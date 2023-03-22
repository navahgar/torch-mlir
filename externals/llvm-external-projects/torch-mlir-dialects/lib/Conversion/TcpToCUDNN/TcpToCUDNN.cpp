//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/TcpToCUDNN/TcpToCUDNN.h"

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "mlir/Dialect/CUDNN/IR/CUDNNOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/IR/BuiltinTypes.h>

namespace mlir {

#define GEN_PASS_DEF_CONVERTTCPTOCUDNN
#include "torch-mlir-dialects/Conversion/Passes.h.inc"

namespace tcp {

namespace {

::mlir::cudnn::TensorDescType TensorTypeToTensorDesc(RankedTensorType type) {
  MLIRContext *ctx = type.getContext();
  return mlir::cudnn::TensorDescType::get(ctx, /*shape=*/type.getShape(), /*element_type=*/type.getElementType(), /*alignment=*/1, /*stride=*/{});
}

bool IsOpSupportedByCUDNN(Operation &op) {
  return isa<tcp::ConstOp>(op) || isa<tcp::Conv2DOp>(op) || isa<tcp::ClampOp>(op) || isa<tcp::YieldOp>(op);
}


class ConvertIsolatedGroupOp : public OpRewritePattern<IsolatedGroupOp> {
public:
  using OpRewritePattern<IsolatedGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IsolatedGroupOp group,
                                PatternRewriter &rewriter) const final {
    if (IsGroupSupportedByCUDNN(group)) {
      // llvm::dbgs() << "SUPPORTED " << group << "nnn";
    }

    auto handle = rewriter.create<::mlir::cudnn::GetCurrentHandle>(group.getLoc());

    auto buildAndExec = rewriter.create<::mlir::cudnn::BuildAndExecGraphOp>(
        group.getLoc(), ValueTypeRange<ResultRange>(group.getOuts()), handle, group.getIns());

    Block& b = buildAndExec.getConstructor().emplaceBlock();

    rewriter.setInsertionPoint(&b, b.end());
    auto yieldOp = cast<YieldOp>(group.getBody().front().getTerminator());
    rewriter.create<::mlir::cudnn::BuildGraphOp>(group.getLoc(), yieldOp.getIns().front());

    Block &groupBlock = group.getBody().front();
    while (groupBlock.rbegin() != groupBlock.rend()) {
      groupBlock.back().moveBefore(&b, b.begin());
    }

    ::llvm::DenseMap<mlir::Value, mlir::Value> tensorToTensorDesc;
    for (auto groupArg : group.getBody().getArguments()) {
      tensorToTensorDesc[groupArg] = b.addArgument(TensorTypeToTensorDesc(groupArg.getType().cast<RankedTensorType>()), group->getLoc());
      groupArg.replaceAllUsesWith(tensorToTensorDesc[groupArg]);
    }
    group->replaceAllUsesWith(buildAndExec);
    group.erase();
    return success();
  }

private:
  static bool IsGroupSupportedByCUDNN(IsolatedGroupOp group) {
    return llvm::all_of(group.getBody().getOps(), IsOpSupportedByCUDNN);
  }
};

class ConvertReluOp : public OpRewritePattern<ClampOp> {
public:
  using OpRewritePattern<ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ClampOp clamp,
                                PatternRewriter &rewriter) const final {
    auto cudnnParent = clamp->getParentOfType<::mlir::cudnn::BuildAndExecGraphOp>();
    if (!cudnnParent) {
      return success();
    }

    auto resultTy = clamp->getResultTypes().front().cast<RankedTensorType>();
    auto inputTy = clamp.getIn().getType().cast<RankedTensorType>();
    // Check if the clamp op is a relu with f32.
    if (!inputTy.getElementType().isa<FloatType>())
      return failure();
    if (clamp.getMaxFloat())
      return failure();
    if (!clamp.getMinFloat() || !clamp.getMinFloat()->isZero())
      return failure();

    auto cudnnRelu = rewriter.replaceOpWithNewOp<cudnn::PointWiseReluOp>(
            clamp,
            TensorTypeToTensorDesc(resultTy),
            clamp.getIn(),
            TypeAttr::get(inputTy.getElementType()),
            rewriter.getF64FloatAttr(0.0));
    clamp->replaceAllUsesWith(cudnnRelu);
    return success();
  }
};

class ConvertConv2DOp : public OpRewritePattern<Conv2DOp> {
public:
  using OpRewritePattern<Conv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOp conv2d,
                                PatternRewriter &rewriter) const final {
    auto cudnnParent = conv2d->getParentOfType<::mlir::cudnn::BuildAndExecGraphOp>();
    if (!cudnnParent)
      return success();

    auto inputTy = conv2d.getInput().getType().cast<RankedTensorType>();
    auto resultTy = conv2d->getResultTypes().front().cast<RankedTensorType>();
    auto cudnnConv = rewriter.replaceOpWithNewOp<cudnn::ConvolutionOp>(
                conv2d,
                TensorTypeToTensorDesc(resultTy),
                conv2d.getInput(),
                conv2d.getWeight(),
                TypeAttr::get(inputTy.getElementType()),
                rewriter.getF32FloatAttr(1.0),  // alpha
                rewriter.getF32FloatAttr(0.0),  // beta
                rewriter.getI32IntegerAttr(4),  // spatial_dim_count
                conv2d.getStrideAttr(), // spatial_stride
                conv2d.getPadAttr(),  // pre_padding
                rewriter.getDenseI64ArrayAttr({}), // post_padding
                conv2d.getDilationAttr());  // dilation
    conv2d->replaceAllUsesWith(cudnnConv);
    return success();
  }
};

class ConvertYieldOp : public OpRewritePattern<YieldOp> {
public:
  using OpRewritePattern<YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(YieldOp yield, PatternRewriter &rewriter) const final {
    auto cudnnParent = yield->getParentOfType<::mlir::cudnn::BuildAndExecGraphOp>();
    if (cudnnParent)
      yield.erase();

    return success();
  }
};

class ConvertTcpToCUDNN : public ConvertTcpToCUDNNBase<ConvertTcpToCUDNN> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<cudnn::CUDNNDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    patterns.add<ConvertIsolatedGroupOp>(context);
    patterns.add<ConvertReluOp>(context);
    patterns.add<ConvertConv2DOp>(context);
    patterns.add<ConvertYieldOp>(context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTcpToCUDNNPass() {
  return std::make_unique<ConvertTcpToCUDNN>();
}

} // namespace tcp
} // namespace mlir
