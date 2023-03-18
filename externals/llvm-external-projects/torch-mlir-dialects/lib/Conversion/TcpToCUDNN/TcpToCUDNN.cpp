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

namespace mlir {

#define GEN_PASS_DEF_CONVERTTCPTOCUDNN
#include "torch-mlir-dialects/Conversion/Passes.h.inc"

namespace tcp {

namespace {

class ConvertIsolatedGroupOp : public OpRewritePattern<IsolatedGroupOp> {
public:
  using OpRewritePattern<IsolatedGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IsolatedGroupOp op,
                                PatternRewriter &rewriter) const final {
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
