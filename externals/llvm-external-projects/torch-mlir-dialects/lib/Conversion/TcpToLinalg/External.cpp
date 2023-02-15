//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/TcpToLinalg/TcpToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

#include <iostream>

using namespace mlir;
using namespace mlir::tcp;

namespace {

Value applyPad(Location loc, Value input, ArrayRef<int64_t> pad,
               Attribute padAttr, OpBuilder &rewriter) {
  // Input should be padded if necessary.
  if (llvm::all_of(pad, [](int64_t p) { return p == 0; }))
    return input;

  ShapedType inputTy = input.getType().cast<ShapedType>();
  Type inputETy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();

  assert((inputShape.size() * 2) == pad.size());

  SmallVector<int64_t, 4> paddedShape;
  SmallVector<OpFoldResult, 8> lowIndices;
  SmallVector<OpFoldResult, 8> highIndices;
  for (int i = 0, s = inputShape.size(); i < s; i++) {
    auto lowPad = pad[i * 2];
    auto highPad = pad[i * 2 + 1];
    if (ShapedType::isDynamic(inputShape[i]))
      paddedShape.push_back(inputShape[i]);
    else
      paddedShape.push_back(inputShape[i] + highPad + lowPad);
    lowIndices.push_back(rewriter.getIndexAttr(lowPad));
    highIndices.push_back(rewriter.getIndexAttr(highPad));
  }

  Value padValue = rewriter.create<arith::ConstantOp>(loc, padAttr);

  return rewriter.create<tensor::PadOp>(
      loc, RankedTensorType::get(paddedShape, inputETy), input, lowIndices,
      highIndices, padValue);
}

class ConvertConv2DOp : public OpConversionPattern<Conv2DOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op->getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();

    auto inputType = input.getType().template cast<RankedTensorType>();
    auto weightType = weight.getType().template cast<RankedTensorType>();
    auto resultType = OpConversionPattern::getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .template cast<RankedTensorType>();
    if (!inputType.hasStaticShape() && !weightType.hasStaticShape() &&
        !resultType.hasStaticShape())
      return b.notifyMatchFailure(
          op, "tcp.conv2d currently only supports static shapes");

    DenseI64ArrayAttr padAttr = op.getPadAttr();
    DenseI64ArrayAttr strideAttr = op.getStrideAttr();
    DenseI64ArrayAttr dilationAttr = op.getDilationAttr();

    Attribute zeroAttr = b.getZeroAttr(inputType.getElementType());
    llvm::SmallVector<int64_t> pad;
    pad.resize(4, 0);
    llvm::append_range(pad, padAttr.asArrayRef());
    input = applyPad(loc, input, pad, zeroAttr, b);

    Value emptyTensor = b.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    Attribute resultZeroAttr = b.getZeroAttr(resultType.getElementType());
    Value zero = b.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor =
        b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor})
            .result();

    auto stride = b.getI64TensorAttr(strideAttr.asArrayRef());
    auto dilation = b.getI64TensorAttr(dilationAttr.asArrayRef());
    auto conv2d = b.create<linalg::Conv2DNchwFchwOp>(
        loc, resultType, ValueRange{input, weight}, ValueRange{zeroTensor},
        stride, dilation);
    b.replaceOp(op, conv2d->getResult(0));
    return success();
  }
};

} // namespace

void mlir::TcpToLinalg::populateExternalPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<Conv2DOp>();
  patterns.add<ConvertConv2DOp>(typeConverter, context);
}
