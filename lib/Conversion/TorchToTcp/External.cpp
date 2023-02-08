#include "torch-mlir/Conversion/TorchToTcp/TorchToTcp.h"

#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertAtenConvolutionOp : public OpConversionPattern<AtenConvolutionOp> {
public:
  using OpConversionPattern<AtenConvolutionOp>::OpConversionPattern;
  using OpAdaptor = typename AtenConvolutionOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto weight = adaptor.getWeight();

    auto inputTy = input.getType().cast<RankedTensorType>();
    auto weightTy = weight.getType().cast<RankedTensorType>();
    auto outputTy = getTypeConverter()
                        ->convertType(op.getType())
                        .template cast<RankedTensorType>();

    if (!inputTy || !weightTy || !outputTy)
      return rewriter.notifyMatchFailure(op,
                                         "Input, weight and output to "
                                         "Convolution must be ranked tensors");

    if (inputTy.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: only 2D convolutions supported");

    auto inputElemTy = inputTy.getElementType();
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    auto weightShape = makeShapeTorchCompatible(weightTy.getShape());

    if (!adaptor.getBias().getType().template isa<Torch::NoneType>()) {
      return rewriter.notifyMatchFailure(op, "Bias is not yet supported");
    }

    SmallVector<int64_t, 2> stride;
    if (!matchPattern(adaptor.getStride(), m_TorchListOfConstantInts(stride)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const stride list unsupported");

    SmallVector<int64_t, 2> padding_2d;
    if (!matchPattern(adaptor.getPadding(),
                      m_TorchListOfConstantInts(padding_2d)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const padding list unsupported");
    // TOSA uses 4D padding {t, b, l, r} while Torch defines 2D padding {t, l}.
    // The Torch OFM computation uses 2*pad in each spatial direction, implying
    // the same t=b and l=r values for TOSA.
    SmallVector<int64_t> padding(
        {padding_2d[0], padding_2d[0], padding_2d[1], padding_2d[1]});

    SmallVector<int64_t, 2> dilation;
    if (!matchPattern(adaptor.getDilation(),
                      m_TorchListOfConstantInts(dilation)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dilation list unsupported");

    int64_t outputHDim;
    int64_t outputWDim;
    if (inputTy.hasStaticShape()) {
      outputHDim = (inputShape[1] + padding[0] + padding[1] -
                    dilation[0] * (weightShape[1] - 1) - 1) /
                       stride[0] +
                   1;
      outputWDim = (inputShape[2] + padding[2] + padding[3] -
                    dilation[1] * (weightShape[2] - 1) - 1) /
                       stride[1] +
                   1;
    } else {
      outputHDim = kUnknownSize;
      outputWDim = kUnknownSize;
    }

    rewriter.replaceOpWithNewOp<tcp::Conv2DOp>(op, outputTy, input, weight,
                                               padding, stride, dilation);
    return success();
  }
};

} // namespace

void torch_to_tcp::populateExternalPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenConvolutionOp>();
  patterns.add<ConvertAtenConvolutionOp>(typeConverter, context);
}
