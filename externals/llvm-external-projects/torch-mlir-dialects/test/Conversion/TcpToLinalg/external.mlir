// RUN: torch-mlir-dialects-opt <%s -convert-tcp-to-linalg -split-input-file --mlir-print-ir-after-all | FileCheck %s

// CHECK-LABEL: func.func @test_conv2d(
// CHECK-SAME:                %[[ARG0:.*]]: tensor<1x128x28x28xf32>,
// CHECK-SAME:                %[[ARG1:.*]]: tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32> {
// CHECK:         %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[PAD:.*]] = tensor.pad %[[ARG0]] low[0, 0, 1, 1] high[0, 0, 1, 1]
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<1x128x28x28xf32>
// CHECK:         %[[C1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[C1]] : f32) outs(%[[EMPTY]] : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
// CHECK:         %[[CONV:.*]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:              ins(%[[PAD]], %[[ARG1]] : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>)
// CHECK-SAME:              outs(%[[FILL]] : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
func.func @test_conv2d(%arg0 : tensor<1x128x28x28xf32>, %arg1 : tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32> {
  %10 = tcp.conv2d %arg0, %arg1
            {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}
            : tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32> -> tensor<1x128x28x28xf32>
  return %10 : tensor<1x128x28x28xf32>
}
