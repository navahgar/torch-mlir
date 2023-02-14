// RUN: torch-mlir-dialects-opt %s -split-input-file -tcp-fuse-ops-for-cudnn -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_relu_conv_fusion(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:           %[[ARG1:.*]]: tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32>
// CHECK:          %[[GROUP:.*]] = tcp.group attributes {group_type = "relu-conv"} {
// CHECK:            %[[CLAMP:.*]] = tcp.clamp %[[ARG0]] {min_float = 0.000000e+00 : f32} : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
// CHECK:            %[[CONV:.*]] = tcp.conv2d %[[CLAMP]], %[[ARG1]]
// CHECK-SAME:                     {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
// CHECK-SAME:                    : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
// CHECK:            tcp.yield %[[CONV]] : tensor<?x?x?x?xf32>
// CHECK:          } : tensor<?x?x?x?xf32>
// CHECK:          return %[[GROUP]] : tensor<?x?x?x?xf32>
func.func @test_relu_conv_fusion(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcp.clamp %arg0 {min_float = 0.0 : f32} : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
  %10 = tcp.conv2d %0, %arg1
            {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
            : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
  return %10 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_clamp_w_max_do_not_fuse(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:           %[[ARG1:.*]]: tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[CLAMP:.*]] = tcp.clamp %arg0 {max_float = 4.500000e+00 : f32, min_float = 0.000000e+00 : f32} : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
// CHECK:         %[[CONV:.*]] = tcp.conv2d %[[CLAMP]], %[[ARG1]]
// CHECK-SAME:                     {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
// CHECK-SAME:                    : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
// CHECK:         return %[[CONV]] : tensor<?x?x?x?xf32>
func.func @test_clamp_w_max_do_not_fuse(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcp.clamp %arg0 {min_float = 0.0 : f32, max_float = 4.5 : f32} : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
  %10 = tcp.conv2d %0, %arg1
            {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
            : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
  return %10 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_multiple_fusions(
// CHECK-SAME:           %[[ARG0:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:           %[[ARG1:.*]]: tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32>
// CHECK:          %[[GROUP0:.*]] = tcp.group attributes {group_type = "tanh-conv"} {
// CHECK:            %[[TANH:.*]] = tcp.tanh %[[ARG0]] : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
// CHECK:            %[[CONV0:.*]] = tcp.conv2d %[[TANH]], %[[ARG1]]
// CHECK-SAME:                     {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
// CHECK-SAME:                    : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
// CHECK:            tcp.yield %[[CONV0]] : tensor<?x?x?x?xf32>
// CHECK:          } : tensor<?x?x?x?xf32>
// CHECK:          %[[GROUP1:.*]] = tcp.group attributes {group_type = "relu-conv"} {
// CHECK:            %[[CLAMP:.*]] = tcp.clamp %[[ARG0]] {min_float = 0.000000e+00 : f32} : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
// CHECK:            %[[CONV1:.*]] = tcp.conv2d %[[CLAMP]], %[[ARG1]]
// CHECK-SAME:                     {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
// CHECK-SAME:                    : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
// CHECK:            tcp.yield %[[CONV1]] : tensor<?x?x?x?xf32>
// CHECK:          } : tensor<?x?x?x?xf32>
// CHECK:          %[[MUL:.*]] = tcp.mul %[[GROUP0]], %[[GROUP1]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
// CHECK:          return %[[MUL]] : tensor<?x?x?x?xf32>
func.func @test_multiple_fusions(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32> {
  %10 = tcp.tanh %arg0 : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
  %11 = tcp.conv2d %10, %arg1
            {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
            : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
  %20 = tcp.clamp %arg0 {min_float = 0.0 : f32} : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
  %21 = tcp.conv2d %20, %arg1
            {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>}
            : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
  %30 = tcp.mul %11, %21 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
  return %30 : tensor<?x?x?x?xf32>
}
