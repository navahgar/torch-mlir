// RUN: torch-mlir-dialects-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[GROUP:.*]] = tcp.group
// CHECK-SAME:                      attributes {group_type = "elementwise"} {
// CHECK:            %[[ADD:.*]] = tcp.add %[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:         } : tensor<?x?xf32>
// CHECK:         return %[[GROUP]] : tensor<?x?xf32>
func.func @test_group(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) { group_type = "elementwise" } : () -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_isolated_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[IGROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]]
// CHECK-SAME:                      attributes {group_type = "elementwise"} {
// CHECK:         ^bb0(%[[BBARG0:.*]]: tensor<?x?xf32>, %[[BBARG1:.*]]: tensor<?x?xf32>):
// CHECK:            %[[ADD:.*]] = tcp.add %[[BBARG0]], %[[BBARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:            tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:         } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[IGROUP]] : tensor<?x?xf32>
func.func @test_isolated_group(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.isolated_group" (%arg0, %arg1) ({
    ^bb0(%0 : tensor<?x?xf32>, %1 : tensor<?x?xf32>) :
      %2 = tcp.add %0, %1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) { group_type = "elementwise" } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
