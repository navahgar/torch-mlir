// RUN: torch-mlir-opt <%s -convert-torch-to-tcp -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.convolution(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.vtensor<[?,?,?,?],f32>,
// CHECK-SAME:                                 %[[ARG_1:.*]]: !torch.vtensor<[?,?,3,3],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[T_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_1:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?,3,3],f32> -> tensor<?x?x3x3xf32>
// CHECK:           %[[T_2:.*]] = torch.constant.none
// CHECK:           %[[T_4:.*]] = torch.constant.int 2
// CHECK:           %[[T_5:.*]] = torch.constant.int 1
// CHECK:           %[[T_6:.*]] = torch.constant.int 4
// CHECK:           %[[T_7:.*]] = torch.constant.int 3
// CHECK:           %[[T_8:.*]] = torch_c.to_i64 %[[T_7]]
// CHECK:           %[[T_9:.*]] = torch.prim.ListConstruct %[[T_4]], %[[T_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_10:.*]] = torch.prim.ListConstruct %[[T_6]], %[[T_4]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_11:.*]] = torch.prim.ListConstruct %[[T_7]], %[[T_5]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T_12:.*]] = torch.prim.ListConstruct  : () -> !torch.list<int>
// CHECK:           %[[T_13:.*]] = torch.constant.bool false
// CHECK:           %[[T_14:.*]] = tcp.conv2d %[[T_0]], %[[T_1]]
// CHECK-SAME:                {dilation = array<i64: 3, 1>, pad = array<i64: 4, 4, 2, 2>, stride = array<i64: 2, 1>} : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[T_15:.*]] = torch_c.from_builtin_tensor %[[T_14]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[T_15]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.convolution(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,3,3],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %none = torch.constant.none
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4
  %int3 = torch.constant.int 3
  %1 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %false = torch.constant.bool false
  %5 = torch.aten.convolution %arg0, %arg1, %none, %1, %2, %3, %false, %4, %int3 : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?,?,?],f32>
  return %5 : !torch.vtensor<[?,?,?,?],f32>
}
