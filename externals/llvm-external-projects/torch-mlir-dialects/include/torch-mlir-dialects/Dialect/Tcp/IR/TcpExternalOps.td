#ifndef TORCH_MLIR_DIALECT_TCP_EXTERNAL
#define TORCH_MLIR_DIALECT_TCP_EXTERNAL

def Tcp_TensorOrScalar : AnyTypeOf<[Tcp_Tensor, Tcp_Scalar]>;

def Tcp_YieldOp : Tcp_Op<"yield", [Terminator, Pure]> {
  let summary = "yield operator";

  let description = [{
    Yield
  }];

  let arguments = (ins
    Variadic<Tcp_TensorOrScalar>:$ins
  );

  let assemblyFormat = "$ins attr-dict `:` type($ins)";
}

def Tcp_GroupOp : Tcp_Op<"group", []> {
  let summary = "Group";

  let description = [{
    An op that represents a group of operations.
  }];

  let arguments = (ins
    StrAttr:$group_type
  );

  let results = (outs
    Variadic<Tcp_TensorOrScalar>:$outs
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = "attr-dict-with-keyword $body `:` type($outs)";
}

def Tcp_IsolatedGroupOp : Tcp_Op<"isolated_group", [IsolatedFromAbove]> {
  let summary = "IsolatedGroup";

  let description = [{
    Isolated group
  }];

  let arguments = (ins
    Variadic<Tcp_TensorOrScalar>:$ins,
    StrAttr:$group_type
  );

  let results = (outs
    Variadic<Tcp_TensorOrScalar>:$outs
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = "$ins attr-dict-with-keyword $body `:` type($ins) `->` type($outs)";
}

#endif // TORCH_MLIR_DIALECT_TCP_EXTERNAL
