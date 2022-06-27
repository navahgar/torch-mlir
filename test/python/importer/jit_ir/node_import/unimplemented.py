import torch
import torch_mlir

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

class Inner(object):
    # CHECK-LABEL: func.func private @__torch__.Inner.foo(
    # CHECK-SAME:      %[[ARG:.*]]: !torch.nn.Module<"__torch__.Inner">) {
    # CHECK:         torch.constant.int 42
    # CHECK:         torch.store "cls", %[[ARG]] : !torch.nn.Module<"__torch__.Inner">
    # CHECK:         %[[DICT:.*]] = torch.prim.DictConstruct keys() values() -> !torch.dict<str, tensor>
    # CHECK:         torch.store "this_dict", %[[DICT]] : !torch.dict<str, tensor>
    # CHECK:         torch.load "this_dict" : !torch.dict<str, tensor>
    # CHECK:         torch.constant.str "key"
    # CHECK:         return
    # CHECK:       }

    @classmethod
    def foo(cls):
        this_dict = {}
        this_dict["key"] = 42
        return this_dict

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()

    def forward(self, data):
        return data

output_type = torch_mlir.OutputType.RAW
mod = torch_mlir.compile(Model(), [torch.tensor([0, 1, 2, 3])], output_type)
print(mod)
