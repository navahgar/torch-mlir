#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::tcp {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTcpFuseElementwiseOpsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTcpFuseOpsForCudnnPass();

} // namespace mlir::tcp