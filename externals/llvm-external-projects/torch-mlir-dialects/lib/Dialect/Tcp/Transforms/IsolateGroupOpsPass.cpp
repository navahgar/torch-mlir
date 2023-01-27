#include "torch-mlir-dialects/Dialect/Tcp/Transforms/IsolateGroupOpsPass.h"
#include "./PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"
#include "torch-mlir-dialects/Dialect/Tcp/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"

#include <iostream>

using namespace mlir;

namespace mlir::tcp {

namespace {

class IsolateGroups : public OpRewritePattern<GroupOp> {
public:
  using OpRewritePattern<GroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GroupOp groupOp,
                                PatternRewriter &rewriter) const override {
    // Collect the values used in the given GroupOp. Those will be the inputs
    // to the IsolatedGroup op.
    std::vector<Value> inputs;
    llvm::SmallDenseSet<Value> defs;
    for (auto &op : groupOp.getBody().front()) {
      for (auto operand : op.getOperands()) {
        if (defs.find(operand) == defs.end()) {
          inputs.push_back(operand);
        }
      }
      defs.insert(op.getResults().begin(), op.getResults().end());
    }

    auto isolatedGroupOp = rewriter.create<IsolatedGroupOp>(
        groupOp.getLoc(), groupOp.getResultTypes(), inputs,
        groupOp.getGroupType());

    isolatedGroupOp.getBody().takeBody(groupOp.getBody());
    auto &isolatedGroupBlock = isolatedGroupOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&isolatedGroupBlock);
      for (size_t n = 0; n < inputs.size(); ++n) {
        isolatedGroupBlock.addArgument(inputs[n].getType(), groupOp.getLoc());
        // TODO: Replace the following with `replaceUseIf` once we update the
        // LLVM base version since that method was added recently.
        //
        // rewriter.replaceUseIf(inputs[n], isolatedGroupBlock.getArgument(n),
        //     [&](OpOperand& opOperand) {
        //       return (opOperand.getOwner()->getParentOp() == groupOp);
        //     });
        for (OpOperand &operand :
             llvm::make_early_inc_range(inputs[n].getUses())) {
          if (operand.getOwner()->getParentOp() == isolatedGroupOp)
            rewriter.updateRootInPlace(operand.getOwner(), [&]() {
              operand.set(isolatedGroupBlock.getArgument(n));
            });
        }
      }
    }
    for (unsigned n = 0; n < groupOp.getNumResults(); ++n) {
      rewriter.replaceAllUsesWith(groupOp->getOpResult(n),
                                  isolatedGroupOp->getOpResult(n));
    }
    rewriter.eraseOp(groupOp);
    return success();
  }
};

class TcpIsolateGroupOpsPass
    : public TcpIsolateGroupOpsBase<TcpIsolateGroupOpsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    patterns.add<IsolateGroups>(context);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTcpIsolateGroupOpsPass() {
  return std::make_unique<TcpIsolateGroupOpsPass>();
}

} // namespace mlir::tcp