// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The patterns of pulse-merge-waits (adjacent waits on a line) and of
// pulse-simplify-control-flow (redundant syncs, untimed schedule blocks, and
// the is_set and scf.for patterns).

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "cudaq-pulse/Transforms/Passes.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace {

/// Remove sync ops with only one operand (no-op synchronization).
struct RemoveSingleOperandSync : public mlir::OpRewritePattern<pulse::SyncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::SyncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getNumOperands() <= 1) {
      if (op.getNumResults() == 1 && op.getNumOperands() == 1) {
        rewriter.replaceOp(op, op.getOperand(0));
      } else if (op.getNumResults() == 0) {
        rewriter.eraseOp(op);
      } else {
        return mlir::failure();
      }
      return mlir::success();
    }
    return mlir::failure();
  }
};

/// Remove consecutive duplicate sync ops on the same set of lines.
struct RemoveRedundantSync : public mlir::OpRewritePattern<pulse::SyncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::SyncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prevOp = op->getPrevNode();
    if (!prevOp)
      return mlir::failure();

    auto prevSync = mlir::dyn_cast<pulse::SyncOp>(prevOp);
    if (!prevSync)
      return mlir::failure();

    if (prevSync.getNumResults() != op.getNumOperands())
      return mlir::failure();

    bool allMatch = true;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (op.getOperand(i) != prevSync.getResult(i)) {
        allMatch = false;
        break;
      }
    }

    if (!allMatch)
      return mlir::failure();

    // This sync immediately follows the previous one on the same lines
    rewriter.replaceOp(op, prevSync.getResults());
    return mlir::success();
  }
};

/// The constant length in cycles of a wait that has no attributes (so not a
/// variable, answer-length or scheduled wait), or nullopt.
static std::optional<int64_t> plainWaitCycles(pulse::WaitOp wait) {
  if (!wait->getAttrDictionary().empty())
    return std::nullopt;
  auto duration = wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>();
  if (!duration)
    return std::nullopt;
  llvm::APInt cycles;
  if (!mlir::matchPattern(duration.getCycles(), mlir::m_ConstantInt(&cycles)))
    return std::nullopt;
  return cycles.getSExtValue();
}

/// A wait right after another on the same line idles the line as one wait
/// of both lengths does.
struct MergeAdjacentWaits : public mlir::OpRewritePattern<pulse::WaitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::WaitOp second,
                  mlir::PatternRewriter &rewriter) const override {
    auto first = second.getLine().getDefiningOp<pulse::WaitOp>();
    if (!first || !first.getUpdatedLine().hasOneUse() ||
        first->getBlock() != second->getBlock())
      return mlir::failure();
    std::optional<int64_t> a = plainWaitCycles(first);
    std::optional<int64_t> b = plainWaitCycles(second);
    if (!a || !b)
      return mlir::failure();
    rewriter.setInsertionPoint(second);
    // The first wait's location: the merged wait starts where it did.
    mlir::Location loc = first.getLoc();
    auto cycles =
        mlir::arith::ConstantIntOp::create(rewriter, loc, *a + *b, 64);
    auto duration = pulse::DurationFromIntOp::create(
        rewriter, loc, second.getDuration().getType(), cycles);
    auto merged = pulse::WaitOp::create(
        rewriter, loc, second.getType(), first.getLine(), duration,
        /*start_cycles=*/std::nullopt, /*duration_cycles=*/std::nullopt,
        /*period_cycles=*/llvm::ArrayRef<int64_t>());
    rewriter.replaceOp(second, merged);
    rewriter.eraseOp(first);
    return mlir::success();
  }
};

/// A schedule block without drives, readouts or waits has no duration and
/// schedules the same inlined.
struct InlineUntimedScheduleBlock
    : public mlir::OpRewritePattern<pulse::ScheduleBlockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::ScheduleBlockOp block,
                  mlir::PatternRewriter &rewriter) const override {
    bool timed =
        block.getBody()
            .walk([](mlir::Operation *op) {
              return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(
                         op)
                         ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
            })
            .wasInterrupted();
    if (timed)
      return mlir::failure();
    mlir::Block &body = block.getBody().front();
    mlir::Operation *yield = body.getTerminator();
    llvm::SmallVector<mlir::Value> results(yield->getOperands());
    // A block that passes a value through yields one of its body's own
    // arguments. Inlining destroys the arguments, so such a result is the
    // input the argument stood for.
    for (mlir::Value &result : results)
      if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(result);
          argument && argument.getOwner() == &body)
        result = block.getInputs()[argument.getArgNumber()];
    rewriter.eraseOp(yield);
    rewriter.inlineBlockBefore(&body, block, block.getInputs());
    rewriter.replaceOp(block, results);
    return mlir::success();
  }
};

} // namespace

namespace pulse {
void populateMergeWaitsPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<MergeAdjacentWaits>(patterns.getContext());
}

void populateSimplifyControlFlowPatterns(mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *ctx = patterns.getContext();
  patterns.add<RemoveSingleOperandSync, RemoveRedundantSync,
               InlineUntimedScheduleBlock>(ctx);
  // Branch on pulse.is_set, and inline or drop loops that run once or
  // never. scf.if patterns are left out: they may turn a branch over
  // lines and tones into arith.select.
  pulse::IsSetOp::getCanonicalizationPatterns(patterns, ctx);
  mlir::scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
}
} // namespace pulse
