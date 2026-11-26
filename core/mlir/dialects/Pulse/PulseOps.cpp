// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseResources.h"
#include "cudaq-pulse/Dialect/Pulse/PulseTimingFormat.h"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

using namespace mlir;

namespace pulse {

// ===----------------------------------------------------------------------===//
// Verifiers check an op on its own: its types and its attributes. A check of
// a value, a duration, sigma or measurement that another op computes, looks
// across ops and is in the pulse-verify pass (transforms/PulseVerify.cpp),
// with the module-wide checks (linearity, monotone time, measurement
// causality): folding may turn such a value into a constant, and must not
// turn an op that verified into one that does not.
// ===----------------------------------------------------------------------===//

LogicalResult CustomSamplesOp::verify() {
  if (getSamples().empty())
    return emitOpError("samples array must not be empty");
  if (auto rate = getRate()) {
    if (rate->size() != 2 || (*rate)[0] < 1 || (*rate)[1] < 1)
      return emitOpError("rate must be two positive integers, n samples every "
                         "d cycles");
    auto count = static_cast<int64_t>(getSamples().size());
    int64_t span;
    if (__builtin_mul_overflow(count, (*rate)[1], &span))
      return emitOpError() << count << " samples at " << (*rate)[0] << " every "
                           << (*rate)[1]
                           << " cycles last more cycles than an i64 holds";
    if (span % (*rate)[0])
      return emitOpError() << count << " samples at " << (*rate)[0] << " every "
                           << (*rate)[1]
                           << " cycles do not last a whole number of cycles";
  }
  return success();
}

int64_t CustomSamplesOp::getDurationCycles() {
  auto count = static_cast<int64_t>(getSamples().size());
  if (auto rate = getRate())
    return count * (*rate)[1] / (*rate)[0];
  return count;
}

LogicalResult WaitOp::verify() {
  auto lowest = (*this)->getAttrOfType<IntegerAttr>("pulse.min_cycles");
  auto highest = (*this)->getAttrOfType<IntegerAttr>("pulse.max_cycles");
  if (!(*this)->hasAttr("pulse.variable_wait")) {
    if (lowest || highest)
      return emitOpError("pulse.min_cycles and pulse.max_cycles apply to a "
                         "pulse.variable_wait wait");
    return success();
  }
  if (!lowest)
    return emitOpError("a pulse.variable_wait wait needs pulse.min_cycles");
  if (lowest.getInt() < 0)
    return emitOpError("pulse.min_cycles must not be negative, got ")
           << lowest.getInt();
  if (highest && highest.getInt() < lowest.getInt())
    return emitOpError("pulse.max_cycles (")
           << highest.getInt() << ") is below pulse.min_cycles ("
           << lowest.getInt() << ")";
  return success();
}

LogicalResult GetDriveLineOp::verify() {
  // A broadcast line names its qudits once each, lowest index first: the
  // line is found by its first.
  if (getOthers().empty())
    return success();
  int64_t last = -1;
  for (Value qudit : getOperation()->getOperands()) {
    auto alloc = qudit.getDefiningOp<QuditAllocOp>();
    if (!alloc)
      return emitOpError("a drive line of several qudits names qudits of the "
                         "quantum system (pulse.qudit_alloc)");
    int64_t index = alloc.getQuditAttr().getInt();
    if (index <= last)
      return emitOpError("a drive line of several qudits names each once, "
                         "lowest index first; got qudit ")
             << index << " after qudit " << last;
    last = index;
  }
  return success();
}

LogicalResult SyncOp::verify() {
  // Sync semantics: align time of multiple lines. Two or more lines required.
  if (getLines().size() < 2)
    return emitOpError("sync requires at least 2 lines, got ")
           << getLines().size();
  if (getLines().size() != getSyncedLines().size())
    return emitOpError("number of input lines (")
           << getLines().size() << ") must match number of output lines ("
           << getSyncedLines().size() << ")";
  Operation *op = getOperation();
  for (unsigned index = 0, count = op->getNumOperands(); index < count;
       ++index) {
    Type line = op->getOperand(index).getType();
    Type synced = op->getResult(index).getType();
    if (line != synced)
      return emitOpError("line ")
             << index << " is a " << line << ", but its result is a " << synced;
  }
  return success();
}

namespace {
/// `arith.cmpi ne, (pulse.outcome %m), 0` is `pulse.is_set %m`.
struct OutcomeNonzeroToIsSet : OpRewritePattern<arith::CmpIOp> {
  explicit OutcomeNonzeroToIsSet(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/1, {IsSetOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(arith::CmpIOp cmp,
                                PatternRewriter &rewriter) const override {
    if (cmp.getPredicate() != arith::CmpIPredicate::ne)
      return failure();
    Value lhs = cmp.getLhs(), rhs = cmp.getRhs();
    auto outcome = lhs.getDefiningOp<OutcomeOp>();
    Value other = rhs;
    if (!outcome) {
      outcome = rhs.getDefiningOp<OutcomeOp>();
      other = lhs;
    }
    APInt zero;
    if (!outcome || !matchPattern(other, m_ConstantInt(&zero)) ||
        !zero.isZero())
      return failure();
    rewriter.replaceOpWithNewOp<IsSetOp>(cmp, cmp.getType(),
                                         outcome.getMeasurement());
    return success();
  }
};
} // namespace

void IsSetOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<OutcomeNonzeroToIsSet>(ctx);
}

/// The types AnyTimedType admits: lines, tones, measurements, and the
/// classical dialect's links and call handles.
static bool isTimedValueType(Type type) {
  return isa<DriveLineType, ReadoutLineType, ToneType, MeasurementType>(type) ||
         type.getDialect().getNamespace() == "classical";
}

LogicalResult ScheduleBlockOp::verifyRegions() {
  // RegionBranchOpInterface checks that the operands match the block
  // arguments and the yielded values the results.
  Block &body = getBody().front();
  if (body.empty() || !isa<YieldOp>(body.back()))
    return emitOpError("body must end in pulse.yield");
  // Lines, tones and measurements reach the body only as block arguments,
  // so the boundary lists every timed value that crosses it.
  Operation *self = getOperation();
  WalkResult walk = getBody().walk([&](Operation *inner) {
    for (Value operand : inner->getOperands()) {
      if (!isTimedValueType(operand.getType()))
        continue;
      Region *region = operand.getParentRegion();
      if (region && self->isProperAncestor(region->getParentOp()))
        continue;
      if (region == &getBody())
        continue;
      inner->emitOpError("uses a line, tone, measurement or link from outside "
                         "its pulse.schedule_block; pass it as an operand");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

void ScheduleBlockOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
    return;
  }
  regions.push_back(RegionSuccessor(getOperation(), getOutputs()));
}

OperandRange ScheduleBlockOp::getEntrySuccessorOperands(RegionSuccessor) {
  return getInputs();
}

// ===----------------------------------------------------------------------===//
// Folders — fold away no-op operations (zero shift, scale by 1, double neg).
// pulse-fold-constants applies them.
//
// Rewrites that need more than one op are passes: adjacent waits in
// pulse-merge-waits and redundant syncs in pulse-simplify-control-flow
// (transforms/Canonicalize.cpp), phase folding into drives in
// pulse-fold-phases (transforms/VirtualZ.cpp), square pulse fusion in
// pulse-fuse-pulses (transforms/Fusion.cpp).
// ===----------------------------------------------------------------------===//

OpFoldResult WaitOp::fold(FoldAdaptor adaptor) {
  // Cannot fold via adaptor since DurationType is a custom type.
  return nullptr;
}

// SyncOp has variadic results → multi-result fold signature
LogicalResult SyncOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  // Whether a sync is redundant depends on the lines' times, a program-wide
  // analysis that pulse-simplify-control-flow (RemoveRedundantSync) does; one
  // op on its own never folds.
  return failure();
}

OpFoldResult ShiftFrequencyOp::fold(FoldAdaptor adaptor) {
  // Fold shift_frequency by 0 Hz → identity (return input tone)
  if (auto freq = dyn_cast_or_null<FloatAttr>(adaptor.getFrequencyHz())) {
    if (freq.getValueAsDouble() == 0.0)
      return getTone();
  }
  return nullptr;
}

OpFoldResult ShiftPhaseOp::fold(FoldAdaptor adaptor) {
  // Fold shift_phase by 0 rad → identity (virtual-Z of angle 0 is no-op)
  if (auto phase = dyn_cast_or_null<FloatAttr>(adaptor.getPhaseRad())) {
    if (phase.getValueAsDouble() == 0.0)
      return getTone();
  }
  return nullptr;
}

OpFoldResult SetPhaseOp::fold(FoldAdaptor adaptor) {
  // set_phase cannot be folded away in general (even setting to 0 is
  // semantically meaningful — it resets the phase)
  return nullptr;
}

OpFoldResult SetFrequencyOp::fold(FoldAdaptor adaptor) { return nullptr; }

OpFoldResult PulseAddOp::fold(FoldAdaptor adaptor) { return nullptr; }

OpFoldResult PulseSubOp::fold(FoldAdaptor adaptor) {
  // fold x - x → zero waveform (would need zero waveform constant)
  return nullptr;
}

OpFoldResult PulseMulOp::fold(FoldAdaptor adaptor) { return nullptr; }

OpFoldResult PulseScaleOp::fold(FoldAdaptor adaptor) {
  // Fold scale(pulse, 1.0) → pulse (identity)
  if (auto scale = dyn_cast_or_null<FloatAttr>(adaptor.getScale())) {
    if (scale.getValueAsDouble() == 1.0)
      return getPulse();
  }
  return nullptr;
}

OpFoldResult PulseNegOp::fold(FoldAdaptor adaptor) {
  // neg(neg(x)) → x
  if (auto innerNeg = getPulse().getDefiningOp<PulseNegOp>())
    return innerNeg.getPulse();
  return nullptr;
}

// ===----------------------------------------------------------------------===//
// Canonicalization patterns
//
// The pulse rewrites are the pattern passes in transforms/. The op-level
// hooks below register no patterns yet; patterns added to them run in the
// generic MLIR canonicalizer, and so when a call's values are bound.
// ===----------------------------------------------------------------------===//

void WaitOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {}

void ShiftFrequencyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {}

void ShiftPhaseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {}

void SetPhaseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {}

void SetFrequencyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {}

void PulseNegOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {}

} // namespace pulse

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.cpp.inc"
