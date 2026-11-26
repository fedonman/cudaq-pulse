// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fusion pass: merge adjacent drive ops on the same line with same-amplitude
// square waveforms into a single drive with summed duration.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "cudaq-pulse/Transforms/Passes.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace {

static std::optional<int64_t> traceConstantI64(mlir::Value v) {
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static bool sameSSAValue(mlir::Value a, mlir::Value b) {
  if (a == b)
    return true;
  auto *aOp = a.getDefiningOp();
  auto *bOp = b.getDefiningOp();
  if (!aOp || !bOp)
    return false;
  auto aCst = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(aOp);
  auto bCst = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(bOp);
  if (aCst && bCst)
    return aCst.value().bitwiseIsEqual(bCst.value());
  return false;
}

/// Whether `op` is or holds a barrier: a wait whose length comes from an
/// answer, or a wait for a blocking classical call. Operations after a
/// barrier start after it, so two drives either side of one are not
/// adjacent in time.
static bool holdsBarrier(mlir::Operation *op) {
  auto barrier = [](mlir::Operation *inner) {
    if (inner->hasAttr("classical.variable"))
      return true;
    if (inner->getName().getStringRef() != "classical.wait")
      return false;
    auto callee = inner->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    mlir::Operation *callable =
        callee ? mlir::SymbolTable::lookupNearestSymbolFrom(inner, callee)
               : nullptr;
    auto mode =
        callable ? callable->getAttrOfType<mlir::StringAttr>("mode") : nullptr;
    return mode && mode.getValue() == "blocking";
  };
  return op
      ->walk([&](mlir::Operation *inner) {
        return barrier(inner) ? mlir::WalkResult::interrupt()
                              : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

static double phaseAttr(pulse::DriveOp drive, llvm::StringRef name) {
  if (auto attr = drive->getAttrOfType<mlir::FloatAttr>(name))
    return attr.getValueAsDouble();
  return 0.0;
}

struct FuseAdjacentSquareDrives
    : public mlir::OpRewritePattern<pulse::DriveOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::DriveOp firstDrive,
                  mlir::PatternRewriter &rewriter) const override {
    auto firstWf = firstDrive.getPulse().getDefiningOp<pulse::SquarePulseOp>();
    if (!firstWf)
      return mlir::failure();

    auto updatedLine = firstDrive.getUpdatedLine();
    if (!updatedLine.hasOneUse())
      return mlir::failure();

    auto *user = *updatedLine.getUsers().begin();
    auto secondDrive = mlir::dyn_cast<pulse::DriveOp>(user);
    if (!secondDrive || secondDrive.getLine() != updatedLine)
      return mlir::failure();

    auto secondWf =
        secondDrive.getPulse().getDefiningOp<pulse::SquarePulseOp>();
    if (!secondWf)
      return mlir::failure();

    if (!sameSSAValue(firstWf.getAmpReal(), secondWf.getAmpReal()) ||
        !sameSSAValue(firstWf.getAmpImag(), secondWf.getAmpImag()))
      return mlir::failure();

    if (secondDrive.getTone() != firstDrive.getUpdatedTone())
      return mlir::failure();

    if (secondDrive->getBlock() != firstDrive->getBlock())
      return mlir::failure();
    // A drive written into an idle window stays apart from the kernel's own.
    if (firstDrive->hasAttr("pulse.written") !=
        secondDrive->hasAttr("pulse.written"))
      return mlir::failure();
    for (mlir::Operation *op = firstDrive->getNextNode(); op != secondDrive;
         op = op->getNextNode())
      if (holdsBarrier(op))
        return mlir::failure();

    // A frame change folded into the second drive (virtual Z) or a different
    // per-pulse phase would be lost by playing both as one pulse.
    if (phaseAttr(secondDrive, "frame_phase_offset") != 0.0 ||
        phaseAttr(firstDrive, "phase_offset") !=
            phaseAttr(secondDrive, "phase_offset"))
      return mlir::failure();

    auto dur1 = traceConstantI64(firstWf.getDuration());
    auto dur2 = traceConstantI64(secondWf.getDuration());
    if (!dur1 || !dur2)
      return mlir::failure();
    int64_t fusedDur = *dur1 + *dur2;

    auto loc = firstDrive.getLoc();
    auto i64Ty = rewriter.getIntegerType(64);
    auto durConst =
        mlir::arith::ConstantIntOp::create(rewriter, loc, fusedDur, 64);
    auto fusedWf = pulse::SquarePulseOp::create(
        rewriter, loc, firstWf.getType(), durConst.getResult(),
        firstWf.getAmpReal(), firstWf.getAmpImag());

    auto fusedDrive = pulse::DriveOp::create(
        rewriter, loc, firstDrive.getUpdatedLine().getType(),
        firstDrive.getUpdatedTone().getType(), firstDrive.getLine(),
        fusedWf.getResult(), firstDrive.getTone(), firstDrive.getStartCycles(),
        fusedDur, firstDrive.getPeriodCycles());

    for (llvm::StringRef name : {"frame_phase_offset", "phase_offset"})
      if (auto a = firstDrive->getAttr(name))
        fusedDrive->setAttr(name, a);
    fusedDrive->setAttr("fused", rewriter.getUnitAttr());
    // Two drives written into an idle window make a written drive.
    if (firstDrive->hasAttr("pulse.written"))
      fusedDrive->setAttr("pulse.written", rewriter.getUnitAttr());

    rewriter.replaceOp(secondDrive, fusedDrive->getResults());
    rewriter.replaceOp(firstDrive, fusedDrive->getResults());
    return mlir::success();
  }
};

} // namespace

namespace pulse {
void populateFusionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<FuseAdjacentSquareDrives>(patterns.getContext());
}
} // namespace pulse
