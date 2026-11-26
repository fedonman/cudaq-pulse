// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The cudaq-pulse passes and conversions, and their registration under the
// names cudaq-pulse-opt and the Python extension run them by.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <string>

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseToQOpPass();

std::unique_ptr<mlir::Pass> createPulseVerifyPass();

void populateMergeWaitsPatterns(mlir::RewritePatternSet &patterns);
void populateSimplifyControlFlowPatterns(mlir::RewritePatternSet &patterns);
void populateVirtualZPatterns(mlir::RewritePatternSet &patterns);
void populateFusionPatterns(mlir::RewritePatternSet &patterns);

/// A pass that is one greedy rewrite of a function with its patterns.
struct RewriteStage {
  llvm::StringLiteral argument;
  llvm::StringLiteral description;
  /// Adds the patterns; null for pulse-fold-constants and
  /// pulse-remove-unused, which have none.
  void (*populate)(mlir::RewritePatternSet &);
  /// The rewrite folds ops and merges equal constants.
  bool foldConstants = false;
  /// The rewrite removes unreachable blocks and unused block arguments.
  bool removeUnused = false;
  /// A call's values that reach the ops these patterns match run them again
  /// (see foldBinding in the extension).
  bool onBinding = false;
  /// The rewrite changes the lengths of ops or their order on a line, so it
  /// takes only an unscheduled function (pulse.schedule_state).
  bool unscheduledOnly = false;
};

/// The pattern passes: pulse-fold-constants, pulse-remove-unused,
/// pulse-merge-waits, pulse-simplify-control-flow, pulse-fold-phases and
/// pulse-fuse-pulses.
llvm::ArrayRef<RewriteStage> rewriteStages();
/// The stage registered as `argument`, or null.
const RewriteStage *findRewriteStage(llvm::StringRef argument);
/// One greedy rewrite of a function with the patterns of all `stages`, which
/// folds and removes as any of them does. The stages share a walk of the
/// function, so a pipeline that runs several back to back pays for one.
std::unique_ptr<mlir::Pass>
createPulseRewritePass(llvm::ArrayRef<const RewriteStage *> stages);
/// MLIR's loop-invariant code motion, as pulse-hoist-loop-invariants.
std::unique_ptr<mlir::Pass> createPulseHoistLoopInvariantsPass();

/// asap and alap: every readout's outcome is usable `readoutLatency` cycles
/// after the readout ends (on several controllers, the module's table).
std::unique_ptr<mlir::Pass> createPulseScheduleAsapPass(int64_t readoutLatency);
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass(int64_t readoutLatency);
/// pulse-schedule-asap and pulse-schedule-alap with the latency as a pass
/// option, for cudaq-pulse-opt.
std::unique_ptr<mlir::Pass> createPulseScheduleAsapOptionsPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAlapOptionsPass();
std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap);
/// pulse-schedule-rcp with the machine model as pass options, for
/// cudaq-pulse-opt; the extension creates the pass above.
std::unique_ptr<mlir::Pass> createPulseScheduleRcpOptionsPass();
std::unique_ptr<mlir::Pass> createPulseInlineScheduleBlocksPass();
std::unique_ptr<mlir::Pass> createPulseExpandCallsPass();
/// Write out every placed pulse.call of `module`, callees first: a copy of
/// the callee's ops at the call's start and periods, and a wait on each line
/// the callee plays on where the call lasts longer than it. Callees no call
/// runs any more are erased. With `callSites`, each copied op's
/// location names the call (for pulse-verify's diagnostics).
mlir::LogicalResult expandCalls(mlir::ModuleOp module, bool callSites);

/// pulse-verify's check of a module with placed pulse.call ops: each callee
/// keeps the rules for one, no call runs back into itself, and the program
/// with its calls written out passes every check.
mlir::LogicalResult checkCalls(mlir::ModuleOp module);

/// For a `module` that lists its controllers' longest programs (its
/// pulse.program_limits, from Controller.max_program_duration): fails if a
/// scheduled function of it can last longer than one of them with its
/// waits of run-time length at their highest, or has such a wait with no
/// highest length where it holds back a line of such a controller
/// (PulseProgramLimits.cpp). The program as scheduled is the schedule
/// passes' to check.
mlir::LogicalResult checkWaitLimits(mlir::ModuleOp module);
/// For the handler of a schedule that failed with a call's values, `error`:
/// where the schedule's check said the program then lasts longer than a
/// controller's longest program, throws that sentence as an error of the
/// values (std::invalid_argument); else throws `error` again (the
/// extension's ProgramLimitError.cpp).
[[noreturn]] void rethrowPastProgramLimit(const std::exception &error);
/// Whether `module` holds a pulse.call.
bool moduleHasCalls(mlir::ModuleOp module);
std::unique_ptr<mlir::Pass>
createPulseClassicalCheckPass(int64_t readoutLatency);
/// pulse-classical-check with its options, for cudaq-pulse-opt; the
/// extension creates the pass above.
std::unique_ptr<mlir::Pass> createPulseClassicalCheckOptionsPass();
/// pulse-frame-changes: place the frame changes of a scheduled program on
/// their tones' timelines and mark those no pulse plays with.
std::unique_ptr<mlir::Pass> createPulseFrameChangesPass();
/// Whether pulse-frame-changes would change `op`: it holds a frame change,
/// or a loop or branch the pass marked.
bool hasFrameChangeWork(mlir::Operation *op);
std::unique_ptr<mlir::Pass> createPulseUnrollClassicalLoopsPass();
/// pulse-lower-selects with no parameter taken at run time, for
/// cudaq-pulse-opt (where the `runtime` option sets them).
std::unique_ptr<mlir::Pass> createPulseLowerSelectsPass();
/// pulse-lower-selects keeping the selects of the parameter kinds in
/// `runtime` (amplitude, phase, frequency).
std::unique_ptr<mlir::Pass>
createPulseLowerSelectsPass(llvm::ArrayRef<std::string> runtime);
} // namespace pulse

namespace qop {
std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass();
} // namespace qop

namespace cudm {
std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass();
} // namespace cudm

namespace pulse {
/// Register the conversions and the pulse passes. Registration is global and
/// not thread-safe: call it once, before any pipeline is parsed.
inline void registerPulsePasses() {
  mlir::registerPass(createPulseToQOpPass);
  mlir::registerPass(qop::createQOpToCuDensityMatPass);
  mlir::registerPass(cudm::createCuDensityMatToLLVMPass);

  mlir::registerPass(createPulseVerifyPass);
  for (const RewriteStage &stage : rewriteStages())
    mlir::registerPass([&stage] { return createPulseRewritePass(&stage); });
  mlir::registerPass(createPulseHoistLoopInvariantsPass);
  mlir::registerPass(createPulseScheduleAsapOptionsPass);
  mlir::registerPass(createPulseScheduleAlapOptionsPass);
  mlir::registerPass(createPulseInlineScheduleBlocksPass);
  mlir::registerPass(createPulseExpandCallsPass);
  mlir::registerPass(createPulseClassicalCheckOptionsPass);
  mlir::registerPass(createPulseFrameChangesPass);
  mlir::registerPass(createPulseUnrollClassicalLoopsPass);
  mlir::registerPass([] { return createPulseLowerSelectsPass(); });
  mlir::registerPass(createPulseScheduleRcpOptionsPass);
}
} // namespace pulse
