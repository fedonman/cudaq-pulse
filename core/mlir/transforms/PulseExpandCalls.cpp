// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Writing placed pulse.call ops out (pulse-expand-calls, pulse::expandCalls).
// Apart from the schedule passes, whose code it does not share.

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <functional>
#include <memory>

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseCallOp.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/Affine.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Transforms/Passes.h"

namespace {

static bool isLineType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType>(type);
}

/// For each line argument of `callee`, a function a pulse.call runs, the
/// local end of the last drive, readout or wait on it (in the last iteration
/// of the loops around it), where any plays on it.
static llvm::DenseMap<unsigned, int64_t> lineEnds(mlir::func::FuncOp callee) {
  llvm::DenseMap<unsigned, int64_t> ends;
  mlir::Block &entry = callee.getBody().front();
  llvm::DenseMap<mlir::Value, int> argOf;
  auto rootArg = [&](mlir::Value line) -> int {
    llvm::SmallVector<mlir::Value, 8> chain;
    int found = -1;
    for (mlir::Value value = line; value;
         value = pulse::linearPredecessor(value)) {
      if (auto known = argOf.find(value); known != argOf.end()) {
        found = known->second;
        break;
      }
      chain.push_back(value);
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
          argument && argument.getOwner() == &entry) {
        found = static_cast<int>(argument.getArgNumber());
        break;
      }
    }
    for (mlir::Value value : chain)
      argOf[value] = found;
    return found;
  };
  callee.walk([&](mlir::Operation *op) {
    if (!mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op))
      return;
    llvm::SmallVector<int64_t, 2> last;
    for (mlir::scf::ForOp loop : pulse::enclosingLoops(op, callee)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(1);
      if (trips <= 0)
        return;
      last.push_back(trips - 1);
    }
    std::optional<int64_t> start = pulse::scheduledStart(op, last);
    std::optional<int64_t> duration = pulse::durationCycles(op);
    int arg = rootArg(op->getOperand(0));
    if (!start || !duration || arg < 0)
      return;
    int64_t end = *start + *duration;
    auto [it, inserted] = ends.try_emplace(static_cast<unsigned>(arg), end);
    if (!inserted)
      it->second = std::max(it->second, end);
  });
  return ends;
}

/// Write the placed `call` out: a copy of its callee's ops at the call's
/// start and periods, as a placed schedule block is inlined. A call that lasts
/// longer than its callee holds each line the callee plays on until its end: a
/// wait after the line's last op. With `callSites`, each copied op's location
/// names the call.
static mlir::LogicalResult
expandCall(pulse::CallOp call, mlir::func::FuncOp callee, bool callSites) {
  std::optional<int64_t> start = call.getStartCycles();
  if (!start)
    return call.emitOpError("is not placed; schedule it before writing it "
                            "out");
  if (callee
          .walk([](pulse::ReadoutOp readout) {
            return readout->hasAttr(pulse::kReadoutNumberAttr) ||
                           readout->hasAttr(pulse::kReadoutIterationAttr)
                       ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
          })
          .wasInterrupted())
    return call.emitOpError("calls @")
           << call.getCallee()
           << ", whose readouts are numbered in the kernel "
              "(pulse.readout_number); such a call cannot be written out";
  if (callee
          .walk([](pulse::ScheduleBlockOp) {
            return mlir::WalkResult::interrupt();
          })
          .wasInterrupted())
    return call.emitOpError("calls @")
           << call.getCallee()
           << ", which holds a pulse.schedule_block; write its blocks out "
              "first";
  llvm::ArrayRef<int64_t> outer = call.getPeriodCycles();
  mlir::Block &body = callee.getBody().front();
  mlir::IRMapping mapping;
  mapping.map(body.getArguments(), call.getArgs());
  mlir::OpBuilder builder(call);
  auto shift = [&](mlir::Operation *op) {
    std::optional<int64_t> local = pulse::startCycles(op);
    if (!local)
      return;
    pulse::setStartCycles(op, *start + *local);
    llvm::SmallVector<int64_t> periods(outer);
    llvm::append_range(periods, pulse::periodCycles(op));
    pulse::setPeriodCycles(op, periods);
  };
  for (mlir::Operation &op : body.without_terminator()) {
    mlir::Operation *copy = builder.clone(op, mapping);
    copy->walk(shift);
    if (callSites)
      copy->walk([&](mlir::Operation *inner) {
        inner->setLoc(mlir::CallSiteLoc::get(inner->getLoc(), call.getLoc()));
      });
  }
  llvm::SmallVector<mlir::Value> results;
  for (mlir::Value value : body.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(value));
  // The k-th line result continues the k-th line operand (pulse-verify).
  int64_t duration = call.getDurationCycles().value_or(0);
  auto length =
      callee->getAttrOfType<mlir::IntegerAttr>(pulse::kLengthCyclesAttr);
  if (length && duration > length.getInt()) {
    llvm::DenseMap<unsigned, int64_t> ends = lineEnds(callee);
    // Of each line type, the operands in order: the k-th result of a type
    // continues the k-th operand of it.
    llvm::DenseMap<mlir::Type, llvm::SmallVector<unsigned, 2>> operandsOf;
    for (auto [index, arg] : llvm::enumerate(call.getArgs()))
      if (isLineType(arg.getType()))
        operandsOf[arg.getType()].push_back(static_cast<unsigned>(index));
    llvm::DenseMap<mlir::Type, size_t> rankOf;
    for (mlir::Value &result : results) {
      if (!isLineType(result.getType()))
        continue;
      auto &operands = operandsOf[result.getType()];
      size_t rank = rankOf[result.getType()]++;
      if (rank >= operands.size())
        return call.emitOpError("calls @")
               << call.getCallee() << ", which returns a line it does not take";
      unsigned arg = operands[rank];
      auto end = ends.find(arg);
      if (end == ends.end() || end->second >= duration)
        continue;
      mlir::Location loc = call.getLoc();
      mlir::Value cycles = mlir::arith::ConstantIntOp::create(
          builder, loc, duration - end->second, 64);
      mlir::Value span = pulse::DurationFromIntOp::create(
          builder, loc, pulse::DurationType::get(call.getContext()), cycles);
      auto wait =
          pulse::WaitOp::create(builder, loc, result.getType(), result, span);
      // Its own setters: the generic setTiming has its hot callers.
      wait.setStartCycles(*start + end->second);
      wait.setDurationCycles(duration - end->second);
      wait.setPeriodCycles(outer);
      result = wait->getResult(0);
    }
  }
  call->replaceAllUsesWith(results);
  call->erase();
  return mlir::success();
}

/// Write out every pulse.call of `module` (pulse::expandCalls).
static mlir::LogicalResult expandAllCalls(mlir::ModuleOp module,
                                          bool callSites) {
  mlir::SymbolTable symbols(module);
  llvm::DenseSet<mlir::Operation *> done, active, called;
  bool failed = false;
  std::function<void(mlir::func::FuncOp)> expandIn =
      [&](mlir::func::FuncOp function) {
        if (done.contains(function) || failed)
          return;
        if (!active.insert(function).second) {
          function.emitError("calls itself through pulse.call");
          failed = true;
          return;
        }
        llvm::SmallVector<pulse::CallOp> calls;
        function.walk([&](pulse::CallOp call) { calls.push_back(call); });
        for (pulse::CallOp call : calls) {
          auto callee = symbols.lookup<mlir::func::FuncOp>(call.getCallee());
          if (!callee || callee.isExternal()) {
            call.emitOpError("calls '")
                << call.getCallee() << "', which has no body here";
            failed = true;
            return;
          }
          called.insert(callee);
          expandIn(callee);
          if (failed || mlir::failed(expandCall(call, callee, callSites))) {
            failed = true;
            return;
          }
        }
        active.erase(function);
        done.insert(function);
      };
  for (auto function :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>()))
    expandIn(function);
  if (failed)
    return mlir::failure();
  // A function a call ran plays on the lines it is given, so it is no
  // program of its own: once no call runs it, it is taken out.
  for (mlir::Operation *function : called)
    if (mlir::SymbolTable::symbolKnownUseEmpty(function, module))
      function->erase();
  return mlir::success();
}

/// Write out every pulse.call of the module (pulse::expandCalls).
struct PulseExpandCallsPass
    : public mlir::PassWrapper<PulseExpandCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseExpandCallsPass)

  llvm::StringRef getArgument() const override { return "pulse-expand-calls"; }
  llvm::StringRef getDescription() const override {
    return "Write out placed pulse.call ops at their starts";
  }
  void runOnOperation() override {
    if (mlir::failed(expandAllCalls(getOperation(), false)))
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseExpandCallsPass() {
  return std::make_unique<PulseExpandCallsPass>();
}

mlir::LogicalResult expandCalls(mlir::ModuleOp module, bool callSites) {
  return expandAllCalls(module, callSites);
}
} // namespace pulse
