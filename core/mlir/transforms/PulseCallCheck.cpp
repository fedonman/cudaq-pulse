// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pulse-verify's check of a module with placed pulse.call ops
// (pulse::checkCalls). Apart from the pass, whose code it does not share.

#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <functional>

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

#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Transforms/Passes.h"

namespace {

static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op);
}

/// The values a callee's `value` may continue: entry-block arguments of
/// `callee` where it continues one, a null value where it continues
/// anything else. Both arms of a branch and a loop's initial and yielded
/// value are followed.
static void rootsOf(mlir::func::FuncOp callee, mlir::Value first,
                    llvm::SmallVectorImpl<mlir::Value> &roots,
                    llvm::DenseSet<mlir::Value> &seen) {
  // A stack of its own, not the machine's: a callee another compiler wrote
  // (pulse.load) may chain as many values as it has ops.
  llvm::SmallVector<mlir::Value, 8> open{first};
  // The first of two is followed first, as a recursive walk would.
  auto both = [&](mlir::Value one, mlir::Value other) {
    open.push_back(other);
    open.push_back(one);
  };
  while (!open.empty()) {
    mlir::Value value = open.pop_back_val();
    if (!value || !seen.insert(value).second)
      continue;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() == &callee.getBody().front()) {
        roots.push_back(value);
        continue;
      }
      mlir::Operation *parent = argument.getOwner()->getParentOp();
      if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
          loop && argument.getArgNumber() > 0) {
        unsigned index = argument.getArgNumber() - 1;
        both(loop.getInitArgs()[index],
             loop.getBody()->getTerminator()->getOperand(index));
        continue;
      }
    } else {
      auto result = mlir::cast<mlir::OpResult>(value);
      mlir::Operation *op = result.getOwner();
      unsigned index = result.getResultNumber();
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        both(loop.getInitArgs()[index],
             loop.getBody()->getTerminator()->getOperand(index));
        continue;
      }
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        both(branch.thenYield().getOperand(index),
             branch.getElseRegion().empty()
                 ? mlir::Value()
                 : branch.elseYield().getOperand(index));
        continue;
      }
    }
    if (mlir::Value before = pulse::linearPredecessor(value))
      open.push_back(before);
    else
      roots.push_back(mlir::Value());
  }
}

/// The rules for a function a pulse.call runs: it takes its lines and links
/// as arguments (it acquires none), and returns each line, tone and link it
/// takes, in the order it takes them, so a call's k-th line (tone, link)
/// result continues its k-th line (tone, link) operand. With a recorded
/// length, every timed op of it is placed, and the length is where its last
/// op ends, placed frame changes included.
static mlir::LogicalResult checkCallee(mlir::func::FuncOp callee) {
  bool failed = false;
  auto threaded = [](mlir::Type type) {
    return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                     pulse::ToneType, classical::LinkType>(type);
  };
  callee.walk([&](mlir::Operation *op) {
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp,
                  classical::GetLinkOp>(op)) {
      op->emitError("@") << callee.getSymName()
                         << " is run by pulse.call, and takes its lines "
                            "and links as arguments; it cannot acquire one";
      failed = true;
    }
  });
  mlir::Operation *ret = callee.getBody().front().getTerminator();
  auto args = callee.getArguments();
  llvm::DenseMap<mlir::Type, std::pair<unsigned, unsigned>> counts;
  for (mlir::BlockArgument arg : args)
    if (threaded(arg.getType()))
      ++counts[arg.getType()].first;
  for (auto [index, value] : llvm::enumerate(ret->getOperands())) {
    mlir::Type type = value.getType();
    if (!threaded(type))
      continue;
    unsigned rank = counts[type].second++;
    llvm::SmallVector<mlir::Value, 2> roots;
    llvm::DenseSet<mlir::Value> seen;
    rootsOf(callee, value, roots, seen);
    bool same = !roots.empty() && llvm::all_of(roots, [&](mlir::Value root) {
      auto argument = mlir::dyn_cast_or_null<mlir::BlockArgument>(root);
      if (!argument)
        return false;
      unsigned argRank = 0;
      for (mlir::BlockArgument earlier :
           args.take_front(argument.getArgNumber()))
        argRank += earlier.getType() == type;
      return argRank == rank;
    });
    if (!same) {
      ret->emitError("@") << callee.getSymName() << " returns as result "
                          << index << " a " << type
                          << " that is not its argument of that type in the "
                             "same place; a function pulse.call runs returns "
                             "each line, tone and link it takes, in the order "
                             "it takes them";
      failed = true;
    }
  }
  for (auto &[type, count] : counts)
    if (count.first != count.second) {
      ret->emitError("@") << callee.getSymName() << " takes " << count.first
                          << " of " << type << " and returns " << count.second
                          << "; a function pulse.call runs returns each line, "
                             "tone and link it takes";
      failed = true;
    }
  callee.walk([&](pulse::ScheduleBlockOp block) {
    block.emitError("@") << callee.getSymName()
                         << " is run by pulse.call; write its schedule blocks "
                            "out first";
    failed = true;
  });
  auto recorded =
      callee->getAttrOfType<mlir::IntegerAttr>(pulse::kLengthCyclesAttr);
  if (failed || !recorded)
    return mlir::failure(failed);
  callee.walk([&](mlir::scf::ForOp loop) {
    if (!pulse::constantTripCount(loop)) {
      loop.emitError("@") << callee.getSymName()
                          << " records a length, and a loop in it needs "
                             "constant bounds";
      failed = true;
    }
  });
  if (failed)
    return mlir::failure();
  int64_t end = 0;
  callee.walk([&](mlir::Operation *op) {
    std::optional<int64_t> start, duration;
    llvm::ArrayRef<int64_t> periods;
    if (auto call = mlir::dyn_cast<pulse::CallOp>(op)) {
      start = call.getStartCycles();
      duration = call.getDurationCycles();
      periods = call.getPeriodCycles();
    } else if (isTimedOp(op) ||
               mlir::isa<classical::SubmitOp, classical::WaitOp>(op)) {
      start = pulse::startCycles(op);
      duration = pulse::durationCycles(op);
      periods = pulse::periodCycles(op);
    } else if (mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                         pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op)) {
      // A placed frame change is written out at its start too, so it ends
      // the callee as well (it lasts no time unless it says otherwise). One
      // not placed yet is placed after the call is written out
      // (pulse-frame-changes), as in any program.
      start = pulse::startCycles(op);
      if (!start)
        return;
      duration = pulse::durationCycles(op).value_or(0);
      periods = pulse::periodCycles(op);
    } else {
      return;
    }
    if (!start || !duration) {
      op->emitError("operation is not placed, but @")
          << callee.getSymName() << " records a length ("
          << pulse::kLengthCyclesAttr << ")";
      failed = true;
      return;
    }
    int64_t at = *start + *duration;
    for (auto [loop, period] :
         llvm::zip(pulse::enclosingLoops(op, callee), periods)) {
      int64_t trips = *pulse::constantTripCount(loop);
      if (trips <= 0)
        return;
      at += (trips - 1) * period;
    }
    end = std::max(end, at);
  });
  if (!failed && end != recorded.getInt()) {
    callee.emitError("records a length of ")
        << recorded.getInt() << " cycles, but its operations end at " << end;
    failed = true;
  }
  return mlir::failure(failed);
}

} // namespace

namespace pulse {
/// Whether a pulse.call has been verified in this process
/// (PulseCallOp.cpp): until one has, no module holds one that matters.
bool callsVerified();
/// Refuses each pulse.call of `function`: the schedule passes do not place
/// one yet.
mlir::LogicalResult refuseCalls(mlir::func::FuncOp function);
/// checkCallee, for a caller that writes calls out itself (pulse.load).
mlir::LogicalResult checkCalledFunction(mlir::func::FuncOp callee);
} // namespace pulse

mlir::LogicalResult pulse::checkCalledFunction(mlir::func::FuncOp callee) {
  return checkCallee(callee);
}

bool pulse::moduleHasCalls(mlir::ModuleOp module) {
  if (!callsVerified())
    return false;
  return module
      .walk([](pulse::CallOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

mlir::LogicalResult pulse::refuseCalls(mlir::func::FuncOp function) {
  if (!callsVerified())
    return mlir::success();
  mlir::WalkResult found = function.walk([](pulse::CallOp call) {
    call.emitError("the schedule passes do not place a pulse.call yet; "
                   "write the ops of @")
        << call.getCallee() << " in its place before scheduling";
    return mlir::WalkResult::interrupt();
  });
  return mlir::failure(found.wasInterrupted());
}

/// A module with pulse.call: every function a call runs keeps the rules for
/// one (checkCallee), no call runs back into itself, and the program, with
/// its calls written out (pulse::expandCalls), passes every check. Errors in
/// a written-out call name the call.
///
/// The module is checked as it is written out only: a call is judged by the
/// ops it writes out, so a line or tone a callee hands back without playing
/// on it is not used by the call.
mlir::LogicalResult pulse::checkCalls(mlir::ModuleOp module) {
  mlir::SymbolTableCollection symbols;
  llvm::SetVector<mlir::Operation *> callees;
  module.walk([&](pulse::CallOp call) {
    if (auto target = symbols.lookupNearestSymbolFrom<mlir::func::FuncOp>(
            call, call.getCalleeAttr());
        target && !target.isExternal())
      callees.insert(target);
  });
  bool failed = false;
  enum Colour { White, Grey, Black };
  llvm::DenseMap<mlir::Operation *, Colour> colour;
  std::function<void(mlir::func::FuncOp)> visit =
      [&](mlir::func::FuncOp function) {
        colour[function] = Grey;
        function.walk([&](pulse::CallOp call) {
          auto target = symbols.lookupNearestSymbolFrom<mlir::func::FuncOp>(
              call, call.getCalleeAttr());
          if (!target || target.isExternal())
            return;
          if (colour.lookup(target) == Grey) {
            call.emitOpError("calls @")
                << call.getCallee() << ", which calls back into it";
            failed = true;
          } else if (colour.lookup(target) == White) {
            visit(target);
          }
        });
        colour[function] = Black;
      };
  for (auto function : module.getOps<mlir::func::FuncOp>())
    if (colour.lookup(function) == White && !function.isExternal())
      visit(function);
  for (mlir::Operation *callee : callees)
    failed |= mlir::failed(checkCallee(mlir::cast<mlir::func::FuncOp>(callee)));
  if (failed)
    return mlir::failure();
  // The copy is checked quietly first; only a copy that fails is written
  // out again with each op's location naming its call, to report.
  auto checkCopy = [&](bool report) {
    // The quiet check gives the copy's warnings, the program's, and keeps
    // its errors; the check that reports gives the errors only.
    mlir::ScopedDiagnosticHandler quiet(
        module.getContext(), [report](mlir::Diagnostic &diagnostic) {
          bool warning =
              diagnostic.getSeverity() == mlir::DiagnosticSeverity::Warning;
          return mlir::success(report ? warning : !warning);
        });
    mlir::OwningOpRef<mlir::ModuleOp> written = module.clone();
    if (mlir::failed(pulse::expandCalls(*written, /*callSites=*/report)))
      return mlir::failure();
    mlir::PassManager pm(module.getContext());
    // The module is verified; writing calls out keeps it so.
    pm.enableVerifier(false);
    pm.addPass(pulse::createPulseVerifyPass());
    return pm.run(*written);
  };
  if (mlir::succeeded(checkCopy(false)))
    return mlir::success();
  (void)checkCopy(true);
  return mlir::failure();
}
