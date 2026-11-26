// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Module-level verification pass for the Pulse dialect.
// Checks linearity, monotone time, measurement causality, and waveform
// validity beyond what individual op verifiers can check, through structured
// control flow: the arms of an `scf.if` are alternative paths, and an
// `scf.for` body runs once per iteration.

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <functional>

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/ClassicalTiming.h"
#include "cudaq-pulse/Analysis/Durations.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Analysis/Untimed.h"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace pulse {
/// Whether `module` holds a pulse.call (PulseCallCheck.cpp).
bool moduleHasCalls(mlir::ModuleOp module);
/// pulse-verify's check of a module with placed calls (PulseCallCheck.cpp).
mlir::LogicalResult checkCalls(mlir::ModuleOp module);
} // namespace pulse

namespace {

static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op);
}

/// The measurement results a branch condition is computed from, not
/// through a link or an answer (see the scheduler's conditionMeasurements):
/// pulse-classical-check checks a branch on an answer.
static void conditionMeasurements(mlir::Value value,
                                  llvm::SmallVectorImpl<mlir::Value> &found,
                                  llvm::DenseSet<mlir::Value> &seen) {
  // The values still to follow, the next one last: an operation's operands
  // in their order, each with all it is computed from before the next.
  llvm::SmallVector<mlir::Value, 16> todo{value};
  while (!todo.empty()) {
    mlir::Value next = todo.pop_back_val();
    if (mlir::isa<classical::LinkType>(next.getType()) ||
        !seen.insert(next).second)
      continue;
    if (mlir::isa<pulse::MeasurementType>(next.getType())) {
      found.push_back(next);
      continue;
    }
    mlir::Operation *def = next.getDefiningOp();
    if (!def || mlir::isa<classical::WaitOp>(def))
      continue;
    for (mlir::Value operand : llvm::reverse(def->getOperands()))
      todo.push_back(operand);
  }
}

/// Checks the schedule attributes of one function, walking the program in
/// execution order. Loops are checked on their first two iterations and
/// their last: with one period per line, those bound every iteration. An op
/// is reported once, at the first iteration it is wrong in.
class TimingChecker {
  pulse::AllocationTracer trace;

public:
  bool hadError = false;

  void checkBlock(mlir::Block &block) {
    for (mlir::Operation &op : block.without_terminator())
      checkOp(&op);
  }

  /// A timed or placed function (pulse.schedule_state) is scheduled as a
  /// whole: every timed op that runs has a schedule.
  void checkFunction(mlir::func::FuncOp function) {
    checkBlock(function.getBody().front());
    pulse::ScheduleState state = pulse::scheduleStateOf(function);
    if (state != pulse::ScheduleState::Unscheduled && unscheduled &&
        report(unscheduled))
      unscheduled->emitError("operation is not scheduled, but its function is ")
          << pulse::stringifyScheduleState(state);
  }

private:
  /// Whether to report a fault of `op`: not if it has been reported.
  bool report(mlir::Operation *op) {
    hadError = true;
    return reported.insert(op).second;
  }

  void error(mlir::Operation *op, const llvm::Twine &message) {
    if (report(op))
      op->emitError(message);
  }

  void checkOp(mlir::Operation *op) {
    if (isTimedOp(op))
      return checkTimed(op);
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return checkLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return checkIf(branch);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        checkBlock(block);
  }

  void checkTimed(mlir::Operation *op) {
    auto start = pulse::scheduledStart(op, iteration);
    auto duration = pulse::scheduledDuration(op);
    if (!start && !duration) {
      if (!unscheduled)
        unscheduled = op;
      return;
    }
    if (!start)
      return error(op, "a scheduled operation needs start_cycles");
    if (!duration)
      return error(op, "a scheduled operation needs duration_cycles");
    if (!op->hasAttr("classical.variable")) {
      std::optional<int64_t> cycles = cyclesOf(op);
      if (!cycles)
        return error(op, "a scheduled operation needs a known length");
      if (*cycles != *duration && report(op))
        op->emitError("duration_cycles is ")
            << *duration << ", but the operation takes " << *cycles
            << " cycles";
    }
    if (pulse::periodCycles(op).size() != iteration.size()) {
      error(op,
            "scheduled operation needs one period_cycles entry per enclosing "
            "scf.for");
      return;
    }
    if (*start < 0 && report(op))
      op->emitError("operation is scheduled before time zero: start ")
          << *start;
    mlir::Operation *physicalLine = trace(op->getOperand(0));
    if (!physicalLine) {
      error(op, "cannot trace timed operation to a physical line");
      return;
    }
    if (*start < measured && report(op))
      op->emitError("operation in a measurement-conditioned branch starts at ")
          << *start << ", before its measurement completes at " << measured;
    auto it = lastEnd.find(physicalLine);
    if (it != lastEnd.end() && *start < it->second && report(op))
      op->emitError("operation overlaps or precedes its physical-line "
                    "predecessor: start ")
          << *start << " < prior end " << it->second;
    lastEnd[physicalLine] =
        std::max(lastEnd.lookup(physicalLine), *start + *duration);
  }

  void checkLoop(mlir::scf::ForOp loop) {
    auto trips = pulse::constantTripCount(loop);
    if (!trips) {
      for (mlir::Operation &inner : loop.getBody()->without_terminator())
        if (isTimedOp(&inner) && pulse::hasStart(&inner)) {
          error(&inner, "a scheduled loop needs constant bounds");
          return;
        }
      return;
    }
    // Checking iterations relies on one period per line.
    if (!checkPeriods(loop))
      return;
    iteration.push_back(0);
    for (int64_t k = 0; k < *trips; ++k) {
      if (k == 2 && *trips > 3)
        k = *trips - 1;
      iteration.back() = k;
      checkBlock(*loop.getBody());
    }
    iteration.pop_back();
  }

  /// Within one iteration of `loop`, every op on a physical line must move
  /// by the same non-negative period from one iteration to the next.
  bool checkPeriods(mlir::scf::ForOp loop) {
    size_t level = iteration.size();
    bool ok = true;
    llvm::DenseMap<mlir::Operation *, int64_t> periodOf;
    loop.getBody()->walk([&](mlir::Operation *op) {
      if (!isTimedOp(op))
        return;
      llvm::ArrayRef<int64_t> periods = pulse::periodCycles(op);
      if (periods.size() <= level)
        return;
      int64_t period = periods[level];
      if (period < 0) {
        ok = false;
        return error(op, "period_cycles must not be negative");
      }
      mlir::Operation *line = trace(op->getOperand(0));
      auto [it, inserted] = periodOf.try_emplace(line, period);
      if (!inserted && it->second != period) {
        ok = false;
        error(op, "operations on one line in a loop must share its period");
      }
    });
    return ok;
  }

  void checkIf(mlir::scf::IfOp branch) {
    int64_t savedMeasured = measured;
    llvm::SmallVector<mlir::Value> measurements;
    llvm::DenseSet<mlir::Value> seen;
    conditionMeasurements(branch.getCondition(), measurements, seen);
    for (mlir::Value measurement : measurements)
      if (auto readout = measurement.getDefiningOp<pulse::ReadoutOp>()) {
        size_t depth = pulse::enclosingLoops(readout).size();
        auto start = pulse::scheduledStart(
            readout, llvm::ArrayRef(iteration).take_front(depth));
        auto duration = pulse::scheduledDuration(readout);
        if (start && duration)
          measured = std::max(measured, *start + *duration);
      }

    auto before = lastEnd;
    checkBlock(*branch.thenBlock());
    if (!branch.getElseRegion().empty()) {
      auto afterThen = std::move(lastEnd);
      lastEnd = std::move(before);
      checkBlock(*branch.elseBlock());
      for (auto [line, end] : afterThen)
        lastEnd[line] = std::max(lastEnd.lookup(line), end);
    }
    measured = savedMeasured;
  }

  /// The cycles a drive, readout or wait takes, as the scheduler places it.
  static std::optional<int64_t> cyclesOf(mlir::Operation *op) {
    if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
      return pulse::waveformCycles(drive.getPulse());
    if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
      return pulse::waveformCycles(readout.getPulse());
    return pulse::waitCycles(mlir::cast<pulse::WaitOp>(op));
  }

  llvm::DenseMap<mlir::Operation *, int64_t> lastEnd;
  llvm::DenseSet<mlir::Operation *> reported;
  /// The first timed op without a schedule, if any.
  mlir::Operation *unscheduled = nullptr;
  llvm::SmallVector<int64_t, 2> iteration;
  int64_t measured = std::numeric_limits<int64_t>::min();
};

/// What an op verifier cannot check without looking at the ops that compute
/// its operands. A duration, sigma or risefall is checked once it is a
/// constant (or arithmetic on constants: folding must not make valid IR
/// invalid); one computed from the function's arguments is checked when they
/// are bound.
class WaveformCheck {
public:
  bool hadError = false;

  void visit(mlir::Operation *op) {
    if (op->getNumResults() == 1 &&
        mlir::isa<pulse::WaveformType>(op->getResult(0).getType()))
      return visitWaveform(op);
    if (auto outcome = mlir::dyn_cast<pulse::OutcomeOp>(op))
      return checkMeasurement(op, outcome.getMeasurement());
    if (auto isSet = mlir::dyn_cast<pulse::IsSetOp>(op))
      return checkMeasurement(op, isSet.getMeasurement());
    if (auto wait = mlir::dyn_cast<pulse::WaitOp>(op))
      return checkBounds(wait);
  }

private:
  void visitWaveform(mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<pulse::SquarePulseOp, pulse::CosinePulseOp, pulse::CustomOp>(
            [&](auto op) { checkDuration(op, op.getDuration()); })
        .Case<pulse::GaussianPulseOp, pulse::DRAGPulseOp, pulse::TanhRampOp>(
            [&](auto op) {
              checkDuration(op, op.getDuration());
              checkSigma(op, op.getSigma());
            })
        .Case([&](pulse::GaussianSquarePulseOp op) {
          checkDuration(op, op.getDuration());
          checkSigma(op, op.getSigma());
          checkRisefall(op);
        })
        .Case<pulse::PulseAddOp, pulse::PulseSubOp, pulse::PulseMulOp>(
            [&](auto op) { checkEqualLengths(op, op.getLhs(), op.getRhs()); });
  }

  void checkDuration(mlir::Operation *op, mlir::Value duration) {
    auto cycles = pulse::constantInteger(duration);
    if (cycles && *cycles <= 0) {
      op->emitOpError("duration must be positive, got ") << *cycles;
      hadError = true;
    }
  }

  /// NaN compares false with everything, so `sigma <= 0` would pass it.
  void checkSigma(mlir::Operation *op, mlir::Value sigma) {
    auto constant = sigma.getDefiningOp<mlir::arith::ConstantOp>();
    auto value = constant ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())
                          : nullptr;
    if (!value)
      return;
    double number = value.getValueAsDouble();
    if (!(number > 0.0) || !std::isfinite(number)) {
      op->emitOpError("sigma must be positive and finite, got ") << number;
      hadError = true;
    }
  }

  void checkRisefall(pulse::GaussianSquarePulseOp op) {
    auto risefall = pulse::constantInteger(op.getRisefall());
    if (!risefall)
      return;
    if (*risefall <= 0) {
      op.emitOpError("risefall must be positive, got ") << *risefall;
      hadError = true;
      return;
    }
    auto duration = pulse::constantInteger(op.getDuration());
    if (duration && *duration > 0 && *risefall > *duration / 2) {
      op.emitOpError("risefall (")
          << *risefall << ") exceeds half the duration (" << *duration << ")";
      hadError = true;
    }
  }

  /// Waveform arithmetic is element-wise, so both operands must have the
  /// same length; otherwise one would be silently truncated.
  void checkEqualLengths(mlir::Operation *op, mlir::Value lhs,
                         mlir::Value rhs) {
    auto left = lengthOf(lhs);
    auto right = lengthOf(rhs);
    if (left && right && *left != *right) {
      op->emitOpError("operands have durations ")
          << *left << " and " << *right
          << "; waveform arithmetic needs equal durations";
      hadError = true;
    }
  }

  /// The length of `waveform` in cycles, if known: from either operand of
  /// waveform arithmetic. Each value is computed once.
  std::optional<int64_t> lengthOf(mlir::Value waveform) {
    auto known = lengths.find(waveform);
    if (known != lengths.end())
      return known->second;
    std::optional<int64_t> length;
    mlir::Operation *def = waveform.getDefiningOp();
    if (mlir::isa_and_nonnull<pulse::PulseAddOp, pulse::PulseSubOp,
                              pulse::PulseMulOp>(def)) {
      length = lengthOf(def->getOperand(0));
      if (!length)
        length = lengthOf(def->getOperand(1));
    } else if (mlir::isa_and_nonnull<pulse::PulseScaleOp, pulse::PulseNegOp>(
                   def)) {
      length = lengthOf(def->getOperand(0));
    } else {
      length = pulse::waveformCycles(waveform);
    }
    lengths[waveform] = length;
    return length;
  }

  /// A measurement comes from a readout on every path: through either arm
  /// of an scf.if, a loop's initial and yielded value (only the yielded one
  /// for the result of a loop that runs), a schedule block's operand and
  /// yielded value. A function argument is the caller's.
  void checkMeasurement(mlir::Operation *op, mlir::Value measurement) {
    if (measurement.getDefiningOp<pulse::ReadoutOp>())
      return;
    if (mlir::Operation *wrong = sourceOtherThanReadout(measurement)) {
      op->emitOpError("measurement must be produced by pulse.readout, got ")
          << wrong->getName();
      hadError = true;
    }
  }

  /// An op that is not a readout and that `measurement` may come from, or
  /// null. Each value is followed once; a loop's carried value is assumed
  /// to come from a readout while it is being followed.
  mlir::Operation *sourceOtherThanReadout(mlir::Value measurement) {
    auto [it, inserted] = sources.try_emplace(measurement, nullptr);
    if (!inserted)
      return it->second;
    auto follow = [&](mlir::Value a, mlir::Value b = {}) -> mlir::Operation * {
      if (mlir::Operation *wrong = sourceOtherThanReadout(a))
        return wrong;
      return b ? sourceOtherThanReadout(b) : nullptr;
    };
    mlir::Operation *wrong = nullptr;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(measurement)) {
      mlir::Operation *parent = argument.getOwner()->getParentOp();
      unsigned index = argument.getArgNumber();
      if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
        if (index > 0)
          wrong =
              follow(loop.getInitArgs()[index - 1],
                     loop.getBody()->getTerminator()->getOperand(index - 1));
      } else if (auto block =
                     mlir::dyn_cast_or_null<pulse::ScheduleBlockOp>(parent)) {
        wrong = follow(block.getInputs()[index]);
      }
    } else {
      auto result = mlir::cast<mlir::OpResult>(measurement);
      mlir::Operation *def = result.getOwner();
      unsigned index = result.getResultNumber();
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def))
        wrong = follow(branch.thenYield().getOperand(index),
                       branch.elseYield().getOperand(index));
      else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(def)) {
        // After an iteration, the result is what the last one yielded.
        mlir::Value yielded =
            loop.getBody()->getTerminator()->getOperand(index);
        auto trips = pulse::constantTripCount(loop);
        wrong = trips && *trips > 0
                    ? follow(yielded)
                    : follow(loop.getInitArgs()[index], yielded);
      } else if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(def))
        wrong =
            follow(block.getBody().front().getTerminator()->getOperand(index));
      else if (!mlir::isa<pulse::ReadoutOp>(def))
        wrong = def;
    }
    sources[measurement] = wrong;
    return wrong;
  }

  /// A variable wait whose length is a constant must be within its bounds.
  void checkBounds(pulse::WaitOp wait) {
    auto lowest = wait->getAttrOfType<mlir::IntegerAttr>("pulse.min_cycles");
    auto conversion =
        wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>();
    if (!lowest || !wait->hasAttr("pulse.variable_wait") || !conversion)
      return;
    auto highest = wait->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles");
    auto cycles = pulse::constantInteger(conversion.getCycles());
    if (!cycles || (*cycles >= lowest.getInt() &&
                    (!highest || *cycles <= highest.getInt())))
      return;
    wait.emitOpError("a pulse.variable_wait wait of ")
        << *cycles << " cycles is outside its bounds [" << lowest.getInt()
        << ", " << (highest ? std::to_string(highest.getInt()) : "none") << "]";
    hadError = true;
  }

  llvm::DenseMap<mlir::Value, std::optional<int64_t>> lengths;
  llvm::DenseMap<mlir::Value, mlir::Operation *> sources;
};

struct PulseVerifyPass
    : public mlir::PassWrapper<PulseVerifyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseVerifyPass)

  llvm::StringRef getArgument() const override { return "pulse-verify"; }
  llvm::StringRef getDescription() const override {
    return "Module-level verification of Pulse IR constraints";
  }

  void runOnOperation() override {
    auto module = getOperation();
    // A module with pulse.call is checked as it is written out.
    if (pulse::moduleHasCalls(module)) [[unlikely]] {
      if (mlir::failed(pulse::checkCalls(module)))
        signalPassFailure();
      return;
    }
    bool hadError = false;

    // Check 1: Linearity -- lines must be consumed. The warning names the
    // qudit instead of attaching the op: printing an op numbers every value
    // of its function, which is linear in the program per warning.
    auto warnUnused = [](mlir::Operation *op, mlir::Value line,
                         llvm::StringRef kind) {
      if (!line.use_empty())
        return;
      auto diagnostic = mlir::emitWarning(op->getLoc())
                        << kind << " line allocated but never used";
      if (int64_t qudit = pulse::quditOf(op); qudit >= 0)
        diagnostic << " (qudit " << qudit << ")";
    };
    module.walk([&](pulse::GetDriveLineOp op) {
      warnUnused(op, op.getLine(), "drive");
    });
    module.walk([&](pulse::GetReadoutLineOp op) {
      warnUnused(op, op.getLine(), "readout");
    });
    // A line or tone is consumed at most once on every path: its uses must
    // lie in different arms of a conditional, and none inside a loop that
    // does not carry it. A use that only reads it (a time read) consumes
    // nothing.
    auto checkLinear = [&](mlir::Value value, mlir::Operation *where) {
      if (!pulse::isLinearPulseType(value.getType()))
        return;
      llvm::SmallVector<mlir::Operation *> users;
      for (mlir::OpOperand &use : value.getUses())
        if (!pulse::readsWithoutConsuming(use.getOwner()))
          users.push_back(use.getOwner());
      if (!pulse::allMutuallyExclusive(users)) {
        where->emitError("linear pulse value is consumed more than once "
                         "on one path");
        hadError = true;
        return;
      }
      for (mlir::Operation *user : users)
        for (mlir::Operation *loop = user->getParentOp(); loop;
             loop = loop->getParentOp()) {
          if (loop == where || loop->isAncestor(where))
            break;
          if (mlir::isa<mlir::scf::ForOp>(loop)) {
            user->emitError("linear pulse value from outside an scf.for is "
                            "used in its body; carry it through iter_args");
            hadError = true;
            return;
          }
        }
    };
    // One walk for each op's own checks: an op the schedule passes cannot
    // time, linearity, the values of waveforms
    // and of the waits and measurements that use them (WaveformCheck), and
    // acquisitions. Timing and lowering identify a line by the op that
    // acquires it, so an acquisition in a loop body cannot stand for a new
    // line per iteration.
    WaveformCheck waveforms;
    module.walk([&](mlir::Operation *op) {
      // The schedule passes would time an op they do not know as nothing.
      if (pulse::untimed(op)) {
        pulse::emitUntimed(op);
        hadError = true;
      }
      for (mlir::Value result : op->getResults())
        checkLinear(result, op);
      for (mlir::Region &region : op->getRegions())
        for (mlir::Block &block : region)
          for (mlir::Value arg : block.getArguments())
            checkLinear(arg, op);
      waveforms.visit(op);
      if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        return;
      if (op->getParentOfType<mlir::scf::ForOp>()) {
        op->emitError("a line cannot be acquired inside an scf.for; acquire "
                      "it before the loop and carry it through iter_args");
        hadError = true;
      }
      if (mlir::failed(pulse::checkOneAcquisition(op)))
        hadError = true;
    });

    // Check 1b: classical calls. A line-like link cannot be acquired in a
    // loop either (V1 and V2 of the classical-call rules follow).
    module.walk([&](classical::GetLinkOp op) {
      if (!op->getParentOfType<mlir::scf::ForOp>())
        return;
      op->emitError("a link cannot be acquired inside an scf.for; acquire it "
                    "before the loop and carry it through iter_args");
      hadError = true;
    });
    // V1: an answer that arrives while the program runs cannot change the
    // timeline, so no duration or loop bound may be computed from one.
    llvm::DenseMap<mlir::Value, bool> fromAnswer;
    std::function<bool(mlir::Value)> dependsOnAnswer =
        [&](mlir::Value value) -> bool {
      auto [it, inserted] = fromAnswer.try_emplace(value, false);
      if (!inserted)
        return it->second;
      mlir::Operation *def = value.getDefiningOp();
      bool result = false;
      if (def && mlir::isa<classical::WaitOp>(def))
        result = true;
      else if (def && def->getNumRegions() == 0)
        result = llvm::any_of(def->getOperands(), dependsOnAnswer);
      fromAnswer[value] = result;
      return result;
    };
    bool hasAnswers = false;
    module.walk([&](classical::WaitOp) { hasAnswers = true; });
    if (hasAnswers) {
      auto rejectDuration = [&](mlir::Operation *op, mlir::Value duration) {
        if (!dependsOnAnswer(duration))
          return;
        op->emitError("a duration cannot depend on the answer of a classical "
                      "call: the answer arrives while the program runs, after "
                      "its timeline is fixed; branch on the answer to choose "
                      "between durations instead");
        hadError = true;
      };
      module.walk([&](mlir::Operation *op) {
        if (auto w = mlir::dyn_cast<pulse::SquarePulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::GaussianPulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::DRAGPulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::CosinePulseOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::TanhRampOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto w = mlir::dyn_cast<pulse::GaussianSquarePulseOp>(op)) {
          rejectDuration(op, w.getDuration());
          rejectDuration(op, w.getRisefall());
        } else if (auto w = mlir::dyn_cast<pulse::CustomOp>(op))
          rejectDuration(op, w.getDuration());
        else if (auto d = mlir::dyn_cast<pulse::DurationFromIntOp>(op)) {
          // A wait marked variable may take its length from an answer.
          bool variable = !d->use_empty() &&
                          llvm::all_of(d->getUsers(), [](mlir::Operation *u) {
                            return mlir::isa<pulse::WaitOp>(u) &&
                                   u->hasAttr("classical.variable");
                          });
          if (!variable)
            rejectDuration(op, d.getCycles());
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
          for (mlir::Value bound :
               {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()})
            if (dependsOnAnswer(bound)) {
              loop.emitError("a loop bound cannot depend on the answer of a "
                             "classical call");
              hadError = true;
              break;
            }
      });
    }
    // V2 for variable waits: a region whose length is known only at run time
    // cannot sit in a schedule block or in one arm of a branch.
    module.walk([&](pulse::WaitOp op) {
      if (!op->hasAttr("classical.variable"))
        return;
      if (!op->hasAttr("pulse.variable_wait")) {
        op.emitError(
            "a classical.variable wait must also be pulse.variable_wait, "
            "with its bounds in pulse.min_cycles and pulse.max_cycles");
        hadError = true;
      } else if (op->getParentOfType<pulse::ScheduleBlockOp>()) {
        op.emitError(classical::kVariableWaitInBlock);
        hadError = true;
      } else if (op->getParentOfType<mlir::scf::IfOp>()) {
        op.emitError("a variable wait cannot be inside one arm of an scf.if; "
                     "let the answer be 0 where no wait is wanted");
        hadError = true;
      }
    });
    // V2: a blocking wait stops the whole program until the answer arrives,
    // so it cannot sit in a schedule block, which has a fixed length, or in
    // one arm of a branch.
    mlir::SymbolTableCollection symbols;
    module.walk([&](classical::WaitOp op) {
      auto callable = symbols.lookupNearestSymbolFrom<classical::CallableOp>(
          op, op.getCalleeAttr());
      if (!callable || callable.getMode() != "blocking")
        return;
      if (op->getParentOfType<pulse::ScheduleBlockOp>()) {
        op.emitError(classical::kBlockingWaitInBlock);
        hadError = true;
      } else if (op->getParentOfType<mlir::scf::IfOp>()) {
        op.emitError("a blocking wait cannot be inside one arm of an scf.if; "
                     "wait before or after the branch");
        hadError = true;
      }
    });
    // A call of a controller that blocks on calls holds every line, which a
    // schedule block, placed whole, cannot.
    if (pulse::blocksOnCalls(module))
      module.walk([&](classical::SubmitOp op) {
        if (op->getParentOfType<pulse::ScheduleBlockOp>()) {
          op.emitError(classical::kBlockingCallInBlock);
          hadError = true;
        }
      });

    hadError |= waveforms.hadError;
    // A wait whose length is not a constant is scheduled at a placeholder
    // length until its arguments are bound; only a variable wait, which
    // carries its bounds, may be.
    module.walk([&](pulse::WaitOp op) {
      auto conversion =
          op.getDuration().getDefiningOp<pulse::DurationFromIntOp>();
      if (op->hasAttr("pulse.variable_wait") ||
          (conversion && pulse::constantInteger(conversion.getCycles())))
        return;
      op.emitError(
          "a wait whose length is not a constant must be marked "
          "pulse.variable_wait, with its bounds in pulse.min_cycles and "
          "pulse.max_cycles");
      hadError = true;
    });

    // Check 3: Monotone, non-overlapping time on each physical line, and no
    // conditional operation before its measurement. SSA line values change
    // after every operation, so trace them back to allocation.
    module.walk([&](mlir::func::FuncOp funcOp) {
      if (funcOp.isExternal())
        return;
      TimingChecker checker;
      checker.checkFunction(funcOp);
      hadError |= checker.hadError;
    });

    if (hadError)
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseVerifyPass() {
  return std::make_unique<PulseVerifyPass>();
}
} // namespace pulse
