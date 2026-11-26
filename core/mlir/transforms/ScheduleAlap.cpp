// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deterministic ASAP, ALAP and resource-constrained scheduling passes for the
// Pulse dialect, over structured control flow.
//
// A pass simulates the program in order. Every iteration of an `scf.for` is
// simulated until the loop-carried lines all advance by the same amount per
// iteration, after which the remaining iterations are known to repeat that
// shift. The two arms of a measurement-conditioned `scf.if` both start once
// the measurement is available, and the lines leaving the `if` are ready when
// the later arm is done.
//
// Results are written as `start_cycles` / `duration_cycles`, plus
// `period_cycles` for ops in loops (see Analysis/Timing.h). When an op's start
// is not affine in its iteration (a loop's first or last iterations behave
// differently), that iteration is peeled off the loop and the function is
// scheduled again.
//
// A classical call is a `classical.submit`, which holds its link for a few
// cycles and makes a future available a budgeted latency after it ends, and a
// `classical.wait`, a zero-length op at the time the answer is taken. In a
// function that makes calls, every value carries the time it is available, so
// a pulse whose amplitude, tone or branch condition is computed from an
// answer starts after it; a function without calls skips that bookkeeping.
// Eager submissions keep their ASAP time under ALAP. A blocking wait, like a
// variable wait, is a barrier: nothing after it starts before it, nothing
// before it ends after.
//
// A barrier holds back only its timeline (see Timelines below): the lines
// that depend on it. Each timeline keeps its own floor, end and ceiling, its
// own order on each link it shares and its own lanes under rcp, and a loop
// advances the timelines its body plays on. With one timeline (a function
// on one line, or without a run-time wait) a barrier holds back every line.
//
// A `pulse.schedule_block` is scheduled first, innermost first, on its own
// with its own policy. Its shape (length, and when each operand is first
// needed and each result is available) makes it one rigid op for the
// enclosing schedule. Once that is placed, the block's ops are moved to
// their place in it and the block is inlined.

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallBitVector.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>

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

#include "cudaq-pulse/Analysis/Affine.h"
#include "cudaq-pulse/Analysis/ClassicalTiming.h"
#include "cudaq-pulse/Analysis/Controllers.h"
#include "cudaq-pulse/Analysis/Durations.h"
#include "cudaq-pulse/Analysis/LineDelay.h"
#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Analysis/Untimed.h"

#include <map>

namespace pulse {
/// Refuses each pulse.call of `function`: the schedule passes do not place
/// one yet (PulseCallCheck.cpp).
mlir::LogicalResult refuseCalls(mlir::func::FuncOp function);
} // namespace pulse

namespace {

constexpr int64_t kNever = std::numeric_limits<int64_t>::max() / 4;

static std::optional<int64_t> traceConstantI64(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
  return std::nullopt;
}

static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                   classical::SubmitOp, classical::WaitOp>(op);
}

/// The shape a scheduled `pulse.schedule_block` presents to its parent, in
/// the block's local time. An offset of -1 marks an operand nothing in the
/// block waits for, or a result that passes operand `source` through.
struct BlockShape {
  int64_t length = 0;
  llvm::SmallVector<int64_t> in, out, source;
};

static BlockShape shapeOf(pulse::ScheduleBlockOp block) {
  BlockShape shape;
  auto array = [&](llvm::StringRef name) {
    llvm::SmallVector<int64_t> values;
    if (auto attr = block->getAttrOfType<mlir::DenseI64ArrayAttr>(name))
      values.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
    return values;
  };
  if (auto length = block->getAttrOfType<mlir::IntegerAttr>("length_cycles"))
    shape.length = length.getInt();
  shape.in = array("in_offset_cycles");
  shape.out = array("out_offset_cycles");
  shape.source = array("out_source");
  return shape;
}

/// The cycles `op` takes. The forward pass checks (placedDuration) that
/// every pulse and wait it places has a known length.
static int64_t durationOf(mlir::Operation *op) {
  if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op))
    return shapeOf(block).length;
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return pulse::waveformCycles(drive.getPulse()).value_or(0);
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return pulse::waveformCycles(readout.getPulse()).value_or(0);
  if (mlir::isa<classical::SubmitOp>(op))
    return classical::intAttr(op, "classical.submit_cycles", 1);
  if (mlir::isa<classical::WaitOp>(op))
    return 0;
  // A variable wait is scheduled at its lowest length, or at its length,
  // clamped to its bounds, once an answer has been bound to it.
  if (op->hasAttr("classical.variable")) {
    int64_t lowest = classical::intAttr(op, "pulse.min_cycles");
    int64_t length = lowest;
    if (auto conversion = mlir::cast<pulse::WaitOp>(op)
                              .getDuration()
                              .getDefiningOp<pulse::DurationFromIntOp>())
      if (auto cycles = pulse::boundNumber(conversion.getCycles()))
        length = std::max(lowest, static_cast<int64_t>(*cycles));
    if (auto highest = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles"))
      length = std::min(length, highest.getInt());
    return length;
  }
  return pulse::waitCycles(mlir::cast<pulse::WaitOp>(op)).value_or(0);
}

/// The cycles `op` keeps its line idle before it starts: its
/// `pulse.hold_cycles`, which spacing for an update bus sets, else 0.
static int64_t holdOf(mlir::Operation *op) {
  return classical::intAttr(op, "pulse.hold_cycles", 0);
}

/// The cycles a drive, readout or wait takes as the forward pass places it,
/// once it is known to be valid: the schedule places an op for the cycles it
/// takes, so an unknown length is an error rather than a placement that
/// assumes some length.
static mlir::FailureOr<int64_t> placedDuration(mlir::Operation *op) {
  auto checkPulse = [&](mlir::Value waveform) -> mlir::FailureOr<int64_t> {
    std::optional<int64_t> cycles = pulse::waveformCycles(waveform);
    if (!cycles)
      return op->emitError("cannot determine the duration of this pulse; "
                           "specialize the waveform to a constant duration "
                           "first");
    if (*cycles <= 0)
      return op->emitError("a pulse must last at least one cycle, got ")
             << *cycles;
    return *cycles;
  };
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return checkPulse(drive.getPulse());
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return checkPulse(readout.getPulse());
  auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
  if (!wait || op->hasAttr("classical.variable"))
    return durationOf(op);
  std::optional<int64_t> cycles = pulse::waitCycles(wait);
  if (!cycles)
    return wait.emitError("cannot determine the length of this wait; give "
                          "it a constant length, or make it a "
                          "pulse.variable_wait with its bounds");
  if (*cycles < 0)
    return wait.emitError("a wait cannot last ") << *cycles << " cycles";
  return *cycles;
}

/// From the end of a submission until its future is available.
static int64_t budgetOf(mlir::Operation *submit) {
  return classical::intAttr(submit, "classical.budget_cycles");
}

/// How many times a submission in the body of `loop` is in flight at once
/// in steady state when the loop's iterations shift by `period`: once per
/// run of it in an iteration (the trips of the loops between them), for
/// ceil(busy / period) iterations each. None when that does not fit.
static std::optional<int64_t> inFlight(mlir::Operation *submit,
                                       mlir::scf::ForOp loop, int64_t period) {
  int64_t busy = durationOf(submit) + budgetOf(submit);
  int64_t runs = (busy + period - 1) / period;
  for (auto inner = submit->getParentOfType<mlir::scf::ForOp>();
       inner && inner != loop;
       inner = inner->getParentOfType<mlir::scf::ForOp>()) {
    auto trips = pulse::constantTripCount(inner);
    if (!trips)
      return std::nullopt;
    if (__builtin_mul_overflow(runs, *trips, &runs))
      return std::nullopt;
  }
  return runs;
}

/// Whether `loop`'s body makes a classical call.
static bool holdsCalls(mlir::scf::ForOp loop) {
  return loop.getBody()
      ->walk([](classical::SubmitOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

/// Lines, tones and measurement results: the values that become available
/// at a time. A tone is busy while a drive or readout plays on it, so pulses
/// that share a tone run in the order the tone is threaded through them.
static bool isTimedType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                   pulse::ToneType, pulse::MeasurementType>(type) ||
         pulse::isClassicalType(type);
}

static bool isLineType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType>(type);
}

/// Visit the timed ops and schedule blocks of `region`, but not the ops in
/// those blocks: they carry a schedule local to their block.
template <typename Fn>
static void forEachScheduled(mlir::Region &region, Fn &&fn) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (mlir::isa<pulse::ScheduleBlockOp>(op)) {
      fn(op);
      return mlir::WalkResult::skip();
    }
    if (isTimedOp(op))
      fn(op);
    return mlir::WalkResult::advance();
  });
}

/// How a loop-carried value takes part in the steady-state test.
enum class Carried : uint8_t {
  Untimed,
  /// Must advance by the loop's period every iteration.
  Timed,
  /// A tone that no pulse in the body plays on keeps its time (also a link,
  /// and, backward, a measurement the body reads).
  Tone,
};

static mlir::Value toneOperand(mlir::Operation *op) {
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return drive.getTone();
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return readout.getTone();
  return {};
}

static mlir::Value toneResult(mlir::Operation *op) {
  if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op))
    return drive.getUpdatedTone();
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
    return readout.getUpdatedTone();
  return {};
}

/// The measurement results a branch condition is computed from. A link is
/// not followed: the calls before a call on its link end before it starts,
/// and each starts after the measurements it sends, as
/// pulse-classical-check checks (S10, S11), so theirs are ready before
/// its own. Without `throughAnswers`, neither is an answer: a call starts
/// after what it sends (S10) and its answer is due later still, before any
/// op of a branch on it starts (S12), so a branch on an answer is bounded
/// by the answer alone.
static void conditionMeasurements(mlir::Value value,
                                  llvm::SmallVectorImpl<mlir::Value> &found,
                                  llvm::DenseSet<mlir::Value> &seen,
                                  bool throughAnswers) {
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
    if (!def || (!throughAnswers && mlir::isa<classical::WaitOp>(def)))
      continue;
    for (mlir::Value operand : llvm::reverse(def->getOperands()))
      todo.push_back(operand);
  }
}

static llvm::SmallVector<mlir::Value>
conditionMeasurements(mlir::Value cond, bool throughAnswers = true) {
  llvm::SmallVector<mlir::Value> found;
  llvm::DenseSet<mlir::Value> seen;
  conditionMeasurements(cond, found, seen, throughAnswers);
  return found;
}

struct Resources {
  bool enabled = false;
  int64_t maxDrives = 0;
  int64_t maxReadouts = 0;
  int64_t readoutLatency = 0;
  int64_t switchPenalty = 0;
  /// By qudit, the delay of its readout line (see LineDelays); empty
  /// without one.
  llvm::ArrayRef<int64_t> readoutDelays;
};

// ── Timelines ────────────────────────────────────────────────────────────

/// The lines of a function in timelines: a run-time wait (a variable wait,
/// or a blocking wait for an answer) holds back the lines of its timeline
/// only. Two lines share a timeline when one waits for the other: an op
/// joins the timelines of the lines, tones, measurements and answers it is
/// computed from, an op in a branch joins the condition's, a schedule block
/// (placed whole) or a sync joins its inputs', and a loop-carried value
/// joins what it starts from and what each iteration yields.
///
/// Other lines are held back too where their schedule could not just move
/// with the wait: under rcp/alap_rcp when there are more lines than lanes, a
/// blocking wait whose answer nothing uses (it stops everything), and a link
/// with more timelines than slots, which merges them until each has one.
struct Timelines {
  /// The number of timelines, or 0 when one barrier would hold back every
  /// line: a function without run-time waits, or with one timeline.
  int64_t count = 0;
  /// Under rcp/alap_rcp, the drive and readout lines of each timeline: its
  /// own lanes.
  llvm::SmallVector<int64_t> drives, readouts;
};

class TimelineAnalysis {
public:
  TimelineAnalysis(mlir::func::FuncOp function, const Resources &resources)
      : function(function), resources(resources) {}

  /// Find the timelines and write `classical.timeline` on the allocations,
  /// calls and schedule blocks of the function, or remove it if there is
  /// one. Returns whether the calls' attributes changed.
  bool run(Timelines &result) {
    // Every call of a controller that blocks on calls holds every line.
    everyLine =
        pulse::blocksOnCalls(function->getParentOfType<mlir::ModuleOp>());
    // Without a run-time wait there is one timeline: skip the analysis.
    bool numbered = false;
    auto scan = function.walk([&](mlir::Operation *op) {
      if (op->hasAttr("classical.variable") ||
          (mlir::isa<classical::WaitOp>(op) &&
           op->hasAttr("classical.blocking")))
        return mlir::WalkResult::interrupt();
      numbered |= op->hasAttr(classical::kTimelineAttr);
      return mlir::WalkResult::advance();
    });
    if (!scan.wasInterrupted()) {
      if (numbered)
        function.walk([](mlir::Operation *op) {
          op->removeAttr(classical::kTimelineAttr);
        });
      return numbered;
    }
    visitBlock(function.getBody().front(), kNone);
    result = decide();
    bool changed = stale;
    if (result.count == 0) {
      if (stale)
        for (auto [op, c] : nodes)
          op->removeAttr(classical::kTimelineAttr);
      return changed;
    }
    mlir::Builder b(function.getContext());
    for (auto [op, c] : nodes)
      op->setAttr(classical::kTimelineAttr, b.getI64IntegerAttr(idOf[find(c)]));
    return true;
  }

private:
  static constexpr unsigned kNone = ~0u;

  unsigned make() {
    parent.push_back(static_cast<unsigned>(parent.size()));
    return parent.back();
  }
  unsigned find(unsigned x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  /// The joined class; the earlier one names it, so numbering is stable.
  unsigned join(unsigned a, unsigned b) {
    if (a == kNone)
      return b;
    if (b == kNone)
      return a;
    a = find(a);
    b = find(b);
    if (a == b)
      return a;
    if (b < a)
      std::swap(a, b);
    parent[b] = a;
    return a;
  }
  unsigned of(mlir::Value value) const {
    auto it = classOf.find(value);
    return it == classOf.end() ? kNone : it->second;
  }
  static bool isLink(mlir::Value value) {
    return mlir::isa<classical::LinkType>(value.getType());
  }
  unsigned operandClass(mlir::Operation *op, unsigned context) {
    unsigned c = context;
    for (mlir::Value operand : op->getOperands())
      if (!isLink(operand))
        c = join(c, of(operand));
    return c;
  }
  void give(mlir::ValueRange values, unsigned c) {
    if (c == kNone)
      return;
    for (mlir::Value value : values)
      if (!isLink(value))
        classOf[value] = c;
  }
  void note(mlir::Operation *op, unsigned c) {
    stale |= op->hasAttr(classical::kTimelineAttr);
    nodes.push_back({op, c});
  }
  /// A call with no inputs that carry time goes with what uses its answer.
  unsigned orphan(unsigned c) { return c != kNone ? c : make(); }

  void visitBlock(mlir::Block &block, unsigned context) {
    for (mlir::Operation &op : block)
      if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
        visitOp(&op, context);
  }

  void visitOp(mlir::Operation *op, unsigned context) {
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op)) {
      unsigned c = join(make(), context);
      lines.push_back({op, c});
      note(op, c);
      give(op->getResults(), c);
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      auto yield = loop.getBody()->getTerminator();
      auto trips = pulse::constantTripCount(loop);
      bool runs = trips && *trips > 0;
      for (auto [init, arg, result] :
           llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                     loop.getResults())) {
        if (isLink(arg))
          continue;
        // A value the body does not read, in a loop that runs, reaches
        // nothing from before the loop: each iteration yields its own.
        bool reaches = !runs || !arg.use_empty();
        unsigned c = join(make(), reaches ? join(context, of(init)) : context);
        classOf[arg] = c;
        classOf[result] = c;
      }
      visitBlock(*loop.getBody(), context);
      for (auto [arg, yielded] :
           llvm::zip(loop.getRegionIterArgs(), yield->getOperands()))
        if (!isLink(arg))
          join(of(arg), of(yielded));
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      unsigned c = join(context, of(branch.getCondition()));
      visitBlock(*branch.thenBlock(), c);
      if (!branch.getElseRegion().empty())
        visitBlock(*branch.elseBlock(), c);
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        unsigned r = join(c, of(branch.thenYield().getOperand(index)));
        if (!branch.getElseRegion().empty())
          r = join(r, of(branch.elseYield().getOperand(index)));
        give(result, r);
      }
      return;
    }
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      unsigned c = operandClass(op, context);
      if (c == kNone)
        c = make();
      give(block.getBody().getArguments(), c);
      visitBlock(block.getBody().front(), c);
      for (mlir::Value yielded :
           block.getBody().front().getTerminator()->getOperands())
        if (!isLink(yielded))
          c = join(c, of(yielded));
      give(op->getResults(), c);
      note(op, c);
      return;
    }
    if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
      unsigned c = orphan(operandClass(op, context));
      give(submit.getFuture(), c);
      note(op, c);
      return;
    }
    if (auto wait = mlir::dyn_cast<classical::WaitOp>(op)) {
      unsigned c = orphan(operandClass(op, context));
      give(wait.getValues(), c);
      note(op, c);
      if (op->hasAttr("classical.blocking")) {
        runtimeWait = true;
        everyLine |= llvm::all_of(wait.getValues(),
                                  [](mlir::Value v) { return v.use_empty(); });
      }
      return;
    }
    runtimeWait |= op->hasAttr("classical.variable");
    unsigned c = operandClass(op, context);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &nested : region)
        visitBlock(nested, c);
    give(op->getResults(), c);
  }

  /// Number the classes that hold lines in allocation order; a call or
  /// block whose class holds none goes with the first line.
  void number() {
    llvm::DenseSet<unsigned> withLines;
    for (auto [op, c] : lines)
      withLines.insert(find(c));
    for (auto [op, c] : nodes)
      if (!withLines.count(find(c)))
        join(c, lines.front().second);
    idOf.clear();
    for (auto [op, c] : lines)
      idOf.try_emplace(find(c), static_cast<int64_t>(idOf.size()));
  }

  Timelines decide() {
    Timelines result;
    if (!runtimeWait || everyLine || lines.size() < 2)
      return result;
    if (resources.enabled) {
      int64_t drives = 0, readouts = 0;
      for (auto [op, c] : lines)
        ++(mlir::isa<pulse::GetDriveLineOp>(op) ? drives : readouts);
      if (drives > resources.maxDrives || readouts > resources.maxReadouts)
        return result;
    }
    auto module = function->getParentOfType<mlir::ModuleOp>();
    // Merge the timelines of a link round-robin until it has a slot for each.
    while (true) {
      number();
      std::map<std::pair<int64_t, int64_t>, llvm::SmallVector<unsigned>>
          perLink;
      for (auto [op, c] : nodes) {
        if (!mlir::isa<classical::SubmitOp>(op))
          continue;
        auto &list =
            perLink[{classical::intAttr(op, "classical.device"),
                     classical::intAttr(op, classical::kInstanceAttr, -1)}];
        if (!llvm::is_contained(list, find(c)))
          list.push_back(find(c));
      }
      bool merged = false;
      for (auto &[key, list] : perLink) {
        auto slots = static_cast<size_t>(std::max<int64_t>(
            1, classical::linkModel(module, key.first).slots));
        for (size_t k = slots; k < list.size(); ++k) {
          join(list[k], list[k % slots]);
          merged = true;
        }
      }
      if (!merged)
        break;
    }
    number();
    if (idOf.size() < 2)
      return result;
    result.count = static_cast<int64_t>(idOf.size());
    if (resources.enabled) {
      result.drives.assign(idOf.size(), 0);
      result.readouts.assign(idOf.size(), 0);
      for (auto [op, c] : lines)
        ++(mlir::isa<pulse::GetDriveLineOp>(op)
               ? result.drives
               : result.readouts)[idOf[find(c)]];
    }
    return result;
  }

  mlir::func::FuncOp function;
  Resources resources;
  llvm::SmallVector<unsigned> parent;
  llvm::DenseMap<mlir::Value, unsigned> classOf;
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>> lines, nodes;
  llvm::DenseMap<unsigned, int64_t> idOf;
  bool runtimeWait = false, everyLine = false, stale = false;
};

struct Sample {
  llvm::SmallVector<int64_t, 2> iteration;
  int64_t start;
};

/// A loop run whose iterations from `from` on were not simulated because
/// each one shifts the previous by `period`; or, with `periods`, shifts
/// each timed op by the period of its own (see LoopGroups).
struct Extrapolation {
  mlir::scf::ForOp loop;
  int64_t period;
  std::shared_ptr<const llvm::DenseMap<mlir::Operation *, int64_t>> periods;

  std::optional<int64_t> periodOf(mlir::Operation *op) const {
    if (!periods)
      return period;
    auto it = periods->find(op);
    if (it == periods->end())
      return std::nullopt;
    return it->second;
  }
};

/// Iterations `lo` < i < `hi` of one run of `loop` (in the iterations
/// `outer` of the loops around it) that the scheduler did not simulate
/// although the pass compares a loop's lanes over its first iterations only
/// and would otherwise have simulated them (see Scheduler::LaneCycle): each
/// starts every op, nested ones too, `period` per iteration after it does
/// in iteration `anchor` (or, with `periods`, its own period). The schedule
/// is written as if they had been simulated.
struct Cover {
  mlir::scf::ForOp loop;
  llvm::SmallVector<int64_t, 2> outer;
  int64_t anchor, lo, hi, period;
  /// With `periods`, each op's period of its own, as in Extrapolation.
  std::shared_ptr<const llvm::DenseMap<mlir::Operation *, int64_t>> periods;

  int64_t periodOf(mlir::Operation *op) const {
    return periods ? periods->lookup(op) : period;
  }
};

/// The carried values of a loop in groups that no op of its body relates:
/// each op's timing depends on the values of one group only, so each group
/// may move by a period of its own. `ofArg` has each carried value's group
/// and `ofOp` each timed op's in the body, nested ones included; `outside`
/// the measurements from before the loop that its branches test, which
/// hold back the ops in them until they are ready.
///
/// A branch relates the group of its condition and those of what it
/// decides one way only: forward, the condition holds back the arms and
/// what follows them; backward, they hold back the condition. Each of
/// `branches` is such a pair, the condition's group first. A group so held
/// back may move faster than the one holding it back: the constraint
/// between them, once slack, only grows slacker. (Groups that hold each
/// other back can then only move together.)
///
/// Backward, a call's wait moves with what its answer decides rather than
/// with its request, and so does a lazy request, due by its wait, with the
/// link it threads: `ofWaitBackward` has the group of each such wait and
/// request, `ofArgBackward` each carried value's group backward, and
/// `mixedWaits` whether some wait's answer decides several groups or some
/// link is due by several.
struct LoopGroups {
  unsigned count = 0;
  /// Where a link is due by several groups (so `mixedWaits`), the groups
  /// backward with each lazy request whose link goes on to an eager one
  /// moving with that one, which keeps its forward time and bounds it in
  /// link order: steady while that bound binds, so the group of its waits
  /// is held to move no faster than the eager request's (a branch, the
  /// waits' group deciding). None when that leaves a link due by several.
  std::shared_ptr<const LoopGroups> linkBound;
  llvm::SmallVector<unsigned> ofArg, ofArgBackward;
  llvm::DenseMap<mlir::Operation *, unsigned> ofOp;
  llvm::DenseMap<mlir::Operation *, unsigned> ofWaitBackward;
  bool mixedWaits = false;
  /// The groups of the drives, and of the readouts, in the body, which take
  /// lanes under resources.
  llvm::SmallBitVector driveGroups, readoutGroups;
  llvm::SmallVector<mlir::Value> outside;
  llvm::SmallVector<std::pair<unsigned, unsigned>> branches;
  /// Of `linkBound`'s branches, each lazy request and the eager one after
  /// it on the link: forward right after it, the eager one bounds it
  /// backward too (its answer's due time is never before its forward
  /// start), so without resources the pair holds however its groups move.
  struct LinkPair {
    std::pair<unsigned, unsigned> branch;
    mlir::Operation *lazy, *eager;
  };
  llvm::SmallVector<LinkPair> linkPairs;
  /// Of `branches`, those some op relates other than through answers (a
  /// branch on a measurement), and the lazy requests whose answers relate
  /// each other one.
  llvm::SmallVector<std::pair<unsigned, unsigned>> strictBranches;
  llvm::SmallVector<std::pair<std::pair<unsigned, unsigned>,
                              llvm::SmallVector<mlir::Operation *, 2>>>
      lazyBranches;

  /// The group of a timed op in the body, the way the pass runs.
  unsigned groupOf(mlir::Operation *op, bool backward) const {
    if (backward)
      if (auto it = ofWaitBackward.find(op); it != ofWaitBackward.end())
        return it->second;
    return ofOp.lookup(op);
  }
  /// The group of each carried value, the way the pass runs.
  llvm::ArrayRef<unsigned> argGroups(bool backward) const {
    return backward ? ofArgBackward : ofArg;
  }
};

/// The waits of calls whose answers `value` is computed from.
static void conditionWaits(mlir::Value value,
                           llvm::SmallVectorImpl<mlir::Operation *> &found,
                           llvm::DenseSet<mlir::Value> &seen) {
  // As conditionMeasurements follows: the next value last.
  llvm::SmallVector<mlir::Value, 16> todo{value};
  while (!todo.empty()) {
    mlir::Value next = todo.pop_back_val();
    if (!seen.insert(next).second)
      continue;
    mlir::Operation *def = next.getDefiningOp();
    if (!def)
      continue;
    if (mlir::isa<classical::WaitOp>(def)) {
      found.push_back(def);
      continue;
    }
    for (mlir::Value operand : llvm::reverse(def->getOperands()))
      todo.push_back(operand);
  }
}

/// Whether `value` is computed from a measurement other than through the
/// answer of a call.
static bool readsMeasurement(mlir::Value value,
                             llvm::DenseSet<mlir::Value> &seen) {
  // As conditionMeasurements follows, up to the first measurement.
  llvm::SmallVector<mlir::Value, 16> todo{value};
  while (!todo.empty()) {
    mlir::Value next = todo.pop_back_val();
    if (!seen.insert(next).second)
      continue;
    if (mlir::isa<pulse::MeasurementType>(next.getType()))
      return true;
    mlir::Operation *def = next.getDefiningOp();
    if (!def || mlir::isa<classical::WaitOp>(def))
      continue;
    for (mlir::Value operand : llvm::reverse(def->getOperands()))
      todo.push_back(operand);
  }
  return false;
}

/// The groups of `loop`'s carried values, or none when their independence
/// cannot be shown: a carried call value other than a link, or a carried
/// measurement the body reads; a call op that keeps state the values it
/// reads do not carry; a timed value other than a measurement from outside
/// the loop read in it (other than through what it carries); a schedule
/// block or another region op; a timed op that reads no carried value; or a
/// line value that does not reach the loop's yield through timed ops and
/// syncs. In a body with a loop of its own, a branch joins the groups it
/// relates.
static std::shared_ptr<const LoopGroups> loopGroups(mlir::scf::ForOp loop) {
  unsigned n = loop.getNumRegionIterArgs();
  for (mlir::Value arg : loop.getRegionIterArgs()) {
    mlir::Type type = arg.getType();
    if (mlir::isa<pulse::MeasurementType>(type)
            ? !arg.use_empty()
            : isTimedType(type) && !isLineType(type) &&
                  !mlir::isa<pulse::ToneType, classical::LinkType>(type))
      return nullptr;
  }
  // A call's state off its values: a barrier, a wait of a run-time length,
  // or a link order of an instance's own. And a wait in a loop of the body
  // whose answers nothing reads: only its own time marks where it ends.
  if (loop.getBody()
          ->walk([&](mlir::Operation *op) {
            bool unread = mlir::isa<classical::WaitOp>(op) &&
                          op->getParentOfType<mlir::scf::ForOp>() != loop &&
                          op->use_empty();
            return unread || op->hasAttr("classical.blocking") ||
                           op->hasAttr("classical.variable") ||
                           op->hasAttr(classical::kInstanceAttr)
                       ? mlir::WalkResult::interrupt()
                       : mlir::WalkResult::advance();
          })
          .wasInterrupted())
    return nullptr;
  llvm::SmallVector<mlir::Value> outside;
  llvm::DenseMap<mlir::Value, llvm::SmallBitVector> roots;
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    llvm::SmallBitVector bits(n);
    bits.set(static_cast<unsigned>(i));
    roots[arg] = std::move(bits);
  }
  auto rootsOf = [&](mlir::Value value) {
    auto found = roots.find(value);
    return found == roots.end() ? llvm::SmallBitVector(n) : found->second;
  };
  auto inside = [&](mlir::Value value) {
    return loop->isAncestor(value.getParentRegion()->getParentOp());
  };
  auto reachesYield = [](mlir::Value line) {
    if (line.use_empty())
      return false;
    return llvm::all_of(line.getUsers(), [](mlir::Operation *user) {
      return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                       pulse::SyncOp, mlir::scf::YieldOp, mlir::scf::ForOp>(
          user);
    });
  };
  // The ops that relate the values they read: timed ops and syncs.
  llvm::SmallVector<std::pair<mlir::Operation *, llvm::SmallBitVector>> related;
  // What each branch's condition and what it decides read, whether the
  // one holds back the other only through answers, and the lazy requests
  // among those.
  struct Decides {
    llvm::SmallBitVector condition, decided;
    bool answers;
    llvm::SmallVector<mlir::Operation *, 2> lazy;
  };
  llvm::SmallVector<Decides> decides;
  auto eagerAnswer = [](mlir::Operation *wait) {
    auto submit =
        mlir::cast<classical::WaitOp>(wait).getFuture().getDefiningOp();
    return submit && submit->hasAttr("classical.eager");
  };
  auto lazyOf = [&](llvm::ArrayRef<mlir::Operation *> waits) {
    llvm::SmallVector<mlir::Operation *, 2> lazy;
    for (mlir::Operation *wait : waits)
      if (!eagerAnswer(wait))
        if (mlir::Operation *submit =
                mlir::cast<classical::WaitOp>(wait).getFuture().getDefiningOp())
          lazy.push_back(submit);
    return lazy;
  };
  // What the answers of each wait decide.
  llvm::DenseMap<mlir::Operation *, llvm::SmallBitVector> waitDecides;
  bool directed = true;
  loop.getBody()->walk([&](mlir::scf::ForOp) { directed = false; });
  // In a body without loops of its own an answer does not join what it is
  // a value of with its call: it holds it back, as a branch's condition
  // does what it decides. The waits each value is computed from, and what
  // each wait's call reads.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *, 1>> answers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallBitVector> callOf;
  // What `value` reads, its answers' calls included.
  auto readsOf = [&](mlir::Value value) {
    llvm::SmallBitVector bits = rootsOf(value);
    bits.resize(n);
    if (auto it = answers.find(value); it != answers.end())
      for (mlir::Operation *wait : it->second)
        bits |= callOf.lookup(wait);
    return bits;
  };
  bool ok = true;
  std::function<void(mlir::Block &, const llvm::SmallBitVector &)> visit =
      [&](mlir::Block &block, const llvm::SmallBitVector &ambient) {
        for (mlir::Operation &op : block.without_terminator()) {
          for (mlir::Value operand : op.getOperands())
            if (isTimedType(operand.getType()) && !inside(operand)) {
              if (mlir::isa<pulse::MeasurementType>(operand.getType()))
                outside.push_back(operand);
              else
                ok = false;
            }
          for (mlir::Value result : op.getResults())
            if (isLineType(result.getType()) && !reachesYield(result))
              ok = false;
          if (!ok)
            return;
          if (auto inner = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
            llvm::SmallVector<llvm::SmallBitVector> state;
            for (mlir::Value init : inner.getInitArgs())
              state.push_back(readsOf(init) | ambient);
            auto yield = inner.getBody()->getTerminator();
            size_t mark = related.size();
            // Each round adds roots, of which there are `n`.
            for (bool changed = true; changed;) {
              related.resize(mark, {nullptr, llvm::SmallBitVector()});
              for (auto [arg, bits] :
                   llvm::zip(inner.getRegionIterArgs(), state))
                roots[arg] = bits;
              visit(*inner.getBody(), ambient);
              if (!ok)
                return;
              changed = false;
              for (auto [bits, value] :
                   llvm::zip(state, yield->getOperands())) {
                llvm::SmallBitVector next = bits | rootsOf(value);
                if (next != bits) {
                  bits = std::move(next);
                  changed = true;
                }
              }
            }
            for (auto [result, bits] : llvm::zip(inner.getResults(), state))
              roots[result] = bits;
            continue;
          }
          if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            for (mlir::Value measurement :
                 conditionMeasurements(branch.getCondition()))
              if (!inside(measurement))
                outside.push_back(measurement);
            llvm::SmallBitVector condition =
                readsOf(branch.getCondition()) | ambient;
            size_t mark = related.size();
            visit(*branch.thenBlock(), condition);
            if (!branch.getElseRegion().empty())
              visit(*branch.elseBlock(), condition);
            if (!ok)
              return;
            llvm::SmallBitVector decided(n);
            for (auto [index, result] : llvm::enumerate(branch.getResults())) {
              llvm::SmallBitVector bits =
                  readsOf(branch.thenYield().getOperand(index));
              if (!branch.getElseRegion().empty())
                bits |= readsOf(branch.elseYield().getOperand(index));
              if (!directed)
                bits |= condition;
              decided |= bits;
              roots[result] = std::move(bits);
            }
            if (directed) {
              for (size_t i = mark; i < related.size(); ++i)
                decided |= related[i].second;
              llvm::SmallVector<mlir::Operation *> waits;
              llvm::DenseSet<mlir::Value> seen;
              conditionWaits(branch.getCondition(), waits, seen);
              for (mlir::Operation *wait : waits) {
                auto [it, inserted] = waitDecides.try_emplace(wait, n);
                it->second |= decided;
              }
              llvm::DenseSet<mlir::Value> read;
              bool answers = ambient.none() && !waits.empty() &&
                             !readsMeasurement(branch.getCondition(), read);
              decides.push_back(
                  {condition, std::move(decided), answers, lazyOf(waits)});
            }
            continue;
          }
          if (op.getNumRegions()) {
            ok = false;
            return;
          }
          llvm::SmallBitVector bits =
              directed ? llvm::SmallBitVector(n) : ambient;
          llvm::SmallVector<mlir::Operation *, 1> from;
          for (mlir::Value operand : op.getOperands()) {
            bits |= directed ? rootsOf(operand) : readsOf(operand);
            if (auto it = answers.find(operand);
                directed && it != answers.end())
              for (mlir::Operation *wait : it->second)
                if (!llvm::is_contained(from, wait))
                  from.push_back(wait);
          }
          if (directed && mlir::isa<classical::WaitOp>(op)) {
            callOf[&op] = bits;
            for (mlir::Value result : op.getResults())
              answers[result] = {&op};
          } else {
            // A line or tone an answer holds back moves with what it is.
            for (mlir::Value result : op.getResults()) {
              roots[result] = bits;
              if (!from.empty() && !isTimedType(result.getType()))
                answers[result] = from;
            }
          }
          if (bits.any())
            for (mlir::Operation *wait : from) {
              decides.push_back(
                  {callOf.lookup(wait), bits, true, lazyOf(wait)});
              auto [it, inserted] = waitDecides.try_emplace(wait, n);
              it->second |= bits;
            }
          if (isTimedOp(&op) || mlir::isa<pulse::SyncOp>(op))
            related.emplace_back(&op, std::move(bits));
        }
      };
  visit(*loop.getBody(), llvm::SmallBitVector(n));
  if (!ok)
    return nullptr;

  llvm::SmallVector<unsigned> parent(n);
  std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&](unsigned i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  };
  auto join = [&](const llvm::SmallBitVector &bits) {
    int first = bits.find_first();
    for (unsigned bit : bits.set_bits())
      parent[find(bit)] = find(static_cast<unsigned>(first));
  };
  for (auto &[op, bits] : related) {
    if (bits.none())
      return nullptr;
    join(bits);
  }
  auto yield = loop.getBody()->getTerminator();
  for (auto [i, value] : llvm::enumerate(yield->getOperands())) {
    llvm::SmallBitVector bits = readsOf(value);
    bits.set(static_cast<unsigned>(i));
    join(bits);
  }
  auto groups = std::make_shared<LoopGroups>();
  llvm::DenseMap<unsigned, unsigned> index;
  for (unsigned i = 0; i < n; ++i) {
    auto [it, inserted] = index.try_emplace(find(i), groups->count);
    if (inserted)
      ++groups->count;
    groups->ofArg.push_back(it->second);
  }
  for (auto &[op, bits] : related)
    if (isTimedOp(op))
      groups->ofOp[op] =
          index.lookup(find(static_cast<unsigned>(bits.find_first())));
  // A branch holds each group its condition reads to each group it decides.
  // The lines of a condition, and of what is decided, lie in few groups
  // (in one, where the body's syncs join every line), so the groups of each
  // are gathered first, in the order of their first lines, and paired:
  // the pairs, and their order, that pairing every line with every line
  // gives. With one group there is no pair.
  llvm::SmallVector<unsigned> seenBy(groups->count, 0);
  unsigned stamp = 0;
  auto groupsIn = [&](const llvm::SmallBitVector &bits) {
    llvm::SmallVector<unsigned, 2> found;
    ++stamp;
    for (unsigned bit : bits.set_bits()) {
      unsigned group = groups->ofArg[bit];
      if (seenBy[group] != stamp) {
        seenBy[group] = stamp;
        found.push_back(group);
      }
    }
    return found;
  };
  if (groups->count > 1)
    for (auto &[condition, decided, answers, lazy] : decides) {
      llvm::SmallVector<unsigned, 2> deciding = groupsIn(condition);
      llvm::SmallVector<unsigned, 2> held = groupsIn(decided);
      for (unsigned c : deciding)
        for (unsigned d : held) {
          std::pair<unsigned, unsigned> branch{c, d};
          if (branch.first == branch.second)
            continue;
          if (!llvm::is_contained(groups->branches, branch))
            groups->branches.push_back(branch);
          if (!answers && !llvm::is_contained(groups->strictBranches, branch))
            groups->strictBranches.push_back(branch);
          auto it = llvm::find_if(groups->lazyBranches, [&](const auto &entry) {
            return entry.first == branch;
          });
          if (it == groups->lazyBranches.end())
            it = groups->lazyBranches.insert(it, {branch, {}});
          for (mlir::Operation *submit : lazy)
            if (!llvm::is_contained(it->second, submit))
              it->second.push_back(submit);
        }
    }
  groups->driveGroups.resize(groups->count);
  groups->readoutGroups.resize(groups->count);
  for (auto [op, group] : groups->ofOp) {
    if (mlir::isa<pulse::DriveOp>(op))
      groups->driveGroups.set(group);
    else if (mlir::isa<pulse::ReadoutOp>(op))
      groups->readoutGroups.set(group);
  }
  for (auto &[wait, decided] : waitDecides) {
    std::optional<unsigned> only;
    for (unsigned d : decided.set_bits()) {
      unsigned group = index.lookup(find(d));
      if (only && *only != group)
        groups->mixedWaits = true;
      only = group;
    }
    if (only)
      groups->ofWaitBackward[wait] = *only;
  }
  bool waitsMixed = groups->mixedWaits;
  auto variant = std::make_shared<LoopGroups>(*groups);
  // Which groups each carried link and lazy request are due by backward;
  // `bound`, each lazy request going on to an eager one with that one.
  auto assignDue = [&](LoopGroups &into, bool bound) {
    into.ofArgBackward = into.ofArg;
    llvm::SmallVector<std::optional<unsigned>> dueBy(n);
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      mlir::Operation *op = submit;
      std::optional<unsigned> group;
      if (!op->hasAttr("classical.eager"))
        for (mlir::Operation *user : submit.getFuture().getUsers())
          if (auto it = into.ofWaitBackward.find(user);
              it != into.ofWaitBackward.end()) {
            if (group && *group != it->second)
              into.mixedWaits = true;
            group = it->second;
          }
      if (bound && group && submit.getUpdatedLink().hasOneUse())
        if (auto next = mlir::dyn_cast<classical::SubmitOp>(
                *submit.getUpdatedLink().getUsers().begin());
            next && next->getBlock() == op->getBlock() &&
            next->hasAttr("classical.eager")) {
          unsigned eager = into.ofOp.lookup(next);
          std::pair<unsigned, unsigned> held{eager, *group};
          if (eager != *group) {
            if (!llvm::is_contained(into.branches, held))
              into.branches.push_back(held);
            into.linkPairs.push_back({held, op, next});
            auto it = llvm::find_if(into.lazyBranches, [&](const auto &entry) {
              return entry.first == held;
            });
            if (it == into.lazyBranches.end())
              it = into.lazyBranches.insert(it, {held, {}});
            if (!llvm::is_contained(it->second, op))
              it->second.push_back(op);
          }
          group = eager;
        }
      if (group)
        into.ofWaitBackward[op] = *group;
      unsigned own = group.value_or(into.ofOp.lookup(op));
      llvm::SmallBitVector roots = rootsOf(submit.getLink());
      for (unsigned i : roots.set_bits()) {
        if (!mlir::isa<classical::LinkType>(
                loop.getRegionIterArgs()[i].getType()))
          continue;
        if (dueBy[i] && *dueBy[i] != own)
          into.mixedWaits = true;
        dueBy[i] = own;
      }
    });
    for (auto [i, group] : llvm::enumerate(dueBy))
      if (group)
        into.ofArgBackward[i] = *group;
  };
  assignDue(*groups, /*bound=*/false);
  if (groups->mixedWaits && !waitsMixed) {
    assignDue(*variant, /*bound=*/true);
    if (!variant->mixedWaits) {
      variant->outside = outside;
      groups->linkBound = std::move(variant);
    }
  }
  groups->outside = std::move(outside);
  return groups;
}

/// Where a Scheduler without grids would trace lines: nowhere.
struct NoLineTrace {};

/// The most iterations after which a loop's iterations may repeat for the
/// scheduler to run that many in each (see Scheduler::regroup).
constexpr int64_t kMostRegrouped = 8;

/// The most iterations of a loop as written regrouping runs in each, over
/// every time it regroups the loop.
constexpr int64_t kMostRegroupedInAll = 16;

/// The carried times after each of a loop's last iterations simulated, up
/// to the most two runs of kMostRegrouped iterations compare.
class RecentStates {
public:
  void push(llvm::ArrayRef<int64_t> state) {
    if (states.size() < kSize)
      states.emplace_back(state.begin(), state.end());
    else
      states[next].assign(state.begin(), state.end());
    next = (next + 1) % kSize;
    ++pushed;
  }

  /// The state pushed `back` pushes before the last one.
  llvm::ArrayRef<int64_t> before(int64_t back) const {
    return states[(next + kSize - 1 - static_cast<size_t>(back)) % kSize];
  }

  int64_t pushed = 0;

private:
  static constexpr size_t kSize = 2 * kMostRegrouped + 1;
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 0> states;
  size_t next = 0;
};

/// Schedules one region: a function body, or a schedule block's body with
/// every block argument available at time 0. `Gridded` when some line is on
/// a grid coarser than one program cycle (see LineGrid) or, under rcp, lines
/// are on several controllers (see Controllers): both trace lines to their
/// allocations. Without, a schedule does no grid or controller work at all.
template <bool Gridded>
class Scheduler {
public:
  Scheduler(mlir::Region &region, Resources resources, bool classical,
            const pulse::LineGrid &grid, const pulse::Controllers &controllers,
            const Timelines *timelines = nullptr)
      : region(&region), resources(resources), classical(classical),
        lanesPer(timelines && timelines->count > 1 ? timelines : nullptr),
        grid(grid), controllers(controllers) {
    auto module = region.getParentOp()->getParentOfType<mlir::ModuleOp>();
    if (module && pulse::blocksOnCalls(module))
      holdControllers(pulse::controlSystemOf(module), controllers);
    auto count = static_cast<size_t>(isolated() ? lanesPer->count : 1);
    floors.assign(count, 0);
    ends.assign(count, 0);
    iterationEnds.assign(count, -1);
    ceilings.assign(count, kNever);
  }

  mlir::LogicalResult forward() {
    resetLanes(0, /*backward=*/false);
    if (mlir::failed(forwardBlock(region->front())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult backward() {
    if (classical) {
      // Eager submissions and unused answers keep their forward times.
      for (auto &[op, list] : samples)
        if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op) ||
            op->hasAttr("classical.variable"))
          for (const Sample &sample : list)
            forwardStarts[op][sample.iteration] = sample.start;
      forwardExtrapolations = extrapolations;
    }
    // The backward pass ends no operation after the forward schedule's last
    // operation (lastEnd): the makespan may end later, at a readout line's
    // hold (for the next operation on the line, of which there is none) or
    // at a sync's instant (it plays nothing). A region that is a schedule
    // block's body or holds a block keeps the makespan: a block's shape is
    // as it was.
    lastEnd = kNever;
    if (tailNoted && !uncapped && tailMayPassOps())
      noteLastEnd();
    samples.clear();
    extrapolations.clear();
    covers.clear();
    resetLanes(makespan, /*backward=*/true);
    return backwardBlock(region->front());
  }

  /// Whether a hold or a sync may end after the forward schedule's last
  /// operation. Where every iteration was simulated (no loop extrapolated,
  /// none skipped) and none ends after the operations' ends, the makespan is
  /// the last end, and the backward pass is the same without it.
  bool tailMayPassOps() const {
    return !extrapolations.empty() || !covers.empty() || tailsEnd > opsEnd;
  }

  /// Set the last end (see backward) from the forward schedule, outside a
  /// region that is a schedule block's body or holds one.
  LLVM_ATTRIBUTE_NOINLINE void noteLastEnd() {
    if (blocksAround())
      return;
    for (auto &[op, list] : samples)
      for (const Sample &sample : list)
        forwardStarts[op][sample.iteration] = sample.start;
    forwardExtrapolations = extrapolations;
    lastEnd = writtenLastEnd() - testShrinkLastEnd();
  }

  /// Test only: CUDAQ_PULSE_TEST_SHRINK_LAST_END moves the last end that
  /// many cycles early, to show a backward schedule that breaks it
  /// scheduled again without it. Anything but a non-negative integer is
  /// ignored.
  static int64_t testShrinkLastEnd() {
    int64_t cycles = 0;
    if (const char *shrink = std::getenv("CUDAQ_PULSE_TEST_SHRINK_LAST_END"))
      if (llvm::StringRef(shrink).getAsInteger(10, cycles) || cycles < 0)
        cycles = 0;
    return cycles;
  }

  /// The end of the last operation of the forward schedule as it is
  /// written: of each timed op, its last iteration (forwardStartAt). The
  /// makespan where a loop has no constant trip count.
  int64_t writtenLastEnd() const {
    int64_t latest = 0;
    for (auto &[op, list] : samples) {
      if (!isTimedOp(op) || list.empty())
        continue;
      auto loops = pulse::enclosingLoops(op, region->getParentOp());
      llvm::SmallVector<int64_t, 2> last;
      for (mlir::scf::ForOp loop : loops) {
        std::optional<int64_t> trips = pulse::constantTripCount(loop);
        if (!trips)
          return makespan;
        last.push_back(*trips - 1);
      }
      int64_t end = forwardStartAt(op, last, loops) + durationOf(op);
      for (const Sample &sample : list)
        end = std::max(end, sample.start + durationOf(op));
      latest = std::max(latest, end);
    }
    return std::min(latest, makespan);
  }

  /// Whether the region is a schedule block's body or holds a block.
  bool blocksAround() const {
    return mlir::isa_and_nonnull<pulse::ScheduleBlockOp>(
               region->getParentOp()) ||
           region
               ->walk([](pulse::ScheduleBlockOp) {
                 return mlir::WalkResult::interrupt();
               })
               .wasInterrupted();
  }

  int64_t makespan = 0;
  /// Where the backward pass ends its last operation (see backward).
  int64_t lastEnd = kNever;
  /// Whether the forward pass noted a readout line's hold or a sync, either
  /// of which may end after the last operation.
  bool tailNoted = false;
  /// The latest end the forward pass gave an operation, and the latest
  /// hold or sync it noted (see tailMayPassOps).
  int64_t opsEnd = 0, tailsEnd = 0;
  /// Schedule without the last end, as before it (see cappedHolds).
  bool uncapped = false;

  /// Whether the round under the last end can be written (as
  /// scheduleRounds checks every round) and keeps to it (cappedHolds).
  LLVM_ATTRIBUTE_NOINLINE bool keepsLastEnd() {
    bool written = !skippedNegative && !answeredEarly &&
                   !(slotsOverrun && heldTooMany()) &&
                   llvm::all_of(samples, [](const auto &entry) {
                     return llvm::all_of(entry.second, [](const Sample &s) {
                       return s.start >= 0;
                     });
                   });
    return written && cappedHolds();
  }

  /// Whether the backward schedule keeps to the last end: every op, pulse,
  /// wait or call, ends by it and, with lanes, starts no earlier than its
  /// forward start (lanes the capped ops took could push one there, into a
  /// call it waits out on a controller that blocks on calls). Where it does
  /// not, or the round cannot be written, scheduleRounds schedules the round
  /// again without the last end, as before it.
  bool cappedHolds() const {
    if (lastEnd == kNever)
      return true;
    for (auto &[op, list] : samples) {
      if (!isTimedOp(op))
        continue;
      auto loops = pulse::enclosingLoops(op, region->getParentOp());
      for (const Sample &sample : list) {
        if (sample.start + durationOf(op) > lastEnd)
          return false;
        if (resources.enabled &&
            sample.start < forwardStartAt(op, sample.iteration, loops))
          return false;
      }
    }
    return true;
  }

  /// `end`, the latest end of `op` in the current iteration, capped at the
  /// last end. An op whose forward end is after the last end (only under
  /// the test hook) keeps `end`, and cappedHolds then fails the round.
  int64_t capAtLastEnd(mlir::Operation *op, int64_t end) {
    if (end <= lastEnd)
      return end;
    return capPastLastEnd(op, end);
  }

  LLVM_ATTRIBUTE_NOINLINE int64_t capPastLastEnd(mlir::Operation *op,
                                                 int64_t end) {
    auto loops = pulse::enclosingLoops(op, region->getParentOp());
    if (forwardStartAt(op, iteration, loops) + durationOf(op) > lastEnd)
      return end;
    return lastEnd;
  }

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Sample, 1>> samples;
  llvm::SmallVector<Extrapolation> extrapolations;
  /// The iterations a LaneCycle skipped (see Cover).
  llvm::SmallVector<Cover> covers;
  /// An iteration the backward pass did not simulate starts an op before
  /// zero.
  bool skippedNegative = false;
  /// The backward pass takes an answer before it is due.
  bool answeredEarly = false;
  /// The backward pass keeps a request at its forward start where its link
  /// has no slot free for it, as slots handed out in reverse program order
  /// tell: requests holding theirs for different times may still fit
  /// (heldTooMany).
  bool slotsOverrun = false;

  /// Whether the requests the pass placed, as sampled, hold more of a
  /// link's slots at once than it has: those of the iterations it did not
  /// simulate held more.
  bool heldTooMany() {
    // Per link, its slots and when each request takes (+1) and frees (-1)
    // one.
    std::map<LinkKey,
             std::pair<int64_t, llvm::SmallVector<std::pair<int64_t, int>>>>
        spans;
    auto hold = [&](mlir::Operation *submit, int64_t start, int64_t busy) {
      auto &[slots, found] = spans[linkOf(submit)];
      slots = classical::intAttr(submit, "classical.slots", 64);
      found.push_back({start, 1});
      found.push_back({start + busy, -1});
    };
    forEachScheduled(*region, [&](mlir::Operation *op) {
      auto it = samples.find(op);
      if (it == samples.end())
        return;
      auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op);
      if (!block && !mlir::isa<classical::SubmitOp>(op))
        return;
      for (const Sample &sample : it->second) {
        if (!block) {
          hold(op, sample.start, durationOf(op) + budgetOf(op));
          continue;
        }
        for (const BlockRequest &request : requestsOf(block, false))
          hold(request.op, sample.start + request.start, request.busy);
      }
    });
    for (auto &[key, entry] : spans) {
      auto &[slots, found] = entry;
      // A slot freed at a time is free for a request starting then.
      llvm::sort(found);
      int64_t held = 0;
      for (auto [at, step] : found) {
        held += step;
        if (held > slots)
          return true;
      }
    }
    return false;
  }

  /// Whether every call holds every line from its submission until its
  /// answer is due (`pulse.blocks_on_calls`): lines are done before it, and
  /// none starts again before then.
  bool blocksOnCalls = false;

  /// On several controllers, some of which block on calls: per controller,
  /// whether it does, when its lines end and may start again (forward), and
  /// when they must be done by (backward). Empty otherwise.
  llvm::SmallVector<bool, 4> blocking;
  llvm::SmallVector<int64_t, 4> heldEnds, heldFloors, heldCeilings;

  /// Whether a loop whose iterations repeat only every few of them stops
  /// the pass, to be run that many iterations at a time (`regroup`).
  bool mayRegroup = true;
  /// Such a loop and how many iterations it repeats after, found when it
  /// had more than two runs of them left: the pass stopped there, so its
  /// schedule is not complete.
  struct Regroup {
    mlir::scf::ForOp loop;
    int64_t iterations;
  };
  std::optional<Regroup> regroup;
  /// How many iterations as written each loop regrouped before runs in each.
  llvm::DenseMap<mlir::Operation *, int64_t> regroupedBy;
  /// Whether the backward pass, finding a loop's first iterations kept
  /// forward starts that do not repeat, stops to have them peeled
  /// (`headPeel`) rather than simulating every iteration.
  bool mayPeelHead = true;
  /// Such a loop and how many of its first iterations to peel.
  std::optional<Regroup> headPeel;

private:
  /// The earliest start of each group's ops in the iteration being
  /// simulated, for a loop whose groups move by periods of their own.
  struct Track {
    const LoopGroups *groups = nullptr;
    llvm::SmallVector<int64_t, 4> earliest;
    /// Without groups yet, every start recorded, for them once built.
    llvm::SmallVector<std::pair<mlir::Operation *, int64_t>, 8> starts;
  };

  /// The carried times before and after an iteration whose values moved by
  /// different shifts: only two such in a row can repeat group by group, so
  /// a loop's groups wait for the second.
  struct Unshifted {
    llvm::SmallVector<int64_t> before, after;
  };

  // ── Shared state ──────────────────────────────────────────────────────

  /// The lanes of one kind of a timeline by when they are free, earliest
  /// (forward; latest, backward) first, ties to the lower index: the lane
  /// a scan of every lane would take, in O(log lanes) per op.
  struct LaneHeap {
    llvm::SmallVector<uint32_t, 4> heap, at;
    bool backward = false;

    void reset(size_t n, bool isBackward) {
      backward = isBackward;
      heap.resize(n);
      at.resize(n);
      for (size_t i = 0; i < n; ++i)
        heap[i] = at[i] = static_cast<uint32_t>(i);
    }

    size_t best() const { return heap[0]; }

    bool before(uint32_t a, uint32_t b, llvm::ArrayRef<int64_t> lanes) const {
      int64_t x = lanes[a], y = lanes[b];
      if (x != y)
        return backward ? x > y : x < y;
      return a < b;
    }

    void swap(size_t i, size_t j) {
      std::swap(heap[i], heap[j]);
      at[heap[i]] = static_cast<uint32_t>(i);
      at[heap[j]] = static_cast<uint32_t>(j);
    }

    void siftUp(size_t i, llvm::ArrayRef<int64_t> lanes) {
      while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!before(heap[i], heap[parent], lanes))
          return;
        swap(i, parent);
        i = parent;
      }
    }

    void siftDown(size_t i, llvm::ArrayRef<int64_t> lanes) {
      for (size_t n = heap.size();;) {
        size_t child = 2 * i + 1;
        if (child >= n)
          return;
        if (child + 1 < n && before(heap[child + 1], heap[child], lanes))
          ++child;
        if (!before(heap[child], heap[i], lanes))
          return;
        swap(i, child);
        i = child;
      }
    }

    /// Lane `lane` changed its time.
    void update(size_t lane, llvm::ArrayRef<int64_t> lanes) {
      siftUp(at[lane], lanes);
      siftDown(at[lane], lanes);
    }

    /// Every lane may have changed.
    void rebuild(llvm::ArrayRef<int64_t> lanes) {
      for (size_t i = heap.size() / 2; i-- > 0;)
        siftDown(i, lanes);
    }
  };

  /// A lane's time and owner before an arm of a branch first changed it,
  /// so the other arm starts from the lanes before the branch, and where
  /// the lane's note of an enclosing arm is (`earlier`; -1: none).
  struct LaneUndo {
    bool readout;
    uint32_t timeline, lane;
    int64_t time, owner, earlier;
  };

  void resetLanes(int64_t value, bool backward) {
    linkLanes.clear();
    laneReset = value;
    laneUndo.clear();
    armMarks.clear();
    byHeap = false;
    if (!resources.enabled)
      return;
    // Timelines that wait independently each use lanes of their own: as
    // many as they have lines, which the hardware has.
    size_t count = floors.size();
    driveLanes.assign(count, {});
    readoutLanes.assign(count, {});
    driveUses.assign(count, 0);
    readoutUses.assign(count, 0);
    for (size_t t = 0; t < count; ++t) {
      int64_t drives = isolated() ? lanesPer->drives[t] : resources.maxDrives;
      int64_t readouts =
          isolated() ? lanesPer->readouts[t] : resources.maxReadouts;
      driveLanes[t].assign(static_cast<size_t>(std::max<int64_t>(1, drives)),
                           value);
      readoutLanes[t].assign(
          static_cast<size_t>(std::max<int64_t>(1, readouts)), value);
    }
    if (several()) {
      driveOwners = driveLanes;
      readoutOwners = readoutLanes;
      for (Lanes *owners : {&driveOwners, &readoutOwners})
        for (auto &list : *owners)
          std::fill(list.begin(), list.end(), kNoOwner);
    }
    // Where no two controllers' starts may differ, every lane gap is 0 and
    // an op takes the earliest (latest, backward) lane, lowest first.
    byHeap = !several() || !startsMayDiffer();
    if (byHeap) {
      driveHeaps.assign(count, {});
      readoutHeaps.assign(count, {});
      for (size_t t = 0; t < count; ++t) {
        driveHeaps[t].reset(driveLanes[t].size(), backward);
        readoutHeaps[t].reset(readoutLanes[t].size(), backward);
      }
    }
    driveNoted.assign(count, {});
    readoutNoted.assign(count, {});
    for (size_t t = 0; t < count; ++t) {
      driveNoted[t].assign(driveLanes[t].size(), -1);
      readoutNoted[t].assign(readoutLanes[t].size(), -1);
    }
  }

  /// Whether the starts of two controllers may differ (their start
  /// uncertainties are not all 0), so that a lane's next op of another
  /// controller may have to wait the spread (laneGap).
  bool startsMayDiffer() const {
    for (int64_t c = 0, n = controllers.count(); c < n; ++c)
      if (controllers.uncertainty(c) != 0.0)
        return true;
    return false;
  }

  /// Give lane `lane` of a kind on timeline `t` the time `time` and owner
  /// `owner`, noting what it was in an arm of a branch.
  void setLane(bool readout, size_t t, size_t lane, int64_t time,
               int64_t owner) {
    auto &lanes = (readout ? readoutLanes : driveLanes)[t];
    noteLane(readout, t, lane);
    lanes[lane] = time;
    if (several())
      (readout ? readoutOwners : driveOwners)[t][lane] = owner;
    if (byHeap)
      (readout ? readoutHeaps : driveHeaps)[t].update(lane, lanes);
  }

  /// In an arm of a branch, note lane `lane` of a kind on timeline `t` as
  /// it is, unless the arm noted it already: an arm notes each lane once,
  /// so the notes of the arms being scheduled are at most as many as the
  /// lanes times how deep the arms are.
  void noteLane(bool readout, size_t t, size_t lane) {
    if (armMarks.empty())
      return;
    int64_t &noted = (readout ? readoutNoted : driveNoted)[t][lane];
    if (noted >= static_cast<int64_t>(armMarks.back()))
      return;
    laneUndo.push_back(
        {readout, static_cast<uint32_t>(t), static_cast<uint32_t>(lane),
         (readout ? readoutLanes : driveLanes)[t][lane],
         several() ? (readout ? readoutOwners : driveOwners)[t][lane]
                   : kNoOwner,
         noted});
    noted = static_cast<int64_t>(laneUndo.size()) - 1;
  }

  /// Before code that moves many lanes at once (a loop skipping
  /// iterations): in an arm of a branch, note every lane as it was.
  void noteEveryLane() {
    if (armMarks.empty())
      return;
    for (bool readout : {false, true}) {
      Lanes &all = readout ? readoutLanes : driveLanes;
      for (size_t t = 0; t < all.size(); ++t)
        for (size_t lane = 0; lane < all[t].size(); ++lane)
          noteLane(readout, t, lane);
    }
  }

  /// An arm of a branch begins: its notes start at the end of the list.
  size_t beginArm() {
    armMarks.push_back(laneUndo.size());
    return laneUndo.size();
  }

  /// A branch ends (after the merge of its arms, or after its then arm
  /// without an else): its notes become the enclosing arm's, a lane that
  /// arm noted already keeping that note; outside every branch there are
  /// none.
  void endArm() {
    size_t mark = armMarks.pop_back_val();
    if (armMarks.empty()) {
      for (const LaneUndo &u : laneUndo)
        (u.readout ? readoutNoted : driveNoted)[u.timeline][u.lane] = -1;
      laneUndo.clear();
      return;
    }
    auto outer = static_cast<int64_t>(armMarks.back());
    size_t kept = mark;
    for (size_t i = mark; i < laneUndo.size(); ++i) {
      LaneUndo u = laneUndo[i];
      int64_t &noted =
          (u.readout ? readoutNoted : driveNoted)[u.timeline][u.lane];
      if (u.earlier >= outer) {
        noted = u.earlier;
        continue;
      }
      noted = static_cast<int64_t>(kept);
      laneUndo[kept++] = u;
    }
    laneUndo.resize(kept);
  }

  /// After code that moved many lanes at once.
  void reorderLanes() {
    if (!byHeap)
      return;
    for (size_t t = 0; t < driveLanes.size(); ++t) {
      driveHeaps[t].rebuild(driveLanes[t]);
      readoutHeaps[t].rebuild(readoutLanes[t]);
    }
  }

  /// A lane's time and owner as the then arm of a branch left it.
  struct ArmLane {
    int64_t then, thenOwner;
  };
  using ArmLanes =
      llvm::SmallDenseMap<std::tuple<uint8_t, uint32_t, uint32_t>, ArmLane, 4>;

  /// After the then arm of a branch: note the lanes it changed as it left
  /// them, and put them back as they were before it (from `mark` on the
  /// undo list).
  void undoArm(size_t mark, ArmLanes &changed) {
    while (laneUndo.size() > mark) {
      LaneUndo u = laneUndo.pop_back_val();
      auto &lanes = (u.readout ? readoutLanes : driveLanes)[u.timeline];
      int64_t owner = several() ? (u.readout ? readoutOwners
                                             : driveOwners)[u.timeline][u.lane]
                                : kNoOwner;
      auto [it, inserted] = changed.try_emplace(
          std::make_tuple(uint8_t(u.readout), u.timeline, u.lane));
      if (inserted)
        it->second = {lanes[u.lane], owner};
      lanes[u.lane] = u.time;
      (u.readout ? readoutNoted : driveNoted)[u.timeline][u.lane] = u.earlier;
      if (several())
        (u.readout ? readoutOwners : driveOwners)[u.timeline][u.lane] = u.owner;
      if (byHeap)
        (u.readout ? readoutHeaps : driveHeaps)[u.timeline].update(u.lane,
                                                                   lanes);
    }
  }

  /// After the else arm: each lane either arm changed (the then arm's in
  /// `changed`, the else arm's on the undo list from `mark`) as busy as the
  /// busier arm left it (`later`; the less busy, backward), its owner
  /// either arm's.
  void mergeArms(size_t mark, ArmLanes &changed, bool later) {
    for (size_t i = mark; i < laneUndo.size(); ++i) {
      const LaneUndo &u = laneUndo[i];
      auto [it, inserted] = changed.try_emplace(
          std::make_tuple(uint8_t(u.readout), u.timeline, u.lane));
      // Only the else arm changed it: the then arm left it as it was.
      if (inserted)
        it->second = {u.time, u.owner};
    }
    for (auto &[key, arm] : changed) {
      auto [readout, t, lane] = key;
      int64_t now = (readout ? readoutLanes : driveLanes)[t][lane];
      int64_t merged =
          later ? std::max(now, arm.then) : std::min(now, arm.then);
      int64_t owner = kNoOwner;
      bool ownerChanged = false;
      if (several()) {
        int64_t mine = (readout ? readoutOwners : driveOwners)[t][lane];
        owner = mine;
        if (mine != arm.thenOwner)
          owner = mine == kNoOwner            ? arm.thenOwner
                  : arm.thenOwner == kNoOwner ? mine
                                              : kAnyOwner;
        ownerChanged = owner != mine;
      }
      if (merged != now || ownerChanged)
        setLane(readout, t, lane, merged, owner);
    }
  }

  // ── Controllers ───────────────────────────────────────────────────────

  /// A lane's owner before any op takes it, and one that may be any
  /// controller.
  static constexpr int64_t kNoOwner = -2, kAnyOwner = -1;

  /// Whether the lines are on several controllers: never without line
  /// tracing.
  bool several() const {
    if constexpr (Gridded)
      return controllers.enabled();
    else
      return false;
  }

  /// The controller that plays the line `line` is a version of, or -1.
  int64_t controllerOfLine(mlir::Value line) {
    if constexpr (Gridded)
      return controllers.of(line, lines);
    else
      return kAnyOwner;
  }

  /// The controller that plays the line `op` is on, or -1.
  int64_t controllerOf(mlir::Operation *op) {
    return controllerOfLine(op->getOperand(0));
  }

  /// The controller that read `measurement`, or -1.
  int64_t readerOf(mlir::Value measurement) {
    auto readout = measurement.getDefiningOp<pulse::ReadoutOp>();
    return readout ? controllerOf(readout) : -1;
  }

  /// The controllers the ops of schedule block `block` play on (-1: one
  /// not known), as the flat schedule finds each op's.
  llvm::ArrayRef<int64_t> playersOf(pulse::ScheduleBlockOp block) {
    auto [it, inserted] = blockPlayers.try_emplace(block.getOperation());
    auto &players = it->second;
    if (inserted)
      block.getBody().walk([&](mlir::Operation *op) {
        if (!isTimedOp(op))
          return;
        int64_t owner = controllerOf(op);
        if (!llvm::is_contained(players, owner))
          players.push_back(owner);
      });
    return players;
  }

  /// On several controllers, the outcomes block input `input` carries (a
  /// measurement, or a value computed from one or from an answer), each
  /// with how much later than on its reader it is usable on every
  /// controller `block` plays on, as the flat schedule waits for each op
  /// in a branch on it.
  llvm::SmallVector<std::pair<mlir::Value, int64_t>, 2>
  carriedOutcomes(pulse::ScheduleBlockOp block, mlir::Value input) {
    llvm::SmallVector<std::pair<mlir::Value, int64_t>, 2> found;
    if (!several() || isLineType(input.getType()) ||
        mlir::isa<pulse::ToneType>(input.getType()))
      return found;
    for (mlir::Value measurement : conditionMeasurements(input)) {
      int64_t reader = readerOf(measurement), most = 0;
      for (int64_t player : playersOf(block))
        most = std::max(most, controllers.extra(reader, player));
      found.push_back({measurement, most});
    }
    return found;
  }

  /// How much later than scheduled the readout `op` ends once its line is
  /// delayed, and so its outcome is usable.
  int64_t readoutDelayOf(mlir::Operation *op) {
    llvm::ArrayRef<int64_t> delays = resources.readoutDelays;
    if (delays.empty())
      return 0;
    mlir::Operation *allocation = trace(op->getOperand(0));
    if (!allocation)
      return 0;
    int64_t qudit = pulse::quditOf(allocation);
    if (qudit < 0 || qudit >= static_cast<int64_t>(delays.size()))
      return 0;
    return delays[qudit];
  }

  /// In a branch on readouts, no op of `controller` (-1: any) starts before
  /// their outcomes reach it.
  int64_t armFloorOf(int64_t controller) const {
    if (armFloors.empty())
      return 0;
    int64_t latest = 0;
    for (const auto &readers : armFloors)
      for (auto [reader, time] : readers)
        latest = std::max(latest, time + controllers.extra(reader, controller));
    return latest;
  }

  /// The earliest start of an op of each controller in an arm, as
  /// (controller, start) pairs (`armStarts`).
  using ArmStarts = llvm::SmallVector<std::pair<int64_t, int64_t>, 4>;

  /// The op or sync of `owner` (-1: of no one controller) in the arm being
  /// scheduled backward starts at `start`. A start of kNever says nothing,
  /// as for a controller with no op in the arm.
  void noteArmStart(int64_t owner, int64_t start) {
    if (owner < 0) {
      armStart = std::min(armStart, start);
      return;
    }
    if (start == kNever)
      return;
    if (armIndexed) {
      if (static_cast<size_t>(owner) >= armSlots.size())
        armSlots.resize(owner + 1, {0, 0});
      auto &[stamp, at] = armSlots[owner];
      if (stamp == armStamp) {
        armStarts[at].second = std::min(armStarts[at].second, start);
        return;
      }
      stamp = armStamp;
      at = static_cast<uint32_t>(armStarts.size());
      armStarts.push_back({owner, start});
      return;
    }
    for (auto &[c, earliest] : armStarts)
      if (c == owner) {
        earliest = std::min(earliest, start);
        return;
      }
    armStarts.push_back({owner, start});
    if (armStarts.size() > kArmIndexedAfter)
      indexArmStarts();
  }

  /// Index `armStarts` by controller (it is long).
  void indexArmStarts() {
    ++armStamp;
    armIndexed = true;
    for (auto [i, entry] : llvm::enumerate(armStarts)) {
      if (static_cast<size_t>(entry.first) >= armSlots.size())
        armSlots.resize(entry.first + 1, {0, 0});
      armSlots[entry.first] = {armStamp, static_cast<uint32_t>(i)};
    }
  }

  /// The arm's starts by controller, taken out: none are left.
  ArmStarts takeArmStarts() {
    ArmStarts taken;
    if (!armStarts.empty()) {
      taken = std::move(armStarts);
      armStarts.clear();
    }
    armIndexed = false;
    return taken;
  }

  /// Make `starts` the arm's starts by controller (there are none).
  void putArmStarts(ArmStarts &starts) {
    if (starts.empty())
      return;
    armStarts = std::move(starts);
    starts.clear();
    if (armStarts.size() > kArmIndexedAfter)
      indexArmStarts();
  }

  /// How long after a lane's op of `owner` an op of `controller` may start
  /// on it: their starts may be that far apart.
  int64_t laneGap(int64_t owner, int64_t controller) const {
    if (owner == kNoOwner || (owner >= 0 && owner == controller))
      return 0;
    return controllers.spread(owner, controller);
  }

  // ── Calls that hold controllers ───────────────────────────────────────

  /// From the controllers of `system` that block on calls: every call holds
  /// every line, or, on several controllers whose lines are traced, the
  /// lines of those that block that the call involves (`heldBy`).
  void holdControllers(pulse::ControlSystemAttr system,
                       const pulse::Controllers &all) {
    llvm::ArrayRef<pulse::ControllerAttr> listed = system.getControllers();
    if (!Gridded || !all.enabled() ||
        static_cast<int64_t>(listed.size()) != all.count()) {
      blocksOnCalls = true;
      return;
    }
    owners = all;
    auto count = static_cast<size_t>(all.count());
    blocking.assign(count, false);
    for (auto [c, controller] : llvm::enumerate(listed))
      blocking[c] = controller.getBlocksOnCalls();
    heldEnds.assign(count, 0);
    heldFloors.assign(count, 0);
    heldCeilings.assign(count, kNever);
  }

  bool holding() const { return !blocking.empty(); }

  /// The controller of the line timed op `op` is on, or -1, whatever the
  /// policy.
  int64_t heldOwnerOf(mlir::Operation *op) {
    if constexpr (Gridded)
      return owners.of(op->getOperand(0), lines);
    else
      return kAnyOwner;
  }

  /// No op of a held controller starts before the answers it waits for
  /// are due; a line of no known controller waits for every one.
  int64_t heldFloorOf(mlir::Operation *op) {
    int64_t c = heldOwnerOf(op);
    return c >= 0 ? heldFloors[c] : *llvm::max_element(heldFloors);
  }

  int64_t heldCeilingOf(mlir::Operation *op) {
    int64_t c = heldOwnerOf(op);
    return c >= 0 ? heldCeilings[c] : *llvm::min_element(heldCeilings);
  }

  /// Move the times an iteration changed by `distance`, for the iterations
  /// not simulated.
  static void moveChanged(llvm::MutableArrayRef<int64_t> times,
                          llvm::ArrayRef<int64_t> before, int64_t distance) {
    for (auto [now, was] : llvm::zip(times, before))
      if (now != was)
        now += distance;
  }

  void noteHeldEnd(mlir::Operation *op, int64_t end) {
    int64_t c = heldOwnerOf(op);
    if (c >= 0) {
      heldEnds[c] = std::max(heldEnds[c], end);
      return;
    }
    for (int64_t &at : heldEnds)
      at = std::max(at, end);
  }

  /// The controllers that block on calls which `submit` holds: the one
  /// that issues it (whose instance of a per-controller link runs it, else
  /// those that read the measurements it sends, else any), and those that
  /// play an op that uses its answer, computes with it or branches on it. A
  /// line of no known controller, or an answer used in a way not followed
  /// here, holds them all.
  llvm::ArrayRef<int64_t> heldBy(classical::SubmitOp submit) {
    auto [it, inserted] = heldCalls.try_emplace(submit);
    if (!inserted)
      return it->second;
    llvm::SmallVector<bool, 4> held(blocking.size(), false);
    auto hold = [&](int64_t c) {
      if (c >= 0)
        held[c] = true;
      else
        std::fill(held.begin(), held.end(), true);
    };
    auto played = [](mlir::Operation *op) {
      return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op);
    };
    int64_t instance = classical::intAttr(submit, classical::kInstanceAttr, -1);
    llvm::SmallVector<mlir::Value> sent = valueMeasurements(submit);
    if (instance >= 0 && instance < static_cast<int64_t>(held.size()))
      held[instance] = true;
    else if (sent.empty())
      hold(kAnyOwner);
    for (mlir::Value measurement : sent) {
      auto readout = measurement.getDefiningOp<pulse::ReadoutOp>();
      hold(readout ? heldOwnerOf(readout) : kAnyOwner);
    }
    llvm::SmallVector<mlir::Value> answers;
    for (mlir::Operation *user : submit.getFuture().getUsers()) {
      if (auto wait = mlir::dyn_cast<classical::WaitOp>(user))
        llvm::append_range(answers, wait.getValues());
      else
        hold(kAnyOwner);
    }
    llvm::DenseSet<mlir::Value> seen;
    while (!answers.empty()) {
      mlir::Value value = answers.pop_back_val();
      if (!seen.insert(value).second)
        continue;
      for (mlir::OpOperand &use : value.getUses()) {
        mlir::Operation *user = use.getOwner();
        unsigned index = use.getOperandNumber();
        if (played(user)) {
          hold(heldOwnerOf(user));
        } else if (mlir::isa<classical::SubmitOp>(user)) {
          // An answer sent on to another call is that call's.
        } else if (mlir::isa<mlir::scf::YieldOp>(user)) {
          mlir::Operation *parent = user->getParentOp();
          if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
            answers.push_back(loop.getRegionIterArgs()[index]);
            answers.push_back(loop->getResult(index));
          } else if (mlir::isa<mlir::scf::IfOp>(parent)) {
            answers.push_back(parent->getResult(index));
          } else {
            hold(kAnyOwner);
          }
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(user);
                   loop && index >= loop.getNumControlOperands()) {
          index -= loop.getNumControlOperands();
          answers.push_back(loop.getRegionIterArgs()[index]);
          answers.push_back(loop->getResult(index));
        } else {
          // A branch on it, a loop over it or a block given it holds what
          // it plays.
          if (user->getNumRegions())
            user->walk([&](mlir::Operation *inner) {
              if (played(inner))
                hold(heldOwnerOf(inner));
            });
          llvm::append_range(answers, user->getResults());
        }
      }
    }
    for (auto [c, is] : llvm::enumerate(held))
      if (is && blocking[c])
        it->second.push_back(static_cast<int64_t>(c));
    return it->second;
  }

  /// The measurements a value is computed from, not through other timed
  /// values or answers.
  static void valueMeasurements(mlir::Value value,
                                llvm::SmallVectorImpl<mlir::Value> &found,
                                llvm::DenseSet<mlir::Value> &seen) {
    // The values still to follow, the next one last: what a value is
    // computed from in order (a branch's condition, then what each arm
    // yields; an operation's operands), each to its end before the next.
    llvm::SmallVector<mlir::Value, 16> todo{value};
    while (!todo.empty()) {
      mlir::Value next = todo.pop_back_val();
      if (!seen.insert(next).second)
        continue;
      if (mlir::isa<pulse::MeasurementType>(next.getType())) {
        found.push_back(next);
        continue;
      }
      if (isTimedType(next.getType()))
        continue;
      mlir::Operation *def = next.getDefiningOp();
      if (!def || mlir::isa<classical::WaitOp>(def))
        continue;
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
        auto index = mlir::cast<mlir::OpResult>(next).getResultNumber();
        if (!branch.getElseRegion().empty())
          todo.push_back(branch.elseYield().getOperand(index));
        todo.push_back(branch.thenYield().getOperand(index));
        todo.push_back(branch.getCondition());
        continue;
      }
      if (def->getNumRegions())
        continue;
      for (mlir::Value operand : llvm::reverse(def->getOperands()))
        todo.push_back(operand);
    }
  }

  /// The measurements the operands of timed op `op` other than its line and
  /// tone are computed from.
  static llvm::SmallVector<mlir::Value> valueMeasurements(mlir::Operation *op) {
    llvm::SmallVector<mlir::Value> found;
    llvm::DenseSet<mlir::Value> seen;
    mlir::Value tone = toneOperand(op);
    for (mlir::Value operand : op->getOperands().drop_front())
      if (operand != tone)
        valueMeasurements(operand, found, seen);
    return found;
  }

  // ── Timelines ─────────────────────────────────────────────────────────

  bool isolated() const { return lanesPer != nullptr; }

  /// The timeline of a call, block, line or tone allocation, or of the
  /// first line or tone `op` plays on.
  int64_t timelineOf(mlir::Operation *op) {
    if (!isolated())
      return 0;
    auto [it, inserted] = timelineCache.try_emplace(op, 0);
    if (!inserted)
      return it->second;
    if (op->hasAttr(classical::kTimelineAttr)) {
      it->second = classical::intAttr(op, classical::kTimelineAttr);
    } else {
      for (mlir::Value operand : op->getOperands())
        if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                      pulse::ToneType>(operand.getType())) {
          it->second = timelineOfValue(operand);
          break;
        }
    }
    return it->second;
  }

  int64_t timelineOfValue(mlir::Value value) {
    if (!isolated())
      return 0;
    mlir::Operation *allocation = trace(value);
    return allocation ? classical::intAttr(allocation, classical::kTimelineAttr)
                      : 0;
  }

  /// No op of timeline `t` starts before this: its last run-time wait, and
  /// the condition of the branch the op is in.
  int64_t floorOf(int64_t t) const { return std::max(floor, floors[t]); }

  /// Where a timeline's lines end if nothing later uses them.
  int64_t endOf(int64_t t) const { return isolated() ? ends[t] : makespan; }

  /// A link as one timeline sees it, on one controller for a link with an
  /// instance per controller: its share of the slots, and its own order of
  /// submissions.
  using LinkKey = std::tuple<int64_t, int64_t, int64_t>;
  LinkKey linkOf(mlir::Operation *submit) {
    return {classical::intAttr(submit, "classical.device"),
            classical::intAttr(submit, classical::kInstanceAttr, -1),
            timelineOf(submit)};
  }

  /// Whether a submission keeps an order of its own on its link
  /// (`linkReady`, `linkLate`) rather than the link value threaded through
  /// the program: with timelines, or on its controller's instance.
  bool ownOrder(mlir::Operation *submit) const {
    return isolated() || submit->hasAttr(classical::kInstanceAttr);
  }

  void record(mlir::Operation *op, int64_t start) {
    samples[op].push_back({iteration, start});
    // Only the backward pass tracks the earliest starts.
    for (Track *track : tracks)
      if (!track->groups)
        track->starts.push_back({op, start});
      else if (track->groups->ofOp.contains(op)) {
        int64_t &earliest =
            track->earliest[track->groups->groupOf(op, /*backward=*/true)];
        earliest = std::min(earliest, start);
      }
  }

  /// Lower the earliest start `arm` of a branch's arm to `simulated`, a
  /// start in the iteration last simulated, moved by `distance` to where it
  /// is in an iteration the backward pass extrapolated.
  void lowerArmStart(int64_t &arm, int64_t simulated, int64_t distance) {
    if (simulated == kNever)
      return;
    int64_t start;
    if (__builtin_add_overflow(simulated, distance, &start)) {
      skippedNegative = true;
      return;
    }
    if (start < 0)
      skippedNegative = true;
    arm = std::min(arm, start);
  }

  // ── Lanes over a loop's iterations ─────────────────────────────────────

  using Lanes = llvm::SmallVector<llvm::SmallVector<int64_t>, 1>;

  /// The lanes, and how many ops took each timeline's, before an iteration.
  struct LaneState {
    Lanes drives, readouts, driveOwners, readoutOwners;
    llvm::SmallVector<int64_t, 1> driveUses, readoutUses;
  };

  LaneState laneState() const {
    return {driveLanes,    readoutLanes, driveOwners,
            readoutOwners, driveUses,    readoutUses};
  }

  /// Whether lanes of equal times are interchangeable for the rest of the
  /// schedule: on one controller, with no branch whose arms both take lanes,
  /// as their lanes merge lane by lane. An arm that takes none leaves each
  /// lane where the other arm's is or before it.
  bool lanesByTime() {
    if (!lanesKnown) {
      lanesKnown = true;
      auto takesLanes = [](mlir::Region &arm) {
        return arm
            .walk([](mlir::Operation *op) {
              return isTimedOp(op) && !mlir::isa<pulse::WaitOp>(op)
                         ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
            })
            .wasInterrupted();
      };
      lanesInterchangeable =
          !several() && !region
                             ->walk([&](mlir::scf::IfOp branch) {
                               return takesLanes(branch.getThenRegion()) &&
                                              takesLanes(branch.getElseRegion())
                                          ? mlir::WalkResult::interrupt()
                                          : mlir::WalkResult::advance();
                             })
                             .wasInterrupted();
    }
    return lanesInterchangeable;
  }

  /// Whether an iteration moved the lanes by `shift`, so that the next one
  /// plays as it did moved by `shift`: on each timeline, the lanes of a kind
  /// an op took are those before moved by `shift`, as a multiset when lanes
  /// of equal times are interchangeable, else lane by lane with the same
  /// owners.
  bool lanesMoved(const LaneState &before, int64_t shift) {
    return lanesMovedBy(before, shift, shift);
  }

  /// lanesMoved, the drive lanes by `driveShift` and the readout lanes by
  /// `readoutShift`.
  bool lanesMovedBy(const LaneState &before, int64_t driveShift,
                    int64_t readoutShift) {
    bool byTime = lanesByTime();
    auto moved = [&](const Lanes &was, const Lanes &now, const Lanes &wasOwners,
                     const Lanes &nowOwners, llvm::ArrayRef<int64_t> wasUses,
                     llvm::ArrayRef<int64_t> nowUses, int64_t shift) {
      for (size_t t = 0; t < now.size(); ++t) {
        if (wasUses[t] == nowUses[t])
          continue;
        if (byTime) {
          llvm::SmallVector<int64_t, 4> a(was[t].begin(), was[t].end());
          llvm::SmallVector<int64_t, 4> b(now[t].begin(), now[t].end());
          llvm::sort(a);
          llvm::sort(b);
          for (auto [x, y] : llvm::zip(a, b))
            if (y != x + shift)
              return false;
          continue;
        }
        if (several() && wasOwners[t] != nowOwners[t])
          return false;
        for (auto [x, y] : llvm::zip(was[t], now[t]))
          if (y != x + shift)
            return false;
      }
      return true;
    };
    return moved(before.drives, driveLanes, before.driveOwners, driveOwners,
                 before.driveUses, driveUses, driveShift) &&
           moved(before.readouts, readoutLanes, before.readoutOwners,
                 readoutOwners, before.readoutUses, readoutUses, readoutShift);
  }

  /// How many iterations moving as the one before a loop's lanes are
  /// compared over, past which they are not: lanes that will move as the
  /// iterations do have once each was taken again (none without resources).
  int64_t laneTriesPerLoop() const {
    if (!resources.enabled)
      return 0;
    size_t most = 0;
    for (auto [drives, readouts] : llvm::zip(driveLanes, readoutLanes))
      most = std::max(most, drives.size() + readouts.size());
    return 2 * static_cast<int64_t>(most) + 2;
  }

  /// Add the lanes now to `history`, keeping the last kMostRegrouped.
  void pushLanes(llvm::SmallVectorImpl<LaneState> &history) const {
    if (static_cast<int64_t>(history.size()) == kMostRegrouped)
      history.erase(history.begin());
    history.push_back(laneState());
  }

  /// lanesMoved, counting the tries left down when not.
  bool lanesSteady(const LaneState &before, int64_t shift, int64_t &tries) {
    if (tries <= 0)
      return false;
    if (lanesMoved(before, shift))
      return true;
    --tries;
    return false;
  }

  /// Moves the lanes an iteration took by `distance`, as the iterations not
  /// simulated do.
  void moveLanes(const LaneState &before, int64_t distance) {
    moveLanesBy(before, distance, distance);
  }

  /// moveLanes, the drive lanes by `driveDistance` and the readout lanes by
  /// `readoutDistance`.
  void moveLanesBy(const LaneState &before, int64_t driveDistance,
                   int64_t readoutDistance) {
    notePicksSkipped();
    noteEveryLane();
    for (size_t t = 0; t < driveLanes.size(); ++t) {
      if (driveUses[t] != before.driveUses[t])
        for (int64_t &lane : driveLanes[t])
          lane += driveDistance;
      if (readoutUses[t] != before.readoutUses[t])
        for (int64_t &lane : readoutLanes[t])
          lane += readoutDistance;
    }
    reorderLanes();
  }

  // ── Lanes past the iterations they are compared over ──────────────────

  /// The times of the lanes the ops of an iteration took, in the order they
  /// took them, and whether a lane held one back: its time was later than
  /// the op could start (forward), or earlier than it could end (backward).
  /// Not `complete` when a loop in it moved lanes past iterations it did
  /// not simulate.
  struct LanePicks {
    llvm::SmallVector<int64_t, 4> drives, readouts;
    bool held = false, complete = true;
    void clear() {
      drives.clear();
      readouts.clear();
      held = false;
      complete = true;
    }
  };

  /// Note that an op took a drive (or readout) lane of time `lane`, which
  /// held it back when `held`, in the iteration of each loop watched.
  void notePick(bool isDrive, int64_t lane, bool held) {
    for (LanePicks *picks : picking) {
      (isDrive ? picks->drives : picks->readouts).push_back(lane);
      picks->held |= held;
    }
  }

  /// Lanes moved past iterations not simulated: the iterations watched
  /// around them did not note every lane taken.
  void notePicksSkipped() {
    for (LanePicks *picks : picking)
      picks->complete = false;
  }

  /// A loop's lanes where the steady test finds them not moved as the
  /// iteration did: past the iterations the lanes are compared over
  /// (laneTriesPerLoop), after which the pass would simulate every iteration
  /// left, or, for a loop whose carried values move by periods of their own
  /// (LoopGroups), where its lanes did not move with them. For a loop on one
  /// controller and timeline, with no call. It stands in for the lanes'
  /// part of the steady test only (see `steady`). After an iteration that
  /// moved every op, nested ones too, as it moved them in the one before
  /// (by the shift, or by the shift of its group), each kind of lane by the
  /// shift of the ops that take it, it is steady when
  ///  - the lanes it did not take that are later (forward; earlier,
  ///    backward) than every lane it gave a time to stay apart, and the
  ///    others moved by the shift (lane by lane; as a multiset where lanes
  ///    of equal times are interchangeable, lanesByTime). Without the lanes
  ///    apart every iteration after plays as this one moved. It then also
  ///    does with them while it gives no lane a time as late as theirs: if
  ///    that holds to the last iteration, or there are none, it is steady
  ///    and they stay; else the loop skips, moved, to the last iteration
  ///    that does and goes on from there. And where lanes of equal times are
  ///    interchangeable, the iteration noted every lane it took and none held
  ///    an op back, every iteration after plays as it moved as well: without
  ///    the lanes apart none waits, and with them each op takes a lane at
  ///    least as early. Each op hands the earliest lane on to a later time,
  ///    so the lanes left at the end are the latest of the lanes before and
  ///    those the ops gave times to: those of the last iterations, and the
  ///    lanes apart or the last the iterations took;
  ///  - or its lanes moved by n times the shift over its last n iterations
  ///    (n from 2 to twice the product of the numbers of drive and readout
  ///    lanes and their sum, and 2), each of which moved every op as the
  ///    last did: they repeat every n iterations.
  /// The iterations not simulated are noted as Covers, and the schedule is
  /// written as when every iteration is simulated.
  class LaneCycle {
    using Shifts = llvm::SmallVector<std::optional<int64_t>, 4>;

  public:
    LaneCycle(Scheduler &scheduler, mlir::scf::ForOp loop, int64_t count,
              int64_t sign)
        : s(scheduler), loop(loop), count(count), sign(sign),
          outer(scheduler.iteration.begin(),
                std::prev(scheduler.iteration.end())) {
      on = s.resources.enabled && !s.several() && !s.classical &&
           !s.holding() && s.floors.size() == 1 &&
           !loop.getBody()
                ->walk([](pulse::ScheduleBlockOp) {
                  return mlir::WalkResult::interrupt();
                })
                .wasInterrupted();
      if (!on)
        return;
      forEachScheduled(loop.getRegion(),
                       [&](mlir::Operation *op) { ops.push_back(op); });
      // Each kind of lane repeats after at most as many iterations as it
      // has lanes, both after the product.
      int64_t drives = static_cast<int64_t>(s.driveLanes[0].size()),
              readouts = static_cast<int64_t>(s.readoutLanes[0].size());
      longest = 2 * (drives * readouts + drives + readouts) + 2;
      steps.resize(static_cast<size_t>(longest) + 1);
    }
    LaneCycle(const LaneCycle &) = delete;
    LaneCycle &operator=(const LaneCycle &) = delete;
    ~LaneCycle() { llvm::erase(s.picking, &picks); }

    /// After iteration `k`: whether the loop is steady, or (`skip`) how many
    /// iterations it may skip. It decides the lanes' part of the steady test
    /// only. `shift` is none unless every other part held for the
    /// iteration: each carried value (line, tone, link or other) moved by
    /// `shift` as in the one before, on every grid, with the links' slots
    /// moved too and every value then past what the body reads from outside
    /// the loop (`bounded`, which forward also tells for an iteration whose
    /// other parts did not hold).
    bool steady(int64_t k, std::optional<int64_t> shift, bool bounded) {
      Moves moves;
      if (shift)
        moves = {true, *shift, *shift, *shift, nullptr, {}, {*shift}};
      return decide(k, moves, bounded);
    }

    /// `steady` for a loop whose carried values move by periods of their
    /// own (`groups`): `shifts` is none unless every other part of the
    /// steady test of such a loop held, the lanes' part (groupLanesMoved)
    /// aside.
    bool steadyGroups(int64_t k, const LoopGroups &groups, const Shifts *shifts,
                      bool bounded) {
      Moves moves;
      if (shifts) {
        auto shiftOf =
            [&](const llvm::SmallBitVector &kind) -> std::optional<int64_t> {
          std::optional<int64_t> found;
          for (unsigned group : kind.set_bits()) {
            int64_t moved = (*shifts)[group].value_or(0);
            if (found && *found != moved)
              return std::nullopt;
            found = moved;
          }
          return found.value_or(0);
        };
        std::optional<int64_t> drives = shiftOf(groups.driveGroups),
                               readouts = shiftOf(groups.readoutGroups);
        if (drives && readouts) {
          moves = {true, *drives, *readouts, 0, &groups, *shifts, {}};
          for (const std::optional<int64_t> &shift : *shifts)
            moves.key.push_back(shift ? *shift : kNever);
        }
      }
      return decide(k, moves, bounded);
    }

    /// Whether the decision on iteration `k` found the loop steady.
    bool found(int64_t k) const { return took && filled > 0 && step(0).k == k; }

    /// How many iterations the last decision found the loop may skip, or 0.
    int64_t skip() const { return skipBy; }

    /// For a loop of groups, how far its groups move over `copies`
    /// iterations; none past 2**63.
    std::optional<llvm::SmallVector<int64_t, 4>>
    groupDistances(int64_t copies) const {
      return Scheduler::groupDistances(moves.shifts, copies);
    }
    const LoopGroups *groups() const { return moves.groups; }

    /// Move the lanes as the `copies` iterations after the last one
    /// simulated, to the loop's end, do.
    void move(int64_t copies) {
      took = false;
      const Step &now = step(0);
      int64_t k = now.k;
      s.covers.push_back({loop, outer, k, sign > 0 ? k : -1,
                          sign > 0 ? count : k, 0, periods()});
      s.notePicksSkipped();
      s.noteEveryLane();
      moveSkipped(copies);
      s.reorderLanes();
    }

    /// The lanes part of move().
    void moveSkipped(int64_t copies) {
      const Step &now = step(0);
      if (cycle > 0) {
        // The iteration of the last cycle that the last one skipped
        // repeats.
        int64_t back = (cycle - copies % cycle) % cycle;
        const LaneState &from = step(back).lanes;
        const LaneState &first = step(cycle).lanes;
        auto kind = [&](llvm::SmallVectorImpl<int64_t> &lanes,
                        llvm::ArrayRef<int64_t> was, int64_t firstUses,
                        int64_t uses, int64_t shift) {
          if (firstUses != uses)
            for (auto [lane, before] : llvm::zip(lanes, was))
              lane = before + (copies + back) * shift;
        };
        kind(s.driveLanes[0], from.drives[0], first.driveUses[0],
             s.driveUses[0], moves.drives);
        kind(s.readoutLanes[0], from.readouts[0], first.readoutUses[0],
             s.readoutUses[0], moves.readouts);
        return;
      }
      const LaneState &was = step(1).lanes;
      if (was.driveUses[0] != now.lanes.driveUses[0])
        moveApart(s.driveLanes[0], drivesApart, now.picks.drives, copies,
                  moves.drives);
      if (was.readoutUses[0] != now.lanes.readoutUses[0])
        moveApart(s.readoutLanes[0], readoutsApart, now.picks.readouts, copies,
                  moves.readouts);
    }

    /// Skip the iterations the last decision found the loop may, moving
    /// the lanes as they do; the iteration last skipped keeps its samples,
    /// those of the last one simulated moved.
    void skipIterations() {
      int64_t copies = skipBy, k = step(0).k, to = k + sign * copies;
      skipBy = 0;
      s.covers.push_back(
          {loop, outer, k, sign > 0 ? k : to, sign > 0 ? to : k, 0, periods()});
      s.notePicksSkipped();
      s.noteEveryLane();
      const LaneState &was = step(1).lanes;
      const Step &now = step(0);
      auto moveKind = [&](llvm::SmallVectorImpl<int64_t> &lanes,
                          llvm::ArrayRef<size_t> apart, int64_t wasUses,
                          int64_t isUses, int64_t shift) {
        if (wasUses == isUses)
          return;
        for (auto [i, lane] : llvm::enumerate(lanes))
          if (!llvm::is_contained(apart, i))
            lane += copies * shift;
      };
      moveKind(s.driveLanes[0], drivesApart, was.driveUses[0],
               now.lanes.driveUses[0], moves.drives);
      moveKind(s.readoutLanes[0], readoutsApart, was.readoutUses[0],
               now.lanes.readoutUses[0], moves.readouts);
      s.reorderLanes();
      // The samples and covers of the iteration last skipped.
      size_t depth = outer.size();
      for (mlir::Operation *op : ops) {
        auto it = s.samples.find(op);
        if (it == s.samples.end())
          continue;
        auto &list = it->second;
        auto [begin, end] = runOf(list, list.size(), k);
        int64_t by = copies * opShift(op);
        for (size_t i = begin; i < end; ++i) {
          Sample moved = list[i];
          moved.iteration[depth] = to;
          moved.start += by;
          list.push_back(std::move(moved));
        }
      }
      for (size_t i = 0, n = s.covers.size(); i < n; ++i)
        if (inIteration(s.covers[i].outer, k)) {
          Cover moved = s.covers[i];
          moved.outer[depth] = to;
          s.covers.push_back(std::move(moved));
        }
      filled = 0;
    }

  private:
    /// How an iteration moved what the lanes' part of the test depends on:
    /// each kind of lane by the shift of the ops that take it, and each op
    /// by `all` or, in a loop of `groups`, by its group's of `shifts`; `key`
    /// tells two iterations that moved alike.
    struct Moves {
      bool ok = false;
      int64_t drives = 0, readouts = 0, all = 0;
      const LoopGroups *groups = nullptr;
      Shifts shifts;
      llvm::SmallVector<int64_t, 4> key;
    };

    struct Step {
      int64_t k = 0;
      bool ok = false, bounded = false, logged = false;
      llvm::SmallVector<int64_t, 4> key;
      /// How many iterations up to this one moved alike.
      int64_t run = 0;
      LaneState lanes;
      /// The sums of the times of the drive and readout lanes.
      int64_t driveSum = 0, readoutSum = 0;
      LanePicks picks;
    };

    /// The iteration `back` iterations before the last one noted.
    Step &step(int64_t back) {
      return steps[static_cast<size_t>((next + steps.size() - 1 - back) %
                                       steps.size())];
    }
    const Step &step(int64_t back) const {
      return steps[static_cast<size_t>((next + steps.size() - 1 - back) %
                                       steps.size())];
    }

    int64_t opShift(mlir::Operation *op) const {
      if (!moves.groups)
        return moves.all;
      return moves.shifts[moves.groups->groupOf(op, sign < 0)].value_or(0);
    }

    /// The period of each op along the loop, for its Covers.
    std::shared_ptr<const llvm::DenseMap<mlir::Operation *, int64_t>>
    periods() const {
      auto found =
          std::make_shared<llvm::DenseMap<mlir::Operation *, int64_t>>();
      for (mlir::Operation *op : ops)
        (*found)[op] = sign * opShift(op);
      return found;
    }

    bool decide(int64_t k, const Moves &now, bool bounded) {
      took = false;
      skipBy = 0;
      if (!on)
        return false;
      // An iteration asked about again (as a loop of groups after its
      // steady test) keeps what was noted of it.
      bool again = filled > 0 && step(0).k == k;
      if (!again && filled > 0 && step(0).k != k - sign)
        filled = 0;
      if (!again) {
        bool logged = llvm::is_contained(s.picking, &picks);
        next = (next + 1) % steps.size();
        filled =
            std::min<int64_t>(filled + 1, static_cast<int64_t>(steps.size()));
        Step &noted = step(0);
        noted.k = k;
        noted.logged = logged;
        noted.lanes.drives = s.driveLanes;
        noted.lanes.readouts = s.readoutLanes;
        noted.lanes.driveUses = s.driveUses;
        noted.lanes.readoutUses = s.readoutUses;
        noted.driveSum = sum(s.driveLanes[0]);
        noted.readoutSum = sum(s.readoutLanes[0]);
        std::swap(noted.picks, picks);
        picks.clear();
        if (!logged)
          s.picking.push_back(&picks);
      }
      Step &noted = step(0);
      noted.ok = now.ok;
      noted.bounded = bounded;
      noted.key = now.key;
      // How many iterations up to this one moved alike.
      noted.run = 0;
      if (now.ok)
        noted.run = filled > 1 && step(1).ok && step(1).key == now.key
                        ? step(1).run + 1
                        : 1;
      if (!now.ok || sign * now.drives < 0 || sign * now.readouts < 0 ||
          (now.drives == 0 && now.readouts == 0))
        return false;
      if constexpr (Gridded)
        if (now.drives % s.grid.lcm() != 0 || now.readouts % s.grid.lcm() != 0)
          return false;
      int64_t left = sign > 0 ? count - 1 - k : k;
      if (left < 1)
        return false;
      moves = now;
      cycle = 0;
      closed = false;
      if (movedApart()) {
        const Step &last = step(0);
        // Lanes apart are taken again as the loop reaches them; for a loop
        // of groups the steady test may find it steady after, so the loop
        // is not ended before.
        if ((drivesApart.empty() && readoutsApart.empty()) ||
            (!now.groups && s.lanesByTime() && last.picks.complete &&
             !last.picks.held)) {
          closed = true;
          took = true;
        } else if (int64_t reach = reached(); reach >= left) {
          took = true;
        } else if (reach >= 1) {
          skipBy = reach;
        }
      }
      if (!took && !skipBy)
        took = repeats();
      return took;
    }

    static bool samePeriods(const Cover &a, const Cover &b) {
      if (!a.periods || !b.periods)
        return a.periods == b.periods;
      if (a.periods->size() != b.periods->size())
        return false;
      for (auto [op, period] : *a.periods)
        if (b.periods->lookup(op) != period || !b.periods->contains(op))
          return false;
      return true;
    }

    static int64_t sum(llvm::ArrayRef<int64_t> lanes) {
      int64_t total = 0;
      for (int64_t lane : lanes)
        total += lane;
      return total;
    }

    /// Whether `iteration` is in iteration `k` of this run of the loop.
    bool inIteration(llvm::ArrayRef<int64_t> iteration, int64_t k) const {
      return iteration.size() > outer.size() &&
             iteration.take_front(outer.size()) == llvm::ArrayRef(outer) &&
             iteration[outer.size()] == k;
    }

    /// The samples of `list` before `end` in iteration `k`, the last ones.
    std::pair<size_t, size_t> runOf(llvm::ArrayRef<Sample> list, size_t end,
                                    int64_t k) const {
      size_t begin = end;
      while (begin > 0 && inIteration(list[begin - 1].iteration, k))
        --begin;
      return {begin, end};
    }

    /// Whether every op of the last `n` iterations, nested ones too, started
    /// as much after it did in the iteration before as the last iteration
    /// moved it, and the loops in them skipped the same iterations.
    bool opsMoved(int64_t n) const {
      int64_t k = step(0).k;
      size_t depth = outer.size();
      for (mlir::Operation *op : ops) {
        auto it = s.samples.find(op);
        if (it == s.samples.end())
          continue;
        int64_t shift = opShift(op);
        llvm::ArrayRef<Sample> list = it->second;
        auto later = runOf(list, list.size(), k);
        for (int64_t i = 1; i <= n; ++i) {
          auto earlier = runOf(list, later.first, k - sign * i);
          if (later.second - later.first != earlier.second - earlier.first)
            return false;
          for (size_t j = 0; j < later.second - later.first; ++j) {
            const Sample &a = list[earlier.first + j];
            const Sample &b = list[later.first + j];
            if (llvm::ArrayRef(a.iteration).drop_front(depth + 1) !=
                    llvm::ArrayRef(b.iteration).drop_front(depth + 1) ||
                b.start - a.start != shift)
              return false;
          }
          later = earlier;
        }
      }
      auto coversIn = [&](int64_t kk) {
        llvm::SmallVector<const Cover *> found;
        for (const Cover &cover : s.covers)
          if (inIteration(cover.outer, kk))
            found.push_back(&cover);
        return found;
      };
      auto now = coversIn(k);
      for (int64_t i = 1; i <= n; ++i) {
        auto before = coversIn(k - sign * i);
        if (before.size() != now.size())
          return false;
        for (auto [a, b] : llvm::zip(before, now))
          if (a->loop != b->loop || a->anchor != b->anchor || a->lo != b->lo ||
              a->hi != b->hi || a->period != b->period ||
              !samePeriods(*a, *b) ||
              llvm::ArrayRef(a->outer).drop_front(depth + 1) !=
                  llvm::ArrayRef(b->outer).drop_front(depth + 1))
            return false;
      }
      return true;
    }

    /// For the lanes of one kind before (`was`) and after (`is`) an
    /// iteration that took some of them: the lanes it did not take that are
    /// later (forward; earlier, backward) than every lane it gave a time
    /// to, in `apart`, if the others moved by `shift`, as a multiset when
    /// `byTime`, else lane by lane.
    bool apartFrom(llvm::ArrayRef<int64_t> was, llvm::ArrayRef<int64_t> is,
                   llvm::SmallVectorImpl<size_t> &apart, bool byTime,
                   int64_t shift) const {
      apart.clear();
      std::optional<int64_t> edge;
      for (auto [before, after] : llvm::zip(was, is))
        if (before != after && (!edge || sign * (after - *edge) > 0))
          edge = after;
      if (!edge)
        return false;
      llvm::SmallVector<int64_t, 8> from, to;
      for (auto [i, before, after] : llvm::enumerate(was, is)) {
        if (before == after && sign * (after - *edge) > 0) {
          apart.push_back(i);
          continue;
        }
        if (!byTime && after != before + shift)
          return false;
        from.push_back(before);
        to.push_back(after);
      }
      if (byTime) {
        llvm::sort(from);
        llvm::sort(to);
        for (auto [x, y] : llvm::zip(from, to))
          if (y != x + shift)
            return false;
      }
      return true;
    }

    /// The first case of the class comment: the lanes apart and the others
    /// moved by the shift.
    bool movedApart() {
      drivesApart.clear();
      readoutsApart.clear();
      if (filled < 2)
        return false;
      const Step &before = step(1), &now = step(0);
      if (!now.logged || !before.ok || !before.bounded || !now.bounded ||
          before.key != now.key)
        return false;
      bool byTime = s.lanesByTime();
      auto kind = [&](const Lanes &was, const Lanes &is, int64_t wasUses,
                      int64_t isUses, llvm::SmallVectorImpl<size_t> &apart,
                      int64_t shift) {
        apart.clear();
        return wasUses == isUses ||
               apartFrom(was[0], is[0], apart, byTime, shift);
      };
      return kind(before.lanes.drives, now.lanes.drives,
                  before.lanes.driveUses[0], now.lanes.driveUses[0],
                  drivesApart, moves.drives) &&
             kind(before.lanes.readouts, now.lanes.readouts,
                  before.lanes.readoutUses[0], now.lanes.readoutUses[0],
                  readoutsApart, moves.readouts) &&
             opsMoved(1);
    }

    /// How many iterations after the last one the loop plays as it did,
    /// moved, giving no lane a time as late (forward; early, backward) as
    /// those apart: every lane it took was at most as late as the latest it
    /// left.
    int64_t reached() const {
      int64_t most = std::numeric_limits<int64_t>::max();
      auto kind = [&](llvm::ArrayRef<int64_t> lanes,
                      llvm::ArrayRef<size_t> apart, int64_t shift) {
        if (apart.empty() || shift == 0)
          return;
        std::optional<int64_t> edge, near;
        for (auto [i, lane] : llvm::enumerate(lanes)) {
          bool isApart = llvm::is_contained(apart, i);
          std::optional<int64_t> &bound = isApart ? near : edge;
          if (!bound || (isApart ? sign * (lane - *bound) < 0
                                 : sign * (lane - *bound) > 0))
            bound = lane;
        }
        int64_t gap = sign * (*near - *edge) - 1;
        most = std::min(most, gap < 0 ? 0 : gap / (sign * shift));
      };
      kind(s.driveLanes[0], drivesApart, moves.drives);
      kind(s.readoutLanes[0], readoutsApart, moves.readouts);
      return most;
    }

    /// The second case of the class comment.
    bool repeats() {
      int64_t most = std::min(longest, filled - 1);
      const Step &now = step(0);
      for (int64_t n = 2; n <= most && n <= now.run; ++n) {
        const Step &first = step(n);
        if (!first.bounded)
          continue;
        // The sums of the lanes taken move as they do.
        if ((first.lanes.driveUses[0] != now.lanes.driveUses[0] &&
             now.driveSum - first.driveSum !=
                 static_cast<int64_t>(s.driveLanes[0].size()) * n *
                     moves.drives) ||
            (first.lanes.readoutUses[0] != now.lanes.readoutUses[0] &&
             now.readoutSum - first.readoutSum !=
                 static_cast<int64_t>(s.readoutLanes[0].size()) * n *
                     moves.readouts))
          continue;
        if (s.lanesMovedBy(first.lanes, n * moves.drives, n * moves.readouts) &&
            opsMoved(n)) {
          cycle = n;
          return true;
        }
      }
      return false;
    }

    /// Move the lanes of one kind as `copies` iterations that took lanes at
    /// `taken` (the last iteration's, in order) moved by `shift` do. Lanes
    /// in `apart` stay where the iterations do not reach them; else
    /// (`closed`) they take the latest of themselves and of the lanes the
    /// iterations gave up last.
    void moveApart(llvm::SmallVectorImpl<int64_t> &lanes,
                   llvm::ArrayRef<size_t> apart, llvm::ArrayRef<int64_t> taken,
                   int64_t copies, int64_t shift) const {
      if (!closed) {
        for (auto [i, lane] : llvm::enumerate(lanes))
          if (!llvm::is_contained(apart, i))
            lane += copies * shift;
        return;
      }
      llvm::SmallVector<int64_t, 8> latest;
      for (size_t i : apart)
        latest.push_back(lanes[i]);
      for (int64_t copy = copies;
           copy >= 1 && !taken.empty() && latest.size() < 2 * apart.size();
           --copy)
        for (int64_t lane : llvm::reverse(taken)) {
          if (latest.size() == 2 * apart.size())
            break;
          latest.push_back(lane + copy * shift);
        }
      if (sign > 0)
        llvm::sort(latest, std::greater<int64_t>());
      else
        llvm::sort(latest);
      size_t at = 0;
      for (auto [i, lane] : llvm::enumerate(lanes)) {
        if (at < apart.size() && apart[at] == i)
          lane = latest[at++];
        else
          lane += copies * shift;
      }
    }

    Scheduler &s;
    mlir::scf::ForOp loop;
    int64_t count, sign;
    /// The iterations of the loops around this run.
    llvm::SmallVector<int64_t, 2> outer;
    bool on = false, took = false, closed = false;
    int64_t skipBy = 0;
    /// The most iterations the lanes may repeat after.
    int64_t longest = 2;
    /// The iterations the lanes repeat after, or 0 for the first case.
    int64_t cycle = 0;
    Moves moves;
    llvm::SmallVector<mlir::Operation *> ops;
    /// The last iterations noted, a ring: `next` follows the last one, and
    /// `filled` of them are of this run, one after another.
    llvm::SmallVector<Step, 0> steps;
    size_t next = 0;
    int64_t filled = 0;
    LanePicks picks;
    llvm::SmallVector<size_t, 2> drivesApart, readoutsApart;
  };

  // ── Loops whose lines move by periods of their own ────────────────────

  /// Whether a loop's carried values may move by a period per group (see
  /// LoopGroups): without timelines, whose state the groups share, or
  /// resources on several controllers.
  bool groupsApply() const {
    return !lanesBound && !isolated() && !(resources.enabled && several()) &&
           !holding();
  }

  const LoopGroups *groupsOf(mlir::scf::ForOp loop) {
    auto [it, inserted] = groupCache.try_emplace(loop, nullptr);
    if (inserted)
      it->second = loopGroups(loop);
    return it->second.get();
  }

  using GroupShifts = llvm::SmallVector<std::optional<int64_t>, 4>;

  /// slotsHoldSteadyState for a loop whose groups move by `shifts`, the
  /// way the pass runs (`sign`): each submission by its group's.
  bool slotsHoldGroups(mlir::scf::ForOp loop, const LoopGroups &groups,
                       const GroupShifts &shifts, int64_t sign) {
    if (!classical)
      return true;
    std::map<LinkKey, std::pair<int64_t, int64_t>> perLink;
    bool ok = true;
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      mlir::Operation *op = submit;
      int64_t period =
          groups.ofOp.contains(op)
              ? sign * shifts[groups.groupOf(op, sign < 0)].value_or(0)
              : 0;
      std::optional<int64_t> held;
      if (period > 0)
        held = inFlight(op, loop, period);
      auto &[need, slots] = perLink[linkOf(op)];
      if (!held) {
        ok = false;
        return;
      }
      if (__builtin_add_overflow(need, *held, &need)) {
        ok = false;
        return;
      }
      slots = classical::intAttr(op, "classical.slots", 64);
    });
    for (auto &[device, entry] : perLink)
      ok = ok && entry.first <= entry.second;
    return ok;
  }

  /// The shift of the groups of a loop's submissions `ops`, the way the pass
  /// runs, each that of the op of the body it is in; none when they differ.
  static std::optional<int64_t>
  requestsShift(mlir::scf::ForOp loop, const LoopGroups &groups,
                const GroupShifts &shifts,
                llvm::ArrayRef<mlir::Operation *> ops, bool backward) {
    std::optional<int64_t> shift;
    for (mlir::Operation *op : ops) {
      mlir::Operation *timed = op;
      while (timed != loop && !groups.ofOp.contains(timed))
        timed = timed->getParentOp();
      if (timed == loop)
        return std::nullopt;
      int64_t moved = shifts[groups.groupOf(timed, backward)].value_or(0);
      if (shift && *shift != moved)
        return std::nullopt;
      shift = moved;
    }
    return shift;
  }

  /// Under resources, whether an iteration of a loop whose groups moved by
  /// `shifts` moved its lanes from `before` by as much: each kind's by the
  /// one shift of the groups its ops are in. True without resources.
  bool groupLanesMoved(const LaneState &before, const LoopGroups &groups,
                       const GroupShifts &shifts) {
    if (!resources.enabled)
      return true;
    auto shiftOf =
        [&](const llvm::SmallBitVector &kind) -> std::optional<int64_t> {
      std::optional<int64_t> shift;
      for (unsigned group : kind.set_bits()) {
        int64_t moved = shifts[group].value_or(0);
        if (shift && *shift != moved)
          return std::nullopt;
        shift = moved;
      }
      return shift.value_or(0);
    };
    std::optional<int64_t> drives = shiftOf(groups.driveGroups),
                           readouts = shiftOf(groups.readoutGroups);
    return drives && readouts && lanesMovedBy(before, *drives, *readouts);
  }

  /// Move the lanes as the iterations a loop's groups skip `distances`
  /// do (see groupLanesMoved).
  void moveGroupLanes(const LaneState &before, const LoopGroups &groups,
                      llvm::ArrayRef<int64_t> distances) {
    if (!resources.enabled)
      return;
    auto distanceOf = [&](const llvm::SmallBitVector &kind) -> int64_t {
      int first = kind.find_first();
      return first < 0 ? 0 : distances[static_cast<unsigned>(first)];
    };
    moveLanesBy(before, distanceOf(groups.driveGroups),
                distanceOf(groups.readoutGroups));
  }

  /// How far each group of carried values moved from `before` to `after`
  /// (none: nothing timed in it moved), or none when a group's values moved
  /// by different amounts; `backward`, by their groups backward.
  static std::optional<GroupShifts> groupShifts(const LoopGroups &groups,
                                                llvm::ArrayRef<int64_t> before,
                                                llvm::ArrayRef<int64_t> after,
                                                llvm::ArrayRef<Carried> kinds,
                                                bool backward) {
    GroupShifts shifts(groups.count);
    for (auto [group, b, a, kind] :
         llvm::zip(groups.argGroups(backward), before, after, kinds)) {
      if (kind == Carried::Untimed || (kind == Carried::Tone && a == b))
        continue;
      std::optional<int64_t> &shift = shifts[group];
      if (shift && *shift != a - b)
        return std::nullopt;
      shift = a - b;
    }
    return shifts;
  }

  /// Each group's shift times `copies`, or none past 2**63.
  static std::optional<llvm::SmallVector<int64_t, 4>>
  groupDistances(const GroupShifts &shifts, int64_t copies) {
    llvm::SmallVector<int64_t, 4> distances(shifts.size(), 0);
    for (auto [shift, distance] : llvm::zip(shifts, distances))
      if (shift)
        if (__builtin_mul_overflow(copies, *shift, &distance))
          return std::nullopt;
    return distances;
  }

  /// Advance each carried value that moves by its group's distance; none,
  /// leaving `carried` as it was, past 2**63.
  static bool extrapolateGroups(llvm::MutableArrayRef<int64_t> carried,
                                llvm::ArrayRef<int64_t> previous,
                                llvm::ArrayRef<Carried> kinds,
                                const LoopGroups &groups, bool backward,
                                llvm::ArrayRef<int64_t> distances) {
    llvm::SmallVector<int64_t> moved(carried.begin(), carried.end());
    for (auto [time, now, before, kind, group] :
         llvm::zip(moved, carried, previous, kinds, groups.argGroups(backward)))
      if (kind == Carried::Timed || (kind == Carried::Tone && now != before))
        if (__builtin_add_overflow(now, distances[group], &time))
          return false;
    llvm::copy(moved, carried.begin());
    return true;
  }

  /// Each timed op's period under `shifts` of its group, times `sign`: 0
  /// in a group none of whose values move (a tone nothing plays on).
  static std::shared_ptr<const llvm::DenseMap<mlir::Operation *, int64_t>>
  opPeriods(const LoopGroups &groups, const GroupShifts &shifts, int64_t sign) {
    auto periods =
        std::make_shared<llvm::DenseMap<mlir::Operation *, int64_t>>();
    for (mlir::Operation *op : llvm::make_first_range(groups.ofOp))
      (*periods)[op] = sign * shifts[groups.groupOf(op, sign < 0)].value_or(0);
    return periods;
  }

  /// Whether each group's shift is whole steps of every grid and points
  /// the way the pass runs (`sign` 1 forward, -1 backward), and some group
  /// moves.
  bool groupsMove(const GroupShifts &shifts, int64_t sign) const {
    bool moves = false;
    for (const std::optional<int64_t> &shift : shifts) {
      if (!shift)
        continue;
      if (sign * *shift < 0)
        return false;
      if constexpr (Gridded)
        if (*shift % grid.lcm() != 0)
          return false;
      moves = true;
    }
    return moves;
  }

  /// Whether, backward without resources, `branch` holds nothing in its
  /// condition's group back: it relates the groups only through answers
  /// (no branch on a measurement), and each lazy request among them ended
  /// forward, in this iteration, right where the eager one after it on its
  /// link starts (see LoopGroups::linkPairs). An eager request keeps its
  /// forward time, and so does such a lazy one, which it bounds; no wait is
  /// placed before its forward time, after its answer is due.
  bool answersHoldNothing(const LoopGroups &groups,
                          std::pair<unsigned, unsigned> branch) const {
    if (llvm::is_contained(groups.strictBranches, branch))
      return false;
    auto it = llvm::find_if(groups.lazyBranches, [&](const auto &entry) {
      return entry.first == branch;
    });
    if (it == groups.lazyBranches.end())
      return false;
    for (mlir::Operation *lazy : it->second) {
      auto pair = llvm::find_if(groups.linkPairs,
                                [&](const auto &p) { return p.lazy == lazy; });
      if (pair == groups.linkPairs.end() ||
          forwardStartOf(lazy) + durationOf(lazy) !=
              forwardStartOf(pair->eager))
        return false;
    }
    return true;
  }

  /// Whether groupsMove, each group a branch holds back moves with the
  /// group holding it back or ahead of it, with every op in the body where
  /// its group puts it (backward, each wait with what its answers decide):
  /// then what holds it back stays behind.
  bool groupsSteady(const LoopGroups &groups, const GroupShifts &shifts,
                    int64_t sign) const {
    if (!groupsMove(shifts, sign) || (sign < 0 && groups.mixedWaits))
      return false;
    bool ahead = false;
    for (auto [condition, decided] : groups.branches) {
      if (sign < 0 && !resources.enabled && !lanesBound &&
          answersHoldNothing(groups, std::pair(condition, decided)))
        continue;
      auto [holding, held] = sign > 0 ? std::pair(condition, decided)
                                      : std::pair(decided, condition);
      int64_t from = shifts[holding].value_or(0);
      int64_t to = shifts[held].value_or(0);
      if (to == from)
        continue;
      if (sign * to < sign * from)
        return false;
      ahead = true;
    }
    if (!ahead)
      return true;
    // Each op's last two samples, this iteration's and the one before.
    for (mlir::Operation *op : llvm::make_first_range(groups.ofOp)) {
      unsigned group = groups.groupOf(op, sign < 0);
      auto found = samples.find(op);
      if (found == samples.end() || found->second.size() < 2)
        return false;
      const Sample &now = found->second.back();
      const Sample &before = found->second[found->second.size() - 2];
      if (llvm::ArrayRef(now.iteration) != llvm::ArrayRef(iteration) ||
          llvm::ArrayRef(before.iteration).drop_back() !=
              llvm::ArrayRef(iteration).drop_back() ||
          before.iteration.back() != iteration.back() - sign ||
          now.start - before.start != shifts[group].value_or(0))
        return false;
    }
    return true;
  }

  mlir::LogicalResult tripCount(mlir::scf::ForOp loop, int64_t &count) {
    auto trips = pulse::constantTripCount(loop);
    if (!trips)
      return loop.emitError("cannot schedule a loop with non-constant bounds");
    count = *trips;
    return mlir::success();
  }

  /// All carried timed values advanced by the same `period`; tones may also
  /// stay where they were.
  static bool uniformShift(llvm::ArrayRef<int64_t> before,
                           llvm::ArrayRef<int64_t> after,
                           llvm::ArrayRef<Carried> kinds, int64_t &period) {
    std::optional<int64_t> shift;
    for (auto [b, a, kind] : llvm::zip(before, after, kinds)) {
      if (kind == Carried::Untimed || (kind == Carried::Tone && a == b))
        continue;
      if (shift && *shift != a - b)
        return false;
      shift = a - b;
    }
    if (!shift)
      return false;
    period = *shift;
    return true;
  }

  /// The most iterations of `loop` regrouping may run in each now.
  int64_t mostRegrouped(mlir::scf::ForOp loop) const {
    return std::min(kMostRegrouped,
                    kMostRegroupedInAll /
                        std::max<int64_t>(1, regroupedBy.lookup(loop)));
  }

  /// Note the carried times after an iteration of `loop` that did not
  /// repeat the one before, with `left` iterations after it; if the last
  /// two runs of some fewer than kMostRegrouped iterations, from two on,
  /// each moved every timed value by one shift the way the pass runs
  /// (`sign`), and more than two such runs are left, ask for the loop to be
  /// regrouped by that many.
  void noteRecent(RecentStates &recent, mlir::scf::ForOp loop,
                  llvm::ArrayRef<int64_t> state, llvm::ArrayRef<Carried> kinds,
                  int64_t left, int64_t sign) {
    recent.push(state);
    const LoopGroups *groups = nullptr;
    bool grouped = false;
    int64_t most = mostRegrouped(loop);
    for (int64_t n = 2; n <= most && 2 * n < recent.pushed; ++n) {
      if (left < 2 * n)
        return;
      int64_t first, second;
      if (uniformShift(recent.before(2 * n), recent.before(n), kinds, first) &&
          uniformShift(recent.before(n), state, kinds, second) &&
          first == second && sign * first > 0 &&
          (!Gridded || first % grid.lcm() == 0)) {
        regroup = Regroup{loop, n};
        return;
      }
      // Or each group by a shift of its own.
      if (!grouped) {
        grouped = true;
        groups = groupsApply() ? groupsOf(loop) : nullptr;
      }
      if (!groups)
        continue;
      auto earlier = groupShifts(*groups, recent.before(2 * n),
                                 recent.before(n), kinds, sign < 0);
      auto later =
          groupShifts(*groups, recent.before(n), state, kinds, sign < 0);
      if (earlier && later && *earlier == *later &&
          groupsMove(*earlier, sign)) {
        regroup = Regroup{loop, n};
        return;
      }
    }
  }

  /// Note the lanes before an iteration of `loop` (`history`, the last
  /// kMostRegrouped of them, this one's last), after which its carried
  /// times moved by `shift` as they did in the one before but its lanes did
  /// not; if they moved by as much each over the last run of some fewer
  /// than kMostRegrouped iterations, from two on, and more than two such
  /// runs are left (`left` iterations after this one), ask for the loop to
  /// be regrouped by that many.
  void noteRecentLanes(llvm::ArrayRef<LaneState> history, mlir::scf::ForOp loop,
                       int64_t shift, int64_t left) {
    int64_t most =
        std::min(mostRegrouped(loop), static_cast<int64_t>(history.size()));
    for (int64_t n = 2; n <= most; ++n) {
      if (left < 2 * n)
        return;
      if (lanesMoved(history[history.size() - n], n * shift)) {
        regroup = Regroup{loop, n};
        return;
      }
    }
  }

  /// Advance the carried values that move with the loop by `distance`.
  static void extrapolate(llvm::MutableArrayRef<int64_t> carried,
                          llvm::ArrayRef<int64_t> previous,
                          llvm::ArrayRef<Carried> kinds, int64_t distance) {
    for (auto [time, before, kind] : llvm::zip(carried, previous, kinds))
      if (kind == Carried::Timed || (kind == Carried::Tone && time != before))
        time += distance;
  }

  /// Move the carried values as the iterations `cycle` skips do, the way
  /// the pass runs (`backward`); false, leaving them, past 2**63.
  static bool skipCarried(const LaneCycle &cycle,
                          llvm::MutableArrayRef<int64_t> carried,
                          llvm::ArrayRef<int64_t> previous,
                          llvm::ArrayRef<Carried> kinds, int64_t shift,
                          bool backward) {
    if (const LoopGroups *groups = cycle.groups()) {
      auto distances = cycle.groupDistances(cycle.skip());
      return distances && extrapolateGroups(carried, previous, kinds, *groups,
                                            backward, *distances);
    }
    extrapolate(carried, previous, kinds, cycle.skip() * shift);
    return true;
  }

  static Carried carriedKind(mlir::Type type, bool timed) {
    // A link, like a tone, keeps its time through a body that sends nothing.
    if (mlir::isa<pulse::ToneType, classical::LinkType>(type))
      return Carried::Tone;
    return timed ? Carried::Timed : Carried::Untimed;
  }

  // ── Grids ─────────────────────────────────────────────────────────────

  /// The grid every line of `sync` is on.
  int64_t syncGrid(pulse::SyncOp sync) {
    if (!grid.perLine())
      return grid.lcm();
    int64_t step = 1;
    for (mlir::Value line : sync.getLines())
      step = std::lcm(step, grid.of(line, lines));
    return step;
  }

  // ── Forward (ASAP) ────────────────────────────────────────────────────

  int64_t ready(mlir::Value value) const { return readyAt.lookup(value); }

  /// On several controllers, the measurements a branch condition comes
  /// from, through answers, each with the controller that reads it out; and
  /// per such controller, the latest its outcomes may be ready by the
  /// branches on the condition the backward pass has passed (late() reads it
  /// with each measurement's own bound): a branch bounds the condition once,
  /// not every measurement.
  struct AnswerGroup {
    mlir::Value condition;
    llvm::SmallVector<std::pair<mlir::Value, int64_t>, 4> measured;
    llvm::SmallVector<int64_t, 2> readers;
    llvm::SmallVector<int64_t, 2> latest;
  };

  AnswerGroup &answerGroup(mlir::Value condition) {
    auto [it, inserted] = groupOf.try_emplace(
        condition, static_cast<unsigned>(answerGroups.size()));
    if (!inserted)
      return answerGroups[it->second];
    AnswerGroup &group = answerGroups.emplace_back();
    group.condition = condition;
    for (mlir::Value measurement : conditionMeasurements(condition)) {
      int64_t reader = readerOf(measurement);
      auto slot = static_cast<unsigned>(llvm::find(group.readers, reader) -
                                        group.readers.begin());
      if (slot == group.readers.size()) {
        group.readers.push_back(reader);
        group.latest.push_back(kNever);
      }
      group.measured.push_back({measurement, reader});
      memberOf[measurement].push_back({it->second, slot});
    }
    return group;
  }

  /// `time`, or the latest the groups of `measurement` let it be ready
  /// (only measurements are in groups).
  int64_t withGroups(mlir::Value measurement, int64_t time) const {
    if (memberOf.empty())
      return time;
    auto it = memberOf.find(measurement);
    if (it == memberOf.end())
      return time;
    for (auto [group, slot] : it->second)
      time = std::min(time, answerGroups[group].latest[slot]);
    return time;
  }

  /// Bound the measurements of the groups on a condition inside `loop` by
  /// what its branches asked, before the loop's next iteration forgets the
  /// bounds of its body's values.
  void settleGroups(mlir::Operation *loop) {
    for (AnswerGroup &group : answerGroups) {
      mlir::Value condition = group.condition;
      mlir::Operation *at = condition.getDefiningOp();
      if (!at)
        at = condition.getParentBlock()->getParentOp();
      if (!loop->isAncestor(at))
        continue;
      for (auto [slot, latest] : llvm::enumerate(group.latest)) {
        if (latest == kNever)
          continue;
        for (auto [measurement, reader] : group.measured)
          if (reader == group.readers[slot])
            bound(measurement, latest);
        latest = kNever;
      }
    }
  }

  /// On several controllers, when the latest measurement a branch condition
  /// comes from (through answers) is ready, and when the outcomes reach each
  /// controller: what forwardIf takes from them, found once per condition
  /// while none of them is placed again (forwardLoop forgets them at each
  /// iteration).
  struct ConditionReady {
    bool any = false;
    int64_t ready = 0;
    /// Each controller that read one of the measurements, and when the
    /// latest of them is ready.
    llvm::SmallVector<std::pair<int64_t, int64_t>, 2> readers;
  };

  const ConditionReady &conditionReadyOf(mlir::Value condition) {
    auto [it, inserted] = conditionReady.try_emplace(condition);
    ConditionReady &at = it->second;
    if (!inserted)
      return at;
    for (auto [measurement, reader] : answerGroup(condition).measured) {
      int64_t time = ready(measurement);
      if (!at.any) {
        at.any = true;
        at.ready = time;
      }
      at.ready = std::max(at.ready, time);
      auto known = at.readers.begin();
      while (known != at.readers.end() && known->first != reader)
        ++known;
      if (known == at.readers.end())
        at.readers.push_back({reader, time});
      else
        known->second = std::max(known->second, time);
    }
    return at;
  }

  /// When every operand of `op` is available: data from answers included.
  /// A submission with its own order on a link (`ownOrder`) does not wait
  /// for the link it is threaded through.
  int64_t operandsReady(mlir::Operation *op) const {
    int64_t time = 0;
    bool own = ownOrder(op);
    for (mlir::Value operand : op->getOperands())
      if (!own || !mlir::isa<classical::LinkType>(operand.getType()))
        time = std::max(time, ready(operand));
    return time;
  }

  /// A link's request slots: a heap of the times they become free (forward,
  /// min-heap) or must be free from (backward, max-heap). A link's slots
  /// bound every policy: they are hardware. Also the start of the request
  /// placed on it last in the pass's order, the latest forward and the
  /// earliest backward: every request placed after it starts no earlier
  /// (forward) or ends its submission by it (backward), so only the slots
  /// held past it (heldPast) can delay one.
  struct LinkSlots {
    llvm::SmallVector<int64_t> free;
    std::optional<int64_t> last;
  };

  LinkSlots &slotsOf(mlir::Operation *submit) {
    auto [it, inserted] = linkLanes.try_emplace(linkOf(submit));
    if (inserted)
      it->second.free.assign(
          static_cast<size_t>(std::max<int64_t>(
              1, classical::intAttr(submit, "classical.slots", 64))),
          laneReset);
    return it->second;
  }

  llvm::SmallVector<int64_t> &lanesOf(mlir::Operation *submit) {
    return slotsOf(submit).free;
  }

  /// The links a loop's requests use, each with its submissions.
  using LoopLinks = llvm::SmallVector<
      std::pair<LinkKey, llvm::SmallVector<mlir::Operation *, 2>>, 1>;

  LoopLinks linksOf(mlir::scf::ForOp loop) {
    LoopLinks links;
    if (!classical)
      return links;
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      LinkKey key = linkOf(submit);
      auto it = llvm::find_if(
          links, [&](const auto &entry) { return entry.first == key; });
      if (it == links.end())
        it =
            &links.emplace_back(key, llvm::SmallVector<mlir::Operation *, 2>{});
      it->second.push_back(submit);
    });
    return links;
  }

  /// The largest budget of a request on a link, which bounds how long past
  /// its submission one holds its slot.
  int64_t linkBudget(const LinkKey &key) {
    if (linkBudgets.empty())
      region->walk([&](classical::SubmitOp submit) {
        int64_t &most = linkBudgets[linkOf(submit)];
        most = std::max(most, budgetOf(submit));
      });
    return linkBudgets.lookup(key);
  }

  /// Whether a slot free at (forward) or from (backward) `free` can delay a
  /// request placed after the link's last: forward, one free past its
  /// start; backward, one to be free before its start plus the largest
  /// budget, where every request placed after it has its slot free again.
  bool heldPast(int64_t free, const LinkSlots &slots, int64_t budget,
                bool backward) const {
    if (!slots.last)
      return true;
    return backward ? free < *slots.last + budget : free > *slots.last;
  }

  /// Per link of a loop, its last request and the slots held past it, in
  /// order.
  struct HeldSlots {
    std::optional<int64_t> last;
    llvm::SmallVector<int64_t, 4> held;
  };

  llvm::SmallVector<HeldSlots, 1> heldSlots(const LoopLinks &links,
                                            bool backward) {
    llvm::SmallVector<HeldSlots, 1> state(links.size());
    for (auto [link, slots] : llvm::zip(links, state)) {
      auto it = linkLanes.find(link.first);
      if (it == linkLanes.end())
        continue;
      int64_t budget = backward ? linkBudget(link.first) : 0;
      slots.last = it->second.last;
      for (int64_t free : it->second.free)
        if (heldPast(free, it->second, budget, backward))
          slots.held.push_back(free);
      llvm::sort(slots.held);
    }
    return state;
  }

  /// Whether an iteration moved each link's last request and held slots
  /// from `before` by the shift its requests move by (`shiftOf`; none when
  /// they move by several), so that the iterations after it hold the slots
  /// as it did, moved as they are.
  bool slotsMoved(const LoopLinks &links, llvm::ArrayRef<HeldSlots> before,
                  bool backward,
                  llvm::function_ref<
                      std::optional<int64_t>(llvm::ArrayRef<mlir::Operation *>)>
                      shiftOf) {
    if (before.size() != links.size())
      return false;
    auto now = heldSlots(links, backward);
    for (auto [link, was, is] : llvm::zip(links, before, now)) {
      std::optional<int64_t> shift = shiftOf(link.second);
      if (!shift || !was.last || !is.last || *is.last != *was.last + *shift ||
          is.held.size() != was.held.size())
        return false;
      for (auto [x, y] : llvm::zip(was.held, is.held))
        if (y != x + *shift)
          return false;
    }
    return true;
  }

  /// Move each link's last request and held slots by `distanceOf` its
  /// requests, as the iterations not simulated do (see slotsMoved).
  void moveSlots(const LoopLinks &links, bool backward,
                 llvm::function_ref<int64_t(llvm::ArrayRef<mlir::Operation *>)>
                     distanceOf) {
    for (const auto &[key, ops] : links) {
      auto it = linkLanes.find(key);
      if (it == linkLanes.end() || !it->second.last)
        continue;
      int64_t distance = distanceOf(ops);
      int64_t budget = backward ? linkBudget(key) : 0;
      // Only held slots move, the way the heap orders them: it stays one.
      for (int64_t &free : it->second.free)
        if (heldPast(free, it->second, budget, backward))
          free += distance;
      *it->second.last += distance;
    }
  }

  /// Whether a loop whose iterations shift by `period` keeps every link's
  /// requests within its slots in steady state, so that extrapolating it
  /// cannot exceed them: per link, the submissions of one iteration stay in
  /// flight for ceil(busy / period) iterations each.
  bool slotsHoldSteadyState(mlir::scf::ForOp loop, int64_t period) {
    if (!classical)
      return true;
    std::map<LinkKey, std::pair<int64_t, int64_t>> perLink;
    bool ok = true;
    loop.getBody()->walk([&](classical::SubmitOp submit) {
      mlir::Operation *op = submit;
      std::optional<int64_t> held;
      if (period > 0)
        held = inFlight(op, loop, period);
      auto &[need, slots] = perLink[linkOf(op)];
      if (!held) {
        ok = false;
        return;
      }
      if (__builtin_add_overflow(need, *held, &need)) {
        ok = false;
        return;
      }
      slots = classical::intAttr(op, "classical.slots", 64);
    });
    for (auto &[device, entry] : perLink)
      ok = ok && entry.first <= entry.second;
    return ok;
  }

  /// Take the earliest free slot of a link for a request starting no earlier
  /// than `start`; it is busy until the answer is due.
  int64_t takeSlot(mlir::Operation *submit, int64_t start, int64_t busy) {
    LinkSlots &slots = slotsOf(submit);
    auto &heap = slots.free;
    auto later = std::greater<int64_t>();
    std::pop_heap(heap.begin(), heap.end(), later);
    if (heap.back() > start) {
      start = heap.back();
      lanesBound = true;
    }
    heap.back() = start + busy;
    std::push_heap(heap.begin(), heap.end(), later);
    slots.last = std::max(slots.last.value_or(start), start);
    return start;
  }

  /// The backward mirror: the latest a request ending its slot by the slot's
  /// bound may start, for one that is busy for `busy`.
  int64_t takeSlotBackward(mlir::Operation *submit, int64_t start, int64_t busy,
                           bool pinned) {
    LinkSlots &slots = slotsOf(submit);
    auto &heap = slots.free;
    std::pop_heap(heap.begin(), heap.end());
    if (heap.back() - busy < start) {
      // A slot no later request took is free however long this one holds
      // it, past the end of the program too.
      if (pinned) {
        if (heap.back() < laneReset)
          slotsOverrun = true;
      } else {
        start = heap.back() - busy;
        lanesBound = true;
      }
    }
    heap.back() = start;
    std::push_heap(heap.begin(), heap.end());
    slots.last = std::min(slots.last.value_or(start), start);
    return start;
  }

  /// One execution of a submission in a scheduled block, in the block's
  /// local time: when it starts and how long it holds its link's slot.
  struct BlockRequest {
    int64_t start, busy;
    mlir::Operation *op;
  };

  static void collectRequests(mlir::Block &body,
                              llvm::SmallVectorImpl<int64_t> &iteration,
                              llvm::SmallVectorImpl<BlockRequest> &found) {
    for (mlir::Operation &op : body) {
      if (mlir::isa<classical::SubmitOp>(op)) {
        if (auto start = pulse::scheduledStart(&op, iteration))
          found.push_back({*start, durationOf(&op) + budgetOf(&op), &op});
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        int64_t trips = pulse::constantTripCount(loop).value_or(0);
        iteration.push_back(0);
        for (int64_t k = 0; k < trips; ++k) {
          iteration.back() = k;
          collectRequests(*loop.getBody(), iteration, found);
        }
        iteration.pop_back();
      } else {
        // Both arms of a branch hold their slots, as the check counts them.
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            collectRequests(nested, iteration, found);
      }
    }
  }

  /// The requests of `block`, ordered by local start (forward) or by local
  /// end, latest first (backward).
  llvm::ArrayRef<BlockRequest> requestsOf(pulse::ScheduleBlockOp block,
                                          bool backward) {
    auto [it, inserted] = blockRequests.try_emplace(block.getOperation());
    auto &[forwardOrder, backwardOrder] = it->second;
    if (inserted) {
      llvm::SmallVector<int64_t, 2> iteration;
      collectRequests(block.getBody().front(), iteration, forwardOrder);
      llvm::stable_sort(forwardOrder, [](const auto &a, const auto &b) {
        return a.start < b.start;
      });
      backwardOrder = forwardOrder;
      llvm::stable_sort(backwardOrder, [](const auto &a, const auto &b) {
        return a.start + a.busy > b.start + b.busy;
      });
    }
    return backward ? backwardOrder : forwardOrder;
  }

  /// Move a block placed at `start` until the requests in it fit the slots
  /// its links have left, later (forward) or earlier (backward), and take
  /// those slots. The block's own schedule fits empty links, so moving it
  /// past the requests around it always succeeds.
  int64_t takeBlockSlots(pulse::ScheduleBlockOp block, int64_t start,
                         bool backward) {
    llvm::ArrayRef<BlockRequest> requests = requestsOf(block, backward);
    if (requests.empty())
      return start;
    while (true) {
      std::map<LinkKey, llvm::SmallVector<int64_t>> trial;
      int64_t shift = 0;
      for (const BlockRequest &request : requests) {
        LinkKey key = linkOf(request.op);
        auto it = trial.find(key);
        if (it == trial.end())
          it = trial.try_emplace(key, lanesOf(request.op)).first;
        auto &heap = it->second;
        int64_t at = start + request.start;
        if (backward) {
          std::pop_heap(heap.begin(), heap.end());
          shift = at + request.busy - heap.back();
          heap.back() = at;
          std::push_heap(heap.begin(), heap.end());
        } else {
          auto later = std::greater<int64_t>();
          std::pop_heap(heap.begin(), heap.end(), later);
          shift = heap.back() - at;
          heap.back() = at + request.busy;
          std::push_heap(heap.begin(), heap.end(), later);
        }
        if (shift > 0)
          break;
      }
      if (shift <= 0) {
        for (auto &[key, heap] : trial)
          linkLanes[key].free = std::move(heap);
        for (const BlockRequest &request : requests) {
          std::optional<int64_t> &last = linkLanes[linkOf(request.op)].last;
          int64_t at = start + request.start;
          last = backward ? std::min(last.value_or(at), at)
                          : std::max(last.value_or(at), at);
        }
        return start;
      }
      start += backward ? -shift : shift;
      if constexpr (Gridded)
        start = backward ? pulse::LineGrid::roundDown(start, grid.lcm())
                         : pulse::LineGrid::roundUp(start, grid.lcm());
      lanesBound = true;
    }
  }

  void forwardSubmit(classical::SubmitOp submit) {
    mlir::Operation *op = submit;
    int64_t t = timelineOf(op);
    bool own = ownOrder(op);
    int64_t link = own ? linkReady[linkOf(op)] : ready(submit.getLink());
    int64_t start =
        std::max({link, floorOf(t), operandsReady(op), armFloorOf(kAnyOwner)});
    if (blocksOnCalls)
      start = std::max(start, endOf(t));
    llvm::ArrayRef<int64_t> held;
    if (holding()) {
      held = heldBy(submit);
      for (int64_t c : held)
        start = std::max(start, heldEnds[c]);
    }
    int64_t duration = durationOf(op), budget = budgetOf(op);
    start = takeSlot(op, start, duration + budget);
    record(op, start);
    readyAt[submit.getUpdatedLink()] = start + duration;
    if (own)
      linkReady[linkOf(op)] = start + duration;
    readyAt[submit.getFuture()] = start + duration + budget;
    if (blocksOnCalls)
      floors[t] = std::max(floors[t], start + duration + budget);
    for (int64_t c : held)
      heldFloors[c] = std::max(heldFloors[c], start + duration + budget);
    opsEnd = std::max(opsEnd, start + duration);
    noteEnd(start + duration, t);
  }

  void forwardWait(classical::WaitOp wait) {
    int64_t t = timelineOf(wait);
    int64_t start =
        std::max({ready(wait.getFuture()), floorOf(t), armFloorOf(kAnyOwner)});
    // A blocking wait is a barrier: every line of its timeline is done
    // before it.
    if (wait->hasAttr("classical.blocking"))
      start = std::max(start, endOf(t));
    record(wait, start);
    for (mlir::Value value : wait.getValues())
      readyAt[value] = start;
    if (wait->hasAttr("classical.blocking"))
      floors[t] = std::max(floors[t], start);
    opsEnd = std::max(opsEnd, start);
    noteEnd(start, t);
  }

  void place(mlir::Operation *op, int64_t start, int64_t t, int64_t duration) {
    record(op, start);
    int64_t end = start + duration;
    int64_t lineEnd = end;
    if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op)) {
      int64_t latency = several() ? controllers.own(controllerOf(op))
                                  : resources.readoutLatency;
      if (resources.enabled)
        lineEnd += latency;
      if (lineEnd != end) {
        tailNoted = true;
        tailsEnd = std::max(tailsEnd, lineEnd);
      }
      readyAt[readout.getResult()] = end + latency + readoutDelayOf(op);
    }
    readyAt[op->getResult(0)] = lineEnd;
    if (mlir::Value tone = toneResult(op))
      readyAt[tone] = end;
    if (holding())
      noteHeldEnd(op, lineEnd);
    opsEnd = std::max(opsEnd, end);
    noteEnd(lineEnd, t);
  }

  void noteEnd(int64_t end, int64_t t) {
    makespan = std::max(makespan, end);
    iterationEnd = std::max(iterationEnd, end);
    ends[t] = std::max(ends[t], end);
    iterationEnds[t] = std::max(iterationEnds[t], end);
  }

  mlir::LogicalResult forwardBlock(mlir::Block &block) {
    for (mlir::Operation &op : block.without_terminator())
      if (mlir::failed(forwardOp(&op)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult forwardOp(mlir::Operation *op) {
    if (classical) {
      if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
        forwardSubmit(submit);
        return mlir::success();
      }
      if (auto wait = mlir::dyn_cast<classical::WaitOp>(op)) {
        forwardWait(wait);
        return mlir::success();
      }
    }
    if (isTimedOp(op)) {
      mlir::FailureOr<int64_t> duration = placedDuration(op);
      if (mlir::failed(duration))
        return mlir::failure();
      int64_t t = timelineOf(op);
      int64_t start =
          std::max(ready(op->getOperand(0)) + holdOf(op), floorOf(t));
      int64_t owner = several() ? controllerOf(op) : kAnyOwner;
      if (!armFloors.empty())
        start = std::max(start, armFloorOf(owner));
      if (holding())
        start = std::max(start, heldFloorOf(op));
      if (mlir::Value tone = toneOperand(op))
        start = std::max(start, ready(tone));
      if (classical) {
        start = std::max(start, operandsReady(op));
        // An outcome it computes with reaches its controller later than the
        // reader's.
        if (several())
          for (mlir::Value measurement : valueMeasurements(op))
            start = std::max(
                start, ready(measurement) +
                           controllers.extra(readerOf(measurement), owner));
        // A variable wait is a barrier: every line of its timeline is done
        // before it.
        if (op->hasAttr("classical.variable"))
          start = std::max(start, endOf(t));
      }
      int64_t step = 1;
      if constexpr (Gridded)
        step = grid.of(op->getOperand(0), lines);
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        bool isDrive = mlir::isa<pulse::DriveOp>(op);
        auto &lanes = isDrive ? driveLanes[t] : readoutLanes[t];
        ++(isDrive ? driveUses : readoutUses)[t];
        size_t best = 0;
        int64_t free = kNever;
        if (byHeap) {
          best = (isDrive ? driveHeaps : readoutHeaps)[t].best();
          free = lanes[best];
        } else {
          // Another controller's op ends a lane only as surely as the two
          // starts are known.
          auto &owners = isDrive ? driveOwners[t] : readoutOwners[t];
          for (size_t i = 0; i < lanes.size(); ++i) {
            int64_t at = lanes[i] + laneGap(owners[i], owner);
            if (at < free) {
              free = at;
              best = i;
            }
          }
        }
        notePick(isDrive, free, free > start);
        start = std::max(start, free);
        if (step > 1)
          start = pulse::LineGrid::roundUp(start, step);
        setLane(!isDrive, t, best, start + *duration + resources.switchPenalty,
                owner);
      }
      if (step > 1)
        start = pulse::LineGrid::roundUp(start, step);
      place(op, start, t, *duration);
      if (classical && op->hasAttr("classical.variable"))
        floors[t] = std::max(floors[t], start + *duration);
      return mlir::success();
    }
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      forwardScheduleBlock(block);
      return mlir::success();
    }
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(op)) {
      int64_t t = timelineOf(op);
      int64_t time = floorOf(t);
      for (mlir::Value line : sync.getLines()) {
        time = std::max(time, ready(line));
        if constexpr (Gridded)
          if (!armFloors.empty())
            time = std::max(time, armFloorOf(controllerOfLine(line)));
      }
      if constexpr (Gridded)
        time = pulse::LineGrid::roundUp(time, syncGrid(sync));
      for (mlir::Value result : sync.getResults())
        readyAt[result] = time;
      tailNoted = true;
      tailsEnd = std::max(tailsEnd, time);
      noteEnd(time, t);
      return mlir::success();
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return forwardLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return forwardIf(branch);
    // Any other op releases its timed results when its timed operands are;
    // with calls, every result is released when all its operands are. That
    // is wrong for an op that could itself play (Untimed.h).
    if (pulse::untimed(op))
      return pulse::emitUntimed(op);
    // Nor does the pass go into the regions of an op that comes here.
    if (op->getNumRegions())
      return op->emitError("'")
             << op->getName().getStringRef()
             << "' holds ops the schedule passes do not go into; write them "
                "in its place before scheduling";
    if (classical) {
      int64_t time = operandsReady(op);
      for (mlir::Value result : op->getResults())
        readyAt[result] = time;
      return mlir::success();
    }
    int64_t time = 0;
    for (mlir::Value operand : op->getOperands())
      if (isTimedType(operand.getType()))
        time = std::max(time, ready(operand));
    for (mlir::Value result : op->getResults())
      if (isTimedType(result.getType()))
        readyAt[result] = time;
    return mlir::success();
  }

  /// Start the block once every operand is there when the block needs it.
  void forwardScheduleBlock(pulse::ScheduleBlockOp block) {
    BlockShape shape = shapeOf(block);
    auto inputs = block.getInputs();
    int64_t t = timelineOf(block);
    int64_t start = floorOf(t);
    for (auto [input, offset] : llvm::zip(inputs, shape.in))
      if (offset >= 0) {
        start = std::max(start, ready(input) - offset);
        // An outcome reaches the block's other controllers later.
        for (auto [measurement, extra] : carriedOutcomes(block, input))
          start = std::max(start, ready(measurement) + extra - offset);
      }
    // A block placed whole is held on every controller.
    if (holding())
      start = std::max(start, *llvm::max_element(heldFloors));
    if constexpr (Gridded)
      start = pulse::LineGrid::roundUp(start, grid.lcm());
    if (classical)
      start = takeBlockSlots(block, start, /*backward=*/false);
    record(block, start);
    for (auto [result, offset, source] :
         llvm::zip(block.getOutputs(), shape.out, shape.source))
      readyAt[result] = offset >= 0 ? start + offset : ready(inputs[source]);
    for (int64_t &at : heldEnds)
      at = std::max(at, start + shape.length);
    noteEnd(start + shape.length, t);
  }

  mlir::LogicalResult forwardLoop(mlir::scf::ForOp loop) {
    int64_t count;
    if (mlir::failed(tripCount(loop, count)))
      return mlir::failure();
    auto args = loop.getRegionIterArgs();
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    llvm::SmallVector<Carried> kinds;
    llvm::SmallVector<int64_t> carried;
    for (mlir::Value init : loop.getInitArgs()) {
      kinds.push_back(carriedKind(init.getType(), isTimedType(init.getType())));
      carried.push_back(ready(init));
    }
    // Times the body reads from outside the loop; like the enclosing floor,
    // they stop mattering once every carried value is past them.
    int64_t outside = floor;
    loop.getBody()->walk([&](mlir::Operation *inner) {
      for (mlir::Value operand : inner->getOperands())
        if (isTimedType(operand.getType()) &&
            !loop->isAncestor(operand.getParentRegion()->getParentOp()))
          outside = std::max(outside, ready(operand));
    });
    // So do the floors of the timelines carried values are on.
    llvm::SmallVector<int64_t> outsideOf;
    for (mlir::Value init : loop.getInitArgs())
      outsideOf.push_back(
          std::max(outside, isLineType(init.getType()) || !isolated()
                                ? floors[timelineOfValue(init)]
                                : *llvm::max_element(floors)));

    iteration.push_back(0);
    std::optional<int64_t> lastShift;
    int64_t laneTries = laneTriesPerLoop();
    std::optional<LaneCycle> cycle;
    const LoopGroups *groups = nullptr;
    bool grouped = false;
    std::optional<Unshifted> unshifted;
    std::optional<GroupShifts> lastGroups;
    RecentStates recent;
    // Lanes that repeat only every few iterations regroup a loop of calls
    // (a loop without is summarized by the controller check as it is).
    bool regroupLanes = resources.enabled && mayRegroup && holdsCalls(loop);
    llvm::SmallVector<LaneState, 0> laneHistory;
    LoopLinks links = linksOf(loop);
    for (int64_t k = 0; k < count; ++k) {
      iteration.back() = k;
      // Each iteration places its readouts again, so the conditions on them
      // are found again.
      if (!conditionReady.empty())
        conditionReady.clear();
      // Slots held past an iteration's last request may delay the next
      // ones: their iterations repeat only if the iteration moved them too.
      llvm::SmallVector<HeldSlots, 1> slotsBefore;
      if (!links.empty() && !lanesBound)
        slotsBefore = heldSlots(links, false);
      int64_t savedEnd = iterationEnd;
      iterationEnd = 0;
      auto savedEnds = iterationEnds;
      std::fill(iterationEnds.begin(), iterationEnds.end(), -1);
      auto floorsBefore = floors;
      auto heldFloorsBefore = heldFloors, heldEndsBefore = heldEnds;
      auto linksBefore = linkReady;
      // Only an iteration moving as the one before can be steady.
      LaneState lanesBefore;
      if (lastShift && laneTries > 0)
        lanesBefore = laneState();
      if (regroupLanes)
        pushLanes(laneHistory);
      LaneState groupLanes;
      if (resources.enabled && groupsApply())
        groupLanes = laneState();
      for (auto [arg, time] : llvm::zip(args, carried))
        readyAt[arg] = time;
      if (mlir::failed(forwardBlock(*loop.getBody())))
        return mlir::failure();
      llvm::SmallVector<int64_t> next;
      for (mlir::Value value : yield.getOperands())
        next.push_back(ready(value));
      int64_t thisEnd = iterationEnd;
      iterationEnd = std::max(savedEnd, iterationEnd);
      auto theseEnds = iterationEnds;
      for (auto [end, saved] : llvm::zip(iterationEnds, savedEnds))
        end = std::max(end, saved);

      int64_t shift;
      bool shifted = uniformShift(carried, next, kinds, shift);
      bool repeated = shifted && lastShift == shift;
      bool steady =
          !lanesBound && repeated && shift >= 0 &&
          (!Gridded || shift % grid.lcm() == 0) &&
          slotsHoldSteadyState(loop, shift) &&
          (!resources.enabled || lanesSteady(lanesBefore, shift, laneTries)) &&
          slotsMoved(links, slotsBefore, false,
                     [&](llvm::ArrayRef<mlir::Operation *>) { return shift; });
      bool bounded = true;
      for (auto [before, after, kind, bound] :
           llvm::zip(carried, next, kinds, outsideOf))
        if (kind == Carried::Timed ||
            (kind == Carried::Tone && after != before))
          bounded = bounded && after >= bound;
      steady = steady && bounded;
      // Past their tries, a LaneCycle stands in for the lanes' part of the
      // test only: it is told the shift only when every other part holds.
      if (resources.enabled && !steady && laneTries <= 0) {
        if (!cycle)
          cycle.emplace(*this, loop, count, 1);
        bool others = !lanesBound && repeated && shift >= 0 &&
                      (!Gridded || shift % grid.lcm() == 0) &&
                      slotsHoldSteadyState(loop, shift) &&
                      slotsMoved(links, slotsBefore, false,
                                 [&](llvm::ArrayRef<mlir::Operation *>) {
                                   return shift;
                                 }) &&
                      bounded;
        steady = cycle->steady(k, others ? std::optional(shift) : std::nullopt,
                               bounded);
      }
      lastShift = shifted ? std::optional<int64_t>(shift) : std::nullopt;
      // Lines that move by periods of their own repeat too, group by group.
      std::optional<GroupShifts> nowGroups;
      if (!shifted && k + 1 < count && groupsApply()) [[unlikely]] {
        if (!grouped && !unshifted) {
          unshifted = Unshifted{carried, next};
        } else if (!grouped) {
          grouped = true;
          groups = groupsOf(loop);
          if (groups)
            lastGroups = groupShifts(*groups, unshifted->before,
                                     unshifted->after, kinds, false);
          unshifted.reset();
        }
        if (groups)
          nowGroups = groupShifts(*groups, carried, next, kinds, false);
      } else if (!grouped) {
        unshifted.reset();
      }
      bool mixed = nowGroups && bounded && lastGroups == nowGroups &&
                   groupsSteady(*groups, *nowGroups, 1) &&
                   slotsHoldGroups(loop, *groups, *nowGroups, 1) &&
                   groupLanesMoved(groupLanes, *groups, *nowGroups) &&
                   slotsMoved(links, slotsBefore, false,
                              [&](llvm::ArrayRef<mlir::Operation *> ops) {
                                return requestsShift(loop, *groups, *nowGroups,
                                                     ops, false);
                              });
      // A LaneCycle may stand in for the lanes' part of that test, and only
      // for it, too.
      bool cycleMixed = false;
      if (resources.enabled && nowGroups && !mixed && groupsApply()) {
        if (!cycle)
          cycle.emplace(*this, loop, count, 1);
        bool others = bounded && lastGroups == nowGroups &&
                      groupsSteady(*groups, *nowGroups, 1) &&
                      slotsHoldGroups(loop, *groups, *nowGroups, 1) &&
                      slotsMoved(links, slotsBefore, false,
                                 [&](llvm::ArrayRef<mlir::Operation *> ops) {
                                   return requestsShift(loop, *groups,
                                                        *nowGroups, ops, false);
                                 });
        cycleMixed = mixed = cycle->steadyGroups(
            k, *groups, others ? &*nowGroups : nullptr, bounded);
      }
      bool groupsRepeated = nowGroups && lastGroups == nowGroups;
      lastGroups = std::move(nowGroups);
      llvm::SmallVector<int64_t> previous = std::move(carried);
      carried = std::move(next);
      if (mixed) [[unlikely]] {
        auto distances = groupDistances(*lastGroups, count - 1 - k);
        if (distances && extrapolateGroups(carried, previous, kinds, *groups,
                                           false, *distances)) {
          if (cycleMixed)
            cycle->move(count - 1 - k);
          else
            moveGroupLanes(groupLanes, *groups, *distances);
          moveSlots(links, false, [&](llvm::ArrayRef<mlir::Operation *> ops) {
            return *requestsShift(loop, *groups, *lastGroups, ops, false) *
                   (count - 1 - k);
          });
          // Each line ends its last iteration where it is carried to, and
          // each wait in the body where its group moves it.
          for (auto [init, time, group] :
               llvm::zip(loop.getInitArgs(), carried, groups->ofArg))
            if ((isLineType(init.getType()) ||
                 mlir::isa<classical::LinkType>(init.getType())) &&
                (*distances)[group] > 0)
              noteEnd(time, 0);
          for (auto [op, group] : groups->ofOp) {
            if (!mlir::isa<classical::WaitOp>(op) ||
                op->template getParentOfType<mlir::scf::ForOp>() != loop)
              continue;
            auto found = samples.find(op);
            if (found != samples.end() && !found->second.empty() &&
                llvm::ArrayRef(found->second.back().iteration) ==
                    llvm::ArrayRef(iteration))
              noteEnd(found->second.back().start + (*distances)[group], 0);
          }
          extrapolations.push_back(
              {loop, 0, opPeriods(*groups, *lastGroups, 1)});
          break;
        }
      }
      if (steady && k + 1 < count) {
        int64_t remaining = count - 1 - k;
        int64_t distance = remaining * shift;
        extrapolate(carried, previous, kinds, distance);
        moveSlots(links, false,
                  [&](llvm::ArrayRef<mlir::Operation *>) { return distance; });
        // What the iterations not simulated move: the ends and run-time
        // waits of the timelines the body plays on, and their links.
        if (!isolated()) {
          noteEnd(thisEnd + distance, 0);
        } else {
          for (auto [t, end] : llvm::enumerate(theseEnds))
            if (end >= 0)
              noteEnd(end + distance, static_cast<int64_t>(t));
        }
        for (auto &[key, time] : linkReady) {
          auto before = linksBefore.find(key);
          if (before == linksBefore.end() || before->second != time)
            time += distance;
        }
        for (auto [now, before] : llvm::zip(floors, floorsBefore))
          if (now != before)
            now += distance;
        moveChanged(heldFloors, heldFloorsBefore, distance);
        moveChanged(heldEnds, heldEndsBefore, distance);
        if (cycle && cycle->found(k))
          cycle->move(remaining);
        else if (resources.enabled)
          moveLanes(lanesBefore, distance);
        extrapolations.push_back({loop, shift, nullptr});
        break;
      }
      if (regroup)
        break;
      // Regrouping is for a loop that repeats only every few iterations:
      // one whose groups or lanes repeat every iteration is left to the
      // checks above.
      if (k >= 2 && !repeated && !groupsRepeated && mayRegroup) {
        noteRecent(recent, loop, carried, kinds, count - 1 - k, 1);
        if (regroup)
          break;
      }
      if (repeated && bounded && !lanesBound && laneTries <= 0 &&
          !laneHistory.empty() && !lanesMoved(laneHistory.back(), shift)) {
        noteRecentLanes(laneHistory, loop, shift, count - 1 - k);
        if (regroup)
          break;
      }
      // Iterations that play as this one moved, short of lanes the loop
      // has not reached (see LaneCycle).
      if (cycle && cycle->skip() > 0) [[unlikely]] {
        int64_t copies = cycle->skip();
        if (skipCarried(*cycle, carried, previous, kinds, shift, false)) {
          cycle->skipIterations();
          k += copies;
        }
      }
    }
    iteration.pop_back();
    for (auto [result, time] : llvm::zip(loop.getResults(), carried))
      readyAt[result] = time;
    return mlir::success();
  }

  /// Whether what `branch` yields is there only once its condition is: when
  /// the condition reads a measurement, or comes from a classical call, an
  /// arm may set a tone or a parameter without a timed op, and the op that
  /// reads it after the if still waits for the condition.
  bool yieldsAfterCondition(mlir::scf::IfOp branch) const {
    if (classical)
      return true;
    return !conditionMeasurements(branch.getCondition()).empty();
  }

  mlir::LogicalResult forwardIf(mlir::scf::IfOp branch) {
    int64_t savedFloor = floor;
    size_t savedArmFloors = armFloors.size();
    // On several controllers each outcome reaches each controller apart,
    // so the measurements an answer comes from bound the arm too, and each
    // controller's ops wait until the outcomes reach it.
    if (several()) {
      const ConditionReady &at = conditionReadyOf(branch.getCondition());
      if (at.any) {
        floor = std::max(floor, at.ready);
        armFloors.push_back(at.readers);
      }
    } else {
      for (mlir::Value measurement :
           conditionMeasurements(branch.getCondition(), false))
        floor = std::max(floor, ready(measurement));
    }
    if (classical)
      floor = std::max(floor, ready(branch.getCondition()));

    auto savedLinks = linkReady;
    size_t mark = beginArm();
    if (mlir::failed(forwardBlock(*branch.thenBlock())))
      return mlir::failure();
    llvm::SmallVector<int64_t> results;
    for (mlir::Value value : branch.thenYield().getOperands())
      results.push_back(ready(value));
    if (!branch.getElseRegion().empty()) {
      ArmLanes thenLanes;
      undoArm(mark, thenLanes);
      auto thenLinks = std::move(linkReady);
      linkReady = std::move(savedLinks);
      if (mlir::failed(forwardBlock(*branch.elseBlock())))
        return mlir::failure();
      for (auto [time, value] :
           llvm::zip(results, branch.elseYield().getOperands()))
        time = std::max(time, ready(value));
      mergeArms(mark, thenLanes, /*later=*/true);
      for (auto &[key, time] : thenLinks) {
        auto [it, inserted] = linkReady.try_emplace(key, time);
        if (!inserted)
          it->second = std::max(it->second, time);
      }
    }
    if (yieldsAfterCondition(branch))
      for (int64_t &time : results)
        time = std::max({time, floor, armFloorOf(kAnyOwner)});
    for (auto [result, time] : llvm::zip(branch.getResults(), results))
      readyAt[result] = time;
    floor = savedFloor;
    armFloors.resize(savedArmFloors);
    endArm();
    return mlir::success();
  }

  // ── Backward (ALAP) ───────────────────────────────────────────────────

  /// The latest time `value` may become available without delaying the
  /// schedule. Lines left unconsumed may run until the makespan, or until
  /// the end of their timeline: a later run-time wait elsewhere does not
  /// move them. Nothing waits for an unused measurement.
  int64_t late(mlir::Value value) {
    auto it = lateAt.find(value);
    if (it != lateAt.end())
      return withGroups(value, it->second);
    mlir::Type type = value.getType();
    if (mlir::isa<pulse::MeasurementType>(type))
      return withGroups(value, kNever);
    // Nothing waits for an unused future or untimed value; a line, tone or
    // link may run until the makespan.
    if (classical && !mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                                pulse::ToneType, classical::LinkType>(type))
      return kNever;
    if (isolated() && !mlir::isa<classical::LinkType>(type))
      return ends[timelineOfValue(value)];
    return makespan;
  }

  /// The forward start of a call op in the current iteration, also for an
  /// iteration the forward pass extrapolated.
  int64_t forwardStartOf(mlir::Operation *op) const {
    auto loops = pulse::enclosingLoops(op, region->getParentOp());
    return forwardStartAt(op, iteration, loops);
  }

  /// forwardStartOf in iteration `at` of the loops around `op`, `loops`.
  int64_t forwardStartAt(mlir::Operation *op, llvm::ArrayRef<int64_t> at,
                         llvm::ArrayRef<mlir::scf::ForOp> loops) const {
    auto starts = forwardStarts.find(op);
    if (starts == forwardStarts.end())
      return 0;
    const auto &byIteration = starts->second;
    llvm::SmallVector<int64_t, 2> key(at);
    auto it = byIteration.find(key);
    if (it != byIteration.end())
      return it->second;
    // The latest iteration simulated before this one of the innermost loop
    // the forward pass extrapolated, in the same iteration of the loops
    // around it, shifted by its period.
    for (int64_t level = static_cast<int64_t>(at.size()) - 1; level >= 0;
         --level) {
      if (level >= static_cast<int64_t>(loops.size()))
        continue;
      std::optional<int64_t> period;
      for (const Extrapolation &e : forwardExtrapolations)
        if (e.loop == loops[level]) {
          period = e.periodOf(op).value_or(0);
          break;
        }
      if (!period)
        continue;
      key.assign(at.begin(), at.begin() + level + 1);
      auto before = byIteration.lower_bound(key);
      if (before == byIteration.begin())
        continue;
      --before;
      const auto &seen = before->first;
      if (static_cast<int64_t>(seen.size()) <= level ||
          !std::equal(at.begin(), at.begin() + level, seen.begin()))
        continue;
      llvm::SmallVector<int64_t, 2> moved(at);
      moved[level] = seen[level];
      return forwardStartAt(op, moved, loops) +
             (at[level] - seen[level]) * *period;
    }
    // Otherwise the latest simulated iteration before this one.
    key.assign(at.begin(), at.end());
    auto before = byIteration.lower_bound(key);
    if (before == byIteration.begin())
      return 0;
    --before;
    if (at.empty())
      return 0;
    return before->second;
  }

  /// How many of the first iterations of `loop`, the one the backward pass
  /// is in, an extrapolation from iteration `k` moving each op by
  /// `shiftOf(op)` each would misplace: those holding an op kept at its
  /// forward start that the forward pass started off that line.
  int64_t pinnedHead(mlir::scf::ForOp loop, int64_t k,
                     llvm::function_ref<int64_t(mlir::Operation *)> shiftOf) {
    size_t depth = iteration.size() - 1;
    llvm::SmallVector<int64_t, 2> saved = iteration;
    int64_t head = 0;
    for (mlir::Operation *op : pinned) {
      if (!loop->isProperAncestor(op))
        continue;
      auto starts = forwardStarts.find(op);
      if (starts == forwardStarts.end())
        continue;
      llvm::SmallVector<int64_t, 2> key(saved.begin(), saved.begin() + depth);
      for (auto it = starts->second.lower_bound(key);
           it != starts->second.end(); ++it) {
        const auto &at = it->first;
        if (at.size() <= depth ||
            !std::equal(saved.begin(), saved.begin() + depth, at.begin()) ||
            at[depth] >= k)
          break;
        iteration.assign(at.begin(), at.end());
        iteration[depth] = k;
        if (it->second != forwardStartOf(op) + (k - at[depth]) * shiftOf(op))
          head = std::max(head, at[depth] + 1);
      }
    }
    iteration = std::move(saved);
    return head;
  }

  void backwardSubmit(classical::SubmitOp submit) {
    mlir::Operation *op = submit;
    int64_t t = timelineOf(op);
    int64_t duration = durationOf(op), budget = budgetOf(op);
    int64_t start;
    // A call that holds every line keeps its forward time, as a blocking
    // wait does: what follows starts after its answer is due.
    bool kept = blocksOnCalls || op->hasAttr("classical.eager");
    llvm::ArrayRef<int64_t> held;
    if (holding()) {
      held = heldBy(submit);
      kept = kept || !held.empty();
    }
    if (kept) {
      start = forwardStartOf(op);
      pinned.insert(op);
    } else {
      int64_t link = late(submit.getUpdatedLink());
      if (ownOrder(op)) {
        auto it = linkLate.find(linkOf(op));
        link = it == linkLate.end() ? endOf(t) : it->second;
      }
      int64_t end = std::min(link, ceilings[t]);
      int64_t needed = late(submit.getFuture());
      if (needed != kNever)
        end = std::min(end, needed - budget);
      // A call ends by the last end, as a pulse does.
      end = capAtLastEnd(op, end);
      start = end - duration;
    }
    start = takeSlotBackward(op, start, duration + budget, kept);
    record(op, start);
    armStart = std::min(armStart, start);
    for (mlir::Value operand : op->getOperands())
      bound(operand, start);
    if (blocksOnCalls)
      ceilings[t] = std::min(ceilings[t], start);
    for (int64_t c : held)
      heldCeilings[c] = std::min(heldCeilings[c], start);
    if (ownOrder(op)) {
      auto [it, inserted] = linkLate.try_emplace(linkOf(op), start);
      if (!inserted)
        it->second = std::min(it->second, start);
    }
  }

  void backwardWait(classical::WaitOp wait) {
    int64_t t = timelineOf(wait);
    int64_t start = kNever;
    for (mlir::Value value : wait.getValues())
      start = std::min(start, late(value));
    if (start == kNever || wait->hasAttr("classical.blocking")) {
      start = forwardStartOf(wait);
      pinned.insert(wait);
    }
    // A timeline's share of a link holds only the requests it times; an
    // answer it took late would hold its slot after its due time.
    else if (isolated()) {
      start = std::min(start, forwardStartOf(wait));
      pinned.insert(wait);
    }
    start = std::min(start, ceilings[t]);
    // An answer is taken by the last end, as a pulse ends by it.
    start = capAtLastEnd(wait, start);
    // An eager request keeps its forward time, so where lanes or slots
    // handed out in reverse moved what its answer decides earlier, the
    // answer would be taken before it is due.
    if (auto submit = wait.getFuture().getDefiningOp<classical::SubmitOp>();
        submit && submit->hasAttr("classical.eager") &&
        start < forwardStartOf(submit) + durationOf(submit) + budgetOf(submit))
      answeredEarly = true;
    record(wait, start);
    armStart = std::min(armStart, start);
    bound(wait.getFuture(), start);
    if (wait->hasAttr("classical.blocking"))
      ceilings[t] = std::min(ceilings[t], start);
  }

  void bound(mlir::Value value, int64_t time) {
    auto [it, inserted] = lateAt.try_emplace(value, time);
    if (!inserted)
      it->second = std::min(it->second, time);
  }

  mlir::LogicalResult backwardBlock(mlir::Block &block) {
    for (mlir::Operation &op : llvm::reverse(block.without_terminator()))
      if (mlir::failed(backwardOp(&op)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult backwardOp(mlir::Operation *op) {
    if (classical) {
      if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
        backwardSubmit(submit);
        return mlir::success();
      }
      if (auto wait = mlir::dyn_cast<classical::WaitOp>(op)) {
        backwardWait(wait);
        return mlir::success();
      }
    }
    if (classical && op->hasAttr("classical.variable")) {
      // A variable wait keeps its forward time; nothing before it ends later.
      int64_t start = forwardStartOf(op);
      pinned.insert(op);
      record(op, start);
      armStart = std::min(armStart, start);
      bound(op->getOperand(0), start);
      int64_t &ceiling = ceilings[timelineOf(op)];
      ceiling = std::min(ceiling, start);
      return mlir::success();
    }
    if (isTimedOp(op)) {
      int64_t t = timelineOf(op);
      int64_t duration = durationOf(op);
      int64_t end = late(op->getResult(0));
      if (classical)
        end = std::min(end, ceilings[t]);
      if (holding())
        end = std::min(end, heldCeilingOf(op));
      bool isReadout = mlir::isa<pulse::ReadoutOp>(op);
      int64_t owner = several() ? controllerOf(op) : kAnyOwner;
      if (isReadout) {
        int64_t latency =
            several() ? controllers.own(owner) : resources.readoutLatency;
        if (resources.enabled)
          end -= latency;
        int64_t needed = late(mlir::cast<pulse::ReadoutOp>(op).getResult());
        if (needed != kNever)
          end = std::min(end, needed - latency - readoutDelayOf(op));
      }
      if (mlir::Value tone = toneResult(op))
        end = std::min(end, late(tone));
      end = capAtLastEnd(op, end);
      int64_t step = 1;
      if constexpr (Gridded)
        step = grid.of(op->getOperand(0), lines);
      if (resources.enabled && !mlir::isa<pulse::WaitOp>(op)) {
        auto &lanes = isReadout ? readoutLanes[t] : driveLanes[t];
        ++(isReadout ? readoutUses : driveUses)[t];
        size_t best = 0;
        int64_t free = -kNever;
        if (byHeap) {
          best = (isReadout ? readoutHeaps : driveHeaps)[t].best();
          free = lanes[best];
        } else {
          auto &owners = isReadout ? readoutOwners[t] : driveOwners[t];
          for (size_t i = 0; i < lanes.size(); ++i) {
            int64_t at = lanes[i] - laneGap(owners[i], owner);
            if (at > free) {
              free = at;
              best = i;
            }
          }
        }
        notePick(!isReadout, free, free < end);
        end = std::min(end, free);
        if (step > 1)
          end = pulse::LineGrid::roundDown(end - duration, step) + duration;
        setLane(isReadout, t, best, end - duration - resources.switchPenalty,
                owner);
      }
      int64_t start = end - duration;
      if (step > 1)
        start = pulse::LineGrid::roundDown(start, step);
      record(op, start);
      noteArmStart(owner, start);
      bound(op->getOperand(0), start - holdOf(op));
      if (mlir::Value tone = toneOperand(op))
        bound(tone, start);
      if (classical) {
        for (mlir::Value operand : op->getOperands())
          bound(operand, start);
        if (several())
          for (mlir::Value measurement : valueMeasurements(op))
            bound(measurement,
                  start - controllers.extra(readerOf(measurement), owner));
      }
      return mlir::success();
    }
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      backwardScheduleBlock(block);
      return mlir::success();
    }
    if (auto sync = mlir::dyn_cast<pulse::SyncOp>(op)) {
      int64_t time = endOf(timelineOf(op));
      for (mlir::Value result : sync.getResults())
        time = std::min(time, late(result));
      if constexpr (Gridded)
        time = pulse::LineGrid::roundDown(time, syncGrid(sync));
      for (mlir::Value line : sync.getLines())
        bound(line, time);
      if (!several() || sync.getLines().empty())
        armStart = std::min(armStart, time);
      else
        for (mlir::Value line : sync.getLines())
          noteArmStart(controllerOfLine(line), time);
      return mlir::success();
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return backwardLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return backwardIf(branch);
    int64_t time = kNever;
    for (mlir::Value result : op->getResults())
      if (classical || isTimedType(result.getType()))
        time = std::min(time, late(result));
    if (time != kNever)
      for (mlir::Value operand : op->getOperands())
        if (classical || isTimedType(operand.getType()))
          bound(operand, time);
    return mlir::success();
  }

  /// Start the block as late as every result's consumer allows.
  void backwardScheduleBlock(pulse::ScheduleBlockOp block) {
    BlockShape shape = shapeOf(block);
    auto inputs = block.getInputs();
    int64_t t = timelineOf(block);
    // A block before a run-time wait ends by its start, like any op.
    int64_t start = endOf(t) - shape.length;
    if (classical)
      start = std::min(start, ceilings[t] - shape.length);
    if (holding())
      start = std::min(start, *llvm::min_element(heldCeilings) - shape.length);
    for (auto [result, offset, source] :
         llvm::zip(block.getOutputs(), shape.out, shape.source)) {
      int64_t needed = late(result);
      if (needed == kNever)
        continue;
      if (offset >= 0)
        start = std::min(start, needed - offset);
      else
        bound(inputs[source], needed);
    }
    if constexpr (Gridded)
      start = pulse::LineGrid::roundDown(start, grid.lcm());
    if (classical)
      start = takeBlockSlots(block, start, /*backward=*/true);
    record(block, start);
    armStart = std::min(armStart, start);
    for (auto [input, offset] : llvm::zip(inputs, shape.in))
      if (offset >= 0) {
        bound(input, start + offset);
        for (auto [measurement, extra] : carriedOutcomes(block, input))
          bound(measurement, start + offset - extra);
      }
  }

  mlir::LogicalResult backwardLoop(mlir::scf::ForOp loop) {
    int64_t count;
    if (mlir::failed(tripCount(loop, count)))
      return mlir::failure();
    auto args = loop.getRegionIterArgs();
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    llvm::SmallVector<mlir::Value> bodyValues;
    // Whether the body bounds a value from before the loop.
    bool readsOutside = false;
    loop.getBody()->walk([&](mlir::Operation *inner) {
      for (mlir::Value result : inner->getResults())
        bodyValues.push_back(result);
      for (mlir::Region &region : inner->getRegions())
        for (mlir::Block &nested : region)
          for (mlir::Value arg : nested.getArguments())
            bodyValues.push_back(arg);
      for (mlir::Value operand : inner->getOperands())
        if ((isTimedType(operand.getType()) ||
             (classical && !mlir::matchPattern(operand, mlir::m_Constant()))) &&
            !loop->isAncestor(operand.getParentRegion()->getParentOp()))
          readsOutside = true;
    });
    llvm::SmallVector<Carried> kinds;
    llvm::SmallVector<int64_t> carried;
    for (auto [arg, result] : llvm::zip(args, loop.getResults())) {
      mlir::Type type = result.getType();
      // A measurement the body reads is due by the next iteration's use of
      // it, so it moves with the loop, as a tone does, when that use moves.
      bool due = mlir::isa<pulse::MeasurementType>(type) && !arg.use_empty();
      kinds.push_back(due ? Carried::Tone
                          : carriedKind(type, isLineType(type)));
      carried.push_back(late(result));
    }

    iteration.push_back(0);
    std::optional<int64_t> lastShift;
    int64_t laneTries = laneTriesPerLoop();
    std::optional<LaneCycle> cycle;
    const LoopGroups *groups = nullptr;
    bool grouped = false;
    std::optional<Unshifted> unshifted;
    std::optional<GroupShifts> lastGroups;
    Track track;
    bool simulatedFirst = false;
    RecentStates recent;
    // Lanes that repeat only every few iterations regroup a loop of calls
    // (a loop without is summarized by the controller check as it is).
    bool regroupLanes = resources.enabled && mayRegroup && holdsCalls(loop);
    llvm::SmallVector<LaneState, 0> laneHistory;
    LoopLinks links = linksOf(loop);
    for (int64_t k = count - 1; k >= 0; --k) {
      iteration.back() = k;
      llvm::SmallVector<HeldSlots, 1> slotsBefore;
      if (!links.empty() && !lanesBound)
        slotsBefore = heldSlots(links, true);
      if (!answerGroups.empty())
        settleGroups(loop);
      for (mlir::Value value : bodyValues)
        lateAt.erase(value);
      for (mlir::Value arg : args)
        lateAt.erase(arg);
      for (auto [value, time] : llvm::zip(yield.getOperands(), carried))
        bound(value, time);
      auto ceilingsBefore = ceilings;
      auto heldCeilingsBefore = heldCeilings;
      auto linksBefore = linkLate;
      // Only an iteration moving as the one before can be steady.
      LaneState lanesBefore;
      if (lastShift && laneTries > 0)
        lanesBefore = laneState();
      if (regroupLanes)
        pushLanes(laneHistory);
      LaneState groupLanes;
      if (resources.enabled && groupsApply())
        groupLanes = laneState();
      if (groups) {
        track.earliest.assign(groups->count, kNever);
        tracks.push_back(&track);
      } else if (unshifted) {
        track.starts.clear();
        tracks.push_back(&track);
      }
      bool tracked = groups || unshifted;
      // The earliest starts of this iteration, in the arm the loop is in.
      int64_t armBefore = std::exchange(armStart, kNever);
      ArmStarts armStartsBefore = takeArmStarts();
      mlir::LogicalResult simulated = backwardBlock(*loop.getBody());
      if (tracked)
        tracks.pop_back();
      if (mlir::failed(simulated))
        return mlir::failure();
      int64_t iterationArmStart = armStart;
      ArmStarts iterationArmStarts = takeArmStarts();
      putArmStarts(armStartsBefore);
      armStart = std::min(armBefore, armStart);
      for (auto [c, start] : iterationArmStarts)
        noteArmStart(c, start);
      llvm::SmallVector<int64_t> next;
      for (mlir::Value arg : args)
        next.push_back(late(arg));

      int64_t shift;
      bool shifted = uniformShift(carried, next, kinds, shift);
      bool repeated = shifted && lastShift == shift;
      bool steady =
          !lanesBound && repeated && shift <= 0 &&
          (!Gridded || shift % grid.lcm() == 0) &&
          slotsHoldSteadyState(loop, -shift) &&
          (!resources.enabled || lanesSteady(lanesBefore, shift, laneTries)) &&
          slotsMoved(links, slotsBefore, true,
                     [&](llvm::ArrayRef<mlir::Operation *>) { return shift; });
      // Past their tries, a LaneCycle stands in for the lanes' part of the
      // test only: it is told the shift only when every other part holds.
      if (resources.enabled && !steady && laneTries <= 0) {
        if (!cycle)
          cycle.emplace(*this, loop, count, -1);
        bool others = !lanesBound && repeated && shift <= 0 &&
                      (!Gridded || shift % grid.lcm() == 0) &&
                      slotsHoldSteadyState(loop, -shift) &&
                      slotsMoved(links, slotsBefore, true,
                                 [&](llvm::ArrayRef<mlir::Operation *>) {
                                   return shift;
                                 });
        steady = cycle->steady(k, others ? std::optional(shift) : std::nullopt,
                               true);
      }
      lastShift = shifted ? std::optional<int64_t>(shift) : std::nullopt;
      // Lines that move by periods of their own repeat too, group by group.
      std::optional<GroupShifts> nowGroups;
      if (!shifted && k > 0 && groupsApply()) [[unlikely]] {
        if (!grouped && !unshifted) {
          unshifted = Unshifted{carried, next};
        } else if (!grouped) {
          grouped = true;
          groups = groupsOf(loop);
          if (groups && groups->mixedWaits && groups->linkBound)
            groups = groups->linkBound.get();
          track.groups = groups;
          if (groups) {
            lastGroups = groupShifts(*groups, unshifted->before,
                                     unshifted->after, kinds, true);
            track.earliest.assign(groups->count, kNever);
            for (auto [op, start] : track.starts)
              if (groups->ofOp.contains(op)) {
                int64_t &earliest =
                    track.earliest[groups->groupOf(op, /*backward=*/true)];
                earliest = std::min(earliest, start);
              }
          }
          track.starts.clear();
          unshifted.reset();
        }
        if (groups)
          nowGroups = groupShifts(*groups, carried, next, kinds, true);
      } else if (!grouped) {
        unshifted.reset();
      }
      // Each group that moves has an op to move: the syncs of one without
      // would not start where its ops do.
      bool mixed = nowGroups && !track.earliest.empty() &&
                   lastGroups == nowGroups &&
                   groupsSteady(*groups, *nowGroups, -1);
      if (mixed)
        for (auto [shift, earliest] : llvm::zip(*nowGroups, track.earliest))
          if (shift.value_or(0) != 0 && earliest == kNever)
            mixed = false;
      bool others = mixed && slotsHoldGroups(loop, *groups, *nowGroups, -1) &&
                    slotsMoved(links, slotsBefore, true,
                               [&](llvm::ArrayRef<mlir::Operation *> ops) {
                                 return requestsShift(loop, *groups, *nowGroups,
                                                      ops, true);
                               });
      mixed = others && groupLanesMoved(groupLanes, *groups, *nowGroups);
      // A LaneCycle may stand in for the lanes' part of that test, and only
      // for it, too.
      bool cycleMixed = false;
      if (resources.enabled && nowGroups && !mixed && groupsApply()) {
        if (!cycle)
          cycle.emplace(*this, loop, count, -1);
        cycleMixed = mixed = cycle->steadyGroups(
            k, *groups, others ? &*nowGroups : nullptr, true);
      }
      if (mixed && k > 0 && !pinned.empty()) {
        const GroupShifts &shifts = *nowGroups;
        int64_t head = pinnedHead(loop, k, [&](mlir::Operation *op) {
          return groups->ofOp.contains(op)
                     ? shifts[groups->groupOf(op, /*backward=*/true)].value_or(
                           0)
                     : 0;
        });
        if (head > 0 && mayPeelHead) {
          headPeel = Regroup{loop, head};
          break;
        }
        mixed = head == 0;
      }
      bool groupsRepeated = nowGroups && lastGroups == nowGroups;
      lastGroups = std::move(nowGroups);
      llvm::SmallVector<int64_t> previous = std::move(carried);
      carried = std::move(next);
      if (mixed) [[unlikely]] {
        // A measurement from before the loop is bounded by every
        // iteration's branches, the first iteration's the earliest: skip to
        // it and simulate it.
        bool first = !groups->outside.empty();
        int64_t copies = first ? k - 1 : k;
        auto distances = groupDistances(*lastGroups, copies);
        if (distances && extrapolateGroups(carried, previous, kinds, *groups,
                                           true, *distances)) {
          if (cycleMixed)
            cycle->move(copies);
          else
            moveGroupLanes(groupLanes, *groups, *distances);
          moveSlots(links, true, [&](llvm::ArrayRef<mlir::Operation *> ops) {
            return *requestsShift(loop, *groups, *lastGroups, ops, true) *
                   copies;
          });
          // The iterations not simulated start their ops earlier, by
          // their group's distance in the first, and so the arm it is in.
          for (auto [earliest, distance] :
               llvm::zip(track.earliest, *distances))
            lowerArmStart(armStart, earliest, distance);
          extrapolations.push_back(
              {loop, 0, opPeriods(*groups, *lastGroups, -1)});
          if (first) {
            simulatedFirst = true;
            k = 1;
            continue;
          }
          break;
        }
      }
      // Ops kept at their forward starts repeat only where the forward
      // pass's did: peel the first iterations that do not.
      if (steady && k > 0 && !pinned.empty()) {
        int64_t head =
            pinnedHead(loop, k, [&](mlir::Operation *) { return shift; });
        if (head > 0 && mayPeelHead) {
          headPeel = Regroup{loop, head};
          break;
        }
        steady = head == 0;
      }
      // A value from before the loop is bounded by every iteration, the
      // first the earliest: skip to it and simulate it.
      if (steady && k > (readsOutside ? 1 : 0)) {
        int64_t distance = (readsOutside ? k - 1 : k) * shift;
        // The first iteration skipped starts the arm earliest.
        lowerArmStart(armStart, iterationArmStart, distance);
        for (auto [c, inIteration] : iterationArmStarts) {
          int64_t start = kNever;
          lowerArmStart(start, inIteration, distance);
          if (start != kNever)
            noteArmStart(c, start);
        }
        extrapolate(carried, previous, kinds, distance);
        moveSlots(links, true,
                  [&](llvm::ArrayRef<mlir::Operation *>) { return distance; });
        for (auto [now, before] : llvm::zip(ceilings, ceilingsBefore))
          if (now != before)
            now += distance;
        moveChanged(heldCeilings, heldCeilingsBefore, distance);
        for (auto &[key, time] : linkLate) {
          auto before = linksBefore.find(key);
          if (before == linksBefore.end() || before->second != time)
            time += distance;
        }
        if (cycle && cycle->found(k))
          cycle->move(readsOutside ? k - 1 : k);
        else if (resources.enabled)
          moveLanes(lanesBefore, distance);
        extrapolations.push_back({loop, -shift, nullptr});
        if (readsOutside) {
          simulatedFirst = true;
          k = 1;
          continue;
        }
        break;
      }
      if (regroup || headPeel)
        break;
      if (count - 1 - k >= 2 && !repeated && !groupsRepeated && mayRegroup) {
        noteRecent(recent, loop, carried, kinds, k, -1);
        if (regroup)
          break;
      }
      if (repeated && !lanesBound && laneTries <= 0 && !laneHistory.empty() &&
          !lanesMoved(laneHistory.back(), shift)) {
        noteRecentLanes(laneHistory, loop, shift, k);
        if (regroup)
          break;
      }
      // Iterations that play as this one moved, short of lanes the loop
      // has not reached (see LaneCycle).
      if (cycle && cycle->skip() > 0) [[unlikely]] {
        int64_t copies = cycle->skip();
        if (skipCarried(*cycle, carried, previous, kinds, shift, true)) {
          cycle->skipIterations();
          k -= copies;
        }
      }
    }
    // The first iteration was simulated for what it bounds before the loop;
    // its samples would leave a gap the extrapolation already spans.
    if (simulatedFirst) [[unlikely]] {
      size_t depth = iteration.size() - 1;
      llvm::ArrayRef<int64_t> outer =
          llvm::ArrayRef(iteration).take_front(depth);
      loop.getBody()->walk([&](mlir::Operation *op) {
        auto found = samples.find(op);
        if (found == samples.end())
          return;
        auto &list = found->second;
        while (!list.empty() && list.back().iteration.size() > depth &&
               list.back().iteration[depth] == 0 &&
               llvm::ArrayRef(list.back().iteration).take_front(depth) ==
                   outer) {
          if (list.back().start < 0)
            skippedNegative = true;
          list.pop_back();
        }
      });
      llvm::erase_if(covers, [&](const Cover &cover) {
        return cover.outer.size() > depth && cover.outer[depth] == 0 &&
               llvm::ArrayRef(cover.outer).take_front(depth) == outer;
      });
    }
    iteration.pop_back();
    for (auto [init, time] : llvm::zip(loop.getInitArgs(), carried))
      bound(init, time);
    return mlir::success();
  }

  mlir::LogicalResult backwardIf(mlir::scf::IfOp branch) {
    int64_t savedArmStart = armStart;
    armStart = kNever;
    ArmStarts savedArmStarts = takeArmStarts();
    auto savedLinks = linkLate;
    size_t mark = beginArm();
    for (auto [value, result] :
         llvm::zip(branch.thenYield().getOperands(), branch.getResults()))
      bound(value, late(result));
    if (mlir::failed(backwardBlock(*branch.thenBlock())))
      return mlir::failure();
    if (!branch.getElseRegion().empty()) {
      ArmLanes thenLanes;
      undoArm(mark, thenLanes);
      auto thenLinks = std::move(linkLate);
      linkLate = std::move(savedLinks);
      for (auto [value, result] :
           llvm::zip(branch.elseYield().getOperands(), branch.getResults()))
        bound(value, late(result));
      if (mlir::failed(backwardBlock(*branch.elseBlock())))
        return mlir::failure();
      mergeArms(mark, thenLanes, /*later=*/false);
      for (auto &[key, time] : thenLinks) {
        auto [it, inserted] = linkLate.try_emplace(key, time);
        if (!inserted)
          it->second = std::min(it->second, time);
      }
    }
    if (several()) {
      // Each outcome reaches every controller of an arm before its ops.
      int64_t earliest = armStart;
      for (auto [c, start] : armStarts)
        earliest = std::min(earliest, start);
      if (earliest != kNever) {
        AnswerGroup &group = answerGroup(branch.getCondition());
        for (auto [reader, bounded] : llvm::zip(group.readers, group.latest)) {
          int64_t latest = armStart == kNever
                               ? kNever
                               : armStart - controllers.extra(reader, -1);
          for (auto [c, start] : armStarts)
            latest = std::min(latest, start - controllers.extra(reader, c));
          bounded = std::min(bounded, latest);
        }
        if (classical)
          bound(branch.getCondition(), earliest);
      }
    } else if (armStart != kNever) {
      for (mlir::Value measurement :
           conditionMeasurements(branch.getCondition(), false))
        bound(measurement, armStart);
      if (classical)
        bound(branch.getCondition(), armStart);
    }
    if (yieldsAfterCondition(branch)) {
      int64_t needed = kNever;
      for (mlir::Value result : branch.getResults())
        needed = std::min(needed, late(result));
      if (needed != kNever) {
        if (several()) {
          AnswerGroup &group = answerGroup(branch.getCondition());
          for (auto [reader, bounded] : llvm::zip(group.readers, group.latest))
            bounded = std::min(bounded, needed - controllers.extra(reader, -1));
        } else
          for (mlir::Value measurement :
               conditionMeasurements(branch.getCondition(), false))
            bound(measurement, needed);
        bound(branch.getCondition(), needed);
      }
    }
    armStart = std::min(savedArmStart, armStart);
    // The arm's starts join the ones before the branch.
    ArmStarts armOnly = takeArmStarts();
    putArmStarts(savedArmStarts);
    for (auto [c, start] : armOnly)
      noteArmStart(c, start);
    endArm();
    return mlir::success();
  }

  mlir::Region *region;
  Resources resources;
  llvm::DenseMap<mlir::Value, int64_t> readyAt;
  llvm::DenseMap<mlir::Value, int64_t> lateAt;
  /// The controllers each schedule block plays on (playersOf).
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int64_t, 2>> blockPlayers;
  /// Per branch condition, on several controllers: answerGroup, and the
  /// groups each measurement is in (group, slot of its reader);
  /// conditionReadyOf.
  std::vector<AnswerGroup> answerGroups;
  llvm::DenseMap<mlir::Value, unsigned> groupOf;
  llvm::DenseMap<mlir::Value,
                 llvm::SmallVector<std::pair<unsigned, unsigned>, 1>>
      memberOf;
  llvm::DenseMap<mlir::Value, ConditionReady> conditionReady;
  /// Under rcp, the drive and readout lanes of each timeline.
  Lanes driveLanes, readoutLanes;
  /// Their order (byHeap: one controller, or controllers whose starts may
  /// not differ).
  llvm::SmallVector<LaneHeap, 1> driveHeaps, readoutHeaps;
  bool byHeap = false;
  /// In arms of branches, each lane as it was before the arm first changed
  /// it, in order; where each arm's notes begin (armMarks); and where each
  /// lane's latest note is (-1: none).
  llvm::SmallVector<LaneUndo, 0> laneUndo;
  llvm::SmallVector<size_t, 4> armMarks;
  Lanes driveNoted, readoutNoted;
  llvm::SmallVector<int64_t, 2> iteration;
  /// No op starts before the condition of the branch it is in.
  int64_t floor = 0;
  int64_t iterationEnd = 0;
  int64_t armStart = kNever;
  // Classical calls.
  bool classical = false;
  /// The lines' timelines, when there is more than one.
  const Timelines *lanesPer = nullptr;
  pulse::AllocationTracer trace;
  llvm::DenseMap<mlir::Operation *, int64_t> timelineCache;
  /// Per timeline: no op starts before its last run-time wait (forward),
  /// where its ops end (in the current loop iteration: -1 for none), and
  /// under ALAP nothing before a run-time wait ends after it.
  llvm::SmallVector<int64_t, 1> floors, ends, iterationEnds, ceilings;
  /// For submissions with their own order (`ownOrder`), when their link is
  /// free (forward) or must be free from (backward), in that order.
  std::map<LinkKey, int64_t> linkReady, linkLate;
  /// Request slots per link and timeline.
  llvm::DenseMap<LinkKey, LinkSlots> linkLanes;
  llvm::DenseMap<LinkKey, int64_t> linkBudgets;
  int64_t laneReset = 0;
  /// A link's slots delayed a request: loop iterations may differ, so they
  /// are all simulated.
  bool lanesBound = false;
  llvm::DenseMap<mlir::Operation *, std::shared_ptr<const LoopGroups>>
      groupCache;
  /// The loops whose ops `record` notes the earliest start of, innermost
  /// last.
  llvm::SmallVector<Track *, 2> tracks;
  /// The requests of each block, by local start and by local end.
  llvm::DenseMap<mlir::Operation *, std::pair<llvm::SmallVector<BlockRequest>,
                                              llvm::SmallVector<BlockRequest>>>
      blockRequests;
  /// The forward start of each op, by iteration of the loops around it: kept
  /// per op, each op's ordered by iteration only, so that no tree is ordered
  /// by the ops' addresses (it would take another shape, and make another
  /// number of comparisons, in another build).
  llvm::DenseMap<mlir::Operation *,
                 std::map<llvm::SmallVector<int64_t, 2>, int64_t>>
      forwardStarts;
  /// The ops the backward pass placed at their forward starts.
  llvm::SmallPtrSet<mlir::Operation *, 8> pinned;
  llvm::SmallVector<Extrapolation> forwardExtrapolations;
  // Lines whose controllers start operations only every few program cycles.
  pulse::LineGrid grid;
  [[no_unique_address]] std::conditional_t<Gridded, pulse::AllocationTracer,
                                           NoLineTrace> lines;
  /// Under rcp on several controllers: which plays each line, and what
  /// crossing between two costs.
  pulse::Controllers controllers;
  /// When calls hold controllers (`holding`), which plays each line under
  /// any policy, and the controllers each call holds.
  pulse::Controllers owners;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int64_t, 2>> heldCalls;
  /// In branches on readouts, each branch's readers and when the latest
  /// outcome each read is ready; an op of a controller waits for them to
  /// reach it (armFloorOf; forward; empty outside one).
  llvm::SmallVector<llvm::SmallVector<std::pair<int64_t, int64_t>, 2>, 2>
      armFloors;
  /// By controller, the earliest start of an op of it in the arm being
  /// scheduled; `armStart` has those of no one controller (backward).
  /// Most arms hold ops of few controllers, so this is a short list of
  /// (controller, start); empty (one controller, or an arm of no one
  /// controller's ops), it costs nothing to save and restore.
  ArmStarts armStarts;
  /// Once `armStarts` is long, where each controller's entry is in it: the
  /// entries stamped `armStamp` are its, older ones belong to lists since
  /// put aside.
  llvm::SmallVector<std::pair<uint64_t, uint32_t>, 0> armSlots;
  uint64_t armStamp = 0;
  bool armIndexed = false;
  static constexpr size_t kArmIndexedAfter = 8;
  /// The controller of each lane's last op.
  Lanes driveOwners, readoutOwners;
  /// Per timeline, how many ops have taken its drive and readout lanes.
  llvm::SmallVector<int64_t, 1> driveUses, readoutUses;
  bool lanesKnown = false, lanesInterchangeable = false;
  /// Where the iterations of the loops being watched (LaneCycle) note the
  /// lanes they take, innermost last.
  llvm::SmallVector<LanePicks *, 2> picking;
};

// ── Writing the schedule ─────────────────────────────────────────────────

/// Why an op's samples are not affine in its iterations: the iterations of
/// `loop` to peel off its front and back, or none to unroll it fully.
struct NotAffine {
  mlir::scf::ForOp loop;
  int64_t peelFront = 0;
  int64_t peelBack = 0;
};

/// The iterations to peel off `loop` so that of the starts `line` it plays
/// one op at, by iteration, one run of equal steps between consecutive
/// iterations is left: the one over the most iterations, counting for the
/// run at an end past which the scheduler extrapolated the iterations it
/// did not simulate, as they repeat it; any other run has two steps or
/// more. None when there is no such run.
static NotAffine peelAround(mlir::scf::ForOp loop,
                            llvm::ArrayRef<std::pair<int64_t, int64_t>> line,
                            int64_t trips, llvm::ArrayRef<bool> covered = {}) {
  // Two starts run on when their iterations follow each other, or when the
  // iterations between them are covered (see Cover), per iteration.
  auto next = [&](size_t i) {
    return line[i].first == line[i - 1].first + 1 ||
           (!covered.empty() && covered[i]);
  };
  auto stepTo = [&](size_t i) {
    return (line[i].second - line[i - 1].second) /
           (line[i].first - line[i - 1].first);
  };
  bool before = line.front().first > 0, after = line.back().first < trips - 1;
  if (before && after)
    return {loop};
  size_t first = 0, last = 0;
  int64_t longest = 0;
  bool extended = false;
  for (size_t from = 0; from + 1 < line.size();) {
    size_t to = from + 1;
    if (!next(to)) {
      from = to;
      continue;
    }
    int64_t step = stepTo(to);
    while (to + 1 < line.size() && next(to + 1) && stepTo(to + 1) == step)
      ++to;
    // The run at the end extrapolated past spans the iterations past it too.
    bool extends = (before && from == 0) || (after && to == line.size() - 1);
    int64_t span = (extends && before ? 0 : line[from].first);
    span = (extends && after ? trips - 1 : line[to].first) - span;
    if (extends ? span >= 1 && span >= longest : span >= 2 && span > longest) {
      first = from;
      last = to;
      longest = span;
      extended = extends;
    }
    from = to;
  }
  if (longest == 0)
    return {loop};
  return {loop, extended && before ? 0 : line[first].first,
          extended && after ? 0 : trips - 1 - line[last].first};
}

/// The start of an op at `iteration` of the loops around it (`loops`): the
/// sample there, or, in iterations a Cover spans, the start in its anchor
/// moved by its period. None where neither is.
static std::optional<int64_t>
coveredStart(mlir::Operation *op, llvm::ArrayRef<int64_t> iteration,
             llvm::ArrayRef<mlir::scf::ForOp> loops,
             const llvm::DenseMap<llvm::ArrayRef<int64_t>, int64_t> &at,
             llvm::ArrayRef<Cover> covers) {
  if (auto it = at.find(iteration); it != at.end())
    return it->second;
  for (const Cover &cover : covers) {
    auto level =
        static_cast<size_t>(llvm::find(loops, cover.loop) - loops.begin());
    if (level >= loops.size() || cover.outer.size() != level ||
        iteration.take_front(level) != llvm::ArrayRef(cover.outer) ||
        iteration[level] <= cover.lo || iteration[level] >= cover.hi)
      continue;
    llvm::SmallVector<int64_t, 2> anchor(iteration.begin(), iteration.end());
    anchor[level] = cover.anchor;
    if (auto start = coveredStart(op, anchor, loops, at, covers))
      return *start + (iteration[level] - cover.anchor) * cover.periodOf(op);
  }
  return std::nullopt;
}

/// Complete `line`, the starts along loop `level` through `iteration` that
/// samples give, with those of the iterations Covers span, as when every
/// iteration is simulated: the first and last of a covered run of the loop
/// itself (markCovered marks the iterations between), and every other
/// iteration of the loop some sample is in.
static void
coveredLine(llvm::SmallVectorImpl<std::pair<int64_t, int64_t>> &line,
            mlir::Operation *op, llvm::ArrayRef<mlir::scf::ForOp> loops,
            int64_t level, llvm::ArrayRef<int64_t> iteration,
            llvm::ArrayRef<Sample> samples,
            const llvm::DenseMap<llvm::ArrayRef<int64_t>, int64_t> &at,
            llvm::ArrayRef<Cover> covers, int64_t trips) {
  auto outer = iteration.take_front(static_cast<size_t>(level));
  llvm::SmallVector<int64_t> candidates;
  for (const Sample &other : samples)
    if (llvm::ArrayRef(other.iteration).take_front(outer.size()) == outer)
      candidates.push_back(other.iteration[static_cast<size_t>(level)]);
  for (const Cover &cover : covers)
    if (cover.loop == loops[static_cast<size_t>(level)] &&
        llvm::ArrayRef(cover.outer) == outer && cover.lo + 1 <= cover.hi - 1) {
      candidates.push_back(cover.lo + 1);
      candidates.push_back(cover.hi - 1);
    }
  llvm::sort(candidates);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  llvm::SmallVector<int64_t, 2> at_(iteration.begin(), iteration.end());
  for (int64_t k : candidates) {
    if (k < 0 || k >= trips)
      continue;
    at_[static_cast<size_t>(level)] = k;
    if (auto start = coveredStart(op, at_, loops, at, covers))
      line.push_back({k, *start});
  }
}

/// Mark each start of `line` (sorted, one per iteration) that follows the
/// one before over iterations a Cover of `loop` in `outer` spans.
static void markCovered(llvm::ArrayRef<std::pair<int64_t, int64_t>> line,
                        llvm::SmallVectorImpl<bool> &covered,
                        mlir::scf::ForOp loop, llvm::ArrayRef<int64_t> outer,
                        llvm::ArrayRef<Cover> covers) {
  covered.assign(line.size(), false);
  for (size_t i = 1; i < line.size(); ++i)
    for (const Cover &cover : covers)
      if (cover.loop == loop && llvm::ArrayRef(cover.outer) == outer &&
          line[i - 1].first == cover.lo + 1 && line[i].first == cover.hi - 1)
        covered[i] = true;
}

static std::optional<NotAffine>
writeTiming(mlir::Operation *op, mlir::Operation *root,
            llvm::ArrayRef<Sample> samples,
            llvm::ArrayRef<Extrapolation> extrapolations,
            llvm::ArrayRef<Cover> covers = {}) {
  auto loops = pulse::enclosingLoops(op, root);
  int64_t duration = durationOf(op);
  auto write = [&](std::optional<int64_t> start,
                   llvm::ArrayRef<int64_t> periods) {
    pulse::setTiming(op, start, duration, periods);
  };
  if (samples.empty()) {
    // Never executed (inside a zero-trip loop).
    write(std::nullopt, {});
    return std::nullopt;
  }
  if (loops.empty() && samples.size() == 1) {
    write(samples.front().start, {});
    return std::nullopt;
  }

  llvm::DenseMap<llvm::ArrayRef<int64_t>, int64_t> at;
  for (const Sample &sample : samples)
    at[sample.iteration] = sample.start;

  // The per-iteration shift along each loop, from any two simulated
  // neighbors, else from the extrapolation that skipped them.
  llvm::SmallVector<int64_t, 2> periods(loops.size(), 0);
  for (auto [level, loop] : llvm::enumerate(loops)) {
    bool found = false;
    for (const Sample &sample : samples) {
      llvm::SmallVector<int64_t, 2> next(sample.iteration.begin(),
                                         sample.iteration.end());
      ++next[level];
      auto it = at.find(next);
      if (it != at.end()) {
        periods[level] = it->second - sample.start;
        found = true;
        break;
      }
    }
    for (const Extrapolation &e : extrapolations)
      if (e.loop == loop && !found)
        periods[level] = e.periodOf(op).value_or(0);
  }
  int64_t base = samples.front().start;
  for (auto [k, period] : llvm::zip(samples.front().iteration, periods))
    base -= k * period;

  for (const Sample &sample : samples) {
    int64_t expected = base;
    for (auto [k, period] : llvm::zip(sample.iteration, periods))
      expected += k * period;
    if (expected == sample.start)
      continue;
    // Find the innermost loop along which this sample breaks the pattern,
    // and whether only the loop's first or last iteration differs.
    for (int64_t level = static_cast<int64_t>(loops.size()) - 1; level >= 0;
         --level) {
      auto trips = pulse::constantTripCount(loops[level]).value_or(0);
      // The simulated iterations along this loop, from the samples: an
      // extrapolated loop has few of its trips among them.
      llvm::SmallVector<std::pair<int64_t, int64_t>> line;
      for (const Sample &other : samples) {
        bool along = true;
        for (auto [i, k] : llvm::enumerate(other.iteration))
          if (static_cast<int64_t>(i) != level && k != sample.iteration[i])
            along = false;
        if (along && other.iteration[level] >= 0 &&
            other.iteration[level] < trips)
          line.push_back({other.iteration[level], at.lookup(other.iteration)});
      }
      llvm::SmallVector<bool> covered;
      if (!covers.empty())
        coveredLine(line, op, loops, level, sample.iteration, samples, at,
                    covers, trips);
      llvm::sort(line);
      line.erase(std::unique(line.begin(), line.end(),
                             [](const auto &a, const auto &b) {
                               return a.first == b.first;
                             }),
                 line.end());
      if (!covers.empty())
        markCovered(line, covered, loops[level],
                    llvm::ArrayRef(sample.iteration).take_front(level), covers);
      llvm::SmallVector<int64_t> steps;
      for (size_t i = 1; i < line.size(); ++i)
        steps.push_back((line[i].second - line[i - 1].second) /
                        (covered.empty() || !covered[i]
                             ? 1
                             : line[i].first - line[i - 1].first));
      if (llvm::all_equal(steps) &&
          (steps.empty() || steps[0] == periods[level]))
        continue;
      bool front = line.front().first == 0 && steps.size() >= 2 &&
                   llvm::all_equal(llvm::ArrayRef(steps).drop_front());
      bool back = line.back().first == trips - 1 && steps.size() >= 2 &&
                  llvm::all_equal(llvm::ArrayRef(steps).drop_back());
      if (front || back)
        return NotAffine{loops[level], front, !front && back};
      return peelAround(loops[level], line, trips, covered);
    }
    return NotAffine{loops.back()};
  }
  for (const Extrapolation &e : extrapolations)
    for (auto [level, loop] : llvm::enumerate(loops))
      if (e.loop == loop && periods[level] != e.periodOf(op))
        return NotAffine{loop};

  write(base, periods);
  return std::nullopt;
}

static mlir::Value indexConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                 int64_t value) {
  return mlir::arith::ConstantIndexOp::create(builder, loc, value);
}

// ── The kernel's numbers of readouts ────────────────────────────────────
//
// Peeling and unrolling copy a loop's readouts. Each keeps the number and
// the loop iteration the kernel gave it (pulse::kernelReadoutKey): a module
// built from a kernel numbers its readouts (`pulse.readout_number`) before
// any pass runs; in one that does not, the readouts of a function are
// numbered in walk order before the first of them is copied here. A readout
// in a loop being peeled or unrolled carries how its iterations map to the
// kernel's (`pulse.readout_iteration`).

static bool isReadoutOp(mlir::Operation *op) {
  return mlir::isa<pulse::ReadoutOp>(op) ||
         op->getName().getStringRef() == "pulse.iq_acquire";
}

static llvm::SmallVector<mlir::Operation *>
readoutsIn(llvm::ArrayRef<mlir::Operation *> ops) {
  llvm::SmallVector<mlir::Operation *> readouts;
  for (mlir::Operation *op : ops)
    op->walk([&](mlir::Operation *inner) {
      if (isReadoutOp(inner))
        readouts.push_back(inner);
    });
  return readouts;
}

/// The readouts in `loop`, each with its iteration map written out, its
/// function's readouts numbered first if they are not.
static llvm::SmallVector<mlir::Operation *>
loopReadouts(mlir::scf::ForOp loop) {
  llvm::SmallVector<mlir::Operation *> readouts;
  for (mlir::Operation &op : loop.getBody()->without_terminator())
    op.walk([&](mlir::Operation *inner) {
      if (isReadoutOp(inner))
        readouts.push_back(inner);
    });
  if (readouts.empty())
    return readouts;
  mlir::MLIRContext *ctx = loop->getContext();
  if (!readouts.front()->hasAttr(pulse::kReadoutNumberAttr)) {
    mlir::Operation *scope = loop->getParentOfType<mlir::func::FuncOp>();
    if (!scope)
      for (scope = loop; scope->getParentOp();)
        scope = scope->getParentOp();
    pulse::numberReadouts(scope);
  }
  for (mlir::Operation *op : readouts) {
    if (op->hasAttr(pulse::kReadoutIterationAttr))
      continue;
    llvm::SmallVector<int64_t> map;
    for (size_t depth = 0, n = pulse::enclosingLoops(op).size(); depth < n;
         ++depth)
      map.append({static_cast<int64_t>(depth), 1, 0});
    op->setAttr(pulse::kReadoutIterationAttr,
                mlir::DenseI64ArrayAttr::get(ctx, map));
  }
  return readouts;
}

/// Rewrite the iteration maps of `readouts` for the loop at `depth`: its
/// iteration t now was its iteration scale * t + offset before; or, with
/// `outside`, the readouts are copies of its iteration `offset` out of it.
static void remapReadouts(llvm::ArrayRef<mlir::Operation *> readouts,
                          int64_t depth, int64_t scale, int64_t offset,
                          bool outside) {
  for (mlir::Operation *op : readouts) {
    llvm::SmallVector<int64_t> map(
        op->getAttrOfType<mlir::DenseI64ArrayAttr>(pulse::kReadoutIterationAttr)
            .asArrayRef());
    for (size_t i = 0; i + 2 < map.size(); i += 3) {
      if (map[i] == depth) {
        map[i + 2] += map[i + 1] * offset;
        map[i + 1] = outside ? 0 : map[i + 1] * scale;
        if (outside)
          map[i] = -1;
      } else if (outside && map[i] > depth) {
        --map[i];
      }
    }
    op->setAttr(pulse::kReadoutIterationAttr,
                mlir::DenseI64ArrayAttr::get(op->getContext(), map));
  }
}

/// Execute the first iteration of `loop` before it.
static void peelFront(mlir::scf::ForOp loop) {
  int64_t lb = *pulse::constantIndex(loop.getLowerBound());
  int64_t step = *pulse::constantIndex(loop.getStep());
  auto readouts = loopReadouts(loop);
  mlir::OpBuilder builder(loop);
  mlir::IRMapping map;
  map.map(loop.getInductionVar(), loop.getLowerBound());
  for (auto [arg, init] :
       llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
    map.map(arg, init);
  // The copies are kept only where they hold readouts to renumber.
  llvm::SmallVector<mlir::Operation *> clones;
  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    mlir::Operation *copy = builder.clone(op, map);
    if (!readouts.empty())
      clones.push_back(copy);
  }
  if (!readouts.empty()) {
    int64_t depth = static_cast<int64_t>(pulse::enclosingLoops(loop).size());
    remapReadouts(readoutsIn(clones), depth, 1, 0, true);
    remapReadouts(readouts, depth, 1, 1, false);
  }
  llvm::SmallVector<mlir::Value> next;
  for (mlir::Value value : loop.getBody()->getTerminator()->getOperands())
    next.push_back(map.lookupOrDefault(value));
  loop.getInitArgsMutable().assign(next);
  loop.setLowerBound(indexConstant(builder, loop.getLoc(), lb + step));
  if (pulse::constantTripCount(loop) == 0) {
    loop.replaceAllUsesWith(next);
    loop.erase();
  }
}

/// Execute the last iteration of `loop` after it.
static void peelBack(mlir::scf::ForOp loop) {
  int64_t lb = *pulse::constantIndex(loop.getLowerBound());
  int64_t step = *pulse::constantIndex(loop.getStep());
  int64_t trips = *pulse::constantTripCount(loop);
  auto readouts = loopReadouts(loop);
  mlir::OpBuilder builder(loop->getContext());
  builder.setInsertionPointAfter(loop);
  mlir::IRMapping map;
  map.map(loop.getInductionVar(),
          indexConstant(builder, loop.getLoc(), lb + (trips - 1) * step));
  for (auto [arg, result] :
       llvm::zip(loop.getRegionIterArgs(), loop.getResults()))
    map.map(arg, result);
  llvm::SmallPtrSet<mlir::Operation *, 8> clones;
  for (mlir::Operation &op : loop.getBody()->without_terminator())
    clones.insert(builder.clone(op, map));
  if (!readouts.empty())
    remapReadouts(readoutsIn(llvm::to_vector(clones)),
                  static_cast<int64_t>(pulse::enclosingLoops(loop).size()), 1,
                  trips - 1, true);
  for (auto [result, value] : llvm::zip(
           loop.getResults(), loop.getBody()->getTerminator()->getOperands())) {
    mlir::Value last = map.lookupOrDefault(value);
    result.replaceUsesWithIf(last, [&](mlir::OpOperand &use) {
      for (mlir::Operation *owner = use.getOwner(); owner;
           owner = owner->getParentOp())
        if (clones.contains(owner))
          return false;
      return true;
    });
  }
  mlir::OpBuilder bounds(loop);
  loop.setUpperBound(
      indexConstant(bounds, loop.getLoc(), lb + (trips - 1) * step));
  if (trips - 1 == 0) {
    loop.replaceAllUsesWith(loop.getInitArgs());
    loop.erase();
  }
}

/// Run `factor` iterations of `loop` in each of its iterations, executing
/// those left over after it. The loop has at least 2 x `factor` trips.
static void unrollBy(mlir::scf::ForOp loop, int64_t factor) {
  for (int64_t left = *pulse::constantTripCount(loop) % factor; left > 0;
       --left)
    peelBack(loop);
  int64_t step = *pulse::constantIndex(loop.getStep());
  mlir::Block *body = loop.getBody();
  mlir::Operation *yield = body->getTerminator();
  llvm::SmallVector<mlir::Operation *> ops;
  for (mlir::Operation &op : body->without_terminator())
    ops.push_back(&op);
  auto readouts = loopReadouts(loop);
  int64_t depth =
      readouts.empty()
          ? 0
          : static_cast<int64_t>(pulse::enclosingLoops(loop).size());
  llvm::SmallVector<mlir::Value> yielded(yield->getOperands());
  llvm::SmallVector<mlir::Value> carried = yielded;
  mlir::OpBuilder builder(yield);
  mlir::Value iv = loop.getInductionVar();
  for (int64_t copy = 1; copy < factor; ++copy) {
    mlir::IRMapping map;
    if (!iv.use_empty())
      map.map(iv, mlir::arith::AddIOp::create(
                      builder, loop.getLoc(), iv,
                      indexConstant(builder, loop.getLoc(), copy * step)));
    for (auto [arg, value] : llvm::zip(loop.getRegionIterArgs(), carried))
      map.map(arg, value);
    llvm::SmallVector<mlir::Operation *> clones;
    for (mlir::Operation *op : ops) {
      mlir::Operation *clone = builder.clone(*op, map);
      if (!readouts.empty())
        clones.push_back(clone);
    }
    if (!readouts.empty())
      remapReadouts(readoutsIn(clones), depth, factor, copy, false);
    for (auto [value, original] : llvm::zip(carried, yielded))
      value = map.lookupOrDefault(original);
  }
  remapReadouts(readouts, depth, factor, 0, false);
  yield->setOperands(carried);
  mlir::OpBuilder bounds(loop);
  loop.setStep(indexConstant(bounds, loop.getLoc(), step * factor));
}

static void unrollFully(mlir::scf::ForOp loop) {
  while (auto trips = pulse::constantTripCount(loop)) {
    if (*trips == 0) {
      loop.replaceAllUsesWith(loop.getInitArgs());
      loop.erase();
      return;
    }
    bool last = *trips == 1;
    peelFront(loop);
    if (last)
      return;
  }
}

/// A loop in which two ops on one physical line move by different periods
/// per iteration. Each op's timing is affine, but the line's timeline is not
/// one pattern repeated, which happens when the first iteration differs.
static std::optional<NotAffine> periodMismatch(mlir::Region &region,
                                               mlir::Operation *root) {
  std::optional<NotAffine> problem;
  pulse::AllocationTracer trace;
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (problem || mlir::isa<pulse::ScheduleBlockOp>(op))
      return mlir::WalkResult::skip();
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op);
    if (!loop)
      return mlir::WalkResult::advance();
    size_t level = pulse::enclosingLoops(loop, root).size();
    llvm::DenseMap<mlir::Operation *, int64_t> periodOf;
    auto check = [&](mlir::Value line, int64_t period) {
      auto [it, inserted] = periodOf.try_emplace(trace(line), period);
      if (!inserted && it->second != period)
        problem = NotAffine{loop, 1};
    };
    forEachScheduled(loop.getRegion(), [&](mlir::Operation *timed) {
      llvm::ArrayRef<int64_t> periods = pulse::periodCycles(timed);
      if (problem || periods.size() <= level)
        return;
      if (mlir::isa<classical::WaitOp>(timed))
        return;
      if (!mlir::isa<pulse::ScheduleBlockOp>(timed))
        return check(timed->getOperand(0), periods[level]);
      for (mlir::Value input : timed->getOperands())
        if (isLineType(input.getType()))
          check(input, periods[level]);
    });
    return mlir::WalkResult::advance();
  });
  return problem;
}

// ── Schedule blocks ──────────────────────────────────────────────────────

/// The schedule blocks in `region` that no other block in it encloses, once
/// each line in it, outside those blocks, has one timeline
/// (checkOneAcquisition).
static mlir::FailureOr<llvm::SmallVector<pulse::ScheduleBlockOp>>
placeableBlocks(mlir::Region &region) {
  llvm::SmallVector<pulse::ScheduleBlockOp> blocks;
  mlir::WalkResult walk =
      region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
          blocks.push_back(block);
          return mlir::WalkResult::skip();
        }
        if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op) &&
            mlir::failed(pulse::checkOneAcquisition(op)))
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      });
  if (walk.wasInterrupted())
    return mlir::failure();
  return blocks;
}

/// The schedule blocks in `region` that no other block in it encloses.
static llvm::SmallVector<pulse::ScheduleBlockOp>
outermostBlocks(mlir::Region &region) {
  llvm::SmallVector<pulse::ScheduleBlockOp> blocks;
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(op)) {
      blocks.push_back(block);
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });
  return blocks;
}

/// When a value from outside a block is first needed in it: the earliest
/// local start of a timed op that depends on it, or of a branch arm
/// conditioned on it.
class FirstUse {
public:
  explicit FirstUse(pulse::ScheduleBlockOp block) : block(block) {}

  int64_t of(mlir::Value value) {
    earliest = kNever;
    visited.clear();
    visitUses(value);
    return earliest;
  }

private:
  void visitUses(mlir::Value value) {
    for (mlir::OpOperand &use : value.getUses())
      visit(use);
  }

  void note(mlir::Operation *op) {
    if (std::optional<int64_t> start = pulse::startCycles(op))
      earliest = std::min(earliest, *start);
  }

  void visit(mlir::OpOperand &use) {
    mlir::Operation *op = use.getOwner();
    if (op->getParentOp() == block &&
        op->hasTrait<mlir::OpTrait::IsTerminator>())
      return;
    if (!visited.insert(&use).second)
      return;
    // Later ops in the chain start no earlier than this one.
    if (isTimedOp(op))
      return note(op);
    unsigned index = use.getOperandNumber();
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
      mlir::Operation *parent = yield->getParentOp();
      visitUses(parent->getResult(index));
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
        visitUses(loop.getRegionIterArgs()[index]);
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      if (index >= loop.getNumControlOperands())
        visitUses(
            loop.getRegionIterArgs()[index - loop.getNumControlOperands()]);
      return;
    }
    if (op->getNumRegions() > 0) {
      // A branch condition: both arms wait for it, and so does whatever
      // reads what the branch yields (a tone an arm changes without a timed
      // op), as in the flat schedule.
      op->walk([&](mlir::Operation *inner) {
        if (isTimedOp(inner))
          note(inner);
      });
      for (mlir::Value result : op->getResults())
        visitUses(result);
      return;
    }
    for (mlir::Value result : op->getResults())
      visitUses(result);
  }

  pulse::ScheduleBlockOp block;
  llvm::DenseSet<mlir::OpOperand *> visited;
  int64_t earliest = kNever;
};

/// When a block's readout's outcome is usable: `latency` cycles after the
/// readout ends on its line, delayed by `delays` (by qudit, see
/// Resources::readoutDelays).
struct Readiness {
  int64_t latency = 0;
  llvm::ArrayRef<int64_t> delays;
  /// On several controllers, their latencies: an outcome a branch yields
  /// values under reaches the other controllers later.
  const pulse::Controllers *controllers = nullptr;

  /// How much later than on its reader the outcome of `measurement` is
  /// usable on every controller, as the flat schedule's branches on it
  /// yield (a measurement not read out in the block: of any reader).
  int64_t peerOf(mlir::Value measurement,
                 pulse::AllocationTracer &trace) const {
    if (!controllers)
      return 0;
    auto readout = measurement.getDefiningOp<pulse::ReadoutOp>();
    int64_t reader =
        readout ? controllers->of(readout->getOperand(0), trace) : -1;
    return controllers->extra(reader, -1);
  }

  int64_t delayOf(mlir::Operation *op, pulse::AllocationTracer &trace) const {
    if (delays.empty())
      return 0;
    mlir::Operation *allocation = trace(op->getOperand(0));
    int64_t qudit = allocation ? pulse::quditOf(allocation) : -1;
    if (qudit < 0 || qudit >= static_cast<int64_t>(delays.size()))
      return 0;
    return delays[qudit];
  }
};

/// The feedback latency every readout waits under `resources`: on several
/// `controllers`, the longest any controller waits for its own.
static int64_t readoutLatencyOf(const Resources &resources,
                                const pulse::Controllers &controllers) {
  return controllers.enabled() ? controllers.own(-1) : resources.readoutLatency;
}

/// What a schedule block keeps of its caller's `resources`: the feedback
/// latency and the readout lines' delays (rcp's lanes refuse blocks).
static Resources latencyOf(const Resources &resources) {
  Resources kept;
  kept.readoutLatency = resources.readoutLatency;
  kept.readoutDelays = resources.readoutDelays;
  return kept;
}

static Readiness readinessOf(const Resources &resources,
                             const pulse::Controllers &controllers) {
  return Readiness{readoutLatencyOf(resources, controllers),
                   resources.readoutDelays,
                   controllers.enabled() ? &controllers : nullptr};
}

/// When each value in a scheduled block becomes available, in local time,
/// or which operand it passes through unchanged.
class Availability {
public:
  struct State {
    int64_t time = 0;
    int source = -1;
  };

  Availability(pulse::ScheduleBlockOp block,
               llvm::MutableArrayRef<int64_t> firstUse, bool classical,
               const Readiness &readiness)
      : block(block), firstUse(firstUse), classical(classical),
        readiness(readiness) {
    for (auto [index, arg] : llvm::enumerate(block.getBody().getArguments()))
      state[arg] = State{0, static_cast<int>(index)};
  }

  void run() { visitBlock(block.getBody().front()); }

  State of(mlir::Value value) const { return state.lookup(value); }

  /// The local time every op of the block is done by.
  int64_t length = 0;

private:
  /// The latest of `states`: the source they all pass through, else a
  /// time. An operand reached here is needed by now if nothing needs it
  /// earlier.
  State join(llvm::ArrayRef<State> states) {
    if (!states.empty() && llvm::all_of(states, [&](const State &s) {
          return s.source >= 0 && s.source == states.front().source;
        }))
      return states.front();
    int64_t time = 0;
    for (const State &s : states)
      time = std::max(time, s.source >= 0
                                ? std::max<int64_t>(firstUse[s.source], 0)
                                : s.time);
    for (const State &s : states)
      if (s.source >= 0 && firstUse[s.source] < 0)
        firstUse[s.source] = time;
    return State{time, -1};
  }

  /// The states `value` is computed from: the measurements, answers and
  /// operands it reads.
  void statesOf(mlir::Value value, llvm::SmallVectorImpl<State> &found,
                llvm::DenseSet<mlir::Value> &seen) const {
    if (mlir::isa<classical::LinkType>(value.getType()) ||
        !seen.insert(value).second)
      return;
    if (auto it = state.find(value); it != state.end()) {
      found.push_back(it->second);
      return;
    }
    if (mlir::Operation *def = value.getDefiningOp())
      for (mlir::Value operand : def->getOperands())
        statesOf(operand, found, seen);
  }

  /// The iteration argument of `loop` that `value` passes on with no timed
  /// op between them (itself, or a frame change of it), else -1.
  static int passedOn(mlir::scf::ForOp loop, mlir::Value value) {
    while (true) {
      auto args = loop.getRegionIterArgs();
      if (auto it = llvm::find(args, value); it != args.end())
        return static_cast<int>(it - args.begin());
      mlir::Operation *def = value.getDefiningOp();
      if (!def || isTimedOp(def) || def->getNumRegions() > 0)
        return -1;
      mlir::Value only;
      for (mlir::Value operand : def->getOperands())
        if (isTimedType(operand.getType())) {
          if (only)
            return -1;
          only = operand;
        }
      if (!only)
        return -1;
      value = only;
    }
  }

  /// Join the carried states of each cycle of arguments that pass each
  /// other on, when `skipped` trips are not a whole number of its turns.
  void joinCycles(mlir::scf::ForOp loop, llvm::MutableArrayRef<State> carried,
                  int64_t skipped) {
    auto yields = loop.getBody()->getTerminator()->getOperands();
    llvm::SmallVector<int> from;
    for (mlir::Value value : yields)
      from.push_back(passedOn(loop, value));
    llvm::SmallVector<bool> done(from.size(), false);
    for (size_t start = 0; start < from.size(); ++start) {
      llvm::SmallVector<int> cycle;
      int at = static_cast<int>(start);
      while (at >= 0 && !done[at] && cycle.size() <= from.size() &&
             !llvm::is_contained(cycle, at)) {
        cycle.push_back(at);
        at = from[at];
      }
      if (at != static_cast<int>(start) || cycle.size() < 2)
        continue;
      for (int member : cycle)
        done[member] = true;
      if (skipped % static_cast<int64_t>(cycle.size()) == 0)
        continue;
      llvm::SmallVector<State> states;
      for (int member : cycle)
        states.push_back(carried[member]);
      State latest = join(states);
      for (int member : cycle)
        carried[member] = latest;
    }
  }

  State joinValues(mlir::ValueRange values) {
    llvm::SmallVector<State> states;
    for (mlir::Value value : values)
      if (isTimedType(value.getType()))
        states.push_back(state.lookup(value));
    return join(states);
  }

  void visitBlock(mlir::Block &body) {
    for (mlir::Operation &op : body.without_terminator())
      visitOp(&op);
  }

  void visitOp(mlir::Operation *op) {
    if (isTimedOp(op)) {
      auto start = pulse::scheduledStart(op, iteration);
      if (!start) {
        State through = joinValues(op->getOperands());
        for (mlir::Value result : op->getResults())
          state[result] = through;
        return;
      }
      int64_t end = *start + durationOf(op);
      length = std::max(length, end);
      for (mlir::Value result : op->getResults())
        state[result] = State{end, -1};
      // Its outcome after the feedback latency and the line's delay.
      if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op))
        state[readout.getResult()] =
            State{end + readiness.latency + readiness.delayOf(op, trace), -1};
      if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op))
        state[submit.getFuture()] = State{end + budgetOf(op), -1};
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      llvm::SmallVector<State> carried;
      for (mlir::Value init : loop.getInitArgs())
        carried.push_back(state.lookup(init));
      // The first iteration, then the last ones, each from what the one
      // before it carries. A value moves to at most one other iteration
      // argument per trip and through at most all of them, so what the last
      // iteration reads (a measurement an arm keeps, a tone passed from
      // argument to argument) was made in one of the last K + 1 trips.
      int64_t args = loop.getNumRegionIterArgs();
      int64_t from = std::max<int64_t>(1, trips - 1 - args);
      llvm::SmallVector<int64_t> runs;
      if (trips > 0)
        runs.push_back(0);
      for (int64_t k = from; k < trips; ++k)
        runs.push_back(k);
      iteration.push_back(0);
      for (int64_t k : runs) {
        iteration.back() = k;
        for (auto [arg, s] : llvm::zip(loop.getRegionIterArgs(), carried))
          state[arg] = s;
        visitBlock(*loop.getBody());
        for (auto [s, value] :
             llvm::zip(carried, loop.getBody()->getTerminator()->getOperands()))
          s = state.lookup(value);
      }
      iteration.pop_back();
      // Arguments that only pass each other on, in a cycle, end in an order
      // that depends on the trips skipped: take the latest of the cycle
      // unless the skipped trips are a whole number of turns.
      if (from > 1)
        joinCycles(loop, carried, from - 1);
      for (auto [result, s] : llvm::zip(loop.getResults(), carried))
        state[result] = s;
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      visitBlock(*branch.thenBlock());
      llvm::SmallVector<State> results;
      for (mlir::Value value : branch.thenYield().getOperands())
        results.push_back(state.lookup(value));
      if (!branch.getElseRegion().empty()) {
        visitBlock(*branch.elseBlock());
        for (auto [s, value] :
             llvm::zip(results, branch.elseYield().getOperands()))
          s = join({s, state.lookup(value)});
      }
      // What a branch on a measurement or an answer yields is there only
      // once its condition is, as in the flat schedule (yieldsAfterCondition).
      mlir::Value condition = branch.getCondition();
      llvm::SmallVector<mlir::Value> measured =
          conditionMeasurements(condition);
      if (classical || !measured.empty()) {
        llvm::SmallVector<State> at;
        llvm::DenseSet<mlir::Value> seen;
        statesOf(condition, at, seen);
        // On several controllers, once its outcomes reach every controller
        // (an outcome passed in: the caller waits for it, see
        // carriedOutcomes).
        for (mlir::Value measurement : measured)
          if (int64_t peer = readiness.peerOf(measurement, trace)) {
            State s = state.lookup(measurement);
            if (s.source < 0)
              at.push_back(State{s.time + peer, -1});
          }
        for (State &s : results) {
          at.push_back(s);
          s = join(at);
          at.pop_back();
        }
      }
      for (auto [result, s] : llvm::zip(branch.getResults(), results))
        state[result] = s;
      return;
    }
    State through = joinValues(op->getOperands());
    for (mlir::Value result : op->getResults())
      if (isTimedType(result.getType()))
        state[result] = through;
  }

  pulse::ScheduleBlockOp block;
  llvm::MutableArrayRef<int64_t> firstUse;
  bool classical;
  const Readiness &readiness;
  pulse::AllocationTracer trace;
  llvm::DenseMap<mlir::Value, State> state;
  llvm::SmallVector<int64_t, 2> iteration;
};

/// Record the shape of `block`, whose body has a local schedule; a
/// readout's outcome is usable as `readiness` says.
static void writeShape(pulse::ScheduleBlockOp block, bool classical,
                       const Readiness &readiness) {
  FirstUse firstUse(block);
  llvm::SmallVector<int64_t> in;
  for (mlir::Value arg : block.getBody().getArguments()) {
    int64_t first = firstUse.of(arg);
    in.push_back(first == kNever ? -1 : first);
  }
  Availability availability(block, in, classical, readiness);
  availability.run();
  llvm::SmallVector<int64_t> out, source;
  int64_t length = availability.length;
  for (mlir::Value value :
       block.getBody().front().getTerminator()->getOperands()) {
    Availability::State s = availability.of(value);
    out.push_back(s.source >= 0 ? -1 : s.time);
    source.push_back(s.source);
    // A measurement handed out is usable later than the block ends; the
    // block does not last longer for it.
    if (!mlir::isa<pulse::MeasurementType>(value.getType()))
      length = std::max(length, s.source >= 0 ? 0 : s.time);
  }
  mlir::MLIRContext *context = block.getContext();
  block->setAttr(
      "length_cycles",
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), length));
  block->setAttr("in_offset_cycles", mlir::DenseI64ArrayAttr::get(context, in));
  block->setAttr("out_offset_cycles",
                 mlir::DenseI64ArrayAttr::get(context, out));
  block->setAttr("out_source", mlir::DenseI64ArrayAttr::get(context, source));
}

/// Replace `block` by its body.
static void inlineBody(pulse::ScheduleBlockOp block) {
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
  yield->erase();
  mlir::IRRewriter rewriter(block.getContext());
  rewriter.inlineBlockBefore(&body, block, block.getInputs());
  rewriter.replaceOp(block, results);
}

/// Move the ops of the placed `block` to its start and period, then inline
/// it.
static void inlineScheduleBlock(pulse::ScheduleBlockOp block) {
  std::optional<int64_t> start = pulse::startCycles(block);
  llvm::ArrayRef<int64_t> outer = pulse::periodCycles(block);
  block.getBody().walk([&](mlir::Operation *op) {
    if (!isTimedOp(op))
      return;
    std::optional<int64_t> local = pulse::startCycles(op);
    if (!start || !local) {
      // The block or this op never runs.
      pulse::setStartCycles(op, std::nullopt);
      pulse::setPeriodCycles(op, {});
      return;
    }
    pulse::setStartCycles(op, *start + *local);
    llvm::SmallVector<int64_t> periods(outer);
    llvm::append_range(periods, pulse::periodCycles(op));
    pulse::setPeriodCycles(op, periods);
  });
  inlineBody(block);
}

// ── Duration budgets ─────────────────────────────────────────────────────
//
// `max_duration_cycles` on a schedule block, or on a controller of the
// module's control system, bounds how long every execution of the block, or
// the whole program, may take. A wait whose length is not known yet is
// scheduled at its current length; the check adds what each execution of it can
// add.

/// How many cycles longer than scheduled `op` can take: 0 for an op of
/// known length, -1 for a wait without a highest length.
static int64_t stretchOf(mlir::Operation *op) {
  if (mlir::isa<classical::WaitOp>(op))
    return op->hasAttr("classical.blocking") ? -1 : 0;
  auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
  if (!wait || !op->hasAttr("pulse.variable_wait"))
    return 0;
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>()) {
    bool known = op->hasAttr("classical.variable")
                     ? pulse::boundNumber(conversion.getCycles()).has_value()
                     : traceConstantI64(conversion.getCycles()).has_value();
    if (known)
      return 0;
  }
  auto highest = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles");
  if (!highest)
    return -1;
  return std::max<int64_t>(0, highest.getInt() - durationOf(op));
}

/// The executions of `op` per execution of `root`.
static int64_t executionsIn(mlir::Operation *op, mlir::Operation *root) {
  int64_t count = 1;
  for (mlir::scf::ForOp loop : pulse::enclosingLoops(op, root))
    count *= pulse::constantTripCount(loop).value_or(0);
  return count;
}

/// What the ops in `root` whose length is not known yet can add to it, or
/// the first one with no bound.
static std::pair<int64_t, mlir::Operation *> stretchIn(mlir::Operation *root) {
  int64_t total = 0;
  mlir::Operation *unbounded = nullptr;
  root->walk([&](mlir::Operation *op) {
    if (!isTimedOp(op))
      return mlir::WalkResult::advance();
    int64_t stretch = stretchOf(op);
    if (stretch < 0) {
      unbounded = op;
      return mlir::WalkResult::interrupt();
    }
    if (stretch > 0)
      total += stretch * executionsIn(op, root);
    return mlir::WalkResult::advance();
  });
  return {total, unbounded};
}

/// The feedback latency the schedule of `root` waits after a readout
/// something in it depends on, `latency` under its resources; 0 if nothing
/// does.
static int64_t waitedLatency(mlir::Operation *root, int64_t latency) {
  if (latency <= 0)
    return 0;
  bool used = root->walk([](pulse::ReadoutOp readout) {
                    return readout.getResult().use_empty()
                               ? mlir::WalkResult::advance()
                               : mlir::WalkResult::interrupt();
                  })
                  .wasInterrupted();
  return used ? latency : 0;
}

/// The clause of a budget's error that names `latency`, if any.
static void nameLatency(mlir::InFlightDiagnostic &diagnostic, int64_t latency) {
  if (latency > 0)
    diagnostic << ", waiting " << latency
               << " cycles of feedback latency after the readouts its "
                  "operations depend on";
}

/// Fail with the error for `what` (a kernel, quoted), whose executions take
/// `length` cycles as scheduled, against `budget`, if it can take longer;
/// `latency`: what it waits after its readouts (see waitedLatency).
static mlir::LogicalResult checkBudget(mlir::Operation *root,
                                       mlir::Operation *at,
                                       llvm::StringRef what, int64_t length,
                                       int64_t budget, int64_t latency) {
  auto [stretch, unbounded] = stretchIn(root);
  if (unbounded && mlir::isa<classical::WaitOp>(unbounded))
    return unbounded->emitError()
           << what << " has max_duration_cycles=" << budget
           << ", but it waits for an answer of a blocking classical function, "
              "which has no bound; use latency_mode=\"bounded\" or "
              "\"timeout\", or drop max_duration_cycles";
  if (unbounded)
    return unbounded->emitError()
           << what << " has max_duration_cycles=" << budget
           << ", but this wait in it has no highest length; bound it with "
              "wait(..., bounds=(lowest, highest)), or drop "
              "max_duration_cycles";
  if (length + stretch <= budget)
    return mlir::success();
  auto diagnostic = at->emitError()
                    << what << " takes " << (stretch ? "up to " : "")
                    << length + stretch << " cycles";
  if (stretch)
    diagnostic << " (" << length << " as scheduled, its variable waits add up "
               << "to " << stretch << ")";
  diagnostic << ", more than max_duration_cycles=" << budget;
  nameLatency(diagnostic, latency);
  diagnostic << "; let its operations overlap, shorten them or raise "
                "max_duration_cycles";
  return mlir::failure();
}

static mlir::LogicalResult checkBlockBudget(pulse::ScheduleBlockOp block,
                                            int64_t latency) {
  auto budget = block->getAttrOfType<mlir::IntegerAttr>("max_duration_cycles");
  if (!budget)
    return mlir::success();
  std::string what =
      "each call of kernel '" + block.getCallee().value_or("block").str() + "'";
  return checkBudget(block, block, what, shapeOf(block).length, budget.getInt(),
                     waitedLatency(block, latency));
}

/// Whether `op` plays on a line the control system `clocks` gives another
/// controller than `controller` (-1: any).
static bool playsElsewhere(mlir::Operation *op,
                           const pulse::ClockAttributes &clocks,
                           int64_t controller, pulse::AllocationTracer &trace) {
  if (controller < 0 ||
      !mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op))
    return false;
  mlir::Operation *allocation = trace(op->getOperand(0));
  if (!allocation)
    return false;
  int64_t qudit = pulse::quditOf(allocation);
  const llvm::SmallVector<int64_t> &owners =
      mlir::isa<pulse::GetDriveLineOp>(allocation) ? clocks.driveOwner
                                                   : clocks.readoutOwner;
  if (qudit < 0 || qudit >= static_cast<int64_t>(owners.size()))
    return false;
  return owners[qudit] >= 0 && owners[qudit] != controller;
}

/// The end of the last execution of the last op of the scheduled
/// `function`, with each line delayed as `clocks` (its module's; none when
/// null) delays it; with `controller`, of the ops on its lines and on lines
/// of no known controller.
static int64_t programLength(mlir::func::FuncOp function,
                             const pulse::ClockAttributes *clocks = nullptr,
                             int64_t controller = -1) {
  std::optional<pulse::LineDelays> lines;
  if (clocks && clocks->delayed())
    lines.emplace(*clocks);
  pulse::AllocationTracer trace;
  int64_t length = 0;
  function.walk([&](mlir::Operation *op) {
    if (!isTimedOp(op) ||
        (clocks && playsElsewhere(op, *clocks, controller, trace)))
      return;
    std::optional<int64_t> start = pulse::startCycles(op);
    if (!start)
      return;
    auto loops = pulse::enclosingLoops(op, function);
    llvm::SmallVector<int64_t, 2> last;
    for (mlir::scf::ForOp loop : loops) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      if (trips == 0)
        return;
      last.push_back(trips - 1);
    }
    int64_t end = *start;
    for (auto [k, period] : llvm::zip(last, pulse::periodCycles(op)))
      end += std::max<int64_t>(0, k * period);
    end += pulse::scheduledDuration(op).value_or(0);
    if (lines)
      end += lines->of(op);
    length = std::max(length, end);
  });
  return length;
}

/// The longest program a schedule counts, in cycles, as building the module
/// checks it before its lines are delayed; a delay is at most as long, so
/// the delayed length does not overflow.
constexpr int64_t kLongestProgram = int64_t(1) << 60;

/// Fail if the line delays of `clocks` take the scheduled `function` past
/// kLongestProgram.
static mlir::LogicalResult
checkDelayedLength(mlir::func::FuncOp function,
                   const pulse::ClockAttributes &clocks) {
  if (!clocks.delayed() || programLength(function, &clocks) <= kLongestProgram)
    return mlir::success();
  return function->emitError()
         << "the program lasts more than 2**60 cycles with its lines delayed "
            "(their delay_ns), more than a schedule counts";
}

static mlir::LogicalResult checkProgramBudget(mlir::func::FuncOp function,
                                              int64_t latency) {
  auto module = function->getParentOfType<mlir::ModuleOp>();
  std::optional<int64_t> budget =
      module ? pulse::programBudget(module) : std::nullopt;
  if (!budget)
    return mlir::success();
  std::string what =
      "kernel '" + module.getSymName().value_or("main").str() + "'";
  return checkBudget(function, function, what, programLength(function), *budget,
                     latency);
}

/// The module attribute that lists each controller's longest program, from
/// Controller.max_program_duration: by entry, a dictionary of its
/// `controller` index, `name`, `cycles` (program cycles) and `duration` (as
/// declared, for messages).
inline constexpr llvm::StringLiteral kProgramLimitsAttr =
    "pulse.program_limits";

/// Fail if the scheduled `function` lasts longer than a controller's
/// longest program: to the end of its last op on the lines that controller
/// plays (every line, with one), line delays included. `latency`: the
/// feedback latency the schedule waits after a readout something depends
/// on (0: none), which the message names.
static mlir::LogicalResult
checkProgramLimits(mlir::func::FuncOp function,
                   const pulse::ClockAttributes &clocks, int64_t latency) {
  auto module = function->getParentOfType<mlir::ModuleOp>();
  auto limits = module
                    ? module->getAttrOfType<mlir::ArrayAttr>(kProgramLimitsAttr)
                    : nullptr;
  if (!limits)
    return mlir::success();
  // A wait whose length a call sets counts at its length as scheduled: each
  // call's values are scheduled, and checked, again.
  for (mlir::Attribute entry : limits) {
    auto fields = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
    auto controller =
        fields ? fields.getAs<mlir::IntegerAttr>("controller") : nullptr;
    auto cycles = fields ? fields.getAs<mlir::IntegerAttr>("cycles") : nullptr;
    auto name = fields ? fields.getAs<mlir::StringAttr>("name") : nullptr;
    auto duration =
        fields ? fields.getAs<mlir::StringAttr>("duration") : nullptr;
    if (!controller || !cycles || !name || !duration)
      continue;
    int64_t length = programLength(function, &clocks, controller.getInt());
    if (length <= cycles.getInt())
      continue;
    auto diagnostic = function->emitError()
                      << "kernel '" << module.getSymName().value_or("main")
                      << "' lasts " << length << " cycles, more than "
                      << name.getValue() << "'s max_program_duration of "
                      << duration.getValue() << " (" << cycles.getInt()
                      << " cycles)";
    nameLatency(diagnostic, latency);
    diagnostic << "; let its operations overlap, shorten them or raise "
                  "max_program_duration";
    return mlir::failure();
  }
  return mlir::success();
}

// ── Driver ───────────────────────────────────────────────────────────────

/// Rounds of a region's scheduling that may peel several iterations at once.
constexpr int64_t kWidePeelRounds = 8;

/// Rounds of a region's scheduling that may regroup a loop's iterations.
constexpr int64_t kRegroupRounds = 4;

/// Schedule `region`, its schedule blocks done, in rounds until its schedule
/// is affine.
template <bool Gridded>
static mlir::LogicalResult
scheduleRounds(mlir::Region &region, mlir::Operation *root, Resources resources,
               bool isAlap, bool classical, const Timelines &timelines,
               const pulse::LineGrid &grid,
               const pulse::Controllers &controllers) {
  // Each round either writes an affine schedule, removes an iteration from
  // a loop, (a bounded number of times) regroups one, or schedules the same
  // round once more without the alap last end, so this terminates.
  int64_t widePeels = 0, regroups = 0, headPeels = 0;
  llvm::DenseMap<mlir::Operation *, int64_t> regroupedBy;
  // Schedule the next round without the alap last end (see below).
  bool uncapped = false;
  while (true) {
    Scheduler<Gridded> scheduler(region, resources, classical, grid,
                                 controllers, &timelines);
    scheduler.mayRegroup = regroups < kRegroupRounds;
    scheduler.mayPeelHead = headPeels < kRegroupRounds;
    scheduler.regroupedBy = regroupedBy;
    scheduler.uncapped = std::exchange(uncapped, false);
    auto regrouped = [&] {
      if (scheduler.headPeel) {
        ++headPeels;
        for (int64_t i = 0; i < scheduler.headPeel->iterations; ++i)
          peelFront(scheduler.headPeel->loop);
        return true;
      }
      if (!scheduler.regroup)
        return false;
      ++regroups;
      auto [loop, factor] = *scheduler.regroup;
      int64_t &by = regroupedBy.try_emplace(loop, 1).first->second;
      by *= factor;
      unrollBy(loop, factor);
      return true;
    };
    if (mlir::failed(scheduler.forward()))
      return mlir::failure();
    if (regrouped())
      continue;
    if (isAlap) {
      if (mlir::failed(scheduler.backward()))
        return mlir::failure();
      if (regrouped())
        continue;
      // A round under the last end that cannot be written or does not keep
      // to it is scheduled again without it, as before it; only where that
      // cannot be written either does the forward schedule stand.
      if (scheduler.lastEnd != kNever && !scheduler.keepsLastEnd()) {
        uncapped = true;
        continue;
      }
      // Handing out resources in reverse program order can need time before
      // zero, take an answer before it is due, or leave a request kept at
      // its forward start no slot; the forward resource schedule is always
      // feasible.
      bool feasible = !scheduler.skippedNegative && !scheduler.answeredEarly &&
                      !(scheduler.slotsOverrun && scheduler.heldTooMany()) &&
                      llvm::all_of(scheduler.samples, [](const auto &entry) {
                        return llvm::all_of(entry.second, [](const Sample &s) {
                          return s.start >= 0;
                        });
                      });
      if (!feasible) {
        scheduler = Scheduler<Gridded>(region, resources, classical, grid,
                                       controllers, &timelines);
        scheduler.mayRegroup = false;
        scheduler.mayPeelHead = false;
        if (mlir::failed(scheduler.forward()))
          return mlir::failure();
      }
    }

    // The first problem of each loop that neither holds nor is held by the
    // loop of one before it: peeling or unrolling one leaves the others'
    // loops in place, and copies of a loop that peeling made (one per
    // iteration peeled around it) are then all fixed in one round. With a
    // problem for each copy, whether a loop is apart is a walk up from it,
    // not a comparison with every problem.
    llvm::SmallVector<NotAffine> problems;
    llvm::DenseSet<mlir::Operation *> problemLoops, aroundProblems;
    auto apart = [&](mlir::scf::ForOp loop) {
      if (aroundProblems.contains(loop))
        return false;
      for (mlir::Operation *op = loop; op; op = op->getParentOp())
        if (problemLoops.contains(op))
          return false;
      return true;
    };
    forEachScheduled(region, [&](mlir::Operation *op) {
      auto it = scheduler.samples.find(op);
      llvm::ArrayRef<Sample> samples;
      if (it != scheduler.samples.end())
        samples = it->second;
      auto problem = writeTiming(op, root, samples, scheduler.extrapolations,
                                 scheduler.covers);
      if (!problem || !apart(problem->loop))
        return;
      problems.push_back(*problem);
      problemLoops.insert(problem->loop);
      mlir::Operation *around = problem->loop;
      while (around && aroundProblems.insert(around).second)
        around = around->getParentOp();
    });
    if (problems.empty())
      if (auto problem = periodMismatch(region, root))
        problems.push_back(*problem);
    if (problems.empty())
      break;
    // Each round schedules the whole region again: a few may peel several
    // iterations of a loop, the rest at most one or unroll.
    bool wide = llvm::any_of(problems, [](const NotAffine &problem) {
      return problem.peelFront + problem.peelBack > 1;
    });
    bool narrow = wide && ++widePeels > kWidePeelRounds;
    for (NotAffine &problem : problems) {
      if (narrow && problem.peelFront + problem.peelBack > 1)
        problem.peelFront = problem.peelBack = 0;
      for (int64_t i = 0; i < problem.peelFront; ++i)
        peelFront(problem.loop);
      for (int64_t i = 0; i < problem.peelBack; ++i)
        peelBack(problem.loop);
      if (!problem.peelFront && !problem.peelBack)
        unrollFully(problem.loop);
    }
  }

  return mlir::success();
}

/// Whether some of the several controllers of the module of `op` block on
/// calls.
static bool holdsControllers(mlir::Operation *op) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return false;
  pulse::ControlSystemAttr system = pulse::controlSystemOf(module);
  return system && system.getControllers().size() > 1 &&
         pulse::blocksOnCalls(module);
}

static mlir::LogicalResult
scheduleRegion(mlir::Region &region, mlir::Operation *root, Resources resources,
               bool isAlap, bool classical, const Timelines &timelines,
               const pulse::LineGrid &grid,
               const pulse::Controllers &controllers) {
  auto blocks = placeableBlocks(region);
  if (mlir::failed(blocks))
    return mlir::failure();
  for (pulse::ScheduleBlockOp block : *blocks) {
    if (mlir::failed(
            scheduleRegion(block.getBody(), block, latencyOf(resources),
                           block.getPolicy() == pulse::SchedulePolicy::ALAP,
                           classical, timelines, grid, controllers)))
      return mlir::failure();
    writeShape(block, classical, readinessOf(resources, controllers));
    if (mlir::failed(
            checkBlockBudget(block, readoutLatencyOf(resources, controllers))))
      return mlir::failure();
  }
  // Lines on several controllers are traced under every policy: an outcome
  // reaches each controller at its own time.
  bool traced = grid.enabled() || controllers.enabled();
  if (mlir::failed(
          traced
              ? scheduleRounds<true>(region, root, resources, isAlap, classical,
                                     timelines, grid, controllers)
              : scheduleRounds<false>(region, root, resources, isAlap,
                                      classical, timelines, grid, controllers)))
    return mlir::failure();

  // Peeling may have copied blocks, so collect them again.
  for (pulse::ScheduleBlockOp block : outermostBlocks(region))
    inlineScheduleBlock(block);
  return mlir::success();
}

static mlir::LogicalResult schedule(mlir::func::FuncOp function,
                                    Resources resources, bool isAlap) {
  // A declaration has no body to schedule.
  if (function.isExternal())
    return mlir::success();
  // Nor do the passes place a pulse.call yet.
  if (mlir::failed(pulse::refuseCalls(function))) [[unlikely]]
    return mlir::failure();
  // The passes time the ops of one block, in their order.
  if (!function.getBody().hasOneBlock())
    return function.emitError(
        "the schedule passes time a function of one block; this one "
        "branches between several");
  if (resources.enabled &&
      (resources.maxDrives <= 0 || resources.maxReadouts <= 0))
    return function.emitError("resource limits must be positive");
  if (resources.enabled) {
    auto blocks = outermostBlocks(function.getBody());
    if (!blocks.empty())
      return blocks.front().emitError(
          "resource-constrained scheduling (rcp, alap_rcp) does not support "
          "kernels with their own schedule yet; drop schedule= from the "
          "called kernels, or flatten them with flatten_schedules=True");
  }
  mlir::FailureOr<bool> annotated = classical::annotateClassicalCalls(function);
  if (mlir::failed(annotated))
    return mlir::failure();
  bool classical = *annotated;
  bool blocksOnCalls =
      pulse::blocksOnCalls(function->getParentOfType<mlir::ModuleOp>());
  if (classical)
    for (pulse::ScheduleBlockOp block : outermostBlocks(function.getBody())) {
      mlir::WalkResult found = block.walk([blocksOnCalls](mlir::Operation *op) {
        if (op->hasAttr("classical.variable")) {
          op->emitError(classical::kVariableWaitInBlock);
          return mlir::WalkResult::interrupt();
        }
        if (mlir::isa<classical::WaitOp>(op) &&
            op->hasAttr("classical.blocking")) {
          op->emitError(classical::kBlockingWaitInBlock);
          return mlir::WalkResult::interrupt();
        }
        if (blocksOnCalls && mlir::isa<classical::SubmitOp>(op)) {
          op->emitError(classical::kBlockingCallInBlock);
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (found.wasInterrupted())
        return mlir::failure();
    }
  Timelines timelines;
  if (classical && TimelineAnalysis(function, resources).run(timelines) &&
      mlir::failed(classical::annotateClassicalCalls(function)))
    return mlir::failure();
  // Ops that read a value a measurement selects wait for it, as ops that
  // read an answer do.
  bool data = classical || function->hasAttr("pulse.selected_values");
  const pulse::ClockAttributes clocks(
      function->getParentOfType<mlir::ModuleOp>());
  // An outcome waits for its readout to end on the delayed line, its fine
  // delay rounded up.
  resources.readoutDelays = clocks.readoutReady;
  int64_t waited = waitedLatency(
      function, readoutLatencyOf(resources, pulse::Controllers(clocks)));
  if (mlir::failed(scheduleRegion(
          function.getBody(), function, resources, isAlap, data, timelines,
          pulse::LineGrid(clocks), pulse::Controllers(clocks))) ||
      mlir::failed(checkProgramBudget(function, waited)) ||
      mlir::failed(checkDelayedLength(function, clocks)) ||
      mlir::failed(checkProgramLimits(function, clocks, waited)))
    return mlir::failure();
  bool placed = pulse::applyLineDelays(function, clocks);
  function->setAttr(
      pulse::kScheduleStateAttr,
      pulse::ScheduleStateAttr::get(function.getContext(),
                                    placed ? pulse::ScheduleState::Placed
                                           : pulse::ScheduleState::Timed));
  return mlir::success();
}

template <bool IsAlap>
struct PulseSchedulePass
    : public mlir::PassWrapper<PulseSchedulePass<IsAlap>,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseSchedulePass<IsAlap>)

  llvm::StringRef getArgument() const override {
    return IsAlap ? "pulse-schedule-alap" : "pulse-schedule-asap";
  }

  llvm::StringRef getDescription() const override {
    return IsAlap ? "Assign a dependency-correct ALAP pulse schedule"
                  : "Assign a dependency-correct ASAP pulse schedule";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    // The pass marks every function with a schedule state, an attribute of
    // the pulse dialect, even a function that holds no pulse operation.
    registry.insert<pulse::PulseDialect>();
  }

  PulseSchedulePass() = default;
  explicit PulseSchedulePass(int64_t readoutLatency) {
    resources.readoutLatency = readoutLatency;
  }

  void runOnOperation() override {
    if (mlir::failed(schedule(this->getOperation(), resources, IsAlap)))
      this->signalPassFailure();
  }

  /// No lanes: only the feedback latency every readout's outcome waits.
  Resources resources;
};

/// pulse-schedule-asap and pulse-schedule-alap as cudaq-pulse-opt runs
/// them, with the feedback latency as an option. The extension builds
/// PulseSchedulePass instead: options cost a command-line parser per pass.
template <bool IsAlap>
struct PulseScheduleOptionsPass
    : public mlir::PassWrapper<PulseScheduleOptionsPass<IsAlap>,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseScheduleOptionsPass<IsAlap>)

  PulseScheduleOptionsPass() = default;
  PulseScheduleOptionsPass(const PulseScheduleOptionsPass &other)
      : mlir::PassWrapper<PulseScheduleOptionsPass<IsAlap>,
                          mlir::OperationPass<mlir::func::FuncOp>>(other) {}

  llvm::StringRef getArgument() const override {
    return IsAlap ? "pulse-schedule-alap" : "pulse-schedule-asap";
  }
  llvm::StringRef getDescription() const override {
    return IsAlap ? "Assign a dependency-correct ALAP pulse schedule"
                  : "Assign a dependency-correct ASAP pulse schedule";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<pulse::PulseDialect>();
  }
  void runOnOperation() override {
    Resources resources;
    resources.readoutLatency = readoutLatency;
    if (mlir::failed(schedule(this->getOperation(), resources, IsAlap)))
      this->signalPassFailure();
  }

  mlir::Pass::Option<int64_t> readoutLatency{
      *this, "readout-latency",
      llvm::cl::desc("Cycles after a readout ends before its outcome is "
                     "usable"),
      llvm::cl::init(0)};
};

struct PulseScheduleRcpPass
    : public mlir::PassWrapper<PulseScheduleRcpPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseScheduleRcpPass)

  PulseScheduleRcpPass(int64_t maxDrives, int64_t maxReadouts,
                       int64_t readoutLatency, int64_t switchPenalty,
                       bool isAlap)
      : resources{true,           maxDrives,     maxReadouts,
                  readoutLatency, switchPenalty, {}},
        isAlap(isAlap) {}

  llvm::StringRef getArgument() const override { return "pulse-schedule-rcp"; }
  llvm::StringRef getDescription() const override {
    return "Assign a resource-constrained pulse schedule";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    // The pass marks every function with a schedule state, an attribute of
    // the pulse dialect, even a function that holds no pulse operation.
    registry.insert<pulse::PulseDialect>();
  }
  void runOnOperation() override {
    if (mlir::failed(schedule(getOperation(), resources, isAlap)))
      signalPassFailure();
  }

  Resources resources;
  bool isAlap;
};

/// pulse-schedule-rcp as cudaq-pulse-opt runs it, with the machine model as
/// options. The extension builds PulseScheduleRcpPass instead: options cost
/// a command-line parser per pass.
struct PulseScheduleRcpOptionsPass
    : public mlir::PassWrapper<PulseScheduleRcpOptionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseScheduleRcpOptionsPass)

  PulseScheduleRcpOptionsPass() = default;
  PulseScheduleRcpOptionsPass(const PulseScheduleRcpOptionsPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override { return "pulse-schedule-rcp"; }
  llvm::StringRef getDescription() const override {
    return "Assign a resource-constrained pulse schedule";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    // The pass marks every function with a schedule state, an attribute of
    // the pulse dialect, even a function that holds no pulse operation.
    registry.insert<pulse::PulseDialect>();
  }
  void runOnOperation() override {
    Resources resources{true,           maxDrives,     maxReadouts,
                        readoutLatency, switchPenalty, {}};
    if (mlir::failed(schedule(getOperation(), resources, alap)))
      signalPassFailure();
  }

  Option<int64_t> maxDrives{*this, "max-drives",
                            llvm::cl::desc("Drives that may play at once"),
                            llvm::cl::init(4)};
  Option<int64_t> maxReadouts{*this, "max-readouts",
                              llvm::cl::desc("Readouts that may play at once"),
                              llvm::cl::init(2)};
  Option<int64_t> readoutLatency{
      *this, "readout-latency",
      llvm::cl::desc(
          "Cycles after a readout before its line and result are ready"),
      llvm::cl::init(0)};
  Option<int64_t> switchPenalty{
      *this, "switch-penalty",
      llvm::cl::desc("Cycles a drive or readout unit stays busy after each op"),
      llvm::cl::init(0)};
  Option<bool> alap{*this, "alap",
                    llvm::cl::desc("Place each op as late as it may go "
                                   "(alap_rcp)"),
                    llvm::cl::init(false)};
};

/// Drop every schedule block, so that its ops are scheduled with the
/// enclosing ones.
struct PulseInlineScheduleBlocksPass
    : public mlir::PassWrapper<PulseInlineScheduleBlocksPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseInlineScheduleBlocksPass)

  llvm::StringRef getArgument() const override {
    return "pulse-inline-schedule-blocks";
  }
  llvm::StringRef getDescription() const override {
    return "Inline pulse.schedule_block ops, ignoring their policies";
  }
  void runOnOperation() override {
    llvm::SmallVector<pulse::ScheduleBlockOp> blocks;
    getOperation().walk(
        [&](pulse::ScheduleBlockOp block) { blocks.push_back(block); });
    for (pulse::ScheduleBlockOp block : blocks)
      inlineBody(block);
  }
};

/// Whether `loop` holds a classical call or a variable wait.
static bool holdsClassical(mlir::scf::ForOp loop) {
  return loop.getBody()
      ->walk([](mlir::Operation *op) {
        if (op->getDialect() && op->getDialect()->getNamespace() == "classical")
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

/// Unroll every loop that holds classical calls, so each execution of a call
/// is an op of its own: simulation binds answers one execution at a time.
struct PulseUnrollClassicalLoopsPass
    : public mlir::PassWrapper<PulseUnrollClassicalLoopsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseUnrollClassicalLoopsPass)

  llvm::StringRef getArgument() const override {
    return "pulse-unroll-classical-loops";
  }
  llvm::StringRef getDescription() const override {
    return "Fully unroll loops that hold classical calls";
  }
  void runOnOperation() override {
    while (true) {
      llvm::SmallVector<mlir::scf::ForOp> loops;
      getOperation().walk<mlir::WalkOrder::PreOrder>(
          [&](mlir::scf::ForOp loop) {
            if (!holdsClassical(loop))
              return mlir::WalkResult::advance();
            loops.push_back(loop);
            return mlir::WalkResult::skip();
          });
      if (loops.empty())
        return;
      for (mlir::scf::ForOp loop : loops) {
        if (!pulse::constantTripCount(loop)) {
          loop.emitError("cannot unroll a loop with non-constant bounds");
          return signalPassFailure();
        }
        unrollFully(loop);
      }
    }
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseUnrollClassicalLoopsPass() {
  return std::make_unique<PulseUnrollClassicalLoopsPass>();
}

std::unique_ptr<mlir::Pass>
createPulseScheduleAsapPass(int64_t readoutLatency) {
  return std::make_unique<PulseSchedulePass<false>>(readoutLatency);
}

std::unique_ptr<mlir::Pass>
createPulseScheduleAlapPass(int64_t readoutLatency) {
  return std::make_unique<PulseSchedulePass<true>>(readoutLatency);
}

std::unique_ptr<mlir::Pass> createPulseScheduleAsapOptionsPass() {
  return std::make_unique<PulseScheduleOptionsPass<false>>();
}

std::unique_ptr<mlir::Pass> createPulseScheduleAlapOptionsPass() {
  return std::make_unique<PulseScheduleOptionsPass<true>>();
}

std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap) {
  return std::make_unique<PulseScheduleRcpPass>(
      maxDrives, maxReadouts, readoutLatency, switchPenalty, isAlap);
}

std::unique_ptr<mlir::Pass> createPulseScheduleRcpOptionsPass() {
  return std::make_unique<PulseScheduleRcpOptionsPass>();
}

std::unique_ptr<mlir::Pass> createPulseInlineScheduleBlocksPass() {
  return std::make_unique<PulseInlineScheduleBlocksPass>();
}
} // namespace pulse
