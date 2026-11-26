// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Checks on the classical calls of a scheduled pulse program:
//
// - every request and answer fits a slot of its link;
// - a call with a deadline can meet it: its submission and budgeted latency
//   end within `deadline_ns` of the start of the submission;
// - every answer is taken no earlier than it is due (S7), and a future is
//   waited on in the loop iteration that submitted it;
// - a call with a reaction time reacts in it: its answer is due within
//   `reaction_deadline_ns` of the end of the readouts its request carries;
// - at no time does a link hold more requests than `slots` (S8);
// - the submissions on one link go out one after another, in program order
//   (S9), also inside branches (S11);
// - a call starts after the readouts and answers it sends end (S10), and no
//   op in a branch on an answer starts before it is taken (S12): the
//   scheduler bounds such a branch by the answer alone.
//
// Where lines wait independently (`classical.timeline`), a link's slots and
// order are per timeline: each holds no more requests than its share and
// sends its own in program order.
//
// A long loop is checked from one copy of its body, its requests moving by
// their periods: the diagnostics are those of checking every iteration.

#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/ClassicalTiming.h"
#include "cudaq-pulse/Analysis/Timing.h"

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace pulse {
/// What the waits of run-time length of a scheduled `module` add under its
/// controllers' longest programs, if it lists any (PulseProgramLimits.cpp).
mlir::LogicalResult checkWaitLimits(mlir::ModuleOp module);
} // namespace pulse

namespace {

/// Loops of at least this many trips are checked from one copy of their
/// body.
constexpr int64_t kSummarizedTrips = 64;

/// The times the calls of a function are counted and ordered at: starts,
/// and the lengths added to them, of at most this many cycles either way,
/// so that the sums and differences of a few of them fit 64 bits. A call
/// with a time outside it is reported, not computed with. (A loaded
/// module's times are within 2^50.)
constexpr int64_t kMostCycles = int64_t(1) << 61;

static bool withinCycles(int64_t cycles) {
  return cycles >= -kMostCycles && cycles <= kMostCycles;
}

/// pulse::scheduledStart within kMostCycles: none when `op` has no start,
/// or its start in `iteration` is outside it or does not fit 64 bits. An
/// op outside loops needs no periods.
static std::optional<int64_t> startWithin(mlir::Operation *op,
                                          llvm::ArrayRef<int64_t> iteration) {
  std::optional<int64_t> start = pulse::startCycles(op);
  if (!start)
    return std::nullopt;
  int64_t time = *start;
  if (!iteration.empty())
    for (auto [k, period] : llvm::zip(iteration, pulse::periodCycles(op))) {
      int64_t moved;
      if (__builtin_mul_overflow(k, period, &moved))
        return std::nullopt;
      if (__builtin_add_overflow(time, moved, &time))
        return std::nullopt;
    }
  return withinCycles(time) ? std::optional<int64_t>(time) : std::nullopt;
}

/// The executions of a submission over the iterations of one loop, or its
/// one execution: in iteration k of `count` it starts at start + k *
/// period, holds its link for `duration` and a slot for `busy`. The runs of
/// one loop's body share a `group`; a single execution has none (-1).
struct Run {
  int64_t start, period, count, duration, busy;
  int64_t group;
  mlir::Operation *op;
};

/// A link, its instance on one controller (-1: the link shared), and one
/// timeline's share of that (-1: all of it).
using LinkKey = std::tuple<int64_t, int64_t, int64_t>;

/// The readouts and answers whose values reach `values` through ops without
/// regions and the results of branches; not through a loop's values.
static void sentValues(mlir::ValueRange values,
                       llvm::SmallVectorImpl<mlir::Operation *> &found) {
  llvm::SmallVector<mlir::Value, 8> work;
  for (mlir::Value value : values) {
    // Most are a readout's or an answer's own.
    mlir::Operation *def = value.getDefiningOp();
    if (mlir::isa_and_nonnull<pulse::ReadoutOp, classical::WaitOp>(def))
      found.push_back(def);
    else
      work.push_back(value);
  }
  if (work.empty())
    return;
  llvm::SmallDenseSet<mlir::Value, 8> seen;
  while (!work.empty()) {
    mlir::Value value = work.pop_back_val();
    if (mlir::isa<classical::LinkType>(value.getType()) ||
        !seen.insert(value).second)
      continue;
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      continue;
    if (mlir::isa<pulse::ReadoutOp, classical::WaitOp>(def)) {
      found.push_back(def);
      continue;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
      unsigned index = mlir::cast<mlir::OpResult>(value).getResultNumber();
      work.push_back(branch.thenYield().getOperand(index));
      if (!branch.getElseRegion().empty())
        work.push_back(branch.elseYield().getOperand(index));
      work.push_back(branch.getCondition());
      continue;
    }
    if (def->getNumRegions() == 0)
      work.append(def->getOperands().begin(), def->getOperands().end());
  }
}

/// Every execution of every submission, per link, in program order. Both
/// arms of a branch are listed, so capacity is checked for the worst case.
/// The first loop around a submission with at least `summarizeFrom` trips
/// (none when 0) is kept as runs; every other loop is walked iteration by
/// iteration.
///
/// With `Ordering` (a function with a branch), the walk also checks the
/// order of the calls a branch on an answer relies on: the scheduler bounds
/// such a branch by the answer alone, not by the measurements the call, or
/// an earlier call on its link, sent. That holds when
///
/// - S10: a call starts after every readout and every answer it sends ends;
/// - S11: a call starts after the one before it on its link (in program
///   order, on its path) ends;
/// - S12: no op in a branch on an answer starts before the answer is taken.
///
/// S9 checks S11 outside branches. Starts are affine in the iteration, so
/// these are checked at each loop's first two iterations and its last, as
/// pulse-verify checks lines: a loop kept as runs is walked again at the
/// other two, recording nothing.
struct Submissions {
  explicit Submissions(int64_t summarizeFrom) : summarizeFrom(summarizeFrom) {}

  int64_t summarizeFrom;
  std::map<LinkKey, std::vector<Run>> perLink, ordered;
  bool failed = false;

  /// Collect the submissions of `block`, checking their order with
  /// `Ordering` (a function with a branch).
  template <bool Ordering>
  void collect(mlir::Block &block, llvm::SmallVectorImpl<int64_t> &iteration,
               bool inBranch) {
    for (mlir::Operation &op : block) {
      if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
        auto start = startWithin(&op, iteration);
        int64_t duration = pulse::scheduledDuration(&op).value_or(1);
        int64_t budget = classical::intAttr(&op, "classical.budget_cycles");
        if (!start || !withinCycles(duration) || !withinCycles(budget)) {
          if (pulse::hasStart(&op))
            past(submit);
          continue;
        }
        LinkKey device{classical::intAttr(&op, "classical.device"),
                       classical::intAttr(&op, classical::kInstanceAttr, -1),
                       classical::intAttr(&op, classical::kTimelineAttr, -1)};
        if (!Ordering || recording) {
          int64_t period = 0;
          if (level) {
            iteration[*level] = 1;
            auto next = startWithin(&op, iteration);
            iteration[*level] = 0;
            if (!next) {
              past(submit);
              continue;
            }
            period = *next - *start;
          }
          Run run{
              *start, period, level ? trips : 1, duration, duration + budget,
              group,  &op};
          perLink[device].push_back(run);
          if (!inBranch)
            ordered[device].push_back(run);
        }
        if (Ordering && !skipped)
          orderSubmit(submit, *start, duration, device, inBranch, iteration);
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        int64_t count = pulse::constantTripCount(loop).value_or(0);
        iteration.push_back(0);
        if ((!Ordering || recording) && !level && summarizeFrom > 0 &&
            count >= summarizeFrom) {
          level = iteration.size() - 1;
          group = groups++;
          trips = count;
          collect<Ordering>(*loop.getBody(), iteration, inBranch);
          level.reset();
          group = -1;
          if (Ordering && !skipped) {
            recording = false;
            for (int64_t k : {int64_t{1}, count - 1}) {
              iteration.back() = k;
              collect<Ordering>(*loop.getBody(), iteration, inBranch);
            }
            recording = true;
          }
        } else if (!Ordering) {
          for (int64_t k = 0; k < count; ++k) {
            iteration.back() = k;
            collect<Ordering>(*loop.getBody(), iteration, inBranch);
          }
        } else {
          for (int64_t k = 0; k < count; ++k) {
            if (!recording && k == 2 && count > 3)
              k = count - 1;
            iteration.back() = k;
            bool skip = k >= 2 && k != count - 1;
            skipped += skip;
            collect<Ordering>(*loop.getBody(), iteration, inBranch);
            skipped -= skip;
          }
        }
        iteration.pop_back();
      } else if constexpr (!Ordering) {
        bool branch = inBranch || mlir::isa<mlir::scf::IfOp>(op);
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            collect<Ordering>(nested, iteration, branch);
      } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        if (!skipped) {
          orderIf(branch, iteration);
        } else {
          for (mlir::Region &region : op.getRegions())
            for (mlir::Block &nested : region)
              collect<Ordering>(nested, iteration, true);
        }
      } else if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                           classical::WaitOp>(op)) {
        if (answered != kNone && !skipped)
          if (auto start = startIn(&op, iteration))
            afterAnswer(&op, *start);
      } else {
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            collect<Ordering>(nested, iteration, inBranch);
      }
    }
  }

private:
  void error(mlir::Operation *op, const llvm::Twine &message) {
    if (reported.insert(op).second)
      op->emitError(message);
    failed = true;
  }

  /// S12.
  void afterAnswer(mlir::Operation *op, int64_t start) {
    if (start < answered)
      error(op, "an operation in a branch on an answer starts at " +
                    llvm::Twine(start) + ", before the answer is taken at " +
                    llvm::Twine(answered));
  }

  /// A call with a time the check does not compute with.
  void past(classical::SubmitOp submit) {
    error(submit, "the call of '" + submit.getCallee() +
                      "' starts, or lasts, past the cycles a schedule counts "
                      "in");
  }

  /// The start of `op`, an operation the order of calls depends on, in the
  /// iteration walked; one with a time the check does not compute with is
  /// reported, not passed over.
  std::optional<int64_t> startIn(mlir::Operation *op,
                                 llvm::ArrayRef<int64_t> iteration) {
    std::optional<int64_t> start = startWithin(op, iteration);
    if (!start && pulse::hasStart(op))
      pastOperation(op);
    return start;
  }

  void pastOperation(mlir::Operation *op) {
    error(op, "an operation the order of calls depends on starts, or lasts, "
              "past the cycles a schedule counts in");
  }

  /// When `op`, a readout or an answer the op walked depends on, ends in the
  /// iteration walked: the loops around it are among those around the op.
  std::optional<int64_t> endOf(mlir::Operation *op,
                               llvm::ArrayRef<int64_t> iteration) {
    size_t depth = 0;
    for (mlir::Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp())
      depth += mlir::isa<mlir::scf::ForOp>(parent);
    if (depth > iteration.size())
      return std::nullopt;
    std::optional<int64_t> start = startIn(op, iteration.take_front(depth));
    if (!start)
      return std::nullopt;
    int64_t duration = pulse::scheduledDuration(op).value_or(0);
    if (!withinCycles(duration)) {
      pastOperation(op);
      return std::nullopt;
    }
    return *start + duration;
  }

  void orderSubmit(classical::SubmitOp submit, int64_t start, int64_t duration,
                   const LinkKey &key, bool inBranch,
                   llvm::ArrayRef<int64_t> iteration) {
    mlir::Operation *op = submit;
    afterAnswer(op, start);
    // S10.
    llvm::SmallVector<mlir::Operation *, 4> sent;
    sentValues(submit.getArgs(), sent);
    for (mlir::Operation *from : sent) {
      std::optional<int64_t> end = endOf(from, iteration);
      if (end && start < *end) {
        bool readout = mlir::isa<pulse::ReadoutOp>(from);
        error(op, "the call of '" + submit.getCallee() + "' starts at " +
                      llvm::Twine(start) + ", before the " +
                      (readout ? "readout" : "answer") + " it sends ends at " +
                      llvm::Twine(*end));
        break;
      }
    }
    // S11, inside branches.
    int64_t *previous = nullptr;
    for (auto &[link, end] : linkEnds)
      if (link == key)
        previous = &end;
    if (!previous) {
      linkEnds.push_back({key, start});
      previous = &linkEnds.back().second;
    } else if (start < *previous && inBranch) {
      error(op, "submission on link " + llvm::Twine(std::get<0>(key)) +
                    " starts at " + llvm::Twine(start) +
                    ", before the one before it on its path ends at " +
                    llvm::Twine(*previous));
    }
    *previous = std::max(*previous, start + duration);
  }

  void orderIf(mlir::scf::IfOp branch,
               llvm::SmallVectorImpl<int64_t> &iteration) {
    int64_t saved = answered;
    llvm::SmallVector<mlir::Operation *, 4> sent;
    sentValues(branch.getCondition(), sent);
    for (mlir::Operation *from : sent)
      if (mlir::isa<classical::WaitOp>(from))
        if (std::optional<int64_t> taken = endOf(from, iteration))
          answered = std::max(answered, *taken);
    auto before = linkEnds;
    collect<true>(*branch.thenBlock(), iteration, true);
    if (!branch.getElseRegion().empty()) {
      auto afterThen = std::move(linkEnds);
      linkEnds = std::move(before);
      collect<true>(*branch.elseBlock(), iteration, true);
      for (auto &[key, end] : afterThen) {
        bool merged = false;
        for (auto &[link, mine] : linkEnds)
          if (link == key) {
            mine = std::max(mine, end);
            merged = true;
          }
        if (!merged)
          linkEnds.push_back({key, end});
      }
    }
    answered = saved;
  }

  static constexpr int64_t kNone = std::numeric_limits<int64_t>::min();

  /// Whether runs are recorded: not when a loop kept as runs is walked
  /// again for the order.
  bool recording = true;
  /// How many loops around the op walked are at an iteration the order is
  /// not checked at.
  int skipped = 0;
  /// The level of `iteration` of the loop being kept as runs, its trips and
  /// its runs' group.
  std::optional<size_t> level;
  int64_t trips = 0, group = -1, groups = 0;
  /// The end of the latest submission on each link, on the path walked.
  llvm::SmallVector<std::pair<LinkKey, int64_t>, 4> linkEnds;
  /// The latest answer a branch around the op walked is on.
  int64_t answered = kNone;
  llvm::DenseSet<mlir::Operation *> reported;
};

static int64_t floorDiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  return a % b != 0 && a < 0 ? q - 1 : q;
}

static int64_t ceilDiv(int64_t a, int64_t b) { return -floorDiv(-a, b); }

/// Whether a run's last execution starts within kMostCycles, as its first
/// does: every execution then starts within it, and ends within three
/// times it.
static bool fits(const Run &run) {
  int64_t span, last;
  if (__builtin_mul_overflow(run.count - 1, run.period, &span))
    return false;
  if (__builtin_add_overflow(run.start, span, &last))
    return false;
  return withinCycles(last);
}

/// The most requests `runs` hold at once and the earliest time they do (0
/// when none is ever held), from every execution: a sweep over interval
/// ends, each half-open (a slot freed at t can be reused at t). The
/// executions of a run that does not move from one iteration to the next
/// are all at one time, and counted there together.
static std::pair<int64_t, int64_t> sweepInFlight(llvm::ArrayRef<Run> runs) {
  std::vector<std::pair<int64_t, int64_t>> events;
  for (const Run &run : runs) {
    if (run.period == 0) {
      events.push_back({run.start, +run.count});
      events.push_back({run.start + run.busy, -run.count});
      continue;
    }
    for (int64_t k = 0; k < run.count; ++k) {
      int64_t start = run.start + k * run.period;
      events.push_back({start, +1});
      events.push_back({start + run.busy, -1});
    }
  }
  std::sort(events.begin(), events.end());
  int64_t inFlight = 0, worst = 0, when = 0;
  for (auto [time, delta] : events) {
    // The counts of runs that do not move can add up past 64 bits.
    if (__builtin_add_overflow(inFlight, delta, &inFlight))
      inFlight = delta > 0 ? std::numeric_limits<int64_t>::max()
                           : std::numeric_limits<int64_t>::min();
    if (inFlight > worst) {
      worst = inFlight;
      when = time;
    }
  }
  return {worst, when};
}

/// sweepInFlight from each loop's runs rather than every execution, or none
/// when that would not be cheaper. The count only rises where a request
/// starts. Past a loop's first iterations, each start of a run finds no
/// more of the loop's requests in flight than the start a period before
/// (as many until its last iterations, fewer in them): of the starts while
/// no request outside the loop is in flight, only the first is counted.
static std::optional<std::pair<int64_t, int64_t>>
summarizedInFlight(llvm::ArrayRef<Run> runs) {
  int64_t executions = 0;
  std::map<int64_t, llvm::SmallVector<const Run *, 4>> groups;
  std::vector<int64_t> singleStarts, singleEnds;
  for (const Run &run : runs) {
    executions += run.count;
    if (run.group < 0) {
      singleStarts.push_back(run.start);
      singleEnds.push_back(run.start + run.busy);
      continue;
    }
    if (run.period <= 0 || !fits(run))
      return std::nullopt;
    auto &group = groups[run.group];
    if (!group.empty() && group.front()->period != run.period)
      return std::nullopt;
    group.push_back(&run);
  }
  if (groups.empty())
    return std::nullopt;
  std::sort(singleStarts.begin(), singleStarts.end());
  std::sort(singleEnds.begin(), singleEnds.end());

  // Where each group's requests are in flight, and each single request.
  struct Span {
    int64_t begin, end, group;
  };
  std::vector<Span> spans;
  for (auto &[id, group] : groups) {
    Span span{std::numeric_limits<int64_t>::max(),
              std::numeric_limits<int64_t>::min(), id};
    for (const Run *run : group) {
      span.begin = std::min(span.begin, run->start);
      span.end = std::max(
          span.end, run->start + (run->count - 1) * run->period + run->busy);
    }
    spans.push_back(span);
  }
  for (const Run &run : runs)
    if (run.group < 0 && run.busy > 0)
      spans.push_back({run.start, run.start + run.busy, -1});

  // The starts to count at, within as much work as sweeping every
  // execution would take.
  int64_t loopRuns = static_cast<int64_t>(runs.size() - singleStarts.size());
  int64_t budget = 4 * executions / (loopRuns + 1) + 64;
  std::vector<int64_t> starts(singleStarts);
  for (auto &[id, group] : groups) {
    int64_t period = group.front()->period, trips = group.front()->count;
    for (const Run *run : group) {
      // From `first` on, the loop's requests in flight at each start of the
      // run are those at the start before, moved by a period, but those of
      // iterations past the last.
      int64_t first = 0;
      for (const Run *other : group) {
        int64_t lo =
            floorDiv(run->start - other->start - other->busy, period) + 1;
        if (lo <= floorDiv(run->start - other->start, period))
          first = std::max(first, -lo);
      }
      first = std::min(first, trips);
      std::vector<std::pair<int64_t, int64_t>> ranges;
      if (first > 0)
        ranges.push_back({0, first - 1});
      for (const Span &span : spans) {
        if (span.group == id)
          continue;
        int64_t lo =
            std::max<int64_t>(ceilDiv(span.begin - run->start, period), 0);
        int64_t hi =
            std::min(ceilDiv(span.end - run->start, period) - 1, trips - 1);
        if (lo <= hi)
          ranges.push_back({lo, hi});
      }
      std::sort(ranges.begin(), ranges.end());
      std::vector<std::pair<int64_t, int64_t>> merged;
      for (auto range : ranges) {
        if (!merged.empty() && range.first <= merged.back().second + 1)
          merged.back().second = std::max(merged.back().second, range.second);
        else
          merged.push_back(range);
      }
      int64_t steady = first;
      for (auto [lo, hi] : merged) {
        if (lo <= steady && steady <= hi)
          steady = hi + 1;
        if (hi - lo + 1 > budget - static_cast<int64_t>(starts.size()))
          return std::nullopt;
        for (int64_t k = lo; k <= hi; ++k)
          starts.push_back(run->start + k * period);
      }
      if (steady < trips)
        starts.push_back(run->start + steady * period);
    }
  }

  int64_t worst = 0, when = 0;
  for (int64_t t : starts) {
    int64_t count =
        (std::upper_bound(singleStarts.begin(), singleStarts.end(), t) -
         singleStarts.begin()) -
        (std::upper_bound(singleEnds.begin(), singleEnds.end(), t) -
         singleEnds.begin());
    for (auto &[id, group] : groups)
      for (const Run *run : group) {
        int64_t hi =
            std::min(floorDiv(t - run->start, run->period), run->count - 1);
        int64_t lo = std::max<int64_t>(
            floorDiv(t - run->start - run->busy, run->period) + 1, 0);
        count += std::max<int64_t>(0, hi - lo + 1);
      }
    if (count > worst || (count == worst && count > 0 && t < when)) {
      worst = count;
      when = t;
    }
  }
  return std::pair<int64_t, int64_t>{worst, when};
}

/// A submission that starts before the one before it in program order
/// ends: its op, its start and that end.
struct OutOfOrder {
  mlir::Operation *op;
  int64_t start, end;
};

/// The first k in [0, last] where a + k * d < 0.
static std::optional<int64_t> firstNegative(int64_t a, int64_t d,
                                            int64_t last) {
  if (last < 0 || (a >= 0 && d >= 0))
    return std::nullopt;
  int64_t k = a < 0 ? 0 : a / -d + 1;
  return k <= last ? std::optional<int64_t>(k) : std::nullopt;
}

/// The first out-of-order execution of `runs`, which are in program order.
/// A loop's runs (one group) repeat as a block, so whether each execution
/// starts before the one before it ends is affine in the iteration.
static std::optional<OutOfOrder> firstOutOfOrder(llvm::ArrayRef<Run> runs) {
  std::optional<int64_t> previousEnd;
  for (size_t i = 0; i < runs.size();) {
    size_t j = i + 1;
    if (runs[i].group >= 0)
      while (j < runs.size() && runs[j].group == runs[i].group)
        ++j;
    llvm::ArrayRef<Run> block = runs.slice(i, j - i);
    const Run &head = block.front(), &tail = block.back();
    if (previousEnd && head.start < *previousEnd)
      return OutOfOrder{head.op, head.start, *previousEnd};
    int64_t trips = head.count;
    auto m = static_cast<int64_t>(block.size());
    // The first execution out of order by its place among the block's
    // (iteration * m + position).
    std::optional<int64_t> first;
    for (int64_t p = 1; p < m; ++p) {
      const Run &a = block[p - 1], &b = block[p];
      if (auto k = firstNegative(b.start - a.start - a.duration,
                                 b.period - a.period, trips - 1))
        if (!first || *k * m + p < *first)
          first = *k * m + p;
    }
    if (auto k =
            firstNegative(head.start + head.period - tail.start - tail.duration,
                          head.period - tail.period, trips - 2))
      if (!first || (*k + 1) * m < *first)
        first = (*k + 1) * m;
    if (first) {
      int64_t k = *first / m, p = *first % m;
      const Run &b = block[p];
      const Run &a = p > 0 ? block[p - 1] : tail;
      int64_t before = p > 0 ? k : k - 1;
      return OutOfOrder{b.op, b.start + k * b.period,
                        a.start + before * a.period + a.duration};
    }
    previousEnd = tail.start + (trips - 1) * tail.period + tail.duration;
    i = j;
  }
  return std::nullopt;
}

/// The readouts whose measurements reach `values` through ops without
/// regions, into `readouts`; false if one comes from elsewhere (a branch
/// result or a loop argument), whose readout is not known statically.
static bool carriedReadouts(mlir::ValueRange values,
                            llvm::SetVector<mlir::Operation *> &readouts) {
  llvm::SmallVector<mlir::Value> work(values.begin(), values.end());
  llvm::DenseSet<mlir::Value> seen;
  while (!work.empty()) {
    mlir::Value value = work.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    mlir::Operation *def = value.getDefiningOp();
    if (def && mlir::isa<pulse::ReadoutOp>(def)) {
      readouts.insert(def);
      continue;
    }
    if (!def || def->getNumRegions() > 0) {
      if (mlir::isa<pulse::MeasurementType>(value.getType()))
        return false;
      continue;
    }
    work.append(def->getOperands().begin(), def->getOperands().end());
  }
  return true;
}

/// How many cycles longer than scheduled `op` can take: 0 for an op of
/// known length, -1 for a wait without a highest length.
static int64_t stretchOf(mlir::Operation *op) {
  if (mlir::isa<classical::WaitOp>(op))
    return op->hasAttr("classical.blocking") ? -1 : 0;
  auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
  if (!wait || !op->hasAttr("pulse.variable_wait"))
    return 0;
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>())
    if (pulse::constantIndex(conversion.getCycles()))
      return 0;
  auto highest = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles");
  if (!highest)
    return -1;
  return std::max<int64_t>(0, highest.getInt() -
                                  pulse::scheduledDuration(op).value_or(0));
}

/// The ops of a function whose length is not known until it runs, in
/// program order, for what they can add between two ops.
struct Stretches {
  explicit Stretches(mlir::func::FuncOp function) : function(function) {
    int64_t position = 0;
    function.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      order[op] = position++;
      if (int64_t stretch = stretchOf(op))
        ops.push_back({op, stretch});
    });
  }

  /// What the ops that can run after `from` and before `to` can add, per
  /// execution of the innermost loop holding both: those between them in
  /// program order, and every one in a loop that holds `to` but not
  /// `from`. -1 if one has no bound.
  int64_t between(mlir::Operation *from, mlir::Operation *to) {
    if (ops.empty())
      return 0;
    mlir::Operation *root = function, *span = nullptr;
    for (mlir::scf::ForOp loop : pulse::enclosingLoops(to, function)) {
      if (!loop->isProperAncestor(from)) {
        span = loop;
        break;
      }
      root = loop;
    }
    int64_t low = order.lookup(from), high = order.lookup(to), total = 0;
    for (auto [op, stretch] : ops) {
      int64_t at = order.lookup(op);
      bool inSpan = span && span->isProperAncestor(op);
      if (!root->isProperAncestor(op) || (!inSpan && (at <= low || at >= high)))
        continue;
      if (stretch < 0)
        return -1;
      int64_t count = 1;
      for (mlir::scf::ForOp loop : pulse::enclosingLoops(op, root))
        count *= pulse::constantTripCount(loop).value_or(0);
      total += stretch * count;
    }
    return total;
  }

  mlir::func::FuncOp function;
  llvm::DenseMap<mlir::Operation *, int64_t> order;
  llvm::SmallVector<std::pair<mlir::Operation *, int64_t>> ops;
};

/// A time in any iteration of the loops around an operation: a start, a
/// period times an iteration, and sums of them over the loops. Starts,
/// periods and trip counts are 64-bit numbers a module may state freely, so
/// the sums are kept wider, and held at the widest number either way where
/// even that would overflow.
using Wide = __int128;

static Wide wideAdd(Wide a, Wide b) {
  Wide sum;
  if (__builtin_add_overflow(a, b, &sum)) {
    Wide most = static_cast<Wide>(~static_cast<unsigned __int128>(0) >> 1);
    return b < 0 ? -most - 1 : most;
  }
  return sum;
}

static Wide wideMul(Wide a, Wide b) {
  Wide product;
  if (__builtin_mul_overflow(a, b, &product)) {
    Wide most = static_cast<Wide>(~static_cast<unsigned __int128>(0) >> 1);
    return (a < 0) != (b < 0) ? -most - 1 : most;
  }
  return product;
}

static std::string wideText(Wide value) {
  if (value >= std::numeric_limits<int64_t>::min() &&
      value <= std::numeric_limits<int64_t>::max())
    return std::to_string(static_cast<int64_t>(value));
  bool negative = value < 0;
  auto left = negative ? -static_cast<unsigned __int128>(value)
                       : static_cast<unsigned __int128>(value);
  std::string digits;
  for (; left; left /= 10)
    digits.push_back(static_cast<char>('0' + static_cast<int>(left % 10)));
  if (negative)
    digits.push_back('-');
  return std::string(digits.rbegin(), digits.rend());
}

/// The period of a timed operation in the loop at `level` around it,
/// outermost first; none stated counts as none, as in scheduledStart.
/// (Outside every loop the periods are not read at all.)
static Wide periodAt(llvm::ArrayRef<int64_t> periods, size_t level) {
  return level < periods.size() ? Wide(periods[level]) : Wide(0);
}

/// Check that every execution of `submit`, a call to `callable` with
/// `reaction_deadline_ns`, has its answer due within it of the end of the
/// latest readout it carries, or of its own start. The submission and each
/// readout start affinely in the iteration, so each difference is largest
/// at a corner of the submission's iteration box: per loop, at its last
/// iteration where the difference grows with it and at its first
/// otherwise, which is the first such corner when the loops are counted
/// with the outermost changing fastest. The latest readout is at least as
/// late as each, so the smallest of their worst cases bounds the
/// reaction. A wait between a readout and the submission whose length
/// is known only when the program runs counts at its highest length. A
/// request that leaves within `latency` (the feedback latency the schedule
/// waits) of its readouts' end could not leave sooner, and the error says
/// so.
static bool checkReaction(classical::SubmitOp submit,
                          classical::CallableOp callable,
                          mlir::func::FuncOp function, double clock,
                          Stretches &stretches, int64_t latency) {
  if (!pulse::hasStart(submit))
    return true;
  double reactionNs = callable.getReactionDeadlineNs()->convertToDouble();
  int64_t limit = static_cast<int64_t>(std::floor(reactionNs * clock + 1e-9));
  llvm::SetVector<mlir::Operation *> readouts;
  if (!carriedReadouts(submit.getArgs(), readouts)) {
    submit.emitError("'")
        << callable.getSymName()
        << "' has reaction_deadline_ns, but a measurement it carries comes out "
           "of a "
           "branch or a loop iteration, so its readout is not known; read it "
           "out where the request is made, or drop reaction_deadline_ns";
    return false;
  }
  auto loops = pulse::enclosingLoops(submit, function);
  size_t depth = loops.size();
  llvm::SmallVector<int64_t, 2> trips;
  for (mlir::scf::ForOp loop : loops) {
    trips.push_back(pulse::constantTripCount(loop).value_or(0));
    if (trips.back() == 0)
      return true;
  }
  int64_t duration = pulse::scheduledDuration(submit).value_or(1);
  int64_t budget = classical::intAttr(submit, "classical.budget_cycles");
  // (worst reaction, its iteration) from each reference: the readouts, or
  // the submission itself.
  llvm::SmallVector<mlir::Operation *> references(readouts.begin(),
                                                  readouts.end());
  if (references.empty())
    references.push_back(submit);
  std::optional<Wide> bound;
  llvm::SmallVector<int64_t, 2> worstAt;
  llvm::ArrayRef<int64_t> sent =
      depth ? pulse::periodCycles(submit) : llvm::ArrayRef<int64_t>();
  Wide due = wideAdd(Wide(*pulse::startCycles(submit)),
                     wideAdd(Wide(duration), Wide(budget)));
  for (mlir::Operation *reference : references) {
    size_t own = pulse::enclosingLoops(reference, function).size();
    if (!pulse::hasStart(reference) || own > depth)
      continue;
    int64_t stretch = reference == submit.getOperation()
                          ? 0
                          : stretches.between(reference, submit);
    if (stretch < 0) {
      submit.emitError("'")
          << callable.getSymName()
          << "' has reaction_deadline_ns, but between a readout it carries and "
             "the "
             "submission the program waits for a time with no bound (a "
             "blocking answer, or a wait without a highest length); submit "
             "before it, bound the wait with bounds=(lowest, highest), or "
             "drop reaction_deadline_ns";
      return false;
    }
    llvm::ArrayRef<int64_t> read =
        depth ? pulse::periodCycles(reference) : llvm::ArrayRef<int64_t>();
    Wide from = *pulse::startCycles(reference);
    if (reference != submit.getOperation())
      from =
          wideAdd(from, Wide(pulse::scheduledDuration(reference).value_or(0)));
    std::optional<Wide> worst = wideAdd(wideAdd(due, Wide(stretch)), -from);
    llvm::SmallVector<int64_t, 2> at(depth, 0);
    for (size_t level = 0; level < depth; ++level) {
      Wide moves = wideAdd(periodAt(sent, level),
                           -(level < own ? periodAt(read, level) : Wide(0)));
      Wide grows = wideMul(moves, Wide(trips[level] - 1));
      if (grows > 0) {
        worst = wideAdd(*worst, grows);
        at[level] = trips[level] - 1;
      }
    }
    if (worst && (!bound || *worst < *bound)) {
      bound = worst;
      worstAt = at;
    }
  }
  if (!bound || *bound <= limit)
    return true;
  auto diagnostic = submit.emitError("'")
                    << callable.getSymName() << "' cannot react in time: ";
  if (depth) {
    diagnostic << "in iteration (";
    llvm::interleaveComma(worstAt, diagnostic);
    diagnostic << ") ";
  }
  // The request left as soon as the outcomes it carries were usable.
  bool waited =
      !readouts.empty() && latency > 0 &&
      wideAdd(*bound, -wideAdd(Wide(duration), Wide(budget))) <= latency;
  diagnostic << "its answer is due " << wideText(*bound) << " cycles after "
             << (readouts.empty() ? "the submission starts"
                                  : "the readouts it carries end");
  if (waited)
    diagnostic << " (" << latency
               << " of them the feedback latency before the request can "
                  "leave)";
  diagnostic << ", but reaction_deadline_ns allows " << limit << " (slack "
             << wideText(wideAdd(Wide(limit), -*bound)) << "); "
             << (waited ? "" : "submit it sooner, ")
             << "lower max_latency_ns, use a faster link or raise "
                "reaction_deadline_ns";
  return false;
}

/// Check the classical calls of `module` (see the file comment), loops of
/// at least `summarizeFrom` trips (none when 0) from one copy of their
/// body; `latency`: the feedback latency its schedule waits after readouts.
static mlir::LogicalResult checkClassicalCalls(mlir::ModuleOp module,
                                               int64_t summarizeFrom,
                                               int64_t latency) {
  if (!classical::declaresClassicalCalls(module))
    return mlir::success();
  bool failed = false;
  double clock = classical::moduleClockGHz(module);

  // On several controllers, the most start uncertainty a function's calls
  // add to its budget (the link's when a call's is not known).
  pulse::Controllers controllers(module);
  llvm::StringMap<std::optional<double>> uncertainty;
  if (controllers.enabled()) {
    pulse::AllocationTracer trace;
    classical::TakersByValue takers;
    module.walk([&](classical::SubmitOp submit) {
      auto found =
          classical::callUncertaintyNs(submit, controllers, trace, takers);
      auto [it, inserted] = uncertainty.try_emplace(submit.getCallee(), found);
      if (!inserted && it->second)
        it->second = found
                         ? std::optional<double>(std::max(*it->second, *found))
                         : std::nullopt;
    });
  }

  for (auto callable : module.getOps<classical::CallableOp>()) {
    classical::LinkModel link =
        classical::linkModel(module, callable.getDevice());
    int64_t request = classical::requestBytes(callable);
    int64_t answer = classical::answerBytes(callable);
    if (request > link.slotBytes || answer > link.slotBytes) {
      callable.emitError(request > link.slotBytes ? "a request" : "an answer")
          << " of '" << callable.getSymName() << "' takes "
          << std::max(request, answer) << " bytes, but a slot of link "
          << callable.getDevice() << " holds " << link.slotBytes;
      failed = true;
    }
    if (auto deadline = callable.getDeadlineNs()) {
      auto calls = uncertainty.find(callable.getSymName());
      int64_t due =
          classical::submitCycles(callable, link, clock) +
          classical::budgetCycles(callable, link, clock, std::nullopt,
                                  calls == uncertainty.end() ? std::nullopt
                                                             : calls->second);
      int64_t limit = classical::ceilCycles(deadline->convertToDouble(), clock);
      if (due > limit) {
        callable.emitError("'")
            << callable.getSymName() << "' cannot meet its deadline: its "
            << "answer is due " << due << " cycles after the submission "
            << "starts, but the deadline is " << limit << " (slack "
            << limit - due
            << "); raise deadline_ns, lower max_latency_ns or use a "
               "faster link";
        failed = true;
      }
    }
  }

  bool reactions =
      llvm::any_of(module.getOps<classical::CallableOp>(),
                   [](classical::CallableOp callable) {
                     return callable.getReactionDeadlineNs().has_value();
                   });
  mlir::SymbolTable symbols(module);

  module.walk([&](mlir::func::FuncOp function) {
    if (function.isExternal())
      return;
    if (mlir::failed(classical::annotateClassicalCalls(function))) {
      failed = true;
      return;
    }
    if (reactions) {
      Stretches stretches(function);
      function.walk([&](classical::SubmitOp submit) {
        auto callable =
            symbols.lookup<classical::CallableOp>(submit.getCallee());
        if (callable && callable.getReactionDeadlineNs() &&
            !checkReaction(submit, callable, function, clock, stretches,
                           latency))
          failed = true;
      });
    }
    // S7: each answer is taken no earlier than it is due. Start times are
    // affine in the iteration, so the time between the two is least at a
    // corner of the iteration box: per loop, at its last iteration where
    // the time shrinks with it. Only when that least is negative is a
    // corner named: the first one that fails with the outermost loop
    // changing fastest, found a loop at a time from the innermost, each
    // kept at its first iteration while the loops outside it can still
    // make the time negative. The walk also finds whether the function has
    // a branch.
    bool branches = false;
    function.walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::scf::IfOp>(op)) {
        branches = true;
        return;
      }
      auto wait = mlir::dyn_cast<classical::WaitOp>(op);
      if (!wait)
        return;
      mlir::Value future = wait.getFuture();
      auto submit = future.getDefiningOp<classical::SubmitOp>();
      if (!submit) {
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(future);
        if (future.getDefiningOp<mlir::scf::ForOp>() ||
            (argument &&
             mlir::isa<mlir::scf::ForOp>(argument.getOwner()->getParentOp()))) {
          wait.emitError("the answer of '")
              << wait.getCallee()
              << "' is taken in another loop iteration than its submission";
          failed = true;
        }
        return;
      }
      if (!pulse::hasStart(submit) || !pulse::hasStart(wait))
        return;
      auto loops = pulse::enclosingLoops(wait, function);
      if (loops != pulse::enclosingLoops(submit, function)) {
        wait.emitError("the answer of '")
            << wait.getCallee()
            << "' is taken in another loop than its submission";
        failed = true;
        return;
      }
      int64_t duration = pulse::scheduledDuration(submit).value_or(1);
      int64_t budget = classical::intAttr(submit, "classical.budget_cycles");
      size_t depth = loops.size();
      llvm::SmallVector<int64_t, 4> last(depth);
      for (size_t level = 0; level < depth; ++level) {
        int64_t trips = pulse::constantTripCount(loops[level]).value_or(0);
        if (trips == 0)
          return;
        last[level] = trips - 1;
      }
      llvm::ArrayRef<int64_t> sent =
          depth ? pulse::periodCycles(submit) : llvm::ArrayRef<int64_t>();
      llvm::ArrayRef<int64_t> took =
          depth ? pulse::periodCycles(wait) : llvm::ArrayRef<int64_t>();
      Wide due = wideAdd(Wide(*pulse::startCycles(submit)),
                         wideAdd(Wide(duration), Wide(budget)));
      Wide taken = *pulse::startCycles(wait);
      // What the last iteration of each loop adds to the time between,
      // and the least the loops outside a level can add together.
      llvm::SmallVector<Wide, 4> adds(depth), outside(depth);
      Wide least = 0;
      for (size_t level = 0; level < depth; ++level) {
        outside[level] = least;
        adds[level] =
            wideMul(wideAdd(periodAt(took, level), -periodAt(sent, level)),
                    Wide(last[level]));
        if (adds[level] < 0)
          least = wideAdd(least, adds[level]);
      }
      Wide between = wideAdd(taken, -due);
      if (wideAdd(between, least) >= 0)
        return;
      for (size_t level = depth; level-- > 0;) {
        if (wideAdd(between, outside[level]) < 0)
          continue;
        between = wideAdd(between, adds[level]);
        taken =
            wideAdd(taken, wideMul(periodAt(took, level), Wide(last[level])));
        due = wideAdd(due, wideMul(periodAt(sent, level), Wide(last[level])));
      }
      wait.emitError("the answer of '")
          << wait.getCallee() << "' is taken at " << wideText(taken)
          << ", before it is due at " << wideText(due);
      failed = true;
    });

    Submissions submissions{summarizeFrom};
    llvm::SmallVector<int64_t> iteration;
    mlir::Block &body = function.getBody().front();
    if (branches)
      submissions.collect<true>(body, iteration, false);
    else
      submissions.collect<false>(body, iteration, false);
    failed |= submissions.failed;
    // A call whose last execution in a loop would start or end past what
    // 64 bits count is no schedule: it is reported, and its link's calls
    // are not counted or ordered from times that do not exist.
    for (auto &[key, runs] : submissions.perLink)
      for (const Run &run : runs)
        if (!fits(run)) {
          run.op->emitError("the call of '")
              << mlir::cast<classical::SubmitOp>(run.op).getCallee()
              << "' starts at " << run.start << " and " << run.period
              << " cycles later in each of " << run.count
              << " iterations: the last is past the cycles a schedule counts "
                 "in";
          failed = true;
          break;
        }
    // S8: requests in flight at once.
    for (auto &[key, runs] : submissions.perLink) {
      auto [device, instance, timeline] = key;
      if (runs.empty() || !llvm::all_of(runs, fits))
        continue;
      int64_t capacity =
          classical::intAttr(runs.front().op, "classical.slots", 64);
      auto summarized = summarizedInFlight(runs);
      auto [worst, when] = summarized ? *summarized : sweepInFlight(runs);
      if (worst > capacity) {
        auto diag = runs.front().op->emitError("link ");
        diag << device;
        if (instance >= 0)
          diag << " on controllers[" << instance << "]";
        diag << " holds " << worst << " requests";
        if (timeline >= 0)
          diag << " for lines that wait on their own";
        diag << " at once at " << when << ", more than ";
        if (timeline >= 0)
          diag << "the " << capacity << " of its slots they have";
        else
          diag << "its " << capacity << " slots";
        failed = true;
      }
    }
    // S9: one submission at a time on a link, in program order.
    for (auto &[key, runs] : submissions.ordered) {
      if (!llvm::all_of(runs, fits))
        continue;
      std::optional<OutOfOrder> found = firstOutOfOrder(runs);
      if (found) {
        found->op->emitError("submission on link ")
            << std::get<0>(key) << " starts at " << found->start
            << ", before the one before it in program order ends at "
            << found->end;
        failed = true;
      }
    }
  });
  return mlir::failure(failed);
}

struct PulseClassicalCheckPass
    : public mlir::PassWrapper<PulseClassicalCheckPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseClassicalCheckPass)

  llvm::StringRef getArgument() const override {
    return "pulse-classical-check";
  }
  llvm::StringRef getDescription() const override {
    return "Check the classical calls of a scheduled pulse program: slot "
           "sizes, deadlines, answer timing, link capacity and order";
  }

  PulseClassicalCheckPass() = default;
  explicit PulseClassicalCheckPass(int64_t readoutLatency)
      : readoutLatency(readoutLatency) {}

  void runOnOperation() override {
    if (mlir::failed(checkClassicalCalls(getOperation(), kSummarizedTrips,
                                         readoutLatency)) ||
        mlir::failed(pulse::checkWaitLimits(getOperation())))
      signalPassFailure();
  }

  int64_t readoutLatency = 0;
};

/// pulse-classical-check as cudaq-pulse-opt runs it, with the trips from
/// which a loop is summarized as an option. The extension builds
/// PulseClassicalCheckPass instead: options cost a command-line parser per
/// pass.
struct PulseClassicalCheckOptionsPass
    : public mlir::PassWrapper<PulseClassicalCheckOptionsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseClassicalCheckOptionsPass)

  PulseClassicalCheckOptionsPass() = default;
  PulseClassicalCheckOptionsPass(const PulseClassicalCheckOptionsPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override {
    return "pulse-classical-check";
  }
  llvm::StringRef getDescription() const override {
    return "Check the classical calls of a scheduled pulse program: slot "
           "sizes, deadlines, answer timing, link capacity and order";
  }
  void runOnOperation() override {
    if (mlir::failed(checkClassicalCalls(getOperation(), summarizeTrips,
                                         readoutLatency)) ||
        mlir::failed(pulse::checkWaitLimits(getOperation())))
      signalPassFailure();
  }

  Option<int64_t> readoutLatency{
      *this, "readout-latency",
      llvm::cl::desc("Cycles the schedule waits after a readout ends before "
                     "its outcome is usable"),
      llvm::cl::init(0)};

  Option<int64_t> summarizeTrips{
      *this, "summarize-trips",
      llvm::cl::desc("Check a loop of at least this many trips from one copy "
                     "of its body (0: every iteration)"),
      llvm::cl::init(kSummarizedTrips)};
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass>
createPulseClassicalCheckPass(int64_t readoutLatency) {
  return std::make_unique<PulseClassicalCheckPass>(readoutLatency);
}

std::unique_ptr<mlir::Pass> createPulseClassicalCheckOptionsPass() {
  return std::make_unique<PulseClassicalCheckOptionsPass>();
}
} // namespace pulse
