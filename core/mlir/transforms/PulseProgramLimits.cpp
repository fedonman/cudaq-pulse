// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// What a scheduled program's waits of run-time length add to it under the
// controllers' longest programs (pulse::checkWaitLimits), for a module
// that lists them in pulse.program_limits
// (Controller.max_program_duration). The schedule passes hold the program
// as scheduled to those limits; a wait whose length an answer sets is
// scheduled at its lowest length, and this holds the program to them with
// each such wait at its highest. Apart from the passes, whose code it does
// not share: pulse-classical-check calls it, and a module without limits
// runs the look for them and nothing else.

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

#include <algorithm>
#include <limits>

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
#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Transforms/Passes.h"

#include <optional>
#include <string>

namespace {

/// The ops a schedule times, as the passes have them.
static bool isTimedOp(mlir::Operation *op) {
  return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                   classical::SubmitOp, classical::WaitOp>(op);
}

/// How many cycles longer than scheduled one execution of a wait of
/// run-time length can take: of a wait on a line whose length an answer
/// sets, its highest length less the length it is scheduled at (0 once the
/// answer is bound to it); -1 where it has no highest length, as a wait for
/// the answer of a `blocking` classical function has none.
static int64_t stretchOf(mlir::Operation *op, bool blocking) {
  if (blocking)
    return -1;
  auto wait = mlir::dyn_cast<pulse::WaitOp>(op);
  if (!wait || !op->hasAttr("pulse.variable_wait"))
    return 0;
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>())
    if (pulse::boundNumber(conversion.getCycles()))
      return 0;
  auto highest = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles");
  if (!highest)
    return -1;
  // The passes schedule it at its lowest length.
  return std::max<int64_t>(0, highest.getInt() -
                                  classical::intAttr(op, "pulse.min_cycles"));
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

/// Call `fn(op, end)` with the end of the last execution of each timed op
/// of the scheduled `function` that runs, its line's delay included (the
/// passes have moved each op by it); with `controller`, of the ops on its
/// lines and on lines of no known controller.
template <typename Fn>
static void forEachEnd(mlir::func::FuncOp function,
                       const pulse::ClockAttributes &clocks, int64_t controller,
                       Fn fn) {
  pulse::AllocationTracer trace;
  function.walk([&](mlir::Operation *op) {
    if (!isTimedOp(op) || playsElsewhere(op, clocks, controller, trace))
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
    fn(op, end);
  });
}

/// a + b and a * b of lengths in cycles, none negative, held at the highest
/// number where they pass it.
static int64_t cappedAdd(int64_t a, int64_t b) {
  int64_t sum = 0;
  return llvm::AddOverflow(a, b, sum) ? std::numeric_limits<int64_t>::max()
                                      : sum;
}

static int64_t cappedMul(int64_t a, int64_t b) {
  int64_t product = 0;
  return llvm::MulOverflow(a, b, product) ? std::numeric_limits<int64_t>::max()
                                          : product;
}

/// A wait of the scheduled function whose length is known only when the
/// program runs: its place among the function's ops in their order, the
/// outermost loop it is in (null: none), its timeline, the start of its
/// first execution as scheduled, and how many cycles longer than scheduled
/// its executions can take together (-1: it has no highest length).
/// `blocking`: a wait for the answer of a blocking classical function, at
/// the end of which the answer is there, rather than a wait on a line whose
/// length an answer sets; it has no highest length.
struct RunTimeWait {
  mlir::Operation *op, *loop;
  int64_t place, timeline, start, stretch;
  bool blocking;
};

/// The timeline of `op`, of the `count` the schedule kept apart (one when
/// it kept none), as the scheduler has it (Scheduler::timelineOf): its own,
/// or that of the first line or tone it plays on.
static int64_t timelineIn(mlir::Operation *op, int64_t count,
                          pulse::AllocationTracer &trace) {
  if (count < 2)
    return 0;
  int64_t timeline = 0;
  if (op->hasAttr(classical::kTimelineAttr)) {
    timeline = classical::intAttr(op, classical::kTimelineAttr);
  } else {
    for (mlir::Value operand : op->getOperands())
      if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                    pulse::ToneType>(operand.getType())) {
        if (mlir::Operation *allocation = trace(operand))
          timeline = classical::intAttr(allocation, classical::kTimelineAttr);
        break;
      }
  }
  return std::clamp<int64_t>(timeline, 0, count - 1);
}

/// The outermost loop of `function` that `op` is in, if any.
static mlir::Operation *outermostLoop(mlir::Operation *op,
                                      mlir::func::FuncOp function) {
  auto loops = pulse::enclosingLoops(op, function);
  return loops.empty() ? nullptr : loops.front().getOperation();
}

/// The waits of run-time length of `function` that run, in the order of
/// its ops, with the place of every timed op in `places`; each wait with
/// its timeline of the `count` the schedule kept apart (with none kept
/// apart they are in one: such a wait then holds back every line). A wait
/// whose length a call's floats set is not one of them: each call is
/// scheduled, and checked, with its values.
static llvm::SmallVector<RunTimeWait>
runTimeWaits(mlir::func::FuncOp function, int64_t count,
             llvm::DenseMap<mlir::Operation *, int64_t> &places) {
  llvm::SmallVector<RunTimeWait> waits;
  pulse::AllocationTracer trace;
  int64_t place = 0;
  // Timed ops hold no op, so the walk meets them in the function's order.
  function.walk([&](mlir::Operation *op) {
    if (!isTimedOp(op))
      return;
    places[op] = place++;
    bool blocking =
        mlir::isa<classical::WaitOp>(op) && op->hasAttr("classical.blocking");
    if (!blocking && !op->hasAttr("classical.variable"))
      return;
    int64_t stretch = stretchOf(op, blocking);
    std::optional<int64_t> start = pulse::startCycles(op);
    if (stretch == 0 || !start)
      return;
    // Each execution adds its own: a loop's body once per trip. (In an arm
    // of an if, where pulse-verify refuses such a wait, it counts as if the
    // arm always ran.)
    auto loops = pulse::enclosingLoops(op, function);
    for (mlir::scf::ForOp loop : loops) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      if (trips == 0)
        return;
      if (stretch > 0)
        stretch = cappedMul(stretch, trips);
    }
    waits.push_back({op, loops.empty() ? nullptr : loops.front().getOperation(),
                     place - 1, timelineIn(op, count, trace), *start, stretch,
                     blocking});
  });
  return waits;
}

/// The waits of run-time length of one timeline, in the order of the
/// function's ops: `added[k]` is what the first k of them add together,
/// `unbounded` the first without a highest length (the size: none), and
/// `starts[k]` the earliest start, as scheduled, of those from the k-th on.
struct TimelineWaits {
  llvm::SmallVector<const RunTimeWait *> waits;
  llvm::SmallVector<int64_t> added, starts;
  size_t unbounded = 0;

  void close() {
    added.assign(waits.size() + 1, 0);
    starts.assign(waits.size() + 1, std::numeric_limits<int64_t>::max());
    unbounded = waits.size();
    for (size_t k = 0; k < waits.size(); ++k) {
      added[k + 1] =
          cappedAdd(added[k], std::max<int64_t>(0, waits[k]->stretch));
      if (waits[k]->stretch < 0 && unbounded == waits.size())
        unbounded = k;
    }
    for (size_t k = waits.size(); k-- > 0;)
      starts[k] = std::min(starts[k + 1], waits[k]->start);
  }
};

/// The timelines the schedule kept apart in `function`: one more than the
/// highest classical.timeline it wrote on the function's ops, 0 where it
/// wrote none (it then keeps none apart).
static int64_t timelinesOf(mlir::func::FuncOp function) {
  int64_t count = 0;
  function.walk([&](mlir::Operation *op) {
    if (op->hasAttr(classical::kTimelineAttr))
      count =
          std::max(count, classical::intAttr(op, classical::kTimelineAttr) + 1);
  });
  return count;
}

/// pulse::checkWaitLimits for one scheduled `function` of the module whose
/// lines `clocks` gives to its controllers.
///
/// A wait of run-time length is scheduled at its lowest length and can last
/// its highest, and what it adds moves the ops of its timeline that come
/// after it, and no other op (the timelines of the schedule passes: the
/// lines a sync or a value ties to the wait). An op comes before the wait,
/// and is not moved, when it is before it in the function's order, ends by
/// the wait's first start as scheduled, and shares no loop with it; every
/// other op of the timeline counts as after it. So an op on a controller's
/// lines can end as late as it is scheduled to plus what the waits of its
/// timeline that it is not before add, and the controller's part of the
/// program lasts to the latest of those ends. Such a wait with no highest
/// length, or a wait for the answer of a blocking classical function, where
/// an op on the controller's lines is not before it, has no bound to hold
/// to the limit: it is an error, as under a timing budget. (A wait on a
/// line is itself an op of that line; a wait for an answer, which every
/// controller counts, is not after itself.)
static mlir::LogicalResult checkFunction(mlir::func::FuncOp function,
                                         mlir::ArrayAttr limits,
                                         const pulse::ClockAttributes &clocks,
                                         llvm::StringRef kernel) {
  int64_t count = timelinesOf(function);
  llvm::DenseMap<mlir::Operation *, int64_t> places;
  llvm::SmallVector<RunTimeWait> waits = runTimeWaits(function, count, places);
  // Without such a wait the program lasts what it is scheduled to, which
  // the schedule passes have held to the limits.
  if (waits.empty())
    return mlir::success();
  llvm::SmallVector<TimelineWaits> byTimeline(std::max<int64_t>(1, count));
  for (const RunTimeWait &wait : waits)
    byTimeline[wait.timeline].waits.push_back(&wait);
  for (TimelineWaits &timeline : byTimeline)
    timeline.close();
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
    // The op that can end last: its end as scheduled and what the waits
    // add to it.
    int64_t length = 0, stretch = 0;
    const RunTimeWait *unbounded = nullptr;
    {
      pulse::AllocationTracer trace;
      forEachEnd(
          function, clocks, controller.getInt(),
          [&](mlir::Operation *op, int64_t end) {
            const TimelineWaits &timeline =
                byTimeline[timelineIn(op, count, trace)];
            int64_t place = places.lookup(op);
            // The waits up to the op in the function's order, itself
            // included, move it.
            size_t k = std::partition_point(
                           timeline.waits.begin(), timeline.waits.end(),
                           [&](const RunTimeWait *wait) {
                             return wait->place < place ||
                                    (wait->place == place && !wait->blocking);
                           }) -
                       timeline.waits.begin();
            int64_t added = timeline.added[k];
            const RunTimeWait *open = timeline.unbounded < k
                                          ? timeline.waits[timeline.unbounded]
                                          : nullptr;
            // Of the later ones, those it does not end before, and those
            // in a loop with it, whose later trips come after them.
            mlir::Operation *loop = k < timeline.waits.size()
                                        ? outermostLoop(op, function)
                                        : nullptr;
            if (end > timeline.starts[k] || loop)
              for (const RunTimeWait *wait :
                   llvm::ArrayRef(timeline.waits).drop_front(k)) {
                if (wait->op == op ||
                    (end <= wait->start && !(loop && loop == wait->loop)))
                  continue;
                if (wait->stretch < 0)
                  open = open ? open : wait;
                else
                  added = cappedAdd(added, wait->stretch);
              }
            if (open && (!unbounded || open->place < unbounded->place))
              unbounded = open;
            if (cappedAdd(end, added) > cappedAdd(length, stretch)) {
              length = end;
              stretch = added;
            }
          });
    }
    if (unbounded) {
      auto diagnostic = unbounded->op->emitError()
                        << "kernel '" << kernel << "' runs under "
                        << name.getValue() << "'s max_program_duration of "
                        << duration.getValue();
      if (unbounded->blocking)
        return diagnostic
               << ", but it waits for an answer of a blocking classical "
                  "function, which has no bound; use latency_mode=\"bounded\" "
                  "or \"timeout\", or drop max_program_duration";
      return diagnostic
             << ", but this wait in it has no highest length; bound it with "
                "wait(..., bounds=(lowest, highest)), or drop "
                "max_program_duration";
    }
    int64_t longest = cappedAdd(length, stretch);
    if (longest <= cycles.getInt())
      continue;
    // As scheduled it fits (the schedule passes checked that), so it is
    // the waits that take it past the limit.
    return function->emitError()
           << "kernel '" << kernel << "' lasts up to " << longest << " cycles ("
           << length << " as scheduled, the waits of run-time length that "
           << "hold it back add up to " << stretch << "), more than "
           << name.getValue() << "'s max_program_duration of "
           << duration.getValue() << " (" << cycles.getInt()
           << " cycles); let its operations overlap, shorten them or raise "
              "max_program_duration";
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult pulse::checkWaitLimits(mlir::ModuleOp module) {
  auto limits = module->getAttrOfType<mlir::ArrayAttr>("pulse.program_limits");
  if (!limits)
    return mlir::success();
  const pulse::ClockAttributes clocks(module);
  llvm::StringRef kernel = module.getSymName().value_or("main");
  for (auto function : module.getOps<mlir::func::FuncOp>())
    if (!function.isExternal() &&
        mlir::failed(checkFunction(function, limits, clocks, kernel)))
      return mlir::failure();
  return mlir::success();
}
