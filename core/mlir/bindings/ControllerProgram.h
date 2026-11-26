// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The walk behind ScheduledKernel.export_to_controller() and the controller
// limit checks, included by bindings.cpp inside its anonymous namespace.
//
// The walk lists every timed op, sync and classical call with the frame of
// its tone and the branches it lies in. It runs in one of three ways:
//
// - Every: once per loop iteration (format /1).
// - Sampled: a loop whose iterations repeat (the same frequencies from
//   iteration 1 on, every op moving by one period, no iteration overlapping
//   the next, no reference into the previous iteration) is walked at
//   iterations 0, 1 and the last only; the records of iteration 1 stand for
//   iterations 1 to n - 2. The limit checks run on these records.
// - Rolled: such a loop becomes one record followed by its body at its first
//   iteration, with each phase's step per iteration; an iteration 0 that
//   differs is peeled off first (format /2).
//
// A loop that does not repeat, and one holding a segment boundary, is walked
// at every iteration in every way, so all three describe the same program.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallPtrSet.h"

#include "ControllerExpressions.h"

// Record kinds are compared byte by byte, not with std::strcmp: its cost
// depends on where the two strings sit relative to a page end (its vector
// path takes a slower way near one), so the same comparison cost more or
// less as the library's strings moved. Both are kept out of line, so that
// the inliner's choices elsewhere in this file stay as they were.

/// Whether the record kind `kind` is the name `name`: a comparison of known
/// length, which stops at the first byte that differs (at the latest at the
/// end of `kind`, whose terminator differs from a byte of a longer name).
template <size_t N>
[[gnu::noinline]] bool isKind(const char *kind, const char (&name)[N]) {
  for (size_t i = 0; i < N; ++i)
    if (kind[i] != name[i])
      return false;
  return true;
}

/// Whether the record kinds `a` and `b` are the same.
[[gnu::noinline]] inline bool sameKind(const char *a, const char *b) {
  for (;; ++a, ++b) {
    if (*a != *b)
      return false;
    if (!*a)
      return true;
  }
}

struct ControllerRecord {
  /// A branch, its arm, the readout and answer records its condition is
  /// computed from, its test, for the test "expression" the walk's
  /// expression of the condition, whose leaves are places in those lists
  /// (else -1), and for the test "answer" the result of the answer it is
  /// (else -1).
  using Condition =
      std::tuple<int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>,
                 std::string, int32_t, int64_t>;

  const char *kind = "";
  mlir::Operation *op = nullptr;
  int64_t line = -1;
  int64_t qudit = -1;
  std::optional<int64_t> start;
  std::optional<int64_t> duration;
  // The iteration of each enclosing loop; -1 for a loop kept rolled.
  std::vector<int64_t> iteration;
  double frequency = std::numeric_limits<double>::quiet_NaN();
  double phase = std::numeric_limits<double>::quiet_NaN();
  std::vector<Condition> conditions;
  std::vector<int64_t> joined;
  std::vector<int64_t> refs;
  std::string callee;
  std::string mode;
  double timeout = std::numeric_limits<double>::quiet_NaN();
  // A frame update: its change (shift_phase, ...) and its value, an
  // expression of the walk.
  llvm::StringRef update;
  int32_t value = -1;
  int64_t segment = 0;
  // What the event has that is known only at run time: its index in the
  // walk's `known`, else -1.
  int32_t known = -1;
  // Sampled: the executions this record stands for, and (loop instance, 1
  // for its iteration 1, 2 for its last) for each sampled loop around it.
  int64_t weight = 1;
  std::vector<std::pair<int64_t, int8_t>> samples;
  // Rolled: the phase's and the segment's step per iteration of each rolled
  // loop around it, outermost first. A loop record has its iterations,
  // first iteration, period, the number of records of its body after it,
  // and the end of its last iteration.
  struct Step {
    double phase;
    int64_t segment;
  };
  std::vector<Step> steps;
  int64_t trips = 0;
  int64_t first = 0;
  int64_t period = 0;
  int64_t size = 0;
  std::optional<int64_t> end;

  std::optional<int64_t> stop() const {
    if (end)
      return end;
    if (start && duration)
      return *start + *duration;
    return std::nullopt;
  }
  bool played() const {
    return kind[0] == 'd' || (kind[0] == 'r' && kind[2] == 'a');
  }
  bool frameUpdate() const { return !update.empty(); }
};

struct ControllerWalk : ControllerExpressions {
  using WaveformNode =
      std::tuple<std::string, std::vector<double>, std::vector<int64_t>>;
  using Condition = ControllerRecord::Condition;
  enum class Walk { Every, Sampled, Rolled };

  /// A tone's frame as the simulator tracks it (pulse::ToneState): `phase`
  /// at `timeNs`, the end of its latest drive or the place of a later frame
  /// change; a tone detuned from its frame frequency gains phase from there
  /// on.
  struct Frame {
    double frequencyHz = 0.0;
    double frameFrequencyHz = 0.0;
    double phase = 0.0;
    double timeNs = 0.0;

    double phaseAt(double timeNs) const {
      double detuning = frequencyHz - frameFrequencyHz;
      if (detuning == 0.0)
        return phase;
      return phase + detuning * (2.0 * M_PI) * 1.0e-9 * (timeNs - this->timeNs);
    }
    /// The frame at a frame change at `start` cycles, if that is later: it
    /// runs at its old frequency until then.
    void advanceTo(std::optional<int64_t> start, double clockGHz) {
      if (!start)
        return;
      double changeNs = static_cast<double>(*start) / clockGHz;
      if (changeNs > timeNs) {
        phase = phaseAt(changeNs);
        timeNs = changeNs;
      }
    }
    /// Whether two arms leave the frame the same: without a detuning, the
    /// time of the last drive does not matter.
    bool operator==(const Frame &other) const {
      return same(frequencyHz, other.frequencyHz) &&
             same(frameFrequencyHz, other.frameFrequencyHz) &&
             same(phase, other.phase) &&
             (same(timeNs, other.timeNs) || frequencyHz == frameFrequencyHz);
    }
  };

  ControllerWalk(double clockGHz, Walk walk) : clockGHz(clockGHz), walk(walk) {}

  double clockGHz;
  Walk walk;
  /// Sampled: a loop of more than this many trips whose iterations play
  /// the same but not one period apart (repeatsOutOfStep), in a module that
  /// computes nothing at run time, is not walked (`declined`): the check's
  /// summarizing walk keeps it rolled from this many trips on.
  static constexpr int64_t kDeclineAfter = 64;
  bool declined = false;
  llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
  std::vector<ControllerRecord> result;

  /// With `track`, the arguments of `main` each record's tone frequency,
  /// frame phase and waveform depend on, parallel to `result`. A qref
  /// argument stands for its qudit's native frequency (see nativeOffset).
  bool track = false;
  using Dependencies = std::array<std::vector<int64_t>, 3>;
  std::vector<Dependencies> dependencies;
  /// With `track`, parallel to `result`: the qref argument (see
  /// nativeOffset) whose native frequency a record's tone plays (its line was
  /// never retuned), or -1.
  std::vector<int64_t> natives;
  /// What a tone's frequency, the frequency of its frame (where it
  /// started) and its phase depend on.
  struct FrameDependencies {
    std::vector<int64_t> frequency, frame, phase;
    bool retuned = false;
  };
  std::map<mlir::Operation *, FrameDependencies> frameDependencies;

  /// The arguments of `main` *value* is computed from.
  static void argumentsOf(mlir::Value value, std::vector<int64_t> &out) {
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(v)) {
        mlir::Block *owner = argument.getOwner();
        if (owner->isEntryBlock() &&
            mlir::isa_and_nonnull<mlir::func::FuncOp>(owner->getParentOp()))
          out.push_back(argument.getArgNumber());
        continue;
      }
      mlir::Operation *def = v.getDefiningOp();
      if (def && def->getNumRegions() == 0)
        work.append(def->getOperands().begin(), def->getOperands().end());
    }
  }

  static void merge(std::vector<int64_t> &into,
                    const std::vector<int64_t> &from) {
    into.insert(into.end(), from.begin(), from.end());
    std::sort(into.begin(), into.end());
    into.erase(std::unique(into.begin(), into.end()), into.end());
  }

  FrameDependencies &frameDependenciesOf(mlir::Value tone) {
    mlir::Operation *root = toneRoots(tone);
    auto [it, inserted] = frameDependencies.try_emplace(root);
    if (inserted && root) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root) &&
          !root->hasAttr("frequency_hz")) {
        argumentsOf(root->getOperand(0), it->second.frequency);
        for (int64_t &argument : it->second.frequency)
          argument += nativeOffset(root);
      } else if (root->getName().getStringRef() == "pulse.tone" &&
                 root->getNumOperands() == 2) {
        argumentsOf(root->getOperand(0), it->second.frequency);
        argumentsOf(root->getOperand(1), it->second.phase);
      }
      merge(it->second.frequency, {});
      merge(it->second.phase, {});
      it->second.frame = it->second.frequency;
    }
    return it->second;
  }

  /// The dependencies and native tone of drive or readout `op`.
  LLVM_ATTRIBUTE_NOINLINE void trackPlayed(mlir::Operation *op) {
    Dependencies &found = dependencies.back();
    FrameDependencies &frame = frameDependenciesOf(op->getOperand(2));
    mlir::Operation *root = toneRoots(op->getOperand(2));
    if (!frame.retuned && root &&
        mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root) &&
        !root->hasAttr("frequency_hz"))
      if (auto argument =
              mlir::dyn_cast<mlir::BlockArgument>(root->getOperand(0)))
        if (argument.getOwner()->isEntryBlock() &&
            mlir::isa_and_nonnull<mlir::func::FuncOp>(
                argument.getOwner()->getParentOp()))
          natives.back() = argument.getArgNumber() + nativeOffset(root);
    found[0] = frame.frequency;
    found[1] = frame.phase;
    // A retuned tone gains phase at its detuning from the frame.
    if (frame.retuned) {
      merge(found[1], frame.frequency);
      merge(found[1], frame.frame);
    }
    argumentsOf(op->getOperand(1), found[2]);
    merge(found[2], {});
  }

  /// What frame change `op` (named `name`) makes its tone depend on.
  LLVM_ATTRIBUTE_NOINLINE void trackFrameChange(mlir::Operation *op,
                                                llvm::StringRef name) {
    FrameDependencies &frame = frameDependenciesOf(op->getOperand(0));
    std::vector<int64_t> found;
    argumentsOf(op->getOperand(1), found);
    bool phase = name == "pulse.shift_phase" || name == "pulse.set_phase";
    auto &field = phase ? frame.phase : frame.frequency;
    if (name.starts_with("pulse.set_"))
      field.clear();
    merge(field, found);
    frame.retuned |= !phase;
  }

  pulse::AllocationTracer trace;
  pulse::ToneRoots toneRoots;
  llvm::SmallVector<int64_t, 2> iteration;
  std::vector<Condition> branches;
  int64_t nextBranch = 0;
  llvm::DenseMap<mlir::Operation *, Frame> frames;
  // The time each line has reached (the end of its latest event), or none
  // when an event on it is not scheduled.
  llvm::DenseMap<mlir::Operation *, std::optional<int64_t>> reached;
  // While the arms of scf.ifs are walked (ifDepth of them): each change to
  // frames and reached, with what it replaced, oldest first. A branch undoes
  // the changes of its then arm before its else arm, and merges after it
  // what either arm changed, instead of copying every tone's frame and every
  // line's reach at each branch.
  struct FrameUndo {
    mlir::Operation *root;
    bool had;
    Frame old;
  };
  struct ReachedUndo {
    mlir::Operation *allocation;
    bool had;
    std::optional<int64_t> old;
  };
  llvm::SmallVector<FrameUndo, 16> frameUndo;
  llvm::SmallVector<ReachedUndo, 16> reachedUndo;
  int ifDepth = 0;
  /// What an arm left of what it changed (or, of what only the other arm
  /// changed, what it had before), each root and allocation once, as notes
  /// whose `had` and `old` say whether it left one and what. An arm changes
  /// few as a rule, so these are lists, and small sets find a root or an
  /// allocation in them by a linear search until they hold many.
  struct ArmLeft {
    llvm::SmallVector<FrameUndo, 4> frames;
    llvm::SmallVector<ReachedUndo, 4> reached;
    llvm::SmallPtrSet<mlir::Operation *, 8> roots, allocations;
  };
  // Sampled: the sampled loops open now.
  std::vector<std::pair<int64_t, int8_t>> sampling;
  int64_t nextInstance = 0;
  llvm::DenseMap<mlir::Operation *, bool> boundaries;
  // The segment of each event, when the program has classical calls. A loop
  // holding a segment boundary is sampled at every iteration, so that the
  // walk visits the boundaries in program order; one rolled counts the
  // boundaries of the iterations it does not walk (SegmentWalk::skip).
  std::optional<SegmentWalk> segments;

  static Frame initialFrame(mlir::Operation *root) {
    if (!root)
      return {};
    if (auto frequency = root->getAttrOfType<mlir::FloatAttr>("frequency_hz"))
      return {frequency.getValueAsDouble(), frequency.getValueAsDouble(), 0.0,
              0.0};
    // A runtime qudit's line: its frequency is the carrier's.
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root))
      return {kUnknown, kUnknown, 0.0, 0.0};
    if (root->getName().getStringRef() == "pulse.tone" &&
        root->getNumOperands() == 2) {
      double frequency = numberOf(root->getOperand(0));
      return {frequency, frequency, numberOf(root->getOperand(1)), 0.0};
    }
    return {};
  }

  /// The frame of `tone`'s root, from its initial frame. A caller that
  /// changes it passes `change`, for an scf.if arm to undo.
  Frame &frameOf(mlir::Value tone, bool change = false) {
    mlir::Operation *root = toneRoots(tone);
    auto [it, inserted] = frames.try_emplace(root);
    if (inserted) {
      it->second = initialFrame(root);
      if (ifDepth > 0)
        frameUndo.push_back({root, false, {}});
    } else if (change && ifDepth > 0) {
      frameUndo.push_back({root, true, it->second});
    }
    return it->second;
  }

  /// Sets the frame of `root`, noting what it was within an scf.if arm.
  void setFrame(mlir::Operation *root, const Frame &frame) {
    auto [it, inserted] = frames.try_emplace(root, frame);
    if (ifDepth > 0)
      frameUndo.push_back({root, !inserted, inserted ? Frame() : it->second});
    if (!inserted)
      it->second = frame;
  }

  /// Sets the time `allocation`'s line has reached, noting what it was
  /// within an scf.if arm.
  void setReached(mlir::Operation *allocation, std::optional<int64_t> time) {
    auto [it, inserted] = reached.try_emplace(allocation, time);
    reachedTo(it, inserted, time);
  }

  /// Sets the time the line at `it` has reached, which `inserted` says was
  /// added just now, noting what it was within an scf.if arm.
  void reachedTo(
      llvm::DenseMap<mlir::Operation *, std::optional<int64_t>>::iterator it,
      bool inserted, std::optional<int64_t> time) {
    if (!inserted && it->second == time)
      return;
    if (ifDepth > 0)
      reachedUndo.push_back(
          {it->first, !inserted, inserted ? std::nullopt : it->second});
    it->second = time;
  }

  /// Undoes the changes to frames and reached since the marks, newest
  /// first, and returns what they had come to.
  ArmLeft undoArm(size_t frameMark, size_t reachedMark) {
    ArmLeft left;
    while (frameUndo.size() > frameMark) {
      FrameUndo undo = frameUndo.pop_back_val();
      auto found = frames.find(undo.root);
      bool has = found != frames.end();
      if (left.roots.insert(undo.root).second)
        left.frames.push_back({undo.root, has, has ? found->second : Frame()});
      if (!undo.had) {
        if (has)
          frames.erase(found);
      } else if (has) {
        found->second = undo.old;
      } else {
        frames.try_emplace(undo.root, undo.old);
      }
    }
    while (reachedUndo.size() > reachedMark) {
      ReachedUndo undo = reachedUndo.pop_back_val();
      auto found = reached.find(undo.allocation);
      bool has = found != reached.end();
      if (left.allocations.insert(undo.allocation).second)
        left.reached.push_back(
            {undo.allocation, has, has ? found->second : std::nullopt});
      if (!undo.had) {
        if (has)
          reached.erase(found);
      } else if (has) {
        found->second = undo.old;
      } else {
        reached.try_emplace(undo.allocation, undo.old);
      }
    }
    return left;
  }

  /// After both arms of an scf.if: a frame both leave alike stays, and one
  /// they leave apart is unknown from here on, a frame an arm has none of
  /// being its initial frame; a line has reached the later of the two
  /// times, or none if either arm leaves it none. Only what either arm
  /// changed can differ: `then` is what the then arm left of its changes,
  /// and the else arm's changes are noted from the marks on.
  void mergeArms(const ArmLeft &then, size_t frameMark, size_t reachedMark) {
    // Where the then arm changed nothing, it left what the else arm's
    // oldest note of a change has from before the branch.
    ArmLeft before;
    for (size_t i = frameMark; i < frameUndo.size(); ++i)
      if (!then.roots.contains(frameUndo[i].root) &&
          before.roots.insert(frameUndo[i].root).second)
        before.frames.push_back(frameUndo[i]);
    for (size_t i = reachedMark; i < reachedUndo.size(); ++i)
      if (!then.allocations.contains(reachedUndo[i].allocation) &&
          before.allocations.insert(reachedUndo[i].allocation).second)
        before.reached.push_back(reachedUndo[i]);
    if (ifDepth == 0) {
      frameUndo.resize(frameMark);
      reachedUndo.resize(reachedMark);
    }
    auto mergeFrame = [&](const FrameUndo &theirs) {
      auto found = frames.find(theirs.root);
      bool has = found != frames.end();
      if (!has && !theirs.had)
        return;
      Frame ours = has ? found->second : initialFrame(theirs.root);
      Frame other = theirs.had ? theirs.old : initialFrame(theirs.root);
      if (!(other == ours))
        ours = {kUnknown, kUnknown, kUnknown, ours.timeNs};
      else
        ours.timeNs = std::max(ours.timeNs, other.timeNs);
      if (!has)
        return setFrame(theirs.root, ours);
      if (ifDepth > 0)
        frameUndo.push_back({theirs.root, true, found->second});
      found->second = ours;
    };
    for (const FrameUndo &left : then.frames)
      mergeFrame(left);
    for (const FrameUndo &was : before.frames)
      mergeFrame(was);
    auto mergeReached = [&](const ReachedUndo &theirs) {
      if (!theirs.had)
        return;
      auto [it, inserted] = reached.try_emplace(theirs.allocation, theirs.old);
      if (inserted)
        return reachedTo(it, true, theirs.old);
      std::optional<int64_t> ours = it->second;
      reachedTo(it, false,
                ours && theirs.old
                    ? std::optional<int64_t>(std::max(*ours, *theirs.old))
                    : std::nullopt);
    };
    for (const ReachedUndo &left : then.reached)
      mergeReached(left);
    for (const ReachedUndo &was : before.reached)
      mergeReached(was);
  }

  // ── Run-time values ──────────────────────────────────────────────────
  //
  // What an event has that is known only at run time: on a tone whose frame
  // depends on run time, the frame changes since its latest drive or readout;
  // the expressions of the waveform operands, a variable wait's length and a
  // submission's operands computed from readouts and answers. Looked for
  // only where the module can compute something at run time (`runTime`).

  bool runTime = false;

  /// Whether `value` depends on a readout or an answer.
  static bool fromRunTime(mlir::Value value) {
    mlir::Operation *first = value.getDefiningOp();
    if (!first || first->hasTrait<mlir::OpTrait::ConstantLike>())
      return false;
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      mlir::Operation *def = v.getDefiningOp();
      if (!def)
        continue;
      if (mlir::isa<pulse::ReadoutOp, classical::WaitOp, mlir::scf::IfOp>(def))
        return true;
      work.append(def->getOperands().begin(), def->getOperands().end());
    }
    return false;
  }

  /// Whether a measurement reaches anything but branch conditions and
  /// submissions; true past a few ops, which only costs a slower walk.
  static bool measurementIsUsed(mlir::Value value, int depth = 0) {
    for (mlir::Operation *user : value.getUsers()) {
      if (mlir::isa<mlir::scf::IfOp, classical::SubmitOp>(user))
        continue;
      if (depth == 4 ||
          (!mlir::isa<pulse::IsSetOp, pulse::OutcomeOp>(user) &&
           !mlir::isa<mlir::arith::ArithDialect>(user->getDialect())))
        return true;
      for (mlir::Value next : user->getResults())
        if (measurementIsUsed(next, depth + 1))
          return true;
    }
    return false;
  }

  /// The tones whose frame a controller tracks at run time: those with a
  /// frame change computed from a readout or an answer, or made in a branch.
  llvm::DenseSet<mlir::Operation *> tracked;
  /// Whether any event can have something known only at run time: a tracked
  /// tone, a classical call, or a measurement used other than as a branch
  /// condition.
  bool detail = false;

  void scan(mlir::Operation *op) {
    if (mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                  pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op)) {
      if (fromRunTime(op->getOperand(1)) ||
          op->getParentOfType<mlir::scf::IfOp>()) {
        tracked.insert(toneRoots(op->getOperand(0)));
        detail = true;
      }
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      // A virtual Z folded into a drive in an arm is a frame change there.
      branch->walk([&](pulse::DriveOp drive) {
        if (drive->hasAttr("frame_phase_offset")) {
          tracked.insert(toneRoots(drive->getOperand(2)));
          detail = true;
        }
      });
    } else if (mlir::isa<classical::SubmitOp>(op)) {
      detail = true;
    } else if (mlir::isa<pulse::ReadoutOp>(op) && !detail) {
      detail = measurementIsUsed(op->getResult(2));
    }
  }

  struct FrameUpdate {
    uint64_t seq;
    llvm::StringRef op;
    int32_t value; // an expression
    std::vector<Condition> conditions;
    // Where it is made: its scheduled start, at the walk's iteration.
    std::optional<int64_t> start;
  };
  // Each tracked tone's id, in the order tones are first played, and the
  // frame changes on it since its latest drive or readout.
  llvm::DenseMap<mlir::Operation *, int64_t> toneIds;
  std::map<mlir::Operation *, std::vector<FrameUpdate>> pending;
  uint64_t nextUpdate = 0;

  /// Whether the controller's frames persist from one shot to the next
  /// (`persistent_frames` in the control system): then the walk keeps the
  /// line each tone
  /// last played on, for finalFrames().
  bool persistent = false;
  std::map<mlir::Operation *, int64_t> playedOn;

  /// The frame a tone ends the shot in: on line `line`, its frequency and
  /// its phase at `endNs`; a tracked tone has its id and the frame changes
  /// on it since its latest drive or readout instead.
  struct FinalFrame {
    int64_t line;
    double frequencyHz = kUnknown, phase = kUnknown;
    int64_t tone = -1;
    std::vector<FrameUpdate> updates;
  };

  /// The frame each tone with a line ends the shot in, by line, the tone a
  /// line allocates first; the shot ends at `endNs`.
  std::vector<FinalFrame> finalFrames(double endNs) {
    std::vector<std::pair<mlir::Operation *, int64_t>> tones;
    for (auto &[root, frame] : frames) {
      int64_t line = -1;
      if (mlir::isa_and_nonnull<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(
              root))
        line = lineOf(root);
      else if (auto found = playedOn.find(root); found != playedOn.end())
        line = found->second;
      if (line >= 0)
        tones.emplace_back(root, line);
    }
    // In the order of their ops' addresses first, as a std::map of the frames
    // listed them: the order below compares the tones of ops in different
    // blocks by address, so where it is not transitive its outcome depends
    // on the order it starts from.
    std::sort(tones.begin(), tones.end(), [](const auto &a, const auto &b) {
      return std::less<mlir::Operation *>()(a.first, b.first);
    });
    auto allocates = [](mlir::Operation *root) {
      return mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root);
    };
    std::sort(tones.begin(), tones.end(), [&](const auto &a, const auto &b) {
      if (a.second != b.second)
        return a.second < b.second;
      if (allocates(a.first) != allocates(b.first))
        return allocates(a.first);
      if (a.first->getBlock() == b.first->getBlock())
        return a.first->isBeforeInBlock(b.first);
      return a.first < b.first;
    });
    std::vector<FinalFrame> out;
    for (auto &[root, line] : tones) {
      FinalFrame &final = out.emplace_back();
      final.line = line;
      if (tracked.contains(root)) {
        final.tone =
            toneIds.try_emplace(root, static_cast<int64_t>(toneIds.size()))
                .first->second;
        if (auto found = pending.find(root); found != pending.end())
          final.updates = found->second;
        continue;
      }
      const Frame &frame = frames[root];
      final.frequencyHz = frame.frequencyHz;
      final.phase = frame.phaseAt(endNs);
    }
    return out;
  }

  /// A tone tracked at run time: the line it is on (a line's own tone's
  /// line, else the first it plays on, else -1) and the frame every shot
  /// starts it in, its frequency and phase (NaN where not known here: a
  /// runtime qudit's carrier frequency).
  struct RuntimeTone {
    int64_t line = -1;
    double frequencyHz = kUnknown, phase = kUnknown;
  };

  /// Each tracked tone with an id, by id.
  std::vector<RuntimeTone> runtimeTones() const {
    std::vector<RuntimeTone> out(toneIds.size());
    for (auto &[root, id] : toneIds) {
      RuntimeTone &tone = out[id];
      Frame reset = initialFrame(root);
      tone.frequencyHz = reset.frequencyHz;
      tone.phase = reset.phase;
      if (mlir::isa_and_nonnull<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(
              root))
        tone.line = lineOf(root);
    }
    for (const ControllerRecord &r : result)
      if (r.known >= 0 && r.line >= 0)
        if (int64_t id = known[r.known].tone; id >= 0 && out[id].line < 0)
          out[id].line = r.line;
    return out;
  }

  /// On a tracked tone (`tone` at least 0), the frame changes since its
  /// latest drive or readout and its own phase offset; the waveform
  /// operands computed from a readout or an answer, as (node, operand
  /// index, expression); a variable wait's length (at least 0) and bounds;
  /// a submission's operands.
  struct Known {
    int64_t tone = -1;
    std::vector<FrameUpdate> updates;
    std::optional<double> offset;
    std::vector<std::tuple<int64_t, int64_t, int32_t>> waveform;
    int32_t length = -1;
    std::optional<int64_t> minCycles, maxCycles;
    std::optional<std::vector<int32_t>> operands;
  };
  std::vector<Known> known;
  /// The waveform of each drive and readout runTimeOf() saw, and whether
  /// some number of it is known only at run time: the same at every
  /// execution.
  llvm::DenseMap<mlir::Operation *, std::pair<std::vector<WaveformNode>, bool>>
      played;

  /// Whether record `y`'s run-time values are record `x`'s one iteration on
  /// (see sameExpression); a tracked tone never repeats.
  bool sameKnown(const ControllerRecord &x, const ControllerRecord &y,
                 int64_t a, int64_t b) const {
    if (x.known < 0 || y.known < 0)
      return x.known == y.known;
    const Known &p = known[x.known], &q = known[y.known];
    if (p.tone >= 0 || q.tone >= 0 || p.waveform.size() != q.waveform.size() ||
        (p.length < 0) != (q.length < 0) || p.minCycles != q.minCycles ||
        p.maxCycles != q.maxCycles ||
        p.operands.has_value() != q.operands.has_value())
      return false;
    for (size_t i = 0; i < p.waveform.size(); ++i) {
      const auto &[n, k, e] = p.waveform[i];
      const auto &[m, l, f] = q.waveform[i];
      if (n != m || k != l || !sameExpression(e, f, a, b))
        return false;
    }
    if (p.length >= 0 && !sameExpression(p.length, q.length, a, b))
      return false;
    if (p.operands) {
      if (p.operands->size() != q.operands->size())
        return false;
      for (size_t i = 0; i < p.operands->size(); ++i)
        if (!sameExpression((*p.operands)[i], (*q.operands)[i], a, b))
          return false;
    }
    return true;
  }

  /// Moves the record references of record `r`'s run-time values with
  /// `shift`.
  template <typename Shift>
  void shiftKnown(const ControllerRecord &r, Shift shift) {
    if (r.known < 0)
      return;
    auto move = [&](int32_t root) {
      eachRecord(root, [&](Expression &e) { shift(e.event); });
    };
    Known &k = known[r.known];
    for (auto &[node, operand, expr] : k.waveform)
      move(expr);
    if (k.length >= 0)
      move(k.length);
    if (k.operands)
      for (int32_t expr : *k.operands)
        move(expr);
  }

  /// The expression of a branch condition computed from `readouts` and
  /// `answers`, its leaves numbering their places in those lists rather
  /// than records: a record's copy of the condition moves with the lists.
  LLVM_ATTRIBUTE_NOINLINE int32_t conditionExpression(
      mlir::Value condition, const std::vector<int64_t> &readouts,
      const std::vector<int64_t> &answers) {
    int32_t root = expression(condition);
    eachRecord(root, [&](Expression &e) {
      const auto &list = e.kind == Expression::Readout ? readouts : answers;
      e.event = llvm::find(list, e.event) - list.begin();
    });
    return root;
  }

  /// A frame change's value: its number, else its expression.
  int32_t frameValue(mlir::Value value) {
    double constant = number(value);
    return std::isnan(constant) ? expression(value)
                                : add(Expression::Number, constant);
  }

  /// A change on tracked tone `root`, made where `at` (a frame change, or a
  /// drive that folds one in) is scheduled.
  void frameUpdate(mlir::Operation *root, llvm::StringRef op, int32_t value,
                   mlir::Operation *at) {
    pending[root].push_back({nextUpdate++, op, value, branches,
                             pulse::scheduledStart(at, iteration)});
  }

  // The run-time work of the walk stays out of line, so that the per-op
  // path of a walk without it is unchanged.

  /// A frame change `op` (named `name`), on a tone that may be tracked.
  LLVM_ATTRIBUTE_NOINLINE void frameChange(mlir::Operation *op,
                                           llvm::StringRef name) {
    if (mlir::Operation *root = toneRoots(op->getOperand(0));
        tracked.contains(root))
      frameUpdate(root, name, frameValue(op->getOperand(1)), op);
  }

  // The frame changes pending before each open branch, and after its then
  // arm.
  std::vector<std::pair<decltype(pending), decltype(pending)>> pendingArms;

  LLVM_ATTRIBUTE_NOINLINE void enterBranch() {
    pendingArms.emplace_back(pending, decltype(pending){});
  }

  LLVM_ATTRIBUTE_NOINLINE void enterElse() {
    auto &[before, then] = pendingArms.back();
    then = std::move(pending);
    pending = before;
  }

  LLVM_ATTRIBUTE_NOINLINE void leaveBranch(const Condition &arms) {
    auto [before, then] = std::move(pendingArms.back());
    pendingArms.pop_back();
    mergePending(before, then, arms);
  }

  /// The frame changes pending after a branch: those of both arms, by
  /// program order. One pending before the branch that a drive or readout
  /// in one arm took is pending only on the other arm's path.
  void
  mergePending(std::map<mlir::Operation *, std::vector<FrameUpdate>> &before,
               std::map<mlir::Operation *, std::vector<FrameUpdate>> &then,
               const Condition &branch) {
    std::set<mlir::Operation *> roots;
    for (auto *side : {&before, &then, &pending})
      for (auto &entry : *side)
        roots.insert(entry.first);
    auto seqs = [](std::map<mlir::Operation *, std::vector<FrameUpdate>> &side,
                   mlir::Operation *root) {
      std::set<uint64_t> out;
      auto found = side.find(root);
      if (found != side.end())
        for (auto &update : found->second)
          out.insert(update.seq);
      return out;
    };
    for (mlir::Operation *root : roots) {
      std::set<uint64_t> old = seqs(before, root), inThen = seqs(then, root),
                         inElse = seqs(pending, root);
      std::map<uint64_t, FrameUpdate> merged;
      auto take =
          [&](std::map<mlir::Operation *, std::vector<FrameUpdate>> &side,
              bool arm, const std::set<uint64_t> &other) {
            auto found = side.find(root);
            if (found == side.end())
              return;
            for (auto &update : found->second) {
              if (merged.count(update.seq))
                continue;
              if (old.count(update.seq) && !other.count(update.seq)) {
                update.conditions.push_back(branch);
                std::get<1>(update.conditions.back()) = arm;
              }
              merged.emplace(update.seq, std::move(update));
            }
          };
      take(then, true, inElse);
      take(pending, false, inThen);
      std::vector<FrameUpdate> out;
      for (auto &[seq, update] : merged)
        out.push_back(std::move(update));
      if (out.empty())
        pending.erase(root);
      else
        pending[root] = std::move(out);
    }
  }

  /// The operands of `value`'s waveform nodes (numbered as `waveform` does,
  /// `nodes` its result) computed from a readout or an answer, as (node,
  /// operand index, operand).
  static void
  runTimeOperands(mlir::Value value, const std::vector<WaveformNode> &nodes,
                  int64_t &next,
                  std::vector<std::tuple<int64_t, int64_t, mlir::Value>> &out) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def) {
      ++next;
      return;
    }
    llvm::SmallVector<std::pair<int64_t, mlir::Value>, 4> numbers;
    for (mlir::Value operand : def->getOperands()) {
      if (mlir::isa<pulse::WaveformType>(operand.getType()))
        runTimeOperands(operand, nodes, next, out);
      else
        numbers.emplace_back(static_cast<int64_t>(numbers.size()), operand);
    }
    int64_t node = next++;
    const std::vector<double> &numbersKnown = std::get<1>(nodes[node]);
    for (auto &[index, operand] : numbers)
      if (std::isnan(numbersKnown[index]) && fromRunTime(operand))
        out.emplace_back(node, index, operand);
  }

  /// What the next record, a drive, readout or wait, has that is known only
  /// at run time: its index in `known`, else -1.
  LLVM_ATTRIBUTE_NOINLINE int32_t runTimeOf(mlir::Operation *op) {
    Known next;
    if (mlir::isa<pulse::WaitOp>(op)) {
      if (!fromRunTime(op->getOperand(1)) ||
          !op->hasAttr("pulse.variable_wait"))
        return -1;
      next.length = expression(op->getOperand(1));
      if (auto bound = op->getAttrOfType<mlir::IntegerAttr>("pulse.max_cycles"))
        next.maxCycles = bound.getInt();
      if (auto bound = op->getAttrOfType<mlir::IntegerAttr>("pulse.min_cycles"))
        next.minCycles = bound.getInt();
      known.push_back(std::move(next));
      return static_cast<int32_t>(known.size()) - 1;
    }
    auto &[nodes, unknown] = played.try_emplace(op).first->second;
    if (nodes.empty()) {
      auto cached = [this](mlir::Value value) { return number(value); };
      waveform(op->getOperand(1), nodes, cached);
      unknown = llvm::any_of(nodes, [](const WaveformNode &node) {
        return llvm::any_of(std::get<1>(node),
                            [](double v) { return std::isnan(v); });
      });
    }
    if (unknown) {
      std::vector<std::tuple<int64_t, int64_t, mlir::Value>> operands;
      int64_t count = 0;
      runTimeOperands(op->getOperand(1), nodes, count, operands);
      for (auto &[node, index, operand] : operands)
        next.waveform.emplace_back(node, index, expression(operand));
    }
    mlir::Operation *root =
        tracked.empty() ? nullptr : toneRoots(op->getOperand(2));
    if (root && tracked.contains(root)) {
      next.tone =
          toneIds.try_emplace(root, static_cast<int64_t>(toneIds.size()))
              .first->second;
      if (mlir::isa<pulse::DriveOp>(op)) {
        if (auto shift =
                op->getAttrOfType<mlir::FloatAttr>("frame_phase_offset"))
          frameUpdate(root, "shift_phase",
                      add(Expression::Number, shift.getValueAsDouble()), op);
        if (auto own = op->getAttrOfType<mlir::FloatAttr>("phase_offset"))
          next.offset = own.getValueAsDouble();
      }
      if (auto found = pending.find(root); found != pending.end()) {
        next.updates = std::move(found->second);
        pending.erase(found);
      }
    }
    if (next.tone < 0 && next.waveform.empty())
      return -1;
    known.push_back(std::move(next));
    return static_cast<int32_t>(known.size()) - 1;
  }

  /// The operands of a submission, in argument order.
  LLVM_ATTRIBUTE_NOINLINE int32_t submission(classical::SubmitOp request) {
    Known next;
    next.operands.emplace();
    for (mlir::Value arg : request.getArgs())
      next.operands->push_back(expression(arg));
    known.push_back(std::move(next));
    return static_cast<int32_t>(known.size()) - 1;
  }

  /// The waveform expression of `value`, children before their parent.
  static int64_t waveform(mlir::Value value, std::vector<WaveformNode> &nodes) {
    auto plain = [](mlir::Value number) { return numberOf(number); };
    return waveform(value, nodes, plain);
  }

  /// waveform(), its numbers from `number`.
  template <typename Number>
  static int64_t waveform(mlir::Value value, std::vector<WaveformNode> &nodes,
                          Number &number) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def) {
      nodes.emplace_back("argument", std::vector<double>{},
                         std::vector<int64_t>{});
      return static_cast<int64_t>(nodes.size()) - 1;
    }
    std::vector<double> numbers;
    std::vector<int64_t> children;
    for (mlir::Value operand : def->getOperands()) {
      if (mlir::isa<pulse::WaveformType>(operand.getType()))
        children.push_back(waveform(operand, nodes, number));
      else
        numbers.push_back(number(operand));
    }
    if (auto samples = def->getAttrOfType<mlir::ArrayAttr>("samples"))
      for (mlir::Attribute sample : samples)
        if (auto number = mlir::dyn_cast<mlir::FloatAttr>(sample))
          numbers.push_back(number.getValueAsDouble());
    std::string name = def->getName().stripDialect().str();
    if (auto custom = def->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      name += ":" + custom.getValue().str();
    nodes.emplace_back(name, std::move(numbers), std::move(children));
    return static_cast<int64_t>(nodes.size()) - 1;
  }

  /// The readout records and answer records `value` is computed from.
  void inputs(mlir::Value value, std::vector<int64_t> &readouts,
              std::vector<int64_t> &answers) {
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      mlir::Operation *def = v.getDefiningOp();
      if (!def || mlir::isa<classical::SubmitOp>(def))
        continue;
      if (mlir::isa<pulse::ReadoutOp, classical::WaitOp>(def)) {
        auto found = latest.find(def);
        if (found != latest.end()) {
          auto &list = mlir::isa<pulse::ReadoutOp>(def) ? readouts : answers;
          if (llvm::find(list, found->second) == list.end())
            list.push_back(found->second);
        }
        continue;
      }
      if (def->getNumRegions() == 0)
        work.append(def->getOperands().begin(), def->getOperands().end());
    }
  }

  int64_t lineOf(mlir::Operation *allocation) const {
    return allocation ? lineIndex.lookup(allocation) : -1;
  }

  ControllerRecord &record(const char *kind, mlir::Operation *op) {
    ControllerRecord &r = result.emplace_back();
    r.kind = kind;
    r.op = op;
    r.iteration.assign(iteration.begin(), iteration.end());
    r.conditions = branches;
    if (walk == Walk::Sampled)
      r.samples = sampling;
    if (segments)
      r.segment = segments->visit(op, iteration);
    if (track) {
      dependencies.emplace_back();
      natives.push_back(-1);
    }
    return r;
  }

  void block(mlir::Block &body) {
    for (mlir::Operation &op : body)
      step(&op);
  }

  void timed(mlir::Operation *op) {
    requireExportedMode(op);
    mlir::Operation *allocation = trace(op->getOperand(0));
    int64_t qubit = allocation ? pulse::quditOf(allocation) : -1;
    auto start = pulse::scheduledStart(op, iteration);
    auto duration = pulse::scheduledDuration(op);
    double frequency = kUnknown, phase = kUnknown;
    if (!mlir::isa<pulse::WaitOp>(op)) {
      bool drive = mlir::isa<pulse::DriveOp>(op);
      Frame &frame = frameOf(op->getOperand(2), drive);
      double offset = 0.0;
      if (drive) {
        if (auto shift =
                op->getAttrOfType<mlir::FloatAttr>("frame_phase_offset"))
          frame.phase += shift.getValueAsDouble();
        if (auto own = op->getAttrOfType<mlir::FloatAttr>("phase_offset"))
          offset = own.getValueAsDouble();
      }
      frequency = frame.frequencyHz;
      if (start)
        phase = frame.phaseAt(static_cast<double>(*start) / clockGHz) + offset;
      if (drive) {
        if (start && duration) {
          double endNs = static_cast<double>(*start + *duration) / clockGHz;
          frame.phase = frame.phaseAt(endNs);
          frame.timeNs = endNs;
        } else {
          frame = {kUnknown, kUnknown, kUnknown, frame.timeNs};
        }
      }
    }
    if (mlir::isa<pulse::ReadoutOp>(op))
      latest[op] = static_cast<int64_t>(result.size());
    if (persistent && !mlir::isa<pulse::WaitOp>(op))
      playedOn[toneRoots(op->getOperand(2))] = lineOf(allocation);
    auto [line, inserted] = reached.try_emplace(allocation, 0);
    reachedTo(line, inserted,
              start && duration && line->second
                  ? std::optional<int64_t>(
                        std::max(*line->second, *start + *duration))
                  : std::nullopt);
    int32_t knownIndex = runTime && detail ? runTimeOf(op) : -1;
    const char *kind = mlir::isa<pulse::DriveOp>(op)     ? "drive"
                       : mlir::isa<pulse::ReadoutOp>(op) ? "readout"
                                                         : "wait";
    ControllerRecord &r = record(kind, op);
    r.line = lineOf(allocation);
    r.qudit = qubit;
    r.start = start;
    r.duration = duration;
    r.frequency = frequency;
    r.phase = phase;
    r.known = knownIndex;
    if (track && kind[0] != 'w')
      trackPlayed(op);
  }

  /// A frame change some path does not play with: the frame of its tone
  /// after it, at an execution.
  struct Carried {
    mlir::Operation *op = nullptr;
    std::vector<int64_t> iteration;
    std::optional<int64_t> start;
    double frequency = kUnknown, phase = kUnknown;
    // Its value: a number, else an expression of the walk.
    double number = kUnknown;
    int32_t value = -1;
    // Its update on a tracked tone.
    uint64_t seq = std::numeric_limits<uint64_t>::max();
    // With `track`, what the frequency and phase after it depend on.
    std::vector<int64_t> frequencyDependencies, phaseDependencies;
  };
  /// The latest executions of each change a loop or branch lists
  /// (`pulse.carried`); a loop leaves its last iteration's.
  std::map<int64_t, std::vector<Carried>> carried;

  Carried capture(mlir::Operation *op) {
    Carried c;
    c.op = op;
    c.iteration.assign(iteration.begin(), iteration.end());
    c.start = pulse::scheduledStart(op, iteration);
    const Frame &frame = frameOf(op->getOperand(0));
    c.frequency = frame.frequencyHz;
    if (c.start)
      c.phase = frame.phaseAt(static_cast<double>(*c.start) / clockGHz);
    c.number = number(op->getOperand(1));
    if (std::isnan(c.number))
      c.value = expression(op->getOperand(1));
    if (runTime && tracked.contains(toneRoots(op->getOperand(0))))
      c.seq = nextUpdate - 1;
    if (track) {
      const FrameDependencies &depends = frameDependenciesOf(op->getOperand(0));
      c.frequencyDependencies = depends.frequency;
      c.phaseDependencies = depends.phase;
      if (depends.retuned) {
        merge(c.phaseDependencies, depends.frequency);
        merge(c.phaseDependencies, depends.frame);
      }
    }
    return c;
  }

  /// A frame update of its own for an execution of a change, with the
  /// tone's frame after it. On a tracked tone it takes the changes pending
  /// on it up to this one, as the next drive would.
  LLVM_ATTRIBUTE_NOINLINE void frameUpdateRecord(const Carried &c) {
    mlir::Value tone = c.op->getOperand(0);
    mlir::Operation *root = toneRoots(tone);
    int64_t line = -1, qudit = -1;
    if (mlir::isa_and_nonnull<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(
            root)) {
      line = lineOf(root);
      qudit = pulse::quditOf(root);
    } else {
      // A tone of its own: the line it last played on.
      for (size_t i = result.size(); i-- > 0;)
        if (result[i].played() &&
            toneRoots(result[i].op->getOperand(2)) == root) {
          line = result[i].line;
          qudit = result[i].qudit;
          break;
        }
    }
    int32_t knownIndex = -1;
    if (runTime && root && tracked.contains(root)) {
      Known next;
      next.tone =
          toneIds.try_emplace(root, static_cast<int64_t>(toneIds.size()))
              .first->second;
      if (auto found = pending.find(root); found != pending.end()) {
        auto &updates = found->second;
        auto taken = std::stable_partition(
            updates.begin(), updates.end(),
            [&](const FrameUpdate &u) { return u.seq <= c.seq; });
        next.updates.assign(updates.begin(), taken);
        updates.erase(updates.begin(), taken);
        if (updates.empty())
          pending.erase(found);
      }
      known.push_back(std::move(next));
      knownIndex = static_cast<int32_t>(known.size()) - 1;
    }
    ControllerRecord &r = record("frame_update", c.op);
    if (track) {
      dependencies.back()[0] = c.frequencyDependencies;
      dependencies.back()[1] = c.phaseDependencies;
    }
    r.line = line;
    r.qudit = qudit;
    r.start = c.start;
    r.duration = c.start ? std::optional<int64_t>(0) : std::optional<int64_t>();
    r.frequency = c.frequency;
    r.phase = c.phase;
    r.update = c.op->getName().getStringRef().drop_front(6);
    r.value = c.value >= 0 ? c.value : add(Expression::Number, c.number);
    r.known = knownIndex;
  }

  /// A change marked `pulse.unconsumed`, or one a loop or branch lists.
  LLVM_ATTRIBUTE_NOINLINE void pendingChange(mlir::Operation *op) {
    if (op->hasAttr("pulse.unconsumed"))
      return frameUpdateRecord(capture(op));
    if (auto id = op->getAttrOfType<mlir::IntegerAttr>("pulse.carried"))
      carried[id.getInt()].push_back(capture(op));
  }

  /// The frame updates `site` lists in `name`: the latest execution of
  /// each.
  LLVM_ATTRIBUTE_NOINLINE void listedUpdates(mlir::Operation *site,
                                             llvm::StringRef name) {
    auto ids = site->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
    if (!ids)
      return;
    for (int64_t id : ids.asArrayRef())
      if (auto found = carried.find(id);
          found != carried.end() && !found->second.empty())
        frameUpdateRecord(found->second.back());
  }

  /// The executions of the changes listed in `loop` that its walk added
  /// since `sizes`: only the last.
  void lastIteration(mlir::DenseI64ArrayAttr ids,
                     const std::vector<size_t> &sizes) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      std::vector<Carried> &all = carried[ids[i]];
      if (all.size() > sizes[i] + 1)
        all.erase(all.begin() + static_cast<std::ptrdiff_t>(sizes[i]),
                  all.end() - 1);
    }
  }

  /// A rolled loop: the executions of its changes walked since `entry`
  /// become the one of its last iteration, moved from the last two walked
  /// as every number moves from one iteration to the next.
  void rolledIterations(mlir::scf::ForOp loop, size_t depth, int64_t trips,
                        const std::map<int64_t, std::vector<Carried>> &entry) {
    auto ids = loop->getAttrOfType<mlir::DenseI64ArrayAttr>("pulse.carried_in");
    if (!ids)
      return;
    for (int64_t id : ids.asArrayRef()) {
      std::vector<Carried> &all = carried[id];
      auto before = entry.find(id);
      size_t from = before == entry.end() ? 0 : before->second.size();
      if (all.size() < from + 2)
        continue;
      const Carried &previous = all[all.size() - 2];
      Carried last = all.back();
      int64_t more = trips - 1 - last.iteration[depth];
      last.iteration[depth] = trips - 1;
      last.start = pulse::scheduledStart(last.op, last.iteration);
      last.phase = extend(previous.phase, last.phase, more);
      all.resize(from);
      all.push_back(std::move(last));
    }
  }

  void sync(pulse::SyncOp op) {
    std::vector<int64_t> lines;
    std::optional<int64_t> time = 0;
    for (mlir::Value value : op.getLines()) {
      mlir::Operation *allocation = trace(value);
      lines.push_back(lineOf(allocation));
      auto found = reached.find(allocation);
      if (found == reached.end())
        continue;
      if (!found->second || !time)
        time = std::nullopt;
      else
        time = std::max(*time, *found->second);
    }
    if (time)
      for (mlir::Value value : op.getLines())
        setReached(trace(value), time);
    ControllerRecord &r = record("sync", op);
    r.start = time;
    r.duration = time ? std::optional<int64_t>(0) : std::optional<int64_t>();
    r.joined = std::move(lines);
  }

  /// A submission (with the readout records whose measurements it sends) or
  /// the wait for its answer, on its link.
  void call(mlir::Operation *op) {
    bool submit = mlir::isa<classical::SubmitOp>(op);
    std::vector<int64_t> refs, answers;
    if (submit)
      for (mlir::Value operand : op->getOperands())
        inputs(operand, refs, answers);
    std::string callee;
    if (auto symbol = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      callee = symbol.getValue().str();
    std::string mode;
    double timeoutNs = kUnknown;
    if (!submit) {
      latest[op] = static_cast<int64_t>(result.size());
      auto [wait, timeout] = waitMode(op);
      mode = wait;
      timeoutNs = timeout;
    }
    int32_t knownIndex = -1;
    if (runTime)
      if (auto request = mlir::dyn_cast<classical::SubmitOp>(op))
        knownIndex = submission(request);
    ControllerRecord &r = record(submit ? "submit" : "answer", op);
    r.known = knownIndex;
    r.line = classical::intAttr(op, "classical.device");
    // A call has no qudit: its slot holds the controller whose instance of
    // a per-controller link runs it (-1: none).
    r.qudit = classical::intAttr(op, classical::kInstanceAttr, -1);
    r.start = pulse::scheduledStart(op, iteration);
    r.duration = pulse::scheduledDuration(op);
    r.refs = std::move(refs);
    r.callee = std::move(callee);
    r.mode = std::move(mode);
    r.timeout = timeoutNs;
  }

  void step(mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op)) {
      timed(op);
      return;
    }
    if (auto barrier = mlir::dyn_cast<pulse::SyncOp>(op)) {
      sync(barrier);
      return;
    }
    if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op)) {
      call(op);
      return;
    }
    if (name == "pulse.shift_phase" || name == "pulse.set_phase" ||
        name == "pulse.shift_frequency" || name == "pulse.set_frequency") {
      if (track)
        trackFrameChange(op, name);
      double value = number(op->getOperand(1));
      Frame &frame = frameOf(op->getOperand(0), true);
      if (runTime && !tracked.empty())
        frameChange(op, name.drop_front(6));
      frame.advanceTo(pulse::scheduledStart(op, iteration), clockGHz);
      if (name == "pulse.shift_phase")
        frame.phase += value;
      else if (name == "pulse.set_phase")
        frame.phase = value;
      else if (name == "pulse.shift_frequency")
        frame.frequencyHz += value;
      else
        frame.frequencyHz = value;
      if (op->hasAttr("pulse.unconsumed") || op->hasAttr("pulse.carried"))
        pendingChange(op);
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      auto ids =
          loop->getAttrOfType<mlir::DenseI64ArrayAttr>("pulse.carried_in");
      std::vector<size_t> sizes;
      if (ids)
        for (int64_t id : ids.asArrayRef())
          sizes.push_back(carried[id].size());
      // A sampled walk that counts segments visits each boundary; a rolled one
      // counts those of the iterations it does not walk.
      if (walk == Walk::Every || trips < 2 ||
          (segments && walk == Walk::Sampled && holdsBoundary(loop)))
        everyIteration(loop, trips);
      else if (walk == Walk::Sampled)
        sampled(loop, trips);
      else
        rolled(loop, trips);
      if (ids)
        lastIteration(ids, sizes);
      if (loop->hasAttr("pulse.exit_updates"))
        listedUpdates(loop, "pulse.exit_updates");
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      mlir::Value condition = branch.getCondition();
      std::vector<int64_t> readouts, answers;
      inputs(condition, readouts, answers);
      std::string test = "expression";
      int32_t computed = -1;
      int64_t answerResult = -1;
      mlir::Operation *def = condition.getDefiningOp();
      if (mlir::isa_and_nonnull<pulse::IsSetOp>(def) && readouts.size() == 1)
        test = "is_set";
      else if (mlir::isa_and_nonnull<classical::WaitOp>(def) &&
               answers.size() == 1) {
        test = "answer";
        answerResult = mlir::cast<mlir::OpResult>(condition).getResultNumber();
      } else
        computed = conditionExpression(condition, readouts, answers);
      if (computed >= 0) {
        // A condition computed in another way than its test names.
        const Expression &e = expressions[computed];
        if (e.kind == Expression::Answer) {
          test = "answer";
          answerResult = e.result;
          computed = -1;
        } else if (isSet(e)) {
          test = "is_set";
          computed = -1;
        }
      }
      int64_t id = nextBranch++;
      size_t frameMark = frameUndo.size(), reachedMark = reachedUndo.size();
      ++ifDepth;
      bool tracking = runTime && !tracked.empty();
      if (tracking)
        enterBranch();
      decltype(frameDependencies) dependenciesBefore, dependenciesThen;
      if (track)
        dependenciesBefore = frameDependencies;
      branches.emplace_back(id, 1, std::move(readouts), std::move(answers),
                            std::move(test), computed, answerResult);
      if (branch->hasAttr("pulse.then_updates"))
        listedUpdates(branch, "pulse.then_updates");
      block(*branch.thenBlock());
      ArmLeft thenLeft = undoArm(frameMark, reachedMark);
      if (track) {
        dependenciesThen = std::move(frameDependencies);
        frameDependencies = std::move(dependenciesBefore);
      }
      if (tracking)
        enterElse();
      std::get<1>(branches.back()) = 0;
      if (branch->hasAttr("pulse.else_updates"))
        listedUpdates(branch, "pulse.else_updates");
      if (!branch.getElseRegion().empty())
        block(*branch.elseBlock());
      if (tracking)
        leaveBranch(branches.back());
      branches.pop_back();
      --ifDepth;
      mergeArms(thenLeft, frameMark, reachedMark);
      // After the branch, a frame depends on what either arm set.
      for (auto &[root, theirs] : dependenciesThen) {
        auto &ours = frameDependencies[root];
        merge(ours.frequency, theirs.frequency);
        merge(ours.frame, theirs.frame);
        merge(ours.phase, theirs.phase);
        ours.retuned |= theirs.retuned;
      }
      return;
    }
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &nested : region)
        block(nested);
  }

  // ── Loops ────────────────────────────────────────────────────────────

  void everyIteration(mlir::scf::ForOp loop, int64_t trips) {
    iteration.push_back(0);
    for (int64_t k = 0; k < trips; ++k) {
      iteration.back() = k;
      block(*loop.getBody());
    }
    iteration.pop_back();
  }

  /// Whether `loop` holds a variable or blocking wait, which ends a segment
  /// in every iteration.
  bool holdsBoundary(mlir::scf::ForOp loop) {
    auto [it, inserted] = boundaries.try_emplace(loop, false);
    if (inserted)
      loop->walk([&it = it](mlir::Operation *op) {
        if (op->hasAttr("classical.variable") ||
            op->hasAttr("classical.blocking"))
          it->second = true;
      });
    return it->second;
  }

  struct State {
    llvm::DenseMap<mlir::Operation *, Frame> frames;
    llvm::DenseMap<mlir::Operation *, std::optional<int64_t>> reached;
    llvm::DenseMap<mlir::Operation *, int64_t> latest;
    int64_t nextBranch;
    size_t size;
    std::map<mlir::Operation *, std::vector<FrameUpdate>> pending;
    llvm::DenseMap<mlir::Operation *, int64_t> toneIds;
    uint64_t nextUpdate;
    std::map<int64_t, std::vector<Carried>> carried;
    std::map<mlir::Operation *, FrameDependencies> frameDependencies;
    size_t boundaries = 0;
    SegmentWalk::Mark segments;
  };

  State save() const {
    State state{frames,
                reached,
                latest,
                nextBranch,
                result.size(),
                pending,
                toneIds,
                nextUpdate,
                carried,
                track ? frameDependencies
                      : std::map<mlir::Operation *, FrameDependencies>()};
    if (segments) {
      state.boundaries = segments->boundaries.size();
      state.segments = segments->mark();
    }
    return state;
  }

  void restore(State &state) {
    frames = std::move(state.frames);
    reached = std::move(state.reached);
    latest = std::move(state.latest);
    nextBranch = state.nextBranch;
    result.resize(state.size);
    pending = std::move(state.pending);
    toneIds = std::move(state.toneIds);
    nextUpdate = state.nextUpdate;
    carried = std::move(state.carried);
    if (track) {
      dependencies.resize(state.size);
      natives.resize(state.size);
      frameDependencies = std::move(state.frameDependencies);
    }
    if (segments) {
      segments->boundaries.resize(state.boundaries);
      segments->skipped =
          state.segments.opened - static_cast<int64_t>(state.boundaries);
      segments->current = std::move(state.segments.current);
    }
  }

  /// Whether two frame states have the same frequencies, so that the body
  /// moves every phase by the same amount from one to the next.
  static bool
  sameFrequencies(const llvm::DenseMap<mlir::Operation *, Frame> &a,
                  const llvm::DenseMap<mlir::Operation *, Frame> &b) {
    auto get = [](const llvm::DenseMap<mlir::Operation *, Frame> &frames,
                  mlir::Operation *root) {
      auto found = frames.find(root);
      return found == frames.end() ? initialFrame(root) : found->second;
    };
    for (const auto *frames : {&a, &b})
      for (const auto &[root, frame] : *frames) {
        Frame x = get(a, root), y = get(b, root);
        if (!same(x.frequencyHz, y.frequencyHz) ||
            !same(x.frameFrequencyHz, y.frameFrequencyHz))
          return false;
      }
    return true;
  }

  /// The period by which records [b, e) follow records [a, b), or none when
  /// they do not match one to one, move by different amounts, or (with
  /// `apart`) one iteration overlaps the next.
  /// Whether record `y` of an iteration starting at `b` plays what record
  /// `x` of the one before, starting at `a`, does, at any time.
  bool sameShape(const ControllerRecord &x, const ControllerRecord &y, size_t a,
                 size_t b) const {
    return sameKind(x.kind, y.kind) && x.op == y.op &&
           x.start.has_value() == y.start.has_value() && x.trips == y.trips &&
           x.first == y.first && x.size == y.size &&
           x.steps.size() == y.steps.size() &&
           sameKnown(x, y, static_cast<int64_t>(a), static_cast<int64_t>(b));
  }

  /// Whether the iterations [a, b) and [b, e) play the same, only not one
  /// period apart: their ops move by different periods, or an iteration
  /// overlaps the next (see periodOf).
  bool repeatsOutOfStep(size_t a, size_t b, size_t e) const {
    if (b - a != e - b)
      return false;
    for (size_t i = 0; i < b - a; ++i)
      if (!sameShape(result[a + i], result[b + i], a, b))
        return false;
    return true;
  }

  std::optional<int64_t> periodOf(size_t a, size_t b, size_t e, bool apart) {
    if (b - a != e - b)
      return std::nullopt;
    std::optional<int64_t> period;
    std::optional<int64_t> earliest, latestEnd;
    for (size_t i = 0; i < b - a; ++i) {
      const ControllerRecord &x = result[a + i], &y = result[b + i];
      if (!sameShape(x, y, a, b))
        return std::nullopt;
      if (!x.start)
        continue;
      int64_t shift = *y.start - *x.start;
      if (period && *period != shift)
        return std::nullopt;
      period = shift;
      if (x.frameUpdate())
        continue;
      earliest = earliest ? std::min(*earliest, *x.start) : *x.start;
      if (auto stop = x.stop())
        latestEnd = latestEnd ? std::max(*latestEnd, *stop) : *stop;
    }
    if (!period)
      return int64_t{0};
    if (apart && latestEnd && *latestEnd - *earliest > *period)
      return std::nullopt;
    return period;
  }

  /// Whether a record of [b, e) refers to one of [a, b): an iteration that
  /// uses the previous one's readouts or answers.
  bool refersBack(size_t a, size_t b, size_t e) const {
    auto into = [&](const std::vector<int64_t> &refs) {
      return llvm::any_of(refs, [&](int64_t ref) {
        return ref >= static_cast<int64_t>(a) && ref < static_cast<int64_t>(b);
      });
    };
    for (size_t i = b; i < e; ++i) {
      if (into(result[i].refs))
        return true;
      for (const Condition &c : result[i].conditions)
        if (into(std::get<2>(c)) || into(std::get<3>(c)))
          return true;
    }
    return false;
  }

  void sampled(mlir::scf::ForOp loop, int64_t trips) {
    if (trips < 3)
      return everyIteration(loop, trips);
    State entry = save();
    int64_t instance = nextInstance++;
    iteration.push_back(0);
    size_t a = result.size();
    block(*loop.getBody());
    auto after = frames;
    size_t b = result.size();
    iteration.back() = 1;
    sampling.emplace_back(instance, 1);
    block(*loop.getBody());
    sampling.pop_back();
    size_t e = result.size();
    bool same = sameFrequencies(after, frames) && !refersBack(a, b, e);
    if (!same || !periodOf(a, b, e, true)) {
      bool outOfStep = same && !runTime && trips > kDeclineAfter &&
                       repeatsOutOfStep(a, b, e);
      iteration.pop_back();
      restore(entry);
      if (outOfStep) {
        declined = true;
        return;
      }
      return everyIteration(loop, trips);
    }
    for (size_t i = b; i < e; ++i)
      result[i].weight *= trips - 2;
    iteration.back() = trips - 1;
    sampling.emplace_back(instance, 2);
    block(*loop.getBody());
    sampling.pop_back();
    iteration.pop_back();
  }

  static double extend(double before, double after, int64_t times) {
    if (std::isnan(after) || std::isnan(before))
      return after;
    return after + static_cast<double>(times) * (after - before);
  }

  void rolled(mlir::scf::ForOp loop, int64_t trips) {
    State entry = save();
    // A body that changes or plays a tracked tone's frame lists frame
    // changes that differ from one iteration to the next.
    auto tracks = [&] {
      if (nextUpdate != entry.nextUpdate)
        return true;
      for (size_t i = entry.size; i < result.size(); ++i)
        if (result[i].known >= 0 && known[result[i].known].tone >= 0)
          return true;
      return false;
    };
    size_t depth = iteration.size();
    iteration.push_back(0);
    auto frames0 = frames;
    size_t a0 = result.size();
    block(*loop.getBody());
    // The segments at the start of each iteration walked, and after the last.
    std::array<SegmentWalk::Mark, 4> marks;
    if (segments) {
      marks[0] = entry.segments;
      marks[1] = segments->mark();
    }
    // The states before and after the first iteration exported rolled.
    auto framesFirst = frames0, framesNext = frames;
    auto reachedNext = reached;
    auto latestNext = latest;
    size_t a1 = result.size();
    iteration.back() = 1;
    block(*loop.getBody());
    size_t a2 = result.size();
    if (segments)
      marks[2] = segments->mark();
    int64_t first = 0;
    size_t a = a0, b = a1, e = a2;
    if (!sameFrequencies(framesFirst, framesNext)) {
      first = 1;
      bool repeats = trips >= 3 && sameFrequencies(framesNext, frames);
      if (repeats) {
        framesFirst = std::move(framesNext);
        framesNext = frames;
        reachedNext = reached;
        latestNext = latest;
        iteration.back() = 2;
        block(*loop.getBody());
        if (segments)
          marks[3] = segments->mark();
        a = a1, b = a2, e = result.size();
      } else {
        e = 0;
      }
    }
    std::optional<int64_t> period =
        e && !tracks() ? periodOf(a, b, e, false) : std::optional<int64_t>();
    // The iterations exported rolled open as many segments each.
    if (period && segments &&
        !SegmentWalk::steady(llvm::ArrayRef(marks).slice(first, 3)))
      period.reset();

    std::vector<ControllerRecord::Step> steps;
    for (size_t i = 0; period && i < b - a; ++i) {
      const ControllerRecord &x = result[a + i], &y = result[b + i];
      double step = y.phase - x.phase;
      if (!same(x.frequency, y.frequency) ||
          std::isnan(x.phase) != std::isnan(y.phase))
        period.reset();
      for (size_t l = 0; period && l < x.steps.size(); ++l)
        if (!same(x.steps[l].phase, y.steps[l].phase) ||
            x.steps[l].segment != y.steps[l].segment)
          period.reset();
      steps.push_back({step, segments ? y.segment - x.segment : 0});
    }
    if (!period) {
      iteration.pop_back();
      restore(entry);
      return everyIteration(loop, trips);
    }
    // The state after the last iteration: every number moves by the same
    // amount per iteration from the first exported one on.
    int64_t more = trips - first - 2;
    if (segments && !segments->skip(more, marks[first + 1]))
      throw std::overflow_error("a rolled loop opens more than 2**63 segments");
    // A frame moves as from the next-to-last iteration walked to the last:
    // the state before the first may have last changed at any earlier time,
    // not one period before, and a later change that retunes the frame reads
    // its time.
    auto framesAfter = frames;
    for (auto &[root, frame] : framesAfter) {
      auto found = framesNext.find(root);
      Frame before =
          found == framesNext.end() ? initialFrame(root) : found->second;
      frame.phase = extend(before.phase, frame.phase, more);
      frame.timeNs = extend(before.timeNs, frame.timeNs, more);
    }
    // A line's reach moves as from the next-to-last iteration walked to the
    // last: a line first played in the loop has no reach before it.
    for (auto &[allocation, time] : reached) {
      auto next = reachedNext.find(allocation);
      if (next == reachedNext.end())
        continue;
      if (time && next->second)
        time = *time + more * (*time - *next->second);
      else
        time = std::nullopt;
    }
    frames = std::move(framesAfter);
    if (!carried.empty())
      rolledIterations(loop, depth, trips, entry.carried);
    result.resize(b);
    if (track) {
      // A record of the body stands for every iteration: it depends on
      // what any does. A shift adds to what a frame depends on and a set
      // replaces it, so from the second iteration on each depends alike.
      for (size_t i = a; i < b; ++i) {
        for (size_t f = 0; f < dependencies[i].size(); ++f)
          merge(dependencies[i][f], dependencies[i + (b - a)][f]);
        if (natives[i] != natives[i + (b - a)])
          natives[i] = -1;
      }
      dependencies.resize(b);
      natives.resize(b);
      dependencies.insert(dependencies.begin() + static_cast<std::ptrdiff_t>(a),
                          Dependencies());
      natives.insert(natives.begin() + static_cast<std::ptrdiff_t>(a), -1);
    }
    ControllerRecord node;
    node.kind = "loop";
    node.op = loop;
    node.iteration.assign(iteration.begin(), iteration.begin() + depth);
    node.conditions = branches;
    node.trips = trips - first;
    node.first = first;
    node.period = *period;
    node.size = static_cast<int64_t>(b - a);
    // Frame updates take no time: the loop spans the rest of its body,
    // unless the body has nothing else.
    bool timed = std::any_of(
        result.begin() + a, result.begin() + b,
        [](const ControllerRecord &r) { return r.start && !r.frameUpdate(); });
    for (size_t i = a; i < b; ++i) {
      ControllerRecord &r = result[i];
      r.iteration[depth] = -1;
      r.steps.insert(r.steps.begin(), steps[i - a]);
      if (timed && r.frameUpdate())
        continue;
      if (r.start && !node.start)
        node.start = r.start;
      else if (r.start)
        node.start = std::min(*node.start, *r.start);
      if (auto stop = r.stop()) {
        int64_t last = *stop + (node.trips - 1) * node.period;
        node.end = node.end ? std::max(*node.end, last) : last;
      }
    }
    node.duration = node.start && node.end
                        ? std::optional<int64_t>(*node.end - *node.start)
                        : std::nullopt;
    // The body moves one place on; references into it follow.
    auto shift = [&](int64_t &ref) {
      if (ref >= static_cast<int64_t>(a))
        ++ref;
    };
    for (size_t i = a; i < b; ++i) {
      ControllerRecord &r = result[i];
      shiftKnown(r, shift);
      for (int64_t &ref : r.refs)
        shift(ref);
      for (Condition &c : r.conditions) {
        for (int64_t &ref : std::get<2>(c))
          shift(ref);
        for (int64_t &ref : std::get<3>(c))
          shift(ref);
      }
    }
    for (auto &[op, ref] : latestNext)
      shift(ref);
    latest = std::move(latestNext);
    result.insert(result.begin() + static_cast<std::ptrdiff_t>(a),
                  std::move(node));
    iteration.pop_back();
  }
};

// ── Waveforms ─────────────────────────────────────────────────────────────

/// The distinct waveforms the drives and readouts of a walk play, and which
/// one each op plays: ops with the same expression share one.
struct ControllerWaveforms {
  using Nodes = std::vector<ControllerWalk::WaveformNode>;
  using Played = decltype(ControllerWalk::played);

  std::vector<Nodes> distinct;
  llvm::DenseMap<mlir::Operation *, int64_t> of;

  /// Add the waveform `op` plays, taking it from `known` (the walk's
  /// `played`) where the walk computed it.
  void add(mlir::Operation *op, Played *known = nullptr) {
    if (of.count(op))
      return;
    // An op computing its waveform as an earlier one does, from the same
    // values, plays the same waveform.
    mlir::Operation *shape = op->getOperand(1).getDefiningOp();
    if (shape && shape->getNumRegions() != 0)
      shape = nullptr;
    if (shape) {
      auto same = byShape.find(shape);
      if (same != byShape.end()) {
        of[op] = same->second;
        return;
      }
    }
    Nodes nodes;
    if (auto found = known ? known->find(op) : Played::iterator();
        known && found != known->end())
      nodes = std::move(found->second.first);
    else
      ControllerWalk::waveform(op->getOperand(1), nodes);
    std::string key;
    for (const auto &[name, numbers, children] : nodes) {
      key += name;
      key += '\0';
      key.append(reinterpret_cast<const char *>(numbers.data()),
                 numbers.size() * sizeof(double));
      key += '\1';
      key.append(reinterpret_cast<const char *>(children.data()),
                 children.size() * sizeof(int64_t));
      key += '\2';
    }
    auto [it, inserted] = index.try_emplace(
        std::move(key), static_cast<int64_t>(distinct.size()));
    if (inserted)
      distinct.push_back(std::move(nodes));
    of[op] = it->second;
    if (shape)
      byShape.try_emplace(shape, it->second);
  }

private:
  /// Ops without regions alike as CSE compares them: name, attributes,
  /// properties, result types and the very same operands.
  struct SameShape : llvm::DenseMapInfo<mlir::Operation *> {
    static unsigned getHashValue(const mlir::Operation *op) {
      auto *mutableOp = const_cast<mlir::Operation *>(op);
      llvm::hash_code hash = llvm::hash_combine(
          mutableOp->getName().getAsOpaquePointer(),
          mutableOp->getRawDictionaryAttrs().getAsOpaquePointer());
      for (mlir::Value operand : mutableOp->getOperands())
        hash = llvm::hash_combine(hash, operand.getAsOpaquePointer());
      return static_cast<unsigned>(hash);
    }
    static bool isEqual(const mlir::Operation *a, const mlir::Operation *b) {
      if (a == b)
        return true;
      if (a == getEmptyKey() || a == getTombstoneKey() || b == getEmptyKey() ||
          b == getTombstoneKey())
        return false;
      auto *x = const_cast<mlir::Operation *>(a);
      auto *y = const_cast<mlir::Operation *>(b);
      return x->getName() == y->getName() &&
             x->getRawDictionaryAttrs() == y->getRawDictionaryAttrs() &&
             x->getOperands() == y->getOperands() &&
             x->getResultTypes() == y->getResultTypes() &&
             x->getName().compareOpProperties(x->getPropertiesStorage(),
                                              y->getPropertiesStorage());
    }
  };

  std::unordered_map<std::string, int64_t> index;
  llvm::DenseMap<mlir::Operation *, int64_t, SameShape> byShape;
};

// ── Controller clocks ─────────────────────────────────────────────────────

/// A line event that does not start and last whole multiples of its
/// controller's grid: (index, start, duration) in program cycles.
using OffGridEvent = std::tuple<int64_t, int64_t, int64_t>;

/// The records of a walk of every execution in the cycles of a controller
/// counting `num` of its cycles every `den` program cycles: each start,
/// duration and end rounded up, an end kept apart where it is not the start
/// plus the duration. Returns the drives, readouts and waits off its grid of
/// `grid` program cycles.
inline std::vector<OffGridEvent>
ownCycles(std::vector<ControllerRecord> &records, int64_t num, int64_t den,
          int64_t grid) {
  std::vector<OffGridEvent> offGrid;
  auto own = [&](std::optional<int64_t> &cycles) {
    if (cycles)
      *cycles = (*cycles * num + den - 1) / den;
  };
  for (size_t index = 0; index < records.size(); ++index) {
    ControllerRecord &r = records[index];
    if (grid > 1 && r.start && r.duration &&
        (isKind(r.kind, "drive") || isKind(r.kind, "readout") ||
         isKind(r.kind, "wait")) &&
        (*r.start % grid || *r.duration % grid))
      offGrid.emplace_back(static_cast<int64_t>(index), *r.start, *r.duration);
    if (num == den)
      continue;
    std::optional<int64_t> end = r.stop();
    own(r.start);
    own(r.duration);
    own(end);
    if (end && *end != *r.start + *r.duration)
      r.end = end;
  }
  return offGrid;
}

/// The starts of the frame changes a walk lists on tracked tones, taken or
/// pending, converted as ownCycles converts its records' starts.
inline void ownFrameUpdateCycles(ControllerWalk &walk, int64_t num,
                                 int64_t den) {
  if (num == den)
    return;
  auto own = [&](std::vector<ControllerWalk::FrameUpdate> &updates) {
    for (ControllerWalk::FrameUpdate &update : updates)
      if (update.start)
        *update.start = (*update.start * num + den - 1) / den;
  };
  for (ControllerWalk::Known &known : walk.known)
    own(known.updates);
  for (auto &entry : walk.pending)
    own(entry.second);
}

// ── Limit checks ──────────────────────────────────────────────────────────

/// What a `Controller` checks on the schedule; -1 for a limit not set.
struct ControllerLimitSpec {
  int64_t issue = 1;
  int64_t gap = 1;
  // Samples per cycle: samplesNum every samplesDen cycles.
  int64_t samplesNum = 1;
  int64_t samplesDen = 1;
  int64_t waveformSamples = -1;
  // The waveform memory: its allocation unit, whether all lines share one,
  // whether one stored shape plays at any amplitude, the built-in shapes it
  // plays without memory, and whether a constant one needs none.
  int64_t granule = 1;
  bool sharedMemory = false;
  bool scaling = true;
  std::set<std::string> synthesized;
  bool hold = false;
  int64_t maxFrequenciesPerLine = -1;
  int64_t maxEvents = -1;
  std::map<int64_t, int64_t> detectors;
  int64_t detectorRecovery = 0;
  int64_t feedbackLatency = 0;
  bool answerTimeouts = true;
  std::set<std::string> timeoutFunctions;
};

/// The key of `shape_key` in cudaq_pulse.controller: equal for waveforms
/// that differ only in overall amplitude, known or not; none when another
/// number is not known. Without `scaled`, equal only for equal waveforms,
/// and none when any number is not known.
inline std::optional<std::string>
shapeKey(const std::vector<ControllerWalk::WaveformNode> &nodes, int64_t index,
         bool scaled = true) {
  static const std::map<std::string, std::vector<std::string>> parameters = {
      {"square", {"duration", "amplitude", "amplitude_imag"}},
      {"gaussian", {"duration", "amplitude", "sigma"}},
      {"gaussian_square", {"duration", "amplitude", "sigma", "risefall"}},
      {"drag", {"duration", "amplitude", "sigma", "beta"}},
      {"cosine", {"duration", "amplitude"}},
      {"tanh_ramp", {"duration", "amplitude", "sigma"}},
      {"custom", {"duration"}},
      {"scale", {"factor"}},
  };
  static const std::map<std::string, std::set<size_t>> amplitudes = {
      {"square", {1, 2}}, {"gaussian", {1}}, {"gaussian_square", {1}},
      {"drag", {1}},      {"cosine", {1}},   {"tanh_ramp", {1}},
  };
  auto text = [](double v) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.17g", v == 0.0 ? 0.0 : v);
    return std::string(buffer);
  };
  const auto &[name, numbers, children] = nodes[index];
  std::string shape = name.substr(0, name.find(':'));
  std::string custom = name.find(':') == std::string::npos
                           ? ""
                           : name.substr(name.find(':') + 1);
  if (!scaled) {
    std::string key = name + "(";
    for (double v : numbers) {
      if (std::isnan(v))
        return std::nullopt;
      key += text(v) + ",";
    }
    key += ")[";
    for (int64_t child : children) {
      auto inner = shapeKey(nodes, child, false);
      if (!inner)
        return std::nullopt;
      key += *inner + ";";
    }
    return key + "]";
  }
  if (shape == "scale" || shape == "neg")
    return children.empty() ? std::nullopt : shapeKey(nodes, children[0]);
  if (shape == "custom_samples") {
    double peak = 0.0;
    for (double v : numbers)
      peak = std::max(peak, std::fabs(v));
    if (peak == 0.0)
      peak = 1.0;
    std::string key = shape;
    for (double v : numbers)
      key += "," + text(std::round(v / peak * 1e12) / 1e12);
    return key;
  }
  auto names = parameters.find(shape);
  if (names == parameters.end() || names->second.size() != numbers.size()) {
    std::string key = shape + "(";
    for (double v : numbers)
      key += (std::isnan(v) ? "None" : text(v)) + ",";
    key += ")[";
    for (int64_t child : children) {
      auto inner = shapeKey(nodes, child);
      if (!inner)
        return std::nullopt;
      key += *inner + ";";
    }
    return key + "]";
  }
  std::string key = shape + "|" + custom;
  auto skip = amplitudes.find(shape);
  for (size_t i = 0; i < numbers.size(); ++i)
    if (skip == amplitudes.end() || !skip->second.count(i)) {
      if (std::isnan(numbers[i]))
        return std::nullopt;
      key += "|" + names->second[i] + "=" + text(numbers[i]);
    }
  return key;
}

/// The op name of a waveform's root, through scale and neg.
inline std::string
rootShape(const std::vector<ControllerWalk::WaveformNode> &nodes) {
  int64_t index = static_cast<int64_t>(nodes.size()) - 1;
  while (index >= 0) {
    const auto &[name, numbers, children] = nodes[index];
    if ((name != "scale" && name != "neg") || children.empty())
      return name;
    index = children[0];
  }
  return "";
}

/// The limits `records` exceed. With `unresolved`, a tone or waveform whose
/// value is not known counts only where the limit is exceeded whatever its
/// value; `*unresolved` is set when only some values would exceed one.
inline std::vector<std::string>
checkControllerLimits(const std::vector<ControllerRecord> &records,
                      const ControllerLimitSpec &limits,
                      bool *unresolved = nullptr) {
  std::vector<std::string> found;
  std::vector<size_t> timed;
  int64_t played = 0;
  for (size_t i = 0; i < records.size(); ++i) {
    const ControllerRecord &r = records[i];
    if (r.played() && r.start && r.duration) {
      timed.push_back(i);
      played += r.weight;
    }
  }
  if (limits.maxEvents >= 0 && played > limits.maxEvents)
    found.push_back(std::to_string(played) +
                    " drives and readouts; the controller plays at most " +
                    std::to_string(limits.maxEvents));

  auto exclusive = [&](const ControllerRecord &a, const ControllerRecord &b) {
    if (a.conditions.empty() || b.conditions.empty() ||
        a.iteration != b.iteration)
      return false;
    for (const auto &c : b.conditions)
      for (const auto &d : a.conditions)
        if (std::get<0>(c) == std::get<0>(d) &&
            std::get<1>(c) != std::get<1>(d))
          return true;
    return false;
  };
  // A pair from a sampled loop's iteration 1 to its last stands for the
  // pair from one iteration to the next, which iterations 0 and 1 show.
  auto jumped = [](const ControllerRecord &a, const ControllerRecord &b) {
    for (const auto &[instance, sample] : a.samples)
      if (sample == 1)
        for (const auto &[other, later] : b.samples)
          if (other == instance && later == 2)
            return true;
    return false;
  };
  // Each record with the latest earlier one it can follow on one path.
  auto pairs = [&](std::vector<size_t> items) {
    std::stable_sort(items.begin(), items.end(), [&](size_t x, size_t y) {
      return *records[x].start < *records[y].start;
    });
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t k = 1; k < items.size(); ++k)
      for (size_t j = k; j-- > 0;) {
        if (jumped(records[items[j]], records[items[k]]))
          break;
        if (!exclusive(records[items[j]], records[items[k]])) {
          out.emplace_back(items[j], items[k]);
          break;
        }
      }
    return out;
  };

  // A record of a sampled loop's iteration 1 stands for iterations 1 to
  // n - 2: its message says so.
  auto times = [](const ControllerRecord &r) {
    return r.weight > 1 ? " (in each of " + std::to_string(r.weight) +
                              " iterations of a loop)"
                        : std::string();
  };
  std::map<int64_t, std::vector<size_t>> byLine;
  for (size_t i : timed)
    byLine[records[i].line].push_back(i);
  ControllerWaveforms waveforms;
  std::vector<std::optional<std::optional<std::string>>> keys;
  for (auto &[line, items] : byLine) {
    for (auto [p, q] : pairs(items)) {
      const ControllerRecord &previous = records[p], &event = records[q];
      int64_t distance = *event.start - *previous.start;
      if (distance != limits.issue && distance < limits.gap)
        found.push_back(
            "line " + std::to_string(line) + ": " + event.kind + " at " +
            std::to_string(*event.start) + " starts " +
            std::to_string(distance) + " cycles after the " + previous.kind +
            " at " + std::to_string(*previous.start) +
            "; the controller needs exactly " + std::to_string(limits.issue) +
            " or at least " + std::to_string(limits.gap) + times(event));
    }
    if (limits.maxFrequenciesPerLine >= 0) {
      std::set<double> tones;
      int64_t unknown = 0;
      for (size_t i : items) {
        if (!std::isnan(records[i].frequency))
          tones.insert(records[i].frequency);
        else if (records[i].played())
          unknown += records[i].weight;
      }
      int64_t known = static_cast<int64_t>(tones.size());
      if (known > limits.maxFrequenciesPerLine)
        found.push_back("line " + std::to_string(line) + ": " +
                        std::to_string(tones.size()) +
                        " tone frequencies; the controller holds " +
                        std::to_string(limits.maxFrequenciesPerLine) +
                        " per line");
      else if (unresolved && known + unknown > limits.maxFrequenciesPerLine)
        *unresolved = true;
    }
  }
  if (limits.waveformSamples >= 0) {
    // One memory per line, or one for all.
    std::map<int64_t, std::vector<size_t>> memories;
    for (auto &[line, items] : byLine)
      for (size_t i : items)
        if (records[i].played())
          memories[limits.sharedMemory ? -1 : line].push_back(i);
    std::vector<int8_t> costless;
    for (auto &[line, items] : memories) {
      std::map<std::string, int64_t> shapes;
      int64_t unshared = 0, unknown = 0;
      for (size_t i : items) {
        const ControllerRecord &r = records[i];
        waveforms.add(r.op);
        size_t shape = static_cast<size_t>(waveforms.of.lookup(r.op));
        if (keys.size() <= shape) {
          keys.resize(shape + 1);
          costless.resize(shape + 1, -1);
        }
        const auto &nodes = waveforms.distinct[shape];
        if (costless[shape] < 0) {
          std::string root = rootShape(nodes);
          costless[shape] = limits.synthesized.count(root) ||
                            (limits.hold && root == "square");
        }
        if (costless[shape])
          continue;
        if (!keys[shape])
          keys[shape] = shapeKey(nodes, static_cast<int64_t>(nodes.size()) - 1,
                                 limits.scaling);
        const std::optional<std::string> &key = *keys[shape];
        int64_t samples =
            (*r.duration * limits.samplesNum + limits.samplesDen - 1) /
            limits.samplesDen;
        samples =
            (samples + limits.granule - 1) / limits.granule * limits.granule;
        if (key)
          shapes[*key + "@" + std::to_string(*r.duration)] = samples;
        else if (unresolved)
          unknown += samples * r.weight;
        else
          unshared += samples * r.weight;
      }
      int64_t used = unshared;
      for (auto &[key, samples] : shapes)
        used += samples;
      if (used > limits.waveformSamples)
        found.push_back(
            (limits.sharedMemory
                 ? std::string("its waveforms take ")
                 : "line " + std::to_string(line) + ": its waveforms take ") +
            std::to_string(used) + " samples; the controller's " +
            (limits.sharedMemory ? "shared " : "") + "waveform memory holds " +
            std::to_string(limits.waveformSamples));
      else if (unresolved && used + unknown > limits.waveformSamples)
        *unresolved = true;
    }
  }
  if (!limits.detectors.empty()) {
    std::map<int64_t, std::vector<size_t>> byDetector;
    for (size_t i : timed) {
      const ControllerRecord &r = records[i];
      auto detector = limits.detectors.find(r.qudit);
      if (isKind(r.kind, "readout") && detector != limits.detectors.end())
        byDetector[detector->second].push_back(i);
    }
    for (auto &[detector, items] : byDetector)
      for (auto [p, q] : pairs(items)) {
        const ControllerRecord &previous = records[p], &event = records[q];
        int64_t free = *previous.stop() + limits.detectorRecovery;
        if (*event.start < free)
          found.push_back(
              "detector " + std::to_string(detector) +
              ": the readout of qudit " + std::to_string(event.qudit) + " at " +
              std::to_string(*event.start) +
              " starts before the detector is free at " + std::to_string(free) +
              " (the readout of qudit " + std::to_string(previous.qudit) +
              " at " + std::to_string(*previous.start) + ")" + times(event));
      }
  }
  if (limits.feedbackLatency)
    for (size_t i : timed) {
      const ControllerRecord &event = records[i];
      for (const auto &condition : event.conditions)
        for (int64_t ref : std::get<2>(condition)) {
          auto end = records[ref].stop();
          if (!end || records[ref].end)
            continue;
          int64_t ready = *end + limits.feedbackLatency;
          if (*event.start < ready)
            found.push_back(
                "line " + std::to_string(event.line) + ": the " + event.kind +
                " at " + std::to_string(*event.start) +
                " depends on the readout ending at " + std::to_string(*end) +
                "; the controller needs " +
                std::to_string(limits.feedbackLatency) +
                " cycles of feedback latency (start at " +
                std::to_string(ready) +
                " or later; schedule with rcp and readout_latency_cycles >= "
                "feedback_latency_cycles)" +
                times(event));
        }
    }
  if (!limits.answerTimeouts)
    for (const ControllerRecord &r : records)
      if (isKind(r.kind, "answer") &&
          (r.mode == "timeout" || limits.timeoutFunctions.count(r.callee)))
        found.push_back(
            "the wait for " + r.callee + " at " +
            (r.start ? std::to_string(*r.start) : "None") +
            " has a timeout; the controller cannot stop waiting for "
            "an answer, so take it with result() (no timeout)" +
            times(r));
  return found;
}

// ── Python objects ────────────────────────────────────────────────────────

/// Builds the events of a walk as the dictionaries of the export format,
/// without reading the IR. `waveform` turns the nodes of a waveform into
/// (expression, amplitude); it is called once per distinct waveform, and
/// the events that play one share what it returns. Without it, the
/// expression and amplitude are built here. `parameters` names the numbers
/// of each waveform shape, and `amplitudes` holds the shapes with an
/// overall amplitude.
class ControllerEvents {
public:
  /// `compact` (format /2) leaves out what is derivable or the default: the
  /// index (an event's place in pre-order), the qudit (its line's), the end,
  /// an empty iteration, segment 0, empty conditions and lists, a sync's or
  /// call's line, a timeout that is none and a first iteration of 0; its
  /// `waveform` returns (index into the program's waveforms, None).
  ControllerEvents(const ControllerWalk &walk,
                   const ControllerWaveforms &waveforms, nb::object waveform,
                   nb::handle parameters, nb::handle amplitudes, bool compact,
                   bool exportable = true)
      : records(walk.result), walk(walk), waveforms(waveforms),
        waveform(std::move(waveform)), compact(compact), exportable(exportable),
        key(keys()), shapes(shapesOf(parameters, amplitudes)),
        made(waveforms.distinct.size()) {}

  nb::list build() {
    // Nothing built here is garbage until it is returned: a collection now
    // would only walk the heap.
    CollectorPause pause;
    size_t at = 0;
    return range(at, records.size());
  }

  /// The frame each tone ends the shot in (ControllerWalk::finalFrames),
  /// as the export's `frame_state`.
  nb::list frameState(const std::vector<ControllerWalk::FinalFrame> &all) {
    nb::list out = list(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
      const ControllerWalk::FinalFrame &final = all[i];
      nb::dict item;
      put(item, key.line, integer(final.line));
      if (final.tone >= 0) {
        put(item, key.tone, integer(final.tone));
        put(item, key.frameUpdates, frameUpdates(final.updates));
      } else {
        put(item, key.frequency, number(final.frequencyHz));
        put(item, key.phase, number(final.phase));
      }
      set(out, i, std::move(item));
    }
    return out;
  }

private:
  using Expression = ControllerWalk::Expression;
  using Nodes = ControllerWaveforms::Nodes;
  using Operands = std::vector<std::tuple<int64_t, int64_t, int32_t>>;

  struct CollectorPause {
    int enabled = PyGC_Disable();
    ~CollectorPause() {
      if (enabled)
        PyGC_Enable();
    }
  };

  // The keys and names of an export, interned once for the life of the
  // process: an export sets them on every event.
  struct Keys {
    PyObject *amplitude = intern("amplitude"), *answer = intern("answer"),
             *answers = intern("answers"), *arm = intern("arm"),
             *body = intern("body"), *branch = intern("branch"),
             *conditions = intern("conditions"), *constant = intern("const"),
             *controller = intern("controller"),
             *duration = intern("duration_cycles"), *end = intern("end_cycles"),
             *segment = intern("segment"),
             *segmentStep = intern("segment_step"),
             *expression = intern("expression"),
             *first = intern("first_iteration"),
             *frameUpdates = intern("frame_updates"),
             *frequency = intern("frequency_hz"),
             *function = intern("function"), *id = intern("id"),
             *im = intern("im"), *index = intern("index"),
             *iteration = intern("iteration"), *kind = intern("kind"),
             *length = intern("length"), *line = intern("line"),
             *lines = intern("lines"), *link = intern("link"),
             *maxCycles = intern("max_cycles"),
             *measurements = intern("measurements"),
             *minCycles = intern("min_cycles"), *name = intern("name"),
             *of = intern("of"), *op = intern("op"),
             *operands = intern("operands"), *params = intern("params"),
             *path = intern("path"), *period = intern("period_cycles"),
             *phase = intern("phase_rad"),
             *phaseOffset = intern("phase_offset_rad"),
             *phaseStep = intern("phase_step_rad"), *qudit = intern("qudit"),
             *re = intern("re"), *readout = intern("readout"),
             *readouts = intern("readouts"), *result = intern("result"),
             *runtime = intern("runtime"), *samples = intern("samples"),
             *shape = intern("shape"), *start = intern("start_cycles"),
             *test = intern("test"), *timeout = intern("timeout_ns"),
             *term = intern("term"), *table = intern("terms"),
             *tone = intern("tone"), *tones = intern("tones"),
             *trips = intern("trips"), *type = intern("type"),
             *unknown = intern("unknown"), *value = intern("value"),
             *values = intern("values"), *wait = intern("wait"),
             *waveform = intern("waveform");
    std::array<PyObject *, Expression::Terms> terms;
    std::array<PyObject *, Expression::Types> types;

    Keys() {
      for (size_t i = 0; i < terms.size(); ++i)
        terms[i] = intern(Expression::names[i]);
      for (size_t i = 0; i < types.size(); ++i)
        types[i] = intern(Expression::typeNames[i]);
    }
    static PyObject *intern(const char *text) {
      PyObject *object = PyUnicode_InternFromString(text);
      if (!object)
        throw nb::python_error();
      return object;
    }
  };
  static const Keys &keys() {
    static const Keys *shared = new Keys();
    return *shared;
  }

  // The names of a waveform shape's numbers, where its amplitude and scale
  // factor are among them, and whether it has an overall amplitude.
  struct Shape {
    std::vector<nb::object> names;
    int real = -1, imag = -1, factor = -1;
    bool amplitude = false;
  };
  using Shapes = llvm::StringMap<Shape>;

  /// The shapes of `parameters` (each shape's parameter names) and
  /// `amplitudes` (the shapes with an overall amplitude). The exports pass
  /// the same read-only mappings every time, which are read once.
  static std::shared_ptr<const Shapes> shapesOf(nb::handle parameters,
                                                nb::handle amplitudes) {
    // Never destroyed: its names are Python objects, and static objects
    // are destroyed at exit, after the interpreter has freed them.
    struct Read {
      nb::ft_mutex mutex;
      PyObject *parameters = nullptr, *amplitudes = nullptr;
      std::shared_ptr<const Shapes> shapes;
    };
    static Read *read = new Read();
    {
      nb::ft_lock_guard guard(read->mutex);
      if (read->shapes && parameters.ptr() == read->parameters &&
          amplitudes.ptr() == read->amplitudes)
        return read->shapes;
    }
    auto shapes = std::make_shared<Shapes>();
    for (nb::handle shape : parameters) {
      Shape &info = (*shapes)[nb::cast<std::string>(shape)];
      info.amplitude = PySequence_Contains(amplitudes.ptr(), shape.ptr()) == 1;
      for (nb::handle name : parameters[shape]) {
        auto text = nb::cast<std::string>(name);
        int index = static_cast<int>(info.names.size());
        if (text == "amplitude")
          info.real = index;
        else if (text == "amplitude_imag")
          info.imag = index;
        else if (text == "factor")
          info.factor = index;
        info.names.push_back(nb::borrow(name));
      }
    }
    nb::ft_lock_guard guard(read->mutex);
    // Held while they are read, so that no other mapping takes their
    // addresses.
    Py_XDECREF(read->parameters);
    Py_XDECREF(read->amplitudes);
    read->parameters = Py_NewRef(parameters.ptr());
    read->amplitudes = Py_NewRef(amplitudes.ptr());
    read->shapes = shapes;
    return shapes;
  }

  const std::vector<ControllerRecord> &records;
  const ControllerWalk &walk;
  const ControllerWaveforms &waveforms;
  nb::object waveform;
  bool compact;
  /// Whether a value the export's vocabulary has no term for is an error
  /// (an export), or written as {"unknown": what it is computed from} (a
  /// check, which never computes it).
  bool exportable;
  const Keys &key;
  std::shared_ptr<const Shapes> shapes;
  std::vector<std::optional<std::pair<nb::object, nb::object>>> made;
  // One object per string and per float: an export repeats few of them
  // many times.
  llvm::StringMap<nb::object> strings;
  llvm::DenseMap<uint64_t, nb::object> floats;

  static nb::object checked(PyObject *object) {
    if (!object)
      throw nb::python_error();
    return nb::steal(object);
  }
  static nb::object none() { return nb::none(); }
  static nb::list list(size_t size) {
    return nb::steal<nb::list>(
        checked(PyList_New(static_cast<Py_ssize_t>(size))).release());
  }
  /// Steals `value` into slot `index` of a new `list` of more items.
  static void set(nb::handle list, size_t index, nb::object value) {
#ifdef Py_LIMITED_API
    if (PyList_SetItem(list.ptr(), static_cast<Py_ssize_t>(index),
                       value.release().ptr()) != 0)
      throw nb::python_error();
#else
    PyList_SET_ITEM(list.ptr(), static_cast<Py_ssize_t>(index),
                    value.release().ptr());
#endif
  }
  static void put(nb::handle dict, PyObject *name, nb::handle value) {
    if (PyDict_SetItem(dict.ptr(), name, value.ptr()) != 0)
      throw nb::python_error();
  }
  static nb::object integer(int64_t value) {
    return checked(PyLong_FromLongLong(value));
  }
  static nb::object optional(const std::optional<int64_t> &v) {
    return v ? integer(*v) : none();
  }
  static nb::list ints(const std::vector<int64_t> &values) {
    nb::list out = list(values.size());
    for (size_t i = 0; i < values.size(); ++i)
      set(out, i, integer(values[i]));
    return out;
  }
  nb::object text(llvm::StringRef value) {
    auto [it, inserted] = strings.try_emplace(value);
    if (inserted)
      it->second = checked(PyUnicode_FromStringAndSize(
          value.data(), static_cast<Py_ssize_t>(value.size())));
    return it->second;
  }
  nb::object real(double value) {
    auto bits = std::bit_cast<uint64_t>(value);
    // Two NaNs are the map's own empty and erased keys.
    if (bits >= llvm::DenseMapInfo<uint64_t>::getTombstoneKey())
      return checked(PyFloat_FromDouble(value));
    auto [it, inserted] = floats.try_emplace(bits);
    if (inserted)
      it->second = checked(PyFloat_FromDouble(value));
    return it->second;
  }
  /// A number, None where it is NaN.
  nb::object number(double value) {
    return std::isnan(value) ? none() : real(value);
  }

  /// The names of a node's numbers, if its shape names them all.
  const Shape *named(llvm::StringRef shape, size_t numbers) const {
    auto found = shapes->find(shape);
    if (found == shapes->end() || found->second.names.size() != numbers)
      return nullptr;
    return &found->second;
  }

  /// The waveform expression from node `index` down.
  nb::object tree(const Nodes &nodes, int64_t index) {
    const auto &[name, numbers, children] = nodes[index];
    auto [shape, custom] = llvm::StringRef(name).split(':');
    nb::dict node;
    put(node, key.shape, text(shape));
    if (!custom.empty())
      put(node, key.name, text(custom));
    const Shape *names = named(shape, numbers.size());
    if (shape == "custom_samples") {
      nb::list samples = list(numbers.size());
      for (size_t i = 0; i < numbers.size(); ++i)
        set(samples, i, real(numbers[i]));
      put(node, key.samples, samples);
    } else if (names) {
      nb::dict params;
      for (size_t i = 0; i < numbers.size(); ++i)
        put(params, names->names[i].ptr(), number(numbers[i]));
      put(node, key.params, params);
    } else if (!numbers.empty()) {
      nb::list values = list(numbers.size());
      for (size_t i = 0; i < numbers.size(); ++i)
        set(values, i, number(numbers[i]));
      put(node, key.values, values);
    }
    if (!children.empty()) {
      nb::list of = list(children.size());
      for (size_t i = 0; i < children.size(); ++i)
        set(of, i, tree(nodes, children[i]));
      put(node, key.of, of);
    }
    return node;
  }

  /// The overall complex amplitude of node `index`, through scale and neg,
  /// in Python's complex arithmetic; None where it is unknown.
  nb::object complexAmplitude(const Nodes &nodes, int64_t index) {
    const auto &[name, numbers, children] = nodes[index];
    llvm::StringRef shape = llvm::StringRef(name).split(':').first;
    const Shape *names = named(shape, numbers.size());
    if (shape == "scale" || shape == "neg") {
      if (children.empty())
        return none();
      nb::object inner = complexAmplitude(nodes, children[0]);
      if (inner.is_none())
        return inner;
      if (shape == "neg")
        return checked(PyNumber_Negative(inner.ptr()));
      if (!names || names->factor < 0 || std::isnan(numbers[names->factor]))
        return none();
      return checked(
          PyNumber_Multiply(inner.ptr(), real(numbers[names->factor]).ptr()));
    }
    if (shape == "custom_samples") {
      if (numbers.empty())
        return none();
      double peak = numbers[0];
      for (double sample : numbers)
        if (std::fabs(sample) > std::fabs(peak))
          peak = sample;
      return checked(PyComplex_FromDoubles(peak, 0.0));
    }
    if (!names || !names->amplitude || names->real < 0)
      return none();
    double re = numbers[names->real];
    double im = names->imag < 0 ? 0.0 : numbers[names->imag];
    if (std::isnan(re) || std::isnan(im))
      return none();
    return checked(PyComplex_FromDoubles(re, im));
  }

  /// (expression, amplitude as {re, im}) of a waveform, as the export has
  /// them.
  std::pair<nb::object, nb::object> built(const Nodes &nodes) {
    if (nodes.empty())
      return {none(), none()};
    int64_t root = static_cast<int64_t>(nodes.size()) - 1;
    nb::object overall = complexAmplitude(nodes, root);
    if (overall.is_none())
      return {tree(nodes, root), none()};
    nb::dict parts;
    put(parts, key.re, real(PyComplex_RealAsDouble(overall.ptr())));
    put(parts, key.im, real(PyComplex_ImagAsDouble(overall.ptr())));
    return {tree(nodes, root), parts};
  }

  /// The records `at` to `stop`, a loop's body after it, as events.
  nb::list range(size_t &at, size_t stop) {
    size_t count = 0;
    for (size_t i = at; i < stop; ++i, ++count)
      if (isKind(records[i].kind, "loop"))
        i += static_cast<size_t>(records[i].size);
    nb::list out = list(count);
    for (size_t k = 0; k < count; ++k) {
      size_t index = at++;
      const ControllerRecord &r = records[index];
      nb::dict event = common(r, index);
      if (isKind(r.kind, "loop")) {
        put(event, key.trips, integer(r.trips));
        if (!compact || r.first)
          put(event, key.first, integer(r.first));
        put(event, key.period, integer(r.period));
        put(event, key.body, range(at, at + static_cast<size_t>(r.size)));
      } else {
        details(r, event);
        if (r.known >= 0)
          runTime(r, walk.known[r.known], event);
      }
      set(out, k, std::move(event));
    }
    return out;
  }

  // The kind of an event in format /2, named for what the controller does
  // rather than for a modality.
  static const char *structuredKind(const char *kind) {
    if (isKind(kind, "drive"))
      return "play";
    if (isKind(kind, "submit"))
      return "call";
    return kind;
  }

  nb::dict common(const ControllerRecord &r, size_t index) {
    nb::dict event;
    bool sync = isKind(r.kind, "sync");
    bool loop = isKind(r.kind, "loop");
    bool call = (r.kind[0] == 's' && !sync) || r.kind[0] == 'a';
    if (!compact)
      put(event, key.index, integer(static_cast<int64_t>(index)));
    put(event, key.kind, text(compact ? structuredKind(r.kind) : r.kind));
    // A frame update on a tone that plays on no line has none.
    bool unplaced = r.kind[0] == 'f' && r.line < 0;
    if (!compact)
      put(event, key.line, sync || loop || unplaced ? none() : integer(r.line));
    else if (!sync && !loop && !call)
      put(event, key.line, unplaced ? none() : integer(r.line));
    if (!compact)
      put(event, key.qudit,
          sync || loop ? none() : integer(call ? -1 : r.qudit));
    put(event, key.start, optional(r.start));
    put(event, key.duration, optional(r.duration));
    if (!compact)
      put(event, key.end, optional(r.stop()));
    if (!compact || !r.iteration.empty()) {
      nb::list iteration = list(r.iteration.size());
      for (size_t i = 0; i < r.iteration.size(); ++i)
        set(iteration, i,
            r.iteration[i] < 0 ? none() : integer(r.iteration[i]));
      put(event, key.iteration, iteration);
    }
    int64_t segment = r.start ? r.segment : 0;
    if (!compact || segment)
      put(event, key.segment, integer(segment));
    if (r.start &&
        llvm::any_of(r.steps, [](const auto &s) { return s.segment; })) {
      nb::list steps = list(r.steps.size());
      for (size_t i = 0; i < r.steps.size(); ++i)
        set(steps, i, integer(r.steps[i].segment));
      put(event, key.segmentStep, steps);
    }
    if (!compact || !r.conditions.empty()) {
      nb::list conditions = list(r.conditions.size());
      for (size_t i = 0; i < r.conditions.size(); ++i) {
        const auto &[branch, arm, readouts, answers, test, computed, answer] =
            r.conditions[i];
        nb::dict condition;
        put(condition, key.branch, integer(branch));
        put(condition, key.arm, nb::bool_(arm != 0));
        if (!compact || !readouts.empty())
          put(condition, key.readouts, ints(readouts));
        if (!compact || !answers.empty())
          put(condition, key.answers, ints(answers));
        put(condition, key.test, text(test));
        if (answer >= 0)
          put(condition, key.result, integer(answer));
        if (computed >= 0)
          put(condition, key.expression,
              expression(computed, readouts, answers));
        set(conditions, i, std::move(condition));
      }
      put(event, key.conditions, conditions);
    }
    return event;
  }

  /// The frame of the tone of a play, readout or frame update: in format /2
  /// as its `tones`.
  void frame(const ControllerRecord &r, nb::handle event) {
    if (!compact) {
      put(event, key.frequency, number(r.frequency));
      put(event, key.phase, number(r.phase));
      return;
    }
    // A list, so that an event may later play more than one carrier.
    nb::dict tone;
    put(tone, key.frequency, number(r.frequency));
    put(tone, key.phase, number(r.phase));
    if (!r.steps.empty()) {
      nb::list steps = list(r.steps.size());
      for (size_t i = 0; i < r.steps.size(); ++i)
        set(steps, i, number(r.steps[i].phase));
      put(tone, key.phaseStep, steps);
    }
    nb::list tones = list(1);
    set(tones, 0, std::move(tone));
    put(event, key.tones, tones);
  }

  void details(const ControllerRecord &r, nb::handle event) {
    if (r.played()) {
      auto at = static_cast<size_t>(waveforms.of.lookup(r.op));
      auto &shape = made[at];
      if (!shape) {
        const Nodes &nodes = waveforms.distinct[at];
        if (waveform.is_none()) {
          shape.emplace(built(nodes));
        } else {
          nb::tuple pair = nb::cast<nb::tuple>(waveform(nodes));
          shape.emplace(pair[0], pair[1]);
        }
      }
      put(event, key.waveform, shape->first);
      if (!compact)
        put(event, key.amplitude, shape->second);
      frame(r, event);
    } else if (r.kind[0] == 'f') {
      put(event, key.op, text(r.update));
      put(event, key.value, expression(r.value));
      frame(r, event);
    } else if (isKind(r.kind, "sync")) {
      put(event, key.lines, ints(r.joined));
    } else if (r.kind[0] == 's' || r.kind[0] == 'a') {
      put(event, key.link, integer(r.line));
      if (r.qudit >= 0)
        put(event, key.controller, integer(r.qudit));
      put(event, key.function, text(r.callee));
      if (r.kind[0] == 's') {
        put(event, key.measurements, ints(r.refs));
      } else {
        put(event, key.wait, text(r.mode));
        if (!compact || !std::isnan(r.timeout))
          put(event, key.timeout, number(r.timeout));
      }
    }
  }

  // ── Run-time values ──────────────────────────────────────────────────

  /// A number as the JSON text "%.17g" of it reads back: an int where that
  /// text has no fraction or exponent.
  nb::object textNumber(double value) {
    if (std::isfinite(value) && value == std::trunc(value) &&
        std::fabs(value) < 1e17)
      return integer(static_cast<int64_t>(value));
    return real(value);
  }

  /// A branch condition's expression (see conditionExpression), with the
  /// records its leaves' places in `readouts` and `answers` hold.
  LLVM_ATTRIBUTE_NOINLINE nb::object
  expression(int32_t index, const std::vector<int64_t> &readouts,
             const std::vector<int64_t> &answers) {
    return write(index, [&](int32_t at) {
      const Expression &e = walk.expressions[at];
      nb::object out = leaf(at);
      if (e.kind == Expression::Readout)
        put(out, key.readout, integer(readouts[e.event]));
      else if (e.kind == Expression::Answer)
        put(out, key.answer, integer(answers[e.event]));
      return out;
    });
  }

  /// Expression `index` over what `leaf` writes of what is no term: a term
  /// in place where one term uses it, else in the `terms` of the
  /// expression, by its place there where it is used (see namedTerms).
  template <typename Leaf>
  nb::object write(int32_t index, Leaf leaf) {
    return walk.written(
        index, leaf,
        [&](int n) {
          nb::dict out;
          put(out, key.term, integer(n));
          return nb::object(out);
        },
        [&](const Expression &e, nb::handle of, nb::handle table) {
          nb::dict out;
          if (table.is_valid())
            put(out, key.table, table);
          put(out, key.op, key.terms[e.term]);
          put(out, key.type, key.types[e.type]);
          put(out, key.of, of);
          return nb::object(out);
        });
  }

  /// Expression `index` as the export gives it.
  nb::object expression(int32_t index) {
    return write(index, [&](int32_t at) { return leaf(at); });
  }

  /// What is no term, as the export gives it.
  nb::object leaf(int32_t index) {
    const Expression &e = walk.expressions[index];
    if (e.kind == Expression::Number)
      return textNumber(e.number);
    if (e.kind == Expression::Unknown) {
      if (exportable)
        unexportable(e.name);
      nb::dict out;
      put(out, key.unknown, text(e.name));
      return out;
    }
    nb::dict out;
    switch (e.kind) {
    case Expression::Constant:
      if (e.type == Expression::Bool)
        put(out, key.constant, nb::bool_(e.number != 0.0));
      else
        put(out, key.constant, textNumber(e.number));
      break;
    case Expression::Readout:
      put(out, key.readout, integer(e.event));
      break;
    case Expression::Answer:
      put(out, key.answer, integer(e.event));
      put(out, key.result, integer(e.result));
      break;
    default:
      break;
    }
    return out;
  }

  /// Raises UnsupportedError for a value computed at run time from `what`
  /// (an op's MLIR name, else what it is), which no expression names.
  [[noreturn]] static void unexportable(llvm::StringRef what) {
    std::string message =
        "cannot export a value the program computes at run time from ";
    if (what.contains('.'))
      message += "the operation '" + what.str() + "'";
    else
      message += what.str();
    message += ": a controller program computes only the terms of its "
               "expression vocabulary (see the controller backends guide); "
               "compute this value while the kernel compiles";
    nb::object error =
        nb::module_::import_("cudaq_pulse._errors").attr("UnsupportedError");
    PyErr_SetString(error.ptr(), message.c_str());
    throw nb::python_error();
  }

  /// The name of number `operand` of a node: its parameter's, where the
  /// shape names all its numbers, else its index.
  nb::object parameter(const std::string &name, size_t numbers,
                       int64_t operand) {
    if (const Shape *names =
            named(llvm::StringRef(name).split(':').first, numbers))
      return names->names[static_cast<size_t>(operand)];
    return text(std::to_string(operand));
  }

  /// The run-time operands of node `index`, by parameter name; none when it
  /// has none.
  nb::object computed(const Nodes &nodes, int64_t index,
                      const Operands &operands) {
    const auto &[name, numbers, children] = nodes[index];
    nb::dict out;
    for (const auto &[at, operand, expr] : operands)
      if (at == index)
        put(out, parameter(name, numbers.size(), operand).ptr(),
            expression(expr));
    return NB_DICT_GET_SIZE(out.ptr()) ? nb::object(out) : none();
  }

  /// A copy of waveform `tree` (node `index` of `nodes`) with its operands
  /// known only at run time in the `runtime` of their nodes.
  nb::object withRunTime(const Nodes &nodes, int64_t index, nb::handle tree,
                         const Operands &operands) {
    const auto &children = std::get<2>(nodes[index]);
    nb::object node = checked(PyDict_Copy(tree.ptr()));
    if (!children.empty()) {
      // Borrowed from `node`, a copy no other thread sees.
      PyObject *old = PyDict_GetItem(node.ptr(), key.of);
      nb::list of = list(children.size());
      for (size_t i = 0; i < children.size(); ++i)
        set(of, i, withRunTime(nodes, children[i], listItem(old, i), operands));
      put(node, key.of, of);
    }
    nb::object here = computed(nodes, index, operands);
    if (!here.is_none())
      put(node, key.runtime, here);
    return node;
  }

  /// Format /2: the run-time operands of a waveform as a list of the paths
  /// to their nodes (the index in `of` at each node from the root) with
  /// their operands by parameter name.
  void paths(const Nodes &nodes, int64_t index, nb::list &path, nb::list &out,
             const Operands &operands) {
    nb::object here = computed(nodes, index, operands);
    if (!here.is_none()) {
      nb::dict item;
      put(item, key.path,
          checked(PyList_GetSlice(path.ptr(), 0, PY_SSIZE_T_MAX)));
      put(item, key.params, here);
      out.append(item);
    }
    const auto &children = std::get<2>(nodes[index]);
    for (size_t i = 0; i < children.size(); ++i) {
      path.append(i);
      paths(nodes, children[i], path, out, operands);
      PyList_SetSlice(path.ptr(), nb::len(path) - 1, nb::len(path), nullptr);
    }
  }

  nb::list frameUpdates(const std::vector<ControllerWalk::FrameUpdate> &all) {
    nb::list out = list(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
      const auto &update = all[i];
      nb::list conditions = list(update.conditions.size());
      for (size_t j = 0; j < update.conditions.size(); ++j) {
        const auto &[branch, arm, readouts, answers, test, computed, answer] =
            update.conditions[j];
        nb::dict condition;
        if (!compact || !answers.empty())
          put(condition, key.answers, ints(answers));
        put(condition, key.arm, nb::bool_(arm != 0));
        put(condition, key.branch, integer(branch));
        if (computed >= 0)
          put(condition, key.expression,
              expression(computed, readouts, answers));
        if (!compact || !readouts.empty())
          put(condition, key.readouts, ints(readouts));
        if (answer >= 0)
          put(condition, key.result, integer(answer));
        put(condition, key.test, text(test));
        set(conditions, j, std::move(condition));
      }
      nb::dict item;
      if (!compact || !update.conditions.empty())
        put(item, key.conditions, conditions);
      put(item, key.id, integer(static_cast<int64_t>(update.seq)));
      put(item, key.op, text(update.op));
      put(item, key.start, optional(update.start));
      put(item, key.value, expression(update.value));
      set(out, i, std::move(item));
    }
    return out;
  }

  /// What `r` has that is known only at run time, after the rest of
  /// `event`; format /1 adds the waveform's in place, in its nodes.
  void runTime(const ControllerRecord &r, const ControllerWalk::Known &known,
               nb::handle event) {
    if (!known.waveform.empty() && r.played()) {
      const Nodes &nodes = waveforms.distinct[waveforms.of.lookup(r.op)];
      Operands sorted = known.waveform;
      std::stable_sort(sorted.begin(), sorted.end(),
                       [](const auto &a, const auto &b) {
                         return std::make_pair(std::get<0>(a), std::get<1>(a)) <
                                std::make_pair(std::get<0>(b), std::get<1>(b));
                       });
      int64_t root = static_cast<int64_t>(nodes.size()) - 1;
      if (!compact) {
        put(event, key.waveform,
            withRunTime(nodes, root, PyDict_GetItem(event.ptr(), key.waveform),
                        sorted));
      } else {
        nb::list path, out;
        paths(nodes, root, path, out, sorted);
        put(event, key.runtime, out);
      }
    }
    if (known.tone >= 0) {
      put(event, key.tone, integer(known.tone));
      put(event, key.frameUpdates, frameUpdates(known.updates));
      if (known.offset)
        put(event, key.phaseOffset, textNumber(*known.offset));
    }
    if (known.length >= 0) {
      put(event, key.length, expression(known.length));
      if (!compact || known.minCycles)
        put(event, key.minCycles, optional(known.minCycles));
      if (!compact || known.maxCycles)
        put(event, key.maxCycles, optional(known.maxCycles));
    }
    if (known.operands) {
      nb::list operands = list(known.operands->size());
      for (size_t i = 0; i < known.operands->size(); ++i)
        set(operands, i, expression((*known.operands)[i]));
      put(event, key.operands, operands);
    }
  }
};
