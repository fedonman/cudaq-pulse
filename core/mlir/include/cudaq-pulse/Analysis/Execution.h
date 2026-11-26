// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Walk a scheduled pulse program in execution order along one path.
//
// Loops run every iteration. A measurement-conditioned `scf.if` takes the
// arm that the path's outcome for that measurement selects. Each tone's
// frame (frequency and phase) is tracked as frame changes and drives
// execute, so every drive occurrence sees the frame it plays in. Frame
// values are affine in the entry function's float arguments (Affine.h), so a
// program whose frequencies and phases are kernel arguments walks without
// binding them.
//
// Include after the Pulse dialect op and type headers.

#pragma once

#include "cudaq-pulse/Analysis/Affine.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include <cmath>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace pulse {

/// A tone's rotating frame. `phase` is the frame phase at `timeNs`, the end
/// of its latest drive or the place of a later frame change; from there a
/// tone detuned from its frame frequency accrues phase.
struct ToneState {
  Affine frequencyHz = 0.0;
  Affine frameFrequencyHz = 0.0;
  Affine phase = 0.0;
  double timeNs = 0.0;
};

/// One execution of a readout: the op and its loop iteration.
struct MeasurementKey {
  mlir::Operation *readout = nullptr;
  std::vector<int64_t> iteration;
  bool operator<(const MeasurementKey &other) const {
    return std::tie(readout, iteration) <
           std::tie(other.readout, other.iteration);
  }
};

/// The outcome chosen for each measurement on one path.
using Path = std::map<MeasurementKey, int64_t>;

/// One execution of a drive, readout or wait.
struct Occurrence {
  mlir::Operation *op = nullptr;
  std::vector<int64_t> iteration;
  int64_t startCycles = 0;
  int64_t durationCycles = 0;
  /// For drives and readouts: the frame of the op's tone as it starts.
  ToneState tone;
};

inline std::optional<double> constantNumber(mlir::Value value) {
  if (auto answer = boundAnswer(value))
    return answer;
  mlir::Attribute attribute = constantAttribute(value);
  if (!attribute)
    return std::nullopt;
  if (auto number = mlir::dyn_cast<mlir::FloatAttr>(attribute))
    return number.getValueAsDouble();
  if (auto number = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
    return static_cast<double>(number.getInt());
  return std::nullopt;
}

/// The allocation, or `pulse.tone`, that a tone value descends from, for
/// walks that visit every op of a program. It remembers every value on the
/// way: a tone threads through every drive and frame change on its line, so
/// walking back from each one would make the walk quadratic.
class ToneRoots {
public:
  mlir::Operation *operator()(mlir::Value tone) {
    llvm::SmallVector<mlir::Value, 16> chain;
    mlir::Operation *root = nullptr;
    mlir::Value current = tone;
    while (true) {
      auto known = roots.find(current);
      if (known != roots.end()) {
        root = known->second;
        break;
      }
      chain.push_back(current);
      mlir::Value previous = linearPredecessor(current);
      if (!previous) {
        root = current.getDefiningOp();
        break;
      }
      current = previous;
    }
    for (mlir::Value value : chain)
      roots[value] = root;
    return root;
  }

private:
  llvm::DenseMap<mlir::Value, mlir::Operation *> roots;
};

/// On a tone's allocation or `pulse.tone`: its frame where the program
/// starts, as [frequency Hz, frame frequency Hz, phase, time ns].
inline constexpr llvm::StringLiteral kToneStateAttr = "pulse.tone_state";

class ExecutionWalker {
public:
  ExecutionWalker(double clockGHz, const Path &path)
      : clockGHz(clockGHz), path(path) {}

  /// Called for every drive, readout and wait occurrence.
  std::function<void(const Occurrence &)> onTimed;
  /// Called for every drive occurrence, with its tone frame.
  std::function<void(const Occurrence &)> onDrive;
  /// Called for every readout occurrence; returning true ends the walk.
  std::function<bool(const Occurrence &, const MeasurementKey &)> onReadout;
  /// Give readout occurrences their tone's frame (drives always have it).
  bool readoutTones = false;
  /// Called for every `classical.submit` occurrence; returning true ends
  /// the walk. A branch on an answer no constant stands for yet is skipped,
  /// like one on a measurement without an outcome: its arms run after the
  /// answer is due.
  std::function<bool(const Occurrence &)> onSubmit;
  /// An op of the function's body the walk ends at, before running it.
  mlir::Operation *stopBefore = nullptr;

  /// The readout whose measurement `value` is, if the walk has seen it.
  std::optional<MeasurementKey> measurementKeyOf(mlir::Value value) const {
    auto it = measurementOf.find(value);
    if (it == measurementOf.end())
      return std::nullopt;
    return it->second;
  }

  /// A number computed from constants and bound answers alone.
  static std::optional<double> number(mlir::Value value) {
    return boundNumber(value);
  }

  /// Walk `function`; on failure `errorOp` and `error` say why.
  bool run(mlir::func::FuncOp function) {
    return block(function.getBody().front());
  }

  /// Each tone's frame where the walk ended, by the allocation or
  /// `pulse.tone` it descends from.
  const llvm::DenseMap<mlir::Operation *, ToneState> &toneStates() const {
    return tones;
  }

  /// Measurements without an outcome on the path that a branch tested. The
  /// walk skips such a branch: its arms run after the measurement completes.
  std::vector<MeasurementKey> undecided;
  mlir::Operation *errorOp = nullptr;
  std::string error;

private:
  bool fail(mlir::Operation *op, const std::string &message) {
    errorOp = op;
    error = message;
    return false;
  }

  ToneState &toneOf(mlir::Value tone) {
    mlir::Operation *root = roots(tone);
    auto [it, inserted] = tones.try_emplace(root);
    if (inserted && root) {
      // A program that starts part-way through a shot (a simulation
      // window) gives each tone the frame it is in there.
      if (auto seed =
              root->getAttrOfType<mlir::DenseF64ArrayAttr>(kToneStateAttr);
          seed && seed.size() == 4)
        it->second = {seed[0], seed[1], seed[2], seed[3]};
      else if (auto frequency =
                   root->getAttrOfType<mlir::FloatAttr>("frequency_hz"))
        it->second = {frequency.getValueAsDouble(),
                      frequency.getValueAsDouble(), 0.0, 0.0};
      else if (root->getName().getStringRef() == "pulse.tone" &&
               root->getNumOperands() == 2) {
        Affine frequency = affineNumber(root->getOperand(0)).value_or(0.0);
        it->second = {frequency, frequency,
                      affineNumber(root->getOperand(1)).value_or(0.0), 0.0};
      }
    }
    return it->second;
  }

  bool timed(mlir::Operation *op, Occurrence &occurrence) {
    auto start = scheduledStart(op, iteration);
    auto duration = scheduledDuration(op);
    if (!start || !duration)
      return fail(op, "operation is not scheduled; schedule the program and "
                      "specialize all numeric values first");
    occurrence = {op, iteration, *start, *duration, {}};
    if (onTimed)
      onTimed(occurrence);
    return true;
  }

  bool block(mlir::Block &body) {
    for (mlir::Operation &op : body.without_terminator()) {
      if (&op == stopBefore) {
        stopped = true;
        return true;
      }
      if (!step(&op))
        return false;
      if (stopped)
        return true;
    }
    return true;
  }

  bool step(mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (auto drive = mlir::dyn_cast<DriveOp>(op)) {
      Occurrence occurrence;
      if (!timed(op, occurrence))
        return false;
      ToneState &tone = toneOf(drive.getTone());
      occurrence.tone = tone;
      if (onDrive)
        onDrive(occurrence);
      if (auto frame = op->getAttrOfType<mlir::FloatAttr>("frame_phase_offset"))
        tone.phase += frame.getValueAsDouble();
      double endNs = static_cast<double>(occurrence.startCycles +
                                         occurrence.durationCycles) /
                     clockGHz;
      tone.phase += (tone.frequencyHz - tone.frameFrequencyHz) * (2.0 * M_PI) *
                    1.0e-9 * (endNs - tone.timeNs);
      tone.timeNs = endNs;
      return true;
    }
    if (auto readout = mlir::dyn_cast<ReadoutOp>(op)) {
      Occurrence occurrence;
      if (!timed(op, occurrence))
        return false;
      if (readoutTones)
        occurrence.tone = toneOf(readout.getTone());
      MeasurementKey key{op, iteration};
      measurementOf[readout.getResult()] = key;
      if (onReadout && onReadout(occurrence, key))
        stopped = true;
      return true;
    }
    if (mlir::isa<WaitOp>(op)) {
      Occurrence occurrence;
      return timed(op, occurrence);
    }
    if (name == "classical.submit") {
      Occurrence occurrence;
      if (!timed(op, occurrence))
        return false;
      if (onSubmit && onSubmit(occurrence))
        stopped = true;
      return true;
    }
    if (name == "classical.wait")
      return true;
    if (name == "pulse.shift_phase" || name == "pulse.set_phase" ||
        name == "pulse.shift_frequency" || name == "pulse.set_frequency") {
      auto value = affineNumber(op->getOperand(1));
      if (!value) {
        // An answer not bound yet: the change happens after it is due.
        if (pendingAnswer(op->getOperand(1)))
          return true;
        return fail(op, "frame change is neither a constant nor affine in the "
                        "kernel's float arguments");
      }
      ToneState &tone = toneOf(op->getOperand(0));
      // The frame runs at its old frequency until the change: after a wait,
      // that is later than its tone's latest drive.
      if (auto start = scheduledStart(op, iteration)) {
        double changeNs = static_cast<double>(*start) / clockGHz;
        if (changeNs > tone.timeNs) {
          tone.phase += (tone.frequencyHz - tone.frameFrequencyHz) *
                        (2.0 * M_PI) * 1.0e-9 * (changeNs - tone.timeNs);
          tone.timeNs = changeNs;
        }
      }
      if (name == "pulse.shift_phase")
        tone.phase += *value;
      else if (name == "pulse.set_phase")
        tone.phase = *value;
      else if (name == "pulse.shift_frequency")
        tone.frequencyHz += *value;
      else
        tone.frequencyHz = *value;
      return true;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op))
      return forLoop(loop);
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op))
      return ifOp(branch);
    // A call plays its callee's pulses, which are walked once it is written
    // out (pulse-expand-calls).
    if (name == "pulse.call")
      return fail(op, "a pulse.call is run once it is written out; run "
                      "pulse-expand-calls first");
    return true;
  }

  void carry(mlir::ValueRange from, mlir::ValueRange to) {
    for (auto [source, target] : llvm::zip(from, to)) {
      auto it = measurementOf.find(source);
      if (it != measurementOf.end())
        measurementOf[target] = it->second;
    }
  }

  bool forLoop(mlir::scf::ForOp loop) {
    auto trips = constantTripCount(loop);
    if (!trips)
      return fail(loop, "loop bounds are not constants");
    auto yield = loop.getBody()->getTerminator()->getOperands();
    carry(loop.getInitArgs(), loop.getRegionIterArgs());
    iteration.push_back(0);
    for (int64_t k = 0; k < *trips && !stopped; ++k) {
      iteration.back() = k;
      if (!block(*loop.getBody()))
        return false;
      carry(yield, loop.getRegionIterArgs());
    }
    iteration.pop_back();
    carry(*trips ? yield : loop.getInitArgs(), loop.getResults());
    return true;
  }

  /// Whether `value` is computed from an answer of a classical call that no
  /// constant stands for yet.
  static bool pendingAnswer(mlir::Value value) {
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      mlir::Operation *def = v.getDefiningOp();
      if (!def)
        continue;
      if (def->getName().getStringRef() == "classical.wait") {
        if (!def->hasAttr("classical.answers"))
          return true;
        continue;
      }
      if (def->getNumRegions() == 0)
        work.append(def->getOperands().begin(), def->getOperands().end());
    }
    return false;
  }

  /// The value of a branch condition, computed from measurement outcomes
  /// and bound answers as a controller computes the condition's
  /// expression.
  std::optional<double> evaluate(mlir::Value value) {
    auto outcome = [&](mlir::Value measurement) -> std::optional<int64_t> {
      auto key = measurementOf.find(measurement);
      if (key == measurementOf.end())
        return std::nullopt;
      auto chosen = path.find(key->second);
      if (chosen == path.end()) {
        undecided.push_back(key->second);
        return std::nullopt;
      }
      return chosen->second;
    };
    auto leaf = [&](mlir::Value v) -> std::optional<double> {
      if (auto answer = boundAnswer(v))
        return answer;
      mlir::Operation *def = v.getDefiningOp();
      if (auto isSet = mlir::dyn_cast_or_null<IsSetOp>(def)) {
        auto level = outcome(isSet.getMeasurement());
        return level ? std::optional<double>(*level != 0) : std::nullopt;
      }
      if (auto level = mlir::dyn_cast_or_null<OutcomeOp>(def)) {
        auto chosen = outcome(level.getMeasurement());
        return chosen ? std::optional<double>(*chosen) : std::nullopt;
      }
      return std::nullopt;
    };
    return computedNumber(value, leaf);
  }

  bool ifOp(mlir::scf::IfOp branch) {
    size_t known = undecided.size();
    auto taken = evaluate(branch.getCondition());
    if (!taken) {
      if (undecided.size() > known || pendingAnswer(branch.getCondition()))
        return true;
      return fail(branch, "cannot evaluate this branch condition from "
                          "measurement outcomes");
    }
    if (*taken == 0.0 && branch.getElseRegion().empty())
      return true;
    mlir::Block &arm =
        *taken != 0.0 ? *branch.thenBlock() : *branch.elseBlock();
    if (!block(arm))
      return false;
    carry(arm.getTerminator()->getOperands(), branch.getResults());
    return true;
  }

  double clockGHz;
  const Path &path;
  llvm::DenseMap<mlir::Operation *, ToneState> tones;
  ToneRoots roots;
  llvm::DenseMap<mlir::Value, MeasurementKey> measurementOf;
  std::vector<int64_t> iteration;
  bool stopped = false;
};

} // namespace pulse
