// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PulseToQOp conversion pass: lower a scheduled pulse program to the qop
// Hamiltonian/Lindblad construction for one time evolution.
//
// Drives are found by walking the program in execution order
// (Analysis/Execution.h), so retained loops and measurement-conditioned
// branches lower without unrolling the IR. A drive op that plays in every
// iteration of its loops with a fixed period and phase step becomes one
// pulse-train callback. A program that measures is lowered along one
// measurement-outcome path, given as the module attribute `pulse.path`
// (entries [readout index, iteration..., outcome], readouts numbered in
// walk order), over the window [qop.t_start, qop.t_end]: the window must
// end by the first measurement on the path that has no outcome, and the
// caller projects the state at each measurement between windows.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
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

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include "cudaq-pulse/Analysis/Execution.h"
#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Runtime/WaveformCallback.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <optional>

using namespace mlir;

namespace {

using pulse::Affine;
using pulse::affineNumber;
using pulse::constantNumber;

/// A drive callback's kind and parameters. A parameter is affine in the
/// program's float arguments; a constant parameter has no argument terms.
struct WaveformCallbackData {
  int32_t kind;
  SmallVector<Affine> parameters;
};

/// The parameters of `callback` if none depends on the arguments.
static std::optional<SmallVector<double>>
constantParameters(const WaveformCallbackData &callback) {
  SmallVector<double> values;
  for (const Affine &parameter : callback.parameters) {
    if (!parameter.isConstant())
      return std::nullopt;
    values.push_back(parameter.constant);
  }
  return values;
}

/// The phase attributes that pulse-fold-phases leaves on a drive.
static double phaseOffsetOf(Operation *drive) {
  double offset = 0.0;
  for (StringRef name : {"phase_offset", "frame_phase_offset"})
    if (auto attribute = drive->getAttrOfType<FloatAttr>(name))
      offset += attribute.getValueAsDouble();
  return offset;
}

/// The parts of a drive occurrence's callback that vary between occurrences
/// of the same op.
struct DriveTiming {
  double startNs, durationNs;
  Affine phase, detuning;
};

/// In the qudit's frame a tone at w_tone drives a^dagger with
/// E(t) = envelope e^{-i theta(t)}, theta(t) = theta_0 + (w_tone - w_q) t:
/// the tone runs ahead of the frame at the detuning. theta_0 holds the
/// tone's frame phase, which shift_phase(theta) advances by theta, so a
/// shift_phase(theta) before a pulse is a virtual Z(theta) (McKay et al.,
/// PRA 96, 022330 (2017)): the pulse turns about an axis at -theta.
static DriveTiming driveTiming(const pulse::Occurrence &occurrence,
                               double clockGHz, double qubitFrequencyHz,
                               double phaseOffset) {
  const double scale = 1.0 / clockGHz;
  const double startNs = static_cast<double>(occurrence.startCycles) * scale;
  const pulse::ToneState &tone = occurrence.tone;
  const Affine qubitFrequency(qubitFrequencyHz);
  const Affine theta = tone.phase +
                       (tone.frequencyHz - tone.frameFrequencyHz) *
                           (2.0 * M_PI) * 1.0e-9 * (startNs - tone.timeNs) +
                       (tone.frameFrequencyHz - qubitFrequency) * (2.0 * M_PI) *
                           1.0e-9 * startNs +
                       Affine(phaseOffset);
  return {startNs, static_cast<double>(occurrence.durationCycles) * scale,
          -theta, (qubitFrequency - tone.frequencyHz) * (2.0 * M_PI) * 1.0e-9};
}

/// The callback of `waveform`, a built-in or custom_samples op, played with
/// `timing` and multiplied by `coefficient` times the real `factor`.
static std::optional<WaveformCallbackData>
leafCallback(Operation *waveform, const DriveTiming &timing, double clockGHz,
             double driveScale, int32_t quadrature,
             std::complex<double> coefficient, const Affine &factor = 1.0) {
  const double scale = 1.0 / clockGHz;
  const double gain = std::abs(coefficient) * driveScale;
  // Common layout: start, duration, real amplitude, imaginary amplitude,
  // sigma, beta, rise/fall, phase-at-start, quadrature, detuning (rad/ns),
  // followed by sample data. A complex coefficient advances the phase.
  WaveformCallbackData result{0,
                              {timing.startNs, timing.durationNs, 0.0, 0.0, 0.0,
                               0.0, 0.0, timing.phase + std::arg(coefficient),
                               static_cast<double>(quadrature),
                               timing.detuning}};
  auto scaled = [&](Value value) -> std::optional<Affine> {
    auto number = affineNumber(value);
    if (!number)
      return std::nullopt;
    auto product = number->times(factor);
    if (!product)
      return std::nullopt;
    return *product * gain;
  };
  auto set = [&](int32_t kind, Value amplitude, Value sigma = {},
                 Value beta = {}) -> bool {
    auto amplitudeValue = scaled(amplitude);
    if (!amplitudeValue)
      return false;
    result.kind = kind;
    result.parameters[2] = *amplitudeValue;
    if (sigma) {
      auto sigmaValue = affineNumber(sigma);
      if (!sigmaValue)
        return false;
      result.parameters[4] = *sigmaValue * scale;
    }
    if (beta) {
      auto betaValue = affineNumber(beta);
      if (!betaValue)
        return false;
      result.parameters[5] = *betaValue * scale;
    }
    return true;
  };

  if (auto op = dyn_cast<pulse::SquarePulseOp>(waveform)) {
    auto real = scaled(op.getAmpReal());
    auto imag = scaled(op.getAmpImag());
    if (!real || !imag)
      return std::nullopt;
    result.kind = 1;
    result.parameters[2] = *real;
    result.parameters[3] = *imag;
  } else if (auto op = dyn_cast<pulse::GaussianPulseOp>(waveform)) {
    if (!set(2, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::DRAGPulseOp>(waveform)) {
    if (!set(3, op.getAmplitude(), op.getSigma(), op.getBeta()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::CosinePulseOp>(waveform)) {
    if (!set(4, op.getAmplitude()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::TanhRampOp>(waveform)) {
    if (!set(5, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::GaussianSquarePulseOp>(waveform)) {
    if (!set(6, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
    auto riseFall = constantNumber(op.getRisefall());
    if (!riseFall)
      return std::nullopt;
    result.parameters[6] = *riseFall * scale;
  } else if (auto op = dyn_cast<pulse::CustomSamplesOp>(waveform)) {
    result.kind = 7;
    for (Attribute sample : op.getSamples()) {
      auto value = dyn_cast<FloatAttr>(sample);
      if (!value)
        return std::nullopt;
      auto product = factor.times(value.getValueAsDouble());
      result.parameters.push_back(*product * gain);
    }
  } else {
    return std::nullopt;
  }
  return result;
}

/// The complex envelope of `waveform` at `time` ns after it starts, before
/// the carrier and the drive scale are applied. Sampled waveforms take
/// constant parameters only.
static std::optional<std::complex<double>>
envelopeAt(Value waveform, double time, double durationNs, double clockGHz) {
  Operation *op = waveform.getDefiningOp();
  if (!op)
    return std::nullopt;
  auto binary = [&](Value lhs, Value rhs,
                    auto combine) -> std::optional<std::complex<double>> {
    auto left = envelopeAt(lhs, time, durationNs, clockGHz);
    auto right = envelopeAt(rhs, time, durationNs, clockGHz);
    if (!left || !right)
      return std::nullopt;
    return combine(*left, *right);
  };
  if (auto add = dyn_cast<pulse::PulseAddOp>(op))
    return binary(add.getLhs(), add.getRhs(), std::plus<>());
  if (auto sub = dyn_cast<pulse::PulseSubOp>(op))
    return binary(sub.getLhs(), sub.getRhs(), std::minus<>());
  if (auto mul = dyn_cast<pulse::PulseMulOp>(op))
    return binary(mul.getLhs(), mul.getRhs(), std::multiplies<>());
  if (auto neg = dyn_cast<pulse::PulseNegOp>(op)) {
    auto value = envelopeAt(neg.getPulse(), time, durationNs, clockGHz);
    return value ? std::optional(-*value) : std::nullopt;
  }
  if (auto scaled = dyn_cast<pulse::PulseScaleOp>(op)) {
    auto factor = constantNumber(scaled.getScale());
    auto value = envelopeAt(scaled.getPulse(), time, durationNs, clockGHz);
    if (!factor || !value)
      return std::nullopt;
    return *factor * *value;
  }
  // With no carrier, the X and Y callbacks are half the envelope's real and
  // imaginary parts.
  const DriveTiming local{0.0, durationNs, 0.0, 0.0};
  double parts[2];
  for (int32_t quadrature : {0, 1}) {
    auto callback = leafCallback(op, local, clockGHz, 1.0, quadrature, 1.0);
    auto parameters = callback ? constantParameters(*callback) : std::nullopt;
    if (!parameters || cudaq_pulse::evaluateWaveform(
                           callback->kind, parameters->data(),
                           parameters->size(), time, parts[quadrature]))
      return std::nullopt;
  }
  return std::complex<double>(2.0 * parts[0], 2.0 * parts[1]);
}

/// A waveform as a linear combination of terms: built-in waveforms, and
/// products, which are sampled. A term's weight is a constant complex
/// coefficient times a real factor that may depend on the arguments.
struct WaveformTerm {
  Operation *waveform;
  std::complex<double> coefficient;
  Affine factor;
};

static bool collectTerms(Value waveform, std::complex<double> coefficient,
                         const Affine &factor,
                         SmallVectorImpl<WaveformTerm> &terms) {
  Operation *op = waveform.getDefiningOp();
  if (!op)
    return false;
  if (auto add = dyn_cast<pulse::PulseAddOp>(op))
    return collectTerms(add.getLhs(), coefficient, factor, terms) &&
           collectTerms(add.getRhs(), coefficient, factor, terms);
  if (auto sub = dyn_cast<pulse::PulseSubOp>(op))
    return collectTerms(sub.getLhs(), coefficient, factor, terms) &&
           collectTerms(sub.getRhs(), -coefficient, factor, terms);
  if (auto neg = dyn_cast<pulse::PulseNegOp>(op))
    return collectTerms(neg.getPulse(), -coefficient, factor, terms);
  if (auto scaled = dyn_cast<pulse::PulseScaleOp>(op)) {
    auto scale = affineNumber(scaled.getScale());
    if (!scale)
      return false;
    if (scale->isConstant())
      return collectTerms(scaled.getPulse(), coefficient * scale->constant,
                          factor, terms);
    auto product = factor.times(*scale);
    return product &&
           collectTerms(scaled.getPulse(), coefficient, *product, terms);
  }
  terms.push_back({op, coefficient, factor});
  return true;
}

/// The callbacks of one drive occurrence for one quadrature: one per term
/// of its waveform. The control Hamiltonian is linear in the envelope, so a
/// sum of waveforms is exactly a sum of callbacks. A product is sampled once
/// per cycle of the line's controller (every `grid` program cycles) into
/// piecewise-constant samples, its real and imaginary parts as two callbacks.
static std::optional<SmallVector<WaveformCallbackData>>
getWaveformCallbacks(const pulse::Occurrence &occurrence, double clockGHz,
                     double qubitFrequencyHz, double driveScale,
                     int32_t quadrature, int64_t grid) {
  Operation *drive = occurrence.op;
  SmallVector<WaveformTerm> terms;
  if (!collectTerms(drive->getOperand(1), 1.0, 1.0, terms))
    return std::nullopt;
  const DriveTiming timing =
      driveTiming(occurrence, clockGHz, qubitFrequencyHz, phaseOffsetOf(drive));
  SmallVector<WaveformCallbackData> callbacks;
  for (const WaveformTerm &term : terms) {
    if (!isa<pulse::PulseMulOp>(term.waveform)) {
      auto callback = leafCallback(term.waveform, timing, clockGHz, driveScale,
                                   quadrature, term.coefficient, term.factor);
      if (!callback)
        return std::nullopt;
      callbacks.push_back(std::move(*callback));
      continue;
    }
    const int64_t step =
        grid > 1 && occurrence.durationCycles % grid == 0 ? grid : 1;
    const auto count = occurrence.durationCycles / step;
    SmallVector<double> real, imaginary;
    for (int64_t k = 0; k < count; ++k) {
      auto value = envelopeAt(term.waveform->getResult(0),
                              (k + 0.5) * static_cast<double>(step) / clockGHz,
                              timing.durationNs, clockGHz);
      if (!value)
        return std::nullopt;
      real.push_back(value->real());
      imaginary.push_back(value->imag());
    }
    auto samples = [&](ArrayRef<double> values, double phase) {
      const double gain = std::abs(term.coefficient) * driveScale;
      WaveformCallbackData callback{
          7,
          {timing.startNs, timing.durationNs, 0.0, 0.0, 0.0, 0.0, 0.0,
           timing.phase + std::arg(term.coefficient) + phase,
           static_cast<double>(quadrature), timing.detuning}};
      for (double value : values)
        callback.parameters.push_back(term.factor * (value * gain));
      callbacks.push_back(std::move(callback));
    };
    samples(real, 0.0);
    if (llvm::any_of(imaginary, [](double v) { return v != 0.0; }))
      samples(imaginary, M_PI / 2.0);
  }
  return callbacks;
}

/// Leaf operators on qudit `target`: Pauli spin handlers for two levels,
/// otherwise d x d matrices built from the ladder operator a, with
/// X = a + a^dagger, Y = i (a^dagger - a), Z = 1 - 2 n. The number and
/// anharmonic operators are matrices at every d.
class QuditOperators {
public:
  /// `modeOf` maps each simulated qudit to its mode; empty, mode and qudit
  /// indices are the same.
  QuditOperators(OpBuilder &b, Location loc,
                 const DenseMap<int64_t, int64_t> &levels,
                 const DenseMap<int64_t, int64_t> &modeOf)
      : b(b), loc(loc), levels(levels), modeOf(modeOf) {}

  int64_t levelsOf(int64_t qudit) const {
    auto it = levels.find(qudit);
    return it == levels.end() ? 2 : it->second;
  }

  bool simulated(int64_t qudit) const {
    return modeOf.empty() || modeOf.count(qudit);
  }

  /// A qudit an operator acted on that is not simulated, if any.
  std::optional<int64_t> unsimulated() const { return outside; }

  Value leaf(int64_t qudit, StringRef kind) {
    auto handlerTy = qop::HandlerType::get(b.getContext());
    int64_t d = levelsOf(qudit);
    int64_t mode = qudit;
    if (!modeOf.empty()) {
      auto it = modeOf.find(qudit);
      if (it == modeOf.end()) {
        outside = qudit;
        mode = 0;
      } else {
        mode = it->second;
      }
    }
    auto parsed = qop::symbolizeHandlerKind(kind);
    if (d == 2 && parsed) {
      auto target =
          arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(mode));
      return qop::SpinOp::create(
          b, loc, handlerTy, target,
          qop::HandlerKindAttr::get(b.getContext(), *parsed));
    }
    // Column-major, interleaved real and imaginary parts.
    SmallVector<double> data(2 * d * d, 0.0);
    auto set = [&](int64_t row, int64_t column, double re, double im) {
      data[2 * (column * d + row)] += re;
      data[2 * (column * d + row) + 1] += im;
    };
    for (int64_t n = 1; n < d; ++n) {
      double amplitude = std::sqrt(static_cast<double>(n));
      // a |n> = sqrt(n) |n-1>: entry (n-1, n); a^dagger: entry (n, n-1).
      if (kind == "spin_x") {
        set(n - 1, n, amplitude, 0.0);
        set(n, n - 1, amplitude, 0.0);
      } else if (kind == "spin_y") {
        set(n, n - 1, 0.0, amplitude);
        set(n - 1, n, 0.0, -amplitude);
      } else if (kind == "spin_lowering") {
        set(n - 1, n, amplitude, 0.0);
      } else if (kind == "spin_raising") {
        set(n, n - 1, amplitude, 0.0);
      }
    }
    for (int64_t n = 0; n < d; ++n) {
      double level = static_cast<double>(n);
      if (kind == "spin_z")
        set(n, n, 1.0 - 2.0 * level, 0.0);
      else if (kind == "spin_i")
        set(n, n, 1.0, 0.0);
      else if (kind == "number")
        set(n, n, level, 0.0);
      else if (kind == "anharmonic")
        set(n, n, level * (level - 1.0), 0.0);
    }
    auto type = RankedTensorType::get({d, d, 2}, b.getF64Type());
    auto tensor = arith::ConstantOp::create(
        b, loc, DenseFPElementsAttr::get(type, ArrayRef<double>(data)));
    return qop::MatrixLeafOp::create(b, loc, handlerTy,
                                     b.getDenseI64ArrayAttr({mode}), tensor,
                                     b.getDenseI64ArrayAttr({d}));
  }

  Value product(ArrayRef<std::pair<int64_t, StringRef>> factors, double re,
                double im = 0.0) {
    SmallVector<Value> leaves;
    for (auto [qudit, kind] : factors)
      leaves.push_back(leaf(qudit, kind));
    auto scalar = qop::ConstScalarOp::create(
        b, loc, qop::ScalarType::get(b.getContext()), b.getF64FloatAttr(re),
        b.getF64FloatAttr(im));
    return qop::MakeProductOp::create(
        b, loc, qop::ProductType::get(b.getContext()), scalar, leaves);
  }

  Value drive(int64_t qudit, StringRef kind, StringRef callbackName,
              const WaveformCallbackData &callback) {
    return timed({{qudit, kind}}, callbackName, callback);
  }

  /// The product of `factors` with the time-dependent coefficient of
  /// `callback`.
  Value timed(ArrayRef<std::pair<int64_t, StringRef>> factors,
              StringRef callbackName, const WaveformCallbackData &callback) {
    auto scalar = qop::CallbackScalarOp::create(
        b, loc, qop::ScalarType::get(b.getContext()),
        FlatSymbolRefAttr::get(b.getContext(), callbackName));
    scalar->setAttr("cudm.callback_kind", b.getI32IntegerAttr(callback.kind));
    // A parameter that depends on the program's float arguments is its
    // constant part plus (slot, argument) pairs with their coefficients.
    SmallVector<double> constants, coefficients;
    SmallVector<int64_t> arguments;
    for (auto [slot, parameter] : llvm::enumerate(callback.parameters)) {
      constants.push_back(parameter.constant);
      for (auto [argument, coefficient] : parameter.coefficients()) {
        arguments.push_back(static_cast<int64_t>(slot));
        arguments.push_back(argument);
        coefficients.push_back(coefficient);
      }
    }
    scalar->setAttr("cudm.callback_params", b.getDenseF64ArrayAttr(constants));
    if (!arguments.empty()) {
      scalar->setAttr("cudm.callback_param_arguments",
                      b.getDenseI64ArrayAttr(arguments));
      scalar->setAttr("cudm.callback_param_coefficients",
                      b.getDenseF64ArrayAttr(coefficients));
    }
    SmallVector<Value> leaves;
    for (auto [qudit, kind] : factors)
      leaves.push_back(leaf(qudit, kind));
    return qop::MakeProductOp::create(
        b, loc, qop::ProductType::get(b.getContext()), scalar, leaves);
  }

private:
  OpBuilder &b;
  Location loc;
  const DenseMap<int64_t, int64_t> &levels;
  const DenseMap<int64_t, int64_t> &modeOf;
  std::optional<int64_t> outside;
};

/// One callback for every occurrence of a drive op, when each occurrence
/// differs from the first only by a fixed start period and phase step per
/// enclosing loop and every iteration plays. `callback` is the first
/// occurrence's; the other occurrences of the same op can differ from it
/// only in their timing.
static std::optional<WaveformCallbackData>
pulseTrain(ArrayRef<pulse::Occurrence> occurrences,
           const WaveformCallbackData &callback, double clockGHz,
           double qubitFrequencyHz) {
  Operation *op = occurrences.front().op;
  auto loops = pulse::enclosingLoops(op);
  if (loops.empty())
    return std::nullopt;
  const size_t depth = loops.size();
  int64_t total = 1;
  SmallVector<int64_t> counts, strides(depth, 1);
  for (auto loop : loops) {
    counts.push_back(pulse::constantTripCount(loop).value_or(0));
    total *= counts.back();
  }
  if (static_cast<int64_t>(occurrences.size()) != total || total < 2)
    return std::nullopt;
  for (size_t level = depth - 1; level > 0; --level)
    strides[level - 1] = strides[level] * counts[level];

  // The occurrences must come in iteration order, one per iteration.
  const double offset = phaseOffsetOf(op);
  SmallVector<DriveTiming> timing;
  timing.reserve(occurrences.size());
  std::vector<int64_t> expected(depth, 0);
  for (const pulse::Occurrence &occurrence : occurrences) {
    if (occurrence.iteration != expected)
      return std::nullopt;
    timing.push_back(
        driveTiming(occurrence, clockGHz, qubitFrequencyHz, offset));
    for (size_t level = depth; level-- > 0;) {
      if (++expected[level] < counts[level])
        break;
      expected[level] = 0;
    }
  }

  const DriveTiming &base = timing.front();
  SmallVector<double> periods(depth, 0.0);
  SmallVector<Affine> steps(depth, 0.0);
  for (size_t level = 0; level < depth; ++level) {
    if (counts[level] < 2)
      continue;
    const DriveTiming &next = timing[strides[level]];
    periods[level] = next.startNs - base.startNs;
    steps[level] = next.phase - base.phase;
    if (periods[level] <= 0.0)
      return std::nullopt;
  }
  for (auto [occurrence, t] : llvm::zip(occurrences, timing)) {
    double start = base.startNs;
    Affine phase = base.phase;
    for (auto [k, period, step] :
         llvm::zip(occurrence.iteration, periods, steps)) {
      start += k * period;
      phase += step * static_cast<double>(k);
    }
    if (std::abs(t.startNs - start) > 1e-9 || !t.phase.near(phase, 1e-9) ||
        t.durationNs != base.durationNs || t.detuning != base.detuning)
      return std::nullopt;
  }
  const auto &first = callback.parameters;
  SmallVector<Affine> header(first.begin(),
                             first.begin() + cudaq_pulse::kWaveformHeader);
  // Square pulses that abut with a continuous carrier phase are one longer
  // square pulse; fold such innermost levels into the duration.
  size_t levels = loops.size();
  auto tolerance = [](double b) { return 1e-9 * std::max(1.0, std::abs(b)); };
  while (callback.kind == 1 && levels > 0) {
    size_t level = levels - 1;
    const double duration = header[1].constant;
    const Affine advance = header[9] * duration;
    if (counts[level] >= 2 &&
        (std::abs(periods[level] - duration) > tolerance(duration) ||
         !steps[level].near(advance, tolerance(advance.constant))))
      break;
    header[1] = header[1] * static_cast<double>(counts[level]);
    --levels;
  }
  if (levels == 0) {
    WaveformCallbackData square{callback.kind, header};
    square.parameters.append(first.begin() + cudaq_pulse::kWaveformHeader,
                             first.end());
    return square;
  }
  WaveformCallbackData train{callback.kind | cudaq_pulse::kTrainFlag, header};
  train.parameters.push_back(static_cast<double>(levels));
  for (size_t level = 0; level < levels; ++level) {
    train.parameters.push_back(periods[level]);
    train.parameters.push_back(static_cast<double>(counts[level]));
    train.parameters.push_back(steps[level]);
  }
  train.parameters.append(first.begin() + cudaq_pulse::kWaveformHeader,
                          first.end());
  return train;
}

/// Each drive of `driveOrder` once for each qudit its line names, with the
/// place of the qudit among them: one qudit, or on a broadcast line each of
/// the line's. Not inlined, like memberQudit: the pass is then compiled as
/// it is without them.
LLVM_ATTRIBUTE_NOINLINE SmallVector<std::pair<Operation *, unsigned>>
drivenQudits(ArrayRef<Operation *> driveOrder, pulse::AllocationTracer &trace) {
  SmallVector<std::pair<Operation *, unsigned>> driven;
  driven.reserve(driveOrder.size());
  for (Operation *op : driveOrder) {
    Operation *allocation = trace(op->getOperand(0));
    unsigned reached = allocation ? allocation->getNumOperands() : 1;
    for (unsigned member = 0; member < reached; ++member)
      driven.push_back({op, member});
  }
  return driven;
}

/// The index of qudit `member` of the qudits the line `allocation` names
/// (a broadcast line names several), or -1.
LLVM_ATTRIBUTE_NOINLINE int64_t memberQudit(Operation *allocation,
                                            unsigned member) {
  auto alloc =
      allocation->getOperand(member).getDefiningOp<pulse::QuditAllocOp>();
  return alloc ? alloc.getQuditAttr().getInt() : -1;
}

struct PulseToQOpPass
    : public PassWrapper<PulseToQOpPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseToQOpPass)

  StringRef getArgument() const final { return "pulse-to-qop"; }
  StringRef getDescription() const final {
    return "Lower pulse dialect ops to qop Hamiltonian/Lindblad construction";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<qop::QOpDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getContext());

    func::FuncOp mainFunc;
    module.walk([&](func::FuncOp fn) {
      if (fn.getName() == "main")
        mainFunc = fn;
    });
    if (!mainFunc) {
      module.emitError("no @main function found");
      return signalPassFailure();
    }

    const double clockGHz = pulse::clockGHz(module);

    const pulse::LineGrid lineGrid(module);
    // The simulated quantum system. Its frequencies are the qudits' own,
    // which differ from the tones the program was compiled at when it is
    // simulated on another system; a qudit it gives none for is at its drive
    // line's tone.
    auto system = module->getAttrOfType<pulse::QuantumSystemAttr>(
        pulse::kQuantumSystemAttr);
    auto field = [&](auto get) {
      return system ? get(system) : decltype(get(system))();
    };
    DenseMap<int64_t, double> quditFreqHz;
    if (auto frequencies =
            field([](auto qs) { return qs.getQuditFrequencyHz(); }))
      for (auto [index, frequency] : llvm::enumerate(frequencies.asArrayRef()))
        quditFreqHz[static_cast<int64_t>(index)] = frequency;
    DenseMap<int64_t, int64_t> levels;
    if (auto counts = field([](auto qs) { return qs.getLevelCounts(); }))
      for (auto [index, d] : llvm::enumerate(counts.asArrayRef()))
        levels[static_cast<int64_t>(index)] = d;
    // The simulated qudits, one mode each in this order. Without it every
    // qudit index up to the largest is a mode.
    SmallVector<int64_t> modes;
    DenseMap<int64_t, int64_t> modeOf;
    if (auto attr = module->getAttrOfType<DenseI64ArrayAttr>("pulse.modes")) {
      for (int64_t qudit : attr.asArrayRef()) {
        if (qudit < 0 || !modeOf.try_emplace(qudit, modes.size()).second) {
          module.emitError("pulse.modes must list distinct qudit indices");
          return signalPassFailure();
        }
        modes.push_back(qudit);
      }
      if (modes.empty()) {
        module.emitError("pulse.modes must list at least one qudit");
        return signalPassFailure();
      }
    }

    // Readouts are numbered in walk order for `pulse.path`.
    SmallVector<Operation *> readouts;
    Operation *notIq = nullptr;
    module.walk([&](Operation *op) {
      // A drive line nothing uses is dead, and standard passes erase it: its
      // frequency is not the qudit's.
      if (isa<pulse::GetDriveLineOp>(op) && !op->use_empty()) {
        if (int64_t qi = pulse::quditOf(op); qi >= 0)
          if (auto freqAttr = op->getAttrOfType<FloatAttr>("frequency_hz"))
            quditFreqHz.try_emplace(qi, freqAttr.getValueAsDouble());
      }
      if (auto readout = dyn_cast<pulse::ReadoutOp>(op)) {
        if (readout.getMode() != pulse::ReadoutMode::IQ && !notIq)
          notIq = op;
        readouts.push_back(op);
      }
      if (op->getName().getStringRef() == "pulse.iq_acquire")
        readouts.push_back(op);
    });
    if (notIq) {
      notIq->emitError("readout mode ")
          << pulse::stringifyReadoutMode(
                 cast<pulse::ReadoutOp>(notIq).getMode())
          << " is not supported by the simulator yet; only iq is";
      return signalPassFailure();
    }
    Operation *unanswered = nullptr;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "classical.wait" &&
          !op->hasAttr("classical.answers")) {
        unanswered = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (unanswered) {
      unanswered->emitError(
          "a program with classical calls evolves in intervals between their "
          "answers; lower it with each answer bound, as pulse.simulate() "
          "does");
      return signalPassFailure();
    }
    // Without its arguments, a variable wait stands at its lowest length. A
    // wait whose length is an answer takes it from the bound answer.
    Operation *unbound = nullptr;
    module.walk([&](pulse::WaitOp wait) {
      auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>();
      IntegerAttr cycles;
      if (!wait->hasAttr("pulse.variable_wait") ||
          wait->hasAttr("classical.variable") ||
          (conversion &&
           matchPattern(conversion.getCycles(), m_Constant(&cycles))))
        return WalkResult::advance();
      unbound = wait;
      return WalkResult::interrupt();
    });
    if (unbound) {
      unbound->emitError("the length of this wait depends on the program's "
                         "float arguments; bind them (specialize the module) "
                         "before lowering it");
      return signalPassFailure();
    }
    pulse::Path path;
    auto pathAttr = module->getAttrOfType<ArrayAttr>("pulse.path");
    if (!readouts.empty() && !pathAttr) {
      readouts.front()->emitError(
          "a program that measures evolves in intervals between measurements; "
          "lower it along a measurement-outcome path (pulse.path), as "
          "pulse.simulate() does");
      return signalPassFailure();
    }
    if (pathAttr)
      for (Attribute entry : pathAttr) {
        auto values = dyn_cast<DenseI64ArrayAttr>(entry);
        if (!values || values.size() < 2 || values[0] < 0 ||
            values[0] >= static_cast<int64_t>(readouts.size())) {
          module.emitError("pulse.path entries are [readout, iteration..., "
                           "outcome] with a valid readout index");
          return signalPassFailure();
        }
        auto array = values.asArrayRef();
        path[{readouts[array.front()],
              std::vector<int64_t>(array.begin() + 1, array.end() - 1)}] =
            array.back();
      }

    std::optional<double> windowStart, windowEnd;
    if (auto attr = module->getAttrOfType<FloatAttr>("qop.t_start"))
      windowStart = attr.getValueAsDouble();
    if (auto attr = module->getAttrOfType<FloatAttr>("qop.t_end"))
      windowEnd = attr.getValueAsDouble();

    // Collect the drives played on this path, in execution order.
    int64_t endCycles = 0;
    std::map<pulse::MeasurementKey, double> measurementEndNs;
    SmallVector<Operation *> driveOrder;
    DenseMap<Operation *, SmallVector<pulse::Occurrence>> drives;
    pulse::ExecutionWalker walker(clockGHz, path);
    walker.onTimed = [&](const pulse::Occurrence &occurrence) {
      endCycles = std::max(endCycles,
                           occurrence.startCycles + occurrence.durationCycles);
    };
    walker.onDrive = [&](const pulse::Occurrence &occurrence) {
      double startNs = occurrence.startCycles / clockGHz;
      double endNs =
          (occurrence.startCycles + occurrence.durationCycles) / clockGHz;
      if ((windowEnd && startNs >= *windowEnd) ||
          (windowStart && endNs <= *windowStart))
        return;
      auto &list = drives[occurrence.op];
      if (list.empty())
        driveOrder.push_back(occurrence.op);
      list.push_back(occurrence);
    };
    walker.onReadout = [&](const pulse::Occurrence &occurrence,
                           const pulse::MeasurementKey &key) {
      measurementEndNs[key] =
          (occurrence.startCycles + occurrence.durationCycles) / clockGHz;
      return false;
    };
    if (!walker.run(mainFunc)) {
      walker.errorOp->emitError(walker.error);
      return signalPassFailure();
    }
    // A branch on a measurement without an outcome is skipped: that is only
    // right if the measurement completes no earlier than the window ends.
    for (const auto &key : walker.undecided) {
      double endNs = measurementEndNs[key];
      if (!windowEnd || endNs < *windowEnd) {
        key.readout->emitError("no outcome on pulse.path for this measurement, "
                               "which completes at ")
            << endNs << " ns, before the evolution window ends";
        return signalPassFailure();
      }
    }

    // Carry the scheduled pulse interval into the evolution lowering. Pulse
    // time is expressed in virtual clock units; cuDensityMat callbacks and
    // integrators use nanoseconds.
    if (endCycles > 0) {
      const double endNs = static_cast<double>(endCycles) / clockGHz;
      if (!module->hasAttr("qop.t_start"))
        module->setAttr("qop.t_start", b.getF64FloatAttr(0.0));
      if (!module->hasAttr("qop.t_end"))
        module->setAttr("qop.t_end", b.getF64FloatAttr(endNs));
      if (!module->hasAttr("qop.step_count"))
        module->setAttr("qop.step_count",
                        b.getI64IntegerAttr(std::max<int64_t>(100, endCycles)));
      if (!module->hasAttr("qop.integrator"))
        module->setAttr("qop.integrator", b.getStringAttr("rk4"));
    }

    // The source program is replaced by the operator construction.
    auto &block = mainFunc.getBody().front();
    Operation *returnOp = block.getTerminator();
    SmallVector<Operation *> sourceOps;
    for (Operation &op : block.without_terminator())
      sourceOps.push_back(&op);
    b.setInsertionPoint(returnOp);
    auto loc = returnOp->getLoc();
    QuditOperators ops(b, loc, levels, modeOf);

    auto opTy = qop::OpType::get(b.getContext());
    auto superOpTy = qop::SuperOpType::get(b.getContext());

    SmallVector<Value> allProducts;

    // Evolution uses one rotating frame per qubit. Bare qubit frequencies are
    // therefore removed from H; tone detuning is encoded in each drive
    // callback below. This avoids a numerically stiff multi-GHz lab-frame ODE.
    auto appendInteractions =
        [&](StringRef pairAttrName, StringRef strengthAttrName,
            function_ref<void(int64_t, int64_t, double)> append)
        -> LogicalResult {
      auto pairs = module->getAttrOfType<DenseI64ArrayAttr>(pairAttrName);
      auto strengths =
          module->getAttrOfType<DenseF64ArrayAttr>(strengthAttrName);
      if (!pairs && !strengths)
        return success();
      if (!pairs || !strengths || pairs.size() != 2 * strengths.size()) {
        module.emitError("interaction pair and strength metadata disagree for ")
            << pairAttrName;
        return failure();
      }
      for (auto [index, strength] : llvm::enumerate(strengths.asArrayRef())) {
        int64_t first = pairs[2 * index];
        int64_t second = pairs[2 * index + 1];
        if (first < 0 || second < 0 || first == second ||
            !quditFreqHz.count(first) || !quditFreqHz.count(second)) {
          module.emitError("interaction references an inactive qubit in ")
              << pairAttrName;
          return failure();
        }
        if (!ops.simulated(first) || !ops.simulated(second))
          continue;
        append(first, second, strength * 2.0 * M_PI * 1.0e-9);
      }
      return success();
    };
    // An exchange coupling g (a^dagger b + a b^dagger) between qudits whose
    // frames rotate at w_a and w_b turns at w_a - w_b in those frames:
    //   g (e^{i D t} a^dagger b + h.c.)
    //     = g/2 [cos(D t) (X X + Y Y) + sin(D t) (Y X - X Y)],
    // with D = w_a - w_b. The counter-rotating terms a b and a^dagger
    // b^dagger turn at w_a + w_b and are dropped.
    int couplingIdx = 0;
    auto exchange = [&](int64_t a, int64_t b, double g) {
      const double detuning =
          (quditFreqHz[a] - quditFreqHz[b]) * 2.0 * M_PI * 1.0e-9;
      if (detuning == 0.0) {
        allProducts.push_back(
            ops.product({{a, "spin_x"}, {b, "spin_x"}}, g / 2));
        allProducts.push_back(
            ops.product({{a, "spin_y"}, {b, "spin_y"}}, g / 2));
        return;
      }
      // A square callback from t = 0 that never ends is
      // (amplitude / 2) cos(D t), or sin(D t) in the Y quadrature.
      auto rotating = [&](double amplitude, int32_t quadrature) {
        return WaveformCallbackData{
            1,
            {0.0, std::numeric_limits<double>::max(), amplitude, 0.0, 0.0, 0.0,
             0.0, 0.0, static_cast<double>(quadrature), detuning}};
      };
      const std::string name = "coupling_" + std::to_string(couplingIdx++);
      const std::pair<int64_t, StringRef> terms[][2] = {
          {{a, "spin_x"}, {b, "spin_x"}},
          {{a, "spin_y"}, {b, "spin_y"}},
          {{a, "spin_y"}, {b, "spin_x"}},
          {{a, "spin_x"}, {b, "spin_y"}}};
      const double amplitudes[] = {g, g, g, -g};
      for (auto [index, factors] : llvm::enumerate(terms))
        allProducts.push_back(
            ops.timed(factors, name + "_" + std::to_string(index),
                      rotating(amplitudes[index], index / 2)));
    };
    if (failed(appendInteractions("pulse.coupling_pairs",
                                  "pulse.coupling_strength_hz", exchange)))
      return signalPassFailure();

    // The static terms of the quantum system, each between simulated
    // qudits. A zz term of rate zeta, E11 - E10 - E01 + E00 = zeta, is
    // zeta n_a n_b, which leaves the frequencies with the other qudit in |0>
    // unchanged.
    if (auto terms = field([](auto qs) { return qs.getStaticTerms(); })) {
      for (Attribute entry : terms) {
        auto term = cast<pulse::StaticTermAttr>(entry);
        ArrayRef<int64_t> qudits = term.getQudits();
        if (!llvm::all_of(
                qudits, [&](int64_t q) { return quditFreqHz.count(q) != 0; })) {
          module.emitError("static term ")
              << term << " names a qudit the program does not use";
          return signalPassFailure();
        }
        if (!llvm::all_of(qudits, [&](int64_t q) { return ops.simulated(q); }))
          continue;
        const double rate =
            term.getRateHz().convertToDouble() * 2.0 * M_PI * 1.0e-9;
        switch (term.getKind()) {
        case pulse::StaticTermKind::ZZ:
          allProducts.push_back(ops.product(
              {{qudits[0], "number"}, {qudits[1], "number"}}, rate));
          break;
        }
      }
    }

    // A multilevel transmon's anharmonicity, (alpha / 2) n (n - 1).
    if (auto anharmonicity =
            field([](auto qs) { return qs.getAnharmonicityHz(); }))
      for (auto [index, alpha] : llvm::enumerate(anharmonicity.asArrayRef()))
        if (alpha != 0.0 && ops.levelsOf(index) > 2 && ops.simulated(index))
          allProducts.push_back(
              ops.product({{static_cast<int64_t>(index), "anharmonic"}},
                          0.5 * alpha * 2.0 * M_PI * 1.0e-9));

    // 2. Time-dependent drive terms
    int driveIdx = 0;
    pulse::AllocationTracer trace;
    // A drive on a broadcast line drives each qudit its line names, with
    // that qudit's own frequency and amplitude scale.
    for (const std::pair<Operation *, unsigned> &target :
         drivenQudits(driveOrder, trace)) {
      Operation *op = target.first;
      auto &occurrences = drives[op];
      auto *allocation = trace(op->getOperand(0));
      int64_t qubitIdx = allocation ? pulse::quditOf(allocation) : -1;
      if (target.second)
        qubitIdx = memberQudit(allocation, target.second);
      if (qubitIdx < 0) {
        op->emitError("cannot determine the qudit of this drive; the line "
                      "must come from a pulse.get_drive_line of a "
                      "pulse.qudit_alloc");
        return signalPassFailure();
      }
      if (!ops.simulated(qubitIdx)) {
        op->emitError("drive on qudit ")
            << qubitIdx << ", which is not one of pulse.modes";
        return signalPassFailure();
      }

      auto qubitFrequency = quditFreqHz.find(qubitIdx);
      if (qubitFrequency == quditFreqHz.end()) {
        op->emitError("cannot determine the physical qubit frequency for "
                      "rotating-frame lowering");
        return signalPassFailure();
      }
      double driveScale = 1.0;
      if (auto scales =
              field([](auto qs) { return qs.getDriveScaleRadPerNs(); })) {
        if (qubitIdx < 0 || static_cast<size_t>(qubitIdx) >= scales.size() ||
            scales.asArrayRef()[qubitIdx] <= 0.0) {
          op->emitError("drive amplitude scale is missing or non-positive");
          return signalPassFailure();
        }
        driveScale = scales.asArrayRef()[qubitIdx];
      }

      const int64_t grid =
          lineGrid.enabled() ? lineGrid.of(op->getOperand(0), trace) : 1;
      static constexpr llvm::StringLiteral quadratures[] = {"spin_x", "spin_y"};
      for (auto [quadrature, kind] : llvm::enumerate(quadratures)) {
        // Clang < 16 cannot capture a structured binding by name.
        auto lower = [&clockGHz, &qubitFrequency, &driveScale,
                      &quadrature = quadrature,
                      &grid](const pulse::Occurrence &occurrence) {
          return getWaveformCallbacks(occurrence, clockGHz,
                                      qubitFrequency->second, driveScale,
                                      static_cast<int32_t>(quadrature), grid);
        };
        auto first = lower(occurrences.front());
        if (!first) {
          op->emitError(
              "waveform cannot be lowered to a cuDensityMat callback; "
              "schedule the program, use built-in or custom_samples "
              "waveforms and their arithmetic, and keep float arguments "
              "affine (products of two arguments, and arguments in sampled "
              "waveform products, need the arguments bound)");
          return signalPassFailure();
        }
        // Each term of the waveform becomes one pulse train if every term
        // can; otherwise every occurrence is lowered on its own.
        SmallVector<WaveformCallbackData> callbacks;
        for (const WaveformCallbackData &term : *first) {
          auto train =
              pulseTrain(occurrences, term, clockGHz, qubitFrequency->second);
          if (!train)
            break;
          callbacks.push_back(std::move(*train));
        }
        if (callbacks.size() != first->size()) {
          callbacks = std::move(*first);
          for (const pulse::Occurrence &occurrence :
               llvm::drop_begin(occurrences)) {
            auto more = lower(occurrence);
            if (!more) {
              op->emitError("waveform cannot be lowered to a cuDensityMat "
                            "callback");
              return signalPassFailure();
            }
            llvm::append_range(callbacks, *more);
          }
        }
        for (auto [index, callback] : llvm::enumerate(callbacks)) {
          std::string name = "drive_envelope_" + std::to_string(driveIdx) +
                             (callbacks.size() > 1 ? "_" + std::to_string(index)
                                                   : std::string()) +
                             (quadrature ? "_y" : "_x");
          allProducts.push_back(ops.drive(qubitIdx, kind, name, callback));
        }
      }
      driveIdx++;
    }

    // 3. Assemble Hamiltonian: qop.make_sum
    Value hamiltonian;
    if (!allProducts.empty()) {
      hamiltonian = qop::MakeSumOp::create(b, loc, opTy, allProducts);
    } else {
      // Trivial Hamiltonian: identity
      int64_t first = modes.empty() ? 0 : modes.front();
      hamiltonian = qop::MakeSumOp::create(
          b, loc, opTy, ValueRange{ops.product({{first, "spin_i"}}, 0.0)});
    }

    // 4. Dissipators from module attributes (T1, T2)
    SmallVector<Value> collapseOps;

    DenseF64ArrayAttr t1Ns = field([](auto qs) { return qs.getT1Ns(); });
    DenseF64ArrayAttr t2Ns = field([](auto qs) { return qs.getT2Ns(); });
    // A qudit with thermal excited-state population p relaxes down at
    // (1 - p) / T1 and up at p / T1 (detailed balance): populations relax
    // to p at the measured rate 1 / T1.
    auto thermal =
        module->getAttrOfType<DenseF64ArrayAttr>("pulse.thermal_populations");
    if (t1Ns) {
      for (int64_t qi = 0; qi < (int64_t)t1Ns.size(); qi++) {
        double t1 = t1Ns[qi];
        if (t1 > 0 && ops.simulated(qi)) {
          double gamma = 1.0 / t1;
          double p =
              thermal && qi < (int64_t)thermal.size() ? thermal[qi] : 0.0;
          auto lowering = ops.product({{qi, "spin_lowering"}},
                                      std::sqrt((1.0 - p) * gamma));
          collapseOps.push_back(
              qop::MakeSumOp::create(b, loc, opTy, ValueRange{lowering}));
          if (p > 0) {
            auto raising =
                ops.product({{qi, "spin_raising"}}, std::sqrt(p * gamma));
            collapseOps.push_back(
                qop::MakeSumOp::create(b, loc, opTy, ValueRange{raising}));
          }
        }
      }
    }

    if (t2Ns) {
      for (int64_t qi = 0; qi < (int64_t)t2Ns.size(); qi++) {
        double t2 = t2Ns[qi];
        double t1 = t1Ns && qi < (int64_t)t1Ns.size() ? t1Ns[qi] : 0.0;
        double gammaPhi = 0.0;
        if (t2 > 0) {
          gammaPhi = 1.0 / t2;
          if (t1 > 0)
            gammaPhi -= 1.0 / (2.0 * t1);
          if (gammaPhi < 0)
            gammaPhi = 0;
        }
        if (gammaPhi > 0 && ops.simulated(qi)) {
          // Off-diagonal elements between levels j and k decay at
          // gammaPhi (j - k)^2; for a qubit sqrt(gammaPhi / 2) Z is the same.
          auto dephase =
              ops.levelsOf(qi) == 2
                  ? ops.product({{qi, "spin_z"}}, std::sqrt(gammaPhi / 2.0))
                  : ops.product({{qi, "number"}}, std::sqrt(2.0 * gammaPhi));
          collapseOps.push_back(
              qop::MakeSumOp::create(b, loc, opTy, ValueRange{dephase}));
        }
      }
    }

    // 5. Construct Lindblad super-operator and evolve the state under it
    auto lindblad =
        qop::LindbladOp::create(b, loc, superOpTy, hamiltonian, collapseOps);
    qop::EvolveOp::create(b, loc, lindblad);
    if (auto qudit = ops.unsimulated()) {
      module.emitError("an operator acts on qudit ")
          << *qudit << ", which is not one of pulse.modes";
      return signalPassFailure();
    }

    if (!modes.empty()) {
      SmallVector<int64_t> extents;
      for (int64_t qudit : modes)
        extents.push_back(ops.levelsOf(qudit));
      module->setAttr("qop.mode_extents", b.getDenseI64ArrayAttr(extents));
    } else if (!levels.empty()) {
      int64_t modes = 0;
      if (auto attr = module->getAttrOfType<IntegerAttr>("qop.qudit_count"))
        modes = attr.getInt();
      else if (auto attr =
                   module->getAttrOfType<IntegerAttr>("pulse.qudit_count"))
        modes = attr.getInt();
      for (auto [qudit, d] : levels)
        modes = std::max(modes, qudit + 1);
      SmallVector<int64_t> extents;
      for (int64_t qudit = 0; qudit < modes; ++qudit)
        extents.push_back(ops.levelsOf(qudit));
      module->setAttr("qop.mode_extents", b.getDenseI64ArrayAttr(extents));
    }

    // The QOp graph above is self-contained. Consume the source Pulse
    // program, regions included, so the next conversion stage receives only
    // QOp plus standard dialect ops.
    for (Operation *op : llvm::reverse(sourceOps)) {
      op->dropAllUses();
      op->dropAllReferences();
    }
    for (Operation *op : llvm::reverse(sourceOps))
      op->erase();
    // Classical calls are answered before lowering (bound along the path);
    // their declarations and link parameters have no QOp counterpart.
    for (Operation &op : llvm::make_early_inc_range(*module.getBody()))
      if (op.getDialect() && op.getDialect()->getNamespace() == "classical")
        op.erase();
    module->removeAttr("classical.links");
  }
};

} // namespace

namespace pulse {

std::unique_ptr<mlir::Pass> createPulseToQOpPass() {
  return std::make_unique<PulseToQOpPass>();
}

} // namespace pulse
