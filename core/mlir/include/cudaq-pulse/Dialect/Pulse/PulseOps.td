// -*- tablegen -*-
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_DIALECT_PULSE_OPS
#define CUDAQ_PULSE_DIALECT_PULSE_OPS

include "cudaq-pulse/Dialect/Pulse/PulseDialect.td"
include "cudaq-pulse/Dialect/Pulse/PulseAttrs.td"
include "cudaq-pulse/Dialect/Pulse/PulseTypes.td"
include "cudaq-pulse/Dialect/Pulse/PulseTiming.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"

class PulseOp<string mnemonic, list<Trait> traits = []> :
    Op<PulseDialect, mnemonic, traits>;

// Effects on lines and tones are on the values an op takes and returns
// (PulseResources.h).
def Pulse_LineResource : Resource<"::pulse::LineResource">;
def Pulse_ToneResource : Resource<"::pulse::ToneResource">;

//===----------------------------------------------------------------------===//
// Qudit allocation
//===----------------------------------------------------------------------===//

def QuditAllocOp : PulseOp<"qudit_alloc", [Pure]> {
  let summary = "A qudit of the quantum system, by its index";
  let description = [{
    Qudit `qudit` of the quantum system, by its index there. Lines and
    tones are obtained from it with `pulse.get_drive_line` and
    `pulse.get_readout_line`, which find the index here.

    ```mlir
    %q = pulse.qudit_alloc {qudit = 3 : i64} : !pulse.qudit
    ```
  }];
  let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$qudit);
  let results = (outs QuditType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Line operations
//===----------------------------------------------------------------------===//

def GetDriveLineOp : PulseOp<"get_drive_line", [Pure]> {
  let summary = "Get the drive line and tone for a qudit";
  let description = [{
    Returns the drive line and native tone associated with a qudit.
    The returned drive line starts at time t=0.

    A broadcast drive line reaches several qudits with one tone: the op
    names all of them, lowest index first, and a drive on the line drives
    every one. Its `frequency_hz` is the frequency of the line's tone.

    ```mlir
    %line, %tone = pulse.get_drive_line %q0, %q1 {frequency_hz = 5.0e+09 : f64}
        : (!pulse.qudit, !pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    ```
  }];
  let arguments = (ins QuditType:$qudit, Variadic<QuditType>:$others);
  let results = (outs DriveLineType:$line, ToneType:$tone);
  let assemblyFormat = "$qudit (`,` $others^)? attr-dict `:` functional-type(operands, results)";
  // The builder of a line of one qudit, as before the line could name more.
  let builders = [
    OpBuilder<(ins "::mlir::Type":$line, "::mlir::Type":$tone,
                   "::mlir::Value":$qudit), [{
      $_state.addOperands(qudit);
      $_state.addTypes(line);
      $_state.addTypes(tone);
    }]>
  ];
  let hasVerifier = 1;
}

def GetReadoutLineOp : PulseOp<"get_readout_line", [Pure]> {
  let summary = "Get the readout line and tone for a qudit";
  let description = [{
    The readout line and readout tone of a qudit. As with
    `pulse.get_drive_line`, the line starts at time 0 and is a linear
    value: each op that uses it returns the updated line, which the next
    op on the line consumes.
  }];
  let arguments = (ins QuditType:$qudit);
  let results = (outs ReadoutLineType:$line, ToneType:$tone);
  let assemblyFormat = "$qudit attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Tone operations
//===----------------------------------------------------------------------===//

def ToneOp : PulseOp<"tone"> {
  let summary = "Create a pulse tone with explicit frequency and phase";
  let description = [{
    A tone with an explicit carrier frequency (Hz) and frame phase
    (radians), rather than the native tone of a line. Tones are linear
    values: `pulse.drive`, `pulse.readout` and the phase and frequency
    ops return the updated tone.

    Each `pulse.tone` is a new frame, even with the same frequency and
    phase as another: it allocates its result, so it is not merged with
    another one or moved out of a loop, and is removed only when unused.
  }];
  let arguments = (ins
    AnyFloat:$frequency_hz,
    AnyFloat:$phase_rad
  );
  let results = (outs Res<ToneType, "", [MemAlloc<Pulse_ToneResource>]>:$tone);
  let assemblyFormat = "$frequency_hz `,` $phase_rad attr-dict `:` qualified(type($frequency_hz)) `,` qualified(type($phase_rad)) `->` qualified(type($tone))";
}

//===----------------------------------------------------------------------===//
// Waveform construction
//===----------------------------------------------------------------------===//

def SquarePulseOp : PulseOp<"square", [Pure]> {
  let summary = "Create a constant-amplitude (square) pulse";
  let description = [{
    A constant envelope `amp_real + i amp_imag` over `duration` clock
    cycles. A constant `duration` must be positive; one that is a function
    argument is checked once it is specialized.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amp_real,
    AnyFloat:$amp_imag
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amp_real `,` $amp_imag attr-dict `:` type($duration) `,` type($amp_real) `,` type($amp_imag) `->` type($pulse)";
}

def GaussianPulseOp : PulseOp<"gaussian", [Pure]> {
  let summary = "Create a Gaussian pulse";
  let description = [{
    A Gaussian envelope over `duration` clock cycles, centered in the
    pulse, with peak `amplitude` and standard deviation `sigma` in clock
    cycles. Constant `duration` and `sigma` must be positive.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `->` type($pulse)";
}

def GaussianSquarePulseOp : PulseOp<"gaussian_square", [Pure]> {
  let summary = "Create a flat-topped Gaussian pulse";
  let description = [{
    A Gaussian rising edge, a flat top at the given amplitude, and a
    Gaussian falling edge. `risefall` is the duration of each edge in clock cycles.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma,
    I64:$risefall
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma `,` $risefall attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `,` type($risefall) `->` type($pulse)";
}

def DRAGPulseOp : PulseOp<"drag", [Pure]> {
  let summary = "Create a DRAG pulse for reducing leakage";
  let description = [{
    A DRAG envelope, `G + i beta dG/dt` for the Gaussian `G` of
    `pulse.gaussian`: the imaginary part is
    `-beta (t - duration / 2) / sigma^2` times `G`. `sigma` and `beta` are
    in clock cycles. Leakage of a transmon is smallest near
    `beta = -1 / alpha`, for the anharmonicity `alpha` in radians per
    clock cycle.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma,
    AnyFloat:$beta
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma `,` $beta attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `,` type($beta) `->` type($pulse)";
}

def CosinePulseOp : PulseOp<"cosine", [Pure]> {
  let summary = "Create a raised-cosine envelope pulse";
  let description = [{
    A raised-cosine envelope `amplitude (1 - cos(2 pi t / duration)) / 2`
    over `duration` clock cycles, zero at both ends.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude attr-dict `:` type($duration) `,` type($amplitude) `->` type($pulse)";
}

def TanhRampOp : PulseOp<"tanh_ramp", [Pure]> {
  let summary = "Create a hyperbolic-tangent ramp pulse";
  let description = [{
    Smooth ramp from 0 to `amplitude` using tanh shaping. `sigma` controls
    the steepness of the ramp.
  }];
  let arguments = (ins
    I64:$duration,
    AnyFloat:$amplitude,
    AnyFloat:$sigma
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$duration `,` $amplitude `,` $sigma attr-dict `:` type($duration) `,` type($amplitude) `,` type($sigma) `->` type($pulse)";
}

def CustomOp : PulseOp<"custom", [Pure]> {
  let summary = "Create a custom waveform via callback function";
  let description = [{
    A waveform whose envelope is the function `callee`, for compiler
    experiments. Execution needs samples, so the GPU lowering rejects it
    until a pass specializes it to `pulse.custom_samples`.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    I64:$duration
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$callee `,` $duration attr-dict `:` type($duration) `->` type($pulse)";
}

def CustomSamplesOp : PulseOp<"custom_samples", [Pure]> {
  let summary = "Create a waveform from raw sample data";
  let description = [{
    A waveform from explicit real samples, each held constant until the
    next. Without `rate` there is one sample per clock cycle, so the
    duration is the number of samples; with `rate = array<i64: n, d>` there
    are `n` samples every `d` cycles (the rate of the line's controller the
    samples were made for), and the samples must last a whole number of
    cycles. The array must not be empty.
  }];
  let arguments = (ins
    F64ArrayAttr:$samples,
    OptionalAttr<DenseI64ArrayAttr>:$rate
  );
  let results = (outs WaveformType:$pulse);
  let assemblyFormat = "$samples attr-dict `:` type($pulse)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// The duration in clock cycles: the samples at their rate.
    int64_t getDurationCycles();
  }];
}

//===----------------------------------------------------------------------===//
// Pulse application operations
//===----------------------------------------------------------------------===//

def DriveOp : PulseOp<"drive", []> {
  let summary = "Apply a waveform to a drive line at a tone";
  let description = [{
    Applies a waveform to a drive line at the specified tone's frequency and
    phase. Advances line time by pulse duration. Returns the updated drive
    line and tone. The tone's frame is as the frame changes before the drive
    on that tone left it: a change takes effect at the next pulse.
  }];
  let arguments = !con((ins
    DriveLineType:$line,
    WaveformType:$pulse,
    ToneType:$tone
  ),
                       PulseTimingArgs);
  let results = (outs DriveLineType:$updated_line, ToneType:$updated_tone);
  let assemblyFormat = [{
    $line `,` $pulse `,` $tone `` custom<Timing>("getOperationName()", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` qualified(type($line))
    `,` qualified(type($pulse)) `,` qualified(type($tone))
    `->` qualified(type($updated_line)) `,` qualified(type($updated_tone))
  }];
}

def ReadoutOp : PulseOp<"readout", []> {
  let summary = "Apply a readout pulse and produce a measurement result";
  let description = [{
    Plays `pulse` on a readout line at `tone` and measures the qudit,
    returning the updated line and tone and a `!pulse.measurement`. The
    measurement completes at the end of the pulse: `pulse.is_set` and
    `pulse.outcome` read it, and `pulse-verify` checks that ops
    conditioned on it start after it completes. `mode` names what the
    acquisition gives: `iq` (demodulated and discriminated into a level, what
    the frontend emits), `discriminated`, `trace` or `counts`. Only `iq` is
    simulated and exported so far; the others are refused there.

    Readouts are numbered in walk order, the kernel's order: a module built
    from a kernel in which a pass can copy a readout (one in a loop, or one
    after an if or a select) gives each readout its number,
    `pulse.readout_number`, before any pass runs (in another module,
    scheduling numbers a function's readouts before it first copies one);
    without it, a readout's number is its place among the readouts. A copy
    keeps the number: an op that reads a measurement run in both arms of an
    if on it, or a readout of a loop that scheduling peels or unrolls, which
    also gets `pulse.readout_iteration`: three integers (depth, scale,
    offset) per loop of the kernel around it, outermost first; its iteration
    of that loop is scale * k + offset, k the iteration of its enclosing
    `scf.for` at depth `depth`, or `offset` where `depth` is -1.

    ```mlir
    %r1, %t1, %m = pulse.readout %r0, %w, %t0, iq : ...
    ```
  }];
  let arguments = !con((ins
    ReadoutLineType:$line,
    WaveformType:$pulse,
    ToneType:$tone,
    EnumProp<ReadoutMode>:$mode
  ),
                       PulseTimingArgs);
  let results = (outs
    ReadoutLineType:$updated_line,
    ToneType:$updated_tone,
    MeasurementType:$result
  );
  let assemblyFormat = [{
    $line `,` $pulse `,` $tone `,` $mode `` custom<Timing>("getOperationName()", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` qualified(type($line))
    `,` qualified(type($pulse)) `,` qualified(type($tone))
    `->` qualified(type($updated_line)) `,` qualified(type($updated_tone))
    `,` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Measurement results
//===----------------------------------------------------------------------===//

def OutcomeOp : PulseOp<"outcome", [Pure]> {
  let summary = "Level observed by a readout";
  let description = [{
    The level `k` in `[0, d)` that the `pulse.readout` producing
    `measurement` observed on its `d`-level qudit. Compare it with
    `arith.cmpi` to branch on a particular level of a qudit:

    ```mlir
    %k = pulse.outcome %m : !pulse.measurement -> i64
    %c2 = arith.constant 2 : i64
    %is2 = arith.cmpi eq, %k, %c2 : i64
    scf.if %is2 { ... }
    ```
  }];
  let arguments = (ins MeasurementType:$measurement);
  let results = (outs I64:$level);
  let assemblyFormat = [{
    $measurement attr-dict `:` qualified(type($measurement)) `->` type($level)
  }];
}

def IsSetOp : PulseOp<"is_set", [Pure]> {
  let summary = "Whether a readout observed a nonzero level";
  let description = [{
    `true` when the `pulse.readout` producing `measurement` observed any
    level other than 0: the truth value a kernel tests with `if result:`.
    Measurement-conditioned feedback is an `scf.if` on this condition, with
    the lines and tones either branch advances passed through `scf.yield`:

    ```mlir
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
      %d, %t = pulse.drive %d0, %x, %t0 : ...
      scf.yield %d, %t : !pulse.drive_line, !pulse.tone
    } else {
      scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
    }
    ```
  }];
  let arguments = (ins MeasurementType:$measurement);
  let results = (outs I1:$condition);
  let assemblyFormat = [{
    $measurement attr-dict `:` qualified(type($measurement)) `->` type($condition)
  }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Timing operations
//===----------------------------------------------------------------------===//

def WaitOp : PulseOp<"wait", [AllTypesMatch<["line", "updated_line"]>]> {
  let summary = "Idle a line for a specified duration";
  let description = [{
    Idles a drive or readout line for `duration` and returns the updated
    line, of the same kind. The scheduler reads the length from a
    `pulse.duration_from_int` of a constant. A wait does not change any
    tone's frame phase (unlike an OpenPulse delay, which advances the
    frame); the carrier runs on, so a detuned tone's phase in its qudit's
    frame advances with the elapsed time as it does during pulses.

    A wait marked `pulse.variable_wait` takes its length from a value computed
    from the function's float arguments, so it is known only when they are
    bound. It carries `pulse.min_cycles` and, optionally, `pulse.max_cycles`: the
    bounds of its length in clock cycles. Without bound arguments, the module
    is scheduled with the wait at `pulse.min_cycles`; binding them folds the
    length to a constant within the bounds, and scheduling again gives the
    exact schedule.
  }];
  let arguments = !con((ins
    AnyLineType:$line,
    DurationType:$duration
  ),
                       PulseTimingArgs);
  let results = (outs AnyLineType:$updated_line);
  let assemblyFormat = "$line `,` $duration `` custom<Timing>(\"getOperationName()\", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SyncOp : PulseOp<"sync", []> {
  let summary = "Synchronize multiple lines to the same time";
  let description = [{
    Aligns time state of multiple lines. All returned lines will have
    time = max(T1, ..., Tk). Tones pass through unchanged: a frame change
    made before a sync applies to the next pulse on its tone after it.
  }];
  let arguments = (ins Variadic<AnyLineType>:$lines);
  let results = (outs Variadic<AnyLineType>:$synced_lines);
  let assemblyFormat = "$lines attr-dict `:` qualified(type($lines)) `->` qualified(type($synced_lines))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def GetCurrentTimeOp : PulseOp<"get_current_time", [Pure]> {
  let summary = "Get the current time of a line as a timepoint";
  let description = [{
    The time a line has reached, as a `!pulse.timepoint`. The line is not
    consumed: a line may be read any number of times, and then played on,
    and a read inside a loop of a line the loop does not carry reads the
    same time every iteration.
  }];
  let arguments = (ins AnyLineType:$line);
  let results = (outs TimepointType:$time);
  let assemblyFormat = "$line attr-dict `:` qualified(type($line)) `->` qualified(type($time))";
}

def TimepointToIntOp : PulseOp<"timepoint_to_int", [Pure]> {
  let summary = "Convert a timepoint to integer cycles";
  let description = [{
    A timepoint as a number of clock cycles from the start of the
    program.
  }];
  let arguments = (ins TimepointType:$time);
  let results = (outs I64:$time_in_cycles);
  let assemblyFormat = "$time attr-dict `:` qualified(type($time)) `->` qualified(type($time_in_cycles))";
}

def TimepointSubtractOp : PulseOp<"timepoint_subtract", [Pure]> {
  let summary = "Compute the duration between two timepoints";
  let description = [{
    The duration `lhs - rhs` between two timepoints.
  }];
  let arguments = (ins TimepointType:$lhs, TimepointType:$rhs);
  let results = (outs DurationType:$duration);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def DurationFromIntOp : PulseOp<"duration_from_int", [Pure]> {
  let summary = "Convert an integer to a duration";
  let description = [{
    A `!pulse.duration` of `cycles` clock cycles, as `pulse.wait` takes.
  }];
  let arguments = (ins I64:$cycles);
  let results = (outs DurationType:$duration);
  let assemblyFormat = "$cycles attr-dict `:` functional-type(operands, results)";
}

def DurationToIntOp : PulseOp<"duration_to_int", [Pure]> {
  let summary = "Convert a duration to an integer";
  let description = [{
    The number of clock cycles of a duration.
  }];
  let arguments = (ins DurationType:$duration);
  let results = (outs I64:$cycles);
  let assemblyFormat = "$duration attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Tone modulation operations (zero-duration, instantaneous)
//===----------------------------------------------------------------------===//

def ShiftFrequencyOp : PulseOp<"shift_frequency", []> {
  let summary = "Apply a relative frequency shift to a tone";
  let description = [{
    Adds `frequency_hz` to the tone's carrier frequency. The frame phase
    stays continuous, and gains phase at the new frequency from where the
    program makes the change, after the waits before it on the tone's line:
    shifted by `+d` before a wait of `T` and by `-d` after it, the frame is
    `2 pi d T` further on. It takes no time.
  }];
  let arguments = !con((ins ToneType:$tone, F64:$frequency_hz),
                       PulseTimingArgs);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $frequency_hz `` custom<Timing>(\"getOperationName()\", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` qualified(type($tone)) `,` type($frequency_hz) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def ShiftPhaseOp : PulseOp<"shift_phase", []> {
  let summary = "Apply a relative phase shift to a tone (virtual Z gate)";
  let description = [{
    Advances the tone's frame phase by `phase_rad`: a virtual Z rotation,
    in the convention of McKay et al., PRA 96, 022330 (2017). A pulse
    played after it turns about an axis rotated by `-phase_rad`. It takes
    no time and takes effect at the next pulse on the tone, and
    `pulse-fold-phases` folds it into the next drive.
  }];
  let arguments = !con((ins ToneType:$tone, F64:$phase_rad),
                       PulseTimingArgs);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $phase_rad `` custom<Timing>(\"getOperationName()\", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` qualified(type($tone)) `,` type($phase_rad) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SetPhaseOp : PulseOp<"set_phase", []> {
  let summary = "Set absolute phase of a tone";
  let description = [{
    Sets the tone's frame phase to `phase_rad`, with the sign of
    `pulse.shift_phase`. It takes no time.
  }];
  let arguments = !con((ins ToneType:$tone, F64:$phase_rad),
                       PulseTimingArgs);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $phase_rad `` custom<Timing>(\"getOperationName()\", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` qualified(type($tone)) `,` type($phase_rad) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SetFrequencyOp : PulseOp<"set_frequency", []> {
  let summary = "Set absolute frequency of a tone";
  let description = [{
    Sets the tone's carrier frequency to `frequency_hz`. The frame phase
    stays continuous, and gains phase at the new frequency from where the
    program makes the change, after the waits before it on the tone's line.
    It takes no time.
  }];
  let arguments = !con((ins ToneType:$tone, F64:$frequency_hz),
                       PulseTimingArgs);
  let results = (outs ToneType:$updated_tone);
  let assemblyFormat = "$tone `,` $frequency_hz `` custom<Timing>(\"getOperationName()\", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` qualified(type($tone)) `,` type($frequency_hz) `->` qualified(type($updated_tone))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Waveform algebra
//===----------------------------------------------------------------------===//

def PulseAddOp : PulseOp<"add", [Pure, Commutative]> {
  let summary = "Add two waveforms element-wise";
  let description = [{
    The element-wise sum of two waveforms. Both must have the same
    duration when it is known; a mismatch is an error rather than a
    truncation.
  }];
  let arguments = (ins WaveformType:$lhs, WaveformType:$rhs);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
}

def PulseSubOp : PulseOp<"sub", [Pure]> {
  let summary = "Subtract two waveforms element-wise";
  let description = [{
    The element-wise difference `lhs - rhs` of two waveforms of the same
    duration.
  }];
  let arguments = (ins WaveformType:$lhs, WaveformType:$rhs);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
}

def PulseMulOp : PulseOp<"mul", [Pure, Commutative]> {
  let summary = "Multiply two waveforms element-wise";
  let description = [{
    The element-wise product of two waveforms of the same duration. The
    drive is not linear in it, so execution samples it once per clock
    cycle into piecewise-constant samples.
  }];
  let arguments = (ins WaveformType:$lhs, WaveformType:$rhs);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
}

def PulseScaleOp : PulseOp<"scale", [Pure]> {
  let summary = "Scale a waveform by a scalar factor";
  let description = [{
    The waveform `pulse` times a real or complex `scale`.
  }];
  let arguments = (ins
    WaveformType:$pulse,
    AnyTypeOf<[AnyFloat, AnyComplex]>:$scale
  );
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$pulse `,` $scale attr-dict `:` qualified(type($pulse)) `,` type($scale) `->` qualified(type($result))";
  let hasFolder = 1;
}

def PulseNegOp : PulseOp<"neg", [Pure]> {
  let summary = "Negate a waveform";
  let description = [{
    The waveform `pulse` negated.
  }];
  let arguments = (ins WaveformType:$pulse);
  let results = (outs WaveformType:$result);
  let assemblyFormat = "$pulse attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Schedule region
//===----------------------------------------------------------------------===//

def YieldOp : PulseOp<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "Yield from a pulse region";
  let description = [{
    Ends the body of a `pulse.schedule_block`, passing the lines, tones
    and measurements that leave it to the block's results.
  }];
  let arguments = (ins Variadic<AnyTimedType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def ScheduleBlockOp : PulseOp<"schedule_block", [
    RecursiveMemoryEffects, SingleBlock,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getEntrySuccessorOperands"]>]> {
  let summary = "A subroutine scheduled on its own, then placed as one unit";
  let description = [{
    The body is scheduled in isolation with its own `policy` (`asap` or
    `alap`), every input available at local time 0. The enclosing schedule
    then places the whole block at one start time, so the relative timing of
    the ops inside is the same wherever the block appears.

    Lines, tones and measurements cross the boundary explicitly, as with
    `scf.for` iteration arguments: each operand is passed to the matching
    block argument, and the `pulse.yield` operands become the results. A
    line or tone that enters the block continues as one of its results.
    Values of other types (waveforms, constants) may be used from the
    enclosing region directly.

    ```mlir
    %d1, %t1 = pulse.schedule_block asap (%d0, %t0)
        : (!pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone) {
    ^bb0(%d: !pulse.drive_line, %t: !pulse.tone):
      %d2, %t2 = pulse.drive %d, %x, %t : ...
      pulse.yield %d2, %t2 : !pulse.drive_line, !pulse.tone
    }
    ```

    Scheduling inlines the block, so later passes see flat IR. Before that,
    the scheduler records the block's shape on it: `length_cycles`, for each
    operand the local time it is first needed (`in_offset_cycles`, -1 if no
    timed op depends on it), and for each result the local time it is
    available (`out_offset_cycles`), or -1 with `out_source` naming the operand
    it passes through unchanged. With `max_duration_cycles`, scheduling fails
    if the block can take longer than that many cycles.
  }];
  let arguments = !con((ins EnumProp<SchedulePolicy>:$policy,
                       Variadic<AnyTimedType>:$inputs,
                       OptionalAttr<StrAttr>:$callee),
                       PulseTimingArgs);
  let results = (outs Variadic<AnyTimedType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $policy ` ` `(` $inputs `)` `` custom<Timing>("getOperationName()", $start_cycles, $duration_cycles, $period_cycles, $callee) attr-dict `:` functional-type($inputs, $outputs)
    $body
  }];
  let hasRegionVerifier = 1;
}

def CallOp : PulseOp<"call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Run a function scheduled on its own, placed as one unit";
  let description = [{
    Runs `callee`, a function of the module with a schedule of its own, at
    one start time: every timed op of the callee keeps its start relative to
    the call's start, wherever and however often it is called.

    Lines, tones and measurements cross as with `pulse.schedule_block`: each
    timed operand is the callee's matching argument, and each result is what
    the callee returns at that place. A line or tone that enters the call
    continues as one of its results. Numbers (`f64`, `i64`) are the values at
    the call.

    The callee records its length in `pulse.length_cycles`. A placed call
    lasts `duration_cycles`, at least that length. When it lasts longer,
    every line the callee plays on is held until the call's end, so nothing
    else plays on it before then.

    ```mlir
    func.func private @x90(%d: !pulse.drive_line, %t: !pulse.tone)
        -> (!pulse.drive_line, !pulse.tone)
        attributes {pulse.length_cycles = 40 : i64} { ... }

    %d1, %t1 = pulse.call @x90(%d0, %t0) <{duration_cycles = 40 : i64, start_cycles = 8 : i64}>
        : (!pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone)
    ```
  }];
  let arguments = !con((ins FlatSymbolRefAttr:$callee,
                            Variadic<AnyType>:$args),
                       PulseTimingArgs);
  let results = (outs Variadic<AnyTimedType>:$results);
  let assemblyFormat = [{
    $callee `(` $args `)` `` custom<Timing>("getOperationName()", $start_cycles, $duration_cycles, $period_cycles) attr-dict `:` functional-type($args, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IQ operations
//===----------------------------------------------------------------------===//

def IQAcquireOp : PulseOp<"iq_acquire", []> {
  let summary = "Acquire raw IQ data from a readout line";
  let description = [{
    Plays `pulse` on a readout line at `tone` and returns the raw
    demodulated IQ point instead of a discriminated measurement. Like
    `pulse.readout`, it returns the updated line and tone, so one tone can
    be acquired with again. Unlike `pulse.readout`, it is not scheduled: it
    gets no start or duration, and the controller export has no event for
    it.
  }];
  let arguments = (ins
    ReadoutLineType:$line,
    WaveformType:$pulse,
    ToneType:$tone
  );
  let results = (outs ReadoutLineType:$updated_line, ToneType:$updated_tone,
                      IQDataType:$iq_data);
  let assemblyFormat = [{
    $line `,` $pulse `,` $tone attr-dict `:` qualified(type($line))
    `,` qualified(type($pulse)) `,` qualified(type($tone))
    `->` qualified(type($updated_line)) `,` qualified(type($updated_tone))
    `,` qualified(type($iq_data))
  }];
}

def ExtractIQOp : PulseOp<"extract_iq", [Pure]> {
  let summary = "Extract I and Q components from IQ data";
  let description = [{
    The in-phase and quadrature components of an IQ point.
  }];
  let arguments = (ins IQDataType:$iq_data);
  let results = (outs F64:$i_value, F64:$q_value);
  let assemblyFormat = "$iq_data attr-dict `:` qualified(type($iq_data)) `->` type($i_value) `,` type($q_value)";
}

def CreateIQOp : PulseOp<"create_iq", [Pure]> {
  let summary = "Create IQ data from I and Q values";
  let description = [{
    An IQ point from its in-phase and quadrature components.
  }];
  let arguments = (ins F64:$i_value, F64:$q_value);
  let results = (outs IQDataType:$iq_data);
  let assemblyFormat = "$i_value `,` $q_value attr-dict `:` type($i_value) `,` type($q_value) `->` qualified(type($iq_data))";
}

def IQMagnitudeOp : PulseOp<"iq_magnitude", [Pure]> {
  let summary = "Compute magnitude of IQ data: sqrt(I^2 + Q^2)";
  let description = [{
    The magnitude `sqrt(I^2 + Q^2)` of an IQ point.
  }];
  let arguments = (ins IQDataType:$iq_data);
  let results = (outs F64:$magnitude);
  let assemblyFormat = "$iq_data attr-dict `:` qualified(type($iq_data)) `->` type($magnitude)";
}

def IQPhaseOp : PulseOp<"iq_phase", [Pure]> {
  let summary = "Compute phase of IQ data: atan2(Q, I)";
  let description = [{
    The phase `atan2(Q, I)` of an IQ point, in radians.
  }];
  let arguments = (ins IQDataType:$iq_data);
  let results = (outs F64:$phase);
  let assemblyFormat = "$iq_data attr-dict `:` qualified(type($iq_data)) `->` type($phase)";
}

#endif // CUDAQ_PULSE_DIALECT_PULSE_OPS
