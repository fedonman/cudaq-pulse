# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the frame-phase figure of the operations guide.

Usage: python docs/user_guide/figures/frame_phase.py

Needs matplotlib (the ``[viz]`` extra). Writes ``frame_phase.svg`` and its
``-dark`` twin next to this file: the phase of a tone's frame over a kernel
that shifts its phase and then its frequency, above the pulses that take
it. The page describes the operations one by one, so the kernel is the one
below. Each pulse's phase and frequency are read from the controller
export, and the times of the two shifts from the scheduled IR; the line
between the pulses is the one frame that passes through all of them.
"""

import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import cudaq_pulse as pulse
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials" / "figures"))
from _style import WIDTH_IN, save_themed  # noqa: E402
from _timeline import Timeline, draw_on, layout  # noqa: E402

PHASE_STEP = math.pi / 2
DETUNING_HZ = 2.5e6


@pulse.kernel
def frames(q):
    line = get_drive_line(q)
    x90 = line.get_calibration(pulse.gates.X90)
    drive(line, x90)
    shift_phase(line.tone, PHASE_STEP)
    drive(line, x90)
    shift_frequency(line.tone, DETUNING_HZ)
    wait(line, 100 * pulse.ns)
    drive(line, x90)


def turns(phase: float) -> str:
    """*phase* in radians as a multiple of π."""
    part = Fraction(phase / math.pi).limit_denominator(16)
    assert abs(float(part) * math.pi - phase) < 1e-9, phase
    if part == 0:
        return "0"
    top = "π" if abs(part.numerator) == 1 else f"{abs(part.numerator)}π"
    return ("-" if part < 0 else "") + (
        top if part.denominator == 1 else f"{top}/{part.denominator}"
    )


def frame_phase() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    # Without passes, each shift stays an operation of the scheduled IR.
    scheduled = pulse.compile(frames, quantum_system=qs, passes=()).lower(qs.qudits[0])
    clock = qs.clock_ghz
    program = scheduled.export_to_controller()
    (line,) = (ln for ln in program["lines"] if ln["role"] == "drive")
    drives = [
        (
            event["start_cycles"] / clock,
            event["end_cycles"] / clock,
            event["phase_rad"],
            event["frequency_hz"] - line["native_frequency_hz"],
        )
        for event in program["events"]
        if event["kind"] == "drive"
    ]
    shifts = sorted(
        (int(start) / clock, op)
        for op, start in re.findall(
            r"= pulse\.(shift_phase|shift_frequency) .*start_cycles = (\d+)", scheduled.mlir()
        )
    )
    assert [op for _, op in shifts] == ["shift_phase", "shift_frequency"], shifts

    # The frame as a path: flat or sloped between the shifts, a step at shift_phase.
    path, steps = [(0.0, 0.0)], {}
    t, phase, offset_hz = 0.0, 0.0, 0.0
    pending = list(shifts)

    def advance(to: float) -> None:
        nonlocal t, phase
        phase += 2 * math.pi * offset_hz * (to - t) * 1e-9
        t = to
        path.append((t, phase))

    for start, _end, exported, detuning in drives:
        jump_at = None
        while pending and pending[0][0] <= start:
            when, op = pending.pop(0)
            advance(when)
            if op == "shift_phase":
                jump_at = len(path)
            else:
                offset_hz = detuning
        advance(start)
        # An exported phase may come wrapped to one turn.
        missing = (exported - phase + math.pi) % (2 * math.pi) - math.pi
        exported = phase + missing
        if jump_at is None:
            assert abs(missing) < 1e-9, (start, missing)
        else:
            # The step is what the next pulse's exported phase asks for.
            steps[path[jump_at - 1][0]] = missing
            path[jump_at:] = [(when, value + missing) for when, value in path[jump_at:]]
            path.insert(jump_at, (path[jump_at - 1][0], path[jump_at - 1][1] + missing))
            phase += missing
        assert abs(phase - exported) < 1e-9 and abs(offset_hz - detuning) < 1e-3
    advance(drives[-1][1])
    assert not pending and len(steps) == 1
    ((step_ns, step),) = steps.items()
    # The path was fitted to the second pulse's phase and the third's frequency; what the
    # kernel asked for and the third pulse's exported phase are the checks.
    assert abs(step - PHASE_STEP) < 1e-9 and abs(drives[-1][3] - DETUNING_HZ) < 1e-3
    drives = [
        (start, end, value, detuning)
        for (start, end, _, detuning), value in zip(
            drives, (dict(path)[start] for start, *_ in drives), strict=True
        )
    ]
    (slope_ns,) = (when for when, op in shifts if op == "shift_frequency")
    detuning_mhz = drives[-1][3] / 1e6
    timeline = Timeline.of(scheduled)
    lanes = layout(timeline)
    end = drives[-1][1]

    def figure(p):
        height = 2.0 + lanes[2] + 0.55
        fig = plt.figure(figsize=(WIDTH_IN, height))
        left = 1.25 / WIDTH_IN
        top = fig.add_axes((left, (lanes[2] + 0.85) / height, 0.985 - left, 1.75 / height))
        low = fig.add_axes((left, 0.42 / height, 0.985 - left, (lanes[2] + 0.12) / height))
        draw_on(
            low,
            timeline,
            p,
            lanes=lanes,
            xlim=(0, end * 1.01),
            xlabel=True,
            syncs="none",
            pulse_labels={(pl.lane, round(pl.start_ns, 3)): "X90" for pl in timeline.pulses},
        )
        times, values = zip(*path, strict=True)
        top.plot(times, np.array(values) / math.pi, color=p.drive, lw=1.6)
        for start, _end, exported, _ in drives:
            top.plot(
                [start], [exported / math.pi], "o", ms=5, color=p.drive, zorder=3, clip_on=False
            )
            top.annotate(
                turns(exported),
                (start, exported / math.pi),
                xytext=(4 if start == 0 else 0, 6),
                textcoords="offset points",
                ha="left" if start == 0 else "center",
                va="bottom",
                fontsize=8.5,
                color=p.text,
            )
        top.annotate(
            f"shift_phase, {turns(step)}",
            (step_ns, step / math.pi / 2),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            color=p.muted,
        )
        top.annotate(
            f"shift_frequency, {detuning_mhz:+g} MHz",
            (slope_ns, dict(path)[slope_ns] / math.pi),
            xytext=(8, -5),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=p.muted,
        )
        top.set_xlim(0, end * 1.01)
        top.set_ylim(-0.12, max(values) / math.pi + 0.3)
        ticks = np.arange(0, max(values) / math.pi + 0.01, 0.5)
        top.set_yticks(ticks, [turns(tick * math.pi) for tick in ticks])
        top.set_ylabel("frame phase (rad)")
        top.tick_params(axis="x", labelbottom=False)
        top.spines["bottom"].set_color(p.rule)
        return fig

    told = ", ".join(
        f"{turns(exported)} for the pulse from {start:g} ns" for start, _end, exported, _ in drives
    )
    save_themed(
        figure,
        "frame_phase",
        title="A tone's frame phase over a phase shift and a frequency shift",
        desc=(
            "The phase of a drive tone's frame against time, above the three X90 pulses that "
            f"take it. A shift_phase of {turns(step)} at {step_ns:g} ns is a step; a "
            f"shift_frequency of {detuning_mhz:+g} MHz at {slope_ns:g} ns makes the phase rise "
            f"steadily from there. The phase each pulse starts with: {told}. Each pulse's "
            "small Q quadrature is drawn dashed."
        ),
        directory=HERE,
    )


def main() -> None:
    frame_phase()


if __name__ == "__main__":
    main()
