#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scorer of the challenge "Find the pi pulse".

    python challenges/pi_pulse/score.py my_pulse.py

my_pulse.py defines a kernel named pi_pulse that takes one qudit.

The submission never runs in the scorer's process. A second Python process (_child.py) loads the
file, compiles pi_pulse for qubit Z4 of pulse.quantum_system("transmon_surface17") and writes what
it would send to the drive line: for each pulse its start, its samples, its frequency and its
phase. The scorer reads only those numbers, checks them against the limits below, plays them
itself from level 0 and reports the populations of levels 1 and 2 (three levels, T1 and T2*;
nothing is sampled, so every run prints the same numbers). A submission can therefore change
nothing about the rules, the qubit or the simulation.

Limits (the clock is the system's; the three limits are the challenge's, the system ships none):
only drive pulses, waits and frame changes; at most MAX_DURATION_NS in all; every sample at most
MAX_AMPLITUDE in magnitude; every pulse within MAX_DETUNING_HZ of the qubit's 0-1 frequency.
"""

import math
import sys
from pathlib import Path

import cudaq_pulse as pulse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exchange import (  # noqa: E402  (after the path it is found on)
    InvalidSubmissionError,
    ask_child,
    digits,
    read_pulses,
    run_scorer,
    said,
)

QUDIT = "Z4"
MAX_DURATION_NS = 20.0  # half of the shipped X pulse
MAX_AMPLITUDE = 0.5  # the shipped X pulse peaks at 0.13
MAX_DETUNING_HZ = 50.0e6
BRONZE_P1 = 0.98
SILVER_P1, SILVER_P2 = 0.999, 2.0e-4
MAX_PULSES = 64
MAX_PROBLEMS = 8  # printed
CHILD = HERE / "_child.py"


def read_submission(path):
    """The pulses of the file's pi_pulse as plain numbers, from the second process."""
    data = ask_child(CHILD, [path, QUDIT])
    if "error" in data:
        raise InvalidSubmissionError(f"it does not load or compile:{said(data['error'])}")
    return read_pulses(data)


def check(pulses, others, total, system, qudit):
    """(problems, duration in ns, peak amplitude) of the pulses against the limits."""
    problems = [f"it has {kind}; only drive pulses and waits are allowed" for kind in others]
    if len(pulses) > MAX_PULSES:
        problems.append(f"it has {len(pulses)} pulses; the limit is {MAX_PULSES}")
    end, peak, busy = max(total, 0), 0.0, 0  # waits after the last pulse count
    for p in pulses:
        if p["start"] < busy:
            problems.append("two pulses overlap on the drive line")
        busy = p["start"] + len(p["samples"])
        end = max(end, busy)
        peak = max(peak, max(abs(s) for s in p["samples"]))
        detuning_mhz = (p["frequency_hz"] - qudit.dressed_frequency_hz) / 1e6
        if abs(detuning_mhz) > MAX_DETUNING_HZ / 1e6:
            limit_mhz = MAX_DETUNING_HZ / 1e6
            problems.append(
                f"a pulse is {detuning_mhz:+g} MHz from the qubit; the limit is {limit_mhz:g} MHz"
            )
    duration_ns = end / system.clock_ghz
    if duration_ns > MAX_DURATION_NS:
        problems.append(f"it lasts {duration_ns:g} ns; the limit is {MAX_DURATION_NS:g} ns")
    if peak > MAX_AMPLITUDE:
        problems.append(
            f"a sample reaches {digits(peak, MAX_AMPLITUDE)}; the limit is {MAX_AMPLITUDE:g}"
        )
    return problems, duration_ns, peak


def replay(pulses, system, qudit):
    """Level populations after the scorer plays the pulses itself from level 0."""
    to_ns = 1.0 / system.clock_ghz
    # The line's own tone: the qubit's 0-1 frequency dressed by its couplings.
    native = qudit.dressed_frequency_hz
    steps, clock, frequency, frame, last = [], 0, native, 0.0, 0
    for p in pulses:
        # A pulse's phase is its frame's at its start. The scorer's own tone, off the qubit by
        # frequency - native since the last pulse began, has gathered part of it already.
        frame += 2.0 * math.pi * (frequency - native) * (p["start"] - last) * to_ns * 1e-9
        steps.append(
            (
                (p["start"] - clock) * to_ns,
                p["frequency_hz"] - frequency,
                p["phase_rad"] - frame,
                [s.real for s in p["samples"]],
                [s.imag for s in p["samples"]],
                len(p["samples"]),
            )
        )
        clock, last = p["start"] + len(p["samples"]), p["start"]
        frequency, frame = p["frequency_hz"], p["phase_rad"]

    @pulse.kernel
    def driven(target):
        line = get_drive_line(target)
        for gap_ns, df, dphi, real, imag, count in steps:
            if gap_ns > 0:
                wait(line, gap_ns * pulse.ns)
            if df != 0.0:
                shift_frequency(line.tone, df)
            if dphi != 0.0:
                shift_phase(line.tone, dphi)
            quadrature = wf_mul(custom_samples(imag), square(count, 1j))
            drive(line, wf_add(custom_samples(real), quadrature))

    program = pulse.compile(driven, quantum_system=system, schedule="asap").lower(qudit)
    return pulse.simulate(program, initial_state=[1.0, 0.0, 0.0]).level_populations(qudit)


def warm_up(system, qudit):
    """Everything the scorer does after the second process, done once before it starts."""
    pulses = [
        {"start": 4 * n, "samples": [0.01 + 0.01j] * 4, "frequency_hz": hz, "phase_rad": n}
        for n, hz in enumerate((qudit.dressed_frequency_hz, 1.0e6 + qudit.dressed_frequency_hz))
    ]
    check(pulses, ["a readout"], 8, system, qudit)
    replay(pulses, system, qudit)


def main(path):
    system = pulse.quantum_system("transmon_surface17")
    qudit = system.qudit(QUDIT)
    warm_up(system, qudit)
    p1 = p2 = 0.0
    try:
        pulses, others, total = read_submission(path)
        problems, duration_ns, peak = check(pulses, others, total, system, qudit)
        if pulses and not problems:
            _, p1, p2 = replay(pulses, system, qudit)
    except InvalidSubmissionError as error:
        print(f"INVALID {error}\nBRONZE fail\nSILVER fail\nSCORE none")
        return 1
    for problem in problems[:MAX_PROBLEMS]:
        print(f"INVALID {problem}")
    bronze = not problems and p1 >= BRONZE_P1
    silver = not problems and p1 >= SILVER_P1 and p2 <= SILVER_P2
    print(f"BRONZE {'pass' if bronze else 'fail'}  (P1 >= {BRONZE_P1:g})")
    print(f"SILVER {'pass' if silver else 'fail'}  (P1 >= {SILVER_P1:g} and P2 <= {SILVER_P2:g})")
    valid = "no" if problems else "yes"
    numbers = "P1=none P2=none" if problems else f"P1={p1:.6f} P2={p2:.6f}"
    print(f"SCORE {numbers} duration_ns={duration_ns:g} peak_amplitude={peak:.4f} valid={valid}")
    return 1 if problems else 0


if __name__ == "__main__":
    run_scorer(main, "usage: python challenges/pi_pulse/score.py my_pulse.py", seeded=False)
