#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scorer of the challenge "Reset one qubit".

    python challenges/reset_one_qubit/score.py my_reset.py [seed]

my_reset.py defines a kernel ``reset(qudit)``.

The submission never runs in the scorer's process. A second Python process (_child.py) loads the
file, compiles reset for qubit D1 of pulse.quantum_system("transmon_surface17") on a controller
with LATENCY_NS of feedback latency, and writes what it plays: each pulse and readout with its
start, its samples, its frequency and its phase, and for a pulse in a branch the readout it
depends on. The scorer reads only those numbers, checks them against the limits below, and plays
them itself from STATES states that only it knows, with D1's decay and readout errors. A state's
result is what is outside level 0 BUDGET_NS after the start. Nothing is sampled.

Limits (the scorer's, checked on the numbers): only D1's pulses, waits and readouts; every readout
is D1's calibrated one and plays in every shot, not in a branch; no pulse while D1 is read out; a
pulse in a branch depends on one readout and starts LATENCY_NS or more after it ends; pulses are
at D1's frequency with phase 0 and at most MAX_AMPLITUDE in magnitude; at most MAX_PULSES pulses
and MAX_READOUTS readouts; everything, waits included, ends within BUDGET_NS.

Each run draws new states unless a seed is given; the seed is printed at the end, so a run can be
repeated.
"""

import math
import secrets
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
from _qubit import LATENCY_NS, QUBIT, calibrated_readout, one_round, the_qubit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _exchange import (  # noqa: E402  (after the path it is found on)
    InvalidSubmissionError,
    ask_child,
    counted,
    digits,
    driven,
    read_driven,
    run_scorer,
    said,
    same_pulse,
)

BUDGET_NS = 1200  # a reset ends within it, and the result is read then
MAX_AMPLITUDE = 0.5  # the calibrated X pulse peaks at 0.13
MAX_PULSES = 16
MAX_READOUTS = 2  # the controller starts a readout LATENCY_NS after the one before: no more fit
STATES = 8
BRONZE = 0.021  # what waiting for ever leaves: D1's thermal population
SILVER = 0.003
CHUNK = 252  # the most samples one custom_samples() waveform holds
LINE_CYCLES = 10_000  # more clock cycles than any reset that passes the check lasts


def overlap(one, other):
    return one["start"] < other["end"] and other["start"] < one["end"]


def check(drives, readouts, waits, others, system, qudit, calibrated):
    """(problems, the end of the last pulse, readout or wait in ns) of what is played."""
    to_ns = 1.0 / system.clock_ghz
    latency = round(LATENCY_NS * system.clock_ghz)
    # D1's drive tone: its 0-1 frequency dressed by its couplings.
    native = qudit.dressed_frequency_hz
    problems = [
        f"it has {kind}; only pulses, waits and readouts of {QUBIT} are played" for kind in others
    ]
    if any(item["qudit"] != qudit.index for item in drives + readouts):
        problems.append(f"it plays on a qubit that is not {QUBIT}")
    if any(readout["in_branch"] for readout in readouts):
        problems.append(
            "a readout is inside an if on an outcome; a readout plays in every shot, only pulses and waits go in a branch"
        )
    if len(drives) > MAX_PULSES or len(readouts) > MAX_READOUTS:
        problems.append(
            f"it has {counted(len(drives), 'pulse')} and {counted(len(readouts), 'readout')}; the limits are {MAX_PULSES} and {MAX_READOUTS}"
        )
        return problems, 0.0  # nothing more is looked at
    if not all(same_pulse(readout, calibrated) for readout in readouts):
        problems.append(f"a readout is not {QUBIT}'s calibrated readout")
    if any(overlap(a, b) for i, a in enumerate(readouts) for b in readouts[:i]):
        problems.append("two readouts overlap")
    for i, drive in enumerate(drives):
        at = f"the pulse at {drive['start'] * to_ns:g} ns"
        peak = max(abs(sample) for sample in drive["samples"])
        if peak > MAX_AMPLITUDE:
            problems.append(
                f"{at} reaches {digits(peak, MAX_AMPLITUDE)}; the limit is {MAX_AMPLITUDE:g}"
            )
        turn = drive["phase_rad"] % (2.0 * math.pi)
        if abs(drive["frequency_hz"] - native) > 1e-3 or min(turn, 2.0 * math.pi - turn) > 1e-9:
            problems.append(
                f"{at} is not at {QUBIT}'s frequency with phase 0; frame changes are not played"
            )
        if any(overlap(drive, readout) for readout in readouts):
            problems.append(f"{at} plays while {QUBIT} is read out")
        if len(drive["depends"]) > 1:
            problems.append(f"{at} depends on more than one readout")
        for readout, _ in drive["depends"][:1]:
            ready = readouts[readout]["end"] + latency
            if drive["start"] < ready:
                problems.append(
                    f"{at} depends on the readout that ends at {readouts[readout]['end'] * to_ns:g} ns; the controller needs {LATENCY_NS} ns, so it starts at {ready * to_ns:g} ns or later"
                )
        for other in drives[:i]:
            arms = {
                (d["depends"][0][0], d["depends"][0][1]) for d in (drive, other) if d["depends"]
            }
            two_arms = len(arms) == 2 and len({readout for readout, _ in arms}) == 1
            if overlap(drive, other) and not two_arms:
                problems.append(f"{at} overlaps another pulse on the drive line")
    ends = [item["end"] for item in drives + readouts] + [wait[2] for wait in waits]
    end_ns = max(ends, default=0) * to_ns
    if end_ns > BUDGET_NS:
        problems.append(
            f"it ends at {end_ns:g} ns, its last pulse, readout or wait; the limit is {BUDGET_NS} ns"
        )
    return problems, end_ns


def pieces(track):
    """A stretch of the drive line in pieces of at most CHUNK samples: (real, imag), or None for silence."""
    cut = [track[at : at + CHUNK] for at in range(0, len(track), CHUNK)]
    return [
        ([s.real for s in piece], [s.imag for s in piece]) if any(piece) else None for piece in cut
    ]


def plan_of(drives, readouts, readout_cycles):
    """The checked pulses and readouts as steps of the scorer's own kernel, in time order.

    A step is (kind, wait before it, readout it depends on or -1, what plays if that readout is
    set, what plays if not, length), in clock cycles. Pulses that overlap are the two arms of one
    branch (check allows nothing else): they become one stretch of the drive line with what each
    arm plays on it, silence where an arm is idle.
    """
    stretches = []  # [start, end, readout or None, {arm: samples}]
    for drive in sorted(drives, key=lambda d: d["start"]):
        readout, arm = drive["depends"][0] if drive["depends"] else (None, True)
        if not stretches or drive["start"] >= stretches[-1][1]:
            stretches.append([drive["start"], drive["start"], readout, {}])
        stretch = stretches[-1]
        stretch[1] = max(stretch[1], drive["end"])
        track = stretch[3].setdefault(arm, [])
        track.extend([0j] * (drive["start"] - stretch[0] - len(track)))
        track.extend(drive["samples"])
    # A readout's outcome is named by its place in the kernel, so by its place in time.
    order = sorted(range(len(readouts)), key=lambda i: readouts[i]["start"])
    place = {readout: n for n, readout in enumerate(order)}
    steps = [
        (readout["start"], 0, "readout", -1, None, None, readout_cycles) for readout in readouts
    ]
    for start, end, readout, tracks in stretches:
        arms = [tracks.get(arm, []) for arm in (True, False)]
        arms = [pieces(track + [0j] * (end - start - len(track))) for track in arms]
        for n, (if_set, if_not) in enumerate(zip(*arms, strict=True)):
            length = min(CHUNK, end - start - n * CHUNK)
            which = -1 if readout is None else place[readout]
            steps.append((start + n * CHUNK, 1, "drive", which, if_set, if_not, length))
    steps.sort(key=lambda step: step[:2])
    plan, clock = [], {"readout": 0, "drive": 0}
    for start, _, kind, which, if_set, if_not, length in steps:
        plan.append((kind, start - clock[kind], which, if_set, if_not, length))
        clock[kind] = start + length
    return plan


def drive_plan(plan, system, qudit, states):
    """What each state leaves outside level 0 at BUDGET_NS: the scorer plays the plan itself."""

    @pulse.kernel
    def driven_here(target):
        drive_line = get_drive_line(target)
        readout_line = get_readout_line(target)
        measurement = readout_line.get_calibration(pulse.gates.READOUT)
        outcomes = []
        for kind, gap, which, if_set, if_not, length in plan:
            if kind == "readout":
                if gap > 0:
                    wait(readout_line, gap)
                outcomes.append(readout(readout_line, measurement))
                continue
            if gap > 0:
                wait(drive_line, gap)
            first = second = None
            if if_set is not None:
                quadrature = wf_mul(custom_samples(if_set[1]), square(length, 1j))
                first = wf_add(custom_samples(if_set[0]), quadrature)
            if if_not is not None:
                quadrature = wf_mul(custom_samples(if_not[1]), square(length, 1j))
                second = wf_add(custom_samples(if_not[0]), quadrature)
            if which < 0 and first is not None:
                drive(drive_line, first)
            elif which < 0 or (first is None and second is None):
                wait(drive_line, length)
            elif second is None:
                if outcomes[which]:
                    drive(drive_line, first)
                else:
                    wait(drive_line, length)
            elif first is None:
                if outcomes[which]:
                    wait(drive_line, length)
                else:
                    drive(drive_line, second)
            else:
                if outcomes[which]:
                    drive(drive_line, first)
                else:
                    drive(drive_line, second)

    program = pulse.compile(driven_here, quantum_system=system, schedule="asap").lower(qudit)
    left = []
    for state in states:
        result = pulse.simulate(
            program, initial_state=state, readout="model", t_end=BUDGET_NS * pulse.ns
        )
        left.append(1.0 - float(result.level_populations(qudit)[0]))
    return left, program


def on_the_lines(drives, readouts):
    """What plays when: the readouts' starts, and the drive line's samples in each branch arm."""
    tracks = {}
    for drive in drives:
        arm = [(readouts[readout]["start"], arm) for readout, arm in drive["depends"]]
        track = tracks.setdefault(tuple(arm), np.zeros(LINE_CYCLES, dtype=complex))
        track[drive["start"] : drive["end"]] += drive["samples"]
    return sorted(readout["start"] for readout in readouts), tracks


def same_lines(one, other):
    """Whether two programs play the same; an arm that plays only silence is no arm."""
    one, other = (
        (starts, {arm: t for arm, t in tracks.items() if t.any()})
        for starts, tracks in (one, other)
    )
    return (
        one[0] == other[0]
        and one[1].keys() == other[1].keys()
        and all(
            np.allclose(track, other[1][arm], rtol=0.0, atol=1e-9) for arm, track in one[1].items()
        )
    )


def drive_again(drives, readouts, calibrated, system, qudit, states):
    """What each state leaves outside level 0, played by the scorer; raises if it cannot be sure."""
    plan = plan_of(drives, readouts, len(calibrated["samples"]))
    left, program = drive_plan(plan, system, qudit, states)
    mine = read_driven(driven(program))
    if not same_lines(on_the_lines(drives, readouts), on_the_lines(mine[0], mine[1])):
        raise ValueError("the scorer's own program is not the one it checked")
    if not all(-1e-9 <= rest <= 1.0 + 1e-9 for rest in left):
        raise ValueError("a population is not a number between 0 and 1")
    return left


def draw(seed):
    """The states of one run: one from each slice of the level-1 population, each with a phase."""
    rng = np.random.default_rng(seed)
    excited = rng.permutation((np.arange(STATES) + rng.random(STATES)) / STATES)
    phases = rng.uniform(0.0, 2.0 * math.pi, STATES)
    states = [
        [math.sqrt(1.0 - p), math.sqrt(p) * complex(math.cos(phi), math.sin(phi)), 0.0]
        for p, phi in zip(excited, phases, strict=True)
    ]
    return excited, states


def warm_up(system, qudit, controller):
    """D1's calibrated readout; and everything the scorer does later, done once before the
    second process starts: a reset of its own is checked, played back and simulated."""
    compiled = pulse.compile(calibrated_readout, quantum_system=system).lower(qudit)
    calibrated = read_driven(driven(compiled))[1][0]
    options = {"control_system": controller, "schedule": "rcp", "flatten_schedules": True}
    program = pulse.compile(one_round, quantum_system=system, **options).lower(qudit)
    drives, readouts, waits, others = read_driven(driven(program))
    problems, _ = check(drives, readouts, waits, others, system, qudit, calibrated)
    assert not problems, problems
    drive_again(drives, readouts, calibrated, system, qudit, draw(1)[1][:1])
    return calibrated


def main(path, seed=None):
    given = seed is not None
    system, qudit, controller = the_qubit()
    calibrated = warm_up(system, qudit, controller)
    try:
        reply = ask_child(HERE / "_child.py", [path])
        if "error" in reply:
            raise InvalidSubmissionError(f"it does not load or compile:{said(reply['error'])}")
        drives, readouts, waits, others = read_driven(reply)
        problems, end_ns = check(drives, readouts, waits, others, system, qudit, calibrated)
    except InvalidSubmissionError as error:
        problems = [str(error)]
    if problems:
        for problem in problems[:8]:
            print(f"INVALID {problem}")
        print("BRONZE fail\nSILVER fail\nSCORE none")
        return 1
    # The states are drawn here, after the second process has ended.
    seed = seed if given else secrets.randbelow(2**31)
    excited, states = draw(seed)
    try:
        left = drive_again(drives, readouts, calibrated, system, qudit, states)
    except Exception as error:  # fail closed: nothing the scorer did not play exactly is scored
        print(f"INVALID the scorer could not play it back, so nothing is scored:{said(error)}")
        print("BRONZE fail\nSILVER fail\nSCORE none")
        return 1
    print(
        f"your reset: {counted(len(readouts), 'readout')} and {counted(len(drives), 'pulse')}, over at {end_ns:g} ns"
    )
    for index, (p, rest) in enumerate(zip(excited, left, strict=True), start=1):
        print(
            f"state {index}: starts with {p:.3f} in level 1, ends with {rest:.5f} outside level 0"
        )
    worst, mean = max(left), sum(left) / STATES
    repeat = " (repeat run)" if given else ""
    for rung, line, why in (
        ("BRONZE", BRONZE, " (what waiting for ever leaves)"),
        ("SILVER", SILVER, ""),
    ):
        within = sum(rest < line for rest in left)
        verdict = "pass" if within == STATES else "fail"
        print(
            f"{rung} {verdict}{repeat}: {within} of {STATES} states end with less than {line:g} outside level 0{why}"
        )
    if given:
        print("SCORE none (a run with a given seed repeats known states: it is not a score)")
    else:
        print(f"SCORE worst={worst:.5f} mean={mean:.5f}")
    print(f"SEED {seed} (give it as the second argument to repeat this run)")
    return 0


if __name__ == "__main__":
    run_scorer(main, "usage: python challenges/reset_one_qubit/score.py my_reset.py [seed]")
