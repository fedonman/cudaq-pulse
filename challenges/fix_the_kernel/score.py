#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scorer of the challenge "Fix the kernel".

    python challenges/fix_the_kernel/score.py my_kernel.py [seed]

my_kernel.py is a fixed copy of broken.py: it defines a kernel ``two_turns(qudit, turn, gap_ns)``
that turns the qubit by ``turn`` of a full flip with one pulse, waits ``gap_ns``, turns it again by
as much with a second pulse, and reads it out when the second pulse ends.

The submission never runs in the scorer's process. A second Python process (_child.py) loads the
file, compiles two_turns for qubit D1 of pulse.quantum_system("transmon_surface17") on the bench
controller, once for each of CASES pairs of turn and gap_ns that the scorer draws, and writes what
each plays: the pulses and the readout with their starts, samples, frequencies and phases, or the
compiler's message. The scorer reads only those numbers and checks each case on them:

- two pulses, a wait and one readout are played, and nothing else;
- every sample, the readout's too, is at most MAX_AMPLITUDE in magnitude, every pulse within
  MAX_DETUNING_HZ of the qubit's 0-1 frequency and at most MAX_PULSE_NS long;
- the second pulse starts gap_ns after the first ends, and the readout starts when the second
  pulse ends: MIN_READOUT_NS or longer, on the readout line's own tone, reaching
  MIN_READOUT_AMPLITUDE or more;
- played by the scorer from level 0, each pulse alone leaves sin(pi turn / 2)^2 in level 1 to
  within PULSE_TOLERANCE, and the two with their wait sin(pi turn)^2 to within PAIR_TOLERANCE.

Each run draws new cases unless a seed is given; the seed is printed at the end, so a run can be
repeated.
"""

import math
import secrets
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
from _bench import MAX_AMPLITUDE, QUBIT, bench_readout, the_bench

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
    scorer_of,
)

drive_from_level_0 = scorer_of("pi_pulse").replay

SMALL_TURNS, LARGE_TURNS = (0.10, 0.70, 8), (0.80, 1.00, 4)  # from, to, how many cases
CASES = SMALL_TURNS[2] + LARGE_TURNS[2]
GAP_RANGE_NS = (4, 200)  # whole ns
MAX_DETUNING_HZ = 50.0e6
MAX_PULSE_NS = 64
MIN_READOUT_NS = 300
MIN_READOUT_AMPLITUDE = 0.01  # a readout that reaches less is silent
PULSE_TOLERANCE, PAIR_TOLERANCE = 0.02, 0.03


def judge(turn, gap_ns, reply, system, qudit, tone):
    """None if the case does what two_turns is for, else the first thing that is wrong.

    *tone* is the readout line's own: its frequency and phase with no frame change.
    """
    if not isinstance(reply, dict):
        raise InvalidSubmissionError("its result could not be read")
    if "error" in reply:
        return f"it does not compile:{said(reply['error'])}"
    try:
        drives, readouts, waits, others = read_driven(reply)
    except InvalidSubmissionError as error:
        return str(error)
    clock = system.clock_ghz
    if others:
        return f"it has {others[0]}; two pulses, a wait and one readout are played"
    if any(item["qudit"] != qudit.index for item in drives + readouts):
        return f"it plays on a qubit that is not {QUBIT}"
    if len(drives) != 2 or len(readouts) != 1:
        return f"it plays {counted(len(drives), 'pulse')} and {counted(len(readouts), 'readout')}; two_turns plays two pulses and one readout"
    in_branch = [drive["depends"] for drive in drives] + [readouts[0]["in_branch"]]
    if any(in_branch) or any(wait[3] for wait in waits):
        return "a pulse, a wait or the readout depends on a measurement"
    first, second = sorted(drives, key=lambda drive: drive["start"])
    measurement = readouts[0]
    for name, drive in (("first", first), ("second", second)):
        peak = max(abs(sample) for sample in drive["samples"])
        if peak > MAX_AMPLITUDE:
            return f"a sample of the {name} pulse reaches {digits(peak, MAX_AMPLITUDE)}; the bench plays at most {MAX_AMPLITUDE:g}"
        detuning_mhz = (drive["frequency_hz"] - qudit.dressed_frequency_hz) / 1e6
        if abs(detuning_mhz) > MAX_DETUNING_HZ / 1e6:
            return f"the {name} pulse is {detuning_mhz:+g} MHz from the qubit; the limit is {MAX_DETUNING_HZ / 1e6:g} MHz"
        if len(drive["samples"]) > MAX_PULSE_NS * clock:
            return f"the {name} pulse lasts {len(drive['samples']) / clock:g} ns; the limit is {MAX_PULSE_NS} ns"
    gap = second["start"] - first["end"]
    if gap != round(gap_ns * clock):
        return (
            f"the second pulse starts {gap / clock:g} ns after the first ends; gap_ns is {gap_ns:g}"
        )
    if measurement["start"] != second["end"]:
        return f"the readout starts at {measurement['start'] / clock:g} ns; the second pulse ends at {second['end'] / clock:g} ns"
    if len(measurement["samples"]) < MIN_READOUT_NS * clock:
        return f"the readout lasts {len(measurement['samples']) / clock:g} ns; it takes {MIN_READOUT_NS} ns or more"
    peak = max(abs(sample) for sample in measurement["samples"])
    if peak > MAX_AMPLITUDE:
        return f"a sample of the readout reaches {digits(peak, MAX_AMPLITUDE)}; the bench plays at most {MAX_AMPLITUDE:g}"
    if peak < MIN_READOUT_AMPLITUDE:
        return f"the readout reaches {peak:.3g}; one that reaches less than {MIN_READOUT_AMPLITUDE:g} is silent"
    turned = (measurement["phase_rad"] - tone["phase_rad"]) % (2.0 * math.pi)
    if (
        abs(measurement["frequency_hz"] - tone["frequency_hz"]) > 1e-3
        or min(turned, 2.0 * math.pi - turned) > 1e-9
    ):
        return (
            "the readout is not on the readout line's own tone: its frequency or phase was changed"
        )
    one, pair = math.sin(math.pi * turn / 2) ** 2, math.sin(math.pi * turn) ** 2
    try:
        for name, drive in (("first", first), ("second", second)):
            alone = float(drive_from_level_0([{**drive, "start": 0}], system, qudit)[1])
            if not abs(alone - one) <= PULSE_TOLERANCE:
                return f"from level 0 the {name} pulse leaves {alone:.3f} in level 1; a turn of {turn:g} leaves {one:.3f}"
        moved = [{**drive, "start": drive["start"] - first["start"]} for drive in (first, second)]
        both = float(drive_from_level_0(moved, system, qudit)[1])
    except Exception as error:  # the numbers do not make playable pulses
        return f"its pulses could not be played:{said(error)}"
    if not abs(both - pair) <= PAIR_TOLERANCE:
        return f"from level 0 the two pulses leave {both:.3f} in level 1; two turns of {turn:g} leave {pair:.3f}"
    return None


def draw(rng):
    """The cases of one run: one turn from each slice of the two ranges, each with a gap."""
    turns = []
    for low, high, count in (SMALL_TURNS, LARGE_TURNS):
        turns += list(low + (high - low) * (np.arange(count) + rng.random(count)) / count)
    gaps = rng.integers(GAP_RANGE_NS[0], GAP_RANGE_NS[1] + 1, CASES)
    order = rng.permutation(CASES)
    return [(round(float(turns[i]), 2), float(gaps[i])) for i in order]


def warm_up(system, qudit):
    """The readout line's tone; and everything the scorer does later, done once before the
    second process starts: a case that passes is judged, pulses played and all."""
    compiled = pulse.compile(bench_readout, quantum_system=system).lower(qudit)
    tone = read_driven(driven(compiled))[1][0]
    turn, gap_ns, count = 0.5, 10.0, 64
    shape = [
        0.165 * turn * math.exp(-((k - count / 2) ** 2) / (2.0 * 16.0**2)) for k in range(count)
    ]
    drive = {"qudit": qudit.index, "re": shape, "im": [0.0] * count, "depends": []}
    drive.update(frequency_hz=float(qudit.dressed_frequency_hz), phase_rad=0.0)
    second = count + round(gap_ns * system.clock_ghz)
    reply = {
        "drives": [{**drive, "start_cycles": 0}, {**drive, "start_cycles": second}],
        "readouts": [
            {
                "qudit": qudit.index,
                "start_cycles": second + count,
                "re": [sample.real for sample in tone["samples"]],
                "im": [sample.imag for sample in tone["samples"]],
                "frequency_hz": tone["frequency_hz"],
                "phase_rad": tone["phase_rad"],
                "in_branch": False,
            }
        ],
        "waits": [],
        "others": [],
    }
    problem = judge(turn, gap_ns, reply, system, qudit, tone)
    assert problem is None, problem
    draw(np.random.default_rng(1))
    return tone


def main(path, seed=None):
    given = seed is not None
    seed = seed if given else secrets.randbelow(2**31)
    system, qudit, _ = the_bench()
    tone = warm_up(system, qudit)
    cases = draw(np.random.default_rng(seed))
    try:
        reply = ask_child(HERE / "_child.py", [path], {"cases": cases})
        if "error" in reply:
            raise InvalidSubmissionError(f"it does not load:{said(reply['error'])}")
        replies = reply.get("cases")
        if type(replies) is not list or len(replies) != CASES:
            raise InvalidSubmissionError("its result could not be read")
        problems = [
            judge(turn, gap_ns, one, system, qudit, tone)
            for (turn, gap_ns), one in zip(cases, replies, strict=True)
        ]
        if len(set(problems)) == 1 and str(problems[0]).startswith("it does not compile"):
            raise InvalidSubmissionError(problems[0])  # the same message for every case
    except InvalidSubmissionError as error:
        print(f"INVALID {error}\nBRONZE fail\nSILVER fail\nSCORE none")
        return 1
    for index, ((turn, gap_ns), problem) in enumerate(zip(cases, problems, strict=True), start=1):
        verdict = "ok" if problem is None else f"FAIL {problem}"
        print(f"case {index:2d}: turn {turn:.2f}, gap {gap_ns:3.0f} ns: {verdict}")
    small = [
        problem is None
        for (turn, _), problem in zip(cases, problems, strict=True)
        if turn <= SMALL_TURNS[1]
    ]
    passed = sum(problem is None for problem in problems)
    bronze, silver = all(small), passed == CASES
    repeat = " (repeat run)" if given else ""
    print(
        f"BRONZE {'pass' if bronze else 'fail'}{repeat}: {sum(small)} of {len(small)} cases with a turn of at most {SMALL_TURNS[1]:g} do what two_turns is for"
    )
    print(f"SILVER {'pass' if silver else 'fail'}{repeat}: {passed} of {CASES} cases do")
    if given:
        print("SCORE none (a run with a given seed repeats known cases: it is not a score)")
    else:
        print(f"SCORE {passed} of {CASES} cases")
    print(f"SEED {seed} (give it as the second argument to repeat this run)")
    return 0


if __name__ == "__main__":
    run_scorer(main, "usage: python challenges/fix_the_kernel/score.py my_kernel.py [seed]")
