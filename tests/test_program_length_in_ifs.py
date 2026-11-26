# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A call's float wait lengths in an if arm count only as far as they make
that arm the longer one: each call is checked as the same program with
those lengths written in is when built. The kernels take two lines, a
drive line and a readout line, whose operations follow each other, so the
schedule is what refuses them past the controller's longest program; op
after op they may sum to twice that."""

from __future__ import annotations

import math
import pickle
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError, Controller, ControlSystem

from tests._handles import Q

# 5000 ns on the 2 GHz clock of Q's quantum system: 10,000 cycles.
MACHINE = ControlSystem(controllers=[Controller(max_program_duration=5_000)])
# Either check's own sentence: the sum's, or the schedule's.
LIMIT = (
    r"the longest program of the controller \(Controller\.max_program_duration\)"
    r"|lasts \d+ cycles, more than controllers\[0\]'s max_program_duration of"
)
BOUNDS = (4, 10**6)
LENGTHS = [4.0, 1999.5, 3000.0, 4400.4, 6000.0, 7600.0, 9700.0, 12_000.0]
SHAPES = ["longer other arm", "both arms", "no else", "nested in a loop", "loop in an arm"]


def _hold(d, n, written):
    if written:
        wait(d, n)
    else:
        wait(d, n, bounds=BOUNDS)


def _kernel(shape, written=None):
    """*shape*, whose waits a and then b last the floats x and y, or the
    lengths *written* in."""
    w = written is not None

    @pulse.kernel
    def kernel(q, x: float, y: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        sync(d, r)
        a, b = (x, y) if written is None else written
        if shape == "longer other arm":
            if m:
                _hold(d, a, w)
            else:
                drive(d, square(4000, 0.1))
            _hold(d, b, w)
        elif shape == "both arms":
            if m:
                _hold(d, a, w)
                drive(d, square(500, 0.1))
            else:
                _hold(d, b, w)
        elif shape == "no else":
            _hold(d, a, w)
            if m:
                _hold(d, b, w)
        elif shape == "nested in a loop":
            for _ in range(3):
                if m:
                    if m:
                        _hold(d, a, w)
                    else:
                        drive(d, square(2500, 0.1))
                else:
                    _hold(d, b, w)
        elif shape == "loop in an arm":
            if m:
                drive(d, square(3000, 0.1))
            else:
                for _ in range(2):
                    _hold(d, a, w)
                    if m:
                        drive(d, square(1000, 0.1))
                    else:
                        _hold(d, b, w)

    return kernel


def _round(x):
    return math.floor(x + 0.5)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_call_is_checked_as_the_program_with_its_lengths_written_in(shape):
    compiled = pulse.compile(_kernel(shape), control_system=MACHINE)
    waits = compiled.lower(Q[0], 4.0, 4.0).generic.waits
    assert len(waits.length) == 5
    cycles, stop, limited, repeats = waits.length[:4]
    # The sum stops at the longest program, 10,000 cycles, times two lines.
    most = 10_000
    assert (stop, limited) == (2 * most, 1)
    newly = 0
    outcomes = set()
    for x in LENGTHS:
        for y in LENGTHS:
            written = (_round(x), _round(y))
            try:
                pulse.compile(_kernel(shape, written), control_system=MACHINE).lower(Q[0], 0.0, 0.0)
                fits = True
            except CompilationError as error:
                assert re.search(LIMIT, str(error)), error
                fits = False
            if fits:
                assert compiled.lower(Q[0], x, y) is not None, (x, y)
            else:
                with pytest.raises(ValueError, match=LIMIT):
                    compiled.lower(Q[0], x, y)
            # Each wait added in full, as the check was before.
            summed = cycles + sum(
                r * (_round(wait.length((x, y))) - wait.lowest)
                for r, wait in zip(repeats, waits.waits, strict=True)
            )
            newly += fits and summed > most
            outcomes.add(fits)
    assert outcomes == {True, False}
    # An arm that is empty, else, is never the longer one.
    assert bool(newly) == (shape != "no else")


def test_an_unpickled_module_checks_waits_in_ifs_as_compiled():
    waits = (
        pulse.compile(_kernel("longer other arm"), control_system=MACHINE)
        .lower(Q[0], 4.0, 4.0)
        .generic.waits
    )
    again = pickle.loads(pickle.dumps(waits))
    assert again.length == waits.length and len(waits.length) == 5
    assert again.ok.__code__.co_code == waits.ok.__code__.co_code
    again.check([5000.0, 4.0])
    # Within twice the longest program the sum passes, and the schedule decides.
    again.check([5000.0, 6000.0])
    with pytest.raises(ValueError, match=LIMIT):
        again.check([5000.0, 16_000.0])


def test_waits_outside_ifs_keep_the_sum():
    @pulse.kernel
    def flat(q, x: float):
        d = get_drive_line(q)
        for _ in range(3):
            wait(d, x, bounds=BOUNDS)

    waits = pulse.compile(flat, control_system=MACHINE).lower(Q[0], 4.0).generic.waits
    assert len(waits.length) == 4


def _two_lines_in_an_arm(cycles):
    @pulse.kernel
    def in_an_arm(a, b):
        d = get_drive_line(a)
        e = get_drive_line(b)
        r = get_readout_line(a)
        m = readout(r, square(100, 0.05))
        sync(d, e, r)
        if m:
            drive(d, square(cycles, 0.1))
            drive(e, square(cycles, 0.1))

    return in_an_arm


def test_two_lines_that_play_together_in_an_arm_count_once():
    # Each drive is under the 10,000 cycles; op after op the arm sums to
    # 12,100, which the sum alone refused, and it lasts 6100.
    lowered = pulse.compile(_two_lines_in_an_arm(6000), control_system=MACHINE).lower(Q[0], Q[1])
    assert max(e.start_cycles + e.duration_cycles for e in lowered.schedule) == 6100
    # Past the limit as scheduled, the schedule refuses it, as the sum did.
    with pytest.raises(CompilationError, match="lasts 10052 cycles, more than controllers"):
        pulse.compile(_two_lines_in_an_arm(9952), control_system=MACHINE).lower(Q[0], Q[1])
    # Past what three lines can last, the sum refuses it before any schedule.
    with pytest.raises(CompilationError, match="the program lasts more than 10000 cycles"):
        pulse.compile(_two_lines_in_an_arm(16_000), control_system=MACHINE).lower(Q[0], Q[1])
