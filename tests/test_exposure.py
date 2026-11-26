# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ScheduledKernel.exposure``: per qudit and per outermost loop, the cycles
of one round in which the qudit is driven, read out and idle, and the number
of rounds; exact, or marked where the schedule does not fix the figure."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import pytest

from tests._handles import Q
from tests.test_idle_numbered import (
    loop_in_an_arm_in_a_loop,
    measured_branch,
    nested_loops,
    ramsey,
)
from tests.test_idle_replacement import middle_tone
from tests.test_load import _as_written, _links


@pulse.kernel
def rounds(a, b, n: int):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    for _ in range(n):
        drive(da, square(16, 0.1))
        readout(rb, square(100, 0.05))
        wait(da, 200)
        sync(da, rb)
    drive(da, square(40, 0.1))


@pulse.classical(max_latency_ns=400, latency_mode="blocking")
def decode(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def held(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    ra = get_readout_line(a)
    drive(db, square(40, 0.1))
    for _ in range(3):
        m = readout(ra, square(100, 0.05))
        decode.submit([m]).result()
        drive(da, square(20, 0.3))


@pulse.kernel
def only_read(a, b):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    readout(rb, square(100, 0.05))


@pulse.kernel
def beside_a_loop(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(1000, 0.1))
    for _ in range(10):
        drive(db, square(100, 0.1))


@pulse.kernel
def two_loops_at_once(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(10):
        drive(da, square(100, 0.1))
    for _ in range(5):
        drive(db, square(200, 0.1))


@pulse.kernel
def tone_in_an_inner_loop(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(10):
        drive(da, square(100, 0.1))
        for _ in range(2):
            drive(db, square(50, 0.1), tone=da.tone)
        sync(da, db)


def _rows(program, qudit=None):
    return [
        (r.qudit, r.loop, r.rounds, r.cycles)
        + (r.drive_cycles, r.readout_cycles, r.idle_cycles, r.exact)
        for r in program.exposure
        if qudit is None or r.qudit == qudit
    ]


def _length(program):
    return max(e.start_cycles + e.duration_cycles for e in program.schedule)


def test_a_loop_is_one_row_per_qudit_with_its_rounds():
    program = pulse.compile(rounds).lower(Q[0], Q[1], 5)
    assert _rows(program) == [
        (0, None, 1, 80, 80, 0, 0, True),
        (0, 0, 5, 216, 16, 0, 200, True),
        (1, None, 1, 80, 0, 0, 80, True),
        (1, 0, 5, 216, 0, 100, 116, True),
    ]


@pytest.mark.parametrize("n", [1, 2, 3, 40])
def test_the_rows_add_up_to_the_programs_length(n):
    program = pulse.compile(rounds).lower(Q[0], Q[1], n)
    for qudit in (0, 1):
        rows = _rows(program, qudit)
        assert sum(r[2] * r[3] for r in rows) == _length(program)
        assert all(r[3] == r[4] + r[5] + r[6] for r in rows)


def test_a_million_rounds_are_the_row_of_two():
    few = pulse.compile(rounds).lower(Q[0], Q[1], 2)
    many = pulse.compile(rounds).lower(Q[0], Q[1], 1_000_000)
    assert [r[:2] + r[3:] for r in _rows(many)] == [r[:2] + r[3:] for r in _rows(few)]
    assert [r[2] for r in _rows(many)] == [1, 1_000_000, 1, 1_000_000]


def test_a_loop_on_another_line_takes_none_of_a_qudits_time():
    # Lines run on their own: the loop on b's line runs while a is driven.
    program = pulse.compile(beside_a_loop).lower(Q[0], Q[1])
    assert _length(program) == 1000
    assert _rows(program) == [
        (0, None, 1, 1000, 1000, 0, 0, True),
        (1, 0, 10, 100, 100, 0, 0, True),
    ]


def test_loops_that_run_at_the_same_time_are_each_their_own_qudits():
    program = pulse.compile(two_loops_at_once).lower(Q[0], Q[1])
    assert _length(program) == 1000
    assert _rows(program) == [
        (0, 0, 10, 100, 100, 0, 0, True),
        (1, 1, 5, 200, 200, 0, 0, True),
    ]


def test_a_tone_played_from_another_line_in_an_inner_loop_is_drive_time_of_its_qudit():
    program = pulse.compile(tone_in_an_inner_loop).lower(Q[0], Q[1])
    assert _rows(program) == [
        (0, 0, 10, 200, 200, 0, 0, True),
        (1, 0, 10, 200, 100, 0, 100, True),
    ]


def test_more_pulses_than_a_round_holds_are_the_rounds_length_and_not_exact():
    # A schedule no kernel here compiles to: in a round of 100 cycles qudit
    # 0 is driven for all 100 while its tone plays from qudit 1's line too.
    class Module:
        def site_tree(self):
            inner = ("for", 2, [("drive", None, 1, 0, 0, 50, [100, 50], False)])
            body = [("drive", None, 0, 0, 0, 100, [100], False), inner]
            return [("drive", 0), ("drive", 1)], [("for", 10, body)]

    from cudaq_pulse._exposure import rows

    found = [row for row in rows(Module()) if row.qudit == 0]
    assert [
        (r.loop, r.rounds, r.cycles, r.drive_cycles, r.idle_cycles, r.exact) for r in found
    ] == [(0, 10, 100, 100, 0, False)]


def test_a_wait_is_idle_time():
    program = pulse.compile(ramsey).lower(Q[0], Q[1])
    assert _rows(program) == [
        (0, None, 1, 480, 80, 0, 400, True),
        (1, None, 1, 480, 40, 0, 440, True),
    ]


def test_an_inner_loop_counts_every_iteration_in_its_round():
    program = pulse.compile(nested_loops).lower(Q[0], Q[1])
    # The loop is the whole program: there is no row for outside it.
    assert _rows(program, 0) == [(0, 0, 3, 120, 80, 0, 40, True)]
    assert _rows(program, 1) == [(1, 0, 3, 120, 120, 0, 0, True)]


def test_a_pulse_at_the_qudits_tone_from_another_line_is_drive_time():
    program = pulse.compile(middle_tone).lower(Q[0], Q[1])
    (a,) = _rows(program, 0)
    (b,) = _rows(program, 1)
    assert a[4] == 40 + 100 + 40
    assert b[4] == 100


def test_a_qudit_with_only_a_readout_line_has_its_row():
    program = pulse.compile(only_read).lower(Q[0], Q[1])
    assert _rows(program, 1) == [(1, None, 1, 100, 0, 100, 0, True)]
    assert _rows(program, 0) == [(0, None, 1, 100, 40, 0, 60, True)]


def test_arms_that_differ_count_the_less_busy_one_and_are_not_exact():
    program = pulse.compile(measured_branch).lower(Q[0], Q[1])
    # One arm drives qudit 0 for 40 cycles, the other waits.
    assert _rows(program, 0) == [(0, None, 1, 160, 40, 0, 120, False)]
    assert _rows(program, 1) == [(1, None, 1, 160, 40, 40, 80, True)]


def test_rounds_that_overlap_are_folded_into_the_period_and_not_exact():
    # Qudit 1's readout of a round runs while its drive of the round before
    # does: 40 cycles of each in a period of 48, of which 8 are idle.
    program = pulse.compile(loop_in_an_arm_in_a_loop).lower(Q[0], Q[1])
    assert _rows(program, 1)[1] == (1, 0, 3, 48, 40, 40, 8, False)


def test_a_loaded_program_reads_as_its_compiled_twin():
    system = pulse.quantum_system("transmon_surface17")
    machine = _links()
    native, text = _as_written(system, machine)
    loaded = pulse.load(
        text, entry="main", qudits=[system.qudits[0], system.qudits[2]], control_system=machine
    )
    assert loaded.exposure == native.exposure
    # Qudit 0's readout runs on into the next round's drive; qudit 2 is
    # driven only if the decoder says so, at a period of its own: 16 is the
    # shift read() gives that conditional drive.
    assert _rows(native) == [
        (0, None, 1, 1132, 0, 0, 1132, True),
        (0, 0, 2, 100, 16, 100, 0, False),
        (2, None, 1, 1300, 0, 0, 1300, True),
        (2, 0, 2, 16, 0, 0, 16, False),
    ]


def test_a_stretch_that_waits_for_an_answer_as_long_as_it_takes_is_not_exact():
    program = pulse.compile(held).lower(Q[0], Q[1])
    assert [(row.qudit, row.exact) for row in program.exposure if row.loop == 0] == [(0, False)]
    # Qudit 1 has no pulse in the loop: its time there is in its one row.
    assert [(row.loop, row.exact) for row in program.exposure if row.qudit == 1] == [(None, False)]


def test_decay_and_dephasing_are_of_one_rounds_idle_time():
    program = pulse.compile(ramsey).lower(Q[0], Q[1])
    row = program.exposure[0]
    assert (row.qudit, row.idle_cycles) == (0, 400)
    # The ladder's qudits: T1 50 us, T2 star 30 us.
    idle_us = 400 / program.quantum_system.clock_ghz / 1000
    assert row.decay == pytest.approx(1 - math.exp(-idle_us / 50))
    assert row.dephasing == pytest.approx((1 - math.exp(-idle_us * (1 / 30 - 1 / 100))) / 2)
    assert 0 < row.dephasing < row.decay < 1e-2


def test_it_prints_as_a_table():
    text = str(pulse.compile(ramsey).lower(Q[0], Q[1]).exposure).splitlines()
    assert text[0].split() == [
        "qudit", "loop", "rounds", "cycles", "drive", "readout", "idle", "idle", "%",
        "exact", "decay", "dephasing",
    ]  # fmt: skip
    assert text[1].split()[:9] == ["0", "-", "1", "480", "80", "0", "400", "83.3", "yes"]
    assert isinstance(pulse.compile(ramsey).lower(Q[0], Q[1]).exposure, pulse.compiler.Exposure)
