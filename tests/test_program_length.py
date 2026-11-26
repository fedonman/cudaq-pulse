# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A program lasts at most its controller's longest program, and at most
2**60 cycles without one; a loop's length multiplies without overflow.
The module checks it as it is built, op after op, and a call checks it
again with the lengths its floats give its waits.
"""

from __future__ import annotations

import dis
import math
import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError, Controller, ControlSystem
from cudaq_pulse.controller import DriveLine

from tests._handles import Q

PAST_THE_BACKSTOP = r"the program lasts more than 2\*\*60 cycles"


def _waits(trips, cycles):
    @pulse.kernel
    def waits(q):
        d = get_drive_line(q)
        for _ in range(trips):
            wait(d, cycles)

    return waits


def test_a_program_of_2_60_cycles_compiles():
    assert _waits(128, 2**53).lower(Q[0]) is not None


@pytest.mark.parametrize("trips", [129, 300, 2000])
def test_a_program_past_2_60_cycles_is_an_error_at_its_loop(trips):
    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP) as info:
        _waits(trips, 2**53).lower(Q[0])
    assert "for _ in range(trips):" in "\n".join(info.value.__notes__)


def test_a_loop_length_past_2_63_does_not_wrap():
    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP):
        _waits(2**62, 4).lower(Q[0])

    @pulse.kernel
    def nested(q):
        d = get_drive_line(q)
        for _ in range(2**31):
            for _ in range(2**31):
                wait(d, 4)

    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP):
        nested.lower(Q[0])


def _delayed(cycles):
    # Q's quantum system runs a 2 GHz clock: a cycle is 0.5 ns.
    return ControlSystem(controllers=[Controller(drive_lines=[DriveLine(0, delay_ns=cycles / 2)])])


@pytest.mark.parametrize("cycles", [2**61, 10**19, 2**64, 10**30])
def test_a_line_delay_longer_than_a_program_is_an_error(cycles):
    with pytest.raises(CompilationError, match=r"\(its delay_ns\), more than a program lasts"):
        pulse.compile(_waits(1, 4), control_system=_delayed(cycles)).lower(Q[0])


def test_a_rolled_loop_its_line_delay_takes_past_2_60_cycles_is_an_error():
    # 64 trips of 2**53 cycles: 2**59 cycles, which a delay of 2**58 keeps short enough.
    loop = pulse.compile(_waits(64, 2**53), control_system=_delayed(2**58), **_ROLLED)
    assert loop.lower(Q[0]).duration / pulse.ns == pytest.approx((2**59 + 2**58) / 2)
    longer = pulse.compile(_waits(64, 2**53), control_system=_delayed(2**59 + 2**20), **_ROLLED)
    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP + " with its lines delayed"):
        longer.lower(Q[0])


def test_waits_of_run_time_length_are_checked_when_bound():
    @pulse.kernel
    def variable(q, n: float):
        d = get_drive_line(q)
        for _ in range(2**12):
            wait(d, n, bounds=(0, 2**53))

    compiled = pulse.compile(variable)
    assert compiled.lower(Q[0], 4.0) is not None
    with pytest.raises(ValueError, match=PAST_THE_BACKSTOP):
        compiled.lower(Q[0], float(2**52))


def test_a_program_past_the_controllers_longest_program_names_the_limit():
    # 5000 ns on the 2 GHz clock of Q's quantum system: 10,000 cycles.
    machine = ControlSystem(controllers=[Controller(max_program_duration=5_000)])
    assert pulse.compile(_waits(100, 100), control_system=machine).lower(Q[0]) is not None
    with pytest.raises(CompilationError, match=r"Controller\.max_program_duration"):
        pulse.compile(_waits(101, 100), control_system=machine).lower(Q[0])


def test_the_longest_program_may_be_a_time():
    assert Controller(max_program_duration=10 * pulse.us).max_program_duration == 10_000
    assert Controller(max_program_duration=1.5).max_program_duration == 1.5
    machine = ControlSystem(controllers=[Controller(max_program_duration=10 * pulse.us)])
    with pytest.raises(CompilationError, match=r"Controller\.max_program_duration"):
        pulse.compile(_waits(1000, 100), control_system=machine).lower(Q[0])


@pytest.mark.parametrize("wrong", [0, -1, True, math.nan, math.inf, "8"])
def test_max_program_duration_is_positive_or_none(wrong):
    with pytest.raises(pulse.InvalidValueError, match="max_program_duration"):
        Controller(max_program_duration=wrong)


def test_a_long_rolled_loop_past_the_controllers_longest_program_fails_fast():
    machine = ControlSystem(controllers=[Controller(max_program_duration=10**6)])
    with pytest.raises(CompilationError, match=r"Controller\.max_program_duration"):
        pulse.compile(_waits(10**15, 4), control_system=machine).lower(Q[0])


_LONGEST = ControlSystem(controllers=[Controller(max_program_duration=5_000)])
# Loops stay rolled, so a wait in one is one wait run several times.
_ROLLED = {"unroll_threshold": 0}


def _held(d, n, bounds):
    if bounds is None:
        wait(d, n)
    else:
        wait(d, n, bounds=bounds)


def _sweep(trips, bounds, written=None):
    @pulse.kernel
    def sweep(q, x: float, y: float):
        d = get_drive_line(q)
        a, b = (x, y) if written is None else written
        drive(d, square(100, 0.1))
        for _ in range(trips):
            _held(d, a, bounds if written is None else None)
        if trips == 1:
            _held(d, b, bounds if written is None else None)

    return sweep


@pytest.mark.parametrize("trips, bounds", [(3, (4, 10**6)), (3, None), (1, (4, 10**6)), (1, None)])
def test_a_call_fits_exactly_when_its_lengths_written_in_fit(trips, bounds):
    # One wait three times, folded into its bounds; or two once, summed.
    compiled = pulse.compile(_sweep(trips, bounds), control_system=_LONGEST, **_ROLLED)
    lowest = 0 if bounds is None else bounds[0]
    values = [4.0, 3299.4, 3299.5, 3300.49, 3300.5, 4950.0, 9895.5, 9896.5, 20_000.0]
    outcomes = set()
    for x in values:
        for y in values:
            written = (math.floor(x + 0.5), math.floor(y + 0.5))
            try:
                pulse.compile(
                    _sweep(trips, None, written), control_system=_LONGEST, **_ROLLED
                ).lower(Q[0], 0.0, 0.0)
                fits = True
            except CompilationError as error:
                assert "max_program_duration" in str(error)
                fits = False
            outcomes.add(fits)
            if fits:
                assert compiled.lower(Q[0], x, y) is not None, (x, y)
            else:
                with pytest.raises(ValueError, match=r"Controller\.max_program_duration"):
                    compiled.lower(Q[0], x, y)
    assert outcomes == {True, False}
    waits = compiled.lower(Q[0], float(lowest), float(lowest)).generic.waits
    # The most the program may last is its own test only with two waits.
    assert (10_000 in waits.ok.__code__.co_consts) == (trips == 1)


def test_waits_whose_bounds_keep_the_program_short_test_no_length():
    waits = pulse.compile(_sweep(3, None), **_ROLLED).lower(Q[0], 4.0, 4.0).generic.waits
    assert waits.length is not None and waits.length[3] == [3]
    assert 2**60 not in waits.ok.__code__.co_consts


# Unrolled, each trip's wait is a wait of its own, as each of a surface-code
# round's thousands of waits is: past about ten thousand of them, a check
# whose expression nested a level deeper per wait no longer compiled.
_MANY = 20_000


def _many_waits(shape):
    @pulse.kernel
    def many(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        sync(d, r)
        if shape == "in an if":
            if m:
                for _ in range(_MANY):
                    wait(d, tau)
                    drive(d, square(8, 0.1))
        else:
            for _ in range(_MANY):
                wait(d, tau)
                drive(d, square(8, 0.1))

    return many


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["in sequence", "in an if"])
def test_a_call_checks_twenty_thousand_waits_of_one_float(shape):
    # 1 ms on the 2 GHz clock of Q's quantum system: 2,000,000 cycles.
    machine = ControlSystem(controllers=[Controller(max_program_duration=10**6)])
    # Its loop unrolled: 20,000 trips of three calls (wait, drive, square).
    compiled = pulse.compile(_many_waits(shape), control_system=machine, unroll_threshold=3 * _MANY)
    waits = compiled.lower(Q[0], 4.0).generic.waits
    assert len(waits.waits) == len(waits.length[3]) == _MANY
    assert len(waits.length) == (5 if shape == "in an if" else 4)
    # With each wait k cycles long the program lasts its cycles + 20,000 k.
    cycles, most = waits.length[:2]
    k = (most - cycles) // _MANY
    again = pickle.loads(pickle.dumps(waits))
    assert again.ok.__code__.co_code == waits.ok.__code__.co_code
    for checked in (waits, again):
        checked.check([k + 0.49])
        with pytest.raises(ValueError, match=r"Controller\.max_program_duration"):
            checked.check([k + 0.5])
    # The kernel takes two lines, so op after op it may last twice the
    # longest program; its schedule may not, and decides.
    assert waits.length[1:3] == (4_000_000, 1)
    fits = (2_000_000 - cycles) // _MANY
    assert compiled.lower(Q[0], fits + 0.49) is not None
    # One half more rounds every wait up a cycle.
    with pytest.raises(ValueError, match=r"lasts \d+ cycles, more than controllers\[0\]'s"):
        compiled.lower(Q[0], fits + 0.5)
    with pytest.raises(ValueError, match=r"Controller\.max_program_duration"):
        compiled.lower(Q[0], k + 0.5)
    # One length for the waits of one float, and one test of their bounds.
    assert len({id(w.length) for w in waits.waits}) == len({id(w.length) for w in again.waits}) == 1
    compares = [op for op in dis.get_instructions(waits.ok) if op.opname == "COMPARE_OP"]
    assert len(compares) < 10


# ── The scheduled program ────────────────────────────────────────────────────


@pulse.kernel
def drive_in_if(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    bit = readout(r, square(400, 0.05))
    if bit:
        drive(d, square(40, 0.3))


@pulse.kernel
def repeat_until_success(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(400, 0.05))
    for _ in range(100):
        if m:
            drive(d, square(360, 0.5))
            sync(d, r)
            m = readout(r, square(400, 0.05))


def _latency_system(limit_ns):
    controller = Controller(feedback_latency_cycles=400, max_program_duration=limit_ns)
    return ControlSystem(controllers=[controller])


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
@pytest.mark.parametrize(
    ("kernel", "limit_ns", "length"),
    [(drive_in_if, 300, 840), (repeat_until_success, 40_000, 116_400)],
)
def test_a_schedule_that_waits_for_feedback_past_the_longest_program_is_an_error(
    kernel, limit_ns, length, schedule
):
    # Played op after op the programs fit (440 and 76,400 cycles of 600 and
    # 80,000); each waits 400 cycles after every readout it branches on.
    # Before, the schedule was not checked and both compiled.
    machine = _latency_system(limit_ns)
    with pytest.raises(CompilationError) as caught:
        pulse.compile(kernel, schedule=schedule, control_system=machine).lower(Q[0])
    assert str(caught.value) == (
        f"kernel '{kernel.__name__}' lasts {length} cycles, more than controllers[0]'s "
        f"max_program_duration of {limit_ns} ns ({2 * limit_ns} cycles), waiting 400 cycles "
        "of feedback latency after the readouts its operations depend on; let its operations "
        "overlap, shorten them or raise max_program_duration"
    )
    # Long enough, it compiles.
    longer = _latency_system(length / 2)
    lowered = pulse.compile(kernel, schedule=schedule, control_system=longer).lower(Q[0])
    assert max(e.start_cycles + e.duration_cycles for e in lowered.schedule) == length


def test_each_controllers_longest_program_bounds_the_lines_it_plays():
    # Nothing checks several controllers as the module is built.
    controllers = [
        Controller(drive_lines=[0], readout_lines=[0], max_program_duration=100),
        Controller(drive_lines=[1], readout_lines=[1]),
    ]
    machine = ControlSystem(controllers=controllers)
    waits = _waits(3, 100)
    assert pulse.compile(waits, control_system=machine).lower(Q[1]) is not None
    with pytest.raises(CompilationError) as caught:
        pulse.compile(waits, control_system=machine).lower(Q[0])
    assert str(caught.value) == (
        "kernel 'waits' lasts 300 cycles, more than controllers[0]'s max_program_duration "
        "of 100 ns (200 cycles); let its operations overlap, shorten them or raise "
        "max_program_duration"
    )


def _together(lines, cycles):
    @pulse.kernel
    def together(a, b, c):
        for q in (a, b, c)[:lines]:
            drive(get_drive_line(q), square(cycles, 0.1))

    return together


def test_operations_that_play_together_count_once():
    # 500 ns on the 2 GHz clock of Q's quantum system: 1000 cycles.
    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    lowered = pulse.compile(_together(3, 400), control_system=machine).lower(Q[0], Q[1], Q[2])
    assert max(e.start_cycles + e.duration_cycles for e in lowered.schedule) == 400
    # Each line alone may last as long as the program.
    assert pulse.compile(_together(3, 1000), control_system=machine).lower(Q[0], Q[1], Q[2])

    @pulse.kernel
    def in_two_parts(a, b, c):
        lines = [get_drive_line(q) for q in (a, b, c)]
        for line in lines:
            drive(line, square(600, 0.1))
        sync(*lines)
        drive(lines[0], square(404, 0.1))

    # Op after op it sums to 2204 cycles, which three lines may; scheduled
    # it lasts 1004, which the controller does not run.
    with pytest.raises(CompilationError) as caught:
        pulse.compile(in_two_parts, control_system=machine).lower(Q[0], Q[1], Q[2])
    assert str(caught.value) == (
        "kernel 'in_two_parts' lasts 1004 cycles, more than controllers[0]'s "
        "max_program_duration of 500 ns (1000 cycles); let its operations overlap, shorten "
        "them or raise max_program_duration"
    )


def test_one_line_is_checked_as_it_is_built():
    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    assert pulse.compile(_together(1, 1000), control_system=machine).lower(Q[0], Q[1], Q[2])
    with pytest.raises(CompilationError, match="the program lasts more than 1000 cycles"):
        pulse.compile(_together(1, 1004), control_system=machine).lower(Q[0], Q[1], Q[2])


def test_the_sum_stops_a_kernel_no_schedule_can_save():
    @pulse.kernel
    def runaway(a, b):
        d = get_drive_line(a)
        e = get_drive_line(b)
        for _ in range(10**15):
            drive(d, square(100, 0.1))
            drive(e, square(100, 0.1))

    machine = ControlSystem(controllers=[Controller(max_program_duration=10**6)])
    with pytest.raises(CompilationError, match="the program lasts more than 2000000 cycles"):
        pulse.compile(runaway, control_system=machine).lower(Q[0], Q[1])

    @pulse.kernel
    def in_turn(a, b, trips: int):
        d = get_drive_line(a)
        e = get_drive_line(b)
        for _ in range(trips):
            drive(d, square(100, 0.1))
            sync(d, e)
            drive(e, square(100, 0.1))
            sync(d, e)

    # Two lines that take turns: 10,000 trips sum to the longest program
    # and last it; one more trip is past it as scheduled, and twice as many
    # and one are past what any schedule of two lines lasts.
    compiled = pulse.compile(in_turn, control_system=machine, unroll_threshold=0)
    assert compiled.lower(Q[0], Q[1], 10_000) is not None
    with pytest.raises(CompilationError, match="lasts 2000200 cycles, more than controllers"):
        compiled.lower(Q[0], Q[1], 10_001)
    with pytest.raises(CompilationError, match="the program lasts more than 2000000 cycles"):
        compiled.lower(Q[0], Q[1], 20_001)


def test_lines_taken_in_a_called_kernel_count():
    @pulse.kernel
    def one(q):
        drive(get_drive_line(q), square(900, 0.1))

    @pulse.kernel
    def all_three(a, b, c):
        one(a)
        one(b)
        one(c)

    # 2700 cycles op after op on three lines, 900 as scheduled, under 1000.
    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    lowered = pulse.compile(all_three, control_system=machine).lower(Q[0], Q[1], Q[2])
    assert max(e.start_cycles + e.duration_cycles for e in lowered.schedule) == 900


def test_without_a_limit_no_line_is_counted(monkeypatch):
    # Only a builder given a limit wraps its buffer to count its lines: a
    # compile without one never asks for the wrap.
    import gc
    import weakref

    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    limited = []
    limit_cycles = PackedIRBuilder.limit_cycles

    def counted(self, cycles):
        limited.append(cycles)
        limit_cycles(self, cycles)

    monkeypatch.setattr(PackedIRBuilder, "limit_cycles", counted)
    assert pulse.compile(_together(3, 400)).lower(Q[0], Q[1], Q[2]) is not None
    no_limit = ControlSystem(controllers=[Controller()])
    assert pulse.compile(_together(3, 400), control_system=no_limit).lower(Q[0], Q[1], Q[2])
    assert limited == []
    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    assert pulse.compile(_together(3, 400), control_system=machine).lower(Q[0], Q[1], Q[2])
    assert limited == [1000]

    # The wrap holds its builder weakly: a builder under a limit is freed
    # when its last reference goes, without the cyclic collector.
    builder = PackedIRBuilder()
    assert "get_buffer" not in vars(builder)
    limit_cycles(builder, 1000)
    assert "get_buffer" in vars(builder) and len(builder.get_buffer()) == 2
    gone = weakref.ref(builder)
    wrap = weakref.ref(vars(builder)["get_buffer"])
    gc.disable()
    try:
        del builder
        # Nor do the wrap's functions hold each other.
        assert gone() is None and wrap() is None
    finally:
        gc.enable()


# ── Waits whose length is an answer, known only when the program runs ──────


@pulse.classical(max_latency_ns=50, name="how_long")
def _how_long(bits: pulse.Bits) -> int:
    return 900


def _answered(shape):
    """A readout, a call on it, then on the drive line of a (b's line is
    taken and unused) a wait of 4 to 900 cycles set by the call's answer
    and a 600-cycle drive: once, three times in a rolled loop, or with the
    drive in an if arm beside a 300-cycle one."""

    @pulse.kernel
    def answered(a, b):
        d = get_drive_line(a)
        get_drive_line(b)
        r = get_readout_line(a)
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        sync(d, r)
        if shape == "plain":
            wait(d, n, bounds=(4, 900))
            drive(d, square(600, 0.1))
        elif shape == "loop":
            for _ in range(3):
                wait(d, n, bounds=(4, 900))
                drive(d, square(600, 0.1))
        else:
            wait(d, n, bounds=(4, 900))
            if m:
                drive(d, square(600, 0.1))
            else:
                drive(d, square(300, 0.1))

    return answered


def _answering(cycles=None):
    from cudaq_pulse.classical import Link

    limit = None if cycles is None else cycles / 2
    controller = Controller(max_program_duration=limit)
    return ControlSystem(controllers=[controller], links={"controller": Link()})


@pytest.mark.parametrize(
    ("shape", "options", "stretch"),
    [("plain", {}, 896), ("loop", {"unroll_threshold": 0}, 3 * 896), ("before an if", {}, 896)],
)
def test_a_wait_set_by_an_answer_counts_at_its_highest_length(shape, options, stretch):
    kernel = _answered(shape)
    free = pulse.compile(kernel, control_system=_answering(), **options).lower(Q[0], Q[1])
    scheduled = max(e.start_cycles + e.duration_cycles for e in free.schedule)
    # Exactly what it can last compiles, and one cycle less (half a ns on
    # this 2 GHz clock) does not: the schedule holds the wait at its lowest
    # length, and the check adds what each execution of it can add.
    longest = scheduled + stretch
    fits = pulse.compile(kernel, control_system=_answering(longest), **options)
    assert fits.lower(Q[0], Q[1]) is not None
    with pytest.raises(CompilationError) as caught:
        pulse.compile(kernel, control_system=_answering(longest - 1), **options).lower(Q[0], Q[1])
    assert (
        f"kernel 'answered' lasts up to {longest} cycles ({scheduled} as scheduled, the waits "
        f"of run-time length that hold it back add up to {stretch}), more than controllers[0]'s "
        "max_program_duration"
    ) in str(caught.value)


def test_a_wait_set_by_an_answer_is_in_no_arm_of_an_if():
    # pulse-verify refuses such a wait inside an arm, with or without a
    # limit, so the stretch counts no arm against another.
    @pulse.kernel
    def in_an_arm(a):
        d = get_drive_line(a)
        r = get_readout_line(a)
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        sync(d, r)
        if m:
            wait(d, n, bounds=(4, 900))
        else:
            drive(d, square(2000, 0.1))

    for cycles in (None, 100_000):
        with pytest.raises(RuntimeError, match="a variable wait cannot be inside one arm") as said:
            pulse.compile(in_an_arm, control_system=_answering(cycles)).lower(Q[0])
        # The error of the pass itself, as without a limit.
        assert type(said.value) is RuntimeError

    # Without that pass (passes=()) the wait is scheduled, the arms one
    # after the other, and counts as if its arm always ran: more than the
    # program can last, never less.
    free = pulse.compile(in_an_arm, control_system=_answering(), passes=()).lower(Q[0])
    scheduled = max(e.start_cycles + e.duration_cycles for e in free.schedule)
    assert scheduled == 201 + 4 + 2000
    fits = pulse.compile(in_an_arm, control_system=_answering(scheduled + 896), passes=())
    assert fits.lower(Q[0]) is not None
    with pytest.raises(CompilationError, match=f"lasts up to {scheduled + 896} cycles"):
        pulse.compile(in_an_arm, control_system=_answering(scheduled + 895), passes=()).lower(Q[0])


def test_one_line_with_such_a_wait_is_refused_as_before():
    @pulse.kernel
    def one_line(a):
        r = get_readout_line(a)
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        wait(r, n, bounds=(4, 900))
        readout(r, square(600, 0.05))

    free = pulse.compile(one_line, control_system=_answering()).lower(Q[0])
    scheduled = max(e.start_cycles + e.duration_cycles for e in free.schedule)
    assert pulse.compile(one_line, control_system=_answering(scheduled + 896)).lower(Q[0])
    with pytest.raises(CompilationError, match="max_program_duration"):
        pulse.compile(one_line, control_system=_answering(scheduled + 895)).lower(Q[0])


# ── Such a wait holds back the lines tied to it, and no other ──────────────


LOWEST = pytest.mark.parametrize("lowest", [4, 0])


def _apart(long, shape="plain", lowest=4):
    """Line e (b's drive line) plays one drive of *long* cycles that nothing
    ties to the rest. On a's lines: a readout, a call on it, then a wait of
    *lowest* to 100 cycles set by the call's answer and a 100-cycle drive:
    once, or six times in two rolled loops, one in the other. The answer is
    there at 201, where a wait of 0 cycles also ends."""

    @pulse.kernel
    def apart(a, b):
        d = get_drive_line(a)
        r = get_readout_line(a)
        e = get_drive_line(b)
        drive(e, square(long, 0.1))
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        sync(d, r)
        if shape == "plain":
            wait(d, n, bounds=(lowest, 100))
            drive(d, square(100, 0.1))
        else:
            for _ in range(2):
                for _ in range(3):
                    wait(d, n, bounds=(lowest, 100))
                    drive(d, square(100, 0.1))

    return apart


def _end(lowered, qudit=None):
    """The end of the last scheduled event, or of the last on *qudit*'s lines."""
    events = [e for e in lowered.schedule if qudit is None or e.qudit == qudit]
    return max(e.start_cycles + e.duration_cycles for e in events)


def _limits(kernel, machine, longest, said, **options):
    """*kernel* compiles on machine(longest) and, one cycle less, is the
    error that has *said*."""
    assert pulse.compile(kernel, control_system=machine(longest), **options).lower(Q[0], Q[1])
    with pytest.raises(CompilationError) as caught:
        pulse.compile(kernel, control_system=machine(longest - 1), **options).lower(Q[0], Q[1])
    assert said in str(caught.value)


@LOWEST
def test_a_wait_set_by_an_answer_holds_back_only_the_lines_tied_to_it(lowest):
    # Line e ends at 950 whatever the answer is, and a's lines at 301 + the
    # wait as scheduled, 100 less the wait later at most: the program never
    # lasts more than 950.
    kernel = _apart(950, lowest=lowest)
    free = pulse.compile(kernel, control_system=_answering()).lower(Q[0], Q[1])
    assert (_end(free, 0), _end(free, 1)) == (301 + lowest, 950)
    assert pulse.compile(kernel, control_system=_answering(1000)).lower(Q[0], Q[1])
    _limits(kernel, _answering, 950, "kernel 'apart' lasts 950 cycles, more than controllers[0]")
    # With a shorter drive on e, a's lines last longest: with their wait.
    _limits(
        _apart(300, lowest=lowest),
        _answering,
        401,
        f"kernel 'apart' lasts up to 401 cycles ({301 + lowest} as scheduled, the waits of "
        f"run-time length that hold it back add up to {100 - lowest}), more than controllers[0]",
    )


def _two_controllers(limited):
    """Controller 0 plays a's lines and controller 1 b's; the one *limited*
    has the limit."""
    from cudaq_pulse.classical import Link

    def machine(cycles):
        limits = [None, None]
        limits[limited] = cycles / 2
        controllers = [
            Controller(drive_lines=[q], readout_lines=[q], max_program_duration=limits[q])
            for q in (0, 1)
        ]
        return ControlSystem(controllers=controllers, links={"controller": Link()})

    return machine


@LOWEST
def test_a_wait_on_another_controllers_line_adds_nothing_to_a_controller(lowest):
    # The limit is controller 1's, which plays line e alone; the wait is on
    # controller 0's line.
    kernel = _apart(950, lowest=lowest)
    assert pulse.compile(kernel, control_system=_two_controllers(1)(1000)).lower(Q[0], Q[1])
    _limits(
        kernel,
        _two_controllers(1),
        950,
        "kernel 'apart' lasts 950 cycles, more than controllers[1]",
    )
    # Controller 0's lines last 301 + the wait as scheduled, and what the
    # wait can add more.
    _limits(
        kernel,
        _two_controllers(0),
        401,
        f"kernel 'apart' lasts up to 401 cycles ({301 + lowest} as scheduled, the waits of "
        f"run-time length that hold it back add up to {100 - lowest}), more than controllers[0]",
    )


@pytest.mark.parametrize("bounds", [(4, 100), (0, 100), (0, 5000)])
def test_a_wait_adds_nothing_to_a_controller_that_only_counts_the_call_before_it(bounds):
    # Controller 1 plays line e, 150 cycles, and counts the call, whose
    # answer is there at 201: it is done by then whatever the answer is,
    # also where the wait, 0 cycles as scheduled, starts and ends at 201.
    @pulse.kernel
    def apart(a, b):
        d = get_drive_line(a)
        r = get_readout_line(a)
        e = get_drive_line(b)
        drive(e, square(150, 0.1))
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        sync(d, r)
        wait(d, n, bounds=bounds)
        drive(d, square(100, 0.1))

    assert pulse.compile(apart, control_system=_two_controllers(1)(1000)).lower(Q[0], Q[1])
    _limits(
        apart,
        _two_controllers(1),
        201,
        "kernel 'apart' lasts 201 cycles, more than controllers[1]",
    )


def test_what_follows_a_wait_of_no_cycles_counts_with_it():
    # Two waits that are 0 cycles as scheduled, one after the other at 201,
    # the last operations of the line: the second ends where the first
    # starts and comes after it, so both count.
    @pulse.kernel
    def in_turn(a, b):
        d = get_drive_line(a)
        r = get_readout_line(a)
        drive(get_drive_line(b), square(150, 0.1))
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        sync(d, r)
        wait(d, n, bounds=(0, 100))
        wait(d, n, bounds=(0, 200))

    for machine in (_answering, _two_controllers(0)):
        _limits(
            in_turn,
            machine,
            501,
            "kernel 'in_turn' lasts up to 501 cycles (201 as scheduled, the waits of run-time "
            "length that hold it back add up to 300), more than controllers[0]",
        )


@LOWEST
def test_such_waits_in_rolled_loops_count_once_for_each_time_they_run(lowest):
    kernel = _apart(300, "loops", lowest)
    options = {"unroll_threshold": 0}
    free = pulse.compile(kernel, control_system=_answering(), **options).lower(Q[0], Q[1])
    scheduled = _end(free, 0)
    assert scheduled > 300
    added = 6 * (100 - lowest)
    _limits(
        kernel,
        _answering,
        scheduled + added,
        f"kernel 'apart' lasts up to {scheduled + added} cycles ({scheduled} as scheduled, the "
        f"waits of run-time length that hold it back add up to {added}), more than controllers[0]",
        **options,
    )
    # Line e, were it the longest, would count without them.
    _limits(
        _apart(5000, "loops", lowest),
        _answering,
        5000,
        "kernel 'apart' lasts 5000 cycles, more than controllers[0]",
        **options,
    )


def _two_waits(lowest):
    @pulse.kernel
    def two_waits(a, b):
        # On each qudit's lines a readout, a call on it and a wait set by
        # its answer, of *lowest* to 100 cycles for a and to 300 for b, then
        # a drive.
        for q, highest, cycles in ((a, 100, 250), (b, 300, 100)):
            d = get_drive_line(q)
            r = get_readout_line(q)
            m = readout(r, square(100, 0.05))
            n = _how_long.submit([m]).result()
            sync(d, r)
            wait(d, n, bounds=(lowest, highest))
            drive(d, square(cycles, 0.1))

    return two_waits


def _linked(**kwargs):
    from cudaq_pulse.classical import Link

    link = Link(**kwargs.pop("link", {}))

    def machine(cycles=None):
        limit = None if cycles is None else cycles / 2
        controller = Controller(max_program_duration=limit, **kwargs)
        return ControlSystem(controllers=[controller], links={"controller": link})

    return machine


@LOWEST
def test_two_such_waits_that_nothing_ties_each_hold_back_their_own_lines(lowest):
    machine = _linked()
    kernel = _two_waits(lowest)
    free = pulse.compile(kernel, control_system=machine()).lower(Q[0], Q[1])
    a, b = _end(free, 0), _end(free, 1)
    # a's lines end last as scheduled, and b's can end last when it runs.
    more = (100 - lowest, 300 - lowest)
    assert a > b and b + more[1] > a + more[0]
    _limits(
        kernel,
        machine,
        b + more[1],
        f"kernel 'two_waits' lasts up to {b + more[1]} cycles ({b} as scheduled, the waits of "
        f"run-time length that hold it back add up to {more[1]}), more than controllers[0]",
    )


@LOWEST
@pytest.mark.parametrize(
    "tied", [{"link": {"slots": 1}}, {"blocks_on_calls": True}], ids=["one slot", "blocks on calls"]
)
def test_two_such_waits_add_up_where_the_schedule_ties_their_lines(tied, lowest):
    # A link with one slot for the two calls, or a controller that plays
    # nothing while a call is out: each wait can hold back every line.
    machine = _linked(**tied)
    kernel = _two_waits(lowest)
    free = pulse.compile(kernel, control_system=machine()).lower(Q[0], Q[1])
    scheduled = _end(free)
    added = 400 - 2 * lowest
    _limits(
        kernel,
        machine,
        scheduled + added,
        f"kernel 'two_waits' lasts up to {scheduled + added} cycles ({scheduled} as scheduled, "
        f"the waits of run-time length that hold it back add up to {added}), more than "
        "controllers[0]",
    )


def test_the_lowest_of_two_limits_on_a_builder_holds():
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    stops = []
    for first, second in ((1000, 400), (400, 1000)):
        builder = PackedIRBuilder()
        builder.limit_cycles(first)
        builder.limit_cycles(second)
        buffer = builder.get_buffer()
        stops.append(list(buffer)[-1])
    # The last record's sum is the lowest limit's, on one line here.
    assert stops == [400, 400]


@pytest.mark.parametrize("other", [400, 3000], ids=["lower", "higher"])
@pytest.mark.parametrize("order", ["before", "after"])
def test_a_second_limit_keeps_the_module_under_the_lowest(monkeypatch, other, order):
    # The module built from a builder limited twice knows it is under a
    # controller's limit, the lowest, whichever of the two came first.
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    limit_cycles = PackedIRBuilder.limit_cycles

    def twice(self, cycles):
        for limit in (other, cycles) if order == "before" else (cycles, other):
            limit_cycles(self, limit)

    monkeypatch.setattr(PackedIRBuilder, "limit_cycles", twice)

    @pulse.kernel
    def one_float(q, tau: float):
        d = get_drive_line(q)
        wait(d, tau)
        drive(d, square(8, 0.1))

    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    waits = pulse.compile(one_float, control_system=machine).lower(Q[0], 4.0).generic.waits
    lowest = min(other, 1000)
    assert waits.length[1:3] == (lowest, 1)
    with pytest.raises(ValueError, match=r"Controller\.max_program_duration"):
        waits.check([float(lowest)])


# ── Such a wait with no highest length has nothing to hold to the limit ────

NO_HIGHEST = (
    r"kernel 'unbounded' runs under controllers\[0\]'s max_program_duration of 50000 ns, but "
    r"this wait in it has no highest length; bound it with wait\(\.\.\., bounds=\(lowest, "
    r"highest\)\), or drop max_program_duration"
)


def _unbounded(bounds):
    """_apart(950) with its wait bounded by *bounds* (None: not at all)."""
    options = {} if bounds is None else {"bounds": bounds}

    @pulse.kernel
    def unbounded(a, b):
        d = get_drive_line(a)
        r = get_readout_line(a)
        e = get_drive_line(b)
        drive(e, square(950, 0.1))
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        sync(d, r)
        wait(d, n, **options)
        drive(d, square(100, 0.1))

    return unbounded


@pytest.mark.parametrize("bounds", [(4, None), None], ids=["no highest", "no bounds"])
def test_a_wait_set_by_an_answer_with_no_highest_length_is_refused_under_a_limit(bounds):
    kernel = _unbounded(bounds)
    # Without a limit it compiles, as it does bounded under one.
    assert pulse.compile(kernel, control_system=_answering()).lower(Q[0], Q[1])
    assert pulse.compile(_unbounded((4, 100)), control_system=_answering(100_000)).lower(Q[0], Q[1])
    with pytest.raises(CompilationError, match=NO_HIGHEST):
        pulse.compile(kernel, control_system=_answering(100_000)).lower(Q[0], Q[1])
    # On several controllers it is refused for the one whose lines it holds
    # back.
    with pytest.raises(CompilationError, match=NO_HIGHEST):
        pulse.compile(kernel, control_system=_two_controllers(0)(100_000)).lower(Q[0], Q[1])


@pytest.mark.parametrize("bounds", [(4, None), (0, None), None])
def test_such_a_wait_is_not_refused_for_a_controller_it_does_not_hold_back(bounds):
    # Controller 1 plays line e, which nothing ties to the wait, and counts
    # the call, whose answer is there at 201, before the wait: also a wait
    # that may last 0 cycles, which starts and ends with that answer.
    kernel = _unbounded(bounds)
    assert pulse.compile(kernel, control_system=_two_controllers(1)(100_000)).lower(Q[0], Q[1])


def test_such_a_wait_under_a_limit_is_an_error_of_the_kernel_at_a_call_too():
    # With a float among its parameters the kernel is scheduled again at
    # each call: no values make the wait bounded.
    @pulse.kernel
    def unbounded(a, b, tau: float):
        d = get_drive_line(a)
        r = get_readout_line(a)
        wait(get_drive_line(b), tau)
        m = readout(r, square(100, 0.05))
        n = _how_long.submit([m]).result()
        sync(d, r)
        wait(d, n, bounds=(4, None))
        drive(d, square(100, 0.1))

    with pytest.raises(CompilationError, match=NO_HIGHEST) as said:
        pulse.compile(unbounded, control_system=_answering(100_000)).lower(Q[0], Q[1], 8.0)
    assert not isinstance(said.value, ValueError)


# ── Nor has the answer of a blocking classical function a latest time ──────


@pulse.classical(max_latency_ns=50, latency_mode="blocking", name="blocking_bit")
def _blocking_bit(bits: pulse.Bits) -> pulse.Bits:
    return [0]


def _blocked(used, after=False):
    """Line e (b's drive line) plays 150 cycles that nothing ties to the
    rest. On a's lines: a readout, a blocking call on it, and a drive that
    the answer decides (*used*) or that follows the call without using it.
    With *after*, line e plays again at the end of the kernel."""

    @pulse.kernel
    def blocked(a, b):
        d = get_drive_line(a)
        r = get_readout_line(a)
        e = get_drive_line(b)
        drive(e, square(150, 0.1))
        m = readout(r, square(100, 0.05))
        answer = _blocking_bit.submit([m]).result()
        sync(d, r)
        if used:
            if answer[0]:
                drive(d, square(100, 0.1))
        else:
            drive(d, square(100, 0.1))
        if after:
            drive(e, square(150, 0.1))

    return blocked


BLOCKING = (
    r"kernel 'blocked' runs under controllers\[{}\]'s max_program_duration of 50000 ns, but it "
    r"waits for an answer of a blocking classical function, which has no bound; use "
    r'latency_mode="bounded" or "timeout", or drop max_program_duration'
)


@pytest.mark.parametrize("used", [True, False])
def test_a_blocking_call_is_refused_under_a_limit(used):
    kernel = _blocked(used)
    assert pulse.compile(kernel, control_system=_answering()).lower(Q[0], Q[1])
    with pytest.raises(CompilationError, match=BLOCKING.format(0)):
        pulse.compile(kernel, control_system=_answering(100_000)).lower(Q[0], Q[1])
    # On several controllers, for the one whose line follows the answer.
    with pytest.raises(CompilationError, match=BLOCKING.format(0)):
        pulse.compile(kernel, control_system=_two_controllers(0)(100_000)).lower(Q[0], Q[1])


@pytest.mark.parametrize(("used", "after"), [(True, False), (True, True), (False, False)])
def test_a_blocking_call_is_not_refused_for_a_controller_with_nothing_after_it(used, after):
    # Controller 1 plays line e and counts the call itself, as scheduled.
    # Where the answer is used, nothing ties line e to it, whenever e plays;
    # where nothing uses it the call holds every line, and e is done.
    kernel = _blocked(used, after)
    assert pulse.compile(kernel, control_system=_two_controllers(1)(100_000)).lower(Q[0], Q[1])


def test_a_blocking_call_whose_answer_nothing_uses_holds_every_line():
    # So line e, which plays again after it, follows it on controller 1.
    with pytest.raises(CompilationError, match=BLOCKING.format(1)):
        pulse.compile(_blocked(False, True), control_system=_two_controllers(1)(100_000)).lower(
            Q[0], Q[1]
        )
