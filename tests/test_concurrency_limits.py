# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``max_concurrent_drives`` and ``max_concurrent_readouts``, when a control
system with controllers is given them, hold under every schedule: ``rcp``
and ``alap_rcp`` keep to them, and a program another schedule places past
them fails to compile, naming the limit, the count and the first cycle."""

from __future__ import annotations

import pickle
import random

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import controller
from cudaq_pulse.controller import ControllerLimitError, DriveLine, ReadoutLine

_RCP = ("rcp", "alap_rcp")


@pytest.fixture(autouse=True)
def _checked_as_placed(monkeypatch):
    """These tests are of the check: asap and alap keep the schedule they
    place, which compile would otherwise place again as rcp does past a
    limit (test_placed_by_limits.py)."""
    from cudaq_pulse import compiler

    monkeypatch.setattr(compiler, "_PLACED_BY_LIMITS", {})


@pulse.kernel
def drives_then_readouts(qs: list):
    for q in qs:
        drive(get_drive_line(q), gaussian(80, 0.5, 20))
    for q in qs:
        readout(get_readout_line(q), square(400, 0.1))


@pulse.kernel
def drives_only(qs: list):
    for q in qs:
        drive(get_drive_line(q), gaussian(80, 0.5, 20))


@pulse.kernel
def back_to_back(q, r, s):
    a = get_drive_line(q)
    b = get_drive_line(r)
    c = get_drive_line(s)
    drive(a, square(40, 0.1))
    drive(b, square(40, 0.1))
    sync(a, b, c)
    drive(c, square(40, 0.1))


@pulse.kernel
def arms(q, r, s, t, u):
    m = readout(get_readout_line(q), square(100, 0.05))
    if m:
        drive(get_drive_line(r), square(40, 0.1))
        drive(get_drive_line(s), square(40, 0.1))
    else:
        drive(get_drive_line(t), square(40, 0.1))
        drive(get_drive_line(u), square(40, 0.1))


@pulse.kernel
def arms_and_more(q, r, s, t, u):
    ro = get_readout_line(q)
    d = get_drive_line(u)
    m = readout(ro, square(100, 0.05))
    sync(ro, d)
    if m:
        drive(get_drive_line(r), square(40, 0.1))
        drive(get_drive_line(s), square(40, 0.1))
    else:
        drive(get_drive_line(t), square(40, 0.1))
    drive(d, square(40, 0.1))


@pulse.kernel
def loop_in_an_arm(q, r, s, t):
    m = readout(get_readout_line(q), square(100, 0.05))
    if m:
        a = get_drive_line(r)
        b = get_drive_line(s)
        for _ in range(3):
            drive(a, gaussian(40, 0.1, 10))
            drive(b, gaussian(40, 0.1, 10))
            sync(a, b)
    else:
        drive(get_drive_line(t), square(200, 0.1))


@pulse.kernel
def branch_in_a_loop(q, r, s, t):
    m = readout(get_readout_line(q), square(100, 0.05))
    for _ in range(3):
        if m:
            a = get_drive_line(r)
            b = get_drive_line(s)
            for _ in range(3):
                drive(a, gaussian(40, 0.1, 10))
                drive(b, gaussian(40, 0.1, 10))
                sync(a, b)
        else:
            drive(get_drive_line(t), square(200, 0.1))


@pulse.kernel
def after_a_wait(q, r, gap: float):
    a = get_drive_line(q)
    b = get_drive_line(r)
    wait(a, gap)
    drive(a, square(40, 0.1))
    drive(b, square(40, 0.1))


@pulse.kernel
def in_a_loop(qs: list):
    for _ in range(3):
        for q in qs:
            drive(get_drive_line(q), square(40, 0.1))
        readout(get_readout_line(qs[0]), square(60, 0.05))


def _system(controllers=None, **limits):
    if controllers is None:
        controllers = [pulse.Controller(drive_lines=range(17), readout_lines=range(17))]
    control = pulse.ControlSystem(controllers=controllers, **limits)
    return pulse.quantum_system("transmon_surface17", control_system=control)


def _lower(kernel, system, schedule, *args):
    return pulse.compile(kernel, quantum_system=system, schedule=schedule).lower(*args)


def test_asap_past_the_drive_limit_fails_naming_the_limit_count_and_cycle():
    system = _system(max_concurrent_drives=6)
    with pytest.raises(ControllerLimitError) as caught:
        _lower(drives_only, system, "asap", system.qudits[:13])
    assert caught.value.violations == (
        "13 drives play at once from cycle 0; the control system plays at most 6 "
        '(max_concurrent_drives): compile with schedule="rcp" or "alap_rcp", which keep to it',
    )
    # Before this check, the same program compiled and controller.check found nothing.
    free = _system()
    unchecked = _lower(drives_only, free, "asap", free.qudits[:13])
    assert controller.check(unchecked) == []


def test_alap_past_the_limits_fails_at_the_cycle_they_are_passed():
    system = _system(max_concurrent_drives=6, max_concurrent_readouts=4)
    with pytest.raises(ControllerLimitError) as caught:
        _lower(drives_then_readouts, system, "alap", system.qudits[:13])
    assert caught.value.violations == (
        "13 drives play at once from cycle 320; the control system plays at most 6 "
        '(max_concurrent_drives): compile with schedule="rcp" or "alap_rcp", which keep to it',
        "13 readouts play at once from cycle 0; the control system plays at most 4 "
        '(max_concurrent_readouts): compile with schedule="rcp" or "alap_rcp", which keep to it',
    )


def test_readouts_past_their_limit_fail_alone():
    system = _system(max_concurrent_readouts=4)
    with pytest.raises(ControllerLimitError) as caught:
        _lower(drives_then_readouts, system, "asap", system.qudits[:5])
    assert caught.value.violations == (
        "5 readouts play at once from cycle 0; the control system plays at most 4 "
        '(max_concurrent_readouts): compile with schedule="rcp" or "alap_rcp", which keep to it',
    )


def test_the_count_names_the_most_at_once_when_it_grows():
    system = _system(max_concurrent_drives=2)

    @pulse.kernel
    def growing(q, r, s, t):
        drive(get_drive_line(q), square(400, 0.1))
        drive(get_drive_line(r), square(400, 0.1))
        drive(get_drive_line(s), square(400, 0.1))
        a = get_drive_line(t)
        wait(a, 100)
        drive(a, square(40, 0.1))

    with pytest.raises(
        ControllerLimitError, match="3 drives play at once from cycle 0, and up to 4;"
    ):
        _lower(growing, system, "asap", *system.qudits[:4])


@pytest.mark.parametrize("schedule", _RCP)
def test_the_rcp_schedules_always_fit(schedule):
    for drives, readouts, count in ((6, 4, 13), (1, 1, 5), (2, 3, 8), (17, 8, 17)):
        system = _system(max_concurrent_drives=drives, max_concurrent_readouts=readouts)
        for kernel in (drives_then_readouts, in_a_loop):
            lowered = _lower(kernel, system, schedule, system.qudits[:count])
            assert controller.check(lowered) == []
    system = _system(max_concurrent_drives=1, max_concurrent_readouts=1)
    for kernel in (arms, arms_and_more):
        assert controller.check(_lower(kernel, system, schedule, *system.qudits[:5])) == []


@pytest.mark.parametrize("schedule", _RCP)
@pytest.mark.parametrize("unroll_threshold", [None, 0, 1])
def test_the_rcp_schedules_fit_with_loops_in_branch_arms(schedule, unroll_threshold):
    # The loop's pulses and the other arm's are in one execution of the
    # branch, in different arms, whatever the loop's iteration.
    for drives in (1, 2):
        system = _system(max_concurrent_drives=drives)
        for kernel in (loop_in_an_arm, branch_in_a_loop):
            compiled = pulse.compile(
                kernel, quantum_system=system, schedule=schedule, unroll_threshold=unroll_threshold
            )
            assert controller.check(compiled.lower(*system.qudits[:4])) == []


def test_a_loop_in_an_arm_counts_with_its_own_arm_only():
    system = _system(max_concurrent_drives=2)
    _lower(loop_in_an_arm, system, "asap", *system.qudits[:4])
    with pytest.raises(ControllerLimitError, match="2 drives play at once from cycle 100;"):
        system = _system(max_concurrent_drives=1)
        _lower(loop_in_an_arm, system, "asap", *system.qudits[:4])


@pytest.mark.parametrize("schedule", _RCP)
def test_the_rcp_schedules_fit_with_line_delays(schedule):
    # A delay moves a line's pulses after they are scheduled; the limits
    # count them where the schedule placed them, as rcp does.
    lines = [DriveLine(q, delay_ns=7.0 * (q % 3)) for q in range(6)]
    readouts = [ReadoutLine(q, delay_ns=3.0 * (q % 2)) for q in range(6)]
    system = _system(
        [pulse.Controller(drive_lines=lines, readout_lines=readouts)],
        max_concurrent_drives=2,
        max_concurrent_readouts=1,
    )
    lowered = _lower(drives_then_readouts, system, schedule, system.qudits[:6])
    assert controller.check(lowered) == []


@pytest.mark.parametrize("schedule", _RCP)
def test_the_rcp_schedules_fit_on_several_controllers(schedule):
    controllers = [
        pulse.Controller(drive_lines=range(0, 8), readout_lines=range(0, 8)),
        pulse.Controller(drive_lines=range(8, 17), readout_lines=range(8, 17)),
    ]
    system = _system(controllers, max_concurrent_drives=3, max_concurrent_readouts=2)
    _lower(drives_then_readouts, system, schedule, system.qudits[:12])
    with pytest.raises(ControllerLimitError, match="12 drives play at once from cycle 0;"):
        _lower(drives_then_readouts, system, "asap", system.qudits[:12])


def test_exactly_at_the_limit_fits():
    system = _system(max_concurrent_drives=6, max_concurrent_readouts=6)
    _lower(drives_then_readouts, system, "asap", system.qudits[:6])
    # A pulse ending at a cycle frees its place for one starting then.
    system = _system(max_concurrent_drives=2)
    _lower(back_to_back, system, "asap", *system.qudits[:3])


def test_pulses_in_different_arms_of_a_branch_never_play_together():
    system = _system(max_concurrent_drives=2)
    # Two drives in each arm, at the same cycles: two at once on either path.
    _lower(arms, system, "asap", *system.qudits[:5])
    # Then one more beside them is three on the path through the first arm.
    with pytest.raises(ControllerLimitError, match="3 drives play at once from cycle 100;"):
        _lower(arms_and_more, system, "asap", *system.qudits[:5])
    system = _system(max_concurrent_drives=3)
    _lower(arms_and_more, system, "asap", *system.qudits[:5])


def test_without_limits_given_nothing_is_checked():
    # The defaults are only the rcp schedules' lanes.
    system = _system()
    assert system.control_system.max_concurrent_drives is None
    lowered = _lower(drives_then_readouts, system, "asap", system.qudits[:13])
    assert controller.check(lowered) == []
    # Nor is a control system without controllers, whatever it is given.
    bare = pulse.ControlSystem(max_concurrent_drives=2, max_concurrent_readouts=1)
    system = pulse.quantum_system("transmon_surface17", control_system=bare)
    _lower(drives_then_readouts, system, "asap", system.qudits[:13])
    # A limit given checks that kind only.
    system = _system(max_concurrent_readouts=13)
    _lower(drives_then_readouts, system, "asap", system.qudits[:13])


def test_a_call_whose_values_place_pulses_past_a_limit_fails_when_made():
    system = _system(max_concurrent_drives=1)
    compiled = pulse.compile(after_a_wait, quantum_system=system, schedule="asap")
    q, r = system.qudits[:2]
    compiled.lower(q, r, 40.0)
    with pytest.raises(ControllerLimitError, match="2 drives play at once from cycle 20;"):
        compiled.lower(q, r, 20.0)


def test_a_limit_not_given_is_none_and_stays_none_in_copies_and_files():
    controllers = [pulse.Controller(drive_lines=[0], readout_lines=[0])]
    given = pulse.ControlSystem(controllers=controllers, max_concurrent_drives=4)
    default = pulse.ControlSystem(controllers=controllers)
    assert (given.max_concurrent_drives, default.max_concurrent_drives) == (4, None)
    assert given != default and repr(given) != repr(default)
    assert given._machine != default._machine
    assert given.as_dict()["max_concurrent_drives"] == 4
    assert default.as_dict()["max_concurrent_drives"] is None
    for control in (given, default):
        assert pickle.loads(pickle.dumps(control)) == control
        system = pulse.quantum_system("transmon_surface17", control_system=control)
        assert pulse.QuantumSystem.from_dict(system.as_dict()).control_system == control


def _drifting(trips, tail=None):
    # Loops of periods 100 and 101 cycles: their pulses first meet at 4800.
    # With *tail*, a third line plays a long pulse from that cycle.
    @pulse.kernel
    def drifting(q, r, s):
        a = get_drive_line(q)
        b = get_drive_line(r)
        wait(b, 50)
        for _ in range(trips):
            drive(a, square(4, 0.1))
            wait(a, 96)
        for _ in range(trips):
            drive(b, square(4, 0.1))
            wait(b, 97)
        if tail is not None:
            c = get_drive_line(s)
            wait(c, tail)
            drive(c, square(400, 0.1))

    return drifting


def test_loops_of_different_periods_are_counted_copy_by_copy_up_to_a_bound():
    system = _system(max_concurrent_drives=1)
    qudits = system.qudits[:3]
    with pytest.raises(ControllerLimitError, match="2 drives play at once from cycle 4800;"):
        _lower(_drifting(100), system, "asap", *qudits)
    # A pulse where the loops are, later: still the first cycle, from every copy.
    with pytest.raises(ControllerLimitError, match="2 drives play at once from cycle 4800;"):
        _lower(_drifting(100, 9800), system, "asap", *qudits)
    # Far more copies than it plays one by one: a count found near the pulse
    # is named at its cycle, not as the first ...
    with pytest.raises(ControllerLimitError) as caught:
        _lower(_drifting(10**15, 9800), system, "asap", *qudits)
    assert caught.value.violations[0].startswith(
        "2 drives play at once at cycle 9800 (rolled loops of different periods, too many "
        "copies to count one by one for the first such cycle);"
    )
    # ... and with none found there, an error rather than a walk of every copy.
    with pytest.raises(pulse.UnsupportedError, match="more than 65536 copies of rolled loops"):
        _lower(_drifting(10**15), system, "asap", *qudits)
    system = _system(max_concurrent_drives=2)
    _lower(_drifting(10**15), system, "asap", *system.qudits[:3])


def _counted_plainly(limit, played):
    """What ``_AtOnce.sweep`` finds, counted the plain way: at each pulse's
    start, that pulse and every one before it that has not ended, in the
    order they started."""
    from cudaq_pulse._concurrency import _together

    first, at_first, most = None, 0, 0
    for k, (start, _, _, _) in enumerate(played):
        flying = [p for p in played[:k] if p[1] > start] + [played[k]]
        if len(flying) <= limit:
            continue
        count = _together(
            [({(c["branch"], ()): c["arm"] for c in p[2]["conditions"]}, 1) for p in flying]
        )
        if count <= limit:
            continue
        if first is None or start < first:
            first, at_first = start, count
        elif start == first:
            at_first = max(at_first, count)
        most = max(most, count)
    return first, at_first, most


@pytest.mark.parametrize("seed", range(40))
def test_the_sweep_counts_what_counting_every_pulse_in_flight_does(seed):
    # Starts tie, a pulse ends at the cycle another starts, pulses of no
    # length, and pulses in the arms of up to three branches.
    from cudaq_pulse._concurrency import _AtOnce, _first

    rng = random.Random(seed)
    for _ in range(60):
        played = []
        for index in range(rng.randrange(1, 40)):
            start = rng.randrange(0, 24) * 4
            conditions = [
                {"branch": b, "arm": rng.choice(("then", "else"))}
                for b in range(3)
                if rng.random() < 0.3
            ]
            event = {"index": index, "conditions": conditions}
            played.append((start, start + rng.choice((0, 4, 4, 8, 20, 60)), event, ()))
        played.sort(key=_first)
        for limit in (1, 2, 5, 12):
            count = _AtOnce(limit, [], {})
            count.sweep(played)
            assert (count.first, count.at_first, count.most) == _counted_plainly(limit, played)


def _bounded_plainly(count, items, floor, weights):
    """What ``_AtOnce._bound`` gives, the plain way: the entries by start,
    those in flight kept in a list made again at every entry."""
    from cudaq_pulse._concurrency import _together

    entries = sorted(
        (
            (*count.timed[i], count._arms(count.events[i]["conditions"], ()), weights[i])
            for i, _, _ in items
        ),
        key=lambda e: (e[0], e[1]),
    )
    most = 0
    active = []
    for lo, hi, arms, weight in entries:
        active = [a for a in active if a[0] > lo]
        active.append((hi, arms, weight))
        if sum(a[2] for a in active) <= max(most, floor):
            continue
        most = max(most, _together([(a[1], a[2]) for a in active]))
    return most


@pytest.mark.parametrize("seed", range(40))
def test_the_bound_is_what_listing_every_pulse_in_flight_gives(seed):
    # Starts tie, a pulse ends at the cycle another starts, pulses of no
    # length, pulses in the arms of up to three branches or in none, and
    # entries that stand for several pulses (a loop's weight).
    from cudaq_pulse._concurrency import _AtOnce

    class _Weighted(_AtOnce):
        def _weight(self, node):
            return self.weights[node]

    rng = random.Random(1000 + seed)
    for _ in range(60):
        events, timed, weights = [], {}, {}
        branched = rng.random() < 0.7
        for index in range(rng.randrange(1, 40)):
            start = rng.randrange(0, 24) * 4
            conditions = [
                {"branch": b, "arm": rng.choice(("then", "else"))}
                for b in range(3)
                if branched and rng.random() < 0.3
            ]
            events.append({"index": index, "conditions": conditions})
            timed[index] = (start, start + rng.choice((0, 4, 4, 8, 20, 60)))
            weights[index] = rng.choice((1, 1, 1, 2, 5, 0))
        items = [(index, 0, ()) for index in timed]
        rng.shuffle(items)
        for floor in (0, 1, 2, 5, 12):
            count = _Weighted(floor, events, timed)
            count.weights = weights
            assert count._bound(items, floor) == _bounded_plainly(count, items, floor, weights)
