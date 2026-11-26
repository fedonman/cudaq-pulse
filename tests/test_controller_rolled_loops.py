# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller checks of rolled loops.

The checks reason about a rolled loop once per body, with its trip count
applied to what they count, so a loop of 10**15 iterations is checked in
the time and memory of one; those tests run in a child process with a CPU,
wall and memory limit. What a check reports is what it reports with every
iteration played, which the loops past the native walk's cutoff are
compared against.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import subprocess
import sys
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, controller
from cudaq_pulse.controller import ControllerLimitError, check

from tests import _child
from tests._handles import Q, ladder_target, with_line_delays

# The child's limits: a check of one body takes well under a second and a
# few hundred MB.
CPU_S = 30
RSS_MB = 600
# What a controller that plays rolled loops unrolled supports.
_NO_LOOP = [c for c in pulse.Capability if c is not pulse.Capability.LOOP]

_KERNELS = """
import json
import cudaq_pulse as pulse
from cudaq_pulse.controller import ControllerLimitError
from cudaq_pulse import Controller
from tests._handles import Q

TRIPS = {trips}
NO_LOOP = [c for c in pulse.Capability if c is not pulse.Capability.LOOP]

@pulse.kernel
def drives(q):
    d = get_drive_line(q)
    for _ in range(TRIPS):
        drive(d, square(8, 0.1))

@pulse.kernel
def close(q):
    d = get_drive_line(q)
    for _ in range(TRIPS):
        drive(d, square(5, 0.1))
        drive(d, gaussian(8, 0.1, 2.0))

@pulse.kernel
def runtime(q: pulse.Qudit):
    d = get_drive_line(q)
    for _ in range(TRIPS):
        drive(d, square(5, 0.1))
        drive(d, gaussian(8, 0.1, 2.0))

@pulse.kernel
def across(q, r):
    a = get_drive_line(q)
    b = get_drive_line(r)
    for _ in range(TRIPS):
        drive(a, square(5, 0.1))
        drive(b, square(5, 0.1))
        sync(a, b)

@pulse.kernel
def parallel(q, r):
    a = get_drive_line(q)
    b = get_drive_line(r)
    for _ in range(TRIPS):
        drive(a, square(5, 0.1))
    for _ in range(TRIPS):
        drive(b, square(5, 0.1))

@pulse.kernel
def beside_long(q, r):
    a = get_drive_line(q)
    b = get_drive_line(r)
    drive(b, square(4000, 0.1))
    for _ in range(TRIPS):
        drive(a, square(5, 0.1))

@pulse.kernel
def in_an_arm(q, r, s, t):
    m = readout(get_readout_line(q), square(100, 0.05))
    if m:
        a = get_drive_line(r)
        b = get_drive_line(s)
        for _ in range(TRIPS):
            drive(a, gaussian(40, 0.1, 10))
            drive(b, gaussian(40, 0.1, 10))
            sync(a, b)
    else:
        drive(get_drive_line(t), square(200, 0.1))

@pulse.kernel
def once(q):
    drive(get_drive_line(q), square(8, 0.1))

@pulse.kernel
def three(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    for _ in range(3):
        drive(d, square(8, 0.1))
    for _ in range(TRIPS):
        once(q)
        wait(d, 4)
    for _ in range(TRIPS):
        sync(d, r)
        readout(r, square(40, 0.05))
        sync(r, d)

@pulse.classical(max_latency_ns=200)
def detuning() -> float:
    return 0.0

@pulse.kernel
def retuned(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    x = detuning.submit().result()
    for _ in range(TRIPS):
        if x > 0.0:
            shift_frequency(d.tone, x)
            drive(d, square(64, 0.01))
            shift_frequency(d.tone, -x)
        sync(d, r)
        m = readout(r, square(40, 0.05))
        sync(d, r)
        if m:
            drive(d, square(8, 0.1))

options = {{"control_system": Controller({limits})}}
handles = (Q[0],)
{system}
try:
    lowered = pulse.compile({kernel}, **options).lower(*handles)
except ControllerLimitError as error:
    print(json.dumps({{"count": len(error.violations), "message": str(error)}}))
else:
    print(json.dumps({{"ops": lowered.metrics.op_count}}))
"""


def _in_a_child(kernel: str, trips: int, limits: str = "", system: str = "") -> dict[str, object]:
    """What compiling *kernel* of *trips* iterations under *limits* gives, in
    a child process that is stopped past CPU_S seconds or RSS_MB MB; with
    *system*, code that sets the compile's ``options`` and ``handles``."""
    code = _child.limits(CPU_S, RSS_MB) + _KERNELS.format(
        trips=trips, limits=limits, kernel=kernel, system=system
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=4 * CPU_S,
    )
    assert out.returncode == 0, f"the child exited {out.returncode}: {out.stderr[-2000:]}"
    result: dict[str, object] = json.loads(out.stdout.splitlines()[-1])
    return result


@pulse.kernel
def close(q):
    d = get_drive_line(q)
    for _ in range(3000):
        drive(d, square(5, 0.1))
        drive(d, gaussian(8, 0.1, 2.0))


def test_a_rolled_loop_reports_each_iteration_as_it_does_now():
    lowered = pulse.compile(close).lower(Q[0])
    assert lowered.metrics.op_count == 7
    limits = Controller(
        instruction_cycles=4, min_idle_cycles=4, max_amplitude=0.05, max_pulses=7, supports=_NO_LOOP
    )
    over = "peaks at amplitude 0.1; the controller plays at most 0.05 (Controller.max_amplitude)"
    assert check(lowered, limits) == [
        "6000 drives and readouts; the controller plays at most 7. It does not support "
        "'loop', so rolled loops are unrolled for it: the loop of 3000 iterations from "
        "cycle 0 plays 5998 more",
        *(
            f"line 0: drive at {13 * k + 5} starts 5 cycles after the drive at {13 * k}; "
            "the controller needs exactly 4 or at least 8"
            for k in range(3000)
        ),
        *(
            f"line 0 (drive of qudit 0): the drive at {start} {over}"
            for k in range(3000)
            for start in (13 * k, 13 * k + 5)
        ),
    ]


def test_a_rolled_loop_reports_each_short_pulse():
    lowered = pulse.compile(close).lower(Q[0])
    assert check(lowered, Controller(min_pulse_cycles=8)) == [
        f"line 0 (drive of qudit 0): the drive at {13 * k} lasts 5 cycles; the controller "
        "plays none shorter than 8 (Controller.min_pulse_cycles)"
        for k in range(3000)
    ]
    assert check(lowered, Controller(min_pulse_cycles=5)) == []


def test_a_rolled_loop_reports_each_long_pulse():
    lowered = pulse.compile(close).lower(Q[0])
    assert check(lowered, Controller(max_pulse_cycles=5)) == [
        f"line 0 (drive of qudit 0): the drive at {13 * k + 5} lasts 8 cycles; the controller "
        "plays none longer than 5 (Controller.max_pulse_cycles)"
        for k in range(3000)
    ]
    assert check(lowered, Controller(max_pulse_cycles=8)) == []


def test_a_rolled_loop_of_a_few_iterations_compiles_under_a_controller_in_a_child():
    assert _in_a_child("drives", 1000) == {"ops": 5}


def test_a_rolled_loop_of_10_15_iterations_is_checked_in_bounded_time_and_memory():
    assert _in_a_child("drives", 10**15) == {"ops": 5}


def _given(limit: int | None, schedule: str, qudits: int, placed: bool = True) -> str:
    """Child code compiling with *schedule* for one controller and, unless
    ``None``, a ``max_concurrent_drives`` of *limit*; unless *placed*, asap
    and alap keep the schedule they place past the limit (the check alone)."""
    given = "" if limit is None else f", max_concurrent_drives={limit}"
    return (
        (
            ""
            if placed
            else "import cudaq_pulse.compiler\ncudaq_pulse.compiler._PLACED_BY_LIMITS.clear()\n"
        )
        + "from cudaq_pulse import ControlSystem\n"
        f"options = {{'schedule': '{schedule}', 'control_system': ControlSystem("
        f"controllers=[Controller()]{given})}}\n"
        f"handles = tuple(Q[:{qudits}])"
    )


@pytest.mark.parametrize(
    ("kernel", "qudits", "first"),
    [("across", 2, 0), ("parallel", 2, 0), ("beside_long", 2, 0), ("in_an_arm", 4, 100)],
)
def test_rolled_loops_of_10_15_iterations_are_held_to_concurrency_limits_in_bounds(
    kernel, qudits, first
):
    # Loops beside others of their kind: in step with another, beside
    # another loop or a long pulse, in one arm of a branch.
    plain = _in_a_child(kernel, 10**15, system=_given(None, "asap", qudits))
    assert _in_a_child(kernel, 10**15, system=_given(2, "asap", qudits)) == plain
    over = _in_a_child(kernel, 10**15, system=_given(1, "asap", qudits, placed=False))
    assert over["count"] == 1
    assert f"2 drives play at once from cycle {first}; the control system plays at most 1" in str(
        over["message"]
    )
    # Compiled, it is placed as rcp places it, in bounds too.
    assert "ops" in _in_a_child(kernel, 10**15, system=_given(1, "asap", qudits))


@pytest.mark.parametrize(
    ("kernel", "qudits", "trips"),
    # The rcp schedules place a loop beside a long pulse, or in a branch arm,
    # in time that grows with its trips, with or without limits.
    [
        ("across", 2, 10**15),
        ("parallel", 2, 10**15),
        ("beside_long", 2, 10**5),
        ("in_an_arm", 4, 10**5),
    ],
)
@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_rcp_loops_fit_concurrency_limits_in_bounds(kernel, qudits, trips, schedule):
    for limit in (1, 2):
        assert "ops" in _in_a_child(kernel, trips, system=_given(limit, schedule, qudits))


def test_the_second_and_third_loops_of_10_15_iterations_stay_rolled_in_a_child():
    assert _in_a_child("three", 10**15) == _in_a_child("three", 100)


def test_a_violation_in_every_iteration_of_10_15_is_counted_not_listed():
    few = _in_a_child("close", 20, "instruction_cycles=4, min_idle_cycles=4")
    many = _in_a_child("close", 10**15, "instruction_cycles=4, min_idle_cycles=4")
    assert many["count"] == 10**15
    assert many["message"] == str(few["message"]).replace(
        "... and 10 more", f"... and {10**15 - 10} more"
    )


def test_a_check_of_run_time_frame_changes_in_branches_takes_bounded_memory():
    # Each shot leaves a frame change pending on the path its reset skips.
    message = (
        "the program exceeds the controller's limits:\n"
        "  line 0: its waveforms take 72 samples; the controller's waveform memory holds 64"
    )
    for shots_count in (2, 400):
        found = _in_a_child("retuned", shots_count, "waveform_memory_samples=64")
        assert found == {"count": 1, "message": message}


def test_the_event_count_of_a_rolled_loop_is_its_trip_count_times_its_body():
    found = _in_a_child("drives", 10**15 + 1, "max_pulses=10**15, supports=NO_LOOP")
    assert found["count"] == 1
    assert found["message"] == (
        "the program does not fit the control system:\n  1000000000000001 drives and readouts; "
        "the controller plays at most 1000000000000000. It does not support 'loop', so rolled "
        "loops are unrolled for it: the loop of 1000000000000001 iterations from cycle 0 plays "
        "1000000000000000 more"
    )


def _every_iteration(monkeypatch: pytest.MonkeyPatch, lowered: Any, limits: Any) -> list[str]:
    """What ``check`` reports with every iteration of every loop played."""
    walk = controller._walk

    def played(*args: Any, repeats: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", played)
        return list(check(lowered, limits))


def _loops(n: int) -> dict[str, Any]:
    @pulse.kernel
    def close(q):
        d = get_drive_line(q)
        for _ in range(n):
            drive(d, square(5, 0.1))
            drive(d, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def branches(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(q)
        other = get_readout_line(r)
        for _ in range(n):
            drive(line, gaussian(16, 0.1, 4.0))
            sync(line, probe, other)
            m = readout(probe, square(40, 0.05))
            readout(other, square(40, 0.05))
            sync(probe, line, other)
            if m:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    @pulse.kernel
    def before(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(q)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)

    @pulse.kernel
    def nested(q):
        d = get_drive_line(q)
        for _ in range(n):
            for _ in range(n):
                drive(d, square(5, 0.1))
                wait(d, 2)
            drive(d, gaussian(8, 0.1, 2.0))

    return {"close": close, "branches": branches, "before": before, "nested": nested}


LIMITS = [
    Controller(
        instruction_cycles=4, min_idle_cycles=4, max_amplitude=0.05, max_pulses=7, supports=_NO_LOOP
    ),
    Controller(
        instruction_cycles=2,
        detectors={0: 0, 1: 0},
        detector_recovery_cycles=30,
        max_frequencies_per_line=1,
        waveform_memory_samples=50,
        feedback_latency_cycles=60,
    ),
    Controller(feedback_latency_cycles=500, max_pulses=10, supports=_NO_LOOP),
    Controller(instruction_cycles=2, waveform_memory_samples=10, waveform_memory="shared"),
    # A float's amplitude is checked when it is called with one.
    Controller(instruction_cycles=4, min_idle_cycles=4, max_amplitude=0.05),
]


@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize("name", ["close", "branches", "before", "nested"])
@pytest.mark.parametrize("schedule", ["asap", "rcp"])
def test_a_long_rolled_loop_reports_what_every_iteration_played_does(
    monkeypatch, trips, name, schedule
):
    lowered = pulse.compile(_loops(trips)[name], schedule=schedule, unroll_threshold=0)
    lowered = lowered.lower(Q[0], Q[1]) if name in ("branches", "before") else lowered.lower(Q[0])
    repeats: list[Any] = []
    controller._walk(lowered, values=False, summary=(repeats, None))
    assert repeats, "the native walk summarized the loop"
    for limits in LIMITS:
        found = check(lowered, limits)
        assert (type(found) is list) == (len(found) <= 10_000)
        assert list(found) == _every_iteration(monkeypatch, lowered, limits)


def _timed(event: dict[str, Any]) -> tuple[Any, ...]:
    return event["kind"], event["line"], event["start_cycles"], event["end_cycles"]


def _unsynced(n: int) -> dict[str, Any]:
    """Loops whose lines move by different periods."""

    @pulse.kernel
    def two_lines(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def then(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, gaussian(8, 0.1, 2.0))
        sync(a, b)
        drive(a, square(6, 0.2))
        drive(b, square(2, 0.1))

    @pulse.kernel
    def two_readouts(q, r):
        p = get_readout_line(q)
        o = get_readout_line(r)
        for _ in range(n):
            readout(p, square(40, 0.05))
            readout(o, square(48, 0.05))

    @pulse.kernel
    def feedback(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            if m:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    @pulse.kernel
    def early(q, r):
        line = get_drive_line(q)
        other = get_drive_line(r)
        probe = get_readout_line(q)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)
            drive(other, square(5, 0.1))

    @pulse.kernel
    def early_on_the_slower_line(q, r):
        other = get_drive_line(q)
        line = get_drive_line(r)
        probe = get_readout_line(r)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            drive(other, square(5, 0.1))
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)

    @pulse.kernel
    def after(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        other = get_drive_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            drive(line, square(8, 0.1))
            drive(other, square(4, 0.1))
        if m:
            drive(line, square(8, 0.1))

    @pulse.kernel
    def nested(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            for _ in range(n):
                drive(a, square(5, 0.1))
                wait(a, 2)
            drive(b, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def partly_synced(q, r):
        a = get_drive_line(q)
        p = get_readout_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(16, 0.1))
            sync(a, p)
            readout(p, square(40, 0.05))
            drive(b, square(4, 0.1))
            wait(b, 2)

    return {
        "two_lines": two_lines,
        "then": then,
        "two_readouts": two_readouts,
        "feedback": feedback,
        "early": early,
        "early_on_the_slower_line": early_on_the_slower_line,
        "after": after,
        "nested": nested,
        "partly_synced": partly_synced,
    }


_LEFT_TO_THE_WALK = (
    "a long loop whose iterations repeat out of step is checked from its summarizing walk"
)


# The native check walks a loop that does not repeat one period apart at
# every iteration; one of more trips than the summarizing walk keeps rolled
# (rcp peels a few first) it leaves to that walk.
@pytest.mark.parametrize("schedule", ["asap", "rcp"])
def test_the_native_check_leaves_a_long_loop_out_of_step_to_the_summarizing_walk(schedule):
    limits = Controller(instruction_cycles=4, max_amplitude=0.05)
    left = []
    for trips in (64, 200):
        program = pulse.compile(
            _unsynced(trips)["two_lines"], schedule=schedule, unroll_threshold=0
        )
        found = controller._native_check(program.lower(Q[0], Q[1]), limits)
        left.append(found == [_LEFT_TO_THE_WALK])
    assert left == [False, True]


@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize(
    "name",
    [
        "two_lines",
        "then",
        "two_readouts",
        "feedback",
        "early",
        "early_on_the_slower_line",
        "after",
        "nested",
        "partly_synced",
    ],
)
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_loop_whose_lines_move_by_different_periods_reports_what_every_iteration_does(
    monkeypatch, trips, name, schedule
):
    from cudaq_pulse import _loops

    lowered = pulse.compile(_unsynced(trips)[name], schedule=schedule, unroll_threshold=0)
    lowered = lowered.lower(Q[0], Q[1])
    repeats: list[Any] = []
    periods: list[Any] = []
    lines, events, *_ = controller._walk(lowered, values=False, summary=(repeats, periods))
    if repeats:
        rest = len(events) - max(r[1] for r in repeats)
        full = controller._walk(lowered, values=False)[1]
        assert [_timed(e) for e in events[len(events) - rest :]] == [
            _timed(e) for e in full[len(full) - rest :]
        ]
    summarized = 0
    for limits in LIMITS:
        found = check(lowered, limits)
        assert list(found) == _every_iteration(monkeypatch, lowered, limits)
        if periods:
            try:
                _loops.violations(lowered, events, repeats, limits, lines, None, None, periods)
            except _loops.UnsyncedError:
                continue
            summarized += 1
    if periods and name != "feedback":
        assert summarized, "some limits are checked on the summarized loop"


# Controllers on clocks of their own: a 4-cycle timestep, half the program
# clock, and two thirds of it (a 3-cycle grid).
OWN_CLOCKS = [{"cycles_per_timestep": 4}, {"clock_ghz": 1.0}, {"clock_ghz": 4 / 3}]


def _lowered(kernel: Any, schedule: str, **options: Any) -> Any:
    compiled = pulse.compile(kernel, schedule=schedule, unroll_threshold=0, **options)
    return compiled.lower(*Q[: kernel.__wrapped__.__code__.co_argcount])


@pytest.mark.parametrize("clock", range(len(OWN_CLOCKS)))
@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize("name", [*_loops(1), *_unsynced(1)])
@pytest.mark.parametrize("schedule", ["asap", "rcp"])
def test_a_long_rolled_loop_checked_on_a_controllers_own_clock_reports_what_every_iteration_does(
    monkeypatch, clock, trips, name, schedule
):
    kernels = {**_loops(trips), **_unsynced(trips)}
    lowered = _lowered(kernels[name], schedule)
    for limits in LIMITS:
        limits = dataclasses.replace(limits, **OWN_CLOCKS[clock])
        found = check(lowered, limits)
        assert list(found) == _every_iteration(monkeypatch, lowered, limits)


def _on_grid(n: int) -> dict[str, Any]:
    """Loops whose operations last whole timesteps of every grid of
    OWN_CLOCKS."""

    @pulse.kernel
    def drives(q):
        d = get_drive_line(q)
        for _ in range(n):
            drive(d, square(12, 0.1))
            drive(d, gaussian(24, 0.3, 4.0))

    @pulse.kernel
    def feedback(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(n):
            m = readout(probe, square(48, 0.05))
            if m:
                drive(line, square(12, 0.1))
            else:
                drive(line, square(24, 0.2))

    @pulse.kernel
    def two_lines(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(12, 0.1))
            drive(b, gaussian(24, 0.1, 4.0))

    @pulse.kernel
    def nested(q):
        d = get_drive_line(q)
        for _ in range(n):
            for _ in range(n):
                drive(d, square(12, 0.1))
                wait(d, 12)
            drive(d, gaussian(24, 0.1, 2.0))

    @pulse.kernel
    def scaled(q, amp: float):
        d = get_drive_line(q)
        for _ in range(n):
            drive(d, square(12, amp))
            wait(d, 12)

    return {
        "drives": drives,
        "feedback": feedback,
        "two_lines": two_lines,
        "nested": nested,
        "scaled": scaled,
    }


def _on(kernel: Any, schedule: str, limits: Any) -> Any:
    """*kernel* compiled under *limits*; one of a float is checked when
    called with one, as its export is."""
    machine = ControlSystem(controllers=[limits])
    if kernel.__wrapped__.__name__ != "scaled":
        return _lowered(kernel, schedule, control_system=machine)
    compiled = pulse.compile(kernel, schedule=schedule, unroll_threshold=0, control_system=machine)
    return compiled.lower(Q[0], 0.6)


@pulse.classical(max_latency_ns=200)
def _delay_of(bits: pulse.Bits) -> float:
    return 48.0 + 24.0 * bits[0]


@pulse.kernel
def _delayed_at_run_time(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(100):
        drive(d, square(24, 0.1))
    m = readout(ro, square(96, 0.05))
    wait(d, _delay_of.submit([m]).result(), bounds=(48, 384))
    drive(d, square(24, 0.1))


# Its own cycles are converted in the export's terms of what is computed at
# run time, so a check walk of a module that computes any gives them as the
# export does.
@pytest.mark.parametrize("clock", range(len(OWN_CLOCKS)))
def test_a_check_on_a_controllers_own_clock_converts_what_runs_at_run_time_as_its_export(clock):
    limits = dataclasses.replace(Controller(), **OWN_CLOCKS[clock])
    lowered = _on(_delayed_at_run_time, "asap", limits)
    own = controller._native_clock(lowered.clocks)
    repeats: list[Any] = []
    checked = controller._walk(
        lowered, values=False, own=own, waveforms=False, summary=(repeats, None)
    )[1]
    exported = controller._walk(lowered, values=False, own=own, waveforms=False)[1]
    assert repeats

    def answered(value: Any, events: list[Any]) -> Any:
        # An answer by the event's kind and start: the summarized loop
        # before it has fewer events.
        if isinstance(value, dict):
            if "answer" in value:
                event = events[value["answer"]]
                return {**value, "answer": (event["kind"], event["start_cycles"])}
            return {k: answered(v, events) for k, v in value.items()}
        if isinstance(value, list):
            return [answered(v, events) for v in value]
        return value

    lengths = [
        [answered(e["length"], events) for e in events if e["kind"] == "wait"]
        for events in (checked, exported)
    ]
    assert lengths[0] == lengths[1]


def _compiled(kernel: Any, schedule: str, limits: Any) -> Any:
    """The ops of *kernel* compiled under *limits*, or the limits it exceeds."""
    try:
        lowered = _on(kernel, schedule, limits)
    except ControllerLimitError as error:
        return list(error.violations)
    return lowered.metrics.op_count


@pytest.mark.parametrize("clock", range(len(OWN_CLOCKS)))
@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize("name", list(_on_grid(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_long_rolled_loop_compiled_for_a_controllers_own_clock_is_summarized_and_exact(
    monkeypatch, clock, trips, name, schedule
):
    kernel = _on_grid(trips)[name]
    lowered = _on(kernel, schedule, dataclasses.replace(Controller(), **OWN_CLOCKS[clock]))
    own = controller._native_clock(lowered.clocks)
    repeats: list[Any] = []
    periods: list[Any] = []
    controller._walk(lowered, values=False, own=own, summary=(repeats, periods))
    program: list[Any] = []
    controller._walk(lowered, values=False, summary=(program, []))
    # The lanes of two lines' alternate drives under rcp never repeat.
    assert program or (name, schedule) == ("two_lines", "rcp")
    assert len(repeats) == len(program), "summarized in the controller's cycles as in the program's"
    walk = controller._walk
    for limits in LIMITS:
        limits = dataclasses.replace(limits, **OWN_CLOCKS[clock])
        found = _compiled(kernel, schedule, limits)

        def played(*args: Any, repeats: Any = None, periods: Any = None, **kwargs: Any) -> Any:
            return walk(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(controller, "_walk", played)
            assert found == _compiled(kernel, schedule, limits)


def test_a_rolled_loop_of_10_15_iterations_on_a_controllers_own_clock_is_checked_in_bounds():
    for clock in ("cycles_per_timestep=4", "clock_ghz=1.0"):
        few = _in_a_child("close", 20, f"instruction_cycles=4, min_idle_cycles=4, {clock}")
        many = _in_a_child("close", 10**15, f"instruction_cycles=4, min_idle_cycles=4, {clock}")
        each = int(str(few["count"])) // 20
        assert each and many["count"] == each * 10**15
        assert many["message"] == str(few["message"]).replace(
            f"... and {20 * each - 10} more", f"... and {each * 10**15 - 10} more"
        )


def _runtime(n: int) -> dict[str, Any]:
    """Loops of runtime qudits, whose lines are bound per call."""

    @pulse.kernel
    def drives(q: pulse.Qudit):
        d = get_drive_line(q)
        for _ in range(n):
            drive(d, square(12, 0.1))
            drive(d, gaussian(24, 0.3, 4.0))

    @pulse.kernel
    def feedback(q: pulse.Qudit, r: pulse.Qudit):
        a = get_drive_line(q)
        p = get_readout_line(r)
        for _ in range(n):
            drive(a, square(12, 0.1))
            sync(a, p)
            m = readout(p, square(48, 0.05))
            sync(p, a)
            if m:
                drive(a, square(12, 0.2))

    @pulse.kernel
    def tones(q: pulse.Qudit, r: pulse.Qudit):
        d = get_drive_line(q)
        e = get_drive_line(r)
        for _ in range(n):
            drive(d, square(12, 0.1))
            drive(e, square(12, 0.1))
        set_frequency(e.tone, 5.1e9)
        for _ in range(n):
            drive(e, square(12, 0.1))

    @pulse.kernel
    def two_lines(q: pulse.Qudit, r: pulse.Qudit):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(12, 0.1))
            drive(b, gaussian(24, 0.1, 4.0))

    return {"drives": drives, "feedback": feedback, "tones": tones, "two_lines": two_lines}


@pytest.mark.parametrize("clock", [None, *range(len(OWN_CLOCKS))])
@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize("name", list(_runtime(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_long_rolled_loop_of_runtime_qudits_is_summarized_and_checked_as_every_iteration(
    monkeypatch, clock, trips, name, schedule
):
    kernel = _runtime(trips)[name]
    generic = _lowered(kernel, schedule)
    generic = generic._generic or generic
    assert generic._role.slots
    program: list[Any] = []
    controller._walk(generic, values=False, summary=(program, []))
    assert program or (name, schedule) == ("two_lines", "rcp")
    walk = controller._walk

    def played(*args: Any, repeats: Any = None, periods: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    # A runtime qudit's own frequency is one tone wherever it plays.
    tones = [Controller(max_frequencies_per_line=1), Controller(max_frequencies_per_line=2)]
    for limits in [*LIMITS, *tones]:
        if clock is not None:
            limits = dataclasses.replace(limits, **OWN_CLOCKS[clock])
        found, exact = controller._compile_check(generic, limits)
        compiled = _compiled(kernel, schedule, limits)
        with monkeypatch.context() as patch:
            patch.setattr(controller, "_walk", played)
            every, exactly = controller._compile_check(generic, limits)
            assert (list(found), exact) == (list(every), exactly)
            assert compiled == _compiled(kernel, schedule, limits)


def test_a_rolled_loop_of_10_15_iterations_of_a_runtime_qudit_is_checked_in_bounds():
    for limits in (
        "instruction_cycles=4, min_idle_cycles=4",
        "max_frequencies_per_line=1, max_amplitude=0.05",
    ):
        few = _in_a_child("runtime", 20, limits)
        many = _in_a_child("runtime", 10**15, limits)
        each = int(str(few["count"])) // 20
        assert each and many["count"] == each * 10**15
        assert many["message"] == str(few["message"]).replace(
            f"... and {20 * each - 10} more", f"... and {each * 10**15 - 10} more"
        )


_CAPABILITIES = [c.value for c in pulse.Capability]


def _systems() -> dict[str, Any]:
    """Control systems of one controller, or of two playing two qudits each."""

    def two(
        reference: float | None = None,
        delays: Any = None,
        skew: int = 0,
        cores: bool = False,
        **each: Any,
    ) -> Any:
        # A processor leaves a check with float values to each call.
        processors = [pulse.Processor(links=["cores"])] if cores else []
        controllers = [
            pulse.Controller(
                drive_lines=[0, 1], readout_lines=[0, 1], processors=processors, **each
            ),
            pulse.Controller(
                drive_lines=[2, 3], readout_lines=[2, 3], clock_skew_cycles=skew, **each
            ),
        ]
        return pulse.ControlSystem(
            controllers=with_line_delays(controllers, delays or {}),
            links={"cores": pulse.Link()} if cores else {},
            reference_clock_ghz=reference,
        )

    one = pulse.Controller(
        drive_lines=range(4), readout_lines=range(4), instruction_cycles=4, min_idle_cycles=4
    )
    return {
        "one": pulse.ControlSystem(controllers=[one]),
        "unshared": two(instruction_cycles=2),
        "delayed": two(1.0, {("drive", 2): 3.0}, cores=True, instruction_cycles=2),
        "skewed": two(1.0, skew=1, instruction_cycles=2),
        "shared": two(1.0, instruction_cycles=2, feedback_latency_cycles=60, peer_latency_cycles=8),
        "counted": two(1.0, max_pulses=10, max_frequencies_per_line=1, waveform_memory_samples=50),
        "unrolled": two(1.0, supports=[c for c in _CAPABILITIES if c != "loop"], max_pulses=10),
        "unforwarded": two(1.0, supports=[c for c in _CAPABILITIES if c != "forward_outcome"]),
        "gridded": two(1.0, cycles_per_timestep=4, max_amplitude=0.05),
        "processed": two(1.0, cores=True, instruction_cycles=2),
        "clocked": pulse.ControlSystem(
            controllers=[
                pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=1.0),
                pulse.Controller(
                    drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5, instruction_cycles=2
                ),
            ],
            reference_clock_ghz=0.5,
        ),
    }


def _across(n: int) -> dict[str, Any]:
    """Loops of two qudits, which two controllers play."""

    @pulse.kernel
    def synced(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(12, 0.1))
            sync(a, b)
            drive(b, square(24, 0.1))
            sync(b, a)

    @pulse.kernel
    def forwarded(q, r):
        p = get_readout_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            m = readout(p, square(48, 0.05))
            sync(p, b)
            if m:
                drive(b, square(12, 0.1))
            sync(b, p)

    @pulse.kernel
    def apart(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(12, 0.1))
            drive(a, gaussian(24, 0.3, 4.0))
        sync(a, b)
        for _ in range(n):
            drive(a, square(12, 0.1))
            drive(b, square(12, 0.1))

    @pulse.kernel
    def odd(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(10, 0.1))
            sync(a, b)
            drive(b, square(12, 0.1))
            sync(b, a)

    @pulse.kernel
    def scaled(q, r, amp: float):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(12, amp))
            sync(a, b)
            drive(b, square(12, amp))
            sync(b, a)

    @pulse.kernel
    def loose(q, r, amp: float):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(12, amp))
            drive(b, square(12, amp))

    return {
        "synced": synced,
        "forwarded": forwarded,
        "apart": apart,
        "odd": odd,
        "scaled": scaled,
        "loose": loose,
    }


def _on_system(kernel: Any, schedule: str, qpu: Any) -> Any:
    """The ops of *kernel* compiled for *qpu* (and called, with a float), or
    the limits it exceeds."""
    try:
        compiled = pulse.compile(kernel, quantum_system=qpu, schedule=schedule, unroll_threshold=0)
        handles = (qpu.qudits[0], qpu.qudits[2])
        if kernel.__wrapped__.__name__ in ("scaled", "loose"):
            return compiled.lower(*handles, 0.6).metrics.op_count
        return compiled.lower(*handles).metrics.op_count
    except ControllerLimitError as error:
        return list(error.violations)


@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize("name", list(_across(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
@pytest.mark.parametrize("system", list(_systems()))
def test_a_long_rolled_loop_on_a_control_system_is_checked_as_every_iteration(
    monkeypatch, system, schedule, name, trips
):
    from cudaq_pulse import _system

    qpu = ladder_target(4, control_system=_systems()[system])
    kernel = _across(trips)[name]
    kept: list[bool] = []
    summarized = _system.ControlSystem._summarized

    def spied(self: Any, *args: Any) -> Any:
        found = summarized(self, *args)
        kept.append(found is not None)
        return found

    with monkeypatch.context() as patch:
        patch.setattr(_system.ControlSystem, "_summarized", spied)
        found = _on_system(kernel, schedule, qpu)
    walk = controller._walk

    def played(*args: Any, repeats: Any = None, periods: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", played)
        assert found == _on_system(kernel, schedule, qpu)
    if system in ("unshared", "delayed", "skewed", "clocked"):
        # A sync of controllers without a shared reference clock, or with
        # line delays or skew, lines up their pulses within a cycle, and
        # controllers of different clocks count each line's periods in their
        # own cycles: every iteration is played.
        assert not kept
        return
    # A pulse before its forwarded outcome, or that needs a capability, is
    # reported for each copy, as is a pulse off its controller's grid.
    played_out = name == "odd" and system == "gridded"
    played_out |= name == "forwarded" and system == "unforwarded"
    assert kept and any(kept) != played_out


def test_a_rolled_loop_of_10_15_iterations_across_controllers_is_checked_in_bounds():
    system = (
        "from tests._handles import ladder_target\n"
        "def _controller(lines):\n"
        "    return pulse.Controller(drive_lines=lines, readout_lines=lines, instruction_cycles=4,"
        " min_idle_cycles=4)\n"
        "qpu = ladder_target(4, control_system=pulse.ControlSystem("
        "controllers=[_controller([0, 1]), _controller([2, 3])], reference_clock_ghz=1.0))\n"
        "options = {'quantum_system': qpu}\n"
        "handles = (qpu.qudits[0], qpu.qudits[2])\n"
    )
    few = _in_a_child("across", 20, system=system)
    more = _in_a_child("across", 21, system=system)
    many = _in_a_child("across", 10**15, system=system)
    count = int(str(few["count"]))
    each = int(str(more["count"])) - count
    assert each and many["count"] == count + each * (10**15 - 20)
    assert many["message"] == str(few["message"]).replace(
        f"... and {count - 10} more", f"... and {count + each * (10**15 - 20) - 10} more"
    )


def test_past_10_000_violations_they_are_built_when_read(monkeypatch):
    lowered = pulse.compile(_loops(6000)["close"], unroll_threshold=0).lower(Q[0])
    limits = LIMITS[0]
    lazy = check(lowered, limits)
    assert type(lazy) is not list and len(lazy) == 1 + 6000 + 12000
    every = _every_iteration(monkeypatch, lowered, limits)
    assert list(lazy) == every
    assert lazy[0] == every[0] and lazy[-1] == every[-1] and lazy[6001] == every[6001]
    assert lazy[5:9] == every[5:9] and lazy[-3:] == every[-3:]
    with pytest.raises(IndexError):
        lazy[len(every)]
    error = ControllerLimitError(lazy)
    assert error.violations is lazy
    assert str(error) == str(ControllerLimitError(every))
    copy = pickle.loads(pickle.dumps(error))
    assert len(copy.violations) == len(every) and copy.violations[-1] == every[-1]
