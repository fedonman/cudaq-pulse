# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Waits whose length comes from a classical answer: barriers and segments."""

from __future__ import annotations

import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q
from tests.classical._schedule_matrix import line_timelines
from tests.property_settings import examples

POLICIES = ("asap", "alap", "rcp", "alap_rcp")


@pulse.classical(max_latency_ns=50)
def next_tau(x: float) -> float:
    return 2.0 * x


@pulse.classical(max_latency_ns=50)
def tau_index(bits: pulse.Bits) -> int:
    return 10 * bits[0]


def _mlir(kernel, *args, **options):
    return pulse.compile(kernel, **options).lower(*args).mlir()


@pulse.kernel
def single(q, tau0: float):
    d = get_drive_line(q)
    tau = next_tau.submit(tau0).result()
    drive(d, square(40, 0.1))
    wait(d, tau, bounds=(16, 4000))
    drive(d, square(40, 0.1))


def test_a_wait_from_an_answer_is_a_variable_wait():
    text = _mlir(single, Q[0], 10.0)
    assert re.search(
        r"pulse.wait .*classical.variable, .*pulse.max_cycles = 4000 : i64, "
        r"pulse.min_cycles = 16 : i64, pulse.variable_wait",
        text,
    )
    # Rounded to the nearest cycle: (x + 0.5) cast to an integer.
    assert "arith.addf" in text and "arith.fptosi" in text


def test_integer_answers_and_no_bounds():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        n = tau_index.submit([readout(r, square(100, 0.05))]).result()
        wait(d, n)
        drive(d, square(40, 0.1))

    text = _mlir(k, Q[0])
    assert re.search(r"classical.variable, .*pulse.min_cycles = 0 : i64, pulse.variable_wait", text)
    assert "pulse.max_cycles" not in text


def test_bounds_errors():
    def bad_bounds(q):
        d = get_drive_line(q)
        wait(d, next_tau.submit(1.0).result(), bounds=(10, 5))

    def fixed_outside_bounds(q):
        d = get_drive_line(q)
        wait(d, 30, bounds=(0, 20))

    def not_a_pair(q):
        d = get_drive_line(q)
        wait(d, next_tau.submit(1.0).result(), bounds=5)

    for body, message in (
        (bad_bounds, "lowest <= highest"),
        (fixed_outside_bounds, "outside its bounds"),
        (not_a_pair, r"takes \(lowest, highest\)"),
    ):
        with pytest.raises(pulse.CompilationError, match=message):
            pulse.compile(pulse.kernel(body)).lower(Q[0])

    # A concrete length within its bounds is a plain wait.
    @pulse.kernel
    def fixed_with_bounds(q):
        d = get_drive_line(q)
        wait(d, 10, bounds=(0, 20))

    assert "pulse.variable_wait" not in pulse.compile(fixed_with_bounds).lower(Q[0]).mlir()


def test_a_variable_wait_cannot_be_in_a_block_or_a_branch_arm():
    @pulse.kernel(schedule="asap")
    def inner(d, x):
        wait(d, x)

    @pulse.kernel
    def in_block(q):
        d = get_drive_line(q)
        inner(d, next_tau.submit(1.0).result())

    @pulse.kernel
    def in_branch(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        x = next_tau.submit(1.0).result()
        if readout(r, square(100, 0.05)):
            wait(d, x)

    with pytest.raises(RuntimeError, match="cannot be inside a pulse.schedule_block"):
        pulse.compile(in_block).lower(Q[0])
    with pytest.raises(RuntimeError, match="cannot be inside one arm of an scf.if"):
        pulse.compile(in_branch).lower(Q[0])
    # Flattened, the kernel is no block, and the wait is allowed.
    pulse.compile(in_block, flatten_schedules=True).lower(Q[0])


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("policy", POLICIES)
def test_segments_and_their_shift_ranges(policy, tied):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(400, 0.01))
        if tied:
            sync(d, e)
        a = next_tau.submit(1.0).result()
        wait(d, a, bounds=(8, 100))
        drive(d, square(40, 0.1))
        b = next_tau.submit(a).result()
        wait(d, b)
        drive(e, square(40, 0.1))

    low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
    segments = low.segments()
    assert [e.opened_by for e in segments] == ["start", "wait", "wait"]
    assert segments[1].max_shift_cycles == 92 and segments[2].max_shift_cycles is None
    calls = low.classical_report()
    assert [c.segment for c in calls] == [0, 1]
    spectator = [e for e in low.schedule if e.kind == "drive" and e.qudit == 1]
    if tied:
        # Synced with the 400-cycle drive on the other line, the first
        # barrier waits for it, and that line's next drive for the barriers.
        assert segments[1].start_cycles == 400 + 8
        assert spectator[1].start >= segments[2].start_cycles
    else:
        # Untied, it starts once its answer is due, and the other line
        # plays on without waiting for either barrier.
        assert segments[1].start_cycles == calls[0].due_cycles + 8 < 400
        assert spectator[1].start == 400


def test_a_program_without_variable_waits_has_one_segment():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(40, next_tau.submit(1.0).result()))

    assert pulse.compile(k).lower(Q[0]).segments() == [
        pulse.compiler.Segment(0, "start", -1, 0, 0, 0)
    ]


# ── Segment invariants ─────────────────────────────────────────────────────────

_STEP = st.tuples(st.sampled_from(("drive", "readout", "variable_wait", "call")), st.integers(0, 2))


def _program(steps):
    @pulse.kernel
    def k(q0, q1, q2):
        qs = (q0, q1, q2)
        x = 1.0
        for kind, q in steps:
            d = get_drive_line(qs[q])
            r = get_readout_line(qs[q])
            if kind == "drive":
                drive(d, square(30, 0.1))
            elif kind == "readout":
                readout(r, square(50, 0.05))
            elif kind == "call":
                x = next_tau.submit(x).result()
            elif kind == "variable_wait":
                wait(d, next_tau.submit(x).result(), bounds=(4, None))

    return k


@settings(max_examples=examples(40), deadline=None)
@given(steps=st.lists(_STEP, min_size=1, max_size=20), policy=st.sampled_from(POLICIES))
def test_segment_invariants(steps, policy):
    low = pulse.compile(_program(steps), schedule=policy).lower(Q[0], Q[1], Q[2])
    events = low.schedule
    barriers = [
        i
        for i, e in enumerate(events)
        if e.kind == "wait" and e.link is None and e.duration_cycles == 4
    ]
    variable = sum(1 for kind, _ in steps if kind == "variable_wait")
    assert len(low.segments()) == variable + 1
    # A barrier holds back the lines of its timeline: with one timeline,
    # every line and call.
    timelines = line_timelines(low)
    single = len(set(timelines.values())) == 1
    for i in barriers:
        start = events[i].start_cycles
        end = start + events[i].duration_cycles
        timeline = timelines[events[i].line]
        for j, e in enumerate(events):
            held = single if e.link is not None else timelines[e.line] == timeline
            if not held:
                continue
            if j < i:
                assert e.start_cycles + e.duration_cycles <= start
            elif j > i:
                assert e.start_cycles >= end


@settings(max_examples=examples(40), deadline=None)
@given(steps=st.lists(_STEP, min_size=1, max_size=20), policy=st.sampled_from(POLICIES))
def test_the_check_walk_counts_the_segments_of_the_export(steps, policy):
    from cudaq_pulse.controller import _AMPLITUDES, _PARAMETERS

    module = pulse.compile(_program(steps), schedule=policy).lower(Q[0], Q[1], Q[2])._concrete()
    _, checked = module.controller_events(_PARAMETERS, _AMPLITUDES)
    _, exported, _ = module.controller_program(_PARAMETERS, _AMPLITUDES, True, exportable=False)
    assert [e["segment"] for e in checked] == [
        e["segment"] for e in exported if e["kind"] != "frame_update"
    ]


def test_a_wait_from_a_float_parameter_beside_calls_is_no_barrier():
    @pulse.kernel
    def k(q, p, tau: float):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(400, 0.01))
        wait(d, tau, bounds=(0, 100))
        drive(d, square(40, 0.1))
        a = next_tau.submit(1.0).result()
        wait(d, a, bounds=(8, 100))
        drive(d, square(40, 0.1))

    compiled = pulse.compile(k, schedule="asap")
    low = compiled.lower(Q[0], Q[1], 10.4)
    waits = [line for line in low.mlir().splitlines() if "pulse.wait" in line]
    assert len(waits) == 2
    # The parameter's wait: bound and rounded, no barrier; the answer's: a barrier.
    assert "classical.variable" not in waits[0] and "duration_cycles = 10 : i64" in waits[0]
    assert "classical.variable" in waits[1] and "pulse.min_cycles = 8" in waits[1]
    drives = [e for e in low.schedule if e.kind == "drive" and e.line == 0]
    assert drives[0].start == 10
    assert [e.opened_by for e in low.segments()] == ["start", "wait"]
    # One compilation for every length; a length outside its bounds is a call error.
    compiled.lower(Q[0], Q[1], 50.0)
    assert compiled.cache_info().compiled == 1
    with pytest.raises(ValueError, match="bounds"):
        compiled.lower(Q[0], Q[1], 120.0)


@pulse.kernel
def offset_by_tau(q, p, tau: float):
    d = get_drive_line(q)
    e = get_drive_line(p)
    drive(e, square(400, 0.01))
    drive(d, square(40, 0.1))
    a = next_tau.submit(10.0).result()
    wait(d, a + tau, bounds=(8, 1000))
    drive(d, square(40, 0.1))


def test_a_wait_from_an_answer_and_a_float_compiles_once():
    compiled = pulse.compile(offset_by_tau)
    for k in range(1000):
        low = compiled.lower(Q[0], Q[1], 0.5 * k)
    assert compiled.cache_info().compiled == 1
    # The answer keeps the wait a barrier that opens a segment, at its lowest
    # length until it is known; it waits for the answer, not for the other
    # line's 400-cycle drive.
    assert [e.opened_by for e in low.segments()] == ["start", "wait"]
    (wait_event,) = [e for e in low.schedule if e.kind == "wait" and e.link is None]
    (call,) = low.classical_report()
    assert wait_event.start == max(40, call.due_cycles) and wait_event.duration_cycles == 8
    assert "%arg0: f64" in low.generic.mlir()


@pytest.mark.parametrize(
    ("tau", "message"),
    [
        (
            1500.0,
            r"next_tau\[0\] \+ tau \(its answers at 0\) = 1500.0 .* above its highest length 1000",
        ),
        (float("nan"), "argument 'tau' is nan; a float argument must be a number"),
        (float("-inf"), "must be a finite number"),
    ],
)
def test_the_float_part_of_a_wait_from_an_answer_is_checked_at_call_time(tau, message):
    compiled = pulse.compile(offset_by_tau)
    with pytest.raises(ValueError, match=message):
        compiled.lower(Q[0], Q[1], tau)
    compiled.lower(Q[0], Q[1], 30.0)
    assert compiled.cache_info().compiled == 1


@pytest.mark.parametrize("tau", [2.0, -50.0, 100.0])
def test_a_float_part_below_the_lowest_length_is_clamped_with_the_answer(tau):
    @pulse.kernel
    def k(q, tau: float):
        d = get_drive_line(q)
        wait(d, next_tau.submit(10.0).result() + tau, bounds=(8, 100))
        drive(d, square(40, 0.1))

    # Only the float part's highest length is checked when called: the
    # answer adds to it, and the run clamps the sum to the bounds.
    compiled = pulse.compile(k)
    low = compiled.lower(Q[0], tau)
    assert [e.opened_by for e in low.segments()] == ["start", "wait"]
    with pytest.raises(ValueError, match="above its highest length 100"):
        compiled.lower(Q[0], 101.0)
    assert compiled.cache_info().compiled == 1


@pytest.mark.parametrize("tau", [30.4, 980.0, -15.0])
def test_a_wait_from_an_answer_and_a_float_runs_their_sum_clamped(tau):
    from cudaq_pulse.quantum_systems import PhysicalQudit

    qudits = {i: PhysicalQudit(i, 5e9 + 1e8 * i, 2.0, 0.0, anharmonicity_hz=-2e8) for i in range(2)}
    dev = pulse.quantum_system(name="decay", qudits=qudits)
    q, p = dev.qudits
    length = min(max(round(2.0 * 10.0 + tau), 8), 1000)
    low = pulse.compile(offset_by_tau, quantum_system=dev).lower(q, p, tau)
    # The drive on d, then the wait once its answer is due, where the static
    # schedule puts them; the other line plays on.
    drive_d = next(e for e in low.schedule if e.kind == "drive" and e.line == 0)
    (variable,) = [e for e in low.schedule if e.kind == "wait" and e.link is None]

    @pulse.kernel
    def fixed(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(400, 0.01))
        if drive_d.start:
            wait(d, drive_d.start)
        drive(d, square(40, 0.1))
        if variable.start > drive_d.end:
            wait(d, variable.start - drive_d.end)
        wait(d, length)
        drive(d, square(40, 0.1))

    mixed = pulse.simulate(low)
    reference = pulse.simulate(
        pulse.compile(fixed, quantum_system=dev, schedule="asap").lower(q, p)
    )
    np.testing.assert_allclose(mixed.density_matrix, reference.density_matrix, atol=1e-9)


@pulse.classical(max_latency_ns=50, latency_mode="blocking")
def blocking_tau(x: float) -> float:
    return x


@pytest.mark.parametrize("barrier", ["variable", "blocking", "variable_in_loop"])
def test_pulses_either_side_of_a_barrier_are_not_fused(barrier):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(400, 0.05))
        if barrier == "variable":
            wait(d, next_tau.submit(10.0).result(), bounds=(8, 100))
        elif barrier == "blocking":
            drive(d, square(40, blocking_tau.submit(0.1).result()))
        else:
            for _ in range(2):
                wait(d, next_tau.submit(10.0).result(), bounds=(8, 100))
        drive(e, square(30, 0.05))

    low = pulse.compile(k, unroll_threshold=0).lower(Q[0], Q[1])
    spectator = [e for e in low.schedule if e.kind == "drive" and e.qudit == 1]
    assert [e.end - e.start for e in spectator] == [400, 30]
    assert spectator[1].start >= low.segments()[-1].start_cycles


def test_a_loop_that_hands_each_answer_to_the_next_call_compiles():
    """The length of each wait is looked for once, however many calls came
    before it: the loop is written out, and each call takes the answer of the
    one before."""

    @pulse.kernel
    def adaptive(q, tau0: float):
        d = get_drive_line(q)
        tau = tau0
        for _ in range(60):
            tau = next_tau.submit(tau).result()
            drive(d, square(40, 0.1))
            wait(d, tau, bounds=(16, 4000))

    for policy in POLICIES:
        text = _mlir(adaptive, Q[0], 10.0, schedule=policy)
        assert text.count("classical.variable") == 60
