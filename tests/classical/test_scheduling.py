# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduling classical calls under every policy, with property checks."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import ControlSystem
from cudaq_pulse.classical import Link
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q
from tests._work import (
    cpu_seconds_each,
    native_seconds_each,
    python_work,
)
from tests.property_settings import examples

POLICIES = ("asap", "alap", "rcp", "alap_rcp")
CLOCK = 2.0  # the test quantum system's clock, GHz
READOUT, ARM, LATENCY_NS = 300, 40, 1000
BUDGET = math.ceil(LATENCY_NS * CLOCK)


@pulse.classical(max_latency_ns=LATENCY_NS)
def decode(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=LATENCY_NS, submit="lazy", name="decode_lazy")
def decode_lazy(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=100, link=1)
def tune(x: float) -> float:
    return x


@pulse.classical(max_latency_ns=50, latency_mode="blocking", link=2)
def calibrate(x: float) -> float:
    return x


@pulse.kernel
def feedback(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(READOUT, 0.05))
    if decode.submit([m]).result()[0]:
        drive(d, square(ARM, 0.1))


def _events(low, kind):
    return [e for e in low.schedule if e.kind == kind]


def _makespan(low):
    return max(e.start_cycles + e.duration_cycles for e in low.schedule)


@pytest.mark.parametrize("policy", POLICIES)
def test_feedback_makespan_is_readout_submit_latency_and_arm(policy):
    low = pulse.compile(feedback, schedule=policy).lower(Q[0])
    assert _makespan(low) == READOUT + 1 + BUDGET + ARM
    (call,) = low.classical_report()
    assert call.submit_cycles == READOUT and call.due_cycles == READOUT + 1 + BUDGET
    assert _events(low, "drive")[0].start_cycles == call.due_cycles


@pytest.mark.parametrize("policy", POLICIES)
def test_readout_latency_delays_the_submission(policy):
    machine = ControlSystem(controllers=[pulse.Controller(feedback_latency_cycles=50)])
    low = pulse.compile(feedback, schedule=policy, control_system=machine).lower(Q[0])
    (call,) = low.classical_report()
    assert call.submit_cycles == READOUT + 50


@pytest.mark.parametrize("feedback_ns", [0, 100, 250])
def test_feedback_latency_of_the_link_adds_to_the_budget(feedback_ns):
    machine = ControlSystem(links={"host": Link(feedback_ns=feedback_ns)})
    low = pulse.compile(feedback, control_system=machine).lower(Q[0])
    assert _makespan(low) == READOUT + 1 + BUDGET + math.ceil(feedback_ns * CLOCK) + ARM


@pulse.kernel
def answered_before_a_loop(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    n = decode.submit([readout(r, square(100, 0.05))]).result()[0]
    for _ in range(4):
        if readout(r, square(100, 0.05)):
            drive(d, square(40, 0.1))
        if n:
            drive(d, square(30, 0.1))
        wait(d, 200)


@pytest.mark.parametrize("schedule", POLICIES)
def test_an_answer_taken_before_a_loop_is_taken_before_its_first_iteration_tests_it(schedule):
    lowered = pulse.compile(answered_before_a_loop, schedule=schedule).lower(Q[0])
    events = lowered.export_to_controller()["events"]
    # Every iteration branches on the answer, the first at its due time.
    [answer] = [e for e in events if e["kind"] == "answer"]
    tested = [e for e in events if any(c["answers"] for c in e["conditions"])]
    assert len(tested) == 4
    assert min(e["start_cycles"] for e in tested) >= answer["end_cycles"]


def _late_consumer(fn):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        r = get_readout_line(q)
        drive(e, square(8000, 0.01))
        m = readout(r, square(READOUT, 0.05))
        if fn.submit([m]).result()[0]:
            drive(d, square(ARM, 0.1))

    return k


@pytest.mark.parametrize("policy", ["alap", "alap_rcp"])
def test_eager_submission_keeps_its_asap_time_under_alap(policy):
    low = pulse.compile(_late_consumer(decode), schedule=policy).lower(Q[0], Q[1])
    (call,) = low.classical_report()
    assert call.submit_cycles == READOUT or call.submit_cycles > READOUT
    readout_end = _events(low, "readout")[0].start_cycles + READOUT
    assert call.submit_cycles == readout_end
    # The conditional drive is pushed to the end, so the answer has slack.
    assert call.slack_cycles == 8000 - ARM - call.due_cycles


def test_lazy_submission_floats_to_just_in_time_under_alap():
    low = pulse.compile(_late_consumer(decode_lazy), schedule="alap").lower(Q[0], Q[1])
    (call,) = low.classical_report()
    assert call.slack_cycles == 0
    assert call.due_cycles == 8000 - ARM
    asap = pulse.compile(_late_consumer(decode_lazy), schedule="asap").lower(Q[0], Q[1])
    assert asap.classical_report()[0].submit_cycles == READOUT


def _rounds(n, fn=decode):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(n):
            m = readout(r, square(READOUT, 0.05))
            if fn.submit([m]).result()[0]:
                drive(d, square(ARM, 0.1))

    return k


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("rolled", [True, False])
def test_rounds_overlap_their_calls(policy, rolled):
    threshold = 0 if rolled else 10_000
    low = pulse.compile(_rounds(8), schedule=policy, unroll_threshold=threshold).lower(Q[0])
    calls = low.classical_report()
    assert len(calls) == 8
    assert [c.submit_cycles for c in calls] == [READOUT * (k + 1) for k in range(8)]
    # Eight requests are in flight at once on the default 64-slot link.
    in_flight = max(
        sum(c.submit_cycles <= t < c.due_cycles for c in calls)
        for t in (c.submit_cycles for c in calls)
    )
    assert in_flight == min(8, math.ceil((1 + BUDGET) / READOUT))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_link_slots_bound_requests_in_flight(policy, slots):
    machine = ControlSystem(links={"host": Link(slots=slots)})
    low = pulse.compile(
        _rounds(6), schedule=policy, control_system=machine, unroll_threshold=0
    ).lower(Q[0])
    calls = low.classical_report()
    busy = [(c.submit_cycles, c.due_cycles) for c in calls]
    for start, _ in busy:
        assert sum(s <= start < e for s, e in busy) <= slots
    # Request k waits for the answer of request k - slots.
    for k in range(slots, len(calls)):
        assert calls[k].submit_cycles >= calls[k - slots].due_cycles


def test_submissions_on_one_link_go_out_in_order():
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        futures = []
        for _ in range(5):
            futures.append(decode.submit([m]))
        for f in futures:
            f.result()

    machine = ControlSystem(links={"host": Link(fixed_ns=10)})
    calls = pulse.compile(k, control_system=machine).lower(Q[0]).classical_report()
    starts = [c.submit_cycles for c in calls]
    assert starts == sorted(starts)
    for a, b in zip(calls, calls[1:], strict=False):
        assert b.submit_cycles >= a.submit_cycles + a.submit_duration_cycles


def test_links_are_independent():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        a = tune.submit(0.1)
        b = calibrate.submit(0.2)
        drive(d, square(ARM, a.result() + b.result()))

    low = pulse.compile(k, schedule="asap").lower(Q[0])
    calls = low.classical_report()
    assert [c.submit_cycles for c in calls] == [0, 0]
    assert {c.link for c in calls} == {1, 2}


@pytest.mark.parametrize("policy", POLICIES)
def test_nested_branches_on_answers(policy):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        if decode.submit([m]).result()[0]:
            m2 = readout(r, square(READOUT, 0.05))
            if decode.submit([m2]).result()[0]:
                drive(d, square(ARM, 0.1))

    low = pulse.compile(k, schedule=policy).lower(Q[0])
    first, second = low.classical_report()
    assert second.submit_cycles >= first.due_cycles + READOUT
    assert _makespan(low) == 2 * (READOUT + 1 + BUDGET) + ARM


@pytest.mark.parametrize("policy", ["asap", "alap"])
@pytest.mark.parametrize("flatten", [False, True])
def test_a_block_returns_its_future(policy, flatten):
    @pulse.kernel(schedule="alap")
    def measure_and_submit(r):
        m = readout(r, square(READOUT, 0.05))
        return decode.submit([m])

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        answer = measure_and_submit(r)
        drive(d, square(ARM, 0.2))
        if answer.result()[0]:
            drive(d, square(ARM, 0.1))

    low = pulse.compile(k, schedule=policy, flatten_schedules=flatten).lower(Q[0])
    (call,) = low.classical_report()
    assert call.due_cycles == READOUT + 1 + BUDGET
    assert _makespan(low) == call.due_cycles + ARM


def test_timeout_budgets_the_timeout():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        ok, fix = decode.submit([m]).result(timeout=400 * pulse.ns)
        if ok:
            drive(d, square(ARM, 0.1))

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.due_cycles == READOUT + 1 + math.ceil(400 * CLOCK)


def test_blocking_waits_start_segments():
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(ARM, 0.1))
        x = calibrate.submit(0.5).result()
        drive(d, square(ARM, x))
        y = calibrate.submit(x).result()
        drive(d, square(ARM, y))

    for policy in ("asap", "alap"):
        low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
        calls = low.classical_report()
        assert [c.segment for c in calls] == [0, 1]
        # Nothing before the first answer ends after it.
        first_answer = calls[0].taken_cycles
        assert _events(low, "drive")[0].start_cycles + ARM <= first_answer


def _looped(shape: str, trips: int):
    """A loop of *trips* iterations that opens no segment ("none"), one
    followed by a blocking answer ("outside"), one that takes a blocking
    answer in every iteration ("inside"), one that holds both a loop that
    opens none and a blocking answer ("nested"), one whose only blocking
    answer is in a loop inside it ("inner"), and one that waits on each of
    two lines for the length an answer gives ("variable")."""

    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(ARM, 0.1))
        for _ in range(trips):
            drive(d, square(8, 0.1))
            wait(d, 16)
            if shape == "nested":
                for _ in range(3):
                    drive(e, square(8, 0.1))
                    wait(e, 4)
            if shape == "inner":
                for _ in range(3):
                    drive(e, square(8, calibrate.submit(0.5).result()))
            if shape in ("inside", "nested"):
                drive(d, square(8, calibrate.submit(0.5).result()))
            if shape == "variable":
                wait(d, tune.submit(8.0).result(), bounds=(0, 40))
                drive(d, square(8, 0.1))
                wait(e, tune.submit(4.0).result(), bounds=(0, 40))
                drive(e, square(8, 0.1))
        if shape != "inside":
            drive(d, square(ARM, tune.submit(0.5).result()))
        if shape in ("outside", "nested"):
            drive(d, square(ARM, calibrate.submit(0.25).result()))
        drive(e, square(ARM, 0.1))

    return k


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize(
    ("shape", "loops", "boundaries"),
    [
        ("none", 1, 0),
        ("outside", 1, 1),
        ("inside", 1, 5),
        ("nested", 2, 6),
        ("inner", 2, 15),
        ("variable", 1, 10),
    ],
)
def test_a_loop_kept_as_a_loop_has_the_segments_of_its_iterations_written_out(
    shape, loops, boundaries, schedule
):
    """The segments are asked for without walking the iterations of a loop
    that opens none: they are those of the same program with its loops
    written out, which has no loop to pass over."""
    trips = 5
    rolled, unrolled = (
        pulse.compile(_looped(shape, trips), schedule=schedule, unroll_threshold=threshold).lower(
            Q[0], Q[1]
        )
        for threshold in (0, 10**6)
    )
    # Under asap the loops stay as written (one may be peeled), a loop
    # inside a loop included; alap may write an inner one out.
    kept = rolled.mlir().count("scf.for")
    assert kept >= (loops if schedule == "asap" else 1)
    assert "scf.for" not in unrolled.mlir()
    segments, after = rolled._segments()
    assert len(segments) == 1 + boundaries
    assert (segments, after) == unrolled._segments()
    assert [s.index for s in segments] == list(range(len(segments)))
    assert after[0] is None and all(0 <= a < k for k, a in enumerate(after[1:], start=1))
    if shape == "variable":
        # Two lines that wait in turn: a segment follows the one its own
        # line was in, not the one opened just before it.
        assert any(a != k - 1 for k, a in enumerate(after[1:], start=1))
    # The walk that gives each event its segment visits every iteration,
    # and reaches the same last segment.
    assert max(rolled._event_segments(syncs=True)) == len(segments) - 1
    assert [c.segment for c in rolled.classical_report()] == [
        c.segment for c in unrolled.classical_report()
    ]
    exported = rolled.export_to_controller(version=2)
    assert exported["segments"] == unrolled.export_to_controller(version=2)["segments"]
    assert len(exported["segments"]) == len(segments)
    assert rolled.export_to_controller(version=1)["segments"] == exported["segments"]


def test_a_missed_deadline_is_rejected():
    @pulse.classical(max_latency_ns=1000, deadline_ns=800)
    def tight(b: pulse.Bits) -> pulse.Bits:
        return b

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        tight.submit([readout(r, square(READOUT, 0.05))]).result()

    with pytest.raises(
        RuntimeError, match=r"cannot meet its deadline.*slack -401\); raise deadline_ns"
    ):
        pulse.compile(k).lower(Q[0])


def test_deadline_slack_is_reported():
    @pulse.classical(max_latency_ns=1000, deadline_ns=1500)
    def loose(b: pulse.Bits) -> pulse.Bits:
        return b

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        loose.submit([readout(r, square(READOUT, 0.05))]).result()

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.deadline_slack_cycles == math.ceil(1500 * CLOCK) - (1 + BUDGET)


def test_a_request_too_big_for_its_slot_is_rejected():
    machine = ControlSystem(links={"host": Link(slot_bytes=24 + 0 + 1 - 1 + 1)})

    @pulse.classical(max_latency_ns=1)
    def big(b: pulse.Bits) -> None:
        return b

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(READOUT, 0.05))
        big.submit([m] * 16)

    with pytest.raises(RuntimeError, match="a request of 'big__16' takes 26 bytes"):
        pulse.compile(k, control_system=machine).lower(Q[0])


def _many(n, links):
    def same(x: float) -> float:
        return x

    functions = [
        pulse.classical(same, name=f"f{k}", max_latency_ns=20, link=k) for k in range(links)
    ]

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        for i in range(n):
            a = functions[i % links].submit(0.1).result()
            drive(d, square(8, a))

    return k


@pytest.mark.slow
@pytest.mark.parametrize("links", [1, 4])
def test_very_many_calls_schedule_in_linear_time(links):
    # The Python work is counted, the same at any load. The time, which also
    # takes in native code, is the best of five in CPU time: one wall-clock
    # sample per size failed on a loaded machine.
    runs, work = {}, {}
    for n in (250, 2000):
        kernel = _many(n, links)

        def run(kernel=kernel):
            return pulse.compile(kernel, schedule="asap").lower(Q[0])

        work[n] = python_work(run)
        runs[n] = run
        calls = run().classical_report()
        assert len(calls) == n
        # Each call waits for the previous drive; answers are due 41 later.
        assert calls[-1].due_cycles == calls[-1].submit_cycles + 1 + 40
    # 8x the calls: 8x the time when linear, 64x when quadratic.
    assert all(work[2000][count] <= 8 * work[250][count] for count in work[250]), work
    timings = cpu_seconds_each(runs)
    assert timings[2000] < 24 * timings[250], timings


@pulse.classical(max_latency_ns=200)
def first_guess() -> tuple[float, float]:
    return 0.0, 0.0


@pulse.classical(max_latency_ns=200)
def next_guess(bits: pulse.Bits) -> tuple[float, float]:
    return 0.0, 0.0


def _shots(n):
    @pulse.kernel
    def search(q):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        x, done = first_guess.submit().result()
        for _ in range(2):
            bits = []
            for _ in range(n):
                if done < 0.5:
                    shift_frequency(d.tone, x)
                    drive(d, square(64, 0.01))
                    shift_frequency(d.tone, -x)
                sync(d, ro)
                bits.append(readout(ro, square(40, 0.05)))
                sync(d, ro)
            x, done = next_guess.submit(bits).result()

    return search


@pytest.mark.slow
def test_shots_that_branch_on_one_answer_schedule_in_linear_time():
    # Every shot of a round branches on the answer the round before sent its
    # measurements to; the time, native, is of whole compiles (a program
    # lowers once, so lowering it again would time nothing).
    runs = {}
    for n in (100, 1600):
        kernel = _shots(n)
        runs[n] = lambda kernel=kernel: pulse.compile(kernel).lower(Q[0])
    timings = native_seconds_each(runs)
    # 16x the shots: 16x the time when linear, 256x when quadratic.
    assert timings[1600] < 64 * timings[100], timings


@pulse.classical(max_latency_ns=60, link="decoder", submit="lazy")
def record_syndrome(bits: pulse.Bits) -> pulse.Bits:
    return bits


@pulse.classical(max_latency_ns=120, link="decoder")
def decode_syndromes() -> pulse.Bits:
    return [0]


@pulse.kernel
def decoded(q, r):
    # The decoder keeps every round's syndrome: decode_syndromes() takes
    # none, and its answer depends on those of the calls before it on the
    # link alone.
    d = get_drive_line(q)
    ro = get_readout_line(r)
    for k in range(6):
        record_syndrome.submit([readout(ro, square(80, 0.05))])
        if k % 3 == 2 and decode_syndromes.submit().result()[0]:
            drive(d, square(16, 0.3))


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_correction_from_decoder_state_waits_for_every_syndrome_it_holds(schedule):
    machine = ControlSystem(links={"decoder": Link(fixed_ns=10)})
    lowered = pulse.compile(decoded, schedule=schedule, control_system=machine).lower(Q[0], Q[1])
    events = lowered.export_to_controller()["events"]
    corrections = [e for e in events if e["kind"] == "drive" and e["conditions"]]
    assert len(corrections) == 2
    for drive in corrections:
        [condition] = drive["conditions"]
        [answer] = condition["answers"]
        submit = max(e["index"] for e in events if e["kind"] == "submit" and e["index"] < answer)
        syndromes = [e for e in events if e["kind"] == "readout" and e["index"] < submit]
        assert syndromes
        assert drive["start_cycles"] >= max(e["end_cycles"] for e in syndromes)


# ── Properties ───────────────────────────────────────────────────────────────

_STEP = st.tuples(
    st.sampled_from(("readout", "submit", "wait", "drive", "drive_from_answer")),
    st.integers(0, 2),
    st.integers(0, 1),
)


def _program(steps):
    @pulse.kernel
    def k(q0, q1, q2):
        qs = (q0, q1, q2)
        ms = []
        futures = []
        answers = []
        for kind, q, link in steps:
            d = get_drive_line(qs[q])
            r = get_readout_line(qs[q])
            if kind == "readout":
                ms.append(readout(r, square(60, 0.05)))
            elif kind == "submit" and ms:
                fn = decode if link == 0 else decode_lazy
                futures.append(fn.submit([ms[-1]]))
            elif kind == "wait" and futures:
                answers.append(futures.pop(0).result())
            elif kind == "drive":
                drive(d, square(20, 0.1))
            elif kind == "drive_from_answer" and answers:
                if answers[-1][0]:
                    drive(d, square(20, 0.1))
        for f in futures:
            f.result()

    return k


@settings(max_examples=examples(40), deadline=None)
@given(steps=st.lists(_STEP, min_size=1, max_size=25), policy=st.sampled_from(POLICIES))
def test_scheduling_invariants(steps, policy):
    low = pulse.compile(_program(steps), schedule=policy).lower(Q[0], Q[1], Q[2])
    calls = low.classical_report()
    for call in calls:
        # S7: every answer is taken no earlier than it is due.
        assert call.taken_cycles is None or call.taken_cycles >= call.due_cycles
        assert call.first_use_cycles is None or call.first_use_cycles >= call.due_cycles
        assert call.due_cycles == call.submit_cycles + call.submit_duration_cycles + BUDGET
    # S9: one link sends one request at a time, in program order.
    starts = [c.submit_cycles for c in calls]
    assert starts == sorted(starts)
    # No two pulses overlap on a line, and the makespan covers everything.
    by_line: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for e in low.schedule:
        assert e.start_cycles >= 0
        if e.link is None:
            by_line.setdefault(("line", e.line), []).append((e.start_cycles, e.duration_cycles))
    for spans in by_line.values():
        spans.sort()
        for (s0, d0), (s1, _) in zip(spans, spans[1:], strict=False):
            assert s0 + d0 <= s1
    if calls:
        assert _makespan(low) >= max(c.taken_cycles or 0 for c in calls)


# ── Regressions ──────────────────────────────────────────────────────────────


def test_a_helper_passed_a_future_takes_its_answer():
    def answer(future):
        return future.result()[0]

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        if answer(decode.submit([readout(r, square(READOUT, 0.05))])):
            drive(d, square(ARM, 0.1))

    (call,) = pulse.compile(k).lower(Q[0]).classical_report()
    assert call.first_use_cycles == call.due_cycles


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_blocking_wait_waits_for_the_lines_it_feeds(policy, tied):
    @pulse.kernel
    def k(q, p):
        d = get_drive_line(q)
        e = get_drive_line(p)
        drive(e, square(8000, 0.01))
        if tied:
            sync(d, e)
        x = calibrate.submit(0.5).result()
        drive(d, square(ARM, x))

    low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
    (call,) = low.classical_report()
    fed = [e for e in _events(low, "drive") if e.line == 0]
    if tied:
        assert call.taken_cycles == 8000
        assert fed[0].start_cycles == 8000
    else:
        # Only the line that uses the answer waits for it.
        assert call.taken_cycles == call.due_cycles < 8000
        assert fed[0].start_cycles == call.due_cycles


@pytest.mark.parametrize("callee", ["asap", "alap"])
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_block_waits_for_a_slot_the_caller_holds(callee, policy):
    @pulse.kernel(schedule=callee)
    def twice(r):
        m = readout(r, square(10, 0.05))
        a = decode.submit([m])
        b = decode.submit([m])
        a.result()
        b.result()

    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        f = decode.submit([readout(r, square(10, 0.05))])
        twice(r)
        f.result()

    machine = ControlSystem(links={"host": Link(slots=2)})
    calls = pulse.compile(k, schedule=policy, control_system=machine).lower(Q[0]).classical_report()
    first, *block = calls
    # The block's second request needs the slot the first call frees.
    assert block[1].submit_cycles == first.due_cycles
    assert block[0].submit_cycles == block[1].submit_cycles - 1


@pytest.mark.parametrize("passes", [None, ()])
def test_a_blocking_wait_in_a_block_is_rejected_without_the_verifier(passes):
    @pulse.kernel(schedule="asap")
    def inner(d):
        drive(d, square(ARM, calibrate.submit(0.5).result()))

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        inner(d)

    with pytest.raises(RuntimeError, match="return the future from the kernel and take the answer"):
        pulse.compile(k, passes=passes).lower(Q[0])


@pytest.mark.parametrize("callee", ["asap", "alap"])
@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_a_future_taken_with_a_timeout_in_a_block_budgets_the_timeout(callee, policy):
    @pulse.kernel(schedule=callee)
    def take(future, d, t):
        ok, fix = future.result(timeout=400 * pulse.ns)
        if fix[0]:
            drive(d, square(ARM, 0.1), tone=t)

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        take(decode.submit([readout(r, square(READOUT, 0.05))]), d, d.tone)

    (call,) = pulse.compile(k, schedule=policy).lower(Q[0]).classical_report()
    assert call.due_cycles == READOUT + 1 + math.ceil(400 * CLOCK)
    assert call.taken_cycles >= call.due_cycles and call.first_use_cycles >= call.due_cycles


# ── A controller that blocks on calls ────────────────────────────────────────

_BLOCKING = ControlSystem(controllers=[pulse.Controller(blocks_on_calls=True)])


@pulse.kernel
def _blocked_shots(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    flip = False
    for shot in range(3):
        drive(d, square(ARM, 0.25))
        sync(d, r)
        m = readout(r, square(READOUT, 0.05))
        if shot:
            if flip:
                drive(d, square(ARM, 0.5))
        flip = decode.submit([m]).result()[0]


def _held(low):
    """The drives and readouts that play while a call is out."""
    played = [e for e in low.schedule if e.kind in ("drive", "readout")]
    return [
        (e.kind, e.start_cycles)
        for call in low.classical_report()
        for e in played
        if e.start_cycles < call.due_cycles and e.end > call.submit_cycles
    ]


@pytest.mark.parametrize("policy", POLICIES)
def test_a_controller_that_blocks_on_calls_plays_nothing_while_one_is_out(policy):
    plain = pulse.compile(_blocked_shots, schedule=policy).lower(Q[0])
    assert _held(plain)
    low = pulse.compile(_blocked_shots, schedule=policy, control_system=_BLOCKING).lower(Q[0])
    assert _held(low) == []
    played = [e for e in low.schedule if e.kind in ("drive", "readout")]
    for call in low.classical_report():
        # Every line is done before a call goes out.
        assert all(e.end <= call.submit_cycles for e in played if e.start < call.submit_cycles)
    # As one Controller alone.
    alone = pulse.compile(_blocked_shots, schedule=policy, control_system=_BLOCKING.controllers[0])
    assert alone.lower(Q[0]).schedule == low.schedule


def test_a_program_scheduled_for_another_controller_fails_the_check():
    from cudaq_pulse import controller

    blocking = pulse.Controller(blocks_on_calls=True)
    plain = pulse.compile(_blocked_shots, schedule="asap").lower(Q[0])
    found = controller.check(plain, blocking)
    first = plain.classical_report()[0]
    assert found[0] == (
        f"line 1: the readout at {first.submit_cycles} plays while the call to decode__1 sent at "
        f"{first.submit_cycles} is out, until its answer is due at {first.due_cycles}; the "
        "controller blocks on calls (Controller.blocks_on_calls)"
    )
    blocked = pulse.compile(_blocked_shots, schedule="asap", control_system=_BLOCKING).lower(Q[0])
    assert controller.check(blocked, blocking) == []


@pulse.kernel
def _rolled(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    for _ in range(40):
        drive(d, square(ARM, 0.1))
        sync(d, r)
        m = readout(r, square(READOUT, 0.05))
        decode.submit([m]).result()


@pytest.mark.parametrize("policy", POLICIES)
def test_a_rolled_loop_of_calls_holds_the_lines_every_iteration(policy):
    rolled = pulse.compile(_rolled, schedule=policy, control_system=_BLOCKING, unroll_threshold=0)
    unrolled = pulse.compile(_rolled, schedule=policy, control_system=_BLOCKING)
    low = rolled.lower(Q[0])
    assert low.schedule == unrolled.lower(Q[0]).schedule
    assert _held(low) == [] and len(low.classical_report()) == 40


@pulse.kernel
def _armed(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(READOUT, 0.05))
    if m:
        drive(d, square(ARM, 0.1))
        decode.submit([m]).result()
        drive(d, square(ARM, 0.2))
    else:
        drive(d, square(ARM, 0.3))
    drive(d, square(ARM, 0.4))


@pytest.mark.parametrize("policy", POLICIES)
def test_a_call_in_one_arm_holds_the_lines_in_it_and_after_it(policy):
    low = pulse.compile(_armed, schedule=policy, control_system=_BLOCKING).lower(Q[0])
    (call,) = low.classical_report()
    drives = [e for e in low.schedule if e.kind == "drive"]
    before, after, other, last = drives
    assert before.end <= call.submit_cycles and after.start >= call.due_cycles
    assert last.start >= call.due_cycles


def test_a_call_in_a_kernel_with_its_own_schedule_does_not_compile():
    @pulse.kernel(schedule="asap")
    def inner(r):
        decode.submit([readout(r, square(READOUT, 0.05))]).result()

    @pulse.kernel
    def outer(q):
        inner(get_readout_line(q))

    with pytest.raises(RuntimeError, match="the controller blocks on calls"):
        pulse.compile(outer, control_system=_BLOCKING).lower(Q[0])
    pulse.compile(outer, control_system=_BLOCKING, flatten_schedules=True).lower(Q[0])


def test_blocking_on_calls_is_exported_and_kept_in_system_files():
    blocking = pulse.Controller(blocks_on_calls=True)
    assert blocking.as_dict()["blocks_on_calls"] is True
    assert "blocks_on_calls" not in pulse.Controller().as_dict()
    assert pulse.Controller(**blocking.as_dict()) == blocking
    low = pulse.compile(_blocked_shots, control_system=blocking).lower(Q[0])
    assert low.export_to_controller()["limits"]["blocks_on_calls"] is True
    assert low.export_to_controller(version=2)["controller"]["blocks_on_calls"] is True
    with pytest.raises(pulse.InvalidValueError, match="blocks_on_calls must be a bool"):
        pulse.Controller(blocks_on_calls=1)  # type: ignore[arg-type]


def _one_each(*blocking):
    """A controller per qudit 0, 1 and 2; those in *blocking* block on calls."""
    return ControlSystem(
        controllers=[
            pulse.Controller(drive_lines=[q], readout_lines=[q], blocks_on_calls=q in blocking)
            for q in range(3)
        ]
    )


_LONG = 3 * BUDGET


@pulse.kernel
def _across(q0, q1, q2):
    # Qudit 0's controller reads the measurement the call sends and qudit
    # 1's branches on its answer; qudit 2's takes no part in it.
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    d1 = get_drive_line(q1)
    d2 = get_drive_line(q2)
    drive(d1, square(_LONG, 0.1))
    drive(d2, square(_LONG, 0.1))
    m = readout(r0, square(READOUT, 0.05))
    flip = decode.submit([m]).result()[0]
    drive(d0, square(_LONG, 0.2))
    if flip:
        drive(d1, square(ARM, 0.5))


def _playing(low):
    """The qudits whose drives or readouts play while a call is out."""
    return {
        e.qudit
        for call in low.classical_report()
        for e in low.schedule
        if e.kind in ("drive", "readout")
        and e.start_cycles < call.due_cycles
        and e.end > call.submit_cycles
    }


def _across_on(policy, *blocking):
    system = _one_each(*blocking)
    return pulse.compile(_across, schedule=policy, control_system=system).lower(Q[0], Q[1], Q[2])


@pytest.mark.parametrize("policy", POLICIES)
def test_a_call_holds_the_blocking_controller_that_reads_its_measurement(policy):
    assert _playing(_across_on(policy)) == {0, 1, 2}
    playing = _playing(_across_on(policy, 0))
    assert 0 not in playing
    if policy in ("asap", "rcp"):
        assert playing == {1, 2}


@pytest.mark.parametrize("policy", POLICIES)
def test_a_call_holds_the_blocking_controller_that_takes_its_answer(policy):
    low = _across_on(policy, 1)
    (call,) = low.classical_report()
    # The call waits for qudit 1's drive before it.
    assert call.submit_cycles >= _LONG
    playing = _playing(low)
    assert 1 not in playing
    if policy in ("alap", "alap_rcp"):
        assert playing == {0, 2}


@pytest.mark.parametrize("policy", POLICIES)
def test_a_call_holds_both_the_controller_it_reads_and_the_one_it_answers(policy):
    assert _playing(_across_on(policy, 0, 1)) & {0, 1} == set()
    # A controller that blocks on calls but takes no part in one plays on.
    assert _playing(_across_on(policy, 2)) == _playing(_across_on(policy))


@pulse.kernel
def _issued(q0, q1, q2):
    # The call sends no measurement and its answer goes unused: it holds
    # only the controllers that issue it, every one that plays, over a
    # shared link.
    d1 = get_drive_line(q1)
    d2 = get_drive_line(q2)
    drive(d1, square(_LONG, 0.1))
    drive(d2, square(_LONG, 0.1))
    tune.submit(0.5).result()
    drive(d2, square(_LONG, 0.2))


@pytest.mark.parametrize("policy", POLICIES)
def test_a_call_holds_the_blocking_controller_that_issues_it(policy):
    from cudaq_pulse.controller import _compile_check

    args = (Q[0], Q[1], Q[2])
    plain = pulse.compile(_issued, schedule=policy, control_system=_one_each()).lower(*args)
    assert 2 in _playing(plain)
    found, _ = _compile_check(plain, None, _one_each(2))
    assert found and {v.split(":")[0] for v in found} == {"controllers[2]"}
    held = pulse.compile(_issued, schedule=policy, control_system=_one_each(2)).lower(*args)
    assert 2 not in _playing(held)


@pulse.kernel
def _rolled_across(q0, q1, q2):
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    for _ in range(5):
        drive(d1, square(_LONG, 0.1))
        if decode.submit([readout(r0, square(READOUT, 0.05))]).result()[0]:
            drive(d1, square(ARM, 0.5))


@pytest.mark.parametrize("policy", POLICIES)
def test_the_check_names_each_controller_a_call_holds(policy):
    from cudaq_pulse.controller import _compile_check

    plain = _across_on(policy)
    (call,) = plain.classical_report()
    found, _ = _compile_check(plain, None, _one_each(0, 1))
    assert {v.split(":")[0] for v in found} == {"controllers[0]", "controllers[1]"}
    assert all(
        f"plays while the call to decode__1 sent at {call.submit_cycles} is out" in v for v in found
    )
    assert _compile_check(plain, None, _one_each(2))[0] == []
    # Every iteration of a rolled loop, rolled as unrolled.
    args = (Q[0], Q[1], Q[2])
    rolled = pulse.compile(
        _rolled_across, schedule=policy, control_system=_one_each(), unroll_threshold=0
    )
    assert len(_compile_check(rolled.lower(*args), None, _one_each(1))[0]) == 5
    held = pulse.compile(
        _rolled_across, schedule=policy, control_system=_one_each(1), unroll_threshold=0
    ).lower(*args)
    unrolled = pulse.compile(_rolled_across, schedule=policy, control_system=_one_each(1))
    assert held.schedule == unrolled.lower(*args).schedule
    assert 1 not in _playing(held)


def test_who_takes_an_answer_is_found_along_a_line_of_any_length():
    # A branch on an answer gives the lines its arms touch, and finding the
    # controllers that take the answer follows each of those lines through
    # every sync and branch after it. That was a call for every op on the
    # line: about 34,000 of them ended the process (a segmentation fault).
    from cudaq_pulse.quantum_systems import PhysicalQudit

    @pulse.classical(max_latency_ns=60)
    def decide(bits: pulse.Bits) -> pulse.Bits:
        return [bits[0]]

    @pulse.kernel
    def long_line(qs):
        d = [get_drive_line(qs[i]) for i in range(4)]
        r = [get_readout_line(qs[i]) for i in range(4)]
        m = readout(r[0], square(32, 0.05))
        if decide.submit([m]).result()[0]:
            sync(r[1], d[0])
        else:
            sync(r[1], d[3])
        for _ in range(60_000):
            sync(r[1], d[3])
            wait(d[3], 4)

    # Who takes an answer matters on several controllers only.
    controllers = [
        pulse.Controller(clock_ghz=2.0, drive_lines=lines, readout_lines=lines)
        for lines in ([0, 1], [2, 3])
    ]
    system = pulse.quantum_system(
        name="long_line",
        qudits={i: PhysicalQudit(i, 4.5e9 + 1.0e5 * i, 37.7, 9.2) for i in range(4)},
        control_system=ControlSystem(controllers=controllers, reference_clock_ghz=2.0),
        clock_ghz=2.0,
    )
    lowered = pulse.compile(long_line, quantum_system=system, unroll_threshold=10**6).lower(
        system.qudits[0:4]
    )
    assert sum(1 for event in lowered.schedule if event.kind == "wait") == 60_000
