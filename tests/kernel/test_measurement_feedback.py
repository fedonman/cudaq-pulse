# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-conditioned feedback: retained as an if region end to end."""

from __future__ import annotations

import collections
import random

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError, Parameter

from tests._handles import Q
from tests._ir import python_ir
from tests._work import python_work
from tests.kernel.ir_corpus import active_reset, feedback_in_loop, negated_feedback


def _compile(kernel, qudit_count=1, **kwargs):
    return pulse.compile(kernel, **kwargs).lower(*[Q[i] for i in range(qudit_count)])


def _makespan(compiled):
    return max(e.start_cycles + e.duration_cycles for e in compiled.schedule)


def test_active_reset_retains_the_conditional():
    ir = python_ir(active_reset, Q[0])
    kinds = [op.kind for op in ir.ops]
    start = kinds.index("scf.if")
    assert kinds[start:] == [
        "scf.if",
        "pulse.drive",
        "scf.yield",
        "scf.else",
        "scf.yield",
        "scf.if_end",
        "pulse.wait",
    ]
    cond = ir.ops[start].operands[0]
    assert cond.vtype == "measurement"
    end = ir.ops[kinds.index("scf.if_end")]
    assert end.attrs == {"result_names": ["d0", "d0.tone"]}
    # The wait after the if continues the line the if produced.
    assert ir.ops[-1].operands[0] is end.results[0]


def test_negation_swaps_the_arms():
    ir = python_ir(negated_feedback, Q[0], Q[1])
    kinds = [op.kind for op in ir.ops]
    then_arm = kinds[kinds.index("scf.if") + 1 : kinds.index("scf.else")]
    else_arm = kinds[kinds.index("scf.else") + 1 : kinds.index("scf.if_end")]
    assert "pulse.wait" in then_arm and "pulse.drive" not in then_arm
    assert "pulse.drive" in else_arm and "pulse.shift_phase" in else_arm


def test_compiled_feedback_verifies():
    for kernel, n in ((active_reset, 1), (negated_feedback, 2), (feedback_in_loop, 1)):
        compiled = _compile(kernel, n)
        assert "pulse.is_set" in compiled.mlir()
        compiled._module.run_passes(["pulse-verify"])


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_schedule_allows_for_the_longer_arm(schedule):
    # readout (300) then the conditional X (40) and a wait (10)
    assert _makespan(_compile(active_reset, schedule=schedule)) == 300 + 40 + 10
    # else arm (wait 40 on d0) vs then arm (shift + 40 on d1); both start
    # after the readout, then a 20 drive on d0.
    assert _makespan(_compile(negated_feedback, 2, schedule=schedule)) == 300 + 40 + 20


def test_conditional_waits_for_the_measurement():
    events = _compile(active_reset, schedule="asap").schedule
    readout = next(e for e in events if e.kind == "readout")
    drive = next(e for e in events if e.kind == "drive")
    assert drive.start_cycles >= readout.start_cycles + readout.duration_cycles


def test_feedback_inside_a_retained_loop():
    ir = python_ir(feedback_in_loop, Q[0])
    kinds = [op.kind for op in ir.ops]
    assert kinds.count("scf.for") == 1 and kinds.count("scf.if") == 1
    compiled = _compile(feedback_in_loop, schedule="asap")
    assert compiled.mlir().count("scf.for") == 1
    # Each readout may start while the previous iteration's conditional
    # drive is still playing on the drive line.
    assert _makespan(compiled) == 5 * 300 + 40


def test_python_state_cannot_depend_on_a_measurement():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05))
        n = [1]
        if bit:
            n = [2]
        wait(d0, 10 * n[0])

    # Numbers are chosen (test_measurement_choice.py); other Python values are not.
    with pytest.raises(CompilationError, match="'n' would depend on a measurement result"):
        k(Q[0])


def test_one_armed_return_is_rejected():

    @pulse.kernel(unroll_threshold=0)
    def k(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        bit = readout(r0, square(300, 0.05))
        for _ in range(2):
            if bit:
                return
            wait(d0, 10)

    with pytest.raises(CompilationError, match="must continue after it"):
        k(Q[0])


def test_break_out_of_an_arm_is_rejected():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        for i in range(3):
            bit = readout(r0, square(300, 0.05))
            if bit:
                break
            wait(d0, 10 + i)

    with pytest.raises(CompilationError, match="must continue after it"):
        k(Q[0])


@pytest.mark.parametrize("condition", ["amp == 0.5", "amp != 0.5", "amp < 1", "0 <= amp"])
def test_symbolic_comparison_raises(condition):
    source = (
        f"def k(q0, amp):\n    d0 = get_drive_line(q0)\n    if {condition}:\n        wait(d0, 10)\n"
    )
    namespace: dict = {}
    exec(compile(source, "<symbolic>", "exec"), namespace)
    kernel = pulse.kernel(namespace["k"])
    with pytest.raises(CompilationError, match="cannot decide"):
        kernel(Q[0], Parameter("amp", 0, "f64"))


def test_comparing_a_measurement_raises():

    @pulse.kernel
    def k(q0):
        r0 = get_readout_line(q0)
        bit = readout(r0, square(10, 0.2))
        if bit == 0:
            wait(r0, 10)

    with pytest.raises(CompilationError, match=r"branch on it directly with `if result:`"):
        k(Q[0])


def test_only_measurements_can_be_conditions():

    @pulse.kernel
    def k(q0):
        d0 = get_drive_line(q0)
        if d0:
            wait(d0, 10)

    with pytest.raises(CompilationError, match="only measurement results"):
        k(Q[0])


_HELD_HEAD = """\
def k(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    t0 = d0.tone
    r = get_readout_line(q0)
"""
_HELD_READ = "    m = readout(r, square(100, 0.05))\n    sync(d0, d1, r)\n"
_ARM = "square(20, 0.1)"
_AFTER = "square(8, 0.1)"

# (lines before the readout, the rest of the body, the body with plain names)
_HELD = {
    "unused list": (
        "    lines = [d0, d1]\n",
        f"    if m:\n        drive(d0, {_ARM})\n        drive(d1, {_ARM})\n",
        None,
    ),
    "else arm": (
        "    lines = [d0, d1]\n",
        f"    if m:\n        pass\n    else:\n        drive(lines[0], {_ARM})\n"
        f"    drive(lines[0], {_AFTER})\n",
        f"    if m:\n        pass\n    else:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "both arms": (
        "    lines = [d0, d1]\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(lines[1], {_ARM})\n"
        f"    drive(lines[0], {_AFTER})\n    drive(d1, {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n    drive(d1, {_AFTER})\n",
    ),
    "one line twice": (
        "    lines = [d0, d0]\n",
        f"    if m:\n        drive(lines[0], {_ARM})\n    drive(lines[1], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "two lists and an alias": (
        "    e = d0\n    xs = [e]\n    ys = [d1, d0]\n",
        f"    if m:\n        drive(e, {_ARM})\n    else:\n        drive(ys[1], {_ARM})\n"
        f"    drive(xs[0], {_AFTER})\n    drive(ys[0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d0, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n    drive(d1, {_AFTER})\n",
    ),
    "only the list": (
        "    lines = [d0, d1]\n",
        f"    del d0\n    if m:\n        drive(lines[0], {_ARM})\n    drive(lines[0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "tuple": (
        "    lines = (d0, d1)\n",
        f"    if m:\n        drive(lines[0], {_ARM})\n    drive(lines[0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    drive(d0, {_AFTER})\n",
    ),
    "dict": (
        "    lines = {'a': d0, 'b': d1}\n",
        f"    if m:\n        drive(lines['a'], {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(lines['a'], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n",
    ),
    "nested containers": (
        "    lines = {'a': [d0, (d1,)], 'b': [[d0]]}\n",
        f"    if m:\n        drive(lines['b'][0][0], {_ARM})\n"
        f"    else:\n        drive(lines['a'][1][0], {_ARM})\n"
        f"    drive(lines['a'][0], {_AFTER})\n    drive(lines['a'][1][0], {_AFTER})\n",
        f"    if m:\n        drive(d0, {_ARM})\n    else:\n        drive(d1, {_ARM})\n"
        f"    drive(d0, {_AFTER})\n    drive(d1, {_AFTER})\n",
    ),
    "tones": (
        "    tones = [t0]\n    lines = [d0]\n",
        f"    if m:\n        drive(lines[0], {_ARM}, tone=tones[0])\n"
        f"    drive(d0, {_AFTER}, tone=tones[0])\n",
        f"    if m:\n        drive(d0, {_ARM}, tone=t0)\n    drive(d0, {_AFTER}, tone=t0)\n",
    ),
    "nested ifs": (
        "    lines = [d0, d1]\n",
        "    m2 = readout(r, square(100, 0.05))\n"
        f"    if m:\n        drive(lines[0], {_ARM})\n"
        f"        if m2:\n            drive(lines[1], {_ARM})\n"
        f"    drive(lines[1], {_AFTER})\n",
        "    m2 = readout(r, square(100, 0.05))\n"
        f"    if m:\n        drive(d0, {_ARM})\n        if m2:\n            drive(d1, {_ARM})\n"
        f"    drive(d1, {_AFTER})\n",
    ),
}


def _get_n() -> int:
    return 5


def _held_kernel(before, rest, tag, condition="m"):
    if condition != "m":
        rest = "    n = get_n.submit().result()\n" + rest.replace(
            "    if m:", f"    if {condition}:"
        )
    source = _HELD_HEAD + before + _HELD_READ + rest
    scope = vars(pulse) | {"__name__": tag, "get_n": pulse.classical(_get_n, max_latency_ns=0)}
    namespace: dict = {}
    exec(compile(source, f"<{tag}>", "exec"), scope, namespace)
    return pulse.kernel(namespace["k"])


def _yields_once(kernel):
    for op in python_ir(kernel, Q[0], Q[1]).ops:
        if op.kind in ("scf.yield", "scf.if_end", "scf.for_end"):
            vids = [v.vid for v in (op.operands or op.results)]
            assert len(vids) == len(set(vids)), op


@pytest.mark.parametrize("condition", ["m", "n > 3"])
@pytest.mark.parametrize("case", list(_HELD))
def test_lines_held_in_containers_cross_a_runtime_branch(case, condition):
    before, rest, plain = _HELD[case]
    held = _held_kernel(before, rest, case, condition)
    if condition == "m":  # the Python IR builder makes no classical calls
        _yields_once(held)
    compiled = _compile(held, 2, schedule="asap")
    compiled._module.run_passes(["pulse-verify"])
    reference = _compile(_held_kernel("", plain or rest, case, condition), 2, schedule="asap")
    assert compiled.schedule == reference.schedule


@pytest.mark.parametrize("unroll_threshold", [64, 0])
def test_lines_held_in_a_list_cross_a_branch_in_a_loop(unroll_threshold):
    loop = (
        "    for _ in range(3):\n"
        "        m = readout(r, square(100, 0.05))\n"
        "        sync(d0, d1, r)\n"
        "        if m:\n"
        "            drive({0}, square(20, 0.1))\n"
        "        else:\n"
        "            drive({1}, square(30, 0.1))\n"
        "    drive({0}, square(8, 0.1))\n"
    )
    held = _held_kernel("    lines = [d0, d1]\n", loop.format("lines[0]", "lines[1]"), "loop")
    plain = _held_kernel("", loop.format("d0", "d1"), "loop")
    _yields_once(held)
    compiled = _compile(held, 2, schedule="asap", unroll_threshold=unroll_threshold)
    compiled._module.run_passes(["pulse-verify"])
    assert compiled.mlir().count("scf.for") == (unroll_threshold == 0)
    reference = _compile(plain, 2, schedule="asap", unroll_threshold=unroll_threshold)
    assert compiled.schedule == reference.schedule


_ROUNDS = (0, 1)


def test_nested_feedback_ending_an_unrolled_loop_body():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(10, 0.05))
        for _ in _ROUNDS:
            if m:
                if m:
                    drive(d, square(10, 0.1))

    events = k.lower(Q[0]).schedule
    assert [e.kind for e in events] == ["readout", "drive", "drive"]


class _Outcomes(list):
    pass


class _Record(dict):
    pass


# What a kernel keeps its outcomes in, and how it adds one.
_KEPT = {
    "list": (list, "append"),
    "list subclass": (_Outcomes, "append"),
    "dict subclass": (_Record, "setdefault"),
    "set": (set, "add"),
    "deque": (collections.deque, "append"),
}


def _rounds(n, kept="list"):
    make, add = _KEPT[kept]

    @pulse.kernel(unroll_threshold=10**9)
    def rounds(q):
        d = get_drive_line(q)  # noqa: F821
        r = get_readout_line(q)  # noqa: F821
        outcomes = make()
        for k in range(n):
            m = readout(r, square(40, 0.05))  # noqa: F821
            sync(d, r)  # noqa: F821
            if m:
                drive(d, square(16, 0.1))  # noqa: F821
            getattr(outcomes, add)(k)

    return rounds


@pytest.mark.slow
@pytest.mark.parametrize("kept", list(_KEPT))
def test_branches_beside_a_growing_container_of_outcomes_trace_in_linear_work(kept):
    # An arm that leaves a container as it was costs nothing for its length.
    work = {
        n: python_work(lambda n=n: pulse.compile(_rounds(n, kept)).lower(Q[0])) for n in (50, 200)
    }
    assert all(work[200][count] <= 4 * work[50][count] for count in work[50]), work


def _reset_each(n):
    @pulse.kernel
    def reset_each(qudits):
        lines = [get_drive_line(q) for q in qudits]
        probes = [get_readout_line(q) for q in qudits]
        bits = []
        for k in range(n):
            sync(lines[k], probes[k])
            bits.append(readout(probes[k], square(40, 0.05)))
        for k in range(n):
            if bits[k]:
                drive(lines[k], square(16, 0.1))

    return reset_each


@pytest.mark.slow
def test_a_branch_on_each_of_many_qudits_traces_in_linear_work():
    # Each arm drives one of the lines a list holds: a branch saves, restores
    # and merges what its arms change, not every line and tone, nor the list.
    work = {
        n: python_work(lambda n=n: pulse.compile(_reset_each(n)).lower(Q[0:n])) for n in (32, 256)
    }
    assert all(work[256][count] <= 8 * work[32][count] for count in work[32]), work


# Loops the trace tries to roll: each iteration moves a name to the other
# line, and an arm reads it before it is rebound, so rolling stops inside the
# if and the loop is unrolled. An if around the loop goes on tracing its arm
# after that attempt.
@pulse.kernel
def _swap_in_loop(q0, q1):
    a = get_drive_line(q0)
    b = get_drive_line(q1)
    p = get_readout_line(q0)
    m = readout(p, square(40, 0.05))
    for _ in range(40):
        if m:
            drive(a, square(16, 0.1))
        a, b = b, a


@pulse.kernel
def _swap_in_loop_in_if(q0, q1):
    a = get_drive_line(q0)
    b = get_drive_line(q1)
    p = get_readout_line(q0)
    r = get_readout_line(q1)
    m0 = readout(p, square(40, 0.05))
    m1 = readout(r, square(40, 0.05))
    if m0:
        drive(a, square(8, 0.2))
        for _ in range(40):
            if m1:
                drive(a, square(16, 0.1))
            a, b = b, a
        drive(b, square(8, 0.3))
    else:
        drive(b, square(8, 0.4))
    drive(a, square(8, 0.5))
    drive(b, square(8, 0.6))


@pulse.kernel
def _swap_list_in_loop_in_if(q0, q1, q2):
    lines = [get_drive_line(q0), get_drive_line(q1), get_drive_line(q2)]
    p = get_readout_line(q0)
    m0 = readout(p, square(40, 0.05))
    m1 = readout(p, square(40, 0.05))
    x = lines[0]
    y = lines[1]
    if m0:
        for _ in range(40):
            if m1:
                drive(x, square(16, 0.1))
            x, y = y, x
        drive(lines[2], square(8, 0.3))
    drive(x, square(8, 0.5))
    drive(lines[2], square(8, 0.7))


def test_a_loop_that_stops_rolling_inside_an_arm_traces_as_unrolled():
    for kernel, n in ((_swap_in_loop, 2), (_swap_in_loop_in_if, 2), (_swap_list_in_loop_in_if, 3)):
        compiled = _compile(kernel, n, schedule="asap")
        compiled._module.run_passes(["pulse-verify"])
        assert "scf.for" not in compiled.mlir(), kernel
        unrolled = _compile(kernel, n, schedule="asap", unroll_threshold=10**9)
        assert compiled.schedule == unrolled.schedule, kernel


def test_a_line_handed_out_again_and_driven_in_an_arm_is_carried_out_of_the_if():
    # get_drive_line(q0) in the arm returns the line d0 holds, which is shared
    # from then on: its drive there still makes the line a result of the if.
    @pulse.kernel
    def again(q0, q1):
        d0 = get_drive_line(q0)
        r = get_readout_line(q1)
        m = readout(r, square(40, 0.05))
        if m:
            line = get_drive_line(q0)
            drive(line, square(16, 0.1))
        else:
            drive(d0, square(24, 0.1))
        drive(d0, square(8, 0.2))

    @pulse.kernel
    def named(q0, q1):
        d0 = get_drive_line(q0)
        r = get_readout_line(q1)
        m = readout(r, square(40, 0.05))
        if m:
            drive(d0, square(16, 0.1))
        else:
            drive(d0, square(24, 0.1))
        drive(d0, square(8, 0.2))

    _yields_once(again)
    end = next(op for op in python_ir(again, Q[0], Q[1]).ops if op.kind == "scf.if_end")
    assert end.attrs == {"result_names": ["d0", "d0.tone"]}
    compiled = _compile(again, 2, schedule="asap")
    compiled._module.run_passes(["pulse-verify"])
    assert compiled.schedule == _compile(named, 2, schedule="asap").schedule


def test_a_line_first_acquired_in_an_arm_is_not_carried_out_of_the_if():
    # q1's line is acquired in the arm that plays on it: the if carries d0
    # only, and the schedule is the one of acquiring the line before the if.
    @pulse.kernel
    def in_arm(q0, q1):
        d0 = get_drive_line(q0)
        r = get_readout_line(q0)
        m = readout(r, square(40, 0.05))
        if m:
            d1 = get_drive_line(q1)
            drive(d1, square(16, 0.1))
            sync(d1, d0)
            drive(d0, square(8, 0.3))
        drive(d0, square(8, 0.2))

    @pulse.kernel
    def before(q0, q1):
        d0 = get_drive_line(q0)
        r = get_readout_line(q0)
        d1 = get_drive_line(q1)
        m = readout(r, square(40, 0.05))
        if m:
            drive(d1, square(16, 0.1))
            sync(d1, d0)
            drive(d0, square(8, 0.3))
        drive(d0, square(8, 0.2))

    end = next(op for op in python_ir(in_arm, Q[0], Q[1]).ops if op.kind == "scf.if_end")
    assert end.attrs == {"result_names": ["d0", "d0.tone"]}
    compiled = _compile(in_arm, 2, schedule="asap")
    compiled._module.run_passes(["pulse-verify"])
    assert compiled.schedule == _compile(before, 2, schedule="asap").schedule


def _traced_lines(monkeypatch, kernel, qudit_count):
    """The channels of each trace compiling *kernel* makes (``_Lines``)."""
    from cudaq_pulse.kernel import _emitter

    made = []

    class Recorded(_emitter._Lines):
        __slots__ = ()

        def __init__(self):
            super().__init__()
            made.append(self)

    monkeypatch.setattr(_emitter, "_Lines", Recorded)
    _compile(kernel, qudit_count)
    return made


def test_a_trace_without_a_measurement_conditioned_if_notes_no_channel(monkeypatch):
    # The branch journal is only for an if's arms: a trace without one gives
    # no channel a place and never makes its channels ones that note.
    @pulse.kernel
    def plain(q0, q1):
        lines = [get_drive_line(q0), get_drive_line(q1)]
        r = get_readout_line(q0)
        bits = [readout(r, square(40, 0.05))]
        for _ in range(3):
            sync(lines[0], lines[1], r)
            drive(lines[0], square(16, 0.1))
            shift_phase(lines[1].tone, 0.2)
        wait(lines[1], 8 * len(bits))

    (lines,) = _traced_lines(monkeypatch, plain, 2)
    assert type(lines).__name__ == "Recorded"
    assert not hasattr(lines, "journal")
    assert not any(hasattr(held, "seq") for held in lines.held.values())

    # With an if, each channel has its place, and no journal is open after it.
    (lines,) = _traced_lines(monkeypatch, _swap_in_loop_in_if, 2)
    assert type(lines).__name__ == "JournalLines"
    assert lines.journal is None
    assert [held.seq for held in lines.held.values()] == list(range(len(lines.held)))


# A container of each kind, and a change of it that keeps its length or
# adds to it in place.
_CHANGED = {
    "deque": ("collections.deque([1, 2])", "values.rotate(1)"),
    "bytearray": ("bytearray(b'ab')", "values.append(3)"),
    "ordered dict": ("collections.OrderedDict(a=1, b=2)", "values.move_to_end('a')"),
    "dict, |= on a key it has": ("{'a': 1}", "values |= {'a': 2}"),
    "set, |=": ("{1}", "values |= {2}"),
    "list in a dict, +=": ("{'a': []}", "values['a'] += [2]"),
    "list subclass": ("_Outcomes([1])", "values.append(2)"),
}


@pytest.mark.parametrize("case", list(_CHANGED))
def test_an_arm_that_changes_a_container_of_any_kind_is_rejected(case):
    made, change = _CHANGED[case]
    source = (
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        f"    values = {made}\n"
        "    if readout(r, square(40, 0.05)):\n"
        f"        {change}\n"
        "        drive(d, square(16, 0.1))\n"
    )
    namespace: dict = {}
    scope = vars(pulse) | {"collections": collections, "_Outcomes": _Outcomes}
    exec(compile(source, f"<{case}>", "exec"), scope, namespace)
    with pytest.raises(CompilationError, match="may only change list entries that hold lines"):
        pulse.compile(pulse.kernel(namespace["k"])).lower(Q[0])


@pytest.mark.parametrize("made", ["collections.deque()", "bytearray()"])
def test_a_loop_that_adds_to_a_deque_or_bytearray_traces_each_iteration(made):
    source = (
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        f"    values = {made}\n"
        "    for _ in range(4):\n"
        "        readout(r, square(40, 0.05))\n"
        "        values.append(1)\n"
        "    drive(d, square(8 * len(values), 0.1))\n"
    )
    namespace: dict = {}
    exec(compile(source, "<adds>", "exec"), vars(pulse) | {"collections": collections}, namespace)
    events = pulse.compile(pulse.kernel(namespace["k"])).lower(Q[0]).schedule
    assert [e.duration_cycles for e in events if e.kind == "drive"] == [32]


def _beside_a_list(condition, listed, change=""):
    source = (
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    r = get_readout_line(q)\n"
        "    n = get_n.submit().result()\n"
        "    outcomes = [0.5]\n"
        "    for _ in range(3):\n"
        "        m = readout(r, square(40, 0.05))\n"
        "        m2 = readout(r, square(40, 0.05))\n"
        "        sync(d, r)\n"
        f"        if {condition}:\n"
        f"            drive(d, square(16, 0.1)){change}\n"
        "        drive(d, square(8, 0.2 if n > 3 else 0.1))\n"
        + ("        outcomes.append(m)\n" if listed else "")
    )
    scope = vars(pulse) | {"get_n": pulse.classical(_get_n, max_latency_ns=0)}
    namespace: dict = {}
    exec(compile(source, "<beside>", "exec"), scope, namespace)
    return pulse.kernel(unroll_threshold=10**9)(namespace["k"])


_CONDITIONS = [
    "m and not m2",
    "(n > 3) & (n < 9)",
    "(n > 3) | (n < 1)",
    "n in (1, 5)",
    "(n > 3) is True",
    "any([n > 3, n < 1])",
    "all([n > 1, n < 9])",
]


@pytest.mark.parametrize("condition", _CONDITIONS)
def test_runtime_conditions_beside_a_growing_list_trace_as_without_it(condition):
    # The list is copied only if an arm changes it: these arms do not.
    listed = pulse.compile(_beside_a_list(condition, True)).lower(Q[0]).schedule
    assert listed == pulse.compile(_beside_a_list(condition, False)).lower(Q[0]).schedule


@pytest.mark.parametrize("condition", ["m and m2", "(n > 3) & (n < 9)", "any([n > 3, n < 1])"])
def test_an_arm_on_a_combined_condition_that_changes_a_list_entry_is_rejected(condition):
    kernel = _beside_a_list(condition, True, change="; outcomes[0] = 0.25")
    with pytest.raises(CompilationError, match="may only change list entries that hold lines"):
        pulse.compile(kernel).lower(Q[0])


def test_an_arm_that_changes_a_list_entry_is_rejected():
    @pulse.kernel
    def changes(q):
        d = get_drive_line(q)  # noqa: F821
        r = get_readout_line(q)  # noqa: F821
        values = [1.0, 2.0]
        if readout(r, square(40, 0.05)):  # noqa: F821
            values[0] = 3.0
            drive(d, square(16, 0.1))  # noqa: F821

    with pytest.raises(CompilationError, match="may only change list entries that hold lines"):
        pulse.compile(changes).lower(Q[0])


def test_an_arm_whose_plain_python_call_reorders_a_list_is_rejected():
    rng = random.Random(0)

    @pulse.kernel
    def shuffles(q):
        d = get_drive_line(q)  # noqa: F821
        r = get_readout_line(q)  # noqa: F821
        values = [1.0, 2.0, 3.0, 4.0]
        if readout(r, square(40, 0.05)):  # noqa: F821
            rng.shuffle(values)
            drive(d, square(16, 0.1))  # noqa: F821

    with pytest.raises(CompilationError, match="may only change list entries that hold lines"):
        pulse.compile(shuffles).lower(Q[0])


def _events(kernel, *qudits):
    return [
        (e.kind, e.line, e.start_cycles, e.duration_cycles)
        for e in pulse.compile(kernel).lower(*qudits).schedule
    ]


@pulse.classical(max_latency_ns=400)
def _corrections(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0], bits[0]]


def test_corrections_from_an_answer_drive_lines_held_in_a_list():
    @pulse.kernel
    def listed(q0, q1, q2):
        data = [get_drive_line(q0), get_drive_line(q1)]  # noqa: F821
        probe = get_readout_line(q2)  # noqa: F821
        for _ in range(2):
            fix = _corrections.submit([readout(probe, square(80, 0.05))]).result()  # noqa: F821
            for k in range(2):
                if fix[k]:
                    drive(data[k], square(16, 0.3))  # noqa: F821

    @pulse.kernel
    def named(q0, q1, q2):
        d0 = get_drive_line(q0)  # noqa: F821
        d1 = get_drive_line(q1)  # noqa: F821
        probe = get_readout_line(q2)  # noqa: F821
        for _ in range(2):
            fix = _corrections.submit([readout(probe, square(80, 0.05))]).result()  # noqa: F821
            if fix[0]:
                drive(d0, square(16, 0.3))  # noqa: F821
            if fix[1]:
                drive(d1, square(16, 0.3))  # noqa: F821

    assert _events(listed, Q[0], Q[1], Q[2]) == _events(named, Q[0], Q[1], Q[2])


class _ReadCounted(list):
    """A list of lines that counts how often its items are read."""

    reads = 0

    def __getitem__(self, i):
        _ReadCounted.reads += 1
        return list.__getitem__(self, i)


def _read_counted(body: str, tag: str):
    scope = vars(pulse) | {"ReadCounted": _ReadCounted}
    exec(compile("def k(qudits):\n" + body, f"<{tag}>", "exec"), scope)
    return pulse.kernel(scope["k"])


# Kernels holding their drive lines in a list subclass, the qudits each takes, and how often one
# compile reads the list's items: main's counts, which tracing an if must not change.
_SUBCLASS_READS = {
    "across an if": (
        2,
        7,
        """\
    lines = ReadCounted([get_drive_line(q) for q in qudits])
    probes = [get_readout_line(q) for q in qudits]
    bits = []
    bits.append(readout(probes[0], square(40, 0.05)))
    if bits[0]:
        drive(lines[0], square(8, 0.1))
    else:
        drive(lines[1], square(16, 0.1))
    drive(lines[0], square(8, 0.2))
""",
    ),
    "entries swapped in the then arm": (
        2,
        5,
        """\
    lines = ReadCounted([get_drive_line(q) for q in qudits])
    probes = [get_readout_line(q) for q in qudits]
    bits = []
    bits.append(readout(probes[0], square(40, 0.05)))
    if bits[0]:
        lines[0], lines[1] = lines[1], lines[0]
    wait(probes[1], 8)
""",
    ),
    "nine lines, an empty if": (
        3,
        10,
        """\
    lines = ReadCounted([get_drive_line(q) for q in qudits] * 3)
    probes = [get_readout_line(q) for q in qudits]
    bits = []
    bits.append(readout(probes[0], square(40, 0.05)))
    if bits[0]:
        pass
    wait(probes[1], 8)
""",
    ),
    "swapped in a rolled loop": (
        2,
        6,
        """\
    lines = ReadCounted([get_drive_line(q) for q in qudits])
    probes = [get_readout_line(q) for q in qudits]
    for _ in range(3):
        lines[0], lines[1] = lines[1], lines[0]
        wait(probes[0], 8)
        wait(probes[1], 8)
    wait(probes[1], 8)
""",
    ),
}


@pytest.mark.parametrize("case", list(_SUBCLASS_READS))
def test_a_list_subclass_of_lines_is_read_as_often_as_before(case):
    qudits, reads, body = _SUBCLASS_READS[case]
    kernel = _read_counted(body, case)
    _ReadCounted.reads = 0
    pulse.compile(kernel).lower(Q[0:qudits])
    assert _ReadCounted.reads == reads


# Kernels whose lists of lines alias, are copied or held elsewhere, or change
# between and inside measurement-conditioned ifs, each if the last statement
# of its loop: its else arm only jumps back to the loop, and the if takes it
# as changing nothing instead of tracing it. That must leave what tracing it
# does.
_EMPTY_ELSE = {
    "copied, aliased and held in a dict": """\
    lines = [get_drive_line(q) for q in qudits]
    probes = [get_readout_line(q) for q in qudits]
    copy = list(lines)
    alias = lines
    keyed = {"a": lines}
    bits = [readout(probes[k], square(40, 0.05)) for k in range(4)]
    for k in range(4):
        drive(alias[k], square(8, 0.2))
        if bits[k]:
            drive(copy[k], square(8, 0.1))
            drive(keyed["a"][-1], square(16, 0.1))
            lines[0], lines[3] = lines[3], lines[0]
    drive(lines[0], square(8, 0.3))
""",
    "a copy read between ifs": """\
    lines = [get_drive_line(q) for q in qudits]
    probes = [get_readout_line(q) for q in qudits]
    copy = list(lines)
    bits = [readout(probes[k], square(40, 0.05)) for k in range(4)]
    x = lines[1]
    for k in range(4):
        wait(copy[1], 8)
        if bits[k]:
            drive(lines[k], square(8, 0.1))
    drive(x, square(8, 0.2))
    drive(lines[1], square(8, 0.3))
""",
    "every line synced through a list of all": """\
    lines = [get_drive_line(q) for q in qudits] * 3
    probes = [get_readout_line(q) for q in qudits]
    sync(*lines, *probes)
    bits = [readout(probes[k], square(40, 0.05)) for k in range(4)]
    for k in range(4):
        if bits[k]:
            drive(lines[k + 12], square(8, 0.1))
""",
    "changed between ifs": """\
    lines = [get_drive_line(q) for q in qudits]
    probes = [get_readout_line(q) for q in qudits]
    bits = [readout(probes[k], square(40, 0.05)) for k in range(4)]
    for k in range(4):
        lines[0], lines[2] = lines[2], lines[0]
        lines[1:2] = [lines[3]]
        lines += [lines[0]]
        del lines[-1]
        if bits[k]:
            drive(lines[k], square(8, 0.1))
    drive(lines[1], square(8, 0.2))
""",
    "changed in an arm": """\
    lines = [get_drive_line(q) for q in qudits]
    probes = [get_readout_line(q) for q in qudits]
    bits = [readout(probes[k], square(40, 0.05)) for k in range(4)]
    for k in range(4):
        if bits[k]:
            lines.append(lines[k])
    drive(lines[0], square(8, 0.2))
""",
}


def _traced(body: str, tag: str) -> str:
    scope = dict(vars(pulse))
    exec(compile("def k(qudits):\n" + body, f"<{tag}>", "exec"), scope)
    try:
        return pulse.compile(pulse.kernel(scope["k"])).lower(Q[0:12]).mlir()
    except (CompilationError, RuntimeError) as e:
        return f"error: {type(e).__name__}: {e}"


@pytest.mark.parametrize("case", list(_EMPTY_ELSE))
def test_an_else_arm_that_only_jumps_back_is_taken_as_traced(case, monkeypatch):
    from cudaq_pulse.kernel import _emitter

    body = _EMPTY_ELSE[case]
    arms = []
    trace_arm = _emitter.PulseIREmitter._trace_arm

    def counted(self, view, branch, join, start, snapshot):
        arms.append(start)
        return trace_arm(self, view, branch, join, start, snapshot)

    monkeypatch.setattr(_emitter.PulseIREmitter, "_trace_arm", counted)
    taken = _traced(body, case)
    skipped = len(arms)
    arms.clear()
    # No arm is a jump: every else arm is traced.
    monkeypatch.setattr(_emitter, "JUMP", -1)
    assert _traced(body, case) == taken
    assert taken.startswith("error") or len(arms) > skipped


# An old value of a line: `t` names line 0 as it was before the drive, which
# was through `a`. Each kernel binds a list place or a name to it in an arm
# of a measurement-conditioned if, where it is the line's current value, as
# get_drive_line(q0) is: every name and place holds line 0 on both paths, and
# the kernel is the one written with get_drive_line(q0) in its place.
_BEFORE_OLD_LINE = """\
    lines = [get_drive_line(q0), get_drive_line(q1)]
    a = lines[0]
    t = a
    drive({first}, square(16, 0.1))
    m = readout(get_readout_line(q0), square(40, 0.05))
"""
_OLD_LINE = {
    "a place, driven after": (
        "a",
        "    if m:\n        lines[0] = {old}\n    drive(lines[0], square(16, 0.1))\n",
    ),
    "a place, waited on after": (
        "a",
        "    if m:\n        lines[0] = {old}\n    wait(lines[0], 16)\n",
    ),
    "a place, the other name driven after": (
        "a",
        "    if m:\n        lines[0] = {old}\n    drive(a, square(16, 0.1))\n",
    ),
    "a place, the old name driven first": (
        "t",
        "    if m:\n        lines[0] = {old}\n    drive(lines[0], square(16, 0.1))\n",
    ),
    "a place, the old name driven first, waited on after": (
        "t",
        "    if m:\n        lines[0] = {old}\n    wait(lines[0], 16)\n",
    ),
    "a place, the old name driven first and the other after": (
        "t",
        "    if m:\n        lines[0] = {old}\n    drive(a, square(16, 0.1))\n",
    ),
    "a place and the old name swapped": (
        "t",
        "    if m:\n        t, lines[0] = lines[0], {old}\n"
        "    drive(lines[0], square(16, 0.1))\n    drive(t, square(16, 0.1))\n",
    ),
    "a place, driven in the arm": (
        "a",
        "    if m:\n        lines[0] = {old}\n        drive(lines[0], square(16, 0.1))\n"
        "    drive(lines[0], square(16, 0.1))\n    drive(a, square(16, 0.1))\n",
    ),
    "a place, the line driven in the arm through another name": (
        "a",
        "    if m:\n        lines[0] = {old}\n        drive(a, square(16, 0.1))\n"
        "    drive(lines[0], square(16, 0.1))\n",
    ),
    "a place, in the else arm": (
        "a",
        "    if m:\n        drive(lines[1], square(16, 0.1))\n    else:\n        lines[0] = {old}\n"
        "    drive(lines[0], square(16, 0.1))\n",
    ),
    "a name, in each arm": (
        "a",
        "    if m:\n        u = {old}\n    else:\n        u = a\n    drive(u, square(16, 0.1))\n",
    ),
}


def _paths(source: str, **options):
    """The schedule of kernel ``k(q0, q1, q2)`` of *source*, and the drives
    it plays when its measurements are set and when they are not."""
    scope = dict(vars(pulse))
    exec(compile(source, "<k>", "exec"), scope)
    lowered = pulse.compile(pulse.kernel(scope["k"]), **options).lower(Q[0], Q[1], Q[2])
    events = [(e.kind, e.qudit, e.start_cycles, e.duration_cycles) for e in lowered.schedule]
    return events, [lowered._concrete().drive_frames([], outcome)[0] for outcome in (1, 0)]


@pytest.mark.parametrize("case", list(_OLD_LINE))
def test_an_old_value_of_a_line_bound_in_an_arm_is_its_current_value(case):
    first, body = _OLD_LINE[case]
    head = "def k(q0, q1, q2):\n" + _BEFORE_OLD_LINE.format(first=first)
    for options in ({}, {"passes": ()}):
        old = _paths(head + body.format(old="t"), **options)
        assert old == _paths(head + body.format(old="get_drive_line(q0)"), **options)


# Two names bound in an arm to one line, which they then hold on both paths:
# the if makes one result for them, as for one name.
_ONE_LINE_TWO_NAMES = {
    "bound in the then arm": (
        "    if m:\n        drive(a, square(16, 0.1))\n        t = a\n"
        "    drive(t, square(16, 0.1))\n",
        "    if m:\n        drive(a, square(16, 0.1))\n    drive(a, square(16, 0.1))\n",
    ),
    "bound in both arms": (
        "    if m:\n        drive(a, square(16, 0.1))\n        t = a\n    else:\n        t = a\n"
        "    drive(t, square(16, 0.1))\n    drive(a, square(16, 0.1))\n",
        "    if m:\n        drive(a, square(16, 0.1))\n"
        "    drive(a, square(16, 0.1))\n    drive(a, square(16, 0.1))\n",
    ),
}


@pytest.mark.parametrize("case", list(_ONE_LINE_TWO_NAMES))
def test_two_names_an_arm_binds_to_one_line_take_one_result(case):
    two, one = _ONE_LINE_TWO_NAMES[case]
    head = (
        "def k(q0, q1, q2):\n    a = get_drive_line(q0)\n    t = a\n"
        "    m = readout(get_readout_line(q0), square(40, 0.05))\n"
    )
    for options in ({}, {"passes": ()}):
        assert _paths(head + two, **options) == _paths(head + one, **options)


# A helper that binds its parameter to another value leaves the names its
# caller passed as they were, as in Python: a line or tone follows only what
# the helper does to it.
_HELPERS = """\
def rebind(x, y):
    x = y
    drive(x, square(16, 0.1))

def rebind_only(x, y):
    x = y

def another_waveform(w):
    w = square(8, 0.2)

def rebind_in_an_arm(x, y, m):
    if m:
        x = y
        drive(x, square(16, 0.1))

"""
_REBOUND_PARAMETER = {
    "rebound and driven": (
        "    rebind(a, b)\n    wait(a, 8)\n",
        "    drive(b, square(16, 0.1))\n    wait(a, 8)\n",
    ),
    "rebound only": (
        "    rebind_only(a, b)\n    wait(a, 8)\n    wait(b, 8)\n",
        "    wait(a, 8)\n    wait(b, 8)\n",
    ),
    "a waveform": (
        "    w = square(16, 0.1)\n    another_waveform(w)\n    drive(a, w)\n",
        "    w = square(16, 0.1)\n    drive(a, w)\n",
    ),
}


@pytest.mark.parametrize("case", list(_REBOUND_PARAMETER))
def test_a_helper_binding_its_parameter_leaves_the_callers_names(case):
    called, written = _REBOUND_PARAMETER[case]
    head = _HELPERS + "def k(q0, q1, q2):\n    a = get_drive_line(q0)\n    b = get_drive_line(q1)\n"
    assert _paths(head + called) == _paths(head + written)


def test_a_helper_binding_its_parameter_in_an_arm_leaves_the_callers_line():
    # The helper's name holds another line on each path after its if, so the
    # if may not be verified; the caller's wait is on its own line either way.
    head = _HELPERS + (
        "def k(q0, q1, q2):\n    a = get_drive_line(q0)\n    b = get_drive_line(q1)\n"
        "    m = readout(get_readout_line(q0), square(40, 0.05))\n"
    )
    called = _paths(head + "    rebind_in_an_arm(a, b, m)\n    wait(a, 8)\n", passes=())
    written = "    if m:\n        drive(b, square(16, 0.1))\n    wait(a, 8)\n"
    assert called == _paths(head + written, passes=())
