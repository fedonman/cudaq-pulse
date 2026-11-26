# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The classical check's two timing bounds over the loops around a call.

An answer is taken no earlier than it is due, and a call with a reaction
deadline has its answer due within it, in every iteration of the loops
around them. Both times are affine in the iteration, so the check takes
their extreme over the corners of the iteration box without visiting the
corners. These tests hold it to what visiting every corner gives, the
corner it names included, and to loops nested deeper than corners can be
counted."""

from __future__ import annotations

import itertools
import random
import re

import pytest
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline

# A call of @f: one cycle to submit, and 100 of budget at this clock.
_DUE_AFTER = 101
_READOUT = 100

_HEADER = """module attributes {{
    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>}} {{
  classical.callable @f {{dispatch_key = 3809224601 : i64, device = 0 : i64,
      args = [{args}], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"{deadline}}}
  func.func @test(%r0: !pulse.readout_line, %rt0: !pulse.tone, %ro: !pulse.waveform) {{
    %l0 = classical.get_link {{device = 0 : i64}} : !classical.link
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index
"""


def _timing(start, duration, periods):
    period = f"period_cycles = array<i64: {', '.join(map(str, periods))}>, " if periods else ""
    return f"<{{duration_cycles = {duration} : i64, {period}start_cycles = {start} : i64}}>"


def _loops(trips, body_at):
    """The loops, outermost first, with body_at(level) giving the lines
    written at the top of the loop at *level* (len(trips) is inside all)."""
    lines = list(body_at(0))
    for level, count in enumerate(trips):
        prior = "%l0" if level == 0 else f"%arg{level - 1}"
        lines.append(f"    %ub{level} = arith.constant {count} : index")
        lines.append(
            f"    %end{level} = scf.for %i{level} = %lb to %ub{level} step %step "
            f"iter_args(%arg{level} = {prior}) -> (!classical.link) {{"
        )
        lines += body_at(level + 1)
    return lines


def _closing(depth):
    lines = []
    for level in reversed(range(depth)):
        forwarded = "%submitted" if level == depth - 1 else f"%end{level + 1}"
        lines += [f"    scf.yield {forwarded} : !classical.link", "    }"]
    return [*lines, "    return", "  }", "}", ""]


def _answer(trips, submit, sent, take, took):
    """A call submitted at *submit* and its answer taken at *take*, inside
    loops of *trips*; *sent* and *took* are their periods per loop."""
    depth = len(trips)
    last = "%l0" if not depth else f"%arg{depth - 1}"

    def body_at(level):
        if level < depth:
            return []
        return [
            f"    %submitted, %h = classical.submit {last}, @f() {_timing(submit, 1, sent)} "
            ": () -> (!classical.link, !classical.future)",
            f"    %v = classical.wait %h, @f {_timing(take, 0, took)} : f64",
        ]

    lines = [_HEADER.format(args="", deadline="").rstrip()]
    return "\n".join(lines + _loops(trips, body_at) + _closing(depth))


def _reaction(trips, own, read_at, read, submit, sent, deadline):
    """A readout inside the first *own* loops, carried by a call submitted
    inside all of them, which must react within *deadline* cycles."""
    depth = len(trips)
    last = "%l0" if not depth else f"%arg{depth - 1}"

    def body_at(level):
        lines = []
        if level == own:
            lines.append(
                f"    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, iq "
                f"{_timing(read_at, _READOUT, read)} : !pulse.readout_line, !pulse.waveform, "
                "!pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement"
            )
        if level == depth:
            lines.append(
                f"    %submitted, %h = classical.submit {last}, @f(%m) "
                f"{_timing(submit, 1, sent)} : (!pulse.measurement) -> "
                "(!classical.link, !classical.future)"
            )
        return lines

    header = _HEADER.format(args='"bits:1"', deadline=f", reaction_deadline_ns = {float(deadline)}")
    return "\n".join([header.rstrip()] + _loops(trips, body_at) + _closing(depth))


def _errors(text):
    module = MLIRPipeline().parse_to_module(text)
    try:
        module.run_passes(["pulse-classical-check"])
    except RuntimeError as error:
        return str(error)
    return ""


def _corners(trips):
    """Every corner of the iteration box, the outermost loop changing
    fastest: the order the check names the first failing one in."""
    for corner in range(2 ** len(trips)):
        yield [count - 1 if corner >> level & 1 else 0 for level, count in enumerate(trips)]


def _at(start, periods, iteration):
    return start + sum(k * period for k, period in zip(iteration, periods, strict=False))


def _taken_before_due(trips, submit, sent, take, took):
    """What visiting every corner reports: None, or (taken, due) at the
    first corner where the answer is taken before it is due."""
    if 0 in trips:
        return None
    for iteration in _corners(trips):
        due = _at(submit, sent, iteration) + _DUE_AFTER
        taken = _at(take, took, iteration)
        if taken < due:
            return taken, due
    return None


def _worst_reaction(trips, own, read_at, read, submit, sent):
    """(worst reaction, its iteration): the first corner with the most
    between the readout's end and the answer being due."""
    worst = None
    for iteration in _corners(trips):
        due = _at(submit, sent, iteration) + _DUE_AFTER
        reaction = due - (_at(read_at, read, iteration[:own]) + _READOUT)
        if worst is None or reaction > worst[0]:
            worst = reaction, iteration
    return worst


def _taken(message):
    found = re.search(r"is taken at (-?\d+), before it is due at (-?\d+)", message)
    return (int(found.group(1)), int(found.group(2))) if found else None


def test_an_answer_taken_too_early_is_found_as_by_visiting_every_corner():
    random.seed(7)
    failures = 0
    for case in range(400):
        depth = random.randrange(0, 6)
        trips = [
            random.choice([0, 1, 1, 2, 3, 5]) if case % 9 == 0 else random.randrange(1, 6)
            for _ in range(depth)
        ]
        sent = [random.randrange(-20, 21) for _ in range(depth)]
        took = [random.randrange(-20, 21) for _ in range(depth)]
        submit = random.randrange(0, 50)
        take = submit + _DUE_AFTER + random.randrange(-60, 120)
        expected = _taken_before_due(trips, submit, sent, take, took)
        found = _taken(_errors(_answer(trips, submit, sent, take, took)))
        assert found == expected, (trips, submit, sent, take, took)
        failures += expected is not None
    # Both outcomes are well represented.
    assert 80 < failures < 320


def test_ties_between_corners_name_the_first():
    # Every loop moves the answer two cycles earlier each iteration, and the
    # answer is one cycle late already: every corner fails, the first named.
    trips = [3, 3, 3]
    text = _answer(trips, 0, [0, 0, 0], _DUE_AFTER - 1, [-2, -2, -2])
    assert _taken(_errors(text)) == (_DUE_AFTER - 1, _DUE_AFTER)
    # A corner fails with at least two of the three loops at their last
    # iteration: the first has the two outermost there.
    text = _answer(trips, 0, [0, 0, 0], _DUE_AFTER + 5, [-2, -2, -2])
    assert _taken(_errors(text)) == (_DUE_AFTER + 5 - 8, _DUE_AFTER)
    assert _taken_before_due(trips, 0, [0, 0, 0], _DUE_AFTER + 5, [-2, -2, -2]) == (
        _DUE_AFTER - 3,
        _DUE_AFTER,
    )


def test_a_reaction_deadline_is_bounded_as_by_visiting_every_corner():
    random.seed(11)
    failures = 0
    for _ in range(300):
        depth = random.randrange(0, 5)
        own = random.randrange(0, depth + 1)
        trips = [random.randrange(1, 6) for _ in range(depth)]
        read = [random.randrange(-20, 21) for _ in range(own)]
        sent = [random.randrange(0, 41) for _ in range(depth)]
        read_at = random.randrange(0, 50)
        submit = read_at + _READOUT + random.randrange(0, 40)
        deadline = _DUE_AFTER + random.randrange(0, 200)
        worst, iteration = _worst_reaction(trips, own, read_at, read, submit, sent)
        message = _errors(_reaction(trips, own, read_at, read, submit, sent, deadline))
        case = (trips, own, read_at, read, submit, sent, deadline)
        if worst <= deadline:
            assert "cannot react in time" not in message, case
            continue
        failures += 1
        where = f"in iteration ({', '.join(map(str, iteration))}) " if depth else ""
        expected = (
            f"'f' cannot react in time: {where}its answer is due {worst} cycles after the "
            f"readouts it carries end, but reaction_deadline_ns allows {deadline} "
            f"(slack {deadline - worst})"
        )
        assert expected in message, (case, message)
    assert 60 < failures < 280


@pytest.mark.parametrize("depth", [40, 64, 65, 120, 200])
def test_loops_nested_deeper_than_corners_can_be_counted(depth):
    # Each loop runs once: one execution, whatever the depth.
    once = [1] * depth
    zero = [0] * depth
    assert _errors(_answer(once, 0, zero, _DUE_AFTER, zero)) == ""
    assert _taken(_errors(_answer(once, 0, zero, _DUE_AFTER - 1, zero))) == (
        _DUE_AFTER - 1,
        _DUE_AFTER,
    )
    # The check also lists every execution of a call, so only the two
    # innermost loops repeat here. Only the innermost moves the answer
    # earlier: the one failing corner has every other loop at its first
    # iteration.
    thrice = [1] * (depth - 2) + [3, 3]
    inner = [0] * (depth - 1) + [-1]
    assert _taken(_errors(_answer(thrice, 0, zero, _DUE_AFTER + 2, inner))) is None
    assert _taken(_errors(_answer(thrice, 0, zero, _DUE_AFTER + 1, inner))) == (
        _DUE_AFTER - 1,
        _DUE_AFTER,
    )
    # A deadline the last iteration of both misses by one cycle.
    sent = [1] * depth
    message = _errors(_reaction(thrice, 0, 0, [], _READOUT, sent, _DUE_AFTER + 3))
    where = ", ".join(["0"] * (depth - 2) + ["2", "2"])
    assert f"in iteration ({where}) its answer is due {_DUE_AFTER + 4} cycles" in message
    assert "cannot react" not in _errors(
        _reaction(thrice, 0, 0, [], _READOUT, sent, _DUE_AFTER + 4)
    )


def test_the_corners_are_as_the_check_counts_them():
    # The helper's order: the outermost loop changes fastest.
    assert list(itertools.islice(_corners([2, 3]), 4)) == [[0, 0], [1, 0], [0, 2], [1, 2]]
