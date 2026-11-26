# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The values a controller program computes at run time are exported as
expressions in one vocabulary of terms, ``ExpressionTerm``, whatever
operations the compiler computes them with."""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
import warnings

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, _validation
from cudaq_pulse._errors import ExecutionError, InvalidValueError, UnsupportedError
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.controller import (
    _AMPLITUDES,
    _PARAMETERS,
    FORMAT,
    ExpressionTerm,
    evaluate_expression,
    unroll,
)

from tests._handles import Q
from tests.kernel.test_runtime_conditions import _kernel as _traced


@pulse.classical(max_latency_ns=200, name="expression_inputs")
def _inputs(bits: pulse.Bits) -> tuple[float, int]:
    return 0.25 * bits[0], 3 + bits[0]


_HEAD = (
    "    d = get_drive_line(q)\n"
    "    ro = get_readout_line(q)\n"
    "    m = readout(ro, square(100, 0.05))\n"
)


def _kernel(body, *, bound=False):
    """A kernel running *body* on a float ``x`` and an int ``n``: answers of
    a call, or (*bound*) parameters, which the compiler folds."""
    source = "def k(q, x: float, n: int):\n" if bound else "def k(q):\n"
    source += _HEAD
    if not bound:
        source += "    x, n = _inputs.submit([m]).result()\n"
    source += "".join(f"    {line}\n" for line in body)
    namespace = {"_inputs": _inputs}
    exec(source, namespace)  # noqa: S102
    return pulse.kernel(namespace["k"])


def _value(program):
    """The phase the program shifts by, or the length it waits."""
    for event in program["events"] if program["format"] == FORMAT else unroll(program)["events"]:
        if event["kind"] == "wait":
            return event.get("length") or event["duration_cycles"]
        if event.get("op") == "shift_phase":
            return event["value"]
        for update in event.get("frame_updates", ()):
            if update["op"] == "shift_phase":
                return update["value"]
    raise AssertionError("no phase shift or wait")


def _answer(program):
    return next(e["index"] for e in program["events"] if e["kind"] == "answer")


def _ops(expression):
    if not isinstance(expression, dict):
        return set()
    found = {expression["op"]} if "op" in expression else set()
    for child in expression.get("of", ()):
        found |= _ops(child)
    return found


@pytest.fixture(autouse=True)
def _selects(monkeypatch):
    monkeypatch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)


def _if(condition, then, other, use="shift_phase(d.tone, y)"):
    return [f"if {condition}:", f"    y = {then}", "else:", f"    y = {other}", use]


_PHASES = {
    "add": ["shift_phase(d.tone, x + 0.5)"],
    "sub": ["shift_phase(d.tone, 0.5 - x)"],
    "mul": ["shift_phase(d.tone, x * 3.0)"],
    "div": ["shift_phase(d.tone, x / 3.0)"],
    "neg": ["shift_phase(d.tone, -x)"],
    "to_float": ["shift_phase(d.tone, n * 0.5)"],
    "max": _if("x > 0.1", "x", "0.1"),
    "min": _if("x < 0.1", "x", "0.1"),
    "ge": _if("x >= 0.1", "x", "0.1"),
    "le": _if("x <= 0.1", "0.3", "0.1"),
    "lt": _if("n < 4", "0.3", "0.1"),
    "gt": _if("n > 3", "0.3", "0.1"),
    "eq": _if("n == 3", "0.3", "0.1"),
    "ne": _if("x != 0.25", "0.3", "0.1"),
    "select": _if("n == 3", "x", "0.1"),
}
_WAITS = {
    "add": ["wait(d, n + 4)"],
    "sub": ["wait(d, 10 - n)"],
    "mul": ["wait(d, n * 4)"],
    "floor_div": ["wait(d, n // 2 + 1)"],
    "rem": ["wait(d, n % 2 + 4)"],
    "to_int": ["wait(d, int(x * 80.0))"],
}
_INPUTS = [(0.05, 2), (0.1, 3), (0.25, 5), (0.3, 4)]


def _folded(body, x, n, machine=None):
    """What the compiler computes for *body* at x and n."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled = pulse.compile(_kernel(body, bound=True), control_system=machine)
        return _value(compiled.lower(Q[0], x, n).export_to_controller())


@pytest.mark.parametrize(
    ("term", "body"),
    [*_PHASES.items(), *((f"wait {term}", body) for term, body in _WAITS.items())],
)
def test_each_term_computes_what_the_compiler_computes(term, body):
    lowered = pulse.compile(_kernel(body)).lower(Q[0])
    program = lowered.export_to_controller()
    expression = _value(program)
    assert term.removeprefix("wait ") in _ops(expression), expression
    answer = _answer(program)
    for x, n in _INPUTS:
        got = evaluate_expression(expression, answers={answer: [x, n]})
        assert got == _folded(body, x, n), (x, n)
    # Format /2 exports the same expression.
    assert _value(lowered.export_to_controller(version=2)) == expression


@pytest.mark.parametrize("term", sorted(_WAITS))
def test_each_term_waits_as_long_as_the_emulated_program(term):
    lowered = pulse.compile(_kernel(_WAITS[term])).lower(Q[0])
    expression = _value(program := lowered.export_to_controller())
    args = lowered.options.schedule_args()
    module = lowered._concrete().simulation_module(*args)
    readout, iteration = module.next_measurement([])[:2]
    call = module.next_call([[readout, *iteration, 1]])[0]
    for x, n in _INPUTS:
        bound = module.bind_call(call, [x, float(n)], *args)
        [wait] = (e for e in bound.events() if e[0] == "wait")
        assert evaluate_expression(expression, answers={_answer(program): [x, n]}) == wait[4]


def test_a_readout_is_set_when_it_observed_any_level_but_0():
    program = pulse.compile(_kernel(_if("m", "0.3", "0.1"))).lower(Q[0]).export_to_controller()
    readout = next(e["index"] for e in program["events"] if e["kind"] == "readout")
    expression = _value(program)
    assert expression["of"][0] == {
        "op": "ne",
        "type": "int",
        "of": [{"readout": readout}, {"const": 0}],
    }
    levels = {0: 0.1, 1: 0.3, 2: 0.3}
    for level, phase in levels.items():
        assert evaluate_expression(expression, readouts={readout: level}) == phase


def test_a_controller_clock_clamps_and_converts_a_run_time_length():
    machine = ControlSystem(controllers=[Controller(clock_ghz=2.0, cycles_per_timestep=4)])
    body = ["wait(d, n * 4, bounds=(8, 400))"]
    program = (
        pulse.compile(_kernel(body), control_system=machine).lower(Q[0]).export_to_controller()
    )
    expression = _value(program)
    assert {"controller_cycles", "max", "min"} <= _ops(expression)
    answer = _answer(program)
    for n in (2, 7, 100):
        got = evaluate_expression(expression, answers={answer: [0.0, n]})
        assert got == _folded(body, 0.0, n, machine) == 4 * n
    # Outside its bounds, which a kernel cannot wait for, the run clamps it.
    for n, cycles in ((1, 8), (150, 400)):
        assert evaluate_expression(expression, answers={answer: [0.0, n]}) == cycles


def test_every_term_is_exercised():
    exported = set()
    for body in (*_PHASES.values(), *_WAITS.values()):
        exported |= _ops(_value(pulse.compile(_kernel(body)).lower(Q[0]).export_to_controller()))
    # The controller's cycles come from a controller clock, tested above.
    assert exported | {"controller_cycles"} == set(ExpressionTerm)


# ── Terms several terms use, and terms many deep ─────────────────────────────

# How many terms deep an expression's text nests (kNested of the native walk).
_NESTED = 65


def _nesting(expression):
    """How many terms deep the text of *expression* and of its terms nests."""
    deepest = 0
    for top in (expression, *expression.get("terms", ())):
        todo = [(top, 1)]
        while todo:
            node, depth = todo.pop()
            if isinstance(node, dict) and "of" in node:
                deepest = max(deepest, depth)
                todo.extend((child, depth + 1) for child in node["of"])
    return deepest


def _emulated_wait(lowered, x, n):
    """The length the emulated program waits when the call answers x and n."""
    args = lowered.options.schedule_args()
    module = lowered._concrete().simulation_module(*args)
    readout, iteration = module.next_measurement([])[:2]
    call = module.next_call([[readout, *iteration, 1]])[0]
    bound = module.bind_call(call, [x, float(n)], *args)
    [wait] = (e for e in bound.events() if e[0] == "wait")
    return wait[4]


def _add(kind, *of):
    return {"op": "add", "type": kind, "of": list(of)}


def test_a_term_two_terms_use_is_in_the_terms_of_its_expression_once():
    body = ["t = x * 2.0", *["t = t + t"] * 3, "shift_phase(d.tone, t)"]
    program = pulse.compile(_kernel(body)).lower(Q[0]).export_to_controller()
    answer = _answer(program)
    expression = _value(program)
    product = {"op": "mul", "type": "float", "of": [{"answer": answer, "result": 0}, {"const": 2}]}
    assert expression == {
        "terms": [
            product,
            _add("float", {"term": 0}, {"term": 0}),
            _add("float", {"term": 1}, {"term": 1}),
        ],
        **_add("float", {"term": 2}, {"term": 2}),
    }
    assert next(iter(expression)) == "terms"
    assert evaluate_expression(expression, answers={answer: [0.25, 3]}) == 4.0
    # A term one term uses is written where it is used, as it was.
    once = pulse.compile(_kernel(["shift_phase(d.tone, x * 2.0 + 0.5)"])).lower(Q[0])
    assert _value(once.export_to_controller()) == _add(
        "float", {**product, "of": [{"answer": answer, "result": 0}, {"const": 2}]}, {"const": 0.5}
    )


_DOUBLED = {
    "phase": (
        ["t = x * 1.0e-9"],
        "shift_phase(d.tone, t)\n    drive(d, square(20, 0.2))",
        0.25e-9 * 2.0**40,
    ),
    "length": (["t = x * 1.0e-9"], "wait(d, int(t), bounds=(0, 4000))", int(0.25e-9 * 2.0**40)),
    "cycles": (["t = n * 1"], "wait(d, t % 1000, bounds=(0, 4000))", 3 * 2**40 % 1000),
    "amplitude": (["t = x * 1.0e-13"], "drive(d, square(40, t))", None),
    "condition": (
        ["c = n > 3", "e = x > 0.1"],
        "if c:\n        drive(d, square(40, 0.1))\n    drive(d, square(20, 0.2))",
        None,
    ),
}


@pytest.mark.parametrize("value", _DOUBLED)
def test_a_value_that_uses_a_term_twice_forty_times_over_exports_each_term_once(value):
    first, use, expected = _DOUBLED[value]
    step = "c = (c | c) & (c | e)" if value == "condition" else "t = t + t"
    lowered = pulse.compile(_kernel([*first, *[step] * 40, use])).lower(Q[0])
    program = lowered.export_to_controller()
    assert len(json.dumps(program)) < 40_000
    rolled = lowered.export_to_controller(version=2)
    assert unroll(rolled) == program
    module = lowered._concrete()
    _, checked = module.controller_events(_PARAMETERS, _AMPLITUDES)
    _, exported, _ = module.controller_program(
        _PARAMETERS, _AMPLITUDES, lowered.classical != (), exportable=False
    )
    assert _computed(checked) == _computed(exported)
    assert list(pulse.controller.check(lowered)) == []
    if expected is not None:
        expression = _value(program)
        assert 39 <= len(expression["terms"]) <= 41
        assert evaluate_expression(expression, answers={_answer(program): [0.25, 3]}) == expected


def _chain(depth, shared):
    step = "t = t + t % 3" if shared else "t = t + 1"
    return ["t = n * 2", *[step] * depth, "wait(d, t, bounds=(0, 8000))"]


def _chained(depth, shared, n):
    t = n * 2
    for _ in range(depth):
        t = t + t % 3 if shared else t + 1
    return t


@pytest.mark.parametrize(
    ("depth", "shared"),
    [
        (63, False),
        (64, False),
        (65, False),
        (66, False),
        (130, False),
        (2000, False),
        (2000, True),
    ],
)
def test_an_expression_of_any_depth_is_written_read_back_evaluated_and_emulated(depth, shared):
    lowered = pulse.compile(_kernel(_chain(depth, shared))).lower(Q[0])
    program = lowered.export_to_controller()
    expression = _value(program)
    # No text nests deeper than _NESTED terms: a term that deep is named.
    assert _nesting(expression) <= _NESTED
    if not shared:
        # The length is a few terms over the chain.
        assert depth // _NESTED <= len(expression.get("terms", ())) <= (depth + 8) // _NESTED
    assert json.loads(json.dumps(program)) == program
    assert json.loads(json.dumps(program, indent=1)) == program
    assert copy.deepcopy(program) == program
    assert unroll(lowered.export_to_controller(version=2)) == program
    answer = _answer(program)
    for n in (2, 3, 7):
        waited = _chained(depth, shared, n)
        assert evaluate_expression(expression, answers={answer: [0.0, n]}) == waited
        assert _emulated_wait(lowered, 0.0, n) == waited


@pytest.mark.parametrize("depth", [63, 64, 65, 66, 129, 130])
def test_a_term_is_named_where_the_text_would_nest_deeper(depth):
    # The product and *depth* sums over it: depth + 1 terms deep.
    body = ["t = x * 2.0", *["t = t + 0.5"] * depth, "shift_phase(d.tone, t)"]
    expression = _value(pulse.compile(_kernel(body)).lower(Q[0]).export_to_controller())
    assert len(expression.get("terms", ())) == depth // _NESTED
    assert _nesting(expression) == min(depth + 1, _NESTED)
    assert evaluate_expression(expression, answers={2: [0.25, 3]}) == 0.5 + 0.5 * depth


@pytest.mark.parametrize(
    ("steps", "rolled"),
    [(["t = t + t", "t = t + t"], True), (["t = t + 1"] * 200, True)],
    ids=["shared", "deep"],
)
def test_a_loop_whose_iterations_compute_alike_stays_rolled(steps, rolled):
    source = (
        "def k(q):\n    d = get_drive_line(q)\n    ro = get_readout_line(q)\n"
        "    for _ in range(4):\n"
        "        x, n = _inputs.submit([readout(ro, square(100, 0.05))]).result()\n"
        "        t = n * 2\n"
        + "".join(f"        {step}\n" for step in steps)
        + "        drive(d, square(40, 0.1))\n"
        "        wait(d, t, bounds=(8, 4000))\n"
    )
    scope = {"_inputs": _inputs}
    exec(source, scope)  # noqa: S102
    lowered = pulse.compile(pulse.kernel(scope["k"])).lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    assert any(e["kind"] == "loop" and e["trips"] == 4 for e in structured["events"]) == rolled
    flat = lowered.export_to_controller()
    assert unroll(structured) == flat
    lengths = [e["length"] for e in flat["events"] if e["kind"] == "wait"]
    answers = [e["index"] for e in flat["events"] if e["kind"] == "answer"]
    assert len(lengths) == len(answers) == 4
    expected = 3 * 2 * 4 if len(steps) == 2 else 3 * 2 + 200
    for length, answer in zip(lengths, answers, strict=True):
        assert "terms" in length
        assert evaluate_expression(length, answers={answer: [0.0, 3]}) == expected
    assert list(pulse.controller.check(lowered)) == []


def test_a_float_expression_two_thousand_terms_deep_computes_what_python_computes():
    for step in ("t = t + 0.5", "t = t + t * 0.001"):
        body = ["t = x * 2.0", *[step] * 2000, "shift_phase(d.tone, t)"]
        program = pulse.compile(_kernel(body)).lower(Q[0]).export_to_controller()
        expression = _value(program)
        assert _nesting(expression) <= _NESTED
        assert json.loads(json.dumps(program, indent=1)) == program
        t = 0.25 * 2.0
        for _ in range(2000):
            t = t + 0.5 if step.endswith("0.5") else t + t * 0.001
        assert evaluate_expression(expression, answers={_answer(program): [0.25, 3]}) == t


def test_a_deep_expression_over_a_term_its_every_step_uses_computes_what_python_computes():
    body = [
        "s = x * 2.0",
        *["s = s * 1.0001 + 0.1"] * 40,
        "t = s",
        *["t = s + t * 0.5"] * 2000,
        "shift_phase(d.tone, t)",
    ]
    program = pulse.compile(_kernel(body)).lower(Q[0]).export_to_controller()
    expression = _value(program)
    assert _nesting(expression) <= _NESTED
    assert len(json.dumps(expression)) < 400_000
    s = 0.25 * 2.0
    for _ in range(40):
        s = s * 1.0001 + 0.1
    t = s
    for _ in range(2000):
        t = s + t * 0.5
    assert evaluate_expression(expression, answers={_answer(program): [0.25, 3]}) == t


def test_an_expression_without_a_term_used_twice_or_deep_has_no_terms():
    for body in (*_PHASES.values(), *_WAITS.values(), ["wait(d, n * 4, bounds=(8, 400))"]):
        expression = _value(pulse.compile(_kernel(body)).lower(Q[0]).export_to_controller())
        assert "term" not in json.dumps(expression)


def test_a_controller_clock_converts_a_length_that_names_terms():
    machine = ControlSystem(controllers=[Controller(clock_ghz=2.0, cycles_per_timestep=4)])
    body = ["t = n * 2", "wait(d, t + t, bounds=(8, 400))"]
    program = (
        pulse.compile(_kernel(body), control_system=machine).lower(Q[0]).export_to_controller()
    )
    expression = _value(program)
    assert next(iter(expression)) == "terms"
    assert expression["op"] == "controller_cycles"
    assert len(expression["terms"]) == 1
    assert "terms" not in json.dumps({**expression, "terms": None}["of"])
    answer = _answer(program)
    for n in (2, 7, 100):
        got = evaluate_expression(expression, answers={answer: [0.0, n]})
        assert got == _folded(body, 0.0, n, machine) == 4 * n
    for n, cycles in ((1, 8), (150, 400)):
        assert evaluate_expression(expression, answers={answer: [0.0, n]}) == cycles


def test_a_length_on_a_controller_clock_nests_three_terms_more_at_most():
    machine = ControlSystem(controllers=[Controller(clock_ghz=2.0, cycles_per_timestep=4)])
    deepest = 0
    for depth in (10, *range(56, 68), 300):
        body = ["t = n * 2", *["t = t + 1"] * depth, "wait(d, t, bounds=(8, 4000))"]
        lowered = pulse.compile(_kernel(body), control_system=machine).lower(Q[0])
        plain = _value(pulse.compile(_kernel(body)).lower(Q[0]).export_to_controller())
        expression = _value(lowered.export_to_controller())
        assert expression["op"] == "controller_cycles"
        assert _nesting(plain) <= _NESTED
        # Its bounds and the conversion are terms over the length.
        assert _nesting(plain) <= _nesting(expression) <= _nesting(plain) + 3
        deepest = max(deepest, _nesting(expression))
    assert deepest == _NESTED + 3


def test_a_named_term_is_computed_once_and_only_from_earlier_terms():
    zero = {"op": "div", "type": "int", "of": [{"const": 1}, {"const": 0}]}
    twice = _add("int", {"term": 0}, {"term": 0})
    with pytest.raises(ExecutionError, match="div of 1 by 0"):
        evaluate_expression({"terms": [zero], **twice})
    seven = _add("int", {"const": 3}, {"const": 4})
    assert (
        evaluate_expression({"terms": [seven, twice], **_add("int", {"term": 1}, {"term": 0})})
        == 21
    )
    for terms, use in (
        ([seven], {"term": 1}),
        ([seven], {"term": -1}),
        ([seven], {"term": True}),
        ([seven], {"term": 0.0}),
        ([seven], {"term": [0]}),
        ([twice], {"term": 0}),
        ([seven, _add("int", {"term": 2}, {"const": 1}), seven], {"term": 1}),
    ):
        for of in ([use, {"const": 1}], [{"term": 0}, use]):
            with pytest.raises(InvalidValueError, match="not an earlier term"):
                evaluate_expression({"terms": terms, **_add("int", *of)})
    # A later term is no operand of an earlier one, computed already or not.
    later = [_add("int", {"term": 1}, {"const": 1}), seven]
    for of in ([{"term": 1}, {"term": 0}], [{"term": 0}, {"term": 1}]):
        with pytest.raises(InvalidValueError, match="not an earlier term"):
            evaluate_expression({"terms": later, **_add("int", *of)})
    with pytest.raises(InvalidValueError, match="not an earlier term"):
        evaluate_expression({"terms": [], **_add("int", {"term": 0}, {"const": 1})})
    inner = {"terms": [seven], **_add("int", {"term": 0}, {"const": 0})}
    for whole in (_add("int", inner, {"const": 0}), {"terms": [inner], **twice}):
        with pytest.raises(InvalidValueError, match="only a whole expression has terms"):
            evaluate_expression(whole)

    class Counted(list):
        asked = 0

        def __getitem__(self, key):
            Counted.asked += 1
            return super().__getitem__(key)

    read = _add("int", {"answer": 4, "result": 0}, {"const": 1})
    four = {"terms": [read, twice], **_add("int", {"term": 1}, {"term": 1})}
    assert evaluate_expression(four, answers={4: Counted([2])}) == 12
    assert Counted.asked == 1
    with pytest.raises(InvalidValueError, match="term of no expression"):
        evaluate_expression({"term": 0})


# ── Forms that mean the same ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "forms",
    [
        [_if("x > 0.1", "x", "0.1"), _if("0.1 < x", "x", "0.1")],
        [_if("x < 0.1", "x", "0.1"), _if("0.1 > x", "x", "0.1")],
        [["shift_phase(d.tone, x * 2.0)"], ["shift_phase(d.tone, 2.0 * x)"]],
        [_if("m", "0.3", "0.1"), _if("not m", "0.1", "0.3")],
    ],
    ids=["float max", "float min", "mul", "readout"],
)
def test_an_expression_written_in_equivalent_ways_exports_the_same(forms):
    exported = [_value(pulse.compile(_kernel(f)).lower(Q[0]).export_to_controller()) for f in forms]
    assert all(e == exported[0] for e in exported), exported
    assert _ops(exported[0]) & {"max", "min", "mul", "select"}


def _computing(form):
    """The program shifting by ``n * 0.5`` with ``n`` replaced by *form*, MLIR
    computing ``%form`` from ``%n`` and ``%three``."""
    lowered = pulse.compile(_kernel(["shift_phase(d.tone, n * 0.5)"])).lower(Q[0])
    text = lowered._concrete().print()
    line = next(line for line in text.splitlines() if "arith.sitofp" in line)
    result, n = re.match(r"\s*(%\S+) = arith.sitofp (%\S+) :", line).groups()
    computed = "\n".join(
        [
            "%three = arith.constant 3 : i64",
            *form.replace("%n", n).split("; "),
            f"{result} = arith.sitofp %form : i64 to f64",
        ]
    )
    module = MLIRPipeline().parse_to_module(text.replace(line, computed))
    object.__setattr__(lowered, "_module", module)
    return lowered.export_to_controller()


@pytest.mark.parametrize(
    ("extremum", "forms"),
    [
        (
            "max",
            [
                "%form = arith.maxsi %n, %three : i64",
                "%c = arith.cmpi sgt, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi sge, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi slt, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
                "%c = arith.cmpi sle, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
            ],
        ),
        (
            "min",
            [
                "%form = arith.minsi %n, %three : i64",
                "%c = arith.cmpi slt, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi sle, %n, %three : i64; %form = arith.select %c, %n, %three : i64",
                "%c = arith.cmpi sgt, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
                "%c = arith.cmpi sge, %n, %three : i64; %form = arith.select %c, %three, %n : i64",
            ],
        ),
    ],
)
def test_the_integer_max_and_min_export_the_same_however_computed(extremum, forms):
    exported = [_computing(form) for form in forms]
    answer = {"answer": _answer(exported[0]), "result": 1}
    integer = {"op": extremum, "type": "int", "of": [answer, {"const": 3}]}
    for program in exported:
        assert _value(program) == {
            "op": "mul",
            "type": "float",
            "of": [{"op": "to_float", "type": "int", "of": [integer]}, {"const": 0.5}],
        }


def test_a_float_max_keeps_the_order_of_its_operands_which_nan_tells_apart():
    def exported(form):
        program = pulse.compile(_kernel(form)).lower(Q[0]).export_to_controller()
        return _value(program), _answer(program)

    (one, answer), (other, _) = (
        exported(_if("x > 0.1", "x", "0.1")),
        exported(_if("x < 0.1", "0.1", "x")),
    )
    assert one["op"] == other["op"] == "max"
    assert other["of"] == one["of"][::-1]
    # As the kernels run: x > 0.1 is false for NaN, and so is x < 0.1.
    assert evaluate_expression(one, answers={answer: [math.nan, 0]}) == 0.1
    assert math.isnan(evaluate_expression(other, answers={answer: [math.nan, 0]}))


# ── Branch conditions ────────────────────────────────────────────────────────

_BRANCH = ["if n > 3:", "    drive(d, square(40, 0.1))", "drive(d, square(20, 0.2))"]


def _branching(form):
    """The program driving first only when ``%cond``, MLIR computing it by
    *form* from the answers ``%x`` and ``%n`` and the measurement ``%m``."""
    lowered = pulse.compile(_kernel(_BRANCH)).lower(Q[0])
    text = lowered._concrete().print()
    line = next(line for line in text.splitlines() if "arith.cmpi sgt" in line)
    condition, answers = re.match(r"\s*(%\S+) = arith.cmpi sgt, (%[^#\s]+)#1,", line).groups()
    measurement = re.search(r"(%\S+) = pulse.readout ", text).group(1).split(", ")[-1]
    constants = [
        "%one = arith.constant 1 : i64",
        "%two = arith.constant 2 : i64",
        "%three = arith.constant 3 : i64",
        "%tenth = arith.constant 0.1 : f64",
        "%true = arith.constant true",
        "%false = arith.constant false",
    ]
    computed = "; ".join([*constants, form]).replace("%cond", condition)
    for name, value in (("%x", f"{answers}#0"), ("%n", f"{answers}#1"), ("%m", measurement)):
        computed = re.sub(rf"{name}\b", value, computed)
    module = MLIRPipeline().parse_to_module(text.replace(line, "\n".join(computed.split("; "))))
    object.__setattr__(lowered, "_module", module)
    return lowered


def _condition(program):
    """The condition of the drive played only on one arm."""
    if program["format"] != FORMAT:
        program = unroll(program)
    [condition] = next(e["conditions"] for e in program["events"] if e["conditions"])
    return condition


_CONDITIONS = {
    "select compared": (
        "%big = arith.cmpf ogt, %x, %tenth : f64; "
        "%z = arith.select %big, %n, %two : i64; %cond = arith.cmpi eq, %z, %three : i64"
    ),
    "integer arithmetic": (
        "%a = arith.muli %n, %three : i64; %b = arith.floordivsi %a, %two : i64; "
        "%c = arith.remsi %n, %two : i64; %d = arith.subi %b, %c : i64; "
        "%cond = arith.cmpi sge, %d, %three : i64"
    ),
    "float arithmetic": (
        "%f = arith.sitofp %n : i64 to f64; %g = arith.mulf %f, %x : f64; "
        "%h = arith.subf %g, %tenth : f64; %cond = arith.cmpf olt, %h, %x : f64"
    ),
    "and": (
        "%p = arith.cmpi sgt, %n, %three : i64; %q = arith.cmpf ogt, %x, %tenth : f64; "
        "%cond = arith.andi %p, %q : i1"
    ),
    "or": (
        "%p = arith.cmpi sgt, %n, %three : i64; %q = arith.cmpf ogt, %x, %tenth : f64; "
        "%cond = arith.ori %p, %q : i1"
    ),
    "exclusive or": (
        "%p = arith.cmpi sgt, %n, %three : i64; %q = arith.cmpf ogt, %x, %tenth : f64; "
        "%cond = arith.xori %p, %q : i1"
    ),
    "readout plus answer": (
        "%s = pulse.is_set %m : !pulse.measurement -> i1; %e = arith.extui %s : i1 to i64; "
        "%t = arith.addi %e, %n : i64; %cond = arith.cmpi sgt, %t, %three : i64"
    ),
    "readout selects": (
        "%s = pulse.is_set %m : !pulse.measurement -> i1; "
        "%z = arith.select %s, %x, %tenth : f64; %cond = arith.cmpf oge, %z, %x : f64"
    ),
    "readout level": (
        "%k = pulse.outcome %m : !pulse.measurement -> i64; "
        "%l = arith.muli %k, %n : i64; %cond = arith.cmpi slt, %l, %three : i64"
    ),
    # Terms that several others use.
    "a comparison and its negation": (
        "%le = arith.cmpi sle, %n, %three : i64; %not = arith.xori %le, %true : i1; "
        "%q = arith.cmpf ogt, %x, %tenth : f64; %a = arith.andi %le, %q : i1; "
        "%b = arith.ori %not, %q : i1; %cond = arith.xori %a, %b : i1"
    ),
    "a negation and its comparison": (
        "%le = arith.cmpi sle, %n, %three : i64; %not = arith.xori %le, %true : i1; "
        "%q = arith.cmpf ogt, %x, %tenth : f64; %b = arith.ori %not, %q : i1; "
        "%a = arith.andi %le, %q : i1; %cond = arith.xori %b, %a : i1"
    ),
    "a comparison negated twice": (
        "%le = arith.cmpi sle, %n, %three : i64; %not = arith.xori %le, %true : i1; "
        "%e = arith.cmpi eq, %le, %false : i1; %q = arith.cmpf ogt, %x, %tenth : f64; "
        "%a = arith.andi %not, %q : i1; %cond = arith.ori %a, %e : i1"
    ),
    "a negation used twice": (
        "%le = arith.cmpi sle, %n, %three : i64; %not = arith.xori %le, %true : i1; "
        "%q = arith.cmpf ogt, %x, %tenth : f64; %a = arith.andi %not, %q : i1; "
        "%b = arith.ori %not, %le : i1; %cond = arith.xori %a, %b : i1"
    ),
    "a float comparison and its negation": (
        "%le = arith.cmpf ole, %x, %tenth : f64; %not = arith.xori %le, %true : i1; "
        "%p = arith.cmpi sgt, %n, %three : i64; %a = arith.andi %le, %p : i1; "
        "%cond = arith.ori %a, %not : i1"
    ),
    "a level squared": (
        "%k = pulse.outcome %m : !pulse.measurement -> i64; %l = arith.muli %k, %k : i64; "
        "%j = arith.addi %l, %n : i64; %cond = arith.cmpi sgt, %j, %three : i64"
    ),
    "a readout set and not": (
        "%s = pulse.is_set %m : !pulse.measurement -> i1; %p = arith.cmpi sgt, %n, %three : i64; "
        "%a = arith.andi %s, %p : i1; %b = arith.xori %s, %true : i1; %cond = arith.ori %a, %b : i1"
    ),
    "a choice doubled": (
        "%big = arith.cmpf ogt, %x, %tenth : f64; %z = arith.select %big, %n, %two : i64; "
        "%y = arith.addi %z, %z : i64; %w = arith.muli %y, %z : i64; "
        "%cond = arith.cmpi sge, %w, %three : i64"
    ),
    "a max and its comparison": (
        "%c = arith.cmpi sgt, %n, %three : i64; %mx = arith.select %c, %n, %three : i64; "
        "%e = arith.cmpi sge, %mx, %two : i64; %cond = arith.andi %c, %e : i1"
    ),
    "a sum of its sums": (
        "%a = arith.muli %n, %three : i64; %b = arith.addi %a, %a : i64; "
        "%c = arith.addi %b, %a : i64; %d = arith.subi %c, %b : i64; "
        "%cond = arith.cmpi sge, %d, %three : i64"
    ),
    "an answer squared": (
        "%f = arith.mulf %x, %x : f64; %g = arith.addf %f, %f : f64; "
        "%cond = arith.cmpf ogt, %g, %tenth : f64"
    ),
}
_BRANCH_INPUTS = [
    (x, n, level) for x in (0.05, 0.1, 0.25, math.nan) for n in (2, 3, 4, 6) for level in (0, 1)
]


@pytest.mark.parametrize("form", _CONDITIONS.values(), ids=_CONDITIONS)
def test_a_branch_condition_runs_the_arm_the_emulation_runs(form):
    lowered = _branching(form)
    condition = _condition(lowered.export_to_controller())
    assert condition["test"] == "expression"
    assert _condition(lowered.export_to_controller(version=2)) == condition
    args = lowered.options.schedule_args()
    module = lowered._concrete().simulation_module(*args)
    readout, iteration = module.next_measurement([])[:2]
    for x, n, level in _BRANCH_INPUTS:
        path = [[readout, *iteration, level]]
        bound = module.bind_call(module.next_call(path)[0], [x, float(n)], *args)
        played = any(drive[0] == 0 for drive in bound.drive_frames(path)[0])
        value = evaluate_expression(
            condition["expression"],
            readouts={r: level for r in condition["readouts"]},
            answers={a: [x, n] for a in condition["answers"]},
        )
        assert type(value) is bool
        assert (value == condition["arm"]) == played, (x, n, level)


@pulse.kernel
def _looped(q):
    d = get_drive_line(q)  # noqa: F821
    ro = get_readout_line(q)  # noqa: F821
    x, n = _inputs.submit([readout(ro, square(100, 0.05))]).result()  # noqa: F821
    for _ in range(4):
        m = readout(ro, square(100, 0.05))  # noqa: F821
        if m:
            drive(d, square(40, 0.1))  # noqa: F821
        if n > 3:
            drive(d, square(30, 0.1))  # noqa: F821
        wait(d, 200)  # noqa: F821


def test_a_condition_in_a_rolled_loop_computes_from_the_readout_of_its_iteration():
    lowered = pulse.compile(_looped).lower(Q[0])
    text = lowered._concrete().print()
    line = next(line for line in text.splitlines() if "pulse.is_set" in line)
    condition, measurement = re.match(r"\s*(%\S+) = pulse.is_set (%\S+) :", line).groups()
    level = (
        f"%k = pulse.outcome {measurement} : !pulse.measurement -> i64\n"
        f"%one = arith.constant 1 : i64\n{condition} = arith.cmpi eq, %k, %one : i64"
    )
    module = MLIRPipeline().parse_to_module(text.replace(line, level))
    object.__setattr__(lowered, "_module", module)
    structured = lowered.export_to_controller(version=2)
    [loop] = (e for e in structured["events"] if e["kind"] == "loop")
    assert loop["trips"] == 4
    flat = lowered.export_to_controller()
    assert [e["conditions"] for e in unroll(structured)["events"]] == [
        e["conditions"] for e in flat["events"]
    ]
    readouts = [e["index"] for e in flat["events"] if e["kind"] == "readout"][1:]
    tested = [
        c["expression"]["of"][0]
        for e in flat["events"]
        for c in e["conditions"]
        if c["expression"]["op"] == "eq"
    ]
    assert tested == [{"readout": r} for r in readouts]


def test_a_condition_of_the_kernel_exports_its_expression():
    lowered = pulse.compile(_kernel(["if (n * 3 - 4) // 2 >= n % 2 + 1:", *_BRANCH[1:]]))
    condition = _condition(lowered.lower(Q[0]).export_to_controller())
    assert condition["test"] == "expression"
    assert {"ge", "floor_div", "rem", "mul", "sub"} <= _ops(condition["expression"])
    answers = {a: [0.0, 4] for a in condition["answers"]}
    assert evaluate_expression(condition["expression"], answers=answers) is True


@pytest.mark.parametrize(
    "forms",
    [
        [
            "%cond = arith.cmpi sgt, %n, %three : i64",
            "%cond = arith.cmpi slt, %three, %n : i64",
            "%le = arith.cmpi sle, %n, %three : i64; %cond = arith.xori %le, %true : i1",
            "%le = arith.cmpi sle, %n, %three : i64; %cond = arith.xori %true, %le : i1",
            "%le = arith.cmpi sle, %n, %three : i64; %cond = arith.cmpi eq, %le, %false : i1",
            "%le = arith.cmpi sle, %n, %three : i64; %cond = arith.select %le, %false, %true : i1",
        ],
        [
            "%cond = arith.cmpf ogt, %x, %tenth : f64",
            "%cond = arith.cmpf olt, %tenth, %x : f64",
            "%gt = arith.cmpf ogt, %x, %tenth : f64; %cond = arith.cmpi ne, %gt, %false : i1",
            "%gt = arith.cmpf ogt, %x, %tenth : f64; %cond = arith.select %gt, %true, %false : i1",
        ],
        [
            "%cond = pulse.is_set %m : !pulse.measurement -> i1",
            "%k = pulse.outcome %m : !pulse.measurement -> i64; "
            "%zero = arith.constant 0 : i64; %cond = arith.cmpi ne, %k, %zero : i64",
            "%k = pulse.outcome %m : !pulse.measurement -> i64; "
            "%zero = arith.constant 0 : i64; %cond = arith.cmpi ne, %zero, %k : i64",
        ],
        [
            "%p = arith.cmpi sgt, %n, %three : i64; %cond = arith.andi %p, %true : i1",
            "%cond = arith.cmpi sgt, %n, %three : i64",
        ],
    ],
    ids=["integer", "float", "readout", "and true"],
)
def test_a_condition_written_in_equivalent_ways_exports_the_same(forms):
    exported = [_condition(_branching(form).export_to_controller()) for form in forms]
    assert all(e == exported[0] for e in exported), exported


def test_a_float_condition_negated_keeps_what_nan_tells_apart():
    # not (x <= 0.1) holds of NaN, which x > 0.1 does not.
    negated = _condition(
        _branching(
            "%le = arith.cmpf ole, %x, %tenth : f64; %cond = arith.xori %le, %true : i1"
        ).export_to_controller()
    )
    greater = _condition(
        _branching("%cond = arith.cmpf ogt, %x, %tenth : f64").export_to_controller()
    )
    assert negated["expression"] != greater["expression"]
    answers = {a: [math.nan, 0] for a in negated["answers"]}
    assert evaluate_expression(negated["expression"], answers=answers) is True
    assert evaluate_expression(greater["expression"], answers=answers) is False


def test_a_condition_a_plain_test_names_has_no_expression():
    branch = ["if m:", *_BRANCH[1:]]
    program = pulse.compile(_kernel(branch)).lower(Q[0]).export_to_controller()
    assert _condition(program) == {
        "branch": 0,
        "arm": True,
        "readouts": [0],
        "answers": [],
        "test": "is_set",
    }


# ── Conditions the kernel frontend traces ────────────────────────────────────

# Answers n (int), x (float) and b (bool), and a readout m: the comparisons of
# conditions, bool() and `not` that the frontend lowers to i1 compares.
_TRACED = [
    "(n > 3) == (x > 0.1)",
    "(n > 3) ^ (x > 0.1)",
    "((n > 3) == (x > 0.1)) == (n < 7)",
    "not ((n > 3) == (x > 0.1))",
    "b == (x > 0.1)",
    "(n > 3) == False",
    "(x > 0.1) == False",
    "(n > 3) == 2",
    "b != True",
    "bool(n)",
    "x == n",
]


def _traced_events(condition: str, version: int) -> list[dict]:
    lowered = pulse.compile(_traced(condition)).lower(Q[0])
    return lowered.export_to_controller(version=version)["events"]


def _traced_condition(condition: str, version: int) -> dict:
    events = _traced_events(condition, version)
    [played] = (e for e in events if e["kind"] in ("drive", "play") and e.get("conditions"))
    [found] = played["conditions"]
    return found


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("condition", _TRACED)
def test_a_traced_condition_exports_the_branch_python_takes(condition, version):
    events = _traced_events(condition, version)
    [played] = (e for e in events if e["kind"] in ("drive", "play") and e.get("conditions"))
    [found] = played["conditions"]
    assert found["test"] == "expression"
    n, x, b = (i for i, e in enumerate(events) if e["kind"] == "answer")
    for values in itertools.product((0, 5, 9), (0.05, 0.5, math.nan), (False, True)):
        answers = dict(zip((n, x, b), ([v] for v in values), strict=True))
        held = evaluate_expression(found["expression"], answers=answers)
        expected = eval(condition, {}, dict(zip("nxb", values, strict=True)))  # noqa: S307
        assert (held == found["arm"]) == bool(expected), values


@pytest.mark.parametrize(
    "forms",
    [
        ["(n > 3) == True", "True == (n > 3)", "(n > 3) != False", "bool(n > 3)", "(n > 3) == 1"],
        ["(n > 3) == False", "(n > 3) != True"],
        ["(n > 3) != (x > 0.1)", "(n > 3) ^ (x > 0.1)"],
        ["b != True", "b == False"],
    ],
    ids=["is", "is not", "differ", "bool is not"],
)
def test_a_traced_condition_written_in_equivalent_ways_exports_the_same(forms):
    exported = [_traced_condition(form, 1) for form in forms]
    assert all(e == exported[0] for e in exported), exported


@pytest.mark.parametrize(
    ("condition", "arm"), [("m", True), ("bool(m)", True), ("not m", False), ("bool(not m)", False)]
)
def test_the_truth_of_a_readout_is_its_plain_test(condition, arm):
    found = _traced_condition(condition, 1)
    assert found["test"] == "is_set" and found["arm"] is arm and "expression" not in found


# ── No operation of the compiler in any export ───────────────────────────────

_DIALECT = re.compile(
    r"(?<![\w-])(arith|math|scf|cf|func|llvm|index|pulse|classical|affine|builtin)\.\w"
)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _no_dialect(program):
    prefixed = [s for s in _strings(program) if _DIALECT.search(s)]
    assert prefixed == []


@pytest.mark.parametrize("version", [1, 2])
def test_no_export_of_the_corpus_names_an_operation_of_the_compiler(version):
    dev = _validation._device(ControlSystem())
    for kernel in _validation._corpus().values():
        lowered = pulse.compile(kernel, quantum_system=dev).lower(*dev.qudits[:2])
        _no_dialect(lowered.export_to_controller(version=version))
    # Format /2 is for controllers on the program clock.
    clocks = [Controller(clock_ghz=2.0, cycles_per_timestep=4)] if version == 1 else []
    machine = ControlSystem(controllers=clocks)
    for body in (
        *_PHASES.values(),
        *_WAITS.values(),
        ["wait(d, n * 4, bounds=(8, 400))"],
        _BRANCH,
        ["if (n * 3 - 4) // 2 >= n % 2 + 1:", *_BRANCH[1:]],
    ):
        lowered = pulse.compile(_kernel(body), control_system=machine).lower(Q[0])
        _no_dialect(lowered.export_to_controller(version=version))


def _computed(events):
    """What each event computes at run time: its length and bounds, its
    submitted values, its waveform's operands and its frame changes."""

    def operands(node):
        if not isinstance(node, dict):
            return []
        found = [(node["shape"], node["runtime"])] if "runtime" in node else []
        for child in node.get("of", ()):
            found += operands(child)
        return found

    keys = ("length", "min_cycles", "max_cycles", "operands")
    return [
        (
            event["kind"],
            {key: event[key] for key in keys if key in event},
            operands(event.get("waveform")),
            [(update["op"], update["value"]) for update in event.get("frame_updates", ())],
        )
        for event in events
    ]


@pytest.mark.parametrize(
    "body",
    [
        *([*body, "drive(d, square(20, 0.2))"] for body in _PHASES.values()),
        *_WAITS.values(),
        ["wait(d, n * 4, bounds=(8, 400))"],
        _BRANCH,
        ["if (n * 3 - 4) // 2 >= n % 2 + 1:", *_BRANCH[1:]],
    ],
)
def test_the_check_walk_computes_at_run_time_as_the_export(body):
    lowered = pulse.compile(_kernel(body)).lower(Q[0])
    module = lowered._concrete()
    _, checked = module.controller_events(_PARAMETERS, _AMPLITUDES)
    _, exported, _ = module.controller_program(
        _PARAMETERS, _AMPLITUDES, lowered.classical != (), exportable=False
    )
    assert _computed(checked) == _computed(exported)
    _no_dialect(checked)


@pytest.mark.parametrize("version", [1, 2])
def test_an_operation_outside_the_vocabulary_is_an_export_error(version):
    lowered = pulse.compile(_kernel(["shift_phase(d.tone, x / 3.0)"])).lower(Q[0])
    text = lowered._concrete().print()
    assert "arith.divf" in text
    object.__setattr__(
        lowered, "_module", MLIRPipeline().parse_to_module(text.replace("arith.divf", "arith.remf"))
    )
    with pytest.raises(UnsupportedError, match="'arith.remf'.*expression vocabulary"):
        lowered.export_to_controller(version=version)


# ── The terms' semantics ─────────────────────────────────────────────────────


def _term(op, kind, *operands, **extra):
    return {"op": op, "type": kind, "of": [{"const": v} for v in operands], **extra}


_INT_MIN, _INT_MAX = -(2**63), 2**63 - 1


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        (_term("div", "int", 7, -2), -3),
        (_term("div", "int", -7, 2), -3),
        (_term("floor_div", "int", 7, -2), -4),
        (_term("floor_div", "int", -7, 2), -4),
        (_term("rem", "int", -7, 2), -1),
        (_term("rem", "int", 7, -2), 1),
        (_term("add", "int", _INT_MAX, 1), _INT_MIN),
        (_term("mul", "int", 2**62, 4), 0),
        (_term("neg", "int", _INT_MIN), _INT_MIN),
        (_term("sub", "int", _INT_MIN, 1), _INT_MAX),
        (_term("div", "float", 1, 0), math.inf),
        (_term("div", "float", -1, 0), -math.inf),
        (_term("div", "float", 1, -0.0), -math.inf),
        (_term("add", "float", 1, 2), 3.0),
        (_term("to_float", "int", 2**53 + 1), float(2**53)),
        (_term("to_int", "float", -2.7), -2),
        (_term("to_int", "float", 2.7), 2),
        (_term("max", "int", 3, 5), 5),
        (_term("min", "float", -0.0, 0.0), 0.0),
        (_term("lt", "float", math.nan, 1.0), False),
        (_term("ne", "float", math.nan, math.nan), True),
        (_term("eq", "float", math.nan, math.nan), False),
        (_term("eq", "bool", True, True), True),
        (_term("select", "int", False, 1, 2), 2),
        (_term("controller_cycles", "int", 12, num=1, den=2, timestep=4), 6),
    ],
)
def test_a_term_computes_as_the_guide_specifies(expression, value):
    got = evaluate_expression(expression)
    assert got == value and type(got) is type(value)


def test_a_float_term_of_nan_is_as_ieee_754_computes_it():
    assert math.isnan(evaluate_expression(_term("div", "float", 0, 0)))
    assert math.isnan(evaluate_expression(_term("add", "float", math.inf, -math.inf)))
    assert evaluate_expression(_term("max", "float", math.nan, 1.0)) == 1.0
    assert math.isnan(evaluate_expression(_term("max", "float", 1.0, math.nan)))
    assert math.isnan(evaluate_expression(_term("min", "float", 1.0, math.nan)))


@pytest.mark.parametrize(
    "expression",
    [
        _term("div", "int", 1, 0),
        _term("floor_div", "int", 1, 0),
        _term("rem", "int", 1, 0),
        _term("div", "int", _INT_MIN, -1),
        _term("rem", "int", _INT_MIN, -1),
        _term("to_int", "float", math.nan),
        _term("to_int", "float", math.inf),
        _term("to_int", "float", 2.0**63),
        _term("controller_cycles", "int", 10, num=1, den=2, timestep=4),
    ],
)
def test_what_has_no_value_fails_the_run(expression):
    with pytest.raises(ExecutionError, match="fails the run"):
        evaluate_expression(expression)


def test_an_expression_outside_the_vocabulary_or_its_inputs_is_refused():
    with pytest.raises(InvalidValueError, match="'arith.addf' is not a term"):
        evaluate_expression({"op": "arith.addf", "of": [{"const": 1}, {"const": 2}]})
    with pytest.raises(InvalidValueError, match="readout event 4"):
        evaluate_expression({"readout": 4})
    with pytest.raises(InvalidValueError, match="result 1 given for answer event 2"):
        evaluate_expression({"answer": 2, "result": 1}, answers={2: [0.5]})
