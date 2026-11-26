# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A value a kernel computes from a term it uses several times, or through
thousands of operations, compiles in time that grows with its operations.

``tau = tau + tau`` written forty times is forty operations, each using the
one before it twice: written out it has 2**40 terms. Every walk of such a
value visits an operation once, and its texts name a shared term once.
"""

from __future__ import annotations

import contextlib
import math
import random
import signal

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import (
    ARITHMETIC,
    AnswerValue,
    Parameter,
    ParameterExpression,
    SymbolicValueError,
    UndecidableComparison,
    arithmetic_source,
    arithmetic_text,
    contains_answer,
    expression_leaves,
    expression_nodes,
    is_arithmetic,
    parameter_names,
    without_answers,
)

from tests._handles import Q

DOUBLINGS = 40


@contextlib.contextmanager
def _within(seconds: int):
    """Fail the test, where it would otherwise hang the suite, if its body
    takes longer: a walk that visits a doubled term as written out does not
    end."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def late(*_: object) -> None:
        raise AssertionError(f"did not finish in {seconds} s")

    before = signal.signal(signal.SIGALRM, late)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, before)


def _kernel(body: str, signature: str = "q, tau0: float", **names: object) -> pulse.Kernel:
    source = f"def k({signature}):\n    d = get_drive_line(q)\n{body}"
    scope = dict(vars(pulse))
    scope.update(names)
    exec(compile(source, "<shared terms>", "exec"), scope)
    return pulse.kernel(scope["k"])


def _doubled(name: str, times: int = DOUBLINGS) -> str:
    return f"    {name} = {name} + {name}\n" * times


def test_a_wait_length_doubled_forty_times_compiles():
    k = _kernel("    tau = tau0 * 1.0\n" + _doubled("tau") + "    wait(d, tau, bounds=(0, 4000))\n")
    with _within(20):
        lowered = k.lower(Q[0], 100 / 2**DOUBLINGS)
        assert lowered.schedule[0].duration_cycles == 100
        assert k.lower(Q[0], 7 / 2**DOUBLINGS).schedule[0].duration_cycles == 7


def test_the_error_for_a_doubled_length_names_each_term_once():
    k = _kernel("    tau = tau0 * 1.0\n" + _doubled("tau") + "    wait(d, tau, bounds=(0, 4000))\n")
    with _within(20), pytest.raises(ValueError) as raised:
        k.lower(Q[0], 1.0)
    message = str(raised.value)
    assert "the wait length t40 + t40 with t1 = tau0 * 1.0, t2 = t1 + t1, " in message
    assert "t40 = t39 + t39" in message
    assert len(message) < 1200


def test_an_amplitude_and_a_phase_from_a_chain_of_two_thousand_sums_compile():
    body = (
        "    amp = tau0 * 1.0\n"
        "    for i in range(2000):\n"
        "        amp = amp + 1.0e-5\n"
        "    drive(d, square(40, amp))\n"
        "    shift_phase(d.tone, amp)\n"
    )
    with _within(20):
        lowered = _kernel(body).lower(Q[0], 0.01)
    assert [e.kind for e in lowered.schedule].count("drive") == 1


def test_a_wait_length_from_a_chain_of_two_thousand_sums_compiles():
    body = (
        "    tau = tau0 * 1.0\n"
        "    for i in range(2000):\n"
        "        tau = tau + 1.0\n"
        "    wait(d, tau, bounds=(0, 40000))\n"
    )
    k = _kernel(body)
    with _within(20):
        assert k.lower(Q[0], 10.0).schedule[0].duration_cycles == 2010
        with pytest.raises(ValueError, match="outside its bounds"):
            k.lower(Q[0], 50000.0)


@pytest.mark.parametrize("logic", ["c & c", "(c | c) & (c | e)", "c ^ e ^ c"])
def test_a_condition_doubled_forty_times_compiles(logic):
    @pulse.classical(max_latency_ns=50, name="shared_terms_flag")
    def flag(x: float) -> bool:
        return x > 1.0

    body = (
        "    c = flag.submit(tau0).result()\n"
        "    e = flag.submit(tau0 + 1.0).result()\n"
        "    c = c & e\n"
        + f"    c = {logic}\n" * DOUBLINGS
        + "    if c:\n        drive(d, square(40, 0.1))\n"
    )
    with _within(20):
        text = _kernel(body, flag=flag).lower(Q[0], 2.0).mlir()
    # One operation of the condition per line of the kernel, a few at most.
    assert text.count("\n") < 40 * DOUBLINGS


def test_ands_between_comparisons_of_conditions_compile_as_deep_as_they_did():
    @pulse.classical(max_latency_ns=50, name="shared_terms_mixed_flag")
    def flag(x: float) -> bool:
        return x > 1.0

    body = (
        "    c = flag.submit(tau0).result()\n"
        "    e = flag.submit(tau0 + 1.0).result()\n"
        "    f = flag.submit(tau0 + 2.0).result()\n"
        "    for i in range(120):\n"
        "        c = (c & e) ^ f\n"
        "    if c:\n        drive(d, square(40, 0.1))\n"
    )
    with _within(60):
        assert "pulse.drive" in _kernel(body, flag=flag).lower(Q[0], 2.0).mlir()


def test_a_condition_of_two_thousand_ands_compiles():
    @pulse.classical(max_latency_ns=50, name="shared_terms_chain_flag")
    def flag(x: float) -> bool:
        return x > 1.0

    body = (
        "    c = flag.submit(tau0).result()\n"
        "    e = flag.submit(tau0 + 1.0).result()\n"
        "    for i in range(2000):\n"
        "        c = c & e\n"
        "    if c:\n        drive(d, square(40, 0.1))\n"
    )
    with _within(60):
        text = _kernel(body, flag=flag).lower(Q[0], 2.0).mlir()
    assert "pulse.drive" in text


# -- The walks against the same walks of the value written out -------------------


def _written_names(value) -> tuple[str, ...]:
    if isinstance(value, Parameter):
        return (value.name,)
    if isinstance(value, (ParameterExpression, UndecidableComparison)):
        return _written_names(value.lhs) + _written_names(value.rhs)
    return ()


def _written_source(value) -> str | None:
    """``arithmetic_source`` as it was: every operation written in place."""
    if isinstance(value, Parameter):
        return f"a[{value.index}]"
    if isinstance(value, ParameterExpression):
        if value.op == "neg":
            inner = _written_source(value.lhs)
            return None if inner is None else f"(-{inner})"
        if value.op == "cast":
            return _written_source(value.lhs) if value.dtype == "f64" else None
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(value.op)
        lhs, rhs = _written_source(value.lhs), _written_source(value.rhs)
        if symbol is None or lhs is None or rhs is None:
            return None
        return f"({lhs} {symbol} {rhs})"
    if isinstance(value, (int, float)):
        return repr(float(value))
    return None


_PRECEDENCE = {"add": 1, "sub": 1, "mul": 2, "div": 2}


def _written_text(value) -> str:
    """``arithmetic_text`` as it was."""
    if isinstance(value, Parameter):
        return value.name
    if isinstance(value, AnswerValue):
        return value.label or "answer"
    if isinstance(value, ParameterExpression):
        if value.op == "neg":
            return f"-{_written_operand(value.lhs, 3)}"
        if value.op == "cast":
            return _written_text(value.lhs)
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(value.op, value.op)
        level = _PRECEDENCE.get(value.op, 3)
        lhs = _written_operand(value.lhs, level)
        rhs = _written_operand(value.rhs, level + (value.op in ("sub", "div")))
        return f"{lhs} {symbol} {rhs}"
    return repr(value)


def _written_operand(value, level: int) -> str:
    text = _written_text(value)
    if isinstance(value, ParameterExpression) and _PRECEDENCE.get(value.op, 3) < level:
        return f"({text})"
    return text


def _written_repr(value) -> str:
    if isinstance(value, UndecidableComparison):
        return f"({_written_repr(value.lhs)} {value.op} {_written_repr(value.rhs)})"
    if isinstance(value, ParameterExpression):
        if value.op in ("neg", "cast"):
            return f"ParameterExpression({value.op}, {_written_repr(value.lhs)}, {value.dtype})"
        return (
            f"ParameterExpression({value.op}, {_written_repr(value.lhs)}, "
            f"{_written_repr(value.rhs)})"
        )
    return repr(value)


def _value(node, a, known):
    """What *node* computes from the arguments *a*, each operation once."""
    if isinstance(node, Parameter):
        return a[node.index]
    if not isinstance(node, ParameterExpression):
        return float(node)
    if id(node) not in known:
        lhs = _value(node.lhs, a, known)
        if node.op == "neg":
            known[id(node)] = -lhs
        elif node.op == "cast":
            known[id(node)] = lhs
        else:
            rhs = _value(node.rhs, a, known)
            if node.op == "div":
                known[id(node)] = lhs / rhs if rhs else math.nan
            else:
                known[id(node)] = {"add": lhs + rhs, "sub": lhs - rhs, "mul": lhs * rhs}[node.op]
    return known[id(node)]


_PARAMETERS = [Parameter(f"p{i}", i, "f64") for i in range(3)]


def _random_expression(rng: random.Random, operations: int, shares: bool):
    """Arithmetic of *operations* operations over three parameters and a few
    numbers; with *shares*, operations use earlier ones again."""
    pool = [*_PARAMETERS, 0.5, 2.0, -1.25, 7]
    made: list = []
    for _ in range(operations):
        op = rng.choice(["add", "sub", "mul", "div", "neg", "cast", "add", "mul"])

        def operand():
            if made and rng.random() < 0.7:
                return made[-1] if not shares else rng.choice(made[-6:])
            return rng.choice(pool)

        lhs = operand()
        if op == "neg":
            node = ParameterExpression("neg", lhs)
        elif op == "cast":
            node = ParameterExpression("cast", lhs, dtype="f64")
        else:
            rhs = rng.choice(pool) if not shares else operand()
            node = ParameterExpression(op, lhs, rhs)
        made.append(node)
    return made[-1]


def _computed(source: str, a: list[float]) -> float:
    try:
        return eval(f"lambda a: {source}", {"__builtins__": {}})(a)
    except ZeroDivisionError:
        return math.nan


def _written_is_arithmetic(value) -> bool:
    if isinstance(value, ParameterExpression):
        if value.op in ("add", "sub", "mul", "div"):
            return _written_is_arithmetic(value.lhs) and _written_is_arithmetic(value.rhs)
        if value.op == "neg" or (value.op == "cast" and value.dtype == "f64"):
            return _written_is_arithmetic(value.lhs)
        return False
    return isinstance(value, (Parameter, int, float))


def _written_answer(value) -> bool:
    if isinstance(value, (ParameterExpression, UndecidableComparison)):
        return _written_answer(value.lhs) or _written_answer(value.rhs)
    return isinstance(value, AnswerValue)


def _written_without_answers(value):
    if isinstance(value, AnswerValue):
        return 0.0
    if isinstance(value, ParameterExpression):
        return ParameterExpression(
            value.op,
            _written_without_answers(value.lhs),
            _written_without_answers(value.rhs),
            dtype=value.dtype,
        )
    return value


_ANSWER = AnswerValue(7, "f64", (), "f[0]")


def _random_tree(rng: random.Random, depth: int):
    """Arithmetic that shares nothing, with operations on both sides, and
    with the operands and operations the walks refuse or pass over."""
    if depth == 0 or rng.random() < 0.2:
        return rng.choice([*_PARAMETERS, 0.5, 2.0, -1.25, 7, True, _ANSWER, None])
    op = rng.choice(["add", "sub", "mul", "div", "neg", "cast", "add", "mul", "floordiv", "i64"])
    lhs = _random_tree(rng, depth - 1)
    if op == "neg":
        return ParameterExpression("neg", lhs)
    if op == "cast" or op == "i64":
        return ParameterExpression("cast", lhs, dtype="f64" if op == "cast" else "i64")
    return ParameterExpression(op, lhs, _random_tree(rng, depth - 1))


def test_an_expression_that_shares_nothing_reads_as_it_did():
    rng = random.Random(11)
    for _ in range(1500):
        root = _random_tree(rng, rng.randrange(1, 7))
        if not isinstance(root, ParameterExpression):
            continue
        assert arithmetic_source(root) == _written_source(root)
        assert arithmetic_text(root) == _written_text(root)
        assert repr(root) == _written_repr(root)
        compared = UndecidableComparison(root, "<", _random_tree(rng, 2))
        other = UndecidableComparison(_random_tree(rng, 3), "==", None)
        both = UndecidableComparison(compared, "and", other)
        assert repr(both) == _written_repr(both)
        assert parameter_names(root) == _written_names(root)
        assert parameter_names(both) == _written_names(both)
        assert is_arithmetic(root) == _written_is_arithmetic(root)
        assert contains_answer(root) == _written_answer(root)
        assert contains_answer(both) == _written_answer(both)
        assert repr(without_answers(root)) == _written_repr(_written_without_answers(root))


def test_a_message_that_joins_two_values_names_their_terms_once():
    p = _PARAMETERS[0]
    s, r = p * 2.0, p + 1.0
    a, b = s + s, r * r
    with pytest.raises(SymbolicValueError) as raised:
        bool(a < b)
    message = str(raised.value)
    assert message.count(" with ") == 1
    assert (
        "ParameterExpression(add, t1, t1) < ParameterExpression(mul, t2, t2) with t1 = " in message
    )
    # Values that share nothing read as they did.
    with pytest.raises(
        SymbolicValueError, match=r"cannot decide Parameter\('p0', idx=0, f64\) < 1.0 at"
    ):
        bool(p < 1.0)


def test_a_none_operand_is_written():
    p = _PARAMETERS[0]
    assert repr(p == None) == "(Parameter('p0', idx=0, f64) == None)"  # noqa: E711
    assert arithmetic_text(ParameterExpression("add", p, None)) == "p0 + None"


def test_a_chain_is_written_in_place_as_deep_as_it_was():
    def chain(n: int):
        value = _PARAMETERS[0]
        for _ in range(n):
            value = value + 1.0
        return value

    # 190 parentheses deep is written as it was; deeper, where Python stops
    # reading parentheses, as named terms. Both compute the same.
    assert arithmetic_source(chain(190)) == _written_source(chain(190))
    deeper = arithmetic_source(chain(191))
    assert deeper.startswith("(t0 := (a[0] + 1.0), t1 := (t0 + 1.0), ")
    assert _computed(deeper, [0.5]) == 191.5 == _computed(arithmetic_source(chain(190)), [0.5]) + 1
    # A cast adds no parentheses.
    cast = chain(100)
    for _ in range(150):
        cast = ParameterExpression("cast", cast, dtype="f64")
    assert ":=" not in arithmetic_source(cast)


def test_names_are_given_once_in_a_text_made_of_texts():
    from cudaq_pulse.kernel._conditions import _text

    answer = AnswerValue(7, "f64", (), "f[0]")
    x = answer * 2.0
    u = x + x
    y = answer - 1.0
    w = y * y
    assert _text(u < 3) == "(t1 + t1 < 3) with t1 = f[0] * 2.0"
    assert _text((u < 3) & (w > u)) == (
        "((t1 + t1 < 3) and (t2 * t2 > t1 + t1)) with t1 = f[0] * 2.0, t2 = f[0] - 1.0"
    )
    c = u < 3
    d = UndecidableComparison(c, "and", c)
    assert _text(UndecidableComparison(d, "or", d)) == (
        "(t3 or t3) with t1 = f[0] * 2.0, t2 = (t1 + t1 < 3), t3 = (t2 and t2)"
    )


def test_the_source_of_a_shared_term_computes_the_value_written_out():
    rng = random.Random(5)
    named = 0
    for _ in range(1500):
        root = _random_expression(rng, rng.randrange(1, 140), shares=True)
        source = arithmetic_source(root)
        assert source is not None
        named += ":=" in source
        a = [rng.uniform(-3.0, 3.0) for _ in _PARAMETERS]
        got, want = _computed(source, a), _value(root, a, {})
        assert got == want or (math.isnan(got) and math.isnan(want)), source
        # As long as its operations, not as its terms written out.
        operations = len(expression_nodes(root, ARITHMETIC))
        assert len(source) < 60 * operations + 60
        assert len(arithmetic_text(root)) < 60 * operations + 60
        assert len(repr(root)) < 120 * operations + 120
    assert named > 800


def test_a_shared_term_is_named_once():
    p, q = _PARAMETERS[0], _PARAMETERS[1]
    x = p * 2.0
    assert arithmetic_text((x + x) * (x + 1.0)) == "(t1 + t1) * (t1 + 1.0) with t1 = p0 * 2.0"
    assert arithmetic_source(x + x) == "(t0 := (a[0] * 2.0), t1 := (t0 + t0), t1)[-1]"
    assert repr((x + x) > x) == (
        "(ParameterExpression(add, t1, t1) > t1) with "
        "t1 = ParameterExpression(mul, Parameter('p0', idx=0, f64), 2.0)"
    )
    # A term used once is written where it is used, as before.
    assert arithmetic_text(x * (q + 1.0)) == "p0 * 2.0 * (p1 + 1.0)"
    assert arithmetic_source(x * (q + 1.0)) == "((a[0] * 2.0) * (a[1] + 1.0))"
    # Its parameters are read once; those of a tree once per use, as before.
    assert parameter_names((x + x) * q) == ("p0", "p1")
    assert parameter_names((p + p) * q) == ("p0", "p0", "p1")


def test_the_walks_visit_a_doubled_term_once():
    value = _PARAMETERS[0] * 2.0
    for _ in range(DOUBLINGS):
        value = value + value
    with _within(20):
        assert len(expression_nodes(value)) == DOUBLINGS + 1
        assert list(expression_leaves(value)) == [_PARAMETERS[0], 2.0]
        assert parameter_names(value) == ("p0",)
        assert is_arithmetic(value) and not contains_answer(value)
        assert _computed(arithmetic_source(value), [3.0]) == 6.0 * 2**DOUBLINGS
        assert len(repr(value > 1.0)) < 4000
        with pytest.raises(SymbolicValueError, match="float argument 'p0'") as raised:
            float(value)
        assert raised.value.names == ("p0",)
