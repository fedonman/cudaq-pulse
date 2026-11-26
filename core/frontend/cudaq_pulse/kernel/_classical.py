# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracing calls to classical functions inside a kernel.

``fn.submit(*args)`` in a kernel records a ``classical.submit`` and gives a
``Future``; its ``.result()`` records a ``classical.wait`` and gives the
answer as ``AnswerValue``s, which later pulse ops take like float kernel
parameters and which branches test. The tracer calls these instead of the
methods themselves, and imports this module when a kernel first makes a
classical call or branches on an answer.
"""

from __future__ import annotations

import numbers
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..classical import (
    _F64,
    _KIND_BITS,
    _KIND_FLOATS,
    _KIND_IQ,
    _LIST_NAME,
    ClassicalFunction,
    ValueType,
    _enter_slow,
    _spelled,
)
from ._emitter import Future
from .ir_builder import (
    AnswerValue,
    CompilationError,
    IRValue,
    UndecidableComparison,
    is_symbolic,
    written_together,
)

if TYPE_CHECKING:
    from .packed_ir_builder import PackedIRBuilder

# The values an f64 argument takes as they are: _numbers would pass them
# straight to _get_numeric_vid.
_DIRECT_F64 = frozenset({float, int, AnswerValue})


def _builder_supports(builder: Any) -> None:
    if not hasattr(builder, "classical_submit"):
        raise CompilationError("classical calls need the packed IR builder", code="other")


def _wrong_count(
    function: ClassicalFunction[..., Any], spec: ValueType, position: int, got: int
) -> CompilationError:
    """A call passing a list of another length than its parameter's."""
    parameter = f"argument {position}"
    code = getattr(function.__wrapped__, "__code__", None)
    if code is not None and position < code.co_argcount:
        parameter = repr(code.co_varnames[position])
    spelled, unit = _spelled(spec)
    return CompilationError(
        f"{function.name}.submit(): {parameter} takes {spec.count} {unit}"
        f"{'s' if spec.count != 1 else ''} ({spelled}), got {got}",
        code="wrong_argument",
    )


# What a list of each kind holds, for an error.
_HOLDS = {
    _KIND_BITS: "a list of outcomes",
    _KIND_IQ: "a list of IQ points",
    _KIND_FLOATS: "a list of numbers",
}


def _is_readout(value: Any) -> bool:
    return isinstance(value, IRValue) and value.vtype == "measurement"


def _local_name(value: Any) -> str | None:
    """The name the traced kernel binds *value* to, if it has one: a local
    of the emitter running its code (``_emitter``'s ``locals``)."""
    frame = sys._getframe(1)
    while frame is not None:
        found = getattr(frame.f_locals.get("self"), "locals", None)
        if found.__class__ is dict:
            for name, bound in found.items():
                if bound is value and name.isidentifier():
                    return str(name)
            return None
        frame = frame.f_back  # type: ignore[assignment]
    return None


def _one_readout(
    function: ClassicalFunction[..., Any], position: int, kind: str, holds: str, value: Any
) -> CompilationError:
    """A call passing one readout where its parameter takes a list."""
    code = getattr(function.__wrapped__, "__code__", None)
    names = list(code.co_varnames[: code.co_argcount]) if code is not None else []
    parameter = names[position] if position < len(names) else f"argument {position}"
    shown = _local_name(value)
    listed = f"[{shown or 'm'}]"
    arguments = [listed if k == position else n for k, n in enumerate(names)] or [listed]
    called = function.base_name
    return CompilationError(
        f"{called}.submit(): {parameter} is {kind} ({holds}) but {shown or 'the argument'} is"
        f" one readout. Write {called}.submit({', '.join(arguments)}).",
        code="wrong_argument",
    )


def _measurements(
    value: Any, spec: ValueType, function: ClassicalFunction[..., Any], position: int
) -> list[int]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        if not spec.is_scalar and _is_readout(value):
            holds = (
                "a list of IQ points" if spec.descriptor.startswith("iq") else "a list of outcomes"
            )
            raise _one_readout(function, position, _spelled(spec)[0], holds, value)
        items = [value]
    if len(items) != spec.count:
        raise _wrong_count(function, spec, position, len(items))
    vids = []
    for item in items:
        if not (isinstance(item, IRValue) and item.vtype == "measurement"):
            raise CompilationError(
                f"{function.name}: argument {position} ({spec.descriptor}) takes measurement "
                f"results from readout(), got {item!r}",
                code="wrong_argument",
            )
        vids.append(item.vid)
    return vids


def _numbers(
    builder: Any, value: Any, spec: ValueType, function: ClassicalFunction[..., Any], position: int
) -> list[int]:
    name = function.name
    if spec.is_scalar:
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif _is_readout(value):
        raise _one_readout(function, position, _spelled(spec)[0], "a list of numbers", value)
    else:
        raise CompilationError(
            f"{name}: argument {position} is {spec.descriptor}",
            code="wrong_argument",
            fix=f"pass a list of {spec.count} numbers",
        )
    if len(items) != spec.count:
        raise _wrong_count(function, spec, position, len(items))
    dtype = "f64" if spec.source == "float" else "i64"
    vids = []
    builder._answer_integers = True
    try:
        for item in items:
            if isinstance(item, bool) or not (isinstance(item, (int, float)) or is_symbolic(item)):
                raise CompilationError(
                    f"{name}: argument {position} ({spec.descriptor}) takes numbers, got {item!r}",
                    code="wrong_argument",
                )
            if dtype == "i64" and isinstance(item, float) and not item.is_integer():
                raise CompilationError(
                    f"{name}: argument {position} ({spec.descriptor}) takes integers, got {item!r}",
                    code="wrong_argument",
                )
            vids.append(
                builder._get_numeric_vid(
                    int(item) if dtype == "i64" and isinstance(item, float) else item, dtype
                )
            )
    finally:
        builder._answer_integers = False
    return vids


def submit(
    builder: Any,
    function: ClassicalFunction[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any] | None,
) -> Future:
    """Record ``function.submit(*args)``."""
    _builder_supports(builder)
    if kwargs:
        raise CompilationError(
            f"{function.name}.submit() takes positional arguments only",
            code="wrong_argument",
        )
    try:
        arity = len(function.args)
    except TypeError as error:  # a mistake in its declaration
        raise CompilationError(str(error), code="wrong_argument") from None
    if len(args) != arity:
        # A template has args == (): its calls give the list lengths.
        plan = function._plan
        if plan is None and getattr(function.implementation, "_route", None) is not None:
            # A CUDA-Q QEC function: its arguments pick the version sent.
            version, wired = function.implementation._route(args)  # type: ignore[union-attr]
            return submit(builder, version, wired, None)
        if plan is None or len(args) != len(plan):
            arity = arity if plan is None else len(plan)
            raise CompilationError(
                f"{function.name}.submit() takes {arity} arguments, got {len(args)}",
                code="wrong_argument",
            )
        return _submit_template(builder, function, args, plan)
    operands: list[int] = []
    for position, (spec, value) in enumerate(zip(function.args, args, strict=False)):
        if spec.source == "measurement":
            operands.extend(_measurements(value, spec, function, position))
        elif spec is _F64 and value.__class__ in _DIRECT_F64:
            # What _numbers gives a plain number or an answer, without its checks.
            operands.append(builder._get_numeric_vid(value, "f64"))
        else:
            operands.extend(_numbers(builder, value, spec, function, position))
    index = builder.classical_function(function)
    vid = builder.classical_submit(index, operands)
    return Future(function, index, vid, builder.flow)


def _submit_template(
    builder: Any,
    template: ClassicalFunction[..., Any],
    args: Sequence[Any],
    plan: tuple[Any, ...],
) -> Future:
    """Record a call of a template, *args* one per entry of its *plan*: one
    pass over the arguments makes the operands and the version's key (per
    unsized list its length, negative for a ``list[int]`` of numbers), then
    one lookup finds the version."""
    name = template.name
    operands: list[int] = []
    key: Any = 0 if template._n == 1 else []
    for position, (kind, value) in enumerate(zip(plan, args, strict=False)):
        if kind.__class__ is not int:  # a wire type
            if kind.source == "measurement":
                operands.extend(_measurements(value, kind, template, position))
            elif kind is _F64 and value.__class__ in _DIRECT_F64:
                # What _numbers gives a plain number or an answer, without its checks.
                operands.append(builder._get_numeric_vid(value, "f64"))
            else:
                operands.extend(_numbers(builder, value, kind, template, position))
        else:
            if value.__class__ is not list and value.__class__ is not tuple:
                _not_a_list(template, position, value)
            if not value:
                raise CompilationError(
                    f"{name}.submit(): {_parameter(template, position)!r} is an empty list",
                    code="wrong_argument",
                )
            if kind >= _KIND_FLOATS and (
                kind == _KIND_FLOATS or not _measured(value, template, position)
            ):
                operands.extend(_number_list(builder, value, name, position, kind == _KIND_FLOATS))
                n = len(value) if kind == _KIND_FLOATS else -len(value)
            else:
                for item in value:
                    if not (isinstance(item, IRValue) and item.vtype == "measurement"):
                        raise CompilationError(
                            f"{name}.submit(): {_parameter(template, position)!r}"
                            f" ({_LIST_NAME[kind]}) takes measurement results from readout(),"
                            f" got {item!r}",
                            code="wrong_argument",
                        )
                    operands.append(item.vid)
                n = len(value)
            if key.__class__ is int:
                key = n
            else:
                key.append(n)
    if key.__class__ is list:
        key = tuple(key)
    version = template._fast.get(key)
    if version is None:
        version = _enter_slow(builder, template, key)
    index = builder.classical_function(version)
    vid = builder.classical_submit(index, operands)
    return Future(version, index, vid, builder.flow)


def _parameter(template: ClassicalFunction[..., Any], position: int) -> str:
    shape = template._shape
    assert shape is not None
    return shape.names[position]


def _measured(value: Sequence[Any], template: ClassicalFunction[..., Any], position: int) -> bool:
    """Whether a ``list[int]`` holds measurements (levels) or numbers."""
    measured = [isinstance(v, IRValue) and v.vtype == "measurement" for v in value]
    if any(measured) and not all(measured):
        raise CompilationError(
            f"{template.name}.submit(): {_parameter(template, position)!r} is list[int]: all"
            " measurements (sent as levels) or all numbers, not both",
            code="wrong_argument",
        )
    return measured[0]


def _number_list(
    builder: Any, items: Sequence[Any], name: str, position: int, floats: bool
) -> list[int]:
    dtype = "f64" if floats else "i64"
    vids = []
    builder._answer_integers = True
    try:
        for item in items:
            if isinstance(item, bool) or not (isinstance(item, (int, float)) or is_symbolic(item)):
                raise CompilationError(
                    f"{name}: argument {position} takes numbers, got {item!r}",
                    code="wrong_argument",
                )
            if dtype == "i64" and isinstance(item, float):
                if not item.is_integer():
                    raise CompilationError(
                        f"{name}: argument {position} takes integers, got {item!r}",
                        code="wrong_argument",
                    )
                item = int(item)
            vids.append(builder._get_numeric_vid(item, dtype))
    finally:
        builder._answer_integers = False
    return vids


def _not_a_list(template: ClassicalFunction[..., Any], position: int, value: Any) -> None:
    parameter = _parameter(template, position)
    shape = template._shape
    assert shape is not None
    plan = shape.plan[position]
    kind = _LIST_NAME[plan]
    if _is_readout(value):
        holds = _HOLDS.get(plan, "a list of outcomes or numbers")
        raise _one_readout(template, position, kind, holds, value)
    raise CompilationError(
        f"{template.name}.submit(): {parameter!r} is {kind}; pass a list, got {value!r}",
        code="wrong_argument",
    )


def _group(function: ClassicalFunction[..., Any], answers: list[AnswerValue]) -> Any:
    """Answers as the kernel sees them: per result a value or a list."""
    grouped: list[Any] = []
    cursor = 0
    for spec in function.results:
        part = answers[cursor : cursor + spec.answer_count]
        cursor += spec.answer_count
        grouped.append(part[0] if spec.is_scalar else list(part))
    return grouped[0] if len(grouped) == 1 else tuple(grouped)


def take(builder: Any, future: Future, args: Sequence[Any], kwargs: dict[str, Any] | None) -> Any:
    """Record ``future.result(timeout=...)``."""
    timeout = None
    if args:
        if len(args) > 1:
            raise CompilationError(
                "result() takes at most one argument, timeout", code="wrong_argument"
            )
        timeout = args[0]
    for key, value in (kwargs or {}).items():
        if key != "timeout":
            raise CompilationError(
                f"result() got an unexpected keyword {key!r}", code="wrong_argument"
            )
        timeout = value
    function = future.function
    timeout_ns = None
    if timeout is not None:
        from ._durations import timeout_ns as _timeout_ns

        timeout_ns = _timeout_ns(builder, timeout)
        if function.latency_mode == "blocking":
            raise CompilationError(
                f"{function.name} blocks until it answers; it has no timeout",
                code="wrong_argument",
            )
    if future.answers is not None:
        if timeout_ns != future.timeout_ns:
            raise CompilationError(
                "an answer is taken once; call result() with the same timeout",
                code="wrong_argument",
            )
        return future.answers
    if not function.results:
        raise CompilationError(
            f"{function.name} returns nothing; there is no answer to take",
            code="wrong_argument",
        )
    builder.check_wait(future.flow, function.name)
    timeout = timeout_ns is not None or function.latency_mode == "timeout"
    dtypes = (["i1"] if timeout else []) + [
        spec.answer for spec in function.results for _ in range(spec.answer_count)
    ]
    answers = builder.classical_wait(future.index, future.vid, dtypes, timeout_ns)
    for k, answer in enumerate(answers):
        answer.label = f"{function.name}[{k}]"
    if timeout:
        ok, rest = answers[0], answers[1:]
        value = _group(function, rest)
        result: Any = (ok, *value) if isinstance(value, tuple) else (ok, value)
    else:
        result = _group(function, answers)
    future.answers = result
    future.timeout_ns = timeout_ns
    return result


def condition(builder: Any, value: Any) -> IRValue:
    """The ``i1`` a branch on *value*, computed from an answer, tests."""
    if isinstance(value, AnswerValue):
        builder.check_scope(value.scope, "an answer of a classical call")
        if value.dtype == "i1":
            return IRValue(value.vid, "i1")
        return IRValue(builder.classical_compare("!=", value, 0), "i1")
    if isinstance(value, UndecidableComparison):
        if value.lhs.__class__ is AnswerValue and value.rhs.__class__ in _NUMBERS:
            try:
                return IRValue(builder.classical_compare(value.op, value.lhs, value.rhs), "i1")
            except CompilationError:
                # classical_compare rejects an answer bit before it records anything.
                if value.lhs.dtype != "i1":
                    raise
        from ._conditions import is_condition, measured, measurement_error

        lhs, rhs, op = value.lhs, value.rhs, value.op
        if measured(value):
            raise measurement_error(lhs, op, rhs)
        if op == "and" or op == "or":
            # Its left condition, then its right, then itself, as a call for
            # each would compute them, from a stack kept here: a chain of
            # them may be thousands long, and a call for each would be one
            # more call per condition than a comparison of conditions had.
            # One that several parts of the condition being traced use
            # (c = a & b, then c | (c & d)) is computed where it is first met
            # and read after that.
            made = getattr(builder, "conditions_made", None)
            outermost = made is None
            if made is None:
                builder.conditions_made = made = {}
            try:
                found = made.get(id(value))
                if found is not None:
                    return found  # type: ignore[no-any-return]
                # The ands and ors on the way down, how far each is (0: nothing, 1:
                # its left is computed, 2: its right too), and its left's id.
                stack = [value]
                steps = [0]
                lefts = [0]
                while stack:
                    node = stack[-1]
                    step = steps[-1]
                    operand = node.rhs if step else node.lhs
                    if step < 2:
                        if id(operand) in made:
                            vid = made[id(operand)].vid
                        elif _is_logic(operand):
                            if measured(operand):
                                raise measurement_error(operand.lhs, operand.op, operand.rhs)
                            stack.append(operand)
                            steps.append(0)
                            lefts.append(0)
                            continue
                        else:
                            vid = condition(builder, operand).vid
                        steps[-1] = step + 1
                        if step == 0:
                            lefts[-1] = vid
                            continue
                        made[id(node)] = IRValue(
                            builder.compare_bits(node.op, lefts[-1], vid), "i1"
                        )
                    stack.pop()
                    steps.pop()
                    lefts.pop()
                return made[id(value)]  # type: ignore[no-any-return]
            finally:
                if outermost:
                    builder.conditions_made = None
        if is_condition(lhs) or is_condition(rhs):
            # Once within the condition being traced, as an and or an or.
            made = getattr(builder, "conditions_made", None)
            if made is not None:
                found = made.get(id(value))
                if found is None:
                    made[id(value)] = found = IRValue(_compare_bits(builder, op, lhs, rhs), "i1")
                return found
            builder.conditions_made = {}
            try:
                return IRValue(_compare_bits(builder, op, lhs, rhs), "i1")
            finally:
                builder.conditions_made = None
        if lhs is None or rhs is None:
            # An answer is a number, never None.
            return IRValue(builder.bit_constant(op == "!="), "i1")
        if isinstance(lhs, IRValue) or isinstance(rhs, IRValue):
            raise CompilationError(
                f"cannot compare {value!r}: a pulse value is not a number",
                code="wrong_argument",
            )
        return IRValue(builder.classical_compare(op, lhs, rhs), "i1")
    from ._emitter import _NotMeasurement

    if value.__class__ is _NotMeasurement and value.value.vtype == "i1":
        bit = value.value.vid
        return IRValue(builder.compare_bits("==", bit, builder.bit_constant(False)), "i1")
    raise CompilationError(f"cannot branch on {value!r}", code="runtime_value")


def _is_logic(value: Any) -> bool:
    """Whether ``condition`` computes *value* as the and or the or of two
    conditions."""
    return (
        value.__class__ is UndecidableComparison
        and (value.op == "and" or value.op == "or")
        and not (value.lhs.__class__ is AnswerValue and value.rhs.__class__ in _NUMBERS)
    )


_NUMBERS = frozenset({int, float, bool})


def _compare_bits(builder: PackedIRBuilder, op: str, lhs: Any, rhs: Any) -> int:
    """The ``i1`` of *lhs* ``op`` *rhs*, one of them a bool known only when
    the program runs and the other such a bool or a number, as Python
    compares bools."""
    from ._conditions import constant_bool, is_condition

    if op not in ("==", "!="):
        (left, right), shared = written_together((lhs, rhs))
        raise CompilationError(
            f"cannot order the conditions in ({left} {op} {right}){shared}, which are known only "
            "when the program runs; compare them with == or !=, or combine them with "
            "and, or, not",
            code="runtime_value",
        )
    for constant, bit in ((lhs, rhs), (rhs, lhs)):
        if is_condition(constant):
            continue
        if constant is None:
            return builder.bit_constant(op == "!=")
        if not isinstance(constant, numbers.Real):
            given, constant = constant, constant_bool(constant)
            if constant is None:
                (first, second), shared = written_together((bit, given))
                raise CompilationError(
                    f"cannot compare the condition {first} with {second}{shared}; a condition "
                    "known only when the program runs compares with another such condition, "
                    "True or False",
                    code="runtime_value",
                )
        if constant != 0 and constant != 1:
            return builder.bit_constant(op == "!=")
        vid = condition(builder, bit).vid
        if bool(constant) == (op == "=="):
            return vid
        return builder.compare_bits("==", vid, builder.bit_constant(False))
    return builder.compare_bits(op, condition(builder, lhs).vid, condition(builder, rhs).vid)
