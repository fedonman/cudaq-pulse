# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conditions known only when the program runs, as a kernel's Python
combines them: ``&``, ``|`` and ``^``, ``in``, ``any()`` and ``all()``, and
the bool a conditional expression, ``and`` or ``or`` chooses. The tracer
imports this module when a kernel first does one of these.
"""

from __future__ import annotations

import numbers
import operator
import threading
from typing import TYPE_CHECKING, Any, NoReturn

from ._bytecode_normalize import FOR_ITER, LOAD_CONST, POP_TOP
from ._emitter import _REJOIN_MESSAGE, _Choice, _choice_map, _NotMeasurement, _watch_choices
from .ir_builder import (
    COMPARISONS,
    AnswerValue,
    CompilationError,
    IRValue,
    ParameterExpression,
    SymbolicValueError,
    UndecidableComparison,
    arithmetic_text,
    contains_answer,
    expression_leaves,
    expression_nodes,
    is_symbolic,
    parameter_names,
    shared_text,
    written_together,
)

if TYPE_CHECKING:
    from ._bytecode_normalize import DecodedCode
    from ._emitter import PulseIREmitter


def negated(value: _NotMeasurement) -> Any:
    """``not`` of a condition on answers as the comparison it is; a
    measurement's stays itself."""
    inner = value.value
    if answer_bit(inner):
        # The IR value carries no regions to check its uses against, so () checks none.
        return UndecidableComparison(AnswerValue(inner.vid, "i1", ()), "==", False)
    return value


def answer_bit(value: Any) -> bool:
    """Whether *value* is the ``i1`` a condition on answers is tested as."""
    return value.__class__ is IRValue and value.vtype == "i1"


def constant_bool(value: Any) -> Any:
    """*value* if it is a bool (Python's or NumPy's) or the int 0 or 1, which
    ``&``, ``|`` and ``^`` treat as one; else None."""
    cls = value.__class__
    if cls is bool or (cls is int and (value == 0 or value == 1)):
        return value
    if cls.__name__ == "bool" and cls.__module__ == "numpy":
        return bool(value)
    return None


def measured(value: Any) -> bool:
    """Whether the bool *value* is computed from a measurement result."""
    cls = value.__class__
    if cls is IRValue:
        return bool(value.vtype == "measurement")
    if cls is _NotMeasurement:
        return measured(value.value)
    if cls is UndecidableComparison:
        # Kept on the comparison: one built from others asks only them, so a
        # condition of n operations is asked n times, not n * n.
        try:
            return value.measured  # type: ignore[no-any-return]
        except AttributeError:
            pass
        for node in expression_nodes(value, COMPARISONS, _unasked):
            node.measured = measured(node.lhs) or measured(node.rhs)
        return value.measured  # type: ignore[no-any-return]
    return False


def _unasked(value: Any) -> bool:
    return not hasattr(value, "measured")


def is_condition(value: Any) -> bool:
    """Whether *value* is a bool known only when the program runs that is
    computed from answers or float arguments: a comparison, an answer bit,
    or ``not`` of one."""
    cls = value.__class__
    if cls is UndecidableComparison:
        return not measured(value)
    if cls is AnswerValue:
        return bool(value.dtype == "i1")
    return cls is _NotMeasurement and answer_bit(value.value)


# Conditions on measurement results kept in a value. Each is a choice of
# bools (vtype "condition"): on a measurement, whether it is set, and on the
# i1 of a condition on answers, whether it holds. An if on one that a single
# measurement or answer does not decide tests its i1, a select of true and
# false per condition, which lowering turns into the ifs on the measurements.

_held = threading.local()


def note_builder(builder: Any) -> None:
    """Note *builder* as the one an if on a choice on this thread tests in."""
    _held.builder = builder


def truth_value(value: Any, scope: tuple[int, ...], builder: Any = None) -> Any:
    """*value*, a bool, a measurement result, ``not`` of one, a condition
    kept in a value or (given *builder*) a condition on answers, as the
    condition it is; None if it is none of these."""
    cls = value.__class__
    if cls is bool:
        return value
    if cls is _Choice:
        return value if value.vtype == "condition" else None
    held = True
    if cls is _NotMeasurement and not answer_bit(value.value):
        value, held, cls = value.value, False, value.value.__class__
    if cls is IRValue and value.vtype == "measurement":
        return _leaf(value, held, scope)
    if builder is not None and is_condition(value):
        from ._classical import condition

        if cls is _NotMeasurement:
            value = negated(value)
        return _leaf(condition(builder, value), True, scope)
    return None


def _leaf(cond: IRValue, held: bool, scope: tuple[int, ...]) -> _Choice:
    leaf = _Choice(cond, held, not held, "", None, scope)
    leaf.vtype = "condition"
    return leaf


def _truth_arm(value: Any) -> bool:
    return is_condition(value) or truth_value(value, ()) is not None


def truth_choice(cond: IRValue, a: Any, b: Any) -> bool:
    """Whether an if on *cond*, a measurement result or a condition, whose
    arms leave *a* and *b* chooses a condition kept in a value: each a bool,
    a measurement result, ``not`` of one or a condition, and the choice
    depending on a measurement result."""
    if cond.vtype != "measurement" and cond.vtype != "i1":
        return False
    if not (_truth_arm(a) and _truth_arm(b)) or (a.__class__ is bool and b.__class__ is bool):
        return False
    if cond.vtype == "measurement":
        return True
    return any(v.__class__ is not bool and not is_condition(v) for v in (a, b))


def chosen_condition(
    builder: Any, cond: IRValue, a: Any, b: Any, name: str, line: int | None, scope: tuple[int, ...]
) -> Any:
    """The condition an if on *cond* leaves where its arms left *a* and *b*."""
    then = truth_value(a, scope, builder)
    other = truth_value(b, scope, builder)
    if then is other or (then.__class__ is bool and other.__class__ is bool and then == other):
        return then
    _watch_choices(builder)
    chosen = _Choice(cond, then, other, name, line, scope)
    chosen.vtype = "condition"
    return chosen


def condition_bit(choice: _Choice) -> IRValue:
    """The i1 an if tests where *choice*, a condition kept in a value, is
    its condition: a select of true and false per measurement or answer."""
    builder = _held.builder

    def bit(value: Any) -> int:
        if value.__class__ is bool:
            return int(builder.bit_constant(value))
        return _bit_select(builder, value.cond.vid, bit(value.then), bit(value.other))

    return IRValue(bit(choice), "i1")


def _bit_select(builder: Any, cond: int, then: int, other: int) -> int:
    """The id of an i1 that is *then* where *cond* (a measurement, whether
    it is set, or an i1) holds and *other* where not: a select the packed
    builder records once in each region."""
    from .packed_ir_builder import _SELECT, _header

    key = ((cond, then, other), "bit")
    vid = builder._numeric_vids.get(key)
    if vid is not None:
        return int(vid)
    vid = builder._next_id
    builder._next_id += 1
    builder._put((_header(_SELECT, 4), vid, cond, then, other))
    builder._numeric_vids[key] = vid
    builder.selects = True
    return int(vid)


def not_a_number(choice: _Choice, operation: str) -> NoReturn:
    """Raise for *operation* of *choice*, a condition kept in a value."""
    symbol = _SYMBOLS.get(operation) or _OPERATION_SYMBOLS.get(operation)
    symbol = f"`{symbol}`" if symbol else operation
    raise CompilationError(
        f"cannot trace {symbol} of {choice!r}: a condition on measurement results is known "
        "only when the program runs, and only an if tests it (`if x:`); it combines with "
        "and, or, not, &, |, ^, == and != of conditions, and is not a number",
        code="runtime_value",
    )


_OPERATION_SYMBOLS = {"div": "/"}


def compared(other: Any) -> Any:
    """*other* as what a condition kept in a value compares with: ``not``
    of a measurement result is its truth; a measurement result itself stays
    its level."""
    if other.__class__ is _NotMeasurement and not answer_bit(other.value):
        held = truth_value(other, ())
        return other if held is None else held
    return other


def _truth_logic(symbol: str, lhs: Any, rhs: Any) -> Any:
    """``lhs & rhs``, ``lhs | rhs`` or ``lhs ^ rhs`` of conditions, one at
    least kept in a value; None for anything else, such as a measurement
    result itself, whose bits are its level's."""
    if lhs.__class__ is IRValue or rhs.__class__ is IRValue:
        return None
    a, b = truth_value(lhs, ()), truth_value(rhs, ())
    if a is None or b is None or not (a.__class__ is _Choice or b.__class__ is _Choice):
        return None
    return _choice_map(_LOGIC[symbol], a, b)


_LOGIC = {"&": operator.and_, "|": operator.or_, "^": operator.ne}


def measurement_error(lhs: Any, symbol: str, rhs: Any) -> CompilationError:
    """The error for ``lhs symbol rhs``, which a measurement result takes part
    in: a test of it with a condition on answers needs nested ifs, and one
    with anything else is a test of the result itself."""
    (left, right), shared = written_together((lhs, rhs), _text)
    text = f"{left} {symbol} {right}{shared}"
    if _answered(lhs) or _answered(rhs):
        return CompilationError(
            f"cannot trace ({text}): it tests a measurement result and a condition on "
            "answers together, which one branch cannot; test them in nested ifs, "
            "`if result:` and in its arm an if on the condition",
            code="runtime_value",
        )
    return CompilationError(
        f"cannot trace ({text}): a measurement result is known only when the program "
        "runs, so it is not compared or combined while the kernel traces",
        code="runtime_value",
        fix=_BRANCH_ON_IT,
    )


_BRANCH_ON_IT = "branch on it directly with `if result:` or `if not result:`"


def measurement_operand_error(name: str) -> CompilationError:
    """The error for *name* (an operator or function) given a measurement result."""
    return CompilationError(
        f"cannot trace {name} of a measurement result: it is known only when the program runs",
        code="runtime_value",
        fix=_BRANCH_ON_IT,
    )


def operator_error(
    error: TypeError, fn: Any, operands: tuple[Any, Any] | None
) -> CompilationError | None:
    """The error for the unary or binary operator *fn* that raised *error*
    on a pulse value (the binary one's *operands* given), naming how to test
    a measurement result; None for any other operand."""
    symbol = _SYMBOLS.get(getattr(fn, "__name__", ""), "an operator")
    if operands is not None:
        lhs, rhs = operands
        if measured(lhs) or measured(rhs):
            return measurement_error(lhs, symbol, rhs)
        return None
    if "'IRValue'" not in str(error):
        return None
    return CompilationError(
        f"cannot trace `{symbol}` of a pulse value",
        code="runtime_value",
        fix=f"to test a measurement result, {_BRANCH_ON_IT}",
    )


_SYMBOLS = {
    "invert": "~",
    "neg": "-",
    "pos": "+",
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "lshift": "<<",
    "rshift": ">>",
    "matmul": "@",
}


def _text(value: Any, names: list[str] | None = None) -> str:
    """*value* in an error, a measurement result as "measurement"; *names*
    as ``shared_text`` takes them."""
    cls = value.__class__
    if cls is IRValue:
        return "measurement" if value.vtype == "measurement" else repr(value)
    if cls is _NotMeasurement:
        return f"(not {_text(value.value, names)})"
    if cls is UndecidableComparison:
        return shared_text(value, _comparison_text, _text, COMPARISONS, names)
    if cls is AnswerValue or is_symbolic(value):
        return arithmetic_text(value, names)
    return repr(value)


def _comparison_text(node: Any, lhs: str, rhs: str, plain: Any) -> str:
    return f"({lhs} {node.op} {rhs})"


def _answered(value: Any) -> bool:
    """Whether *value* holds a condition on answers."""
    if value.__class__ is UndecidableComparison:
        return any(_answered(leaf) for leaf in expression_leaves(value, COMPARISONS))
    if value.__class__ is _NotMeasurement:
        return answer_bit(value.value)
    return contains_answer(value)


_LOGIC_WORDS = {"&": "and", "|": "or"}


def bit_logic(symbol: str, lhs: Any, rhs: Any) -> Any:
    """``lhs & rhs``, ``lhs | rhs`` or ``lhs ^ rhs``, as Python computes them
    on bools, where one is known only when the program runs: the value
    when the other operand decides it, else a condition traced as ``and``,
    ``or`` or ``!=``. Measurement results and numbers raise."""
    word = _LOGIC_WORDS.get(symbol)
    for a, b in ((lhs, rhs), (rhs, lhs)):
        constant = constant_bool(a)
        if constant is not None and is_condition(b):
            if word is None:
                return UndecidableComparison(b, "!=", constant)
            # True | b and False & b are the constant; True & b and False | b are b.
            return constant if constant == (word == "or") else b
    if not (is_condition(lhs) and is_condition(rhs)):
        kept = _truth_logic(symbol, lhs, rhs)
        if kept is not None:
            return kept
        if measured(lhs) or measured(rhs):
            raise measurement_error(lhs, symbol, rhs)
        (left, right), shared = written_together((lhs, rhs))
        raise CompilationError(
            f"cannot trace {left} {symbol} {right}{shared}: it is known only when the program "
            f"runs; `{symbol}` combines only conditions (comparisons and answer bits) and True or "
            "False; bitwise operations on numbers known only when the program runs are not "
            "supported",
            code="runtime_value",
        )
    if lhs.__class__ is _NotMeasurement:
        lhs = negated(lhs)
    if rhs.__class__ is _NotMeasurement:
        rhs = negated(rhs)
    return UndecidableComparison(lhs, word or "!=", rhs)


def bit_select(cond: AnswerValue, then: Any, other: Any) -> Any:
    """*then* where the answer bit *cond* is set, else *other*: bools, at
    least one known only when the program runs."""
    if then is True:
        return bit_logic("|", cond, other)
    if other is False:
        return bit_logic("&", cond, then)
    unset = UndecidableComparison(cond, "==", False)
    if then is False:
        return bit_logic("&", unset, other)
    if other is True:
        return bit_logic("|", unset, then)
    return bit_logic("|", bit_logic("&", cond, then), bit_logic("&", unset, other))


def runtime_value(value: Any) -> bool:
    """Whether *value* is known only when the program runs, so that a set or
    dict cannot look it up by its hash."""
    cls = value.__class__
    if cls is IRValue:
        return bool(value.vtype == "measurement")
    return cls is UndecidableComparison or cls is _NotMeasurement or is_symbolic(value)


# Past this many items, `answer in range(...)` of a float answer is too
# many comparisons; one of an integer answer is two whatever its length.
_MAX_MEMBERS = 64


def contains(left: Any, right: Any, error: CompilationError) -> Any:
    """``left in right`` for a tuple, list or range *right*, where comparing
    *left* with its items raised *error* because the outcome is known only
    when the program runs: an ``or`` of the comparisons, as ``in`` tests
    each item with ``is`` and ``==``."""
    cls = right.__class__
    if (
        isinstance(error, SymbolicValueError)
        or not (cls is tuple or cls is list or cls is range)
        or measured(left)
    ):
        raise error
    if cls is range and left.__class__ is AnswerValue and left.dtype == "i64":
        start, stop, step = right.start, right.stop, right.step
        if not right:
            return False
        if step > 0:
            found: Any = UndecidableComparison(left >= start, "and", left < stop)
        else:
            found = UndecidableComparison(left <= start, "and", left > stop)
        if step != 1 and step != -1:
            offset = ParameterExpression("sub", left, start, dtype="i64")
            remainder = ParameterExpression("mod", offset, abs(step), dtype="i64")
            on_step = UndecidableComparison(remainder, "==", 0)
            found = UndecidableComparison(found, "and", on_step)
        return found
    if len(right) > _MAX_MEMBERS:
        raise CompilationError(
            f"cannot trace {left!r} in a {cls.__name__} of {len(right)}: it is known only "
            f"when the program runs, and `in` would compare it with each of more than "
            f"{_MAX_MEMBERS} items; compare it with <, <= and == instead",
            code="runtime_value",
        ) from error
    found = False
    for item in right:
        if item is left:
            return True
        if (
            left.__class__ is AnswerValue
            and not isinstance(item, numbers.Real)
            and constant_bool(item) is None
        ):
            continue  # an answer is a number, equal to no other kind of value
        term = left == item
        if term.__class__ is bool:
            if term:
                return True
            continue
        found = term if found is False else UndecidableComparison(found, "or", term)
    return found


def any_or_all(either: bool, items: Any, error: CompilationError, generated: bool) -> Any:
    """``any(items)`` if *either*, else ``all(items)``, whose evaluation
    raised *error* because the truth of an item is known only when the
    program runs: for a list or tuple, an ``or`` (an ``and``) of the items'
    truth; *generated* items, a generator expression's, raise."""
    if items.__class__ is not list and items.__class__ is not tuple:
        if not generated:
            raise error
        name, word = ("any", "or") if either else ("all", "and")
        raise CompilationError(
            f"{name}() of a generator of conditions known only when the program runs is "
            f"not traced; give it a list, {name}([...]), or combine them with `{word}`",
            code="runtime_value",
        ) from error
    found: Any = not either
    for item in items:
        if item.__class__ is AnswerValue and item.dtype != "i1":
            item = UndecidableComparison(item, "!=", 0)
        elif not is_condition(item):
            kept = truth_value(item, ())
            if kept.__class__ is _Choice and (
                found.__class__ is bool or found.__class__ is _Choice
            ):
                found = _choice_map(operator.or_ if either else operator.and_, found, kept)
                continue
            if measured(item) or item.__class__ is IRValue:
                raise error
            if bool(item) == either:
                return either
            continue
        found = bit_logic("|" if either else "&", found, item)
    return found


def bits_chosen(cond: IRValue, a: Any, b: Any) -> bool:
    """Whether an if on *cond* whose arms leave *a* and *b* chooses a bool
    known only when the program runs: *cond* is a condition on answers, and
    *a* and *b* are such conditions or bools, at least one a condition."""
    if cond.vtype != "i1":
        return False
    a_bit = is_condition(a)
    b_bit = is_condition(b)
    return (a_bit or b_bit) and (a_bit or a.__class__ is bool) and (b_bit or b.__class__ is bool)


def add_built(stack: list[Any], containers: list[tuple[Any, Any]]) -> None:
    """Add to *containers*, with a copy, each list, set or dict on *stack*
    not in them yet: what a comprehension around a branch builds."""
    known = {id(obj) for obj, _ in containers}
    for value in stack:
        if value.__class__ in (list, set, dict) and id(value) not in known:
            containers.append((value, value.copy()))


def hashed_operand(emitter: PulseIREmitter, left: Any, right: Any) -> Any:
    """What ``left in right`` tests for a set or dict *right*: its items
    compared one by one when *left* is known only at run time, as such a
    value has no hash to look up."""
    if right.__class__ is dict and emitter._dict_items is not None:
        emitter.b.dict_whole.add(id(right))
    if runtime_value(left):
        return tuple(right)
    return right


def stack_choices(
    emitter: PulseIREmitter,
    cond: IRValue,
    then_stack: list[Any],
    saved_stack: list[Any],
    jump_if_true: bool,
    prog: DecodedCode,
    index: int,
    first: int,
    chosen: list[tuple[str | int, Any, Any]],
) -> None:
    """Add to *chosen* the stack slots the arms of the if on *cond* at
    ``prog.instrs[index]`` left different values in, as (slot, then
    value, else value), each checked to be a pair the if can choose.

    An arm that left a slot as it was before the if is the short cut of
    an ``and`` (a jump if false) or an ``or``: the slot holds the value
    the jump tested, which is false, or true, in that arm."""
    for slot, (a, b) in enumerate(zip(then_stack, emitter.stack, strict=False)):
        if a is not b:
            if slot < len(saved_stack):
                kept = saved_stack[slot]
                if a is kept:
                    a = jump_if_true
                elif b is kept:
                    b = jump_if_true
            if not bits_chosen(cond, a, b):
                emitter._check_choosable("", a, b, prog, index, first, cond)
            chosen.append((slot, a, b))


def chosen_value(
    emitter: PulseIREmitter,
    cond: IRValue,
    a: Any,
    b: Any,
    name: str,
    line: int | None,
    scope: tuple[int, ...],
) -> Any:
    """What an if on *cond* leaves where its arms left *a* and *b*: for
    bools on answers, the condition ``cond and a or not cond and b``;
    else the choice between them."""
    if bits_chosen(cond, a, b):
        return bit_select(AnswerValue(cond.vid, "i1", scope), a, b)
    if truth_choice(cond, a, b):
        return chosen_condition(emitter.b, cond, a, b, name, line, scope)
    _watch_choices(emitter.b)
    return _Choice(cond, a, b, name, line, scope)


def filtered(emitter: PulseIREmitter, containers: list[tuple[Any, Any]]) -> bool:
    """Whether the arm just traced added to what a comprehension or
    generator expression builds."""
    yielded = emitter.locals.get(".yielded")
    # A list put back from a slot log (no copy) keeps its length.
    return any(
        len(obj) != len(copy)
        for obj, copy in containers
        if copy is not None and (obj is yielded or any(obj is v for v in emitter.stack))
    )


def identity(left: Any, right: Any) -> Any:
    """``left is right``, which ``_identity_unknown`` holds depends on the
    program's run: ``left == right`` where both are bools, a condition on
    answers and True or False or another such condition, as Python has one
    True and one False; anything else raises."""
    if not (measured(left) or measured(right)) and _bool(left) and _bool(right):
        if left.__class__ is _NotMeasurement:
            left = negated(left)
        return UndecidableComparison(left, "==", right)
    raise identity_error(left, right)


def _bool(value: Any) -> bool:
    return value is True or value is False or is_condition(value)


def identity_error(left: Any, right: Any) -> CompilationError:
    """The error for ``left is right``, which ``_identity_unknown`` holds
    depends on the program's run; a kernel specializes on the float
    arguments of one that depends on nothing else."""
    if measured(left) or measured(right):
        return measurement_error(left, "is", right)
    (first, second), shared = written_together((left, right))
    message = (
        f"cannot decide {first} is {second}{shared} at trace time: it is known only when the "
        "program runs; compare values with == or !="
    )

    def run_time(value: Any) -> bool:
        if value.__class__ is UndecidableComparison:
            return any(run_time(leaf) for leaf in expression_leaves(value, COMPARISONS))
        return isinstance(value, (IRValue, _NotMeasurement)) or contains_answer(value)

    names = parameter_names(left) + parameter_names(right)
    if not names or run_time(left) or run_time(right):
        return CompilationError(message, code="runtime_value")
    return SymbolicValueError(message, tuple(dict.fromkeys(names)), "the identity test is")


def rejoin_error(view: DecodedCode, branch: int) -> CompilationError:
    """The error for the if at ``view.instrs[branch]``, one arm of which
    leaves where the other continues. Python 3.14 compiles ``any()`` and
    ``all()`` of a generator expression to a loop whose if leaves it, on
    the first true (false) item, to push True (False)."""
    instrs = view.instrs
    target = instrs[branch][1]
    if (
        branch > 0
        and instrs[branch - 1][0] is FOR_ITER
        and target.__class__ is int
        and target + 1 < len(instrs)
        and instrs[target][0] is POP_TOP
        and instrs[target + 1][0] is LOAD_CONST
        and instrs[target + 1][1].__class__ is bool
    ):
        name, word = ("any", "or") if instrs[target + 1][1] else ("all", "and")
        return CompilationError(
            f"{name}() of a generator of conditions known only when the program runs is "
            f"not traced; give it a list, {name}([...]), or combine them with `{word}`",
            code="runtime_value",
        )
    return CompilationError(_REJOIN_MESSAGE, code="runtime_value")
