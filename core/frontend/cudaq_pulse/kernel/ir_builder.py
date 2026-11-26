# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared IR types and builder for the pulse kernel compiler.

This module contains the core data structures used by the bytecode
compiler: Op, IRValue, Parameter, PythonIRBuilder, OP_TABLE, LINEAR_TYPES,
and CompilationError.
"""

from __future__ import annotations

import numbers
from collections import namedtuple
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

from .._errors import CompilationError as CompilationError
from .._errors import InternalError

Op = namedtuple("Op", ["kind", "operands", "results", "attrs"])


class SymbolicValueError(CompilationError):
    """Python code needed the value of a runtime ``float`` argument.

    ``names`` lists the arguments involved, and ``operation`` what needed
    the value. A kernel call specializes on the value of such an argument
    instead of keeping it symbolic.
    """

    def __init__(self, message: str, names: tuple[str, ...] = (), operation: str = ""):
        super().__init__(message, code="symbolic_parameter")
        self.names = names
        self.operation = operation


def parameter_names(value: Any) -> tuple[str, ...]:
    """Names of the kernel parameters a symbolic value depends on, as it
    names them from left to right; a term that several others use is read
    once."""
    if isinstance(value, Parameter):
        return (value.name,)
    if isinstance(value, (ParameterExpression, UndecidableComparison)):
        return tuple(leaf.name for leaf in expression_leaves(value) if isinstance(leaf, Parameter))
    return ()


def expression_leaves(value: Any, kinds: tuple[Any, ...] | None = None) -> Iterator[Any]:
    """What the expression *value* is computed from, from left to right:
    the operands of its operations that are not such operations themselves.
    The operations are arithmetic and comparisons, or those of *kinds*
    (``ARITHMETIC``, ``COMPARISONS``); anything else is an operand. An
    operation that several others use is entered once, so an expression that
    doubles a term forty times is forty steps, not 2**40; and nothing here
    calls itself, so an expression may be thousands of operations deep."""
    if kinds is None:
        kinds = _EXPRESSIONS
    if value.__class__ not in kinds:
        yield value
        return
    seen = {id(value)}
    stack = [value.rhs, value.lhs]
    while stack:
        item = stack.pop()
        if item.__class__ in kinds:
            if id(item) not in seen:
                seen.add(id(item))
                stack.append(item.rhs)
                stack.append(item.lhs)
        else:
            yield item


def expression_nodes(
    value: Any,
    kinds: tuple[Any, ...] | None = None,
    entering: Callable[[Any], bool] | None = None,
) -> list[Any]:
    """The operations of the expression *value* (those of *kinds*, as
    ``expression_leaves`` takes them), each once however many others use
    it, and each after its operands, left before right. With *entering*,
    only the operations it holds for, and nothing below one it does not
    hold for. No recursion."""
    if kinds is None:
        kinds = _EXPRESSIONS
    if value.__class__ not in kinds or (entering is not None and not entering(value)):
        return []
    nodes: list[Any] = []
    seen = {id(value)}
    # The operations on the way down, and how many operands of each are entered.
    stack = [value]
    entered = [0]
    while stack:
        node = stack[-1]
        step = entered[-1]
        if step == 2:
            nodes.append(node)
            stack.pop()
            entered.pop()
            continue
        entered[-1] = step + 1
        operand = node.rhs if step else node.lhs
        if (
            operand.__class__ in kinds
            and id(operand) not in seen
            and (entering is None or entering(operand))
        ):
            seen.add(id(operand))
            stack.append(operand)
            entered.append(0)
    return nodes


def shared_text(
    value: Any,
    written: Callable[[Any, str, str, Callable[[Any], bool]], str],
    leaf: Callable[[Any, list[str] | None], str],
    kinds: tuple[Any, ...] | None = None,
    names: list[str] | None = None,
) -> str:
    """The expression *value* as text: ``written(node, lhs, rhs, plain)`` for
    each operation from the texts of its operands, where ``plain(operand)``
    says the operand is an operation written out in place;
    ``leaf(operand, names)`` for everything else. An operation that several
    others use is written once, under a name (``t1``, ``t2``, ...), and the
    names follow the text: ``t2 + t2 with t1 = tau * 2.0, t2 = t1 + t1``. So
    the text is as long as the expression has operations, not as long as it
    is written out. *names* is the list of names of a text this one is part
    of: its names are added there and follow that text, once."""
    nodes = expression_nodes(value, kinds)
    if not nodes:
        return leaf(value, names)
    outermost = names is None
    if names is None:
        names = []
    uses: dict[int, int] = {}
    if len(nodes) > 1:
        for node in nodes:
            uses[id(node.lhs)] = uses.get(id(node.lhs), 0) + 1
            uses[id(node.rhs)] = uses.get(id(node.rhs), 0) + 1
    texts: dict[int, str] = {}
    in_place: set[int] = set()

    def plain(operand: Any) -> bool:
        return id(operand) in in_place

    def of(operand: Any) -> str:
        text = texts.get(id(operand))
        if text is None:
            text = leaf(operand, names)
            if operand.__class__ in _EXPRESSIONS:
                # An operation of another kind, which the leaf wrote: once.
                texts[id(operand)] = text
        return text

    for node in nodes:
        lhs = of(node.lhs)
        # Negation and casts have one operand; None is an operand elsewhere.
        if node.rhs is None and node.__class__ is ParameterExpression and node.op in _UNARY:
            rhs = ""
        else:
            rhs = of(node.rhs)
        text = written(node, lhs, rhs, plain)
        # A cast is written as its operand: one that others share is named
        # unless its operand already is a name or is not an operation.
        if uses.get(id(node), 0) > 1 and (text != lhs or id(node.lhs) in in_place):
            name = f"t{len(names) + 1}"
            names.append(f"{name} = {text}")
            text = name
        else:
            in_place.add(id(node))
        texts[id(node)] = text
    text = texts[id(value)]
    return f"{text} with {', '.join(names)}" if outermost and names else text


_UNARY = ("neg", "cast")


def written_together(
    values: tuple[Any, ...], write: Callable[[Any, list[str] | None], str] | None = None
) -> tuple[list[str], str]:
    """The texts of *values* that one message joins (their reprs, or
    ``write(value, names)``), and what follows the joined text: the names of
    the terms they share, given once for all of them, or nothing."""
    names: list[str] = []
    texts = [(write or _repr_with)(value, names) for value in values]
    return texts, f" with {', '.join(names)}" if names else ""


def _repr_with(value: Any, names: list[str] | None) -> str:
    if value.__class__ in _EXPRESSIONS:
        return shared_text(value, _node_repr, _leaf_repr, None, names)
    return repr(value)


class _UndecidableComparisons:
    """Comparison operators that defer to ``UndecidableComparison``, and the
    bool operators ``&``, ``|`` and ``^`` that ``bit_logic`` traces."""

    __slots__ = ()
    __hash__ = object.__hash__

    def __eq__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        return UndecidableComparison(self, "==", other)

    def __ne__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        return UndecidableComparison(self, "!=", other)

    def __lt__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, "<", other)

    def __le__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, "<=", other)

    def __gt__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, ">", other)

    def __ge__(self, other: Any) -> UndecidableComparison:
        return UndecidableComparison(self, ">=", other)

    # On bools &, | and ^ commute, so each is its own reflection.
    def __and__(self, other: Any) -> Any:
        from ._conditions import bit_logic

        return bit_logic("&", self, other)

    def __or__(self, other: Any) -> Any:
        from ._conditions import bit_logic

        return bit_logic("|", self, other)

    def __xor__(self, other: Any) -> Any:
        from ._conditions import bit_logic

        return bit_logic("^", self, other)

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__


class UndecidableComparison(_UndecidableComparisons):
    """A comparison whose outcome is only known when the program runs.

    Comparing a symbolic parameter or an IR value builds one of these
    instead of falling back to object identity, and so does comparing one
    of these; using it as a condition raises, so a kernel never silently
    takes one branch. ``op`` is a comparison, or ``"and"`` or ``"or"`` of
    two such bools (see ``bit_logic``).
    """

    __slots__ = ("lhs", "measured", "op", "rhs")
    # Whether it is computed from a measurement result: set when first
    # asked for (``_conditions.measured``), not when it is made.
    measured: bool

    def __init__(self, lhs: Any, op: str, rhs: Any):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def __bool__(self) -> bool:
        from ._conditions import measured, measurement_error

        if measured(self):
            raise measurement_error(self.lhs, self.op, self.rhs)
        if self.lhs.__class__ is IRValue or self.rhs.__class__ is IRValue:
            raise CompilationError(
                f"cannot compare a runtime IR value at trace time ({self.lhs!r} {self.op} "
                f"{self.rhs!r}); branch on a measurement result directly with "
                "`if result:` or `if not result:`",
                code="runtime_value",
            )
        names = parameter_names(self)
        if not names or contains_answer(self):
            raise CompilationError(
                f"cannot decide {self!r} at trace time: it is known only when the program "
                "runs; use it as an if condition, alone or with and, or, not",
                code="runtime_value",
            )
        (lhs, rhs), shared = written_together((self.lhs, self.rhs))
        raise SymbolicValueError(
            f"cannot decide {lhs} {self.op} {rhs}{shared} at trace time: a float "
            "argument has no value while the kernel compiles",
            names,
            f"the comparison {self.op}",
        )

    def __repr__(self) -> str:
        return shared_text(self, _node_repr, _leaf_repr)


class _SymbolicNumber(_UndecidableComparisons):
    """Arithmetic shared by symbolic kernel parameters and expressions."""

    __slots__ = ()

    def _binary(self, op: str, other: Any) -> ParameterExpression:
        return ParameterExpression(op, self, other)

    def _rbinary(self, op: str, other: Any) -> ParameterExpression:
        return ParameterExpression(op, other, self)

    def __add__(self, other: Any) -> ParameterExpression:
        return self._binary("add", other)

    def __radd__(self, other: Any) -> ParameterExpression:
        return self._rbinary("add", other)

    def __sub__(self, other: Any) -> ParameterExpression:
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> ParameterExpression:
        return self._rbinary("sub", other)

    def __mul__(self, other: Any) -> ParameterExpression:
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> ParameterExpression:
        return self._rbinary("mul", other)

    def __truediv__(self, other: Any) -> ParameterExpression:
        return self._binary("div", other)

    def __rtruediv__(self, other: Any) -> ParameterExpression:
        return self._rbinary("div", other)

    def __floordiv__(self, other: Any) -> ParameterExpression:
        return self._binary("floordiv", other)

    def __rfloordiv__(self, other: Any) -> ParameterExpression:
        return self._rbinary("floordiv", other)

    def __mod__(self, other: Any) -> ParameterExpression:
        return self._binary("mod", other)

    def __rmod__(self, other: Any) -> ParameterExpression:
        return self._rbinary("mod", other)

    def __pow__(self, other: Any) -> ParameterExpression:
        return self._binary("pow", other)

    def __rpow__(self, other: Any) -> ParameterExpression:
        return self._rbinary("pow", other)

    def __neg__(self) -> ParameterExpression:
        return ParameterExpression("neg", self)

    def cast(self, dtype: str) -> ParameterExpression:
        if dtype not in ("i64", "f64"):
            raise InternalError(f"unsupported symbolic cast to {dtype}")
        return ParameterExpression("cast", self, dtype=dtype)

    def _not_concrete(self, operation: str) -> NoReturn:
        names = parameter_names(self)
        if not names:
            raise CompilationError(
                f"{operation} needs the value of {arithmetic_text(self)}, the answer of a "
                "classical call, which is known only when the program runs; use it as a "
                "pulse op argument (amplitude, phase, frequency) or in an if condition",
                code="runtime_value",
            )
        quoted = ", ".join(map(repr, dict.fromkeys(names)))
        arguments = "argument" if len(set(names)) == 1 else "arguments"
        raise SymbolicValueError(
            f"{operation} needs the value of float {arguments} {quoted}, which is only known "
            "when the kernel runs; use it only as a pulse op argument (amplitude, phase, "
            "frequency, wait length), or annotate it as int to compile one kernel per value",
            names,
            operation,
        )

    def __bool__(self) -> bool:
        self._not_concrete("bool()")

    def __float__(self) -> float:
        self._not_concrete("float()")

    def __int__(self) -> int:
        self._not_concrete("int()")

    def __complex__(self) -> complex:
        self._not_concrete("complex()")

    def __index__(self) -> int:
        self._not_concrete("an index")


class Parameter(_SymbolicNumber):
    """Sentinel for a symbolic kernel parameter (compile-once, evaluate-many).

    Instances track an index and type so the packed IR builder can emit
    a PARAM opcode instead of a concrete value.
    """

    __slots__ = ("name", "index", "dtype")

    def __init__(self, name: str, index: int, dtype: str = "unknown"):
        if dtype not in ("unknown", "i64", "f64"):
            raise InternalError(f"unsupported parameter dtype {dtype!r}")
        self.name = name
        self.index = index
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"Parameter({self.name!r}, idx={self.index}, {self.dtype})"


class ParameterExpression(_SymbolicNumber):
    """A symbolic arithmetic expression rooted in kernel parameters."""

    __slots__ = ("op", "lhs", "rhs", "dtype")

    def __init__(self, op: str, lhs: Any, rhs: Any = None, *, dtype: str = "unknown"):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self.dtype = dtype

    def __repr__(self) -> str:
        return shared_text(self, _node_repr, _leaf_repr)


# The operations an expression is made of, for ``expression_nodes``. The
# walks test a value's own class against these: a subclass of one would be
# taken for an operand.
ARITHMETIC: tuple[type, ...] = (ParameterExpression,)
COMPARISONS: tuple[type, ...] = (UndecidableComparison,)
_EXPRESSIONS: tuple[type, ...] = (ParameterExpression, UndecidableComparison)


def _leaf_repr(value: Any, names: list[str] | None) -> str:
    return repr(value)


def _node_repr(node: Any, lhs: str, rhs: str, plain: Callable[[Any], bool]) -> str:
    if node.__class__ is UndecidableComparison:
        return f"({lhs} {node.op} {rhs})"
    if node.op in ("neg", "cast"):
        return f"ParameterExpression({node.op}, {lhs}, {node.dtype})"
    return f"ParameterExpression({node.op}, {lhs}, {rhs})"


class AnswerValue(_SymbolicNumber):
    """A value a classical function returns, known only when the program runs.

    It takes part in arithmetic like a float kernel parameter, and can set
    amplitudes, phases and frequencies or decide a branch, but not a
    duration. ``dtype`` is ``"i1"``, ``"i64"`` or ``"f64"``; ``scope`` is the
    regions open where it was taken, outside which it is not defined.
    """

    __slots__ = ("vid", "dtype", "scope", "label")

    def __init__(self, vid: int, dtype: str, scope: tuple[int, ...], label: str = "") -> None:
        self.vid = vid
        self.dtype = dtype
        self.scope = scope
        self.label = label

    @property
    def vtype(self) -> str:
        return self.dtype

    def __hash__(self) -> int:
        raise CompilationError(
            f"{self!r} is known only when the program runs, so it has no hash: it cannot be "
            "a set member or a dict key; compare it with == or test it with `in` a tuple",
            code="runtime_value",
        )

    def __repr__(self) -> str:
        return f"AnswerValue({self.label or self.vid}, {self.dtype})"


class Selection(_SymbolicNumber):
    """A value a condition selects while the program runs: ``then`` where
    ``cond`` (a measurement, or a boolean IR value) holds, ``other`` where it
    does not. ``then`` and ``other`` are numbers, symbolic numbers, waveforms
    or selections; the builder emits one select of them."""

    __slots__ = ("cond", "then", "other")
    __hash__ = object.__hash__

    def __init__(self, cond: IRValue, then: Any, other: Any) -> None:
        self.cond = cond
        self.then = then
        self.other = other


def contains_answer(value: Any) -> bool:
    """Whether a symbolic value is computed from an answer of a classical call."""
    if isinstance(value, AnswerValue):
        return True
    if isinstance(value, (ParameterExpression, UndecidableComparison)):
        return any(isinstance(leaf, AnswerValue) for leaf in expression_leaves(value))
    return False


def is_symbolic(value: Any) -> bool:
    """Return whether *value* is a symbolic parameter or expression."""
    return isinstance(value, _SymbolicNumber)


_ARITHMETIC = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def is_arithmetic(value: Any) -> bool:
    """Whether *value* is ``+ - * /`` and negation of parameters and numbers:
    what ``arithmetic_source`` can compute."""
    cls = value.__class__
    if cls is Parameter or cls is float or cls is int:
        return True
    if cls is ParameterExpression:
        for node in expression_nodes(value, ARITHMETIC):
            op = node.op
            if op in _ARITHMETIC:
                if not (_arithmetic_operand(node.lhs) and _arithmetic_operand(node.rhs)):
                    return False
            elif op == "neg" or (op == "cast" and node.dtype == "f64"):
                if not _arithmetic_operand(node.lhs):
                    return False
            else:
                return False
        return True
    return isinstance(value, numbers.Real)


# Whether an operand of arithmetic is a parameter, a number, or
# arithmetic itself, which ``is_arithmetic`` tests where it comes.
def _arithmetic_operand(value: Any) -> bool:
    cls = value.__class__
    return (
        cls is ParameterExpression
        or cls is Parameter
        or cls is float
        or cls is int
        or isinstance(value, numbers.Real)
    )


def arithmetic_source(value: Any) -> str | None:
    """Python source that computes *value* from ``a``, the float arguments
    in parameter order, with the f64 operations the module computes it with;
    None unless it is ``+ - * /`` and negation of parameters and numbers."""
    if isinstance(value, Parameter):
        return f"a[{value.index}]"
    if isinstance(value, ParameterExpression):
        nodes = expression_nodes(value, ARITHMETIC)
        # Written in place, an operation used twice is computed twice and
        # one two hundred deep is more parentheses deep than Python reads.
        # Before either, every operation is a named term of one flat
        # tuple, computed once and in the same order: (t0 := a[0] * 2.0,
        # t1 := t0 + t0)[-1].
        depth: dict[int, int] = {}
        flat = False
        for node in nodes:
            # A cast is written as its operand, without parentheses of its own.
            deep = max(depth.get(id(node.lhs), 0), depth.get(id(node.rhs), 0))
            deep += node.op != "cast"
            depth[id(node)] = deep
            if deep > _WRITTEN_IN_PLACE:
                flat = True
        if not flat and len(nodes) > 1:
            used: set[int] = set()
            for node in nodes:
                for operand in (node.lhs, node.rhs):
                    if id(operand) in depth:
                        if id(operand) in used:
                            flat = True
                        used.add(id(operand))
        texts: dict[int, str | None] = {}
        terms: list[str] = []
        for node in nodes:
            lhs = texts[id(node.lhs)] if id(node.lhs) in texts else _source_operand(node.lhs)
            text: str | None
            if node.op == "neg":
                text = None if lhs is None else f"(-{lhs})"
            elif node.op == "cast":
                text = lhs if node.dtype == "f64" else None
            else:
                symbol = _ARITHMETIC.get(node.op)
                rhs = texts[id(node.rhs)] if id(node.rhs) in texts else _source_operand(node.rhs)
                text = (
                    None
                    if symbol is None or lhs is None or rhs is None
                    else f"({lhs} {symbol} {rhs})"
                )
            if text is None:
                return None
            if flat and node.op != "cast":
                terms.append(f"t{len(terms)} := {text}")
                text = f"t{len(terms) - 1}"
            texts[id(node)] = text
        if flat and terms:
            return f"({', '.join(terms)}, {texts[id(value)]})[-1]"
        return texts[id(value)]
    if isinstance(value, numbers.Real):
        return repr(float(value))
    return None


# The parentheses deep an expression is still written in place in its
# source. Python reads 200; the tests a wait's source is written into add a
# few.
_WRITTEN_IN_PLACE = 190


def _source_operand(value: Any) -> str | None:
    if isinstance(value, Parameter):
        return f"a[{value.index}]"
    if isinstance(value, numbers.Real):
        return repr(float(value))
    return None


def without_answers(value: Any) -> Any:
    """*value* with every answer of a classical call taken as 0: the part of
    a wait length that its float parameters give."""
    if isinstance(value, AnswerValue):
        return 0.0
    if isinstance(value, ParameterExpression):
        made: dict[int, Any] = {}
        for node in expression_nodes(value, ARITHMETIC):
            lhs, rhs = node.lhs, node.rhs
            made[id(node)] = ParameterExpression(
                node.op,
                made[id(lhs)] if id(lhs) in made else 0.0 if isinstance(lhs, AnswerValue) else lhs,
                made[id(rhs)] if id(rhs) in made else 0.0 if isinstance(rhs, AnswerValue) else rhs,
                dtype=node.dtype,
            )
        return made[id(value)]
    return value


def arithmetic_text(value: Any, names: list[str] | None = None) -> str:
    """*value* as the user wrote it, roughly: ``tau / 2``, ``2.0 * tau + 40.0``;
    *names* as ``shared_text`` takes them."""
    if isinstance(value, Parameter):
        return value.name
    if isinstance(value, AnswerValue):
        return value.label or "answer"
    if isinstance(value, ParameterExpression):
        return shared_text(value, _arithmetic_node_text, arithmetic_text, ARITHMETIC, names)
    if value.__class__ is UndecidableComparison:
        return shared_text(value, _node_repr, _leaf_repr, None, names)
    return repr(value)


_PRECEDENCE = {"add": 1, "sub": 1, "mul": 2, "div": 2}


def _arithmetic_node_text(node: Any, lhs: str, rhs: str, plain: Callable[[Any], bool]) -> str:
    if node.op == "neg":
        return f"-{_operand_text(node.lhs, lhs, 3, plain)}"
    if node.op == "cast":
        return lhs
    symbol = _ARITHMETIC.get(node.op, node.op)
    level = _PRECEDENCE.get(node.op, 3)
    lhs = _operand_text(node.lhs, lhs, level, plain)
    rhs = _operand_text(node.rhs, rhs, level + (node.op in ("sub", "div")), plain)
    return f"{lhs} {symbol} {rhs}"


# *text*, which writes the operand *value*, in parentheses where it is
# an operation written in place that binds less than *level*.
def _operand_text(value: Any, text: str, level: int, plain: Callable[[Any], bool]) -> str:
    if plain(value) and _PRECEDENCE.get(value.op, 3) < level:
        return f"({text})"
    return text


class IRValue(_UndecidableComparisons):
    __slots__ = ("vid", "vtype", "name", "chan")

    def __init__(self, vid: int, vtype: str, name: str = ""):
        self.vid = vid
        self.vtype = vtype
        self.name = name
        # The qudit channel cell this line or tone value is, or was, a
        # version of; set by the tracer.
        self.chan: Any = None

    def __bool__(self) -> bool:
        what = "a measurement result" if self.vtype == "measurement" else repr(self)
        raise CompilationError(
            f"cannot decide {what} at trace time: it is known only when the program runs; "
            "branch on it directly with `if result:` or `if not result:`",
            code="runtime_value",
        )

    def __repr__(self) -> str:
        return f"{self.name or f'%v{self.vid}'}:{self.vtype}"


LINEAR_TYPES = frozenset({"drive_line", "readout_line", "tone"})

# The vid of a loop-carried value that has no value entering the loop.
UNDEFINED_VID = -1

# The longest duration, in cycles: a time a float holds to the cycle.
MAX_CYCLES = 2**53

# (value_arg_count, attr_names, result_types)
#   value_arg_count: leading args that are IR values; -1 = variadic all-values
#   result_types: None = mirror operand types
OP_TABLE: dict[str, tuple[int, tuple[str, ...], tuple[str, ...] | None]] = {
    "get_drive_line": (1, (), ("drive_line", "tone")),
    "get_readout_line": (1, (), ("readout_line", "tone")),
    "gaussian": (0, ("duration", "amplitude", "sigma"), ("waveform",)),
    "square": (0, ("duration", "amplitude"), ("waveform",)),
    "drag": (0, ("duration", "amplitude", "sigma", "beta"), ("waveform",)),
    "cosine": (0, ("duration", "amplitude"), ("waveform",)),
    "tanh_ramp": (0, ("duration", "amplitude", "sigma"), ("waveform",)),
    "gaussian_square": (0, ("duration", "amplitude", "sigma", "width"), ("waveform",)),
    "custom": (0, ("duration", "name"), ("waveform",)),
    "custom_samples": (0, ("samples",), ("waveform",)),
    "drive": (3, (), ("drive_line", "tone")),
    "readout": (3, (), ("readout_line", "tone", "measurement")),
    "wait": (1, ("duration",), None),
    "sync": (-1, (), None),
    "shift_phase": (1, ("phase_rad",), ("tone",)),
    "set_phase": (1, ("phase_rad",), ("tone",)),
    "shift_frequency": (1, ("freq_hz",), ("tone",)),
    "set_frequency": (1, ("freq_hz",), ("tone",)),
    "wf_add": (2, (), ("waveform",)),
    "wf_sub": (2, (), ("waveform",)),
    "wf_mul": (2, (), ("waveform",)),
    "wf_scale": (1, ("scale",), ("waveform",)),
    "wf_neg": (1, (), ("waveform",)),
}


# The ops with times (pulse.custom is sampled before it is emitted).
_TIMED_OPS = frozenset(
    {"gaussian", "square", "drag", "cosine", "tanh_ramp", "gaussian_square", "wait"}
)
_UNCOUNTED_OPS = frozenset({"pulse.qudit_arg", "pulse.qudit_alloc", "scf.yield"})


class PythonIRBuilder:
    """An in-memory IR builder that keeps each op as Python data, so tests can
    inspect what a kernel traces to; compilation uses ``PackedIRBuilder``."""

    # Accepts scf.for regions, so the tracer may emit a loop once.
    supports_regions = True
    # The samples every line plays per program cycle; see PackedIRBuilder.
    sample_rate: tuple[int, int] | None = None
    # Each line's sample rate, when they differ; see PackedIRBuilder.
    sample_rate_of: Any = None
    tracer: Any = None

    def __init__(self, name: str = "main"):
        self.name = name
        self.ops: list[Op] = []
        self._next_id = 0
        # The trace's loop-unroll threshold; None for the default.
        self.unroll_threshold: int | None = None
        # What the traced kernel returned.
        self.result: Any = None

    def emit(
        self,
        kind: str,
        operands: tuple[IRValue, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        vid = self._next_id
        count = len(result_types)
        results: tuple[IRValue, ...]
        if count == 1:
            results = (IRValue(vid, result_types[0]),)
        elif count == 2:
            results = (IRValue(vid, result_types[0]), IRValue(vid + 1, result_types[1]))
        else:
            results = tuple([IRValue(vid + i, rt) for i, rt in enumerate(result_types)])
        self._next_id = vid + count
        if attrs and kind.removeprefix("pulse.") in _TIMED_OPS:
            from ._durations import in_program_cycles

            in_program_cycles(kind.removeprefix("pulse."), attrs, getattr(self, "clock_ghz", 2.0))
        self.ops.append(Op(kind, operands, results, attrs or {}))
        return results

    def mark(self) -> int:
        """A position that ``insert`` can later insert at."""
        return len(self.ops)

    def insert(self, index: int, kind: str, operands: tuple[IRValue, ...]) -> None:
        """Insert a result-less op before ``ops[index]``."""
        self.ops.insert(index, Op(kind, operands, (), {}))

    def op_count(self) -> int:
        """Pulse-program ops, not counting qudit binding or region yields."""
        return sum(op.kind not in _UNCOUNTED_OPS for op in self.ops)

    def checkpoint(self) -> tuple[int, int]:
        return len(self.ops), self._next_id

    def rollback(self, checkpoint: tuple[int, int]) -> None:
        op_count, self._next_id = checkpoint
        del self.ops[op_count:]
