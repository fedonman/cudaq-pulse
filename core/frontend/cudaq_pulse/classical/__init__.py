# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classical co-processing at QPU runtime.

``classical`` is a classical real-time dialect for co-processing at QPU
runtime: the GPU, host or FPGA services that a pulse program calls while the
QPU runs, to decode syndromes, update calibrations or receive bulk readout
data, and whose answers later pulses use.

``pulse.classical`` is both this module and the decorator that declares such
a function. Its type hints say what it takes and returns::

    import cudaq_pulse as pulse

    @pulse.classical(max_latency_ns=3000)
    def decode(syndromes: pulse.Bits) -> bool:
        return bool(syndromes[0] ^ syndromes[3])

    @pulse.kernel
    def round(...):
        ...
        answer = decode.submit([m0, m1, m2, m3])   # send the request
        if answer.result():                        # take the answer
            drive(...)

``pulse.Bits`` takes measurements read as bits, ``list[int]`` measured
levels or numbers, ``list[complex]`` IQ points and ``list[float]`` numbers;
``float``, ``int`` and ``complex`` take one value. A list's length comes from
the call: a function called with lists of two lengths compiles to two
versions, each named by its lengths (``decode__4``). Results are typed the
same way, and a list result's length is inferred from the body per version.
``pulse.Length[pulse.Bits, 8]`` states a fixed length, for a quantum system that
takes one size.

Inside a kernel, ``.submit(...)`` sends a request and returns a future at
once; pulses that do not need the answer keep playing. ``.result()`` is where
the answer is taken: it is due the maximum latency after the request is sent,
and the pulses whose amplitude, phase, frequency or branch condition come from
it start no earlier. An answer sets a duration only through a variable
wait, ``wait(line, answer, bounds=(lo, hi))``, which begins a new segment of
the timeline. Outside a kernel, calling the function runs its body, its
reference implementation, directly.

The body is what simulation runs. On hardware the service registered under
the function's dispatch key answers instead: the 32-bit FNV-1a hash of its
compiled name, as CUDA-Q Realtime uses, unless the firmware of the
processor that runs it fixes it (its ``dispatch_keys``).
``call.dispatch_key`` and the call manifest give each one.
"""

from __future__ import annotations

import builtins
import collections.abc
import math
import sys
import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Required, TypedDict, Unpack, overload

from .._errors import InvalidTypeError, InvalidValueError
from .._units import Time, as_nanoseconds

if TYPE_CHECKING:
    from ._link import Link as Link
from .._native import infer_list_lengths

__all__ = [
    "Bits",
    "ClassicalDecorator",
    "ClassicalFunction",
    "Link",
    "Pending",
    "external",
    "function",
    "dispatch_key",
    "random_index",
]

_LATENCY_MODES = ("bounded", "timeout", "blocking")
_SUBMIT_POLICIES = ("eager", "lazy")
# The handler schema of the CUDA-Q Realtime dispatcher.
_MAX_ARGS = 8
_MAX_RESULTS = 4


# The wire types of arguments and results. A function's parameter and return
# annotations become these when it is declared (``pulse.Bits`` of a known
# length is ``bits:n``, ``float`` is ``f64``, ``list[int]`` of measurements is
# ``levels:n``, and so on). They are what the IR, the wire schema and the
# simulator read; users never name them.


class ValueType:
    """The type of one argument or result of a classical function.

    ``descriptor`` is its spelling in the IR, ``count`` how many values it
    holds, ``source`` what a kernel passes for it (``"measurement"``,
    ``"float"`` or ``"integer"``), and ``answer`` the type of the values a
    kernel gets back for it (``"i1"``, ``"i64"`` or ``"f64"``).
    """

    __slots__ = ("count", "descriptor", "source", "answer", "answer_count", "wire_bytes")

    def __init__(
        self,
        descriptor: str,
        count: int,
        source: str,
        answer: str,
        answer_count: int,
        wire_bytes: int,
    ) -> None:
        self.descriptor = descriptor
        self.count = count
        self.source = source
        self.answer = answer
        self.answer_count = answer_count
        self.wire_bytes = wire_bytes

    @property
    def is_scalar(self) -> bool:
        return ":" not in self.descriptor and "[" not in self.descriptor

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ValueType) and other.descriptor == self.descriptor

    def __hash__(self) -> int:
        return hash(self.descriptor)

    def __repr__(self) -> str:
        return f"<{self.descriptor}>"


def _count(n: int, what: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise InvalidValueError(f"{what} needs a positive whole number of values, got {n!r}")
    return n


class _Bits(ValueType):
    """``n`` measurement results, sent as ``outcome != 0``, bit-packed; as a
    result, ``n`` booleans."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a pulse.Bits value")
        super().__init__(f"bits:{n}", n, "measurement", "i1", n, (n + 7) // 8)


class _OneBit(_Bits):
    """``bits:1`` answered as one ``bool`` (a ``-> bool`` result)."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(1)

    @property
    def is_scalar(self) -> bool:
        return True


class _Levels(ValueType):
    """``n`` measured levels, one byte each; as a result, ``n`` integers."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a list[int] of measurements")
        super().__init__(f"levels:{n}", n, "measurement", "i64", n, n)


class _IQ(ValueType):
    """The IQ points of ``n`` readouts, as ``2n`` 32-bit floats (I, then Q)."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a list[complex]")
        super().__init__(f"iq:{n}", n, "measurement", "f64", 2 * n, 8 * n)


class _OnePoint(_IQ):
    """``iq:1`` taken as one ``complex`` (a ``complex`` parameter)."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(1)

    @property
    def is_scalar(self) -> bool:
        return True


def _width(bits: int) -> int:
    if bits not in (64, 32):
        raise InvalidValueError(f"a number has 64 or 32 bits, got {bits!r}")
    return bits


class _Float(ValueType):
    """One floating-point number, of 64 bits (what ``float`` is) or 32."""

    __slots__ = ()

    def __init__(self, bits: int = 64) -> None:
        bits = _width(bits)
        super().__init__(f"f{bits}", 1, "float", "f64", 1, bits // 8)


class _Integer(ValueType):
    """One signed integer, of 64 bits (the default) or 32."""

    __slots__ = ()

    def __init__(self, bits: int = 64) -> None:
        bits = _width(bits)
        super().__init__(f"i{bits}", 1, "integer", "i64", 1, bits // 8)


class _Byte(ValueType):
    """One unsigned byte."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("u8", 1, "integer", "i64", 1, 1)


class _Floats(ValueType):
    """``n`` floating-point numbers, of 64 bits (``list[float]``) or 32 each."""

    __slots__ = ()

    def __init__(self, n: int, bits: int = 64) -> None:
        n = _count(n, "a list[float]")
        bits = _width(bits)
        super().__init__(f"f{bits}[{n}]", n, "float", "f64", n, n * bits // 8)


class _Integers(ValueType):
    """``n`` signed 32-bit integers."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a list[int]")
        super().__init__(f"i32[{n}]", n, "integer", "i64", n, 4 * n)


class _Bytes(ValueType):
    """``n`` unsigned bytes."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "a byte array")
        super().__init__(f"u8[{n}]", n, "integer", "i64", n, n)


def dispatch_key(name: str) -> int:
    """The dispatch key of a function compiled as *name*, unless its
    controller fixes one: 32-bit FNV-1a, the hash CUDA-Q Realtime registers
    handlers by."""
    value = 2166136261
    for byte in name.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


# Keys built-in protocols send, by the protocol: a user function given one
# (by its name's hash or a controller) is a compile error. These are the
# dispatch_key of enqueue_syndromes, get_corrections, get_logical_measurements
# and reset_decoder, which pulse.classical(cudaq_qec.qec.<name>) sends.
_RESERVED_KEYS = {
    0x7ED8BE82: "CUDA-Q QEC's decoding server",
    0x882D5BA1: "CUDA-Q QEC's decoding server",
    0x3056A149: "CUDA-Q QEC's decoding server",
    0x977A59CF: "CUDA-Q QEC's decoding server",
}

#: A CUDA-QX decoder's status for corrections it has not decoded yet.
_NOT_READY = 4


class _NotReadyError(Exception):
    """Raised by a decoder's handler with nothing decoded: the server's
    NOT_READY, which a wait with a timeout takes as a late answer."""


# Type checkers see a list of ints, so a body returns any list of bits and
# ``pulse.Bits()``, an external's placeholder result, checks too (an alias made
# with a ``type`` statement can't be called).
if typing.TYPE_CHECKING:
    Bits: typing.TypeAlias = list[int]  # noqa: UP040
else:

    class Bits(list):
        """Measurements read as bits (``outcome != 0``), bit-packed on the wire.

        A kernel passes a list of measurements, ``fn.submit([m0, m1])``; the
        body reads ``bits[0]``, ``len(bits)`` or iterates. As a result, the
        bits are answered as booleans a kernel can branch on. The length
        comes from the call, or is stated, ``pulse.Length[pulse.Bits, 8]``.
        ``pulse.Bits()`` is an empty list, an external's placeholder result.
        """

        __slots__ = ()


# ── Reading type hints ───────────────────────────────────────────────────────

# List kinds: measurements read as bits, measurements read as IQ points,
# numbers as f64, and list[int] (measured levels or i32 numbers, by what
# the call passes).
_KIND_BITS, _KIND_IQ, _KIND_FLOATS, _KIND_INTS = 2, 3, 4, 5
_LIST_KIND: dict[Any, int] = {
    Bits: _KIND_BITS,
    list[complex]: _KIND_IQ,
    list[float]: _KIND_FLOATS,
    list[int]: _KIND_INTS,
    collections.abc.Sequence[complex]: _KIND_IQ,
    collections.abc.Sequence[float]: _KIND_FLOATS,
    collections.abc.Sequence[int]: _KIND_INTS,
}


# The list kind of a ``typing.Sequence[...]`` hint, which is not equal
# to the ``collections.abc`` one it stands for.
def _sequence_kind(t: Any) -> int | None:
    if getattr(t, "_name", None) == "Sequence" and typing.get_origin(t) is collections.abc.Sequence:
        return _LIST_KIND.get(collections.abc.Sequence[t.__args__])
    return None


_LIST_NAME = {
    _KIND_BITS: "pulse.Bits",
    _KIND_IQ: "list[complex]",
    _KIND_FLOATS: "list[float]",
    _KIND_INTS: "list[int]",
}
_F64, _I64 = _Float(), _Integer()
_SCALAR_ARG: dict[Any, ValueType] = {float: _F64, int: _I64, complex: _OnePoint()}
_SCALAR_RESULT: dict[Any, ValueType] = {float: _F64, int: _I64, bool: _OneBit()}
_VARIADIC = 0x04 | 0x08  # CO_VARARGS | CO_VARKEYWORDS


def _sized(kind: int, n: int, measured: bool = True) -> ValueType:
    if kind == _KIND_BITS:
        return _Bits(n)
    if kind == _KIND_IQ:
        return _IQ(n)
    if kind == _KIND_FLOATS:
        return _Floats(n)
    return _Levels(n) if measured else _Integers(n)


_NOT_A_NAME: Any = object()
_HINTS: dict[tuple[int, str], Any] = {}
_BUILTIN_NAMES = {"float": float, "int": int, "bool": bool, "complex": complex, "None": None}


# An annotation's value. A ``from __future__`` string is looked up when
# it is a plain or dotted name or a one-level ``list[...]``/``tuple[...]``,
# else evaluated; each is read once per module and string.
def _hint(t: Any, fn: Any) -> Any:
    if t.__class__ is not str:
        return t
    g = fn.__globals__
    key = (id(g), t)
    try:
        return _HINTS[key]
    except KeyError:
        pass
    value = _name(t, g)
    if value is _NOT_A_NAME:
        value = _subscript(t, g)
    if value is _NOT_A_NAME:
        try:
            value = eval(t, g)  # the function's own annotation
        except Exception as error:
            raise InvalidTypeError(
                f"{fn.__qualname__}: can't read the type hint {t!r} ({error}); a hint is read"
                " in the function's module, so the names it uses must be defined there"
            ) from None
    _HINTS[key] = value
    return value


def _name(t: str, g: dict[str, Any]) -> Any:
    if t in _BUILTIN_NAMES and t not in g:
        return _BUILTIN_NAMES[t]
    if not t.replace(".", "").replace("_", "a").isalnum() or t[:1].isdigit():
        return _NOT_A_NAME
    head, *rest = t.split(".")
    if head in g:
        value = g[head]
    elif hasattr(builtins, head):
        value = getattr(builtins, head)
    else:
        return _NOT_A_NAME  # eval raises the NameError
    for part in rest:
        value = getattr(value, part, _NOT_A_NAME)
        if value is _NOT_A_NAME:
            return _NOT_A_NAME
    return value


# ``tuple[float, float]``, ``list[int]`` or ``pulse.Length[pulse.Bits,
# 8]``, one level deep, without eval.
def _subscript(t: str, g: dict[str, Any]) -> Any:
    i = t.find("[")
    if i <= 0 or t[-1] != "]" or "[" in t[i + 1 :]:
        return _NOT_A_NAME
    head = _name(t[:i].strip(), g)
    items = [_item(s.strip(), g) for s in t[i + 1 : -1].split(",")]
    if head is _NOT_A_NAME or any(x is _NOT_A_NAME for x in items):
        return _NOT_A_NAME
    try:
        return head[tuple(items)] if len(items) > 1 else head[items[0]]
    except TypeError:
        return _NOT_A_NAME


def _item(s: str, g: dict[str, Any]) -> Any:
    if s.isdigit():
        return int(s)
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'" and s[1:-1].isidentifier():
        return s[1:-1]
    return _name(s, g)


def _result_items(t: Any) -> tuple[Any, ...]:
    if t is None or t is type(None):
        return ()
    if typing.get_origin(t) is tuple:
        return typing.get_args(t)
    if isinstance(t, type) and issubclass(t, tuple) and hasattr(t, "_fields"):
        hints = typing.get_type_hints(t)
        return tuple(hints[f] for f in t._fields)
    return (t,)


class _Rule:
    """A list result whose length each version infers from the body."""

    __slots__ = ("kind",)

    def __init__(self, kind: int) -> None:
        self.kind = kind


class _Shape:
    """A function's declaration read from its hints: parameter names, per
    parameter a wire type or an unsized list kind (an int), and per result
    a wire type or a ``_Rule``; how many parameters are unsized and whether
    any result is a rule."""

    __slots__ = ("names", "plan", "results", "unsized", "rules")

    def __init__(
        self,
        names: tuple[str, ...],
        plan: tuple[Any, ...],
        results: tuple[Any, ...],
        unsized: int,
        rules: bool,
    ):
        self.names, self.plan, self.results = names, plan, results
        self.unsized, self.rules = unsized, rules


# Read a function's annotations from ``__code__`` and
# ``__annotations__`` (no ``inspect.signature`` or ``get_type_hints``).
def _read_hints(fn: Any) -> _Shape:
    where = fn.__qualname__
    code = fn.__code__
    if code.co_flags & _VARIADIC or code.co_kwonlyargcount:
        raise InvalidTypeError(f"{where}: only positional parameters are sent")
    names = code.co_varnames[: code.co_argcount]
    ann = fn.__annotations__
    plan: list[Any] = []
    unsized = 0
    for name in names:
        if name not in ann:
            raise InvalidTypeError(
                f"{where}: parameter {name!r} needs a type, such as float or pulse.Bits"
            )
        t = _hint(ann[name], fn)
        spec: Any = _SCALAR_ARG.get(t)
        if spec is None:
            kind = _LIST_KIND.get(t)
            if kind is None:
                kind = _sequence_kind(t)
            if kind is None:
                spec = _stated_parameter(where, name, t)
            else:
                spec = kind
                unsized += 1
        plan.append(spec)
    if "return" not in ann:
        raise InvalidTypeError(f"{where}: annotate what it returns (-> None if it answers nothing)")
    returned = _hint(ann["return"], fn)
    if len(plan) <= _MAX_ARGS:
        try:
            scalar = _SCALAR_RESULT.get(returned)
        except TypeError:  # unhashable: not a type, the checks below say so
            scalar = None
        if scalar is not None:
            return _Shape(tuple(names), tuple(plan), (scalar,), unsized, False)
    items = _result_items(returned)
    results: list[Any] = []
    for t in items:
        spec = _SCALAR_RESULT.get(t)
        if spec is None:
            kind = _LIST_KIND.get(t)
            if kind is None:
                kind = _sequence_kind(t)
            if kind is None:
                spec = _stated_result(where, t)
            elif kind == _KIND_IQ:
                _bad_result(where, t)
            else:
                spec = _Rule(kind)
        results.append(spec)
    if len(plan) > _MAX_ARGS:
        raise InvalidTypeError(f"{where}: at most {_MAX_ARGS} arguments, got {len(plan)}")
    if len(results) > _MAX_RESULTS:
        raise InvalidTypeError(f"{where}: at most {_MAX_RESULTS} results, got {len(results)}")
    rules = any(r.__class__ is _Rule for r in results)
    return _Shape(tuple(names), tuple(plan), tuple(results), unsized, rules)


# A parameter of fixed length, ``pulse.Length[kind, n]``: *n*
# measurements read as bits for ``pulse.Bits``, else *n* numbers.
def _stated_parameter(where: str, name: str, t: Any) -> ValueType:
    if typing.get_origin(t) is not typing.Annotated:
        _bad_parameter(where, name, t)
    base, meta = t.__origin__, t.__metadata__
    kind = _LIST_KIND.get(base)
    if kind is None:
        kind = _sequence_kind(base)
    if kind is None:
        if base in _SCALAR_ARG or base is bool:
            raise InvalidTypeError(
                f"{where}: parameter {name!r} is {_spell(base)}; pulse.Length applies to"
                " pulse.Bits and lists"
            )
        _bad_parameter(where, name, base)
    return _sized(kind, _length(where, kind, meta, "parameter"), measured=False)


# A list result of fixed length, ``pulse.Length[kind, n]``.
def _stated_result(where: str, t: Any) -> ValueType:
    if typing.get_origin(t) is not typing.Annotated:
        _bad_result(where, t)
    base, meta = t.__origin__, t.__metadata__
    kind = _LIST_KIND.get(base)
    if kind is None:
        kind = _sequence_kind(base)
    if kind is None or kind == _KIND_IQ:
        if base in _SCALAR_RESULT or base is complex:
            raise InvalidTypeError(
                f"{where}: its result is {_spell(base)}; pulse.Length applies to pulse.Bits"
                " and lists"
            )
        _bad_result(where, base)
    return _sized(kind, _length(where, kind, meta, "result"), measured=False)


# The one positive whole number a ``pulse.Length`` states.
def _length(where: str, kind: int, meta: tuple[Any, ...], what: str) -> int:
    spelled = _spell_length(kind, meta)
    if len(meta) != 1:
        raise InvalidTypeError(
            f"{where}: {spelled}: state one length, such as pulse.Length[{_LIST_NAME[kind]}, 8]"
        )
    n = meta[0]
    if n.__class__ is int and n > 0:
        return int(n)
    tail = ""
    if n.__class__ is str and what == "result":
        tail = f"; a list result without one takes its length from the body: -> {_LIST_NAME[kind]}"
    elif n.__class__ is str and what == "parameter":
        tail = f"; a list parameter without one takes its length from the call: {_LIST_NAME[kind]}"
    raise InvalidTypeError(
        f"{where}: {spelled}: a length is a positive whole number, got {n!r}{tail}"
    )


_SPELLED = {
    "bits": ("pulse.Bits", "bit"),
    "levels": ("list[int]", "measurement"),
    "iq": ("list[complex]", "IQ point"),
    "f64": ("list[float]", "number"),
    "i32": ("list[int]", "number"),
}


# A list type's ``pulse.Length[...]`` spelling and what it counts
# (``"bit"``, ``"number"``...), for errors.
def _spelled(spec: ValueType) -> tuple[str, str]:
    head = spec.descriptor.split(":")[0].split("[")[0]
    name, unit = _SPELLED.get(head, (spec.descriptor, "value"))
    return f"pulse.Length[{name}, {spec.count}]", unit


# A handler's or body's answer of the wrong length, at run time.
def _wrong_length(function: ClassicalFunction[..., Any], position: int, got: int) -> str:
    spec = function.results[position]
    spelled, unit = _spelled(spec)
    which = f" as result {position}" if len(function.results) > 1 else ""
    return (
        f"{function.name} returned {got} {unit}{'s' if got != 1 else ''}{which}; its result is"
        f" {spelled}"
    )


def _spell_length(kind: int, meta: tuple[Any, ...]) -> str:
    items = ", ".join(f'"{m}"' if m.__class__ is str else repr(m) for m in meta)
    return f"pulse.Length[{_LIST_NAME[kind]}, {items}]"


def _bad_parameter(where: str, name: str, t: Any) -> typing.NoReturn:
    from ..ops import MeasurementResult

    if t is bool:
        raise InvalidTypeError(
            f"{where}: parameter {name!r} is bool; a measured bit is pulse.Bits"
            " (submit [m], read bits[0])"
        )
    if t is MeasurementResult:
        raise InvalidTypeError(
            f"{where}: parameter {name!r} is MeasurementResult; say how to read it: pulse.Bits,"
            " list[int] (levels) or complex (an IQ point)"
        )
    if typing.get_origin(t) in (list, collections.abc.Sequence):
        raise InvalidTypeError(
            f"{where}: parameter {name!r} is {_spell(t)}; lists hold float, int or complex,"
            " and measured bits are pulse.Bits"
        )
    raise InvalidTypeError(
        f"{where}: parameter {name!r} can't be {_spell(t)}; use float, int, complex,"
        " pulse.Bits, list[float], list[int] or list[complex]"
    )


def _bad_result(where: str, t: Any) -> typing.NoReturn:
    if t is complex or typing.get_args(t) == (complex,):
        raise InvalidTypeError(
            f"{where}: can't return {_spell(t)}: a kernel can't use a complex answer;"
            " return tuple[float, float] instead"
        )
    raise InvalidTypeError(
        f"{where}: can't return {_spell(t)}; use float, int, bool, pulse.Bits, list[float],"
        " list[int], a tuple of them, or None"
    )


def _spell(t: Any) -> str:
    if isinstance(t, type) and not typing.get_args(t):
        return t.__name__
    return repr(t).replace("typing.", "")


# ── Classical functions ──────────────────────────────────────────────────────


class _LazyTypes:
    """Stands in for ``args`` or ``results`` until the hints are read.
    Touching it resolves the function, which replaces both attributes with
    tuples, so later reads are plain attribute loads."""

    __slots__ = ("owner", "which")

    def __init__(self, owner: ClassicalFunction[..., Any], which: str) -> None:
        self.owner = owner
        self.which = which

    def _real(self) -> tuple[ValueType, ...]:
        _resolve(self.owner)
        real: tuple[ValueType, ...] = getattr(self.owner, self.which)
        return real

    def __len__(self) -> int:
        return len(self._real())

    def __iter__(self) -> Iterator[ValueType]:
        return iter(self._real())

    def __getitem__(self, i: Any) -> Any:
        return self._real()[i]

    def __eq__(self, other: object) -> bool:
        return self._real() == other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(self._real())


_NO_VERSIONS: Mapping[Any, Any] = types.MappingProxyType({})


# Refuse *name* for a function of the user's when it is the name
# cudaq-pulse gives a version of a CUDA-Q QEC decoder message
# (``get_corrections__3_1``): a program's functions, and the simulation's
# replay of a decoder's calls, are found by name.
def _not_a_message(name: str, what: str, fix: str = "give it another name") -> None:
    from . import _cudaqx

    found = _cudaqx._message_version(name)
    if found is None:
        return
    message, numbers = found
    raise InvalidValueError(
        f"{what}: that is the name cudaq-pulse gives the CUDA-Q QEC decoder message "
        f"{message.name} for decoder {numbers[0]}"
        + (f" and length {numbers[1]}" if len(numbers) > 1 else "")
        + f", and a call of it would be taken for that message; {fix}"
    )


class ClassicalFunction[**P, R]:
    """A classical function a pulse kernel can call while the QPU runs.

    Made by ``@pulse.classical(...)`` or ``@pulse.classical.external(...)``.
    In a kernel, ``.submit(*args)`` sends a request and returns a future
    whose ``.result()`` takes the answer. Outside a kernel, calling it runs
    its reference implementation.

    A function with a list parameter of no stated length is a template: each
    set of list lengths it is called with in a kernel is a version, named
    ``name__<lengths>`` (``decode__4``, ``levels__i3`` for three i32 numbers),
    dispatched by the hash of that name. ``specialize(...)`` gives one version.
    """

    # Set on an instance only when given, so that a function without one
    # holds what it held before budgets (the kernel guard copies it).
    reaction_deadline_ns: float | None = None

    def __init__(
        self,
        implementation: Callable[P, R] | None,
        *,
        name: str | None = None,
        max_latency_ns: float | Time,
        min_latency_ns: float | Time = 0.0,
        nominal_latency_ns: float | Time | None = None,
        latency_mode: str = "bounded",
        timeout_ns: float | Time | None = None,
        submit: str = "eager",
        deadline_ns: float | Time | None = None,
        reaction_deadline_ns: float | Time | None = None,
        link: int | str = 0,
        _signature: Callable[..., Any] | None = None,
        _dispatch_key: int | None = None,
        _key_route: int | None = None,
    ) -> None:
        max_latency_ns = as_nanoseconds(max_latency_ns)
        min_latency_ns = as_nanoseconds(min_latency_ns)
        nominal_latency_ns = as_nanoseconds(nominal_latency_ns)
        timeout_ns = as_nanoseconds(timeout_ns)
        deadline_ns = as_nanoseconds(deadline_ns)
        hinted = implementation if implementation is not None else _signature
        reaction_deadline_ns = as_nanoseconds(reaction_deadline_ns)
        if name is None:
            if hinted is None:
                raise InvalidTypeError("an external classical function needs a name")
            name = hinted.__name__
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise InvalidValueError(
                f"a classical function name must be an identifier, got {name!r}"
            )
        # The functions cudaq-pulse declares itself give their dispatch key.
        if _dispatch_key is None and "__" in name:
            _not_a_message(name, f"a classical function can't be named {name!r}")
        self.name = name
        self.implementation = implementation
        self.args: tuple[ValueType, ...] = _LazyTypes(self, "args")  # type: ignore[assignment]
        self.results: tuple[ValueType, ...] = _LazyTypes(self, "results")  # type: ignore[assignment]
        nominal = max_latency_ns if nominal_latency_ns is None else nominal_latency_ns
        for label, value in (
            ("max_latency_ns", max_latency_ns),
            ("min_latency_ns", min_latency_ns),
            ("nominal_latency_ns", nominal),
        ):
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise InvalidValueError(
                    f"{name}: {label} must be a finite number >= 0, got {value!r}"
                )
        if not min_latency_ns <= nominal <= max_latency_ns:
            raise InvalidValueError(
                f"{name}: latencies must satisfy min <= nominal <= max, got "
                f"{min_latency_ns}, {nominal}, {max_latency_ns}"
            )
        self.min_latency_ns = float(min_latency_ns)
        self.nominal_latency_ns = float(nominal)
        self.max_latency_ns = float(max_latency_ns)
        if latency_mode not in _LATENCY_MODES:
            raise InvalidValueError(
                f"{name}: latency_mode must be one of {', '.join(_LATENCY_MODES)}, "
                f"got {latency_mode!r}"
            )
        if (latency_mode == "timeout") != (timeout_ns is not None):
            raise InvalidValueError(
                f"{name}: timeout_ns goes with latency_mode='timeout' (or pass "
                "timeout to result() at the call)"
            )
        if timeout_ns is not None and not timeout_ns > 0:
            raise InvalidValueError(f"{name}: timeout_ns must be positive, got {timeout_ns!r}")
        self.latency_mode = latency_mode
        self.timeout_ns = None if timeout_ns is None else float(timeout_ns)
        if submit not in _SUBMIT_POLICIES:
            raise InvalidValueError(f"{name}: submit must be 'eager' or 'lazy', got {submit!r}")
        self.submit_policy = submit
        if deadline_ns is not None and not deadline_ns > 0:
            raise InvalidValueError(f"{name}: deadline_ns must be positive, got {deadline_ns!r}")
        self.deadline_ns = None if deadline_ns is None else float(deadline_ns)
        if reaction_deadline_ns is not None and not (
            isinstance(reaction_deadline_ns, (int, float))
            and math.isfinite(reaction_deadline_ns)
            and reaction_deadline_ns > 0
        ):
            raise InvalidValueError(
                f"{name}: reaction_deadline_ns must be a positive number, "
                f"got {reaction_deadline_ns!r}"
            )
        if reaction_deadline_ns is not None:
            self.reaction_deadline_ns = float(reaction_deadline_ns)
        if isinstance(link, str):
            if not link.isidentifier():
                raise InvalidValueError(f"{name}: a link name must be an identifier, got {link!r}")
        elif isinstance(link, bool) or not isinstance(link, int) or link < 0:
            raise InvalidValueError(
                f"{name}: link must be a link name or a whole number >= 0, got {link!r}"
            )
        self.link = link
        # The key it is dispatched by unless its controller fixes one: its
        # compiled name's hash, or a built-in protocol's key.
        self._dispatch_key = dispatch_key(name) if _dispatch_key is None else _dispatch_key
        # For a built-in that shares its key (CUDA-Q QEC's decoder calls): the
        # first argument that picks it out, the decoder id. None for a
        # function with a key of its own.
        self._key_route = _key_route
        # For an on-device random source: how many equally likely values.
        self.uniform: int | None = None
        self.__doc__ = hinted.__doc__ if hinted is not None else None
        self.__wrapped__ = hinted
        # Every attribute is set here, in one order, so all functions share
        # one attribute layout (attributes added later, or in other orders,
        # make every read slower).
        self._error: TypeError | None = None
        # A template's declaration, the plan its submit reads (per parameter
        # a wire type, or the kind of a list of unstated length) and how
        # many lengths its calls give; None, None, 0 for a function of one
        # schema.
        self._shape: _Shape | None = None
        self._plan: tuple[Any, ...] | None = None
        self._n = 0
        self._made: dict[Any, ClassicalFunction[P, R]] | None = None
        # The versions a submit takes without a per-program check: _made.
        self._fast: Mapping[Any, ClassicalFunction[P, R]] = _NO_VERSIONS
        # A version's template and lengths.
        self._template: ClassicalFunction[P, R] | None = None
        self._key: Any = None
        self.base_name = name
        # A template's list-result lengths per version (classical._infer).
        self._length_rules: Any = None

    # A function of one schema given as wire types, for the functions
    # cudaq-pulse declares itself.
    @classmethod
    def _declared(
        cls,
        implementation: Callable[..., Any] | None,
        args: Sequence[ValueType],
        results: Sequence[ValueType],
        **options: Any,
    ) -> ClassicalFunction[..., Any]:
        function: ClassicalFunction[..., Any] = cls(implementation, **options)
        function.args = tuple(args)
        function.results = tuple(results)
        return function

    @property
    def __name__(self) -> str:
        return self.name

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the reference implementation on *args* (outside a kernel)."""
        if self.implementation is None:
            how = ""
            if self.__wrapped__ is not None:
                how = " it is declared with @pulse.classical.external, so"
            raise InvalidTypeError(
                f"{self.base_name} runs on link {self.link!r};{how} it has no body to run here."
                " Give it a handler to simulate it:"
                f' pulse.simulate(..., handlers={{"{self.base_name}": fn}})'
            )
        if kwargs or len(args) != len(self.args):
            if getattr(self.implementation, "_route", None) is not None:
                # A CUDA-Q QEC function: its arguments pick the version.
                return self.implementation(*args, **kwargs)
            # A template has args == (): its calls give the list lengths.
            expected = len(self.args) if self._plan is None else len(self._plan)
            if kwargs or len(args) != expected:
                raise InvalidTypeError(
                    f"{self.name}() takes {expected} positional arguments, got {len(args)}"
                )
        return self.implementation(*args)  # type: ignore[call-arg]

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Pending[R]:
        """Call the function; outside a kernel this runs it at once."""
        if kwargs:
            self(*args, **kwargs)
        return Answer(self(*args))  # type: ignore[call-arg]

    def specialize(self, **lengths: int) -> ClassicalFunction[P, R]:
        """The version of this function for list parameters of *lengths*
        (by parameter name), as a kernel call with lists of those lengths
        makes it. A function of one schema is its own version."""
        _resolve(self)
        if self._shape is None:
            if lengths:
                raise InvalidTypeError(
                    f"{self.name} has one schema; it has no lengths to specialize"
                )
            return self
        names = _unsized_names(self)
        unknown = sorted(set(lengths) - set(names))
        if unknown:
            raise InvalidTypeError(
                f"{self.name}.specialize(): {', '.join(unknown)} isn't a list parameter of"
                " unstated length"
            )
        missing = [n for n in names if n not in lengths]
        if missing:
            raise InvalidTypeError(
                f"{self.name}.specialize(): give lengths for {', '.join(missing)}"
            )
        key: list[int] = []
        for name in names:
            n = lengths[name]
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                raise InvalidTypeError(
                    f"{self.name}.specialize(): {name} must be a positive length"
                )
            key.append(n)
        made = key[0] if self._n == 1 else tuple(key)
        # No kernel is compiling here: a name is refused as a value.
        _version_name(self, made)
        version = _version(self, made)
        return typing.cast("ClassicalFunction[P, R]", version)

    # A declaration, like a function: a copy is the function itself.
    def __copy__(self) -> ClassicalFunction[P, R]:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ClassicalFunction[P, R]:
        return self

    def __reduce_ex__(self, protocol: Any) -> Any:
        from .._pickling import classical_reduce

        return classical_reduce(self, protocol)

    def __repr__(self) -> str:
        try:
            _resolve(self)
        except TypeError:
            return f"<classical function {self.name}, not declared correctly>"
        if self._shape is not None:
            plan = self._shape.plan
            names = self._shape.names
            args = ", ".join(
                f"{_LIST_NAME[p]} {n}" if p.__class__ is int else p.descriptor
                for n, p in zip(names, plan, strict=True)
            )
            results = ", ".join(
                f"{_LIST_NAME[r.kind]}" if r.__class__ is _Rule else r.descriptor
                for r in self._shape.results
            )
        else:
            args = ", ".join(a.descriptor for a in self.args)
            results = ", ".join(r.descriptor for r in self.results)
        return f"<classical function {self.name}({args}) -> ({results}) on link {self.link!r}>"


# Read a declaration's hints, once. Never touch ``__dict__``: that
# changes the attribute layout every read goes through.
def _resolve(function: ClassicalFunction[..., Any]) -> None:
    if function.args.__class__ is not _LazyTypes:  # type: ignore[comparison-overlap]
        return
    if function._error is not None:
        raise function._error
    hinted = function.__wrapped__
    try:
        shape = _read_hints(hinted)
        if not shape.unsized:
            if shape.rules:
                stated = {
                    n: p.count
                    for n, p in zip(shape.names, shape.plan, strict=True)
                    if not p.is_scalar
                }
                function.results = _result_specs(function, shape, stated, "")
            else:
                function.results = shape.results
            function.args = shape.plan
            return
    except TypeError as error:
        function._error = error
        raise
    function.results = () if shape.rules else shape.results
    # A template: args == () sends every kernel submit to the arity check,
    # where the template path starts (see kernel._classical.submit).
    function.args = ()
    function._shape = shape
    function._plan = shape.plan
    function._n = shape.unsized
    function._made = {}
    function._fast = function._made


# A version's results: rules made sized with its input lengths.
def _result_specs(
    function: ClassicalFunction[..., Any],
    shape: _Shape,
    lengths: dict[str, int],
    text: str,
) -> tuple[ValueType, ...]:
    rules = [(i, r) for i, r in enumerate(shape.results) if r.__class__ is _Rule]
    if not rules:
        return shape.results
    out = list(shape.results)
    inferred = [(i, _LIST_NAME[r.kind]) for i, r in rules]
    found: dict[int, int] = {}
    positions = tuple(i for i, _ in inferred)
    native = infer_list_lengths(function.__wrapped__, shape.names, lengths, positions, len(out))
    if native is not None:
        found = dict(zip(positions, native, strict=True))
    elif inferred:
        known = function._length_rules
        if known is None:
            from ._infer import LengthRules

            bits = frozenset(
                n for n, p in zip(shape.names, shape.plan, strict=False) if p is _KIND_BITS
            )
            # An external's stub is a declaration: its placeholder return,
            # such as `return outcomes`, gives the result's length.
            known = function._length_rules = LengthRules(
                function.__wrapped__, shape.names, inferred, len(out), bits
            )
        where = f"{function.name} at {text}" if text else function.name
        try:
            found = dict(zip((i for i, _ in inferred), known.lengths(where, lengths), strict=True))
        except TypeError as error:
            raise InvalidTypeError(f"{error} {_length_advice(function, shape, inferred)}") from None
    for i, r in rules:
        n = found[i]
        if n <= 0:
            where = f"{function.name} at {text}" if text else function.name
            if function.implementation is None:
                raise InvalidTypeError(
                    f"{where}: its {_LIST_NAME[r.kind]} result has no length: the placeholder"
                    f" says only its type. {_length_advice(function, shape, inferred)}"
                )
            raise InvalidTypeError(f"{where}: its {_LIST_NAME[r.kind]} result would be empty")
        out[i] = _sized(r.kind, n, measured=False)
    return tuple(out)


# How to give a list result a length the compiler can see.
def _length_advice(
    function: ClassicalFunction[..., Any], shape: _Shape, inferred: list[tuple[int, str]]
) -> str:
    kind = _LIST_NAME[shape.results[inferred[0][0]].kind]
    fixed = f"give it a fixed size: -> pulse.Length[{kind}, 8]"
    if function.implementation is None:
        plan = zip(shape.names, shape.plan, strict=True)
        inputs = [n for n, p in plan if p.__class__ is int or not p.is_scalar]
        if not inputs:
            return f"An external's body is a placeholder; {fixed}."
        return (
            f"An external's body is a placeholder: return {inputs[0]}, or a slice such as"
            f" {inputs[0]}[:2], for a length the input gives, or {fixed}."
        )
    return f"Write it in plain Python, or {fixed}."


# ── Versions ─────────────────────────────────────────────────────────────────

# __init__'s attribute order.
_COPIED = (
    "name",
    "implementation",
    "args",
    "results",
    "min_latency_ns",
    "nominal_latency_ns",
    "max_latency_ns",
    "latency_mode",
    "timeout_ns",
    "submit_policy",
    "deadline_ns",
    "link",
    "_dispatch_key",
    "_key_route",
    "uniform",
    "__doc__",
    "__wrapped__",
    "_error",
    "_shape",
    "_plan",
    "_n",
    "_made",
    "_fast",
    "_template",
    "_key",
    "base_name",
    "_length_rules",
)


def _unsized_names(template: ClassicalFunction[..., Any]) -> list[str]:
    shape = template._shape
    assert shape is not None
    return [shape.names[i] for i, p in enumerate(shape.plan) if p.__class__ is int]


def _key_items(key: Any) -> tuple[int, ...]:
    return (key,) if key.__class__ is int else tuple(key)


def _lengths_text(template: ClassicalFunction[..., Any], key: Any) -> str:
    return ", ".join(
        f"{name}={abs(n)}"
        for name, n in zip(_unsized_names(template), _key_items(key), strict=True)
    )


# The name of the version of *template* for *key*: the template's and its
# lengths (``decode__4``, ``i`` before the length of a ``list[int]`` of
# numbers), which must not be a decoder message version's.
def _version_name(template: ClassicalFunction[..., Any], key: Any) -> str:
    suffix = "_".join(str(n) if n >= 0 else f"i{-n}" for n in _key_items(key))
    name = f"{template.name}__{suffix}"
    _not_a_message(
        name,
        f"the version of {template.name} for {_lengths_text(template, key)} "
        f"would be named {name!r}",
        f"rename {template.name}",
    )
    return name


# The version of *template* for *key*: per unsized parameter its
# length, negative for a ``list[int]`` of numbers (i32) rather than
# measurements (levels).
def _make(template: ClassicalFunction[..., Any], key: Any) -> ClassicalFunction[..., Any]:
    from ..kernel.ir_builder import CompilationError

    shape = template._shape
    assert shape is not None
    version: ClassicalFunction[..., Any] = object.__new__(ClassicalFunction)
    for attr in _COPIED:
        setattr(version, attr, getattr(template, attr))
    if "reaction_deadline_ns" in template.__dict__:
        version.reaction_deadline_ns = template.reaction_deadline_ns
    version._shape, version._plan, version._n, version._made = None, None, 0, None
    version._fast = _NO_VERSIONS
    items = iter(_key_items(key))
    args = list(shape.plan)
    for i, p in enumerate(args):
        if p.__class__ is int:
            n = next(items)
            args[i] = _sized(p, abs(n), n >= 0)
    version.args = tuple(args)
    text = _lengths_text(template, key)
    if not version.results:  # () only when a result's length is per version
        lengths = {n: p.count for n, p in zip(shape.names, args, strict=True) if not p.is_scalar}
        try:
            version.results = _result_specs(template, shape, lengths, text)
        except TypeError as error:
            raise CompilationError(str(error), code="wrong_argument") from None
    try:
        version.name = _version_name(template, key)
    except InvalidValueError as error:
        raise CompilationError(str(error), code="wrong_argument") from None
    version._dispatch_key = dispatch_key(version.name)
    version._template = template
    version._key = key
    version.base_name = template.name
    return version


def _version(template: ClassicalFunction[..., Any], key: Any) -> ClassicalFunction[..., Any]:
    made = template._made
    assert made is not None
    version = made.get(key)
    if version is None:
        version = made[key] = _make(template, key)
    return version


# The version of *template* for *key*, as an unpickled program names it.
def _unpickled_version(
    template: ClassicalFunction[..., Any], key: Any
) -> ClassicalFunction[..., Any]:
    _resolve(template)
    return _version(template, key)


# A submit whose version isn't made yet: ``builder._versions[template]``
# maps each key to its version, in first use order.
def _enter_slow(builder: Any, template: ClassicalFunction[..., Any], key: Any) -> Any:
    versions = builder._versions
    if versions is None:
        versions = builder._versions = {}
    inner = versions.get(template)
    if inner is None:
        inner = versions[template] = {}
    else:
        entered = inner.get(key)
        if entered is not None:
            return entered
    version = _version(template, key)
    inner[key] = version
    return version


# What answers each function of a program, by name: a handler given
# for its name, else one given for its template's name (serving every
# version), else its body.
def _handlers_for(
    functions: Sequence[ClassicalFunction[..., Any]],
    handlers: Mapping[str, Callable[..., Any]] | None,
) -> dict[str, Callable[..., Any]]:
    chosen: dict[str, Callable[..., Any]] = {
        f.name: f.implementation for f in functions if f.implementation is not None
    }
    for f in functions:
        handler = None
        if handlers:
            handler = handlers.get(f.name) or handlers.get(f.base_name)
        if handler is not None:
            chosen[f.name] = handler
    for name, handler in (handlers or {}).items():
        chosen.setdefault(name, handler)
    return chosen


class Pending[R](Protocol):
    """What ``fn.submit(...)`` returns: a future while a kernel is traced, an
    answer already taken when called outside a kernel.

    ``.result()`` gives the declared results; with ``timeout`` (a time, or a
    number of nanoseconds, of whole program cycles) it gives
    ``(in_time, *results)``.
    """

    @overload
    def result(self, timeout: None = None) -> R: ...
    @overload
    def result[*Ts](self: Pending[tuple[*Ts]], timeout: float | Time) -> tuple[bool, *Ts]: ...
    @overload
    def result(self, timeout: float | Time) -> tuple[bool, R]: ...
    def result(self, timeout: float | Time | None = None) -> Any: ...


class Answer:
    """The answer of a classical function called outside a kernel; its
    ``submit`` types it as a ``Pending``."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def done(self) -> bool:
        return True

    def result(self, timeout: float | Time | None = None) -> Any:
        if timeout is not None:
            if isinstance(self._value, tuple):
                return (True, *self._value)
            return (True, self._value)
        return self._value


class _Options(TypedDict, total=False):
    """The keyword options of ``pulse.classical`` and ``external``."""

    max_latency_ns: Required[float | Time]
    min_latency_ns: float | Time
    nominal_latency_ns: float | Time | None
    latency_mode: str
    timeout_ns: float | Time | None
    submit: str
    deadline_ns: float | Time | None
    reaction_deadline_ns: float | Time | None
    link: int | str
    name: str | None


@overload
def function[**Q, T](
    implementation: Callable[Q, T], /, **options: Unpack[_Options]
) -> ClassicalFunction[Q, T]: ...
@overload
def function[**Q, T](
    implementation: None = None, /, **options: Unpack[_Options]
) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]: ...
def function(implementation: Callable[..., Any] | None = None, /, **options: Any) -> Any:
    """Declare a classical function; ``pulse.classical`` is this decorator.

    The parameters' and result's type hints say what it takes and returns:
    ``float``, ``int`` and ``complex`` (one IQ point) take one value;
    ``pulse.Bits`` (measurements read as bits), ``list[int]`` (measured
    levels, or numbers), ``list[complex]`` (IQ points) and ``list[float]``
    take a list whose length comes from the call. Results are ``float``,
    ``int``, ``bool``, ``pulse.Bits``, ``list[float]``, ``list[int]``, a tuple
    of them or ``None``; a list result's length is inferred from the body.

    ``pulse.Length`` states a fixed length where one wire schema is wanted:
    ``pulse.Length[pulse.Bits, 8]`` takes eight measurements
    read as bits and ``pulse.Length[list[int], 8]`` eight numbers; a result
    ``-> pulse.Length[pulse.Bits, 2]`` answers two bits.

    Keyword options: ``max_latency_ns`` (required), ``min_latency_ns``,
    ``nominal_latency_ns``, ``latency_mode`` (``"bounded"``, ``"timeout"``
    with ``timeout_ns``, or ``"blocking"``), ``submit`` (``"eager"`` or
    ``"lazy"``), ``deadline_ns``, ``reaction_deadline_ns``, ``link`` (the
    link its requests travel on: a name from ``ControlSystem.links``, such as
    ``"controller"``, or a link id; default 0; ``device`` is not used for
    it, because a cudaq-pulse quantum system is the QPU) and ``name``. A latency,
    timeout or deadline is a time (``400 * pulse.ns``) or a number of ns.

    With ``reaction_deadline_ns``, compiling fails unless every call's answer
    is due within that long of the end of the latest readout whose outcome
    the request carries (of the start of the submission when it carries
    none): the time from measuring to being able to act on the answer,
    queueing, link and ``max_latency_ns`` included.

    The hints are read when the function is first used (a program that
    calls it is compiled), and a mistake in them is raised there.
    """
    if implementation is not None:
        return _declare(implementation, options)

    def decorate(fn: Callable[..., Any]) -> ClassicalFunction[..., Any]:
        return _declare(fn, options)

    return decorate


def _declare(fn: Callable[..., Any], options: dict[str, Any]) -> ClassicalFunction[..., Any]:
    if fn.__class__ is not types.FunctionType:
        from . import _cudaqx

        if _cudaqx.is_cudaq_qec(fn):
            return _cudaqx.declare(fn, options)
    if not isinstance(fn, types.FunctionType) or fn.__name__ == "<lambda>":
        raise InvalidTypeError(
            f"@pulse.classical takes a function defined with def, whose type hints say what it"
            f" takes and returns; got {fn!r}"
        )
    return ClassicalFunction(fn, **options)


def external[**Q, T](
    **options: Unpack[_Options],
) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]:
    """Declare a classical function served elsewhere: a declaration whose
    hints give its schema and whose body is a placeholder for its result,
    which never runs::

        @pulse.classical.external(max_latency_ns=50, link="feedback")
        def table(index: pulse.Length[pulse.Bits, 3]) -> float:
            return float()

        @pulse.classical.external(max_latency_ns=900, link="gpu")
        def decode(syndrome: pulse.Bits) -> tuple[float, pulse.Bits]:
            return (float(), syndrome[:2])

    A placeholder is a value of the result's type: ``float()``,
    ``pulse.Bits()`` (its length stated), an input (``return outcomes``) or a
    slice of one, whose length the result takes; a ``None`` result needs no
    return. A list parameter of no stated length takes its length from the
    call, one version per length as for ``pulse.classical``; state the
    length the quantum system takes, ``pulse.Length[pulse.Bits, 3]``, and a call of
    another length is a compile error.

    It takes the keyword options of ``pulse.classical``. Simulating a program
    that calls it needs a handler for it.
    """

    def decorate(stub: Callable[Q, T]) -> ClassicalFunction[Q, T]:
        if not isinstance(stub, types.FunctionType):
            raise InvalidTypeError(
                f"@pulse.classical.external takes a stub defined with def, got {stub!r}"
            )
        declared: ClassicalFunction[Q, T] = ClassicalFunction(None, _signature=stub, **options)
        _resolve(declared)
        return declared

    return decorate


def random_index(
    n: int,
    /,
    *,
    name: str = "random_index",
    max_latency_ns: float | Time = 0.0,
    link: int | str = 0,
    **options: Any,
) -> ClassicalFunction[[], int]:
    """An on-device random source: a classical function of no arguments whose
    answer is an integer drawn uniformly from ``0 .. n - 1``.

    On hardware the controller's random number generator answers it (the
    manifest marks it ``"random"``). Simulation without shots follows every
    value, each with probability 1/n; with shots each shot draws one.
    Emulation draws from its own seeded generator. Use it to pick a
    randomized-benchmarking gate per shot, branching on the index.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise InvalidValueError(f"random_index needs at least 2 values, got {n!r}")
    import random

    def draw() -> int:
        return random.randrange(n)

    function = ClassicalFunction._declared(
        draw,
        (),
        (_I64,),
        name=name,
        max_latency_ns=max_latency_ns,
        link=link,
        **options,
    )
    function.uniform = n
    return function


if TYPE_CHECKING:
    _AnyFunction = ClassicalFunction[..., Any]
    _IndexFunction = ClassicalFunction[[], int]


class _Declare(Protocol):
    @overload
    def __call__[**Q, T](
        self, implementation: Callable[Q, T], /, **options: Unpack[_Options]
    ) -> ClassicalFunction[Q, T]: ...
    @overload
    def __call__[**Q, T](
        self, implementation: None = None, /, **options: Unpack[_Options]
    ) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]: ...


class _External(Protocol):
    def __call__[**Q, T](
        self, **options: Unpack[_Options]
    ) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]: ...


class ClassicalDecorator(_Declare, Protocol):
    """What type checkers take ``pulse.classical`` to be: the decorator, with
    this module's names as attributes.

    A module can't be declared callable, so ``cudaq_pulse`` declares its
    ``classical`` attribute with this type for checkers only; at run time it
    is this module.
    """

    Bits: type[Bits]
    Link: type[Link]
    ClassicalFunction: type[_AnyFunction]
    Pending: type[Pending[Any]]
    function: _Declare
    external: _External
    dispatch_key: Callable[[str], int]
    random_index: Callable[..., _IndexFunction]
    # Submodules, once imported.
    calibration: types.ModuleType
    emulation: types.ModuleType
    loop: types.ModuleType
    wire: types.ModuleType


class _ClassicalModule(types.ModuleType):
    """This module, callable as the ``pulse.classical`` decorator."""

    def __call__(self, implementation: Callable[..., Any] | None = None, /, **options: Any) -> Any:
        return function(implementation, **options)


def __getattr__(name: str) -> Any:
    if name == "Link":
        from ._link import Link

        globals()["Link"] = Link
        return Link
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _ClassicalModule
