# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A compiled kernel for a controller, and what a controller can play.

``ScheduledKernel.export_to_controller()`` (``export`` here) writes a scheduled kernel in
the form an out-of-tree controller backend lowers to its own instructions,
format ``FORMAT`` (``cudaq-pulse.controller/1``): the lines and the qudits
they belong to, and every drive, readout, wait, sync and classical call with
its exact start and end, its waveform (the shape and its parameters, and
optionally its samples), its tone's frequency and frame phase, the
measurement-conditioned branches it lies in, and the segments and calls of
``ScheduledKernel.export_to_emulator()``. The schedule is final: a backend places each
event at its ``start_cycles`` and does not schedule again.

Times count the controller's own clock cycles, at ``clock_ghz``. When the
controller's clock is not the program clock, the export also gives
``program_clock_ghz``, every time is converted to its cycles (exactly for
operations on its lines, which are on its grid, and rounded up for the
others) and every waveform's time parameters are scaled with them. With a
shared reference clock (``ControlSystem.reference_clock_ghz``) it also gives
``reference_clock_ghz``.

The number after the slash is the format's major version. Within one version,
new keys and new event kinds may be added, so a backend ignores keys it does
not know and refuses event kinds it does not know; a change that removes or
changes the meaning of a key gets a new version. ``export(version=2)`` gives
the structured format ``cudaq-pulse.controller/2``: loops whose iterations
repeat stay loops, each distinct waveform is listed once, and what follows
from the rest is left out; ``unroll`` turns it into format /1.

A ``Controller`` states what a controller can play. Given to ``compile()``
with the rest of its control system (``ControlSystem``), it makes
``compile()`` check every scheduled program against it and raise
``ControllerLimitError`` naming each violated limit, so that a program the
controller cannot play fails at compile time. ``check`` returns the
violations without raising.

What a controller computes, for the classical functions it runs itself,
its ``Processor`` states: the cycles each class of operation takes, how
many operations and bytes of state a function may hold, and how many calls
and bytes of requests and answers one shot may use. ``usage`` reports, for
every classical function of a program, what it holds on the processor that
runs it (its operations, read from its body, its state in bytes and its
compute time against its ``max_latency_ns``), each link's calls, slots and
bytes per shot, and the limits they exceed. ``Processor.usage`` does the
same for one function on its own: its cycles always, its time when the
processor has a clock. A processor runs a body with a fixed number of
scalars of state, loops of a trip count known when the program compiles,
no ``while``, recursion, allocation or exceptions.
"""

from __future__ import annotations

import copy
import math
import operator
import os
import sys
import threading
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from ._errors import (
    ExecutionError,
    InvalidTypeError,
    InvalidValueError,
    UnsupportedError,
)
from .kernel.ir_builder import CompilationError

try:
    # hashlib's own blake2b, without hashlib's import of OpenSSL (a millisecond).
    from _blake2 import blake2b as _blake2b
except ImportError:  # pragma: no cover - CPython builds without the _blake2 module
    from hashlib import blake2b as _blake2b

if TYPE_CHECKING:
    from fractions import Fraction

    import numpy as np
    import numpy.typing as npt

    from . import _clocks
    from ._classical_report import ClassicalCall
    from ._controller_cost import ControllerUsage as ControllerUsage
    from ._controller_cost import FunctionCost as FunctionCost
    from ._controller_cost import FunctionUsage as FunctionUsage
    from ._controller_cost import LimitExceeded as LimitExceeded
    from ._controller_cost import LinkUsage as LinkUsage
    from ._controller_cost import ProcessorUsage as ProcessorUsage
    from ._controller_cost import UsageReport as UsageReport
    from ._controller_cost import usage as usage
    from ._device import QuantumSystem
    from ._op_classes import ExpressionTerm as ExpressionTerm
    from ._op_classes import OpClass as OpClass
    from ._op_tables import OpCounts as OpCounts
    from ._op_tables import OpCycles as OpCycles
    from ._system import Controller, ControlSystem
    from ._system import DriveLine as DriveLine
    from ._system import ReadoutLine as ReadoutLine
    from ._system import UpdateBus as UpdateBus
    from .compiler import ScheduledKernel

    # The real and the imaginary parts of a waveform's samples.
    _Samples = tuple[list[float], list[float]]

__all__ = [
    "FORMAT",
    "ControllerLimitError",
    "ControllerUsage",
    "DriveLine",
    "Event",
    "ExpressionTerm",
    "FunctionCost",
    "FunctionUsage",
    "LimitExceeded",
    "LinkUsage",
    "OpClass",
    "OpCounts",
    "OpCycles",
    "ProcessorUsage",
    "ReadoutLine",
    "UpdateBus",
    "UsageReport",
    "WaveformMemoryScope",
    "check",
    "evaluate_expression",
    "export",
    "sample_waveform",
    "shape_key",
    "usage",
    "unroll",
]

FORMAT = "cudaq-pulse.controller/1"
# The structured format, ``export_to_controller(version=2)``.
_STRUCTURED = "cudaq-pulse.controller/2"
# Every format this cudaq-pulse writes.
_FORMATS = (FORMAT, _STRUCTURED)
# The names format /2 gives a link's fields where they differ from the /1
# names; unroll() turns them back.
_LINK_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "fixed_ns": "send_ns",
        "bytes_per_ns": "bandwidth_bytes_per_ns",
        "feedback_ns": "answer_delay_ns",
    }
)


class Event(StrEnum):
    """The kind of an event of a format /2 program: what the controller
    does, named alike for every modality.

    ``PLAY`` plays a waveform on a line and ``READOUT`` plays one and
    acquires a measurement result; ``WAIT`` idles a line, ``SYNC`` aligns
    lines and ``LOOP`` repeats its body. ``CALL`` sends a request to a
    classical function over a link and ``ANSWER`` takes its answer. An event
    in a branch keeps its kind and lists its ``conditions``; a frame's
    phase and frequency are carried by the events that play on it, and
    ``FRAME_UPDATE`` is a phase or frequency change that no later event
    plays with on some path, at its place on the timeline. The export holds
    the plain strings, so ``Event.PLAY == "play"``.
    """

    PLAY = "play"
    READOUT = "readout"
    WAIT = "wait"
    SYNC = "sync"
    LOOP = "loop"
    CALL = "call"
    ANSWER = "answer"
    FRAME_UPDATE = "frame_update"


_EVENT_KINDS = frozenset(Event)
# The /2 kinds of /1 events whose /1 names differ.
_KINDS: Mapping[str, str] = MappingProxyType({"drive": "play", "submit": "call"})
_FLAT_KINDS: Mapping[str, str] = MappingProxyType({v: k for k, v in _KINDS.items()})


# The parameter names of each waveform op, in operand order.
_PARAMETERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "square": ("duration", "amplitude", "amplitude_imag"),
        "gaussian": ("duration", "amplitude", "sigma"),
        "gaussian_square": ("duration", "amplitude", "sigma", "risefall"),
        "drag": ("duration", "amplitude", "sigma", "beta"),
        "cosine": ("duration", "amplitude"),
        "tanh_ramp": ("duration", "amplitude", "sigma"),
        "custom": ("duration",),
        "scale": ("factor",),
    }
)
# The operands an overall amplitude multiplies, per waveform op.
_AMPLITUDES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "square": (1, 2),
        "gaussian": (1,),
        "gaussian_square": (1,),
        "drag": (1,),
        "cosine": (1,),
        "tanh_ramp": (1,),
    }
)
# The names of those operands.
_AMPLITUDE_NAMES: Mapping[str, frozenset[str]] = MappingProxyType(
    {shape: frozenset(_PARAMETERS[shape][i] for i in at) for shape, at in _AMPLITUDES.items()}
)
_PLAYED = frozenset({"drive", "readout", "play"})
# The keys a format /2 waveform has beside its expression.
_IDS = ("id", "shape_id")


class ControllerLimitError(CompilationError):
    """A scheduled program exceeds a limit of the controller it is for.

    ``violations`` has every limit exceeded: a tuple, or past 10,000 of
    them in rolled loops a sequence that builds each one when read, with an
    exact ``len()``. For a control system with processors, ``usage`` is the
    program's ``UsageReport``: what each processor holds against its limits,
    and each limit exceeded with by how much (``usage.exceeded``); the
    message ends with its table.
    """

    def __init__(
        self,
        violations: Sequence[str],
        *,
        system: bool = False,
        usage: UsageReport | None = None,
    ) -> None:
        self.violations: Sequence[str] = tuple(violations) if _listed(violations) else violations
        self.usage = usage
        shown = "\n  ".join(self.violations[:10])
        more = len(self.violations) - 10
        tail = f"\n  ... and {more} more" if more > 0 else ""
        what = "does not fit the control system" if system else "exceeds the controller's limits"
        table = ""
        if usage is not None:
            from ._controller_cost import _format_usage

            table = "\n" + _format_usage(usage, exceeds=False)
        super().__init__(f"the program {what}:\n  {shown}{tail}{table}", code="controller_limit")


def _listed(violations: Sequence[str]) -> bool:
    """Whether *violations* are not the lazy ones of a summarized loop."""
    if isinstance(violations, (list, tuple)):
        return True
    from ._loops import Violations

    return not isinstance(violations, Violations)


def _limit_error(lowered: ScheduledKernel, violations: Sequence[str]) -> ControllerLimitError:
    """The error for the *violations* of *lowered*, with its usage report
    when its control system has processors and they were checked: always
    for one call, and at compile time without float or qudit parameters."""
    system = lowered.options.machine._system
    report = None
    if system is not None and system._placement:
        placed = lowered._role is not None and lowered._role.slots > 0
        if lowered._generic is not None or not (lowered.parameters or placed):
            from ._controller_cost import usage

            report = usage(lowered)
    return ControllerLimitError(violations, system=system is not None, usage=report)


class WaveformMemoryScope(StrEnum):
    """Which lines one waveform memory of a controller serves
    (``Controller.waveform_memory``); the strings ``"per_line"`` and
    ``"shared"`` mean the same."""

    #: Each line has a memory of its own.
    PER_LINE = "per_line"
    #: One memory serves every line, and holds a waveform they share once.
    SHARED = "shared"


# ── Waveforms ────────────────────────────────────────────────────────────────


def _number(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _tree(nodes: Sequence[tuple[str, Sequence[float], Sequence[int]]]) -> dict[str, Any] | None:
    """The waveform expression of an event as nested dictionaries."""
    if not nodes:
        return None

    def build(index: int) -> dict[str, Any]:
        name, numbers, children = nodes[index]
        shape, _, custom = name.partition(":")
        node: dict[str, Any] = {"shape": shape}
        if custom:
            node["name"] = custom
        names = _PARAMETERS.get(shape)
        if shape == "custom_samples":
            node["samples"] = [float(v) for v in numbers]
        elif names is not None and len(names) == len(numbers):
            node["params"] = {k: _number(v) for k, v in zip(names, numbers, strict=True)}
        elif numbers:
            node["values"] = [_number(v) for v in numbers]
        if children:
            node["of"] = [build(child) for child in children]
        return node

    return build(len(nodes) - 1)


def _amplitude(node: Mapping[str, Any] | None) -> complex | None:
    """The overall complex amplitude of a waveform, through scale and neg."""
    if node is None:
        return None
    shape = node["shape"]
    if shape == "scale":
        inner = _amplitude(node["of"][0])
        factor = node.get("params", {}).get("factor")
        return None if inner is None or factor is None else inner * factor
    if shape == "neg":
        inner = _amplitude(node["of"][0])
        return None if inner is None else -inner
    if shape == "custom_samples":
        return complex(max(node["samples"], key=abs)) if node.get("samples") else None
    if shape not in _AMPLITUDES:
        return None
    params = node.get("params")
    if params is None:
        return None
    real = params.get("amplitude")
    imag = params.get("amplitude_imag", 0.0)
    if real is None or imag is None:
        return None
    return complex(real, imag)


def shape_key(node: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    """A key equal for waveforms that differ only in overall amplitude.

    Scale and negation drop out, and so does the amplitude of a built-in
    shape, known or not; custom samples are normalized to their largest
    magnitude. ``None`` when a parameter other than the amplitude is not
    known while the program compiles.
    """
    if node is None:
        return None
    shape = node["shape"]
    if shape in ("scale", "neg"):
        return shape_key(node["of"][0])
    if shape == "custom_samples":
        samples = node.get("samples", [])
        peak = max((abs(v) for v in samples), default=0.0) or 1.0
        return (shape, tuple(round(v / peak, 12) for v in samples))
    params = node.get("params")
    if params is None:
        children = tuple(shape_key(child) for child in node.get("of", ()))
        if any(child is None for child in children):
            return None
        return (shape, tuple(node.get("values", ())), children)
    skip = _AMPLITUDE_NAMES.get(shape, frozenset())
    kept = tuple((k, v) for k, v in params.items() if k not in skip)
    if any(v is None or isinstance(v, dict) for _, v in kept):
        return None
    return (shape, node.get("name"), kept)


def _exact(value: float) -> Fraction:
    """*value* as the decimal it is written as, so that rates such as 2.4
    samples per cycle divide exactly."""
    from fractions import Fraction

    return Fraction(repr(float(value)))


# The terms of the expressions of run-time values; see evaluate_expression.
_ARITHMETIC = frozenset({"add", "sub", "mul", "div", "neg", "max", "min"})
_COMPARISONS: Mapping[str, Any] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}
_INTEGER = frozenset({"floor_div", "rem"})
_CONVERSIONS = frozenset({"to_float", "to_int", "controller_cycles"})
# Every op an exported expression of a run-time value may have (ExpressionTerm).
_EXPRESSION_TERMS = frozenset({*_ARITHMETIC, *_COMPARISONS, *_INTEGER, *_CONVERSIONS, "select"})
_TYPES = {"int": int, "float": float, "bool": bool}
_INT_MIN = -(1 << 63)


def evaluate_expression(
    expression: Any,
    readouts: Mapping[int, int] | None = None,
    answers: Mapping[int, Sequence[float | int | bool]] | None = None,
) -> float | int | bool:
    """The value an exported run-time *expression* takes, as a controller
    must compute it (see the controller backends guide for each term).

    *readouts* gives the level each readout event the expression reads
    observed, and *answers* the results of each answer event, both by the
    event index the expression names. A frame change's value may be a
    number, which is its value. A term that several terms use is in the
    expression's ``terms`` and ``{"term": n}`` where it is used; it is
    computed once. An expression may be any number of terms deep. Raises
    ``ExecutionError`` where the controller must fail the run (a division
    by zero, say), and ``InvalidValueError`` for an input not given or a
    term not of ``ExpressionTerm``.
    """
    value: float | int | bool = _evaluate(expression, readouts or {}, answers or {})
    return value


def _evaluate(
    expression: Any,
    readouts: Mapping[int, int],
    answers: Mapping[int, Sequence[float | int | bool]],
) -> Any:
    if not _is_term(expression):
        return _leaf(expression, readouts, answers)
    terms = expression.get("terms", ())
    known: dict[int, Any] = {}
    # The terms being computed, innermost last: each with its type, the
    # values of its operands so far, its place in ``terms`` if it is one of
    # them, and how many of ``terms`` it may use. Operands are computed in
    # order, as nested calls would.
    computing: list[tuple[Mapping[str, Any], Any, list[Any], int | None, int]] = [
        (expression, _kind(expression), [], None, len(terms))
    ]
    while True:
        node, kind, values, place, earlier = computing[-1]
        of = node["of"]
        if len(values) < len(of):
            operand = of[len(values)]
            if isinstance(operand, Mapping) and "term" in operand:
                n = operand["term"]
                if type(n) is not int or not 0 <= n < earlier:
                    raise InvalidValueError(
                        f"{operand!r} is not an earlier term of the expression's terms"
                    )
                if n in known:
                    values.append(known[n])
                else:
                    computing.append((terms[n], _kind(terms[n], inner=True), [], n, n))
            elif _is_term(operand):
                computing.append((operand, _kind(operand, inner=True), [], None, earlier))
            else:
                values.append(_leaf(operand, readouts, answers))
            continue
        value = _computed(node, kind, values)
        computing.pop()
        if not computing:
            return value
        if place is not None:
            known[place] = value
        computing[-1][2].append(value)


def _with_leaves(expression: Any, leaf: Callable[[Any], Any]) -> Any:
    """A copy of *expression*, and of its ``terms``, with ``leaf(x)`` for
    each ``x`` that is not computed from operands, however deep it is."""
    if not isinstance(expression, dict) or "of" not in expression:
        return leaf(expression)

    def copied(top: dict[str, Any]) -> dict[str, Any]:
        copying: list[tuple[dict[str, Any], list[Any]]] = [(top, [])]
        while True:
            node, of = copying[-1]
            if len(of) < len(node["of"]):
                operand = node["of"][len(of)]
                if isinstance(operand, dict) and "of" in operand:
                    copying.append((operand, []))
                else:
                    of.append(leaf(operand))
                continue
            copying.pop()
            copy = {**node, "of": of}
            if not copying:
                return copy
            copying[-1][1].append(copy)

    root = copied(expression)
    if "terms" in root:
        root["terms"] = [copied(term) for term in root["terms"]]
    return root


def _is_term(expression: Any) -> bool:
    """Whether *expression* is computed from operands: not a number, a
    constant, a readout's level, an answer's result or a use of a term."""
    return isinstance(expression, Mapping) and not (
        "const" in expression
        or "readout" in expression
        or "answer" in expression
        or "term" in expression
    )


def _kind(expression: Mapping[str, Any], inner: bool = False) -> Any:
    if inner and "terms" in expression:
        raise InvalidValueError("only a whole expression has terms, not a term of it")
    op = expression.get("op")
    if op not in _EXPRESSION_TERMS:
        raise InvalidValueError(f"{op!r} is not a term of the expression vocabulary")
    return _TYPES[expression["type"]]


def _leaf(
    expression: Any,
    readouts: Mapping[int, int],
    answers: Mapping[int, Sequence[float | int | bool]],
) -> Any:
    if not isinstance(expression, Mapping):
        if isinstance(expression, (int, float)):
            return expression
        raise InvalidValueError(f"not an expression: {expression!r}")
    if "const" in expression:
        return expression["const"]
    if "readout" in expression:
        index = expression["readout"]
        if index not in readouts:
            raise InvalidValueError(f"no level given for readout event {index}")
        return int(readouts[index])
    if "answer" in expression:
        index, result = expression["answer"], expression["result"]
        if index not in answers or not 0 <= result < len(answers[index]):
            raise InvalidValueError(f"no result {result} given for answer event {index}")
        return answers[index][result]
    raise InvalidValueError(f"{expression!r} is a term of no expression's terms")


def _computed(expression: Mapping[str, Any], kind: Any, values: list[Any]) -> Any:
    """What term *expression* of type *kind* computes of its operands'
    *values*."""
    op = expression["op"]
    if op == "select":
        condition, then, other = values
        return kind(then if condition else other)
    if op == "to_float":
        return float(int(values[0]))
    if op == "to_int":
        x = float(values[0])
        if not math.isfinite(x) or not _INT_MIN <= math.trunc(x) < -_INT_MIN:
            raise _failed(f"to_int of {x!r}, which no 64-bit integer holds")
        return math.trunc(x)
    if op == "controller_cycles":
        cycles = int(values[0])
        if cycles % expression["timestep"]:
            raise _failed(
                f"{cycles} program cycles, not a whole number of the line's timesteps of "
                f"{expression['timestep']}"
            )
        return cycles * expression["num"] // expression["den"]
    values = [kind(v) for v in values]
    if op in _COMPARISONS:
        return _COMPARISONS[op](*values)
    if op == "neg":
        return _integer(-values[0]) if kind is int else -values[0]
    a, b = values
    if op == "max":
        return a if a > b else b
    if op == "min":
        return a if a < b else b
    if kind is float:
        if op == "div":
            if b == 0.0:
                if a != a or a == 0.0:
                    return math.nan
                return math.copysign(math.inf, a) * math.copysign(1.0, b)
            return a / b
        return a + b if op == "add" else a - b if op == "sub" else a * b
    if op in ("add", "sub", "mul"):
        return _integer(a + b if op == "add" else a - b if op == "sub" else a * b)
    if b == 0 or (a == _INT_MIN and b == -1):
        raise _failed(f"{op} of {a} by {b}")
    quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
    if op == "div":
        return quotient
    if op == "floor_div":
        return a // b
    return a - b * quotient


def _integer(value: int) -> int:
    """*value* in 64-bit two's complement, as integer arithmetic wraps."""
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value


def _failed(what: str) -> ExecutionError:
    return ExecutionError(f"the controller fails the run: {what}")


def sample_waveform(
    node: Mapping[str, Any], duration_cycles: int, samples_per_cycle: float = 1
) -> npt.NDArray[np.complex128]:
    """The complex envelope of a waveform, ``samples_per_cycle`` per cycle.

    Sample ``k`` is the envelope at ``k / samples_per_cycle`` cycles from the
    start, with the definitions of the waveform ops (times in cycles), the
    same the simulator plays. ``samples_per_cycle`` may be fractional, but
    the waveform must then last a whole number of samples. Raises
    ``ValueError`` when it does not, or for a waveform the export cannot
    evaluate: a ``pulse.custom`` op of hand-written IR, whose samples the
    runtime computes, or one whose parameters are not numbers.
    """
    import numpy as np

    real, imag = _sample(node, duration_cycles, _exact(samples_per_cycle))
    samples = np.empty(len(real), complex)
    samples.real = real
    samples.imag = imag
    return samples


def _sample(node: Mapping[str, Any], duration_cycles: int, ratio: Fraction | int) -> _Samples:
    """``sample_waveform``'s samples at *ratio* samples per cycle, as the
    lists of floats the export writes: the simulator's own envelopes,
    sampled natively, without NumPy."""
    # Absolute, so that the import costs no relative name's resolution.
    from cudaq_pulse._native._cudaq_pulse_native import sample_envelope

    # The kinds of the simulator's waveform callbacks (WaveformCallback.h).
    kinds = {
        "gaussian": 2,
        "drag": 3,
        "cosine": 4,
        "tanh_ramp": 5,
        "gaussian_square": 6,
        "custom_samples": 7,
    }
    per, over = ratio.numerator, ratio.denominator
    if int(duration_cycles) * per % over:
        raise InvalidValueError(
            f"{duration_cycles} cycles at {per / over:g} samples per cycle is "
            f"{int(duration_cycles) * per / over:g} samples, not a whole number"
        )
    count = int(duration_cycles) * per // over
    rate = per if over == 1 else per / over

    def evaluate(node: Mapping[str, Any]) -> _Samples:
        shape = node["shape"]
        children = [evaluate(child) for child in node.get("of", ())]
        params = node.get("params", {})
        if shape == "scale" and isinstance(params["factor"], (int, float)):
            (re, im), factor = children[0], params["factor"]
            return [x * factor for x in re], [y * factor for y in im]
        if shape == "neg":
            re, im = children[0]
            return [-x for x in re], [-y for y in im]
        if shape in ("add", "sub"):
            (are, aim), (bre, bim) = children
            sign = 1.0 if shape == "add" else -1.0
            return [x + sign * y for x, y in zip(are, bre, strict=True)], [
                x + sign * y for x, y in zip(aim, bim, strict=True)
            ]
        if shape == "mul":
            (are, aim), (bre, bim) = children
            return (
                [ar * br - ai * bi for ar, ai, br, bi in zip(are, aim, bre, bim, strict=True)],
                [ar * bi + ai * br for ar, ai, br, bi in zip(are, aim, bre, bim, strict=True)],
            )
        # The callback's parameters: start, duration, amplitude (real and
        # imaginary), sigma, beta, risefall, then the phase, quadrature and
        # rotation an envelope leaves at 0, then any samples.
        if shape == "custom_samples":
            held = (0.0, max(duration_cycles, 1), *[0.0] * 8, *node["samples"])
            return sample_envelope(kinds[shape], held, count, rate)
        if shape not in _AMPLITUDES or any(
            not isinstance(v, (int, float)) for v in params.values()
        ):
            raise ValueError(f"cannot sample a {shape!r} waveform without numeric parameters")
        if shape == "square":
            return [float(params["amplitude"])] * count, [float(params["amplitude_imag"])] * count
        get = params.get
        shaped = (get("sigma", 0.0), get("beta", 0.0), get("risefall", 0.0))
        envelope = (0.0, get("duration"), get("amplitude"), 0.0, *shaped, 0.0, 0.0, 0.0)
        return sample_envelope(kinds[shape], envelope, count, rate)

    return evaluate(node)


# ── Export ───────────────────────────────────────────────────────────────────


class _Waveforms:
    """The distinct waveforms of a format /2 program, in first use order."""

    def __init__(self) -> None:
        self.table: list[dict[str, Any] | None] = []
        self._index: dict[str, int] = {}

    def __call__(
        self, nodes: Sequence[tuple[str, Sequence[float], Sequence[int]]]
    ) -> tuple[int, None]:
        tree = _tree(nodes)
        key = repr(tree)
        index = self._index.get(key)
        if index is None:
            index = self._index[key] = len(self.table)
            self.table.append(tree)
        return index, None

    def inline(self, events: Sequence[dict[str, Any]]) -> None:
        """Give each event of *events* with ``runtime`` a waveform with
        those operands in place, keeping only the waveforms events play."""
        table: list[dict[str, Any] | None] = []
        index: dict[str, int] = {}
        for event in _played(events):
            tree = self.table[event["waveform"]]
            runtime = event.pop("runtime", None)
            if runtime is not None and tree is not None:
                tree = _with_expressions(tree, runtime)
            key = repr(tree)
            if key not in index:
                index[key] = len(table)
                table.append(tree)
            event["waveform"] = index[key]
        self.table, self._index = table, index

    def identified(self) -> list[dict[str, Any] | None]:
        """The table, each waveform with its ``id`` and ``shape_id``."""
        out: list[dict[str, Any] | None] = []
        for key, tree in zip(self._index, self.table, strict=True):
            if tree is None:
                out.append(None)
                continue
            entry = {"id": _digest(key), **tree}
            shape = shape_key(tree)
            if shape is not None:
                entry["shape_id"] = _digest(repr(shape))
            out.append(entry)
        return out


def _digest(text: str) -> str:
    """A content id: 16 hex digits of a hash of *text*."""
    return _blake2b(text.encode(), digest_size=8).hexdigest()


def _walk(
    lowered: ScheduledKernel,
    values: bool = True,
    own: _clocks.ControllerClock | None = None,
    waveforms: bool = True,
    slots: list[int] | None = None,
    summary: tuple[list[tuple[int, ...]], list[tuple[int, int, int]] | None] | None = None,
    segments: list[Any] | None = None,
) -> tuple[
    list[Any],
    list[dict[str, Any]],
    dict[str, Any],
    Sequence[tuple[int, int, int]],
    list[str],
]:
    """The lines, events, and frames of *lowered* (``_with_frames``), at
    the values of its call; without *values*, a module with float parameters
    has ``None`` where they reach, and ``{"unknown": ...}`` for a value it
    computes at run time that the export has no term for.

    The lines are the dictionaries of format /1. With *summary*, (repeats,
    periods), for a check, the events are those the check walk gives: in a
    module that computes nothing at run time, a long rolled loop whose
    iterations repeat has its events once, and the list *repeats* gets its
    copies (``_loops``); with a list *periods* too, a loop whose lines move
    by different periods as well, and *periods* gets their (repeat, line,
    period). With *own* too, only loops whose periods are whole timesteps of
    its grid, their periods in its cycles.

    With *own*, the events' times are in that controller's cycles (their
    waveforms' too, with *waveforms*), what they compute at run time too
    (``_clocks.own_run_time``), and the line events off its grid are
    returned as (index, start, duration) in program cycles, with the
    violations of what they compute at run time. *slots* gets the
    runtime-qudit slot of each line (-1 for none)."""
    if values:
        if lowered.parameters:
            raise ValueError(
                "the module has float parameters "
                f"({', '.join(lowered.parameters)}); lower it with their values first"
            )
        if lowered._role is not None and lowered._unbound():
            raise InvalidValueError("the module has runtime qudits; lower it with handles first")
    # Waveforms and frames depend on the call's values, which only the
    # concrete module holds.
    module = lowered._concrete()
    off_grid: Sequence[tuple[int, int, int]] = ()
    problems: list[str] = []
    if summary is not None:
        repeats, periods = summary
        with lowered._lock:
            if own is None:
                lines, events = module.controller_events(
                    _PARAMETERS, _AMPLITUDES, slots, repeats, periods, segments
                )
            else:
                ratio = own.ratio
                lines, events, off_grid, ends = module.controller_events_on(
                    _PARAMETERS,
                    _AMPLITUDES,
                    ratio.numerator,
                    ratio.denominator,
                    own.grid,
                    slots,
                    repeats,
                    periods,
                )
                for index, end in ends:
                    events[index]["end_cycles"] = end
        frames: dict[str, Any] = {}
    else:
        # A check (without *values*) needs no value the export cannot name.
        with lowered._lock:
            if own is None:
                lines, events, frames = module.controller_program(
                    _PARAMETERS,
                    _AMPLITUDES,
                    lowered.classical != (),
                    slots=slots,
                    exportable=values,
                )
            else:
                ratio = own.ratio
                lines, events, frames, off_grid = module.controller_program(
                    _PARAMETERS,
                    _AMPLITUDES,
                    lowered.classical != (),
                    own=(ratio.numerator, ratio.denominator, own.grid),
                    slots=slots,
                    exportable=values,
                )
    if own is not None:
        from . import _clocks

        if waveforms and ratio != 1:
            # The events of one op share its waveform's parameters.
            factor = float(ratio)
            scaled: set[int] = set()
            for event in _played(events):
                tree = event["waveform"]
                if tree is not None:
                    _clocks._scale(tree, factor, scaled)
        if _computes_at_run_time(lowered):
            from . import _clocks

            for event in events:
                if event["kind"] in _clocks._GRIDDED:
                    problems.extend(
                        _clocks.own_run_time(own, lines[event["line"]], event, waveforms)
                    )
    fine = (lowered._generic or lowered).fine_delays
    if fine is not None:
        for line in lines:
            line["fine_delay_ns"] = fine.get((line["role"], line["qudit"]), 0.0)
    return lines, events, frames, off_grid, problems


def _walk_structured(
    lowered: ScheduledKernel, table: _Waveforms, slots: list[int] | None = None
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any], tuple[()], list[str]]:
    """``_walk`` for format /2, on the program clock: the lines as (role,
    qudit), loops whose iterations repeat stay loops, and events give their
    waveform's index in *table*. *slots* gets each line's runtime-qudit
    slot (-1 for none)."""
    if lowered.parameters:
        raise InvalidValueError(
            "the module has float parameters "
            f"({', '.join(lowered.parameters)}); lower it with their values first"
        )
    if lowered._role is not None and lowered._unbound():
        raise InvalidValueError("the module has runtime qudits; lower it with handles first")
    module = lowered._concrete()
    with lowered._lock:
        lines, events, frames = module.controller_program(
            _PARAMETERS,
            _AMPLITUDES,
            lowered.classical != (),
            True,
            table,
            slots=slots,
        )
    if any("runtime" in event for event in _played(events)):
        table.inline(events)
    return lines, events, frames, (), []


def _computes_at_run_time(lowered: ScheduledKernel) -> bool:
    """Whether the program computes values at run time: from the answers of
    classical calls, or values a readout selects while it runs."""
    return bool(lowered.classical) or lowered.runtime_selects


def _native_clock(clocks: _clocks.ClockPlan | None) -> _clocks.ControllerClock | None:
    """The controller whose cycles the native walk counts the events in, when
    the program has one and it is not on the program clock or times in
    steps of more than one cycle."""
    if clocks is None or clocks.owners is not None:
        return None
    own = clocks.controllers[0]
    return None if own.ratio == 1 and own.grid == 1 else own


def _one_clock(clocks: _clocks.ClockPlan | None) -> bool:
    """Whether the program's lines all count program cycles one at a time."""
    return clocks is None or (clocks.owners is None and _native_clock(clocks) is None)


def _unknown_kinds(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """The kinds of *events*, in loops too, that format /2 does not have,
    sorted."""
    found: set[str] = set()
    stack = [events]
    while stack:
        for event in stack.pop():
            if event["kind"] == "loop":
                stack.append(event["body"])
            elif event["kind"] not in _EVENT_KINDS:
                found.add(str(event["kind"]))
    return sorted(found)


def _played(events: Sequence[dict[str, Any]]) -> Any:
    """The drives and readouts of *events*, in the bodies of loops too."""
    for event in events:
        if event["kind"] == "loop":
            yield from _played(event["body"])
        elif event["kind"] in _PLAYED:
            yield event


def export(
    lowered: ScheduledKernel,
    sample_rate_ghz: float | None = None,
    controller: int | None = None,
    version: int = 1,
) -> dict[str, Any]:
    """A scheduled kernel for a controller backend, as JSON-serializable data.

    With *sample_rate_ghz*, each drive and readout also carries its
    envelope's samples at that rate (``samples``: real and imaginary parts),
    which must be a whole multiple of the clock. *version* 2 gives format
    ``cudaq-pulse.controller/2``: a loop whose iterations repeat is one
    ``loop`` event holding its body, each distinct waveform is listed once,
    and what follows from the rest is left out (``unroll`` gives format /1
    back). Raises ``ControllerLimitError`` when the limits of its
    controller are exceeded. On a control system of several
    controllers, the program of the one *controller* indexes (see
    ``_export_part``); without it, a program on more than one is a
    ``ValueError``. Without runtime qudits, each line has the frequency of
    its own tone (``_native_frequencies``).
    """
    program = _export(lowered, sample_rate_ghz, controller, version)
    if lowered._role is None or lowered._role.slots == 0:
        _native_frequencies(program, lowered.quantum_system)
    return program


def _native_frequencies(program: dict[str, Any], quantum_system: Any) -> None:
    """Give each line of *program* the frequency of its own tone, at which
    every shot starts its frame: its qudit's 0-1 frequency on a drive line,
    its readout's frequency (else the 0-1 frequency) on a readout line, as
    the compiled module records them; in place. Format /1 gives its qudit's
    (``native_frequency_hz``), format /2 one for each of its ``qudits``
    (``native_frequencies_hz``)."""
    lines = program["lines"]
    structured = program["format"] == _STRUCTURED
    reached = [q for line in lines for q in (line["qudits"] if structured else (line["qudit"],))]
    if not reached:
        return
    _, drive, readout = quantum_system._frequency_tables(max(reached) + 1)
    for line in lines:
        table = drive if line["role"] == "drive" else readout
        if structured:
            line["native_frequencies_hz"] = [table[q] for q in line["qudits"]]
        else:
            line["native_frequency_hz"] = table[line["qudit"]]


def _export(
    lowered: ScheduledKernel,
    sample_rate_ghz: float | None,
    controller: int | None,
    version: int,
) -> dict[str, Any]:
    """``export`` without the lines' native frequencies."""
    from .classical.wire import _export as emulation

    machine = lowered.options.machine
    limits = machine.controller
    system = machine._system
    role = lowered._role
    if version != 1:
        if version != 2:
            raise InvalidValueError(f"version must be 1 or 2, got {version!r}")
        if system is not None:
            if len(system.controllers) > 1:
                raise UnsupportedError(
                    "format /2 is written for one controller; export version 1 for a control "
                    "system of several"
                )
            if controller is not None:
                _controller_index(controller, 1)
            program = _export_structured(lowered, sample_rate_ghz, emulation, role)
            if not system._shares:
                return program
            return system._shares.structured(program, system)  # type: ignore[no-any-return]
        if controller is not None:
            _controller_index(controller, 1)
        return _export_structured(lowered, sample_rate_ghz, emulation, role)
    plan: _ExportPlan | None = None
    slots: list[int] | None = None
    if system is not None or role is not None or controller is not None:
        if system is not None and limits is None:
            return _export_part(lowered, sample_rate_ghz, controller)
        if controller is not None:
            _controller_index(controller, 1)
        if role is not None:
            plan = role.export_plan
            if plan is None:
                slots = []
    clocks = (lowered._generic or lowered).clocks
    native = None if clocks is None else _native_clock(clocks)
    # In the controller's own cycles from here on.
    lines, events, frames, off, problems = _walk(lowered, True, native, True, slots)
    if system is not None:
        if system._one_box:
            assert limits is not None
            _check_at_export(lowered, limits, clocks, (lines, events, frames, off, problems))
        else:
            violations = system._violations(
                lowered, lines, events, clocks, off, problems, True, None, None
            )
            if violations:
                raise _limit_error(lowered, violations)
        if system._shares:
            tone_slots = system._shares.flat(system, lines, events)
    elif limits is not None:
        _check_at_export(lowered, limits, clocks, (lines, events, frames, off, problems))
    if clocks is None:
        clock = lowered.quantum_system.clock_ghz
    else:
        from . import _clocks

        clock = clocks.controllers[0].clock_ghz
    if sample_rate_ghz is not None:
        _add_samples(events, sample_rate_ghz, clock)
    base = emulation(lowered, False)
    program = {
        "format": FORMAT,
        "quantum_system": lowered.quantum_system.name,
        "clock_ghz": clock,
        **({} if clocks is None else _clocks.clock_keys(clocks, clocks.controllers[0])),
        "schedule": lowered.options.schedule,
        "sample_rate_ghz": sample_rate_ghz,
        "qudits": list(lowered.qudits),
        "lines": lines,
        "events": events,
        "segments": base["segments"],
        "calls": base["calls"],
        "links": base["links"],
        "manifest": base["manifest"],
        "limits": None if limits is None else limits.as_dict(),
    }
    if role is not None:
        if plan is None or len(plan[1]) != len(events):
            if slots is None:
                slots = []
                lines = _walk(lowered, own=native, slots=slots)[0]
            plan = _export_plan(lowered, role, slots, lines, events)
        program["program_id"] = role.program_id
        program["symbols"] = _symbols(lowered, plan[0])
        for event, template in zip(events, plan[1], strict=True):
            if template is not None:
                symbols = template[0].copy()
                for key in template[1]:
                    symbols[key] = symbols[key][:]
                event["symbols"] = symbols
    if frames:
        program = _with_frames(program, frames, _slot_names(events, role))
        if system is not None and system._shares:
            system._shares.framed(program, tone_slots)
    return program


def _with_frames(
    program: dict[str, Any], frames: Mapping[str, Any], slots: Mapping[int, str]
) -> dict[str, Any]:
    """*program* with what the native *frames* hold right after its
    ``events``: its ``frame_state``, then its ``runtime_tones``, of the tones
    on its ``lines`` (``_runtime_tones``; *slots* names a runtime qudit's)."""
    added = []
    if "frame_state" in frames:
        added.append(("frame_state", frames["frame_state"]))
    if "runtime_tones" in frames:
        table = _runtime_tones(frames["runtime_tones"], program["lines"], slots)
        if table:
            added.append(("runtime_tones", table))
    items = list(program.items())
    at = list(program).index("events") + 1
    return dict([*items[:at], *added, *items[at:]])


def _runtime_tones(
    found: Sequence[tuple[int, float | None, float | None]],
    lines: Sequence[Mapping[str, Any]],
    slots: Mapping[int, str],
) -> dict[str, dict[str, Any]]:
    """The ``runtime_tones`` of an export: each tone tracked at run time
    that *found* gives, on one of *lines* or on none, by its id, with its
    line and the frame every shot starts it in. On the line of a runtime
    qudit (its slot in *slots*, by line) the tone is the line's own, and its
    reset frequency is the symbol of the slot's native frequency."""
    roles = {line["line"]: line["role"] for line in lines}
    table = {}
    for tone, (line, frequency, phase) in enumerate(found):
        if line >= 0 and line not in roles:
            continue
        entry: dict[str, Any] = {
            "line": None if line < 0 else line,
            "reset_frequency_hz": frequency,
            "reset_phase": phase,
        }
        slot = slots.get(line)
        if slot is not None:
            native = "f01" if roles[line] == "drive" else "f_ro"
            entry["symbols"] = {"qudit": slot, "reset_frequency_hz": [f"{slot}.{native}"]}
        table[str(tone)] = entry
    return table


def _slot_names(events: Sequence[Mapping[str, Any]], role: Any) -> dict[int, str]:
    """The slot of each runtime qudit's line, as format /1 *events* name it."""
    if role is None:
        return {}
    return {
        event["line"]: event["symbols"]["qudit"]
        for event in events
        if "qudit" in event.get("symbols", ())
    }


def _check_at_export(
    lowered: ScheduledKernel,
    limits: Controller,
    clocks: Any,
    walked: tuple[Any, Any, Any, Any, Any] | None,
) -> None:
    """Raises ``ControllerLimitError`` for the limits an export exceeds.

    On one clock, compile() checked the module it scheduled, and a call
    whose values some limit depends on was checked when it bound them; on
    another, the events in the controller's cycles are checked here, from
    the format /1 *walked* (``None``: walk them).
    """
    if not _one_clock(clocks) or (not lowered.scheduled and not lowered.arguments):
        violations = _limit_violations(lowered, limits, clocks, walked)
        if violations:
            raise _limit_error(lowered, violations)


def _export_structured(
    lowered: ScheduledKernel, sample_rate_ghz: float | None, emulation: Any, role: Any = None
) -> dict[str, Any]:
    """``export`` in format ``cudaq-pulse.controller/2``; with the *role* of
    a module of list or qudit parameters, its symbols as format /1 gives
    them."""
    from . import _clocks

    waveforms = _Waveforms()
    limits = lowered.options.machine.controller
    clocks = (lowered._generic or lowered).clocks
    on_program_clock = clocks is None or (
        clocks.owners is None and clocks.controllers[0].ratio == 1
    )
    if not on_program_clock or (lowered._generic or lowered).fine_delays is not None:
        raise UnsupportedError(
            "format /2 is written for a controller on the program clock, without line "
            "delays; export version 1 for this machine"
        )
    slots: list[int] | None = None if role is None else []
    raw_lines, events, frames, _, _ = _walk_structured(lowered, waveforms, slots)
    if limits is not None:
        _check_at_export(lowered, limits, clocks, None)
    own = None if clocks is None else clocks.controllers[0]
    clock = lowered.quantum_system.clock_ghz if own is None else own.clock_ghz
    samples_of: list[dict[str, list[float]] | None] = []
    # The controller's converter rate, else sample_rate_ghz, else one sample
    # per cycle.
    own_rate = None if limits is None else limits.sample_rate_ghz
    rate = clock if own_rate is None else own_rate
    if sample_rate_ghz is not None:
        if not sample_rate_ghz > 0:
            raise InvalidValueError(f"sample_rate_ghz must be positive, got {sample_rate_ghz}")
        if own_rate is not None and sample_rate_ghz != own_rate:
            raise InvalidValueError(
                f"sample_rate_ghz={sample_rate_ghz}; the controller samples at {own_rate} GHz"
            )
        from fractions import Fraction

        ratio = Fraction(*_clocks.made_for(sample_rate_ghz, clock))
        rate = sample_rate_ghz
        samples_of = _add_samples(events, sample_rate_ghz, clock, waveforms, ratio)
    base = emulation(lowered, False)
    program: dict[str, Any] = {
        "format": _STRUCTURED,
        "quantum_system": lowered.quantum_system.name,
        "clock_ghz": clock,
    }
    if clocks is not None:
        program.update(_clocks.clock_keys(clocks, clocks.controllers[0]))
    program["sample_rate_ghz"] = rate
    program["schedule"] = lowered.options.schedule
    program["qudits"] = list(lowered.qudits)
    # A list, so that a line may later address many qudits.
    program["lines"] = [
        {"line": index, "role": role, "qudits": [qudit]}
        for index, (role, qudit) in enumerate(raw_lines)
    ]
    if clocks is not None and clocks.controllers[0].grid > 1 and _computes_at_run_time(lowered):
        _on_timesteps(clocks.controllers[0], program["lines"], events)
    program["waveforms"] = waveforms.identified()
    if sample_rate_ghz is not None:
        program["samples"] = samples_of
    generic = lowered._generic
    if generic is not None and generic.parameters:
        with lowered._lock:
            roles = generic._module.parameter_roles()
        program["parameters"] = {
            name: {"value": value, "reaches": reaches}
            for name, value, reaches in zip(
                generic.parameters, lowered.arguments, roles, strict=True
            )
        }
    program["events"] = events
    if frames:
        names = {} if slots is None else {i: f"s{k}" for i, k in enumerate(slots) if k >= 0}
        program = _with_frames(program, frames, names)
    program["segments"] = base["segments"]
    program["calls"] = base["calls"]
    program["links"] = base["links"]
    program["manifest"] = base["manifest"]
    program["controller"] = None if limits is None else _structured_controller(limits)
    if role is not None:
        assert slots is not None
        program["program_id"] = role.program_id
        program["symbols"] = _symbols(lowered, _symbol_plan(role, slots, program["lines"]))
        _structured_symbols(lowered, role, events, slots)
    program["links"] = {
        k: {_LINK_KEYS.get(f, f): v for f, v in link.items()}
        for k, link in program["links"].items()
    }
    program["programs"] = {}
    program["bindings"] = {}
    return program


def _on_timesteps(
    own: _clocks.ControllerClock, lines: Sequence[Mapping[str, Any]], events: list[dict[str, Any]]
) -> None:
    """Give each wait of format /2 *events*, in loops too, whose length is
    computed at run time the length format /1 gives it on *own*'s
    timesteps, in place; the check at export has found any it can decide
    is off them."""
    from . import _clocks

    for event in events:
        if event["kind"] == "loop":
            _on_timesteps(own, lines, event["body"])
        elif event["kind"] == "wait" and "length" in event:
            _clocks.own_run_time(own, lines[event["line"]], event, False)


def _structured_controller(limits: Controller) -> dict[str, Any]:
    """The controller's fields as format /2 writes them, its ``processors``
    always listed."""
    data = limits.as_dict()
    data.setdefault("processors", [])
    return data


def _flat_controller(data: Mapping[str, Any]) -> dict[str, Any]:
    """A format /2 controller as format /1 writes it, its ``processors``
    listed only when it has some."""
    return {k: v for k, v in data.items() if k != "processors" or v}


def _with_expressions(
    tree: Mapping[str, Any], runtime: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A copy of waveform *tree* with the operands known at run time
    (``path`` to a node, its ``params`` by name) as expressions in its
    parameters; one the node has no parameter for stays in its ``runtime``."""
    out = copy.deepcopy(dict(tree))
    for item in runtime:
        node = out
        for step in item["path"]:
            node = node["of"][step]
        params = node.get("params", {})
        for name, expression in item["params"].items():
            if name in params:
                params[name] = expression
            else:
                node.setdefault("runtime", {})[name] = expression
    return out


def _run_time_operands(tree: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The operands of format /2 waveform *tree* that are expressions, as
    (``path``, ``params``) items in pre-order."""
    out: list[dict[str, Any]] = []

    def visit(node: Mapping[str, Any], path: list[int]) -> None:
        params = {k: v for k, v in node.get("params", {}).items() if isinstance(v, dict)}
        params.update(node.get("runtime", {}))
        if params:
            out.append({"path": list(path), "params": params})
        for i, child in enumerate(node.get("of", ())):
            visit(child, [*path, i])

    if tree is not None:
        visit(tree, [])
    return out


def _with_runtime(
    tree: Mapping[str, Any], runtime: Sequence[Mapping[str, Any]], resolve: Any = None
) -> dict[str, Any]:
    """A copy of waveform *tree* in format /1: each operand of *runtime*
    (paths to nodes, their operands by name) ``None`` in its node's
    parameters and, with *resolve*, resolved in the node's ``runtime``."""
    out = copy.deepcopy(dict(tree))
    for item in runtime:
        node = out
        for step in item["path"]:
            node = node["of"][step]
        node.pop("runtime", None)
        params = node.get("params", {})
        for name in item["params"]:
            if name in params:
                params[name] = None
        if resolve is not None:
            node["runtime"] = {k: resolve(v) for k, v in item["params"].items()}
    return out


def unroll(program: Mapping[str, Any]) -> dict[str, Any]:
    """A format /2 export in format /1: every event once per loop iteration.

    A program already in format /1 is returned as it is. Times, iterations
    and references are those of the /1 export of the same kernel; a phase
    is its first iteration's plus its steps, equal up to rounding; branches
    are numbered in the order they first appear.
    """
    if program["format"] == FORMAT:
        return dict(program)
    if program["format"] != _STRUCTURED:
        raise InvalidValueError(f"cannot unroll format {program['format']!r}")
    flat_lines, handle_of, handles = _handles(program)
    qudit_of = {line["line"]: line["qudits"][0] for line in program["lines"]}
    waveforms = [
        None if entry is None else {k: v for k, v in entry.items() if k not in _IDS}
        for entry in program["waveforms"]
    ]
    run_time = [_run_time_operands(tree) for tree in waveforms]
    waveforms = [
        _with_runtime(tree, operands) if tree is not None and operands else tree
        for tree, operands in zip(waveforms, run_time, strict=True)
    ]
    amplitudes = [_amplitude(tree) for tree in waveforms]
    samples = program.get("samples")
    # Every event of /2, loops included, by its place in pre-order: the
    # index references give.
    place: dict[int, int] = {}

    def number(items: Sequence[Mapping[str, Any]]) -> None:
        for item in items:
            place[id(item)] = len(place)
            if item["kind"] == "loop":
                number(item["body"])

    number(program["events"])
    unknown = _unknown_kinds(program["events"])
    if unknown:
        raise InvalidValueError(f"cannot unroll events of kind {', '.join(map(repr, unknown))}")
    events: list[dict[str, Any]] = []
    latest: dict[int, int] = {}
    branches: dict[tuple[Any, ...], int] = {}

    def place_of(leaf: Any) -> Any:
        if not isinstance(leaf, dict):
            return leaf
        if "readout" in leaf:
            return {"readout": latest[leaf["readout"]]}
        if "answer" in leaf:
            return {"answer": latest[leaf["answer"]], "result": leaf["result"]}
        return dict(leaf)

    def resolve(expression: Any) -> Any:
        """A run-time expression with its /2 places as /1 indices."""
        return _with_leaves(expression, place_of)

    def frame_updates(
        updates: Sequence[Mapping[str, Any]], branch: Any, shift: int = 0
    ) -> list[dict[str, Any]]:
        """Frame changes with their /2 places as /1 indices, made *shift*
        cycles after their place in the loops' first iterations."""
        return [
            {
                **({"tone": update["tone"]} if "tone" in update else {}),
                "conditions": [
                    {
                        "answers": [latest[r] for r in c.get("answers", ())],
                        "arm": c["arm"],
                        "branch": branch(c),
                        **({"expression": resolve(c["expression"])} if "expression" in c else {}),
                        "readouts": [latest[r] for r in c.get("readouts", ())],
                        **({"result": c["result"]} if "result" in c else {}),
                        "test": c["test"],
                    }
                    for c in update.get("conditions", ())
                ],
                "id": update["id"],
                "op": update["op"],
                "start_cycles": (
                    None if update["start_cycles"] is None else update["start_cycles"] + shift
                ),
                "value": resolve(update["value"]),
            }
            for update in updates
        ]

    def expand(items: Sequence[Mapping[str, Any]], loops: list[tuple[Any, ...]]) -> None:
        # loops: (depth, iteration, iterations since the first, period, the
        # branches around the loop) for each rolled loop around *items*.
        for item in items:
            kind = _FLAT_KINDS.get(item["kind"], item["kind"])
            if kind == "loop":
                outer = {c["branch"] for c in item.get("conditions", ())}
                depth = len(item.get("iteration", ()))
                first = item.get("first_iteration", 0)
                for j in range(item["trips"]):
                    loop = (depth, first + j, j, item["period_cycles"], outer)
                    expand(item["body"], [*loops, loop])
                continue
            shift = sum(j * period for _, _, j, period, _ in loops)
            start, duration = item["start_cycles"], item["duration_cycles"]
            if start is not None:
                start += shift
            iteration = list(item.get("iteration", ()))
            for depth, k, _, _, _ in loops:
                iteration[depth] = k

            def branch(condition: Mapping[str, Any]) -> int:
                inner = tuple(j for _, _, j, _, outer in loops if condition["branch"] not in outer)
                return branches.setdefault((condition["branch"], inner), len(branches))

            conditions = [
                {
                    "branch": branch(condition),
                    "arm": condition["arm"],
                    "readouts": [latest[r] for r in condition.get("readouts", ())],
                    "answers": [latest[r] for r in condition.get("answers", ())],
                    "test": condition["test"],
                    **({"result": condition["result"]} if "result" in condition else {}),
                    **(
                        {"expression": resolve(condition["expression"])}
                        if "expression" in condition
                        else {}
                    ),
                }
                for condition in item.get("conditions", ())
            ]
            if kind == "sync":
                line, qudit = None, None
            elif kind in ("submit", "answer"):
                line, qudit = item["link"], -1
            elif "qudits" in item:
                qudit = item["qudits"][0]
                line = handle_of[(item["line"], qudit)]
            else:
                line, qudit = item["line"], qudit_of.get(item["line"], -1)
                if handles is not None and line is not None:
                    line = handle_of[(line, qudit)]
            event: dict[str, Any] = {
                "index": len(events),
                "kind": kind,
                "line": line,
                "qudit": qudit,
                **({"tone_slot": item["tone_slot"]} if "tone_slot" in item else {}),
                "start_cycles": start,
                "duration_cycles": duration,
                "end_cycles": None if start is None or duration is None else start + duration,
                "iteration": iteration,
                "segment": item.get("segment", 0)
                + sum(
                    j * step
                    for (_, _, j, _, _), step in zip(
                        loops, item.get("segment_step", ()), strict=False
                    )
                ),
                "conditions": conditions,
            }
            if kind in _PLAYED or kind == "frame_update":
                (tone,) = item["tones"]
                phase = tone["phase_rad"]
                if phase is not None:
                    phase += sum(
                        j * step
                        for (_, _, j, _, _), step in zip(
                            loops, tone.get("phase_step_rad", ()), strict=False
                        )
                        if j and step is not None
                    )
            if kind == "frame_update":
                event["op"] = item["op"]
                event["value"] = item["value"]
                event["frequency_hz"] = tone["frequency_hz"]
                event["phase_rad"] = phase
            elif kind in _PLAYED:
                amplitude = amplitudes[item["waveform"]]
                event["waveform"] = waveforms[item["waveform"]]
                event["amplitude"] = (
                    None if amplitude is None else {"re": amplitude.real, "im": amplitude.imag}
                )
                event["frequency_hz"] = tone["frequency_hz"]
                event["phase_rad"] = phase
                if run_time[item["waveform"]]:
                    event["waveform"] = _with_runtime(
                        event["waveform"], run_time[item["waveform"]], resolve
                    )
            elif kind == "sync":
                event["lines"] = (
                    list(item["lines"])
                    if handles is None
                    else [h for line in item["lines"] for h in handles[line]]
                )
            elif kind in ("submit", "answer"):
                event["link"] = item["link"]
                if "controller" in item:
                    event["controller"] = item["controller"]
                event["function"] = item["function"]
                if kind == "submit":
                    event["measurements"] = [latest[r] for r in item["measurements"]]
                else:
                    event["wait"] = item["wait"]
                    event["timeout_ns"] = item.get("timeout_ns")
            if "tone" in item:
                event["tone"] = item["tone"]
                event["frame_updates"] = frame_updates(item["frame_updates"], branch, shift)
                if "phase_offset_rad" in item:
                    event["phase_offset_rad"] = item["phase_offset_rad"]
            if "length" in item:
                event["length"] = resolve(item["length"])
                event["min_cycles"] = item.get("min_cycles")
                event["max_cycles"] = item.get("max_cycles")
            if "operands" in item:
                event["operands"] = [resolve(operand) for operand in item["operands"]]
            if kind in _PLAYED and samples is not None and samples[item["waveform"]] is not None:
                event["samples"] = samples[item["waveform"]]
            if "symbols" in item:
                event["symbols"] = {
                    k: v[:] if type(v) is list else v for k, v in item["symbols"].items()
                }
            latest[place[id(item)]] = event["index"]
            events.append(event)

    expand(program["events"], [])
    frame_state = [
        {
            **frame,
            "frame_updates": frame_updates(
                frame["frame_updates"],
                lambda c: branches.setdefault((c["branch"], ()), len(branches)),
            ),
        }
        if "frame_updates" in frame
        else dict(frame)
        for frame in program.get("frame_state", ())
    ]
    flat_keys = {"controller": "limits"}
    from . import _clocks

    # Format /1 states the rate after the schedule, and only when sampled.
    rate = program["sample_rate_ghz"] if "samples" in program else None
    if rate is not None and _clocks.made_for(rate, program["clock_ghz"])[1] != 1:
        raise InvalidValueError(
            f"format /1 samples at a whole multiple of the clock; this program samples "
            f"at {rate} GHz on a {program['clock_ghz']} GHz clock"
        )
    flat: dict[str, Any] = {}
    for k, v in program.items():
        if k in ("programs", "bindings", "waveforms", "samples", "parameters", "sample_rate_ghz"):
            continue
        flat[flat_keys.get(k, k)] = v
        if k == "schedule":
            flat["sample_rate_ghz"] = rate
    flat["format"] = FORMAT
    if flat["limits"] is not None:
        flat["limits"] = _flat_controller(flat["limits"])
    names = {v: k for k, v in _LINK_KEYS.items()}
    flat["links"] = {
        k: {names.get(f, f): v for f, v in link.items()} for k, link in flat["links"].items()
    }
    flat["lines"] = flat_lines
    flat["events"] = events
    if "frame_state" in flat:
        flat["frame_state"] = frame_state
    if handles is not None:
        for key in ("frame_state", "runtime_tones"):
            if key in flat:
                flat[key] = _on_handles(flat[key], handle_of, handles)
        flat["segments"] = _segments_on_handles(flat["segments"], events, handles)
    return flat


def _handles(
    program: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int], list[list[int]] | None]:
    """The format /1 lines of format /2 *program*, the /1 line of each
    (/2 line, qudit), and the /1 lines of each /2 line (``None`` when no
    line is shared, and the lines are the same). A shared line has one /1
    line for each of its qudits the program plays at, in the line's order,
    with the line's ``qudits`` and ``max_concurrent_tones``."""
    if not any("max_concurrent_tones" in line for line in program["lines"]):
        same = [
            {"line": line["line"], "role": line["role"], "qudit": line["qudits"][0]}
            for line in program["lines"]
        ]
        for handle, line in zip(same, program["lines"], strict=True):
            if "native_frequencies_hz" in line:
                handle["native_frequency_hz"] = line["native_frequencies_hz"][0]
            elif "broadcast" in line:
                raise _no_broadcast_unroll(line)
        return same, {(line["line"], line["qudit"]): line["line"] for line in same}, None
    played: set[tuple[int, int]] = set()
    stack = [program["events"]]
    while stack:
        for event in stack.pop():
            if event["kind"] == "loop":
                stack.append(event["body"])
            elif "qudits" in event:
                played.add((event["line"], event["qudits"][0]))
    for key in ("frame_state", "runtime_tones"):
        found = program.get(key, ())
        for item in found.values() if isinstance(found, Mapping) else found:
            if "qudits" in item:
                played.add((item["line"], item["qudits"][0]))
    lines: list[dict[str, Any]] = []
    handle_of: dict[tuple[int, int], int] = {}
    handles: list[list[int]] = []
    for line in program["lines"]:
        shared = "max_concurrent_tones" in line
        qudits = [q for q in line["qudits"] if not shared or (line["line"], q) in played]
        mine = []
        for qudit in qudits or line["qudits"][:1]:
            entry: dict[str, Any] = {"line": len(lines), "role": line["role"], "qudit": qudit}
            if shared:
                entry["qudits"] = list(line["qudits"])
                entry["max_concurrent_tones"] = line["max_concurrent_tones"]
            if "native_frequencies_hz" in line:
                at = line["qudits"].index(qudit)
                entry["native_frequency_hz"] = line["native_frequencies_hz"][at]
            elif "broadcast" in line:
                raise _no_broadcast_unroll(line)
            handle_of[(line["line"], qudit)] = len(lines)
            mine.append(len(lines))
            lines.append(entry)
        handles.append(mine)
    return lines, handle_of, handles


def _no_broadcast_unroll(line: Mapping[str, Any]) -> UnsupportedError:
    return UnsupportedError(
        f"line {line['line']} is a broadcast drive line, which format /1 writes with its "
        "qudits on the line and on each of its events; unrolling one is not supported "
        "yet: export the program in format /1, as export_to_controller(version=1)"
    )


def _on_handles(
    found: Any, handle_of: Mapping[tuple[int, int], int], handles: Sequence[Sequence[int]]
) -> Any:
    """``frame_state`` or ``runtime_tones`` of format /2 with each entry on
    the format /1 line of its qudit (``_handles``)."""

    def entry(item: Mapping[str, Any]) -> dict[str, Any]:
        if "qudits" not in item:
            line = item.get("line")
            if line is None or not 0 <= line < len(handles):
                return dict(item)
            return {**item, "line": handles[line][0]}
        qudit = item["qudits"][0]
        return {
            k: handle_of[(item["line"], qudit)] if k == "line" else v
            for k, v in item.items()
            if k != "qudits"
        }

    if isinstance(found, Mapping):
        return {key: entry(item) for key, item in found.items()}
    return [entry(item) for item in found]


def _segments_on_handles(
    segments: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    handles: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """*segments* of format /2 with the line of each opened by a variable
    wait the format /1 line of that wait: segment k is opened by the k-th
    barrier of the format /1 *events* (a variable wait, or a blocking wait
    for an answer)."""
    barriers = [
        e
        for e in events
        if (e["kind"] == "wait" and "length" in e)
        or (e["kind"] == "answer" and e["wait"] == "blocking")
    ]
    out = []
    for s in segments:
        if s.get("opened_by") == "wait" and 0 <= s["line"] < len(handles):
            k = s["index"] - 1
            barrier = barriers[k] if 0 <= k < len(barriers) else None
            line = (
                barrier["line"]
                if barrier is not None and barrier["line"] in handles[s["line"]]
                else handles[s["line"]][0]
            )
            s = {**s, "line": line}
        out.append(dict(s))
    return out


def _add_samples(
    events: Sequence[dict[str, Any]],
    sample_rate_ghz: float,
    clock: float,
    waveforms: _Waveforms | None = None,
    ratio: Fraction | int | None = None,
) -> list[dict[str, list[float]] | None]:
    """Give each drive and readout of *events* its envelope's samples at
    *sample_rate_ghz*, a whole multiple of the *clock* its times count; with
    *waveforms* (format /2), the samples of each of its waveforms instead,
    at *ratio* samples per cycle."""
    if ratio is None:
        per_cycles = round(sample_rate_ghz / clock)
        if per_cycles < 1 or abs(sample_rate_ghz / clock - per_cycles) > 1e-9:
            raise ValueError(
                f"sample_rate_ghz={sample_rate_ghz} is not a whole multiple of the "
                f"{clock} GHz clock"
            )
        ratio = per_cycles
    # The events of one op share its waveform, so each is sampled once.
    sampled: dict[tuple[int, int], dict[str, list[float]]] = {}
    samples_of: list[dict[str, list[float]] | None] = []
    if waveforms is not None:
        samples_of = [None] * len(waveforms.table)
    for event in _played(events):
        index: int = event["waveform"] if waveforms is not None else -1
        tree = event["waveform"] if waveforms is None else waveforms.table[index]
        duration = event["duration_cycles"]
        if tree is None or duration is None:
            continue
        key = (id(tree), duration)
        if key not in sampled:
            try:
                real, imag = _sample(tree, duration, ratio)
            except ValueError as e:
                where = event.get("index", "of waveform")
                raise ValueError(f"event {where} ({event['kind']}): {e}") from None
            sampled[key] = {"re": real, "im": imag}
        if waveforms is None:
            event["samples"] = sampled[key]
        else:
            samples_of[index] = sampled[key]
    return samples_of


def _controller_index(controller: Any, count: int) -> int:
    """*controller*, checked as the index of one of *count* controllers."""
    if isinstance(controller, bool) or not isinstance(controller, int):
        raise InvalidTypeError(
            f"controller= is the index of a controller of the control system, got {controller!r}"
        )
    if not 0 <= controller < count:
        raise InvalidValueError(
            f"controller={controller}, and the program's control system has "
            + ("one controller (0)" if count == 1 else f"{count} controllers (0 to {count - 1})")
        )
    return controller


def _part(
    lowered: ScheduledKernel, controller: int | None
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[int | None],
    list[list[int]],
    int,
    dict[str, Any],
]:
    """Of a program on a control system of several controllers: its lines
    and events in program cycles, each line's controller, the controllers
    taking part in each event, and the controller whose program is wanted:
    *controller*, else the only one the program uses (``ValueError`` when it
    uses more), and the program's frames (``_walk``). A controller the
    program does not use gets an empty part."""
    system = lowered.options.machine._system
    assert system is not None
    lines, events, frames, _, _ = _walk(lowered, slots=None)
    owners, _ = system._owners(lines)
    parts = system._parts(events, owners)
    index = _part_index(
        controller,
        len(system.controllers),
        _known_owners(owners),
        lambda: sorted({c for part in parts for c in part}),
    )
    return lines, events, owners, parts, index, frames


def _known_owners(owners: Sequence[int | None]) -> list[int]:
    return sorted({o for o in owners if o is not None})


def _part_index(
    controller: int | None, count: int, used: list[int], calling: Callable[[], list[int]]
) -> int:
    """The controller of *count* whose part is wanted: *controller*, else
    the only one that plays a line (of *used*), else the only one that makes
    a call (*calling* gives them)."""
    if controller is not None:
        return _controller_index(controller, count)
    if len(used) > 1:
        raise InvalidValueError(
            f"the program plays lines of {len(used)} controllers ("
            + ", ".join(f"controllers[{i}]" for i in used)
            + f"); pass controller= (0 to {count - 1}) to export one controller's program"
        )
    if not used:
        # Without lines, the program is its calls.
        used = calling()
        if len(used) > 1:
            raise InvalidValueError(
                f"the program's calls are made on {len(used)} controllers ("
                + ", ".join(f"controllers[{i}]" for i in used)
                + f"); pass controller= (0 to {count - 1}) to export one controller's "
                "program"
            )
    return used[0] if used else 0


# The nested values of an event that converting it to its controller's cycles
# rewrites (``_clocks.off_grid``).
_CONVERTED = ("frame_updates", "waveform")


class _Parts:
    """What the exports of each controller's part of one program share: the
    program walked in program cycles and checked against the whole control
    system, once. A program keeps it (``ScheduledKernel._export_parts``)
    until every controller's part has been exported, and each export copies
    the events it takes, so that what it returns shares nothing with the
    next. Once the check fails, only the violations and what chooses the
    controller are kept."""

    __slots__ = (
        "key",
        "lines",
        "events",
        "frames",
        "owners",
        "parts",
        "used",
        "calling",
        "violations",
        "tone_slots",
        "plan",
        "slot_names",
        "members",
        "played",
        "synced",
        "exported",
        "program",
        "lock",
    )

    def __init__(self, lowered: ScheduledKernel, system: Any) -> None:
        self.key = _parts_key(lowered)
        self.lines, self.events, self.frames, _, _ = _walk(lowered, slots=None)
        self.owners, _ = system._owners(self.lines)
        self.parts: list[list[int]] | None = system._parts(self.events, self.owners)
        self.used = _known_owners(self.owners)
        self.calling: list[int] | None = None
        self.violations: list[str] | None = None
        self.tone_slots: Mapping[int, int] | None = None
        self.plan: _ExportPlan | None = None
        self.slot_names: dict[int, str] | None = None
        self.members: list[list[int]] = []
        self.played: list[list[dict[str, Any]]] = []
        self.synced: dict[int, dict[int, list[int]]] = {}
        self.exported: set[int] = set()
        self.program: tuple[Any, Any] | None = None
        self.lock = threading.Lock()

    def whole(self, lowered: ScheduledKernel) -> tuple[Any, Any]:
        """What each part's emulation data reads of the whole program, its
        segments and calls (``classical.wire._program``), found once."""
        with self.lock:
            if self.program is None:
                from .classical.wire import _program

                self.program = _program(lowered)
            return self.program

    def calls(self) -> list[int]:
        """The controllers that take part in the program's calls."""
        if self.calling is None:
            assert self.parts is not None
            self.calling = sorted({c for part in self.parts for c in part})
        return self.calling

    def check(self, lowered: ScheduledKernel, system: Any, clocks: Any) -> list[str]:
        """The violations of the whole control system, found once; then what
        every part needs of the whole program."""
        with self.lock:
            if self.violations is None:
                self.violations = self._checked(lowered, system, clocks)
            return self.violations

    def _checked(self, lowered: ScheduledKernel, system: Any, clocks: Any) -> list[str]:
        events = self.events
        # The checks count each controller's events in its cycles, in place (a
        # controller on another clock, or timing in steps what is computed at
        # run time): each event's times and its frame updates' and waveform's.
        checked = events
        if clocks is not None and any(
            c.ratio != 1 or (c.grid > 1 and _computes_at_run_time(lowered))
            for c in clocks.controllers
        ):
            memo: dict[int, Any] = {}
            checked = [
                {**e, **{k: copy.deepcopy(e[k], memo) for k in _CONVERTED if k in e}}
                for e in events
            ]
        violations: list[str] = system._violations(
            lowered, self.lines, checked, clocks, (), (), True, None, None
        )
        if violations:
            self.calls()
            self.lines, self.events, self.frames, self.owners, self.parts = [], [], {}, [], []
            return violations
        if system._shares:
            # From the whole program, in program cycles: a shared line is one
            # controller's, and its slots are those of every export of the
            # program.
            self.tone_slots = system._shares.flat(system, self.lines, events)
        role = lowered._role
        if role is not None:
            plan = role.export_plan
            if plan is None or len(plan[1]) != len(events):
                slots: list[int] = []
                walked = _walk(lowered, slots=slots)[0]
                plan = _export_plan(lowered, role, slots, walked, events)
            self.plan = plan
        count = len(system.controllers)
        self.members = [[] for _ in range(count)]
        assert self.parts is not None
        for position, part in enumerate(self.parts):
            for c in part:
                self.members[c].append(position)
        self.played = [[] for _ in range(count)]
        for line, owner in zip(self.lines, self.owners, strict=True):
            if owner is not None:
                self.played[owner].append(line)
        owners = self.owners
        for position, event in enumerate(events):
            if event["kind"] == "sync":
                synced: dict[int, list[int]] = {}
                for line in event["lines"]:
                    owner = owners[line]
                    if owner is not None:
                        synced.setdefault(owner, []).append(line)
                self.synced[position] = synced
        return violations

    def names(self, role: Any) -> dict[int, str]:
        """``_slot_names`` of the program's events with their symbols."""
        if self.slot_names is None:
            names: dict[int, str] = {}
            if role is not None and self.plan is not None:
                for event, template in zip(self.events, self.plan[1], strict=True):
                    if template is not None and "qudit" in template[0]:
                        names[event["line"]] = template[0]["qudit"]
            self.slot_names = names
        return self.slot_names


def _parts_key(lowered: ScheduledKernel) -> tuple[Any, ...]:
    """What a program's walk depends on that a program keeps, compared by
    identity: none of it is replaced once the program is made."""
    return (lowered.options, lowered.quantum_system, lowered.arguments, lowered.carriers)


_parts_lock = threading.Lock()


def _shared_parts(lowered: ScheduledKernel, system: Any) -> _Parts:
    """The ``_Parts`` of *lowered*: the one it keeps for its key, else new,
    kept when its control system has more than one controller."""
    with _parts_lock:
        kept: _Parts | None = getattr(lowered, "_export_parts", None)
    if kept is not None:
        key = _parts_key(lowered)
        if all(a is b for a, b in zip(kept.key, key, strict=True)):
            return kept
    shared = _Parts(lowered, system)
    if len(system.controllers) > 1:
        with _parts_lock:
            lowered._export_parts = shared
    return shared


def _exported(lowered: ScheduledKernel, shared: _Parts, index: int, count: int) -> None:
    """Note that the part of controller *index* of *count* is exported: once
    every part is, *lowered* keeps *shared* no longer."""
    with _parts_lock:
        shared.exported.add(index)
        if len(shared.exported) == count and getattr(lowered, "_export_parts", None) is shared:
            del lowered._export_parts


def _copied(value: Any, memo: dict[int, Any]) -> Any:
    """*value*, export data, with its dicts and lists copied; one shared by
    two values in *memo*'s copy stays shared."""
    cls = value.__class__
    if cls is dict:
        found = memo.get(id(value))
        if found is None:
            found = memo[id(value)] = {}
            for key, item in value.items():
                found[key] = _copied(item, memo)
        return found
    if cls is list:
        found = memo.get(id(value))
        if found is None:
            found = memo[id(value)] = []
            found.extend([_copied(item, memo) for item in value])
        return found
    return value


def _export_part(
    lowered: ScheduledKernel, sample_rate_ghz: float | None, controller: int | None
) -> dict[str, Any]:
    """``export`` on a control system of several controllers: the program of
    one, in its own cycles, with its lines and limits and the events it
    takes part in (see ``ControlSystem._parts``); a sync lists its lines
    alone. Line and event ids are the program's, so an event names one
    another controller plays (a readout whose outcome it is sent) by the id
    that controller's program gives it. The walk and the check of the whole
    system are shared by the exports of every controller (``_Parts``)."""
    from .classical.wire import _export as emulation

    machine = lowered.options.machine
    system = machine._system
    assert system is not None
    clocks = (lowered._generic or lowered).clocks
    count = len(system.controllers)
    shared = _shared_parts(lowered, system)
    index = _part_index(controller, count, shared.used, shared.calls)
    violations = shared.check(lowered, system, clocks)
    if violations:
        _exported(lowered, shared, index, count)
        raise _limit_error(lowered, violations)
    role = lowered._role
    plan = shared.plan
    events = shared.events
    memo: dict[int, Any] = {}
    mine: list[dict[str, Any]] = []
    for position in shared.members[index]:
        source = events[position]
        if source["kind"] == "sync":
            # A sync lists its lines alone.
            synced = shared.synced[position].get(index, ())
            event: dict[str, Any] = {
                key: list(synced) if key == "lines" else _copied(value, memo)
                for key, value in source.items()
            }
        else:
            event = _copied(source, memo)
        if plan is not None:
            template = plan[1][position]
            if template is not None:
                symbols = template[0].copy()
                for key in template[1]:
                    symbols[key] = symbols[key][:]
                event["symbols"] = symbols
        mine.append(event)
    played = [_copied(line, memo) for line in shared.played[index]]
    own = None
    keys: dict[str, float] = {}
    if clocks is not None:
        from . import _clocks

        own = clocks.controllers[index]
        keys = _clocks.clock_keys(clocks, own)
        if own.ratio != 1 or own.grid > 1:
            alone = _clocks.ClockPlan(clocks.program_ghz, (own,), clocks.reference_clock_ghz)
            _clocks.off_grid(alone, shared.lines, mine, True, True, _computes_at_run_time(lowered))
    clock = lowered.quantum_system.clock_ghz if own is None else own.clock_ghz
    if sample_rate_ghz is not None:
        _add_samples(mine, sample_rate_ghz, clock)
    functions: frozenset[str] = frozenset(
        e["function"] for e in mine if e["kind"] in ("submit", "answer")
    )
    base = emulation(lowered, False, (index, own, functions, ()), shared.whole(lowered))
    exported = {
        "format": FORMAT,
        "quantum_system": lowered.quantum_system.name,
        "clock_ghz": clock,
        **keys,
        "controller": index,
        "schedule": lowered.options.schedule,
        "sample_rate_ghz": sample_rate_ghz,
        "qudits": sorted({line["qudit"] for line in played}),
        "lines": played,
        "events": mine,
        "segments": base["segments"],
        "calls": base["calls"],
        "links": base["links"],
        "manifest": base["manifest"],
        "limits": system.controllers[index].as_dict(),
    }
    if role is not None:
        assert plan is not None
        exported["program_id"] = role.program_id
        exported["symbols"] = _symbols(lowered, plan[0])
    frames = shared.frames
    if "runtime_tones" in frames:
        # A controller's part has the run-time tones of its lines, and no
        # frame_state.
        tones = {"runtime_tones": _copied(frames["runtime_tones"], memo)}
        exported = _with_frames(exported, tones, shared.names(role))
        if system._shares:
            assert shared.tone_slots is not None
            system._shares.framed(exported, shared.tone_slots)
    _exported(lowered, shared, index, count)
    return exported


_DEPENDENT_FIELDS = ("frequency_hz", "phase_rad", "waveform")
# The kinds of event, in either format, that name their line's qudit.
_ON_LINE = _PLAYED | {"wait", "frame_update"}


# Where a symbol's value comes from in a call: its float arguments, the
# carrier bound to a slot, or that carrier's 0-1 or readout frequency.
_ARGUMENT, _CARRIER, _NATIVE, _READOUT = range(4)


def _symbols(
    lowered: ScheduledKernel, templates: list[tuple[dict[str, Any], int, int]]
) -> list[dict[str, Any]]:
    """Every symbol of a call's module, with its value in the call: its
    float arguments, its slots (the carriers bound) and the native
    frequencies of each slot that gets a line."""
    arguments = lowered.arguments
    carriers = lowered.carriers
    found = []
    for template, source, k in templates:
        entry = template.copy()
        if source == _ARGUMENT:
            entry["value"] = arguments[k]
        elif source == _CARRIER:
            entry["value"] = carriers[k]
        elif source == _NATIVE:
            entry["value"] = lowered.quantum_system.frequencies[carriers[k]]
        else:
            target = lowered.quantum_system
            carrier = carriers[k]
            entry["value"] = target._frequency_tables(carrier + 1)[2][carrier]
        found.append(entry)
    return found


# Each symbol without its value and where the value comes from; each event's
# symbols, with the keys whose values are lists.
_ExportPlan = tuple[
    list[tuple[dict[str, Any], int, int]],
    list[tuple[dict[str, Any], tuple[str, ...]] | None],
]


def _symbol_plan(
    role: Any, slots: Sequence[int], lines: Sequence[Mapping[str, Any]]
) -> list[tuple[dict[str, Any], int, int]]:
    """Each symbol of a role without its value, and where the value comes
    from (see ``_symbols``). By the runtime-qudit *slots* of *lines*, a slot
    with a drive line has its 0-1 frequency (``s0.f01``), and one with a
    readout line its readout frequency (``s0.f_ro``): the native frequency
    of each such line's tone."""
    lined = {(slot, line["role"]) for slot, line in zip(slots, lines, strict=True)}
    symbols: list[tuple[dict[str, Any], int, int]] = []
    argument = 0
    for name, kind, parameter, position in role.symbols:
        entry: dict[str, Any] = {"symbol": name, "kind": kind, "parameter": parameter}
        if position is not None:
            entry["position"] = position
        if kind == "slot":
            slot = int(name[1:])
            symbols.append((entry, _CARRIER, slot))
            if (slot, "drive") in lined:
                native = {"symbol": f"{name}.f01", "kind": "frequency", "slot": name}
                symbols.append((native, _NATIVE, slot))
            if (slot, "readout") in lined:
                native = {"symbol": f"{name}.f_ro", "kind": "frequency", "slot": name}
                symbols.append((native, _READOUT, slot))
        else:
            symbols.append((entry, _ARGUMENT, argument))
            argument += 1
    return symbols


def _named_dependencies(role: Any, found: Sequence[Sequence[Any]]) -> list[dict[str, list[str]]]:
    """The arguments of `main` each event depends on (see
    ``controller_dependencies``), as the names of their symbols by field."""
    names = [name for name, kind, _, _ in role.symbols if kind != "slot"]
    names += [f"s{k}.f01" for k in range(role.slots)]
    names += [f"s{k}.f_ro" for k in range(role.slots)]
    return [
        {
            field: [names[a] for a in arguments]
            for field, arguments in zip(_DEPENDENT_FIELDS, item, strict=True)
            if arguments
        }
        for item in found
    ]


def _structured_symbols(
    lowered: ScheduledKernel, role: Any, events: list[dict[str, Any]], slots: Sequence[int]
) -> None:
    """Gives each event of a format /2 program the symbols its fields depend
    on, as format /1 gives them; an event of a loop's body, those of any of
    its iterations. They come from the module the role compiled; a loop
    that rolls only at this call's values is ``UnsupportedError``."""
    structured = role.structured
    if structured is None:
        with lowered._lock:
            found = lowered._module.controller_rolled_dependencies()
        named = _named_dependencies(role, [item for _, _, item in found])
        structured = role.structured = [
            (_KINDS.get(kind, kind), trips, symbols)
            for (kind, trips, _), symbols in zip(found, named, strict=True)
        ]
    place = iter(structured)

    def give(items: list[dict[str, Any]]) -> bool:
        for event in items:
            entry = next(place, None)
            kind = event["kind"]
            if entry is None or entry[0] != kind or entry[1] != event.get("trips", -1):
                return False
            symbols: dict[str, Any] = {}
            line = event.get("line")
            if kind in _ON_LINE and line is not None:
                if 0 <= line < len(slots) and slots[line] >= 0:
                    symbols["qudit"] = f"s{slots[line]}"
            for key, names in entry[2].items():
                symbols[key] = names[:]
            if symbols:
                event["symbols"] = symbols
            if kind == "loop" and not give(event["body"]):
                return False
        return True

    if not give(events) or next(place, None) is not None:
        raise UnsupportedError(
            "format /2 cannot name the symbols of this call: a loop repeats only at "
            "its values; export version 1 for it"
        )


def _export_plan(
    lowered: ScheduledKernel,
    role: Any,
    slots: Sequence[int],
    lines: Sequence[Mapping[str, Any]],
    events: list[dict[str, Any]],
) -> _ExportPlan:
    """What a role's exports share, built once per role from a call's
    *events*, its *lines* and their runtime-qudit *slots*."""
    symbols = _symbol_plan(role, slots, lines)
    dependencies = role.dependencies
    if dependencies is None:
        with lowered._lock:
            found = lowered._module.controller_dependencies()
        dependencies = role.dependencies = _named_dependencies(role, found)
    aligned = len(dependencies) == len(events)
    templates: list[tuple[dict[str, Any], tuple[str, ...]] | None] = []
    for index, event in enumerate(events):
        template: dict[str, Any] = {}
        line = event["line"]
        if event["kind"] in _ON_LINE and line is not None:
            if 0 <= line < len(slots) and slots[line] >= 0:
                template["qudit"] = f"s{slots[line]}"
        if aligned:
            template.update(dependencies[index])
        lists = tuple(k for k, v in template.items() if type(v) is list)
        templates.append((template, lists) if template else None)
    plan: _ExportPlan = (symbols, templates)
    role.export_plan = plan
    return plan


# ── Checks ───────────────────────────────────────────────────────────────────


def _memory(limits: Controller) -> tuple[int, int, bool, bool, list[str], bool]:
    """The waveform memory of *limits* as the native check takes it:
    (samples, granule, shared, scaling, synthesized shapes, hold)."""
    samples = -1 if limits.waveform_memory_samples is None else limits.waveform_memory_samples
    shared = limits.waveform_memory is WaveformMemoryScope.SHARED
    return samples, 1, shared, True, [], False


def _walks_every(
    lowered: ScheduledKernel, limits: Controller, clocks: _clocks.ClockPlan | None
) -> bool:
    """Whether a check of *limits* walks every execution rather than a walk
    that summarizes loops: for calls that hold the lines, and for update
    buses with a violation to list, which the summarizing walk, without
    frame updates, cannot. On one clock the native check says whether
    there is one (``_bus_fast``); a call binds a runtime qudit's bus."""
    if limits.blocks_on_calls:
        return True
    if not limits.update_buses:
        return False
    if not _one_clock(clocks) or (lowered._role is not None and lowered._role.slots > 0):
        return True
    return bool(_bus_fast(lowered, lowered._concrete(), limits))


def _checks_natively(limits: Controller, clocks: _clocks.ClockPlan | None) -> bool:
    """Whether the native check counts every limit of *limits* on *clocks*."""
    return (
        _one_clock(clocks)
        and not (limits.max_pulses is not None and limits._runs_loops)
        and not limits.blocks_on_calls
    )


def _native_check(
    lowered: ScheduledKernel, limits: Controller, unresolved: list[str] | None = None
) -> list[str]:
    """The limits *lowered* exceeds as the native check finds them: a loop
    whose iterations repeat is checked once, its first iterations and its
    last standing for all of them, so a violation inside it is reported
    once. With *unresolved*, a tone or waveform whose value is not known yet
    counts only if the limit is exceeded whatever its value; a limit that
    only some values exceed is appended to *unresolved*."""
    num, den = _samples_per_own_cycle(limits, lowered.quantum_system.clock_ghz)
    module = lowered._concrete()
    found: list[str]
    with lowered._lock:
        found, maybe = module.controller_check(
            limits.instruction_cycles,
            limits.instruction_cycles + limits.min_idle_cycles,
            num,
            den,
            _memory(limits),
            -1 if limits.max_frequencies_per_line is None else limits.max_frequencies_per_line,
            -1 if limits.max_pulses is None else limits.max_pulses,
            dict(limits.detectors),
            limits.detector_recovery_cycles,
            limits.feedback_latency_cycles,
            limits.supports_timeouts,
            [fn.name for fn in lowered.classical if fn.latency_mode == "timeout"],
            unresolved is not None,
        )
    if maybe and unresolved is not None:
        unresolved.append("tone frequencies or waveform memory")
    # Any found sends the check to its walk, which counts the amplitudes too.
    if limits.max_amplitude is not None and not found:
        found.extend(_amplitude_violations(lowered, module, limits.max_amplitude, unresolved))
    if _checks_pulse_lengths(limits) and not found:
        found.extend(_pulse_length_violations(lowered, module, limits))
    # A value a call sets is an update here; the walk tells it apart.
    if limits.update_buses and not found:
        found.extend(_bus_fast(lowered, module, limits))
    return found


def _found_natively(
    lowered: ScheduledKernel, limits: Controller, unresolved: list[str] | None = None
) -> bool:
    """Whether *lowered* exceeds a limit of *limits* (``_checks_natively``):
    a check then lists its violations, one per execution, from a walk."""
    if _native_check(lowered, limits, unresolved):
        return True
    return (
        limits is not lowered.options.machine.controller
        and _sample_rates(lowered, limits) is not None
    )


def _limit_violations(
    lowered: ScheduledKernel,
    limits: Controller,
    clocks: _clocks.ClockPlan | None,
    walked: tuple[Any, Any, Any, Any, Any] | None = None,
    unresolved: list[str] | None = None,
) -> Sequence[str]:
    """The limits a call's schedule exceeds, as ``_violations`` finds them,
    one per execution. On one clock, without limits the native check does
    not count, the native check says whether there are any. They are listed
    from the events of *walked* (``_walk``'s format /1, in the controller's
    cycles), else of a walk made here that has a long rolled loop whose
    iterations repeat once (``_loops``).

    With *unresolved*, a tone or waveform whose value is not known yet
    counts only if the limit is exceeded whatever its value; a limit that
    only some values exceed is appended to *unresolved*."""
    if _checks_natively(limits, clocks):
        maybe: list[str] | None = None if unresolved is None else []
        if not _found_natively(lowered, limits, maybe):
            if unresolved is not None and maybe:
                unresolved.extend(maybe)
            return []
    if walked is not None:
        lines, events, _, off, problems = walked
        return _violations(lowered, events, limits, lines, clocks, off, problems, True, unresolved)
    repeats: list[tuple[int, ...]] = []
    for unsynced in (True, False):
        periods: list[tuple[int, int, int]] | None = [] if unsynced else None
        lines, events, _, off, problems = _walk(
            lowered,
            unresolved is None,
            _native_clock(clocks),
            False,
            # Held lines are checked call by call, every iteration walked.
            summary=None if _walks_every(lowered, limits, clocks) else (repeats, periods),
        )
        if not repeats:
            break
        from ._loops import UnsyncedError, violations

        if problems:
            # A summarizing walk's problems name only the copies it plays.
            lines, events, _, off, problems = _walk(
                lowered, unresolved is None, _native_clock(clocks), False
            )
            break
        own = _native_clock(clocks)
        try:
            return violations(
                lowered, events, repeats, limits, lines, unresolved, None, periods or (), own, off
            )
        except UnsyncedError:
            repeats = []
    return _violations(lowered, events, limits, lines, clocks, off, problems, False, unresolved)


def _exclusive(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two events lie in different arms of one branch execution. A
    walk numbers each execution of a branch apart, so the events of one
    branch id are of one execution, a loop in an arm or not; a summarizing
    walk's copies of a loop share its body's ids, and its callers compare
    only events of one copy (``_loops._Pairs``)."""
    if not a["conditions"] or not b["conditions"]:
        return False
    arms = {c["branch"]: c["arm"] for c in a["conditions"]}
    for c in b["conditions"]:
        arm = arms.get(c["branch"])
        if arm is not None and arm != c["arm"]:
            return True
    return False


_STARTED = operator.itemgetter("start_cycles", "index")


def _pairs(
    events: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Each event with the latest earlier event it can follow on one path:
    for an event without conditions, the one before it."""
    ordered = sorted(events, key=_STARTED)
    pairs = []
    for k in range(1, len(ordered)):
        event = ordered[k]
        if not event["conditions"]:
            pairs.append((ordered[k - 1], event))
            continue
        for j in range(k - 1, -1, -1):
            if not _exclusive(ordered[j], event):
                pairs.append((ordered[j], event))
                break
    return pairs


def _violations(
    lowered: ScheduledKernel,
    events: Sequence[dict[str, Any]],
    limits: Controller,
    lines: Sequence[Mapping[str, Any]],
    clocks: _clocks.ClockPlan | None,
    off: Sequence[tuple[int, int, int]] = (),
    run_time: Sequence[str] = (),
    waveforms: bool = True,
    unresolved: list[str] | None = None,
    natives: Sequence[int] | None = None,
    *,
    held: bool = True,
    timed: Sequence[Mapping[str, Any]] | None = None,
    answers: Sequence[Mapping[str, Any]] | None = None,
    updates: Sequence[Mapping[str, Any]] | None = None,
    peaks: dict[str, float | None] | None = None,
) -> list[str]:
    """The limits *events* exceed. With *clocks*, the events are in the
    controller's own cycles: the native walk's, with *off* its line events
    off the grid and *run_time* the violations of what they compute at run
    time, or else (lines on several controllers) converted here in place,
    their waveforms too unless *waveforms* is false (the limits count
    waveforms alike either way). With *unresolved*, a tone or waveform
    whose value is not known yet counts only if the limit is exceeded
    whatever its value; a limit that only some values exceed is appended to
    *unresolved* instead, to be checked with them. *natives* has, by event
    index, the qudit argument whose native frequency the event plays, or -1.
    *held* false leaves the lines held by calls to the caller, which knows
    the controller each call holds. *timed* and *answers*, when the caller
    has them, are the timed drives and readouts of *events* and its answers,
    in order; then only the update-bus check reads all of *events*, or only
    *updates*, the frame updates, drives and readouts of *events* on the
    controller's lines, when the caller has them. *peaks* holds the sampled
    peak of each envelope (``_over_amplitude``), shared with the checks of
    other controllers of the same events."""
    if clocks is None:
        found = []
    else:
        from . import _clocks

        if clocks.owners is None:
            found = _clocks.off_grid_messages(clocks.controllers[0], lines, events, off)
            found.extend(run_time)
        else:
            found = _clocks.off_grid(
                clocks, lines, events, True, waveforms, _computes_at_run_time(lowered)
            )
    if timed is None:
        timed = [e for e in events if e["kind"] in _PLAYED and e["end_cycles"] is not None]
    if answers is None:
        answers = () if limits.supports_timeouts else [e for e in events if e["kind"] == "answer"]
    if limits.max_pulses is not None and len(timed) > limits.max_pulses:
        found.extend(_event_count(timed, limits.max_pulses, limits._runs_loops))
    by_line: dict[int, list[Mapping[str, Any]]] = {}
    for event in timed:
        by_line.setdefault(event["line"], []).append(event)
    issue, gap = limits.instruction_cycles, limits.instruction_cycles + limits.min_idle_cycles
    num, den = _samples_per_own_cycle(limits, lowered.quantum_system.clock_ghz)
    shared = limits.waveform_memory is WaveformMemoryScope.SHARED
    stored: dict[Any, int] = {}
    unknown_stored = 0
    if peaks is None:
        peaks = {}
    for line, items in sorted(by_line.items()):
        for previous, event in _pairs(items):
            distance = event["start_cycles"] - previous["start_cycles"]
            if distance != issue and distance < gap:
                found.append(
                    f"line {line}: {event['kind']} at {event['start_cycles']} starts "
                    f"{distance} cycles after the {previous['kind']} at "
                    f"{previous['start_cycles']}; the controller "
                    f"needs exactly {issue} or at least {gap}"
                )
        if limits.max_frequencies_per_line is not None:
            tones = {e["frequency_hz"] for e in items if e.get("frequency_hz") is not None}
            if len(tones) > limits.max_frequencies_per_line:
                found.append(
                    f"line {line}: {len(tones)} tone frequencies; the controller holds "
                    f"{limits.max_frequencies_per_line} per line"
                )
            elif unresolved is not None:
                unknown = [e for e in items if e.get("frequency_hz") is None]
                if natives is None:
                    count = len(unknown)
                else:
                    # A runtime qudit's own frequency is one tone however
                    # often it plays: an upper bound for every placement.
                    keys = [natives[e["index"]] for e in unknown]
                    count = len({k for k in keys if k >= 0}) + keys.count(-1)
                if len(tones) + count > limits.max_frequencies_per_line:
                    unresolved.append(f"line {line}: tone frequencies")
        if limits.max_amplitude is not None:
            found.extend(
                _over_amplitude(lines[line], items, limits.max_amplitude, unresolved, peaks)
            )
        if _checks_pulse_lengths(limits):
            for e in items:
                duration = e["end_cycles"] - e["start_cycles"]
                violation = _pulse_length(lines[line], e["kind"], duration, limits)
                if violation is not None:
                    found.append(f"{violation[0]}{e['start_cycles']}{violation[1]}")
        if limits.waveform_memory_samples is not None:
            shapes = stored if shared else {}
            unknown_samples = 0
            for e in items:
                key = shape_key(e.get("waveform"))
                samples = -(-e["duration_cycles"] * num // den)
                if key is not None:
                    shapes[(key, e["duration_cycles"])] = samples
                elif unresolved is not None:
                    unknown_samples += samples
                else:
                    shapes[("event", e["index"])] = samples
            used = sum(shapes.values())
            if not shared and used > limits.waveform_memory_samples:
                found.append(
                    f"line {line}: its waveforms take {used} samples; the controller's "
                    f"waveform memory holds {limits.waveform_memory_samples}"
                )
            elif not shared and unresolved is not None:
                if used + unknown_samples > limits.waveform_memory_samples:
                    unresolved.append(f"line {line}: waveform memory")
            unknown_stored += unknown_samples
    if shared and limits.waveform_memory_samples is not None:
        used = sum(stored.values())
        if used > limits.waveform_memory_samples:
            found.append(
                f"its waveforms take {used} samples; the controller's shared waveform "
                f"memory holds {limits.waveform_memory_samples}"
            )
        elif unresolved is not None and used + unknown_stored > limits.waveform_memory_samples:
            unresolved.append("waveform memory")
    if limits.detectors:
        by_acquirer: dict[int, list[Mapping[str, Any]]] = {}
        for event in timed:
            if event["kind"] == "readout" and event["qudit"] in limits.detectors:
                by_acquirer.setdefault(limits.detectors[event["qudit"]], []).append(event)
        for detector, items in sorted(by_acquirer.items()):
            for previous, event in _pairs(items):
                free = previous["end_cycles"] + limits.detector_recovery_cycles
                if event["start_cycles"] < free:
                    found.append(
                        f"detector {detector}: the readout of qudit {event['qudit']} at "
                        f"{event['start_cycles']} starts before the detector is free at {free} "
                        f"(the readout of qudit {previous['qudit']} at {previous['start_cycles']})"
                    )
    if limits.update_buses:
        on_lines = events if updates is None else updates
        found.extend(_bus_violations(lowered, on_lines, limits, lines, clocks, unresolved))
    if limits.feedback_latency_cycles:
        for event in timed:
            for condition in event["conditions"]:
                for ref in condition["readouts"]:
                    readout = events[ref]
                    if readout["end_cycles"] is None:
                        continue
                    ready = readout["end_cycles"] + limits.feedback_latency_cycles
                    if event["start_cycles"] < ready:
                        found.append(
                            f"line {event['line']}: the {event['kind']} at {event['start_cycles']} "
                            f"depends on the readout ending at {readout['end_cycles']}; the "
                            f"controller needs {limits.feedback_latency_cycles} cycles of feedback "
                            f"latency (start at {ready} or later; schedule with rcp and "
                            "readout_latency_cycles >= feedback_latency_cycles)"
                        )
    if not limits.supports_timeouts:
        functions = {fn.name: fn for fn in lowered.classical}
        for event in answers:
            fn = functions.get(event["function"])
            if event["wait"] == "timeout" or (fn is not None and fn.latency_mode == "timeout"):
                found.append(
                    f"the wait for {event['function']} at {event['start_cycles']} has a timeout; "
                    "the controller cannot stop waiting for an answer, so take it with "
                    "result() (no timeout)"
                )
    if held and limits.blocks_on_calls:
        found.extend(_held(lowered))
    return found


def _held(lowered: ScheduledKernel) -> list[str]:
    """The drives and readouts of *lowered* that play while one of its
    calls is out, from its submission until its answer is due, for a
    controller that blocks on calls; in program cycles."""
    if not lowered.classical or not lowered.scheduled:
        return []
    events = _walk(lowered, values=False)[1]
    return [
        _held_text(submit, call, event)
        for submit, call, event in _played_while_out(events, lowered.classical_report())
    ]


# How far apart two phases may be and still be the same, in radians.
_PHASE_TOLERANCE = 1e-6


def _bus_violations(
    lowered: ScheduledKernel,
    events: Sequence[Mapping[str, Any]],
    limits: Controller,
    lines: Sequence[Mapping[str, Any]],
    clocks: _clocks.ClockPlan | None,
    unresolved: list[str] | None,
) -> list[str]:
    """The updates (``_updates``) on one of ``limits.update_buses`` that
    start less than its ``cycles_per_update`` after the update before them
    on the bus, on one path; *events* as ``_violations`` takes them. With
    *unresolved*, a bus whose updates need the call's values is noted there
    instead."""
    if clocks is not None and clocks.owners is None:
        clock_ghz = clocks.controllers[0].clock_ghz
    else:
        clock_ghz = lowered.quantum_system.clock_ghz
    clashes = _bus_clashes(lowered.quantum_system, events, limits, lines, clock_ghz, unresolved)
    return [_clash_text(limits, *clash) for clash in clashes]


def _clash_text(
    limits: Controller,
    index: int,
    previous: Mapping[str, Any],
    event: Mapping[str, Any],
    gap: int,
) -> str:
    bus = limits.update_buses[index]
    return (
        f"update bus {index} ({bus._named()}): the {event['kind']} on line "
        f"{event['line']} at {event['start_cycles']} updates its tone {gap} cycles "
        f"after the {previous['kind']} on line {previous['line']} at "
        f"{previous['start_cycles']}; the bus carries one update per "
        f"{bus.cycles_per_update} cycles (Controller.update_buses)"
    )


def _bus_clashes(
    target: QuantumSystem,
    events: Sequence[Mapping[str, Any]],
    limits: Controller,
    lines: Sequence[Mapping[str, Any]],
    clock_ghz: float,
    unresolved: list[str] | None,
    cycles: Sequence[int] | None = None,
) -> list[tuple[int, Mapping[str, Any], Mapping[str, Any], int]]:
    """(bus, previous, update, gap) for each update on one of
    ``limits.update_buses`` that starts less than its *cycles* (by bus;
    else its ``cycles_per_update``) after the update before it on the bus,
    on one path, in time order on each bus; *events* in cycles of
    *clock_ghz*, of a program on quantum system *target*."""
    # A controller that lists its lines plays only those.
    plays = {(line._role, q) for line in limits._all_lines for q in line.qudits}
    bus_of = {}
    for line in lines:
        handle = (line["role"], line["qudit"])
        if line["qudit"] < 0 and unresolved is not None:
            # Which bus a runtime qudit's line is on, each call says.
            unresolved.append("update buses")
            return []
        if plays and handle not in plays:
            continue
        for index, bus in enumerate(limits.update_buses):
            if handle in bus._lines:
                bus_of[line["line"]] = index
    if not bus_of:
        return []
    updates = _updates(target, events, lines, bus_of, clock_ghz, unresolved)
    if updates is None:
        return []
    if cycles is None:
        cycles = [bus.cycles_per_update for bus in limits.update_buses]
    by_bus: list[list[Mapping[str, Any]]] = [[] for _ in limits.update_buses]
    for e in updates:
        by_bus[bus_of[e["line"]]].append(e)
    found = []
    for index, on_bus in enumerate(by_bus):
        for previous, event in _pairs(on_bus):
            gap = event["start_cycles"] - previous["start_cycles"]
            if gap < cycles[index]:
                found.append((index, previous, event, gap))
    return found


# A rolled loop of more trips is played at iterations 0, 1, 2 and its last
# two only (``_bus_fast``).
_SUMMARY_TRIPS = 5


def _bus_fast(lowered: ScheduledKernel, module: Any, limits: Controller) -> list[str]:
    """``_bus_violations`` on one clock, on the format /2 walk of *module*,
    each loop it keeps rolled (each tone's frequency the same in every
    iteration, its phase one step on) played at iterations 0, 1, 2 and its
    last two. Whether a pulse is an update depends only on the pulse before
    it on its line, and a clash only on the update before it on its bus;
    from iteration 1 on, both are those of the iteration before, shifted by
    one period and one phase step, so iterations 1 and 2 clash as every
    later one does, iteration 0 is the entry and the last two what follows
    the loop follows. It finds a violation if any execution has one; a
    caller lists them all from a walk of every execution."""
    with lowered._lock:
        found_lines, events, _ = module.controller_program(
            _PARAMETERS, _AMPLITUDES, False, True, _Waveforms(), exportable=False
        )
    lines = [{"line": i, "role": r, "qudit": q} for i, (r, q) in enumerate(found_lines)]
    played: list[dict[str, Any]] = []
    _copies(events, played, 0, ())
    clock_ghz = lowered.quantum_system.clock_ghz
    clashes = _bus_clashes(lowered.quantum_system, played, limits, lines, clock_ghz, None)
    return [_clash_text(limits, *clash) for clash in clashes]


def _copies(
    events: Sequence[Mapping[str, Any]],
    out: list[dict[str, Any]],
    shift: int,
    trips: tuple[int, ...],
) -> None:
    """The format /2 *events* as format /1 events into *out*, each rolled
    loop at the iterations ``_bus_fast`` plays: shifted *shift* cycles, at
    iteration *trips* of each rolled loop around them (outermost first)."""
    for event in events:
        kind = event["kind"]
        if kind == "loop":
            count = event["trips"]
            picked: Iterable[int] = (
                range(count) if count <= _SUMMARY_TRIPS else (0, 1, 2, count - 2, count - 1)
            )
            for k in picked:
                _copies(event["body"], out, shift + k * event["period_cycles"], (*trips, k))
            continue
        start = event.get("start_cycles")
        copy: dict[str, Any] = {
            "kind": _FLAT_KINDS.get(kind, kind),
            "index": len(out),
            "line": event.get("line"),
            "start_cycles": None if start is None else start + shift,
            "conditions": event.get("conditions", []),
            "iteration": _iteration(event.get("iteration", []), trips),
        }
        duration = event.get("duration_cycles")
        copy["end_cycles"] = None if start is None or duration is None else start + shift + duration
        tones = event.get("tones")
        if tones is not None:
            tone = tones[0] if len(tones) == 1 else {}
            phase = tone.get("phase_rad")
            if phase is not None:
                steps = tone.get("phase_step_rad", ())
                phase += sum(k * step for k, step in zip(trips, steps, strict=False))
            copy["frequency_hz"] = tone.get("frequency_hz")
            copy["phase_rad"] = phase
        out.append(copy)


def _iteration(given: Sequence[int | None], copies: tuple[int, ...]) -> list[int]:
    """An event's iteration in format /2, a rolled loop's ``None``, with the
    rolled loops' iterations *copies* in their places."""
    rolled = iter(copies)
    return [next(rolled, 0) if k is None else k for k in given]


def _updates(
    target: QuantumSystem,
    events: Sequence[Mapping[str, Any]],
    lines: Sequence[Mapping[str, Any]],
    bus_of: Mapping[int, int],
    clock_ghz: float,
    unresolved: list[str] | None,
) -> list[Mapping[str, Any]] | None:
    """The events on the lines of *bus_of* that update their line's tone:
    each frame update, and each drive or readout whose frequency, or whose
    phase other than as its tone runs on, differs from that of the drive or
    readout before it on its line, on one path; the first on a line is set
    before the shot. A frequency or phase known only at run time is an
    update. With *unresolved*, ``None`` when one waits for the call's
    values, after noting it there."""
    played: dict[int, list[Mapping[str, Any]]] = {}
    found = []
    for event in events:
        line = event.get("line")
        if line is None or line not in bus_of:
            continue
        if event["kind"] == "frame_update":
            found.append(event)
        elif event["kind"] in _PLAYED and event["end_cycles"] is not None:
            played.setdefault(line, []).append(event)
            if unresolved is not None and None in (event["frequency_hz"], event["phase_rad"]):
                unresolved.append("update buses")
                return None
    reached = [lines[line]["qudit"] for line in played]
    _, drive, readout = target._frequency_tables(max(reached, default=-1) + 1)
    for line, items in played.items():
        qudit = lines[line]["qudit"]
        table = drive if lines[line]["role"] == "drive" else readout
        native = None if qudit < 0 else table[qudit]
        found.extend(
            event
            for previous, event in _pairs(items)
            if _retunes(previous, event, native, clock_ghz)
        )
    return found


def _retunes(
    previous: Mapping[str, Any], event: Mapping[str, Any], native: float | None, clock_ghz: float
) -> bool:
    """Whether drive or readout *event* plays another tone than *previous*,
    the one before it on its line: another frequency, or a phase the tone
    does not reach by running on. A tone at its frame's frequency keeps its
    phase; one retuned from its line's *native* frequency gains phase at the
    detuning."""
    before, after = previous["frequency_hz"], event["frequency_hz"]
    if before is None or after is None or before != after:
        return True
    if previous["phase_rad"] is None or event["phase_rad"] is None:
        return True
    turned = event["phase_rad"] - previous["phase_rad"]
    if _same_angle(turned, 0.0):
        return False
    if native is None:
        return True
    seconds = (event["start_cycles"] - previous["start_cycles"]) / clock_ghz * 1e-9
    return not _same_angle(turned, 2.0 * math.pi * (after - native) * seconds)


def _same_angle(a: float, b: float) -> bool:
    off = math.remainder(a - b, 2.0 * math.pi)
    return abs(off) < _PHASE_TOLERANCE


def _played_while_out(
    events: Sequence[Mapping[str, Any]],
    calls: Sequence[ClassicalCall],
    holds: Callable[[int, Mapping[str, Any]], bool] | None = None,
) -> list[tuple[Mapping[str, Any], ClassicalCall, Mapping[str, Any]]]:
    """Each drive or readout of *events* (in program cycles) that plays
    while one of *calls* is out, from its submission until its answer is
    due, with the call's submit event and the call; with *holds*, only
    those ``holds(i, event)`` gives for the call that is submit *i*."""
    import bisect

    played = sorted(
        (e for e in events if e["kind"] in _PLAYED and e["end_cycles"] is not None),
        key=lambda e: e["start_cycles"],
    )
    if not played:
        return []
    starts = [e["start_cycles"] for e in played]
    longest = max(e["end_cycles"] - e["start_cycles"] for e in played)
    submits = [e for e in events if e["kind"] == "submit"]
    found = []
    for index, (submit, call) in enumerate(zip(submits, calls, strict=True)):
        sent, due = submit["start_cycles"], call.due_cycles
        for k in range(bisect.bisect_left(starts, sent - longest), bisect.bisect_left(starts, due)):
            event = played[k]
            if (
                event["end_cycles"] > sent
                and not _exclusive(submit, event)
                and (holds is None or holds(index, event))
            ):
                found.append((submit, call, event))
    return found


def _held_text(submit: Mapping[str, Any], call: ClassicalCall, event: Mapping[str, Any]) -> str:
    return (
        f"line {event['line']}: the {event['kind']} at {event['start_cycles']} "
        f"plays while the call to {submit['function']} sent at {submit['start_cycles']} is out, "
        f"until its answer is due at {call.due_cycles}; the controller blocks on calls "
        "(Controller.blocks_on_calls)"
    )


def _over_amplitude(
    line: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    most: float,
    unresolved: list[str] | None = None,
    peaks: dict[str, float | None] | None = None,
) -> list[str]:
    """The events of *line* whose envelope peaks above *most*. With
    *unresolved*, an envelope whose peak needs the call's values is noted
    there, to be checked with them. Each distinct envelope is sampled once,
    once for all the lines that share *peaks*."""
    found = []
    if peaks is None:
        peaks = {}
    for event in events:
        tree = event.get("waveform")
        key = repr(tree)
        if key not in peaks:
            peaks[key] = _peak_amplitude(tree)
        peak = peaks[key]
        if peak is None and tree is not None and unresolved is not None:
            unresolved.append(f"line {line['line']}: amplitude")
            unresolved = None
        if peak is not None and peak > most:
            found.append(
                f"line {line['line']} ({line['role']} of qudit {line['qudit']}): the "
                f"{event['kind']} at {event['start_cycles']} peaks at amplitude {peak:.6g}; "
                f"the controller plays at most {most:g} (Controller.max_amplitude)"
            )
    return found


def _amplitude_violations(
    lowered: ScheduledKernel, module: Any, most: float, unresolved: list[str] | None
) -> list[str]:
    """``_over_amplitude`` on one clock, on the format /2 walk of *module*:
    each distinct waveform is sampled once, and a loop whose iterations
    repeat is checked once. With *unresolved*, a waveform a float parameter
    sets is noted there instead, to be checked with each call's values."""
    table = _Waveforms()
    with lowered._lock:
        lines, events, _ = module.controller_program(
            _PARAMETERS, _AMPLITUDES, lowered.classical != (), True, table, exportable=False
        )
    peaks: dict[int, float | None] = {}
    by_line: dict[int, list[str]] = {}
    for event in _played(events):
        index = event["waveform"]
        if index not in peaks:
            peaks[index] = _peak_amplitude(table.table[index])
            if peaks[index] is None and unresolved is not None and _unset(table.table[index]):
                unresolved.append(f"line {event['line']}: amplitude")
        peak = peaks[index]
        if peak is not None and peak > most:
            role, qudit = lines[event["line"]]
            by_line.setdefault(event["line"], []).append(
                f"line {event['line']} ({role} of qudit {qudit}): the "
                f"{_FLAT_KINDS.get(event['kind'], event['kind'])} at {event['start_cycles']} "
                f"peaks at amplitude {peak:.6g}; the controller plays at most {most:g} "
                "(Controller.max_amplitude)"
            )
    return [message for _, messages in sorted(by_line.items()) for message in messages]


def _checks_pulse_lengths(limits: Controller) -> bool:
    """Whether *limits* bound the length of a drive or readout."""
    return bool(limits.min_pulse_cycles) or limits.max_pulse_cycles is not None


def _pulse_length(
    line: Mapping[str, Any], kind: str, duration: int, limits: Controller
) -> tuple[str, str] | None:
    """The violation of a *kind* of *duration* cycles on *line* shorter than
    ``limits.min_pulse_cycles`` or longer than ``limits.max_pulse_cycles``,
    before and after its start; ``None`` when its length is allowed."""
    if duration < limits.min_pulse_cycles:
        bound = f"none shorter than {limits.min_pulse_cycles} (Controller.min_pulse_cycles)"
    elif limits.max_pulse_cycles is not None and duration > limits.max_pulse_cycles:
        bound = f"none longer than {limits.max_pulse_cycles} (Controller.max_pulse_cycles)"
    else:
        return None
    return (
        f"line {line['line']} ({line['role']} of qudit {line['qudit']}): the {kind} at ",
        f" lasts {duration} cycles; the controller plays {bound}",
    )


def _pulse_length_violations(
    lowered: ScheduledKernel, module: Any, limits: Controller
) -> list[str]:
    """``_pulse_length`` on one clock, on the format /2 walk of *module*: a
    loop whose iterations repeat is checked once."""
    with lowered._lock:
        lines, events, _ = module.controller_program(
            _PARAMETERS, _AMPLITUDES, lowered.classical != (), True, _Waveforms(), exportable=False
        )
    found = []
    for event in _played(events):
        duration = event["duration_cycles"]
        if duration is None:
            continue
        role, qudit = lines[event["line"]]
        line = {"line": event["line"], "role": role, "qudit": qudit}
        kind = _FLAT_KINDS.get(event["kind"], event["kind"])
        violation = _pulse_length(line, kind, duration, limits)
        if violation is not None:
            found.append(f"{violation[0]}{event['start_cycles']}{violation[1]}")
    return found


def _unset(node: Mapping[str, Any]) -> bool:
    """Whether a value of waveform *node* waits for a float argument."""
    return None in node.get("params", {}).values() or any(
        _unset(child) for child in node.get("of", ())
    )


# Samples per cycle the peak of an envelope is found at.
_PEAK_SAMPLES_PER_CYCLE = 8


def _peak_amplitude(node: Mapping[str, Any] | None) -> float | None:
    """The largest magnitude of the envelope of *node*, over its own
    duration; ``None`` when it is not built from numbers alone."""
    if node is None:
        return None
    duration = _span(node)
    if duration is None:
        return None
    try:
        samples = sample_waveform(node, math.ceil(duration), _PEAK_SAMPLES_PER_CYCLE)
    except (ValueError, TypeError, KeyError):
        return None
    return float(abs(samples).max()) if len(samples) else 0.0


def _span(node: Mapping[str, Any]) -> float | None:
    """The duration, in cycles, a waveform tree states for itself."""
    duration = node.get("params", {}).get("duration")
    if duration is not None:
        return float(duration)
    if node["shape"] == "custom_samples":
        return float(len(node.get("samples", ())) or 1)
    for child in node.get("of", ()):
        found = _span(child)
        if found is not None:
            return found
    return None


def _samples_per_own_cycle(limits: Controller, program_ghz: float) -> tuple[int, int]:
    """The controller's samples per cycle of its clock, as (num, den)."""
    if limits.sample_rate_ghz is None:
        return 1, 1
    from . import _clocks

    return _clocks.made_for(limits.sample_rate_ghz, limits.clock_ghz or program_ghz)


def _event_count(timed: Sequence[Mapping[str, Any]], most: int, runs_loops: bool) -> list[str]:
    """The violation of ``max_pulses`` (*most*) by *timed*, which exceeds it
    unrolled; a controller that *runs_loops* holds a rolled body once."""
    if runs_loops:
        held = sum(1 for e in timed if not any(e["iteration"]))
        if held <= most:
            return []
        return [
            f"{held} drives and readouts, a rolled loop's body counted once; the "
            f"controller holds at most {most}"
        ]
    text = f"{len(timed)} drives and readouts; the controller plays at most {most}"
    loops = _rolled_loops(timed)
    if loops:
        grown = "; ".join(
            f"the loop of {trips} iterations from cycle {start} plays {extra} more"
            for trips, start, extra in loops
        )
        text += f". It does not support 'loop', so rolled loops are unrolled for it: {grown}"
    return [text]


def _rolled_loops(timed: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, int]]:
    """Each outermost rolled loop of *timed* (in program order): its trip
    count, its first start cycle and the events past its first iteration."""
    loops: list[list[int]] = []
    previous = -1
    for e in timed:
        iteration = e["iteration"]
        if not iteration:
            previous = -1
            continue
        if previous < 0 or iteration[0] < previous:
            loops.append([0, e["start_cycles"], 0])
        loop = loops[-1]
        loop[0] = max(loop[0], iteration[0] + 1)
        loop[2] += any(iteration)
        previous = iteration[0]
    return [(trips, start, extra) for trips, start, extra in loops if extra]


def check(lowered: ScheduledKernel, limits: Controller | None = None) -> Sequence[str]:
    """The limits *lowered* exceeds, of *limits* or else its controller's:
    a list, or past 10,000 violations in rolled loops a sequence that builds
    each one when read (``ControllerLimitError.violations``).

    Of a module with float parameters (``lowered.generic``), the limits it
    exceeds whatever their values; a call's values are checked when it is
    made, so a limit never fails first at export. A shared line of *limits*
    is checked as compiling for it checks it (the tones it plays at once).
    """
    machine = lowered.options.machine
    limits = limits if limits is not None else machine.controller
    # The drives and readouts at once its control system was given, when
    # checked for that system's controller (or controllers).
    over: list[str] = []
    if limits is machine.controller and (
        machine.controller is not None or machine._system is not None
    ):
        from .compiler import _over_concurrency

        over = _over_concurrency(lowered)[0]
    if limits is None:
        return over
    found = _compile_check(lowered, limits)[0]
    if any(len(line) > 2 for line in limits._all_lines):
        from ._shared_lines import against

        found = [*found, *against(lowered, limits)]
    return [*found, *over] if over else found


def _compile_check(
    lowered: ScheduledKernel, limits: Controller | None, system: ControlSystem | None = None
) -> tuple[Sequence[str], bool]:
    """The limits *lowered* exceeds whatever its float values, and whether
    that is exact: ``False`` when some values may exceed others. With
    *system*, of its controllers (*limits* is then its only one, or ``None``)."""
    if system is not None and system._one_box:
        found, exact = _compile_check(lowered, limits)
        if not system._placement:
            return found, exact
        if lowered.parameters or (lowered._role is not None and lowered._role.slots > 0):
            # When calls are due can depend on a call's values.
            return found, False
        from ._controller_cost import _usage

        extra = _usage(lowered).violations
        return ([*found, *extra] if extra else found), exact
    machine = lowered.options.machine
    if limits is machine.controller:
        clocks = (lowered._generic or lowered).clocks
    else:
        from . import _clocks

        clocks = _clocks.plan(limits, lowered.quantum_system.clock_ghz, machine.reference_clock_ghz)
    slots = lowered._role is not None and lowered._role.slots > 0
    if system is None and not slots:
        assert limits is not None
        maybe: list[str] = []
        if _checks_natively(limits, clocks) and not _found_natively(
            lowered, limits, maybe if lowered.parameters else None
        ):
            return [], not maybe
    # Held lines are checked call by call, every iteration walked.
    repeats: list[tuple[int, ...]] | None = (
        None if limits is not None and _walks_every(lowered, limits, clocks) else []
    )
    # Loops of lines of different periods summarized too, then not, then none.
    passes: tuple[bool | None, ...] = (True, False, None)
    if system is not None:
        if (
            repeats is None
            or not system._lines_up
            or slots
            and not system._plays_everything
            or any(c.update_buses for c in system.controllers)
        ):
            repeats = None
        elif clocks is not None and clocks.owners is not None:
            # A controller counting cycles of its own is walked in full.
            if any(c.ratio != 1 for c in clocks.controllers):
                repeats = None
    # The copies of rolled loops the next walk plays one by one, as far as
    # the walk before it shows: past those a check plays, it stops here.
    walked = every = 0
    for unsynced in passes:
        periods: list[tuple[int, int, int]] | None = [] if unsynced else None
        if walked:
            from ._loops import refuse_past_budget

            refuse_past_budget(walked)
        lines, events, _, off, problems = _walk(
            lowered,
            values=False,
            own=_native_clock(clocks),
            waveforms=False,
            summary=None if repeats is None else (repeats, periods),
        )
        if not repeats:
            break
        # A summarizing walk's problems name only the copies it plays.
        if not problems:
            from ._loops import summarized_check

            checked = summarized_check(
                lowered, limits, system, clocks, slots, lines, events, repeats, periods, off
            )
            if checked is not None:
                return checked
        from ._loops import walked_copies

        again = unsynced and not problems
        # What a walk of every iteration plays, from the walk in which every
        # loop still is one; the next summarizes those of one period, or none.
        every = max(every, walked_copies(events, repeats, periods or ()))
        walked = walked_copies(events, repeats, periods or (), unsynced=True) if again else every
        repeats = [] if again else None
    if system is not None and slots and not system._plays_everything:
        # Which controller plays a line depends on the qudits a call binds.
        return [], False
    unresolved: list[str] | None = [] if lowered.parameters or slots else None
    natives = None
    if slots and limits is not None and limits.max_frequencies_per_line is not None:
        with lowered._lock:
            natives = lowered._module.controller_native_tones()
    if system is None:
        assert limits is not None
        found = _violations(
            lowered, events, limits, lines, clocks, off, problems, False, unresolved, natives
        )
    else:
        found = system._violations(
            lowered, lines, events, clocks, off, problems, False, unresolved, natives
        )
    if slots and limits is not None and limits.detectors:
        assert unresolved is not None
        # Acquirers are given per carrier, which each call binds.
        unresolved.append("detectors")
    if limits is not None and limits is not machine.controller:
        found.extend(_sample_rate_violations(lowered, limits, clocks, events))
    return found, not unresolved


def _call_violations(lowered: ScheduledKernel) -> Sequence[str]:
    """The limits one call's values exceed, as its export checks them."""
    from .compiler import _over_concurrency

    found = _system_call_violations(lowered)
    over = _over_concurrency(lowered)[0]
    return [*found, *over] if over else found


def _system_call_violations(lowered: ScheduledKernel) -> Sequence[str]:
    """``_call_violations`` but the drives and readouts at once."""
    machine = lowered.options.machine
    clocks = (lowered._generic or lowered).clocks
    system = machine._system
    if system is not None and system._one_box:
        assert machine.controller is not None
        found = _limit_violations(lowered, machine.controller, clocks)
        if not system._placement:
            return found
        from ._controller_cost import _usage

        extra = _usage(lowered).violations
        return [*found, *extra] if extra else found
    if system is not None:
        repeats: list[tuple[int, ...]] | None = []
        if (
            not system._lines_up
            or system._blocking
            or any(c.update_buses for c in system.controllers)
        ):
            repeats = None
        elif clocks is not None and clocks.owners is not None:
            # A controller counting cycles of its own is walked in full.
            if any(c.ratio != 1 for c in clocks.controllers):
                repeats = None
        # Summarized while the system's controllers line up, loops of lines
        # of different periods too, then not, else every iteration.
        periods: list[tuple[int, int, int]] | None = []
        while True:
            lines, events, _, off, problems = _walk(
                lowered,
                own=_native_clock(clocks),
                summary=None if repeats is None else (repeats, periods),
            )
            if not repeats:
                break
            from ._loops import finished

            # A summarizing walk's problems name only the copies it plays.
            if not problems:
                parts = system._summarized(
                    lowered, lines, events, repeats, clocks, off, None, None, None, periods or ()
                )
                if parts is not None:
                    return finished(parts)
            if periods is not None and not problems:
                repeats, periods = [], None
            else:
                repeats = None
        return system._violations(lowered, lines, events, clocks, off, problems, True, None, None)
    assert machine.controller is not None
    return _limit_violations(lowered, machine.controller, clocks)


def _sample_rate_violations(
    lowered: ScheduledKernel,
    limits: Controller,
    clocks: _clocks.ClockPlan | None,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """The sample arrays that *limits*' lines would play at another rate than
    the one they were made for: the rate of the lines *lowered* was compiled
    for. *events* are its events, else walked here."""
    rates = _sample_rates(lowered, limits)
    if rates is None:
        return []
    made_ghz, plays_ghz = rates
    if events is None:
        events = _walk(lowered, values=False, own=_native_clock(clocks), waveforms=False)[1]
    return [
        f"line {e['line']}: the {e['kind']} at {e['start_cycles']} plays samples made for "
        f"{made_ghz:g} GHz, and the controller's lines play {plays_ghz:g} GHz"
        for e in events
        if e["kind"] in _PLAYED and _has_samples(e.get("waveform"))
    ]


def _sample_rates(lowered: ScheduledKernel, limits: Controller) -> tuple[float, float] | None:
    """The rate, in GHz, the samples of *lowered* were made for and the one
    *limits*' lines play, when they differ."""
    from . import _clocks

    machine = lowered.options.machine
    made = None
    program = lowered.quantum_system.clock_ghz
    if machine.controller is not None:
        made = _clocks.sample_rate(machine.controller, program)
    plays = _clocks.sample_rate(limits, program)
    if made == plays:
        return None
    made_ghz = program if made is None else program * made[0] / made[1]
    plays_ghz = program if plays is None else program * plays[0] / plays[1]
    return made_ghz, plays_ghz


def _has_samples(node: Mapping[str, Any] | None) -> bool:
    if node is None:
        return False
    return node["shape"] == "custom_samples" or any(_has_samples(c) for c in node.get("of", ()))


# ── pulse.controller(...) ────────────────────────────────────────────────────


if TYPE_CHECKING:
    _Limits = Controller
    _Usage = UsageReport


class ControllerNames(Protocol):
    """The names this module exports, as attributes: what a module
    satisfies of ``ControllerModule``, which it can't be called as."""

    FORMAT: str
    ControllerLimitError: type[ControllerLimitError]
    Event: type[Event]
    FunctionCost: type[FunctionCost]
    LimitExceeded: type[LimitExceeded]
    LinkUsage: type[LinkUsage]
    ControllerUsage: type[ControllerUsage]
    FunctionUsage: type[FunctionUsage]
    ProcessorUsage: type[ProcessorUsage]
    UsageReport: type[UsageReport]
    WaveformMemoryScope: type[WaveformMemoryScope]
    if TYPE_CHECKING:
        DriveLine: type[DriveLine]
        ReadoutLine: type[ReadoutLine]
        UpdateBus: type[UpdateBus]
        ExpressionTerm: type[ExpressionTerm]
        OpClass: type[OpClass]
        OpCounts: type[OpCounts]
        OpCycles: type[OpCycles]

    def check(self, lowered: ScheduledKernel, limits: _Limits | None = None) -> Sequence[str]: ...
    def evaluate_expression(
        self,
        expression: Any,
        readouts: Mapping[int, int] | None = None,
        answers: Mapping[int, Sequence[float | int | bool]] | None = None,
    ) -> float | int | bool: ...
    def export(
        self,
        lowered: ScheduledKernel,
        sample_rate_ghz: float | None = None,
        controller: int | None = None,
        version: int = 1,
    ) -> dict[str, Any]: ...
    def sample_waveform(
        self, node: Mapping[str, Any], duration_cycles: int, samples_per_cycle: float = 1
    ) -> npt.NDArray[np.complex128]: ...
    def shape_key(self, node: Mapping[str, Any] | None) -> tuple[Any, ...] | None: ...
    def unroll(self, program: Mapping[str, Any]) -> dict[str, Any]: ...
    def usage(self, lowered: ScheduledKernel) -> _Usage: ...


class ControllerModule(ControllerNames, Protocol):
    """What type checkers take ``pulse.controller`` to be: the loader of
    controller descriptions, with this module's names as attributes.

    ``pulse.controller("vendor.model", drive_lines=..., readout_lines=...)``
    is the ``Controller`` an installed description (or a TOML file, by path)
    states, with the installation's wiring; each file is parsed once. A
    module can't be declared callable, so ``cudaq_pulse`` declares its
    ``controller`` attribute with this type for checkers only; at run time
    it is this module.
    """

    def __call__(self, description: str | os.PathLike[str], /, **wiring: Any) -> Controller: ...


class _Loaded(str):
    """The name of a ``_controller_cost`` attribute (or of another module's,
    by ``_ELSEWHERE``), imported on first use and then kept in this module's
    namespace, where later reads find it as a plain attribute. Being its own
    name, it is built without a call into Python."""

    def __get__(self, instance: types.ModuleType | None, owner: type) -> Any:
        if instance is None:
            return self
        name = str(self)
        module = _ELSEWHERE.get(name)
        if module is None:
            import cudaq_pulse._controller_cost as reports
        else:
            import importlib

            reports = importlib.import_module(module)
        value = instance.__dict__[name] = getattr(reports, name)
        return value


# The names _Loaded imports from elsewhere than _controller_cost: building
# an enum costs more than the rest of an export.
_ELSEWHERE = {
    "DriveLine": "cudaq_pulse._system",
    "ReadoutLine": "cudaq_pulse._system",
    "UpdateBus": "cudaq_pulse._system",
    "ExpressionTerm": "cudaq_pulse._op_classes",
    "OpClass": "cudaq_pulse._op_classes",
    "OpCounts": "cudaq_pulse._op_tables",
    "OpCycles": "cudaq_pulse._op_tables",
}


class _ControllerModule(types.ModuleType):
    """This module, callable as ``pulse.controller(description, **wiring)``."""

    def __call__(self, description: str | os.PathLike[str], /, **wiring: Any) -> Controller:
        from ._system import load_controller

        return load_controller(description, **wiring)

    def __dir__(self) -> list[str]:
        loaded = [
            name for name, value in vars(_ControllerModule).items() if isinstance(value, _Loaded)
        ]
        return sorted({*super().__dir__(), *loaded})

    # An export builds no report and analyzes no function: building their
    # dataclasses and the analyzer costs more than the export itself.
    if TYPE_CHECKING:
        # ControllerModule's attributes are settable, and a descriptor with
        # no __set__ is not, to a checker.
        FunctionCost: Any
        ControllerUsage: Any
        FunctionUsage: Any
        LimitExceeded: Any
        LinkUsage: Any
        ProcessorUsage: Any
        UsageReport: Any
        usage: Any
        ExpressionTerm: Any
        OpClass: Any
        OpCounts: Any
        OpCycles: Any
        DriveLine: Any
        ReadoutLine: Any
        UpdateBus: Any

    FunctionCost = _Loaded("FunctionCost")
    ControllerUsage = _Loaded("ControllerUsage")
    FunctionUsage = _Loaded("FunctionUsage")
    LimitExceeded = _Loaded("LimitExceeded")
    LinkUsage = _Loaded("LinkUsage")
    ProcessorUsage = _Loaded("ProcessorUsage")
    UsageReport = _Loaded("UsageReport")
    usage = _Loaded("usage")
    ExpressionTerm = _Loaded("ExpressionTerm")
    OpClass = _Loaded("OpClass")
    OpCounts = _Loaded("OpCounts")
    OpCycles = _Loaded("OpCycles")
    DriveLine = _Loaded("DriveLine")
    ReadoutLine = _Loaded("ReadoutLine")
    UpdateBus = _Loaded("UpdateBus")


sys.modules[__name__].__class__ = _ControllerModule
