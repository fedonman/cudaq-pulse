# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durations while tracing: cycles, or times with units converted to cycles.

A duration (a waveform's length or flat top, a wait and its bounds, a
``round_duration`` length) is an int number of cycles or a ``pulse.Time``; a
plain float there is ambiguous and an error. A waveform's shape times
(``sigma``, DRAG ``beta``) are cycles, fractional ones too, or times.

A time converts exactly to cycles of the program clock; a duration that is
not a whole number of them is an error. In an expression known only when
the program runs, ``f.result() * pulse.ns``, each time becomes its number of
cycles: one multiply by a constant the compile fixes.
"""

from __future__ import annotations

import math
import numbers
import operator
from collections.abc import Mapping
from typing import Any

from .._units import Time, UnitError, in_cycles
from .ir_builder import (
    ARITHMETIC,
    MAX_CYCLES,
    CompilationError,
    IRValue,
    ParameterExpression,
    arithmetic_text,
    expression_leaves,
    expression_nodes,
    is_symbolic,
)
from .packed_ir_builder import MAX_CUSTOM_SAMPLES

# The times of each op: (attribute, whole cycles, the call with it as {}).
TIMES: dict[str, tuple[tuple[str, bool, str], ...]] = {
    "gaussian": (("duration", True, "gaussian({}, ...)"), ("sigma", False, "")),
    "square": (("duration", True, "square({}, ...)"),),
    "drag": (
        ("duration", True, "drag({}, ...)"),
        ("sigma", False, ""),
        ("beta", False, ""),
    ),
    "cosine": (("duration", True, "cosine({}, ...)"),),
    "tanh_ramp": (("duration", True, "tanh_ramp({}, ...)"), ("sigma", False, "")),
    "gaussian_square": (
        ("duration", True, "gaussian_square({}, ...)"),
        ("sigma", False, ""),
        ("width", True, "gaussian_square(duration, amplitude, sigma, {})"),
    ),
    "custom": (("duration", True, "custom({}, ...)"),),
    "wait": (
        ("duration", True, "wait(line, {})"),
        ("bounds", True, "wait(line, duration, bounds=(0, {}))"),
    ),
}


# The fewest cycles of each whole-cycle time: a waveform plays for at least
# one cycle; a wait, a flat top and a bound may be 0.
LEAST: dict[tuple[str, str], int] = {
    (fname, name): int(name == "duration" and fname != "wait")
    for fname, spec in TIMES.items()
    for name, whole, _ in spec
    if whole
}


class Cycles(float):
    """The value of a ``float`` parameter the kernel is compiled for: a
    number of cycles, as the parameter is when the program computes with it,
    so a duration takes it, and what is computed from it, like a run-time
    length."""

    __slots__ = ()

    def _keep(self, value: Any) -> Any:
        return Cycles(value) if value.__class__ is float else value

    def _with(self, other: Any) -> bool:
        if other.__class__ is Time:
            raise UnitError(
                f"{float(self)!r} {other!r}: a float parameter is a number of cycles, "
                "and mixes with a time only multiplied by one (tau * pulse.ns)"
            )
        return isinstance(other, (int, float)) and other.__class__ is not bool

    def __add__(self, other: Any) -> Any:
        return self._keep(float.__add__(self, other)) if self._with(other) else NotImplemented

    def __radd__(self, other: Any) -> Any:
        return self._keep(float.__radd__(self, other)) if self._with(other) else NotImplemented

    def __sub__(self, other: Any) -> Any:
        return self._keep(float.__sub__(self, other)) if self._with(other) else NotImplemented

    def __rsub__(self, other: Any) -> Any:
        return self._keep(float.__rsub__(self, other)) if self._with(other) else NotImplemented

    def __mul__(self, other: Any) -> Any:
        if other.__class__ is Time:
            return other * float(self)
        return self._keep(float.__mul__(self, other)) if self._with(other) else NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        return self._keep(float.__truediv__(self, other)) if self._with(other) else NotImplemented

    def __rtruediv__(self, other: Any) -> Any:
        return self._keep(float.__rtruediv__(self, other)) if self._with(other) else NotImplemented

    def __floordiv__(self, other: Any) -> Any:
        return self._keep(float.__floordiv__(self, other)) if self._with(other) else NotImplemented

    def __rfloordiv__(self, other: Any) -> Any:
        return self._keep(float.__rfloordiv__(self, other)) if self._with(other) else NotImplemented

    def __mod__(self, other: Any) -> Any:
        return self._keep(float.__mod__(self, other)) if self._with(other) else NotImplemented

    def __neg__(self) -> Cycles:
        return Cycles(-float(self))

    def __pos__(self) -> Cycles:
        return self

    def __abs__(self) -> Cycles:
        return Cycles(abs(float(self)))


# Errors name a value's type; this is the float the user passed.
Cycles.__name__ = Cycles.__qualname__ = "float"


def in_program_cycles(fname: str, attrs: dict[str, Any], clock_ghz: float) -> None:
    """Replace the times in *attrs*, the attributes of a *fname* call, by
    cycles of the *clock_ghz* program clock; reject plain floats in its
    durations."""
    for name, whole, call in TIMES[fname]:
        value = attrs.get(name)
        if value is None:
            continue
        if name == "bounds":
            if isinstance(value, (tuple, list)) and not _whole(value):
                attrs[name] = type(value)(
                    v if v is None else cycles(fname, name, v, True, call, clock_ghz) for v in value
                )
            continue
        n = value
        if n.__class__ is not int:
            n = attrs[name] = cycles(fname, name, value, whole, call, clock_ghz)
            if not whole:
                continue
            if n.__class__ is Cycles:
                # A float argument the kernel is compiled for, as a duration.
                if not LEAST[fname, name] <= n <= MAX_CYCLES:
                    raise _out_of_range(fname, name, value, n)
                continue
            if n.__class__ is bool or not isinstance(n, numbers.Integral):
                continue
            # An integer of another type, such as numpy's: its int value.
            n = attrs[name] = operator.index(n)
        if whole and not LEAST[fname, name] <= n <= MAX_CYCLES:
            raise _out_of_range(fname, name, value, n)


def _out_of_range(fname: str, what: str, value: Any, n: float) -> CompilationError:
    """The error for *value*, a duration of *n* cycles outside LEAST to MAX_CYCLES."""
    shown = f"{n:g}" if n.__class__ is Cycles else f"{n}"
    given = (
        f"{what} {value} is {shown} cycles"
        if value.__class__ is Time
        else f"{what} is {shown} cycles"
    )
    if n > MAX_CYCLES:
        rule = "a duration is at most 2**53 cycles"
    elif LEAST[fname, what]:
        rule = "a waveform lasts at least 1 cycle"
    elif fname == "wait":
        rule = "a wait cannot be negative"
    else:
        rule = "a duration cannot be negative"
    return CompilationError(f"{fname}(): {given}; {rule}", code="invalid_duration")


def _whole(bounds: tuple[Any, ...] | list[Any]) -> bool:
    for v in bounds:
        if v is not None and v.__class__ is not int:
            return False
    return True


def cycles(fname: str, what: str, value: Any, whole: bool, call: str, clock_ghz: float) -> Any:
    """*value*, the *what* of a *fname* call, in program cycles."""
    cls = value.__class__
    if cls is Time:
        numerator, denominator = in_cycles(value, clock_ghz)
        if not whole:
            return numerator / denominator
        if denominator != 1:
            raise CompilationError(
                f"{fname}(): {what} {value} is {numerator / denominator:g} cycles of the "
                f"{clock_ghz:g} GHz program clock, and durations are whole cycles; "
                f"round it with pulse.round_duration(line, {_written(value)})",
                code="invalid_duration",
            )
        return numerator
    if whole and isinstance(value, float) and cls is not Cycles:
        raise float_duration(fname, what, value, call)
    if is_symbolic(value) and _has_time(value):
        if unit(value, fname, what) != 1:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} is a number, not a time: "
                "a duration is a number of cycles or a time",
                code="invalid_duration",
            )
        return _replaced(value, clock_ghz)
    return value


def float_duration(
    fname: str, what: str, value: float, call: str, cycles_only: bool = False
) -> CompilationError:
    """The error for the plain float *value* in a duration; *cycles_only*
    for one whose name gives its unit, which takes no time."""
    shown = f"{value:g}"
    whole = str(round(value)) if value == value and abs(value) != float("inf") else shown
    in_cycles_call = call.format(whole)
    plural = "" if whole == "1" else "s"
    if cycles_only:
        return CompilationError(
            f"{fname}(): {what} is a float ({value!r}); use an int number of cycles:\n"
            f"    {in_cycles_call}    # {whole} cycle{plural}",
            code="invalid_duration",
        )
    in_ns_call = call.format(f"{shown} * pulse.ns")
    width = max(len(in_cycles_call), len(in_ns_call)) + 4
    return CompilationError(
        f"{fname}(): {what} is a float ({value!r}). Durations are an int number of "
        "cycles, or a time with units:\n"
        f"    {in_cycles_call.ljust(width)}# {whole} cycle{plural}\n"
        f"    {in_ns_call.ljust(width)}# {shown} nanoseconds",
        code="invalid_duration",
    )


def _written(value: Time) -> str:
    text = repr(value)
    number, unit_name = text.split()
    return f"{number} * pulse.{unit_name}"


def _has_time(value: Any) -> bool:
    if value.__class__ is Time:
        return True
    if value.__class__ is ParameterExpression:
        return any(leaf.__class__ is Time for leaf in expression_leaves(value, ARITHMETIC))
    return False


def unit(value: Any, fname: str, what: str) -> int:
    """1 if *value* is a time, 0 if a number; raises for arithmetic that
    mixes them."""
    if value.__class__ is Time:
        return 1
    if value.__class__ is not ParameterExpression:
        return 0
    # Each operation once, after its operands, as they are written.
    units: dict[int, int] = {}
    for node in expression_nodes(value, ARITHMETIC):
        lhs, rhs = node.lhs, node.rhs
        units[id(node)] = _node_unit(
            node,
            units[id(lhs)] if id(lhs) in units else int(lhs.__class__ is Time),
            units[id(rhs)] if id(rhs) in units else int(rhs.__class__ is Time),
            fname,
            what,
        )
    return units[id(value)]


def _node_unit(value: Any, left: int, right: int, fname: str, what: str) -> int:
    """``unit`` of the operation *value* whose operands have units *left*
    and *right*."""
    op = value.op
    if op in ("neg", "cast"):
        return left
    if op in ("add", "sub"):
        if left != right:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} mixes a time with a number "
                "of cycles; give both with units (n * pulse.ns) or both in cycles",
                code="invalid_duration",
            )
        return left
    if op == "mul":
        if left and right:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} is a time times a time",
                code="invalid_duration",
            )
        return left + right
    if op == "div":
        if right and not left:
            raise CompilationError(
                f"{fname}(): {what} {arithmetic_text(value)} is a number over a time",
                code="invalid_duration",
            )
        return left - right
    if left or right:
        raise CompilationError(
            f"{fname}(): {what} {arithmetic_text(value)} takes {op} of a time",
            code="invalid_duration",
            fix="use + - * / on times",
        )
    return 0


def _replaced(value: Any, clock_ghz: float) -> Any:
    if value.__class__ is Time:
        numerator, denominator = in_cycles(value, clock_ghz)
        return numerator / denominator
    if value.__class__ is ParameterExpression:
        made: dict[int, Any] = {}
        for node in expression_nodes(value, ARITHMETIC):
            lhs, rhs = node.lhs, node.rhs
            made[id(node)] = ParameterExpression(
                node.op,
                made[id(lhs)] if id(lhs) in made else _replaced(lhs, clock_ghz),
                made[id(rhs)] if id(rhs) in made else _replaced(rhs, clock_ghz),
                dtype=node.dtype,
            )
        return made[id(value)]
    return value


def timeout_ns(builder: Any, timeout: Any) -> float:
    """*timeout*, a time or a number of nanoseconds, in nanoseconds; it is
    whole cycles of the program clock."""
    if (
        is_symbolic(timeout)
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
    ):
        raise CompilationError(
            "result(timeout=...) takes a time known when compiling: "
            "a number of nanoseconds, or a time with units (350 * pulse.ns)",
            code="invalid_duration",
        )
    time = timeout if timeout.__class__ is Time else Time(timeout)
    if not time._n > 0:
        raise CompilationError(
            f"result(): timeout must be positive, got {timeout!r}", code="wrong_argument"
        )
    clock_ghz = getattr(builder, "clock_ghz", 2.0)
    cycles("result", "timeout", time, True, "f(...).result(timeout={})", clock_ghz)
    # Whole cycles of it are exactly its nanoseconds, which its float rounds.
    return float(time)


# ── Lines on other clocks ─────────────────────────────────────────────────────


def traced_round_duration(builder: Any, args: list[Any], kwargs: dict[str, Any] | None) -> int:
    """``round_duration(line, duration, direction)`` while tracing: a number,
    from the grid of the line's controller."""
    from .. import _clocks
    from ._emitter import _bound, _parameter_names
    from .ir_builder import IRValue, SymbolicValueError

    line, n, direction = _bound(_clocks.round_duration, args, kwargs or {}).args
    if not isinstance(line, IRValue) or line.vtype not in ("drive_line", "readout_line"):
        raise CompilationError(
            f"round_duration() takes a line from get_drive_line or get_readout_line, got {line!r}",
            code="wrong_argument",
        )
    if is_symbolic(n):
        names = tuple(dict.fromkeys(_parameter_names(n)))
        raise SymbolicValueError(
            "round_duration() needs a duration known while the kernel compiles; it "
            f"depends on {', '.join(map(repr, names))}",
            names,
            "a round_duration() duration",
        )
    if n.__class__ is Time:
        numerator, denominator = in_cycles(n, getattr(builder, "clock_ghz", 2.0))
        n = numerator / denominator
    elif isinstance(n, float) and n.__class__ is not Cycles:
        raise float_duration("round_duration", "duration", n, "pulse.round_duration(line, {})")
    grid_of = getattr(builder, "line_grid", None)
    grid = 1
    if grid_of is not None:
        qudit = getattr(builder, "_line_qudit", {}).get(line.vid, -1)
        grid = grid_of(line.vtype.removesuffix("_line"), qudit)
    try:
        return _clocks.round_on_grid(n, grid, direction)
    except ValueError as e:
        raise CompilationError(str(e), code="invalid_duration") from None


def at_rate(count: int, rate: tuple[int, int], envelope: Any) -> tuple[int, float, Any]:
    """For ``custom()`` over *count* cycles at the lines' *rate*: its number
    of samples, the cycles between them, and *envelope* of the sample times
    in those (``t = k + 0.5``) instead of cycles."""
    count = samples_at(count, rate)
    scale = rate[1] / rate[0]
    return count, scale, lambda t: envelope(t * scale)


def samples_at(cycles: int, rate: tuple[int, int]) -> int:
    """How many samples ``custom()`` takes over *cycles* at the lines' *rate*."""
    n, d = rate
    if cycles * n % d:
        step = d // math.gcd(n, d)
        raise CompilationError(
            f"custom(): a duration of {cycles} clock cycles is not a whole number of "
            f"samples on these lines, which play {n} every {d} cycles (their "
            f"controller's rate); give it a multiple of {step} cycles, as "
            f"pulse.round_duration(line, {cycles}) does",
            code="invalid_waveform",
        )
    count = cycles * n // d
    most = MAX_CUSTOM_SAMPLES - 2
    if count > most:
        raise CompilationError(
            f"custom(): a duration of {cycles} clock cycles is {count} samples at "
            f"these lines' rate ({n} every {d} cycles), over the limit of {most}; "
            "split the envelope into consecutive drives",
            code="limit_exceeded",
        )
    return count


def samples_made_for(
    samples: Any, sample_rate_ghz: Any, line_rate: tuple[int, int] | None, program_ghz: float
) -> tuple[int, int] | None:
    """The ``rate`` of ``custom_samples(samples, sample_rate_ghz)``: the
    rate it was made for (the program clock without *sample_rate_ghz*),
    which must be the lines' *line_rate*. ``None`` for one per cycle."""
    made = (1, 1)
    if sample_rate_ghz is not None:
        try:
            from .. import _clocks

            made = _clocks.made_for(sample_rate_ghz, program_ghz)
        except ValueError as e:
            raise CompilationError(f"custom_samples(): {e}", code="invalid_waveform") from None
    plays = line_rate or (1, 1)
    if made != plays:
        made_ghz = program_ghz * made[0] / made[1]
        plays_ghz = program_ghz * plays[0] / plays[1]
        count = len(samples) if isinstance(samples, (list, tuple)) else "the"
        raise CompilationError(
            f"custom_samples(): {count} samples made for {made_ghz:g} GHz "
            f"({_rate_text(made)}) on lines that play {plays_ghz:g} GHz "
            f"({_rate_text(plays)}), which would change how long they last; "
            f"resample them to {plays_ghz:g} GHz and pass "
            f"sample_rate_ghz={plays_ghz:g}, or build the waveform with "
            "custom(duration, f), which samples at the lines' rate",
            code="invalid_waveform",
        )
    if made == (1, 1):
        return None
    try:
        count = len(samples)
    except TypeError:
        return made
    if count * made[1] % made[0]:
        raise CompilationError(
            f"custom_samples(): {count} samples at {_rate_text(made)} do not last a "
            f"whole number of clock cycles; give a multiple of "
            f"{made[0] // math.gcd(*made)} samples",
            code="invalid_waveform",
        )
    return made


def _rate_ghz(rate: tuple[int, int] | None, program_ghz: float) -> float:
    return program_ghz if rate is None else program_ghz * rate[0] / rate[1]


def _rate_text(rate: tuple[int, int]) -> str:
    n, d = rate
    if d == 1:
        return "one sample per clock cycle" if n == 1 else f"{n} samples per clock cycle"
    return f"one sample every {d} clock cycles" if n == 1 else f"{n} samples every {d} cycles"


def custom_samples_duration(builder: Any, attrs: dict[str, Any]) -> int | None:
    """The duration in program cycles of a ``custom_samples`` op of *attrs*
    (its ``rate`` set) on lines that play *builder*'s sample rate."""
    line_rate = getattr(builder, "sample_rate", None)
    made_for = attrs.pop("sample_rate_ghz", None)
    if made_for is not None and getattr(builder, "sample_rate_of", None) is not None:
        # The line that plays them is checked when it does.
        from .. import _clocks

        try:
            n, d = _clocks.made_for(made_for, getattr(builder, "clock_ghz", 2.0))
        except ValueError as e:
            raise CompilationError(f"custom_samples(): {e}", code="invalid_waveform") from None
        line_rate = None if n == d else (n, d)
    rate = None
    if line_rate is not None or made_for is not None:
        rate = samples_made_for(
            attrs["samples"], made_for, line_rate, getattr(builder, "clock_ghz", 2.0)
        )
    try:
        if rate is None:
            return len(attrs["samples"])
        attrs["rate"] = rate
        return len(attrs["samples"]) * rate[1] // rate[0]
    except TypeError:
        return None


def traced_duration(builder: Any, fname: str, attrs: dict[str, Any]) -> int | None:
    """The duration in program cycles of a *fname* waveform once the times
    in its *attrs* are converted (the builder checks the rest); ``None`` when
    not known while tracing."""
    in_program_cycles(fname, attrs, getattr(builder, "clock_ghz", 2.0))
    duration = attrs.get("duration")
    return duration if duration.__class__ is int else None


def custom_samples(
    builder: Any, duration: Any, attrs: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """The attributes of the custom_samples op a custom() call of *duration*
    makes on *builder*'s lines, and its duration in program cycles."""
    from ._emitter import _sample_envelope

    if duration.__class__ is not int or not 1 <= duration <= MAX_CYCLES:
        in_program_cycles("custom", attrs, getattr(builder, "clock_ghz", 2.0))
        duration = attrs.get("duration")
    rate = getattr(builder, "sample_rate", None)
    attrs = {"samples": _sample_envelope(duration, attrs.get("name"), rate)}
    if rate is None:
        return attrs, len(attrs["samples"])
    attrs["rate"] = rate
    return attrs, len(attrs["samples"]) * rate[1] // rate[0]


def track_samples(
    em: Any,
    fname: str,
    op_name: str | None,
    operands: tuple[Any, ...],
    out_types: tuple[str, ...],
    attrs: Mapping[str, Any],
    vid: int,
    envelope: tuple[Any, Any] | None = None,
) -> None:
    """Record the rates the samples of the waveform *vid*, which the op
    *op_name* (of *fname*) made of *operands*, were made for, and how to
    make it again: of a custom() *envelope* (duration, function), or of
    its operands made again."""
    sampled = em._sampled
    assert sampled is not None
    if fname in ("custom", "custom_samples"):
        recipe = None if envelope is None else ("custom", *envelope, out_types)
        sampled[vid] = (frozenset({attrs.get("rate")}), recipe)
        return
    made: frozenset[Any] = frozenset().union(
        *(sampled[o.vid][0] for o in operands if o.__class__ is IRValue and o.vid in sampled)
    )
    if not made:
        sampled.pop(vid, None)
    elif op_name is None:
        sampled[vid] = (made, None)
    else:
        sampled[vid] = (made, ("op", op_name, operands, out_types, dict(attrs)))


def for_line(em: Any, fname: str, operands: tuple[Any, ...]) -> tuple[Any, ...]:
    """The *operands* of the drive or readout *fname* that the emitter *em*
    traces, with the samples of
    its waveform made for the rate of its line: what custom() made is
    sampled again for it, and samples given for another rate fail."""
    sampled = em._sampled
    assert sampled is not None
    wave = operands[1]
    record = sampled.get(wave.vid) if wave.__class__ is IRValue else None
    if record is None:
        return operands
    b = em.b
    qudit = b._line_qudit.get(operands[0].vid, -1)
    if qudit < 0:
        raise CompilationError(
            f"{fname}(): plays samples on a runtime qudit's line, and the control "
            "system's controllers play different sample rates, so the rate they "
            "need depends on the qudit a call binds; play them on a static qudit's "
            "line, or give the controllers one sample rate",
            code="control_system",
        )
    rate = b.sample_rate_of(fname, qudit)
    again = _resampled(em, wave, rate)
    if again is None:
        program = getattr(b, "clock_ghz", 2.0)
        shown = " and ".join(sorted(f"{_rate_ghz(m, program):g} GHz" for m in record[0]))
        plays = _rate_ghz(rate, program)
        raise CompilationError(
            f"{fname}(): the waveform on the {fname} line of qudit {qudit} holds "
            f"samples made for {shown}, and the line's controller plays {plays:g} GHz "
            f"({_rate_text(rate or (1, 1))}), which would change how long they last; "
            f"make them for it with sample_rate_ghz={plays:g}, or build the waveform "
            "with custom(duration, f), which samples it at the rate of the line that "
            "plays it",
            code="invalid_waveform",
        )
    return operands if again is wave else (operands[0], again, *operands[2:])


def _resampled(em: Any, wave: IRValue, rate: tuple[int, int] | None) -> IRValue | None:
    """*wave* with its samples made for *rate*: itself, or made again
    here; ``None`` when samples it holds were given for another."""
    sampled = em._sampled
    assert sampled is not None
    record = sampled.get(wave.vid)
    if record is None:
        return wave
    made, recipe = record
    if made == {rate}:
        return wave
    if recipe is None:
        return None
    from ._emitter import _OP_NAMES, _sample_envelope

    b = em.b
    if recipe[0] == "custom":
        _, duration, name, out_types = recipe
        attrs: dict[str, Any] = {"samples": _sample_envelope(duration, name, rate)}
        count = len(attrs["samples"])
        if rate is not None:
            attrs["rate"] = rate
            count = count * rate[1] // rate[0]
        again: IRValue = b.emit(_OP_NAMES["custom_samples"], (), out_types, attrs)[0]
        em._durations[again.vid] = count
    else:
        _, op_name, operands, out_types, attrs = recipe
        made_again = []
        for operand in operands:
            if operand.__class__ is IRValue:
                operand = _resampled(em, operand, rate)
                if operand is None:
                    return None
            made_again.append(operand)
        again = b.emit(op_name, tuple(made_again), out_types, dict(attrs))[0]
        duration = em._durations.get(wave.vid)
        if duration is not None:
            em._durations[again.vid] = duration
    sampled[again.vid] = (frozenset({rate}), recipe)
    return again
