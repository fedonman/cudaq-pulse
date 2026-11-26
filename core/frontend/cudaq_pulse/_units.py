# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock times: ``20 * pulse.ns``, ``4 * pulse.us``.

A kernel counts time in cycles of the quantum system's program clock; a bare int in
a duration is a number of cycles. A time with units is converted to cycles
when the kernel compiles, for the quantum system it compiles for, so one kernel
written in nanoseconds runs on quantum systems with different clocks.

A time is a float of nanoseconds that also holds its exact value, a
fraction: ``0.1 * pulse.us`` is exactly 100 ns, and converts to whole
cycles exactly or not at all. Arithmetic keeps the unit where it means a
time (time times a number, time plus time, time over a number, minus a
time) and rejects the rest (time times time, time plus a number of cycles).

The fraction is held as two coprime ints, so arithmetic on times needs no
``fractions``; ``exact_ns`` gives it as a ``Fraction``.
"""

from __future__ import annotations

import math
import numbers
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ._errors import InvalidValueError

if TYPE_CHECKING:
    from fractions import Fraction

__all__ = ["Time", "ns", "us", "ms", "s"]


# *value*, a real number, as the coprime numerator and positive
# denominator its decimal digits give: ``0.1`` is 1/10.
def _exact(value: Any) -> tuple[int, int]:
    cls = value.__class__
    if cls is int:
        return value, 1
    # Below 1e15 an integral float's digits are the integer itself.
    if cls is float and value.is_integer() and abs(value) < 1e15:
        return int(value), 1
    if isinstance(value, numbers.Rational):
        return _reduced(int(value.numerator), int(value.denominator))
    value = float(value)
    if not math.isfinite(value):
        raise InvalidValueError(f"a time is a finite number of nanoseconds, not {value!r}")
    # The digits of repr(value), as fractions.Fraction(repr(value)) reads them.
    mantissa, _, exponent = repr(value).partition("e")
    whole, _, digits = mantissa.partition(".")
    numerator = int(whole + digits)
    denominator = 10 ** len(digits)
    shift = int(exponent) if exponent else 0
    if shift >= 0:
        numerator *= 10**shift
    else:
        denominator *= 10**-shift
    return _reduced(numerator, denominator)


def _reduced(numerator: int, denominator: int) -> tuple[int, int]:
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    g = math.gcd(numerator, denominator)
    return (numerator // g, denominator // g) if g > 1 else (numerator, denominator)


def _is_number(value: Any) -> bool:
    cls = value.__class__
    if cls is int or cls is float:
        return True
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, Time))


class UnitError(TypeError):
    """Arithmetic that mixes a time with a number of cycles, or makes
    something other than a time."""


class Time(float):
    """A wall-clock time: a float of nanoseconds holding its exact value.

    Made by multiplying a unit, ``20 * pulse.ns``; durations, waits and
    ``pulse.round_duration`` take it, and convert it to cycles of the clock the
    kernel compiles for.

    ``float(t)`` is the time in nanoseconds, whatever unit made it. A bare
    int duration in a kernel is a number of clock cycles instead, so it
    depends on the quantum system's clock.
    Times compare with times; comparing one with a bare number, with ``==``
    or an ordering, raises ``TypeError``: write ``t == 20 * pulse.ns`` or
    ``float(t) == 20.0``. A time never hashes like a number, so a set or dict
    can hold ``20 * pulse.ns`` and ``20.0`` as different keys, and finding a
    number in it never compares it with a time; ``in`` on a list or tuple
    compares with ``==``, and raises when it meets a time and a number.
    """

    __slots__ = ("_n", "_d")
    _n: int
    _d: int

    def __new__(cls, nanoseconds: Fraction | int | float) -> Time:
        numerator, denominator = _exact(nanoseconds)
        return _make(numerator, denominator, cls)

    @property
    def exact_ns(self) -> Fraction:
        """The time in nanoseconds, exactly."""
        from fractions import Fraction

        return Fraction(self._n, self._d)

    def __reduce__(self) -> tuple[Any, ...]:
        return (_time, (self._n, self._d))

    # A time is immutable, so a copy is the time itself.
    def __copy__(self) -> Time:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Time:
        return self

    def __repr__(self) -> str:
        n, d = self._n, self._d
        for unit, scale in (("s", 10**9), ("ms", 10**6), ("us", 10**3)):
            if abs(n) >= scale * d:
                return f"{n / (d * scale):g} {unit}"
        return f"{n / d:g} ns"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self) if not spec else float.__format__(self, spec)

    # ── Arithmetic that keeps the unit ──────────────────────────────────

    def __mul__(self, other: Any) -> Time:
        if other.__class__ is Time:
            raise UnitError(f"{self} * {other} is a time times a time, which is not a time")
        if _is_number(other):
            n, d = _exact(other)
            a, b = self._n, self._d
            g, h = math.gcd(a, d), math.gcd(n, b)
            return _make((a // g) * (n // h), (b // h) * (d // g))
        return NotImplemented

    # Not an alias of __mul__: editors would type 100 * pulse.ns as float.
    def __rmul__(self, other: Any) -> Time:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        if other.__class__ is Time:
            n, d = other._n, other._d
            if n == 0:
                raise ZeroDivisionError(f"Fraction({d}, 0)")
            # Correctly rounded, as float() of the Fraction is, and 0.0 rather
            # than -0.0 when it is zero, as the Fraction's sign is its own.
            numerator, denominator = self._n * d, self._d * n
            if denominator < 0:
                numerator, denominator = -numerator, -denominator
            return numerator / denominator
        if _is_number(other):
            n, d = _exact(other)
            if n == 0:
                raise ZeroDivisionError(f"Fraction({d}, 0)")
            return _make(*_reduced(self._n * d, self._d * n))
        return NotImplemented

    def __rtruediv__(self, other: Any) -> Any:
        if _is_number(other):
            raise UnitError(f"{other!r} / {self} is a number over a time, which is not a time")
        return NotImplemented

    def __add__(self, other: Any) -> Time:
        if other.__class__ is Time:
            b, d = self._d, other._d
            return _make(*_reduced(self._n * d + other._n * b, b * d))
        if other == 0 and other.__class__ is int:
            return self
        if _is_number(other):
            raise _mixed(self, "+", other)
        return NotImplemented

    def __radd__(self, other: Any) -> Time:
        if other == 0 and other.__class__ is int:
            return self
        if _is_number(other):
            raise _mixed(other, "+", self)
        return NotImplemented

    def __sub__(self, other: Any) -> Time:
        if other.__class__ is Time:
            b, d = self._d, other._d
            return _make(*_reduced(self._n * d - other._n * b, b * d))
        if _is_number(other):
            raise _mixed(self, "-", other)
        return NotImplemented

    def __rsub__(self, other: Any) -> Any:
        if _is_number(other):
            raise _mixed(other, "-", self)
        return NotImplemented

    def __neg__(self) -> Time:
        return _make(-self._n, self._d)

    def __pos__(self) -> Time:
        return self

    def __abs__(self) -> Time:
        return _make(abs(self._n), self._d)

    def __floordiv__(self, other: Any) -> Any:
        if other.__class__ is Time:
            return (self._n * other._d) // (self._d * other._n)
        raise UnitError(f"{self} // {other!r}: divide a time by a time, or use /")

    def __rfloordiv__(self, other: Any) -> Any:
        raise UnitError(f"{other!r} // {self} is a number over a time, which is not a time")

    def __mod__(self, other: Any) -> Time:
        if other.__class__ is Time:
            b, d = self._d, other._d
            return _make(*_reduced((self._n * d) % (other._n * b), b * d))
        raise UnitError(f"{self} % {other!r}: the remainder of a time is by a time")

    def __rmod__(self, other: Any) -> Any:
        raise UnitError(f"{other!r} % {self}: the remainder of a time is by a time")

    def __pow__(self, other: Any, modulo: Any = None) -> Any:
        raise UnitError(f"{self} ** {other!r} is not a time")

    def __rpow__(self, other: Any, modulo: Any = None) -> Any:
        raise UnitError(f"{other!r} ** {self} raises to a time")

    # ── Comparisons: only with times ────────────────────────────────────
    # Each compares the two cross products, the denominators being positive.

    def _other(self, op: str, other: Any) -> tuple[int, int] | None:
        if other.__class__ is Time:
            return self._n * other._d, other._n * self._d
        if _is_number(other):
            raise _compared(self, op, other)
        return None

    def __eq__(self, other: object) -> bool:
        pair = self._other("==", other)
        return NotImplemented if pair is None else pair[0] == pair[1]

    def __ne__(self, other: object) -> bool:
        pair = self._other("!=", other)
        return NotImplemented if pair is None else pair[0] != pair[1]

    def __hash__(self) -> int:
        # Every number hashes below sys.hash_info.modulus in magnitude, so a
        # time never shares a hash with one, and sets and dicts never
        # compare the two.
        h = float.__hash__(self)
        return h + _HASH_OFFSET if h >= 0 else h - _HASH_OFFSET

    def __lt__(self, other: Any) -> bool:
        pair = self._other("<", other)
        return NotImplemented if pair is None else pair[0] < pair[1]

    def __le__(self, other: Any) -> bool:
        pair = self._other("<=", other)
        return NotImplemented if pair is None else pair[0] <= pair[1]

    def __gt__(self, other: Any) -> bool:
        pair = self._other(">", other)
        return NotImplemented if pair is None else pair[0] > pair[1]

    def __ge__(self, other: Any) -> bool:
        pair = self._other(">=", other)
        return NotImplemented if pair is None else pair[0] >= pair[1]


_new_float = float.__new__


# The time of *numerator* / *denominator* ns, given coprime with a
# positive denominator.
def _make(numerator: int, denominator: int, cls: type[Time] = Time) -> Time:
    try:
        self = _new_float(cls, numerator / denominator)
    except OverflowError:
        raise InvalidValueError(
            f"a time is at most {sys.float_info.max!r} ns; this one is longer"
        ) from None
    self._n = numerator
    self._d = denominator
    return self


_HASH_OFFSET = 1 << (sys.hash_info.width - 2)


def _compared(time: Time, op: str, number: Any) -> UnitError:
    return UnitError(
        f"{time} {op} {number!r} compares a time with a bare number; compare it with a "
        f"time ({number!r} * pulse.ns), or compare float(t), which is the time in "
        "nanoseconds"
    )


def _time(numerator: int, denominator: int) -> Time:
    return _make(*_reduced(numerator, denominator))


def _mixed(left: Any, op: str, right: Any) -> UnitError:
    cycles = right if left.__class__ is Time else left
    return UnitError(
        f"{left!r} {op} {right!r} mixes a time with {cycles!r} cycles; give both with "
        f"units ({cycles!r} * pulse.ns) or both in cycles"
    )


def as_nanoseconds(value: Any) -> Any:
    """A time a parameter takes in ns, given as a number or a ``Time``: a
    ``Time`` as its float of nanoseconds, anything else as it is, for the
    parameter's own checks."""
    return float(value) if value.__class__ is Time else value


class _Unit(tuple[str, int]):
    """A unit's name and nanoseconds, made a ``Time`` on first use and then
    kept in the module's namespace, where later reads find it as a plain
    attribute."""

    def __get__(self, module: ModuleType | None, owner: type) -> Any:
        if module is None:
            return self
        name, nanoseconds = self
        unit = module.__dict__[name] = Time(nanoseconds)
        return unit


class _Module(ModuleType):
    ns = _Unit(("ns", 1))
    us = _Unit(("us", 10**3))
    ms = _Unit(("ms", 10**6))
    s = _Unit(("s", 10**9))


sys.modules[__name__].__class__ = _Module

if TYPE_CHECKING:
    ns: Time
    us: Time
    ms: Time
    s: Time


_CYCLES_PER_NS: dict[float, tuple[int, int]] = {}


def cycles_per_ns(clock_ghz: float) -> tuple[int, int]:
    """Cycles of a *clock_ghz* clock in one nanosecond, exactly as written:
    a coprime numerator and positive denominator."""
    found = _CYCLES_PER_NS.get(clock_ghz)
    if found is None:
        found = _CYCLES_PER_NS[clock_ghz] = _exact(clock_ghz)
    return found


def in_cycles(time: Time, clock_ghz: float) -> tuple[int, int]:
    """*time* in cycles of a *clock_ghz* clock, exactly: a coprime numerator
    and positive denominator."""
    n, d = cycles_per_ns(clock_ghz)
    return _reduced(time._n * n, time._d * d)
