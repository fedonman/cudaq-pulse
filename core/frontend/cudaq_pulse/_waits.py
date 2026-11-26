# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checks of a module's variable waits, which bind runs on every call's
values. Only a kernel with a wait whose length a float sets loads them."""

from __future__ import annotations

import functools
import marshal
import math
from collections.abc import Sequence
from types import FunctionType
from typing import Any

from ._errors import InvalidValueError
from .kernel.ir_builder import MAX_CYCLES

# Longest wait a call may ask for: lengths stay exact in a float.
_MAX_WAIT_CYCLES = MAX_CYCLES

# ``PulseModule.program_length()``, with the loops and ifs around the waits
# when one is in an if.
_Length = tuple[int, int, int, Sequence[int]] | tuple[int, int, int, Sequence[int], Sequence[int]]

# Python compiles a chain of ``+`` one level deeper per term, and fails with
# a RecursionError past some thousands of terms: a longer sum adds its terms
# in groups of this many.
_SUM_TERMS = 32


def _summed(terms: Sequence[str]) -> str:
    """The sum of *terms*, expressions of the call's values ``a``, added left
    to right as ``t0 + t1 + ...`` adds them, as an expression that nests no
    deeper with more terms: past ``_SUM_TERMS`` terms, each group after the
    first adds its terms to the sum so far, held in ``_s``."""
    if len(terms) <= _SUM_TERMS:
        return " + ".join(terms)
    more = _SUM_TERMS - 1
    groups = [" + ".join(terms[:_SUM_TERMS])]
    groups.extend(
        " + ".join(["_s", *terms[at : at + more]]) for at in range(_SUM_TERMS, len(terms), more)
    )
    return f"({', '.join(f'_s := {group}' for group in groups)})[-1]"


@functools.lru_cache(maxsize=1024)
def _computed(source: str) -> Any:
    """The length *source* computes from the call's values ``a``, compiled
    once for all its waits: every wait of one float has the same source."""
    return eval(f"lambda a: {source}", {"__builtins__": {}})


class _WaitLength:
    """A wait whose length the module computes from its float arguments.

    The call's values give its length, rounded to the nearest cycle with ties
    up as the module rounds it, and checked against its bounds before the
    call runs. For a length that also takes answers of classical calls this
    is its float part, the length with its answers at 0, which *text* names:
    it has no lowest length (*lowest* is None) and must be finite and at most
    the highest, since the run clamps the whole length to the bounds.
    """

    __slots__ = ("source", "length", "lowest", "highest", "text", "kernel")

    def __init__(
        self,
        expression: Any,
        lowest: int | None,
        highest: int,
        kernel: str,
        text: str | None = None,
    ) -> None:
        from .kernel.ir_builder import arithmetic_source, arithmetic_text

        self.source = arithmetic_source(expression)
        self.length = _computed(self.source)
        self.lowest = lowest
        self.highest = None if highest < 0 else highest
        self.text = arithmetic_text(expression) if text is None else text
        self.kernel = kernel

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuilt_wait, (self.source, self.lowest, self.highest, self.text, self.kernel))

    def check(self, values: Sequence[float]) -> None:
        """Raise ``ValueError`` saying why the length is not allowed."""
        try:
            x = self.length(values)
        except ZeroDivisionError:
            x = math.nan
        if not math.isfinite(x):
            raise ValueError(
                f"{self.kernel}(): the wait length {self.text} is {x!r}; it must be a "
                "finite number of clock cycles"
            )
        cycles = math.floor(x + 0.5)
        if self.lowest is None:
            if self.highest is not None and cycles > self.highest:
                raise ValueError(
                    f"{self.kernel}(): the wait length {self.text} = {x!r} rounds to "
                    f"{cycles} clock cycles, above its highest length {self.highest}"
                )
            return
        if cycles < 0 and self.lowest == 0:
            raise ValueError(
                f"{self.kernel}(): the wait length {self.text} = {x!r} rounds to "
                f"{cycles} clock cycles; a wait cannot be negative"
            )
        highest = _MAX_WAIT_CYCLES if self.highest is None else self.highest
        if not self.lowest <= cycles <= highest:
            bounds = (self.lowest, self.highest)
            raise ValueError(
                f"{self.kernel}(): the wait length {self.text} = {x!r} rounds to "
                f"{cycles} clock cycles, outside its bounds {bounds!r}"
            )


class _WaitLengths:
    """The variable waits of a module, checked together on every call, and
    with them the program's length: (its cycles op after op with each wait
    at its lowest length, the most it may last so, 1 if a controller's
    longest program sets that, how often each wait with a lowest length
    runs, and when one is in an if the loops and ifs around them), as
    ``PulseModule.program_length()`` gives it. Under a controller's longest
    program the most it may last op after op is that length times the lines
    the program takes: past it no schedule fits, and within it the schedule
    of each call is checked."""

    __slots__ = ("length", "ok", "waits")

    # By wait, the steps of the lengths a controller with a coarser clock
    # than the program's can time (none: every length).
    grids: tuple[int, ...] = ()

    def __init__(
        self,
        waits: Sequence[tuple[Any, ...]],
        kernel: str,
        grids: Sequence[int] = (),
        length: _Length | None = None,
    ) -> None:
        self.waits = tuple(
            _WaitLength(wait[0], wait[1], wait[2], kernel, wait[3] if len(wait) > 3 else None)
            for wait in waits
        )
        self.length = length
        # The program's length within the most it may last, tested on its
        # own unless the waits' bounds already keep it there, or the one
        # wait it depends on has a highest length that does.
        highest = {id(w): _highest(w) for w in self.waits}
        # The program's length with the call's lengths, when it is tested.
        lasts = None
        timed = self._timed()
        if timed:
            assert length is not None
            cycles, most, _, repeats = length[:4]
            spare = most - cycles
            runs = [(w, w.lowest or 0, r) for w, r in zip(timed, repeats, strict=True) if r]
            if sum(r * (highest[id(w)] - lowest) for w, lowest, r in runs) > spare:
                if len(length) > 4:
                    lasts = _longer_in_ifs(timed, length[4], cycles)
                elif len(runs) == 1:
                    ((w, lowest, r),) = runs
                    highest[id(w)] = lowest + spare // r
                else:
                    lasts = _summed(
                        [str(cycles)]
                        + [
                            f"{r} * (({w.source} + 0.5) // 1 - {w.lowest})"
                            for w, r in zip(timed, repeats, strict=True)
                        ]
                    )
        # floor(x + 0.5) is in [lowest, highest] exactly when
        # lowest <= x + 0.5 < highest + 1, which NaN fails; without a lowest,
        # -1e400 (minus infinity) < x + 0.5 also fails for minus infinity.
        tests = [
            f"{'-1e400 <' if w.lowest is None else f'{w.lowest} <='} {w.source} + 0.5 < "
            f"{highest[id(w)] + 1}"
            for w in self.waits
        ]
        if grids:
            tests.extend(
                f"({w.source} + 0.5) // 1 % {grid} == 0"
                for w, grid in zip(self.waits, grids, strict=True)
                if w.lowest is not None and grid > 1
            )
        if lasts is not None:
            tests.append(f"{lasts} <= {most}")
        # Each test once, where it first comes: waits of one length and
        # bounds pass or fail it together, so the others add nothing.
        self.ok = eval(f"lambda a: {' and '.join(dict.fromkeys(tests))}", {"__builtins__": {}})

    def _timed(self) -> tuple[_WaitLength, ...]:
        """The waits the program's length counts at their lowest length."""
        if self.length is None or not self.length[3]:
            return ()
        return tuple(w for w in self.waits if w.lowest is not None)

    def __reduce__(self) -> tuple[Any, ...]:
        # The check as compiled: the stamp of the pickle names the Python
        # whose marshal format wrote it.
        check = marshal.dumps(self.ok.__code__)
        return (_rebuilt_wait_lengths, (type(self), self.waits, self.grids, check, self.length))

    def check(self, values: Sequence[float]) -> None:
        try:
            if self.ok(values):
                return
        except ZeroDivisionError:
            pass
        for wait in self.waits:
            wait.check(values)
        for wait, grid in zip(self.waits, self.grids, strict=False):
            cycles = math.floor(wait.length(values) + 0.5)
            if wait.lowest is not None and cycles % grid:
                lower = cycles // grid * grid
                raise InvalidValueError(
                    f"{wait.kernel}(): the wait length {wait.text} rounds to {cycles} clock "
                    f"cycles; its line moves in timesteps of {grid} cycles, and "
                    f"{cycles} cycles isn't a whole number of timesteps: the nearest "
                    f"lengths are {lower} and {lower + grid}"
                )
        timed = self._timed()
        if timed:
            assert self.length is not None
            cycles, most, controller, repeats = self.length[:4]
            if len(self.length) > 4:
                longer = _longer_in_ifs(timed, self.length[4])
                cycles += eval(f"lambda a: {longer}", {"__builtins__": {}})(values)
            else:
                for wait, runs in zip(timed, repeats, strict=False):
                    assert wait.lowest is not None
                    cycles += runs * (math.floor(wait.length(values) + 0.5) - wait.lowest)
            if cycles > most:
                kernel = timed[0].kernel
                if controller:
                    raise ValueError(
                        f"{kernel}(): with these wait lengths the program lasts more than "
                        f"{most} cycles op after op, more than any schedule of it fits in the "
                        "longest program of the controller (Controller.max_program_duration)"
                    )
                raise ValueError(
                    f"{kernel}(): with these wait lengths the program lasts more than 2**60 "
                    "cycles, more than a schedule counts"
                )


class _GridWaitLengths(_WaitLengths):
    """Variable waits on lines that move in timesteps of more than one
    cycle (*grids*, by wait). Its own class keeps the grids off the waits of
    every module on one clock."""

    __slots__ = ("grids",)

    def __init__(
        self,
        waits: Sequence[tuple[Any, ...]],
        kernel: str,
        grids: Sequence[int],
        length: _Length | None = None,
    ) -> None:
        self.grids = tuple(grids)
        super().__init__(waits, kernel, self.grids, length)


def _highest(wait: _WaitLength) -> int:
    return _MAX_WAIT_CYCLES if wait.highest is None else wait.highest


def _longer_in_ifs(
    timed: Sequence[_WaitLength], excess: Sequence[int], cycles: int | None = None
) -> str:
    """The cycles the lengths of *timed* add to the program's with each at
    its lowest, as an expression of the call's values ``a``, from the loops
    and ifs around them (``PulseModule.program_length()``): a loop adds its
    trips times its body's, an if what its longer arm lasts past the longer
    arm with the lowest lengths. Each region is a name assigned in turn, and
    each sum adds its terms in groups, so the expression nests no deeper
    with the program. With *cycles*, the program's length: *cycles* and
    those cycles, added in the order the length test adds them."""
    steps: list[str] = []
    at = 1

    def named(terms: list[str]) -> str:
        if not terms:
            return "0"
        k = len(steps)
        steps.append(f"(_v{k} := {_summed(terms)}) == _v{k}")
        return f"_v{k}"

    def body() -> list[str]:
        nonlocal at
        terms: list[str] = []
        while at < len(excess) and excess[at] not in (3, 4):
            kind = excess[at]
            if kind == 0:
                wait = timed[excess[at + 1]]
                at += 2
                terms.append(f"(({wait.source} + 0.5) // 1 - {wait.lowest})")
            elif kind == 1:
                trips = excess[at + 1]
                at += 2
                inner = named(body())
                at += 1
                terms.append(f"{trips} * {inner}")
            else:
                then, other = excess[at + 1], excess[at + 2]
                at += 3
                longer_then = named(body())
                longer_else = "0"
                if excess[at] == 4:
                    at += 1
                    longer_else = named(body())
                at += 1
                k = len(steps)
                steps.append(
                    f"(_t{k} := {then} + {longer_then}) == _t{k} and "
                    f"(_e{k} := {other} + {longer_else}) == _e{k}"
                )
                terms.append(f"(_t{k} if _t{k} >= _e{k} else _e{k}) - {max(then, other)}")
        return terms

    terms = body()
    if steps:
        longer = f"({' and '.join(steps)} and {_summed(terms) or '0'})"
        return longer if cycles is None else f"{cycles} + {longer}"
    if cycles is None:
        return _summed(terms) or "0"
    return _summed([str(cycles), *(terms or ["0"])])


def wait_lengths(
    waits: Sequence[tuple[Any, ...]],
    kernel: str,
    grid: int,
    grids: Sequence[int] | None = None,
    length: _Length | None = None,
) -> _WaitLengths:
    """The checks of *waits*, each on *grid*, or on its own of *grids*, and
    of the program's *length*."""
    if grids is not None:
        return _GridWaitLengths(waits, kernel, grids, length)
    if grid == 1:
        return _WaitLengths(waits, kernel, (), length)
    return _GridWaitLengths(waits, kernel, [grid] * len(waits), length)


def _rebuilt_wait(
    source: str, lowest: int | None, highest: int | None, text: str, kernel: str
) -> _WaitLength:
    wait = _WaitLength.__new__(_WaitLength)
    wait.source = source
    wait.length = _computed(source)
    wait.lowest = lowest
    wait.highest = highest
    wait.text = text
    wait.kernel = kernel
    return wait


def _rebuilt_wait_lengths(
    cls: type[_WaitLengths],
    waits: tuple[_WaitLength, ...],
    grids: tuple[int, ...],
    check: bytes,
    length: _Length | None = None,
) -> _WaitLengths:
    lengths = cls.__new__(cls)
    lengths.waits = waits
    lengths.length = length
    if isinstance(lengths, _GridWaitLengths):
        lengths.grids = grids
    lengths.ok = FunctionType(marshal.loads(check), {"__builtins__": {}})
    return lengths
