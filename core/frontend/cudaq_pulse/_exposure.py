# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a schedule does to each qudit over time: per qudit and per loop, the
cycles of one round in which it is driven, read out and idle, and how many
rounds there are. Read from the scheduled module in the rolled form, so the
work grows with the program's operations, not with how often they run.

A row is exact, or marked as not exact where the schedule does not fix the
figure: a wait of run-time length or an answer in the stretch, a loop whose
operations repeat with different periods, a branch whose arms keep the
qudit busy for different lengths (the less busy arm is counted)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ._errors import UnsupportedError
from ._idle_windows import read

__all__ = ["Exposure", "QuditExposure"]


@dataclass(frozen=True)
class QuditExposure:
    """One qudit in one stretch of the program: ``loop`` is the loop's place
    among the program's outermost loops that hold an operation (0 for the
    first), or ``None`` for the rest of the program's length: everything
    outside those loops, and the loops the qudit has no pulse in (left out
    when that is no time at all). ``cycles`` is the length of one round,
    during which the qudit is driven for ``drive_cycles`` (a pulse on its
    drive line, or at its tone from another line), read out for
    ``readout_cycles`` and idle for ``idle_cycles``; a wait is idle, and a
    drive during a readout counts in both. ``rounds`` is how often the
    stretch runs. ``exact`` is false where the schedule does not fix these
    figures.

    ``decay`` and ``dephasing`` are what the idle time of one round costs,
    from the qudit's T1 and T2 star: the probability ``1 - exp(-t / T1)``
    that an excited qudit relaxes, and the probability
    ``(1 - exp(-t / T_phi)) / 2`` of a phase flip from pure dephasing,
    ``1 / T_phi = 1 / T2 - 1 / (2 T1)``. A T1 or T2 star the quantum system
    gives as 0 counts as none. What a pulse costs is not in them."""

    qudit: int
    loop: int | None
    rounds: int
    cycles: int
    drive_cycles: int
    readout_cycles: int
    idle_cycles: int
    exact: bool
    decay: float = 0.0
    dephasing: float = 0.0


class Exposure(tuple[QuditExposure, ...]):
    """The rows of ``ScheduledKernel.exposure``, by qudit and then in
    program order; ``print`` it for a table."""

    __slots__ = ()

    def __str__(self) -> str:
        table: list[tuple[str, ...]] = [
            ("qudit", "loop", "rounds", "cycles", "drive", "readout", "idle", "idle %")
            + ("exact", "decay", "dephasing")
        ]
        for row in self:
            share = 100.0 * row.idle_cycles / row.cycles if row.cycles else 0.0
            table.append(
                (
                    str(row.qudit),
                    "-" if row.loop is None else str(row.loop),
                    str(row.rounds),
                    str(row.cycles),
                    str(row.drive_cycles),
                    str(row.readout_cycles),
                    str(row.idle_cycles),
                    f"{share:.1f}",
                    "yes" if row.exact else "no",
                    f"{row.decay:.2e}",
                    f"{row.dephasing:.2e}",
                )
            )
        widths = [max(len(line[k]) for line in table) for k in range(len(table[0]))]
        return "\n".join(
            "  ".join(cell.rjust(width) for cell, width in zip(line, widths, strict=True))
            for line in table
        )


class _Busy:
    """One execution of a pulse that keeps a qudit busy: where it lies
    (``chain``: the loops and branches around it, outermost first, as
    ("for", loop, trips, depth) and ("if", branch, arm)), its interval in
    the first iteration of every loop around it, whether it is a readout,
    and its start's shift per iteration of each loop around it."""

    __slots__ = ("chain", "start", "end", "readout", "shifts")

    def __init__(
        self, chain: tuple[Any, ...], start: int, end: int, readout: bool, shifts: tuple[int, ...]
    ) -> None:
        self.chain = chain
        self.start = start
        self.end = end
        self.readout = readout
        self.shifts = shifts


def _chain(event: Any) -> tuple[Any, ...]:
    # read() numbers loops and branches from one counter as it enters them,
    # so around one event a smaller number is further out.
    around: list[tuple[Any, ...]] = [
        ("for", loop, trips, depth) for depth, (loop, _, trips) in enumerate(event.path)
    ]
    around += [("if", branch, arm) for branch, arm in event.arms]
    return tuple(sorted(around, key=lambda entry: entry[1]))


def _union(spans: list[tuple[int, int]]) -> int:
    total = 0
    reach = None
    for start, end in sorted(spans):
        if reach is None or start > reach:
            total += end - start
            reach = end
        elif end > reach:
            total += end - reach
            reach = end
    return total


def _folded(spans: list[tuple[int, int]], period: int) -> list[tuple[int, int]]:
    """*spans* as they lie in one round of a loop that repeats them every
    *period* cycles: a pulse that runs on into the next round is busy time
    at the start of that one."""
    out: list[tuple[int, int]] = []
    for start, end in spans:
        if end - start >= period:
            return [(0, period)]
        start, end = start % period, start % period + end - start
        out.append((start, min(end, period)))
        if end > period:
            out.append((0, end - period))
    return out


def _measure(found: list[_Busy], depth: int, period: int = 0) -> tuple[int, int, int, bool]:
    """Of the pulses *found*, all inside the same *depth* loops and arms:
    the cycles of drive, of readout and of either in one execution, and
    whether the schedule fixes them. With *period*, they are the body of a
    loop of that period, and one execution is one round of a long run."""
    here = [busy for busy in found if len(busy.chain) == depth]
    spans = [(b.start, b.end) for b in here]
    exact = True
    if period > 0:
        # Rounds that overlap: the first and the last differ from the rest.
        exact = _union(_folded(spans, period)) == _union(spans)
        drive = _union(_folded([(b.start, b.end) for b in here if not b.readout], period))
        readout = _union(_folded([(b.start, b.end) for b in here if b.readout], period))
        either = _union(_folded(spans, period))
    else:
        drive = _union([(b.start, b.end) for b in here if not b.readout])
        readout = _union([(b.start, b.end) for b in here if b.readout])
        either = _union(spans)
    inside: dict[int, list[_Busy]] = {}
    for busy in found:
        if len(busy.chain) > depth:
            inside.setdefault(busy.chain[depth][1], []).append(busy)
    for group in inside.values():
        entry = group[0].chain[depth]
        if entry[0] == "for":
            _, _, trips, loop_depth = entry
            d, r, e, fixed = _measure(group, depth + 1)
            fixed = fixed and len({busy.shifts[loop_depth] for busy in group}) == 1
            drive, readout, either = drive + trips * d, readout + trips * r, either + trips * e
            exact = exact and fixed
            continue
        arms = [
            _measure([busy for busy in group if busy.chain[depth][2] == arm], depth + 1)
            for arm in (0, 1)
        ]
        # The less busy arm: the qudit idles at most as long as reported.
        d, r, e, fixed = min(arms, key=lambda arm: arm[2])
        drive, readout, either = drive + d, readout + r, either + e
        exact = exact and fixed and arms[0] == arms[1]
    return drive, readout, either, exact


def rows(module: Any) -> list[QuditExposure]:
    """The exposure of every qudit of scheduled *module* that has a line."""
    lines, items = read(module)
    qudits = sorted({qudit for _, qudit in lines if qudit >= 0})
    busy: dict[int, list[_Busy]] = {qudit: [] for qudit in qudits}
    # The trips of the program's outermost loops that hold an operation, in
    # program order.
    loops: dict[int, int] = {}
    # Outermost loops (None: outside them) next to a wait of run-time length
    # or an answer: what runs there has no fixed length.
    loose: set[int | None] = set()
    pending = False
    last: int | None = None
    length = 0
    for item in items:
        if item.__class__ is tuple:
            if item[0] == "barrier":
                loose.add(last)
                pending = True
            continue
        event = item
        if event.kind == "sync":
            continue
        chain = _chain(event)
        outer = chain[0][1] if chain and chain[0][0] == "for" else None
        last = outer
        if pending:
            loose.add(outer)
            pending = False
        if any(copy for _, copy, _ in event.path):
            continue
        length = max(
            length,
            event.end
            + sum(
                (trips - 1) * shift
                for (_, _, trips), shift in zip(event.path, event.shifts, strict=True)
            ),
        )
        if outer is not None:
            loops.setdefault(outer, chain[0][2])
        if event.kind not in ("drive", "readout") or event.line < 0:
            continue
        role, owner = lines[event.line]
        on: list[tuple[int, bool]] = []
        if owner >= 0:
            on.append((owner, role == "readout"))
        if event.tone >= 0 and event.tone != event.line:
            tone_role, tone_owner = lines[event.tone]
            if tone_role == "drive" and tone_owner >= 0 and tone_owner != owner:
                on.append((tone_owner, False))
        for qudit, is_readout in on:
            busy[qudit].append(_Busy(chain, event.start, event.end, is_readout, event.shifts))

    order = {loop: place for place, loop in enumerate(loops)}
    out: list[QuditExposure] = []
    for qudit in qudits:
        mine = busy[qudit]
        looped = 0
        inside: list[QuditExposure] = []
        # Lines run on their own: a loop the qudit has no pulse in takes none
        # of its time, which is then time outside the loops.
        for loop, trips in loops.items():
            found = [b for b in mine if b.chain and b.chain[0][1] == loop]
            if not found:
                continue
            own = {b.shifts[0] for b in found}
            period = max(own)
            drive, readout, either, exact = _measure(found, 1, period)
            exact = exact and len(own) == 1 and loop not in loose
            if either > period:
                # More than a round holds: the figures are the round's most.
                drive, readout = min(drive, period), min(readout, period)
                either, exact = period, False
            looped += trips * period
            inside.append(
                QuditExposure(
                    qudit, order[loop], trips, period, drive, readout, period - either, exact
                )
            )
        found = [b for b in mine if not b.chain or b.chain[0][0] != "for"]
        drive, readout, either, exact = _measure(found, 0)
        rest = length - looped
        if either > rest:
            # The qudit's loops and what it does outside them run at the same
            # time: no row of it is a stretch of its own.
            rest = either
            inside = [replace(row, exact=False) for row in inside]
            out.append(QuditExposure(qudit, None, 1, rest, drive, readout, 0, False))
        elif rest or not inside:
            # What is left of the program's length is as fixed as every
            # stretch of it.
            exact = exact and not loose
            out.append(QuditExposure(qudit, None, 1, rest, drive, readout, rest - either, exact))
        out.extend(inside)
    return out


def report(program: Any) -> Exposure:
    """``ScheduledKernel.exposure`` of *program*."""
    try:
        with program._lock:
            found = rows(program._module)
    except UnsupportedError:
        raise UnsupportedError(
            "exposure: the program is not scheduled without its arguments' values"
        ) from None
    system = program.quantum_system
    t1s, t2s = system._description.t1_times, system._description.t2_times
    out = []
    for row in found:
        idle_us = row.idle_cycles / system.clock_ghz / 1000.0
        t1, t2 = t1s.get(row.qudit, 0.0), t2s.get(row.qudit, 0.0)
        relax = 1.0 / t1 if t1 > 0 else 0.0
        pure = max((1.0 / t2 if t2 > 0 else 0.0) - relax / 2.0, 0.0)
        out.append(
            QuditExposure(
                row.qudit,
                row.loop,
                row.rounds,
                row.cycles,
                row.drive_cycles,
                row.readout_cycles,
                row.idle_cycles,
                row.exact,
                -math.expm1(-idle_us * relax),
                -math.expm1(-idle_us * pure) / 2.0,
            )
        )
    return Exposure(out)
