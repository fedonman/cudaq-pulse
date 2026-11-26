# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The controller checks of a program whose long rolled loops the native
walk summarized (``ControllerWalk::summarizeLoop``): the events of one
iteration stand for those of every later one, so each check works on a
loop's body once and counts its copies. A check finds what it finds with
every iteration played, in the same order; past ``LAZY_AFTER`` violations
they are built when read. So is a schedule of more than ``LAZY_AFTER``
events (``Schedule``). Only a check that met such a loop, or such a
schedule, loads this."""

from __future__ import annotations

import bisect
import math
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, overload

from ._errors import UnsupportedError
from .controller import (
    _PLAYED,
    WaveformMemoryScope,
    _checks_pulse_lengths,
    _exclusive,
    _native_clock,
    _peak_amplitude,
    _pulse_length,
    _sample_rates,
    _samples_per_own_cycle,
    shape_key,
)

if TYPE_CHECKING:
    from fractions import Fraction

    from ._clocks import ClockPlan, ControllerClock
    from ._system import Controller, ControlSystem
    from .compiler import ScheduledKernel

# Up to this many violations are a list, as without summarized loops.
LAZY_AFTER = 10_000
# The copies of loops a check plays one by one, at most: those whose events
# interleave in time with others on one line, or whose feedback depends on a
# readout before the loop.
_MAX_PLAYED = 1 << 20


class UnsyncedError(Exception):
    """A check pairs events of lines that a loop's copies move by different
    periods: the caller walks the program again with only loops of one
    period summarized."""


class _Rep:
    """Copies of the events of ``body``: copy j starts j * ``period``
    cycles after the first, which the events of ``body`` are; or, with
    ``lines``, j times the period of its event's line, ``at`` the line of
    each event of the first copy."""

    __slots__ = ("begin", "end", "copies", "period", "depth", "body", "orig", "lines", "at")

    def __init__(
        self,
        begin: int,
        end: int,
        copies: int,
        period: int,
        depth: int,
        lines: dict[int, int] | None = None,
    ) -> None:
        self.begin, self.end, self.copies = begin, end, copies
        self.period, self.depth = period, depth
        self.lines = lines
        self.at: dict[int, int] = {}
        self.body: list[Any] = []
        self.orig = self

    def keeping(self, body: list[Any]) -> _Rep:
        kept = _Rep(self.begin, self.end, self.copies, self.period, self.depth, self.lines)
        kept.body, kept.orig, kept.at = body, self.orig, self.at
        return kept


def _period(rep: _Rep, lines: Iterable[int]) -> int:
    """The period of *rep*'s copies on *lines*, which they must share."""
    if rep.lines is None:
        return rep.period
    found = {rep.lines[line] for line in lines if line in rep.lines}
    if len(found) > 1:
        raise UnsyncedError
    return found.pop() if found else 0


class _Shift:
    """The cycles a copy starts after the first where the lines of a loop
    move by periods of their own: ``base``, and for the time of the event
    of each index in ``by``, that many more."""

    __slots__ = ("base", "by")

    def __init__(self, base: int, by: dict[int, int]) -> None:
        self.base, self.by = base, by


# The cycles a copy starts after the first: one for every event, or a _Shift.
_Moved = int | _Shift


def _at(piece: int | tuple[int, int], shift: _Moved) -> int:
    """A time of a violation, *piece* (or (time, the index of the event
    whose copy it moves with)), in a copy *shift* after the first."""
    if isinstance(piece, int):
        return piece + (shift if isinstance(shift, int) else shift.base)
    value, key = piece
    if isinstance(shift, int):
        return value + shift
    return value + shift.base + shift.by.get(key, 0)


def _moved(shift: _Moved, offset: int, j: int, period: int | dict[int, int]) -> _Moved:
    """*shift* moved to copy *j* of copies *offset* + j * *period* cycles
    after the first, or by event index *period*."""
    if isinstance(period, int):
        if isinstance(shift, int):
            return shift + offset + j * period
        return _Shift(shift.base + offset + j * period, shift.by)
    base, by = (shift, {}) if isinstance(shift, int) else (shift.base, shift.by)
    moved = dict(by)
    for key, step in period.items():
        moved[key] = moved.get(key, 0) + j * step
    return _Shift(base + offset, moved)


class _Text:
    """A violation: its pieces joined, an int piece shifted by the cycles
    its copy starts after the first; a piece (time, index) by those the
    copy of the event of that index starts after its first."""

    __slots__ = ("pieces",)

    def __init__(self, *pieces: str | int | tuple[int, int]) -> None:
        self.pieces = pieces

    def render(self, shift: _Moved) -> str:
        return "".join(p if isinstance(p, str) else str(_at(p, shift)) for p in self.pieces)


class _Copies:
    """The violations ``parts`` of copies ``first`` to ``count`` - 1 of a
    loop, copy j shifted by ``offset`` + j * ``period`` cycles, or with
    ``period`` by event index, each event's time by j times its own."""

    __slots__ = ("parts", "first", "count", "period", "offset", "inner", "size")

    def __init__(
        self,
        parts: list[Any],
        first: int,
        count: int,
        period: int | dict[int, int],
        offset: int = 0,
    ) -> None:
        self.parts, self.first, self.count = parts, first, count
        self.period, self.offset = period, offset
        self.inner = _length(parts)
        self.size = (count - first) * self.inner


def _length(parts: Sequence[Any]) -> int:
    return sum(p.size if type(p) is _Copies else 1 for p in parts)


def _render(part: Any, shift: _Moved) -> str:
    return part if type(part) is str else part.render(shift)


def _each(parts: Sequence[Any], shift: _Moved) -> Iterator[str]:
    for part in parts:
        if type(part) is _Copies:
            if not part.size:
                continue
            for j in range(part.first, part.count):
                yield from _each(part.parts, _moved(shift, part.offset, j, part.period))
        else:
            yield _render(part, shift)


class Violations(Sequence[str]):
    """The limits a program exceeds, each built when read: the checks of a
    rolled loop's body stand for every iteration it plays. ``len()`` is
    exact; indexing, slicing and iterating give what a list of every
    violation would."""

    __slots__ = ("_parts", "_size")

    def __init__(self, parts: list[Any]) -> None:
        self._parts = parts
        self._size = _length(parts)

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> list[str]: ...
    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(self._size))]
        k = operator.index(index)
        if k < 0:
            k += self._size
        if not 0 <= k < self._size:
            raise IndexError("violation index out of range")
        parts: Sequence[Any] = self._parts
        shift: _Moved = 0
        while True:
            for part in parts:
                if type(part) is _Copies:
                    if k < part.size:
                        j, k = divmod(k, part.inner)
                        shift = _moved(shift, part.offset, part.first + j, part.period)
                        parts = part.parts
                        break
                    k -= part.size
                elif k == 0:
                    return _render(part, shift)
                else:
                    k -= 1

    def __iter__(self) -> Iterator[str]:
        return _each(self._parts, 0)

    def __repr__(self) -> str:
        return f"<{self._size} controller violations>"


class _Block:
    """Nodes of ``PulseModule.event_tree()``: ops, and loops as (trips,
    their body's block), with where each starts among the events listed."""

    __slots__ = ("items", "offsets", "size")

    def __init__(self, nodes: Sequence[Any]) -> None:
        self.items: list[Any] = []
        self.offsets: list[int] = []
        self.size = 0
        for node in nodes:
            if len(node) == 2:
                body = _Block(node[1])
                size = node[0] * body.size
                if not size:
                    continue
                node = (node[0], body)
            else:
                size = 1
            self.items.append(node)
            self.offsets.append(self.size)
            self.size += size


class Schedule(Sequence[Any]):
    """``ScheduledKernel.schedule`` past ``LAZY_AFTER`` events: each event built when
    read from its op and the iteration of each loop around it, whose start
    is the op's first plus each iteration's shift. ``len()`` is exact;
    indexing, slicing and iterating give what a list of every event would."""

    __slots__ = ("_event", "_root")

    def __init__(self, nodes: Sequence[Any], event: Callable[..., Any]) -> None:
        self._root = _Block(nodes)
        self._event = event

    def __len__(self) -> int:
        return self._root.size

    @overload
    def __getitem__(self, index: int) -> Any: ...
    @overload
    def __getitem__(self, index: slice) -> list[Any]: ...
    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(self._root.size))]
        k = operator.index(index)
        if k < 0:
            k += self._root.size
        if not 0 <= k < self._root.size:
            raise IndexError("schedule index out of range")
        block, at = self._root, []
        while True:
            j = bisect.bisect_right(block.offsets, k) - 1
            item = block.items[j]
            k -= block.offsets[j]
            if len(item) == 2:
                trips, block = item
                i, k = divmod(k, block.size)
                at.append(i)
            else:
                return self._made(item, at)

    def __iter__(self) -> Iterator[Any]:
        return self._each(self._root, [])

    def _each(self, block: _Block, at: list[int]) -> Iterator[Any]:
        for item in block.items:
            if len(item) == 2:
                trips, body = item
                at.append(0)
                for i in range(trips):
                    at[-1] = i
                    yield from self._each(body, at)
                at.pop()
            else:
                yield self._made(item, at)

    def _made(self, op: tuple[Any, ...], at: Sequence[int]) -> Any:
        kind, line, qudit, start, duration, shifts = op
        if start is not None:
            start += sum(i * s for i, s in zip(at, shifts, strict=True) if s is not None)
        return self._event(kind, line, qudit, start, duration)

    def latest_end(self, timed_only: bool = False) -> int | None:
        """The latest end of an event, or ``None`` when some event has no
        start or duration yet; with *timed_only*, the latest end of those
        that have both."""
        end = 0
        work: list[tuple[_Block, tuple[int, ...]]] = [(self._root, ())]
        while work:
            block, trips = work.pop()
            for item in block.items:
                if len(item) == 2:
                    work.append((item[1], (*trips, item[0])))
                    continue
                _, _, _, start, duration, shifts = item
                if start is None or duration is None:
                    if timed_only:
                        continue
                    return None
                # Each loop's iterations shift the op alike: the latest takes
                # each loop's last iteration that moves it later.
                late = sum(
                    (t - 1) * s
                    for t, s in zip(trips, shifts, strict=True)
                    if s is not None and s > 0
                )
                end = max(end, start + late + duration)
        return end

    def __repr__(self) -> str:
        return f"<{self._root.size} scheduled events>"


def _tree(
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    periods: Sequence[Sequence[int]] = (),
) -> tuple[list[Any], list[tuple[_Rep, ...]]]:
    """The indices of *events* with their repeats as nested nodes, and the
    repeats each event lies in, outermost first; *periods* the (repeat,
    line, period) of repeats whose lines have periods of their own."""
    count = len(events)
    lines: dict[int, dict[int, int]] = {}
    for index, line, period in periods:
        lines.setdefault(index, {})[line] = period
    reps = sorted(
        (
            _Rep(r[0], r[1], r[2], r[3], r[4], lines.get(k, {}) if r[3] < 0 else None)
            for k, r in enumerate(repeats)
            if r[1] > r[0]
        ),
        key=lambda r: (r.begin, -r.end),
    )
    for rep in reps:
        if rep.lines is not None:
            # A classical call's submit or answer, i events into the body, is
            # on a line of its own, kCallLine - i (ControllerEventWalk).
            for i in range(rep.begin, rep.end):
                event = events[i]
                call = event["kind"] in ("submit", "answer")
                rep.at[i] = -(1 << 62) - (i - rep.begin) if call else event["line"]
    chains: list[tuple[_Rep, ...]] = [()] * count
    at = 0

    def build(lo: int, hi: int, chain: tuple[_Rep, ...]) -> list[Any]:
        nonlocal at
        nodes: list[Any] = []
        i = lo
        while i < hi:
            if at < len(reps) and reps[at].begin == i and reps[at].end <= hi:
                rep = reps[at]
                at += 1
                rep.body = build(rep.begin, rep.end, chain + (rep,))
                nodes.append(rep)
                i = rep.end
            else:
                chains[i] = chain
                nodes.append(i)
                i += 1
        return nodes

    return build(0, count, ()), chains


def _keep(nodes: list[Any], keep: Callable[[int], bool]) -> list[Any]:
    kept: list[Any] = []
    for node in nodes:
        if type(node) is int:
            if keep(node):
                kept.append(node)
        else:
            body = _keep(node.body, keep)
            if body:
                kept.append(node.keeping(body))
    return kept


def _split(nodes: list[Any], key: Callable[[int], Any]) -> dict[Any, list[Any]]:
    """``_keep`` of *nodes* for each value *key* gives an event other than
    ``None``, keeping the events it gives that value, in one pass."""
    split: dict[Any, list[Any]] = {}
    for node in nodes:
        if type(node) is int:
            found = key(node)
            if found is not None:
                split.setdefault(found, []).append(node)
        else:
            for found, body in _split(node.body, key).items():
                split.setdefault(found, []).append(node.keeping(body))
    return split


def _count(nodes: list[Any]) -> int:
    return sum(1 if type(n) is int else n.copies * _count(n.body) for n in nodes)


def _events_of(nodes: list[Any]) -> Iterator[int]:
    for node in nodes:
        if type(node) is int:
            yield node
        else:
            yield from _events_of(node.body)


def _each_event(
    nodes: list[Any], make: Callable[[int], Any], lines: Iterable[int] | None = None
) -> list[Any]:
    """The violation *make* gives for each event of *nodes* (or ``None``),
    in program order; their events are on *lines*, else each on its own."""
    parts: list[Any] = []
    for node in nodes:
        if type(node) is int:
            part = make(node)
            if part is not None:
                parts.append(part)
        else:
            made: list[int] = []

            def making(i: int, made: list[int] = made) -> Any:
                part = make(i)
                if part is not None:
                    made.append(i)
                return part

            body = _each_event(node.body, making, lines)
            if body:
                if lines is not None:
                    on: Iterable[int] = lines
                else:
                    on = {node.at[i] for i in made} if node.lines is not None else ()
                parts.append(_Copies(body, 0, node.copies, _period_of(node, on, made)))
    return parts


def _period_of(rep: _Rep, lines: Iterable[int], made: Iterable[int]) -> int | dict[int, int]:
    """The period of *rep*'s copies on *lines*, else (lines of different
    periods) that of the line of each event of *made*, the indices of the
    events whose violations it repeats, which tag their times."""
    try:
        return _period(rep, lines)
    except UnsyncedError:
        assert rep.lines is not None
        return {i: rep.lines.get(rep.at[i], 0) for i in made}


class _Budget:
    def __init__(self) -> None:
        self.left = _MAX_PLAYED

    def spend(self, copies: int) -> None:
        self.left -= copies
        if self.left < 0:
            raise ValueError(
                f"the controller check plays more than {_MAX_PLAYED} copies of rolled loops "
                "one by one, whose events interleave with others on one line or depend on a "
                "readout before the loop; give the loops fewer iterations"
            )


def walked_copies(
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    periods: Sequence[Sequence[int]] = (),
    unsynced: bool = False,
) -> int:
    """The copies of rolled loops a walk of every iteration plays: of each
    loop of *repeats*, its copies times those of the loops around it. With
    *unsynced*, of the loops whose lines move by periods of their own
    alone, which a walk that summarizes only loops of one period plays."""
    tree, _ = _tree(events, repeats, periods)

    def count(nodes: list[Any], times: int) -> int:
        total = 0
        for node in nodes:
            if type(node) is not int:
                if unsynced and node.lines is None:
                    # Summarized by that walk: its body is listed once.
                    total += count(node.body, times)
                    continue
                inner = times * int(node.copies)
                total += inner + count(node.body, inner)
        return total

    return count(tree, 1)


def refuse_past_budget(copies: int) -> None:
    """Stop a check that would walk *copies* of rolled loops one by one when
    they are more than a check plays (``_MAX_PLAYED``): walking them takes
    time and memory in proportion to them."""
    if copies > _MAX_PLAYED:
        raise UnsupportedError(
            f"the controller check would play {copies} copies of rolled loops one by one, more "
            f"than the {_MAX_PLAYED} it plays: their events interleave with others on one line "
            "or depend on a readout before the loop, so they cannot be counted from one copy; "
            "give the loops fewer iterations"
        )


# An event of a check: (index, shift in cycles, the copies it is in).
_At = tuple[int, int, tuple[Any, ...]]


class _Pairs:
    """Each event of a line or detector with the latest earlier one it can
    follow on one path (``controller._pairs``), a loop's copies alike."""

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]],
        pair: Callable[[Mapping[str, Any], int, Mapping[str, Any], int], Any],
        budget: _Budget,
        lines: Iterable[int],
    ) -> None:
        self.events, self.pair, self.budget = events, pair, budget
        self.lines = tuple(lines)
        self.windows: dict[int, tuple[int, int]] = {}

    def period(self, rep: _Rep) -> int:
        return _period(rep, self.lines)

    def window(self, node: Any) -> tuple[int, int]:
        """The first and last start of *node* (all its copies)."""
        if type(node) is int:
            start = self.events[node]["start_cycles"]
            return start, start
        found = self.windows.get(id(node))
        if found is None:
            lo, hi = self.body_window(node)
            found = self.windows[id(node)] = (lo, hi + (node.copies - 1) * self.period(node))
        return found

    def body_window(self, rep: _Rep) -> tuple[int, int]:
        spans = [self.window(child) for child in rep.body]
        return min(s[0] for s in spans), max(s[1] for s in spans)

    def ordered(self, items: list[tuple[Any, int, tuple[Any, ...]]]) -> list[Any]:
        """*items* in start order, a repeat a block of copies that each come
        whole; one that does not is played copy by copy."""
        while True:
            spans = [self.window(node) for node, _, _ in items]
            order = sorted(range(len(items)), key=lambda k: (spans[k][0] + items[k][1], k))
            bad = None
            for k in order:
                node = items[k][0]
                if type(node) is not int:
                    lo, hi = self.body_window(node)
                    if hi - lo > self.period(node):
                        bad = k
                        break
            if bad is None:
                for a, b in zip(order, order[1:], strict=False):
                    hi = spans[a][1] + items[a][1]
                    lo = spans[b][0] + items[b][1]
                    if hi > lo or (hi == lo and a > b):
                        bad = a if type(items[a][0]) is not int else b
                        break
            if bad is None:
                return [items[k] for k in order]
            node, dt, path = items[bad]
            self.budget.spend(node.copies)
            items = (
                items[:bad]
                + [
                    (child, dt + j * self.period(node), path + ((id(node.orig), j),))
                    for j in range(node.copies)
                    for child in node.body
                ]
                + items[bad + 1 :]
            )

    def apart(self, a: _At, b: _At) -> bool:
        """Whether events *a* and *b* lie in different arms of one branch
        execution: never in two copies of one loop, which share the arms of
        the branches around it and run those in it apart."""
        for x, y in zip(a[2], b[2], strict=False):
            if x != y:
                if (x if type(x) is int else x[0]) == (y if type(y) is int else y[0]):
                    return False
                break
        return _exclusive(self.events[a[0]], self.events[b[0]])

    def walk(
        self, items: list[tuple[Any, int, tuple[Any, ...]]], seen: list[_At]
    ) -> tuple[list[Any], _At | None]:
        """The pairs of *items* after the events *seen* (in start order,
        which *items* are appended to), and the last of them. An event's
        mate is the latest one seen that is not in another arm of a branch
        execution it is in, a loop in an arm too."""
        events = self.events
        parts: list[Any] = []
        last = seen[-1] if seen else None
        for node, dt, path in self.ordered(items):
            if type(node) is int:
                event = events[node]
                at = (node, dt, path)
                for mate in reversed(seen):
                    if not self.apart(mate, at):
                        part = self.pair(events[mate[0]], mate[1], event, dt)
                        if part is not None:
                            parts.append(part)
                        break
                last = at
                seen.append(at)
                continue
            rid, depth = id(node.orig), len(path)
            inner = path + (rid,)
            first, last = self.walk([(child, dt, inner) for child in node.body], seen)
            parts.extend(first)
            assert last is not None
            if node.copies > 1:
                period = self.period(node)
                # A copy follows the last event of the copy before it.
                again, end = self.walk(
                    [(child, 0, inner) for child in node.body],
                    [(last[0], last[1] - dt - period, _copied(last[2], depth, (rid, -1)))],
                )
                if again:
                    parts.append(_Copies(again, 1, node.copies, period, dt))
                assert end is not None
                last = (
                    end[0],
                    dt + (node.copies - 1) * period + end[1],
                    _copied(end[2], depth, (rid, node.copies - 1)),
                )
                seen.append(last)
        return parts, last


def _copied(path: tuple[Any, ...], depth: int, copy: tuple[int, int]) -> tuple[Any, ...]:
    """*path* with its loop at *depth* in copy *copy* (its id, copy)."""
    return (*path[:depth], copy, *path[depth + 1 :])


def _rolled_loops(
    nodes: list[Any], events: Sequence[Mapping[str, Any]]
) -> list[tuple[int, int, int]]:
    """``controller._rolled_loops`` of the events of *nodes*, every copy."""
    loops: list[list[int]] = []
    previous = -1

    def scan(nodes: list[Any]) -> None:
        nonlocal previous
        for node in nodes:
            if type(node) is int:
                e = events[node]
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
                continue
            scan(node.body)
            rest = node.copies - 1
            if rest:
                loop = loops[-1]
                loop[2] += rest * _count(node.body)
                if node.depth == 0:
                    previous += rest
                    loop[0] = max(loop[0], previous + 1)

    scan(nodes)
    return [(trips, start, extra) for trips, start, extra in loops if extra]


def violations(
    lowered: ScheduledKernel,
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    limits: Controller,
    lines: Sequence[Mapping[str, Any]],
    unresolved: list[str] | None,
    sample_rate: tuple[float, float] | None = None,
    periods: Sequence[Sequence[int]] = (),
    own: ControllerClock | None = None,
    off: Sequence[tuple[int, int, int]] = (),
    natives: Sequence[int] | None = None,
) -> list[str] | Violations:
    """``controller._violations`` of the events of a summarizing walk
    (``parts``)."""
    return finished(
        parts(
            lowered,
            events,
            repeats,
            limits,
            lines,
            unresolved,
            sample_rate,
            periods,
            own,
            off,
            natives,
        )
    )


def finished(found: list[Any]) -> list[str] | Violations:
    """The violations of *found*, parts of ``parts``: a list, or past
    ``LAZY_AFTER`` built when read."""
    if _length(found) <= LAZY_AFTER:
        return list(_each(found, 0))
    return Violations(found)


class _Labeled:
    """A violation of one controller of several: ``label``, then ``part``."""

    __slots__ = ("label", "part")

    def __init__(self, label: str, part: Any) -> None:
        self.label, self.part = label, part

    def render(self, shift: _Moved) -> str:
        return self.label + _render(self.part, shift)


def labeled(found: list[Any], label: str) -> list[Any]:
    """The parts *found* of ``parts``, each violation after *label*."""
    out: list[Any] = []
    for part in found:
        if type(part) is _Copies:
            out.append(
                _Copies(
                    labeled(part.parts, label), part.first, part.count, part.period, part.offset
                )
            )
        elif type(part) is str:
            out.append(label + part)
        else:
            out.append(_Labeled(label, part))
    return out


def _timeouts(
    lowered: ScheduledKernel, events: Sequence[Mapping[str, Any]], tree: list[Any]
) -> list[Any]:
    """The waits for an answer among *events* that have a timeout, for a
    controller that cannot stop waiting."""
    functions = {fn.name: fn for fn in lowered.classical}

    def timeout(i: int) -> _Text | None:
        event = events[i]
        if event["kind"] != "answer":
            return None
        fn = functions.get(event["function"])
        if event["wait"] != "timeout" and (fn is None or fn.latency_mode != "timeout"):
            return None
        return _Text(
            f"the wait for {event['function']} at ",
            (event["start_cycles"], i),
            " has a timeout; the controller cannot stop waiting for an answer, so take it "
            "with result() (no timeout)",
        )

    return _each_event(tree, timeout)


class Shared:
    """What ``parts`` of the controllers of one control system share, made
    once for all of them: the tree of the system's events with the loops
    each lies in; by controller, the pulses it plays (``owner`` gives the
    controller of an event that is played, else ``None``); and the waits
    with a timeout, the same for every controller that cannot stop
    waiting."""

    __slots__ = ("tree", "chains", "timed", "owner", "_timeouts")

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]],
        repeats: Sequence[Sequence[int]],
        periods: Sequence[Sequence[int]],
        owner: Callable[[int], int | None],
    ) -> None:
        self.tree, self.chains = _tree(events, repeats, periods)
        self.owner = owner
        self.timed: dict[int, list[Any]] = _split(self.tree, owner)
        self._timeouts: list[Any] | None = None

    def timeouts(self, lowered: ScheduledKernel, events: Sequence[Mapping[str, Any]]) -> list[Any]:
        if self._timeouts is None:
            self._timeouts = _timeouts(lowered, events, self.tree)
        return self._timeouts


def parts(
    lowered: ScheduledKernel,
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    limits: Controller,
    lines: Sequence[Mapping[str, Any]],
    unresolved: list[str] | None,
    sample_rate: tuple[float, float] | None = None,
    periods: Sequence[Sequence[int]] = (),
    own: ControllerClock | None = None,
    off: Sequence[tuple[int, int, int]] = (),
    natives: Sequence[int] | None = None,
    whole: Shared | None = None,
    index: int | None = None,
) -> list[Any]:
    """``controller._violations`` of the events of a summarizing walk, as
    violations and the copies of a loop's (``finished`` lists them), with
    *sample_rate* (made, plays) when a line plays samples made for another
    rate (``controller._sample_rate_violations``), and *periods* the line
    periods of its unsynced loops; in the cycles of *own*, when the walk
    counted them, with *off* its events off the grid, and *natives* by event
    the runtime qudit whose own frequency its tone plays, or -1.
    ``UnsyncedError`` when a check pairs events of lines of different
    periods. With *whole*, for controller *index* of a control system:
    the pulses it plays among *events*, which are the whole system's."""
    mine: Callable[[int], bool] | None = None
    if whole is None:
        tree, chains = _tree(events, repeats, periods)
        timed = _keep(
            tree, lambda i: events[i]["kind"] in _PLAYED and events[i]["end_cycles"] is not None
        )
    else:
        assert index is not None
        tree, chains = whole.tree, whole.chains
        timed = whole.timed.get(index, [])
        owner = whole.owner

        def mine(i: int) -> bool:
            return owner(i) == index

    budget = _Budget()
    found: list[Any] = []
    if off:
        assert own is not None
        found.extend(_off_grid(tree, events, lines, own, off))
    most = limits.max_pulses
    if most is not None:
        total = _count(timed)
        if limits._runs_loops:
            held = sum(1 for i in timed if type(i) is int and not any(events[i]["iteration"]))
            if held > most:
                found.append(
                    f"{held} drives and readouts, a rolled loop's body counted once; the "
                    f"controller holds at most {most}"
                )
        elif total > most:
            text = f"{total} drives and readouts; the controller plays at most {most}"
            loops = _rolled_loops(timed, events)
            if loops:
                grown = "; ".join(
                    f"the loop of {trips} iterations from cycle {start} plays {extra} more"
                    for trips, start, extra in loops
                )
                text += (
                    ". It does not support 'loop', so rolled loops are unrolled for it: " + grown
                )
            found.append(text)
    issue, gap = limits.instruction_cycles, limits.instruction_cycles + limits.min_idle_cycles
    num, den = _samples_per_own_cycle(limits, lowered.quantum_system.clock_ghz)
    on_lines = sorted({events[i]["line"] for i in _events_of(timed)})
    shared = limits.waveform_memory is WaveformMemoryScope.SHARED
    stored: dict[Any, int] = {}
    stored_each = 0
    unknown_stored = 0

    def spacing(
        previous: Mapping[str, Any], dp: int, event: Mapping[str, Any], de: int
    ) -> _Text | None:
        distance = event["start_cycles"] + de - previous["start_cycles"] - dp
        if distance == issue or distance >= gap:
            return None
        return _Text(
            f"line {event['line']}: {event['kind']} at ",
            event["start_cycles"] + de,
            f" starts {distance} cycles after the {previous['kind']} at ",
            previous["start_cycles"] + dp,
            f"; the controller needs exactly {issue} or at least {gap}",
        )

    for line in on_lines:
        items = _keep(timed, lambda i: events[i]["line"] == line)  # noqa: B023
        found.extend(
            _Pairs(events, spacing, budget, (line,)).walk([(n, 0, ()) for n in items], [])[0]
        )
        templates = list(_events_of(items))
        if limits.max_frequencies_per_line is not None:
            tones = {
                events[i]["frequency_hz"]
                for i in templates
                if events[i].get("frequency_hz") is not None
            }
            if len(tones) > limits.max_frequencies_per_line:
                found.append(
                    f"line {line}: {len(tones)} tone frequencies; the controller holds "
                    f"{limits.max_frequencies_per_line} per line"
                )
            elif unresolved is not None:
                unknown = _keep(items, lambda i: events[i].get("frequency_hz") is None)
                if natives is None:
                    count = _count(unknown)
                else:
                    count = len({natives[i] for i in _events_of(unknown) if natives[i] >= 0})
                    count += _count(_keep(unknown, lambda i: natives[i] < 0))
                if len(tones) + count > limits.max_frequencies_per_line:
                    unresolved.append(f"line {line}: tone frequencies")
        if limits.max_amplitude is not None:
            found.extend(_amplitudes(lines[line], items, events, limits.max_amplitude, unresolved))
        if _checks_pulse_lengths(limits):
            found.extend(_pulse_lengths(lines[line], items, events, limits))
        if limits.waveform_memory_samples is not None:
            shapes = stored if shared else {}
            unknown_samples = 0
            each = 0
            for i in templates:
                e = events[i]
                key = shape_key(e.get("waveform"))
                samples = -(-e["duration_cycles"] * num // den)
                if key is not None:
                    shapes[(key, e["duration_cycles"])] = samples
                else:
                    copies = _count(_keep(items, lambda k: k == i))  # noqa: B023
                    if unresolved is not None:
                        unknown_samples += copies * samples
                    else:
                        each += copies * samples
            used = sum(shapes.values()) + each
            if not shared and used > limits.waveform_memory_samples:
                found.append(
                    f"line {line}: its waveforms take {used} samples; the controller's "
                    f"waveform memory holds {limits.waveform_memory_samples}"
                )
            elif not shared and unresolved is not None:
                if used + unknown_samples > limits.waveform_memory_samples:
                    unresolved.append(f"line {line}: waveform memory")
            stored_each += each
            unknown_stored += unknown_samples
    if shared and limits.waveform_memory_samples is not None:
        used = sum(stored.values()) + stored_each
        if used > limits.waveform_memory_samples:
            found.append(
                f"its waveforms take {used} samples; the controller's shared waveform "
                f"memory holds {limits.waveform_memory_samples}"
            )
        elif unresolved is not None and used + unknown_stored > limits.waveform_memory_samples:
            unresolved.append("waveform memory")
    if limits.detectors:
        detectors = limits.detectors
        hold = limits.detector_recovery_cycles

        def acquired(
            previous: Mapping[str, Any], dp: int, event: Mapping[str, Any], de: int
        ) -> _Text | None:
            free = previous["end_cycles"] + dp + hold
            if event["start_cycles"] + de >= free:
                return None
            return _Text(
                f"detector {detectors[event['qudit']]}: the readout of qudit {event['qudit']} at ",
                event["start_cycles"] + de,
                " starts before the detector is free at ",
                free,
                f" (the readout of qudit {previous['qudit']} at ",
                previous["start_cycles"] + dp,
                ")",
            )

        readouts = _keep(
            timed, lambda i: events[i]["kind"] == "readout" and events[i]["qudit"] in detectors
        )
        for detector in sorted({detectors[events[i]["qudit"]] for i in _events_of(readouts)}):
            items = _keep(readouts, lambda i: detectors[events[i]["qudit"]] == detector)  # noqa: B023
            joint = {events[i]["line"] for i in _events_of(items)}
            found.extend(
                _Pairs(events, acquired, budget, joint).walk([(n, 0, ()) for n in items], [])[0]
            )
    if limits.feedback_latency_cycles:
        found.extend(
            _Feedback(events, chains, limits.feedback_latency_cycles, budget, mine).walk(timed, {})
        )
    if not limits.supports_timeouts:
        if whole is None:
            found.extend(_timeouts(lowered, events, tree))
        else:
            found.extend(whole.timeouts(lowered, events))
    if sample_rate is not None:
        from .controller import _has_samples

        made_ghz, plays_ghz = sample_rate

        def made_for(i: int) -> _Text | None:
            e = events[i]
            if e["kind"] not in _PLAYED or not _has_samples(e.get("waveform")):
                return None
            return _Text(
                f"line {e['line']}: the {e['kind']} at ",
                (e["start_cycles"], i),
                f" plays samples made for {made_ghz:g} GHz, and the controller's lines play "
                f"{plays_ghz:g} GHz",
            )

        found.extend(_each_event(tree, made_for))
    return found


class _OnProgramClock:
    """A violation stated in program cycles of an event whose copies are
    shifted in a controller's own cycles: ``make`` of its start, ``at``,
    shifted as the event of index ``key``."""

    __slots__ = ("make", "at", "num", "den", "key")

    def __init__(self, make: Callable[[int], str], at: int, ratio: Fraction, key: int) -> None:
        self.make, self.at, self.key = make, at, key
        self.num, self.den = ratio.numerator, ratio.denominator

    def render(self, shift: _Moved) -> str:
        # The periods are whole own cycles of whole timesteps, so a shift
        # is whole program cycles.
        return self.make(self.at + _at((0, self.key), shift) * self.den // self.num)


def _off_grid(
    tree: list[Any],
    events: Sequence[Mapping[str, Any]],
    lines: Sequence[Mapping[str, Any]],
    own: ControllerClock,
    off: Sequence[tuple[int, int, int]],
) -> list[Any]:
    """``_clocks.off_grid_messages`` of every copy of the events of *tree*
    that *off* has (index, start, duration) in program cycles."""
    from ._clocks import _off_grid as message

    at = {index: (start, duration) for index, start, duration in off}

    def make(i: int) -> _OnProgramClock | None:
        found = at.get(i)
        if found is None:
            return None
        start, duration = found
        line, kind = events[i]["line"], events[i]["kind"]
        return _OnProgramClock(
            lambda s: message(own, lines[line], kind, s, duration), start, own.ratio, i
        )

    return _each_event(tree, make)


def _amplitudes(
    line: Mapping[str, Any],
    items: list[Any],
    events: Sequence[Mapping[str, Any]],
    most: float,
    unresolved: list[str] | None,
) -> list[Any]:
    """``controller._over_amplitude`` of the events of *items*."""
    noted = unresolved is None

    def over(i: int) -> _Text | None:
        nonlocal noted
        event = events[i]
        tree = event.get("waveform")
        peak = _peak_amplitude(tree)
        if peak is None and tree is not None and not noted:
            assert unresolved is not None
            unresolved.append(f"line {line['line']}: amplitude")
            noted = True
        if peak is None or peak <= most:
            return None
        return _Text(
            f"line {line['line']} ({line['role']} of qudit {line['qudit']}): the "
            f"{event['kind']} at ",
            event["start_cycles"],
            f" peaks at amplitude {peak:.6g}; the controller plays at most {most:g} "
            "(Controller.max_amplitude)",
        )

    return _each_event(items, over, (line["line"],))


def _pulse_lengths(
    line: Mapping[str, Any],
    items: list[Any],
    events: Sequence[Mapping[str, Any]],
    limits: Controller,
) -> list[Any]:
    """``controller._pulse_length`` of the events of *items*."""

    def off_length(i: int) -> _Text | None:
        event = events[i]
        duration = event["end_cycles"] - event["start_cycles"]
        violation = _pulse_length(line, event["kind"], duration, limits)
        if violation is None:
            return None
        return _Text(violation[0], event["start_cycles"], violation[1])

    return _each_event(items, off_length, (line["line"],))


# The copies of a loop whose lines move by periods of their own that the
# feedback check plays one by one, at most, where an event in a loop inside
# it may start too early after a readout of its copy; past them the loops
# are walked in full.
_MAX_EARLY = 1 << 14


def _moves(period: int | dict[int, int] | None, key: int) -> int:
    """The cycles copies of *period* (``_Copies.period``) move a time of
    the event of index *key* by per copy."""
    if period is None:
        return 0
    return period if isinstance(period, int) else period.get(key, 0)


class _Feedback:
    """The events that start before the readouts they depend on are ready
    (``Controller.feedback_latency_cycles``). A copy of a loop depends
    on the copies of the readouts in it, or in the copy before it, as the
    first copy does; on a readout before the loop only a few first copies
    can be too early, and those are checked one by one.

    Where the lines of a loop move by periods of their own, an event and
    the readout of its copy on lines of different periods drift apart by
    the difference every copy, so it is too early in the copies before, or
    from, the copy where the drift crosses the latency: the loop's copies
    are checked in runs between those copies, each run as its first copy.
    Where that event is in a loop inside, whose first copies may then be
    too early, the copies of the outer loop where they may are checked one
    by one."""

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]],
        chains: Sequence[tuple[_Rep, ...]],
        latency: int,
        budget: _Budget,
        mine: Callable[[int], bool] | None = None,
    ) -> None:
        self.events, self.chains, self.latency, self.budget = events, chains, latency, budget
        # Of a control system's events, whether one is the controller's own.
        self.mine = mine
        # The copy each run of copies of a loop is checked as, and the
        # period of the copies of each loop being walked (``walk``).
        self.at: dict[_Rep, int] = {}
        self.moving: dict[_Rep, int | dict[int, int]] = {}

    def apart(self, i: int, ref: int) -> tuple[list[_Rep], int, _Rep | None]:
        """The loops event *i* is in and readout *ref* is not, whose copy
        decides how far apart the two are, the cycles the readout's copy
        (the last of each loop it is in alone) starts after its first, and
        the loop whose copies move the two by different periods, if any.
        ``UnsyncedError`` for two such loops, or one inside a loop the
        readout is not in."""
        mine, theirs = self.chains[i], self.chains[ref]
        line, readout = self.events[i]["line"], self.events[ref]["line"]
        separate: list[_Rep] = []
        drift = None
        for rep in mine:
            if rep not in theirs and not rep.begin - (rep.end - rep.begin) <= ref < rep.begin:
                separate.append(rep)
                continue
            if rep.lines is not None and _period(rep, (line,)) != _period(rep, (readout,)):
                if drift is not None or separate:
                    raise UnsyncedError
                drift = rep
        later = sum(
            (rep.copies - 1) * _period(rep, (readout,)) for rep in theirs if rep not in mine
        )
        return separate, later, drift

    def period(self, rep: _Rep, i: int) -> int:
        """The period of *rep*'s copies of event *i*."""
        return _period(rep, (self.events[i]["line"],))

    def refs(self, i: int) -> Iterator[tuple[int, Mapping[str, Any]]]:
        for condition in self.events[i]["conditions"]:
            for ref in condition["readouts"]:
                readout = self.events[ref]
                if readout["end_cycles"] is not None and (self.mine is None or self.mine(ref)):
                    yield ref, readout

    def timing(
        self, i: int, ref: int, fixed: Mapping[_Rep, int | None]
    ) -> tuple[int, int, int, _Rep | None, int, bool] | None:
        """Event *i* and readout *ref* at the *fixed* copies of the loops
        they are in (a loop not yet in *fixed* at its first copy): the
        cycles the readout is ready after the event starts, the readout's
        end as a violation states it, the index of the event whose copies
        it moves with, the loop whose copies move the two apart and the
        cycles they do per copy, and whether the event is in a loop inside
        that one that the readout is not; ``None`` for copies of a loop
        that are never early."""
        separate, later, drift = self.apart(i, ref)
        key = i if drift is None else ref
        lead = back = 0
        for rep in separate:
            if rep not in fixed:
                continue
            j = fixed[rep]
            if j is None:
                return None
            lead += j * self.period(rep, i)
            back += j * (self.period(rep, i) if key == i else _moves(self.moving.get(rep), key))
        end = self.events[ref]["end_cycles"] + later
        room = end + self.latency - self.events[i]["start_cycles"] - lead
        if drift is None:
            return room, end - lead, key, None, 0, False
        step = _period(drift, (self.events[ref]["line"],)) - self.period(drift, i)
        j = fixed.get(drift)
        at = self.at.get(drift, 0) if j is None else j
        return room + at * step, end - back, key, drift, step, bool(separate)

    def walk(self, nodes: list[Any], fixed: dict[_Rep, int | None]) -> list[Any]:
        parts: list[Any] = []
        for node in nodes:
            if type(node) is int:
                parts.extend(self.late(node, fixed))
                continue
            rep = node.orig
            period = self.body_period(node, rep)
            head = min(rep.copies, self.head(node, rep, fixed))
            if head:
                self.budget.spend(head)
            for j in range(head):
                self.moving[rep] = period
                body = self.walk(node.body, {**fixed, rep: j})
                if body:
                    parts.append(_Copies(body, j, j + 1, period))
            if head < rep.copies:
                cuts, busy = self.cuts(node, rep, fixed, head)
                cuts = sorted({head, rep.copies, *cuts})
                for first, end in zip(cuts, cuts[1:], strict=False):
                    self.moving[rep] = period
                    if any(lo <= first < hi for lo, hi in busy):
                        if end - first > min(_MAX_EARLY, self.budget.left):
                            raise UnsyncedError
                        self.budget.spend(end - first)
                        for j in range(first, end):
                            body = self.walk(node.body, {**fixed, rep: j})
                            if body:
                                parts.append(_Copies(body, j, j + 1, period))
                        continue
                    self.at[rep] = first
                    body = self.walk(node.body, {**fixed, rep: None})
                    if body:
                        parts.append(_Copies(body, first, end, period))
                self.at.pop(rep, None)
            self.moving.pop(rep, None)
        return parts

    def body_period(self, node: _Rep, rep: _Rep) -> int | dict[int, int]:
        """The period of *rep*'s copies of the events of *node* that
        depend on readouts, or by event index, where they and the readouts
        of their copies move by different periods."""
        if rep.lines is None:
            return rep.period
        lines = set()
        for i in _events_of(node.body):
            if self.events[i]["conditions"]:
                lines.add(self.events[i]["line"])
                for ref, readout in self.refs(i):
                    if rep.begin - (rep.end - rep.begin) <= ref < rep.end:
                        lines.add(readout["line"])
        try:
            return _period(rep, lines)
        except UnsyncedError:
            size = rep.end - rep.begin
            return {
                k: rep.lines.get(rep.at[k if k >= rep.begin else k + size], 0)
                for k in range(rep.begin - size, rep.end)
            }

    def cuts(
        self, node: _Rep, rep: _Rep, fixed: Mapping[_Rep, int | None], head: int
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """The copies of *rep* after *head* where an event of *node* and a
        readout it depends on, moving apart, become or stop being early,
        and the runs of copies where an event in a loop inside may be."""
        if rep.lines is None:
            return [], []
        found: list[int] = []
        busy: list[tuple[int, int]] = []
        saved = self.at.pop(rep, None)
        for i in _events_of(node.body):
            for ref, _ in self.refs(i):
                timing = self.timing(i, ref, fixed)
                if timing is None or timing[3] is not rep:
                    continue
                room, _, _, _, step, inside = timing
                # Early at copy j: room + j * step > 0.
                if step > 0:
                    cut = -room // step + 1
                    early = (cut, rep.copies)
                else:
                    cut = -(-room // -step)
                    early = (0, cut)
                if head < cut < rep.copies:
                    found.append(cut)
                if inside:
                    busy.append(early)
        if saved is not None:
            self.at[rep] = saved
        return found, busy

    def head(self, node: _Rep, rep: _Rep, fixed: Mapping[_Rep, int | None]) -> int:
        """The first copies of *rep* that may start too early after a readout
        outside it."""
        copies = 0
        for i in _events_of(node.body):
            for ref, _ in self.refs(i):
                if rep not in self.apart(i, ref)[0]:
                    continue
                timing = self.timing(i, ref, fixed)
                if timing is not None and timing[0] > 0:
                    period = self.period(rep, i)
                    copies = max(
                        copies,
                        math.ceil(timing[0] / period) if period > 0 else rep.copies,
                    )
        return copies

    def late(self, i: int, fixed: Mapping[_Rep, int | None]) -> list[_Text]:
        event = self.events[i]
        found = []
        for ref, _ in self.refs(i):
            timing = self.timing(i, ref, fixed)
            if timing is None or timing[0] <= 0:
                continue
            _, end, key, _, _, _ = timing
            found.append(
                _Text(
                    f"line {event['line']}: the {event['kind']} at ",
                    (event["start_cycles"], i),
                    " depends on the readout ending at ",
                    (end, key),
                    f"; the controller needs {self.latency} cycles of feedback latency (start at ",
                    (end + self.latency, key),
                    " or later; schedule with rcp and "
                    "readout_latency_cycles >= feedback_latency_cycles)",
                )
            )
        return found


def loops_of(
    repeats: Sequence[Sequence[int]], periods: Sequence[Sequence[int]] | None
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The loops a summarizing walk's *repeats* and line *periods* keep
    rolled, whatever clock it counts their periods in."""
    return (
        [(*r[:3], r[3] < 0, *r[4:]) for r in repeats],
        [tuple(p[:2]) for p in periods or ()],
    )


def summarized_check(
    lowered: ScheduledKernel,
    limits: Controller | None,
    system: ControlSystem | None,
    clocks: ClockPlan | None,
    slots: bool,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[dict[str, Any]],
    repeats: list[tuple[int, ...]],
    periods: list[tuple[int, int, int]] | None,
    off: Sequence[tuple[int, int, int]],
) -> tuple[Sequence[str], bool] | None:
    """``controller._compile_check`` of the events of a summarizing walk with
    *repeats* (and line *periods*); ``None`` when the loops must be walked
    with fewer summarized."""
    machine = lowered.options.machine
    natives = None
    if slots and limits is not None and limits.max_frequencies_per_line is not None:
        with lowered._lock:
            natives, again, moved = lowered._module.controller_native_tones(
                True, periods is not None
            )
        if loops_of(repeats, periods) != loops_of(again, moved):
            return None
    rates = (
        None if limits is None or limits is machine.controller else _sample_rates(lowered, limits)
    )
    pending: list[str] | None = [] if lowered.parameters or slots else None
    if system is not None:
        parts = system._summarized(
            lowered, lines, events, repeats, clocks, off, pending, natives, rates, periods or ()
        )
        if parts is None:
            return None
        found: Sequence[str] = finished(parts)
    else:
        assert limits is not None
        try:
            found = violations(
                lowered,
                events,
                repeats,
                limits,
                lines,
                pending,
                rates,
                periods or (),
                _native_clock(clocks),
                off,
                natives,
            )
        except UnsyncedError:
            return None
    if slots and limits is not None and limits.detectors:
        assert pending is not None
        # Acquirers are given per carrier, which each call binds.
        pending.append("detectors")
    return found, not pending
