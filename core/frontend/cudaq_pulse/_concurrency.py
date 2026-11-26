# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drives and readouts a program plays at once, against the
``max_concurrent_drives`` and ``max_concurrent_readouts`` its control system
gives.

They are counted as the ``rcp`` and ``alap_rcp`` schedules count them, so a
program those schedules place fits: on every controller together, in the
cycles the schedule placed each pulse at (before its line's delay moves
it), a pulse holding its place from its start up to its end, where the next
may start. A wait takes no place. Pulses in different arms of one execution
of a branch never play together.

A long rolled loop is counted from the check walk's summary (``_loops``),
never played iteration by iteration. Pulses that share time are first
bounded: a loop by the most one copy plays times the copies that overlap,
and when that bound is within the limit nothing more is counted. Past it,
loops of one period are counted exactly from the copies near where
something starts or ends, as what plays between repeats with the period. A
group of loops of different periods that the bound does not clear is played
copy by copy when its copies fit ``_MAX_COPIES``. Past that, a count past the
limit among its copies near where something starts or ends is reported at
its cycle, not as the first; with none there, the check stops with an error
rather than walk every iteration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

from ._errors import UnsupportedError

if TYPE_CHECKING:
    from .compiler import ScheduledKernel

_KINDS = (("drive", "max_concurrent_drives"), ("readout", "max_concurrent_readouts"))

# The copies of rolled loops the check plays one by one, at most.
_MAX_COPIES = 1 << 16

# A pulse being counted: (start, end, event, the copies of loops it is in).
_Played = tuple[int, int, Mapping[str, Any], tuple[Any, ...]]
# A node of a group: (``_loops._tree`` node, shift in cycles, its copies).
_Item = tuple[Any, int, tuple[Any, ...]]


def violations(lowered: ScheduledKernel) -> tuple[list[str], bool]:
    """What *lowered* plays at once past the limits its control system
    gives, and whether that is exact: ``False`` when a pulse's time depends
    on values not known yet, which each call's values then settle."""
    machine = lowered.options.machine
    limits = {
        kind: getattr(machine, name) for kind, name in _KINDS if getattr(machine, name) is not None
    }
    if not limits:
        return [], True
    from .controller import _walk

    shifts: Mapping[tuple[str, int], int] = {}
    if machine._line_delays:
        from .compiler import _line_delays

        clocks = (lowered._generic or lowered).clocks
        shifts = _line_delays(machine, lowered.quantum_system, clocks)[0]
    repeats: list[tuple[int, ...]] = []
    periods: list[tuple[int, int, int]] = []
    segments: list[Any] = []
    lines, events, _, _, _ = _walk(
        lowered, values=False, waveforms=False, summary=(repeats, periods), segments=segments
    )
    # What follows a wait whose length is computed at run time, or a wait
    # for an answer, starts later in a run than it is placed, while lines
    # the wait does not hold back keep their times: the pulses of such a
    # program are counted as they can slide, unless their lines all wait
    # together.
    sliding = bool(lowered.classical) and _slides_apart(events, limits)
    if repeats and sliding and not periods and segments:
        # Counted from the trips of each loop the walk listed, and the rest
        # of its trips as blocks, where its loops allow that.
        timed = _timed(events, lines, limits, shifts)
        if timed is None:
            return [], False
        rolled = _rolled(limits, events, timed, lines, repeats, segments)
        if rolled is not None:
            return rolled, True
    if repeats and sliding:
        # Else pulses that can slide are counted from every copy of a loop.
        from . import _loops

        copies = _loops.walked_copies(events, repeats, periods)
        if copies > _loops._MAX_PLAYED:
            raise UnsupportedError(
                f"the controller check would count {copies} copies of rolled loops one by "
                f"one, more than the {_loops._MAX_PLAYED} it plays: what follows a wait "
                "whose length is computed at run time starts later in a run than it is "
                "placed, and the control system limits what plays at once; sync the lines "
                "that play after such a wait, or give the loops fewer iterations"
            )
        repeats = []
        lines, events, _, _, _ = _walk(lowered, values=False, waveforms=False)
    timed = _timed(events, lines, limits, shifts)
    if timed is None:
        return [], False
    found = []
    if not repeats:
        for kind, name in _KINDS:
            if kind not in limits:
                continue
            count = _AtOnce(limits[kind], events, timed)
            if sliding:
                found.extend(_sliding(lowered, count, kind, name, lines))
                continue
            count.sweep(
                sorted(
                    ((*timed[i], events[i], ()) for i in timed if events[i]["kind"] == kind),
                    key=_first,
                )
            )
            found.extend(count.message(kind, name))
        return found, True
    from . import _loops

    tree, chains = _loops._tree(events, repeats, periods)
    kinds = {i: events[i]["kind"] for i in timed}
    split = _loops._split(tree, kinds.get)
    for kind, name in _KINDS:
        if kind not in limits:
            continue
        nodes = split.get(kind, [])
        count = _AtOnce(limits[kind], events, timed)
        count.enclosing = _enclosing(events, chains, nodes)
        count.group([(node, 0, ()) for node in nodes])
        found.extend(count.message(kind, name))
    return found, True


def _first(item: _Played) -> tuple[int, int]:
    return item[0], item[1]


def _timed(
    events: Sequence[Mapping[str, Any]],
    lines: Sequence[Mapping[str, Any]],
    limits: Mapping[str, int],
    shifts: Mapping[tuple[str, int], int],
) -> dict[int, tuple[int, int]] | None:
    """By event, each pulse of a kind with a limit: its start and end before
    its line's delay; ``None`` when a time is not known yet."""
    timed: dict[int, tuple[int, int]] = {}
    for index, event in enumerate(events):
        if event["kind"] not in limits:
            continue
        start, end = event["start_cycles"], event["end_cycles"]
        if start is None or end is None:
            return None
        if shifts:
            line = lines[event["line"]]
            shift = shifts.get((line["role"], line["qudit"]), 0)
            start, end = start - shift, end - shift
        timed[index] = (start, end)
    return timed


def _rolled(
    limits: Mapping[str, int],
    events: Sequence[Mapping[str, Any]],
    timed: Mapping[int, tuple[int, int]],
    lines: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    segments: Sequence[Any],
) -> list[str] | None:
    """What can play at once past *limits* in some run, counted from a
    walk that lists three trips of each long loop (``_slides.over_rolled``);
    ``None`` when its loops cannot be counted so."""
    from . import _slides

    found: list[str] = []
    for kind, name in _KINDS:
        if kind not in limits:
            continue
        count = _AtOnce(limits[kind], events, timed)
        played = sorted(
            ((*timed[i], events[i]) for i in timed if events[i]["kind"] == kind),
            key=_slides.by_start,
        )

        def together(active: list[Any], count: _AtOnce = count) -> int:
            return _together([(count._arms(pulse[2]["conditions"], ()), 1) for pulse in active])

        branched = any(pulse[2]["conditions"] for pulse in played)
        counted, over = _slides.over_rolled(
            played, count.limit, lines, events, repeats, segments, together if branched else None
        )
        if not counted:
            return None
        if over is not None:
            found.append(_can_play(over, kind, name, count.limit))
    return found


def _can_play(found: Any, kind: str, name: str, limit: int) -> str:
    beyond = f", and up to {found.most}" if found.most > found.at_first else ""
    return (
        f"{found.at_first} {kind}s can play at once from cycle {found.first}{beyond}: "
        f"{found.cause}; the control system plays at most {limit} ({name}): sync the "
        "lines after the wait, so that they wait together, or give the wait bounds that keep "
        "them apart"
    )


def _slides_apart(events: Sequence[Mapping[str, Any]], limits: Mapping[str, int]) -> bool:
    """Whether pulses of a kind *limits* counts can slide against one
    another: one is past the first segment, and they are not all on lines
    that wait together."""
    from . import _slides

    for kind in limits:
        counted = [e for e in events if e["kind"] == kind]
        if any(e["segment"] for e in counted) and not _slides.one_timeline(counted):
            return True
    return False


def _sliding(
    lowered: ScheduledKernel,
    count: _AtOnce,
    kind: str,
    name: str,
    lines: Sequence[Mapping[str, Any]],
) -> list[str]:
    """The *kind*s of *count* that can play at once past its limit in some
    run, when what follows each wait of a run-time length starts as much
    later as the wait can be longer (``_slides``)."""
    from . import _slides

    events, timed = count.events, count.timed
    played = sorted(
        ((*timed[i], events[i]) for i in timed if events[i]["kind"] == kind),
        key=_slides.by_start,
    )

    def together(active: list[Any]) -> int:
        return _together([(count._arms(pulse[2]["conditions"], ()), 1) for pulse in active])

    branched = any(pulse[2]["conditions"] for pulse in played)
    found = _slides.over(lowered, played, count.limit, lines, together if branched else None)
    if found is None:
        return []
    return [_can_play(found, kind, name, count.limit)]


def _second(item: tuple[int, int]) -> int:
    return item[1]


class _AtOnce:
    """The pulses of one kind at once past *limit*: the first cycle they
    are more, how many then, and the most at once."""

    __slots__ = (
        "limit",
        "events",
        "timed",
        "first",
        "at_first",
        "most",
        "enclosing",
        "copies",
        "sampled",
        "first_sampled",
    )

    def __init__(
        self,
        limit: int,
        events: Sequence[Mapping[str, Any]],
        timed: Mapping[int, tuple[int, int]],
    ) -> None:
        self.limit = limit
        self.events = events
        self.timed = timed
        self.first: int | None = None
        self.at_first = 0
        self.most = 0
        # By branch id, the rolled loops (by id) its executions repeat in.
        self.enclosing: Mapping[int, frozenset[int]] = {}
        # The copies of rolled loops played one by one so far.
        self.copies = 0
        # Whether the pulses swept now are copies near edges only (loops of
        # different periods past _MAX_COPIES), and whether first came from
        # such a sweep, so that it is not shown to be the first.
        self.sampled = False
        self.first_sampled = False

    def message(self, kind: str, name: str) -> list[str]:
        if self.first is None:
            return []
        beyond = f", and up to {self.most}" if self.most > self.at_first else ""
        when = f"from cycle {self.first}{beyond}"
        if self.first_sampled:
            when = (
                f"at cycle {self.first} (rolled loops of different periods, too many copies to "
                "count one by one for the first such cycle)"
            )
        return [
            f"{self.at_first} {kind}s play at once {when}; the control "
            f'system plays at most {self.limit} ({name}): compile with schedule="rcp" or '
            '"alap_rcp", which keep to it'
        ]

    # -- Pulses ------------------------------------------------------------

    def sweep(self, played: Sequence[_Played]) -> None:
        """Count *played*, sorted by start; a pulse ending at a cycle frees
        its place for one starting then."""
        # The pulses in flight, the one that ends first on top: (end, the
        # pulse's place in *played*). Their arms are worked out only where
        # more are in flight than the limit, in the order they started.
        active: list[tuple[int, int]] = []
        arms: dict[int, dict[tuple[Any, ...], Any]] = {}
        for order, (start, end, _, _) in enumerate(played):
            while active and active[0][0] <= start:
                arms.pop(heappop(active)[1], None)
            heappush(active, (end, order))
            if len(active) <= self.limit:
                continue
            together = []
            for _, k in sorted(active, key=_second):
                if k not in arms:
                    arms[k] = self._arms(played[k][2]["conditions"], played[k][3])
                together.append((arms[k], 1))
            count = _together(together)
            if count <= self.limit:
                continue
            if self.first is None or start < self.first:
                self.first, self.at_first = start, count
                self.first_sampled = self.sampled
            elif start == self.first:
                self.at_first = max(self.at_first, count)
            self.most = max(self.most, count)

    def _arms(
        self, conditions: Sequence[Mapping[str, Any]], path: tuple[Any, ...]
    ) -> dict[tuple[Any, ...], Any]:
        """The arm of each branch execution of *conditions*, for a pulse in
        the copies *path*. The walk numbers each execution of a branch; the
        copies of a summarized loop around the branch share its numbers,
        told apart by the copy."""
        arms = {}
        for condition in conditions:
            branch = condition["branch"]
            loops = self.enclosing.get(branch, frozenset())
            copy = tuple(c for c in path if (c[0] if type(c) is tuple else c) in loops)
            arms[(branch, copy)] = condition["arm"]
        return arms

    # -- Groups of nodes ---------------------------------------------------

    def group(self, items: list[_Item]) -> None:
        """Count *items*, the nodes of one kind that may share time."""
        items = [part for item in items for part in self._by_period(item)]
        if not items:
            return
        if self._bound(items, self.limit) <= self.limit:
            return
        windows = [_shifted(self._window(node), dt) for node, dt, _ in items]
        crowded = []
        for k, (node, _, _) in enumerate(items):
            if type(node) is int:
                continue
            lo, hi = windows[k]
            alone = all(w[1] <= lo or w[0] >= hi for j, w in enumerate(windows) if j != k)
            if not (alone and self._span(node) <= self._period(node)):
                crowded.append(k)
        if not crowded:
            played = []
            for node, dt, path in items:
                if type(node) is int:
                    start, end = self.timed[node]
                    played.append((start + dt, end + dt, self.events[node], path))
                else:
                    # Its copies follow each other with no other pulse of the
                    # kind meanwhile: each plays as the first does.
                    self.group([(child, dt, (*path, id(node.orig))) for child in node.body])
            self.sweep(sorted(played, key=_first))
            return
        loops = [items[k] for k in crowded]
        cycles = {self._period(node) for node, _, _ in loops}
        if len(cycles) == 1 and min(cycles) > 0:
            # One period: between two cycles where something starts or ends,
            # what plays repeats with the period, so the copies within a
            # period and a copy's length after each such cycle show it all.
            period = cycles.pop()
            reach = period + max(self._span(node) for node, _, _ in loops)
            edges = sorted({edge for w in windows for edge in w})
            expanded = [item for k, item in enumerate(items) if k not in set(crowded)]
            for node, dt, path in loops:
                expanded.extend(self._copies(node, dt, path, period, edges, reach))
            self.group(expanded)
            return
        # Loops of different periods: every copy, when they fit _MAX_COPIES.
        if self.copies + sum(int(node.copies) for node, _, _ in loops) > _MAX_COPIES:
            # Else their copies near where something starts or ends: what they
            # play is played, so a count past the limit there is one, though
            # not shown to be the first.
            edges = sorted({edge for w in windows for edge in w})
            near = [item for k, item in enumerate(items) if k not in set(crowded)]
            for node, dt, path in loops:
                step = self._period(node)
                if step <= 0:
                    break
                near.extend(self._copies(node, dt, path, step, edges, step + self._span(node)))
            else:
                self.sampled = True
                try:
                    self.group(near)
                finally:
                    self.sampled = False
                if self.first is not None:
                    return
        expanded = [item for k, item in enumerate(items) if k not in set(crowded)]
        for node, dt, path in loops:
            self._spend(node.copies)
            step = self._period(node)
            expanded.extend(
                (child, dt + j * step, (*path, (id(node.orig), j)))
                for j in range(node.copies)
                for child in node.body
            )
        self.group(expanded)

    def _copies(
        self,
        node: Any,
        dt: int,
        path: tuple[Any, ...],
        period: int,
        edges: Sequence[int],
        reach: int,
    ) -> list[_Item]:
        """The children of the copies of *node* that play within *reach*
        cycles after one of *edges*."""
        lo = self._window(node)[0] + dt
        span = self._span(node)
        last = node.copies - 1
        chosen: set[int] = set()
        for edge in edges:
            first = max(0, (edge - span - lo) // period + 1)
            final = min(last, -(-(edge + reach - lo) // period) - 1)
            if final >= first:
                self._spend(final - first + 1)
                chosen.update(range(first, final + 1))
        return [
            (child, dt + j * period, (*path, (id(node.orig), j)))
            for j in sorted(chosen)
            for child in node.body
        ]

    def _spend(self, copies: int) -> None:
        self.copies += copies
        if self.copies > _MAX_COPIES:
            raise UnsupportedError(
                "the max_concurrent_drives and max_concurrent_readouts check would play more "
                f"than {_MAX_COPIES} copies of rolled loops one by one: rolled loops of "
                "different periods overlap and could exceed the limit; give the loops "
                "one period (a sync of their lines in each iteration), fewer iterations, or "
                'schedule them with schedule="rcp"'
            )

    def _by_period(self, item: _Item) -> list[_Item]:
        """*item*, or a loop whose lines move by different periods as one
        loop per period."""
        node, dt, path = item
        if type(node) is int or node.lines is None:
            return [item]
        from . import _loops

        by_period: dict[int, set[int]] = {}
        for i in _loops._events_of(node.body):
            line = self.events[i]["line"]
            by_period.setdefault(node.lines.get(line, 0), set()).add(line)
        events = self.events
        parts = []
        for period, lines in by_period.items():

            def on(i: int, lines: set[int] = lines) -> bool:
                return events[i]["line"] in lines

            kept = node.keeping(_loops._keep(node.body, on))
            kept.lines, kept.period = None, period
            parts.append((kept, dt, path))
        return parts

    def _period(self, node: Any, least: bool = False) -> int:
        """The cycles between copies of *node*; of a loop whose lines move by
        different periods, the longest (or with *least*, the shortest)."""
        if node.lines is None:
            return int(node.period)
        from ._loops import _events_of

        found = {node.lines.get(self.events[i]["line"], 0) for i in _events_of(node.body)}
        return int(min(found) if least else max(found))

    def _window(self, node: Any) -> tuple[int, int]:
        """The first start and last end of *node*'s pulses, all its copies."""
        if type(node) is int:
            return self.timed[node]
        spans = [self._window(child) for child in node.body]
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        return lo, hi + (node.copies - 1) * self._period(node)

    def _span(self, node: Any) -> int:
        """The cycles from the first start to the last end of one copy."""
        spans = [self._window(child) for child in node.body]
        return max(s[1] for s in spans) - min(s[0] for s in spans)

    def _bound(self, items: Sequence[_Item], floor: int = 0) -> int:
        """At least the most of *items* that play at once, when that is past
        *floor*: a loop counts the most one copy plays times the copies that
        overlap."""
        entries = []
        for node, dt, path in items:
            lo, hi = _shifted(self._window(node), dt)
            entries.append((lo, hi, self._common(node, path), self._weight(node)))
        entries.sort(key=lambda e: (e[0], e[1]))
        most = 0
        # The entries in flight: a heap with the one that ends first on top,
        # (end, the entry's place in *entries*); their places in the order
        # they came in; the sum of their weights; and how many of them are
        # in an arm of a branch.
        active: list[tuple[int, int]] = []
        flying: dict[int, None] = {}
        total = armed = 0
        for order, (lo, hi, arms, weight) in enumerate(entries):
            while active and active[0][0] <= lo:
                gone = heappop(active)[1]
                del flying[gone]
                total -= entries[gone][3]
                if entries[gone][2]:
                    armed -= 1
            heappush(active, (hi, order))
            flying[order] = None
            total += weight
            if arms:
                armed += 1
            if total <= max(most, floor):
                continue
            if not armed:
                # None is in a branch: all in flight play together.
                most = total
                continue
            most = max(most, _together([(entries[k][2], entries[k][3]) for k in flying]))
        return most

    def _weight(self, node: Any) -> int:
        """The most pulses one node plays at once, at most."""
        if type(node) is int:
            return 1
        inner = self._bound([(child, 0, ()) for child in node.body])
        period, span = self._period(node, least=True), self._span(node)
        copies = int(node.copies)
        overlap = copies if period <= 0 else min(copies, max(1, -(-span // period)))
        return inner * overlap

    def _common(self, node: Any, path: tuple[Any, ...]) -> dict[tuple[Any, ...], Any]:
        """The branch arms every pulse of *node* is in."""
        if type(node) is int:
            return self._arms(self.events[node]["conditions"], path)
        from ._loops import _events_of

        shared: set[tuple[Any, Any]] | None = None
        for i in _events_of(node.body):
            mine = {(c["branch"], c["arm"]) for c in self.events[i]["conditions"]}
            shared = mine if shared is None else shared & mine
            if not shared:
                return {}
        conditions = [{"branch": b, "arm": a} for b, a in sorted(shared or (), key=repr)]
        return self._arms(conditions, path)


def _shifted(window: tuple[int, int], dt: int) -> tuple[int, int]:
    return window[0] + dt, window[1] + dt


def _together(active: Sequence[tuple[Mapping[tuple[Any, ...], Any], int]]) -> int:
    """The most pulses of *active* ((arms, pulses)) that play together: those
    of one path through the branches, no two in different arms of one
    branch execution."""
    count = 0
    keyed = []
    for arms, weight in active:
        if arms:
            keyed.append((arms, weight))
        else:
            count += weight
    # Groups that share no branch execution take their best path each.
    while keyed:
        group = [keyed.pop()]
        keys = set(group[0][0])
        grown = True
        while grown:
            grown = False
            for entry in keyed[:]:
                if keys.isdisjoint(entry[0]):
                    continue
                keyed.remove(entry)
                group.append(entry)
                keys.update(entry[0])
                grown = True
        count += _best(group, {})
    return count


def _best(
    group: Sequence[tuple[Mapping[tuple[Any, ...], Any], int]], chosen: dict[tuple[Any, ...], Any]
) -> int:
    """The most of *group* (arms, pulses) that one choice of arms, extending
    *chosen*, plays together."""
    open_key = None
    for arms, _ in group:
        open_key = next((key for key in arms if key not in chosen), None)
        if open_key is not None:
            break
    if open_key is None:
        return sum(weight for _, weight in group)
    best = 0
    for arm in {arms[open_key] for arms, _ in group if open_key in arms}:
        chosen[open_key] = arm
        kept = [entry for entry in group if entry[0].get(open_key, arm) == arm]
        best = max(best, _best(kept, chosen))
        del chosen[open_key]
    return best


def _enclosing(
    events: Sequence[Mapping[str, Any]], chains: Sequence[tuple[Any, ...]], nodes: list[Any]
) -> dict[int, frozenset[int]]:
    """By branch id, the summarized loops around the branch: those around
    every pulse of *nodes* in an arm of it. A loop inside an arm repeats
    the pulses of one execution of the branch."""
    from ._loops import _events_of

    found: dict[int, frozenset[int]] = {}
    for index in _events_of(nodes):
        loops = frozenset(id(rep.orig) for rep in chains[index])
        for condition in events[index]["conditions"]:
            branch = condition["branch"]
            had = found.get(branch)
            found[branch] = loops if had is None else had & loops
    return found
