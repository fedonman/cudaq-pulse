# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How many pulses can play at once when parts of the program slide.

A wait whose length is computed at run time is scheduled at its lowest
length, and what follows it on its lines (a segment, see
``compiler.Segment``) starts later at run time by up to the wait's range.
Lines the wait does not hold back keep their times, so two pulses the
schedule places apart can meet.

The segments are a tree: each follows the one its wait ended. Pulses of one
segment keep their scheduled distances, and the segments that follow a
segment slide against it by 0 to their wait's range, each on its own. So
with ``own(s, t)`` the pulses of segment ``s`` playing at cycle ``t`` as
scheduled, the most that can play at once at ``t``, in and after ``s``, is

    most(s, t) = own(s, t) + sum over the c that follow s of
                 max over 0 <= d <= range(c) of most(c, t - d)

and ``most(0, t)`` is the most over every run, at the cycle ``t`` of a run
whose segments on the way all start as scheduled. The lengths of two waits
are taken to be independent; two that are computed from one answer move
together, and are counted as if they did not.

A count over time is a step function: ``[(cycle, count), ...]`` by cycle,
each count holding up to the next cycle, 0 before the first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from .compiler import ScheduledKernel

# The waits a message names, at most.
NAMED = 3

# The cycle a count that holds for good ends at.
_NEVER = float("inf")

# A pulse: (start, end, event).
Pulse = tuple[int, int, Mapping[str, Any]]
Steps = list[tuple[int, int]]


def own(pulses: Sequence[Pulse], together: Callable[[list[Pulse]], int] | None = None) -> Steps:
    """The pulses of *pulses* (sorted by start, of one segment) playing at
    once as scheduled, over time; a pulse ending at a cycle is over for one
    starting then. *together* counts those of a set that play in one run,
    where it is not all of them (arms of a branch)."""
    steps: Steps = []
    active: list[tuple[int, int]] = []
    k = 0
    while k < len(pulses) or active:
        if active and (k == len(pulses) or active[0][0] <= pulses[k][0]):
            at = active[0][0]
        else:
            at = pulses[k][0]
        while active and active[0][0] <= at:
            heappop(active)
        while k < len(pulses) and pulses[k][0] == at:
            heappush(active, (pulses[k][1], k))
            k += 1
        count = len(active)
        if together is not None and count > 1:
            count = together([pulses[i] for _, i in sorted(active, key=_place)])
        if not steps or steps[-1][1] != count:
            steps.append((at, count))
    return steps


def _place(item: tuple[int, int]) -> int:
    return item[1]


def by_start(pulse: Pulse) -> tuple[int, int]:
    """The order pulses are counted in: by start, then end."""
    return pulse[0], pulse[1]


def slid(steps: Steps, most: int | None) -> Steps:
    """The most *steps* reaches at each cycle when it starts 0 to *most*
    cycles later (``None``: any number)."""
    if not steps or most == 0:
        return steps
    if most is None:
        out: Steps = []
        highest = 0
        for at, count in steps:
            if count > highest:
                highest = count
                out.append((at, count))
        return out
    # Each count holds from its cycle to the next one plus *most*; the last
    # one, when it is not 0 (after a slide by any number), for good.
    spans = [
        (at, steps[k + 1][0] + most if k + 1 < len(steps) else _NEVER, count)
        for k, (at, count) in enumerate(steps)
        if count
    ]
    edges = sorted(
        {int(edge) for start, end, _ in spans for edge in (start, end) if edge != _NEVER}
    )
    out = []
    held: list[tuple[int, float]] = []
    k = 0
    for cycle in edges:
        at = cycle
        while k < len(spans) and spans[k][0] <= at:
            heappush(held, (-spans[k][2], spans[k][1]))
            k += 1
        while held and held[0][1] <= at:
            heappop(held)
        count = -held[0][0] if held else 0
        if not out or out[-1][1] != count:
            out.append((at, count))
    return out


def added(parts: Sequence[Steps]) -> Steps:
    """The sum of the step functions *parts*."""
    parts = [part for part in parts if part]
    if len(parts) == 1:
        return parts[0]
    changes: dict[int, int] = {}
    for part in parts:
        before = 0
        for at, count in part:
            changes[at] = changes.get(at, 0) + count - before
            before = count
    out: Steps = []
    total = 0
    for at in sorted(changes):
        if changes[at]:
            total += changes[at]
            out.append((at, total))
    return out


def most_at_once(
    pulses: Mapping[int, Sequence[Pulse]],
    after: Sequence[int | None],
    ranges: Sequence[int | None],
    together: Callable[[list[Pulse]], int] | None = None,
    blocks: bool = True,
    counted: Mapping[int, Steps] | None = None,
) -> Steps:
    """The most of *pulses* (by segment, each sorted by start) that can
    play at once over time (see the module). Segment ``k`` follows segment
    ``after[k]`` (``None`` for segment 0) and starts 0 to ``ranges[k]``
    cycles later than scheduled against it (``None``: any number).

    With *blocks*, segments that follow one another alone, each one's
    pulses after the one before's (the trips of a loop with a wait each),
    are counted as one that slides by the sum of their ranges: every trip
    as far as the last can. That never counts fewer, and it keeps the work
    in proportion to the pulses however many such segments there are.

    *counted* gives, for segments of its keys, a count over time already
    made (the copies of a loop that were not walked, see ``rolled``), added
    to the count of the segment's pulses; such a segment is kept as it is."""
    counted = counted or {}
    # The segments that hold pulses, and those they follow.
    needed = set()
    for segment in (*pulses, *counted):
        at: int | None = segment
        while at is not None and at not in needed:
            needed.add(at)
            at = after[at]
    # A segment that cannot slide plays with the one it follows.
    home: dict[int, int] = {}
    following: dict[int, list[int]] = {}
    held: dict[int, Sequence[Pulse]] = {}
    joined: set[int] = set()
    for segment in sorted(needed):
        before = after[segment]
        if before is None:
            home[segment] = segment
        elif ranges[segment] == 0 and segment not in counted:
            home[segment] = home[before]
        else:
            home[segment] = segment
            following.setdefault(home[before], []).append(segment)
        if segment in pulses:
            at = home[segment]
            if at in held:
                held[at] = [*held[at], *pulses[segment]]
                joined.add(at)
            else:
                held[at] = pulses[segment]
    for at in joined:
        held[at] = sorted(held[at], key=by_start)
    needed = {segment for segment in needed if home[segment] == segment}
    slides: dict[int, int | None] = {segment: ranges[segment] for segment in needed}
    if blocks:
        _merge(needed, following, held, slides, counted)
    # A segment follows one of a lower number: the last first.
    found: dict[int, Steps] = {}
    for segment in sorted(needed, reverse=True):
        parts = [own(held[segment], together)] if held.get(segment) else []
        if segment in counted:
            parts.append(counted[segment])
        for later in following.get(segment, ()):
            parts.append(slid(found.pop(later), slides[later]))
        found[segment] = added(parts)
    return found.get(0, [])


def _merge(
    needed: set[int],
    following: dict[int, list[int]],
    held: dict[int, Sequence[Pulse]],
    slides: dict[int, int | None],
    kept: Mapping[int, Steps],
) -> None:
    """Make each run of segments that follow one another alone one segment
    (see ``most_at_once``), in place: the first of the run holds the pulses
    of all and slides by the sum of their ranges. A segment joins the run
    only when its pulses start at or after the end of those before it in
    the run, which sliding later keeps so; what follows the run slides
    against all of it as far as against its first segment. A segment of
    *kept* is in no run."""
    for head in sorted(needed):
        if head not in needed or len(following.get(head, ())) != 1:
            continue
        first = last = following[head][0]
        if first in kept:
            continue
        block = [*held.get(first, ())]
        end = max((pulse[1] for pulse in block), default=None)
        total = slides[first]
        # How far the last of the run can slide against its first.
        within: int | None = 0
        while len(following.get(last, ())) == 1:
            follower = following[last][0]
            if follower in kept:
                break
            theirs = held.get(follower, ())
            if theirs:
                if end is not None and theirs[0][0] < end:
                    break
                end = max(pulse[1] for pulse in theirs)
                block.extend(theirs)
            more = slides[follower]
            total = None if total is None or more is None else total + more
            within = None if within is None or more is None else within + more
            following.pop(last)
            needed.discard(follower)
            last = follower
        if last != first:
            held[first] = block
            slides[first] = total
            rest = following.pop(last, None)
            if rest is not None:
                following[first] = rest
                # What follows the run follows its last segment: against a
                # pulse earlier in the run it slides that much further.
                for later in rest:
                    more = slides[later]
                    slides[later] = None if within is None or more is None else more + within


class OrderError(Exception):
    """A loop's pulses do not come in the order of their segments: its
    trips are not counted as a block (``rolled``)."""


class Loop(NamedTuple):
    """A loop whose trips after the last one walked are not listed: the
    ``first`` segment those trips open, how many each opens (``step``),
    how many such ``trips`` there are, the loop's ``period`` in cycles, and
    ``played``, the pulses counted of the last trip walked, each of whose
    copies, a period later per trip, is ``step`` segments on."""

    first: int
    step: int
    trips: int
    period: int
    played: Sequence[Pulse]


def rolled(
    pulses: Mapping[int, Sequence[Pulse]],
    after: Mapping[int, int | None],
    ranges: Mapping[int, int | None],
    loops: Sequence[Loop],
    together: Callable[[list[Pulse]], int] | None = None,
) -> tuple[dict[int, Sequence[Pulse]], list[int | None], list[int | None], dict[int, Steps]]:
    """What ``most_at_once`` takes (pulses, after, ranges, counted), for the
    segments a walk listed (*pulses*, *after* and *ranges* by segment)
    and the trips of *loops* it did not.

    The trips of a loop not walked are counted as a block per timeline of
    the loop: it plays, from a period after the last trip walked to the
    end of the last trip, the most that trips in a row play at once on
    that timeline (a trip's last pulses are in one segment with the next
    trip's first; those of the first trip not walked that sit in a listed
    segment are counted there); it follows the last segment the last trip
    walked opened on
    the timeline and slides by the trips times what a trip can; and a
    segment one of those trips opened that is listed (what follows the
    loop) follows the block, sliding against it as far again. That never
    counts fewer than the trips one by one would, for a loop whose pulses
    come in the order of their segments on each timeline, a later
    segment's after an earlier one's; ``OrderError`` for a loop where they
    do not (line delays can move one before another), which is counted
    trip by trip."""
    # A block by (loop, the segments the last walked trip opened on one
    # timeline): the segment it follows, its range, its pulses of one trip.
    blocks: dict[tuple[int, int], tuple[int, int | None, list[Pulse]]] = {}
    of: list[dict[int, int]] = []
    for k, loop in enumerate(loops):
        anchors = range(loop.first - loop.step, loop.first)
        group: dict[int, int] = {}
        for anchor in anchors:
            before = after[anchor]
            group[anchor] = group[before] if before in group else anchor
        of.append(group)
        for anchor in anchors:
            head = group[anchor]
            last, most, played = blocks.get((k, head), (anchor, 0, []))
            more = ranges[anchor]
            most = None if most is None or more is None else most + more
            blocks[(k, head)] = (max(last, anchor), most, played)

    def block(k: int, segment: int) -> tuple[int, int]:
        """The block of loop *k* that the trips' segment *segment* is in."""
        loop = loops[k]
        if segment >= loop.first:
            segment = loop.first - loop.step + (segment - loop.first) % loop.step
        return k, of[k][segment]

    def skipped(segment: int) -> int | None:
        for k, loop in enumerate(loops):
            if loop.first <= segment < loop.first + loop.step * loop.trips:
                return k
        return None

    for k, loop in enumerate(loops):
        for pulse in loop.played:
            blocks[block(k, pulse[2]["segment"] + loop.step)][2].append(pulse)
    # Every node by where it comes in the order of segments: a block after
    # the segment it follows.
    order: dict[tuple[int, int, int], Any] = {(segment, 0, 0): segment for segment in after}
    # A segment one of those trips opened that is listed, for its pulses
    # (what follows the loop before the next wait) or as the one another
    # follows, plays with the last trip: it follows the block.
    beyond = {segment for segment in pulses if segment not in after}
    beyond.update(before for before in after.values() if before is not None and before not in after)
    for segment in beyond:
        order[(segment, 0, 0)] = segment
    for (k, head), (last, _, _) in blocks.items():
        order[(last, 1, head)] = (k, head)
    place = {node: at for at, key in enumerate(sorted(order)) for node in [order[key]]}
    out_pulses: dict[int, Sequence[Pulse]] = {place[k]: played for k, played in pulses.items()}
    # A pulse of the next trip that sits in a segment the last walked trip
    # opened (before the trip's first wait) plays there with that trip's
    # last pulses: it is listed, and the block holds the trips after it.
    for loop in loops:
        for pulse in loop.played:
            segment = pulse[2]["segment"] + loop.step
            if segment < loop.first:
                at = place[segment]
                copy = (pulse[0] + loop.period, pulse[1] + loop.period, pulse[2])
                out_pulses[at] = sorted([*out_pulses.get(at, ()), copy], key=by_start)
    out_after: list[int | None] = [None] * len(place)
    out_ranges: list[int | None] = [0] * len(place)
    counted: dict[int, Steps] = {}
    for segment, before in after.items():
        if before is not None:
            out_after[place[segment]] = place[before]
        out_ranges[place[segment]] = ranges[segment]
    for segment in beyond:
        inside = skipped(segment)
        assert inside is not None
        node = block(inside, segment)
        out_after[place[segment]] = place[node]
        out_ranges[place[segment]] = _times(blocks[node][1], loops[inside].trips)
    for (k, head), (last, most, played) in blocks.items():
        loop = loops[k]
        at = place[(k, head)]
        out_after[at] = place[last]
        out_ranges[at] = _times(most, loop.trips)
        if played:
            played = sorted(played, key=by_start)
            # As many trips in a row as one trip's pulses span, and one
            # more: each a period and a trip's segments on from the last.
            # Pulses of trips further apart play at no common time.
            reach = max(pulse[1] for pulse in played) - played[0][0]
            rows = -(-reach // loop.period) + 1
            if rows > _ROWS:
                raise OrderError("a trip's pulses span too many periods")
            both = [
                (
                    (pulse[0] + row * loop.period, pulse[1] + row * loop.period, pulse[2]),
                    pulse[2]["segment"] + row * loop.step,
                )
                for row in range(rows)
                for pulse in played
            ]
            ended, before, upto = 0, -1, 0
            for pulse, segment in sorted(both, key=_by_segment):
                if segment != before:
                    ended, before = max(ended, upto), segment
                if pulse[0] < ended:
                    raise OrderError("a pulse before one of an earlier segment")
                upto = max(upto, pulse[1])
            pair = sorted((pulse for pulse, _ in both), key=by_start)
            highest = max(count for _, count in own(pair, together))
            # The trips the block holds: from the second not walked for a
            # pulse listed above, from the first for any other.
            spans = [
                (
                    pulse[0] + loop.period * (2 if segment + loop.step < loop.first else 1),
                    pulse[1] + loop.period * loop.trips,
                )
                for pulse, segment in both[: len(played)]
            ]
            spans = [span for span in spans if span[0] < span[1]]
            if spans:
                counted[at] = [
                    (min(span[0] for span in spans), highest),
                    (max(span[1] for span in spans), 0),
                ]
    return out_pulses, out_after, out_ranges, counted


# The trips in a row whose pulses ``rolled`` compares, at most.
_ROWS = 64


def _by_segment(item: tuple[Pulse, int]) -> tuple[int, int]:
    return item[1], item[0][0]


def _times(most: int | None, trips: int) -> int | None:
    return None if most is None else most * trips


def first_over(steps: Steps, limit: int) -> tuple[int, int, int] | None:
    """Where *steps* first passes *limit*: the cycle, the count then, and
    the most it reaches; ``None`` when it never does."""
    first = None
    for at, count in steps:
        if count > limit and first is None:
            first = (at, count)
    if first is None:
        return None
    return first[0], first[1], max(count for _, count in steps)


def one_timeline(events: Sequence[Mapping[str, Any]]) -> bool:
    """Whether *events* (those one limit counts) are on lines that wait
    together, as far as their segments tell: every segment but the first
    is of one timeline, so lines with events in one such segment wait
    together, and what waits together plays as it is placed. A line with
    events only in the first segment is taken to wait with no other."""
    group: dict[int, int] = {}

    def of(line: int) -> int:
        while group.setdefault(line, line) != line:
            line = group[line]
        return line

    lines: dict[int, int] = {}
    for event in events:
        line = of(event["line"])
        segment = event["segment"]
        if segment:
            first = lines.setdefault(segment, line)
            if first != line:
                group[line] = of(first)
    return len({of(line) for line in group}) <= 1


class Over(NamedTuple):
    """Pulses that can play at once past a limit: the first cycle they can,
    how many then and at most, and the waits that let them (a clause for a
    message)."""

    first: int
    at_first: int
    most: int
    cause: str


def over(
    lowered: ScheduledKernel,
    played: Sequence[Pulse],
    limit: int,
    lines: Sequence[Mapping[str, Any]],
    together: Callable[[list[Pulse]], int] | None = None,
) -> Over | None:
    """Where *played* (pulses of *lowered* that one limit counts, sorted by
    start) can be more than *limit* at once in some run."""
    segments, after = lowered._segments()
    ranges: list[int | None] = [0]
    for segment, before in zip(segments[1:], after[1:], strict=True):
        most, least = segment.max_shift_cycles, segments[before or 0].max_shift_cycles
        ranges.append(None if most is None or least is None else most - least)
    pulses: dict[int, list[Pulse]] = {}
    for pulse in played:
        pulses.setdefault(pulse[2]["segment"], []).append(pulse)

    def wait(k: int) -> str:
        segment = segments[k]
        if segment.opened_by == "answer":
            on = f"the blocking wait on link {segment.line}"
        else:
            line = lines[segment.line] if segment.line >= 0 else None
            on = f"the wait on {line['role'][0]}{line['qudit']}" if line else "the wait"
        return f"{on} that ends at cycle {segment.start_cycles}"

    # The waits whose range moves a pulse counted: those on the way to it.
    seen: set[int] = set()
    for k in pulses:
        at: int | None = k
        while at and at not in seen:
            seen.add(at)
            at = after[at]
    slid = sorted(k for k in seen if ranges[k] != 0)
    found = first_over(most_at_once(pulses, after, ranges, together), limit)
    if found is None:
        return None
    named = []
    for k in slid[:NAMED]:
        later = "any number of" if ranges[k] is None else f"up to {ranges[k]}"
        named.append(f"{wait(k)} ({later} cycles later)")
    if len(slid) > NAMED:
        named.append(f"{len(slid) - NAMED} more such waits")
    cause = (
        f"what follows {', '.join(named)} starts later in a run than it is placed, each "
        "wait's length counted as independent of the others'"
    )
    return Over(*found, cause)


def over_rolled(
    played: Sequence[Pulse],
    limit: int,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    segments: Sequence[Any],
    together: Callable[[list[Pulse]], int] | None = None,
) -> tuple[bool, Over | None]:
    """``over`` from a walk that lists three trips of each long loop:
    *played* its pulses that one limit counts, *events* all it listed,
    *repeats* its loops ((first event, end, copies, period, depth, ...)) and
    *segments* what its segment walk kept: the waits it visited ((kind,
    line, start, lowest, highest, the segment before), each opening the
    next segment) and, per loop in order, (the waits visited before the
    trips it skipped, the first segment those open, the segments a trip
    opens, the trips). The first result is whether the loops can be counted
    so (``rolled``): each of one period, in no other, its pulses a trip
    apart in segment as the trips' waits are; if not, they are counted
    trip by trip."""
    boundaries, skips = segments
    if len(skips) != len(repeats):
        return False, None
    # The segment each visited wait opens: the next, after those skipped.
    table: dict[int, tuple[Any, ...]] = {}
    after: dict[int, int | None] = {0: None}
    ranges: dict[int, int | None] = {0: 0}
    k, jumped = 0, 0
    for visited, (kind, line, start, lowest, highest, before) in enumerate(boundaries):
        while k < len(skips) and skips[k][0] <= visited:
            jumped += skips[k][2] * skips[k][3]
            k += 1
        opened = visited + 1 + jumped
        table[opened] = (kind, line, start + lowest)
        after[opened] = before
        ranges[opened] = None if highest < 0 else highest - lowest
    pulses: dict[int, list[Pulse]] = {}
    for pulse in played:
        pulses.setdefault(pulse[2]["segment"], []).append(pulse)
    loops = []
    more = 0
    for (begin, end, copies, period, depth, *_), (_, first, step, trips) in zip(
        repeats, skips, strict=True
    ):
        inside = [pulse for pulse in played if begin <= pulse[2]["index"] < end]
        if step == 0 and not inside:
            # No wait in the loop and no pulse counted: nothing to slide.
            continue
        if period <= 0 or depth or trips != copies - 1 or step <= 0:
            return False, None
        for pulse in inside:
            # The same pulse a trip earlier is a trip's segments back.
            earlier = events[pulse[2]["index"] - (end - begin)]
            if earlier["kind"] != pulse[2]["kind"] or (
                pulse[2]["segment"] - earlier["segment"] != step
            ):
                return False, None
        loops.append(Loop(first, step, trips, period, inside))
        # Each trip not listed has the waits of the last one listed; the
        # last trip's move a pulse only when something follows the loop.
        last = first + step * trips
        follows = any(first <= segment < last for segment in pulses) or any(
            before is not None and first <= before < last for before in after.values()
        )
        more += (trips if follows else trips - 1) * sum(
            1 for anchor in range(first - step, first) if ranges[anchor] != 0
        )
    try:
        made = rolled(pulses, after, ranges, loops, together)
    except OrderError:
        return False, None
    found = first_over(most_at_once(made[0], made[1], made[2], together, counted=made[3]), limit)
    if found is None:
        return True, None
    seen: set[int] = set()
    for segment in (*pulses, *(loop.first - 1 for loop in loops)):
        at: int | None = segment
        while at and at not in seen and at in table:
            seen.add(at)
            at = after[at]
    slid = sorted(segment for segment in seen if ranges[segment] != 0)
    named = []
    for segment in slid[:NAMED]:
        kind, line, start = table[segment]
        if kind == "answer":
            on = f"the blocking wait on link {line}"
        else:
            on = (
                f"the wait on {lines[line]['role'][0]}{lines[line]['qudit']}"
                if line >= 0
                else "the wait"
            )
        later = "any number of" if ranges[segment] is None else f"up to {ranges[segment]}"
        named.append(f"{on} that ends at cycle {start} ({later} cycles later)")
    if len(slid) + more > NAMED:
        named.append(f"{len(slid) + more - NAMED} more such waits")
    cause = (
        f"what follows {', '.join(named)} starts later in a run than it is placed, each "
        "wait's length counted as independent of the others'"
    )
    return True, Over(*found, cause)
