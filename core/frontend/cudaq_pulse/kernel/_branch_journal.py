# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The channels of a trace from its first measurement-conditioned if on,
imported by that if; a trace without one runs none of this.

While an if is traced, each op that rebinds a channel notes it in the if's
journal (``_Lines.journal``) with its value entering the if, and the if
restores and compares only those channels; outside an if no journal is
open. Each channel has its place in ``_Lines.held`` (``_Held.seq``), which
orders an if's results as the channels were acquired.
"""

from __future__ import annotations

import itertools
import operator
import sys
from typing import TYPE_CHECKING, Any

from .._native._cudaq_pulse_native import identity_keys
from ._emitter import (
    PulseIREmitter,
    StackVal,
    _Chan,
    _first,
    _Held,
    _is_pure_host_call,
    _Lines,
    _slot_changes,
    _slot_holds,
)
from .ir_builder import LINEAR_TYPES, IRValue


class _LoggedSlots(dict):  # type: ignore[type-arg]
    """The slot records (id(line or tone) -> (list, index, value)) from the
    trace's first measurement-conditioned if on, which note each value whose
    record they write or drop (``written``) and each place a record names in
    a list ``places`` follows, so that the next if makes only those records
    canonical (``_canonical_list``). Before that if the records are a dict."""

    __slots__ = ("written", "places")

    if TYPE_CHECKING:
        written: dict[int, IRValue]
        places: dict[int, list[Any]]

    def __setitem__(self, key: int, entry: tuple[list[Any], int, IRValue]) -> None:
        self.written[key] = entry[2]
        known = self.places.get(id(entry[0]))
        if known is not None:
            # _place, inline: this runs at every write of a record.
            at = known[1]
            i = entry[1]
            if i.__class__ is not int:
                i = operator.index(i)
            found = at.get(key)
            if found is None:
                at[key] = i
            elif found.__class__ is int:
                if found != i:
                    at[key] = {found, i}
            else:
                found.add(i)
        dict.__setitem__(self, key, entry)

    def __delitem__(self, key: int) -> None:
        self.written[key] = dict.__getitem__(self, key)[2]
        dict.__delitem__(self, key)

    def pop(self, key: int, *default: Any) -> Any:
        entry = dict.pop(self, key, *default)
        if entry is not None and entry is not (default[0] if default else None):
            self.written[key] = entry[2]
        return entry

    def update(self, items: Any) -> None:  # type: ignore[override]
        for key, entry in items:
            self[key] = entry


def _places(lines: _Lines, obj: list[Any]) -> tuple[dict[int, Any], list[int]]:
    """The places of each line or tone in *obj*, by id, as ``lines.places``
    keeps them (found once by reading the list, then from the records
    written; one place as an int, more as a set, so that a list of lines
    adds no container per line for the collector to walk), and the ids of
    the lines and tones the tracer added at its end since it was last looked
    at without writing their records (`+=`)."""
    known = lines.places.get(id(obj))
    n = len(obj)
    if known is None or known[0] is not obj:
        at: dict[int, set[int]] = {}
        start = 0
        lines.places[id(obj)] = [obj, at, n]
    else:
        at = known[1]
        start = known[2]
        known[2] = n
        if start > n:
            start = 0
    added: list[int] = []
    for i in range(start, n):
        v = obj[i]
        if v.__class__ is IRValue and v.vtype in LINEAR_TYPES:
            _place(at, id(v), i)
            added.append(id(v))
    return at, added


def _place(at: dict[int, Any], key: int, i: Any) -> None:
    """Add place *i* of *key* to *at*: an int for one place, a set for more.
    An index the kernel gave as another integer type (``numpy.int64``, an
    ``IntEnum`` member) is kept as an int."""
    if i.__class__ is not int:
        i = operator.index(i)
    found = at.get(key)
    if found is None:
        at[key] = i
    elif found.__class__ is int:
        if found != i:
            at[key] = {found, i}
    else:
        found.add(i)


def _canonical_list(emitter: PulseIREmitter, obj: list[Any], full: bool) -> None:
    """Record the lines and tones of *obj*, a list an if's then arm put back
    from its slot log, at their places in it, as the if records every list
    it puts back, in the order it puts them back: each at its last place in
    the list. With *full*, every one; else only those whose record was
    written or dropped since the last if (``written``): the others are as
    the last if recorded them, and the lists are the same."""
    slots = emitter._slots
    if full:
        at: dict[int, Any] = {}
        for i, v in enumerate(obj):
            if v.__class__ is IRValue and v.vtype in LINEAR_TYPES:
                dict.__setitem__(slots, id(v), (obj, i, v))
                _place(at, id(v), i)
        emitter._lines.places[id(obj)] = [obj, at, len(obj)]
        return
    written = slots.written  # type: ignore[attr-defined]
    n = len(obj)
    known = emitter._lines.places.get(id(obj))
    if known is not None and known[0] is obj and known[2] == n:
        # Nothing added at its end since it was last looked at.
        if not written:
            return
        at = known[1]
    else:
        at, added = _places(emitter._lines, obj)
        for key in added:
            # Added without a record: recorded as the if records every item.
            if key not in written:
                place = at[key]
                written[key] = obj[place if place.__class__ is int else max(place)]
        if not written:
            return
    if at.keys().isdisjoint(written):
        return
    # Only the written lines and tones the list holds, in the order written.
    for key in [key for key in written if key in at]:
        found = at[key]
        value = written[key]
        if found.__class__ is int:
            # One place: as below, without the sets; none left, no entry.
            i = found
            if not -n <= i < n:
                del at[key]
                continue
            place = i % n if i < 0 else i
            if obj[place] is value:
                if place != i:
                    # The place it holds, counted from the front.
                    at[key] = place
                dict.__setitem__(slots, key, (obj, place, value))
            else:
                del at[key]
            continue
        if not found:
            continue
        valid = {i % n if i < 0 else i for i in found if -n <= i < n}
        valid = {i for i in valid if obj[i] is value}
        if valid:
            # The places it holds, counted from the front.
            at[key] = valid if len(valid) > 1 else next(iter(valid))
            dict.__setitem__(slots, key, (obj, max(valid), value))
        else:
            del at[key]


def _log_by_list(
    log: dict[tuple[int, int], tuple[list[Any], int, IRValue]],
) -> dict[int, list[tuple[int, IRValue]]]:
    """A slot log's entries by list id: (index, the value it held), in index order."""
    by_list: dict[int, list[tuple[int, IRValue]]] = {}
    for (key, i), (_, _, old) in log.items():
        if key in by_list:
            by_list[key].append((i, old))
        else:
            by_list[key] = [(i, old)]
    for entries in by_list.values():
        entries.sort(key=_first)
    return by_list


def _logged_slot_changes(
    containers: list[tuple[Any, Any]],
    made: dict[int, list[Any]],
    log: dict[tuple[int, int], tuple[list[Any], int, IRValue]],
    copies: list[tuple[Any, Any]],
    arm: Logged | None = None,
) -> list[tuple[Any, int, IRValue]] | None:
    """``_slot_changes`` of an arm's *containers* and then its *copies*, in
    their order: a container without a copy, put back from the arm's slot
    *log*, by the slots the log holds (or by its copy in *made*, if code
    changed it), in index order, as a comparison with a copy would find."""
    changes: list[tuple[Any, int, IRValue]] = []
    by_list: dict[int, list[tuple[int, IRValue]]] | None = None
    for obj, copy in containers:
        if copy is None:
            copy = made.get(id(obj))
            if copy is None:
                if by_list is None:
                    by_list = _log_by_list(log)
                    if arm is not None:
                        # Its arm's slots are put back from it too.
                        arm.by_list = by_list
                for i, old in by_list.get(id(obj), ()):
                    new = obj[i]
                    if new is old:
                        continue
                    if not (
                        isinstance(new, IRValue)
                        and isinstance(old, IRValue)
                        and new.vtype == old.vtype
                        and new.vtype in LINEAR_TYPES
                    ):
                        return None
                    changes.append((obj, i, new))
                continue
        found = _slot_changes([(obj, copy)])
        if found is None:
            return None
        changes += found
    if copies:
        found = _slot_changes(copies)
        if found is None:
            return None
        changes += found
    return changes


def _materialize(
    arms: tuple[tuple[Any, ...], ...],
    obj: list[Any],
) -> None:
    """Before code changes *obj*, a list each arm in *arms* (innermost first)
    puts back from its slot log: note in each such arm's record of changed
    containers, unless there already, the list as it was when that arm began
    (its current contents with the slot logs of that arm and of the arms in
    it undone)."""
    key = id(obj)
    # Only arms that put lists back from a slot log (not those of an if
    # holding none, nor the record between ifs, _Between).
    arms = tuple(arm for arm in arms if len(arm) == 5 and arm[2] is not None)
    if not any(key in backed and key not in mutated for mutated, _, _, backed, _ in arms):
        return
    contents: list[Any] | None = None
    for mutated, _, log, backed, lengths in arms:
        if contents is None:
            contents = list(obj)
        for (owner, i), (_, _, old) in log.items():
            if owner == key and i < len(contents):
                contents[i] = old
        if key in lengths:
            # What the tracer added without a note (at the end) since then.
            del contents[lengths[key] :]
        if key in backed and key not in mutated:
            mutated[key] = (obj, list(contents))


def _put_back_order(
    containers: list[tuple[Any, Any]], repeats: list[tuple[int, list[Any]]]
) -> list[tuple[Any, Any]]:
    """The order in which an if puts back its *containers* when names repeat
    a list of lines among them: each such list where its last name comes
    (*repeats*: each further name of one, in order, with the number of
    containers before it), the others where they are, with any container
    an arm added after the names (``_conditions.add_built``) at the end.
    Putting a list back records its lines and tones as its own, so one that
    two lists hold is recorded as the later list's: the order is the one a
    record for each name would give, which the if kept before it kept one
    for each list."""
    # Each repeated list's last name, in the order those names come.
    last: dict[int, tuple[int, list[Any]]] = {}
    for entry in repeats:
        key = id(entry[1])
        if key in last:
            del last[key]
        last[key] = entry
    moved = iter(last.values())
    at, obj = next(moved)
    back: list[tuple[Any, Any]] = []
    for index, entry in enumerate(containers):
        while at <= index:
            back.append((obj, None))
            at, obj = next(moved, (len(containers) + 1, obj))
        if entry[1] is not None or id(entry[0]) not in last:
            back.append(entry)
    while at <= len(containers):
        back.append((obj, None))
        at, obj = next(moved, (len(containers) + 1, obj))
    return back


class _Between(dict):  # type: ignore[type-arg]
    """What code changed between a trace's measured ifs, outside them: its
    record of changed containers (the emitter's ``_mutated`` there), as an
    arm keeps one, from the trace's first measured if on. A container is
    noted when code first changes it, and never copied; the next if
    records afresh the slots of each list of lines it holds that was
    changed (``JournalLines.watch``). ``plain``: the lists ``_Lines.plain``
    held when a call the trace does not follow, between ifs, emptied it,
    as there it does not (None: it is as it should be)."""

    __slots__ = ("plain",)

    if TYPE_CHECKING:
        plain: dict[int, tuple[list[Any], int]] | None

    def __contains__(self, key: object) -> bool:
        # Asked only before a container changes: noted, so never copied.
        dict.__setitem__(self, key, None)
        return True


def _reach(
    emitter: PulseIREmitter, old: dict[int, Any], slots: _LoggedSlots, top: _Between
) -> None:
    """Give *slots*, the trace's slot records from its first measured if on,
    to *emitter* and to each emitter of an inlined call it is in that holds
    the records it held (*old*), and *top* as the record of changed
    containers to each of those outside any arm or rolled body. Each of
    those emitters is running a method on the stack, up to the one that
    began the trace."""
    emitter._slots = slots
    if emitter._mutated is None:
        emitter._mutated = top
    if not emitter._depth:
        return
    frame: Any = sys._getframe(1)
    while frame is not None:
        other = frame.f_locals.get("self") if "self" in frame.f_code.co_varnames else None
        if isinstance(other, PulseIREmitter):
            if other._slots is old:
                other._slots = slots
            if other._mutated is None:
                other._mutated = top
            if not other._depth:
                return
        frame = frame.f_back


class JournalLines(_Lines):
    """``_Lines`` that note what an op rebinds in an open journal, and give
    each channel acquired its place."""

    __slots__ = ()

    # Where a cell is among the channels: its channel's place, then its own.
    CELL_ORDER = operator.attrgetter("held.seq", "index")

    @staticmethod
    def materialize(arms: tuple[tuple[Any, ...], ...], obj: list[Any]) -> None:
        """``_materialize``, for the emitter's ``_note_mutation``."""
        _materialize(arms, obj)

    @staticmethod
    def logged(
        emitter: PulseIREmitter,
        containers: list[tuple[Any, Any]],
        lazy: list[Any],
        backed: dict[int, list[Any]],
        repeats: list[tuple[int, list[Any]]] | None,
    ) -> Logged:
        """At a measured if that holds lists of lines (*containers* without
        a copy; *backed*, by id), put back from its arms' slot logs: the
        trace's channels and slot records from then on note what changes
        them (``SlotJournalLines``), and the containers *lazy*, copied when
        first changed, become the if's ``Logged`` record. A list that more
        than one name holds is in *containers* once, at its first name;
        *repeats* has each further name of one (``_put_back_order``)."""
        lines: Any = emitter._lines
        fresh = True
        if lines.__class__ is SlotJournalLines:
            fresh = lines.watch(emitter, containers)
        else:
            if lines.__class__ is _Lines:
                # As the if would at its first: its channels numbered.
                for seq, h in enumerate(lines.held.values()):
                    h.seq = seq
                lines.journal = None
            if lines.__class__ is not JournalLines:
                lines.__class__ = JournalLines
            lines.slotlog = getattr(lines, "slotlog", None)
            lines.__class__ = SlotJournalLines
            lines.start(emitter)
        logged = Logged(lazy)
        logged.fresh = fresh
        logged.backed = backed
        logged.repeats = repeats
        return logged

    def add(self, key: tuple[Any, ...], values: tuple[IRValue, ...]) -> None:
        # Only a channel not held is added, and ``held`` only ever loses its
        # newest entries: its length is the channel's place.
        held = _Held(values, key)
        held.seq = len(self.held)
        self.held[key] = held

    @staticmethod
    def now(
        a: IRValue, b: IRValue, then_held: dict[_Chan, IRValue], journal: dict[_Chan, IRValue]
    ) -> tuple[IRValue, IRValue]:
        """What an if's then arm left on a line or tone *a*, and its else arm
        on *b*, once the else arm is traced: the value each one's channel had
        at that arm's end (for *a*, what the then arm left on a channel it
        changed, in *then_held*, else what the channel held entering the if),
        or the value itself if it is no channel's."""
        chan = a.chan
        if chan is not None:
            if chan in then_held:
                a = then_held[chan]
            elif chan in journal:
                a = journal[chan]
            else:
                a = chan.value
        chan = b.chan
        return a, b if chan is None else chan.value

    @classmethod
    def join_name(
        cls,
        name: str,
        a: IRValue,
        b: IRValue,
        then_held: dict[_Chan, IRValue],
        journal: dict[_Chan, IRValue],
        named: dict[_Chan, str],
        twins: list[tuple[str, str]],
        merged: dict[str, Any],
        results: list[tuple[str, IRValue, IRValue]],
    ) -> None:
        """Join the values *a* and *b* of one line or tone that an if's arms
        left on *name*, one an old value or the pair another name carries:
        an old value stands for its channel's value at that arm's end
        (``now``), as an op on it would take; *name* then holds one value
        (*merged*), takes the result of the name that carries the same pair
        (*twins*), or carries it (*named*, *results*)."""
        chan = a.chan
        a, b = cls.now(a, b, then_held, journal)
        if a is b:
            merged[name] = a
        elif chan in named:
            twins.append((name, named[chan]))
        else:
            named[chan] = name
            results.append((name, a, b))

    @classmethod
    def list_now(
        cls,
        listed: list[Any],
        obj: Any,
        key: tuple[int, int],
        a: IRValue,
        b: IRValue,
        then_held: dict[_Chan, IRValue],
        journal: dict[_Chan, IRValue],
    ) -> None:
        """Note in *listed* the list place *obj[key[1]]* an if's arms left on
        values *a* and *b* of one line or tone, one an old value, as its
        channel's values at the arms' ends (``now``), unless they are one."""
        a, b = cls.now(a, b, then_held, journal)
        if a is not b:
            listed.append((obj, key[1], a, b, None))

    def rebind_linear(
        self,
        emitter: PulseIREmitter,
        fname: str,
        call_args: list[StackVal],
        results: tuple[IRValue, ...],
    ) -> None:
        """``_Lines.rebind_linear``, noting in an open journal each channel
        it moves."""
        slots = emitter._slots
        names: dict[int, str | None] | None = None
        if fname == "sync" and len(call_args) == 2 and call_args[0] is not call_args[1]:
            # Two distinct lines: their names are found in one pass.
            first, second = call_args
            if first.chan is not None and second.chan is not None and len(results) == 2:
                # Both are claimable, so each claims its own result.
                self.rebind_pair(emitter, first, results[0], second, results[1])
                return
            found = emitter._local_names(first, second)
            names = {id(first): found[0], id(second): found[1]}
        # The operands each result may replace, by type, the first in operand
        # order last: a sync over n lines rebinds in O(n), not O(n^2).
        unclaimed: dict[str, list[tuple[str | None, IRValue, Any, Any]]] = {}
        for a in reversed(call_args):
            if a.__class__ is IRValue and a.vtype in LINEAR_TYPES:
                slot = slots.get(id(a))
                if slot is not None and not _slot_holds(slot, a):
                    slot = None
                name = emitter._find_local_name(a) if names is None else names[id(a)]
                if name is not None or slot is not None or a.chan is not None:
                    if a.vtype in unclaimed:
                        unclaimed[a.vtype].append((name, a, slot, a.chan))
                    else:
                        unclaimed[a.vtype] = [(name, a, slot, a.chan)]

        for res in results:
            if res.vtype not in unclaimed or not unclaimed[res.vtype]:
                continue
            name, op_val, slot, chan = unclaimed[res.vtype].pop()
            if chan is not None and chan.value is op_val:
                journal = self.journal
                if journal is not None and chan not in journal:
                    journal[chan] = op_val
                chan.value = res
                res.chan = chan
                if chan.shared:
                    for other, value in emitter.locals.items():
                        if value is op_val:
                            emitter.locals[other] = res
            if name is not None:
                emitter.locals[name] = res
            if slot is not None:
                slot[0][slot[1]] = res
                slots.pop(id(op_val), None)
                slots[id(res)] = (slot[0], slot[1], res)

    def rebind_pair(
        self, emitter: PulseIREmitter, old: IRValue, new: IRValue, old2: IRValue, new2: IRValue
    ) -> None:
        """``_Lines.rebind_pair``, noting in an open journal the channels it
        moves."""
        chan = old.chan
        chan2 = old2.chan
        journal = self.journal
        if journal is not None:
            # Noted for _rebind_one too, which only this calls.
            if chan is not None and chan not in journal:
                journal[chan] = chan.value
            if chan2 is not None and chan2 not in journal:
                journal[chan2] = chan2.value
        if (chan is not None and chan.shared) or (chan2 is not None and chan2.shared):
            emitter._rebind_one(old, new)
            emitter._rebind_one(old2, new2)
            return
        name, name2 = identity_keys(emitter.locals, old, old2)
        if chan is not None:
            chan.value = new
            new.chan = chan
        if name is not None:
            emitter.locals[name] = new
        emitter._rebind_slot(old, new)
        if chan2 is not None:
            chan2.value = new2
            new2.chan = chan2
        if name2 is not None:
            emitter.locals[name2] = new2
        emitter._rebind_slot(old2, new2)

    def restore_container(self, emitter: PulseIREmitter, obj: Any, copy: Any) -> None:
        """*emitter*'s ``_restore_container``, as an if restores a container
        an arm changed: a list of more than 8 lines and tones has its slots
        tracked in one step."""
        if isinstance(obj, _Held):
            for i, value in enumerate(copy):
                obj[i] = value
        elif isinstance(obj, list):
            obj[:] = copy
            if (
                len(copy) > 8
                and obj.__class__ is list
                and set(map(type, copy)) == {IRValue}
                and LINEAR_TYPES.issuperset(map(operator.attrgetter("vtype"), copy))
            ):
                # Lines or tones only, as a measurement-conditioned if restores
                # a list of them: every slot at once. A subclass is read by
                # index (its own len() and indexing run).
                entries = zip(itertools.repeat(obj), itertools.count(), copy, strict=False)
                emitter._slots.update(zip(map(id, copy), entries, strict=False))
            else:
                emitter._track_slots(obj, 0)
        elif isinstance(obj, (dict, set)):
            obj.clear()
            obj.update(copy)
        else:
            obj.clear()
            obj.extend(copy)


class SlotJournalLines(JournalLines):
    """``JournalLines`` from the trace's first measured if that holds a list
    of lines on: each op that writes a line or tone into a list slot notes,
    in the open arm's slot log (``slotlog``), the value the slot held, and
    what code changes between ifs is noted (``_Between``)."""

    __slots__ = ()

    @staticmethod
    def canonical_list(emitter: PulseIREmitter, obj: list[Any], full: bool) -> None:
        """``_canonical_list``, for the emitter's ``_runtime_branch``."""
        _canonical_list(emitter, obj, full)

    @staticmethod
    def put_back_order(
        containers: list[tuple[Any, Any]], repeats: list[tuple[int, list[Any]]]
    ) -> list[tuple[Any, Any]]:
        """``_put_back_order``, for the emitter's ``_runtime_branch``."""
        return _put_back_order(containers, repeats)

    @staticmethod
    def log_by_list(
        log: dict[tuple[int, int], tuple[list[Any], int, IRValue]],
    ) -> dict[int, list[tuple[int, IRValue]]]:
        """``_log_by_list``, for the emitter's ``_runtime_branch``."""
        return _log_by_list(log)

    def start(self, emitter: PulseIREmitter) -> None:
        """Begin what the trace keeps from its first measured if with a list
        of lines to put back from its arms' slot logs on: slot records that
        note what they write, and a record of what code changes between
        ifs (``_Between``). The if may be in an arm of another."""
        self.canon = None
        self.places = {}
        self.dirty = True
        top = _Between()
        top.plain = None
        self.between = top
        self.watched = (*getattr(self, "watched", ()), (top, (), None, True, None))
        # Code that runs where the record is not the emitter's (around the
        # rolled body, arm or inlined call this if is in) is not noted until
        # an if at the trace's top level makes it so.
        self.unwatched = emitter._depth > 0 or emitter._mutated is not None
        old = emitter._slots
        slots = _LoggedSlots(old)
        slots.written = {}
        slots.places = self.places
        _reach(emitter, old, slots, top)

    def watch(self, emitter: PulseIREmitter, containers: list[tuple[Any, Any]]) -> bool:
        """At a measured if, before its arms: whether code changed one of
        the lists of lines it puts back from its slot logs (*containers*
        without a copy) since the last if, outside an arm, as its record of
        changed containers shows; if so, or if code ran where no record was
        kept, the if records their slots afresh (``dirty``, returned: an if
        in its arm may clear it first)."""
        top = self.between
        current = emitter._mutated
        if current is None:
            # Back from an arm, rolled body or inlined call the if that began
            # the record was in: what ran since was not noted.
            _reach(emitter, emitter._slots, emitter._slots, top)  # type: ignore[arg-type]
            watched = getattr(self, "watched", ())
            if not watched or watched[-1][0] is not top:
                # The record, last among the arms an untraced call looks at.
                self.watched = (*watched, (top, (), None, True, None))
            self.dirty = True
            if not emitter._depth:
                # From here on, each arm and rolled body begins inside it.
                self.unwatched = False
            return True
        if self.unwatched:
            self.dirty = True
            if current is top:
                top.clear()
                # Each emitter from here to the trace's top took the record
                # from the one that called it, so all that runs is noted in
                # it from now on (an inlined call's ifs, too): this if records
                # every slot, the next ones only what changed.
                self.unwatched = False
            return True
        if top or current is not top:
            for obj, copy in containers:
                if copy is None and (
                    dict.__contains__(top, id(obj)) or dict.__contains__(current, id(obj))
                ):
                    self.dirty = True
                    break
        if current is top:
            top.clear()
        return self.dirty

    def untraced(
        self,
        emitter: PulseIREmitter,
        mutated: dict[int, Any],
        backed: Any,
        fn: Any,
        args: list[StackVal],
        kwargs: dict[str, StackVal] | None,
    ) -> None:
        """A call the trace does not follow, in an arm whose lists of lines
        and tones (*backed*) are put back from its slot log, or between ifs
        (*mutated* the ``_Between``): one that may change a list (not pure,
        or given it) has each such list copied as it was when the arm began
        (``_materialize``), or between ifs noted as changed."""
        if mutated.__class__ is _Between:
            between = emitter._mutated is mutated
            if between:
                # Between ifs the call leaves ``plain`` as it was.
                if mutated.plain is None:
                    plain = getattr(self, "plain", None)
                    mutated.plain = dict(plain) if plain else {}
            else:
                mutated.plain = None
            if between or self.slotlog is None:
                # Outside an arm: a list it is given, or any list if it is
                # not pure, is noted as changed.
                for value in (*args, *(kwargs or {}).values()):
                    if isinstance(value, list):
                        dict.__setitem__(mutated, id(value), None)
                if not _is_pure_host_call(fn):
                    self.dirty = True
            return
        given = {
            id(value) for value in (*args, *(kwargs or {}).values()) if isinstance(value, list)
        }
        pure = _is_pure_host_call(fn)
        arms = self.watched
        for key, obj in backed.items():
            if key not in mutated and (not pure or key in given):
                _materialize(arms, obj)

    def take_plain(self) -> None:
        """Before a measured if reads ``plain``: what it held when calls
        between ifs emptied it, as there they do not."""
        top = self.between
        if top.plain is not None:
            self.plain = top.plain
            top.plain = None

    def rebind_linear(
        self,
        emitter: PulseIREmitter,
        fname: str,
        call_args: list[StackVal],
        results: tuple[IRValue, ...],
    ) -> None:
        """``JournalLines.rebind_linear``, noting first in the open slot log
        each list slot it may write: those holding a line or tone it is
        given."""
        log = self.slotlog
        if log is not None:
            _log_held(emitter._slots, log, call_args)
        JournalLines.rebind_linear(self, emitter, fname, call_args, results)

    def rebind_pair(
        self, emitter: PulseIREmitter, old: IRValue, new: IRValue, old2: IRValue, new2: IRValue
    ) -> None:
        """``JournalLines.rebind_pair``, noting first in the open slot log
        the list slots it may write."""
        log = self.slotlog
        if log is not None:
            _log_held(emitter._slots, log, (old, old2))
        JournalLines.rebind_pair(self, emitter, old, new, old2, new2)


def _log_held(slots: dict[int, Any], log: dict[tuple[int, int], Any], values: Any) -> None:
    """Note in an arm's slot *log* each plain list slot recorded as holding
    one of *values* (lines and tones an op rebinds), with that value, unless
    the slot is noted already. A subclass's slots are not put back from the
    log: its own indexing is not run for it."""
    for value in values:
        if value.__class__ is IRValue and value.vtype in LINEAR_TYPES:
            slot = slots.get(id(value))
            if slot is not None and slot[0].__class__ is list and _slot_holds(slot, value):
                obj = slot[0]
                at = slot[1] + len(obj) if slot[1] < 0 else slot[1]
                log.setdefault((id(obj), at), (obj, at, value))


class Logged(list):  # type: ignore[type-arg]
    """The containers a measured if copies when an arm first changes them
    (each with its length), at an if that holds lists of lines put back
    from its arms' slot logs (``backed``, by id): true even when it holds
    none of the others, as it keeps each arm's ``log``, and ``made``, the
    lists of lines code changed in it, as they were when it began.
    ``fresh``: whether the if records every slot of the lists it puts back
    (``SlotJournalLines.watch``). ``repeats``: each further name of a list
    in ``backed``, or None if each has one name (``_put_back_order``)."""

    __slots__ = ("fresh", "backed", "repeats", "lengths", "log", "made", "around_log", "by_list")

    if TYPE_CHECKING:
        fresh: bool
        backed: dict[int, list[Any]]
        repeats: list[tuple[int, list[Any]]] | None
        lengths: dict[int, int]
        log: dict[tuple[int, int], tuple[list[Any], int, IRValue]]
        made: dict[int, list[Any]]
        around_log: dict[tuple[int, int], tuple[list[Any], int, IRValue]] | None
        by_list: dict[int, list[tuple[int, IRValue]]] | None

    def __bool__(self) -> bool:
        return True

    def begin(self, lines: Any, mutated: dict[int, Any], around: tuple[Any, ...]) -> None:
        """Before an arm: its slot log opened, and the lists it puts back
        from it watched, with their lengths (the tracer adds to a list, as
        LIST_APPEND and `+=` may, without noting it, only at its end)."""
        log: dict[tuple[int, int], tuple[list[Any], int, IRValue]] = {}
        self.log = log
        self.by_list = None
        self.around_log = lines.slotlog
        lines.slotlog = log
        backed = self.backed
        self.lengths = {key: len(obj) for key, obj in backed.items()}
        lines.watched = (
            (mutated, tuple(obj for obj, _ in self), log, backed, self.lengths),
            *around,
        )

    def end(
        self,
        mutated: dict[int, Any],
        containers: list[tuple[Any, Any]],
        copies: list[tuple[Any, Any]],
    ) -> list[tuple[Any, int, IRValue]] | None:
        """After an arm: ``made``, and the list slots it rebound
        (``_logged_slot_changes``)."""
        log = self.log
        lengths = self.lengths
        made: dict[int, list[Any]] = {}
        for key, obj in self.backed.items():
            noted = mutated.get(key)
            if noted is not None and noted[1] is not None:
                made[key] = noted[1]
            elif len(obj) != lengths[key]:
                # Added to without a note: as it was, its first places with
                # the arm's writes to them undone.
                n = lengths[key]
                copy = obj[:n]
                for (owner, i), (_, _, old) in log.items():
                    if owner == key and i < n:
                        copy[i] = old
                made[key] = copy
        self.made = made
        return _logged_slot_changes(containers, made, log, copies, self)
