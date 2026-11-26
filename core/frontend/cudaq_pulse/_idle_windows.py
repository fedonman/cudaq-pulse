# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The idle windows of a scheduled program, for idle replacement.

A trace keeps a record of the drives, readouts, waits, syncs and frame
changes it emits (``Sites``), and the module built from it carries each
one's number, so the times the schedule gave them can be read back per
traced operation (``read``). From those, ``gaps`` finds where each qudit's
drive line has nothing to do and which parts of that the qudit itself
idles through, and ``report`` lists the windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._errors import CompilationError, InternalError, UnsupportedError

if TYPE_CHECKING:
    from .kernel._emitter import PulseIREmitter

__all__ = ["IdleWindow", "IdleWindows"]

# The kind of each numbered op, as the packed buffer's site record gives it.
_SITE_KINDS = {"d": 0, "r": 1, "w": 2, "s": 3, "f": 4}

_LETTERS = {
    "pulse.drive": "d",
    "pulse.readout": "r",
    "pulse.wait": "w",
    "pulse.sync": "s",
    "pulse.shift_phase": "f",
    "pulse.set_phase": "f",
    "pulse.shift_frequency": "f",
    "pulse.set_frequency": "f",
}


# The pulse calls that emit a numbered op.
_SITE_CALLS = frozenset(kind.removeprefix("pulse.") for kind in _LETTERS)


class Site:
    """One numbered op of a trace: its kind (the op's name), where its
    record starts in the packed buffer, the regions open when it was
    emitted, the kernel instructions being traced then (innermost first),
    its length in cycles when the trace knows it, and the regions in which
    the tracer was between two kernel statements right before it
    (``enter``) and right after it (``leave``): the points where something
    can be written next to it, in its own region and, for the first or
    last op of a called kernel, in the caller's. ``number`` is its place
    among the kernel's own ops, or, for an op written into an idle window,
    a negative number."""

    __slots__ = ("kind", "mark", "scope", "frames", "cycles", "enter", "leave", "number")

    def __init__(
        self,
        kind: str,
        mark: int,
        scope: tuple[int, ...],
        frames: Any,
        cycles: int | None,
        enter: set[tuple[int, ...]],
        number: int,
    ) -> None:
        self.kind = kind
        self.mark = mark
        self.scope = scope
        self.frames = frames
        self.cycles = cycles
        self.enter = enter
        self.leave: set[tuple[int, ...]] = set()
        self.number = number

    def describe(self) -> str:
        """``drive() at line 12 of cycle()``."""
        name = f"{self.kind.removeprefix('pulse.')}()"
        if not self.frames:
            return name
        prog, index = self.frames[0]
        line = prog.location(index)[1]
        where = f"{prog.code.co_name}()"
        return f"{name} at line {line} of {where}" if line else f"{name} in {where}"


class Sites:
    """The record one trace keeps of its numbered ops, in program order.

    With a *plan*, the trace also writes it: at the points the plan names,
    the waits and fills of its tiles (``rule`` gives the fill)."""

    def __init__(self, plan: Plan | None = None, rule: Any = None) -> None:
        self.log: list[Site] = []
        # The kernel's own ops recorded so far (all of them without a plan).
        self.hosts = 0
        # Whether a pulse call is being emitted (calls nest: a calibrated
        # gate emits its ops through the same entry), or a tile written.
        self.busy = False
        # By number of the next op of the kernel: the regions in which a
        # statement began since the op before it.
        self.entering: dict[int, set[tuple[int, ...]]] = {}
        self.plan = plan
        self.rule = rule
        # The tile being written (its number), and the fill being traced.
        self.inserting: int | None = None
        self.filling: _Filling | None = None
        # The tiles of the plan already written, by number, with the
        # buffer's cursor then: a loop body traced again writes its tiles
        # again.
        self.done: dict[int, int] = {}
        # The trace's waveform lengths, by value id (the tracer's own dict).
        self.durations: dict[int, Any] = {}
        # The fill's length by (qudit, window), once it has been traced.
        self.lengths: dict[tuple[int, int], int] = {}
        # The (qudit, window) a fill traced into writes nothing for.
        self.nothing: set[tuple[int, int]] = set()

    def attach(self, builder: Any, tracer: type[PulseIREmitter] | None) -> type[PulseIREmitter]:
        """Record what *builder* is given from here on; returns the tracer
        to trace with, which is *tracer* (or the one *builder*'s sample
        rates take) and notes the points between statements."""
        from .kernel._emitter import tracing_frames
        from .kernel.packed_ir_builder import _SITE

        # A kernel is traced again when a float parameter turns out to set
        # a length: the record starts over with each trace.
        log = self.log
        log.clear()
        self.hosts = 0
        self.entering.clear()
        self.done.clear()
        self.lengths.clear()
        self.nothing.clear()
        self.busy = False
        self.inserting = self.filling = None
        emit = builder.emit
        rollback = builder.rollback
        submit = builder.classical_submit
        wait = builder.classical_wait

        def submitted(*args: Any, **kwargs: Any) -> Any:
            if self.filling is not None:
                raise self.filling.refuse("calls a classical function")
            return submit(*args, **kwargs)

        def waited(*args: Any, **kwargs: Any) -> Any:
            if self.filling is not None:
                raise self.filling.refuse("takes the answer of a classical function")
            return wait(*args, **kwargs)

        def recorded(
            kind: str,
            operands: tuple[Any, ...] = (),
            result_types: tuple[str, ...] = (),
            attrs: dict[str, Any] | None = None,
        ) -> tuple[Any, ...]:
            filling = self.filling
            if filling is not None:
                filling.take(kind, operands, attrs or {}, builder._cur, self.durations)
            if kind not in _LETTERS:
                return emit(kind, operands, result_types, attrs)  # type: ignore[no-any-return]
            mark = builder._cur
            scope = tuple(builder._scope)
            results = emit(kind, operands, result_types, attrs)
            cycles = None
            if kind == "pulse.wait":
                cycles = (attrs or {}).get("duration")
            elif kind in ("pulse.drive", "pulse.readout"):
                cycles = self.durations.get(operands[1].vid)
            if cycles.__class__ is not int:
                cycles = None
            if self.inserting is None:
                number = self.hosts
                self.hosts += 1
                enter = self.entering.pop(number, set())
            else:
                number = -1 - self.inserting
                enter = set()
            log.append(Site(kind, mark, scope, tracing_frames(), cycles, enter, number))
            # The op's number travels with it in the packed buffer, so the
            # module is built with it, before any pass runs.
            builder._record(_SITE, [number, _SITE_KINDS[_LETTERS[kind]]])
            return results  # type: ignore[no-any-return]

        def rolled_back(checkpoint: tuple[Any, ...]) -> None:
            # A loop body traced again: what it emitted is gone.
            rollback(checkpoint)
            cursor = checkpoint[0]
            while log and log[-1].mark >= cursor:
                if log.pop().number >= 0:
                    self.hosts -= 1
            for number in [number for number in self.entering if number > self.hosts]:
                del self.entering[number]
            # A tile written at the cursor was written before what is undone.
            for tile in [tile for tile, mark in self.done.items() if mark > cursor]:
                del self.done[tile]
            if self.filling is not None:
                self.filling.undo(cursor)

        builder.emit = recorded
        builder.rollback = rolled_back
        builder.classical_submit = submitted
        builder.classical_wait = waited
        builder._sites = self
        base: type[Any]
        if tracer is not None:
            base = tracer
        elif builder.tracer is None:
            from .kernel._emitter import PulseIREmitter

            base = PulseIREmitter
        else:
            # Lines of different sample rates, or a broadcast drive line.
            base = builder.tracer
        sited = _SITED.get(base)
        if sited is None:
            sited = _SITED[base] = type(
                f"Sited{base.__name__}", (_SiteTracer, base), {"__slots__": ()}
            )
        return sited

    def finish(self) -> None:
        """After the filled program's trace: each tile planned was written.
        (The module was built with each op's number from the packed
        buffer's site records, which also check each op's kind.)"""
        plan = self.plan
        if plan is not None and len(self.done) != plan.tiles:
            raise InternalError(
                "idle replacement: the program traced differently the second time, so "
                "not every idle window was written"
            )


class _Filling:
    """The fill being traced into a window: what it may do (drive and wait
    on its own qudit's drive line, at that line's tone, and shift the
    tone's phase by constants that add up to zero) and how long it is."""

    __slots__ = ("name", "line", "tone", "ops", "drives")

    def __init__(self, name: str, line: Any, tone: Any) -> None:
        self.name = name
        self.line = line
        self.tone = tone
        # (cursor of the buffer, what, value): "length" in cycles, "phase"
        # in radians, "for" with a trip count, "end".
        self.ops: list[tuple[int, str, Any]] = []
        # The cursors of the pulses it plays.
        self.drives: list[int] = []

    def refuse(self, what: str) -> CompilationError:
        return CompilationError(
            f"idle replacement: fill {self.name}() {what}; in this version a fill drives and "
            "waits on its own qudit's drive line, at that line's tone, and nothing else",
            code="unsupported_construct",
        )

    def take(
        self,
        kind: str,
        operands: tuple[Any, ...],
        attrs: dict[str, Any],
        mark: int,
        durations: dict[int, Any],
    ) -> None:
        """Check the op the fill emits, and note what it adds."""
        if kind == "pulse.drive":
            if operands[0].chan is not self.line:
                raise self.refuse("drives another line than its qudit's drive line")
            if operands[2].chan is not self.tone:
                raise self.refuse("plays a tone that is not its drive line's own")
            cycles = durations.get(operands[1].vid)
            if cycles.__class__ is not int:
                raise self.refuse("plays a waveform whose length is not known when it compiles")
            self.ops.append((mark, "length", cycles))
            self.drives.append(mark)
        elif kind == "pulse.wait":
            if operands[0].chan is not self.line:
                raise self.refuse("waits on another line than its qudit's drive line")
            cycles = attrs.get("duration", 0)
            if cycles.__class__ is not int or attrs.get("bounds") is not None:
                raise self.refuse("waits a length that is not a whole number of cycles")
            self.ops.append((mark, "length", cycles))
        elif kind == "pulse.shift_phase":
            phase = attrs.get("phase_rad", 0.0)
            if operands[0].chan is not self.tone or not isinstance(phase, (int, float)):
                raise self.refuse("shifts a phase that is not a number, or of another tone")
            self.ops.append((mark, "phase", float(phase)))
        elif kind == "scf.for":
            trips = max(0, -(-(attrs.get("ub", 0) - attrs.get("lb", 0)) // attrs.get("step", 1)))
            self.ops.append((mark, "for", trips))
        elif kind == "scf.for_end":
            self.ops.append((mark, "end", None))
        elif kind == "scf.yield":
            pass
        elif kind in _LETTERS or kind.startswith("scf.") or kind in _ACQUIRING:
            raise self.refuse(f"uses {kind.removeprefix('pulse.').removeprefix('scf.')}")

    def undo(self, cursor: int) -> None:
        ops = self.ops
        while ops and ops[-1][0] >= cursor:
            ops.pop()
        drives = self.drives
        while drives and drives[-1] >= cursor:
            drives.pop()

    def pulses(self) -> int:
        """How many pulses it plays, a loop's body counted once per
        iteration (a loop of no iterations plays none)."""
        drives = set(self.drives)
        stack = [1]
        count = 0
        for mark, what, value in self.ops:
            if what == "for":
                stack.append(stack[-1] * value)
            elif what == "end":
                stack.pop()
            elif what == "length" and mark in drives:
                count += stack[-1]
        return count

    def totals(self) -> tuple[int, float]:
        """Its length in cycles and the phase it leaves its tone shifted
        by, a loop's body counted once per iteration."""
        stack: list[list[Any]] = [[1, 0, 0.0]]
        for _, what, value in self.ops:
            if what == "for":
                stack.append([value, 0, 0.0])
            elif what == "end":
                trips, length, phase = stack.pop()
                stack[-1][1] += trips * length
                stack[-1][2] += trips * phase
            elif what == "length":
                stack[-1][1] += value
            else:
                stack[-1][2] += value
        return int(stack[0][1]), float(stack[0][2])


# What a fill may not make: a line or a tone it does not have yet.
_ACQUIRING = ("pulse.get_drive_line", "pulse.get_readout_line", "pulse.tone")


class _SiteTracer:
    """Notes, for the record of its builder, the points between kernel
    statements, and writes the record's plan at them; emits as the tracer
    it is mixed into."""

    __slots__ = ()

    def _emit_pulse_call(self, fname: str, call_args: list[Any]) -> tuple[Any, ...]:
        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        if sites.busy:
            return super()._emit_pulse_call(fname, call_args)  # type: ignore[misc,no-any-return]
        sites.durations = self._durations  # type: ignore[attr-defined]
        before = sites.hosts
        if fname in _SITE_CALLS:
            self._statement_begins(before)
        sites.busy = True
        try:
            results: tuple[Any, ...] = super()._emit_pulse_call(fname, call_args)  # type: ignore[misc]
        finally:
            sites.busy = False
        if sites.hosts > before:
            self._statement_ends()
        return results

    def _dispatch_call(self, callable_val: Any, call_args: list[Any], kwargs: Any) -> None:
        # A call of a kernel is a statement too: what its first op starts
        # after or its last op ends before can be written around the call,
        # in the caller's region (the kernel may be scheduled on its own).
        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        if sites.busy or getattr(callable_val, "__pulse_kernel_fn__", None) is None:
            super()._dispatch_call(callable_val, call_args, kwargs)  # type: ignore[misc]
            return
        before = sites.hosts
        self._statement_begins(before)
        super()._dispatch_call(callable_val, call_args, kwargs)  # type: ignore[misc]
        if sites.hosts > before:
            self._statement_ends()

    def _statement_begins(self, number: int) -> None:
        """A statement begins here, before the kernel's op *number*: note
        the point, or write what the plan has for it."""
        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        scope = tuple(builder._scope)
        if sites.plan is None:
            sites.entering.setdefault(number, set()).add(scope)
        else:
            self._write(sites.plan.before.get(number), scope)

    def _statement_ends(self) -> None:
        """A statement that emitted ops of the kernel ends here."""
        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        scope = tuple(builder._scope)
        if sites.plan is None:
            # Without a plan the record holds the kernel's ops only.
            sites.log[sites.hosts - 1].leave.add(scope)
        else:
            self._write(sites.plan.after.get(sites.hosts - 1), scope)

    def trace(self, prog: Any) -> Any:
        """Trace the kernel, then write what the plan has for its end."""
        result = super().trace(prog)  # type: ignore[misc]
        builder: Any = self.b  # type: ignore[attr-defined]
        plan = builder._sites.plan
        if plan is not None:
            self._write(plan.end, tuple(builder._scope))
        return result

    def _write(self, tiles: list[Tile] | None, scope: tuple[int, ...]) -> None:
        """Write here, in the region *scope*, those of *tiles* that belong
        to it and are not written yet."""
        if not tiles:
            return
        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        tiles = [tile for tile in tiles if tile.scope == scope and tile.number not in sites.done]
        if not tiles:
            return
        sites.busy = True
        # The regions a fill opens (a loop of its own) are closed again when
        # it is written: the kernel's later regions keep the numbers they
        # had in the trace the plan was made from.
        regions = builder._next_scope
        try:
            for tile in tiles:
                sites.done[tile.number] = builder._cur
                held = self._drive_channel(tile.qudit)
                sites.inserting = tile.number
                for item in tile.items:
                    if item[0] == "wait":
                        self._emit_pulse_call("wait", [held.cells[0].value, item[1]])
                    else:
                        self._fill(held, item[1])
        finally:
            sites.busy = False
            sites.inserting = None
            sites.filling = None
        builder._next_scope = regions
        self._refresh_lines()  # type: ignore[attr-defined]

    def _drive_channel(self, qudit: int) -> Any:
        """The drive channel the trace holds of *qudit*."""
        qudits = self.b._qref_to_qudit  # type: ignore[attr-defined]
        for key, held in self._lines.held.items():  # type: ignore[attr-defined]
            if len(key) == 2 and key[1] == "get_drive_line" and qudits.get(key[0]) == qudit:
                return held
        raise InternalError(f"idle replacement: the trace holds no drive line of qudit {qudit}")

    def _fill(self, held: Any, window: int) -> None:
        """Trace the rule's fill into a window of *window* cycles on the
        channel *held*, as a call ``fill(qudit, window)`` written here, and
        complete it with waits to the window's length."""
        from .kernel._emitter import _channel_values, _restore_channels

        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        rule = sites.rule
        line = held.cells[0]
        if sites.plan is not None and sites.plan.pads:
            self._emit_pulse_call("wait", [line.value, window])
            return
        qudit = builder._qref_to_qudit.get(held.key[0])
        length = None
        if rule.align != "start":
            # The wait before the fill is as long as the fill leaves over:
            # the fill is traced once to learn its length, and undone.
            length = sites.lengths.get((qudit, window))
            if length is None:
                channels = self._lines.held  # type: ignore[attr-defined]
                count, values = len(channels), _channel_values(channels)
                checkpoint = builder.checkpoint()
                length = self._traced_fill(held, window, qudit)
                builder.rollback(checkpoint)
                _restore_channels(channels, count, values)
                self._refresh_lines()  # type: ignore[attr-defined]
                sites.lengths[qudit, window] = length
            spare = window - length
            lead = spare if rule.align == "end" else spare // 2
            if lead:
                self._emit_pulse_call("wait", [line.value, lead])
        else:
            lead = 0
        traced = self._traced_fill(held, window, qudit)
        if length is not None and traced != length:
            raise CompilationError(
                f"idle replacement: fill {rule.fill.__name__}() has another length each time "
                "it is traced; a fill must trace the same way every time",
                code="unsupported_construct",
            )
        if window - traced - lead:
            self._emit_pulse_call("wait", [line.value, window - traced - lead])

    def _traced_fill(self, held: Any, window: int, qudit: int) -> int:
        """Trace the fill here; its length in cycles."""
        from .kernel.ir_builder import IRValue

        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        fill = sites.rule.fill
        name = fill.__name__
        line, tone = held.cells
        calls, readouts = len(builder.classical), builder._n_readouts
        filling = sites.filling = _Filling(name, line, tone)
        try:
            if getattr(fill, "schedule", None) is not None:
                raise filling.refuse("has a schedule= of its own")
            self._dispatch_call(fill, [IRValue(held.key[0], "qref"), window], None)
            self.stack.pop()  # type: ignore[attr-defined]
        finally:
            sites.filling = None
        if len(builder.classical) != calls or builder._n_readouts != readouts:
            raise filling.refuse("calls a classical function or reads out")
        length, phase = filling.totals()
        if not filling.pulses():
            # It plays no pulse here: the window is planned again as left
            # alone (_idle._filled).
            sites.nothing.add((qudit, window))
        if abs(phase) > 1e-9:
            raise filling.refuse(
                f"leaves its tone's phase shifted by {phase:g} rad (its shifts must add up to zero)"
            )
        if length > window:
            raise CompilationError(
                f"idle replacement: fill {name}() takes {length} cycles, {length - window} more "
                f"than its window of {window} cycles on qudit {qudit}",
                fix="shorten the fill, or raise threshold= so that this window is left alone",
                code="invalid_duration",
            )
        return length


_SITED: dict[type[Any], type[Any]] = {}


# ---------------------------------------------------------------- reading the schedule


class _Event:
    """One execution of a numbered op: its site, kind, line and the line
    whose tone it plays, its start and end in cycles, its start's shift per
    iteration of each loop around it, and where it lies: ``path`` holds
    (loop, copy, trips) per loop around it, outermost first, ``arms`` the
    (branch, arm) pairs around it. A sync is an instant where its lines
    meet; ``lines`` holds them."""

    __slots__ = ("site", "kind", "line", "tone", "start", "end", "shifts", "path", "arms", "lines")

    def __init__(
        self,
        site: int | None,
        kind: str,
        line: int,
        tone: int,
        start: int,
        end: int,
        shifts: tuple[int, ...],
        path: tuple[tuple[int, int, int], ...],
        arms: tuple[tuple[int, int], ...],
        lines: tuple[int, ...] = (),
    ) -> None:
        self.site = site
        self.kind = kind
        self.line = line
        self.tone = tone
        self.start = start
        self.end = end
        self.shifts = shifts
        self.path = path
        self.arms = arms
        self.lines = lines

    def advance(self, depth: int, loop: int, iterations: int) -> int:
        """How much later this event runs *iterations* iterations of *loop*
        (at *depth*) on, 0 if it is not in that loop's second copy."""
        path = self.path
        if len(path) > depth and path[depth][0] == loop and path[depth][1] == 1:
            return iterations * self.shifts[depth]
        return 0


def read(module: Any) -> tuple[list[tuple[str, int]], list[Any]]:
    """The lines of scheduled *module*, (role, qudit) by index, and its
    numbered ops in program order, each loop's body listed for its first
    iteration and, when it has more, its second: items are an ``_Event``,
    ``("barrier",)`` where a wait of run-time length is, and ``("advance",
    loop, depth, iterations)`` after the second copy of a loop of more than
    two iterations, which stands for the iterations that follow it."""
    lines, tree = module.site_tree()
    out: list[Any] = []
    counter = [0]

    def walk(
        nodes: list[Any],
        copies: tuple[int, ...],
        path: tuple[tuple[int, int, int], ...],
        arms: tuple[tuple[int, int], ...],
    ) -> None:
        for node in nodes:
            kind = node[0]
            if kind == "for":
                trips, body = node[1], node[2]
                counter[0] += 1
                loop = counter[0]
                for copy in range(min(trips, 2)):
                    walk(body, (*copies, copy), (*path, (loop, copy, trips)), arms)
                if trips > 2:
                    out.append(("advance", loop, len(path), trips - 2))
            elif kind == "if":
                counter[0] += 1
                branch = counter[0]
                walk(node[1], copies, path, (*arms, (branch, 0)))
                walk(node[2], copies, path, (*arms, (branch, 1)))
            elif kind == "sync":
                out.append(_Event(node[1], "sync", -1, -1, 0, 0, (), path, arms, tuple(node[2])))
            elif kind == "answer":
                out.append(("barrier",))
            else:
                _, site, line, tone, start, duration, shifts, variable = node
                if start is None or duration is None or None in shifts:
                    raise UnsupportedError(
                        "idle replacement: the program is not scheduled without its "
                        "arguments' values"
                    )
                if variable:
                    out.append(("barrier",))
                at = start + sum(k * shift for k, shift in zip(copies, shifts, strict=True))
                out.append(
                    _Event(site, kind, line, tone, at, at + duration, tuple(shifts), path, arms)
                )

    walk(tree, (), (), ())
    return [(str(role), int(qudit)) for role, qudit in lines], out


def numbered(items: list[Any]) -> list[Any]:
    """*items* (``read()``'s) of a program that was not traced, such as one
    ``pulse.load`` read, with each op numbered as a trace numbers it: the
    same number for every execution of one op of the kernel (each copy of a
    loop's body), in program order. ``gaps()`` tells a gap's executions and
    a loop's syncs apart by these numbers. A program that has any number
    of its own is returned as it is."""
    events = [item for item in items if item.__class__ is not tuple]
    if any(event.site is not None for event in events):
        return items
    # read() numbers a loop (and a branch) anew in each copy of the loops
    # around it; the loop of the kernel is its place in its parent's body:
    # the parent's, and the order it comes in among the loops of one copy
    # of the parent.
    kernel_loop: dict[int, tuple[Any, ...]] = {}
    loops_in: dict[tuple[Any, ...], int] = {}
    # An op is its loop of the kernel and how many ops came before it in the
    # same copy of that loop's body, both arms of its branches counted in
    # turn: every copy is read in the same order.
    seen: dict[tuple[Any, ...], int] = {}
    number: dict[tuple[Any, ...], int] = {}
    for event in events:
        parent: tuple[Any, ...] = ()
        copies: tuple[int, ...] = ()
        for loop, copy, _ in event.path:
            if loop not in kernel_loop:
                nth = loops_in.get((parent, copies), 0)
                loops_in[parent, copies] = nth + 1
                kernel_loop[loop] = (parent, nth)
            parent = kernel_loop[loop]
            copies = (*copies, copy)
        nth = seen.get(event.path, 0)
        seen[event.path] = nth + 1
        event.site = number.setdefault((parent, nth), len(number))
    return items


# ---------------------------------------------------------------- gaps and their idle parts

_PLAYS = ("drive", "readout")
# What the program orders with whatever is written on the qudit's line in
# a gap: what uses the qudit's tone, and a sync of its line.
_CHAINED = ("tone", "frame", "frequency", "sync")
# What plays on the qudit in a gap.
_ON_QUDIT = ("tone", "busy", "readout")
_FREQUENCY = ("shift_frequency", "set_frequency")


class _Timed:
    """A time that moves with the loop iterations of the event it comes
    from."""

    __slots__ = ("at", "event")

    def __init__(self, at: int, event: _Event | None) -> None:
        self.at = at
        self.event = event

    def advance(self, depth: int, loop: int, iterations: int) -> None:
        if self.event is not None:
            self.at += self.event.advance(depth, loop, iterations)


class Gap:
    """A stretch in which a qudit's drive line has no operation: from the
    end of ``lo`` (a drive or wait on the line; None at the program's
    start) to the start of ``hi`` (None at its end), with the events on the
    qudit in between, each (event, start, end, what): ``"tone"`` while its
    tone plays from another line, ``"readout"`` while its own readout runs,
    ``"frame"`` where the kernel changes the tone's phase, ``"frequency"``
    where it changes its frequency, ``"sync"`` where a sync of the line
    meets, and ``"busy"`` for a pulse at its tone that began before the gap.

    ``first`` and ``last`` say whether the qudit has no operation before
    the gap and none after it; ``barrier`` whether a wait of run-time
    length comes before its end; ``detuned`` whether the kernel changed the
    tone's frequency before it. ``rounds`` counts the executions the gap
    stands for."""

    __slots__ = (
        "qudit",
        "lo",
        "hi",
        "start",
        "end",
        "inner",
        "first",
        "last",
        "barrier",
        "detuned",
        "rounds",
        "own",
        "varies",
        "executions",
        "holes",
        "repeats",
        "uneven",
        "read_before",
        "read_after",
        "repeats_op",
    )

    def __init__(
        self,
        qudit: int,
        lo: _Event | None,
        hi: _Event | None,
        start: int,
        end: int,
        inner: list[tuple[_Event, int, int, str]],
        first: bool,
        barrier: bool,
        detuned: bool,
    ) -> None:
        self.qudit = qudit
        self.lo = lo
        self.hi = hi
        self.start = start
        self.end = end
        self.inner = inner
        self.first = first
        self.last = False
        self.barrier = barrier
        self.detuned = detuned
        # The executions this one stands for, and, on the gap that stands
        # for every execution of one gap of the kernel, all of them.
        self.own = self.rounds = _rounds(lo, hi)
        # Whether the executions it stands for differ in length or in what
        # happens on the qudit meanwhile, and all of them.
        self.varies = False
        self.executions: list[Gap] = [self]
        # Stretches of it in which the qudit is read out or its tone plays
        # in every iteration of a loop the gap runs through, which are not
        # listed one by one; and whether something on the qudit repeats
        # with another period than the gap does.
        self.holes: list[tuple[int, int]] = []
        # What makes the holes: "readout", "tone", "busy", "frame",
        # "frequency" or "sync".
        self.repeats: set[str] = set()
        self.uneven = False
        # With ``first`` (``last``): whether the qudit is read out or its
        # tone plays in a gap before (after) it that ``first`` (``last``)
        # holds for too, which syncs cut from it.
        self.read_before = False
        self.read_after = False
        # Whether it lies in the body of a loop where the qudit has an op of
        # its own: that op is before it and after it in other rounds, so it
        # is neither leading nor trailing.
        self.repeats_op = False

    def key(self) -> tuple[Any, ...]:
        """What the executions of one gap of the kernel share."""
        return (
            self.qudit,
            None if self.lo is None else self.lo.site,
            None if self.hi is None else self.hi.site,
        )

    def shape(self) -> tuple[Any, ...]:
        """Its length and what happens when, from its start."""
        return (
            self.end - self.start,
            tuple(
                (event.site, a - self.start, b - self.start, what)
                for event, a, b, what in self.inner
            ),
            # A window between the qudit's ops of a loop is not leading in
            # any round.
            self.first and not self.repeats_op,
            self.last,
            self.barrier,
            self.detuned,
        )

    def wraps(self) -> bool:
        """Whether it runs across the end of a loop iteration: the kernel
        writes the op it ends at no later than the op it starts after."""
        lo, hi = self.lo, self.hi
        if lo is None or hi is None or lo.site is None or hi.site is None:
            return False
        return hi.site <= lo.site

    def branched(self) -> bool:
        """Whether one of its events lies in an arm of a branch."""
        events = [self.lo, self.hi, *(event for event, _, _, _ in self.inner)]
        return any(event is not None and event.arms for event in events)

    def ordered(self) -> bool:
        """Whether the pulses at the qudit's tone and the changes of its
        frame in the gap come in the order the program writes them: what
        is written into the gap is written between them."""
        at = self.start
        # In a gap within one iteration, what cuts it is written after the
        # op it starts after.
        after = -1 if self.lo is None or self.lo.site is None or self.wraps() else self.lo.site
        for event, start, _, what in self.inner:
            if what in _CHAINED:
                if start < at or (after >= 0 and (event.site is None or event.site < after)):
                    return False
                at = start
        return at <= self.end

    def idle(self) -> list[tuple[int, int]]:
        """The parts of it the qudit idles through, as (start, end) in
        order: not while its tone plays or its readout runs, and cut where
        its frame changes."""
        parts: list[tuple[int, int]] = []
        at = self.start
        busy = [(start, end) for _, start, end, _ in self.inner] + self.holes
        for start, end in sorted(busy):
            start = min(max(start, self.start), self.end)
            end = min(max(end, self.start), self.end)
            if start > at:
                parts.append((at, start))
            at = max(at, end)
        if self.end > at:
            parts.append((at, self.end))
        return parts


def _rounds(lo: _Event | None, hi: _Event | None) -> int:
    """How many executions a gap from *lo* to *hi* stands for: a loop's
    second copy stands for every iteration after the first, and a gap from
    the first copy into the second for every pass from one iteration to the
    next."""
    a = () if lo is None else lo.path
    b = () if hi is None else hi.path
    rounds = 1
    for depth in range(min(len(a), len(b))):
        if a[depth] == b[depth]:
            _, copy, trips = a[depth]
            if copy:
                rounds *= trips - 1
            continue
        if a[depth][0] == b[depth][0]:
            rounds *= a[depth][2] - 1
        break
    return rounds


class _Line:
    """What a walk knows of one qudit's drive line: where its open gap
    starts and after which event, and what happened on the qudit since."""

    __slots__ = ("since", "inner", "started", "detuned", "gaps", "holes", "repeats")

    def __init__(self) -> None:
        self.since = _Timed(0, None)
        self.inner: list[tuple[_Event, _Timed, _Timed, str]] = []
        # Of the open gap: where a loop it runs through repeats what
        # happens on the qudit, as (start, end) that move with their event.
        self.holes: list[tuple[_Timed, _Timed]] = []
        self.repeats: set[str] = set()
        self.started = False
        self.detuned = False
        self.gaps: list[Gap] = []


def gaps(lines: list[tuple[str, int]], items: list[Any]) -> dict[int, list[Gap]]:
    """Per qudit with a drive line, the gaps of that line in program order;
    the last one ends where the last line does. Executions of one gap of
    the kernel (the iterations of a loop, and the copies the schedule made
    of one) are one ``Gap``, the shortest, which lists them all; ``varies``
    says they differ."""
    state: dict[int, _Line] = {}
    drive_of: dict[int, int] = {}
    for index, (role, qudit) in enumerate(lines):
        if role == "drive" and qudit >= 0 and qudit not in drive_of:
            drive_of[qudit] = index
            state[qudit] = _Line()
    # Where each line's latest op ends: a sync meets where the last of its
    # lines is free.
    free: dict[int, _Timed] = {}
    barrier = False
    # Each qudit's own readouts, at the first iteration of the loops around
    # them. The program orders a readout with nothing on the drive line, so
    # one written after a gap, or of another iteration, may run during it:
    # they are laid over the gaps at the end.
    readouts: dict[int, list[_Event]] = {qudit: [] for qudit in state}
    anchored, crowded = _sync_shifts(lines, items)

    def close(qudit: int, hi: _Event | None, end: int, restart: int) -> None:
        """End the open gap of *qudit* at *hi*, at *end*; the next one
        starts at *restart*, and what still runs then runs on in it."""
        line = state[qudit]
        inner = [(event, a.at, b.at, what) for event, a, b, what in line.inner]
        line.gaps.append(
            Gap(
                qudit,
                line.since.event,
                hi,
                line.since.at,
                end,
                inner,
                not line.started,
                barrier,
                line.detuned,
            )
        )
        line.gaps[-1].holes = [(a.at, b.at) for a, b in line.holes]
        line.gaps[-1].repeats = line.repeats
        line.holes = []
        line.repeats = set()
        # What still runs then is busy time of the next gap; a pulse at the
        # tone that has not begun is one of its events.
        line.inner = [
            (event, a, b, what if a.at >= restart else "busy")
            for event, a, b, what in line.inner
            if b.at > restart
        ]

    for item in items:
        if item.__class__ is tuple:
            if item[0] == "barrier":
                barrier = True
            else:
                _, loop, depth, iterations = item
                for timed in free.values():
                    timed.advance(depth, loop, iterations)
                for line in state.values():
                    for a, b in line.holes:
                        a.advance(depth, loop, iterations)
                        b.advance(depth, loop, iterations)
                    opened = line.since.event
                    if opened is None or not opened.advance(depth, loop, 1):
                        # The gap runs through the loop: what happens on
                        # the qudit in its second iteration happens in
                        # every one up to the last, which is listed again.
                        again = [
                            (a.at, event, what)
                            for event, a, _, what in line.inner
                            if event.advance(depth, loop, 1)
                        ]
                        if again:
                            start, first, _ = min(again, key=lambda found: found[0])
                            line.repeats.update(what for _, _, what in again)
                            hole = _Timed(start, first), _Timed(start, first)
                            hole[1].advance(depth, loop, iterations)
                            line.holes.append(hole)
                    line.since.advance(depth, loop, iterations)
                    for _, a, b, _ in line.inner:
                        a.advance(depth, loop, iterations)
                        b.advance(depth, loop, iterations)
            continue
        event: _Event = item
        if event.kind == "sync":
            meet = max((free[index].at for index in event.lines if index in free), default=0)
            latest = max(
                (free[index] for index in event.lines if index in free),
                key=lambda timed: timed.at,
                default=None,
            )
            event.start = event.end = meet
            if latest is not None and latest.event is not None:
                shifts = latest.event.shifts
                event.shifts = (*shifts, *(0,) * len(event.path))[: len(event.path)]
            else:
                event.shifts = (0,) * len(event.path)
            for index in event.lines:
                # Each of its lines goes on from where they met. A sync is
                # no time of its own: the gap goes on through it, cut there.
                free[index] = _Timed(meet, event)
                role, qudit = lines[index] if 0 <= index < len(lines) else ("", -1)
                if role == "drive" and drive_of.get(qudit) == index:
                    line = state[qudit]
                    if (
                        not event.arms
                        and event.path
                        and _where(event) in anchored
                        and not any((qudit, loop) in crowded for loop, _, _ in event.path)
                    ):
                        # A sync of the line in a loop ends the gap, in
                        # every copy of the loop alike: the stretches
                        # between the line's syncs and ops there are gaps
                        # of every round.
                        event.shifts = anchored[_where(event)]
                        close(qudit, event, meet, meet)
                        line.since = _Timed(meet, event)
                    else:
                        line.inner.append((event, _Timed(meet, event), _Timed(meet, event), "sync"))
            continue
        role, qudit = lines[event.line] if event.line >= 0 else ("", -1)
        if event.line >= 0 and event.kind in ("drive", "readout", "wait"):
            known = free.get(event.line)
            if known is None or event.end >= known.at:
                free[event.line] = _Timed(event.end, event)
            if role == "drive" and drive_of.get(qudit) == event.line:
                close(qudit, event, event.start, event.end)
                line = state[qudit]
                line.since = _Timed(event.end, event)
                line.started = True
            elif role == "readout" and event.kind == "readout" and qudit in state:
                if not any(copy for _, copy, _ in event.path):
                    readouts[qudit].append(event)
        if event.tone < 0 or event.tone == event.line:
            continue
        tone_role, owner = lines[event.tone]
        if tone_role != "drive" or owner not in state:
            continue
        if event.kind in _PLAYS:
            what = "tone"
            state[owner].started = True
        else:
            what = "frequency" if event.kind in _FREQUENCY else "frame"
        state[owner].inner.append(
            (event, _Timed(event.start, event), _Timed(event.end, event), what)
        )
        if what == "frequency":
            state[owner].detuned = True

    found: dict[int, list[Gap]] = {}
    length = max((timed.at for timed in free.values()), default=0)
    # By qudit, the outermost loops in which its line has an op of its own.
    owned: dict[int, set[int]] = {}
    for item in items:
        if item.__class__ is not tuple and item.path and item.kind in ("drive", "readout", "wait"):
            role, owner = lines[item.line] if 0 <= item.line < len(lines) else ("", -1)
            if role == "drive":
                owned.setdefault(owner, set()).add(item.path[0][0])
    for qudit, line in state.items():
        end = max(length, line.since.at)
        close(qudit, None, end, end)
        for gap in line.gaps:
            _lay(gap, readouts[qudit])
            # A pulse at the tone that repeats with another period than the
            # line's ops is elsewhere in every round.
            beat = gap.lo or gap.hi
            if beat is not None and any(
                what != "sync" and _out_of_step(beat, event) for event, _, _, what in gap.inner
            ):
                gap.uneven = True
        # No operation of the line follows the last gap, nor the gaps that
        # only syncs cut from it.
        tail = len(line.gaps) - 1
        while tail > 0 and _at_sync(line.gaps[tail - 1].hi):
            tail -= 1
        mine = owned.get(qudit, ())
        for gap in line.gaps:
            gap.repeats_op = _in_body(gap) and gap.lo.path[0][0] in mine  # type: ignore[union-attr]
        seen = False
        for gap in line.gaps:
            if not gap.first:
                break
            gap.read_before = seen
            seen = seen or any(what in _ON_QUDIT for _, _, _, what in gap.inner)
        seen = False
        for gap in line.gaps[tail:][::-1]:
            # In the body of a loop where the qudit has an op of its own,
            # that op follows the window in the next round: not trailing.
            if not gap.repeats_op:
                gap.last = True
            gap.read_after = seen
            seen = seen or any(what in _ON_QUDIT for _, _, _, what in gap.inner)
        found[qudit] = _merged(line.gaps)
    return found


def _at_sync(event: _Event | None) -> bool:
    return event is not None and event.kind == "sync"


def _outer(event: _Event) -> int | None:
    """The outermost loop around *event*, None outside any."""
    return event.path[0][0] if event.path else None


def _where(event: _Event) -> tuple[Any, ...]:
    """A sync's op, and the loops around it."""
    return (event.site, *(loop for loop, _, _ in event.path))


def _sync_shifts(
    lines: list[tuple[str, int]], items: list[Any]
) -> tuple[dict[tuple[Any, ...], tuple[int, ...]], set[tuple[int, int]]]:
    """The syncs in loops, by site, whose shift per iteration of each loop
    around them is known, with those shifts: wherever the sync runs in a
    loop's second copy, it meets after an op of that loop on its lines, or
    after such a sync (the walk of ``gaps()`` without its gaps), and every
    execution gives the same. And the (qudit, outermost loop) pairs where
    the qudit is read out or its tone plays outside that loop while it
    runs: a gap of every round there would be laid over its first round
    only, so its syncs do not end the qudit's gaps."""
    span: dict[int, list[int]] = {}
    on_qudit: list[tuple[int, int | None, int, int]] = []

    def note(event: _Event) -> None:
        if event.path:
            outer = event.path[0][0]
            low, high = span.setdefault(outer, [event.start, event.end])
            span[outer] = [min(low, event.start), max(high, event.end)]
        if event.kind == "readout" and 0 <= event.line < len(lines):
            role, qudit = lines[event.line]
            if role == "readout":
                on_qudit.append((qudit, _outer(event), event.start, event.end))
        # A pulse at its tone from another line, and a change of the
        # tone's frame or frequency there.
        if 0 <= event.tone < len(lines) and event.tone != event.line:
            role, qudit = lines[event.tone]
            if role == "drive":
                on_qudit.append((qudit, _outer(event), event.start, event.end))

    free: dict[int, _Timed] = {}
    known: dict[int, tuple[int, ...] | None] = {}
    found: dict[tuple[Any, ...], list[tuple[int, ...] | None]] = {}
    # The places each sync's op is in, in loops or not: one in several (a
    # loop the schedule copied or peeled) is written in all of them, so
    # none of them ends a gap.
    places: dict[int, set[tuple[Any, ...]]] = {}
    for item in items:
        if item.__class__ is tuple:
            if item[0] != "barrier":
                _, loop, depth, iterations = item
                for timed in free.values():
                    timed.advance(depth, loop, iterations)
                    if timed.event is not None and timed.event.path:
                        outer = timed.event.path[0][0]
                        if outer in span:
                            span[outer][1] = max(span[outer][1], timed.at)
            continue
        event: _Event = item
        if event.kind != "sync":
            note(event)
            if event.line >= 0 and event.kind in ("drive", "readout", "wait"):
                last = free.get(event.line)
                if last is None or event.end >= last.at:
                    free[event.line] = _Timed(event.end, event)
            continue
        latest = max(
            (free[index] for index in event.lines if index in free),
            key=lambda timed: timed.at,
            default=None,
        )
        # Where it meets (gaps() gives it that again), as of its loops.
        event.start = event.end = latest.at if latest is not None else 0
        note(event)
        if event.site is not None:
            places.setdefault(event.site, set()).add(_where(event))
        source = latest.event if latest is not None else None
        shifts: tuple[int, ...] | None = None
        if source is not None:
            given = known.get(id(source)) if source.kind == "sync" else source.shifts
            if given is not None:
                shifts = (*given, *(0,) * len(event.path))[: len(event.path)]
                for depth, (loop, copy, _) in enumerate(event.path):
                    if copy and (depth >= len(source.path) or source.path[depth][0] != loop):
                        shifts = None
                        break
        known[id(event)] = shifts
        # gaps() gives every sync its shifts again as it walks.
        event.shifts = shifts or (0,) * len(event.path)
        if event.site is not None and any(copy for _, copy, _ in event.path):
            found.setdefault(_where(event), []).append(shifts)
        for index in event.lines:
            free[index] = _Timed(latest.at if latest is not None else 0, event)
    crowded: set[tuple[int, int]] = set()
    for qudit, around, start, end in on_qudit:
        low, high = (start, end) if around is None else span[around]
        for loop, (a, b) in span.items():
            if loop != around and a < high and low < b:
                crowded.add((qudit, loop))
    anchored = {
        site: seen[0]
        for site, seen in found.items()
        if seen[0] is not None
        and all(shifts == seen[0] for shifts in seen)
        and len(places[site[0]]) == 1
    }
    return anchored, crowded


def _lay(gap: Gap, readouts: list[_Event]) -> None:
    """Lay over *gap* the executions of its qudit's own *readouts* (each
    given at the first iteration of its loops) that run during it. In a
    loop that holds both, the gap stands for every iteration and a readout
    of any other iteration may be the one that runs during it, so every
    distance between two iterations is laid; a readout that repeats there
    with another period makes the gap uneven. Readouts too many to list
    are one hole from the first to the last."""
    beat = gap.lo or gap.hi
    for event in readouts:
        start = event.start
        terms = []
        shared = beat is not None
        for depth, (loop, _, trips) in enumerate(event.path):
            period = event.shifts[depth]
            shared = (
                shared
                and beat is not None
                and depth < len(beat.path)
                and beat.path[depth][0] == loop
            )
            if shared:
                assert beat is not None
                if beat.shifts[depth] != period:
                    gap.uneven = True
                start += beat.path[depth][1] * period
                terms.append((period, 1 - trips, trips - 1))
            else:
                terms.append((period, 0, trips - 1))
        terms = sorted((term for term in terms if term[0] > 0), reverse=True)
        length = event.end - event.start
        found: list[int] = []
        if not _starts(start, terms, gap.start - length, gap.end, found):
            # Its earliest and latest execution that can run during the gap.
            first = start + sum(min(p * a, p * b) for p, a, b in terms)
            last = start + sum(max(p * a, p * b) for p, a, b in terms) + length
            gap.holes.append((max(first, gap.start), min(last, gap.end)))
            gap.repeats.add("readout")
            continue
        gap.inner += [(event, at, at + length, "readout") for at in sorted(found)]


def _starts(
    offset: int, terms: list[tuple[int, int, int]], low: int, high: int, found: list[int]
) -> bool:
    """Add to *found* every ``offset + sum(m * step)`` strictly between
    *low* and *high*, m whole per term (step, least m, most m), steps
    positive and longest first; False once there are more than 64."""
    if not terms:
        if low < offset < high:
            found.append(offset)
        return len(found) <= 64
    (step, least, most), rest = terms[0], terms[1:]
    under = sum(min(s * a, s * b) for s, a, b in rest)
    over = sum(max(s * a, s * b) for s, a, b in rest)
    first = max(least, (low - over - offset) // step + 1)
    last = min(most, -((offset + under - high) // step) - 1)
    if last - first > 64:
        return False
    return all(_starts(offset + m * step, rest, low, high, found) for m in range(first, last + 1))


def _out_of_step(a: _Event, b: _Event) -> bool:
    """Whether *a* and *b* repeat with different periods in a loop that
    holds both."""
    for depth in range(min(len(a.path), len(b.path))):
        if a.path[depth][0] != b.path[depth][0]:
            return False
        if a.shifts[depth] != b.shifts[depth]:
            return True
    return False


def _merged(executions: list[Gap]) -> list[Gap]:
    """One gap per gap of the kernel, in the order of their first
    executions: the shortest execution, which is the one a fill must fit."""
    merged: dict[tuple[Any, ...], Gap] = {}
    for gap in executions:
        first = merged.setdefault(gap.key(), gap)
        if first is gap:
            continue
        gap.executions = first.executions
        first.executions.append(gap)
        gap.rounds += first.rounds
        gap.varies = first.varies or gap.shape() != first.shape()
        if gap.end - gap.start < first.end - first.start:
            merged[gap.key()] = gap
        else:
            first.rounds, first.varies = gap.rounds, gap.varies
    for gap in merged.values():
        if gap.varies:
            for execution in gap.executions:
                execution.varies = True
        # What is written into it is written into every execution: the
        # qudit is read out before (after) it only if before (after) each.
        gap.read_before = all(execution.read_before for execution in gap.executions)
        gap.read_after = all(execution.read_after for execution in gap.executions)
    return list(merged.values())


def _steady(gap: Gap, fills: set[tuple[int, int]], forward: bool) -> bool:
    """Whether the executions of *gap*, which differ, can all take what is
    written for the shortest: they differ only by more idle at one end.
    A schedule that places as soon as possible (*forward*) keeps what is
    written at the gap's start, one that places as late as possible at its
    end; where that is the end with the idle to spare, nothing may play on
    the qudit on the far side of a fill, which sits further that way in a
    longer round."""
    length = gap.end - gap.start
    flags = gap.shape()[2:]
    from_start = [(event.site, a - gap.start, b - gap.start, w) for event, a, b, w in gap.inner]
    from_end = [(event.site, a - gap.end, b - gap.end, w) for event, a, b, w in gap.inner]
    at_start = at_end = True
    for other in gap.executions:
        if other.shape()[2:] != flags or other.end - other.start < length:
            return False
        inner = other.inner
        at_start &= from_start == [
            (event.site, a - other.start, b - other.start, w) for event, a, b, w in inner
        ]
        at_end &= from_end == [
            (event.site, a - other.end, b - other.end, w) for event, a, b, w in inner
        ]
    if not fills:
        return at_start or at_end
    first, last = min(u for u, _ in fills), max(v for _, v in fills)
    if at_start and (forward or all(b <= first for _, _, b, _ in gap.inner)):
        return True
    return at_end and (not forward or all(a >= last for _, a, _, _ in gap.inner))


# ---------------------------------------------------------------- the report


@dataclass(frozen=True)
class IdleWindow:
    """One idle window of a compiled program: a stretch of at least the
    threshold in which a qudit's drive line has no operation, none of its
    tones plays, and its readout does not run.

    ``start`` and ``end`` are in clock cycles, of the window's first
    execution; ``rounds`` counts its executions (the iterations of a loop
    it lies in), and when they differ the shortest is given. ``before``
    names the operation the window ends at, as ``"drive() at line 12 of
    cycle()"``, or is ``None`` for a window that lasts to the program's
    end. ``filled`` says whether the window was replaced, and ``result``
    what was done: ``"filled by echo()"``, or why it was left alone.
    """

    qudit: int
    start: int
    end: int
    rounds: int
    before: str | None
    filled: bool
    result: str

    @property
    def cycles(self) -> int:
        """The window's length in clock cycles."""
        return self.end - self.start


class IdleWindows(tuple[IdleWindow, ...]):
    """The idle windows of a compiled program, in the order of their
    qudits and, per qudit, of the program. ``print()`` shows them as a
    table."""

    __slots__ = ()

    def __str__(self) -> str:
        if not self:
            return "no idle windows"
        rows = [("qudit", "start", "end", "cycles", "rounds", "before", "result")]
        for w in self:
            rows.append(
                (
                    str(w.qudit),
                    str(w.start),
                    str(w.end),
                    str(w.cycles),
                    str(w.rounds),
                    w.before or "the end of the program",
                    w.result,
                )
            )
        widths = [max(len(row[i]) for row in rows) for i in range(7)]
        return "\n".join(
            "  ".join(
                cell.rjust(widths[i]) if i < 5 else cell.ljust(widths[i])
                for i, cell in enumerate(row)
            ).rstrip()
            for row in rows
        )


def _in_body(gap: Gap) -> bool:
    """Whether *gap* starts and ends in the body of one loop."""
    lo, hi = gap.lo, gap.hi
    return (
        lo is not None
        and hi is not None
        and bool(lo.path)
        and bool(hi.path)
        and lo.path[0][0] == hi.path[0][0]
    )


def kept(gap: Gap, part: tuple[int, int], rule: Any, log: list[Site], forward: bool) -> str | None:
    """Why the idle part *part* of *gap* is left alone under *rule*
    whatever the fill, or None; *forward* says whether the schedule places
    operations as soon as possible."""
    lo = gap.lo
    # A window in the body of a loop where the qudit has an op of its own
    # is written once for every round: it is not the qudit's leading
    # window, even if the first round's comes before its first operation.
    if gap.first and not rule.leading and not gap.repeats_op:
        # Before the qudit's first operation; a pulse at its tone from
        # another line and its own readout are operations too.
        if not gap.read_before and not any(
            what in _ON_QUDIT and a < part[0] for _, a, _, what in gap.inner
        ):
            return "left alone: before the qudit's first operation (leading=True fills it)"
    if gap.last and not rule.trailing:
        if not gap.read_after and not any(
            what in _ON_QUDIT and a >= part[1] for _, a, _, what in gap.inner
        ):
            return "left alone: after the qudit's last operation (trailing=True fills it)"
    if gap.branched():
        return "left alone: in or next to an arm of a branch"
    if gap.barrier:
        return "left alone: after a wait whose length is known only when the program runs"
    if gap.detuned:
        return "left alone: the kernel changed the frequency of the qudit's tone"
    if gap.holes:
        if gap.repeats & {"readout", "tone", "busy"}:
            return (
                "left alone: a loop reads the qudit out or plays its tone in every round meanwhile"
            )
        if gap.repeats & {"frame", "frequency"}:
            return (
                "left alone: a loop changes the frame of the qudit's tone in every round meanwhile"
            )
        # A sync of the line in a loop ends the gap, unless it lies in a
        # branch (whose reason comes first), its shift per round is not
        # known, or the qudit is read out or its tone plays outside the
        # loop while it runs.
        return "left alone: a loop syncs the qudit's line in every round meanwhile"
    if gap.uneven or (gap.varies and (gap.wraps() or not _steady(gap, {part}, forward))):
        return "left alone: its rounds differ"
    if not gap.ordered():
        return "left alone: the program writes what surrounds it in another order than it runs"
    if gap.wraps():
        assert lo is not None and lo.site is not None
        # What plays on the qudit's tone before its first op of the next
        # iteration would have to be written around from two places. A sync
        # there that meets where the gap ends cuts nothing off.
        if any(
            what in _CHAINED
            and (event.site is None or event.site < lo.site)
            and not (what == "sync" and a == gap.end)
            for event, a, _, what in gap.inner
        ):
            return "left alone: its tone plays or changes at the start of the next round"
    return None


# ---------------------------------------------------------------- what is written where


class Tile:
    """What is written at one point of the program on one qudit's drive
    line: waits and fills back to back, as ``("wait", cycles)`` and
    ``("fill", cycles)``, in the region *scope*."""

    __slots__ = ("qudit", "items", "scope", "number")

    def __init__(
        self, qudit: int, items: list[tuple[str, int]], scope: tuple[int, ...], number: int
    ) -> None:
        self.qudit = qudit
        self.items = items
        self.scope = scope
        self.number = number


class Plan:
    """The tiles to write right before an op of the kernel and right after
    one, by the op's number. With *pads*, each fill is written as a wait
    of its window's length."""

    def __init__(self, pads: bool = False) -> None:
        self.before: dict[int, list[Tile]] = {}
        self.after: dict[int, list[Tile]] = {}
        # What is written after the kernel's last statement.
        self.end: list[Tile] = []
        self.pads = pads
        self.tiles = 0
        # By tile number, for each tile that holds a fill: how many times
        # the windows it fills run, as the windows report them.
        self.runs: dict[int, int] = {}

    def add(
        self,
        where: str,
        site: int,
        qudit: int,
        items: list[tuple[str, int]],
        scope: Any,
        runs: int = 0,
    ) -> None:
        tile = Tile(qudit, items, scope, self.tiles)
        if runs and any(kind == "fill" for kind, _ in items):
            self.runs[self.tiles] = runs
        if where == "end":
            self.end.append(tile)
        else:
            points = self.before if where == "before" else self.after
            points.setdefault(site, []).append(tile)
        self.tiles += 1


def _site(event: _Event | None, log: list[Site]) -> Site | None:
    if event is None or event.site is None or not 0 <= event.site < len(log):
        return None
    return log[event.site]


def _region(gap: Gap, log: list[Site]) -> tuple[int, ...]:
    """The region a gap lies in: the innermost that holds both the op it
    starts after and the op it ends at."""
    lo, hi = _site(gap.lo, log), _site(gap.hi, log)
    if lo is None or hi is None:
        return ()
    region = []
    for a, b in zip(lo.scope, hi.scope, strict=False):
        if a != b:
            break
        region.append(a)
    return tuple(region)


def _point(
    a: _Event | None, b: _Event | None, region: tuple[int, ...], log: list[Site], last: bool
) -> tuple[str, int] | None:
    """Where the stretch from *a* to *b* is written: right before *b*, else
    right after *a*, if the tracer is between two statements of *region*
    there; a stretch to the program's end (*b* None, *last*) also after
    the kernel's last statement. An op the passes made of several traced
    ones has no point after it: that point is inside it."""
    site = _site(b, log)
    if site is not None and region in site.enter:
        assert b is not None and b.site is not None
        return "before", b.site
    site = _site(a, log)
    if site is not None and region in site.leave:
        assert a is not None and a.site is not None
        if a.kind not in ("drive", "readout", "wait") or site.cycles == a.end - a.start:
            return "after", a.site
    if b is None and last and region == ():
        return "end", 0
    return None


def _tiles(
    gap: Gap, fills: set[tuple[int, int]], log: list[Site]
) -> list[tuple[str, int, list[tuple[str, int]], tuple[int, ...]]] | None:
    """The whole of *gap* as waits and fills (its idle parts in *fills*),
    cut where the kernel plays or changes the qudit's tone: each piece with
    the point it is written at, or None if one has none. The whole gap is
    written so that its line has no slack left: every piece then stays
    where it is planned under every schedule policy.

    A gap across the end of a loop iteration is written at the end of the
    loop's body: its last piece after the op before it, never before the
    op it ends at, which starts the next iteration."""
    region = _region(gap, log)
    wraps = gap.wraps()
    # In a gap across the end of an iteration, only what the body writes
    # after the op the gap starts after cuts it: that op's number.
    after = gap.lo.site if wraps and gap.lo is not None and gap.lo.site is not None else -1
    bounds = [
        (event, start)
        for event, start, _, what in gap.inner
        if what in _CHAINED and (after < 0 or (event.site is not None and event.site >= after))
    ]
    edges = [gap.start, *(start for _, start in bounds), gap.end]
    points = [gap.lo, *(event for event, _ in bounds), None if wraps else gap.hi]
    parts = gap.idle()
    out = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        items: list[tuple[str, int]] = []

        def wait(cycles: int, items: list[tuple[str, int]] = items) -> None:
            if items and items[-1][0] == "wait":
                items[-1] = ("wait", items[-1][1] + cycles)
            else:
                items.append(("wait", cycles))

        at = a
        for u, v in parts:
            if v <= a or u >= b:
                continue
            if u > at:
                wait(u - at)
            if (u, v) in fills:
                items.append(("fill", v - u))
            else:
                wait(v - u)
            at = v
        if b > at:
            wait(b - at)
        where = _point(points[k], points[k + 1], region, log, gap.hi is None)
        if where is None:
            return None
        out.append((*where, items, region))
    return out


_NO_POINT = "left alone: the program has no statement to write it before or after"


def _shortened(gap: Gap, cycles: int) -> Gap:
    """*gap* without its first *cycles* cycles, which something written
    before it takes."""
    import copy

    rest = copy.copy(gap)
    rest.start = gap.start + cycles
    rest.first = False
    rest.inner = [
        (event, a, b, what if a >= rest.start else "busy")
        for event, a, b, what in gap.inner
        if b > rest.start
    ]
    return rest


def _last_round(listed: list[Gap], index: int, log: list[Site]) -> int | str:
    """For the gap *listed[index]*, which runs across the end of a loop
    iteration and is written at the end of the loop's body: the gap after
    the loop's last iteration, where what is written runs once more, by
    its place in *listed*; or why the gap cannot be written."""
    gap = listed[index]
    assert gap.lo is not None
    length = gap.end - gap.start
    for place in range(index + 1, len(listed)):
        after = listed[place]
        if after.lo is not None and after.lo.site == gap.lo.site and not after.wraps():
            break
    else:
        return _NO_POINT
    if after.varies or after.branched() or after.end - after.start < length:
        return "left alone: there is less room for it after the loop's last round"
    # What plays on the qudit there must be what plays in every round.
    rounds = {(event.site, a - gap.start, b - gap.start) for event, a, b, _ in gap.inner}
    for event, a, b, _ in after.inner:
        if a < after.start + length:
            if (event.site, a - after.start, b - after.start) not in rounds:
                return "left alone: the qudit is not idle as long after the loop's last round"
    # The rest of that gap is written too, so that the last round's part
    # has no slack: it must have a place.
    rest = _shortened(after, length)
    if rest.end > rest.start and _tiles(rest, set(), log) is None:
        return _NO_POINT
    return place


# Why a window is left unfilled under rcp or alap_rcp: those policies give
# each drive lane one time it is free from, so a pulse written into one
# line's idle time can take a lane another line's pulse used (until the
# scheduler fills such gaps). LANES: its fill, tried with those kept before
# it, moved or stretched the kernel's operations. UNTRIED: it was not tried
# (past the windows tried one at a time, or there only because another
# window was left unfilled).
LANES = "left alone: its fill would move another operation under this schedule policy's drive lanes"
UNTRIED = (
    "left alone: under this schedule policy each window's fill is tried with the "
    "others kept, and this one was not tried"
)


def plan(
    found: dict[int, list[Gap]],
    rule: Any,
    log: list[Site],
    threshold: int,
    chosen: frozenset[int] | None,
    forward: bool,
    pads: bool = False,
    allowed: frozenset[tuple[int, int]] | None = None,
    refused: frozenset[tuple[int, int]] = frozenset(),
    earlier: dict[tuple[int, int], str] | None = None,
    nothing: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[Plan, IdleWindows]:
    """What to write for *rule* into the gaps *found*, and the windows with
    what is done in each; *forward* says whether the schedule places
    operations as soon as possible. With *allowed* (windows as (qudit, end)),
    only those windows are filled: the others are left alone, those in
    *refused* for the rcp policies' lane accounting (``LANES``), those in
    *earlier* (the reasons a plan without *allowed* gave the windows it did
    not fill) for that reason, any other as not tried (``UNTRIED``). A window whose length the fill
    writes nothing for on its qudit (*nothing*, as (qudit, cycles)) is left
    alone and says so."""
    planned = Plan(pads)
    name = "a wait" if pads else f"{rule.fill.__name__}()"
    rows = []
    for qudit in sorted(found):
        if chosen is not None and qudit not in chosen:
            continue
        listed = found[qudit]
        # By place in listed: the cycles a gap loses to the last round of
        # a loop's gap written before it.
        taken: dict[int, int] = {}
        for index, gap in enumerate(listed):
            if index in taken:
                gap = _shortened(gap, taken[index])
            parts = [part for part in gap.idle() if part[1] - part[0] >= threshold]
            reasons = {part: kept(gap, part, rule, log, forward) for part in parts}
            if nothing:
                for part in parts:
                    if reasons[part] is None and (qudit, part[1] - part[0]) in nothing:
                        reasons[part] = (
                            f"left alone: {rule.fill.__name__}() plays no pulse in a window "
                            f"of {part[1] - part[0]} cycles"
                        )
            fills = {part for part in parts if reasons[part] is None}
            held = False
            if allowed is not None:
                for part in [part for part in fills if (qudit, part[1]) not in allowed]:
                    key = (qudit, part[1])
                    reasons[part] = LANES if key in refused else (earlier or {}).get(key, UNTRIED)
                    fills.discard(part)
                    held = True
            if fills and gap.varies and not _steady(gap, fills, forward):
                reasons.update(dict.fromkeys(fills, "left alone: its rounds differ"))
                fills = set()
            rounds = gap.rounds
            if fills and gap.wraps():
                last = _last_round(listed, index, log)
                if last.__class__ is str:
                    reasons.update(dict.fromkeys(fills, last))
                    fills = set()
                else:
                    taken[last] = gap.end - gap.start  # type: ignore[index]
                    rounds += listed[last].rounds  # type: ignore[index]
            if gap.varies and not fills and index not in taken and not held:
                rows += _apart(gap, rule, log, threshold, forward)
                continue
            if fills or (index in taken and gap.end > gap.start):
                tiles = _tiles(gap, fills, log)
                if tiles is None:
                    # Not after a loop's gap: _last_round() found its place.
                    reasons.update(dict.fromkeys(fills, _NO_POINT))
                else:
                    for where, site, items, region in tiles:
                        planned.add(where, site, qudit, items, region, rounds)
            for part in parts:
                reason = reasons[part]
                rows.append(
                    IdleWindow(
                        qudit,
                        part[0],
                        part[1],
                        rounds,
                        _before(gap, part, log),
                        reason is None,
                        reason or f"filled by {name}",
                    )
                )
    return planned, IdleWindows(rows)


# ---------------------------------------------------------------- the proof


def _shape(item: Any) -> Any:
    """What an item of a program read back must share with the same item
    of the program before its windows were filled."""
    if item.__class__ is tuple:
        # Loops are numbered as they are met; a fill may hold some.
        return (item[0], *item[2:])
    where = (tuple((copy, trips) for _, copy, trips in item.path), len(item.arms))
    if item.kind == "sync":
        return (item.site, "sync", item.lines, where)
    return (item.site, item.kind, item.line, item.tone, item.start, item.end, item.shifts, where)


def prove(
    lines: list[tuple[str, int]],
    base: list[Any],
    module: Any,
    log: list[Site],
    name: str,
    forward: bool,
    runs: dict[int, int] | None = None,
) -> None:
    """Check that *module*, the program of *base* with its windows filled,
    is that program with something written into its gaps: every drive,
    readout, wait and sync of the kernel is where it was, with its loop
    periods, and so is every change of a frame but a phase shift (the
    passes fold a phase shift into the pulse after it, which may now be a
    fill's); what was written has no slack on the side the schedule places
    from; and no written pulse plays while its qudit is read out or its
    tone plays from another line. A ``CompilationError`` names the first op
    that moved. With *runs* (``Plan.runs``), each tile that holds a fill
    must run as many times as the windows it fills report."""
    filled_lines, final = read(module)
    if filled_lines != lines:
        raise InternalError("idle replacement: the filled program has other lines")
    base = [item for item in base if item.__class__ is tuple or item.kind != "shift_phase"]
    # The kernel's own ops, and of the loops only those that hold some.
    loops = {
        loop
        for item in final
        if item.__class__ is not tuple and (item.site is None or item.site >= 0)
        for loop, _, _ in item.path
    }
    own = [
        item
        for item in final
        if (
            (item[0] != "advance" or item[1] in loops)
            if item.__class__ is tuple
            else ((item.site is None or item.site >= 0) and item.kind != "shift_phase")
        )
    ]
    for was, now in zip(base, own, strict=False):
        a, b = _shape(was), _shape(now)
        if a == b:
            continue
        site = _site(was if was.__class__ is not tuple else None, log)
        if site is None or a[:2] != b[:2] or a[1] == "sync":
            break
        if a[4] != b[4]:
            moved = f"starts at cycle {b[4]} instead of {a[4]}"
        elif a[6] != b[6] and len(a[6]) == len(b[6]):
            moved = (
                f"repeats every {', '.join(map(str, b[6]))} cycles instead of every "
                f"{', '.join(map(str, a[6]))}"
            )
        elif a[6] != b[6] or a[7] != b[7]:
            moved = "is in other iterations of its loops than it was"
        else:
            moved = f"ends at cycle {b[5]} instead of {a[5]}"
        raise CompilationError(
            f"idle replacement: with the idle windows of {name}() filled, "
            f"{site.describe()} {moved}",
            code="timing",
        )
    else:
        if len(base) == len(own):
            _back_to_back(final, name, forward)
            _clear_of_the_qudit(lines, final, name)
            if runs:
                _runs_as_reported(final, runs, loops, name)
            return
    raise InternalError(
        f"idle replacement: {name}() traced differently the second time, so its filled "
        "program is not the program with its windows filled"
    )


def _runs_as_reported(final: list[Any], runs: dict[int, int], loops: set[int], name: str) -> None:
    """Check that each tile in *runs* runs, in the program read back as
    *final*, as many times as the windows it fills report: an op listed in
    a loop's second copy stands for every iteration after the first. Only
    the kernel's own *loops* count; a loop of the fill is the fill's."""
    seen: dict[int, set[tuple[Any, ...]]] = {}
    ran: dict[int, int] = {}
    # The tiles with a written pulse.
    played: set[int] = set()
    # Where the last of the kernel's own ops is: copies of a loop body the
    # schedule made (peeled or unrolled) keep their ops' numbers and the
    # loops around them, but each is after another op.
    after = -1
    for place, item in enumerate(final):
        if item.__class__ is tuple:
            continue
        if item.site is None or item.site >= 0:
            after = place
            continue
        tile = -1 - item.site
        if item.kind == "drive":
            played.add(tile)
        where = (after, *(entry for entry in item.path if entry[0] in loops))
        if tile not in runs or where in seen.setdefault(tile, set()):
            continue
        seen[tile].add(where)
        count = 1
        for _, copy, trips in where[1:]:
            count *= trips - 1 if copy else 1
        ran[tile] = ran.get(tile, 0) + count
    for tile, reported in runs.items():
        if tile not in played:
            raise InternalError(
                f"idle replacement: a fill written into {name}() plays no pulse where the "
                "windows it fills report it filled"
            )
        if ran.get(tile, 0) != reported:
            raise InternalError(
                f"idle replacement: a fill written into {name}() runs {ran.get(tile, 0)} "
                f"times where the windows it fills report {reported}"
            )


def _meets(offset: int, terms: list[tuple[int, int, int]], low: int, high: int) -> bool:
    """Whether ``offset + sum(m * step)`` lies strictly between *low* and
    *high* for some whole m per term (step, least m, most m), steps
    positive and longest first."""
    if not terms:
        return low < offset < high
    (step, least, most), rest = terms[0], terms[1:]
    under = sum(min(s * a, s * b) for s, a, b in rest)
    over = sum(max(s * a, s * b) for s, a, b in rest)
    first = max(least, (low - over - offset) // step + 1)
    last = min(most, -((offset + under - high) // step) - 1)
    if last - first > 4096:
        # Loops whose periods do not nest: more cases than are worth trying.
        return True
    return any(_meets(offset + m * step, rest, low, high) for m in range(first, last + 1))


def _overlap(a: _Event, b: _Event) -> bool:
    """Whether some execution of *a* overlaps some execution of *b*, over
    every iteration of the loops around them."""

    def first(event: _Event) -> int:
        return event.start - sum(
            copy * shift for (_, copy, _), shift in zip(event.path, event.shifts, strict=True)
        )

    terms = []
    shared = True
    for depth in range(max(len(a.path), len(b.path))):
        mine = a.path[depth] if depth < len(a.path) else None
        other = b.path[depth] if depth < len(b.path) else None
        shared = shared and mine is not None and other is not None and mine[0] == other[0]
        if shared and a.shifts[depth] == b.shifts[depth]:
            # Any two iterations of a loop that holds both.
            assert mine is not None
            terms.append((a.shifts[depth], 1 - mine[2], mine[2] - 1))
            continue
        if mine is not None:
            terms.append((a.shifts[depth], 0, mine[2] - 1))
        if other is not None:
            terms.append((b.shifts[depth], 1 - other[2], 0))
    terms = sorted((term for term in terms if term[0] > 0), reverse=True)
    # a starts before b ends, and ends after b starts.
    return _meets(first(a) - first(b), terms, a.start - a.end, b.end - b.start)


def _clear_of_the_qudit(lines: list[tuple[str, int]], final: list[Any], name: str) -> None:
    """Check that no pulse written into a window plays while its qudit's
    own readout runs or its tone plays from another line, in any iteration
    of the loops around either: the gaps are found from two iterations of
    each loop, and this is what they must have got right."""
    readout_of = {qudit: index for index, (role, qudit) in enumerate(lines) if role == "readout"}
    written: dict[int, list[_Event]] = {}
    busy: dict[int, list[_Event]] = {}
    for item in final:
        if item.__class__ is tuple or item.kind not in _PLAYS:
            continue
        role, qudit = lines[item.line] if item.line >= 0 else ("", -1)
        if item.site is not None and item.site < 0:
            if role == "drive":
                written.setdefault(qudit, []).append(item)
        elif item.kind == "readout" and role == "readout":
            busy.setdefault(qudit, []).append(item)
        if item.tone >= 0 and item.tone != item.line and (item.site is None or item.site >= 0):
            tone_role, owner = lines[item.tone]
            if tone_role == "drive":
                busy.setdefault(owner, []).append(item)
    for qudit, pulses in written.items():
        for other in busy.get(qudit, ()):
            for pulse in pulses:
                if _overlap(pulse, other):
                    what = (
                        "its own readout runs"
                        if other.line == readout_of.get(qudit)
                        else "its tone plays from another line"
                    )
                    raise InternalError(
                        f"idle replacement: a pulse written into an idle window of {name}() "
                        f"on qudit {qudit} (first at cycle {pulse.start}) plays while {what} "
                        f"(first at cycle {other.start}); until this is fixed, qudits= can "
                        "leave the qudit out"
                    )


def _back_to_back(final: list[Any], name: str, forward: bool) -> None:
    """Check that what was written into a gap has no slack on the side the
    schedule places from: under a policy that places as soon as possible
    (*forward*) each written op starts where the op before it on its line
    ends (or where a sync of the line meets); under one that places as late
    as possible the op after a written one starts where it ends. A gap
    written to the cycle is then tight on both sides; one written for the
    shortest of its rounds has its spare idle on the other."""
    ends: dict[int, int] = {}
    latest: dict[int, _Event] = {}
    tiled: set[int] = set()

    def moved(what: str) -> CompilationError:
        return CompilationError(
            f"idle replacement: what was written into an idle window of {name}() does not "
            f"stay in place: {what}",
            code="timing",
        )

    for item in final:
        if item.__class__ is tuple:
            if item[0] == "advance":
                # The iterations of a loop that are not listed: its lines
                # are that much further on.
                _, loop, depth, iterations = item
                for line, event in latest.items():
                    ends[line] += event.advance(depth, loop, iterations)
            continue
        if item.kind == "sync":
            # Its lines go on from where the last of them is free.
            meet = max((ends.get(line, 0) for line in item.lines), default=0)
            last = max(
                (latest[line] for line in item.lines if line in latest),
                key=lambda event: event.end,
                default=None,
            )
            for line in item.lines:
                if not forward and line in tiled and ends[line] != meet:
                    raise moved(
                        f"line {line} is free at cycle {ends[line]}, before the sync it was "
                        f"written up to, at {meet}"
                    )
                ends[line] = meet
                # The sync is the op after what was written before it: what
                # follows it on the line is another stretch, which a gap
                # that ends at a sync may have left unwritten.
                tiled.discard(line)
                if last is not None:
                    # The sync is in its own iteration of the loops around
                    # it, and repeats as the op it waited for does.
                    item.shifts = (*last.shifts, *(0,) * len(item.path))[: len(item.path)]
                    latest[line] = item
            continue
        if item.kind not in ("drive", "readout", "wait"):
            continue
        line = item.line
        written = item.site is not None and item.site < 0
        expected = ends.get(line, 0)
        if (written if forward else line in tiled) and item.start != expected:
            raise moved(
                f"a {item.kind} on line {line} starts at cycle {item.start}, not at "
                f"{expected} where the operation before it ends"
            )
        if written:
            tiled.add(line)
        else:
            tiled.discard(line)
        ends[line] = item.end
        latest[line] = item


def _apart(gap: Gap, rule: Any, log: list[Site], threshold: int, forward: bool) -> list[IdleWindow]:
    """The windows of a gap whose rounds differ and that is left alone:
    those of each kind of round, with the rounds of that kind."""
    kinds: dict[tuple[Any, ...], list[Any]] = {}
    for execution in gap.executions:
        kinds.setdefault(execution.shape(), [execution, 0])[1] += execution.own
    rows = []
    for execution, rounds in kinds.values():
        for part in execution.idle():
            if part[1] - part[0] >= threshold:
                reason = kept(execution, part, rule, log, forward)
                rows.append(
                    IdleWindow(
                        gap.qudit,
                        part[0],
                        part[1],
                        rounds,
                        _before(execution, part, log),
                        False,
                        reason or "left alone: its rounds differ",
                    )
                )
    return rows


def report(
    found: dict[int, list[Gap]],
    rule: Any,
    log: list[Site],
    threshold: int,
    chosen: frozenset[int] | None,
    forward: bool,
) -> IdleWindows:
    """The windows of *found* under *rule*, none of them filled: the idle
    parts of at least *threshold* cycles on the qudits *chosen* (None:
    every qudit); *forward* says whether the schedule places operations as
    soon as possible."""
    rows = []
    for qudit in sorted(found):
        if chosen is not None and qudit not in chosen:
            continue
        for gap in found[qudit]:
            parts = {part for part in gap.idle() if part[1] - part[0] >= threshold}
            if gap.varies and not (parts and _steady(gap, parts, forward)):
                rows += _apart(gap, rule, log, threshold, forward)
                continue
            for part in gap.idle():
                if part[1] - part[0] < threshold:
                    continue
                reason = kept(gap, part, rule, log, forward)
                rows.append(
                    IdleWindow(
                        qudit,
                        part[0],
                        part[1],
                        gap.rounds,
                        _before(gap, part, log),
                        False,
                        reason or "not filled: the rule has no fill",
                    )
                )
    return IdleWindows(rows)


def _before(gap: Gap, part: tuple[int, int], log: list[Site]) -> str | None:
    """What the idle part *part* of *gap* ends at."""
    ending = gap.hi
    for event, start, _, _ in gap.inner:
        if start == part[1]:
            ending = event
            break
    if ending is None:
        return None
    if ending.site is None or not 0 <= ending.site < len(log):
        return f"{ending.kind}()"
    return log[ending.site].describe()
