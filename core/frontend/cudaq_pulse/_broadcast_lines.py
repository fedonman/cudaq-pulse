# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Broadcast drive lines: a ``DriveLine`` that reaches several qudits with
one tone, ``DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.1e9)``.

A kernel gets the line with ``get_drive_line(q)`` of any of its qudits, or
with ``get_drive_line(a, b, c)`` of all of them; either way it is the one
line, with one tone, and a drive on it drives every qudit it reaches. The
program's IR names the line by all its qudits, lowest first, and the
scheduler, the frame bookkeeping and the controller checks see one line of
one tone, as a dedicated line is; the simulator drives each qudit, off
resonance by the difference between the tone's frequency and its own; the
exports write the line once, with its ``qudits``, and one event per drive.

Only a control system with a broadcast line loads this: its model
(``machine_model``) sets up the trace and the module of each program
compiled for it, and its exports are written here (``flat``,
``structured``, ``framed``)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from . import _shared_lines
from ._errors import (
    CompilationError,
    InvalidTypeError,
    InvalidValueError,
    UnsupportedError,
    _ranges,
)
from ._system import DriveLine

if TYPE_CHECKING:
    from ._system import ControlSystem, _Line

# ── The line ────────────────────────────────────────────────────────────────

# A line's default max_concurrent_tones (an int 1 is this object).
_ONE_TONE = 1


class _BroadcastDriveLine(DriveLine):
    """``DriveLine(qudits, broadcast=True)``: ``(qudits, 1, delay_ns,
    frequency_hz)``, a ``DriveLine`` that writes itself with what it was
    declared with. A class of its own so that no other line asks whether it
    is one."""

    __slots__ = ()

    def __repr__(self) -> str:
        qudits, _, delay, hz = self
        return (
            f"DriveLine({list(qudits)}{f', delay_ns={delay!r}' if delay else ''}"
            f", broadcast=True{'' if hz is None else f', frequency_hz={hz!r}'})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        qudits, _, delay, hz = self
        return (_shared_lines.line, (DriveLine, qudits, delay, 1, True, hz))

    def _undelayed(self) -> Any:
        return tuple.__new__(_BroadcastDriveLine, (self[0], 1, 0.0, self[3]))

    def _data(self) -> dict[str, Any]:
        """The line as a system file holds it, without the defaults."""
        qudits, _, delay, hz = self
        found: dict[str, Any] = {"qudits": list(qudits), "broadcast": True}
        if hz is not None:
            found["frequency_hz"] = hz
        if delay:
            found["delay_ns"] = delay
        return found


def line(
    cls: type[_Line], reached: tuple[int, ...], delay_ns: Any, tones: Any, broadcast: Any, hz: Any
) -> Any:
    """A *cls* line reaching the qudits *reached* (checked) that was given
    ``broadcast=`` or ``frequency_hz=``: the broadcast line, with its delay
    and frequency checked."""
    from ._system import _delay_ns

    name = cls.__name__
    said = f"{name}({list(reached)}, broadcast=True)"
    if not isinstance(broadcast, bool):
        raise InvalidTypeError(f"{name}.broadcast is True or False, got {broadcast!r}")
    if not broadcast:
        raise InvalidValueError(
            f"{name}({list(reached)}, frequency_hz={hz!r}): frequency_hz is the frequency "
            "of a broadcast line's one tone; give broadcast=True with it, or leave it out"
        )
    if cls._role != "drive":
        raise UnsupportedError(
            f"{said}: a readout line whose one probe reads out several qudits is not "
            "supported yet; broadcast=True is for a DriveLine"
        )
    if len(reached) == 1:
        raise InvalidValueError(
            f"{name}({reached[0]}) reaches one qudit, so broadcast=True says nothing; leave it out"
        )
    if tones is not _ONE_TONE:
        raise InvalidValueError(
            f"{said} drives one tone that reaches every qudit; max_concurrent_tones is for "
            "a line with a tone per qudit"
        )
    if hz is not None:
        if isinstance(hz, bool) or not isinstance(hz, (int, float)):
            raise InvalidTypeError(f"{name}.frequency_hz is a number of Hz, got {hz!r}")
        hz = float(hz)
        if not 0.0 < hz < math.inf:
            raise InvalidValueError(
                f"{name}.frequency_hz is the frequency of the line's tone in Hz, a positive "
                f"number; got {hz!r}"
            )
    if delay_ns.__class__ is not float or not 0.0 <= delay_ns < math.inf:
        delay_ns = _delay_ns(delay_ns, f"{name}({list(reached)}).delay_ns")
    # Item 1 is never 0, as a shared line's: the delay filters see it.
    return tuple.__new__(_BroadcastDriveLine, (reached, 1, delay_ns, hz))


def _named(qudits: Sequence[int]) -> str:
    return f"the broadcast drive line of qudits {_ranges(list(qudits))}"


def _lines(system: ControlSystem) -> Mapping[int, _Line]:
    """By each qudit a broadcast line of *system* reaches, the line. Made
    once per system."""
    cache = system.__dict__
    try:
        return cache["_broadcast_lines"]  # type: ignore[no-any-return]
    except KeyError:
        pass
    found: dict[int, _Line] = {}
    for controller in system.controllers:
        for each in cast("Sequence[_Line]", controller.drive_lines):
            if len(each) > 3:
                found.update(dict.fromkeys(each[0], each))
    object.__setattr__(system, "_broadcast_lines", found)
    return found


def _tone_hz(found: _Line, target: Any) -> tuple[float | None, str | None]:
    """The frequency of broadcast line *found*'s tone on quantum system
    *target*: the one it states, else the one its qudits share; without
    either, ``None`` and the error (no error when *target* lacks one of
    its qudits, which is an error of its own)."""
    qudits, _, _, hz = found
    frame = target.frequency_frame
    if frame == "own_carrier":
        # Each tone of such a system is an offset from its own qudit's
        # carrier. The line's one tone is at the carrier of every qudit it
        # reaches, and a change of its frequency is that offset from each.
        if hz is not None:
            return None, (
                f"quantum system {target.name!r}: {_named(qudits)} states its tone as "
                f"frequency_hz={hz!r}, and the system's qudits give no frequency_hz: the "
                "line's tone is at the carrier of each qudit it reaches, and no frequency "
                "places it; leave frequency_hz out, and detune the tone in the kernel "
                "(shift_frequency(line.tone, ...)) by the offset every qudit is to see"
            )
        return 0.0, None
    if frame == "shared_reference" and hz is not None:
        return None, (
            f"quantum system {target.name!r}: {_named(qudits)} states its tone as "
            f"frequency_hz={hz!r}, and the system's qudits give Offsets from a reference it "
            "does not state, so the tone cannot be placed among them; leave frequency_hz out "
            "where the line's qudits share one Offset"
        )
    if hz is not None:
        return hz, None
    known = target.frequencies
    try:
        own = [known[q] for q in qudits]
    except KeyError:
        return None, None
    if len(set(own)) == 1:
        return own[0], None
    if frame == "shared_reference":
        mhz = [f"{f / 1e6:g}" for f in own]
        return None, (
            f"quantum system {target.name!r}: {_named(qudits)} has one tone, and its qudits "
            f"are at Offsets of {', '.join(mhz[:-1])} and {mhz[-1]} MHz from their reference; "
            "a broadcast line cannot state its tone on a system of Offsets yet: its qudits "
            "share one Offset, or the system gives their frequencies"
        )
    ghz = [f"{f / 1e9:g}" for f in own]
    return None, (
        f"quantum system {target.name!r}: {_named(qudits)} has one tone, and its qudits are "
        f"at {', '.join(ghz[:-1])} and {ghz[-1]} GHz; state the tone's frequency, as "
        f"DriveLine({list(qudits)}, broadcast=True, frequency_hz=...): each qudit is then "
        "driven off resonance by the difference from its own"
    )


# ── Compiling for a control system with a broadcast line ────────────────────

_MODELS: dict[Any, Any] = {}


def machine_model(model: Any) -> Any:
    """The class of the model that compiles for a control system with a
    broadcast line: *model* (``compiler._MachineModel``) whose
    ``_clocked_builder`` also sets the trace up for the line, and whose
    ``_clocked_module`` gives the module's line all its qudits."""
    found = _MODELS.get(model)
    if found is None:
        found = _MODELS[model] = type(
            "_BroadcastMachineModel",
            (model,),
            {
                "__module__": __name__,
                "__doc__": "``compiler._MachineModel`` of a system with a broadcast line.",
                "_clocked_builder": _clocked_builder,
                "_clocked_module": _clocked_module,
                "_emulation": staticmethod(_emulation),
                "_emulation_part": staticmethod(_emulation_part),
                "__reduce__": _model_reduce,
            },
        )
    return found


def _model_reduce(self: Any) -> tuple[Any, ...]:
    """The model as ``_rebuilt_model`` and its constructor inputs: its class
    is made when a control system first needs it, so a pickle does not name
    it."""
    from ._pickling import frozen_reduce

    return (_rebuilt_model, (frozen_reduce(self)[1][1],))


def _rebuilt_model(inputs: dict[str, Any]) -> Any:
    from .compiler import _MachineModel

    return machine_model(_MachineModel)(**inputs)


class _Reached:
    """A broadcast line as a trace takes it: its qudits, the tracer's value
    for each (the line is acquired from its lowest), and the error of a
    line whose tone has no frequency on the quantum system."""

    __slots__ = ("qudits", "values", "vids", "error")

    def __init__(self, qudits: tuple[int, ...], error: str | None, builder: Any) -> None:
        from .kernel.ir_builder import IRValue

        self.qudits = qudits
        self.error = error
        first = builder._next_id
        builder._next_id = first + len(qudits)
        self.values = [IRValue(first + k, "qref") for k in range(len(qudits))]
        # What the builder records of the line's qudits once it is acquired,
        # by the id of each one's value: the qudit, and that the program
        # takes a line on it. The ids are older than any the trace gives, so
        # a loop body traced again keeps them.
        self.vids = {first + k: q for k, q in enumerate(qudits)}


def _clocked_builder(self: Any, builder: Any, target: Any) -> Any:
    """``_MachineModel._clocked_builder``, and *builder* set up for the
    broadcast lines of the model's system: its tracer, and by qudit the
    line that reaches it."""
    from .compiler import _MachineModel

    if not hasattr(builder, "_next_id"):
        # pulse.load: a function written by another compiler, not traced here.
        raise UnsupportedError(
            "load() does not give a loaded function a broadcast drive line yet, and this "
            "control system has one (DriveLine([...], broadcast=True)); load on a control "
            "system whose drive lines each reach one qudit or have a tone per qudit"
        )
    clocks = _MachineModel._clocked_builder(self, builder, target)
    reached: dict[int, _Reached] = {}
    for found in dict.fromkeys(_lines(self._system).values()):
        entry = _Reached(found[0], _tone_hz(found, target)[1], builder)
        reached.update(dict.fromkeys(found[0], entry))
    builder.broadcast_lines = reached
    # On a system whose qudits give no frequency, the ids of the lines'
    # values: no tone crosses between such a line and a qudit (_own_tone).
    builder.broadcast_carriers = (
        frozenset(entry.values[0].vid for entry in reached.values())
        if target.frequency_frame == "own_carrier"
        else None
    )
    builder.tracer = _tracer(builder.tracer)
    return clocks


def armed_like(found: Any, builder: Any) -> None:
    """Set *found*, a new builder, up for the broadcast lines as *builder*
    was before its trace, so that the same trace writes the same records
    (``PackedIRBuilder.finder``)."""
    found.broadcast_lines = builder.broadcast_lines
    found.broadcast_carriers = builder.broadcast_carriers
    found.tracer = builder.tracer
    found._next_id += sum(
        len(entry.qudits) for entry in dict.fromkeys(builder.broadcast_lines.values())
    )


def _clocked_module(
    self: Any, module: Any, target: Any, clocks: Any, max_duration_cycles: int | None
) -> Any:
    """``_MachineModel._clocked_module``, and each broadcast line *module*
    acquired (from its lowest qudit) given all its qudits and its tone's
    frequency (the ``pulse-broadcast-lines`` pass)."""
    from .compiler import _MachineModel

    fine_delays = _MachineModel._clocked_module(self, module, target, clocks, max_duration_cycles)
    entries = []
    for found in dict.fromkeys(_lines(self._system).values()):
        hz = _tone_hz(found, target)[0]
        if hz is not None:
            entries.append(
                f"{{qudits = [{', '.join(map(str, found[0]))}], frequency_hz = {hz!r} : f64}}"
            )
    if entries:
        module.set_attributes(f"{{pulse.broadcast_lines = [{', '.join(entries)}]}}")
        module.run_passes(["pulse-broadcast-lines"])
    return fine_delays


_TRACERS: dict[Any, Any] = {}


def _tracer(base: Any) -> Any:
    """The tracer *base* (``None``: the plain one) that takes a broadcast
    line by any of its qudits, or by all of them."""
    from .kernel._emitter import PulseIREmitter

    base = base or PulseIREmitter
    found = _TRACERS.get(base)
    if found is None:
        found = _TRACERS[base] = type(
            f"Broadcast{base.__name__}", (_Broadcast, base), {"__slots__": ()}
        )
    return found


class _Broadcast:
    """Traces ``get_drive_line`` of a broadcast line's qudits as the one
    line, then as the tracer it is mixed into: the line is that of its
    lowest qudit, whichever of its qudits the kernel names, so every pulse
    on it follows the one before and its tone is one frame."""

    __slots__ = ()

    def _emit_pulse_call(self, fname: str, call_args: list[Any]) -> tuple[Any, ...]:
        builder: Any = self.b  # type: ignore[attr-defined]
        if fname == "get_drive_line":
            call_args = _line_arguments(builder, call_args)
        elif builder.broadcast_carriers and len(call_args) > 2 and fname in ("drive", "readout"):
            _own_tone(builder, fname, call_args)
        return super()._emit_pulse_call(fname, call_args)  # type: ignore[misc,no-any-return]

    def _value_attribute(self, value: Any, name: str) -> Any:
        chan = value.chan
        if chan is not None and value.vtype == "drive_line" and name != "tone":
            builder: Any = self.b  # type: ignore[attr-defined]
            entry = builder.broadcast_lines.get(builder._qref_to_qudit.get(chan.held.key[0]))
            if entry is not None and chan.held.key[0] == entry.values[0].vid:
                raise CompilationError(
                    f"{_named(entry.qudits)} has one tone, line.tone, which reaches every "
                    f"qudit; it has no line.{name}",
                    code="unknown_attribute",
                )
        return super()._value_attribute(value, name)  # type: ignore[misc]


def _own_tone(builder: Any, fname: str, call_args: list[Any]) -> None:
    """On a quantum system whose qudits give no frequency: refuse *fname*
    (a drive or readout with a tone given) on a broadcast line at a tone
    that is not the line's, or on another line at a broadcast line's tone.
    The line's tone is at the carrier of each qudit it reaches and is no
    single qudit's, its lowest's included, which the system's own check of
    two qudits' tones (``compiler._refuse_other_qudits_tones``) takes the
    line for. Values whose line the trace does not record are that check's."""
    owners = []
    for value in (call_args[0], call_args[2]):
        chan = getattr(value, "chan", None)
        if chan is None:
            return
        owners.append(chan.held.key[0])
    line, tone = owners
    carriers = builder.broadcast_carriers
    if line != tone and (line in carriers or tone in carriers):
        entry = builder.broadcast_lines[builder._qref_to_qudit[line if line in carriers else tone]]
        raise CompilationError(
            f"{fname}(): tone= is "
            + (
                f"not the tone of {_named(entry.qudits)}, which the line is"
                if line in carriers
                else f"the tone of {_named(entry.qudits)}, and the line is another"
            )
            + "; the qudits of the quantum system give no frequency_hz, so the broadcast "
            "line's tone is at the carrier of each qudit it reaches and the system does not "
            "state how far that is from any other tone",
            code="control_system",
            fix="drive the line at its own tone, or give the qudits their frequencies",
        )


def _line_arguments(builder: Any, call_args: list[Any]) -> list[Any]:
    """The arguments of a ``get_drive_line`` call as the plain tracer takes
    them: for one or all qudits of a broadcast line, the tracer's value for
    the line's lowest qudit; else *call_args*, of one qudit. Qudits that
    are not those of one broadcast line are an error."""
    from .kernel.ir_builder import IRValue

    several = len(call_args) > 1
    qudits = []
    for arg in call_args:
        if arg.__class__ is not IRValue or arg.vtype != "qref":
            if not several:
                return call_args
            raise CompilationError(
                "get_drive_line() takes a qudit, or all the qudits of a broadcast drive "
                f"line; got {arg!r}",
                code="wrong_argument",
            )
        qudits.append(builder._qref_to_qudit.get(arg.vid))
    if not qudits:
        return call_args
    entry = builder.broadcast_lines.get(qudits[0])
    if several:
        given = _ranges(sorted(q for q in qudits if q is not None and q >= 0))
        if any(q is None or q < 0 for q in qudits):
            raise CompilationError(
                "get_drive_line() of several qudits names the qudits of a broadcast drive "
                "line, which a run-time qudit cannot be; pass the qudits when the kernel "
                "is lowered",
                code="wrong_argument",
            )
        if entry is None:
            raise CompilationError(
                f"get_drive_line() names qudits {given}, and no broadcast drive line of the "
                f"control system reaches qudit {qudits[0]}; several qudits name a line that "
                f"is declared as DriveLine([{', '.join(map(str, sorted(set(qudits))))}], "
                "broadcast=True)",
                code="control_system",
            )
        if len(qudits) != len(entry.qudits) or set(qudits) != set(entry.qudits):
            raise CompilationError(
                f"get_drive_line() names qudits {given} of {_named(entry.qudits)}; name "
                "all its qudits, each once, or any one of them",
                code="control_system",
            )
    elif entry is None:
        return call_args
    if entry.error is not None:
        raise CompilationError(entry.error, code="control_system")
    # The line's qudits are the program's, whichever the kernel holds.
    builder._qref_to_qudit.update(entry.vids)
    builder._line_qudit.update(entry.vids)
    if builder.qudit_count <= entry.qudits[-1]:
        builder.qudit_count = entry.qudits[-1] + 1
    return [entry.values[0]]


# ── Checks ──────────────────────────────────────────────────────────────────


def violations(
    system: ControlSystem,
    lowered: Any,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    unresolved: list[str] | None,
) -> list[str]:
    """``_shared_lines.violations``, and the run-time qudits of *lowered*
    that are bound to a qudit of a broadcast line (``_run_time``)."""
    found = _shared_lines.violations(system, lowered, lines, events, unresolved)
    return [*found, *_run_time(system, lowered)]


def summarized(
    system: ControlSystem,
    lowered: Any,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    unresolved: list[str] | None,
    periods: Any = None,
) -> list[str] | None:
    """``_shared_lines.summarized``, and ``_run_time``."""
    found = _shared_lines.summarized(system, lowered, lines, events, repeats, unresolved, periods)
    return None if found is None else [*found, *_run_time(system, lowered)]


def _run_time(system: ControlSystem, lowered: Any) -> list[str]:
    """A violation for each run-time qudit of *lowered* that takes a drive
    line and is bound to a qudit a broadcast line reaches: its line was
    traced as that qudit's alone, when the kernel compiled without knowing
    the qudit, and the line that reaches the qudit drives all its qudits.
    Checked when the qudits are bound; a program without run-time qudits,
    or with none bound to such a qudit, is not walked."""
    carriers = getattr(lowered, "carriers", ())
    if lowered._role is None or not carriers:
        return []
    reached = _lines(system)
    if not any(carrier in reached for carrier in carriers):
        return []
    from .controller import _walk

    slots: list[int] = []
    found = []
    for entry, slot in zip(_walk(lowered, values=False, slots=slots)[0], slots, strict=False):
        if slot >= 0 and entry["role"] == "drive" and entry["qudit"] in reached:
            found.append(
                f"line {entry['line']} is the drive line of run-time qudit {slot}, bound to "
                f"qudit {entry['qudit']}, which {_named(reached[entry['qudit']][0])} "
                "reaches: a run-time qudit's line is that qudit's alone, and the broadcast "
                "line drives all its qudits; pass the qudit when the kernel is lowered (a "
                "parameter without the Qudit annotation), or bind the run-time qudit to a "
                "qudit with a drive line of its own"
            )
    return found


# ── Exports ─────────────────────────────────────────────────────────────────


class _Entry(dict[str, Any]):
    """A broadcast line's entry in an export's ``lines``: it has the one
    frequency of its tone (``native_frequency_hz``) where the export gives
    a line the frequency of each of its qudits."""

    __slots__ = ("hz",)
    # The frequency the line states for its tone, if it states one.
    hz: float | None

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "native_frequencies_hz":
            key, value = "native_frequency_hz", value[0]
        if key == "native_frequency_hz" and self.hz is not None:
            value = self.hz
        dict.__setitem__(self, key, value)

    def __reduce__(self) -> tuple[Any, ...]:
        return (dict, (dict(self),))


def _entry(entry: dict[str, Any], found: _Line) -> _Entry:
    made = _Entry(entry)
    made.hz = found[3]
    return made


class _Slots(dict[int, int]):
    """``_shared_lines.flat``'s tone slots by handle, with the qudits of
    each handle that is a broadcast line."""

    __slots__ = ("broadcast",)
    broadcast: dict[int, list[int]]


def _on_line(event: dict[str, Any], after: str, qudits: list[int]) -> None:
    _shared_lines._insert(event, after, [("qudits", list(qudits))])


def flat(
    system: ControlSystem, lines: list[dict[str, Any]], events: Sequence[dict[str, Any]]
) -> dict[int, int]:
    """``_shared_lines.flat`` (format /1 *lines* and *events*, in place),
    and each broadcast line written as one: its entry gets, after its
    ``qudit`` (the lowest), its ``qudits`` and ``"broadcast": true``, and
    each event on it its ``qudits``."""
    slots = _Slots(_shared_lines.flat(system, lines, events))
    reached = _lines(system)
    on: dict[int, list[int]] = {}
    for at, entry in enumerate(lines):
        found = reached.get(entry["qudit"]) if entry["role"] == "drive" else None
        if found is not None and entry["qudit"] == found[0][0]:
            made = _entry(entry, found)
            _shared_lines._insert(made, "qudit", [("qudits", list(found[0])), ("broadcast", True)])
            lines[at] = made
            on[entry["line"]] = list(found[0])
    slots.broadcast = on
    if on:
        for event in events:
            qudits = on.get(event.get("line"))  # type: ignore[arg-type]
            if qudits is not None and "qudit" in event:
                _on_line(event, "qudit", qudits)
    return slots


def _emulation(
    lowered: Any, with_events: bool = True, part: Any = None, program: Any = None
) -> dict[str, Any]:
    """``classical.wire._export`` (the emulator export), and each drive on a
    broadcast line given the line's ``qudits`` after its ``qudit``, the
    lowest of them."""
    from .classical.wire import _export

    return _with_qudits_driven(lowered, _export(lowered, with_events, part, program))


def _emulation_part(lowered: Any, controller: int | None) -> dict[str, Any]:
    """``classical.wire._export_part`` (the emulator export of one controller
    of several), with the line's ``qudits`` as ``_emulation`` writes them."""
    from .classical.wire import _export_part

    return _with_qudits_driven(lowered, _export_part(lowered, controller))


def _with_qudits_driven(lowered: Any, exported: dict[str, Any]) -> dict[str, Any]:
    reached = _lines(lowered.options.machine._system)
    for event in exported["events"]:
        if event["kind"] == "drive":
            found = reached.get(event["qudit"])
            if found is not None and event["qudit"] == found[0][0]:
                _on_line(event, "qudit", list(found[0]))
    return exported


def framed(program: dict[str, Any], slots: Mapping[int, int]) -> None:
    """``_shared_lines.framed`` (the ``frame_state`` and ``runtime_tones``
    of format /1 *program*, in place), and each entry on a broadcast line
    given the line's ``qudits`` after its ``line``."""
    _shared_lines.framed(program, slots)
    on: dict[int, list[int]] = getattr(slots, "broadcast", {})
    if not on:
        return
    for key in ("frame_state", "runtime_tones"):
        found = program.get(key)
        if found is None:
            continue
        for item in found.values() if isinstance(found, Mapping) else found:
            qudits = on.get(item.get("line"))
            if qudits is not None:
                _on_line(item, "qudit" if "qudit" in item else "line", qudits)


def structured(program: dict[str, Any], system: ControlSystem) -> dict[str, Any]:
    """``_shared_lines.structured`` (format /2 *program*), and each
    broadcast line written as one: its entry has its ``qudits`` and
    ``"broadcast": true``, and each event, frame and run-time tone on it
    the line's ``qudits`` after its ``line``."""
    program = _shared_lines.structured(program, system)
    reached = _lines(system)
    on: dict[int, list[int]] = {}
    lines = list(program["lines"])
    for at, entry in enumerate(lines):
        found = reached.get(entry["qudits"][0]) if entry["role"] == "drive" else None
        if (
            found is not None
            and "max_concurrent_tones" not in entry
            and entry["qudits"] == [found[0][0]]
        ):
            made = _entry({**entry, "qudits": list(found[0]), "broadcast": True}, found)
            lines[at] = made
            on[entry["line"]] = list(found[0])
    if not on:
        return program

    def written(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for event in items:
            if event["kind"] == "loop":
                event = {**event, "body": written(event["body"])}
            else:
                qudits = on.get(event.get("line"))  # type: ignore[arg-type]
                if qudits is not None and event["kind"] != "sync":
                    event = dict(event)
                    _on_line(event, "line", qudits)
            out.append(event)
        return out

    program = {**program, "lines": lines, "events": written(program["events"])}
    for key in ("frame_state", "runtime_tones"):
        found_frames = program.get(key)
        if found_frames is None:
            continue
        if isinstance(found_frames, Mapping):
            program[key] = {name: _with_qudits(item, on) for name, item in found_frames.items()}
        else:
            program[key] = [_with_qudits(item, on) for item in found_frames]
    return program


def _with_qudits(item: Mapping[str, Any], on: Mapping[int, list[int]]) -> dict[str, Any]:
    out = dict(item)
    qudits = on.get(out.get("line"))  # type: ignore[arg-type]
    if qudits is not None and "qudits" not in out:
        _on_line(out, "line", qudits)
    return out
