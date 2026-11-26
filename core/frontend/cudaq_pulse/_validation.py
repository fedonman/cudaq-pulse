# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``backends.validate`` runs: the export test corpus and its checks.

Imported by ``validate`` alone, so nothing here costs anything until a
backend is validated.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import cudaq_pulse as pulse

from ._device import Qudit
from ._system import Controller, ControlSystem, DriveLine, ReadoutLine
from .backends import Backend, IncompatibleBackendError
from .controller import (
    _FORMATS,
    _KINDS,
    FORMAT,
    ControllerLimitError,
    Event,
    unroll,
)
from .kernel._kernel import Kernel
from .ops import (
    drag,
    drive,
    gaussian,
    get_drive_line,
    get_readout_line,
    readout,
    shift_frequency,
    shift_phase,
    square,
    sync,
    wait,
)
from .quantum_systems import PhysicalQudit

if TYPE_CHECKING:
    from .compiler import ScheduledKernel

# The kinds whose timing and order a backend reports and is checked on: the
# others (waits, syncs, loops) follow from them.
_TIMED = frozenset({Event.PLAY, Event.READOUT, Event.CALL, Event.ANSWER})
_QUDITS = 2


# ── Corpus ───────────────────────────────────────────────────────────────────


def _branch(stretch: int = 1) -> Kernel[[Qudit, Qudit]]:
    """The program with a branch: a pulse on a readout's outcome, which the
    schedule starts once the controller's feedback latency has passed; its
    pulses *stretch* times as long."""
    s = stretch

    @pulse.kernel
    def branches(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        drive(d, gaussian(32 * s, 0.5, 8.0 * s))
        sync(d, ro)
        m = readout(ro, square(80 * s, 0.05))
        sync(d, ro)
        if m:
            drive(d, gaussian(32 * s, 0.5, 8.0 * s))

    return branches


def _build(stretch: int = 1) -> dict[str, Kernel[[Qudit, Qudit]]]:
    """One program per kind of event, named for it, one with a branch, one
    that branches on whether an answer came in time and on the answer, one
    that does so for a function declared with a timeout, one whose
    amplitude and phase come from an answer and one whose frequency does,
    changed ahead of the pulse that takes it. Every pulse and wait lasts
    *stretch* times its cycles."""
    s = stretch

    @pulse.classical(max_latency_ns=200, name="echo")
    def echo(bits: pulse.Bits) -> pulse.Bits:
        return [bits[0]]

    @pulse.classical(max_latency_ns=200, latency_mode="timeout", timeout_ns=300, name="timed_echo")
    def timed_echo(bits: pulse.Bits) -> pulse.Bits:
        return [bits[0]]

    @pulse.classical(max_latency_ns=200, name="gain")
    def gain(bits: pulse.Bits) -> float:
        return 0.1 + 0.2 * bits[0]

    @pulse.classical(max_latency_ns=200, name="retune")
    def retune() -> tuple[float, float]:
        return 2.5e6 / s, 2.0

    @pulse.kernel
    def plays(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        e = get_drive_line(r)
        drive(d, gaussian(32 * s, 0.4, 8.0 * s))
        drive(e, square(16 * s, 0.1))
        shift_phase(d.tone, 0.5)
        drive(d, drag(32 * s, 0.2, 8.0 * s, 0.3))

    @pulse.kernel
    def acquires(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        drive(d, gaussian(32 * s, 0.5, 8.0 * s))
        sync(d, ro)
        readout(ro, square(120 * s, 0.05))

    @pulse.kernel
    def waits(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        drive(d, square(16 * s, 0.1))
        wait(d, 100 * s)
        drive(d, square(16 * s, 0.1))

    @pulse.kernel
    def syncs(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        e = get_drive_line(r)
        ro = get_readout_line(q)
        drive(d, gaussian(48 * s, 0.3, 12.0 * s))
        drive(e, square(16 * s, 0.1))
        sync(d, e, ro)
        drive(e, square(16 * s, 0.1))
        readout(ro, square(80 * s, 0.05))

    @pulse.kernel
    def loops(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(6):
            drive(d, gaussian(16 * s, 0.2, 4.0 * s))
            sync(d, ro)
            readout(ro, square(40 * s, 0.05))
            sync(d, ro)

    @pulse.kernel
    def calls(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(80 * s, 0.05))
        echo.submit([m])
        drive(d, square(16 * s, 0.1))

    @pulse.kernel
    def answers(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(80 * s, 0.05))
        if echo.submit([m]).result()[0]:
            drive(d, gaussian(32 * s, 0.5, 8.0 * s))

    @pulse.kernel
    def answer_timeout(q: Qudit, r: Qudit) -> None:
        # The answer comes in time and its bit is 0: a backend that took
        # one condition for the other would drive the wrong qudit.
        d = get_drive_line(q)
        e = get_drive_line(r)
        a = get_readout_line(q)
        b = get_readout_line(r)
        m = readout(a, square(80 * s, 0.05))
        in_time, bits = echo.submit([m]).result(timeout=300)
        if in_time:
            drive(e, gaussian(32 * s, 0.5, 8.0 * s))
        if bits[0]:
            drive(d, gaussian(32 * s, 0.5, 8.0 * s))
        sync(d, e, a, b)
        readout(a, square(80 * s, 0.05))
        readout(b, square(80 * s, 0.05))

    @pulse.kernel
    def timeout_function(q: Qudit, r: Qudit) -> None:
        # As answer_timeout, with the timeout the function's own.
        d = get_drive_line(q)
        e = get_drive_line(r)
        a = get_readout_line(q)
        b = get_readout_line(r)
        m = readout(a, square(80 * s, 0.05))
        # result() gives (in_time, bits) for a function declared with a
        # timeout, which its type, the function's answer, does not say.
        in_time, bits = timed_echo.submit([m]).result()
        if in_time:
            drive(e, gaussian(32 * s, 0.5, 8.0 * s))
        if bits[0]:  # type: ignore[index]
            drive(d, gaussian(32 * s, 0.5, 8.0 * s))
        sync(d, e, a, b)
        readout(a, square(80 * s, 0.05))
        readout(b, square(80 * s, 0.05))

    @pulse.kernel
    def runtime(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(80 * s, 0.05))
        amp = gain.submit([m]).result()
        shift_phase(d.tone, amp)
        drive(d, square(16 * s, amp))

    @pulse.kernel
    def runtime_frequency(q: Qudit, r: Qudit) -> None:
        # The detuning is made where the first pulse ends, 200 * s cycles
        # before the pulse that takes it (its frame update's start_cycles):
        # over them the frame gains half a turn at 1 GHz, so the two pulses
        # undo each other, where applied at the second pulse they add up.
        d = get_drive_line(q)
        ro = get_readout_line(q)
        detuning, amplitude = retune.submit().result()
        drive(d, gaussian(32 * s, 2.0, 8.0 * s))
        shift_frequency(d.tone, detuning)
        wait(d, 200 * s)
        drive(d, gaussian(32 * s, amplitude, 8.0 * s))
        sync(d, ro)
        readout(ro, square(80 * s, 0.05))

    @pulse.kernel
    def frame_updates(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        drive(d, gaussian(32 * s, 0.4, 8.0 * s))
        shift_phase(d.tone, 0.5)
        shift_frequency(d.tone, 1.0e6)

    @pulse.kernel
    def shared_line(q: Qudit, r: Qudit) -> None:
        a = get_readout_line(q)
        b = get_readout_line(r)
        sync(a, b)
        readout(a, square(80 * s, 0.05))
        readout(b, square(80 * s, 0.05))

    @pulse.kernel
    def shared_drive_line(q: Qudit, r: Qudit) -> None:
        # Both tones of one drive line at once, with the same envelope:
        # two frequencies on the corpus's system, one where its qudits
        # share a frequency.
        d = get_drive_line(q)
        e = get_drive_line(r)
        a = get_readout_line(q)
        b = get_readout_line(r)
        sync(d, e)
        drive(d, gaussian(32 * s, 0.5, 8.0 * s))
        drive(e, gaussian(32 * s, 0.5, 8.0 * s))
        sync(d, e, a, b)
        readout(a, square(80 * s, 0.05))
        readout(b, square(80 * s, 0.05))

    @pulse.kernel
    def several_controllers(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        e = get_drive_line(r)
        a = get_readout_line(q)
        b = get_readout_line(r)
        drive(d, gaussian(32 * s, 0.5, 8.0 * s))
        drive(e, gaussian(32 * s, 0.5, 8.0 * s))
        sync(d, e, a, b)
        readout(a, square(80 * s, 0.05))
        readout(b, square(80 * s, 0.05))

    return {
        Event.PLAY: plays,
        Event.READOUT: acquires,
        Event.WAIT: waits,
        Event.SYNC: syncs,
        Event.LOOP: loops,
        Event.CALL: calls,
        Event.ANSWER: answers,
        Event.FRAME_UPDATE: frame_updates,
        "branch": _branch(s),
        "answer_timeout": answer_timeout,
        "timeout_function": timeout_function,
        "runtime": runtime,
        "runtime_frequency": runtime_frequency,
        "shared_line": shared_line,
        "shared_drive_line_two_tones": shared_drive_line,
        "shared_drive_line_one_tone": shared_drive_line,
        "several_controllers": several_controllers,
    }


_built: dict[int, dict[str, Kernel[[Qudit, Qudit]]]] = {}
# Every pulse and wait of the corpus lasts a whole number of these cycles.
_CORPUS_STEP = 4
# The shortest pulse of the corpus.
_CORPUS_SHORTEST = 16


def _corpus(stretch: int = 1) -> Mapping[str, Kernel[[Qudit, Qudit]]]:
    """The corpus with its pulses *stretch* times as long, built on first use."""
    if stretch not in _built:
        _built[stretch] = _build(stretch)
    return _built[stretch]


def _stretch(system: ControlSystem, program_ghz: float) -> int:
    """The fewest times as long the corpus's pulses and waits must last for
    every controller of *system* to play them: on its lines' timestep, and no
    shorter than its ``min_pulse_cycles``. 1 wherever the corpus plays as
    written."""
    from . import _clocks

    whole, shortest = 1, 1
    for controller in system.controllers:
        plan = _clocks.plan(controller, program_ghz)
        if plan is None:
            grid, ratio = 1, Fraction(1)
        else:
            grid, ratio = plan.controllers[0].grid, plan.controllers[0].ratio
        whole = math.lcm(whole, grid // math.gcd(grid, _CORPUS_STEP))
        # Its minimum, in its own cycles, in program cycles.
        least = -(-controller.min_pulse_cycles * ratio.denominator // ratio.numerator)
        shortest = max(shortest, -(-least // _CORPUS_SHORTEST))
    return -(-shortest // whole) * whole


def _unsupported_event(program: dict[str, Any]) -> dict[str, Any]:
    """*program* with an event of a kind no format has."""
    probe = copy.deepcopy(program)
    last = probe["events"][-1]
    probe["events"].append({**last, "kind": "forward_outcome"})
    return probe


def _newer_format(program: dict[str, Any]) -> dict[str, Any]:
    """*program* in a format this cudaq-pulse does not write."""
    probe = copy.deepcopy(program)
    probe["format"] = f"cudaq-pulse.controller/{len(_FORMATS) + 1}"
    return probe


# Exports a backend must refuse with IncompatibleBackendError, and what each
# is refused for.
_PROBES: Mapping[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str]] = {
    "unsupported_event": (_unsupported_event, "'forward_outcome'"),
    "newer_format": (_newer_format, "a format it does not read"),
}


# ── Report ───────────────────────────────────────────────────────────────────


_TALLIED = {"PASS": "passed", "FAIL": "failed", "REFUSED": "refused", "SKIP": "skipped"}


class _Report:
    """What ``validate`` found: one line per program, ``PASS``, ``FAIL``,
    ``REFUSED`` (the backend raised ``IncompatibleBackendError``) or
    ``SKIP`` (the program exceeds the controller's limits)."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.lines: list[tuple[str, str, str]] = []

    @property
    def ok(self) -> bool:
        """Whether no program failed."""
        return all(status != "FAIL" for status, _, _ in self.lines)

    def _add(self, status: str, name: str, detail: str = "") -> None:
        self.lines.append((status, name, detail))

    def __str__(self) -> str:
        rows = [f"{s} {n}: {d}" if d else f"{s} {n}" for s, n, d in self.lines]
        counts = Counter(s for s, _, _ in self.lines)
        tally = ", ".join(f"{n} {_TALLIED[s]}" for s, n in counts.items()) or "nothing run"
        rows.append(f"backend {self.backend!r}: {tally}")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"<validation of {self.backend!r}: {'ok' if self.ok else 'failed'}>"


# ── Checks ───────────────────────────────────────────────────────────────────


def _control_system(backend: Backend, controller: Controller | None) -> ControlSystem:
    """The backend's control system, with *controller* in place of its one."""
    system = backend.control_system()
    if isinstance(system, Controller):
        system = ControlSystem(controllers=[system])
    if controller is not None:
        system = dataclasses.replace(system, controllers=[controller])
    return system


def _one(system: ControlSystem) -> Controller | None:
    return system.controllers[0] if len(system.controllers) == 1 else None


def _feedline() -> ReadoutLine:
    """One readout line reaching both qudits, playing both tones at once."""
    return ReadoutLine(list(range(_QUDITS)), max_concurrent_tones=_QUDITS)


def _shared_line(system: ControlSystem) -> ControlSystem:
    """*system*'s first controller alone, its readout lines one feedline."""
    controller = dataclasses.replace(system.controllers[0], readout_lines=[_feedline()])
    return ControlSystem(controllers=[controller])


def _shared_drive_line(system: ControlSystem) -> ControlSystem:
    """*system*'s first controller alone, its drive lines one line reaching
    both qudits and playing both tones at once; its readout lines as they
    are (one for each qudit when it lists none)."""
    base = system.controllers[0]
    line = DriveLine(list(range(_QUDITS)), max_concurrent_tones=_QUDITS)
    readouts = list(base.readout_lines) or list(range(_QUDITS))
    controller = dataclasses.replace(base, drive_lines=[line], readout_lines=readouts)
    return ControlSystem(controllers=[controller])


def _several_controllers(system: ControlSystem) -> ControlSystem:
    """Two controllers made from *system*'s first: one drives qudit 0 and
    reads out both qudits on one feedline, the other drives qudit 1; they
    share a reference clock, so a sync across them lines up."""
    base = system.controllers[0]
    clock = base.clock_ghz or 1.0
    return ControlSystem(
        controllers=[
            dataclasses.replace(base, drive_lines=[0], readout_lines=[_feedline()]),
            dataclasses.replace(base, drive_lines=[1], readout_lines=[]),
        ],
        reference_clock_ghz=clock,
    )


# The control systems the programs that need wiring of their own compile for,
# made from the one validated.
_WIRED: Mapping[str, Callable[[ControlSystem], ControlSystem]] = {
    "shared_line": _shared_line,
    "shared_drive_line_two_tones": _shared_drive_line,
    "shared_drive_line_one_tone": _shared_drive_line,
    "several_controllers": _several_controllers,
}


def _device(system: ControlSystem, spacing_hz: float = 1.5e8) -> Any:
    """The corpus's quantum system: its qudits *spacing_hz* apart."""
    limits = _one(system)
    clock = limits.clock_ghz if limits is not None and limits.clock_ghz else 1.0
    return pulse.quantum_system(
        name="validation" if spacing_hz else "validation-one-frequency",
        qudits={
            i: PhysicalQudit(i, 5.0e9 + spacing_hz * i, 50.0, 30.0, anharmonicity_hz=-2.5e8)
            for i in range(_QUDITS)
        },
        clock_ghz=clock,
    )


# The programs that run on a quantum system of their own: both qudits at one
# frequency, so that the two tones of the shared drive line are one.
_ONE_FREQUENCY = frozenset({"shared_drive_line_one_tone"})


def _formats(backend: Backend) -> list[str]:
    read = getattr(backend, "formats", None)
    return [f for f in _FORMATS if read is None and f == FORMAT or read and f in read]


def _program_lines(program: Mapping[str, Any], flat: Mapping[str, Any]) -> dict[int, int]:
    """By line of *flat*, ``unroll(program)``, the line of *program* it is:
    format /2 lists a shared line once, where format /1 has a line for each
    of its qudits. Empty for a format /1 *program*."""
    if program["format"] == FORMAT:
        return {}
    physical = {(line["role"], tuple(line["qudits"])): line["line"] for line in program["lines"]}
    return {
        line["line"]: physical[(line["role"], tuple(line.get("qudits", (line["qudit"],))))]
        for line in flat["lines"]
    }


def _place(kind: Any, line: Any) -> Any:
    """Where an event's order is checked: a call or answer on its link, whose
    number may also be a line's."""
    return ("link", line) if kind in (Event.CALL, Event.ANSWER) else line


def _named(place: Any) -> str:
    return f"link {place[1]}" if isinstance(place, tuple) else f"line {place}"


def _expected(program: Mapping[str, Any]) -> tuple[int, dict[Any, list[tuple[Any, ...]]]]:
    """The programs *program* is for, and its timed events by line, each
    ``(kind, start_cycles, duration_cycles)`` in program order."""
    flat = unroll(program)
    line_of = _program_lines(program, flat)
    by_line: dict[Any, list[tuple[Any, ...]]] = {}
    for event in flat["events"]:
        kind = _KINDS.get(event["kind"], event["kind"])
        if kind in _TIMED and event["start_cycles"] is not None:
            line = event["line"]
            if kind in (Event.PLAY, Event.READOUT):
                line = line_of.get(line, line)
            by_line.setdefault(_place(kind, line), []).append(
                (kind, event["start_cycles"], event["duration_cycles"])
            )
    return max(1, len(program.get("programs") or ())), by_line


def _emitted(programs: Sequence[Iterable[Mapping[str, Any]]]) -> dict[Any, list[tuple[Any, ...]]]:
    by_line: dict[Any, list[tuple[Any, ...]]] = {}
    for events in programs:
        for event in events:
            kind = _KINDS.get(str(event["kind"]), str(event["kind"]))
            if kind in _TIMED:
                by_line.setdefault(_place(kind, event["line"]), []).append(
                    (kind, event["start_cycles"], event.get("duration_cycles"))
                )
    return by_line


def _timing(
    expected: Mapping[Any, list[tuple[Any, ...]]], emitted: Mapping[Any, list[tuple[Any, ...]]]
) -> str | None:
    """The first difference in timing or order, or None."""
    for line in sorted(set(expected) | set(emitted), key=repr):
        want, got = expected.get(line, []), emitted.get(line, [])
        for k, (a, b) in enumerate(zip(want, got, strict=False)):
            same = a[0] == b[0] and a[1] == b[1] and (b[2] is None or a[2] == b[2])
            if not same:
                return (
                    f"{_named(line)}: event {k} is a {b[0]} at {b[1]} cycles; "
                    f"the controller program has a {a[0]} at {a[1]}"
                )
        if len(want) != len(got):
            return (
                f"{_named(line)}: emitted {len(got)} events, the controller program has {len(want)}"
            )
    return None


def _limits(
    limits: Controller | None,
    emitted: Mapping[Any, list[tuple[Any, ...]]],
    toned: frozenset[Any] = frozenset(),
) -> str | None:
    """The first declared limit the emitted events exceed, or None. The
    spacing of pulses is per tone, so it is not checked on the *toned*
    lines, which play several at once."""
    if limits is None:
        return None
    played = sum(
        1 for events in emitted.values() for e in events if e[0] in (Event.PLAY, Event.READOUT)
    )
    # A controller that runs loops holds a rolled body once, and emits it
    # once per iteration.
    if limits.max_pulses is not None and not limits._runs_loops and played > limits.max_pulses:
        return f"emitted {played} pulses; the controller plays at most {limits.max_pulses}"
    for line, events in emitted.items():
        if line in toned:
            continue
        starts = sorted(e[1] for e in events if e[0] in (Event.PLAY, Event.READOUT))
        for a, b in zip(starts, starts[1:], strict=False):
            if b - a < limits.instruction_cycles:
                return (
                    f"line {line}: two pulses start {b - a} cycles apart; the controller "
                    f"issues one per {limits.instruction_cycles}"
                )
    return None


def _simulated(lowered: ScheduledKernel) -> list[float]:
    """The probability of outcome 1 of each measurement, in end-time order."""
    result = pulse.simulate(lowered)
    p = [0.0] * len(result.measurements)
    for branch in result.branches:
        for k, level in enumerate(branch.levels):
            if level == 1:
                p[k] += branch.probability
    return p


def _emulation(
    backend: Backend, artifact: Any, lowered: ScheduledKernel, shots_count: int, tolerance: float
) -> str | None:
    """How the backend's emulator differs from the simulator, or None."""
    rows = [list(row) for row in backend.emulate(artifact, shots_count=shots_count)]  # type: ignore[attr-defined]
    expected = _simulated(lowered)
    if len(rows) != shots_count:
        return f"the emulator returned {len(rows)} shots, expected {shots_count}"
    if any(len(row) != len(expected) for row in rows):
        return f"the emulator reports a shot without {len(expected)} measurements"
    for k, p in enumerate(expected):
        mean = sum(1 for row in rows if row[k] == 1) / shots_count
        # The tolerance, widened by three standard errors of the shots.
        allowed = tolerance + 3.0 * math.sqrt(p * (1.0 - p) / shots_count)
        if abs(mean - p) > allowed:
            return (
                f"measurement {k}: the emulator gives 1 in {mean:.3f} of shots, "
                f"the simulator {p:.3f} (tolerance {allowed:.3f})"
            )
    return None


def _emit(
    report: _Report,
    backend: Backend,
    name: str,
    program: dict[str, Any],
    lowered: ScheduledKernel,
    limits: Controller | None,
    shots_count: int,
    tolerance: float,
    emulated: bool = True,
) -> bool:
    """Emit one program and check it; whether the backend emitted it.
    Without *emulated* (one controller's part of a program), its emulation
    is not compared with the simulator's."""
    try:
        artifact = backend.emit(program)
    except IncompatibleBackendError as e:
        report._add("REFUSED", name, str(e))
        return False
    except Exception as e:
        report._add(
            "FAIL",
            name,
            f"emit() raised {type(e).__name__}: {e}; a program it cannot emit is refused "
            "with IncompatibleBackendError",
        )
        return False
    decode = getattr(backend, "events", None)
    if decode is not None:
        programs = list(decode(artifact))
        count, expected = _expected(program)
        if len(programs) != count:
            s = "" if len(programs) == 1 else "s"
            report._add("FAIL", name, f"emitted {len(programs)} program{s}, expected {count}")
            return True
        emitted = _emitted(programs)
        # Format /2 lists a shared line once; format /1 has a line per tone.
        toned = frozenset(
            line["line"]
            for line in program["lines"]
            if program["format"] != FORMAT and line.get("max_concurrent_tones", 1) > 1
        )
        problem = _timing(expected, emitted) or _limits(limits, emitted, toned)
        if problem is not None:
            report._add("FAIL", name, problem)
            return True
    if emulated and getattr(backend, "emulate", None) is not None:
        problem = _emulation(backend, artifact, lowered, shots_count, tolerance)
        if problem is not None:
            report._add("FAIL", name, problem)
            return True
    report._add("PASS", name)
    return True


def validate(
    backend: Backend,
    controller: Controller | None = None,
    shots_count: int = 1000,
    tolerance: float = 0.05,
) -> _Report:
    """See ``cudaq_pulse.backends.validate``."""
    report = _Report(backend.name)
    system = _control_system(backend, controller)
    dev = _device(system)
    stretch = _stretch(system, dev.clock_ghz)
    formats = _formats(backend)
    tagged = len(formats) > 1
    emitted = 0
    first: dict[str, Any] | None = None
    for kind, kernel in _corpus(stretch).items():
        wired = _WIRED.get(kind)
        on = system if wired is None else wired(system)
        parts = len(on.controllers) > 1
        if parts and FORMAT not in formats:
            # Format /2 is written for one controller.
            continue
        on_dev = _device(system, spacing_hz=0.0) if kind in _ONE_FREQUENCY else dev
        try:
            compiled = pulse.compile(kernel, quantum_system=on_dev, control_system=on)
            lowered = compiled.lower(*on_dev.qudits[:_QUDITS])
        except ControllerLimitError as e:
            report._add("SKIP", str(kind), f"exceeds the controller's limits: {e.violations[0]}")
            continue
        if parts:
            # One controller's part of the program each, in format /1.
            for k, controller in enumerate(on.controllers):
                name = f"{kind}[{k}] (/1)" if tagged else f"{kind}[{k}]"
                program = lowered.export_to_controller(controller=k)
                emitted += _emit(
                    report,
                    backend,
                    name,
                    program,
                    lowered,
                    controller,
                    shots_count,
                    tolerance,
                    emulated=False,
                )
            continue
        for fmt in formats:
            version = 1 if fmt == FORMAT else 2
            name = f"{kind} (/{version})" if tagged else str(kind)
            program = lowered.export_to_controller(version=version)
            first = first or program
            emitted += _emit(
                report, backend, name, program, lowered, _one(on), shots_count, tolerance
            )
    if first is not None:
        for name, (make, what) in _PROBES.items():
            try:
                backend.emit(make(first))
            except IncompatibleBackendError:
                report._add("PASS", name)
            except Exception as e:
                report._add(
                    "FAIL",
                    name,
                    f"emit() raised {type(e).__name__} for {what}; it should raise "
                    "IncompatibleBackendError",
                )
            else:
                report._add(
                    "FAIL",
                    name,
                    f"emit() accepted {what}; it should raise IncompatibleBackendError",
                )
    if not emitted and any(s == "REFUSED" for s, _, _ in report.lines):
        report._add("FAIL", "corpus", "the backend refused every program")
    return report


def opted_in(
    descriptions: Iterable[tuple[str, Mapping[str, Any], Controller]],
    get: Callable[[str], Backend],
    shots_count: int = 1000,
    tolerance: float = 0.05,
) -> list[tuple[str, _Report]]:
    """The CI hook: for each controller description, its name, its parsed
    table and the controller it describes, the report of validating the
    backend its table opts in with ``backend = "..."``; descriptions
    without one are left out."""
    out = []
    for name, table, controller in descriptions:
        chosen = table.get("backend")
        if chosen:
            out.append((name, validate(get(chosen), controller, shots_count, tolerance)))
    return out
