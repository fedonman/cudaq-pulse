# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering a traced kernel to a scheduled MLIR module, and its inspection.

``cudaq_pulse.compile(k)`` returns a callable ``CompiledKernel`` (see
``kernel/_kernel.py``); this module holds what it is built from: the options
that pick passes and a schedule, the native build of one specialization, and
``ScheduledKernel``, the read-only view of one compiled module.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

import cudaq_pulse.classical as classical

from ._errors import InvalidTypeError, InvalidValueError, _closest
from ._native import PackedValueError, PulseModuleBuilder
from ._units import Time, _time
from .kernel._bytecode_normalize import UnsupportedInterpreterError
from .kernel._schedule_blocks import check_root_schedule, root_schedule
from .kernel._unroll import DEFAULT_UNROLL_THRESHOLD, resolve_unroll_threshold
from .kernel.ir_builder import CompilationError, SymbolicValueError

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

    from . import _clocks
    from ._classical_report import ClassicalCall as ClassicalCall
    from ._device import QuantumSystem
    from ._exposure import Exposure as Exposure
    from ._exposure import QuditExposure as QuditExposure
    from ._idle_windows import IdleWindow as IdleWindow
    from ._idle_windows import IdleWindows as IdleWindows
    from ._programs import _Role
    from ._system import Controller, ControlSystem
    from ._waits import _WaitLengths
    from .classical import Link
    from .controller import UsageReport, _Parts
    from .kernel._kernel import CompiledKernel, Kernel

    type _Kernel = ScheduledKernel

__all__ = [
    "DEFAULT_PASSES",
    "CompileMetrics",
    "IRLevel",
    "ScheduledKernel",
    "Pass",
    "ScheduledEvent",
    "UnsupportedInterpreterError",
    "compile",
]


class _Module(ModuleType):
    # ClassicalCall is built with the first report: building a dataclass
    # costs more than an import of this module should. A property rather
    # than a module __getattr__, which every ``from .compiler import`` would
    # call to look for ``__path__``.
    @property
    def ClassicalCall(self) -> type[ClassicalCall]:  # noqa: N802
        from ._classical_report import ClassicalCall

        return ClassicalCall

    # The types of ScheduledKernel.idle_windows, loaded with the first
    # program whose idle windows are replaced.
    @property
    def IdleWindow(self) -> type[IdleWindow]:  # noqa: N802
        from ._idle_windows import IdleWindow

        return IdleWindow

    @property
    def IdleWindows(self) -> type[IdleWindows]:  # noqa: N802
        from ._idle_windows import IdleWindows

        return IdleWindows

    # The types of ScheduledKernel.exposure, loaded with the first one read.
    @property
    def Exposure(self) -> type[Exposure]:  # noqa: N802
        from ._exposure import Exposure

        return Exposure

    @property
    def QuditExposure(self) -> type[QuditExposure]:  # noqa: N802
        from ._exposure import QuditExposure

        return QuditExposure


sys.modules[__name__].__class__ = _Module


_VALID_SCHEDULES = frozenset({"asap", "alap", "rcp", "alap_rcp"})


class Pass(StrEnum):
    """An optimization pass of ``pulse.compile(k, passes=)``.

    Each member is a string, and the plain string (``"fold_phases"``) is
    accepted wherever a member is. Passes run in the order given, any subset
    in any order; the pattern passes (all but ``VERIFY`` and
    ``HOIST_LOOP_INVARIANTS``) that follow one another run as one rewrite of
    the program, and each of them also removes the operations its rewrites
    leave unused.
    """

    #: Check waveform durations and that each line is used once at each step.
    VERIFY = "verify"
    #: Compute constant arithmetic at compile time and merge equal constants.
    FOLD_CONSTANTS = "fold_constants"
    #: Remove operations whose results nothing uses, unreachable blocks and
    #: unused block arguments.
    REMOVE_UNUSED = "remove_unused"
    #: Join constant waits in a row on one line into one wait.
    MERGE_WAITS = "merge_waits"
    #: Remove syncs that order nothing, inline schedule blocks that take no
    #: time, branch on ``is_set``, and inline or remove loops that run once
    #: or never.
    SIMPLIFY_CONTROL_FLOW = "simplify_control_flow"
    #: Fold constant frame changes (``shift_phase``) into the next drive.
    FOLD_PHASES = "fold_phases"
    #: Merge back-to-back square-pulse drives of the same amplitude into one.
    FUSE_PULSES = "fuse_pulses"
    #: Move operations that compute the same value in every iteration out of
    #: loops.
    HOIST_LOOP_INVARIANTS = "hoist_loop_invariants"


# The cudaq-pulse-opt pass each member runs.
_PASS_MAP = {
    Pass.VERIFY: "pulse-verify",
    Pass.FOLD_CONSTANTS: "pulse-fold-constants",
    Pass.REMOVE_UNUSED: "pulse-remove-unused",
    Pass.MERGE_WAITS: "pulse-merge-waits",
    Pass.SIMPLIFY_CONTROL_FLOW: "pulse-simplify-control-flow",
    Pass.FOLD_PHASES: "pulse-fold-phases",
    Pass.FUSE_PULSES: "pulse-fuse-pulses",
    Pass.HOIST_LOOP_INVARIANTS: "pulse-hoist-loop-invariants",
}

DEFAULT_PASSES: tuple[Pass, ...] = (
    Pass.VERIFY,
    Pass.FOLD_CONSTANTS,
    Pass.REMOVE_UNUSED,
    Pass.FOLD_PHASES,
    Pass.FUSE_PULSES,
)


# *passes* as members; an InvalidValueError names any that is not one.
def _passes(passes: Sequence[str]) -> tuple[Pass, ...]:
    if isinstance(passes, str):
        raise InvalidValueError(
            f"passes must be a sequence of pass names, not the string {passes!r}"
        )
    members = []
    names = [p.value for p in Pass]
    for name in passes:
        try:
            members.append(Pass(name))
        except ValueError:
            close = _closest(str(name), names)
            hint = "" if close is None else f" Did you mean {close!r}?"
            raise InvalidValueError(
                f"Unknown pass {name!r}.{hint} The passes are: {', '.join(names)}"
            ) from None
    return tuple(members)


class IRLevel(StrEnum):
    """A level of the program's IR, for ``mlir(level)``: each lowers the one
    before it. The plain string (``"qop"``) is accepted wherever a member is.
    """

    #: The scheduled Pulse dialect.
    PULSE = "pulse"
    #: Quantum operators: the program's Hamiltonian and dissipators.
    QOP = "qop"
    #: cuDensityMat operators and evolution.
    CUDM = "cudm"
    #: LLVM dialect, ready to compile.
    LLVM = "llvm"


_LEVELS = tuple(IRLevel)

# The parameters a measurement can select while the program runs, in the
# order lowering asks about them, and those any target takes that way.
_RUNTIME_KINDS = ("amplitude", "phase", "frequency")
# ScheduledKernel.schedule is a list up to this many events.
_LISTED_EVENTS = 10_000
_SELECTED_AT_RUNTIME: frozenset[str] = frozenset()


# Whether a program compiled for *target* can take a value of
# *parameter_kind* that a measurement selects while it runs.
#
# *target* is the quantum system the program is built for. *parameter_kind* is
# ``"amplitude"`` (of a waveform, or the factor of ``wf_scale``),
# ``"phase"`` (of ``shift_phase`` and ``set_phase``) or ``"frequency"``
# (of ``shift_frequency`` and ``set_frequency``). Durations, waveform
# shapes and anything else that moves ops in time are never asked about.
#
# Where this holds, the measurement selects the value once, every op that
# uses it reads that run-time value, and the schedule starts those ops
# once the measurement is in. Everywhere else, each op that uses the value
# runs in both arms of an if on the measurement, with that arm's value.
# Simulation gives the same result either way. No target takes any kind
# at run time: ``export_to_controller`` gives a kept value as ``null`` with
# its expression beside it (the select on the measurement), and
# ``export_to_emulator`` carries no parameters.
def _can_select_at_runtime(target: QuantumSystem, parameter_kind: str) -> bool:
    return parameter_kind in _SELECTED_AT_RUNTIME


class _Links(dict[int, "Link"]):
    """The links of a control system: a read-only, hashable mapping by id,
    which keeps the names the model gave them (``names``), so that a model
    made from another's links (``dataclasses.replace``) keeps them too."""

    names: Mapping[str, int] = MappingProxyType({})

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted(self.items(), key=lambda item: item[0])))

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import rebuilt_links

        return (rebuilt_links, (_Links, dict(self), dict(self.names)))

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("the links are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly  # type: ignore[assignment]


# The drives and readouts the rcp and alap_rcp schedules play at once when
# the control system gives no limit.
_RCP_DRIVES = 4
_RCP_READOUTS = 2

# Defaults of the mapping fields, read-only: __post_init__ replaces whatever
# a field is given.
_NO_DELAYS: Mapping[tuple[str, int], float] = MappingProxyType({})
_NO_LINKS: Mapping[int | str, Link] = MappingProxyType({})


@dataclass(frozen=True)
class _MachineModel:
    """What compiling for a ``ControlSystem`` (``ControlSystem._machine``)
    needs of it: the lanes and switch penalty of the ``rcp`` and
    ``alap_rcp`` schedules, and the readout latency of every schedule; the
    links by id, a named link at
    the lowest id no other link has, in the order given (a link id a
    program uses that is not listed gets ``Link()``; an unknown name is an
    error); the controller a program is for, when it is checked as one
    controller; the reference clock and line delays; and the system when
    its wiring, capabilities or processors are checked too. An instance
    made for ``compile(control_system=)`` given one ``Controller`` has
    ``_bare`` set, not a field: one more would cost every import.
    """

    max_concurrent_drives: int | None = None
    max_concurrent_readouts: int | None = None
    readout_latency_cycles: int = 0
    line_switch_penalty_cycles: int = 0
    links: Mapping[int | str, Link] = _NO_LINKS
    controller: Controller | None = None
    reference_clock_ghz: float | None = None
    # (role, carrier) -> the delay in ns of each delayed line of _system.
    _line_delays: Mapping[tuple[str, int], float] = field(default=_NO_DELAYS, repr=False)
    link_names: Mapping[str, int] = field(init=False, repr=False, compare=False)
    # The ControlSystem this model was made from, whose wiring and
    # capabilities every program is checked against.
    _system: ControlSystem | None = field(default=None, repr=False)
    _plain: bool = field(init=False, repr=False, compare=False)
    if TYPE_CHECKING:
        # The drives and readouts the rcp schedules play at once: the limits
        # given, else _RCP_DRIVES and _RCP_READOUTS. Not fields, as _bare.
        _drive_lanes: int = field(init=False)
        _readout_lanes: int = field(init=False)

        # The emulator export of a program on one controller or none
        # (classical.wire sets it when it loads).
        _emulation: ClassVar[Any]
        _emulation_part: ClassVar[Any]

        def _clocked_builder(self, builder: Any, target: QuantumSystem) -> Any: ...

        def _clocked_module(
            self, module: Any, target: QuantumSystem, clocks: Any, max_duration_cycles: int | None
        ) -> Mapping[tuple[str, int], float] | None: ...

    def __post_init__(self) -> None:
        drives, readouts = self.max_concurrent_drives, self.max_concurrent_readouts
        object.__setattr__(self, "_drive_lanes", _RCP_DRIVES if drives is None else drives)
        object.__setattr__(self, "_readout_lanes", _RCP_READOUTS if readouts is None else readouts)
        if self.controller is not None:
            from ._system import Controller

            if not isinstance(self.controller, Controller):
                raise TypeError(f"the controller must be a Controller, got {self.controller!r}")
        if self.reference_clock_ghz is not None and not self.reference_clock_ghz > 0:
            raise InvalidValueError(
                "ControlSystem.reference_clock_ghz must be positive, "
                f"got {self.reference_clock_ghz}"
            )
        # Nothing of clocks to do: no controller, control system, reference
        # clock or delays.
        object.__setattr__(
            self,
            "_plain",
            self.controller is None
            and self._system is None
            and self.reference_clock_ghz is None
            and not self._line_delays,
        )
        if isinstance(self.links, _Links):
            # Another model's links, as dataclasses.replace passes them: by
            # id, with their names beside them.
            named = set(self.links.names.values())
            given: dict[int | str, Link] = {
                key: value for key, value in self.links.items() if key not in named
            }
            for name, link_id in sorted(self.links.names.items(), key=lambda item: item[1]):
                given[name] = self.links[link_id]
        else:
            given = dict(self.links)
        links = _Links()
        names: list[str] = []
        for key, value in given.items():
            if isinstance(key, str):
                if not key.isidentifier():
                    raise ValueError(f"ControlSystem.links names are identifiers, got {key!r}")
                names.append(key)
            elif isinstance(key, bool) or not isinstance(key, int) or key < 0:
                raise ValueError(
                    f"ControlSystem.links keys are names or link ids >= 0, got {key!r}"
                )
            if not isinstance(value, classical.Link):
                raise TypeError(f"ControlSystem.links[{key!r}] must be a Link, got {value!r}")
            if isinstance(key, int):
                dict.__setitem__(links, key, value)
        by_name: dict[str, int] = {}
        next_id = 0
        for name in names:
            while next_id in links:
                next_id += 1
            dict.__setitem__(links, next_id, given[name])
            by_name[name] = next_id
        links.names = MappingProxyType(by_name)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "link_names", links.names)
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.max_concurrent_drives,
                    self.max_concurrent_readouts,
                    self.readout_latency_cycles,
                    self.line_switch_penalty_cycles,
                    self.links,
                    tuple(sorted(by_name.items())),
                    self.controller,
                    self.reference_clock_ghz,
                    tuple(self._line_delays.items()),
                    self._system,
                )
            ),
        )

    def __hash__(self) -> int:
        return self._hash  # type: ignore[no-any-return,attr-defined]

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MachineModel):
            return NotImplemented
        return (
            self.max_concurrent_drives,
            self.max_concurrent_readouts,
            self.readout_latency_cycles,
            self.line_switch_penalty_cycles,
            dict(self.links),
            dict(self.link_names),
            self.controller,
            self.reference_clock_ghz,
            dict(self._line_delays),
            self._system,
        ) == (
            other.max_concurrent_drives,
            other.max_concurrent_readouts,
            other.readout_latency_cycles,
            other.line_switch_penalty_cycles,
            dict(other.links),
            dict(other.link_names),
            other.controller,
            other.reference_clock_ghz,
            dict(other._line_delays),
            other._system,
        )

    def link_id(self, link: int | str) -> int:
        """The id of *link*, a link name of this model or a link id."""
        if isinstance(link, str):
            found = self.link_names.get(link)
            if found is None:
                if getattr(self, "_bare", False):
                    raise CompilationError(
                        f"no link named {link!r}: compile() was given one Controller, which has"
                        " no links; give compile() a pulse.ControlSystem that describes the whole"
                        f" control system, with its links (links={{{link!r}: pulse.Link(...)}})",
                        code="control_system",
                    )
                known = ", ".join(repr(n) for n in self.link_names) or "none"
                raise CompilationError(
                    f"no link named {link!r} in the control system (its links: {known}); "
                    "add it to ControlSystem(links={...})",
                    code="control_system",
                )
            return found
        return link

    def link_name(self, link_id: int) -> str | None:
        """The name of the link with id *link_id*, if it has one."""
        for name, found in self.link_names.items():
            if found == link_id:
                return name
        return None

    def link(self, link: int | str) -> Link:
        """The ``Link`` of *link* (a name or an id; ``Link()`` if unlisted)."""
        found = self.links.get(self.link_id(link))
        return found if found is not None else _unlisted_link()

    def links_attribute(self, used: Sequence[int]) -> str:
        """The module attribute ``classical.links`` for the links in *used*."""
        entries = ", ".join(
            f'"{link}" = {(self.links.get(link) or _unlisted_link()).attribute()}'
            for link in sorted(set(used))
        )
        return f"{{classical.links = {{{entries}}}}}"


_DEFAULT_MACHINE = _MachineModel()
_UNLISTED_LINK: list[classical.Link] = []


# ``Link()``, the link of a program's link its machine does not list
# (one, made when first needed: links are immutable).
def _unlisted_link() -> classical.Link:
    if not _UNLISTED_LINK:
        _UNLISTED_LINK.append(classical.Link())
    return _UNLISTED_LINK[0]


@dataclass(frozen=True)
class ScheduledEvent:
    """One timed operation of a compiled kernel.

    ``line`` numbers the kernel's drive and readout lines in allocation order.
    ``start_cycles`` and ``duration_cycles`` are ``None`` until the kernel is
    scheduled (a kernel with float parameters is scheduled when it is called
    with their values). ``start`` and ``end`` are the scheduled times in
    clock cycles, and raise ``ValueError`` before then.

    A call to a classical function adds a ``"submit"`` event, while the
    request holds its link, and a zero-length ``"answer"`` event, when the
    answer is taken. For those, ``line`` is the link id and ``qudit`` is -1.
    """

    kind: str
    line: int
    qudit: int
    start_cycles: int | None
    duration_cycles: int | None

    @property
    def start(self) -> int:
        """The clock cycle the event starts at; ``ValueError`` if not scheduled."""
        if self.start_cycles is None:
            raise ValueError(f"{self.kind} on line {self.line} is not scheduled")
        return self.start_cycles

    @property
    def end(self) -> int:
        """The clock cycle the event ends at; ``ValueError`` if not scheduled."""
        if self.duration_cycles is None:
            raise ValueError(f"{self.kind} on line {self.line} is not scheduled")
        return self.start + self.duration_cycles

    @property
    def link(self) -> int | None:
        """The link of a classical call's event, else ``None``."""
        return self.line if self.kind in ("submit", "answer") else None


@dataclass(frozen=True)
class Segment:
    """One segment of a compiled kernel, from ``ScheduledKernel.segments()``.

    A variable wait (its length computed from a classical answer) or a
    blocking classical wait is a barrier for the lines that depend on it:
    those it waits on or whose pulses use its answer, and every line tied to
    them by a ``sync``, a shared tone, a branch or a kernel with its own
    ``schedule=``. They sync before it, and what follows on them shifts with
    its run-time length; other lines keep their time. Segment 0 starts the
    program; segment k starts when the k-th barrier, in program order, ends.
    ``start_cycles`` is the segment's start in the static schedule, which places
    every barrier at its lowest length. At run time the segment starts
    ``shift_cycles`` later, with ``min_shift_cycles <= shift_cycles <=
    max_shift_cycles`` (``None``: unbounded); the bound adds up the barriers
    before it on the same lines. ``opened_by`` is ``"start"``, ``"wait"`` (a
    variable wait on line ``line``) or ``"answer"`` (a blocking wait on link
    ``line``).
    """

    index: int
    opened_by: str
    line: int
    start_cycles: int
    min_shift_cycles: int
    max_shift_cycles: int | None


@dataclass()
class CompileMetrics:
    """Per-stage timing of one compilation (milliseconds) and its size."""

    #: Milliseconds spent tracing the kernel's bytecode into the packed
    #: operation buffer (stage 1).
    trace_ms: float = 0.0
    #: Milliseconds spent sending the packed buffer to C++, building the MLIR
    #: module from it and setting the module's link attributes and clocks
    #: (stage 2, "FFI to C++").
    ffi_ms: float = 0.0
    #: Milliseconds spent lowering runtime selects and running the compiler
    #: passes on the module (stage 3).
    passes_ms: float = 0.0
    #: Milliseconds spent scheduling the module (stage 4). A module that is only
    #: scheduled again for other line delays reports just this stage.
    schedule_ms: float = 0.0
    #: Milliseconds for the sum of the four stages: ``trace_ms``, ``ffi_ms``,
    #: ``passes_ms`` and ``schedule_ms``.
    total_ms: float = 0.0
    #: The number of pulse-program operations the kernel recorded, not counting
    #: qudit binding or region yields.
    op_count: int = 0
    #: The name of the quantum system the kernel was compiled for.
    quantum_system: str = ""


@dataclass(frozen=True)
class CompileOptions:
    """Passes, schedule policy, control system, unroll threshold and unit
    warnings of a compilation. ``unroll_threshold`` is a non-negative
    ``int``, 8 by default; 0 never unrolls (see ``cudaq_pulse.compile``)."""

    schedule: str = "alap"
    passes: tuple[Pass, ...] = DEFAULT_PASSES
    machine: _MachineModel = _DEFAULT_MACHINE
    flatten_schedules: bool = False
    unroll_threshold: int = DEFAULT_UNROLL_THRESHOLD
    # Not a field, so making and comparing options costs no more: a
    # compilation with unit warnings has options of a subclass
    # (kernel/_unit_warnings.py).
    unit_warnings = False

    def tracer(self, builder: Any, target: QuantumSystem) -> Any:
        """The tracer of a compilation whose ``unit_warnings`` is set: the
        one that warns of bare numbers, or (``_idle._SiteOptions``) the one
        that also keeps idle replacement's record of the operations."""
        from .kernel._unit_warnings import unit_checked

        return unit_checked(builder, target)

    @staticmethod
    def make(
        schedule: str = "alap",
        passes: Sequence[str] | None = None,
        control_system: ControlSystem | Controller | None = None,
        flatten_schedules: bool = False,
        unroll_threshold: int | None = None,
        unit_warnings: bool = False,
    ) -> CompileOptions:
        if schedule not in _VALID_SCHEDULES:
            close = _closest(str(schedule).lower(), _VALID_SCHEDULES)
            hint = f" Did you mean {close!r}?" if close else ""
            raise InvalidValueError(
                f"Unknown schedule policy {schedule!r}.{hint} "
                f"Choose from: {', '.join(sorted(_VALID_SCHEDULES))}"
            )
        passes = DEFAULT_PASSES if passes is None else _passes(passes)
        if control_system is None:
            machine = _DEFAULT_MACHINE
        else:
            try:
                machine = control_system._machine
            except AttributeError:
                raise InvalidTypeError(
                    "compile(control_system=) takes a pulse.ControlSystem or a "
                    f"pulse.Controller, got {control_system!r}"
                ) from None
            drives, readouts = machine.max_concurrent_drives, machine.max_concurrent_readouts
            if drives is not None and drives <= 0:
                raise ValueError("ControlSystem.max_concurrent_drives must be positive")
            if readouts is not None and readouts <= 0:
                raise ValueError("ControlSystem.max_concurrent_readouts must be positive")
            if machine.readout_latency_cycles < 0:
                raise ValueError("the readout latency cannot be negative")
            if machine.line_switch_penalty_cycles < 0:
                raise ValueError("ControlSystem.line_switch_penalty_cycles cannot be negative")
        options = CompileOptions(
            schedule,
            passes,
            machine,
            bool(flatten_schedules),
            resolve_unroll_threshold(unroll_threshold),
        )
        if unit_warnings:
            from .kernel._unit_warnings import with_unit_warnings

            return with_unit_warnings(options)
        return options

    def schedule_args(self) -> tuple[str, int, int, int, int]:
        m = self.machine
        return (
            self.schedule,
            m._drive_lanes,
            m._readout_lanes,
            int(m.readout_latency_cycles),
            int(m.line_switch_penalty_cycles),
        )


# The error for ``bind(values)`` with the wrong number of *values*.
def _bind_count_error(parameters: tuple[str, ...], values: Any) -> InvalidTypeError:
    try:
        count = len(values)
    except TypeError:
        return InvalidTypeError(
            f"bind() takes a list or tuple of values, one per float parameter, "
            f"got {type(values).__name__}"
        )
    got = f"{count} value" if count == 1 else f"{count} values"
    if not parameters:
        return InvalidTypeError(
            f"bind() got {got} for a module with no float parameters; a module of one "
            "call has its values already"
        )
    names = ", ".join(parameters)
    if count < len(parameters):
        missing = ", ".join(repr(p) for p in parameters[count:])
        detail = f"missing the value of {missing}"
    else:
        detail = f"{count - len(parameters)} more than it has parameters"
    return InvalidTypeError(
        f"bind() got {got} for {len(parameters)} float parameters ({names}); {detail}"
    )


# The types of the values bind() has checked and taken as real numbers. What
# the check decides about a value other than a NaN, it decides by the value's
# type, so a value of one of these types needs no more than float().
_REAL_TYPES: set[type] = {float, int}


# *values* as floats, one per parameter, or the error for their count or
# naming the first parameter whose value is not a real number, or is NaN.
def _bind_checked(parameters: tuple[str, ...], values: Any) -> list[float]:
    try:
        fits = len(values) == len(parameters)
    except TypeError:
        fits = False
    if not fits:
        raise _bind_count_error(parameters, values)
    out = []
    for name, v in zip(parameters, values, strict=False):
        if (
            isinstance(v, (bool, str, bytes))
            or not hasattr(v, "__float__")
            or (type(v).__module__ == "numpy" and type(v).__name__ in ("bool", "bool_"))
        ):
            raise InvalidTypeError(
                f"bind(): the value of {name!r} must be a real number, got {type(v).__name__} {v!r}"
            )
        if isinstance(v, complex):
            raise InvalidTypeError(
                f"bind(): the value of {name!r} must be a real number, got complex {v!r}"
            )
        v = float(v)
        if v != v:
            raise InvalidValueError(
                f"bind(): the value of {name!r} is nan; a float argument must be a number"
            )
        out.append(v)
    return out


class ScheduledKernel:
    """One compiled module of a kernel: read-only text, schedule and metrics.

    ``kernel.lower(*args)`` and ``compiled.lower(*args)`` return the module
    of one call. Float parameters stay function arguments of the module, and
    the call's values are kept beside it in ``arguments``, so a call with new
    values shares the module; ``generic`` is the module without values.
    ``mlir()`` of a call shows its values substituted.

    A wait whose length is computed from float parameters keeps the module
    generic too: without values it is scheduled at its lowest length, and a
    call's schedule, text and simulation come from the module with the call's
    values bound and scheduled again: the call is ``rescheduled``.

    ``qudits`` holds the quantum system qudits the program drives or measures (gets
    a line on), in increasing order: the qudits ``simulate()`` models.

    A module pickles as MLIR bytecode with its quantum system and options, and runs
    unpickled without compiling again, in any process of the build that
    compiled it; another build raises ``UnsupportedError``. A call's view
    pickles its values with its generic module.

    Qudit parameters annotated ``Qudit`` or ``Register`` are runtime qudits:
    the module takes them as slots, numbered in parameter order, and a call
    binds each slot to its handle's carrier (``carriers``), on the call's
    quantum system. Calls with other handles of the same shape share the module.

    The module with a call's values substituted is shared by every view
    with them once they recur, so exporting or lowering the same call again
    does not build it again; values seen once keep nothing. The latest few
    are kept per module, fewer for a large one.
    """

    __slots__ = (
        "_module",
        "_text",
        "_lock",
        "_concrete_module",
        "quantum_system",
        "metrics",
        "parameters",
        "arguments",
        "qudits",
        "options",
        "scheduled",
        "_generic",
        "waits",
        "classical",
        "runtime_selects",
        "_arity",
        # What exporting each controller's part shares, until every part is
        # exported (``controller._Parts``); unset until the first export.
        "_export_parts",
    )

    # Its controllers' clocks, when one is not the program clock or they
    # share a reference clock; a call's view reads its generic module's.
    clocks: _clocks.ClockPlan | None = None
    # Each line's fine delay in ns, when the machine has line delays; a
    # call's view reads its generic module's.
    fine_delays: Mapping[tuple[str, int], float] | None = None
    # Whether the module is the trace of a compile for other line delays,
    # scheduled again for its own (``with_line_delays``, which makes a
    # _ClockedLowered); a call's view reads the module at the end of its
    # chain of generics (``rescheduled_for_line_delays``).
    _reused_trace: bool = False
    # How many values bind takes: set by a generic module's first bind.
    _arity: int
    _export_parts: _Parts
    # A module with list or qudit parameters is a _Placed: what its calls
    # share, and the carrier of each runtime qudit of a call, in slot order.
    _role: _Role | None = None
    carriers: tuple[int, ...] = ()

    def __init__(
        self,
        module: Any,
        *,
        quantum_system: QuantumSystem,
        metrics: CompileMetrics,
        parameters: tuple[str, ...],
        qudits: tuple[int, ...],
        options: CompileOptions,
        scheduled: bool = True,
        waits: _WaitLengths | None = None,
    ) -> None:
        self._module = module
        self._text: str | None = None
        self._lock = threading.Lock()
        self._concrete_module: Any = None
        self.quantum_system = quantum_system
        self.metrics = metrics
        self.parameters = parameters
        self.arguments: tuple[float, ...] = ()
        self.qudits = qudits
        self.options = options
        self.scheduled = scheduled
        # The module a call's view shares; None for that module itself.
        self._generic: ScheduledKernel | None = None
        self.waits = waits
        # The classical functions the program calls, in declaration order.
        self.classical: tuple[Any, ...] = ()
        # Whether ops read values measurements select while the program runs.
        self.runtime_selects = False

    def bind(self, values: Sequence[float]) -> ScheduledKernel:
        """This module with values for its float parameters, sharing it.

        *values* holds one real number per entry of ``parameters``, in that
        order. Too few or too many values are an ``InvalidTypeError`` naming
        the parameters, and a wait length computed from the values that is not
        a non-negative number of cycles within the wait's bounds a
        ``ValueError``. A value that is not a real number (a bool, a numpy
        bool or a str) is an ``InvalidTypeError``, and a NaN an
        ``InvalidValueError``, each raised here and naming the parameter, on
        the first bind of a module and on every later one; an infinity is an
        ``InvalidValueError`` naming it where the module is used: run,
        simulated, exported or printed. The call's ``arguments`` are floats,
        whatever real numbers were given.

        A module of one call, as ``lower()`` with values returns, has no
        parameters left: binding no values returns that call, and binding any
        is an ``InvalidTypeError``."""
        try:
            arity = self._arity
            arguments = tuple(values)
        except (TypeError, AttributeError):
            # How many values a generic module takes is kept by its first
            # bind, which then binds as every later one does; a call's view
            # keeps none and takes none.
            if self._generic is None:
                if not hasattr(self, "_arity"):
                    self._arity = len(self.parameters)
                    return self.bind(values)
            elif not _bind_checked(self.parameters, values):
                # A call's values are bound: binding none keeps them.
                return self
            return self._with_values(_bind_checked(self.parameters, values))
        if len(arguments) != arity:
            raise _bind_count_error(self.parameters, arguments)
        for v in arguments:
            if v.__class__ is not float or v != v:
                # Numbers of types the check has passed before (an int, a
                # numpy scalar) are made floats as they are; any other
                # value, or a NaN, goes to the check, for the error naming
                # its parameter or for the values as floats.
                if _REAL_TYPES.issuperset(map(type, arguments)):
                    try:
                        floats = tuple(map(float, arguments))
                        total = sum(floats)
                    except Exception:
                        # One that float() does not take (an int too large):
                        # the check raises for it, or for a NaN before it.
                        total = math.nan
                    if total == total:
                        arguments = floats
                    else:
                        # A NaN among them, or infinities of both signs.
                        arguments = tuple(_bind_checked(self.parameters, arguments))
                else:
                    floats = tuple(_bind_checked(self.parameters, arguments))
                    _REAL_TYPES.update(map(type, arguments))
                    arguments = floats
                break
        # The view is made here as in _with_values: a call to it would be a
        # third of a bind.
        waits = self.waits
        if waits is not None:
            try:
                waits.check(arguments)
            except TypeError:
                _bind_checked(self.parameters, arguments)
                raise
        view = _View()
        view._module = self._module
        view._text = None
        view._lock = self._lock
        view._concrete_module = None
        view.quantum_system = self.quantum_system
        view.metrics = self.metrics
        view.parameters = ()
        view.arguments = arguments
        view.qudits = self.qudits
        view.options = self.options
        view.scheduled = self.scheduled
        view._generic = self
        view.waits = waits
        view.classical = self.classical
        view.runtime_selects = self.runtime_selects
        return view

    # ``bind`` for *values* known to be one real number per parameter and
    # none NaN, such as a kernel call's. Whether each is a finite float is
    # checked where the module uses them (``_concrete`` and simulation).
    def _with_values(self, values: Sequence[float]) -> ScheduledKernel:
        waits = self.waits
        if waits is not None:
            waits.check(values)
        view = _View()
        view._module = self._module
        view._text = None
        view._lock = self._lock
        view._concrete_module = None
        view.quantum_system = self.quantum_system
        view.metrics = self.metrics
        view.parameters = ()
        view.arguments = tuple(values)
        view.qudits = self.qudits
        view.options = self.options
        view.scheduled = self.scheduled
        view._generic = self
        view.waits = waits
        view.classical = self.classical
        view.runtime_selects = self.runtime_selects
        return view

    @property
    def generic(self) -> ScheduledKernel:
        """The module without values: the one a call's view shares, or this
        module when it holds no values. ``generic.bind(values)`` makes a call
        of it with new values."""
        generic = self._generic
        return self if generic is None else generic

    @property
    def rescheduled(self) -> bool:
        """Whether the call's float values set a wait length, so its times
        come from scheduling the module again with them bound, rather than
        from the module's own schedule.

        ``True`` for a call whose float values set a wait length, and for a
        module that could not be scheduled without values. ``False`` for a
        module scheduled without values (its waits stand at their lowest
        lengths) and for a call of it whose values move no op, which keeps
        that module's times. Whether the module itself was scheduled again
        for new line delays is ``rescheduled_for_line_delays``.

        It records how ``lower()`` or ``bind()`` made the program, so
        reading it does no work: a call's schedule is still computed when it
        is first read. Copies and pickles keep it.
        """
        return not self.scheduled or (self.waits is not None and bool(self.arguments))

    @property
    def rescheduled_for_line_delays(self) -> bool:
        """Whether the module was compiled again with only other line delays,
        scheduling its kernel's earlier trace again for the new delays
        rather than tracing the kernel anew.

        ``True`` for such a module and for every call of it. ``False`` for a
        module traced by its own compile, such as the first compile of a
        kernel, a compile with the line delays it was last compiled with,
        any compile of a kernel with list or qudit parameters, and a compile
        after a change to what the kernel read while it was traced. It does
        not depend on ``rescheduled``: a call of such a module whose float
        values set a wait length is both.

        It records how the module was made, so reading it does no work.
        Copies and pickles keep it.
        """
        program = self
        while program._generic is not None:
            program = program._generic
        return program._reused_trace

    # Whether the module has runtime qudits and no carriers for them.
    def _unbound(self) -> bool:
        return self._role is not None and self._role.slots > 0 and not self.carriers

    # The module with the values substituted (a private copy), or the
    # module itself when there are no values.
    def _concrete(self) -> Any:
        if not self.arguments:
            return self._module
        if self._concrete_module is None:
            arguments: Sequence[float] = self.arguments
            for v in arguments:
                if v.__class__ is not float and v.__class__ is not int or v - v != 0.0:
                    arguments = self._checked_arguments()
                    break
            with self._lock:
                module = self._module.specialize(list(arguments), [], *self.options.schedule_args())
            self._concrete_module = module
        return self._concrete_module

    # The call's values as floats, or the error naming the first that is
    # not a real number or not finite.
    def _checked_arguments(self) -> list[float]:
        generic = self._generic
        names = generic.parameters if generic is not None else ()
        values = _bind_checked(names, self.arguments)
        for name, v in zip(names, values, strict=False):
            if v - v != 0.0:
                raise InvalidValueError(
                    f"argument {name!r} is {v!r}; a float argument must be a finite number"
                )
        return values

    def mlir(self, level: IRLevel | str = IRLevel.PULSE) -> str:
        """The module as MLIR text at *level* (an ``IRLevel`` or its name):
        ``"pulse"``, the scheduled Pulse dialect; ``"qop"``, ``"cudm"`` and
        ``"llvm"``, after lowering it that far (pulse -> qop -> cudm ->
        LLVM). Every level but ``"pulse"`` needs the float parameters' values,
        and a program that measures lowers only along a measurement-outcome
        path, as ``simulate`` does."""
        if level == "pulse":
            if self._text is None:
                module = self._concrete()
                with self._lock:
                    self._text = module.print()
            return self._text
        if level not in _LEVELS:
            raise InvalidValueError(f"unknown IR level {level!r}; choose from {', '.join(_LEVELS)}")
        if self.parameters:
            raise ValueError(
                "the module has float parameters "
                f"({', '.join(self.parameters)}); lower it with their values first"
            )
        if self._unbound():
            raise InvalidValueError("the module has runtime qudits; lower it with handles first")
        if level == "llvm":
            module = self._concrete()
            with self._lock:
                text: str = module.run_full_lowering()
            return text
        from ._native._cudaq_pulse_native import MLIRPipeline

        pipeline = MLIRPipeline()
        text = pipeline.run_pulse_to_qop(self.mlir())
        return text if level == "qop" else pipeline.run_qop_to_cudm(text)

    @property
    def duration(self) -> Time:
        """How long the program runs: the latest end of its operations, over
        both arms of every branch, as a time (``float(duration)`` is in ns).

        Only a lowered program has a duration: a kernel or ``CompiledKernel``
        has none until it is lowered for a call, since the call's arguments
        decide it. A wait a float parameter sets counts once the module has
        values."""
        # The clock exactly as its float holds it, as Fraction(clock_ghz) is.
        numerator, denominator = self.quantum_system.clock_ghz.as_integer_ratio()
        module = self._concrete() if self.rescheduled or self.carriers else self._module
        with self._lock:
            events = module.event_tree(_LISTED_EVENTS)
        if type(events) is list:
            end: int | None = 0
            for _, _, _, start, length in events:
                if start is None or length is None:
                    end = None
                    break
                end = max(end, start + length)
        else:
            from ._loops import Schedule

            end = Schedule(events[0], ScheduledEvent).latest_end()
        if end is None:
            raise ValueError(
                "the module is scheduled when it has the values of its float "
                f"parameters ({', '.join(self.parameters)}); lower it with them first"
            )
        if end.__class__ is not int:
            return Time(end / self.quantum_system.clock_ghz)
        return _time(end * denominator, numerator)

    @property
    def schedule(self) -> Sequence[ScheduledEvent]:
        """The drives, readouts and waits with their timing.

        Ops in a loop are listed once per iteration; both arms of a
        measurement-conditioned branch are listed. Without values, a wait
        whose length a float parameter sets is listed at its lowest length.
        Up to 10,000 events this is a list; past that, a sequence that
        builds each event when it is read, from its op and its loops'
        iterations: ``len()`` is exact, and indexing, slicing and iterating
        give what the list would.
        """
        module = self._concrete() if self.rescheduled or self.carriers else self._module
        with self._lock:
            events = module.event_tree(_LISTED_EVENTS)
        if type(events) is list:
            return [ScheduledEvent(*event) for event in events]
        from ._loops import Schedule

        return Schedule(events[0], ScheduledEvent)

    def segments(self) -> list[Segment]:
        """The segments of the program, separated by variable and blocking waits.

        A program without such waits has one segment.
        """
        return self._segments()[0]

    # The segments, and the segment each one follows (``None`` for the
    # first): the one its barrier ends, on the lines the barrier holds
    # back. Its shift adds to that segment's.
    def _segments(self) -> tuple[list[Segment], list[int | None]]:
        module = self._module if self.scheduled else self._concrete()
        with self._lock:
            boundaries = module.segment_boundaries()
        segments = [Segment(0, "start", -1, 0, 0, 0)]
        after: list[int | None] = [None]
        for k, (kind, line, start, lowest, highest, before) in enumerate(boundaries, start=1):
            most = segments[before].max_shift_cycles
            extra = None if most is None or highest < 0 else most + highest - lowest
            segments.append(Segment(k, kind, line, start + lowest, 0, extra))
            after.append(before)
        return segments, after

    # The segment of each event of ``schedule`` (with *syncs*, of the
    # controller export), in the same order.
    def _event_segments(self, syncs: bool) -> list[int]:
        if not self.classical:
            return []
        module = self._concrete() if syncs or self.rescheduled else self._module
        with self._lock:
            return list(module.event_segments(syncs))

    @property
    def controller_usage(self) -> UsageReport:
        """What the classical functions use of the processors of the
        quantum system's control system (``cudaq_pulse.controller.UsageReport``),
        built when read; ``print`` it for a table."""
        from .controller import usage

        return usage(self)

    def classical_report(self) -> list[ClassicalCall]:
        """Every call to a classical function, with its timing and slack.

        Calls in a loop are listed once per iteration, in program order.
        """
        return (_report or _load_report())(self)

    @property
    def idle_windows(self) -> IdleWindows | None:
        """The idle windows of a program lowered by ``compiled.replace_idle(rule)``:
        each stretch of at least the rule's threshold in which a qudit idles,
        with what was done there (see ``IdleWindow``). ``None`` for a program
        whose idle windows were not replaced."""
        return getattr(self.metrics, "idle_windows", None)

    @property
    def exposure(self) -> Exposure:
        """What the schedule does to each qudit over time, built when read:
        per qudit and per outermost loop it has a pulse in, the cycles of
        one round in which the qudit is driven, read out and idle, the number of rounds, and
        what the idle time costs at the qudit's T1 and T2 star (see
        ``QuditExposure``); ``print`` it for a table. Loops stay rolled, so
        it costs the same for 2 rounds as for a million. ``UnsupportedError``
        for a program whose schedule depends on its arguments' values."""
        from ._exposure import report

        return report(self)

    def export_to_emulator(self, *, controller: int | None = None) -> dict[str, Any]:
        """The program for a controller emulator: its schedule by segment, its
        calls and the manifest of its classical functions (see
        ``cudaq_pulse.classical.emulation``). On a control system of several
        controllers, that of the one *controller* indexes, in its cycles."""
        from .classical.wire import _emulation_export as export

        return export(self, controller)

    def export_to_controller(
        self,
        sample_rate_ghz: float | None = None,
        *,
        controller: int | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        """The program for an out-of-tree controller backend: its lines and
        every timed operation with its waveform, tone, frame and branches
        (see ``cudaq_pulse.controller``); with *sample_rate_ghz*, each
        envelope's samples too. On a control system of several controllers,
        the program of the one *controller* indexes, in its own cycles (a
        program that plays the lines of more than one needs it). Exporting
        each controller in turn walks and checks the whole program once:
        the program keeps that until every controller's part is exported.
        ``version=2`` gives the structured format, which keeps loops
        rolled."""
        from .controller import export

        return export(self, sample_rate_ghz, controller, version)

    def timeline(
        self, *, figsize: tuple[float, float] = (14, 4), title: str | None = None
    ) -> Figure:
        """Plot the schedule, one row per line, in a figure of *figsize*
        inches with *title* (needs matplotlib); see ``viz.plot_schedule``."""
        from .viz.timeline import plot_schedule

        return plot_schedule(self, figsize=figsize, title=title)

    # Read-only: a copy is the module itself.
    def __copy__(self) -> ScheduledKernel:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ScheduledKernel:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import program_reduce

        return program_reduce(self)

    def __repr__(self) -> str:
        params = f", parameters={list(self.parameters)}" if self.parameters else ""
        return (
            f"<ScheduledKernel {self.metrics.op_count} ops on {self.quantum_system.name}{params}>"
        )


class _View(ScheduledKernel):
    """A call's view of a module, made by ``bind``: ``_View()`` runs no
    ``__init__`` and is the cheapest way to make a ``ScheduledKernel``."""

    __slots__ = ()
    if TYPE_CHECKING:

        def __init__(self) -> None: ...

    else:
        __init__ = object.__init__


def _lowered(
    module: Any,
    clocks: _clocks.ClockPlan | None,
    fine_delays: Mapping[tuple[str, int], float] | None,
    role: _Role | None = None,
    **fields: Any,
) -> ScheduledKernel:
    if role is None and clocks is None and fine_delays is None:
        return ScheduledKernel(module, **fields)
    from ._programs import _ClockedLowered, _Placed

    lowered: ScheduledKernel
    if role is not None:
        lowered = _Placed(module, role=role, **fields)
    else:
        lowered = _ClockedLowered(module, **fields)
        lowered._reused_trace = False
    lowered.clocks = clocks
    lowered.fine_delays = fine_delays
    return lowered


_report: Callable[[ScheduledKernel], list[ClassicalCall]] | None = None


def _load_report() -> Callable[[ScheduledKernel], list[ClassicalCall]]:
    global _report
    from ._classical_report import report

    _report = report
    return report


# The error for a literal the extension rejected, with the kernel line
# that wrote it. The kernel is traced again up to that line, never running
# its host code again: a trace that reaches host code first stops there and
# names that line as the one at or before it. When the trace differs (the
# kernel read state that changed), the line is left out rather than
# guessed.
def _rejected_literal(
    fn: Any, trace_args: Sequence[Any], traced: Any, error: PackedValueError
) -> CompilationError:
    from .kernel._emitter import add_source_note

    message, word = error.args
    limited = vars(traced).get("get_buffer")
    if limited is not None:
        # A builder under a controller's longest program says the limit.
        message = limited.restated(message)
    arms = message.startswith("arms:")
    if arms:
        located = _different_lines(message, None)
    else:
        located = CompilationError(message, code="wrong_argument")
    finder = traced.finder(word)
    try:
        if not finder.locate(fn, trace_args):
            return located
    except Exception:
        # The trace failed. When *traced* is a trace that failed too (its
        # record is in a loop traced rolled, where finding it does not stop
        # the trace), the two are one trace if they wrote the same.
        if finder.frames is None:
            return located
    written = finder.get_buffer()
    if traced.get_buffer()[: len(written)] != written:
        return located
    if arms:
        located = _different_lines(message, finder.labels)
    if finder.frames:
        frames, heading = finder.frames, None
    elif finder.host_frames:
        frames = finder.host_frames
        heading = (
            "while tracing the kernel (written at or after the last line, which calls "
            "host code; finding the line stops there, so host code does not run twice):"
        )
    else:
        return located
    located.pulse_source = frames[0]  # type: ignore[attr-defined]
    located.pulse_callers = frames[1:]  # type: ignore[attr-defined]
    add_source_note(located, heading)
    return located


# The error for a name, or a place of a list, that holds different
# lines (or different tones) after the two arms of an if on a
# measurement, and is then operated on, or that a loop compiled as one
# loop would have to carry between its iterations. *message* is the
# extension's: ``arms:``, the place of that result of the if, ``,tone``
# when it is a tone and ``,loop`` for the loop. *labels* is what each
# result of the if is carried by, when known.
def _different_lines(message: str, labels: Sequence[str] | None) -> CompilationError:
    place, *marks = message[5:].split(",")
    what = "a name or a place of a list"
    if labels is not None and place.isdigit() and int(place) < len(labels):
        what = labels[int(place)]
    kind = "tone" if "tone" in marks else "line"
    fix = (
        f"do in each arm what follows on that arm's {kind}, or bind it to the same {kind} "
        "in both arms"
    )
    if "loop" in marks:
        # Which iterations read the name is known only when each is traced.
        so = "so the loop cannot carry it from one iteration to the next"
        fix += (
            "; when no iteration reads it before binding it, a larger unroll_threshold= "
            "compiles the loop one iteration at a time"
        )
    elif kind == "tone":
        so = "so no operation after the if can be given it"
    else:
        so = "so no operation after the if can be written on it"
    return CompilationError(
        f"after this if, {what} holds one {kind} when the measurement is set and another when "
        f"it is not (its arms bind it differently, as a, b = b, a in one arm does), {so}",
        fix=fix,
        code="runtime_value",
    )


# Why a trace of *fn* failed with *failed*, when what it wrote before
# failing says so: a record the extension rejects (an if whose arms leave
# different lines in one place) leaves the tracer with a line it has lost
# track of, and the trace fails at a later operation on that line with an
# error about the operation. Only a line the failing operation was given
# is looked up: another the kernel holds may be lost as well, and is not
# why this operation failed.
def _cause_in_program(
    fn: Any,
    trace_args: Sequence[Any],
    builder: Any,
    target: QuantumSystem,
    parameters: tuple[str, ...],
    slots: int,
    failed: CompilationError,
) -> CompilationError | None:
    from .kernel.ir_builder import LINEAR_TYPES, IRValue

    # A line the tracer has lost track of is no channel's value and has no
    # tone. Two failures are that line's: an operation given it finds no
    # tone among its operands (play_calibrated, where the operand error is
    # raised), and an attribute of it is read (_value_attribute). Any other
    # failure is its own, whatever lines the kernel holds.
    trace = failed.__traceback__
    if trace is None:
        return None
    while trace.tb_next is not None:
        trace = trace.tb_next
    held = trace.tb_frame.f_locals
    site = trace.tb_frame.f_code.co_name
    given: Sequence[Any] = ()
    if site == "_value_attribute":
        given = (held.get("value"),)
    elif site == "play_calibrated":
        operands = held.get("call_args")
        # Identity, not ``in``: a line compared with ``==`` is an error.
        if operands.__class__ is list and any(item is None for item in operands):
            given = operands
    lost = [
        item.vid
        for item in given
        if item.__class__ is IRValue and item.vtype in LINEAR_TYPES and item.chan is None
    ]
    if not lost:
        return None
    count = builder.qudit_count
    tables = target._tables
    if tables[0] < count:
        tables = target._frequency_tables(count)
    try:
        PulseModuleBuilder().build_from_packed(
            builder.get_buffer(),
            target.clock_ghz,
            count,
            tables[1],
            list(parameters),
            ["f64"] * len(parameters),
            name=builder.name,
            slot_count=slots,
            readout_freqs=tables[2],
            lost_lines=lost,
        )
    except PackedValueError as error:
        if error.args[0].startswith(("arms:", "sync():")):
            return _rejected_literal(fn, trace_args, builder, error)
    except Exception:  # noqa: S110
        # What was traced ends where the trace failed: not a whole program.
        pass
    return None


_BUDGETS = ("max_duration_cycles", "reaction_deadline_ns", "max_program_duration")


def build(
    fn: Any,
    trace_args: Sequence[Any],
    parameters: tuple[str, ...],
    target: QuantumSystem,
    options: CompileOptions,
    slots: int = 0,
    role: _Role | None = None,
    max_duration_cycles: int | None = None,
) -> tuple[ScheduledKernel, Any]:
    """Trace *fn* and build its module on *target*.

    *trace_args* holds a ``Parameter`` sentinel for every float parameter
    named in *parameters*; those stay function arguments of the module, as
    do the *slots* runtime qudits (slot k traces as the qudit of index
    -1 - k). A *role* names the
    symbols of a module with list or qudit parameters, and gets the rest of
    what its calls share. With *max_duration_cycles*, scheduling fails if
    the program can take longer.
    Returns the module and the builder, whose record of readouts and host
    calls the caller needs. Raises ``SymbolicValueError`` naming the float
    parameters an integer of the program (a waveform duration, a cast to
    int) was computed from: the module needs their values.
    """
    from .kernel.bytecode_bridge import _trace_kernel_with_builder
    from .kernel.packed_ir_builder import PackedIRBuilder

    metrics = CompileMetrics(quantum_system=target.name)
    name = getattr(fn, "__name__", "kernel")
    t0 = time.perf_counter()
    machine = options.machine
    builder = PackedIRBuilder(
        name=name,
        clock_ghz=target.clock_ghz,
        flatten_schedules=options.flatten_schedules,
    )
    builder.unroll_threshold = options.unroll_threshold
    # A coupled system's tones are a Mapping that works higher ones out when
    # read; the emitter only calls .get on it.
    builder.line_tones = target._line_tones  # type: ignore[assignment]
    if target._description._own_carrier:
        _refuse_other_qudits_tones(builder, target, fn, trace_args)
    builder.machine = machine
    builder.locate_structural = frozenset(parameters)
    plain = machine._plain
    clocks = None if plain else machine._clocked_builder(builder, target)
    # One try for the trace and the module: a try statement costs a bytecode.
    try:
        if options.unit_warnings:
            checked = options.tracer(builder, target)
            builder.returned = _trace_kernel_with_builder(fn, builder, trace_args, checked)
        else:
            builder.returned = _trace_kernel_with_builder(fn, builder, trace_args)
        needed = tuple(sorted(builder.structural & set(parameters)))
        if needed:
            error = SymbolicValueError(
                f"{name}(): {', '.join(needed)} must be known while the kernel compiles",
                needed,
                "a value that must be an integer, such as a waveform duration",
            )
            frames = getattr(builder, "structural_at", {}).get(needed[0])
            if frames:
                error.pulse_source = frames[0]  # type: ignore[attr-defined]
                error.pulse_callers = frames[1:]  # type: ignore[attr-defined]
            raise error
        check_root_schedule(options.schedule, builder)
        qudits = builder.qudits
        # The quantum system's tables, of which the module records qudits 0 to
        # qudit_count - 1: when the target defines all of those, it defines every
        # qudit of the program, else they are checked.
        qudit_count = builder.qudit_count
        carrier_count, freq, readout_freqs = target._tables
        if carrier_count < qudit_count:
            carrier_count, freq, readout_freqs = target._frequency_tables(qudit_count)
            if carrier_count < qudit_count:
                missing = [q for q in qudits if q not in target.frequencies]
                if missing:
                    # Runtime qudit k is qudit -1 - k, of no target.
                    missing = [q for q in missing if q >= 0]
                    if missing:
                        defined = sorted(target.frequencies)
                        span = f"{defined[0]}..{defined[-1]}" if defined else "none"
                        raise CompilationError(
                            f"{name}() uses qudit {', '.join(map(str, missing))}, which quantum "
                            f"system {target.name!r} does not define (it defines qudits {span})",
                            code="control_system",
                        )
        buf: Any = builder.get_buffer()
        routes = builder.routes
        links = [link for link, _, _ in routes] if routes else []
        metrics.trace_ms = (time.perf_counter() - t0) * 1000
        metrics.op_count = builder.op_count

        t0 = time.perf_counter()
        if slots:
            module = PulseModuleBuilder().build_from_packed(
                buf,
                target.clock_ghz,
                qudit_count,
                freq,
                list(parameters),
                ["f64"] * len(parameters),
                name=name,
                slot_count=slots,
                readout_freqs=readout_freqs,
            )
        elif parameters:
            module = PulseModuleBuilder().build_from_packed(
                buf,
                target.clock_ghz,
                qudit_count,
                freq,
                list(parameters),
                ["f64"] * len(parameters),
                name=name,
                readout_freqs=readout_freqs,
            )
        else:
            module = PulseModuleBuilder().build_from_packed(
                buf,
                target.clock_ghz,
                qudit_count,
                freq,
                name=name,
                readout_freqs=readout_freqs,
            )
    except PackedValueError as error:
        raise _rejected_literal(fn, trace_args, builder, error) from None
    except CompilationError as failed:
        # The failure may be the late sign of something traced earlier.
        cause = None
        if failed.__class__ is CompilationError:
            cause = _cause_in_program(fn, trace_args, builder, target, parameters, slots, failed)
        if cause is None:
            raise
        raise cause from None
    if links:
        system = machine._system
        module.set_attributes(
            machine.links_attribute(links)
            if system is None or machine.controller is not None
            else system._links_attribute(machine, links, target.clock_ghz)
        )
    fine_delays = (
        None
        if plain and max_duration_cycles is None
        else machine._clocked_module(module, target, clocks, max_duration_cycles)
    )
    metrics.ffi_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    runtime_selects = False
    if builder.selects:
        runtime_selects = module.lower_selects(
            [kind for kind in _RUNTIME_KINDS if _can_select_at_runtime(target, kind)]
        )
    cpp_passes = [_PASS_MAP[p] for p in options.passes]
    if cpp_passes:
        module.run_passes(cpp_passes)
    metrics.passes_ms = (time.perf_counter() - t0) * 1000

    # Waveform durations never depend on a float parameter (a float that
    # reaches one is specialized on); a wait whose length does is scheduled at
    # its lowest length, and again with each call's values.
    t0 = time.perf_counter()
    scheduled = True
    if clocks is None or clocks.readout_latency_cycles is None:
        schedule_args = options.schedule_args()
    else:
        from ._programs import _schedule_args

        schedule_args = _schedule_args(options, clocks)
    try:
        module.schedule(*schedule_args)
    except Exception as e:
        text = str(e)
        # Delayed lines take the program past what a schedule counts, with
        # every wait at its lowest length: with any values.
        at = text.find("the program lasts more than 2**60 cycles with its lines delayed")
        if at >= 0:
            raise CompilationError(text[at:].partition("\n")[0], code="limit_exceeded") from None
        # A scheduled program longer than a controller runs.
        at = text.find("'s max_program_duration of ")
        if at >= 0:
            at = text.rfind("kernel '", 0, at)
            raise CompilationError(text[at:].partition("\n")[0], code="limit_exceeded") from None
        # A call over a per-controller link that uses several controllers
        # fails with any values too. The error names the function as the
        # kernel calls it, not the version of it the module holds.
        for line in text.splitlines():
            at = line.find("the call of '")
            if at >= 0 and ", which has one instance per controller, uses " in line:
                line = line[at:]
                symbol = line.split("'", 2)[1]
                called = next((f.base_name for f in builder.classical if f.name == symbol), symbol)
                line = line.replace(f"'{symbol}'", f"'{called}'", 1)
                raise CompilationError(f"{name}(): {line}", code="control_system") from None
        # A timing budget counts every wait at its highest length, so it
        # fails the same way with any values.
        if not parameters or any(name in text for name in _BUDGETS):
            raise
        scheduled = False
    if scheduled and not plain:
        module = _spaced(module, machine, target, schedule_args)
    metrics.schedule_ms = (time.perf_counter() - t0) * 1000
    metrics.total_ms = metrics.trace_ms + metrics.ffi_ms + metrics.passes_ms + metrics.schedule_ms

    waits = None
    if builder.variable_waits:
        from ._waits import wait_lengths

        waits = wait_lengths(
            builder.variable_waits,
            name,
            1 if clocks is None else clocks.lcm,
            builder.wait_grids,
            module.program_length(),
        )
    if plain and role is None:
        lowered = ScheduledKernel(
            module,
            quantum_system=target,
            metrics=metrics,
            parameters=parameters,
            qudits=builder.used_qudits or qudits,
            options=options,
            scheduled=scheduled,
            waits=waits,
        )
    else:
        used = builder.used_qudits
        if slots:
            qudits = tuple(q for q in qudits if q >= 0)
            used = tuple(q for q in used if q >= 0)
        lowered = _lowered(
            module,
            clocks,
            fine_delays,
            role,
            quantum_system=target,
            metrics=metrics,
            parameters=parameters,
            qudits=used or qudits,
            options=options,
            scheduled=scheduled,
            waits=waits,
        )
    calls = builder.classical
    if calls:
        lowered.classical = called = _ClassicalFunctions(calls)
        called.routes = builder.routes  # type: ignore[assignment]
    lowered.runtime_selects = runtime_selects
    if role is not None:
        role.slots = slots
        role.lines = builder.used_slots
        role.used = used
        role.every = qudits
        role.program_id = role.identify(buf, name, parameters, target, options)
    if not plain:
        try:
            lowered = _check_controller(lowered)
        except CompilationError as error:
            # A try statement costs a bytecode where the check passes.
            return _placed_by_limits(
                error, fn, trace_args, parameters, target, options, slots, role, max_duration_cycles
            )
    return lowered, builder


#: The policy that places a program by its control system's limits, for
#: each that does not (``_placed_by_limits``).
_PLACED_BY_LIMITS = {"asap": "rcp", "alap": "alap_rcp"}


# A program asap or alap places past a limit of its control system, which
# declares how many drives or readouts play at once, built again as rcp or
# alap_rcp places it, when that keeps to every limit; else *error*, the limits
# the program is over. Only a program the controller check refuses comes here;
# one compiled with ``keeps_schedule`` options (idle replacement's fills) keeps
# its schedule and its error.
def _placed_by_limits(
    error: CompilationError,
    fn: Any,
    trace_args: Sequence[Any],
    parameters: tuple[str, ...],
    target: QuantumSystem,
    options: CompileOptions,
    slots: int,
    role: _Role | None,
    max_duration_cycles: int | None,
) -> tuple[ScheduledKernel, Any]:
    from .controller import ControllerLimitError

    machine = options.machine
    placed = _PLACED_BY_LIMITS.get(options.schedule)
    if (
        not isinstance(error, ControllerLimitError)
        or placed is None
        or getattr(options, "keeps_schedule", False)
        or (machine.max_concurrent_drives is None and machine.max_concurrent_readouts is None)
    ):
        raise error
    try:
        return build(
            fn,
            trace_args,
            parameters,
            target,
            dataclasses.replace(options, schedule=placed),
            slots,
            role,
            max_duration_cycles,
        )
    except Exception:
        # Placed by the limits, the program fails otherwise: the limit it is
        # over is the error to give.
        raise error from None


# Set *builder* up for the model's controllers; their clock plan.
# ``_MachineModel._clocked_builder``: the model of a control system with a
# broadcast line has its own (``_broadcast_lines``).
def _clocked_builder(
    self: _MachineModel, builder: Any, target: QuantumSystem
) -> _clocks.ClockPlan | None:
    from . import _clocks

    system = self._system
    controller = self.controller
    if system is not None and controller is None:
        clocks = system._clock_plan(target.clock_ghz)
    elif controller is None and self.reference_clock_ghz is None:
        clocks = None
    else:
        clocks = _clocks.plan(controller, target.clock_ghz, self.reference_clock_ghz)
        if clocks is not None and controller is not None:
            # The controller counts its feedback latency in its own cycles;
            # the scheduler waits it in the program's (a new plan: plans
            # are cached).
            from fractions import Fraction

            latency = Fraction(self.readout_latency_cycles) / clocks.controllers[0].ratio
            clocks = dataclasses.replace(clocks, readout_latency_cycles=math.ceil(latency))
    if clocks is not None:
        builder.line_grid = clocks.grid_of
        builder.wait_grid = clocks.lcm
        if clocks.owners is not None:
            builder.wait_grids = []
    if controller is not None:
        builder.sample_rate = _clocks.sample_rate(controller, target.clock_ghz)
        if controller.max_instructions is not None:
            builder.limit_instructions(controller.max_instructions)
        if controller.max_program_duration is not None:
            builder.limit_cycles(
                max(1, math.floor(controller.max_program_duration * target.clock_ghz + 1e-9))
            )
    elif system is not None:
        rates = [_clocks.sample_rate(c, target.clock_ghz) for c in system.controllers]
        if len(set(rates)) == 1:
            builder.sample_rate = rates[0]
        else:
            from .kernel._mixed_rates import MixedRates

            owners = system._line_owners
            builder.sample_rate_of = lambda role, qudit: (
                None if (owner := owners.get((role, qudit))) is None else rates[owner]
            )
            builder.tracer = MixedRates
    return clocks


# Give *module* its control system (``pulse.control_system``): the
# controller and delay of each line, and each controller's grid, clock
# ratio, feedback and peer latency, start uncertainty, frames, calls and
# *max_duration_cycles*, where any differs from the default; the fine
# delays.
def _clocked_module(
    self: _MachineModel,
    module: Any,
    target: QuantumSystem,
    clocks: _clocks.ClockPlan | None,
    max_duration_cycles: int | None,
) -> Mapping[tuple[str, int], float] | None:
    from . import _clocks

    shifts: Mapping[tuple[str, int], int] = {}
    fine_delays = None
    if self._line_delays:
        shifts, fine_delays = _line_delays(self, target, clocks)
    lines = _lines(self, shifts, fine_delays, target.clock_ghz)
    system = self._system
    if system is not None and self.controller is None:
        controllers = system._controllers_attribute(target.clock_ghz, max_duration_cycles)
    else:
        controllers = _controller_attribute(self.controller, clocks, max_duration_cycles)
    if limits := _program_limits(self, target.clock_ghz):
        module.set_attributes(limits)
    if lines or controllers:
        text = "{pulse.control_system = #pulse.control_system<clock_period_fs = "
        text += _clocks.clock_period_fs(target.clock_ghz)
        if lines:
            text += f", lines = {lines}"
        if controllers:
            text += f", controllers = {controllers}"
        module.set_attributes(text + ">}")
    return fine_delays


# Methods of the model, so that the model of a control system with a
# broadcast line (``_broadcast_lines``) sets up the trace and the module for
# one, and no other compilation asks whether it has one.
_MachineModel._clocked_builder = _clocked_builder  # type: ignore[method-assign]
_MachineModel._clocked_module = _clocked_module  # type: ignore[method-assign]


# For a quantum system whose qudits give no frequencies: make *builder*
# refuse a drive or readout at the tone of another qudit. Each tone is
# then an offset from its own qudit's carrier, and the system does not
# state how far apart two carriers are. A tone whose qudit the trace does
# not know is refused too. Only such a system's builder is wrapped; every
# other compile emits as it did.
def _refuse_other_qudits_tones(
    builder: Any, target: QuantumSystem, fn: Any, trace_args: Sequence[Any]
) -> None:
    emit = builder.emit
    lines = ("pulse.get_drive_line", "pulse.get_readout_line")
    # The qudit (by the value it was made from) of each line and tone the
    # builder made, for a value whose channel the trace does not record.
    made: dict[int, int] = {}

    def owner(value: Any) -> int | None:
        chan = getattr(value, "chan", None)
        return made.get(value.vid) if chan is None else chan.held.key[0]

    def named(qref: int) -> str:
        qudit = builder._qref_to_qudit.get(qref)
        if qudit is None:
            return "another qudit"
        if qudit >= 0:
            layout = target.layout
            return f"qudit {layout.index(qudit) if qudit in layout else qudit}"
        return _runtime_qudit(fn, trace_args, qudit)

    def checked(kind: str, operands: Any = (), result_types: Any = (), attrs: Any = None) -> Any:
        played = kind == "pulse.drive" or kind == "pulse.readout"
        ours = theirs = None
        if played:
            line, tone = operands[0], operands[2]
            ours, theirs = owner(line), owner(tone)
            chan = getattr(line, "chan", None)
            if chan is not None and len(chan.held.cells) > 1 and chan.held.cells[1].value is tone:
                # The line's own tone as it is now.
                theirs = ours
            op = kind.removeprefix("pulse.")
            reason = (
                f"the qudits of quantum system {target.name!r} give no frequency_hz, so each "
                "tone is an offset from its own qudit's carrier and the system does not state "
                "how far apart two carriers are"
            )
            if ours is None or theirs is None:
                raise CompilationError(
                    f"{op}(): the trace cannot tell which qudit's tone tone= is (the tone, or "
                    "the line, is a value of its own: what a loop carries, or what the arms of "
                    f"an if on a measurement bind differently); {reason}",
                    code="control_system",
                    fix="read the tone from its line where it is used, as line.tone or "
                    "line.tones[(1, 2)], or give the qudits their frequencies",
                )
            # One qudit, by its index where the builder knows both.
            mine, other = map(builder._qref_to_qudit.get, (ours, theirs))
            if ours != theirs if mine is None or other is None else mine != other:
                raise CompilationError(
                    f"{op}(): tone= is the tone of {named(theirs)}, and the line is that of "
                    f"{named(ours)}; {reason}",
                    code="control_system",
                    fix="give the qudits their frequencies, or drive the line at its own tone",
                )
        results = emit(kind, operands, result_types, attrs)
        for result in results:
            # An id is used again after the trace takes a body back.
            made.pop(result.vid, None)
        if played:
            # Both known, or the drive was refused.
            if ours is not None and theirs is not None:
                made[results[0].vid], made[results[1].vid] = ours, theirs
        elif kind in lines:
            made[results[0].vid] = made[results[1].vid] = operands[0].vid
        elif results and not kind.startswith("scf."):
            # A tone or line an op makes from one of its kind is that one's
            # qudit's; from several, in their order.
            for vtype in ("tone", "drive_line", "readout_line"):
                after = [r for r in results if r.vtype == vtype]
                before = [o for o in operands if getattr(o, "vtype", None) == vtype]
                if after and len(after) == len(before):
                    for result, source in zip(after, before, strict=True):
                        found = owner(source)
                        if found is not None:
                            made[result.vid] = found
        return results

    builder.emit = checked


# How an error names the runtime qudit that traces as qudit *slot*
# (-1 - k): by the kernel's parameter.
def _runtime_qudit(fn: Any, trace_args: Sequence[Any], slot: int) -> str:
    from ._device import Qudit, Register

    code = getattr(fn, "__code__", None)
    names = code.co_varnames[: code.co_argcount] if code is not None else ()
    for name, value in zip(names, trace_args, strict=False):
        if value.__class__ is Qudit and value.index == slot:
            return f"the qudit {name!r}"
        if value.__class__ is Register:
            for k, qudit in enumerate(value):
                if qudit.index == slot:
                    return f"the qudit {name}[{k}]"
    return "a runtime qudit"


# ``pulse.program_limits``: each controller's ``max_program_duration``,
# which the scheduled program is checked against; empty for none.
def _program_limits(machine: _MachineModel, program_ghz: float) -> str:
    from ._system import _label

    system = machine._system
    controllers = (
        [machine.controller]
        if machine.controller is not None or system is None
        else list(system.controllers)
    )
    entries = []
    for index, controller in enumerate(controllers):
        if controller is None or controller.max_program_duration is None:
            continue
        ns = controller.max_program_duration
        cycles = max(1, math.floor(ns * program_ghz + 1e-9))
        name = '"' + _label(index, controller).replace("\\", "\\\\").replace('"', '\\"') + '"'
        entries.append(
            f"{{controller = {index} : i64, cycles = {cycles} : i64, name = {name}, "
            f'duration = "{ns:g} ns"}}'
        )
    return f"{{pulse.program_limits = [{', '.join(entries)}]}}" if entries else ""


# The ``controllers`` of a program's control system on one controller:
# its grid, frames, calls and *max_duration_cycles*; empty when all are
# the defaults.
def _controller_attribute(
    controller: Controller | None,
    clocks: _clocks.ClockPlan | None,
    max_duration_cycles: int | None,
) -> str:
    fields = []
    if clocks is not None and clocks.controllers[0].grid > 1:
        fields.append(f"grid_cycles = {clocks.controllers[0].grid}")
    if controller is not None:
        if controller.persistent_frames:
            fields.append("persistent_frames")
        if controller.blocks_on_calls:
            fields.append("blocks_on_calls")
    if max_duration_cycles is not None:
        fields.append(f"max_duration_cycles = {int(max_duration_cycles)}")
    return f"[#pulse.controller<{', '.join(fields)}>]" if fields else ""


# The ``lines`` of a program's control system: each line the system
# gives a controller of (on several) or that *shifts* moves, with its
# controller and delay in program cycles, and each readout line's fine
# delay (*fine_delays*, ns) rounded up to the program's cycles, when a
# readout on it ends that much later; empty for none.
def _lines(
    machine: _MachineModel,
    shifts: Mapping[tuple[str, int], int],
    fine_delays: Mapping[tuple[str, int], float] | None = None,
    program_ghz: float = 0.0,
) -> str:
    system = machine._system if machine.controller is None else None
    owners: Mapping[tuple[str, int], int] = {} if system is None else system._line_owners
    fine = {
        key: math.ceil(ns * program_ghz - 1e-9)
        for key, ns in (fine_delays or {}).items()
        if key[0] == "readout" and ns > 0
    }
    lines = []
    for role, qudit in sorted({*owners, *shifts, *fine}):
        line = f"#pulse.line<{role}, qudits = [{qudit}]"
        if shift := shifts.get((role, qudit)):
            line += f", delay_cycles = {shift}"
        if owner := owners.get((role, qudit)):
            line += f", controller = {owner}"
        if cycles := fine.get((role, qudit)):
            line += f", fine_cycles = {cycles}"
        lines.append(line + ">")
    return f"[{', '.join(lines)}]" if lines else ""


def _line_delays(
    machine: _MachineModel,
    target: QuantumSystem,
    clocks: _clocks.ClockPlan | None,
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], float]]:
    from . import _clocks

    system = machine._system if machine.controller is None else None
    try:
        return _clocks.line_delays(
            machine._line_delays,
            machine.controller,
            clocks,
            target.clock_ghz,
            None if system is None else system.controllers,
            None if system is None else system._line_owners,
        )
    except ValueError as e:
        raise CompilationError(str(e)) from None


# *module*, scheduled, with the updates on the update buses of the
# controllers that ask for it spaced (``_bus_spacing``).
def _spaced(module: Any, machine: _MachineModel, target: QuantumSystem, args: Any) -> Any:
    system = machine._system
    given = (machine.controller,) if system is None else system.controllers
    spacing = [c for c in given if c is not None and c.space_updates and c.update_buses]
    if not spacing:
        return module
    from ._bus_spacing import spaced

    return spaced(module, spacing, target, args)


# The drives and readouts *lowered* plays at once past the limits its
# control system gives (``_concurrency``), and whether that is exact.
def _over_concurrency(lowered: ScheduledKernel) -> tuple[list[str], bool]:
    machine = lowered.options.machine
    if machine.max_concurrent_drives is None and machine.max_concurrent_readouts is None:
        return [], True
    from ._concurrency import violations

    return violations(lowered)


# *lowered*, checked against its controller limits: those it exceeds
# whatever its float values fail now, and when some values may exceed
# others, each call's values are checked when they are bound.
def _check_controller(lowered: ScheduledKernel) -> ScheduledKernel:
    machine = lowered.options.machine
    limits = machine.controller
    system = machine._system
    if limits is None and system is None:
        return lowered
    from .controller import _compile_check, _limit_error

    exact = False
    # A float wait length places the ops after it per call: the schedule
    # without values is not one any call plays.
    if lowered.scheduled and lowered.waits is None:
        violations, exact = _compile_check(lowered, limits, system)
        over, known = _over_concurrency(lowered)
        if over:
            violations = [*violations, *over]
        exact = exact and known
        if violations:
            raise _limit_error(lowered, violations)
    if exact:
        return lowered
    from ._programs import _CheckedPerCall, _CheckedPlaced

    return (_CheckedPerCall if lowered._role is None else _CheckedPlaced)(lowered)


def with_line_delays(lowered: _Kernel, options: CompileOptions) -> _Kernel | None:
    """*lowered*, a scheduled module, compiled again for *options*, which
    differ from its own only in the machine's line delays:
    its module before scheduling is kept, and only scheduling, the checks and
    the exports are done again. ``None`` when scheduling it again fails, for
    a full compile to report, and for a module with list or qudit
    parameters, whose calls share what its own schedule gives them."""
    if lowered._role is not None:
        return None
    target = lowered.quantum_system
    clocks = lowered.clocks
    fine_delays = None
    shifts: Mapping[tuple[str, int], int] = {}
    if options.machine._line_delays:
        shifts, fine_delays = _line_delays(options.machine, target, clocks)
    if clocks is None or clocks.readout_latency_cycles is None:
        schedule_args = options.schedule_args()
    else:
        from ._programs import _schedule_args

        schedule_args = _schedule_args(options, clocks)
    t0 = time.perf_counter()
    try:
        with lowered._lock:
            module = lowered._module.with_line_delays(
                _lines(options.machine, shifts, fine_delays, target.clock_ghz), *schedule_args
            )
    except Exception:
        return None
    module = _spaced(module, options.machine, target, schedule_args)
    old = lowered.metrics
    metrics = CompileMetrics(quantum_system=old.quantum_system)
    metrics.op_count = old.op_count
    metrics.schedule_ms = (time.perf_counter() - t0) * 1000
    metrics.total_ms = metrics.schedule_ms
    from ._programs import _ClockedLowered

    # Always a _ClockedLowered, the class that holds the mark below, even
    # with no clocks and no line delays left.
    result = _ClockedLowered(
        module,
        quantum_system=target,
        metrics=metrics,
        parameters=lowered.parameters,
        qudits=lowered.qudits,
        options=options,
        waits=lowered.waits,
    )
    result.clocks = clocks
    result.fine_delays = fine_delays
    result.classical = lowered.classical
    result.runtime_selects = lowered.runtime_selects
    result._reused_trace = True
    if options.schedule not in _PLACED_BY_LIMITS:
        return _check_controller(result)
    from .controller import ControllerLimitError

    try:
        return _check_controller(result)
    except ControllerLimitError:
        # Placed past a limit: a full compile places it by the limits.
        return None


class _ClassicalFunctions(tuple[Any, ...]):
    """The classical functions a program calls, in declaration order, with
    ``routes``: per function, its link id, dispatch key and the controller
    that fixes the key (None for a hashed or built-in key)."""

    routes: list[tuple[int, int, str | None]]
    # Per compiled name, its key and what fixes it, made when first read.
    by_name: dict[str, tuple[int, str | None]]


_NO_KEYS: dict[str, tuple[int, str | None]] = {}


# Per compiled name of a classical function *lowered* calls, its
# dispatch key and the controller that fixes it.
def _dispatch_keys(lowered: ScheduledKernel) -> dict[str, tuple[int, str | None]]:
    called = lowered.classical
    if not called:
        return _NO_KEYS
    keys: dict[str, tuple[int, str | None]] | None = called.__dict__.get("by_name")
    if keys is None:
        keys = called.by_name = {  # type: ignore[attr-defined]
            function.name: (key, fixed_by)
            for function, (_, key, fixed_by) in zip(called, _routes(lowered), strict=True)
        }
    return keys


# Per classical function *lowered* calls, its link id, dispatch key and
# the controller that fixes the key.
def _routes(lowered: ScheduledKernel) -> Sequence[tuple[int, int, str | None]]:
    routes: Sequence[tuple[int, int, str | None]] = getattr(lowered.classical, "routes", ())
    return routes


# Per link id, the keys fixed on it (by function name), each with what
# fixes them: a processor's ``dispatch_keys`` on its ``links``.
def _dispatch_tables(machine: _MachineModel) -> dict[int, dict[str, tuple[int, str]]]:
    system = machine._system
    if system is not None and machine.controller is None:
        controllers: Sequence[Any] = system.controllers
    else:
        controllers = () if machine.controller is None else (machine.controller,)
    tables: dict[int, dict[str, tuple[int, str]]] = {}
    for position, controller in enumerate(controllers):
        label = f"controllers[{position}]" + (f" ({controller.model})" if controller.model else "")
        for index, processor in enumerate(controller.processors):
            if processor.dispatch_keys:
                _fix_keys(
                    machine,
                    tables,
                    f"{label}.processors[{index}]",
                    processor.links,
                    processor.dispatch_keys,
                )
    return tables


# Enter the keys *given* by *label* in *tables*, on each of *links*.
def _fix_keys(
    machine: _MachineModel,
    tables: dict[int, dict[str, tuple[int, str]]],
    label: str,
    links: Sequence[int | str],
    given: Mapping[str, int],
) -> None:
    for link in links:
        if isinstance(link, str) and link not in machine.link_names:
            continue
        table = tables.setdefault(machine.link_id(link), {})
        taken = {key: name for name, (key, _) in table.items()}
        for name, key in given.items():
            known = table.get(name)
            if known is not None and known[0] != key:
                raise CompilationError(
                    f"{label}.dispatch_keys gives {name} {key}, but {known[1]} also serves"
                    f" link {link!r} and gives it {known[0]}; give {name} one key, or put"
                    " the processors on different links",
                    code="control_system",
                )
            owner = taken.get(key, name)
            if owner != name:
                raise CompilationError(
                    f"{label}.dispatch_keys gives {name} {key}, but {table[owner][1]} also"
                    f" serves link {link!r} and gives that key to {owner}; give each"
                    " function its own key, or put the processors on different links",
                    code="control_system",
                )
            table[name] = (key, label)


def compile[**P](
    kernel: Kernel[P],
    /,
    *,
    schedule: str | None = None,
    passes: Sequence[str] | None = None,
    control_system: ControlSystem | Controller | None = None,
    quantum_system: QuantumSystem | None = None,
    flatten_schedules: bool = False,
    unroll_threshold: int | None = None,
    unit_warnings: bool = False,
) -> CompiledKernel[P]:
    """Compile a ``@cudaq_pulse.kernel`` without its arguments.

    Returns a ``CompiledKernel``, a family of specializations. Calling it
    with the kernel's arguments runs the kernel on the quantum system of the qudit
    handles passed and returns what the kernel returns. Float parameters
    stay symbolic: the module takes them as function arguments, so calls
    with new values never recompile. Parameters of any other type (qudits,
    ints) shape the program; each distinct value, on each quantum system, is
    compiled on first use and kept for the life of the compiled object.

    Parameters
    ----------
    kernel:
        A ``@cudaq_pulse.kernel``.
    schedule:
        Scheduling policy of *kernel* itself: ``"asap"``, ``"alap"``,
        ``"rcp"``, ``"alap_rcp"``; by default the kernel's own
        ``schedule=``, else ``"alap"``. Kernels it calls keep their own.
    passes:
        Optimization passes (``Pass`` members or their names), run in the
        order given; ``DEFAULT_PASSES`` by default, ``()`` for none.
    control_system:
        The ``ControlSystem`` to compile for: every program is checked
        against its controllers' wiring, capabilities and limits, and its
        links and lanes set the schedule. One ``Controller`` stands for a
        system of that controller alone; what it cannot take (a line it
        does not play, a link it does not have) is an error that says to
        describe the whole system. By default the control system of
        *quantum_system*, if it has one.
    flatten_schedules:
        Ignore the ``schedule=`` of every called kernel and schedule the whole
        program with *schedule*.
    unroll_threshold:
        A ``for ... in range(n)`` loop whose variable the body never reads is
        unrolled while tracing when n times the number of calls in its body
        is at most this, and emitted once as a rolled loop otherwise. By
        default the kernel's own ``unroll_threshold=``, else 8. A
        non-negative ``int``; 0 never unrolls, so every such loop stays
        rolled, even one of no or one iteration. Either way the
        program, its schedule and its simulation are the same.
    unit_warnings:
        Issue a ``UnitWarning`` at each kernel line that gives a time
        without units (a bare number as a duration, a wait, or a pulse's
        ``sigma``, ``beta`` or ``width``, which counts clock cycles) or a
        frequency far below the quantum system's carriers (one probably in
        MHz or kHz rather than Hz). Off by default. The warnings are
        issued when the kernel traces; they do not change the program.
    quantum_system:
        Pin the compiled kernel to one quantum system: a call with handles of
        another raises ``QuantumSystemMismatchError``, and a call without handles
        runs on it. A pinned kernel without static parameters is built now.
    """
    from .kernel._kernel import CompiledKernel, Kernel

    if not isinstance(kernel, Kernel):
        raise TypeError(f"compile() expects a @cudaq_pulse.kernel, got {type(kernel).__name__}")
    if control_system is None and quantum_system is not None:
        control_system = quantum_system.control_system
    options = CompileOptions.make(
        root_schedule(kernel.schedule, schedule),
        passes,
        control_system,
        flatten_schedules,
        kernel.unroll_threshold if unroll_threshold is None else unroll_threshold,
        unit_warnings,
    )
    return CompiledKernel(kernel, options, quantum_system)
