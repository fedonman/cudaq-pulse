# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantum systems and qudit handles.

A ``QuantumSystem`` is a read-only description of one machine: its calibrated
physics and its layout, which maps the user's qudit numbering to its
carriers. ``qs.qudits[i]`` is a ``Qudit``: a handle that
names one carrier of that quantum system. A kernel called with handles runs on their
quantum system, so one compiled kernel serves every quantum system it is called with.

Quantum systems compare by content: two quantum systems built from the same values are equal
and share compiled kernels. The content is the physics, layout and
calibration (a shipped one by name, measurement run and file, any other by its
values); ``name`` and ``source`` are labels, and equal quantum systems may differ
in them.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

from ._errors import (
    IndexOutOfRangeError,
    InvalidTypeError,
    InvalidValueError,
    NotFoundError,
    Problem,
    PulseError,
    QuantumSystemDefinitionError,
    _closest,
    _count,
    _ranges,
)
from .quantum_systems._registry import (
    NoProcessQuantumSystemError,
    UnknownQuantumSystemError,
    _registry,
    make_quantum_system,
)
from .quantum_systems.base import (
    _DESCRIPTION_FIELDS,
    _DESCRIPTIONS,
    _DRIVE_FRAMES,
    _TRANSMON,
    _Description,
    _DressedTones,
    dressed_frequencies,
)
from .quantum_systems.models import SimulationModel

try:
    # hashlib's own blake2b, without hashlib's import of OpenSSL (a millisecond).
    from _blake2 import blake2b as _blake2b
except ImportError:  # pragma: no cover - CPython builds without the _blake2 module
    from hashlib import blake2b as _blake2b

if TYPE_CHECKING:
    from array import array

    from ._system import ControlSystem
    from ._units import Time
    from .gates import Gate
    from .quantum_systems.base import (
        Coupling,
        DragPulse,
        PhysicalQudit,
        ReadoutPulse,
        ResidualZZ,
        VirtualZ,
    )
    from .quantum_systems.calibration import Calibration
    from .quantum_systems.drift import ParameterDriftModel

    type _MaybeSystem = QuantumSystem | None

__all__ = [
    "QuantumSystem",
    "QuantumSystemMismatchError",
    "NoQuantumSystemError",
    "Qudit",
    "Register",
    "quantum_system",
]


class NoQuantumSystemError(NoProcessQuantumSystemError):
    """A kernel was called without qudit handles and was not compiled for a
    quantum system, so it has no quantum system to run on. The process quantum system is never used
    for a kernel call."""


class QuantumSystemMismatchError(PulseError, RuntimeError):
    """A kernel call mixes handles of different quantum systems, or of an unpinned one."""


_CARRIER = "qudit"
_CONFIG_KEYS = (
    "format",
    "written_by",
    "name",
    "qudits",
    "couplings",
    "residual_zz",
    "model",
    "attribution",
    "layout",
    "calibration",
    "measurement_run",
    "control_system",
    "clock_ghz",
    "frames",
)
# The keys that build a quantum system rather than name a registered one.
_BUILT_KEYS = ("qudits", "couplings", "residual_zz", "model", "attribution")
# Reserved for a model of where carriers are and how they move.
_RESERVED_CONFIG = frozenset({"zones", "sites", "moves", "transport"})


def _read_only(array: Any) -> Any:
    array.setflags(write=False)
    return array


# Content hash of everything a compiled kernel depends on, and of the
# calibration the quantum system holds: quantum systems of one fingerprint are one quantum
# system.
def _fingerprint(target: _Description, layout: tuple[int, ...], calibration: Any) -> str:
    qudits = [
        (
            q.index,
            q.frequency_hz,
            q.anharmonicity_hz,
            q.t1_us,
            q.t2_star_us,
            q.t2_echo_us,
            q.thermal_population,
            None if (drive := q.drive) is None else (type(drive).__name__, drive._values),
            None if (readout := q.readout) is None else (type(readout).__name__, readout._values),
            q._calibrations._content,  # type: ignore[attr-defined]
        )
        for q in sorted(target.qudits.values(), key=lambda q: q.index)
    ]
    couplings = [
        (c.qudits, c.strength_hz, c._calibrations._content)  # type: ignore[attr-defined]
        for c in target.couplings
    ]
    residual_zz = [(x.qudits, x.rate_hz) for x in target.residual_zz]
    content: tuple[Any, ...] = (
        target.model,
        target.clock_ghz,
        qudits,
        couplings,
        residual_zz,
        layout,
    )
    # Dressed frames are the default and add nothing; bare ones do.
    content += target._frames_content
    # Uncalibrated quantum systems keep their fingerprints.
    if calibration is not None:
        content += (calibration._identity,)
    return _blake2b(repr(content).encode(), digest_size=16).hexdigest()


class Qudit:
    """A handle naming one carrier of a quantum system.

    ``index`` is the carrier's index in the quantum system (the layout is applied
    when the handle is created); ``logical`` is the position the user
    indexed. A handle is identity, not location.
    """

    __slots__ = ("quantum_system", "carrier", "index", "logical", "_key")

    quantum_system: QuantumSystem
    carrier: str
    index: int
    logical: int

    def __init__(self, quantum_system: QuantumSystem, index: int, logical: int) -> None:
        self.quantum_system = quantum_system
        self.carrier = _CARRIER
        self.index = index
        self.logical = logical
        self._key = (quantum_system.fingerprint, _CARRIER, index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Qudit) and other._key == self._key

    def __hash__(self) -> int:
        return hash(self._key)

    # Read-only: a copy is the handle itself.
    def __copy__(self) -> Qudit:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Qudit:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (Qudit, (self.quantum_system, self.index, self.logical))

    @property
    def dressed_frequency_hz(self) -> float | None:
        """The qudit's 0-1 frequency in Hz with its quantum system's
        couplings included: the frequency its drive tone plays at. Without
        couplings it is the qudit's ``frequency_hz``, which stays the bare
        value the Hamiltonian uses. Where the qudits give their frequencies
        as offsets from a shared reference it is an ``Offset`` from it, and
        where they give none it is ``None``."""
        system = self.quantum_system
        frequency = float(system.frequencies[self.index])
        frame = system._description._frame
        if frame == "absolute":
            return frequency
        if frame == "shared_reference":
            from .quantum_systems.base import Offset

            return Offset(frequency)
        return None

    def __repr__(self) -> str:
        return (
            f"<Qudit {self.logical} -> {self.carrier} {self.index} of {self.quantum_system.name}>"
        )

    if TYPE_CHECKING:

        @overload
        def get_calibration(
            self,
            gate: Literal[Gate.X, Gate.X90, Gate.Y, Gate.Y90],
            transition: tuple[int, int] = (0, 1),
        ) -> DragPulse: ...
        @overload
        def get_calibration(
            self,
            gate: Literal[Gate.Z, Gate.S, Gate.T],
            transition: tuple[int, int] = (0, 1),
        ) -> VirtualZ: ...
        @overload
        def get_calibration(
            self, gate: Literal[Gate.READOUT], transition: tuple[int, int] = (0, 1)
        ) -> ReadoutPulse: ...
        @overload
        def get_calibration(
            self, gate: Gate, transition: tuple[int, int] = (0, 1)
        ) -> DragPulse | VirtualZ | ReadoutPulse: ...

    def get_calibration(self, gate: Any, transition: tuple[int, int] = (0, 1)) -> Any:
        """The calibrated pulse of *gate* on this qudit of its quantum system, as
        ``get_drive_line(q).get_calibration(gate)`` (``X`` to ``T``) and
        ``get_readout_line(q).get_calibration(pulse.gates.READOUT)`` give it
        in a kernel; see ``PhysicalQudit.get_calibration``."""
        return self.quantum_system._description.qudits[self.index].get_calibration(gate, transition)


class Register(Sequence[Qudit]):
    """An ordered selection of qudit handles of one quantum system."""

    __slots__ = ("_qudits",)

    def __init__(self, qudits: Sequence[Qudit]) -> None:
        self._qudits = tuple(qudits)

    def __len__(self) -> int:
        return len(self._qudits)

    @overload
    def __getitem__(self, index: int) -> Qudit: ...
    @overload
    def __getitem__(self, index: slice | Sequence[int]) -> Register: ...

    def __getitem__(self, index: Any) -> Qudit | Register:
        qudit: Qudit
        try:
            if index.__class__ is int:
                qudit = self._qudits[index]
                return qudit
            if isinstance(index, slice):
                return Register(self._qudits[index])
            if isinstance(index, Sequence) and not isinstance(index, str):
                return Register([self._qudits[i] for i in index])
            qudit = self._qudits[index]
        except IndexError:
            raise _out_of_range(index, len(self._qudits), self._owner()) from None
        return qudit

    def _owner(self) -> str:
        return "the register"

    def __iter__(self) -> Iterator[Qudit]:
        return iter(self._qudits)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Register) and other._qudits == self._qudits

    def __hash__(self) -> int:
        return hash(self._qudits)

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self._qudits,))

    def __repr__(self) -> str:
        return f"Register({[q.logical for q in self._qudits]})"


class _Handles:
    """``qs.qudits``: index with an int, a list of ints or a slice."""

    __slots__ = ("_all",)

    def __init__(self, qudits: tuple[Qudit, ...]) -> None:
        self._all = _DeviceRegister(qudits)

    def __len__(self) -> int:
        return len(self._all)

    def __iter__(self) -> Iterator[Qudit]:
        return iter(self._all)

    @overload
    def __getitem__(self, index: int) -> Qudit: ...
    @overload
    def __getitem__(self, index: slice | Sequence[int]) -> Register: ...

    def __getitem__(self, index: Any) -> Qudit | Register:
        handles: Qudit | Register = self._all[index]
        return handles

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Handles, (self._all._qudits,))


class _DeviceRegister(Register):
    """All the qudits of a quantum system, which its errors name."""

    __slots__ = ()

    def _owner(self) -> str:
        return (
            f"quantum system {self._qudits[0].quantum_system.name!r}"
            if self._qudits
            else "the quantum system"
        )


# The error for *index*, which names a qudit past the *n* of *owner*.
def _out_of_range(index: Any, n: int, owner: str) -> IndexOutOfRangeError:
    wrong = index
    if isinstance(index, Sequence) and not isinstance(index, str):
        wrong = next((i for i in index if isinstance(i, int) and not -n <= i < n), index)
    if n == 0:
        return IndexOutOfRangeError(f"qudit index {wrong!r} is out of range: {owner} has no qudits")
    return IndexOutOfRangeError(
        f"qudit index {wrong!r} is out of range: {owner} has {_count(n, 'qudit')}, "
        f"indexed 0 to {n - 1} (or -{n} to -1)"
    )


class QuantumSystem:
    """A read-only machine description; see the module docstring.

    ``clock_ghz`` is the program clock: kernels count time in its cycles.
    ``pulse.quantum_system(..., clock_ghz=)`` (a system file's ``"clock_ghz"``) sets
    it; else the fastest clock of the control
    system's controllers, when every other one divides it (controllers whose
    clocks do not divide evenly need it declared); else the quantum system's own. Each
    controller may run its own clock, a whole-cycle ratio of it (see
    ``cudaq_pulse.controller``).

    ``qudit_freq[q]`` is the 0-1 transition frequency of qudit ``q`` (Hz) and
    ``transition_freq[q, i, j]`` that of levels ``i`` and ``j`` (the upper
    level's energy less the lower's, in either order of ``i`` and ``j``),
    over the levels the system's model gives each qudit (NaN past a qudit's
    last level); both are indexed by the user's qudit numbering and are
    read-only arrays.
    ``control_system`` is the ``ControlSystem`` that plays it, if given;
    ``compile(kernel, quantum_system=dev)`` checks every program against it.

    A quantum system built with ``parameter_drift=`` drifts (see
    ``cudaq_pulse.quantum_systems.drift``): ``physical_qudits`` are its
    calibrated values, all that compiling reads, and ``true_qudits`` the
    values ``simulate`` plays, which drift as it runs and as
    ``advance(duration)`` moves time on; ``parameter_drift`` is the model of
    each drifting qudit and ``parameter_drift_time`` the drift time so far.
    It equals its non-drifting twin: the same fingerprint, compiled programs
    and exports. Without drift, ``true_qudits`` is ``physical_qudits``,
    ``parameter_drift`` and ``parameter_drift_time`` are ``None`` and
    ``advance`` raises.
    """

    # The registered name the quantum system was made from, as ``as_dict`` writes it.
    _spec: str | None = None
    _drift = False
    # Each qudit's dressed 0-1 frequency, in a system whose couplings move
    # some (dressed_frequencies); else None.
    _dressed: dict[int, float] | None = None
    # What a kernel whose qudits are runtime parameters depends on, made by
    # the first such kernel (_programs._structure).
    _structure: str | None = None

    def __init__(
        self,
        target: _Description,
        *,
        name: str,
        layout: Sequence[int] | None = None,
        source: str = "",
        calibration: Any = None,
        control_system: ControlSystem | None = None,
        clock_ghz: float | None = None,
        _variant: bool = False,
    ) -> None:
        if control_system is not None:
            from ._system import ControlSystem

            if not isinstance(control_system, ControlSystem):
                raise InvalidTypeError(
                    f"QuantumSystem control_system must be a ControlSystem, got {control_system!r}"
                )
        problems: list[Problem] = []
        if clock_ghz is not None or (control_system is not None and control_system._needs_clock):
            from ._clocks import _program_clock

            clock, problems = _program_clock(control_system, clock_ghz, target.clock_ghz)
            if clock is None:
                raise QuantumSystemDefinitionError(problems=problems)
            if clock != target.clock_ghz:
                target = target.replace(clock_ghz=clock)
        layout = (
            tuple(sorted(target.qudits))
            if layout is None
            else _layout(layout, target, name, problems)
        )
        if control_system is not None:
            from ._system import _wiring_problems

            problems.extend(_wiring_problems(control_system, target))
        if layout is None or problems:
            raise QuantumSystemDefinitionError(problems=problems)
        self.control_system = control_system
        # The target before its calibration was applied, as ``as_dict`` writes it.
        self._base_target = target
        self._description = target
        self.name = name
        self.source = source
        # The Calibration the target's fitted values came from, if any.
        self.calibration = calibration
        self.layout = layout
        self.clock_ghz = float(target.clock_ghz)
        self._arrays: tuple[Any, Any] | None = None
        # The frequency tables a compiled module records, as
        # _frequency_tables returns them: (count, 0-1 frequencies, readout
        # tones); none before the first compile.
        self._tables: tuple[int, Any, Any] = (-1, None, None)
        # A variant (replace) makes the rest when it is first read
        # (__getattr__): most are only simulated against, which reads none of it.
        # One with couplings reads its line tones through the dressed method,
        # which makes it first (a class attribute would not).
        if not _variant:
            self._finish()
        elif target.couplings:
            self._line_tones = self._dressed_line_tones  # type: ignore[method-assign]

    # The frequencies, fingerprint, identity and qudit handles.
    def _finish(self) -> None:
        target, layout, control_system = self._description, self.layout, self.control_system
        # The tones' frequencies: each qudit's 0-1 frequency, dressed by the
        # couplings in a system that has some (the Hamiltonian keeps the
        # bare frequency_hz). A system without couplings does no more than
        # test that it has none.
        self.frequencies: Mapping[int, float] = target.frequencies
        if target.couplings:
            dressed = dressed_frequencies(target)
            if dressed is not None:
                self.frequencies = self._dressed = dressed
                self._line_tones = self._dressed_line_tones  # type: ignore[method-assign]
        self.fingerprint = _fingerprint(target, layout, self.calibration)
        # What equality compares: the fingerprint alone without a control
        # system, so comparing such quantum systems costs one string comparison.
        self._identity: Any = (
            self.fingerprint if control_system is None else (self.fingerprint, control_system)
        )
        self.qudits = _Handles(tuple(Qudit(self, p, i) for i, p in enumerate(layout)))

    def _frequency_arrays(self) -> tuple[Any, Any]:
        if self._arrays is None:
            import numpy as np

            target = self._description
            model = target.model if isinstance(target.model, SimulationModel) else SimulationModel()
            qudits = [target.qudits[p] for p in self.layout]
            levels = [model.levels_of(p) for p in self.layout]
            f01 = np.array([q._frame_hz for q in qudits], dtype=float)
            energies = np.full((len(qudits), max(levels, default=model.levels)), np.nan)
            for row, qudit, n in zip(energies, qudits, levels, strict=False):
                row[:n] = model.energies_hz(qudit, n)
            # A transition's frequency: its upper level's energy less its
            # lower's, for [q, i, j] and [q, j, i] alike; 0.0 between levels
            # of one energy, in either order (adding 0.0 makes a -0.0 one).
            level = np.arange(energies.shape[1])
            transitions = (energies[:, None, :] - energies[:, :, None]) * np.sign(
                level - level[:, None]
            ) + 0.0
            self._arrays = (_read_only(f01), _read_only(transitions))
        return self._arrays

    # The default transition of carrier *index*, and the frequency of
    # each transition its line has a tone for: every transition of a
    # drive line, only the default of a readout line.
    def _line_tones(
        self, index: int, readout: bool
    ) -> tuple[tuple[int, int], Mapping[tuple[int, int], float]]:
        target = self._description
        model = target.model if isinstance(target.model, SimulationModel) else SimulationModel()
        default = model.default_transition
        if index < 0:
            # A runtime qudit: its own tone is bound per call, and no other.
            return default, {}
        qudit = target.qudits[index]
        if readout:
            return default, {default: self._frequency_tables(index + 1)[2][index]}
        return default, model.transitions_hz(qudit, model.levels_of(index))

    # ``_line_tones`` of a system whose couplings dress its qudits: each tone
    # of a drive line at the difference of the qudit's dressed levels, the
    # 0-1 one its dressed frequency, a higher one worked out when a program
    # first plays it (quantum_systems.base._DressedTones).
    def _dressed_line_tones(
        self, index: int, readout: bool
    ) -> tuple[tuple[int, int], Mapping[tuple[int, int], float]]:
        default, tones = QuantumSystem._line_tones(self, index, readout)
        # The tone table: made first in a variant (__getattr__ finishes it).
        self.frequencies  # noqa: B018
        if self._dressed is None or readout or index < 0:
            return default, tones
        return default, _DressedTones(self._description, index, tones)

    # The 0-1 frequency of carriers 0 to *count* - 1 (or more), and the
    # tone of each one's readout line (its readout's ``frequency_hz``, else
    # its 0-1 frequency), by carrier index, as a compiled module records
    # them, after how many carriers from 0 on the target defines: *count*
    # or more, else the first it lacks (0 Hz in the tables, which then end
    # at its last carrier). Kept (``_tables``) until a caller needs more
    # carriers: a compile makes them for its program's qudits only, not
    # for the whole quantum system.
    def _frequency_tables(self, count: int) -> tuple[int, array[float], array[float]]:
        if self._tables[0] >= count:
            return self._tables
        from array import array

        frequencies, qudits = self.frequencies, self._description.qudits
        carriers: Sequence[int] = range(count)
        try:
            table = array("d", map(frequencies.__getitem__, carriers))
        except KeyError:
            size = min(count, max(frequencies, default=-1) + 1)
            made = self._tables[1]
            if made is not None and len(made) >= size:
                return self._tables
            carriers = [carrier for carrier in sorted(frequencies) if carrier < size]
            table = array("d", bytes(8 * size))
            for carrier in carriers:
                table[carrier] = frequencies[carrier]
            # The first carrier the target lacks.
            count = next((i for i, carrier in enumerate(carriers) if i != carrier), len(carriers))
        for carrier in carriers:
            if qudits[carrier].readout is not None:
                break
        else:
            # Without a readout, a readout line plays at the 0-1 frequency.
            self._tables = (count, table, table)
            return self._tables
        from operator import attrgetter

        components = list(map(attrgetter("readout"), map(qudits.__getitem__, carriers)))
        try:
            given = list(map(attrgetter("frequency_hz"), components))
        except AttributeError:
            given = [None if r is None else r.frequency_hz for r in components]
        if not _ONLY_HZ(map(type, given)):
            _offset_readouts(self, carriers, given)
        if len(given) == len(table) and None not in given:
            readouts = array("d", given)
        else:
            readouts = array("d", table)
            for carrier, frequency in zip(carriers, given, strict=True):
                if frequency is not None:
                    readouts[carrier] = frequency
        self._tables = (count, table, readouts)
        return self._tables

    @property
    def qudit_freq(self) -> Any:
        """Read-only array; ``qudit_freq[q]`` gives qudit ``q``'s 0-1 transition
        frequency in Hz.
        """
        return self._frequency_arrays()[0]

    @property
    def transition_freq(self) -> Any:
        """Read-only array of transition frequencies in Hz.

        ``transition_freq[q, i, j]`` gives the transition between levels ``i`` and ``j`` of
        qudit ``q``. Qudits use the user's numbering; entries beyond a qudit's last modeled
        level are NaN.
        """
        return self._frequency_arrays()[1]

    def qudit(self, label: str) -> Qudit:
        """The handle of the qudit labelled *label* (``PhysicalQudit.label``), as in
        ``qs.qudit("D1")``; a ``NotFoundError`` names the labels there are."""
        qudits = self._description.qudits
        for handle in self.qudits:
            if qudits[handle.index].label == label:
                return handle
        labels = [qudits[p].label for p in self.layout if qudits[p].label]
        carrier = next((q.index for q in qudits.values() if q.label == label), None)
        if carrier is not None:
            raise NotFoundError(
                f"qudit {label!r} is carrier {carrier} of {self._description.name}, which the "
                f"layout of quantum system {self.name!r} leaves out"
            )
        close = _closest(str(label), labels)
        hint = f"did you mean {close!r}? " if close else ""
        known = f"its labels are {', '.join(labels)}" if labels else "its qudits have no labels"
        raise NotFoundError(
            f"quantum system {self.name!r} has no qudit labelled {label!r}; {hint}{known}"
        )

    def coupling(self, a: Qudit, b: Qudit) -> Coupling:
        """The coupling of the quantum system's qudits *a* and *b* (handles, in either
        order), as calibrated: ``qs.coupling(a, b).get_calibration(
        pulse.gates.CZ)`` is its CZ. A ``NotFoundError`` names the qudits *a*
        couples to."""
        for handle in (a, b):
            if handle.__class__ is not Qudit:
                raise InvalidTypeError(
                    f"QuantumSystem.coupling takes two qudit handles, as qs.qudits[i]; got "
                    f"{handle!r}"
                )
            if handle.quantum_system is not self and handle.quantum_system != self:
                raise QuantumSystemMismatchError(
                    f"{handle!r} is a qudit of {handle.quantum_system.name!r}, not of {self.name!r}"
                )
        return self._description.coupling(a.index, b.index)

    # ``line.get_calibration(gate, transition)`` of the drive (or
    # readout) line of carrier *index*, while a kernel is traced.
    def _line_calibration(
        self, index: int, readout: bool, gate: Any, transition: Any
    ) -> DragPulse | VirtualZ | ReadoutPulse:
        from .quantum_systems.base import _DRIVE_GATES, _gate_label, _gate_name, _qudit_calibration

        name = _gate_name(gate)
        if (name == "readout") is not readout:
            line, other = ("readout", "drive") if readout else ("drive", "readout")
            if name == "cz":
                fix = "it is a coupling's: qs.coupling(a, b).get_calibration(pulse.gates.CZ)"
            else:
                fix = (
                    f"it is the {other} line's: "
                    f"get_{other}_line(q).get_calibration({_gate_label(name)})"
                )
            gates = "pulse.gates.READOUT" if readout else ", ".join(map(_gate_label, _DRIVE_GATES))
            raise InvalidValueError(
                f"a {line} line has the calibrations of {gates}, not {_gate_label(name)}; {fix}"
            )
        pulse: DragPulse | VirtualZ | ReadoutPulse = _qudit_calibration(
            self._description.qudits[index], name, transition
        )
        return pulse

    @staticmethod
    def load(path: str | os.PathLike[str], /) -> QuantumSystem:
        """The quantum system of a system file, as ``as_dict`` writes one: JSON with
        its format (``cudaq-pulse.system/1``), its quantum system (a
        registered name, or every value inline), layout, calibration,
        measurement run, control system and program clock, each as data. A file
        with problems raises ``QuantumSystemDefinitionError`` listing all of them,
        each with its path in the file, its line and its column; ``python -m
        cudaq_pulse.doctor --system FILE`` runs the same checks."""
        return _from_file(path)

    @staticmethod
    def from_dict(data: Mapping[str, Any], /) -> QuantumSystem:
        """The quantum system of a system file's data, as ``as_dict`` returns it or
        ``json.load`` reads it; see ``load``."""
        return _from_data(data)

    if TYPE_CHECKING:
        # Read from the quantum system's description by __getattr__.
        @property
        def physical_qudits(self) -> Mapping[int, PhysicalQudit]: ...
        @property
        def couplings(self) -> Sequence[Coupling]: ...
        @property
        def residual_zz(self) -> Sequence[ResidualZZ]: ...
        @property
        def model(self) -> SimulationModel | type[SimulationModel] | None: ...
        @property
        def attribution(self) -> str: ...
        @property
        def anharmonicities(self) -> dict[int, float | None]: ...
        @property
        def t1_times(self) -> dict[int, float]: ...
        @property
        def t2_times(self) -> dict[int, float]: ...
        @property
        def t2_echo_times(self) -> dict[int, float | None]: ...
        @property
        def thermal_populations(self) -> dict[int, float]: ...
        @property
        def coupling_map(self) -> list[tuple[int, ...]]: ...
        def connectivity_graph(self) -> dict[int, list[int]]: ...
        def drive_amplitude_scale(self, qudit_index: int) -> float: ...
        @property
        def frequency_frame(self) -> str: ...

        # Parameter drift: read by __getattr__ without drift, and set by the
        # drifting subclass (quantum_systems.drift).
        @property
        def true_qudits(self) -> Mapping[int, PhysicalQudit]: ...
        @property
        def parameter_drift(self) -> Mapping[int, ParameterDriftModel] | None: ...
        @property
        def parameter_drift_time(self) -> Time | None: ...
        def advance(self, duration: Time) -> None: ...

    def __getattr__(self, name: str) -> Any:
        """The description's fields, properties and methods, read from the
        description the quantum system was built from when asked for:
        ``physical_qudits`` (its ``qudits``; ``qudits`` are the handles),
        ``couplings``, ``residual_zz``, ``model``, ``attribution``,
        ``anharmonicities``, ``t1_times``, ``t2_times``, ``t2_echo_times``,
        ``thermal_populations``, ``coupling_map``, ``connectivity_graph()`` and
        ``drive_amplitude_scale(i)``."""
        if name == "physical_qudits":
            return self.__dict__["_description"].qudits
        if (
            name in ("qudits", "fingerprint", "_identity", "frequencies")
            and "layout" in self.__dict__
        ):
            self._finish()
            return self.__dict__[name]
        if name in {
            "couplings",
            "residual_zz",
            "model",
            "attribution",
            "anharmonicities",
            "t1_times",
            "t2_times",
            "t2_echo_times",
            "thermal_populations",
            "coupling_map",
            "connectivity_graph",
            "drive_amplitude_scale",
            "frequency_frame",
        }:
            return getattr(self.__dict__["_description"], name)
        if name == "true_qudits":
            return self.__dict__["_description"].qudits
        if name == "parameter_drift" or name == "parameter_drift_time":
            return None
        if name == "advance":
            system = self.name

            def advance(duration: Any) -> None:
                from ._errors import UnsupportedError

                raise UnsupportedError(
                    f"quantum system {system!r} is built without parameter drift, so it has no "
                    "drift time to advance; pass parameter_drift= to pulse.quantum_system"
                )

            return advance
        raise AttributeError(
            f"'QuantumSystem' object has no attribute {name!r}", name=name, obj=self
        )

    def replace(self, **changes: Any) -> QuantumSystem:
        """A variant of the quantum system, as ``dataclasses.replace`` makes one:
        the quantum system ``pulse.quantum_system(name=..., qudits=...)`` builds
        from this one's ``name``, ``qudits`` (its ``physical_qudits``),
        ``couplings``, ``residual_zz``, ``model``, ``attribution``, ``frames``,
        ``layout``, ``control_system`` and ``clock_ghz``, with *changes* in their place:
        ``qs.replace(qudits={**qs.physical_qudits, 0: qudit})`` changes one
        qudit. The values are the calibrated ones, so the variant has no
        ``calibration`` of its own.

        ``parameter_drift`` and ``parameter_drift_seed`` give the variant
        drift, as ``pulse.quantum_system`` takes them; a variant without them
        does not drift. ``qs.replace(parameter_drift=True)`` changes nothing
        else, so it keeps the calibration and the fingerprint."""
        described = self._description
        if "parameter_drift" in changes or "parameter_drift_seed" in changes:
            from .quantum_systems.drift import _replaced

            return _replaced(self, changes)
        layout = changes.pop(
            "layout", None if self.layout == tuple(sorted(described.qudits)) else self.layout
        )
        control_system = changes.pop("control_system", self.control_system)
        # The program clock, when it is not the description's own.
        clock_ghz = changes.pop(
            "clock_ghz", None if self.clock_ghz == described.clock_ghz else self.clock_ghz
        )
        unknown = sorted(changes.keys() - _DESCRIPTION_FIELDS - {"frames"})
        if unknown:
            raise InvalidTypeError(
                f"QuantumSystem.replace() got an unexpected keyword argument {unknown[0]!r}; "
                f"it takes {', '.join(_DESCRIPTION_FIELDS[:-1])}, frames, layout, "
                "control_system, clock_ghz, parameter_drift and parameter_drift_seed"
            )
        return QuantumSystem(
            described.replace(**changes),
            name=changes.get("name", self.name),
            layout=layout,
            source=f"{self.name}.replace()",
            control_system=control_system,
            clock_ghz=clock_ghz,
            _variant=True,
        )

    def as_dict(self) -> dict[str, Any]:
        """The quantum system as JSON-serializable data, which ``QuantumSystem.from_dict``
        reads back: its format (``cudaq-pulse.system/1``), the version that
        wrote it, its quantum system (a registered name, or every value
        inline), layout and calibration, and its control system."""
        from ._system import _device_data

        return _device_data(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuantumSystem) and other._identity == self._identity

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    # Read-only: a copy is the quantum system itself.
    def __copy__(self) -> QuantumSystem:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> QuantumSystem:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import device_reduce

        return device_reduce(self)

    def __repr__(self) -> str:
        return f"<QuantumSystem {self.name} ({len(self.layout)} qudits)>"


# The types of readout frequencies that are frequencies in Hz (or none
# given), and whether some types are all among them: any other takes
# _offset_readouts, which adds a float type it meets (numpy.float64).
_HZ_TYPES = {float, int, type(None)}
_ONLY_HZ = _HZ_TYPES.issuperset


# Refuse a readout frequency of *system* that is an ``Offset`` (or as a
# system file holds one) when its qudits give their frequencies in Hz:
# it would be read as a frequency in Hz. *given* are the readout
# frequencies of *carriers*. A system whose qudits give offsets from a
# shared reference checked its readouts when it was built.
def _offset_readouts(system: QuantumSystem, carriers: Sequence[int], given: Sequence[Any]) -> None:
    from .quantum_systems.base import Offset

    kinds = set(map(type, given))
    if Offset not in kinds and not any(issubclass(kind, Mapping) for kind in kinds):
        # Numbers of other types: the float types among them are looked at
        # once, here and where a system file is written.
        from . import _system

        floats = [k for k in kinds if issubclass(k, float) and not issubclass(k, Offset)]
        _HZ_TYPES.update(floats)
        _system._HZ_TYPES.update(floats)
        return
    target = system._description
    if target._frame != "absolute":
        return
    for carrier, frequency in zip(carriers, given, strict=True):
        if frequency.__class__ is Offset or isinstance(frequency, Mapping):
            raise InvalidValueError(
                f"the readout of qudit {carrier} of quantum system {target.name!r} gives its "
                f"frequency_hz as {frequency!r}, an offset from a shared reference, and the "
                "system's qudits give their frequencies in Hz; an Offset readout frequency "
                "needs qudits that give Offsets from the same reference"
            )


# ``_offset_readouts`` for the system file of *quantum_system*, whose
# qudits are *target*'s: the file would hold an ``Offset`` as its number.
def _offset_readouts_in_file(quantum_system: QuantumSystem, target: _Description) -> None:
    carriers = sorted(target.qudits)
    readouts = [target.qudits[carrier].readout for carrier in carriers]
    given = [None if readout is None else readout.frequency_hz for readout in readouts]
    _offset_readouts(quantum_system, carriers, given)


# The quantum system of *target*, registered as *spec* (``None`` for a
# ``_Description``), with *calibration* applied and *clock_ghz* (if
# given) its program clock.
def _calibrated(
    target: _Description,
    spec: str | None,
    source: str,
    layout: Sequence[int] | None,
    calibration: Any,
    control_system: ControlSystem | None = None,
    clock_ghz: float | None = None,
) -> QuantumSystem:
    base = target
    if calibration is not None:
        target = calibration._apply(target)
    elif spec == "transmon_surface17":
        from .quantum_systems.calibration import shipped_calibration

        calibration = shipped_calibration("transmon_surface17")
    name = target.name if spec is None else spec
    made = QuantumSystem(
        target,
        name=name,
        layout=layout,
        source=source,
        calibration=calibration,
        control_system=control_system,
        clock_ghz=clock_ghz,
    )
    made._spec = spec
    made._base_target = base
    return made


# The description of the registered quantum system *spec* names.
def _target(spec: Any, problems: list[Problem], key: str = "name") -> _Description | None:
    from .quantum_systems._registry import available_quantum_systems, make_quantum_system

    if spec is None:
        problems.append(
            Problem(
                "is missing",
                path=key,
                fix="give a registered name ("
                + ", ".join(available_quantum_systems())
                + '), or "qudits" to build one',
            )
        )
        return None
    if not isinstance(spec, str):
        problems.append(
            Problem(
                f"must be a registered name, got {type(spec).__name__}",
                path=key,
                value=spec,
            )
        )
        return None
    try:
        return make_quantum_system(spec)
    except (UnknownQuantumSystemError, ValueError) as error:
        problems.append(Problem(Exception.__str__(error), path=key, value=spec))
        return None


# *layout* as carrier indices, checked against *target* when it is known.
def _layout(
    layout: Any, target: _Description | None, name: str, problems: list[Problem]
) -> tuple[int, ...] | None:
    if isinstance(layout, (str, bytes)) or not isinstance(layout, Sequence):
        problems.append(
            Problem(
                f"must be a list of carrier indices, got {type(layout).__name__}",
                path="layout",
                value=layout,
            )
        )
        return None
    # The last index after the loop; it stays 0 for an empty layout, which
    # then fails the check that every entry named a new carrier.
    k = 0
    carriers: list[int] = []
    first: dict[int, int] = {}
    known = target.qudits if target is not None else None
    for k, entry in enumerate(layout):
        path = f"layout[{k}]"
        index = getattr(type(entry), "__index__", None)
        if index is None or isinstance(entry, bool):
            problems.append(
                Problem(f"must be a carrier index, got {entry!r}", path=path, value=entry)
            )
            continue
        carrier: int = index(entry)
        if known is not None and carrier not in known:
            problems.append(
                Problem(
                    f"carrier {carrier} is not in quantum system {name!r} (it has "
                    f"{_ranges(known)})",
                    path=path,
                    value=carrier,
                )
            )
        elif carrier in first:
            problems.append(
                Problem(
                    f"carrier {carrier} is already named by layout[{first[carrier]}]",
                    path=path,
                    value=carrier,
                    fix="each qudit names a different carrier",
                )
            )
        else:
            first[carrier] = k
        carriers.append(carrier)
    return tuple(carriers) if len(first) > k else _empty(layout, problems)


# No layout: the problem of an empty one, unless its entries had others.
def _empty(layout: Sequence[Any], problems: list[Problem]) -> tuple[int, ...] | None:
    if len(layout) == 0:
        problems.append(
            Problem(
                "must name at least one carrier, got an empty list",
                path="layout",
                value=layout,
                fix="leave layout out for every carrier",
            )
        )
    return None


# The config's calibration: a Calibration, or a shipped name or a path,
# for the config's ``measurement_run`` (the file's default if absent).
def _calibration(config: Mapping[str, Any], problems: list[Problem]) -> Any:
    from .quantum_systems.calibration import Calibration

    given = config.get("calibration")
    measurement_run = config.get("measurement_run")
    if given is None:
        if measurement_run is not None:
            problems.append(
                Problem(
                    f"{measurement_run!r} is given without a calibration",
                    path="measurement_run",
                    value=measurement_run,
                    fix=(
                        'add "calibration", a shipped name or a file path, '
                        'or remove "measurement_run"'
                    ),
                )
            )
        return None
    if isinstance(given, Calibration):
        if measurement_run is not None and measurement_run != given.measurement_run:
            problems.append(
                Problem(
                    f"asks for measurement run {measurement_run!r} of a calibration loaded for "
                    f"{given.measurement_run!r}",
                    path="measurement_run",
                    value=measurement_run,
                    fix=f"pass Calibration.load(..., measurement_run={measurement_run!r})",
                )
            )
            return None
        return given
    if isinstance(given, Mapping):
        try:
            return Calibration(given, name="inline", measurement_run=measurement_run)
        except (KeyError, TypeError, ValueError) as error:
            problems.append(
                Problem(f"is not calibration data: {Exception.__str__(error)}", path="calibration")
            )
            return None
    if not isinstance(given, (str, os.PathLike)):
        problems.append(
            Problem(
                "must be a shipped calibration name, the path of a calibration file, "
                f"calibration data or a Calibration, got {type(given).__name__}",
                path="calibration",
                value=given,
            )
        )
        return None
    try:
        return Calibration.load(given, measurement_run=measurement_run)
    except NotFoundError as error:
        problems.append(Problem(str(error), path="calibration", value=given))
    except ValueError as error:
        import json

        if isinstance(error, json.JSONDecodeError):
            problems.append(
                Problem(f"the calibration file is not valid JSON: {error}", path="calibration")
            )
        else:
            problems.append(Problem(str(error), path="measurement_run", value=measurement_run))
    return None


# The quantum system of *config*, or ``None`` after adding its problems in the
# order of the config's keys. ``pulse.quantum_system`` gives the system it built
# from its keywords as the config's ``"name"``.
def _definition(config: Mapping[str, Any], source: str, problems: list[Problem]) -> _MaybeSystem:
    found = len(problems)
    quantum_system = _checked(config, source, problems)
    order = {str(key): i for i, key in enumerate(config)}
    problems[found:] = sorted(
        problems[found:], key=lambda p: order.get(p.path.split("[")[0].split(".")[0], -1)
    )
    return quantum_system


def _checked(config: Mapping[str, Any], source: str, problems: list[Problem]) -> _MaybeSystem:
    for key in sorted(set(config) & _RESERVED_CONFIG):
        problems.append(
            Problem(
                "is reserved for a model of where carriers are and how they move, "
                "which is not available yet",
                path=key,
                fix=f"remove {key!r}",
            )
        )
    for key in sorted(set(config) - set(_CONFIG_KEYS) - _RESERVED_CONFIG, key=str):
        close = _closest(str(key), _CONFIG_KEYS)
        problems.append(
            Problem(
                "is not a key of a system file",
                path=str(key),
                value=key,
                fix=f"did you mean {close!r}?"
                if close
                else "the keys are " + ", ".join(_CONFIG_KEYS),
            )
        )
    spec = config.get("name")
    described = spec if spec.__class__ in _DESCRIPTIONS else None
    built = described is not None or "qudits" in config
    if not built and config.keys() & _BUILT_KEYS:
        for key in sorted(config.keys() & _BUILT_KEYS):
            problems.append(
                Problem(
                    'builds a quantum system, with "qudits"; a registered name gives its own',
                    path=key,
                    fix='give "qudits" to build one, or remove ' + repr(key),
                )
            )
    if built:
        spec = None
    control = config.get("control_system")
    # A quantum system by name or built from keywords, without a format or a
    # control system, needs none of the system file machinery.
    if described is not None and control is None:
        target: _Description | None = described
    elif "format" in config or built or control is not None:
        from ._system import _config_systems

        target, control = _config_systems(config, source, problems, described)
    else:
        target = _target(spec, problems)
    name = spec if isinstance(spec, str) else getattr(target, "name", "")
    layout = config.get("layout")
    if layout is not None:
        layout = _layout(layout, target, name, problems)
    calibration = _calibration(config, problems)
    clock_ghz = config.get("clock_ghz")
    if clock_ghz is not None or (control is not None and control._needs_clock):
        from ._clocks import _program_clock

        problems.extend(
            _program_clock(control, clock_ghz, 0.0 if target is None else target.clock_ghz)[1]
        )
    frames = config.get("frames", "dressed")
    if target is not None and frames != target.frames:
        if frames.__class__ is str and frames in _DRIVE_FRAMES:
            target = target.replace(frames=frames)
        else:
            problems.append(
                Problem(
                    f'is {frames!r}; it is "dressed" (the default: a system with couplings '
                    'drives each qudit at its dressed frequency) or "bare"',
                    path="frames",
                )
            )
    if problems or target is None:
        return None
    try:
        return _calibrated(
            target,
            spec if isinstance(spec, str) else None,
            source,
            layout,
            calibration,
            control,
            clock_ghz,
        )
    except (KeyError, ValueError) as error:
        problems.append(
            Problem(
                f"does not fit quantum system {name!r}: {Exception.__str__(error)}",
                path="calibration",
            )
        )
        return None


# The line and column of each value of the JSON *text*, by its path.
def _json_positions(text: str) -> dict[str, tuple[int, int]]:
    import json

    decoder = json.JSONDecoder()
    found: dict[str, int] = {}

    def skip(i: int) -> int:
        while text[i] in " \t\r\n":
            i += 1
        return i

    def value(i: int, path: str) -> int:
        i = skip(i)
        found[path] = i
        opening = text[i]
        if opening not in "{[":
            return decoder.raw_decode(text, i)[1]
        closing = "}" if opening == "{" else "]"
        i = skip(i + 1)
        n = 0
        while text[i] != closing:
            if opening == "{":
                key, i = decoder.raw_decode(text, i)
                i = skip(i) + 1
                i = value(i, f"{path}.{key}" if path else key)
            else:
                i = value(i, f"{path}[{n}]")
                n += 1
            i = skip(i)
            if text[i] == ",":
                i = skip(i + 1)
        return i + 1

    try:
        value(0, "")
    except (IndexError, ValueError):
        pass
    return {
        path: (text.count("\n", 0, i) + 1, i - text.rfind("\n", 0, i)) for path, i in found.items()
    }


def _located(problems: list[Problem], file: str, text: str) -> list[Problem]:
    positions = _json_positions(text)
    located = []
    for problem in problems:
        # A path the file has no value at (a missing key) points at the
        # nearest value that holds it.
        path = problem.path
        while path and path not in positions:
            path = path[: max(path.rfind("."), path.rfind("["), 0)]
        line, column = positions.get(path, (None, None))
        located.append(dataclasses.replace(problem, file=file, line=line, column=column))
    return located


def quantum_system(
    name: str | None = None,
    *,
    qudits: Mapping[int, PhysicalQudit] | None = None,
    couplings: Sequence[Coupling] = (),
    residual_zz: Sequence[ResidualZZ] = (),
    model: SimulationModel | type[SimulationModel] | None = _TRANSMON,
    attribution: str = "",
    layout: Sequence[int] | None = None,
    calibration: Calibration | str | os.PathLike[str] | None = None,
    measurement_run: str | None = None,
    control_system: ControlSystem | None = None,
    clock_ghz: float | None = None,
    parameter_drift: bool | ParameterDriftModel | Mapping[int, ParameterDriftModel] = False,
    parameter_drift_seed: int | None = None,
    frames: str = "dressed",
) -> QuantumSystem:
    """A quantum system: a registered one by name, or one built from its qudits.

    ``quantum_system("transmon_surface17")`` is a registered quantum system, with
    every carrier in order (a name takes its argument after a colon, as in
    ``"transmon_lattice:4"``); ``cudaq_pulse.quantum_systems.available_quantum_systems()``
    lists the names. A string is always a name: ``QuantumSystem.load(path)`` reads
    a system file, and ``QuantumSystem.from_dict(data)`` its data. ``quantum_system()`` is
    the process quantum system, named by ``CUDAQ_PULSE_QUANTUM_SYSTEM`` (``python -m
    cudaq_pulse --quantum-system NAME`` sets it); without one it raises
    ``NoProcessQuantumSystemError``.

    ``quantum_system(name="chip", qudits={0: PhysicalQudit(...), ...})`` builds
    one: ``qudits`` are its ``PhysicalQudit`` carriers by index, ``couplings``
    and ``residual_zz`` its ``Coupling`` and ``ResidualZZ`` entries, ``model``
    how the simulator treats its qudits and how many levels each has (a
    ``SimulationModel`` class or instance, ``TransmonModel`` by default,
    ``TransmonModel(levels={0: 4})`` to give qudit 0 four levels, or ``None``
    for a system with no simulation model) and ``attribution`` where its
    values come from. Its couplings and residual ZZ entries are checked, and
    a problem is at its path (``couplings[3].qudits[1]``). These keywords
    build a system, so a registered name takes none of them.

    ``layout[i]`` is the carrier that qudit ``i`` names; by default every
    carrier of the quantum system, in order. ``calibration`` is a
    ``Calibration``, the name of a shipped one or the path of a calibration
    file, and ``measurement_run`` one of its measurement runs; the calibration's couplings
    replace the system's, and ``QuantumSystem.calibration`` holds it. The
    ``transmon_surface17`` quantum system uses the shipped ``transmon_surface17`` calibration by
    default. ``control_system`` is a ``ControlSystem``, whose lines must be
    carriers of the quantum system. ``clock_ghz`` is the program clock
    kernels count cycles of (``QuantumSystem.clock_ghz``); by default the fastest
    controller clock when the others divide it, else the quantum system's
    (2 GHz for one built here).
    Controller clocks that do not divide evenly into one need it, and the
    problem lists the clocks that work.

    ``frames`` is where the drive tones of a system with couplings play:
    ``"dressed"`` (the default) at each qudit's 0-1 frequency with the
    couplings included (``Qudit.dressed_frequency_hz``), so that a qudit
    does not precess at its neighbours' shift; ``"bare"`` at its
    ``frequency_hz``. A system without couplings is the same either way.

    ``parameter_drift`` makes the simulated system's qudit parameters drift
    (``cudaq_pulse.quantum_systems.drift``): ``True`` for the default
    ``ParameterDrift()`` on every qudit, a drift model for that model on
    every qudit, or ``{qudit: model}`` for only those qudits (keyed as
    ``physical_qudits``). Compiling reads only the calibrated values, so the
    system equals its non-drifting twin; ``simulate`` plays its true values,
    ``QuantumSystem.true_qudits``. ``parameter_drift_seed`` (an int) makes
    the drift repeat; by default each system drifts its own way.

    Arguments with problems raise ``QuantumSystemDefinitionError`` listing all of
    them, each at the argument it is in.
    """
    if qudits is not None:
        if name.__class__ is not str:
            _not_a_name(name)
            raise InvalidTypeError(
                "pulse.quantum_system(qudits=...) builds a quantum system and needs its name=, "
                f"got {type(name).__name__}"
            )
        described: _Description = _Description(
            name=name,
            qudits=qudits,
            couplings=couplings,
            residual_zz=residual_zz,
            model=model,
            attribution=attribution,
            # A clock that is no positive number is the program clock's problem.
            **(
                {"clock_ghz": clock_ghz}
                if clock_ghz is not None
                and isinstance(clock_ghz, (int, float))
                and not isinstance(clock_ghz, bool)
                and 0 < clock_ghz < float("inf")
                else {}
            ),
        )
    elif name.__class__ is str and name.partition(":")[0] not in _registry():
        _unknown_target(name)
    elif couplings or residual_zz or attribution or model is not _TRANSMON:
        _not_a_name(name)
        given = "couplings" if couplings else "residual_zz" if residual_zz else "attribution"
        raise InvalidTypeError(
            f"pulse.quantum_system's {'model' if model is not _TRANSMON else given} builds a "
            "quantum system with qudits=; a registered name gives its own"
        )
    elif name.__class__ is str:
        if (
            layout is None
            and calibration is None
            and measurement_run is None
            and control_system is None
            and clock_ghz is None
            and not parameter_drift
            and parameter_drift_seed is None
            and frames == "dressed"
        ):
            return _calibrated(
                make_quantum_system(name), name, f"quantum system {name}", None, None
            )
    elif name is None:
        if (
            layout is None
            and calibration is None
            and measurement_run is None
            and control_system is None
            and clock_ghz is None
            and not parameter_drift
            and parameter_drift_seed is None
            and frames == "dressed"
        ):
            from .quantum_systems._registry import current_quantum_system

            return current_quantum_system()
        from .quantum_systems._registry import ENV_VAR, _no_quantum_system_message

        name = os.environ.get(ENV_VAR)
        if not name:
            raise NoProcessQuantumSystemError(_no_quantum_system_message())
        if name.partition(":")[0] not in _registry():
            _unknown_target(name)
    elif isinstance(name, str):
        if name.partition(":")[0] not in _registry():
            _unknown_target(name)
    else:
        _not_a_name(name)
        fix = (
            "; QuantumSystem.from_dict(data) reads a system file's data"
            if isinstance(name, Mapping)
            else ""
        )
        raise InvalidTypeError(
            "pulse.quantum_system takes a registered name, or name= and qudits= to build "
            f"one, got {type(name).__name__}{fix}"
        )
    if isinstance(control_system, Mapping) or isinstance(calibration, Mapping):
        given = "control_system" if isinstance(control_system, Mapping) else "calibration"
        raise InvalidTypeError(
            f"pulse.quantum_system's {given} is a "
            + ("ControlSystem" if given == "control_system" else "Calibration, a name or a path")
            + ", not data; QuantumSystem.from_dict(data) reads a system file's data"
        )
    arguments = {
        "name": name if qudits is None else described,
        "layout": layout,
        "calibration": calibration,
        "measurement_run": measurement_run,
        "control_system": control_system,
        "clock_ghz": clock_ghz,
    }
    defined: dict[str, Any] = {k: v for k, v in arguments.items() if v is not None}
    if frames != "dressed":
        defined["frames"] = frames
    problems: list[Problem] = []
    found = _definition(defined, "arguments", problems)
    if found is None:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in pulse.quantum_system()", problems=problems
        )
    if parameter_drift or parameter_drift_seed is not None:
        from .quantum_systems.drift import _replaced

        drift = {"parameter_drift": parameter_drift, "parameter_drift_seed": parameter_drift_seed}
        found = _replaced(found, drift)
    return found


# Raise for a QuantumSystem given where ``pulse.quantum_system`` takes a name.
def _not_a_name(name: Any) -> None:
    if isinstance(name, QuantumSystem):
        raise InvalidTypeError(
            "pulse.quantum_system takes a registered name or the keywords that build a "
            "quantum system, not a QuantumSystem; qs.replace(...) makes a variant of "
            f"one, as qs.replace(layout=[0, 1]) for {name.name!r} on its first two qudits"
        )


# Raise ``UnknownQuantumSystemError`` for *name*, which names no registered
# quantum system, naming the way to read a file when it looks like the path of one.
def _unknown_target(name: str) -> None:
    if os.sep in name or name.endswith(".json") or os.path.exists(name):
        try:
            make_quantum_system(name)
        except UnknownQuantumSystemError as error:
            raise UnknownQuantumSystemError(
                f"{Exception.__str__(error)}; a string is a registered quantum system name: "
                f"to read a system file, use pulse.QuantumSystem.load({name!r})"
            ) from None
    make_quantum_system(name)


# ``QuantumSystem.from_dict``: the quantum system of a system file's data.
def _from_data(data: Any) -> QuantumSystem:
    if not isinstance(data, Mapping):
        raise InvalidTypeError(f"quantum system data must be a mapping, got {type(data).__name__}")
    problems: list[Problem] = []
    found = _definition(data, "data", problems)
    if found is None:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in the quantum system data", problems=problems
        )
    return found


# ``QuantumSystem.load``: the quantum system of a system file.
def _from_file(path: str | os.PathLike[str]) -> QuantumSystem:
    import json

    file = os.fspath(path)
    with open(file, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        problem = Problem(
            f"not valid JSON: {error.msg}", file=file, line=error.lineno, column=error.colno
        )
        raise QuantumSystemDefinitionError(
            f"1 problem in quantum system file {file}", problems=[problem]
        ) from error
    problems: list[Problem] = []
    if isinstance(data, Mapping):
        found = _definition(data, f"file {file}", problems)
    else:
        found = None
        problems.append(
            Problem(f"the file must hold a JSON object, got {type(data).__name__}", value=data)
        )
    if found is None:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in quantum system file {file}",
            problems=_located(problems, file, text),
        )
    return found
