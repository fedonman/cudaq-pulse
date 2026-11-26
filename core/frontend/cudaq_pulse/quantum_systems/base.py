# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The parts of a quantum system, architecture-neutral.

``pulse.quantum_system(name=..., qudits=..., couplings=..., model=...)``
builds a quantum system from them: its qudits' frequencies,
anharmonicities, decoherence, drive and readout, its couplings, and the
pulses its gates are calibrated for. Simulation lowers it to the module
attributes of the Pulse -> QOp lowering.
"""

from __future__ import annotations

import math
import numbers
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from itertools import chain
from operator import attrgetter, eq, is_, methodcaller
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .._errors import (
    InvalidTypeError,
    InvalidValueError,
    NotFoundError,
    Problem,
    QuantumSystemDefinitionError,
    UnsupportedError,
    _count,
)
from .models import SimulationModel, TransmonModel

if TYPE_CHECKING:
    from collections.abc import Iterable
    from dataclasses import dataclass as _dataclass_when_used
    from typing import ClassVar, Literal, NoReturn, Self, overload

    from ..gates import Gate

    _NONE: type[None]
else:
    from .._errors import _dataclass_when_used

_TRANSMON = TransmonModel()

# The default of the mapping fields: __post_init__ keeps a read-only copy of
# whatever a field is given, so every instance may share it.
if TYPE_CHECKING:
    _EMPTY: Mapping[Any, Any]
_EMPTY = MappingProxyType({})


class _Field(int):
    """A field of a component, read from its ``_values`` at this index."""

    __slots__ = ()

    def __get__(self, component: Any, owner: Any = None) -> Any:
        if component is None:
            return self
        return component._values[self]


class _Component:
    """A read-only value of named fields (``_fields``, the keywords of its
    ``__init__``), held in ``_values`` and read through ``_Field``
    descriptors, compared, hashed, pickled and printed by them.

    A plain slotted class rather than a dataclass: the shipped system makes
    one per qudit, and making a dataclass of this many fields costs as much
    as the rest of the import.
    """

    __slots__ = ("_values",)
    if TYPE_CHECKING:
        _fields: ClassVar[tuple[str, ...]]
        _values: tuple[Any, ...]
    _fields = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Component):
            return NotImplemented
        return type(other) is type(self) and other._values == self._values

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._values))

    def __repr__(self) -> str:
        defaults = type(self).__init__.__kwdefaults__ or {}
        given = ", ".join(
            f"{name}={value!r}"
            for name, value in zip(self._fields, self._values, strict=True)
            if value != defaults.get(name)
        )
        return f"{type(self).__name__}({given})"

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import rebuilt

        values: Iterable[Any] = self._values
        if not {MappingProxyType, _Calibrations}.isdisjoint(map(type, values)):
            values = map(_plain_value, values)
        return (rebuilt, (type(self), dict(zip(self._fields, values, strict=True))))

    def replace(self, **changes: Any) -> Self:
        """A copy with the fields *changes* names set to their values."""
        values = dict(zip(self._fields, self._values, strict=True))
        values.update(changes)
        return type(self)(**values)

    __replace__ = replace


class Readout(_Component):
    """How a qudit is read out: the base of the readout modalities, such as
    ``DispersiveReadout``. Simulation and emulation pick a readout's model by
    its type, and refuse a type they have no model for."""

    __slots__ = ()

    @property
    def frequency_hz(self) -> float | None:
        """The frequency its readout line plays at by default (``None``: the
        qudit's 0-1 frequency)."""
        return None

    # How long after a readout the next one must wait.
    def _depletion_ns(self) -> float:
        return 0.0


class Drive(_Component):
    """How a qudit is driven: a drive of no stated modality, and the base of
    the drive modalities, such as ``MicrowaveDrive``.

    ``amplitude_scale_rad_per_ns`` is the Rabi rate in rad/ns of a pulse of
    amplitude 1, whatever carries the drive. The simulator turns a pulse's
    amplitude into a Rabi rate by it; without it, by the scale that makes
    the qudit's ``X`` calibration a pi rotation, if that gives an amplitude;
    else 1.0. Simulation picks a drive's model by its type: it drives a
    ``Drive`` and a ``MicrowaveDrive`` so, and refuses another subclass, which
    it has no model for. A subclass with fields of its own states them all
    in ``_fields``.
    """

    __slots__ = ()
    _fields = ("amplitude_scale_rad_per_ns",)

    def __init__(self, *, amplitude_scale_rad_per_ns: float | None = None) -> None:
        object.__setattr__(self, "_values", (amplitude_scale_rad_per_ns,))

    if TYPE_CHECKING:

        @property
        def amplitude_scale_rad_per_ns(self) -> float | None:
            """The Rabi rate in rad/ns of a pulse of amplitude 1."""
            ...

    else:
        (amplitude_scale_rad_per_ns,) = map(_Field, range(1))


class DispersiveReadout(Readout):
    """A dispersive readout through a resonator: the physics of the qudit's
    readout. Every field may be left out.

    ``frequency_hz`` is the resonator's frequency, where the readout line
    plays by default; on a system whose qudits give their frequencies as
    offsets from a shared reference it is an ``Offset`` from the same
    reference, and a system whose qudits give no frequencies takes none.
    The readout model in simulation and emulation
    (``IQModel``) pulls the resonator by ``(2k - 1) dispersive_shift_hz``
    for level ``k``, with linewidth ``linewidth_hz`` (kappa / 2 pi), and
    needs both. ``error_1to0`` and ``error_0to1`` are the probabilities of
    reporting 0 for ``|1>`` and 1 for ``|0>``. The resonator empties in
    ``depletion_ns`` (else ln(100) / kappa) at measurement efficiency
    ``efficiency``.

    The readout pulse the quantum system is tuned for is the qudit's ``READOUT``
    calibration, a ``ReadoutPulse``.
    """

    __slots__ = ()
    _fields = (
        "frequency_hz",
        "dispersive_shift_hz",
        "linewidth_hz",
        "efficiency",
        "error_1to0",
        "error_0to1",
        "depletion_ns",
    )

    def __init__(
        self,
        *,
        frequency_hz: float | None = None,
        dispersive_shift_hz: float | None = None,
        linewidth_hz: float | None = None,
        efficiency: float = 0.5,
        error_1to0: float = 0.0,
        error_0to1: float = 0.0,
        depletion_ns: float | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "_values",
            (
                frequency_hz,
                dispersive_shift_hz,
                linewidth_hz,
                efficiency,
                error_1to0,
                error_0to1,
                depletion_ns,
            ),
        )

    if TYPE_CHECKING:

        @property
        def frequency_hz(self) -> float | None:
            """The resonator's frequency, where the readout line plays by default."""
            ...

        @property
        def dispersive_shift_hz(self) -> float | None:
            """The dispersive shift chi / 2 pi."""
            ...

        @property
        def linewidth_hz(self) -> float | None:
            """The resonator's effective linewidth kappa / 2 pi."""
            ...

        @property
        def efficiency(self) -> float:
            """The measurement efficiency."""
            ...

        @property
        def error_1to0(self) -> float:
            """The probability that ``|1>`` is reported 0."""
            ...

        @property
        def error_0to1(self) -> float:
            """The probability that ``|0>`` is reported 1."""
            ...

        @property
        def depletion_ns(self) -> float | None:
            """The time the resonator takes to empty after a readout."""
            ...

    else:
        # In the order of _fields.
        (
            frequency_hz,
            dispersive_shift_hz,
            linewidth_hz,
            efficiency,
            error_1to0,
            error_0to1,
            depletion_ns,
        ) = map(_Field, range(7))

    def _depletion_ns(self) -> float:
        depletion, linewidth = self._values[6], self._values[2]
        if depletion is not None:
            return float(depletion)
        if linewidth:
            return math.log(100.0) / (2 * math.pi * float(linewidth) * 1e-9)
        return 0.0


class MicrowaveDrive(Drive):
    """A microwave drive: the physics of the qudit's drive.

    Its one field is ``Drive``'s ``amplitude_scale_rad_per_ns``, the Rabi
    rate in rad/ns of a pulse of amplitude 1. The tuned pulses are the
    qudit's calibrations (``get_calibration``).
    """

    __slots__ = ()


class DragPulse(_Component):
    """The calibrated pulse of a drive-line gate (``X``, ``X90``, ``Y``,
    ``Y90``): a DRAG pulse, a Gaussian of width ``sigma_ns`` over
    ``duration_ns`` with peak ``amplitude`` and DRAG coefficient
    ``beta_ns``, at phase ``phase_rad`` (``Y`` is ``X`` at pi / 2).

    In a kernel, ``drive(line, pulse)`` plays it: ``drag(duration_ns * ns,
    amplitude, sigma_ns * ns, beta_ns * ns)``, times ``exp(i phase_rad)``
    when the phase is not 0. An ``amplitude`` of ``None`` is one not
    published: ``get_calibration`` gives the pulse the amplitude that makes
    its rotation, at the qudit's drive scale.
    """

    __slots__ = ()
    _fields = ("duration_ns", "sigma_ns", "amplitude", "beta_ns", "phase_rad")

    def __init__(
        self,
        *,
        duration_ns: float,
        sigma_ns: float,
        amplitude: float | None = None,
        beta_ns: float = 0.0,
        phase_rad: float = 0.0,
    ) -> None:
        object.__setattr__(self, "_values", (duration_ns, sigma_ns, amplitude, beta_ns, phase_rad))

    if TYPE_CHECKING:

        @property
        def duration_ns(self) -> float:
            """The pulse's length."""
            ...

        @property
        def sigma_ns(self) -> float:
            """The width of its Gaussian."""
            ...

        @property
        def amplitude(self) -> float | None:
            """Its peak amplitude (``None``: not published)."""
            ...

        @property
        def beta_ns(self) -> float:
            """Its DRAG coefficient."""
            ...

        @property
        def phase_rad(self) -> float:
            """The phase it plays at."""
            ...

    else:
        duration_ns, sigma_ns, amplitude, beta_ns, phase_rad = map(_Field, range(5))


class VirtualZ(_Component):
    """The calibration of a phase gate (``Z``, ``S``, ``T``): a virtual Z
    rotation by ``phase_rad``, which takes no time and plays no pulse.

    In a kernel, ``drive(line, z)`` plays it as ``shift_phase(line.tone,
    z.phase_rad)`` (at ``tone=`` if given): the frame of the tone advances,
    and every later pulse at that tone turns about an axis rotated by
    ``-phase_rad``, so that the qudit's state, read in the frame, is
    ``Rz(phase_rad)`` applied to the state the pulses alone make. The state
    a simulation returns is in the frame: a virtual Z after the last pulse
    does not change it (see Phase frames and virtual Z gates in the
    operations guide).
    """

    __slots__ = ()
    _fields = ("phase_rad",)

    def __init__(self, *, phase_rad: float) -> None:
        object.__setattr__(self, "_values", (phase_rad,))

    if TYPE_CHECKING:

        @property
        def phase_rad(self) -> float:
            """The rotation, the phase the tone's frame advances by."""
            ...

    else:
        phase_rad = _Field(0)


class ReadoutPulse(_Component):
    """The calibrated readout pulse of a qudit (``READOUT``): of peak
    ``amplitude`` it holds ``photon_number`` photons in the resonator (with
    no ``amplitude``, any amplitude played does), lasts ``duration_ns``,
    has edges of width ``sigma_ns`` and is integrated over
    ``integration_ns`` (else the readout's length).

    In a kernel, ``readout(line, pulse)`` plays it: ``gaussian_square`` with
    edges of 2 ``sigma_ns`` given a ``sigma_ns``, else ``square``, at
    ``amplitude`` (1.0 when not published).
    """

    __slots__ = ()
    _fields = ("duration_ns", "amplitude", "sigma_ns", "integration_ns", "photon_number")

    def __init__(
        self,
        *,
        duration_ns: float | None = None,
        amplitude: float | None = None,
        sigma_ns: float | None = None,
        integration_ns: float | None = None,
        photon_number: float = 4.0,
    ) -> None:
        object.__setattr__(
            self, "_values", (duration_ns, amplitude, sigma_ns, integration_ns, photon_number)
        )

    if TYPE_CHECKING:

        @property
        def duration_ns(self) -> float | None:
            """The pulse's length."""
            ...

        @property
        def amplitude(self) -> float | None:
            """Its peak amplitude, at which it holds ``photon_number``."""
            ...

        @property
        def sigma_ns(self) -> float | None:
            """The width of its edges."""
            ...

        @property
        def integration_ns(self) -> float | None:
            """The time the readout signal is integrated over."""
            ...

        @property
        def photon_number(self) -> float:
            """The photons in the resonator at ``amplitude``."""
            ...

    else:
        duration_ns, amplitude, sigma_ns, integration_ns, photon_number = map(_Field, range(5))


class FluxCZ(_Component):
    """The calibrated CZ of a coupling (``CZ``), made by a flux pulse: it
    takes ``duration_ns`` with ``buffer_ns`` of idle time around it, and
    its qudits interact at ``interaction_frequency_hz``; ``None`` where not
    given. Flux lines are not modeled yet: a kernel leaves its qudits idle
    for its ``duration_ns``.
    """

    __slots__ = ()
    _fields = ("duration_ns", "buffer_ns", "interaction_frequency_hz")

    def __init__(
        self,
        *,
        duration_ns: float,
        buffer_ns: float | None = None,
        interaction_frequency_hz: float | None = None,
    ) -> None:
        object.__setattr__(self, "_values", (duration_ns, buffer_ns, interaction_frequency_hz))

    if TYPE_CHECKING:

        @property
        def duration_ns(self) -> float:
            """How long the gate takes."""
            ...

        @property
        def buffer_ns(self) -> float | None:
            """The idle time before and after it."""
            ...

        @property
        def interaction_frequency_hz(self) -> float | None:
            """The frequency at which its qudits interact."""
            ...

    else:
        duration_ns, buffer_ns, interaction_frequency_hz = map(_Field, range(3))


# The values of pulse.gates.Gate (the enum loads with pulse.gates, which
# building a target does not need): the gates of a qudit's drive line, of
# its readout line and of a coupling.
_DRIVE_GATES = ("x", "x90", "y", "y90", "z", "s", "t")
_QUDIT_GATES = ("x", "x90", "y", "y90", "z", "s", "t", "readout")
_COUPLING_GATES = ("cz",)
_GATES = ("x", "x90", "y", "y90", "z", "s", "t", "readout", "cz")
if TYPE_CHECKING:
    _CALIBRATION_TYPES: Mapping[str, Any]
_CALIBRATION_TYPES = MappingProxyType(
    {
        "x": DragPulse,
        "x90": DragPulse,
        "y": DragPulse,
        "y90": DragPulse,
        "z": VirtualZ,
        "s": VirtualZ,
        "t": VirtualZ,
        "readout": ReadoutPulse,
        "cz": FluxCZ,
    }
)
# The rotation of each drive-line gate, pi over this: an unpublished X's
# amplitude makes it, and an unpublished Z, S or T is a virtual Z by it.
_PI_DIVISORS = MappingProxyType({"x": 1, "x90": 2, "y": 1, "y90": 2, "z": 1, "s": 2, "t": 4})
_DEFAULT_TRANSITION = (0, 1)


def _gate_label(name: str) -> str:
    return f"pulse.gates.{name.upper()}"


# The value of *gate*, a ``pulse.gates.Gate``; the caller checks where
# it belongs. A string that is no gate's is matched against *near*.
def _gate_name(gate: Any, near: Sequence[str] = _GATES) -> str:
    if isinstance(gate, str):
        if gate in _GATES:
            return gate if gate.__class__ is str else str.__str__(gate)
        from .._errors import _closest

        close = _closest(str.__str__(gate), near)
        hint = f"did you mean {_gate_label(close)}? " if close else ""
        raise InvalidValueError(
            f"{gate!r} is not a gate; {hint}the gates are {', '.join(map(_gate_label, _GATES))}"
        )
    raise InvalidTypeError(f"a gate is one of {', '.join(map(_gate_label, _GATES))}, got {gate!r}")


def _transition(transition: Any) -> tuple[int, int]:
    if (
        isinstance(transition, Sequence)
        and not isinstance(transition, str)
        and len(transition) == 2
    ):
        lower, upper = transition
        if lower.__class__ is int and upper.__class__ is int and 0 <= lower < upper:
            return _DEFAULT_TRANSITION if (lower, upper) == _DEFAULT_TRANSITION else (lower, upper)
    raise InvalidValueError(
        f"a transition is (lower, upper) levels, 0 <= lower < upper, as (1, 2); got {transition!r}"
    )


def _calibration_key(name: str, transition: tuple[int, int]) -> Any:
    return name if transition == _DEFAULT_TRANSITION else (name, transition)


_COMPONENT_VALUES = attrgetter("_values")


class _Calibrations(dict[Any, Any]):
    """Checked calibrations by gate: read-only, and with what a quantum system's
    fingerprint takes of them (``_content``), made once however many qudits
    share them."""

    __slots__ = ("_content",)

    def __init__(self, checked: Mapping[Any, Any]) -> None:
        dict.__init__(self, checked)
        # A gate's calibration has one type, so the values say what it is.
        self._content = (tuple(self), tuple(map(_COMPONENT_VALUES, self.values())))

    def _read_only(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise InvalidTypeError("calibrations are read-only; replace() makes a changed copy")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _read_only
    __ior__ = _read_only

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self._content)

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Calibrations, (dict(self),))


# The calibrations of a qudit or coupling that gives none.
_NO_CALIBRATIONS = _Calibrations({})


# *given*, calibrations by gate (or by ``(gate, transition)``), as a
# checked read-only mapping keyed by gate value, or ``(value,
# transition)`` for a transition other than (0, 1).
def _calibrations(given: Any, owner: str, allowed: Sequence[str]) -> Mapping[Any, Any]:
    if given.__class__ is _Calibrations:
        return given
    if not isinstance(given, Mapping):
        raise InvalidTypeError(
            f"the calibrations of {owner} map gates to calibrations, as "
            f"{{pulse.gates.{allowed[0].upper()}: ...}}; got {type(given).__name__}"
        )
    checked: dict[Any, Any] = {}
    for key, value in given.items():
        gate, transition = key if key.__class__ is tuple else (key, _DEFAULT_TRANSITION)
        name = _gate_name(gate, allowed)
        if name not in allowed:
            where = "a coupling's" if name in _COUPLING_GATES else "a qudit's"
            raise InvalidValueError(
                f"{_gate_label(name)} is {where} calibration, not one of {owner}; "
                f"{owner} has calibrations of {', '.join(map(_gate_label, allowed))}"
            )
        expected = _CALIBRATION_TYPES[name]
        if value.__class__ is not expected:
            raise InvalidTypeError(
                f"the {_gate_label(name)} calibration of {owner} must be a "
                f"{expected.__name__}, got {type(value).__name__}"
            )
        checked[_calibration_key(name, _transition(transition))] = value
    return _Calibrations(checked)


def _json_value(value: Any) -> bool:
    if value is None or value.__class__ in (str, int, float, bool):
        return True
    if value.__class__ in (list, tuple):
        return all(map(_json_value, value))
    if isinstance(value, Mapping):
        return all(k.__class__ is str and _json_value(v) for k, v in value.items())
    return False


def _metadata(given: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(given, Mapping) or not all(k.__class__ is str for k in given):
        raise InvalidTypeError(
            f"the metadata of qudit {index} is a mapping with str keys, got {given!r}"
        )
    for key, value in given.items():
        if not _json_value(value):
            raise InvalidTypeError(
                f"the metadata of qudit {index} holds JSON values only (str, int, float, bool, "
                f"None, lists and str-keyed mappings of them); {key!r} is {type(value).__name__}"
            )
    return MappingProxyType(dict(given))


# The area of *pulse*'s Gaussian envelope at amplitude 1.
def _area_ns(pulse: DragPulse, owner: str) -> float:
    duration, sigma = pulse._values[0], pulse._values[1]
    if not (_finite(duration) and duration > 0 and _finite(sigma) and sigma > 0):
        raise InvalidValueError(
            f"the DragPulse of {owner} needs a positive duration_ns and sigma_ns to give its "
            f"amplitude, got {duration!r} and {sigma!r}"
        )
    return float(
        sigma * math.sqrt(2.0 * math.pi) * math.erf(duration / (2.0 * math.sqrt(2.0) * sigma))
    )


def _qudit_name(qudit: PhysicalQudit) -> str:
    return f"qudit {qudit.index} ({qudit.label})" if qudit.label else f"qudit {qudit.index}"


def _published(calibrations: Mapping[Any, Any]) -> str:
    if not calibrations:
        return "it publishes none"
    names = [
        _gate_label(k) if k.__class__ is not tuple else f"{_gate_label(k[0])} {k[1]}"
        for k in calibrations
    ]
    return "it publishes " + ", ".join(names)


# *calibrations*, which are keyed by gate value, keyed by
# ``pulse.gates.Gate`` (with the transition, for one other than (0, 1)):
# read-only.
def _by_gate(calibrations: Mapping[Any, Any]) -> Mapping[Any, Any]:
    if not calibrations:
        return _EMPTY
    from ..gates import Gate

    return MappingProxyType(
        {
            (Gate(key) if key.__class__ is str else (Gate(key[0]), key[1])): value
            for key, value in calibrations.items()
        }
    )


# ``PhysicalQudit.get_calibration``: the calibration *gate* of *qudit*, derived as
# ``pulse.gates`` describes where it is not published.
def _qudit_calibration(qudit: PhysicalQudit, gate: Any, transition: Any) -> Any:
    name = _gate_name(gate, _QUDIT_GATES)
    at = _transition(transition)
    if name == "cz":
        raise InvalidValueError(
            f"{_gate_label('cz')} is calibrated on a coupling, not on {_qudit_name(qudit)}: "
            "qs.coupling(a, b).get_calibration(pulse.gates.CZ)"
        )
    found: Any = qudit._calibrations.get(_calibration_key(name, at))
    if found is not None:
        if found.__class__ is DragPulse and found._values[2] is None:
            return found.replace(
                amplitude=math.pi
                / _PI_DIVISORS[name]
                / (_drive_scale(qudit) * _area_ns(found, _qudit_name(qudit)))
            )
        return found
    if _CALIBRATION_TYPES[name] is VirtualZ:
        return VirtualZ(phase_rad=math.pi / _PI_DIVISORS[name])
    if name == "x90":
        x = _qudit_calibration(qudit, "x", at)
        return x.replace(amplitude=x._values[2] / 2)
    if name in ("y", "y90"):
        base = _qudit_calibration(qudit, "x" if name == "y" else "x90", at)
        return base.replace(phase_rad=base._values[4] + math.pi / 2)
    which = _gate_label(name) if at == _DEFAULT_TRANSITION else f"{_gate_label(name)} {at}"
    raise NotFoundError(
        f"{_qudit_name(qudit)} publishes no {which} calibration; {_published(qudit._calibrations)}"
    )


# The drives whose first field is their Rabi rate per unit amplitude.
_SCALED_DRIVES = (MicrowaveDrive, Drive)
# The kind a system file writes for each drive it holds.
_DRIVE_KINDS: Mapping[str, type[Drive]] = MappingProxyType(
    {"microwave": MicrowaveDrive, "drive": Drive}
)


# The Rabi rate in rad/ns of amplitude 1 on *qudit*'s drive, if it gives one, else 1.0.
def _drive_scale(qudit: PhysicalQudit) -> float:
    drive = qudit.drive
    if drive.__class__ in _SCALED_DRIVES:
        explicit = drive._values[0]
        if explicit is not None:
            return float(explicit)
    return 1.0


class _QuditFields:
    """The fields of a ``PhysicalQudit``, which its ``__init__`` stores through
    this class: the same layout, without PhysicalQudit's read-only ``__setattr__``."""

    __slots__ = (
        "index",
        "frequency_hz",
        "t1_us",
        "t2_star_us",
        "anharmonicity_hz",
        "label",
        "drive",
        "readout",
        "t2_echo_us",
        "thermal_population",
        "_calibrations",
        "metadata",
    )

    index: int
    # None on a qudit made without one (PhysicalQudit(0, None, ...)). Typed
    # as the float every other qudit has, so that programs which compute
    # with it check as before.
    frequency_hz: float
    t1_us: float
    t2_star_us: float
    anharmonicity_hz: float | None
    label: str
    drive: Drive | None
    readout: Readout | None
    # Hahn-echo T2 (us), which the simulation dephases at on request, and
    # the excited-state population at equilibrium (a fraction), from which
    # it starts and towards which T1 relaxes.
    t2_echo_us: float | None
    thermal_population: float
    # What calibrations= gives, which get_calibration reads.
    _calibrations: Mapping[Any, DragPulse | VirtualZ | ReadoutPulse]
    metadata: Mapping[str, Any]
    if TYPE_CHECKING:
        # The frequency its frame turns at, which the models, the simulator
        # and the exports compute with: ``frequency_hz``, read from the same
        # slot under this name, or 0.0 for a qudit a system holds without one.
        _frame_hz: float
        # The absolute frequency a circuit model is evaluated at (a
        # Calibration's exchange): ``frequency_hz`` under a third name, which
        # a qudit a system holds without an absolute frequency refuses.
        _circuit_hz: float


_QuditFields._frame_hz = _QuditFields.__dict__["frequency_hz"]
_QuditFields._circuit_hz = _QuditFields.__dict__["frequency_hz"]

# A qubit's fields but its metadata, in the order of PhysicalQudit's arguments.
_QUDIT_STATE = attrgetter(*_QuditFields.__slots__[:-1])


class PhysicalQudit(_QuditFields):
    """One qudit of a quantum system.

    ``frequency_hz`` is its 0-1 frequency, more than 0. ``None`` says the
    system does not state it: every frequency of its programs is then an
    offset from the qudit's own carrier (a tone at 0 Hz is on resonance,
    ``set_frequency(tone, 1e6)`` is 1 MHz above it), as on a control stack
    that owns the carrier and takes offsets only; a system whose qudits
    give none declares no couplings, which act by the difference of two
    qudits' frequencies. ``Offset(hz)`` gives the frequency as an offset
    from one reference the system's qudits share and the system does not
    state: every difference is then known, and couplings compute on the
    offsets. A system's qudits give their frequencies all one way.
    ``anharmonicity_hz`` is where a level above the first sits off evenly
    spaced levels; ``None`` (the default) is a qudit that gives none, which
    a ``TransmonModel`` of more than two levels refuses, naming the qudit.
    ``drive`` and ``readout`` are its physics: how it is driven and read
    out, each by the component of its modality (``MicrowaveDrive``,
    ``DispersiveReadout``), or ``None`` when not given. ``calibrations=``
    gives the pulses it is tuned for, by gate: ``{pulse.gates.X:
    DragPulse(...), pulse.gates.READOUT: ReadoutPulse(...)}`` (a key
    ``(gate, (lower, upper))`` for a transition other than (0, 1)),
    ``calibrations`` gives them as published, and ``get_calibration`` reads
    them, deriving what is not given. ``metadata`` is the user's own data
    (JSON values), which the library never reads, and which does not make
    a quantum system differ; values the library knows are fields, never metadata.
    A qubit is read-only: ``replace`` makes a changed copy.
    """

    __slots__ = ()
    __match_args__ = ("index", "frequency_hz", "t1_us", "t2_star_us")

    def __init__(
        self,
        index: int,
        frequency_hz: float | None,
        t1_us: float,
        t2_star_us: float,
        *,
        anharmonicity_hz: float | None = None,
        label: str = "",
        drive: Drive | None = None,
        readout: Readout | None = None,
        t2_echo_us: float | None = None,
        thermal_population: float = 0.0,
        calibrations: Mapping[Any, DragPulse | VirtualZ | ReadoutPulse] = _NO_CALIBRATIONS,
        metadata: Mapping[str, Any] = _EMPTY,
    ) -> None:
        if calibrations.__class__ is not _Calibrations:
            calibrations = _calibrations(calibrations, f"qudit {index}", _QUDIT_GATES)
        if metadata is not _EMPTY:
            metadata = _metadata(metadata, index)
        cls = self.__class__
        object.__setattr__(self, "__class__", _QuditFields)
        self.index = index
        self.frequency_hz = frequency_hz  # type: ignore[assignment]
        self.t1_us = t1_us
        self.t2_star_us = t2_star_us
        self.anharmonicity_hz = anharmonicity_hz
        self.label = label
        self.drive = drive
        self.readout = readout
        self.t2_echo_us = t2_echo_us
        self.thermal_population = thermal_population
        self._calibrations = calibrations
        self.metadata = metadata
        self.__class__ = cls

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in _QuditFields.__slots__)

    def __hash__(self) -> int:
        return hash((self.index, self.frequency_hz, self.label))

    def __repr__(self) -> str:
        shown = ", ".join(
            f"{n.lstrip('_')}={_plain_value(getattr(self, n))!r}" for n in _QuditFields.__slots__
        )
        return f"{type(self).__name__}({shown})"

    def __reduce__(self) -> tuple[Any, ...]:
        # Most fields are keyword-only: a new instance takes them as state.
        state = _QUDIT_STATE(self)
        if self.metadata:
            state = (*state, dict(self.metadata))
        return (object.__new__, (type(self),), state)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        cls = self.__class__
        object.__setattr__(self, "__class__", _QuditFields)
        (
            self.index,
            self.frequency_hz,
            self.t1_us,
            self.t2_star_us,
            self.anharmonicity_hz,
            self.label,
            self.drive,
            self.readout,
            self.t2_echo_us,
            self.thermal_population,
            self._calibrations,
            *metadata,
        ) = state
        self.metadata = MappingProxyType(metadata[0]) if metadata else _EMPTY
        self.__class__ = cls

    def replace(self, **changes: Any) -> Self:
        """A copy with the fields *changes* names set to their values."""
        values = {n: getattr(self, n) for n in _QuditFields.__slots__}
        values["calibrations"] = values.pop("_calibrations")
        values.update(changes)
        return type(self)(**values)

    __replace__ = replace

    @property
    def depletion_ns(self) -> float:
        """How long after a readout the next one must wait: for a
        ``DispersiveReadout``, its ``depletion_ns`` if given, else the time
        for its photons to fall to 1% at its ``linewidth_hz``, ln(100) / kappa;
        else 0.
        """
        readout = self.readout
        return 0.0 if readout is None else readout._depletion_ns()

    @property
    def calibrations(
        self,
    ) -> Mapping[Gate | tuple[Gate, tuple[int, int]], DragPulse | ReadoutPulse]:
        """The calibrations the qudit publishes, as ``calibrations=`` gives
        them: by ``pulse.gates.Gate``, or by ``(gate, (lower, upper))`` for a
        transition other than (0, 1); read-only. Only what is published:
        ``get_calibration`` also derives the rest (``X90`` from ``X``, an
        unpublished amplitude). ``replace(calibrations={**q.calibrations,
        pulse.gates.X: pulse})`` changes one."""
        return _by_gate(self._calibrations)

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
        """The calibrated pulse of *gate* on the qubit's *transition*, by the
        rules of ``pulse.gates``: ``X``, ``X90``, ``Y`` and ``Y90`` a
        ``DragPulse`` (``X90`` half of ``X``, ``Y`` and ``Y90`` at +90
        degrees, where not published; an unpublished amplitude makes the
        rotation), ``Z``, ``S`` and ``T`` a ``VirtualZ`` (by pi, pi / 2
        and pi / 4, where not published), ``READOUT`` a ``ReadoutPulse``. A
        calibration it neither publishes nor derives is a ``NotFoundError``
        naming the qubit."""
        return _qudit_calibration(self, gate, transition)


def _plain_value(value: Any) -> Any:
    return dict(value) if type(value) in (MappingProxyType, _Calibrations) else value


class Offset(float):
    """A qudit's frequency as an offset in Hz from one reference that the
    system's qudits share and the system does not state:
    ``PhysicalQudit(1, Offset(1.5e8), 50.0, 30.0)`` is 150 MHz above the
    qudit at ``Offset(0.0)``. Every difference of two such qudits'
    frequencies is known, so couplings and whatever derives from them
    compute as on a system with frequencies, and every frequency of the
    system's programs is an offset from the same reference.

    It is a ``float`` that says what it is measured from: arithmetic on it
    gives a plain ``float``, and it equals only another ``Offset`` of the
    same value, so a qudit at ``Offset(5e9)`` is not the qudit at 5 GHz.
    It is ordered against other ``Offset`` values only: ``<`` with a number
    is an ``InvalidTypeError`` (a ``TypeError``, as between a ``str`` and a
    number); ``float(offset)`` is the number. An object that is not a
    number decides the comparison itself, so ``pytest.approx(1.5e8)`` equals
    ``Offset(1.5e8)`` either way round.

    An offset is an ``Offset`` itself: the package tells one by its exact
    class, and takes an instance of a subclass for the ``float`` it is.

    This holds with the ``Offset`` on the left, and on either side of an
    ``int`` or a ``float``. With another kind of number on the left
    (a ``Decimal``, a ``complex``, a ``Fraction``, a NumPy float) Python
    asks that number first, and it compares by value:
    ``Decimal(1) == Offset(1.0)`` is true where ``Offset(1.0) ==
    Decimal(1)`` is false, and ``Decimal(1) < Offset(2.0)`` is an answer,
    not an error.
    """

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if other.__class__ is Offset:
            return float.__eq__(self, other)
        # A number is not an offset; anything else may know better.
        return False if isinstance(other, numbers.Number) else NotImplemented

    def __ne__(self, other: object) -> bool:
        if other.__class__ is Offset:
            return float.__ne__(self, other)
        return True if isinstance(other, numbers.Number) else NotImplemented

    def __lt__(self, other: Any) -> bool:
        return float.__lt__(self, other) if _orders(self, other, "<") else NotImplemented

    def __le__(self, other: Any) -> bool:
        return float.__le__(self, other) if _orders(self, other, "<=") else NotImplemented

    def __gt__(self, other: Any) -> bool:
        return float.__gt__(self, other) if _orders(self, other, ">") else NotImplemented

    def __ge__(self, other: Any) -> bool:
        return float.__ge__(self, other) if _orders(self, other, ">=") else NotImplemented

    def __hash__(self) -> int:
        return hash((Offset, float(self)))

    def __repr__(self) -> str:
        return f"Offset({float(self)!r})"


# Whether *offset* is ordered against *other*, another ``Offset``; a
# plain number is a ``TypeError``, and anything else decides itself.
def _orders(offset: Offset, other: Any, sign: str) -> bool:
    if other.__class__ is Offset:
        return True
    if isinstance(other, numbers.Number):
        raise InvalidTypeError(
            f"'{sign}' is not supported between {offset!r}, an offset from a shared reference, "
            f"and the number {other!r}; compare float(offset), or another Offset"
        )
    return False


# The offset a qudit's *frequency* gives, as an ``Offset`` or as a
# system file holds one (``{"offset_hz": value}``), if it is finite.
def _offset_hz(frequency: Any) -> float | None:
    if frequency.__class__ is Offset:
        return float(frequency) if math.isfinite(frequency) else None
    if not (isinstance(frequency, Mapping) and set(frequency) == {"offset_hz"}):
        return None
    frequency = frequency["offset_hz"]
    return float(frequency) if _finite(frequency) else None


class _HeldQudit(PhysicalQudit):
    """A qudit whose frequency is not an absolute one (an ``Offset``, or
    ``None``) as a quantum system holds it. It prints, compares, hashes,
    pickles and is replaced as the ``PhysicalQudit`` it was made from."""

    __slots__ = ()

    @property
    def _frame_hz(self) -> float:  # type: ignore[override]
        # A plain number: the Offset stays in frequency_hz.
        return float(self.frequency_hz)

    @property
    def _circuit_hz(self) -> float:  # type: ignore[override]
        given = self.frequency_hz
        said = "no frequency_hz" if given is None else f"its frequency_hz as {given!r}"
        raise UnsupportedError(
            f"qudit {self.index} gives {said}, and a Calibration of a system with couplings "
            "reads the absolute frequencies of their qudits, where its circuit model gives a "
            "coupling's exchange; a system whose qudits give offsets or no frequencies takes "
            "no Calibration of its couplings"
        )

    @staticmethod
    def of(qudit: PhysicalQudit) -> PhysicalQudit:
        if isinstance(qudit, _HeldQudit):
            return qudit
        offset = _offset_hz(qudit.frequency_hz)
        if offset is None:
            made = PhysicalQudit.replace(qudit)
            object.__setattr__(made, "__class__", _OwnCarrierQudit)
        else:
            changes: dict[str, Any] = {"frequency_hz": Offset(offset)}
            readout = qudit.readout
            given = getattr(readout, "frequency_hz", None)
            if given is not None and given.__class__ is not Offset:
                # As a system file holds it; checked before a qudit is held.
                changes["readout"] = readout.replace(frequency_hz=Offset(_offset_hz(given)))  # type: ignore[union-attr, arg-type]
            made = PhysicalQudit.replace(qudit, **changes)
            object.__setattr__(made, "__class__", _HeldQudit)
        return made

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhysicalQudit):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in _QuditFields.__slots__)

    __hash__ = PhysicalQudit.__hash__

    def __repr__(self) -> str:
        return repr(self.replace())

    def __reduce__(self) -> tuple[Any, ...]:
        return self.replace().__reduce__()

    def replace(self, **changes: Any) -> PhysicalQudit:  # type: ignore[override]
        """A ``PhysicalQudit`` with the fields *changes* names set to their values."""
        values = {n: getattr(self, n) for n in _QuditFields.__slots__}
        values["calibrations"] = values.pop("_calibrations")
        values.update(changes)
        return PhysicalQudit(**values)

    __replace__ = replace  # type: ignore[assignment]


class _OwnCarrierQudit(_HeldQudit):
    """A qudit without a frequency (``frequency_hz=None``) as a quantum
    system holds it: 0.0 is the frequency its frame turns at, so that every
    frequency computed with it is an offset from its own carrier."""

    __slots__ = ()
    _frame_hz = 0.0


class Coupling:
    """A coupling of the qudits ``qudits`` of a quantum system: two in this
    release, by the system's qudit index, in the order the coupling names
    them (``transmon_surface17``'s are (auxiliary, data)).

    The system's ``model`` decides the operator; a ``TransmonModel``
    coupling is the exchange g (a^dagger b + a b^dagger), in the
    rotating-wave approximation, and ``strength_hz`` is g / 2 pi. A
    ``strength_hz`` of ``None`` is one not measured yet: the system
    compiles, schedules and exports, and ``simulate`` refuses it.
    ``calibrations=`` gives the gates it is tuned for, ``{pulse.gates.CZ:
    FluxCZ(...)}``, which ``calibrations`` gives back and ``get_calibration``
    reads. A coupling is
    read-only: ``replace`` makes a changed copy.

    A plain slotted class, as ``PhysicalQudit`` is: its calibrations are
    private storage, which a dataclass field could not be and still be the
    constructor's ``calibrations=``.
    """

    __slots__ = ("qudits", "strength_hz", "_calibrations")
    __match_args__ = ("qudits", "strength_hz")

    qudits: tuple[int, ...]
    strength_hz: float | None
    # What calibrations= gives, which get_calibration reads.
    _calibrations: Mapping[Any, FluxCZ]

    def __init__(
        self,
        qudits: Sequence[int],
        strength_hz: float | None,
        calibrations: Mapping[Any, FluxCZ] = _NO_CALIBRATIONS,
    ) -> None:
        if qudits.__class__ is not tuple and isinstance(qudits, Sequence):
            qudits = tuple(qudits)
        if calibrations.__class__ is not _Calibrations:
            calibrations = _calibrations(
                calibrations, f"the coupling of qudits {qudits}", _COUPLING_GATES
            )
        set_ = object.__setattr__
        set_(self, "qudits", qudits)
        set_(self, "strength_hz", strength_hz)
        set_(self, "_calibrations", calibrations)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.qudits, self.strength_hz, self._calibrations) == (
            other.qudits,
            other.strength_hz,
            other._calibrations,
        )

    def __hash__(self) -> int:
        return hash((self.qudits, self.strength_hz))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(qudits={self.qudits!r}, strength_hz={self.strength_hz!r}, "
            f"calibrations={_plain_value(self._calibrations)!r})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.qudits, self.strength_hz, self._calibrations))

    def replace(self, **changes: Any) -> Self:
        """A copy with the fields *changes* names set to their values."""
        values: dict[str, Any] = {
            "qudits": self.qudits,
            "strength_hz": self.strength_hz,
            "calibrations": self._calibrations,
        }
        values.update(changes)
        return type(self)(**values)

    __replace__ = replace

    @property
    def calibrations(self) -> Mapping[Gate | tuple[Gate, tuple[int, int]], FluxCZ]:
        """The calibrations the coupling publishes, as ``calibrations=`` gives
        them: by ``pulse.gates.Gate`` (``CZ``), or by ``(gate, (lower,
        upper))`` for a transition other than (0, 1); read-only."""
        return _by_gate(self._calibrations)

    def get_calibration(self, gate: Any, transition: tuple[int, int] = (0, 1)) -> FluxCZ:
        """The calibration of *gate* (``pulse.gates.CZ``) on the coupling; a
        ``NotFoundError`` if it publishes none."""
        name = _gate_name(gate, _COUPLING_GATES)
        if name not in _COUPLING_GATES:
            raise InvalidValueError(
                f"{_gate_label(name)} is calibrated on a qudit, not on a coupling: "
                f"qs.qudits[i].get_calibration({_gate_label(name)})"
            )
        at = _transition(transition)
        found: FluxCZ | None = self._calibrations.get(_calibration_key(name, at))
        if found is None:
            which = _gate_label(name) if at == _DEFAULT_TRANSITION else f"{_gate_label(name)} {at}"
            a, b = self.qudits
            raise NotFoundError(
                f"the coupling of qudits {a} and {b} publishes no {which} calibration; "
                f"{_published(self._calibrations)}"
            )
        return found


@_dataclass_when_used(frozen=True)
class ResidualZZ:
    """Residual ZZ between the qudits ``qudits`` (two, by the system's qudit
    index), in addition to what the couplings produce.

    ``rate_hz`` is the ZZ rate zeta / 2 pi, zeta = E11 - E10 - E01 + E00,
    modeled as zeta n_a n_b.
    """

    qudits: tuple[int, ...]
    rate_hz: float

    def __post_init__(self) -> None:
        qudits = self.qudits
        if qudits.__class__ is not tuple and isinstance(qudits, Sequence):
            object.__setattr__(self, "qudits", tuple(qudits))


# The operator a coupling stands for, by the simulation model of its system:
# the entry of the model's class or of its nearest base class here.
_COUPLING_OPERATORS: Mapping[type[SimulationModel], str] = MappingProxyType(
    {TransmonModel: "exchange"}
)


# The operator of the couplings of a system with the model *model*, or
# None if the model gives them none.
def _coupling_operator(model: Any) -> str | None:
    for cls in type(model).__mro__:
        operator = _COUPLING_OPERATORS.get(cls)
        if operator is not None:
            return operator
    return None


# The eigenvalues of the real symmetric *matrix* and its eigenvectors
# (``vectors[i][k]``, component *i* of eigenvector *k*), by cyclic Jacobi
# rotations: plain Python, as a coupled system's dressed frequencies are
# made when it is first compiled for, which loads no numerical library.
def _jacobi(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[float(i == j) for j in range(n)] for i in range(n)]
    scale = max((abs(x) for row in a for x in row), default=0.0) or 1.0
    for _ in range(100):
        off = sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n))
        if off <= (1e-30 * scale) ** 2:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if apq == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


# A coupled group this large is solved with numpy: plain Python grows as
# the cube of its size (about 1 s for 100 qudits).
_PLAIN_JACOBI_LIMIT = 32


# Each qudit's 0-1 frequency in Hz, by index, among the qudits of
# *frequencies* (their bare 0-1 frequencies) with the exchange couplings
# *pairs* ((qudits, strength_hz)) between them, and the weight on each
# qudit of the eigenstate each one takes: an eigenvalue of the
# single-excitation matrix ``diag(f) + J``, solved for each group of
# qudits the couplings join. Each qudit takes the eigenstate that holds
# most of it, the states going to qudits in order of weight, so that two
# qudits near resonance, mixed, each still take one.
def _single_excitation_weights(
    frequencies: Mapping[int, float], pairs: list[Any]
) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    out = {q: float(f) for q, f in frequencies.items()}
    weights = {q: {q: 1.0} for q in out}
    joined: dict[int, set[int]] = {q: {q} for q in out}
    for (a, b), _ in pairs:
        group = joined[a] | joined[b]
        for q in group:
            joined[q] = group
    groups = {id(g): sorted(g) for g in joined.values() if len(g) > 1}
    for qudits in groups.values():
        n = len(qudits)
        row = {qudit: i for i, qudit in enumerate(qudits)}
        # About the group's mean, so that kHz shifts keep their digits
        # against GHz frequencies.
        mean = sum(out[q] for q in qudits) / n
        matrix = [[0.0] * n for _ in qudits]
        for q in qudits:
            matrix[row[q]][row[q]] = out[q] - mean
        for (a, b), strength in pairs:
            if a in row:
                matrix[row[a]][row[b]] += strength
                matrix[row[b]][row[a]] += strength
        if n > _PLAIN_JACOBI_LIMIT:
            import numpy as np

            w, v = np.linalg.eigh(np.array(matrix))
            values, vectors = w.tolist(), v.tolist()
        else:
            values, vectors = _jacobi(matrix)
        ranked = sorted(
            ((vectors[i][k] ** 2, i, k) for i in range(n) for k in range(n)), reverse=True
        )
        rows: set[int] = set()
        states: set[int] = set()
        for _, i, k in ranked:
            if i in rows or k in states:
                continue
            rows.add(i)
            states.add(k)
            out[qudits[i]] = values[k] + mean
            weights[qudits[i]] = {qudits[r]: vectors[r][k] ** 2 for r in range(n)}
    return out, weights


# Each qudit's 0-1 frequency in Hz with the couplings *pairs*
# (_single_excitation_weights).
def _single_excitation(frequencies: Mapping[int, float], pairs: list[Any]) -> dict[int, float]:
    return _single_excitation_weights(frequencies, pairs)[0]


# The exchange couplings of known strength of *description*, as
# (qudits, strength_hz); none when its model gives no exchange.
def _exchange_pairs(description: Any) -> list[Any]:
    if _coupling_operator(description.model) != "exchange":
        return []
    return [(c.qudits, c.strength_hz) for c in description.couplings if c.strength_hz]


# The description last asked for and its table.
_LAST_DRESSED: list[tuple[Any, Any]] = [(None, None)]
_DRESSED: weakref.WeakKeyDictionary[Any, dict[int, float] | None] = weakref.WeakKeyDictionary()


def dressed_frequencies(description: Any) -> dict[int, float] | None:
    """Each qudit's 0-1 frequency in Hz with the system's couplings
    included, by qudit index, or None when no coupling moves one (no
    exchange coupling of known strength) or the system keeps bare frames
    (``frames="bare"``). Computed once per description.

    The exchange conserves the number of excitations, so with every other
    qudit in its ground level a qudit's dressed 0-1 frequency is an
    eigenvalue of the single-excitation matrix ``diag(f) + J``, exactly,
    whatever the qudits' levels and anharmonicities."""
    # The registry gives a named system's description itself on every
    # call: the last one asked for is answered without hashing it.
    last = _LAST_DRESSED[0]
    if last[0] is description:
        remembered: dict[int, float] | None = last[1]
        return remembered
    try:
        dressed = _DRESSED[description]
    except KeyError:
        pass
    else:
        _LAST_DRESSED[0] = (description, dressed)
        return dressed
    pairs = _exchange_pairs(description) if description.frames == "dressed" else []
    dressed = (
        _single_excitation({q: p._frame_hz for q, p in description.qudits.items()}, pairs)
        if pairs
        else None
    )
    _DRESSED[description] = dressed
    _LAST_DRESSED[0] = (description, dressed)
    return dressed


_NOTHING_MOVED: Mapping[int, float] = MappingProxyType({})
_LAYOUTS: weakref.WeakKeyDictionary[Any, dict[tuple[int, ...], Mapping[int, float]]] = (
    weakref.WeakKeyDictionary()
)


def layout_frequencies(description: Any, qudits: Sequence[int]) -> Mapping[int, float]:
    """For a simulation of the qudits *qudits* of *description* alone: the
    0-1 frequency in Hz to give each of them, by index, so that with the
    couplings between them it is at its dressed frequency on the whole
    system. A qudit whose dropped neighbours dress it (the others in their
    ground level) is moved by their share; the others are left out. Kept
    per description and set of qudits."""
    dressed = dressed_frequencies(description)
    if dressed is None:
        return _NOTHING_MOVED
    kept = tuple(sorted(set(qudits)))
    known = _LAYOUTS.setdefault(description, {})
    found = known.get(kept)
    if found is not None:
        return found
    pairs = _exchange_pairs(description)
    inside = set(kept)
    # Nothing coupled dropped: the kept couplings dress the kept qudits as
    # the whole system does.
    if all((a in inside) == (b in inside) for (a, b), _ in pairs):
        known[kept] = MappingProxyType({})
        return known[kept]
    within_pairs = [(q, s) for q, s in pairs if inside.issuperset(q)]
    bare = {q: description.qudits[q]._frame_hz for q in kept}
    given = dict(bare)
    # Newton on the spectrum: the kept qudits' eigenvalues, in order, are
    # to be the whole system's dressed frequencies of them, in order (which
    # qudit a state of two near-resonant qudits counts as can differ between
    # the two). An eigenvalue moves with each bare frequency by the weight
    # its state has there (Hellmann-Feynman). Exact to a millihertz, unless
    # no frequencies can give that spectrum: two kept qudits near resonance
    # split by at least twice their coupling, which their dropped
    # neighbours can narrow. Then the closest it gets, by damped steps.
    targets = sorted(dressed[q] for q in kept)

    def missed(frequencies: dict[int, float]) -> tuple[float, list[Any], Any]:
        within, weights = _single_excitation_weights(frequencies, within_pairs)
        states = sorted((within[q], q) for q in kept)
        miss = [t - value for t, (value, _) in zip(targets, states, strict=True)]
        return max(abs(m) for m in miss), miss, [weights[q] for _, q in states]

    worst, miss, rows = missed(given)
    # Levenberg-Marquardt: (J^T J + mu I) step = J^T miss, mu raised until a
    # step helps and lowered after one does, so that a nearly singular J
    # (two kept qudits at resonance) still gives a finite step.
    mu = 1e-9
    for _ in range(200):
        if worst < 1e-3 or mu > 1e12:
            break
        jacobian = [[row.get(r, 0.0) for r in kept] for row in rows]
        n = len(kept)
        normal = [
            [
                sum(jacobian[k][i] * jacobian[k][j] for k in range(n)) + (mu if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]
        gradient = [sum(jacobian[k][i] * miss[k] for k in range(n)) for i in range(n)]
        step = _solve(normal, gradient)
        trial = {r: given[r] + d for r, d in zip(kept, step, strict=True)}
        trial_worst, trial_miss, trial_rows = missed(trial)
        if trial_worst < worst:
            given, worst, miss, rows = trial, trial_worst, trial_miss, trial_rows
            mu /= 10
        else:
            mu *= 10
    if worst > 1.0:
        _warn_layout(description, kept, miss, rows, pairs)
    known[kept] = MappingProxyType({q: f for q, f in given.items() if f != bare[q]})
    return known[kept]


# A dressed level above the first is worked out among the qudits within
# this many couplings of its qudit, as the whole coupled group of a large
# chip would not fit in memory: each coupling further moves it by about
# (g / detuning)^2 of the last: under a millihertz on transmon_surface17,
# but tens of kHz on a chain of qudits 80 MHz apart with 30 MHz couplings,
# and up to about a MHz on such a chain of nine.
_HOPS = 3


# Dressed levels above the first: by description, (qudit, level) -> energy
# in Hz from the ground state; and the anharmonicities simulate gives a kept
# part (layout_anharmonicities), by description and set of qudits.
_LEVELS: weakref.WeakKeyDictionary[Any, dict[tuple[int, int], float]] = weakref.WeakKeyDictionary()
# Per description: (kept qudits, their simulated levels) -> anharmonicities.
_ANHARMONICITIES: weakref.WeakKeyDictionary[
    Any, dict[tuple[tuple[int, ...], tuple[int, ...]], Mapping[int, float]]
] = weakref.WeakKeyDictionary()


# The qudits the couplings *pairs* join to *qudit*, it among them.
def _group_of(
    description: Any, qudit: int, pairs: list[Any], hops: int | None = _HOPS
) -> list[int]:
    group, edge = {qudit}, set([qudit])
    while edge and (hops is None or hops > 0):
        reached = set()
        for (a, b), _ in pairs:
            if a in edge and b not in group:
                reached.add(b)
            elif b in edge and a not in group:
                reached.add(a)
        group |= reached
        edge = reached
        if hops is not None:
            hops -= 1
    return sorted(group)


# The basis, eigenvalues and eigenvectors (columns) of the qudits of
# *energies* (each its levels' bare energies in Hz) with the exchange
# *pairs* in the sector of *excitations* excitations.
def _sector(
    energies: Mapping[int, list[float]], pairs: list[Any], excitations: int
) -> tuple[list[tuple[int, ...]], Any, Any]:
    import numpy as np

    qudits = sorted(energies)
    tops = [len(energies[q]) - 1 for q in qudits]

    def fill(i: int, left: int) -> list[tuple[int, ...]]:
        if i == len(qudits) - 1:
            return [(left,)] if left <= tops[i] else []
        return [(n, *rest) for n in range(min(left, tops[i]) + 1) for rest in fill(i + 1, left - n)]

    basis = fill(0, excitations)
    where = {state: k for k, state in enumerate(basis)}
    col = {q: i for i, q in enumerate(qudits)}
    h = np.zeros((len(basis), len(basis)))
    for k, state in enumerate(basis):
        h[k, k] = sum(energies[q][state[col[q]]] for q in qudits)
        for (a, b), g in pairs:
            for x, y in ((a, b), (b, a)):
                na, nb = state[col[x]], state[col[y]]
                if nb and na < tops[col[x]]:
                    moved = list(state)
                    moved[col[x]] += 1
                    moved[col[y]] -= 1
                    h[where[tuple(moved)], k] += g * ((na + 1) * nb) ** 0.5
    values, vectors = np.linalg.eigh(h)
    return basis, values, vectors


# The dressed energy of *qudit* in its level *level*, the others in their
# ground level (the eigenstate holding most of that bare state), with that
# state's components on the basis.
def _level(
    energies: Mapping[int, list[float]], pairs: list[Any], qudit: int, level: int
) -> tuple[float, Any, list[tuple[int, ...]]]:
    basis, values, vectors = _sector(energies, pairs, level)
    qudits = sorted(energies)
    bare = tuple(level if q == qudit else 0 for q in qudits)
    k = basis.index(bare)
    state = int(abs(vectors[k, :]).argmax())
    return float(values[state]), vectors[:, state], basis


def dressed_level_energy(description: Any, qudit: int, level: int) -> float:
    """The energy in Hz of *qudit*'s level *level* with the couplings of
    *description* included, every other qudit in its ground level: the
    eigenvalue of the *level*-excitation sector of the qudits its couplings
    join to it whose state holds most of that bare state. Level 1 is its
    dressed 0-1 frequency; a higher level is worked out among the qudits
    within _HOPS couplings of it. Every tone of its line is the difference
    of two of these. Kept per description."""
    if level == 0:
        return 0.0
    dressed = dressed_frequencies(description)
    if dressed is None:
        model = description.model
        return float(model.energies_hz(description.qudits[qudit], level + 1)[level])
    if level == 1:
        return dressed[qudit]
    return _group_level(description, qudit, level)


# *qudit*'s dressed level *level* (from 2) among the qudits within _HOPS
# couplings of it; *qudit* itself has *level* + 1 levels whatever its model
# keeps, as a simulation may keep more.
def _group_level(description: Any, qudit: int, level: int) -> float:
    known = _LEVELS.setdefault(description, {})
    found = known.get((qudit, level))
    if found is None:
        pairs = _exchange_pairs(description)
        group = _group_of(description, qudit, pairs)
        model = description.model
        energies = {
            q: model.energies_hz(
                description.qudits[q],
                level + 1 if q == qudit else min(level, model.levels_of(q) - 1) + 1,
            )
            for q in group
        }
        within = [(q, s) for q, s in pairs if set(q) <= set(group)]
        found = _level(energies, within, qudit, level)[0]
        known[qudit, level] = found
    return found


class _DressedTones(Mapping[tuple[int, int], float]):
    """The tones of a coupled qudit's drive line, by (lower, upper) level:
    each the difference of its dressed levels, worked out when it is first
    read."""

    __slots__ = ("_description", "_qudit", "_bare", "_made")

    def __init__(self, description: Any, qudit: int, bare: Mapping[tuple[int, int], float]):
        self._description, self._qudit, self._bare = description, qudit, bare
        self._made: dict[tuple[int, int], float] = {}

    def __getitem__(self, transition: tuple[int, int]) -> float:
        made = self._made.get(transition)
        if made is None:
            if transition not in self._bare:
                from .._errors import NotFoundError

                raise NotFoundError(
                    f"the drive line has no tone {transition}; it has "
                    + ", ".join(map(str, self._bare))
                )
            lower, upper = transition
            made = dressed_level_energy(
                self._description, self._qudit, upper
            ) - dressed_level_energy(self._description, self._qudit, lower)
            self._made[transition] = made
        return made

    def __contains__(self, transition: object) -> bool:
        # Without working the tone out.
        return transition in self._bare

    def __iter__(self) -> Any:
        return iter(self._bare)

    def __len__(self) -> int:
        return len(self._bare)


def layout_anharmonicities(
    description: Any, qudits: Sequence[int], levels: Sequence[int] | None = None
) -> Mapping[int, float]:
    """For a simulation of the qudits *qudits* of *description* alone, at
    least three levels each: the anharmonicity in Hz to give each kept qudit
    whose group reaches a qudit left out, so that its dressed level 2, with
    the kept couplings and layout_frequencies, is the whole system's. Empty
    when no kept qudit's group reaches one. Kept per description and set of
    qudits."""
    if dressed_frequencies(description) is None:
        return _NOTHING_MOVED
    kept = sorted(set(qudits))
    # Each kept qudit's simulated levels (*levels*, in the order of *qudits*).
    count = dict(zip(qudits, levels, strict=True)) if levels is not None else {}
    shape = tuple(min(count.get(q, 3), 3) for q in kept)
    known = _ANHARMONICITIES.setdefault(description, {})
    found = known.get((tuple(kept), shape))
    if found is not None:
        return found
    pairs = _exchange_pairs(description)
    inside = set(kept)
    # A kept qudit whose group reaches a dropped qudit: its level 2 has
    # that qudit's share.
    simulated = dict(zip(kept, shape, strict=True))
    marked = [
        q
        for q in kept
        if simulated[q] > 2 and not inside.issuperset(_group_of(description, q, pairs))
    ]
    if not marked:
        known[tuple(kept), shape] = _NOTHING_MOVED
        return _NOTHING_MOVED
    within = [(q, s) for q, s in pairs if inside.issuperset(q)]
    frequencies = {
        q: layout_frequencies(description, kept).get(q, description.qudits[q]._frame_hz)
        for q in kept
    }
    alphas = {q: float(description.qudits[q].anharmonicity_hz or 0.0) for q in kept}

    def energies() -> dict[int, list[float]]:
        return {
            q: [0.0, frequencies[q], 2 * frequencies[q] + alphas[q]][: simulated[q]] for q in kept
        }

    # Level 2 where the 0-2 and 1-2 tones a program drives put it.
    targets = {q: dressed_level_energy(description, q, 2) for q in marked}
    for _ in range(30):
        miss, rows = [], []
        for q in marked:
            value, vector, basis = _level(energies(), within, q, 2)
            miss.append(targets[q] - value)
            col = {r: i for i, r in enumerate(kept)}
            rows.append(
                [
                    float(sum(vector[k] ** 2 for k, b in enumerate(basis) if b[col[r]] == 2))
                    for r in marked
                ]
            )
        if max(abs(m) for m in miss) < 1e-3:
            break
        for r, d in zip(marked, _solve(rows, miss), strict=True):
            alphas[r] += d
    found = MappingProxyType({q: alphas[q] for q in marked})
    known[tuple(kept), shape] = found
    return found


# Warn that the qudits *kept* of *description*, simulated alone, are
# as far as *miss* from their dressed frequencies (``layout_frequencies``):
# naming the kept qudits of the states that miss, by the weight each
# state has on them, and the left-out qudits coupled to those.
def _warn_layout(
    description: Any, kept: tuple[int, ...], miss: list[float], rows: list[Any], pairs: list[Any]
) -> None:
    import os
    import warnings

    from .._errors import _DressedLayoutWarning

    missing = sorted(
        {max(row, key=row.get) for row, m in zip(rows, miss, strict=True) if abs(m) > 1.0}
    )
    near = sorted(
        {b if a in missing else a for (a, b), _ in pairs if (a in missing) != (b in missing)}
        - set(kept)
    )
    if not near:
        near = sorted({b if a in kept else a for (a, b), _ in pairs if (a in kept) != (b in kept)})

    def listed(qudits: list[int]) -> str:
        return ("qudit " if len(qudits) == 1 else "qudits ") + ", ".join(map(str, qudits))

    one = len(missing) == 1
    warnings.warn(
        f"simulate: {listed(missing)} {'is' if one else 'are'} so near resonance that, "
        f"simulated without {listed(near)}, no frequencies give {'it' if one else 'them'} "
        f"the dressed {'frequency it has' if one else 'frequencies they have'} on the whole "
        f"quantum system: {'it is' if one else 'they are'} simulated up to "
        f"{max(abs(m) for m in miss) / 1e3:.3g} kHz from {'it' if one else 'them'}. Simulate "
        f"{listed(near)} too to have {'it' if one else 'them'} exactly.",
        _DressedLayoutWarning,
        # At the program's call of simulate, not in the package.
        skip_file_prefixes=(os.path.dirname(os.path.dirname(__file__)) + os.sep,),
    )


# The solution of the linear system *a* x = *b*, by Gaussian elimination
# with partial pivoting.
def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if m[col][col] == 0.0:
            continue
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        if m[r][r] != 0.0:
            x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


# The fields of a _Description, in the order its constructor takes them.
_DESCRIPTION_FIELDS = (
    "name",
    "qudits",
    "couplings",
    "residual_zz",
    "model",
    "attribution",
    "clock_ghz",
)
# What the drive tones of a system with couplings are at: each qudit's
# dressed 0-1 frequency, or its bare frequency_hz.
_DRIVE_FRAMES = ("dressed", "bare")


class _Description:
    """What ``pulse.quantum_system`` builds a quantum system from: the values
    of its keywords, read-only (``qudits`` a read-only mapping, ``couplings``
    and ``residual_zz`` tuples). ``clock_ghz`` is the clock that converts
    nanoseconds to cycles. The model also decides the operator of the
    couplings: the exchange for ``TransmonModel``. Descriptions compare by
    content, so one unpickled is equal to the one pickled.

    The couplings and residual ZZ entries are checked when it is built, and
    a ``QuantumSystemDefinitionError`` lists every problem, each at its path
    (``couplings[3].qudits[1]``).

    A plain slotted class rather than a dataclass: every quantum system has
    one, and making the dataclass cost more than importing the rest of the
    module.
    """

    __slots__ = (*_DESCRIPTION_FIELDS, "__weakref__")

    # Whether its qudits give no frequencies, each tone then an offset from
    # its own qudit's carrier; a compile asks once.
    _own_carrier: ClassVar[bool] = False
    # What its frequencies are measured from (``frequency_frame``).
    _frame: ClassVar[str] = "absolute"

    name: str
    qudits: Mapping[int, PhysicalQudit]
    couplings: tuple[Coupling, ...]
    residual_zz: tuple[ResidualZZ, ...]
    model: SimulationModel | None
    attribution: str
    clock_ghz: float
    # Where the drive tones of a system with couplings play (_DRIVE_FRAMES): not a
    # field, so that the default costs a description nothing to build; a
    # description with bare frames is a _BareDescription.
    frames = "dressed"
    # What it adds to a quantum system's fingerprint (_device._fingerprint).
    _frames_content: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        qudits: Mapping[int, PhysicalQudit] = _EMPTY,
        couplings: Sequence[Coupling] = (),
        residual_zz: Sequence[ResidualZZ] = (),
        model: SimulationModel | type[SimulationModel] | None = _TRANSMON,
        attribution: str = "",
        clock_ghz: float = 2.0,
    ) -> None:
        if not clock_ghz > 0:
            raise InvalidValueError(f"clock_ghz must be positive, got {clock_ghz}")
        if isinstance(model, type) and issubclass(model, SimulationModel):
            model = model()
        elif model is not None and not isinstance(model, SimulationModel):
            raise InvalidTypeError(
                "model must be a SimulationModel class or instance, such as TransmonModel, "
                f"or None; got {model!r}"
            )
        set_ = object.__setattr__
        set_(self, "name", name)
        set_(self, "qudits", MappingProxyType(dict(qudits)))
        set_(self, "couplings", tuple(couplings))
        set_(self, "residual_zz", tuple(residual_zz))
        set_(self, "model", model)
        set_(self, "attribution", attribution)
        set_(self, "clock_ghz", clock_ghz)
        # Every problem with the target's qudits and entries, each at its path in it.
        problems = _qudits_problems(self.qudits) + _entries_problems(self)
        if problems:
            # A frequency that is an Offset, or left out, is reported by the
            # checks of the qudits as any value that is not a frequency in Hz
            # is. Here, where a system whose qudits have frequencies and
            # nothing to report never comes, such a system is told from a
            # mistake and held.
            problems = _unstated(self, problems)
            if not problems:
                return
            raise QuantumSystemDefinitionError(
                f"{_count(len(problems), 'problem')} in the quantum system {self.name!r}",
                problems=problems,
            )

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def replace(self, **changes: Any) -> _Description:
        """A copy with *changes*, built and checked as a new one, as
        ``dataclasses.replace`` makes one."""
        values: dict[str, Any] = {
            "name": self.name,
            "qudits": self.qudits,
            "couplings": self.couplings,
            "residual_zz": self.residual_zz,
            "model": self.model,
            "attribution": self.attribution,
            "clock_ghz": self.clock_ghz,
        }
        frames = changes.pop("frames", self.frames)
        if frames not in _DRIVE_FRAMES:
            raise InvalidValueError(
                f'frames must be "dressed" or "bare", got {frames!r}: "dressed" drives each '
                'qudit of a system with couplings at its dressed frequency, "bare" at its '
                "frequency_hz"
            )
        values.update(changes)
        described = _Description(**values)
        # Without a coupling that dresses, the frames are the same: the
        # default description.
        if frames == "bare" and _exchange_pairs(described):
            return _BareDescription(**values)
        return described

    def __repr__(self) -> str:
        values = ", ".join(f"{f}={getattr(self, f)!r}" for f in _DESCRIPTION_FIELDS)
        return f"_Description({values})"

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f) for f in _DESCRIPTION_FIELDS)

    def __hash__(self) -> int:
        return hash((self.name, self.clock_ghz, len(self.qudits), len(self.couplings)))

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import _plain, rebuilt

        return (rebuilt, (type(self), {f: _plain(getattr(self, f)) for f in _DESCRIPTION_FIELDS}))

    @property
    def qudit_count(self) -> int:
        return len(self.qudits)

    @property
    def frequency_frame(self) -> str:
        """What the frequencies of the system and of its programs are
        measured from: ``"absolute"`` (the qudits give their frequencies),
        ``"shared_reference"`` (each qudit gives an ``Offset`` from one
        reference the system does not state) or ``"own_carrier"`` (the
        qudits give none: each tone's frequency is an offset from its own
        qudit's carrier)."""
        return self._frame

    @property
    def qudit_indices(self) -> list[int]:
        return sorted(self.qudits.keys())

    @property
    def frequencies(self) -> dict[int, float]:
        return {q.index: q._frame_hz for q in self.qudits.values()}

    @property
    def anharmonicities(self) -> dict[int, float | None]:
        return {q.index: q.anharmonicity_hz for q in self.qudits.values()}

    @property
    def t1_times(self) -> dict[int, float]:
        """T1 in microseconds, keyed by qudit index."""
        return {q.index: q.t1_us for q in self.qudits.values()}

    @property
    def t2_times(self) -> dict[int, float]:
        """T2* in microseconds, keyed by qudit index."""
        return {q.index: q.t2_star_us for q in self.qudits.values()}

    @property
    def t2_echo_times(self) -> dict[int, float | None]:
        """Hahn-echo T2 in microseconds, keyed by qudit index (None if unknown)."""
        return {q.index: q.t2_echo_us for q in self.qudits.values()}

    @property
    def thermal_populations(self) -> dict[int, float]:
        """Equilibrium excited-state population, keyed by qudit index."""
        return {q.index: q.thermal_population for q in self.qudits.values()}

    @property
    def coupling_map(self) -> list[tuple[int, ...]]:
        """The qudits of each coupling, in the order of ``couplings``."""
        return [c.qudits for c in self.couplings]

    def coupling(self, a: int, b: int) -> Coupling:
        """The coupling of qudits *a* and *b*, in either order (the system's
        qudit indices, as ``Qudit.index``); a ``NotFoundError`` names the
        qudits *a* couples to."""
        for c in self.couplings:
            if c.qudits == (a, b) or c.qudits == (b, a):
                return c
        partners = sorted(q for c in self.couplings if a in c.qudits for q in c.qudits if q != a)
        if partners:
            which = f"qudit {a} couples to {', '.join(map(str, partners))}"
        else:
            which = f"qudit {a} has no couplings"
        raise NotFoundError(f"qudits {a} and {b} of {self.name} are not coupled; {which}")

    def connectivity_graph(self) -> dict[int, list[int]]:
        """The qudits each qudit couples to, by qudit index, for every qudit."""
        g: dict[int, list[int]] = {idx: [] for idx in self.qudits}
        for c in self.couplings:
            a, b = c.qudits
            g[a].append(b)
            g[b].append(a)
        return g

    def drive_amplitude_scale(self, qudit_index: int) -> float:
        """Return the conversion from pulse amplitude to radians/ns.

        A ``Drive`` or a ``MicrowaveDrive`` may give
        ``amplitude_scale_rad_per_ns`` explicitly; else, if the qudit's ``X``
        calibration gives an amplitude, the scale that makes that pulse a pi
        rotation, from its truncated-Gaussian area. A qudit with neither, or
        no drive, uses 1.0: pulse amplitudes are already angular rates in
        rad/ns. Another subclass of ``Drive`` is ``UnsupportedError``.
        """
        if qudit_index not in self.qudits:
            raise NotFoundError(f"qudit {qudit_index} is not in quantum system {self.name!r}")
        drive = self.qudits[qudit_index].drive
        if drive is None:
            return 1.0
        if not isinstance(drive, MicrowaveDrive) and drive.__class__ is not Drive:
            raise UnsupportedError(
                f"no drive model for {type(drive).__name__} (qudit {qudit_index}); the "
                "simulator models a Drive and a MicrowaveDrive"
            )
        explicit = drive.amplitude_scale_rad_per_ns
        if explicit is not None:
            if explicit <= 0:
                raise InvalidValueError(
                    f"amplitude_scale_rad_per_ns must be positive for qudit {qudit_index}"
                )
            return float(explicit)
        qudit = self.qudits[qudit_index]
        x: Any = qudit._calibrations.get("x")
        if x is not None and x.amplitude:
            return math.pi / (float(x.amplitude) * _area_ns(x, _qudit_name(qudit)))
        return 1.0


class _BareDescription(_Description):
    """A description whose system plays each qudit's drive tone at its bare
    ``frequency_hz``, couplings or not (``frames="bare"``)."""

    __slots__ = ()
    frames = "bare"
    _frames_content = ("bare",)


class _BareSharedReferenceDescription(_BareDescription):
    """A description with bare frames whose qudits give their frequencies
    as offsets from one reference they share. (A system whose qudits give
    none has no couplings to dress, so bare frames are its default.)"""

    __slots__ = ()
    _frame = "shared_reference"

    @property
    def frequencies(self) -> dict[int, float]:
        return {q.index: float(q.frequency_hz) for q in self.qudits.values()}


# The error that *needs* the strengths of *target*'s couplings at
# *indices*, which are not measured yet.
def _unmeasured(
    target: _Description,
    indices: Sequence[int],
    needs: str,
    fix: str = "give the strength in Hz",
) -> InvalidValueError:
    named = [
        f"couplings[{i}] (qudits {' and '.join(map(str, target.couplings[i].qudits))})"
        for i in indices
    ]
    listed = named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]
    verb = "has" if len(named) == 1 else "have"
    return InvalidValueError(
        f"{needs}; {listed} of {target.name!r} {verb} strength_hz None, not measured yet: {fix}"
    )


# The couplings and residual ZZ entries last found to have no problem, and the
# qudits of their system. A system made from that one with
# dataclasses.replace, or with only its couplings' strengths changed (as a
# calibration does), checks only what changed.
_CHECKED: tuple[Sequence[Any], Sequence[Any], frozenset[int]] = ((), (), frozenset())


# Every problem with *target*'s couplings and residual ZZ entries.
def _entries_problems(target: _Description) -> list[Problem]:
    global _CHECKED
    known = target.qudits
    couplings, residual_zz = target.couplings, target.residual_zz
    checked_couplings, checked_residual_zz, checked_qudits = _CHECKED
    if residual_zz is checked_residual_zz and known.keys() == checked_qudits:
        if couplings is checked_couplings or _strengths_changed(couplings, checked_couplings):
            return []
    if (not couplings or _plain_entries(couplings, Coupling, known)) and (
        not residual_zz or _plain_entries(residual_zz, ResidualZZ, known)
    ):
        _CHECKED = (couplings, residual_zz, frozenset(known))
        return []
    problems: list[Problem] = []
    groups: tuple[tuple[str, Sequence[Any]], ...] = (
        ("couplings", couplings),
        ("residual_zz", residual_zz),
    )
    for kind, entries in groups:
        pairs: dict[tuple[int, int], int] = {}
        for i, entry in enumerate(entries):
            found, pair = _entry_problems(kind, i, entry, known, pairs)
            problems += found
            if pair is not None:
                pairs[pair] = i
    return problems


class _OwnCarrierDescription(_Description):
    """The description of a system whose qudits give no frequencies. Its
    own class, so that the one question a compile asks of a description is
    a class attribute and a system with frequencies stores nothing for it."""

    __slots__ = ()
    _own_carrier = True
    _frame = "own_carrier"


class _SharedReferenceDescription(_Description):
    """The description of a system whose qudits give their frequencies as
    offsets from one reference they share."""

    __slots__ = ()
    _frame = "shared_reference"

    @property
    def frequencies(self) -> dict[int, float]:
        # Plain numbers, as every system gives: the offsets.
        return {q.index: float(q.frequency_hz) for q in self.qudits.values()}


# The classes a description has.
_DESCRIPTIONS = (
    _Description,
    _OwnCarrierDescription,
    _SharedReferenceDescription,
    _BareDescription,
    _BareSharedReferenceDescription,
)

# What the checks read of every qudit at once.
_QUDIT_VALUES = attrgetter(
    "frequency_hz", "t1_us", "t2_star_us", "t2_echo_us", "thermal_population"
)


# Every problem with the values of *qudits*, a system's qudits by index.
# Checks every qudit at once first, so the common case, where all of them
# are fine, stays cheap.
def _qudits_problems(qudits: Mapping[int, Any]) -> list[Problem]:
    if not qudits:
        return []
    values = list(qudits.values())
    if set(map(type, values)) == {PhysicalQudit}:
        frequencies, t1s, t2_stars, t2_echoes, thermals = zip(
            *map(_QUDIT_VALUES, values), strict=True
        )
        if (
            _finite_floats(frequencies)
            and min(frequencies) > 0.0
            and _finite_floats(t1s + t2_stars)
            and min(t1s + t2_stars) >= 0.0
            and _finite_floats(t2_echoes, optional=True)
            and min(filter(None, t2_echoes), default=0.0) >= 0.0
            and _finite_floats(thermals)
            and min(thermals) >= 0.0
            and max(thermals) < 1.0
        ):
            return []
    problems: list[Problem] = []
    for index, qudit in qudits.items():
        if isinstance(qudit, PhysicalQudit):
            problems += _qudit_value_problems(qudit, f"qudits[{index}]")
    return problems


_FRAMES = {
    "absolute": "as frequencies",
    "own_carrier": "not at all (None)",
    "shared_reference": "as offsets from a shared reference",
}


# What a qudit's *frequency* is measured from.
def _frequency_frame(frequency: Any) -> str:
    if frequency is None:
        return "own_carrier"
    return "absolute" if _offset_hz(frequency) is None else "shared_reference"


# The *problems* of *description*, whose checks found some, without
# those that are a qudit's frequency given as an ``Offset`` (or as a system
# file holds one) or left out, and with what such a system may not be:
# its qudits give their frequencies all one way, and declare nothing that
# needs a frequency they do not state. Without problems, *description*
# holds its qudits as such a system does and has its class.
def _unstated(description: _Description, problems: list[Problem]) -> list[Problem]:
    ways: dict[str, list[int]] = {}
    for index, qudit in description.qudits.items():
        if isinstance(qudit, PhysicalQudit):
            ways.setdefault(_frequency_frame(qudit.frequency_hz), []).append(index)
    if ways.keys() <= {"absolute"}:
        return problems
    # A frequency of another form is not a problem of its qudit.
    unstated = {
        f"qudits[{index}].frequency_hz"
        for way, found in ways.items()
        if way != "absolute"
        for index in found
    }
    problems = [p for p in problems if p.path not in unstated]
    if len(ways) > 1:
        from .._errors import _ranges

        said = "; ".join(
            f"{_FRAMES[way]} (qudits {_ranges(found)})" for way, found in sorted(ways.items())
        )
        # At the first qudit that differs from the one before it.
        first = next(iter(ways.values()))
        other = min(found[0] for found in ways.values() if found is not first)
        problems.append(
            Problem(
                f"the qudits give their frequencies in more than one way: {said}",
                path=f"qudits[{other}].frequency_hz",
                fix="give every qudit its 0-1 frequency, every qudit an Offset from one shared "
                "reference, or None to all of them",
            )
        )
        return problems
    own_carrier = "own_carrier" in ways
    problems += _readout_frequency_problems(description, own_carrier)
    if own_carrier:
        problems += _own_carrier_problems(description)
    if not problems:
        set_ = object.__setattr__
        held = {
            index: _HeldQudit.of(q) if isinstance(q, PhysicalQudit) else q
            for index, q in description.qudits.items()
        }
        set_(description, "qudits", MappingProxyType(held))
        set_(
            description,
            "__class__",
            _OwnCarrierDescription
            if own_carrier
            else _BareSharedReferenceDescription
            if description.frames == "bare"
            else _SharedReferenceDescription,
        )
    return problems


# A readout's ``frequency_hz`` that a system whose qudits give offsets
# or no frequencies cannot read: with offsets, one that is not an
# ``Offset`` from the same reference (or as a system file holds one);
# without frequencies, any.
def _readout_frequency_problems(description: _Description, own_carrier: bool) -> list[Problem]:
    problems = []
    for index, qudit in description.qudits.items():
        frequency = getattr(getattr(qudit, "readout", None), "frequency_hz", None)
        if frequency is None or (not own_carrier and _offset_hz(frequency) is not None):
            continue
        if own_carrier:
            why = (
                "the qudits give no frequency_hz, so the system does not state where the "
                "readout's frequency is measured from"
            )
            fix = (
                "leave the readout's frequency_hz out (its line then plays at the qudit's own tone)"
            )
        else:
            why = (
                "the qudits give their frequencies as offsets from a shared reference, and "
                "the readout's frequency is not a finite Offset from it"
            )
            fix = (
                "give it as Offset(hz) from the reference the qudits share, or leave it out "
                "(its line then plays at the qudit's own tone)"
            )
        problems.append(
            Problem(
                f"{frequency!r} is a readout frequency, and {why}",
                path=f"qudits[{index}].readout.frequency_hz",
                value=frequency,
                fix=f"{fix}, or give the qudits their frequencies",
            )
        )
    return problems


# What a system whose qudits leave their frequencies out may not
# declare: whatever acts by the difference of two qudits' frequencies,
# which it does not state.
def _own_carrier_problems(description: _Description) -> list[Problem]:
    if not description.couplings:
        return []
    pairs = ", ".join(str(tuple(c.qudits)) for c in description.couplings[:4])
    more = "" if len(description.couplings) <= 4 else ", ..."
    return [
        Problem(
            f"the qudits give no frequency_hz, and a coupling ({pairs}{more}) acts by the "
            "difference of its qudits' frequencies, which the system then does not state",
            path="couplings",
            fix="give the qudits their frequencies, or state the interaction as a number: "
            "residual_zz=[ResidualZZ((a, b), rate_hz=...)]",
        )
    ]


# The problems with the values of *qudit*, the qudit at *where*.
def _qudit_value_problems(qudit: PhysicalQudit, where: str) -> list[Problem]:
    problems: list[Problem] = []

    def check(field: str, ok: bool, message: str, fix: str) -> None:
        if not ok:
            value = getattr(qudit, field)
            problems.append(
                Problem(f"{value!r} {message}", path=f"{where}.{field}", value=value, fix=fix)
            )

    frequency = qudit.frequency_hz
    check(
        "frequency_hz",
        _finite(frequency) and frequency > 0,
        "is not a positive, finite number of Hz",
        "give the 0-1 frequency in Hz, more than 0; or, for a system that does not state "
        "it, Offset(hz) from a reference its qudits share, or None (null in a file) for "
        "each qudit's own carrier",
    )
    for field, name in (("t1_us", "T1"), ("t2_star_us", "T2*")):
        value = getattr(qudit, field)
        check(
            field,
            _finite(value) and value >= 0,
            "is not a finite number of microseconds, 0 or more",
            f"give {name} in microseconds, 0 or more (0 is no decay)",
        )
    echo = qudit.t2_echo_us
    check(
        "t2_echo_us",
        echo is None or (_finite(echo) and echo >= 0),
        "is not a finite number of microseconds, 0 or more",
        "give the Hahn-echo T2 in microseconds, 0 or more, or None (null in a file) if it "
        "is not measured",
    )
    thermal = qudit.thermal_population
    check(
        "thermal_population",
        _finite(thermal) and 0 <= thermal < 1,
        "is not a finite number in [0, 1)",
        "give the excited-state population as a fraction, 0 or more and less than 1",
    )
    return problems


# What the checks read of every entry or gate at once.
_QUDITS = attrgetter("qudits")
_STRENGTH = attrgetter("strength_hz")
_CALIBRATIONS = attrgetter("_calibrations")
_VALUES = methodcaller("values")
_RATE = attrgetter("rate_hz")

_NONE = type(None)


# Whether *values* are all finite floats (or None, if *optional*): their
# sum is finite unless one is inf or NaN, or the floats overflow, which
# then only takes the slower checks.
def _finite_floats(values: Sequence[Any], optional: bool = False) -> bool:
    types = set(map(type, values))
    if not (types <= {float, _NONE} if optional else types <= {float}):
        return False
    total = sum(filter(None, values), 0.0)
    return total - total == 0.0


# Whether *entries*, couplings or residual ZZ entries of a system with the
# qudits *known*, have nothing to report; checked in C for every entry at
# once, by class, so any other entry takes ``_entry_problems``.
def _plain_entries(entries: Sequence[Any], cls: type, known: Mapping[int, Any]) -> bool:
    if set(map(type, entries)) != {cls}:
        return False
    qudits = list(map(_QUDITS, entries))
    if set(map(type, qudits)) != {tuple} or set(map(len, qudits)) != {2}:
        return False
    flat = list(chain.from_iterable(qudits))
    if set(map(type, flat)) != {int} or not known.keys() >= set(flat):
        return False
    if any(map(eq, flat[::2], flat[1::2])) or len(set(map(frozenset, qudits))) < len(qudits):
        return False
    if cls is ResidualZZ:
        return _finite_floats(list(map(_RATE, entries)))
    if not _finite_floats(list(map(_STRENGTH, entries)), optional=True):
        return False
    gates = list(chain.from_iterable(map(_VALUES, filter(None, map(_CALIBRATIONS, entries)))))
    if not gates:
        return True
    durations, buffers, frequencies = zip(*map(_COMPONENT_VALUES, gates), strict=True)
    given = [f for f in frequencies if f is not None]
    return (
        _finite_floats(durations)
        and min(durations) >= 0.0
        and _finite_floats(buffers, optional=True)
        and min(filter(None, buffers), default=0.0) >= 0.0
        and _finite_floats(given)
        and min(given, default=1.0) > 0.0
    )


# Whether *couplings* are the *checked* ones with only their strengths
# changed, to strengths with nothing to report.
def _strengths_changed(couplings: Sequence[Any], checked: Sequence[Any]) -> bool:
    return (
        len(couplings) == len(checked)
        and set(map(type, couplings)) <= {Coupling}
        and all(map(is_, map(_QUDITS, couplings), map(_QUDITS, checked)))
        and all(map(is_, map(_CALIBRATIONS, couplings), map(_CALIBRATIONS, checked)))
        and _finite_floats(list(map(_STRENGTH, couplings)), optional=True)
    )


def _pair_key(qudits: tuple[int, ...]) -> tuple[int, int]:
    a, b = qudits
    return (a, b) if a < b else (b, a)


# The numbers that are not a value in a unit: a bool, and an Offset, which
# is a qudit's frequency measured from a shared reference and nothing else.
_NOT_VALUES = (bool, Offset)


# Whether *value* is a finite real number, not a bool or an ``Offset`` (in
# as many steps as when it was not a bool).
def _finite(value: Any) -> bool:
    return (
        value.__class__ not in _NOT_VALUES
        and isinstance(value, numbers.Real)
        and math.isfinite(value)
    )


# Every problem with *entry*, entry *index* of the system's *kind*
# (``"couplings"`` or ``"residual_zz"``), each at its path in the system,
# and its pair of qudits if they are two distinct known qudits of a new
# pair. *pairs* holds the earlier entries' pairs, by entry index.
def _entry_problems(
    kind: str, index: int, entry: Any, known: Mapping[int, Any], pairs: Mapping[Any, int]
) -> tuple[list[Problem], tuple[int, int] | None]:
    where = f"{kind}[{index}]"
    cls = Coupling if kind == "couplings" else ResidualZZ
    if not isinstance(entry, cls):
        problem = Problem(
            f"must be a {cls.__name__}, got {type(entry).__name__}", path=where, value=entry
        )
        return [problem], None
    noun = "coupling" if cls is Coupling else "residual ZZ entry"
    problems = _qudit_problems(entry.qudits, where, known, pairs, noun)
    pair = None if problems else _pair_key(entry.qudits)
    if isinstance(entry, Coupling):
        value = entry.strength_hz
        if value is not None and not _finite(value):
            problems.append(
                Problem(
                    f"{value!r} is not a finite number of Hz",
                    path=f"{where}.strength_hz",
                    value=value,
                    fix="give the strength in Hz, or None (null in a file) if it is not "
                    "measured yet",
                )
            )
        for key, cz in entry._calibrations.items():
            problems += _flux_cz_problems(cz, f"{where}.calibrations[{_key_label(key)}]")
        return problems, pair
    if not _finite(entry.rate_hz):
        problems.append(
            Problem(
                f"{entry.rate_hz!r} is not a finite number of Hz",
                path=f"{where}.rate_hz",
                value=entry.rate_hz,
                fix="give the ZZ rate in Hz",
            )
        )
    return problems, pair


# The problems with *qudits*, the qudits of the entry at *where*.
def _qudit_problems(
    qudits: Any, where: str, known: Mapping[int, Any], pairs: Mapping[Any, int], noun: str
) -> list[Problem]:
    path = f"{where}.qudits"
    fix = "give the indices of its two qudits, as (0, 1)"
    if not isinstance(qudits, tuple):
        return [
            Problem(
                f"must be a tuple of qudit indices, got {type(qudits).__name__}",
                path=path,
                value=qudits,
                fix=fix,
            )
        ]
    if len(qudits) != 2:
        return [
            Problem(
                f"names {_count(len(qudits), 'qudit')}; a {noun} is between two qudits "
                "in this release",
                path=path,
                value=qudits,
                fix=fix,
            )
        ]
    problems = []
    for j, q in enumerate(qudits):
        if q.__class__ is bool or not isinstance(q, int):
            problems.append(
                Problem(
                    f"must be a qudit index (an int), got {type(q).__name__}",
                    path=f"{path}[{j}]",
                    value=q,
                )
            )
        elif q not in known:
            from .._errors import _ranges

            problems.append(
                Problem(
                    f"{q} is not a qudit of the quantum system",
                    path=f"{path}[{j}]",
                    value=q,
                    fix=f"its qudits are {_ranges(known)}" if known else "it has no qudits",
                )
            )
    if problems:
        return problems
    a, b = qudits
    if a == b:
        return [
            Problem(
                f"names qudit {a} twice",
                path=path,
                value=qudits,
                fix=f"a {noun} is between two distinct qudits",
            )
        ]
    earlier = pairs.get(_pair_key(qudits))
    if earlier is not None:
        kind = where.partition("[")[0]
        return [
            Problem(
                f"names the same pair of qudits as {kind}[{earlier}]",
                path=path,
                value=qudits,
                fix=f"give each pair one {noun}; ({a}, {b}) and ({b}, {a}) are the same pair",
            )
        ]
    return []


# A calibration's key as a problem's path names it.
def _key_label(key: Any) -> str:
    if key.__class__ is tuple:
        return f"{key[0].upper()}, {key[1]}"
    return str.__str__(key).upper()


def _flux_cz_problems(gate: FluxCZ, where: str) -> list[Problem]:
    problems = []
    for name, required in (("duration_ns", True), ("buffer_ns", False)):
        value = getattr(gate, name)
        if (value is not None or required) and not (_finite(value) and value >= 0):
            problems.append(
                Problem(
                    f"{value!r} is not a finite time of 0 ns or more",
                    path=f"{where}.{name}",
                    value=value,
                    fix="give it in ns" if required else "give it in ns, or None",
                )
            )
    frequency = gate.interaction_frequency_hz
    if frequency is not None and not (_finite(frequency) and frequency > 0):
        problems.append(
            Problem(
                f"{frequency!r} is not a finite frequency above 0 Hz",
                path=f"{where}.interaction_frequency_hz",
                value=frequency,
                fix="give it in Hz, or None",
            )
        )
    return problems
