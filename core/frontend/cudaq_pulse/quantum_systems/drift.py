# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parameter drift of a simulated quantum system.

``pulse.quantum_system(..., parameter_drift=True)`` builds a quantum system
whose qudit parameters drift while it runs, as a real chip's do. It holds
two sets of values:

- the calibrated values (``qs.physical_qudits``), which are all that
  compiling reads: a drifting system compiles, caches, fingerprints and
  exports exactly as its non-drifting twin does;
- the true values (``qs.true_qudits``, each the same ``PhysicalQudit`` type),
  which ``pulse.simulate`` plays. They start at the calibrated values (a
  freshly calibrated chip) and move as drift time passes.

Drift time follows the run clock of what the system runs. Each
``simulate`` call advances it by its shots: ``shots_count`` times the
program's duration plus ``reset_time`` (without ``shots_count``, one shot).
The true values are held for the call, except that a call whose shots take
longer than the shortest correlation time of the system's drift models is
split, by whole shots, into chunks each at least that long, and each chunk
plays the true values at its start. ``qs.advance(20 * pulse.ms)`` moves
time on outside ``simulate``: the idle time between experiments.

``parameter_drift`` takes ``True`` (the default model, ``ParameterDrift()``,
on every qudit), a model (on every qudit) or ``{qudit: model}`` (only those
qudits, keyed as ``physical_qudits`` is). A model is a ``ParameterDrift``
or any object with the two methods of ``ParameterDriftModel``.
``parameter_drift_seed`` seeds every qudit's own random stream.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .._device import QuantumSystem, Qudit, _Handles
from .._errors import InvalidTypeError, InvalidValueError, UnsupportedError
from .._units import Time
from .base import (
    MicrowaveDrive,
    PhysicalQudit,
    _frequency_frame,
    _HeldQudit,
    _qudit_value_problems,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..compiler import ScheduledKernel
    from ..kernel._kernel import _Entry
    from ..runtime.simulate import SimulationResult

__all__ = ["OrnsteinUhlenbeck", "ParameterDrift", "ParameterDriftModel", "TelegraphT1"]

_SECOND_NS = 1e9


def _seconds(value: Any, what: str) -> float:
    """*value*, a ``Time`` that is not negative, in seconds."""
    if value.__class__ is not Time:
        raise InvalidTypeError(
            f"{what} is a time, as 10 * pulse.s; got {type(value).__name__} {value!r}"
        )
    if not float(value) >= 0:
        raise InvalidValueError(f"{what} must not be negative, got {value!r}")
    return float(value) / _SECOND_NS


@runtime_checkable
class ParameterDriftModel(Protocol):
    """How one qudit's true parameters drift: any object with these two
    methods. The state is the true qudit itself, so a model needs no state
    of its own; randomness must come from *rng* for ``parameter_drift_seed``
    to repeat a run. A model may also have ``correlation_time`` (a
    ``Time``): ``simulate`` then splits calls longer than it. Models reach
    worker processes pickled."""

    def start(self, calibrated: PhysicalQudit, rng: np.random.Generator) -> PhysicalQudit:
        """The true qudit when the system is built, from its *calibrated* one."""
        ...

    def advance(
        self,
        true: PhysicalQudit,
        calibrated: PhysicalQudit,
        seconds: float,
        rng: np.random.Generator,
    ) -> PhysicalQudit:
        """The true qudit *seconds* after *true*: a changed copy
        (``true.replace(...)``) of any of the fields the simulator reads."""
        ...


@dataclasses.dataclass(frozen=True)
class OrnsteinUhlenbeck:
    """A stationary Gaussian process around zero, with standard deviation
    ``sigma`` and correlation time ``correlation_time``: an offset that
    wanders and returns. ``ParameterDrift`` uses it for a qudit's frequency
    offset (``sigma`` in Hz) and its drive amplitude (``sigma`` relative to
    the calibrated amplitude)."""

    sigma: float
    correlation_time: Time

    def __post_init__(self) -> None:
        if not self.sigma >= 0:
            raise InvalidValueError(
                f"OrnsteinUhlenbeck sigma must not be negative, got {self.sigma!r}"
            )
        if not _seconds(self.correlation_time, "OrnsteinUhlenbeck correlation_time") > 0:
            raise InvalidValueError(
                "OrnsteinUhlenbeck correlation_time must be positive, "
                f"got {self.correlation_time!r}"
            )

    def _step(self, x: float, seconds: float, rng: np.random.Generator) -> float:
        """The exact update of *x* over *seconds*."""
        decay = math.exp(-seconds * _SECOND_NS / float(self.correlation_time))
        return x * decay + self.sigma * math.sqrt(1 - decay * decay) * rng.standard_normal()


@dataclasses.dataclass(frozen=True)
class TelegraphT1:
    """T1 switching at random between the calibrated value and
    ``low_fraction`` of it, staying a mean ``dwell_time`` in each, as when
    a two-level defect near the qudit switches in and out of resonance."""

    low_fraction: float
    dwell_time: Time

    def __post_init__(self) -> None:
        if not 0 < self.low_fraction <= 1:
            raise InvalidValueError(
                f"TelegraphT1 low_fraction is in (0, 1], got {self.low_fraction!r}"
            )
        if not _seconds(self.dwell_time, "TelegraphT1 dwell_time") > 0:
            raise InvalidValueError(
                f"TelegraphT1 dwell_time must be positive, got {self.dwell_time!r}"
            )


def _hour() -> Time:
    return Time(3600 * _SECOND_NS)


@dataclasses.dataclass(frozen=True)
class ParameterDrift:
    """The built-in drift model: a qudit's frequency, T1 and drive
    amplitude drift independently, each part ``None`` to leave it fixed.

    The defaults are of the size measured on superconducting transmons
    (see the user guide): the frequency wanders by about 20 kHz over an
    hour (``OrnsteinUhlenbeck(20e3, 3600 * pulse.s)``), T1 drops to half
    its calibrated value and back, staying about ten minutes in each
    (``TelegraphT1(0.5, 600 * pulse.s)``), and the drive amplitude wanders
    by about 0.2% over an hour (``OrnsteinUhlenbeck(0.002, 3600 * pulse.s)``).
    They are slow: a run of seconds barely drifts, so an example that
    wants visible drift passes faster parts."""

    frequency: OrnsteinUhlenbeck | None = dataclasses.field(
        default_factory=lambda: OrnsteinUhlenbeck(20e3, _hour())
    )
    t1: TelegraphT1 | None = dataclasses.field(
        default_factory=lambda: TelegraphT1(0.5, Time(600 * _SECOND_NS))
    )
    amplitude: OrnsteinUhlenbeck | None = dataclasses.field(
        default_factory=lambda: OrnsteinUhlenbeck(0.002, _hour())
    )

    def __post_init__(self) -> None:
        for name, kind in (
            ("frequency", OrnsteinUhlenbeck),
            ("t1", TelegraphT1),
            ("amplitude", OrnsteinUhlenbeck),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, kind):
                raise InvalidTypeError(
                    f"ParameterDrift {name} is a {kind.__name__} or None, got {value!r}"
                )
        if self.frequency is None and self.t1 is None and self.amplitude is None:
            raise InvalidValueError(
                "ParameterDrift(frequency=None, t1=None, amplitude=None) drifts nothing; "
                "leave parameter_drift out for a system without drift"
            )

    @property
    def correlation_time(self) -> Time:
        """The shortest time scale of its parts: a correlation time or a
        T1 dwell time."""
        parts = (self.frequency, self.amplitude)
        times = [float(p.correlation_time) for p in parts if p is not None]
        if self.t1 is not None:
            times.append(float(self.t1.dwell_time))
        return Time(min(times))

    def start(self, calibrated: PhysicalQudit, rng: np.random.Generator) -> PhysicalQudit:
        """The calibrated qudit: drift starts from a freshly calibrated chip."""
        return calibrated

    def advance(
        self,
        true: PhysicalQudit,
        calibrated: PhysicalQudit,
        seconds: float,
        rng: np.random.Generator,
    ) -> PhysicalQudit:
        changes: dict[str, Any] = {}
        if self.frequency is not None:
            calibrated_hz = calibrated._frame_hz
            offset = true._frame_hz - calibrated_hz
            changes["frequency_hz"] = calibrated_hz + self.frequency._step(offset, seconds, rng)
        if self.t1 is not None:
            low = calibrated.t1_us * self.t1.low_fraction
            high = true.t1_us != low or low == calibrated.t1_us
            # Switches in *seconds*: Poisson at 1 / dwell_time; an odd number flips it.
            rate = _SECOND_NS / float(self.t1.dwell_time)
            if int(rng.poisson(rate * seconds)) % 2:
                high = not high
            changes["t1_us"] = calibrated.t1_us if high else low
        if self.amplitude is not None:
            nominal = _drive_scale(calibrated)
            relative = _drive_scale(true) / nominal - 1.0
            scale = nominal * (1.0 + self.amplitude._step(relative, seconds, rng))
            drive = true.drive if true.drive is not None else MicrowaveDrive()
            changes["drive"] = drive.replace(amplitude_scale_rad_per_ns=scale)
        return true.replace(**changes)


class _OneQudit:
    """The part of a description ``drive_amplitude_scale`` reads, for one qudit."""

    __slots__ = ("name", "qudits")

    def __init__(self, qudit: PhysicalQudit) -> None:
        self.name = qudit.label or f"qudit {qudit.index}"
        self.qudits = {qudit.index: qudit}


def _drive_scale(qudit: PhysicalQudit) -> float:
    """The rad/ns per unit amplitude the simulator drives *qudit* with."""
    from .base import _Description

    one: Any = _OneQudit(qudit)
    return _Description.drive_amplitude_scale(one, qudit.index)


# ── Building a drifting system ──────────────────────────────────────────────


def _models(system: QuantumSystem, parameter_drift: Any) -> dict[int, Any]:
    """The model of each drifting qudit of *system*, by carrier index."""
    carriers = system.physical_qudits
    if parameter_drift is True:
        default = ParameterDrift()
        return dict.fromkeys(carriers, default)
    if isinstance(parameter_drift, Mapping):
        unknown = [k for k in parameter_drift if k.__class__ is not int or k not in carriers]
        if unknown:
            raise InvalidValueError(
                f"parameter_drift names qudit {unknown[0]!r}, which quantum system "
                f"{system.name!r} does not define (it defines qudits "
                f"{', '.join(map(str, carriers))})"
            )
        models = {k: parameter_drift[k] for k in carriers if k in parameter_drift}
    else:
        models = dict.fromkeys(carriers, parameter_drift)
    for model in models.values():
        if not isinstance(model, ParameterDriftModel):
            raise InvalidTypeError(
                "parameter_drift is True, a drift model (a ParameterDrift, or an object with "
                "start(calibrated, rng) and advance(true, calibrated, seconds, rng)) or a "
                f"dict of them by qudit; got {type(model).__name__}"
            )
    return models


def _correlation_ns(models: Mapping[int, Any]) -> float | None:
    """The shortest correlation time of *models*, in ns; ``None`` when no
    model has one."""
    times = []
    for model in models.values():
        time = getattr(model, "correlation_time", None)
        if time is not None:
            times.append(float(time))
    return min(times, default=None)


def _drifting(system: QuantumSystem, parameter_drift: Any, seed: int | None) -> QuantumSystem:
    """*system* with *parameter_drift*: the same calibrated values, its own
    true values. ``None`` or ``False`` is *system* without drift."""
    if parameter_drift is None or parameter_drift is False:
        return system if system.__class__ is QuantumSystem else system._twin
    if system.model is None:
        raise UnsupportedError(
            f"quantum system {system.name!r} has no simulation model (its model is None), so "
            "nothing would play its true values; parameter_drift needs a model"
        )
    if seed is not None and (seed.__class__ is not int or seed < 0):
        raise InvalidTypeError(f"parameter_drift_seed is a non-negative int or None, got {seed!r}")
    twin = system if system.__class__ is QuantumSystem else system._twin
    models = _models(twin, parameter_drift)
    streams = np.random.SeedSequence(seed)
    rngs = {
        carrier: np.random.default_rng(
            np.random.SeedSequence(streams.entropy, spawn_key=(carrier,))
        )
        for carrier in models
    }
    calibrated = twin.physical_qudits
    true = dict(calibrated)
    for carrier, model in models.items():
        true[carrier] = _checked(model.start(calibrated[carrier], rngs[carrier]), carrier, "start")
    return _DriftingQuantumSystem._made(twin, models, rngs, true, 0.0)


def _checked(qudit: Any, carrier: int, method: str) -> PhysicalQudit:
    """*qudit*, what a drift model's *method* returned for qudit *carrier*,
    if it is that qudit's ``PhysicalQudit`` with values a quantum system can
    have: checked here, so the error names the model's method and not the
    system's definition at the next ``simulate``."""
    if qudit.__class__ is not PhysicalQudit or qudit.index != carrier:
        if isinstance(qudit, _HeldQudit):
            raise UnsupportedError(
                f"qudit {carrier} gives no absolute frequency_hz (it is {qudit.frequency_hz!r}), "
                "so its true value has no frequency to drift from; parameter_drift needs qudits "
                "with frequencies"
            )
        raise InvalidTypeError(
            f"a drift model's {method}() returns the qudit's PhysicalQudit (index {carrier}), "
            f"as true.replace(...) makes; got {qudit!r}"
        )
    problems = _qudit_value_problems(qudit, f"{method}()")
    if problems:
        frequency = qudit.frequency_hz
        # Whatever class the model returned: a true value without an
        # absolute frequency is refused, on a system with frequencies and on
        # one whose qudits give offsets or none.
        drifts = (
            ""
            if frequency is not None and _frequency_frame(frequency) == "absolute"
            else f" (its frequency_hz is {frequency!r}: parameter_drift needs qudits with "
            "frequencies, and a true value keeps one)"
        )
        raise InvalidValueError(
            f"a drift model's {method}() returned qudit {carrier} with values a quantum system "
            f"cannot have{drifts}",
            problems=problems,
        )
    return qudit


def _replaced(system: QuantumSystem, changes: dict[str, Any]) -> QuantumSystem:
    """``system.replace(**changes)`` where *changes* name ``parameter_drift``
    or ``parameter_drift_seed``: a variant with that drift (none without
    ``parameter_drift``). Without other changes it keeps the calibration, so
    its fingerprint is *system*'s."""
    changes = dict(changes)
    parameter_drift = changes.pop("parameter_drift", None)
    seed = changes.pop("parameter_drift_seed", None)
    twin = system if system.__class__ is QuantumSystem else system._twin
    if changes:
        twin = twin.replace(**changes)
    if seed is not None and (parameter_drift is None or parameter_drift is False):
        raise InvalidValueError("parameter_drift_seed seeds parameter_drift, which is not given")
    return _drifting(twin, parameter_drift, seed)


class _DriftingQuantumSystem(QuantumSystem):
    """A quantum system with parameter drift (see the module docstring):
    equal to its non-drifting twin, which it compiles as."""

    _drift = True

    # Set by _made.
    _twin: QuantumSystem
    _models: dict[int, Any]
    _rngs: dict[int, np.random.Generator]
    _true: dict[int, PhysicalQudit]
    _true_view: Mapping[int, PhysicalQudit]
    _clock_ns: float
    _played: QuantumSystem | None
    # The program simulated last and its duration in ns, which every call reads.
    _shot: tuple[ScheduledKernel | None, float]
    _drift_lock: threading.Lock
    _correlation: float | None

    @classmethod
    def _made(
        cls,
        twin: QuantumSystem,
        models: dict[int, Any],
        rngs: dict[int, np.random.Generator],
        true: dict[int, PhysicalQudit],
        clock_ns: float,
    ) -> _DriftingQuantumSystem:
        made = cls.__new__(cls)
        state = vars(made)
        state.update(vars(twin))
        state.pop("_pickled", None)
        made.qudits = _Handles(tuple(Qudit(made, p, i) for i, p in enumerate(twin.layout)))
        made._twin = twin
        made._models = models
        made._rngs = rngs
        made._true = true
        made._true_view = MappingProxyType(true)
        made._clock_ns = clock_ns
        made._played = None
        made._shot = (None, 0.0)
        made._drift_lock = threading.Lock()
        made._correlation = _correlation_ns(models)
        return made

    @property
    def true_qudits(self) -> Mapping[int, PhysicalQudit]:
        """The qudits' true parameters now, by carrier index as
        ``physical_qudits``: what ``simulate`` plays."""
        return self._true_view

    @property
    def parameter_drift(self) -> Mapping[int, Any]:
        """The drift model of each drifting qudit, by carrier index."""
        return MappingProxyType(self._models)

    @property
    def parameter_drift_time(self) -> Time:
        """The drift time since the system was built."""
        return Time(self._clock_ns)

    def advance(self, duration: Time) -> None:
        """Move drift time on by *duration*, a ``Time`` (``20 * pulse.ms``)."""
        if duration.__class__ is not Time:
            raise InvalidTypeError(
                "advance() takes a time, as 20 * pulse.ms; got "
                f"{type(duration).__name__} {duration!r}"
            )
        if not float(duration) >= 0:
            raise InvalidValueError(f"drift time only moves forward; got {duration!r}")
        self._advance_ns(float(duration))

    def _advance_ns(self, ns: float) -> None:
        if ns <= 0:
            return
        seconds = ns / _SECOND_NS
        calibrated = self._description.qudits
        with self._drift_lock:
            true = self._true
            # Every qudit moves on, or none does: a model's bad value leaves
            # the true values and the clock as they were (the random streams
            # of the models up to and including it have still moved on).
            moved = {}
            for carrier, model in self._models.items():
                qudit = model.advance(
                    true[carrier], calibrated[carrier], seconds, self._rngs[carrier]
                )
                # The built-in model returns true.replace(...), which needs no check.
                if model.__class__ is not ParameterDrift:
                    qudit = _checked(qudit, carrier, "advance")
                moved[carrier] = qudit
            true.update(moved)
            self._clock_ns += ns
            self._played = None

    def _playing(self) -> QuantumSystem:
        """A non-drifting quantum system of the true values now, for
        ``simulate``: built once per drift time."""
        with self._drift_lock:
            played = self._played
            if played is None:
                twin = self._twin
                played = QuantumSystem.__new__(QuantumSystem)
                vars(played).update(vars(twin))
                vars(played).pop("_pickled", None)
                played._description = twin._description.replace(qudits=self._true)
                self._played = played
            return played

    @staticmethod
    def _options(arguments: Mapping[str, Any]) -> dict[str, Any]:
        """The arguments of ``_simulate_lowered`` in *arguments* (its locals)
        that ``_simulated`` passes on."""
        return {name: arguments[name] for name in _OPTIONS}

    def _simulated(
        self,
        run: Callable[..., SimulationResult],
        lowered: ScheduledKernel,
        reset_time: Time | None,
        options: dict[str, Any],
    ) -> SimulationResult:
        """``simulate`` of *lowered* on this system: *run* is
        ``_simulate_lowered`` and *options* its other arguments."""
        reset_ns = _reset_ns(reset_time, lowered, self)
        shots_count = options["shots_count"]
        shots = shots_count or 1
        kept, duration_ns = self._shot
        if kept is not lowered:
            duration_ns = float(lowered.duration)
            self._shot = (lowered, duration_ns)
        shot_ns = duration_ns + reset_ns
        correlation = self._correlation
        chunks = 1
        if shots_count is not None and correlation is not None and correlation > 0:
            chunks = max(1, min(shots_count, int(shots * shot_ns // correlation)))
        if chunks == 1:
            result = run(lowered, quantum_system=self._playing(), **options)
            self._advance_ns(shots * shot_ns)
            return result
        base, extra = divmod(shots, chunks)
        sizes = [base + (i < extra) for i in range(chunks)]
        seed = options["seed"]
        seeds: list[int | None]
        if seed is None:
            seeds = [None] * chunks
        else:
            spawned = np.random.SeedSequence(seed).spawn(chunks)
            seeds = [int(s.generate_state(1)[0]) for s in spawned]
        results = []
        for size, chunk_seed in zip(sizes, seeds, strict=True):
            chunk = dict(options, shots_count=size, seed=chunk_seed)
            results.append(run(lowered, quantum_system=self._playing(), **chunk))
            self._advance_ns(size * shot_ns)
        return _merged(results, sizes)

    def __reduce__(self) -> tuple[Any, ...]:
        with self._drift_lock:
            state = (dict(self._true), dict(self._rngs), self._clock_ns)
        return (_rebuilt, (self._twin, self._models, *state))

    def __repr__(self) -> str:
        return (
            f"<QuantumSystem {self.name} ({len(self.layout)} qudits, parameter drift on "
            f"{len(self._models)})>"
        )


def _rebuilt(
    twin: QuantumSystem,
    models: dict[int, Any],
    true: dict[int, PhysicalQudit],
    rngs: dict[int, np.random.Generator],
    clock_ns: float,
) -> _DriftingQuantumSystem:
    return _DriftingQuantumSystem._made(twin, models, rngs, true, clock_ns)


# ── Compiling and simulating ────────────────────────────────────────────────


def _rebound(lowered: ScheduledKernel, system: QuantumSystem) -> ScheduledKernel:
    """*lowered*, a view of the same module, for *system*."""
    view = lowered.__class__.__new__(lowered.__class__)
    for cls in lowered.__class__.__mro__:
        for name in cls.__dict__.get("__slots__", ()):
            try:
                value = getattr(lowered, name)
            except AttributeError:
                continue
            object.__setattr__(view, name, value)
    if hasattr(lowered, "__dict__"):
        vars(view).update(vars(lowered))
    view.quantum_system = system
    return view


def _entry_for(entry: _Entry, system: QuantumSystem) -> _Entry:
    """A kernel cache entry whose program is *system*'s."""
    if entry.lowered.quantum_system is system:
        return entry
    made = entry.__class__.__new__(entry.__class__)
    for name in entry.__class__.__slots__:
        try:
            object.__setattr__(made, name, getattr(entry, name))
        except AttributeError:
            continue
    made.lowered = _rebound(entry.lowered, system)
    return made


def _kept(entry: _Entry, system: QuantumSystem, kernel: Any) -> tuple[_Entry, _Entry]:
    """The entry to keep in *kernel*'s cache for *entry*, compiled for
    *system* (drifting): its twin's, so no cached program plays drift for
    another system; and the entry to return. The specializations the kernel
    keeps to schedule again are made its twin's too."""
    twin = system._twin
    traced = kernel._traced
    for key, kept in list(traced.items()):
        if kept.lowered.quantum_system is system:
            traced[key] = _entry_for(kept, twin)
    return _entry_for(entry, twin), _entry_for(entry, system)


def _reset_ns(reset_time: Any, lowered: ScheduledKernel, system: QuantumSystem) -> float:
    if reset_time is None:
        qudits = system._twin._description.qudits
        return 5e3 * max((float(qudits[q].t1_us) for q in lowered.qudits), default=0.0)
    if reset_time.__class__ is not Time:
        raise InvalidTypeError(
            "simulate's reset_time is a time, as 100 * pulse.us; got "
            f"{type(reset_time).__name__} {reset_time!r}"
        )
    reset_ns = float(reset_time)
    if not reset_ns >= 0:
        raise InvalidValueError(f"simulate's reset_time must not be negative, got {reset_time!r}")
    return reset_ns


_OPTIONS = (
    "levels",
    "initial_state",
    "shots_count",
    "seed",
    "backend",
    "t_start",
    "t_end",
    "dt_ns",
    "integrator",
    "dephasing",
    "handlers",
    "readout",
    "latency_branching",
    "branches",
)


def _merged(results: list[SimulationResult], sizes: list[int]) -> SimulationResult:
    """One result of the chunks *results* of a call, of *sizes* shots each:
    the state and the branch probabilities weighted by shots, the records
    in shot order."""
    from ..runtime.simulate import Branch, SimulationResult

    total = sum(sizes)
    weights = [size / total for size in sizes]
    density = sum(w * r.density_matrix for w, r in zip(weights, results, strict=True))
    merged: dict[tuple[Any, ...], float] = {}
    for w, r in zip(weights, results, strict=True):
        for b in r.branches:
            key = (b.levels, b.calls)
            merged[key] = merged.get(key, 0.0) + w * b.probability
    records = [r.records for r in results if r.records is not None]
    first = results[0]
    return SimulationResult(
        density_matrix=np.asarray(density),
        dims=first.dims,
        qudits=first.qudits,
        measurements=first.measurements,
        branches=[Branch(levels, p, calls) for (levels, calls), p in merged.items()],
        records=np.concatenate(records) if records else None,
        times_ns=first.times_ns,
    )
