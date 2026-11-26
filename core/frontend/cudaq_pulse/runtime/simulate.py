# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation of pulse programs that measure and branch on the outcome.

The program evolves in intervals between measurements. A measurement of
qudit ``q`` at the end of its readout projects the state onto each level
``k`` of ``q``; each outcome continues along its own path, where branches on
that measurement take the arm the outcome selects. Summing the unnormalized
state of every path gives the ensemble

    rho' = sum_m U_m P_m rho P_m^dagger U_m^dagger

exactly, and with ``shots_count`` the branch weights also sample per-shot records.

Intervals are lowered by the native pipeline (pulse -> qop) along the path,
then evolved either by a NumPy reference integrator (``backend="cpu"``) or
by the cuDensityMat runtime (``backend="gpu"``), from the projected state.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from .._errors import (
    ExecutionError,
    InvalidTypeError,
    InvalidValueError,
    NotFoundError,
    UnsupportedError,
    _positive_int,
)
from .._native._cudaq_pulse_native import evaluate_callback
from .._units import Time, as_nanoseconds
from ..compiler import _LISTED_EVENTS, ScheduledEvent
from . import _propagate
from ._attributes import _VALID_INTEGRATORS, _f64, _level_attributes, _target_attributes

try:
    # hashlib's own blake2b, without hashlib's import of OpenSSL (a millisecond).
    from _blake2 import blake2b as _blake2b
except ImportError:  # pragma: no cover - CPython builds without the _blake2 module
    from hashlib import blake2b as _blake2b

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from .._device import Qudit
    from ..compiler import ScheduledKernel
    from ..kernel._kernel import CompiledKernel, Kernel
    from ..quantum_systems.models import SimulationModel
    from ._calls import CallRecord

__all__ = ["Branch", "BranchCoverage", "Measurement", "SimulationResult", "simulate"]

# Branches whose weight falls below this fraction of the initial state's are
# not followed.
_NEGLIGIBLE = 1e-14


class BranchCoverage(StrEnum):
    """Which measurement branches ``simulate`` follows; the strings
    ``"all"`` and ``"sampled"`` mean the same."""

    #: Every outcome of every measurement: the exact ensemble.
    ALL = "all"
    #: Only the outcomes the shots draw, at most ``shots_count`` branches.
    SAMPLED = "sampled"


@dataclass(frozen=True)
class Measurement:
    """One execution of a readout: which one, on which (quantum system) qudit, and
    when."""

    readout: int
    iteration: tuple[int, ...]
    qudit: int
    start_ns: float
    end_ns: float


@dataclass(frozen=True)
class Branch:
    """One path of the ensemble: the levels its measurements gave, its
    probability, and the classical calls it made.

    ``levels`` holds one entry per entry of ``SimulationResult.measurements``,
    -1 where this branch did not measure. With ``branches="sampled"``,
    ``probability`` is the fraction of the shots that took the branch."""

    levels: tuple[int, ...]
    probability: float
    #: The classical calls this branch made, in time order.
    calls: tuple[CallRecord, ...] = ()

    def answers(self, function: str) -> list[Any]:
        """The answers of this branch's calls to *function* (its name), in
        call order. A function it did not call is a ``KeyError`` listing
        those it called."""
        found = [call.answer for call in self.calls if call.function == function]
        if not found:
            called = list(dict.fromkeys(call.function for call in self.calls))
            raise NotFoundError(
                f"this branch made no call to {function!r}; it called "
                + (", ".join(map(repr, called)) if called else "no classical function")
            )
        return found


@dataclass
class SimulationResult:
    """The ensemble final state, and per-shot records if shots were taken.

    The state spans the qudits the program uses, ``qudits`` (quantum system
    indices, increasing), with ``dims[i]`` levels on ``qudits[i]``;
    ``density_matrix`` is indexed with ``qudits[0]`` as the least
    significant digit. Qudits the program does not use are left out, with
    their couplings, as classical spectators in their ground state.
    ``measurements`` lists every measurement any path made, in the order of
    their end times; ``branches`` holds one ``Branch`` per path, with its
    measured levels (one per entry of ``measurements``, -1 where that path
    did not measure), its probability and its classical calls.
    ``records`` holds one row of levels per shot. With
    ``branches="sampled"``, ``branches`` holds the branches the shots took,
    each with the fraction of shots that took it, and the state is their
    average. ``postselect`` gives the result over the branches a test
    keeps, and ``acceptance`` the fraction of the ensemble it kept.
    """

    density_matrix: npt.NDArray[np.complex128]
    dims: tuple[int, ...]
    qudits: tuple[int, ...]
    measurements: list[Measurement]
    branches: list[Branch]
    records: npt.NDArray[np.int64] | None = None
    times_ns: tuple[float, float] = (0.0, 0.0)
    #: The probability of the branches this result keeps: 1.0 unless
    #: ``postselect`` made it.
    acceptance: float = 1.0
    _states: list[Any] = field(default_factory=list, repr=False)

    def postselect(self, keep: Callable[[Branch], bool], /) -> SimulationResult:
        """The result over the branches *keep* accepts, a test of a
        ``Branch``, as ``lambda b: b.levels[0] == 0`` or ``lambda b:
        b.answers("herald")[-1]``: its state is theirs, normalized, each
        kept branch's probability is divided by their total, which is the
        result's ``acceptance`` (times this result's), and ``records``
        holds the kept branches' shots. Where branches of the same levels
        are both kept and not (calls that took IQ points, or timeouts with
        ``latency_branching``), a shot's levels do not say which it took,
        and ``records`` is ``None``.

        A test that keeps no branch raises ``InvalidValueError``. A result
        merged from the shot chunks of a quantum system with parameter drift
        holds no state per branch, and raises ``UnsupportedError``."""
        branches = self.branches
        states = self._states
        if len(states) != len(branches):
            raise UnsupportedError(
                "this result holds no state per branch to post-select (a simulation of a "
                "quantum system with parameter drift merges its shot chunks); post-select its "
                "records, or simulate a system without drift"
            )
        kept = [bool(keep(branch)) for branch in branches]
        accepted = sum(b.probability for b, k in zip(branches, kept, strict=True) if k)
        if not accepted > 0.0:
            raise InvalidValueError(
                f"the test keeps none of the {len(branches)} branches, or only branches of "
                "probability 0; nothing is left to post-select"
            )
        dim = self.density_matrix.shape[0]
        rho = np.zeros((dim, dim), dtype=np.complex128)
        for branch, state, k in zip(branches, states, kept, strict=True):
            if not k or branch.probability <= 0.0:
                continue
            density = _Backend.density(state)
            rho += density * (branch.probability / float(np.real(np.trace(density))))
        records = self.records
        if records is not None:
            taken = {b.levels for b, k in zip(branches, kept, strict=True) if k}
            if taken.isdisjoint(b.levels for b, k in zip(branches, kept, strict=True) if not k):
                rows = [tuple(row) in taken for row in records.tolist()]
                records = records[np.array(rows, dtype=bool)] if rows else records
            else:
                records = None
        return SimulationResult(
            density_matrix=rho / accepted,
            dims=self.dims,
            qudits=self.qudits,
            measurements=self.measurements,
            branches=[
                replace(b, probability=b.probability / accepted)
                for b, k in zip(branches, kept, strict=True)
                if k
            ],
            records=records,
            times_ns=self.times_ns,
            acceptance=self.acceptance * accepted,
            _states=[s for s, k in zip(states, kept, strict=True) if k],
        )

    def level_populations(self, qudit: int | Qudit | None = None) -> list[float]:
        """The probability of each level of *qudit*, from level 0 up: the
        diagonal of its reduced density matrix, summing to 1.

        *qudit* is a quantum system qudit index or a qudit handle, one of
        ``qudits``; it may be left out when the result models one qudit."""
        mode = self._mode(qudit)
        dims = self.dims
        diagonal = np.real(np.diagonal(self.density_matrix)).reshape(dims[::-1])
        axis = len(dims) - 1 - mode
        others = tuple(a for a in range(len(dims)) if a != axis)
        return [float(p) for p in diagonal.sum(axis=others)]

    def level_population(self, level: int, qudit: int | Qudit | None = None) -> float:
        """The probability that *qudit* is in *level*, as in
        ``level_populations(qudit)[level]``."""
        populations = self.level_populations(qudit)
        if level.__class__ is not int or not 0 <= level < len(populations):
            which = f"qudit {self.qudits[self._mode(qudit)]}"
            raise InvalidValueError(
                f"{which} has levels 0 to {len(populations) - 1} in this result "
                f"({len(populations)} levels); got level {level!r}"
            )
        return populations[level]

    def _mode(self, qudit: int | Qudit | None) -> int:
        """The position in ``qudits`` of *qudit*, a quantum system index or handle."""
        qudits = self.qudits
        if qudit is None:
            if len(qudits) == 1:
                return 0
            raise InvalidValueError(
                f"the result models qudits {list(qudits)}; pass qudit=, one of them"
            )
        index = qudit if isinstance(qudit, int) else getattr(qudit, "index", None)
        if index.__class__ is not int or index not in qudits:
            raise InvalidValueError(
                f"qudit {qudit!r} is not in the result; it models qudits {list(qudits)}"
            )
        return qudits.index(index)


def simulate(
    program: Kernel[...] | CompiledKernel[...] | ScheduledKernel,
    /,
    *args: Any,
    levels: int | dict[int, int] | None = None,
    initial_state: npt.ArrayLike | None = None,
    shots_count: int | None = None,
    seed: int | None = None,
    backend: str = "cpu",
    t_end: float | Time | None = None,
    dt_ns: float | Time | None = None,
    integrator: str = "rk4",
    dephasing: str = "t2_star",
    handlers: Mapping[str, Callable[..., Any]] | None = None,
    readout: str = "ideal",
    latency_branching: bool = False,
    quantum_system: Any = None,
    branches: BranchCoverage | str = BranchCoverage.ALL,
    reset_time: Time | None = None,
) -> SimulationResult:
    """Simulate one kernel call on the quantum system it is compiled for.

    ``simulate(kernel, *args)`` compiles the call as ``kernel.lower(*args)``
    does and simulates it, as ``compile(kernel)`` takes the kernel first and
    its options by keyword. A program compiled already is simulated without
    compiling it again, such as one a worker process receives pickled:
    ``simulate(program, *values)`` binds the values of its run-time float
    parameters as ``program.bind(values)`` does.

    Parameters
    ----------
    program:
        A ``@cudaq_pulse.kernel`` or the ``CompiledKernel`` of one, called
        with *args*; a module with run-time float parameters, such as
        ``lowered.generic``, bound with *args*; or the module of one call,
        from ``kernel.lower(*args)`` or ``bind``, with no *args*.
    args:
        For a kernel, its arguments by position: the qudit handles, then its
        other parameters. For a module with run-time parameters, their
        values, in the order of ``program.parameters``.
    levels:
        Levels per qudit. By default each qudit has the levels its quantum system's
        simulation model declares for it (``QuantumSystem.model``; 3 for
        a ``TransmonModel``); an int sets every qudit's, and a dict by quantum system
        qudit index sets those it names. A transmon with three or more
        levels includes its anharmonicity.
    initial_state:
        A state vector or density matrix over ``lowered.qudits``, the
        qudits the program uses, with the first the least significant
        digit (the layout of ``SimulationResult.density_matrix``). By default each qudit starts in
        its thermal mixture: its excited level holds the quantum system's
        ``thermal_population`` p, and higher levels follow the same
        Boltzmann ratio p / (1 - p) per level (``|0...0>`` when every p is
        0). T1 then relaxes it back there, with downward and upward rates
        (1 - p) / T1 and p / T1, whatever the initial state.
    shots_count:
        Sample this many per-shot measurement records: a positive int.
        ``None`` (the default) samples none.
    seed:
        Seed for sampling the records.
    backend:
        ``"cpu"`` for the NumPy reference integrator (RK4), ``"gpu"`` for
        the cuDensityMat runtime.
    t_end:
        End of the simulation: a time (``40 * pulse.ns``) or a number of
        ns; the end of the schedule by default.
    dt_ns:
        Integration step: a time or a number of ns. By default 0.05 ns, or
        shorter when the simulation holds a faster frequency: a coupling
        turns at its qudits' detuning, and a qudit's levels at the spread of
        their energies in its frame (of four or more transmon levels, a
        multiple of its anharmonicity). The step resolves it with 20 steps
        per period (0.024 ns at 2.1 GHz), a whole fraction of a clock
        cycle. A step passed here that takes
        fewer than 20 steps per period of the fastest frequency the
        simulation holds, a qudit's anharmonicity with three or more levels
        or the detuning of coupled qudits, warns with a ``PulseWarning``
        that names that frequency and a step that resolves it. A step so
        coarse that the integrator diverges raises ``ExecutionError``,
        naming the same and the line that called ``simulate``, rather
        than returning a state that is not finite.
    integrator:
        cuDensityMat integrator for the GPU backend: ``"rk1"``, ``"rk2"``,
        ``"rk4"``, ``"magnus"`` or ``"crank_nicolson"``.
    dephasing:
        Dephase at each qudit's Ramsey T2* (``"t2_star"``, the default) or
        its Hahn-echo T2 (``"t2_echo"``), which every simulated qudit must
        have.
    handlers:
        For a program that calls classical functions, what answers each call,
        by function name: a Python callable taking the call's arguments (bits
        and levels as lists of ints, numbers as floats), by default the
        function's own body. Each call's answer is bound to its path, and the
        path is scheduled again, so answers that set amplitudes, frequencies,
        branches or wait lengths take effect exactly.
    readout:
        ``"ideal"`` (the default) reports the measured level. ``"model"``
        reports through each qubit's assignment errors (its
        ``DispersiveReadout``'s ``error_1to0`` and ``error_0to1``) and ``"model_with_decay"`` adds
        relaxation during the readout window; branches and calls see the
        reported outcome, and the state is projected on the true one. A
        call that takes IQ points (``complex`` or ``list[complex]``) gets
        points synthesized per shot at each readout's true level, from its
        tone's frequency and its pulse's amplitude (``IQModel``; with
        ``"model_with_decay"``, relaxation during the window too); it needs
        ``shots_count``, and each shot from the call on is simulated as its own
        trajectory.
    quantum_system:
        A ``QuantumSystem`` to simulate instead of the one the program
        was compiled for: its qudits' frequencies, T1, T2, thermal
        populations, drive strength and readout parameters apply, while the
        program (its tones and pulses) stays as compiled. A system with
        parameter drift (``cudaq_pulse.quantum_systems.drift``) plays its
        true values, ``true_qudits``, whether it is this one or the one the
        program is compiled for, and the call advances its drift time.
    latency_branching:
        With a timeout, branch on whether the answer came in time with the
        probability of the function's latency model; by default, without
        shots, the answer is in time when the nominal latency is within the
        timeout. With shots the in-time outcome is always sampled.
    branches:
        ``"all"`` (the default) follows every outcome of every measurement,
        so the state is the exact ensemble. ``"sampled"`` follows only the
        outcomes the shots draw: at most ``shots_count`` branches, so a program
        that measures many times inside one shot (an adaptive loop of
        readouts and calls) does not cost time exponential in its
        measurements. A loop that waits each round on a variable or blocking
        wait costs the same host time per round at any length; other programs
        with calls are scheduled again whole after each answer, so their host
        time per interval grows with the rounds before it. The state is then
        the average over the shots'
        branches, and ``result.branches`` gives each one's share of the
        shots. It needs ``shots_count``. A ``BranchCoverage`` or its string.
    reset_time:
        The time between shots, after each run of the program, as
        ``200 * pulse.us``: a system with parameter drift advances its drift
        time by ``shots_count`` (or one) times the program's duration plus
        this. By default 5 times the longest calibrated T1 of the program's
        qudits, a passive reset. It changes nothing else, and nothing for a
        system without drift. A call whose shots take longer than the
        shortest correlation time of the drift models is split into chunks
        of whole shots, each at least that long, and each plays the true
        values at its start; the result merges them, weighted by shots, with
        the records in shot order.
    """
    from ..compiler import ScheduledKernel
    from ..kernel._kernel import CompiledKernel, Kernel

    if isinstance(program, ScheduledKernel):
        if not args:
            lowered = program
        elif program.parameters:
            lowered = program.bind(args)
        else:
            raise InvalidTypeError(
                "simulate() got arguments with the module of a call, which has its "
                "arguments already; pass the kernel and its arguments, as in "
                "simulate(kernel, q), or the module alone"
            )
    elif isinstance(program, (Kernel, CompiledKernel)):
        lowered = program.lower(*args)
    else:
        raise InvalidTypeError(
            "simulate() expects a kernel and its arguments, as in simulate(kernel, q), "
            "a module and the values of its run-time parameters, as in "
            "simulate(lowered.generic, 0.5), or the module of one call; "
            f"got {type(program).__name__}"
        )
    if reset_time is not None:
        options: dict[str, Any] = {
            "levels": levels,
            "initial_state": initial_state,
            "shots_count": shots_count,
            "seed": seed,
            "backend": backend,
            "t_start": 0.0,
            "t_end": t_end,
            "dt_ns": dt_ns,
            "integrator": integrator,
            "dephasing": dephasing,
            "handlers": handlers,
            "readout": readout,
            "latency_branching": latency_branching,
            "branches": branches,
        }
        system = lowered.quantum_system if quantum_system is None else quantum_system
        if getattr(system, "_drift", False) is True:
            return system._simulated(_simulate_lowered, lowered, reset_time, options)  # type: ignore[no-any-return]
        from ..quantum_systems.drift import _reset_ns

        _reset_ns(reset_time, lowered, system)
        return _simulate_lowered(lowered, quantum_system=quantum_system, **options)
    return _simulate_lowered(
        lowered,
        levels=levels,
        initial_state=initial_state,
        shots_count=shots_count,
        seed=seed,
        backend=backend,
        t_end=t_end,
        dt_ns=dt_ns,
        integrator=integrator,
        dephasing=dephasing,
        handlers=handlers,
        readout=readout,
        latency_branching=latency_branching,
        quantum_system=quantum_system,
        branches=branches,
    )


def _simulate_lowered(
    lowered: ScheduledKernel,
    *,
    levels: int | dict[int, int] | None = None,
    initial_state: npt.ArrayLike | None = None,
    shots_count: int | None = None,
    seed: int | None = None,
    backend: str = "cpu",
    t_start: float = 0.0,
    t_end: float | Time | None = None,
    dt_ns: float | Time | None = None,
    integrator: str = "rk4",
    dephasing: str = "t2_star",
    handlers: Mapping[str, Callable[..., Any]] | None = None,
    readout: str = "ideal",
    latency_branching: bool = False,
    quantum_system: Any = None,
    branches: BranchCoverage | str = BranchCoverage.ALL,
) -> SimulationResult:
    if shots_count is not None and (shots_count.__class__ is not int or shots_count < 1):
        shots_count = _positive_int("shots_count", shots_count)
    if branches not in ("all", "sampled"):
        raise InvalidValueError(f"unknown branches {branches!r}; choose 'all' or 'sampled'")
    if branches == "sampled" and not shots_count:
        raise InvalidValueError(
            "branches='sampled' follows the branches of the shots; pass shots_count >= 1"
        )
    if backend not in ("cpu", "gpu"):
        raise InvalidValueError(f"unknown backend {backend!r}; choose 'cpu' or 'gpu'")
    if dephasing not in ("t2_star", "t2_echo"):
        raise InvalidValueError(f"unknown dephasing {dephasing!r}; choose 't2_star' or 't2_echo'")
    if readout not in ("ideal", "model", "model_with_decay"):
        raise InvalidValueError(
            f"unknown readout {readout!r}; choose 'ideal', 'model' or 'model_with_decay'"
        )
    if t_end is not None:
        t_end = as_nanoseconds(t_end)
    if integrator not in _VALID_INTEGRATORS:
        raise InvalidValueError(
            f"Unknown integrator {integrator!r}. "
            f"Choose from: {', '.join(sorted(_VALID_INTEGRATORS))}"
        )
    if lowered.parameters:
        raise InvalidValueError(
            "the module has float parameters "
            f"({', '.join(lowered.parameters)}); pass their values, as in "
            "simulate(program, 0.5), or lower the kernel with them"
        )
    compiled_for = lowered.quantum_system
    if quantum_system is None:
        if compiled_for._drift:
            options = compiled_for._options(locals())
            return compiled_for._simulated(_simulate_lowered, lowered, None, options)  # type: ignore[no-any-return]
        target = compiled_for._description
    else:
        target = _check_quantum_system(lowered, quantum_system)
        if quantum_system._drift:
            options = quantum_system._options(locals())
            return quantum_system._simulated(_simulate_lowered, lowered, None, options)  # type: ignore[no-any-return]
    try:
        step = _STEPS.get(
            (
                id(compiled_for._description),
                id(target),
                compiled_for.clock_ghz,
                lowered.qudits,
                levels,
                dt_ns,
            )
        )
    except TypeError:  # levels by qudit, a dict: unhashable, and never kept
        step = None
    if step is None:
        step = _step(lowered, target, levels, dt_ns)
    # _step refused a target without a model; a target does not change.
    model: Callable[..., SimulationResult] = target.model._simulate  # type: ignore[union-attr]
    try:
        return model(
            lowered,
            target,
            levels=levels,
            initial_state=initial_state,
            shots_count=shots_count,
            seed=seed,
            backend=backend,
            t_start=t_start,
            t_end=t_end,
            dt_ns=step,
            integrator=integrator,
            clock_ghz=compiled_for.clock_ghz,
            dephasing=dephasing,
            handlers=handlers,
            readout=readout,
            latency_branching=latency_branching,
            sampled=branches == "sampled",
        )
    except _DivergedError as e:
        raise _diverged_error(lowered, target, levels, as_nanoseconds(dt_ns), e.step_ns) from None


def _step(
    lowered: ScheduledKernel,
    target: Any,
    levels: int | dict[int, int] | None,
    dt_ns: float | Time | None,
) -> float:
    """The integration step of a simulation of *lowered* on *target*: *dt_ns*,
    checked, or the default step; kept in ``_STEPS`` unless it warns or its
    model may change."""
    _simulation_model(target)
    given = as_nanoseconds(dt_ns)
    step: float
    if given is None:
        step = _default_step(lowered, target, levels)
    elif given <= 0:
        raise InvalidValueError("dt_ns must be positive")
    else:
        step = given
        # A step of fewer than 20 per period of what the simulation holds
        # warns, as the default step keeps, at every call.
        fastest, source = _fastest(target, lowered.qudits, levels)
        if fastest and fastest * step > _MAX_HZ_NS:
            _warn_coarse_step(target, source, fastest, step, lowered.quantum_system.clock_ghz)
            return step
    if levels.__class__ is not dict and target.model.__class__ in _read_only_models():
        compiled = lowered.quantum_system._description
        key = (
            id(compiled),
            id(target),
            lowered.quantum_system.clock_ghz,
            lowered.qudits,
            levels,
            dt_ns,
        )
        _remember(key, step, compiled, target)
    return step


def _remember(key: tuple[Any, ...], step: float, *targets: Any) -> None:
    """Keep *step* in ``_STEPS`` under *key*, which holds the ids of
    *targets*, and forget it when one of them is freed."""
    import weakref
    from functools import partial

    for target in targets:
        if id(target) not in _WATCHED:
            ref = weakref.ref(target, partial(_forget, id(target)))
            _WATCHED.setdefault(id(target), ref)
    if len(_STEPS) >= _MAX_STEPS:
        _STEPS.clear()
    _STEPS[key] = step


def _forget(target_id: int, _: Any = None) -> None:
    """Drop every step of the target whose id is *target_id*, which is being
    freed: run before its memory is, so no other object has its id yet."""
    _WATCHED.pop(target_id, None)
    for key in _STEPS.copy():
        if target_id in (key[0], key[1]):
            _STEPS.pop(key, None)


def _read_only_models() -> tuple[type, ...]:
    from ..quantum_systems.models import TransmonModel

    return (TransmonModel,)


# The step resolved for each compiled target, simulated target, clock,
# qudits, levels and dt_ns (by the ids of the targets): a call that hits it
# runs none of the checks that made it. Every input of the step is in the
# key or fixed with an object whose id is: targets, their qubits and
# couplings are frozen, and the models kept (_read_only_models) are
# read-only, with immutable classes. A quantum system may change its target or
# clock, both read into the key at each call. An entry holds no object; it
# is dropped when either target is freed (_forget, before the id can be
# reused), and the cache is emptied when it holds _MAX_STEPS. Each read and
# write is one dict operation on an immutable key and float, atomic in the
# free-threaded build too: a race recomputes a step, never returns another's.
_MAX_STEPS = 256
_STEPS: dict[tuple[Any, ...], float] = {}
# A weak reference to each target with steps in _STEPS, by id.
_WATCHED: dict[int, Any] = {}


def _simulation_model(target: Any) -> Callable[..., SimulationResult]:
    """The simulator of *target*'s model, the one place a system is matched
    to a model."""
    model: SimulationModel | None = target.model
    if model is None:
        raise UnsupportedError(
            f"the system {target.name!r} has no simulation model (its model is None). "
            "Export and hardware runs work."
        )
    return model._simulate


def _check_quantum_system(lowered: ScheduledKernel, quantum_system: Any) -> Any:
    """The description of *quantum_system*, which defines every qudit of
    *lowered* and measures its frequencies from what the program's are
    measured from."""
    from .._device import QuantumSystem

    if not isinstance(quantum_system, QuantumSystem):
        raise InvalidTypeError(
            f"quantum_system must be a QuantumSystem, got {type(quantum_system).__name__}"
        )
    if quantum_system._tables[0] <= max(lowered.qudits):
        # A system no program was compiled for: its frequency tables are
        # made, which refuses a readout frequency it cannot read.
        quantum_system._frequency_tables(max(lowered.qudits) + 1)
    description = quantum_system._description
    for qudit in lowered.qudits:
        if qudit not in description.qudits:
            missing = [q for q in lowered.qudits if q not in description.qudits]
            raise InvalidValueError(
                f"the quantum system does not define qudits {missing} that the program uses"
            )
    if description._frame != lowered.quantum_system._description._frame:
        raise _frames_differ(lowered.quantum_system._description, description)
    return description


# What the frequencies of a system of each frequency_frame are.
_FRAMES = {
    "absolute": "its qudits give their frequencies",
    "shared_reference": "its frequencies are offsets from a reference its qudits share",
    "own_carrier": "each of its tones is an offset from its own qudit's carrier",
}


def _frames_differ(compiled: Any, simulated: Any) -> InvalidValueError:
    """The refusal of a program compiled for *compiled* on *simulated*, a
    system whose frequencies are measured from something else: the tones
    would be numbers in one frame played on qudits in another."""
    ours, theirs = compiled._frame, simulated._frame
    return InvalidValueError(
        f"the program was compiled for quantum system {compiled.name!r}, whose frequency_frame "
        f"is {ours!r} ({_FRAMES[ours]}), and quantum_system= is {simulated.name!r}, whose "
        f"frequency_frame is {theirs!r} ({_FRAMES[theirs]}): the program's tones would be read "
        "against qudit frequencies measured from something else. Simulate it on a system of the "
        f"same frequency_frame, or compile the kernel for {simulated.name!r}"
    )


_DEFAULT_STEP_NS = 0.05
_STEPS_PER_PERIOD = 20


def _default_step(
    lowered: ScheduledKernel, target: Any = None, levels: int | dict[int, int] | None = None
) -> float:
    """The integration step: 0.05 ns, or 20 steps per period of the fastest
    frequency the simulation holds (see ``_fastest``), rounded down to a
    whole fraction of a clock cycle.

    Couplings turn at the detuning of the compiled quantum system's qudits; levels
    at the spread of the simulated *target*'s (the compiled one by default)."""
    compiled = lowered.quantum_system._description
    active = lowered.qudits
    fastest = _level_spread(compiled if target is None else target, active, levels)[0]
    if len(active) > 1:
        qudits = compiled.qudits
        for c in compiled.couplings:
            a, b = c.qudits
            if a in active and b in active:
                detuning = abs(qudits[a]._frame_hz - qudits[b]._frame_hz)
                if detuning > fastest:
                    fastest = detuning
    step = _DEFAULT_STEP_NS
    if fastest > 0:
        step = min(step, 1e9 / (_STEPS_PER_PERIOD * fastest))
    # A whole number of steps per clock cycle keeps pulse edges on the grid.
    cycle = 1.0 / lowered.quantum_system.clock_ghz
    return cycle / math.ceil(cycle / step - 1e-9)


def _level_spread(
    target: Any, qudits: tuple[int, ...], levels: int | dict[int, int] | None
) -> tuple[float, int, int]:
    """The widest spread, in Hz, of the level offsets (``_level_offsets_hz``)
    of one of *qudits*, with the qudit and its levels; zero with two levels
    each, whose offsets are both zero."""
    spread, qudit, count = 0.0, 0, 0
    model = target.model
    for q in qudits:
        if levels is None:
            d = model.levels_of(q)
        elif isinstance(levels, dict):
            d = levels[q] if q in levels else model.levels_of(q)
        else:
            d = int(levels)
        if d > 2:
            offsets = _level_offsets_hz(model, target.qudits[q], d)
            width = max(offsets) - min(offsets)
            if width > spread:
                spread, qudit, count = width, q, d
    return spread, qudit, count


def _fastest(
    target: Any, active: tuple[int, ...], levels: int | dict[int, int] | None
) -> tuple[float, tuple[str, int, int]]:
    """The fastest frequency, in Hz, that a simulation of *active* on
    *target* holds, with its source: ``("levels", qudit, levels)``,
    ``("coupling", qudit, qudit)``, or ``("", 0, 0)`` if none turns.

    Each qudit turns in the frame of its 0-1 transition: a coherence of two
    of its levels turns at the difference of their energies' offsets from
    evenly spaced levels, which is zero with two levels, and an exchange
    coupling at the qudits' detuning."""
    fastest = 0.0
    source = ("", 0, 0)
    if levels.__class__ is not int or levels > 2:
        fastest, q, d = _level_spread(target, active, levels)
        source = ("levels", q, d)
    qudits = target.qudits
    for c in target.couplings:
        a, b = c.qudits
        if a in active and b in active:
            detuning = abs(qudits[a]._frame_hz - qudits[b]._frame_hz)
            if detuning > fastest:
                fastest, source = detuning, ("coupling", a, b)
    return fastest, source


# The largest frequency (Hz) times step (ns) that keeps 20 steps per period.
_MAX_HZ_NS = 1e9 / _STEPS_PER_PERIOD * (1 + 1e-9)


def _level_offsets_hz(model: Any, qudit: Any, levels: int) -> list[float]:
    """Each level's energy less that of evenly spaced levels at the 0-1
    frequency: the frequency it turns at in the qudit's frame."""
    f01 = qudit._frame_hz
    return [e - k * f01 for k, e in enumerate(model.energies_hz(qudit, levels))]


def _hz(frequency: float) -> str:
    if frequency >= 1e9:
        return f"{frequency / 1e9:.4g} GHz"
    return f"{frequency / 1e6:.4g} MHz"


def _cause(target: Any, source: tuple[str, int, int], fastest: float) -> str:
    """What turns at *fastest*, from ``_fastest``'s *source*."""
    kind, a, b = source
    if kind == "levels":
        offsets = _level_offsets_hz(target.model, target.qudits[a], b)
        low, high = sorted((offsets.index(min(offsets)), offsets.index(max(offsets))))
        return (
            f"qudit {a} has {b} levels, and in its frame the coherence of levels "
            f"{low} and {high} turns at {_hz(fastest)}"
        )
    return f"the exchange coupling of qudits {a} and {b} turns at their detuning, {_hz(fastest)}"


def _suggestion(fastest: float, clock_ghz: float) -> str:
    """The largest step, as printed, that keeps 20 steps per period of
    *fastest* and a whole number of steps per clock cycle, and why."""
    bound = 1e9 / (_STEPS_PER_PERIOD * fastest)
    cycle = 1.0 / clock_ghz
    per_cycle = math.ceil(cycle / bound - 1e-9)
    # The step as printed must not exceed the bound once rounded.
    while float(f"{cycle / per_cycle:.3g}") > bound * (1 + 1e-9):
        per_cycle += 1
    return (
        f"dt_ns={cycle / per_cycle:.3g} or less ({per_cycle} steps per {cycle:g} ns "
        "clock cycle keep pulse edges on the step grid)"
    )


def _package_dir() -> str:
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warn_coarse_step(
    target: Any, source: tuple[str, int, int], fastest: float, dt_ns: float, clock_ghz: float
) -> None:
    import warnings

    from .._errors import _CoarseStepWarning

    warnings.warn(
        f"simulate(dt_ns={dt_ns:g}): {_cause(target, source, fastest)}, so a step of "
        f"{dt_ns:g} ns takes {1e9 / (fastest * dt_ns):.3g} steps per period of it. A "
        f"fixed-step integrator damps and dephases what it resolves with fewer than "
        f"{_STEPS_PER_PERIOD} steps per period, so the result may be wrong. "
        f"Pass {_suggestion(fastest, clock_ghz)}.",
        _CoarseStepWarning,
        skip_file_prefixes=(_package_dir(),),
    )


class _DivergedError(Exception):
    """An integration step of *step_ns* let the state grow without bound."""

    def __init__(self, step_ns: float) -> None:
        super().__init__(step_ns)
        self.step_ns = step_ns


def _diverged_error(
    lowered: ScheduledKernel,
    target: Any,
    levels: int | dict[int, int] | None,
    dt_ns: float | None,
    step_ns: float,
) -> ExecutionError:
    """The error for a run whose state *step_ns* let grow without bound,
    with the note of the line that called ``simulate``."""
    import sys

    fastest, source = _fastest(target, lowered.qudits, levels)
    call = "simulate()" if dt_ns is None else f"simulate(dt_ns={dt_ns:g})"
    text = (
        f"{call}: the integrator diverged: with a step of {step_ns:.3g} ns its state "
        "grew without bound, so there is no result."
    )
    if fastest and fastest * step_ns > _MAX_HZ_NS:
        cause = _cause(target, source, fastest)
        text += (
            f" {cause[0].upper()}{cause[1:]}, and a step of {step_ns:.3g} ns takes "
            f"{1e9 / (fastest * step_ns):.3g} steps per period of it: a fixed-step "
            f"integrator needs {_STEPS_PER_PERIOD} to be accurate and diverges with "
            f"fewer than about 2. Pass {_suggestion(fastest, lowered.quantum_system.clock_ghz)}."
        )
    else:
        text += (
            " The step resolves every frequency of the qudits' levels and couplings, so "
            "a drive or decay rate is too fast for it: pass a smaller dt_ns, such as "
            f"dt_ns={step_ns / 10:.3g}."
        )
    error = ExecutionError(text)
    package = _package_dir()
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame = frame.f_back  # type: ignore[assignment]
    if frame is not None:
        error.add_note(
            f'simulate was called at File "{frame.f_code.co_filename}", line {frame.f_lineno}'
        )
    return error


def _simulate_compiled(
    compiled: Any,
    target: Any,
    *,
    levels: int | dict[int, int] | None,
    initial_state: npt.ArrayLike | None,
    shots_count: int | None,
    seed: int | None,
    backend: str,
    t_start: float,
    t_end: float | None,
    dt_ns: float,
    integrator: str,
    clock_ghz: float,
    dephasing: str = "t2_star",
    handlers: Mapping[str, Callable[..., Any]] | None = None,
    readout: str = "ideal",
    latency_branching: bool = False,
    sampled: bool = False,
) -> SimulationResult:
    # A cached specialization's module is shared by every call of it, from
    # any thread: an interval sets the module attributes it lowers with and
    # reads the lowering under the module's lock, in one step. The GPU path,
    # and a call whose values set wait lengths, bind the float arguments into
    # a private copy of the module.
    lock = compiled._lock
    arguments = list(compiled.arguments)
    for v in arguments:
        if v.__class__ is not float and v.__class__ is not int or v - v != 0.0:
            arguments = compiled._checked_arguments()
            break
    functions = tuple(getattr(compiled, "classical", ()))
    tree = bool(functions) or readout != "ideal"
    if functions:
        # Calls are bound one execution at a time, on the concrete module
        # with loops that hold calls unrolled.
        from .._programs import _schedule_args

        module = compiled._concrete().simulation_module(
            *_schedule_args(compiled.options, compiled.clocks)
        )
        arguments = []
        lock = threading.RLock()
    elif backend == "gpu" or compiled.waits is not None:
        module = compiled._concrete()
    elif compiled.carriers:
        from .._programs import _placement

        # With float values, the placement's module evaluated at them: no
        # copy of the module per value.
        module = _placement(compiled) if compiled.arguments else compiled._concrete()
    else:
        module = compiled._module
    if compiled.runtime_selects:
        # A path plays the values its measurements select: on a copy, each op
        # that reads a run-time value runs in both arms of an if on the
        # measurement, at the time the schedule gave it.
        module = module.resolve_selects()
        lock = threading.RLock()
    qudits = compiled.qudits
    if not qudits:
        raise InvalidValueError("the program uses no qudits; there is nothing to simulate")
    mode = {q: i for i, q in enumerate(qudits)}
    dims, level_fields = _level_attributes(target, qudits, levels)
    base = _target_attributes(
        target, qudits, integrator=integrator, dephasing=dephasing, levels=level_fields
    )

    path_end = t_end is None and bool(functions)
    if t_end is None:
        # The latest end of the events that have a start and a length. Past
        # _LISTED_EVENTS of them a loop kept as a loop comes as a loop, whose
        # latest end is that of its last iteration: its events are not
        # listed once per iteration.
        events = module.event_tree(_LISTED_EVENTS)
        if events.__class__ is list:
            ends = [
                start + length
                for _, _, _, start, length in events
                if start is not None and length is not None
            ]
            latest = max(ends, default=0)
        else:
            from .._loops import Schedule

            end = Schedule(events[0], ScheduledEvent).latest_end(timed_only=True)
            assert end is not None
            latest = end
        t_end = latest / clock_ghz
    if t_end < t_start:
        raise InvalidValueError(f"t_end ({t_end}) must not precede t_start ({t_start})")

    dim = math.prod(dims)
    state = _initial_state(initial_state, dim)
    if initial_state is None:
        state = _thermal_state(target, qudits, dims, state)
    engine: _Backend
    if backend == "cpu":
        engine = _CpuBackend(module, dims, dt_ns, base, lock, arguments)
    else:
        # A program with classical calls lowers only with their answers
        # bound: the backend lowers its placeholders, as each interval does.
        probe = module.placeholders() if functions else module
        engine = _GpuBackend(probe, dims, dt_ns, base, lock, integrator)
    state = engine.prepare(state)

    if tree:
        return _simulate_tree(
            module,
            engine,
            dims,
            state,
            t_start,
            None if path_end else t_end,
            shots_count,
            np.random.default_rng(seed),
            functions,
            handlers,
            compiled,
            target,
            readout,
            latency_branching,
            clock_ghz,
            mode,
            sampled,
        )
    # Only shots draw: numpy.random, and the hashing it imports, load for them.
    rng: np.random.Generator = None if shots_count is None else np.random.default_rng(seed)  # type: ignore[assignment]
    seen: dict[tuple[int, tuple[int, ...]], Measurement] = {}
    leaves: list[tuple[Any, list[tuple[tuple[int, tuple[int, ...]], int]], int]] = []
    found = module.next_measurement([])
    if found is None or found[4] > t_end:
        if t_end > t_start:
            state = engine.evolve(state, [], t_start, t_end)
        leaves.append((state, [], shots_count or 0))
    else:
        _measurement_tree(
            module,
            engine,
            dims,
            mode,
            state,
            (t_start, t_end),
            shots_count or 0,
            rng,
            sampled,
            leaves,
            seen,
        )

    # The kernel's numbers are looked up only where something was measured.
    measurements, column = _columns(_kernel_numbered(module, seen)) if seen else ([], {})
    rho = np.zeros((dim, dim), dtype=np.complex128)
    branches = []
    rows = []
    for leaf, path, count in leaves:
        density = engine.density(leaf)
        if sampled:
            density = density * (count / (shots_count or 1) / float(np.real(np.trace(density))))
        rho += density
        outcomes = [-1] * len(measurements)
        for key, k in path:
            outcomes[column[key]] = k
        branches.append(Branch(tuple(outcomes), float(np.real(np.trace(density)))))
        rows.extend([outcomes] * count)
    records = None
    if shots_count is not None:
        records = np.array(rows, dtype=np.int64).reshape(len(rows), len(measurements))
        rng.shuffle(records)
    return SimulationResult(
        density_matrix=rho,
        dims=dims,
        qudits=qudits,
        measurements=measurements,
        branches=branches,
        records=records,
        times_ns=(t_start, t_end),
        _states=[leaf for leaf, _, _ in leaves],
    )


def _kernel_numbered(
    module: Any, seen: dict[tuple[int, tuple[int, ...]], Measurement]
) -> dict[tuple[int, tuple[int, ...]], Measurement]:
    """*seen*, by the module's readout index and loop iteration, each
    measurement given the number and iteration the kernel gave its readout:
    peeling and unrolling a loop copy its readouts, and every schedule
    reports them as the kernel made them."""
    keys = list(seen)
    numbered = module.readout_keys([[index, *iteration] for index, iteration in keys])
    return {
        key: replace(seen[key], readout=number, iteration=tuple(iteration))
        for key, (number, iteration) in zip(keys, numbered, strict=True)
    }


def _columns(
    seen: dict[tuple[int, tuple[int, ...]], Measurement],
) -> tuple[list[Measurement], dict[tuple[int, tuple[int, ...]], int]]:
    """The measurements of *seen* in the order they end, and the column of
    each path key (readout index and iteration in the module) among them.

    Copies of one execution of a kernel's readout (an op that reads a
    measurement runs in both arms of an if on it) share one column: a shot
    runs one of them, and the column holds its level. The measurement
    listed is the copy that ends first."""
    keys = sorted(seen, key=lambda k: (seen[k].end_ns, seen[k].readout, seen[k].iteration))
    measurements: list[Measurement] = []
    of: dict[tuple[int, tuple[int, ...]], int] = {}
    column = {}
    for key in keys:
        m = seen[key]
        column[key] = of.setdefault((m.readout, m.iteration), len(measurements))
        if column[key] == len(measurements):
            measurements.append(m)
    return measurements, column


def _measurement_tree(
    module: Any,
    engine: _Backend,
    dims: tuple[int, ...],
    mode: Mapping[int, int],
    state: Any,
    window: tuple[float, float],
    shots: int,
    rng: np.random.Generator,
    sampled: bool,
    leaves: list[tuple[Any, list[tuple[tuple[int, tuple[int, ...]], int]], int]],
    seen: dict[tuple[int, tuple[int, ...]], Measurement],
) -> None:
    """Walk the outcome paths of a program without classical calls from
    *state* over *window*, appending each path's end (state, outcomes, shots)
    to *leaves* in depth-first order and each measurement to *seen*.

    Shots are drawn at each measurement in that order, as each path is
    reached. Paths that reach the same window together evolve together
    (``_Backend.evolve_together``): siblings, and without *sampled* every
    path of a depth."""
    t_start, t_end = window

    # A path is [outcomes, state, time, shots]; one that measures gets its
    # measurement, outcome weights and followed outcomes appended.
    def advance(paths: list[list[Any]]) -> list[Any]:
        """Evolve *paths* to their next measurements and measure them; paths
        over the same window evolve together. Returns each one's
        projections, or None where it ends."""
        found = []
        windows: dict[tuple[float, float], list[int]] = {}
        for i, path in enumerate(paths):
            entries = [[index, *iteration, k] for (index, iteration), k in path[0]]
            next_one = module.next_measurement(entries)
            boundary = t_end if next_one is None or next_one[4] > t_end else next_one[4]
            found.append(next_one)
            if boundary > path[2]:
                windows.setdefault((path[2], boundary), []).append(i)
                path.append(entries)
        for (t, boundary), members in windows.items():
            evolved = engine.evolve_together(
                [paths[i][1] for i in members], [paths[i].pop() for i in members], t, boundary
            )
            for i, state in zip(members, evolved, strict=True):
                paths[i][1] = state
        projections: list[Any] = []
        for path, next_one in zip(paths, found, strict=True):
            if next_one is None or next_one[4] > t_end:
                projections.append(None)
                continue
            index, iteration, qudit, start, end = next_one
            path.append(Measurement(index, tuple(iteration), qudit, start, end))
            state = path[1]
            path[1] = None
            projected = [
                engine.project(state, dims, mode[qudit], k) for k in range(dims[mode[qudit]])
            ]
            # Exact propagation leaves an empty level at a trace of +-1e-17.
            projections.append((projected, np.maximum([engine.weight(s) for s in projected], 0.0)))
        return projections

    def outcome_counts(count: int, weights: npt.NDArray[np.float64]) -> Any:
        total = weights.sum()
        return (
            rng.multinomial(count, weights / total)
            if count and total > 0
            else np.zeros(len(weights), dtype=np.int64)
        )

    def follow(paths: list[list[Any]]) -> None:
        """Walk the paths shots take from *paths*, each to its end."""
        for path, projection in zip(paths, advance(paths), strict=True):
            if projection is None:
                leaves.append((path[1], path[0], path[3]))
                continue
            measurement = path[4]
            key = (measurement.readout, measurement.iteration)
            seen[key] = measurement
            projected, weights = projection
            counts = outcome_counts(path[3], weights)
            # A path of many measurements is improbable as a whole; each
            # followed outcome is kept at unit weight instead. A measurement
            # that ends before t_start projects the initial state; evolution
            # still starts at t_start.
            t = max(measurement.end_ns, path[2])
            total = weights.sum()
            follow(
                [
                    [[*path[0], (key, k)], branch / w, t, int(counts[k])]
                    for k, (branch, w) in enumerate(zip(projected, weights, strict=False))
                    if counts[k] and w > _NEGLIGIBLE * total
                ]
            )

    def draw(path: list[Any]) -> None:
        """Give the shots of *path* to its outcomes, as followed in order."""
        if len(path) == 4:
            leaves.append((path[1], path[0], path[3]))
            return
        measurement, weights, children = path[4:]
        seen[(measurement.readout, measurement.iteration)] = measurement
        counts = outcome_counts(path[3], weights)
        for k, child in children:
            child[3] = int(counts[k])
            draw(child)

    norm = engine.weight(state)
    root = [[], state, t_start, shots]
    if sampled:
        follow([root])
    else:
        # Every outcome above the floor is followed whatever the shots draw,
        # so the tree is grown a measurement deep at a time, every path at
        # that depth evolving at once, and the shots drawn after.
        frontier = [root]
        while frontier:
            deeper: list[list[Any]] = []
            for path, projection in zip(frontier, advance(frontier), strict=True):
                if projection is None:
                    continue
                projected, weights = projection
                t = max(path[4].end_ns, path[2])
                key = (path[4].readout, path[4].iteration)
                children = [
                    (k, [[*path[0], (key, k)], branch, t, 0])
                    for k, (branch, w) in enumerate(zip(projected, weights, strict=False))
                    if w > _NEGLIGIBLE * norm
                ]
                path += [weights, children]
                deeper.extend(child for _, child in children)
            frontier = deeper
        draw(root)


def _simulate_tree(
    module: Any,
    engine: _Backend,
    dims: tuple[int, ...],
    state: Any,
    t_start: float,
    t_end: float | None,
    shots_count: int | None,
    rng: np.random.Generator,
    functions: tuple[Any, ...],
    handlers: Mapping[str, Callable[..., Any]] | None,
    compiled: Any,
    target: Any,
    readout: str,
    latency_branching: bool,
    clock_ghz: float,
    mode: Mapping[int, int],
    sampled: bool = False,
) -> SimulationResult:
    """Simulate a program with classical calls or a readout model.

    Without *t_end* each path runs to the end of its own schedule, which
    variable waits make path-dependent."""
    from .._programs import _schedule_args
    from ..classical import _handlers_for
    from ._calls import IQModel, ReadoutModel, simulate_tree, takes_iq

    chosen = _handlers_for(functions, handlers)
    model = (
        None
        if readout == "ideal"
        else ReadoutModel(target, compiled.qudits, dims, decay=readout == "model_with_decay")
    )
    leaves, seen_raw = simulate_tree(
        module,
        engine,
        dims,
        state,
        t_start,
        t_end,
        shots_count,
        rng,
        functions,
        chosen,
        _schedule_args(compiled.options, compiled.clocks),
        model,
        latency_branching,
        clock_ghz,
        mode,
        (
            IQModel(target, compiled.qudits, decay=readout == "model_with_decay")
            if any(takes_iq(f) for f in functions)
            else None
        ),
        sampled,
    )
    seen = {key: Measurement(*value) for key, value in seen_raw.items()}
    if t_end is None:
        t_end = max((leaf.end for leaf in leaves), default=t_start)
    measurements, column = _columns(seen)
    dim = math.prod(dims)
    rho = np.zeros((dim, dim), dtype=np.complex128)
    branches = []
    rows = []
    for leaf in leaves:
        density = engine.density(leaf.state)
        if sampled:
            density = density * (
                leaf.count / (shots_count or 1) / float(np.real(np.trace(density)))
            )
        else:
            density = leaf.weight * density
        rho += density
        outcomes = [-1] * len(measurements)
        for key, k in leaf.path:
            outcomes[column[key]] = k
        branches.append(
            Branch(tuple(outcomes), float(np.real(np.trace(density))), tuple(leaf.calls))
        )
        rows.extend([outcomes] * leaf.count)
    records = None
    if shots_count is not None:
        records = np.array(rows, dtype=np.int64).reshape(len(rows), len(measurements))
        rng.shuffle(records)
    return SimulationResult(
        density_matrix=rho,
        dims=dims,
        qudits=compiled.qudits,
        measurements=measurements,
        branches=branches,
        records=records,
        times_ns=(t_start, t_end),
        _states=[leaf.state for leaf in leaves],
    )


def _initial_state(state: npt.ArrayLike | None, dim: int) -> npt.NDArray[np.complex128]:
    if state is None:
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0
        return psi
    array = np.asarray(state, dtype=np.complex128)
    if array.shape not in ((dim,), (dim, dim)):
        raise InvalidValueError(f"initial_state must have shape ({dim},) or ({dim}, {dim})")
    return array


def _thermal_state(
    target: Any,
    qudits: tuple[int, ...],
    dims: tuple[int, ...],
    cold: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """The product of the thermal mixtures of *qudits* (*dims* levels each),
    or *cold* if none is hot."""
    populations = [float(target.qudits[q].thermal_population) for q in qudits]
    if not any(populations):
        return cold
    diagonal: npt.NDArray[Any] = np.ones(1)
    for i in reversed(range(len(dims))):
        p = populations[i]
        weights = (p / (1.0 - p)) ** np.arange(dims[i], dtype=float)
        diagonal = np.kron(diagonal, weights / weights.sum())
    return np.diag(diagonal).astype(np.complex128)


def _level_mask(dims: tuple[int, ...], qudit: int, level: int) -> npt.NDArray[np.bool_]:
    stride = math.prod(dims[:qudit])
    indices = np.arange(math.prod(dims))
    return np.asarray((indices // stride) % dims[qudit] == level)


def _interval_attributes(entries: list[list[int]], t0: float, t1: float, steps: int) -> str:
    path = ", ".join(f"array<i64: {', '.join(map(str, e))}>" for e in entries)
    return (
        f"{{pulse.path = [{path}], qop.t_start = {_f64(t0)}, qop.t_end = {_f64(t1)}, "
        f"qop.step_count = {steps} : i64}}"
    )


class _Backend:
    """Interval evolution and projection for one state representation.

    *base* holds the module attributes of the target and levels, in MLIR
    syntax without braces; every interval sets them, then its own. The text
    of *base* is the same in every interval: the module's context parses it
    once.
    """

    # Whether evolve_together runs states as one batch; if not, a walk
    # gains nothing by gathering siblings first.
    batches = False

    def __init__(
        self, module: Any, dims: tuple[int, ...], dt_ns: float, base: str, lock: Any
    ) -> None:
        self.module = module
        self.dims = dims
        self.dim = math.prod(dims)
        self.dt_ns = dt_ns
        self.base = "{" + base + "}"
        self.lock = lock

    def steps(self, t0: float, t1: float) -> int:
        return max(1, math.ceil((t1 - t0) / self.dt_ns - 1e-9))

    def prepare(self, state: npt.NDArray[np.complex128]) -> Any:
        return state

    def evolve(self, state: Any, entries: list[list[int]], t0: float, t1: float) -> Any:
        raise NotImplementedError

    def evolve_together(
        self, states: list[Any], paths: list[list[list[int]]], t0: float, t1: float
    ) -> list[Any]:
        """Evolve each state along its path (``evolve``) over the same window."""
        return [self.evolve(s, e, t0, t1) for s, e in zip(states, paths, strict=True)]

    @staticmethod
    def project(state: Any, dims: tuple[int, ...], qudit: int, level: int) -> Any:
        mask = _level_mask(dims, qudit, level)
        if state.ndim == 1:
            return state * mask
        return state * np.outer(mask, mask)

    @staticmethod
    def weight(state: Any) -> float:
        if state.ndim == 1:
            return float(np.vdot(state, state).real)
        return float(np.trace(state).real)

    @staticmethod
    def density(state: Any) -> npt.NDArray[np.complex128]:
        if state.ndim == 1:
            return np.asarray(np.outer(state, state.conj()))
        return np.asarray(state)


def _keep_map(key: Any, propagator: npt.NDArray[np.complex128], h: float) -> None:
    """Cache the RK4 map *propagator* of steps of *h*, unless it diverges.

    A map that evolves states has no entry above 1 (each is a matrix element
    of the image of a unit-trace-norm operator); one with a larger or
    non-finite entry amplifies what it maps. Checked once, as it is built:
    the cache keeps no map that diverges, and a hit is not checked again."""
    if not np.abs(propagator).max() <= 2.0:
        raise _DivergedError(h)
    _propagate.keep(key, propagator)


def _bounded(state: Any, start: Any) -> bool:
    """Whether every entry of *state*, evolved from *start*, is finite and
    at most twice what evolution keeps it to: *start*'s trace bounds each
    entry of a density matrix, and its norm each of a vector."""
    if start.ndim == 1:
        bound = math.sqrt(float(np.vdot(start, start).real))
    else:
        bound = abs(complex(np.trace(start)))
    return bool(np.abs(state).max() <= 2.0 * bound)


def _full_operator(factors: list[Any], dims: tuple[int, ...]) -> npt.NDArray[np.complex128]:
    """The product of single-mode factors on the whole register."""
    on_mode: dict[int, npt.NDArray[np.complex128]] = {}
    for mode, d, data in factors:
        values = np.asarray(data, dtype=float)
        matrix = (values[0::2] + 1j * values[1::2]).reshape((d, d), order="F")
        on_mode[mode] = matrix @ on_mode[mode] if mode in on_mode else matrix
    full: npt.NDArray[np.complex128] = np.ones((1, 1), dtype=np.complex128)
    for mode in reversed(range(len(dims))):
        full = np.asarray(
            _propagate.kron(full, on_mode.get(mode, np.eye(dims[mode]))), dtype=np.complex128
        )
    return full


def _step_times(t0: float, h: float, lo: int, hi: int) -> npt.NDArray[np.float64]:
    """The start, middle and end of steps lo .. hi - 1, flattened.

    Callbacks are right-continuous on half-open intervals: like the
    cuDensityMat integrator, a step's end is sampled just inside the step so
    that a pulse ending there is not lost."""
    starts = t0 + h * np.arange(lo, hi)
    ends = np.maximum(
        starts,
        starts
        + h
        - np.maximum(h * 1e-9, 8 * np.finfo(float).eps * np.maximum(1.0, np.abs(starts + h))),
    )
    return np.stack([starts, starts + 0.5 * h, ends], axis=1).ravel()


def _step_range(support: tuple[float, float], t0: float, h: float, steps: int) -> tuple[int, int]:
    """The steps, of an interval from *t0* in steps of *h*, that a drive term
    with this support can be nonzero on."""
    start, end = support
    # Clamped as floats first: a term that plays throughout has a huge or
    # infinite duration.
    lo = int(math.floor(min(max((start - t0) / h - 1, 0.0), float(steps))))
    hi = int(math.ceil(min(max((end - t0) / h + 1, 0.0), float(steps))))
    return lo, hi


def _active_ranges(
    driven: list[tuple[int, Any, Any]], t0: float, h: float, steps: int
) -> list[tuple[int, int]]:
    """The ranges of steps, merged and in order, on which some drive term
    can be nonzero; every coefficient is zero on the other steps."""
    ranges = []
    for _, params, _ in driven:
        lo, hi = _step_range(_propagate.support(params), t0, h, steps)
        if hi > lo:
            ranges.append((lo, hi))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# Past this many ranges of steps on which drive terms play, each range is
# stepped with its own terms alone (_Stepper.play_apart). Up to it, asking
# every term in every range costs no more than sorting the terms into
# their ranges does.
_APART_RANGES = 3

# A driven stretch of a register too large for kept maps builds the
# Hamiltonians of its steps (three per step) in parts; a part and the one
# built after it take at most about this many bytes.
_STRETCH_BYTES = 1 << 25

# The states of one batched GPU run take at most this many bytes.
_BATCH_BYTES = 1 << 28


class _Stepper:
    """Steps one interval's density matrix in order (see ``_propagate``):
    exactly where the Lindbladian is constant, else with RK4, reusing kept
    maps on small registers."""

    def __init__(
        self,
        dims: tuple[int, ...],
        static: npt.NDArray[np.complex128],
        driven: list[tuple[int, Any, Any]],
        jumps: list[npt.NDArray[np.complex128]],
        decay: npt.NDArray[np.complex128],
        h: float,
    ) -> None:
        self.dim = static.shape[0]
        self.static = static
        self.driven = driven
        # Terms with one operator (every pulse of a line's quadrature) add
        # into one coefficient: a train of n pulses costs n evaluations of
        # its own steps, not n over the whole train.
        rows: dict[bytes, int] = {}
        operators: list[npt.NDArray[np.complex128]] = []
        self.row_of: list[int] = []
        for _, _, op in driven:
            row = rows.setdefault(op.tobytes(), len(operators))
            if row == len(operators):
                operators.append(op)
            self.row_of.append(row)
        self.supports = [_propagate.support(params) for _, params, _ in driven]
        self.operators = (
            np.stack(operators)
            if operators
            else np.zeros((0, self.dim, self.dim), dtype=np.complex128)
        )
        self.jumps = jumps
        self.decay = decay
        self.h = h
        self.done = 0
        self.key = (
            dims,
            h,
            static.tobytes(),
            b"".join(c.tobytes() for c in jumps),
            self.operators.tobytes(),
        )
        self._dissipator: npt.NDArray[np.complex128] | None = None
        self._terms: npt.NDArray[np.complex128] | None = None
        # An exact propagator costs about (12 + log2 of its norm) products of
        # (dim^2)^2 matrices; an RK4 step about 4 (4 + 2 jumps) products of
        # dim^2 matrices and its Python overhead.
        self._rk4_cost = 4 * (4 + 2 * len(jumps)) * self.dim**3 + 5e4
        self._exact_base = float(self.dim) ** 6

    def _exact_pays(self, n: int, hamiltonian: npt.NDArray[np.complex128]) -> bool:
        if self.dim <= 8:
            return True
        norm = 2 * float(np.abs(hamiltonian).sum(axis=1).max()) + float(
            np.abs(self.decay).sum(axis=1).max()
        )
        squarings = max(0.0, math.log2(max(norm * n * self.h, 1.0)))
        return bool((12 + squarings) * self._exact_base < n * self._rk4_cost)

    def idle(self, rho: Any, n: int) -> Any:
        return self.constant(rho, n, np.zeros(len(self.operators))) if n > 0 else rho

    def constant(self, rho: Any, n: int, values: npt.NDArray[np.float64]) -> Any:
        """*n* steps with every drive coefficient at *values*."""
        hamiltonian = self.static + np.tensordot(values, self.operators, axes=1)
        self.done += n
        if not self._exact_pays(n, hamiltonian):
            hs = np.broadcast_to(hamiltonian, (n * 3, self.dim, self.dim))
            return self._rk4(rho, (hs,))
        key = (self.key, "exact", values.tobytes(), n)
        propagator = _propagate.cached(key)
        if propagator is None:
            if self._dissipator is None:
                self._dissipator = _propagate.dissipator(self.jumps, self.dim, self.key[3])
            generator = _propagate.liouvillian(hamiltonian, self._dissipator)
            propagator = _propagate.expm(generator * (n * self.h))
            _propagate.keep(key, propagator)
        return _propagate.apply(propagator, rho)

    def play(self, rho: Any, t0: float, lo: int, hi: int) -> Any:
        """Steps lo .. hi - 1 (``done`` is lo), where drive terms play."""
        values = np.zeros((len(self.operators), hi - lo, 3))
        for (kind, params, _), row, support in zip(
            self.driven, self.row_of, self.supports, strict=False
        ):
            a, b = _step_range(support, t0, self.h, hi)
            a = max(a, lo)
            if b <= a:
                continue
            times = _step_times(t0, self.h, a, b).tolist()
            values[row, a - lo : b - lo, :] += np.asarray(
                evaluate_callback(kind, params, times), dtype=float
            ).reshape(b - a, 3)
        flat = np.all(values[:, :, 0] == values[:, :, 1], axis=0) & np.all(
            values[:, :, 1] == values[:, :, 2], axis=0
        )
        same = np.zeros(hi - lo, dtype=bool)
        same[1:] = flat[1:] & flat[:-1] & np.all(values[:, 1:, 0] == values[:, :-1, 0], axis=0)
        j = 0
        pending = 0
        count = hi - lo
        while j < count:
            k = j + 1
            if flat[j]:
                while k < count and same[k]:
                    k += 1
            if flat[j] and k - j >= 2:
                if pending < j:
                    rho = self._stretch(rho, values[:, pending:j, :])
                rho = self.constant(rho, k - j, np.ascontiguousarray(values[:, j, 0]))
                pending = k
            j = k
        if pending < count:
            rho = self._stretch(rho, values[:, pending:, :])
        return rho

    def ranges(self, t0: float, steps: int) -> list[tuple[int, int]]:
        """``_active_ranges`` of this stepper's drive terms over *steps*
        steps from *t0*, from the supports it keeps."""
        ranges = []
        for support in self.supports:
            lo, hi = _step_range(support, t0, self.h, steps)
            if hi > lo:
                ranges.append((lo, hi))
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for lo, hi in ranges:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def play_apart(self, rho: Any, t0: float, steps: int, ranges: list[tuple[int, int]]) -> Any:
        """All *steps* steps from *t0*: the idle stretches and each of
        *ranges* (``ranges(t0, steps)``) in turn, as ``idle`` and ``play``
        step them, but each range asking only its own terms.

        A term's steps lie in one range, the last that starts at or before
        them, and ``play`` adds nothing for a term outside the range it
        steps. So a copy of this stepper that holds a range's terms alone,
        in the order they have here, plays the range with the same
        coefficients, added in the same order: the result is the same to
        the last bit, and n terms in n ranges cost n evaluations where
        asking every term in every range costs n times n tests."""
        import copy
        from bisect import bisect_right

        starts = [lo for lo, _ in ranges]
        members: list[list[tuple[Any, int, tuple[float, float]]]] = [[] for _ in ranges]
        for term in zip(self.driven, self.row_of, self.supports, strict=True):
            lo, hi = _step_range(term[2], t0, self.h, steps)
            if hi > lo:
                members[bisect_right(starts, lo) - 1].append(term)
        part = copy.copy(self)
        for (lo, hi), terms in zip(ranges, members, strict=True):
            part.driven, part.row_of, part.supports = (
                list(column) for column in zip(*terms, strict=True)
            )
            rho = part.idle(rho, lo - part.done)
            rho = part.play(rho, t0, lo, hi)
        rho = part.idle(rho, steps - part.done)
        self.done = part.done
        return rho

    def _stretch(self, rho: Any, values: npt.NDArray[np.float64]) -> Any:
        """RK4 over steps whose coefficients (terms x steps x 3) vary."""
        n = values.shape[1]
        self.done += n
        if self.dim > _propagate.MAX_MAPPED_DIM:
            flat = values.reshape(len(self.operators), n * 3).T
            rows = 3 * max(1, _STRETCH_BYTES // (96 * self.dim * self.dim))
            return self._rk4(
                rho, (self._hamiltonians(flat[lo : lo + rows]) for lo in range(0, 3 * n, rows))
            )
        # A digest, not the coefficients: a long pulse train's run to
        # megabytes, and the cache keeps thousands of keys.
        digest = _blake2b(np.round(values, 12).tobytes(), digest_size=16).digest()
        key = (self.key, "rk4", digest)
        propagator = _propagate.cached(key)
        if propagator is None:
            if self._dissipator is None:
                self._dissipator = _propagate.dissipator(self.jumps, self.dim, self.key[3])
            if self._terms is None:
                self._terms = _propagate.commutators(self.operators)
            propagator = _propagate.rk4_map(
                _propagate.liouvillian(self.static, self._dissipator), self._terms, values, self.h
            )
            _keep_map(key, propagator, self.h)
        return _propagate.apply(propagator, rho)

    def _hamiltonians(self, coefficients: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """The Hamiltonian at each row of drive coefficients (rows x terms)."""
        hs: npt.NDArray[np.complex128] = np.tensordot(coefficients, self.operators, axes=1)
        hs += self.static
        return hs

    def _rk4(self, rho: Any, parts: Iterable[npt.NDArray[np.complex128]]) -> Any:
        """RK4 steps on *rho*, each under three Hamiltonians (its start,
        middle and end), from *parts* in order."""
        h = self.h
        jumps, decay = self.jumps, self.decay
        adjoints = [c.conj().T for c in jumps]

        def lindblad(r: Any, hm: Any) -> Any:
            out = -1j * (hm @ r - r @ hm)
            for c, ca in zip(jumps, adjoints, strict=False):
                out += c @ r @ ca
            if jumps:
                out -= 0.5 * (decay @ r + r @ decay)
            return out

        start = rho
        # A state that diverges raises _DivergedError below, not numpy's warnings.
        with np.errstate(over="ignore", invalid="ignore"):
            for hs in parts:
                for step in range(len(hs) // 3):
                    h0, hm, h1 = hs[3 * step], hs[3 * step + 1], hs[3 * step + 2]
                    k1 = lindblad(rho, h0)
                    k2 = lindblad(rho + 0.5 * h * k1, hm)
                    k3 = lindblad(rho + 0.5 * h * k2, hm)
                    k4 = lindblad(rho + h * k3, h1)
                    rho = rho + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not _bounded(rho, start):
            raise _DivergedError(h)
        return rho


class _CpuBackend(_Backend):
    """Fixed-step RK4 on the density matrix: the reference for the GPU path.

    The lowering takes the float arguments as values, so a call with new
    values lowers no differently from the last one.
    """

    def __init__(
        self,
        module: Any,
        dims: tuple[int, ...],
        dt_ns: float,
        base: str,
        lock: Any,
        arguments: list[float],
    ) -> None:
        super().__init__(module, dims, dt_ns, base, lock)
        self.arguments = arguments

    def prepare(self, state: npt.NDArray[np.complex128]) -> Any:
        return self.density(state)

    def evolve(self, state: Any, entries: list[list[int]], t0: float, t1: float) -> Any:
        steps = self.steps(t0, t1)
        with self.lock:
            self.module.set_attributes(self.base, _interval_attributes(entries, t0, t1, steps))
            _, hamiltonian, collapse, *_ = self.module.operator_program(self.arguments)
        static = np.zeros((self.dim, self.dim), dtype=np.complex128)
        driven = []
        for re, im, kind, params, factors in hamiltonian:
            if kind:
                # An unrolled program lists every pulse it plays; only those
                # that overlap the interval act in it.
                start, end = _propagate.support(params)
                if end <= t0 or start >= t1:
                    continue
                driven.append((kind, params, _full_operator(factors, self.dims)))
            else:
                static += (re + 1j * im) * _full_operator(factors, self.dims)
        jumps = [(re + 1j * im) * _full_operator(f, self.dims) for re, im, _, _, f in collapse]
        decay = sum((c.conj().T @ c for c in jumps), np.zeros_like(static))
        if t1 <= t0:
            return state
        h = (t1 - t0) / steps
        stepper = _Stepper(self.dims, static, driven, jumps, decay, h)
        if self.dims[1:] and any(k == 1 and p[1] >= 1e300 for k, p, _ in driven):
            # Couplings of detuned qudits play throughout: idle stretches
            # are propagated in the frame that turns with them.
            from ._idle import step_through

            rho = step_through(stepper, self.dims, state, t0, steps, self.dt_ns)
            if rho is not None:
                return rho
        if driven:
            ranges = stepper.ranges(t0, steps)
            if len(ranges) > _APART_RANGES:
                # Drives apart from each other: each stretch of them asks
                # only its own terms.
                return stepper.play_apart(state, t0, steps, ranges)
            for lo, hi in ranges:
                state = stepper.idle(state, lo - stepper.done)
                state = stepper.play(state, t0, lo, hi)
        return stepper.idle(state, steps - stepper.done)


class _GpuBackend(_Backend):
    """Each interval is lowered and JIT-run by the cuDensityMat runtime."""

    batches = True

    def __init__(
        self,
        module: Any,
        dims: tuple[int, ...],
        dt_ns: float,
        base: str,
        lock: Any,
        integrator: str,
    ) -> None:
        super().__init__(module, dims, dt_ns, base, lock)
        from ._jit import JITCompiler

        self.compiler = JITCompiler()
        self.integrator = integrator
        # Collapse operators depend on the target alone. A program that
        # measures lowers only along a measurement path over a window, so ask
        # along the empty path over an empty window at 0 ns, which ends before
        # any measurement; every interval then sets its own path and window.
        with lock:
            module.set_attributes(self.base, _interval_attributes([], 0.0, 0.0, 1))
            _, _, collapse, *_ = module.operator_program()
        self.dissipative = bool(collapse)

    def prepare(self, state: npt.NDArray[np.complex128]) -> Any:
        if self.dissipative and state.ndim == 1:
            return self.density(state)
        return state

    def evolve(self, state: Any, entries: list[list[int]], t0: float, t1: float) -> Any:
        steps = self.steps(t0, t1)
        # An idle stretch long enough to pay for lowering and running the
        # rest in parts (_idle.GPU_IDLE_STEPS) is propagated on the host.
        if steps >= 1024:
            from ._idle import through_gpu

            results = through_gpu(self, [state], entries, t0, t1, steps)
            if results is not None:
                return results[0]
        mixed = state.ndim == 2
        attributes = _interval_attributes(entries, t0, t1, steps)
        attributes = attributes[:-1]
        if mixed and not self.dissipative:
            attributes += ", qop.mixed_state"
        with self.lock:
            self.module.set_attributes(self.base, attributes + "}")
            text = self.module.print()
        (result,) = self._execute(text, [state], mixed)
        if not _bounded(result, state):
            raise _DivergedError((t1 - t0) / steps)
        return result

    def evolve_together(
        self, states: list[Any], paths: list[list[list[int]]], t0: float, t1: float
    ) -> list[Any]:
        """Evolve each state along its path over the same window. States
        whose paths lower to the same operators there (sibling branches that
        no outcome between them changes) evolve as one batched run, member
        for member what each would give alone."""
        if len(states) == 1:
            return [self.evolve(states[0], paths[0], t0, t1)]
        steps = self.steps(t0, t1)
        programs = []
        with self.lock:
            for entries in paths:
                self.module.set_attributes(self.base, _interval_attributes(entries, t0, t1, steps))
                programs.append(self.module.operator_program())
        groups: list[list[int]] = []
        for i, program in enumerate(programs):
            for group in groups:
                first = group[0]
                if states[first].ndim == states[i].ndim and programs[first] == program:
                    group.append(i)
                    break
            else:
                groups.append([i])
        results: list[Any] = [None] * len(states)
        limit = max(1, _BATCH_BYTES // states[0].nbytes)
        for group in groups:
            for lo in range(0, len(group), limit):
                members = group[lo : lo + limit]
                if len(members) > 1 and steps >= 1024:
                    from ._idle import through_gpu

                    # As evolve: idle stretches on the host, member by
                    # member, and the rest of the interval as one batch.
                    split = through_gpu(
                        self, [states[i] for i in members], paths[members[0]], t0, t1, steps
                    )
                    if split is not None:
                        for i, result in zip(members, split, strict=True):
                            results[i] = result
                        continue
                if len(members) == 1:
                    results[members[0]] = self.evolve(states[members[0]], paths[members[0]], t0, t1)
                    continue
                mixed = states[members[0]].ndim == 2
                attributes = _interval_attributes(paths[members[0]], t0, t1, steps)[:-1]
                if mixed and not self.dissipative:
                    attributes += ", qop.mixed_state"
                with self.lock:
                    self.module.set_attributes(self.base, attributes + "}")
                    text = self.module.print()
                evolved = self._execute(text, [states[i] for i in members], mixed)
                for i, result in zip(members, evolved, strict=True):
                    if not _bounded(result, states[i]):
                        raise _DivergedError((t1 - t0) / steps)
                    results[i] = result
        return results

    def _execute(self, text: str, states: list[Any], mixed: bool) -> list[Any]:
        """Run one interval, the module *text*, on the GPU from each of
        *states*, as one batch."""
        jit_module = self.compiler.compile_pulse_mlir(text)
        # cuDensityMat stores a density matrix with the ket index fastest,
        # and a batch member after member.
        buffer = np.ascontiguousarray(
            np.concatenate([s.ravel(order="F") for s in states])
            if len(states) > 1
            else states[0].ravel(order="F")
            if mixed
            else states[0]
        )
        library = self.compiler._runtime.library
        if library.cudm_set_initial_state(buffer.ctypes.data, buffer.nbytes, len(states)) != 0:
            raise ExecutionError("could not set the cuDensityMat initial state")
        results = self.compiler.load_and_run(
            jit_module, qudit_count=len(self.dims), state_dim=self.dim, count=len(states)
        )
        if mixed:
            return [r.to_numpy().ravel().reshape((self.dim, self.dim), order="F") for r in results]
        return [r.to_numpy().ravel() for r in results]
