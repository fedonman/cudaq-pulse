# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transmon quantum systems.

``transmon_surface17`` is the 17-qubit distance-three surface-code quantum system of
Krinner et al., Nature 605, 669 (2022), arXiv:2112.03708, built from the
values the paper publishes (``_transmon_surface17``, with a citation for each).
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

from .._errors import InvalidValueError
from . import _transmon_surface17 as _k
from .base import (
    _DRIVE_GATES,
    Coupling,
    DispersiveReadout,
    DragPulse,
    FluxCZ,
    MicrowaveDrive,
    PhysicalQudit,
    ReadoutPulse,
    _Calibrations,
    _Description,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .._device import QuantumSystem
    from .calibration import Calibration

#: The CZs of one error correction cycle of ``transmon_surface17``, layer by layer
#: (Fig. 3a and the edge labels of Fig. 6c): each layer is three CZs on
#: disjoint pairs, each pair the (auxiliary, data) qudit indices of a
#: coupling. Layers 1-4 measure the Z stabilizers, 5-8 the X stabilizers.
CZ_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((13, 0), (14, 4), (15, 2)),  # Z1-D1 Z2-D5 Z3-D3
    ((13, 3), (14, 7), (15, 4)),  # Z1-D4 Z2-D8 Z3-D5
    ((14, 3), (15, 1), (16, 5)),  # Z2-D4 Z3-D2 Z4-D6
    ((14, 6), (15, 5), (16, 8)),  # Z2-D7 Z3-D6 Z4-D9
    ((10, 0), (11, 4), (12, 6)),  # X2-D1 X3-D5 X4-D7
    ((10, 1), (11, 5), (12, 7)),  # X2-D2 X3-D6 X4-D8
    ((9, 2), (10, 4), (11, 8)),  # X1-D3 X2-D5 X3-D9
    ((9, 1), (10, 3), (11, 7)),  # X1-D2 X2-D4 X3-D8
)


def transmon_surface17(calibration: Any = "transmon_surface17") -> QuantumSystem:
    """17-qubit surface-code quantum system of Krinner et al., Nature 605, 669 (2022).

    Qudits 0-16 are D1-D9, X1-X4, Z1-Z4, the column order of the paper's
    Tab. I; ``PhysicalQudit.label`` carries the name. The 24 couplings are the
    auxiliary-data pairs of the stabilizers of Tab. II::

        Z1: D1 D4         X1: D2 D3
        Z2: D4 D5 D7 D8   X2: D1 D2 D4 D5
        Z3: D2 D3 D5 D6   X3: D5 D6 D8 D9
        Z4: D6 D9         X4: D7 D8

    Data qubits idle at 3.74-4.14 GHz (their lower sweet spot) and
    auxiliary qubits at 5.89-6.33 GHz (their upper one), with idle
    frequencies, anharmonicities, T1 and Ramsey T2* from Tab. I.

    The paper's device-wide J / 2 pi ~ 7 MHz (Methods A) is the
    ``|11> <-> |20>`` coupling of the CZ at each pair's interaction point (7.4
    MHz measured on X2-D4 at 4.4 GHz, Hellings et al., arXiv:2503.04610).
    *calibration* (a ``Calibration``, the name of a shipped one, or None)
    supplies the system-wide charge coupling g_c (fitted to the X2-D4
    measurement, 3.12 MHz in ``transmon_surface17``), and each coupling is the
    exchange J_qq = g_c ``<0|n_a|1> <0|n_d|1>`` at the qubits' idle points:
    5.67-6.08 MHz with the shipped ``transmon_surface17``. With
    ``calibration=None`` it is the published value over sqrt(2), 4.95 MHz.
    Each coupling's ``CZ`` calibration is a ``FluxCZ``: 98 ns on average,
    with 15 ns buffers, at the pair's interaction frequency of Fig. 6c.
    ``CZ_LAYERS`` gives the time steps the CZs run in.
    Each qudit's ``X`` is the published 40 ns DRAG pulse with sigma = 10 ns;
    its amplitude is not published, so amplitudes are angular rates in
    rad/ns and ``get_calibration`` gives the pi pulse's. Readout physics is
    that of Tab. I, and the ``READOUT`` pulse that of App. F1. The shipped
    ``transmon_surface17`` calibration fits what the paper does not publish:
    each ``X``'s DRAG coefficient (0.51-0.58 ns), an ``X`` of each qudit's
    1-2 transition (``transition=(1, 2)``, 20 ns) and readout assignment
    errors with which ``readout="model"`` reproduces Tab. I's two-state
    readout errors. ``calibration=None`` leaves them out.

    No residual ZZ is listed: the paper gives no per-pair values, only
    that the calculated idle ZZ is below about 8 kHz. Simulated with three
    or more levels, the couplings produce it; ``transmon_surface17_idle_zz_hz`` gives
    the charge model's value, -4.5 to -7.3 kHz with ``transmon_surface17``.
    """
    if calibration is None:
        from .._device import _calibrated

        return _calibrated(
            _published_surface17(), None, "transmon_surface17(calibration=None)", None, None
        )
    from .._device import quantum_system

    if calibration == "transmon_surface17":
        return quantum_system("transmon_surface17")
    return quantum_system("transmon_surface17", calibration=calibration)


# The description ``transmon_surface17(calibration)`` is built from.
def _described_surface17(calibration: Any = "transmon_surface17") -> _Description:
    if isinstance(calibration, str):
        return _calibrated_surface17(calibration)
    target = _published_surface17()
    return target if calibration is None else calibration._apply(target)


_BUILT: dict[str | None, _Description] = {}
_BUILT_LOCK = threading.Lock()


def _calibrated_surface17(name: str) -> _Description:
    from .calibration import shipped_calibration

    built = _BUILT.get(name)
    if built is None:
        with _BUILT_LOCK:
            built = _BUILT.get(name)
            if built is None:
                built = _BUILT[name] = _build_surface17(shipped_calibration(name))
    return built


def _published_surface17() -> _Description:
    built = _BUILT.get(None)
    if built is None:
        with _BUILT_LOCK:
            built = _BUILT.get(None)
            if built is None:
                built = _BUILT[None] = _build_surface17(None)
    return built


# The published values with *calibration* applied, built once: what
# ``calibration._apply`` of the published description gives, without
# building the published qudits and couplings first.
def _build_surface17(calibration: Calibration | None) -> _Description:
    qudits = {}
    drive = MicrowaveDrive()
    x = DragPulse(duration_ns=_k.DRIVE_DURATION_NS, sigma_ns=_k.DRIVE_SIGMA_NS)
    # One calibration mapping per kind of qudit (data, X and Z auxiliaries),
    # built as checked: these values are known to be right.
    calibrations = {
        kind: _Calibrations(
            {
                "x": x,
                "readout": ReadoutPulse(
                    duration_ns=duration_ns,
                    sigma_ns=_k.READOUT_SIGMA_NS,
                    integration_ns=_k.READOUT_INTEGRATION_NS,
                ),
            }
        )
        for kind, duration_ns in _k.READOUT_DURATION_NS.items()
    }
    pulses: Mapping[str, Any] = {}
    made: dict[Any, Any] = {}
    if calibration is not None:
        pulses, idle_exchange = calibration._qudit_pulses(), calibration._idle_exchange
    for index, q in enumerate(_k.QUDITS):
        given = calibrations[q.label[0]]
        error_0to1 = error_1to0 = 0.0
        values = pulses.get(q.label)
        if values is not None:
            beta, pulse, errors = values
            if beta is not None or pulse is not None:
                given = _calibrated_pulses(given, beta, pulse, made)
            if errors is not None:
                error_0to1, error_1to0 = errors
        qudits[index] = PhysicalQudit(
            index=index,
            frequency_hz=q.idle_frequency_hz,
            anharmonicity_hz=q.anharmonicity_hz,
            t1_us=q.t1_us,
            t2_star_us=q.t2_ramsey_us,
            t2_echo_us=q.t2_echo_us,
            thermal_population=q.thermal_population,
            label=q.label,
            drive=drive,
            readout=DispersiveReadout(
                frequency_hz=q.readout_frequency_hz,
                dispersive_shift_hz=q.dispersive_shift_hz,
                linewidth_hz=q.kappa_eff_hz,
                error_1to0=error_1to0,
                error_0to1=error_0to1,
            ),
            calibrations=given,
        )
    couplings = []
    for e in _k.EDGES:
        a, d = _k.INDEX[e.auxiliary], _k.INDEX[e.data]
        strength = None
        if calibration is not None:
            at_a, at_d = qudits[a].frequency_hz, qudits[d].frequency_hz
            strength = calibration._exchange(idle_exchange, f"{e.auxiliary}-{e.data}", at_a, at_d)
            if strength is None:
                strength = calibration._exchange(
                    idle_exchange, f"{e.data}-{e.auxiliary}", at_d, at_a
                )
        if strength is None:
            strength = _k.coupling_11_20_hz(e.auxiliary, e.data) / math.sqrt(2.0)
        couplings.append(
            Coupling(
                (a, d),
                strength_hz=strength,
                calibrations=_Calibrations(
                    {
                        "cz": FluxCZ(
                            duration_ns=_k.CZ_DURATION_NS,
                            buffer_ns=_k.CZ_BUFFER_NS,
                            interaction_frequency_hz=e.interaction_frequency_hz,
                        )
                    }
                ),
            )
        )
    return _Description(
        name="transmon_surface17",
        qudits=qudits,
        couplings=couplings,
        attribution=_k.CITATION,
    )


# A qudit's calibrations *given* with the values of
# ``Calibration._qudit_pulses``: its drive-line pulses take the DRAG
# coefficient *beta* and the 1-2 ``X`` *pulse* is added. Qudits given
# equal pulses share them through *made*, as a pickle then stores them once.
def _calibrated_pulses(given: Any, beta: Any, pulse: Any, made: dict[Any, Any]) -> Any:
    key = (id(given), beta, pulse)
    calibrated = made.get(key)
    if calibrated is None:
        values = dict(given)
        if beta is not None:
            for gate, found in given.items():
                if found.__class__ is DragPulse and gate in _DRIVE_GATES:
                    duration, sigma, amplitude, _, phase = found._values
                    found = DragPulse(
                        duration_ns=duration,
                        sigma_ns=sigma,
                        amplitude=amplitude,
                        beta_ns=beta * 1e9,
                        phase_rad=phase,
                    )
                    values[gate] = made.setdefault(found._values, found)
        if pulse is not None:
            duration, sigma, amplitude, beta_12 = pulse
            found = DragPulse(
                duration_ns=duration * 1e9,
                sigma_ns=sigma * 1e9,
                amplitude=amplitude * 1e-9,
                beta_ns=beta_12 * 1e9,
            )
            values["x", (1, 2)] = made.setdefault(found._values, found)
        calibrated = made[key] = _Calibrations(values)
    return calibrated


def transmon_surface17_idle_zz_hz(
    calibration: Any = "transmon_surface17",
) -> dict[tuple[int, int], float]:
    """Idle ZZ of each coupled pair of the ``transmon_surface17`` quantum system, keyed by the
    coupling's (auxiliary, data) qudit indices.

    zeta = E_11 - E_10 - E_01 + E_00 of the dressed levels of the
    charge-coupled pair (charge basis, five levels each, no rotating-wave
    approximation) at the idle sweet spots, with the circuits and couplings
    of *calibration*: -4.5 to -7.3 kHz with the shipped ``transmon_surface17``, below
    the paper's "below about 8 kHz". The simulation's Duffing ladder with
    exchange J_qq reproduces it within a few percent.
    """
    from .calibration import shipped_calibration

    if isinstance(calibration, str):
        calibration = shipped_calibration(calibration)
    return {
        (_k.INDEX[e.auxiliary], _k.INDEX[e.data]): calibration.idle_zz_hz(f"{e.auxiliary}-{e.data}")
        for e in _k.EDGES
    }


def transmon_lattice(
    qudit_count: int,
    *,
    base_frequency_hz: float = 5.0e9,
    frequency_range_hz: float = 100e6,
    anharmonicity_hz: float = -330e6,
    coupling_strength_hz: float = 3.0e6,
    t1_us: float = 25.0,
    t2_star_us: float = 15.0,
    topology: str = "linear",
) -> QuantumSystem:
    """A transmon quantum system of *qudit_count* uniform qudits on a chain,
    ring or grid, named ``"transmon_lattice:N"`` (``"transmon_lattice:N:ring"``
    or ``":grid"`` for the other topologies), as the registry names it.

    Parameters
    ----------
    qudit_count : int
        Number of qudits.
    base_frequency_hz : float
        Frequency of qudit 0.
    frequency_range_hz : float
        Total frequency range, from qudit 0 to qudit N-1: qudit i is at
        ``base_frequency_hz + i * frequency_range_hz / (N - 1)``.
    anharmonicity_hz : float
        Uniform anharmonicity (negative for transmon).
    coupling_strength_hz : float
        Uniform nearest-neighbor coupling.
    t1_us, t2_star_us : float
        Uniform decoherence times.
    topology : str
        ``"linear"`` (chain), ``"ring"``, or ``"grid"`` (square lattice
        with side length = ceil(sqrt(qudit_count))).
    """
    from .._device import _calibrated

    built = _described_lattice(
        qudit_count,
        base_frequency_hz=base_frequency_hz,
        frequency_range_hz=frequency_range_hz,
        anharmonicity_hz=anharmonicity_hz,
        coupling_strength_hz=coupling_strength_hz,
        t1_us=t1_us,
        t2_star_us=t2_star_us,
        topology=topology,
    )
    return _calibrated(built, None, f"transmon_lattice({qudit_count})", None, None)


# The description ``transmon_lattice(...)`` is built from.
def _described_lattice(
    qudit_count: int,
    *,
    base_frequency_hz: float = 5.0e9,
    frequency_range_hz: float = 100e6,
    anharmonicity_hz: float = -330e6,
    coupling_strength_hz: float = 3.0e6,
    t1_us: float = 25.0,
    t2_star_us: float = 15.0,
    topology: str = "linear",
) -> _Description:
    if qudit_count < 1:
        raise InvalidValueError(f"qudit_count must be >= 1, got {qudit_count}")

    import math as _math

    qudits = {}
    for i in range(qudit_count):
        qudits[i] = PhysicalQudit(
            index=i,
            frequency_hz=base_frequency_hz + i * frequency_range_hz / max(qudit_count - 1, 1),
            anharmonicity_hz=anharmonicity_hz,
            t1_us=t1_us,
            t2_star_us=t2_star_us,
            label=f"Q{i}",
        )

    edges: list[tuple[int, int]] = []
    if topology == "linear":
        name = f"transmon_lattice:{qudit_count}"
        edges = [(i, i + 1) for i in range(qudit_count - 1)]
    elif topology == "ring":
        name = f"transmon_lattice:{qudit_count}:ring"
        edges = [(i, (i + 1) % qudit_count) for i in range(qudit_count)]
    elif topology == "grid":
        name = f"transmon_lattice:{qudit_count}:grid"
        side = _math.ceil(_math.sqrt(qudit_count))
        for i in range(qudit_count):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < qudit_count:
                edges.append((i, i + 1))
            if r + 1 < side and i + side < qudit_count:
                edges.append((i, i + side))
    else:
        raise InvalidValueError(f"Unknown topology {topology!r}. Use 'linear', 'ring', or 'grid'.")

    couplings = [Coupling((a, b), coupling_strength_hz) for a, b in edges]

    return _Description(
        name=name,
        qudits=qudits,
        couplings=couplings,
    )
