# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module attributes that carry a quantum system's calibration into the lowering."""

from __future__ import annotations

import math
import struct
from itertools import compress
from operator import attrgetter
from typing import Any

from .._errors import InvalidValueError
from ..quantum_systems.base import _unmeasured, layout_anharmonicities, layout_frequencies

_QUDITS = attrgetter("qudits")

_VALID_INTEGRATORS = frozenset(
    {
        "rk1",
        "rk2",
        "rk4",
        "magnus",
        "crank_nicolson",
    }
)


def _level_attributes(
    target: Any, qudits: tuple[int, ...], levels: int | dict[int, int] | None
) -> tuple[tuple[int, ...], str]:
    """Levels of each simulated qudit, in the order of *qudits*, and the
    ``#pulse.quantum_system`` fields that model them (indexed by quantum
    system qudit).

    A qudit has the levels its target's model declares for it
    (``model.levels_of``) unless *levels* sets them: an int for every
    qudit, a dict by quantum system qudit index for those it names."""
    model = target.model
    if levels is None:
        dims = tuple(model.levels_of(q) for q in qudits)
    elif isinstance(levels, dict):
        dims = tuple(levels[q] if q in levels else model.levels_of(q) for q in qudits)
    else:
        dims = (int(levels),) * len(qudits)
    if any(d < 2 for d in dims):
        raise InvalidValueError("every qudit needs at least two levels")
    # Qudits the program does not use are not simulated, and keep the levels
    # they would have.
    extent = _extent(qudits)
    if levels is None or isinstance(levels, dict):
        by_index = [model.levels_of(q) for q in range(extent)]
    else:
        by_index = [int(levels)] * extent
    for q, d in zip(qudits, dims, strict=False):
        by_index[q] = d
    fields = [f"level_counts = [{', '.join(map(str, by_index))}]"]
    if any(d > 2 for d in dims):
        alphas = [0.0] * _extent(qudits)
        for q, d in zip(qudits, dims, strict=False):
            qudit = target.qudits[q]
            if qudit.anharmonicity_hz is None:
                # The model refuses the levels if they need the anharmonicity.
                model.energies_hz(qudit, d)
                continue
            alphas[q] = float(qudit.anharmonicity_hz)
        if target.couplings:
            # A qudit simulated without a qudit its couplings reach keeps that
            # qudit's share of its 1-2 transition.
            for q, alpha in layout_anharmonicities(target, qudits, dims).items():
                alphas[q] = alpha
        fields.append(f"anharmonicity_hz = {_f64_list(alphas)}")
    return dims, ", ".join(fields)


def _extent(qudits: tuple[int, ...]) -> int:
    """The length of an attribute indexed by quantum system qudit."""
    return max(qudits) + 1


def _f64_literal(value: float) -> str:
    value = float(value)
    if math.isfinite(value):
        return format(value, ".17e")
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016X}"


def _f64(value: float) -> str:
    return f"{_f64_literal(value)} : f64"


def _f64_array(values: list[float]) -> str:
    return "array<f64: " + ", ".join(_f64_literal(v) for v in values) + ">"


def _f64_list(values: list[float]) -> str:
    return "[" + ", ".join(map(_f64_literal, values)) + "]"


def _target_attributes(
    target: Any,
    qudits: tuple[int, ...],
    *,
    integrator: str,
    dephasing: str = "t2_star",
    levels: str = "",
) -> str:
    """Module attributes, in MLIR syntax and comma-separated, that the
    cuDensityMat lowering reads, but for the time window, which each interval
    sets: ``pulse.quantum_system`` with the *levels* fields
    (``_level_attributes``), and beside it what it does not describe.

    *qudits* are the simulated qudits, one mode each in this order; the
    others are left out, along with their couplings. Each simulated qudit is
    at its frequency on *target*, which may differ from the frequency the
    program was compiled for, moved where couplings to left-out qudits dress
    it (``layout_frequencies``); the others' are 0 Hz. Qudits dephase at
    their Ramsey T2* (*dephasing*
    ``"t2_star"``) or their Hahn-echo T2 (``"t2_echo"``).
    """
    qudit_count = _extent(qudits)
    frequencies = ["0.0"] * qudit_count
    t1_ns = [0.0] * qudit_count
    t2_ns = [0.0] * qudit_count
    thermal = [0.0] * qudit_count
    drive_scales = [1.0] * qudit_count
    for qi in qudits:
        qudit = target.qudits[qi]
        t2_us = qudit.t2_star_us if dephasing == "t2_star" else qudit.t2_echo_us
        if t2_us is None:
            raise InvalidValueError(f"qudit {qi} has no Hahn-echo T2 to dephase at")
        # A frequency is finite, so it is spelled as _f64_literal would.
        frequencies[qi] = f"{qudit._frame_hz:.17e}"
        # _f64_literal makes each a float.
        t1_ns[qi] = max(0.0, qudit.t1_us) * 1.0e3
        t2_ns[qi] = max(0.0, t2_us) * 1.0e3
        thermal[qi] = qudit.thermal_population
        drive_scales[qi] = target.drive_amplitude_scale(qi)
    attrs = [
        f"qop.qudit_count = {len(qudits)} : i64",
        f"pulse.modes = array<i64: {', '.join(map(str, qudits))}>",
        f'qop.integrator = "{integrator}"',
    ]
    if any(thermal):
        attrs.append(f"pulse.thermal_populations = {_f64_array(thermal)}")
    active = set(qudits)
    # pulse.coupling_* is the exchange, the operator of the couplings of the
    # one model that simulates (quantum_systems.base._COUPLING_OPERATORS).
    couplings = target.couplings
    if couplings:
        # The drive tones are at the whole system's dressed frequencies: a
        # simulated qudit whose couplings to left-out qudits dress it is
        # given the frequency that the kept couplings dress to the same.
        for qi, hz in layout_frequencies(target, qudits).items():
            frequencies[qi] = f"{hz:.17e}"
        couplings = list(compress(couplings, map(active.issuperset, map(_QUDITS, couplings))))
    if couplings:
        strengths = [c.strength_hz for c in couplings]
        if None in strengths:
            raise _unmeasured(
                target,
                [
                    i
                    for i, c in enumerate(target.couplings)
                    if c.strength_hz is None and active.issuperset(c.qudits)
                ],
                "simulating needs the strength of every coupling of the program's qudits",
                "give the strength in Hz, or leave those qudits out of the program",
            )
        pairs = ", ".join(f"{c.qudits[0]}, {c.qudits[1]}" for c in couplings)
        attrs.append(f"pulse.coupling_pairs = array<i64: {pairs}>")
        attrs.append("pulse.coupling_strength_hz = " + _f64_array(strengths))
    system = (
        f"qudit_frequency_hz = [{', '.join(frequencies)}], {levels}{levels and ', '}"
        f"t1_ns = [{', '.join(_f64_literal(v) for v in t1_ns)}], "
        f"t2_ns = [{', '.join(_f64_literal(v) for v in t2_ns)}], "
        f"drive_scale_rad_per_ns = [{', '.join(_f64_literal(v) for v in drive_scales)}]"
    )
    residual = target.residual_zz
    if residual:
        residual = list(compress(residual, map(active.issuperset, map(_QUDITS, residual))))
    if residual:
        terms = ", ".join(
            f"#pulse.static_term<zz, qudits = [{x.qudits[0]}, {x.qudits[1]}], "
            f"rate_hz = {_f64_literal(x.rate_hz)}>"
            for x in residual
        )
        system += f", static_terms = [{terms}]"
    return f"{', '.join(attrs)}, pulse.quantum_system = #pulse.quantum_system<{system}>"
