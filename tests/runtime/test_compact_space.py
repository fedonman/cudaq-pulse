# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The simulated space holds the qudits a program uses, not every index up to
the largest: a pair at indices 13 and 3 of the 17-qudit transmon_surface17 quantum system is a
two-mode problem, with the same physics as on a quantum system of just those two."""

from __future__ import annotations

import dataclasses

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems.base import layout_anharmonicities, layout_frequencies

_TRANSMON_SURFACE17 = pulse.quantum_system("transmon_surface17")


def _slice(indices):
    """A quantum system of the transmon_surface17 qudits *indices*, renumbered 0, 1, ... in
    increasing order, with the couplings and residual ZZ between them. Each
    qudit is at the frequency, and the anharmonicity, its left-out
    neighbours dress it to, as when transmon_surface17 simulates those
    qudits (dressed frames)."""
    target = _TRANSMON_SURFACE17
    new = {q: i for i, q in enumerate(sorted(indices))}
    moved = layout_frequencies(target._description, tuple(new))
    # At three levels a kept qudit also keeps its left-out neighbours' share
    # of its 1-2 transition.
    alphas = layout_anharmonicities(target._description, tuple(new))

    def carrier(q, i):
        physical = target.physical_qudits[q].replace(index=i)
        if q in moved:
            physical = physical.replace(frequency_hz=moved[q])
        if q in alphas:
            physical = physical.replace(anharmonicity_hz=alphas[q])
        return physical

    def renumbered(entries, replace):
        return [
            replace(e, qudits=tuple(new[q] for q in e.qudits))
            for e in entries
            if all(q in new for q in e.qudits)
        ]

    sliced = pulse.quantum_system(
        name="transmon-surface17-slice",
        qudits={i: carrier(q, i) for q, i in new.items()},
        couplings=renumbered(target.couplings, pulse.quantum_systems.Coupling.replace),
        residual_zz=renumbered(target.residual_zz, dataclasses.replace),
        clock_ghz=target.clock_ghz,
    )
    return sliced.qudits, new


@pulse.kernel
def pair(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, gaussian(40, 0.08, 10.0))
    drive(db, gaussian(40, 0.03, 10.0))
    sync(da, db)
    wait(da, 20)


@pulse.kernel
def drive_and_measure(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    ra = get_readout_line(a)
    drive(da, gaussian(40, 0.08, 10.0))
    drive(db, gaussian(40, 0.03, 10.0))
    sync(da, db, ra)
    readout(ra, square(40, 0.05))


@pulse.kernel
def triple(a, b, c):
    da = get_drive_line(a)
    db = get_drive_line(b)
    dc = get_drive_line(c)
    drive(da, gaussian(40, 0.08, 10.0))
    drive(db, gaussian(40, 0.05, 10.0))
    drive(dc, gaussian(40, 0.03, 10.0))


def _same(a, b):
    assert a.dims == b.dims
    np.testing.assert_allclose(a.density_matrix, b.density_matrix, atol=1e-12)


def test_a_high_index_pair_is_a_two_mode_problem():
    q = _TRANSMON_SURFACE17.qudits
    lowered = pair.lower(q[13], q[3])
    assert lowered.qudits == (3, 13)
    result = pulse.simulate(lowered, levels=3, t_end=2.0)
    assert result.qudits == (3, 13)
    assert result.dims == (3, 3)
    assert result.density_matrix.shape == (9, 9)


def test_bound_but_unused_qudits_are_not_simulated():
    q = _TRANSMON_SURFACE17.qudits

    @pulse.kernel
    def one_of_two(a, b):
        da = get_drive_line(a)
        drive(da, gaussian(40, 0.08, 10.0))

    result = pulse.simulate(one_of_two.lower(q[13], q[3]), t_end=2.0, levels=2)
    assert result.qudits == (13,) and result.dims == (2,)


@pytest.mark.parametrize("levels", [2, 3])
def test_a_slice_simulates_like_the_device_of_its_qudits(levels):
    q = _TRANSMON_SURFACE17.qudits
    h, new = _slice((13, 3))
    on_chip = pulse.simulate(pair.lower(q[13], q[3]), levels=levels)
    alone = pulse.simulate(pair.lower(h[new[13]], h[new[3]]), levels=levels)
    _same(on_chip, alone)
    assert on_chip.qudits == (3, 13) and alone.qudits == (0, 1)


def test_modes_are_ordered_by_device_index_whatever_the_argument_order():
    q = _TRANSMON_SURFACE17.qudits
    h, new = _slice((13, 0, 3))
    for order in ((13, 0, 3), (3, 13, 0), (0, 3, 13)):
        on_chip = pulse.simulate(triple.lower(*(q[i] for i in order)), levels=2)
        alone = pulse.simulate(triple.lower(*(h[new[i]] for i in order)), levels=2)
        assert on_chip.qudits == (0, 3, 13)
        _same(on_chip, alone)
    # The argument order changes which qudit gets which drive.
    first = pulse.simulate(triple.lower(q[13], q[0], q[3]), levels=2)
    second = pulse.simulate(triple.lower(q[3], q[13], q[0]), levels=2)
    assert not np.allclose(first.density_matrix, second.density_matrix, atol=1e-6)


@pytest.mark.parametrize("measured", [13, 3])
def test_measurements_report_device_qudits(measured):
    q = _TRANSMON_SURFACE17.qudits
    h, new = _slice((13, 3))
    other = 3 if measured == 13 else 13
    on_chip = pulse.simulate(
        drive_and_measure.lower(q[measured], q[other]), shots_count=200, seed=5, levels=2
    )
    alone = pulse.simulate(
        drive_and_measure.lower(h[new[measured]], h[new[other]]), shots_count=200, seed=5, levels=2
    )
    assert [m.qudit for m in on_chip.measurements] == [measured]
    assert [m.qudit for m in alone.measurements] == [new[measured]]
    _same(on_chip, alone)
    assert [(b.levels, b.probability) for b in on_chip.branches] == [
        (b.levels, pytest.approx(b.probability)) for b in alone.branches
    ]
    np.testing.assert_array_equal(on_chip.records, alone.records)


def test_an_initial_state_spans_the_used_qudits():
    q = _TRANSMON_SURFACE17.qudits
    h, new = _slice((13, 3))
    psi = np.zeros(4)
    psi[0b10] = 1.0  # qudits[1] = 13 excited
    on_chip = pulse.simulate(pair.lower(q[13], q[3]), initial_state=psi, t_end=5.0, levels=2)
    alone = pulse.simulate(
        pair.lower(h[new[13]], h[new[3]]), initial_state=psi, t_end=5.0, levels=2
    )
    _same(on_chip, alone)
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        pulse.simulate(pair.lower(q[13], q[3]), initial_state=np.zeros(2**14), levels=2)


@pytest.mark.gpu
def test_gpu_simulates_a_high_index_pair_like_the_cpu():
    q = _TRANSMON_SURFACE17.qudits
    lowered = drive_and_measure.lower(q[13], q[3])
    cpu = pulse.simulate(lowered, levels=3)
    gpu = pulse.simulate(lowered, levels=3, backend="gpu")
    assert gpu.qudits == cpu.qudits == (3, 13) and gpu.dims == (3, 3)
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=1e-6)
    assert [b.levels for b in gpu.branches] == [b.levels for b in cpu.branches]
