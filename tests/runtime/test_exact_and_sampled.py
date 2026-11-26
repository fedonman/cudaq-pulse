# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU integrator's exact and kept propagators, and sampled paths.

Stretches with a constant Lindbladian (idle waits, square pulses in their
frame) are propagated exactly; driven stretches on small registers by their
RK4 map, built at once. Both must agree with stepping RK4 on the state, and
``branches="sampled"`` must follow only the branches shots take.
"""

from __future__ import annotations

import math
import time
import tracemalloc

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import ControlSystem
from cudaq_pulse.classical import Link
from cudaq_pulse.quantum_systems import MicrowaveDrive, PhysicalQudit
from cudaq_pulse.runtime import _propagate
from cudaq_pulse.runtime import simulate as simulate_module

from tests._handles import Q
from tests.runtime._lowering import on

_PI = math.pi / 20.0


def _target(t1_us=30.0, t2_us=20.0, thermal=0.02, offset_hz=0.0):
    return pulse.quantum_system(
        name="propagation-test",
        qudits={
            0: PhysicalQudit(
                index=0,
                frequency_hz=5.0e9 + offset_hz,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=t2_us,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
                thermal_population=thermal,
            )
        },
    )


@pulse.kernel
def _mixed_pulses(q0):
    d = get_drive_line(q0)
    drive(d, gaussian(120, _PI, 30))
    # Short enough that stepping does not damp a qutrit's 0-2 coherence.
    wait(d, 40)
    shift_phase(d.tone, 0.3)
    drive(d, square(40, _PI / 2))
    drive(d, drag(60, _PI, 15, 0.4))
    shift_frequency(d.tone, 3e6)
    drive(d, square(80, _PI / 3))


def _simulated(lowered, monkeypatch, *, maps=True, exact=True, **kwargs):
    """The density matrix with kept RK4 maps and exact propagation on or
    off (off: every stretch stepped by RK4 on the state)."""
    with monkeypatch.context() as m:
        if not maps:
            m.setattr(_propagate, "MAX_MAPPED_DIM", 0)
        if not exact:
            m.setattr(simulate_module._Stepper, "_exact_pays", lambda self, n, h: False)
        _propagate.clear_cache()
        return pulse.simulate(lowered, **kwargs).density_matrix


@pytest.mark.parametrize("levels", [2, 3])
def test_kept_rk4_maps_give_what_stepping_gives(levels, monkeypatch):
    lowered = on(_target(), _mixed_pulses, Q[0])
    stepped = _simulated(lowered, monkeypatch, maps=False, exact=False, levels=levels)
    for _ in range(2):  # built, then kept
        with monkeypatch.context() as m:
            m.setattr(simulate_module._Stepper, "_exact_pays", lambda self, n, h: False)
            rho = pulse.simulate(lowered, levels=levels).density_matrix
        np.testing.assert_allclose(rho, stepped, atol=1e-10)


@pytest.mark.parametrize("levels", [2, 3])
def test_exact_stretches_agree_with_stepping_within_its_error(levels, monkeypatch):
    lowered = on(_target(), _mixed_pulses, Q[0])
    errors = []
    for dt in (0.05, 0.025):
        exact = _simulated(lowered, monkeypatch, levels=levels, dt_ns=dt)
        stepped = _simulated(lowered, monkeypatch, exact=False, levels=levels, dt_ns=dt)
        errors.append(np.abs(exact - stepped).max())
    # They differ by RK4's error on the square pulses, which falls as dt^4.
    assert errors[0] < 1e-7
    assert errors[1] < errors[0] / 8 or errors[0] < 1e-12


@pulse.kernel
def _excite_and_wait(q0, wait_cycles: int):
    d = get_drive_line(q0)
    drive(d, square(40, _PI))
    wait(d, wait_cycles)


def test_a_long_idle_decays_exactly_and_quickly():
    target = _target(t1_us=30.0, thermal=0.0)
    short = pulse.simulate(on(target, _excite_and_wait, Q[0], 200_000), levels=2).density_matrix
    start = time.perf_counter()
    long = pulse.simulate(on(target, _excite_and_wait, Q[0], 400_000), levels=2).density_matrix
    elapsed = time.perf_counter() - start
    # 100 us more of pure decay.
    assert long[1, 1].real / short[1, 1].real == pytest.approx(math.exp(-100.0 / 30.0), rel=1e-9)
    # Stepping 0.05 ns at a time would take minutes.
    assert elapsed < 2.0


def test_a_detuned_system_is_propagated_like_stepping(monkeypatch):
    lowered = on(_target(), _mixed_pulses, Q[0])
    detuned = _target(offset_hz=250e3)
    stepped = _simulated(lowered, monkeypatch, maps=False, exact=False, quantum_system=detuned)
    rho = _simulated(lowered, monkeypatch, quantum_system=detuned)
    np.testing.assert_allclose(rho, stepped, atol=1e-7)


@pulse.kernel
def _long_gaussian(q0):
    d = get_drive_line(q0)
    drive(d, gaussian(40, _PI, 10))


def test_a_long_driven_stretch_builds_its_hamiltonians_a_part_at_a_time():
    # 16 levels are too many for a kept map, so the pulse steps RK4 on the
    # state, 8400 steps at the default step. Their Hamiltonians (three of
    # 16 x 16 a step) take about 100 MB together.
    lowered = on(_target(), _long_gaussian, Q[0])
    tracemalloc.start()
    try:
        pulse.simulate(lowered, levels=16)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 48 << 20


def test_parts_of_a_driven_stretch_step_as_the_whole_stretch(monkeypatch):
    lowered = on(_target(), _mixed_pulses, Q[0])
    whole = pulse.simulate(lowered, levels=6).density_matrix
    with monkeypatch.context() as m:
        # Seven steps a part, and a shorter last part.
        m.setattr(simulate_module, "_STRETCH_BYTES", 7 * 96 * 36)
        parts = pulse.simulate(lowered, levels=6).density_matrix
    assert parts.tobytes() == whole.tobytes()


def test_expm_matches_a_diagonalizable_exponential():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    a = 40.0 * (a + a.conj().T)  # large norm: many squarings
    values, vectors = np.linalg.eigh(a)
    expected = vectors @ np.diag(np.exp(-1j * values)) @ vectors.conj().T
    np.testing.assert_allclose(_propagate.expm(-1j * a), expected, atol=1e-9)


@pytest.mark.parametrize(("rows", "columns"), [(1, 1), (2, 3), (4, 4)])
def test_kron_is_numpys_kron(rows, columns):
    rng = np.random.default_rng(rows * columns)
    a = rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
    for b in (rng.normal(size=(3, 2)), np.eye(columns), -a.T):
        for x, y in ((a, b), (b, a)):
            expected = np.kron(x, y)
            got = _propagate.kron(x, y)
            assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes()


@pulse.kernel
def _three_measurements(q0):
    d = get_drive_line(q0)
    r = get_readout_line(q0)
    for _ in range(3):
        drive(d, square(40, _PI / 2))
        sync(d, r)
        readout(r, square(200, 0.05))
        sync(d, r)


def test_sampled_branches_have_the_frequencies_of_the_exact_branches():
    lowered = on(_target(t1_us=0.0, t2_us=0.0, thermal=0.0), _three_measurements, Q[0])
    exact = {b.levels: b.probability for b in pulse.simulate(lowered, levels=2).branches}
    shots_count = 4000
    sampled = pulse.simulate(lowered, shots_count=shots_count, seed=5, branches="sampled", levels=2)
    assert len(sampled.branches) <= shots_count
    assert sum(b.probability for b in sampled.branches) == pytest.approx(1.0)
    assert np.trace(sampled.density_matrix).real == pytest.approx(1.0)
    for branch in sampled.branches:
        p = exact[branch.levels]
        assert abs(branch.probability - p) < 5 * math.sqrt(p / shots_count)
    assert sampled.records is not None and sampled.records.shape == (shots_count, 3)


def test_sampled_branches_need_shots():
    lowered = on(_target(), _three_measurements, Q[0])
    with pytest.raises(ValueError, match="shots_count"):
        pulse.simulate(lowered, branches="sampled", levels=2)
    with pytest.raises(ValueError, match="unknown branches 'some'; choose 'all' or 'sampled'"):
        pulse.simulate(lowered, shots_count=1, branches="some", levels=2)


def test_branch_coverage_takes_its_members_or_their_strings():
    lowered = on(_target(), _three_measurements, Q[0])
    by_string = pulse.simulate(lowered, shots_count=50, seed=2, branches="sampled", levels=2)
    by_member = pulse.simulate(
        lowered, shots_count=50, seed=2, branches=pulse.BranchCoverage.SAMPLED, levels=2
    )
    assert by_member.branches == by_string.branches
    assert pulse.BranchCoverage("all") is pulse.BranchCoverage.ALL


_MACHINE = ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8)})


@pulse.classical(max_latency_ns=200, link="controller")
def _halve_or_double(bit: pulse.Bits, wait_cycles: float) -> float:
    return min(max(wait_cycles * (2.0 if bit[0] else 0.5), 20.0), 20000.0)


@pulse.kernel
def _adaptive_loop(q0, rounds: int):
    d = get_drive_line(q0)
    r = get_readout_line(q0)
    w = 2000.0
    drive(d, square(40, _PI))
    wait(d, 2000)
    for _ in range(rounds):
        sync(d, r)
        m = readout(r, square(200, 0.05))
        w = _halve_or_double.submit([m], w).result()
        sync(d, r)
        if m:
            drive(d, square(40, _PI))
        drive(d, square(40, _PI))
        wait(d, w, bounds=(20, 20000))


def test_a_long_adaptive_loop_follows_one_path_per_shot():
    dev = _target(t1_us=5.0)
    lowered = pulse.compile(_adaptive_loop, quantum_system=dev, control_system=_MACHINE).lower(
        dev.qudits[0], 60
    )
    result = pulse.simulate(lowered, shots_count=3, seed=1, branches="sampled", levels=2)
    assert 1 <= len(result.branches) <= 3
    assert all(len(branch.calls) == 60 for branch in result.branches)
    assert result.records is not None and result.records.shape == (3, 60)


def test_kept_propagators_are_keyed_compactly():
    @pulse.kernel(unroll_threshold=0)
    def train(q0):
        d = get_drive_line(q0)
        for _ in range(4):
            drive(d, gaussian(400, _PI, 100))

    _propagate.clear_cache()
    pulse.simulate(on(_target(), train, Q[0]), levels=2)
    sizes = [
        sum(len(part) for part in key[2:] if isinstance(part, bytes)) for key in _propagate._cache
    ]
    # A key names a stretch by a digest of its coefficients, whatever its
    # length: the cache would otherwise hold every stretch's samples.
    assert sizes and max(sizes) <= 1024


@pulse.kernel
def _pulses_apart(q0):
    # Written out: a loop's pulses would be one term, a train.
    d = get_drive_line(q0)
    drive(d, square(24, _PI / 2))
    wait(d, 40)
    shift_phase(d.tone, 0.3)
    drive(d, gaussian(60, _PI, 15))
    wait(d, 24)
    drive(d, square(24, _PI / 4))
    wait(d, 36)
    shift_frequency(d.tone, 3e6)
    drive(d, gaussian(48, _PI / 2, 12))
    wait(d, 28)
    # Two that touch, and one after them.
    drive(d, drag(40, _PI, 10, 0.4))
    drive(d, square(16, _PI / 3))
    wait(d, 32)
    drive(d, square(8, _PI))


@pulse.kernel
def _lines_apart(q0, q1):
    # Two lines at once, one longer than the other, then both idle.
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    drive(d0, gaussian(24, _PI, 6))
    drive(d1, square(40, _PI / 2))
    sync(d0, d1)
    wait(d0, 30)
    sync(d0, d1)
    drive(d0, square(40, _PI / 3))
    drive(d1, gaussian(24, _PI, 6))
    sync(d0, d1)
    wait(d0, 26)
    sync(d0, d1)
    drive(d1, drag(32, _PI, 8, 0.4))
    sync(d0, d1)
    wait(d1, 34)
    sync(d0, d1)
    drive(d0, square(16, _PI))
    drive(d1, square(16, _PI / 2))
    sync(d0, d1)
    wait(d0, 30)
    sync(d0, d1)
    drive(d0, gaussian(36, _PI / 2, 9))


def _two_qudits():
    return pulse.quantum_system(
        name="propagation-test-pair",
        qudits={
            q: PhysicalQudit(
                index=q,
                frequency_hz=5.0e9 + 0.3e9 * q,
                anharmonicity_hz=-200.0e6,
                t1_us=30.0,
                t2_star_us=20.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
            for q in range(2)
        },
    )


@pytest.mark.parametrize("stepping", ["exact and maps", "maps", "rk4"])
@pytest.mark.parametrize("levels", [2, 3])
@pytest.mark.parametrize("program", ["one line", "two lines"])
def test_drives_apart_step_with_their_own_terms_to_the_same_bits(
    program, levels, stepping, monkeypatch
):
    """Past ``_APART_RANGES`` stretches of drives, each stretch is stepped
    with its own terms alone. A term adds nothing outside its stretch and
    the terms of a stretch add in the order they have, so the density
    matrix is the one that asking every term in every stretch gives, bit
    for bit: whether stretches are propagated exactly, by kept maps or by
    RK4 on the state."""
    if program == "one line":
        lowered = on(_target(), _pulses_apart, Q[0])
    else:
        lowered = on(_two_qudits(), _lines_apart, Q[0], Q[1])
    stepper = simulate_module._Stepper
    play_apart, play = stepper.play_apart, stepper.play
    apart: list[int] = []
    asked: list[int] = []

    def counting_apart(self, rho, t0, steps, ranges):
        apart.append(len(ranges))
        return play_apart(self, rho, t0, steps, ranges)

    def counting(self, rho, t0, lo, hi):
        asked.append(len(self.driven))
        return play(self, rho, t0, lo, hi)

    # To the end, and to the middle of a pulse of the fourth stretch.
    for window in ({}, {"t_end": 105.0}):
        results, terms_asked = [], []
        for limit in (simulate_module._APART_RANGES, 10**9):
            apart.clear()
            asked.clear()
            with monkeypatch.context() as m:
                m.setattr(simulate_module, "_APART_RANGES", limit)
                m.setattr(stepper, "play_apart", counting_apart)
                m.setattr(stepper, "play", counting)
                results.append(
                    _simulated(
                        lowered,
                        m,
                        maps=stepping != "rk4",
                        exact=stepping == "exact and maps",
                        levels=levels,
                        **window,
                    )
                )
            terms_asked.append(sum(asked))
            # Each stretch asks its own terms alone, or every stretch asks all.
            assert bool(apart) == (limit == simulate_module._APART_RANGES)
            if apart:
                assert max(apart) > simulate_module._APART_RANGES
        own, every = results
        assert own.dtype == every.dtype and own.shape == every.shape
        assert own.tobytes() == every.tobytes()
        assert terms_asked[0] * 3 < terms_asked[1]


def test_up_to_three_stretches_of_drives_ask_every_term(monkeypatch):
    """The same steps as before for a program with few stretches."""

    @pulse.kernel
    def three(q0):
        d = get_drive_line(q0)
        drive(d, square(24, _PI / 2))
        wait(d, 40)
        drive(d, gaussian(60, _PI, 15))
        wait(d, 24)
        drive(d, square(8, _PI))

    def refuse(*_):
        raise AssertionError("three stretches were stepped apart")

    monkeypatch.setattr(simulate_module._Stepper, "play_apart", refuse)
    rho = pulse.simulate(on(_target(), three, Q[0]), levels=2).density_matrix
    assert np.isclose(np.trace(rho).real, 1.0)
