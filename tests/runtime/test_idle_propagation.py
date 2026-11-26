# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle stretches of coupled registers are propagated exactly.

Detuned exchange couplings turn in the qudits' frames, so an idle coupled
register is not constant there; in the frame that turns with the couplings
it is, and ``_idle`` propagates the stretch at once. These tests check that
against stepping at a fine step (which converges to the same evolution),
against the closed form of an exchange oscillation, and that the GPU path
runs its drives on the device and its idle stretches on the host.
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import Coupling, MicrowaveDrive, PhysicalQudit, ResidualZZ
from cudaq_pulse.runtime import _idle, _jit, _propagate
from cudaq_pulse.runtime import simulate as simulate_module

from tests.runtime.test_gpu_backend_logic import _CpuStandIn

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

_DT_NS = 0.05


def _system(n, *, t1_us=20.0, t2_us=12.0, thermal=0.0, g_hz=6.0e6, zz_hz=0.0):
    return pulse.quantum_system(
        name="idle",
        qudits={
            q: PhysicalQudit(
                q,
                5.0e9 + 0.27e9 * q,
                t1_us,
                t2_us,
                anharmonicity_hz=-230.0e6,
                thermal_population=thermal,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
            for q in range(n)
        },
        couplings=[Coupling((q, q + 1), g_hz) for q in range(n - 1)],
        residual_zz=[ResidualZZ((q, q + 1), zz_hz) for q in range(n - 1)] if zz_hz else [],
    )


@pulse.kernel
def ramsey(qs, tau: float):
    for q in qs:
        drive(get_drive_line(q), gaussian(40, 0.06, 10))
    wait(get_drive_line(qs[0]), tau)
    shift_phase(get_drive_line(qs[0]).tone, 0.7)
    for q in qs:
        drive(get_drive_line(q), gaussian(40, 0.06, 10))


@pulse.kernel
def idle(q0, q1, tau: float):
    wait(get_drive_line(q0), tau)
    wait(get_drive_line(q1), tau)


def _lowered(system, kernel, *args):
    return pulse.compile(kernel, quantum_system=system).lower(*args)


def _rho(lowered, *, dt_ns=_DT_NS, fast=True, monkeypatch=None, **kwargs):
    _propagate.clear_cache()
    if fast:
        return pulse.simulate(lowered, dt_ns=dt_ns, **kwargs).density_matrix
    with monkeypatch.context() as m:
        m.setattr(_idle, "step_through", lambda *a, **k: None)
        return pulse.simulate(lowered, dt_ns=dt_ns, **kwargs).density_matrix


@pytest.fixture
def gpu_on_cpu(monkeypatch):
    monkeypatch.setattr(_jit, "JITCompiler", lambda: None)
    monkeypatch.setattr(simulate_module, "_GpuBackend", _CpuStandIn)
    monkeypatch.setattr(_CpuStandIn, "runs", [])


@pytest.fixture
def advances(monkeypatch):
    calls = []
    advance = _idle.Idle.advance

    def counted(self, state, t, duration_ns):
        calls.append(duration_ns)
        return advance(self, state, t, duration_ns)

    monkeypatch.setattr(_idle.Idle, "advance", counted)
    return calls


@pulse.kernel
def idle3(q0, q1, q2, tau: float):
    for q in (q0, q1, q2):
        wait(get_drive_line(q), tau)


def _random_state(dim):
    rng = np.random.default_rng(dim)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize(
    ("n", "levels", "cycles", "parts"),
    [
        (2, 3, 600.0, {}),
        (2, 2, 600.0, {"thermal": 0.03, "zz_hz": -40e3}),
        (3, 2, 600.0, {"t1_us": 0.0, "t2_us": 0.0}),
        (3, 3, 160.0, {"thermal": 0.02}),
    ],
)
def test_an_idle_coupled_register_evolves_as_fine_stepping_does(
    n, levels, cycles, parts, advances, monkeypatch
):
    system = _system(n, **parts)
    args = [*system.qudits[0:n]] if n == 3 else [system.qudits[0], system.qudits[1]]
    lowered = _lowered(system, idle3 if n == 3 else idle, *args, cycles)
    kwargs = {"levels": levels, "initial_state": _random_state(levels**n)}
    exact = _rho(lowered, **kwargs)
    assert sum(advances) == pytest.approx(cycles / system.clock_ghz)
    stepped = _rho(lowered, fast=False, monkeypatch=monkeypatch, **kwargs)
    fine = _rho(lowered, dt_ns=_DT_NS / 4, fast=False, monkeypatch=monkeypatch, **kwargs)
    # Stepping errs by RK4's error, which falls as the step's fourth power
    # (most on the level-2 coherences of a random state, which turn fastest);
    # the exact propagation is nearer the fine run than stepping is.
    assert np.abs(stepped - fine).max() < 1e-3
    assert np.abs(exact - fine).max() < 0.1 * np.abs(stepped - fine).max() + 1e-12
    assert np.trace(exact).real == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(exact, exact.conj().T, atol=1e-15)


@pytest.mark.parametrize("levels", [2, 3])
def test_drives_around_an_idle_stretch_step_as_before(levels, advances, monkeypatch):
    system = _system(2, thermal=0.02)
    lowered = _lowered(system, ramsey, system.qudits[0:2], 600.0)
    exact = _rho(lowered, levels=levels)
    assert advances
    stepped = _rho(lowered, fast=False, monkeypatch=monkeypatch, levels=levels)
    np.testing.assert_allclose(exact, stepped, atol=1e-6)


def test_an_exchange_oscillation_follows_its_closed_form(advances):
    # Two qubits detuned by D exchange an excitation at
    # P(01) = 4 g^2 / (4 g^2 + D^2) sin^2(sqrt(4 g^2 + D^2) t / 2) (g, D in rad/ns).
    g_hz, d_hz = 6.0e6, 0.27e9
    system = _system(2, t1_us=0.0, t2_us=0.0, g_hz=g_hz)
    g, d = 2 * math.pi * g_hz * 1e-9, 2 * math.pi * d_hz * 1e-9
    initial = np.zeros(4)
    initial[1] = 1.0  # qudit 0 excited
    for cycles in (0.0, 3.0, 1001.0, 40000.0):
        rho = pulse.simulate(
            _lowered(system, idle, system.qudits[0], system.qudits[1], cycles),
            levels=2,
            initial_state=initial,
            t_end=cycles / system.clock_ghz,
        ).density_matrix
        t = cycles / system.clock_ghz
        omega = math.sqrt(4 * g * g + d * d)
        swapped = 4 * g * g / omega**2 * math.sin(omega * t / 2) ** 2
        assert rho[2, 2].real == pytest.approx(swapped, abs=1e-10)
        assert rho[1, 1].real == pytest.approx(1 - swapped, abs=1e-10)
    assert advances


def test_a_stretch_too_short_to_pay_is_stepped(advances):
    system = _system(2)
    pulse.simulate(_lowered(system, ramsey, system.qudits[0:2], 2.0), levels=2, dt_ns=_DT_NS)
    assert not advances


def _exchange(g, rate):
    """The four terms the lowering gives a coupling g turning at *rate*."""
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]])

    def term(quadrature, amplitude, op):
        return (1, [0.0, 1.8e308, amplitude, 0, 0, 0, 0, 0, quadrature, rate], op)

    return [
        term(0, g, np.kron(x, x)),
        term(0, g, np.kron(y, y)),
        term(1, g, np.kron(x, y)),
        term(1, -g, np.kron(y, x)),
    ]


def test_a_frame_that_does_not_hold_is_not_used():
    static = np.zeros((4, 4), dtype=complex)
    lowering = np.kron(np.eye(2), [[0, 1], [0, 0]]).astype(complex)
    assert _idle.propagator((2, 2), static, _exchange(0.04, 1.7), [lowering], 0.05) is not None
    # A collapse operator that the couplings' rotation does not turn by a
    # phase (sigma_x on a coupled qubit) leaves no frame to propagate in.
    flip = np.kron(np.eye(2), [[0, 1], [1, 0]]).astype(complex)
    assert _idle.propagator((2, 2), static, _exchange(0.04, 1.7), [flip], 0.05) is None
    # Nor does a term at its rate in both senses (one quadrature alone).
    assert _idle.propagator((2, 2), static, _exchange(0.04, 1.7)[:1], [], 0.05) is None


@pytest.mark.parametrize("t1_us", [0.0, 15.0])
def test_the_gpu_path_runs_drives_on_the_device_and_idles_on_the_host(t1_us, gpu_on_cpu, advances):
    system = _system(2, t1_us=t1_us, t2_us=t1_us / 2)
    lowered = _lowered(system, ramsey, system.qudits[0:2], 600.0)
    gpu = pulse.simulate(lowered, backend="gpu", levels=3, dt_ns=_DT_NS).density_matrix
    steps = [s for _, _, s in _CpuStandIn.runs]
    # The two drive stretches on the device, the wait between them on the host.
    assert len(steps) == 2 and all(s > 0 for s in steps)
    total_ns = 680 / system.clock_ghz
    assert sum(advances) + sum(steps) * _DT_NS == pytest.approx(total_ns, abs=1e-9)
    cpu = pulse.simulate(lowered, backend="cpu", levels=3, dt_ns=_DT_NS).density_matrix
    np.testing.assert_allclose(gpu, cpu, atol=1e-9 if t1_us else 1e-6)


def test_an_uncoupled_gpu_wait_is_propagated_on_the_host(gpu_on_cpu, advances):
    system = _system(1)

    @pulse.kernel
    def t1(q, tau: float):
        drive(get_drive_line(q), gaussian(40, 0.08, 10))
        wait(get_drive_line(q), tau)

    lowered = _lowered(system, t1, system.qudits[0], 4000.0)
    gpu = pulse.simulate(lowered, backend="gpu", levels=2, dt_ns=_DT_NS).density_matrix
    assert len(_CpuStandIn.runs) == 1 and advances
    cpu = pulse.simulate(lowered, backend="cpu", levels=2, dt_ns=_DT_NS).density_matrix
    np.testing.assert_allclose(gpu, cpu, atol=1e-10)


def test_short_gpu_idles_stay_on_the_device(gpu_on_cpu, advances):
    system = _system(2)
    lowered = _lowered(system, ramsey, system.qudits[0:2], 40.0)
    pulse.simulate(lowered, backend="gpu", levels=2, dt_ns=_DT_NS)
    assert not advances
    # One run of the whole evolution: its 1200 steps, of which 400 idle.
    assert [s for _, _, s in _CpuStandIn.runs] == [round(60.0 / _DT_NS)]
    assert simulate_module._GpuBackend is _CpuStandIn


@pytest.fixture
def probes(monkeypatch):
    """Of each call of ``_idle.through_gpu``: whether it propagated a stretch
    on the host, and the operators on the whole register built during it
    (for a call that declined, the ones the check built; a call that splits
    also runs the stand-in's device runs, which build their own)."""
    calls = []
    built = []
    through_gpu, full_operator = _idle.through_gpu, simulate_module._full_operator

    def counted(factors, dims):
        if built:
            built[-1] += 1
        return full_operator(factors, dims)

    def probed(*args):
        built.append(0)
        try:
            result = through_gpu(*args)
        finally:
            count = built.pop()
        calls.append((result is not None, count))
        return result

    monkeypatch.setattr(simulate_module, "_full_operator", counted)
    monkeypatch.setattr(_idle, "through_gpu", probed)
    return calls


def test_a_gpu_interval_driven_throughout_builds_no_host_operators(gpu_on_cpu, advances, probes):
    system = _system(2)

    @pulse.kernel
    def driven(q0, q1):
        get_drive_line(q1)
        drive(get_drive_line(q0), square(200, 0.01))

    lowered = _lowered(system, driven, system.qudits[0], system.qudits[1])
    pulse.simulate(lowered, backend="gpu", levels=2, dt_ns=_DT_NS)
    # One interval of 2000 steps with no idle stretch: asked once, declined
    # from the drive's support alone, and run whole on the device.
    assert probes == [(False, 0)]
    assert not advances
    assert [s for _, _, s in _CpuStandIn.runs] == [2000]


def test_a_dissipative_gpu_register_above_the_cutoff_builds_no_host_operators(
    gpu_on_cpu, advances, probes
):
    system = _system(3)
    assert 4**3 > _idle.MAX_DISSIPATIVE_DIM
    lowered = _lowered(system, idle3, *system.qudits[0:3], 120.0)
    pulse.simulate(lowered, backend="gpu", levels=4, dt_ns=_DT_NS)
    # The whole interval is idle, 1200 steps, but the register keeps no
    # superoperator: declined on its size, and stepped on the device.
    assert probes == [(False, 0)]
    assert not advances
    assert [s for _, _, s in _CpuStandIn.runs] == [1200]
