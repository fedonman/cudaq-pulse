# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement and feedback simulation against analytic results.

Every test runs the CPU reference; the GPU-marked tests check that the
cuDensityMat backend agrees with it.
"""

from __future__ import annotations

import math
import warnings

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import Coupling, MicrowaveDrive, PhysicalQudit

from tests._handles import Q
from tests.runtime._lowering import on

# 40 virtual units at 2 GHz is 20 ns; with H = amp X / 2 these amplitudes
# rotate by pi and pi / 2.
_PI = math.pi / 20.0
_HALF_PI = math.pi / 40.0


def _target(n=1, t1_us=0.0):
    return pulse.quantum_system(
        name="simulate-test",
        qudits={
            q: PhysicalQudit(
                index=q,
                frequency_hz=5.0e9 + 0.1e9 * q,
                anharmonicity_hz=-200.0e6,
                t1_us=t1_us,
                t2_star_us=0.0,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
            for q in range(n)
        },
    )


def _active_reset(prepare):
    @pulse.kernel
    def reset(q0):
        d0 = get_drive_line(q0)
        r0 = get_readout_line(q0)
        drive(d0, square(40, prepare))
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))

    return reset


@pulse.kernel
def conditional_flip(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    sync(d0, r0)
    bit = readout(r0, square(200, 0.05))
    if bit:
        drive(d1, square(40, _PI))


@pulse.kernel
def measure(q0):
    r0 = get_readout_line(q0)
    readout(r0, square(200, 0.05))


def _run(kernel, n=1, target=None, levels=2, **kwargs):
    """Simulate *kernel* with two levels per qudit unless *levels* says
    otherwise: the checks compare with two-level formulas."""
    lowered = on(target or _target(n), kernel, *[Q[i] for i in range(n)])
    return pulse.simulate(lowered, levels=levels, **kwargs)


def _rabi(amplitude):
    @pulse.kernel
    def rabi(q0):
        d0 = get_drive_line(q0)
        drive(d0, square(40, amplitude))

    return rabi


def test_rabi_matches_the_analytic_rotation():
    for amplitude in (_HALF_PI, _PI, 0.03):
        rho = _run(_rabi(amplitude)).density_matrix
        assert rho[1, 1].real == pytest.approx(math.sin(amplitude * 20.0 / 2) ** 2, abs=1e-8)


def test_simulate_takes_the_kernel_and_its_arguments():
    @pulse.kernel
    def rabi(q0, amplitude: float):
        d0 = get_drive_line(q0)
        drive(d0, square(40, amplitude))

    dev = _target()
    q = dev.qudits[0]
    expected = pulse.simulate(rabi.lower(q, _PI), levels=2).density_matrix
    np.testing.assert_array_equal(pulse.simulate(rabi, q, _PI, levels=2).density_matrix, expected)
    compiled = pulse.compile(rabi)
    np.testing.assert_array_equal(
        pulse.simulate(compiled, q, _PI, shots_count=2, seed=1, levels=2).density_matrix, expected
    )
    with pytest.raises(pulse.InvalidTypeError, match="module of a call, which has its arguments"):
        pulse.simulate(rabi.lower(q, _PI), q, levels=2)
    with pytest.raises(
        pulse.InvalidTypeError, match=r"expects a kernel and its arguments.*got str"
    ):
        pulse.simulate("rabi", q, levels=2)  # type: ignore[arg-type]


@pulse.kernel
def _rabi_with_amplitude(q0, amplitude: float):
    d0 = get_drive_line(q0)
    drive(d0, square(40, amplitude))


@pytest.mark.parametrize("kind", ["kernel", "compiled", "with_parameters", "bound", "call"])
def test_simulate_takes_a_kernel_or_a_program_compiled_already(kind):
    dev = _target()
    q = dev.qudits[0]
    compiled = pulse.compile(_rabi_with_amplitude)
    generic = compiled.lower(q, _PI).generic
    assert generic is not None and generic.parameters == ("amplitude",)
    program, args = {
        "kernel": (_rabi_with_amplitude, (q, _PI)),
        "compiled": (compiled, (q, _PI)),
        "with_parameters": (generic, (_PI,)),
        "bound": (generic.bind([_PI]), ()),
        "call": (compiled.lower(q, _PI), ()),
    }[kind]
    rho = pulse.simulate(program, *args, levels=2).density_matrix
    assert rho[1, 1].real == pytest.approx(math.sin(_PI * 20.0 / 2) ** 2, abs=1e-8)


def test_simulate_binds_a_program_s_parameters_as_bind_does():
    @pulse.kernel
    def wait_for(q0, gap: float):
        d0 = get_drive_line(q0)
        wait(d0, gap, bounds=(0, 100))
        drive(d0, square(40, _PI))

    dev = _target()
    q = dev.qudits[0]
    generic = pulse.compile(wait_for).lower(q, 10.0).generic
    assert generic is not None
    with pytest.raises(pulse.InvalidValueError, match=r"float parameters \(gap\); pass their"):
        pulse.simulate(generic, levels=2)
    with pytest.raises(ValueError, match="bounds") as from_bind:
        generic.bind([500.0])
    with pytest.raises(type(from_bind.value), match="bounds"):
        pulse.simulate(generic, 500.0, levels=2)
    with pytest.raises(pulse.InvalidTypeError, match="module of a call, which has its arguments"):
        pulse.simulate(generic.bind([10.0]), 10.0, levels=2)


def test_active_reset_returns_the_excited_state_to_ground():
    result = _run(_active_reset(_PI), shots_count=200, seed=1)
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    assert result.records is not None and (result.records[:, 0] == 1).all()


def test_active_reset_of_a_superposition():
    result = _run(_active_reset(_HALF_PI), shots_count=4000, seed=2)
    # Either outcome ends in |0>: the ensemble is pure.
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    probabilities = {b.levels: b.probability for b in result.branches}
    assert probabilities[(0,)] == pytest.approx(0.5, abs=1e-8)
    assert probabilities[(1,)] == pytest.approx(0.5, abs=1e-8)
    ones = result.records[:, 0].mean()
    assert abs(ones - 0.5) < 4 * math.sqrt(0.25 / 4000)


def test_conditional_flip_correlates_the_qudits():
    result = _run(conditional_flip, 2, shots_count=1000, seed=3)
    expected = np.zeros((4, 4))
    expected[0, 0] = expected[3, 3] = 0.5  # |00> and |11>, qubit 0 least significant
    np.testing.assert_allclose(result.density_matrix, expected, atol=1e-7)
    assert sorted((b.levels, b.probability) for b in result.branches) == [
        ((0,), pytest.approx(0.5)),
        ((1,), pytest.approx(0.5)),
    ]


def test_decay_during_the_readout_sets_the_outcome_probability():
    t1_us = 0.5
    result = _run(measure, target=_target(1, t1_us), initial_state=[0.0, 1.0])
    # The readout is 100 ns; the measurement completes at its end.
    p1 = math.exp(-100.0 / (t1_us * 1e3))
    assert {b.levels: b.probability for b in result.branches}[(1,)] == pytest.approx(p1, rel=1e-6)
    np.testing.assert_allclose(np.diag(result.density_matrix).real, [1 - p1, p1], atol=1e-7)


def test_measuring_a_qutrit_dephases_all_three_levels():
    psi = np.ones(3) / math.sqrt(3)
    result = _run(measure, levels=3, initial_state=psi, shots_count=3000, seed=4)
    np.testing.assert_allclose(result.density_matrix, np.eye(3) / 3, atol=1e-8)
    assert {b.levels[0]: b.probability for b in result.branches} == pytest.approx(
        {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    )
    counts = np.bincount(result.records[:, 0], minlength=3) / 3000
    assert np.all(np.abs(counts - 1 / 3) < 4 * math.sqrt(2 / 9 / 3000))


@pulse.kernel
def repeated_reset(q0):
    d0 = get_drive_line(q0)
    r0 = get_readout_line(q0)
    drive(d0, square(40, _HALF_PI))
    for _ in range(3):
        sync(d0, r0)
        bit = readout(r0, square(200, 0.05))
        if bit:
            drive(d0, square(40, _PI))


def test_resets_in_a_retained_loop():
    assert "scf.for" in repeated_reset.lower(Q[0]).mlir()
    result = _run(repeated_reset, shots_count=500, seed=5)
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    # Three measurements, one per iteration; after the first reset every
    # later one reads 0.
    assert [m.iteration for m in result.measurements] == [(0,), (1,), (2,)]
    likely = sorted(b.levels for b in result.branches if b.probability > 1e-9)
    assert likely == [(0, 0, 0), (1, 0, 0)]
    assert (result.records[:, 1:] == 0).all()


def test_a_retained_loop_simulates_like_its_unrolling():
    @pulse.kernel
    def rolled(q0):
        d0 = get_drive_line(q0)
        for _ in range(4):
            drive(d0, square(40, 0.02))
            shift_phase(d0.tone, 0.7)

    @pulse.kernel
    def unrolled(q0):
        d0 = get_drive_line(q0)
        for i in range(4):
            drive(d0, square(40, 0.02 + 0 * i))
            shift_phase(d0.tone, 0.7)

    a = _run(rolled).density_matrix
    b = _run(unrolled).density_matrix
    assert "scf.for" in rolled.lower(Q[0]).mlir()
    np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kernel, n", [(_active_reset(_HALF_PI), 1), (conditional_flip, 2)], ids=["reset", "flip"]
)
def test_gpu_backend_matches_the_reference(kernel, n):
    cpu = _run(kernel, n)
    gpu = _run(kernel, n, backend="gpu")
    np.testing.assert_allclose(gpu.density_matrix, cpu.density_matrix, atol=1e-6)
    assert [b.levels for b in gpu.branches] == [b.levels for b in cpu.branches]


def test_each_qudit_has_its_model_s_levels_unless_simulate_sets_them():
    from cudaq_pulse.quantum_systems import TransmonModel

    lowered = on(_target(1), measure, Q[0])
    assert pulse.simulate(lowered).dims == (TransmonModel.levels,) == (3,)
    target = pulse.quantum_system(
        name="simulate-test", qudits=_target(2).physical_qudits, model=TransmonModel(levels={1: 4})
    )
    lowered = on(target, conditional_flip, Q[0], Q[1])
    assert pulse.simulate(lowered).dims == (3, 4)
    assert pulse.simulate(lowered, levels=2).dims == (2, 2)
    assert pulse.simulate(lowered, levels={0: 2}).dims == (2, 4)


def test_level_populations_are_the_diagonal_of_a_qudit_s_reduced_state():
    one = _run(_rabi(_HALF_PI), levels=3)
    rho = one.density_matrix
    assert one.level_populations() == [float(rho[k, k].real) for k in range(3)]
    assert one.level_population(1) == float(rho[1, 1].real)
    assert one.level_population(1, qudit=0) == one.level_population(1, qudit=Q[0])

    two = _run(conditional_flip, 2, levels={0: 2, 1: 3})
    assert two.dims == (2, 3)
    diagonal = np.real(np.diagonal(two.density_matrix)).reshape(3, 2)
    assert two.level_populations(qudit=0) == pytest.approx(list(diagonal.sum(axis=0)), abs=1e-15)
    assert two.level_populations(qudit=1) == pytest.approx(list(diagonal.sum(axis=1)), abs=1e-15)
    assert sum(two.level_populations(1)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda r: r.level_populations(), r"the result models qudits \[0, 1\]; pass qudit="),
        (lambda r: r.level_population(1), r"models qudits \[0, 1\]; pass qudit="),
        (lambda r: r.level_population(1, qudit=5), r"qudit 5 is not in the result; it models "),
        (lambda r: r.level_population(2, qudit=0), r"qudit 0 has levels 0 to 1 .*got level 2"),
        (lambda r: r.level_population(-1, qudit=1), r"qudit 1 has levels 0 to 2 .*got level -1"),
        (lambda r: r.level_population(True, qudit=1), r"got level True"),
        (lambda r: r.level_populations(qudit="q1"), r"qudit 'q1' is not in the result"),
    ],
)
def test_level_population_names_the_qudits_and_levels_it_has(call, message):
    result = _run(conditional_flip, 2, levels={0: 2, 1: 3})
    with pytest.raises(pulse.InvalidValueError, match=message):
        call(result)


@pulse.kernel
def _shaped(q0):
    d0 = get_drive_line(q0)
    drive(d0, gaussian(400, 1e-6, 80.0))


# Levels 0 and 2 in equal parts: their coherence turns at the anharmonicity.
_ZERO_TWO = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)


def test_a_step_too_coarse_for_the_anharmonicity_warns_at_the_caller():
    with pytest.warns(pulse.PulseWarning) as caught:
        coarse = _run(_shaped, levels=3, initial_state=_ZERO_TWO, dt_ns=0.5)
    (warning,) = caught
    assert issubclass(warning.category, RuntimeWarning)
    assert warning.filename == __file__
    message = str(warning.message)
    assert message.startswith(
        "simulate(dt_ns=0.5): qudit 0 has 3 levels, and in its frame the coherence of "
        "levels 0 and 2 turns at 200 MHz, so a step of 0.5 ns takes 10 steps per period"
    )
    assert "Pass dt_ns=0.25 or less (2 steps per 0.5 ns clock cycle" in message
    # The coarse step damps the coherence the warning names; the step it
    # suggests keeps it.
    assert abs(coarse.density_matrix[0, 2]) < 0.45
    fine = _run(_shaped, levels=3, initial_state=_ZERO_TWO, dt_ns=0.25)
    assert abs(fine.density_matrix[0, 2]) == pytest.approx(0.5, abs=5e-3)


@pytest.mark.parametrize(
    "options",
    [
        {"levels": 3},  # the default step
        {"levels": 3, "dt_ns": 0.25},  # 20 steps per period
        {"levels": 2, "dt_ns": 2.0},  # two levels turn together in the frame
    ],
)
def test_a_step_that_resolves_the_simulation_does_not_warn(options, recwarn):
    _run(_shaped, **options)
    assert not [w for w in recwarn if issubclass(w.category, pulse.PulseWarning)]


def test_the_check_follows_the_model_s_levels_not_a_transmon_s():
    # Evenly spaced levels (no anharmonicity) all turn with the frame.
    harmonic = pulse.quantum_system(
        name="harmonic",
        qudits={0: PhysicalQudit(0, 5.0e9, 0.0, 0.0, anharmonicity_hz=0.0)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pulse.PulseWarning)
        _run(_shaped, target=harmonic, levels=4, dt_ns=2.0)
    # With four levels, the fastest coherence is of levels 0 and 3, at 3 alpha.
    with pytest.warns(pulse.PulseWarning, match=r"levels 0 and 3 turns at 600 MHz"):
        _run(_shaped, levels=4, dt_ns=0.25)


@pytest.mark.parametrize(
    ("levels", "step"),
    # Transmon level k sits k (k - 1) / 2 alpha off the frame: at -200 MHz,
    # 4 levels spread over 600 MHz, 5 over 1.2 GHz and 6 over 2 GHz.
    [(2, 0.05), (3, 0.05), (4, 0.05), (5, 0.5 / 12), (6, 0.025)],
)
def test_the_default_step_takes_20_steps_per_period_of_the_levels_spread(levels, step):
    from cudaq_pulse.runtime.simulate import _default_step

    target = _target()
    lowered = on(target, _shaped, Q[0])
    assert _default_step(lowered, target._description, levels) == step
    assert _default_step(lowered, target._description, {0: levels}) == step


@pytest.mark.parametrize("levels", [3, {0: 3}])
def test_a_step_too_coarse_warns_at_every_call_not_only_the_first(levels):
    lowered = on(_target(), _shaped, Q[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pulse.PulseWarning)
        for _ in range(3):
            pulse.simulate(lowered, levels=levels, dt_ns=0.5)
            pulse.simulate(lowered, levels=levels, dt_ns=0.25)
    assert [str(w.message)[:20] for w in caught] == ["simulate(dt_ns=0.5):"] * 3


@pulse.kernel
def _short_shaped(q0):
    d0 = get_drive_line(q0)
    drive(d0, gaussian(80, 1e-6, 16.0))


def test_the_default_step_resolves_six_levels_where_0_05_ns_did_not(recwarn):
    zero_five = np.zeros(6)
    zero_five[0] = zero_five[5] = 1 / math.sqrt(2.0)
    options = {"levels": 6, "initial_state": zero_five}
    default = _run(_short_shaped, **options).density_matrix
    assert not [w for w in recwarn if issubclass(w.category, pulse.PulseWarning)]
    with pytest.warns(pulse.PulseWarning, match=r"levels 0 and 5 turns at 2 GHz"):
        coarse = _run(_short_shaped, dt_ns=0.05, **options).density_matrix
    fine = _run(_short_shaped, dt_ns=0.025 / 8, **options).density_matrix
    assert np.abs(default - fine).max() < np.abs(coarse - fine).max() / 4


@pytest.mark.parametrize(
    ("levels", "spread", "steps", "suggested"),
    [
        # Four levels hold one RK4 map per stretch, checked as it is built.
        (4, "levels 0 and 3 turns at 600 MHz", "1.67", "dt_ns=0.0833 or less"),
        # Five step the state itself, checked at the end of the stretch.
        (5, "levels 0 and 4 turns at 1.2 GHz", "0.833", "dt_ns=0.0385 or less"),
    ],
)
def test_a_step_that_diverges_raises_naming_the_qudit_and_a_step_that_works(
    levels, spread, steps, suggested
):
    with pytest.warns(pulse.PulseWarning), pytest.raises(pulse.ExecutionError) as caught:
        _run(_shaped, levels=levels, dt_ns=1.0)
    message = str(caught.value)
    assert message.startswith(
        "simulate(dt_ns=1): the integrator diverged: with a step of 1 ns its state grew "
        f"without bound, so there is no result. Qudit 0 has {levels} levels, and in its "
        f"frame the coherence of {spread}, and a step of 1 ns takes {steps} steps per period"
    )
    assert f"Pass {suggested}" in message
    (note,) = caught.value.__notes__
    assert note.startswith(f'simulate was called at File "{__file__}", line ')
    assert caught.value.__cause__ is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", pulse.PulseWarning)
        result = _run(_shaped, levels=levels, dt_ns=float(suggested.split("=")[1].split()[0]))
    assert np.isfinite(result.density_matrix).all()


@pulse.kernel
def _strong(q0):
    d0 = get_drive_line(q0)
    drive(d0, gaussian(40, 40.0, 8.0))


def test_a_drive_too_fast_for_the_step_raises_rather_than_returning_nan():
    # Two levels turn with the frame, so the step resolves every frequency
    # of the qudit; a 40 rad/ns drive is what diverges.
    with pytest.raises(pulse.ExecutionError) as caught:
        _run(_strong, dt_ns=0.5)
    assert str(caught.value) == (
        "simulate(dt_ns=0.5): the integrator diverged: with a step of 0.5 ns its state grew "
        "without bound, so there is no result. The step resolves every frequency of the "
        "qudits' levels and couplings, so a drive or decay rate is too fast for it: pass a "
        "smaller dt_ns, such as dt_ns=0.05."
    )
    assert np.isfinite(_run(_strong, dt_ns=0.005).density_matrix).all()


def test_a_gpu_result_that_is_not_finite_raises(monkeypatch):
    from cudaq_pulse.runtime import simulate as simulate_module

    backend = simulate_module._GpuBackend

    def init(self, module, dims, dt_ns, base, lock, integrator):
        simulate_module._Backend.__init__(self, module, dims, dt_ns, base, lock)
        self.dissipative = True

    monkeypatch.setattr(backend, "__init__", init)
    monkeypatch.setattr(
        backend, "_execute", lambda self, text, states, mixed: [s * np.nan for s in states]
    )
    with pytest.raises(pulse.ExecutionError, match=r"^simulate\(\): the integrator diverged"):
        _run(_shaped, levels=3, backend="gpu")


def test_a_step_too_coarse_for_coupled_qudits_names_their_detuning():
    pair = pulse.quantum_system(
        name="pair",
        qudits={
            q: PhysicalQudit(q, 5.0e9 + 2.0e9 * q, 0.0, 0.0, anharmonicity_hz=-200.0e6)
            for q in range(2)
        },
        couplings=[Coupling((0, 1), 5.0e6)],
    )
    with pytest.warns(
        pulse.PulseWarning,
        match=r"the exchange coupling of qudits 0 and 1 turns at their detuning, 2 GHz, "
        r"so a step of 0\.05 ns takes 10 steps per period",
    ):
        _run(conditional_flip, 2, target=pair, dt_ns=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pulse.PulseWarning)
        _run(conditional_flip, 2, target=pair, dt_ns=0.025)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_the_default_end_of_a_loop_kept_as_a_loop_is_that_of_its_events(schedule, monkeypatch):
    """Without ``t_end`` a simulation runs to the latest end of the
    program's events. A long loop kept as a loop gives it from its last
    iteration, without listing an event per iteration: the same end, and
    the same state, as listing them gives."""
    from cudaq_pulse._native._cudaq_pulse_native import PulseModule
    from cudaq_pulse.runtime import simulate as simulate_module

    @pulse.kernel
    def rounds(q0, q1):
        d = get_drive_line(q0)
        e = get_drive_line(q1)
        drive(d, square(8, 0.1))
        for _ in range(4):
            drive(d, square(8, 0.05))
            wait(d, 6)
        drive(d, square(4, 0.2))
        # The other line's loop ends after the first line's last pulse.
        for _ in range(5):
            wait(e, 10)
            drive(e, square(8, 0.05))

    lowered = pulse.compile(rounds, schedule=schedule, unroll_threshold=0).lower(Q[0], Q[1])
    assert "scf.for" in lowered.mlir()
    end = float(lowered.duration)
    event_tree = PulseModule.event_tree
    made: list[object] = []
    forms: list[type] = []

    def listing(self, limit):
        events = event_tree(self, limit)
        forms.append(type(events))
        return events

    def run():
        return pulse.simulate(lowered, levels=2, dt_ns=0.5, seed=0)

    monkeypatch.setattr(PulseModule, "event_tree", listing)
    monkeypatch.setattr(
        simulate_module, "ScheduledEvent", lambda *event: made.append(event) or event
    )
    listed = run()
    assert forms == [list]
    monkeypatch.setattr(simulate_module, "_LISTED_EVENTS", 3)
    rolled = run()
    assert forms == [list, tuple]
    # Neither builds an event to read its end.
    assert made == []
    assert listed.times_ns == rolled.times_ns == (0.0, end)
    assert np.array_equal(listed.density_matrix, rolled.density_matrix)
    given = pulse.simulate(lowered, levels=2, dt_ns=0.5, seed=0, t_end=end)
    assert np.array_equal(given.density_matrix, rolled.density_matrix)


def test_the_latest_timed_end_of_a_schedule_is_that_of_its_timed_events():
    from cudaq_pulse._loops import Schedule
    from cudaq_pulse.compiler import ScheduledEvent

    nodes = [
        ("drive", 0, 0, 5, 8, []),
        ("wait", 0, 0, None, None, []),
        (
            3,
            [
                ("wait", 0, 0, None, 4, []),
                ("drive", 0, 0, 20, 4, [10]),
                # Later in each outer iteration, earlier in each inner one.
                (2, [("drive", 1, 1, 30, 6, [10, -3]), ("readout", 1, 1, 33, 2, [12, 5])]),
                # No iteration: none of its events is one.
                (0, [("drive", 0, 0, 10**6, 1, [10, 1])]),
            ],
        ),
        ("answer", 2, -1, 40, 0, []),
    ]
    schedule = Schedule(nodes, ScheduledEvent)
    ends = [
        e.start_cycles + e.duration_cycles
        for e in schedule
        if e.start_cycles is not None and e.duration_cycles is not None
    ]
    assert len(schedule) == 2 + 3 * (2 + 2 * 2) + 1 and len(ends) == len(schedule) - 4
    assert schedule.latest_end(timed_only=True) == max(ends) == 33 + 2 * 12 + 5 + 2
    assert schedule.latest_end() is None
    assert Schedule([("wait", 0, 0, None, None, [])], ScheduledEvent).latest_end(True) == 0
