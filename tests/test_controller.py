# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exporting a compiled kernel to a controller, and checking its limits."""

from __future__ import annotations

import json
import math

import cudaq_pulse as pulse
import cudaq_pulse.controller as controller_module
import numpy as np
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.controller import FORMAT, ControllerLimitError, check, sample_waveform, shape_key
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q, ladder_target
from tests._work import native_seconds_each
from tests.classical.fake_cudaq_qec import qec
from tests.property_settings import examples


@pulse.kernel
def drive_and_branch(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    d1 = get_drive_line(r)
    drive(d, gaussian(16, 0.5, 4.0))
    shift_phase(d.tone, 0.5)
    drive(d, drag(16, 0.25, 4.0, 0.3))
    sync(d, ro)
    m = readout(ro, square(100, 0.05))
    sync(ro, d1)
    if m:
        drive(d1, wf_scale(square(20, 0.3), -0.5))
    else:
        drive(d1, gaussian(20, 0.1, 5.0))


def _program(kernel=drive_and_branch, schedule="asap", machine=None, **kwargs):
    compiled = pulse.compile(kernel, schedule=schedule, control_system=machine)
    return compiled.lower(Q[0], Q[1]).export_to_controller(**kwargs)


def test_export_lists_every_timed_operation_with_its_waveform_and_tone():
    program = _program()
    assert program["format"] == FORMAT == "cudaq-pulse.controller/1"
    assert json.loads(json.dumps(program)) == program
    assert program["lines"] == [
        {"line": 0, "role": "drive", "qudit": 0, "native_frequency_hz": 5.0e9},
        {"line": 1, "role": "readout", "qudit": 0, "native_frequency_hz": 5.0e9},
        {"line": 2, "role": "drive", "qudit": 1, "native_frequency_hz": 5.1e9},
    ]
    kinds = [(e["kind"], e["line"], e["start_cycles"], e["end_cycles"]) for e in program["events"]]
    assert kinds == [
        ("drive", 0, 0, 16),
        ("drive", 0, 16, 32),
        ("sync", None, 32, 32),
        ("readout", 1, 32, 132),
        ("sync", None, 132, 132),
        ("drive", 2, 132, 152),
        ("drive", 2, 132, 152),
    ]
    assert [e["index"] for e in program["events"]] == list(range(7))
    first, second, _, readout, _, then, _ = program["events"]
    assert first["waveform"] == {
        "shape": "gaussian",
        "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 4.0},
    }
    assert first["duration_cycles"] == 16 and first["iteration"] == []
    assert second["waveform"]["shape"] == "drag"
    assert second["waveform"]["params"]["beta"] == pytest.approx(0.3)
    # The frame change before a drive is the frame it plays in.
    assert first["phase_rad"] == 0.0
    assert second["phase_rad"] == pytest.approx(0.5)
    assert first["frequency_hz"] == pytest.approx(5.0e9)
    assert then["frequency_hz"] == pytest.approx(5.1e9)
    assert readout["amplitude"] == {"re": 0.05, "im": 0.0}
    assert then["waveform"]["shape"] == "scale"
    assert then["amplitude"]["re"] == pytest.approx(-0.15)


def test_export_marks_the_branch_and_the_measurement_it_depends_on():
    events = _program()["events"]
    readout, then, otherwise = events[3], events[5], events[6]
    assert readout["conditions"] == []
    assert then["conditions"] == [
        {"branch": 0, "arm": True, "readouts": [3], "answers": [], "test": "is_set"}
    ]
    assert otherwise["conditions"] == [
        {"branch": 0, "arm": False, "readouts": [3], "answers": [], "test": "is_set"}
    ]


def test_export_names_the_lines_a_sync_joins():
    events = _program()["events"]
    assert events[2]["lines"] == [0, 1] and events[4]["lines"] == [1, 2]
    assert events[2]["qudit"] is None and events[2]["duration_cycles"] == 0


@pulse.kernel
def staggered(q, r):
    d = get_drive_line(q)
    d1 = get_drive_line(r)
    ro = get_readout_line(q)
    for _ in range(3):
        drive(d, gaussian(24, 0.5, 6.0))
        drive(d1, square(8, 0.1))
        sync(d, d1, ro)
        readout(ro, square(40, 0.05))
        drive(d1, square(12, 0.1))
        sync(d, d1, ro)


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_a_sync_is_a_barrier_between_the_events_of_its_lines(schedule):
    events = _program(staggered, schedule=schedule)["events"]
    syncs = [e for e in events if e["kind"] == "sync"]
    assert len(syncs) == 6
    assert [s["iteration"] for s in syncs] == [[0], [0], [1], [1], [2], [2]]
    for barrier in syncs:
        joined = set(barrier["lines"])
        assert joined == {0, 1, 2}
        for event in events:
            if event["kind"] == "sync" or event["line"] not in joined:
                continue
            if event["index"] < barrier["index"]:
                assert event["end_cycles"] <= barrier["start_cycles"], (barrier, event)
            else:
                assert event["start_cycles"] >= barrier["start_cycles"], (barrier, event)


@pulse.kernel
def detuned(q, r):
    d = get_drive_line(q)
    shift_frequency(d.tone, 2.0e6)
    drive(d, square(16, 0.1))
    wait(d, 100)
    drive(d, square(16, 0.1))
    shift_phase(d.tone, 0.25)
    drive(d, square(16, 0.1))


def test_a_detuned_tone_gains_phase_between_its_drives():
    # fold_phases folds the phase shift into the last drive.
    lowered = pulse.compile(detuned, schedule="asap").lower(Q[0], Q[1])
    assert "frame_phase_offset" in lowered.mlir()
    program = lowered.export_to_controller()
    clock = program["clock_ghz"]
    drives = [e for e in program["events"] if e["kind"] == "drive"]
    assert [e["start_cycles"] for e in drives] == [0, 116, 132]
    for event, extra in zip(drives, (0.0, 0.0, 0.25), strict=True):
        assert event["frequency_hz"] == pytest.approx(5.0e9 + 2.0e6)
        expected = 2 * math.pi * 2.0e6 * event["start_cycles"] / clock * 1e-9 + extra
        assert event["phase_rad"] == pytest.approx(expected)


@pulse.kernel
def tone_in_one_arm(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        drive(d, square(20, 0.3))
    drive(d, gaussian(20, 0.2, 5.0))


@pulse.kernel
def phase_in_one_arm(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    drive(d, gaussian(20, 0.2, 5.0))
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        shift_phase(d.tone, 0.5)
        drive(d, square(20, 0.3))
    drive(d, gaussian(20, 0.2, 5.0))


def test_a_frame_only_one_arm_touches_is_unchanged_in_the_other():
    last = _program(tone_in_one_arm)["events"][-1]
    assert last["frequency_hz"] == pytest.approx(5.0e9)
    assert last["phase_rad"] == 0.0
    last = _program(phase_in_one_arm)["events"][-1]
    assert last["frequency_hz"] is None and last["phase_rad"] is None


@pulse.kernel
def idle_readout_line(q, r):
    ro = get_readout_line(q)
    wait(ro, 10)


def test_a_line_is_listed_with_its_role_even_when_it_only_waits():
    program = _program(idle_readout_line)
    assert program["lines"] == [
        {"line": 0, "role": "readout", "qudit": 0, "native_frequency_hz": 5.0e9}
    ]
    (event,) = program["events"]
    assert event["kind"] == "wait" and event["end_cycles"] == 10
    assert "waveform" not in event


@pulse.kernel
def other_tones(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(16, 0.1), tone=d.tones[(1, 2)])
    shift_frequency(d.tone, 2.0e6)
    drive(d, square(16, 0.1))
    sync(d, r)
    readout(r, square(100, 0.05))


def test_each_line_gives_the_frequency_of_its_own_tone():
    qs = pulse.quantum_system("transmon_surface17", layout=[0])
    physical = qs.physical_qudits[qs.layout[0]]
    assert physical.readout is not None
    lowered = pulse.compile(other_tones, quantum_system=qs).lower(qs.qudits[0])
    flat = lowered.export_to_controller()
    native = {line["role"]: line["native_frequency_hz"] for line in flat["lines"]}
    # The drive line's tone is at the qudit's dressed frequency (its couplings).
    assert native == {
        "drive": qs.qudits[0].dressed_frequency_hz,
        "readout": physical.readout.frequency_hz,
    }
    # No drive plays the line's own tone at its own frequency; the line gives it.
    played = [e["frequency_hz"] for e in flat["events"] if e["kind"] == "drive"]
    assert native["drive"] not in played
    assert played[1] == pytest.approx(native["drive"] + 2.0e6)
    structured = lowered.export_to_controller(version=2)
    assert [line["native_frequencies_hz"] for line in structured["lines"]] == [
        [native[line["role"]]] for line in structured["lines"]
    ]
    assert controller_module.unroll(structured)["lines"] == flat["lines"]


def test_export_samples_envelopes_at_a_multiple_of_the_clock():
    program = _program(sample_rate_ghz=4 * pulse.quantum_system().clock_ghz)
    first = program["events"][0]
    samples = np.asarray(first["samples"]["re"]) + 1j * np.asarray(first["samples"]["im"])
    assert len(samples) == 16 * 4
    expected = sample_waveform(first["waveform"], 16, 4)
    np.testing.assert_allclose(samples, expected)
    t = np.arange(64) / 4 - 8
    np.testing.assert_allclose(samples.real, 0.5 * np.exp(-0.5 * t**2 / 16.0))
    assert "samples" not in program["events"][2]
    with pytest.raises(ValueError, match="whole multiple"):
        _program(sample_rate_ghz=1.5 * pulse.quantum_system().clock_ghz)


def test_drag_samples_carry_the_derivative_in_quadrature():
    node = {
        "shape": "drag",
        "params": {"duration": 10, "amplitude": 1.0, "sigma": 2.0, "beta": 0.5},
    }
    samples = sample_waveform(node, 10)
    t = np.arange(10) - 5.0
    g = np.exp(-0.5 * t**2 / 4.0)
    np.testing.assert_allclose(samples, g - 1j * 0.5 * t * g / 4.0)


def test_gaussian_square_samples_are_flat_between_their_edges():
    node = {
        "shape": "gaussian_square",
        "params": {"duration": 40, "amplitude": 0.5, "sigma": 3.0, "risefall": 10},
    }
    samples = sample_waveform(node, 40).real
    np.testing.assert_allclose(samples[10:30], 0.5)
    np.testing.assert_allclose(samples[:10], 0.5 * np.exp(-0.5 * (np.arange(10) - 10) ** 2 / 9))
    np.testing.assert_allclose(samples[30:], 0.5 * np.exp(-0.5 * np.arange(10) ** 2 / 9))


def ramp(t):
    return t / 8


@pulse.kernel
def sampled_envelopes(q, r):
    d = get_drive_line(q)
    drive(d, custom(8, ramp))
    drive(d, custom_samples([0.0, 0.5, 1.0, 0.5]))


def test_custom_envelopes_are_exported_as_their_samples():
    program = _program(sampled_envelopes, sample_rate_ghz=pulse.quantum_system().clock_ghz)
    traced, given = program["events"]
    expected = [(k + 0.5) / 8 for k in range(8)]
    assert traced["waveform"]["shape"] == "custom_samples"
    assert traced["waveform"]["samples"] == pytest.approx(expected)
    assert traced["samples"]["re"] == pytest.approx(expected)
    assert given["waveform"] == {"shape": "custom_samples", "samples": [0.0, 0.5, 1.0, 0.5]}
    assert traced["amplitude"]["re"] == pytest.approx(expected[-1])


def test_a_waveform_a_python_function_plays_cannot_be_sampled():
    node = {"shape": "custom", "name": "ramp_envelope", "params": {"duration": 32.0}}
    with pytest.raises(ValueError, match="cannot sample a 'custom' waveform"):
        sample_waveform(node, 32)


def test_shape_key_ignores_overall_amplitude_only():
    a = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 4.0}}
    b = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.1, "sigma": 4.0}}
    c = {"shape": "gaussian", "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 3.0}}
    scaled = {"shape": "scale", "params": {"factor": -2.0}, "of": [b]}
    assert shape_key(a) == shape_key(b) == shape_key(scaled)
    assert shape_key(a) != shape_key(c)
    one = {"shape": "custom", "name": "one", "params": {"duration": 8.0}}
    two = {"shape": "custom", "name": "two", "params": {"duration": 8.0}}
    assert shape_key(one) != shape_key(two)


def test_shape_key_ignores_an_amplitude_not_known_while_compiling():
    known = {
        "shape": "drag",
        "params": {"duration": 16.0, "amplitude": 0.5, "sigma": 4.0, "beta": 1.0},
    }
    unknown = {"shape": "drag", "params": {**known["params"], "amplitude": None}}
    assert shape_key(unknown) == shape_key(known)
    square = {
        "shape": "square",
        "params": {"duration": 8.0, "amplitude": None, "amplitude_imag": None},
    }
    assert shape_key(square) is not None
    sigma = {"shape": "drag", "params": {**known["params"], "sigma": None}}
    assert shape_key(sigma) is None


@pulse.kernel
def two_amplitudes(q, a: float, b: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, a, 10.0))
    drive(d, gaussian(40, b, 10.0))


def test_waveforms_of_one_shape_at_float_amplitudes_share_waveform_memory():
    machine = ControlSystem(
        controllers=[
            Controller(waveform_memory_samples=60, sample_rate_ghz=pulse.quantum_system().clock_ghz)
        ]
    )
    lowered = pulse.compile(two_amplitudes, control_system=machine).lower(Q[0], 0.5, 0.25)
    assert check(lowered.generic) == []
    assert len(lowered.export_to_controller()["events"]) == 2


@pulse.kernel
def two_readouts(q, r):
    ra = get_readout_line(q)
    rb = get_readout_line(r)
    readout(ra, square(40, 0.05))
    readout(rb, square(40, 0.05))


@pulse.kernel
def close_drives(q, r):
    d = get_drive_line(q)
    drive(d, square(5, 0.1))
    drive(d, gaussian(8, 0.1, 2.0))


def test_limits_report_instruction_spacing_detectors_and_feedback():
    limits = Controller(instruction_cycles=4, min_idle_cycles=4)
    found = check(pulse.compile(close_drives).lower(Q[0], Q[1]), limits)
    assert len(found) == 1 and "starts 5 cycles after" in found[0]
    limits = Controller(detectors={0: 0, 1: 0}, detector_recovery_cycles=30)
    found = check(pulse.compile(two_readouts, schedule="asap").lower(Q[0], Q[1]), limits)
    assert len(found) == 1 and "detector 0" in found[0]
    # Serialized by the scheduler, the readouts fit.
    machine = ControlSystem(max_concurrent_readouts=1, line_switch_penalty_cycles=30)
    lowered = pulse.compile(two_readouts, schedule="rcp", control_system=machine).lower(Q[0], Q[1])
    assert check(lowered, limits) == []
    limits = Controller(feedback_latency_cycles=50)
    lowered = pulse.compile(drive_and_branch, schedule="asap").lower(Q[0], Q[1])
    assert len(check(lowered, limits)) == 2
    machine = ControlSystem(controllers=[limits])
    lowered = pulse.compile(drive_and_branch, schedule="rcp", control_system=machine).lower(
        Q[0], Q[1]
    )
    assert check(lowered, limits) == []


def _reset_each(n):
    @pulse.kernel
    def reset_each(qudits):
        lines = [get_drive_line(q) for q in qudits]
        probes = [get_readout_line(q) for q in qudits]
        bits = []
        for k in range(n):
            sync(lines[k], probes[k])
            bits.append(readout(probes[k], square(40, 0.05)))
        for k in range(n):
            if bits[k]:
                drive(lines[k], square(16, 0.1))

    return reset_each


@pytest.mark.slow
def test_a_branch_on_each_of_many_qudits_is_walked_in_linear_time():
    # The controller check and the export walk the program: at a branch they
    # undo and merge what its arms change, not every tone's frame and every
    # line's time. The time, native, is of whole compiles and exports.
    machine = ControlSystem(controllers=[Controller(feedback_latency_cycles=50)])
    wide = ladder_target(512).qudits
    runs = {}
    for n in (64, 512):
        kernel = _reset_each(n)

        def run(kernel=kernel, n=n):
            compiled = pulse.compile(kernel, schedule="rcp", control_system=machine)
            compiled.lower(wide[0:n]).export_to_controller()

        runs[n] = run
    timings = native_seconds_each(
        runs, methods=("controller_events", "controller_program"), repeats=3
    )
    # 8x the branches: 8x the time when linear, 64x when quadratic.
    assert timings[512] < 24 * timings[64], timings


def test_limits_report_clock_tones_memory_and_size():
    lowered = pulse.compile(drive_and_branch, schedule="asap").lower(Q[0], Q[1])
    clock = lowered.quantum_system.clock_ghz
    # Every operation starts and lasts whole cycles of a controller at half
    # the clock; at an eighth, the readout and the two drives after it do not.
    assert check(lowered, Controller(clock_ghz=clock / 2)) == []
    found = check(lowered, Controller(clock_ghz=clock / 8))
    assert len(found) == 3 and all("moves in timesteps of 8 cycles" in f for f in found)
    assert check(lowered, Controller(max_frequencies_per_line=1)) == []
    # Line 0 plays two shapes of 16 cycles, the readout line one of 100,
    # line 2 two of 20.
    found = check(lowered, Controller(sample_rate_ghz=4.0, waveform_memory_samples=64))
    assert [f.split(":")[0] for f in found] == ["line 1", "line 2"]
    found = check(lowered, Controller(max_pulses=4))
    assert "5 drives and readouts" in found[0] and "at most 4" in found[0]
    # Both arms of a branch play on one line at one time without conflict.
    assert check(lowered, Controller(instruction_cycles=4)) == []


def test_a_sample_rate_sets_the_samples_a_waveform_takes():
    lowered = pulse.compile(drive_and_branch, schedule="asap").lower(Q[0], Q[1])
    clock = lowered.quantum_system.clock_ghz
    # At one and a half samples per cycle, the readout's 100 cycles take 150
    # samples and line 2's two shapes of 20 cycles 30 each.
    found = check(lowered, Controller(sample_rate_ghz=1.5 * clock, waveform_memory_samples=60))
    assert [f.split(":")[0] for f in found] == ["line 1"]
    assert "150 samples" in found[0]
    with pytest.raises(ValueError, match="no whole-cycle ratio"):
        check(lowered, Controller(sample_rate_ghz=clock / 10_007, waveform_memory_samples=60))


@pytest.mark.parametrize("rate", [0, -1.0, float("inf"), float("nan"), True, "2"])
def test_a_sample_rate_is_a_positive_number(rate):
    with pytest.raises(pulse.InvalidValueError, match="sample_rate_ghz must be a positive number"):
        Controller(sample_rate_ghz=rate)


def test_samples_per_cycles_is_no_longer_a_limit():
    with pytest.raises(TypeError, match="samples_per_cycles"):
        Controller(samples_per_cycles=2)  # type: ignore[call-arg]


def test_limits_of_the_controller_fail_the_compilation():
    machine = ControlSystem(controllers=[Controller(instruction_cycles=4, min_idle_cycles=4)])
    with pytest.raises(ControllerLimitError, match="starts 5 cycles after") as caught:
        pulse.compile(close_drives, control_system=machine).lower(Q[0], Q[1])
    assert len(caught.value.violations) == 1
    assert isinstance(caught.value, pulse.CompilationError)
    with pytest.raises(TypeError, match="Controller"):
        ControlSystem(controllers=[{"instruction_cycles": 4}])  # type: ignore[arg-type]


@pulse.kernel
def strong_drives(q, amplitude: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(16, 1.5))
    drive(d, gaussian(16, amplitude, 4.0))
    drive(d, wf_scale(drag(16, 0.5, 4.0, 16.0), 1.5))
    readout(r, square(32, 0.1))


def test_only_the_controller_limits_the_amplitude():
    lowered = pulse.compile(strong_drives).lower(Q[0], 3.0)
    assert lowered.export_to_controller()["events"][0]["amplitude"]["re"] == 1.5
    found = check(lowered, Controller(max_amplitude=1.0))
    assert found[0] == (
        "line 0 (drive of qudit 0): the drive at 0 peaks at amplitude 1.5; the controller "
        "plays at most 1 (Controller.max_amplitude)"
    )
    # The quadrature of the DRAG pulse takes its peak above 0.75.
    assert [f.split(": the ")[1].split(" peaks")[0] for f in found] == [
        "drive at 0",
        "drive at 16",
        "drive at 32",
    ]
    assert check(lowered, Controller(max_amplitude=3.0)) == []
    assert "max_amplitude" not in Controller(clock_ghz=1.0).as_dict()
    machine = ControlSystem(controllers=[Controller(max_amplitude=2.0)])
    compiled = pulse.compile(strong_drives, control_system=machine)
    assert compiled.lower(Q[0], 0.5).export_to_controller()["limits"]["max_amplitude"] == 2.0
    with pytest.raises(ControllerLimitError, match=r"peaks at amplitude 3; .* at most 2 \("):
        compiled.lower(Q[0], 3.0).export_to_controller()
    for wrong in (0, -1.0, math.inf, math.nan, True, "1"):
        with pytest.raises(pulse.InvalidValueError, match="max_amplitude must be a positive"):
            Controller(max_amplitude=wrong)  # type: ignore[arg-type]


@pulse.kernel
def short_pulses(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    drive(d, square(8, 0.1))
    drive(d, gaussian(16, 0.3, 4.0))
    drive(get_drive_line(r), square(8, 0.2))
    sync(d, ro)
    readout(ro, square(100, 0.05))


def test_a_controller_may_require_a_minimum_pulse_length():
    lowered = pulse.compile(short_pulses).lower(Q[0], Q[1])
    assert check(lowered, Controller()) == check(lowered, Controller(min_pulse_cycles=8)) == []
    assert check(lowered, Controller(min_pulse_cycles=16)) == [
        "line 0 (drive of qudit 0): the drive at 0 lasts 8 cycles; the controller plays none "
        "shorter than 16 (Controller.min_pulse_cycles)",
        "line 2 (drive of qudit 1): the drive at 116 lasts 8 cycles; the controller plays none "
        "shorter than 16 (Controller.min_pulse_cycles)",
    ]
    found = check(lowered, Controller(min_pulse_cycles=101))
    assert [f.split(": the ")[1].split(" lasts")[0] for f in found] == [
        "drive at 0",
        "drive at 8",
        "readout at 24",
        "drive at 116",
    ]
    # In the controller's own cycles: 8 cycles of the 2 GHz program clock are 4 at 1 GHz.
    own = check(lowered, Controller(clock_ghz=1.0, min_pulse_cycles=8))
    assert [f.split(" lasts ")[1].split(";")[0] for f in own] == ["4 cycles", "4 cycles"]
    machine = ControlSystem(controllers=[Controller(min_pulse_cycles=16)])
    with pytest.raises(ControllerLimitError, match=r"lasts 8 cycles; .* shorter than 16 \("):
        pulse.compile(short_pulses, control_system=machine).lower(Q[0], Q[1])
    machine = ControlSystem(controllers=[Controller(min_pulse_cycles=8)])
    program = pulse.compile(short_pulses, control_system=machine).lower(Q[0], Q[1])
    assert program.export_to_controller()["limits"]["min_pulse_cycles"] == 8
    assert program.export_to_controller(version=2)["controller"]["min_pulse_cycles"] == 8
    assert "min_pulse_cycles" not in Controller().as_dict()
    for wrong in (-1, 1.5, True, "8"):
        with pytest.raises(pulse.InvalidValueError, match="min_pulse_cycles must be an int"):
            Controller(min_pulse_cycles=wrong)  # type: ignore[arg-type]


def test_each_controller_of_a_system_checks_its_own_minimum_pulse_length():
    def system(first, second):
        return ControlSystem(
            controllers=[
                Controller(drive_lines=[0], readout_lines=[0], min_pulse_cycles=first),
                Controller(drive_lines=[1], readout_lines=[1], min_pulse_cycles=second),
            ]
        )

    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(short_pulses, control_system=system(16, 0)).lower(Q[0], Q[1])
    (violation,) = caught.value.violations
    assert "(drive of qudit 0): the drive at 0 lasts 8 cycles" in violation
    pulse.compile(short_pulses, control_system=system(0, 8)).lower(Q[0], Q[1])


def test_a_controller_may_cap_the_length_of_a_pulse():
    lowered = pulse.compile(short_pulses).lower(Q[0], Q[1])
    assert check(lowered, Controller(max_pulse_cycles=100)) == []
    assert check(lowered, Controller(max_pulse_cycles=99)) == [
        "line 1 (readout of qudit 0): the readout at 24 lasts 100 cycles; the controller "
        "plays none longer than 99 (Controller.max_pulse_cycles)",
    ]
    found = check(lowered, Controller(max_pulse_cycles=8))
    assert [f.split(": the ")[1].split(" lasts")[0] for f in found] == [
        "drive at 8",
        "readout at 24",
    ]
    # Both bounds: each pulse is reported once, against the bound it breaks.
    both = check(lowered, Controller(min_pulse_cycles=16, max_pulse_cycles=16))
    assert [f.split(" lasts ")[1].split(" (")[0] for f in both] == [
        "8 cycles; the controller plays none shorter than 16",
        "100 cycles; the controller plays none longer than 16",
        "8 cycles; the controller plays none shorter than 16",
    ]
    # In the controller's own cycles: the 100-cycle readout of the 2 GHz program
    # clock lasts 50 cycles at 1 GHz.
    assert check(lowered, Controller(clock_ghz=1.0, max_pulse_cycles=50)) == []
    (own,) = check(lowered, Controller(clock_ghz=1.0, max_pulse_cycles=49))
    assert "the readout at 12 lasts 50 cycles" in own
    machine = ControlSystem(controllers=[Controller(max_pulse_cycles=64)])
    with pytest.raises(ControllerLimitError, match=r"lasts 100 cycles; .* longer than 64 \("):
        pulse.compile(short_pulses, control_system=machine).lower(Q[0], Q[1])
    machine = ControlSystem(controllers=[Controller(max_pulse_cycles=128)])
    program = pulse.compile(short_pulses, control_system=machine).lower(Q[0], Q[1])
    assert program.export_to_controller()["limits"]["max_pulse_cycles"] == 128
    assert program.export_to_controller(version=2)["controller"]["max_pulse_cycles"] == 128
    assert "max_pulse_cycles" not in Controller().as_dict()
    for wrong in (0, -1, 1.5, True, "8"):
        with pytest.raises(
            pulse.InvalidValueError, match="max_pulse_cycles must be None or an int"
        ):
            Controller(max_pulse_cycles=wrong)  # type: ignore[arg-type]
    with pytest.raises(pulse.InvalidValueError, match=r"at least min_pulse_cycles \(16\), got 8"):
        Controller(min_pulse_cycles=16, max_pulse_cycles=8)
    Controller(min_pulse_cycles=16, max_pulse_cycles=16)


def test_each_controller_of_a_system_checks_its_own_maximum_pulse_length():
    def system(first, second):
        return ControlSystem(
            controllers=[
                Controller(drive_lines=[0], readout_lines=[0], max_pulse_cycles=first),
                Controller(drive_lines=[1], readout_lines=[1], max_pulse_cycles=second),
            ]
        )

    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(short_pulses, control_system=system(64, None)).lower(Q[0], Q[1])
    (violation,) = caught.value.violations
    assert "(readout of qudit 0): the readout at 24 lasts 100 cycles" in violation
    pulse.compile(short_pulses, control_system=system(None, 8)).lower(Q[0], Q[1])


def _after_a_loop(n: int) -> pulse.Kernel:
    @pulse.kernel
    def after(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            drive(line, square(8, 0.1))
        if m:
            drive(line, square(8, 0.1))

    return after


# The branch after the loop tests the readout of its last iteration, a value
# the export's vocabulary has no term for: checking the limits needs none.
@pytest.mark.parametrize(
    "limits",
    [
        Controller(max_amplitude=0.05),
        Controller(cycles_per_timestep=4, max_amplitude=0.05),
        Controller(clock_ghz=1.0, max_amplitude=0.05),
    ],
    ids=["program_clock", "timesteps", "own_clock"],
)
def test_a_branch_on_a_readout_carried_out_of_a_loop_is_checked_as_written_out(limits):
    rolled = pulse.compile(_after_a_loop(65), unroll_threshold=0, passes=()).lower(Q[0], Q[1])
    unrolled = pulse.compile(_after_a_loop(65), unroll_threshold=10**6, passes=()).lower(Q[0], Q[1])
    found = check(rolled, limits)
    assert len(found) == 66
    assert found == check(unrolled, limits)
    # An export still has to name it.
    with pytest.raises(pulse.UnsupportedError, match="'scf.for'"):
        rolled.export_to_controller()


def test_a_limit_error_lists_ten_violations_and_counts_the_rest():
    error = ControllerLimitError([f"violation {k}" for k in range(13)])
    assert "violation 9" in str(error) and "violation 10" not in str(error)
    assert "and 3 more" in str(error) and len(error.violations) == 13


def test_limits_are_part_of_the_control_system_identity():
    a = ControlSystem(controllers=[Controller(instruction_cycles=4, detectors={0: 0})])
    b = ControlSystem(controllers=[Controller(instruction_cycles=4, detectors={0: 0})])
    c = ControlSystem(controllers=[Controller(instruction_cycles=2)])
    assert a == b and hash(a) == hash(b)
    assert a != c and a != ControlSystem()
    assert Controller(detectors={0: 0, 1: 0}).as_dict()["detectors"] == {"0": 0, "1": 0}
    with pytest.raises(ValueError, match="instruction_cycles"):
        Controller(instruction_cycles=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        Controller(detector_recovery_cycles=-1)
    with pytest.raises(ValueError, match="positive or None"):
        Controller(waveform_memory_samples=0)


def test_the_limits_travel_with_the_program():
    limits = Controller(instruction_cycles=1, max_frequencies_per_line=2)
    program = _program(machine=ControlSystem(controllers=[limits]))
    assert program["limits"] == limits.as_dict()
    assert _program()["limits"] is None


@pulse.kernel
def ramsey(q, phase: float):
    d = get_drive_line(q)
    drive(d, gaussian(16, 0.25, 4.0))
    shift_phase(d.tone, phase)
    drive(d, gaussian(16, 0.25, 4.0))


def test_float_arguments_are_exported_at_their_values():
    lowered = pulse.compile(ramsey).lower(Q[0], 0.75)
    assert lowered.generic is not lowered
    with pytest.raises(ValueError, match="float parameters"):
        lowered.generic.export_to_controller()
    program = lowered.export_to_controller()
    assert program["events"][1]["phase_rad"] == pytest.approx(0.75)
    assert not math.isnan(program["events"][0]["phase_rad"])


def test_a_kernel_with_float_parameters_is_checked_when_compiled():
    machine = ControlSystem(controllers=[Controller(instruction_cycles=4, min_idle_cycles=40)])
    with pytest.raises(ControllerLimitError, match="starts 16 cycles after"):
        pulse.compile(ramsey, control_system=machine).lower(Q[0], 0.75)


@pulse.kernel
def long_train(q):
    d = get_drive_line(q)
    for _ in range(400):
        drive(d, square(8, 0.1))
        wait(d, 4)


def test_events_without_branches_pair_without_testing_arms(monkeypatch):
    # The spacing check pairs each pulse with the one before it on its line;
    # with no branches, no pair needs the test for arms of one branch, at
    # compile or at export.
    tested = []
    exclusive = controller_module._exclusive
    monkeypatch.setattr(
        controller_module, "_exclusive", lambda a, b: tested.append(1) or exclusive(a, b)
    )
    machine = ControlSystem(
        controllers=[Controller(drive_lines=[0], instruction_cycles=4, min_idle_cycles=4)]
    )
    lowered = pulse.compile(long_train, control_system=machine, unroll_threshold=10**6).lower(Q[0])
    assert len(lowered.export_to_controller()["events"]) == 800
    assert tested == []


def _scanned(events):
    """``controller._pairs`` as a scan back from each event to the first it
    can follow."""
    ordered = sorted(events, key=lambda e: (e["start_cycles"], e["index"]))
    pairs = []
    for k in range(1, len(ordered)):
        for j in range(k - 1, -1, -1):
            if not controller_module._exclusive(ordered[j], ordered[k]):
                pairs.append((ordered[j], ordered[k]))
                break
    return pairs


@given(
    st.lists(
        st.tuples(
            st.integers(0, 6),
            st.integers(0, 1),
            st.lists(st.tuples(st.integers(0, 2), st.integers(0, 1)), max_size=3),
        ),
        max_size=24,
    )
)
@settings(max_examples=examples(200))
def test_pairs_follow_the_latest_event_on_one_path(made):
    events = [
        {
            "index": index,
            "start_cycles": start,
            "iteration": [iteration],
            "conditions": [{"branch": b, "arm": arm} for b, arm in dict(conditions).items()],
        }
        for index, (start, iteration, conditions) in enumerate(made)
    ]
    found = [(a["index"], b["index"]) for a, b in controller_module._pairs(events)]
    assert found == [(a["index"], b["index"]) for a, b in _scanned(events)]


@pulse.kernel
def retuned(q, shift: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, 0.5, 10.0))
    shift_frequency(d.tone, shift)
    drive(d, gaussian(40, 0.5, 10.0))


def test_a_float_tone_shift_beyond_the_limit_fails_at_the_call_not_at_export():
    machine = ControlSystem(controllers=[Controller(max_frequencies_per_line=1)])
    compiled = pulse.compile(retuned, control_system=machine)
    with pytest.raises(ControllerLimitError, match="2 tone frequencies"):
        compiled.lower(Q[0], 50e6)
    # No shift plays one tone, which fits.
    assert len(compiled.lower(Q[0], 0.0).export_to_controller()["events"]) == 2
    # Two tones fit a line that plays two.
    two = ControlSystem(controllers=[Controller(max_frequencies_per_line=2)])
    lowered = pulse.compile(retuned, control_system=two).lower(Q[0], 50e6)
    assert len(lowered.export_to_controller()["events"]) == 2


@pulse.kernel
def two_widths(q, a: float, b: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, 0.5, a))
    drive(d, gaussian(40, 0.5, b))


def test_waveforms_of_float_widths_take_memory_by_their_values():
    machine = ControlSystem(
        controllers=[
            Controller(waveform_memory_samples=60, sample_rate_ghz=pulse.quantum_system().clock_ghz)
        ]
    )
    compiled = pulse.compile(two_widths, control_system=machine)
    # One width is one shape, stored once: it fits.
    assert len(compiled.lower(Q[0], 10.0, 10.0).export_to_controller()["events"]) == 2
    with pytest.raises(ControllerLimitError, match="80 samples"):
        compiled.lower(Q[0], 10.0, 12.0)


@pulse.classical(max_latency_ns=200)
def _delay_for(x: float) -> int:
    return 8


@pulse.kernel
def delayed(q, x: float):
    d = get_drive_line(q)
    t = _delay_for.submit(x).result()
    drive(d, square(16, 0.1))
    wait(d, t, bounds=(4, 64))


@pytest.mark.parametrize(
    ("limits", "version"),
    [({"max_amplitude": 1.0}, 2), ({"cycles_per_timestep": 2}, 1), ({"sample_rate_ghz": 1.0}, 1)],
)
def test_a_float_parameter_a_call_is_given_is_checked_at_its_value(limits, version):
    machine = ControlSystem(controllers=[Controller(**limits)])
    lowered = pulse.compile(delayed, control_system=machine).lower(Q[0], 0.25)
    submitted = lowered.export_to_controller(version=version)["events"][0]
    assert submitted["operands"] == [{"const": 0.25}]


def test_a_call_is_checked_once_per_distinct_values(monkeypatch):
    checked = []
    check_call = controller_module._call_violations

    def counted(view):
        checked.append(tuple(view.arguments))
        return check_call(view)

    monkeypatch.setattr(controller_module, "_call_violations", counted)
    machine = ControlSystem(controllers=[Controller(max_frequencies_per_line=1)])
    compiled = pulse.compile(retuned, control_system=machine)
    for shift in (0.0, 0.0, 0.0, -0.0):
        compiled.lower(Q[0], shift)
    # -0.0 is a different value to the controller: its own check.
    assert checked == [(0.0,), (-0.0,)]
    # A module no float value can push past a limit is not checked per call.
    lowered = pulse.compile(ramsey, control_system=machine).lower(Q[0], 0.75)
    assert type(lowered.generic) is pulse.ScheduledKernel


def test_a_checked_call_names_a_value_that_is_not_a_finite_number():
    machine = ControlSystem(controllers=[Controller(max_frequencies_per_line=1)])
    compiled = pulse.compile(retuned, control_system=machine)
    with pytest.raises(pulse.InvalidValueError, match="argument 'shift' is nan"):
        compiled.lower(Q[0], float("nan"))
    generic = compiled.lower(Q[0], 0.0).generic
    with pytest.raises(pulse.InvalidValueError, match="argument 'shift' is inf"):
        generic.bind([float("inf")])
    with pytest.raises(pulse.InvalidTypeError, match="'shift'"):
        generic.bind(["fast"])
    with pytest.raises(pulse.InvalidTypeError, match="missing the value of 'shift'"):
        generic.bind([])


@pulse.kernel
def waited(q, tau: float):
    d = get_drive_line(q)
    drive(d, square(16, 0.1))
    wait(d, tau)
    drive(d, square(16, 0.1))


def test_a_float_wait_length_is_checked_against_the_limits_at_the_call():
    machine = ControlSystem(controllers=[Controller(instruction_cycles=4, min_idle_cycles=40)])
    compiled = pulse.compile(waited, control_system=machine)
    with pytest.raises(ControllerLimitError, match="starts 20 cycles after"):
        compiled.lower(Q[0], 4.0)
    assert len(compiled.lower(Q[0], 40.0).export_to_controller()["events"]) == 3


@given(
    shift=st.sampled_from([0.0, 25e6, 50e6]),
    widths=st.tuples(st.sampled_from([8.0, 10.0]), st.sampled_from([8.0, 10.0])),
    tones=st.integers(1, 3),
    samples=st.sampled_from([40, 60, 80, 200]),
)
@settings(max_examples=examples(40))
def test_a_call_that_lowers_exports(shift, widths, tones, samples):
    @pulse.kernel
    def program(q, shift: float, a: float, b: float):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.5, a))
        shift_frequency(d.tone, shift)
        drive(d, gaussian(40, 0.25, b))
        shift_frequency(d.tone, 25e6)
        drive(d, gaussian(40, 0.5, 10.0))

    limits = Controller(max_frequencies_per_line=tones, waveform_memory_samples=samples)
    # The call fails exactly when its program, compiled free, exceeds them.
    expected = check(pulse.compile(program).lower(Q[0], shift, *widths), limits)
    compiled = pulse.compile(program, control_system=ControlSystem(controllers=[limits]))
    if expected:
        with pytest.raises(ControllerLimitError) as raised:
            compiled.lower(Q[0], shift, *widths)
        assert list(raised.value.violations) == expected
    else:
        lowered = compiled.lower(Q[0], shift, *widths)
        assert len(lowered.export_to_controller()["events"]) == 3
        assert check(lowered) == []


# ── Classical calls ──────────────────────────────────────────────────────────

enqueue_syndromes = pulse.classical(qec.enqueue_syndromes)
get_corrections = pulse.classical(qec.get_corrections, max_latency_ns=400)


@pulse.classical(max_latency_ns=400)
def check_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def decoded(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    enqueue_syndromes.submit(0, [m], 0)
    fix = get_corrections.submit(0, 1, False).result()
    sync(d, ro)
    if fix[0]:
        drive(d, square(20, 0.3))


@pulse.kernel
def timed_out(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    ok, fix = check_bit.submit([m]).result(timeout=300 * pulse.ns)
    sync(d, ro)
    if ok:
        drive(d, square(20, 0.3))


def test_export_carries_calls_with_their_measurements_and_waits():
    program = pulse.compile(decoded, schedule="asap").lower(Q[0]).export_to_controller()
    submits = [e for e in program["events"] if e["kind"] == "submit"]
    answer = next(e for e in program["events"] if e["kind"] == "answer")
    readout = next(e for e in program["events"] if e["kind"] == "readout")
    enqueue, get = submits
    assert enqueue["function"].startswith("enqueue_syndromes")
    assert enqueue["measurements"] == [readout["index"]]
    assert get["measurements"] == []
    assert answer["wait"] == "bounded" and answer["timeout_ns"] is None
    assert all(e["end_cycles"] == e["start_cycles"] + e["duration_cycles"] for e in submits)
    fix = next(e for e in program["events"] if e["conditions"])
    assert fix["conditions"][0]["answers"] == [answer["index"]]
    assert fix["conditions"][0]["test"] == "answer"
    assert {f["name"] for f in program["manifest"]["functions"]} == {
        enqueue["function"],
        get["function"],
    }
    # Links are not lines.
    assert all(line["role"] in ("drive", "readout") for line in program["lines"])


@pulse.kernel
def settled_by_float(q, wait_ns: float):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(2):
        drive(d, square(40, 0.1))
        wait(d, wait_ns * pulse.ns, bounds=(100 * pulse.ns, 200 * pulse.us))
        sync(d, ro)
        if check_bit.submit([readout(ro, square(300, 0.05))]).result()[0]:
            drive(d, square(40, 0.1))
        sync(d, ro)


@pulse.kernel
def settled_by_int(q, wait_ns: int):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(2):
        drive(d, square(40, 0.1))
        wait(d, wait_ns * pulse.ns)
        sync(d, ro)
        if check_bit.submit([readout(ro, square(300, 0.05))]).result()[0]:
            drive(d, square(40, 0.1))
        sync(d, ro)


@pulse.kernel
def settled(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(2):
        drive(d, square(40, 0.1))
        wait(d, 50 * pulse.us)
        sync(d, ro)
        if check_bit.submit([readout(ro, square(300, 0.05))]).result()[0]:
            drive(d, square(40, 0.1))
        sync(d, ro)


def _call_times(calls):
    return [(c["submit_cycles"], c["taken_cycles"]) for c in calls]


def _event_call_times(events):
    submits = [e["start_cycles"] for e in events if e["kind"] == "submit"]
    answers = [e["start_cycles"] for e in events if e["kind"] == "answer"]
    return list(zip(submits, answers, strict=True))


@pytest.mark.parametrize(
    ("kernel", "arguments"),
    [(settled_by_float, (50_000.0,)), (settled_by_int, (50_000,)), (settled, ())],
    ids=["float", "int", "literal"],
)
def test_calls_are_timed_as_the_events_after_a_wait_an_argument_sets(kernel, arguments):
    from cudaq_pulse.classical.emulation import emulate

    lowered = pulse.compile(kernel).lower(Q[0], *arguments)
    flat = lowered.export_to_controller()
    structured = lowered.export_to_controller(version=2)
    times = _event_call_times(flat["events"])
    # Each shot's readout comes after the 50 us wait: 100,000 cycles at 2 GHz.
    assert len(times) == 2 and times[0][0] > 100_000
    assert _event_call_times(controller_module.unroll(structured)["events"]) == times
    report = [(c.submit_cycles, c.taken_cycles) for c in lowered.classical_report()]
    assert report == times
    assert _call_times(flat["calls"]) == times
    assert _call_times(structured["calls"]) == times
    assert _call_times(controller_module.unroll(structured)["calls"]) == times
    emulated = lowered.export_to_emulator()
    assert _call_times(emulated["calls"]) == times
    assert _event_call_times(emulated["events"]) == times
    clock = lowered.quantum_system.clock_ghz
    traces = emulate(lowered, seed=1).calls[0]
    assert [round(t.submit_ns * clock) for t in traces] == [s for s, _ in times]


@pulse.classical(max_latency_ns=400)
def two_bits(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 2]:
    return [bits[0], not bits[0]]


@pulse.kernel
def flag_and_bits(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    ok, fix = two_bits.submit([m]).result(timeout=300 * pulse.ns)
    sync(d, e, ro)
    if ok:
        drive(d, square(20, 0.3))
    if fix[0]:
        drive(e, square(20, 0.3))
    if fix[1]:
        drive(e, square(20, 0.2))


def _answer_conditions(program):
    """(qudit driven, arm, test, result) of each condition of a drive."""
    qudit = {line["line"]: line.get("qudits", [line.get("qudit")])[0] for line in program["lines"]}
    return [
        (qudit[e["line"]], c["arm"], c["test"], c.get("result"))
        for e in program["events"]
        if e["kind"] in ("drive", "play")
        for c in e.get("conditions", ())
    ]


def test_a_branch_on_an_answer_names_the_result_it_reads():
    lowered = pulse.compile(flag_and_bits, schedule="rcp").lower(Q[0], Q[1])
    flat = lowered.export_to_controller()
    structured = lowered.export_to_controller(version=2)
    [answer] = [e for e in flat["events"] if e["kind"] == "answer"]
    assert answer["wait"] == "timeout"
    # Result 0 is whether the answer came in time, then the function's bits.
    expected = [(0, True, "answer", 0), (1, True, "answer", 1), (1, True, "answer", 2)]
    assert _answer_conditions(flat) == expected
    assert _answer_conditions(structured) == expected
    for e in flat["events"]:
        for c in e["conditions"]:
            assert c["answers"] == [answer["index"]]
            assert list(c) == ["branch", "arm", "readouts", "answers", "test", "result"]
    assert json.loads(json.dumps(structured)) == structured
    unrolled = controller_module.unroll(structured)
    assert [e["conditions"] for e in unrolled["events"] if e["conditions"]] == [
        e["conditions"] for e in flat["events"] if e["conditions"]
    ]


def test_a_branch_on_an_answer_without_a_timeout_counts_from_the_first_value():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(100, 0.05))
        fix = two_bits.submit([m]).result()
        sync(d, ro)
        if fix[1]:
            drive(d, square(20, 0.3))

    lowered = pulse.compile(k, schedule="rcp").lower(Q[0])
    for version in (1, 2):
        program = lowered.export_to_controller(version=version)
        assert _answer_conditions(program) == [(0, True, "answer", 1)]


def test_only_a_branch_on_an_answer_has_a_result():
    for program in (_program(), _program(version=2)):
        for e in program["events"]:
            assert all("result" not in c for c in e.get("conditions", ()))


def test_branches_on_the_in_time_flag_and_on_the_answer_export_apart():
    # `if fix[0] == 1:` and `if not ok:` exported the same condition, but
    # for its arm: {"answers": [3], "test": "answer"}.
    @pulse.kernel
    def k(q, r):
        s0 = readout(get_readout_line(q), square(200, 0.05))
        s1 = readout(get_readout_line(r), square(200, 0.05))
        ok, fix = check_bit.submit([s0, s1]).result(timeout=300)
        if fix[0] == 1:
            drive(get_drive_line(q), gaussian(40, 0.5, 10.0))
        if not ok:
            drive(get_drive_line(r), gaussian(40, 0.5, 10.0))

    lowered = pulse.compile(k, schedule="rcp").lower(Q[0], Q[1])
    for version in (1, 2):
        program = lowered.export_to_controller(version=version)
        assert _answer_conditions(program) == [(0, True, "answer", 1), (1, False, "answer", 0)]


def test_events_after_a_variable_wait_are_in_its_segment():
    @pulse.classical(max_latency_ns=100)
    def tau() -> float:
        return 8.0

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))
        wait(d, tau.submit().result(), bounds=(4, 40))
        drive(d, square(20, 0.1))

    program = pulse.compile(k).lower(Q[0]).export_to_controller()
    drives = [e for e in program["events"] if e["kind"] == "drive"]
    assert [e["segment"] for e in drives] == [0, 1]
    assert len(program["segments"]) == 2


def test_a_controller_without_timeouts_rejects_a_timed_wait():
    limits = Controller(supports_timeouts=False)
    assert check(pulse.compile(decoded).lower(Q[0]), limits) == []
    lowered = pulse.compile(timed_out).lower(Q[0])
    answer = next(e for e in lowered.export_to_controller()["events"] if e["kind"] == "answer")
    assert answer["wait"] == "timeout" and answer["timeout_ns"] == pytest.approx(300)
    with pytest.raises(ControllerLimitError, match="cannot stop waiting"):
        pulse.compile(timed_out, control_system=ControlSystem(controllers=[limits])).lower(Q[0])


@pulse.classical(max_latency_ns=400, latency_mode="timeout", timeout_ns=250)
def timed_bits(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 2]:
    return [bits[0], not bits[0]]


def _answer_waits(program):
    return [(e["wait"], e.get("timeout_ns")) for e in program["events"] if e["kind"] == "answer"]


def test_a_function_declared_with_a_timeout_exports_a_timeout_wait():
    # It exported "bounded" with no timeout, though result 0 is the flag.
    @pulse.kernel
    def k(q, r):
        d = get_drive_line(q)
        e = get_drive_line(r)
        ro = get_readout_line(q)
        m = readout(ro, square(100, 0.05))
        ok, fix = timed_bits.submit([m]).result()
        sync(d, e, ro)
        if ok:
            drive(d, square(20, 0.3))
        if fix[1]:
            drive(e, square(20, 0.3))

    lowered = pulse.compile(k, schedule="rcp").lower(Q[0], Q[1])
    flat = lowered.export_to_controller()
    structured = lowered.export_to_controller(version=2)
    unrolled = controller_module.unroll(structured)
    expected = [(0, True, "answer", 0), (1, True, "answer", 2)]
    for program in (flat, structured, unrolled):
        assert _answer_waits(program) == [("timeout", pytest.approx(250))]
        assert _answer_conditions(program) == expected
    assert json.loads(json.dumps(structured)) == structured


def test_a_timeout_at_the_call_outranks_the_functions():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(100, 0.05))
        ok, _ = timed_bits.submit([m]).result(timeout=150)
        sync(d, ro)
        if ok:
            drive(d, square(20, 0.3))

    lowered = pulse.compile(k, schedule="rcp").lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    for program in (
        lowered.export_to_controller(),
        structured,
        controller_module.unroll(structured),
    ):
        assert _answer_waits(program) == [("timeout", pytest.approx(150))]


def test_a_function_without_a_timeout_still_exports_a_bounded_wait():
    lowered = pulse.compile(decoded, schedule="rcp").lower(Q[0])
    structured = lowered.export_to_controller(version=2)
    for program in (
        lowered.export_to_controller(),
        structured,
        controller_module.unroll(structured),
    ):
        assert _answer_waits(program) == [("bounded", None)]
    assert "timeout_ns" not in next(e for e in structured["events"] if e["kind"] == "answer")


# ── Values known at run time ────────────────────────────────────────────────


@pulse.classical(max_latency_ns=200)
def settings(bits: pulse.Bits) -> tuple[float, float, float]:
    return 0.1 * bits[0], 1.0e5 * bits[0], 40.0 + bits[0]


@pulse.kernel
def answered(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    amp, detuning, delay = settings.submit([m]).result()
    shift_frequency(d.tone, detuning)
    drive(d, square(20, amp * 0.5))
    wait(d, delay, bounds=(8, 400))
    shift_phase(d.tone, amp)
    drive(d, square(20, 0.2))


def test_values_from_answers_are_exported_as_expressions():
    program = pulse.compile(answered).lower(Q[0]).export_to_controller()
    events = program["events"]
    answer = next(e for e in events if e["kind"] == "answer")["index"]
    first, second = (e for e in events if e["kind"] == "drive")
    assert first["amplitude"] is None and first["frequency_hz"] is None
    assert first["waveform"]["runtime"] == {
        "amplitude": {
            "op": "mul",
            "type": "float",
            "of": [{"answer": answer, "result": 0}, {"const": 0.5}],
        }
    }
    [shift] = first["frame_updates"]
    [phase] = second["frame_updates"]
    wait = next(e for e in events if e["kind"] == "wait")
    # Each change is where the program makes it on its tone's line: the
    # shot's start, then the end of the wait before it.
    assert shift == {
        "id": shift["id"],
        "op": "shift_frequency",
        "start_cycles": 0,
        "value": {"answer": answer, "result": 1},
        "conditions": [],
    }
    assert phase == {
        "id": phase["id"],
        "op": "shift_phase",
        "start_cycles": wait["end_cycles"],
        "value": {"answer": answer, "result": 0},
        "conditions": [],
    }
    assert phase["id"] != shift["id"]
    assert "runtime" not in second["waveform"]
    readout = next(e for e in events if e["kind"] == "readout")
    # The readout tone's frame does not depend on run time.
    assert first["tone"] == second["tone"] and "tone" not in readout
    assert wait["start_cycles"] == first["end_cycles"]
    assert (wait["min_cycles"], wait["max_cycles"]) == (8, 400)

    def answers(expr):
        if "answer" in expr:
            return [(expr["answer"], expr["result"])]
        return [a for child in expr.get("of", []) for a in answers(child)]

    assert answers(wait["length"]) == [(answer, 2)]
    submit = next(e for e in events if e["kind"] == "submit")
    assert submit["operands"] == [{"readout": readout["index"]}]
    assert json.loads(json.dumps(events)) == events


@pulse.classical(max_latency_ns=200)
def detuning() -> float:
    return 0.0


@pulse.kernel
def retuned_by_answer(q):
    d = get_drive_line(q)
    x = detuning.submit().result()
    for _ in range(3):
        shift_frequency(d.tone, x)
        drive(d, square(16, 0.1))
        wait(d, 8)
        shift_frequency(d.tone, -x)


def retuned_by_answer_changes(events):
    """The places of the frame changes of `retuned_by_answer`'s export, checked
    against its drives and waits: each drive takes the change after the wait
    before it, at that wait's end, and its own, at its start (a change before
    the first pulse of an iteration on a tone the iteration before played is
    placed at the pulse); the last change is a frame update where the last
    wait ends. Returns every change's (id, start_cycles)."""
    drives = [e for e in events if e["kind"] == "drive"]
    waits = [e for e in events if e["kind"] == "wait"]
    [kept] = (e for e in events if e["kind"] == "frame_update")
    assert len(drives) == len(waits) == 3
    places = []
    for k, drive in enumerate(drives):
        made = [u["start_cycles"] for u in drive["frame_updates"]]
        expected = [drive["start_cycles"]]
        if k:
            expected.insert(0, waits[k - 1]["end_cycles"])
        assert made == expected
        places += [(u["id"], u["start_cycles"]) for u in drive["frame_updates"]]
    [last] = kept["frame_updates"]
    assert last["start_cycles"] == kept["start_cycles"] == waits[-1]["end_cycles"]
    return [*places, (last["id"], last["start_cycles"])]


def test_each_run_time_frame_change_is_exported_with_where_it_is_made():
    lowered = pulse.compile(retuned_by_answer).lower(Q[0])
    flat = lowered.export_to_controller()
    places = retuned_by_answer_changes(flat["events"])
    assert [i for i, _ in places] == list(range(6))
    # Format /2 places them as it places its events.
    structured = lowered.export_to_controller(version=2)
    assert retuned_by_answer_changes(controller_module.unroll(structured)["events"]) == places


def test_a_frame_changes_place_is_in_its_controllers_own_cycles():
    program = pulse.compile(retuned_by_answer).lower(Q[0]).export_to_controller()
    places = retuned_by_answer_changes(program["events"])
    slow = ControlSystem(controllers=[Controller(clock_ghz=program["clock_ghz"] / 2)])
    own = pulse.compile(retuned_by_answer, control_system=slow).lower(Q[0]).export_to_controller()
    assert retuned_by_answer_changes(own["events"]) == [(i, -(-start // 2)) for i, start in places]


@pulse.kernel
def retuned_tones(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    upper = d.tones[(1, 2)]
    x = detuning.submit().result()
    shift_frequency(d.tone, x)
    drive(d, square(16, 0.1))
    m = readout(r, square(40, 0.05))
    if m:
        shift_phase(upper, 0.25)
    drive(d, square(16, 0.1), tone=upper)
    readout(r, square(40, 0.05))


@pulse.kernel
def fixed_tones(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(16, 0.1))
    readout(r, square(40, 0.05))
    drive(d, square(16, 0.1), tone=d.tones[(1, 2)])


def test_each_tone_tracked_at_run_time_is_exported_with_its_reset_frame():
    lowered = pulse.compile(retuned_tones).lower(Q[0])
    flat = lowered.export_to_controller()
    # The frames the tones start in, from a program that changes none.
    fixed = pulse.compile(fixed_tones).lower(Q[0]).export_to_controller()
    assert "runtime_tones" not in fixed
    assert "runtime_tones" not in pulse.compile(fixed_tones).lower(Q[0]).export_to_controller(
        version=2
    )
    drive_hz, _, upper_hz = (e["frequency_hz"] for e in fixed["events"])
    assert flat["runtime_tones"] == {
        "0": {"line": 0, "reset_frequency_hz": drive_hz, "reset_phase": 0.0},
        "1": {"line": 0, "reset_frequency_hz": upper_hz, "reset_phase": 0.0},
    }
    assert upper_hz != drive_hz
    played = [e for e in flat["events"] if e["kind"] == "drive"]
    assert [(e["tone"], e["line"]) for e in played] == [(0, 0), (1, 0)]
    assert list(flat)[list(flat).index("events") + 1] == "runtime_tones"
    persistent = ControlSystem(controllers=[Controller(persistent_frames=True)])
    kept = (
        pulse.compile(retuned_tones, control_system=persistent).lower(Q[0]).export_to_controller()
    )
    keys = list(kept)
    assert keys[keys.index("events") + 1 :][:2] == ["frame_state", "runtime_tones"]
    assert kept["runtime_tones"] == flat["runtime_tones"]
    assert json.loads(json.dumps(flat["runtime_tones"])) == flat["runtime_tones"]
    structured = lowered.export_to_controller(version=2)
    assert structured["runtime_tones"] == flat["runtime_tones"]
    assert controller_module.unroll(structured)["runtime_tones"] == flat["runtime_tones"]


@pulse.kernel
def retuned_slot(q: pulse.Qudit):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(40, 0.05))
    if m:
        shift_phase(d.tone, 0.25)
        shift_phase(r.tone, 0.5)
    drive(d, square(16, 0.1))
    readout(r, square(40, 0.05))


@pytest.mark.parametrize("version", [1, 2])
def test_a_runtime_qudits_tones_reset_to_the_symbols_of_its_native_frequencies(version):
    compiled = pulse.compile(retuned_slot)
    for qudit in (Q[1], Q[2]):
        program = compiled.lower(qudit).export_to_controller(version=version)
        values = {s["symbol"]: s.get("value") for s in program["symbols"]}
        roles = {line["line"]: line["role"] for line in program["lines"]}
        native = {"drive": "s0.f01", "readout": "s0.f_ro"}
        tones = program["runtime_tones"]
        assert sorted(roles[t["line"]] for t in tones.values()) == ["drive", "readout"]
        for tone in tones.values():
            symbol = native[roles[tone["line"]]]
            assert tone["symbols"] == {"qudit": "s0", "reset_frequency_hz": [symbol]}
            assert tone["reset_frequency_hz"] == values[symbol]
            assert tone["reset_phase"] == 0.0


@pulse.classical(max_latency_ns=200)
def scaled(bits: pulse.Bits, value: float) -> float:
    return value * (bits[0] + bits[1])


@pulse.kernel
def chained(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    a = readout(ro, square(100, 0.05))
    b = readout(ro, square(100, 0.05))
    first = scaled.submit([a, b], 0.5).result()
    second = scaled.submit([b, a], first * 2.0).result()
    drive(d, square(20, second))


def test_a_submission_carries_its_arguments_in_order():
    events = pulse.compile(chained).lower(Q[0]).export_to_controller()["events"]
    a, b = (e["index"] for e in events if e["kind"] == "readout")
    first, second = (e for e in events if e["kind"] == "submit")
    answer = next(e for e in events if e["kind"] == "answer")["index"]
    assert first["operands"] == [{"readout": a}, {"readout": b}, {"const": 0.5}]
    assert second["operands"][:2] == [{"readout": b}, {"readout": a}]
    assert second["operands"][2] == {
        "op": "mul",
        "type": "float",
        "of": [{"answer": answer, "result": 0}, {"const": 2.0}],
    }


@pulse.kernel
def chosen_by_readout(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    if m:
        value = 0.3
    else:
        value = 0.1
    shift_phase(d.tone, value)
    drive(d, square(20, value))


def test_a_value_a_readout_selects_at_run_time_is_exported_as_its_select(monkeypatch):
    monkeypatch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)
    events = pulse.compile(chosen_by_readout).lower(Q[0]).export_to_controller()["events"]
    readout = next(e for e in events if e["kind"] == "readout")["index"]
    [drive] = (e for e in events if e["kind"] == "drive")
    select = {
        "op": "select",
        "type": "float",
        "of": [
            {"op": "ne", "type": "int", "of": [{"readout": readout}, {"const": 0}]},
            {"const": 0.3},
            {"const": 0.1},
        ],
    }
    assert drive["conditions"] == []
    assert drive["waveform"]["params"]["amplitude"] is None
    assert drive["waveform"]["runtime"] == {"amplitude": select}
    [shift] = drive["frame_updates"]
    assert (shift["op"], shift["value"]) == ("shift_phase", select)


@pulse.kernel
def frame_in_arms(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    shift_phase(d.tone, 0.25)
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        drive(d, square(20, 0.1))
        shift_phase(d.tone, 0.5)
    else:
        shift_frequency(d.tone, 1.0e6)
    drive(d, square(20, 0.1))


def test_frame_updates_carry_the_arm_they_happen_in():
    program = _program(frame_in_arms)
    readout = next(e for e in program["events"] if e["kind"] == "readout")["index"]
    in_arm, after = (e for e in program["events"] if e["kind"] == "drive")
    assert [(u["op"], u["value"]) for u in in_arm["frame_updates"]] == [("shift_phase", 0.25)]
    # The drive in the if arm took the shift from before the branch, so
    # after the branch it is pending only on the else arm's path.
    ops = [
        (u["op"], u["value"], [(c["readouts"], c["arm"]) for c in u["conditions"]])
        for u in after["frame_updates"]
    ]
    assert ops == [
        ("shift_phase", 0.25, [([readout], False)]),
        ("shift_phase", 0.5, [([readout], True)]),
        ("shift_frequency", 1.0e6, [([readout], False)]),
    ]
    # The shift from before the branch is one change, listed on both paths.
    assert after["frame_updates"][0]["id"] == in_arm["frame_updates"][0]["id"]
    assert len({u["id"] for u in after["frame_updates"]}) == 3
    assert all("length" not in e for e in program["events"] if e["kind"] == "wait")


@pulse.kernel
def decided_at_compile_time(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for k in range(3):
        shift_phase(d.tone, 0.1 * k)
        drive(d, square(20, 0.1 * (k + 1)))
    m = readout(ro, square(100, 0.05))
    sync(d, ro)
    if m:
        drive(d, square(20, 0.1))
    wait(d, 16)
    drive(d, square(20, 0.1))


def test_a_program_computing_nothing_at_run_time_exports_no_run_time_keys():
    events = pulse.compile(decided_at_compile_time).lower(Q[0]).export_to_controller()["events"]
    keys = {"tone", "frame_updates", "phase_offset_rad", "length", "operands"}
    assert all(not keys & e.keys() for e in events)
    assert all("runtime" not in (e.get("waveform") or {}) for e in events)
