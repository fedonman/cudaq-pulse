# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clocks of a control system: the program clock its controllers give a
quantum system, and the reference clock they lock to."""

from __future__ import annotations

import dataclasses
import json
import pickle

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import QuantumSystemDefinitionError
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import ControllerLimitError

from tests._handles import ladder_target, with_line_delays


def _system(*clocks, tolerance=0, **kwargs):
    """A control system of one controller per clock, each playing the drive
    and readout lines of two qudits of four."""
    return pulse.ControlSystem(
        controllers=[
            pulse.Controller(
                drive_lines=[2 * k, 2 * k + 1],
                readout_lines=[2 * k, 2 * k + 1],
                clock_ghz=c,
                alignment_tolerance_cycles=tolerance,
            )
            for k, c in enumerate(clocks)
        ],
        **kwargs,
    )


def _device(*clocks, system=None, **config):
    control = system or _system(*clocks)
    return ladder_target(4, control_system=control, **config)


# ── The program clock ────────────────────────────────────────────────────────


def test_the_program_clock_is_the_fastest_controller_clock_the_others_divide():
    assert _device(1.0, 0.5).clock_ghz == 1.0
    assert _device(0.25, 1.0).clock_ghz == 1.0
    # A controller without a clock runs on the program clock.
    assert _device(None, 0.5).clock_ghz == 0.5
    # Without any controller clock, the quantum system's.
    assert _device(None, None).clock_ghz == ladder_target(4).clock_ghz


def test_uneven_controller_clocks_need_a_declared_program_clock():
    with pytest.raises(QuantumSystemDefinitionError) as caught:
        _device(2.4, 1.0)
    [problem] = caught.value.problems
    assert problem.path == "clock_ghz"
    # Every clock that works, never one picked for the user.
    assert "set 'clock_ghz' in its config to one of 12 GHz" in problem.message
    assert "2.4 GHz (controllers[0]'s; then controllers[1] every 12 program cycles)" in (
        problem.message
    )
    assert _device(2.4, 1.0, clock_ghz=12.0).clock_ghz == 12.0
    assert _device(2.4, 1.0, clock_ghz=1.0).clock_ghz == 1.0


def test_a_declared_program_clock_has_a_whole_cycle_ratio_to_every_controller():
    with pytest.raises(QuantumSystemDefinitionError, match="no whole-cycle ratio") as caught:
        _device(1.0, 0.5, clock_ghz=3.1415926)
    assert [p.path for p in caught.value.problems] == ["clock_ghz"]
    assert _device(1.0, 0.5, clock_ghz=2.0).clock_ghz == 2.0


def test_a_device_built_directly_derives_its_program_clock_too():
    target = ladder_target(4)._description
    assert pulse.QuantumSystem(target, name="t", control_system=_system(1.0, 0.5)).clock_ghz == 1.0
    with pytest.raises(QuantumSystemDefinitionError, match="do not divide evenly"):
        pulse.QuantumSystem(target, name="t", control_system=_system(2.4, 1.0))


def test_a_derived_program_clock_round_trips_through_the_system_file():
    quantum_system = _device(1.0, 0.5)
    again = pulse.QuantumSystem.from_dict(json.loads(json.dumps(quantum_system.as_dict())))
    assert again.clock_ghz == 1.0 and again == quantum_system


# ── The reference clock ──────────────────────────────────────────────────────


def test_every_controller_clock_is_a_whole_multiple_of_the_reference_clock():
    assert _device(1.0, 0.5, system=_system(1.0, 0.5, reference_clock_ghz=0.25)).clock_ghz == 1.0
    with pytest.raises(QuantumSystemDefinitionError) as caught:
        _device(system=_system(1.0, None, reference_clock_ghz=0.3))
    paths = [p.path for p in caught.value.problems]
    # The second runs on the program clock, 1 GHz: neither is a multiple.
    assert paths == [
        "control_system.controllers[0].clock_ghz",
        "control_system.controllers[1].clock_ghz",
    ]
    assert "its 1 GHz clock (the program clock) is not a whole multiple of the 0.3 GHz" in str(
        caught.value
    )


# ── Scheduling on each controller's grid ─────────────────────────────────────
#
# Two controllers at 1 and 0.5 GHz on a 2 GHz program clock: qudits 0 and 1
# move in steps of 2 program cycles, qudits 2 and 3 in steps of 4.

SCHEDULES = ("asap", "alap", "rcp", "alap_rcp")
LINKS = {"controller": Link(fixed_ns=10)}


def _mixed(**kwargs):
    # Pulses a sync lines up across the two land within one of their cycles.
    kwargs.setdefault("tolerance", 1)
    return _device(system=_system(1.0, 0.5, **kwargs), clock_ghz=2.0)


def _grid(qudit):
    return 2 if qudit < 2 else 4


# The loop stays rolled, so its body is placed on the grid once for every
# iteration.
@pulse.kernel(unroll_threshold=0)
def mixed(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    ra = get_readout_line(a)
    drive(da, gaussian(40, 0.1, 10.0))
    wait(db, 8)
    drive(db, square(pulse.round_duration(db, 15), 0.1))
    sync(da, db, ra)
    drive(da, gaussian(40, 0.1, 10.0))
    readout(ra, square(pulse.round_duration(ra, 99), 0.05))
    for _ in range(4):
        drive(db, square(12, 0.1))


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_operation_starts_and_ends_on_its_own_controllers_grid(schedule):
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule=schedule, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    events = [e for e in lowered.schedule if e.kind in ("drive", "readout", "wait")]
    assert {e.qudit for e in events} == {0, 2}
    assert all(e.start % _grid(e.qudit) == 0 and e.end % _grid(e.qudit) == 0 for e in events)
    text = lowered.mlir()
    assert "#pulse.line<drive, qudits = [2], controller = 1>" in text
    assert "#pulse.controller<grid_cycles = 2, " in text
    assert "#pulse.controller<grid_cycles = 4, " in text


@pulse.kernel
def back_to_back(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(2, 0.1))
    drive(da, square(2, 0.1))
    drive(db, square(4, 0.1))
    drive(db, square(4, 0.1))


def test_each_line_keeps_its_own_controllers_finer_grid():
    qpu = _mixed()
    lowered = pulse.compile(back_to_back, schedule="asap", passes=[], quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    starts = {(e.qudit, e.start) for e in lowered.schedule if e.kind == "drive"}
    assert starts == {(0, 0), (0, 2), (2, 0), (2, 4)}


@pulse.kernel
def twice(q: pulse.Qudit):
    d = get_drive_line(q)
    drive(d, square(4, 0.1))
    drive(d, square(4, 0.2))


@pulse.kernel
def short_twice(q: pulse.Qudit):
    d = get_drive_line(q)
    drive(d, square(2, 0.1))
    drive(d, square(2, 0.2))


@pulse.kernel
def shaped_anywhere(q: pulse.Qudit):
    d = get_drive_line(q)
    drive(d, custom(8, ramp))


def test_a_runtime_qudits_one_schedule_is_on_every_controllers_grid():
    # Compiled once for any qudit, its lines start where every grid meets...
    qpu = _mixed()
    compiled = pulse.compile(twice, schedule="asap", passes=[], quantum_system=qpu)
    for qudit in (0, 2):
        lowered = compiled.lower(qpu.qudits[qudit])
        assert [e.start for e in lowered.schedule if e.kind == "drive"] == [0, 4]
    # ...and each call checks its lengths on the grid of the controller it binds.
    short = pulse.compile(short_twice, schedule="asap", passes=[], quantum_system=qpu)
    assert [e.start for e in short.lower(qpu.qudits[0]).schedule if e.kind == "drive"] == [0, 4]
    with pytest.raises(ControllerLimitError, match="line d2 moves in timesteps of 4 cycles"):
        short(qpu.qudits[2])
    # Which rate the samples need depends on the qudit.
    with pytest.raises(pulse.CompilationError, match="runtime qudit's line"):
        pulse.compile(shaped_anywhere, quantum_system=qpu).lower(qpu.qudits[0])


def test_a_sync_meets_on_every_lines_grid():
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule="asap", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    drives = [e for e in lowered.schedule if e.kind == "drive" and e.qudit == 0]
    # Qudit 2's line is busy until 8 + 16 = 24 and qudit 0's until 40; the
    # sync meets at 40, a step of both grids.
    assert drives[1].start == 40


@pulse.kernel
def callee_body(q):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    drive(d, square(4, 0.1))


callee = pulse.kernel(schedule="asap")(callee_body.__pulse_kernel_fn__)


@pulse.kernel
def calls_a_block(a, b):
    da = get_drive_line(a)
    drive(da, square(2, 0.1))
    callee(a)
    callee(b)


@pytest.mark.parametrize("schedule", ("asap", "alap"))
def test_a_kernel_with_its_own_schedule_is_placed_on_its_lines_grid(schedule):
    qpu = _mixed()
    lowered = pulse.compile(calls_a_block, schedule=schedule, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    drives = [e for e in lowered.schedule if e.kind == "drive"]
    assert all(e.start % _grid(e.qudit) == 0 for e in drives)


@pulse.classical(max_latency_ns=60, link="controller")
def correction(bits: pulse.Length[pulse.Bits, 1]) -> float:
    return 0.1 * bits[0]


@pulse.kernel
def with_a_call(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    amplitude = correction.submit([m]).result()
    drive(d, square(8, amplitude))


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_classical_calls_keep_each_controllers_lines_on_its_grid(schedule):
    qpu = _mixed(links=LINKS)
    for qudit in (0, 2):
        lowered = pulse.compile(with_a_call, schedule=schedule, quantum_system=qpu).lower(
            qpu.qudits[qudit]
        )
        assert all(
            e.start % _grid(qudit) == 0 for e in lowered.schedule if e.kind in ("drive", "readout")
        )


# ── Off the grid ─────────────────────────────────────────────────────────────


@pulse.kernel
def off_grid(q):
    d = get_drive_line(q)
    drive(d, gaussian(41, 0.1, 10.0))
    wait(d, 5)


def test_a_length_off_its_controllers_grid_fails_to_compile_naming_the_line_and_the_fix():
    qpu = _mixed()
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(off_grid, quantum_system=qpu).lower(qpu.qudits[0])
    drive, wait = caught.value.violations
    assert drive == (
        "line d0 moves in timesteps of 2 cycles; 41 cycles isn't a whole number of "
        "timesteps (the drive at cycle 0; pulse.round_duration(line, 41) gives 42, or "
        "40 rounding down)"
    )
    assert wait.startswith("line d0 moves in timesteps of 2 cycles; 5 cycles isn't")
    with pytest.raises(ControllerLimitError, match="line d2 moves in timesteps of 4 cycles"):
        pulse.compile(off_grid, quantum_system=qpu).lower(qpu.qudits[2])


@pulse.classical(max_latency_ns=60, link="controller")
def wait_length(x: float) -> float:
    return x


@pulse.kernel
def answer_wait_off_grid(q):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    wait(d, wait_length.submit(1.0).result(), bounds=(6, 402))
    drive(d, square(8, 0.1))


def test_a_wait_from_an_answer_has_its_bounds_on_every_controllers_grid():
    qpu = _mixed(links=LINKS)
    # Qudit 0's own grid is 2, but every later op on every line moves by it.
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(answer_wait_off_grid, quantum_system=qpu).lower(qpu.qudits[0])
    assert "move in timesteps of 4 cycles; its bounds (6, 402) aren't whole" in str(caught.value)


@pulse.kernel
def variable_wait(q, tau: float):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    wait(d, tau * 2.0, bounds=(0, 1000))
    drive(d, square(8, 0.1))


def test_a_variable_wait_off_its_controllers_grid_fails_at_the_call():
    qpu = _mixed()
    compiled = pulse.compile(variable_wait, quantum_system=qpu)
    assert [e.start for e in compiled.lower(qpu.qudits[2], 6.0).schedule if e.kind == "drive"] == [
        0,
        20,
    ]
    with pytest.raises(ValueError, match="nearest lengths are 4 and 8"):
        compiled.lower(qpu.qudits[2], 3.0)
    # Qudit 0's grid takes a length of 6.
    assert compiled.lower(qpu.qudits[0], 3.0).schedule


# ── Each controller's limits in its own cycles ───────────────────────────────


@pulse.kernel
def feedback(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    if m:
        drive(d, square(8, 0.1))


def test_rcp_waits_each_readers_feedback_latency_in_program_cycles():
    controllers = [
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=1.0),
        pulse.Controller(
            drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5, feedback_latency_cycles=10
        ),
    ]
    qpu = _device(system=pulse.ControlSystem(controllers=controllers), clock_ghz=2.0)
    lowered = pulse.compile(feedback, schedule="rcp", quantum_system=qpu).lower(qpu.qudits[2])
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    # 10 cycles at 0.5 GHz are 40 of the program's.
    assert drive.start >= readout.end + 40


@pulse.classical(max_latency_ns=100)
def echo_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def call_then_branch(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(400, 0.05))
    fix = echo_bit.submit([m]).result()
    if fix[0]:
        drive(d, square(40, 0.3))


def _one_controller_at(clock_ghz):
    """One controller at *clock_ghz* with 100 cycles of its own feedback
    latency, under a 2 GHz program."""
    controller = pulse.Controller(
        drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=clock_ghz, feedback_latency_cycles=100
    )
    return _device(system=pulse.ControlSystem(controllers=[controller]), clock_ghz=2.0)


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
@pytest.mark.parametrize(("clock_ghz", "latency"), [(0.5, 400), (1.0, 200), (2.0, 100)])
def test_one_controller_at_another_clock_waits_its_feedback_latency_in_program_cycles(
    clock_ghz, latency, schedule
):
    # 100 cycles at 0.5 GHz are 400 of the 2 GHz program's. Before, they
    # were waited as 100 program cycles, and rcp failed its own check.
    qpu = _one_controller_at(clock_ghz)
    lowered = pulse.compile(feedback, schedule=schedule, quantum_system=qpu).lower(qpu.qudits[0])
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    assert drive.start - readout.end == latency
    assert pulse.controller.check(lowered) == []
    # A request on the outcome leaves once the latency has passed.
    lowered = pulse.compile(call_then_branch, schedule=schedule, quantum_system=qpu).lower(
        qpu.qudits[0]
    )
    [call] = lowered.classical_report()
    readout = next(e for e in lowered.schedule if e.kind == "readout")
    assert call.submit_cycles >= readout.end + latency
    assert pulse.controller.check(lowered) == []


def test_each_lines_delay_is_split_by_its_own_controllers_fine_delay():
    controllers = [
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], max_fine_delay_ns=1.0),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3]),
    ]

    def compiled(delays):
        system = pulse.ControlSystem(controllers=with_line_delays(controllers, delays))
        qpu = _device(system=system)
        return pulse.compile(feedback, quantum_system=qpu).lower(qpu.qudits[0])

    assert compiled({("drive", 0): 2.3}).fine_delays[("drive", 0)] == pytest.approx(0.3)
    with pytest.raises(pulse.CompilationError, match="the controller has no fine delay"):
        compiled({("drive", 2): 2.3})


# ── Samples at each line's rate ──────────────────────────────────────────────


def ramp(t):
    return 0.1 + 0.01 * t


@pulse.kernel
def shaped_on_both(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    envelope = custom(8, ramp)
    drive(da, envelope)
    drive(db, envelope)


def test_custom_samples_the_envelope_at_the_rate_of_the_line_that_plays_it():
    qpu = _mixed()
    lowered = pulse.compile(shaped_on_both, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    drives = [e for e in lowered.schedule if e.kind == "drive"]
    assert [e.duration_cycles for e in drives] == [8, 8]
    text = lowered.mlir()
    assert "rate = array<i64: 1, 2>" in text and "rate = array<i64: 1, 4>" in text


@pulse.kernel
def shaped(q):
    d = get_drive_line(q)
    drive(d, custom(8, ramp))


@pulse.kernel
def held_for_two_cycles(q):
    d = get_drive_line(q)
    drive(d, custom_samples([ramp(t) for t in (1, 1, 3, 3, 5, 5, 7, 7)]))


def test_a_custom_envelope_is_simulated_at_its_own_controllers_rate():
    # On qudit 0's controller at 1 GHz, each sample is held for two cycles.
    qpu = _mixed()
    lowered = pulse.compile(shaped, quantum_system=qpu).lower(qpu.qudits[0])
    plain = _device(None, None)
    held = pulse.compile(held_for_two_cycles, quantum_system=plain).lower(plain.qudits[0])
    a = pulse.simulate(lowered).density_matrix
    b = pulse.simulate(held).density_matrix
    np.testing.assert_allclose(a, b, atol=1e-12)


@pulse.kernel
def made_for_one_ghz(q):
    d = get_drive_line(q)
    drive(d, custom_samples([0.1, 0.2, 0.3, 0.4], 1.0))


@pulse.kernel
def scaled_custom(q):
    d = get_drive_line(q)
    drive(d, wf_scale(custom(8, ramp), 0.5))


def test_sample_arrays_play_only_on_lines_of_the_rate_they_were_made_for():
    qpu = _mixed()
    assert pulse.compile(made_for_one_ghz, quantum_system=qpu).lower(qpu.qudits[0]).schedule
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(made_for_one_ghz, quantum_system=qpu).lower(qpu.qudits[2])
    message = str(caught.value)
    assert message.startswith(
        "drive(): the waveform on the drive line of qudit 2 holds samples made for 1 GHz, "
        "and the line's controller plays 0.5 GHz (one sample every 4 clock cycles)"
    )
    assert "sample_rate_ghz=0.5" in message and "custom(duration, f)" in message


def test_a_waveform_made_of_custom_is_made_again_at_its_lines_rate():
    qpu = _mixed()
    lowered = pulse.compile(scaled_custom, quantum_system=qpu).lower(qpu.qudits[2])
    assert "rate = array<i64: 1, 4>" in lowered.mlir()
    assert [e.duration_cycles for e in lowered.schedule if e.kind == "drive"] == [8]


# ── One export per controller, in its own cycles ─────────────────────────────


def _timed(program):
    return [e for e in program["events"] if e["kind"] in ("drive", "readout", "wait")]


def test_each_controller_exports_its_part_of_the_program_in_its_own_cycles():
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule="asap", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    with pytest.raises(ValueError) as caught:
        lowered.export_to_controller()
    assert str(caught.value) == (
        "the program plays lines of 2 controllers (controllers[0], controllers[1]); "
        "pass controller= (0 to 1) to export one controller's program"
    )
    for index, qudit, cycles in ((0, 0, 2), (1, 2, 4)):
        program = lowered.export_to_controller(controller=index)
        assert program["controller"] == index
        assert program["clock_ghz"] == (1.0, 0.5)[index] and program["program_clock_ghz"] == 2.0
        assert program["limits"] == qpu.control_system.controllers[index].as_dict()
        assert program["qudits"] == [qudit] and {line["qudit"] for line in program["lines"]} == {
            qudit
        }
        mine = [e for e in lowered.schedule if e.kind in ("drive", "readout", "wait")]
        mine = [e for e in mine if e.qudit == qudit]
        assert [(e["start_cycles"], e["duration_cycles"]) for e in _timed(program)] == [
            (e.start // cycles, e.duration_cycles // cycles) for e in mine
        ]
        # The sync lists only the lines this controller plays.
        [sync] = [e for e in program["events"] if e["kind"] == "sync"]
        ids = {line["line"] for line in program["lines"]}
        assert sync["lines"] and set(sync["lines"]) <= ids
        json.dumps(program)
    with pytest.raises(ValueError, match=r"controller=2, and the program's control system has 2"):
        lowered.export_to_controller(controller=2)
    with pytest.raises(TypeError, match="index of a controller"):
        lowered.export_to_controller(controller="1")  # type: ignore[arg-type]


@pulse.kernel
def mixed_retuned(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    ra = get_readout_line(a)
    m = readout(ra, square(pulse.round_duration(ra, 99), 0.05))
    if m:
        shift_phase(da.tone, 0.25)
    drive(da, gaussian(40, 0.1, 10.0))
    drive(db, square(pulse.round_duration(db, 15), 0.1))


def test_each_controller_exports_the_run_time_tones_of_its_own_lines():
    qpu = _mixed()
    lowered = pulse.compile(mixed_retuned, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    first, second = (lowered.export_to_controller(controller=index) for index in (0, 1))
    [drive] = [e for e in first["events"] if e["kind"] == "drive"]
    assert first["runtime_tones"] == {
        str(drive["tone"]): {
            "line": drive["line"],
            "reset_frequency_hz": qpu.frequencies[0],
            "reset_phase": 0.0,
        }
    }
    assert "runtime_tones" not in second


def test_a_program_on_one_controller_exports_without_naming_it():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(with_a_call, quantum_system=qpu).lower(qpu.qudits[2])
    program = lowered.export_to_controller()
    assert program["controller"] == 1 and program["clock_ghz"] == 0.5
    assert program == lowered.export_to_controller(controller=1)
    drive = next(e for e in program["events"] if e["kind"] == "drive")
    assert drive["duration_cycles"] == 2 and drive["waveform"]["params"]["duration"] == 2.0
    # The other controller takes no part: no lines, events or calls.
    other = lowered.export_to_controller(controller=0)
    assert other["lines"] == [] and other["events"] == [] and other["calls"] == []


def test_samples_at_a_rate_of_each_controllers_clock():
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule="asap", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    program = lowered.export_to_controller(sample_rate_ghz=1.0, controller=1)
    first = next(e for e in program["events"] if e["kind"] == "drive")
    assert len(first["samples"]["re"]) == 2 * first["duration_cycles"]
    with pytest.raises(ValueError, match="whole multiple of the 0.5 GHz clock"):
        lowered.export_to_controller(sample_rate_ghz=0.75, controller=1)
    assert lowered.export_to_controller(sample_rate_ghz=1.0, controller=0)["events"]


@pulse.kernel
def read_then_correct(a, b):
    ra = get_readout_line(a)
    db = get_drive_line(b)
    m = readout(ra, square(100, 0.05))
    amplitude = correction.submit([m]).result()
    drive(db, square(8, amplitude))


def test_a_call_is_submitted_by_its_measuring_controller_and_answered_to_its_taker():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(read_then_correct, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    first, second = (lowered.export_to_controller(controller=k) for k in (0, 1))
    assert [e["kind"] for e in first["events"]] == ["readout", "submit"]
    assert [e["kind"] for e in second["events"]] == ["answer", "drive"]
    answer, drive = second["events"]
    # Event ids are the program's, so the answer's value keeps its name.
    assert drive["waveform"]["runtime"]["amplitude"]["answer"] == answer["index"]
    for program in (first, second):
        assert [c["function"] for c in program["calls"]] == ["correction"]
        assert [f["name"] for f in program["manifest"]["functions"]] == ["correction"]
    starts = {e.kind: e.start for e in lowered.schedule}
    assert first["events"][1]["start_cycles"] == starts["submit"] // 2
    assert answer["start_cycles"] == starts["answer"] // 4


def test_each_controllers_emulation_export_is_its_part_in_its_cycles():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(read_then_correct, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    with pytest.raises(ValueError, match="pass controller="):
        lowered.export_to_emulator()
    first, second = (lowered.export_to_emulator(controller=k) for k in (0, 1))
    assert first["controller"] == 0 and first["clock_ghz"] == 1.0
    assert second["controller"] == 1 and second["clock_ghz"] == 0.5
    assert [e["kind"] for e in first["events"]] == ["readout", "submit"]
    assert [e["kind"] for e in second["events"]] == ["answer", "drive"]
    starts = {e.kind: e.start for e in lowered.schedule}
    assert first["events"][0]["start_cycles"] == starts["readout"] // 2
    assert second["events"][1]["start_cycles"] == starts["drive"] // 4
    assert list(first["links"]) == list(second["links"]) == ["0"]


@pulse.kernel
def answer_wait(q):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    wait(d, wait_length.submit(1.0).result(), bounds=(8, 400))
    drive(d, square(8, 0.1))


def test_a_run_time_wait_is_exported_in_its_controllers_cycles():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(answer_wait, quantum_system=qpu).lower(qpu.qudits[2])
    wait = next(e for e in lowered.export_to_controller()["events"] if e["kind"] == "wait")
    assert (wait["min_cycles"], wait["max_cycles"]) == (2, 100)
    assert wait["length"]["op"] == "controller_cycles" and wait["length"]["den"] == 4


def test_a_run_time_frame_change_is_placed_in_its_controllers_cycles():
    from tests.test_controller import retuned_by_answer, retuned_by_answer_changes

    qpu = _mixed(links={"default": Link(fixed_ns=10)})
    lowered = pulse.compile(retuned_by_answer, quantum_system=qpu).lower(qpu.qudits[2])
    program = lowered.export_to_controller()
    assert program["clock_ghz"] == 0.5
    assert len(retuned_by_answer_changes(program["events"])) == 6


def test_each_controller_exports_the_reference_clock():
    qpu = _mixed(reference_clock_ghz=0.25)
    lowered = pulse.compile(mixed, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    for index in (0, 1):
        assert lowered.export_to_controller(controller=index)["reference_clock_ghz"] == 0.25


def test_drives_of_one_waveform_have_its_times_in_their_controllers_cycles():
    qpu = _mixed()
    lowered = pulse.compile(mixed, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    # The two gaussians of 40 program cycles at 2 GHz, on a 1 GHz controller.
    drives = [
        e for e in lowered.export_to_controller(controller=0)["events"] if e["kind"] == "drive"
    ]
    assert len(drives) == 2
    for drive in drives:
        assert drive["duration_cycles"] == 20
        assert drive["waveform"]["params"] == {"duration": 20.0, "amplitude": 0.1, "sigma": 5.0}


def test_a_controller_faster_than_the_program_clock_counts_more_cycles():
    qpu = _device(4.0, 1.0, clock_ghz=2.0)
    lowered = pulse.compile(shaped_twice_plain, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    program = lowered.export_to_controller(controller=0)
    assert program["clock_ghz"] == 4.0 and program["program_clock_ghz"] == 2.0
    drive = next(e for e in program["events"] if e["kind"] == "drive")
    assert drive["duration_cycles"] == 16 and drive["waveform"]["params"]["duration"] == 16.0
    drive = next(
        e for e in lowered.export_to_controller(controller=1)["events"] if e["kind"] == "drive"
    )
    assert drive["duration_cycles"] == 4


@pulse.kernel
def shaped_twice_plain(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(8, 0.1))
    drive(db, square(8, 0.1))


def test_a_one_controller_device_exports_as_before():
    qpu = _device(0.5)
    lowered = pulse.compile(mixed, schedule="asap", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[1]
    )
    program = lowered.export_to_controller()
    assert "controller" not in program and program == lowered.export_to_controller(controller=0)
    assert "controller" not in lowered.export_to_emulator()
    with pytest.raises(ValueError, match="has one controller"):
        lowered.export_to_controller(controller=1)


@pulse.kernel(unroll_threshold=10**6)
def rounds_on_both(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    ra = get_readout_line(a)
    for _ in range(3):
        drive(da, gaussian(40, 0.1, 10.0))
        drive(db, drag(40, 0.2, 10.0, 2.0))
        sync(da, db, ra)
        readout(ra, square(40, 0.05))
        sync(da, db, ra)


@pytest.mark.parametrize("clocks", [(2.0, 2.0), (1.0, 0.5)])
def test_checking_each_controllers_export_leaves_it_as_exported_unchecked(clocks, monkeypatch):
    qpu = _device(system=_system(*clocks, tolerance=1), clock_ghz=2.0)
    lowered = pulse.compile(rounds_on_both, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    checked = [lowered.export_to_controller(controller=k) for k in (0, 1)]
    assert all(len(program["events"]) > 6 for program in checked)
    monkeypatch.setattr(pulse.ControlSystem, "_violations", lambda *args: [])
    assert [lowered.export_to_controller(controller=k) for k in (0, 1)] == checked


@pulse.kernel(unroll_threshold=10**6)
def rounds_at(a, b, amplitude: float):
    da = get_drive_line(a)
    db = get_drive_line(b)
    ra = get_readout_line(a)
    for _ in range(3):
        drive(da, gaussian(40, amplitude, 10.0))
        drive(db, drag(40, 0.2, 10.0, 2.0))
        sync(da, db, ra)
        readout(ra, square(40, 0.05))
        sync(da, db, ra)


def _rounds(clocks, amplitude=0.1, **limits):
    system = _system(*clocks, tolerance=1)
    if limits:
        system = pulse.ControlSystem(
            controllers=[
                system.controllers[0],
                dataclasses.replace(system.controllers[1], **limits),
            ]
        )
    qpu = _device(system=system, clock_ghz=2.0)
    return pulse.compile(rounds_at, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2], amplitude
    )


def _walks_and_checks(monkeypatch):
    """Count the walks of a whole program and the checks of a whole system."""
    from cudaq_pulse import controller

    counts = {"walks": 0, "checks": 0}
    walk = controller._walk
    check = pulse.ControlSystem._violations

    def counted_walk(*args, **kwargs):
        counts["walks"] += 1
        return walk(*args, **kwargs)

    def counted_check(*args, **kwargs):
        counts["checks"] += 1
        return check(*args, **kwargs)

    monkeypatch.setattr(controller, "_walk", counted_walk)
    monkeypatch.setattr(pulse.ControlSystem, "_violations", counted_check)
    return counts


@pytest.mark.parametrize("clocks", [(2.0, 2.0), (1.0, 0.5)])
def test_exporting_every_controller_walks_and_checks_the_program_once(clocks, monkeypatch):
    from cudaq_pulse import controller

    lowered = _rounds(clocks)
    counts = _walks_and_checks(monkeypatch)
    shared = [lowered.export_to_controller(controller=k) for k in (0, 1)]
    assert counts == {"walks": 1, "checks": 1}
    # Each export alone walks and checks the program again; what it gives is
    # the same, byte for byte, in any order and with samples.
    monkeypatch.setattr(controller, "_shared_parts", controller._Parts)
    alone = [lowered.export_to_controller(controller=k) for k in (1, 0)][::-1]
    assert counts == {"walks": 3, "checks": 3}
    assert json.dumps(shared) == json.dumps(alone)
    sampled = lowered.export_to_controller(4.0, controller=1)
    monkeypatch.undo()
    assert lowered.export_to_controller(4.0, controller=1) == sampled
    assert lowered.export_to_controller(controller=0) == shared[0]


@pulse.kernel(unroll_threshold=10**6)
def turned_rounds(a, b, c):
    da, db, dc = get_drive_line(a), get_drive_line(b), get_drive_line(c)
    for _ in range(3):
        shift_phase(da.tone, 0.5)
        shift_phase(db.tone, 0.5)
        shift_phase(dc.tone, 0.5)
        drive(da, square(16, 0.1))
        drive(db, square(16, 0.1))
        drive(dc, square(16, 0.1))
        sync(da, db, dc)


@pytest.mark.parametrize("clocks", [(2.0, 2.0), (1.0, 0.5)])
def test_exports_on_spaced_update_buses_walk_and_check_once(clocks, monkeypatch):
    from cudaq_pulse import controller

    bus = controller.UpdateBus(drive_lines=[0, 1], cycles_per_update=8)
    first, second = _system(*clocks, tolerance=1).controllers
    system = pulse.ControlSystem(
        controllers=[
            dataclasses.replace(first, update_buses=[bus], space_updates=True),
            dataclasses.replace(second, update_buses=[controller.UpdateBus(drive_lines=[2, 3])]),
        ]
    )
    qpu = _device(system=system, clock_ghz=2.0)
    lowered = pulse.compile(turned_rounds, schedule="asap", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[1], qpu.qudits[2]
    )
    counts = _walks_and_checks(monkeypatch)
    shared = [lowered.export_to_controller(controller=k) for k in (0, 1)]
    assert counts == {"walks": 1, "checks": 1}
    # After the first, set before the shot, the bus holds the second update
    # of each round back by its cycles.
    starts = [
        [e["start_cycles"] for e in shared[0]["events"] if e["kind"] == "drive" and e["line"] == k]
        for k in (0, 1)
    ]
    gaps = [b - a for a, b in zip(*starts, strict=True)]
    assert gaps[0] == 0 and min(gaps[1:]) >= 8
    assert shared[0]["limits"]["space_updates"] is True
    monkeypatch.setattr(controller, "_shared_parts", controller._Parts)
    alone = [lowered.export_to_controller(controller=k) for k in (1, 0)][::-1]
    assert json.dumps(shared) == json.dumps(alone)
    # Unspaced, the second update of the second and third rounds clashes on
    # the bus, and the system check finds it on its controller's events.
    unspaced = _device(system=_system(*clocks, tolerance=1), clock_ghz=2.0)
    plain = pulse.compile(turned_rounds, schedule="asap", quantum_system=unspaced).lower(
        unspaced.qudits[0], unspaced.qudits[1], unspaced.qudits[2]
    )
    tight = pulse.ControlSystem(
        controllers=[dataclasses.replace(system.controllers[0], space_updates=False), second]
    )
    lines, events, _, _, _ = controller._walk(plain, slots=None)
    found = tight._violations(
        plain, lines, events, (plain._generic or plain).clocks, (), (), True, None, None
    )
    assert len(found) == 2
    assert all(
        v.startswith("controllers[0]: update bus 0 (drive lines of qudits 0, 1)") for v in found
    )


def test_a_program_keeps_its_walk_until_every_controller_is_exported():
    lowered = _rounds((1.0, 0.5))
    assert not hasattr(lowered, "_export_parts")
    first = lowered.export_to_controller(controller=1)
    kept = lowered._export_parts
    # Again, or changed by its caller: the next export shares nothing with it.
    first["events"][0]["start_cycles"] = -1
    first["lines"][0]["qudit"] = -1
    again = lowered.export_to_controller(controller=1)
    assert again["events"][0]["start_cycles"] >= 0 and again["lines"][0]["qudit"] == 2
    assert lowered._export_parts is kept
    lowered.export_to_controller(controller=0)
    assert not hasattr(lowered, "_export_parts")
    # A pickled program leaves it behind.
    lowered.export_to_controller(controller=0)
    assert not hasattr(pickle.loads(pickle.dumps(lowered)), "_export_parts")


def test_a_programs_walk_is_not_shared_with_other_values_or_controllers():
    lowered = _rounds((2.0, 2.0), amplitude=0.1)
    lowered.export_to_controller(controller=1)
    generic = lowered.generic
    other = generic.bind([0.3])
    peaks = [
        e["waveform"]["params"]["amplitude"]
        for e in other.export_to_controller(controller=0)["events"]
        if e["kind"] == "drive"
    ]
    assert peaks == [0.3] * 3
    fresh = _rounds((2.0, 2.0), 0.3)
    assert other.export_to_controller(controller=0) == fresh.export_to_controller(controller=0)
    # The same kernel and values on other controllers is another program.
    limited = _rounds((2.0, 2.0), max_amplitude=0.5)
    assert limited.export_to_controller(controller=1)["limits"]["max_amplitude"] == 0.5
    assert lowered.export_to_controller(controller=0)["limits"].get("max_amplitude") is None
    # A program whose control system was replaced walks and checks again.
    kept = _rounds((2.0, 2.0))
    kept.export_to_controller(controller=0)
    kept.options = limited.options
    assert kept.export_to_controller(controller=1)["limits"]["max_amplitude"] == 0.5


def test_every_controllers_export_raises_the_systems_violations(monkeypatch):
    lowered = _rounds((1.0, 0.5))
    counts = _walks_and_checks(monkeypatch)
    check = pulse.ControlSystem._violations

    def over(*args):
        return [*check(*args), "controllers[1]: too much"]

    monkeypatch.setattr(pulse.ControlSystem, "_violations", over)
    raised = []
    for k in (0, 0, 1):
        with pytest.raises(ControllerLimitError) as caught:
            lowered.export_to_controller(controller=k)
        raised.append(caught.value.violations)
        if k == 0:
            # A failed check keeps its violations and not the program.
            assert lowered._export_parts.events == []
            # A part chosen wrongly is named so first, as before.
            with pytest.raises(pulse.InvalidValueError, match="2 controllers"):
                lowered.export_to_controller(controller=2)
            with pytest.raises(pulse.InvalidValueError, match="plays lines of 2 controllers"):
                lowered.export_to_controller()
    assert list(raised[0]) == ["controllers[1]: too much"]
    assert raised[0] == raised[1] == raised[2]
    assert counts == {"walks": 1, "checks": 1}
    # Once every part has been asked for, the program keeps nothing.
    assert not hasattr(lowered, "_export_parts")


def test_the_amplitude_check_samples_each_envelope_once_and_reports_every_pulse(monkeypatch):
    from cudaq_pulse import controller

    line = {"line": 0, "role": "drive", "qudit": 0}
    # One shape at two amplitudes, alternating, each event with its own tree.
    events = [
        {
            "kind": "drive",
            "start_cycles": 40 * k,
            "waveform": {
                "shape": "square",
                "params": {"duration": 40.0, "amplitude": a, "amplitude_imag": 0.0},
            },
        }
        for k, a in enumerate([0.9, 0.3] * 3)
    ]
    sampled = []
    peak = controller._peak_amplitude
    monkeypatch.setattr(
        controller, "_peak_amplitude", lambda tree: sampled.append(tree) or peak(tree)
    )
    found = controller._over_amplitude(line, events, 0.5)
    assert len(sampled) == 2
    assert [v.split(" peaks")[0].rsplit(" ", 1)[1] for v in found] == ["0", "80", "160"]
    assert all("amplitude 0.9;" in v for v in found)


@pulse.kernel
def two_envelopes_everywhere(*qudits):
    for q in qudits:
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.4, 10.0))
        drive(d, drag(40, 0.2, 10.0, 2.0))


def _samplings(count, clocks, monkeypatch):
    """The envelopes the amplitude checks sample to compile a program on
    *count* controllers of two qudits each (on *clocks* in turn) and export
    every controller."""
    from cudaq_pulse import controller

    controllers = [
        pulse.Controller(
            drive_lines=[2 * k, 2 * k + 1], clock_ghz=clocks[k % len(clocks)], max_amplitude=0.5
        )
        for k in range(count)
    ]
    qpu = ladder_target(2 * count, control_system=pulse.ControlSystem(controllers=controllers))
    sampled = []
    peak = controller._peak_amplitude
    monkeypatch.setattr(
        controller, "_peak_amplitude", lambda tree: sampled.append(tree) or peak(tree)
    )
    lowered = pulse.compile(two_envelopes_everywhere, quantum_system=qpu).lower(
        *(qpu.qudits[q] for q in range(2 * count))
    )
    for k in range(count):
        lowered.export_to_controller(controller=k)
    monkeypatch.undo()
    return len(sampled)


@pytest.mark.parametrize("clocks", [(1.0,), (1.0, 0.5)])
def test_the_system_check_samples_each_envelope_once_for_every_controller(clocks, monkeypatch):
    # Each controller's check takes the peaks the others sampled: the
    # samplings do not grow with the controllers. On two clocks an envelope
    # in each controller's cycles is its own.
    few = _samplings(2, clocks, monkeypatch)
    assert few > 0
    assert _samplings(8, clocks, monkeypatch) == few


# ── Budgets between controllers ──────────────────────────────────────────────
#
# Without a shared reference clock, each controller's start is known to one
# of its own cycles: 2 program cycles at 1 GHz and 4 at 0.5 GHz.


@pulse.kernel
def branch_across(a, b):
    ra = get_readout_line(a)
    db = get_drive_line(b)
    m = readout(ra, square(100, 0.05))
    sync(ra, db)
    if m:
        drive(db, square(8, 0.1))


def _forwarding(reference=None, skew=0, tolerance=0, delays=None):
    controllers = [
        pulse.Controller(
            drive_lines=[0, 1],
            readout_lines=[0, 1],
            clock_ghz=1.0,
            feedback_latency_cycles=3,
            peer_latency_cycles={1: 2},
            alignment_tolerance_cycles=tolerance,
        ),
        pulse.Controller(
            drive_lines=[2, 3],
            readout_lines=[2, 3],
            clock_ghz=0.5,
            clock_skew_cycles=skew,
            alignment_tolerance_cycles=tolerance,
        ),
    ]
    if delays:
        controllers = with_line_delays(controllers, delays)
    system = pulse.ControlSystem(controllers=controllers, reference_clock_ghz=reference)
    return _device(system=system, clock_ghz=2.0)


def _gap(qpu, schedule="rcp"):
    lowered = pulse.compile(branch_across, schedule=schedule, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    return drive.start - readout.end


@pulse.kernel
def slow_branch(a, b):
    ra = get_readout_line(a)
    db = get_drive_line(b)
    m = readout(ra, square(10000, 0.05))
    if m:
        drive(db, square(10000, 0.1))


@pytest.mark.parametrize(("clock_ghz", "gap"), [(1.234567, 10000), (1.0486, 50), (1.0485, 20)])
def test_controllers_far_faster_than_the_program_clock_schedule_their_feedback(clock_ghz, gap):
    # 1.234567 GHz controllers of a 10 MHz program count 1234567/10000 of
    # their cycles per program cycle: a clock ratio of any size is exact.
    controllers = [
        pulse.Controller(
            drive_lines=[0, 1],
            readout_lines=[0, 1],
            clock_ghz=clock_ghz,
            feedback_latency_cycles=3,
            peer_latency_cycles={1: 2},
        ),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=clock_ghz),
    ]
    qpu = _device(system=pulse.ControlSystem(controllers=controllers), clock_ghz=0.01)
    lowered = pulse.compile(slow_branch, schedule="rcp", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    assert drive.start - readout.end == gap


@pytest.mark.parametrize(
    ("facts", "gap"),
    [
        ({"feedback_latency_cycles": 2**41}, 4398046511112),
        ({"clock_skew_cycles": 2**40}, 2199023255560),
        ({"peer_latency_cycles": 2**41}, 4398046511112),
    ],
)
def test_latencies_of_any_size_that_the_scheduler_counts_schedule(facts, gap):
    controllers = [
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=1.0, **facts),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5),
    ]
    qpu = _device(system=pulse.ControlSystem(controllers=controllers), clock_ghz=2.0)
    assert _gap(qpu) == gap


def test_a_latency_past_what_the_scheduler_counts_names_its_controllers():
    # 2**62 cycles of a 1 GHz controller are 2**63 cycles of a 2 GHz program.
    controllers = [
        pulse.Controller(
            drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=1.0, feedback_latency_cycles=2**62
        ),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5),
    ]
    qpu = _device(system=pulse.ControlSystem(controllers=controllers), clock_ghz=2.0)
    with pytest.raises(
        pulse.InvalidValueError,
        match=r"controllers\[0\].*: the outcome of its readout reaches .* program cycles after the "
        r"readout ends, more than the 2\*\*63 - 1 the scheduler counts; lower its "
        "feedback_latency_cycles",
    ):
        _gap(qpu)


def test_an_outcome_sent_to_another_controller_is_budgeted_with_both_start_uncertainties():
    qpu = _forwarding()
    # Feedback 3 and peer 2 cycles at 1 GHz are 10 program cycles; the start
    # uncertainty adds 2 + 4. Every schedule waits for it.
    assert _gap(qpu, schedule="asap") == _gap(qpu, schedule="alap") == 16
    # A schedule made for the same controllers without the latencies (the
    # start uncertainty alone puts the drive at 108) is refused by this
    # system's check.
    from cudaq_pulse.controller import _compile_check

    quick = [
        dataclasses.replace(c, feedback_latency_cycles=0, peer_latency_cycles=0)
        for c in qpu.control_system.controllers
    ]
    near = _device(system=pulse.ControlSystem(controllers=quick), clock_ghz=2.0)
    lowered = pulse.compile(branch_across, schedule="asap", quantum_system=near).lower(
        near.qudits[0], near.qudits[2]
    )
    [message], _ = _compile_check(lowered, None, qpu.control_system)
    assert message.startswith(
        "controllers[1]: the drive on line 1 at 108 branches on the readout of controllers[0] "
        "ending at 100, whose outcome reaches it 16 cycles later, at 116 (10 of feedback and "
        "peer latency, and 6 of the two controllers' start uncertainty, without a shared "
        "reference_clock_ghz); start it at 116 or later: schedule with rcp"
    )
    # rcp waits for it.
    assert _gap(qpu) == 16
    # A shared reference leaves the latency, and a skew adds its cycles, the
    # drive starting on its line's grid of 4.
    assert _gap(_forwarding(reference=0.5)) == 12
    assert _gap(_forwarding(reference=0.5, skew=1)) == 16
    assert _gap(_forwarding(skew=1)) == 20


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_rcp_waits_for_an_outcome_from_where_its_delayed_readout_ends(schedule):
    # Gaps from the readout's end as played, line delays included: 16 as in
    # test_an_outcome_sent_to_another_controller_is_budgeted_with_both_start_uncertainties.
    assert _gap(_forwarding(), schedule) == 16
    # 2 ns on the reader's readout line end it 4 program cycles later; the
    # drive on the other controller waits for it.
    assert _gap(_forwarding(delays={("readout", 0): 2.0}), schedule) == 16
    # 4 ns on the user's drive line only move the drive 8 cycles later.
    assert _gap(_forwarding(delays={("drive", 2): 4.0}), schedule) == 24
    # On the reader's own controller the drive waits its feedback latency
    # after the delayed readout.
    qpu = _forwarding(delays={("readout", 0): 2.0})
    lowered = pulse.compile(feedback, schedule=schedule, quantum_system=qpu).lower(qpu.qudits[0])
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    assert drive.start - readout.end == 6


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_an_outcome_used_on_its_own_controller_needs_only_its_feedback_latency(schedule):
    # 3 cycles at 1 GHz: 6 of the program's, under every schedule.
    qpu = _forwarding()
    lowered = pulse.compile(feedback, schedule=schedule, quantum_system=qpu).lower(qpu.qudits[0])
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    assert drive.start - readout.end == 6


def _call_across(reference=None, reach_both=True):
    a = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=1.0)
    b = pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5)
    link = Link(fixed_ns=10, controllers=None if reach_both else [a])
    system = pulse.ControlSystem(
        controllers=[a, b], links={"controller": link}, reference_clock_ghz=reference
    )
    return _device(system=system, clock_ghz=2.0)


def _due(qpu, b=2):
    lowered = pulse.compile(read_then_correct, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[b]
    )
    [call] = lowered.classical_report()
    return call.due_cycles - call.submit_cycles


def test_an_answer_on_a_link_between_controllers_is_budgeted_with_their_start_uncertainty():
    shared = _due(_call_across(reference=0.5))
    # 1 ns at 1 GHz and 2 ns at 0.5 GHz: 6 program cycles more.
    assert _due(_call_across()) == shared + 6
    # A link that reaches one controller joins none.
    assert _due(_call_across(reach_both=False), b=1) == _due(_call_across(reference=0.5), b=1)


def test_an_answer_sent_and_taken_on_one_controller_is_budgeted_without_uncertainty():
    # Measured and corrected on controllers[0], over a link that reaches both.
    assert _due(_call_across(), b=1) == _due(_call_across(reference=0.5), b=1)
    # Taken on controllers[1], the same call adds both uncertainties.
    qpu = _call_across()
    assert _due(qpu) == _due(qpu, b=1) + 6


@pulse.kernel
def branch_on_both(a, b):
    ra = get_readout_line(a)
    da = get_drive_line(a)
    db = get_drive_line(b)
    m = readout(ra, square(100, 0.05))
    sync(da, db)
    if m:
        drive(da, square(8, 0.1))
        drive(db, square(8, 0.1))


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_rcp_waits_for_an_outcome_only_as_long_as_it_takes_to_reach_each_controller(schedule):
    qpu = _forwarding(tolerance=1)
    lowered = pulse.compile(branch_on_both, schedule=schedule, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    readout = next(e for e in lowered.schedule if e.kind == "readout")
    mine, other = (e for e in lowered.schedule if e.kind == "drive")
    # On the reader: its feedback latency, 3 cycles at 1 GHz (alap_rcp then
    # ends it with the other); on the other controller: 16, as in
    # test_an_outcome_sent_to_another_controller_is_budgeted_with_both_start_uncertainties.
    assert mine.start - readout.end == (6 if schedule == "rcp" else 16)
    assert other.start - readout.end == 16
    # One controller's program waits its own feedback latency alone.
    lowered = pulse.compile(feedback, schedule=schedule, quantum_system=qpu).lower(qpu.qudits[0])
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    assert drive.start - readout.end == 6


@pulse.kernel
def chosen_across(a, b):
    ra = get_readout_line(a)
    db = get_drive_line(b)
    m = readout(ra, square(100, 0.05))
    if m:
        value = 0.3
    else:
        value = 0.1
    drive(db, square(8, value))


def test_rcp_waits_for_an_outcome_a_pulse_computes_with_to_reach_its_controller(monkeypatch):
    monkeypatch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)
    qpu = _forwarding()

    def gap(b, schedule="rcp"):
        lowered = pulse.compile(chosen_across, schedule=schedule, quantum_system=qpu).lower(
            qpu.qudits[0], qpu.qudits[b]
        )
        readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
        return drive.start - readout.end

    assert gap(1) == 6
    assert gap(2) == 16
    # Every schedule waits as long.
    assert gap(1, "asap") == 6
    assert gap(2, "asap") == 16
    # Scheduled for the same controllers without the latencies, the check
    # names it.
    from cudaq_pulse.controller import _compile_check

    quick = [
        dataclasses.replace(c, feedback_latency_cycles=0, peer_latency_cycles=0)
        for c in qpu.control_system.controllers
    ]
    near = _device(system=pulse.ControlSystem(controllers=quick), clock_ghz=2.0)
    early = pulse.compile(chosen_across, schedule="asap", quantum_system=near).lower(
        near.qudits[0], near.qudits[2]
    )
    [message], _ = _compile_check(early, None, qpu.control_system)
    assert "computes with the readout of controllers" in message


@pulse.kernel
def two_drives(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(8, 0.1))
    drive(db, square(8, 0.1))


def test_rcp_keeps_pulses_of_two_controllers_on_one_lane_apart_by_their_start_uncertainty():
    def second_start(b, reference=None):
        system = _system(1.0, 0.5, max_concurrent_drives=1, reference_clock_ghz=reference)
        qpu = _device(system=system, clock_ghz=2.0)
        lowered = pulse.compile(two_drives, schedule="rcp", quantum_system=qpu).lower(
            qpu.qudits[0], qpu.qudits[b]
        )
        first, second = (e for e in lowered.schedule if e.kind == "drive")
        assert first.start == 0
        return second.start

    # One drive at a time: the second, on controllers[1], also waits the 2 + 4
    # program cycles the two starts may be apart, then its grid of 4.
    assert second_start(2) == 16
    assert second_start(2, reference=0.5) == 8
    # On the same controller, right after.
    assert second_start(1) == 8


@pulse.kernel
def paired(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    sync(da, db)
    drive(da, square(8, 0.1))
    wait(db, 4)
    drive(db, square(8, 0.1))


def test_pulses_a_sync_lines_up_across_controllers_land_within_their_alignment_tolerance():
    def compiled(tolerance=0, reference=None, skew=0, delays=None):
        controllers = [
            pulse.Controller(
                drive_lines=[0, 1],
                readout_lines=[0, 1],
                clock_ghz=1.0,
                alignment_tolerance_cycles=tolerance,
            ),
            pulse.Controller(
                drive_lines=[2, 3],
                readout_lines=[2, 3],
                clock_ghz=0.5,
                alignment_tolerance_cycles=tolerance,
                clock_skew_cycles=skew,
            ),
        ]
        system = pulse.ControlSystem(
            controllers=with_line_delays(controllers, delays or {}), reference_clock_ghz=reference
        )
        qpu = _device(system=system, clock_ghz=2.0)
        return pulse.compile(paired, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])

    with pytest.raises(ControllerLimitError) as caught:
        compiled()
    first, second = caught.value.violations
    assert first == (
        "controllers[0]: the sync at 0 lines up the drive on line 0 with the pulses of "
        "controllers[1], but they land within 1 of its cycles of where they are meant to "
        "(1 without a shared reference_clock_ghz), more than its alignment_tolerance_cycles "
        "of 0; share a reference_clock_ghz between them, or raise its "
        "alignment_tolerance_cycles to 1"
    )
    # A wait before a pulse leaves it lined up.
    assert second.startswith("controllers[1]: the sync at 0 lines up the drive on line 1 ")
    compiled(tolerance=1)
    compiled(reference=0.5)
    with pytest.raises(ControllerLimitError, match="2 of clock_skew_cycles"):
        compiled(reference=0.5, skew=2, tolerance=1)
    # A line delay rounded to its grid (0.4 ns of a 2 ns step: 0.2 of a 0.5
    # GHz cycle) adds to it.
    with pytest.raises(ControllerLimitError, match="raise its alignment_tolerance_cycles to 2"):
        compiled(tolerance=1, delays={("drive", 2): 2.4})
