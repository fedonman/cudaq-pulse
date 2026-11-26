# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clocks: the quantum system's program clock, a controller's own clock and the grid
it puts its lines on, ``round_duration``, the reference clock and the exports in a
controller's own cycles."""

from __future__ import annotations

import dataclasses
import json
import math
from fractions import Fraction

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import Controller, ControlSystem, _clocks
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import ControllerLimitError, check, sample_waveform

from tests._handles import LADDER, Q, ladder_target

SCHEDULES = ("asap", "alap", "rcp", "alap_rcp")


def _machine(clock_ghz: float, **kwargs):
    return ControlSystem(controllers=[Controller(clock_ghz=clock_ghz)], **kwargs)


# ── The program clock ────────────────────────────────────────────────────────


def test_a_config_sets_the_program_clock_and_the_fingerprint_follows():
    default = ladder_target(4)
    same = ladder_target(4, clock_ghz=2.0)
    slow = ladder_target(4, clock_ghz=1.0)
    assert default.clock_ghz == same.clock_ghz == 2.0 and default == same
    assert default.fingerprint == same.fingerprint
    assert slow.clock_ghz == 1.0 and slow._description.clock_ghz == 1.0
    assert slow.fingerprint != default.fingerprint


@pytest.mark.parametrize("bad", [0, -1.0, "2", True, math.inf])
def test_a_config_clock_must_be_a_positive_number(bad):
    with pytest.raises(ValueError, match="clock_ghz"):
        ladder_target(4, clock_ghz=bad)


def test_the_program_clock_comes_from_controllers_that_divide_evenly():
    assert _clocks.program_clock_ghz(None, {}, 2.0) == 2.0
    assert _clocks.program_clock_ghz(None, {"a": 2.0, "b": 1.0, "c": 0.5}, 9.0) == 2.0
    assert _clocks.program_clock_ghz(1.5, {"a": 3.0, "b": 0.5}, 9.0) == 1.5


def test_uneven_controller_clocks_need_a_declared_program_clock():
    with pytest.raises(ValueError) as caught:
        _clocks.program_clock_ghz(None, {"awg": 2.4, "readout": 1.0}, 2.0)
    message = str(caught.value)
    assert "set 'clock_ghz'" in message
    # Every clock that works, never one picked for the user.
    assert "12 GHz, which every controller's clock divides" in message
    assert "2.4 GHz (awg's; then readout every 12 program cycles)" in message
    assert "1 GHz (readout's; then awg every 5 program cycles)" in message


def test_a_clock_without_a_whole_cycle_ratio_is_an_error():
    with pytest.raises(ValueError, match="no whole-cycle ratio"):
        _clocks.clock_ratio(math.pi, 2.0)
    assert _clocks.clock_ratio(0.5, 2.0) == Fraction(1, 4)
    assert _clocks.clock_ratio(2.4, 1.0).denominator == 5


# ── round_duration ───────────────────────────────────────────────────────────


@pulse.kernel
def rounded(q):
    d = get_drive_line(q)
    drive(d, square(pulse.round_duration(d, 37), 0.1))
    drive(d, square(pulse.round_duration(d, 37, "down"), 0.1))
    drive(d, square(pulse.round_duration(d, 37, direction=pulse.RoundDirection.NEAREST), 0.1))
    drive(d, square(pulse.round_duration(d, 1.25 * pulse.ns), 0.1))
    drive(d, square(pulse.round_duration(d, 21 * pulse.ns, pulse.RoundDirection.DOWN), 0.1))


def _durations(lowered):
    return [e.duration_cycles for e in lowered.schedule if e.kind == "drive"]


def test_round_duration_rounds_onto_the_line_grid():
    def durations(machine=None):
        return _durations(pulse.compile(rounded, passes=[], control_system=machine).lower(Q[0]))

    assert durations() == [37, 37, 37, 3, 42]
    # A 0.5 GHz controller starts every 4 cycles of the 2 GHz program clock.
    assert durations(_machine(0.5)) == [40, 36, 36, 4, 40]
    # A faster controller times every program cycle.
    assert durations(_machine(4.0)) == [37, 37, 37, 3, 42]
    assert pulse.RoundDirection("nearest") is pulse.RoundDirection.NEAREST
    assert not hasattr(pulse, "cycles_on")


@pytest.mark.parametrize(
    ("member", "text", "cycles"),
    [
        (pulse.RoundDirection.UP, "up", 40),
        (pulse.RoundDirection.DOWN, "down", 36),
        (pulse.RoundDirection.NEAREST, "nearest", 36),
    ],
)
def test_each_round_direction_is_its_string(member, text, cycles):
    assert list(pulse.RoundDirection) == [
        pulse.RoundDirection.UP,
        pulse.RoundDirection.DOWN,
        pulse.RoundDirection.NEAREST,
    ]
    assert member == text and str(member) == text and pulse.RoundDirection(text) is member

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(pulse.round_duration(d, 37, member), 0.1))
        drive(d, square(pulse.round_duration(d, 37, text), 0.1))

    assert _durations(pulse.compile(k, passes=[], control_system=_machine(0.5)).lower(Q[0])) == [
        cycles,
        cycles,
    ]


def test_round_duration_needs_a_line_and_a_known_length():
    @pulse.kernel
    def specialized(q, n: float):
        d = get_drive_line(q)
        drive(d, square(pulse.round_duration(d, n), 0.1))

    @pulse.kernel
    def not_a_line(q):
        d = get_drive_line(q)
        drive(d, square(pulse.round_duration(d.tone, 4), 0.1))

    @pulse.kernel
    def a_number(q):
        d = get_drive_line(q)
        drive(d, square(pulse.round_duration(3, 4), 0.1))

    @pulse.kernel
    def bad_mode(q):
        d = get_drive_line(q)
        drive(d, square(pulse.round_duration(d, 4, "sideways"), 0.1))

    # Like a duration, a float that reaches it is specialized on.
    with pytest.warns(RuntimeWarning, match="round_duration"):
        lowered = pulse.compile(specialized, control_system=_machine(0.5)).lower(Q[0], 13.0)
    assert _durations(lowered) == [16]
    with pytest.raises(pulse.CompilationError, match="takes a line"):
        pulse.compile(not_a_line).lower(Q[0])
    with pytest.raises(pulse.CompilationError, match="takes a line"):
        pulse.compile(a_number).lower(Q[0])
    with pytest.raises(pulse.CompilationError, match="up, down, nearest"):
        pulse.compile(bad_mode).lower(Q[0])
    with pytest.raises(TypeError, match="inside a @cudaq_pulse.kernel"):
        pulse.round_duration(None, 4)


# ── Scheduling on the grid ───────────────────────────────────────────────────


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
def test_every_operation_starts_and_ends_on_its_controllers_grid(schedule):
    lowered = pulse.compile(mixed, schedule=schedule, control_system=_machine(0.5)).lower(
        Q[0], Q[1]
    )
    events = [e for e in lowered.schedule if e.kind in ("drive", "readout", "wait")]
    assert events and all(e.start % 4 == 0 and e.end % 4 == 0 for e in events)
    assert "controllers = [#pulse.controller<grid_cycles = 4>]" in lowered.mlir()
    # The program clock runs on; only the starts moved.
    plain = pulse.compile(mixed, schedule=schedule).lower(Q[0], Q[1])
    assert len(plain.schedule) == len(lowered.schedule)


def test_a_sync_meets_on_the_grid():
    lowered = pulse.compile(mixed, schedule="asap", control_system=_machine(0.5)).lower(Q[0], Q[1])
    drives = [e for e in lowered.schedule if e.kind == "drive" and e.line == 0]
    # Line 0 is busy until 40; the sync meets there, on the grid.
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
    drive(da, square(4, 0.1))
    callee(a)
    callee(b)


@pytest.mark.parametrize("schedule", ("asap", "alap"))
def test_a_kernel_with_its_own_schedule_is_placed_on_the_grid(schedule):
    lowered = pulse.compile(calls_a_block, schedule=schedule, control_system=_machine(0.5)).lower(
        Q[0], Q[1]
    )
    assert all(e.start % 4 == 0 for e in lowered.schedule if e.kind == "drive")


@pulse.classical(max_latency_ns=60, link="controller")
def correction(bits: pulse.Bits) -> float:
    return 0.1 * bits[0]


@pulse.kernel
def with_a_call(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    amplitude = correction.submit([m]).result()
    drive(d, square(8, amplitude))


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_classical_calls_keep_lines_on_the_grid(schedule):
    machine = _machine(0.5, links={"controller": Link(fixed_ns=10)})
    lowered = pulse.compile(with_a_call, schedule=schedule, control_system=machine).lower(Q[0])
    assert all(e.start % 4 == 0 for e in lowered.schedule if e.kind in ("drive", "readout"))
    program = lowered.export_to_controller()
    assert program["clock_ghz"] == 0.5 and program["program_clock_ghz"] == 2.0
    report = lowered.classical_report()
    call = program["calls"][0]
    assert call["submit_cycles"] == -(-report[0].submit_cycles // 4)
    assert call["due_cycles"] == -(-report[0].due_cycles // 4)


# ── Off the grid ─────────────────────────────────────────────────────────────


@pulse.kernel
def off_grid(q):
    d = get_drive_line(q)
    drive(d, gaussian(41, 0.1, 10.0))
    wait(d, 5)


def test_a_length_off_the_grid_fails_to_compile_naming_the_line_and_the_fix():
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(off_grid, control_system=_machine(1.0)).lower(Q[3])
    drive, wait = caught.value.violations
    assert drive == (
        "line d3 moves in timesteps of 2 cycles; 41 cycles isn't a whole number of "
        "timesteps (the drive at cycle 0; pulse.round_duration(line, 41) gives 42, or "
        "40 rounding down)"
    )
    assert wait.startswith("line d3 moves in timesteps of 2 cycles; 5 cycles isn't")
    assert "(the wait at cycle 42; pulse.round_duration(line, 5) gives 6" in wait
    # A controller that starts every cycle takes it.
    assert pulse.compile(off_grid, control_system=_machine(2.0)).lower(Q[3]).schedule


def test_check_against_a_slower_controller_reports_what_is_off_its_grid():
    lowered = pulse.compile(off_grid).lower(Q[0])
    found = check(lowered, Controller(clock_ghz=1.0))
    assert len(found) == 2 and all("moves in timesteps of 2 cycles" in f for f in found)
    assert "(the wait at cycle 41;" in found[1]
    assert (
        "the wait starts at cycle 41, which isn't a whole number of timesteps "
        "(the nearest starts are 40 and 42)"
    ) in found[1]
    assert check(lowered, Controller(clock_ghz=4.0)) == []


@pulse.kernel
def variable_wait(q, tau: float):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    wait(d, tau * 2.0, bounds=(0, 1000))
    drive(d, square(8, 0.1))


@pulse.classical(max_latency_ns=60, link="controller")
def wait_length(x: float) -> float:
    return x


@pulse.kernel
def answer_wait_on_grid(q):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    wait(d, wait_length.submit(1.0).result(), bounds=(8, 400))
    drive(d, square(8, 0.1))


@pulse.kernel
def answer_wait_off_grid(q):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    wait(d, wait_length.submit(1.0).result(), bounds=(6, 402))
    drive(d, square(8, 0.1))


def test_a_wait_from_an_answer_has_its_bounds_on_every_lines_grid():
    links = {"controller": Link(fixed_ns=10)}
    machine = _machine(0.5, links=links)
    assert pulse.compile(answer_wait_on_grid, control_system=machine).lower(Q[0]).schedule
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(answer_wait_off_grid, control_system=machine).lower(Q[0])
    message = str(caught.value)
    assert message.startswith("wait() on line d0: its length comes from")
    assert "every later operation on every line moves by it" in message
    assert "move in timesteps of 4 cycles; its bounds (6, 402) aren't whole" in message
    assert "for 6, use 4 or 8; for 402, use 400 or 404" in message
    # On the program clock every bound is on the grid.
    assert pulse.compile(answer_wait_off_grid, control_system=ControlSystem(links=links)).lower(
        Q[0]
    )


def test_a_variable_wait_off_the_grid_fails_at_the_call():
    compiled = pulse.compile(variable_wait, control_system=_machine(0.5))
    lowered = compiled.lower(Q[0], 6.0)
    assert [e.start for e in lowered.schedule if e.kind == "drive"] == [0, 20]
    with pytest.raises(ValueError, match="nearest lengths are 4 and 8"):
        compiled.lower(Q[0], 3.0)


# ── Exports in the controller's own cycles ───────────────────────────────────


def test_the_export_counts_the_controllers_own_cycles():
    lowered = pulse.compile(mixed, schedule="asap", control_system=_machine(0.5)).lower(Q[0], Q[1])
    program = lowered.export_to_controller()
    assert program["clock_ghz"] == 0.5 and program["program_clock_ghz"] == 2.0
    assert "reference_clock_ghz" not in program
    timed = ("drive", "readout", "wait")
    played = [e for e in program["events"] if e["kind"] in timed]
    assert [(e["start_cycles"], e["duration_cycles"], e["end_cycles"]) for e in played] == [
        (e.start // 4, e.duration_cycles // 4, e.end // 4)
        for e in lowered.schedule
        if e.kind in timed
    ]
    params = played[0]["waveform"]["params"]
    assert params == {"duration": 10.0, "amplitude": 0.1, "sigma": 2.5}
    json.dumps(program)


def test_samples_at_a_rate_of_the_controllers_clock():
    lowered = pulse.compile(mixed, schedule="asap", control_system=_machine(0.5)).lower(Q[0], Q[1])
    program = lowered.export_to_controller(sample_rate_ghz=1.0)
    first = next(e for e in program["events"] if e["kind"] == "drive")
    # Two samples per own cycle: the envelope at every other program cycle.
    samples = np.array(first["samples"]["re"])
    assert len(samples) == 2 * first["duration_cycles"] == 20
    node = {"shape": "gaussian", "params": {"duration": 40.0, "amplitude": 0.1, "sigma": 10.0}}
    np.testing.assert_allclose(samples, sample_waveform(node, 40).real[::2], rtol=1e-12)
    with pytest.raises(ValueError, match="whole multiple"):
        lowered.export_to_controller(sample_rate_ghz=0.75)


def test_the_emulation_export_counts_the_controllers_own_cycles():
    lowered = pulse.compile(mixed, schedule="asap", control_system=_machine(0.5)).lower(Q[0], Q[1])
    data = lowered.export_to_emulator()
    assert data["clock_ghz"] == 0.5 and data["program_clock_ghz"] == 2.0
    starts = [e["start_cycles"] for e in data["events"]]
    assert starts == [e.start // 4 for e in lowered.schedule]


def test_every_waveform_time_is_scaled():
    from cudaq_pulse.controller import _PARAMETERS

    times = {"duration", "sigma", "beta", "risefall"}
    shapes = {s for s, names in _PARAMETERS.items() if "duration" in names}
    assert set(_clocks._TIMES_OF) == shapes
    for shape in shapes:
        assert _clocks._TIMES_OF[shape] == tuple(k for k in _PARAMETERS[shape] if k in times)


def test_every_time_in_the_exports_is_converted():
    from tests.classical import _schedule_matrix as matrix

    known = {
        *_clocks.EVENT_TIMES,
        *_clocks.EVENT_BOUNDS,
        *_clocks.EVENT_LOWER_BOUNDS,
        *_clocks.EMULATION_EVENT_TIMES,
        *_clocks.SEGMENT_TIMES,
        *_clocks.SEGMENT_LOWER_BOUNDS,
        *_clocks.CALL_TIMES,
    }
    found = set()

    def walk(data):
        if isinstance(data, dict):
            found.update(k for k in data if k.endswith("_cycles"))
            for key, value in data.items():
                if key != "limits":  # the controller's own, as given
                    walk(value)
        elif isinstance(data, list):
            for value in data:
                walk(value)

    machine = dataclasses.replace(matrix.MACHINE, controllers=[Controller(clock_ghz=1.0)])
    for case in matrix.all_cases()[::7]:
        if matrix.expected_error(case) is not None:
            continue
        try:
            lowered = matrix.lower_case(case, Q, control_system=machine)
        except (ControllerLimitError, ValueError):
            continue
        walk(lowered.export_to_controller())
        walk(lowered.export_to_emulator())
    assert "submit_cycles" in found and "min_shift_cycles" in found
    assert found <= known


def test_a_single_clock_export_is_unchanged():
    lowered = pulse.compile(mixed, control_system=_machine(2.0)).lower(Q[0], Q[1])
    program = lowered.export_to_controller()
    assert program["clock_ghz"] == 2.0
    assert "program_clock_ghz" not in program and "reference_clock_ghz" not in program
    assert "grid_cycles" not in lowered.mlir()


def test_a_faster_controller_counts_more_cycles():
    lowered = pulse.compile(off_grid, control_system=_machine(4.0)).lower(Q[0])
    program = lowered.export_to_controller()
    assert program["clock_ghz"] == 4.0 and program["program_clock_ghz"] == 2.0
    drive = next(e for e in program["events"] if e["kind"] == "drive")
    assert drive["duration_cycles"] == 82 and drive["waveform"]["params"]["duration"] == 82.0
    assert "grid_cycles" not in lowered.mlir()


def test_a_run_time_wait_has_its_bounds_in_the_controllers_cycles():
    from tests.test_controller import answered

    for clock_ghz, bounds in ((None, (8, 400)), (1.0, (4, 200)), (4.0, (16, 800))):
        machine = ControlSystem() if clock_ghz is None else _machine(clock_ghz)
        program = pulse.compile(answered, control_system=machine).lower(Q[0]).export_to_controller()
        wait = next(e for e in program["events"] if e["kind"] == "wait")
        assert (wait["min_cycles"], wait["max_cycles"]) == bounds


@pulse.classical(max_latency_ns=200)
def settings_of(bits: pulse.Bits) -> tuple[float, float]:
    return 0.1 * bits[0], 40.0 + bits[0]


@pulse.kernel
def timed_at_run_time(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    m = readout(ro, square(100, 0.05))
    amp, delay = settings_of.submit([m]).result()
    wait(d, delay, bounds=(8, 400))
    drive(d, gaussian(40, 0.1, delay * 0.25))
    drive(d, drag(40, amp, 8.0, delay))


def _answer_of(program):
    return {"answer": next(e for e in program["events"] if e["kind"] == "answer")["index"]}


def test_what_a_program_computes_at_run_time_is_in_the_controllers_cycles():
    machine = _machine(1.0)
    program = (
        pulse.compile(timed_at_run_time, control_system=machine).lower(Q[0]).export_to_controller()
    )
    answer = _answer_of(program)
    delay = {**answer, "result": 1}
    wait = next(e for e in program["events"] if e["kind"] == "wait")
    gaussian_, drag_ = (e for e in program["events"] if e["kind"] == "drive")
    in_program_cycles = {
        "op": "to_int",
        "type": "float",
        "of": [{"op": "add", "type": "float", "of": [delay, {"const": 0.5}]}],
    }
    # The length clamped as the run clamps it, then in own cycles: 1 for 2.
    clamped = {
        "op": "min",
        "type": "int",
        "of": [
            {"op": "max", "type": "int", "of": [in_program_cycles, {"const": 8}]},
            {"const": 400},
        ],
    }
    assert json.loads(json.dumps(wait["length"])) == {
        "op": "controller_cycles",
        "type": "int",
        "of": [clamped],
        "num": 1,
        "den": 2,
        "timestep": 2,
    }
    assert (wait["min_cycles"], wait["max_cycles"]) == (4, 200)
    # Time operands scale by the ratio as the fixed ones do; others don't.
    assert gaussian_["waveform"]["params"]["duration"] == 20.0
    assert gaussian_["waveform"]["runtime"] == {
        "sigma": {
            "op": "mul",
            "type": "float",
            "of": [{"op": "mul", "type": "float", "of": [delay, {"const": 0.25}]}, {"const": 0.5}],
        }
    }
    assert drag_["waveform"]["params"]["sigma"] == 4.0
    assert drag_["waveform"]["runtime"] == {
        "amplitude": {**answer, "result": 0},
        "beta": {"op": "mul", "type": "float", "of": [delay, {"const": 0.5}]},
    }
    # On the program clock, nothing changes.
    plain = pulse.compile(timed_at_run_time).lower(Q[0]).export_to_controller()
    wait = next(e for e in plain["events"] if e["kind"] == "wait")
    assert wait["length"] == in_program_cycles
    assert (wait["min_cycles"], wait["max_cycles"]) == (8, 400)
    assert next(e for e in plain["events"] if e["kind"] == "drive")["waveform"]["runtime"] == {
        "sigma": {"op": "mul", "type": "float", "of": [delay, {"const": 0.25}]}
    }


def test_a_clock_that_only_steps_checks_a_run_time_length_on_its_timesteps():
    # The ratio is 1, so only the timestep check is added; times keep their
    # values.
    machine = ControlSystem(controllers=[Controller(clock_ghz=2.0, cycles_per_timestep=4)])
    program = (
        pulse.compile(timed_at_run_time, control_system=machine).lower(Q[0]).export_to_controller()
    )
    wait = next(e for e in program["events"] if e["kind"] == "wait")
    assert wait["length"]["op"] == "controller_cycles"
    assert (wait["length"]["num"], wait["length"]["den"], wait["length"]["timestep"]) == (1, 1, 4)
    assert (wait["min_cycles"], wait["max_cycles"]) == (8, 400)
    drive = next(e for e in program["events"] if e["kind"] == "drive")
    assert drive["waveform"]["runtime"]["sigma"]["of"][1] == {"const": 0.25}


def _line_event(**parts):
    return {"index": 3, "kind": "drive", "line": 0, **parts}


_LINE = {"role": "drive", "qudit": 0}
_HALF = _clocks.ControllerClock("awg", 1.0, Fraction(1, 2))


def test_a_run_time_length_known_now_to_be_off_the_timesteps_is_a_compile_error():
    select = {
        "op": "select",
        "type": "int",
        "of": [
            {"op": "ne", "type": "int", "of": [{"readout": 1}, {"const": 0}]},
            {"const": 9},
            {"const": 12},
        ],
    }
    event = _line_event(kind="wait", length=select, min_cycles=None, max_cycles=None)
    [message] = _clocks.own_run_time(_HALF, _LINE, event)
    assert "line d0, on the awg at 1 GHz: the wait that is event 3" in message
    assert "length, computed at run time, of 9 cycles" in message
    assert "timesteps of 2 cycles" in message
    event = _line_event(
        waveform={"shape": "square", "params": {"duration": None}, "runtime": {"duration": select}}
    )
    [message] = _clocks.own_run_time(_HALF, _LINE, event)
    assert "waveform's duration, computed at run time, of 9 cycles" in message
    # On the timesteps for every outcome, or not known now: converted.
    eights = {**select, "of": [select["of"][0], {"const": 8}, {"const": 8}]}
    event = _line_event(kind="wait", length=eights)
    assert _clocks.own_run_time(_HALF, _LINE, event) == []
    event = _line_event(kind="wait", length={"answer": 0, "result": 0})
    assert _clocks.own_run_time(_HALF, _LINE, event) == []
    assert event["length"]["op"] == "controller_cycles"


def test_a_run_time_length_that_names_terms_keeps_them_at_its_root():
    select = {
        "op": "select",
        "type": "int",
        "of": [
            {"op": "ne", "type": "int", "of": [{"readout": 1}, {"const": 0}]},
            {"const": 9},
            {"const": 12},
        ],
    }
    twice = {"op": "add", "type": "int", "of": [{"term": 0}, {"term": 0}]}
    event = _line_event(
        kind="wait", length={"terms": [select], **twice}, min_cycles=4, max_cycles=40
    )
    # Each use of the term takes either value: 18, 21 or 24 cycles.
    [message] = _clocks.own_run_time(_HALF, _LINE, event)
    assert "length, computed at run time, of 21 cycles" in message
    length = event["length"]
    assert list(length)[:2] == ["terms", "op"]
    assert length["terms"] == [select]
    assert length["op"] == "controller_cycles"
    assert "terms" not in str(length["of"])
    eights = {**select, "of": [select["of"][0], {"const": 8}, {"const": 8}]}
    event = _line_event(kind="wait", length={"terms": [eights], **twice})
    assert _clocks.own_run_time(_HALF, _LINE, event) == []
    assert pulse.controller.evaluate_expression(event["length"], readouts={1: 1}) == 8
    event = _line_event(
        waveform={
            "shape": "square",
            "params": {"duration": None},
            "runtime": {"duration": {"terms": [select], **twice}},
        }
    )
    [message] = _clocks.own_run_time(_HALF, _LINE, event, True)
    assert "waveform's duration, computed at run time, of 21 cycles" in message
    duration = event["waveform"]["runtime"]["duration"]
    assert list(duration)[:2] == ["terms", "op"]
    assert duration["op"] == "controller_cycles"
    # A chain deeper than Python nests calls.
    deep = {"const": 4}
    for _ in range(3000):
        deep = {"op": "add", "type": "int", "of": [deep, {"const": 2}]}
    assert _clocks._values(deep) == {6004}


def test_a_run_time_operand_the_export_cannot_tell_is_a_time_is_a_compile_error():
    def unnamed():
        computed = {"0": {"answer": 0, "result": 0}}
        return {"shape": "chirp", "values": [None, 0.1], "runtime": computed}

    event = _line_event(waveform=unnamed())
    [message] = _clocks.own_run_time(_HALF, _LINE, event)
    assert "plays a 'chirp' waveform whose operand 0 is computed at run time" in message
    assert "compute it while the kernel compiles" in message
    # Left as it was, never exported in the wrong cycles.
    assert event["waveform"] == unnamed()
    # Where the ratio is 1 there is nothing to convert.
    steps = _clocks.ControllerClock("awg", 2.0, Fraction(1), grid=4)
    assert _clocks.own_run_time(steps, _LINE, _line_event(waveform=unnamed())) == []


def test_a_run_time_violation_fails_the_export_and_the_check(monkeypatch):
    lowered = pulse.compile(timed_at_run_time, control_system=_machine(1.0)).lower(Q[0])
    monkeypatch.setattr(_clocks, "own_run_time", lambda *args: ["not convertible"])
    with pytest.raises(ControllerLimitError, match="not convertible"):
        lowered.export_to_controller()
    assert "not convertible" in check(lowered, Controller(clock_ghz=1.0))


def test_lines_on_several_controllers_convert_what_they_compute_at_run_time():
    plan = _clocks.ClockPlan(
        2.0,
        (
            _clocks.ControllerClock("awg", 2.0, Fraction(1)),
            _clocks.ControllerClock("readout", 1.0, Fraction(1, 2)),
        ),
        owners={("readout", 0): 1},
    )
    lines = [{"role": "drive", "qudit": 0}, {"role": "readout", "qudit": 0}]
    length = {"answer": 0, "result": 0}

    def events():
        return [
            {
                "index": 0,
                "kind": "wait",
                "line": i,
                "length": dict(length),
                "min_cycles": 8,
                "max_cycles": 40,
                "start_cycles": 0,
                "duration_cycles": None,
                "end_cycles": None,
            }
            for i in (0, 1)
        ]

    converted = events()
    assert _clocks.off_grid(plan, lines, converted, True, True, True) == []
    assert converted[0]["length"] == length and converted[0]["max_cycles"] == 40
    assert converted[1]["length"]["op"] == "controller_cycles"
    assert (converted[1]["min_cycles"], converted[1]["max_cycles"]) == (4, 20)
    # A program that computes nothing at run time keeps its bounds.
    kept = events()
    assert _clocks.off_grid(plan, lines, kept, True, True) == []
    assert kept == events()


# ── The reference clock ──────────────────────────────────────────────────────


def test_a_reference_clock_divides_every_controller_clock_and_is_exported():
    machine = _machine(0.5, reference_clock_ghz=0.25)
    program = pulse.compile(mixed, control_system=machine).lower(Q[0], Q[1]).export_to_controller()
    assert program["reference_clock_ghz"] == 0.25
    machine = ControlSystem(controllers=[Controller()], reference_clock_ghz=0.5)
    program = pulse.compile(mixed, control_system=machine).lower(Q[0], Q[1]).export_to_controller()
    assert program["reference_clock_ghz"] == 0.5 and "program_clock_ghz" not in program
    with pytest.raises(ValueError, match="not a whole multiple of the 0.3 GHz reference"):
        pulse.compile(mixed, control_system=_machine(0.5, reference_clock_ghz=0.3)).lower(
            Q[0], Q[1]
        )
    with pytest.raises(ValueError, match="must be positive"):
        ControlSystem(reference_clock_ghz=0.0)
    assert ControlSystem(reference_clock_ghz=0.5) != ControlSystem()


def test_without_a_reference_each_controller_start_is_known_to_one_own_cycle():
    plan = _clocks.ClockPlan(
        2.0,
        (
            _clocks.ControllerClock("awg", 2.0, Fraction(1)),
            _clocks.ControllerClock("readout", 0.5, Fraction(1, 4)),
        ),
    )
    assert plan.start_uncertainty_ns() == {"awg": 0.5, "readout": 2.0}
    assert plan.budget_uncertainty_ns("awg", "readout") == 2.5
    assert plan.budget_uncertainty_ns("awg", "awg") == 0.0
    assert plan.start_uncertainty_ns({"readout": 1}) == {"awg": 0.5, "readout": 4.0}
    shared = _clocks.ClockPlan(2.0, plan.controllers, reference_clock_ghz=0.5)
    assert shared.budget_uncertainty_ns("awg", "readout") == 0.0
    assert shared.start_uncertainty_ns({"readout": 1}) == {"awg": 0.0, "readout": 2.0}


# ── Sampled waveforms ────────────────────────────────────────────────────────


def ramp(t):
    return 0.1 + 0.01 * t


@pulse.kernel
def shaped(q):
    d = get_drive_line(q)
    drive(d, custom(8, ramp))


@pulse.kernel
def shaped_off_grid(q):
    d = get_drive_line(q)
    drive(d, custom(9, ramp))


def _played(lowered):
    return next(e for e in lowered.export_to_controller()["events"] if e["kind"] == "drive")


def test_custom_samples_the_envelope_at_the_lines_rate():
    # One sample per program cycle, at the middle of each...
    drive = _played(pulse.compile(shaped).lower(Q[0]))
    assert drive["waveform"]["samples"] == pytest.approx([ramp(k + 0.5) for k in range(8)])
    # ...one per cycle of a controller at half the clock, at t = 1, 3, 5, 7...
    lowered = pulse.compile(shaped, control_system=_machine(1.0)).lower(Q[0])
    assert _durations(lowered) == [8]
    assert "rate = array<i64: 1, 2>" in lowered.mlir()
    drive = _played(lowered)
    assert drive["duration_cycles"] == 4
    assert drive["waveform"]["samples"] == pytest.approx([ramp(t) for t in (1, 3, 5, 7)])
    # ...and two per cycle on a controller that plays two.
    machine = ControlSystem(controllers=[Controller(sample_rate_ghz=4.0)])
    drive = _played(pulse.compile(shaped, control_system=machine).lower(Q[0]))
    assert drive["duration_cycles"] == 8
    expected = [ramp((k + 0.5) / 2) for k in range(16)]
    assert drive["waveform"]["samples"] == pytest.approx(expected)
    with pytest.raises(pulse.CompilationError, match="a multiple of 2 cycles"):
        pulse.compile(shaped_off_grid, control_system=_machine(1.0)).lower(Q[0])


HELD = [ramp(t) for t in (1, 1, 3, 3, 5, 5, 7, 7)]


@pulse.kernel
def held_for_two_cycles(q):
    d = get_drive_line(q)
    drive(d, custom_samples(HELD))


def test_a_custom_envelope_is_simulated_at_its_controllers_rate():
    # Each sample at half the clock is held for two cycles of the program's.
    lowered = pulse.compile(shaped, control_system=_machine(1.0)).lower(Q[0])
    held = pulse.compile(held_for_two_cycles).lower(Q[0])
    a = pulse.simulate(lowered).density_matrix
    b = pulse.simulate(held).density_matrix
    np.testing.assert_allclose(a, b, atol=1e-12)


HALF_RATE = [ramp(t) for t in (1, 3, 5, 7)]


@pulse.kernel
def at_half_rate(q):
    d = get_drive_line(q)
    drive(d, custom_samples(HALF_RATE, 1.0))


@pulse.kernel
def at_program_rate(q):
    d = get_drive_line(q)
    drive(d, custom_samples([0.1, 0.2, 0.3, 0.4]))


@pulse.kernel
def at_double_rate(q):
    d = get_drive_line(q)
    drive(d, custom_samples([0.1, 0.2, 0.3], sample_rate_ghz=4.0))


def test_sample_arrays_play_only_at_the_rate_they_were_made_for():
    # Made for the program clock, they play on it...
    assert _durations(pulse.compile(at_program_rate).lower(Q[0])) == [4]
    # ...and nowhere else: never as a longer or shorter pulse.
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(at_program_rate, control_system=_machine(1.0)).lower(Q[0])
    message = str(caught.value)
    assert message.startswith(
        "custom_samples(): 4 samples made for 2 GHz (one sample per clock cycle) on lines "
        "that play 1 GHz (one sample every 2 clock cycles)"
    )
    assert "pass sample_rate_ghz=1" in message and "custom(duration, f)" in message
    machine = ControlSystem(controllers=[Controller(sample_rate_ghz=4.0)])
    with pytest.raises(pulse.CompilationError, match="lines that play 4 GHz"):
        pulse.compile(at_program_rate, control_system=machine).lower(Q[0])
    # Made for the line's rate, they last their samples at it.
    lowered = pulse.compile(at_half_rate, control_system=_machine(1.0)).lower(Q[0])
    assert _durations(lowered) == [8] and _played(lowered)["duration_cycles"] == 4
    with pytest.raises(pulse.CompilationError, match="made for 1 GHz .* lines that play 2 GHz"):
        pulse.compile(at_half_rate).lower(Q[0])
    # Two per cycle, an odd count does not fill whole cycles.
    with pytest.raises(pulse.CompilationError, match="give a multiple of 2 samples"):
        pulse.compile(at_double_rate, control_system=machine).lower(Q[0])


def test_check_against_another_rate_reports_the_sample_arrays():
    lowered = pulse.compile(at_program_rate).lower(Q[0])
    found = check(lowered, Controller(clock_ghz=1.0))
    assert found == [
        "line 0: the drive at 0 plays samples made for 2 GHz, and the controller's lines play 1 GHz"
    ]
    assert check(lowered, Controller(instruction_cycles=2)) == []


# ── Simulation ───────────────────────────────────────────────────────────────


@pulse.kernel
def product(q):
    d = get_drive_line(q)
    drive(d, wf_mul(gaussian(40, 0.2, 10.0), square(40, 0.5)))


def test_a_product_is_simulated_at_its_controllers_clock():
    fast = pulse.compile(product).lower(Q[0])
    slow = pulse.compile(product, control_system=_machine(0.5)).lower(Q[0])
    llvm_fast, llvm_slow = fast.mlir("llvm"), slow.mlir("llvm")
    assert llvm_fast != llvm_slow
    a = pulse.simulate(fast).density_matrix
    b = pulse.simulate(slow).density_matrix
    # Ten samples instead of forty of one smooth envelope: close, not equal.
    assert not np.array_equal(a, b) and np.allclose(a, b, atol=5e-3)


def test_the_ladder_device_is_untouched():
    assert LADDER.clock_ghz == 2.0
