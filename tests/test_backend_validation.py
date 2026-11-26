# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validating a controller backend against the export test corpus."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, _validation, backends
from cudaq_pulse.controller import _EVENT_KINDS, _FORMATS, ReadoutLine, unroll

_PLUGINS = Path(__file__).resolve().parent / "plugins"
_KNOWN = _EVENT_KINDS | {"drive", "readout", "submit"}
_ON_LINE = {"drive", "readout", "wait", "frame_update"}


class ToyBackend:
    """A backend that emits each controller's events as tuples, reports them
    back, and emulates every measurement with outcome 0."""

    name = "toy"
    formats = _FORMATS

    def __init__(self, limits: Controller | None = None, shift: int = 0) -> None:
        self.limits = limits or Controller(clock_ghz=1.0)
        self.shift = shift

    def control_system(self) -> ControlSystem:
        return ControlSystem(controllers=[self.limits])

    def emit(self, program: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
        if program.get("format") not in _FORMATS:
            raise backends.IncompatibleBackendError(f"cannot read {program.get('format')!r}")
        flat = unroll(program)
        unknown = {e["kind"] for e in flat["events"]} - _KNOWN
        if unknown:
            raise backends.IncompatibleBackendError(f"cannot emit {sorted(unknown)}")
        # It reports the program's lines: a format /2 shared line once.
        lines = _validation._program_lines(program, flat)
        return [
            [
                {
                    **e,
                    "line": lines.get(e["line"], e["line"]) if e["kind"] in _ON_LINE else e["line"],
                    "start_cycles": e["start_cycles"] + (self.shift if k == 1 else 0),
                }
                for k, e in enumerate(flat["events"])
            ]
        ]

    def events(self, artifact: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        return artifact

    def emulate(self, artifact: Any, *, shots_count: int) -> list[list[int]]:
        readouts = sum(1 for e in artifact[0] if e["kind"] == "readout")
        return [[0] * readouts for _ in range(shots_count)]

    def run(self, artifact: Any, *, shots_count: int = 1) -> Any:
        return self.emulate(artifact, shots_count=shots_count)


@pytest.fixture
def ideal(monkeypatch):
    """Every simulated measurement gives 0, as the toy emulator does."""
    monkeypatch.setattr(_validation, "_simulated", lambda lowered: [0.0] * _readouts(lowered))


def _readouts(lowered):
    return sum(
        1
        for e in unroll(lowered.export_to_controller(version=2))["events"]
        if e["kind"] == "readout"
    )


def test_the_corpus_has_a_program_for_every_event_kind():
    assert set(pulse.Event) <= set(_validation._corpus())


def test_a_conforming_backend_passes_in_every_format_it_reads(ideal):
    report = backends.validate(ToyBackend())
    assert report.ok, str(report)
    lines = str(report).splitlines()
    assert "PASS play (/1)" in lines and "PASS loop (/2)" in lines
    assert "PASS unsupported_event" in lines and "PASS newer_format" in lines
    assert "PASS shared_line (/1)" in lines and "PASS shared_line (/2)" in lines
    for name in ("shared_drive_line_two_tones", "shared_drive_line_one_tone"):
        assert f"PASS {name} (/1)" in lines and f"PASS {name} (/2)" in lines
    assert "PASS several_controllers[0] (/1)" in lines
    assert "PASS several_controllers[1] (/1)" in lines
    assert "PASS answer_timeout (/1)" in lines and "PASS answer_timeout (/2)" in lines
    assert "PASS timeout_function (/1)" in lines and "PASS timeout_function (/2)" in lines
    assert lines[-1] == "backend 'toy': 36 passed"


def test_programs_of_several_controllers_are_exported_in_format_1_only(ideal, monkeypatch):
    backend = ToyBackend()
    monkeypatch.setattr(backend, "formats", (_FORMATS[1],))
    report = str(backends.validate(backend))
    assert "PASS shared_line" in report.splitlines()
    assert "several_controllers" not in report


def test_the_shared_drive_line_program_drives_both_tones_of_one_line_at_once(ideal):
    feedline = ReadoutLine([0, 1], max_concurrent_tones=2)
    for readout_lines, kept in (((), [(0,), (1,)]), ([feedline], [(0, 1)])):
        base = Controller(clock_ghz=1.0, readout_lines=readout_lines)
        system = _validation._shared_drive_line(ControlSystem(controllers=[base]))
        (controller,) = system.controllers
        assert [(line.qudits, line.max_concurrent_tones) for line in controller.drive_lines] == [
            ((0, 1), 2)
        ]
        # Its readout lines stay as the controller has them.
        assert [line.qudits for line in controller.readout_lines] == kept
        corpus = _validation._corpus()
        assert corpus["shared_drive_line_two_tones"] is corpus["shared_drive_line_one_tone"]
        # On the corpus's system the line's two tones differ in frequency;
        # on the system whose qudits share one, they are one tone.
        for spacing_hz, frequencies in ((1.5e8, [5.0e9, 5.15e9]), (0.0, [5.0e9, 5.0e9])):
            dev = _validation._device(system, spacing_hz=spacing_hz)
            kernel = corpus["shared_drive_line_one_tone"]
            lowered = pulse.compile(kernel, quantum_system=dev, control_system=system).lower(
                *dev.qudits[:2]
            )
            program = lowered.export_to_controller(version=2)
            drives = [e for e in program["events"] if e["kind"] == "play"]
            assert [(e["line"], e["tone_slot"], e["qudits"]) for e in drives] == [
                (0, 0, [0]),
                (0, 1, [1]),
            ]
            assert [e["tones"][0]["frequency_hz"] for e in drives] == frequencies
            starts = {(e["start_cycles"], e["duration_cycles"], e["waveform"]) for e in drives}
            assert len(starts) == 1
        assert program["lines"][0]["max_concurrent_tones"] == 2
        assert sum(e["kind"] == "readout" for e in program["events"]) == 2


def test_a_backend_without_shared_drive_lines_refuses_that_program(ideal):
    class Dedicated(ToyBackend):
        def emit(self, program):
            shared = [x for x in program["lines"] if x["role"] == "drive" and "qudits" in x]
            if any(len(line["qudits"]) > 1 for line in shared):
                raise backends.IncompatibleBackendError("one drive line per qudit")
            return super().emit(program)

    report = backends.validate(Dedicated())
    lines = str(report).splitlines()
    assert report.ok, str(report)
    for name in ("shared_drive_line_two_tones", "shared_drive_line_one_tone"):
        assert f"REFUSED {name} (/1): one drive line per qudit" in lines
        assert f"REFUSED {name} (/2): one drive line per qudit" in lines
    assert "PASS shared_line (/1)" in lines and lines[-1] == "backend 'toy': 32 passed, 4 refused"


def test_a_backend_of_one_tone_per_line_emits_the_one_tone_program(ideal):
    class OneTone(ToyBackend):
        def emit(self, program):
            shared = {x["line"] for x in program["lines"] if "max_concurrent_tones" in x}
            played = {
                e["tones"][0]["frequency_hz"]
                for e in program["events"]
                if e["kind"] == "play" and e["line"] in shared
            }
            if len(played) > 1:
                raise backends.IncompatibleBackendError("one tone for every qudit of a line")
            return super().emit(program)

    backend = OneTone()
    backend.formats = (_FORMATS[1],)
    lines = str(backends.validate(backend)).splitlines()
    assert "PASS shared_drive_line_one_tone" in lines
    assert "REFUSED shared_drive_line_two_tones: one tone for every qudit of a line" in lines


def test_each_controller_s_part_is_checked_on_its_own(ideal):
    system = _validation._several_controllers(
        ControlSystem(controllers=[Controller(clock_ghz=1.0)])
    )
    dev = _validation._device(system)
    compiled = pulse.compile(
        _validation._corpus()["several_controllers"], quantum_system=dev, control_system=system
    )
    lowered = compiled.lower(*dev.qudits[:2])
    feedline, other = (lowered.export_to_controller(controller=k) for k in (0, 1))
    assert [line["qudits"] for line in feedline["lines"] if line["role"] == "readout"] == [
        [0, 1],
        [0, 1],
    ]
    assert [e["tone_slot"] for e in feedline["events"] if e["kind"] == "readout"] == [0, 1]
    assert [(line["role"], line["qudit"]) for line in other["lines"]] == [("drive", 1)]
    assert _validation._expected(other)[1] == {1: [("play", 0, 32)]}


def test_pulses_on_the_tones_of_a_shared_line_may_start_together():
    limits = Controller(instruction_cycles=4)
    together = {0: [("readout", 0, 80), ("readout", 0, 80)]}
    assert _validation._limits(limits, together) is not None
    assert _validation._limits(limits, together, frozenset({0})) is None


def test_the_corpus_tells_where_a_run_time_frequency_change_is_made():
    dev = _validation._device(ControlSystem())
    compiled = pulse.compile(_validation._corpus()["runtime_frequency"], quantum_system=dev)
    lowered = compiled.lower(*dev.qudits[:2])
    events = lowered.export_to_controller()["events"]
    first, second = (e for e in events if e["kind"] == "drive")
    [change] = second["frame_updates"]
    # An emulator that took the change at the second pulse would miss the
    # phase the frame gains between them.
    assert change["op"] == "shift_frequency"
    assert change["start_cycles"] == first["end_cycles"] == second["start_cycles"] - 200
    (p,) = _validation._simulated(lowered)
    assert p < 0.1


@pytest.mark.parametrize("name", ["answer_timeout", "timeout_function"])
def test_the_corpus_branches_on_whether_an_answer_came_in_time_and_on_the_answer(name):
    dev = _validation._device(ControlSystem())
    lowered = pulse.compile(_validation._corpus()[name], quantum_system=dev).lower(*dev.qudits[:2])
    for version in (1, 2):
        program = lowered.export_to_controller(version=version)
        flat = unroll(program)
        [answer] = [e for e in flat["events"] if e["kind"] == "answer"]
        assert answer["wait"] == "timeout"
        assert answer["timeout_ns"] == pytest.approx(300)
        conditions = [c for e in flat["events"] for c in e["conditions"]]
        assert [(c["answers"], c["test"], c["result"]) for c in conditions] == [
            ([answer["index"]], "answer", 0),
            ([answer["index"]], "answer", 1),
        ]
    # In time, with bit 0: qudit 1 is driven, qudit 0 is not, so an
    # emulator that took one condition for the other disagrees.
    first, again, other = _validation._simulated(lowered)
    assert again < 0.05 < other


def test_a_backend_that_accepts_what_it_cannot_read_fails(monkeypatch):
    monkeypatch.syspath_prepend(str(_PLUGINS))
    from dummy_controller import DummyBackend

    sys.modules.pop("dummy_controller")
    report = backends.validate(DummyBackend())
    assert not report.ok
    text = str(report)
    assert (
        "FAIL unsupported_event: emit() accepted 'forward_outcome'; "
        "it should raise IncompatibleBackendError" in text
    )
    assert "FAIL newer_format: emit() raised ValueError" in text


def test_timing_that_differs_from_the_controller_program_fails(ideal):
    report = backends.validate(ToyBackend(shift=3))
    assert not report.ok
    assert (
        "FAIL play (/1): line 1: event 0 is a play at 51 cycles; "
        "the controller program has a play at 48" in str(report)
    )


def test_a_call_is_checked_on_its_link_apart_from_the_line_of_its_number(ideal, monkeypatch):
    # A sequencer gives its events back in time order, so a drive on line 0
    # may come before a call over link 0 that the program makes first.
    backend = ToyBackend()
    emit = backend.emit
    monkeypatch.setattr(
        backend,
        "emit",
        lambda program: [sorted(p, key=lambda e: e["start_cycles"]) for p in emit(program)],
    )
    lines = str(backends.validate(backend)).splitlines()
    assert "PASS call (/1)" in lines and "PASS answer (/1)" in lines


def test_a_wrong_number_of_programs_fails(ideal, monkeypatch):
    backend = ToyBackend()
    monkeypatch.setattr(backend, "events", lambda artifact: artifact * 2)
    assert "FAIL play (/1): emitted 2 programs, expected 1" in str(backends.validate(backend))


def test_a_program_beyond_the_declared_controller_is_skipped(ideal, monkeypatch):
    backend = ToyBackend()
    monkeypatch.setattr(backend, "control_system", lambda: ControlSystem())
    no_loop = [c for c in pulse.Capability if c is not pulse.Capability.LOOP]
    limits = Controller(clock_ghz=1.0, max_pulses=2, supports=no_loop)
    report = backends.validate(backend, controller=limits)
    assert "SKIP play: exceeds the controller's limits" in str(report)
    assert report.ok


def test_the_branch_runs_on_a_controller_with_long_feedback_latency(ideal):
    # 2 us of feedback latency at 1 GHz: the branch's pulse waits for it.
    # Before, it came right after its readout and the program was skipped.
    backend = ToyBackend(Controller(clock_ghz=1.0, feedback_latency_cycles=2000))
    report = backends.validate(backend)
    lines = str(report).splitlines()
    assert report.ok, str(report)
    assert "PASS branch (/1)" in lines and "PASS branch (/2)" in lines


def test_the_branch_s_pulse_waits_for_the_feedback_latency():
    slow = ControlSystem(controllers=[Controller(feedback_latency_cycles=2000)])
    dev = _validation._device(slow)
    events = (
        pulse.compile(_validation._corpus()["branch"], quantum_system=dev, control_system=slow)
        .lower(*dev.qudits[:2])
        .export_to_controller()["events"]
    )
    readout_end = next(e["end_cycles"] for e in events if e["kind"] == "readout")
    [branched] = [e for e in events if e["conditions"]]
    assert branched["kind"] == "drive"
    assert branched["start_cycles"] == readout_end + 2000
    # Without a latency, it starts where the readout ends.
    dev = _validation._device(ControlSystem())
    unspaced = (
        pulse.compile(_validation._corpus()["branch"], quantum_system=dev)
        .lower(*dev.qudits[:2])
        .export_to_controller()["events"]
    )
    assert "wait" not in [e["kind"] for e in unspaced]
    [branched] = [e for e in unspaced if e["conditions"]]
    readout_end = next(e["end_cycles"] for e in unspaced if e["kind"] == "readout")
    assert branched["start_cycles"] == readout_end


def _passes(report):
    return [line for line in str(report).splitlines() if line.startswith("PASS")]


@pytest.mark.parametrize(
    ("limits", "stretch"),
    [
        (Controller(clock_ghz=1.0, min_pulse_cycles=64), 4),
        (Controller(clock_ghz=1.0, min_pulse_cycles=50), 4),
        (Controller(clock_ghz=1.0, cycles_per_timestep=8), 2),
        (Controller(clock_ghz=1.0, sample_rate_ghz=2.4), 5),
        (Controller(clock_ghz=1.0, cycles_per_timestep=8, min_pulse_cycles=100), 8),
    ],
)
def test_the_corpus_lasts_long_enough_for_the_controller_to_play_it(ideal, limits, stretch):
    # As written, the corpus has a pulse or a wait the controller cannot play.
    backend = ToyBackend(limits)
    dev = _validation._device(backend.control_system())
    with pytest.raises(pulse.controller.ControllerLimitError):
        pulse.compile(
            _validation._corpus()[pulse.Event.WAIT],
            quantum_system=dev,
            control_system=backend.control_system(),
        ).lower(*dev.qudits[:2])
    assert _validation._stretch(backend.control_system(), dev.clock_ghz) == stretch
    report = backends.validate(backend)
    assert report.ok, str(report)
    assert "SKIP" not in str(report)
    # Its pulses and waits last *stretch* times as long; it passes what a
    # controller without the limit passes.
    assert _passes(report) == _passes(backends.validate(ToyBackend()))
    played = (
        pulse.compile(_validation._corpus(stretch)[pulse.Event.WAIT], quantum_system=dev)
        .lower(*dev.qudits[:2])
        .export_to_controller()["events"]
    )
    assert [e["duration_cycles"] for e in played] == [16 * stretch, 100 * stretch, 16 * stretch]


def test_the_corpus_is_as_written_wherever_it_plays_as_written():
    for limits in (
        Controller(clock_ghz=1.0),
        Controller(clock_ghz=1.0, min_pulse_cycles=16),
        Controller(clock_ghz=1.0, cycles_per_timestep=4),
        Controller(clock_ghz=0.5, sample_rate_ghz=1.0),
        Controller(clock_ghz=1.0, feedback_latency_cycles=2000),
    ):
        system = ControlSystem(controllers=[limits])
        assert _validation._stretch(system, _validation._device(system).clock_ghz) == 1
    assert _validation._corpus(1) is _validation._corpus()
    # Two controllers: the corpus fits both.
    two = ControlSystem(
        controllers=[
            Controller(clock_ghz=1.0, min_pulse_cycles=40, drive_lines=[0]),
            Controller(clock_ghz=1.0, cycles_per_timestep=8, drive_lines=[1]),
        ]
    )
    assert _validation._stretch(two, 1.0) == 4
    # A controller at half the program clock counts its minimum in its own cycles.
    half = ControlSystem(controllers=[Controller(clock_ghz=0.5, min_pulse_cycles=16)])
    assert _validation._stretch(half, 1.0) == 2


def test_emitted_pulses_must_respect_the_declared_limits():
    limits = Controller(instruction_cycles=4, max_pulses=2)
    close = {0: [("play", 0, 16), ("play", 2, 16)]}
    assert _validation._limits(limits, close) == (
        "line 0: two pulses start 2 cycles apart; the controller issues one per 4"
    )
    many = {0: [("play", 0, 4), ("play", 8, 4)], 1: [("readout", 0, 4)]}
    assert _validation._limits(limits, many) is None
    no_loop = [c for c in pulse.Capability if c is not pulse.Capability.LOOP]
    unrolled = Controller(instruction_cycles=4, max_pulses=2, supports=no_loop)
    assert _validation._limits(unrolled, many) == "emitted 3 pulses; the controller plays at most 2"
    assert _validation._limits(None, many) is None


def test_an_emulator_that_disagrees_with_the_simulator_fails(monkeypatch):
    monkeypatch.setattr(_validation, "_simulated", lambda lowered: [0.5] * _readouts(lowered))
    report = backends.validate(ToyBackend(), shots_count=200)
    assert "FAIL readout (/1): measurement 0: the emulator gives 1 in 0.000 of shots" in str(report)


def test_the_emulator_is_compared_with_the_simulator():
    lowered = pulse.compile(
        _validation._corpus()[pulse.Event.READOUT],
        quantum_system=_validation._device(ControlSystem()),
    ).lower(*_validation._device(ControlSystem()).qudits[:2])
    (p,) = _validation._simulated(lowered)
    assert 0.0 < p < 1.0


def test_a_backend_that_refuses_everything_fails(ideal, monkeypatch):
    backend = ToyBackend()

    def refuse(program):
        raise backends.IncompatibleBackendError("no")

    monkeypatch.setattr(backend, "emit", refuse)
    report = backends.validate(backend)
    assert "REFUSED play (/1): no" in str(report)
    assert "FAIL corpus: the backend refused every program" in str(report)


def test_validate_checks_its_arguments():
    with pytest.raises(ValueError, match="shots_count must be positive"):
        backends.validate(ToyBackend(), shots_count=0)
    with pytest.raises(backends.IncompatibleBackendError, match="is not a"):
        backends.validate(object())  # type: ignore[arg-type]
    with pytest.raises(backends.BackendNotFoundError):
        backends.validate("no_such_backend")


def test_only_controllers_that_opt_in_are_validated(ideal):
    found = {"toy": ToyBackend()}
    limits = Controller(clock_ghz=1.0)
    reports = _validation.opted_in(
        [("vendor/a", {"backend": "toy"}, limits), ("vendor/b", {}, limits)], found.__getitem__
    )
    assert [name for name, _ in reports] == ["vendor/a"] and reports[0][1].ok


def test_validation_costs_nothing_until_it_is_called():
    code = (
        "import sys, cudaq_pulse as p; from cudaq_pulse import backends; "
        "print('cudaq_pulse._validation' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
