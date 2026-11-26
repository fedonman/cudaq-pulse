# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Broadcast drive lines: a DriveLine whose one tone reaches several qudits
at once, what a kernel gets for it, and what the IR, the simulator, the
checks and the exports make of it."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.controller import DriveLine, ReadoutLine, unroll
from cudaq_pulse.quantum_systems import Offset, PhysicalQudit


def _qudits(count, step_hz=0.0):
    return pulse.quantum_system(
        name="chip",
        qudits={
            i: PhysicalQudit(i, 5.0e9 + step_hz * i, 50.0, 30.0, anharmonicity_hz=-2.5e8)
            for i in range(count)
        },
        clock_ghz=1.0,
    )


def _box(*drive_lines, readout_lines=(0, 1, 2), **limits):
    return pulse.Controller(
        clock_ghz=1.0, drive_lines=list(drive_lines), readout_lines=list(readout_lines), **limits
    )


BEAM = DriveLine([0, 1, 2], broadcast=True)


@pulse.kernel
def by_one(q):
    drive(get_drive_line(q), gaussian(40, 0.3, 10))


@pulse.kernel
def by_all(a, b, c):
    drive(get_drive_line(a, b, c), gaussian(40, 0.3, 10))


@pulse.kernel
def by_two_names(a, b):
    beam = get_drive_line(a)
    drive(beam, gaussian(40, 0.3, 10))
    shift_phase(beam.tone, 0.3)
    drive(get_drive_line(b), gaussian(40, 0.3, 10))


def _lowered(kernel, qs, box, *indices, **options):
    compiled = pulse.compile(kernel, quantum_system=qs, control_system=box, **options)
    return compiled.lower(*(qs.qudits[i] for i in indices))


def _acquisitions(lowered):
    return [line.strip() for line in lowered.mlir().splitlines() if "pulse.get_drive_line" in line]


# ── The line ────────────────────────────────────────────────────────────────


def test_a_broadcast_line_is_declared_on_the_line():
    line = DriveLine([0, 1, 2], broadcast=True)
    assert line.qudits == (0, 1, 2)
    assert line.broadcast
    assert line.frequency_hz is None
    assert line.max_concurrent_tones == 1
    assert line.delay_ns == 0.0
    assert repr(line) == "DriveLine([0, 1, 2], broadcast=True)"
    stated = DriveLine([0, 1, 2], 4.0, broadcast=True, frequency_hz=5.1e9)
    assert (stated.frequency_hz, stated.delay_ns) == (5.1e9, 4.0)
    assert repr(stated) == (
        "DriveLine([0, 1, 2], delay_ns=4.0, broadcast=True, frequency_hz=5100000000.0)"
    )
    assert eval(repr(stated)) == stated
    assert stated != line
    assert line != DriveLine([0, 1, 2])
    assert pickle.loads(pickle.dumps(stated)) == stated
    assert hash(pickle.loads(pickle.dumps(line))) == hash(line)


def test_other_lines_are_not_broadcast():
    for line in (DriveLine(0), DriveLine(0, delay_ns=4.0), DriveLine([0, 1]), ReadoutLine([0, 1])):
        assert not line.broadcast
        assert line.frequency_hz is None
    assert DriveLine([0, 1], broadcast=False) == DriveLine([0, 1])


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (
            lambda: DriveLine([0, 1, 2], broadcast=True, max_concurrent_tones=2),
            pulse.InvalidValueError,
            r"DriveLine\(\[0, 1, 2\], broadcast=True\) drives one tone that reaches every "
            "qudit; max_concurrent_tones is for a line with a tone per qudit",
        ),
        (
            lambda: DriveLine([0], broadcast=True),
            pulse.InvalidValueError,
            r"DriveLine\(0\) reaches one qudit, so broadcast=True says nothing; leave it out",
        ),
        (
            lambda: DriveLine([0, 1], frequency_hz=5.0e9),
            pulse.InvalidValueError,
            "frequency_hz is the frequency of a broadcast line's one tone",
        ),
        (
            lambda: ReadoutLine([0, 1, 2], broadcast=True),
            pulse.UnsupportedError,
            "a readout line whose one probe reads out several qudits is not supported yet",
        ),
        (
            lambda: DriveLine([0, 1], broadcast=1),
            pulse.InvalidTypeError,
            "DriveLine.broadcast is True or False, got 1",
        ),
        (
            lambda: DriveLine([0, 1], broadcast=True, frequency_hz="5 GHz"),
            pulse.InvalidTypeError,
            "DriveLine.frequency_hz is a number of Hz",
        ),
        (
            lambda: DriveLine([0, 1], broadcast=True, frequency_hz=0.0),
            pulse.InvalidValueError,
            "a positive number",
        ),
        (
            lambda: DriveLine([0, 1], broadcast=True, delay_ns=-1.0),
            pulse.InvalidValueError,
            "delays cannot be negative",
        ),
    ],
)
def test_a_broadcast_line_is_checked(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_the_keywords_are_ignored_on_one_qudit_given_as_an_int():
    # Checking them would cost every DriveLine(q); the list form is checked.
    assert DriveLine(0, broadcast=True) == DriveLine(0)
    assert DriveLine(0, 4.0, frequency_hz=5.0e9) == DriveLine(0, 4.0)
    assert not DriveLine(0, broadcast=True).broadcast
    assert ReadoutLine(0, broadcast=True) == ReadoutLine(0)


def test_a_qudit_is_on_one_drive_line():
    with pytest.raises(pulse.InvalidValueError, match="qudit 1 is reached by two drive lines"):
        pulse.Controller(drive_lines=[DriveLine([0, 1], broadcast=True), 1])


def test_a_system_file_holds_a_broadcast_line():
    system = pulse.ControlSystem(
        controllers=[
            _box(DriveLine([0, 1], 4.0, broadcast=True, frequency_hz=5.1e9), 2),
            pulse.Controller(clock_ghz=1.0, drive_lines=[DriveLine([3, 4], broadcast=True)]),
        ],
        reference_clock_ghz=1.0,
    )
    qs = pulse.quantum_system("transmon_surface17", layout=[0, 1, 2, 3, 4], control_system=system)
    data = qs.as_dict()
    lines = [c["drive_lines"] for c in data["control_system"]["controllers"]]
    assert lines == [
        [{"qudits": [0, 1], "broadcast": True, "frequency_hz": 5.1e9, "delay_ns": 4.0}, 2],
        [{"qudits": [3, 4], "broadcast": True}],
    ]
    back = pulse.QuantumSystem.from_dict(json.loads(json.dumps(data)))
    assert back == qs
    assert back.control_system.controllers[0].drive_lines[0] == DriveLine(
        [0, 1], 4.0, broadcast=True, frequency_hz=5.1e9
    )
    assert pickle.loads(pickle.dumps(system)) == system


# ── The kernel and the IR ───────────────────────────────────────────────────


def test_any_of_its_qudits_or_all_of_them_give_the_one_line():
    qs = _qudits(3)
    box = _box(BEAM)
    line = (
        "%line, %tone = pulse.get_drive_line %0, %1, %2 {frequency_hz = 5.000000e+09 : f64} "
        ": (!pulse.qudit, !pulse.qudit, !pulse.qudit) -> (!pulse.drive_line, !pulse.tone)"
    )
    for lowered in (
        _lowered(by_one, qs, box, 0),
        _lowered(by_one, qs, box, 1),
        _lowered(by_one, qs, box, 2),
        _lowered(by_all, qs, box, 0, 1, 2),
        _lowered(by_all, qs, box, 2, 0, 1),
    ):
        assert _acquisitions(lowered) == [line]
        # The drive reaches all three, whichever the kernel holds.
        assert lowered.qudits == (0, 1, 2)


def test_pulses_on_the_line_follow_each_other_whichever_qudit_named_it():
    lowered = _lowered(by_two_names, _qudits(3), _box(BEAM), 0, 2)
    assert len(_acquisitions(lowered)) == 1
    first, second = lowered.export_to_controller()["events"]
    assert (first["start_cycles"], second["start_cycles"]) == (0, 40)
    # The line has one tone: a phase shift turns every later drive on it.
    assert (first["phase_rad"], second["phase_rad"]) == (0.0, 0.3)


def test_a_called_kernel_gets_the_line_its_caller_holds():
    @pulse.kernel
    def caller(a, b):
        by_one(a)
        by_one(b)

    lowered = _lowered(caller, _qudits(3), _box(BEAM), 0, 1, passes=["verify"])
    assert len(_acquisitions(lowered)) == 1
    starts = [e["start_cycles"] for e in lowered.export_to_controller()["events"]]
    assert starts == [0, 40]


def test_the_line_is_taken_in_a_loop_and_before_a_branch():
    @pulse.kernel
    def rounds(a, b, r):
        for _ in range(20):
            drive(get_drive_line(a), gaussian(40, 0.3, 10))
            drive(get_drive_line(b), gaussian(40, 0.3, 10))
        if readout(get_readout_line(r), square(80, 0.05)):
            drive(get_drive_line(a), square(40, 0.2))
        else:
            drive(get_drive_line(b), square(20, 0.2))

    for threshold in (8, 100):
        lowered = _lowered(rounds, _qudits(3), _box(BEAM), 1, 2, 0, unroll_threshold=threshold)
        assert len(_acquisitions(lowered)) == 1
        assert lowered.qudits == (0, 1, 2)
        played = [e for e in lowered.export_to_controller()["events"] if e["kind"] == "drive"]
        assert [e["start_cycles"] for e in played[:40]] == list(range(0, 1600, 40))
        assert len(played) == 42


def test_a_line_of_its_own_beside_the_broadcast_line():
    @pulse.kernel
    def both(a, d):
        beam = get_drive_line(a)
        own = get_drive_line(d)
        sync(beam, own)
        drive(beam, square(40, 0.2))
        drive(own, square(40, 0.2))

    qs = _qudits(4)
    lowered = _lowered(both, qs, _box(BEAM, 3, readout_lines=(0, 1, 2, 3)), 1, 3)
    assert [text.split(" {")[0] for text in _acquisitions(lowered)] == [
        "%line, %tone = pulse.get_drive_line %0, %1, %2",
        "%line_1, %tone_2 = pulse.get_drive_line %3",
    ]


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        (
            (0, 1),
            r"get_drive_line\(\) names qudits 0..1 of the broadcast drive line of qudits 0..2; "
            "name all its qudits, each once, or any one of them",
        ),
        ((0, 1, 3), "names qudits 0..1, 3 of the broadcast drive line of qudits 0..2"),
        ((3, 0, 1), "no broadcast drive line of the control system reaches qudit 3"),
    ],
)
def test_several_qudits_are_checked_against_the_line(indices, match):
    @pulse.kernel
    def some(a, b):
        drive(get_drive_line(a, b), square(40, 0.2))

    @pulse.kernel
    def three(a, b, c):
        drive(get_drive_line(a, b, c), square(40, 0.2))

    qs = _qudits(4)
    box = _box(BEAM, 3, readout_lines=(0, 1, 2, 3))
    with pytest.raises(pulse.CompilationError, match=match):
        _lowered(some if len(indices) == 2 else three, qs, box, *indices)


@pytest.mark.parametrize(
    "box",
    [None, _box(0, 1, 2), _box(DriveLine([0, 1, 2], max_concurrent_tones=3))],
    ids=["no control system", "dedicated lines", "a shared line"],
)
def test_several_qudits_need_a_broadcast_line(box):
    qs = _qudits(3)
    with pytest.raises(pulse.CompilationError) as caught:
        if box is None:
            pulse.compile(by_all, quantum_system=qs).lower(*qs.qudits[:3])
        else:
            _lowered(by_all, qs, box, 0, 1, 2)
    assert "get_drive_line() got 3 qudits: several qudits name a broadcast drive line" in str(
        caught.value
    )
    assert "DriveLine([0, 1, 2], broadcast=True)" in str(caught.value)


def test_the_line_has_one_tone():
    @pulse.kernel
    def transition(q):
        beam = get_drive_line(q)
        drive(beam, square(40, 0.2), tone=beam.tones[(1, 2)])

    with pytest.raises(pulse.CompilationError, match="has one tone, line.tone"):
        _lowered(transition, _qudits(3), _box(BEAM), 0)


# ── The tone's frequency ────────────────────────────────────────────────────


def test_the_tone_is_at_the_frequency_its_qudits_share():
    lowered = _lowered(by_one, _qudits(3), _box(BEAM), 1)
    (line,) = lowered.export_to_controller()["lines"]
    assert line["native_frequency_hz"] == 5.0e9
    assert lowered.export_to_controller()["events"][0]["frequency_hz"] == 5.0e9


def test_qudits_at_different_frequencies_need_the_line_to_state_its_tone():
    qs = _qudits(3, step_hz=1.5e8)
    with pytest.raises(pulse.CompilationError) as caught:
        _lowered(by_one, qs, _box(BEAM), 1)
    assert str(caught.value).startswith(
        "quantum system 'chip': the broadcast drive line of qudits 0..2 has one tone, and its "
        "qudits are at 5, 5.15 and 5.3 GHz; state the tone's frequency, as "
        "DriveLine([0, 1, 2], broadcast=True, frequency_hz=...)"
    )
    # A program that leaves the line alone compiles.
    lowered = _lowered(by_one, _qudits(4, step_hz=1.5e8), _box(BEAM, 3), 3)
    assert lowered.qudits == (3,)


def test_a_stated_tone_is_the_line_s_whatever_its_qudits_are_at():
    qs = _qudits(3, step_hz=1.5e8)
    box = _box(DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.1e9))
    lowered = _lowered(by_one, qs, box, 2)
    assert "pulse.get_drive_line %0, %1, %2 {frequency_hz = 5.100000e+09 : f64}" in lowered.mlir()
    for version in (1, 2):
        program = lowered.export_to_controller(version=version)
        assert program["lines"][0]["native_frequency_hz"] == 5.1e9
    assert lowered.export_to_controller()["events"][0]["frequency_hz"] == 5.1e9


def _framed(frequencies):
    return pulse.quantum_system(
        name="chip",
        qudits={i: PhysicalQudit(i, f, 50.0, 30.0) for i, f in enumerate(frequencies)},
        clock_ghz=1.0,
    )


def test_on_offsets_from_one_reference_the_tone_is_the_offset_its_qudits_share():
    qs = _framed([Offset(1.5e8)] * 3)
    assert qs.frequency_frame == "shared_reference"
    lowered = _lowered(by_one, qs, _box(BEAM), 1)
    assert "pulse.get_drive_line %0, %1, %2 {frequency_hz = 1.500000e+08 : f64}" in lowered.mlir()
    assert lowered.export_to_controller()["lines"][0]["native_frequency_hz"] == 1.5e8
    # A frequency is not an offset from that reference, and the line cannot
    # state an offset yet.
    stated = _box(DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.0e9))
    with pytest.raises(pulse.CompilationError, match="states its tone as frequency_hz=5"):
        _lowered(by_one, qs, stated, 1)
    apart = _framed([Offset(0.0), Offset(1.5e8), Offset(3.0e8)])
    with pytest.raises(pulse.CompilationError, match="Offsets of 0, 150 and 300 MHz"):
        _lowered(by_one, apart, _box(BEAM), 1)


def test_on_own_carriers_the_tone_is_at_the_carrier_of_every_qudit_it_reaches():
    qs = _framed([None] * 3)
    assert qs.frequency_frame == "own_carrier"

    @pulse.kernel
    def detuned(q):
        beam = get_drive_line(q)
        drive(beam, gaussian(40, 0.3, 10))
        shift_frequency(beam.tone, 2.0e6)
        drive(beam, gaussian(40, 0.3, 10))

    lowered = _lowered(detuned, qs, _box(BEAM), 1)
    assert "pulse.get_drive_line %0, %1, %2 {frequency_hz = 0.000000e+00 : f64}" in lowered.mlir()
    for version in (1, 2):
        (line,) = lowered.export_to_controller(version=version)["lines"]
        assert (line["qudits"], line["broadcast"]) == ([0, 1, 2], True)
        assert line["native_frequency_hz"] == 0.0
    first, second = lowered.export_to_controller()["events"]
    # The offset every reached qudit sees from its own carrier.
    assert (first["frequency_hz"], second["frequency_hz"]) == (0.0, 2.0e6)
    assert first["qudits"] == second["qudits"] == [0, 1, 2]
    assert [e["qudits"] for e in lowered.export_to_emulator()["events"]] == [[0, 1, 2]] * 2

    # Each qudit's own line, driven and detuned alike, is the same evolution.
    @pulse.kernel
    def each(a, b, c):
        lines = [get_drive_line(a), get_drive_line(b), get_drive_line(c)]
        sync(lines[0], lines[1], lines[2])
        for line in lines:
            drive(line, gaussian(40, 0.3, 10))
        for line in lines:
            shift_frequency(line.tone, 2.0e6)
        for line in lines:
            drive(line, gaussian(40, 0.3, 10))

    separate = pulse.compile(each, quantum_system=qs).lower(*qs.qudits[:3])
    together = pulse.simulate(lowered, levels=2, dt_ns=0.5, seed=0)
    apart = pulse.simulate(separate, levels=2, dt_ns=0.5, seed=0)
    np.testing.assert_allclose(together.density_matrix, apart.density_matrix, rtol=0, atol=1e-9)
    populations = np.real(np.diag(together.density_matrix))
    assert populations[0b001] == pytest.approx(populations[0b100])
    assert populations[0b001] > 1e-3


def test_on_own_carriers_the_line_states_no_frequency():
    qs = _framed([None] * 3)
    box = _box(DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.0e9))
    with pytest.raises(pulse.CompilationError) as caught:
        _lowered(by_one, qs, box, 1)
    assert (
        "the broadcast drive line of qudits 0..2 states its tone as frequency_hz=5000000000.0, "
        "and the system's qudits give no frequency_hz"
    ) in str(caught.value)
    # A program that leaves the line alone compiles.
    four = _framed([None] * 4)
    stated = DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.0e9)
    lowered = _lowered(by_one, four, _box(stated, 3, readout_lines=(0, 1, 2, 3)), 3)
    assert lowered.qudits == (3,)


def test_on_own_carriers_no_tone_crosses_between_the_line_and_another_qudit():
    qs = _framed([None] * 4)
    box = _box(BEAM, 3, readout_lines=(0, 1, 2, 3))

    @pulse.kernel
    def others_tone_on_the_line(a, d):
        beam = get_drive_line(a)
        own = get_drive_line(d)
        drive(beam, square(40, 0.2), tone=own.tone)

    @pulse.kernel
    def the_lines_tone_on_another(a, d):
        beam = get_drive_line(a)
        own = get_drive_line(d)
        drive(own, square(40, 0.2), tone=beam.tone)

    @pulse.kernel
    def each_its_own(a, b, d):
        beam = get_drive_line(a)
        own = get_drive_line(d)
        drive(beam, square(40, 0.2), tone=get_drive_line(b).tone)
        drive(own, square(40, 0.2))

    for kernel in (others_tone_on_the_line, the_lines_tone_on_another):
        with pytest.raises(pulse.CompilationError, match="does not state how far"):
            _lowered(kernel, qs, box, 1, 3)

    # The line's tone is no single qudit's: not its lowest qudit's either.
    @pulse.kernel
    def a_readout_tone_on_the_line(a):
        beam = get_drive_line(a)
        drive(beam, square(40, 0.2), tone=get_readout_line(a).tone)

    @pulse.kernel
    def the_lines_tone_on_a_readout(a):
        beam = get_drive_line(a)
        readout(get_readout_line(a), square(80, 0.05), tone=beam.tone)

    for index in (0, 1):
        with pytest.raises(pulse.CompilationError, match="is not the tone of the broadcast"):
            _lowered(a_readout_tone_on_the_line, qs, box, index)
        with pytest.raises(pulse.CompilationError, match="and the line is another"):
            _lowered(the_lines_tone_on_a_readout, qs, box, index)
    lowered = _lowered(each_its_own, qs, box, 0, 2, 3)
    assert len(_acquisitions(lowered)) == 2


# ── The simulator ───────────────────────────────────────────────────────────


@pulse.kernel
def each_on_its_own_line(a, b, c):
    first = get_drive_line(a)
    second = get_drive_line(b)
    third = get_drive_line(c)
    sync(first, second, third)
    drive(first, gaussian(40, 0.3, 10))
    drive(second, gaussian(40, 0.3, 10))
    drive(third, gaussian(40, 0.3, 10))


def test_a_drive_on_the_line_drives_every_qudit():
    qs = _qudits(3)
    broadcast = pulse.simulate(_lowered(by_one, qs, _box(BEAM), 1), levels=2, dt_ns=0.5, seed=0)
    separate = pulse.simulate(
        pulse.compile(each_on_its_own_line, quantum_system=qs).lower(*qs.qudits[:3]),
        levels=2,
        dt_ns=0.5,
        seed=0,
    )
    np.testing.assert_allclose(broadcast.density_matrix, separate.density_matrix, rtol=0, atol=1e-9)
    # Every qudit left its ground state by the same amount.
    populations = np.real(np.diag(broadcast.density_matrix))
    assert populations[0b001] == pytest.approx(populations[0b010])
    assert populations[0b001] == pytest.approx(populations[0b100])
    assert populations[0b001] > 1e-3


def test_each_qudit_is_driven_off_resonance_by_its_own_difference():
    # The tone is on qudit 0; qudits 1 and 2 are 150 and 300 MHz away, far
    # off resonance for this pulse, and stay near their ground states.
    qs = _qudits(3, step_hz=1.5e8)
    box = _box(DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.0e9))
    result = pulse.simulate(_lowered(by_one, qs, box, 2), levels=2, dt_ns=0.25, seed=0)
    matrix = np.asarray(result.density_matrix).reshape((2,) * 6)
    excited = [
        float(np.real(np.einsum("abcabc->abc", matrix).sum(axis=tuple({0, 1, 2} - {axis}))[1]))
        for axis in range(3)
    ]
    # Qudit 0 is the last axis of the density matrix: the further a qudit is
    # from the tone, the less it is excited.
    far, near, on_resonance = excited
    assert on_resonance > 0.05
    assert far < near < on_resonance / 5


# ── The exports ─────────────────────────────────────────────────────────────


def test_format_1_writes_the_line_once_with_its_qudits():
    lowered = _lowered(by_one, _qudits(3), _box(BEAM), 1)
    program = lowered.export_to_controller(version=1)
    assert program["qudits"] == [0, 1, 2]
    assert program["lines"] == [
        {
            "line": 0,
            "role": "drive",
            "qudit": 0,
            "qudits": [0, 1, 2],
            "broadcast": True,
            "native_frequency_hz": 5.0e9,
        }
    ]
    (event,) = program["events"]
    assert list(event)[:5] == ["index", "kind", "line", "qudit", "qudits"]
    assert (event["kind"], event["line"], event["qudit"], event["qudits"]) == (
        "drive",
        0,
        0,
        [0, 1, 2],
    )
    assert "tone_slot" not in event
    json.dumps(program)


def test_format_2_writes_the_line_once_with_its_qudits():
    lowered = _lowered(by_one, _qudits(3), _box(BEAM), 1)
    program = lowered.export_to_controller(version=2)
    assert program["lines"] == [
        {
            "line": 0,
            "role": "drive",
            "qudits": [0, 1, 2],
            "broadcast": True,
            "native_frequency_hz": 5.0e9,
        }
    ]
    assert program["events"] == [
        {
            "kind": "play",
            "line": 0,
            "qudits": [0, 1, 2],
            "start_cycles": 0,
            "duration_cycles": 40,
            "waveform": 0,
            "tones": [{"frequency_hz": 5.0e9, "phase_rad": 0.0}],
        }
    ]
    assert json.loads(json.dumps(program))["lines"] == program["lines"]


def test_frames_name_the_line_s_qudits():
    @pulse.kernel
    def frames(q):
        beam = get_drive_line(q)
        shift_phase(beam.tone, 0.25)
        drive(beam, square(40, 0.2))
        shift_frequency(beam.tone, 1.0e6)
        drive(beam, square(40, 0.2))
        set_phase(beam.tone, 0.5)

    lowered = _lowered(frames, _qudits(3), _box(BEAM, persistent_frames=True), 1)
    for version in (1, 2):
        program = lowered.export_to_controller(version=version)
        assert program["frame_state"] == [
            {"line": 0, "qudits": [0, 1, 2], "frequency_hz": 5.001e9, "phase_rad": 0.5}
        ]


def test_frame_updates_name_the_line_s_qudits():
    @pulse.kernel
    def rounds(q, r):
        beam = get_drive_line(q)
        own = get_readout_line(r)
        for _ in range(20):
            drive(beam, gaussian(40, 0.3, 10))
            shift_phase(beam.tone, 0.1)
        sync(beam, own)
        if readout(own, square(80, 0.05)):
            drive(beam, square(20, 0.1))

    lowered = _lowered(rounds, _qudits(3), _box(BEAM), 0, 1)
    events = lowered.export_to_controller(version=2)["events"]
    updates = [event for event in events if event["kind"] == "frame_update"]
    assert updates
    for update in updates:
        assert list(update)[:3] == ["kind", "line", "qudits"]
        assert (update["line"], update["qudits"]) == (0, [0, 1, 2])
    flat = lowered.export_to_controller(version=1)["events"]
    updates = [event for event in flat if event["kind"] == "frame_update"]
    assert updates
    assert all(update["qudits"] == [0, 1, 2] for update in updates)


def test_a_rolled_loop_on_the_line_stays_a_loop():
    @pulse.kernel
    def rounds(q):
        beam = get_drive_line(q)
        for _ in range(20):
            drive(beam, gaussian(40, 0.3, 10))

    lowered = _lowered(rounds, _qudits(3), _box(BEAM), 0)
    (loop,) = lowered.export_to_controller(version=2)["events"]
    assert (loop["kind"], loop["trips"]) == ("loop", 20)
    assert loop["body"][0]["qudits"] == [0, 1, 2]
    flat = lowered.export_to_controller(version=1)["events"]
    assert len(flat) == 20
    assert all(event["qudits"] == [0, 1, 2] for event in flat)


def test_a_shared_readout_line_beside_the_broadcast_line():
    @pulse.kernel
    def both(a, b):
        beam = get_drive_line(a)
        first = get_readout_line(a)
        second = get_readout_line(b)
        drive(beam, square(40, 0.2))
        sync(beam, first, second)
        readout(first, square(80, 0.05))
        readout(second, square(80, 0.05))

    box = _box(BEAM, readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=3)])
    lowered = _lowered(both, _qudits(3), box, 0, 1)
    program = lowered.export_to_controller(version=2)
    assert program["lines"] == [
        {
            "line": 0,
            "role": "drive",
            "qudits": [0, 1, 2],
            "broadcast": True,
            "native_frequency_hz": 5.0e9,
        },
        {
            "line": 1,
            "role": "readout",
            "qudits": [0, 1, 2],
            "max_concurrent_tones": 3,
            "native_frequencies_hz": [5.0e9, 5.0e9, 5.0e9],
        },
    ]
    driven, _, first, second = program["events"]
    assert (driven["qudits"], "tone_slot" in driven) == ([0, 1, 2], False)
    assert [(e["tone_slot"], e["qudits"]) for e in (first, second)] == [(0, [0]), (1, [1])]
    flat = lowered.export_to_controller(version=1)
    assert [line.get("broadcast", line.get("max_concurrent_tones")) for line in flat["lines"]] == [
        True,
        3,
        3,
    ]


def test_each_controller_exports_its_own_broadcast_line():
    system = pulse.ControlSystem(
        controllers=[
            pulse.Controller(clock_ghz=1.0, drive_lines=[DriveLine([0, 1], broadcast=True)]),
            pulse.Controller(clock_ghz=1.0, drive_lines=[DriveLine([2, 3], broadcast=True)]),
        ],
        reference_clock_ghz=1.0,
    )

    @pulse.kernel
    def beams(a, b, c, d):
        first = get_drive_line(a, b)
        second = get_drive_line(d)
        sync(first, second)
        drive(first, square(40, 0.2))
        drive(second, square(40, 0.2))

    lowered = _lowered(beams, _qudits(4), system, 0, 1, 2, 3)
    for controller, qudits in ((0, [0, 1]), (1, [2, 3])):
        program = lowered.export_to_controller(controller=controller)
        (line,) = program["lines"]
        assert (line["qudit"], line["qudits"], line["broadcast"]) == (qudits[0], qudits, True)
        (event,) = (e for e in program["events"] if e["kind"] == "drive")
        assert event["qudits"] == qudits
        (driven,) = (
            e
            for e in lowered.export_to_emulator(controller=controller)["events"]
            if e["kind"] == "drive"
        )
        assert (driven["qudit"], driven["qudits"]) == (qudits[0], qudits)


def test_a_delayed_broadcast_line_is_timed_as_a_delayed_line_of_its_own():
    @pulse.kernel
    def aligned(q, r):
        beam = get_drive_line(q)
        own = get_readout_line(r)
        sync(beam, own)
        drive(beam, square(40, 0.2))
        readout(own, square(40, 0.05))

    def starts(box):
        events = _lowered(aligned, _qudits(3), box, 0, 2).export_to_controller()["events"]
        return [(e["kind"], e["start_cycles"]) for e in events]

    delayed = starts(_box(DriveLine([0, 1, 2], delay_ns=4.0, broadcast=True)))
    assert delayed == starts(_box(DriveLine(0, delay_ns=4.0), 1, 2))
    assert delayed != starts(_box(BEAM))


def test_the_emulator_export_lists_the_qudits_of_a_broadcast_drive():
    @pulse.kernel
    def both(a, d):
        beam = get_drive_line(a)
        own = get_drive_line(d)
        sync(beam, own)
        drive(beam, square(40, 0.2))
        drive(own, square(40, 0.2))
        readout(get_readout_line(d), square(80, 0.05))

    qs = _qudits(4)
    box = _box(BEAM, 3, readout_lines=(0, 1, 2, 3))
    events = _lowered(both, qs, box, 1, 3).export_to_emulator()["events"]
    on_beam, on_own = (e for e in events if e["kind"] == "drive")
    assert list(on_beam)[:5] == ["kind", "line", "link", "qudit", "qudits"]
    assert (on_beam["qudit"], on_beam["qudits"]) == (0, [0, 1, 2])
    assert "qudits" not in on_own
    assert all("qudits" not in e for e in events if e["kind"] != "drive")

    # A program that leaves the line alone exports as on dedicated lines,
    # byte for byte.
    @pulse.kernel
    def own_line(d):
        drive(get_drive_line(d), square(40, 0.2))
        readout(get_readout_line(d), square(80, 0.05))

    dedicated = _box(0, 1, 2, 3, readout_lines=(0, 1, 2, 3))
    with_beam = _lowered(own_line, qs, box, 3).export_to_emulator()
    without = _lowered(own_line, qs, dedicated, 3).export_to_emulator()
    assert json.dumps(with_beam) == json.dumps(without)


def test_a_loaded_function_is_refused_a_broadcast_line():
    line = "!pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone"
    module = (
        "func.func @round(%a : !pulse.qudit) attributes "
        "{pulse.schedule_state = #pulse.schedule_state<timed>} {\n"
        "  %d, %t = pulse.get_drive_line %a : (!pulse.qudit) -> "
        "(!pulse.drive_line, !pulse.tone)\n"
        "  %c40 = arith.constant 40 : i64\n"
        "  %amp = arith.constant 0.2 : f64\n"
        "  %zero = arith.constant 0.0 : f64\n"
        "  %w = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform\n"
        "  %d1, %t1 = pulse.drive %d, %w, %t "
        f"<{{duration_cycles = 40 : i64, start_cycles = 0 : i64}}> : {line}\n"
        "  return\n"
        "}\n"
    )
    qs = _qudits(4)
    box = _box(BEAM, 3, readout_lines=(0, 1, 2, 3))
    with pytest.raises(pulse.UnsupportedError, match="load.. does not give a loaded function"):
        pulse.load(module, entry="round", qudits=[qs.qudits[3]], control_system=box)
    dedicated = _box(0, 1, 2, 3, readout_lines=(0, 1, 2, 3))
    program = pulse.load(module, entry="round", qudits=[qs.qudits[3]], control_system=dedicated)
    assert program.export_to_controller()["lines"][0]["qudit"] == 3
    several = module.replace(
        "(%a : !pulse.qudit)", "(%a : !pulse.qudit, %b : !pulse.qudit)"
    ).replace(
        "pulse.get_drive_line %a : (!pulse.qudit)",
        "pulse.get_drive_line %a, %b : (!pulse.qudit, !pulse.qudit)",
    )
    with pytest.raises(pulse.CompilationError, match="a drive line of several qudits names"):
        pulse.load(several, entry="round", qudits=qs.qudits[:2], control_system=dedicated)


def test_unrolling_a_broadcast_line_is_refused():
    program = _lowered(by_one, _qudits(3), _box(BEAM), 0).export_to_controller(version=2)
    for given in (program, json.loads(json.dumps(program))):
        with pytest.raises(pulse.UnsupportedError, match="line 0 is a broadcast drive line"):
            unroll(given)

    # Beside a shared line too.
    @pulse.kernel
    def both(a, b):
        beam = get_drive_line(a)
        first = get_readout_line(a)
        second = get_readout_line(b)
        drive(beam, square(40, 0.2))
        sync(beam, first, second)
        readout(first, square(80, 0.05))
        readout(second, square(80, 0.05))

    box = _box(BEAM, readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=3)])
    program = _lowered(both, _qudits(3), box, 0, 1).export_to_controller(version=2)
    with pytest.raises(pulse.UnsupportedError, match="line 0 is a broadcast drive line"):
        unroll(program)


def test_a_line_reaching_a_qudit_the_quantum_system_lacks_is_refused():
    with pytest.raises(pulse.CompilationError, match="does not define"):
        _lowered(by_one, _qudits(2), _box(BEAM), 0)


def test_a_pickle_is_read_by_a_process_that_made_no_broadcast_line(tmp_path):
    lowered = _lowered(by_one, _qudits(3), _box(BEAM), 0)
    system = pulse.ControlSystem(controllers=[_box(BEAM)])
    assert type(pickle.loads(pickle.dumps(system._machine))) is type(system._machine)
    path = tmp_path / "program.pickle"
    path.write_bytes(pickle.dumps((system, lowered)))
    code = (
        "import pickle, sys\n"
        "import cudaq_pulse\n"
        "system, lowered = pickle.loads(open(sys.argv[1], 'rb').read())\n"
        "line = lowered.export_to_controller(version=2)['lines'][0]\n"
        "assert (line['qudits'], line['broadcast']) == ([0, 1, 2], True), line\n"
        "assert system.controllers[0].drive_lines[0].broadcast\n"
        "module = sys.modules['cudaq_pulse._broadcast_lines']\n"
        "assert '__getattr__' not in vars(module)\n"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)


def test_a_compiled_program_pickles():
    lowered = _lowered(by_one, _qudits(3), _box(BEAM), 0)
    back = pickle.loads(pickle.dumps(lowered))
    for version in (1, 2):
        assert back.export_to_controller(version=version) == lowered.export_to_controller(
            version=version
        )


# ── Checks ──────────────────────────────────────────────────────────────────


def test_the_controller_s_limits_apply_to_the_line():
    @pulse.kernel
    def strong(q):
        drive(get_drive_line(q), square(40, 0.9))

    with pytest.raises(pulse.controller.ControllerLimitError, match="peaks at amplitude 0.9"):
        _lowered(strong, _qudits(3), _box(BEAM, max_amplitude=0.5), 1)
    lowered = _lowered(by_one, _qudits(3), _box(BEAM, max_amplitude=0.5), 1)
    assert pulse.controller.check(lowered, _box(BEAM, max_amplitude=0.5)) == []
    assert pulse.controller.check(lowered, _box(BEAM, max_amplitude=0.2))


def test_runtime_qudits_are_checked_when_bound():
    @pulse.kernel
    def driven(q: pulse.Qudit):
        drive(get_drive_line(q), square(40, 0.2))
        readout(get_readout_line(q), square(80, 0.05))

    @pulse.kernel
    def read(q: pulse.Qudit):
        readout(get_readout_line(q), square(80, 0.05))

    qs = _qudits(4)
    box = _box(BEAM, 3, readout_lines=(0, 1, 2, 3))
    compiled = pulse.compile(driven, quantum_system=qs, control_system=box)
    assert compiled.lower(qs.qudits[3]).export_to_controller()["lines"][0]["qudit"] == 3
    for index in (0, 1, 2):
        with pytest.raises(pulse.controller.ControllerLimitError) as caught:
            compiled.lower(qs.qudits[index])
        assert (
            f"line 0 is the drive line of run-time qudit 0, bound to qudit {index}, which the "
            "broadcast drive line of qudits 0..2 reaches"
        ) in str(caught.value)
    assert compiled.lower(qs.qudits[3]).qudits == (3,)
    # Its readout line is the qudit's own.
    reading = pulse.compile(read, quantum_system=qs, control_system=box)
    assert reading.lower(qs.qudits[1]).export_to_controller()["lines"] == [
        {"line": 0, "role": "readout", "qudit": 1}
    ]


def test_an_error_is_located_as_on_a_dedicated_line():
    @pulse.kernel
    def flat(q):
        line = get_drive_line(q)
        drive(line, square(40, 0.2))
        drive(line, gaussian(40, 0.3, 0.0))

    def error(box):
        with pytest.raises(pulse.CompilationError) as caught:
            _lowered(flat, _qudits(3), box, 0)
        return str(caught.value), getattr(caught.value, "__notes__", None)

    assert error(_box(BEAM)) == error(_box(0, 1, 2))


def test_idle_replacement_and_the_line_trace_together():
    @pulse.kernel
    def echo(qudit, window):
        line = get_drive_line(qudit)
        lead = (window - 40) // 2
        wait(line, lead)
        drive(line, square(40, 0.1))
        wait(line, window - 40 - lead)

    @pulse.kernel
    def gap(b, d):
        beam = get_drive_line(b)
        own = get_drive_line(d)
        drive(beam, square(40, 0.1))
        drive(own, square(400, 0.1))
        sync(beam, own)
        drive(beam, square(40, 0.1))

    qs = _qudits(4)
    box = _box(BEAM, 3, readout_lines=(0, 1, 2, 3))
    rule = pulse.IdleReplacement(echo, threshold=100 * pulse.ns)
    compiled = pulse.compile(gap, quantum_system=qs, control_system=box).replace_idle(rule)
    lowered = compiled.lower(qs.qudits[1], qs.qudits[3])
    beams = [text for text in _acquisitions(lowered) if "%0, %1, %2" in text]
    assert len(beams) == 1
    assert len(_acquisitions(lowered)) == 2
    assert lowered.qudits == (0, 1, 2, 3)


def test_unit_warnings_and_the_line_trace_together():
    @pulse.kernel
    def bare(q):
        drive(get_drive_line(q), square(40, 0.2))

    with pytest.warns(pulse.UnitWarning, match="duration 40 has no units"):
        lowered = _lowered(bare, _qudits(3), _box(BEAM), 2, unit_warnings=True)
    assert lowered.qudits == (0, 1, 2)


# ── Cost to programs without one ────────────────────────────────────────────


def test_no_other_control_system_loads_the_broadcast_line_code():
    code = (
        "import sys\n"
        "import cudaq_pulse as pulse\n"
        "from cudaq_pulse.controller import DriveLine, ReadoutLine\n"
        "for lines in ([0, 1], [DriveLine([0, 1], max_concurrent_tones=2)]):\n"
        "    controller = pulse.Controller(drive_lines=lines,\n"
        "        readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2)])\n"
        "    qs = pulse.quantum_system('transmon_surface17', layout=[0, 1],\n"
        "        control_system=pulse.ControlSystem(controllers=[controller]))\n"
        "    @pulse.kernel\n"
        "    def k(a, b):\n"
        "        da = get_drive_line(a)\n"
        "        db = get_drive_line(b)\n"
        "        sync(da, db)\n"
        "        drive(da, square(200, 0.05))\n"
        "        drive(db, square(200, 0.05))\n"
        "        readout(get_readout_line(a), square(200, 0.05))\n"
        "    low = pulse.compile(k, quantum_system=qs).lower(*qs.qudits[:2])\n"
        "    low.export_to_controller(version=1)\n"
        "    low.export_to_controller(version=2)\n"
        "    pulse.controller.check(low, controller)\n"
        "    repr(controller), qs.as_dict()\n"
        "assert 'cudaq_pulse._broadcast_lines' not in sys.modules, 'loaded'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_program_that_leaves_the_line_alone_exports_as_on_dedicated_lines():
    @pulse.kernel
    def own(d):
        line = get_drive_line(d)
        shift_phase(line.tone, 0.1)
        drive(line, square(40, 0.2))
        readout(get_readout_line(d), square(80, 0.05))

    qs = _qudits(4)
    with_beam = _lowered(own, qs, _box(BEAM, 3, readout_lines=(0, 1, 2, 3)), 3)
    without = _lowered(own, qs, _box(0, 1, 2, 3, readout_lines=(0, 1, 2, 3)), 3)
    assert with_beam.mlir() == without.mlir()
    for version in (1, 2):
        ours = with_beam.export_to_controller(version=version)
        theirs = without.export_to_controller(version=version)
        # The program; the export also describes the controller itself.
        for key in ("format", "qudits", "lines", "events", "segments", "waveforms"):
            assert json.dumps(ours.get(key)) == json.dumps(theirs.get(key)), key
