# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The toy instruction-set lowering of examples/isa_lowering: its listings,
their timing against the compiler's schedule, both arms of a branch, a
shared line, a branch across controllers, a decoder call, the emulator's
latency check, validation and refusals."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import backends
from cudaq_pulse.backends import IncompatibleBackendError
from cudaq_pulse.quantum_systems import PhysicalQudit

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "isa_lowering" / "toy_isa.py"
if not _EXAMPLE.exists():
    pytest.skip("needs the source tree's examples/", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("toy_isa", _EXAMPLE)
assert _spec is not None and _spec.loader is not None
toy_isa = importlib.util.module_from_spec(_spec)
sys.modules["toy_isa"] = toy_isa
_spec.loader.exec_module(toy_isa)


@pytest.fixture(scope="module")
def qpu() -> pulse.QuantumSystem:
    return pulse.quantum_system(
        name="toy-qpu",
        qudits={q: PhysicalQudit(q, 5.0e9 + 3.0e8 * q, 50.0, 30.0) for q in range(3)},
        clock_ghz=1.0,
        control_system=toy_isa.ToyBackend().control_system(),
    )


def _lower(kernel, qpu, *qudits):
    return pulse.compile(kernel, quantum_system=qpu, schedule="rcp").lower(
        *(qpu.qudits[q] for q in qudits)
    )


@pulse.kernel
def _ramsey_then_reset(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    x90 = gaussian(40, 0.25, 10.0)
    drive(d, x90)
    shift_frequency(d.tone, 2.0e6)
    wait(d, 100)
    shift_phase(d.tone, math.pi / 2)
    drive(d, x90)
    shift_frequency(d.tone, -2.0e6)
    sync(d, r)
    m = readout(r, square(200, 0.05))
    sync(d, r)
    if m:
        drive(d, gaussian(40, 0.5, 10.0))


@pulse.kernel
def _either(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(200, 0.05))
    sync(d, r)
    if m:
        drive(d, gaussian(40, 0.5, 10.0))
    else:
        drive(d, square(80, 0.1))
    sync(d, r)
    drive(d, gaussian(40, 0.25, 10.0))
    readout(r, square(200, 0.05))


@pulse.kernel
def _read_both(a, b):
    ra, rb = get_readout_line(a), get_readout_line(b)
    sync(ra, rb)
    readout(ra, square(200, 0.05))
    readout(rb, square(200, 0.05))


@pulse.kernel
def _flip_other(measured, other):
    r = get_readout_line(measured)
    p = get_drive_line(other)
    m = readout(r, square(200, 0.05))
    sync(r, p)
    if m:
        drive(p, gaussian(40, 0.5, 10.0))


_GOLDEN = """\
.chan ch0 drive q0
.chan ch1.0 readout q0
.wave w0 40
.wave w1 200
set_freq ch0 5000000000
play ch0 w0 0.25
wait 35
set_freq ch0 5002000000
shift_phase ch0 2.827433
play ch0 w0 0.25
wait 10
set_freq ch1.0 5000000000
acquire ch1.0 w1 0.05 m4
wait 70
branch !m4 end0
set_freq ch0 5000000000
shift_phase ch0 0.502655
play ch0 w0 0.5
shift_phase ch0 -0.502655
end0:"""


def _pulses(scheduled):
    return sorted(
        (e.kind, e.line, e.start_cycles, e.duration_cycles)
        for e in scheduled.schedule
        if e.kind != "wait"
    )


def _run(programs, outcomes):
    """The pulses the emulated controllers play: the plays and
    acquisitions among the instructions they issue."""
    return [step for step in toy_isa.run(programs, outcomes) if step.plays()]


def _played(programs):
    return sorted(
        (e["kind"], e["line"], e["start_cycles"], e["duration_cycles"])
        for p in programs
        for e in toy_isa.events(p)
    )


def test_a_ramsey_and_reset_lower_to_the_golden_listing(qpu):
    program = toy_isa.lower(_lower(_ramsey_then_reset, qpu, 0).export_to_controller())
    assert str(program) == _GOLDEN
    # One table entry serves the three Gaussians, at their gains.
    assert len(program.waveforms) == 2


def test_read_back_listings_play_the_schedule(qpu):
    for kernel, qudits in ((_ramsey_then_reset, (0,)), (_either, (0,)), (_read_both, (0, 1))):
        scheduled = _lower(kernel, qpu, *qudits)
        program = toy_isa.lower(scheduled.export_to_controller())
        assert _played([program]) == _pulses(scheduled), kernel


def test_both_arms_of_a_branch_end_in_step(qpu):
    program = toy_isa.lower(_lower(_either, qpu, 0).export_to_controller())
    assert "jump end0" in program.code and "else0:" in program.code
    taken = _run([program], {"m0": 1})
    skipped = _run([program], {"m0": 0})
    assert [p.instruction.split()[2] for p in taken] == ["w0", "w1", "w1", "w0"]
    assert [p.instruction.split()[2] for p in skipped] == ["w0", "w2", "w1", "w0"]
    # What follows the branch starts at the same cycle whichever arm played.
    assert [p.start_cycles for p in taken[2:]] == [p.start_cycles for p in skipped[2:]]


def test_a_shared_feedline_plays_one_tone_per_qudit(qpu):
    program = toy_isa.lower(_lower(_read_both, qpu, 0, 1).export_to_controller())
    assert [c for c in program.code if c.startswith("acquire")] == [
        "acquire ch0.0 w0 0.05 m1",
        "acquire ch0.1 w0 0.05 m2",
    ]


def test_a_branch_on_another_controllers_outcome(qpu):
    scheduled = _lower(_flip_other, qpu, 0, 2)
    reader, user = (toy_isa.lower(scheduled.export_to_controller(controller=k)) for k in range(2))
    assert "acquire ch0.0 w0 0.05 m0" in reader.code
    assert user.code[:2] == ["wait 80", "branch !m0 end0"]
    assert _played([reader, user]) == _pulses(scheduled)
    flipped = _run([reader, user], {"m0": 1})
    assert [(p.controller, p.start_cycles) for p in flipped][-1] == (1, 320)
    assert all(p.controller == 0 for p in _run([reader, user], {"m0": 0}))


def test_the_emulator_gives_back_each_instruction_it_issues(qpu):
    scheduled = _lower(_flip_other, qpu, 0, 2)
    reader, user = (toy_isa.lower(scheduled.export_to_controller(controller=k)) for k in range(2))
    flipped = toy_isa.run([reader, user], {"m0": 1})
    assert [str(step) for step in flipped] == [
        "controller 0 at   0: set_freq ch0.0 5000000000",
        "controller 0 at   0: acquire ch0.0 w0 0.05 m0",
        "controller 1 at   0: wait 80",
        "controller 1 at 320: branch !m0 end0",
        "controller 1 at 320: set_freq ch0 5600000000",
        "controller 1 at 320: play ch0 w0 0.5",
    ]
    # Each step is where the listing has it; a skipped arm and labels are not issued.
    for step in flipped:
        program = (reader, user)[step.controller]
        assert program.code[step.counter] == step.instruction
    skipped = toy_isa.run([reader, user], {"m0": 0})
    assert [step.counter for step in skipped if step.controller == 1] == [0, 1]


def _parity(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0] ^ bits[1]]


_decode = pulse.classical(_parity, link="decoder", max_latency_ns=100)


@pulse.kernel
def _qec_round(data, ancilla):
    s0 = readout(get_readout_line(data), square(200, 0.05))
    s1 = readout(get_readout_line(ancilla), square(200, 0.05))
    fix = _decode.submit([s0, s1]).result()
    if fix[0]:
        drive(get_drive_line(data), gaussian(40, 0.5, 10.0))


def test_a_decoder_call_is_submitted_and_answered_where_the_schedule_has_them(qpu):
    scheduled = _lower(_qec_round, qpu, 0, 1)
    program = toy_isa.lower(scheduled.export_to_controller())
    assert [c for c in program.code if c.split()[0] in ("submit", "answer", "branch")] == [
        "submit L0 m0 m1",
        "answer L0 a3",
        "branch !a3 end0",
    ]
    timed = {e["kind"]: e["start_cycles"] for e in toy_isa.events(program)}
    # The answer is due 4 + 100 + 8 cycles after the request leaves.
    assert (timed["submit"], timed["answer"]) == (280, 392)
    assert [p.start_cycles for p in _run([program], {"a3": 1})][-1] == 392
    assert len(_run([program], {"a3": 0})) == 2


@pulse.kernel
def _qec_round_with_deadline(data, ancilla):
    s0 = readout(get_readout_line(data), square(200, 0.05))
    s1 = readout(get_readout_line(ancilla), square(200, 0.05))
    in_time, fix = _decode.submit([s0, s1]).result(timeout=60 * pulse.ns)
    if fix[0]:
        drive(get_drive_line(data), gaussian(40, 0.5, 10.0))
    if not in_time:
        drive(get_drive_line(ancilla), gaussian(40, 0.5, 10.0))


def test_a_call_with_a_deadline_is_polled_and_its_flag_tested(qpu):
    scheduled = _lower(_qec_round_with_deadline, qpu, 0, 1)
    program = toy_isa.lower(scheduled.export_to_controller())
    calls = ("submit", "poll", "branch")
    assert [c for c in program.code if c.split()[0] in calls] == [
        "submit L0 m0 m1",
        "poll L0 a3",
        "branch !a3 end0",
        "branch a3.ok end1",
    ]
    timed = {e["kind"]: e["start_cycles"] for e in toy_isa.events(program)}
    # The poll is at the deadline: 4 + 60 + 8 cycles after the request.
    assert (timed["submit"], timed["answer"]) == (280, 352)
    late = [p.instruction for p in _run([program], {})]
    assert late[-1] == "play ch2 w1 0.5"
    in_time = [p.instruction for p in _run([program], {"a3.ok": 1, "a3": 1})]
    assert in_time[-1] == "play ch1 w1 0.5"


def test_the_emulator_fails_a_branch_before_its_outcome_arrives(qpu):
    scheduled = _lower(_flip_other, qpu, 0, 2)
    reader, user = (toy_isa.lower(scheduled.export_to_controller(controller=k)) for k in range(2))
    user.code[0] = "wait 79"
    with pytest.raises(RuntimeError, match="branches on m0 at cycle 316, before cycle 320"):
        _run([reader, user], {"m0": 1})


def test_the_backend_passes_validation():
    report = backends.validate(
        toy_isa.ToyBackend(), controller=toy_isa.toy_controller(feedback_latency_cycles=0)
    )
    assert report.ok, str(report)
    assert str(report).splitlines()[-1] == "backend 'toy': 18 passed, 2 skipped"


def test_what_the_toy_sequencer_cannot_play_is_refused(qpu):
    reset = _lower(_ramsey_then_reset, qpu, 0).export_to_controller()
    readout = next(e for e in reset["events"] if e["kind"] == "readout")
    flip = next(e for e in reset["events"] if e["conditions"])
    # A branch arm playing from 460 to 500, and a readout at 480 that always
    # plays: one sequencer can't issue it while it runs either arm.
    n = len(reset["events"])
    later = {**flip, "index": n, "start_cycles": 500}
    busy = {**readout, "index": n + 1, "start_cycles": 480}
    with pytest.raises(IncompatibleBackendError, match="no wait goes from cycle 500 to 480"):
        toy_isa.lower({**reset, "events": [*reset["events"], later, busy]})
    off = {
        **reset,
        "events": [{**e, "start_cycles": e["start_cycles"] + 2} for e in reset["events"]],
    }
    with pytest.raises(IncompatibleBackendError, match="no wait goes from cycle 0 to 2"):
        toy_isa.lower(off)
    with pytest.raises(IncompatibleBackendError, match="reads cudaq-pulse.controller/1 only"):
        toy_isa.lower({**reset, "format": "cudaq-pulse.controller/2"})
