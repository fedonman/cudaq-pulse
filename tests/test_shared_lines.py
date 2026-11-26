# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared lines: a DriveLine or ReadoutLine reaching several qudits, the
tones it plays at once, and what the checks and the exports make of it."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.controller import DriveLine, ReadoutLine, unroll

from tests import _child


def _system(readout_lines, drive_lines=(0, 1, 2, 3), **limits):
    controller = pulse.Controller(
        drive_lines=list(drive_lines), readout_lines=readout_lines, **limits
    )
    qudits = sorted(
        {q for line in controller.readout_lines for q in line.qudits}
        | {q for line in controller.drive_lines for q in line.qudits}
    )
    return pulse.quantum_system(
        "transmon_surface17",
        layout=qudits,
        control_system=pulse.ControlSystem(controllers=[controller]),
    )


def _feedline(tones, qudits=(0, 1, 2, 3), **limits):
    return _system([ReadoutLine(list(qudits), max_concurrent_tones=tones)], **limits)


@pulse.kernel
def multiplexed(a, b, c, d):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    rc = get_readout_line(c)
    rd = get_readout_line(d)
    sync(ra, rb, rc, rd)
    readout(ra, square(200, 0.05))
    readout(rb, square(200, 0.05))
    readout(rc, square(200, 0.05))
    readout(rd, square(200, 0.05))


@pulse.kernel
def in_turn(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    readout(ra, square(200, 0.05))
    sync(ra, rb)
    readout(rb, square(200, 0.05))


@pulse.kernel
def together(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    sync(ra, rb)
    readout(ra, square(200, 0.05))
    readout(rb, square(200, 0.05))


@pulse.kernel
def rolled_together(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    for _ in range(500):
        sync(ra, rb)
        readout(ra, square(100, 0.05))
        readout(rb, square(100, 0.05))


@pulse.kernel
def rolled_in_turn(a, b):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    for _ in range(500):
        readout(ra, square(100, 0.05))
        sync(ra, rb)
        readout(rb, square(100, 0.05))
        sync(ra, rb)


def _lowered(kernel, qs, *qudits, **options):
    return pulse.compile(kernel, quantum_system=qs, **options).lower(
        *(qs.qudits[q] for q in qudits)
    )


def _native(qs, role, qudit):
    """The frequency of the own tone of qudit *qudit*'s *role* line: its
    readout's, or its dressed 0-1 frequency for a drive line."""
    physical = qs.physical_qudits[qs.layout[qudit]]
    if role == "readout":
        return physical.readout.frequency_hz
    return qs.qudits[qudit].dressed_frequency_hz


# ── The line ────────────────────────────────────────────────────────────────


def test_a_line_reaches_one_qudit_or_several():
    dedicated = ReadoutLine(5, delay_ns=12.0)
    assert dedicated.qudit == 5
    assert dedicated.qudits == (5,)
    assert dedicated.max_concurrent_tones == 1
    assert repr(dedicated) == "ReadoutLine(5, delay_ns=12.0)"
    shared = ReadoutLine([0, 1, 2, 3], delay_ns=12.0, max_concurrent_tones=4)
    assert shared.qudits == (0, 1, 2, 3)
    assert shared.delay_ns == 12.0
    assert shared.max_concurrent_tones == 4
    assert repr(shared) == "ReadoutLine([0, 1, 2, 3], delay_ns=12.0, max_concurrent_tones=4)"
    assert repr(DriveLine(range(3))) == "DriveLine([0, 1, 2])"
    # A list of one qudit is that qudit's dedicated line.
    assert DriveLine([4], delay_ns=2.0) == DriveLine(4, delay_ns=2.0)
    assert DriveLine([4]).qudit == 4
    assert shared == ReadoutLine((0, 1, 2, 3), 12.0, max_concurrent_tones=4)
    assert shared != ReadoutLine([0, 1, 2, 3], 12.0, max_concurrent_tones=3)
    assert shared != DriveLine([0, 1, 2, 3], 12.0, max_concurrent_tones=4)
    assert hash(shared) == hash(ReadoutLine([0, 1, 2, 3], 12.0, max_concurrent_tones=4))
    for line in (dedicated, shared, DriveLine([0, 1])):
        assert pickle.loads(pickle.dumps(line)) == line
        assert type(pickle.loads(pickle.dumps(line))) is type(line)


def test_a_shared_line_has_no_one_qudit():
    with pytest.raises(
        pulse.InvalidValueError,
        match=r"reaches qudits 0\.\.3, so it has no one qudit; read its \.qudits",
    ):
        _ = ReadoutLine([0, 1, 2, 3]).qudit


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (
            lambda: ReadoutLine([0, 1], max_concurrent_tones=3),
            pulse.InvalidValueError,
            r"ReadoutLine\(\[0, 1\]\) reaches 2 qudits, so it plays 1 to 2 tones at once; "
            "got max_concurrent_tones=3",
        ),
        (
            lambda: ReadoutLine([0, 1], max_concurrent_tones=0),
            pulse.InvalidValueError,
            "plays 1 to 2 tones at once; got max_concurrent_tones=0",
        ),
        (
            lambda: DriveLine(3, max_concurrent_tones=2),
            pulse.InvalidValueError,
            r"DriveLine\(3\) reaches one qudit, so it plays one tone at once; "
            "got max_concurrent_tones=2",
        ),
        (
            lambda: DriveLine([3], max_concurrent_tones=2),
            pulse.InvalidValueError,
            "reaches one qudit, so it plays one tone at once",
        ),
        (
            lambda: DriveLine([3, 4], max_concurrent_tones=True),
            pulse.InvalidTypeError,
            "DriveLine.max_concurrent_tones is an int, got True",
        ),
        (
            lambda: DriveLine([3, 4], max_concurrent_tones=1.0),
            pulse.InvalidTypeError,
            "max_concurrent_tones is an int, got 1.0",
        ),
        (
            lambda: DriveLine(3, max_concurrent_tones=1.0),
            pulse.InvalidTypeError,
            "max_concurrent_tones is an int",
        ),
        (
            lambda: DriveLine([]),
            pulse.InvalidValueError,
            "DriveLine reaches no qudit; give it at least one",
        ),
        (
            lambda: DriveLine([1, 1]),
            pulse.InvalidValueError,
            r"DriveLine\(\[1, 1\]\) lists a qudit twice",
        ),
        (
            lambda: DriveLine([1, -1]),
            pulse.InvalidValueError,
            "DriveLine has a negative carrier index -1",
        ),
        (
            lambda: DriveLine([1, "2"]),
            pulse.InvalidTypeError,
            r"reaches qudits by carrier index, got '2'",
        ),
        (
            lambda: DriveLine([1, True]),
            pulse.InvalidTypeError,
            "reaches qudits by carrier index, got True",
        ),
        (
            lambda: DriveLine("01"),
            pulse.InvalidTypeError,
            "DriveLine reaches a qudit or a list of qudits, by carrier index, got '01'",
        ),
        (
            lambda: ReadoutLine([0, 1], delay_ns=-1.0),
            pulse.InvalidValueError,
            "delays cannot be negative",
        ),
    ],
)
def test_a_line_is_checked(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_a_controller_reaches_a_qudit_by_one_line_of_a_kind():
    with pytest.raises(
        pulse.InvalidValueError,
        match=r"qudit 1 is reached by two readout lines, ReadoutLine\(\[0, 1\]\) and "
        r"ReadoutLine\(1\); "
        "choosing between lines is not supported yet: reach each qudit by one readout line",
    ):
        pulse.Controller(readout_lines=[ReadoutLine([0, 1]), 1])
    with pytest.raises(pulse.InvalidValueError, match="qudit 2 is reached by two drive lines"):
        pulse.Controller(drive_lines=[DriveLine([0, 1, 2]), DriveLine([2, 3])])
    # Two dedicated lines of one qudit are the carrier listed twice, as before.
    with pytest.raises(pulse.InvalidValueError, match=r"lists a carrier twice: \[1, 1\]"):
        pulse.Controller(readout_lines=[ReadoutLine(1), 1])
    # A drive line and a readout line may reach the same qudits.
    both = pulse.Controller(drive_lines=[DriveLine([0, 1])], readout_lines=[ReadoutLine([0, 1])])
    assert both.drive_lines == (DriveLine([0, 1]),)


def test_a_bare_list_is_not_a_line():
    with pytest.raises(
        pulse.InvalidTypeError,
        match=r"or a line reaching several qudits as DriveLine\(\[0, 1, 2, 3\], "
        r"max_concurrent_tones=\.\.\.\), got \[0, 1\]",
    ):
        pulse.Controller(drive_lines=[[0, 1]])


def test_a_control_system_reaches_a_qudit_by_one_line_of_a_kind():
    first = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=2)])
    second = pulse.Controller(readout_lines=[2])
    with pytest.raises(pulse.QuantumSystemDefinitionError) as caught:
        pulse.ControlSystem(controllers=[first, second])
    (problem,) = caught.value.problems
    assert problem.path == "controllers[1].readout_lines[0]"
    assert problem.message == (
        "qudit 2 is reached by two readout lines, ReadoutLine([0, 1, 2], "
        "max_concurrent_tones=2) on controllers[0] and ReadoutLine(2); choosing between "
        "lines is not supported yet"
    )
    assert problem.fix == "reach each qudit by one readout line"


def test_every_qudit_of_a_line_is_in_the_quantum_system():
    controller = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 30])])
    with pytest.raises(
        pulse.QuantumSystemDefinitionError, match="qudit 30 is not in the quantum system"
    ):
        pulse.quantum_system(
            "transmon_surface17", control_system=pulse.ControlSystem(controllers=[controller])
        )


def test_a_system_file_holds_a_shared_line():
    qs = _system(
        [ReadoutLine([0, 1, 2], delay_ns=12.0, max_concurrent_tones=2), 3],
        drive_lines=[DriveLine([0, 1]), 2, DriveLine(3, delay_ns=4.0)],
    )
    data = qs.as_dict()
    controller = data["control_system"]["controllers"][0]
    assert controller["readout_lines"] == [
        {"qudits": [0, 1, 2], "max_concurrent_tones": 2, "delay_ns": 12.0},
        3,
    ]
    assert controller["drive_lines"] == [{"qudits": [0, 1]}, 2, {"qudit": 3, "delay_ns": 4.0}]
    back = pulse.QuantumSystem.from_dict(json.loads(json.dumps(data)))
    assert back == qs
    assert back.control_system.controllers[0].readout_lines[0] == ReadoutLine(
        [0, 1, 2], 12.0, max_concurrent_tones=2
    )


@pytest.mark.parametrize(
    ("entry", "path", "message"),
    [
        (
            {"qudits": [0, 1, 2], "max_concurrent_tones": 5},
            "readout_lines[0]",
            "reaches 3 qudits, so it plays 1 to 3 tones at once; got max_concurrent_tones=5",
        ),
        ({"qudits": [0, 1], "tones": 2}, "readout_lines[0].tones", "is not a known key"),
        ({"qudits": []}, "readout_lines[0]", "reaches no qudit"),
    ],
)
def test_a_system_file_shared_line_is_checked(entry, path, message):
    data = _feedline(2, (0, 1, 2)).as_dict()
    data["control_system"]["controllers"][0]["readout_lines"] = [entry]
    with pytest.raises(pulse.QuantumSystemDefinitionError) as caught:
        pulse.QuantumSystem.from_dict(data)
    (problem,) = caught.value.problems
    assert problem.path == f"control_system.controllers[0].{path}"
    assert message in problem.message


def test_a_line_s_delay_reaches_each_of_its_qudits():
    qs = _system([ReadoutLine([0, 1], delay_ns=10.0, max_concurrent_tones=2)], drive_lines=())
    control = qs.control_system
    assert dict(control._line_delays) == {("readout", 0): 10.0, ("readout", 1): 10.0}
    assert control._undelayed().controllers[0].readout_lines == (
        ReadoutLine([0, 1], max_concurrent_tones=2),
    )


# ── Checks ──────────────────────────────────────────────────────────────────


def test_a_feedline_reads_out_its_qudits_together():
    qs = _feedline(4)
    lowered = _lowered(multiplexed, qs, 0, 1, 2, 3)
    assert lowered.export_to_controller(version=2)["lines"][0] == {
        "line": 0,
        "role": "readout",
        "qudits": [0, 1, 2, 3],
        "max_concurrent_tones": 4,
        "native_frequencies_hz": [_native(qs, "readout", q) for q in range(4)],
    }


def test_a_feedline_plays_at_most_its_tones_at_once():
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(multiplexed, _feedline(2), 0, 1, 2, 3)
    assert list(caught.value.violations) == [
        "the readout line of qudits 0..3 (controllers[0].readout_lines[0]) plays 3 tones at once "
        "from cycle 0 (the readouts of qudits 0, 1, 2), and does so 1 more time; it plays at most "
        "2 (max_concurrent_tones): play some after the others end (a sync of their lines between "
        "them), or raise its max_concurrent_tones"
    ]


def test_a_one_tone_line_is_checked_too():
    # Two handles of a feedline that plays one tone at once must not overlap.
    with pytest.raises(
        pulse.controller.ControllerLimitError, match="plays 2 tones at once from cycle 0"
    ):
        _lowered(together, _feedline(1, (0, 1)), 0, 1)
    lowered = _lowered(in_turn, _feedline(1, (0, 1)), 0, 1)
    events = lowered.export_to_controller(version=2)["events"]
    assert [
        (e["kind"], e.get("tone_slot"), e.get("qudits")) for e in events if e["kind"] == "readout"
    ] == [
        ("readout", 0, [0]),
        ("readout", 0, [1]),
    ]


def test_one_handle_of_a_shared_line_needs_no_tone_check():
    @pulse.kernel
    def alone(a):
        r = get_readout_line(a)
        readout(r, square(200, 0.05))
        readout(r, square(200, 0.05))

    lowered = _lowered(alone, _feedline(1, (0, 1)), 0)
    assert lowered.export_to_controller(version=2)["events"][0]["tone_slot"] == 0


def test_dedicated_lines_never_load_the_shared_line_code():
    code = (
        "import sys\n"
        "import cudaq_pulse as pulse\n"
        "controller = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1])\n"
        "qs = pulse.quantum_system('transmon_surface17', layout=[0, 1],\n"
        "    control_system=pulse.ControlSystem(controllers=[controller]))\n"
        "@pulse.kernel\n"
        "def k(a, b):\n"
        "    ra = get_readout_line(a)\n"
        "    rb = get_readout_line(b)\n"
        "    sync(ra, rb)\n"
        "    readout(ra, square(200, 0.05))\n"
        "    readout(rb, square(200, 0.05))\n"
        "low = pulse.compile(k, quantum_system=qs).lower(*qs.qudits[:2])\n"
        "low.export_to_controller()\n"
        "assert 'cudaq_pulse._shared_lines' not in sys.modules, 'loaded'\n"
        "assert 'cudaq_pulse._broadcast_lines' not in sys.modules, 'loaded'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_branches_that_never_play_together_share_a_tone():
    @pulse.kernel
    def either(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        m = readout(ra, square(200, 0.05))
        sync(ra, rb)
        if m:
            readout(ra, square(200, 0.05))
        else:
            readout(rb, square(200, 0.05))

    _lowered(either, _feedline(1, (0, 1)), 0, 1, schedule="rcp")


def test_tone_frequencies_count_per_physical_line():
    qs = _feedline(4, max_frequencies_per_line=2)
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(multiplexed, qs, 0, 1, 2, 3)
    assert list(caught.value.violations) == [
        "the readout line of qudits 0..3 (controllers[0].readout_lines[0]): 4 tone frequencies; "
        "the controller holds 2 per line"
    ]


def test_waveform_memory_counts_per_physical_line():
    @pulse.kernel
    def shapes(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, gaussian(200, 0.05, 40.0))

    # Each handle stores one 200-sample shape; the line stores both.
    qs = _feedline(2, (0, 1), waveform_memory_samples=300)
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(shapes, qs, 0, 1)
    assert list(caught.value.violations) == [
        "the readout line of qudits 0..1 (controllers[0].readout_lines[0]): its waveforms take "
        "400 samples; the controller's waveform memory holds 300 per line"
    ]
    _lowered(shapes, _feedline(2, (0, 1), waveform_memory_samples=400), 0, 1)


def test_spacing_and_amplitude_are_per_handle():
    # Each handle is a tone slot of its own sequencer: two handles start
    # together however many cycles one instruction takes, and the amplitude
    # limit is each tone's (the sum of concurrent tones is not checked).
    @pulse.kernel
    def loud(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.6))
        readout(rb, square(200, 0.6))

    qs = _feedline(2, (0, 1), instruction_cycles=8, max_amplitude=1.0)
    _lowered(loud, qs, 0, 1)
    with pytest.raises(pulse.controller.ControllerLimitError, match="peaks at amplitude 0.6"):
        _lowered(loud, _feedline(2, (0, 1), max_amplitude=0.5), 0, 1)


def test_a_rolled_loop_on_a_feedline_is_checked_every_iteration():
    qs = _feedline(1, (0, 1))
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(rolled_together, qs, 0, 1, unroll_threshold=10)
    (message,) = caught.value.violations
    assert "plays 2 tones at once from cycle 0" in message
    assert "and does so 499 more times" in message
    _lowered(rolled_together, _feedline(2, (0, 1)), 0, 1, unroll_threshold=10)
    _lowered(rolled_in_turn, qs, 0, 1, unroll_threshold=10)


def test_a_program_is_checked_against_a_controller_with_a_feedline():
    lowered = _lowered(together, _system([0, 1], drive_lines=(0, 1)), 0, 1)
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=1)])
    (message,) = pulse.controller.check(lowered, feedline)
    assert message.startswith(
        "the readout line of qudits 0..1 (controllers[0].readout_lines[0]) plays 2 tones at once"
    )
    roomy = pulse.Controller(readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2)])
    assert list(pulse.controller.check(lowered, roomy)) == []


def _walked(lowered, feedline):
    """The shared-line violations of *lowered* against *feedline* with
    every iteration walked, as a check counted them before it counted
    rolled loops from their summary."""
    from cudaq_pulse import _shared_lines
    from cudaq_pulse._system import _one_controller
    from cudaq_pulse.controller import _walk

    lines, events, _, _, _ = _walk(lowered, values=False)
    return _shared_lines.violations(_one_controller(feedline), lowered, lines, events, None)


@pytest.fixture
def no_walk_of_every_round(monkeypatch):
    """Fail, rather than take the machine's memory, if a check walks a
    program without summarizing its loops."""
    from cudaq_pulse import controller

    walk = controller._walk

    def summarizing(lowered, *args, summary=None, **options):
        assert summary is not None, "a check walked every iteration"
        return walk(lowered, *args, summary=summary, **options)

    monkeypatch.setattr(controller, "_walk", summarizing)


def _rounds(trips, together=True, beside=False):
    """A kernel of *trips* rounds kept as a loop: two readouts on one
    feedline, at once or in turn; with *beside*, a third qudit of the
    feedline is read out while the loop runs."""

    @pulse.kernel
    def rounds(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        if beside:
            readout(get_readout_line(c), square(4000, 0.05))
        for _ in range(trips):
            sync(ra, rb)
            readout(ra, square(100, 0.05))
            if not together:
                sync(ra, rb)
            readout(rb, square(100, 0.05))

    return rounds


@pytest.mark.parametrize("together", [True, False])
@pytest.mark.parametrize("beside", [False, True])
@pytest.mark.parametrize("tones", [1, 2, 3])
def test_a_check_against_a_feedline_counts_a_rolled_loop_as_walking_it_does(
    together, beside, tones
):
    from cudaq_pulse import _shared_lines

    system = _system([0, 1, 2], drive_lines=(0, 1, 2))
    lowered = _lowered(_rounds(300, together, beside), system, 0, 1, 2, unroll_threshold=10)
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=tones)])
    found = _shared_lines.against(lowered, feedline)
    assert found == _walked(lowered, feedline)
    assert bool(found) == (tones < together + beside + 1)


def test_a_check_against_a_feedline_does_not_grow_with_the_rounds(no_walk_of_every_round):
    from tests._work import python_work

    system = _system([0, 1, 2], drive_lines=(0, 1, 2))
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=1)])

    def checked(trips):
        lowered = _lowered(_rounds(trips), system, 0, 1, 2, unroll_threshold=10)
        work = python_work(lambda: pulse.controller.check(lowered, feedline))
        return list(pulse.controller.check(lowered, feedline)), work

    few, work_few = checked(1_000)
    many, work_many = checked(2_700_000_000)
    assert "plays 2 tones at once from cycle 0" in few[0]
    assert "and does so 999 more times" in few[0]
    assert "and does so 2699999999 more times" in many[0]
    assert work_many == work_few


def test_rounds_beside_another_readout_are_counted_from_their_period(no_walk_of_every_round):
    # The readout beside the loop is on the feedline while the loop runs, so
    # the rounds are not counted from the first alone: they are from their
    # period, between the cycles where the readout beside them starts and ends.
    from tests._work import python_work

    system = _system([0, 1, 2], drive_lines=(0, 1, 2))

    def checked(trips, tones):
        feedline = pulse.Controller(
            readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=tones)]
        )
        lowered = _lowered(_rounds(trips, beside=True), system, 0, 1, 2, unroll_threshold=10)
        work = python_work(lambda: pulse.controller.check(lowered, feedline))
        return list(pulse.controller.check(lowered, feedline)), work

    few, work_few = checked(1_000, 1)
    many, work_many = checked(2_700_000_000, 1)
    assert "from cycle 0 (the readouts of qudits 0, 1), and does so 1039 more times" in few[0]
    assert (
        "from cycle 0 (the readouts of qudits 0, 1), and does so 2700000039 more times" in many[0]
    )
    assert work_many == work_few
    # With room for the rounds' two tones, the line is past its limit only in
    # the last 40 rounds, where the third readout plays.
    few, work_few = checked(1_000, 2)
    many, work_many = checked(2_700_000_000, 2)
    assert "plays 3 tones at once from cycle 96000 " in few[0]
    assert "plays 3 tones at once from cycle 269999996000 " in many[0]
    assert "and does so 39 more times" in few[0] and "and does so 39 more times" in many[0]
    assert work_many == work_few


def _rounds_and_one(trips, before, length, together):
    """Two readouts a round on one feedline, kept as a loop of *trips*
    rounds, and one readout of a third qudit of the feedline, *length*
    cycles long, *before* cycles after the start."""

    @pulse.kernel
    def rounds(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        wait(rc, before)
        readout(rc, square(length, 0.05))
        for _ in range(trips):
            if together:
                sync(ra, rb)
            readout(ra, square(100, 0.05))
            readout(rb, square(160, 0.05))

    return rounds


@pytest.mark.parametrize("together", [True, False])
@pytest.mark.parametrize("tones", [1, 2])
@pytest.mark.parametrize(
    "before, length",
    [(0, 20), (0, 500), (15_990, 30), (16_000, 160), (24_123, 777), (47_800, 100), (47_990, 500)],
    ids=["first", "first-few", "across-two", "one-round", "middle", "last", "past-the-last"],
)
def test_a_readout_beside_some_rounds_is_counted_in_those_rounds(before, length, tones, together):
    # A count from the loop's period must find the rounds a readout beside
    # the loop plays in, wherever they are: the first, the last, some between.
    from cudaq_pulse import _shared_lines

    system = _system([0, 1, 2], drive_lines=(0, 1, 2))
    kernel = _rounds_and_one(300, before, length, together)
    lowered = _lowered(kernel, system, 0, 1, 2, unroll_threshold=10, schedule="asap")
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=tones)])
    found = _shared_lines.against(lowered, feedline)
    assert found == _walked(lowered, feedline)
    # One tone at most: every placing has a round's readout under the third.
    assert found or tones == 2


def _drifting(trips):
    """Two readouts on one feedline in a loop of *trips* rounds, with no
    sync between their lines: each line repeats by its own period."""

    @pulse.kernel
    def drifting(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        for _ in range(trips):
            readout(ra, square(100, 0.05))
            readout(rb, square(160, 0.05))

    return drifting


def test_rounds_on_lines_of_their_own_periods_are_counted_from_the_periods(
    no_walk_of_every_round,
):
    # Compiled for dedicated lines, then checked against a feedline: the two
    # lines repeat by 100 and by 160 cycles, so what they play together
    # repeats every 800.
    from tests._work import python_work

    dedicated = _system([0, 1], drive_lines=(0, 1))
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=1)])

    def checked(trips):
        lowered = _lowered(_drifting(trips), dedicated, 0, 1, unroll_threshold=10)
        work = python_work(lambda: pulse.controller.check(lowered, feedline))
        return list(pulse.controller.check(lowered, feedline)), work

    few, work_few = checked(1_000)
    many, work_many = checked(2_700_000_000)
    assert "and does so 1499 more times" in few[0]
    assert "and does so 4049999999 more times" in many[0]
    assert work_many == work_few
    roomy = pulse.Controller(readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2)])
    lowered = _lowered(_drifting(2_700_000_000), dedicated, 0, 1, unroll_threshold=10)
    assert list(pulse.controller.check(lowered, roomy)) == []


def test_rounds_on_lines_of_their_own_periods_are_as_walking_them():
    from cudaq_pulse import _shared_lines

    dedicated = _system([0, 1], drive_lines=(0, 1))
    lowered = _lowered(_drifting(300), dedicated, 0, 1, unroll_threshold=10)
    for tones in (1, 2):
        feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=tones)])
        assert _shared_lines.against(lowered, feedline) == _walked(lowered, feedline)


def test_compiling_rounds_on_lines_of_their_own_periods_does_not_walk_them(
    no_walk_of_every_round,
):
    # Compiling for the feedline makes the same count from the periods.
    lowered = _lowered(_drifting(2_700_000_000), _feedline(2, (0, 1)), 0, 1, unroll_threshold=10)
    assert len(lowered.schedule) == 5_400_000_000
    with pytest.raises(pulse.PulseError, match="and does so 4049999999 more times"):
        _lowered(_drifting(2_700_000_000), _feedline(1, (0, 1)), 0, 1, unroll_threshold=10)


def test_rounds_in_one_arm_of_a_branch_do_not_play_with_the_other_arm():
    # The arms of a branch never play together, whatever round of a loop in
    # one of them: counted as a walk of every round does.
    from cudaq_pulse import _shared_lines

    @pulse.kernel
    def either(a, b, c):
        ra = get_readout_line(a)
        rc = get_readout_line(c)
        m = readout(get_readout_line(b), square(100, 0.05))
        sync(ra, rc)
        if m:
            readout(rc, square(4000, 0.05))
        else:
            for _ in range(1000):
                readout(ra, square(100, 0.05))

    system = _system([0, 1, 2], drive_lines=(0, 1, 2))
    lowered = _lowered(either, system, 0, 1, 2, unroll_threshold=10)
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 2], max_concurrent_tones=1)])
    assert _shared_lines.against(lowered, feedline) == _walked(lowered, feedline) == []
    # Compiling for the feedline counted the loop's rounds against the other
    # arm's readout, and refused the program for 40 tones too many.
    shared = _system([ReadoutLine([0, 2], max_concurrent_tones=1), 1], drive_lines=(0, 1, 2))
    _lowered(either, shared, 0, 1, 2, unroll_threshold=10)


def _rounds_in_an_arm(trips):
    """*trips* rounds in one arm of a branch whose other arm is empty,
    beside a readout of another qudit of the feedline."""

    @pulse.kernel
    def rounds(a, b, c):
        ra = get_readout_line(a)
        rc = get_readout_line(c)
        m = readout(get_readout_line(b), square(100, 0.05))
        sync(ra, rc)
        readout(rc, square(4000, 0.05))
        if m:
            for _ in range(trips):
                readout(ra, square(100, 0.05))

    return rounds


def test_rounds_in_an_arm_with_nothing_in_the_other_are_counted_from_their_period(
    no_walk_of_every_round,
):
    # Nothing on the line is in the branch's other arm, so nothing is told
    # apart by the branch: the rounds are counted, not walked.
    system = _system([0, 1, 2], drive_lines=(0, 1, 2))
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 2], max_concurrent_tones=1)])

    def checked(trips):
        kernel = _rounds_in_an_arm(trips)
        lowered = _lowered(kernel, system, 0, 1, 2, unroll_threshold=10, schedule="asap")
        return list(pulse.controller.check(lowered, feedline))

    found = checked(2_700_000_000)
    assert len(found) == 1 and "plays 2 tones at once" in found[0]
    # The readout beside the rounds covers the same rounds whatever their number.
    assert found == checked(300)


def test_rounds_in_an_arm_with_nothing_in_the_other_are_counted_as_walking_them_does():
    from cudaq_pulse import _shared_lines

    system = _system([0, 1, 2], drive_lines=(0, 1, 2))
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 2], max_concurrent_tones=1)])
    kernel = _rounds_in_an_arm(300)
    lowered = _lowered(kernel, system, 0, 1, 2, unroll_threshold=10, schedule="asap")
    found = _shared_lines.against(lowered, feedline)
    assert found and found == _walked(lowered, feedline)


def _four_periods(trips):
    """Four readouts a round on one feedline with no sync: what the four
    lines play together repeats only after about a million rounds."""

    @pulse.kernel
    def rounds(a, b, c, d):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        rd = get_readout_line(d)
        for _ in range(trips):
            readout(ra, square(1000, 0.05))
            readout(rb, square(1001, 0.05))
            readout(rc, square(1003, 0.05))
            readout(rd, square(1007, 0.05))

    return rounds


def test_periods_that_repeat_together_too_rarely_are_refused_before_any_round_is_played(
    no_walk_of_every_round,
):
    system = _system([0, 1, 2, 3], drive_lines=(0, 1, 2, 3))
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 2, 3], max_concurrent_tones=1)])
    lowered = _lowered(_four_periods(2_700_000_000), system, 0, 1, 2, 3, unroll_threshold=10)
    # What is swept before the refusal is small (measured: about two
    # thousand events at most, a stretch two of the lines share), never the
    # stretch of all four, which is copies by the thousand million.
    from cudaq_pulse import _shared_lines

    swept = []
    sweep = _shared_lines._AtOnce.sweep

    def counting(self, played, weigh=None):
        swept.append(len(played))
        return sweep(self, played, weigh)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_shared_lines._AtOnce, "sweep", counting)
        with pytest.raises(pulse.UnsupportedError, match=r"would play \d+ copies of rolled loops"):
            pulse.controller.check(lowered, feedline)
    assert swept and max(swept) < 10_000


def _rounds_of_rounds(trips):
    """*trips* rounds that each hold a loop of 300, beside a readout of a
    third qudit of the feedline."""

    @pulse.kernel
    def rounds(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        readout(get_readout_line(c), square(4000, 0.05))
        for _ in range(trips):
            for _ in range(300):
                sync(ra, rb)
                readout(ra, square(100, 0.05))
                readout(rb, square(100, 0.05))

    return rounds


def test_a_check_that_would_walk_too_many_rounds_says_so(no_walk_of_every_round):
    # Rounds that hold a loop, beside a readout on the feedline, are played
    # one by one: a few thousand are, millions are refused before the walk.
    system = _system([0, 1, 2], drive_lines=(0, 1, 2))
    feedline = pulse.Controller(readout_lines=[ReadoutLine([0, 1, 2], max_concurrent_tones=1)])
    lowered = _lowered(_rounds_of_rounds(5_000_000), system, 0, 1, 2, unroll_threshold=10)
    with pytest.raises(
        pulse.UnsupportedError,
        match=r"would play \d+ copies of rolled loops one by one, more than the 1048576",
    ):
        pulse.controller.check(lowered, feedline)


def test_compiling_rounds_that_would_be_walked_one_by_one_stops_with_an_error(
    no_walk_of_every_round,
):
    with pytest.raises(pulse.UnsupportedError) as refused:
        _lowered(
            _rounds_of_rounds(5_000_000), _feedline(3, (0, 1, 2)), 0, 1, 2, unroll_threshold=10
        )
    assert "copies of rolled loops one by one" in str(refused.value)
    assert "give the loops fewer iterations" in str(refused.value)


@pulse.kernel
def on_every_feedline(*qs):
    lines = []
    for q in qs:
        lines.append(get_readout_line(q))
    for _ in range(100):
        sync(*lines)
        for r in lines:
            readout(r, square(100, 0.05))


def _reads_of_feedlines(count, monkeypatch, unroll_threshold):
    """The events the shared-line checks read to compile and export a
    program on *count* feedlines of two qudits each, its event count, and
    whether the check summarized its rolled loop."""
    from cudaq_pulse import _shared_lines
    from cudaq_pulse.quantum_systems import PhysicalQudit

    lines = [ReadoutLine([2 * k, 2 * k + 1], max_concurrent_tones=2) for k in range(count)]
    qs = pulse.quantum_system(
        name="feedlines",
        clock_ghz=1.0,
        qudits={
            q: PhysicalQudit(q, 5.0e9 + 1.0e7 * q, 50.0, 30.0, anharmonicity_hz=-2.0e8)
            for q in range(2 * count)
        },
        control_system=pulse.ControlSystem(controllers=[pulse.Controller(readout_lines=lines)]),
    )
    counts = {"reads": 0, "summaries": 0}
    timed_on, summarized = _shared_lines._timed_on, _shared_lines.summarized

    def counted(event, of):
        counts["reads"] += 1
        return timed_on(event, of)

    def summary(*args):
        counts["summaries"] += 1
        return summarized(*args)

    monkeypatch.setattr(_shared_lines, "_timed_on", counted)
    monkeypatch.setattr(_shared_lines, "summarized", summary)
    compiled = pulse.compile(
        on_every_feedline, quantum_system=qs, unroll_threshold=unroll_threshold
    )
    lowered = compiled.lower(*(qs.qudits[q] for q in range(2 * count)))
    events = len(lowered.export_to_controller()["events"])
    monkeypatch.undo()
    return counts["reads"], events, counts["summaries"] > 0


@pytest.mark.parametrize("unroll_threshold", [10**6, 10])
def test_the_shared_line_checks_read_each_event_once(monkeypatch, unroll_threshold):
    # Every iteration walked (unrolled) or a rolled loop summarized: the
    # checks read each event a fixed number of times, however many feedlines
    # it is spread over, not once per feedline.
    few, few_events, summarized = _reads_of_feedlines(4, monkeypatch, unroll_threshold)
    many, many_events, _ = _reads_of_feedlines(32, monkeypatch, unroll_threshold)
    assert summarized == (unroll_threshold == 10)
    assert many_events > 7 * few_events
    assert 0 < many / many_events <= 1.1 * few / few_events
    assert many <= 3 * many_events


def test_runtime_qudits_on_a_feedline_are_checked_when_bound():
    @pulse.kernel
    def pair(a: pulse.Qudit, b: pulse.Qudit):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, square(200, 0.05))

    qs = _system([ReadoutLine([0, 1], max_concurrent_tones=1), 2, 3])
    compiled = pulse.compile(pair, quantum_system=qs)
    compiled.lower(qs.qudits[2], qs.qudits[3])
    compiled.lower(qs.qudits[0], qs.qudits[2])
    with pytest.raises(pulse.controller.ControllerLimitError, match="plays 2 tones at once"):
        compiled.lower(qs.qudits[0], qs.qudits[1])


# ── Exports ─────────────────────────────────────────────────────────────────


def _played(events):
    for event in events:
        if event["kind"] == "loop":
            yield from _played(event["body"])
        else:
            yield event


def test_format_2_gives_each_tone_its_slot_and_qudit():
    qs = _feedline(4)
    lowered = _lowered(multiplexed, qs, 0, 1, 2, 3)
    program = lowered.export_to_controller(version=2)
    assert program["lines"] == [
        {
            "line": 0,
            "role": "readout",
            "qudits": [0, 1, 2, 3],
            "max_concurrent_tones": 4,
            "native_frequencies_hz": [_native(qs, "readout", q) for q in range(4)],
        }
    ]
    assert [
        {k: e[k] for k in ("kind", "line", "tone_slot", "qudits", "start_cycles") if k in e}
        for e in program["events"]
    ] == [
        {"kind": "sync", "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 0, "qudits": [0], "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 1, "qudits": [1], "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 2, "qudits": [2], "start_cycles": 0},
        {"kind": "readout", "line": 0, "tone_slot": 3, "qudits": [3], "start_cycles": 0},
    ]
    # The keys follow "line", in the order of every other event's.
    assert list(program["events"][1])[:5] == ["kind", "line", "tone_slot", "qudits", "start_cycles"]
    assert program["events"][0]["lines"] == [0]
    again = _lowered(multiplexed, _feedline(4), 0, 1, 2, 3).export_to_controller(version=2)
    assert json.dumps(again, sort_keys=False) == json.dumps(program, sort_keys=False)


def test_format_2_numbers_physical_lines_by_first_use():
    @pulse.kernel
    def mixed(a, b):
        da = get_drive_line(a)
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        drive(da, gaussian(40, 0.5, 10.0))
        sync(da, ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, square(200, 0.05))

    qs = _feedline(2, (0, 1))
    lowered = _lowered(mixed, qs, 0, 1)
    program = lowered.export_to_controller(version=2)
    assert program["lines"] == [
        {
            "line": 0,
            "role": "drive",
            "qudits": [0],
            "native_frequencies_hz": [_native(qs, "drive", 0)],
        },
        {
            "line": 1,
            "role": "readout",
            "qudits": [0, 1],
            "max_concurrent_tones": 2,
            "native_frequencies_hz": [_native(qs, "readout", 0), _native(qs, "readout", 1)],
        },
    ]
    drive_event = next(e for e in program["events"] if e["kind"] == "play")
    assert "tone_slot" not in drive_event and "qudits" not in drive_event
    sync_event = next(e for e in program["events"] if e["kind"] == "sync")
    assert sync_event["lines"] == [0, 1]


def test_a_program_on_dedicated_lines_exports_as_before():
    @pulse.kernel
    def drives(a, b):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, gaussian(40, 0.5, 10.0))
        sync(da, db)
        drive(db, gaussian(40, 0.5, 10.0))

    # The same program on a system whose readout feedline it does not play.
    plain = _lowered(drives, _system([0, 1], drive_lines=(0, 1)), 0, 1)
    shared = _lowered(
        drives, _system([ReadoutLine([0, 1], max_concurrent_tones=2)], drive_lines=(0, 1)), 0, 1
    )
    for version in (1, 2):
        a = plain.export_to_controller(version=version)
        b = shared.export_to_controller(version=version)
        key = "limits" if version == 1 else "controller"
        a.pop(key), b.pop(key)
        assert json.dumps(a) == json.dumps(b)


def test_format_2_relines_frame_updates():
    @pulse.kernel
    def framed(a, b):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        sync(ra, rb)
        readout(ra, square(200, 0.05))
        readout(rb, square(200, 0.05))
        shift_phase(rb.tone, 0.5)

    program = _lowered(framed, _feedline(2, (0, 1)), 0, 1).export_to_controller(version=2)
    update = next(e for e in program["events"] if e["kind"] == "frame_update")
    assert (update["line"], update["tone_slot"], update["qudits"]) == (0, 1, [1])


def test_format_2_gives_handles_taking_turns_the_lowest_free_slot():
    @pulse.kernel
    def three(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        sync(ra, rb, rc)
        readout(ra, square(100, 0.05))
        readout(rb, square(100, 0.05))
        sync(ra, rb, rc)
        readout(rc, square(100, 0.05))
        readout(ra, square(100, 0.05))
        sync(ra, rb, rc)
        readout(rb, square(100, 0.05))
        readout(rc, square(100, 0.05))

    # Each pair of the three handles overlaps somewhere, so they take turns
    # at the line's two slots, tone by tone: in start order, then the order
    # of the handles' first use, each the lowest free slot.
    lowered = _lowered(three, _feedline(2, (0, 1, 2)), 0, 1, 2, schedule="asap")
    events = [
        e for e in lowered.export_to_controller(version=2)["events"] if e["kind"] == "readout"
    ]
    assert [(e["qudits"][0], e["start_cycles"], e["tone_slot"]) for e in events] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 100, 1),
        (0, 100, 0),
        (1, 200, 0),
        (2, 200, 1),
    ]


def test_format_2_refuses_handles_taking_turns_in_a_rolled_loop():
    @pulse.kernel
    def three(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        for _ in range(100):
            sync(ra, rb, rc)
            readout(ra, square(100, 0.05))
            readout(rb, square(100, 0.05))
            sync(ra, rb, rc)
            readout(rc, square(100, 0.05))
            readout(ra, square(100, 0.05))
            sync(ra, rb, rc)
            readout(rb, square(100, 0.05))
            readout(rc, square(100, 0.05))
            sync(ra, rb, rc)

    lowered = _lowered(
        three, _feedline(2, (0, 1, 2)), 0, 1, 2, schedule="asap", unroll_threshold=10
    )
    with pytest.raises(
        pulse.UnsupportedError, match="format /2 gives each event of a rolled loop one tone slot"
    ):
        lowered.export_to_controller(version=2)


def _slots(program):
    """(qudit, start, tone slot) of each readout of an export, either format."""
    return sorted(
        (e["qudits"][0] if "qudits" in e else e["qudit"], e["start_cycles"], e["tone_slot"])
        for e in _played(program["events"])
        if e["kind"] == "readout"
    )


def test_format_1_gives_each_line_its_shared_line_and_tone_slot():
    qs = _feedline(4)
    lowered = _lowered(multiplexed, qs, 0, 1, 2, 3)
    program = lowered.export_to_controller()
    assert program["lines"] == [
        {
            "line": q,
            "role": "readout",
            "qudit": q,
            "qudits": [0, 1, 2, 3],
            "max_concurrent_tones": 4,
            "native_frequency_hz": _native(qs, "readout", q),
        }
        for q in range(4)
    ]
    readouts = [e for e in program["events"] if e["kind"] == "readout"]
    assert [(e["line"], e["qudit"], e["tone_slot"]) for e in readouts] == [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
    ]
    assert list(readouts[0])[:5] == ["index", "kind", "line", "qudit", "tone_slot"]
    assert "tone_slot" not in program["events"][0]
    assert _slots(program) == _slots(lowered.export_to_controller(version=2))


def test_format_1_gives_the_slots_of_format_2_to_handles_taking_turns():
    @pulse.kernel
    def three(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        sync(ra, rb, rc)
        readout(ra, square(100, 0.05))
        readout(rb, square(100, 0.05))
        sync(ra, rb, rc)
        readout(rc, square(100, 0.05))
        readout(ra, square(100, 0.05))

    lowered = _lowered(three, _feedline(2, (0, 1, 2)), 0, 1, 2, schedule="asap")
    flat = lowered.export_to_controller()
    assert _slots(flat) == _slots(lowered.export_to_controller(version=2))
    assert _slots(flat) == [(0, 0, 0), (0, 100, 0), (1, 0, 1), (2, 100, 1)]


def test_format_1_gives_slots_to_handles_taking_turns_in_a_rolled_loop():
    @pulse.kernel
    def three(a, b, c):
        ra = get_readout_line(a)
        rb = get_readout_line(b)
        rc = get_readout_line(c)
        for _ in range(20):
            sync(ra, rb, rc)
            readout(ra, square(100, 0.05))
            readout(rb, square(100, 0.05))
            sync(ra, rb, rc)
            readout(rc, square(100, 0.05))
            readout(ra, square(100, 0.05))
            sync(ra, rb, rc)

    lowered = _lowered(
        three, _feedline(2, (0, 1, 2)), 0, 1, 2, schedule="asap", unroll_threshold=10
    )
    with pytest.raises(pulse.UnsupportedError, match="one tone slot in every iteration"):
        lowered.export_to_controller(version=2)
    slots = _slots(lowered.export_to_controller())
    assert len(slots) == 80
    assert slots[:4] == [(0, 0, 0), (0, 100, 0), (0, 200, 0), (0, 300, 0)]
    assert {slot for _, _, slot in slots} == {0, 1}


def test_a_program_on_a_dedicated_line_of_a_system_with_a_feedline_has_no_slots():
    @pulse.kernel
    def drives(a):
        drive(get_drive_line(a), gaussian(40, 0.5, 10.0))

    qs = _feedline(2, (0, 1), drive_lines=(0, 1))
    program = _lowered(drives, qs, 0).export_to_controller()
    assert program["lines"] == [
        {"line": 0, "role": "drive", "qudit": 0, "native_frequency_hz": _native(qs, "drive", 0)}
    ]
    assert all("tone_slot" not in e for e in program["events"])


def test_a_program_on_a_delayed_shared_line_exports_in_format_1():
    qs = _system([ReadoutLine([0, 1], delay_ns=12.0, max_concurrent_tones=2)], drive_lines=(0, 1))
    lowered = _lowered(together, qs, 0, 1)
    program = lowered.export_to_controller()
    assert [
        (line["qudit"], line["qudits"], "fine_delay_ns" in line) for line in program["lines"]
    ] == [(0, [0, 1], True), (1, [0, 1], True)]
    assert _slots(program) == [(0, 24, 0), (1, 24, 1)]
    with pytest.raises(pulse.UnsupportedError, match="without line delays; export version 1"):
        lowered.export_to_controller(version=2)


@pulse.kernel
def flip_on_feedline(a, b, c):
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    dc = get_drive_line(c)
    sync(ra, rb)
    outcome = readout(ra, square(200, 0.05))
    readout(rb, square(200, 0.05))
    sync(ra, dc)
    if outcome:
        drive(dc, gaussian(40, 0.5, 10.0))


def _two_controllers():
    # The feedline's controller runs at half the program clock.
    left = pulse.Controller(
        clock_ghz=1.0,
        drive_lines=[0, 1],
        readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2)],
        feedback_latency_cycles=40,
        peer_latency_cycles=30,
    )
    right = pulse.Controller(
        clock_ghz=2.0,
        drive_lines=[2, 3],
        readout_lines=[2, 3],
        feedback_latency_cycles=80,
        peer_latency_cycles=60,
    )
    return pulse.quantum_system(
        "transmon_surface17",
        layout=[0, 1, 2, 3],
        control_system=pulse.ControlSystem(controllers=[left, right], reference_clock_ghz=0.01),
    )


def test_format_1_of_several_controllers_exports_a_shared_line():
    lowered = _lowered(flip_on_feedline, _two_controllers(), 0, 1, 2, schedule="rcp")
    left = lowered.export_to_controller(controller=0)
    right = lowered.export_to_controller(controller=1)
    assert (left["clock_ghz"], left["program_clock_ghz"]) == (1.0, 2.0)
    assert [
        (line["qudit"], line["qudits"], line["max_concurrent_tones"]) for line in left["lines"]
    ] == [
        (0, [0, 1], 2),
        (1, [0, 1], 2),
    ]
    readouts = [e for e in left["events"] if e["kind"] == "readout"]
    # In the left controller's own cycles: 200 program cycles are 100 of its.
    assert [
        (e["qudit"], e["tone_slot"], e["start_cycles"], e["duration_cycles"]) for e in readouts
    ] == [
        (0, 0, 0, 100),
        (1, 1, 0, 100),
    ]
    # Event ids are the program's: the right controller's drive branches on
    # the left one's readout by its index.
    (flip,) = (e for e in right["events"] if e["kind"] == "drive")
    assert flip["conditions"][0]["readouts"] == [readouts[0]["index"]]
    assert all("qudits" not in line for line in right["lines"])
    assert all("tone_slot" not in e for e in right["events"])
    with pytest.raises(pulse.UnsupportedError, match="format /2 is written for one controller"):
        lowered.export_to_controller(controller=0, version=2)


def test_unroll_gives_each_tone_of_a_shared_line_its_own_line():
    lowered = _lowered(multiplexed, _feedline(4), 0, 1, 2, 3)
    flat = lowered.export_to_controller()
    unrolled = unroll(lowered.export_to_controller(version=2))
    assert unrolled["lines"] == flat["lines"]
    assert unrolled["events"] == flat["events"]


def test_unroll_numbers_a_shared_line_s_tones_in_its_order():
    @pulse.kernel
    def taken_out_of_order(a, b, c):
        rc = get_readout_line(c)
        ra = get_readout_line(a)
        sync(ra, rc)
        readout(ra, square(100, 0.05))
        sync(ra, rc)
        readout(rc, square(100, 0.05))

    lowered = _lowered(taken_out_of_order, _feedline(2, (0, 1, 2)), 0, 1, 2)
    unrolled = unroll(lowered.export_to_controller(version=2))
    assert [(line["line"], line["qudit"]) for line in unrolled["lines"]] == [(0, 0), (1, 2)]
    readouts = [e for e in unrolled["events"] if e["kind"] == "readout"]
    assert [(e["line"], e["qudit"]) for e in readouts] == [(0, 0), (1, 2)]
    assert _slots(unrolled) == _slots(lowered.export_to_controller())


@pulse.classical(max_latency_ns=200)
def _delay(bits: pulse.Bits) -> int:
    return 40 + 40 * bits[0]


def test_unroll_puts_a_variable_wait_and_its_segment_on_its_tone():
    @pulse.kernel
    def waits_on_a_tone(a, b, c):
        rc = get_readout_line(c)
        rb = get_readout_line(b)
        ra = get_readout_line(a)
        n = _delay.submit([readout(rb, square(100, 0.05))]).result()
        sync(rb, rc)
        wait(rc, n)
        readout(rc, square(100, 0.05))
        readout(ra, square(100, 0.05))

    lowered = _lowered(waits_on_a_tone, _feedline(2, (0, 1, 2)), 0, 1, 2)
    flat = lowered.export_to_controller()
    unrolled = unroll(lowered.export_to_controller(version=2))

    def by_tone(program):
        tone = {line["line"]: line["qudit"] for line in program["lines"]}
        segments = [(s["opened_by"], tone.get(s["line"])) for s in program["segments"]]
        events = [
            (e["kind"], tone[e["line"]], e["tone_slot"], e["start_cycles"])
            for e in program["events"]
            if "tone_slot" in e
        ]
        return segments, events

    assert by_tone(unrolled) == by_tone(flat)
    assert by_tone(flat)[0] == [("start", None), ("wait", 2)]


def test_a_shared_line_simulates_as_its_qudits_own_lines():
    shared = pulse.simulate(_lowered(together, _feedline(2, (0, 1)), 0, 1), shots_count=10)
    plain = pulse.simulate(
        _lowered(together, _system([0, 1], drive_lines=(0, 1)), 0, 1), shots_count=10
    )
    np.testing.assert_allclose(shared.density_matrix, plain.density_matrix)


# A shared line plays at once only the tones that can play together in one
# execution: of the tones playing when one starts, the most that lie in no
# two different arms of one branch execution.


def _feed_and_condition(tones):
    """Qudits 0-3 on one feedline playing *tones* at once; qudit 4 on its own
    readout line, read for the branches."""
    return _system([ReadoutLine([0, 1, 2, 3], max_concurrent_tones=tones), 4], drive_lines=())


@pulse.kernel
def arms_and_a_third(q0, q1, q2, q3, q4):
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)
    r2 = get_readout_line(q2)
    if readout(get_readout_line(q4), square(40, 0.05)):
        readout(r0, square(400, 0.05))
    else:
        readout(r1, square(400, 0.05))
    wait(r2, 200)
    readout(r2, square(40, 0.05))


@pulse.kernel
def nested_arms(q0, q1, q2, q3, q4):
    r4 = get_readout_line(q4)
    r3 = get_readout_line(q3)
    m = readout(r4, square(40, 0.05))
    n = readout(r4, square(40, 0.05))
    if m:
        readout(get_readout_line(q0), square(400, 0.05))
    else:
        if n:
            readout(get_readout_line(q1), square(400, 0.05))
        else:
            readout(get_readout_line(q2), square(400, 0.05))
    wait(r3, 300)
    readout(r3, square(40, 0.05))


@pulse.kernel
def elif_chain(q0, q1, q2, q3, q4):
    r4 = get_readout_line(q4)
    r3 = get_readout_line(q3)
    m = readout(r4, square(40, 0.05))
    n = readout(r4, square(40, 0.05))
    if m:
        readout(get_readout_line(q0), square(400, 0.05))
    elif n:
        readout(get_readout_line(q1), square(400, 0.05))
    else:
        readout(get_readout_line(q2), square(400, 0.05))
    wait(r3, 300)
    readout(r3, square(40, 0.05))


@pulse.kernel(unroll_threshold=0)
def arms_in_a_rolled_loop(q0, q1, q2, q3, q4):
    r0 = get_readout_line(q0)
    r1 = get_readout_line(q1)
    r2 = get_readout_line(q2)
    r4 = get_readout_line(q4)
    for _ in range(4):
        sync(r0, r1, r2, r4)
        if readout(r4, square(40, 0.05)):
            readout(r0, square(400, 0.05))
        else:
            readout(r1, square(400, 0.05))
        wait(r2, 200)
        readout(r2, square(40, 0.05))


def _tones(events):
    """(start, end, tone_slot, arm by branch) of each tone of exported
    *events* on a shared line, a rolled loop's at its first iteration."""
    for event in events:
        if event["kind"] == "loop":
            yield from _tones(event["body"])
        elif event["kind"] in ("play", "drive", "readout") and "tone_slot" in event:
            start = event["start_cycles"]
            arms = {c["branch"]: c["arm"] for c in event.get("conditions") or ()}
            yield start, start + event["duration_cycles"], event["tone_slot"], arms


def _assert_slots_serve_every_path(program, tones):
    """Each tone of *program* on its shared line has one of its *tones* tone
    slots, and two that play at once share one only in other arms of one
    branch execution."""
    found = list(_tones(program["events"]))
    assert found
    for k, (start, end, slot, arms) in enumerate(found):
        assert slot in range(tones)
        for start2, end2, slot2, arms2 in found[:k]:
            if slot2 == slot and start2 < end and start < end2:
                assert any(arms2.get(branch, arm) != arm for branch, arm in arms.items())


@pytest.mark.parametrize(
    "kernel", [arms_and_a_third, nested_arms, elif_chain, arms_in_a_rolled_loop]
)
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_tones_in_different_arms_of_a_branch_never_play_together(kernel, schedule):
    # Each branch's arms play one readout at a time, beside one more on the
    # line: two tones at once on a line that plays two, and two tone slots
    # serve every path.
    lowered = _lowered(kernel, _feed_and_condition(2), 0, 1, 2, 3, 4, schedule=schedule)
    for version in (1, 2):
        _assert_slots_serve_every_path(lowered.export_to_controller(version=version), 2)


def test_tones_in_different_arms_share_a_tone_slot():
    lowered = _lowered(arms_and_a_third, _feed_and_condition(2), 0, 1, 2, 3, 4)
    for version, qudit in ((1, lambda e: e["qudit"]), (2, lambda e: e["qudits"][0])):
        events = lowered.export_to_controller(version=version)["events"]
        slots = {qudit(e): e["tone_slot"] for e in events if "tone_slot" in e}
        assert slots == {0: 0, 1: 0, 2: 1}


@pulse.kernel
def three_arms(q0, q1, q2, q3, q4):
    r4 = get_readout_line(q4)
    m = readout(r4, square(40, 0.05))
    n = readout(r4, square(40, 0.05))
    if m:
        readout(get_readout_line(q0), square(400, 0.05))
    elif n:
        readout(get_readout_line(q1), square(400, 0.05))
    else:
        readout(get_readout_line(q2), square(400, 0.05))


def test_the_arms_of_an_elif_chain_at_once_share_one_slot():
    # One readout plays at once on every path, so a two-tone line holds the
    # three arms' readouts, which start together under alap, in one slot.
    lowered = _lowered(three_arms, _feed_and_condition(2), 0, 1, 2, 3, 4, schedule="alap")
    for version in (1, 2):
        events = lowered.export_to_controller(version=version)["events"]
        assert [e["tone_slot"] for e in events if "tone_slot" in e] == [0, 0, 0]


@pulse.kernel
def first_choice_undone(q0, q1, q2, q3, q4):
    r2 = get_readout_line(q2)
    r3 = get_readout_line(q3)
    if readout(get_readout_line(q4), square(40, 0.05)):
        readout(get_readout_line(q0), square(400, 0.05))
    else:
        readout(get_readout_line(q1), square(120, 0.05))
        wait(r2, 80)
        readout(r2, square(320, 0.05))
    wait(r3, 200)
    readout(r3, square(400, 0.05))


def test_a_slot_that_leaves_a_later_tone_none_is_chosen_again():
    # Lowest first, qudit 1's readout would take qudit 0's slot (other arms)
    # and qudit 2's the other, leaving qudit 3's, beside qudits 0 and 2,
    # none; qudit 1's takes the other slot instead.
    lowered = _lowered(first_choice_undone, _feed_and_condition(2), 0, 1, 2, 3, 4, schedule="asap")
    for version in (1, 2):
        program = lowered.export_to_controller(version=version)
        _assert_slots_serve_every_path(program, 2)
        assert [e["tone_slot"] for e in program["events"] if "tone_slot" in e] == [0, 1, 0, 0, 1, 1]


@pulse.kernel
def slots_apart_and_together(q0, q1, q2, q3, q4):
    r0 = get_readout_line(q0)
    r3 = get_readout_line(q3)
    readout(r0, square(100, 0.05))
    if readout(get_readout_line(q4), square(40, 0.05)):
        readout(get_readout_line(q1), square(400, 0.05))
    else:
        readout(get_readout_line(q2), square(80, 0.05))
        wait(r3, 70)
        readout(r3, square(290, 0.05))
    wait(r0, 200)
    readout(r0, square(200, 0.05))


def test_tones_no_one_slot_each_serves_on_every_path_are_refused_at_export():
    # No path plays more than two readouts at once, so the check accepts. But
    # qudit 0's two readouts each play beside qudit 1's in the then arm, so
    # on a line of two slots they take one slot, and in the else arm qudit
    # 2's and 3's chain them, the first beside the one, the second beside the
    # other, so they take different ones.
    lowered = _lowered(
        slots_apart_and_together, _feed_and_condition(2), 0, 1, 2, 3, 4, schedule="asap"
    )
    for version in (1, 2):
        with pytest.raises(pulse.UnsupportedError, match="one tone slot for every path"):
            lowered.export_to_controller(version=version)


def _counted(tones, *arms):
    """How often, and with which tones (by position), a line of *tones* is
    full when each of tones playing at once from cycle 0, with *arms* (by
    branch) each, starts."""
    from types import SimpleNamespace

    from cudaq_pulse._shared_lines import _AtOnce

    events = [
        {"index": k, "conditions": [{"branch": b, "arm": a} for b, a in each.items()]}
        for k, each in enumerate(arms)
    ]
    count = _AtOnce(SimpleNamespace(tones=tones))
    count.sweep([(0, 10, event, ()) for event in events])
    return count.count, [e["index"] for e in count.playing]


def test_branches_that_do_not_nest_count_without_the_tree():
    # Branch 2 lies in no arm for tone 1 and in arm False of branch 1 for
    # tone 2: not a tree, which a walk never gives. Tones 0 and 1 play
    # together, tone 2 with neither, so tone 3 (in no branch) is the third.
    nested = ({1: True}, {2: True}, {1: False, 2: False}, {})
    assert _counted(2, *nested) == (1, [0, 1, 3])
    assert _counted(3, *nested) == (0, [])
    # Here every two tones of 0-3 apart are joined by a third neither is
    # apart from (0-1-2-3 a path of "not apart"), so they split neither way
    # and all count: never fewer than can play together.
    path = ({1: True, 2: True}, {3: True}, {1: False}, {2: False, 3: False}, {})
    assert _counted(3, *path) == (1, [0, 1, 2, 3, 4])


# A shared line of h handles, each reading out once in every arm of an elif
# chain of a arms of its own: 2**h paths of h tones at once. The check finds
# the most that play together from how branches nest, in time polynomial in
# the tones, run here in a child process stopped past a CPU and memory limit.
_CHAINS_CPU_S = 10
_CHAINS_RSS_MB = 800


def _chains(handles, arms):
    """The source of a kernel ``chains`` of *handles* x *arms* readouts."""
    lines = ["@pulse.kernel", "def chains(qs):", f"    rm = get_readout_line(qs[{handles}])"]
    lines += [f"    r{i} = get_readout_line(qs[{i}])" for i in range(handles)]
    lines += [
        f"    m{i}_{j} = readout(rm, square(40, 0.05))"
        for i in range(handles)
        for j in range(arms - 1)
    ]
    lines.append("    sync(rm, " + ", ".join(f"r{i}" for i in range(handles)) + ")")
    for i in range(handles):
        for j in range(arms):
            test = "if" if j == 0 else "elif" if j < arms - 1 else "else"
            lines.append(f"    {test} m{i}_{j}:" if j < arms - 1 else "    else:")
            lines.append(f"        readout(r{i}, square(400, 0.05))")
    return "\n".join(lines)


_CHAINS = """
import json
import cudaq_pulse as pulse
from cudaq_pulse.controller import ControllerLimitError, ReadoutLine

{kernel}

found = []
for tones in ({handles}, {handles} - 1):
    controller = pulse.Controller(
        drive_lines=[],
        readout_lines=[ReadoutLine(list(range({handles})), max_concurrent_tones=tones), {handles}],
    )
    system = pulse.quantum_system(
        "transmon_surface17",
        layout=list(range({handles} + 1)),
        control_system=pulse.ControlSystem(controllers=[controller]),
    )
    try:
        pulse.compile(chains, quantum_system=system).lower(system.qudits[: {handles} + 1])
    except ControllerLimitError as error:
        found.append(list(error.violations))
    else:
        found.append([])
print(json.dumps(found))
"""


@pytest.mark.parametrize(("handles", "arms"), [(10, 5), (8, 8), (8, 13), (8, 25)])
def test_tones_of_many_branches_are_counted_in_bounded_time(handles, arms):
    code = _child.limits(_CHAINS_CPU_S, _CHAINS_RSS_MB) + _CHAINS.format(
        kernel=_chains(handles, arms), handles=handles
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=4 * _CHAINS_CPU_S,
    )
    assert out.returncode == 0, f"the child exited {out.returncode}: {out.stderr[-2000:]}"
    fits, over = json.loads(out.stdout.splitlines()[-1])
    # Each path plays one readout per handle at once: as many as the line
    # plays fit, one fewer does not.
    assert fits == []
    assert len(over) == 1
    assert f"plays {handles} tones at once" in over[0]


@pytest.mark.parametrize("schedule", ["asap", "rcp"])
def test_an_arm_and_a_tone_beside_it_still_overlap(schedule):
    # On a line that plays one tone, the readout in either arm and the one
    # beside it play together on every path.
    with pytest.raises(pulse.controller.ControllerLimitError) as caught:
        _lowered(arms_and_a_third, _feed_and_condition(1), 0, 1, 2, 3, 4, schedule=schedule)
    assert list(caught.value.violations) == [
        "the readout line of qudits 0..3 (controllers[0].readout_lines[0]) plays 2 tones at once "
        "from cycle 200 (the readouts of qudits 0, 2); it plays at most 1 (max_concurrent_tones): "
        "play some after the others end (a sync of their lines between them), or raise its "
        "max_concurrent_tones"
    ]
