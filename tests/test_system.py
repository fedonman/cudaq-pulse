# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control systems: controllers, their wiring, links and system files."""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import math
import os
import pickle
import tomllib
import warnings
from collections.abc import Mapping
from fractions import Fraction

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse._errors import (
    InvalidValueError,
    QuantumSystemDefinitionError,
    UnsupportedError,
    UpgradeNotice,
)
from cudaq_pulse._system import installed_descriptions
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import ControllerLimitError, DriveLine, ReadoutLine
from cudaq_pulse.quantum_systems.base import (
    Coupling,
    DispersiveReadout,
    MicrowaveDrive,
    PhysicalQudit,
)
from cudaq_pulse.quantum_systems.models import SimulationModel

from tests._work import native_seconds_each

TRANSMON_SURFACE17 = "transmon_surface17"


@pulse.kernel
def drive_and_branch(q, r):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    d1 = get_drive_line(r)
    drive(d, gaussian(16, 0.5, 4.0))
    sync(d, ro)
    m = readout(ro, square(100, 0.05))
    sync(ro, d1)
    if m:
        drive(d1, square(20, 0.3))


@pulse.classical(max_latency_ns=400, link="feedback")
def flip(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def corrected(q, r):
    ro = get_readout_line(q)
    d = get_drive_line(r)
    m = readout(ro, square(100, 0.05))
    fix = flip.submit([m]).result()
    sync(ro, d)
    if fix[0]:
        drive(d, square(20, 0.3))


def _qpu(*controllers, links=None, layout=(0, 1, 2, 3), **config):
    system = pulse.ControlSystem(controllers=list(controllers), links=links or {})
    return pulse.QuantumSystem.from_dict(
        {"name": TRANSMON_SURFACE17, "layout": list(layout), "control_system": system, **config}
    )


def _two_boxes(**second):
    return (
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], model="box.a"),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], model="box.b", **second),
    )


# ── Definitions ──────────────────────────────────────────────────────────────


def test_a_controller_checks_its_wiring_fields():
    with pytest.raises(TypeError, match="carrier index"):
        pulse.Controller(drive_lines=["q0"])
    with pytest.raises(ValueError, match="twice"):
        pulse.Controller(readout_lines=[1, 1])
    with pytest.raises(ValueError, match="did you mean 'readout'"):
        pulse.Controller(supports=["play", "raedout"])
    with pytest.raises(ValueError, match="negative"):
        pulse.Controller(clock_skew_cycles=-1)
    box = pulse.Controller(drive_lines=range(3), supports=["play", pulse.Capability.SYNC])
    assert box.drive_lines == (DriveLine(0), DriveLine(1), DriveLine(2))
    assert box.supports == (pulse.Capability.PLAY, pulse.Capability.SYNC)
    assert box.supports_capability(pulse.Capability.SYNC)
    assert not box.supports_capability(pulse.Capability.BRANCH)
    assert pulse.Controller().supports_capability(pulse.Capability.BRANCH)
    assert hash(box) == hash(pulse.Controller(drive_lines=[0, 1, 2], supports=["play", "sync"]))
    assert isinstance(box, Controller)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"instruction_cycles": 0}, r"instruction_cycles must be at least 1, got 0$"),
        ({"min_idle_cycles": -1}, r"min_idle_cycles cannot be negative, got -1$"),
        ({"detector_recovery_cycles": -1}, r"detector_recovery_cycles cannot be negative, got -1$"),
    ],
    ids=["instruction_cycles", "min_idle_cycles", "detector_recovery_cycles"],
)
def test_a_controller_cycle_count_error_names_the_value(kwargs, message):
    with pytest.raises(InvalidValueError, match=message):
        pulse.Controller(**kwargs)


def test_a_control_system_reports_every_wiring_problem_at_once():
    a = pulse.Controller(drive_lines=[0, 1], peer_latency_cycles={0: 4})
    b = pulse.Controller(drive_lines=[1, 2])
    stranger = pulse.Controller()
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.ControlSystem(
            controllers=[a, b], links={"feedback": pulse.Link(controllers=[stranger])}
        )
    paths = [p.path for p in raised.value.problems]
    assert paths == [
        "controllers[1].drive_lines[0]",
        "controllers[0].peer_latency_cycles",
        "links.feedback.controllers[0]",
    ]
    assert "also on controllers[0]" in str(raised.value)
    with pytest.raises(TypeError, match="is not a pulse.Controller"):
        pulse.ControlSystem(controllers=[pulse.Processor()])
    with pytest.raises(ValueError, match="keyed by name"):
        pulse.ControlSystem(controllers=[a], links={0: pulse.Link()})


@pytest.mark.parametrize("count", [2, 10])
def test_a_controller_listed_again_names_every_earlier_copy(count):
    # Few controllers are checked pair by pair, many by id: the same errors.
    boxes = [pulse.Controller(model=f"box{i}") for i in range(count)]
    a, b = boxes[0], boxes[1]
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.ControlSystem(controllers=[*boxes, a, b, a])
    assert [str(p) for p in raised.value.problems] == [
        f"controllers[{i}]: is the same Controller as controllers[{j}]; "
        "give each controller of the system its own Controller"
        for i, j in ((count, 0), (count + 1, 1), (count + 2, 0), (count + 2, count))
    ]
    # A link may list an equal Controller; it stands for the first equal one.
    twin = dataclasses.replace(b)
    system = pulse.ControlSystem(controllers=boxes, links={"f": pulse.Link(controllers=[twin, a])})
    assert system.as_dict()["links"]["f"]["controllers"] == [1, 0]
    with pytest.raises(QuantumSystemDefinitionError, match="not a controller of the system"):
        pulse.ControlSystem(controllers=boxes, links={"f": pulse.Link(controllers=[Controller()])})


def test_wiring_names_carriers_of_the_quantum_system():
    box = pulse.Controller(drive_lines=[0, 40], readout_lines=[41], detectors={42: 0})
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        _qpu(box)
    assert [p.path for p in raised.value.problems] == [
        "control_system.controllers[0].drive_lines[1]",
        "control_system.controllers[0].readout_lines[0]",
        "control_system.controllers[0].detectors",
    ]
    assert "it has qudits 0..16" in str(raised.value)
    assert isinstance(raised.value, ValueError)


def test_the_device_holds_its_control_system():
    box = pulse.Controller(drive_lines=range(4), readout_lines=range(4))
    qpu = _qpu(box)
    assert qpu.control_system.controllers == (box,)
    other = _qpu(pulse.Controller(drive_lines=range(4), readout_lines=range(3)))
    assert qpu.fingerprint == other.fingerprint and qpu != other
    legacy = pulse.quantum_system(TRANSMON_SURFACE17, layout=[0, 1, 2, 3])
    assert legacy.control_system is None and legacy.fingerprint == qpu.fingerprint


# ── rcp and the controllers' feedback latency ────────────────────────────────


def _schedule(machine):
    lowered = pulse.compile(drive_and_branch, schedule="rcp", control_system=machine).lower(
        *pulse.quantum_system("transmon_surface17").qudits[:2]
    )
    events = [(e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule]
    return events, lowered.mlir()


def test_rcp_waits_the_feedback_latency_of_the_controllers():
    system = pulse.ControlSystem(controllers=[pulse.Controller(feedback_latency_cycles=50)])
    assert system._machine.readout_latency_cycles == 50
    assert _schedule(ControlSystem()) != _schedule(system)
    # Of several, the slowest that plays readouts, and on another controller
    # the start uncertainty of both too: one cycle each without a reference.
    a, b = _two_boxes(feedback_latency_cycles=50)
    slow = pulse.Controller(drive_lines=[5], feedback_latency_cycles=900)
    assert pulse.ControlSystem(controllers=[a, b, slow])._machine.readout_latency_cycles == 52
    shared = pulse.ControlSystem(controllers=[a, b, slow], reference_clock_ghz=0.5)
    assert shared._machine.readout_latency_cycles == 50


def _slowest_outcome(system, clocks):
    # Every reader and every other user, one pair at a time.
    controllers = system.controllers
    everyone = range(len(controllers))
    readers = [i for i in everyone if controllers[i].readout_lines] or everyone
    users = [i for i in everyone if controllers[i].drive_lines or controllers[i].readout_lines]
    uncertainty = system._start_uncertainty(clocks)
    longest = Fraction(0)
    for reader in readers:
        ratio = Fraction(1) if clocks is None else clocks.controllers[reader].ratio
        source = controllers[reader]
        longest = max(longest, source.feedback_latency_cycles / ratio)
        for user in users or everyone:
            if user != reader:
                peers = source.peer_latency_cycles
                peer = peers.get(user, 0) if isinstance(peers, Mapping) else peers
                slack = uncertainty[reader] + uncertainty[user]
                longest = max(longest, (source.feedback_latency_cycles + peer) / ratio + slack)
    return math.ceil(longest)


@pytest.mark.parametrize("peers", ["one", "some", "mixed"])
def test_the_readout_latency_of_many_controllers_is_their_slowest_pair(peers):
    # Start uncertainties with ties at the top, on readers and on other users.
    count = 24
    skew = [9 if i in (3, 17) else 7 if i == 20 else i % 4 for i in range(count)]
    given = {
        "one": lambda i: 5 + i % 3,
        "some": lambda i: {j: (i + j) % 7 for j in (0, 3, 17, 20) if j != i},
        "mixed": lambda i: {j: 4 for j in (3, 20) if j != i} if i % 2 else 2,
    }[peers]
    controllers = [
        pulse.Controller(
            drive_lines=[i] if i % 5 else [],
            readout_lines=[i] if i % 3 == 0 else [],
            clock_skew_cycles=skew[i],
            peer_latency_cycles=given(i),
            feedback_latency_cycles=10 + (i * 17) % 23,
            clock_ghz=(1.0, 2.0, None)[i % 3],
        )
        for i in range(count)
    ]
    for reference in (None, 0.5):
        system = pulse.ControlSystem(controllers=controllers, reference_clock_ghz=reference)
        assert system._machine.readout_latency_cycles == _slowest_outcome(system, None)
        plan = system._clock_plan(2.0)
        assert plan.readout_latency_cycles == _slowest_outcome(system, plan)


@pytest.mark.parametrize("count", [5, 12])
def test_a_copied_or_unpickled_system_finds_its_own_controllers(count):
    boxes = [pulse.Controller(drive_lines=[i], model=f"box{i}") for i in range(count)]
    system = pulse.ControlSystem(
        controllers=boxes, links={"f": pulse.Link(controllers=[boxes[-1], boxes[1]])}
    )
    everyone = list(range(count))
    assert [system._index(c) for c in boxes] == everyone
    for other in (
        pickle.loads(pickle.dumps(system)),
        copy.deepcopy(system),
        copy.copy(system),
        dataclasses.replace(system),
    ):
        # Many controllers are found by id: each system by its own, never the original's.
        found = vars(other).get("_index")
        assert (found is None) == (count <= 8)
        if found is not None:
            assert found.__self__ is not vars(system)["_index"].__self__
            assert found.__self__._controllers is other.controllers
        assert [other._index(c) for c in other.controllers] == everyone
        assert other.as_dict() == system.as_dict()
    swapped = dataclasses.replace(system, controllers=boxes[::-1])
    assert [swapped._index(c) for c in boxes] == everyone[::-1]
    assert swapped.as_dict()["links"]["f"]["controllers"] == [0, count - 2]
    assert system._index(dataclasses.replace(boxes[2])) == 2
    with pytest.raises(ValueError):
        system._index(pulse.Controller(model="elsewhere"))


# ── Compiling for a control system ───────────────────────────────────────────


def test_compile_uses_the_device_control_system():
    box = pulse.Controller(drive_lines=range(4), readout_lines=range(4), max_pulses=1)
    qpu = _qpu(box)
    with pytest.raises(ControllerLimitError, match="at most 1"):
        pulse.compile(drive_and_branch, quantum_system=qpu)(qpu.qudits[0], qpu.qudits[1])
    # As with the same limits given as one controller.
    with pytest.raises(ControllerLimitError, match="at most 1"):
        pulse.compile(drive_and_branch, control_system=Controller(max_pulses=1)).lower(
            qpu.qudits[0], qpu.qudits[1]
        )
    with pytest.raises(TypeError, match="takes a pulse.ControlSystem or a pulse.Controller"):
        pulse.compile(drive_and_branch, control_system=qpu)  # type: ignore[arg-type]


def test_one_controller_given_to_compile_says_what_it_lacks():
    qpu = pulse.quantum_system("transmon_surface17")
    q, r = qpu.qudits[:2]
    wired = pulse.Controller(drive_lines=[0, 1], readout_lines=[1])
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(drive_and_branch, control_system=wired).lower(q, r)
    assert (
        "line 1 (the readout line of qudit 0) is not on the one Controller compile() was given;"
        " add 0 to its readout_lines, or give compile() a pulse.ControlSystem that describes the"
        " whole control system" in str(raised.value)
    )
    no_branch = [c for c in pulse.Capability if c is not pulse.Capability.BRANCH]
    with pytest.raises(ControllerLimitError, match="needs 'branch', which the one Controller"):
        pulse.compile(drive_and_branch, control_system=Controller(supports=no_branch)).lower(q, r)

    @pulse.classical(max_latency_ns=100, link="decoder")
    def decide(bits: pulse.Bits) -> pulse.Bits:
        return [bits[0]]

    @pulse.kernel
    def decoded(q):
        m = readout(get_readout_line(q), square(100, 0.05))
        decide.submit([m]).result()

    with pytest.raises(pulse.CompilationError, match="no link named 'decoder': compile\\(\\) was"):
        pulse.compile(decoded, control_system=Controller()).lower(q)
    computing = Controller(processors=[pulse.Processor(links=["decoder"])])
    with pytest.raises(
        QuantumSystemDefinitionError, match="the one Controller compile\\(\\) was given"
    ):
        pulse.compile(decoded, control_system=computing)
    # Described as a system, the same program compiles.
    system = pulse.ControlSystem(controllers=[computing], links={"decoder": pulse.Link()})
    pulse.compile(decoded, control_system=system).lower(q)


def test_a_line_on_no_controller_does_not_compile():
    qpu = _qpu(*_two_boxes(), layout=range(6))
    with pytest.raises(
        ControllerLimitError, match="the drive line of qudit 4\\) is on no controller"
    ):
        pulse.compile(drive_and_branch, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[4])
    pulse.compile(drive_and_branch, schedule="rcp", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[3]
    )


def test_each_controller_is_held_to_its_own_limits():
    a, b = _two_boxes(max_pulses=0 + 1)
    qpu = _qpu(a, b)
    compiled = pulse.compile(drive_and_branch, schedule="rcp", quantum_system=qpu)
    # Two events on box.a, one on box.b.
    compiled.lower(qpu.qudits[0], qpu.qudits[2])
    a, b = _two_boxes()
    a = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], model="box.a", max_pulses=1)
    qpu = _qpu(a, b)
    with pytest.raises(ControllerLimitError, match=r"controllers\[0\] \(box.a\): 2 drives"):
        pulse.compile(drive_and_branch, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])


def test_one_controller_on_its_own_clock_is_timed_alike_in_a_system_of_several():
    box = pulse.Controller(drive_lines=range(4), readout_lines=range(4), clock_ghz=1.0)
    # Without a declared program clock, the controller's is the program's.
    assert _qpu(box).clock_ghz == 1.0
    qpu = _qpu(box, clock_ghz=2.0)
    by_system = pulse.compile(drive_and_branch, quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[1]
    )
    machine = ControlSystem(controllers=[Controller(clock_ghz=1.0)])
    by_model = pulse.compile(drive_and_branch, control_system=machine).lower(
        qpu.qudits[0], qpu.qudits[1]
    )
    system, model = by_system.export_to_controller(), by_model.export_to_controller()
    assert system["clock_ghz"] == 1.0 and system["program_clock_ghz"] == qpu.clock_ghz
    assert system["events"] == model["events"]

    @pulse.kernel
    def odd(q):
        d = get_drive_line(q)
        drive(d, square(15, 0.1))

    with pytest.raises(ControllerLimitError, match="line d0 moves in timesteps of 2 cycles"):
        pulse.compile(odd, quantum_system=qpu).lower(qpu.qudits[0])


def _outputs(lowered):
    return (
        lowered.mlir(),
        json.dumps(lowered.export_to_controller(), sort_keys=True),
        json.dumps(lowered.export_to_emulator(), sort_keys=True),
    )


def _delayed_box(delays, **fields):
    """A controller of qudits 0 to 3, its lines delayed as *delays* gives,
    by (role, qudit)."""
    return pulse.Controller(
        drive_lines=[DriveLine(q, delay_ns=delays.get(("drive", q), 0.0)) for q in range(4)],
        readout_lines=[ReadoutLine(q, delay_ns=delays.get(("readout", q), 0.0)) for q in range(4)],
        **fields,
    )


def test_line_delays_and_a_reference_clock_are_the_control_systems():
    delays = {("drive", 0): 2.3, ("readout", 0): 1.0}
    box = _delayed_box(delays, clock_ghz=1.0, max_fine_delay_ns=1.0)
    assert box.drive_lines[0] == DriveLine(0, delay_ns=2.3)
    system = pulse.ControlSystem(controllers=[box], reference_clock_ghz=0.5)
    assert system._machine._line_delays == delays and system._machine.reference_clock_ghz == 0.5
    timed = pulse.quantum_system(TRANSMON_SURFACE17, layout=[0, 1, 2, 3], control_system=system)
    by_system = pulse.compile(drive_and_branch, quantum_system=timed).lower(
        timed.qudits[0], timed.qudits[1]
    )
    by_model = pulse.compile(drive_and_branch, control_system=system).lower(
        timed.qudits[0], timed.qudits[1]
    )
    assert _outputs(by_system) == _outputs(by_model)
    assert by_system.export_to_controller()["reference_clock_ghz"] == 0.5
    lines = {
        (line["role"], line["qudit"]): line for line in by_system.export_to_controller()["lines"]
    }
    assert lines[("drive", 0)]["fine_delay_ns"] == pytest.approx(0.3)


def test_new_line_delays_of_a_control_system_schedule_again_to_the_same_bytes():
    kernel = pulse.kernel(drive_and_branch.__pulse_kernel_fn__)
    for k, delays in enumerate(({("drive", 0): 1.0}, {("drive", 1): 4.0, ("readout", 0): 1.5})):
        system = pulse.ControlSystem(controllers=[_delayed_box(delays)])
        qpu = pulse.quantum_system(TRANSMON_SURFACE17, layout=[0, 1], control_system=system)
        again = pulse.compile(kernel, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[1])
        assert (again.metrics.trace_ms == 0.0) == (k > 0)
        fresh = pulse.kernel(drive_and_branch.__pulse_kernel_fn__)
        full = pulse.compile(fresh, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[1])
        assert _outputs(again) == _outputs(full)


def test_a_control_systems_line_delays_and_reference_clock_are_checked():
    with pytest.raises(ValueError, match=r"DriveLine\(0\).delay_ns is -1.0 ns"):
        pulse.Controller(drive_lines=[DriveLine(0, delay_ns=-1.0)])
    with pytest.raises(ValueError, match="ControlSystem.reference_clock_ghz must be positive"):
        pulse.ControlSystem(reference_clock_ghz=0.0)
    with pytest.raises(TypeError, match="line_delays_ns"):
        pulse.ControlSystem(line_delays_ns={("drive", 0): 1.0})  # type: ignore[call-arg]
    box = pulse.Controller(readout_lines=[ReadoutLine(40, delay_ns=1.0)])
    system = pulse.ControlSystem(controllers=[box])
    with pytest.raises(QuantumSystemDefinitionError, match="qudit 40 is not in the quantum system"):
        pulse.quantum_system(TRANSMON_SURFACE17, control_system=system)


def test_a_control_system_takes_its_times_as_times():
    ns, us = pulse.ns, pulse.us
    lines = {"drive_lines": range(4), "readout_lines": range(4)}
    box = pulse.Controller(max_fine_delay_ns=1 * ns, **lines)
    assert box == pulse.Controller(max_fine_delay_ns=1.0, **lines)
    timed = pulse.Controller(
        max_fine_delay_ns=1 * ns,
        drive_lines=[DriveLine(0, delay_ns=2.5 * ns), 1, 2, 3],
        readout_lines=[ReadoutLine(0, delay_ns=0.001 * us), 1, 2, 3],
    )
    numbered = pulse.Controller(
        max_fine_delay_ns=1.0,
        drive_lines=[DriveLine(0, delay_ns=2.5), 1, 2, 3],
        readout_lines=[ReadoutLine(0, delay_ns=1.0), 1, 2, 3],
    )
    by_time = pulse.ControlSystem(
        controllers=[timed],
        links={"decoder": Link(fixed_ns=0.1 * us, feedback_ns=20 * ns, per_controller=True)},
    )
    by_number = pulse.ControlSystem(
        controllers=[numbered],
        links={"decoder": Link(fixed_ns=100, feedback_ns=20, per_controller=True)},
    )
    assert by_time == by_number
    assert [ns.__class__ for ns in by_time._line_delays.values()] == [float, float]
    qpus = [
        pulse.quantum_system(TRANSMON_SURFACE17, layout=[0, 1], control_system=system)
        for system in (by_time, by_number)
    ]
    assert qpus[0].as_dict() == qpus[1].as_dict()
    with pytest.raises(ValueError, match=r"ReadoutLine\(0\).delay_ns is -1.0 ns"):
        ReadoutLine(0, delay_ns=-1 * ns)


@pytest.mark.parametrize(
    ("name", "value"),
    [("clock_skew_cycles", 4 * pulse.ns), ("peer_latency_cycles", {1: 4 * pulse.ns})],
)
def test_a_controllers_cycles_are_its_own_not_a_time(name, value):
    with pytest.raises(
        pulse.InvalidTypeError, match=f"Controller.{name} is a whole number of the controller's own"
    ):
        pulse.Controller(**{name: value})


def test_a_controllers_feedback_latency_is_a_whole_number_of_cycles():
    # A whole float is that many cycles, as before; a fraction of a cycle and
    # a bool are not cycles, as for the other *_cycles fields.
    assert pulse.Controller(feedback_latency_cycles=6.0).feedback_latency_cycles == 6
    assert type(pulse.Controller(feedback_latency_cycles=6.0).feedback_latency_cycles) is int
    with pytest.raises(
        pulse.InvalidValueError,
        match=r"Controller.feedback_latency_cycles is a whole number of cycles, got 2.5",
    ):
        pulse.Controller(feedback_latency_cycles=2.5)
    with pytest.raises(
        pulse.InvalidTypeError,
        match=r"Controller.feedback_latency_cycles is a whole number of cycles, got True",
    ):
        pulse.Controller(feedback_latency_cycles=True)
    with pytest.raises(pulse.InvalidValueError, match="cannot be negative"):
        pulse.Controller(feedback_latency_cycles=-1)
    # On several controllers, 6.0 compiles to the program 6 does.
    lowered = []
    for feedback in (6, 6.0):
        qpu = _qpu(*_two_boxes(feedback_latency_cycles=feedback))
        compiled = pulse.compile(drive_and_branch, schedule="rcp", quantum_system=qpu)
        lowered.append(compiled.lower(qpu.qudits[0], qpu.qudits[2]))
    assert "feedback_latency_cycles = 6," in lowered[1].mlir()
    assert lowered[1].mlir() == lowered[0].mlir()
    assert lowered[1].schedule == lowered[0].schedule


def test_several_controllers_on_their_own_clocks_schedule_on_their_grids():
    qpu = _qpu(*_two_boxes(clock_ghz=1.0), clock_ghz=2.0)
    lowered = pulse.compile(drive_and_branch, schedule="rcp", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    starts = [e.start for e in lowered.schedule]
    assert starts and all(start % 2 == 0 for start in starts)


def test_a_program_needing_a_capability_a_controller_lacks_does_not_compile():
    a, b = _two_boxes(supports=["play"])
    qpu = _qpu(a, b)
    compiled = pulse.compile(drive_and_branch, quantum_system=qpu)
    with pytest.raises(ControllerLimitError) as raised:
        compiled.lower(qpu.qudits[0], qpu.qudits[2])
    text = str(raised.value)
    # The branch plays on box.b, which forwards nothing but must branch and
    # sync; box.a forwards the outcome.
    assert "controllers[1] (box.b): the drive on line 2 (qudit 2)" in text
    assert "needs 'branch'" in text and "needs 'sync'" in text
    full = pulse.Controller(
        drive_lines=[0, 1],
        readout_lines=[0, 1],
        supports=[c for c in pulse.Capability if c != "forward_outcome"],
    )
    qpu = _qpu(full, b)
    with pytest.raises(
        ControllerLimitError, match=r"controllers\[0\]: the readout .* needs 'forward_outcome'"
    ):
        pulse.compile(drive_and_branch, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    # Both on one controller: nothing to forward.
    pulse.compile(drive_and_branch, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[1])


@pulse.classical(max_latency_ns=400, link="feedback")
def settle(x: float) -> float:
    return x


@pulse.kernel
def plays(q):
    d = get_drive_line(q)
    drive(d, square(20, 0.1))


@pulse.kernel
def acquires(q):
    r = get_readout_line(q)
    readout(r, square(100, 0.05))


@pulse.kernel
def waits_for_an_answer(q):
    d = get_drive_line(q)
    wait(d, settle.submit(8.0).result(), bounds=(0, 800))
    drive(d, square(20, 0.1))


@pulse.kernel
def syncs(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, square(20, 0.1))
    sync(d, r)
    readout(r, square(100, 0.05))


@pulse.kernel
def turns_by_an_outcome(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(100, 0.05))
    phase = 0.1
    if m:
        phase = 0.5
    shift_phase(d.tone, phase)
    drive(d, square(20, 0.1))


@pulse.kernel
def turns_by_a_parameter(q, phase: float):
    d = get_drive_line(q)
    shift_phase(d.tone, phase)
    drive(d, square(20, 0.1))


@pulse.kernel(unroll_threshold=0)
def loops(q):
    d = get_drive_line(q)
    for _ in range(4):
        drive(d, square(20, 0.1))


# For each capability, a program that needs it and the qudits it runs on:
# two on one controller, or (0, 2) across two.
_NEEDING = {
    pulse.Capability.PLAY: (plays, (0,)),
    pulse.Capability.READOUT: (acquires, (0,)),
    pulse.Capability.WAIT: (waits_for_an_answer, (0,)),
    pulse.Capability.SYNC: (syncs, (0,)),
    pulse.Capability.CALL: (corrected, (0, 1)),
    pulse.Capability.ANSWER: (corrected, (0, 1)),
    pulse.Capability.BRANCH: (drive_and_branch, (0, 1)),
    pulse.Capability.RUNTIME_FRAME_UPDATE: (turns_by_an_outcome, (0,)),
    pulse.Capability.FORWARD_OUTCOME: (drive_and_branch, (0, 2)),
    pulse.Capability.LOOP: (loops, (0,)),
}


def test_every_capability_has_a_program_that_needs_it():
    assert set(_NEEDING) == set(pulse.Capability)


@pytest.mark.parametrize("capability", list(pulse.Capability))
def test_a_controller_lacking_only_one_capability_rejects_what_needs_it(capability):
    kernel, placed = _NEEDING[capability]
    others = [c for c in pulse.Capability if c != capability]
    a = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], supports=others)
    b = pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3])
    qpu = _qpu(a, b, links={"feedback": pulse.Link(controllers=[a])})
    qudits = [qpu.qudits[i] for i in placed]
    full = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1])
    complete = _qpu(full, b, links={"feedback": pulse.Link(controllers=[full])})
    pulse.compile(kernel, schedule="rcp", quantum_system=complete).lower(
        *[complete.qudits[i] for i in placed]
    )
    if capability == pulse.Capability.LOOP:
        # Without it, the controller plays the loop unrolled.
        assert pulse.compile(kernel, quantum_system=qpu).lower(*qudits).export_to_controller()
        return
    with pytest.raises(ControllerLimitError, match=f"controllers\\[0\\]: .* needs '{capability}'"):
        pulse.compile(kernel, quantum_system=qpu).lower(*qudits)


def test_a_tone_a_parameter_sets_needs_no_runtime_frame_update():
    others = [c for c in pulse.Capability if c != pulse.Capability.RUNTIME_FRAME_UPDATE]
    qpu = _qpu(pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], supports=others))
    compiled = pulse.compile(turns_by_a_parameter, quantum_system=qpu)
    assert (
        compiled.lower(qpu.qudits[0], 0.3).export_to_controller()["events"][0]["phase_rad"] == 0.3
    )


@pulse.kernel(unroll_threshold=0)
def two_loops(q: pulse.Qudit, r: pulse.Qudit):
    d = get_drive_line(q)
    e = get_drive_line(r)
    for _ in range(30):
        drive(d, square(20, 0.1))
        drive(d, square(20, 0.2))
    for _ in range(10):
        drive(e, square(20, 0.1))


def test_a_controller_without_loop_gets_its_loops_unrolled_for_it_only():
    no_loop = [c for c in pulse.Capability if c != "loop"]
    a, b = _two_boxes(max_pulses=4)
    a = pulse.Controller(drive_lines=[0, 1], model="box.a", max_pulses=4, supports=no_loop)
    qpu = _qpu(a, b)
    roomy = _qpu(pulse.Controller(drive_lines=[0, 1], supports=no_loop), b)
    assert "scf.for" in two_loops.lower(roomy.qudits[0], roomy.qudits[2]).mlir()
    # Without loop, box.a plays all 60 drives of its loop, naming it; box.b
    # holds its loop's body once.
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(two_loops, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    text = str(raised.value)
    assert "controllers[0] (box.a): 60 drives and readouts" in text
    assert "the loop of 30 iterations from cycle 0 plays 58 more" in text
    assert "box.b" not in text
    # Unrolling adds pulses, not waveforms: memory for one shape fits either way.
    for supports in (None, no_loop):
        held = pulse.Controller(drive_lines=[0, 1], waveform_memory_samples=20, supports=supports)
        qpu = _qpu(held, b)
        pulse.compile(two_loops, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    # With loop, a body over max_pulses is counted once and still fails.
    small = pulse.Controller(drive_lines=[0, 1], model="box.a", max_pulses=1)
    qpu = _qpu(small, b)
    with pytest.raises(ControllerLimitError, match="body counted once; the controller holds"):
        pulse.compile(two_loops, quantum_system=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    # A controller without loop counts every iteration of both loops.
    machine = ControlSystem(controllers=[Controller(max_pulses=60, supports=no_loop)])
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(two_loops, control_system=machine).lower(*qpu.qudits[:2])
    text = str(raised.value)
    assert "70 drives and readouts; the controller plays at most 60" in text
    assert text.endswith(
        "the loop of 30 iterations from cycle 0 plays 58 more; "
        "the loop of 10 iterations from cycle 1000 plays 9 more"
    )


def test_a_call_sends_and_answers_only_controllers_its_link_reaches():
    a, b = _two_boxes()
    qpu = _qpu(a, b, links={"feedback": pulse.Link(controllers=[a])})
    compiled = pulse.compile(corrected, quantum_system=qpu)
    compiled.lower(qpu.qudits[0], qpu.qudits[1])
    with pytest.raises(ControllerLimitError) as raised:
        compiled.lower(qpu.qudits[2], qpu.qudits[0])
    assert "over link 'feedback' sends the measurements of controllers[1] (box.b)" in str(
        raised.value
    )
    with pytest.raises(ControllerLimitError, match="answers controllers\\[1\\]"):
        compiled.lower(qpu.qudits[0], qpu.qudits[2])


def _answered_rounds(n):
    # Each round's shots branch on the answer about the round before's.
    @pulse.kernel(unroll_threshold=10**9)
    def rounds(q, r):
        ro = get_readout_line(q)
        d = get_drive_line(r)
        bits = [readout(ro, square(20, 0.05)) for _ in range(n)]
        for _ in range(4):
            fix = flip.submit(bits).result()
            bits = []
            for _ in range(n):
                if fix[0]:
                    drive(d, square(16, 0.1))
                bits.append(readout(ro, square(20, 0.05)))

    return rounds


def _two_box_qpu():
    return _qpu(*_two_boxes(peer_latency_cycles={0: 12}), links={"feedback": Link(slot_bytes=4096)})


@pytest.mark.slow
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_rounds_on_two_controllers_that_branch_on_an_answer_schedule_in_linear_time(schedule):
    # The answer reaches the drive's controller apart from the readouts it
    # comes from; the time, native, is of whole compiles (a program lowers
    # once, so lowering it again would time nothing).
    qpu = _two_box_qpu()
    runs = {}
    for n in (100, 1600):
        kernel = _answered_rounds(n)
        runs[n] = lambda kernel=kernel: pulse.compile(
            kernel, quantum_system=qpu, schedule=schedule
        ).lower(qpu.qudits[0], qpu.qudits[2])
    timings = native_seconds_each(runs, repeats=3)
    # 16x the shots: 16x the time when linear, 256x when quadratic. 1600
    # shots outgrow a shared runner's caches, which can take it past 64x.
    assert timings[1600] < 128 * timings[100], timings


@pulse.kernel
def answered_before_a_loop(q, r):
    ro = get_readout_line(q)
    d = get_drive_line(r)
    fix = flip.submit([readout(ro, square(20, 0.05)), readout(ro, square(30, 0.05))]).result()
    for _ in range(6):
        if fix[0]:
            drive(d, square(16, 0.1))
        wait(d, 40)


@pulse.kernel
def answered_in_each_iteration(q, r):
    ro = get_readout_line(q)
    d = get_drive_line(r)
    for _ in range(6):
        fix = flip.submit([readout(ro, square(20, 0.05)), readout(ro, square(30, 0.05))]).result()
        if fix[0]:
            drive(d, square(16, 0.1))
        drive(d, square(8, 0.1))


def _controller_events(kernel, qpu, schedule):
    lowered = pulse.compile(kernel, quantum_system=qpu, schedule=schedule).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    return lowered, [
        [
            (e["kind"], e["start_cycles"], e["duration_cycles"], len(e["conditions"]))
            for e in lowered.export_to_controller(controller=c)["events"]
        ]
        for c in (0, 1)
    ]


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
@pytest.mark.parametrize("kernel", [answered_before_a_loop, answered_in_each_iteration])
def test_a_rolled_loop_on_two_controllers_that_branches_on_an_answer_schedules_as_unrolled(
    kernel, schedule
):
    # The bounds of each iteration's branches on the answer reach the
    # readouts it comes from, as when every iteration is written out.
    qpu = _two_box_qpu()
    rolled, events = _controller_events(kernel, qpu, schedule)
    assert rolled._concrete().print().count("scf.for") == 1
    unrolled = pulse.kernel(unroll_threshold=10**9)(kernel.__wrapped__)
    assert events == _controller_events(unrolled, qpu, schedule)[1]


def test_runtime_qudits_are_checked_against_the_wiring_per_call():
    @pulse.kernel
    def pulse_one(q: pulse.Qudit):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))

    qpu = _qpu(*_two_boxes(), layout=range(6))
    compiled = pulse.compile(pulse_one, quantum_system=qpu)
    compiled(qpu.qudits[3])
    with pytest.raises(ControllerLimitError, match="qudit 5\\) is on no controller"):
        compiled(qpu.qudits[5])


# ── System files ─────────────────────────────────────────────────────────────


def _round_trip(qpu):
    data = qpu.as_dict()
    assert data["format"] == "cudaq-pulse.system/1"
    assert data["written_by"].startswith("cudaq-pulse ")
    back = pulse.QuantumSystem.from_dict(json.loads(json.dumps(data)))
    assert back == qpu and back.fingerprint == qpu.fingerprint
    assert back.as_dict() == data
    return data


def test_a_qpu_round_trips_through_its_system_file():
    a, b = _two_boxes(
        peer_latency_cycles={0: 12}, supports=["play", "readout", "sync"], backend="toy"
    )
    qpu = _qpu(
        a, b, links={"feedback": pulse.Link(fixed_ns=40.0, controllers=[b])}, layout=range(6)
    )
    data = _round_trip(qpu)
    assert data["name"] == TRANSMON_SURFACE17 and data["calibration"] == "transmon_surface17"
    system = data["control_system"]
    assert system["links"]["feedback"]["controllers"] == [1]
    assert system["controllers"][1]["peer_latency_cycles"] == {"0": 12}
    assert system["controllers"][1]["supports"] == ["play", "readout", "sync"]


def test_a_qudit_s_drive_and_readout_round_trip_through_the_system_file():
    readout = DispersiveReadout(frequency_hz=7.1e9, linewidth_hz=1e6, error_1to0=0.02)
    qudits = {
        0: PhysicalQudit(
            0,
            5.0e9,
            50.0,
            30.0,
            anharmonicity_hz=-2e8,
            drive=MicrowaveDrive(amplitude_scale_rad_per_ns=0.3),
            readout=readout,
        ),
        1: PhysicalQudit(1, 5.1e9, 50.0, 30.0, anharmonicity_hz=-2e8),
    }
    dev = pulse.quantum_system(name="ro", qudits=qudits)
    written = _round_trip(dev)["qudits"]
    assert written[0]["readout"]["kind"] == "dispersive"
    assert written[0]["readout"]["frequency_hz"] == 7.1e9
    assert written[0]["drive"] == {"kind": "microwave", "amplitude_scale_rad_per_ns": 0.3}
    assert written[1]["readout"] is None and written[1]["drive"] is None
    other = PhysicalQudit(
        0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-2e8, readout=readout.replace(linewidth_hz=2e6)
    )
    moved = pulse.quantum_system(name="ro", qudits={**qudits, 0: other})
    assert moved.fingerprint != dev.fingerprint


def test_a_system_file_holds_a_drive_of_no_stated_modality():
    drive = pulse.quantum_systems.Drive(amplitude_scale_rad_per_ns=0.3)
    qudits = {0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-2e8, drive=drive)}
    dev = pulse.quantum_system(name="plain", qudits=qudits)
    written = dev.as_dict()["qudits"][0]["drive"]
    assert written == {"kind": "drive", "amplitude_scale_rad_per_ns": 0.3}
    back = pulse.QuantumSystem.from_dict(json.loads(json.dumps(dev.as_dict())))
    assert back.physical_qudits[0].drive == drive and back.fingerprint == dev.fingerprint
    microwave = qudits[0].replace(drive=MicrowaveDrive(amplitude_scale_rad_per_ns=0.3))
    other = pulse.quantum_system(name="plain", qudits={0: microwave})
    assert other.fingerprint != dev.fingerprint
    # Each kind reads back as its own class, from the same scale.
    for system, kind, made in (
        (dev, "drive", pulse.quantum_systems.Drive),
        (other, "microwave", MicrowaveDrive),
    ):
        data = json.loads(json.dumps(system.as_dict()))
        assert data["qudits"][0]["drive"]["kind"] == kind
        read = pulse.QuantumSystem.from_dict(data).physical_qudits[0].drive
        assert type(read) is made and read.amplitude_scale_rad_per_ns == 0.3


def test_a_system_file_refuses_an_unknown_component_kind_or_field():
    data = pulse.quantum_system(
        name="ro", qudits={0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-2e8)}
    ).as_dict()
    qudit = data["qudits"][0]
    qudit["readout"] = {"kind": "fluorescence"}
    qudit["drive"] = {"kind": "microwave", "amplitude_scale": 0.2}
    with pytest.raises(QuantumSystemDefinitionError) as found:
        pulse.QuantumSystem.from_dict(data)
    message = str(found.value)
    assert "qudits[0].readout.kind" in message and "the kinds are dispersive" in message
    qudit["drive"] = {"kind": "optical"}
    with pytest.raises(QuantumSystemDefinitionError, match="the kinds are microwave, drive"):
        pulse.QuantumSystem.from_dict(data)
    assert "qudits[0].drive.amplitude_scale" in message
    assert "did you mean 'amplitude_scale_rad_per_ns'" in message


def test_line_delays_and_a_reference_clock_round_trip_through_the_system_file():
    a, b = _two_boxes()
    a = dataclasses.replace(a, drive_lines=[DriveLine(0, delay_ns=1.5), *a.drive_lines[1:]])
    b = dataclasses.replace(b, readout_lines=[2, ReadoutLine(3, delay_ns=2.0)])
    system = pulse.ControlSystem(controllers=[a, b], reference_clock_ghz=0.25)
    qpu = pulse.quantum_system(TRANSMON_SURFACE17, layout=[0, 1, 2, 3], control_system=system)
    data = _round_trip(qpu)
    control = data["control_system"]
    assert control["controllers"][0]["drive_lines"][0] == {"qudit": 0, "delay_ns": 1.5}
    assert control["controllers"][1]["readout_lines"] == [2, {"qudit": 3, "delay_ns": 2.0}]
    assert control["reference_clock_ghz"] == 0.25 and "line_delays_ns" not in control
    control["controllers"][0]["drive_lines"][0] = {"qudit": 0, "delay": 1.0}
    control["line_delays_ns"] = {"drive": {"0": 1.0}}
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.from_dict(data)
    found = [(p.path, p.message) for p in raised.value.problems]
    assert ("control_system.line_delays_ns", "is not a known key") in found
    assert ("control_system.controllers[0].drive_lines[0].delay", "is not a known key") in found


def test_a_program_clock_round_trips_through_the_system_file():
    system = pulse.ControlSystem(controllers=[pulse.Controller(clock_ghz=1.0)])
    qpu = pulse.quantum_system(
        TRANSMON_SURFACE17, layout=[0, 1], control_system=system, clock_ghz=4.0
    )
    assert qpu.clock_ghz == 4.0
    assert _round_trip(qpu)["clock_ghz"] == 4.0
    assert "clock_ghz" not in _round_trip(_qpu(pulse.Controller(), layout=[0, 1]))


def test_a_custom_quantum_system_is_written_inline_with_the_new_key_names():
    qpu = pulse.quantum_system(
        name="pair",
        qudits={
            i: PhysicalQudit(i, 5e9 + i * 1e8, 50.0, 40.0, anharmonicity_hz=-2.5e8)
            for i in range(2)
        },
        couplings=[Coupling((0, 1), 3e6)],
    )
    data = _round_trip(qpu)
    assert data["couplings"] == [{"qudits": [0, 1], "strength_hz": 3e6}]
    assert [q["index"] for q in data["qudits"]] == [0, 1]
    assert "control_system" not in data


def test_a_system_file_reports_every_problem_with_its_line(tmp_path):
    qpu = _qpu(*_two_boxes())
    data = qpu.as_dict()
    data["control_system"]["controllers"][1]["drive_lines"] = [2, 30]
    data["control_system"]["controllers"][0]["instruction_cycle"] = 2
    path = tmp_path / "qpu.json"
    path.write_text(json.dumps(data, indent=2))
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.load(path)
    unknown, wiring = raised.value.problems
    assert unknown.path == "control_system.controllers[0].instruction_cycle"
    assert "did you mean 'instruction_cycles'?" in str(raised.value)
    assert wiring.path == "control_system.controllers[1].drive_lines[1]" and wiring.value == 30
    text = path.read_text().splitlines()
    assert unknown.file == wiring.file == str(path)
    assert text[wiring.line - 1].strip().startswith("30")
    assert '"instruction_cycle"' in text[unknown.line - 1]
    assert str(raised.value).splitlines()[1].strip().startswith(f"{path}:{unknown.line}:")


def test_a_newer_system_file_format_is_refused():
    with pytest.raises(QuantumSystemDefinitionError, match="newer than this cudaq-pulse reads"):
        pulse.QuantumSystem.from_dict(
            {"format": "cudaq-pulse.system/2", "name": TRANSMON_SURFACE17}
        )
    with pytest.raises(QuantumSystemDefinitionError, match="not a system file format"):
        pulse.QuantumSystem.from_dict({"format": "something/1", "name": TRANSMON_SURFACE17})


# ── Controller descriptions ──────────────────────────────────────────────────


def test_a_description_is_loaded_by_name_with_the_installation_wiring():
    box = pulse.controller("reference.controller", drive_lines=[0, 1], readout_lines=[0])
    assert isinstance(box, pulse.Controller)
    assert (box.model, box.version) == ("reference.controller", "1")
    assert box.drive_lines == (DriveLine(0), DriveLine(1))
    assert box.readout_lines == (ReadoutLine(0),)
    assert pulse.controller("reference.controller") is pulse.controller("reference.controller")
    with pytest.raises(KeyError, match="did you mean 'reference.controller'"):
        pulse.controller("reference.controler")
    with pytest.raises(TypeError, match="did you mean 'drive_lines'"):
        pulse.controller("reference.controller", drive_line=[0])


@pytest.mark.parametrize("name", sorted(installed_descriptions()))
def test_every_installed_description_loads_and_compiles(name):
    path = installed_descriptions()[name]
    assert "owner" not in tomllib.loads(path.read_text(encoding="utf-8"))
    box = pulse.controller(name, drive_lines=range(4), readout_lines=range(4))
    assert box.model == name
    qpu = _qpu(box)
    pulse.compile(drive_and_branch, schedule="rcp", quantum_system=qpu).lower(
        qpu.qudits[0], qpu.qudits[1]
    )
    # A description that names its backend opts in to the backend's checks.
    validate = getattr(pulse.backends, "validate", None)
    if box.backend is not None and validate is not None:
        validate(pulse.backends.get(box.backend), controller=box)


def test_a_description_file_reports_every_problem_with_its_line(tmp_path):
    path = tmp_path / "box.toml"
    path.write_text('model = "lab.box"\ninstruction_cycle = 2\ndrive_lines = [0]\n')
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.controller(path)
    problems = {p.path: p for p in raised.value.problems}
    assert (
        problems["instruction_cycle"].line == 2
        and "instruction_cycles" in problems["instruction_cycle"].fix
    )
    assert problems["drive_lines"].line == 3 and "installation" in problems["drive_lines"].message
    assert any("no version" in p.message for p in raised.value.problems)
    path.write_text('model = "lab.box"\nversion = "2"\ninstruction_cycles = 3\n')
    box = pulse.controller(path, drive_lines=[0])
    assert box.instruction_cycles == 3 and box.model == "lab.box"
    path.write_text('model = "lab.box"\nversion = "2"\nowner = "the lab"\n')
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.controller(path)
    (problem,) = raised.value.problems
    assert (problem.path, problem.message, problem.line) == (
        "owner",
        "is not a controller field",
        3,
    )


def test_a_description_file_written_again_is_the_one_loaded(tmp_path):
    # Each write keeps the first one's time stamp, as writes within one tick
    # of the file system's clock do, or a copy that keeps times.
    path = tmp_path / "box.toml"

    def write(text):
        stamp = os.stat(path).st_mtime_ns if path.exists() else None
        path.write_text(text)
        if stamp is not None:
            os.utime(path, ns=(stamp, stamp))
            assert os.stat(path).st_mtime_ns == stamp

    write('model = "lab.box"\nversion = "2"\ninstruction_cycles = 3\n')
    first = pulse.controller(path, drive_lines=[0])
    assert first.instruction_cycles == 3
    assert pulse.controller(path) is pulse.controller(path)
    write('model = "lab.box"\nversion = "2"\ninstruction_cycles = 7\n')
    assert pulse.controller(path, drive_lines=[0]).instruction_cycles == 7
    write('model = "lab.box"\nversion = "2"\nowner = "the lab"\n')
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.controller(path)
    assert [p.path for p in raised.value.problems] == ["owner"]
    # The first content again is the controller parsed for it.
    write('model = "lab.box"\nversion = "2"\ninstruction_cycles = 3\n')
    assert pulse.controller(path, drive_lines=[0]) == first


def test_a_description_file_changed_to_the_same_length_is_the_one_loaded(tmp_path):
    path = tmp_path / "box.toml"
    path.write_text('model = "lab.box"\nversion = "2"\ninstruction_cycles = 3\n')
    stamp, size = os.stat(path).st_mtime_ns, os.stat(path).st_size
    assert pulse.controller(path).instruction_cycles == 3
    path.write_text('model = "lab.box"\nversion = "2"\ninstruction_cycles = 7\n')
    os.utime(path, ns=(stamp, stamp))
    assert (os.stat(path).st_mtime_ns, os.stat(path).st_size) == (stamp, size)
    assert pulse.controller(path).instruction_cycles == 7


def test_a_description_file_reads_as_text_whatever_its_line_ends(tmp_path):
    path = tmp_path / "box.toml"
    path.write_bytes(b'model = "lab.box"\r\nversion = "2"\r\ninstruction_cycle = 2\r\n')
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.controller(path)
    assert {p.path: p.line for p in raised.value.problems}["instruction_cycle"] == 3


def test_a_system_file_from_another_description_version_is_noticed_once():
    qpu = _qpu(pulse.controller("reference.controller", drive_lines=range(4)))
    data = qpu.as_dict()
    data["control_system"]["controllers"][0]["version"] = "0-test"
    with pytest.warns(
        UpgradeNotice, match="version 0-test; the installed description is version 1"
    ):
        old = pulse.QuantumSystem.from_dict(data)
    assert old.control_system.controllers[0].version == "0-test"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pulse.QuantumSystem.from_dict(data)
        pulse.QuantumSystem.from_dict(qpu.as_dict())


# ── doctor --system ──────────────────────────────────────────────────────────


def _doctor(path, *flags):
    from cudaq_pulse.doctor import main

    out = io.StringIO()
    return main(["--system", str(path), *flags], out=out), out.getvalue()


def test_doctor_checks_a_system_file(tmp_path):
    a, b = _two_boxes()
    qpu = _qpu(a, b, links={"feedback": pulse.Link(controllers=[a])})
    path = tmp_path / "qpu.json"
    path.write_text(json.dumps(qpu.as_dict(), indent=2))
    code, text = _doctor(path)
    assert code == 0, text
    assert f"{path}: ok, quantum system {TRANSMON_SURFACE17} with 4 qudits" in text
    assert "controllers[1]: box.b, drive lines 2..3" in text
    assert "link 'feedback': reaches controllers [0]" in text
    data = qpu.as_dict()
    data["control_system"]["controllers"][0]["backend"] = "no_such_backend"
    path.write_text(json.dumps(data, indent=2))
    code, text = _doctor(path)
    assert code == 1 and "backend no_such_backend (not installed)" in text
    assert "control_system.controllers[0].backend: backend 'no_such_backend' is not" in text
    code, text = _doctor(path, "--json")
    report = json.loads(text)
    assert code == 1 and report["quantum_system"]["control_system"]["links"] == {"feedback": [0]}
    data["control_system"]["controllers"][0]["readout_lines"] = [0, 99]
    path.write_text(json.dumps(data, indent=2))
    code, text = _doctor(path)
    assert code == 1 and f"{path}:" in text and "readout_lines[1]: qudit 99" in text


def test_a_system_file_keeps_the_simulation_model():
    qpu = pulse.quantum_system(
        name="other-pair",
        qudits={
            i: PhysicalQudit(i, 5e9 + i * 1e8, 50.0, 40.0, anharmonicity_hz=0.0) for i in range(2)
        },
        model=SimulationModel(levels={1: 3}),
    )
    data = qpu.as_dict()
    assert data["model"] == {"kind": "generic", "levels": {"1": 3}}
    again = pulse.QuantumSystem.from_dict(json.loads(json.dumps(data)))
    assert again.model == qpu.model and again == qpu

    @pulse.kernel
    def nothing(q):
        d = get_drive_line(q)
        drive(d, square(20, 0.1))

    with pytest.raises(UnsupportedError, match="SimulationModel has no simulator"):
        pulse.simulate(pulse.compile(nothing).lower(again.qudits[0]))

    data["model"] = None
    assert pulse.QuantumSystem.from_dict(data).model is None
    data["model"] = {"kind": "fluxonium"}
    with pytest.raises(
        QuantumSystemDefinitionError, match="model.kind: must be one of transmon, generic"
    ):
        pulse.QuantumSystem.from_dict(data)


def test_a_system_file_names_a_model_it_cannot_hold():
    class Custom(SimulationModel):
        pass

    target = pulse.quantum_system(
        name="custom-pair",
        qudits={
            i: PhysicalQudit(i, 5e9 + i * 1e8, 50.0, 40.0, anharmonicity_hz=0.0) for i in range(2)
        },
        model=Custom,
    )
    with pytest.raises(InvalidValueError, match="Custom is not one"):
        target.as_dict()
