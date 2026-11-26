# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every schedule starts an operation that depends on a readout once the
readout has ended and the controller's feedback latency has passed: a
branch on it (the direct case), a pulse on a tone a branch on it changed
(A), a measured value out of an if (B), a request that sends it (C)."""

from __future__ import annotations

import os
import subprocess
import sys

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.controller import ReadoutLine
from cudaq_pulse.quantum_systems import PhysicalQudit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POLICIES = ["asap", "alap", None, "rcp", "alap_rcp"]


def _qudits():
    return {
        i: PhysicalQudit(i, 4.5e9 + 1.0e5 * i, 37.7, 9.2, anharmonicity_hz=-215e6) for i in range(2)
    }


def _qpu(latency, clock_ghz=2.0, steps=8):
    qudits = _qudits()
    extra = {"feedback_latency_cycles": latency} if latency else {}
    controller = Controller(
        clock_ghz=clock_ghz,
        cycles_per_timestep=steps,
        drive_lines=[0, 1],
        readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2)],
        max_amplitude=1.0,
        **extra,
    )
    system = ControlSystem(
        controllers=[controller],
        reference_clock_ghz=clock_ghz,
        max_concurrent_drives=2,
        max_concurrent_readouts=2,
    )
    return pulse.quantum_system(
        name=f"latency_{latency}_{clock_ghz}", qudits=qudits, control_system=system, clock_ghz=2.0
    )


def _lowered(kernel, qpu, policy):
    options = {"schedule": policy} if policy else {}
    return pulse.compile(kernel, quantum_system=qpu, **options).lower(qpu.qudits[0:2])


def _starts(lowered, kind):
    return sorted(e.start_cycles for e in lowered.schedule if e.kind == kind)


@pulse.kernel
def drive_in_if(qs):
    d = get_drive_line(qs[0])
    r = get_readout_line(qs[0])
    bit = readout(r, square(400, 0.05))
    if bit:
        drive(d, square(40, 0.3))


@pulse.kernel
def phase_in_if(qs):
    d = get_drive_line(qs[0])
    r = get_readout_line(qs[0])
    bit = readout(r, square(400, 0.05))
    if bit:
        shift_phase(d.tone, 3.14)
    drive(d, square(40, 0.3))


@pulse.kernel
def remeasure_in_if(qs):
    d1 = get_drive_line(qs[1])
    r0 = get_readout_line(qs[0])
    m = readout(r0, square(48, 0.05))
    if m:
        wait(r0, 2000)
        m = readout(r0, square(400, 0.05))
    if m:
        drive(d1, square(40, 0.5))


@pulse.classical(max_latency_ns=100)
def echo_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def call_branch(qs):
    d = get_drive_line(qs[0])
    r = get_readout_line(qs[0])
    m = readout(r, square(400, 0.05))
    fix = echo_bit.submit([m]).result()
    if fix[0]:
        drive(d, square(40, 0.3))


@pulse.kernel
def two_readouts(qs):
    r = get_readout_line(qs[0])
    readout(r, square(400, 0.05))
    readout(r, square(400, 0.05))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("latency", [0, 400])
def test_a_branch_on_a_readout_waits_for_the_latency(latency, policy):
    # The direct case. Before, asap and alap started the drive at 400 and the
    # controller check refused it ("schedule with rcp").
    lowered = _lowered(drive_in_if, _qpu(latency), policy)
    assert _starts(lowered, "drive") == [400 + latency]
    assert pulse.controller.check(lowered) == []


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("latency", [0, 400])
def test_a_pulse_on_a_tone_a_branch_changed_waits_for_the_latency(latency, policy):
    # A: the drive plays the phase the bit chose. Before, asap and alap
    # started it at 400, and nothing refused it.
    lowered = _lowered(phase_in_if, _qpu(latency), policy)
    assert _starts(lowered, "drive") == [400 + latency]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("latency", "drive"), [(0, 2448), (400, 3248)])
def test_a_measured_value_out_of_an_if_waits_for_the_latency(latency, drive, policy):
    # B: the arm (wait and second readout) waits for the first readout's
    # outcome, 48 + 400, and the drive for the second's, 2848 + 400. Before,
    # asap and alap started the drive at 2448.
    lowered = _lowered(remeasure_in_if, _qpu(latency), policy)
    assert _starts(lowered, "drive") == [drive]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("latency", "submit", "drive"), [(0, 400, 608), (400, 800, 1008)])
def test_a_request_that_sends_a_readout_waits_for_the_latency(latency, submit, drive, policy):
    # C: the request leaves once the outcome is usable. Before, asap and
    # alap sent it at 400.
    lowered = _lowered(call_branch, _qpu(latency), policy)
    [call] = lowered.classical_report()
    assert call.submit_cycles == submit
    assert _starts(lowered, "drive") == [drive]


@pytest.mark.parametrize("policy", ["asap", "alap", None])
def test_readouts_nothing_depends_on_do_not_wait(policy):
    # The line is not held for the latency: only rcp holds it.
    lowered = _lowered(two_readouts, _qpu(400), policy)
    assert _starts(lowered, "readout") == [0, 400]
    assert _starts(_lowered(two_readouts, _qpu(400), "rcp"), "readout") == [0, 800]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("clock_ghz", "program"), [(0.5, 400), (1.0, 200)])
def test_one_controller_at_another_clock_waits_in_program_cycles(clock_ghz, program, policy):
    # 100 cycles of feedback latency at 0.5 and 1 GHz under a 2 GHz program.
    qpu = _qpu(100, clock_ghz=clock_ghz, steps=1)
    lowered = _lowered(drive_in_if, qpu, policy)
    assert _starts(lowered, "drive") == [400 + program]
    assert pulse.controller.check(lowered) == []
    [call] = _lowered(call_branch, qpu, policy).classical_report()
    assert call.submit_cycles == 400 + program


@pytest.mark.parametrize("policy", ["asap", "alap", None])
def test_a_kernel_compiled_for_no_control_system_waits_for_the_qudits_one(policy):
    # Lowered on qudits whose controller has 400 cycles of feedback latency.
    # Before, nothing waited and nothing checked.
    qpu = _qpu(400)
    options = {"schedule": policy} if policy else {}
    lowered = pulse.compile(drive_in_if, **options).lower(qpu.qudits[0:2])
    assert _starts(lowered, "drive") == [800]
    assert lowered.options.machine is qpu.control_system._machine


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
@pytest.mark.parametrize("compiled", [True, False])
def test_one_kernel_lowered_on_several_control_systems_gets_each_one(compiled, order):
    # Quantum systems that differ only in their control system share a
    # fingerprint. Before, the cache keyed by it handed the second lowering
    # the first one's schedule and machine, whichever came first.
    plain = _qpu(400).replace(control_system=None)
    systems = [(plain, 400), (_qpu(0), 400), (_qpu(400), 800)]
    kernel = pulse.compile(drive_in_if) if compiled else drive_in_if
    for i in order:
        qpu, drive = systems[i]
        lowered = kernel.lower(qpu.qudits[0:2])
        assert _starts(lowered, "drive") == [drive]
        if qpu.control_system is None:
            assert lowered.options.machine.controller is None
            continue
        # Equal control systems share an entry (drive_in_if is lowered by
        # other tests too).
        assert lowered.options.machine == qpu.control_system._machine
        [controller] = qpu.control_system.controllers
        assert pulse.controller.check(lowered, controller) == []


@pytest.mark.parametrize("plain_first", [True, False])
def test_a_kernel_compiled_for_no_control_system_is_checked_against_the_qudits_one(plain_first):
    # The qudits' controller plays amplitudes up to 0.1: the drive of 0.3 is
    # refused there, and compiles without a control system, in either order.
    weak = Controller(clock_ghz=2.0, drive_lines=[0, 1], readout_lines=[0, 1], max_amplitude=0.1)
    qpu = pulse.quantum_system(
        name="weak", qudits=_qudits(), control_system=ControlSystem(controllers=[weak])
    )
    plain = qpu.replace(control_system=None)
    kernel = pulse.compile(drive_in_if)
    if plain_first:
        assert _starts(kernel.lower(plain.qudits[0:2]), "drive") == [400]
    with pytest.raises(pulse.controller.ControllerLimitError, match="amplitude"):
        kernel.lower(qpu.qudits[0:2])
    assert _starts(kernel.lower(plain.qudits[0:2]), "drive") == [400]


@pytest.mark.parametrize("policy", POLICIES)
def test_a_readout_line_s_fine_delay_is_waited_for_rounded_up(policy):
    # 0.3 ns at 2 GHz is 0.6 of a cycle the hardware adds below the grid:
    # the readout ends as played 1 cycle after its scheduled end. Before,
    # every schedule waited from the scheduled end.
    controller = Controller(
        drive_lines=[0, 1],
        readout_lines=[ReadoutLine(0, delay_ns=0.3), ReadoutLine(1)],
        max_fine_delay_ns=1.0,
        feedback_latency_cycles=400,
    )
    qpu = pulse.quantum_system(
        name="fine_delay", qudits=_qudits(), control_system=ControlSystem(controllers=[controller])
    )
    lowered = _lowered(drive_in_if, qpu, policy)
    assert _starts(lowered, "drive") == [801]


@pytest.mark.parametrize("caller", ["asap", "alap", None])
@pytest.mark.parametrize("callee", ["asap", "alap"])
def test_a_kernel_with_its_own_schedule_waits_for_the_latency(callee, caller):
    @pulse.kernel(schedule=callee)
    def phase_on_bit(q, m):
        d = get_drive_line(q)
        if m:
            shift_phase(d.tone, 3.14)
        drive(d, square(40, 0.3))

    @pulse.kernel
    def top(qs):
        m = readout(get_readout_line(qs[0]), square(400, 0.05))
        phase_on_bit(qs[1], m)

    lowered = _lowered(top, _qpu(400), caller)
    assert _starts(lowered, "drive") == [800]


@pulse.kernel(max_duration_cycles=500)
def budgeted(qs):
    d = get_drive_line(qs[0])
    r = get_readout_line(qs[0])
    m = readout(r, square(400, 0.05))
    if m:
        drive(d, square(40, 0.3))


@pytest.mark.parametrize("policy", POLICIES)
def test_a_budget_the_latency_breaks_names_it(policy):
    with pytest.raises(RuntimeError) as caught:
        _lowered(budgeted, _qpu(400), policy)
    assert (
        "kernel 'budgeted' takes 840 cycles, more than max_duration_cycles=500, waiting 400 "
        "cycles of feedback latency after the readouts its operations depend on; let its "
        "operations overlap, shorten them or raise max_duration_cycles"
    ) in str(caught.value)


def _reacting(reaction_ns):
    @pulse.classical(
        max_latency_ns=100, reaction_deadline_ns=reaction_ns, name=f"react{int(reaction_ns)}"
    )
    def respond(m: pulse.Bits) -> pulse.Bits:
        return [m[0]]

    @pulse.kernel
    def react(qs):
        d = get_drive_line(qs[0])
        r = get_readout_line(qs[0])
        m = readout(r, square(400, 0.05))
        if respond.submit([m]).result()[0]:
            drive(d, square(40, 0.3))

    return react


@pytest.mark.parametrize("policy", POLICIES)
def test_a_reaction_deadline_the_latency_breaks_names_it(policy):
    # The request cannot leave before 800: its answer is due 601 cycles
    # after the readout ends, and 300 ns allow 600.
    with pytest.raises(Exception) as caught:
        _lowered(_reacting(300), _qpu(400), policy)
    assert (
        "cannot react in time: its answer is due 601 cycles after the readouts it carries end "
        "(400 of them the feedback latency before the request can leave), but "
        "reaction_deadline_ns allows 600 (slack -1); lower max_latency_ns, use a faster link "
        "or raise reaction_deadline_ns"
    ) in str(caught.value)
    # 400 ns are met.
    [call] = _lowered(_reacting(400), _qpu(400), policy).classical_report()
    assert call.submit_cycles == 800


@pulse.kernel
def answer_across(qs):
    m = readout(get_readout_line(qs[0]), square(400, 0.05))
    fix = echo_bit.submit([m]).result()
    if fix[0]:
        drive(get_drive_line(qs[1]), square(40, 0.3))


def _two():
    """qudit 0 on controllers[0], which hands an outcome to controllers[1],
    playing qudit 1, 1000 cycles after its own 100: 1100 after a readout."""
    a = Controller(
        clock_ghz=2.0,
        drive_lines=[0],
        readout_lines=[0],
        max_amplitude=1.0,
        feedback_latency_cycles=100,
        peer_latency_cycles=1000,
    )
    b = Controller(
        clock_ghz=2.0,
        drive_lines=[1],
        readout_lines=[1],
        max_amplitude=1.0,
        feedback_latency_cycles=100,
    )
    return pulse.quantum_system(
        name="two_controllers",
        qudits=_qudits(),
        control_system=ControlSystem(controllers=[a, b], reference_clock_ghz=2.0),
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_an_arm_on_an_answer_on_another_controller_waits_for_the_readout_to_reach_it(policy):
    # As rcp does: the drive on controllers[1] waits for the readout's
    # outcome to reach it (400 + 100 of feedback and 1000 of peer latency),
    # though the answer is due at 701. Before, asap and alap started it at 601.
    lowered = _lowered(answer_across, _two(), policy)
    [call] = lowered.classical_report()
    assert call.submit_cycles == 500
    assert _starts(lowered, "drive") == [1500]


def _blocks_on_two(callee):
    @pulse.kernel(schedule=callee)
    def branch_on(q, m):
        if m:
            drive(get_drive_line(q), square(40, 0.3))

    @pulse.kernel
    def bit_into_a_branch(qs):
        m = readout(get_readout_line(qs[0]), square(400, 0.05))
        branch_on(qs[1], m)

    @pulse.kernel(schedule=callee)
    def phase_on(q, m):
        d = get_drive_line(q)
        if m:
            shift_phase(d.tone, 3.14)
        drive(d, square(40, 0.3))

    @pulse.kernel
    def bit_into_a_phase(qs):
        m = readout(get_readout_line(qs[0]), square(400, 0.05))
        phase_on(qs[1], m)

    @pulse.kernel(schedule=callee)
    def readout_and_phase(q, q2):
        d = get_drive_line(q2)
        if readout(get_readout_line(q), square(400, 0.05)):
            shift_phase(d.tone, 3.14)

    @pulse.kernel
    def phase_out_of_a_block(qs):
        readout_and_phase(qs[0], qs[1])
        drive(get_drive_line(qs[1]), square(40, 0.3))

    return [bit_into_a_branch, bit_into_a_phase, phase_out_of_a_block]


@pytest.mark.parametrize("caller", ["asap", "alap", None])
@pytest.mark.parametrize("callee", ["asap", "alap"])
@pytest.mark.parametrize("shape", [0, 1, 2])
def test_a_block_on_two_controllers_waits_for_the_outcome_to_reach_the_other(shape, callee, caller):
    # The readout on controllers[0] ends at 400; its outcome is usable on
    # controllers[1] 1100 later, as the flat schedule waits. Before, the
    # block took only the reader's 100: the drive started at 500 (a phase
    # changed in the block) and compiled, or at 1000 and was refused.
    qpu = _two()
    kernel = _blocks_on_two(callee)[shape]
    options = {"schedule": caller} if caller else {}
    lowered = pulse.compile(kernel, quantum_system=qpu, **options).lower(qpu.qudits[0:2])
    flat = pulse.compile(kernel, quantum_system=qpu, flatten_schedules=True, **options).lower(
        qpu.qudits[0:2]
    )
    assert _starts(lowered, "drive") == _starts(flat, "drive") == [1500]
    assert pulse.controller.check(lowered) == []


def _carried_across(callee, depth):
    """A loop that branches, on qudit 1's tone, on the readout of qudit 0
    made `depth` trips before."""

    @pulse.kernel(schedule=callee, unroll_threshold=0)
    def two_deep(q, q2):
        d = get_drive_line(q2)
        r = get_readout_line(q)
        a = readout(r, square(40, 0.05))
        b = readout(r, square(40, 0.05))
        for _ in range(8):
            if a:
                shift_phase(d.tone, 3.14)
            a = b
            b = readout(r, square(400, 0.05))

    @pulse.kernel(schedule=callee, unroll_threshold=0)
    def three_deep(q, q2):
        d = get_drive_line(q2)
        r = get_readout_line(q)
        a = readout(r, square(40, 0.05))
        b = readout(r, square(40, 0.05))
        c = readout(r, square(40, 0.05))
        for _ in range(8):
            if a:
                shift_phase(d.tone, 3.14)
            a = b
            b = c
            c = readout(r, square(400, 0.05))

    block = two_deep if depth == 2 else three_deep

    @pulse.kernel
    def top(qs):
        block(qs[0], qs[1])
        drive(get_drive_line(qs[1]), square(40, 0.3))

    return top


@pytest.mark.parametrize("caller", ["asap", "alap"])
@pytest.mark.parametrize("callee", ["asap", "alap"])
@pytest.mark.parametrize(("depth", "eighth"), [(2, 2480), (3, 2120)])
def test_a_measurement_carried_in_a_block_waits_for_the_other_controller(
    depth, eighth, callee, caller
):
    # The drive after the block plays the phase the eighth readout chose,
    # which ends at `eighth`: 1100 later on controllers[1]. Before, the
    # block handed the tone out at the readout's end plus 100.
    qpu = _two()
    kernel = _carried_across(callee, depth)
    lowered = pulse.compile(kernel, quantum_system=qpu, schedule=caller).lower(qpu.qudits[0:2])
    flat = pulse.compile(kernel, quantum_system=qpu, schedule=caller, flatten_schedules=True).lower(
        qpu.qudits[0:2]
    )
    for each in (lowered, flat):
        ends = sorted(
            e.start_cycles + e.duration_cycles for e in each.schedule if e.kind == "readout"
        )
        assert ends[7] == eighth
        [drive] = _starts(each, "drive")
        assert drive >= eighth + 1100
        assert pulse.controller.check(each) == []
    if caller == "asap":
        assert _starts(lowered, "drive") == _starts(flat, "drive") == [eighth + 1100]


# alap and alap_rcp end a program at its last operation: a readout line's
# hold after the last readout (for the next operation on the line, of which
# there is none) and a sync's instant (it plays nothing) do not lengthen it.


@pulse.kernel
def measure_while_driving(qs):
    d1 = get_drive_line(qs[1])
    drive(d1, square(40, 0.2))
    readout(get_readout_line(qs[0]), square(400, 0.05))


@pytest.mark.parametrize("policy", ["alap", "alap_rcp"])
def test_alap_ends_at_the_last_readout_not_after_its_hold(policy):
    # rcp holds the readout line for 400 cycles of latency after it ends at
    # 400; alap_rcp used to line the drive up with the hold's end (760-800,
    # an 800-cycle program). Now the drive ends with the readout.
    lowered = _lowered(measure_while_driving, _qpu(400), policy)
    assert sorted((e.kind, e.start_cycles) for e in lowered.schedule) == [
        ("drive", 360),
        ("readout", 0),
    ]
    assert round(float(lowered.duration) * 2) == 400


@pytest.mark.parametrize("policy", ["asap", "rcp"])
def test_asap_and_rcp_keep_a_readout_s_hold_for_the_next_operation(policy):
    # The hold still makes the next operation on the readout line wait.
    lowered = _lowered(two_readouts, _qpu(400), policy)
    assert _starts(lowered, "readout") == ([0, 800] if policy == "rcp" else [0, 400])


@pulse.kernel
def sync_in_an_arm(qs):
    m = readout(get_readout_line(qs[0]), square(400, 0.05))
    if m:
        sync(get_drive_line(qs[1]), get_readout_line(qs[1]))
    drive(get_drive_line(qs[0]), square(40, 0.3))


@pytest.mark.parametrize("policy", ["alap", "alap_rcp"])
def test_alap_does_not_end_at_a_sync_in_an_arm(policy):
    # The sync on controllers[1] waits for the outcome to reach it, 1500;
    # it plays nothing, so the drive ends with the readout at 400 rather
    # than at 1500.
    lowered = _lowered(sync_in_an_arm, _two(), policy)
    assert _starts(lowered, "drive") == [360]
    assert round(float(lowered.duration) * 2) == 400


@pulse.classical(max_latency_ns=400)
def check_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.kernel
def answer_then_sync(qs):
    m = readout(get_readout_line(qs[0]), square(400, 0.05))
    if check_bit.submit([m]).result()[0]:
        sync(get_drive_line(qs[1]), get_readout_line(qs[1]))
    drive(get_drive_line(qs[0]), square(40, 0.3))


# The program's length on main (before the last end) and now, by system and
# policy: the answer taken at its forward time ends the program, and the
# drive ends with it rather than at the sync's instant in the arm.
ANSWER_THEN_SYNC = {
    "one": {
        "asap": (1601, 1601),
        "alap": (1608, 1601),
        "rcp": (1601, 1601),
        "alap_rcp": (1608, 1601),
    },
    "two": {
        "asap": (1301, 1301),
        "alap": (1500, 1301),
        "rcp": (1301, 1301),
        "alap_rcp": (1500, 1301),
    },
}


@pytest.mark.parametrize("policy", ["asap", "alap", "rcp", "alap_rcp"])
@pytest.mark.parametrize("system", ["one", "two"])
def test_an_answer_ends_by_the_last_end_as_a_pulse_does(system, policy):
    # The answer an arm's sync waits for is taken by the last end: the sync
    # at the arm's latest time once pulled the answer past the last end.
    qpu = _qpu(400) if system == "one" else _two()
    lowered = _lowered(answer_then_sync, qpu, policy)
    before, now = ANSWER_THEN_SYNC[system][policy]
    cycles = round(float(lowered.duration) * 2)
    assert cycles == now <= before
    assert pulse.controller.check(lowered) == []
    [answer] = [e.start_cycles for e in lowered.schedule if e.kind == "answer"]
    assert answer == now


def _answer_then_sync_loop(threshold):
    @pulse.kernel(unroll_threshold=threshold)
    def answer_then_sync_loop(qs):
        r0, d0 = get_readout_line(qs[0]), get_drive_line(qs[0])
        d1, r1 = get_drive_line(qs[1]), get_readout_line(qs[1])
        for _ in range(30):
            m = readout(r0, square(400, 0.05))
            if check_bit.submit([m]).result()[0]:
                sync(d1, r1)
            drive(d0, square(40, 0.3))

    return answer_then_sync_loop


@pytest.mark.parametrize("threshold", [0, 10**6])
@pytest.mark.parametrize(
    ("system", "policy", "before", "now"),
    [
        ("one", "alap", 13208, 13201),
        ("one", "alap_rcp", 24808, 24801),
        ("two", "alap", 13100, 12901),
        ("two", "alap_rcp", 16000, 15801),
    ],
)
def test_an_answer_in_a_loop_ends_by_the_last_end(system, policy, before, now, threshold):
    # Rolled and unrolled, 30 trips: the last answer ends the program.
    qpu = _qpu(400) if system == "one" else _two()
    lowered = pulse.compile(
        _answer_then_sync_loop(threshold), quantum_system=qpu, schedule=policy
    ).lower(qpu.qudits[0:2])
    assert round(float(lowered.duration) * 2) == now <= before
    assert pulse.controller.check(lowered) == []


@pulse.classical(max_latency_ns=200)
def delay_of(bits: pulse.Bits) -> float:
    return 48.0 + 24.0 * bits[0]


def _nested_after_a_variable_wait(outer, inner):
    @pulse.kernel
    def nested(qs):
        d0 = get_drive_line(qs[0])
        r0 = get_readout_line(qs[0])
        d1 = get_drive_line(qs[1])
        r1 = get_readout_line(qs[1])
        m = readout(r0, square(96, 0.05))
        wait(d0, delay_of.submit([m]).result(), bounds=(48, 384))
        drive(d0, square(24, 0.1))
        sync(d1, r1)
        readout(r1, square(32, 0.05))
        for _ in range(outer):
            drive(d1, square(16, 0.3))
            for _ in range(inner):
                drive(d1, square(40, 0.2))

    return nested


def _events(lowered):
    return sorted((e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule)


@pytest.mark.parametrize("policy", ["alap", "alap_rcp"])
@pytest.mark.parametrize("system", ["one", "two"])
@pytest.mark.parametrize(("outer", "inner"), [(20, 8), (40, 30), (12, 15)])
def test_alap_ends_nested_rolled_loops_where_unrolled_ones_end(outer, inner, system, policy):
    # The last end of an outer loop counts the iterations its inner loop
    # skips: rolled, the loops end and play as unrolled, every pulse after
    # the variable wait and the readout tied to them by the sync.
    qpu = _qpu(0) if system == "one" else _two()
    kernel = _nested_after_a_variable_wait(outer, inner)
    rolled, unrolled = (
        pulse.compile(
            kernel, quantum_system=qpu, schedule=policy, unroll_threshold=threshold, passes=()
        ).lower(qpu.qudits[0:2])
        for threshold in (0, 10**7)
    )
    assert _events(rolled) == _events(unrolled)
    assert rolled.duration == unrolled.duration


@pulse.kernel(unroll_threshold=0)
def readout_synced_in_a_loop(qs):
    d0 = get_drive_line(qs[0])
    d1 = get_drive_line(qs[1])
    r0 = get_readout_line(qs[0])
    for _ in range(50):
        readout(r0, square(400, 0.05))
        sync(r0, d0)
        drive(d1, square(40, 0.2))


def test_alap_rcp_ends_a_loop_that_syncs_a_readout_line_at_its_last_pulse():
    # The line synced to the readout carries its hold; the loop still ends
    # at the last readout's end, as unrolled.
    qpu = _qpu(400)
    rolled, unrolled = (
        pulse.compile(
            readout_synced_in_a_loop,
            quantum_system=qpu,
            schedule="alap_rcp",
            unroll_threshold=threshold,
            passes=(),
        ).lower(qpu.qudits[0:2])
        for threshold in (0, 10**7)
    )
    assert _events(rolled) == _events(unrolled)
    assert round(float(rolled.duration) * 2) == 39600


def _alap_rcp_beside_a_readout(shrink):
    """The drive and readout starts of measure_while_driving under alap_rcp with
    the test hook CUDAQ_PULSE_TEST_SHRINK_LAST_END set to `shrink`."""
    code = (
        "import cudaq_pulse as pulse\n"
        "from tests.test_feedback_latency_schedules import _qpu, measure_while_driving\n"
        "qpu = _qpu(400)\n"
        "low = pulse.compile(measure_while_driving, quantum_system=qpu, schedule='alap_rcp')"
        ".lower(qpu.qudits[0:2])\n"
        "starts = sorted((e.kind, e.start_cycles) for e in low.schedule)\n"
        "print(starts, round(float(low.duration) * 2))\n"
    )
    env = dict(os.environ, CUDAQ_PULSE_TEST_SHRINK_LAST_END=shrink)
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.strip()


def test_a_schedule_that_breaks_the_last_end_is_scheduled_without_it():
    # The test hook moves the last end 100 cycles early, before the
    # readout's forward end: the backward schedule cannot keep to it, so the
    # round is scheduled again without it, as before the last end (the drive
    # lined up with the hold's end, 760-800), never refused.
    assert _alap_rcp_beside_a_readout("100") == "[('drive', 760), ('readout', 0)] 800"


@pytest.mark.parametrize("shrink", ["0", "-100", "abc", "1e2", " 5", ""])
def test_the_last_end_test_hook_takes_only_a_non_negative_integer(shrink):
    # Anything else is ignored: the drive ends with the readout.
    assert _alap_rcp_beside_a_readout(shrink) == "[('drive', 360), ('readout', 0)] 400"


def _blocking_controller():
    """One controller that blocks on calls, its feedback latency 200 cycles,
    three drive and readout lanes."""
    qudits = {
        i: PhysicalQudit(i, 4.5e9 + 1.0e5 * i, 37.7, 9.2, anharmonicity_hz=-215e6) for i in range(3)
    }
    controller = Controller(
        clock_ghz=2.0,
        drive_lines=[0, 1, 2],
        readout_lines=[ReadoutLine([0, 1], max_concurrent_tones=2), 2],
        max_amplitude=1.0,
        feedback_latency_cycles=200,
        blocks_on_calls=True,
    )
    system = ControlSystem(
        controllers=[controller],
        reference_clock_ghz=2.0,
        max_concurrent_drives=3,
        max_concurrent_readouts=3,
    )
    return pulse.quantum_system(
        name="blocking", qudits=qudits, control_system=system, clock_ghz=2.0
    )


def _call_then_drives(threshold):
    @pulse.kernel(unroll_threshold=threshold)
    def call_then_drives(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        d2 = get_drive_line(qs[2])
        r0 = get_readout_line(qs[0])
        r1 = get_readout_line(qs[1])
        m = readout(r1, square(36, 0.05))
        wait(d2, delay_of.submit([m]).result(), bounds=(48, 384))
        drive(d1, square(96, 0.2))
        m = readout(r0, square(96, 0.05))
        for _ in range(40):
            drive(d0, square(12, 0.2))
        if m:
            readout(r1, square(96, 0.05))

    return call_then_drives


@pytest.mark.parametrize("threshold", [0, 10**6])
def test_alap_rcp_starts_no_operation_before_its_forward_start(threshold):
    # The lanes the drives capped at the last end take would push the d1
    # drive before its forward start, 685, into the call it waits out; the
    # program is scheduled without the last end instead, as before it.
    qpu = _blocking_controller()
    lowered = pulse.compile(
        _call_then_drives(threshold), quantum_system=qpu, schedule="alap_rcp", passes=()
    ).lower(qpu.qudits[0:3])
    d1 = [e.start_cycles for e in lowered.schedule if e.kind == "drive" and e.line == 1]
    assert d1 == [725]
    assert round(float(lowered.duration) * 2) == 1277


def _answer_in_two_loops(threshold):
    @pulse.kernel(unroll_threshold=threshold)
    def answer_in_two_loops(qs):
        d = [get_drive_line(qs[i]) for i in range(4)]
        r = [get_readout_line(qs[i]) for i in range(4)]
        for _ in range(2):
            m = readout(r[2], square(32, 0.05))
            if decide_eagerly.submit([m]).result()[0]:
                sync(r[0], r[3])
                sync(r[3], r[1])
                readout(r[2], square(96, 0.05))
            else:
                readout(r[0], square(96, 0.05))
                drive(d[2], square(16, 0.1))
        for _ in range(5):
            drive(d[0], square(40, 0.1))

    return answer_in_two_loops


@pulse.classical(max_latency_ns=1000)
def decide_eagerly(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


def _two_with_lanes():
    """qudits 0 and 1 on controllers[0] (400 cycles of feedback, 300 to its
    peer), 2 and 3 on controllers[1]; three drive and two readout lanes."""
    qudits = {
        i: PhysicalQudit(i, 4.5e9 + 1.0e5 * i, 37.7, 9.2, anharmonicity_hz=-215e6) for i in range(4)
    }
    a = Controller(
        clock_ghz=2.0,
        drive_lines=[0, 1],
        readout_lines=[0, 1],
        max_amplitude=1.0,
        feedback_latency_cycles=400,
        peer_latency_cycles=300,
    )
    b = Controller(
        clock_ghz=2.0,
        drive_lines=[2, 3],
        readout_lines=[2, 3],
        max_amplitude=1.0,
        feedback_latency_cycles=100,
    )
    system = ControlSystem(
        controllers=[a, b],
        reference_clock_ghz=2.0,
        max_concurrent_drives=3,
        max_concurrent_readouts=2,
    )
    return pulse.quantum_system(
        name="two_lanes", qudits=qudits, control_system=system, clock_ghz=2.0
    )


@pytest.mark.parametrize(("threshold", "cycles"), [(0, 4958), (10**6, 4558)])
def test_a_round_that_cannot_be_written_under_the_last_end_is_scheduled_without_it(
    threshold, cycles
):
    # Rolled, the lanes handed out under the last end take the eager call's
    # answer before it is due: the round is scheduled without the last end,
    # as on main (4958 cycles), not by the forward schedule. Unrolled, the
    # last end holds and the program ends at its last operation.
    qpu = _two_with_lanes()
    lowered = pulse.compile(
        _answer_in_two_loops(threshold), quantum_system=qpu, schedule="alap_rcp"
    ).lower(qpu.qudits[0:4])
    assert round(float(lowered.duration) * 2) == cycles
    assert pulse.controller.check(lowered) == []
