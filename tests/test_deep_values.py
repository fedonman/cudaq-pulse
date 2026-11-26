# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A value computed through very many operations compiles: the walks that
follow what a value is computed from, or what it reaches, keep what is
still to follow on a stack of their own. Each of these ended the process
(a segmentation fault) at about a hundred thousand operations."""

from __future__ import annotations

import cudaq_pulse as pulse
from cudaq_pulse import ControlSystem
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.quantum_systems import PhysicalQudit

_DEEP = 100_000


def _selected(depth):
    """A measurement selects an amplitude, which goes through *depth*
    products to a waveform."""
    chain = "".join(f"  %a{i + 1} = arith.mulf %a{i}, %k : f64\n" for i in range(depth))
    return (
        "func.func @amplitude() {\n"
        "  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit\n"
        "  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)\n"
        "  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> "
        "(!pulse.readout_line, !pulse.tone)\n"
        "  %c40 = arith.constant 40 : i64\n"
        "  %c100 = arith.constant 100 : i64\n"
        "  %zero = arith.constant 0.0 : f64\n"
        "  %low = arith.constant 0.1 : f64\n"
        "  %high = arith.constant 0.3 : f64\n"
        "  %k = arith.constant 0.99999 : f64\n"
        "  %probe = pulse.square %c100, %low, %zero : i64, f64, f64 -> !pulse.waveform\n"
        "  %r1, %rt1, %m = pulse.readout %r, %probe, %rt, iq : !pulse.readout_line, "
        "!pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement\n"
        "  %bit = pulse.is_set %m : !pulse.measurement -> i1\n"
        "  %a0 = arith.select %bit, %high, %low : f64\n"
        f"{chain}"
        f"  %wf = pulse.square %c40, %a{depth}, %zero : i64, f64, f64 -> !pulse.waveform\n"
        "  %d1, %t1 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone "
        "-> !pulse.drive_line, !pulse.tone\n"
        "  return\n"
        "}\n"
    )


def test_a_selected_value_reaches_a_waveform_through_any_number_of_operations():
    # The pulse the value shapes plays in both arms of a branch, each arm
    # with its own copy of the operations.
    lowered = {}
    for depth in (3, _DEEP):
        module = MLIRPipeline().parse_to_module(_selected(depth))
        module.lower_selects([])
        lowered[depth] = module.print()
    for depth, text in lowered.items():
        assert "arith.select" not in text
        assert text.count("scf.if") == 1
        assert text.count("pulse.drive ") == 2
        assert text.count("arith.mulf") == 2 * depth


def _branched(depth):
    """A branch on a bit computed from two measurements through *depth*
    exclusive ors."""
    line = "!pulse.drive_line, !pulse.tone"
    read = "!pulse.readout_line, !pulse.waveform, !pulse.tone"
    chain = "".join(f"  %b{i + 1} = arith.xori %b{i}, %other : i1\n" for i in range(depth))
    return (
        "func.func @condition() {\n"
        "  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit\n"
        f"  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({line})\n"
        "  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> "
        "(!pulse.readout_line, !pulse.tone)\n"
        "  %c40 = arith.constant 40 : i64\n"
        "  %c100 = arith.constant 100 : i64\n"
        "  %zero = arith.constant 0.0 : f64\n"
        "  %low = arith.constant 0.1 : f64\n"
        "  %probe = pulse.square %c100, %low, %zero : i64, f64, f64 -> !pulse.waveform\n"
        "  %wf = pulse.square %c40, %low, %zero : i64, f64, f64 -> !pulse.waveform\n"
        f"  %r1, %rt1, %m0 = pulse.readout %r, %probe, %rt, iq : {read} -> "
        "!pulse.readout_line, !pulse.tone, !pulse.measurement\n"
        f"  %r2, %rt2, %m1 = pulse.readout %r1, %probe, %rt1, iq : {read} -> "
        "!pulse.readout_line, !pulse.tone, !pulse.measurement\n"
        # The later readout is at the far end of the chain: a walk that
        # stopped short of it would place the branch too early.
        "  %b0 = pulse.is_set %m1 : !pulse.measurement -> i1\n"
        "  %other = pulse.is_set %m0 : !pulse.measurement -> i1\n"
        f"{chain}"
        f"  %d2, %t2 = scf.if %b{depth} -> ({line}) {{\n"
        "    %d1, %t1 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, "
        f"!pulse.tone -> {line}\n"
        f"    scf.yield %d1, %t1 : {line}\n"
        "  } else {\n"
        f"    scf.yield %d, %t : {line}\n"
        "  }\n"
        "  return\n"
        "}\n"
    )


def test_a_branch_condition_is_computed_through_any_number_of_operations():
    # The scheduler and the verifier each find the measurements a branch
    # waits for through the operations its condition is computed by.
    for policy in ("asap", "alap"):
        starts = {}
        for depth in (3, _DEEP):
            module = MLIRPipeline().parse_to_module(_branched(depth))
            module.schedule(policy)
            module.run_passes(["pulse-verify"])
            text = module.print()
            starts[depth] = [
                line.split("start_cycles = ")[1].split(" ")[0]
                for line in text.splitlines()
                if "pulse.drive " in line or "pulse.readout " in line
            ]
        assert len(starts[3]) == 3
        assert starts[_DEEP] == starts[3]


@pulse.classical(max_latency_ns=60)
def _first(bits: pulse.Bits) -> float:
    return 0.5


@pulse.classical(max_latency_ns=60)
def _second(x: float) -> float:
    return x


def test_a_calls_argument_is_computed_through_any_number_of_operations():
    # On several controllers the scheduler finds which of them sends a
    # call's arguments, and the controller check what the argument computes.
    source = (
        "def k(qs):\n"
        "    d = [get_drive_line(qs[i]) for i in range(4)]\n"
        "    r = [get_readout_line(qs[i]) for i in range(4)]\n"
        "    v = _first.submit([readout(r[0], square(32, 0.05))]).result()\n"
        + "    v = v * 0.99999\n"
        * _DEEP
        + "    w = _second.submit(v).result()\n"
        "    shift_phase(d[3].tone, w)\n"
        "    drive(d[3], square(40, 0.1))\n"
    )
    scope = {"_first": _first, "_second": _second}
    exec(source, scope)  # noqa: S102
    controllers = [
        pulse.Controller(clock_ghz=2.0, drive_lines=lines, readout_lines=lines)
        for lines in ([0, 1], [2, 3])
    ]
    system = pulse.quantum_system(
        name="deep_values",
        qudits={i: PhysicalQudit(i, 4.5e9 + 1.0e5 * i, 37.7, 9.2) for i in range(4)},
        control_system=ControlSystem(controllers=controllers, reference_clock_ghz=2.0),
        clock_ghz=2.0,
    )
    lowered = pulse.compile(pulse.kernel(scope["k"]), quantum_system=system).lower(
        system.qudits[0:4]
    )
    assert list(pulse.controller.check(lowered)) == []
    kinds = [event.kind for event in lowered.schedule]
    assert kinds.count("drive") == 1
