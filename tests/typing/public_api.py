# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type-checked use of the public API (mypy --strict in CI; not run by pytest).

assert_type pins what users see; a wrong call is marked with the error code
mypy must report, and --warn-unused-ignores fails if it stops reporting it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, assert_type

import cudaq_pulse as pulse
import cudaq_pulse.controller as controller_module
import numpy as np
import numpy.typing as npt
from cudaq_pulse import Controller
from cudaq_pulse.classical import Bits, ClassicalFunction, Pending
from cudaq_pulse.compiler import CompileMetrics, ScheduledEvent
from cudaq_pulse.controller import (
    ControllerModule,
    ControllerNames,
    DriveLine,
    ExpressionTerm,
    WaveformMemoryScope,
)
from cudaq_pulse.ops import Waveform
from cudaq_pulse.quantum_systems import (
    Coupling,
    DispersiveReadout,
    DragPulse,
    Drive,
    FluxCZ,
    MicrowaveDrive,
    PhysicalQudit,
    Readout,
    ReadoutPulse,
    ResidualZZ,
    VirtualZ,
)
from cudaq_pulse.quantum_systems.transmon import CZ_LAYERS

wave = pulse.ops.gaussian(40, 0.1, 10.0)
assert_type(wave, Waveform)
assert_type(pulse.ops.wf_scale(wave, 0.5), Waveform)
pulse.ops.wf_scale(0.5, wave)  # type: ignore[arg-type]
pulse.ops.wf_scale(waveform=wave, scalar=0.5)  # type: ignore[call-arg]
assert_type(pulse.ops.wf_add(wave, wave), Waveform)

dev = pulse.quantum_system("transmon_surface17")
assert_type(dev, pulse.QuantumSystem)
assert_type(pulse.quantum_system("transmon_surface17"), pulse.QuantumSystem)
assert_type(
    pulse.quantum_system("transmon_surface17", layout=[4, 5], clock_ghz=1.0), pulse.QuantumSystem
)
assert_type(pulse.QuantumSystem.from_dict(dev.as_dict()), pulse.QuantumSystem)
assert_type(pulse.QuantumSystem.load("qpu.json"), pulse.QuantumSystem)
pulse.quantum_system({"name": "transmon_surface17"})  # type: ignore[arg-type]
pulse.quantum_system("transmon_surface17", [4, 5])  # type: ignore[call-arg, arg-type]
assert_type(dev.qudits[0], pulse.Qudit)
assert_type(dev.qudits[0:2], pulse.Register)
assert_type(dev.qudits[[0, 1]], pulse.Register)
assert_type(dev.qudits[0:2][0], pulse.Qudit)


@pulse.kernel
def rabi(q: pulse.Qudit, amp: float) -> None:
    pass


@pulse.kernel(cache_size=4)
def idle(q: pulse.Qudit) -> None:
    pass


# A kernel keeps the signature of its function: calls are checked against it.
kernels: list[pulse.Kernel[...]] = [rabi, idle]
assert_type(rabi.cache_info().size, int)
rabi(dev.qudits[0])  # type: ignore[call-arg]
rabi(dev.qudits[0], 0.5, 1)  # type: ignore[call-arg]
rabi(dev.qudits[0], amp="high")  # type: ignore[arg-type]
rabi.lower(dev.qudits[0], ampl=0.5)  # type: ignore[call-arg]


@pulse.kernel
def spread(tau: float, *qudits: pulse.Qudit, amp: float = 0.1) -> None:
    pass


spread(1.0)
spread(1.0, *dev.qudits[0:3], amp=0.2)
assert_type(spread.lower(1.0, dev.qudits[0], dev.qudits[1]), pulse.ScheduledKernel)
spread(1.0, dev.qudits[0:2])  # type: ignore[arg-type]

lowered = rabi.lower(dev.qudits[0], 0.5)
assert_type(lowered, pulse.ScheduledKernel)
assert_type(lowered.mlir(), str)
assert_type(lowered.mlir("llvm"), str)
assert_type(lowered.metrics, CompileMetrics)
assert_type(lowered.schedule, Sequence[ScheduledEvent])
assert_type(lowered.arguments, tuple[float, ...])
pulse.viz.save_schedule(lowered, Path("schedule.png"))
pulse.viz.save_schedule(lowered, "schedule.png", dpi=300)  # type: ignore[call-arg]

compiled = pulse.compile(rabi, schedule="asap", quantum_system=dev)
compiled_kernels: list[pulse.CompiledKernel[...]] = [compiled]
assert_type(compiled.lower(dev.qudits[0], 0.5), pulse.ScheduledKernel)
compiled(dev.qudits[0])  # type: ignore[call-arg]
assert_type(compiled.parameters, tuple[str, ...])
pulse.compile(rabi, schedule=1)  # type: ignore[arg-type]

# Loaded lazily at runtime, but fully typed for checkers.
assert_type(pulse.simulate(lowered, shots_count=10), pulse.SimulationResult)
# Every module has a generic one, which binds a call's new values.
assert_type(lowered.generic, pulse.ScheduledKernel)
assert_type(pulse.simulate(lowered.generic, 0.7), pulse.SimulationResult)
assert_type(lowered.generic.bind([0.7]), pulse.ScheduledKernel)
# Whether a call's values or new line delays scheduled the program again, read-only.
assert_type(lowered.rescheduled, bool)
lowered.rescheduled = False  # type: ignore[misc]
assert_type(lowered.rescheduled_for_line_delays, bool)
lowered.rescheduled_for_line_delays = False  # type: ignore[misc]
# Times take units or numbers of ns.
assert_type(pulse.simulate(lowered, t_end=40 * pulse.ns, dt_ns=0.5), pulse.SimulationResult)
assert_type(pulse.simulate(rabi, dev.qudits[0], 0.5), pulse.SimulationResult)
assert_type(pulse.simulate(compiled, dev.qudits[0], 0.5), pulse.SimulationResult)
simulated = pulse.simulate(lowered, shots_count=10)
kept = simulated.postselect(lambda branch: branch.levels[0] == 0)
assert_type(kept, pulse.SimulationResult)
assert_type(kept.acceptance, float)
simulated.postselect(lambda branch: branch.answers("decode")[-1])


def two_arguments(branch: object, extra: int) -> bool:
    return True


simulated.postselect(two_arguments)  # type: ignore[arg-type]


# pulse.classical is a module at run time; checkers see a callable decorator,
# and a classical function keeps the signature of its body.
@pulse.classical(max_latency_ns=400, link="gpu")
def decode(syndrome: Bits) -> bool:
    return bool(syndrome[0])


@pulse.classical(max_latency_ns=10)
def _scale(x: list[float], k: float) -> tuple[list[float], float]:
    return [k * v for v in x], k


@pulse.classical.external(max_latency_ns=10)
def table(index: pulse.Length[Bits, 2]) -> float:
    return float()


@pulse.classical.external(max_latency_ns=10)
def decode_surface(
    syndrome: pulse.Length[pulse.Bits, 8],
    counts: pulse.Length[list[int], 2],
) -> tuple[float, pulse.Length[pulse.Bits, 2]]:
    return (float(), pulse.Bits())


@pulse.classical.external(max_latency_ns=10)
def _halves(syndrome: pulse.Bits) -> tuple[float, pulse.Bits]:
    return (float(), syndrome[:2])


assert_type(pulse.Bits(), list[int])
assert_type(decode_surface.submit([1] * 8, [3, 4]).result(), tuple[float, list[int]])
assert_type(_halves.submit([0, 1, 1]).result(), tuple[float, list[int]])


def _double(x: float) -> float:
    return 2 * x


decode_as: ClassicalFunction[[Bits], bool] = decode
table_as: ClassicalFunction[[Bits], float] = table
double_as: ClassicalFunction[[float], float] = pulse.classical(_double, max_latency_ns=1.0)
double_too: ClassicalFunction[[float], float] = pulse.classical.function(_double, max_latency_ns=1)
mismatched: ClassicalFunction[[float], bool] = decode  # type: ignore[assignment]
assert_type(decode.submit([1]), Pending[bool])
assert_type(decode.submit([1]).result(), bool)
assert_type(decode.submit([1]).result(timeout=350 * pulse.ns), tuple[bool, bool])
assert_type(decode.submit([1]).result(timeout=700), tuple[bool, bool])
assert_type(_scale.submit([1.0], 2.0).result(), tuple[list[float], float])
assert_type(_scale.submit([1.0], 2.0).result(timeout=9), tuple[bool, list[float], float])
scale_3: ClassicalFunction[[list[float], float], tuple[list[float], float]] = _scale.specialize(x=3)
assert_type(_scale.specialize(x=3).submit([1.0], 2.0).result(), tuple[list[float], float])
assert_type(table.submit([0, 1]).result(), float)
assert_type(pulse.classical.random_index(4), ClassicalFunction[[], int])
assert_type(pulse.classical.dispatch_key("decode"), int)
functions: list[ClassicalFunction[..., Any]] = [decode, _scale, table]
decode.submit(0.5)  # type: ignore[arg-type]
decode.submit([1], [0])  # type: ignore[call-arg]
_scale.submit([1.0], k="x")  # type: ignore[arg-type]
pulse.classical(max_latency_ns="soon")  # type: ignore[call-overload]
pulse.classical(_double)  # type: ignore[call-overload]
pulse.classical(_double, max_latency_ns=1, lengths=3)  # type: ignore[call-overload]  # remnant-ok
pulse.classical.external(max_latency_ns=1, args=())  # type: ignore[call-arg]  # remnant-ok

# pulse.controller is a module and the loader of controller descriptions.
box = pulse.controller("reference.controller", drive_lines=[0, 1])
assert_type(box, pulse.Controller)
assert_type(pulse.Controller(instruction_cycles=2), Controller)
assert_type(pulse.controller.FORMAT, str)
assert_type(ExpressionTerm("select"), ExpressionTerm)
assert_type(pulse.controller.evaluate_expression({"const": 1.0}), float | int | bool)
assert_type(pulse.controller.unroll({"format": "cudaq-pulse.controller/2"}), dict[str, Any])
assert_type(
    pulse.controller.sample_waveform({}, 40, samples_per_cycle=2.5), npt.NDArray[np.complex128]
)
assert_type(pulse.controller.sample_waveform({}, 40).real, npt.NDArray[np.float64])
assert_type(pulse.controller.export(lowered, version=2), dict[str, Any])
assert_type(pulse.controller.WaveformMemoryScope("program"), WaveformMemoryScope)
pulse.controller.sample_waveform({}, 40, samples_per_cycles=2)  # type: ignore[call-arg]
# The module is what checkers take pulse.controller to be, but for the call.
module_names: ControllerNames = controller_module
declared: ControllerNames = pulse.controller
assert_type(pulse.controller, ControllerModule)
control = pulse.ControlSystem(controllers=[box], links={"feedback": pulse.Link(controllers=[box])})
assert_type(control.controllers, Sequence[pulse.Controller])
delayed = pulse.Controller(
    drive_lines=[pulse.controller.DriveLine(0, delay_ns=2 * pulse.ns), 1],
    readout_lines=[pulse.controller.ReadoutLine(0, delay_ns=1.5)],
)
assert_type(delayed.drive_lines, Sequence[DriveLine | int])
feedline = pulse.controller.ReadoutLine([0, 1, 2, 3], delay_ns=12.0, max_concurrent_tones=4)
assert_type(feedline.qudits, tuple[int, ...])
assert_type(feedline.max_concurrent_tones, int)
assert_type(pulse.controller.DriveLine(range(4)).qudits, tuple[int, ...])
pulse.controller.ReadoutLine([0, 1], 12.0, 2)  # type: ignore[call-arg]
pulse.ControlSystem(line_delays_ns={("drive", 0): 1.5})  # type: ignore[call-arg]
pulse.Link(fixed_ns=100 * pulse.ns, feedback_ns=20.0, per_controller=True)
assert_type(box.supports_capability(pulse.Capability.BRANCH), bool)
cores = pulse.Processor(
    links=("feedback",),
    cycles_per_op=pulse.controller.OpCycles(div=20),
    unsupported=[pulse.controller.OpClass.TRIG],
)
assert_type(cores.cycles_per_op.div, int)
pulse.Processor(cycles_per_op={"div": 20})  # type: ignore[arg-type]
pulse.controller.OpCycles(dvi=20)  # type: ignore[call-arg]
assert_type(pulse.controller.OpCounts(add=2).total, int)
assert_type(cores.max_ops_total, int | None)
assert_type(cores.dispatch_keys, Mapping[str, int])
assert_type(pulse.Controller(processors=[cores]).processors, Sequence[pulse.Processor])
qpu = pulse.quantum_system("transmon_surface17", control_system=control)
assert_type(qpu.control_system, pulse.ControlSystem | None)
assert_type(qpu.as_dict(), dict[str, Any])
pulse.compile(rabi, quantum_system=qpu, control_system=control)
pulse.compile(rabi, quantum_system=qpu, control_system=box)
pulse.controller(3)  # type: ignore[arg-type]

readout = DispersiveReadout(frequency_hz=7.0e9, linewidth_hz=1.0e6)
assert_type(readout.frequency_hz, float | None)
assert_type(readout.efficiency, float)
assert_type(readout.replace(linewidth_hz=2.0e6), DispersiveReadout)
DispersiveReadout(7.0e9)  # type: ignore[call-arg]
DispersiveReadout(photon_number=4.0)  # type: ignore[call-arg]
readout.linewidth_hz = 2.0e6  # type: ignore[misc]
MicrowaveDrive(x_amplitude=0.2)  # type: ignore[call-arg]
assert_type(Drive(amplitude_scale_rad_per_ns=0.2).amplitude_scale_rad_per_ns, float | None)
assert_type(MicrowaveDrive().replace(amplitude_scale_rad_per_ns=0.2), MicrowaveDrive)
Drive(0.2)  # type: ignore[call-arg]

# A qudit's tuned pulses are its calibrations, by gate; get_calibration gives
# each as the type of its gate, on the qudit, a handle or a kernel's line.
x = DragPulse(duration_ns=40.0, sigma_ns=10.0)
assert_type(x.amplitude, float | None)
assert_type(x.replace(amplitude=0.1), DragPulse)
DragPulse(duration_ns=40.0)  # type: ignore[call-arg]
qudit = PhysicalQudit(
    0,
    5.0e9,
    50.0,
    30.0,
    anharmonicity_hz=-2.0e8,
    drive=MicrowaveDrive(amplitude_scale_rad_per_ns=0.2),
    readout=readout,
    calibrations={pulse.gates.X: x, pulse.gates.READOUT: ReadoutPulse(duration_ns=300.0)},
    metadata={"fridge": "B"},
)
assert_type(qudit.readout, Readout | None)
assert_type(qudit.metadata, Mapping[str, Any])
assert_type(qudit.get_calibration(pulse.gates.X), DragPulse)
assert_type(qudit.get_calibration(pulse.gates.Y90, transition=(1, 2)), DragPulse)
assert_type(qudit.get_calibration(pulse.gates.READOUT), ReadoutPulse)
assert_type(qudit.get_calibration(pulse.gates.T), VirtualZ)
assert_type(qudit.get_calibration(pulse.gates.S, transition=(1, 2)), VirtualZ)
t = qpu.qudits[0].get_calibration(pulse.gates.T)
assert_type(t, VirtualZ)
assert_type(t.phase_rad, float)
assert_type(t.replace(phase_rad=-t.phase_rad), VirtualZ)
VirtualZ()  # type: ignore[call-arg]
assert_type(qudit.replace(label="Q0"), PhysicalQudit)
assert_type(qudit.frequency_hz + 1.0e6, float)
PhysicalQudit(0, None, 50.0, 30.0)
assert_type(pulse.quantum_systems.Offset(1.5e8) + 1.0, float)
assert_type(pulse.quantum_systems.Offset(1.5e8) < pulse.quantum_systems.Offset(2.0e8), bool)
PhysicalQudit(1, pulse.quantum_systems.Offset(1.5e8), 50.0, 30.0)
assert_type(qpu.frequency_frame, str)
assert_type(qpu.qudits[0].get_calibration(pulse.gates.X90), DragPulse)
gate: pulse.gates.Gate = pulse.gates.CZ


def any_gate(which: pulse.gates.Gate) -> None:
    assert_type(qudit.get_calibration(which), DragPulse | VirtualZ | ReadoutPulse)


# A coupling gives its qudits and strength (None: not measured yet), and its
# CZ calibration is one object of typed fields; leaving a required field out,
# or giving one it does not have, is a type error.
cz = FluxCZ(duration_ns=98.0, buffer_ns=15.0, interaction_frequency_hz=4.4e9)
assert_type(cz.duration_ns, float)
assert_type(cz.interaction_frequency_hz, float | None)
FluxCZ(duration_ns=98.0, fidelity=0.99)  # type: ignore[call-arg]
FluxCZ()  # type: ignore[call-arg]
edge = Coupling((0, 1), 5.7e6, calibrations={pulse.gates.CZ: cz})
assert_type(edge.qudits, tuple[int, ...])
assert_type(edge.strength_hz, float | None)
assert_type(edge.get_calibration(pulse.gates.CZ), FluxCZ)
assert_type(qpu.coupling(qpu.qudits[13], qpu.qudits[0]), Coupling)
Coupling((0, 1), None)
Coupling((0, 1))  # type: ignore[call-arg]
Coupling((0, 1), 5.7e6, gate=cz)  # type: ignore[call-arg]
zz = ResidualZZ((0, 1), 1.5e4)
assert_type(zz.rate_hz, float)
ResidualZZ((0, 1), 1.5e4, zz_coupling=0.5)  # type: ignore[call-arg]
pair = pulse.quantum_system(
    name="pair",
    qudits={0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-2.5e8)},
    couplings=[edge],
    residual_zz=[zz],
)
pulse.quantum_system(name="pair", crosstalk=[zz])  # type: ignore[call-arg]
assert_type(CZ_LAYERS, tuple[tuple[tuple[int, int], ...], ...])
assert_type(pair.coupling_map, list[tuple[int, ...]])
assert_type(pair.coupling(pair.qudits[0], pair.qudits[1]), Coupling)
pulse.quantum_system(name="pair", architecture="transmon")  # type: ignore[call-arg]
