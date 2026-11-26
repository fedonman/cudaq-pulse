# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""From Kernel to Controller: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/kernel_to_controller.html

Generated from docs/tutorials/kernel_to_controller.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% The exported program
import cudaq_pulse as pulse
from cudaq_pulse.quantum_systems import PhysicalQudit

qudit = PhysicalQudit(
    index=0,
    frequency_hz=5.0e9,
    t1_us=50.0,
    t2_star_us=30.0,
    anharmonicity_hz=-2.0e8,
)
system = pulse.quantum_system(
    name="one-qubit",
    qudits={0: qudit},
    clock_ghz=1.0,
)


@pulse.kernel
def drive_and_measure(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    pulse_shape = gaussian(
        duration=40 * pulse.ns,
        amplitude=0.5,
        sigma=10 * pulse.ns,
    )
    drive(drive_line, pulse_shape)
    sync(drive_line, readout_line)
    readout(readout_line, square(200 * pulse.ns, 0.05))


scheduled = drive_and_measure.lower(system.qudits[0])
program = scheduled.export_to_controller()
print("format:", program["format"])

print(f"{'kind':8} {'line':>4} {'start':>6} {'length':>6}")
for event in program["events"]:
    line = str(event["line"])
    start = event["start_cycles"]
    length = event["duration_cycles"]
    print(f"{event['kind']:8} {line:>4} {start:>6} {length:>6}")

# Prints:
#   format: cudaq-pulse.controller/1
#   kind     line  start length
#   drive       0      0     40
#   sync     None     40      0
#   readout     1     40    200


# %% Check a limit in Python
def concurrent_drives(scheduled, limit):
    edges = []
    for event in scheduled.schedule:
        if event.kind == "drive" and event.duration_cycles:
            edges.append((event.start, 1))
            edges.append((event.end, -1))

    active = 0
    violations = []
    # An end sorts before a start at the same time.
    for time, step in sorted(edges):
        active += step
        if step > 0 and active > limit:
            violations.append((time, active))
    return violations


@pulse.kernel
def flip_four(qudits):
    for qudit in qudits:
        drive_line = get_drive_line(qudit)
        flip = drive_line.get_calibration(pulse.gates.X)
        drive(drive_line, flip)


surface = pulse.quantum_system("transmon_surface17")
four = surface.qudits[0:4]
violations = concurrent_drives(flip_four.lower(four), limit=2)
print("(cycle, drives at once):", violations)

# Prints:
#   (cycle, drives at once): [(0, 3), (0, 4)]


# %% Let the compiler keep the limit
limited = pulse.ControlSystem(max_concurrent_drives=2)
compiled = pulse.compile(
    flip_four,
    schedule="rcp",
    control_system=limited,
)
program = compiled.lower(four)

starts = sorted(event.start for event in program.schedule)
print("violations:", concurrent_drives(program, limit=2))
print("pulse starts:", starts)

# Prints:
#   violations: []
#   pulse starts: [0, 0, 80, 80]


# %% A small backend
from collections.abc import Mapping
from typing import Any

from cudaq_pulse import Controller, backends
from cudaq_pulse.backends import IncompatibleBackendError
from cudaq_pulse.controller import FORMAT

Shots = list[list[str]]


def describe(event: Mapping[str, Any]) -> str:
    """One line of text for a drive or readout."""
    start = event["start_cycles"]
    kind = event["kind"]
    line = event["line"]
    shape = event["waveform"]["shape"]
    frequency = f"{event['frequency_hz']:.6g} Hz"
    return f"{start} {kind} line {line} {shape} at {frequency}"


class TextBackend:
    name = "text"
    formats = (FORMAT,)

    def control_system(self) -> Controller:
        return Controller(
            clock_ghz=1.0,
            instruction_cycles=2,
            min_idle_cycles=2,
        )

    def emit(self, program: Mapping[str, Any]) -> list[str]:
        if program["format"] != FORMAT:
            message = f"cannot read {program['format']}"
            raise IncompatibleBackendError(message)

        played = []
        for event in program["events"]:
            if event["kind"] in ("wait", "sync"):
                continue
            playable = event["kind"] in ("drive", "readout")
            if not playable or event["conditions"]:
                message = f"cannot play event {event['index']}"
                raise IncompatibleBackendError(message)
            played.append(describe(event))
        return played

    def run(self, artifact: Any, *, shots_count: int = 1) -> Shots:
        shots = []
        for _ in range(shots_count):
            shots.append(list(artifact))
        return shots


# %% Register it and run a program
backends.register("text", TextBackend)
backend = backends.get("text")

compiled = pulse.compile(
    drive_and_measure,
    quantum_system=system,
    control_system=backend.control_system(),
)
program = compiled.lower(system.qudits[0]).export_to_controller()
artifact = backend.emit(program)

for line in backend.run(artifact)[0]:
    print(line)

# Prints:
#   0 drive line 0 gaussian at 5e+09 Hz
#   40 readout line 1 square at 5e+09 Hz


# %% Validate it
report = backends.validate(
    "text",
    controller=Controller(clock_ghz=1.0),
)
print("ok:", report.ok)
print(str(report).splitlines()[-1])
backends.unregister("text")

# Prints:
#   ok: True
#   backend 'text': 12 passed, 8 refused
