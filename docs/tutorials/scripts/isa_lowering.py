# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering to Your Controller's Instruction Set: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/isa_lowering.html

Generated from docs/tutorials/isa_lowering.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

import sys
from pathlib import Path


# The page imports examples/isa_lowering/toy_isa.py. Run this script from a
# clone of the repository, or save that file beside it.
_EXAMPLES = Path(__file__).resolve().parent.parent.parent.parent / "examples"
sys.path.insert(0, str(_EXAMPLES / "isa_lowering"))


# %% The controller description
import math
from pathlib import Path

import cudaq_pulse as pulse
from toy_isa import DESCRIPTION

description = Path("toy_sequencer.toml")
description.write_text(DESCRIPTION)


# %% Two controllers, wired
from cudaq_pulse.controller import ReadoutLine

# One feedline reads qudits 0 and 1 at once.
feedline = ReadoutLine([0, 1], max_concurrent_tones=2)

first = pulse.controller(
    "toy_sequencer.toml",
    drive_lines=[0, 1],
    readout_lines=[feedline],
    peer_latency_cycles=40,
)
second = pulse.controller(
    "toy_sequencer.toml",
    drive_lines=[2],
    readout_lines=[2],
    peer_latency_cycles=40,
)
print("model:     ", first.model)
print("max pulses:", first.max_pulses)

# Prints:
#   model:      toy.sequencer
#   max pulses: 256


# %% One QPU
from cudaq_pulse.quantum_systems import PhysicalQudit

qudits = {}
for number in range(3):
    qudits[number] = PhysicalQudit(
        index=number,
        frequency_hz=5.0e9 + 3.0e8 * number,
        t1_us=50.0,
        t2_star_us=30.0,
    )


decoder_link = pulse.Link(fixed_ns=4, feedback_ns=8)
control_system = pulse.ControlSystem(
    controllers=[first, second],
    reference_clock_ghz=0.25,
    links={"decoder": decoder_link},
)
qpu = pulse.quantum_system(
    name="toy-qpu",
    qudits=qudits,
    clock_ghz=1.0,
    control_system=control_system,
)

saved = qpu.as_dict()
controllers = saved["control_system"]["controllers"]
print("controllers:", len(controllers))
print("second line drives:", controllers[1]["drive_lines"])

# Prints:
#   controllers: 2
#   second line drives: [2]


# %% A Ramsey sequence with a reset
@pulse.kernel
def ramsey_then_reset(qudit):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    x90 = gaussian(duration=40, amplitude=0.25, sigma=10.0)
    x180 = gaussian(duration=40, amplitude=0.5, sigma=10.0)
    probe = square(duration=200, amplitude=0.05)

    # Ramsey, detuned by 2 MHz while it waits.
    drive(drive_line, x90)
    shift_frequency(drive_line.tone, 2.0e6)
    wait(drive_line, 100)
    shift_phase(drive_line.tone, math.pi / 2)
    drive(drive_line, x90)
    shift_frequency(drive_line.tone, -2.0e6)

    # Read out, and play a π pulse when it reads 1.
    sync(drive_line, readout_line)
    outcome = readout(readout_line, probe)
    sync(drive_line, readout_line)
    if outcome:
        drive(drive_line, x180)


# %% The scheduled IR
import re

compiled = pulse.compile(
    ramsey_then_reset,
    quantum_system=qpu,
    schedule="rcp",
)
scheduled = compiled.lower(qpu.qudits[0])

op_timing = re.compile(
    r"= (pulse\.\w+) .*"
    r"duration_cycles = (\d+) : i64, start_cycles = (\d+)"
)
for op, duration, start in op_timing.findall(scheduled.mlir()):
    print(f"{op:<23} start {start:>3}  length {duration:>3}")

# Prints:
#   pulse.drive             start   0  length  40
#   pulse.shift_frequency   start  40  length   0
#   pulse.wait              start  40  length 100
#   pulse.drive             start 140  length  40
#   pulse.shift_frequency   start 180  length   0
#   pulse.readout           start 180  length 200
#   pulse.drive             start 460  length  40


# %% The export
program = scheduled.export_to_controller()

for event in program["events"]:
    if event["kind"] not in ("drive", "readout"):
        continue
    start = event["start_cycles"]
    gigahertz = event["frequency_hz"] / 1e9
    phase = event["phase_rad"]
    print(f"{event['kind']:<7} at {start:>3}: "
          f"{gigahertz:.3f} GHz, {phase:.6f} rad")

# Prints:
#   drive   at   0: 5.000 GHz, 0.000000 rad
#   drive   at 140: 5.002 GHz, 2.827433 rad
#   readout at 180: 5.000 GHz, 0.000000 rad
#   drive   at 460: 5.000 GHz, 3.330088 rad


reset_pulse = program["events"][6]
condition = reset_pulse["conditions"][0]
print("test:", condition["test"])
print("on readouts:", condition["readouts"])

# Prints:
#   test: is_set
#   on readouts: [4]


# %% The head and the Ramsey part
from toy_isa import lower

toy = lower(program)
listing = str(toy).splitlines()

for line in listing[:10]:
    print(line)

# Prints:
#   .chan ch0 drive q0
#   .chan ch1.0 readout q0
#   .wave w0 40
#   .wave w1 200
#   set_freq ch0 5000000000
#   play ch0 w0 0.25
#   wait 35
#   set_freq ch0 5002000000
#   shift_phase ch0 2.827433
#   play ch0 w0 0.25


# %% The readout and the reset
for line in listing[10:]:
    print(line)

# Prints:
#   wait 10
#   set_freq ch1.0 5000000000
#   acquire ch1.0 w1 0.05 m4
#   wait 70
#   branch !m4 end0
#   set_freq ch0 5000000000
#   shift_phase ch0 0.502655
#   play ch0 w0 0.5
#   shift_phase ch0 -0.502655
#   end0:


# %% Validate it
from cudaq_pulse import backends
from toy_isa import ToyBackend

report = backends.validate(ToyBackend())
lines = str(report).splitlines()
print(lines[-1])

# Prints:
#   backend 'toy': 29 passed, 2 skipped


for result in lines[:-1]:
    if not result.startswith("PASS"):
        name = result.split(":")[0]
        print(name)

# Prints:
#   SKIP runtime
#   SKIP runtime_frequency


# %% Check your own programs
from toy_isa import events

played = []
for event in events(toy):
    played.append((event["kind"], event["line"], event["start_cycles"]))

expected = []
for event in scheduled.schedule:
    if event.kind != "wait":
        expected.append((event.kind, event.line, event.start_cycles))

print("plays as scheduled:", sorted(played) == sorted(expected))

# Prints:
#   plays as scheduled: True
