# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several Qubits: Registers, Sync and Scheduling: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/several_qubits.html

Generated from docs/tutorials/several_qubits.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# %% A register of qubits
import cudaq_pulse as pulse

system = pulse.quantum_system("transmon_surface17")
data = system.qudits[0:3]


@pulse.kernel
def flip_all(qudits):
    for qudit in qudits:
        drive_line = get_drive_line(qudit)
        flip = drive_line.get_calibration(pulse.gates.X)
        drive(drive_line, flip)


scheduled = flip_all.lower(data)

print(f"{'kind':6} {'qudit':>5} {'start':>6}")
for event in scheduled.schedule:
    start = event.start_cycles
    print(f"{event.kind:6} {event.qudit:>5} {start:>6}")

# Prints:
#   kind   qudit  start
#   drive      0      0
#   drive      1      0
#   drive      2      0


result = pulse.simulate(flip_all, data)
for qudit in data:
    label = system.physical_qudits[qudit.index].label
    excited = result.level_populations(qudit)[1]
    print(f"{label} excited: {excited:.3f}")

# Prints:
#   D1 excited: 0.978
#   D2 excited: 0.992
#   D3 excited: 0.984


# %% Line pulses up with sync
@pulse.kernel
def uneven(first, second):
    first_drive = get_drive_line(first)
    second_drive = get_drive_line(second)
    first_readout = get_readout_line(first)
    second_readout = get_readout_line(second)
    first_flip = first_drive.get_calibration(pulse.gates.X)
    second_flip = second_drive.get_calibration(pulse.gates.X)
    first_measure = first_readout.get_calibration(pulse.gates.READOUT)
    second_measure = second_readout.get_calibration(pulse.gates.READOUT)

    for _ in range(3):
        drive(first_drive, first_flip)
    drive(second_drive, second_flip)

    # Both readouts start after the last pulse.
    sync(first_drive, second_drive, first_readout, second_readout)
    readout(first_readout, first_measure)
    readout(second_readout, second_measure)


# %% Two scheduling policies
for policy in ("alap", "asap"):
    compiled = pulse.compile(uneven, schedule=policy)
    program = compiled.lower(data[0], data[1])
    starts = []
    for event in program.schedule:
        if event.qudit == 1 and event.kind == "drive":
            starts.append(event.start_cycles)
    print(f"{policy}: second qubit's pulse at cycle {starts}")

# Prints:
#   alap: second qubit's pulse at cycle [160]
#   asap: second qubit's pulse at cycle [0]


# %% Two coupled qubits
x2 = system.qudit("X2")
coupling = system.coupling(data[0], x2)
assert coupling.strength_hz is not None  # None: not measured yet
idle_zz_hz = system.calibration.idle_zz_hz("X2-D1")

print(f"g / 2π:  {coupling.strength_hz / 1e6:.2f} MHz")
print(f"idle ZZ: {idle_zz_hz / 1e3:.2f} kHz")

# Prints:
#   g / 2π:  5.69 MHz
#   idle ZZ: -5.94 kHz


# %% A Ramsey sequence sees it
@pulse.kernel
def ramsey_next_to(qubit, neighbour, excite: int, tau_ns: float):
    qubit_drive = get_drive_line(qubit)
    neighbour_drive = get_drive_line(neighbour)
    qubit_readout = get_readout_line(qubit)
    half_flip = qubit_drive.get_calibration(pulse.gates.X90)
    neighbour_flip = neighbour_drive.get_calibration(pulse.gates.X)
    measurement = qubit_readout.get_calibration(pulse.gates.READOUT)

    if excite:
        drive(neighbour_drive, neighbour_flip)
    sync(qubit_drive, neighbour_drive)

    drive(qubit_drive, half_flip)
    wait(qubit_drive, tau_ns * pulse.ns)
    drive(qubit_drive, half_flip)
    sync(qubit_drive, qubit_readout)
    return readout(qubit_readout, measurement)


for excite in (0, 1):
    result = pulse.simulate(
        ramsey_next_to,
        data[0],
        x2,
        excite,
        10_000.0,
    )
    excited = result.level_populations(data[0])[1]
    neighbour = "X2 excited" if excite else "X2 in ground"
    print(f"{neighbour:12}  D1 excited: {excited:.3f}")

# Prints:
#   X2 in ground  D1 excited: 0.845
#   X2 excited    D1 excited: 0.829
