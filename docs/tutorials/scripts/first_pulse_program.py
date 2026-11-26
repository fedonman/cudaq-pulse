# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Your First Pulse Program: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/first_pulse_program.html

Generated from docs/tutorials/first_pulse_program.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

import importlib.util


# Blocks under ``if VIZ:`` need matplotlib (``pip install "cudaq-pulse[viz]"``).
VIZ = importlib.util.find_spec("matplotlib") is not None


# %% Pick a quantum system
import cudaq_pulse as pulse

system = pulse.quantum_system("transmon_surface17")
qubit = system.qudits[0]

frequency_ghz = system.qudit_freq[0] / 1e9
print(qubit)
print(f"0-1 transition: {frequency_ghz} GHz")
print(f"clock:          {system.clock_ghz} GHz")

# Prints:
#   <Qudit 0 -> qudit 0 of transmon_surface17>
#   0-1 transition: 3.885 GHz
#   clock:          2.0 GHz


# %% Write a pulse kernel
@pulse.kernel
def rabi(qudit, amplitude: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    flip = gaussian(
        duration=32 * pulse.ns,
        amplitude=amplitude,
        sigma=8 * pulse.ns,
    )

    drive(drive_line, flip)
    # Start the readout after the drive pulse ends.
    sync(drive_line, readout_line)
    return readout(readout_line, square(300 * pulse.ns, 0.05))


# %% Simulate it
result = pulse.simulate(rabi, qubit, 0.16)
levels = result.level_populations(qubit)

for level, population in enumerate(levels):
    print(f"level {level}: {population:.3f}")

# Prints:
#   level 0: 0.036
#   level 1: 0.963
#   level 2: 0.001


# %% Sweep the amplitude
import numpy as np

amplitudes = np.linspace(0, 0.4, 41)
populations = []
for amplitude in amplitudes:
    result = pulse.simulate(rabi, qubit, amplitude)
    populations.append(result.level_population(1))

print("compilations:", rabi.cache_info().compiled)

# Prints:
#   compilations: 1


# %% Take shots
result = pulse.simulate(
    rabi,
    qubit,
    0.16,
    shots_count=1000,
    seed=7,
)
outcomes = np.asarray(result.records)

print("shape:", outcomes.shape)
print("fraction that read 1:", outcomes.mean())

# Prints:
#   shape: (1000, 1)
#   fraction that read 1: 0.956


# %% See what was compiled
scheduled = rabi.lower(qubit, 0.16)

print(f"{'kind':8} {'start':>6} {'length':>6}")
for event in scheduled.schedule:
    start = event.start_cycles
    length = event.duration_cycles
    print(f"{event.kind:8} {start:>6} {length:>6}")

# Prints:
#   kind      start length
#   drive         0     64
#   readout      64    600


# %% Draw it
if VIZ:
    figure = scheduled.timeline()
    figure.savefig("rabi_schedule.svg")


# %% The same program as MLIR
print(scheduled.mlir())

# Prints, wrapped here, with ... for text left out:
#   module @rabi attributes {...} {
#     func.func @main()
#         attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
#       %cst = arith.constant 1.600000e-01 : f64
#       %cst_0 = arith.constant 0.000000e+00 : f64
#       %cst_1 = arith.constant 5.000000e-02 : f64
#       %c600_i64 = arith.constant 600 : i64
#       %cst_2 = arith.constant 1.600000e+01 : f64
#       %c64_i64 = arith.constant 64 : i64
#       %0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
#       %line, %tone = pulse.get_drive_line %0
#           {frequency_hz = 3884968867.2938495 : f64}
#           : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
#       %line_3, %tone_4 = pulse.get_readout_line %0
#           {frequency_hz = 6.769000e+09 : f64}
#           : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
#       %1 = pulse.gaussian %c64_i64, %cst, %cst_2
#           : i64, f64, f64 -> !pulse.waveform
#       %updated_line, %updated_tone = pulse.drive %line, %1, %tone
#           <{duration_cycles = 64 : i64, start_cycles = 0 : i64}>
#           : !pulse.drive_line, !pulse.waveform, !pulse.tone
#           -> !pulse.drive_line, !pulse.tone
#       %2:2 = pulse.sync %updated_line, %line_3
#           : !pulse.drive_line, !pulse.readout_line
#           -> !pulse.drive_line, !pulse.readout_line
#       %3 = pulse.square %c600_i64, %cst_1, %cst_0
#           : i64, f64, f64 -> !pulse.waveform
#       %updated_line_5, %updated_tone_6, %result =
#           pulse.readout %2#1, %3, %tone_4, iq
#           <{duration_cycles = 600 : i64, start_cycles = 64 : i64}>
#           : !pulse.readout_line, !pulse.waveform, !pulse.tone
#           -> !pulse.readout_line, !pulse.tone, !pulse.measurement
#       return
#     }
#   }
