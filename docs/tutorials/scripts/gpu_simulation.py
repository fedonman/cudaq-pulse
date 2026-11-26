# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulating on a GPU: the code of the tutorial, as one script.

https://nvidia.github.io/cudaq-pulse/tutorials/gpu_simulation.html

Generated from docs/tutorials/gpu_simulation.rst by tools/tutorial-scripts.py: edit
the page, not this file.
"""

# Blocks under ``if GPU:`` need the GPU runtime (Linux, an NVIDIA GPU and
# ``pip install "cudaq-pulse[cu12]"`` or ``[cu13]``).
def _gpu_runtime_loads() -> bool:
    from cudaq_pulse.runtime import require_gpu_runtime

    try:
        require_gpu_runtime()
    except Exception:
        return False
    return True


GPU = _gpu_runtime_loads()


# %% Install and check
if GPU:
    from cudaq_pulse.runtime import require_gpu_runtime

    require_gpu_runtime()


# %% A coupled pair on the CPU
import numpy as np

import cudaq_pulse as pulse

system = pulse.quantum_system("transmon_surface17")
x1 = system.qudit("X1")
d2 = system.qudit("D2")

coupling = system.coupling(x1, d2)
cz = coupling.get_calibration(pulse.gates.CZ)
cz_slot = cz.duration_ns * pulse.ns


@pulse.kernel
def evolve_pair(a, b):
    drive_a = get_drive_line(a)
    drive_b = get_drive_line(b)
    half_flip = drive_a.get_calibration(pulse.gates.X90)
    flip = drive_b.get_calibration(pulse.gates.X)

    drive(drive_a, half_flip)
    drive(drive_b, flip)
    sync(drive_a, drive_b)
    wait(drive_a, cz_slot)
    wait(drive_b, cz_slot)


cpu = pulse.simulate(evolve_pair, x1, d2, dt_ns=0.02)
print("dimensions:", cpu.dims)

for name, qudit in (("X1", x1), ("D2", d2)):
    excited = cpu.level_populations(qudit)[1]
    print(f"{name} excited: {excited:.4f}")

# Prints:
#   dimensions: (3, 3)
#   X1 excited: 0.4944
#   D2 excited: 0.9890


# %% The same call on the GPU
if GPU:
    gpu = pulse.simulate(
        evolve_pair,
        x1,
        d2,
        backend="gpu",
        dt_ns=0.02,
    )
    for name, qudit in (("X1", x1), ("D2", d2)):
        excited = gpu.level_populations(qudit)[1]
        print(f"{name} excited: {excited:.4f}")

    difference = np.abs(gpu.density_matrix - cpu.density_matrix)
    print("largest difference:", difference.max())


# %% What runs on the GPU
scheduled = evolve_pair.lower(x1, d2)
llvm = scheduled.mlir("llvm")
print("calls into cuDensityMat:", llvm.count("llvm.call"))

# Prints:
#   calls into cuDensityMat: 88
