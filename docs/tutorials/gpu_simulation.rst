..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Simulating on a GPU
===================

Every simulation in the earlier tutorials ran on the CPU, with the NumPy
reference simulator. The same programs also run on an NVIDIA GPU, through
NVIDIA cuDensityMat, with one argument changed.

What you'll learn:

- how to install and check GPU support;
- how to run a coupled pair on the CPU and then on the GPU;
- what form the program takes on its way to the GPU.

.. note::

   GPU simulation is a research preview: see
   :doc:`../user_guide/gpu_execution` for what it supports. The GPU code on
   this page needs Linux and an NVIDIA GPU, and the tests run it only there.
   Its CPU parts run anywhere.

All the code on this page is also one Python script,
:download:`gpu_simulation.py <scripts/gpu_simulation.py>`.

Install and check
-----------------

GPU support comes as an extra of the same package, for CUDA 12 or CUDA 13.
The CUDA libraries come from PyPI too, so the NVIDIA driver is the only
thing the machine needs.

Install one CUDA extra, unless the environment already has one, then run
the doctor:

.. code-block:: bash

   pip install "cudaq-pulse[cu12]"
   python -m cudaq_pulse.doctor

``cu12`` needs NVIDIA driver 525.60.13 or newer. ``cudaq-pulse[cu13]``
installs the CUDA 13 libraries instead and needs driver 580 or newer. Never
install both. The doctor reports the GPU runtime it loaded, the driver
version and the number of GPUs. It exits with an error when GPU execution
doesn't work.

The same check from Python raises an error that says what is missing:

.. code-block:: python
   :class: gpu

   from cudaq_pulse.runtime import require_gpu_runtime

   require_gpu_runtime()

A coupled pair on the CPU
-------------------------

The ancilla X1 and the data qubit D2 of ``transmon_surface17`` are coupled,
about 2.1 GHz apart. Their coupling carries its CZ calibration, whose
duration sets the slot a CZ gate takes:

.. code-block:: python

   import numpy as np

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   x1 = system.qudit("X1")
   d2 = system.qudit("D2")

   coupling = system.coupling(x1, d2)
   cz = coupling.get_calibration(pulse.gates.CZ)
   cz_slot = cz.duration_ns * pulse.ns

The kernel gives X1 a π/2 pulse and D2 a π pulse, then idles both for the
CZ's slot. Flux pulses aren't simulated yet, so the slot is a wait:

.. code-block:: python

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

.. themed-figure:: figures/pair_schedule.svg
   :alt: Timeline of evolve_pair on the drive lines of X1 and D2: a pi/2 pulse on X1 and a pi pulse on D2, a sync, then the CZ's 98 ns slot as a wait on both lines, to 138 ns.
   :width: 100%

   ``evolve_pair`` on X1 and D2.

What to notice: after the sync, nothing plays during the CZ slot. Only the
coupling acts.

Simulate it on the CPU, with a step fine enough for the pair's detuning:

.. code-block:: python

   cpu = pulse.simulate(evolve_pair, x1, d2, dt_ns=0.02)
   print("dimensions:", cpu.dims)

   for name, qudit in (("X1", x1), ("D2", d2)):
       excited = cpu.level_populations(qudit)[1]
       print(f"{name} excited: {excited:.4f}")

.. code-block:: text
   :class: output

   dimensions: (3, 3)
   X1 excited: 0.4944
   D2 excited: 0.9890

What to notice:

- the simulation holds only X1 and D2, each with three levels, so the
  state is a 9 × 9 density matrix;
- X1 ends half excited and D2 nearly fully, less what decays in the
  program's 138 ns.

.. themed-figure:: figures/pair_density.svg
   :alt: The magnitudes of the 9 by 9 density matrix of the pair after the CPU simulation, rows and columns labelled by the levels of X1 then D2. Almost all of it sits in 01 (0.50) and 11 (0.49), with the coherence between them off the diagonal; levels 2 stay empty.
   :width: 55%

   The magnitudes of the CPU result's density matrix.

What to notice: almost everything sits in two corners of the matrix, X1's
two levels with D2 excited. The off-diagonal pair is X1's superposition.

The step, ``dt_ns=0.02``, resolves the coupling, which turns at the pair's
2.1 GHz detuning in their rotating frames.

The same call on the GPU
------------------------

``backend="gpu"`` runs the same compiled program through cuDensityMat. The
result has the same layout as the CPU's, so the two density matrices can be
compared entry by entry:

.. code-block:: python
   :class: gpu

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

Choosing the integrator
-----------------------

``integrator`` picks the GPU's time-stepping scheme:

- fixed-step Runge-Kutta of order 1, 2 or 4 (``"rk4"`` by default);
- a Magnus expansion;
- Crank-Nicolson, for stiff problems.

Long stretches where nothing plays, such as the CZ slot here, are
propagated exactly on the host instead of stepped.

What runs on the GPU
--------------------

The compiler lowers the pulse program through an operator dialect, QOp,
which writes the Hamiltonian and the Lindblad terms of the decay. It ends
as calls into the cuDensityMat runtime.

You can see the last stage on any machine:

.. code-block:: python

   scheduled = evolve_pair.lower(x1, d2)
   llvm = scheduled.mlir("llvm")
   print("calls into cuDensityMat:", llvm.count("llvm.call"))

.. code-block:: text
   :class: output

   calls into cuDensityMat: 88

.. pulse-visual:: figures/gpu_simulation.json
   :scene: lowering
   :alt: One program, lowered for the GPU and exported. The kernel as written: pulses, waits and syncs on its lines. It holds 21 operations. QOp writes the pair's Hamiltonian and the Lindblad terms of its decay. It holds 82 operations. The cuDensityMat dialect is the cuDensityMat API as IR. It holds 122 operations. LLVM IR turns each of its operations into a call into the cuDensityMat runtime. It holds 1643 operations, 88 of them calls into cuDensityMat. From the same Pulse MLIR, the program exports for a controller as 5 events. Only the last steps differ.

   ``evolve_pair`` at each level of ``mlir()``, one stage at a time, with
   the operations each holds, counted from its text.

What to notice: the program you simulate on the GPU is the same one you
would export to a control system. Only the last steps of compilation
differ.

Where to go next
----------------

- :example:`transmon/transmon_surface17_evolve_gpu.py` runs this pair on
  the GPU and prints its final state.
- :example:`transmon/transmon_surface17_full_pipeline.py` prints each stage
  of the lowering: the Pulse MLIR, the QOp form and the LLVM IR.

For the details:

- :doc:`../user_guide/gpu_execution` lists the driver requirements, the GPU
  errors, and what GPU simulation doesn't support yet;
- :doc:`../architecture/dialects` describes QOp and cuDensityMat.
