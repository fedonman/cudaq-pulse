..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Your First Pulse Program
========================

This tutorial goes from installing CUDA-Q Pulse to a simulated Rabi
oscillation on one transmon qubit. Everything runs on a laptop CPU, with no
GPU or quantum hardware.

What you'll learn:

- how to pick a quantum system and one of its qubits;
- how to write a pulse kernel;
- how to simulate it, sweep its amplitude and take shots;
- how to see the schedule the compiler made.

All the code on this page is also one Python script,
:download:`first_pulse_program.py <scripts/first_pulse_program.py>`.

Install
-------

CUDA-Q Pulse needs Python 3.12 or newer, on Linux or macOS. Install it in a
virtual environment, then run the doctor:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install cudaq-pulse
   python -m cudaq_pulse.doctor

The doctor lists what was installed and checks that it works. Without a
CUDA extra, it ends with:

.. code-block:: text

   Result: OK, compilation and CPU simulation work; GPU simulation is not set up.

That is all this tutorial needs. :doc:`../getting_started` lists the
supported platforms and the GPU extras.

Pick a quantum system
---------------------

A pulse program runs on a *quantum system*: the calibrated physics of a
machine. It holds:

- each qudit's frequencies and coherence times;
- the couplings between qudits;
- the clock that pulses are timed against.

CUDA-Q Pulse ships a model of a 17-qubit superconducting processor built
for a distance-3 surface code. Take its first qubit:

.. code-block:: python

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudits[0]

   frequency_ghz = system.qudit_freq[0] / 1e9
   print(qubit)
   print(f"0-1 transition: {frequency_ghz} GHz")
   print(f"clock:          {system.clock_ghz} GHz")

.. code-block:: text
   :class: output

   <Qudit 0 -> qudit 0 of transmon_surface17>
   0-1 transition: 3.885 GHz
   clock:          2.0 GHz

CUDA-Q Pulse calls it a qudit because a transmon has more than two levels.
The simulator keeps the lowest three, so a pulse that drives too hard shows
up as population in level 2.

.. themed-figure:: figures/qudit_levels.svg
   :alt: The three lowest levels of qudit 0, the levels the simulator keeps: the 0-1 transition at 3.885 GHz, the one the drive is tuned to, and the 1-2 transition at 3.701 GHz, 184 MHz lower.
   :width: 60%

   The three levels the simulator keeps for this qubit.

What to notice: the 1-2 transition sits a little below the 0-1 one. A drive
tuned to 0-1 mostly leaves level 2 alone.

Write a pulse kernel
--------------------

A pulse program is a Python function with the ``@pulse.kernel`` decorator.
Each qudit has two lines:

- a drive line, which carries the microwave pulses that rotate it;
- a readout line, which measures it.

This kernel plays one pulse on the drive line, then reads the qubit out:

.. code-block:: python

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

What the kernel does:

- the drive is a Gaussian pulse, 32 ns long and 8 ns wide, at an amplitude
  you choose for each call;
- ``sync`` lines the two lines up, so the 300 ns readout starts when the
  drive pulse ends;
- the kernel returns the readout's outcome.

.. themed-figure:: figures/rabi_pulse.svg
   :alt: The drive pulse of the rabi kernel at amplitude 0.16: a Gaussian 32 ns long, peaking in the middle and cut off at each end at about 14% of its peak.
   :width: 100%

   The drive pulse at amplitude 0.16, as the drive line plays it.

What to notice: the pulse does not start or end at zero. The Gaussian is
cut off two ``sigma`` from its peak, at about 14% of its height; a longer
duration for the same ``sigma`` brings the ends closer to zero. Its smooth
rise and fall keep its spectrum narrow.

Inside a kernel, the operations are used by their bare names. Editors that
use Pylance or pyright complete and check them with no setup (see
:doc:`../user_guide/editors`).

Durations carry units, so the same kernel runs unchanged on a quantum
system with another clock.

Simulate it
-----------

``pulse.simulate`` compiles the kernel for the qudit's quantum system and
evolves it on the CPU, with the qubit's relaxation and dephasing included:

.. code-block:: python

   result = pulse.simulate(rabi, qubit, 0.16)
   levels = result.level_populations(qubit)

   for level, population in enumerate(levels):
       print(f"level {level}: {population:.3f}")

.. code-block:: text
   :class: output

   level 0: 0.036
   level 1: 0.963
   level 2: 0.001

.. themed-figure:: figures/rabi_levels.svg
   :alt: The population of each level of qudit 0 after the 0.16 pulse: level 0, 0.036; level 1, 0.963; level 2, 0.001.
   :width: 100%

   The population of each level after the pulse, on a log scale.

What to notice: level 2 holds a thousandth of the population, so this
pulse barely leaks out of the qubit.

At amplitude 0.16 the pulse is close to a π pulse (the sweep below puts it
at 0.165), and 96% of the population ends up excited.

Where the rest goes:

- the qubit starts in a thermal mixture with about 2% already excited,
  which the pulse flips the other way;
- it relaxes a little during the 300 ns readout.

To start from a known state instead of the thermal mixture, pass
``initial_state`` to ``pulse.simulate``:
``pulse.simulate(rabi, qubit, 0.16, initial_state=[1, 0, 0])`` starts the
qubit in level 0. The list holds one amplitude per level, so ``[1, 0, 0]``
puts all of the population in level 0 and none in levels 1 and 2.

Sweep the amplitude
-------------------

A Rabi experiment repeats the program over a range of amplitudes:

.. code-block:: python

   import numpy as np

   amplitudes = np.linspace(0, 0.4, 41)
   populations = []
   for amplitude in amplitudes:
       result = pulse.simulate(rabi, qubit, amplitude)
       populations.append(result.level_population(1))

   print("compilations:", rabi.cache_info().compiled)

.. code-block:: text
   :class: output

   compilations: 1

Because ``amplitude`` is annotated ``float``, it stays a parameter of the
compiled program instead of being fixed into it. So the 41 simulations
share one compilation.

.. themed-figure:: figures/rabi_sweep.svg
   :alt: Simulated excited-state population after the 32 ns Gaussian pulse, for drive amplitudes from 0 to 0.4, with a cosine fit. The population rises from 0.02 to 0.96; the fit puts the pi pulse at amplitude 0.165, and the population falls back near 0.33, a full turn.
   :width: 100%

   The excited-state population after the pulse, for each amplitude, with a
   cosine fit.

What to notice: the fit puts the π pulse at 0.165, and the population
returns close to zero near 0.33, a full rotation.

Take shots
----------

A simulation gives probabilities, while an experiment gives one outcome per
shot. Ask for shots, and the simulator samples them from the same
evolution:

.. code-block:: python

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

.. code-block:: text
   :class: output

   shape: (1000, 1)
   fraction that read 1: 0.956

``records`` holds one row per shot and one column per readout, here the
kernel's single readout.

.. themed-figure:: figures/rabi_shots.svg
   :alt: The fraction of the first n of 1000 sampled shots that read 1, against n on a log scale. It wanders at first and settles at 0.956, inside a band of two standard errors around the excited-state population of 0.963, which narrows as the shots add up.
   :width: 100%

   The mean of the first *n* shots against *n*, with a band of two standard
   errors around the population.

What to notice: the mean wanders while there are few shots, and settles
near the population as the band narrows as 1/√n.

See what was compiled
---------------------

``lower`` compiles and schedules a call without running it, so you can see
when each pulse plays:

.. code-block:: python

   scheduled = rabi.lower(qubit, 0.16)

   print(f"{'kind':8} {'start':>6} {'length':>6}")
   for event in scheduled.schedule:
       start = event.start_cycles
       length = event.duration_cycles
       print(f"{event.kind:8} {start:>6} {length:>6}")

.. code-block:: text
   :class: output

   kind      start length
   drive         0     64
   readout      64    600

Times count cycles of the 2 GHz clock. The 32 ns drive pulse is 64
cycles, and the readout starts at cycle 64, where ``sync`` put it.

Draw it
-------

``timeline()`` draws the schedule with matplotlib. Install it with the
``[viz]`` extra:

.. code-block:: bash

   pip install "cudaq-pulse[viz]"

Then draw and save the schedule:

.. code-block:: python
   :class: viz

   figure = scheduled.timeline()
   figure.savefig("rabi_schedule.svg")

.. themed-figure:: figures/rabi_schedule.svg
   :alt: Timeline of one call of the Rabi kernel on D1: a 32 ns Gaussian drive pulse, then, where sync puts it, a 300 ns square readout pulse on the readout line.
   :width: 100%

   The schedule of one call, in nanoseconds: each pulse is its envelope as
   the controller plays it. ``timeline()`` draws the same schedule as bars,
   in clock cycles.

The same program as MLIR
------------------------

``mlir()`` returns the program in the Pulse dialect, the form that compiler
passes, simulators and controller backends read. The text has one operation
per line. This page wraps the long lines to fit, and writes ``{...}`` for
the module's attributes, one long line of settings, the clock and the qubit
among them:

.. code-block:: python

   print(scheduled.mlir())

.. code-block:: text
   :class: output abridged

   module @rabi attributes {...} {
     func.func @main()
         attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
       %cst = arith.constant 1.600000e-01 : f64
       %cst_0 = arith.constant 0.000000e+00 : f64
       %cst_1 = arith.constant 5.000000e-02 : f64
       %c600_i64 = arith.constant 600 : i64
       %cst_2 = arith.constant 1.600000e+01 : f64
       %c64_i64 = arith.constant 64 : i64
       %0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
       %line, %tone = pulse.get_drive_line %0
           {frequency_hz = 3884968867.2938495 : f64}
           : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
       %line_3, %tone_4 = pulse.get_readout_line %0
           {frequency_hz = 6.769000e+09 : f64}
           : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
       %1 = pulse.gaussian %c64_i64, %cst, %cst_2
           : i64, f64, f64 -> !pulse.waveform
       %updated_line, %updated_tone = pulse.drive %line, %1, %tone
           <{duration_cycles = 64 : i64, start_cycles = 0 : i64}>
           : !pulse.drive_line, !pulse.waveform, !pulse.tone
           -> !pulse.drive_line, !pulse.tone
       %2:2 = pulse.sync %updated_line, %line_3
           : !pulse.drive_line, !pulse.readout_line
           -> !pulse.drive_line, !pulse.readout_line
       %3 = pulse.square %c600_i64, %cst_1, %cst_0
           : i64, f64, f64 -> !pulse.waveform
       %updated_line_5, %updated_tone_6, %result =
           pulse.readout %2#1, %3, %tone_4, iq
           <{duration_cycles = 600 : i64, start_cycles = 64 : i64}>
           : !pulse.readout_line, !pulse.waveform, !pulse.tone
           -> !pulse.readout_line, !pulse.tone, !pulse.measurement
       return
     }
   }

What to notice: the kernel's ``drive``, ``sync`` and ``readout`` are
``pulse.drive``, ``pulse.sync`` and ``pulse.readout``, in that order. The
drive and the readout carry the start and duration in cycles listed above,
between ``<{`` and ``}>``. The constants at the top are the program's
numbers, such as the pulse lengths of 64 and 600 cycles and this call's
amplitude, 0.16, which ``lower`` fills in. The drive line's frequency is
qudit 0's 0-1 transition with its couplings to its neighbours included,
31 kHz below the 3.885 GHz printed above: its ``dressed_frequency_hz``.

You don't need to read MLIR to use CUDA-Q Pulse, but it's there when you
want to check what the compiler did. :doc:`../architecture/dialects`
describes it.

Where to go next
----------------

Each of these examples takes one idea from this page further. Run them from
a clone of the repository with ``python examples/<file>``:

- :example:`01_single_qubit_rabi.py` sweeps a square pulse instead, and
  compiles the kernel for a quantum system with a 1 GHz clock.
- :example:`03_t1_t2_dissipator.py` compares a simulated T1 decay with the
  exponential the model predicts, alongside Ramsey and Hahn-echo kernels.
- :example:`08_readout_and_branching.py` branches on a measurement, as in
  an active reset.
- :example:`16_visualization.py` draws the schedules of a three-qubit
  program under two scheduling policies.
- :example:`transmon/transmon_surface17_rabi.py` drives qubit D5 with its
  calibrated π pulse and follows the program down to the operator form the
  simulator evolves.

The next tutorial, :doc:`coherence`, measures T1 and dephasing with the
quantum system's calibrated pulses.

For the details behind this page:

- :doc:`../user_guide/kernels` covers what a kernel can contain;
- :doc:`../user_guide/quantum_systems` covers how quantum systems are
  described and calibrated;
- :doc:`../user_guide/compilation` covers how calls compile and what they
  return.
