..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Several Qubits: Registers, Sync and Scheduling
==============================================

Programs on more than one qubit need two things a single-qubit program
doesn't: a way to pass a group of qubits to a kernel, and a way to say
which pulses must line up in time.

What you'll learn:

- how to drive a register of qubits in parallel;
- how ``sync`` lines pulses up across lines;
- how two scheduling policies place the same pulses;
- how a coupled neighbour shifts a qubit's frequency.

All the code on this page is also one Python script,
:download:`several_qubits.py <scripts/several_qubits.py>`.

A register of qubits
--------------------

Slicing the quantum system's qudits gives a *register*, which a kernel
iterates like a list. Take D1, D2 and D3, and a kernel that plays a π
pulse on each qubit of the register:

.. code-block:: python

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   data = system.qudits[0:3]


   @pulse.kernel
   def flip_all(qudits):
       for qudit in qudits:
           drive_line = get_drive_line(qudit)
           flip = drive_line.get_calibration(pulse.gates.X)
           drive(drive_line, flip)

The loop runs while the kernel compiles, once per qubit. Lower it to see
when each pulse plays:

.. code-block:: python

   scheduled = flip_all.lower(data)

   print(f"{'kind':6} {'qudit':>5} {'start':>6}")
   for event in scheduled.schedule:
       start = event.start_cycles
       print(f"{event.kind:6} {event.qudit:>5} {start:>6}")

.. code-block:: text
   :class: output

   kind   qudit  start
   drive      0      0
   drive      1      0
   drive      2      0

.. themed-figure:: figures/register_parallel.svg
   :alt: Timeline of flip_all on 3 drive lines, D1 drive, D2 drive, D3 drive. Each plays its own calibrated pi pulse, all from 0 to 40 ns.
   :width: 100%

   ``flip_all`` on D1, D2 and D3.

What to notice: each qubit has its own drive line, so the three π pulses
play at the same time.

Each line uses its own qubit's calibrated pulse, since every qubit has its
own frequency and drive strength:

.. code-block:: python

   result = pulse.simulate(flip_all, data)
   for qudit in data:
       label = system.physical_qudits[qudit.index].label
       excited = result.level_populations(qudit)[1]
       print(f"{label} excited: {excited:.3f}")

.. code-block:: text
   :class: output

   D1 excited: 0.978
   D2 excited: 0.992
   D3 excited: 0.984

The simulation holds only the qubits the program uses: a 27-dimensional
space for three three-level transmons. The other 14 qubits stay out of it,
in their ground state.

Line pulses up with sync
------------------------

Pulses on different lines are independent until ``sync`` ties them
together. Every line it names waits for the latest of them.

Here one qubit gets three π pulses and the other one, and both are read
out together afterwards. The kernel finds its lines and pulses, plays the
pulses, lines everything up, and reads both qubits:

.. code-block:: python

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

Two scheduling policies
-----------------------

Where should the second qubit's single pulse go: at the start, or just
before the readouts?

- By default, the compiler schedules every operation as late as possible
  (``"alap"``), which keeps a qubit's state fresh until it is measured.
- ``schedule="asap"`` places operations as early as possible instead.

Compile with each and find the second qubit's pulse:

.. code-block:: python

   for policy in ("alap", "asap"):
       compiled = pulse.compile(uneven, schedule=policy)
       program = compiled.lower(data[0], data[1])
       starts = []
       for event in program.schedule:
           if event.qudit == 1 and event.kind == "drive":
               starts.append(event.start_cycles)
       print(f"{policy}: second qubit's pulse at cycle {starts}")

.. code-block:: text
   :class: output

   alap: second qubit's pulse at cycle [160]
   asap: second qubit's pulse at cycle [0]

.. themed-figure:: figures/alap_asap.svg
   :alt: Two timelines of the uneven kernel on D1 and D2. D1 plays three pi pulses and D2 one, and both readouts start at 120 ns. As late as possible, D2's pulse plays from 80 ns, right before the readouts; as early as possible, from 0 ns, and D2 then idles until it is measured.
   :width: 100%

   ``uneven`` under both policies, in nanoseconds of the 2 GHz clock.

What to notice:

- only the second qubit's π pulse moves;
- as late as possible, it plays right before the readouts at cycle 240;
- as early as possible, the qubit then idles, and decays, for 160 cycles
  before it is measured.

Two coupled qubits
------------------

D1 isn't coupled to D2 or D3, but it is coupled to the ancilla X2, through
an exchange interaction of a few megahertz:

.. code-block:: python

   x2 = system.qudit("X2")
   coupling = system.coupling(data[0], x2)
   assert coupling.strength_hz is not None  # None: not measured yet
   idle_zz_hz = system.calibration.idle_zz_hz("X2-D1")

   print(f"g / 2π:  {coupling.strength_hz / 1e6:.2f} MHz")
   print(f"idle ZZ: {idle_zz_hz / 1e3:.2f} kHz")

.. code-block:: text
   :class: output

   g / 2π:  5.69 MHz
   idle ZZ: -5.94 kHz

Because of the coupling, D1's frequency depends slightly on whether X2 is
excited, by about 6 kHz. This is the residual ZZ interaction, an error
source for every gate played next to a busy neighbour.

.. themed-figure:: figures/zz_shift.svg
   :alt: D1's 0-1 frequency with its neighbour X2 in its ground state and excited. Through their 5.69 MHz coupling, exciting X2 moves D1 by -5.94 kHz, the residual ZZ shift.
   :width: 80%

   D1's frequency with X2 in each state.

What to notice: the shift is a few kilohertz, a millionth of D1's
frequency, yet it adds up over a long wait.

A Ramsey sequence sees it
-------------------------

A Ramsey sequence on D1 can see the shift: excite X2 first or don't, and
compare. The kernel takes both qubits and the wait. It sets the
neighbour's state, then plays the two π/2 pulses of the Ramsey sequence
around the wait:

.. code-block:: python

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

The ``excite`` argument is an integer, so the kernel compiles once for each
value, and the ``if`` is decided while compiling, not during the shot.

.. themed-figure:: figures/ramsey_neighbour.svg
   :alt: Timeline of the Ramsey kernel on D1 next to X2, with X2 excited and a 10 microsecond wait cut short. X2's pi pulse plays first; then D1 plays a pi/2 pulse, waits, plays a second pi/2 pulse and is read out.
   :width: 100%

   The kernel with X2 excited and a 10 µs wait, cut short.

What to notice: X2's π pulse comes first, so D1's wait happens next to an
excited neighbour.

Compare the two cases after 10 µs:

.. code-block:: python

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

.. code-block:: text
   :class: output

   X2 in ground  D1 excited: 0.845
   X2 excited    D1 excited: 0.829

After 10 µs, D1's excited-state population differs by about 0.02 with its
neighbour. That comes from a phase difference of about 0.37 radians: 6 kHz
over 10 µs. With X2 in its ground state D1 keeps in step with its drive
line, which plays at D1's frequency with its couplings included; what
remains is the decay of the wait.

.. themed-figure:: figures/zz_ramsey.svg
   :alt: D1's excited-state population against the Ramsey wait, from 0 to 20 microseconds, with its neighbour X2 in its ground state and excited. The two curves start together and separate as the wait grows, by 0.03 at 20 microseconds.
   :width: 100%

   Ramsey on D1 with its neighbour X2 in the ground state or excited.

What to notice: the curves start together and separate as the ZZ shift
accumulates phase.

For larger groups of coupled qubits, :doc:`gpu_simulation` runs the same
programs on an NVIDIA GPU.

Where to go next
----------------

- :example:`07_multi_qubit_sync.py` builds programs on registers, with
  ``sync`` barriers between groups of lines.
- :example:`10_scheduling_comparison.py` compares as-late-as-possible,
  as-early-as-possible and resource-constrained schedules of one kernel.
- :example:`18_per_kernel_schedules.py` gives each kernel of a stabilizer
  cycle its own scheduling policy.
- :example:`02_two_qubit_cross_resonance.py` and
  :example:`17_ghz_state_prep.py` build two-qubit gates and a GHZ state.
  Their amplitudes are illustrative, not calibrated for this system.

For the details, :doc:`../user_guide/kernels` covers registers, and
:doc:`../user_guide/compilation` what each annotation makes static.
