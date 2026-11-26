..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Lowering to Your Controller's Instruction Set
=============================================

.. note::

   Experimental. The export this page lowers from is a versioned format,
   but how a backend spans several controllers may change.

This tutorial is for people who build control systems or QPUs. You write
a small pass that turns a compiled pulse program into the instructions of
a toy controller.

What you'll learn:

- how to describe a controller, and a QPU built from two of them, as data
- what the compiler hands a backend: the scheduled program, as data
- how to lower that program to an instruction set, one kind of event at
  a time
- how to test the result with ``backends.validate``

The full code is in :example:`isa_lowering/toy_isa.py`. Save it beside
your own code: this page shows it piece by piece and imports it.

All the code on this page is also one Python script,
:download:`isa_lowering.py <scripts/isa_lowering.py>`.

A toy instruction set
---------------------

Each toy controller has one sequencer. It issues these instructions in
order:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Instruction
     - What it does
   * - ``set_freq ch0 5002000000``
     - Sets channel 0's oscillator to 5.002 GHz.
   * - ``shift_phase ch0 0.502655``
     - Adds 0.502655 rad to its phase.
   * - ``play ch0 w0 0.25``
     - Starts waveform ``w0`` on channel 0, at gain 0.25.
   * - ``acquire ch1.0 w1 0.05 m4``
     - Starts ``w1`` on tone 0 of channel 1, and acquires into bit ``m4``.
   * - ``wait 35``
     - Lets 35 timesteps pass.
   * - ``branch !m4 end0``
     - Jumps to ``end0`` when bit ``m4`` is 0.
   * - ``branch m4 end0``
     - Jumps to ``end0`` when bit ``m4`` is 1.
   * - ``jump end0``
     - Jumps to ``end0``.
   * - ``submit L0 m0 m1``
     - Sends bits ``m0`` and ``m1`` to the decoder on link 0.
   * - ``answer L0 a3``
     - Waits for the decoder's answer, into bit ``a3``.
   * - ``poll L0 a3``
     - Takes the answer if it is there. Bit ``a3.ok`` says whether it was.

A few rules complete the picture:

- Only ``wait`` moves time. A ``play`` or ``acquire`` starts a pulse and
  returns.
- A channel that several qudits share has an oscillator per tone, such as
  ``ch1.0`` and ``ch1.1``.
- Oscillators start each shot at phase 0.
- Outcome bits reach every sequencer over a shared bus.
- There are no loops and no registers to compute with.

Describe the controllers as data
--------------------------------

Before any lowering, the compiler needs to know what each controller can
do. You give it that as data.

The controller description
~~~~~~~~~~~~~~~~~~~~~~~~~~

A controller description is a file its maker ships, one per model. These
are the toy model's limits:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: toml
   :start-at: # Pulses start and end
   :end-at: supports_timeouts

The file also names the model and lists what it supports: plays,
readouts, waits, syncs, branches, forwarding outcomes, and decoder calls.
Save it next to your code:

.. code-block:: python

   import math
   from pathlib import Path

   import cudaq_pulse as pulse
   from toy_isa import DESCRIPTION

   description = Path("toy_sequencer.toml")
   description.write_text(DESCRIPTION)

Two controllers, wired
~~~~~~~~~~~~~~~~~~~~~~

Which qudits a controller drives is the installation's part, not the
maker's. Both toy controllers are the same model, wired differently:

.. code-block:: python

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

.. code-block:: text
   :class: output

   model:      toy.sequencer
   max pulses: 256

What to notice:

- The first controller drives qudits 0 and 1, and reads both on one
  feedline.
- The second controller drives and reads qudit 2.
- An outcome takes 40 more cycles to reach the other controller.

One QPU
~~~~~~~

The QPU has three qudits, each with its frequency and coherence times:

.. code-block:: python

   from cudaq_pulse.quantum_systems import PhysicalQudit

   qudits = {}
   for number in range(3):
       qudits[number] = PhysicalQudit(
           index=number,
           frequency_hz=5.0e9 + 3.0e8 * number,
           t1_us=50.0,
           t2_star_us=30.0,
       )

The control system holds both controllers, the clock and the link to the
decoder. The QPU puts it together with the qudits. It is one value, which
``as_dict`` saves as a file:

.. code-block:: python

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

.. code-block:: text
   :class: output

   controllers: 2
   second line drives: [2]

Both controllers count from the 250 MHz reference clock. A call to the
decoder takes 4 ns to go out and 8 ns to come back, on top of its own
latency. Every compile now checks a program against these two controllers.

.. themed-figure:: figures/toy_qpu_wiring.svg
   :alt: Diagram of the toy QPU. 3 qudits at 5, 5.3, 5.6 GHz and 2 toy.sequencer controllers: controller 0 drives qudits 0 and 1, and one feedline reads qudits 0 and 1; controller 1 drives qudit 2, and a readout line reads qudit 2. An outcome takes 40 cycles from one controller to the other. The decoder link of the control system, drawn to both controllers, takes 4 ns out and 8 ns back. The reference clock is 250 MHz and the program clock 1 GHz.
   :width: 100%

   The QPU as ``as_dict`` saves it. Each controller's drive lines are
   solid and its readout lines dashed; the first controller reads its two
   qudits on one feedline. The decoder's link belongs to the control
   system.

The program before lowering
---------------------------

This section shows what a backend receives: the scheduled program.

A Ramsey sequence with a reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel is a detuned Ramsey sequence, then an active reset:

.. code-block:: python

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

What to notice: the reset flips the qudit back when it reads 1.

The scheduled IR
~~~~~~~~~~~~~~~~

Compile with the resource-constrained schedule, ``rcp``. The scheduled
pulse dialect gives each operation a place in time, as the attributes
``start_cycles`` and ``duration_cycles``. The regular expression below
picks each operation's name and those two attributes out of the MLIR
text, to print its start and length in clock cycles:

.. code-block:: python

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

.. code-block:: text
   :class: output

   pulse.drive             start   0  length  40
   pulse.shift_frequency   start  40  length   0
   pulse.wait              start  40  length 100
   pulse.drive             start 140  length  40
   pulse.shift_frequency   start 180  length   0
   pulse.readout           start 180  length 200
   pulse.drive             start 460  length  40

What to notice:

- The last drive sits inside an ``scf.if`` on the readout.
- No phase shift is left. The compiler folded the virtual Z into the
  second drive.

.. themed-figure:: figures/ramsey_reset_schedule.svg
   :alt: Timeline of the scheduled program on qudit 0's drive and readout lines, with a lane for the outcome. An x90 pulse from 0 to 40 ns, a 100 ns wait, a second x90 pulse detuned by +2 MHz from 140 to 180 ns, then the readout from 180 to 380 ns. The outcome is ready 80 ns after the readout ends, and the x180 pulse, which plays only if the readout read 1, runs from 460 to 500 ns.
   :width: 100%

   The same schedule, drawn from the program's export. This QPU's clock
   is 1 GHz, so a cycle is 1 ns. The dashed x180 pulse plays only when the
   readout reads 1, and starts once the outcome is back, after the
   controller's feedback latency.

The export
~~~~~~~~~~

The export is the same program as versioned data. It is the one to read:

.. code-block:: python

   program = scheduled.export_to_controller()

   for event in program["events"]:
       if event["kind"] not in ("drive", "readout"):
           continue
       start = event["start_cycles"]
       gigahertz = event["frequency_hz"] / 1e9
       phase = event["phase_rad"]
       print(f"{event['kind']:<7} at {start:>3}: "
             f"{gigahertz:.3f} GHz, {phase:.6f} rad")

.. code-block:: text
   :class: output

   drive   at   0: 5.000 GHz, 0.000000 rad
   drive   at 140: 5.002 GHz, 2.827433 rad
   readout at 180: 5.000 GHz, 0.000000 rad
   drive   at 460: 5.000 GHz, 3.330088 rad

Each pulse carries its absolute frame. So the sequencer never has to
add up phase itself.

The reset's π pulse is conditional. Its event says on what:

.. code-block:: python

   reset_pulse = program["events"][6]
   condition = reset_pulse["conditions"][0]
   print("test:", condition["test"])
   print("on readouts:", condition["readouts"])

.. code-block:: text
   :class: output

   test: is_set
   on readouts: [4]

It plays when the outcome of event 4, the readout, is set.

The lowering pass
-----------------

The pass builds one ``ToyProgram`` per controller. This section goes
through it in the order a program meets it.

State
~~~~~

A ``ToyProgram`` starts from the program's clock grid, with empty state:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: ToyProgram.__init__
   :dedent: 4

It tracks the code so far, the cycle the sequencer has reached, each
oscillator's frame, and a table of waveforms.

Channels
~~~~~~~~

Each line of the export becomes a channel. The lines of a shared line's
qudits become tones of one channel:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: ToyProgram.wire
   :dedent: 4

An event on a shared line gives its tone in ``tone_slot``:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: ToyProgram.channel
   :dedent: 4

Time
~~~~

Time moves only by ``wait``, in timesteps of 4 cycles:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: ToyProgram.wait_until
   :dedent: 4

Every start in the export is already on that grid. A kernel whose
lengths can't be is refused when it compiles, so this only divides.

What the toy sequencer can't play, it refuses. It never plays it some
other way:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: refuse

Frequency and phase
~~~~~~~~~~~~~~~~~~~

The toy oscillator can only be set to a frequency and shifted in phase.
So the pass shifts by the difference from the frame it had:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: set_frame

Waveforms
~~~~~~~~~

Waveforms that differ only in gain share a table entry, which
``shape_key`` names:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: table_entry

Play and acquire
~~~~~~~~~~~~~~~~

A drive becomes a ``play``. A readout becomes an ``acquire`` into the
bit named for its event index:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: play

The event index is the same in every controller's part of a program. So
a bit has one name everywhere.

Decoder calls
~~~~~~~~~~~~~

A call to the decoder becomes a ``submit`` of the bits it reads:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: submit

Its answer becomes an ``answer``, or a ``poll`` when the call has a
deadline:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: answer

Those are all the events the sequencer issues by itself:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :start-at: EMIT: dict
   :end-at: }

Branches
~~~~~~~~

A branch spans events. First, the pass names the bit it tests:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: tested_bit

Each arm becomes code that ends at the same cycle:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: arm

Each arm also ends with the phases it started with. So what follows the
branch doesn't depend on the outcome:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: restore_phases

The two arms then go behind one ``branch``:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: branch_code

Putting it together
~~~~~~~~~~~~~~~~~~~

The pass groups events into units. A unit is an event that always plays,
or a branch with the events of both its arms:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: units

It refuses what the toy can't express, such as a branch inside a branch:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: check

Last, ``lower`` issues the units in time order:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: lower

The program after lowering
--------------------------

Now lower the Ramsey sequence and read the listing.

The head and the Ramsey part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The head names the channels and the waveform table:

.. code-block:: python

   from toy_isa import lower

   toy = lower(program)
   listing = str(toy).splitlines()

   for line in listing[:10]:
       print(line)

.. code-block:: text
   :class: output

   .chan ch0 drive q0
   .chan ch1.0 readout q0
   .wave w0 40
   .wave w1 200
   set_freq ch0 5000000000
   play ch0 w0 0.25
   wait 35
   set_freq ch0 5002000000
   shift_phase ch0 2.827433
   play ch0 w0 0.25

What to notice:

- The three Gaussians are one table entry, ``w0``.
- The second π/2 pulse plays at the detuned frequency.

The readout and the reset
~~~~~~~~~~~~~~~~~~~~~~~~~

The rest of the listing reads the qudit out and resets it:

.. code-block:: python

   for line in listing[10:]:
       print(line)

.. code-block:: text
   :class: output

   wait 10
   set_freq ch1.0 5000000000
   acquire ch1.0 w1 0.05 m4
   wait 70
   branch !m4 end0
   set_freq ch0 5000000000
   shift_phase ch0 0.502655
   play ch0 w0 0.5
   shift_phase ch0 -0.502655
   end0:

What to notice:

- The readout plays on tone 0 of the feedline, ``ch1.0``.
- The branch waits 70 timesteps: the 200-cycle readout, then 80 cycles of
  feedback latency.
- The π pulse shifts its phase by 2π × 2 MHz × 40 ns = 0.502655 rad,
  then shifts it back before the arm ends.

Test your backend
-----------------

CUDA-Q Pulse ships programs for testing a backend. This section runs the
toy pass through them.

Read a listing back
~~~~~~~~~~~~~~~~~~~

To compare a backend with the schedule, give it ``events``. It reads a
listing back into what it plays:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: events

The backend
~~~~~~~~~~~

The backend wraps the pass and the read-back:

.. literalinclude:: ../../examples/isa_lowering/toy_isa.py
   :language: python
   :class: fragment
   :pyobject: ToyBackend
   :end-before: def run(

Its ``run`` plays listings on an emulator of the toy controllers.

Validate it
~~~~~~~~~~~

The test programs compile with the default schedule, which waits out the
toy controller's feedback latency. So validate against the toy controller
itself:

.. code-block:: python

   from cudaq_pulse import backends
   from toy_isa import ToyBackend

   report = backends.validate(ToyBackend())
   lines = str(report).splitlines()
   print(lines[-1])

.. code-block:: text
   :class: output

   backend 'toy': 29 passed, 2 skipped

The test programs cover:

- one program per kind of event
- a branch, and branches on whether a decoder answered in time, with the
  call's timeout and with the function's own
- values known only at run time
- a shared line, and two controllers

Two are skipped. Print them:

.. code-block:: python

   for result in lines[:-1]:
       if not result.startswith("PASS"):
           name = result.split(":")[0]
           print(name)

.. code-block:: text
   :class: output

   SKIP runtime
   SKIP runtime_frequency

Both set a frame from a decoder's answer at run time. The toy
description doesn't support that, so the compiler turns them away.

Check your own programs
~~~~~~~~~~~~~~~~~~~~~~~

Check your own programs against their schedule the same way:

.. code-block:: python

   from toy_isa import events

   played = []
   for event in events(toy):
       played.append((event["kind"], event["line"], event["start_cycles"]))

   expected = []
   for event in scheduled.schedule:
       if event.kind != "wait":
           expected.append((event.kind, event.line, event.start_cycles))

   print("plays as scheduled:", sorted(played) == sorted(expected))

.. code-block:: text
   :class: output

   plays as scheduled: True

Then pin each listing with a golden copy in your tests.

Where to go next
----------------

- :doc:`isa_limits` goes through six common controller limits with this
  pass, from feedback latency to waveform memory.
- :doc:`kernel_to_controller` packages a backend as a distribution that
  CUDA-Q Pulse finds by its entry point.
- :doc:`../user_guide/controller_backends` documents every key of the
  export.
