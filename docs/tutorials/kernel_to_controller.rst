..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

From Kernel to Controller
=========================

A compiled pulse program doesn't depend on any one control system. A
*controller backend* turns it into what a particular controller plays: its
sequencer instructions and waveform memory.

This tutorial is for people who build or program control systems.

What you'll learn:

- what the exported program holds;
- how to check a hardware limit on it in plain Python;
- how to write, register and validate a small backend;
- how one program splits across two controllers.

All the code on this page is also one Python script,
:download:`kernel_to_controller.py <scripts/kernel_to_controller.py>`.

The exported program
--------------------

The export gives a scheduled program as JSON-ready data: every drive,
readout, wait and sync, with its line and its time in clock cycles.

Start with one qudit at 5 GHz and a 1 GHz clock, so one cycle is one
nanosecond, and a kernel that drives it and reads it out:

.. code-block:: python

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

Export one call and list its events:

.. code-block:: python

   scheduled = drive_and_measure.lower(system.qudits[0])
   program = scheduled.export_to_controller()
   print("format:", program["format"])

   print(f"{'kind':8} {'line':>4} {'start':>6} {'length':>6}")
   for event in program["events"]:
       line = str(event["line"])
       start = event["start_cycles"]
       length = event["duration_cycles"]
       print(f"{event['kind']:8} {line:>4} {start:>6} {length:>6}")

.. code-block:: text
   :class: output

   format: cudaq-pulse.controller/1
   kind     line  start length
   drive       0      0     40
   sync     None     40      0
   readout     1     40    200

.. themed-figure:: figures/exported_program.svg
   :alt: Timeline of the exported drive_and_measure on qudit 0's drive and readout lines, in cycles of the 1 GHz clock: a 40-cycle drive from cycle 0, a sync at cycle 40 that ties both lines, and a 200-cycle readout from there.
   :width: 100%

   The same three events on their lines, one cycle per nanosecond.

What to notice:

- the 40 ns drive pulse is 40 cycles;
- the readout starts where ``sync`` put it;
- a sync belongs to no single line.

The format's name carries a version. Within ``/1``, keys and event kinds
may be added but never change meaning, so a backend written today keeps
reading programs from later releases.

Check a limit in Python
-----------------------

The schedule lists every operation with its line, start and end. So a check
the compiler doesn't know about is a short Python function.

Suppose a controller can play at most two drive lines at once. Each drive
adds a step up at its start and a step down at its end, and the running
count is how many play at once:

.. code-block:: python

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

Try it on four π pulses:

.. code-block:: python

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

.. code-block:: text
   :class: output

   (cycle, drives at once): [(0, 3), (0, 4)]

All four π pulses start at cycle 0, two more than the limit.

.. themed-figure:: figures/drive_count.svg
   :alt: The number of flip_four's drives playing at once, as late as possible: 4 from 0 to 40 ns, then none. The limit of 2 is drawn across it.
   :width: 70%

   The running count of drives at once, against the limit.

What to notice: the count jumps straight to four, and the check reports
each step above two.

Let the compiler keep the limit
-------------------------------

The compiler can schedule around such a limit itself when the control
system states it. Use the resource-constrained policy, ``"rcp"``:

.. code-block:: python

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

.. code-block:: text
   :class: output

   violations: []
   pulse starts: [0, 0, 80, 80]

.. themed-figure:: figures/concurrent_drives.svg
   :alt: Two timelines of flip_four on the drive lines of 4 qudits. As late as possible, all 4 pi pulses play at once, from 0 ns. With the resource-constrained policy and at most 2 drives at once, 2 play from 0 ns and the others from 40 ns.
   :width: 100%

   ``flip_four`` as late as possible, and with ``schedule="rcp"`` under a
   limit of two drives.

What to notice: two pulses play first and two follow, so the limit holds.

A small backend
---------------

A backend is an object with a ``name`` and three methods:

.. list-table::
   :header-rows: 1

   * - Method
     - What it does
   * - ``control_system``
     - says what its controller can play
   * - ``emit``
     - turns an exported program into the controller's own form
   * - ``run``
     - plays it

This one emits a line of text per drive and readout. It is
:example:`controller_backend/text_backend.py`, shown a piece at a time.

It needs the controller interface and the export format's name:

.. literalinclude:: ../../examples/controller_backend/text_backend.py
   :language: python
   :start-after: # [start imports]
   :end-before: # [end imports]

Each drive and readout becomes one line of text:

.. literalinclude:: ../../examples/controller_backend/text_backend.py
   :language: python
   :start-after: # [start describe]
   :end-before: # [end describe]

The class states the controller's limits:

.. literalinclude:: ../../examples/controller_backend/text_backend.py
   :language: python
   :class: continues
   :start-after: # [start class]
   :end-before: # [end class]

Those limits are a 1 GHz clock, one instruction every 2 cycles, and idle
gaps of at least 2 cycles. Compiling against them checks a program before
it reaches the backend.

``emit`` checks the format, skips what takes no instruction, and refuses
what it can't play:

.. literalinclude:: ../../examples/controller_backend/text_backend.py
   :language: python
   :class: continues
   :start-after: # [start emit]
   :end-before: # [end emit]

What to notice: a pulse inside a branch is refused with
``IncompatibleBackendError``, instead of being played as something else.

``run`` plays the emitted text once per shot:

.. literalinclude:: ../../examples/controller_backend/text_backend.py
   :language: python
   :start-after: # [start run]
   :end-before: # [end run]

Register it and run a program
-----------------------------

An installed backend is found by its entry point. ``register`` adds one to
the running process instead:

.. code-block:: python

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

.. code-block:: text
   :class: output

   0 drive line 0 gaussian at 5e+09 Hz
   40 readout line 1 square at 5e+09 Hz

Validate it
-----------

``backends.validate`` runs a corpus of test programs through a backend, one
for each kind of event. Each must be either emitted or refused cleanly:

.. code-block:: python

   report = backends.validate(
       "text",
       controller=Controller(clock_ghz=1.0),
   )
   print("ok:", report.ok)
   print(str(report).splitlines()[-1])
   backends.unregister("text")

.. code-block:: text
   :class: output

   ok: True
   backend 'text': 12 passed, 8 refused

What the text backend does with the corpus:

- it plays drives, readouts, waits, syncs and loops;
- it refuses classical calls, frame updates, branches and values known
  only at run time.

.. themed-figure:: figures/validation_corpus.svg
   :alt: The 20 programs of the validation corpus and what the text backend does with each: play, passed; readout, passed; wait, passed; sync, passed; loop, passed; call, refused; answer, refused; frame_update, refused; branch, refused; answer_timeout, refused; timeout_function, refused; runtime, refused; runtime_frequency, refused; shared_line, passed; shared_drive_line_two_tones, passed; shared_drive_line_one_tone, passed; several_controllers[0], passed; several_controllers[1], passed; unsupported_event, passed; newer_format, passed. 12 pass and 8 are refused cleanly; none fail.
   :width: 100%

   Each program of the corpus, and what the text backend did with it.

What to notice: there is no third mark. Every program either passes or is
refused, and a program the backend can't read passes by being turned away.

Refusing is a pass. What fails validation is a backend that emits a program
it can't play faithfully, or reads a format it doesn't know.

One program, two controllers
----------------------------

On a control system of several controllers, each controller gets the
events of its own lines, in its own cycles, from
``export_to_controller(controller=i)``.

In :example:`23_define_a_qpu.py`, one controller reads D1 out, and another
plays the correction on D3 that a function on the feedback link decides:

.. themed-figure:: figures/two_controllers.svg
   :alt: Timeline of one program on two controllers. Controller 0 reads D1 out; the outcome goes over the feedback link, 40 ns on the link, at most 400 ns in the correction function and 20 ns of feedback, to controller 1, which plays a pulse on D3 only if the qubit read 1, at 561 ns. A sync ties the two controllers' lines at the end of the readout.
   :width: 100%

   The two exports of one program. The dashed pulse plays only when the
   answer says to flip.

What to notice: the sync at the end of the readout ties a line of each
controller together.

Where to go next
----------------

- :example:`controller_backend/toy_controller/__init__.py` is a toy backend
  as an installable distribution, with its entry point in
  :example:`controller_backend/pyproject.toml`.
- :example:`custom_passes/python_pass.py` is the concurrent-drive check
  above, as a script.
- :example:`transmon/transmon_surface17_multiplexed_readout.py` reads four
  qudits through one feedline and shows each readout's tone in the export.
- :doc:`isa_lowering` (experimental) lowers the export to a controller's own
  instructions, and :doc:`isa_limits` shows what the compiler does about
  feedback latency, clocks across controllers, instruction and waveform
  memory, and decoder calls.

For the details:

- :doc:`../user_guide/controller_backends` documents every key of the
  export format, clocks, shared lines and waveform memory;
- :doc:`../user_guide/custom_passes` shows how to load your own MLIR passes
  into ``cudaq-pulse-opt``.
