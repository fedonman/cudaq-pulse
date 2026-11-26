..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

MLIR Passes
===========

CUDA-Q Pulse implements its optimizations, checks and schedules as C++ MLIR
passes on the in-memory ``mlir::ModuleOp``. Each is registered under the
name below, so ``cudaq-pulse-opt`` runs it on its own:

.. code-block:: bash

   cudaq-pulse-opt --pulse-fold-constants --pulse-fold-phases kernel.mlir
   cudaq-pulse-opt '--pulse-schedule-rcp=max-drives=2 alap=true' kernel.mlir

``pulse.compile`` runs the passes of ``passes=``, members of
:class:`cudaq_pulse.Pass` or their names, in the order given, then the
schedule of its policy. Each member is one pass:

=============================  =================================
``Pass`` member                Pass
=============================  =================================
``VERIFY``                     ``pulse-verify``
``FOLD_CONSTANTS``             ``pulse-fold-constants``
``REMOVE_UNUSED``              ``pulse-remove-unused``
``MERGE_WAITS``                ``pulse-merge-waits``
``SIMPLIFY_CONTROL_FLOW``      ``pulse-simplify-control-flow``
``FOLD_PHASES``                ``pulse-fold-phases``
``FUSE_PULSES``                ``pulse-fuse-pulses``
``HOIST_LOOP_INVARIANTS``      ``pulse-hoist-loop-invariants``
=============================  =================================

The default, ``DEFAULT_PASSES``, is ``VERIFY``, ``FOLD_CONSTANTS``,
``REMOVE_UNUSED``, ``FOLD_PHASES`` and ``FUSE_PULSES``.

Idle replacement (:ref:`idle replacement <idle-replacement>`) is no pass
of ``cudaq-pulse-opt``: the idle windows exist only once the program is
scheduled, and what replaces them is a kernel, traced like any other.
``compiled.replace_idle(rule)`` compiles the program, reads its schedule,
traces the kernel again with the fills written at the statements around
each gap, and compiles that. To read the schedule per traced operation, its
trace writes the number of each drive, readout, wait, sync and frame change
into the packed buffer right after the operation (a site record, which a
trace without idle replacement never writes), and the module is built with
each number as the operation's location, which no pass reads and the
printed module does not show; the numbered program compiles as the program
without numbers. An operation written into a window also carries the
attribute ``pulse.written`` until the numbers are removed:
``pulse-merge-waits`` does not merge a written wait and
``pulse-fuse-pulses`` does not fuse a written drive with one of the
kernel's (two written drives fuse into a written one), so the kernel's
operations stay the ones the two schedules are compared by.

Optimization passes
-------------------

All but ``pulse-verify`` and ``pulse-hoist-loop-invariants`` are pattern
passes: each is one greedy rewrite of a function. Pattern passes in a row in
``passes=`` run as a single rewrite with all their patterns, to the point
where none applies, so splitting the work into passes costs no extra walk of
the module. A greedy rewrite removes the operations whose results nothing
uses as it goes, so each pattern pass also removes those its rewrites leave
unused.

pulse-fold-constants
   Applies the folders of the ops (a frame change by zero, a scale by one, a
   double negation, constant arithmetic) and merges equal constants.

pulse-remove-unused
   Removes operations whose results nothing uses, unreachable blocks and
   unused block arguments.

pulse-merge-waits
   Merges waits in a row on one line, each of a constant length and neither
   variable, into one. It takes only an unscheduled function (no
   ``pulse.schedule_state``): a merged wait would leave the schedule's times
   behind.

pulse-simplify-control-flow
   Removes a ``pulse.sync`` of one line and a sync of the lines another sync
   just produced; inlines a ``pulse.schedule_block`` with no drives, readouts
   or waits; branches on ``pulse.is_set``; and inlines or removes ``scf.for``
   loops that run once or never.

pulse-fold-phases
   Folds a ``pulse.shift_phase`` by a constant into the ``frame_phase_offset``
   of the drive that follows on its tone, after moving it past an
   ``scf.if`` whose arms both pass it through unchanged, into a schedule
   block that takes the tone next, or out of the end of a block. A frame
   rotation then costs no operation of its own.

pulse-fuse-pulses
   Merges two drives in a row on one line and tone, whose waveforms are
   square pulses of the same amplitude and of constant durations, with the
   same phase offset and no frame change folded into the second, into one
   drive of the summed duration. A wait for an answer or for a blocking
   classical call between them keeps them apart.

pulse-hoist-loop-invariants
   Moves waveform construction and other operations that compute the same
   value in every iteration out of ``scf.for`` bodies (MLIR's loop-invariant
   code motion).

When a call binds a kernel's ``float`` parameters, the module with the
values substituted is folded with MLIR's canonicalization patterns, and
``pulse-fold-phases`` and ``pulse-fuse-pulses`` run again, if the module ran
them, on the frame changes and square pulses the values reach: a phase or an
amplitude that was a parameter folds or fuses once it is a number.

Measurement choices
-------------------

pulse-lower-selects
   Lowers the ``arith.select`` a measurement-conditioned ``if`` leaves on a
   name its arms set differently. ``pulse.compile`` runs it before
   ``passes=``, only on programs that have one. Its ``runtime`` option lists
   the parameters the quantum system takes at run time (``amplitude``, ``phase``,
   ``frequency``): a use that sets one of those (a waveform's amplitude, the
   factor of ``pulse.scale``, a frame's phase or frequency), through any
   arithmetic, reads the select, so every use shares the one value. Any
   other use (a duration, a waveform's shape, a waveform chosen whole) has
   the first op with effects it reaches copied into both arms of an
   ``scf.if`` on the select's condition, each reading that arm's value; an
   ``elif`` select is copied after the one it is nested in, into the arm
   that reads it. The function is marked ``pulse.selected_values``, so a
   schedule starts an op that reads a selected value, kept or in the arms,
   after the measurement. The compiler chooses ``runtime`` per quantum system, and
   today no quantum system takes a value at run time.

Checks
------

pulse-verify
   Checks that waveform durations are positive, that each drive and readout
   line is consumed once at each step, and, on a scheduled module, that the
   intervals on each physical line neither overlap nor move backwards. In a
   function marked ``timed`` or ``placed`` (``pulse.schedule_state``), every
   timed operation that runs needs its start and duration.

pulse-classical-check
   On a scheduled module that declares classical calls: checks slot sizes,
   deadlines, answer timing, link capacity and the order of calls on each
   link. Scheduling such a module runs it, also for each call's values;
   simulation does not.

Schedules
---------

Each schedule assigns ``start_cycles`` and ``duration_cycles`` to every timed
operation, respecting data dependencies, syncs, the order of pulses on each
tone and the policies of schedule blocks, then moves the ops of each delayed
line by its delay and marks the function ``pulse.schedule_state``:
``placed`` when a delay moved some op, ``timed`` otherwise. A wait for an answer or for a
blocking call holds back only the lines of its timeline, which it opens a new
segment for; see :doc:`../user_guide/classical_calls`.

pulse-schedule-asap, pulse-schedule-alap
   Place each operation as early, or as late, as its dependencies allow; a
   readout's outcome is usable ``readout-latency`` cycles after it ends
   (default 0; on several controllers, the module's table). The line is
   not held for it.

pulse-schedule-rcp
   Resource-constrained: at most ``max-drives`` drives and ``max-readouts``
   readouts play at once, a readout's line and result are ready
   ``readout-latency`` cycles after it ends, and a unit stays busy
   ``switch-penalty`` cycles after each operation. ``alap=true`` gives the
   ``alap_rcp`` policy. ``pulse.compile`` takes these from the ``ControlSystem``.

pulse-inline-schedule-blocks
   Inlines every ``pulse.schedule_block``, ignoring its policy
   (``flatten_schedules=True``).

pulse-unroll-classical-loops
   Fully unrolls loops that hold classical calls; simulation runs it on its
   copy of the module before scheduling it.

Lowering
--------

``pulse-to-qop``, ``qop-to-cudm`` and ``cudm-to-llvm`` lower a scheduled
module for GPU execution; see :doc:`pipeline`.
