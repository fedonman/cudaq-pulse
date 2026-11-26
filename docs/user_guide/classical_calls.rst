..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Classical Calls at QPU Runtime
==============================

``classical`` is a classical real-time dialect for co-processing at QPU
runtime. A pulse program calls GPU, host or FPGA services while the QPU runs:
a decoder turns syndromes into corrections, an estimator turns outcomes into
the next pulse parameters, a calibration service updates a frequency. The
answer is used later in the same shot, to set an amplitude, a phase, a
frequency, the length of a wait or the branch taken.

A call is two timed events. The request is *submitted* on a link, and the
answer is *taken* where the program needs it. The scheduler budgets the
function's worst-case latency between them, so pulses that depend on the
answer start after it is due and pulses that do not are not delayed. It
reports, for every call, how much slack the answer has.

``examples/19_classical_calls.py`` uses everything on this page end to end,
and ``examples/20_adaptive_calibration.py`` calibrates against a drifting
quantum system with the estimators described under "Calibrating while the QPU runs";
see :doc:`../examples`.

Declaring a function
--------------------

``@pulse.classical`` declares a function from its body: the type hints of its
parameters and result say what it takes and returns, and the options give
its latency and the link its requests travel on. The body is the reference
implementation: simulation and emulation run it in place of the service.

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.classical(
       min_latency_ns=3000,
       nominal_latency_ns=3500,
       max_latency_ns=4500,
       link="gpu",
   )
   def decode(syndrome: pulse.Bits) -> bool:
       return bool(syndrome[0])

A parameter is one of:

- ``pulse.Bits``: measurement results read as bits, as in ``decode.submit([m])``;
- ``list[int]``: measurement results read as levels (``0``, ``1``, ``2``, ...
  for a qudit), or a list of integers;
- ``list[complex]``: the IQ points of readouts, or ``complex`` for one, with I
  as the real part and Q as the imaginary part;
- ``float``, ``int`` or ``list[float]``.

A result is ``float``, ``int``, ``bool``, ``pulse.Bits``, ``list[float]``,
``list[int]``, a tuple of them, or ``None``. A ``bool`` or ``pulse.Bits``
result gives values a branch can test. NaN values in classical data pass
through as they are and are not rejected: a ``float`` argument or answer may
be NaN, as an estimator's state is before it has a cost (see "Calibrating
while the QPU runs").

**List lengths.** A list parameter takes its length from the call. A
function called with lists of two lengths has two *versions*, each with its
own schema, named by its lengths: ``flip.submit([m0, m1, m2])`` and
``flip.submit([m0])`` call ``flip__3`` and ``flip__1``. A call record or
report row gives both names: ``call.function`` is ``"flip"`` and
``call.compiled_name`` is ``"flip__3"``, and the call manifest lists every
version with its dispatch key and layout. The length of a list result is read from
the body, for each version:

.. code-block:: python
   :class: fragment

   @pulse.classical(max_latency_ns=100)
   def flip(outcomes: pulse.Bits) -> pulse.Bits:
       return [b ^ 1 for b in outcomes]  # flip__3 answers with 3 bits

Lengths are read from the function's bytecode, not its source: from
slices, concatenation, comprehensions, ``for`` loops, ``while`` loops the
lengths end, branches the lengths decide, a table of equal-length lists, and
counts that cancel (the set bits of a ``pulse.Bits`` and the items a
comprehension keeps, padded back to an input's length), and through the
plain Python functions the body calls (a function that calls itself is not
followed). Where the values
decide it (a comprehension with a condition, a ``while`` loop the values
end, two returns of different lengths), compiling says so and names the
line; return a fixed maximum and a count instead (``positions = [0] *
len(syndrome)``, fill the first ``count`` items, ``return count,
positions``). ``flip.specialize(outcomes=3)`` is the version of one length,
for ``wire.schema``, ``wire.native`` or ``processor.usage``.

**Fixed lengths.** Where one size is wanted, the signature states it with
``pulse.Length`` (``typing.Annotated``, so checkers read the type inside
it); the function then has one version, named as declared:

.. code-block:: python
   :class: fragment

   @pulse.classical(max_latency_ns=4500, link="gpu")
   def frozen(
       syndromes: pulse.Length[pulse.Bits, 24],
       counts: pulse.Length[list[int], 4],
   ) -> pulse.Length[pulse.Bits, 4]:
       return [syndromes[i] ^ syndromes[i + 1] for i in range(4)]

``pulse.Length[pulse.Bits, 24]`` takes 24 bits, and a call with another
length is a compile error that names the size: ``frozen.submit():
'syndromes' takes 24 bits (pulse.Length[pulse.Bits, 24]), got 23``. In a
fixed-size signature ``pulse.Bits`` takes measurement outcomes and
``list[int]`` plain numbers. A length is a positive whole number; a list
result without one takes its length from the body, per version. Functions
whose lengths come from their calls need none of this.

**Dispatch keys.** Each request carries the key its service is registered
under: the 32-bit FNV-1a hash of the compiled name
(``classical.dispatch_key("flip__3")``), the hash CUDA-Q Realtime registers
handlers by. ``call.dispatch_key`` gives it next to ``call.compiled_name``,
and the call manifest's ``"dispatch_key"``. A controller whose firmware
fixes the keys of the functions it runs lists them by name in its
description's ``dispatch_keys``; every function on its links must be
listed, with a fixed size, and the manifest marks each such entry with
``"dispatch_key_fixed_by"``. Two names of one program that hash to one key
on one link are a compile error that asks to rename one; the keys of
CUDA-Q QEC's decoding-server messages are reserved for its decoder calls
(below).

The hints are read when a program that calls the function is first
compiled, and a mistake in them is raised there, naming the function. The
body is a ``def``: a lambda has no hints.

**External functions.** ``pulse.classical.external`` declares a function
that is served elsewhere. Its stub is a declaration: the hints give its
schema, and the body, which never runs, is a placeholder for its result
that gives the result's shape:

.. code-block:: python
   :class: fragment

   @pulse.classical.external(max_latency_ns=50, link="feedback")
   def table(index: pulse.Length[pulse.Bits, 3]) -> float:
       return float()

   @pulse.classical.external(max_latency_ns=4500, link="gpu")
   def decode(syndrome: pulse.Bits) -> tuple[float, pulse.Bits]:
       return (float(), syndrome[:2])

   @pulse.classical.external(max_latency_ns=4500, link="gpu")
   def echo(outcomes: pulse.Bits) -> pulse.Bits:
       return outcomes

A placeholder is a value of the result's type: ``float()``, ``int()``,
``pulse.Bits()`` (an empty list, for a result whose length the signature
states), an input or a slice of one. A ``None`` result needs no return.
mypy and pyright check these bodies like any other. A list parameter of no
stated length works as for ``pulse.classical``: ``decode`` called with 6 and
with 8 bits is two versions, ``decode__6`` and ``decode__8``, each served
under its own id and listed in the manifest with its layout. A service that
takes one size is declared with it, as ``table`` is. Calling an external
function in Python is an error that says to give it a handler.

The latency has a minimum, a nominal value and a maximum. The scheduler
budgets the maximum. Simulation and emulation use a triangular distribution
over ``[min, max]`` with its mode at the nominal value. ``latency_mode``
chooses what happens when an answer could be late: ``"bounded"`` (the
default) trusts the maximum, ``"timeout"`` gives up after ``timeout_ns``, and
``"blocking"`` stops the lines that wait for the answer until it arrives
(see the barriers below). ``submit="lazy"``
sends a request as late as the budget allows rather than as soon as its
arguments are ready, and ``deadline_ns`` bounds how long after its submission
an answer may be due. ``reaction_deadline_ns`` bounds how long after the
readouts a request carries its answer may be due: the whole reaction, from the end of
the measurement through queueing, the link and ``max_latency_ns`` to the
answer being usable. Compiling fails if any execution of a call exceeds it,
with the iteration, the slack and what to change.

.. code-block:: python

   @pulse.classical(max_latency_ns=800, reaction_deadline_ns=1500, link="feedback")
   def correct(syndrome: pulse.Length[pulse.Bits, 8]) -> pulse.Length[pulse.Bits, 8]:
       ...

A call that carries no readout reacts from its submission. Waits between the
readouts and the submission whose length is known only when the program runs
count at their highest length.

Outside a kernel, calling the function runs its body, and ``fn.submit(...)``
returns an answer whose ``result()`` is ready at once. Between-shot code can
use the same functions as kernels.

Links
-----

A link is a path to a classical service with a cost: a fixed time per
request, a bandwidth, a feedback time from an answer arriving to a pulse
using it, a number of requests it holds at once and the largest request it
takes. They belong to the control system, by name:

.. code-block:: python
   :class: fragment

   from cudaq_pulse.classical import Link

   control_system = pulse.ControlSystem(
       links={
           "gpu": Link(fixed_ns=100, bytes_per_ns=10, feedback_ns=50),
           "controller": Link(fixed_ns=4, feedback_ns=8, slots=4),
       }
   )
   scheduled = pulse.compile(kernel, quantum_system=qs, control_system=control_system).lower(q)

A function's ``link=`` names a link of the control system, or gives a link
id: the link's position, so the first is link 0, the default. A link id the
system does not list gets ``Link()``; compiling a function that names a link
the control system does not have is an error. A control system without
controllers, as here, states only its links and lanes: nothing checks what a
controller can play.

Calling from a kernel
---------------------

In a kernel, ``fn.submit(...)`` sends the request and returns a future, and
``future.result()`` takes the answer:

.. code-block:: python
   :class: fragment

   @pulse.kernel
   def feed_forward(q):
       d = get_drive_line(q)
       r = get_readout_line(q)
       readout_time = 500 * pulse.ns
       decode_deadline = 4 * pulse.us
       pi_time = 20 * pulse.ns
       pi_amplitude = 0.16  # square(pi_time, pi_amplitude) is a pi pulse
       m = readout(r, square(readout_time, 0.05))
       in_time, correction = decode.submit([m]).result(timeout=decode_deadline)
       sync(d, r)
       if in_time:
           if correction:
               drive(d, square(pi_time, pi_amplitude))
       else:
           if m:
               drive(d, square(pi_time, pi_amplitude))

With ``timeout`` the result starts with a flag that is true when the
answer came in time; the other values are zero when it did not, and the
program takes its fallback. ``timeout`` is a duration like any other: an
``int`` number of program cycles, or a time such as ``4 * pulse.us``,
which converts once when the kernel compiles. A future is taken in the loop iteration that
submitted it. It may be passed to a kernel the program calls, or returned
from one, and taken there; an answer taken inside a kernel called with its
own ``schedule=`` is used only inside it, so return the future instead when
the caller needs the answer. Answers may be carried from one iteration to
the next, as an estimator's state is.

A wait may take its length from an answer, between bounds. An answer is a
plain number, a number of cycles; multiply it by a unit for an answer in
nanoseconds:

.. code-block:: python
   :class: fragment

   shortest_wait = 100 * pulse.ns
   longest_wait = 20 * pulse.us
   alpha, beta, tau = update_t1.submit([m], alpha, beta, tau).result()
   wait(d, tau * pulse.ns, bounds=(shortest_wait, longest_wait))

Such a variable wait is a barrier for the lines that depend on it, and it
opens a new *segment* for them. The static schedule places it at its lower
bound, and what follows it on those lines shifts by the wait's actual length,
which is clamped to the bounds. ``ScheduledKernel.segments()`` lists the segments. A
variable wait may not be inside a ``schedule_block`` or a branch; a blocking
wait is a barrier in the same way.

A barrier holds back the lines of its *timeline*: lines share one when one
waits for the other, through a ``sync`` naming both, a measurement or an
answer used on both, a branch on one around the other, a tone or a value
carried between them, or a kernel with its own ``schedule=`` that plays on
both. They sync before the barrier; lines on other timelines keep their time,
so qubits that wait on answers of their own run side by side. Timelines also
merge where their schedules could not move independently: when a link has
fewer slots than the timelines that use it, under ``rcp`` and ``alap_rcp``
when there are more lines than lanes, and for a blocking wait whose answer
nothing uses, which holds back every line. Timelines that share a link each
get a share of its slots and budget each request as sent after the others',
so a request on a shared link is due a few cycles later than alone.

A length may also add ``float`` kernel parameters to an answer, as in
``wait(d, (answer + tau) * pulse.ns, bounds=(4 * pulse.ns, 500 * pulse.ns))``. The kernel still compiles once:
``tau`` is taken when the kernel is called, and each call checks that the
length with its answers taken as 0 (here ``tau``) is finite and at most the
upper bound, raising ``ValueError`` otherwise; it may be below the lower
bound, since the answer adds to it. The wait stays a barrier, and its whole
length is clamped to the bounds at run time.

Scheduling and the report
-------------------------

Every schedule (``asap``, ``alap``, ``rcp``, ``alap_rcp``) places calls the
same way:

- A request is submitted once its arguments are ready (eager, the default)
  or as late as the budget allows (lazy). It holds its link for
  ``fixed_ns + bytes / bytes_per_ns``, and a link holds ``slots``
  requests at once: one more waits until a slot is free, so a program never
  fails for having too many requests in flight.
- Its answer is due after the budget: the maximum latency (or the timeout),
  plus the link's feedback time.
- An operation whose amplitude, phase, frequency, duration or branch
  condition comes from the answer starts no earlier than that.
- A kernel called with its own ``schedule=`` keeps its calls where its own
  schedule puts them, eager submissions included, and moves with them: it
  is placed where its requests fit the slots its caller's requests leave.
- A blocking wait, like a variable wait, starts once every line of its
  timeline is done.

``ScheduledKernel.classical_report()`` gives one ``ClassicalCall`` per execution of a
call, in clock cycles: when it is submitted (``submit_cycles``), when its answer
is due (``due_cycles``) and taken, the first operation that uses it, its slack
(``slack_cycles``, how much later the answer could arrive without delaying a
pulse) and, with ``deadline_ns``, how far inside its deadline it is.
``exposed_cycles`` is how long the first use waits for the answer beyond when it
could start anyway: after its line is free and after the readout resonator
has depleted (``PhysicalQudit.depletion_ns``, from its ``DispersiveReadout``'s
``depletion_ns`` or ``linewidth_hz``). The rest of the budget, ``hidden_cycles``, overlaps
work that had to happen anyway, so a latency within the depletion time costs
nothing. The
timeline (``ScheduledKernel.timeline()``) draws each link as a row of its own,
labelled with its name.

.. themed-figure:: figures/call_submit_take.svg
   :alt: Timeline of feed_forward on a drive line, a readout line and the gpu link. The readout ends at 500 ns and decode is submitted then; the request holds the link for 102.5 ns, the answer has a timeout of 4000 ns and 50 ns of feedback, and it is due at 4652.5 ns, 4152.5 ns after its submission, where the correction pulse starts, dashed because it plays only on some outcomes. A second panel shows 4502.5 to 4722.5 ns closer: the end of the timeout, the feedback and the pulse.
   :width: 100%

   ``feed_forward`` of `Calling from a kernel`_, compiled with the ``gpu``
   link of `Links`_. The request is submitted when the readout ends and
   holds the link; the answer is due after the timeout and the link's
   feedback time, and the correction, dashed because it plays only on some
   outcomes, starts then. The time at the right of the link's row is from
   the submission to the answer. The lower panel shows the shaded end of
   the upper one.

After scheduling, ``pulse-classical-check`` verifies that every request and
answer fits its link's slots, that deadlines and reaction times hold, that no
answer is taken before it is due, that no link holds more requests than
``slots``, and that the submissions on a link go out one after
another, in program order.

Simulation
----------

``pulse.simulate`` runs a program with calls per measurement path. When a
call is due, its handler runs on the path's outcomes and numbers, the answer
is bound, and the path is scheduled again, so variable waits take their
actual lengths on each path. Branches share the state rather than copying
it.

.. code-block:: python
   :class: fragment

   result = pulse.simulate(scheduled, initial_state=plus, latency_branching=True)
   for branch in result.branches:
       print(branch.levels, branch.probability, branch.answers("decode"))

- ``handlers={"decode": fn}`` replaces a function's body in every version
  of it, or provides one for a function declared with
  ``pulse.classical.external``; a handler named by a version, such as
  ``"flip__3"``, serves that version only. Each answer's list lengths are
  checked every time a body or handler runs, in simulation and emulation
  (compiled controller code is not checked):
  ``decode_surface returned 3 bits; its result is pulse.Length[pulse.Bits,
  2]``.
- ``latency_branching=True`` branches a timeout on whether the answer came
  in time, with the probability of the latency model; with ``shots_count`` that
  outcome is sampled.
- ``readout="model"`` reports outcomes through each qubit's assignment
  errors, and ``"model_with_decay"`` adds relaxation during the readout.
  Calls and branches see the reported outcome; the state is projected on the
  true one.
- A function that takes IQ points (``complex`` or ``list[complex]``) gets
  them synthesized per shot from a
  dispersive readout model, at the true level of each readout, from the
  frequency of its tone and the amplitude of its pulse. It needs ``shots_count``.
  The model reads each qubit's ``DispersiveReadout`` and its ``READOUT``
  calibration (a ``ReadoutPulse``): it puts the resonator at the readout's
  ``frequency_hz`` and scales the points by the pulse's amplitude over the
  calibration's ``amplitude``; without the first every tone is on
  resonance, and without the second the amplitude does not change the
  points. A tone frequency or amplitude it
  needs and cannot compute where the readout plays (from constants, float
  arguments and earlier answers; an amplitude only of a built-in waveform)
  is ``UnsupportedError``.
- Each of ``result.branches`` lists its path's ``calls``: the function's
  name and its version's (``function``, ``compiled_name``), inputs, answer,
  and whether it came in time; ``branch.answers("decode")`` gives the
  answers of one function, in call order, and
  ``result.postselect(lambda b: b.answers("decode")[-1])`` the result over
  the branches whose last answer is true, with the fraction it kept in
  ``acceptance``.

A whole program with unanswered calls does not lower to LLVM:
``ScheduledKernel.mlir("llvm")`` points to ``pulse.simulate``, which lowers the
program one interval between answers at a time, with the answers bound, on the
CPU or the GPU.

``quantum_system=`` simulates another quantum system than the one the program
was compiled for, keeping the program as compiled: its qudits' frequencies,
T1, T2, drive strength and readout parameters apply. A program compiled from
an estimate is simulated this way on a quantum system that drifts (see
below).

Native handlers
---------------

``cudaq_pulse.classical.wire`` encodes requests and answers in the CUDA-Q
real-time format: a 24-byte header, then the arguments packed by type.
``wire.native(library, symbol, fn)`` makes a handler from a C function with
the CUDA-Q host-call signature, ``void handler(const void *request, void
*answer, size_t slot_bytes)``, so the same compiled handler serves
simulation, emulation and a real dispatcher.

.. _cudaq-qec-decoders:

CUDA-Q QEC decoders
-------------------

Decoders are defined in CUDA-Q QEC (``cudaq_qec``), and a pulse program
calls them as a CUDA-Q kernel does: ``pulse.classical`` takes CUDA-Q QEC's
real-time decoding functions themselves, with a link and a latency budget.
A configured decoder answers in simulation and emulation, and on hardware
the requests are the messages of CUDA-Q QEC's decoding server.

.. code-block:: python
   :class: fragment

   import cudaq_qec

   cudaq_qec.configure_decoders_from_file("decoders.yaml")   # decoder 0

   enqueue_syndromes = pulse.classical(cudaq_qec.qec.enqueue_syndromes, link="gpu")
   get_corrections = pulse.classical(cudaq_qec.qec.get_corrections, link="gpu",
                                     max_latency_ns=75_000)

   @pulse.kernel
   def qec_round(data, ancillas, deadline_ns: int):
       s0 = readout(get_readout_line(ancillas[0]), square(400 * pulse.ns, 0.05))
       s1 = readout(get_readout_line(ancillas[1]), square(400 * pulse.ns, 0.05))
       enqueue_syndromes.submit(0, [s0, s1], 0)       # decoder 0, round 0
       in_time, fix = get_corrections.submit(0, 3, False).result(timeout=deadline_ns * pulse.ns)
       ...

The calls take CUDA-Q QEC's arguments:
``enqueue_syndromes(decoder_id, syndromes, tag)`` sends one round's
measurements and answers nothing; ``get_corrections(decoder_id,
return_size, reset)`` answers ``return_size`` correction bits, and with
``reset`` clears them once read; ``reset_decoder(decoder_id)`` starts the
decoder over. The decoder id and ``return_size`` must be known when the
kernel is compiled (a number, or a kernel argument given to ``lower``):
each message's dispatch key is shared by all decoders, which the server
tells apart by the id, and every pair is one version of the call
(``get_corrections__0_3``: decoder 0, three bits). The calls without an
answer need no ``max_latency_ns`` (it is 0 unless given); the other
options work as a classical function's do. ``cudaq_qec`` is imported only
when such a call has to be answered, and CUDA-Q Pulse takes these functions
from cudaq-qec 0.6.0 to 0.8.1.

Simulation and emulation answer with the decoders the process configured
(``cudaq_qec.configure_decoders``, ``configure_decoders_from_file`` or
``configure_decoders_from_str``), through CUDA-Q QEC's simulation library,
as a CUDA-Q kernel on a simulator does: each enqueued round is decoded and
the corrections accumulate until a reset. One difference: where nothing was
enqueued since the decoder was reset (by ``reset_decoder`` or a get with
``reset``), the get answers NOT_READY, as the decoding server does, rather
than the library's zeros; a wait with a timeout takes it as late. Each path
in simulation, and each shot in emulation, starts from a reset decoder and
replays its own earlier calls of it; ``serve_udp`` and a call outside a
kernel use the library's one state. A handler given for a call
(``handlers={"get_corrections": fn}``) answers instead, taking the
message's fields: ``(decoder_id, tag, syndrome_mapping_id, count,
syndromes)``, ``(decoder_id, return_size, reset)`` or ``(decoder_id,)``.

CUDA-Q QEC installs on Linux (x86_64 and aarch64) with Python 3.11 to 3.13;
``nv-qldpc-decoder`` also needs an NVIDIA GPU. Installed next to
``cudaq-pulse[cu12]``, pip settles on cudaq-qec 0.6.0, the release whose
CUDA-Q brings the cuDensityMat the GPU extra pins; next to plain
``cudaq-pulse`` it takes the latest.

Emulation
---------

``ScheduledKernel.export_to_emulator()`` writes what a controller or an emulator runs
the program from, as JSON-ready data: the static schedule, each event's
segment, the segments, the calls, the links and the call manifest (every
function's dispatch key, link, payload sizes, latency and slots).

``cudaq_pulse.classical.emulation.emulate`` runs the program shot by shot in
virtual time without physics. Outcomes come from a source you give; each
call goes out as real wire bytes, in process or over UDP to a server from
``serve_udp``; and its answer arrives after a modelled transport delay in
each direction plus the service's measured time. It is in time if it arrives
by the time the compiled program allows: the link's feedback time before it
is due (``classical_report()``'s ``due_cycles``). A late answer takes the
timeout arm, as does a decoder's ``NOT_READY`` (status 4, corrections not
decoded yet); a call without a timeout that gets ``NOT_READY`` is an error.
``late_fraction`` is the fraction of the answers the program waits on that
came late; calls it only sends, such as enqueues of syndromes, do not count.
``transport`` is one ``Transport`` for every link, or one per link by name
or id; a name the control system has no link for is an ``InvalidValueError``.

.. code-block:: python
   :class: fragment

   from cudaq_pulse.classical import emulation

   run = emulation.emulate(
       scheduled,
       shots_count=200,
       transport={"gpu": emulation.Transport(fixed_ns=1500, jitter_ns=300)},
       seed=1,
   )
   print(run.late_fraction, run.calls_per_second)

Calibrating while the QPU runs
------------------------------

``cudaq_pulse.classical.calibration`` holds estimators whose state goes in and
comes back out, so the same function is the body of a classical function a
kernel calls between measurements and runs between shots:

- ``t1_update`` and ``t1_wait``: a gamma posterior over the decay rate,
  updated from one "excite, wait, measure" shot, and the next wait.
- ``frequency_update`` and ``ramsey_setting``: a Gaussian posterior over the
  detuning, updated from one Ramsey shot, and the next wait and phase. The
  estimators' Ramsey shot reports 1 with probability
  ``(1 - cos(2 pi d t + phase)) / 2``; two pi/2 pulses with the second's
  tone shifted by ``pi - phase`` give it.
- ``three_point_decay`` and ``three_point_phase``: a decay time and a phase
  error from three points, whatever the amplitude and offset.
- ``golden_section_step`` and ``nelder_mead_step``: ask-and-tell searches;
  each step takes the cost at the point the last one asked for and returns
  the next point and whether the search has converged. Their states pack
  into a ``list[float]`` for calls.

Every time an estimator takes (a wait, a spacing, a coherence time) is a
``pulse.Time`` or a number of ns: ``cal.ramsey_detuning(mean_hz, theta,
2 * pulse.us)`` is ``cal.ramsey_detuning(mean_hz, theta, 2000.0)``. Where a
spacing or wait may be in any unit (``three_point_decay``, ``decay_time``,
``adaptive_t1_update``), a time is in ns and so is the result. A classical
function's arguments are numbers, so in its body they are taken as they are.

.. code-block:: python
   :class: fragment

   from cudaq_pulse.classical import calibration as cal

   @pulse.classical(max_latency_ns=200, link="controller")
   def t1_estimator(
       survived: pulse.Bits, alpha: float, beta: float, wait_ns: float
   ) -> tuple[float, float, float]:
       alpha, beta = cal.t1_update(alpha, beta, bool(survived[0]), wait_ns)
       return alpha, beta, min(max(cal.t1_wait(alpha, beta), 100), 10000)

**The converged pattern.** A kernel's trip counts are fixed when it
compiles. To stop early, the estimator carries a converged flag in its state,
keeps answering with its final point once converged, and the kernel gates its
probe pulses on the flag; values cannot leave the arms of an ``if``, so the
call stays outside it:

.. code-block:: python
   :class: fragment

   for _ in range(iterations):
       if done < 0.5:
           shift_frequency(d.tone, x)
           drive(d, probe)
           shift_frequency(d.tone, -x)
       m = readout(r, ro)
       state, x, done = golden_step.submit([m], state).result()

**Choosing among settings.** An integer answer selects among arms with
``if i == 0: ... elif i == 1: ...``, or a balanced split, ``if i < 2:`` with
nested tests, which needs fewer comparisons for many arms. Every arm is
scheduled to end together, after the longest.

**Randomness.** ``cudaq_pulse.classical.random_index(n)`` is an on-device
random source: an answer drawn uniformly from ``0 .. n - 1`` by the
controller (the manifest marks it ``"random"``). Simulation without shots
follows every value with probability 1/n; with shots each shot draws one.
Answers can be passed to later calls, for example to compute a
randomized-benchmarking sequence's recovery gate.

**Between shots.** A loop compiles once and runs with the current parameters
each iteration. Amplitudes, phases and frequencies can be float parameters of
the kernel. A quantity that shapes the timeline, such as a wait, is read at
run time: ``ParameterStore.reader(name)`` (``cudaq_pulse.classical.loop``) is
a classical function that answers with the store's current value, on
hardware a read of a controller register, and a variable wait takes its
length from it. ``EstimateLog`` records each estimate with what was true,
when it is known.

.. code-block:: python
   :class: fragment

   store = ParameterStore(wait_ns=2000.0)
   read_wait = store.reader("wait_ns", link="controller", bounds=(0.0, 20000.0))

   @pulse.kernel
   def shot(q, amplitude: float):
       ...
       longest_wait = 20 * pulse.us
       wait(d, read_wait.submit().result() * pulse.ns, bounds=(0, longest_wait))

   compiled = pulse.compile(shot, quantum_system=qs, control_system=control_system)
   for i in range(10_000):
       scheduled = compiled.lower(q, amplitude)  # no compiling after the first
       ...
       store.set("wait_ns", next_wait, iteration=i)

**A drifting quantum system.** ``pulse.quantum_system(...,
parameter_drift=True)`` builds a simulated quantum system whose qudit
parameters drift as a chip's do. It holds two sets of values:
``qs.physical_qudits``, the calibrated ones, which are all that compiling
reads, and ``qs.true_qudits``, the same ``PhysicalQudit`` type, which
``pulse.simulate`` plays. So a drifting system has the fingerprint, the
compiled programs and the exports of its non-drifting twin, and a program
compiled for it plays its calibration against the truth. The truth starts at
the calibrated values and moves with drift time:

- each ``simulate`` call advances drift time by its shots: ``shots_count``
  (or one) times the program's duration plus ``reset_time``, the time
  between shots (by default five times the longest calibrated T1 of the
  program's qudits, a passive reset). The true values are held for the call;
  a call longer than the shortest correlation time of the drift is split
  into chunks of whole shots, each at least that long and each played at the
  true values at its start.
- ``qs.advance(20 * pulse.ms)`` moves time on between experiments. It takes
  a time, not a bare number, and raises on a system without drift.
  ``qs.parameter_drift_time`` is the drift time so far.

.. code-block:: python
   :class: fragment

   from cudaq_pulse.quantum_systems.drift import OrnsteinUhlenbeck, ParameterDrift

   # Qudit 0's frequency wanders by 200 kHz with a 10 s correlation time.
   fast = ParameterDrift(frequency=OrnsteinUhlenbeck(200e3, 10 * pulse.s), t1=None, amplitude=None)
   qs = pulse.quantum_system("transmon_surface17", parameter_drift={0: fast}, parameter_drift_seed=5)
   scheduled = fringe.lower(qs.qudits[0])               # the calibrated values
   result = pulse.simulate(scheduled, shots_count=400)  # the true values; time moves on
   offset_hz = qs.true_qudits[0].frequency_hz - qs.physical_qudits[0].frequency_hz
   qs.advance(20 * pulse.ms)

``parameter_drift`` takes ``True`` (the default ``ParameterDrift()`` on every
qudit), one model for every qudit, or ``{qudit: model}``; ``qs.replace(
parameter_drift=True)`` gives drift to a system a factory built.
``parameter_drift_seed`` repeats a run, with each qudit on its own random
stream. A drifting system pickles with its truth and random state, so a
worker continues where it was. The default ``ParameterDrift()`` moves the
frequency by about 20 kHz with an hour's correlation time, halves T1 and back
with a ten-minute dwell time, and moves the drive amplitude by about 0.2% over
an hour: slow drift, of the size reported for transmons in P. V. Klimov et
al., Phys. Rev. Lett. 121, 090502 (2018), J. J. Burnett et al., npj Quantum
Information 5, 54 (2019) and S. Schlör et al., Phys. Rev. Lett. 123, 190502
(2019). Examples that want visible drift in seconds pass faster parts.

A model of your own is any object with ``start(calibrated, rng)`` and
``advance(true, calibrated, seconds, rng)``, each returning the qudit's
``PhysicalQudit`` (``true.replace(...)`` of any field the simulator reads),
and optionally a ``correlation_time``:

.. code-block:: python
   :class: fragment

   @dataclasses.dataclass(frozen=True)
   class LinearFrequencyDrift:
       hz_per_s: float

       def start(self, calibrated, rng):
           return calibrated

       def advance(self, true, calibrated, seconds, rng):
           return true.replace(frequency_hz=true.frequency_hz + self.hz_per_s * seconds)

   qs = pulse.quantum_system("transmon_surface17", parameter_drift={3: LinearFrequencyDrift(50.0)})

Loops of many iterations draw shots without simulating them from
``cal.survival_probability``, ``cal.ramsey_probability`` and
``cal.rabi_probability`` at the true values, with their own generator:
``rng.random() < cal.ramsey_probability(wait_ns, phase, offset_hz)``.

A calibration chain on the controller
-------------------------------------

``examples/21_calibration_protocol.py`` runs the millisecond calibration
chain of

   M. A. Marciniak, R. T. Birke, J. B. Severin, F. Berritta, D. Kjær,
   F. Nilsson, S. N. Themadath, S. Kallatt, J. L. Webb, K. Bentsen, T. Madsen,
   Z. Sun, S. Krøjer, C. W. Warren, J. Hastrup and M. Kjaergaard,
   "Millisecond-Scale Calibration and Benchmarking of Superconducting Qubits",
   arXiv:2602.11912 (2026),

as one program whose estimators are classical functions on the controller's
link, so nothing goes to a host or a GPU between steps: readout assignment
errors, the drive frequency from Ramsey shots at three phases (sparse phase
estimation), the pi and pi/2 amplitudes from pulse trains at three amplitude
scales, T1 from the analytical decay estimator at delays spaced by the last
estimate, and T1 again by the adaptive tracker of

   F. Berritta, J. Benestad, J. A. Krzywda, O. Krause, M. A. Marciniak,
   S. Krøjer, C. W. Warren, E. Hogedal, A. Nylander, I. Ahmad, A. Osman,
   J. Biznárová, M. Rommel, A. F. Roudsari, J. Bylander, G. Tancredi, J. Danon,
   J. Hastrup, F. Kuemmeth and M. Kjaergaard, "Real-time adaptive tracking of
   fluctuating relaxation rates in superconducting qubits", arXiv:2506.09576
   (2025),

whose prior is centred on the previous estimate and whose likelihood uses
the measured assignment errors. Each step is a kernel that returns its
estimator's answer, and the next step plays its pulses with it: the drive
tone is shifted by the measured detuning, the trains use the corrected
amplitudes, the delays come from the T1 estimate.

The estimators in ``cudaq_pulse.classical.calibration`` that the chain uses
(``adaptive_t1_update``, ``binary_search_update``, ``decay_time``,
``sparse_phase`` and their helpers) run on a controller's processor: a
fixed number of scalars of state, closed-form arithmetic, branches and
loops of a trip count fixed when the program compiles, and no allocation
or exceptions. An estimate that cannot be formed is a result
(``decay_time`` returns whether it formed one), not an error.

Compile the chain with ALAP scheduling (the default). A pulse just before a
variable wait then ends where the wait starts, so the qubit decays for the
answer's length; under ASAP the excitation plays as early as it can, and
the qubit also decays through the call's latency.

A program that measures hundreds of times in one shot is simulated one
path per shot: ``pulse.simulate(scheduled, shots_count=n, branches="sampled")``. Idle
waits and square pulses are propagated exactly, so a shot's cost does not
grow with its waits. Under ``branches="sampled"``, a loop that waits each
round for a variable wait or a blocking call (scheduled ``"asap"`` or
``"alap"``, with its lines on one timeline) costs the same host time per
round at any length: each answer is bound and scheduled only in the part of
the program between two such waits. Any other program with calls is
scheduled again whole after each answer, so its host time per interval
grows with the rounds before it.

What it costs the controller
----------------------------

``scheduled.controller_usage`` reads the body of each classical function a
processor runs once and reports how many operations it holds, its state in
bytes, its compute time against its ``max_latency_ns`` and its calls per
shot, with each link's calls, slots and bytes per shot. An expression of
constants costs nothing, as it folds when the program compiles, and
dividing by a constant counts as a multiply. The costs and limits come
from the ``Processor`` of a controller that runs the functions on its
``links``:

.. :spellcheck-disable:

.. code-block:: python
   :class: fragment

   processor = pulse.Processor(
       links=["controller"],
       cycles_per_op=controller.OpCycles(div=8, sqrt=8, exp=12, log=12, trig=12),
       clock_ghz=0.25,       # 4 ns per cycle
       fixed_cost_only=True,
       max_ops_per_function=256,
       max_state_bytes_per_function=512,
   )
   system = pulse.ControlSystem(
       controllers=[pulse.Controller(processors=[processor])],
       links={"controller": Link(fixed_ns=4)},
   )
   scheduled = pulse.compile(chain, quantum_system=qs, control_system=system).lower(q)
   print(scheduled.controller_usage)

.. :spellcheck-enable:

The report is a table: for the chain's adaptive T1 update, 47 operations
(one ``exp``, one ``log``, eight divisions) and 0.5 us at these costs, called
50 times per shot. ``processor.usage(fn)`` is one function's cost on a
processor on its own: its cycles always, its time only when the processor
has a ``clock_ghz``, and no calls per shot, which come from a compiled
program. A function that computes past its budget, holds too many
operations or bytes, or (with ``fixed_cost_only=True``) uses a ``while``
loop, a list or an exception fails to compile with ``ControllerLimitError``
naming it, the limit and the line. Without processors, compiling does not
read function bodies.

Running on hardware
-------------------

A controller that runs these programs needs, beyond its pulse sequencer: a
way to send a request on a link at a scheduled time and to receive the
answer into registers; branching on an answer; frequency, phase and
amplitude updates from an answer; and, for variable waits, a wait whose
length comes from a register, after which the rest of the segment is released
together. The emulation artifact gives every event its segment and its time
from the segment's start for that purpose. A segment holds the lines of its
barrier's timeline, so segments of different timelines move independently: an
segment's shift adds to that of the segment it follows, which is the one before
it unless the segment names another in ``after``.

Not available yet: a channel through the CUDA-Q Realtime host library
(``emulation.RealtimeChannel``), exporting calls as CUDA-Q
``cc.device_call`` operations (``emulation.device_call_text``), and lowering
links to a controller dialect. The wire format, the manifest and the
handler signature are those of CUDA-Q Realtime, so these slot in without
changing programs.

Questions
---------

**Can a parameter be a** ``Sequence`` **rather than a** ``list``\ **?** Yes:
``Sequence[int]``, ``Sequence[float]`` and ``Sequence[complex]``, from
``collections.abc`` or ``typing``, mean what the ``list`` forms do. A type
checker then also accepts a tuple where ``list[int]`` takes only a list;
this guide writes ``list``.

**Why does a type checker show** ``pulse.Bits`` **as** ``list[int]``\ **?**
Checkers see ``pulse.Bits`` as ``list[int]``, so a body may return any list
of bits and an external's placeholder ``pulse.Bits()`` checks; at run time
it is a list type of its own, which tells the compiler the items are bits.
A list is invariant, so a variable typed as a list of another item type
(``list[bool]``, say) is flagged where ``pulse.Bits`` is expected, while a
list written in the call, ``decode.submit([m0, m1])``, is not.
