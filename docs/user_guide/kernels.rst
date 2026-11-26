..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Pulse Kernels
=============

The ``@pulse.kernel`` decorator is the entry point for writing
pulse-level quantum programs. It captures the decorated function's
Python bytecode and traces it into an intermediate representation
that can be compiled to MLIR.

Import Convention
-----------------

A single import gives you everything:

.. code-block:: python

   import cudaq_pulse as pulse

Inside ``@pulse.kernel`` functions, DSL operations (``drive``,
``gaussian``, ``get_drive_line``, etc.) are used as bare names, which
editors complete and check with no configuration (see :doc:`editors`).
Infrastructure stays behind the ``pulse.`` prefix:

.. code-block:: python

   @pulse.kernel
   def rabi_oscillation(qubit, duration, sigma):
       drive_line = get_drive_line(qubit)
       drive(drive_line, gaussian(duration, 0.5, sigma))

   qs = pulse.quantum_system("transmon_surface17")
   duration = 32 * pulse.ns
   sigma = 8 * pulse.ns
   rabi_oscillation(qs.qudits[0], duration, sigma)

Defining a Kernel
-----------------

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def my_kernel(qubit, x_duration, x_sigma):
       drive_line = get_drive_line(qubit)
       waveform = gaussian(x_duration, 0.3, x_sigma)
       drive(drive_line, waveform)

   x_duration = 20 * pulse.ns
   x_sigma = 5 * pulse.ns
   my_kernel(qs.qudits[0], x_duration, x_sigma)

The kernels on this page take their pulse values as arguments, and the calls
pass these; see :ref:`durations`.

Qudit arguments are **qudit handles** of a quantum system: ``qs.qudits[i]``
for one qudit, and ``qs.qudits[0:4]`` or ``qs.qudits[[0, 2]]``
for a register, which a kernel indexes and iterates like a list. The kernel
runs on the quantum system of its handles; see :doc:`quantum_systems`. How arguments bind,
which of them the kernel is specialized on, and what a call returns are
described in :doc:`compilation`.

.. code-block:: python

   @pulse.kernel
   def two(q0, q1, amplitude: float):
       ...

   reg = qs.qudits[0:2]
   two(reg[0], reg[1], 0.3)

A ``*qudits`` parameter takes any number of qudits, one argument each. In
the kernel it is a register of the call's qudits, so it indexes, slices,
iterates, has a ``len()`` and passes on to another kernel as ``*qudits``;
parameters after it are keyword-only:

.. code-block:: python

   @pulse.kernel
   def x_all(amplitude: float, *qudits: pulse.Qudit, duration, sigma):
       for q in qudits:
           drive(get_drive_line(q), gaussian(duration, amplitude, sigma))

   @pulse.kernel
   def x_then_measure_first(*qudits: pulse.Qudit, duration, sigma, readout_duration):
       x_all(0.5, *qudits, duration=duration, sigma=sigma)
       first = get_readout_line(qudits[0])
       sync(first, *[get_drive_line(q) for q in qudits])
       readout(first, square(readout_duration, 0.05))

   readout_duration = 500 * pulse.ns
   x_all(0.3, qs.qudits[0], qs.qudits[2], duration=x_duration, sigma=x_sigma)
   x_then_measure_first(
       *qs.qudits[0:4], duration=x_duration, sigma=x_sigma, readout_duration=readout_duration
   )

Annotated ``Qudit``, the qudits are runtime, as for a ``Register``
parameter: one compilation serves every call with the same number of
qudits. Unannotated, the kernel is specialized on which qudits it gets. A
call may pass none, as with an empty register; ``*qudits`` takes only qudit
handles, so unpack a register into it (``*reg``). See :doc:`compilation`.

Qudits always come from a quantum system: a kernel takes them as parameters, and a
call passes the quantum system's handles (``qs.qudits[i]``, or a register
``qs.qudits[0:4]``). A call without handles raises
``NoQuantumSystemError``, unless the kernel was compiled for a quantum system with
``pulse.compile(k, quantum_system=qs)``.

The kernel language (``get_drive_line``, ``drive``, ``gaussian``, ...) lives
in ``cudaq_pulse.ops``; inside a kernel its names are used bare, as above.

.. _durations:

Durations: cycles or times
--------------------------

A kernel counts time in cycles of the quantum system's clock (``qs.clock_ghz``).
A duration (a waveform's length or flat top, a ``wait`` and its bounds, a
``pulse.round_duration`` length) is either an ``int`` number of cycles or a time
with units: ``20 * pulse.ns``, ``4 * pulse.us``, also ``pulse.ms`` and
``pulse.s``. A time converts to cycles exactly when the kernel compiles,
for the quantum system it compiles for, so a kernel whose durations are times
runs unchanged on quantum systems with different clocks. Shape parameters
(``sigma``, DRAG ``beta``) take cycles, fractional too, or times.

Keep durations out of the kernel body: pass them as arguments, or read
them from a calibration the kernel closes over.

.. code-block:: python

   @pulse.kernel
   def x_then_wait(qubit, gap, x_duration, x_sigma, x_amp):
       d = get_drive_line(qubit)
       x = gaussian(x_duration, x_amp, x_sigma)
       drive(d, x)
       wait(d, gap)
       drive(d, x)

   fast = pulse.quantum_system("transmon_surface17", clock_ghz=2.0)
   slow = pulse.quantum_system("transmon_surface17", clock_ghz=1.0)
   x_amp = 0.2
   gap = 100 * pulse.ns
   x_then_wait.lower(fast.qudits[0], gap, x_duration, x_sigma, x_amp)  # 40, 200, 40 cycles
   x_then_wait.lower(slow.qudits[0], gap, x_duration, x_sigma, x_amp)  # 20, 100, 20 cycles

- A plain float in a duration is an error, since it could be cycles or
  nanoseconds; the error shows both spellings::

     gaussian(): duration is a float (20.0). Durations are an int number of cycles, or a time with units:
         gaussian(20, ...)               # 20 cycles
         gaussian(20 * pulse.ns, ...)    # 20 nanoseconds

- A time that is not a whole number of cycles is an error that names its
  cycles; ``pulse.round_duration(line, 10.3 * pulse.ns)`` rounds it to whole
  :ref:`timesteps <timestep>` of the line instead.
- A time times a number, a time plus a time, a time over a number and minus
  a time are times. A time plus a number of cycles, and a time times a
  time, are errors. ``0.1 * pulse.us`` is exactly 100 ns: a time holds its
  value as a fraction, from the digits it is written with.
- Times compare with times: ``pulse_length <= 40 * pulse.ns``. Comparing a
  time with a bare number, ``==`` or an ordering, is a ``TypeError``, since
  the number could be cycles or nanoseconds; ``float(t)`` is the time in
  nanoseconds, to compare as a number. A time never hashes like a number,
  so ``20 * pulse.ns`` and ``20.0`` are different keys of a ``dict`` or
  ``set``.
- A ``float`` parameter is a plain number, a number of cycles where it sets
  a length. Write ``wait(d, tau * pulse.ns)`` for a ``tau`` in nanoseconds:
  the length stays a runtime parameter, converted with one multiply. A time
  passed to a ``float`` parameter is a ``TypeError``. The answer of a
  classical call converts the same way: ``wait(d, f.result() * pulse.ns,
  bounds=(0, 4 * pulse.us))``.
- ``result(timeout=...)`` takes a time or a number of nanoseconds, as the
  declaration options such as ``max_latency_ns=350`` do, of whole cycles of
  the program clock.

.. _unit-warnings:

A bare ``int`` in a duration is a number of cycles, and a bare number in
``shift_frequency`` or ``set_frequency`` is in Hz, so ``square(20, a)``
lasts 10 ns at 2 GHz and ``shift_frequency(tone, 5)`` shifts by 5 Hz.
Neither is an error. To catch them, compile with
``pulse.compile(k, unit_warnings=True)``: each duration, wait, ``sigma``,
DRAG ``beta`` or flat-top ``width`` given as a bare nonzero number, and
each frequency below a millionth of the quantum system's lowest carrier
(a shift) or a thousandth of it (``set_frequency``), is a
``pulse.UnitWarning`` at its kernel line::

   kernels.py:7: UnitWarning: square(): duration 20 has no units, so it is 20 clock
   cycles (10 ns at the 2 GHz clock); write it with units, as 20 * pulse.ns

A time with units, ``0``, a count from ``pulse.round_duration`` and a
calibrated pulse are never warned. The warnings are issued while the kernel
traces, on its first compilation for each set of arguments; they do not
change the program, and without ``unit_warnings`` nothing is checked.
``warnings.filterwarnings("error", category=pulse.UnitWarning)`` makes them
errors.

Control Flow
------------

Kernels support a subset of Python control flow that can be captured
at trace time:

**For loops** with compile-time integer bounds:

.. code-block:: python

   @pulse.kernel
   def echo_sequence(qubit, gap, x_duration, x_sigma):
       drive_line = get_drive_line(qubit)
       for i in range(5):
           drive(drive_line, gaussian(x_duration, 0.3, x_sigma))
           wait(drive_line, gap)

A loop whose induction variable is never read, like the one above, is
captured once as a structured ``scf.for`` in the kernel IR, so the IR size
does not depend on the trip count, unless it is small: when the trip count
times the number of calls in the body is at most the unroll threshold, the
loop is unrolled while tracing (see below). Loops that read the induction variable,
or that carry Python state from one iteration to the next, are unrolled
exactly at trace time. A name the body binds before reading it, such as an
inner loop's variable, carries nothing, even if the kernel used the name
before the loop. Compiling keeps the same loops as an ``scf.for`` that
carries the lines and tones its body advances; the scheduler times every
iteration. A loop bound given by a ``float`` argument specializes the kernel
on its value; ``while`` loops are rejected.

.. themed-figure:: figures/rolled_loop.svg
   :alt: Two timelines of echo_sequence on one drive line. As exported, the loop is one body, a 20 ns pulse and a 20 ns wait, with 5 trips and a period of 40 ns. The schedule it stands for repeats that body 5 times, from 0 to 200 ns, with a dotted line where each trip starts.
   :width: 100%

   ``echo_sequence`` with a gap as long as its pulse. Kept rolled, the loop
   is exported as one body with its trips and its period; the lower row is
   the schedule it stands for, which is also the unrolled kernel's, with a
   dotted line where each trip starts.

**The unroll threshold** is ``pulse.compile(k, unroll_threshold=n)``, else
``@pulse.kernel(unroll_threshold=n)``, else 8. It is a non-negative ``int``;
0 never unrolls, so every such loop stays rolled. Both forms are the same program: without passes their
schedules are identical, and with the default passes ``fuse_pulses`` may merge
back-to-back pulses of the unrolled form, with the same evolution. A kernel
or compiled kernel keeps the threshold it was given; compile it again for
another.

The default comes from timing both forms, on one transmon, of loops of 1 to
512 iterations whose bodies hold 2, 3 or 6 calls: compiling (trace, lowering
to MLIR, passes and schedule, a fresh ``pulse.compile`` each time) and
simulating on the CPU backend. A rolled loop costs about 0.3-0.6 ms to
compile at any size; unrolled, compiling costs about 7 us per call and
simulating grows with the number of separate pulses (a rolled loop lowers to
one pulse train). The unrolled form is up to 10 % faster below an estimated
size of 6, even at 8-12, and slower beyond: 1.2-1.4x at 24, 2-3x at 96 and
more than 20x at 1500, where simulation dominates. The default, 8, is the
crossover; ``benchmarks/compiler/unroll_threshold.py`` repeats the measurement.

**Unrolled size.** An unrolled loop holds its body once per iteration, so
its size has no limit of its own. A controller given
``Controller(max_instructions=n)`` holds at most ``n`` operations as
the kernel traces them: tracing stops as soon as the program grows past
them, with a ``CompilationError`` at the line of the loop that unrolled it,
naming the loop, about how many operations it unrolls to and the limit. A
rolled loop holds its body once.

**Program length.** A program runs at most its controller's
``Controller(max_program_duration=...)``, a time or a number of ns,
and at most 2**60 clock cycles without one. The schedule decides: the
program lasts to the end of its last operation on the controller's lines,
so operations that play at the same time on different lines count once,
and a scheduled program longer than the limit is a ``CompilationError``
that gives its length. A wait whose length a float sets counts at the
length each call gives it; a call whose values make the scheduled program
longer is a ``ValueError`` when it binds them. A wait whose length is known
only when the program runs (an answer of a classical call) is scheduled at
its lowest length and counts at its highest, once for each time it runs,
for the lines it holds back: those a ``sync`` or a value ties to it. A line
that nothing ties to the wait ends when it is scheduled to. The error then
gives the length as scheduled and what those waits add. Such a wait with
no highest length (``bounds=(lowest, None)``, or no ``bounds``) has nothing
to hold to the limit: where it holds back a line of a controller that sets
``max_program_duration`` it is a ``CompilationError`` at the wait, which
says to bound it. So is a call of a blocking classical function
(``latency_mode="blocking"``), whose answer has no latest time, where an
operation of such a controller comes after its answer.

Building the module from the traced kernel also sums the program op after
op, every iteration of its loops (the longer arm of an ``if``), so that a
program no schedule can save is refused before it is scheduled. The
operations of one line follow each other, so a program on ``n`` lines that
fits sums to at most ``n`` times the limit: past that, and past 2**60
cycles without a limit, it is a ``CompilationError`` (for a wait whose
length a float sets, a ``ValueError`` at the call). Without a controller
limit the error is at the op or loop that passes 2**60 cycles. A line's ``delay_ns``
counts too: a delay longer than 2**60 cycles, or one that takes the
scheduled program past them, is a ``CompilationError``.

**For loops over Python values** known when the kernel is traced: lists,
tuples, dictionaries, ``enumerate()``, ``zip()``, ``reversed()``, a qubit
vector, or any other finite iterable. These loops are always unrolled, one
traced iteration per element, with ``break``, ``continue`` and ``else``
behaving as in Python. An iterator may be endless, and a body that plays
nothing never fills the controller, so tracing such a loop stops with a
``CompilationError`` after 1,000,000 iterations; a controller's
``max_instructions`` bounds these loops as it does ``range()`` ones. Lines
held in lists stay threaded through the loop:

.. code-block:: python

   @pulse.kernel
   def sweep(qubits, x_duration):
       lines = [get_drive_line(q) for q in qubits]
       for amplitude in [0.1, 0.2, 0.3]:
           for line in lines:
               drive(line, square(x_duration, amplitude))

**If/else** with compile-time conditions:

.. code-block:: python

   @pulse.kernel
   def conditional_pulse(qubit, use_drag, x_duration, x_sigma, x_beta):
       drive_line = get_drive_line(qubit)
       if use_drag:
           waveform = drag(x_duration, 0.3, x_sigma, x_beta)
       else:
           waveform = gaussian(x_duration, 0.3, x_sigma)
       drive(drive_line, waveform)

**Measurement feedback**: an ``if`` on a measurement result (``if result:`` or
``if not result:``) is kept as a conditional region:

.. code-block:: python

   @pulse.kernel
   def active_reset(qubit, x_duration, x_sigma, x_beta, readout_duration):
       drive_line = get_drive_line(qubit)
       readout_line = get_readout_line(qubit)
       result = readout(readout_line, square(readout_duration, 0.05))
       sync(drive_line, readout_line)
       if result:
           x = drag(x_duration, 0.5, x_sigma, x_beta)
           drive(drive_line, x)

An ``if`` on the answer of a classical call is kept the same way
(``if correction:``, ``if flip[0]:``, or the in-time flag of
``result(timeout=...)``), and ``and``, ``or`` and ``not`` combine answers
and measurement results in one condition; see :doc:`classical_calls`.

Both branches are traced, and lines and tones they rebind become the
results of the region. Both branches must continue after the ``if``. A
number the branches set differently holds, after the ``if``, the value of
the branch that ran:

.. code-block:: python

   @pulse.kernel
   def flip_on_one(qubit):
       drive_line = get_drive_line(qubit)
       readout_line = get_readout_line(qubit)
       result = readout(readout_line, square(1000, 0.05))
       sync(drive_line, readout_line)
       if result:
           amp = 0.5
       else:
           amp = 0.0
       drive(drive_line, square(40, amp))

The number is one value the measurement selects. On a quantum system that takes it
at run time (an amplitude, a phase or a frequency), every op given it, or a
waveform made from it, reads that value. Otherwise, and always for a
duration or a waveform's shape, each such op runs in each branch with that
branch's value, as if it were written in both. No quantum system takes a selected
value at run time yet. Either way the op waits for the measurement, and
the program schedules and simulates the same. A name assigned in only one branch, with no value
before the ``if``, has none after it: reading it raises a
``CompilationError`` that names it and the line of the ``if``. Other Python
values, qudits, and lines of different kinds must not differ between the
branches. Compiling keeps the
branch as an ``scf.if`` on ``pulse.is_set`` (whether the readout
observed a nonzero level), whose arms yield the lines and tones they
advance; both arms start once the measurement completes, and the code after
the ``if`` waits for the longer arm.

A condition on measurement results can be kept in a name and tested later,
once or in several ``if`` statements:

.. code-block:: python
   :class: fragment

   flip = s0 and not s1
   if flip:
       drive(drive_line, x)

``and``, ``or``, ``not``, a conditional expression, ``any()`` and
``all()`` of results keep their truth, whether each observed a nonzero
level, as an ``if`` tests it; ``&``, ``|``, ``^``, ``==`` and ``!=``
combine such conditions, and in ``and``, ``or`` and conditional expressions
conditions on answers of classical calls mix with them. ``if flip:`` compiles to the ifs ``if s0 and not s1:`` gives: one
``pulse.is_set`` test per readout, in the order the condition reads them,
each where the condition still depends on that readout. A kept condition is
only tested: it is not a number (``0.1 * flip`` and an amplitude or a
duration from it are a ``CompilationError``), and ``&``, ``==`` and the
other operators do not take a measurement result itself, whose level they
would compare.

Python Inside Kernels
---------------------

Besides the control flow above, a kernel body may use:

- list, dictionary and set comprehensions, nested and filtered ones too, and
  set displays such as ``{8, 16}``. A comprehension is unrolled while
  tracing like the ``for`` loop it stands for, so
  ``[get_drive_line(q) for q in qubits]`` builds a list of lines and
  ``[readout(r, wf) for _ in range(12)]`` a list of results.
- generator expressions, such as ``sum(a for a in amps)`` or
  ``sync(*(lines[i] for i in active))``. A generator expression runs to
  the end where it is written, as a list comprehension would, so pulses
  it emits are emitted there, in order, even if what consumes it stops
  early, as ``any()`` does.
- starred assignment, such as ``first, *rest = lines``, and slice
  assignment, such as ``xs[1:3] = ys``.
- ``assert`` and ``raise`` for checks while tracing: a failing
  ``assert n > 0, "msg"`` raises ``CompilationError("assertion failed in
  kernel: msg")``.
- ``sync()`` over lines, lists or tuples of lines, nested too:
  ``sync(lines)`` and ``sync([d0, d1], d2)`` synchronize every line they
  hold.
- calls with ``*args`` and ``**kwargs``, unpacked while tracing, so
  ``sync(*lines)`` synchronizes a list of lines built in a loop and
  ``helper(d, **opts)`` passes a dictionary of options.
- lines and tones kept in any container: a list of lines, a dictionary or
  nested lists. A qudit has one drive and one readout line, so an operation
  always uses the line, and its tone, as they are now, whichever copy of
  them the kernel passes. ``append``, ``insert``, ``extend``, ``pop``,
  ``remove`` and ``del xs[i]`` work on lists of lines.

A ``float`` argument stays a runtime parameter, compiled once for every
value, through ``+``, ``-``, ``*``, ``/`` and negation, including as the length
of a ``wait`` (see :ref:`wait-lengths`). Any other use that needs its value
while tracing, such as ``abs``, ``min``, ``math.sin``, ``//``, a comparison,
a ``range`` bound or a waveform duration, makes it static, so each new value
compiles the kernel again. The first such use warns once per kernel and
argument with a ``cudaq_pulse.PulseWarning`` (also a ``RuntimeWarning``)
that names the argument, the operation and the kernel line, and suggests
computing the value on the host, annotating the argument as ``int``, or
giving it a fixed value;
``warnings.filterwarnings("error", category=cudaq_pulse.PulseWarning)``
makes it an error.

.. _wait-lengths:

Wait lengths
~~~~~~~~~~~~

A ``wait`` whose length is computed from ``float`` arguments keeps them
runtime parameters, so a sweep over a delay compiles once:

.. code-block:: python

   import numpy as np

   @pulse.kernel
   def echo(qubit, tau: float, x_duration, x_sigma, x_beta, readout_duration):
       d = get_drive_line(qubit)
       r = get_readout_line(qubit)
       x90 = drag(x_duration, 0.25, x_sigma, x_beta)
       x180 = drag(x_duration, 0.5, x_sigma, x_beta)
       drive(d, x90)
       wait(d, tau / 2 * pulse.ns)
       drive(d, x180)
       wait(d, tau / 2 * pulse.ns)
       drive(d, x90)
       sync(d, r)
       readout(r, square(readout_duration, 0.05))

   q = qs.qudits[0]
   x_beta = 0.25 * pulse.ns
   compiled_echo = pulse.compile(echo)
   for tau in np.linspace(0, 10_000, 1000):  # nanoseconds
       # one compilation for the whole sweep
       compiled_echo.lower(q, tau, x_duration, x_sigma, x_beta, readout_duration)

.. themed-figure:: figures/runtime_wait.svg
   :alt: Three timelines of the echo kernel on a drive and a readout line, one per call of the compiled kernel: an x90 pulse, a wait, an x180 pulse, a wait, an x90 pulse, then the start of the 500 ns readout, which runs past the right edge at 325 ns. tau = 0 ns: no wait, readout from 60 ns; tau = 100.1 ns: two waits of 50 ns, readout from 160 ns; tau = 200.2 ns: two waits of 100 ns, readout from 260 ns. The x180 pulse's Q quadrature is drawn dashed.
   :width: 100%

   Three calls of ``compiled_echo`` from the sweep. Each wait is half of
   ``tau``, rounded to a clock cycle (half a nanosecond here); the readout
   runs past the right edge. Dashed blue is a pulse's Q quadrature.

- Each call computes the length from its values, in clock cycles (here
  nanoseconds times ``qs.clock_ghz``), and rounds it to the nearest cycle,
  ties up (``floor(x + 0.5)``): 99.5 cycles is 100, 99.49 is 99.
- The call's schedule is exactly that of the kernel compiled with that
  length: ``sync``, ALAP and ASAP placement, loops, branches and called
  kernels see the rounded length, and other lines are not held back by the
  wait. ``lower()`` only binds the values; the schedule, ``mlir()`` and the
  simulation of a call come from the module with them bound and scheduled
  again, which takes tens of microseconds, not a compile: such a call is
  ``rescheduled``.
- ``wait(line, length, bounds=(lowest, highest))`` bounds the length, in
  cycles or as times; ``highest`` may be ``None``, and without ``bounds`` the length
  only has to be non-negative. A call whose length is not finite, is
  negative or falls outside the bounds raises ``ValueError`` naming the
  kernel, the expression and the value, before anything runs.
- Without values (``pulse.compile(k, quantum_system=qs).schedule`` of a kernel that
  only has float parameters, or ``scheduled.generic``) the module is scheduled
  with every such wait at its lowest length, 0 unless ``bounds`` says
  otherwise.
- A length that needs more than arithmetic (``int(tau)``, ``tau // 2``,
  ``round(tau)``) still makes the parameter static, with the warning.
  A literal or static length is not rounded: it must be an ``int`` or a
  time that is a whole number of cycles (see :ref:`durations`), and
  ``bounds`` checks it while tracing.

A float argument of a call is a number: a NaN given to ``k(...)``,
``lower(...)`` or ``bind(...)`` raises ``InvalidValueError`` (a
``ValueError``) naming the argument when it is bound, and an infinity
does where the program is used (run, simulated, exported or printed).
NaN in classical data is allowed (see :doc:`classical_calls`).

Operations check their arguments when they are traced. Passing a tone
where a line belongs, a number where a waveform belongs, or a readout line
to ``drive`` raises a ``CompilationError`` that names the operation and
the argument, such as ``drive(): argument 1 must be a drive line, got a
tone``, before any MLIR is built.

An error raised while a kernel is traced carries a note with the kernel
source lines that led to it, as a traceback shows them: the line in the
kernel and, for an error inside a helper, the helper's line. A misspelled
operation or attribute gets a suggestion, as in ``unknown pulse op:
drvie; did you mean drive()?``. The kernel language is in
``cudaq_pulse.ops``, so ``pulse.drive(...)`` fails with a message that says
to write ``drive(...)`` or ``pulse.ops.drive(...)``.

Unsupported Patterns
--------------------

The following Python constructs are **not** supported inside kernels
and will raise ``CompilationError``:

- ``while`` loops
- Lambdas and function definitions inside the kernel body; define the
  function at module level, where it is inlined
- Generator functions (``yield``)
- ``try`` / ``except`` and ``with`` blocks
- ``match`` with sequence, mapping or class patterns (literal and ``|``
  patterns work)
- Assignments to globals and to attributes (``obj.x = ...``), class
  definitions, ``async`` code and star imports
- Ordering comparisons (``<``, ``>``, ``<=``, ``>=``) of measurement
  results or boolean answers, which are known only when the program runs;
  combine them with ``and``, ``or`` and ``not``

The error names the construct and, where there is one, what to write
instead.

Kernels may read globals and variables captured from an enclosing scope; a
change to one is seen on the next call. Calls to ``cudaq_pulse`` ops are
emitted as IR. Other kernels, and helper functions that take IR values, call
``cudaq_pulse`` ops or call a kernel, are inlined, inside loops too. So are
methods of helper objects and ``functools.partial`` objects of such
functions, and helpers wrapped by a decorator: the wrapper is inlined, and it
calls the original. Helpers bind their arguments as Python does, including
default, keyword-only, ``*args`` and ``**kwargs`` parameters. Any other
Python call, such as ``math.sin`` or ``len``, is evaluated at trace time on
concrete values; if one of them calls a ``cudaq_pulse`` op, say through
``map``, the error names the op and the call.
