..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Calling and Compiling Kernels
=============================

A ``@pulse.kernel`` is called like a Python function. The first call with
new arguments compiles the kernel; later calls reuse the compilation and
only run it.

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def rabi(qubit, amplitude: float, duration, sigma):
       drive_line = get_drive_line(qubit)
       drive(drive_line, gaussian(duration, amplitude, sigma))

   qs = pulse.quantum_system("transmon_surface17")
   q = qs.qudits[0]
   duration = 20 * pulse.ns
   sigma = 5 * pulse.ns

   rabi(q, 0.3, duration, sigma)                    # compiles, then runs; returns None
   rabi(q, 0.4, duration, sigma)                    # same compilation, new amplitude
   scheduled = rabi.lower(q, 0.3, duration, sigma)   # compiles (or reuses), does not run
   print(scheduled.mlir())

Arguments
---------

Arguments bind exactly as for a Python function: positional and keyword
arguments, defaults, keyword-only parameters, and ``TypeError`` for a call
that does not match the signature. A ``*qudits`` parameter takes the
call's remaining positional arguments, which must be qudit handles, and
binds them as one register; a kernel cannot take ``**kwargs``. A kernel,
``kernel.lower`` and ``pulse.compile(kernel)`` keep the decorated
function's signature for type checkers, so editors check each call against
it.

Annotations decide what a compilation depends on:

- A parameter annotated ``float`` is a *runtime* value. It stays a function
  argument of the compiled MLIR module, wherever it flows (an amplitude, a
  phase, a tone frequency, the length of a ``wait``), so calling with a new
  value never recompiles. A call whose values set a wait length schedules
  the module again with them, without tracing it; see
  :ref:`wait-lengths`.
- A parameter annotated ``Qudit`` or ``Register`` is a runtime value too:
  the compiled module takes the qudits as arguments, so calling on other
  qudits of the quantum system never recompiles.
- ``*qudits: Qudit`` is a runtime register of the call's qudits: the kernel
  compiles once for each number of qudits passed. An unannotated
  ``*qudits`` is a static register, like an unannotated register parameter.
  Any other annotation of a ``*`` parameter is a ``TypeError`` when the
  kernel is defined; pass other values as named parameters (a
  ``list[float]`` for a runtime list of numbers).
- Every other parameter (``int``, anything unannotated, including an
  unannotated qudit or a duration passed as a time) is *static*: the kernel
  is specialized on its value. A list or tuple of qudit handles is
  rejected: pass a Register (``qs.qudits[0:4]``) or use ``*qudits``. Annotate a
  qudit parameter to compile once for every placement. A time passed to a
  ``float`` parameter is a ``TypeError``; see :ref:`durations`.
- A ``float`` the program needs while it compiles -- one that sets a
  waveform duration, goes through ``int()``, decides an ``if`` or bounds a
  loop -- is specialized on as well. ``kernel.static_parameters`` and
  ``kernel.runtime_parameters`` show the split.

Return values
-------------

A call returns what the kernel returns: ``None`` without a ``return``, a
measurement's outcome (0, 1, ...) for a measurement made at the top level of
the kernel, values known at compile time, or tuples and lists of these.
Returning a value computed from a ``float`` parameter is an error.

Quantum systems
---------------

A kernel runs on the quantum system of the qudit handles it is called with; see
:doc:`quantum_systems`. One kernel serves every quantum system it is called with, each with
its own specializations. A call without handles raises ``NoQuantumSystemError``,
unless the kernel is compiled for one quantum system (``quantum_system=`` below); the
process quantum system is never used for it. Qudit frequencies and the pulse clock
come from the quantum system.

Caching and Python semantics
----------------------------

Each kernel keeps up to 32 specializations, least recently used first out;
set ``CUDAQ_PULSE_CACHE_SIZE`` or ``@pulse.kernel(cache_size=N)`` to change
that. ``kernel.cache_info()`` reports the specializations held, compiled
and evicted.

Globals and closure variables behave as in Python: the values a kernel read
while it compiled (module globals, closure cells, attribute chains such as
``cfg.amplitude``, the contents of lists and dicts it read) are checked on
every call, and a change compiles the kernel again. A kernel that calls a
host function whose result can change between calls (``random.random()``)
compiles on every call.
A static argument changed in place, such as a NumPy array (``amps[:] = new``)
or an attribute of a dataclass or other object (``p.amplitude = 0.5``),
also compiles the kernel again on the next call.

``pulse.compile``
-----------------

``pulse.compile(kernel)`` compiles ahead of the call and takes no kernel
arguments. It returns a ``CompiledKernel``: calling it runs the kernel and
returns what the kernel returns, and it has the same ``lower`` as the
kernel. Its specializations are kept for as long as the object lives and are
never evicted.

.. code-block:: python

   sweep = pulse.compile(rabi)
   for amplitude in (0.1, 0.2, 0.3):
       sweep(q, amplitude, duration, sigma)   # one compilation for the whole sweep

   pinned = pulse.compile(rabi, quantum_system=qs, schedule="asap")

Options, all keyword-only:

``schedule`` *(str, optional)*
   Scheduling policy of the compiled kernel itself: ``"asap"``, ``"alap"``,
   ``"rcp"`` or ``"alap_rcp"``; by default the kernel's own ``schedule=``,
   else ``"alap"``. Every policy preserves data and physical-line
   dependencies, including the order of pulses that play on the same tone,
   and starts an operation that depends on a readout once the controller's
   ``feedback_latency_cycles`` have passed after the readout ends; the RCP
   variants also enforce the ``ControlSystem``'s lanes. Kernels
   it calls keep their own ``schedule=`` (see `Per-kernel schedules`_).

``flatten_schedules`` *(bool, default False)*
   Ignore the ``schedule=`` of every called kernel and schedule the whole
   program with ``schedule``, for comparison.

``passes`` *(sequence of Pass or str, optional)*
   Optimization passes, run in the order given: members of
   :class:`cudaq_pulse.Pass` or their names. ``DEFAULT_PASSES`` by default
   (``VERIFY``, ``FOLD_CONSTANTS``, ``REMOVE_UNUSED``, ``FOLD_PHASES``,
   ``FUSE_PULSES``); ``()`` runs none. The members are ``VERIFY`` (check the
   IR), ``FOLD_CONSTANTS`` (compute constant arithmetic at compile time),
   ``REMOVE_UNUSED`` (remove operations whose results nothing uses),
   ``MERGE_WAITS`` (join constant waits in a row on one line),
   ``SIMPLIFY_CONTROL_FLOW`` (remove syncs that order nothing, inline blocks
   that take no time, remove loops that run once or never), ``FOLD_PHASES``
   (fold frame changes into drives), ``FUSE_PULSES`` (merge square pulses in
   a row) and ``HOIST_LOOP_INVARIANTS`` (move loop-invariant operations out
   of loops). An unknown name is a ``ValueError`` that names the closest. See
   :doc:`../architecture/mlir_passes`.

``control_system`` *(ControlSystem or Controller, optional)*
   The control system to compile for: every program is checked against its
   controllers' wiring, capabilities and limits, and its links and the
   lanes of the RCP schedules set the schedule. One ``Controller`` stands
   for a system of that controller alone; what it cannot take (a line it
   does not play, a link it does not have) is an error that says to
   describe the whole ``ControlSystem``. By default the quantum system's control
   system, if it has one.

``quantum_system`` *(QuantumSystem, optional)*
   Pin the compiled kernel to one quantum system. A call with handles of another
   quantum system, even an equal one, raises ``QuantumSystemMismatchError``; a call without
   handles runs on it.
   A pinned kernel whose parameters are all ``float`` is built when it is
   compiled, so ``.mlir()``, ``.schedule`` and ``.metrics`` work on the
   compiled kernel itself.

Calling the kernel directly always uses the defaults: ``k(...)`` takes only
the kernel's own arguments.

Per-kernel schedules
--------------------

A kernel declared with ``@pulse.kernel(schedule="asap")`` or
``schedule="alap"`` and called from another kernel is scheduled on its own
with that policy, then placed whole by its caller's policy, so its internal
timing is the same wherever it is called -- also inside a rolled loop. A
called kernel without ``schedule=`` is scheduled together with its caller.
``rcp`` and ``alap_rcp`` apply to a whole program: they are accepted only by
``pulse.compile``, and not yet for programs that call kernels with their own
``schedule=`` (compile with ``flatten_schedules=True`` instead); that
``CompilationError`` has the code ``ErrorCode.SCHEDULE_BLOCK``, so a tool that
compiles programs it did not write can retry with ``flatten_schedules=True``
on the code, without reading the message. A kernel's
policy is part of what it is, so a compiled kernel's specializations depend
on its own policy, ``flatten_schedules`` and the policies of the kernels it
calls. The shape holds when the program is scheduled again, with float
values bound or, in simulation, with the answers of classical calls bound.

Duration budgets
----------------

``@pulse.kernel(max_duration_cycles=n)`` makes compiling fail if any
execution of the kernel can take more than ``n`` clock cycles. On a kernel
called with its own ``schedule=``, it bounds each call, which has one length
wherever it is called: the body of a loop that must fit its cycle is checked
once, whatever the trip count. On the kernel compiled, it bounds the whole
program. A wait whose length is known only when the program runs counts at
its highest length, so a wait without ``bounds=``, or a blocking classical
call, cannot be inside a budget.

.. code-block:: python

   @pulse.kernel(schedule="asap", max_duration_cycles=2000)
   def round_(data, ancilla):
       ...

   @pulse.kernel
   def repeated(data, ancilla):
       for _ in range(10_000):
           round_(data, ancilla)

Compiling ``repeated`` fails with the length of ``round_`` when it exceeds
2000 cycles. A kernel with a budget but without ``schedule=`` is scheduled
with its caller, so its calls have no length of their own: compiling a
caller is an error that names the fix, as is ``flatten_schedules=True``
(code ``ErrorCode.SCHEDULE_BLOCK``).

Inspecting a compilation
------------------------

``kernel.lower(*args)`` and ``compiled.lower(*args)`` return a ``ScheduledKernel``,
the module of one call:

``.mlir(level="pulse")``
   MLIR text at an IR level (``cudaq_pulse.compiler.IRLevel``, or its
   name): ``"pulse"`` for the scheduled pulse dialect (with the call's float
   values substituted), and ``"qop"``, ``"cudm"`` and ``"llvm"`` after
   lowering it that far along Pulse -> QOp -> CuDensityMat -> LLVM.

``.schedule``
   The drives, readouts and waits with their start and duration in clock
   cycles (``ScheduledEvent``). ``event.start`` and ``event.end`` are the
   scheduled times as ``int``; they raise ``ValueError`` on an event that is
   not scheduled yet, whose ``start_cycles`` and ``duration_cycles`` are ``None``.
   Up to 10,000 events it is a list; past that, a sequence that builds each
   event when it is read, with an exact ``len()``, so a rolled loop of many
   iterations is listed without holding every event.

``.duration``
   How long the program runs, as a time: the latest end of its operations,
   over both arms of every branch (``float(duration)`` is in ns). Only a
   ``ScheduledKernel`` has one: a kernel or ``CompiledKernel`` has no duration until
   it is lowered for a call, since the call's arguments decide it.

The quantum system's handles find a qudit by its label, ``qs.qudit("D1")``, and the
quantum system gives the coupling of two of them, ``qs.coupling(a, b)``.

``.timeline(figsize=(14, 4), title=None)``
   A matplotlib figure of the schedule, one row per line.

``.metrics``
   A ``CompileMetrics`` with per-stage timing in milliseconds (``trace_ms``,
   ``ffi_ms``, ``passes_ms``, ``schedule_ms``, ``total_ms``), ``op_count``
   and the quantum system name.

``.arguments`` and ``.generic``
   The call's float values, and the module without them, in which the float
   parameters are function arguments: ``generic.bind(values)`` makes a call
   with new values. A module that holds no values is its own ``generic``.

.. code-block:: python

   program = rabi.lower(q, 0.3, duration, sigma).generic
   for amplitude in (0.1, 0.2, 0.3):
       result = pulse.simulate(program, amplitude, t_end=20 * pulse.ns)

``.rescheduled``
   Whether a call's float values set a wait length, so its times come from
   scheduling the module again with them bound (see :ref:`wait-lengths`),
   read-only: ``False`` for a call whose values move no op, which keeps the
   module's times, and for a module scheduled without values. Reading it
   does no work.

``.rescheduled_for_line_delays``
   Whether the module was compiled again with only other line delays,
   scheduling its kernel's earlier trace again instead of tracing it (see
   :ref:`line-delays`), read-only; a call reads its module's. It does not
   depend on ``.rescheduled``: a call of such a module whose float values
   set a wait length is both. A loop that compiles each round can tell which
   path each round took, and reading it does no work.

Threads
-------

Calls and cache hits take no lock, and the first call for new static
arguments compiles once even when many threads make it at the same time.
The native stages (module construction, passes, scheduling, lowering and JIT
compilation) release the GIL. Simulations of one specialization from several
threads are safe: each sets the module attributes it lowers with and reads
the lowering in one step.

Long Sessions and Memory
------------------------

Compiled kernels share an MLIR context, which keeps every attribute it has
created (constants, qubit frequencies, timing) until it is freed. A context
is retired, and freed once its last kernel is gone, after it has built a
number of modules or a number of packed program words, whichever comes
first. The limits are read from the environment each time a kernel is
compiled:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Variable
     - Default
     - Meaning
   * - ``CUDAQ_PULSE_CONTEXT_MAX_MODULES``
     - 256
     - Modules built in one context before it is retired.
   * - ``CUDAQ_PULSE_CONTEXT_MAX_WORDS``
     - 1048576
     - Packed program words (8 bytes each) built in one context before it
       is retired.

Each must be a whole number from 1 to 2\ :sup:`40`; any other value makes
compiling raise ``ValueError`` naming the variable. Lower values bound the
memory a long session keeps (a loop that compiles a large program for many
quantum systems, for example) at the cost of creating contexts more often, about
0.4 ms each. ``python -m cudaq_pulse.doctor`` prints the limits in effect,
the process quantum system and the kernel cache size.
