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
  ``float`` parameter, or in a runtime ``list[float]``, is a ``TypeError``;
  see :ref:`durations`.
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
or an attribute the kernel reads off a dataclass or other object
(``p.amplitude = 0.5``), also compiles the kernel again on the next call. An
object the kernel passes on, or calls a method of, is compared at all its
attributes. An array passed as a static argument is copied when the kernel
compiles and compared with that copy on every call, at a cost that grows with
its size; a runtime ``list[float]`` parameter is neither copied nor compared.

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

.. themed-figure:: figures/schedule_policies.svg
   :alt: Four timelines of one kernel on the drive lines of D1, D2 and D3: an X pulse on D1, one on D2 and two on D3, then a sync of the three lines. asap: D1 at 0 ns, D2 at 0 ns, D3 at 0 and 40 ns, done at 80 ns; alap: D1 at 40 ns, D2 at 40 ns, D3 at 0 and 40 ns, done at 80 ns; rcp: D1 at 0 ns, D2 at 0 ns, D3 at 40 and 80 ns, done at 120 ns; alap_rcp: D1 at 0 ns, D2 at 40 ns, D3 at 40 and 80 ns, done at 120 ns. asap and alap drive 3 lines at once; rcp and alap_rcp never more than 2. Each pulse's Q quadrature is drawn dashed where it is large enough.
   :width: 100%

   The four policies on one kernel: an X pulse on D1, one on D2 and two on
   D3, then a ``sync`` of the three lines, for a control system with two
   drive lanes (``max_concurrent_drives=2``) and no controllers. ``asap``
   starts each pulse as early as its line allows and ``alap`` as late, and
   without controllers neither is held to the lanes. ``rcp`` and
   ``alap_rcp`` place operations in program order and never move a later
   one ahead of an earlier one, so here D1 and D2 take both lanes first and
   D3's pulses wait, though an order starting with D3 would end at 80 ns.
   ``alap_rcp`` hands out lanes from the end in reverse program order, so
   D1 here gets the first slot. Dashed blue is a pulse's Q quadrature.

.. _idle-replacement:

Idle replacement
----------------

A schedule leaves qudits idle: a data qudit while an ancilla is read out, a
line that waits at a ``sync`` for a longer pulse on another line.
``compiled.replace_idle(rule)`` finds each such stretch that is at least a
threshold long, an idle window, and writes a kernel of yours into it, the
fill: a dynamical-decoupling sequence, an echo, whatever the qudit should do
instead of nothing. The fill is an ordinary kernel that takes the idle qudit
and the window's length in clock cycles:

.. code-block:: python

   import cudaq_pulse as pulse

   qs = pulse.quantum_system("transmon_surface17")
   data, ancilla = qs.qudit("D1"), qs.qudit("X1")

   @pulse.kernel
   def syndrome(data, ancilla):
       data_drive = get_drive_line(data)
       ancilla_drive = get_drive_line(ancilla)
       ancilla_readout = get_readout_line(ancilla)
       half = data_drive.get_calibration(pulse.gates.X90)
       drive(data_drive, half)
       drive(ancilla_drive, ancilla_drive.get_calibration(pulse.gates.X))
       sync(data_drive, ancilla_drive, ancilla_readout)
       readout(ancilla_readout, ancilla_readout.get_calibration(pulse.gates.READOUT))
       sync(data_drive, ancilla_drive, ancilla_readout)   # the data qudit idles until here
       drive(data_drive, half)

   @pulse.kernel
   def echo(qudit, window):
       line = get_drive_line(qudit)
       drive(line, line.get_calibration(pulse.gates.X))

   compiled = pulse.compile(syndrome)
   filled = compiled.replace_idle(pulse.IdleReplacement(echo, threshold=100 * pulse.ns))
   program = filled.lower(data, ancilla)
   print(program.idle_windows)

.. code-block:: text

   qudit  start  end  cycles  rounds  before                           result
       0     80  480     400       1  sync() at line 16 of syndrome()  filled by echo()

The data qudit's line now holds a wait from cycle 80 to 240, the echo from
240 to 320 and a wait to 480; every operation of ``syndrome`` starts where
it did, and the program is as long as it was (280 ns).

.. themed-figure:: figures/idle_window_fill.svg
   :alt: Two timelines of the syndrome kernel on D1's drive line and X1's drive and readout lines. As compiled, D1 has an X90 pulse, then an idle window from 40 to 240 ns while the ancilla is read out, then a second X90 pulse. After replace_idle, the window holds a wait, the echo pulse from 120 to 160 ns and a wait, and every other pulse starts where it did. The X and echo pulses have a small Q quadrature, drawn dashed.
   :width: 100%

   The schedule of ``syndrome`` as compiled and with the idle window
   filled. The axis is in nanoseconds, and one nanosecond is two cycles
   of the system's 2 GHz clock: the shaded stretch, 40 to 240 ns, is the
   window of the report above, cycles 80 to 480. Dashed blue is a pulse's
   Q quadrature, where it is large enough to draw.

**Windows.** A qudit idles while its drive line has no operation, its
line's tone does not play from another line (a cross-resonance pulse at its
tone makes it busy), and its own readout does not run, in any iteration of
the loops around them. A window is such a stretch of at
least ``threshold``, an ``int`` of clock cycles or a time
(``100 * pulse.ns``); a stretch of exactly the threshold is one. A window
ends at the next operation on the line or at the qudit's tone, where a
``sync`` of the line meets, and where the kernel changes the tone's frame.
A ``wait`` the kernel writes is an operation and is kept: only the gaps the
schedule made are windows. The idle before a qudit's first operation and
after its last is left alone unless the rule says ``leading=True`` or
``trailing=True``, and ``qudits=[...]`` keeps the rule to some qudits. A
window in the body of a loop where the qudit has a drive or wait of its own
on its drive line is written once for every round, with the qudit's
operation before and after it in other rounds: it is neither leading nor
trailing, even when the first round's comes before the qudit's first
operation (as ``"alap"`` makes it when the qudit's operation ends the
round). In a loop the qudit sits out, its windows are leading or trailing
as the stretch around the loop is.

**The fill.** ``fill(qudit, window)`` is traced at each window as a kernel
called there, so it reads the qudit's calibrations and may compute its
spacing from ``window``. A window that the fill plays no pulse in (it
writes nothing, or only waits) is left alone, and its ``result`` says so.
A fill shorter than its window is completed with
waits: centred by default, or at the window's start or end with
``align="start"`` or ``align="end"``. A fill longer than its window is a
``CompilationError`` that gives both lengths; shorten the fill or raise the
threshold. In this version a fill drives and waits on its own qudit's drive
line, at that line's tone, and may shift the tone's phase by constants that
add up to zero; a readout, a ``sync``, a change of frequency, another line
or a classical call in a fill is a ``CompilationError``.

**Loops.** A window in a loop is written once, in the loop's body, and the
loop stays rolled: a thousand rounds compile as nine do. A ``sync`` of the
qudit's line in a loop ends a window in every round alike, so the stretch
between two syncs, or between a sync and the qudit's operation, is a window
of every round; a sync in a branch, one whose shift per round is not the
same in every round (as for a loop inside a loop of more rounds), or one in
a loop during which the qudit is read out or its tone plays from outside
the loop, does not, and the window is left alone with the reason. A window that runs
from a qudit's last operation of one round to its first of the next is
written at the end of the body and filled in every round, the last
included, if the qudit idles as long after the loop. Rounds that differ
only by more idle at one end take the fill of the shortest round.

**The report.** ``program.idle_windows`` lists every window with its
qudit, ``start`` and ``end`` in clock cycles, the ``rounds`` it stands for,
the operation it ends at, whether it was ``filled`` and the ``result``:
``"filled by echo()"``, or why it was left alone (before the qudit's first
operation, next to an arm of a branch, after a wait whose length is known
only when the program runs, rounds that differ, a loop that reads the qudit
out in more rounds than are listed, no statement of the kernel to write it
at). ``pulse.IdleReplacement(None, threshold=...)`` replaces
nothing: the program is the one compiled without the rule, with the report.

**What it guarantees.** The program is compiled twice: once without the
rule, whose schedule gives the windows, and once with the fills written in.
The two schedules are then compared: every drive, readout, wait and
``sync`` of the kernel has the start, length and loop periods it had, the
program is as long, and no pulse of a fill plays while its qudit is read
out or its tone plays from another line; if not, compiling fails and names
the operation. What the report says is what the program plays: compiling
checks that every fill plays a pulse and runs as many times as the rounds
of the windows it fills. A phase shift of the kernel in front of a window is folded,
as the passes fold it, into the pulse after it, which may be the fill's:
the fill plays in the shifted frame. Each gap is
written to the cycle, so one fill gives one program under every schedule
policy. Under ``"rcp"`` and ``"alap_rcp"`` a fill's pulses take drive lanes
like any pulse, and those policies give each lane one time it is free from:
a window whose fill would move an operation of the kernel there is left
unfilled, and its ``result`` says so. Under ``"asap"`` and ``"alap"``, fills
that take the control system past one of its limits (``max_concurrent_drives``)
make compiling fail with ``ControllerLimitError``, which names idle
replacement and the way out (``qudits=``, a fill with fewer pulses). A
program that ``"asap"`` or ``"alap"`` places past such a limit without its
fills is placed as ``"rcp"`` or ``"alap_rcp"`` places it, and its windows are
read from and filled under that policy.
``compiled.replace_idle(rule)`` returns a new compiled kernel and leaves
``compiled`` as it is; a kernel whose idle windows are not replaced runs
none of this. A program with filled windows pickles as the program it is,
with the options it was compiled with; ``compiled.replace_idle(rule)``
pickles as the compiled kernel and the rule.

Limits of this version:

- Under ``"rcp"`` and ``"alap_rcp"``, at most 64 windows are tried, each
  with the ones kept before it; the others are left alone as not tried.
- In rolled loops, a window can be left alone where its rounds differ (more
  or less idle, or something else on the qudit, in some rounds), or where
  the qudit idles less after the loop's last round than in the others.
- A fill's phase shifts must add up to zero; shifts that add up to a whole
  turn (2π) are refused too.
- A window ends at every ``sync`` of the qudit's line, whether or not the
  sync moves the line.

Not supported yet, each refused with a message: kernels whose qudits or
lists each call binds (parameters annotated ``Qudit``, ``Register`` or
``list``), a ``wait`` whose length a ``float`` parameter sets, and control
systems with line delays. Not seen yet: a pulse at the tone of another
transition of the qudit (``line.tones[(1, 2)]``) played from another line
does not make the qudit busy.

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

``.exposure``
   What the schedule does to each qudit over time, for estimating what its
   idle time costs. One row per qudit and per outermost loop the qudit has
   a pulse in, with the loop's rounds, and one for the rest of the
   program's length; an example follows. A qudit is driven while a pulse
   plays on its drive line or at its tone from another line; a ``wait`` is
   idle time, and a drive during a readout counts in both. Lines run on
   their own, so a loop the qudit has no pulse in takes none of its time.
   ``decay`` and ``dephasing`` are what the idle time of one round costs at
   the qudit's T1 and T2 star (see ``QuditExposure``); what a pulse costs
   is not in them. Loops stay rolled, so reading it costs the same for 1000
   rounds as for two. ``exact`` is ``no`` where the schedule does not fix
   the figures: a stretch that waits for an answer as long as it takes or
   has a wait of run-time length, a loop whose rounds overlap, whose
   operations repeat with different periods or that runs while the qudit is
   busy outside it, and a branch whose arms keep the qudit busy for
   different lengths, where the less busy arm is counted. It works on a
   program ``pulse.load`` read as on a compiled one, and is built when
   read: a program that never reads it loads none of it.

The exposure of a syndrome round, 1000 times:

.. code-block:: python
   :class: fragment

   @pulse.kernel
   def syndrome(data, ancilla, rounds: int):
       d = get_drive_line(data)
       a = get_drive_line(ancilla)
       r = get_readout_line(ancilla)
       for _ in range(rounds):
           drive(d, square(16, 0.1))
           drive(a, square(16, 0.1))
           sync(d, a, r)
           readout(r, square(200, 0.05))
           sync(d, a, r)

   program = pulse.compile(syndrome).lower(data, ancilla, 1000)
   print(program.exposure)

.. code-block:: text

   qudit  loop  rounds  cycles  drive  readout  idle  idle %  exact     decay  dephasing
       0     0    1000     216     16        0   200    92.6    yes  3.43e-03   5.69e-04
       9     0    1000     216     16      200     0     0.0    yes  0.00e+00   0.00e+00

A round is 216 clock cycles, of which the data qudit is driven for 16 and
idle for 200 while the ancilla is read out.

Loading a scheduled function
----------------------------

A program does not have to start as a kernel. ``pulse.load`` takes a module of
the Pulse dialect that another compiler wrote, as assembly or bytecode, and
returns what ``kernel.lower(...)`` returns: a ``ScheduledKernel``, with its
``schedule``, ``timeline()`` and ``export_to_controller()``, that the
backends take.

.. code-block:: python

   from pathlib import Path

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   text = Path("round.mlir").read_text()
   program = pulse.load(text, entry="round", qudits=[system.qudits[2], system.qudits[0]])
   program.export_to_controller()

The function named by ``entry`` takes one ``!pulse.qudit`` per qudit it plays
on, gets its lines from them and is scheduled: every timed operation has its
``start_cycles`` and ``duration_cycles``, in cycles of the quantum system's
clock. ``qudits`` gives the qudit of each argument, in order, and the lines
get that system's frequencies.

``load`` keeps the schedule as written and does not schedule again. It does
check it: on its qudits and control system the function is verified as every
scheduled program is, and checked against what the controllers can play. A
module that states its clock must state the quantum system's. Attributes
another compiler put on the function or its operations are dropped. The
function plays on the qudits it takes and allocates none. A control system
with line delays, or a controller with ``max_instructions`` or
``max_program_duration``, is refused: ``load`` does not apply or check those
yet.

A module given to ``load`` is another compiler's and is treated as untrusted
input. Nested modules are refused; the entry and the functions it calls are
those of the module's own symbol table; what its calls would write in place
is counted first and refused above 2\ :sup:`20` operations; calls nest at most
256 deep; regions nest at most 256 deep counting the module's and the
function's own, which leaves 254 branches or loops nested in a function,
and the same once its calls are written in place;
starts, durations and periods are counts of 0 to 2\ :sup:`50` cycles, and so
is an operation's start in the last trip of its loops, which are ``scf.for``
loops that count signed, with bounds and a step that are index constants of at
most 2\ :sup:`50`; and the
module ``load`` writes is verified before anything reads it. The text or
bytecode itself is parsed by MLIR as it is given, and the depth of that parse
is not bounded: input nested tens of thousands of levels deep can exhaust the
stack and end the process. Do not load a module from a source you would not
run code from. The same holds for unpickling a compiled program.

A loaded function may hold loops, branches on measurements and calls of
classical functions (:doc:`classical_calls`). The classical functions are the
module's ``classical.callable`` declarations: their types, latency, link and
dispatch key are enough to export the program. The links are the control
system's: the module names each link its calls travel on (``name`` in its
``classical.links``), and ``links=`` gives the control system's name for a
link the module names otherwise, or only numbers
(``links={"host": "nvqlink"}``, ``links={0: "nvqlink"}``). A link the control
system does not have is refused. An answer taken before the latency and the
link allow is refused, and so is a submission given less time than the link
takes. The dispatch keys are the control system's too, as for ``compile``: a
key a controller fixes for a function replaces the module's, and a reserved
key or two functions on one link and key are refused. A function named as
``compile`` names a version of one of CUDA-Q QEC's decoder messages
(``enqueue_syndromes__3_2``: decoder 3, two measurements;
``get_corrections__3_1``; ``reset_decoder__3``) is that message, as a kernel
that calls ``pulse.classical(cudaq_qec.qec.get_corrections, ...)`` compiles
it: it keeps the decoding server's key, and the configured decoder answers it
in simulation. Such a name is never an ordinary function. The module is held
to what ``compile`` writes for the message: its key and types, a message of
a CUDA-Q QEC release, each submission sending the decoder id and the length
its name states as constants (and the mapping 0 for ``enqueue_syndromes``),
and, in a module that calls a decoder message, every answer taken
from its submission's own future (a future yielded from a branch or a loop
is refused there). A CUDA-Q QEC
function given in ``classical=`` gives the latency of the module's versions
of its message.

A round that only sends syndromes and takes the corrections once at the end
looks like this in a module (operations shortened to what matters):

.. code-block:: text

   classical.callable @enqueue_syndromes__3_2 {args = ["i64", "i64", "i64", "i64", "bits:2"], dispatch_key = 2128133762 : i64, ...}
   classical.callable @get_corrections__3_1 {args = ["i64", "i64", "u8"], dispatch_key = 2284673953 : i64, latency_max_ns = 1.0e+03 : f64, ...}
   func.func @main(%a: !pulse.qudit, %c: !pulse.qudit) attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
     ...
     scf.for %i = %c0 to %c4 step %c1 ... {
       %ra1, %ta1, %s0 = pulse.readout ... <{duration_cycles = 100 : i64, period_cycles = array<i64: 100>, start_cycles = 0 : i64}> ...
       %rc1, %tc1, %s1 = pulse.readout ... <{duration_cycles = 100 : i64, period_cycles = array<i64: 100>, start_cycles = 0 : i64}> ...
       %l1, %f = classical.submit %l, @enqueue_syndromes__3_2(%c3, %c7, %c0, %c2, %s0, %s1) <{duration_cycles = 100 : i64, period_cycles = array<i64: 100>, start_cycles = 100 : i64}> ...
       ...
     }
     %l2, %g = classical.submit %l1, @get_corrections__3_1(%c3, %c1, %c1) <{duration_cycles = 100 : i64, start_cycles = 500 : i64}> ...
     %fix = classical.wait %g, @get_corrections__3_1 <{duration_cycles = 0 : i64, start_cycles = 2800 : i64}> ...
     scf.if %fix ... { pulse.drive ... <{duration_cycles = 16 : i64, start_cycles = 2800 : i64}> ... }
     return
   }

.. code-block:: python

   from cudaq_pulse.classical import Link

   machine = pulse.ControlSystem(links={"host": Link(fixed_ns=50, feedback_ns=100)})
   program = pulse.load(
       text, entry="main", qudits=[system.qudits[0], system.qudits[2]], control_system=machine
   )
   [
       (c["function"], c["submit_cycles"], c["taken_cycles"])
       for c in program.export_to_controller()["calls"]
   ]

.. code-block:: text

   [('enqueue_syndromes__3_2', 100, None), ('enqueue_syndromes__3_2', 200, None),
    ('enqueue_syndromes__3_2', 300, None), ('enqueue_syndromes__3_2', 400, None),
    ('get_corrections__3_1', 500, 2800)]

The four rounds send their syndromes and wait for nothing, and the loop stays
rolled. The corrections are asked for once, at 500, and used at 2800, when the
decoder's latency and the link allow.

``classical=`` gives functions declared with
``pulse.classical`` for the module's, by name. Each must take and give what
the module states, with its latency mode and submit policy, and its latency
and implementation replace the module's:

.. code-block:: python

   @pulse.classical(max_latency_ns=400, link="host")
   def decide(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
       return pulse.Bits([bits[0]])

   qudits = [system.qudits[0], system.qudits[2]]
   program = pulse.load(text, entry="main", qudits=qudits, classical=[decide])

A call of another function of the module is a placed ``pulse.call``: it
states its start and its length (``start_cycles`` and ``duration_cycles``,
with one ``period_cycles`` entry per loop around it), and is written in
place, the callee's operations each that much later. The callee takes and
returns the lines and tones it plays on, is scheduled, records its length
(``pulse.length_cycles``) and holds no loop; a call that lasts longer than
its callee holds the callee's lines until its end. A ``func.call`` is
refused.

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
quantum systems, for example) at the cost of creating contexts more often, a
fraction of a millisecond each. ``python -m cudaq_pulse.doctor`` prints the limits in effect,
the process quantum system and the kernel cache size.
