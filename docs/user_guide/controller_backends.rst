..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Writing a Controller Backend
============================

A controller backend runs compiled kernels on a quantum controller: it
lowers a scheduled program to the controller's own instructions and
waveform memory, and runs it there. Backends live outside cudaq-pulse, in
their own distributions, and cudaq-pulse finds the installed ones by a
standard Python entry point. This page shows the program a backend
receives, how a backend states what its controller can play, and a complete
minimal backend; ``examples/controller_backend/`` is the same backend as an
installable distribution.

Exporting a program
-------------------

``ScheduledKernel.export_to_controller()`` returns a compiled kernel as JSON-serializable
data. Its ``format`` is ``"cudaq-pulse.controller/1"``
(``cudaq_pulse.controller.FORMAT``): the schema's name and, after the slash,
its major version. Within one version keys and event kinds may be added, but
none is removed or changes meaning, so a backend ignores keys it does not
know and refuses event kinds it does not know; a change that would break a
backend gets a new version.

At the top level, ``quantum_system`` and ``clock_ghz`` name the quantum system and its
clock, ``schedule`` the schedule policy, ``qudits`` the quantum system qudits the
program uses, ``lines`` each line's index, ``role`` (``"drive"`` or
``"readout"``) and qudit (and, with line delays, its ``fine_delay_ns``;
see below; on a shared line, the line's ``qudits`` and
``max_concurrent_tones``; see `Shared lines`_), and ``limits`` the controller limits it was
checked against (``null`` without). Each line also has
``native_frequency_hz``, the frequency of its own tone (``line.tone``):
its qudit's 0-1 frequency on a drive line, and its readout's
``frequency_hz`` (else the 0-1 frequency) on a readout line. Every shot
starts that tone's frame at this frequency, and its ``phase_rad`` advances
from it when the tone is retuned, so a backend that plays a line's other
frequencies (another transition's tone, a shifted frequency) as offsets
from one oscillator needs nothing from the quantum system. On a system
that does not state its frequencies they are offsets:
``quantum_system.frequency_frame`` is ``"shared_reference"`` when every
``frequency_hz`` and ``native_frequency_hz`` is an offset from one reference
the qudits share, and ``"own_carrier"`` when each line's native frequency
is 0 and its events' frequencies are offsets from its own qudit's carrier
(see :doc:`quantum_systems`); the export's keys are the same. A kernel with
qudit parameters gives it as the slot's symbols instead (``s0.f01``,
``s0.f_ro``; see below) and its lines leave it out. ``events`` lists every drive, readout,
wait, sync, classical call and frame update in program order, once per loop
iteration and both arms of a branch. ``segments``, ``calls``, ``links`` and
``manifest`` are those of ``ScheduledKernel.export_to_emulator()`` (see
:doc:`classical_calls`).

A kernel with list or qudit parameters compiles once for every call, and
its export says what each call binds: ``program_id`` is the same for every
call of that compile, and ``symbols`` lists the call's values, each with
its ``symbol``, ``kind`` and ``value``: a float parameter or element
(``"float"``), a runtime qudit's slot (``"slot"``, ``s0``, the carrier
bound), and the native frequencies of the slot's lines (``"frequency"``):
``s0.f01``, its 0-1 frequency, for a drive line, and ``s0.f_ro``, the
``frequency_hz`` of its readout (else the 0-1 frequency), for a readout
line. Each event names in its ``symbols`` the slot of its line
(``qudit``) and, by field (``frequency_hz``, ``phase_rad``,
``waveform``), the symbols its value depends on; a tone on a slot's line
depends on that line's native frequency. Each event has:

``index``, ``kind``
    Its position in ``events`` and one of ``"drive"``, ``"readout"``,
    ``"wait"``, ``"sync"``, ``"submit"``, ``"answer"`` and
    ``"frame_update"``.
``line``, ``qudit``
    The line it plays on and that line's qudit; ``null`` for a sync. For a
    classical call, ``line`` is its link and ``qudit`` is -1.
``tone_slot``
    Only on a drive, readout, wait or frame update on a shared line (see
    `Shared lines`_): the line's tone it plays, as in ``/2``.
``start_cycles``, ``duration_cycles``, ``end_cycles``
    The final schedule as integers, in clock cycles. A backend places every
    event at its start and does not schedule again.
``iteration``
    The iteration of each loop around it, outermost first.
``segment``
    The segment it starts in. A wait whose length is an answer, and a blocking
    wait for an answer, end one segment of the lines that wait for it and
    start the next; an event shifts by the run-time lengths of its segment's
    barrier and of the barriers of the segments it follows. Each segment follows
    the one before it, or the segment its ``after`` names.
``conditions``
    The measurement-conditioned branches the event lies in, outermost
    first: the ``branch``, the ``arm`` (``true`` for the ``if`` arm), and
    the ``readouts`` and ``answers`` (event indices) the condition is
    computed from. ``test`` is ``"is_set"`` when the condition is whether
    one readout is set, ``"answer"`` when it is one answer value, else
    ``"expression"``, and then ``expression`` is the condition, a bool
    expression (see `Run-time expressions`_) over those readouts and
    answers. A condition with ``test`` ``"answer"`` also has ``result``,
    which result of that answer it is, numbered as in an expression's
    ``{"answer": i, "result": k}``: for a call with a timeout, result 0 is
    the flag that the answer came in time and result ``k + 1`` the
    function's value ``k``. The event plays when every condition is its
    ``arm``. Events in the two arms of one branch may overlap in time; only
    one arm plays.

Drives and readouts also have:

``waveform``, ``amplitude``
    The waveform's expression, ``{"shape": "gaussian", "params":
    {"duration": ..., "amplitude": ..., "sigma": ...}}``, with ``scale``,
    ``neg``, ``add``, ``sub`` and ``mul`` nodes over ``of``, and
    ``samples`` for ``custom_samples`` (which ``custom()`` envelopes become
    when the kernel is traced), at the line's own rate (its controller's
    ``sample_rate_ghz``); and its overall complex amplitude,
    ``{"re": ..., "im": ...}``. Times in parameters are in cycles.
``samples``
    With ``export_to_controller(sample_rate_ghz=...)``, a whole multiple of the
    clock, the envelope sampled at that rate, ``{"re": [...], "im":
    [...]}``. ``cudaq_pulse.controller.sample_waveform`` computes the same
    samples for one waveform. Sample ``k`` of a shape (``gaussian``,
    ``drag``, ...) is its envelope at the start of that sample, ``k``
    samples from the pulse's start, as the simulator plays it; a
    ``custom()`` envelope was sampled at the middle of each sample when the
    kernel was traced, and ``custom_samples`` are played as given. An
    instrument that evaluates a shape at the middle of each sample plays
    these samples with the shape delayed by half a sample.
``frequency_hz``, ``phase_rad``
    The tone's frequency and its frame phase as the event starts, with every
    frame change and virtual Z before it applied and the phase a detuned
    tone gains over time, or ``null`` where the two arms of a branch leave
    them different or a frame change is known only at run time.
``tone``, ``frame_updates``, ``phase_offset_rad``
    Only on a tone whose frame depends on run time: one with a frame change
    computed from a readout or an answer, or made in a branch. The tone's
    id (such tones are numbered in the order they are first played), and
    the frame changes on that tone since its previous drive or readout, in
    program order: each with ``id``, ``op`` (``"shift_phase"``,
    ``"set_phase"``, ``"shift_frequency"`` or ``"set_frequency"``),
    ``value`` (a number, or an expression), ``start_cycles``, where it is
    made on the tone's timeline (see below), from which the frame gains
    phase at its new frequency even where the value is known only later,
    and ``conditions``, the branch arms it is on the path of. One change
    pending before a branch whose arm
    plays on the tone is listed again, with the other arm's condition, at
    the next drive after the branch; the same ``id`` names it both times.
    A controller tracks such a tone's frame itself, applying these instead
    of reading ``frequency_hz`` and ``phase_rad``, each ``id`` once. A drive
    whose own phase offset does not change the frame has ``phase_offset_rad``.

A program with such tones has ``runtime_tones`` (after ``events`` and
``frame_state``): by each
tone's id, as a string, its ``line`` (a line's own tone's line, else the line
it first plays on) and the frame every shot starts it in,
``reset_frequency_hz`` and ``reset_phase`` (in radians), to which a
controller tracking the tone applies its ``frame_updates``. On the line of a
runtime qudit, the entry's ``symbols`` name the symbol its reset frequency
is, ``s0.f01`` on a drive line and ``s0.f_ro`` on a readout line. A
program without such tones has no ``runtime_tones``.

Every shot starts from the reset frame: each tone at its own frequency, with
phase 0. A frame change (``shift_phase``, ``set_phase``, ``shift_frequency``,
``set_frequency``) is a point on the timeline: in the scheduled IR it has
``start_cycles`` and a ``duration_cycles`` of 0. It is where the program
makes it on its tone's line: after the waits, pulses and syncs before it
there and after the readouts it depends on (in the arm of a branch on a
measurement, or with a value computed from one); or, on a tone that next
plays on another line earlier, where its previous drive or readout ends. A frequency shifted before a wait and back
after it so runs shifted through the wait, and the drives after it have
the phase gained. The next drive or readout on
its tone takes it in its ``frequency_hz`` and ``phase_rad``; in a loop, the
next iteration's do. In a loop whose iterations do not play its tone, a
change is at the start of each iteration. A change takes no time: every
drive, readout, wait and loop, and the program, starts and lasts as it does
without it.

The compiler keeps every change the program makes, and where no later
drive or readout takes a change on some path, the export gives that
execution as a ``"frame_update"`` event of its own, decided from the
program alone:

- A change nothing takes is a frame update where it is made, in every
  iteration of a loop around it; a loop kept rolled stays rolled, the frame
  update in its body.
- A change a loop carries out of its last iteration, which nothing after
  the loop takes, is a frame update after the loop, at its place in the
  last iteration.
- A change made before a branch that only one arm takes is a frame update
  at the start of the other arm (or of the missing ``else``), with that
  arm's ``conditions``; the arm that takes it plays with it as before.
- A change a loop carries through a branch, or out of a loop whose
  iterations do not play its tone, is a frame update at every execution;
  a drive or readout that takes it on another path has it in its frame as
  well.

A frame update has ``op`` and ``value``, as in ``frame_updates``, its
``start_cycles`` (its place) and a ``duration_cycles`` of 0, and the tone's
``frequency_hz`` and ``phase_rad`` with the change applied; on a tone whose
frame depends on run time, ``tone`` and the ``frame_updates`` it applies,
as a drive has. A controller tracking that frame applies the event's
``frame_updates``, each ``id`` once, as it does a drive's; the last of them
is the event's own change, which its ``op`` and ``value`` restate, so it is
not applied again. Its ``line`` is its tone's, else the line the tone last
played on, else ``null``. A controller whose frames persist from one shot to the next
declares ``Controller(persistent_frames=True)``; such changes then
have no event, and the program's ``frame_state`` (after ``events``) gives
the frame each tone ends the shot in, by line: its ``line``, and its
``frequency_hz`` and ``phase_rad`` at the end of the shot, or, for a tone
whose frame depends on run time, its ``tone`` and the ``frame_updates`` not
yet taken.

A waveform parameter computed from a readout or an answer is ``null`` in
``params`` and has an expression (see `Run-time expressions`_) in the
node's ``runtime``, by parameter name. A variable
wait's ``duration_cycles`` is its least length, and it also has ``length``,
the expression of its length in cycles, and ``min_cycles`` and
``max_cycles``, its bounds. A program that computes nothing from its
readouts and answers exports none of these keys.

A sync has ``lines``, the lines it joins, and lasts 0 cycles: no event on
those lines before it ends after its start, and none after it starts
before.

A classical call is a ``submit``, while the request holds its link, and an
``answer``, when the answer is taken. Both have ``link`` and ``function``;
a submit has ``measurements``, the readout events whose outcomes it sends,
and ``operands``, its arguments in the order of the function's argument
descriptors (one per scalar or element, and one ``{"readout": i}`` per
measurement), each an expression; an answer has ``wait``,
``"bounded"`` (by the latency budget), ``"blocking"`` or ``"timeout"``,
with ``timeout_ns``. A call taken with ``result(timeout=...)`` waits
with that timeout, and one of a function declared with
``latency_mode="timeout"`` (and taken without one) with the function's
``timeout_ns``.

The events of one operation, and of operations with the same waveform,
share their ``waveform`` and ``amplitude`` objects; copy one before
changing it.

Run-time expressions
--------------------

What a program computes from its readouts and answers (a run-time waveform
parameter, a wait's ``length``, a frame update's ``value``, a submit's
``operands``, a branch's ``expression``) is an expression in a closed
vocabulary of terms, whatever operations the compiler computes it with:
``y = x if x > 0.1`` (else ``0.1``) and ``y = x if 0.1 < x`` are both
``max``. An expression is one of:

``{"const": x}``
    A number, or ``true`` or ``false``.
``{"readout": i}``
    The level the readout event ``i`` observed, an integer; 0 is the
    ground state, and the outcome is set for any other level.
``{"answer": i, "result": k}``
    Result ``k`` of the answer event ``i``: a float for a ``float``
    result, an integer for an ``int`` one, a bool for a bit. A call taken
    with ``result(timeout=...)``, or of a function declared with
    ``latency_mode="timeout"``, has the bool that its answer came in time
    as result 0, before the function's values.
``{"op": term, "type": t, "of": [...]}``
    A term over the expressions in ``of``, computed in the type ``t`` of
    its operands, ``"int"``, ``"float"`` or ``"bool"``. A ``const`` in a
    float term is that float, so ``{"const": 3}`` there is 3.0.
``{"term": n}``
    Term ``n`` of the expression's ``terms``, where that term is used.

An expression that is a term may have ``"terms": [...]`` beside its
``op``, ``type`` and ``of``: terms written once there and used by their
place in the list, the first being term 0. A term is in ``terms`` when
the expression uses it more than once, or when the expression would
otherwise nest more than 65 terms deep; every other term is written where
it is used. A term of ``terms`` uses only terms before it, so computing
the list in order computes each term once. For ``t = x * 0.001`` (``x``
result 0 of answer event 1), ``t = t + t`` and a phase shift of ``t + t``:

.. code-block:: json

   {"terms": [
      {"op": "mul", "type": "float", "of": [{"answer": 1, "result": 0}, {"const": 0.001}]},
      {"op": "add", "type": "float", "of": [{"term": 0}, {"term": 0}]}],
    "op": "add", "type": "float", "of": [{"term": 1}, {"term": 1}]}

``terms`` is only at the root of an expression and belongs to all of it;
code that makes an expression an operand of a new term moves the list to
the new root. The compiler writes no expression that nests more than 65
terms deep, however many terms the program computes it with, which is the
deepest it wrote one before there were ``terms``; the length of a wait on
a controller with a clock of its own is up to three terms more (its
bounds, and the conversion to the controller's cycles), as before. An
expression has no ``terms`` unless it uses a term more than once or is
deeper than that: an export that held no such expression is written as
before.

``cudaq_pulse.controller.ExpressionTerm`` names every term. A value
computed any other way is an ``UnsupportedError`` when the program is
exported, never a term a backend has to guess at;
``cudaq_pulse.controller.evaluate_expression(expression, readouts,
answers)`` computes an expression as the controller must, which is the
reference for a backend's own evaluation. Integers are 64-bit two's
complement and floats IEEE 754 binary64, rounded to nearest, ties to even.

``add``, ``sub``, ``mul``
    ``a + b``, ``a - b`` and ``a * b``. Integers wrap modulo 2\ :sup:`64`;
    floats overflow to ±inf.
``div``
    ``a / b``. An integer quotient is truncated toward zero; a float
    divided by zero is ±inf, and 0 / 0 is NaN.
``floor_div``, ``rem``
    Integers only: the quotient rounded toward −inf, and the remainder
    ``a - b * div(a, b)``, which has the sign of ``a``. An integer ``div``,
    ``floor_div`` or ``rem`` by 0, or of −2\ :sup:`63` by −1, has no value,
    and the controller fails the run.
``neg``
    ``-a``. The integer −2\ :sup:`63` wraps to itself; a float changes
    sign, 0 and NaN too.
``max``, ``min``
    ``max(a, b)`` is ``a`` when ``a > b``, else ``b``, and ``min(a, b)`` is
    ``a`` when ``a < b``, else ``b``. For integers that is the usual
    maximum and minimum, which the export writes with the operands of the
    comparison it comes from. For floats the order is part of the value,
    as the kernel's comparison was: ``max(NaN, 1)`` is 1 and ``max(1,
    NaN)`` NaN, ``max(-0.0, 0.0)`` is 0.0 and ``max(0.0, -0.0)`` −0.0.
``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``
    A bool: ``a == b``, ``a != b``, ``a < b``, ``a <= b``, ``a > b``, ``a
    >= b``. Integers compare signed. Floats compare ordered, so every
    comparison with a NaN is false but ``ne``, which is true. Bools have
    only ``eq`` and ``ne``.
``select``
    ``of`` is ``[c, a, b]``: ``a`` when the bool ``c`` is true, else ``b``;
    ``type`` is the type of ``a`` and ``b``.
``to_float``
    The integer ``a`` as the nearest float (``type`` ``"int"``).
``to_int``
    The float ``a`` truncated toward zero (``type`` ``"float"``); NaN, ±inf
    and floats outside the 64-bit integers have no value, and the
    controller fails the run.
``controller_cycles``
    A length of ``a`` program cycles in the controller's cycles, ``a *
    num / den``, with ``num``, ``den`` and ``timestep`` beside ``of`` (see
    `Clocks`_); the controller fails the run when ``a`` is
    not a whole number of timesteps.

A frontend computes an integer answer used as a time in floats, so
``wait(d, n + 4)`` is ``to_int(add(add(to_float(n), 4), 0.5))``. Whether
a readout is set is ``ne(readout, 0)``, and a bool used as an integer a
``select`` of 1 (or −1) and 0. A comparison has its constant second:
``3 < n`` is ``gt(n, 3)``. Bools combine without terms of their own:
``a and b`` is ``select(a, b, false)``, ``a or b`` is ``select(a, true,
b)``, ``a != b`` (exclusive or) is ``ne(a, b)``, and ``not a`` is
``ne(a, true)``, except that ``not`` of an integer comparison is the
opposite comparison (``not (n <= 3)`` is ``gt(n, 3)``) and of ``eq`` or
``ne`` the other; a float ``not (x <= 0.1)`` stays as it is, since it holds
of NaN and ``x > 0.1`` does not. There is no JSON schema; this section and
``evaluate_expression`` define the expressions.

The structured format
---------------------

``export_to_controller(version=2)`` gives format
``"cudaq-pulse.controller/2"``, which grows with a program's operations
rather than with its executions, as a sequencer runs it. It is written for a
controller on the program clock, in :ref:`timesteps <timestep>` of any
length; for one on its own clock, or with line delays, export
``/1``. It has the keys of ``/1``, except that:

``events``
    Each event's ``kind`` is one of ``cudaq_pulse.Event``, named for what a
    controller does whatever the modality: ``"play"`` (a ``/1`` drive),
    ``"readout"`` (a ``/1`` readout), ``"wait"``, ``"sync"``, ``"loop"``,
    ``"call"`` (a ``/1`` submit), ``"answer"`` and ``"frame_update"``. An
    event in a branch keeps its kind and lists its ``conditions``, and phase
    and frequency changes are in the ``tones`` of the events that play
    them; a change no event plays is a ``"frame_update"``, with its frame
    in ``tones``. A backend refuses a kind it does not know.

    A tree. A loop whose iterations repeat (each event moves by one period,
    and every tone keeps its frequency) is one ``loop`` event: ``trips``
    iterations from ``first_iteration`` (omitted when 0), one
    ``period_cycles`` apart, its ``start_cycles`` and ``duration_cycles``
    spanning them, and its ``body``, the events of its first iteration. An
    event in the body of loops starts ``start_cycles`` plus, for each loop
    around it, the iterations since its first times its period. A first
    iteration that differs from the rest is exported before its loop; a
    loop that does not repeat is exported once per iteration, as in ``/1``.
``waveforms``
    Each distinct waveform expression once; a play or readout gives the
    index of its own in ``waveform``. With ``sample_rate_ghz``, ``samples``
    holds each one's samples, in the same order. Each also has ``id``, 16
    hex digits that depend on its expression alone, so they stay the same
    from one compile, call, line or controller to the next, and
    ``shape_id``, equal for waveforms that differ only in overall amplitude
    and phase (``shape_key``), the ones a controller that scales what it
    plays stores once.
``parameters``
    For a module with ``float`` parameters, each one's ``value`` in this
    call and what it ``reaches``: ``"amplitude"`` (an overall factor of a
    waveform), ``"phase"``, ``"frequency"``, ``"shape"`` (another waveform
    parameter), ``"timing"`` (a length) or ``"other"``. A parameter that
    reaches only amplitudes and phases leaves every ``shape_id`` the same
    for every value. Omitted without parameters.
``controller``
    In place of ``limits``: the controller, its fields under the names
    ``limits`` gives them. What it computes is in ``processors``, always
    listed (empty for a controller without processors), each one as
    ``Processor.as_dict()`` gives it: ``links``, ``clock_ghz``,
    ``cycles_per_op`` (the cost of each class of operation it computes),
    ``unsupported`` (the classes it cannot compute, by name),
    ``max_concurrent_calls``, ``max_ops_per_function``,
    ``max_ops_total``, ``max_state_bytes_per_function``,
    ``max_state_bytes_total``, ``max_calls_per_shot``,
    ``max_bytes_per_shot``, ``fixed_cost_only`` and ``dispatch_keys``.
``quantum_system``
    In place of ``quantum_system``: the name of the quantum system the program is
    for.
``sample_rate_ghz``
    Beside ``clock_ghz``, always a number: the rate at which the
    controller's converters play waveform samples, so that the program
    states its own timing. A waveform of ``n`` samples lasts
    ``n * clock_ghz / sample_rate_ghz`` cycles. The ratio may be
    fractional (2.4 GHz on a 1 GHz clock), but every waveform lasts a whole
    number of samples, and the export raises ``ValueError`` for one that
    does not rather than round. The rate is the controller's, else the
    export's ``sample_rate_ghz``, which must then be the same, else
    ``clock_ghz`` (one sample per cycle). Unlike in ``/1``, it need not
    be a whole multiple of the clock, and ``samples`` are present only when
    the export is given ``sample_rate_ghz``.
``links``
    Each link with ``send_ns``, ``bandwidth_bytes_per_ns``,
    ``answer_delay_ns`` and ``slots`` for ``Link.fixed_ns``,
    ``bytes_per_ns``, ``feedback_ns`` and ``slots``.
``lines``
    Each line's ``qudits``, a list, in the order the program first uses the
    lines. A shared line (see `Shared lines`_) is listed once, with every
    qudit it reaches and its ``max_concurrent_tones``, which a dedicated line
    does not have. In place of ``native_frequency_hz``,
    ``native_frequencies_hz``: the native frequency of each of its
    ``qudits``, in the same order.
Events on a shared line
    A play, readout, wait or frame update on a shared line has, after its
    ``line``, its ``tone_slot`` (0 to the line's ``max_concurrent_tones``
    minus 1: the tone it plays, which a controller maps to one of the line's
    oscillators) and ``qudits``, the one qudit it is for. Tones that play at
    the same time have different slots. Slots are given from the program
    alone, so every export of it gives the same: a line playing one tone at
    once gives every event slot 0; else each qudit keeps one slot for the
    whole program when the line has enough for the qudits whose events
    overlap (lowest free slot first, in the order of their first events);
    else each tone takes the lowest free slot in start order, which a
    rolled loop cannot hold (``UnsupportedError``: unroll it, or give the
    line more tones). A sync lists each of its lines once; ``runtime_tones``
    and ``frame_state`` entries name the line and, on a shared line, the
    ``qudits`` (and ``tone_slot``) of their tone. On dedicated lines none of
    these keys are present, and a program that plays no shared line exports
    as it does on a system without them.
Broadcast lines
    A broadcast drive line (see `Broadcast lines`_) is listed once, with
    ``"broadcast": true``, every qudit it reaches, and one
    ``native_frequency_hz``, its tone's, in place of
    ``native_frequencies_hz``. A play or frame update on it has, after its
    ``line``, the line's ``qudits``, all of which it reaches, and no
    ``tone_slot``; so do its ``runtime_tones`` and ``frame_state`` entries.
Played events and frame updates
    ``tones``, a list of the tones the event plays (a frame update's
    frame), each with its ``frequency_hz``, its ``phase_rad`` in the loops'
    first iterations, and ``phase_step_rad``, the phase added per iteration
    of each loop around it, outermost first.
Segments in a rolled loop
    An event's ``segment`` is that of the loops' first iterations; in a loop
    whose body ends segments (a blocking wait for an answer, or a wait whose
    length is an answer), ``segment_step`` is the segments added per iteration
    of each loop around it, outermost first, omitted where every step is 0.
Values known at run time
    As in ``/1``, and only where a value is computed from a readout or an
    answer: ``tone``, ``frame_updates`` and ``phase_offset_rad`` on a play
    or readout, ``length``, ``min_cycles`` and ``max_cycles`` on a wait, and
    ``operands`` on a call; an expression's ``readout`` and ``answer`` are
    places, like other references, and resolve to the latest execution of
    that place before the event, so a rolled body refers to its own
    events, and a frame change's ``start_cycles`` is given as an event's
    is. A waveform parameter known only at run time is its expression
    in the entry of the program's ``waveforms`` the event plays, where
    ``/1`` has ``null`` and the node's ``runtime``. The program's
    ``runtime_tones`` and ``frame_state`` are as in ``/1``. A loop whose body changes or plays a tone whose frame
    is tracked at run time is exported once per iteration, so that each
    frame change keeps its own ``id``; a loop whose run-time values repeat
    with its iterations stays one loop.
Symbols
    A kernel with list or qudit parameters has ``program_id`` and
    ``symbols`` as in ``/1``, and each event the ``symbols`` its fields
    depend on; an event in a loop's body names those of any of its
    iterations.
Left out
    What follows from the rest or is the default: an event's ``index`` (its
    place in a pre-order walk of the tree, loops included, which is what
    references give), its ``qudit`` (its line's), its ``end_cycles``, an
    empty ``iteration`` (``null`` for a loop kept rolled), a ``segment`` of
    0, empty ``conditions``, ``readouts`` and ``answers``, the ``line`` of
    a sync or a classical call, a ``timeout_ns`` of ``null``, and the
    ``amplitude``, which follows from the waveform. A reference is to the
    latest execution of the event it names.
``programs``, ``bindings``
    Reserved for programs that several controllers share and the tables
    that bind them to each; empty for now.

``cudaq_pulse.controller.unroll(program)`` turns a ``/2`` program into the
``/1`` program of the same kernel, for a backend that needs every
execution; phases agree up to rounding and branches are numbered in the
order they first appear. A ``/2`` program lists a shared line once, so on
one, ``unroll`` numbers the lines in the order of the ``/2`` lines, each
shared line's qudits in its order (those the program plays at), where the
``/1`` export may number them in the order the kernel takes them, and a
sync names every ``/1`` line of a shared line it joins.

Clocks
------

A kernel counts time in cycles of the quantum system's program clock,
``QuantumSystem.clock_ghz`` (``pulse.quantum_system(..., clock_ghz=)``, else the
quantum system's own); a duration written as a time converts to them when the kernel
compiles (see :ref:`durations`). A controller may run its own clock,
``Controller.clock_ghz``, at a whole-cycle ratio of it; may start
and time operations only every ``Controller.cycles_per_timestep`` of
its own cycles (default 1); and may play samples at
``Controller.sample_rate_ghz`` (default one per cycle of its clock),
at any ratio of whole numbers to its clock.

.. _timestep:

A line's **timestep** is the shortest time, in program cycles, at whose
multiples its controller can start and end operations: the fewest whole
program cycles that are a whole number of the controller's steps and of
its samples. On the program clock with one sample per cycle it is
``cycles_per_timestep``; at ``a / b`` of the program clock (in lowest
terms), with one cycle per step, it is ``b``; at 2.4 GHz samples on a
1 GHz clock it is 5 cycles, 12 samples. A length off the samples is an
error like any other off its timesteps; it is never rounded.

The scheduler places every operation on whole timesteps of its line and
meets a sync on a multiple of every synced line's timestep; a length that
is not a whole number of timesteps fails to compile with
``ControllerLimitError``, which names the line and the fix:
``line d3 moves in timesteps of 4 cycles; 22 cycles isn't a whole number of
timesteps``. ``pulse.round_duration(line, duration)`` in a kernel rounds a
length to whole timesteps (``direction="up"`` by default, ``"down"`` or
``"nearest"``, or a ``pulse.RoundDirection``), so a kernel can be written
for any controller::

    drive(line, gaussian(pulse.round_duration(line, 41), 0.1, 10.0))

The export counts the controller's own cycles: ``clock_ghz`` is the
controller's clock, and when that is not the program clock the export also
gives ``program_clock_ghz``, converts every time to the controller's cycles
(exactly for operations on its lines; rounded up for classical calls and
segments) and scales each waveform's time parameters (``duration``,
``sigma``, ``beta``, ``risefall``) with them. A program for a controller at
the program clock is unchanged. ``ControlSystem(reference_clock_ghz=...)``
states a clock the controllers share; every controller's clock must be a
whole multiple of it, and the export gives it as ``reference_clock_ghz``.
Without one, each controller's start is known only to within one of its own
cycles, which budgets between controllers count.

Each ``pulse.Controller`` of a control system may run its own clock.
Without a declared ``"clock_ghz"`` the quantum system's program clock is the
fastest controller clock when it is a whole multiple of every other; clocks
that are not are a ``QuantumSystemDefinitionError`` listing the program clocks
that would work, never one picked for you. Every line is then scheduled on
the timesteps of the controller that plays it, and checked against that
controller's limits in its own cycles. A sync, and a wait whose length
comes from a classical answer, meet where every line's timesteps do. So do
the lines of a runtime qudit, so that the kernel's one schedule fits
whichever qudits a call binds. A line's delay is split by its own
controller's fine delay, and every schedule keeps an operation that depends
on a readout for the reading controller's ``feedback_latency_cycles``
converted to program cycles, counted from where the readout ends on its
delayed line (its fine delay rounded up to a cycle).
``custom(duration, f)`` samples ``f`` at the rate of the line that plays
it, the waveforms made from it too; samples given to
``custom_samples`` play only on lines of the rate they were made for, and
on a runtime qudit's line, whose rate depends on the call, ``custom()`` is
a compile error when the controllers' rates differ.

Each controller gets its own program: ``scheduled.export_to_controller(controller=k)``
is controller ``k``'s part, in its own cycles, with ``"controller": k``,
its ``lines`` and ``limits``, and the events it takes part in: those on its
lines, a sync with only its lines, the submit of a call whose measurements
it reads and the answer it takes, whether to branch on or to compute with.
Event and line ids stay the program's, so a value computed from an answer
names the answer by the same ``index`` in every controller's program. A
program that plays the lines of more than one controller needs
``controller=``; one on a single controller exports it without, and one
that plays no lines is exported by the controller that makes its calls.
A controller the program does not use exports a valid empty program: no
lines, events or calls. ``export_to_emulator(controller=k)`` gives the same
part for an emulator. Exporting every controller in turn walks the program
and checks it against the whole control system once, not once per
controller: the program keeps what its parts share until each one has been
exported, and every export is a fresh copy, whatever the order.

A link is shared by default: every controller it reaches sends its
requests to one instance, which holds ``slots`` requests for all
of them. A link that each controller has its own instance of, a decoder
beside each controller for example, is declared per controller:

.. code-block:: python

   import cudaq_pulse as pulse

   links = {"decoder": pulse.Link(fixed_ns=10, slots=1, per_controller=True)}

or ``per_controller = true`` in the link's table of a system file
(``[control_system.links.decoder]``). Each controller's calls over it then
use that controller's slots, so calls on two controllers are scheduled
without waiting for each other's slots, and each controller exports only
its own calls, with the call's ``"controller"`` in its submit, answer and
``calls`` entries. A call never crosses between controllers, so its
answer is due without the start uncertainty of a second controller. The
call's measurements and the pulses that use its answer must be on one
controller, or the program fails to compile with a ``CompilationError``
naming the function and both controllers.
When the scheduler cannot tell a call's controller (it reads no
measurement and its answer is unused, or leaves the kernel), the link's
calls share one set of slots, as on a shared link. Each controller may
place the link on one of its own processors; ``controller_usage`` counts
each instance's calls on its controller's processor, and its ``links``
have a row per instance.

Times between controllers are budgeted with how well each one's start is
known: its ``clock_skew_cycles``, plus one of its own cycles without a
``reference_clock_ghz``. A pulse that branches on, or computes with,
another controller's readout starts once the outcome can reach it: the
reader's ``feedback_latency_cycles`` and its ``peer_latency_cycles`` to the
user, in the reader's cycles, and both controllers' start uncertainty.
Every schedule waits that long for each pair of reader and user, and a
pulse on the reader's own controller waits its feedback latency alone; a
program scheduled for other controllers that starts the pulse sooner fails
their check with the cycle it may start at. An answer taken on another controller than the one
that sent the request is budgeted with the two controllers' start
uncertainty, and ``due_cycles`` includes it; one controller sending and
taking it adds none. When the scheduler cannot tell which controllers they
are (the request carries no measurement, or the answer leaves the kernel),
the budget adds the start uncertainty of the two controllers the link
reaches that are known least well.

A sync of lines on several controllers lines up the first pulse after it
on each of its lines. Each of those lands within its controller's start
uncertainty of where it is meant to, plus how far its line delay is
rounded to its grid when the controller has no fine delay, all in the
controller's cycles, and that must fit the controller's
``alignment_tolerance_cycles``; otherwise the program fails to compile,
suggesting a shared ``reference_clock_ghz`` or the tolerance it needs.
Under ``rcp`` and ``alap_rcp``, ``max_concurrent_drives`` and
``max_concurrent_readouts`` count pulses on every controller: a pulse that
follows another controller's on the same lane starts their start
uncertainty after it, so the two never overlap. Given to a control system
with controllers, they are checked under every schedule, counted the same
way. A program that ``asap`` places past them is placed as ``rcp`` places
it, and one that ``alap`` places past them as ``alap_rcp`` does; its
``options.schedule`` names the policy that placed it. A program that ``asap``
or ``alap`` places within them keeps its schedule exactly. It fails to compile with a
``ControllerLimitError`` naming the count and the first cycle when that
placement does not keep to every limit either, or when the values a call
binds place its pulses past one.

A wait whose length is computed at run time (``wait(line, value,
bounds=(low, high))``) is scheduled at its lowest length, and what follows
it on its lines starts later in a run, by up to ``high - low`` cycles;
lines the wait does not hold back keep their times. A blocking wait for an
answer is such a wait without an upper end. These limits, and a
shared line's ``max_concurrent_tones``, count pulses as they can slide: two
pulses the schedule places apart are refused when some run plays them at
once, with the first cycle they can and the waits that let them
(``2 drives can play at once from cycle 600: what follows the wait on d0
that ends at cycle 117 (up to 3984 cycles later) starts later in a run
than it is placed``). Each wait's length is counted as independent of the
others', also where two are computed from one answer, and every trip of a
loop with such a wait as able to slide as far as its last. A wait without
an upper bound lets what follows it play at any later time. To keep such
pulses apart, ``sync`` their lines after the wait, so that they wait
together, or give the wait bounds that keep them apart; ``rcp`` and
``alap_rcp`` hold a line back at the wait when the drive or readout limit
could bind. Pulses in the two arms of one branch are kept apart where
they follow the same wait, and counted together where they follow
different ones. A program without such a wait, and pulses on lines that
all wait together, are counted as scheduled. A long loop with such a wait
whose pulses can slide against another line's is counted from the trips
the check walks and the rest of its trips together, whatever their
number. Where that cannot be done (for example a pulse in the loop on a
line its waits do not hold back, a loop in a loop, a long loop of counted
pulses without such a wait in the same program, or line delays that put a
pulse before one of an earlier wait) the program is counted trip by trip,
up to the 1,048,576 copies of loops a check plays; past that the check
stops with an ``UnsupportedError``.

What the program computes at run time is converted by the same rule. A
variable wait's ``length`` is clamped to its bounds, as the run clamps it
(an integer ``max`` and ``min``), and then wrapped in ``{"op":
"controller_cycles", "type": "int", "of": [x], "num": a, "den": b,
"timestep": g}``: the length ``x`` program cycles is ``x * a / b``
of the controller's, and the controller must fail the run when ``x`` is not
a whole number of the line's timesteps of ``g`` program cycles. A
controller with the program's clock but longer timesteps gets the wrapper
too, with ``num`` and ``den`` 1, as that check. ``min_cycles`` rounds down
and ``max_cycles`` up. A run-time ``duration`` is wrapped the same way, and
a run-time ``sigma``, ``beta`` or ``risefall`` is multiplied by ``a / b``
(a float ``mul``). A run-time length whose values are all known while the
kernel compiles (it selects between constants) and include one off the
timesteps is a ``ControllerLimitError`` then. A run-time operand of a
waveform whose parameters the export does not name can't be converted, so
on a controller with its own clock it is a ``ControllerLimitError`` too,
never exported in program cycles.

.. _line-delays:

Line delays
-----------

Cables and electronics delay each line differently. The calibrated delay
of a line belongs to the line, as its controller declares it:
``DriveLine(5, delay_ns=4.0)`` and ``ReadoutLine(5, delay_ns=12.0)`` (from
``cudaq_pulse.controller``) in the controller's ``drive_lines`` and
``readout_lines``, as times (``2.3 * pulse.ns``) or numbers of ns; a bare
carrier index is an undelayed line. Every operation on a delayed line starts
that much later than scheduled.
Whole :ref:`timesteps <timestep>` of the line move its operations, which
are then checked against the controller's limits like any others; the rest,
less than one timestep, is the line's ``fine_delay_ns`` in the export, for
a controller that can delay a line's output by less than that
(``Controller.max_fine_delay_ns``). A controller without fine delay
rounds the delay to the nearest timestep if that is within its
``alignment_tolerance_cycles``, and otherwise the program fails to compile
with an error naming the line, the rest and the fixes::

    from cudaq_pulse.controller import DriveLine, ReadoutLine

    box = pulse.Controller(
        clock_ghz=1.0,
        max_fine_delay_ns=1.0,
        drive_lines=[DriveLine(0, delay_ns=2.3), 1],
        readout_lines=[ReadoutLine(0, delay_ns=41.0), 1],
    )
    control_system = pulse.ControlSystem(controllers=[box])
    # The drive line of qudit 0 moves 2 ns (4 cycles of a 2 GHz program
    # clock) and plays with fine_delay_ns = 0.3.

In a system file a delayed line is ``{"qudit": 0, "delay_ns": 2.3}`` and an
undelayed one its carrier index. Compiling a kernel again with only other
delays, such as a control system calibrated again, reuses its trace and
optimized module and schedules it again, with the same result, byte for
byte, as compiling it anew. The ``rescheduled_for_line_delays`` of what
``lower()`` returns says which it did, so a calibration loop can check
that each round took that path: ``True`` when the earlier trace was
scheduled again, ``False`` when the kernel was traced, such as the first
time, with the line delays it was last compiled with, the first time it
is compiled with line delays at all (only a compile with line delays keeps
its trace), for a kernel with list or qudit parameters, and after a change
to what the kernel read while it was traced. A call's ``rescheduled`` says
something else: whether its float values set a wait length (see
:ref:`wait-lengths`).

Shared lines
------------

A line may reach several qudits: a readout feedline multiplexing the
readout of the qudits along it, or a drive line combining several drive
tones. ``ReadoutLine([0, 1, 2, 3], max_concurrent_tones=4, delay_ns=12.0)``
reads out qudits 0 to 3 and plays up to four tones at once; a line reaching
one qudit, ``ReadoutLine(5)`` or ``ReadoutLine([5])``, is dedicated, as
before. ``max_concurrent_tones`` is 1 by default, and at most the qudits
the line reaches. Every line has ``qudits``, a tuple, and
``max_concurrent_tones``; only a dedicated line has ``qudit``, which is an
error on a shared one. A qudit is reached by at most one line of each kind
in the control system: a second one is an error (choosing between lines is
not supported yet).

A kernel is written as for dedicated lines: ``get_readout_line(q)`` of a
qudit on the feedline gives one of its tones, for ``q``, with ``q``'s
frequency, and each call gives a line of its own::

    from cudaq_pulse.controller import ReadoutLine

    box = pulse.Controller(
        drive_lines=[0, 1, 2, 3],
        readout_lines=[ReadoutLine([0, 1, 2, 3], max_concurrent_tones=4)],
    )

    @pulse.kernel
    def measure(a, b, c, d):
        lines = [get_readout_line(a), get_readout_line(b), get_readout_line(c), get_readout_line(d)]
        sync(lines[0], lines[1], lines[2], lines[3])
        for line in lines:
            readout(line, square(400, 0.05))

The compiler checks every line that reaches more than one qudit, whatever
its ``max_concurrent_tones``, for the tones it plays at once: more than
``max_concurrent_tones`` drives or readouts on it overlapping in time is a
``ControllerLimitError`` naming the line, the cycle and the qudits (with
one tone at once, its qudits' pulses must not overlap at all). Pulses in
the two arms of one branch never play together. The scheduler does not
move pulses to keep within it: play some after the others end, with a
sync between them. A dedicated line is never checked for it, and a
program on dedicated lines costs nothing more to compile or export.

The other limits of a ``Controller`` apply as follows on a shared line.
Per tone, as each tone is its own sequencer: the spacing of its operations
(``instruction_cycles``, ``min_idle_cycles``) and ``max_amplitude`` (the
sum of concurrent tones is not checked yet). Per physical line: its
waveform memory (``waveform_memory_samples`` with ``"per_line"``, the
distinct shapes of all its tones) and ``max_frequencies_per_line`` (the distinct
tone frequencies of all its tones). ``max_pulses`` counts every drive and
readout, as on dedicated lines. A kernel with runtime qudits compiles on
shared lines, and each call is checked for the tones at once when it binds
its qudits.

Format ``/2`` of the export lists a shared line once, with its
``max_concurrent_tones``, and gives each event on it a ``tone_slot`` and
the ``qudits`` it is for (see `The structured format`_). Format ``/1``
keeps a line for each qudit, as the kernel has: each line on a shared line
also has, after its ``qudit``, the shared line's ``qudits`` and
``max_concurrent_tones`` (lines with the same ``role`` and ``qudits`` are
tones of one line), and each drive, readout, wait and frame update on it
its ``tone_slot``, after its ``qudit``, the slot ``/2`` gives it; so do the
``runtime_tones`` and ``frame_state`` entries of a line that keeps one slot.
``/1`` lists every execution, so it gives slots to handles taking turns in
a loop too. Format ``/1`` is the one for a shared line on a control system
of several controllers (each controller's part has its own lines' keys), on
a controller with its own clock, or with line delays, which ``/2`` does not
take. On dedicated lines none of these keys are present. The
simulator plays each tone on its own qudit. In a system file a shared line
is ``{"qudits": [0, 1, 2, 3], "max_concurrent_tones": 4, "delay_ns": 12.0}``,
without the keys at their defaults. ``examples/transmon/transmon_surface17_multiplexed_readout.py``
reads out the four X-check ancillas of ``transmon_surface17`` on one
feedline.

Broadcast lines
---------------

A drive line may reach several qudits with one tone: a beam or an antenna
that drives every qudit under it at once. ``DriveLine([0, 1, 2],
broadcast=True)`` declares one. Where a shared line has a tone per qudit
and a drive on it reaches one qudit, a broadcast line has one tone, and a
drive on it reaches all its qudits; ``max_concurrent_tones`` does not apply
to it, and a line is one or the other. ``broadcast`` and ``frequency_hz``
are for a line given a list of qudits: ``DriveLine([0], broadcast=True)`` is
an error, and on one qudit given as an int, ``DriveLine(0,
broadcast=True)``, they are ignored.

A kernel gets the line from any one of its qudits, or from all of them,
which says in the kernel what the drive reaches; either way it is the one
line::

    from cudaq_pulse.controller import DriveLine

    box = pulse.Controller(drive_lines=[DriveLine([0, 1, 2], broadcast=True)])

    @pulse.kernel
    def together(a, b, c):
        beam = get_drive_line(a, b, c)      # get_drive_line(b) is the same line
        drive(beam, gaussian(400 * pulse.ns, 0.5, 100 * pulse.ns))    # reaches a, b and c
        shift_phase(beam.tone, 0.3)         # the line's one tone

Several qudits must be all the qudits of one broadcast line, each once, in
any order; other qudits are a ``CompilationError``, as are several qudits
where the control system has no broadcast line and the first qudit's line
is not yet taken. A kernel that holds only
one of the line's qudits still drives all of them, and the compiled
program's ``qudits`` lists them all. Pulses on the line follow each other,
as on a dedicated line, and the line has one frame: a phase or frequency
change of ``beam.tone`` applies to every later drive on it, for every
qudit, so there is no virtual Z of one of its qudits alone. The line has no
``tones`` of other transitions and no calibrations.

The tone starts each shot at the frequency the line states,
``DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.1e9)``. It may be
left out when the line's qudits share one frequency, which is then the
tone's; when they differ, a program that takes the line is a
``CompilationError`` asking for it. On a quantum system whose qudits give
an ``Offset`` from one reference, the tone is the ``Offset`` its qudits
share, and a line that states ``frequency_hz``, or whose qudits differ, is
a ``CompilationError``: the line cannot state an offset yet. On a system
whose qudits give no frequency, each tone is an offset from its own
qudit's carrier, and the line's one tone is at the carrier of every qudit
it reaches: a drive on it is resonant with each of them, and a change of
the tone's frequency (``shift_frequency(beam.tone, 1e6)``) is that offset
from each reached qudit's own carrier. The line cannot state
``frequency_hz`` there (a ``CompilationError`` for a program that takes
it), its ``native_frequency_hz`` in the exports is 0.0 as for every line of
such a system, and each event's ``frequency_hz`` is the offset every qudit
in its ``qudits`` sees. A drive on the line at another qudit's tone, or on
another line at the line's tone, is refused as between any two qudits of
such a system. Each qudit is driven off resonance by
the difference between the tone and its own frequency: the simulator
drives every qudit of the line with the same envelope, the tone's
frequency and phase, and that qudit's own amplitude scale.

The scheduler, the frame bookkeeping and the limits of a ``Controller``
treat the line as a dedicated one: one pulse counts once toward
``max_pulses``, and ``max_amplitude``, the grid, the pulse lengths and
``max_frequencies_per_line`` apply to its one tone. A kernel with runtime
qudits compiles for a control system with a broadcast line, and each call
is checked when it binds its qudits: a runtime qudit that takes a drive
line cannot be bound to a qudit of a broadcast line, since its line was
compiled as that qudit's alone (pass such a qudit when the kernel is
lowered). ``ReadoutLine`` has no ``broadcast``.

Both formats of the export write the line once, with ``"broadcast":
true``, all its ``qudits`` and the one ``native_frequency_hz`` of its
tone, and one event for each drive, with the line's ``qudits`` and no
``tone_slot``. In format ``/1`` the line's ``qudit``, and that of its
events, is the lowest of them:

.. code-block:: json

    {"line": 0, "role": "drive", "qudit": 0, "qudits": [0, 1, 2], "broadcast": true,
     "native_frequency_hz": 5100000000.0}

In format ``/2`` the line has ``native_frequency_hz`` in place of
``native_frequencies_hz``, and each play and frame update on it has
``qudits`` after its ``line``:

.. code-block:: json

    {"kind": "play", "line": 0, "qudits": [0, 1, 2], "start_cycles": 0,
     "duration_cycles": 400, "waveform": 0,
     "tones": [{"frequency_hz": 5100000000.0, "phase_rad": 0.0}]}

``frame_state`` and ``runtime_tones`` entries on the line have its
``qudits`` too, and so does each of its drives in ``export_to_emulator()``,
after its ``qudit``. A backend must drive every qudit in ``qudits``: one that
reads only the line's first qudit would drive one. ``unroll`` does not take
a program with a broadcast line yet; export it in format ``/1``. ``load``
does not take a control system with one yet. The part of one controller of
several lists the line's lowest qudit alone in its top-level ``qudits``. A program
that takes no broadcast line exports as it does without one, and a control
system without one loads nothing for them. In a system file the line is
``{"qudits": [0, 1, 2], "broadcast": true, "frequency_hz": 5100000000.0}``.

Waveform memory
---------------

Two plain values of ``Controller`` describe the memory a controller
plays waveforms from, the same in Python and in a controller description
in TOML (``waveform_memory_samples = 65536``, ``waveform_memory = "shared"``):

.. code-block:: python

   import cudaq_pulse as pulse

   limits = pulse.Controller(
       sample_rate_ghz=2.0,
       waveform_memory_samples=65536,     # what one memory holds, in samples
       waveform_memory="shared",   # one memory for every line; "per_line" by default
   )

Compiling checks that every waveform of a program fits at once: in the
memory of its line, or once over all lines when ``"shared"``. The
controller plays one stored shape at any amplitude and phase, so waveforms
that differ only in those share memory, even when a ``float`` parameter sets
the amplitude. A limit that only some ``float`` values exceed is checked once
when a call binds them, never per shot. A shared memory reaches the export's
``controller`` as ``waveform_memory``, with the waveform ids and parameter
roles of ``/2``; a memory per line leaves the key out.

Controller limits
-----------------

``Controller`` states what a controller can play, in its own clock
cycles: its clock, how
closely two operations on one line can follow each other, its waveform
memory and distinct frequencies per line, which detector reads each qudit and how long a
readout holds it, the least feedback latency from a readout to an operation
that branches on it, whether it can stop waiting for an answer after a
timeout, and the most drives and readouts in one program. Given to
``pulse.compile(control_system=...)``, alone or in its ``ControlSystem``,
it makes every compilation check the scheduled program, so that a program the controller
cannot play fails at compile time with ``ControllerLimitError``, which lists
each violated limit. A limit that only some values of a ``float``
parameter exceed, such as the tones a frequency shift adds or the memory
a waveform width takes, is checked when a call supplies the values, once
per distinct values, so a call that compiles also exports.
``cudaq_pulse.controller.check(compiled, limits)`` returns the violations
without raising. A control system without controllers checks nothing
more.

Some instruments require a minimum pulse length.
``Controller(min_pulse_cycles=16)`` makes every drive and readout shorter
than 16 of the controller's cycles a violation naming its line, its start
and its length; on a control system of several controllers, each checks
its own lines. It is 0, no limit, by default, and the export lists it
among the limits only when it is set. Likewise, an instrument that caps a
pulse's length declares ``Controller(max_pulse_cycles=4095)``: every drive
and readout longer than 4095 of the controller's cycles is a violation
naming its line, its start and its length. It is ``None``, no limit, by
default, may not be below ``min_pulse_cycles``, and is listed among the
limits only when it is set.

A controller that plays nothing while one of its classical calls is out
declares ``Controller(blocks_on_calls=True)``. Each call then holds every
line from its submission until its answer is due: the scheduler sends it
once the lines before it are done and starts nothing after it before then,
in every schedule, and a program played on such a controller is checked
for it, naming each pulse that plays while a call is out. A call in a
kernel with its own ``schedule=``, which is placed whole, does not compile
for it. The export lists ``blocks_on_calls`` among the limits only when it
is set. In a control system of several controllers, a call holds each
controller that blocks on calls and issues it (whose instance of a
per-controller link runs it, else those that send its measurements, else,
over a shared link, every controller that plays), reads a measurement the
call sends or plays a pulse that uses its answer (branches on it, or
computes with it); the lines of the others play on while it is out.

Update buses
~~~~~~~~~~~~

Some instruments retune a group of lines over one shared bus that carries
one update at a time. ``Controller(update_buses=[...])`` lists such
groups, each a ``cudaq_pulse.controller.UpdateBus``: its lines by carrier
index, as the controller's own are given, and the cycles of the
controller's clock one update holds the bus for::

    from cudaq_pulse.controller import UpdateBus

    box = pulse.Controller(
        drive_lines=[0, 1, 2, 3],
        readout_lines=[0, 1, 2, 3],
        update_buses=[UpdateBus(drive_lines=[0, 1, 2, 3], cycles_per_update=8)],
    )

An update is what changes the tone a line plays, read from the program
alone:

- each frame update (a frame change no later drive or readout takes, see
  `Exporting a program`_);
- each drive or readout whose ``frequency_hz`` differs from that of the
  drive or readout before it on its line, or whose ``phase_rad`` differs
  other than as the tone runs on: a tone at its frame's frequency keeps
  its phase, and one retuned from the line's own frequency gains phase at
  its detuning, so neither is an update by itself. A frequency or phase
  known only at run time is an update.

The first drive or readout on a line is not an update, its tone being set
before the shot, and neither is a pulse that changes only its envelope or
amplitude. An update holds the bus for the ``cycles_per_update`` before it
takes effect, so updates on one bus must start at least that far apart;
two in the two arms of one branch never meet. A program that has them
closer is a ``ControllerLimitError`` naming the bus, its lines, both
updates and their times. A line is on at most one bus, and a bus reaches
only lines its controller plays. In a system file each bus is
``{"drive_lines": [0, 1], "readout_lines": [], "cycles_per_update": 8}``;
like the lines, the buses are wiring, given when a description is loaded
(``pulse.controller(name, ..., update_buses=[...])``), and the export lists
them among the limits only when there are any.

By default the scheduler does not place updates for the bus, and timing
never changes: space them with waits or syncs. A controller that asks for
it, ``Controller(update_buses=[...], space_updates=True)``, has compiling
space them instead. After scheduling, it takes the updates of each bus in
time order and holds the first that clashes back on its line by the
fewest cycles that put it ``cycles_per_update`` after the one before it,
and schedules the program again: what follows it on its line moves, and
syncs carry the move to their other lines, as the scheduler places them,
on the controller's grid. A delay can change which later pulses are
updates (a tone retuned across a wait gains phase over it), so it checks
the whole program again, and repeats until no update clashes, at most 256
times, else ``CompilationError``. An op is held back in every execution:
in a rolled loop, in every iteration. Updates in the two arms of a branch
stay independent. The export, the simulator and the controller checks all
read the spaced program, so feedback latency, syncs and every other limit
hold on it; a program with nothing to space keeps its schedule. Spacing
works on the program as compiled, so a kernel whose waits a call's
``float`` values set keeps the holds its compile found, and each call is
checked; one with runtime qudits is checked when a call binds them.

The check of a rolled loop does not play every iteration. Whether a pulse
is an update depends only on the pulse before it on its line, and a clash
only on the update before it on its bus; from a loop's iteration 1 on,
both are those of the iteration before, a period later and a phase step
on. On one clock the check plays each loop it keeps rolled at iterations
0 (the entry), 1 and 2 (every later iteration clashes as these do) and
its last two (which what follows the loop follows), and lists violations,
when there are any, from a walk of every execution. A program on
controllers with their own clocks, or on several controllers, is checked
on every execution.

A minimal backend
-----------------

A backend is an object with a ``name`` and three methods, the
``cudaq_pulse.backends.Backend`` protocol: ``control_system()`` returns the
control system to compile for (a ``ControlSystem``, or its one
``Controller``), ``emit(program)`` takes an
exported program and returns the controller's, and ``run(artifact,
shots_count=...)`` runs it. ``formats``, if present, lists the export formats it
reads. A program it cannot emit it refuses with
``IncompatibleBackendError``.

Two more methods are optional, and only ``backends.validate`` uses them
(see `Validating a backend`_). ``events(artifact)`` gives an emitted
artifact back as one list of events per controller program, each a mapping
with its ``kind`` (``cudaq_pulse.Event``, or the ``/1`` name), ``line``
(the link of a call or answer), ``start_cycles`` and optionally
``duration_cycles``, every execution in program order.
``emulate(artifact, shots_count=...)`` runs it on the backend's emulator and
gives each shot's measurement outcomes, in the order the measurements end.

This one emits one line of text per drive and readout:

.. code-block:: python

   from collections.abc import Mapping
   from typing import Any

   from cudaq_pulse import Controller
   from cudaq_pulse.backends import IncompatibleBackendError
   from cudaq_pulse.controller import FORMAT


   class TextBackend:
       name = "text"
       formats = (FORMAT,)

       def control_system(self) -> Controller:
           return Controller(clock_ghz=1.0, instruction_cycles=2, min_idle_cycles=2)

       def emit(self, program: Mapping[str, Any]) -> list[str]:
           if program["format"] != FORMAT:
               raise IncompatibleBackendError(f"cannot read {program['format']}")
           played = []
           for event in program["events"]:
               if event["kind"] in ("wait", "sync"):
                   continue
               if event["kind"] not in ("drive", "readout") or event["conditions"]:
                   raise IncompatibleBackendError(f"cannot play event {event['index']}")
               played.append(
                   f"{event['start_cycles']} {event['kind']} line {event['line']} "
                   f"{event['waveform']['shape']} at {event['frequency_hz']:.6g} Hz"
               )
           return played

       def run(self, artifact: Any, *, shots_count: int = 1) -> list[list[str]]:
           return [list(artifact) for _ in range(shots_count)]

Its distribution registers it under its name in the
``cudaq_pulse.backends`` entry-point group, naming the backend class or any
callable without arguments that returns one:

.. code-block:: toml

   [project.entry-points."cudaq_pulse.backends"]
   text = "text_controller:TextBackend"

Using a backend
---------------

``backends.available()`` lists the installed backends without importing
any, and ``backends.get(name)`` loads one on first use and keeps it.
``backends.register(name, factory)`` adds a backend in the running process
without a distribution, and takes precedence over an installed one of the
same name:

.. code-block:: python

   import cudaq_pulse as pulse
   from cudaq_pulse import backends
   from cudaq_pulse.quantum_systems import PhysicalQudit

   backends.register("text", TextBackend)
   backend = backends.get("text")
   qs = pulse.quantum_system(
       name="one-qubit", qudits={0: PhysicalQudit(0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-2.0e8)}, clock_ghz=1.0
   )


   @pulse.kernel
   def drive_and_measure(q, x_duration, x_sigma, readout_duration):
       drive_line = get_drive_line(q)
       readout_line = get_readout_line(q)
       drive(drive_line, gaussian(x_duration, 0.5, x_sigma))
       sync(drive_line, readout_line)
       readout(readout_line, square(readout_duration, 0.05))


   compiled = pulse.compile(drive_and_measure, quantum_system=qs, control_system=backend.control_system())
   scheduled = compiled.lower(
       qs.qudits[0],
       x_duration=40 * pulse.ns,
       x_sigma=10 * pulse.ns,
       readout_duration=200 * pulse.ns,
   )
   program = scheduled.export_to_controller()
   print(backend.run(backend.emit(program), shots_count=1))
   # [['0 drive line 0 gaussian at 5e+09 Hz', '40 readout line 1 square at 5e+09 Hz']]

Validating a backend
--------------------

``backends.validate(backend, controller=...)`` runs the export test corpus,
which ships with cudaq-pulse, through a backend: a program for each kind of
event (``cudaq_pulse.Event``), one with a branch, ``answer_timeout``, which
branches on whether a call's answer came in time and on the answer itself
(its conditions differ only in ``result``), ``timeout_function``, the
same for a function declared with ``latency_mode="timeout"``, and one
with values known only at run time, compiled for the controller (by
default the backend's control system's) and exported in each format the
backend reads. Four more are wired from that controller: ``shared_line``
reads out both qudits at once on one readout line reaching both
(``ReadoutLine([0, 1],
max_concurrent_tones=2)`` in place of its readout lines),
``shared_drive_line_two_tones`` drives both qudits at once with the same
envelope on one drive line reaching both (``DriveLine([0, 1],
max_concurrent_tones=2)`` in place of its drive lines, its readout lines as
they are) and reads both out; its two tones are at the qudits' two
frequencies. ``shared_drive_line_one_tone`` is the same program on a system
of the corpus's own whose two qudits share a frequency, so that a backend
whose line plays one tone for all its qudits can emit it. And
``several_controllers`` runs on two controllers made from it, one driving
qudit 0 and reading out both qudits on such a line, the other driving qudit
1, with a shared reference clock; each controller's part is emitted on its
own in format ``/1`` (``several_controllers[0]``, ``several_controllers[1]``),
and is not compared with ``emulate``. Each must be emitted or refused with
``IncompatibleBackendError``, and a program with an event kind no format
has, or in a newer format, must be refused. A program that exceeds the
controller's limits is skipped. The corpus's pulses and waits last 16 to 200
cycles, all multiples of 4; on a controller with a longer
``min_pulse_cycles``, or a timestep or sample rate those lengths are off,
every one of them lasts the fewest whole times as long that the controller
plays (16 cycles become 64 for a ``min_pulse_cycles`` of 64), so its
programs are checked rather than skipped. On every other controller the
corpus is as written.

.. code-block:: python

   from cudaq_pulse import Controller, backends

   report = backends.validate("text", controller=Controller(clock_ghz=1.0))
   print(report)
   # PASS play
   # ...
   # REFUSED call: cannot play event 1
   # ...
   # PASS unsupported_event
   # PASS newer_format
   # backend 'text': 12 passed, 8 refused
   print(report.ok)  # True

A backend that accepted the unknown event would instead report ``FAIL
unsupported_event: emit() accepted 'forward_outcome'; it should raise
IncompatibleBackendError``.

``branch`` plays its pulse on the readout's outcome once the controller's
``feedback_latency_cycles`` have passed after the readout ends: every
schedule waits for it, so a controller with long feedback latency runs
``branch`` too.

.. code-block:: python

   slow = Controller(clock_ghz=1.0, feedback_latency_cycles=2000)  # 2 us
   backends.validate(backend, controller=slow)
   # PASS branch

Two optional methods make the checks stronger. With ``events(artifact)``,
which gives the emitted programs back, one list of events per controller
(each with its ``kind``, ``line``, ``start_cycles`` and optionally
``duration_cycles``, every execution in program order), the number of
programs and the timing and order of plays, acquisitions, calls and answers
on each line must match the controller program, and the pulses must respect
the controller's limits. With ``emulate(artifact, shots_count=...)``, which gives
each shot's measurement outcomes in the order they end, the fraction of 1s
of each must match ``cudaq_pulse.simulate`` within ``tolerance`` (0.05 by
default) plus three standard errors of the ``shots_count``. Validating imports
nothing until it is called.

Errors
------

``backends.get`` raises ``BackendNotFoundError`` (a ``LookupError``) for a
name no backend has, listing the installed ones, and
``IncompatibleBackendError`` (a ``TypeError``) for a backend that fails to
import, is not a ``Backend``, or reads none of the formats this version
writes;
both are ``BackendError``. Compiling a program that exceeds the backend's
limits raises ``ControllerLimitError``, a ``CompilationError``.

Cost
----

Importing cudaq-pulse neither imports the backend modules nor looks for
installed backends; that happens on the first call to ``backends``. A
control system without controllers adds nothing to compilation. The
export and the limit checks are native. Exporting in ``/1`` takes time and
memory linear in a program's executions; in ``/2``, and for the checks, a
loop whose iterations repeat costs as much as its first two iterations and
its last, whatever its trip count. A violation inside it is still listed
once per iteration, each message built when it is read.
