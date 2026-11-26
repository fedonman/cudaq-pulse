..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Pulse Operations
================

All operations listed here are used as bare names inside ``@pulse.kernel``
functions. They are intercepted during bytecode tracing and lowered
to MLIR operations in the Pulse dialect.

Channel Access
--------------

.. function:: get_drive_line(qudit) -> Line

   Obtain the drive line of a qudit. The line is passed to ``drive()``,
   ``wait()`` and ``sync()``; its tone, ``line.tone``, is passed to the phase
   and frequency ops, as in ``shift_phase(line.tone, math.pi)``. A line is
   one value: it is not unpacked or iterated.

   ``line.tone`` drives the qudit's default transition, which its quantum system's
   model declares (levels 0 and 1 unless the model says otherwise).
   ``line.tones`` maps each transition the model lists, as a
   ``(lower, upper)`` pair of levels, to its tone, so
   ``line.tones[(1, 2)]`` is the tone of the 1-2 transition; the default
   transition's entry is ``line.tone`` itself. A transition the model does
   not list raises a ``KeyError`` that names the ones it does. The mapping is
   read-only.

   A qudit has one drive line for the whole program. The first call
   acquires it; every later call, in the kernel or in a kernel it calls,
   returns the line in its current state. Pulses on it play one after
   another, and phase and frequency changes carry over: ``x90(q); x90(q)``,
   where ``x90`` calls ``get_drive_line(q)`` itself, plays two pulses in
   sequence. A loop whose body acquires the line of a qudit that exists
   before the loop acquires it once, before the loop, and stays rolled.

.. function:: get_readout_line(qudit) -> Line

   Obtain the readout line of a qudit: the same one value per qudit for the
   whole program, with its tone in ``line.tone``. A readout line has only
   its own tone, so its ``line.tones`` holds that one entry.

Scheduling Operations
---------------------

.. function:: drive(line, waveform, *, tone=None)

   Play a waveform envelope on a drive line, at the line's own tone or, given
   ``tone=``, at another one, such as ``drive(control, cr, tone=target.tone)``
   for a cross-resonance pulse.

.. function:: readout(line, waveform, *, tone=None)

   Acquire a measurement through a readout line, at its own tone unless
   ``tone=`` gives another.

.. function:: wait(line, duration, bounds=None)

   Insert an idle delay of *duration* on a line: an ``int`` number of clock
   cycles or a time, such as ``wait(line, 10 * pulse.ns)`` (see
   :ref:`durations`). A duration computed from ``float`` parameters stays a
   runtime value, rounded to the nearest cycle on each call; *bounds*,
   ``(lowest, highest)`` in cycles or as times, limits it. See
   :ref:`wait-lengths`.

.. function:: sync(line1, line2, ...)

   Synchronize two or more lines to a common time point.
   All lines are padded to the latest time among them.

Phase and Frequency Control
---------------------------

These operations modify the rotating frame of a tone channel.

.. function:: shift_phase(tone, phase)

   Advance the tone's frame phase by *phase* (radians): a virtual Z rotation
   by *phase*, in the convention of McKay et al., PRA 96, 022330 (2017). A
   pulse played after it turns about an axis rotated by ``-phase``, so
   ``shift_phase(t, pi / 2)`` followed by an X90 pulse is an X90 about the
   -Y axis. It takes no time, and ``Pass.FOLD_PHASES`` folds it into the next
   drive. See :ref:`virtual-z`.

.. function:: set_phase(tone, phase)

   Set the tone's frame phase (radians), with the same sign as
   ``shift_phase``.

.. function:: shift_frequency(tone, frequency)

   Add a relative frequency offset (Hz) to the tone.

.. function:: set_frequency(tone, frequency)

   Set the absolute frequency (Hz) of the tone. The phase carries over
   continuously.

In the qudit's frame, which rotates at its 0-1 frequency, a tone at
``w_tone`` drives with ``H = (A/2)(E a^dagger + E^* a)`` and
``E(t) = envelope(t) exp(-i theta(t))``, where the frame phase
``theta(t) = theta_0 + (w_tone - w_q) t``: a tone above the qudit runs ahead of
its frame. The 1-2 transition of a transmon is driven resonantly by a tone at
``w_q + alpha``.

.. themed-figure:: figures/frame_phase.svg
   :alt: The phase of a drive tone's frame against time, above the three X90 pulses that take it. A shift_phase of π/2 at 40 ns is a step; a shift_frequency of +2.5 MHz at 80 ns makes the phase rise steadily from there. The phase each pulse starts with: 0 for the pulse from 0 ns, π/2 for the pulse from 40 ns, π for the pulse from 180 ns. Each pulse's small Q quadrature is drawn dashed.
   :width: 100%

   A frame over three X90 pulses: ``shift_phase(tone, pi / 2)`` after the
   first, and ``shift_frequency(tone, 2.5e6)`` and a 100 ns wait after the
   second. A phase shift is a step and a frequency shift a slope; each
   pulse takes the phase the frame has when it starts. Dashed blue is a
   pulse's Q quadrature.

.. _virtual-z:

Phase frames and virtual Z gates
--------------------------------

A Z rotation needs no pulse: advancing a tone's frame phase by ``phi``
turns every later pulse at that tone about an axis rotated by ``-phi``,
which is the same as applying ``Rz(phi) = exp(-i phi Z / 2)`` to the qudit
before those pulses. The phase gates ``pulse.gates.Z``, ``S`` and ``T`` are
these virtual Z rotations, by pi, pi / 2 and pi / 4 (each equal to its gate
up to a global phase). ``get_calibration`` gives each as a ``VirtualZ``, and
``drive(line, it)`` plays it as ``shift_phase(line.tone, phase_rad)``: it
takes no time and adds no error.

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def t_state(q):
       d = get_drive_line(q)
       drive(d, d.get_calibration(pulse.gates.Y90))   # |0> to |+>
       drive(d, d.get_calibration(pulse.gates.T))     # |+> to |T>, a frame change

The sign convention is that of McKay et al., PRA 96, 022330 (2017): a
positive phase is a positive rotation about +Z, so ``T`` takes ``|+>`` to
``(|0> + e^{i pi/4} |1>) / sqrt(2)``. There are no inverse gates; a
calibration's ``replace`` gives one, ``t.replace(phase_rad=-t.phase_rad)``.

The frame is bookkeeping on the control side, so it changes what later pulses
do, never the state itself. ``simulate`` returns the state in the frame: a
virtual Z after the last pulse leaves ``density_matrix`` as it was, and the
state in the logical basis is ``Rz(phi)`` applied to it, for ``phi`` the
phase the qudit's tone has accumulated. Two consequences:

- A readout cannot see a virtual Z played just before it: a Z rotation
  leaves the populations a readout measures as they were.
- A pulse played after it, such as a tomography rotation before a readout,
  sees the frame and measures the logical state.

A virtual Z shifts the frame of one tone. A pulse played on another line at
that tone, such as a cross-resonance drive at the target's tone, sees the
shift; a pulse at another tone, such as the 1-2 tone, does not. A phase gate
of another transition, ``get_calibration(pulse.gates.S, transition=(1, 2))``,
is played at that tone: ``drive(d, s12, tone=d.tones[(1, 2)])``.

Waveform Constructors
---------------------

Each constructor returns a waveform value that can be passed to
``drive()`` or combined with waveform arithmetic. A *duration* or *width* is
an ``int`` number of clock cycles or a time such as ``20 * pulse.ns``;
*sigma* and *beta* take clock cycles, fractional too, or times. See
:ref:`durations`.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``gaussian(duration, amplitude, sigma)``
     - Gaussian envelope with standard deviation *sigma*:
       ``amplitude * exp(-(t - duration / 2)**2 / (2 * sigma**2))`` at time
       ``t`` from the start of the pulse. It is cut off at the ends of
       *duration*, with no offset subtracted: with ``duration = 4 * sigma``
       it starts and ends at about 14% of *amplitude*.
   * - ``square(duration, amplitude)``
     - Constant-amplitude (flat-top) envelope
   * - ``drag(duration, amplitude, sigma, beta)``
     - DRAG pulse for leakage suppression: ``G + i beta dG/dt`` for the
       Gaussian ``G``, with *beta* in clock cycles or a time, like *sigma*.
       Leakage of a transmon is smallest near ``beta = -1 / alpha``, for the
       anharmonicity ``alpha`` in radians per clock cycle (``2 pi f_alpha /
       f_clock``, negative for a transmon).
   * - ``cosine(duration, amplitude)``
     - Raised-cosine envelope
   * - ``tanh_ramp(duration, amplitude, sigma)``
     - Hyperbolic tangent ramp from zero to the requested amplitude
   * - ``gaussian_square(duration, amplitude, sigma, width)``
     - Flat-top pulse with Gaussian rise/fall edges
   * - ``custom(duration, envelope_fn)``
     - A Python function ``f(t)`` of the time in clock cycles, sampled while
       tracing (below)
   * - ``custom_samples(samples, sample_rate_ghz=None)``
     - Waveform from a non-empty sequence of at most 252 real-valued samples,
       made for one sample per clock cycle or for ``sample_rate_ghz``
       (below)

For ``gaussian_square``, ``width`` is the flat-top width;
``duration - width`` must be a positive even number of clock cycles so the
two edges have the same integer duration.

.. rubric:: Waveform explorer

Pick a waveform and move its sliders or type a value; the line under the
plot is the call that makes it. *combine* applies the waveform arithmetic
below to a second waveform, and *modulate* shows the drive in the qudit's
frame (see :ref:`conventions-frames`).

.. only:: html

   .. raw:: html

      <div class="pulse-waveform-explorer" data-shape="drag"></div>

.. themed-figure:: figures/waveform_envelopes.svg
   :figclass: pwe-fallback
   :width: 100%
   :alt: The I and Q envelopes of gaussian, drag, cosine, tanh_ramp,
         gaussian_square and square, each 40 clock cycles long.

   I (solid) and Q (dashed) of each parametric waveform, 40 clock cycles
   (20 ns) long, as ``sample_waveform`` samples them. With JavaScript on,
   the explorer replaces this figure.

``custom(duration, f)`` calls ``f`` once per sample of the lines it plays on
while the kernel is traced, at the middle of each sample, and holds each
value until the next: the IR records the ``custom_samples`` of those values,
so a custom envelope compiles, schedules and simulates on the CPU and the
GPU like the built-in ones. On the program clock that is once per clock
cycle (``t = 0.5, 1.5, ..., duration - 0.5``); a controller with its own
clock, or several samples per cycle, plays at its own rate, and ``f`` is
sampled at that rate (at half the clock, ``t = 1, 3, ...``). Built-in
envelopes are evaluated continuously, so a function that reproduces one,
such as a Gaussian for ``gaussian``, matches it up to that sampling. ``f``
must be defined outside the kernel and return real numbers that do not
depend on kernel parameters; scale the waveform by a parameter with
``wf_scale(custom(duration, f), amplitude)``. The duration must be known
while tracing and at most 252 samples (250 at a controller's own rate), the
length of one sample record; play a longer envelope as consecutive drives.

A ``custom_samples`` array is made for one rate and plays only at that
rate: one sample per clock cycle, or ``sample_rate_ghz``. On lines whose
controller plays another rate (its ``sample_rate_ghz``)
it is a compile error, never a pulse that lasts longer or shorter than the
samples were made for; resample the array for the line and pass its rate,
or use ``custom``:

.. code-block:: python
   :class: fragment

   @pulse.kernel
   def raw(q):
       d = get_drive_line(q)
       # 1 GHz samples, for a controller at half of a 2 GHz program clock:
       # 8 samples last 16 clock cycles.
       drive(d, custom_samples(samples_1ghz, sample_rate_ghz=1.0))

Waveform Arithmetic
-------------------

Waveforms can be combined algebraically inside kernels:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def combined_waveform(qubit, duration):
       drive_line = get_drive_line(qubit)
       wide_sigma = 8 * pulse.ns
       narrow_sigma = 5 * pulse.ns
       envelope_a = gaussian(duration, 0.5, wide_sigma)
       envelope_b = gaussian(duration, 0.3, narrow_sigma)
       combined = wf_add(envelope_a, envelope_b)
       scaled = wf_scale(envelope_a, 0.5)
       inverted = wf_neg(envelope_a)

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Operation
   * - ``wf_add(left, right)``
     - ``left + right``
   * - ``wf_sub(left, right)``
     - ``left - right``
   * - ``wf_mul(left, right)``
     - ``left * right`` (element-wise)
   * - ``wf_scale(waveform, scalar)``
     - ``scalar * waveform``
   * - ``wf_neg(waveform)``
     - ``-waveform``

The operands of ``wf_add``, ``wf_sub`` and ``wf_mul`` must have the same
duration; a mismatch is a compile error. For simulation and execution, a
sum, difference, scaling or negation lowers exactly: the drive Hamiltonian
is linear in the envelope, so each term plays as its own envelope. A
product is not linear and is sampled once per clock cycle into
piecewise-constant samples, as ``custom_samples`` would be.
