..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

.. _conventions:

Conventions
===========

This page states the units, time bases and signs a program depends on and
its code does not show: what a bare number means, which way a frame phase
turns the pulses after it, which way a virtual detuning moves Ramsey
fringes, and what sign an exported phase has. Each rule comes with a
snippet that shows it, and the snippets run in order, as on the
:doc:`../tutorials/index` pages. ``tests/test_conventions.py`` runs them,
checks what they print, and checks each rule again in a test of its own, so
the page and the code cannot drift apart.

The snippets use one qudit at 5 GHz without decay, whose drive turns it by
1 radian per nanosecond at amplitude 1, and read its state as a Bloch
vector ``(x, y, z)``, with ``|0>`` at ``z = +1``:

.. code-block:: python

   import math
   import warnings

   import cudaq_pulse as pulse
   import numpy as np
   from cudaq_pulse.quantum_systems import MicrowaveDrive, PhysicalQudit

   drive_model = MicrowaveDrive(amplitude_scale_rad_per_ns=1.0)
   qs = pulse.quantum_system(
       name="conventions",
       qudits={0: PhysicalQudit(0, 5.0e9, 0.0, 0.0, drive=drive_model)},
   )
   q = qs.qudits[0]


   def bloch(kernel, *args):
       """The qudit's Bloch vector at the end of the kernel, to 3 places."""
       result = pulse.simulate(kernel, q, *args, levels=2, dt_ns=0.01)
       rho = np.asarray(result.density_matrix)
       x, y, z = 2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real
       return [round(float(v), 3) + 0.0 for v in (x, y, z)]


   def drives(kernel, *args):
       """The drive events of the kernel's controller export."""
       events = kernel.lower(q, *args).export_to_controller()["events"]
       return [e for e in events if e["kind"] == "drive"]

.. _conventions-units:

Units
-----

.. test: test_a_bare_duration_counts_clock_cycles

**Times.** A duration, a wait, and a pulse's ``sigma``, DRAG ``beta`` and
flat-top ``width`` are a number of clock cycles when given as a bare number,
and a time when given with units. A time converts to cycles exactly when
the kernel compiles, for the quantum system it compiles for:

.. code-block:: python

   @pulse.kernel
   def bare(qudit):
       drive(get_drive_line(qudit), square(20, 0.1))


   @pulse.kernel
   def with_units(qudit):
       drive(get_drive_line(qudit), square(20 * pulse.ns, 0.1))


   print(qs.clock_ghz)                           # 2.0
   print(drives(bare)[0]["duration_cycles"])        # 20 cycles, 10 ns
   print(drives(with_units)[0]["duration_cycles"])  # 40 cycles, 20 ns

``pulse.compile(k, unit_warnings=True)`` warns at each bare number (see
:ref:`unit warnings <unit-warnings>`):

.. code-block:: python

   with warnings.catch_warnings(record=True) as caught:
       warnings.simplefilter("always")
       pulse.compile(bare, unit_warnings=True).lower(q)
   print(caught[0].category.__name__)   # UnitWarning

.. test: test_frequencies_are_in_hz

**Frequencies** are in Hz everywhere: ``set_frequency`` sets the tone's
frequency, ``shift_frequency`` adds to it, and a qudit's ``frequency_hz``
and ``anharmonicity_hz`` are in Hz. A transmon's anharmonicity is negative:
its 1-2 transition is at ``frequency_hz + anharmonicity_hz``.

.. code-block:: python

   @pulse.kernel
   def tuned(qudit):
       d = get_drive_line(qudit)
       set_frequency(d.tone, 5.1e9)
       drive(d, square(8, 0.1))
       shift_frequency(d.tone, 2e6)
       drive(d, square(8, 0.1))


   print([e["frequency_hz"] / 1e9 for e in drives(tuned)])   # [5.1, 5.102]

**Phases** are in radians: ``shift_phase``, ``set_phase``, a calibrated
pulse's ``phase_rad`` and the export's ``phase_rad``.

.. test: test_amplitude_times_duration_is_the_rotation_angle

**Amplitudes.** On a qudit whose drive gives ``amplitude_scale_rad_per_ns``
(``MicrowaveDrive(amplitude_scale_rad_per_ns=s)``, or
``Drive(amplitude_scale_rad_per_ns=s)`` for a drive of another modality),
amplitude ``a`` is a Rabi rate of ``a * s`` radians per nanosecond: a square
pulse of amplitude ``a`` for ``t`` ns turns the qudit by ``a * s * t``
radians, so with ``s = 1`` a 20 ns pi pulse has amplitude ``pi / 20``.
Without a scale, amplitude 1 is the system's own unit (see
:doc:`quantum_systems`).

.. code-block:: python

   @pulse.kernel
   def square_pulse(qudit, amplitude: float):
       drive(get_drive_line(qudit), square(20 * pulse.ns, amplitude))


   print(bloch(square_pulse, math.pi / 20))   # [0.0, 0.0, -1.0], a pi pulse
   print(bloch(square_pulse, math.pi / 40))   # [0.0, -1.0, 0.0], a pi/2 pulse

**Coherence** times (``t1_us``, ``t2_star_us``, ``t2_echo_us``) are in
microseconds; ``thermal_population`` is a fraction.

.. _conventions-time:

Time bases
----------

.. test: test_the_program_clock_is_2_ghz_unless_the_system_says_otherwise

The program clock is ``clock_ghz`` of the quantum system: 2 GHz unless the
system (``pulse.quantum_system(..., clock_ghz=)``, or its file's
``"clock_ghz"``) gives another.

.. code-block:: python

   print(pulse.quantum_system("transmon_surface17").clock_ghz)                 # 2.0
   print(pulse.quantum_system("transmon_surface17", clock_ghz=1.0).clock_ghz)  # 1.0

.. test: test_the_schedule_and_the_export_count_cycles_from_the_program_start

The schedule (``ScheduledKernel.schedule``) and the controller export
(``export_to_controller()``) give times as whole cycles of that clock, from
0 at the start of the program: ``start_cycles``, ``duration_cycles``,
``end_cycles``. ``pulse.simulate`` takes ``t_end`` and ``dt_ns`` in
nanoseconds, or as times.

.. code-block:: python

   @pulse.kernel
   def two_pulses(qudit):
       d = get_drive_line(qudit)
       drive(d, square(10 * pulse.ns, 0.1))
       wait(d, 5 * pulse.ns)
       drive(d, square(10 * pulse.ns, 0.1))


   for event in drives(two_pulses):
       print(event["start_cycles"], event["duration_cycles"], event["end_cycles"])
   # 0 20 20
   # 30 20 50

.. _conventions-frames:

Frames and signs
----------------

A tone drives its qudit with ``H = (A/2)(E a^dagger + E^* a)`` in the
qudit's frame, which rotates at the qudit's 0-1 frequency, with
``E(t) = amplitude envelope(t) exp(-i theta(t))`` and ``theta`` the tone's
frame phase (see :doc:`operations`). Each rule below follows from it.

.. test: test_a_real_amplitude_turns_about_x_and_an_imaginary_one_about_y

**Drive axis.** A pulse of positive real amplitude turns the qudit about
+X: an X90 takes ``|0>`` to ``(|0> - i|1>)/sqrt(2)``, Bloch vector
(0, -1, 0). An amplitude ``a exp(i psi)`` turns it about the axis at
``+psi``, so ``a * 1j`` turns about +Y and takes ``|0>`` to (+1, 0, 0).

.. code-block:: python

   @pulse.kernel
   def x90(qudit):
       drive(get_drive_line(qudit), square(2 * pulse.ns, math.pi / 4))


   @pulse.kernel
   def y90(qudit):
       drive(get_drive_line(qudit), square(2 * pulse.ns, math.pi / 4 * 1j))


   print(bloch(x90))   # [0.0, -1.0, 0.0]
   print(bloch(y90))   # [1.0, 0.0, 0.0]

.. test: test_shift_phase_is_a_virtual_z_that_turns_later_pulses_by_minus_phi

**Frame phase.** ``shift_phase(tone, +phi)`` is a virtual Z: the pulses
after it turn about the axis at ``-phi``, which is the logical rotation
``Rz(+phi)`` on the qudit (McKay et al., PRA 96, 022330 (2017)).
``shift_phase(tone, phi)`` then ``square(t, a)`` plays as
``square(t, a * exp(-1j * phi))``, so ``shift_phase(tone, pi / 2)`` then an
X90 is an X90 about -Y:

.. code-block:: python

   @pulse.kernel
   def phase_then_x90(qudit, phi: float):
       d = get_drive_line(qudit)
       shift_phase(d.tone, phi)
       drive(d, square(2 * pulse.ns, math.pi / 4))


   print(bloch(phase_then_x90, math.pi / 2))   # [-1.0, 0.0, 0.0]

.. test: test_the_simulated_state_is_in_the_qudit_frame

**Simulated state.** ``pulse.simulate``'s ``density_matrix`` is in the
qudit's frame. A frame change acts only on the pulses after it, so a
``shift_phase`` after the last pulse leaves ``density_matrix`` as it was;
the logical state is ``Rz(phi) rho Rz(phi)^dagger`` for the phase ``phi``
the tone has accumulated. Likewise, a tone detuned from its qudit does not
turn the simulated state between pulses: its frame phase advances, at
``theta(t) = theta_0 + (w_tone - w_q) t``.

.. code-block:: python

   @pulse.kernel
   def x90_then_phase(qudit):
       d = get_drive_line(qudit)
       drive(d, square(2 * pulse.ns, math.pi / 4))
       shift_phase(d.tone, math.pi / 2)


   print(bloch(x90_then_phase))   # [0.0, -1.0, 0.0], as after the X90 alone

.. test: test_a_virtual_detuning_cancels_an_offset_of_the_same_sign

**Virtual detuning.** In a Ramsey sequence whose tone is
``D = f_qudit - f_tone`` below the qudit, advancing the frame by
``shift_phase(tone, 2 pi v tau)`` before the second pulse, ``tau`` after
the first, gives fringes at ``|D - v|``: a virtual detuning ``v`` cancels
an offset ``D`` of the same sign. A pair at ``+v`` and ``-v``, with
``|v| > |D|``, gives fringes ``v - D`` and ``v + D``, so ``D`` is half
their difference, the ``-v`` fringe minus the ``+v`` one. Here 1 MHz over
250 ns is a quarter turn: cancelled, the two X90s make an X; doubled, the
half turn between them undoes the first:

.. code-block:: python

   @pulse.kernel
   def ramsey(qudit, offset_hz: float, virtual_hz: float):
       d = get_drive_line(qudit)
       set_frequency(d.tone, 5.0e9 - offset_hz)
       drive(d, square(2 * pulse.ns, math.pi / 4))
       wait(d, 248 * pulse.ns)   # the X90s start 250 ns apart
       shift_phase(d.tone, 2 * math.pi * virtual_hz * 250e-9)
       drive(d, square(2 * pulse.ns, math.pi / 4))


   print(bloch(ramsey, 1e6, 1e6)[2])    # -1.0: v = D cancels the offset
   print(bloch(ramsey, 1e6, -1e6)[2])   # 1.0: v = -D doubles it

.. test: test_shift_frequency_raises_the_tone

**Frequency shifts.** ``shift_frequency(tone, +delta)`` raises the tone by
``delta``: the export plays the tone at ``f + delta``, and a qudit that was
at the tone is then ``delta`` below it (``D = -delta``).

.. test: test_a_frequency_shifted_over_a_wait_and_back_leaves_the_phase_it_gained

**Frequency excursions.** A frequency change is made where the program
reaches it on the tone's line, after the waits before it there, and the
frame phase runs on through it. ``shift_frequency(tone, +delta)``, a wait
of ``T`` and ``shift_frequency(tone, -delta)`` leave the frame
``2 pi delta T`` further on, which the pulses after them carry in
``phase_rad``, as with a pulse of the same length in place of the wait. A
change made after the wait does not act during it. Here 1 MHz over 250 ns
is a quarter turn, so the second X90 turns about -Y and leaves the first's
state as it was:

.. code-block:: python

   @pulse.kernel
   def excursion(qudit, delta_hz: float):
       d = get_drive_line(qudit)
       drive(d, square(2 * pulse.ns, math.pi / 4))
       shift_frequency(d.tone, delta_hz)
       wait(d, 250 * pulse.ns)
       shift_frequency(d.tone, -delta_hz)
       drive(d, square(2 * pulse.ns, math.pi / 4))


   print(round(drives(excursion, 1e6)[1]["phase_rad"], 4))   # 1.5708, 2 pi (1 MHz) (250 ns)
   print(bloch(excursion, 1e6))   # [0.0, -1.0, 0.0], as after the first X90
   print(bloch(excursion, 0.0))   # [0.0, 0.0, -1.0], an X without the excursion

.. test: test_an_exported_frame_phase_multiplies_the_samples_by_exp_minus_i_phase
.. test: test_an_exported_frame_phase_counts_the_detuning_from_the_line_frequency

**Export phase.** A drive event of ``export_to_controller()`` gives the
tone's ``frequency_hz``, its frame phase ``phase_rad`` and the waveform's
``amplitude`` as written. ``phase_rad`` is the tone's phase as the event
starts, in the frame that turns at the line's own frequency
(``native_frequency_hz``) from the start of the shot: the ``shift_phase``
and ``set_phase`` before it, plus the phase the tone gained while its
frequency differed from the line's, as in the excursion above. The complex
samples it plays in the tone's frame are ``amplitude * envelope(t) *
exp(-i phase_rad)``: a backend that renders samples multiplies by
``exp(-1j * phase_rad)``, not ``exp(+1j * phase_rad)``.

.. code-block:: python

   @pulse.kernel
   def phase_then_pulse(qudit, phi: float):
       d = get_drive_line(qudit)
       shift_phase(d.tone, phi)
       drive(d, square(8, 0.3))


   (event,) = drives(phase_then_pulse, 0.7)
   print(round(event["phase_rad"], 3), event["amplitude"])   # 0.7 {'re': 0.3, 'im': 0.0}
   amplitude = complex(event["amplitude"]["re"], event["amplitude"]["im"])
   print(np.round(amplitude * np.exp(-1j * event["phase_rad"]), 3))   # (0.229-0.193j)

.. test: test_a_readout_result_is_the_level_and_if_takes_level_1

**Readout results.** A readout's result is the level measured, 1 for a
qudit found excited, and ``if result:`` takes its arm then:
``SimulationResult.branches`` gives the levels of each path.
