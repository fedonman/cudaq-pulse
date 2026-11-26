..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Quantum systems
===============

A *quantum system* is the calibrated physics of a machine (qudit
frequencies and anharmonicities, couplings, residual ZZ, T1/T2, the pulses
its gates are calibrated for, and the pulse clock) together with a
*layout*, which maps the qudit numbering a program uses to its carriers.
It is read-only.

.. code-block:: python

   import cudaq_pulse as pulse

   qs = pulse.quantum_system("transmon_surface17")                  # shipped quantum system
   chip = pulse.quantum_system("transmon_surface17", layout=[4, 5, 6])
   chip = pulse.QuantumSystem.load("chip.json")              # the same, from a system file

   q = chip.qudits[0]          # a handle naming carrier 4
   pair = chip.qudits[0:2]     # a Register: carriers 4 and 5
   some = chip.qudits[[0, 2]]  # carriers 4 and 6

   chip.qudit_freq[0]                 # 0-1 frequency of qudit 0, in Hz
   chip.transition_freq[0, 1, 2]      # 1-2 frequency of qudit 0
   chip.qudit_freq[0] = 5.01e9        # ValueError: assignment destination is read-only

``pulse.quantum_system`` takes a registered name, or builds a quantum system
from ``name=`` and its ``qudits=``, ``couplings=``, ``residual_zz=``,
``model=`` and ``attribution=``; either way optionally ``layout=``,
``calibration=`` and ``measurement_run=``, ``control_system=`` and ``clock_ghz=``.
A string is always a name: ``pulse.QuantumSystem.load(path)`` reads a system file
(as ``QuantumSystem.as_dict()`` writes one, and ``json.dumps`` saves), and
``pulse.QuantumSystem.from_dict(data)`` its data, with the same keys.
``layout[i]`` is the carrier that qudit ``i`` names; by default every
carrier, in order. The ``model`` (``TransmonModel`` by default) decides the
operator of the couplings, for transmons the exchange
g (a^dagger b + a b^dagger):

.. code-block:: python

   from cudaq_pulse.quantum_systems import Coupling, FluxCZ, PhysicalQudit

   cz = FluxCZ(duration_ns=98.0)
   qpu = pulse.quantum_system(
       name="pair",
       qudits={
           i: PhysicalQudit(i, f, 50.0, 30.0, anharmonicity_hz=-2.0e8)
           for i, f in enumerate([4.8e9, 6.9e9])
       },
       couplings=[Coupling((0, 1), strength_hz=5.7e6, calibrations={pulse.gates.CZ: cz})],
   )

A ``PhysicalQudit`` takes its index, 0-1 frequency (Hz), T1 and T2* (us) in
that order, and the rest by keyword. ``anharmonicity_hz`` is left out where
a qudit has none to give; a transmon of more than two levels needs it, and
compiling or simulating one without it is an error naming the qudit.

A system that does not state its qudits' frequencies gives them in one of
two other ways, and ``qs.frequency_frame`` says which (``"absolute"`` for a
system with frequencies):

- **Offsets from a shared reference** (``"shared_reference"``):
  ``PhysicalQudit(1, Offset(1.5e8), 50.0, 30.0)`` is 150 MHz above the qudit
  at ``Offset(0.0)``. ``Offset`` (``cudaq_pulse.quantum_systems``) is a
  ``float`` that says what it is measured from; an offset may be 0 or
  negative. Every difference of two frequencies is known, so couplings, a
  drive at another qudit's tone and everything else compute as on the same
  system with frequencies, and every frequency of its programs is an offset
  from that reference: a line's tone starts at its qudit's offset, and the
  exports give offsets.
- **Each qudit's own carrier** (``"own_carrier"``):
  ``PhysicalQudit(0, None, 50.0, 30.0)`` leaves the frequency out, for a
  control stack that owns the carriers and takes offsets only. A line's
  tone is at 0 Hz, ``set_frequency(tone, 1e6)`` puts it 1 MHz above its
  qudit, and the exports give 0 Hz as each line's ``native_frequency_hz``.
  Such a system declares no ``couplings`` (a coupling acts by the
  difference of its qudits' frequencies; an interaction is given as a
  number, ``residual_zz``), and a drive at another qudit's tone is an
  error naming the two qudits.

A system's qudits give their frequencies all one way; 0 Hz is not a
frequency. A system file holds an offset as ``{"offset_hz": 1.5e8}`` and a
missing frequency as ``null``. Either kind of program schedules, exports
and simulates as that of the same system with frequencies does, less the
reference. ``frequency_hz`` reads what the qudit was given (an ``Offset``,
or ``None``, where the type says ``float``); ``qs.qudit_freq`` and
``qs.frequencies`` are plain numbers, the offset or 0. An ``Offset`` equals
and is ordered against another ``Offset`` only: ``offset == 1.5e8`` is
false and ``offset < 1.5e8`` is an ``InvalidTypeError`` (a ``TypeError``),
as for a ``str`` and a number; ``float(offset)`` is the number.

What needs a frequency such a system does not state is refused, each with
an error that says so:

- a readout's ``frequency_hz`` that is not an offset from the same
  reference: on a ``"shared_reference"`` system it is given as
  ``DispersiveReadout(frequency_hz=Offset(2.0e9))``, and a plain number is
  refused; an ``"own_carrier"`` system takes none. Without one, a readout
  line plays at its qudit's own tone: on an ``"own_carrier"`` system that
  is 0 Hz from the qudit's carrier. An ``Offset`` readout frequency on a
  system whose qudits give their frequencies in Hz is refused too, when
  the system's first program compiles;
- ``simulate(program, quantum_system=other)`` when ``other.frequency_frame``
  is not that of the system the program was compiled for;
- a ``Calibration`` of a system with couplings, which computes each
  coupling's exchange at its qudits' absolute frequencies;
- parameter drift.

On an ``"own_carrier"`` system a line several qudits share counts each
qudit's tones apart (``max_frequencies_per_line``): a tone there is a qudit
and an offset from its carrier, so three qudits' own tones, each 0 Hz, are
three frequencies.

``qs.transition_freq[q, i, j]`` is the upper level's energy less the
lower's, as ``line.tones`` and ``qs.model.transitions_hz(qudit, levels)``
give it: where frequencies are offsets it keeps its sign (a 1-2 tone is
its anharmonicity below the 0-1 tone).

A coupling's ``strength_hz`` is g / 2 pi; ``None`` says it is not measured
yet, and such a system compiles, schedules and exports while
``pulse.simulate`` refuses it, naming the coupling. ``residual_zz`` lists
any ZZ between two qudits beyond what the couplings produce, as
``ResidualZZ((0, 1), rate_hz=-2.0e4)``, rate zeta / 2 pi with
zeta = E11 - E10 - E01 + E00. The values are checked when the system is
built, and one ``QuantumSystemDefinitionError`` lists every problem at its path
(``couplings[1].qudits[0]``).

``qs.replace(...)`` makes a variant of a quantum system, as
``dataclasses.replace`` does: the same build keywords, each defaulting to
the system's own value (``qudits`` to its ``physical_qudits``). Passing a
``QuantumSystem`` to ``pulse.quantum_system`` is a ``TypeError``.

.. code-block:: python

   q0 = qs.physical_qudits[0]
   q0 = q0.replace(frequency_hz=q0.frequency_hz + 1e6)
   detuned = qs.replace(qudits={**qs.physical_qudits, 0: q0})
   first_two = qs.replace(layout=[0, 1])

Arguments or a system file with problems raise ``QuantumSystemDefinitionError``, a
``ValueError``, which lists every problem rather than the first. Each is a
``pulse.Problem`` with its path (the argument, or its place in the file),
the offending value and a fix; for a file, the line and column too:

.. code-block:: text

   QuantumSystemDefinitionError: 2 problems in quantum system file lab.json:
     lab.json:3:20: layout[2]: carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)
     lab.json:4:22: measurement_run: 'x' is given without a calibration; add "calibration", a shipped name or a file path, or remove "measurement_run"

``python -m cudaq_pulse.doctor --system lab.json`` runs the same checks
without a script, one problem per line; ``--json`` prints them as JSON, and
it exits 1 when the file has a problem.

Qudit handles
-------------

A ``Qudit`` names a carrier of one quantum system; it is identity, not location. A
kernel called with handles runs on their quantum system. Passing handles of two
quantum systems to one call, or handles of another quantum system to a kernel compiled for
one, raises ``QuantumSystemMismatchError``, and passing the same qudit twice raises
``TypeError``.

Quantum systems compare by content: two quantum systems built from the same values are
equal and share compiled kernels, but each is its own quantum system, so their
handles do not mix in one call. A layout is part of the content, and an
empty one is a ``QuantumSystemDefinitionError``: leave ``layout`` out for every
carrier.

To change a frequency, edit the quantum system, or pass the frequency to the kernel
as a ``float`` argument and set it on the tone with ``set_frequency``; the
argument stays symbolic, so a frequency sweep compiles once.

Gate calibrations
-----------------

The pulses a quantum system is tuned for are its gate calibrations, named by
``pulse.gates`` (``X``, ``X90``, ``Y``, ``Y90``, ``Z``, ``S``, ``T``,
``READOUT``, ``CZ``). In a
kernel a line gives its qudit's, ready to play; outside one, a qudit handle
and a coupling give the same values:

.. code-block:: python

   qs = pulse.quantum_system("transmon_surface17")

   @pulse.kernel
   def x_then_measure(q):
       d = get_drive_line(q)
       r = get_readout_line(q)
       x = d.get_calibration(pulse.gates.X)        # a DragPulse
       drive(d, x)
       drive(d, x.replace(amplitude=x.amplitude / 4))
       sync(d, r)
       readout(r, r.get_calibration(pulse.gates.READOUT))

   qs.qudits[4].get_calibration(pulse.gates.X).amplitude         # 0.1313: a pi rotation
   a, d = qs.qudit("X1"), qs.qudit("D2")
   qs.coupling(a, d).get_calibration(pulse.gates.CZ).duration_ns   # 98.0

What a quantum system does not publish is derived: an ``X`` without an amplitude
gets the one that makes its area a pi rotation at the qudit's drive scale,
``X90`` is ``X`` at half the amplitude, and ``Y`` and ``Y90`` are ``X`` and
``X90`` at +90 degrees. ``Z``, ``S`` and ``T`` are virtual Z rotations by
pi, pi / 2 and pi / 4, each a ``VirtualZ`` that ``drive`` plays as a phase
shift of the line's tone, with no pulse (see :ref:`virtual-z`). A ``CZ`` or
``READOUT`` the quantum system does not publish
is a ``NotFoundError`` naming the coupling or qudit, and so is another
transition's calibration (``transition=(1, 2)``) it does not give. A ``FluxCZ`` does not play:
flux lines are not modeled yet, so a kernel leaves the qudits idle for its
``duration_ns``.

The ``transmon_surface17`` calibration (below) gives each qubit an ``X`` of
its 1-2 transition, which plays at that transition's tone, for example to
return a qubit from ``|2>`` to ``|1>`` before a herald or a reset:

.. code-block:: python

   @pulse.kernel
   def repair_leakage(q):
       d = get_drive_line(q)
       drive(d, d.get_calibration(pulse.gates.X, transition=(1, 2)), tone=d.tones[(1, 2)])

``calibrations`` of a ``PhysicalQudit`` or ``Coupling`` gives what it
publishes and nothing derived, read-only and keyed as ``calibrations=``
takes them, so a calibration step can change one and keep the rest:

.. code-block:: python

   q = qs.physical_qudits[4]
   x = q.calibrations[pulse.gates.X]                          # a DragPulse, as published
   tuned = q.replace(calibrations={**q.calibrations, pulse.gates.X: x.replace(amplitude=0.13)})
   qs = qs.replace(qudits={**qs.physical_qudits, 4: tuned})

The process quantum system
--------------------------

A script can leave the choice of quantum system to the command line with a *process
quantum system*:

.. code-block:: bash

   CUDAQ_PULSE_QUANTUM_SYSTEM=transmon_surface17 python script.py
   python -m cudaq_pulse --quantum-system transmon_lattice:4 script.py --script-args
   python -m cudaq_pulse --quantum-system transmon_lattice:4 --quiet -m package.module --module-args

The launcher sets ``CUDAQ_PULSE_QUANTUM_SYSTEM`` for the script and passes the
script's own arguments through unparsed. With ``-m MODULE`` it runs a module instead,
as ``python -m`` does, and everything after the module name, dashes included, goes to
the module; the launcher's own options (``--quantum-system``, ``--quiet``, which drops
the banner it prints to stderr) come before ``-m``. ``pulse.quantum_system()`` with no
argument returns the process quantum system, whose handles the script then
passes to its kernels; a kernel call with no qudit handles always raises
``NoQuantumSystemError``.

Registered names are ``transmon_surface17`` and ``transmon_lattice:N``
(``transmon_lattice:N:ring`` and ``transmon_lattice:N:grid`` for the other
topologies), and a system's ``name`` is the name it is registered under:
``transmon_lattice(4).name`` is ``"transmon_lattice:4"``.
``cudaq_pulse.quantum_systems.register_quantum_system`` adds more.

``transmon_surface17`` is the distance-three surface-code quantum system of
Krinner et al., Nature 605, 669 (2022), built from the values the paper
publishes. Its qudits 0-16 are D1-D9, X1-X4 and Z1-Z4 (``PhysicalQudit.label``), and
its 24 couplings are the auxiliary-data pairs of the stabilizers, each with
its CZ's duration, buffers and interaction frequency (its ``CZ``
calibration, below).
``cudaq_pulse.quantum_systems.transmon.CZ_LAYERS`` gives the CZ time steps of a
cycle, each a tuple of (auxiliary, data) qudit pairs.
``python -m cudaq_pulse.doctor`` lists them and reports the process quantum system.
:doc:`transmon_surface17` shows the chip's layout, frequencies, coherence
and calibrated gates in tables and figures.

The paper's J / 2 pi ~ 7 MHz is the ``|11> <-> |20>`` coupling of the CZ at
each pair's interaction point (N. Lacroix, PhD thesis, ETH Zurich (2025),
Diss. ETH No. 31121, https://doi.org/10.3929/ethz-b-000740393, on the same
device); X2-D4's is measured, 7.4 MHz at 4.4 GHz (Hellings et al.,
arXiv:2503.04610). The simulated exchange comes from the
quantum system's calibration (below): one system-wide charge coupling g_c, and the
exchange J_qq = g_c <0|n_a|1> <0|n_d|1> at the idle points is 5.67-6.08 MHz.
The idle ZZ of the charge-coupled pairs is -4.5 to -7.3 kHz
(``cudaq_pulse.quantum_systems.transmon.transmon_surface17_idle_zz_hz``), below the
paper's "below about 8 kHz"; simulated with three levels, the couplings
reproduce it within a few percent. Each qubit also records its Hahn-echo T2
and thermal excited-state population from Tab. I (``QuantumSystem.t2_echo_times``,
``QuantumSystem.thermal_populations``): a simulation starts each qubit in its
thermal mixture and relaxes it there, and dephases at the Hahn-echo T2 with
``simulate(..., dephasing="t2_echo")``.

Calibrations
------------

A calibration holds the values that are fitted rather than published, apart
from the quantum system: for the ``transmon_surface17`` quantum system, the shipped
``transmon_surface17`` calibration holds each qubit's circuit (E_C, E_J,max and the SQUID asymmetry d, fitted to Tab. I and the
other sweet spot of Fig. 1c), each pair's charge coupling, the flux-line
model of Hellings et al., the in-model CZ knobs of all 24 pairs, each
qubit's single-qubit pulses and readout assignment errors, crosstalk,
defect bands and feedline groups. Every value records its
``kind`` (``published``, ``digitized`` or ``fitted``), ``source``,
``uncertainty`` and ``measurement_run``, one of the file's measurement runs.

.. code-block:: python

   qs = pulse.quantum_system("transmon_surface17", calibration="transmon_surface17", measurement_run="2022-main")
   cal = qs.calibration
   cal.entry("qudits", "D4", "ec")        # value with its provenance
   cal.exchange_hz("X2-D4")               # J_qq at the idle points
   cal.exchange_hz("X2-D4", data_hz=4.4e9)  # ... with D4 tuned to 4.4 GHz
   cal.idle_zz_hz("X2-D4")
   cal.flux_for_detuning("D4", 500e6)     # flux quanta for a +500 MHz detuning
   cal.detuning_for_flux("X2", 0.25)

``calibration=`` is a shipped name, the path of a calibration file or a
``cudaq_pulse.quantum_systems.Calibration``; the ``transmon_surface17`` quantum system uses the
``transmon_surface17`` calibration by default. Its couplings, pulses and readout errors replace the system's, so quantum systems with different
calibrations compile separately. Detunings are in Hz; for qubits whose
circuit, including d, is calibrated (``Calibration.flux_qudits``, all 17
here) ``flux_for_detuning`` and ``detuning_for_flux`` convert to and from
flux quanta. Every value in the shipped file is published, digitized from a
published figure, or fitted, and cites its source; fitted values also name
the fitting method.

The coupling is one system-wide charge coupling, g_c = 3.12 MHz
(``coupling.charge_coupling_g_c.calibration_A``), fitted so that X2-D4's
``|11> <-> |20>`` coupling is Hellings et al.'s measured 7.4 MHz at 4.4 GHz.
At each pair's own interaction point it gives J_2 = 7.3-9.6 MHz, the idle
exchange is 5.67-6.08 MHz and the idle ZZ -4.5 to -7.3 kHz. The CZ knobs
are fitted with the same coupling, so the simulated quantum system and its gates
agree.

The CZ knobs (``cz.knobs.per_pair``) are those of the sudden-net-zero pulse
the quantum system uses (Negirneac et al., PRL 126, 220502 (2021)): the plateau
duration t_p, the 2.5 ns transition section t_mid, its amplitude B and the
amplitude offset, fitted in a closed two-transmon model (five levels each,
no rotating-wave approximation, ideal flux line, no spectators) for a
conditional phase of 180 degrees and the least leakage. Their mean
duration, 2 t_p + t_mid, is 65.0 +- 4.8 ns against the published 68(7) ns.
Each record names the coupling it was fitted with in ``coupling_used``,
the system-wide g_c. Each record keeps its predicted phase and leakage and its ``flags``:
nine pairs leak at least 1e-4, a floor set by the 0.417 ns time grid, and
Z2-D7 defaults to an unphysical B = 1.934 (its data qubit moves only 261
MHz), with the in-range alternatives in ``alternatives``.

The papers publish the single-qubit pulse shape (40 ns DRAG pulses, a
Gaussian of sigma = 10 ns truncated at +-2 sigma) but no DRAG coefficients
and no 1-2 pulse; the calibration fits them (``single_qubit_gates``) in a
closed Duffing model of each qubit with its published anharmonicity. Each
qubit's ``X`` (and the ``X90``, ``Y`` and ``Y90`` derived from it) takes
``beta_ns`` 0.51-0.58, which minimizes the leakage and phase error of ``X``
and ``X90`` together: in simulation with each qubit's T1 and T2*, ``X``
then errs by 9.6e-4 and ``X90`` by 8.6e-4 on average, against the
published single-qubit randomized-benchmarking error of 9e-4 (2.7e-3 for
``X`` with beta 0). The ``X`` of the 1-2 transition is 20 ns long
(sigma = 5 ns), with a fitted amplitude and beta; the duration minimizes the
device-mean error of the pulse, decoherence included, and it returns ``|2>``
to ``|1>`` with probability 0.997-0.999. ``beta_ns`` is fitted for these
durations: a pulse replaced with another duration needs its own.

Tab. I gives each qubit's two-state readout error, the mean of P(1|0) and
P(0|1) after a preselection readout, but not the two errors apart.
``readout.assignment`` keeps the published value and the assignment errors
of each qubit's ``DispersiveReadout``: one error for both states, fitted so
that ``simulate(..., readout="model")`` of the paper's characterization
(``|0>``, and ``|1>`` from ``X``, then the qubit's ``READOUT``) reproduces
it. The simulation relaxes the qubit during the readout pulse and takes the
outcome at its end, so the fitted error is the rest; for D2, D9 and X4 the
relaxation alone already exceeds the published error (0.51, 0.64 and 0.86 %
against 0.5, 0.5 and 0.8 %), and their assignment error is 0. Use
``readout="model"`` here: ``"model_with_decay"`` adds relaxation over the
readout again. ``transmon_surface17(calibration=None)``, the published
values alone, keeps beta 0, no 1-2 pulse and ideal assignment.
