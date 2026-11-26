..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0


Quantum system components API
==============================

``cudaq_pulse.quantum_systems`` holds what ``pulse.quantum_system(name=...,
qudits=..., couplings=..., model=...)`` builds a quantum system
(:doc:`quantum_systems`) from: its qudits, couplings and decoherence, the
components of their drive and readout, their gate calibrations and the
simulation models, and the shipped quantum systems.

.. autoclass:: cudaq_pulse.quantum_systems.PhysicalQudit
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.quantum_systems.Offset

Drive and readout
-----------------

A qudit's ``drive`` and ``readout`` are components of their modality: a
``MicrowaveDrive`` and a ``DispersiveReadout`` for a transmon. A drive of
another modality, or of none the system states, is a plain ``Drive``:
``Drive(amplitude_scale_rad_per_ns=s)`` gives the Rabi rate of a pulse of
amplitude 1, which is all the simulator needs of a drive. They hold the
qudit's physics only; the pulses it is tuned for are its calibrations
(below). They are read-only values; ``component.replace(field=value)`` makes
a changed copy.
Simulation picks its model by the component's type and refuses one it has no
model for. In a system file each is a mapping with its ``kind``
(``"microwave"`` or, for a plain ``Drive``, ``"drive"``; ``"dispersive"``)
and its fields.

.. autoclass:: cudaq_pulse.quantum_systems.Drive
   :members:

.. autoclass:: cudaq_pulse.quantum_systems.MicrowaveDrive
   :members:

.. autoclass:: cudaq_pulse.quantum_systems.Readout
   :members:

.. autoclass:: cudaq_pulse.quantum_systems.DispersiveReadout
   :members:

Couplings
---------

.. autoclass:: cudaq_pulse.quantum_systems.Coupling
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.quantum_systems.ResidualZZ
   :members:
   :undoc-members:

Gate calibrations
-----------------

The pulses a quantum system is tuned for are calibrations, by gate
(``pulse.gates``): a qudit's ``calibrations=`` give its ``X`` (and, if
published, ``X90``, ``Y``, ``Y90``) as a ``DragPulse``, its ``READOUT`` as
a ``ReadoutPulse`` and, if published, its ``Z``, ``S`` and ``T`` as a
``VirtualZ``; a coupling's ``calibrations=`` give its ``CZ`` as a ``FluxCZ``. A key
``(gate, (lower, upper))`` holds another transition's. ``get_calibration``
reads them, deriving what is not published; neither a qudit nor a coupling
has another read of them:

.. code-block:: python

   import cudaq_pulse as pulse
   from cudaq_pulse.quantum_systems import DragPulse, PhysicalQudit, ReadoutPulse

   qubit = PhysicalQudit(
       0, 5.0e9, 50.0, 30.0, anharmonicity_hz=-2.0e8,
       calibrations={
           pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0),
           pulse.gates.READOUT: ReadoutPulse(duration_ns=300.0, integration_ns=400.0),
       },
   )
   qubit.get_calibration(pulse.gates.X90)   # X at half the amplitude of a pi rotation

In a system file a qudit's or coupling's ``calibrations`` is a list of
entries, each its ``gate``, its ``transition`` if not ``[0, 1]``, and its
fields. A qudit's ``metadata`` is the user's own data (JSON values): the
library never reads it, it does not make two quantum systems differ, and system
files keep it.

.. automodule:: cudaq_pulse.gates
   :members: Gate

.. autoclass:: cudaq_pulse.quantum_systems.DragPulse
   :members:

.. autoclass:: cudaq_pulse.quantum_systems.VirtualZ
   :members:

.. autoclass:: cudaq_pulse.quantum_systems.ReadoutPulse
   :members:

.. autoclass:: cudaq_pulse.quantum_systems.FluxCZ
   :members:

Simulation models
-----------------

A quantum system's ``model`` says how the simulator treats its qudits and how many
levels each has. Each model class declares its levels; an instance sets a
qudit's own, keyed by the system's qudit index:

.. code-block:: python

   from cudaq_pulse.quantum_systems import PhysicalQudit, TransmonModel

   qubits = {
       i: PhysicalQudit(index=i, frequency_hz=f, anharmonicity_hz=-2.0e8, t1_us=50.0, t2_star_us=30.0)
       for i, f in enumerate([5.0e9, 5.2e9])
   }
   # Qudit 0 has four levels; qudit 1 has the model's default.
   chip = pulse.quantum_system(name="chip", qudits=qubits, model=TransmonModel(levels={0: 4}))

The model also gives a drive line its transitions: ``line.tones`` has a tone
for each ``(lower, upper)`` pair of the qudit's levels, at the frequency
``transitions_hz`` gives it, and the line's own tone, ``line.tone``, is the
one of ``default_transition``. A model numbers its levels from 0; one whose
levels have names says in its docstring which number each name has.

.. autoclass:: cudaq_pulse.quantum_systems.SimulationModel
   :members: levels, default_transition, levels_of, energies_hz, transitions_hz

.. autoclass:: cudaq_pulse.quantum_systems.TransmonModel

Shipped quantum systems
-----------------------

.. autofunction:: cudaq_pulse.quantum_systems.transmon_surface17

.. autodata:: cudaq_pulse.quantum_systems.transmon.CZ_LAYERS
   :no-value:

.. autofunction:: cudaq_pulse.quantum_systems.transmon_lattice

Registering quantum systems
---------------------------

.. autofunction:: cudaq_pulse.quantum_systems.register_quantum_system

.. autofunction:: cudaq_pulse.quantum_systems.available_quantum_systems
