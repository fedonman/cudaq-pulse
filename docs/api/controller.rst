..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Controller Backends API
=======================

See :doc:`../user_guide/controller_backends` for the exported program and
how backends are found, and :doc:`errors` for ``ControllerLimitError`` and
the ``BackendError`` exceptions.

Export and limits
-----------------

.. automethod:: cudaq_pulse.ScheduledKernel.export_to_controller
   :no-index:

.. autodata:: cudaq_pulse.controller.FORMAT

   The ``format`` of the program that ``export_to_controller()`` returns by
   default: the schema's name and, after the slash, its major version.

.. autoclass:: cudaq_pulse.controller.DriveLine
   :members: qudits, qudit, delay_ns, max_concurrent_tones, broadcast, frequency_hz

.. autoclass:: cudaq_pulse.controller.ReadoutLine
   :members: qudits, qudit, delay_ns, max_concurrent_tones

.. autoclass:: cudaq_pulse.controller.UpdateBus
   :members: as_dict

.. autoclass:: cudaq_pulse.controller.WaveformMemoryScope
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.Event
   :members:
   :undoc-members:

.. autofunction:: cudaq_pulse.controller.check

.. autofunction:: cudaq_pulse.controller.export

.. autofunction:: cudaq_pulse.controller.unroll

.. autofunction:: cudaq_pulse.controller.sample_waveform

.. autofunction:: cudaq_pulse.controller.evaluate_expression

.. autoclass:: cudaq_pulse.controller.ExpressionTerm
   :members:
   :undoc-members:

.. autofunction:: cudaq_pulse.controller.shape_key

What classical functions cost
-----------------------------

``processor.usage(fn)`` (:meth:`cudaq_pulse.Processor.usage`) is what one
classical function costs on a processor on its own: a ``FunctionUsage``
with its cycles, and its time when the processor has a clock.

.. autoclass:: cudaq_pulse.controller.FunctionCost
   :members:

.. autoclass:: cudaq_pulse.controller.OpClass
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.controller.OpCounts
   :members: total, replace

.. autoclass:: cudaq_pulse.controller.OpCycles
   :members: replace

What a program uses of each processor and link
----------------------------------------------

``scheduled.controller_usage`` is :func:`~cudaq_pulse.controller.usage` of a
lowered program: each processor's functions and totals, and each link's
calls, slots and bytes per shot (``links``). A program over a processor's
limits does not compile: its ``ControllerLimitError`` carries the report as
``usage``, with each limit exceeded and by how much in ``usage.exceeded``.

.. autofunction:: cudaq_pulse.controller.usage

.. autoclass:: cudaq_pulse.controller.UsageReport
   :members:

.. autoclass:: cudaq_pulse.controller.LimitExceeded
   :members:

.. autoclass:: cudaq_pulse.controller.ControllerUsage

.. autoclass:: cudaq_pulse.controller.ProcessorUsage

.. autoclass:: cudaq_pulse.controller.FunctionUsage
   :members:

.. autoclass:: cudaq_pulse.controller.LinkUsage

Backends
--------

.. autodata:: cudaq_pulse.backends.GROUP

   The entry-point group in which a backend's distribution registers it,
   under the backend's name.

.. autoclass:: cudaq_pulse.backends.Backend
   :members:

.. autofunction:: cudaq_pulse.backends.available

.. autofunction:: cudaq_pulse.backends.get

.. autofunction:: cudaq_pulse.backends.register

.. autofunction:: cudaq_pulse.backends.unregister

.. autofunction:: cudaq_pulse.backends.validate
