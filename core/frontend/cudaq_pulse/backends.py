# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller backends installed beside cudaq-pulse.

A controller backend turns a compiled kernel into a program for one
controller and runs it there. It ships as its own distribution and
registers an entry point in the group ``cudaq_pulse.backends``, under the
backend's name::

    [project.entry-points."cudaq_pulse.backends"]
    toy = "toy_controller:ToyBackend"

The entry point names a ``Backend`` or a callable without arguments that
returns one. ``available()`` lists the installed backends without importing
them, and ``get(name)`` loads one. A backend follows the ``Backend``
protocol: ``control_system()`` is the control system to compile for (a
``ControlSystem``, or its one ``Controller``), so that programs the
controller cannot play fail at compile time;
``emit(program)`` takes ``ScheduledKernel.export_to_controller()`` and returns the
controller's program; ``run(artifact, shots_count=...)`` runs it::

    from cudaq_pulse import backends

    backend = backends.get("toy")
    system = backend.control_system()
    scheduled = pulse.compile(kernel, quantum_system=qs, control_system=system).lower(q)
    artifact = backend.emit(scheduled.export_to_controller())
    result = backend.run(artifact, shots_count=100)

A backend may also list the export formats it reads in ``formats``
(``"cudaq-pulse.controller/1"``, ``cudaq_pulse.controller.FORMAT``, and the
structured ``"cudaq-pulse.controller/2"`` of ``export_to_controller(version=2)``);
``get`` refuses one that reads none of them.
``register(name, factory)`` adds a backend in this process, without a
distribution. Nothing here runs until it is called: importing cudaq-pulse
neither imports this module nor looks for backends.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ._errors import InvalidValueError, NotFoundError, PulseError

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from ._system import Controller, ControlSystem
    from ._validation import _Report

__all__ = [
    "GROUP",
    "Backend",
    "BackendError",
    "BackendNotFoundError",
    "IncompatibleBackendError",
    "available",
    "get",
    "register",
    "unregister",
    "validate",
]

GROUP = "cudaq_pulse.backends"


@runtime_checkable
class Backend(Protocol):
    """A controller backend."""

    name: str

    def control_system(self) -> ControlSystem | Controller:
        """The control system to compile for this controller: a
        ``ControlSystem``, or its one ``Controller``."""
        ...

    def emit(self, program: Mapping[str, Any]) -> Any:
        """The controller's program for ``ScheduledKernel.export_to_controller()``."""
        ...

    def run(self, artifact: Any, *, shots_count: int = 1) -> Any:
        """Run an emitted program on the controller."""
        ...


class BackendError(PulseError):
    """A controller backend cannot be found or used."""


class BackendNotFoundError(BackendError, NotFoundError):
    """No backend of that name is installed or registered."""


class IncompatibleBackendError(BackendError, TypeError):
    """A backend fails to load, is not a ``Backend``, or reads none of the
    export formats this version writes."""


_lock = threading.Lock()
_registered: dict[str, Callable[[], Backend]] = {}
_loaded: dict[tuple[str, str], Backend] = {}


def _entry_points() -> dict[str, EntryPoint]:
    from importlib.metadata import entry_points

    return {ep.name: ep for ep in entry_points(group=GROUP)}


def available() -> list[str]:
    """The names of the backends installed or registered, sorted; none is
    imported."""
    with _lock:
        names = set(_registered)
    return sorted(names | set(_entry_points()))


def register(name: str, factory: Callable[[], Backend]) -> None:
    """Make *factory* the backend *name* in this process; it takes precedence
    over an installed backend of the same name."""
    if not name or not name.isidentifier():
        raise InvalidValueError(f"a backend name is an identifier, got {name!r}")
    with _lock:
        _registered[name] = factory
        _loaded.pop((name, "registered"), None)


def unregister(name: str) -> None:
    """Remove a backend ``register`` added."""
    with _lock:
        _registered.pop(name, None)
        _loaded.pop((name, "registered"), None)


def _instance(name: str, loaded: Any) -> Backend:
    backend = loaded
    if isinstance(loaded, type) or (not isinstance(loaded, Backend) and callable(loaded)):
        backend = loaded()
    if not isinstance(backend, Backend):
        raise IncompatibleBackendError(
            f"backend {name!r} is not a cudaq_pulse.backends.Backend: it needs a name, "
            "control_system(), emit() and run()"
        )
    formats = getattr(backend, "formats", None)
    if formats is not None:
        from .controller import _FORMATS

        if not set(_FORMATS) & set(formats):
            raise IncompatibleBackendError(
                f"backend {name!r} reads {', '.join(map(str, formats)) or 'no format'}; "
                f"this cudaq-pulse exports {' and '.join(_FORMATS)}"
            )
    return backend


def _load(name: str, found: EntryPoint) -> Any:
    try:
        return found.load()
    except Exception as e:
        source = f" from {found.dist.name} {found.dist.version}" if found.dist else ""
        raise IncompatibleBackendError(
            f"backend {name!r} ({found.value}{source}) failed to load: {type(e).__name__}: {e}"
        ) from e


def get(name: str) -> Backend:
    """The backend *name*, loaded on first use and kept."""
    with _lock:
        factory = _registered.get(name)
        key = (name, "registered")
        if factory is not None:
            if key not in _loaded:
                _loaded[key] = _instance(name, factory)
            return _loaded[key]
    found = _entry_points().get(name)
    if found is None:
        installed = ", ".join(available()) or "none"
        raise BackendNotFoundError(
            f"no controller backend {name!r} is installed (installed: {installed}); "
            f"a backend registers an entry point in the {GROUP!r} group"
        )
    key = (name, found.value)
    with _lock:
        backend = _loaded.get(key)
    if backend is None:
        backend = _instance(name, _load(name, found))
        with _lock:
            backend = _loaded.setdefault(key, backend)
    return backend


def validate(
    backend: Backend | str,
    controller: Controller | None = None,
    *,
    shots_count: int = 1000,
    tolerance: float = 0.05,
) -> _Report:
    """Run the export test corpus through *backend* (a ``Backend`` or the
    name of one) and report what it does wrong.

    The corpus has a program for each kind of event (``cudaq_pulse.Event``),
    one with a branch, ``answer_timeout``, which branches on whether a
    call's answer came in time and on the answer itself,
    ``timeout_function``, the same for a function declared with
    ``latency_mode="timeout"``, and two with
    values known only at run time (one a frequency change made ahead of
    the pulse that takes it), compiled
    for *controller* (by default the backend's control system's), in each
    format the backend reads. Four more are wired from that controller:
    ``shared_line``, both qudits read out at once on one readout line,
    ``shared_drive_line_two_tones`` and ``shared_drive_line_one_tone``, both
    driven at once on one drive line (the second on a system whose qudits
    share a frequency), and ``several_controllers``, on two controllers made from it, one of them
    reading out both qudits on such a line, each controller's part in format
    ``/1`` (not compared with ``emulate``). Each must be emitted or refused with
    ``IncompatibleBackendError``; a program
    with an event kind no format has, and one in a newer format, must be
    refused. A program that exceeds the controller's limits is skipped.
    Every pulse and wait of the corpus lasts the fewest whole times as long
    that the controllers play: on their lines' timestep and no shorter than
    their ``min_pulse_cycles``; as written wherever they play it so.

    ``branch`` plays its pulse on the readout's outcome once the
    controller's ``feedback_latency_cycles`` have passed after the readout
    ends: every schedule waits for it.

    A backend may offer two more methods, which make the checks stronger.
    ``events(artifact)`` gives back the emitted programs, one list per
    controller, each event a mapping with its ``kind``, ``line`` (the link
    of a call or answer), ``start_cycles`` and optionally
    ``duration_cycles``, every execution in program order; the count of
    programs and the timing and order of plays, acquisitions, calls and
    answers on each line must then match the controller program, and the
    pulses must respect the controller's limits. ``emulate(artifact,
    shots_count=...)`` gives each shot's measurement outcomes, in the order they
    end; the fraction of 1s of each must match ``cudaq_pulse.simulate``
    within *tolerance* plus three standard errors.

    The report's ``ok`` is whether nothing failed; ``str()`` gives one line
    per program, such as ``FAIL unsupported_event: emit() accepted
    'forward_outcome'; it should raise IncompatibleBackendError``.
    """
    if shots_count < 1:
        raise InvalidValueError(f"shots_count must be positive, got {shots_count}")
    if not tolerance >= 0:
        raise InvalidValueError(f"tolerance must be at least 0, got {tolerance}")
    found = get(backend) if isinstance(backend, str) else backend
    if not isinstance(found, Backend):
        raise IncompatibleBackendError(
            f"{found!r} is not a cudaq_pulse.backends.Backend: it needs a name, "
            "control_system(), emit() and run()"
        )
    from . import _validation

    return _validation.validate(found, controller, shots_count, tolerance)
