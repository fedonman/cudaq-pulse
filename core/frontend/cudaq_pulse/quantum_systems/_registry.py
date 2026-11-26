# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The process quantum system, chosen outside user code.

``quantum_system()`` with no argument returns the quantum system named by
the ``CUDAQ_PULSE_QUANTUM_SYSTEM`` environment variable, which
``python -m cudaq_pulse --quantum-system NAME script.py`` sets for the script it
runs. A name is a registered quantum system, optionally with an argument after a
colon (``transmon_lattice:4``). There is no default: ``quantum_system()`` without
one raises ``NoProcessQuantumSystemError``. Kernel calls never read the process quantum system;
they run on the quantum system of their qudit handles.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._errors import InvalidTypeError, InvalidValueError, NotFoundError, PulseError, _closest
from .base import _DESCRIPTIONS, _Description

if TYPE_CHECKING:
    from .._device import QuantumSystem

ENV_VAR = "CUDAQ_PULSE_QUANTUM_SYSTEM"

QuantumSystemFactory = Callable[[str | None], Any]


class NoProcessQuantumSystemError(PulseError, RuntimeError):
    """``quantum_system()`` was called with no argument and ``CUDAQ_PULSE_QUANTUM_SYSTEM``
    names no process quantum system."""


class UnknownQuantumSystemError(NotFoundError):
    """A quantum system name, given to ``quantum_system()``, in a system file or in
    ``CUDAQ_PULSE_QUANTUM_SYSTEM``, is not registered; the message lists the
    registered ones."""


def _no_argument(name: str, factory: Callable[[], Any]) -> QuantumSystemFactory:
    def make(arg: str | None) -> Any:
        if arg is not None:
            raise InvalidValueError(f"quantum system {name!r} takes no argument, got {arg!r}")
        return factory()

    return make


def _lattice(name: str, factory: Callable[..., Any]) -> QuantumSystemFactory:
    def make(arg: str | None) -> Any:
        if arg is not None and arg.isdigit():
            return factory(int(arg))
        count, _, topology = (arg or "").partition(":")
        if not count.isdigit():
            raise InvalidValueError(
                f"quantum system {name!r} needs a qudit count, e.g. {name}:4 or {name}:4:ring"
            )
        return factory(int(count), topology=topology)

    return make


def _builtin_factories() -> dict[str, QuantumSystemFactory]:
    from .transmon import _described_lattice, _described_surface17

    return {
        "transmon_surface17": _no_argument("transmon_surface17", _described_surface17),
        "transmon_lattice": _lattice("transmon_lattice", _described_lattice),
    }


_lock = threading.Lock()
_factories: dict[str, QuantumSystemFactory] = {}
_resolved: dict[str, QuantumSystem] = {}


def _registry() -> dict[str, QuantumSystemFactory]:
    if not _factories:
        with _lock:
            if not _factories:
                _factories.update(_builtin_factories())
    return _factories


def register_quantum_system(name: str, factory: QuantumSystemFactory) -> None:
    """Make *factory* available as ``quantum_system("name[:arg]")`` and
    ``CUDAQ_PULSE_QUANTUM_SYSTEM=name[:arg]``.

    *factory* receives the text after the colon, or ``None``, and returns a
    ``QuantumSystem`` (or an object with ``to_quantum_system()``), as
    ``pulse.quantum_system(name=..., qudits=...)`` builds one; the name gives
    its qudits, couplings and model. Registering only adds a name: which
    quantum system a process uses is still chosen by the environment.
    """
    if not name or ":" in name or name != name.strip():
        raise InvalidValueError(f"invalid quantum system name {name!r}")
    registry = _registry()
    with _lock:
        registry[name] = factory
        for spec in [s for s in _resolved if s.partition(":")[0] == name]:
            del _resolved[spec]


def available_quantum_systems() -> list[str]:
    """Names that ``quantum_system()`` and ``CUDAQ_PULSE_QUANTUM_SYSTEM`` accept.

    The list is sorted, and it includes names added with
    ``register_quantum_system``. Some names take an argument after a colon:
    ``"transmon_lattice"`` on its own is an error — a lattice needs a qudit
    count, so pass ``"transmon_lattice:4"`` (or ``"transmon_lattice:4:ring"``
    for a ring topology).

    Example::

        available_quantum_systems()
        ['transmon_lattice', 'transmon_surface17']
    """
    registry = _registry()
    with _lock:
        return sorted(registry)


def make_quantum_system(spec: str) -> _Description:
    """The description of a registered name with its optional argument."""
    name, colon, arg = spec.partition(":")
    factory = _registry().get(name)
    if factory is None:
        names = available_quantum_systems()
        close = _closest(name, names)
        hint = f"did you mean {close!r}? " if close else ""
        raise UnknownQuantumSystemError(
            f"no registered quantum system {spec!r}; {hint}the registered names are "
            f"{', '.join(names)}"
        )
    target = factory(arg if colon else None)
    if target.__class__ in _DESCRIPTIONS:
        return target
    if hasattr(target, "to_quantum_system"):
        target = target.to_quantum_system()
    from .._device import QuantumSystem

    if not isinstance(target, QuantumSystem):
        raise InvalidTypeError(
            f"quantum system factory {name!r} returned {type(target).__name__}, not a QuantumSystem"
        )
    return target._description


def current_quantum_system() -> QuantumSystem:
    """The process quantum system; ``NoProcessQuantumSystemError`` if none is
    configured.

    Resolved once per distinct ``CUDAQ_PULSE_QUANTUM_SYSTEM`` value.
    """
    spec = os.environ.get(ENV_VAR)
    if not spec:
        raise NoProcessQuantumSystemError(_no_quantum_system_message())
    resolved = _resolved.get(spec)
    if resolved is None:
        from .._device import _calibrated

        fresh = _calibrated(make_quantum_system(spec), spec, f"env {ENV_VAR}={spec}", None, None)
        with _lock:
            resolved = _resolved.setdefault(spec, fresh)
    return resolved


def _no_quantum_system_message() -> str:
    return (
        "quantum_system() with no argument returns the process quantum system, and none is "
        'configured: name one, as quantum_system("transmon_surface17"), '
        "or set one of:\n"
        f"    {ENV_VAR}=transmon_surface17 python script.py\n"
        "    python -m cudaq_pulse --quantum-system transmon_surface17 script.py\n"
        f"Available quantum systems: {', '.join(available_quantum_systems())}"
    )
