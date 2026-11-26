# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels with list, qudit or register parameters: the module depends on a
list only through its length and, with runtime qudits, on the quantum system only
through its structure, and each call binds them. ``_kernel`` imports it for
the first such kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._device import Qudit, Register
from .._errors import InvalidTypeError
from .._programs import _PLACEMENTS, _Role, _structure
from .._units import Time
from ..compiler import CompileOptions, ScheduledKernel, build
from ._kernel import (
    _FLOAT,
    _QUDIT,
    _VECTOR,
    _Dispatcher,
    _Entry,
    _execute,
    _no_device,
    _passed_twice,
    _pin_mismatch,
    _QuditTwiceError,
    _static_key,
    _systems,
    _ticks,
)
from .ir_builder import Parameter

if TYPE_CHECKING:
    from .._device import QuantumSystem
    from ._kernel import Kernel


class _Placing(_Dispatcher):
    """The dispatcher of a kernel with list, qudit or register parameters:
    the module depends on lists only through their lengths and, with runtime
    qudits, on the quantum system only through its structure; each call binds them."""

    def _device(self, values: tuple[Any, ...], kernel: Kernel[...]) -> QuantumSystem:
        slotted = kernel._slotted
        if len(slotted) == 1 and not kernel._split[0]:
            value = values[slotted[0]]
            if type(value) is Qudit:
                quantum_system: QuantumSystem = value.quantum_system
                pin = self.pin
                if pin is None or pin is quantum_system:
                    return quantum_system
                if quantum_system != pin:
                    raise _pin_mismatch(kernel.__name__, pin, quantum_system)
                return pin
        found: QuantumSystem | None = None
        seen: set[int] = set()
        wrong = None
        try:
            for i in kernel._split[0]:
                found = _systems(values[i], found, kernel, seen)
            for i in kernel._slotted:
                value = values[i]
                if type(value) is not Qudit and not (
                    type(value) is Register or type(value) is list or type(value) is tuple
                ):
                    wrong = i
                    continue
                found = _systems(value, found, kernel, seen)
        except _QuditTwiceError as twice:
            names = kernel._names
            met = [*kernel._split[0], *kernel._slotted]
            raise _passed_twice(
                kernel.__name__, twice.qudit, [(names[i], values[i]) for i in met]
            ) from None
        pin = self.pin
        if wrong is not None:
            if found is None and pin is None:
                raise _no_device(kernel.__name__)
            raise InvalidTypeError(
                f"{kernel.__name__}(): argument {kernel._names[wrong]!r} must be a qudit handle "
                f"or a register of them, not {type(values[wrong]).__name__}"
            )
        if pin is not None:
            if found is not None and found != pin:
                raise _pin_mismatch(kernel.__name__, pin, found)
            return pin
        if found is not None:
            return found
        raise _no_device(kernel.__name__)

    def _key(self, target: QuantumSystem, values: tuple[Any, ...]) -> Any:
        kernel = self.kernel
        static = kernel._split[0]
        # With the control system (the identity holds it): compiled for no
        # control system, a kernel is scheduled and checked for the one of
        # its qudits.
        fingerprint = (
            (target._structure or _structure(target), target.control_system)
            if kernel._slotted
            else target._identity
        )
        lengths = [len(values[i]) for i in kernel._shaped]
        return (fingerprint, static, *[_static_key(values[i]) for i in static], *lengths)

    def lookup(self, values: tuple[Any, ...]) -> _Entry:
        kernel = self.kernel
        target = self._device(values, kernel)
        key = self._key(target, values)
        entry = self._entries.get(key)
        if entry is not None and entry.guard.check():
            entry.tick = next(_ticks)
            if target._drift:
                from ..quantum_systems.drift import _entry_for

                return _entry_for(entry, target)
            return entry
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.guard.check():
                if target._drift:
                    from ..quantum_systems.drift import _entry_for

                    return _entry_for(entry, target)
                return entry
            entry = kernel._specialize(values, target, self.options)
            self.misses += 1
            # Tracing may have made float parameters static.
            key = self._key(target, values)
            if target._drift:
                from ..quantum_systems.drift import _kept

                kept, entry = _kept(entry, target, kernel)
                self._insert(key, kept)
                return entry
            self._insert(key, entry)
        return entry

    def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        values = self.kernel._bind(*args, **kwargs)
        entry = self.lookup(values)
        return _execute(self._bound(entry, values), entry.returns)

    def lower(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ScheduledKernel:
        values = self.kernel._bind(*args, **kwargs)
        return self._bound(self.lookup(values), values)

    def _bound(self, entry: _Entry, values: tuple[Any, ...]) -> ScheduledKernel:
        kernel = self.kernel
        if entry.lowered._role is not None:
            return _relocate(kernel, entry.lowered, values, entry.runtime)
        if entry.runtime:
            return entry.lowered._with_values(kernel._runtime_values(values, entry.runtime))
        return entry.lowered


def _relocate(
    kernel: Kernel[Any], lowered: ScheduledKernel, values: tuple[Any, ...], runtime: tuple[int, ...]
) -> ScheduledKernel:
    """The call's view of a module with list or qudit parameters: its
    float values flattened, and the carrier of each slot."""
    kinds = kernel._kinds
    numbers: list[float] = []
    carriers: list[int] = []
    quantum_system = None
    for i in runtime:
        kind = kinds[i]
        v = values[i]
        if kind == _FLOAT:
            if type(v) is float:
                numbers.append(v)
            else:
                numbers.extend(kernel._runtime_values(values, (i,)))
        elif kind == _VECTOR:
            numbers.extend(_vector_values(kernel, v, i))
        elif kind == _QUDIT:
            carriers.append(v.index)
            quantum_system = v.quantum_system
        else:
            for q in v:
                carriers.append(q.index)
                quantum_system = q.quantum_system
    role = lowered._role
    placement = tuple(carriers)
    views = None
    if not numbers and role is not None and quantum_system is not None:
        views = role.views
        view = views.get((placement, quantum_system))
        if view is not None:
            return view
    # A module with list or qudit parameters is a _Placed or _CheckedPlaced.
    place: Callable[..., ScheduledKernel] = lowered.bind
    view = place(numbers, placement, quantum_system)
    if views is not None and quantum_system is not None and len(views) < _PLACEMENTS:
        views[(placement, quantum_system)] = view
    return view


def _vector_values(kernel: Kernel[Any], value: Any, i: int) -> list[float]:
    try:
        return [x if type(x) is float else _real(x) for x in value]
    except TypeError:
        pass
    name = kernel._names[i]
    for k, x in enumerate(value if isinstance(value, (list, tuple)) else ()):
        if x.__class__ is Time:
            raise InvalidTypeError(
                f"{kernel.__name__}(): {name}[{k}] is a time ({x}), and a list[float] "
                f"element is a plain number; pass {name} in nanoseconds and multiply "
                "its elements by pulse.ns in the kernel, or drop its list[float] "
                "annotation to compile the kernel for it"
            )
        if type(x) is not float:
            try:
                _real(x)
            except TypeError:
                raise InvalidTypeError(
                    f"{kernel.__name__}(): {name}[{k}] must be a real number, "
                    f"not {type(x).__name__}"
                ) from None
    raise InvalidTypeError(
        f"{kernel.__name__}(): argument {name!r} must be a list of real "
        f"numbers, not {type(value).__name__}"
    )


def _build_placed(
    kernel: Kernel[Any],
    fn: Any,
    trace_args: list[Any],
    runtime: tuple[int, ...],
    target: Any,
    options: CompileOptions,
) -> tuple[ScheduledKernel, Any]:
    """``build`` for a kernel with list, qudit or register parameters:
    *trace_args*, the call's values, gets their tracing stand-ins."""
    names = kernel._names
    kinds = kernel._kinds
    for i in kernel._declared_float:
        if i in runtime:
            continue
        value = trace_args[i]
        if value.__class__ is float:
            from ._durations import Cycles

            trace_args[i] = Cycles(value)
        elif kinds.get(i) == _VECTOR:
            from ._durations import Cycles

            # Each element is a float parameter's value, so a duration too.
            trace_args[i] = [Cycles(v) if v.__class__ is float else v for v in value]
    parameters: list[str] = []
    symbols: list[tuple[str, str, str, int | None]] = []
    slotted: list[tuple[str, str, str, int | None]] = []
    for i in runtime:
        kind = kinds[i]
        value = trace_args[i]
        if kind == _FLOAT:
            trace_args[i] = Parameter(names[i], len(parameters), "f64")
            parameters.append(names[i])
            symbols.append((names[i], "float", names[i], None))
        elif kind == _VECTOR:
            elements = []
            for k in range(len(value)):
                element = f"{names[i]}[{k}]"
                elements.append(Parameter(element, len(parameters), "f64"))
                parameters.append(element)
                symbols.append((element, "vector", names[i], k))
            trace_args[i] = elements
        elif kind == _QUDIT:
            # Slot k of the module traces as the qudit of index -1 - k.
            trace_args[i] = Qudit(target, -1 - len(slotted), value.logical)
            slotted.append((f"s{len(slotted)}", "slot", names[i], None))
        else:
            first = len(slotted)
            trace_args[i] = Register(
                [Qudit(target, -1 - first - k, q.logical) for k, q in enumerate(value)]
            )
            for k in range(len(value)):
                slotted.append((f"s{first + k}", "slot", names[i], k))
    # Promoting list parameters may leave only floats: a module of float
    # parameters, bound as any other.
    relocated = any(kinds[i] != _FLOAT for i in runtime)
    role = _Role((*symbols, *slotted)) if relocated else None
    return build(
        fn,
        trace_args,
        tuple(parameters),
        target,
        options,
        len(slotted),
        role,
        kernel.max_duration_cycles,
    )


def _real(value: Any) -> float:
    """*value* as a float, if it is a real number and not a time."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            return float(value.__float__())
        except AttributeError:
            raise InvalidTypeError(f"{type(value).__name__} is not a real number") from None
    if value.__class__ is Time:
        raise InvalidTypeError(f"{value} is a time, not a plain number")
    return float(value)
