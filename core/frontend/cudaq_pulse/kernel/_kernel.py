# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel objects: calling, compiling and caching pulse kernels.

Calling a kernel binds its arguments as Python binds a call, then splits
them by annotation. A ``float`` parameter is a runtime value: it stays a
function argument of the compiled module, so a new value never recompiles.
So is each element of a ``list[float]`` parameter, whose length is part of
the specialization. A parameter annotated ``Qudit`` or ``Register`` is a
runtime qudit: the module takes it as a slot, and a call binds the slot to
its handle's carrier, so calls with other handles, or on a recalibrated
quantum system, share the module. A ``*qudits`` parameter is a register of the
call's qudits, runtime when annotated ``Qudit``, of a length that is part of
the specialization. Every other parameter (unannotated qudits, ints,
anything unannotated) is static: it can shape the program, so the kernel is
specialized on its value.

A specialization is found by its static values and the quantum system, and is used
while its guard holds: the globals, closure cells and attribute chains the
kernel read when it was traced are re-read and compared on every call
(``_guard.py``). A miss traces and builds the module under the dispatcher's
lock; a hit takes no lock.
"""

from __future__ import annotations

import atexit
import copy
import inspect
import itertools
import os
import threading
import types
import typing
import warnings
import weakref
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, NamedTuple

from .._device import (
    NoQuantumSystemError,
    QuantumSystem,
    QuantumSystemMismatchError,
    Qudit,
    Register,
)
from .._errors import InvalidTypeError
from ..compiler import (
    _DEFAULT_MACHINE,
    CompileOptions,
    ScheduledKernel,
    build,
    with_line_delays,
)
from ._guard import Guard, build_guard

if typing.TYPE_CHECKING:
    from .._idle import IdleReplacement
from ._schedule_blocks import root_schedule, validate_kernel_schedule, validate_max_duration
from ._unroll import resolve_unroll_threshold
from .ir_builder import (
    CompilationError,
    IRValue,
    Parameter,
    SymbolicValueError,
    is_symbolic,
)

CACHE_SIZE_ENV = "CUDAQ_PULSE_CACHE_SIZE"
DEFAULT_CACHE_SIZE = 32

_ticks = itertools.count()


# A kernel's guard holds its module globals, which hold the kernel: release
# the compiled modules before the interpreter tears the extension down.
_dispatchers: weakref.WeakSet[_Dispatcher] = weakref.WeakSet()


@atexit.register
def _release_compiled_modules() -> None:
    for dispatcher in list(_dispatchers):
        dispatcher._entries.clear()
        dispatcher.general = None
        dispatcher.kernel._traced.clear()


def _cache_size(value: int | None) -> int:
    if value is None:
        raw = os.environ.get(CACHE_SIZE_ENV, "").strip()
        if not raw:
            return DEFAULT_CACHE_SIZE
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(f"{CACHE_SIZE_ENV} must be a positive integer, got {raw!r}")
        return int(raw)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"cache_size must be a positive integer, got {value!r}")
    return value


# What a runtime parameter is.
_FLOAT, _VECTOR, _QUDIT, _REGISTER = range(4)
_KINDS = {float: _FLOAT, Qudit: _QUDIT, Register: _REGISTER}


# The kind of runtime parameter *annotation* declares, if any: ``float``,
# ``list[float]``, ``Qudit``, or ``Register`` (also ``list[Qudit]`` and
# ``tuple[Qudit, ...]``). A string annotation is evaluated in *scope*.
def _runtime_kind(annotation: Any, scope: dict[str, Any]) -> int | None:
    if isinstance(annotation, str):
        if annotation == "float":
            return _FLOAT
        try:
            annotation = eval(annotation, scope)
        except Exception:
            return None
    try:
        kind = _KINDS.get(annotation)
    except TypeError:
        return None
    if kind is not None:
        return kind
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is list and args == (float,):
        return _VECTOR
    if (origin is list and args == (Qudit,)) or (origin is tuple and args == (Qudit, ...)):
        return _REGISTER
    return None


# A function with *fn*'s signature that returns its bound arguments,
# those of *names* in order.
#
# Python itself binds the call, so defaults, keyword-only parameters and
# every ``TypeError`` behave exactly as for *fn*. A ``*`` parameter, the
# last of *names*, is bound to a ``Register`` of its qudits.
def _make_binder(fn: types.FunctionType, names: tuple[str, ...]) -> Callable[..., tuple[Any, ...]]:
    code = fn.__code__
    npos = code.co_posonlyargcount
    params = list(names[: code.co_argcount])
    if npos:
        params.insert(npos, "/")
    namespace: dict[str, Any] = {}
    if code.co_flags & inspect.CO_VARARGS:
        star = names[-1]
        kernel = fn.__name__

        def register(qudits: tuple[Any, ...]) -> Register:
            for position, value in enumerate(qudits):
                if type(value) is not Qudit:
                    if isinstance(value, (Register, list, tuple)):
                        hint = f"each qudit as its own argument, as {kernel}(..., *register)"
                    else:
                        hint = "other values as named parameters"
                    raise InvalidTypeError(
                        f"{kernel}(): *{star} takes qudit handles, got "
                        f"{type(value).__name__} at position {position}; pass {hint}"
                    )
            return Register(qudits)

        bound = "_register"
        while bound in names:
            bound += "_"
        namespace[bound] = register
        params.append(f"*{star}")
        params.extend(names[code.co_argcount : -1])
        body = f"({', '.join(names[:-1])}{', ' if names[1:] else ''}{bound}({star}),)"
    else:
        if code.co_kwonlyargcount:
            params.append("*")
            params.extend(names[code.co_argcount :])
        body = f"({', '.join(names)},)" if names else "()"
    src = f"def _bind({', '.join(params)}):\n    return {body}\n"
    exec(compile(src, code.co_filename, "exec"), namespace)
    binder: Any = namespace["_bind"]
    binder.__code__ = binder.__code__.replace(co_name=code.co_name, co_qualname=code.co_qualname)
    binder.__name__ = fn.__name__
    binder.__qualname__ = fn.__qualname__
    binder.__defaults__ = fn.__defaults__
    binder.__kwdefaults__ = fn.__kwdefaults__
    return binder  # type: ignore[no-any-return]


def _static_key(value: Any) -> Any:
    t = type(value)
    if t is Qudit:
        # The dispatcher's key holds the quantum system's fingerprint.
        return (Qudit, value.carrier, value.index)
    if t is Register:
        return (Register, tuple((q.carrier, q.index) for q in value))
    if t in (int, bool, float, str, complex, type(None)):
        return (t, value)
    if t in (tuple, list):
        return (t, tuple(_static_key(v) for v in value))
    if t is dict:
        return (t, tuple((k, _static_key(v)) for k, v in value.items()))
    try:
        hash(value)
    except TypeError:
        return (id, id(value))
    return (t, value)


# The types of the static arguments _static_key keys by value alone.
_BY_VALUE = frozenset({Qudit, Register, int, bool, float, str, complex, type(None)})


def _keyed_by_identity(value: Any, found: list[Any]) -> list[Any]:
    """The values in the static argument *value* that ``_static_key`` keys by
    identity (unhashable, or compared by identity), added to *found*:
    changed in place, they keep their key, and the guard compares them."""
    t = type(value)
    if t in (tuple, list):
        for v in value:
            _keyed_by_identity(v, found)
    elif t is dict:
        for v in value.values():
            _keyed_by_identity(v, found)
    elif t not in _BY_VALUE:
        try:
            hash(value)
        except TypeError:
            found.append(value)
        else:
            if t.__eq__ is object.__eq__:  # type: ignore[comparison-overlap]
                found.append(value)
    return found


# ── Return values ────────────────────────────────────────────────────────


class _Measured:
    """A returned measurement: the ordinal of its readout."""

    __slots__ = ("readout",)

    def __init__(self, readout: int):
        self.readout = readout


class _Sequence:
    """A returned list or tuple (*kind*) holding measurements."""

    __slots__ = ("kind", "items")

    def __init__(self, kind: type, items: tuple[Any, ...]):
        self.kind = kind
        self.items = items


# What a call returns, from the value the trace returned: for a value
# only the running program knows, the error calling the kernel raises.
def _return_spec(value: Any, readouts: dict[int, int], name: str) -> Any:
    if isinstance(value, IRValue):
        if value.vtype == "measurement":
            ordinal = readouts.get(value.vid)
            if ordinal is None:
                raise CompilationError(
                    f"{name}() returns a measurement made inside a loop or branch; "
                    "only measurements made at the top level of the kernel can be returned",
                    code="runtime_value",
                )
            return _Measured(ordinal)
        raise CompilationError(
            f"{name}() returns a pulse {value.vtype} value; a kernel returns "
            "measurements or values known at compile time",
            code="unsupported_construct",
        )
    if is_symbolic(value):
        from .ir_builder import contains_answer

        if contains_answer(value):
            return CompilationError(
                f"{name}() returns the answer of a classical call, which only the "
                "running program has; called from Python, a kernel returns "
                "measurements or values known at compile time",
                code="runtime_value",
            )
        raise CompilationError(
            f"{name}() returns a value computed from a float parameter, which is "
            "only known when the kernel runs",
            code="runtime_value",
        )
    if type(value) in (tuple, list):
        items = tuple(_return_spec(v, readouts, name) for v in value)
        for item in items:
            if isinstance(item, CompilationError):
                return item
        if any(isinstance(i, (_Measured, _Sequence)) for i in items):
            return _Sequence(type(value), items)
    return value


def _has_measurements(spec: Any) -> bool:
    if isinstance(spec, _Measured):
        return True
    return isinstance(spec, _Sequence) and any(_has_measurements(i) for i in spec.items)


def _materialize(spec: Any, outcome: Callable[[int], int]) -> Any:
    if isinstance(spec, _Measured):
        return outcome(spec.readout)
    if isinstance(spec, _Sequence):
        return spec.kind(_materialize(i, outcome) for i in spec.items)
    if isinstance(spec, (int, float, complex, str, type(None))):
        return spec
    if isinstance(spec, CompilationError):
        raise CompilationError(*spec.args, code=spec.code)
    # Each call returns its own copy of a container built by the kernel.
    return copy.deepcopy(spec)


def _execute(lowered: ScheduledKernel, returns: Any) -> Any:
    from ..runtime.simulate import _simulate_lowered

    measured = _has_measurements(returns)
    result = _simulate_lowered(lowered, shots_count=1 if measured else None)
    if not measured:
        return _materialize(returns, int)
    assert result.records is not None
    row = result.records[0]
    column = {m.readout: i for i, m in enumerate(result.measurements) if not m.iteration}

    def outcome(readout: int) -> int:
        i = column.get(readout)
        return -1 if i is None else int(row[i])

    return _materialize(returns, outcome)


class _QuditTwiceError(Exception):
    """A qudit ``_systems`` met twice; the caller names the parameters."""

    def __init__(self, qudit: Qudit) -> None:
        self.qudit = qudit


# The quantum system of the handles in *arg*, checked against *qs*, for
# a call of the kernel *fn* (named in errors).
#
# *seen* collects the carriers passed so far: one qudit passed twice is
# an error, as it would be one degree of freedom bound to two names, and
# raises ``_QuditTwiceError``.
def _systems(arg: Any, qs: QuantumSystem | None, fn: Any, seen: set[int]) -> QuantumSystem | None:
    t = type(arg)
    if t is Qudit:
        if arg.index in seen:
            raise _QuditTwiceError(arg)
        seen.add(arg.index)
        quantum_system: QuantumSystem = arg.quantum_system
        if qs is None or qs is quantum_system:
            return quantum_system
        raise QuantumSystemMismatchError(_two_systems(fn.__name__, qs, quantum_system))
    if t is Register or t is tuple or t is list:
        for item in arg:
            qs = _systems(item, qs, fn, seen)
    return qs


# Why *name*() cannot take qudits of both *found* and *quantum_system*.
def _two_systems(name: str, found: QuantumSystem, quantum_system: QuantumSystem) -> str:
    if found == quantum_system:
        return (
            f"{name}() got qudits of two separate quantum systems, both named "
            f"{quantum_system.name!r}, with "
            "the same definition, as two pulse.quantum_system() calls make; pass handles "
            "of one quantum system"
        )
    from .._errors import _ranges

    both = (
        f"both named {quantum_system.name!r}"
        if found.name == quantum_system.name
        else f"named {found.name!r} and {quantum_system.name!r}"
    )
    differences = []
    if found.layout != quantum_system.layout:
        differences.append(
            f"one with layout {_ranges(found.layout)}, the other with layout "
            f"{_ranges(quantum_system.layout)}"
        )
    elif found.fingerprint != quantum_system.fingerprint:
        differences.append("the same layout, but different qudits, couplings or calibration")
    one, other = found.control_system, quantum_system.control_system
    if one != other:

        def described(system: Any) -> str:
            if system is None:
                return "none"
            count = len(system.controllers)
            return f"{count} controller{'' if count == 1 else 's'}"

        def first_difference(a: Any, b: Any, path: str) -> str | None:
            if isinstance(a, dict) and isinstance(b, dict):
                for key in [*a, *(k for k in b if k not in a)]:
                    if a.get(key) != b.get(key):
                        return first_difference(a.get(key), b.get(key), f"{path}.{key}")
                return None
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    return f"{path} has {len(a)} in one and {len(b)} in the other"
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    if x != y:
                        return first_difference(x, y, f"{path}[{i}]")
                return None
            return None if a == b else f"{path} is {a!r} in one and {b!r} in the other"

        where: str | None
        if one is None or other is None:
            where = f"one has {described(one)}, the other {described(other)}"
        else:
            where = first_difference(one.as_dict(), other.as_dict(), "control_system")
        differences.append("their control systems differ" + (f" ({where})" if where else ""))
    return (
        f"{name}() got qudits of two different quantum systems, {both}: "
        f"{'; '.join(differences)}; pass handles of one quantum system"
    )


# How many handles in *value* are of the qudit at *index*.
def _count(value: Any, index: int) -> int:
    t = type(value)
    if t is Qudit:
        return int(value.index == index)
    if t is Register or t is tuple or t is list:
        return sum(_count(item, index) for item in value)
    return 0


# The error for *qudit* passed twice among *arguments*, the (parameter,
# value) pairs in the order ``_systems`` met them.
def _passed_twice(name: str, qudit: Qudit, arguments: list[tuple[str, Any]]) -> InvalidTypeError:
    holders = [
        parameter for parameter, value in arguments for _ in range(_count(value, qudit.index))
    ]
    first, second = holders[0], holders[1]
    if first == second:
        return InvalidTypeError(
            f"{name}() got qudit {qudit.logical} twice in {first!r}; each qudit of an "
            "argument needs to be distinct"
        )
    return InvalidTypeError(
        f"{name}() got qudit {qudit.logical} for both {first!r} and {second!r}; each "
        "argument needs a distinct qudit"
    )


# ── Specializations ──────────────────────────────────────────────────────


class CacheInfo(NamedTuple):
    size: int
    compiled: int
    evicted: int
    capacity: int | None


# Traced programs a kernel keeps for scheduling again with other line delays.
_TRACED_KEPT = 8


class _Entry:
    """One compiled specialization and the guard that keeps it valid."""

    __slots__ = (
        "lowered",
        "guard",
        "returns",
        "tick",
        "pins",
        "runtime",
        "qudit_count",
    )
    # Set only on the entries kept to be scheduled again.
    qudit_count: int

    def __init__(
        self,
        lowered: ScheduledKernel,
        guard: Guard,
        returns: Any,
        pins: tuple[Any, ...],
        runtime: tuple[int, ...],
    ):
        self.lowered = lowered
        self.guard = guard
        self.returns = returns
        self.tick = next(_ticks)
        # The parameters that are function arguments of the module.
        self.runtime = runtime
        # Static arguments keyed by identity are kept alive with the entry.
        self.pins = pins


class _Dispatcher:
    """Binds a call, finds or builds its specialization, and runs it."""

    def __init__(self, kernel: Kernel[...], options: CompileOptions, capacity: int | None) -> None:
        self.kernel = kernel
        self.options = options
        self._capacity = capacity
        self._entries: dict[Any, _Entry] = {}
        self._lock = threading.Lock()
        self.misses = 0
        self.evictions = 0
        self.pin: QuantumSystem | None = None
        # The specialization a pinned all-float kernel is built with.
        self.general: ScheduledKernel | None = None
        _dispatchers.add(self)

    def _device(self, values: tuple[Any, ...], kernel: Kernel[...]) -> QuantumSystem:
        found = None
        seen: set[int] = set()
        try:
            for i in kernel._split[0]:
                found = _systems(values[i], found, kernel, seen)
        except _QuditTwiceError as twice:
            names = kernel._names
            arguments = [(names[i], values[i]) for i in kernel._split[0]]
            raise _passed_twice(kernel.__name__, twice.qudit, arguments) from None
        pin = self.pin
        if pin is not None:
            if found is not None and found is not pin:
                raise _pin_mismatch(kernel.__name__, pin, found)
            return pin
        if found is not None:
            return found
        raise _no_device(kernel.__name__)

    def lookup(self, values: tuple[Any, ...]) -> _Entry:
        kernel = self.kernel
        target = self._device(values, kernel)
        static = kernel._split[0]
        # The identity, not the fingerprint: compiled for no control system,
        # a kernel is scheduled and checked for the one of its qudits.
        key = (target._identity, static, *[_static_key(values[i]) for i in static])
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
            static = kernel._split[0]
            key = (target._identity, static, *[_static_key(values[i]) for i in static])
            if target._drift:
                # Kept for the twin: no cached program plays another system's drift.
                from ..quantum_systems.drift import _kept

                kept, entry = _kept(entry, target, kernel)
                self._insert(key, kept)
                return entry
            self._insert(key, entry)
        return entry

    def _insert(self, key: Any, entry: _Entry) -> None:
        entries = self._entries
        entries[key] = entry
        if self._capacity is not None and len(entries) > self._capacity:
            oldest = min(entries, key=lambda k: entries[k].tick)
            del entries[oldest]
            self.evictions += 1

    def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = self.kernel
        values = kernel._bind(*args, **kwargs)
        entry = self.lookup(values)
        lowered = entry.lowered
        if entry.runtime:
            lowered = lowered._with_values(kernel._runtime_values(values, entry.runtime))
        return _execute(lowered, entry.returns)

    def lower(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ScheduledKernel:
        kernel = self.kernel
        values = kernel._bind(*args, **kwargs)
        entry = self.lookup(values)
        if entry.runtime:
            return entry.lowered._with_values(kernel._runtime_values(values, entry.runtime))
        return entry.lowered

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _pin_mismatch(name: str, pin: QuantumSystem, got: QuantumSystem) -> QuantumSystemMismatchError:
    if got == pin:
        return QuantumSystemMismatchError(
            f"{name}() is compiled for a quantum system of {pin.name!r} and got qudits of "
            "another, separate quantum system of it with the same definition; pass "
            "handles of the quantum system it is compiled for"
        )
    return QuantumSystemMismatchError(
        f"{name}() is compiled for quantum system {pin.name!r} and got qudits of {got.name!r}"
    )


def _no_device(name: str) -> NoQuantumSystemError:
    return NoQuantumSystemError(
        f"{name}() was called without qudit handles, so it has no quantum system: pass "
        "handles of a quantum system (qs.qudits[i]), or compile it for one with "
        "cudaq_pulse.compile(k, quantum_system=qs)"
    )


class Kernel[**P]:
    """A pulse kernel. ``k(*args)`` compiles on first use and runs it.

    ``k.lower(*args)`` returns the compiled module without running it.
    ``cudaq_pulse.compile(k)`` compiles ahead of the call. ``P`` is the
    decorated function's signature, so calls are checked against it.

    A kernel pickles as a function does. One that its module and qualified
    name find, outside ``__main__``, pickles as that name. Any other (in
    ``__main__``, at the prompt, inside a function) pickles as its function
    and options, the function by reference with pickle and by value with
    cloudpickle (Ray, Dask): pickle cannot send one defined inside a
    function. A process keeps the kernels it unpickles this way, so sending
    one again does not compile it again.
    """

    __name__: str
    __qualname__: str

    def __init__(
        self,
        fn: Callable[P, Any],
        *,
        cache_size: int | None = None,
        schedule: str | None = None,
        unroll_threshold: int | None = None,
        max_duration_cycles: int | None = None,
    ) -> None:
        if not isinstance(fn, types.FunctionType):
            raise TypeError(f"@kernel expects a function, got {type(fn).__name__}")
        code = fn.__code__
        self.__wrapped__ = fn
        self.__pulse_kernel_fn__ = fn
        # Called from another kernel, the kernel is scheduled on its own with
        # this policy; None schedules it with its caller.
        self.schedule = validate_kernel_schedule(schedule)
        # The longest any execution of the kernel may take, checked when the
        # program is scheduled.
        self.max_duration_cycles = validate_max_duration(max_duration_cycles)
        # None leaves the threshold to cudaq_pulse.compile(), else the default.
        self.unroll_threshold = (
            None if unroll_threshold is None else resolve_unroll_threshold(unroll_threshold)
        )
        for attr in ("__module__", "__name__", "__qualname__", "__doc__"):
            setattr(self, attr, getattr(fn, attr))
        signature = inspect.signature(fn)
        self.__signature__ = signature
        if code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS):
            # A * parameter is a register of the call's qudits. The kernel is
            # traced as a function of the same code that takes it as its last
            # keyword-only parameter, where the code keeps it.
            name = fn.__name__
            parameters = signature.parameters.values()
            if code.co_flags & inspect.CO_VARKEYWORDS:
                keywords = next(p for p in parameters if p.kind is p.VAR_KEYWORD)
                raise InvalidTypeError(f"pulse kernel {name}() cannot take **{keywords.name}")
            star = next(p for p in parameters if p.kind is p.VAR_POSITIONAL)
            annotation = star.annotation
            if annotation is not star.empty:
                if _runtime_kind(annotation, fn.__globals__) != _QUDIT:
                    raise InvalidTypeError(
                        f"pulse kernel {name}(): *{star.name} takes qudits: annotate it Qudit "
                        "to compile once for every placement, or leave it unannotated to "
                        "specialize on the qudits, and pass other values as named parameters"
                    )
                annotation = tuple[Qudit, ...]
            code = code.replace(
                co_flags=code.co_flags & ~inspect.CO_VARARGS,
                co_kwonlyargcount=code.co_kwonlyargcount + 1,
            )
            traced = types.FunctionType(code, fn.__globals__, name, fn.__defaults__, fn.__closure__)
            traced.__qualname__ = fn.__qualname__
            traced.__kwdefaults__ = fn.__kwdefaults__
            # Called from another kernel, it binds as the function does.
            traced.__signature__ = signature  # type: ignore[attr-defined]
            self.__pulse_kernel_fn__ = traced
            classified = [p.replace(annotation=annotation) if p is star else p for p in parameters]
            signature = signature.replace(parameters=classified)
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        self._names = names
        kinds = {}
        for i, n in enumerate(names):
            kind = _runtime_kind(signature.parameters[n].annotation, fn.__globals__)
            if kind is not None:
                kinds[i] = kind
        self._classify(kinds, tuple(i for i in range(len(names)) if i not in kinds))
        self._promote_lock = threading.Lock()
        # The latest scheduled specializations with line delays of the
        # programs it traced, the last _TRACED_KEPT of them, which other line
        # delays schedule again without tracing.
        self._traced: dict[Any, _Entry] = {}
        self._bind = _make_binder(fn, names)
        self._dispatch = self._dispatcher_type(
            self,
            CompileOptions(
                schedule=root_schedule(self.schedule, None),
                unroll_threshold=resolve_unroll_threshold(self.unroll_threshold),
            ),
            _cache_size(cache_size),
        )

    @property
    def runtime_parameters(self) -> tuple[str, ...]:
        """Parameters annotated ``float``, ``list[float]``, ``Qudit`` or
        ``Register``, and ``*qudits: Qudit``: function arguments of the
        module. An unannotated qudit or register parameter is static. In the
        order of the signature."""
        runtime = {self._names[i] for i in self._split[1]}
        return tuple(n for n in self.__signature__.parameters if n in runtime)

    @property
    def static_parameters(self) -> tuple[str, ...]:
        """Parameters the kernel is specialized on.

        Includes float parameters whose value the program needs while it
        compiles: one that sets a waveform duration, goes through ``int()``
        or decides Python control flow. A wait length is not one of them.
        In the order of the signature.
        """
        static = {self._names[i] for i in self._split[0]}
        return tuple(n for n in self.__signature__.parameters if n in static)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        return self._dispatch.call(args, kwargs)

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> ScheduledKernel:
        """Compile for these arguments without running; see ``ScheduledKernel``."""
        return self._dispatch.lower(args, kwargs)

    def cache_info(self) -> CacheInfo:
        """Specializations held, compiled and evicted by ``k(...)`` calls."""
        d = self._dispatch
        return CacheInfo(len(d._entries), d.misses, d.evictions, d._capacity)

    def cache_clear(self) -> None:
        """Clear the specializations and traced programs cached by this kernel."""
        self._dispatch.clear()
        self._traced.clear()

    def __reduce__(self) -> str | tuple[Any, ...]:
        from .._pickling import kernel_reduce

        return kernel_reduce(self)

    @property
    def __pulse_pickled_fn__(self) -> types.FunctionType:
        """The function a kernel that its name does not find pickles with:
        named after this attribute, so that pickle finds it by reference and
        cloudpickle sends it by value, as each does a function."""
        from .._pickling import function_copy

        return function_copy(self)

    def __repr__(self) -> str:
        return f"<pulse kernel {self.__qualname__}{self.__signature__}>"

    def _runtime_values(self, values: tuple[Any, ...], runtime: tuple[int, ...]) -> list[float]:
        out = []
        for i in runtime:
            v = values[i]
            if v.__class__ is not float or v != v:
                from .._units import Time

                if v.__class__ is Time:
                    name = self._names[i]
                    raise InvalidTypeError(
                        f"{self.__name__}(): argument {name!r} is a time ({v}), and a float "
                        f"parameter is a plain number; pass {name} in nanoseconds and write "
                        f"{name} * pulse.ns in the kernel, or drop its float annotation to "
                        "compile the kernel for it"
                    )
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    try:
                        v = v.__float__()
                    except AttributeError:
                        raise TypeError(
                            f"{self.__name__}(): argument {self._names[i]!r} must be a "
                            f"real number, not {type(v).__name__}"
                        ) from None
                v = float(v)
                if v != v:
                    from .._errors import InvalidValueError

                    raise InvalidValueError(
                        f"{self.__name__}(): argument {self._names[i]!r} is nan; a float "
                        "argument must be a number"
                    )
            out.append(v)
        return out

    def _specialize(self, values: tuple[Any, ...], target: Any, options: CompileOptions) -> _Entry:
        if target.control_system is not None and options.machine is _DEFAULT_MACHINE:
            # Compiled for no control system: scheduled and checked for the
            # one of the qudits it is lowered on.
            options = replace(options, machine=target.control_system._machine)
        if options.machine._line_delays or self._traced:
            found = self._retimed(values, target, options)
            if found is not None:
                return found
        fn = self.__pulse_kernel_fn__
        names = self._names
        while True:
            static, runtime = self._split
            trace_args = list(values)
            try:
                if self._relocatable:
                    from ._placing import _build_placed

                    lowered, builder = _build_placed(self, fn, trace_args, runtime, target, options)
                else:
                    for i in self._declared_float:
                        if i not in runtime and values[i].__class__ is float:
                            from ._durations import Cycles

                            trace_args[i] = Cycles(values[i])
                    for n, i in enumerate(runtime):
                        trace_args[i] = Parameter(names[i], n, "f64")
                    lowered, builder = build(
                        fn,
                        trace_args,
                        tuple(names[i] for i in runtime),
                        target,
                        options,
                        max_duration_cycles=self.max_duration_cycles,
                    )
                break
            except SymbolicValueError as e:
                kinds = self._kinds
                promote = {
                    i
                    for i in runtime
                    if kinds[i] in (_FLOAT, _VECTOR)
                    and any(n == names[i] or n.startswith(f"{names[i]}[") for n in e.names)
                }
                if not promote:
                    raise
                operation = e.operation or "an operation that needs its value"
                where = _traced_line(e)
            promoted = self._promote(promote)
            if promoted:
                _warn_static(self.__name__, [names[i] for i in promoted], operation, where)
        returns = _return_spec(builder.returned, builder.readouts, self.__name__)
        pins = tuple(values[i] for i in static)
        guard = build_guard(
            fn,
            impure=builder.impure,
            dict_items=builder.dict_items,
            dict_whole=builder.dict_whole,
            pinned=() if _BY_VALUE.issuperset(map(type, pins)) else _keyed_by_identity(pins, []),
        )
        entry = _Entry(lowered, guard, returns, pins, runtime)
        if options.machine._line_delays and lowered.scheduled:
            # Only a kept entry is scheduled again, for its qudits.
            entry.qudit_count = builder.qudit_count
            self._keep_traced(self._traced_key(values, target, options), entry)
        return entry

    def _keep_traced(self, key: Any, entry: _Entry) -> None:
        traced = self._traced
        traced.pop(key, None)
        traced[key] = entry
        if len(traced) > _TRACED_KEPT:
            del traced[next(iter(traced))]

    # What a specialization is traced for: everything but line delays,
    # which only move its schedule.
    def _traced_key(self, values: tuple[Any, ...], target: Any, options: CompileOptions) -> Any:
        machine = options.machine
        system = machine._system
        if machine._line_delays and system is not None:
            options = replace(options, machine=system._undelayed()._machine)
        static = self._split[0]
        return (target.fingerprint, options, static, *[_static_key(values[i]) for i in static])

    # The specialization for *values* scheduled again with the line
    # delays of *options*, from one traced with other delays, if any.
    def _retimed(
        self, values: tuple[Any, ...], target: Any, options: CompileOptions
    ) -> _Entry | None:
        prior = self._traced.get(self._traced_key(values, target, options))
        if (
            prior is None
            or prior.lowered.options.machine._line_delays == options.machine._line_delays
            or not prior.guard.check()
        ):
            return None
        lowered = with_line_delays(prior.lowered, options)
        if lowered is None:
            return None
        entry = _Entry(lowered, prior.guard, prior.returns, prior.pins, prior.runtime)
        entry.qudit_count = prior.qudit_count
        self._keep_traced(self._traced_key(values, target, options), entry)
        return entry

    def _classify(self, kinds: dict[int, int], static: tuple[int, ...]) -> None:
        # The kind of each runtime parameter, by index.
        self._kinds = kinds
        runtime = tuple(i for i in sorted(kinds) if i not in static)
        # Runtime qudit and register parameters, and those whose length
        # (lists, registers) is part of the specialization.
        self._slotted = tuple(i for i in runtime if kinds[i] in (_QUDIT, _REGISTER))
        self._shaped = tuple(i for i in runtime if kinds[i] in (_VECTOR, _REGISTER))
        self._relocatable = bool(self._slotted or self._shaped)
        dispatcher: type[_Dispatcher] = _Dispatcher
        if self._relocatable:
            from ._placing import _Placing

            dispatcher = _Placing
        self._dispatcher_type = dispatcher
        # (static, runtime) parameter indices, replaced as one value when a
        # float parameter turns out to shape the program.
        self._split = (static, runtime)
        # The float and list[float] parameters, static or not: a static one's
        # values trace as cycles, so that a duration may take them.
        self._declared_float = tuple(i for i in sorted(kinds) if kinds[i] in (_FLOAT, _VECTOR))

    # Make float parameters static: the program needs their values.
    # Returns those of *indices* that were not static yet, so that threads
    # tracing at once report each parameter once.
    def _promote(self, indices: set[int]) -> list[int]:
        with self._promote_lock:
            static, runtime = self._split
            promoted = [i for i in runtime if i in indices]
            self._split = (
                tuple(sorted({*static, *indices})),
                tuple(i for i in runtime if i not in indices),
            )
        return promoted


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# `` at line N`` for the kernel line where tracing raised *e*, with the
# helper's line when the value was needed inside a helper.
def _traced_line(e: BaseException) -> str:
    source = getattr(e, "pulse_source", None)
    if source is None:
        return ""
    frames = [source, *getattr(e, "pulse_callers", ())]
    prog, index = frames[-1]
    where = f" at line {prog.location(index)[1]}"
    if len(frames) > 1:
        prog, index = frames[0]
        where += f" (in {prog.code.co_name}(), line {prog.location(index)[1]})"
    return where


# Warn that tracing made the float parameters *params* static. Each is
# made static once per kernel, so this warns once per parameter.
def _warn_static(kernel: str, params: list[str], operation: str, where: str = "") -> None:
    from .._errors import _SpecializationWarning

    quoted = ", ".join(repr(p) for p in params)
    one = len(params) == 1
    it = "it" if one else "them"
    warnings.warn(
        f"pulse kernel {kernel}(): float argument{'' if one else 's'} {quoted} "
        f"{'is' if one else 'are'} used in {operation}{where}, which needs "
        f"{'its value' if one else 'their values'} while tracing, so the kernel is "
        f"compiled again for every new value of {quoted}. To keep "
        f"{'it a runtime parameter' if one else 'them runtime parameters'}, compute "
        f"that value on the host and pass it in; otherwise annotate {it} as int, or "
        f"give {it} a fixed value, to make the specialization explicit.",
        _SpecializationWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )


class CompiledKernel[**P]:
    """A kernel compiled ahead of its call by ``cudaq_pulse.compile``.

    ``ck(*args)`` runs the kernel on the quantum system of the qudit handles passed
    and returns what the kernel returns. It holds one specialization per
    quantum system and static argument values, built on first use and never
    evicted. A kernel pinned to a quantum system and without static parameters is
    built when it is compiled.

    It pickles as its kernel (by reference), options and quantum system, without
    its specializations, which compile again on first use where it is
    unpickled; pickle ``ck.lower(*args)`` to send a program that runs
    without compiling.
    """

    def __init__(self, kernel: Kernel[P], opts: CompileOptions, qs: QuantumSystem | None) -> None:
        if qs is not None and not isinstance(qs, QuantumSystem):
            raise TypeError(
                f"quantum_system must be a cudaq_pulse QuantumSystem, got {type(qs).__name__}"
            )
        self.kernel = kernel
        self.quantum_system = qs
        self._dispatch = kernel._dispatcher_type(kernel, opts, None)
        self._dispatch.pin = qs
        if qs is not None and not kernel._split[0] and not kernel._relocatable:
            # Every parameter is a float: the values are not needed to build
            # it, unless one of them shapes the program.
            try:
                self._dispatch.general = self._dispatch.lookup((None,) * len(kernel._names)).lowered
            except Exception:
                if not kernel._split[0]:
                    raise

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        return self._dispatch.call(args, kwargs)

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> ScheduledKernel:
        """The module for these arguments, specialized and scheduled."""
        return self._dispatch.lower(args, kwargs)

    def _view(self) -> ScheduledKernel:
        general = self._dispatch.general
        if general is None:
            if self.kernel._split[0]:
                why = f"is specialized on {', '.join(self.kernel.static_parameters)}"
            else:
                why = (
                    "has no quantum system until it is called; pin one with compile(k, "
                    "quantum_system=qs)"
                )
            raise TypeError(
                f"{self.kernel.__name__}() {why}; inspect one specialization with .lower(...)"
            )
        return general

    def mlir(self, stage: str = "pulse") -> str:
        """MLIR text of the compiled module; see ``ScheduledKernel.mlir``."""
        return self._view().mlir(stage)

    @property
    def schedule(self) -> Sequence[Any]:
        """The scheduled events; see ``ScheduledKernel.schedule``."""
        return self._view().schedule

    def timeline(self, *, figsize: tuple[float, float] = (14, 4), title: str | None = None) -> Any:
        """The schedule's plot; see ``ScheduledKernel.timeline``."""
        return self._view().timeline(figsize=figsize, title=title)

    @property
    def metrics(self) -> Any:
        """The compile metrics, a ``CompileMetrics``; see ``ScheduledKernel.metrics``."""
        return self._view().metrics

    @property
    def parameters(self) -> tuple[str, ...]:
        """The float parameters, function arguments of the module."""
        kernel = self.kernel
        kinds = kernel._kinds
        return tuple(kernel._names[i] for i in kernel._split[1] if kinds[i] in (_FLOAT, _VECTOR))

    def cache_info(self) -> CacheInfo:
        """Return cache statistics for calls to ``ck(...)``.

        Compiled-kernel specializations are never evicted, so ``capacity`` is ``None``.
        """
        d = self._dispatch
        return CacheInfo(len(d._entries), d.misses, d.evictions, None)

    def replace_idle(self, rule: IdleReplacement) -> CompiledKernel[P]:
        """This kernel, compiled as it is, with each idle window of its
        programs replaced as *rule* (a ``pulse.IdleReplacement``) says: each
        stretch of at least the rule's threshold in which a qudit idles gets
        the rule's fill written in, and every operation of the kernel keeps
        its start, which compiling checks. ``lower(...)`` of the result gives
        programs whose ``idle_windows`` report each window and what was done
        there; this one is unchanged."""
        from .._idle import IdledKernel

        return IdledKernel(self, rule)

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import rebuilt_compiled

        d = self._dispatch
        return (rebuilt_compiled, (self.kernel, d.options, self.quantum_system, d.general))

    def __repr__(self) -> str:
        return f"<compiled pulse kernel {self.kernel.__qualname__}{self.kernel.__signature__}>"
