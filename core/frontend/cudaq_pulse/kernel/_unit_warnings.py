# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``UnitWarning`` checks of ``unit_warnings=True``: a tracer that warns
at each time a kernel gives without units and each frequency far below the
quantum system's carriers. Loaded only by a compilation that asks for them,
so a kernel compiled without them traces as before.
"""

from __future__ import annotations

import dataclasses
import numbers
import warnings
from typing import TYPE_CHECKING, Any

from .._errors import PulseWarning
from .._units import Time
from ..compiler import CompileOptions
from ._emitter import PulseIREmitter, tracing_frames

if TYPE_CHECKING:
    from .._device import QuantumSystem


class UnitWarning(PulseWarning):
    """A number given without units that is probably not what was meant: a
    bare number as a duration, wait or other time of a pulse, which counts
    clock cycles, or a frequency so far below the quantum system's carriers
    that it is probably in MHz or kHz rather than Hz.

    Issued while a kernel compiled with ``unit_warnings=True`` traces, at
    the kernel line; ``warnings.filterwarnings("error",
    category=pulse.UnitWarning)`` makes each one an error."""


# The arguments of each op that are times, by position, and their names.
_TIMES = {
    "gaussian": ((0, "duration"), (2, "sigma")),
    "square": ((0, "duration"),),
    "drag": ((0, "duration"), (2, "sigma"), (3, "beta")),
    "cosine": ((0, "duration"),),
    "tanh_ramp": ((0, "duration"), (2, "sigma")),
    "gaussian_square": ((0, "duration"), (2, "sigma"), (3, "width")),
    "custom": ((0, "duration"),),
    "wait": ((1, "duration"),),
}
# The ops that take a frequency in Hz as their second argument, and how far
# below the lowest carrier one is taken for a frequency in other units: a shift
# by a millionth of the carrier (5 kHz at 5 GHz), a frequency of a
# thousandth (5 MHz).
_FREQUENCIES = {"shift_frequency": 1e-6, "set_frequency": 1e-3}


class _UnitWarningOptions(CompileOptions):
    """The options of a compilation with unit warnings: equal only to
    others with them, so its programs are traced apart, and with the repr of
    the same options without them, since the program is the same."""

    unit_warnings = True

    def __repr__(self) -> str:
        return repr(CompileOptions(*(getattr(self, f.name) for f in dataclasses.fields(self))))


def with_unit_warnings(options: CompileOptions) -> CompileOptions:
    """*options*, for a compilation with unit warnings."""
    if options.unit_warnings:
        return options
    return _UnitWarningOptions(*(getattr(options, f.name) for f in dataclasses.fields(options)))


def carrier_hz(quantum_system: QuantumSystem) -> float:
    """The lowest qudit frequency of *quantum_system*, which frequencies are
    held against; its clock rate when it gives none."""
    given = [q._frame_hz for q in quantum_system.physical_qudits.values() if q._frame_hz > 0]
    return float(min(given)) if given else quantum_system.clock_ghz * 1e9


def _bare(value: Any) -> bool:
    """Whether *value* is a nonzero number with no unit: not a time, a
    count of cycles from ``round_duration``, a bool or a value of the
    program."""
    cls = value.__class__
    if cls is not int and not (
        isinstance(value, numbers.Real) and not isinstance(value, (int, Time))
    ):
        return False
    return bool(value != 0)


def _clock(value: float) -> str:
    return f"{value:g} GHz" if value >= 1 else f"{value * 1e3:g} MHz"


def _hz(value: float) -> str:
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(value) >= scale:
            return f"{value / scale:g} {unit}"
    return f"{value:g} Hz"


class _UnitChecks:
    """Warns, then emits as the tracer it is mixed into."""

    __slots__ = ()

    def _emit_pulse_call(self, fname: str, call_args: list[Any]) -> tuple[Any, ...]:
        builder: Any = self.b  # type: ignore[attr-defined]
        times = _TIMES.get(fname)
        if times is not None:
            rounded = builder.__dict__.get("_unit_rounded", ())
            for index, name in times:
                if index < len(call_args):
                    value = call_args[index]
                    if _bare(value) and value not in rounded:
                        self._warn_time(fname, name, value)
        elif fname in _FREQUENCIES:
            if len(call_args) > 1 and _bare(call_args[1]):
                self._check_frequency(fname, call_args[1], _FREQUENCIES[fname])
        results: tuple[Any, ...] = super()._emit_pulse_call(fname, call_args)  # type: ignore[misc]
        if fname == "round_duration" and results[0].__class__ is int:
            # Cycles counted by round_duration are meant as cycles.
            builder.__dict__.setdefault("_unit_rounded", set()).add(results[0])
        return results

    def _warn_time(self, fname: str, name: str, value: Any) -> None:
        clock_ghz = float(getattr(self.b, "clock_ghz", 2.0))  # type: ignore[attr-defined]
        self._warn(
            f"{fname}(): {name} {value!r} has no units, so it is {value!r} clock cycles "
            f"({value / clock_ghz:g} ns at the {_clock(clock_ghz)} clock); write it with "
            f"units, as {value!r} * pulse.ns"
        )

    def _check_frequency(self, fname: str, value: Any, below: float) -> None:
        carrier = float(self.b.unit_scale_hz)  # type: ignore[attr-defined]
        if abs(value) >= below * carrier:
            return
        what = "a shift" if fname == "shift_frequency" else "a frequency"
        self._warn(
            f"{fname}(): {value!r} is {what} of {_hz(float(value))}, far below the quantum "
            f"system's carriers (the lowest is {_hz(carrier)}), so it is probably not in Hz; "
            f"frequencies are in Hz, as {value!r}e6 for {value!r} MHz"
        )

    def _warn(self, message: str) -> None:
        frames = tracing_frames()
        if not frames:
            warnings.warn(message, UnitWarning, stacklevel=2)
            return
        prog, index = frames[0]
        filename, line, *_ = prog.location(index)
        module_globals: dict[str, Any] = self._fn_globals  # type: ignore[attr-defined]
        warnings.warn_explicit(
            message,
            UnitWarning,
            filename,
            line or 0,
            module=module_globals.get("__name__"),
            registry=module_globals.setdefault("__warningregistry__", {}),
        )


_CHECKED: dict[type[PulseIREmitter], type[PulseIREmitter]] = {}


def unit_checked(builder: Any, quantum_system: QuantumSystem) -> type[PulseIREmitter]:
    """The tracer *builder*'s sample rates take, with the unit checks; its
    helpers' tracers, which are of its type, have them too. Frequencies are
    held against *quantum_system*'s carriers."""
    builder.unit_scale_hz = carrier_hz(quantum_system)
    if builder.tracer is None:
        emitter_type: type[PulseIREmitter] = PulseIREmitter
    else:
        emitter_type = builder.tracer
    checked = _CHECKED.get(emitter_type)
    if checked is None:
        checked = type(
            f"UnitChecked{emitter_type.__name__}", (_UnitChecks, emitter_type), {"__slots__": ()}
        )
        _CHECKED[emitter_type] = checked
    return checked
