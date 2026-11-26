# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The control system of a QPU: its controllers, their wiring and links.

A ``Controller`` is one box: what it can play, the lines it drives and
reads out (``drive_lines``, ``readout_lines``: carrier indices of the
quantum system), its latency to its peers, what it supports
(``Capability``), what it computes (``Processor``) and the backend that
lowers its programs. A ``ControlSystem`` is the controllers of one QPU, the links to
the classical services beside them (by name) and the limits that apply to
all of them::

    controllers = [pulse.Controller(drive_lines=range(8), readout_lines=range(8))]
    qpu = pulse.quantum_system(
        "transmon_surface17", control_system=pulse.ControlSystem(controllers=controllers)
    )
    compiled = pulse.compile(kernel, quantum_system=qpu)

``QuantumSystem.as_dict()`` writes a QPU as one JSON-serializable mapping (format
``SYSTEM_FORMAT``), which ``QuantumSystem.from_dict`` reads back (``QuantumSystem.load`` from a
file).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import os
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import partial
from itertools import repeat
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from ._errors import (
    InvalidTypeError,
    InvalidValueError,
    NotFoundError,
    Problem,
    QuantumSystemDefinitionError,
    UpgradeNotice,
    _closest,
    _count,
    _ranges,
)
from ._units import Time, as_nanoseconds
from .classical import Link

if TYPE_CHECKING:
    from dataclasses import dataclass as _dataclass_when_used
    from fractions import Fraction
    from pathlib import Path
    from typing import Self

    from . import _clocks
    from ._capability import Capability
    from ._controller_cost import FunctionUsage
    from ._device import QuantumSystem
    from ._op_classes import OpClass
    from ._op_tables import OpCycles
    from .compiler import ScheduledKernel, _MachineModel
    from .controller import WaveformMemoryScope
    from .quantum_systems.base import _Description
else:
    from ._errors import _dataclass_when_used

__all__ = [
    "SYSTEM_FORMAT",
    "ControlSystem",
    "Controller",
    "DriveLine",
    "Processor",
    "ReadoutLine",
    "UpdateBus",
]

SYSTEM_FORMAT = "cudaq-pulse.system/1"


_PLAYED = frozenset({"drive", "readout"})
# A dedicated line's max_concurrent_tones (an int 1 is this object).
_ONE_TONE = 1
_NEEDS = MappingProxyType({"drive": "play", "readout": "readout", "wait": "wait"})


class _Line(tuple[Any, ...]):
    """A line a controller plays: the qudits it drives or reads out (their
    carrier indices in the quantum system), its calibrated delay in ns,
    from cables and electronics, which the scheduler plays the line early
    by, and the tones it plays at once. A line reaching one qudit is
    dedicated; one reaching several is shared, and a kernel's
    ``get_drive_line(q)`` or ``get_readout_line(q)`` of any of them takes
    one of its tones, for that qudit; a drive line made with
    ``broadcast=True`` has one tone instead, which reaches all its qudits
    at once. A line is read-only and compares by content; it is ``(qudit,
    delay_ns)``, on a shared line ``(qudits, max_concurrent_tones,
    delay_ns)``, and on a broadcast line ``(qudits, 1, delay_ns,
    frequency_hz)``; item 1 of those is never 0, so that a line whose item
    1 is 0 is an undelayed dedicated one."""

    __slots__ = ()
    _role: ClassVar[str]

    if TYPE_CHECKING:

        @property
        def qudit(self) -> int:
            """The qudit it drives or reads out, by carrier index; an error
            on a shared line (read ``qudits``)."""
            ...

        @property
        def qudits(self) -> tuple[int, ...]:
            """The qudits it drives or reads out, by carrier index."""
            ...

        @property
        def delay_ns(self) -> float:
            """Its calibrated delay, in ns."""
            ...

        @property
        def max_concurrent_tones(self) -> int:
            """The most tones it plays at once (1 on a dedicated line)."""
            ...

        @property
        def broadcast(self) -> bool:
            """Whether its one tone reaches all its qudits at once."""
            ...

        @property
        def frequency_hz(self) -> float | None:
            """The frequency a broadcast line states for its tone; ``None``
            on any other line, and on a broadcast line that leaves it to
            the frequency its qudits share."""
            ...

    else:

        @property
        def qudit(self):
            """The qudit it drives or reads out, by carrier index; an error
            on a shared line (read ``qudits``)."""
            qudit = self[0]
            if qudit.__class__ is int:
                return qudit
            raise InvalidValueError(
                f"{self!r} reaches qudits {_ranges(qudit)}, so it has no one qudit; read its "
                ".qudits"
            )

        @property
        def qudits(self):
            """The qudits it drives or reads out, by carrier index."""
            qudit = self[0]
            return (qudit,) if qudit.__class__ is int else qudit

        @property
        def delay_ns(self):
            """Its calibrated delay, in ns."""
            return self[1] if len(self) == 2 else self[2]

        @property
        def max_concurrent_tones(self):
            """The most tones it plays at once (1 on a dedicated line)."""
            return self[1] if len(self) > 2 else 1

        @property
        def broadcast(self):
            """Whether its one tone reaches all its qudits at once."""
            return len(self) > 3

        @property
        def frequency_hz(self):
            """The frequency a broadcast line states for its tone; ``None``
            on any other line, and on a broadcast line that leaves it to
            the frequency its qudits share."""
            return self[3] if len(self) > 3 else None

    def __new__(
        cls,
        qudits: int | Iterable[int],
        delay_ns: float | Time = 0.0,
        *,
        max_concurrent_tones: int = 1,
        broadcast: bool = False,
        frequency_hz: float | None = None,
    ) -> Self:
        name = cls.__name__
        if qudits.__class__ is not int:
            if (
                isinstance(qudits, bool)
                or not isinstance(qudits, (int, Iterable))
                or isinstance(qudits, (str, bytes))
            ):
                raise InvalidTypeError(
                    f"{name} reaches a qudit or a list of qudits, by carrier index, got {qudits!r}"
                )
            if not isinstance(qudits, int):
                from ._shared_lines import line

                shared: Self = line(
                    cls, qudits, delay_ns, max_concurrent_tones, broadcast, frequency_hz
                )
                return shared
            qudits = int(qudits)
        if qudits < 0:
            raise InvalidValueError(f"{name} has a negative carrier index {qudits}")
        if max_concurrent_tones is not _ONE_TONE:
            from ._shared_lines import tone_count

            tone_count(name, qudits, max_concurrent_tones)
        if delay_ns.__class__ is not float or not 0.0 <= delay_ns < math.inf:
            delay_ns = _delay_ns(delay_ns, f"{name}({qudits}).delay_ns")
        return tuple.__new__(cls, (qudits, delay_ns))

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return tuple.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return tuple.__ne__(self, other)

    __hash__ = tuple.__hash__

    def __repr__(self) -> str:
        try:
            qudit, delay = self
        except ValueError:
            from ._shared_lines import text

            return text(self)
        return f"{type(self).__name__}({qudit}{f', delay_ns={delay!r}' if delay else ''})"

    def __reduce__(self) -> tuple[Any, ...]:
        if len(self) > 2:
            from ._shared_lines import line

            qudits, tones, delay = self
            return (line, (type(self), qudits, delay, tones))
        return (type(self), tuple(self))

    def _undelayed(self) -> Self:
        """A shared line without its delay."""
        return type(self)(self[0], max_concurrent_tones=self[1])

    def _data(self) -> int | dict[str, Any]:
        """The line as a system file holds it: its qudit, or with a delay
        ``{"qudit": 5, "delay_ns": 4.0}``; a shared line as ``{"qudits": [0,
        1, 2, 3], "max_concurrent_tones": 4, "delay_ns": 12.0}``, without
        the defaults; a broadcast line as ``{"qudits": [0, 1, 2],
        "broadcast": true, "frequency_hz": 5.1e9}``."""
        try:
            qudit, delay = self
        except ValueError:
            from ._shared_lines import data

            return data(self)
        return {"qudit": qudit, "delay_ns": delay} if delay else qudit


class DriveLine(_Line):
    """A drive line of a controller (``Controller.drive_lines``); see
    ``ReadoutLine``. A bare carrier index in ``drive_lines`` is an undelayed
    ``DriveLine``. A line whose one tone reaches several qudits at once is a
    broadcast line: ``DriveLine([0, 1, 2], broadcast=True)``. A kernel gets
    it with ``get_drive_line(q)`` of any of its qudits or with
    ``get_drive_line(a, b, c)`` of all of them, and a drive on it drives
    every one. Its tone starts at ``frequency_hz``
    (``DriveLine([0, 1, 2], broadcast=True, frequency_hz=5.1e9)``), which
    may be left out when its qudits share one frequency; each qudit is
    driven off resonance by the difference from its own. Where the qudits
    give an ``Offset`` from one reference, the tone is the ``Offset`` they
    share and ``frequency_hz`` is not taken; where they give no frequency,
    the tone is at the carrier of each qudit the line reaches, a change of
    its frequency in a kernel is that offset from each, and ``frequency_hz``
    is not taken. Both keywords are
    for a list of qudits: with one qudit given as an int they are ignored,
    and ``DriveLine([0], broadcast=True)`` is an error."""

    __slots__ = ()
    _role = "drive"


class ReadoutLine(_Line):
    """A readout line of a controller (``Controller.readout_lines``):
    ``ReadoutLine(5, delay_ns=12.0)`` reads out qudit 5, its pulses played
    12 ns early. A bare carrier index in ``readout_lines`` is an undelayed
    ``ReadoutLine``. A feedline reading out several qudits is shared:
    ``ReadoutLine([0, 1, 2, 3], max_concurrent_tones=4)`` plays up to four
    readout tones at once (1 by default, at most the qudits it reaches)."""

    __slots__ = ()
    _role = "readout"


@_dataclass_when_used(frozen=True)
class UpdateBus:
    """A bus that carries the updates of a group of a controller's lines
    (``Controller.update_buses``) one at a time, each holding it for the
    ``cycles_per_update`` of the controller's cycles before the update takes
    effect: ``UpdateBus(drive_lines=[0, 1], cycles_per_update=8)``.
    ``drive_lines`` and ``readout_lines`` name its lines by carrier index,
    as the controller's own do. An update is a frame update, or a drive or
    readout whose tone's frequency, or whose phase other than as the tone
    runs on, differs from the drive or readout before it on its line; see
    "Update buses" in the controller backends guide."""

    drive_lines: Sequence[int] = ()
    readout_lines: Sequence[int] = ()
    cycles_per_update: int = 1

    def __post_init__(self) -> None:
        for name in ("drive_lines", "readout_lines"):
            given = getattr(self, name)
            if isinstance(given, (str, bytes)) or not isinstance(given, Iterable):
                raise InvalidTypeError(
                    f"UpdateBus.{name} is a sequence of carrier indices, got {given!r}"
                )
            qudits = tuple(given)
            if any(isinstance(q, bool) or not isinstance(q, int) or q < 0 for q in qudits):
                raise InvalidValueError(
                    f"UpdateBus.{name} holds carrier indices (ints >= 0), got {list(qudits)}"
                )
            if len(set(qudits)) != len(qudits):
                raise InvalidValueError(f"UpdateBus.{name} lists a carrier twice: {list(qudits)}")
            object.__setattr__(self, name, qudits)
        if not self.drive_lines and not self.readout_lines:
            raise InvalidValueError("UpdateBus lists no lines")
        cycles = self.cycles_per_update
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
            raise InvalidValueError(
                f"UpdateBus.cycles_per_update must be an int of at least 1, got {cycles!r}"
            )

    def __hash__(self) -> int:
        return hash((self.drive_lines, self.readout_lines, self.cycles_per_update))

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

    def as_dict(self) -> dict[str, Any]:
        """The bus as JSON-serializable data, as a system file holds it."""
        return {
            "drive_lines": list(self.drive_lines),
            "readout_lines": list(self.readout_lines),
            "cycles_per_update": self.cycles_per_update,
        }

    @property
    def _lines(self) -> tuple[tuple[str, int], ...]:
        """Its lines as (role, carrier index)."""
        return (
            *(("drive", q) for q in self.drive_lines),
            *(("readout", q) for q in self.readout_lines),
        )

    def _named(self) -> str:
        """Its lines, as a violation names them."""
        parts = []
        for role, qudits in (("drive", self.drive_lines), ("readout", self.readout_lines)):
            if qudits:
                s = "s" if len(qudits) > 1 else ""
                parts.append(f"{role} line{s} of qudit{s} {', '.join(map(str, qudits))}")
        return " and ".join(parts)


def _update_buses(value: Any) -> tuple[UpdateBus, ...]:
    """*value*, ``Controller.update_buses``, as buses (a mapping, as a system
    file states one, is read as ``UpdateBus(**mapping)``), each line on at
    most one. A line the controller does not play has no updates."""
    if isinstance(value, (str, bytes, Mapping, UpdateBus)) or not isinstance(value, Iterable):
        raise InvalidTypeError(
            f"Controller.update_buses is a list of cudaq_pulse.controller.UpdateBus, got {value!r}"
        )
    buses = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            try:
                item = UpdateBus(**item)
            except TypeError as error:
                raise InvalidValueError(f"Controller.update_buses[{index}]: {error}") from None
        elif not isinstance(item, UpdateBus):
            raise InvalidTypeError(
                f"Controller.update_buses[{index}] is not a cudaq_pulse.controller.UpdateBus: "
                f"{item!r}"
            )
        buses.append(item)
    seen: set[tuple[str, int]] = set()
    for index, bus in enumerate(buses):
        for role, qudit in bus._lines:
            if (role, qudit) in seen:
                raise InvalidValueError(
                    f"Controller.update_buses[{index}]: the {role} line of qudit {qudit} is on "
                    "another bus already; a line is on at most one"
                )
            seen.add((role, qudit))
    return tuple(buses)


_QUDIT: Callable[[Any], int] = itemgetter(0)
_DELAY: Callable[[Any], float] = itemgetter(1)
# Lines given as carrier indices only.
_PLAIN = frozenset({int})
# By class, an undelayed line made from a (qudit, 0.0) pair.
_UNDELAYED: dict[type[_Line], Callable[[tuple[int, float]], Any]] = {
    DriveLine: partial(tuple.__new__, DriveLine),
    ReadoutLine: partial(tuple.__new__, ReadoutLine),
}
_NO_DELAY = repeat(0.0)


def _lines_data(lines: Sequence[Any]) -> list[Any]:
    """*lines* as a system file holds them: undelayed dedicated ones as
    carrier indices (a shared line's item 1 is never 0)."""
    if any(map(_DELAY, lines)):
        return [line._data() for line in lines]
    return list(map(_QUDIT, lines))


def _delay_ns(given: Any, where: str) -> float:
    """*given*, a line delay as a time or a number of ns, checked, in ns."""
    from ._units import as_nanoseconds

    ns = as_nanoseconds(given)
    if isinstance(ns, bool) or not isinstance(ns, (int, float)) or not math.isfinite(ns):
        raise InvalidValueError(f"{where} must be a number of ns or a time, got {given!r}")
    if ns < 0:
        raise InvalidValueError(
            f"{where} is {ns} ns; delays cannot be negative: to play a line earlier, "
            "delay every other line instead"
        )
    return float(ns)


def _lines(value: Iterable[Any], name: str, cls: type[_Line]) -> tuple[Any, ...]:
    """*value*, the lines ``Controller.<name>`` gives, as *cls* lines (a
    bare carrier index is an undelayed one)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise InvalidTypeError(f"Controller.{name} is a sequence of carrier indices, got {value!r}")
    given = tuple(value)
    if not given:
        return given
    if set(map(type, given)) <= _PLAIN and min(given) >= 0:
        if len(set(given)) != len(given):
            raise InvalidValueError(f"Controller.{name} lists a carrier twice: {list(given)}")
        return tuple(map(_UNDELAYED[cls], zip(given, _NO_DELAY, strict=False)))
    lines = []
    for line in given:
        if line.__class__ is not cls:
            if isinstance(line, _Line):
                # A broadcast line is of a class of its own (_broadcast_lines).
                if not isinstance(line, cls):
                    raise InvalidTypeError(
                        f"Controller.{name} holds {cls.__name__}s or carrier indices, got {line!r}"
                    )
            else:
                if isinstance(line, bool) or not isinstance(line, int):
                    raise InvalidTypeError(
                        f"Controller.{name} names a line by the carrier index of its "
                        f"qudit, or as {cls.__name__}(qudit, delay_ns=...), or a line "
                        f"reaching several qudits as {cls.__name__}([0, 1, 2, 3], "
                        f"max_concurrent_tones=...), got {line!r}"
                    )
                if line < 0:
                    raise InvalidValueError(
                        f"Controller.{name} has a negative carrier index {line}"
                    )
                line = tuple.__new__(cls, (int(line), 0.0))
        lines.append(line)
    carriers = list(map(_QUDIT, lines))
    if not _PLAIN.issuperset(map(type, carriers)):
        from ._shared_lines import reached_once

        reached_once(lines, cls)
    elif len(set(carriers)) != len(carriers):
        raise InvalidValueError(f"Controller.{name} lists a carrier twice: {carriers}")
    return tuple(lines)


# numpy's bools, which are not ints: like True, not a number of cycles.
_BOOL_TYPES = frozenset({"bool_", "bool"})


def _cycles(value: Any, name: str, *, whole_floats: bool = False) -> int:
    if value.__class__ is Time:
        raise InvalidTypeError(
            f"Controller.{name} is a whole number of the controller's own cycles, "
            f"got the time {value}; divide it by the controller's cycle time"
        )
    if (
        whole_floats
        and not isinstance(value, (bool, int))
        and type(value).__name__ not in _BOOL_TYPES
        and isinstance(value, numbers.Number)
        # Real, or a number not complex at all, as Decimal is.
        and (isinstance(value, numbers.Real) or not isinstance(value, numbers.Complex))
    ):
        # A whole number of another type (6.0, numpy's int64, Fraction(6),
        # Decimal(6)) is that number of cycles; a fraction of a cycle, an
        # infinity or a NaN is not.
        number: Any = value
        whole: int | None
        try:
            whole = int(number)
        except (OverflowError, ValueError):
            whole = None
        if whole is None or whole != number:
            raise InvalidValueError(f"Controller.{name} is a whole number of cycles, got {value!r}")
        value = whole
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidTypeError(f"Controller.{name} is a whole number of cycles, got {value!r}")
    if value < 0:
        raise InvalidValueError(f"Controller.{name} cannot be negative, got {value}")
    return value


_PROCESSOR_LIMITS = (
    "max_ops_per_function",
    "max_ops_total",
    "max_state_bytes_per_function",
    "max_state_bytes_total",
    "max_calls_per_shot",
    "max_bytes_per_shot",
)


def _one_cycle() -> OpCycles:
    from ._op_tables import _ONE_CYCLE

    return _ONE_CYCLE


def _op_class(given: Any) -> Any:
    """*given*, an operation ``Processor.unsupported`` names, as an ``OpClass``."""
    from ._op_tables import _PROCESSOR_OPS

    if isinstance(given, str) and given in _PROCESSOR_OPS:
        if given.__class__ is str:
            from ._op_classes import OpClass

            return OpClass(given)
        return given
    hint = _closest(str(given), _PROCESSOR_OPS)
    raise InvalidValueError(
        f"Processor.unsupported names an unknown operation {given!r}"
        + (f" (did you mean {hint!r}?)" if hint else "")
        + f"; the operations are {', '.join(_PROCESSOR_OPS)}"
    )


def _processor_of(data: Mapping[str, Any]) -> Processor:
    """The processor a description or a system file states as *data*, its
    ``cycles_per_op`` a table of cycles by class name."""
    from ._op_tables import _PROCESSOR_OPS, OpCycles

    given = dict(data)
    table = given.get("cycles_per_op")
    if isinstance(table, Mapping):
        for name in table:
            if name not in _PROCESSOR_OPS:
                hint = _closest(str(name), _PROCESSOR_OPS)
                raise InvalidValueError(
                    f"Processor.cycles_per_op names an unknown operation {name!r}"
                    + (f" (did you mean {hint!r}?)" if hint else "")
                    + f"; the operations are {', '.join(_PROCESSOR_OPS)}"
                )
        given["cycles_per_op"] = OpCycles(**table)
    return Processor(**given)


def _dispatch_keys(given: Any, owner: str) -> Mapping[str, int]:
    if not isinstance(given, Mapping):
        raise InvalidTypeError(f"{owner}.dispatch_keys maps function names to keys, got {given!r}")
    owners: dict[int, str] = {}
    for name, key in given.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise InvalidValueError(
                f"{owner}.dispatch_keys is keyed by function name, got {name!r}"
            )
        if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key <= 0xFFFFFFFF:
            raise InvalidValueError(
                f"{owner}.dispatch_keys[{name!r}] is {key!r}; a dispatch key is a whole "
                "number from 0 to 0xFFFFFFFF"
            )
        if key in owners:
            raise InvalidValueError(
                f"{owner}.dispatch_keys gives {key} to both {owners[key]!r} and {name!r}; "
                "give each function its own key"
            )
        owners[key] = name
    return MappingProxyType(dict(given))


@_dataclass_when_used(frozen=True)
class Processor:
    """A classical processor on a controller, which runs the classical
    functions called on its ``links`` (link names of the control system, or
    link ids). A link belongs to at most one processor of a system; a
    function on no processor runs off the controllers, and only its link is
    checked.

    ``clock_ghz``: its clock (``None``: the controller's, else the
    program's). ``cycles_per_op``: an ``OpCycles``, the cycles of one
    operation of each class, and of ``loop``, a rolled loop's counter and
    jump per iteration; one cycle when not given. ``branch`` is an ``if`` or
    ``else``. ``unsupported``: the classes it cannot run (``OpClass``); a
    function that uses one does not compile, except ``OpClass.LOOP``: a
    processor without it runs each loop unrolled, so its code is counted
    unrolled.
    ``max_concurrent_calls``: the calls it computes at once, from submission
    to the time the answer is due.
    ``max_ops_per_function`` and ``max_ops_total``: the operations one
    function, and all its functions, may hold. ``max_state_bytes_per_function``
    and ``max_state_bytes_total``: the bytes of arguments, results and
    locals of one function, and of all of them. ``max_calls_per_shot`` and
    ``max_bytes_per_shot``: the calls, and the bytes of requests and
    answers, of one shot. ``None`` is no limit. ``fixed_cost_only``: every
    function's cost must be fixed when the program compiles (no ``while``,
    recursion or allocation). ``dispatch_keys``: the dispatch key of each
    function it runs, by function name, when the processor fixes them; it
    then lists every function it runs.
    """

    links: Sequence[int | str] = ("controller",)
    clock_ghz: float | None = None
    cycles_per_op: OpCycles = field(default_factory=_one_cycle)
    unsupported: Sequence[OpClass] = ()
    max_concurrent_calls: int = 1
    max_ops_per_function: int | None = None
    max_ops_total: int | None = None
    max_state_bytes_per_function: int | None = None
    max_state_bytes_total: int | None = None
    max_calls_per_shot: int | None = None
    max_bytes_per_shot: int | None = None
    fixed_cost_only: bool = False
    dispatch_keys: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.links, (str, bytes)) or not isinstance(self.links, Iterable):
            raise InvalidTypeError(
                f"Processor.links is a list of link names or ids, got {self.links!r}"
            )
        links = tuple(self.links)
        for link in links:
            if isinstance(link, str):
                if not link.isidentifier():
                    raise InvalidValueError(f"Processor.links names a link, got {link!r}")
            elif isinstance(link, bool) or not isinstance(link, int) or link < 0:
                raise InvalidTypeError(
                    f"Processor.links holds link names or ids >= 0, got {link!r}"
                )
        if len(set(links)) != len(links):
            raise InvalidValueError(f"Processor.links lists a link twice: {list(links)}")
        object.__setattr__(self, "links", links)
        clock = self.clock_ghz
        if clock is not None and (
            isinstance(clock, bool)
            or not isinstance(clock, (int, float))
            or not math.isfinite(clock)
            or not clock > 0
        ):
            raise InvalidValueError(
                f"Processor.clock_ghz must be a positive number or None, got {clock!r}"
            )
        from ._op_tables import OpCycles

        costs = self.cycles_per_op
        if costs.__class__ is not OpCycles:
            raise InvalidTypeError(
                "Processor.cycles_per_op is an OpCycles, as OpCycles(div=20); got "
                f"{type(costs).__name__}"
            )
        if isinstance(self.unsupported, (str, bytes)) or not isinstance(self.unsupported, Iterable):
            raise InvalidTypeError(
                f"Processor.unsupported is a list of operations, got {self.unsupported!r}"
            )
        unsupported = tuple(dict.fromkeys(map(_op_class, self.unsupported)))
        both = [str(op) for op in unsupported if costs[op] != 1]
        if both:
            raise InvalidValueError(
                f"Processor.unsupported lists {both[0]!r}, which cycles_per_op gives a cost; "
                "an operation is supported or not"
            )
        object.__setattr__(self, "unsupported", unsupported)
        calls = self.max_concurrent_calls
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
            raise InvalidValueError(
                f"Processor.max_concurrent_calls is a whole number >= 1, got {calls!r}"
            )
        for name in _PROCESSOR_LIMITS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise InvalidValueError(
                    f"Processor.{name} is a whole number >= 1 or None, got {value!r}"
                )
        if not isinstance(self.fixed_cost_only, bool):
            raise InvalidTypeError(
                f"Processor.fixed_cost_only is True or False, got {self.fixed_cost_only!r}"
            )
        object.__setattr__(self, "dispatch_keys", _dispatch_keys(self.dispatch_keys, "Processor"))

    def __hash__(self) -> int:
        return hash(
            tuple(
                tuple(sorted(value.items())) if isinstance(value, Mapping) else value
                for value in (getattr(self, f.name) for f in fields(self))
            )
        )

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

    @property
    def loops(self) -> bool:
        """Whether it runs a rolled loop as a loop (``"loop"`` is not
        ``unsupported``); without, its loops are unrolled."""
        return "loop" not in self.unsupported

    def usage(self, function: Any) -> FunctionUsage:
        """What the classical *function* uses of this processor on its own:
        its operations, state and messages, and its cycles on the longest
        path; ``compute_ns`` only when the processor has ``clock_ghz`` (else
        ``None``), and ``calls_per_shot`` ``None`` (a compiled program's
        ``controller_usage`` has both). A function whose list lengths come
        from its calls has a cost per version: ask for one,
        ``processor.usage(flip.specialize(outcomes=3))``."""
        from ._controller_cost import function_usage

        return function_usage(self, function)

    def as_dict(self) -> dict[str, Any]:
        """The processor as JSON-serializable data, as a system file holds it:
        its ``cycles_per_op`` a table of the classes not 1 cycle."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["links"] = list(self.links)
        data["cycles_per_op"] = self.cycles_per_op._given()
        data["unsupported"] = [str(op) for op in self.unsupported]
        data["dispatch_keys"] = dict(self.dispatch_keys)
        return data


@_dataclass_when_used(frozen=True)
class Controller:
    """One controller: what it can play, its wiring and its processors;
    every limit is in its own clock cycles.

    ``clock_ghz``: the controller's clock (``None``: the program clock,
    ``QuantumSystem.clock_ghz``). It must be a whole-cycle ratio of the program
    clock; its program counts its own cycles.
    ``cycles_per_timestep``: the controller starts and times operations only
    every this many of its own cycles; every operation on its lines must
    start and last whole timesteps (see "Clocks" in the controller backends
    guide; ``pulse.round_duration`` rounds a length to them).
    ``instruction_cycles`` and ``min_idle_cycles``: two operations on one
    line start exactly ``instruction_cycles`` apart, or at least
    ``instruction_cycles + min_idle_cycles`` apart, because the controller
    issues one instruction per ``instruction_cycles`` and idles no less than
    ``min_idle_cycles``.
    ``min_pulse_cycles``: the shortest drive or readout its lines play (0:
    no limit), as some instruments require a minimum pulse length; a
    shorter one is a violation naming its line. Exports list it among the
    limits only when it is set.
    ``max_pulse_cycles``: the longest drive or readout its lines play
    (``None``: no limit), as some instruments cap a pulse's length; a longer
    one is a violation naming its line. It may not be below
    ``min_pulse_cycles``. Exports list it among the limits only when it is
    set.
    ``sample_rate_ghz``: the rate its lines play samples at (``None``: one
    per cycle of its clock). Its ratio to the clock may be a fraction, and
    every operation on its lines must start and last a whole number of
    samples, which the timestep includes: at 2.4 GHz on a 1 GHz clock, whole
    multiples of 5 cycles. A length off them is a compile error, never
    rounded.
    ``waveform_memory_samples``: a waveform memory holds this many samples,
    at ``sample_rate_ghz``; waveforms of one shape at different amplitudes
    share it (``shape_key``).
    ``waveform_memory``: ``"per_line"`` (each line has a memory of
    ``waveform_memory_samples``) or ``"shared"`` (one memory serves every
    line); see ``WaveformMemoryScope``.
    ``max_frequencies_per_line``: distinct frequencies one line plays over a program.
    ``max_amplitude``: the largest magnitude of a sample its lines play
    (``None``: no limit). The compiler sets no amplitude limit of its own;
    a drive or readout whose envelope peaks above this one is a violation
    naming its line. Exports list it among the limits only when it is set.
    ``detectors`` and ``detector_recovery_cycles``: the detector reading
    each qudit; readouts on one detector are serialized, each holding it
    for its duration plus ``detector_recovery_cycles``.
    ``feedback_latency_cycles``: the least time from the end of a readout to
    the start of any operation that depends on its outcome (in a branch on
    it, on a tone whose frame a branch on it changed, computed from it, or a
    request that sends it); every schedule waits it.
    ``supports_timeouts``: whether the controller can stop waiting for an
    answer after a timeout (``result(timeout=...)``, ``latency_mode=
    "timeout"``); without, only waits that block or are budgeted are
    allowed.
    ``max_pulses``: the drives and readouts of one program; a controller
    that supports ``"loop"`` holds a rolled loop's body once.
    ``max_instructions``: the operations one program may hold, as the
    kernel traces them (``None``: no limit). An unrolled loop holds its
    body once per iteration, a rolled one once; tracing stops when the
    program grows past the limit, and the error names the loop that
    unrolled it. Exports list it among the limits only when it is set.
    ``max_program_duration``: the longest program the controller runs, a
    time or a number of ns (``None``: no limit). The scheduled program,
    to the end of its last operation on the controller's lines (line delays
    and the waits for feedback, lines and links included), is checked
    against it, on one controller or several, so operations that play
    together on different lines count once; a longer one is a compile error
    giving its length (with a wait whose length floats set, an error at the
    call). A wait whose length is known only when the program runs counts at
    its highest length for the lines it holds back, and is a compile error
    where it has no highest, as is a blocking classical call that an
    operation on those lines follows. When the module is built the compiler
    also sums the
    traced program op after op, every iteration of its loops, and refuses it
    where the sum passes the limit times the lines the program takes: no
    schedule of it fits then. Without the limit, a program may last up to
    2**60 cycles. Exports list it only when it is set.
    ``max_fine_delay_ns``: the longest delay, below one grid step, each line
    can add to its output (0: none), a time or a number of ns; it plays the
    part of a line's calibrated delay (``DriveLine``, ``ReadoutLine``) that
    whole grid steps cannot.
    ``alignment_tolerance_cycles``: how far its pulses may land from where
    they are meant to be on other controllers, and, without fine delay, how
    far a line's delay may be from a whole number of grid steps (the
    compiler rounds it to the nearest one).
    ``persistent_frames``: whether the frame of each tone, its frequency and
    phase, persists from one shot to the next. Without, every shot starts
    from the reset frame, and the export gives a frame change no later
    operation uses on some path as a frame update of its own; with, it
    exports instead the frame each tone ends the shot in (``frame_state``).
    ``blocks_on_calls``: whether the controller plays nothing while one of
    its classical calls is out: each call holds every line from its
    submission until its answer is due, after the lines before it are done.
    The scheduler places calls so, and a program played on such a
    controller is checked for it; a call in a kernel with its own
    ``schedule=`` then does not compile. Exports list it among the limits
    only when it is set. In a control system of several controllers, a
    call holds the lines of each such controller that issues it, reads a
    measurement it sends or plays a pulse that uses its answer.

    ``drive_lines`` and ``readout_lines`` are the drive and readout lines it
    plays: ``DriveLine(5, delay_ns=4.0)`` and ``ReadoutLine(5,
    delay_ns=12.0)``, each with its qudit (its carrier index in the quantum
    system) and calibrated delay, or a bare carrier index for an undelayed
    line (they read back as lines); the only controller of a system that
    lists none plays every line. A line may reach several qudits
    (``ReadoutLine([0, 1, 2, 3], max_concurrent_tones=4)``, a shared line);
    each qudit is reached by at most one line of each kind.
    ``update_buses``: the groups of its lines whose updates share one bus
    (``UpdateBus``), which carries one update at a time; updates on one bus
    must start ``cycles_per_update`` apart, else the program is a violation
    naming the bus, the lines and the times (none by default: no limit).
    Exports list it among the limits only when it is set.
    ``space_updates``: whether compiling spaces the updates on its update
    buses, holding each back as little as it must, instead of refusing them
    (see "Update buses" in the controller backends guide). Off by default;
    exports list it only when it is set.
    ``peer_latency_cycles`` is the time an outcome this controller reads
    takes to reach another controller, in this controller's own cycles: one
    number for every receiver, or a mapping from the receiving controller's
    index in the system (a receiver it does not list: 0).
    ``clock_skew_cycles`` bounds how far its clock is from the system's.
    ``supports`` lists its capabilities (``None``: everything); a program
    that needs one it lacks fails to compile. ``backend`` is the controller
    backend that lowers its programs, by entry-point name. ``model`` and
    ``version`` identify the description it was made from. ``processors``
    are its classical processors (``Processor``): the classical functions
    it computes itself, and their limits.
    """

    clock_ghz: float | None = None
    cycles_per_timestep: int = 1
    instruction_cycles: int = 1
    min_idle_cycles: int = 0
    min_pulse_cycles: int = 0
    max_pulse_cycles: int | None = None
    sample_rate_ghz: float | None = None
    waveform_memory_samples: int | None = None
    max_frequencies_per_line: int | None = None
    max_amplitude: float | None = None
    detectors: Mapping[int, int] = field(default_factory=dict)
    detector_recovery_cycles: int = 0
    feedback_latency_cycles: int = 0
    max_pulses: int | None = None
    max_fine_delay_ns: float | Time = 0.0
    alignment_tolerance_cycles: float = 0
    supports_timeouts: bool = True
    max_instructions: int | None = None
    max_program_duration: float | Time | None = None
    waveform_memory: WaveformMemoryScope | str = "per_line"
    persistent_frames: bool = False
    blocks_on_calls: bool = False
    drive_lines: Sequence[DriveLine | int] = ()
    readout_lines: Sequence[ReadoutLine | int] = ()
    update_buses: Sequence[UpdateBus] = ()
    space_updates: bool = False
    peer_latency_cycles: int | Mapping[int, int] = 0
    clock_skew_cycles: int = 0
    supports: Sequence[Capability | str] | None = None
    backend: str | None = None
    model: str | None = None
    version: str | None = None
    processors: Sequence[Processor] = ()

    def __post_init__(self) -> None:
        for name in ("persistent_frames", "blocks_on_calls", "space_updates"):
            if not isinstance(getattr(self, name), bool):
                raise InvalidValueError(
                    f"Controller.{name} must be a bool, got {getattr(self, name)!r}"
                )
        if self.clock_ghz is not None and not self.clock_ghz > 0:
            raise InvalidValueError(f"Controller.clock_ghz must be positive, got {self.clock_ghz}")
        from .controller import WaveformMemoryScope

        try:
            scope = WaveformMemoryScope(self.waveform_memory)
        except ValueError:
            raise InvalidValueError(
                "Controller.waveform_memory must be 'per_line' or 'shared', "
                f"got {self.waveform_memory!r}"
            ) from None
        object.__setattr__(self, "waveform_memory", scope)
        steps = self.cycles_per_timestep
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise InvalidValueError(
                f"Controller.cycles_per_timestep must be an int of at least 1, got {steps!r}"
            )
        if self.instruction_cycles < 1:
            raise InvalidValueError(
                f"Controller.instruction_cycles must be at least 1, got {self.instruction_cycles!r}"
            )
        for name in ("min_idle_cycles", "detector_recovery_cycles"):
            value = getattr(self, name)
            if value < 0:
                raise InvalidValueError(f"Controller.{name} cannot be negative, got {value!r}")
        object.__setattr__(
            self,
            "feedback_latency_cycles",
            _cycles(self.feedback_latency_cycles, "feedback_latency_cycles", whole_floats=True),
        )
        least = self.min_pulse_cycles
        if isinstance(least, bool) or not isinstance(least, int) or least < 0:
            raise InvalidValueError(
                f"Controller.min_pulse_cycles must be an int of at least 0, got {least!r}"
            )
        most = self.max_pulse_cycles
        if most is not None and (
            isinstance(most, bool) or not isinstance(most, int) or most < max(least, 1)
        ):
            raise InvalidValueError(
                "Controller.max_pulse_cycles must be None or an int of at least 1 and at "
                f"least min_pulse_cycles ({least}), got {most!r}"
            )
        object.__setattr__(self, "max_fine_delay_ns", as_nanoseconds(self.max_fine_delay_ns))
        if self.max_program_duration is not None:
            object.__setattr__(
                self, "max_program_duration", _longest_program(self.max_program_duration)
            )
        for name in ("max_fine_delay_ns", "alignment_tolerance_cycles"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise InvalidValueError(f"Controller.{name} must be a finite number >= 0")
        peak = self.max_amplitude
        if peak is not None and (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(peak)
            or not peak > 0
        ):
            raise InvalidValueError(
                f"Controller.max_amplitude must be a positive number or None, got {peak!r}"
            )
        rate = self.sample_rate_ghz
        if rate is not None and (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not rate > 0
        ):
            raise InvalidValueError(
                f"Controller.sample_rate_ghz must be a positive number or None, got {rate!r}"
            )
        for name in ("waveform_memory_samples", "max_frequencies_per_line", "max_pulses"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise InvalidValueError(f"Controller.{name} must be positive or None")
        size = self.max_instructions
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
            raise InvalidValueError(
                f"Controller.max_instructions must be a positive int or None, got {size!r}"
            )
        object.__setattr__(self, "detectors", MappingProxyType(dict(self.detectors)))
        object.__setattr__(self, "drive_lines", _lines(self.drive_lines, "drive_lines", DriveLine))
        object.__setattr__(
            self, "readout_lines", _lines(self.readout_lines, "readout_lines", ReadoutLine)
        )
        object.__setattr__(self, "update_buses", _update_buses(self.update_buses))
        peers = self.peer_latency_cycles
        if isinstance(peers, Mapping):
            object.__setattr__(
                self,
                "peer_latency_cycles",
                MappingProxyType(
                    {
                        _cycles(k, "peer_latency_cycles keys"): _cycles(v, "peer_latency_cycles")
                        for k, v in peers.items()
                    }
                ),
            )
        else:
            _cycles(peers, "peer_latency_cycles")
        _cycles(self.clock_skew_cycles, "clock_skew_cycles")
        if self.supports is not None:
            if isinstance(self.supports, str):
                raise InvalidTypeError(
                    f"Controller.supports is a list of capabilities, got {self.supports!r}"
                )
            from ._capability import Capability

            supported = []
            for given in self.supports:
                try:
                    supported.append(Capability(given))
                except ValueError:
                    hint = _closest(str(given), [c.value for c in Capability])
                    raise InvalidValueError(
                        f"Controller.supports has an unknown capability {given!r}"
                        + (f" (did you mean {hint!r}?)" if hint else "")
                        + f"; the capabilities are {', '.join(c.value for c in Capability)}"
                    ) from None
            object.__setattr__(self, "supports", tuple(dict.fromkeys(supported)))
        for name in ("backend", "model", "version"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise InvalidTypeError(f"Controller.{name} is a string, got {value!r}")
        if isinstance(self.processors, (str, bytes, Processor, Mapping)) or not isinstance(
            self.processors, Iterable
        ):
            raise InvalidTypeError(
                f"Controller.processors is a list of pulse.Processor, got {self.processors!r}"
            )
        processors = []
        for index, item in enumerate(self.processors):
            if isinstance(item, Mapping):
                # As a description or a system file states it.
                try:
                    item = _processor_of(item)
                except TypeError as error:
                    raise InvalidValueError(f"Controller.processors[{index}]: {error}") from None
            elif not isinstance(item, Processor):
                raise InvalidTypeError(
                    f"Controller.processors[{index}] is not a pulse.Processor: {item!r}"
                )
            processors.append(item)
        object.__setattr__(self, "processors", tuple(processors))

    def __hash__(self) -> int:
        return hash(
            tuple(
                tuple(sorted(value.items())) if isinstance(value, Mapping) else value
                for value in (getattr(self, f.name) for f in fields(self))
            )
        )

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

    def as_dict(self) -> dict[str, Any]:
        """The controller as JSON-serializable data, which ``Controller(**data)``
        reads back after ``_controller_fields``."""
        # Every field in order: the class has none that fields() would leave out.
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        data["detectors"] = {str(k): v for k, v in sorted(self.detectors.items())}
        for name in (
            "max_amplitude",
            "max_instructions",
            "max_program_duration",
            "max_pulse_cycles",
        ):
            if data[name] is None:
                del data[name]
        if not self.min_pulse_cycles:
            del data["min_pulse_cycles"]
        # Memory per line exports as it did before the key existed.
        if self.waveform_memory == "per_line":
            del data["waveform_memory"]
        else:
            data["waveform_memory"] = str(self.waveform_memory)
        if not self.persistent_frames:
            del data["persistent_frames"]
        if not self.blocks_on_calls:
            del data["blocks_on_calls"]
        if self.drive_lines or self.readout_lines:
            data["drive_lines"] = _lines_data(self.drive_lines)
            data["readout_lines"] = _lines_data(self.readout_lines)
        else:
            data["drive_lines"] = []
            data["readout_lines"] = []
        if self.update_buses:
            data["update_buses"] = [bus.as_dict() for bus in self.update_buses]
        else:
            del data["update_buses"]
        if not self.space_updates:
            del data["space_updates"]
        peers = self.peer_latency_cycles
        if not isinstance(peers, int):
            data["peer_latency_cycles"] = {str(k): v for k, v in sorted(peers.items())}
        if self.supports is not None:
            data["supports"] = [str(c) for c in self.supports]
        if self.processors:
            data["processors"] = [p.as_dict() for p in self.processors]
        else:
            del data["processors"]
        return data

    @property
    def _all_lines(self) -> tuple[_Line, ...]:
        """Its drive lines, then its readout lines, each a line."""
        lines: tuple[Any, ...] = (*self.drive_lines, *self.readout_lines)
        return lines

    def supports_capability(self, capability: Capability) -> bool:
        """Whether the controller can do *capability*."""
        return self.supports is None or capability in self.supports

    @property
    def _machine(self) -> _MachineModel:
        """What compiling for this controller alone needs, as
        ``compile(control_system=controller)`` does."""
        return _one_controller(self)._machine

    @property
    def _runs_loops(self) -> bool:
        """Whether a rolled loop's body counts once against ``max_pulses``."""
        return self.supports is None or "loop" in self.supports


def _longest_program(value: Any) -> float:
    """``Controller.max_program_duration`` in ns."""
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        ns = float(as_nanoseconds(value))
        if math.isfinite(ns) and ns > 0:
            return ns
    raise InvalidValueError(
        "Controller.max_program_duration must be a positive time or number of "
        f"nanoseconds, or None; got {value!r}"
    )


# What an error about the system compile() was given as one Controller says.
_DESCRIBE = (
    "give compile() a pulse.ControlSystem that describes the whole control system: its "
    "controllers, the lines each plays and its links"
)


def _one_controller(controller: Controller) -> ControlSystem:
    """The system of *controller* alone, for ``compile(control_system=controller)``:
    what the one controller cannot take is an error that says to describe
    the whole system."""
    try:
        system = ControlSystem(controllers=[controller])
    except QuantumSystemDefinitionError as error:
        raise QuantumSystemDefinitionError(
            f"{_count(len(error.problems), 'problem')} in the control system of the one "
            f"Controller compile() was given; {_DESCRIBE}",
            problems=error.problems,
        ) from None
    object.__setattr__(system._machine, "_bare", True)
    return system


def _known(owners: Iterable[int | None]) -> list[int]:
    return sorted({o for o in owners if o is not None})


class _SeenBy(Sequence[dict[str, Any]]):
    """*events* as the controller *index* sees them: another controller's
    drives and readouts untimed, made as each is read."""

    __slots__ = ("_events", "_index", "_owners")

    def __init__(
        self, events: Sequence[dict[str, Any]], owners: Sequence[int | None], index: int
    ) -> None:
        self._events = events
        self._owners = owners
        self._index = index

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, k: Any) -> Any:
        return self._seen(self._events[k])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return map(self._seen, self._events)

    def _seen(self, e: dict[str, Any]) -> dict[str, Any]:
        if e["kind"] not in _PLAYED or self._owners[e["line"]] == self._index:
            return e
        return {**e, "end_cycles": None}


def _timed_by_controller(
    events: Sequence[dict[str, Any]], owners: Sequence[int | None], count: int
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """For each of *count* controllers, the timed drives, readouts and plays
    of *events* it sees (``_SeenBy``), in order; and the answers."""
    from .controller import _PLAYED as TIMED

    timed: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    answers = []
    for e in events:
        kind = e["kind"]
        if kind in TIMED:
            if e["end_cycles"] is None:
                continue
            if kind in _PLAYED:
                owner = owners[e["line"]]
                if owner is not None:
                    timed[owner].append(e)
            else:
                for mine in timed:
                    mine.append(e)
        elif kind == "answer":
            answers.append(e)
    return timed, answers


def _on_buses(
    controllers: Sequence[Controller],
    events: Sequence[dict[str, Any]],
    owners: Sequence[int | None],
) -> list[list[dict[str, Any]] | None]:
    """For each of *controllers* with update buses that lists its lines, the
    frame updates, drives and readouts of *events* on its lines, in order:
    every event its update-bus check reads (``controller._updates``). ``None``
    for any other, whose check reads every event as it sees them
    (``_SeenBy``): a controller that lists no lines has every line on its
    buses."""
    from .controller import _PLAYED as TIMED

    found: list[list[dict[str, Any]] | None] = [
        [] if c.update_buses and c._all_lines else None for c in controllers
    ]
    for e in events:
        if e["kind"] == "frame_update" or e["kind"] in TIMED:
            line = e["line"]
            if line is not None:
                owner = owners[line]
                if owner is not None:
                    mine = found[owner]
                    if mine is not None:
                        mine.append(e)
    return found


def _cycles_text(cycles: Fraction) -> str:
    return str(cycles.numerator) if cycles.denominator == 1 else f"{float(cycles):g}"


def _takers(
    events: Sequence[Mapping[str, Any]], owners: Sequence[int | None]
) -> dict[int, set[int]]:
    """The controllers that take each answer: those playing a line event
    that tests it or computes a value from it at run time."""
    takers: dict[int, set[int]] = {}
    if not any(event["kind"] == "answer" for event in events):
        return takers
    for event in events:
        if event["kind"] not in _NEEDS:
            continue
        owner = owners[event["line"]]
        if owner is None:
            continue
        for condition in event["conditions"]:
            for ref in condition["answers"]:
                takers.setdefault(ref, set()).add(owner)
        stack = [value for key, value in event.items() if key != "conditions"]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                ref = value.get("answer")
                if type(ref) is int:
                    takers.setdefault(ref, set()).add(owner)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return takers


def _label(index: int, controller: Controller) -> str:
    name = f"controllers[{index}]"
    return f"{name} ({controller.model})" if controller.model else name


# Up to this many controllers, a system finds one by a scan and checks for
# one listed twice pair by pair; above it, by their ids (_Controllers).
_FEW_CONTROLLERS = 8


class _Controllers:
    """The controllers of a system of many, by id: ``index`` and ``in`` as
    on their tuple (the same Controller, else an equal one), without a scan.
    It stays valid because the system is frozen and holds its controllers; a
    copy or an unpickled system is built again and makes its own."""

    __slots__ = ("_controllers", "_positions", "repeats")

    def __init__(self, controllers: tuple[Controller, ...]) -> None:
        self._controllers = controllers
        self._positions = dict(zip(map(id, controllers), range(len(controllers)), strict=True))
        # Whether one is listed twice: fewer ids than controllers.
        self.repeats = len(self._positions) < len(controllers)

    def __contains__(self, controller: object) -> bool:
        return id(controller) in self._positions or controller in self._controllers

    def index(self, controller: Controller) -> int:
        try:
            return self._positions[id(controller)]
        except KeyError:
            return self._controllers.index(controller)


@dataclass(frozen=True)
class ControlSystem:
    """The controllers of one QPU and what connects them; see the module
    docstring.

    A system without controllers states only its links, lanes, reference
    clock and line delays, and nothing checks what its controllers can
    play.
    ``links`` are the links to classical services, by name (``"feedback"``,
    ``"controller"``); a classical function's ``link=`` names one, or gives
    its id: the links' positions, so the first is link 0, the default. A
    link id a program uses that is not listed gets ``Link()``.
    ``Link.controllers`` says which controllers a link reaches. Their slots
    and latencies apply to every schedule, because they are hardware.
    ``max_concurrent_drives`` and ``max_concurrent_readouts``, when given,
    are the most drives and readouts the system plays at once, counted on
    every controller: the ``rcp`` and ``alap_rcp`` schedules keep to them,
    and a program whose schedule plays more at once fails to compile under
    any schedule. ``None`` (not given): the ``rcp`` schedules play at most 4
    drives and 2 readouts at once, and nothing else checks them.
    ``max_concurrent_drives``, ``max_concurrent_readouts`` and
    ``line_switch_penalty_cycles`` apply to the ``rcp`` and ``alap_rcp``
    schedules, which keep pulses of two controllers that follow each other on
    one lane their start uncertainty apart. Every schedule keeps an operation
    that depends on a readout at least the ``feedback_latency_cycles`` of the
    controller that read it after the readout ends (on another controller,
    its peer latency and both start uncertainties too).
    ``reference_clock_ghz`` is a clock every controller locks to, if they
    share one: each controller's clock must be a whole multiple of it, and
    without it each controller's start is known only to within one of its
    own cycles. The calibrated delay of each line is part of the line, as
    its controller declares it (``DriveLine``, ``ReadoutLine``).
    """

    controllers: Sequence[Controller] = ()
    links: Mapping[str, Link] = field(default_factory=dict)
    max_concurrent_drives: int | None = None
    max_concurrent_readouts: int | None = None
    line_switch_penalty_cycles: int = 0
    reference_clock_ghz: float | None = None
    # Set by __post_init__, and declared for type checkers only: as fields
    # they would slow the first use of the class.
    if TYPE_CHECKING:
        # (role, carrier) -> the delay in ns of each delayed line, in line order.
        _line_delays: Mapping[tuple[str, int], float] = field(init=False)
        _machine: _MachineModel = field(init=False)
        _plays_everything: bool = field(init=False)
        # By link id and controller, the processor that runs its calls there.
        _placement: Mapping[int, Mapping[int, int]] = field(init=False)
        # The clocks of the controllers that run their own, by label.
        _own_clocks: Mapping[str, float] = field(init=False)
        # Whether a quantum system of it checks its program clock against its clocks.
        _needs_clock: bool = field(init=False)
        # (role, carrier) -> the index of the controller that plays the line.
        _line_owners: Mapping[tuple[str, int], int] = field(init=False)
        # By program clock, the clocks of a program on several controllers.
        _plans: dict[float, _clocks.ClockPlan | None] = field(init=False)
        # Whether a sync lines up its controllers' pulses exactly, so a check
        # may take a rolled loop's copies for its first (``_summarized``) when
        # they count program cycles.
        _lines_up: bool = field(init=False)
        # The indices of the controllers that block on calls.
        _blocking: tuple[int, ...] = field(init=False)
        _shares: ClassVar[Any]
    else:
        # False, or with a line that reaches several qudits what writes its
        # programs' exports: the _shared_lines module, or with a broadcast
        # line _broadcast_lines; set only then. Not a field.
        _shares = False

    def __post_init__(self) -> None:
        from .compiler import _NO_DELAYS, _MachineModel

        if isinstance(self.controllers, Controller) or not isinstance(self.controllers, Iterable):
            raise InvalidTypeError(
                f"ControlSystem.controllers is a list of pulse.Controller, got {self.controllers!r}"
            )
        controllers = tuple(self.controllers)
        for index, controller in enumerate(controllers):
            if not isinstance(controller, Controller):
                raise InvalidTypeError(
                    f"ControlSystem.controllers[{index}] is not a pulse.Controller: {controller!r}"
                )
        object.__setattr__(self, "controllers", controllers)
        count = len(controllers)
        single = count == 1
        if not isinstance(self.links, Mapping):
            raise InvalidTypeError(
                f"ControlSystem.links maps link names to Link, got {self.links!r}"
            )
        for name, link in self.links.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise InvalidValueError(
                    f"ControlSystem.links are keyed by name (an identifier), got {name!r}"
                )
            if not isinstance(link, Link):
                raise InvalidTypeError(
                    f"ControlSystem.links[{name!r}] must be a Link, got {link!r}"
                )
        object.__setattr__(self, "links", MappingProxyType(dict(self.links)))
        ghz = self.reference_clock_ghz
        if ghz is not None and not ghz > 0:
            raise InvalidValueError(
                f"ControlSystem.reference_clock_ghz must be positive, got {ghz}"
            )
        owners = {
            (role, carrier): index
            for index, controller in enumerate(controllers)
            for role, lines in (
                ("drive", controller.drive_lines),
                ("readout", controller.readout_lines),
            )
            for carrier in map(_QUDIT, lines)
        }
        if owners and not _PLAIN.issuperset(map(type, map(_DELAY, owners))):
            # A shared line: each of its qudits gets its controller and delay.
            from ._shared_lines import tables

            # With a broadcast line, the model that compiles for one.
            owners, delays, shares, _MachineModel = tables(controllers, _MachineModel)  # type: ignore[misc]  # noqa: N806
            object.__setattr__(self, "_shares", shares)
            repeated = False
        else:
            delays = {
                (line._role, line[0]): line[1]
                for controller in controllers
                for line in filter(_DELAY, controller._all_lines)
            }
            # Fewer (role, qudit) than lines: a qudit's line on two controllers.
            repeated = len(owners) < sum(
                len(c.drive_lines) + len(c.readout_lines) for c in controllers
            )
        object.__setattr__(
            self,
            "_line_delays",
            MappingProxyType(dict(sorted(delays.items()))) if delays else _NO_DELAYS,
        )
        if count > _FEW_CONTROLLERS:
            # Many controllers: found by id, and checked pair by pair only
            # when one is listed twice.
            members = _Controllers(controllers)
            object.__setattr__(self, "_index", members.index)
            problems = _structure_problems(
                self, "", repeated, controllers, members, controllers if members.repeats else ()
            )
        else:
            problems = _structure_problems(
                self, "", repeated, controllers, controllers, controllers
            )
        if problems:
            raise QuantumSystemDefinitionError(
                f"{_count(len(problems), 'problem')} in the control system",
                problems=problems,
            )
        object.__setattr__(
            self,
            "_plays_everything",
            single and not (controllers[0].drive_lines or controllers[0].readout_lines),
        )
        object.__setattr__(self, "_placement", MappingProxyType(_placement(self)))
        object.__setattr__(
            self,
            "_own_clocks",
            MappingProxyType(
                {
                    _label(index, c): c.clock_ghz
                    for index, c in enumerate(controllers)
                    if c.clock_ghz is not None
                }
            ),
        )
        object.__setattr__(
            self,
            "_needs_clock",
            bool(self._own_clocks) or self.reference_clock_ghz is not None,
        )
        object.__setattr__(self, "_line_owners", MappingProxyType(owners))
        object.__setattr__(self, "_plans", {})
        object.__setattr__(
            self,
            "_lines_up",
            single
            or (
                self.reference_clock_ghz is not None
                and not self._line_delays
                and not any(c.clock_skew_cycles for c in controllers)
            ),
        )
        object.__setattr__(
            self,
            "_blocking",
            tuple(index for index, c in enumerate(controllers) if c.blocks_on_calls),
        )
        links: dict[int | str, Link] = {name: link for name, link in self.links.items()}
        one = controllers[0] if single else None
        # Programs are checked against the system as one controller (or,
        # without controllers, not at all) when that controller plays every
        # line, supports everything, computes nothing, and no link reaches
        # only some controllers.
        alone = not controllers or (
            one is not None
            and self._plays_everything
            and one.supports is None
            and not one.processors
            and not any(
                link.controllers is not None or link.per_controller for link in links.values()
            )
        )
        object.__setattr__(
            self,
            "_machine",
            _MachineModel(
                max_concurrent_drives=self.max_concurrent_drives,
                max_concurrent_readouts=self.max_concurrent_readouts,
                readout_latency_cycles=self._outcome_latency(None) if controllers else 0,
                line_switch_penalty_cycles=self.line_switch_penalty_cycles,
                links=links,
                controller=one,
                reference_clock_ghz=self.reference_clock_ghz,
                _line_delays=self._line_delays,
                _system=None if alone else self,
            ),
        )

    def __hash__(self) -> int:
        # Kept: a kernel's cache key holds the control system its qudits
        # are lowered for, read at every call.
        found: int | None = self.__dict__.get("_hash")
        if found is None:
            found = hash(
                (
                    self.controllers,
                    tuple(sorted(self.links.items())),
                    self.max_concurrent_drives,
                    self.max_concurrent_readouts,
                    self.line_switch_penalty_cycles,
                    self.reference_clock_ghz,
                )
            )
            object.__setattr__(self, "_hash", found)
        return found

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import frozen_reduce

        return frozen_reduce(self)

    def as_dict(self) -> dict[str, Any]:
        """The control system as JSON-serializable data; links name the
        controllers they reach by index."""
        data: dict[str, Any] = {
            "controllers": [c.as_dict() for c in self.controllers],
            "links": {
                name: {
                    **link.cost(),
                    **(
                        {}
                        if link.controllers is None
                        else {"controllers": [self._index(c) for c in link.controllers]}
                    ),
                }
                for name, link in self.links.items()
            },
            "max_concurrent_drives": self.max_concurrent_drives,
            "max_concurrent_readouts": self.max_concurrent_readouts,
            "line_switch_penalty_cycles": self.line_switch_penalty_cycles,
        }
        if self.reference_clock_ghz is not None:
            data["reference_clock_ghz"] = self.reference_clock_ghz
        return data

    @property
    def _one_box(self) -> bool:
        """Whether its programs are checked as of its one controller, with
        its processors: the controller plays every line and supports
        everything, and no link reaches only some controllers."""
        return (
            self._plays_everything
            and self.controllers[0].supports is None
            and not any(
                link.controllers is not None or link.per_controller for link in self.links.values()
            )
        )

    def _clock_plan(self, program_ghz: float) -> _clocks.ClockPlan | None:
        """The clocks of a program at *program_ghz* on the controllers (of a
        system of several), with each line's controller and the readout
        latency in program cycles; ``None`` when all run at the program clock
        in steps of one cycle and no reference clock is given."""
        try:
            return self._plans[program_ghz]
        except KeyError:
            pass
        from . import _clocks

        controllers = self.controllers
        found = _clocks.controllers_plan(
            controllers,
            [_label(index, c) for index, c in enumerate(controllers)],
            self._line_owners,
            program_ghz,
            self.reference_clock_ghz,
        )
        if found is not None:
            found = dataclasses.replace(found, readout_latency_cycles=self._outcome_latency(found))
        self._plans[program_ghz] = found
        return found

    def _outcome_latency(self, clocks: _clocks.ClockPlan | None) -> int:
        """The longest a readout's outcome takes to be usable, in program
        cycles: on the controller that reads it, after its feedback latency;
        on another, after its ``peer_latency_cycles`` to it and both
        controllers' start uncertainty too. The scheduler holds one readout
        latency, so every readout waits for this one."""
        controllers = self.controllers
        if clocks is None and len(controllers) == 1:
            return controllers[0].feedback_latency_cycles
        from fractions import Fraction

        count = len(controllers)
        everyone = range(count)
        readers = [i for i in everyone if controllers[i].readout_lines] or everyone
        longest = Fraction(0)
        if count < 2:
            for reader in readers:
                ratio = Fraction(1) if clocks is None else clocks.controllers[reader].ratio
                longest = max(longest, controllers[reader].feedback_latency_cycles / ratio)
            return math.ceil(longest)
        users = [
            i for i in everyone if controllers[i].drive_lines or controllers[i].readout_lines
        ] or everyone
        uncertainty = self._start_uncertainty(clocks)
        if len(users) == 1:
            user = users[0]
            for reader in readers:
                source = controllers[reader]
                ratio = Fraction(1) if clocks is None else clocks.controllers[reader].ratio
                longest = max(longest, source.feedback_latency_cycles / ratio)
                if user != reader:
                    peers = source.peer_latency_cycles
                    peer = peers.get(user, 0) if isinstance(peers, Mapping) else peers
                    latency = (source.feedback_latency_cycles + peer) / ratio
                    longest = max(longest, latency + uncertainty[reader] + uncertainty[user])
            return math.ceil(longest)
        # Each reader reaches another user, after its feedback latency and
        # more (no latency or start uncertainty is negative), so that latency
        # alone never decides. Without clocks every ratio is 1, and latencies
        # stay whole numbers of cycles.
        first = second = -1
        for reader in readers:
            source = controllers[reader]
            # A whole number of cycles without clocks, else a Fraction.
            scale: Fraction | None = None if clocks is None else clocks.controllers[reader].ratio
            own = uncertainty[reader]
            peers = source.peer_latency_cycles
            if isinstance(peers, Mapping):
                for user in users:
                    if user != reader:
                        wait: int | Fraction = source.feedback_latency_cycles + peers.get(user, 0)
                        if scale is not None:
                            wait /= scale
                        longest = max(longest, wait + own + uncertainty[user])
                continue
            # One peer latency to every user: the latest is on the user with
            # the largest start uncertainty other than the reader.
            if first < 0:
                for user in users:
                    if first < 0 or uncertainty[user] > uncertainty[first]:
                        first, second = user, first
                    elif second < 0 or uncertainty[user] > uncertainty[second]:
                        second = user
            wait = source.feedback_latency_cycles + peers
            if scale is not None:
                wait /= scale
            other = second if reader == first else first
            longest = max(longest, wait + own + uncertainty[other])
        return math.ceil(longest)

    def _start_uncertainty(self, clocks: _clocks.ClockPlan | None) -> list[Fraction]:
        """How well each controller's start is known, in program cycles: its
        ``clock_skew_cycles``, plus one of its own cycles when the system
        shares no reference clock (*clocks*: the program's, ``None`` when
        every controller runs at the program clock)."""
        from fractions import Fraction

        extra = 0 if self.reference_clock_ghz is not None else 1
        return [
            (c.clock_skew_cycles + extra)
            / (Fraction(1) if clocks is None else clocks.controllers[i].ratio)
            for i, c in enumerate(self.controllers)
        ]

    def _controllers_attribute(self, program_ghz: float, max_duration_cycles: int | None) -> str:
        """The ``controllers`` of the control system of a program at
        *program_ghz* (``#pulse.control_system``): each controller's grid,
        the facts the scheduler derives the latencies and spreads between
        controllers from, in its own cycles (its cycles per program cycle,
        its feedback and peer latency, and its start uncertainty: its
        ``clock_skew_cycles``, plus one cycle when the system shares no
        reference clock), whether it blocks on calls, and
        *max_duration_cycles*; each fact left out at its default."""
        clocks = self._clock_plan(program_ghz)
        extra = 0 if self.reference_clock_ghz is not None else 1
        self._latencies_fit(clocks, extra)
        entries = []
        for a, controller in enumerate(self.controllers):
            fields = []
            if clocks is not None:
                clock = clocks.controllers[a]
                if clock.grid > 1:
                    fields.append(f"grid_cycles = {clock.grid}")
                if clock.ratio != 1:
                    fields.append(f"clock_ratio = {clock.ratio}")
            if controller.feedback_latency_cycles:
                fields.append(f"feedback_latency_cycles = {controller.feedback_latency_cycles}")
            peers = controller.peer_latency_cycles
            if isinstance(peers, Mapping):
                if peers:
                    given = ", ".join(f"{k} = {v}" for k, v in sorted(peers.items()))
                    fields.append(f"peer_latency_cycles = {{{given}}}")
            elif peers:
                fields.append(f"peer_latency_cycles = {peers}")
            if uncertainty := controller.clock_skew_cycles + extra:
                fields.append(f"start_uncertainty_cycles = {uncertainty}")
            if controller.blocks_on_calls:
                fields.append("blocks_on_calls")
            if max_duration_cycles is not None:
                fields.append(f"max_duration_cycles = {int(max_duration_cycles)}")
            entries.append(f"#pulse.controller<{', '.join(fields)}>")
        return f"[{', '.join(entries)}]"

    def _latencies_fit(self, clocks: _clocks.ClockPlan | None, extra: int) -> None:
        """Raise unless every outcome latency between controllers, in program
        cycles, fits the 64 bits the scheduler counts in: the reader's
        feedback and peer latency and both start uncertainties (*extra*: the
        cycle each adds without a shared reference clock)."""
        most = 2**63 - 1
        ratios = (
            [(1, 1)] * len(self.controllers)
            if clocks is None
            else [(c.ratio.numerator, c.ratio.denominator) for c in clocks.controllers]
        )

        def late(controller: Controller, peer: int, num: int, den: int) -> tuple[int, int]:
            return divmod(
                (controller.feedback_latency_cycles + peer + controller.clock_skew_cycles + extra)
                * den,
                num,
            )

        def start(controller: Controller, num: int, den: int) -> tuple[int, int]:
            return divmod((controller.clock_skew_cycles + extra) * den, num)

        def peers(controller: Controller) -> Mapping[int, int]:
            given = controller.peer_latency_cycles
            return given if isinstance(given, Mapping) else {}

        def default(controller: Controller) -> int:
            given = controller.peer_latency_cycles
            return 0 if isinstance(given, Mapping) else given

        # A bound first, in O(C): the latest reader plus the least known start.
        latest = least = 0
        for controller, (num, den) in zip(self.controllers, ratios, strict=True):
            peer = max([default(controller), *peers(controller).values()])
            whole, rest = late(controller, peer, num, den)
            latest = max(latest, whole + (rest > 0))
            whole, rest = start(controller, num, den)
            least = max(least, whole + (rest > 0))
        if latest + least <= most:
            return
        # Then every pair, exactly.
        for r, reader in enumerate(self.controllers):
            num, den = ratios[r]
            for u, user in enumerate(self.controllers):
                if u == r:
                    continue
                user_num, user_den = ratios[u]
                a, a_rest = late(reader, peers(reader).get(u, default(reader)), num, den)
                x, x_rest = start(user, user_num, user_den)
                total = a + x
                if a_rest or x_rest:
                    total += 1 + (a_rest * user_num + x_rest * num > num * user_num)
                if total > most:
                    raise InvalidValueError(
                        f"{_label(r, reader)}: the outcome of its readout reaches "
                        f"{_label(u, user)} {total} program cycles after the readout ends, "
                        "more than the 2**63 - 1 the scheduler counts; lower its "
                        "feedback_latency_cycles or peer_latency_cycles, or the "
                        "clock_skew_cycles of either"
                    )

    def _links_attribute(
        self, machine: _MachineModel, used: Sequence[int], program_ghz: float
    ) -> str:
        """``machine.links_attribute(used)`` on this system: each link that
        joins controllers carries the start uncertainty between the two of
        them known least well, in ns, which the budget of a call whose sender
        and taker the scheduler cannot tell adds. A per-controller link joins
        none (its calls stay on one) and carries its name."""
        from fractions import Fraction

        each = self._start_uncertainty(self._clock_plan(program_ghz))
        entries = []
        for link_id in sorted(set(used)):
            link = machine.link(link_id)
            if link.per_controller:
                entry = link.attribute(0.0, machine.link_name(link_id))
            else:
                reached = link.controllers
                indices = (
                    range(len(self.controllers))
                    if reached is None
                    else sorted({self._index(c) for c in reached})
                )
                worst = sorted((each[i] for i in indices), reverse=True)
                uncertainty = 0.0
                if len(worst) > 1 and worst[0] + worst[1]:
                    uncertainty = float((worst[0] + worst[1]) / Fraction(program_ghz))
                entry = link.attribute(uncertainty)
            entries.append(f'"{link_id}" = {entry}')
        return f"{{classical.links = {{{', '.join(entries)}}}}}"

    def _forwarding(
        self,
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        clocks: _clocks.ClockPlan | None,
        run_time: bool,
    ) -> list[str]:
        """The pulses that branch on, or compute with, a readout of another
        controller before its outcome can reach them: the reader's feedback
        latency, its ``peer_latency_cycles`` to the user, and both
        controllers' start uncertainty, each in its own cycles (events in
        program cycles). Only a program that computes at run time
        (*run_time*) computes with a readout."""
        from fractions import Fraction

        found: list[str] = []
        uncertainty: list[Fraction] | None = None
        for event in events:
            if event["kind"] not in _PLAYED:
                continue
            user = owners[event["line"]]
            if user is None or event["start_cycles"] is None:
                continue
            uses = [
                (ref, "branches on")
                for condition in event["conditions"]
                for ref in condition["readouts"]
            ]
            stack = [v for k, v in event.items() if k != "conditions"] if run_time else []
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    ref = value.get("readout")
                    if type(ref) is int:
                        uses.append((ref, "computes with"))
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
            for ref, verb in dict.fromkeys(uses):
                readout = events[ref]
                reader = owners[readout["line"]]
                if reader is None or reader == user or readout["end_cycles"] is None:
                    continue
                if uncertainty is None:
                    uncertainty = self._start_uncertainty(clocks)
                source = self.controllers[reader]
                peers = source.peer_latency_cycles
                peer = peers.get(user, 0) if isinstance(peers, Mapping) else peers
                ratio = Fraction(1) if clocks is None else clocks.controllers[reader].ratio
                latency = (source.feedback_latency_cycles + peer) / ratio
                spread = uncertainty[reader] + uncertainty[user]
                total = latency + spread
                ready = readout["end_cycles"] + math.ceil(total)
                if event["start_cycles"] >= ready:
                    continue
                found.append(
                    f"{_label(user, self.controllers[user])}: the {event['kind']} on line "
                    f"{event['line']} at {event['start_cycles']} {verb} the readout "
                    f"of {_label(reader, source)} ending at {readout['end_cycles']}, whose "
                    f"outcome reaches it {_cycles_text(total)} cycles later, at {ready} "
                    f"({_cycles_text(latency)} of feedback and peer latency, and "
                    f"{_cycles_text(spread)} of the two controllers' start uncertainty"
                    + (
                        ""
                        if self.reference_clock_ghz is not None
                        else ", without a shared reference_clock_ghz"
                    )
                    + f"); start it at {ready} or later: schedule with rcp, which waits "
                    f"for it, or wait {ready - event['start_cycles']} cycles before it on "
                    "its line"
                )
        return found

    def _alignment(
        self,
        lines: Sequence[Mapping[str, Any]],
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        clocks: _clocks.ClockPlan | None,
        program_ghz: float,
    ) -> list[str]:
        """The pulses a sync lines up across controllers that may land
        further from where they are meant to than their
        ``alignment_tolerance_cycles``: the first pulse after the sync on each
        of its lines lands within its controller's start uncertainty and what
        its line delay is rounded by (each in the controller's cycles)."""
        import bisect

        if (
            self.reference_clock_ghz is not None
            and not self._line_delays
            and not any(c.clock_skew_cycles for c in self.controllers)
        ):
            return []
        by_line: dict[int, list[Mapping[str, Any]]] = {}
        for event in events:
            if event["kind"] in _PLAYED and event["start_cycles"] is not None:
                by_line.setdefault(event["line"], []).append(event)
        for items in by_line.values():
            items.sort(key=lambda e: (e["start_cycles"], e["index"]))
        starts = {line: [e["start_cycles"] for e in items] for line, items in by_line.items()}
        found: list[str] = []
        for sync in events:
            if sync["kind"] != "sync" or sync["start_cycles"] is None:
                continue
            if len(_known(owners[line] for line in sync["lines"])) < 2:
                continue
            paired: list[tuple[int, Mapping[str, Any]]] = []
            for line in sync["lines"]:
                owner = owners[line]
                if owner is None or line not in by_line:
                    continue
                after = bisect.bisect_left(starts[line], sync["start_cycles"])
                if after < len(starts[line]):
                    paired.append((owner, by_line[line][after]))
            parts = _known(owner for owner, _ in paired)
            if len(parts) < 2:
                continue
            trigger = 0 if self.reference_clock_ghz is not None else 1
            for owner in parts:
                controller = self.controllers[owner]
                mine = [e for o, e in paired if o == owner]
                skew = controller.clock_skew_cycles
                rounded = max(
                    self._delay_rounding(lines[e["line"]], owner, clocks, program_ghz) for e in mine
                )
                spread = skew + trigger + rounded
                if spread <= controller.alignment_tolerance_cycles + 1e-9:
                    continue
                terms = [f"{skew} of clock_skew_cycles"] if skew else []
                if trigger:
                    terms.append("1 without a shared reference_clock_ghz")
                if rounded:
                    terms.append(f"{rounded:g} of a rounded line delay")
                fixes = ["share a reference_clock_ghz between them"] if trigger else []
                fixes.append(f"raise its alignment_tolerance_cycles to {math.ceil(spread - 1e-9)}")
                pulses = ", ".join(f"the {e['kind']} on line {e['line']}" for e in mine)
                found.append(
                    f"{_label(owner, controller)}: the sync at {sync['start_cycles']} lines "
                    f"up {pulses} with the pulses of "
                    + ", ".join(_label(o, self.controllers[o]) for o in parts if o != owner)
                    + f", but they land within {spread:g} of its cycles of where they are "
                    f"meant to ({', '.join(terms)}), more than its "
                    f"alignment_tolerance_cycles of {controller.alignment_tolerance_cycles}; "
                    + ", or ".join(fixes)
                )
        return found

    def _delay_rounding(
        self,
        line: Mapping[str, Any],
        owner: int,
        clocks: _clocks.ClockPlan | None,
        program_ghz: float,
    ) -> float:
        """How far a line's delay is rounded to its grid, in its controller's
        cycles: 0 when the controller adds the rest as fine delay."""
        ns = self._line_delays.get((line["role"], line["qudit"]))
        controller = self.controllers[owner]
        if not ns or controller.max_fine_delay_ns > 0:
            return 0.0
        own = None if clocks is None else clocks.controllers[owner]
        step_ns = (1 if own is None else own.grid) / program_ghz
        rest = abs(ns - round(ns / step_ns) * step_ns)
        return 0.0 if rest < 1e-9 else rest * (program_ghz if own is None else own.clock_ghz)

    def _undelayed(self) -> ControlSystem:
        """The system with every line undelayed: what a program is traced
        for, which other delays only schedule again. Made once."""
        found: ControlSystem | None = self.__dict__.get("_undelayed_system")
        if found is None:
            if not self._line_delays:
                return self
            import dataclasses

            if self._shares:
                from ._shared_lines import undelayed

                controllers = undelayed(self.controllers)
            else:
                controllers = [
                    dataclasses.replace(
                        c,
                        drive_lines=list(map(_QUDIT, c.drive_lines)),
                        readout_lines=list(map(_QUDIT, c.readout_lines)),
                    )
                    for c in self.controllers
                ]
            links = {
                name: link
                if link.controllers is None
                else dataclasses.replace(
                    link, controllers=[controllers[self._index(c)] for c in link.controllers]
                )
                for name, link in self.links.items()
            }
            found = dataclasses.replace(self, controllers=controllers, links=links)
            object.__setattr__(self, "_undelayed_system", found)
        return found

    def _index(self, controller: Controller) -> int:
        # A system of many controllers replaces this with _Controllers.index
        # (__post_init__).
        for index, candidate in enumerate(self.controllers):
            if candidate is controller:
                return index
        return self.controllers.index(controller)

    # ── Checks of a compiled program ────────────────────────────────────────

    def _owners(self, lines: Sequence[Mapping[str, Any]]) -> tuple[list[int | None], list[str]]:
        """The index of the controller that plays each line, and a violation
        for each line no controller plays."""
        if self._plays_everything:
            return [0] * len(lines), []
        played = self._line_owners
        owners: list[int | None] = []
        found: list[str] = []
        for line in lines:
            owner = played.get((line["role"], line["qudit"]))
            owners.append(owner)
            if owner is None:
                where = f"line {line['line']} (the {line['role']} line of qudit {line['qudit']})"
                found.append(
                    f"{where} is not on the one Controller compile() was given; add "
                    f"{line['qudit']} to its {line['role']}_lines, or {_DESCRIBE}"
                    if getattr(self._machine, "_bare", False)
                    else f"{where} is on no controller; add {line['qudit']} to the "
                    f"{line['role']}_lines of one of the control system's controllers"
                )
        return owners, found

    def _violations(
        self,
        lowered: ScheduledKernel,
        lines: Sequence[Mapping[str, Any]],
        events: Sequence[dict[str, Any]],
        clocks: _clocks.ClockPlan | None,
        off: Sequence[tuple[int, int, int]],
        run_time: Sequence[str],
        waveforms: bool,
        unresolved: list[str] | None,
        natives: Sequence[int] | None,
    ) -> list[str]:
        """What the program needs of the system that it does not have: a
        controller for each line, each controller's limits on what it plays,
        its capabilities and the reach of the links. *clocks*, *off* and
        *run_time* are as ``controller._violations`` takes them."""
        from . import _clocks
        from .controller import _computes_at_run_time, _violations

        owners, found = self._owners(lines)
        if self._shares:
            found.extend(self._shares.violations(self, lowered, lines, events, unresolved))
        if len(self.controllers) == 1:
            found.extend(
                _violations(
                    lowered,
                    events,
                    self.controllers[0],
                    lines,
                    clocks,
                    off,
                    run_time,
                    waveforms,
                    unresolved,
                    natives,
                )
            )
            found.extend(self._needs(lowered, events, owners, unresolved))
        else:
            needs = self._needs(lowered, events, owners, unresolved)
            needs.extend(self._forwarding(events, owners, clocks, _computes_at_run_time(lowered)))
            needs.extend(
                self._alignment(lines, events, owners, clocks, lowered.quantum_system.clock_ghz)
            )
            if self._blocking:
                needs.extend(self._held(lowered, events, owners))
            if clocks is not None and clocks.owners is not None:
                # Each event in its controller's cycles, once for them all.
                found.extend(
                    _clocks.off_grid(
                        clocks, lines, events, True, waveforms, _computes_at_run_time(lowered)
                    )
                )
            timed, answers = _timed_by_controller(events, owners, len(self.controllers))
            buses = (
                _on_buses(self.controllers, events, owners)
                if any(c.update_buses for c in self.controllers)
                else [None] * len(self.controllers)
            )
            # An envelope's peak depends on its tree alone, which holds any
            # own-clock scaling: one sampling serves every controller.
            peaks: dict[str, float | None] = {}
            for index, controller in enumerate(self.controllers):
                label = _label(index, controller)
                found.extend(
                    f"{label}: {v}"
                    for v in _violations(
                        lowered,
                        _SeenBy(events, owners, index),
                        controller,
                        lines,
                        None,
                        off,
                        run_time,
                        waveforms,
                        unresolved,
                        natives,
                        held=False,
                        timed=timed[index],
                        answers=answers,
                        updates=buses[index],
                        peaks=peaks,
                    )
                )
            found.extend(needs)
        if self._placement:
            if unresolved is None:
                from .controller import usage

                found.extend(usage(lowered).violations)
            else:
                # When calls are due can depend on a call's values.
                unresolved.append("processors")
        return found

    def _held(
        self,
        lowered: ScheduledKernel,
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
    ) -> list[str]:
        """The pulses a controller that blocks on calls plays while a call
        that holds it is out (*events* in program cycles): a call holds each
        that issues it (``_callers``), reads a measurement it sends or plays a
        line event that uses its answer."""
        from .controller import _held_text, _played_while_out

        if not lowered.classical or not lowered.scheduled:
            return []
        blocking = set(self._blocking)
        takers = _takers(events, owners)
        answers: dict[tuple[Any, ...], list[int]] = {}
        for event in events:
            if event["kind"] == "answer":
                taken = (event["link"], event["function"], tuple(event["iteration"]))
                answers.setdefault((*taken, event["start_cycles"]), []).append(event["index"])
        submits = [e for e in events if e["kind"] == "submit"]
        calls = lowered.classical_report()
        used = _known(owners)
        holders = []
        for submit, call in zip(submits, calls, strict=True):
            held: set[int | None] = set(self._callers(submit, events, owners, used, takers))
            held.update(owners[events[ref]["line"]] for ref in submit["measurements"])
            key = (call.link, call.compiled_name, tuple(call.iteration), call.taken_cycles)
            for answer in answers.get(key, ()):
                held.update(takers.get(answer, ()))
            holders.append(held & blocking)
        return [
            f"{_label(owner, self.controllers[owner])}: {_held_text(submit, call, event)}"
            for submit, call, event in _played_while_out(
                events, calls, lambda index, event: owners[event["line"]] in holders[index]
            )
            if (owner := owners[event["line"]]) is not None
        ]

    def _summarized(
        self,
        lowered: ScheduledKernel,
        lines: Sequence[Mapping[str, Any]],
        events: Sequence[dict[str, Any]],
        repeats: Sequence[Sequence[int]],
        clocks: _clocks.ClockPlan | None,
        off: Sequence[tuple[int, int, int]],
        unresolved: list[str] | None,
        natives: Sequence[int] | None,
        rates: tuple[float, float] | None,
        periods: Sequence[Sequence[int]] = (),
    ) -> list[Any] | None:
        """``_violations`` of the events of a summarizing walk, on a system
        that ``_lines_up``, as ``_loops.parts`` gives them, with *rates* those
        of ``controller._sample_rate_violations`` and *periods* the (repeat,
        line, period) of loops whose lines move by periods of their own;
        ``None`` unless every copy of a loop needs and forwards what its
        first does, each period is whole timesteps of every controller, and
        *rates* come last. A sync lines up the pulses of several controllers
        exactly, so ``_alignment`` finds nothing."""
        from . import _clocks, _loops
        from .controller import _computes_at_run_time

        run_time = _computes_at_run_time(lowered)
        # The period of each line of each loop whose lines have their own.
        moves: dict[int, dict[int, int]] = {}
        for index, line, period in periods:
            moves.setdefault(index, {})[line] = period
        if clocks is not None and clocks.owners is not None:
            grids = {c.grid for c in clocks.controllers}
            steps = [r[3] for r in repeats if r[3] >= 0] + [p[2] for p in periods]
            if any(step % grid for step in steps for grid in grids):
                return None
            if _clocks.off_grid(clocks, lines, events, run_time, False, run_time):
                return None
        if rates is not None and self._placement:
            return None
        owners, found = self._owners(lines)
        several = len(self.controllers) > 1
        # A copy needs or forwards what its first does not: where calls are
        # checked (as _needs checks them), when an answer is taken across
        # its bounds (after its loop, or in the copy after it), so that its
        # takers are not the first's; and on several controllers, when
        # another controller's readout of it is tested or computed with
        # after its loop, which a later copy's reaches later. Only a program
        # that computes at run time computes with an answer or a readout.
        checks = any(c.supports is not None for c in self.controllers) or any(
            link.controllers is not None or link.per_controller for link in self.links.values()
        )
        for i, event in enumerate(events if checks or several else ()):
            conditions = event["conditions"]
            if not (conditions or run_time):
                continue
            kind = event["kind"]
            if kind not in _NEEDS:
                continue
            user = owners[event["line"]]
            if user is None:
                continue
            forwards = several and kind in _PLAYED
            if run_time:
                readouts: list[int] = []
                answers: list[int] = []
                stack = [value for key, value in event.items() if key != "conditions"]
                while stack:
                    value = stack.pop()
                    if isinstance(value, dict):
                        ref = value.get("readout")
                        if type(ref) is int:
                            readouts.append(ref)
                        ref = value.get("answer")
                        if type(ref) is int:
                            answers.append(ref)
                        stack.extend(value.values())
                    elif isinstance(value, list):
                        stack.extend(value)
                conditions = [*conditions, {"readouts": readouts, "answers": answers}]
            for condition in conditions:
                if forwards:
                    for ref in condition["readouts"]:
                        if owners[events[ref]["line"]] not in (None, user):
                            for k, r in enumerate(repeats):
                                if r[0] <= ref < r[1] and not r[0] <= i < r[1]:
                                    return None
                                # Copies of lines of different periods drift apart.
                                if (
                                    r[3] < 0
                                    and r[0] <= i < r[1]
                                    and r[0] - (r[1] - r[0]) <= ref < r[1]
                                    and moves.get(k, {}).get(event["line"])
                                    != moves.get(k, {}).get(events[ref]["line"])
                                ):
                                    return None
                if checks:
                    for ref in condition["answers"]:
                        for r in repeats:
                            begin, end = r[0], r[1]
                            if begin <= i < end:
                                if begin - (end - begin) <= ref < begin:
                                    return None
                            elif begin <= ref < end:
                                return None
        if self._needs(lowered, events, owners, unresolved):
            return None
        if self._shares:
            shared = self._shares.summarized(
                self, lowered, lines, events, repeats, unresolved, periods
            )
            if shared is None:
                return None
            found.extend(shared)
        if not several:
            from .controller import _native_clock

            try:
                found.extend(
                    _loops.parts(
                        lowered,
                        events,
                        repeats,
                        self.controllers[0],
                        lines,
                        unresolved,
                        rates,
                        periods,
                        _native_clock(clocks),
                        off,
                        natives,
                    )
                )
            except _loops.UnsyncedError:
                return None
        else:
            if self._forwarding(events, owners, clocks, run_time):
                return None

            # One tree of the events for all the controllers, and the pulses
            # each plays: a controller is checked on its own, not on a copy
            # of every event with the others' hidden.
            # (The walk's pulses are drives and readouts; a "play", which
            # only an export in format /2 holds, is no controller's here.)
            def owner(i: int) -> int | None:
                e = events[i]
                if e["kind"] not in _PLAYED or e["end_cycles"] is None:
                    return None
                found: int | None = owners[e["line"]]
                return found

            whole = _loops.Shared(events, repeats, periods, owner)
            for index, controller in enumerate(self.controllers):
                try:
                    own = _loops.parts(
                        lowered,
                        events,
                        repeats,
                        controller,
                        lines,
                        unresolved,
                        None,
                        periods,
                        None,
                        (),
                        natives,
                        whole,
                        index,
                    )
                except _loops.UnsyncedError:
                    return None
                found.extend(_loops.labeled(own, f"{_label(index, controller)}: "))
        if self._placement:
            if unresolved is None:
                from .controller import usage

                found.extend(usage(lowered).violations)
            else:
                unresolved.append("processors")
        return found

    def _needs(
        self,
        lowered: ScheduledKernel,
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        unresolved: list[str] | None = None,
    ) -> list[str]:
        """The capabilities the program needs that its controllers lack, and
        the calls whose link does not reach the controllers taking part. With
        *unresolved*, a tone or phase a call's values may set is noted there
        instead, to be checked with them."""
        checks_capabilities = any(c.supports is not None for c in self.controllers)
        checks_reach = any(
            link.controllers is not None or link.per_controller for link in self.links.values()
        )
        if not (checks_capabilities or checks_reach):
            return []
        found: list[str] = []
        used = _known(owners)
        users = _takers(events, owners)

        def owner(event: Mapping[str, Any]) -> int | None:
            found: int | None = owners[event["line"]]
            return found

        def where(event: Mapping[str, Any]) -> str:
            text = f"the {event['kind']}"
            if event["kind"] in _NEEDS:
                text += f" on line {event['line']} (qudit {event['qudit']})"
            elif event["kind"] in ("submit", "answer"):
                text += f" of {event['function']}"
            start = event["start_cycles"]
            return text if start is None else f"{text} at cycle {start}"

        def need(index: int, capability: str, event: Mapping[str, Any]) -> None:
            controller = self.controllers[index]
            if controller.supports is None or capability in controller.supports:
                return
            needs = f"{_label(index, controller)}: {where(event)} needs {capability!r}"
            found.append(
                f"{needs}, which the one Controller compile() was given does not support; "
                f"write that part of the program without {capability!r}, or {_DESCRIBE}"
                if getattr(self._machine, "_bare", False)
                else f"{needs}, which the controller does not support; run that part of the "
                "program on a controller that supports it, or write it without "
                f"{capability!r}"
            )

        for event in events:
            kind = event["kind"]
            if kind in _NEEDS:
                index = owner(event)
                if index is None:
                    continue
                need(index, _NEEDS[kind], event)
                if kind in _PLAYED and (
                    event["frequency_hz"] is None or event["phase_rad"] is None
                ):
                    # Without a call's values, the tone may be a parameter's.
                    if unresolved is None:
                        need(index, "runtime_frame_update", event)
                    elif "runtime_frame_update" not in (self.controllers[index].supports or ()):
                        unresolved.append(f"line {event['line']}: frame")
                if event["conditions"]:
                    need(index, "branch", event)
                for condition in event["conditions"]:
                    for ref in condition["readouts"]:
                        source = owner(events[ref])
                        if source is not None and source != index:
                            need(source, "forward_outcome", events[ref])
                    for ref in condition["answers"]:
                        need(index, "answer", events[ref])
            elif kind == "sync":
                for index in _known(owners[line] for line in event["lines"]):
                    need(index, "sync", event)
        for event in events:
            kind = event["kind"]
            if kind == "submit" or kind == "answer":
                callers = self._callers(event, events, owners, used, users)
                for index in callers:
                    need(index, "call" if kind == "submit" else "answer", event)
                verb = "sends the measurements of" if kind == "submit" else "answers"
                self._reach(lowered, event, callers, verb, found)
        return found

    def _parts(
        self, events: Sequence[Mapping[str, Any]], owners: Sequence[int | None]
    ) -> list[list[int]]:
        """The controllers that take part in each event, as ``_needs`` counts
        them: a line event's, a sync's lines', and a call's (``_callers``)."""
        used = _known(owners)
        takers = _takers(events, owners)
        parts: list[list[int]] = []
        for event in events:
            kind = event["kind"]
            if kind in _NEEDS:
                owner = owners[event["line"]]
                parts.append([] if owner is None else [owner])
            elif kind == "sync":
                parts.append(_known(owners[line] for line in event["lines"]))
            elif kind == "submit" or kind == "answer":
                parts.append(self._callers(event, events, owners, used, takers))
            else:
                parts.append(used)
        return parts

    def _callers(
        self,
        event: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        owners: Sequence[int | None],
        used: list[int],
        takers: Mapping[int, set[int]],
    ) -> list[int]:
        """The controllers that take part in a submit or answer *event*: for
        a call over a per-controller link, the one whose instance runs it;
        else a submit's callers (those of its measurements) and an answer's
        takers (those of the events that test it or compute with it), else
        every controller *used* when the link is shared, else the one its
        link leads to (``_home``)."""
        instance = event.get("controller")
        if instance is not None:
            return [instance]
        if event["kind"] == "submit":
            found = _known(owners[events[ref]["line"]] for ref in event["measurements"])
        else:
            found = sorted(takers.get(event["index"], ()))
        if found:
            return found
        if used and not _per_controller(self, event["link"]):
            return used
        return [self._home(event["link"])]

    def _home(self, link_id: int) -> int:
        """The controller a call over *link_id* is made on when the program
        plays no line: the first with a processor that runs it, else the
        first the link reaches, else ``controllers[0]``."""
        placed = self._placement.get(link_id)
        if placed:
            return min(placed)
        reached = self._machine.link(link_id).controllers
        return min(self._index(c) for c in reached) if reached else 0

    def _reach(
        self,
        lowered: ScheduledKernel,
        event: Mapping[str, Any],
        indices: Sequence[int],
        verb: str,
        found: list[str],
    ) -> None:
        machine = lowered.options.machine
        link = machine.link(event["link"])
        if link.per_controller and len(indices) > 1:
            name = machine.link_name(event["link"]) or f"link {event['link']}"
            on = ", ".join(_label(i, self.controllers[i]) for i in indices)
            found.append(
                f"the {event['kind']} of {event['function']} over link {name!r}, which has "
                f"one instance per controller, is on {on}; a call over it runs on one "
                "controller's instance: send its measurements and use its answer on one "
                "controller, or make the link shared (drop per_controller)"
            )
        if link.controllers is None:
            return
        reached = {self._index(c) for c in link.controllers}
        name = machine.link_name(event["link"]) or f"link {event['link']}"
        for index in indices:
            if index not in reached:
                found.append(
                    f"the {event['kind']} of {event['function']} over link {name!r} {verb} "
                    f"{_label(index, self.controllers[index])}, which the link does not "
                    f"reach; add it to the link's controllers, or play those lines on "
                    f"one of {sorted(reached)}"
                )


def _structure_problems(
    system: ControlSystem,
    prefix: str,
    repeated: bool,
    controllers: tuple[Controller, ...],
    members: tuple[Controller, ...] | _Controllers,
    twice: tuple[Controller, ...],
) -> list[Problem]:
    """What is wrong with *system* on its own: lines on two controllers
    (*repeated*: some qudit has a line of a kind on two), peers that are not
    in it, links that reach controllers it does not have. *controllers* are
    its controllers; *members* those a link's controllers are found among
    (them, or of many, ``_Controllers``); *twice* those checked pair by
    pair for one listed twice (them, or none when no id repeats)."""
    problems: list[Problem] = []
    for index, controller in enumerate(twice):
        for other in range(index):
            if controllers[other] is controller:
                problems.append(
                    Problem(
                        f"is the same Controller as controllers[{other}]",
                        path=f"{prefix}controllers[{index}]",
                        fix="give each controller of the system its own Controller",
                    )
                )
    if system._shares:
        from ._shared_lines import reach_problems

        problems.extend(reach_problems(system, prefix))
    for role in ("drive_lines", "readout_lines") if repeated else ():
        first: dict[int, int] = {}
        for index, controller in enumerate(controllers):
            for position, carrier in enumerate(map(_QUDIT, getattr(controller, role))):
                if carrier in first:
                    problems.append(
                        Problem(
                            f"qudit {carrier}'s {role[:-6]} line is also on "
                            f"controllers[{first[carrier]}]",
                            path=f"{prefix}controllers[{index}].{role}[{position}]",
                            value=carrier,
                            fix="a line is on one controller",
                        )
                    )
                else:
                    first[carrier] = index
    for index, controller in enumerate(controllers):
        peers = controller.peer_latency_cycles
        if isinstance(peers, Mapping):
            for peer in peers:
                if peer == index or peer >= len(controllers):
                    problems.append(
                        Problem(
                            f"names controller {peer}, which is "
                            + ("this controller" if peer == index else "not in the system"),
                            path=f"{prefix}controllers[{index}].peer_latency_cycles",
                            value=peer,
                            fix="key it by the index of another controller "
                            f"(0..{len(controllers) - 1})",
                        )
                    )
    for name, link in system.links.items():
        for position, controller in enumerate(link.controllers or ()):
            if controller not in members:
                problems.append(
                    Problem(
                        "is not a controller of the system",
                        path=f"{prefix}links.{name}.controllers[{position}]",
                        fix="list the system's own Controller objects",
                    )
                )
    names = list(system.links)
    claimed: dict[tuple[int, int], str] = {}
    for index, controller in enumerate(controllers):
        for number, processor in enumerate(controller.processors):
            here = f"controllers[{index}].processors[{number}]"
            for position, named in enumerate(processor.links):
                path = f"{prefix}{here}.links[{position}]"
                if isinstance(named, str) and named not in system.links:
                    hint = _closest(named, names)
                    problems.append(
                        Problem(
                            "is not a link of the control system",
                            path=path,
                            value=named,
                            fix=f"did you mean {hint!r}?"
                            if hint
                            else "add it to ControlSystem(links={...})",
                        )
                    )
                    continue
                link_id = names.index(named) if isinstance(named, str) else named
                own = _per_controller(system, link_id)
                key = (link_id, index if own else -1)
                if key in claimed:
                    problems.append(
                        Problem(
                            f"is also a link of {claimed[key]}",
                            path=path,
                            value=named,
                            fix="a link with an instance per controller belongs to one "
                            "processor of each controller"
                            if own
                            else "a link belongs to one processor, or with "
                            "Link(per_controller=True) to one of each controller",
                        )
                    )
                else:
                    claimed[key] = here
    return problems


def _per_controller(system: ControlSystem, link_id: int) -> bool:
    names = list(system.links)
    return link_id < len(names) and system.links[names[link_id]].per_controller


def _placement(system: ControlSystem) -> dict[int, dict[int, int]]:
    """For each processor's link id, the processor that runs its calls on
    each controller: one controller for a shared link, each controller with
    a processor for it for a per-controller link."""
    names = list(system.links)
    found: dict[int, dict[int, int]] = {}
    for index, controller in enumerate(system.controllers):
        for number, processor in enumerate(controller.processors):
            for link in processor.links:
                link_id = names.index(link) if isinstance(link, str) else link
                found.setdefault(link_id, {})[index] = number
    return found


def _wiring_problems(system: ControlSystem, target: _Description) -> list[Problem]:
    """The lines and detectors of *system* that name carriers *target* lacks."""
    carriers = set(target.qudits)
    problems: list[Problem] = []
    # Whether some line reaches a qudit not in it (each (role, qudit) a line reaches).
    stray = not carriers.issuperset(map(_DELAY, system._line_owners))
    for index, controller in enumerate(system.controllers):
        path = f"control_system.controllers[{index}]"
        if stray and system._shares:
            from ._shared_lines import wiring_problems

            problems.extend(wiring_problems(controller, path, carriers))
        for role in ("drive_lines", "readout_lines") if stray and not system._shares else ():
            for position, carrier in enumerate(map(_QUDIT, getattr(controller, role))):
                if carrier not in carriers:
                    problems.append(
                        Problem(
                            f"qudit {carrier} is not in the quantum system",
                            path=f"{path}.{role}[{position}]",
                            value=carrier,
                            fix=f"it has qudits {_ranges(sorted(carriers))}",
                        )
                    )
        for carrier in controller.detectors:
            if carrier not in carriers:
                problems.append(
                    Problem(
                        f"qudit {carrier} is not in the quantum system",
                        path=f"{path}.detectors",
                        value=carrier,
                        fix=f"it has qudits {_ranges(sorted(carriers))}",
                    )
                )
    return problems


# ── Controller descriptions ──────────────────────────────────────────────────

# Paths are strings here: building a control system does not need pathlib.
_DESCRIPTIONS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "controllers")
# What the installation gives, not the description.
_WIRING = (
    "drive_lines",
    "readout_lines",
    "detectors",
    "peer_latency_cycles",
    "clock_skew_cycles",
    "alignment_tolerance_cycles",
    "update_buses",
)
_NOTICED: set[tuple[str, str, str]] = set()


def installed_descriptions() -> dict[str, Path]:
    """The controller descriptions installed with cudaq-pulse, by name
    (``"vendor.model"``)."""
    from pathlib import Path

    return {name: Path(path) for name, path in _installed().items()}


def _installed() -> dict[str, str]:
    """``installed_descriptions()``, by path string."""
    found = {}
    for vendor in sorted(os.listdir(_DESCRIPTIONS)):
        folder = os.path.join(_DESCRIPTIONS, vendor)
        if os.path.isdir(folder):
            for file in sorted(os.listdir(folder)):
                if file.endswith(".toml"):
                    found[f"{vendor}.{file[:-5]}"] = os.path.join(folder, file)
    return found


def _toml_line(text: str, key: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=", text, re.MULTILINE)
    return None if match is None else text.count("\n", 0, match.start()) + 1


_PARSED: dict[tuple[str, bytes], Controller] = {}


def _description(path: str) -> Controller:
    """The controller the description file at *path* states. The file is
    read each time and parsed once for each content it has had, so a file
    written again is the file read, whatever its time stamp says."""
    with open(path, "rb") as f:
        content = f.read()
    key = (path, content)
    parsed = _PARSED.get(key)
    if parsed is None:
        # As a file opened as text reads: UTF-8, any line end as a newline.
        text = content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        parsed = _PARSED[key] = _parse_description(path, text)
    return parsed


_INSTALLED: dict[tuple[str, int], Controller] = {}


def _installed_description(path: str) -> Controller:
    """``_description`` for a description installed with cudaq-pulse, which
    nothing writes while a program runs: parsed once per time stamp, without
    reading the file again."""
    key = (path, os.stat(path).st_mtime_ns)
    parsed = _INSTALLED.get(key)
    if parsed is None:
        parsed = _INSTALLED[key] = _description(path)
    return parsed


def _parse_description(path: str, text: str) -> Controller:
    import tomllib

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        where = re.search(r"\(at line (\d+), column (\d+)\)", str(error))
        raise QuantumSystemDefinitionError(
            "the controller description is not valid TOML",
            problems=[
                Problem(
                    str(error).split(" (at line")[0],
                    file=path,
                    line=int(where.group(1)) if where else None,
                    column=int(where.group(2)) if where else None,
                )
            ],
        ) from None
    known = [f.name for f in fields(Controller) if f.name not in _WIRING]
    problems: list[Problem] = []

    def problem(key: str, message: str, fix: str = "") -> None:
        problems.append(Problem(message, path=key, fix=fix, file=path, line=_toml_line(text, key)))

    for key in data:
        if key in _WIRING:
            problem(
                key,
                "is wiring, which the installation gives",
                f"pass it when loading: pulse.controller(name, {key}=...)",
            )
        elif key not in known:
            hint = _closest(key, known)
            problem(key, "is not a controller field", f"did you mean {hint!r}?" if hint else "")
    for key in ("model", "version"):
        if not isinstance(data.get(key), str):
            problems.append(
                Problem(
                    f"the description has no {key} (a string)", file=path, fix=f'add {key} = "..."'
                )
            )
    if problems:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in the controller description",
            problems=problems,
        )
    given = {k: v for k, v in data.items() if k in known}
    try:
        return Controller(**given)
    except (TypeError, ValueError) as error:
        raise QuantumSystemDefinitionError(
            "the controller description has a bad value",
            problems=[Problem(str(error), file=path)],
        ) from None


def load_controller(description: str | os.PathLike[str], /, **wiring: Any) -> Controller:
    """The controller a description states, with the installation's wiring:
    ``pulse.controller("vendor.model", drive_lines=..., readout_lines=...)``.

    *description* is the name of an installed description or the path of a
    TOML file. A file given by its path is read at each call and parsed
    once for each content it has had, so a file written again is the file
    loaded; an installed description, given by its name, is parsed once
    for each modification time of its file. *wiring* sets any field of
    ``Controller``: its lines, detectors and peers, and overrides of the
    description's values.
    """
    if isinstance(description, str) and not description.endswith(".toml"):
        installed = _installed()
        path = installed.get(description)
        if path is None:
            hint = _closest(description, installed)
            raise NotFoundError(
                f"no controller description named {description!r}"
                + (f" (did you mean {hint!r}?)" if hint else "")
                + f"; installed: {', '.join(installed) or 'none'}, or give the path of a .toml file"
            )
        base = _installed_description(os.path.realpath(path))
    else:
        path = os.fspath(description)
        if not os.path.isfile(path):
            raise NotFoundError(f"no controller description file {path!r}")
        base = _description(os.path.realpath(path))
    if not wiring:
        return base
    names = [f.name for f in fields(Controller)]
    for key in wiring:
        if key not in names:
            hint = _closest(key, names)
            raise InvalidTypeError(
                f"pulse.controller() got an unknown field {key!r}"
                + (f" (did you mean {hint!r}?)" if hint else "")
            )
    return dataclasses.replace(base, **wiring)


def _notice_version(controller: Controller, path: str) -> None:
    """Notice, once, a controller made from another version of an installed
    description."""
    if controller.model is None or controller.version is None:
        return
    installed = installed_descriptions().get(controller.model)
    if installed is None:
        return
    current = load_controller(controller.model).version
    key = (controller.model, controller.version, current or "")
    if current == controller.version or key in _NOTICED:
        return
    _NOTICED.add(key)
    warnings.warn(
        f"{path}: made from {controller.model} version {controller.version}; the installed "
        f"description is version {current}. The file's values are used; to use the installed "
        f"ones, make the controller with pulse.controller({controller.model!r}, ...)",
        UpgradeNotice,
        stacklevel=4,
    )


# ── System files ─────────────────────────────────────────────────────────────

# The keys of a quantum system's data, by the field they set.
_QUDIT_FIELDS = (
    "index",
    "frequency_hz",
    "anharmonicity_hz",
    "t1_us",
    "t2_star_us",
    "label",
    "drive",
    "readout",
    "t2_echo_us",
    "thermal_population",
)
# Written only when a qudit gives them.
_QUDIT_EXTRAS = ("calibrations", "metadata")
_COUPLING_FIELDS = ("qudits", "strength_hz", "calibrations")
_RESIDUAL_ZZ_FIELDS = ("qudits", "rate_hz")
# The keys an entry must give, with the fix for each.
_QUDITS_FIX = "give the indices of the two qudits, as [0, 1]"
_COUPLING_REQUIRED = MappingProxyType(
    {
        "qudits": _QUDITS_FIX,
        "strength_hz": "give the strength in Hz, or null if it is not measured yet",
    }
)
_RESIDUAL_ZZ_REQUIRED = MappingProxyType(
    {"qudits": _QUDITS_FIX, "rate_hz": "give the ZZ rate in Hz"}
)
_CONTROL_KEYS = (
    "controllers",
    "links",
    "max_concurrent_drives",
    "max_concurrent_readouts",
    "line_switch_penalty_cycles",
    "reference_clock_ghz",
)
_LINK_KEYS = (
    "fixed_ns",
    "bytes_per_ns",
    "feedback_ns",
    "slots",
    "slot_bytes",
    "controllers",
    "per_controller",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _component_kinds() -> dict[str, Mapping[str, type[Any]]]:
    """The modality components of a qudit a system file holds, by the member
    they set and the kind it writes."""
    from .quantum_systems.base import _DRIVE_KINDS, DispersiveReadout

    return {"drive": _DRIVE_KINDS, "readout": {"dispersive": DispersiveReadout}}


def _component_data(member: str, component: Any) -> dict[str, Any] | None:
    if component is None:
        return None
    kinds = _component_kinds()[member]
    kind = next((k for k, cls in kinds.items() if type(component) is cls), None)
    if kind is None:
        from ._errors import UnsupportedError

        raise UnsupportedError(
            f"a system file holds a qudit's {member} as one of {', '.join(kinds)}; "
            f"got a {type(component).__name__}"
        )
    return {"kind": kind, **dict(zip(component._fields, component._values, strict=True))}


def _calibration_data(calibrations: Mapping[Any, Any]) -> list[dict[str, Any]]:
    """Calibrations by gate as a system file holds them: one entry per
    calibration, its gate, its transition if not (0, 1), and its fields."""
    data = []
    for key, pulse in calibrations.items():
        gate, transition = key if key.__class__ is tuple else (key, None)
        entry: dict[str, Any] = {"gate": str.__str__(gate)}
        if transition is not None:
            entry["transition"] = list(transition)
        entry.update(zip(pulse._fields, pulse._values, strict=True))
        data.append(entry)
    return data


def _qudit_data(q: Any) -> dict[str, Any]:
    data = {
        key: _component_data(key, value) if key in ("drive", "readout") else _plain(value)
        for key in _QUDIT_FIELDS
        for value in (getattr(q, key),)
    }
    if q._calibrations:
        data["calibrations"] = _calibration_data(q._calibrations)
    if q.metadata:
        data["metadata"] = _plain(q.metadata)
    return data


def _model_kinds() -> dict[str, type]:
    """The simulation models a system file holds, by the kind it writes."""
    from .quantum_systems.models import SimulationModel, TransmonModel

    return {"transmon": TransmonModel, "generic": SimulationModel}


def _model_data(model: Any) -> dict[str, Any] | None:
    """A simulation model as data: its kind and the levels it gives qudits."""
    if model is None:
        return None
    kind = next((k for k, cls in _model_kinds().items() if type(model) is cls), None)
    if kind is None:
        raise InvalidValueError(
            f"a system file holds only the shipped simulation models (transmon, generic); "
            f"{type(model).__name__} is not one, so give the quantum system by registered name"
        )
    data: dict[str, Any] = {"kind": kind}
    if model._levels:
        data["levels"] = {str(q): n for q, n in model._levels.items()}
    return data


def _quantum_data(target: _Description) -> dict[str, Any]:
    return {
        "model": _model_data(target.model),
        "attribution": target.attribution,
        "qudits": [_qudit_data(q) for q in sorted(target.qudits.values(), key=lambda q: q.index)],
        "couplings": [
            {"qudits": list(c.qudits), "strength_hz": c.strength_hz}
            | ({"calibrations": _calibration_data(c._calibrations)} if c._calibrations else {})
            for c in target.couplings
        ],
        "residual_zz": [
            {key: _plain(getattr(x, key)) for key in _RESIDUAL_ZZ_FIELDS}
            for x in target.residual_zz
        ],
    }


# A qudit's readout as a file holds it, a readout's frequency, the types
# of a frequency in Hz (or of none given; _device adds a float type it has
# looked at, as numpy.float64), and whether some types are all among them.
_READOUT = itemgetter("readout")
_READOUT_HZ = itemgetter("frequency_hz")
_HZ_TYPES = {float, int, type(None)}
_ONLY_HZ = _HZ_TYPES.issuperset


def _write_offsets(qudits: list[dict[str, Any]]) -> None:
    """The frequency of each of a file's *qudits*, and of its readout if it
    gives one, each an ``Offset``, written as the file holds one."""
    for qudit in qudits:
        qudit["frequency_hz"] = {"offset_hz": float(qudit["frequency_hz"])}
        readout = qudit.get("readout")
        if readout is not None and readout.get("frequency_hz") is not None:
            readout["frequency_hz"] = {"offset_hz": float(readout["frequency_hz"])}


def _device_data(quantum_system: QuantumSystem) -> dict[str, Any]:
    import json

    from . import __version__
    from .quantum_systems.calibration import Calibration

    data: dict[str, Any] = {"format": SYSTEM_FORMAT, "written_by": f"cudaq-pulse {__version__}"}
    spec = quantum_system._spec
    base = quantum_system._base_target
    data["name"] = spec if spec is not None else base.name
    if spec is None:
        data |= _quantum_data(base)
        # One look per file: qudits whose frequencies are offsets from a
        # shared reference are written as {"offset_hz": value}.
        if base._frame == "shared_reference":
            _write_offsets(data["qudits"])
        elif not _ONLY_HZ(map(type, map(_READOUT_HZ, filter(None, map(_READOUT, data["qudits"]))))):
            # A readout frequency that is no float: an Offset among qudits
            # with frequencies in Hz is refused, as a compile refuses it.
            from ._device import _offset_readouts_in_file

            _offset_readouts_in_file(quantum_system, base)
    data["layout"] = list(quantum_system.layout)
    if base.frames != "dressed":
        data["frames"] = base.frames
    if spec is None or quantum_system.clock_ghz != base.clock_ghz:
        data["clock_ghz"] = quantum_system.clock_ghz
    calibration = quantum_system.calibration
    if calibration is not None:
        shipped = None
        try:
            shipped = Calibration.load(
                calibration.name, measurement_run=calibration.measurement_run
            )
        except (OSError, ValueError):
            pass
        data["calibration"] = (
            calibration.name
            if shipped == calibration
            else json.loads(json.dumps(calibration._data))
        )
        data["measurement_run"] = calibration.measurement_run
    if quantum_system.control_system is not None:
        data["control_system"] = quantum_system.control_system.as_dict()
    return data


def _config_systems(
    config: Mapping[str, Any],
    source: str,
    problems: list[Problem],
    described: _Description | None = None,
) -> tuple[_Description | None, ControlSystem | None]:
    """The quantum and control systems of a quantum system config that is a system
    file, builds its quantum system or gives a control system, each ``None``
    with *problems* noted; *described* is one ``pulse.quantum_system`` built."""
    from ._device import _target

    problems.extend(_check_format(config))
    reader = _Reader(source if source.startswith("file ") else None)
    if described is not None:
        target: _Description | None = described
    elif "qudits" in config:
        target = reader.quantum_system(config)
    else:
        target = _target(config.get("name"), problems)
    control = config.get("control_system")
    if isinstance(control, Mapping):
        control = reader.control_system(control, "control_system")
    elif control is not None and not isinstance(control, ControlSystem):
        problems.append(
            Problem(
                f"must be a ControlSystem or its data, got {type(control).__name__}",
                path="control_system",
                value=control,
            )
        )
        control = None
    problems.extend(reader.problems)
    if control is not None and target is not None:
        problems.extend(_wiring_problems(control, target))
    return target, control


class _Reader:
    """Builds objects from a system file's data, collecting a ``Problem``
    for everything wrong instead of stopping at the first."""

    def __init__(self, file: str | None = None) -> None:
        self.problems: list[Problem] = []
        self.file = file

    def problem(self, path: str, message: str, value: Any = Problem.UNSET, fix: str = "") -> None:
        self.problems.append(Problem(message, path=path, value=value, fix=fix))

    def mapping(self, data: Any, path: str, known: Iterable[str]) -> Mapping[str, Any] | None:
        if not isinstance(data, Mapping):
            self.problem(path, f"must be a mapping, got {type(data).__name__}", data)
            return None
        known = tuple(known)
        for key in data:
            if key not in known:
                hint = _closest(str(key), known)
                self.problem(
                    f"{path}.{key}",
                    "is not a known key",
                    fix=f"did you mean {hint!r}?" if hint else f"the keys are {', '.join(known)}",
                )
        return data

    def component(self, data: Any, path: str, member: str) -> Any:
        """A qudit's *member* (``drive`` or ``readout``): the component of the
        kind *data* names, with its fields."""
        kinds = _component_kinds()[member]
        if not isinstance(data, Mapping):
            self.problem(path, f"must be a mapping, got {type(data).__name__}", data)
            return None
        kind = data.get("kind")
        made = kinds.get(kind) if isinstance(kind, str) else None
        if made is None:
            self.problem(
                f"{path}.kind",
                f"is not a {member} kind" if "kind" in data else "is missing",
                kind if "kind" in data else Problem.UNSET,
                fix=f"the kinds are {', '.join(kinds)}",
            )
            return None
        known = made._fields
        self.mapping(data, path, ("kind", *known))
        return self.build(path, made, **{k: data[k] for k in known if k in data})

    def build[T](self, path: str, make: Callable[..., T], **kwargs: Any) -> T | None:
        try:
            return make(**kwargs)
        except (TypeError, ValueError) as error:
            self.problem(path, str(error))
            return None

    def quantum_system(self, system: Mapping[str, Any]) -> _Description | None:
        """The quantum system a system file builds, from its top-level keys."""
        from .quantum_systems.base import Coupling, PhysicalQudit, ResidualZZ, _Description

        qudits: dict[int, Any] = {}
        # Each qudit's position in the file's list, by its index.
        positions: dict[int, int] = {}
        given = system.get("qudits")
        if not isinstance(given, Sequence) or isinstance(given, (str, bytes)):
            self.problem("qudits", f"must be a list of qudits, got {type(given).__name__}", given)
            given = ()
        for i, item in enumerate(given):
            where = f"qudits[{i}]"
            entry = self.mapping(item, where, (*_QUDIT_FIELDS, *_QUDIT_EXTRAS))
            if entry is not None:
                entry = {
                    key: self.component(value, f"{where}.{key}", key)
                    if key in ("drive", "readout") and value is not None
                    else self.calibrations(value, f"{where}.{key}", "qudit")
                    if key == "calibrations"
                    else value
                    for key, value in entry.items()
                }
                qudit = self.build(where, PhysicalQudit, **entry)
                if qudit is not None:
                    qudits[qudit.index] = qudit
                    positions[qudit.index] = i
        # The entries read, each with its index in the file.
        entries: dict[str, list[tuple[int, Any]]] = {"couplings": [], "residual_zz": []}
        for i, item in enumerate(system.get("couplings", ())):
            where = f"couplings[{i}]"
            entry = self.mapping(item, where, _COUPLING_FIELDS)
            if entry is None or self.missing(entry, where, _COUPLING_REQUIRED):
                continue
            given = {k: entry[k] for k in ("qudits", "strength_hz")}
            if "calibrations" in entry:
                given["calibrations"] = self.calibrations(
                    entry["calibrations"], f"{where}.calibrations", "coupling"
                )
            coupling = self.build(where, Coupling, **given)
            if coupling is not None:
                entries["couplings"].append((i, coupling))
        for i, item in enumerate(system.get("residual_zz", ())):
            where = f"residual_zz[{i}]"
            entry = self.mapping(item, where, _RESIDUAL_ZZ_FIELDS)
            if entry is None or self.missing(entry, where, _RESIDUAL_ZZ_REQUIRED):
                continue
            given = {k: v for k, v in entry.items() if k in _RESIDUAL_ZZ_FIELDS}
            residual = self.build(where, ResidualZZ, **given)
            if residual is not None:
                entries["residual_zz"].append((i, residual))
        if "name" not in system:
            self.problem("name", "is missing", fix="give the quantum system's name")
            return None
        rest = {k: system[k] for k in ("attribution", "clock_ghz") if k in system}
        if "model" in system:
            rest["model"] = self.model(system["model"], "model")
        try:
            return _Description(
                name=system["name"],
                qudits=qudits,
                couplings=[c for _, c in entries["couplings"]],
                residual_zz=[x for _, x in entries["residual_zz"]],
                **rest,
            )
        except QuantumSystemDefinitionError as error:
            # The path in the file of an entry's problem, which the system
            # gives as couplings[3]... by its own index: entries the file
            # gives that could not be read are not in the system. A qudit's
            # problem is at qudits[i] by the qudit's index, so it is found
            # by where that qudit is in the file's list.
            def in_file(entry_path: str) -> str:
                kind, _, rest = entry_path.partition("[")
                index, _, tail = rest.partition("]")
                if kind == "qudits":
                    return f"qudits[{positions[int(index)]}]{tail}"
                return f"{kind}[{entries[kind][int(index)][0]}]{tail}"

            self.problems.extend(
                Problem(
                    p.message,
                    path=in_file(p.path),
                    value=p.value,
                    fix=p.fix,
                )
                for p in error.problems
            )
        except (TypeError, ValueError) as error:
            self.problem("name", str(error))
        return None

    def calibrations(self, data: Any, path: str, owner: str) -> dict[Any, Any]:
        """The calibrations of a qudit or coupling (*owner*), by gate (or
        ``(gate, transition)``), from their entries in the file."""
        from .quantum_systems.base import _CALIBRATION_TYPES, _COUPLING_GATES, _QUDIT_GATES

        allowed = _QUDIT_GATES if owner == "qudit" else _COUPLING_GATES
        found: dict[Any, Any] = {}
        if not isinstance(data, list):
            self.problem(
                path,
                f"must be a list of calibrations, got {type(data).__name__}",
                data,
                fix=f'give each as {{"gate": "{allowed[0]}", ...}}',
            )
            return found
        for j, item in enumerate(data):
            where = f"{path}[{j}]"
            if not isinstance(item, Mapping):
                self.problem(where, f"must be a mapping, got {type(item).__name__}", item)
                continue
            gate = item.get("gate")
            if not (isinstance(gate, str) and gate in allowed):
                self.problem(
                    f"{where}.gate",
                    f"is not a gate of a {owner}" if "gate" in item else "is missing",
                    gate if "gate" in item else Problem.UNSET,
                    fix=f"the gates of a {owner} are {', '.join(allowed)}",
                )
                continue
            made = _CALIBRATION_TYPES[gate]
            self.mapping(item, where, ("gate", "transition", *made._fields))
            pulse = self.build(where, made, **{k: item[k] for k in made._fields if k in item})
            if pulse is None:
                continue
            transition = item.get("transition")
            if transition is None:
                found[gate] = pulse
            else:
                found[(gate, tuple(transition) if isinstance(transition, list) else transition)] = (
                    pulse
                )
        return found

    def missing(self, entry: Mapping[str, Any], path: str, required: Mapping[str, str]) -> bool:
        """Whether *entry* lacks a key of *required*, each noted as a problem
        with the fix *required* gives for it."""
        absent = [key for key in required if key not in entry]
        for key in absent:
            self.problem(f"{path}.{key}", "is missing", fix=required[key])
        return bool(absent)

    def model(self, data: Any, path: str) -> Any:
        if data is None:
            return None
        entry = self.mapping(data, path, ("kind", "levels"))
        if entry is None:
            return None
        kinds = _model_kinds()
        cls = kinds.get(str(entry.get("kind")))
        if cls is None:
            self.problem(f"{path}.kind", f"must be one of {', '.join(kinds)}", entry.get("kind"))
            return None
        levels = entry.get("levels")
        if isinstance(levels, Mapping):
            try:
                levels = {int(q): n for q, n in levels.items()}
            except ValueError:
                self.problem(f"{path}.levels", "keys must be qudit indices", levels)
                return None
        return self.build(path, cls, levels=levels)

    def controller(self, data: Any, path: str) -> Controller | None:
        known = [f.name for f in fields(Controller)]
        entry = self.mapping(data, path, known)
        if entry is None:
            return None
        given = {k: v for k, v in entry.items() if k in known}
        for name in ("detectors", "peer_latency_cycles"):
            value = given.get(name)
            if isinstance(value, Mapping):
                try:
                    given[name] = {int(k): v for k, v in value.items()}
                except ValueError:
                    self.problem(f"{path}.{name}", "is keyed by integers", value)
                    return None
        for name, cls in (("drive_lines", DriveLine), ("readout_lines", ReadoutLine)):
            lines = given.get(name)
            if isinstance(lines, list):
                read = [self.line(item, f"{path}.{name}[{i}]", cls) for i, item in enumerate(lines)]
                if None in read:
                    return None
                given[name] = read
        controller: Controller | None = self.build(path, Controller, **given)
        if controller is not None:
            _notice_version(controller, path if self.file is None else f"{self.file}: {path}")
        return controller

    def line(self, data: Any, path: str, cls: type[_Line]) -> Any:
        """A system file's line: a carrier index, ``{"qudit": 5,
        "delay_ns": 4.0}``, or a shared line ``{"qudits": [0, 1, 2, 3],
        "max_concurrent_tones": 4, "delay_ns": 12.0}``."""
        if not isinstance(data, Mapping):
            return data
        if "qudits" in data:
            known = ("qudits", "max_concurrent_tones", "delay_ns", "broadcast", "frequency_hz")
            entry = self.mapping(data, path, known)
            if entry is None:
                return None
            return self.build(path, cls, **{k: v for k, v in entry.items() if k in known})
        entry = self.mapping(data, path, ("qudit", "delay_ns"))
        if entry is None or self.missing(entry, path, {"qudit": "give its qudit, as 5"}):
            return None
        given = dict(entry)
        return self.build(path, cls, qudits=given.pop("qudit"), **given)

    def control_system(self, data: Any, path: str) -> ControlSystem | None:
        system = self.mapping(data, path, _CONTROL_KEYS)
        if system is None:
            return None
        read = [
            self.controller(item, f"{path}.controllers[{i}]")
            for i, item in enumerate(system.get("controllers", ()))
        ]
        controllers = [c for c in read if c is not None]
        if len(controllers) < len(read):
            return None
        links: dict[str, Link] = {}
        for name, item in dict(system.get("links", {})).items():
            entry = self.mapping(item, f"{path}.links.{name}", _LINK_KEYS)
            if entry is None:
                continue
            given = dict(entry)
            reached = given.pop("controllers", None)
            if reached is not None:
                picked = []
                for i, index in enumerate(reached):
                    if isinstance(index, int) and 0 <= index < len(controllers):
                        picked.append(controllers[index])
                    else:
                        self.problem(
                            f"{path}.links.{name}.controllers[{i}]",
                            "is not the index of a controller of the system",
                            index,
                        )
                given["controllers"] = picked
            link = self.build(f"{path}.links.{name}", Link, **given)
            if link is not None:
                links[name] = link
        rest = {k: system[k] for k in _CONTROL_KEYS[2:] if k in system}
        try:
            return ControlSystem(controllers=controllers, links=links, **rest)
        except QuantumSystemDefinitionError as error:
            self.problems.extend(
                Problem(p.message, path=f"{path}.{p.path}", value=p.value, fix=p.fix)
                for p in error.problems
            )
        except (TypeError, ValueError) as error:
            self.problem(path, str(error))
        return None


def _check_format(config: Mapping[str, Any]) -> list[Problem]:
    """The problem with the config's ``format``, if it is not one this
    version reads."""
    given = config.get("format")
    if given is None:
        return []
    kind, _, major = str(given).rpartition("/")
    current = SYSTEM_FORMAT.rpartition("/")
    if kind != current[0] or not major.isdigit():
        return [
            Problem(
                "is not a system file format",
                path="format",
                value=given,
                fix=f"this cudaq-pulse reads {SYSTEM_FORMAT}",
            )
        ]
    if int(major) > int(current[2]):
        return [
            Problem(
                f"is newer than this cudaq-pulse reads ({SYSTEM_FORMAT})",
                path="format",
                value=given,
                fix="upgrade cudaq-pulse to load it",
            )
        ]
    return []
