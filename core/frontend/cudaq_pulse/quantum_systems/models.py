# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation models: how the simulator treats the qudits of a system.

A quantum system names its model (``QuantumSystem.model``); the model, never a
label, decides the physics. Each model class declares the levels of its
qudits as the class attribute ``levels``, and an instance overrides them per
qudit:

    TransmonModel(levels={0: 4, 2: 2})

keyed by the system's qudit index, the key of ``QuantumSystem.physical_qudits``
and of ``simulate(levels=...)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from .._errors import InvalidTypeError, InvalidValueError, UnsupportedError

if TYPE_CHECKING:
    from ..runtime.simulate import SimulationResult
    from .base import PhysicalQudit

__all__ = ["SimulationModel", "TransmonModel"]


class SimulationModel:
    """A model of a system's qudits, with ``levels`` levels each.

    The base model describes qudits by their 0-1 frequency alone, as evenly
    spaced levels, and has no simulator: a system with it compiles and
    exports, and ``simulate`` refuses it. Subclasses that simulate override
    ``_simulate``.

    A model is read-only: its levels by qudit are fixed when it is made.
    This class and ``TransmonModel`` are immutable, as built-in types are
    (``import cudaq_pulse`` freezes them), so what they give a qudit never
    changes and a step resolved for it stays right
    (``runtime.simulate._STEPS``); subclass them to change it.
    """

    #: Levels per qudit, unless the instance sets a qudit's own.
    levels: ClassVar[int] = 2
    #: The transition, as ``(lower, upper)`` levels, that a line's own tone
    #: (``line.tone``, at the qudit's ``frequency_hz``) drives.
    default_transition: ClassVar[tuple[int, int]] = (0, 1)

    __slots__ = ("_levels",)
    _levels: Mapping[int, int]

    def __init__(self, levels: dict[int, int] | None = None) -> None:
        own: dict[int, int] = {}
        if levels is not None:
            if not isinstance(levels, Mapping):
                raise InvalidTypeError(
                    f"levels must be a dict of levels by qudit index, as in "
                    f"{type(self).__name__}(levels={{0: 4}}); got {type(levels).__name__}"
                )
            for qudit, n in levels.items():
                if qudit.__class__ is not int or qudit < 0:
                    raise InvalidValueError(
                        f"levels keys are qudit indices (ints from 0), got {qudit!r}"
                    )
                if n.__class__ is not int or n < 2:
                    raise InvalidValueError(
                        f"qudit {qudit} needs 2 or more levels (an int), got {n!r}"
                    )
            # Sorted only once every key is an int, since sorting keys of mixed
            # types fails; __hash__ and __repr__ read the levels in this order.
            own = dict(sorted(levels.items()))
        object.__setattr__(self, "_levels", MappingProxyType(own))

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_levels":
            raise AttributeError(f"{type(self).__name__} is read-only")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_levels":
            raise AttributeError(f"{type(self).__name__} is read-only")
        super().__delattr__(name)

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (dict(self._levels) or None,)

    def levels_of(self, qudit: int) -> int:
        """The levels of qudit *qudit*: its own, else the model's."""
        return self._levels.get(qudit, self.levels)

    def energies_hz(self, qudit: PhysicalQudit, levels: int) -> list[float]:
        """The energies of *qudit*'s first *levels* levels, in Hz from its ground
        level: evenly spaced by its 0-1 frequency."""
        return [k * qudit._frame_hz for k in range(levels)]

    def transitions_hz(self, qudit: PhysicalQudit, levels: int) -> dict[tuple[int, int], float]:
        """The frequency in Hz of each transition between *qudit*'s first
        *levels* levels, by ``(lower, upper)`` level, ordered by level."""
        energies = self.energies_hz(qudit, levels)
        return {
            (lower, upper): energies[upper] - energies[lower]
            for lower in range(levels)
            for upper in range(lower + 1, levels)
        }

    def _simulate(self, *args: Any, **options: Any) -> SimulationResult:
        raise UnsupportedError(
            f"{type(self).__name__} has no simulator; this system compiles and "
            "exports, and a model that simulates it is a SimulationModel subclass"
        )

    def __eq__(self, other: object) -> bool:
        return other.__class__ is self.__class__ and other._levels == self._levels

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(self._levels.items())))

    def __repr__(self) -> str:
        own = f"levels={dict(self._levels)!r}" if self._levels else ""
        return f"{type(self).__name__}({own})"


class TransmonModel(SimulationModel):
    """Transmons: anharmonic oscillators, each level ``k`` at
    ``k f01 + k (k - 1) / 2 alpha`` with ``alpha`` the qubit's anharmonicity,
    coupled by exchange."""

    levels: ClassVar[int] = 3

    __slots__ = ()

    def energies_hz(self, qudit: PhysicalQudit, levels: int) -> list[float]:
        f01, alpha = qudit._frame_hz, qudit.anharmonicity_hz
        if alpha is None:
            if levels > 2:
                raise InvalidValueError(
                    f"qudit {qudit.index} gives no anharmonicity_hz, which a transmon of "
                    f"{levels} levels needs: give PhysicalQudit(..., anharmonicity_hz=...), "
                    f"or simulate it with two levels (TransmonModel(levels={{{qudit.index}: 2}}))"
                )
            alpha = 0.0
        return [k * f01 + k * (k - 1) / 2 * alpha for k in range(levels)]

    def _simulate(self, *args: Any, **options: Any) -> SimulationResult:
        from ..runtime.simulate import _simulate_compiled

        return _simulate_compiled(*args, **options)
