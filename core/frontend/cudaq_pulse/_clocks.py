# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clocks: the program's, each controller's, and the grids between them.

A kernel counts time in cycles of the quantum system's *program clock*,
``QuantumSystem.clock_ghz``. A controller may run its own clock: at ``f_k`` against
the program's ``f_p`` it counts ``r = f_k / f_p`` of its cycles per program
cycle, and may start operations only every ``cycles_per_timestep`` of them.
Its lines' timestep (their ``grid``) is the fewest whole program cycles
that are a whole number of those steps: ``b`` for ``r = a / b`` in lowest
terms and one cycle per step. A line's operations start and last whole
timesteps; a sync of lines meets on the least common multiple of theirs. A
controller's program counts in its own cycles: the compiler converts once,
when it exports.

Without a shared reference clock, two controllers' starts are each known to
within one of their own cycles, which budgets across controllers add.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ._errors import InvalidValueError, Problem
from .ops import RoundDirection, round_duration

if TYPE_CHECKING:
    from fractions import Fraction

    from ._system import Controller, ControlSystem

__all__ = ["RoundDirection", "round_duration"]

# Clocks are given in GHz as floats; their ratios are fractions with
# denominators up to this.
_MAX_DENOMINATOR = 10_000
_TOLERANCE = 1e-9


def _definition_error(message: str) -> Exception:
    """The error for a quantum system whose clocks do not fit together."""
    return ValueError(message)


def _ghz(value: float) -> str:
    return f"{value:g} GHz"


def clock_period_fs(clock_ghz: float) -> str:
    """The period in fs of a *clock_ghz* program clock as
    ``#pulse.control_system`` writes it: the fraction of smallest
    denominator whose clock is exactly *clock_ghz* (the first convergent of
    its continued fraction that is), ``1250000/3`` for 2.4 GHz, ``500000``
    for 2 GHz. The packed builder finds the same one."""
    found = _PERIODS.get(clock_ghz)
    if found is None:
        x, y = clock_ghz.as_integer_ratio()
        p0, p1, q0, q1 = 0, 1, 1, 0
        while y:
            term = x // y
            p0, p1, q0, q1 = p1, term * p1 + p0, q1, term * q1 + q0
            if float(p1) / float(q1) == clock_ghz:
                break
            x, y = y, x - term * y
        numerator, denominator = 1_000_000 * q1, p1
        common = math.gcd(numerator, denominator)
        numerator, denominator = numerator // common, denominator // common
        found = _PERIODS[clock_ghz] = (
            str(numerator) if denominator == 1 else f"{numerator}/{denominator}"
        )
    return found


_PERIODS: dict[float, str] = {}


def clock_ratio(clock_ghz: float, program_ghz: float) -> Fraction:
    """``clock_ghz / program_ghz`` as a fraction; an error if it is not one."""
    # Imported here: a quantum system whose clock is only declared needs no fraction.
    from fractions import Fraction

    ratio = Fraction(clock_ghz / program_ghz).limit_denominator(_MAX_DENOMINATOR)
    if abs(float(ratio) * program_ghz - clock_ghz) > _TOLERANCE * clock_ghz:
        raise _definition_error(
            f"a controller's {_ghz(clock_ghz)} clock has no whole-cycle ratio to the "
            f"{_ghz(program_ghz)} program clock; give the quantum system a 'clock_ghz' both "
            "divide into"
        )
    return ratio


def _fraction(ghz: float) -> Fraction:
    from fractions import Fraction

    return Fraction(ghz).limit_denominator(1_000_000)


def _lcm(clocks: Sequence[float]) -> float:
    """The slowest clock every one of *clocks* divides into."""
    values = [_fraction(c) for c in clocks]
    numerator = functools.reduce(math.lcm, (v.numerator for v in values))
    denominator = functools.reduce(math.gcd, (v.denominator for v in values))
    return numerator / denominator


def program_clock_ghz(
    declared: float | None, controller_clocks: Mapping[str, float], default: float
) -> float:
    """The program clock of a quantum system: *declared* (its config's ``clock_ghz``)
    if given, else the fastest of its controllers' clocks when every other
    one divides it, else *default* when it has no controllers.

    Raises ``QuantumSystemDefinitionError`` when the controllers' clocks do not
    divide evenly into one and none is declared, listing the clocks that
    work; it never picks one.
    """
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, (int, float)):
            raise _definition_error(
                f"the quantum system's clock_ghz must be a number, got {declared!r}"
            )
        if not declared > 0 or not math.isfinite(declared):
            raise _definition_error(
                f"the quantum system's clock_ghz must be positive, got {declared}"
            )
        for clock in controller_clocks.values():
            clock_ratio(clock, declared)
        return float(declared)
    if not controller_clocks:
        return float(default)
    fastest = max(controller_clocks.values())
    if all(clock_ratio(c, fastest).numerator == 1 for c in controller_clocks.values()):
        return float(fastest)
    common = _lcm(list(controller_clocks.values()))
    options = [f"{_ghz(common)}, which every controller's clock divides"]
    for name, clock in sorted(controller_clocks.items(), key=lambda item: -item[1]):
        grids = ", ".join(
            f"{other} every {clock_ratio(c, clock).denominator}"
            for other, c in sorted(controller_clocks.items())
            if clock_ratio(c, clock).denominator > 1
        )
        options.append(f"{_ghz(clock)} ({name}'s; then {grids} program cycles)")
    clocks = ", ".join(f"{name} at {_ghz(c)}" for name, c in sorted(controller_clocks.items()))
    raise _definition_error(
        f"the controllers' clocks ({clocks}) do not divide evenly into one, so the "
        "quantum system needs a program clock: set 'clock_ghz' in its config to one of "
        + "; ".join(options)
    )


@dataclass(frozen=True, slots=True)
class ControllerClock:
    """One controller's clock against the program clock."""

    name: str
    clock_ghz: float
    # Own cycles per program cycle.
    ratio: Fraction
    # Its lines' timestep: program cycles between the instants it can start
    # an operation.
    grid: int = 0

    def __post_init__(self) -> None:
        if not self.grid:
            object.__setattr__(self, "grid", self.ratio.denominator)

    def own(self, cycles: int) -> int:
        """*cycles* program cycles in its own, rounded up."""
        return -((-cycles * self.ratio.numerator) // self.ratio.denominator)


@dataclass(frozen=True, slots=True)
class ClockPlan:
    """The controllers of a compiled program and the lines each one plays."""

    program_ghz: float
    controllers: tuple[ControllerClock, ...]
    reference_clock_ghz: float | None = None
    # (role, qudit) -> index into controllers; absent: the first.
    owners: Mapping[tuple[str, int], int] | None = None
    # The scheduler's readout latency in program cycles, when the
    # controllers count theirs in their own (None: the control system's).
    readout_latency_cycles: int | None = None

    def __reduce__(self) -> tuple[Any, ...]:
        owners = None if self.owners is None else dict(self.owners)
        return (
            _rebuilt_plan,
            (
                self.program_ghz,
                self.controllers,
                self.reference_clock_ghz,
                owners,
                self.readout_latency_cycles,
            ),
        )

    def controller_of(self, role: str, qudit: int) -> ControllerClock:
        if self.owners is None:
            return self.controllers[0]
        return self.controllers[self.owners.get((role, qudit), 0)]

    def grid_of(self, role: str, qudit: int) -> int:
        """The grid of a line; with owners, one they do not name (a runtime
        qudit's, which any controller may play) is on every grid."""
        if self.owners is None:
            return self.controllers[0].grid
        owner = self.owners.get((role, qudit))
        return self.lcm if owner is None else self.controllers[owner].grid

    @property
    def lcm(self) -> int:
        """Where every line can meet."""
        return functools.reduce(math.lcm, (c.grid for c in self.controllers), 1)

    def start_uncertainty_ns(
        self, skew_cycles: Mapping[str, int] | None = None
    ) -> dict[str, float]:
        """How well each controller's start is known, in ns: its skew, plus one
        of its own cycles when no reference clock is shared."""
        extra = 0 if self.reference_clock_ghz is not None else 1
        skew = skew_cycles or {}
        return {c.name: (skew.get(c.name, 0) + extra) / c.clock_ghz for c in self.controllers}

    def budget_uncertainty_ns(
        self, a: str, b: str, skew_cycles: Mapping[str, int] | None = None
    ) -> float:
        """The start uncertainty a budget between controllers *a* and *b* adds."""
        if a == b:
            return 0.0
        each = self.start_uncertainty_ns(skew_cycles)
        return each[a] + each[b]


def _rebuilt_plan(
    program_ghz: float,
    controllers: tuple[ControllerClock, ...],
    reference_clock_ghz: float | None,
    owners: dict[tuple[str, int], int] | None,
    readout_latency_cycles: int | None,
) -> ClockPlan:
    return ClockPlan(
        program_ghz,
        controllers,
        reference_clock_ghz,
        None if owners is None else MappingProxyType(owners),
        readout_latency_cycles,
    )


def _check_reference(reference_ghz: float, clocks: Mapping[str, float]) -> None:
    for name, clock in sorted(clocks.items()):
        multiple = clock / reference_ghz
        if abs(multiple - round(multiple)) > _TOLERANCE * multiple or round(multiple) < 1:
            raise _definition_error(
                f"the {name}'s {_ghz(clock)} clock is not a whole multiple of the "
                f"{_ghz(reference_ghz)} reference clock"
            )


def plan(
    limits: Controller | None, program_ghz: float, reference_clock_ghz: float | None = None
) -> ClockPlan | None:
    """The clocks of a program for *limits*' controller, or ``None`` when every
    controller runs at the program clock and no reference clock is given."""
    if limits is None:
        own, steps, samples = None, 1, None
    else:
        own, steps, samples = limits.clock_ghz, limits.cycles_per_timestep, limits.sample_rate_ghz
    if (
        reference_clock_ghz is None
        and steps == 1
        and (own is None or own == program_ghz)
        and (samples is None or _sample_step(samples, program_ghz) == 1)
    ):
        return None
    key = (own, program_ghz, reference_clock_ghz, steps, samples)
    found = _PLANS.get(key)
    if found is None:
        found = _PLANS[key] = _plan(own, program_ghz, reference_clock_ghz, steps, samples)
    return found


_PLANS: dict[tuple[float | None, float, float | None, int, float | None], ClockPlan] = {}


def _sample_step(sample_rate_ghz: float, program_ghz: float) -> int:
    """The program cycles of the shortest whole number of samples."""
    try:
        return made_for(sample_rate_ghz, program_ghz)[1]
    except ValueError:
        raise _definition_error(
            f"a controller's {_ghz(sample_rate_ghz)} sample rate has no whole-cycle ratio "
            f"to the {_ghz(program_ghz)} program clock"
        ) from None


def timestep(ratio: Fraction, cycles_per_timestep: int) -> int:
    """A timestep in program cycles: the fewest whole program cycles that
    are whole steps of *cycles_per_timestep* own cycles, at *ratio* own
    cycles per program cycle."""
    own = ratio.denominator * cycles_per_timestep
    return own // math.gcd(ratio.numerator, own)


def _plan(
    own: float | None,
    program_ghz: float,
    reference: float | None,
    steps: int,
    samples: float | None,
) -> ClockPlan:
    clock = program_ghz if own is None else own
    if reference is not None:
        _check_reference(reference, {"controller": clock})
    ratio = clock_ratio(clock, program_ghz)
    grid = timestep(ratio, steps)
    if samples is not None:
        grid = math.lcm(grid, _sample_step(samples, program_ghz))
    controller = ControllerClock("controller", clock, ratio, grid)
    return ClockPlan(program_ghz, (controller,), reference)


def controllers_plan(
    controllers: Sequence[Controller],
    names: Sequence[str],
    owners: Mapping[tuple[str, int], int],
    program_ghz: float,
    reference_clock_ghz: float | None = None,
) -> ClockPlan | None:
    """The clocks of a program on several controllers, each named by
    *names*, and the controller that plays each line (*owners*, (role,
    qudit) -> index); ``None`` when every one runs at the program clock in
    steps of one cycle and no reference clock is given."""
    plans = [plan(c, program_ghz, reference_clock_ghz) for c in controllers]
    if all(p is None for p in plans):
        return None
    from fractions import Fraction

    clocks = tuple(
        ControllerClock(name, program_ghz, Fraction(1))
        if p is None
        else dataclasses.replace(p.controllers[0], name=name)
        for name, p in zip(names, plans, strict=True)
    )
    return ClockPlan(program_ghz, clocks, reference_clock_ghz, MappingProxyType(dict(owners)))


def sample_rate(limits: Controller, program_ghz: float) -> tuple[int, int] | None:
    """The samples a line of *limits*' controller plays per program cycle, as
    ``(n, d)``: ``n`` every ``d`` cycles; ``None`` when it is one."""
    if limits.sample_rate_ghz is not None:
        n, d = made_for(limits.sample_rate_ghz, program_ghz)
        return None if n == d else (n, d)
    if limits.clock_ghz is None or limits.clock_ghz == program_ghz:
        return None
    rate = clock_ratio(limits.clock_ghz, program_ghz)
    return None if rate == 1 else (rate.numerator, rate.denominator)


def made_for(sample_rate_ghz: Any, program_ghz: float) -> tuple[int, int]:
    """The ``(n, d)`` of samples made at *sample_rate_ghz*: ``n`` every ``d``
    program cycles. Raises ``ValueError`` when that is not a ratio of whole
    numbers."""
    if (
        isinstance(sample_rate_ghz, bool)
        or not isinstance(sample_rate_ghz, (int, float))
        or not math.isfinite(sample_rate_ghz)
        or not sample_rate_ghz > 0
    ):
        raise InvalidValueError(
            f"sample_rate_ghz must be a positive number, got {sample_rate_ghz!r}"
        )
    from fractions import Fraction

    rate = Fraction(sample_rate_ghz / program_ghz).limit_denominator(_MAX_DENOMINATOR)
    if abs(float(rate) * program_ghz - sample_rate_ghz) > _TOLERANCE * sample_rate_ghz:
        raise InvalidValueError(
            f"sample_rate_ghz={sample_rate_ghz:g} has no whole-cycle ratio to the "
            f"{_ghz(program_ghz)} program clock"
        )
    return rate.numerator, rate.denominator


def line_delays(
    delays: Mapping[tuple[str, int], float],
    limits: Controller | None,
    plan: ClockPlan | None,
    program_ghz: float,
    controllers: Sequence[Controller] | None = None,
    owners: Mapping[tuple[str, int], int] | None = None,
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], float]]:
    """How many program cycles each delayed line moves, whole steps of its
    grid (none for a line that does not move), and each line's fine delay
    in ns, the rest, both by (role, qudit). Raises ``ValueError`` for a rest
    its controller can neither add nor tolerate. With several
    *controllers*, each line's limits are those of the one *owners*
    ((role, qudit) -> index) names."""
    fine_max = 0.0 if limits is None else limits.max_fine_delay_ns
    tolerance = 0.0 if limits is None else limits.alignment_tolerance_cycles
    shifts: dict[tuple[str, int], int] = {}
    fine: dict[tuple[str, int], float] = {}
    for (role, qudit), ns in delays.items():
        if controllers is not None and owners is not None:
            owner = owners.get((role, qudit))
            own = None if owner is None else controllers[owner]
            fine_max = 0.0 if own is None else own.max_fine_delay_ns
            tolerance = 0.0 if own is None else own.alignment_tolerance_cycles
        controller = None if plan is None else plan.controller_of(role, qudit)
        grid = 1 if controller is None else controller.grid
        own_ghz = program_ghz if controller is None else controller.clock_ghz
        step_ns = grid / program_ghz
        steps = ns / step_ns
        whole = math.floor(steps + _TOLERANCE) if fine_max > 0 else round(steps)
        rest = ns - whole * step_ns
        if abs(rest) < _TOLERANCE:
            rest = 0.0
        if fine_max > 0 and rest > fine_max + _TOLERANCE:
            raise InvalidValueError(
                f"the {role} line of qudit {qudit} is delayed {ns:g} ns (its delay_ns), "
                f"{rest:g} ns past a whole number of its {step_ns:g} ns timesteps, "
                f"and the controller adds at most {fine_max:g} ns of fine delay; raise its "
                f"max_fine_delay_ns to {rest:g}, or calibrate the delay to "
                f"{whole * step_ns:g} or {(whole + 1) * step_ns:g} ns"
            )
        if fine_max == 0 and abs(rest) * own_ghz > tolerance + _TOLERANCE:
            low = math.floor(steps + _TOLERANCE) * step_ns
            raise InvalidValueError(
                f"the {role} line of qudit {qudit} is delayed {ns:g} ns (its delay_ns), "
                f"{abs(rest):g} ns ({abs(rest) * own_ghz:g} of its controller's cycles) from "
                f"a whole number of its {step_ns:g} ns timesteps, and the controller has no fine "
                f"delay; give it max_fine_delay_ns, raise its alignment_tolerance_cycles to "
                f"{abs(rest) * own_ghz:g}, or calibrate the delay to {low:g} or "
                f"{low + step_ns:g} ns"
            )
        if abs(whole * grid) > 2**60:
            raise InvalidValueError(
                f"the {role} line of qudit {qudit} is delayed {ns:g} ns (its delay_ns), more "
                f"than a program lasts: at most 2**60 cycles, {2**60 / program_ghz:.3g} ns"
            )
        fine[(role, qudit)] = rest if fine_max > 0 else 0.0
        if whole:
            shifts[(role, qudit)] = whole * grid
    return shifts, fine


def off_grid(
    plan: ClockPlan,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[dict[str, Any]],
    convert: bool = False,
    waveforms: bool = True,
    run_time: bool = False,
) -> list[str]:
    """A violation for each line event that does not start and last whole
    cycles of its controller; with *convert*, the events' times are then in
    their controller's own cycles, in place (and with *waveforms*, their
    waveforms' time parameters scaled), in the same pass. With *run_time*
    (the program computes values at run time), what the events compute at
    run time is converted too (``own_run_time``)."""
    found = []
    # The events of one op share their waveform's parameters.
    scaled: set[int] = set()
    single = plan.controllers[0] if plan.owners is None else None
    controller = plan.controllers[0]
    a, b = controller.ratio.numerator, controller.ratio.denominator
    grid = controller.grid
    for event in events:
        # A submit's or an answer's "line" is its link.
        line = event["line"] if event["kind"] in _GRIDDED else None
        if single is None:
            on = event["line"] if event["kind"] == "frame_update" else line
            controller = (
                plan.controllers[0]
                if on is None
                else plan.controller_of(lines[on]["role"], lines[on]["qudit"])
            )
            a, b = controller.ratio.numerator, controller.ratio.denominator
            grid = controller.grid
        start = event["start_cycles"]
        duration = event["duration_cycles"]
        if (
            grid > 1
            and line is not None
            and start is not None
            and duration is not None
            and (start % grid or duration % grid)
        ):
            found.append(_off_grid(controller, lines[line], event["kind"], start, duration))
        if convert and run_time and line is not None and (a != b or grid > 1):
            found.extend(own_run_time(controller, lines[line], event, waveforms))
        if convert and a != b:
            if start is not None:
                event["start_cycles"] = -((-start * a) // b)
            if duration is not None:
                event["duration_cycles"] = -((-duration * a) // b)
            end = event["end_cycles"]
            if end is not None:
                event["end_cycles"] = -((-end * a) // b)
            for update in event.get("frame_updates", ()):
                made = update["start_cycles"]
                if made is not None:
                    update["start_cycles"] = -((-made * a) // b)
            tree = event.get("waveform") if waveforms else None
            if tree is not None:
                _scale(tree, a / b, scaled)
    return found


_GRIDDED = frozenset({"drive", "readout", "wait"})


def own_run_time(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: dict[str, Any],
    waveforms: bool = True,
) -> list[str]:
    """Convert what a line event computes at run time to *controller*'s
    cycles, in place, by the rule for its fixed times: a variable wait's
    length, clamped to its bounds as the run clamps it, and its bounds; and
    (with *waveforms*) its waveform's operands computed at run time. A
    length becomes a ``controller_cycles`` expression, which the controller
    fails at run time when the length is off the line's timesteps. Returns
    the violations: a length that is off them for a value it can be known
    to take now, or an operand the export cannot tell is a time."""
    a, b = controller.ratio.numerator, controller.ratio.denominator
    grid = controller.grid
    found: list[str] = []
    length = event.get("length")
    if length is not None:
        # The terms the length names stay those of the whole expression.
        length, terms = _without_terms(length)
        low = event.get("min_cycles")
        high = event.get("max_cycles")
        if low is not None:
            length = {"op": "max", "type": "int", "of": [length, {"const": low}]}
            event["min_cycles"] = low * a // b
        if high is not None:
            length = {"op": "min", "type": "int", "of": [length, {"const": high}]}
            event["max_cycles"] = -((-high * a) // b)
        event["length"] = _with_terms(_own_length(length, a, b, grid), terms)
        found.extend(_decided(controller, line, event, "length", _with_terms(length, terms)))
    tree = event.get("waveform")
    if tree is not None:
        _own_operands(controller, line, event, tree, a, b, waveforms, found)
    return found


def _without_terms(expression: Any) -> tuple[Any, Any]:
    """*expression* without the ``terms`` it names, and those terms (None
    for none): what an expression is made an operand of."""
    if isinstance(expression, dict) and "terms" in expression:
        return {k: v for k, v in expression.items() if k != "terms"}, expression["terms"]
    return expression, None


def _with_terms(expression: dict[str, Any], terms: Any) -> dict[str, Any]:
    """*expression* naming *terms* (see ``_without_terms``)."""
    return expression if terms is None else {"terms": terms, **expression}


def _own_length(cycles: Any, a: int, b: int, grid: int) -> dict[str, Any]:
    return {
        "op": "controller_cycles",
        "type": "int",
        "of": [cycles],
        "num": a,
        "den": b,
        "timestep": grid,
    }


def _own_operands(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: Mapping[str, Any],
    node: dict[str, Any],
    a: int,
    b: int,
    convert: bool,
    found: list[str],
) -> None:
    computed = node.get("runtime")
    if computed:
        shape = node["shape"]
        times = _TIMES_OF.get(shape, ())
        named = "params" in node
        for key, value in computed.items():
            if not named:
                if a != b:
                    found.append(_unconvertible(controller, line, event, node, key))
                continue
            if key == "duration":
                found.extend(_decided(controller, line, event, "waveform's duration", value))
                if convert:
                    value, terms = _without_terms(value)
                    computed[key] = _with_terms(_own_length(value, a, b, controller.grid), terms)
            elif key in times and a != b and convert:
                factor = a / b
                value, terms = _without_terms(value)
                computed[key] = _with_terms(
                    {
                        "op": "mul",
                        "type": "float",
                        "of": [value, {"const": int(factor) if factor.is_integer() else factor}],
                    },
                    terms,
                )
    for child in node.get("of", ()):
        _own_operands(controller, line, event, child, a, b, convert, found)


def _where(controller: ControllerClock, line: Mapping[str, Any], event: Mapping[str, Any]) -> str:
    return (
        f"line {line_name(line['role'], line['qudit'])}, on {_the(controller.name)} at "
        f"{_ghz(controller.clock_ghz)}: the {event['kind']} that is event {event['index']}"
    )


def _the(name: str) -> str:
    """*name* to follow "on": a control system's ``controllers[i]`` as it is."""
    return name if name.startswith("controllers[") else f"the {name}"


def _decided(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: Mapping[str, Any],
    what: str,
    cycles: Any,
) -> list[str]:
    """The violation of a run-time length that can take a value off the
    line's timesteps, when the values it can take are known now."""
    grid = controller.grid
    values = _values(cycles)
    if grid == 1 or values is None:
        return []
    off = sorted(v for v in values if v % grid)
    if not off:
        return []
    shown = ", ".join(f"{v:g}" for v in off)
    return [
        f"{_where(controller, line, event)} has a {what}, computed at run time, of "
        f"{shown} cycles for some outcomes, which isn't a whole number of the line's "
        f"timesteps of {grid} cycles"
    ]


def _unconvertible(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: Mapping[str, Any],
    node: Mapping[str, Any],
    key: str,
) -> str:
    shape = node.get("name", node["shape"])
    return (
        f"{_where(controller, line, event)} plays a {shape!r} waveform whose operand {key} "
        "is computed at run time, and the export can't tell whether that operand is a time "
        f"to convert to the controller's cycles ({controller.ratio} of a program cycle "
        "each); compute it while the kernel compiles, or give it a waveform shape whose "
        "parameters are named"
    )


# The results of the ops a set of values passes through unchanged in kind.
_VALUE_OPS: Mapping[str, Any] = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "max": lambda x, y: x if x > y else y,
    "min": lambda x, y: x if x < y else y,
    "to_int": lambda x: int(x),
    "to_float": float,
}


def _values(expression: Any, limit: int = 64) -> set[float] | None:
    """The values *expression* can take, when it computes from constants and
    selects between them alone and they are few; else None."""
    terms = expression.get("terms", ()) if isinstance(expression, dict) else ()
    known: dict[int, set[float]] = {}
    # The terms whose values are being gathered, innermost last: each with
    # the operands that give it values, their sets so far, and its place in
    # ``terms`` if it is one of them. A set that is not known, or too
    # large, makes the whole unknown.
    gathering: list[tuple[str, Sequence[Any], list[set[float]], int | None]] = []
    node, place = expression, None
    while True:
        found: set[float] | None = None
        if isinstance(node, (int, float)):
            found = {node}
        elif not isinstance(node, dict):
            return None
        elif "const" in node:
            found = {node["const"]}
        elif "term" in node:
            n = node["term"]
            if n in known:
                found = known[n]
            elif type(n) is int and 0 <= n < len(terms) and all(n != g[3] for g in gathering):
                node, place = terms[n], n
                continue
            else:
                return None
        else:
            op: str = node.get("op", "")
            of = node.get("of", ())
            if op == "select" and len(of) == 3:
                of = of[1:]
            elif op not in _VALUE_OPS or not of:
                return None
            gathering.append((op, of, [], place))
            node, place = of[0], None
            continue
        # A set is found: it is an operand's, or the whole expression's.
        while True:
            if not gathering:
                return found
            op, of, sets, place = gathering[-1]
            assert found is not None
            sets.append(found)
            if len(sets) < len(of):
                node, place = of[len(sets)], None
                break
            gathering.pop()
            if op == "select":
                found = sets[0] | sets[1]
            else:
                fn = _VALUE_OPS[op]
                found = set()
                for args in itertools.product(*sets):
                    found.add(fn(*args))
                    if len(found) > limit:
                        return None
            if place is not None:
                known[place] = found


def off_grid_messages(
    controller: ControllerClock,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    off: Sequence[tuple[int, int, int]],
) -> list[str]:
    """The violations of the line events the native walk found off
    *controller*'s grid, each (index, start, duration) in program cycles."""
    found = []
    for index, start, duration in off:
        event = events[index]
        line = event["line"]
        found.append(_off_grid(controller, lines[line], event["kind"], start, duration))
    return found


def _off_grid(
    controller: ControllerClock,
    line: Mapping[str, Any],
    kind: str,
    start: int,
    duration: int,
) -> str:
    return off_timestep(
        line_name(line["role"], line["qudit"]), controller.grid, kind, start, duration
    )


def line_name(role: str, qudit: int) -> str:
    """``d3`` for the drive line of qudit 3, ``r3`` for its readout line."""
    return f"{role[0]}{qudit}"


def off_timestep(name: str, grid: int, kind: str, start: int, duration: int) -> str:
    """The violation of a *kind* on line *name* that does not start and last
    whole timesteps of *grid* program cycles."""
    problems = []
    if duration % grid:
        problems.append(
            f"{duration} cycles isn't a whole number of timesteps (the {kind} at cycle "
            f"{start}; pulse.round_duration(line, {duration}) gives "
            f"{-(-duration // grid) * grid}, or {duration // grid * grid} rounding down)"
        )
    if start % grid:
        problems.append(
            f"the {kind} starts at cycle {start}, which isn't a whole number of timesteps "
            f"(the nearest starts are {start // grid * grid} and {-(-start // grid) * grid})"
        )
    return f"line {name} moves in timesteps of {grid} cycles; " + "; ".join(problems)


# The times of each kind of export record, in cycles; lower bounds round down.
EVENT_TIMES = ("start_cycles", "duration_cycles", "end_cycles")
# A run-time wait's bounds, which only some events have.
EVENT_BOUNDS = ("max_cycles",)
EVENT_LOWER_BOUNDS = ("min_cycles",)
EMULATION_EVENT_TIMES = ("start_cycles", "duration_cycles", "segment_start_cycles")
SEGMENT_TIMES = ("start_cycles", "max_shift_cycles")
SEGMENT_LOWER_BOUNDS = ("min_shift_cycles",)
CALL_TIMES = (
    "submit_cycles",
    "submit_duration_cycles",
    "due_cycles",
    "deadline_cycles",
    "exposed_cycles",
    "first_use_cycles",
    "taken_cycles",
)


def to_own(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
    controller: ControllerClock,
    lower_bounds: Sequence[str] = (),
) -> None:
    """Convert the *keys* times of *records* (each has every key) to
    *controller*'s own cycles, in place, rounded up (line times are on its
    grid, so exact), and the *lower_bounds* rounded down."""
    a, b = controller.ratio.numerator, controller.ratio.denominator
    for record in records:
        for key in keys:
            value = record[key]
            if value is not None:
                record[key] = -((-value * a) // b)
        for key in lower_bounds:
            value = record[key]
            if value is not None:
                record[key] = value * a // b


def _scale(node: dict[str, Any], factor: float, scaled: set[int] | None = None) -> None:
    """Scale the time parameters of waveform *node* by *factor*, in place;
    with *scaled*, only those not yet scaled, by the ids of their
    ``params``, which it adds to."""
    params = node.get("params")
    if params is not None and (scaled is None or id(params) not in scaled):
        if scaled is not None:
            scaled.add(id(params))
        for key in _TIMES_OF.get(node["shape"], ()):
            value = params[key]
            if value is not None:
                params[key] = value * factor
    children = node.get("of")
    if children:
        for child in children:
            _scale(child, factor, scaled)


# The time parameters of each waveform shape, as ``controller._PARAMETERS``
# names them.
_TIMES_OF = {
    "square": ("duration",),
    "gaussian": ("duration", "sigma"),
    "gaussian_square": ("duration", "sigma", "risefall"),
    "drag": ("duration", "sigma", "beta"),
    "cosine": ("duration",),
    "tanh_ramp": ("duration", "sigma"),
    "custom": ("duration",),
}


def clock_keys(plan: ClockPlan, controller: ControllerClock) -> dict[str, float]:
    """The clock entries of a controller's export beside ``clock_ghz``."""
    keys: dict[str, float] = {}
    if controller.ratio != 1:
        keys["program_clock_ghz"] = plan.program_ghz
    if plan.reference_clock_ghz is not None:
        keys["reference_clock_ghz"] = plan.reference_clock_ghz
    return keys


def _round(n: float, grid: int, direction: RoundDirection) -> int:
    if direction is RoundDirection.UP:
        return -(-math.ceil(n - _TOLERANCE) // grid) * grid
    if direction is RoundDirection.DOWN:
        return math.floor(n + _TOLERANCE) // grid * grid
    return math.floor(n / grid + 0.5) * grid


def round_on_grid(n: Any, grid: int, direction: RoundDirection | str) -> int:
    """``round_duration`` once its line's timestep is known."""
    try:
        mode = RoundDirection(direction)
    except ValueError:
        raise InvalidValueError(
            "round_duration() direction must be one of "
            f"{', '.join(m.value for m in RoundDirection)}, got {direction!r}"
        ) from None
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
        raise InvalidValueError(
            f"round_duration() takes a finite, non-negative number of cycles, got {n!r}"
        )
    return _round(float(n), grid, mode)


def _program_clock(
    system: ControlSystem | None, declared: Any, default: float
) -> tuple[float | None, list[Problem]]:
    """The program clock of a quantum system of *system*: *declared* (its config's
    ``clock_ghz``), else the fastest of the controllers' own clocks when the
    others divide it, else *default*; with the problems that keep it from
    one, and the controllers' clocks that are not whole multiples of the
    system's reference clock."""
    clocks = {} if system is None else system._own_clocks
    try:
        program = program_clock_ghz(declared, clocks, default)
    except ValueError as error:
        return None, [Problem(str(error), path="clock_ghz", value=declared)]
    reference = None if system is None else system.reference_clock_ghz
    if system is None or reference is None:
        return program, []
    problems = []
    for index, controller in enumerate(system.controllers):
        clock = program if controller.clock_ghz is None else controller.clock_ghz
        try:
            _check_reference(reference, {"controller": clock})
        except ValueError:
            problems.append(
                Problem(
                    f"its {clock:g} GHz clock"
                    + (" (the program clock)" if controller.clock_ghz is None else "")
                    + f" is not a whole multiple of the {reference:g} GHz reference clock",
                    path=f"control_system.controllers[{index}].clock_ghz",
                    value=controller.clock_ghz,
                    fix="give it a clock the reference clock divides, or change "
                    "the control system's reference_clock_ghz",
                )
            )
    return program, problems
