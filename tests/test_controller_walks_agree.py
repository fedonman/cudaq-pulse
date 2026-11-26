# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two native walks of a program for a controller give the same events.

``controller_events()`` (and ``controller_events_on()``, on a controller's own
clock) is the walk a check lists violations from; ``controller_program()`` is
the walk of every export, and of a check where the first cannot be used. Here
both play every iteration of every loop and their events are compared as
JSON, key for key, with the check walk run in two ways: tracking the tones'
frame changes, and as a check asks for it (given a list for its repeats),
without.

The export walk says more than the tracking check walk in four places, and
the events are equal once those are taken out (``_alike``):

1. A ``frame_update`` event, which the check walk does not have, is dropped.
   What refers to an event by its place (``index``, ``measurements``, a
   condition's ``readouts`` and ``answers``, the ``readout`` and ``answer``
   leaves of an expression, the events off a controller's grid) is moved to
   the place the event has without them. A tone's id counts the tones in
   the order the events first name them, so the export's are counted again
   without those events.
2. The frame changes such an event took. The check walk keeps them pending
   until the tone plays, so there they are in the ``frame_updates`` of later
   events where the export's event has none of them. Where the check walk
   has one (found by its ``id``), it is taken out of the list and must be
   the change the export took (its ``op`` and ``value``), on the same tone,
   at or after the place of the dropped event, and its conditions must
   begin with those it had there. Not compared: whether a later event holds
   it, which and how many do, and the conditions a later branch adds.
3. A frame change's ``start_cycles``.
4. A condition's ``expression`` and ``result``.

The check walk that does not track has no ``tone``, ``frame_updates`` or
``phase_offset_rad`` either; rules 1 and 4 and those three keys are all that
is taken out of the export's events for it. It is compared where it
summarizes no loop.

Nothing else is allowed to differ: the lines, each line's runtime-qudit
slot, every other key of every event and, on a controller's own clock, the
events off its grid.

A long loop that the check walk summarizes is compared with its every
iteration in ``tests/test_controller_rolled_calls.py`` and
``tests/test_controller_segment_loops.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.controller import _AMPLITUDES, _PARAMETERS, ControllerLimitError
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q, ladder_target
from tests.property_settings import examples

_ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("asap", "alap", "rcp", "alap_rcp")
# A controller's own clock as (its cycles, per program cycles, its grid in
# program cycles): a 4-cycle timestep, half the program clock, two thirds of
# it.
OWN_CLOCKS = [(1, 1, 4), (1, 2, 2), (2, 3, 3)]

Program = tuple[str, Callable[[], Any]]
# The trip counts of the corpus's loops: both walks play every iteration here.
_TRIPS = (3, 20)

# ── The comparison ───────────────────────────────────────────────────────────


def _moved(node: Any, place: dict[int, int], within: str | None = None) -> Any:
    """*node* of an export event with its references to events moved to
    *place*, and without what only the export says of a frame change (in
    ``frame_updates``) and of a condition (in ``conditions``)."""
    if isinstance(node, list):
        return [_moved(item, place, within) for item in node]
    if not isinstance(node, dict):
        return node
    moved = {}
    for key, value in node.items():
        if within == "frame_updates" and key == "start_cycles":
            continue
        if within == "conditions" and key in ("expression", "result"):
            continue
        if key in ("index", "readout", "answer") and type(value) is int:
            moved[key] = place[value]
        elif key in ("measurements", "readouts", "answers"):
            moved[key] = [place[index] for index in value]
        else:
            moved[key] = _moved(value, place, key)
    return moved


def _raw(
    lowered: Any, own: tuple[int, int, int] | None = None, tracked: bool = True
) -> tuple[Any, Any] | None:
    """What the check walk and the export walk give for *lowered*, as they
    come: each a dictionary of the lines, their slots, the events and (on
    the clock *own*) the events off its grid. Without *tracked* the check
    walk is the one a check asks for, and where it summarizes a loop there
    is nothing to compare: None."""
    module = lowered._concrete()
    segments = lowered.classical != ()
    slots: tuple[list[int], list[int]] = ([], [])
    off_grid: tuple[Any, Any] = ((), ())
    repeats: list[Any] | None = None if tracked else []
    if own is None:
        lines, checked = module.controller_events(_PARAMETERS, _AMPLITUDES, slots[0], repeats)
        exported_lines, exported, _ = module.controller_program(
            _PARAMETERS, _AMPLITUDES, segments, slots=slots[1], exportable=False
        )
    else:
        lines, checked, off, ends = module.controller_events_on(
            _PARAMETERS, _AMPLITUDES, *own, slots[0], repeats
        )
        # As controller._walk does with them.
        for index, end in ends:
            checked[index]["end_cycles"] = end
        exported_lines, exported, _, exported_off = module.controller_program(
            _PARAMETERS, _AMPLITUDES, segments, own=own, slots=slots[1], exportable=False
        )
        off_grid = (off, exported_off)
    if repeats:
        return None
    return tuple(
        {"lines": lines, "slots": slots, "events": events, "off_grid": [list(e) for e in off]}
        for lines, slots, events, off in (
            (lines, slots[0], checked, off_grid[0]),
            (exported_lines, slots[1], exported, off_grid[1]),
        )
    )


_TRACKED_ONLY = ("tone", "frame_updates", "phase_offset_rad")


def _alike(check: Any, export: Any, tracked: bool = True) -> tuple[Any, Any]:
    """The two results of ``_raw`` without what only the export says (the
    module's docstring). The check walk's gets ``misplaced``: the frame
    changes that a ``frame_update`` event took and that it has in another
    way than rule 2 allows (none, for the export)."""
    # The export's events without its frame_update events, and the changes
    # those took: id -> (tone, the place of the next event, the change).
    place: dict[int, int] = {}
    kept = []
    taken: dict[int, tuple[Any, int, Any]] = {}
    for event in export["events"]:
        if event["kind"] == "frame_update":
            for update in event.get("frame_updates", ()):
                taken[update["id"]] = (event.get("tone"), len(kept), update)
        else:
            place[event["index"]] = len(kept)
            kept.append(event)
    kept = [_moved(event, place) for event in kept]
    checked = [dict(event) for event in check["events"]]

    if not tracked:
        for event in kept:
            for key in _TRACKED_ONLY:
                event.pop(key, None)
    # The export's tone ids in the order the kept events first name them.
    tones: dict[int, int] = {}
    for event in kept:
        if type(event.get("tone")) is int:
            event["tone"] = tones.setdefault(event["tone"], len(tones))

    # Rule 2. A change that the export's event at the same place has too is
    # pending there in both walks (on a path the frame_update event is not
    # on) and is compared with it.
    misplaced = []
    for index, event in enumerate(checked):
        updates = event.get("frame_updates")
        if not updates:
            continue
        there = kept[index].get("frame_updates", ()) if index < len(kept) else ()
        there = {update["id"] for update in there}
        event["frame_updates"] = []
        for update in updates:
            if update["id"] in there or update["id"] not in taken:
                event["frame_updates"].append(update)
                continue
            tone, after, was = taken[update["id"]]
            was = _moved(was, place, "frame_updates")
            same = {**update, "conditions": update["conditions"][: len(was["conditions"])]}
            if _json(same) != _json(was) or event["tone"] != tones.get(tone) or index < after:
                misplaced.append([index, update])
    return (
        {**check, "events": checked, "misplaced": misplaced},
        {
            **export,
            "events": kept,
            "off_grid": [[place[index], *rest] for index, *rest in export["off_grid"]],
            "misplaced": [],
        },
    )


def _walks(lowered: Any, own: tuple[int, int, int] | None = None) -> tuple[Any, Any]:
    raw = _raw(lowered, own)
    assert raw is not None
    return _alike(*raw)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _difference(check: Any, export: Any) -> str | None:
    """Where the two results of ``_walks`` differ first, or None."""
    for part in ("lines", "slots", "off_grid", "misplaced"):
        if _json(check[part]) != _json(export[part]):
            return f"{part}: check {_json(check[part])[:300]}, export {_json(export[part])[:300]}"
    checked, exported = check["events"], export["events"]
    if len(checked) != len(exported):
        return f"{len(checked)} check events, {len(exported)} export events"
    for index, (a, b) in enumerate(zip(checked, exported, strict=True)):
        keys = sorted(
            key
            for key in {*a, *b}
            if key not in a or key not in b or _json(a[key]) != _json(b[key])
        )
        if keys:
            key = keys[0]
            said = [_json(event[key])[:300] if key in event else "nothing" for event in (a, b)]
            return (
                f"event {index} ({a['kind']}) of {len(checked)} differs in {', '.join(keys)}; "
                f"{key}: check {said[0]}, export {said[1]}"
            )
    return None


def _disagreement(lowered: Any, own: tuple[int, int, int] | None = None) -> str | None:
    """Where the export walk of *lowered* differs first from the check walk
    that tracks frame changes, or from the one that does not; or None."""
    difference = _difference(*_walks(lowered, own))
    if difference is None and (raw := _raw(lowered, own, tracked=False)) is not None:
        difference = _difference(*_alike(*raw, tracked=False))
        if difference is not None:
            difference = f"not tracking, {difference}"
    return difference


# ── Small programs, always run ───────────────────────────────────────────────


@pulse.classical(max_latency_ns=200)
def _settings(bits: pulse.Bits) -> tuple[float, int]:
    return 0.1 * bits[0], 40 + bits[0]


@pulse.kernel
def _pulses(q, r):
    a = get_drive_line(q)
    b = get_drive_line(r)
    drive(a, gaussian(16, 0.1, 4.0))
    shift_phase(a.tone, 0.25)
    sync(a, b)
    drive(b, square(10, 0.2))
    wait(a, 7)
    drive(a, square(5, 0.1))


@pulse.kernel
def _detuned_train(q, r):
    d = get_drive_line(q)
    shift_frequency(d.tone, 3.0e6)
    for _ in range(6):
        drive(d, square(16, 0.1))
        shift_phase(d.tone, 0.37)


@pulse.kernel
def _frame_change_in_a_branch(q, r):
    d = get_drive_line(q)
    p = get_readout_line(q)
    m = readout(p, square(40, 0.05))
    sync(p, d)
    if m:
        shift_phase(d.tone, 0.5)
    drive(d, square(16, 0.1))
    n = readout(p, square(40, 0.05))
    sync(p, d)
    if m and not n:
        drive(d, square(8, 0.1))
    shift_phase(d.tone, 0.25)


@pulse.kernel
def _frame_change_of_a_tone_played_later(q, r):
    # The export has a frame_update event for the change in the arm; the
    # check walk keeps it for the drive after the second readout.
    a = get_drive_line(q)
    b = get_drive_line(r)
    p = get_readout_line(q)
    m = readout(p, square(16, 0.05))
    if m:
        wait(a, 8)
    else:
        shift_phase(b.tone, 0.5)
    n = readout(get_readout_line(r), square(16, 0.05))
    if n:
        drive(b, square(56, 0.2))


@pulse.kernel
def _frame_changes_of_a_tone_never_played(q, r):
    # The other qudit's tone is the first the export's events name, in
    # frame_update events alone.
    a = get_drive_line(q)
    b = get_drive_line(r)
    m = readout(get_readout_line(r), square(40, 0.05))
    if m:
        for _ in range(3):
            shift_phase(b.tone, 0.5)
            sync(a, b)
            shift_phase(a.tone, 0.5)
            drive(a, square(24, 0.3))
    else:
        drive(a, square(64, 0.3))


@pulse.kernel
def _answers(q, r):
    d = get_drive_line(q)
    p = get_readout_line(q)
    for _ in range(3):
        m = readout(p, square(100, 0.05))
        amp, delay = _settings.submit([m]).result()
        shift_phase(d.tone, amp)
        wait(d, delay, bounds=(8, 400))
        if delay > 40:
            drive(d, square(20, amp * 0.5))
        drive(d, square(20, 0.2))


_SMALL = [
    _pulses,
    _detuned_train,
    _frame_change_in_a_branch,
    _frame_change_of_a_tone_played_later,
    _frame_changes_of_a_tone_never_played,
    _answers,
]


def _small(kernel: Any, **options: Any) -> Any:
    return pulse.compile(kernel, **options).lower(Q[0], Q[1])


@pytest.mark.parametrize("own", [None, *OWN_CLOCKS], ids=str)
@pytest.mark.parametrize("unroll_threshold", [0, 10**6], ids=["rolled", "unrolled"])
@pytest.mark.parametrize("kernel", _SMALL, ids=lambda kernel: kernel.__wrapped__.__name__[1:])
def test_the_two_walks_give_the_same_events_for_a_small_program(kernel, unroll_threshold, own):
    lowered = _small(kernel, unroll_threshold=unroll_threshold)
    assert _disagreement(lowered, own) is None


def _events(kernel: Any) -> tuple[list[Any], list[Any]]:
    """The events of the two walks of *kernel* as they come."""
    check, export = _raw(_small(kernel))
    return check["events"], export["events"]


def test_the_small_programs_have_what_only_the_export_walk_says():
    """The small programs have a frame_update event, a change it took, a
    change's start_cycles, a condition's expression and a tone only such
    events name: the comparison above is one of events that differ before
    it."""
    checked, exported = _events(_frame_change_of_a_tone_played_later)
    [update] = (event for event in exported if event["kind"] == "frame_update")
    [change] = update["frame_updates"]
    assert "start_cycles" in change
    # The export's later drive of the tone is without the change; the check
    # walk's has it.
    assert all(e["kind"] != "frame_update" for e in checked)
    [later] = (e for e in exported if e["kind"] == "drive")
    assert not later["frame_updates"]
    [later] = (e for e in checked if e["kind"] == "drive")
    assert [u["id"] for u in later["frame_updates"]] == [change["id"]]

    _, exported = _events(_answers)
    conditions = [c for event in exported for c in event["conditions"]]
    assert any("expression" in condition for condition in conditions)

    # A tone that only frame_update events name is the first of the
    # export's and none of the check walk's.
    checked, exported = _events(_frame_changes_of_a_tone_never_played)
    assert {e["tone"] for e in exported if e["kind"] == "drive"} == {1}
    assert {e["tone"] for e in checked if e["kind"] == "drive"} == {0}


@pytest.mark.parametrize(
    "change",
    [
        lambda event: event.update(start_cycles=event["start_cycles"] + 1),
        lambda event: event.update(duration_cycles=event["duration_cycles"] + 1),
        lambda event: event.update(line=event["line"] + 1),
        lambda event: event.update(conditions=[*event["conditions"], {"branch": 9}]),
        lambda event: event.update(frame_updates=[]),
        lambda event: event.pop("phase_rad"),
        lambda event: event.update(extra=None),
    ],
)
def test_the_comparison_reports_a_changed_event(change):
    check, export = _walks(_small(_frame_change_in_a_branch))
    assert _difference(check, export) is None
    event = next(e for e in export["events"] if e["kind"] == "drive" and e.get("frame_updates"))
    change(event)
    place = export["events"].index(event)
    assert f"event {place} (drive)" in (_difference(check, export) or "")


def test_the_comparison_reports_a_missing_event_and_another_line():
    check, export = _walks(_small(_pulses), OWN_CLOCKS[0])
    assert check["off_grid"], "the program has events off a 4-cycle grid"
    assert "check events" in (_difference(check, {**export, "events": export["events"][:-1]}) or "")
    assert "lines:" in (_difference(check, {**export, "lines": export["lines"][:-1]}) or "")
    assert "off_grid:" in (_difference(check, {**export, "off_grid": []}) or "")


def test_a_change_the_export_took_is_looked_for_in_the_check_walks_events():
    """Rule 2: the change must be the one the export took, on its tone, with
    the conditions it had."""

    def changed(change: Callable[[Any], None]) -> str:
        check, export = _raw(_small(_frame_change_of_a_tone_played_later))
        assert _difference(*_alike(check, export)) is None
        [later] = (e for e in check["events"] if e["kind"] == "drive")
        change(later)
        return _difference(*_alike(check, export)) or ""

    assert "misplaced:" in changed(lambda later: later["frame_updates"][0].update(value=0.75))
    assert "misplaced:" in changed(lambda later: later["frame_updates"][0].update(conditions=[]))
    # One more change than the export took is not one of them.
    extra = {"conditions": [], "id": 99, "op": "shift_phase", "value": 0.5}
    assert "frame_updates" in changed(lambda later: later["frame_updates"].append(extra))


# ── The corpus ───────────────────────────────────────────────────────────────


def _argument_count(kernel: Any) -> int:
    return kernel.__wrapped__.__code__.co_argcount


def _load(relative: str) -> Any:
    path = _ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _loops() -> Iterator[Program]:
    """The loops of the rolled-loop tests, kept rolled."""
    from tests import test_controller_drifting_loops as drifting
    from tests import test_controller_rolled_loops as rolled

    def lowered(kernel: Any, policy: str) -> Any:
        compiled = pulse.compile(kernel, schedule=policy, unroll_threshold=0)
        handles = Q[: _argument_count(kernel)]
        if kernel.__wrapped__.__name__ == "scaled":
            return compiled.lower(Q[0], 0.6)
        return compiled.lower(*handles)

    def drifted(kernel: Any, policy: str) -> Any:
        qpu = pulse.quantum_system("transmon_surface17")
        compiled = pulse.compile(kernel, quantum_system=qpu, schedule=policy, unroll_threshold=0)
        return compiled.lower(*qpu.qudits[: _argument_count(kernel)])

    for trips in _TRIPS:
        for table in (rolled._loops, rolled._unsynced, rolled._on_grid, rolled._runtime):
            for name, kernel in table(trips).items():
                for policy in POLICIES:
                    yield (
                        f"{table.__name__[1:]} {name} {trips} {policy}",
                        lambda kernel=kernel, policy=policy: lowered(kernel, policy),
                    )
        for name, kernel in drifting._drifting(trips).items():
            for policy in POLICIES:
                yield (
                    f"drifting {name} {trips} {policy}",
                    lambda kernel=kernel, policy=policy: drifted(kernel, policy),
                )


def _loops_on_systems() -> Iterator[Program]:
    """The two-qudit loops of the rolled-loop tests on their control
    systems; a loop that does not fit a system is left out by the sweep."""
    from tests import test_controller_rolled_loops as rolled

    def lowered(system: str, kernel: Any, policy: str) -> Any:
        qpu = ladder_target(4, control_system=rolled._systems()[system])
        compiled = pulse.compile(kernel, quantum_system=qpu, schedule=policy, unroll_threshold=0)
        handles = (qpu.qudits[0], qpu.qudits[2])
        if _argument_count(kernel) == 3:
            return compiled.lower(*handles, 0.6)
        return compiled.lower(*handles)

    for trips in _TRIPS:
        for system in rolled._systems():
            for name, kernel in rolled._across(trips).items():
                for policy in POLICIES:
                    yield (
                        f"{system} {name} {trips} {policy}",
                        lambda system=system, kernel=kernel, policy=policy: lowered(
                            system, kernel, policy
                        ),
                    )


def _loops_of_calls() -> Iterator[Program]:
    """The loops of the rolled-call and segment-loop tests."""
    from tests import test_controller_rolled_calls as calls
    from tests import test_controller_segment_loops as rounds

    for trips in _TRIPS:
        for module, table in (
            (calls, calls._calls),
            (calls, calls._frames),
            (calls, calls._branches),
            (rounds, rounds._rounds),
        ):
            for name, kernel in table(trips).items():
                for policy in POLICIES:
                    yield (
                        f"{table.__name__[1:]} {name} {trips} {policy}",
                        lambda module=module, kernel=kernel, policy=policy: module._lowered(
                            kernel, policy
                        ),
                    )


def _expressions() -> Iterator[Program]:
    """The programs of ``test_controller_expressions.py`` that compute at
    run time: phases, wait lengths, conditions; each also compiled as that
    file compiles it, for a controller that selects a value at run time."""
    from tests import test_controller_expressions as expressions

    def selecting(lower: Callable[[], Any]) -> Any:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)
            return lower()

    bodies = {
        f"phase {term}": [*body, "drive(d, square(20, 0.2))"]
        for term, body in expressions._PHASES.items()
    }
    bodies |= {f"wait {term}": body for term, body in expressions._WAITS.items()}
    bodies["wait bounded"] = ["wait(d, n * 4, bounds=(8, 400))"]
    bodies["branch"] = expressions._BRANCH
    bodies["branch computed"] = ["if (n * 3 - 4) // 2 >= n % 2 + 1:", *expressions._BRANCH[1:]]
    for value, (first, use, _) in expressions._DOUBLED.items():
        step = "c = (c | c) & (c | e)" if value == "condition" else "t = t + t"
        bodies[f"doubled {value}"] = [*first, *[step] * 40, use]
    programs: dict[str, Callable[[], Any]] = {
        name: lambda body=body: pulse.compile(expressions._kernel(body)).lower(Q[0])
        for name, body in bodies.items()
    }
    programs |= {
        f"condition {name}": lambda form=form: expressions._branching(form)
        for name, form in expressions._CONDITIONS.items()
    }
    for name, lower in programs.items():
        yield name, lower
        yield f"{name}, selecting", lambda lower=lower: selecting(lower)


def _schedules() -> Iterator[tuple[str, dict[str, Any]]]:
    for policy in POLICIES:
        for mode, unroll in (("rolled", 0), ("unrolled", 10**6)):
            yield f"{policy} {mode}", {"schedule": policy, "unroll_threshold": unroll}


def _kernels() -> Iterator[Program]:
    """The kernel corpus of the IR golden test (the six workloads among
    them) under each schedule, rolled and unrolled, without the kernels
    whose golden record is an error."""
    from cudaq_pulse.kernel.ir_builder import Parameter

    from tests.kernel import ir_corpus

    def traced(case: Any, **options: Any) -> Any:
        # A float or int parameter of the corpus is a value here.
        handles = [
            (0.3 if argument.dtype == "f64" else 3) if isinstance(argument, Parameter) else argument
            for argument in case.args()
        ]
        with warnings.catch_warnings():
            # A float the kernel compares is a value while it traces.
            warnings.simplefilter("ignore")
            return pulse.compile(case.kernel, **options).lower(*handles)

    kernels = {
        case.kernel: case for case in ir_corpus.CASES if "error" not in ir_corpus.trace(case)
    }
    for schedule, options in _schedules():
        for case in kernels.values():
            yield (
                f"{case.name} {schedule}",
                lambda case=case, options=options: traced(case, **options),
            )


def _bugs() -> Iterator[Program]:
    """The bug corpus under each schedule, rolled and unrolled, without its
    waits of a negative length, which are compile errors."""
    from tests.workloads.bug_corpus import ALL_CASES

    def built(case: Any, **options: Any) -> Any:
        kernel = case.build_fn()
        return pulse.compile(kernel, **options).lower(*Q[: _argument_count(kernel)])

    for schedule, options in _schedules():
        for case in ALL_CASES:
            if case.expected != "backward_time_travel":
                yield (
                    f"{case.name} {schedule}",
                    lambda case=case, options=options: built(case, **options),
                )


def _calls(root: str) -> Iterator[Program]:
    """The valid programs of the schedule matrix of classical calls whose
    compiled kernel has the schedule *root*."""
    from tests.classical import _schedule_matrix as matrix

    for case in matrix.all_cases():
        if case.root == root and matrix.expected_error(case) is None:
            yield case.id, lambda case=case: matrix.lower_case(case, Q)


def _benchmarks() -> Iterator[Program]:
    """The perf gate's feedback and call programs and the compile
    benchmark's workloads."""
    cases = _load("benchmarks/perf_gate/cases.py")
    for trips in (2, 16, 64):
        yield f"feedback {trips}", lambda trips=trips: cases._feedback_program(trips)
    for rounds, calls in ((4, 2), (64, 4)):
        yield (
            f"many_calls {rounds} {calls}",
            lambda rounds=rounds, calls=calls: pulse.compile(
                cases._calls.many_calls(rounds, calls),
                quantum_system=cases._DEVICE,
                schedule="asap",
            ).lower(cases.Q[0]),
        )
    for name, (kernel, count) in cases._WORKLOADS.items():
        for policy in POLICIES:
            yield (
                f"{name} {policy}",
                lambda kernel=kernel, count=count, policy=policy: pulse.compile(
                    kernel, quantum_system=cases._DEVICE, schedule=policy
                ).lower(*cases.Q[:count]),
            )


# Each part of the corpus: its programs, the fewest of them that must
# compile (so that a part cannot go empty unnoticed) and the errors that
# leave a program out.
_CORPUS: dict[str, tuple[Callable[[], Iterator[Program]], int, tuple[type[Exception], ...]]] = {
    "loops": (_loops, 200, ()),
    "loops on control systems": (_loops_on_systems, 350, (ControllerLimitError,)),
    "loops of calls": (_loops_of_calls, 200, ()),
    "expressions": (_expressions, 90, ()),
    "kernels": (_kernels, 180, ()),
    "bug corpus": (_bugs, 600, ()),
    **{
        f"calls under {root}": (lambda root=root: _calls(root), 200, ())
        for root in ("default", "asap", "alap", "rcp", "alap_rcp")
    },
    "benchmarks": (_benchmarks, 20, ()),
}


@pytest.mark.level2
@pytest.mark.parametrize("part", list(_CORPUS))
def test_the_two_walks_give_the_same_events_over_the_corpus(part):
    programs, fewest, left_out = _CORPUS[part]
    compared = 0
    found = []
    for name, lower in programs():
        try:
            lowered = lower()
        except left_out:
            continue
        compared += 1
        for own in (None, *OWN_CLOCKS):
            difference = _disagreement(lowered, own)
            if difference is not None:
                found.append((lowered.metrics.op_count, f"{name}, clock {own}: {difference}"))
    assert compared >= fewest, f"only {compared} programs of {part!r} compiled"
    # The smallest programs first.
    assert not found, f"{len(found)} of {4 * compared} differ:\n" + "\n".join(
        message for _, message in sorted(found)[:5]
    )


# ── Random programs ──────────────────────────────────────────────────────────

_STEP = st.tuples(
    st.sampled_from(("drive", "readout", "variable_wait", "call", "phase", "branch", "loop")),
    st.integers(0, 2),
)


def _random_program(steps: list[tuple[str, int]]) -> Any:
    """Drives, readouts, calls, waits an answer sets, frame changes (plain,
    from an answer, in a branch on the latest readout) and short loops, on
    three qudits."""

    @pulse.kernel
    def k(q0, q1, q2):
        qs = (q0, q1, q2)
        amp, delay = 0.1, 40
        m = readout(get_readout_line(q0), square(50, 0.05))
        for kind, q in steps:
            d = get_drive_line(qs[q])
            r = get_readout_line(qs[q])
            if kind == "drive":
                drive(d, square(30, 0.1))
            elif kind == "readout":
                m = readout(r, square(50, 0.05))
            elif kind == "call":
                amp, delay = _settings.submit([m]).result()
            elif kind == "variable_wait":
                wait(d, delay, bounds=(8, 400))
            elif kind == "phase":
                shift_phase(d.tone, amp)
            elif kind == "branch":
                if m:
                    shift_phase(d.tone, 0.5)
                else:
                    drive(d, square(12, 0.2))
            elif kind == "loop":
                for _ in range(3):
                    drive(d, square(10, 0.1))
                    shift_phase(d.tone, 0.25)

    return k


@pytest.mark.level2
@settings(max_examples=examples(60), deadline=None)
@given(
    steps=st.lists(_STEP, min_size=1, max_size=20),
    policy=st.sampled_from(POLICIES),
    unroll_threshold=st.sampled_from((0, 10**6)),
    own=st.sampled_from((None, *OWN_CLOCKS)),
)
def test_the_two_walks_give_the_same_events_for_random_programs(
    steps, policy, unroll_threshold, own
):
    compiled = pulse.compile(
        _random_program(steps), schedule=policy, unroll_threshold=unroll_threshold
    )
    assert _disagreement(compiled.lower(Q[0], Q[1], Q[2]), own) is None
