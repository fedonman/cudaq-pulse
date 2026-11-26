# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw a compiled kernel's schedule as a timeline, from what the compiler exports.

``Timeline.of(scheduled)`` reads a ``ScheduledKernel``: its controller
export, sampled at the clock, gives every pulse's envelope, condition, wait,
frame update and sync, and every call's submit, answer and first use, with
the link's latencies and the function's budget from its manifest. Nothing
is typed in by hand. ``draw`` turns a timeline into a figure in
the theme of a ``Palette``: a lane per line, grouped by controller, with a
lane per link for the calls.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from _style import WIDTH_IN, Palette
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator

LaneKey = tuple[str, int]


@dataclass(frozen=True)
class Lane:
    key: LaneKey
    label: str
    role: str  # "drive", "readout" or "link"
    group: str | None = None


@dataclass(frozen=True)
class Pulse:
    lane: LaneKey
    kind: str
    start_ns: float
    end_ns: float
    t_ns: tuple[float, ...]
    i: tuple[float, ...]
    q: tuple[float, ...]
    conditional: bool


@dataclass(frozen=True)
class Mark:
    lane: LaneKey
    t_ns: float
    text: str


@dataclass(frozen=True)
class Call:
    function: str
    lane: LaneKey
    submit_ns: float
    sent_ns: float
    due_ns: float
    source: tuple[LaneKey, float] | None
    use: tuple[LaneKey, float] | None
    timeout_ns: float | None
    max_latency_ns: float | None
    link: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sync:
    t_ns: float
    lanes: tuple[LaneKey, ...]


@dataclass(frozen=True)
class Forward:
    """A readout's outcome on its way to a pulse that branches on it: ready
    on its own controller *feedback_ns* after the readout ends, and on
    another controller *peer_ns* later still."""

    lane: LaneKey
    source: tuple[LaneKey, float]
    use: tuple[LaneKey, float]
    feedback_ns: float
    peer_ns: float


@dataclass(frozen=True)
class Wait:
    """A wait on a lane: *length_ns* as scheduled, and for a wait whose length
    comes from a classical answer, the bounds the scheduler was given."""

    lane: LaneKey
    start_ns: float
    end_ns: float
    length_ns: float
    min_ns: float | None = None
    max_ns: float | None = None


@dataclass(frozen=True)
class Cut:
    """A stretch of idle time a compressed timeline squeezes, from *start_ns* to
    *end_ns* of the program, drawn from *at_ns* to *at_ns* + *width_ns*."""

    start_ns: float
    end_ns: float
    at_ns: float
    width_ns: float


@dataclass(frozen=True)
class Timeline:
    clock_ghz: float
    lanes: tuple[Lane, ...]
    pulses: tuple[Pulse, ...]
    marks: tuple[Mark, ...]
    calls: tuple[Call, ...]
    syncs: tuple[Sync, ...]
    waits: tuple[Wait, ...] = ()
    forwards: tuple[Forward, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def of(cls, scheduled, *, names: dict[int, str] | None = None) -> Timeline:
        """The timeline of *scheduled*; *names* labels qudits by index."""
        qs = scheduled.quantum_system
        controllers = list(getattr(qs.control_system, "controllers", None) or [])
        clock = qs.clock_ghz
        if len(controllers) > 1:
            exports = [
                (f"controller {i}", scheduled.export_to_controller(clock, controller=i))
                for i in range(len(controllers))
            ]
        else:
            exports = [(None, scheduled.export_to_controller(clock))]
        if names is None:
            physical = getattr(qs, "physical_qudits", None) or {}
            names = {i: getattr(pq, "label", None) or f"qudit {i}" for i, pq in physical.items()}

        def qudit_name(index: int) -> str:
            return names.get(index, f"qudit {index}")

        lanes: list[Lane] = []
        by_index: dict[int, tuple[dict, float]] = {}
        syncs: dict[tuple, Sync] = {}
        for group, program in exports:
            ghz = program["clock_ghz"]
            for line in sorted(program["lines"], key=lambda ln: (ln["qudit"], ln["role"])):
                key = ("line", line["line"])
                label = f"{qudit_name(line['qudit'])} {line['role']}"
                lanes.append(Lane(key, label, line["role"], group))
            for event in program["events"]:
                by_index.setdefault(event["index"], (event, ghz))
                if event["kind"] == "sync" and event["lines"]:
                    t = event["start_cycles"] / ghz
                    keys = tuple(("line", n) for n in event["lines"])
                    old = syncs.get((t,), Sync(t, ()))
                    syncs[(t,)] = Sync(t, tuple(dict.fromkeys(old.lanes + keys)))
        used = {e["line"] for e, _ in by_index.values() if e["kind"] in ("drive", "readout")}
        lanes = [ln for ln in lanes if ln.key[1] in used]

        pulses, marks, waits = [], [], []
        for event, ghz in by_index.values():
            lane = ("line", event["line"])
            if event["kind"] in ("drive", "readout"):
                start, stop = event["start_cycles"] / ghz, event["end_cycles"] / ghz
                i = np.array([0.0, *event["samples"]["re"], 0.0])
                q = np.array([0.0, *event["samples"]["im"], 0.0])
                t = np.array([start, *(start + (np.arange(len(i) - 2) + 0.5) / ghz), stop])
                keep = _simplify(t, [i, q], 0.004 * float(np.max(np.hypot(i, q)) or 1.0))
                pulses.append(
                    Pulse(
                        lane,
                        event["kind"],
                        start,
                        stop,
                        tuple(np.round(t[keep], 3).tolist()),
                        tuple(np.round(i[keep], 6).tolist()),
                        tuple(np.round(q[keep], 6).tolist()),
                        bool(event["conditions"]),
                    )
                )
            elif event["kind"] == "frame_update" and event["value"]:
                marks.append(Mark(lane, event["start_cycles"] / ghz, _frame_text(event)))
            elif event["kind"] == "wait" and event["duration_cycles"]:
                bounds = (
                    (event["min_cycles"] / ghz, event["max_cycles"] / ghz)
                    if "length" in event
                    else (None, None)
                )
                start, stop = event["start_cycles"] / ghz, event["end_cycles"] / ghz
                waits.append(Wait(lane, start, stop, stop - start, *bounds))

        link_names = {v: k for k, v in scheduled.options.machine.link_names.items()}
        links, functions = {}, {}
        for _, program in exports:
            links.update(program.get("links") or {})
            for fn in (program.get("manifest") or {}).get("functions", []):
                functions[fn["name"]] = fn
        # A call is its submit event and the next answer event of the same function:
        # the events carry the times of the program as bound and scheduled.
        events = sorted(by_index.values(), key=lambda item: item[0]["index"])
        calls = []
        call_lanes: dict[int, Lane] = {}
        for k, (submit, ghz) in enumerate(events):
            if submit["kind"] != "submit":
                continue
            answer = next(
                (
                    e
                    for e, _ in events[k + 1 :]
                    if e["kind"] == "answer" and e["function"] == submit["function"]
                ),
                None,
            )
            link = submit["link"]
            lane = call_lanes.setdefault(
                link, Lane(("link", link), f"{link_names.get(link, link)} link", "link")
            )
            source = None
            if submit.get("measurements"):
                readout, rghz = by_index[submit["measurements"][-1]]
                source = (("line", readout["line"]), readout["end_cycles"] / rghz)
            use = None
            if answer is not None:
                users = [
                    (e["start_cycles"] / eghz, ("line", e["line"]))
                    for e, eghz in events
                    if any(answer["index"] in cond["answers"] for cond in e["conditions"])
                    or answer["index"] in _answers_read(e.get("length"))
                ]
                if users:
                    t, key = min(users)
                    use = (key, t)
            fn = functions.get(submit["function"], {})
            calls.append(
                Call(
                    submit["function"].rsplit("__", 1)[0],
                    lane.key,
                    submit["start_cycles"] / ghz,
                    submit["end_cycles"] / ghz,
                    (answer["start_cycles"] if answer else submit["end_cycles"]) / ghz,
                    source,
                    use,
                    answer.get("timeout_ns") if answer else None,
                    (fn.get("latency_ns") or {}).get("max"),
                    dict(links.get(str(link), {})),
                )
            )
        lanes += call_lanes.values()

        forwards = _forwards(events, controllers, exports, clock)
        if forwards:
            lanes.append(Lane(_BUS, "outcome bus", "bus"))
        return cls(
            clock,
            tuple(lanes),
            tuple(pulses),
            tuple(marks),
            tuple(calls),
            tuple(sorted(syncs.values(), key=lambda s: s.t_ns)),
            tuple(waits),
            tuple(forwards),
        )


_BUS: LaneKey = ("bus", 0)


def _forwards(events, controllers, exports, clock: float) -> list[Forward]:
    """The outcomes pulses branch on, from the controllers' latencies: a
    pulse whose condition tests a readout gets one forward from that readout,
    the first such pulse on each lane. Only for a control system that lists
    its controllers, which state the latencies."""
    if not controllers:
        return []
    owner = {}
    for number, (_, program) in enumerate(exports):
        for line in program["lines"]:
            owner[line["line"]] = number if len(exports) > 1 else 0
    by_index = {e["index"]: (e, ghz) for e, ghz in events}
    forwards, seen = [], set()
    for event, ghz in events:
        for condition in event.get("conditions", []):
            for index in condition.get("readouts", []):
                readout, readout_ghz = by_index[index]
                key = (index, event["line"])
                if key in seen:
                    continue
                seen.add(key)
                sender = controllers[owner[readout["line"]]]
                receiver = owner[event["line"]]
                sender_ghz = sender.clock_ghz or clock
                feedback = sender.feedback_latency_cycles / sender_ghz
                peer = 0.0
                if receiver != owner[readout["line"]]:
                    peer = sender.peer_latency_cycles / sender_ghz
                forwards.append(
                    Forward(
                        _BUS,
                        (("line", readout["line"]), readout["end_cycles"] / readout_ghz),
                        (("line", event["line"]), event["start_cycles"] / ghz),
                        feedback,
                        peer,
                    )
                )
    return forwards


def _answers_read(expression: Any) -> set[int]:
    """The answer events an exported expression reads."""
    if isinstance(expression, dict):
        found = {expression["answer"]} if "answer" in expression else set()
        for value in expression.values():
            found |= _answers_read(value)
        return found
    if isinstance(expression, list):
        return set().union(*map(_answers_read, expression)) if expression else set()
    return set()


def compress(timeline: Timeline, longest_ns: float) -> tuple[Timeline, tuple[Cut, ...]]:
    """*timeline* with every stretch longer than *longest_ns* in which no pulse,
    mark or call is under way squeezed to *longest_ns*, and the cuts it made.
    Waits keep their scheduled length and bounds, for their labels."""
    busy = sorted(
        [(pl.start_ns, pl.end_ns) for pl in timeline.pulses]
        + [(c.submit_ns, c.due_ns) for c in timeline.calls]
        + [(m.t_ns, m.t_ns) for m in timeline.marks]
    )
    cuts: list[tuple[float, float]] = []
    reach = busy[0][1] if busy else 0.0
    for a, b in busy[1:]:
        if a - reach > longest_ns:
            cuts.append((reach, a))
        reach = max(reach, b)

    def at(t: float) -> float:
        shift = 0.0
        for a, b in cuts:
            if t >= b:
                shift += (b - a) - longest_ns
            elif t > a:
                return t - shift - (t - a) + (t - a) * longest_ns / (b - a)
        return t - shift

    def moved(times):
        return tuple(at(x) for x in times)

    squeezed = Timeline(
        timeline.clock_ghz,
        timeline.lanes,
        tuple(
            Pulse(
                pl.lane,
                pl.kind,
                at(pl.start_ns),
                at(pl.end_ns),
                moved(pl.t_ns),
                pl.i,
                pl.q,
                pl.conditional,
            )
            for pl in timeline.pulses
        ),
        tuple(Mark(m.lane, at(m.t_ns), m.text) for m in timeline.marks),
        tuple(
            Call(
                c.function,
                c.lane,
                at(c.submit_ns),
                at(c.sent_ns),
                at(c.due_ns),
                c.source and (c.source[0], at(c.source[1])),
                c.use and (c.use[0], at(c.use[1])),
                c.timeout_ns,
                c.max_latency_ns,
                c.link,
            )
            for c in timeline.calls
        ),
        tuple(Sync(at(s.t_ns), s.lanes) for s in timeline.syncs),
        tuple(
            Wait(w.lane, at(w.start_ns), at(w.end_ns), w.length_ns, w.min_ns, w.max_ns)
            for w in timeline.waits
        ),
        tuple(
            Forward(
                f.lane,
                (f.source[0], at(f.source[1])),
                (f.use[0], at(f.use[1])),
                f.feedback_ns,
                f.peer_ns,
            )
            for f in timeline.forwards
        ),
    )
    return squeezed, tuple(Cut(a, b, at(a), longest_ns) for a, b in cuts)


def _simplify(t: np.ndarray, ys: list[np.ndarray], tol: float) -> np.ndarray:
    """The indices of the points that keep every curve of *ys* within *tol* of its
    straight-line outline (Ramer-Douglas-Peucker on time and every curve)."""
    keep = np.zeros(len(t), bool)
    keep[[0, -1]] = True
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        f = (t[a + 1 : b] - t[a]) / (t[b] - t[a])
        err = max(np.max(np.abs(y[a + 1 : b] - (y[a] + f * (y[b] - y[a])))) for y in ys)
        if err > tol:
            k = (
                a
                + 1
                + int(
                    np.argmax(
                        np.max(
                            [np.abs(y[a + 1 : b] - (y[a] + f * (y[b] - y[a]))) for y in ys], axis=0
                        )
                    )
                )
            )
            keep[k] = True
            stack += [(a, k), (k, b)]
    return np.flatnonzero(keep)


def _frame_text(event: dict) -> str:
    value = event["value"]
    if event["op"] in ("shift_phase", "set_phase"):
        frac = value / math.pi
        for d in (1, 2, 3, 4, 6, 8):
            if abs(frac * d - round(frac * d)) < 1e-9:
                n = round(frac * d)
                num = {1: "", -1: "−"}.get(n, f"{n}")
                text = f"{num}π" if d == 1 else f"{num}π/{d}"
                return ("+" if n > 0 else "") + text
        return f"{value:+.3g} rad"
    return f"{value / 1e6:+.3g} MHz"


def _fmt_ns(ns: float) -> str:
    if abs(ns) >= 1000:
        return f"{ns / 1000:.3g} µs"
    return f"{ns:.3g} ns"


# Lanes that sit between the controllers' groups, not in one.
_BETWEEN = ("link", "bus")


def busy_lanes(timeline: Timeline, window: tuple[float, float]) -> set[LaneKey]:
    """The lanes with a pulse, mark or call in *window* (ns)."""
    a, b = window
    keys = {pl.lane for pl in timeline.pulses if pl.end_ns >= a and pl.start_ns <= b}
    keys |= {mk.lane for mk in timeline.marks if a <= mk.t_ns <= b}
    keys |= {c.lane for c in timeline.calls if c.due_ns >= a and c.submit_ns <= b}
    keys |= {f.lane for f in timeline.forwards if f.use[1] >= a and f.source[1] <= b}
    return keys


def layout(
    timeline: Timeline,
    *,
    lane_in: float = 0.5,
    link_in: float = 0.36,
    gap_in: float = 0.2,
    only: set[LaneKey] | None = None,
) -> tuple[list[Lane], dict[LaneKey, tuple[float, float]], float]:
    """The lanes top to bottom (those in *only*, if given), each lane's
    (baseline, height) in inches, and the total.

    Lanes are grouped by controller. A single group has its links below it;
    with several, the links sit between the first group and the rest.
    """
    shown = [ln for ln in timeline.lanes if only is None or ln.key in only]
    groups = list(dict.fromkeys(ln.group for ln in shown if ln.role not in _BETWEEN))
    links = [ln for ln in shown if ln.role in _BETWEEN]
    order: list[Lane] = []
    for i, g in enumerate(groups):
        order += [ln for ln in shown if ln.role not in _BETWEEN and ln.group == g]
        if i == 0 and len(groups) > 1:
            order += links
    if len(groups) <= 1:
        order += links
    y, ys = 0.0, {}
    for k, ln in enumerate(order):
        if k and len(groups) > 1 and ln.role not in _BETWEEN and ln.group != order[k - 1].group:
            y -= gap_in
        y -= link_in if ln.role in _BETWEEN else lane_in
        ys[ln.key] = (y, link_in if ln.role in _BETWEEN else lane_in)
    return order, ys, -y


def draw_on(
    ax,
    timeline: Timeline,
    p: Palette,
    *,
    lanes=None,
    syncs: str = "groups",
    call_labels: dict[str, str] | None = None,
    pulse_labels: dict[tuple[LaneKey, float], str] | None = None,
    xlim: tuple[float, float] | None = None,
    breakdown: bool = False,
    lane_labels: bool = True,
    cuts: tuple[Cut, ...] = (),
    xlabel: bool = False,
):
    """Draw *timeline* on *ax* in the theme of *p*; see ``draw``."""
    order, ys, total = lanes or layout(timeline)
    ax.set_ylim(-total, 0.12)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(p.rule)
    ax.set_yticks([])
    ax.grid(axis="x", color=p.grid, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    end = max([pl.end_ns for pl in timeline.pulses] + [c.due_ns for c in timeline.calls] + [1.0])
    x0, x1 = xlim or (0, end * 1.01)
    ax.set_xlim(x0, x1)
    fig = ax.figure
    ns_per_pt = (x1 - x0) / (fig.get_figwidth() * 72 * ax.get_position().width)
    unit = _time_axis(ax, p, cuts, (x0, x1))
    if xlabel:
        ax.set_xlabel(f"time ({unit})", color=p.muted, loc="right", labelpad=1)

    def visible(a: float, b: float) -> bool:
        return b >= x0 and a <= x1

    taken: dict[tuple, list[tuple[float, float]]] = {}
    labelled: set[tuple] = set()

    def place(key, x, text, size):
        """The x of *text* centred near *x*, nudged right of labels already on row *key*."""
        w = len(text) * size * 0.56 * ns_per_pt
        a = max(x - w / 2, x0)
        for lo, hi in sorted(taken.get(key, [])):
            if a < hi + 3 * ns_per_pt and a + w > lo:
                a = hi + 3 * ns_per_pt
        taken.setdefault(key, []).append((a, a + w))
        return a + w / 2

    group_of = {ln.key: ln.group for ln in timeline.lanes}
    groups = list(dict.fromkeys(ln.group for ln in order if ln.role not in _BETWEEN))
    for ln in order:
        base, h = ys[ln.key]
        ax.axhline(base, color=p.grid, lw=0.8, zorder=0)
        if lane_labels:
            color = {"link": p.classical, "bus": p.readout}.get(ln.role, p.text)
            ax.text(
                -0.015,
                base + h * 0.3,
                ln.label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8.5,
                color=color,
            )
    if lane_labels and len(groups) > 1:
        for g in groups:
            top = max(ys[ln.key][0] + ys[ln.key][1] for ln in order if ln.group == g)
            ax.text(
                -0.015,
                top - 0.02,
                g,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="top",
                fontsize=8,
                color=p.muted,
                fontweight="bold",
            )

    # Pulses, each scaled to the largest of its kind.
    peak: dict[str, float] = {}
    for pl in timeline.pulses:
        peak[pl.kind] = max(peak.get(pl.kind, 0.0), float(np.max(np.hypot(pl.i, pl.q))))
    for pl in timeline.pulses:
        if pl.lane not in ys or not visible(pl.start_ns, pl.end_ns):
            continue
        base, h = ys[pl.lane]
        scale = 0.6 * h / (peak[pl.kind] or 1.0)
        t, i, q = np.asarray(pl.t_ns), np.asarray(pl.i), np.asarray(pl.q)
        mag = np.hypot(i, q)
        color = p.drive if pl.kind == "drive" else p.readout
        alpha = p.fill_alpha * (0.5 if pl.conditional else 1.0)
        ax.fill_between(t, base, base + mag * scale, color=color, alpha=alpha, lw=0)
        style = (0, (3, 1.8)) if pl.conditional else "solid"
        if pl.kind == "drive":
            if np.max(np.abs(i)) > 0.02 * peak[pl.kind]:
                ax.plot(t, base + i * scale, color=color, lw=1.4, ls=style)
            if np.max(np.abs(q)) > 0.02 * peak[pl.kind]:
                ax.plot(t, base + q * scale, color=p.q, lw=1.2, ls=(0, (2.4, 1.6)))
        else:
            ax.plot(t, base + mag * scale, color=color, lw=1.4, ls=style)
        key = (pl.lane, round(pl.start_ns, 3))
        label = (pulse_labels or {}).get(key)
        if label and key not in labelled:
            labelled.add(key)
            text, side = (label, "above") if isinstance(label, str) else label
            if side == "above":
                x = place(pl.lane, (max(pl.start_ns, x0) + min(pl.end_ns, x1)) / 2, text, 8)
                ax.text(
                    x, base + h * 0.66, text, ha="center", va="bottom", fontsize=8, color=p.muted
                )
            else:
                gap, width = 5 * ns_per_pt, len(text) * 8 * 0.56 * ns_per_pt
                if side == "right" and pl.end_ns + gap + width > x1:
                    side = "left"
                x, ha = (
                    (pl.end_ns + gap, "left") if side == "right" else (pl.start_ns - gap, "right")
                )
                ax.text(x, base + h * 0.3, text, ha=ha, va="center", fontsize=8, color=p.muted)

    for mk in timeline.marks:
        if mk.lane not in ys or not visible(mk.t_ns, mk.t_ns):
            continue
        base, h = ys[mk.lane]
        ax.plot(
            [mk.t_ns],
            [base],
            marker="D",
            ms=4.5,
            color=p.drive,
            mec="none",
            zorder=4,
            clip_on=False,
        )
        x = place(mk.lane, mk.t_ns, mk.text, 8)
        ax.text(x, base + h * 0.66, mk.text, ha="center", va="bottom", fontsize=8, color=p.drive)

    for w in timeline.waits:
        if w.lane not in ys or not visible(w.start_ns, w.end_ns):
            continue
        base, h = ys[w.lane]
        y = base + h * 0.12
        # A wait set at run time is dashed: the static schedule shows its lowest length.
        style = "solid" if w.min_ns is None else (0, (2, 1.5))
        ax.plot([w.start_ns, w.end_ns], [y, y], color=p.muted, lw=0.9, ls=style)
        for x in (w.start_ns, w.end_ns):
            ax.plot([x, x], [y - h * 0.07, y + h * 0.07], color=p.muted, lw=0.9)
        if w.min_ns is None:
            text = f"wait {_fmt_ns(w.length_ns)}"
            x = place(w.lane, (max(w.start_ns, x0) + min(w.end_ns, x1)) / 2, text, 8)
            ax.text(x, y + h * 0.1, text, ha="center", va="bottom", fontsize=8, color=p.muted)

    for s in timeline.syncs:
        keys = [k for k in s.lanes if k in ys]
        if syncs == "none" or len(keys) < 2 or not visible(s.t_ns, s.t_ns):
            continue
        if syncs == "groups" and len({group_of[k] for k in keys}) < 2:
            continue
        lo = min(ys[k][0] for k in keys)
        hi = max(ys[k][0] + ys[k][1] for k in keys)
        ax.plot([s.t_ns, s.t_ns], [lo, hi], color=p.muted, lw=0.9, ls=(0, (1, 2)), zorder=1)
        ax.text(s.t_ns, hi + 0.02, "sync", ha="center", va="bottom", fontsize=7.5, color=p.muted)

    # Calls: the outcome down to the link, the latency, the answer up to its pulse.
    arrow = dict(
        arrowstyle="-|>,head_length=3.6,head_width=2",
        color=p.classical,
        lw=1.1,
        shrinkA=0,
        shrinkB=0.5,
        zorder=5,
    )
    for c in timeline.calls:
        if c.lane not in ys or not visible(c.submit_ns, c.due_ns):
            continue
        base, h = ys[c.lane]
        lo, hi = base + h * 0.18, base + h * 0.62
        mid = (lo + hi) / 2
        if breakdown:
            # Each share is drawn where the schedule puts it and labelled with what
            # was declared: the scheduler may align the answer a few cycles later.
            feedback = float(c.link.get("feedback_ns", 0.0))
            wait = c.timeout_ns or c.max_latency_ns or (c.due_ns - feedback - c.sent_ns)
            work = f"timeout {_fmt_ns(wait)}" if c.timeout_ns else f"{c.function} ≤ {_fmt_ns(wait)}"
            parts = [
                (c.submit_ns, c.sent_ns, f"link {_fmt_ns(c.sent_ns - c.submit_ns)}", 1.0),
                (c.sent_ns, min(c.sent_ns + wait, c.due_ns - feedback), work, 0.45),
                (c.due_ns - feedback, c.due_ns, f"feedback {_fmt_ns(feedback)}", 1.0),
            ]
            for k, (a, b, text, alpha) in enumerate(parts):
                if b <= a:
                    continue
                ax.add_patch(
                    Rectangle((a, lo), b - a, hi - lo, color=p.classical, alpha=alpha, lw=0)
                )
                if not (x0 <= (a if k == 0 else b if k == 2 else max(a, x0)) <= x1):
                    continue
                if k != 1 and b - a < 2 * ns_per_pt:
                    continue
                # The link's share ends left of the outcome's arrow, the feedback's starts
                # right of the answer's; the function's sits under its bar.
                x, ha = {0: (a - 3 * ns_per_pt, "right"), 2: (b + 4 * ns_per_pt, "left")}.get(
                    k, ((max(a, x0) + min(b, x1)) / 2, "center")
                )
                y, va = (lo - 0.03, "top") if k == 1 else (mid, "center")
                width = len(text) * 8 * 0.56 * ns_per_pt
                if (ha == "right" and x - width < x0) or (ha == "left" and x + width > x1):
                    # No room beside the bar: sit above its end instead.
                    x, ha, y, va = (
                        (a, "left", hi + 0.03, "bottom")
                        if k == 0
                        else (b, "right", hi + 0.03, "bottom")
                    )
                ax.text(x, y, text, ha=ha, va=va, fontsize=8, color=p.classical)
        else:
            ax.add_patch(
                Rectangle(
                    (c.submit_ns, lo),
                    max(c.sent_ns - c.submit_ns, 1.5 * ns_per_pt),
                    hi - lo,
                    color=p.classical,
                    lw=0,
                    zorder=4,
                )
            )
            ax.plot([c.sent_ns, c.due_ns], [mid, mid], color=p.classical, lw=1.0, zorder=3)
            text = (call_labels or {}).get(c.function, _fmt_ns(c.due_ns - c.submit_ns))
            ax.text(
                c.due_ns + 3 * ns_per_pt,
                mid,
                text,
                ha="left",
                va="center",
                fontsize=8,
                color=p.classical,
            )
        ax.plot([c.due_ns, c.due_ns], [base + h * 0.06, base + h * 0.74], color=p.classical, lw=1.3)
        if c.source and c.source[0] in ys:
            sb, _ = ys[c.source[0]]
            ax.add_patch(FancyArrowPatch((c.source[1], sb), (c.submit_ns, hi), **arrow))
        if c.use and c.use[0] in ys:
            ub, _ = ys[c.use[0]]
            start = base + h * 0.74 if ub > base else lo
            ax.add_patch(FancyArrowPatch((c.due_ns, start), (c.use[1], ub), **arrow))

    # Forwarded outcomes: the readout down to the bus, its latencies, then up
    # (or down) to the pulse that branches on it.
    bus_arrow = {**arrow, "color": p.readout}
    for f in timeline.forwards:
        if f.lane not in ys or not visible(f.source[1], f.use[1]):
            continue
        base, h = ys[f.lane]
        lo, hi = base + h * 0.18, base + h * 0.62
        mid = (lo + hi) / 2
        ready = f.source[1] + f.feedback_ns
        parts = [(f.source[1], ready, f"feedback {_fmt_ns(f.feedback_ns)}", 1.0)]
        if f.peer_ns:
            parts.append((ready, ready + f.peer_ns, f"peer {_fmt_ns(f.peer_ns)}", 0.45))
        for a, b, text, alpha in parts:
            ax.add_patch(Rectangle((a, lo), b - a, hi - lo, color=p.readout, alpha=alpha, lw=0))
            width = len(text) * 8 * 0.56 * ns_per_pt
            if b - a > width + 4 * ns_per_pt:
                ax.text(
                    (a + b) / 2, mid, text, ha="center", va="center", fontsize=8, color=p.surface
                )
            else:
                ax.text(
                    (a + b) / 2, lo - 0.03, text, ha="center", va="top", fontsize=8, color=p.readout
                )
        if f.source[0] in ys:
            sb, _ = ys[f.source[0]]
            ax.add_patch(FancyArrowPatch((f.source[1], sb), (f.source[1], hi), **bus_arrow))
        if f.use[0] in ys:
            ub, _ = ys[f.use[0]]
            start = base + h * 0.74 if ub > base else lo
            ax.plot([f.use[1], f.use[1]], [lo, hi], color=p.readout, lw=1.3)
            ax.add_patch(FancyArrowPatch((f.use[1], start), (f.use[1], ub), **bus_arrow))
    return ax


def _nice(span: float) -> float:
    """A step of 1, 2, 2.5 or 5 times a power of ten near *span*."""
    power = 10 ** math.floor(math.log10(span))
    return min((s * power for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - span))


def _time_axis(ax, p: Palette, cuts: tuple[Cut, ...], xlim: tuple[float, float]) -> str:
    """Ticks in program time for an axis drawn with *cuts*, a break mark at each
    cut, and the unit of the tick labels."""
    if not cuts:
        ax.xaxis.set_major_locator(MaxNLocator(7, steps=[1, 2, 2.5, 5, 10]))
        return "ns"
    # Each stretch between cuts keeps its length, shifted left by what the cuts before
    # it removed.
    stretches, start, shift = [], xlim[0], 0.0
    for cut in cuts:
        stretches.append((start, cut.start_ns, shift))
        shift += (cut.end_ns - cut.start_ns) - cut.width_ns
        start = cut.end_ns
    stretches.append((start, xlim[1] + shift, shift))
    longest = max(b - a for a, b, _ in stretches)
    unit, scale = ("µs", 1000.0) if xlim[1] + shift > 20_000 else ("ns", 1.0)
    step = _nice(longest / 3)
    ticks, labels = [], []
    for a, b, s in stretches:
        for k in range(math.ceil(a / step), math.floor(b / step) + 1):
            x = k * step - s
            if not ticks or x - ticks[-1] > 0.06 * (xlim[1] - xlim[0]):
                ticks.append(x)
                labels.append(f"{k * step / scale:g}")
    ax.set_xticks(ticks, labels)
    for cut in cuts:
        mid = cut.at_ns + cut.width_ns / 2
        ax.axvspan(cut.at_ns, cut.at_ns + cut.width_ns, color=p.muted, alpha=0.06, lw=0)
        for dx in (-0.006, 0.006):
            ax.plot(
                [
                    mid + dx * (xlim[1] - xlim[0]) - 0.004 * (xlim[1] - xlim[0]),
                    mid + dx * (xlim[1] - xlim[0]) + 0.004 * (xlim[1] - xlim[0]),
                ],
                [-0.03, 0.03],
                transform=ax.get_xaxis_transform(),
                color=p.muted,
                lw=1.0,
                clip_on=False,
            )
    return unit


def draw_zoomed(
    timeline: Timeline,
    p: Palette,
    window: tuple[float, float],
    *,
    width_in: float = WIDTH_IN,
    zoom_options: dict | None = None,
    **options,
):
    """The whole of *timeline* above a closer look at *window* (ns), with the
    latency of each call in the window broken down."""
    sizes = {k: options.pop(k) for k in ("lane_in", "link_in") if k in options}
    lanes = layout(timeline, **sizes)
    zoomed = layout(timeline, only=busy_lanes(timeline, window), **sizes)
    total, below = lanes[2], zoomed[2]
    gap = 0.62
    height = (total + 0.12) + (below + 0.12) + gap + 0.42
    fig = plt.figure(figsize=(width_in, height))
    left = 1.25 / width_in
    top = fig.add_axes(
        (left, (0.42 + below + 0.12 + gap) / height, 0.985 - left, (total + 0.12) / height)
    )
    low = fig.add_axes((left, 0.42 / height, 0.985 - left, (below + 0.12) / height))
    end = max([pl.end_ns for pl in timeline.pulses] + [c.due_ns for c in timeline.calls])
    draw_on(top, timeline, p, lanes=lanes, **{"xlim": (0, max(end * 1.01, window[1])), **options})
    draw_on(
        low,
        timeline,
        p,
        lanes=zoomed,
        **{
            **options,
            "xlim": window,
            "breakdown": True,
            "xlabel": True,
            "cuts": (),
            **(zoom_options or {}),
        },
    )
    top.axvspan(*window, color=p.muted, alpha=0.08, lw=0, zorder=0)
    for x_top, x_low in ((window[0], 0.0), (window[1], 1.0)):
        fig.add_artist(
            ConnectionPatch(
                (x_top, -total),
                (x_low, 1.0),
                coordsA="data",
                coordsB="axes fraction",
                axesA=top,
                axesB=low,
                color=p.rule,
                lw=0.8,
                ls=(0, (2, 2)),
            )
        )
    return fig


def draw_stack(
    timelines: list[Timeline],
    titles: list[str],
    p: Palette,
    *,
    width_in: float = WIDTH_IN,
    labels: list[dict] | None = None,
    **options,
):
    """Timelines one above the other on one time axis, each under its title, to
    compare schedules of the same program, or to show the steps of one; *labels*
    gives each its own *pulse_labels*, and *xlim* the stretch of time to show
    (by default, all of the longest)."""
    sizes = {k: options.pop(k) for k in ("lane_in", "link_in") if k in options}
    lanes = [layout(t, **sizes) for t in timelines]
    labels = labels or [options.pop("pulse_labels", None)] * len(timelines)
    gap, bottom = 0.42, 0.42
    height = sum(lane[2] + 0.12 for lane in lanes) + gap * len(lanes) + bottom - 0.1
    end = max(max([pl.end_ns for pl in t.pulses] + [1.0]) for t in timelines)
    xlim = options.pop("xlim", None) or (0, end * 1.01)
    fig = plt.figure(figsize=(width_in, height))
    left, y = 1.25 / width_in, height
    rows = zip(timelines, lanes, titles, labels, strict=True)
    for k, (timeline, lane, title, words) in enumerate(rows):
        y -= gap - (0.1 if k == 0 else 0) + lane[2] + 0.12
        ax = fig.add_axes((left, y / height, 0.985 - left, (lane[2] + 0.12) / height))
        last = k == len(timelines) - 1
        draw_on(
            ax,
            timeline,
            p,
            lanes=lane,
            xlim=xlim,
            xlabel=last,
            pulse_labels=words,
            **options,
        )
        ax.set_title(title, loc="left", fontsize=8.5, color=p.text, pad=4)
        if not last:
            ax.tick_params(axis="x", labelbottom=False)
    return fig


def draw(timeline: Timeline, p: Palette, *, width_in: float = WIDTH_IN, **options):
    """A figure of *timeline* in the theme of *p*, a lane per line and per link.

    *syncs* is ``"all"``, ``"groups"`` (only those that tie lanes of
    different controllers, the default) or ``"none"``. *call_labels* names a
    function's latency (by default, its length); *pulse_labels* puts a word
    above the pulse that starts at a time on a lane; *xlim* zooms in;
    *breakdown* splits each call's latency into the link's, the function's
    (or the timeout's) and the feedback's; *only* keeps those lanes, as
    ``busy_lanes`` gives them for a zoomed window.
    """
    lanes = layout(
        timeline, **{k: options.pop(k) for k in ("lane_in", "link_in", "only") if k in options}
    )
    total = lanes[2]
    fig = plt.figure(figsize=(width_in, total + 0.55))
    left = 1.25 / width_in
    ax = fig.add_axes((left, 0.42 / (total + 0.55), 0.985 - left, (total + 0.12) / (total + 0.55)))
    draw_on(ax, timeline, p, lanes=lanes, xlabel=True, **options)
    return fig
