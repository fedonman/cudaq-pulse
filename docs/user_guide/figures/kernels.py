# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figures of the kernels guide.

Usage: python docs/user_guide/figures/kernels.py

Needs matplotlib (the ``[viz]`` extra). Writes ``rolled_loop.svg`` and
``runtime_wait.svg``, each with its ``-dark`` twin, next to this file. Both
run the page's own code blocks. The first draws the loop of
``echo_sequence`` as the structured export holds it, one body with its
trips and period, above the schedule it stands for. The second draws three
calls of ``echo`` from the page's sweep, each with its waits as the call
rounds them.
"""

import contextlib
import io
import sys
from dataclasses import replace
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials" / "figures"))
from _style import _page_blocks, save_themed  # noqa: E402
from _timeline import Timeline, draw_stack, layout  # noqa: E402

PAGE = HERE.parent / "kernels.rst"
# The first, eleventh and twenty-first delay of the page's sweep.
SWEEP = "np.linspace(0, 10_000, 1000)"
PICKED = (0, 10, 20)


def page_namespace() -> dict:
    """The names of the page's runnable blocks up to the two kernels drawn here."""
    namespace: dict = {"__name__": "guide", "pulse": pulse}
    wanted = (
        "qs = pulse.quantum_system(",
        "x_duration = ",
        "readout_duration = ",
        "def echo_sequence(",
        "def echo(qubit, tau",
    )
    for classes, code, _ in _page_blocks()(PAGE):
        if "fragment" not in classes and any(text in code for text in wanted):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, namespace)
    return namespace


def rolled_loop(page) -> None:
    qs = page["qs"]
    qudit = qs.qudits[0]
    # The page leaves the gap to the caller: here it is as long as the pulse.
    values = (qudit, page["x_duration"], page["x_duration"], page["x_sigma"])
    rolled = pulse.compile(page["echo_sequence"], quantum_system=qs).lower(*values)
    flat = pulse.compile(page["echo_sequence"], quantum_system=qs, unroll_threshold=100).lower(
        *values
    )
    (loop,) = rolled.export_to_controller(version=2)["events"]
    assert loop["kind"] == "loop"
    trips, period = loop["trips"], loop["period_cycles"] / qs.clock_ghz
    whole = Timeline.of(rolled)
    # Rolled or unrolled, the schedule is the same; the IR is not.
    assert whole == Timeline.of(flat)
    assert "scf.for" in rolled.mlir() and "scf.for" not in flat.mlir()
    end = max(pl.end_ns for pl in whole.pulses)
    assert loop["start_cycles"] == 0 and trips * period >= end
    body = replace(
        whole,
        pulses=tuple(pl for pl in whole.pulses if pl.start_ns < period),
        waits=tuple(w for w in whole.waits if w.start_ns < period),
    )
    assert len(body.pulses) * trips == len(whole.pulses)
    (pulse_ns,) = {pl.end_ns - pl.start_ns for pl in whole.pulses}
    assert body.waits[0].length_ns == pulse_ns
    (lane,) = (ln.key for ln in whole.lanes)

    # In the lower row the gaps speak for the waits; five labels would crowd it.
    rounds = replace(whole, waits=())

    def figure(p):
        fig = draw_stack(
            [body, rounds],
            [
                f"the loop as exported: one body, {trips} trips, a period of {period:g} ns",
                "the schedule it stands for",
            ],
            p,
            syncs="none",
            xlim=(0, trips * period * 1.01),
        )
        top, low = fig.axes
        base, h = layout(body)[1][lane]
        top.add_patch(
            Rectangle(
                (period, base), (trips - 1) * period, h * 0.92, color=p.muted, alpha=0.13, lw=0
            )
        )
        top.text(
            period + (trips - 1) * period / 2,
            base + h * 0.36,
            f"{trips - 1} more trips, one every {period:g} ns",
            ha="center",
            va="center",
            fontsize=8,
            color=p.text,
        )
        for k in range(1, trips):
            low.axvline(k * period, color=p.muted, lw=0.9, ls=(0, (1, 2)), zorder=1)
        return fig

    save_themed(
        figure,
        "rolled_loop",
        title="A rolled loop and the schedule it stands for",
        desc=(
            f"Two timelines of echo_sequence on one drive line. As exported, the loop is one "
            f"body, a {body.pulses[0].end_ns - body.pulses[0].start_ns:g} ns pulse and a "
            f"{body.waits[0].length_ns:g} ns wait, with {trips} trips and a period of "
            f"{period:g} ns. The schedule it stands for repeats that body {trips} times, "
            f"from 0 to {trips * period:g} ns, with a dotted line where each trip starts."
        ),
        directory=HERE,
    )


def runtime_wait(page) -> None:
    assert SWEEP in PAGE.read_text()
    qs, qudit = page["qs"], page["q"]
    delays = eval(SWEEP, {"np": np})[list(PICKED)]
    calls = [
        page["compiled_echo"].lower(
            qudit,
            float(tau),
            page["x_duration"],
            page["x_sigma"],
            page["x_beta"],
            page["readout_duration"],
        )
        for tau in delays
    ]
    assert all(call.rescheduled for call in calls)
    timelines = [Timeline.of(call) for call in calls]

    def at(pl):
        return (pl.lane, round(pl.start_ns, 3))

    words = []
    for timeline in timelines:
        drives = [pl for pl in timeline.pulses if pl.kind == "drive"]
        assert len(drives) == 3
        words.append(
            {at(pl): word for pl, word in zip(drives, ("x90", "x180", "x90"), strict=True)}
        )
    waits = [sorted(w.length_ns for w in timeline.waits) for timeline in timelines]
    # Each wait is half the delay, rounded to a clock cycle; a wait of no length is not drawn.
    for tau, lengths in zip(delays, waits, strict=True):
        cycles = int(np.floor(tau / 2 * qs.clock_ghz + 0.5))
        assert lengths == ([cycles / qs.clock_ghz] * 2 if cycles else []), (tau, lengths)

    # The readout is far longer than the echo: show its start and let it run off the edge.
    starts = [next(pl.start_ns for pl in t.pulses if pl.kind == "readout") for t in timelines]
    shown = (0, max(starts) * 1.25)
    readout_ns = {
        pl.end_ns - pl.start_ns for t in timelines for pl in t.pulses if pl.kind == "readout"
    }
    (readout_ns,) = readout_ns
    assert all(start + readout_ns > shown[1] for start in starts)

    def figure(p):
        return draw_stack(
            timelines,
            [f"tau = {tau:.4g} ns" for tau in delays],
            p,
            labels=words,
            syncs="none",
            xlim=shown,
        )

    told = "; ".join(
        f"tau = {tau:.4g} ns: "
        + (f"two waits of {lengths[0]:g} ns" if lengths else "no wait")
        + f", readout from {start:g} ns"
        for tau, lengths, start in zip(delays, waits, starts, strict=True)
    )
    save_themed(
        figure,
        "runtime_wait",
        title="Three calls of one compiled echo",
        desc=(
            "Three timelines of the echo kernel on a drive and a readout line, one per call "
            "of the compiled kernel: an x90 pulse, a wait, an x180 pulse, a wait, an x90 "
            f"pulse, then the start of the {readout_ns:g} ns readout, which runs past the "
            f"right edge at {shown[1]:g} ns. {told}. The x180 pulse's Q quadrature is drawn "
            "dashed."
        ),
        directory=HERE,
    )


def main() -> None:
    page = page_namespace()
    rolled_loop(page)
    runtime_wait(page)


if __name__ == "__main__":
    main()
