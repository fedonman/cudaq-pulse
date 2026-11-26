# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the idle-replacement figure of the compilation guide.

Usage: python docs/user_guide/figures/idle_replacement.py

Needs matplotlib (the ``[viz]`` extra). Writes ``idle_window_fill.svg`` and
its ``-dark`` twin next to this file. It runs the page's own code block (the
``syndrome`` kernel, the ``echo`` fill and ``replace_idle``) and draws the
schedule before and after from the two programs' controller exports; the
shaded window is the row of ``program.idle_windows``.
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials" / "figures"))
from _style import _page_blocks, save_themed  # noqa: E402
from _timeline import Timeline, draw_stack, layout  # noqa: E402

PAGE = HERE.parent / "compilation.rst"


def page_namespace() -> dict:
    """The names of the page's idle-replacement block, run as the page has it."""
    (code,) = (
        code
        for _, code, _ in _page_blocks()(PAGE)
        if "def syndrome(" in code and "replace_idle(" in code
    )
    namespace: dict = {"__name__": "guide"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)
    return namespace


def idle_window_fill(page) -> None:
    qs, data, ancilla = page["qs"], page["data"], page["ancilla"]
    before = Timeline.of(page["compiled"].lower(data, ancilla))
    after = Timeline.of(page["program"])
    (window,) = page["program"].idle_windows
    assert window.filled and "echo" in window.result
    start, end = window.start / qs.clock_ghz, window.end / qs.clock_ghz
    name = qs.physical_qudits[window.qudit].label
    lane = next(ln.key for ln in before.lanes if ln.label == f"{name} drive")

    def at(pl):
        return (pl.lane, round(pl.start_ns, 3))

    def in_window(pl) -> bool:
        return pl.lane == lane and start <= pl.start_ns and pl.end_ns <= end

    def words(timeline):
        kept = {}
        for pl in timeline.pulses:
            if pl.kind == "readout":
                kept[at(pl)] = "readout"
            elif pl.lane != lane:
                kept[at(pl)] = "X"
            else:
                kept[at(pl)] = "echo" if in_window(pl) else "X90"
        return kept

    # The fill is the one pulse in the window; every other pulse is where it was.
    (fill,) = (pl for pl in after.pulses if in_window(pl))
    assert not any(in_window(pl) for pl in before.pulses)
    assert [at(pl) for pl in before.pulses] == [at(pl) for pl in after.pulses if pl is not fill]
    # What the description says: the readout spans the window, and waits fill its ends.
    (probe,) = (pl for pl in after.pulses if pl.kind == "readout")
    assert (probe.start_ns, probe.end_ns) == (start, end)
    assert sorted((w.start_ns, w.end_ns) for w in after.waits if w.lane == lane) == [
        (start, fill.start_ns),
        (fill.end_ns, end),
    ]

    def figure(p):
        timelines = [before, after]
        fig = draw_stack(
            timelines,
            ["compiled", "after replace_idle"],
            p,
            labels=[words(t) for t in timelines],
            syncs="all",
        )
        for ax, timeline in zip(fig.axes, timelines, strict=True):
            base, h = layout(timeline)[1][lane]
            ax.add_patch(
                Rectangle((start, base), end - start, h * 0.92, color=p.muted, alpha=0.13, lw=0)
            )
        base, h = layout(before)[1][lane]
        fig.axes[0].text(
            (start + end) / 2,
            base + h * 0.36,
            f"idle window, {end - start:g} ns",
            ha="center",
            va="center",
            fontsize=8,
            color=p.text,
        )
        return fig

    # The helper draws a pulse's Q where it is over 2% of the largest drive: say which.
    peak = max(float(np.max(np.hypot(pl.i, pl.q))) for pl in after.pulses if pl.kind == "drive")
    with_q = list(
        dict.fromkeys(
            words(after)[at(pl)]
            for pl in after.pulses
            if pl.kind == "drive" and float(np.max(np.abs(pl.q))) > 0.02 * peak
        )
    )
    assert with_q == ["X", "echo"], with_q
    alt = (
        f"Two timelines of the syndrome kernel on {name}'s drive line and "
        f"{qs.physical_qudits[ancilla.index].label}'s drive and readout lines. As compiled, "
        f"{name} has an X90 pulse, then an idle window from {start:g} to {end:g} ns while the "
        f"ancilla is read out, then a second X90 pulse. After replace_idle, the window holds a "
        f"wait, the echo pulse from {fill.start_ns:g} to {fill.end_ns:g} ns and a wait, and "
        f"every other pulse starts where it did. The {' and '.join(with_q)} pulses have a "
        "small Q quadrature, drawn dashed."
    )
    save_themed(
        figure,
        "idle_window_fill",
        title="An idle window before and after its fill",
        desc=alt,
        directory=HERE,
    )


def main() -> None:
    idle_window_fill(page_namespace())


if __name__ == "__main__":
    main()
