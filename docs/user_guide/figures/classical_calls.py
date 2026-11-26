# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the classical calls guide.

Usage: python docs/user_guide/figures/classical_calls.py

Needs matplotlib (the ``[viz]`` extra). Writes ``call_submit_take.svg`` and
its ``-dark`` twin next to this file. It runs three blocks of the page as
they are written: the ``decode`` function, the ``feed_forward`` kernel and
the control system with the ``gpu`` link, which compiles the kernel. The
call's submission, its budget and the answer's first use are read from the
program's controller export.
"""

import sys
from pathlib import Path

import cudaq_pulse as pulse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials" / "figures"))
from _style import _page_blocks, save_themed  # noqa: E402
from _timeline import Timeline, busy_lanes, draw_zoomed, layout  # noqa: E402

PAGE = HERE.parent / "classical_calls.rst"


def page_namespace() -> dict:
    """The page's function, kernel and control system; the last block compiles
    ``kernel`` for ``q`` of ``qs``, which the page leaves to its reader."""
    blocks = [code for _, code, _ in _page_blocks()(PAGE)]

    def block(text: str) -> str:
        """The first block of the page with *text*."""
        return next(code for code in blocks if text in code)

    namespace: dict = {"__name__": "guide"}
    exec(block("def decode(syndrome: pulse.Bits)"), namespace)
    exec(block("def feed_forward(q)"), namespace)
    qs = pulse.quantum_system("transmon_surface17")
    namespace.update(kernel=namespace["feed_forward"], qs=qs, q=qs.qudits[0])
    exec(block("control_system = pulse.ControlSystem("), namespace)
    return namespace


def call_submit_take(page) -> None:
    scheduled = page["scheduled"]
    timeline = Timeline.of(scheduled)
    (call,) = timeline.calls
    (report,) = scheduled.classical_report()
    clock = scheduled.quantum_system.clock_ghz
    assert call.submit_ns == report.submit_cycles / clock
    assert call.due_ns == report.due_cycles / clock == report.first_use_cycles / clock
    (probe,) = (pl for pl in timeline.pulses if pl.kind == "readout")
    assert call.submit_ns == probe.end_ns
    drives = [pl for pl in timeline.pulses if pl.kind == "drive"]
    assert drives and all(pl.conditional and pl.start_ns == call.due_ns for pl in drives)
    feedback = call.link["feedback_ns"]
    assert call.timeout_ns and call.due_ns == call.sent_ns + call.timeout_ns + feedback

    def at(pl):
        return (pl.lane, round(pl.start_ns, 3))

    words = {at(probe): "readout", at(drives[0]): ("correction", "left")}

    # The whole call above, and a closer look at its end: the feedback and the first use.
    window = (call.due_ns - 3 * feedback, drives[0].end_ns + feedback)

    def figure(p):
        fig = draw_zoomed(timeline, p, window, pulse_labels=words, syncs="none")
        # The timeout's label goes on its bar, clear of the time axis below it.
        low = fig.axes[1]
        (label,) = (t for t in low.texts if t.get_text().startswith("timeout "))
        base, h = layout(timeline, only=busy_lanes(timeline, window))[1][call.lane]
        label.set_position(((window[0] + call.due_ns - feedback) / 2, base + h * 0.4))
        label.set_verticalalignment("center")
        label.set_color(p.text)
        return fig

    save_themed(
        figure,
        "call_submit_take",
        title="A classical call from its submission to its answer",
        desc=(
            "Timeline of feed_forward on a drive line, a readout line and the "
            f"{call.link['name']} link. The readout ends at {probe.end_ns:g} ns and decode is "
            f"submitted then; the request holds the link for "
            f"{call.sent_ns - call.submit_ns:g} ns, the answer has a timeout of "
            f"{call.timeout_ns:g} ns and {feedback:g} ns of feedback, and it is due at "
            f"{call.due_ns:g} ns, {call.due_ns - call.submit_ns:g} ns after its submission, "
            "where the correction pulse starts, dashed because it plays only on some "
            "outcomes. A second "
            f"panel shows {window[0]:g} to {window[1]:g} ns closer: the end of the timeout, "
            "the feedback and the pulse."
        ),
        directory=HERE,
    )


def main() -> None:
    call_submit_take(page_namespace())


if __name__ == "__main__":
    main()
