# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Lowering to Your Controller's Instruction Set" tutorial.

Usage: python docs/tutorials/figures/isa_lowering.py

Needs matplotlib (the ``[viz]`` extra). Writes ``toy_qpu_wiring.svg`` and
``ramsey_reset_schedule.svg``, each with its ``-dark`` twin, next to this
file. The first draws the QPU the page builds, from what ``qpu.as_dict()``
saves: the qudits, each controller's lines, the latency between controllers
and the decoder's link. The second draws the program the page compiles and
lowers for that QPU: every pulse, its frequency, the wait and the feedback
latency are read from the program's controller export.
"""

from dataclasses import replace

import matplotlib.pyplot as plt
from _style import WIDTH_IN, page_namespace, save_themed
from _timeline import Timeline, draw, layout
from matplotlib.patches import FancyBboxPatch


def toy_qpu_wiring(page) -> None:
    saved = page["qpu"].as_dict()
    system = saved["control_system"]
    controllers = system["controllers"]
    qudits = {q["index"]: q for q in saved["qudits"]}
    ((link_name, link),) = system["links"].items()
    (peer,) = {c["peer_latency_cycles"] for c in controllers}
    # The drawing is laid out for this QPU: two controllers, one readout line each.
    assert len(controllers) == 2 and all(len(c["readout_lines"]) == 1 for c in controllers)

    def clock(ghz: float) -> str:
        return f"{ghz:g} GHz" if ghz >= 1 else f"{ghz * 1e3:g} MHz"

    (model,) = {c["model"] for c in controllers}

    def read_by(line) -> list[int]:
        """The qudits a readout line reads: one, or a feedline's several."""
        return list(line["qudits"]) if isinstance(line, dict) else [line]

    wiring = [
        (list(c["drive_lines"]), [read_by(line) for line in c["readout_lines"]])
        for c in controllers
    ]
    # Every qudit is driven by one controller and read by one line.
    assert sorted(q for drives, _ in wiring for q in drives) == sorted(qudits)
    assert sorted(q for _, reads in wiring for line in reads for q in line) == sorted(qudits)

    # Qudits top to bottom on the right; each controller level with the middle of its own.
    qudit_y = {index: 1.0 - k for k, index in enumerate(sorted(qudits))}
    box_w, box_h, box_x, qudit_x = 1.5, 0.62, 1.9, 5.6

    def figure(p):
        fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.2))
        ax.axis("off")
        ax.set_xlim(-0.2, 6.9)
        ax.set_ylim(min(qudit_y.values()) - 0.55, 1.75)
        centres = []
        for number, (drives, reads) in enumerate(wiring):
            own = [qudit_y[q] for q in drives]
            y = sum(own) / len(own)
            centres.append(y)
            ax.add_patch(
                FancyBboxPatch(
                    (box_x, y - box_h / 2),
                    box_w,
                    box_h,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    fc="none",
                    ec=p.rule,
                    lw=1.2,
                )
            )
            ax.text(box_x + box_w / 2, y + 0.1, f"controller {number}", ha="center", color=p.text)
            ax.text(box_x + box_w / 2, y - 0.16, model, ha="center", fontsize=8, color=p.muted)
            right = box_x + box_w + 0.03
            fork = right + 1.55

            def side(q: int, y: float = y) -> float:
                """Where the drive line of qudit *q* meets it: on the side away from
                the controller's middle, so the readout lines run between the drives."""
                return 0.1 if qudit_y[q] >= y else -0.1

            for q in drives:
                ax.plot(
                    [right, qudit_x - 0.3],
                    [y + 1.5 * side(q), qudit_y[q] + side(q)],
                    color=p.drive,
                    lw=1.5,
                )
            dashes = dict(color=p.readout, lw=1.5, ls=(0, (4, 2)))
            lone = len(drives) == 1
            level = y - 0.15 if lone else y
            for line in reads:
                # A feedline leaves the controller once and forks to its qudits.
                ax.plot([right, fork], [level, level], **dashes)
                for q in line:
                    ax.plot([fork, qudit_x - 0.3], [level, qudit_y[q] - side(q)], **dashes)
            first = drives[0]
            ax.text(
                qudit_x - 0.5,
                qudit_y[first] + side(first) + 0.07,
                "drive",
                color=p.drive,
                fontsize=8,
                ha="right",
                va="bottom",
            )
            many = [line for line in reads if len(line) > 1]
            word = "one readout feedline" if many else "readout"
            ax.text(
                fork - 0.05,
                level + (-0.05 if lone else 0.04),
                word,
                color=p.readout,
                fontsize=8,
                ha="right",
                va="top" if lone else "bottom",
            )
        for index, y in qudit_y.items():
            ax.plot([qudit_x], [y], "o", ms=30, mfc="none", mec=p.text, mew=1.4)
            ax.text(qudit_x, y, str(index), ha="center", va="center", color=p.text)
            ax.text(
                qudit_x + 0.42,
                y,
                f"qudit {index}\n{qudits[index]['frequency_hz'] / 1e9:g} GHz",
                va="center",
                fontsize=8,
                color=p.muted,
            )
        # Outcomes between the controllers.
        x = box_x + box_w / 2
        ax.annotate(
            "",
            (x, centres[1] + box_h / 2 + 0.03),
            (x, centres[0] - box_h / 2 - 0.03),
            arrowprops=dict(arrowstyle="<|-|>", color=p.muted, lw=1.0),
        )
        ax.text(
            x + 0.1,
            (centres[0] + centres[1]) / 2,
            f"outcomes: {peer} cycles",
            va="center",
            fontsize=8,
            color=p.muted,
        )
        # The link to the classical function, shared by the control system.
        mid = (centres[0] + centres[-1]) / 2
        ax.add_patch(
            FancyBboxPatch(
                (-0.1, mid - 0.3),
                1.05,
                0.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc="none",
                ec=p.classical,
                lw=1.2,
            )
        )
        ax.text(0.425, mid, f"{link_name}\nlink", ha="center", va="center", color=p.classical)
        for y in centres:
            ax.plot([0.98, box_x - 0.03], [mid, y], color=p.classical, lw=1.2)
        ax.text(
            0.425,
            mid - 0.42,
            f"{link['fixed_ns']:g} ns out,\n{link['feedback_ns']:g} ns back",
            ha="center",
            va="top",
            fontsize=8,
            color=p.classical,
        )
        ax.text(
            box_x + box_w / 2,
            1.6,
            f"reference clock {clock(system['reference_clock_ghz'])}, "
            f"program clock {clock(saved['clock_ghz'])}",
            ha="center",
            fontsize=8,
            color=p.muted,
        )
        return fig

    def lines_of(number: int) -> str:
        drives, reads = wiring[number]
        read = "; ".join(
            f"one feedline reads qudits {' and '.join(map(str, line))}"
            if len(line) > 1
            else f"a readout line reads qudit {line[0]}"
            for line in reads
        )
        return (
            f"controller {number} drives qudit{'s' if len(drives) > 1 else ''} "
            f"{' and '.join(map(str, drives))}, and {read}"
        )

    frequencies = ", ".join(f"{qudits[i]['frequency_hz'] / 1e9:g}" for i in sorted(qudits))
    save_themed(
        figure,
        "toy_qpu_wiring",
        title="The toy QPU: qudits, controllers, lines and the decoder link",
        desc=(
            f"Diagram of the toy QPU. {len(qudits)} qudits at {frequencies} GHz and "
            f"{len(controllers)} {model} controllers: {lines_of(0)}; {lines_of(1)}. An outcome "
            f"takes {peer} cycles from one controller to the other. The {link_name} link of "
            f"the control system, drawn to both controllers, takes {link['fixed_ns']:g} ns out "
            f"and {link['feedback_ns']:g} ns back. The reference clock is "
            f"{clock(system['reference_clock_ghz'])} and the program clock "
            f"{clock(saved['clock_ghz'])}."
        ),
    )


def ramsey_reset_schedule(page) -> None:
    scheduled = page["scheduled"]
    # Each pulse of the export carries its absolute frequency, so the figure labels
    # the detuned pulse with it and leaves out the frame updates between them.
    timeline = replace(Timeline.of(scheduled), marks=())
    first, second, flip = (pl for pl in timeline.pulses if pl.kind == "drive")
    (probe,) = (pl for pl in timeline.pulses if pl.kind == "readout")
    (wait,) = timeline.waits
    (forward,) = timeline.forwards
    assert flip.conditional and not (first.conditional or second.conditional)

    clock = scheduled.quantum_system.clock_ghz
    program = scheduled.export_to_controller()
    (line,) = (ln for ln in program["lines"] if ("line", ln["line"]) == first.lane)
    ghz = {
        (("line", event["line"]), round(event["start_cycles"] / clock, 3)): event["frequency_hz"]
        / 1e9
        for event in program["events"]
        if event["kind"] == "drive"
    }

    def at(pl):
        return (pl.lane, round(pl.start_ns, 3))

    def detuning(pl) -> str:
        return f"{ghz[at(pl)] * 1e3 - line['native_frequency_hz'] / 1e6:+g} MHz"

    assert detuning(first) == detuning(flip) == "+0 MHz"
    words = {
        at(first): "x90",
        at(second): f"x90, {detuning(second)}",
        at(probe): "probe",
        at(flip): ("x180 if it read 1", "left"),
    }

    def figure(p):
        fig = draw(timeline, p, pulse_labels=words, syncs="none")
        (ax,) = fig.axes
        # The latency's label goes left of its bar, clear of the time axis below it,
        # in the bar's color: the helper draws it in the page's color when it fits inside.
        (label,) = (t for t in ax.texts if t.get_text().startswith("feedback "))
        base, h = layout(timeline)[1][forward.lane]
        x0, x1 = ax.get_xlim()
        ns_per_pt = (x1 - x0) / (fig.get_figwidth() * 72 * ax.get_position().width)
        label.set_position((forward.source[1] - 5 * ns_per_pt, base + h * 0.4))
        label.set_horizontalalignment("right")
        label.set_verticalalignment("center")
        label.set_color(p.readout)
        return fig

    alt = (
        "Timeline of the scheduled program on qudit 0's drive and readout lines, with a lane "
        f"for the outcome. An x90 pulse from {first.start_ns:g} to {first.end_ns:g} ns, a "
        f"{wait.length_ns:g} ns wait, a second x90 pulse detuned by {detuning(second)} from "
        f"{second.start_ns:g} to {second.end_ns:g} ns, then the readout from "
        f"{probe.start_ns:g} to {probe.end_ns:g} ns. The outcome is ready "
        f"{forward.feedback_ns:g} ns after the readout ends, and the x180 pulse, which plays "
        f"only if the readout read 1, runs from {flip.start_ns:g} to {flip.end_ns:g} ns."
    )
    save_themed(
        figure,
        "ramsey_reset_schedule",
        title="The schedule of the Ramsey sequence and its reset",
        desc=alt,
    )


def main() -> None:
    # The page's blocks up to the export: the QPU, the kernel and its compile.
    page = page_namespace(
        "isa_lowering.rst", stop_before="program = scheduled.export_to_controller()"
    )
    toy_qpu_wiring(page)
    ramsey_reset_schedule(page)


if __name__ == "__main__":
    main()
