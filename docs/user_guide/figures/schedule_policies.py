# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the schedule-policy figure of the compilation guide.

Usage: python docs/user_guide/figures/schedule_policies.py

Needs matplotlib (the ``[viz]`` extra). Writes ``schedule_policies.svg`` and
its ``-dark`` twin next to this file: one kernel compiled under ``asap``,
``alap``, ``rcp`` and ``alap_rcp`` for a control system with two drive
lanes, each schedule read from its controller export. The page has no
kernel on which the four differ, so the kernel is the one below.
"""

import sys
from pathlib import Path

import cudaq_pulse as pulse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials" / "figures"))
from _style import save_themed  # noqa: E402
from _timeline import Timeline, draw_stack  # noqa: E402

POLICIES = ("asap", "alap", "rcp", "alap_rcp")
LANES = 2
QUDITS = ("D1", "D2", "D3")


@pulse.kernel
def three_lines(a, b, c):
    drive_a, drive_b, drive_c = get_drive_line(a), get_drive_line(b), get_drive_line(c)
    drive(drive_a, drive_a.get_calibration(pulse.gates.X))
    drive(drive_b, drive_b.get_calibration(pulse.gates.X))
    drive(drive_c, drive_c.get_calibration(pulse.gates.X))
    drive(drive_c, drive_c.get_calibration(pulse.gates.X))
    sync(drive_a, drive_b, drive_c)


def schedule_policies() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    control_system = pulse.ControlSystem(max_concurrent_drives=LANES)
    qudits = [qs.qudit(name) for name in QUDITS]
    timelines = [
        Timeline.of(
            pulse.compile(
                three_lines, quantum_system=qs, control_system=control_system, schedule=policy
            ).lower(*qudits)
        )
        for policy in POLICIES
    ]

    def drives(timeline):
        return sorted(
            (pl for pl in timeline.pulses if pl.kind == "drive"),
            key=lambda pl: (pl.lane, pl.start_ns),
        )

    def starts(timeline) -> tuple:
        return tuple(pl.start_ns for pl in drives(timeline))

    def most_at_once(timeline) -> int:
        return max(
            sum(other.start_ns <= pl.start_ns < other.end_ns for other in drives(timeline))
            for pl in drives(timeline)
        )

    # The four schedules differ, and only the RCP ones keep to the lanes.
    assert len({starts(t) for t in timelines}) == len(POLICIES)
    at_once = dict(zip(POLICIES, map(most_at_once, timelines), strict=True))
    assert at_once["rcp"] <= LANES and at_once["alap_rcp"] <= LANES
    assert at_once["asap"] > LANES and at_once["alap"] > LANES
    ends = [max(pl.end_ns for pl in drives(t)) for t in timelines]

    def figure(p):
        return draw_stack(
            timelines,
            [f'schedule="{policy}"' for policy in POLICIES],
            p,
            syncs="none",
            lane_in=0.34,
        )

    def told(policy, timeline, end) -> str:
        by_line = {}
        for pl in drives(timeline):
            by_line.setdefault(pl.lane, []).append(f"{pl.start_ns:g}")
        names = {ln.key: ln.label.removesuffix(" drive") for ln in timeline.lanes}
        assert sorted(names[lane] for lane in by_line) == sorted(QUDITS)
        lines = ", ".join(
            f"{name} at {' and '.join(by_line[lane])} ns"
            for name in QUDITS
            for lane in by_line
            if names[lane] == name
        )
        return f"{policy}: {lines}, done at {end:g} ns"

    save_themed(
        figure,
        "schedule_policies",
        title="One kernel under the four schedule policies",
        desc=(
            f"Four timelines of one kernel on the drive lines of {', '.join(QUDITS[:-1])} and "
            f"{QUDITS[-1]}: an X pulse on {QUDITS[0]}, one on {QUDITS[1]} and two on "
            f"{QUDITS[2]}, then a sync of the three lines. "
            + "; ".join(map(told, POLICIES, timelines, ends))
            + f". asap and alap drive {at_once['asap']} lines at once; rcp and alap_rcp never "
            f"more than {LANES}. Each pulse's Q quadrature is drawn dashed where it is large "
            "enough."
        ),
        directory=HERE,
    )


def main() -> None:
    schedule_policies()


if __name__ == "__main__":
    main()
