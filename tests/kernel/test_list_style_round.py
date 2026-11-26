# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A surface-code round written with its lines held in lists schedules as the
same round written with each line fetched where it is used.

The round (benchmarks/perf_gate/cases.py's syndrome round, then a readout of
each ancilla and, on its outcome, a reset pulse) on one controller per
qudit: each ancilla's measurement-conditioned if is traced with every line
of the round held in a list, or with none held. Each line's events, in time
order, are the same.
"""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.quantum_systems import PhysicalQudit

_CR = """\
    h = drag(40, 0.25, 10.0, 0.5)
    cr_p = gaussian(160, 0.12, 40.0)
    cr_m = gaussian(160, -0.12, 40.0)
"""


def _sources(d: int) -> tuple[str, str]:
    data = d * d
    ancillas = (d - 1) * (d - 1) + 2 * (d - 1)
    n = data + ancillas
    place = (
        f"        row = a // {d - 1} if a < {(d - 1) * (d - 1)} else -1\n"
        f"        col = a % {d - 1} if a < {(d - 1) * (d - 1)} else a - {(d - 1) * (d - 1)}\n"
    )
    near = f"                if 0 <= r and r < {d} and 0 <= c and c < {d}:\n"
    listed = (
        "def round_(qudits):\n    lines = []\n    tones = []\n    reads = []\n"
        f"    for i in range({n}):\n"
        "        line = get_drive_line(qudits[i])\n        lines.append(line)\n"
        "        tones.append(line.tone)\n"
        f"    for a in range({ancillas}):\n"
        f"        reads.append(get_readout_line(qudits[{data} + a]))\n"
        + _CR
        + f"    for a in range({ancillas}):\n"
        f"        anc = {data} + a\n" + place + "        drive(lines[anc], h, tone=tones[anc])\n"
        "        for dr in range(2):\n            for dc in range(2):\n"
        "                r = row + dr\n                c = col + dc\n"
        + near
        + f"                    q = r * {d} + c\n"
        "                    sync(lines[anc], lines[q])\n"
        "                    drive(lines[anc], cr_p, tone=tones[q])\n"
        "                    drive(lines[anc], cr_m, tone=tones[q])\n"
        "        drive(lines[anc], h, tone=tones[anc])\n"
        f"    for a in range({ancillas}):\n"
        f"        anc = {data} + a\n"
        "        sync(reads[a], lines[anc])\n"
        "        m = readout(reads[a], square(400, 0.05))\n"
        "        if m:\n            drive(lines[anc], h, tone=tones[anc])\n"
    )
    fetched = (
        "def round_(qudits):\n" + _CR + f"    for a in range({ancillas}):\n"
        f"        anc = {data} + a\n        la = get_drive_line(qudits[anc])\n"
        + place
        + "        drive(la, h, tone=la.tone)\n"
        "        for dr in range(2):\n            for dc in range(2):\n"
        "                r = row + dr\n                c = col + dc\n"
        + near
        + f"                    lq = get_drive_line(qudits[r * {d} + c])\n"
        "                    sync(la, lq)\n"
        "                    drive(la, cr_p, tone=lq.tone)\n"
        "                    drive(la, cr_m, tone=lq.tone)\n"
        "        drive(la, h, tone=la.tone)\n"
        f"    for a in range({ancillas}):\n"
        f"        anc = {data} + a\n"
        "        la = get_drive_line(qudits[anc])\n"
        "        ra = get_readout_line(qudits[anc])\n"
        "        sync(ra, la)\n"
        "        m = readout(ra, square(400, 0.05))\n"
        "        if m:\n            drive(la, h, tone=la.tone)\n"
    )
    return listed, fetched


def _timelines(source: str, d: int) -> dict[tuple[int, str], list[tuple[int, int, str]]]:
    n = d * d + (d - 1) * (d - 1) + 2 * (d - 1)
    system = pulse.quantum_system(
        name=f"surface-{d}",
        qudits={
            i: PhysicalQudit(i, 5.0e9 + 1.0e6 * i, 50.0, 30.0, anharmonicity_hz=-300e6)
            for i in range(n)
        },
        control_system=ControlSystem(
            controllers=[Controller(drive_lines=[i], readout_lines=[i]) for i in range(n)]
        ),
    )
    scope = dict(vars(pulse))
    exec(compile(source, f"<round d={d}>", "exec"), scope)
    lowered = pulse.compile(pulse.kernel(scope["round_"]), quantum_system=system).lower(
        system.qudits[0:n]
    )
    lines: dict[tuple[int, str], list[tuple[int, int, str]]] = {}
    for event in lowered.schedule:
        kind = "readout" if event.kind in ("readout", "acquire") else "drive"
        lines.setdefault((event.qudit, kind), []).append(
            (event.start_cycles, event.duration_cycles, event.kind)
        )
    return {key: sorted(events) for key, events in lines.items()}


@pytest.mark.parametrize("d", [5, 11, 21])
def test_a_round_with_its_lines_in_lists_schedules_as_one_without(d):
    listed, fetched = _sources(d)
    assert _timelines(listed, d) == _timelines(fetched, d)
