# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``viz.plot_schedule`` and ``viz.save_schedule`` draw what was scheduled."""

from __future__ import annotations

from types import SimpleNamespace

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import InvalidValueError, viz

from tests._handles import Q
from tests._tools import import_module

matplotlib = import_module("matplotlib")
matplotlib.use("Agg")
plt = import_module("matplotlib.pyplot", "matplotlib")
_COLORS = {"drive": "#5b8c00", "readout": "#c2185b", "wait": "#b8bdc2"}


@pulse.kernel
def _two_lines(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    r0 = get_readout_line(q0)
    drive(d0, square(40, 0.1))
    wait(d1, 16)
    drive(d1, gaussian(40, 0.1, 10.0))
    sync(d0, r0)
    readout(r0, square(100, 0.05))


@pytest.fixture
def lowered():
    return _two_lines.lower(Q[0], Q[1])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bars(ax):
    """(start, length, row, color) of each drawn event, sorted."""
    return sorted(
        (
            round(patch.get_x()),
            round(patch.get_width()),
            round(patch.get_y() + patch.get_height() / 2),
            matplotlib.colors.to_hex(patch.get_facecolor()),
        )
        for patch in ax.patches
    )


def test_each_line_is_a_row_and_each_event_a_bar(lowered):
    fig = viz.plot_schedule(lowered, title="two lines")
    (ax,) = fig.axes
    rows: dict[int, int] = {}
    for event in lowered.schedule:
        rows.setdefault(event.line, len(rows))
    assert len(rows) == 3
    assert [t.get_text() for t in ax.get_yticklabels()] == [f"line {line}" for line in rows]
    expected = sorted(
        (e.start_cycles, e.duration_cycles, rows[e.line], _COLORS[e.kind]) for e in lowered.schedule
    )
    assert {kind for *_, kind in expected} == set(_COLORS.values())
    assert _bars(ax) == expected
    assert ax.get_title() == "two lines"
    assert ax.get_xlabel() == "Time (clock cycles)"
    end = max(e.start_cycles + e.duration_cycles for e in lowered.schedule)
    assert ax.get_xlim() == pytest.approx((-0.02 * end, 1.05 * end))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Drive", "Readout", "Wait"]


def test_a_schedule_list_plots_like_its_program(lowered):
    from_program = viz.plot_schedule(lowered).axes[0]
    from_list = viz.plot_schedule(list(lowered.schedule)).axes[0]
    assert _bars(from_list) == _bars(from_program)
    assert [t.get_text() for t in from_list.get_yticklabels()] == [
        t.get_text() for t in from_program.get_yticklabels()
    ]


def test_an_empty_schedule_plots_an_empty_axis():
    (ax,) = viz.plot_schedule([]).axes
    assert len(ax.patches) == 0
    assert ax.get_xlim() == pytest.approx((-2.0, 105.0))


def test_an_unscheduled_event_is_an_error():
    event = SimpleNamespace(kind="wait", line=0, start_cycles=None, duration_cycles=None)
    with pytest.raises(InvalidValueError, match="not scheduled"):
        viz.plot_schedule([event])


def test_save_schedule_writes_the_image_and_closes_its_figure(lowered, tmp_path):
    before = plt.get_fignums()
    path = tmp_path / "schedule.png"
    viz.save_schedule(lowered, path, title="saved", figsize=(6, 2))
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert plt.get_fignums() == before
    svg = tmp_path / "schedule.svg"
    viz.save_schedule(lowered, str(svg))
    assert "<svg" in svg.read_text()
    with pytest.raises(TypeError):
        viz.save_schedule(lowered, path, dpi=300)  # type: ignore[call-arg]


def test_timeline_takes_the_keywords_of_plot_schedule(lowered):
    fig = lowered.timeline(figsize=(6, 2), title="two lines")
    assert tuple(fig.get_size_inches()) == (6, 2)
    assert fig.axes[0].get_title() == "two lines"
    compiled = pulse.compile(_two_lines, quantum_system=Q[0].quantum_system)
    with pytest.raises(TypeError):
        lowered.timeline(dpi=100)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        compiled.timeline(dpi=100)  # type: ignore[call-arg]
