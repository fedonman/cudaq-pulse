# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeline visualization for pulse schedules.

Plots a per-line timeline of the drives, readouts and waits of a compiled
kernel using matplotlib.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .._errors import InvalidValueError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..compiler import ScheduledEvent, ScheduledKernel

# The colors of the documentation's figures: each kind of event has one, they
# stay apart under the common color-vision deficiencies, and each is at least
# 4:1 against white. Waits are a hatched neutral.
_DRIVE_COLOR = "#5b8c00"
_READOUT_COLOR = "#c2185b"
_WAIT_COLOR = "#b8bdc2"
_SUBMIT_COLOR = "#00796b"
_TEXT_COLOR = "#5c636a"
_GRID_COLOR = "#e8e8e8"
_CLASSICAL_KINDS = ("submit", "answer")


def _rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def plot_schedule(
    schedule: ScheduledKernel | Sequence[ScheduledEvent],
    figsize: tuple[float, float] = (14, 4),
    title: str | None = None,
) -> Figure:
    """Plot a pulse schedule timeline, one row per drive or readout line.

    The links of classical calls get rows of their own, below the lines: a
    bar while a request is sent and a marker where its answer is taken.

    Args:
        schedule: A compiled kernel call (``kernel.lower(...)``) or its ``schedule``.
        figsize: Figure size (width, height).
        title: Optional figure title.

    Returns:
        matplotlib Figure.
    """
    events = schedule if isinstance(schedule, Sequence) else schedule.schedule
    names = {} if isinstance(schedule, Sequence) else dict(schedule.options.machine.link_names)
    link_label = {v: k for k, v in names.items()}
    # Rows are lines, then the links of classical calls.
    spans: list[tuple[str, tuple[bool, int], int, int]] = []
    for ev in events:
        if ev.start_cycles is None or ev.duration_cycles is None:
            raise InvalidValueError(
                "the module is not scheduled; lower the kernel with its float arguments"
            )
        spans.append(
            (ev.kind, (ev.kind in _CLASSICAL_KINDS, ev.line), ev.start_cycles, ev.duration_cycles)
        )
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    line_ids: list[tuple[bool, int]] = []
    for _, line, _, _ in spans:
        if line not in line_ids:
            line_ids.append(line)
    line_ids.sort(key=lambda row: row[0])

    line_to_row = {lid: i for i, lid in enumerate(line_ids)}
    row_count = max(len(line_ids), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for kind, line, x, w in spans:
        y = line_to_row[line]
        if kind in ("drive", "readout"):
            color = _DRIVE_COLOR if kind == "drive" else _READOUT_COLOR
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y - 0.3), w, 0.6, facecolor=color, edgecolor="none", alpha=0.9
                )
            )
        elif kind == "submit":
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y - 0.3), max(w, 1), 0.6, facecolor=_SUBMIT_COLOR, edgecolor="none"
                )
            )
        elif kind == "answer":
            ax.plot([x], [y], marker="v", color=_SUBMIT_COLOR, markersize=7, markeredgewidth=0)
        elif kind == "wait" and w > 0:
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y - 0.22),
                    w,
                    0.44,
                    facecolor=(*_rgb(_WAIT_COLOR), 0.3),
                    edgecolor=_WAIT_COLOR,
                    linewidth=0,
                    hatch="////",
                )
            )

    ax.set_yticks(range(row_count))
    ax.set_yticklabels(
        [(f"link {link_label.get(n, n)}" if link else f"line {n}") for link, n in line_ids]
    )
    ax.set_xlabel("Time (clock cycles)", color=_TEXT_COLOR)
    ax.set_ylim(-0.5, row_count - 0.5)
    ax.invert_yaxis()

    max_time = max((x + w for _, _, x, w in spans), default=100)
    ax.set_xlim(-max_time * 0.02, max_time * 1.05)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_WAIT_COLOR)
    ax.tick_params(axis="y", length=0, labelcolor="#1a1a1a")
    ax.tick_params(axis="x", color=_WAIT_COLOR, labelcolor=_TEXT_COLOR)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.8)
    for y in range(row_count):
        ax.axhline(y + 0.5, color=_GRID_COLOR, linewidth=0.8, zorder=0)

    if title:
        ax.set_title(title, color="#1a1a1a", pad=18)

    legend_handles = [
        mpatches.Patch(color=_DRIVE_COLOR, label="Drive"),
        mpatches.Patch(color=_READOUT_COLOR, label="Readout"),
        mpatches.Patch(
            facecolor=(*_rgb(_WAIT_COLOR), 0.3), edgecolor=_WAIT_COLOR, label="Wait", hatch="////"
        ),
    ]
    if any(link for link, _ in line_ids):
        legend_handles.append(mpatches.Patch(color=_SUBMIT_COLOR, label="Classical call"))
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
        labelcolor=_TEXT_COLOR,
        handlelength=1.2,
        columnspacing=1.2,
    )
    fig.tight_layout()
    return fig


def save_schedule(
    schedule: ScheduledKernel | Sequence[ScheduledEvent],
    path: str | os.PathLike[str],
    *,
    figsize: tuple[float, float] = (14, 4),
    title: str | None = None,
) -> None:
    """Plot a schedule as ``plot_schedule`` does and save it to a file.

    The extension of *path* picks the format (``.png``, ``.svg``, ``.pdf``
    and the others matplotlib writes). The file is written at 150 dpi, and
    the figure is closed afterwards.

    Args:
        schedule: A compiled kernel call (``kernel.lower(...)``) or its ``schedule``.
        path: The file to write.
        figsize: Figure size (width, height).
        title: Optional figure title.
    """
    fig = plot_schedule(schedule, figsize=figsize, title=title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
