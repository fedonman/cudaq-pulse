# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The look of the tutorial figures, how they are saved, and the page code they plot.

A figure is drawn twice, once per theme of the documentation, by a function
that takes a ``Palette``. ``save_themed`` writes ``<name>.svg`` for the light
theme and ``<name>-dark.svg`` for the dark one, and the ``themed-figure``
directive (``docs/_ext/themed_figure.py``) shows the one that matches the
reader's theme.

Every color has a role. Drive pulses and the I quadrature are green, the Q
quadrature is blue and dashed, readouts are rose and classical calls teal,
in both themes; a second data series takes the Q blue. The four roles stay
apart under simulated protanopia and deuteranopia (CIE76 distance above 16
in both themes), every one is at least 4:1 against its page, and no figure
tells two things apart by color alone: lanes are labelled, Q is dashed and a
conditional pulse is outlined.
"""

import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE = "#4a7fb5"
GREEN = "#76b900"

# The documentation's font, then what a reader without it sees. An SVG shown
# as an image can't use the page's web fonts. Text is laid out with DejaVu Sans,
# which comes with matplotlib, so a figure is drawn the same on every machine;
# it is a little wider than the fonts that show it, so no label overflows.
FONT_STACK = "NVIDIA, Arial, Helvetica, sans-serif"
# Figures are drawn at the width of the content column, so 9 pt text shows
# at about the size of a caption.
WIDTH_IN = 7.2


@dataclass(frozen=True)
class Palette:
    """The colors of one theme: its page, text and chrome, then one color per role."""

    name: str
    surface: str
    text: str
    muted: str
    grid: str
    rule: str
    drive: str
    q: str
    readout: str
    classical: str
    fill_alpha: float


LIGHT = Palette(
    name="light",
    surface="#ffffff",
    text="#1a1a1a",
    muted="#5c636a",
    grid="#e8e8e8",
    rule="#b8bdc2",
    drive="#5b8c00",
    q="#3d6fa3",
    readout="#c2185b",
    classical="#00796b",
    fill_alpha=0.16,
)
DARK = Palette(
    name="dark",
    surface="#111111",
    text="#eeeeee",
    muted="#b4bac1",
    grid="#2c2f33",
    rule="#5a6066",
    drive="#76b900",
    q="#8fb8e8",
    readout="#f06292",
    classical="#4dd0c4",
    fill_alpha=0.22,
)
THEMES = (LIGHT, DARK)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "cudaq-pulse",
    }
)


def _rc(p: Palette) -> dict:
    return {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": p.text,
        "axes.labelcolor": p.muted,
        "axes.edgecolor": p.rule,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": p.grid,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "xtick.color": p.rule,
        "xtick.labelcolor": p.muted,
        "ytick.color": p.rule,
        "ytick.labelcolor": p.muted,
        "ytick.major.size": 0,
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
    }


def _page_blocks():
    """``code_blocks`` of the tutorial test, so a figure runs a page exactly as the test does."""
    spec = importlib.util.spec_from_file_location(
        "_tutorial_pages", HERE.parents[2] / "tests" / "test_tutorials.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.code_blocks


def page_namespace(page: str, stop_before: str | None = None) -> dict:
    """Run the CPU code blocks of tutorial *page*, as a reader would, and return their
    names; with *stop_before*, only the blocks before the first that contains it."""
    namespace: dict = {"__name__": "tutorial"}
    blocks = _page_blocks()(HERE.parent / page)
    # A page may import the examples it links, as the tutorial test lets it.
    examples = HERE.parents[2] / "examples"
    for linked in re.findall(r":example:`([^`<>]+\.py)`", (HERE.parent / page).read_text()):
        folder = str((examples / linked).parent)
        if folder not in sys.path:
            sys.path.insert(0, folder)
    # Pages write files where they run, as the tutorial test lets them.
    with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
        for classes, code, _ in blocks:
            if stop_before is not None and stop_before in code:
                break
            if not {"fragment", "gpu", "viz", "cudaq-qec"} & classes:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(code, namespace)
    return namespace


def output_dir(directory: Path) -> Path:
    """Where a figure is written: *directory*, or ``$PULSE_FIGURES_DIR`` when it is set
    (the drift test draws every figure there to compare it with the committed one)."""
    return Path(os.environ.get("PULSE_FIGURES_DIR") or directory)


def axes(figsize=(6.4, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def plot_axes(p: Palette, height_in: float = 3.0, width_in: float = WIDTH_IN):
    """A figure and axes in the theme of *p*: a baseline, faint horizontal grid, no box."""
    with plt.rc_context(_rc(p)):
        fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.spines["bottom"].set_color(p.rule)
    return fig, ax


def end_label(ax, x, y, text, color, *, dx=5, dy=0, ha="left", va="center", **kw):
    """A direct label at (*x*, *y*), in place of a legend entry."""
    return ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        ha=ha,
        va=va,
        fontsize=kw.pop("fontsize", 8.5),
        **kw,
    )


_HEADER = (
    "<!--\n"
    "   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES."
    " All rights reserved.\n"
    "   SPDX-License-Identifier: Apache-2.0\n"
    "-->\n"
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fonts(text: str) -> str:
    text = re.sub(r"font-family:[^;\"]*", f"font-family:{FONT_STACK}", text)
    return re.sub(r"(font: [\d.]+px) '[^']*'", rf"\1 {FONT_STACK}", text)


def _svg_text(fig, title: str, desc: str) -> str:
    svg = io.StringIO()
    fig.savefig(
        svg,
        format="svg",
        metadata={"Date": None, "Creator": None},
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=True,
    )
    plt.close(fig)
    text = svg.getvalue()
    text = re.sub(r"<\?xml[^>]*\?>\s*|<!DOCTYPE[^>]*>\s*", "", text)
    text = re.sub(r"\s*<metadata>.*?</metadata>", "", text, flags=re.S)
    # Two decimals of a point are far finer than a screen pixel; labels keep theirs.
    text = re.sub(r"<[^>]+>", lambda tag: re.sub(r"(?<=\d\.\d\d)\d+", "", tag[0]), text)
    text = _fonts(text)
    text = text.replace(
        "<svg ",
        '<svg role="img" aria-labelledby="title desc" ',
        1,
    )
    head = f'\n <title id="title">{_escape(title)}</title>\n <desc id="desc">{_escape(desc)}</desc>'
    text = re.sub(r"(<svg [^>]*>)", lambda m: m[1] + head, text, count=1)
    # The repository's pre-commit hooks reject trailing whitespace.
    return _HEADER + "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def save(fig, name: str, directory: Path = HERE) -> None:
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    # The repository's pre-commit hooks reject trailing whitespace.
    text = "\n".join(line.rstrip() for line in _fonts(svg.getvalue()).splitlines()) + "\n"
    (output_dir(directory) / name).write_text(text)


def save_themed(
    draw: Callable[[Palette], "plt.Figure"],
    name: str,
    *,
    title: str,
    desc: str,
    directory: Path = HERE,
) -> list[Path]:
    """Draw *draw* once per theme and write ``<name>.svg`` and ``<name>-dark.svg``.

    *title* and *desc* become the SVG's accessible name and description; the
    page's ``:alt:`` should say the same as *desc*.
    """
    written = []
    for p in THEMES:
        with plt.rc_context(_rc(p)):
            fig = draw(p)
            path = output_dir(directory) / (f"{name}.svg" if p is LIGHT else f"{name}-dark.svg")
            path.write_text(_svg_text(fig, title, desc))
        written.append(path)
    return written
