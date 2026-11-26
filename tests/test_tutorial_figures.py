# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every tutorial figure is what its script draws today, and says what its page says.

Each script in ``docs/tutorials/figures`` redraws its figures into a
temporary folder (``$PULSE_FIGURES_DIR``), and each must match the committed
SVG in a normalized form: the parsed elements and their text, with the
numbers in their attributes rounded to one decimal and compared to within a
fraction of a pixel, and matplotlib's ids and metadata left out. A
matplotlib upgrade that moves a point by a hundredth still matches; a
changed schedule, value or label doesn't. Figures lay text out with DejaVu
Sans, which comes with matplotlib, so they draw the same on every machine;
they draw in vector only, as an embedded image's bytes differ by platform.
To update a figure, run its script: ``python docs/tutorials/figures/<script>.py``.
The data a script writes for an animated figure (``<script>.json``) must match
the committed file too, every number to within 1e-3.

A ``themed-figure`` of a page, the fallback figure of a ``toy-flow`` and
the still of a ``pulse-visual`` each have a ``-dark`` twin, both have a
title and a description, and the page's ``:alt:`` is the description.

The scripts of guide and reference pages listed in ``_PAGE_SCRIPTS`` are
rerun the same way, against the figures committed beside each script; none
of their figures embeds an image. The tests of both themes and the alt
text, and that every figure is a ``themed-figure`` and every SVG in a
``figures`` folder is used, read every page of the docs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
_FIGURES = _DOCS / "figures"
# The figure scripts of pages outside the tutorials, each next to its figures.
_PAGE_SCRIPTS = [
    _DOCS.parent / "user_guide" / "figures" / "idle_replacement.py",
    _DOCS.parent / "user_guide" / "figures" / "waveform_envelopes.py",
    _DOCS.parent / "user_guide" / "figures" / "kernels.py",
    _DOCS.parent / "user_guide" / "figures" / "classical_calls.py",
    _DOCS.parent / "user_guide" / "figures" / "frame_phase.py",
    _DOCS.parent / "user_guide" / "figures" / "schedule_policies.py",
    _DOCS.parent / "architecture" / "figures" / "pipeline.py",
    _DOCS.parent / "api" / "figures" / "waveform_spectra.py",
]
_SCRIPTS = sorted(p for p in _FIGURES.glob("*.py") if not p.name.startswith("_")) + _PAGE_SCRIPTS
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")
_REFERENCE = re.compile(r"#[\w-]+")


def normalized(svg: str) -> tuple[list[tuple], list[float]]:
    """The elements of *svg*, in order, as (tag, attributes, text) with every
    number in an attribute taken out, and those numbers rounded to one decimal.
    Ids and metadata are left out."""
    root = ET.fromstring(svg)
    skip = {id(e) for m in root.iter() if m.tag.endswith("metadata") for e in m.iter()}
    shape, numbers = [], []
    for element in root.iter():
        if id(element) in skip:
            continue
        attrs = []
        for key, value in sorted(element.attrib.items()):
            key = key.split("}")[-1]
            if key == "id":
                continue
            value = _REFERENCE.sub("#", value)
            numbers += [round(float(n), 1) for n in _NUMBER.findall(value)]
            attrs.append((key, _NUMBER.sub("0", value)))
        shape.append((element.tag.split("}")[-1], tuple(attrs), (element.text or "").strip()))
    return shape, numbers


def same_figure(a: str, b: str) -> bool:
    """Whether SVGs *a* and *b* draw the same: the same elements and text, and
    every coordinate within 0.2 (a fraction of a pixel)."""
    (shape_a, num_a), (shape_b, num_b) = normalized(a), normalized(b)
    return (
        shape_a == shape_b
        and len(num_a) == len(num_b)
        and all(abs(x - y) <= 0.2 + 1e-9 for x, y in zip(num_a, num_b, strict=True))
    )


def _themed_figures(page: Path):
    """(light file, alt text) of each ``themed-figure`` of *page*, of the
    ``:fallback:`` figure of each ``toy-flow``, and of each ``pulse-visual``'s
    still, ``<scene>.svg`` next to its JSON."""
    lines = page.read_text().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*\.\. (themed-figure|toy-flow|pulse-visual):: (\S+)", line)
        if not m:
            continue
        options, key = {}, None
        for option in lines[i + 1 :]:
            if option.strip().startswith(":"):
                key, _, value = option.strip()[1:].partition(":")
                options[key] = value.strip()
            elif key and option.strip():
                # An option's value may go on over indented lines.
                options[key] += " " + option.strip()
            else:
                break
        if m[1] == "themed-figure":
            light = m[2]
        elif m[1] == "toy-flow":
            light = options.get("fallback")
        else:
            light = str(Path(m[2]).with_name(f"{options.get('scene')}.svg"))
        if light:
            yield page.parent / light, options.get("alt")


def _desc(svg: Path) -> tuple[str | None, str | None]:
    root = ET.fromstring(svg.read_text())
    ns = "{http://www.w3.org/2000/svg}"
    title, desc = root.find(f"{ns}title"), root.find(f"{ns}desc")
    return (title.text if title is not None else None, desc.text if desc is not None else None)


def test_each_themed_figure_has_both_themes_and_its_page_says_what_it_shows():
    if not _DOCS.exists():
        pytest.skip("needs the source tree")
    for page in sorted(_DOCS.parent.rglob("*.rst")):
        for light, alt in _themed_figures(page):
            dark = light.with_name(f"{light.stem}-dark{light.suffix}")
            assert light.exists() and dark.exists(), f"{page.name}: {light.name} needs both files"
            for svg in (light, dark):
                title, desc = _desc(svg)
                assert title and desc, f"{svg.name} has no title or description"
                assert alt == " ".join(desc.split()), (
                    f"{page.name}: the :alt: of {light.name} is not its description; "
                    "run its figure script and copy the description"
                )


def test_every_figure_follows_the_theme_and_every_svg_is_used():
    if not _DOCS.exists():
        pytest.skip("needs the source tree")
    docs = _DOCS.parent
    used = set()
    for page in sorted(docs.rglob("*.rst")):
        text = page.read_text()
        # The site's own art in ``_static`` (the banner) is no figure of a page.
        plain = [
            m[2]
            for m in re.finditer(r"^\s*\.\. (figure|image):: (\S+)", text, re.M)
            if not m[2].startswith("_static/")
        ]
        assert plain == [], f"{page.name} has a figure that isn't a themed-figure"
        for light, _ in _themed_figures(page):
            used |= {light.resolve(), light.with_name(f"{light.stem}-dark{light.suffix}").resolve()}
    drawn = sorted(p for p in docs.rglob("*.svg") if p.parent.name == "figures")
    assert [str(p.relative_to(docs)) for p in drawn if p.resolve() not in used] == []


def test_no_figure_embeds_an_image():
    if not _DOCS.exists():
        pytest.skip("needs the source tree")
    # matplotlib's PNG of an imshow or a colorbar comes out with other bytes
    # on Linux than on macOS, so its figure would match on one only.
    folders = [_FIGURES, *dict.fromkeys(script.parent for script in _PAGE_SCRIPTS)]
    drawn = [p for folder in folders for p in sorted(folder.glob("*.svg"))]
    embedded = [p.name for p in drawn if "<image" in p.read_text()]
    assert embedded == [], "draw these in vector (pcolormesh, solids.set_rasterized(False))"


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="needs matplotlib")
@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.stem)
def test_each_figure_is_what_its_script_draws(script, tmp_path):
    if not _DOCS.exists():
        pytest.skip("needs the source tree")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        "PULSE_FIGURES_DIR": str(tmp_path),
        "MPLBACKEND": "Agg",
    }
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    env.pop("CUDAQ_PULSE_QUANTUM_SYSTEM", None)
    result = subprocess.run(
        [sys.executable, script.name], cwd=script.parent, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-3000:]
    drawn = sorted(tmp_path.glob("*.svg"))
    assert drawn, f"{script.name} drew nothing"
    again = f"run python {script.relative_to(_DOCS.parents[1]).as_posix()}"
    for svg in drawn:
        committed = script.parent / svg.name
        assert committed.exists(), f"{script.name} draws {svg.name}, which is not committed"
        assert same_figure(svg.read_text(), committed.read_text()), (
            f"{svg.name} is not what {script.name} draws now; {again}"
        )
    for data in sorted(tmp_path.glob("*.json")):
        committed = script.parent / data.name
        assert committed.exists(), f"{script.name} writes {data.name}, which is not committed"
        assert same_data(json.loads(data.read_text()), json.loads(committed.read_text())), (
            f"{data.name} is not what {script.name} writes now; {again}"
        )


def same_data(a, b) -> bool:
    """Whether two animated figures' data say the same: the same structure and
    text, and every number within 1e-3 of the other, relative or absolute (a
    simulated share of shots may move in its sixth digit between machines)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-3 * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_data(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_data(x, y) for x, y in zip(a, b, strict=True))
    return a == b
