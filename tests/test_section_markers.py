# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A source file's contents block lists its section banners, in order.

A banner is a comment line ``── Title ───``. A file that has a block
starting ``Contents, in file order`` lists every banner's title there, one
per line and in the order of the file, each followed by nothing or by two
spaces and a few words. The test reads the sources as text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CORE = Path(__file__).resolve().parents[1] / "core"
_SUFFIXES = {".py": "#", ".cpp": "//", ".h": "//", ".td": "//"}
_HEAD = "Contents, in file order"
_WIDTH = {"#": 78, "//": 79}

pytestmark = pytest.mark.skipif(not _CORE.is_dir(), reason="needs the source tree")


def _marked() -> list[Path]:
    return sorted(
        path
        for path in _CORE.rglob("*")
        if path.suffix in _SUFFIXES and _HEAD in path.read_text(encoding="utf-8")
    )


def _banners(lines: list[str], mark: str) -> list[tuple[str, int]]:
    banner = re.compile(rf"^ *{re.escape(mark)} ── (.+?) ─+$")
    return [(m.group(1), len(line)) for line in lines if (m := banner.match(line))]


def _contents(lines: list[str], mark: str) -> list[str]:
    start = next(i for i, line in enumerate(lines) if _HEAD in line)
    entry = re.compile(rf"^{re.escape(mark)}  +(\S.*?)(?:  +\S.*)?$")
    titles = []
    for line in lines[start + 1 :]:
        if not line.startswith(mark) or line.startswith(mark + "="):
            break
        if m := entry.match(line):
            titles.append(m.group(1))
    return titles


def test_some_file_has_a_contents_block():
    assert _marked()


@pytest.mark.parametrize("path", _marked(), ids=lambda path: path.name)
def test_the_contents_block_lists_the_banners_in_order(path):
    mark = _SUFFIXES[path.suffix]
    lines = path.read_text(encoding="utf-8").splitlines()
    banners = _banners(lines, mark)
    assert _contents(lines, mark) == [title for title, _ in banners]
    assert {width for _, width in banners} == {_WIDTH[mark]}
