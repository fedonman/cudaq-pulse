# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A list of lines changed in place, between measurement-conditioned ifs or
in one of their arms, traces as the same list built anew does.

From a trace's first measurement-conditioned if on, an if puts back the
slots its arms wrote in the lists of lines it holds from a log of those
writes, and records afresh only the slots written since the last if. A new
list is one the last if did not hold, so the if records each of its slots,
as every if once did for every list: each change in place below is checked
against the same change made by building a new list.
"""

from __future__ import annotations

import enum
import operator
import random

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q

# Each way code can change a list of lines in place, and the same change made
# by building a new list. `p` is a line the list does not hold.
_WAYS = {
    "reverse": ("lines.reverse()", "lines = lines[::-1]"),
    "sort": (
        "ranks = {line: -i for i, line in enumerate(lines)}\nlines.sort(key=ranks.get)",
        "ranks = {line: -i for i, line in enumerate(lines)}\nlines = sorted(lines, key=ranks.get)",
    ),
    "insert": ("lines.insert(0, p)", "lines = [p] + lines"),
    "pop": ("lines.pop(0)", "lines = lines[1:]"),
    "rotate": ("lines.append(lines.pop(0))", "lines = lines[1:] + lines[:1]"),
    "item": ("lines[1] = p", "lines = [lines[0], p] + lines[2:]"),
    "negative item": ("lines[-1] = p", "lines = lines[:-1] + [p]"),
    "swap": (
        "lines[0], lines[2] = lines[2], lines[0]",
        "lines = [lines[2], lines[1], lines[0]] + lines[3:]",
    ),
    "slice": ("lines[:] = lines[::-1]", "lines = lines[::-1]"),
    "slice at the front": ("lines[:0] = [p]", "lines = [p] + lines"),
    "del": ("del lines[1]", "lines = lines[:1] + lines[2:]"),
    "+=": ("lines += [p]", "lines = lines + [p]"),
    "*=": ("lines *= 2", "lines = lines * 2"),
    "clear and extend": (
        "kept = list(lines)\nlines.clear()\nlines.extend(kept[::-1])",
        "kept = list(lines)\nlines = kept[::-1]",
    ),
    "operator.setitem": ("operator.setitem(lines, 1, p)", "lines = [lines[0], p] + lines[2:]"),
    "shuffled": (
        "random.Random(1).shuffle(lines)",
        "kept = list(lines)\nrandom.Random(1).shuffle(kept)\nlines = kept",
    ),
}

# Where the change is made: between two ifs, or between them in a loop the
# trace unrolls. (In an arm, a list changed in place and a name bound to a new
# list are not the same program: the name is bound on one path only.)
_PLACES = {
    "between ifs": "    {change}\n",
    "in a loop between ifs": "    for _ in range(2):\n        {change}\n",
}

_KERNEL = """\
def k(qudits):
    lines = [get_drive_line(q) for q in qudits[:4]]
    other = lines[1:]
    p = get_drive_line(qudits[4])
    probe = get_readout_line(qudits[0])
    b = readout(probe, square(40, 0.05))
    x = lines[1]
    if b:
        drive(lines[0], square(16, 0.1))
        drive(other[0], square(16, 0.15))
{place}    if b:
        drive(x, square(8, 0.2))
        drive(lines[1], square(8, 0.3))
        drive(lines[-1], square(8, 0.35))
    else:
        wait(x, 8)
    drive(lines[0], square(8, 0.4))
    sync(*other)
"""


def _traced(change: str, place: str) -> str:
    pad = "        " if place != "between ifs" else "    "
    lines = change.split("\n")
    body = _PLACES[place].format(change=("\n" + pad).join(lines))
    scope = dict(vars(pulse), operator=operator, random=random)
    exec(compile(_KERNEL.format(place=body), "<kernel>", "exec"), scope)
    try:
        lowered = pulse.compile(pulse.kernel(scope["k"])).lower(Q[0:5])
    except (CompilationError, RuntimeError) as e:
        return f"error: {type(e).__name__}: {e}"
    return lowered.mlir() + "\n".join(repr(e) for e in lowered.schedule)


@pytest.mark.parametrize("place", list(_PLACES))
@pytest.mark.parametrize("way", list(_WAYS))
def test_a_list_changed_in_place_traces_as_a_new_one(way, place):
    in_place, anew = _WAYS[way]
    traced = _traced(in_place, place)
    assert not traced.startswith("error"), traced
    assert traced == _traced(anew, place)


class _One(enum.IntEnum):
    ONE = 1


@pytest.mark.parametrize("index", [np.int64(1), _One.ONE, True])
def test_a_list_indexed_by_another_integer_type_traces_as_by_an_int(index):
    def traced(i):
        @pulse.kernel
        def k(a, b, c):
            lines = [get_drive_line(q) for q in (a, b, c)]
            m = readout(get_readout_line(a), square(40, 0.05))
            if m:
                drive(lines[0], square(16, 0.1))
            drive(lines[i], square(16, 0.2))
            if m:
                drive(lines[0], square(16, 0.3))
            drive(lines[1], square(16, 0.4))

        lowered = pulse.compile(k).lower(Q[0], Q[1], Q[2])
        return lowered.mlir() + "\n".join(repr(e) for e in lowered.schedule)

    assert traced(index) == traced(1)


_CALLED = """\
def flip(lines, reads, i):
    m = readout(reads[i], square(40, 0.05))
    if m:
        drive(lines[i], square(16, 0.1))
    else:
        wait(lines[i], 16)


def k(qudits):
    lines = [get_drive_line(q) for q in qudits[:4]]
    reads = [get_readout_line(q) for q in qudits[:4]]
    p = get_drive_line(qudits[4])
    flip(lines, reads, 0)
    flip(lines, reads, 1)
{change}    flip(lines, reads, 0)
    flip(lines, reads, 1)
    drive(lines[0], square(8, 0.4))
    drive(lines[1], square(8, 0.45))
"""


@pytest.mark.parametrize("way", ["reverse", "swap", "item", "rotate"])
def test_a_list_changed_between_ifs_in_called_kernels_traces_as_a_new_one(way):
    # The trace's first if is in a called kernel: the ifs after it put back
    # only what their arms wrote once every caller notes what changes.
    def traced(change):
        body = "".join(f"    {line}\n" for line in change.split("\n"))
        scope = dict(vars(pulse))
        exec(compile(_CALLED.format(change=body), "<kernel>", "exec"), scope)
        scope["flip"] = pulse.kernel(scope["flip"])
        lowered = pulse.compile(pulse.kernel(scope["k"])).lower(Q[0:5])
        return lowered.mlir() + "\n".join(repr(e) for e in lowered.schedule)

    in_place, anew = _WAYS[way]
    assert traced(in_place) == traced(anew)
