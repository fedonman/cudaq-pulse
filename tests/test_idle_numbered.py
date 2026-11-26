# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``numbered()``: a program that was not traced (one ``pulse.load`` read, or
one compiled without an idle-replacement rule) has no op numbers, and
``gaps()`` tells a gap's executions and a loop's syncs apart by them.
Numbered by ``numbered()``, its gaps are the ones its traced program has."""

from __future__ import annotations

import cudaq_pulse as pulse
import cudaq_pulse._idle_windows as iw
import pytest

from tests._handles import Q


@pulse.kernel
def sync_in_a_loop(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(4):
        drive(da, square(40, 0.1))
        drive(db, square(80, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))
    drive(db, square(40, 0.1))


@pulse.kernel
def nested_loops(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(3):
        drive(db, square(120, 0.1))
        for _ in range(5):
            drive(da, square(16, 0.1))
            wait(da, 8)
        sync(da, db)


@pulse.kernel
def measured_branch(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    m = readout(get_readout_line(b), square(40, 0.05))
    if m:
        drive(da, square(40, 0.1))
    else:
        wait(da, 80)
    drive(db, square(40, 0.1))
    drive(da, square(40, 0.1))


@pulse.kernel
def if_in_a_loop(a, b):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    for _ in range(4):
        m = readout(rb, square(40, 0.05))
        if m:
            drive(da, square(40, 0.1))
        else:
            wait(da, 80)
        drive(da, square(16, 0.1))


@pulse.kernel
def loop_in_an_arm_in_a_loop(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(3):
        m = readout(rb, square(40, 0.05))
        if m:
            for _ in range(2):
                drive(da, square(16, 0.1))
                wait(da, 8)
        drive(db, square(40, 0.1))
        sync(da, db)


@pulse.kernel
def ramsey(a, b):
    da = get_drive_line(a)
    drive(da, square(40, 0.1))
    wait(da, 400)
    drive(da, square(40, 0.1))
    drive(get_drive_line(b), square(40, 0.1))


def _shape(found):
    return {
        q: [
            (
                g.start,
                g.end,
                g.rounds,
                g.varies,
                g.wraps(),
                g.ordered(),
                tuple(g.idle()),
                tuple(sorted(g.repeats)),
            )
            for g in gs
        ]
        for q, gs in found.items()
    }


def _traced(kernel, monkeypatch):
    """The gaps idle replacement finds in *kernel*'s traced program."""
    seen = []
    real = iw.gaps

    def spy(lines, items):
        found = real(lines, items)
        seen.append(_shape(found))
        return found

    monkeypatch.setattr(iw, "gaps", spy)
    pulse.compile(kernel).replace_idle(pulse.IdleReplacement(None, threshold=0)).lower(Q[0], Q[1])
    monkeypatch.setattr(iw, "gaps", real)
    return seen[0]


_KERNELS = [
    sync_in_a_loop,
    nested_loops,
    measured_branch,
    if_in_a_loop,
    loop_in_an_arm_in_a_loop,
    ramsey,
]


@pytest.mark.parametrize("kernel", _KERNELS)
def test_a_program_without_numbers_has_its_traced_gaps_once_numbered(kernel, monkeypatch):
    lines, items = iw.read(pulse.compile(kernel).lower(Q[0], Q[1])._module)
    assert all(item.site is None for item in items if item.__class__ is not tuple)
    assert _shape(iw.gaps(lines, iw.numbered(items))) == _traced(kernel, monkeypatch)


def test_without_numbers_a_loops_gaps_run_together():
    # What numbering is for: unnumbered, every execution of every gap of a
    # qudit is taken for one gap's.
    lines, items = iw.read(pulse.compile(sync_in_a_loop).lower(Q[0], Q[1])._module)
    assert [len(gs) for gs in iw.gaps(lines, items).values()] == [1, 1]
    assert [len(gs) for gs in iw.gaps(lines, iw.numbered(items)).values()] == [5, 5]


def test_a_program_with_numbers_of_its_own_is_left_as_it_is(monkeypatch):
    traced = []
    real = iw.read

    def spy(module):
        lines, items = real(module)
        traced.append(items)
        return lines, items

    monkeypatch.setattr(iw, "read", spy)
    pulse.compile(sync_in_a_loop).replace_idle(pulse.IdleReplacement(None, threshold=0)).lower(
        Q[0], Q[1]
    )
    items = traced[0]
    sites = [item.site for item in items if item.__class__ is not tuple]
    assert iw.numbered(items) is items
    assert [item.site for item in items if item.__class__ is not tuple] == sites


@pytest.mark.parametrize("kernel", _KERNELS)
def test_numbers_order_the_ops_as_the_trace_does(kernel, monkeypatch):
    # The traced program and the program compiled without the rule hold the
    # same ops in the same order: every pair of them compares the same way
    # by number as by the trace's site.
    traced = []
    real = iw.read

    def spy(module):
        lines, items = real(module)
        traced.append([item for item in items if item.__class__ is not tuple])
        return lines, items

    monkeypatch.setattr(iw, "read", spy)
    pulse.compile(kernel).replace_idle(pulse.IdleReplacement(None, threshold=0)).lower(Q[0], Q[1])
    monkeypatch.setattr(iw, "read", real)
    _, items = iw.read(pulse.compile(kernel).lower(Q[0], Q[1])._module)
    numbered = [item for item in iw.numbered(items) if item.__class__ is not tuple]
    sites = [event.site for event in traced[0]]
    numbers = [event.site for event in numbered]
    # (A sync's start is where its lines meet, which gaps() sets.)
    assert [(e.kind, e.line) for e in traced[0]] == [(e.kind, e.line) for e in numbered]
    for i, (s, n) in enumerate(zip(sites, numbers, strict=True)):
        for t, m in zip(sites[i:], numbers[i:], strict=True):
            assert (s < t, s == t) == (n < m, n == m)
