# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binding values folds a module with only the canonicalization patterns it
can match: those rooted at an op it holds, closed under the ops each of them
makes (bindings.cpp, ``reachablePatterns``). What a pattern makes is what it
declares, else what ``undeclaredMakes()`` lists for it.

Every pattern must declare what it makes or be listed, or a module would
miss a pattern it can match; and folding with the patterns a module can
match must give what folding with all of them gives
(``CUDAQ_PULSE_FULL_CANONICAL=1``, read once per process, so in a fresh
interpreter each).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cudaq_pulse as pulse
from cudaq_pulse._native import _cudaq_pulse_native as native

from tests._handles import Q

_ROOT = Path(__file__).resolve().parents[1]

# Ops of every dialect whose patterns the canonical set holds: a module
# holding all of them keeps every pattern rooted at one.
_EVERY_ROOT = {
    "scf.if",
    "scf.yield",
    "scf.for",
    "scf.while",
    "scf.condition",
    "scf.execute_region",
    "scf.forall",
    "scf.index_switch",
    "scf.parallel",
    "arith.mulf",
    "arith.cmpf",
    "arith.addf",
    "arith.select",
    "arith.xori",
}


def test_every_canonicalization_pattern_declares_what_it_makes_or_is_listed():
    audit = native._canonical_pattern_audit()
    assert audit["patterns"] > 0
    assert audit["unlisted"] == [], (
        "these canonicalization patterns declare no ops they make and are not "
        "listed in undeclaredMakes() in core/mlir/bindings/bindings.cpp: read "
        f"each one's source and list the ops it makes there: {audit['unlisted']}"
    )
    assert audit["unused_entries"] == [], (
        "these entries of undeclaredMakes() in core/mlir/bindings/bindings.cpp "
        f"match no pattern any more (renamed or removed?): {audit['unused_entries']}"
    )


def test_a_module_keeps_only_the_patterns_its_ops_can_reach():
    drives = ["builtin.module", "func.func", "func.return", "pulse.drive", "arith.constant"]
    assert native._canonical_pattern_audit(drives)["kept"] == []
    branches = [*drives, "scf.if", "scf.yield", "arith.mulf", "arith.cmpf"]
    audit = native._canonical_pattern_audit(branches)
    kept = set(audit["kept"])
    assert len(kept) < audit["patterns"], audit
    # Every pattern rooted at an op the module holds is kept.
    everything = set(native._canonical_pattern_audit(sorted(_EVERY_ROOT))["kept"])
    rooted = {key for key in everything if key.split("|")[0] in branches}
    assert rooted and rooted <= kept, rooted - kept
    # And a loop's are not, as it holds none.
    assert not any(key.startswith("scf.for|") for key in kept), kept


# Kernels whose bound values open folds: loops (of 0, 1 and more trips) and
# ifs (measured, nested, on a bound float, yielding a value) around waits,
# phases and square amplitudes that float arguments set.
_KERNELS = """
import math
import cudaq_pulse as pulse
from tests._handles import Q


@pulse.kernel
def waits_in_a_loop(a, t: float, n: int):
    d = get_drive_line(a)
    for _ in range(n):
        drive(d, square(40, 0.1))
        wait(d, t)


@pulse.kernel
def phases_and_squares(a, phase: float, amp: float):
    d = get_drive_line(a)
    shift_phase(d.tone, phase)
    drive(d, square(40, amp))
    drive(d, square(40, amp))
    shift_phase(d.tone, -phase)
    drive(d, square(40, 0.1))


@pulse.kernel
def measured_if(a, b, amp: float, t: float):
    da = get_drive_line(a)
    m = readout(get_readout_line(b), square(40, 0.05))
    if m:
        drive(da, square(40, amp))
    else:
        wait(da, t)
    drive(da, square(40, amp))


@pulse.kernel
def if_in_a_loop(a, b, amp: float, n: int):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    for _ in range(n):
        m = readout(rb, square(40, 0.05))
        if m:
            drive(da, square(40, amp))
        shift_phase(da.tone, amp * math.pi)


@pulse.kernel
def if_on_a_bound_float(a, amp: float):
    d = get_drive_line(a)
    if amp > 0.5:
        drive(d, square(40, amp))
    else:
        drive(d, square(40, 0.5 * amp))


@pulse.kernel
def nested_ifs(a, b, amp: float, t: float):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    m = readout(rb, square(40, 0.05))
    if m:
        k = readout(rb, square(40, 0.05))
        if k:
            drive(da, square(40, amp))
        else:
            wait(da, t)
    shift_phase(da.tone, amp)


@pulse.kernel
def if_yielding_a_value(a, b, amp: float):
    da = get_drive_line(a)
    m = readout(get_readout_line(b), square(40, 0.05))
    phase = 0.0
    if m:
        phase = amp
    shift_phase(da.tone, phase)
    drive(da, square(40, 0.1))


def programs():
    out = []
    for passes in (None, ("verify", "fold_constants", "fold_phases")):
        options = {} if passes is None else {"passes": passes}
        for v in (0.0, 0.25, 0.5, 1.0):
            for kernel, args in (
                (waits_in_a_loop, (Q[0], 40 + 400 * v, 5)),
                (phases_and_squares, (Q[0], v, v / 2)),
                (measured_if, (Q[0], Q[1], v, 40 + 80 * v)),
                (if_in_a_loop, (Q[0], Q[1], v, 9)),
                (waits_in_a_loop, (Q[0], 40 + 400 * v, 0)),
                (waits_in_a_loop, (Q[0], 40 + 400 * v, 1)),
                (if_in_a_loop, (Q[0], Q[1], v, 1)),
                (if_on_a_bound_float, (Q[0], v)),
                (nested_ifs, (Q[0], Q[1], v, 40 + 80 * v)),
                (if_yielding_a_value, (Q[0], Q[1], v)),
            ):
                out.append(pulse.compile(kernel, **options).lower(*args).mlir())
    return out


for program in programs():
    print(program)
    print("=" * 20)
"""


def _programs(full: bool) -> str:
    env = dict(os.environ, CUDAQ_PULSE_FULL_CANONICAL="1" if full else "0")
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_KERNELS)],
        capture_output=True,
        text=True,
        check=False,
        cwd=_ROOT,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def test_folding_with_the_patterns_a_module_can_match_gives_what_all_of_them_give():
    filtered = _programs(full=False)
    # Two pass lists, four values, ten kernels and arguments.
    assert filtered.count("=" * 20) == 80
    assert filtered == _programs(full=True)


def test_binding_from_many_threads_at_once_folds_as_one_thread_does():
    @pulse.kernel
    def bound(a, phase: float, amp: float):
        d = get_drive_line(a)
        shift_phase(d.tone, phase)
        drive(d, square(40, amp))
        drive(d, square(40, amp))

    values = [(0.1 * k, 0.05 * (k % 4)) for k in range(16)]
    compiled = pulse.compile(bound)
    expected = [compiled.lower(Q[0], *v).mlir() for v in values]
    # A fresh compile in each thread: each module gathers its pattern sets
    # once, from the context's shared ones.
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda v: pulse.compile(bound).lower(Q[0], *v).mlir(), values * 4))
    assert got == expected * 4
