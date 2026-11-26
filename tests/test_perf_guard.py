# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile-time and simulation-memory regression guard.

Skipped unless ``--run-benchmarks`` is given. Each case is timed best-of-N
and divided by a pure-Python calibration loop timed in the same process, so
the checked-in ratios transfer between machines and CI runners. A case fails
only if it is several times slower, relative to the calibration, than the
baseline in ``benchmarks/baselines/frontend_perf.json``. Refresh that file
with ``--update-perf-baseline`` after an intentional change.

Tests marked ``scaling`` compare two sizes of one program in one process
(the work counted, time or peak memory), so they hold on any machine; CI
fails on them. The Python work of each size is counted, and must be the same
or in proportion to the size; the times, which also take in native code, are
the best of many and are bounded loosely enough that a loaded runner cannot
fail them, while a cost that grows with the size still does.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.kernel._bytecode_normalize import _decode
from cudaq_pulse.kernel._emitter import PulseIREmitter
from cudaq_pulse.kernel._unroll import DEFAULT_UNROLL_THRESHOLD
from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
from cudaq_pulse.kernel.ir_builder import Parameter
from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

from tests._handles import Q, ladder_target
from tests._ir import python_ir
from tests._tools import missing
from tests._work import cpu_seconds_each, native_seconds_each, python_work, seconds_each
from tests.kernel.ir_corpus import (
    bicycle_syndrome,
    dd_loop,
    parametric,
    surface_syndrome,
    tone_list_loop,
)

_guard = importlib.import_module("cudaq_pulse.kernel._guard")

pytestmark = pytest.mark.benchmark

_ROOT = Path(__file__).resolve().parents[1]
BASELINE = _ROOT / "benchmarks" / "baselines" / "frontend_perf.json"
TOLERANCE = 3.0
# A rolled loop's cost must not grow with its trip count: loops of ROLLED_TRIPS
# trips, a hundred thousand times apart, do the same Python work and take
# times within FLAT_LOOP_LIMIT of each other. Native code that went over the
# trips at a nanosecond each would add 10 ms, over 30 times a compile.
ROLLED_TRIPS = (100, 10**7)
FLAT_LOOP_LIMIT = 10.0
# A build tree stages the package in build/<dir>/python and cudaq-pulse-opt
# in build/<dir>/bin.
_BUILD_BIN = Path(pulse.__file__).parent.parent.parent / "bin"


def _best(
    fn: Callable[[], object],
    min_reps: int = 5,
    budget_s: float = 0.25,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    fn()
    best = float("inf")
    reps = 0
    deadline = time.perf_counter() + budget_s
    while reps < min_reps or (time.perf_counter() < deadline and reps < 2000):
        t0 = clock()
        fn()
        best = min(best, clock() - t0)
        reps += 1
    return best


def _assert_flat(
    what: str,
    runs: dict[int, Callable[[], object]],
    clock: Callable[[], float] = time.process_time,
) -> None:
    """The run of each size in *runs* does the same Python work, and the
    largest takes less than FLAT_LOOP_LIMIT times the smallest's time on
    *clock*, CPU time unless a run is shorter than a tick of it."""
    small, large = min(runs), max(runs)
    work = {n: python_work(runs[n]) for n in (small, large)}
    seconds = seconds_each({n: runs[n] for n in (small, large)}, clock, 20, run_s=0.125)
    assert work[large] == work[small], (what, work)
    assert seconds[large] < seconds[small] * FLAT_LOOP_LIMIT, (what, seconds)


def _assert_linear(what: str, runs: dict[int, Callable[[], object]], limit: float) -> None:
    """The run of each size in *runs* does Python work in at most proportion
    to its size, and the largest takes less than *limit* times the smallest's
    CPU time; a cost quadratic in the size takes the ratio of sizes squared.
    The runs take milliseconds, long enough for a loaded machine to stretch
    their wall-clock times unevenly; the sizes are timed in turn, so that a
    slow spell stretches both."""
    small, large = min(runs), max(runs)
    work = {n: python_work(runs[n]) for n in (small, large)}
    seconds = cpu_seconds_each({n: runs[n] for n in (small, large)}, run_s=0.25)
    for count in work[small]:
        assert work[large][count] <= work[small][count] * large / small, (what, work)
    assert seconds[large] < seconds[small] * limit, (what, seconds)


def _calibration() -> None:
    table: dict[int, float] = {}
    acc = 0.0
    for i in range(2000):
        table[i % 64] = table.get(i % 64, 0.0) + i * 0.5
        acc += len(str(i)) + table[i % 64] % 7
    assert acc > 0


def _compile_time_workloads():
    spec = importlib.util.spec_from_file_location(
        "compile_time", _ROOT / "benchmarks" / "compiler" / "compile_time.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return {name: (k, n, f) for name, k, n, f in module.WORKLOADS}


def _kernel_ir(kernel, qudit_count: int) -> Callable[[], object]:
    return lambda: python_ir(kernel, *[Q[i] for i in range(qudit_count)])


# A fresh compiled kernel holds no specializations, so these compile.
def _compile(kernel, qudit_count: int) -> Callable[[], object]:
    return lambda: pulse.compile(kernel).lower(*[Q[i] for i in range(qudit_count)])


def _fused(kernel, n: int) -> Callable[[], object]:
    return lambda: pulse.compile(kernel, passes=()).lower(Q[0:n])


def _packed_trace(kernel, n: int) -> Callable[[], object]:
    freq = {i: 5e9 for i in range(n)}
    fn = kernel.__wrapped__

    def run():
        builder = PackedIRBuilder(name="k", clock_ghz=2.0, qudit_freq_hz=freq)
        _trace_kernel_with_builder(fn, builder, [Q[0:n]])

    return run


def _cases() -> dict[str, Callable[[], object]]:
    workloads = _compile_time_workloads()
    cases = {}
    for name in ("bell", "qaoa4", "dd_cpmg8", "vqe_hea"):
        kernel, qudit_count, _ = workloads[name]
        cases[f"tier1_{name}_trace"] = _kernel_ir(kernel, qudit_count)
        cases[f"tier1_{name}_compile"] = _compile(kernel, qudit_count)
    for n in (8, 512):
        cases[f"dd_loop_n{n}"] = _kernel_ir(dd_loop(n), 1)
        cases[f"dd_loop_n{n}_compile"] = _compile(dd_loop(n), 1)
        cases[f"tone_list_loop_n{n}_compile"] = _fused(tone_list_loop(n), 2)
    surface5, n5 = surface_syndrome(5)
    surface9, n9 = surface_syndrome(9)
    bicycle, nb = bicycle_syndrome(6, 6)
    cases["surface_d5_compile"] = _fused(surface5, n5)
    cases["surface_d9_trace"] = _packed_trace(surface9, n9)
    cases["bicycle_6x6_compile"] = _fused(bicycle, nb)
    cases["parametric_compile"] = lambda: pulse.compile(parametric).lower(Q[0], 0.5, 2)
    cases["parametric_trace"] = lambda: _trace_kernel_with_builder(
        parametric.__wrapped__,
        PackedIRBuilder(name="p", clock_ghz=2.0, qudit_freq_hz={0: 5e9}),
        [Q[0], Parameter("amp", 0, "f64"), Parameter("n", 1, "i64")],
    )
    return cases


@pytest.fixture(scope="module")
def ratios() -> dict[str, float]:
    calib = _best(_calibration, min_reps=20)
    return {name: _best(run) / calib for name, run in _cases().items()}


def test_compile_time_within_baseline(ratios, request):
    if request.config.getoption("--update-perf-baseline"):
        BASELINE.write_text(
            json.dumps({k: round(v, 3) for k, v in sorted(ratios.items())}, indent=2) + "\n"
        )
    baseline = json.loads(BASELINE.read_text())
    assert sorted(baseline) == sorted(ratios)
    slow = {
        name: f"{ratio:.2f} vs baseline {baseline[name]:.2f}"
        for name, ratio in ratios.items()
        if ratio > baseline[name] * TOLERANCE
    }
    assert not slow, f"slower than {TOLERANCE}x the baseline ratio: {slow}"


def test_building_the_module_costs_less_than_tracing_the_kernel():
    kernel, qudit_count, _ = _compile_time_workloads()["bell"]
    best = {"trace_ms": float("inf"), "ffi_ms": float("inf")}
    for _ in range(50):
        metrics = _compile(kernel, qudit_count)().metrics
        for stage in best:
            best[stage] = min(best[stage], getattr(metrics, stage))
    assert best["ffi_ms"] < best["trace_ms"], best


def _compile_work(kernel, n: int, args, **options) -> dict[str, object]:
    """What compiling *kernel* does, counted: the calls the tracer
    dispatched, the ops and packed bytes it traced, and the scheduled
    module, with the trip count *n* written as ``N``."""
    work = {"calls": 0, "packed_bytes": 0}
    dispatch, get_buffer = PulseIREmitter._dispatch_call, PackedIRBuilder.get_buffer

    def counted_dispatch(self, *a, **k):
        work["calls"] += 1
        return dispatch(self, *a, **k)

    def counted_buffer(self):
        buffer = get_buffer(self)
        work["packed_bytes"] += len(buffer) * buffer.itemsize
        return buffer

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PulseIREmitter, "_dispatch_call", counted_dispatch)
        patch.setattr(PackedIRBuilder, "get_buffer", counted_buffer)
        lowered = pulse.compile(kernel, **options).lower(*args)
    work["traced_ops"] = lowered.metrics.op_count
    work["module"] = re.sub(rf"(?<!\d){n}(?!\d)", "N", lowered.mlir())
    return work


@pytest.mark.scaling
def test_retained_loop_compile_work_is_flat():
    # A loop over the unroll threshold is kept: compiling it does the same
    # work at any trip count, and no more than the unrolled loop at the
    # threshold does.
    for make, args, options in (
        (dd_loop, (Q[0],), {}),
        (tone_list_loop, (Q[0:2],), {"passes": ()}),
    ):
        retained = [_compile_work(make(n), n, args, **options) for n in (512, 4096)]
        assert retained[0] == retained[1], make.__name__
        n = DEFAULT_UNROLL_THRESHOLD
        unrolled = _compile_work(make(n), n, args, **options)
        for count in ("calls", "packed_bytes", "traced_ops"):
            assert retained[0][count] <= unrolled[count], (make.__name__, count)


def _half_unused(qudit_count: int):
    @pulse.kernel
    def kernel(qudits):
        for q in range(qudit_count):
            line = get_drive_line(qudits[q])
            if q % 2 == 0:
                drive(line, square(20, 0.1))

    return kernel


@pytest.mark.scaling
def test_verify_does_constant_work_per_unused_line(tmp_path):
    # Each unused line is one warning that names its qubit. A warning that
    # attached its op would print it, and printing an op numbers every value
    # of its function: the whole program per unused line. cudaq-pulse-opt
    # shows the warnings that compiles from Python drop.
    opt = shutil.which("cudaq-pulse-opt", path=str(_BUILD_BIN)) or shutil.which("cudaq-pulse-opt")
    if opt is None:
        missing("cudaq-pulse-opt", "cudaq-pulse-opt is not built or on PATH")
    wide = ladder_target(2000).qudits
    for n in (500, 2000):
        module = tmp_path / f"half_unused_{n}.mlir"
        module.write_text(pulse.compile(_half_unused(n), passes=()).lower(wide[0:n]).mlir())
        stderr = subprocess.run(
            [opt, "--pulse-verify", str(module), "-o", os.devnull],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        ).stderr
        warnings = re.findall(r"^\S+: warning: (.*)$", stderr, re.M)
        assert warnings == [
            f"drive line allocated but never used (qudit {q})" for q in range(1, n, 2)
        ]
        # The message, the source line and its caret: nothing attached.
        assert stderr.count("\n") == 3 * len(warnings), stderr[:2000]


@pytest.mark.scaling
def test_compiling_a_kernel_that_reads_a_dict_is_flat_in_its_size():
    def program(n: int):
        table = dict.fromkeys(map("entry{}".format, range(n)), [0.1, 40])
        namespace = {"table": table}
        exec(
            "def kernel(q):\n"
            "    line = get_drive_line(q)\n"
            "    entry = table['entry7']\n"
            "    drive(line, square(entry[1], entry[0]))\n",
            namespace,
        )
        kernel = pulse.kernel(namespace["kernel"])
        compiled = pulse.compile(kernel)
        # Past the call that upgrades the guard: a call measures the steady state.
        for _ in range(1 + _guard.UPGRADE_AFTER):
            compiled.lower(Q[0])
        return lambda: pulse.compile(kernel).lower(Q[0]), lambda: compiled.lower(Q[0])

    # Copying a million entries, or hashing them, takes some 10 ms.
    programs = {n: program(n) for n in (16, 10**6)}
    _assert_flat("compile", {n: runs[0] for n, runs in programs.items()})
    # A cached call takes under a microsecond, so it is timed a hundred at a
    # time: the first of each hundred finds the caches cold from the other size.
    calls = {n: lambda call=runs[1]: [call() for _ in range(100)] for n, runs in programs.items()}
    _assert_flat("call", calls)


@pytest.mark.scaling
def test_a_wide_sync_traces_in_linear_time():
    wide = ladder_target(4000).qudits

    def program(n: int):
        @pulse.kernel
        def kernel(qudits):
            lines = [get_drive_line(q) for q in qudits]
            sync(*lines)
            for line in lines:
                drive(line, square(8, 0.1))

        return lambda: pulse.compile(kernel).lower(wide[0:n])

    # Eight times the lines: about 8x the time when linear, 64x when not.
    _assert_linear("trace", {n: program(n) for n in (500, 4000)}, 16)


@pytest.mark.scaling
def test_augmented_assignments_are_decoded_in_linear_time():
    def decoding(n: int):
        scope = {}
        exec("def f(x):\n" + "    x += 1\n" * n + "    return x\n", scope)
        code = scope["f"].__code__
        return lambda: _decode(code)

    # Eight times the assignments: about 8x the time when linear, 64x when
    # the instruction after each is looked for from the first.
    _assert_linear("decode", {n: decoding(n) for n in (1000, 8000)}, 16)


@pytest.mark.scaling
def test_an_if_costs_what_its_arms_change_not_the_lines_held_in_lists():
    # A readout and a measurement-conditioned if per qudit, with every line
    # held in a list: each if puts back the list slots its arms wrote, not
    # every list the kernel holds.
    wide = ladder_target(1600).qudits

    def program(n: int):
        @pulse.kernel
        def kernel(qudits):
            lines = [get_drive_line(q) for q in qudits]
            reads = [get_readout_line(q) for q in qudits]
            for i in range(len(lines)):
                sync(reads[i], lines[i])
                m = readout(reads[i], square(40, 0.05))
                if m:
                    drive(lines[i], square(8, 0.1))

        return lambda: pulse.compile(kernel).lower(wide[0:n])

    # Eight times the ifs and lines: about 8x the time when linear, 64x when not.
    _assert_linear("trace", {n: program(n) for n in (200, 1600)}, 16)


@pulse.kernel
def _reset_one(lines, reads, i):
    m = readout(reads[i], square(40, 0.05))
    if m:
        drive(lines[i], square(8, 0.1))


@pytest.mark.scaling
def test_an_if_in_a_called_kernel_costs_what_its_arms_change():
    # As above, with each if in a kernel the round calls: the trace's first
    # if is then inside a call, and the ifs after it still put back only
    # what their arms wrote.
    wide = ladder_target(1600).qudits

    def program(n: int):
        @pulse.kernel
        def kernel(qudits):
            lines = [get_drive_line(q) for q in qudits]
            reads = [get_readout_line(q) for q in qudits]
            for i in range(len(lines)):
                _reset_one(lines, reads, i)

        return lambda: pulse.compile(kernel).lower(wide[0:n])

    _assert_linear("trace", {n: program(n) for n in (200, 1600)}, 16)


@pytest.mark.scaling
def test_a_further_name_of_a_list_of_lines_costs_an_if_what_one_of_a_list_of_numbers_does():
    # Measurement-conditioned ifs over a list of lines that further names
    # hold: each if keeps one record of the list and puts it back once, so a
    # further name of it costs an if what a further name of a list of
    # numbers does (the if still looks at each name once). With a record for
    # each name, it cost over three times that.
    def program(n: int, held: str):
        source = (
            "def k(q, r):\n"
            "    lines = [get_drive_line(r)]\n"
            "    numbers = [0.1, 0.2]\n"
            "    probe = get_readout_line(q)\n"
            + "".join(f"    n{i} = {held}\n" for i in range(n))
            + "    for _ in range(24):\n"
            "        if readout(probe, square(40, 0.05)):\n"
            "            drive(lines[0], square(8, numbers[0]))\n"
            "        else:\n"
            "            drive(lines[0], square(8, numbers[1]))\n"
        )
        scope = dict(vars(pulse))
        exec(compile(source, f"<{n} names of {held}>", "exec"), scope)
        kernel = pulse.kernel(scope["k"], unroll_threshold=10_000)
        return lambda: pulse.compile(kernel).lower(Q[0], Q[1])

    # What 240 further names add to the Python work of the 24 ifs.
    added = {}
    for held in ("lines", "numbers"):
        work = {n: python_work(program(n, held)) for n in (16, 256)}
        added[held] = {count: work[256][count] - work[16][count] for count in work[16]}
    for count in ("lines", "calls"):
        assert added["lines"][count] <= added["numbers"][count] * 1.5, (count, added)


@pytest.mark.scaling
def test_the_events_of_a_long_line_are_listed_in_linear_time():
    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            for i in range(n):
                drive(line, square(8 + 0 * i, 0.1))

        # Without fuse_pulses, every drive is an event of its own.
        return pulse.compile(kernel, passes=("verify",), unroll_threshold=100_000).lower(Q[0])

    # Eight times the drives: about 10x the time when linear, 64x when not.
    runs = {n: lambda m=program(n): m.schedule for n in (2000, 16000)}  # noqa: B008
    _assert_linear("schedule", runs, 8 * TOLERANCE)


@pytest.mark.scaling
def test_a_rolled_round_with_branches_is_not_scheduled_pair_by_pair_of_lines():
    from cudaq_pulse.controller import ReadoutLine
    from cudaq_pulse.quantum_systems import PhysicalQudit

    def scheduled(n: int) -> float:
        # A round of n qudits on controllers of seven, kept as a loop: one
        # sync joins every line, neighbours are synced in pairs, and every
        # second qudit is read out and driven again where its result is set.
        groups = [list(range(k, min(n, k + 7))) for k in range(0, n, 7)]
        controllers = [
            Controller(
                clock_ghz=2.0,
                drive_lines=g,
                readout_lines=[ReadoutLine(g, max_concurrent_tones=len(g))],
                supports=["play", "readout", "wait", "sync", "branch", "loop"],
                feedback_latency_cycles=400,
                peer_latency_cycles=400,
            )
            for g in groups
        ]
        system = pulse.quantum_system(
            name=f"round{n}",
            qudits={i: PhysicalQudit(i, 4.5e9 + 1.0e3 * i, 37.7, 9.2) for i in range(n)},
            control_system=ControlSystem(controllers=controllers, reference_clock_ghz=2.0),
        )

        def kernel():
            # Made anew for each compile: a kernel keeps what it compiled.
            @pulse.kernel
            def rounds(qudits):
                for _ in range(1000):
                    lines = [get_drive_line(q) for q in qudits]
                    for line in lines:
                        drive(line, square(48, 0.25))
                    sync(*lines, *[get_readout_line(q) for q in qudits])
                    for first in (0, 1):
                        for i in range(first, n - 1, 2):
                            sync(get_drive_line(qudits[i]), get_drive_line(qudits[i + 1]))
                            drive(get_drive_line(qudits[i]), square(48, 0.25))
                    for i in range(0, n, 2):
                        line = get_drive_line(qudits[i])
                        answer = get_readout_line(qudits[i])
                        sync(line, answer)
                        if readout(answer, square(1000, 0.05)):
                            drive(line, square(48, 0.25))

            return rounds

        return min(
            pulse.compile(kernel(), quantum_system=system)
            .lower(system.qudits[0:n])
            .metrics.schedule_ms
            for _ in range(3)
        )

    # Four times the qudits: the conditions are four times as many and each
    # reads four times the lines. Pairing the lines of every condition with
    # the lines it decides takes 64 times as long; the stage as a whole has
    # parts that take 16.
    small, large = scheduled(100), scheduled(400)
    assert large < small * 8 * TOLERANCE, (small, large)


@pytest.mark.scaling
def test_a_round_on_many_controllers_is_checked_in_linear_time():
    from cudaq_pulse.controller import ReadoutLine, _compile_check
    from cudaq_pulse.quantum_systems import PhysicalQudit

    def program(n: int):
        # A round of n qudits on controllers of seven, kept as a loop: a
        # drive on every qudit, one sync of every line, and every second
        # qudit read out and driven again where its result is set.
        groups = [list(range(k, min(n, k + 7))) for k in range(0, n, 7)]
        controllers = [
            Controller(
                clock_ghz=2.0,
                drive_lines=g,
                readout_lines=[ReadoutLine(g, max_concurrent_tones=len(g))],
                supports=["play", "readout", "wait", "sync", "branch", "loop"],
                feedback_latency_cycles=400,
                peer_latency_cycles=400,
            )
            for g in groups
        ]
        system = pulse.quantum_system(
            name=f"checked{n}",
            qudits={i: PhysicalQudit(i, 4.5e9 + 1.0e3 * i, 37.7, 9.2) for i in range(n)},
            control_system=ControlSystem(controllers=controllers, reference_clock_ghz=2.0),
        )

        @pulse.kernel
        def rounds(qudits):
            for _ in range(1000):
                lines = [get_drive_line(q) for q in qudits]
                for line in lines:
                    drive(line, square(48, 0.25))
                sync(*lines, *[get_readout_line(q) for q in qudits])
                for i in range(0, n, 2):
                    line = get_drive_line(qudits[i])
                    answer = get_readout_line(qudits[i])
                    sync(line, answer)
                    if readout(answer, square(1000, 0.05)):
                        drive(line, square(48, 0.25))

        lowered = pulse.compile(rounds, quantum_system=system).lower(system.qudits[0:n])
        machine = lowered.options.machine
        # The check compiling makes of the program against its control system.
        return lambda: _compile_check(lowered, machine.controller, machine._system)

    # Eight times the qudits and the controllers: about 8x the time when
    # each controller is checked on its own pulses, 64x when each is checked
    # on a copy of every controller's.
    _assert_linear("a round's check", {n: program(n) for n in (70, 560)}, 8 * TOLERANCE)


@pytest.mark.scaling
def test_calls_with_branches_on_their_answers_are_timed_in_linear_time():
    from cudaq_pulse.quantum_systems import PhysicalQudit

    @pulse.classical(max_latency_ns=60)
    def decide(bits: pulse.Bits) -> bool:
        return True

    controllers = [Controller(clock_ghz=2.0, drive_lines=[i], readout_lines=[i]) for i in range(2)]
    system = pulse.quantum_system(
        name="calls",
        qudits={i: PhysicalQudit(i, 4.5e9 + 1.0e5 * i, 37.7, 9.2) for i in range(2)},
        control_system=ControlSystem(controllers=controllers, reference_clock_ghz=2.0),
        clock_ghz=2.0,
    )

    def program(n: int):
        @pulse.kernel
        def kernel(qudits):
            line = get_drive_line(qudits[0])
            answer = get_readout_line(qudits[0])
            for i in range(n):
                if decide.submit([readout(answer, square(32 + 0 * i, 0.05))]).result():
                    drive(line, square(40, 0.1))

        # A fresh compiled kernel holds no specializations, so this compiles.
        return lambda: pulse.compile(kernel, quantum_system=system, unroll_threshold=100_000).lower(
            system.qudits[0:2]
        )

    # On several controllers a call's budget takes the controllers that play
    # what uses its answer: the line a branch on it drives, through every
    # branch after it. Eight times the calls: about 8x the time when the
    # branches are followed once for all calls, 64x when once per call.
    _assert_linear("calls", {n: program(n) for n in (250, 2000)}, 8 * TOLERANCE)


@pytest.mark.scaling
def test_the_pulses_at_once_are_counted_in_linear_time():
    from cudaq_pulse._concurrency import violations
    from cudaq_pulse.quantum_systems import PhysicalQudit

    @pulse.kernel
    def together(qudits):
        for q in qudits:
            drive(get_drive_line(q), square(40, 0.1))

    def program(n: int):
        # Every drive is in flight at once, and the limit lets them be.
        control = ControlSystem(
            controllers=[Controller(drive_lines=range(n), readout_lines=range(n))],
            max_concurrent_drives=n,
        )
        qudits = {i: PhysicalQudit(i, 5.0e9 + 1.0e3 * i, 50.0, 30.0) for i in range(n)}
        system = pulse.quantum_system(name=f"wide{n}", qudits=qudits, control_system=control)
        lowered = pulse.compile(together, quantum_system=system, schedule="asap").lower(
            system.qudits[0:n]
        )
        return lambda: violations(lowered)

    # Eight times the drives: about 8x the time when linear, 64x when each
    # drive is counted against every other in flight.
    _assert_linear("at once", {n: program(n) for n in (500, 4000)}, 8 * TOLERANCE)


@pytest.mark.scaling
def test_the_pulses_of_a_round_kept_as_a_loop_are_bounded_in_linear_time():
    from cudaq_pulse._concurrency import violations
    from cudaq_pulse.quantum_systems import PhysicalQudit

    @pulse.kernel
    def rounds(qudits):
        for _ in range(1000):
            for q in qudits:
                drive(get_drive_line(q), square(40, 0.1))

    def program(n: int):
        # Every drive of a round is in flight at once, and the limit lets
        # them be: the bound on what a round plays settles it.
        control = ControlSystem(
            controllers=[Controller(drive_lines=range(n), readout_lines=range(n))],
            max_concurrent_drives=n,
        )
        qudits = {i: PhysicalQudit(i, 5.0e9 + 1.0e3 * i, 50.0, 30.0) for i in range(n)}
        system = pulse.quantum_system(name=f"round{n}", qudits=qudits, control_system=control)
        lowered = pulse.compile(rounds, quantum_system=system, schedule="asap").lower(
            system.qudits[0:n]
        )
        return lambda: violations(lowered)

    # Eight times the drives: about 8x the time when linear, 64x when each
    # drive is summed with every other in flight.
    _assert_linear("a round's bound", {n: program(n) for n in (500, 4000)}, 8 * TOLERANCE)


@pytest.mark.scaling
def test_the_values_of_a_long_list_are_bound_in_linear_time():
    @pulse.kernel
    def stream(q, amplitudes: list[float], count: int):
        line = get_drive_line(q)
        for i in range(count):
            drive(line, square(20, amplitudes[i]))

    compiled = pulse.compile(stream)
    turn = [0]

    def bind(count: int):
        def run():
            # Other values each time: a call's module is kept by its values.
            turn[0] += 1
            values = [float((i + turn[0]) % 7) / 8 for i in range(count)]
            return compiled.lower(Q[0], values, count).mlir()

        return run

    # Eight times the values: about 10x the time when linear, 64x when what
    # the values reach is gathered once per value.
    _assert_linear("bind", {n: bind(n) for n in (1000, 8000)}, 8 * TOLERANCE)


def _doubled(first: str, line: str, last: str, times: int, **names: object) -> Callable[[], object]:
    """A compile, traced afresh at every call, of a kernel that sets a value
    with *first*, runs *line* on it *times* times and uses it with *last*."""
    turn = [0]

    def run():
        turn[0] += 1
        source = (
            f"def k(q, x: float):\n    d = get_drive_line(q)\n    unused = {turn[0]}\n"
            f"    {first}\n" + f"    {line}\n" * times + last
        )
        scope = dict(vars(pulse))
        scope.update(names)
        exec(compile(source, f"<doubled {times}>", "exec"), scope)
        return pulse.compile(pulse.kernel(scope["k"])).lower(Q[0], 2.0**-times)

    return run


@pytest.mark.scaling
def test_a_wait_length_that_doubles_a_term_is_traced_in_linear_time():
    # v = v + v uses the line before it twice: written out, six more lines
    # are 64 times the terms. Each walk of the value visited them all. The
    # sizes are small so that this fails and does not hang.
    runs = {
        n: _doubled("v = x * 1.0", "v = v + v", "    wait(d, v, bounds=(0, 4000))\n", n)
        for n in (6, 12)
    }
    _assert_linear("trace", runs, 2 * TOLERANCE)
    # The error for the length names each line once: a megabyte written out.
    with pytest.raises(ValueError) as raised:
        pulse.compile(pulse.kernel(_doubled_kernel(16))).lower(Q[0], 1.0)
    assert len(str(raised.value)) < 1000


def _doubled_kernel(times: int):
    source = (
        "def k(q, x: float):\n    d = get_drive_line(q)\n    v = x * 1.0\n"
        + "    v = v + v\n" * times
        + "    wait(d, v, bounds=(0, 4000))\n"
    )
    scope = dict(vars(pulse))
    exec(compile(source, f"<doubled {times}>", "exec"), scope)
    return scope["k"]


@pytest.mark.scaling
@pytest.mark.parametrize(
    ("line", "sizes"), [("c = c & c", (6, 12)), ("c = (c | c) & (c | e)", (4, 8))]
)
def test_a_condition_that_doubles_a_condition_is_traced_in_linear_time(line, sizes):
    @pulse.classical(max_latency_ns=50, name="perf_doubled_flag")
    def flag(x: float) -> bool:
        return x > 1.0

    first = "c = flag.submit(x).result(); e = flag.submit(x + 1.0).result(); c = c & e"
    last = "    if c:\n        drive(d, square(40, 0.1))\n"
    # The second line uses the one before it three times: 81 times the
    # terms for four more lines.
    runs = {n: _doubled(first, line, last, n, flag=flag) for n in sizes}
    _assert_linear("trace", runs, 2 * TOLERANCE)


@pytest.mark.scaling
def test_the_uses_of_a_line_in_the_arms_of_nested_ifs_are_checked_in_linear_time():
    # Each leaf of a tree of ifs nested `depth` deep reads out on one line:
    # 2**depth uses of it, no two on one path, which verify checks.
    def program(depth: int):
        source = ["def kernel(qs):", "    rm = get_readout_line(qs[1])"]
        source += ["    r = get_readout_line(qs[0])"]
        source += [f"    m{d} = readout(rm, square(40, 0.05))" for d in range(depth)]
        source.append("    sync(rm, r)")

        def tree(d: int, indent: str) -> None:
            if d == depth:
                source.append(f"{indent}readout(r, square(400, 0.05))")
                return
            source.append(f"{indent}if m{d}:")
            tree(d + 1, indent + "    ")
            source.append(f"{indent}else:")
            tree(d + 1, indent + "    ")

        tree(0, "    ")
        namespace: dict[str, object] = {}
        exec("\n".join(source) + "\n", namespace)
        kernel = pulse.kernel(namespace["kernel"])
        device = ladder_target(2)
        return lambda: pulse.compile(kernel, quantum_system=device, passes=("verify",)).lower(
            device.qudits[0:2]
        )

    # Eight times the uses: about 8x the time when linear, 64x when every two
    # of them are compared.
    seconds = native_seconds_each({d: program(d) for d in (7, 10)}, ("run_passes",))
    assert seconds[10] < seconds[7] * 8 * TOLERANCE, seconds


@pytest.mark.scaling
def test_rcp_takes_a_lane_in_logarithmic_time():
    def program(lanes: int):
        machine = ControlSystem(max_concurrent_drives=lanes)

        @pulse.kernel
        def kernel(qudits):
            for q in qudits:
                line = get_drive_line(q)
                for i in range(400):
                    drive(line, square(8 + 0 * i, 0.1))

        return lambda: pulse.compile(
            kernel,
            schedule="rcp",
            control_system=machine,
            passes=("verify",),
            unroll_threshold=100_000,
        ).lower(Q[0:16])

    # 128 times the lanes, of which the 16 lines never take more than 16:
    # about the same time when a pulse takes a lane in logarithmic time,
    # several times more when it compares every lane.
    seconds = native_seconds_each({lanes: program(lanes) for lanes in (64, 8192)})
    assert seconds[8192] < seconds[64] * 2.5, seconds


@pytest.mark.scaling
def test_a_long_program_is_exported_and_checked_in_linear_time():
    limits = Controller(
        instruction_cycles=2,
        detectors={0: 0},
        max_frequencies_per_line=2,
        waveform_memory_samples=10**9,
        feedback_latency_cycles=1,
    )
    machine = ControlSystem(controllers=[limits])

    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            probe = get_readout_line(q)
            for i in range(n):
                drive(line, gaussian(16 + 0 * i, 0.1, 4.0))
                sync(line, probe)
                m = readout(probe, square(40, 0.05))
                sync(probe, line)
                if m:
                    drive(line, square(8, 0.1))

        compiled = pulse.compile(
            kernel, schedule="rcp", control_system=machine, unroll_threshold=100_000
        )
        return compiled.lower(Q[0])

    # Eight times the events: about 8x the time when linear, 64x when not.
    runs = {n: program(n).export_to_controller for n in (500, 4000)}
    _assert_linear("export", runs, 8 * TOLERANCE)


def _sweep(compiled, sweep: list[float], monkeypatch) -> None:
    """Lower *compiled* at each length of *sweep*: after the first, none may
    trace the kernel or build a native module, and each shares the first's.

    These count what a compile does rather than time it, so a loaded runner
    can't fail them. What binding a new value costs is measured by
    tools/perf-gate.py (bind_new_value)."""
    from cudaq_pulse import compiler
    from cudaq_pulse.kernel import bytecode_bridge

    first = compiled.lower(Q[0], sweep[0])
    work = {"traces": 0, "modules": 0}
    trace, builder = bytecode_bridge._trace_kernel_with_builder, compiler.PulseModuleBuilder

    def counted_trace(*args, **kwargs):
        work["traces"] += 1
        return trace(*args, **kwargs)

    def counted_builder():
        work["modules"] += 1
        return builder()

    monkeypatch.setattr(bytecode_bridge, "_trace_kernel_with_builder", counted_trace)
    monkeypatch.setattr(compiler, "PulseModuleBuilder", counted_builder)
    lowered = [compiled.lower(Q[0], tau) for tau in sweep[1:]]
    monkeypatch.undo()

    assert work == {"traces": 0, "modules": 0}
    assert compiled.cache_info().compiled == 1
    assert all(low._module is first._module for low in lowered)
    assert [low.arguments for low in lowered] == [(tau,) for tau in sweep[1:]]


@pytest.mark.scaling
def test_a_rolled_loop_is_checked_and_exported_rolled_in_constant_time():
    limits = Controller(
        instruction_cycles=2,
        detectors={0: 0},
        max_frequencies_per_line=2,
        waveform_memory_samples=10**9,
    )
    machine = ControlSystem(controllers=[limits])

    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            probe = get_readout_line(q)
            for _ in range(n):
                drive(line, gaussian(16, 0.1, 4.0))
                shift_phase(line.tone, 0.1)
                sync(line, probe)
                readout(probe, square(40, 0.05))
                sync(probe, line)

        return kernel

    compiles, exports, size = {}, {}, {}
    for n in ROLLED_TRIPS:
        kernel = program(n)
        compiles[n] = lambda k=kernel: pulse.compile(k, control_system=machine).lower(Q[0])
        lowered = compiles[n]()
        exports[n] = lambda m=lowered: m.export_to_controller(version=2)
        size[n] = len(json.dumps(lowered.export_to_controller(version=2)))
    # The same cost at any trip count when the loop stays rolled.
    _assert_flat("compile", compiles)
    _assert_flat("export", exports)
    small, large = ROLLED_TRIPS
    assert size[large] < size[small] + 64, size


@pytest.mark.scaling
def test_the_segments_of_a_rolled_loop_that_opens_none_are_found_in_constant_time():
    @pulse.classical(max_latency_ns=20, link=0)
    def tune(x: float) -> float:
        return x

    @pulse.classical(max_latency_ns=20, latency_mode="blocking", link=0)
    def hold(x: float) -> float:
        return x

    def program(n: int, call: Any):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            for _ in range(n):
                drive(line, square(8, 0.1))
                wait(line, 16)
            drive(line, square(8, call.submit(0.1).result()))

        return kernel

    # A call after the loop, whose answer opens no segment or one: the loop
    # opens none, and its iterations are not walked to find that.
    for call, segments in ((tune, 1), (hold, 2)):
        runs = {}
        for n in ROLLED_TRIPS:
            lowered = pulse.compile(program(n, call), schedule="asap", unroll_threshold=0).lower(
                Q[0]
            )
            assert len(lowered.segments()) == segments
            runs[n] = lowered.segments
        _assert_flat(f"segments, {segments}", runs)

    def inside(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            for _ in range(3):
                for _ in range(n):
                    drive(line, square(8, 0.1))
                    wait(line, 16)
                drive(line, square(8, hold.submit(0.1).result()))

        return kernel

    # A loop that opens none inside one that opens a segment in every
    # iteration: the outer loop is walked, the inner one passed over each time.
    runs = {}
    for n in ROLLED_TRIPS:
        lowered = pulse.compile(inside(n), schedule="asap", unroll_threshold=0).lower(Q[0])
        # Both loops are kept (the outer one may have an iteration peeled).
        assert lowered.mlir().count("scf.for") >= 2
        assert len(lowered.segments()) == 4
        runs[n] = lowered.segments
    _assert_flat("segments, inside", runs)


@pytest.mark.scaling
def test_duration_budgets_of_a_rolled_loop_are_checked_in_constant_time():
    @pulse.kernel(schedule="asap", max_duration_cycles=1000)
    def cycle(q):
        line = get_drive_line(q)
        probe = get_readout_line(q)
        drive(line, gaussian(16, 0.1, 4.0))
        sync(line, probe)
        readout(probe, square(40, 0.05))

    def program(n: int):
        @pulse.kernel(max_duration_cycles=10**9)
        def kernel(q):
            for _ in range(n):
                cycle(q)

        return kernel

    runs = {n: lambda k=program(n): pulse.compile(k).lower(Q[0]) for n in ROLLED_TRIPS}  # noqa: B008
    _assert_flat("compile", runs)


@pytest.mark.scaling
def test_a_waveform_memory_of_a_rolled_loop_is_checked_in_constant_time():
    from cudaq_pulse.controller import check

    limits = Controller(waveform_memory_samples=4096, waveform_memory="shared")

    def program(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            for _ in range(n):
                drive(line, gaussian(16, 0.1, 4.0))
                drive(line, square(8, 0.2))
                shift_phase(line.tone, 0.1)

        return kernel

    runs = {}
    for n in ROLLED_TRIPS:
        lowered = pulse.compile(program(n)).lower(Q[0])
        assert check(lowered, limits) == []
        runs[n] = lambda m=lowered: check(m, limits)
    _assert_flat("check", runs)


@pytest.mark.scaling
def test_a_wait_length_sweep_compiles_once(monkeypatch):
    @pulse.kernel
    def t1(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(40, 0.1))
        wait(d, tau)
        sync(d, r)
        readout(r, square(100, 0.05))

    _sweep(pulse.compile(t1), [float(4 * i + 1) for i in range(1000)], monkeypatch)


@pytest.mark.scaling
def test_exporting_a_call_again_does_not_specialize_again():
    @pulse.kernel
    def pulses(q, amp: float):
        d = get_drive_line(q)
        for _ in range(16):
            drive(d, gaussian(40, amp, 10.0))
            shift_phase(d.tone, 0.1)

    compiled = pulse.compile(pulses, unroll_threshold=100_000)
    values = iter(1e-3 * k for k in range(1, 10**6))
    seconds = cpu_seconds_each(
        {
            "new": lambda: compiled.lower(Q[0], next(values)).export_to_controller(),
            "again": lambda: compiled.lower(Q[0], 0.5).export_to_controller(),
        },
        repeats=20,
        run_s=0.125,
    )
    new, again = seconds["new"], seconds["again"]
    assert compiled.lower(Q[0], 0.5)._concrete() is compiled.lower(Q[0], 0.5)._concrete()
    # Specializing is most of an export of new values: about 3x the time.
    assert again < new / 1.5, (again, new)


@pytest.mark.scaling
def test_a_sweep_of_a_wait_from_an_answer_and_a_float_compiles_once(monkeypatch):

    @pulse.classical(max_latency_ns=50, name="perf_offset")
    def offset(x: float) -> float:
        return x

    @pulse.kernel
    def t1(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(40, 0.1))
        wait(d, offset.submit(8.0).result() + tau, bounds=(0, 8000))
        sync(d, r)
        readout(r, square(100, 0.05))

    _sweep(pulse.compile(t1), [float(4 * i + 1) for i in range(1000)], monkeypatch)


@pytest.mark.scaling
def test_a_loop_that_hands_each_answer_to_the_next_call_compiles_in_linear_time():

    @pulse.classical(max_latency_ns=50, name="perf_next_tau")
    def next_tau(x: float) -> float:
        return 2.0 * x

    def program(n: int):
        @pulse.kernel
        def adaptive(q, tau0: float):
            d = get_drive_line(q)
            tau = tau0
            for _ in range(n):
                tau = next_tau.submit(tau).result()
                drive(d, square(40, 0.1))
                wait(d, tau, bounds=(16, 4000))

        return lambda: pulse.compile(adaptive).lower(Q[0], 10.0)

    # Twice the rounds: about 2x the time when linear. The scheduler asks for
    # each wait's length; a walk from it through the calls behind it, by each
    # call's link and by its argument, took 2^10 times the time for ten more
    # rounds. The sizes are small so that this fails and does not hang.
    _assert_linear("compile", {n: program(n) for n in (10, 20)}, 2 * TOLERANCE)


# Peak memory a simulation of two qutrits may add: their 9 x 9 density matrix
# and operators are tiny, so this only fails if the space grows with the
# qudits' indices (3^14 states for qudits 13 and 3).
SIMULATE_MEMORY_LIMIT_MB = 64

_SIMULATE_PAIRS = """
import json, resource, sys
import cudaq_pulse as pulse

q = pulse.quantum_system("transmon_surface17").qudits


@pulse.kernel
def pair(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, gaussian(40, 0.08, 10.0))
    drive(db, gaussian(40, 0.03, 10.0))


scale = 1 if sys.platform == "darwin" else 1024
growth = {}
for a, b in json.loads(sys.argv[1]):
    lowered = pair.lower(q[a], q[b])
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = pulse.simulate(lowered, levels=3, t_end=2.0)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth[f"{a},{b}"] = [list(result.dims), (after - before) * scale / 2**20]
print(json.dumps(growth))
"""


@pytest.mark.scaling
def test_simulating_a_high_index_pair_takes_little_memory(tmp_path):
    script = tmp_path / "simulate_pairs.py"
    script.write_text(_SIMULATE_PAIRS)
    pairs = [(13, 3), (16, 8), (9, 1)]
    out = subprocess.run(
        [sys.executable, str(script), json.dumps(pairs)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    growth = json.loads(out.stdout)
    assert {k: dims for k, (dims, _) in growth.items()} == {f"{a},{b}": [3, 3] for a, b in pairs}
    assert all(mb < SIMULATE_MEMORY_LIMIT_MB for _, mb in growth.values()), growth


# ── Classical calls ──────────────────────────────────────────────────────────


def _classical_benchmarks():
    spec = importlib.util.spec_from_file_location(
        "classical_calls", _ROOT / "benchmarks" / "classical" / "classical_calls.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.scaling
def test_calls_compile_schedule_and_report_in_linear_time():
    bench = _classical_benchmarks()
    q = bench._device().qudits[0]
    compiles, reports = {}, {}
    for n in (1000, 8000):
        kernel = bench.many_calls(n, 4)
        compiles[n] = lambda k=kernel: pulse.compile(k, schedule="asap").lower(q)
        lowered = compiles[n]()
        reports[n] = lowered.classical_report
        assert len(lowered.classical_report()) == n
    # 8x the calls: linear is 8x, quadratic 64x.
    _assert_linear("compile", compiles, 16)
    _assert_linear("report", reports, 16)
    seconds = seconds_each(
        {"events": lambda m=lowered: m.schedule, "report": lowered.classical_report},
        time.process_time,
        run_s=0.125,
    )
    assert seconds["events"] < seconds["report"] * 4, seconds


def test_a_cached_kernel_that_calls_lowers_in_microseconds():
    bench = _classical_benchmarks()
    dev = bench._device()
    q = dev.qudits[0]
    compiled = pulse.compile(bench.decoded_reset, quantum_system=dev)
    compiled.lower(q)
    per_call = _best(lambda: [compiled.lower(q) for _ in range(100)], 5) / 100
    assert compiled.cache_info().compiled == 1
    assert per_call < 100e-6, per_call


def test_a_call_adds_little_to_simulation_on_the_same_timeline():
    bench = _classical_benchmarks()
    dev = bench._device(t1_us=20.0)
    q = dev.qudits[0]
    plain = pulse.compile(bench.waiting_reset, quantum_system=dev).lower(q)
    called = pulse.compile(bench.decoded_reset, quantum_system=dev).lower(q)
    plus = [2**-0.5, 2**-0.5]
    seconds = seconds_each(
        {
            "plain": lambda: pulse.simulate(plain, initial_state=plus, levels=2, dt_ns=0.5),
            "called": lambda: pulse.simulate(called, initial_state=plus, levels=2, dt_ns=0.5),
        },
        time.perf_counter,
        repeats=3,
        run_s=0,
    )
    a, b = seconds["plain"], seconds["called"]
    # Both take well under a millisecond, so allow the call's own binding and
    # rescheduling (about 0.3 ms) on top.
    assert b < 1.25 * a + 2e-3, (a, b)


@pytest.mark.scaling
def test_compiling_calls_takes_little_memory_per_call():
    import tracemalloc

    bench = _classical_benchmarks()
    q = bench._device().qudits[0]
    peaks = {}
    for n in (500, 4000):
        kernel = bench.many_calls(n, 1)
        pulse.compile(kernel, schedule="asap").lower(q)
        tracemalloc.start()
        pulse.compile(kernel, schedule="asap").lower(q)
        peaks[n] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    per_call = (peaks[4000] - peaks[500]) / 3500
    assert per_call < 2048, per_call


def _if_chain(n):
    src = (
        "def k(a, b):\n    da = get_drive_line(a)\n    db = get_drive_line(b)\n"
        "    m = readout(get_readout_line(b), square(40, 0.05))\n"
        + "    if m:\n        shift_phase(da.tone, 0.1)\n    else:\n        wait(db, 8)\n" * n
        + "    drive(da, square(40, 0.1))\n"
    )
    scope = dict(vars(pulse))
    exec(compile(src, f"<{n} ifs>", "exec"), scope)
    return pulse.kernel(scope["k"], unroll_threshold=0)


@pytest.mark.scaling
def test_tracing_measured_ifs_takes_little_memory_per_if():
    # Each if's arms stop where they join: that end was written into a new
    # copy of the whole function's instructions, kept for every if, so a
    # function of n ifs held n copies of n ifs' instructions.
    import tracemalloc

    peaks = {}
    for n in (300, 2400):
        kernel = _if_chain(n)
        pulse.compile(kernel).lower(Q[0], Q[1])
        tracemalloc.start()
        pulse.compile(_if_chain(n)).lower(Q[0], Q[1])
        peaks[n] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    # About 12 KB an if (the if's own IR); 300 KB an if at 2,400 ifs when
    # each kept its copy.
    per_if = (peaks[2400] - peaks[300]) / 2100
    assert per_if < 65536, per_if


def _measured_line(n):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for i in range(n):
            drive(d, square(8 + 0 * i, 0.1))
            shift_phase(d.tone, 0.1)
            sync(d, r)
            readout(r, square(40, 0.05))
            sync(d, r)

    return pulse.compile(k, unroll_threshold=100_000).lower(Q[0])


def test_walking_a_long_program_along_a_path_is_linear():
    # Every drive's frame comes from its tone, threaded through every op on
    # the line before it: following that chain back from each drive was
    # quadratic in the program's length.
    walks, exports = {}, {}
    for n in (500, 4000):
        lowered = _measured_line(n)
        module = lowered._concrete().simulation_module(*lowered.options.schedule_args())
        walks[n] = lambda m=module: m.next_measurement([])
        exports[n] = lowered.export_to_controller
    # 8x the ops: linear is 8x, quadratic 64x.
    _assert_linear("walk", walks, 16)
    _assert_linear("export", exports, 16)


def test_a_long_idle_simulates_in_constant_time():
    @pulse.kernel
    def idle(q, wait_cycles: int):
        d = get_drive_line(q)
        drive(d, square(40, 0.157))
        wait(d, wait_cycles)

    dev = ladder_target(1)
    compiled = pulse.compile(idle, quantum_system=dev)
    q = dev.qudits[0]
    short, long = compiled.lower(q, 2_000), compiled.lower(q, 400_000)
    times = seconds_each(
        {"1 us": lambda: pulse.simulate(short), "200 us": lambda: pulse.simulate(long)},
        time.perf_counter,
        repeats=3,
        run_s=0,
    )
    assert times["200 us"] < 3 * times["1 us"] + 0.01, times


@pytest.mark.scaling
def test_drives_apart_from_each_other_simulate_in_linear_time():
    def train(n: int):
        @pulse.kernel
        def kernel(q):
            line = get_drive_line(q)
            for _ in range(n):
                drive(line, square(8, 0.1))
                wait(line, 16)

        return kernel

    runs = {}
    for n in (32, 256):
        lowered = pulse.compile(train(n), schedule="asap", unroll_threshold=10**6).lower(Q[0])
        runs[n] = lambda m=lowered: pulse.simulate(m, levels=2, backend="cpu", dt_ns=0.5, seed=0)
    # 8x the drives, each stepped with its own terms: linear is 8x; asking
    # every term at every drive is 64x.
    _assert_linear("simulate", runs, 16)


def test_an_adaptive_loop_of_fifty_rounds_simulates_one_path_per_shot():
    from cudaq_pulse.classical import Link

    @pulse.classical(max_latency_ns=200, link="controller")
    def next_wait(bit: pulse.Bits, w: float) -> float:
        return min(max(w * (1.5 if bit[0] else 0.75), 40.0), 200_000.0)

    @pulse.kernel
    def rounds(q, n: int):
        d = get_drive_line(q)
        r = get_readout_line(q)
        w = 20_000.0
        drive(d, square(40, 0.157))
        wait(d, 20_000)
        for _ in range(n):
            sync(d, r)
            m = readout(r, square(200, 0.05))
            w = next_wait.submit([m], w).result()
            sync(d, r)
            if m:
                drive(d, square(40, 0.157))
            drive(d, square(40, 0.157))
            wait(d, w, bounds=(40, 200_000))

    machine = ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8)})
    dev = ladder_target(1)
    lowered = pulse.compile(rounds, quantum_system=dev, control_system=machine).lower(
        dev.qudits[0], 50
    )
    seconds = _best(
        lambda: pulse.simulate(lowered, shots_count=1, seed=1, branches="sampled", dt_ns=0.125),
        2,
        0,
    )
    # Following all 2^50 paths is out of reach; one path takes about 0.1 s.
    assert seconds < 2.0, seconds
