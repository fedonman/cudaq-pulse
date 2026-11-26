# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context that compiled modules share has the dialects those modules can
hold and never loads another: the lowering to LLVM works on a copy in a
context the calling thread keeps for it."""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native import _cudaq_pulse_native as native
from cudaq_pulse.compiler import IRLevel

from tests._handles import Q
from tests.test_load import MODULE

MODULES = "CUDAQ_PULSE_CONTEXT_MAX_MODULES"
TO_LLVM = ["pulse-to-qop", "qop-to-cudm", "cudm-to-llvm"]


def _compile(amplitude=0.2):
    @pulse.kernel
    def pulses(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        drive(d0, gaussian(40, 0.3, 10.0))
        sync(d0, d1)
        drive(d1, square(16, amplitude))

    return pulses.lower(Q[0], Q[1])


def test_every_level_lowers_and_the_module_stays_as_it_was():
    lowered = _compile()
    text = lowered.mlir()
    levels = {level: lowered.mlir(level) for level in IRLevel}
    assert "qop." in levels[IRLevel.QOP]
    assert "cudm." in levels[IRLevel.CUDM]
    assert "llvm.func @main" in levels[IRLevel.LLVM]
    assert "\n    cudm." not in levels[IRLevel.LLVM]
    assert lowered._concrete().print() == text
    # Lowering again gives the same text: nothing of the first stays behind.
    assert {level: lowered.mlir(level) for level in IRLevel} == levels


def test_lowering_to_llvm_is_the_same_from_any_context(monkeypatch):
    lowered = _compile()
    text = lowered._concrete().run_full_lowering()
    # A module parsed into a context of its own, with every dialect.
    parsed = native.MLIRPipeline().parse_to_module(lowered.mlir())
    assert parsed.run_full_lowering() == text
    # A pickled module, read into the shared context.
    assert pickle.loads(pickle.dumps(lowered)).mlir("llvm") == text
    # A new shared context for every module.
    monkeypatch.setenv(MODULES, "1")
    assert _compile().mlir("llvm") == text


def test_a_thread_lowers_to_llvm_more_often_than_it_keeps_one_context():
    module = _compile()._concrete()
    text = module.run_full_lowering()
    # The thread's lowering context is replaced after 256 lowerings.
    assert all(module.run_full_lowering() == text for _ in range(300))


def test_a_shared_context_refuses_a_pipeline_that_needs_another_dialect():
    module = _compile()._concrete().resolve_selects()
    text = module.print()
    with pytest.raises(
        RuntimeError, match="run_passes: a pass makes operations of the 'cudm' dialect"
    ):
        module.run_passes(TO_LLVM)
    assert module.print() == text
    # The same pipeline runs on a module parsed from the text.
    parsed = native.MLIRPipeline().parse_to_module(text)
    parsed.run_passes(TO_LLVM)
    assert "llvm.call @cudm_evolve" in parsed.print()
    # Passes of the dialects a shared context has still run in it.
    module.run_passes(["pulse-verify", "pulse-to-qop"])
    assert "qop." in module.print()


@pytest.mark.parametrize(
    "dialect, other",
    [
        ("llvm", "  llvm.func @elsewhere(%a: i64) -> i64 {\n    llvm.return %a : i64\n  }\n"),
        ("llvm", "  func.func private @other(!llvm.ptr)\n"),
        (
            "cudm",
            "  func.func @other() {\n    %h = cudm.init_handle : !cudm.handle\n    return\n  }\n",
        ),
    ],
)
def test_a_loaded_module_that_names_a_dialect_of_the_llvm_lowering_is_refused(dialect, other):
    system = pulse.quantum_system("transmon_surface17")
    text = MODULE.rstrip().removesuffix("}") + other + "}\n"
    with pytest.raises(
        pulse.CompilationError,
        match=f'Dialect `{dialect}\' not found|"{dialect}".* unregistered dialect',
    ):
        pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert pulse.load(MODULE, entry="round", qudits=system.qudits[:2]).mlir()


def test_threads_lower_to_llvm_while_others_compile_in_the_same_context(monkeypatch):
    amplitudes = [0.05 + 0.01 * i for i in range(12)]
    expected = {a: (_compile(a).mlir(), _compile(a).mlir("llvm")) for a in amplitudes}
    threads = 12
    for limit in ("256", "3"):
        monkeypatch.setenv(MODULES, limit)
        barrier = threading.Barrier(threads)

        def work(i, barrier=barrier):
            barrier.wait()
            got = []
            for k in range(len(amplitudes)):
                a = amplitudes[(i + k) % len(amplitudes)]
                lowered = _compile(a)
                # Odd threads lower to LLVM; even ones keep compiling and
                # printing in the context the others' modules are in.
                llvm = lowered.mlir("llvm") if (i + k) % 2 else expected[a][1]
                got.append((a, lowered.mlir(), llvm))
            return got

        with ThreadPoolExecutor(threads) as pool:
            results = list(pool.map(work, range(threads)))
        for got in results:
            for a, text, llvm in got:
                assert (text, llvm) == expected[a]
