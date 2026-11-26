# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for parameterized pulse kernels.

Covers: MLIR roundtrip, parameterized compilation, __call__ evaluation,
strict scheduling correctness, E2E sweeps, and backward compatibility.
"""

from __future__ import annotations

import math
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import Parameter

from tests._handles import Q
from tests._work import cpu_seconds_each, python_work

# ===========================================================================
# MLIR dialect roundtrip and verifier tests
# ===========================================================================


class TestDialectRoundtrip:
    """Verify SSA-value-based waveform ops produce valid MLIR."""

    def test_gaussian_concrete_roundtrip(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        mlir = ck.mlir()
        assert "pulse.gaussian" in mlir
        assert "arith.constant 40 : i64" in mlir
        assert re.search(r"arith\.constant\s+3\.0+e-01\s*:\s*f64", mlir)
        assert re.search(r"arith\.constant\s+1\.0+e\+01\s*:\s*f64", mlir)

    def test_square_concrete_roundtrip(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = square(20, 0.1)
            drive(d, wf)

        ck = k.lower(Q[0])
        mlir = ck.mlir()
        assert "pulse.square" in mlir
        assert "arith.constant 20 : i64" in mlir

    def test_drag_concrete_roundtrip(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = drag(40, 0.3, 10.0, 0.5)
            drive(d, wf)

        ck = k.lower(Q[0])
        mlir = ck.mlir()
        assert "pulse.drag" in mlir
        assert "arith.constant 40 : i64" in mlir

    def test_cosine_concrete_roundtrip(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = cosine(40, 0.5)
            drive(d, wf)

        ck = k.lower(Q[0])
        mlir = ck.mlir()
        assert "pulse.cosine" in mlir

    def test_tanh_ramp_concrete_roundtrip(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = tanh_ramp(40, 0.5, 5.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        mlir = ck.mlir()
        assert "pulse.tanh_ramp" in mlir

    def test_gaussian_square_concrete_roundtrip(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian_square(100, 0.5, 10.0, 20)
            drive(d, wf)

        ck = k.lower(Q[0])
        mlir = ck.mlir()
        assert "pulse.gaussian_square" in mlir

    def test_parametric_gaussian_has_block_arg(self):

        @pulse.kernel
        def k(q, amplitude: float):
            d = get_drive_line(q)
            wf = gaussian(64, amplitude, 16.0)
            drive(d, wf)

        ck = k.lower(Q[0], 0.5).generic
        mlir = ck.mlir()
        assert "%arg0: f64" in mlir
        assert "pulse.gaussian" in mlir
        assert "%arg0" in mlir

    def test_parametric_has_param_names_attr(self):

        @pulse.kernel
        def k(q, amplitude: float):
            d = get_drive_line(q)
            wf = gaussian(64, amplitude, 16.0)
            drive(d, wf)

        ck = k.lower(Q[0], 0.5).generic
        assert 'pulse.param_names = ["amplitude"]' in ck.mlir()

    def test_concrete_no_block_args(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        assert "func.func @main()" in ck.mlir()
        assert "%arg" not in ck.mlir()

    def test_verifier_passes_concrete_ops(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = pulse.compile(k, passes=("verify",)).lower(Q[0])
        assert ck.mlir() is not None

    def test_scheduling_produces_timing_attrs(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        assert "start_cycles = 0" in ck.mlir()
        assert "duration_cycles = 40" in ck.mlir()


# ===========================================================================
# Parameterized compilation tests
# ===========================================================================


class TestParametricCompilation:
    """Test that compile() detects parameters and builds parametric MLIR."""

    def test_single_param_amplitude(self):

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf = gaussian(64, amp, 16.0)
            drive(d, wf)

        ck = k.lower(Q[0], 0.5).generic
        assert bool(ck.parameters)
        assert ck.parameters == ("amp",)
        assert "%arg0" in ck.mlir()

    def test_multiple_params(self):
        """A float parameter stays symbolic; an int one specializes the kernel."""

        @pulse.kernel
        def k(q, amp: float, duration: int):
            d = get_drive_line(q)
            wf = gaussian(duration, amp, 16.0)
            drive(d, wf)

        ck = k.lower(Q[0], 0.5, 64).generic
        assert ck.parameters == ("amp",)
        assert "%arg0" in ck.mlir()
        assert "%arg1" not in ck.mlir()
        assert "arith.constant 64 : i64" in ck.mlir()

    def test_mixed_concrete_and_param(self):
        """Duration is literal 64, amplitude is parameterized."""

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf = gaussian(64, amp, 16.0)
            drive(d, wf)

        ck = k.lower(Q[0], 0.5).generic
        mlir = ck.mlir()
        assert "arith.constant 64 : i64" in mlir
        assert "%arg0" in mlir  # amplitude is block arg

    def test_all_concrete_backward_compat(self):
        """Kernel with only qubit args compiles as before (no block args)."""

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(64, 0.5, 16.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        assert ck.parameters == ()
        assert ck.generic is ck
        assert "func.func @main()" in ck.mlir()
        assert "start_cycles" in ck.mlir()  # scheduled

    def test_phase_parameter(self):

        @pulse.kernel
        def k(q, phi: float):
            d = get_drive_line(q)
            shift_phase(d.tone, phi)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = k.lower(Q[0], 0.5).generic
        assert bool(ck.parameters)
        assert ck.parameters == ("phi",)
        assert "pulse.shift_phase" in ck.mlir()

    def test_wait_parameter(self):

        @pulse.kernel
        def k(q, delay: int):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)
            wait(d, delay)

        ck = pulse.compile(k)
        q = Q[0]
        assert "pulse.wait" in ck.lower(q, 50).mlir()
        ck.lower(q, 60)
        assert ck.cache_info().compiled == 2

    def test_param_used_in_multiple_ops(self):

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf1 = gaussian(40, amp, 10.0)
            drive(d, wf1)
            wf2 = gaussian(60, amp, 15.0)
            drive(d, wf2)

        ck = k.lower(Q[0], 0.5).generic
        assert bool(ck.parameters)
        # amp block arg used in two gaussian ops
        mlir = ck.mlir()
        assert mlir.count("pulse.gaussian") == 2
        assert len(re.findall(r"= pulse\.drive ", mlir)) == 2


# ===========================================================================
# __call__ evaluation tests
# ===========================================================================


class TestEvaluation:
    """Test compiled(amplitude=0.5) evaluation via specialize()."""

    @pytest.fixture
    def parametric_amp(self):

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf = gaussian(64, amp, 16.0)
            drive(d, wf)

        return pulse.compile(k)

    def test_eval_kwargs(self, parametric_amp):
        result = parametric_amp.lower(Q[0], amp=0.5)
        assert not bool(result.parameters)
        assert "start_cycles" in result.mlir()
        assert "duration_cycles" in result.mlir()

    def test_eval_positional(self, parametric_amp):
        result = parametric_amp.lower(Q[0], 0.5)
        assert "start_cycles" in result.mlir()

    def test_eval_produces_correct_amplitude(self, parametric_amp):
        result = parametric_amp.lower(Q[0], amp=0.5)
        assert re.search(r"arith\.constant\s+5\.0+e-01\s*:\s*f64", result.mlir())

    def test_eval_different_values(self, parametric_amp):
        r1 = parametric_amp.lower(Q[0], amp=0.25)
        r2 = parametric_amp.lower(Q[0], amp=0.75)
        assert "2.500000e-01" in r1.mlir()
        assert "7.500000e-01" in r2.mlir()

    def test_re_evaluation_independence(self, parametric_amp):
        """Re-evaluation with different values returns distinct results."""
        r1 = parametric_amp.lower(Q[0], amp=0.1)
        r2 = parametric_amp.lower(Q[0], amp=0.9)
        assert "1.000000e-01" in r1.mlir()
        assert "9.000000e-01" in r2.mlir()
        # Original kernel is still parametric
        assert bool(parametric_amp.parameters)

    def test_multi_param_eval(self):

        @pulse.kernel
        def k(q, amp: float, duration: int):
            d = get_drive_line(q)
            wf = gaussian(duration, amp, 16.0)
            drive(d, wf)

        ck = pulse.compile(k)
        result = ck.lower(Q[0], amp=0.5, duration=80)
        assert "start_cycles" in result.mlir()
        assert "duration_cycles = 80" in result.mlir()

    def test_eval_wrong_param_count_raises(self, parametric_amp):
        with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
            parametric_amp.lower(Q[0], 0.5, 0.6)

    def test_eval_missing_kwarg_raises(self, parametric_amp):
        with pytest.raises(TypeError, match="unexpected keyword argument 'wrong_name'"):
            parametric_amp.lower(Q[0], wrong_name=0.5)

    def test_eval_unknown_kwarg_raises(self, parametric_amp):
        with pytest.raises(TypeError, match="unexpected keyword argument 'bogus'"):
            parametric_amp.lower(Q[0], amp=0.5, bogus=1.0)

    def test_eval_non_parametric_raises(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
            pulse.compile(k).lower(Q[0], 0.5)

    def test_eval_mixed_positional_kwargs_raises(self, parametric_amp):
        with pytest.raises(TypeError, match="multiple values for argument 'amp'"):
            parametric_amp.lower(Q[0], 0.5, amp=0.3)

    def test_a_float_argument_must_be_a_real_number(self, parametric_amp):
        with pytest.raises(TypeError, match="argument 'amp' must be a real number, not str"):
            parametric_amp.lower(Q[0], "0.5")

    def test_scheduling_correct_after_eval(self, parametric_amp):
        result = parametric_amp.lower(Q[0], amp=0.5)
        mlir = result.mlir()
        assert "start_cycles = 0 : i64" in mlir
        assert "duration_cycles = 64 : i64" in mlir


# ===========================================================================
# Strict scheduling correctness tests
# ===========================================================================


class TestStrictScheduling:
    """Verify exact numeric timing values after evaluate()."""

    @staticmethod
    def _extract_drive_attrs(mlir: str):
        """Extract (start_cycles, duration_cycles) for each pulse.drive op."""
        drives = []
        for m in re.finditer(r"pulse\.drive.*?\{([^}]*)\}", mlir):
            attrs_str = m.group(1)
            start = int(re.search(r"start_cycles\s*=\s*(\d+)", attrs_str).group(1))
            dur = int(re.search(r"duration_cycles\s*=\s*(\d+)", attrs_str).group(1))
            drives.append((start, dur))
        return drives

    def test_single_drive_duration_param(self):

        @pulse.kernel
        def k(q, dur: int):
            d = get_drive_line(q)
            wf = gaussian(dur, 0.5, 16.0)
            drive(d, wf)

        ck = pulse.compile(k)

        result = ck.lower(Q[0], dur=40)
        drives = self._extract_drive_attrs(result.mlir())
        assert drives == [(0, 40)]

        result = ck.lower(Q[0], dur=80)
        drives = self._extract_drive_attrs(result.mlir())
        assert drives == [(0, 80)]

    def test_two_sequential_drives_amplitude_param(self):

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf1 = gaussian(40, amp, 10.0)
            drive(d, wf1)
            wf2 = gaussian(60, amp, 15.0)
            drive(d, wf2)

        ck = pulse.compile(k)
        result = ck.lower(Q[0], amp=0.5)
        drives = self._extract_drive_attrs(result.mlir())
        assert len(drives) == 2
        assert drives[0] == (0, 40)
        assert drives[1] == (40, 60)

    def test_two_qudit_sync_param_duration(self):

        @pulse.kernel
        def k(q0, q1, dur: int):
            d0 = get_drive_line(q0)
            d1 = get_drive_line(q1)
            wf0 = gaussian(dur, 0.5, 16.0)
            drive(d0, wf0)
            sync(d0, d1)
            wf1 = gaussian(40, 0.5, 10.0)
            drive(d1, wf1)

        ck = pulse.compile(k)
        result = ck.lower(Q[0], Q[1], dur=100)
        drives = self._extract_drive_attrs(result.mlir())
        assert len(drives) == 2
        assert drives[0] == (0, 100)
        assert drives[1][0] == 100  # second drive starts after sync
        assert drives[1][1] == 40

        result = ck.lower(Q[0], Q[1], dur=20)
        drives = self._extract_drive_attrs(result.mlir())
        assert drives[0] == (0, 20)
        assert drives[1][0] == 20

    def test_wait_param_duration(self):

        @pulse.kernel
        def k(q, delay: int):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)
            wait(d, delay)
            wf2 = gaussian(40, 0.3, 10.0)
            drive(d, wf2)

        ck = pulse.compile(k)
        result = ck.lower(Q[0], delay=50)
        drives = self._extract_drive_attrs(result.mlir())
        assert len(drives) == 2
        assert drives[0] == (0, 40)
        assert drives[1] == (40 + 50, 40)

    def test_deterministic_re_evaluation(self):

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf = gaussian(64, amp, 16.0)
            drive(d, wf)

        ck = pulse.compile(k)
        mlirs = [ck.lower(Q[0], amp=0.5).mlir() for _ in range(50)]
        assert all(m == mlirs[0] for m in mlirs), "Re-evaluation must produce identical MLIR text"

    def test_parameter_isolation(self):
        """Changing only amplitude must not affect timing."""

        @pulse.kernel
        def k(q, amp: float, dur: int):
            d = get_drive_line(q)
            wf = gaussian(dur, amp, 16.0)
            drive(d, wf)

        ck = pulse.compile(k)

        r1 = ck.lower(Q[0], amp=0.3, dur=64)
        r2 = ck.lower(Q[0], amp=0.7, dur=64)
        d1 = self._extract_drive_attrs(r1.mlir())
        d2 = self._extract_drive_attrs(r2.mlir())
        assert d1 == d2, "Amplitude changes must not affect timing"

        r3 = ck.lower(Q[0], amp=0.3, dur=100)
        d3 = self._extract_drive_attrs(r3.mlir())
        assert d3[0][1] == 100, "Duration change should update timing"
        assert d1[0][1] == 64

    def test_concrete_vs_evaluate_equivalence(self):
        """Concrete compile and parametric evaluate at same values must match timing."""

        @pulse.kernel
        def concrete(q):
            d = get_drive_line(q)
            wf = gaussian(64, 0.5, 16.0)
            drive(d, wf)

        @pulse.kernel
        def parametric(q, amp: float):
            d = get_drive_line(q)
            wf = gaussian(64, amp, 16.0)
            drive(d, wf)

        ck_concrete = concrete.lower(Q[0])
        ck_param = pulse.compile(parametric)
        ck_eval = ck_param.lower(Q[0], amp=0.5)

        d_concrete = self._extract_drive_attrs(ck_concrete.mlir())
        d_eval = self._extract_drive_attrs(ck_eval.mlir())
        assert d_concrete == d_eval, (
            "Concrete and parametric-evaluated must produce identical timing"
        )


# ===========================================================================
# End-to-end integration tests
# ===========================================================================


class TestE2ESweeps:
    """End-to-end tests: amplitude sweep, duration sweep, QEC parameterized."""

    def test_amplitude_sweep(self):

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf = gaussian(64, amp, 16.0)
            drive(d, wf)

        ck = pulse.compile(k)
        for amp_val in [0.1 * i for i in range(1, 11)]:
            result = ck.lower(Q[0], amp=amp_val)
            assert "start_cycles" in result.mlir()
            assert "duration_cycles = 64" in result.mlir()

    def test_duration_sweep(self):

        @pulse.kernel
        def k(q, dur: int):
            d = get_drive_line(q)
            wf = gaussian(dur, 0.5, 16.0)
            drive(d, wf)

        ck = pulse.compile(k)
        for dur_val in [20, 40, 60, 80, 100, 200]:
            result = ck.lower(Q[0], dur=dur_val)
            assert f"duration_cycles = {dur_val}" in result.mlir()

    def test_phase_sweep(self):

        @pulse.kernel
        def k(q, phi: float):
            d = get_drive_line(q)
            shift_phase(d.tone, phi)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = pulse.compile(k)
        for angle in [0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi]:
            result = ck.lower(Q[0], phi=angle)
            assert "start_cycles" in result.mlir()

    def test_qec_parameterized(self):
        """Surface-code-like kernel with parameterized amplitudes."""

        @pulse.kernel
        def k(q0, q1, amp: float):
            d0 = get_drive_line(q0)
            d1 = get_drive_line(q1)
            wf0 = gaussian(40, amp, 10.0)
            drive(d0, wf0)
            sync(d0, d1)
            wf1 = gaussian(40, amp, 10.0)
            drive(d1, wf1)

        ck = pulse.compile(k)
        assert bool(ck.parameters)

        result = ck.lower(Q[0], Q[1], amp=0.25)
        assert "start_cycles" in result.mlir()
        assert len(re.findall(r"= pulse\.drive ", result.mlir())) == 2

    def test_performance_specialize_vs_recompile(self):
        """Specialize must be faster than full recompile."""

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            wf = gaussian(64, amp, 16.0)
            drive(d, wf)

        def recompile():
            k.cache_clear()
            k.lower(Q[0], 0.1)

        # Parametric: compile once, evaluate many
        ck = pulse.compile(k)
        amps = iter(0.1 * (i + 1) for i in range(10**6))

        def specialize():
            ck.lower(Q[0], amp=next(amps))

        # The Python work, counted, and the best of twenty times.
        work = {"compile": python_work(recompile), "specialize": python_work(specialize)}
        assert work["specialize"]["lines"] < work["compile"]["lines"], work
        times = cpu_seconds_each({"compile": recompile, "specialize": specialize}, repeats=20)
        compile_ms, specialize_ms = 1000 * times["compile"], 1000 * times["specialize"]
        assert specialize_ms < compile_ms, (
            f"specialize ({specialize_ms:.2f}ms) should be faster than compile ({compile_ms:.2f}ms)"
        )


# ===========================================================================
# Kernels without parameters
# ===========================================================================


class TestConcreteKernels:
    """Kernels with only concrete arguments compile as before parameters."""

    def test_existing_single_qudit(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        assert "func.func @main()" in ck.mlir()
        assert "start_cycles = 0" in ck.mlir()
        assert "duration_cycles = 40" in ck.mlir()

    def test_existing_two_qudit_sync(self):

        @pulse.kernel
        def k(q0, q1):
            d0 = get_drive_line(q0)
            d1 = get_drive_line(q1)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf)
            sync(d0, d1)
            drive(d1, wf)

        ck = k.lower(Q[0], Q[1])
        assert "pulse.sync" in ck.mlir()
        assert len(re.findall(r"= pulse\.drive ", ck.mlir())) == 2
        assert "start_cycles" in ck.mlir()

    def test_existing_echo_with_wait(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)
            wait(d, 20)
            wf2 = gaussian(40, 0.3, 10.0)
            drive(d, wf2)

        ck = k.lower(Q[0])
        assert "pulse.wait" in ck.mlir()
        assert "start_cycles" in ck.mlir()

    def test_existing_with_shift_phase(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            shift_phase(d.tone, math.pi / 4)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        assert ck.mlir() is not None
        assert "start_cycles" in ck.mlir()

    def test_existing_with_all_passes(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            sq1 = square(50, 0.2)
            drive(d, sq1)
            sq2 = square(50, 0.2)
            drive(d, sq2)

        ck = k.lower(Q[0])
        assert ck.mlir() is not None

    def test_compile_metrics_populated(self):

        @pulse.kernel
        def k(q):
            d = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf)

        ck = k.lower(Q[0])
        assert ck.metrics.total_ms > 0
        assert ck.metrics.trace_ms > 0
        assert ck.metrics.ffi_ms > 0


# ===========================================================================
# Parameter sentinel type tests
# ===========================================================================


class TestParameterType:
    """Test the Parameter sentinel class itself."""

    def test_parameter_creation(self):
        p = Parameter("amp", 0, "f64")
        assert p.name == "amp"
        assert p.index == 0
        assert p.dtype == "f64"

    def test_parameter_repr(self):
        p = Parameter("amp", 0, "f64")
        assert "amp" in repr(p)

    def test_parameter_arithmetic_builds_expression(self):
        p = Parameter("amp", 0, "f64")
        assert (p + 1).op == "add"
        assert (p * 2).op == "mul"
        assert (p - 0.5).op == "sub"


class TestExpressionsAcrossRegions:
    """A float expression the tracer materialized inside a region (a rolled
    loop, a branch arm, a schedule block) is not reused after it closes."""

    def test_an_expression_used_in_a_rolled_loop_and_after_it(self):
        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            doubled = amp * 2
            for _ in range(4):
                drive(d, square(40, doubled))
            drive(d, square(40, doubled))

        lowered = pulse.compile(k, unroll_threshold=0).lower(Q[0], 0.1)
        assert "scf.for" in lowered.generic.mlir()
        assert len(lowered.schedule) == 5

    def test_an_expression_used_in_both_arms_and_after_the_branch(self):
        @pulse.kernel
        def k(q, phase: float):
            d = get_drive_line(q)
            r = get_readout_line(q)
            half = phase / 2
            m = readout(r, square(40, 0.05))
            sync(d, r)
            if m:
                shift_phase(d.tone, half)
            else:
                shift_phase(d.tone, -half)
            shift_phase(d.tone, half)
            drive(d, square(40, 0.1))

        assert len(k.lower(Q[0], 0.4).schedule) == 2

    def test_an_expression_used_in_a_called_kernel_with_its_own_schedule_and_after(self):
        @pulse.kernel(schedule="asap")
        def pulse_pair(line, a):
            drive(line, square(16, a))
            drive(line, square(16, a))

        @pulse.kernel
        def k(q, amp: float):
            d = get_drive_line(q)
            scaled = amp * 0.5
            pulse_pair(d, scaled)
            drive(d, square(16, scaled))

        assert sum(e.duration_cycles for e in k.lower(Q[0], 0.2).schedule) == 48
