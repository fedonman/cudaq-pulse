# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Times with units: ``20 * pulse.ns`` in every duration, converted to cycles
of the quantum system's clock exactly when the kernel compiles."""

from __future__ import annotations

import pickle
import re
from fractions import Fraction

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.classical import Link
from cudaq_pulse.controller import DriveLine

from tests._handles import Q

ns, us = pulse.ns, pulse.us


def _events(lowered):
    return [(e.kind, e.start_cycles, e.duration_cycles) for e in lowered.schedule]


# ── The arithmetic ───────────────────────────────────────────────────────────


def test_a_time_holds_its_exact_value():
    assert (0.1 * us).exact_ns == 100
    assert (20 * ns).exact_ns == Fraction(20)
    assert (1 / 3 * ns).exact_ns == Fraction("0.3333333333333333")
    assert isinstance(20 * ns, pulse.Time) and isinstance(20 * ns, float)
    assert float(4 * us) == 4000.0
    assert [repr(t) for t in (20 * ns, 4 * us, 2.5 * pulse.ms, 3 * pulse.s)] == [
        "20 ns",
        "4 us",
        "2.5 ms",
        "3 s",
    ]
    assert pickle.loads(pickle.dumps(0.1 * us)).exact_ns == 100


def test_arithmetic_keeps_the_unit_where_it_means_a_time():
    t = 20 * ns
    for value, want in [
        (t * 2, 40),
        (2 * t, 40),
        (t + 4 * ns, 24),
        (t - 4 * ns, 16),
        (t / 4, 5),
        (-t, -20),
        (abs(-t), 20),
        (sum([t, t, t]), 60),
    ]:
        assert value.__class__ is pulse.Time and value.exact_ns == want
    assert t / (4 * ns) == 5.0 and (t / (4 * ns)).__class__ is float
    assert t // (3 * ns) == 6 and (t % (3 * ns)).exact_ns == 2
    assert 4 * ns < t <= 20 * ns and t > 0 * ns and t == 20 * ns and t != 4 * ns
    assert float(t) == 20.0 and float(2 * us) == 2000.0


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (lambda t: t + 4, "mixes a time with 4 cycles"),
        (lambda t: 4 - t, "mixes a time with 4 cycles"),
        (lambda t: t * t, "a time times a time"),
        (lambda t: 3 / t, "a number over a time"),
        (lambda t: t < 5, "compares a time with a bare number"),
        (lambda t: t**2, "is not a time"),
    ],
)
def test_arithmetic_that_is_not_a_time_is_an_error(expression, message):
    with pytest.raises(TypeError, match=message):
        expression(20 * ns)


@pytest.mark.parametrize(
    "compare",
    [
        lambda t, n: t == n,
        lambda t, n: n == t,
        lambda t, n: t != n,
        lambda t, n: t < n,
        lambda t, n: n <= t,
        lambda t, n: t > n,
        lambda t, n: t >= n,
    ],
)
@pytest.mark.parametrize("number", [20, 20.0, 0])
def test_a_time_compared_with_a_bare_number_is_an_error(compare, number):
    message = r"compares a time with a bare number; compare it with a time \(.* \* pulse\.ns\)"
    with pytest.raises(TypeError, match=message + r", or compare float\(t\)"):
        compare(20 * ns, number)
    with pytest.raises(TypeError, match="compares a time with a bare number"):
        _ = 20 * ns == Fraction(20)


def test_a_time_never_hashes_like_a_number():
    times = [20 * ns, 0 * ns, -3 * us, 0.1 * us, 1 / 3 * ns]
    for t in times:
        assert hash(t) != hash(float(t)) and hash(t) != hash(t.exact_ns)
        assert hash(t) == hash(pulse.Time(t.exact_ns))
    keys = {20 * ns: "time", 20.0: "float"}
    assert len(keys) == 2 and keys[20 * ns] == "time" and keys[20] == "float"
    assert 20.0 not in {20 * ns} and 20 * ns in {20 * ns, 20.0}
    assert {20 * ns, 20 * ns, 0.02 * us} == {20 * ns}
    with pytest.raises(TypeError, match="compares a time with a bare number"):
        _ = 20.0 in [20 * ns]
    assert 20 * ns in [20 * ns, 4 * ns] and "a" != 20 * ns


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (lambda: float("nan") * ns, "finite number of nanoseconds, not nan"),
        (lambda: float("inf") * ns, "finite number of nanoseconds, not inf"),
        (lambda: ns * float("-inf"), "finite number of nanoseconds, not -inf"),
        (lambda: ns / float("inf"), "finite number of nanoseconds, not inf"),
        (lambda: 10**400 * ns, r"at most 1\.7976931348623157e\+308 ns"),
        (lambda: 1e308 * pulse.us, r"at most 1\.7976931348623157e\+308 ns"),
    ],
)
def test_a_time_that_is_not_a_finite_float_is_an_error(expression, message):
    with pytest.raises(pulse.InvalidValueError, match=message):
        expression()


_LONGEST = "a duration is at most 2**53 cycles"
_SHORTEST = "a waveform lasts at least 1 cycle"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("wait(d, -5)", "wait(): duration is -5 cycles; a wait cannot be negative"),
        ("wait(d, -4 * pulse.ns)", "wait(): duration -4 ns is -8 cycles; a wait cannot be"),
        ("drive(d, square(-8, 0.1))", f"square(): duration is -8 cycles; {_SHORTEST}"),
        ("drive(d, square(0, 0.1))", f"square(): duration is 0 cycles; {_SHORTEST}"),
        (
            "drive(d, square(0 * pulse.ns, 0.1))",
            f"square(): duration 0 ns is 0 cycles; {_SHORTEST}",
        ),
        ("drive(d, gaussian(0, 0.1, 10.0))", f"gaussian(): duration is 0 cycles; {_SHORTEST}"),
        ("drive(d, drag(-1, 0.1, 10.0, 0.5))", f"drag(): duration is -1 cycles; {_SHORTEST}"),
        ("drive(d, cosine(0, 0.1))", f"cosine(): duration is 0 cycles; {_SHORTEST}"),
        ("drive(d, tanh_ramp(0, 0.1, 4.0))", f"tanh_ramp(): duration is 0 cycles; {_SHORTEST}"),
        (
            "drive(d, gaussian_square(64, 0.1, 4.0, -1))",
            "gaussian_square(): width is -1 cycles; a duration cannot be negative",
        ),
        ("wait(d, 2**53 + 1)", f"wait(): duration is {2**53 + 1} cycles; {_LONGEST}"),
        ("drive(d, square(2**62, 0.1))", f"square(): duration is {2**62} cycles; {_LONGEST}"),
        ("drive(d, square(2**63, 0.1))", f"square(): duration is {2**63} cycles; {_LONGEST}"),
        ("wait(d, 1e30 * pulse.ns)", f"wait(): duration 1e+21 s is {2 * 10**30} cycles"),
    ],
)
def test_a_duration_outside_its_cycles_is_an_error_naming_the_line(call, message):
    source = f"def k(q):\n    d = get_drive_line(q)\n    {call}\n"
    namespace = {"pulse": pulse}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.kernel(namespace["k"]).lower(Q[0])
    assert str(caught.value).startswith(message)
    assert 'File "<kernel>", line 3, in k' in "\n".join(caught.value.__notes__)


def test_a_negative_wait_parameter_is_an_error():
    @pulse.kernel
    def k(q, n: int):
        d = get_drive_line(q)
        wait(d, n)

    with pytest.raises(pulse.CompilationError, match="duration is -1 cycles; a wait cannot be"):
        k.lower(Q[0], -1)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("wait(d, np.int64(-5))", "wait(): duration is -5 cycles; a wait cannot be negative"),
        ("drive(d, square(np.int32(0), 0.1))", f"square(): duration is 0 cycles; {_SHORTEST}"),
        ("drive(d, square(np.int64(2**60), 0.1))", f"is {2**60} cycles; {_LONGEST}"),
        ("drive(d, square(True, 0.1))", "square duration must be an integer, got bool"),
    ],
)
def test_an_integer_of_another_type_follows_the_same_rule(call, message):
    source = f"def k(q):\n    d = get_drive_line(q)\n    {call}\n"
    namespace = {"pulse": pulse, "np": np}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError, match=re.escape(message)):
        pulse.kernel(namespace["k"]).lower(Q[0])


@pytest.mark.filterwarnings("ignore::cudaq_pulse.PulseWarning")
def test_a_float_argument_compiled_into_a_duration_follows_the_same_rule():
    @pulse.kernel
    def k(q, tau: float):
        d = get_drive_line(q)
        drive(d, square(tau, 0.1))

    assert k.lower(Q[0], 8.0) is not None
    with pytest.raises(pulse.CompilationError, match="duration is 0 cycles; a waveform lasts"):
        k.lower(Q[0], 0.0)


def test_the_longest_duration_is_2_53_cycles():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        wait(d, 2**53)
        wait(d, 2**52 * ns)
        wait(d, 0)

    waits = [e.duration_cycles for e in k.lower(Q[0]).schedule if e.kind == "wait"]
    assert waits == [2**53, 2**53, 0]


# ── Durations ────────────────────────────────────────────────────────────────


def test_a_float_duration_is_an_error_that_shows_both_spellings():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, gaussian(20.0, 0.1, 4.0))

    with pytest.raises(pulse.CompilationError) as caught:
        k.lower(Q[0])
    assert str(caught.value).split("\nwhile tracing")[0] == (
        "gaussian(): duration is a float (20.0). Durations are an int number of cycles, "
        "or a time with units:\n"
        "    gaussian(20, ...)               # 20 cycles\n"
        "    gaussian(20 * pulse.ns, ...)    # 20 nanoseconds"
    )


@pytest.mark.parametrize(
    ("call", "what"),
    [
        ("drive(d, square(8.0, 0.1))", "square(): duration"),
        ("drive(d, gaussian_square(40, 0.1, 4.0, 20.0))", "gaussian_square(): width"),
        ("wait(d, 8.0)", "wait(): duration"),
        ("wait(d, 8, bounds=(0, 40.0))", "wait(): bounds"),
        ("wait(d, pulse.round_duration(d, 2.5))", "round_duration(): duration"),
    ],
)
def test_every_duration_rejects_a_float(call, what):
    source = f"def k(q):\n    d = get_drive_line(q)\n    {call}\n"
    namespace = {"pulse": pulse}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError, match="^" + re.escape(f"{what} is a float")):
        pulse.kernel(namespace["k"]).lower(Q[0])


def test_shape_times_take_fractional_cycles():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, drag(40, 0.1, 4.5, 0.5))

    assert k.lower(Q[0]).schedule


def _same_module(with_units, in_cycles, *args):
    def body(kernel):
        return kernel.lower(Q[0], *args).mlir().split("\n", 1)[1]

    assert body(with_units) == body(in_cycles)


@pulse.kernel
def waveforms_with_units(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, gaussian(20 * ns, 0.1, 2.5 * ns))
    drive(d, square(4 * ns, 0.1))
    drive(d, drag(40 * ns, 0.1, 5 * ns, 0.25 * ns))
    drive(d, cosine(10 * ns, 0.1))
    drive(d, tanh_ramp(10 * ns, 0.1, 1 * ns))
    drive(d, gaussian_square(40 * ns, 0.1, 2 * ns, 20 * ns))
    wait(d, 0.1 * us)
    sync(d, r)
    readout(r, square(1 * us, 0.05))


@pulse.kernel
def waveforms_in_cycles(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    drive(d, gaussian(40, 0.1, 5.0))
    drive(d, square(8, 0.1))
    drive(d, drag(80, 0.1, 10.0, 0.5))
    drive(d, cosine(20, 0.1))
    drive(d, tanh_ramp(20, 0.1, 2.0))
    drive(d, gaussian_square(80, 0.1, 4.0, 40))
    wait(d, 200)
    sync(d, r)
    readout(r, square(2000, 0.05))


def test_times_compile_to_the_module_of_their_cycles():
    _same_module(waveforms_with_units, waveforms_in_cycles)


def _envelope(t):
    return 0.1


@pulse.kernel
def custom_with_units(q):
    d = get_drive_line(q)
    drive(d, custom(4 * ns, _envelope))


@pulse.kernel
def custom_in_cycles(q):
    d = get_drive_line(q)
    drive(d, custom(8, _envelope))


def test_a_custom_envelope_takes_a_time():
    _same_module(custom_with_units, custom_in_cycles)


def test_a_time_that_is_not_whole_cycles_is_an_error():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        wait(d, 0.3 * ns)

    with pytest.raises(pulse.CompilationError) as caught:
        k.lower(Q[0])
    assert str(caught.value).startswith(
        "wait(): duration 0.3 ns is 0.6 cycles of the 2 GHz program clock, and durations "
        "are whole cycles; round it with pulse.round_duration(line, 0.3 * pulse.ns)"
    )


def test_round_duration_rounds_a_time_onto_the_grid():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(pulse.round_duration(d, 10.3 * ns), 0.1))

    machine = ControlSystem(controllers=[Controller(clock_ghz=0.5)])
    assert [e[2] for e in _events(k.lower(Q[0]))] == [21]
    assert [e[2] for e in _events(pulse.compile(k, control_system=machine).lower(Q[0]))] == [24]


def test_loops_and_parameters_take_times():
    @pulse.kernel
    def k(q, step, n: int):
        d = get_drive_line(q)
        for i in range(n):
            drive(d, square(8, 0.1))
            wait(d, step * (i + 1))

    events = _events(k.lower(Q[0], 10 * ns, 3))
    assert [e for e in events if e[0] == "wait"] == [
        ("wait", 8, 20),
        ("wait", 36, 40),
        ("wait", 84, 60),
    ]
    assert k.cache_info().compiled == 1
    k.lower(Q[0], 5 * ns, 3)
    assert k.cache_info().compiled == 2

    @pulse.kernel(unroll_threshold=0)
    def rolled(q, step):
        d = get_drive_line(q)
        for _ in range(40):
            drive(d, square(8, 0.1))
            wait(d, step)

    lowered = rolled.lower(Q[0], 10 * ns)
    assert "scf.for" in lowered.generic.mlir()
    assert lowered.schedule[-1].start_cycles == 39 * 28 + 8


def test_a_float_parameter_in_nanoseconds_is_one_run_time_multiply():
    @pulse.kernel
    def k(q, tau: float):
        d = get_drive_line(q)
        drive(d, square(8, 0.1))
        wait(d, tau * ns, bounds=(0, 1 * us))
        drive(d, square(8, 0.1))

    compiled = pulse.compile(k)
    assert [e[1] for e in _events(compiled.lower(Q[0], 30.0)) if e[0] == "drive"] == [0, 68]
    assert [e[1] for e in _events(compiled.lower(Q[0], 50.0)) if e[0] == "drive"] == [0, 108]
    assert compiled.cache_info().compiled == 1
    with pytest.raises(TypeError, match=r"argument 'tau' is a time \(30 ns\)"):
        compiled(Q[0], 30 * ns)


def test_a_time_in_a_run_time_list_float_is_an_error():
    @pulse.kernel
    def k(q, taus: list[float]):
        d = get_drive_line(q)
        for t in taus:
            wait(d, t)

    for lower in (k.lower, pulse.compile(k).lower):
        with pytest.raises(pulse.InvalidTypeError, match=r"taus\[1\] is a time \(100 ns\)"):
            lower(Q[0], [50.0, 100 * ns])
        with pytest.raises(
            pulse.InvalidTypeError, match=r"taus\[1\] must be a real number, not str"
        ):
            lower(Q[0], [50.0, "x"])
    assert [e[2] for e in _events(k.lower(Q[0], [50.0, 100.0]))] == [50, 100]


def test_a_list_float_element_can_be_a_duration_like_a_float_parameter():
    @pulse.kernel
    def k(q, widths: list[float]):
        d = get_drive_line(q)
        drive(d, square(widths[0], 0.1))
        wait(d, widths[1])

    with pytest.warns(pulse.PulseWarning, match="'widths'"):
        assert _events(k.lower(Q[0], [16.0, 8.0])) == [("drive", 0, 16), ("wait", 16, 8)]
    assert "widths" in k.static_parameters
    assert _events(k.lower(Q[0], [12.0, 4.0])) == [("drive", 0, 12), ("wait", 12, 4)]


def test_mixing_units_in_a_run_time_length_is_an_error():
    @pulse.kernel
    def k(q, tau: float):
        d = get_drive_line(q)
        wait(d, tau * ns + 4)

    with pytest.raises(pulse.CompilationError, match="mixes a time with a number of cycles"):
        k.lower(Q[0], 1.0)


@pulse.classical(max_latency_ns=60, link="controller")
def wait_length(x: float) -> float:
    return x


@pulse.kernel
def answer_in_nanoseconds(q):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    f = wait_length.submit(1.0)
    wait(d, f.result() * pulse.ns, bounds=(0, 4 * pulse.us))
    drive(d, square(8, 0.1))


def test_an_answer_in_nanoseconds_converts_with_one_multiply():
    machine = ControlSystem(links={"controller": Link(fixed_ns=10)})
    text = pulse.compile(answer_in_nanoseconds, control_system=machine).lower(Q[0]).mlir()
    taken = text.index("classical.wait")
    converted = text[taken:]
    assert converted.count("arith.mulf") == 1
    assert "arith.constant 2.000000e+00 : f64" in text
    assert "pulse.max_cycles = 8000" in converted


@pulse.classical(max_latency_ns=400)
def check_bit(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


def test_a_timeout_takes_a_time():
    @pulse.kernel
    def timed(q, timeout):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(100, 0.05))
        ok, fix = check_bit.submit([m]).result(timeout=timeout)
        sync(d, ro)
        if ok:
            drive(d, square(20, 0.3))

    with_units = timed.lower(Q[0], 0.3 * us)
    in_ns = timed.lower(Q[0], 300)
    assert with_units.mlir() == in_ns.mlir()
    exports = (with_units.export_to_controller(), in_ns.export_to_controller())
    answers = [next(e for e in p["events"] if e["kind"] == "answer") for p in exports]
    assert answers[0]["timeout_ns"] == answers[1]["timeout_ns"] == 300


def _timeout_error(timeout):
    source = (
        "def k(q):\n"
        "    ro = get_readout_line(q)\n"
        "    m = readout(ro, square(100, 0.05))\n"
        f"    check_bit.submit([m]).result({timeout})\n"
    )
    namespace = {"pulse": pulse, "check_bit": check_bit}
    exec(compile(source, "<kernel>", "exec"), namespace)
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.kernel(namespace["k"]).lower(Q[0])
    return str(caught.value).split("\nwhile tracing")[0]


@pytest.mark.parametrize("timeout", ["timeout=0.3 * pulse.ns", "timeout=0.3"])
def test_a_timeout_off_the_cycles_is_an_error(timeout):
    assert _timeout_error(timeout).startswith(
        "result(): timeout 0.3 ns is 0.6 cycles of the 2 GHz program clock, and durations "
        "are whole cycles; round it with pulse.round_duration(line, 0.3 * pulse.ns)"
    )


@pytest.mark.parametrize(
    ("timeout", "message"),
    [
        ("timeout_ns=300", "result() got an unexpected keyword 'timeout_ns'"),
        ("timeout=0", "result(): timeout must be positive, got 0"),
        ("timeout=-2 * pulse.ns", "result(): timeout must be positive"),
    ],
)
def test_timeout_misuse(timeout, message):
    assert _timeout_error(timeout).startswith(message)


# ── One kernel, two clocks ───────────────────────────────────────────────────


@pulse.kernel
def x_then_wait(q, gap, x_duration, x_sigma, x_amplitude):
    d = get_drive_line(q)
    drive(d, gaussian(x_duration, x_amplitude, x_sigma))
    wait(d, gap)
    drive(d, gaussian(x_duration, x_amplitude, x_sigma))


def test_one_kernel_compiles_for_devices_with_different_clocks():
    fast = pulse.quantum_system("transmon_surface17", clock_ghz=2.0)
    slow = pulse.quantum_system("transmon_surface17", clock_ghz=1.0)
    x_duration, x_sigma, x_amplitude, gap = 20 * ns, 5 * ns, 0.2, 0.1 * us
    on_fast = pulse.compile(x_then_wait, quantum_system=fast).lower(
        fast.qudits[0], gap, x_duration, x_sigma, x_amplitude
    )
    on_slow = pulse.compile(x_then_wait, quantum_system=slow).lower(
        slow.qudits[0], gap, x_duration, x_sigma, x_amplitude
    )
    assert _events(on_fast) == [("drive", 0, 40), ("wait", 40, 200), ("drive", 240, 40)]
    assert _events(on_slow) == [("drive", 0, 20), ("wait", 20, 100), ("drive", 120, 20)]
    # The same nanoseconds on both.
    for lowered, quantum_system in ((on_fast, fast), (on_slow, slow)):
        end = lowered.schedule[-1].start_cycles + lowered.schedule[-1].duration_cycles
        assert end / quantum_system.clock_ghz == 140.0


# ── Times outside kernels ────────────────────────────────────────────────────


@pulse.kernel
def _pulse_then_idle(q):
    d = get_drive_line(q)
    drive(d, gaussian(64, 0.3, 16))
    wait(d, 80)


def test_simulate_takes_its_end_and_step_as_times():
    by_number = pulse.simulate(_pulse_then_idle, Q[0], t_end=40.0, dt_ns=0.125)
    for t_end, dt_ns in ((40 * ns, 0.125 * ns), (0.04 * us, 0.125), (40.0, ns / 8)):
        by_time = pulse.simulate(_pulse_then_idle, Q[0], t_end=t_end, dt_ns=dt_ns)
        assert by_time.times_ns == by_number.times_ns == (0.0, 40.0)
        assert by_time.level_populations() == by_number.level_populations()


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ({"t_end": -1 * ns}, "must not precede"),
        ({"dt_ns": 0 * ns}, "dt_ns must be positive"),
    ],
)
def test_simulate_checks_a_time_as_its_number_of_ns(time, message):
    with pytest.raises(pulse.InvalidValueError, match=message):
        pulse.simulate(_pulse_then_idle, Q[0], **time)


def test_a_classical_function_takes_its_latencies_as_times():
    @pulse.classical(
        max_latency_ns=0.4 * us, min_latency_ns=100 * ns, deadline_ns=2 * us, link="controller"
    )
    def bounded(b: pulse.Bits) -> pulse.Bits:
        return b

    @pulse.classical(
        max_latency_ns=400 * ns, latency_mode="timeout", timeout_ns=1 * us, link="controller"
    )
    def timed(b: pulse.Bits) -> pulse.Bits:
        return b

    assert (bounded.max_latency_ns, bounded.min_latency_ns) == (400.0, 100.0)
    assert (bounded.nominal_latency_ns, bounded.deadline_ns) == (400.0, 2000.0)
    assert timed.timeout_ns == 1000.0
    for value in (bounded.max_latency_ns, bounded.deadline_ns, timed.timeout_ns):
        assert value.__class__ is float
    with pytest.raises(pulse.InvalidValueError, match="min <= nominal <= max"):

        @pulse.classical.external(max_latency_ns=100 * ns, min_latency_ns=0.2 * us)
        def late() -> float:
            return float()


def test_a_machine_takes_its_link_and_line_times_as_times():
    link = Link(fixed_ns=0.1 * us, feedback_ns=20 * ns)
    assert link == Link(fixed_ns=100, feedback_ns=20)
    assert link.fixed_ns.__class__ is float and link.feedback_ns.__class__ is float
    limits = Controller(max_fine_delay_ns=2 * ns)
    assert limits == Controller(max_fine_delay_ns=2.0)
    line = DriveLine(0, delay_ns=3 * ns)
    assert line == DriveLine(0, delay_ns=3.0) and line.delay_ns.__class__ is float
    system = ControlSystem(controllers=[Controller(drive_lines=[line])])
    assert system._line_delays == {("drive", 0): 3.0}
    with pytest.raises(pulse.InvalidValueError, match="cannot be negative"):
        DriveLine(0, delay_ns=-1 * ns)
