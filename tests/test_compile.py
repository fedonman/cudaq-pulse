# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for compiling kernels: ``pulse.compile`` and ``kernel.lower``."""

from __future__ import annotations

import array
import math
import pickle

import cudaq_pulse as pulse
import pytest

from tests._handles import Q, rebuilt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pulse.kernel
def _bell(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf)
    sync(d0, d1)
    drive(d1, wf)


# ---------------------------------------------------------------------------
# Basic compile() usage
# ---------------------------------------------------------------------------


def test_compile_returns_compiled_kernel():
    ck = pulse.compile(_bell)
    assert isinstance(ck, pulse.CompiledKernel)
    assert isinstance(ck.lower(Q[0], Q[1]), pulse.ScheduledKernel)


def test_compile_produces_mlir():
    ck = _bell.lower(Q[0], Q[1])
    mlir = ck.mlir()
    assert "module @" in mlir
    assert "func.func @main()" in mlir
    assert "pulse.gaussian" in mlir
    assert "pulse.drive" in mlir


def test_scheduled_event_start_and_end():
    events = _bell.lower(Q[0], Q[1]).schedule
    assert events
    for e in events:
        assert e.start == e.start_cycles
        assert e.end == e.start_cycles + e.duration_cycles
    unscheduled = pulse.compiler.ScheduledEvent("drive", 0, 0, None, None)
    with pytest.raises(ValueError, match="not scheduled"):
        unscheduled.start  # noqa: B018
    with pytest.raises(ValueError, match="not scheduled"):
        unscheduled.end  # noqa: B018


def test_compile_metrics():
    ck = _bell.lower(Q[0], Q[1])
    m = ck.metrics
    assert isinstance(m, pulse.compiler.CompileMetrics)
    assert m.total_ms > 0
    assert m.trace_ms > 0
    assert m.ffi_ms > 0


def test_compile_no_passes():
    ck = pulse.compile(_bell, passes=()).lower(Q[0], Q[1])
    assert ck.mlir() is not None
    assert "pulse.drive" in ck.mlir()


def test_compile_single_qudit():

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        wf = gaussian(40, 0.3, 10.0)
        drive(d, wf)

    ck = k.lower(Q[0])
    assert "pulse.drive" in ck.mlir()


def _sparse_device():
    from cudaq_pulse.quantum_systems import PhysicalQudit

    qudits = {
        i: PhysicalQudit(i, 5.0e9 + 1e8 * i, 50.0, 30.0, anharmonicity_hz=-300e6) for i in (0, 2)
    }
    return pulse.quantum_system(name="sparse", qudits=qudits)


@pulse.kernel
def _drive_one(q):
    drive(get_drive_line(q), square(40, 0.1))


def test_a_module_records_0_hz_for_a_qudit_its_target_lacks():
    dev = _sparse_device()
    for _ in range(2):
        mlir = _drive_one.lower(dev.qudits[1]).mlir()
        assert (
            "pulse.quantum_system = #pulse.quantum_system<"
            "qudit_frequency_hz = [5.000000e+09, 0.000000e+00, 5.200000e+09]>"
        ) in mlir


def test_a_qudit_the_target_lacks_is_refused_however_its_tables_were_made():
    # Handles of another quantum system reach build() as the qudits it lacks: one
    # in its gap, one past its last, before and after programs on the
    # qudits it defines made its tables.
    from cudaq_pulse.compiler import CompileOptions, build

    options = CompileOptions.make()
    dense = rebuilt(pulse.quantum_system("transmon_surface17"))
    for dev, lacking in ((_sparse_device(), (1, 5)), (dense, (17, 30))):
        kernel = pulse.kernel(_drive_one.__wrapped__)
        handles = dev.qudits
        for handle in (None, handles[0], handles[len(handles) - 1]):
            if handle is not None:
                kernel.lower(handle)
            for index in lacking:
                with pytest.raises(pulse.CompilationError, match=f"uses qudit {index}, which"):
                    build(_drive_one.__wrapped__, [Q[index]], (), dev, options)
        last = handles[len(handles) - 1].index
        build(_drive_one.__wrapped__, [Q[last]], (), dev, options)


def test_compile_folds_phases():

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        shift_phase(d.tone, math.pi / 4)
        wf = gaussian(40, 0.3, 10.0)
        drive(d, wf)

    ck = k.lower(Q[0])
    assert ck.mlir() is not None
    assert ck.metrics.total_ms > 0


def test_compile_with_fusion():

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        sq1 = square(50, 0.2)
        drive(d, sq1)
        sq2 = square(50, 0.2)
        drive(d, sq2)

    ck = k.lower(Q[0])
    assert ck.mlir() is not None


@pulse.kernel
def _ramsey(q, phase: float):
    d = get_drive_line(q)
    drive(d, gaussian(40, 0.3, 10.0))
    shift_phase(d.tone, phase)
    drive(d, gaussian(40, 0.3, 10.0))


def test_a_phase_from_a_float_folds_into_the_drive_once_bound():
    bound = pulse.compile(_ramsey).lower(Q[0], 0.25)
    assert "pulse.shift_phase" in bound.generic.mlir()
    assert "pulse.shift_phase" not in bound.mlir()
    assert "frame_phase_offset = 2.500000e-01" in bound.mlir()
    # Without fold_phases the frame change stays, bound or not.
    assert "pulse.shift_phase" in pulse.compile(_ramsey, passes=()).lower(Q[0], 0.25).mlir()


@pulse.kernel
def _two_squares(q, a: float, b: float):
    d = get_drive_line(q)
    drive(d, square(50, a))
    drive(d, square(50, b))


def test_squares_of_amplitudes_bound_equal_fuse():
    compiled = pulse.compile(_two_squares)
    assert "fused" in compiled.lower(Q[0], 0.2, 0.2).mlir()
    assert "fused" not in compiled.lower(Q[0], 0.2, 0.3).mlir()
    unfused = pulse.compile(_two_squares, passes=("verify", "fold_constants", "fold_phases"))
    assert "fused" not in unfused.lower(Q[0], 0.2, 0.2).mlir()


def test_passes_are_members_or_their_names():
    assert pulse.Pass.FOLD_PHASES == "fold_phases"
    as_members = pulse.compile(_bell, passes=pulse.compiler.DEFAULT_PASSES).lower(Q[0], Q[1])
    as_names = pulse.compile(_bell, passes=[str(p) for p in pulse.compiler.DEFAULT_PASSES])
    assert as_members.mlir() == as_names.lower(Q[0], Q[1]).mlir()
    assert as_members.mlir() == pulse.compile(_bell).lower(Q[0], Q[1]).mlir()


@pytest.mark.parametrize("passes", [(p,) for p in pulse.Pass] + [tuple(pulse.Pass)[::-1]])
def test_every_pass_runs_alone_and_in_any_order(passes):
    lowered = pulse.compile(_ramsey, passes=passes).lower(Q[0], 0.25)
    assert "pulse.drive" in lowered.mlir()
    assert lowered.schedule


def test_an_unknown_pass_names_the_closest_and_lists_them_all():
    with pytest.raises(
        ValueError, match="Unknown pass 'fuse_pulse'. Did you mean 'fuse_pulses'"
    ) as error:
        pulse.compile(_bell, passes=("verify", "fuse_pulse"))
    assert all(p.value in str(error.value) for p in pulse.Pass)
    with pytest.raises(ValueError, match="not the string 'verify'"):
        pulse.compile(_bell, passes="verify")


def test_compile_bad_schedule_raises():
    with pytest.raises(ValueError, match="Unknown schedule"):
        pulse.compile(_bell, schedule="bogus").lower(Q[0], Q[1])


def test_lower_binds_arguments_like_python():
    with pytest.raises(TypeError, match=r"_bell\(\) missing 2 required positional arguments"):
        _bell.lower()


@pulse.kernel
def _two_amplitudes(q0, amplitude: float, detuning: float):
    d0 = get_drive_line(q0)
    drive(d0, square(40, amplitude))
    shift_frequency(d0.tone, detuning)
    drive(d0, square(40, amplitude))


def _unbound_module():
    """The module of a kernel made here, so nothing has bound its values: a
    module's first bind is told from its later ones."""

    @pulse.kernel
    def two(q0, amplitude: float, detuning: float):
        d0 = get_drive_line(q0)
        drive(d0, square(40, amplitude))
        shift_frequency(d0.tone, detuning)
        drive(d0, square(40, amplitude))

    return two.lower(Q[0], 0.1, 1e6).generic


def test_bind_takes_one_real_number_per_parameter():
    generic = _two_amplitudes.lower(Q[0], 0.1, 1e6).generic
    assert generic.parameters == ("amplitude", "detuning")
    expected = generic.bind([0.2, 2e6])
    values = ([0.2, 2e6], (0.2, 2_000_000), [_Real(0.2), 2e6], iter([0.2, 2e6]))
    bound = [generic.bind(v) for v in values]
    assert all(b.mlir() == expected.mlir() for b in bound)
    density = pulse.simulate(expected).density_matrix
    assert all((pulse.simulate(b).density_matrix == density).all() for b in bound)


class _Real:
    def __init__(self, x):
        self.x = x

    def __float__(self):
        return self.x


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            [0.2],
            r"got 1 value for 2 float parameters "
            r"\(amplitude, detuning\); missing the value of 'detuning'",
        ),
        ([], r"missing the value of 'amplitude', 'detuning'"),
        ([0.2, 1e6, 3.0], r"got 3 values .*; 1 more than it has"),
        (0.2, r"takes a list or tuple of values, .*got float"),
    ],
)
def test_bind_names_the_parameters_when_the_count_does_not_fit(values, message):
    generic = _two_amplitudes.lower(Q[0], 0.1, 1e6).generic
    with pytest.raises(pulse.InvalidTypeError, match=message):
        generic.bind(values)


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (
            [True, 1e6],
            pulse.InvalidTypeError,
            r"the value of 'amplitude' must be a real number, got bool True",
        ),
        (
            [0.2, "1e6"],
            pulse.InvalidTypeError,
            r"the value of 'detuning' must be a real number, got str '1e6'",
        ),
        ([0.2, 1j], pulse.InvalidTypeError, r"'detuning' must be a real number, got complex"),
        ([0.2, None], pulse.InvalidTypeError, r"'detuning' must be a real number, got NoneType"),
    ],
)
def test_a_bound_value_that_is_not_a_real_number_fails_where_it_is_bound(values, error, message):
    # The first bind of a module and a later one.
    generic = _unbound_module()
    for _ in range(2):
        with pytest.raises(error, match=message):
            generic.bind(values)
        assert generic.bind([0.3, 2e6]).arguments == (0.3, 2e6)


def test_the_values_of_a_call_are_floats_whatever_numbers_were_bound():
    generic = _unbound_module()
    for _ in range(2):
        for values in ([1, 2], (_Real(1.0), 2), [1.0, 2.0]):
            arguments = generic.bind(values).arguments
            assert arguments == (1.0, 2.0)
            assert [type(v) for v in arguments] == [float, float]


def test_the_first_bind_of_a_module_takes_what_a_later_one_takes():
    for values in (
        lambda: iter([0.2, 2e6]),
        lambda: (v for v in (0.2, 2e6)),
        lambda: {0.2: 0, 2e6: 1},
    ):
        generic = _unbound_module()
        first, later = generic.bind(values()), generic.bind(values())
        assert first.arguments == later.arguments == (0.2, 2e6)
        assert first.mlir() == later.mlir()
    for values in (lambda: iter([0.2]), lambda: 0.2, lambda: None):
        generic = _unbound_module()
        errors = []
        for _ in range(2):
            with pytest.raises(pulse.InvalidTypeError) as raised:
                generic.bind(values())
            errors.append(str(raised.value))
            generic.bind([0.3, 2e6])
        assert errors[0] == errors[1]


class _Volts(float):
    pass


def test_a_number_of_another_type_is_bound_as_a_plain_float():
    np = pytest.importorskip("numpy")
    kinds = (
        lambda: [np.float64(0.25), np.float64(2e6)],
        lambda: [np.float32(0.25), np.float32(2e6)],
        lambda: [_Volts(0.25), _Volts(2e6)],
        lambda: [0.25, np.float64(2e6)],
        lambda: np.array([0.25, 2e6]),
        lambda: [_Real(0.25), 2_000_000],
        lambda: [np.float16(0.25), np.int64(2_000_000)],
        lambda: [0.25, np.int32(2_000_000)],
    )
    for values in kinds:
        generic = _unbound_module()
        for _ in range(3):
            arguments = generic.bind(values()).arguments
            assert arguments == (0.25, 2e6)
            assert [type(v) for v in arguments] == [float, float]


def test_a_bad_value_beside_a_number_of_another_type_fails_where_it_is_bound():
    np = pytest.importorskip("numpy")
    generic = _unbound_module()
    refused = (
        ([np.float64(0.2), True], "'detuning' must be a real number, got bool"),
        ([1, True], "'detuning' must be a real number, got bool"),
        ([_Volts(0.2), "1e6"], "'detuning' must be a real number, got str"),
        ([np.int64(1), None], "'detuning' must be a real number, got NoneType"),
    )
    for _ in range(3):
        for values, message in refused:
            with pytest.raises(pulse.InvalidTypeError, match=message):
                generic.bind(values)
        with pytest.raises(
            pulse.InvalidValueError, match=r"bind\(\): the value of 'detuning' is nan"
        ):
            generic.bind([1, np.float64("nan")])
        both = generic.bind([np.float64(math.inf), np.float64(-math.inf)])
        assert both.arguments == (math.inf, -math.inf)
        # A NaN before a number float() does not take is the error, as it
        # is whatever the types bound before.
        with pytest.raises(pulse.InvalidValueError, match=r"the value of 'amplitude' is nan"):
            generic.bind([math.nan, 10**400])
        with pytest.raises(OverflowError):
            generic.bind([1, 10**400])
        # The numbers beside the refused values are still taken.
        assert generic.bind([np.float64(0.2), np.int64(1)]).arguments == (0.2, 1.0)


def test_a_numpy_bool_is_refused_as_a_bool_is():
    np = pytest.importorskip("numpy")
    for values, name in (
        ([np.True_, 1e6], "amplitude"),
        ([0.2, np.False_], "detuning"),
        ([np.int64(1), np.True_], "detuning"),
        (np.array([True, False]), "amplitude"),
    ):
        generic = _unbound_module()
        for _ in range(3):
            with pytest.raises(
                pulse.InvalidTypeError,
                match=rf"bind\(\): the value of '{name}' must be a real number, got bool",
            ):
                generic.bind(values)
            generic.bind([0.3, 2])


def test_an_infinite_bound_value_fails_where_the_module_is_used():
    generic = _unbound_module()
    for _ in range(2):
        bound = generic.bind([0.2, -math.inf])
        for use in (bound.mlir, lambda bound=bound: pulse.simulate(bound)):
            with pytest.raises(
                pulse.InvalidValueError, match=r"argument 'detuning' is -inf; a float"
            ):
                use()


def test_a_nan_value_fails_where_it_is_bound():
    generic = _two_amplitudes.lower(Q[0], 0.1, 1e6).generic
    with pytest.raises(pulse.InvalidValueError, match=r"bind\(\): the value of 'detuning' is nan"):
        generic.bind([0.2, math.nan])
    with pytest.raises(pulse.InvalidValueError, match=r"argument 'amplitude' is nan"):
        _two_amplitudes.lower(Q[0], math.nan, 1e6)


def test_generic_is_the_module_every_call_rebinds_from():
    call = _two_amplitudes.lower(Q[0], 0.1, 1e6)
    generic = call.generic
    assert generic is not call
    assert generic.parameters == ("amplitude", "detuning")
    assert generic.generic is generic
    assert generic.bind([0.2, 2e6]).generic is generic

    @pulse.kernel
    def no_floats(q):
        drive(get_drive_line(q), square(40, 0.1))

    # A module that holds no values is its own: binding it takes none.
    plain = no_floats.lower(Q[0])
    assert plain.generic is plain
    assert plain.generic.bind([]).mlir() == plain.mlir()


def test_bind_of_a_module_with_its_values_takes_none():
    bound = _two_amplitudes.lower(Q[0], 0.1, 1e6)
    assert bound.parameters == ()
    # Binding no values keeps the call's values: bind returns the call itself.
    assert bound.bind([]) is bound
    assert bound.bind([]).arguments == (0.1, 1e6)
    with pytest.raises(pulse.InvalidTypeError, match=r"no float parameters; a module of one call"):
        bound.bind([0.2])


@pulse.kernel
def _amplitude(q0, amplitude: float):
    drive(get_drive_line(q0), square(8, amplitude))


@pulse.kernel
def _waited(q0, tau: float):
    d0 = get_drive_line(q0)
    drive(d0, square(8, 0.1))
    wait(d0, tau)
    drive(d0, square(8, 0.1))


@pulse.kernel
def _placed_amplitude(q0: pulse.Qudit, amplitude: float):
    drive(get_drive_line(q0), square(8, amplitude))


def test_a_call_bound_with_no_values_exports_its_values():
    again = _amplitude.lower(Q[0], 0.3).bind([])
    assert again.arguments == (0.3,)
    (event,) = again.export_to_controller()["events"]
    assert event["amplitude"] == {"re": 0.3, "im": 0.0}
    assert event["waveform"]["params"]["amplitude"] == 0.3


def test_a_call_bound_with_no_values_keeps_the_wait_its_value_sets():
    again = _waited.lower(Q[0], 30.0).bind([])
    assert again.arguments == (30.0,)
    assert again.schedule == _waited.lower(Q[0], 30.0).schedule
    assert [e.start for e in again.schedule if e.kind == "drive"] == [0, 38]


def test_a_call_with_runtime_qudits_bound_with_no_values_keeps_its_values_and_qudits():
    call = _placed_amplitude.lower(Q[1], 0.2)
    again = call.bind([])
    assert again is call
    assert (again.arguments, again.carriers, again.qudits) == ((0.2,), (1,), (1,))
    (event,) = again.export_to_controller()["events"]
    assert event["amplitude"] == {"re": 0.2, "im": 0.0}


@pulse.kernel
def _amplitudes(q0, amplitudes: list[float]):
    d0 = get_drive_line(q0)
    for amplitude in amplitudes:
        drive(d0, square(8, amplitude))


def _pickled(value):
    return pickle.loads(pickle.dumps(value))


def _limited(kernel):
    """*kernel* compiled under limits that check each call's values when
    they are bound."""
    machine = pulse.ControlSystem(
        controllers=[pulse.Controller(instruction_cycles=4, min_idle_cycles=4, max_amplitude=0.5)]
    )
    return pulse.compile(kernel, control_system=machine)


_CALLS = {
    "floats": lambda: _amplitude.lower(Q[0], 0.3),
    "runtime_qudits": lambda: _placed_amplitude.lower(Q[1], 0.2),
    "float_list": lambda: _amplitudes.lower(Q[0], [0.1, 0.2]),
    "pickled": lambda: _pickled(_amplitude.lower(Q[0], 0.3)),
    "pickled_runtime_qudits": lambda: _pickled(_placed_amplitude.lower(Q[1], 0.2)),
    "limits_wait": lambda: _limited(_waited).lower(Q[0], 30.0),
    "limits_runtime_qudits": lambda: _limited(_placed_amplitude).lower(Q[1], 0.2),
}


@pytest.mark.parametrize("name", list(_CALLS))
def test_a_call_bound_with_no_values_is_that_call(name):
    call = _CALLS[name]()
    assert call.arguments
    assert call.bind([]) is call
    assert call.bind(()) is call


@pytest.mark.parametrize("name", list(_CALLS))
def test_a_call_refuses_values(name):
    with pytest.raises(
        pulse.InvalidTypeError,
        match=r"got 1 value for a module with no float parameters; a module of one call has "
        r"its values already",
    ):
        _CALLS[name]().bind([0.2])


# ---------------------------------------------------------------------------
# PackedIRBuilder tests
# ---------------------------------------------------------------------------


def test_packed_ir_builder_basic():
    """PackedIRBuilder produces valid int64 buffer."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qudit_freq_hz={0: 5e9})
    (q,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (dl, t) = b.emit("pulse.get_drive_line", (q,), ("drive_line", "tone"))
    (wf,) = b.emit(
        "pulse.gaussian", (), ("waveform",), {"duration": 40, "amplitude": 0.5, "sigma": 10.0}
    )
    b.emit("pulse.drive", (dl, wf, t), ("drive_line", "tone"))

    buf = b.get_buffer()
    assert isinstance(buf, array.array)
    assert buf.typecode == "q" and buf.itemsize == 8
    assert len(buf) > 0
    assert b.qudit_count == 1


def test_packed_ir_builder_readout():
    """PackedIRBuilder encodes readout ops."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qudit_freq_hz={0: 5e9})
    (q,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (rl, t) = b.emit("pulse.get_readout_line", (q,), ("readout_line", "tone"))
    (wf,) = b.emit("pulse.square", (), ("waveform",), {"duration": 400, "amplitude": 0.1})
    b.emit("pulse.readout", (rl, wf, t), ("readout_line", "tone", "measurement"))

    buf = b.get_buffer()
    assert (buf[0] & 0xFF) == 1  # ALLOC_READOUT
    assert len(buf) > 8


def test_packed_ir_builder_sync():
    """PackedIRBuilder encodes variable-length sync ops."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qudit_freq_hz={0: 5e9, 1: 5.1e9})
    (q0,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (q1,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (d0, t0) = b.emit("pulse.get_drive_line", (q0,), ("drive_line", "tone"))
    (d1, t1) = b.emit("pulse.get_drive_line", (q1,), ("drive_line", "tone"))
    b.emit("pulse.sync", (d0, d1), ("drive_line", "drive_line"))

    buf = b.get_buffer()
    assert len(buf) > 0


def test_compile_module_available():
    """compile() produces an in-memory PulseModule."""
    ck = _bell.lower(Q[0], Q[1])
    assert ck._module is not None
    assert ck.mlir() is not None


@pulse.kernel
def _stream(q0, amplitudes: list[float], count: int):
    d0 = get_drive_line(q0)
    for i in range(count):
        drive(d0, square(20, amplitudes[i]))


def test_a_call_with_seventy_thousand_values_is_bound():
    # Finding what the values reach went a call deep for every op of the
    # line: from about 65,500 values on, binding them crashed the process.
    count = 70_000
    call = pulse.compile(_stream).lower(Q[0], [float(i % 2) for i in range(count)], count)
    assert call.mlir().count("pulse.drive ") == count
    assert call.arguments[:3] == (0.0, 1.0, 0.0)


_REACHED = {
    "at the top": ("shift_phase(d0.tone, {x})", "drive(d0, square(16, {y}))"),
    "in an arm": (
        "m = readout(r0, square(40, 0.05))",
        "if m:",
        "    shift_phase(d0.tone, {x})",
        "    drive(d0, square(16, {y}))",
        "drive(d0, square(8, 0.1))",
    ),
    "in the other arm": (
        "m = readout(r0, square(40, 0.05))",
        "if m:",
        "    drive(d0, square(8, 0.2))",
        "else:",
        "    shift_phase(d0.tone, 2 * {x} + 0.1)",
        "    drive(d0, square(16, {y} / 2))",
    ),
    "in a loop": (
        "for i in range(40):",
        "    shift_phase(d0.tone, {x})",
        "    drive(d0, square(16, {y}))",
    ),
    "in an arm in a loop": (
        "for i in range(40):",
        "    m = readout(r0, square(40, 0.05))",
        "    if m:",
        "        shift_phase(d0.tone, {x})",
        "        drive(d0, square(16, {y}))",
    ),
}


@pytest.mark.parametrize("where", list(_REACHED))
@pytest.mark.parametrize("passes", [None, ("fold_phases",), ("fuse_pulses",)])
def test_values_bound_inside_an_arm_or_a_loop_fold_as_literals_do(where, passes):
    # The frame changes and square pulses the values reach are folded when
    # they are bound, wherever they are: the call plays what the kernel
    # written with the numbers plays.
    def kernel(signature, x, y):
        head = f"@pulse.kernel\ndef k(q0{signature}):\n    d0 = get_drive_line(q0)\n"
        head += "    r0 = get_readout_line(q0)\n"
        scope = {"pulse": pulse}
        exec(head + "".join(f"    {line.format(x=x, y=y)}\n" for line in _REACHED[where]), scope)  # noqa: S102
        return scope["k"]

    options = {} if passes is None else {"passes": passes}
    call = pulse.compile(kernel(", x: float, y: float", "x", "y"), **options).lower(Q[0], 0.25, 0.4)
    literal = pulse.compile(kernel("", "0.25", "0.4"), **options).lower(Q[0])
    assert call.export_to_controller()["events"] == literal.export_to_controller()["events"]
    assert call.schedule == literal.schedule
