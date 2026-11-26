# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calling kernels: binding, specialization, caching, guards, quantum systems, returns."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.quantum_systems import MicrowaveDrive, PhysicalQudit, TransmonModel
from cudaq_pulse.quantum_systems._registry import ENV_VAR

from tests._handles import Q, rebuilt

_PI_AMPLITUDE = np.pi / 20.0  # a square pulse of 40 cycles (20 ns) at this rate is a pi pulse


def _device(n=4, **kwargs):
    return pulse.quantum_system(
        name="call-test",
        qudits={
            i: PhysicalQudit(
                i,
                5.0e9 + 1e8 * i,
                0.0,
                0.0,
                anharmonicity_hz=-300e6,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
            )
            for i in range(n)
        },
        **kwargs,
    )


@pulse.kernel
def rabi(q, amp: float, *, duration: int = 40):
    d = get_drive_line(q)
    drive(d, square(duration, amp))


@pulse.kernel
def flip_and_measure(q):
    d = get_drive_line(q)
    drive(d, square(40, _PI_AMPLITUDE))
    r = get_readout_line(q)
    sync(d, r)
    return readout(r, square(20, 0.05))


# ── Binding ──────────────────────────────────────────────────────────────


def test_arguments_bind_like_a_python_call():
    q = _device().qudits[0]
    assert rabi.lower(q, 0.1).schedule[0].duration_cycles == 40
    assert rabi.lower(q, amp=0.1, duration=8).schedule[0].duration_cycles == 8
    with pytest.raises(TypeError, match=r"rabi\(\) takes 2 positional arguments but 3"):
        rabi.lower(q, 0.1, 8)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'amp'"):
        rabi.lower(q)
    with pytest.raises(TypeError, match="unexpected keyword argument 'dur'"):
        rabi.lower(q, 0.1, dur=8)


def test_keyword_varargs_kernels_are_rejected():
    with pytest.raises(TypeError, match=r"cannot take \*\*options"):

        @pulse.kernel
        def k(*qs, **options):
            pass


def test_annotations_split_static_and_runtime_parameters():
    assert rabi.runtime_parameters == ("amp",)
    assert rabi.static_parameters == ("q", "duration")


# ── Compile-only and caching ─────────────────────────────────────────────


def test_compile_takes_no_kernel_arguments_and_float_values_never_recompile():
    q = _device().qudits[0]
    ck = pulse.compile(rabi)
    texts = {ck.lower(q, a).mlir() for a in (0.1, 0.2, 0.3)}
    assert len(texts) == 3
    assert ck.cache_info().compiled == 1
    ck.lower(q, 0.1, duration=8)
    assert ck.cache_info().compiled == 2


def test_a_call_without_qudit_handles_has_no_device():
    @pulse.kernel
    def drive_amp(q, amp: float):
        d = get_drive_line(q)
        drive(d, square(40, amp))

    with pytest.raises(pulse.NoQuantumSystemError, match="without qudit handles"):
        drive_amp.lower(0, 0.1)
    # The process quantum system is not a fallback.
    assert os.environ.get(ENV_VAR)
    with pytest.raises(pulse.NoQuantumSystemError):
        drive_amp(0, 0.1)


def test_compiled_kernel_without_a_specialization_explains_lower():
    with pytest.raises(TypeError, match=r"specialized on q, duration; inspect one .*\.lower"):
        pulse.compile(rabi).mlir()


def test_the_lru_holds_cache_size_specializations(monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_CACHE_SIZE", "2")

    @pulse.kernel
    def wait_for(q, n: int):
        d = get_drive_line(q)
        wait(d, n)

    q = _device().qudits[0]
    for n in (4, 8, 12, 4):
        wait_for.lower(q, n)
    info = wait_for.cache_info()
    assert (info.size, info.compiled, info.evicted, info.capacity) == (2, 4, 2, 2)


def test_cache_size_is_validated(monkeypatch):
    monkeypatch.setenv("CUDAQ_PULSE_CACHE_SIZE", "lots")
    with pytest.raises(ValueError, match="CUDAQ_PULSE_CACHE_SIZE must be a positive integer"):

        @pulse.kernel
        def k(q):
            pass

    with pytest.raises(ValueError, match="cache_size"):
        pulse.kernel(cache_size=0)(lambda q: None)
    monkeypatch.delenv("CUDAQ_PULSE_CACHE_SIZE")
    assert pulse.kernel(lambda q, amp=0.5: None).lower is not None


# ── Python semantics of globals and closures ─────────────────────────────

AMPLITUDE = 0.1
SETTINGS = {"duration": 40}


@pulse.kernel
def reads_globals(q):
    d = get_drive_line(q)
    drive(d, square(SETTINGS["duration"], AMPLITUDE))


def test_a_changed_global_is_seen_on_the_next_call():
    global AMPLITUDE
    q = _device().qudits[0]
    first = reads_globals.lower(q)
    assert reads_globals.lower(q) is first
    AMPLITUDE = 0.2
    try:
        second = reads_globals.lower(q)
        assert second is not first and "2.000000e-01" in second.mlir()
        SETTINGS["duration"] = 8
        assert reads_globals.lower(q).schedule[0].duration_cycles == 8
    finally:
        AMPLITUDE = 0.1
        SETTINGS["duration"] = 40
    assert reads_globals.cache_info().compiled == 3


@dataclasses.dataclass
class _Calibration:
    amplitude: float
    durations: list[int]


CALIBRATIONS = {f"q{i}": _Calibration(0.1, [40]) for i in range(1000)}
SCALES = {"a": 0.5, "b": 0.25}


@pulse.kernel
def subscripts_a_dict(q):
    d = get_drive_line(q)
    entry = CALIBRATIONS["q3"]
    drive(d, square(entry.durations[0], entry.amplitude))


@pulse.kernel
def iterates_a_dict(q):
    d = get_drive_line(q)
    for scale in SCALES.values():
        drive(d, square(8, scale))


@pulse.kernel
def checks_a_dict(q):
    d = get_drive_line(q)
    drive(d, square(8, 0.5 if "c" in SCALES else 0.1))


@pulse.kernel
def measures_a_dict(q):
    d = get_drive_line(q)
    drive(d, square(8 * len(SCALES), 0.1))


def test_a_subscripted_dict_is_guarded_at_the_items_read():
    q = _device().qudits[0]
    first = subscripts_a_dict.lower(q)
    entry = CALIBRATIONS["q3"]
    try:
        CALIBRATIONS["q4"] = _Calibration(0.9, [8])
        CALIBRATIONS["new"] = _Calibration(0.9, [8])
        assert subscripts_a_dict.lower(q) is first
        entry.amplitude = 0.2
        assert "2.000000e-01" in subscripts_a_dict.lower(q).mlir()
        entry.durations.append(4)
        entry.durations[0] = 16
        assert subscripts_a_dict.lower(q).schedule[0].duration_cycles == 16
        CALIBRATIONS["q3"] = _Calibration(0.3, [40])
        assert "3.000000e-01" in subscripts_a_dict.lower(q).mlir()
        del CALIBRATIONS["q3"]
        with pytest.raises(CompilationError, match="subscript failed"):
            subscripts_a_dict.lower(q)
    finally:
        CALIBRATIONS["q3"] = entry
        entry.amplitude, entry.durations[:] = 0.1, [40]
        CALIBRATIONS.pop("new")
        CALIBRATIONS["q4"] = _Calibration(0.1, [40])


@pytest.mark.parametrize(
    ("kernel", "change"),
    [
        (iterates_a_dict, lambda: SCALES.update(b=0.75)),
        (checks_a_dict, lambda: SCALES.update(c=1.0)),
        (measures_a_dict, lambda: SCALES.update(c=1.0)),
    ],
    ids=["iterated", "tested with in", "passed to a call"],
)
def test_a_dict_used_whole_is_guarded_whole(kernel, change):
    q = _device().qudits[0]
    first = kernel.lower(q)
    assert kernel.lower(q) is first
    try:
        change()
        assert kernel.lower(q) is not first
    finally:
        SCALES.clear()
        SCALES.update(a=0.5, b=0.25)


class _Tables(NamedTuple):
    amplitudes: np.ndarray
    widths: dict[str, int]


PAIR = ([0.1], [40])
TABLES = _Tables(np.array([0.1, 0.3]), {"x": 40})
ARRAYS = [np.linspace(0.05, 0.15, 8)]


@pulse.kernel
def reads_a_tuple_of_lists(q):
    drive(get_drive_line(q), square(PAIR[1][0], PAIR[0][0]))


@pulse.kernel
def reads_a_namedtuple_by_index(q):
    drive(get_drive_line(q), square(TABLES[1]["x"], TABLES[0][0]))


@pulse.kernel
def reads_a_list_of_arrays(q):
    drive(get_drive_line(q), square(40, ARRAYS[0][0]))


@pytest.mark.parametrize(
    ("kernel", "amplitudes", "widths", "key"),
    [
        (reads_a_tuple_of_lists, PAIR[0], PAIR[1], 0),
        (reads_a_namedtuple_by_index, TABLES.amplitudes, TABLES.widths, "x"),
    ],
    ids=["tuple of lists", "NamedTuple of an array and a dict"],
)
def test_a_change_inside_a_global_tuple_is_seen_on_the_next_call(kernel, amplitudes, widths, key):
    q = _device().qudits[0]
    first = kernel.lower(q)
    assert kernel.lower(q) is first
    try:
        amplitudes[0] = 0.2
        widths[key] = 16
        second = kernel.lower(q)
        assert "2.000000e-01" in second.mlir()
        assert second.schedule[0].duration_cycles == 16
    finally:
        amplitudes[0] = 0.1
        widths[key] = 40


def test_a_global_list_of_arrays_is_cached_until_an_array_changes():
    q = _device().qudits[0]
    first = reads_a_list_of_arrays.lower(q)
    for _ in range(5):
        assert reads_a_list_of_arrays.lower(q) is first
    try:
        ARRAYS[0][0] = 0.2
        assert "2.000000e-01" in reads_a_list_of_arrays.lower(q).mlir()
    finally:
        ARRAYS[0][0] = 0.05


class _Pulses:
    def __init__(self, amplitude):
        self.amplitude = amplitude

    def scaled(self, factor):
        return self.amplitude * factor


PULSES = _Pulses(0.1)


@pulse.kernel
def calls_a_method(q):
    d = get_drive_line(q)
    drive(d, square(40, PULSES.scaled(1.0)))


def test_a_method_of_a_global_is_cached_until_the_global_changes():
    # A bound method is a new object on every read; the guard compares it
    # by the function and object it binds.
    global PULSES
    q = _device().qudits[0]
    first = calls_a_method.lower(q)
    for _ in range(5):
        assert calls_a_method.lower(q) is first
    assert calls_a_method.cache_info().compiled == 1
    previous = PULSES
    PULSES = _Pulses(0.2)
    try:
        assert "2.000000e-01" in calls_a_method.lower(q).mlir()
    finally:
        PULSES = previous
    assert calls_a_method.cache_info().compiled == 2


def test_a_changed_closure_cell_is_seen_on_the_next_call():
    amplitude = 0.1

    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(40, amplitude))

    q = _device().qudits[0]
    assert "1.000000e-01" in k.lower(q).mlir()
    amplitude = 0.3
    assert "3.000000e-01" in k.lower(q).mlir()


def test_a_trace_that_calls_an_impure_host_function_is_never_reused():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(40, random.random()))

    q = _device().qudits[0]
    k.lower(q)
    k.lower(q)
    assert k.cache_info().compiled == 2


# ── Quantum systems ──────────────────────────────────────────────────────────────


def test_the_device_comes_from_the_qudit_handles():
    a, b = _device(), _device(layout=[3, 2, 1, 0])
    assert "frequency_hz = 5.000000e+09" in rabi.lower(a.qudits[0], 0.1).mlir()
    assert "frequency_hz = 5.300000e+09" in rabi.lower(b.qudits[0], 0.1).mlir()
    assert rabi.lower(b.qudits[0], 0.1).qudits == (3,)


def test_devices_from_the_same_config_are_equal_and_share_specializations():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        wait(d, 4)

    a, b = _device(), _device()
    assert a == b and a is not b
    assert k.lower(a.qudits[1]) is k.lower(b.qudits[1])


def test_mixing_devices_in_one_call_is_an_error():
    @pulse.kernel
    def pair(q0, q1):
        sync(get_drive_line(q0), get_drive_line(q1))

    a, other = _device(), _device(n=5)
    with pytest.raises(pulse.QuantumSystemMismatchError, match="two different quantum systems"):
        pair.lower(a.qudits[0], other.qudits[1])
    pair.lower(a.qudits[0], a.qudits[1])


def test_two_separate_devices_with_one_definition_do_not_mix_in_a_call():
    @pulse.kernel
    def pair(q0, q1):
        sync(get_drive_line(q0), get_drive_line(q1))

    a, twin = _device(), _device()
    assert a == twin
    message = (
        r"^pair\(\) got qudits of two separate quantum systems, both named 'call-test', with "
        r"the same definition, as two pulse.quantum_system\(\) calls make; pass handles of one "
        r"quantum system$"
    )
    with pytest.raises(pulse.QuantumSystemMismatchError, match=message):
        pair.lower(a.qudits[0], twin.qudits[1])
    with pytest.raises(pulse.QuantumSystemMismatchError, match=message):
        pair.lower(a.qudits[0:1], (twin.qudits[1],))


def test_a_pinned_compile_rejects_another_device():
    ck = pulse.compile(rabi, quantum_system=_device())
    with pytest.raises(pulse.QuantumSystemMismatchError, match="compiled for quantum system"):
        ck.lower(_device(n=5).qudits[0], 0.1)


def test_a_pinned_compile_rejects_a_separate_device_with_its_definition():
    dev = _device()
    ck = pulse.compile(rabi, quantum_system=dev)
    with pytest.raises(
        pulse.QuantumSystemMismatchError, match="another, separate quantum system of it"
    ):
        ck.lower(_device().qudits[0], 0.1)
    assert ck.lower(dev.qudits[0], 0.1) is not None


def test_a_call_without_a_device_raises(monkeypatch):
    monkeypatch.delenv(ENV_VAR)
    with pytest.raises(pulse.NoQuantumSystemError, match="without qudit handles"):
        rabi.lower(0, 0.1)


def test_registers_index_like_sequences():
    dev = _device(n=10)
    reg = dev.qudits[0:5]
    assert [q.index for q in reg[1:3]] == [1, 2]
    assert [q.index for q in dev.qudits[[5, 7]]] == [5, 7]
    assert reg[0] == dev.qudits[0]

    @pulse.kernel
    def all_of(qs):
        for q in qs:
            d = get_drive_line(q)
            wait(d, 4)

    assert all_of.lower(dev.qudits[[5, 7]]).qudits == (5, 7)


def test_device_values_are_read_only():
    dev = _device()
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        dev.qudit_freq[0] = 5.01e9
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        dev.transition_freq[0, 0, 1] = 5.01e9
    assert dev.transition_freq[1, 0, 1] == dev.qudit_freq[1] == 5.1e9
    assert dev.transition_freq[1, 1, 2] == pytest.approx(5.1e9 - 300e6)
    with pytest.raises(AttributeError):
        dev._description.name = "other"


def test_a_device_from_a_system_file(tmp_path: Path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps({"name": "transmon_surface17", "layout": [4, 5]}))
    dev = pulse.QuantumSystem.load(path)
    assert [q.index for q in dev.qudits] == [4, 5]
    assert dev == pulse.quantum_system("transmon_surface17", layout=[4, 5])
    assert dev != pulse.quantum_system("transmon_surface17")
    with pytest.raises(
        pulse.QuantumSystemDefinitionError, match="lattice: is not a key of a system"
    ):
        pulse.QuantumSystem.from_dict({"name": "transmon_surface17", "lattice": []})
    with pytest.raises(pulse.QuantumSystemDefinitionError, match="zones: is reserved"):
        pulse.QuantumSystem.from_dict({"name": "transmon_surface17", "zones": []})


def test_a_waveform_as_a_frequency_is_reserved():
    @pulse.kernel
    def chirp(q):
        d = get_drive_line(q)
        set_frequency(d.tone, square(8, 0.1))

    with pytest.raises(CompilationError, match="time-varying frequencies"):
        chirp.lower(_device().qudits[0])


def test_a_qudit_the_target_lacks_is_an_error():
    dev = _device(n=2)
    with pytest.raises(pulse.QuantumSystemMismatchError, match="'call-test' and got qudits of"):
        pulse.compile(rabi, quantum_system=dev).lower(Q[5], 0.1)


# ── Frequencies as float arguments ───────────────────────────────────────


@pulse.kernel
def detuned(q, frequency: float):
    d = get_drive_line(q)
    set_frequency(d.tone, frequency)
    drive(d, square(40, 0.02))


def test_a_float_argument_used_as_a_frequency_stays_symbolic():
    q = _device().qudits[0]
    ck = pulse.compile(detuned)
    populations = []
    for offset in (0.0, 10e6, 40e6):
        lowered = ck.lower(q, 5.0e9 + offset)
        assert "%arg0: f64" in lowered.generic.mlir()
        populations.append(pulse.simulate(lowered, dt_ns=0.1).density_matrix[1, 1].real)
    assert ck.cache_info().compiled == 1
    assert populations[0] > populations[1] > populations[2]


# ── Execution and return values ──────────────────────────────────────────


def test_calling_a_kernel_runs_it_and_returns_none_without_a_return():
    assert rabi(_device().qudits[0], 0.1) is None


def test_a_returned_measurement_is_its_outcome():
    # A call samples its outcome unseeded. On three levels this pi pulse
    # leaves 0.5% outside level 1; on two, outcome 1 is certain.
    q = rebuilt(_device(), model=TransmonModel(levels={0: 2})).qudits[0]
    assert [b.levels for b in pulse.simulate(flip_and_measure.lower(q)).branches] == [(1,)]
    assert flip_and_measure(q) == 1
    assert pulse.compile(flip_and_measure)(q) == 1


def test_returned_values_known_at_compile_time_and_tuples_of_measurements():
    @pulse.kernel
    def both(q0, q1):
        r0 = get_readout_line(q0)
        r1 = get_readout_line(q1)
        return readout(r0, square(20, 0.05)), readout(r1, square(20, 0.05)), 7

    dev = _device()
    assert both(dev.qudits[0], dev.qudits[1]) == (0, 0, 7)


def test_returning_a_value_of_a_float_parameter_is_an_error():
    @pulse.kernel
    def echo(q, amp: float):
        return amp * 2

    with pytest.raises(CompilationError, match="computed from a float parameter"):
        echo.lower(_device().qudits[0], 0.1)


@pulse.classical(max_latency_ns=200)
def _doubled(x: float) -> float:
    return 2 * x


@pulse.kernel
def asks(q, amp: float):
    d = get_drive_line(q)
    drive(d, square(16, amp))
    return _doubled.submit(amp).result()


@pulse.kernel
def asks_and_reads(q, amp: float):
    r = get_readout_line(q)
    m = readout(r, square(20, 0.05))
    return m, _doubled.submit(amp).result()


@pytest.mark.parametrize("kernel", [asks, asks_and_reads])
def test_a_kernel_that_returns_an_answer_compiles_and_only_its_call_fails(kernel):
    q = _device().qudits[0]
    submitted = kernel.lower(q, 0.1).export_to_controller()["events"]
    assert [e["operands"] for e in submitted if e["kind"] == "submit"] == [[{"const": 0.1}]]
    with pytest.raises(CompilationError, match="returns the answer of a classical call"):
        kernel(q, 0.1)


# ── Launcher ─────────────────────────────────────────────────────────────


def test_the_launcher_sets_the_process_target(tmp_path: Path):
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import cudaq_pulse as pulse
            print(pulse.quantum_system().name, sys.argv[1:])
            """
        )
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != ENV_VAR}
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "cudaq_pulse",
            "--quantum-system",
            "transmon_lattice:3",
            str(script),
            "-x",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "transmon_lattice:3 ['-x']"
    assert "quantum system transmon_lattice:3" in run.stderr
    bad = subprocess.run(
        [sys.executable, "-m", "cudaq_pulse", "--quantum-system", "nope", str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert bad.returncode != 0 and "no registered quantum system 'nope'" in bad.stderr


def _launch_module(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m cudaq_pulse *argv`` with a probe module ``launcher_probe`` importable."""
    (tmp_path / "launcher_probe.py").write_text(
        textwrap.dedent(
            """
            import sys
            import cudaq_pulse as pulse
            print(pulse.quantum_system().name, sys.argv[1:])
            """
        )
    )
    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(tmp_path), os.environ.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-m", "cudaq_pulse", *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_the_launcher_passes_a_modules_arguments_through(tmp_path: Path):
    plain = _launch_module(
        tmp_path, "--quantum-system", "transmon_lattice:3", "-m", "launcher_probe", "data.json"
    )
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout.strip() == "transmon_lattice:3 ['data.json']"
    assert "quantum system transmon_lattice:3" in plain.stderr
    dashed = _launch_module(
        tmp_path,
        "--quantum-system",
        "transmon_lattice:3",
        "-m",
        "launcher_probe",
        "--sort-keys",
        "--quiet",
        "data.json",
    )
    assert dashed.returncode == 0, dashed.stderr
    assert dashed.stdout.strip() == "transmon_lattice:3 ['--sort-keys', '--quiet', 'data.json']"
    # the module's --quiet is not the launcher's
    assert "quantum system transmon_lattice:3" in dashed.stderr
    attached = _launch_module(
        tmp_path, "--quantum-system", "transmon_lattice:3", "-mlauncher_probe", "data.json"
    )
    assert attached.returncode == 0, attached.stderr
    assert attached.stdout.strip() == "transmon_lattice:3 ['data.json']"
    assert "quantum system transmon_lattice:3" in attached.stderr


def test_the_launcher_quiet_option_drops_the_quantum_system_banner(tmp_path: Path):
    run = _launch_module(
        tmp_path, "--quantum-system", "transmon_lattice:3", "--quiet", "-m", "launcher_probe", "-x"
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "transmon_lattice:3 ['-x']"
    assert "quantum system" not in run.stderr


def test_the_launcher_needs_a_script_or_a_module(tmp_path: Path):
    neither = _launch_module(tmp_path, "--quantum-system", "transmon_lattice:3")
    assert neither.returncode != 0 and "give either a script path or -m MODULE" in neither.stderr
    bare = _launch_module(tmp_path, "--quantum-system", "transmon_lattice:3", "-m")
    assert bare.returncode != 0 and "argument -m: expected a module name" in bare.stderr


def test_the_launcher_lists_the_quantum_systems_for_a_bad_quantum_system_argument(tmp_path: Path):
    run = _launch_module(
        tmp_path, "--quantum-system", "transmon_lattice:four", "-m", "launcher_probe"
    )
    assert run.returncode != 0
    assert "available quantum systems:" in run.stderr and "transmon_surface17" in run.stderr


# ── Floats that shape the program are specialized on ─────────────────────


def test_a_float_that_decides_control_flow_or_a_duration_is_specialized_on():
    @pulse.kernel
    def shaped(q, amp: float, width: float):
        d = get_drive_line(q)
        if amp > 0.5:
            wait(d, 4)
        drive(d, square(int(width), amp))

    q = _device().qudits[0]
    ck = pulse.compile(shaped)
    assert ck.lower(q, 0.7, 16.0).schedule[-1].duration_cycles == 16
    assert len(ck.lower(q, 0.2, 8.0).schedule) == 1
    assert shaped.static_parameters == ("q", "amp", "width")
    assert shaped.runtime_parameters == ()


def test_a_float_only_used_as_a_value_stays_symbolic():
    q = _device().qudits[0]
    lowered = rabi.lower(q, 0.3)
    assert lowered.arguments == (0.3,)
    assert "%arg0: f64" in lowered.generic.mlir()
    assert "3.000000e-01" in lowered.mlir()


def test_simulating_new_float_values_shares_the_module():
    q = _device().qudits[0]
    first = rabi.lower(q, 0.01)
    second = rabi.lower(q, 0.02)
    a = pulse.simulate(first, dt_ns=0.125).density_matrix[1, 1].real
    b = pulse.simulate(second, dt_ns=0.125).density_matrix[1, 1].real
    assert first.generic is second.generic
    # The CPU path evaluates the lowering at the values: no copy of the module.
    assert first._concrete_module is None and second._concrete_module is None
    assert b > a > 0


def test_lowering_the_same_values_again_specializes_once_they_recur():
    q = _device().qudits[0]
    first = rabi.lower(q, 0.3125)
    program = first.export_to_controller()
    second = rabi.lower(q, 0.3125)
    assert second is not first
    assert second._concrete() is not first._concrete()
    again = rabi.lower(q, 0.3125)
    assert again._concrete() is second._concrete()
    assert again.export_to_controller() == program
    assert rabi.lower(q, 0.4125)._concrete() is not again._concrete()


def test_specialized_modules_are_told_apart_by_their_bits():
    q = _device().qudits[0]
    for value in (0.0, -0.0, 0.0, -0.0):
        rabi.lower(q, value)._concrete()
    assert rabi.lower(q, 0.0)._concrete() is rabi.lower(q, 0.0)._concrete()
    assert rabi.lower(q, -0.0)._concrete() is rabi.lower(q, -0.0)._concrete()
    assert rabi.lower(q, 0.0)._concrete() is not rabi.lower(q, -0.0)._concrete()


@pytest.fixture
def specialized_cache():
    """The native cache of rabi's module, its bounds restored afterwards."""
    module = rabi.lower(_device().qudits[0], 1.0).generic._module
    _, entries, ops = module.specialized_cache()
    yield module.specialized_cache
    module.specialized_cache(entries, ops)


def test_values_seen_once_keep_no_specialized_module(specialized_cache):
    q = _device().qudits[0]
    specialized_cache(entries=specialized_cache()[1])
    for k in range(8):
        rabi.lower(q, 3.0 + k)._concrete()
    assert specialized_cache()[0] == 0


def test_the_specialized_modules_kept_are_bounded(specialized_cache):
    q = _device().qudits[0]
    kept = specialized_cache()[1]
    for k in range(2 * kept):
        for _ in range(2):
            rabi.lower(q, 1.0 + k)._concrete()
        # The most recently used survives other values.
        rabi.lower(q, 1.0)._concrete()
    assert specialized_cache()[0] == kept
    assert rabi.lower(q, 1.0)._concrete() is rabi.lower(q, 1.0)._concrete()
    specialized_cache(entries=2)
    for k in range(8):
        for _ in range(2):
            rabi.lower(q, 2.0 + k)._concrete()
    assert specialized_cache()[0] == 2


def test_a_large_module_keeps_fewer_specialized_modules(specialized_cache):
    q = _device().qudits[0]
    specialized_cache(ops=1)
    for k in range(8):
        for _ in range(2):
            rabi.lower(q, 2.0 + k)._concrete()
    assert specialized_cache()[0] == 1


def test_a_module_changed_in_place_is_not_shared():
    q = _device().qudits[0]
    rabi.lower(q, 5.0)._concrete()
    first = rabi.lower(q, 5.0)._concrete()
    assert rabi.lower(q, 5.0)._concrete() is first
    first.schedule("asap")
    assert rabi.lower(q, 5.0)._concrete() is not first


def test_concurrent_simulations_of_one_specialization_agree():
    from concurrent.futures import ThreadPoolExecutor

    q = _device().qudits[0]
    calls = [rabi.lower(q, 0.01 * (i % 4 + 1)) for i in range(16)]
    expected = [pulse.simulate(c, dt_ns=0.125).density_matrix for c in calls]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda c: pulse.simulate(c, dt_ns=0.125).density_matrix, calls))
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_allclose(g, e, atol=1e-12)


def test_a_qudit_passed_twice_is_an_error():
    @pulse.kernel
    def pair(q0, q1):
        pass

    @pulse.kernel
    def cz(control, target):
        pass

    @pulse.kernel
    def layer(qudits):
        pass

    handles = _device().qudits
    q = handles[0]
    with pytest.raises(
        pulse.InvalidTypeError, match=r"^pair\(\) got qudit 0 for both 'q0' and 'q1'; each"
    ):
        pair.lower(q, q)
    message = (
        r"^cz\(\) got qudit 0 for both 'control' and 'target'; each argument needs a "
        r"distinct qudit$"
    )
    with pytest.raises(TypeError, match=message):
        cz(q, q)
    with pytest.raises(TypeError, match=r"^layer\(\) got qudit 1 twice in 'qudits'; each qudit"):
        layer.lower(handles[[0, 1, 1]])


def test_each_call_returns_its_own_container():
    @pulse.kernel
    def listing(q):
        return [1, 2]

    q = _device().qudits[0]
    first, second = listing(q), listing(q)
    assert first == second == [1, 2] and first is not second


def test_the_schedule_option_is_part_of_the_compiled_kernel():
    @pulse.kernel
    def two(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        drive(d0, square(40, 0.1))
        drive(d0, square(40, 0.1))
        drive(d1, square(20, 0.1))
        sync(d0, d1)

    dev = _device()
    args = (dev.qudits[0], dev.qudits[1])
    starts = {
        policy: [e.start_cycles for e in pulse.compile(two, schedule=policy).lower(*args).schedule]
        for policy in ("asap", "alap")
    }
    assert starts["asap"] != starts["alap"]
    with pytest.raises(ValueError, match="Unknown schedule policy"):
        pulse.compile(two, schedule="soon")


@pytest.mark.parametrize(
    ("given", "meant"), [("ASAP", "asap"), ("ALAP_RCP", "alap_rcp"), ("Rcp", "rcp")]
)
def test_a_schedule_in_the_wrong_case_says_what_was_meant(given, meant):
    @pulse.kernel
    def one(q):
        wait(get_drive_line(q), 4)

    message = (
        rf"^Unknown schedule policy '{given}'\. Did you mean '{meant}'\? "
        r"Choose from: alap, alap_rcp, asap, rcp$"
    )
    with pytest.raises(pulse.InvalidValueError, match=message):
        pulse.compile(one, schedule=given)


def test_an_unknown_schedule_with_nothing_close_lists_the_choices():
    @pulse.kernel
    def one(q):
        wait(get_drive_line(q), 4)

    message = r"^Unknown schedule policy 'soon'\. Choose from: alap, alap_rcp, asap, rcp$"
    with pytest.raises(pulse.InvalidValueError, match=message):
        pulse.compile(one, schedule="soon")


# ── Kernels calling kernels ──────────────────────────────────────────────


@pulse.kernel
def pulse_once(q, amp: float):
    d = get_drive_line(q)
    drive(d, square(8, amp))


def test_a_kernel_called_in_a_range_loop_keeps_the_loop_rolled():
    @pulse.kernel
    def play(d, t, amp: float):
        drive(d, square(8, amp), tone=t)

    @pulse.kernel(unroll_threshold=0)
    def train(q, n: int):
        d = get_drive_line(q)
        for _ in range(n):
            play(d, d.tone, 0.01)

    q = _device().qudits[0]
    small, large = train.lower(q, 4), train.lower(q, 64)
    assert small.metrics.op_count == large.metrics.op_count
    assert large.mlir().count("scf.for") == 1
    assert sum(e.kind == "drive" for e in large.schedule) == 64


def test_a_kernel_called_with_the_loop_variable_sees_each_value():
    @pulse.kernel
    def idle(q, i):
        d = get_drive_line(q)
        wait(d, 4 * (i + 1))

    @pulse.kernel
    def ramp(q, n: int):
        for i in range(n):
            idle(q, i)

    q = _device().qudits[0]
    waits = [e.duration_cycles for e in ramp.lower(q, 3).schedule]
    assert waits == [4, 8, 12]


def test_a_kernel_called_on_a_qudit_indexed_by_the_loop_variable():
    @pulse.kernel
    def each(qs, n: int):
        for i in range(n):
            pulse_once(qs[i], 0.02)

    dev = _device()
    lowered = each.lower(dev.qudits[0:4], 4)
    assert sorted(e.qudit for e in lowered.schedule) == [0, 1, 2, 3]


def test_nested_kernel_calls_and_tone_state_carried_across_iterations():
    @pulse.kernel
    def advance(d, t):
        shift_phase(t, 0.25)
        drive(d, square(8, 0.02), tone=t)

    @pulse.kernel
    def block(d):
        advance(d, d.tone)
        advance(d, d.tone)

    @pulse.kernel
    def carried(q, n: int):
        d = get_drive_line(q)
        for _ in range(n):
            block(d)

    q = _device().qudits[0]
    lowered = pulse.compile(carried, passes=(), unroll_threshold=0).lower(q, 5)
    text = lowered.mlir()
    assert text.count("scf.for") == 1 and text.count("pulse.shift_phase") == 2
    # The tone, with its phase, is carried from one iteration to the next.
    assert "!pulse.tone" in next(line for line in text.splitlines() if "scf.yield" in line)
    assert sum(e.kind == "drive" for e in lowered.schedule) == 10


def test_a_helper_that_calls_a_kernel_is_inlined():
    def helper(q):
        pulse_once(q, 0.05)

    @pulse.kernel
    def via_helper(q):
        helper(q)

    assert sum(e.kind == "drive" for e in via_helper.lower(_device().qudits[0]).schedule) == 1


# ── One drive channel per qudit ──────────────────────────────────────────


@pulse.kernel
def x90(q):
    d = get_drive_line(q)
    drive(d, square(16, 0.02))


def test_two_calls_that_acquire_the_same_drive_line_play_in_sequence():
    @pulse.kernel
    def twice(q):
        x90(q)
        x90(q)

    # Without fuse_pulses, which would merge the two pulses into one.
    lowered = pulse.compile(twice, passes=()).lower(_device().qudits[0])
    assert [(e.start_cycles, e.duration_cycles) for e in lowered.schedule] == [(0, 16), (16, 16)]
    assert lowered.mlir().count("pulse.get_drive_line") == 1


def test_a_callee_shares_the_line_its_caller_holds():
    @pulse.kernel
    def outer(q):
        d = get_drive_line(q)
        shift_phase(d.tone, 0.5)
        drive(d, square(8, 0.02))
        x90(q)
        drive(d, square(8, 0.02))

    lowered = pulse.compile(outer, passes=()).lower(_device().qudits[0])
    assert [e.start_cycles for e in lowered.schedule] == [0, 8, 24]
    # The callee's pulse carries the caller's phase.
    assert lowered.mlir().count("pulse.shift_phase") == 1


def test_two_names_for_one_line_stay_in_step():
    @pulse.kernel
    def aliased(q):
        d1 = get_drive_line(q)
        d2 = get_drive_line(q)
        drive(d1, square(8, 0.02))
        drive(d2, square(8, 0.02))
        drive(d1, square(8, 0.02))

    lowered = pulse.compile(aliased, passes=()).lower(_device().qudits[0])
    assert [e.start_cycles for e in lowered.schedule] == [0, 8, 16]


def test_a_callee_acquiring_the_line_in_a_rolled_loop_is_hoisted():
    @pulse.kernel(unroll_threshold=0)
    def train(q, n: int):
        for _ in range(n):
            x90(q)

    q = _device().qudits[0]
    small, large = train.lower(q, 4), train.lower(q, 64)
    assert small.metrics.op_count == large.metrics.op_count
    text = large.mlir()
    assert text.count("scf.for") == 1 and text.count("pulse.get_drive_line") == 1
    assert text.index("pulse.get_drive_line") < text.index("scf.for")
    assert [e.start_cycles for e in large.schedule] == [16 * i for i in range(64)]


def test_each_arm_of_a_branch_acquires_a_line_it_first_uses():
    @pulse.kernel
    def feedback(q0, q1):
        r = get_readout_line(q0)
        if readout(r, square(20, 0.05)):
            x90(q1)
        else:
            x90(q1)
            x90(q1)

    h = _device().qudits
    lowered = pulse.compile(feedback, passes=()).lower(h[0], h[1])
    assert lowered.mlir().count("pulse.get_drive_line") == 2


@pulse.kernel
def _flip_if_set(q, amp: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(20, 0.05))
    sync(d, r)
    if m:
        drive(d, square(40, amp))


@pulse.kernel
def _same_value_inside_and_after_a_branch(q, amp: float):
    d = get_drive_line(q)
    half = amp * 0.5
    # First made inside the branch of the called kernel, then used after it:
    # it must be made again there, where the branch's value does not reach.
    _flip_if_set(q, half)
    drive(d, square(40, half))


def test_a_value_first_made_in_a_branch_is_made_again_after_it():
    q = _device().qudits[0]
    lowered = pulse.compile(_same_value_inside_and_after_a_branch).lower(q, 0.3)
    drives = [e for e in lowered.schedule if e.kind == "drive"]
    assert len(drives) == 2
