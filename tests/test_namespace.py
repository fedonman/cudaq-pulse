# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public namespace is frozen: these names, and only these."""

import re
import subprocess
import sys

import cudaq_pulse as pulse
import pytest

from tests._tools import tool_env

TOP_LEVEL = {
    "kernel",
    "Kernel",
    "compile",
    "load",
    "CompiledKernel",
    "ScheduledKernel",
    "Pass",
    "simulate",
    "SimulationResult",
    "BranchCoverage",
    "quantum_system",
    "QuantumSystem",
    "Qudit",
    "Register",
    "Controller",
    "ControlSystem",
    "Processor",
    "Capability",
    "Link",
    "PulseError",
    "Problem",
    "QuantumSystemDefinitionError",
    "InvalidValueError",
    "InvalidTypeError",
    "NotFoundError",
    "IndexOutOfRangeError",
    "UnsupportedError",
    "CompilationError",
    "ErrorCode",
    "TimingError",
    "ExecutionError",
    "InternalError",
    "PulseWarning",
    "UpgradeNotice",
    "UnitWarning",
    "IdleReplacement",
    "NoProcessQuantumSystemError",
    "NoQuantumSystemError",
    "QuantumSystemMismatchError",
    "GpuRuntimeError",
    "round_duration",
    "RoundDirection",
    "Time",
    "ns",
    "us",
    "ms",
    "s",
    "viz",
    "Bits",
    "Length",
    "classical",
    "controller",
    "backends",
    "gates",
    "Event",
    "__version__",
}

DSL = {
    "get_drive_line",
    "get_readout_line",
    "drive",
    "readout",
    "wait",
    "sync",
    "shift_phase",
    "set_phase",
    "shift_frequency",
    "set_frequency",
    "gaussian",
    "square",
    "drag",
    "cosine",
    "tanh_ramp",
    "gaussian_square",
    "custom",
    "custom_samples",
    "wf_add",
    "wf_sub",
    "wf_mul",
    "wf_scale",
    "wf_neg",
}


def test_dir_is_exactly_the_public_names():
    assert set(dir(pulse)) == TOP_LEVEL
    assert set(pulse.__all__) == TOP_LEVEL - {"__version__"}


def test_every_public_name_resolves():
    for name in TOP_LEVEL:
        assert getattr(pulse, name) is not None, name


@pytest.mark.parametrize(
    "name",
    [
        "qudit_ref",
        "qvec_ref",
        "QuditRef",
        "QvecRef",
        "drive",
        "gaussian",
        "CompileMetrics",
        "ScheduledEvent",
        "DEFAULT_PASSES",
        "Parameter",
    ],
)
def test_names_that_moved_or_went_are_not_top_level(name):
    with pytest.raises(AttributeError):
        getattr(pulse, name)


def test_the_kernel_language_is_in_ops():
    assert DSL <= set(dir(pulse.ops))
    assert len(DSL) == 23


def test_ops_all_is_the_kernel_language_the_tracer_lowers():
    from cudaq_pulse.kernel.ir_builder import OP_TABLE

    assert set(pulse.ops.__all__) == DSL == set(OP_TABLE)
    assert len(pulse.ops.__all__) == len(DSL)


def test_ruff_builtins_are_the_kernel_language():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    builtins = tomllib.loads(pyproject.read_text())["tool"]["ruff"]["builtins"]
    assert sorted(builtins) == sorted(pulse.ops.__all__)


def _stub_imports(stub):
    return [ln for ln in stub.read_text().splitlines() if ln.startswith("from ")]


def test_the_installed_builtins_stub_reexports_exactly_ops_all():
    from pathlib import Path

    # Beside the package, at the root of site-packages (or of build/*/python).
    stub = Path(pulse.__file__).parent.parent / "__builtins__.pyi"
    assert _stub_imports(stub) == [
        f"from cudaq_pulse.ops import {n} as {n}" for n in pulse.ops.__all__
    ]


def test_builtins_stub_reexports_exactly_ops_all(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    generator = root / "cmake" / "write_builtins_stub.py"
    if not generator.exists():
        pytest.skip("needs the source tree; a wheel's tests check the installed stub")
    spec = importlib.util.spec_from_file_location("write_builtins_stub", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "__builtins__.pyi"
    module.main(
        [
            "write_builtins_stub.py",
            str(root / "core/frontend/cudaq_pulse/ops/__init__.py"),
            str(out),
        ]
    )
    assert _stub_imports(out) == [
        f"from cudaq_pulse.ops import {n} as {n}" for n in pulse.ops.__all__
    ]


def test_compiler_types_are_in_the_compiler_module():
    from cudaq_pulse import compiler

    for name in ("CompileMetrics", "ScheduledEvent", "DEFAULT_PASSES"):
        assert hasattr(compiler, name)


def test_device_is_the_one_device_getter():
    import cudaq_pulse.quantum_systems as quantum_systems

    for removed in ("get_device", "Target"):
        assert not hasattr(pulse, removed)
        assert not hasattr(quantum_systems, removed)


def test_the_release_ships_no_neutral_atom_target():
    import cudaq_pulse.quantum_systems as quantum_systems

    assert not [n for n in quantum_systems.available_quantum_systems() if n.startswith("rydberg")]
    assert not [n for n in quantum_systems.__all__ if n.lower().startswith("rydberg")]


# Neutral-atom work is kept out of the release; a line that must name one of
# these says ``scope-ok``.
_OUT_OF_SCOPE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"rydberg",
        r"neutral[ _-]?atoms?",
        r"tweezers?",
        r"\bblockade\b",
        r"atom[ _-]?arrays?",
        r"van der waals",
        r"\batoms?\b",
        r"\d+atoms\b",
    )
] + [re.compile(p) for p in (r"\bC6\b", r"\bRb-?87\b")]
_SCOPE_EXEMPT = {"uv.lock", "tests/test_namespace.py"}


def test_tracked_files_have_no_neutral_atom_remnants():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            env=tool_env(),
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout")
    hits = []
    for name in listed.decode().split("\0"):
        if not name or name in _SCOPE_EXEMPT or not (root / name).is_file():
            continue
        try:
            text = (root / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if "scope-ok" not in line and any(p.search(line) for p in _OUT_OF_SCOPE):
                hits.append(f"{name}:{number}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_viz_loads_on_first_use():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, cudaq_pulse as p; before = 'cudaq_pulse.viz' in sys.modules; "
            "p.viz; print(before, 'cudaq_pulse.viz' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False True"


def test_every_public_error_is_a_pulse_error_and_keeps_its_builtin_base():
    import cudaq_pulse.backends
    import cudaq_pulse.compiler
    import cudaq_pulse.controller
    import cudaq_pulse.quantum_systems
    import cudaq_pulse.runtime

    builtin = {
        "InvalidValueError": ValueError,
        "InvalidTypeError": TypeError,
        "NotFoundError": KeyError,
        "IndexOutOfRangeError": IndexError,
        "UnsupportedError": NotImplementedError,
        "CompilationError": RuntimeError,
        "TimingError": RuntimeError,
        "ExecutionError": RuntimeError,
        "InternalError": RuntimeError,
        "UnsupportedInterpreterError": NotImplementedError,
        "NoProcessQuantumSystemError": RuntimeError,
        "NoQuantumSystemError": RuntimeError,
        "QuantumSystemMismatchError": RuntimeError,
        "QuantumSystemDefinitionError": ValueError,
        "UnknownQuantumSystemError": KeyError,
        "GpuRuntimeError": RuntimeError,
        "GpuSupportNotBuiltError": RuntimeError,
        "GpuRuntimeLoadError": RuntimeError,
        "GpuDeviceUnavailableError": RuntimeError,
        "BackendError": Exception,
        "BackendNotFoundError": KeyError,
        "IncompatibleBackendError": TypeError,
        "ControllerLimitError": RuntimeError,
    }
    errors = {
        name: getattr(module, name)
        for module in (
            pulse,
            cudaq_pulse.backends,
            cudaq_pulse.compiler,
            cudaq_pulse.controller,
            cudaq_pulse.runtime,
            cudaq_pulse.quantum_systems,
        )
        for name in module.__all__
        if isinstance(getattr(module, name), type)
        and issubclass(getattr(module, name), Exception)
        and not issubclass(getattr(module, name), Warning)
    }
    assert set(errors) == set(builtin) | {"PulseError"}
    for name, base in builtin.items():
        assert issubclass(errors[name], pulse.PulseError), name
        assert issubclass(errors[name], base), name


def test_every_public_warning_is_a_pulse_warning():
    assert issubclass(pulse.PulseWarning, UserWarning)
    assert issubclass(pulse.UpgradeNotice, pulse.PulseWarning)
    assert issubclass(pulse.UnitWarning, pulse.PulseWarning)
    assert not issubclass(pulse.PulseWarning, pulse.PulseError)
