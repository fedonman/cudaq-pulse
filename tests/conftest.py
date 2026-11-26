# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import weakref

import pytest
from cudaq_pulse.quantum_systems._registry import ENV_VAR, register_quantum_system
from cudaq_pulse.runtime.gpu import GpuRuntimeError, require_gpu_runtime

import tests.property_settings  # noqa: F401  (loads the hypothesis profile)
from tests import _levels
from tests._handles import ladder_target

# ``quantum_system()`` without an argument returns the process quantum system.
register_quantum_system("ladder", lambda arg: ladder_target(int(arg) if arg else 64))
os.environ[ENV_VAR] = "ladder:8192"
# The verifications that only catch a bug of the compiler itself, which a
# user's compile skips; read by the first compile, in every child too.
os.environ.setdefault("CUDAQ_PULSE_VERIFY", "1")


@pytest.fixture(autouse=True)
def _guard_forms_agree(monkeypatch):
    """Every cache guard a test builds gives one verdict as closures and as
    generated code, checked when the test ends (kernel/_guard.py)."""
    kernel = importlib.import_module("cudaq_pulse.kernel._kernel")
    guards = importlib.import_module("cudaq_pulse.kernel._guard")
    built = weakref.WeakSet()

    def recorded(make):
        def record(*args, **kwargs):
            guard = make(*args, **kwargs)
            built.add(guard)
            return guard

        return record

    monkeypatch.setattr(kernel, "build_guard", recorded(kernel.build_guard))
    monkeypatch.setattr(kernel, "pin_arguments", recorded(kernel.pin_arguments))
    yield
    for guard in list(built):
        if not guard.impure:
            closures, generated = (
                guards._closures(guard.plan, guard.registers)(),
                guards._generated(guard.plan)(),
            )
            assert closures == generated, (
                f"a guard holds {closures} as closures but {generated} as generated code"
            )


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        help="run the timing tests marked `benchmark` (skipped by default)",
    )
    parser.addoption(
        "--run-nightly",
        action="store_true",
        help="run the slow physics checks marked `nightly` (skipped by default)",
    )
    parser.addoption(
        "--update-perf-baseline",
        action="store_true",
        help="rewrite benchmarks/baselines/frontend_perf.json from this run",
    )
    _levels.add_options(parser)


def pytest_configure(config):
    # Test levels and ``--level`` (tests/_levels.py, tests/levels.toml).
    _levels.register(config)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Selecting any GPU test requires a working GPU runtime.

    Runs after ``-m`` deselection, so ``pytest -m "not gpu"`` never touches
    the GPU, while a run that selects GPU tests fails outright, with the
    specific reason, instead of skipping them.
    """
    if not config.getoption("--run-benchmarks"):
        skip = pytest.mark.skip(reason="timing test; pass --run-benchmarks")
        for item in items:
            if item.get_closest_marker("benchmark"):
                item.add_marker(skip)
    if not config.getoption("--run-nightly"):
        skip = pytest.mark.skip(reason="slow physics check; pass --run-nightly")
        for item in items:
            if item.get_closest_marker("nightly"):
                item.add_marker(skip)
    if not any(item.get_closest_marker("gpu") for item in items):
        return
    try:
        require_gpu_runtime()
    except GpuRuntimeError as error:
        raise pytest.UsageError(
            f"GPU tests were selected but cannot run: {error}\n"
            'Deselect them with `-m "not gpu"` to run the compiler-only '
            "suite."
        ) from error
