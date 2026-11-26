# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importing and compiling stay light; simulation loads its dependencies lazily."""

import importlib
import importlib.metadata
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest

from tests._handles import Q

_ROOT = str(Path(pulse.__file__).parent.parent)


def _run(code: str, tmp_path: Path) -> str:
    script = tmp_path / "probe.py"
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=_ROOT)
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_and_compile_do_not_load_numpy_or_the_gpu_runtime(tmp_path):
    loaded = _run(
        "import sys, cudaq_pulse as p\n"
        "@p.kernel\n"
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(40, 0.1))\n"
        "dev = p.quantum_system('transmon_surface17')\n"
        "p.compile(k).lower(dev.qudits[0])\n"
        "print(sorted(m for m in ('numpy', 'cudaq_pulse.runtime.gpu',"
        " 'cudaq_pulse.runtime._jit', 'importlib.metadata') if m in sys.modules))",
        tmp_path,
    )
    assert loaded == "[]"


def test_exposure_and_idle_windows_load_when_first_read(tmp_path):
    loaded = _run(
        "import sys, cudaq_pulse as p\n"
        "@p.kernel\n"
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(40, 0.1))\n"
        "s = p.compile(k).lower(p.quantum_system('transmon_surface17').qudits[0])\n"
        "names = ('cudaq_pulse._exposure', 'cudaq_pulse._idle_windows')\n"
        "before = [m for m in names if m in sys.modules]\n"
        "s.exposure\n"
        "print(before, [m for m in names if m in sys.modules])",
        tmp_path,
    )
    assert loaded == "[] ['cudaq_pulse._exposure', 'cudaq_pulse._idle_windows']"


def test_viz_loads_matplotlib_only_when_it_plots(tmp_path):
    loaded = _run(
        "import sys, cudaq_pulse as p\n"
        "from cudaq_pulse import viz\n"
        "@p.kernel\n"
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(40, 0.1))\n"
        "s = p.compile(k).lower(p.quantum_system('transmon_surface17').qudits[0])\n"
        "s.timeline\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib'))",
        tmp_path,
    )
    assert loaded == "[]"


def test_a_device_config_without_a_control_system_loads_no_clock_modules(tmp_path):
    loaded = _run(
        "import sys, cudaq_pulse as p\n"
        "from cudaq_pulse.quantum_systems import PhysicalQudit\n"
        "@p.kernel\n"
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(40, 0.1))\n"
        "dev = p.quantum_system(\n"
        "    name='t', qudits={0: PhysicalQudit(0, 5e9, 50.0, 30.0, anharmonicity_hz=-300e6)}\n"
        ")\n"
        "p.compile(k, quantum_system=dev).lower(dev.qudits[0])\n"
        "print(sorted(m for m in ('cudaq_pulse._clocks', 'cudaq_pulse._system',"
        " 'cudaq_pulse.kernel._durations') if m in sys.modules))",
        tmp_path,
    )
    assert loaded == "[]"


def test_links_wait_checks_and_the_classical_report_load_on_first_use(tmp_path):
    out = _run(
        "import sys, cudaq_pulse as p\n"
        "@p.kernel\n"
        "def k(q):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(40, 0.1))\n"
        "p.compile(k).lower(p.quantum_system('transmon_surface17').qudits[0])\n"
        "lazy = ('cudaq_pulse.classical._link', 'cudaq_pulse._waits',"
        " 'cudaq_pulse._classical_report', 'cudaq_pulse._system')\n"
        "print(sorted(m for m in lazy if m in sys.modules))\n"
        "print(p.Link is p.classical.Link, 'Link' in dir(p))",
        tmp_path,
    )
    assert out == "[]\nTrue True"


def test_modules_that_are_not_packages_have_no_module_getattr():
    # Each `from module import name` run in a function looks up
    # module.__path__, which calls a module __getattr__ and fails.
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)  # noqa: F821
        drive(d, square(40, 0.1))  # noqa: F821

    pulse.simulate(pulse.compile(k).lower(Q[0]), levels=2)
    pulse.compile(k).lower(Q[0]).classical_report()
    # Loaded but not yet used: lazy names must not be a module __getattr__.
    importlib.import_module("cudaq_pulse.runtime._calls")
    importlib.import_module("cudaq_pulse.classical.emulation")
    importlib.import_module("cudaq_pulse.controller")
    plain = [
        name
        for name, module in list(sys.modules.items())
        if name.startswith("cudaq_pulse")
        and not hasattr(module, "__path__")
        and "__getattr__" in vars(module)
    ]
    assert plain == []


def test_link_pickles_under_its_public_module():
    assert pulse.Link.__module__ == "cudaq_pulse.classical"
    link = pulse.Link(fixed_ns=3 * pulse.ns, per_controller=True)
    data = pickle.dumps(link)
    assert b"cudaq_pulse.classical\x94" in data and b"_link" not in data
    assert pickle.loads(data) == link


def test_simulation_exports_load_on_first_use(tmp_path):
    out = _run(
        "import sys, cudaq_pulse as p\n"
        "before = 'cudaq_pulse.runtime.simulate' in sys.modules\n"
        "f = p.simulate\n"
        "print(before, 'cudaq_pulse.runtime.simulate' in sys.modules, f.__name__,"
        " p.SimulationResult.__name__, 'simulate' in dir(p))",
        tmp_path,
    )
    assert out == "False True simulate SimulationResult True"


def test_dir_lists_no_internals():
    names = set(dir(pulse))
    leaked = {"_device", "_native", "_build_config", "annotations", "TYPE_CHECKING"}
    assert not names & leaked
    assert not {n for n in names if n.startswith("_") and not n.startswith("__")}
    assert set(pulse.__all__) <= names
    assert "__version__" in names
    # Hidden, not removed.
    assert pulse._device.QuantumSystem is pulse.QuantumSystem


@pytest.mark.parametrize(
    "module",
    [
        "cudaq_pulse",
        "cudaq_pulse._device",
        "cudaq_pulse.doctor",
        "cudaq_pulse.compiler",
        "cudaq_pulse.ops",
        "cudaq_pulse.runtime",
        "cudaq_pulse.runtime.simulate",
        "cudaq_pulse.quantum_systems",
        "cudaq_pulse.quantum_systems.transmon_model",
        "cudaq_pulse.viz",
    ],
)
def test_all_names_resolve(module):
    mod = importlib.import_module(module)
    assert mod.__all__
    for name in mod.__all__:
        assert getattr(mod, name) is not None, name


def test_runtime_exports_the_gpu_errors_without_loading_them(tmp_path):
    out = _run(
        "import sys, cudaq_pulse.runtime as r\n"
        "before = 'cudaq_pulse.runtime.gpu' in sys.modules\n"
        "from cudaq_pulse.runtime import gpu\n"
        "print(before, r.GpuRuntimeError is gpu.GpuRuntimeError,"
        " issubclass(r.GpuSupportNotBuiltError, r.GpuRuntimeError))",
        tmp_path,
    )
    assert out == "False True True"


def test_version_comes_from_the_build():
    from cudaq_pulse import _build_config

    assert pulse.__version__ == _build_config.PACKAGE_VERSION
    assert re.fullmatch(
        r"\d+(\.\d+)+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?(\+[\w.]+)?", pulse.__version__
    )
    try:
        installed = importlib.metadata.version("cudaq-pulse")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    assert installed in (None, pulse.__version__)


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'not_a_thing'"):
        pulse.not_a_thing  # noqa: B018


def test_numpy_scalars_and_arrays_are_accepted_without_numpy_in_the_tracer():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(np.int64(40), np.float64(0.1)))
        drive(d, square(40, np.complex128(0.1 + 0.2j)))
        drive(d, square(40, np.array([0.1, 0.2])))
        drive(d, gaussian(np.int32(40), np.float32(0.25), 10.0))
        drive(d, custom_samples(np.linspace(0.0, 0.1, 8)))

    text = pulse.compile(k, passes=()).lower(Q[0]).mlir()
    assert text.count("= pulse.drive ") == 5
    assert "2.000000e-01" in text
