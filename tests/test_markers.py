# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pytest markers the workflows select on are registered, used and applied."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent


def test_every_registered_marker_is_used():
    config = tomllib.loads((_TESTS.parent / "pyproject.toml").read_text())
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    registered = {line.split(":", 1)[0].strip() for line in markers}
    used = set()
    for path in _TESTS.rglob("*.py"):
        # tests/test_levels.py writes the level markers in the files of a
        # small project of its own: that is no use of them in this suite.
        if path.name != "test_levels.py":
            used.update(re.findall(r"pytest\.mark\.(\w+)", path.read_text()))
    # A level is used when the level table gives it to a test.
    used.update(tomllib.loads((_TESTS / "levels.toml").read_text()))
    assert registered <= used


@pytest.mark.slow
def test_the_nightly_physics_selection_runs_every_physics_test():
    def collected(*args):
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
            + list(args)
            + [str(_TESTS / "physics")],
            cwd=_TESTS.parent,
            capture_output=True,
            text=True,
            timeout=300,
        )
        match = re.search(r"(\d+)(?:/\d+)? tests? collected", out.stdout)
        assert match, out.stdout + out.stderr
        return int(match.group(1))

    everything = collected()
    assert everything > 0
    assert collected("-m", "physics and not gpu", "--run-nightly") == everything
