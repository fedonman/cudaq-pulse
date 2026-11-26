# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test levels (tests/_levels.py): the level table names tests that exist,
level 0 stays small, a test has its level from one place; and, in a small
project of its own, how a level is found and what ``--level`` selects."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import _levels

_TESTS = Path(__file__).resolve().parent
_TABLE = _TESTS / "levels.toml"
#: The most cases level 0 may hold. Raising it is a decision, not a fix.
#: The four guards below are at level 0 themselves, so that a run of level 0
#: alone checks the table it selects by.
_MAX_LEVEL0_CASES = 600


def _pytest(cwd, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_TESTS.parent)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    # The child's options and plain output are the ones written here.
    for name in ("PYTEST_ADDOPTS", "FORCE_COLOR", "PY_COLORS"):
        env.pop(name, None)
    env["NO_COLOR"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.fixture(scope="module")
def suite(request, tmp_path_factory):
    """Every test of the suite with its level, and the files that skipped
    themselves: from this session when it collected the whole suite, else
    from a collection in a child process."""
    plugin = request.config.pluginmanager.get_plugin(_levels.PLUGIN)
    if plugin.whole_suite():
        return plugin.collected, plugin.skipped_files
    out = tmp_path_factory.mktemp("levels") / "levels.json"
    # The levels are written before -m deselects, so the GPU tests are in
    # them, and the child asks for no GPU.
    run = _pytest(
        _TESTS.parent, str(_TESTS), "--collect-only", "-q", "-m", "not gpu", f"--level-json={out}"
    )
    assert run.returncode == 0 and out.exists(), (run.stdout + run.stderr)[-3000:]
    return _levels.load(out.read_text())


@pytest.mark.level0
def test_every_table_entry_is_a_test_file_with_a_reason():
    for entry in _levels.load_table(_TABLE):
        assert (_TESTS.parent / entry.path).is_file(), f"{entry.key}: no such file"
        assert Path(entry.path).name.startswith("test_"), entry.key
        assert len(entry.reason) <= 200, f"{entry.key}: the reason is one short line"


@pytest.mark.level0
def test_every_table_entry_names_a_collected_test(suite):
    tests, skipped_files = suite
    stale = _levels.stale_entries(_levels.load_table(_TABLE), tests, skipped_files)
    assert not stale, f"tests/levels.toml names no collected test with: {stale}"


@pytest.mark.level0
def test_a_test_has_its_level_from_a_marker_or_from_the_table(suite):
    tests, _ = suite
    twice = _levels.levelled_twice(tests)
    assert not twice, (
        "these tests carry a level marker and are named in tests/levels.toml; "
        f"remove the table entry: {twice[:10]}"
    )


@pytest.mark.level0
def test_level_0_is_explicit_and_small(suite):
    tests, _ = suite
    assert _levels.DEFAULT_LEVEL == 1
    level0 = [test for test in tests if test.level == 0]
    assert all(test.source != "default" for test in level0)
    assert 0 < len(level0) <= _MAX_LEVEL0_CASES, (
        f"level 0 holds {len(level0)} cases, more than {_MAX_LEVEL0_CASES}: move tests "
        "to level 1 rather than raise the cap"
    )


# A small project with a level table of its own: one test for each place a
# level can come from.

_CONFTEST = """
from pathlib import Path
from tests import _levels

def pytest_addoption(parser):
    _levels.add_options(parser)

def pytest_configure(config):
    here = Path(__file__).parent
    _levels.register(config, here / "levels.toml", here)
"""
_INI = """
[pytest]
addopts = --strict-markers
markers =
    gpu
    level0
    level1
    level2
    level3
"""
_TABLE_TEXT = """
[level0.files]
"test_table.py" = "the file's tests are contract tests"
[level2.tests]
"test_table.py::test_swept" = "a sweep"
"test_table.py::Test*::test_in_a_class" = "a sweep in a class"
[level3.tests]
"test_table.py::test_swept[long-*]" = "the long cases"
"test_table.py::test_of_the_examples[*slow*]" = "a slow example, where the examples are"
[level3.files]
"test_detail.py" = "exact wording"
[level1.tests]
"test_detail.py::test_kept_at_level_1" = "found before merge"
"test_table.py::test_swept[long-2]" = "an entry without * counts before one with"
"""
_TEST_TABLE = """
import pytest

def test_plain():
    pass

@pytest.mark.parametrize("case", ["short-1", "long-1", "long-2"])
def test_swept(case):
    pass

class TestGroup:
    def test_in_a_class(self):
        pass

@pytest.mark.gpu
def test_on_a_gpu():
    pass

@pytest.mark.parametrize("example", [])
def test_of_the_examples(example):
    pass
"""
_TEST_MARKED = """
import pytest

pytestmark = pytest.mark.level2

def test_of_the_module():
    pass

@pytest.mark.level3
def test_marked():
    pass

@pytest.mark.level3
@pytest.mark.parametrize("case", ["a", pytest.param("b", marks=pytest.mark.level0)])
def test_with_a_marked_parameter(case):
    pass

@pytest.mark.level1
class TestMarked:
    def test_of_the_class(self):
        pass

    @pytest.mark.level0
    def test_marked_in_the_class(self):
        pass

@pytest.mark.level3
class TestDerived(TestMarked):
    pass
"""
_TEST_DETAIL = """
def test_wording():
    pass

def test_kept_at_level_1():
    pass
"""
_EXPECTED = {
    "test_table.py::test_plain": (0, "table"),
    "test_table.py::test_swept[short-1]": (2, "table"),
    "test_table.py::test_swept[long-1]": (3, "table"),
    "test_table.py::test_swept[long-2]": (1, "table"),
    "test_table.py::TestGroup::test_in_a_class": (2, "table"),
    "test_table.py::test_on_a_gpu": (0, "table"),
    "test_table.py::test_of_the_examples[NOTSET]": (0, "table"),
    "test_marked.py::test_of_the_module": (2, "module"),
    "test_marked.py::test_marked": (3, "test"),
    "test_marked.py::test_with_a_marked_parameter[a]": (3, "test"),
    "test_marked.py::test_with_a_marked_parameter[b]": (0, "parameter"),
    "test_marked.py::TestMarked::test_of_the_class": (1, "class"),
    "test_marked.py::TestMarked::test_marked_in_the_class": (0, "test"),
    # A subclass's own level counts before its base's.
    "test_marked.py::TestDerived::test_of_the_class": (3, "class"),
    "test_marked.py::TestDerived::test_marked_in_the_class": (0, "test"),
    "test_detail.py::test_wording": (3, "table"),
    "test_detail.py::test_kept_at_level_1": (1, "table"),
    "test_unmarked.py::test_default": (1, "default"),
}


@pytest.fixture
def project(tmp_path):
    files = {
        "conftest.py": _CONFTEST,
        "pytest.ini": _INI,
        "levels.toml": _TABLE_TEXT,
        "test_table.py": _TEST_TABLE,
        "test_marked.py": _TEST_MARKED,
        "test_detail.py": _TEST_DETAIL,
        "test_unmarked.py": "def test_default():\n    pass\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(textwrap.dedent(text).lstrip())
    return tmp_path


def _selected(project, *args):
    run = _pytest(project, "--collect-only", "-q", *args)
    assert run.returncode == 0, run.stdout + run.stderr
    return {line for line in run.stdout.splitlines() if "::" in line}


def _levels_of(project):
    out = project / "levels.json"
    run = _pytest(project, "--collect-only", "-q", f"--level-json={out}")
    assert run.returncode == 0, run.stdout + run.stderr
    return _levels.load(out.read_text())


@pytest.mark.slow
def test_the_closest_marker_then_the_table_then_level_1_give_a_test_its_level(project):
    tests, _ = _levels_of(project)
    assert {test.test: (test.level, test.source) for test in tests} == _EXPECTED


@pytest.mark.slow
def test_a_level_selects_itself_and_the_levels_below(project):
    assert _selected(project) == set(_EXPECTED)
    for level in _levels.LEVELS:
        expected = {test for test, (at, _) in _EXPECTED.items() if at <= level}
        assert _selected(project, "--level", str(level)) == expected
    assert _selected(project, "--level", "3") == set(_EXPECTED)
    # With another selection, both hold.
    assert _selected(project, "--level", "0", "-m", "gpu") == {"test_table.py::test_on_a_gpu"}
    assert _selected(project, "--level", "2", "-k", "swept") == {
        "test_table.py::test_swept[short-1]",
        "test_table.py::test_swept[long-2]",
    }
    assert _pytest(project, "--level", "4").returncode == 4


@pytest.mark.slow
def test_a_run_at_a_level_runs_only_its_tests(project):
    (project / "test_unmarked.py").write_text("def test_default():\n    assert False\n")
    run = _pytest(project, "-q", "--level", "0")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "5 passed, 1 skipped, 12 deselected" in run.stdout
    assert _pytest(project, "-q", "--level", "1").returncode == 1


@pytest.mark.slow
@pytest.mark.parametrize(
    ("file", "text", "said"),
    [
        (
            "test_two.py",
            "import pytest\n@pytest.mark.level0\n@pytest.mark.level2\ndef test_two():\n    pass\n",
            "test_two.py::test_two: its test is marked level0 and level2",
        ),
        (
            "test_two.py",
            "import pytest\npytestmark = [pytest.mark.level1, pytest.mark.level3]\n"
            "def test_two():\n    pass\n",
            "test_two.py::test_two: its module is marked level1 and level3",
        ),
        (
            "levels.toml",
            '[level0.files]\n"test_unmarked.py" = "a contract"\n'
            '[level2.files]\n"test_unmarked.py" = "a sweep"\n',
            "test_unmarked.py::test_default: the level table gives it level 0 "
            "('test_unmarked.py') and level 2 ('test_unmarked.py')",
        ),
    ],
    ids=["a-test", "a-module", "the-table"],
)
def test_two_levels_at_one_scope_are_a_usage_error_that_names_the_test(project, file, text, said):
    (project / file).write_text(text)
    run = _pytest(project, "-q")
    assert run.returncode == 4, run.stdout + run.stderr
    assert said in run.stderr


@pytest.mark.slow
@pytest.mark.parametrize(
    ("table", "said"),
    [
        ('[level1.files]\n"test_unmarked.py" = "the default"\n', "is at level 1 without an entry"),
        ('[level4.files]\n"test_unmarked.py" = "later"\n', "[level4] is not level0 .. level3"),
        ('[level2.files]\n"test_unmarked.py" = ""\n', "needs a one-line reason"),
        ('[level2.files]\n"test_unmarked.py::test_default" = "a test"\n', "does not belong"),
        ('[level2.cases]\n"test_unmarked.py" = "a sweep"\n', "is not files or tests"),
    ],
    ids=["a-level-1-file", "level-4", "no-reason", "a-test-among-files", "another-section"],
)
def test_a_table_that_cannot_be_read_is_a_usage_error(project, table, said):
    (project / "levels.toml").write_text(table)
    run = _pytest(project, "-q")
    assert run.returncode == 4, run.stdout + run.stderr
    assert said in run.stderr


@pytest.mark.slow
def test_the_guards_find_a_stale_entry_and_a_test_levelled_twice(project):
    table = project / "levels.toml"
    tests, skipped = _levels_of(project)
    # The entry for parameters of test_of_the_examples names none, and is
    # not stale: the test was collected without its parameters.
    assert _levels.stale_entries(_levels.load_table(table), tests, skipped) == []
    assert _levels.levelled_twice(tests) == []

    table.write_text(
        table.read_text()
        + '"test_table.py::test_renamed" = "a test that was renamed"\n'
        + '"test_marked.py::test_marked" = "also marked"\n'
        + '"test_skipped.py::test_never_collected" = "its file skips itself"\n'
    )
    (project / "test_skipped.py").write_text(
        "import pytest\npytest.skip('needs a tool', allow_module_level=True)\n"
    )
    tests, skipped = _levels_of(project)
    assert skipped == {"test_skipped.py"}
    assert _levels.stale_entries(_levels.load_table(table), tests, skipped) == [
        "test_table.py::test_renamed"
    ]
    assert _levels.levelled_twice(tests) == ["test_marked.py::test_marked"]
    # The marker still gives the level.
    assert {t.test: t.level for t in tests}["test_marked.py::test_marked"] == 3


@pytest.mark.slow
def test_the_report_counts_the_selected_cases_and_runs_nothing(project):
    run = _pytest(project, "--level-report", "-q")
    assert run.returncode == 0, run.stdout + run.stderr
    lines = [" ".join(line.split()) for line in run.stdout.splitlines()]
    assert "test levels of the 18 selected cases" in lines
    assert ["0 6 2", "1 4 4", "2 3 2", "3 5 3"] == lines[lines.index("level cases files") + 1 :][:4]
    assert "test_table.py 3 1 2 1 table" in lines
    assert "test_marked.py 3 1 1 3 marker" in lines
    assert "test_unmarked.py . 1 . . default" in lines
    assert "test_detail.py . 1 . 1 table" in lines
    assert "passed" not in run.stdout

    run = _pytest(project, "--level-report", "-q", "--level", "1", "-m", "not gpu")
    lines = [" ".join(line.split()) for line in run.stdout.splitlines()]
    assert "test levels of the 9 selected cases" in lines
    assert "test_table.py 2 1 . . table" in lines
