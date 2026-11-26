# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test levels: every test has one level, 0 to 3, and ``--level N`` runs the
tests of level N and lower.

A test's level is, in this order: a ``level0`` .. ``level3`` marker on its
parameter, on the test, on its class, on its module (``pytestmark``); its
entry in the level table (``tests/levels.toml``); level 1. Without
``--level`` nothing is deselected. ``CONTRIBUTING.md`` says what each level
means.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

LEVELS = (0, 1, 2, 3)
DEFAULT_LEVEL = 1
PLUGIN = "cudaq-pulse-test-levels"
_MARKERS = {f"level{level}": level for level in LEVELS}
_TESTS = Path(__file__).resolve().parent


def _pattern(text: str) -> re.Pattern[str]:
    """*text* as a pattern in which ``*`` stands for any characters."""
    return re.compile(".*".join(re.escape(part) for part in text.split("*")), re.DOTALL)


@dataclass
class Entry:
    """One line of the level table: the tests it names, their level and why."""

    key: str
    level: int
    reason: str
    path: str = field(init=False)
    #: How closely the entry names a test: 0 for a file, 1 and 2 for a test
    #: of it, 3 and 4 for parameters of a test; the even one of each pair is
    #: an entry written without ``*``.
    rank: int = field(init=False)

    def __post_init__(self):
        self.path, _, test = self.key.partition("::")
        name, bracket, param = test.partition("[")
        if bracket and not param.endswith("]"):
            raise ValueError(f"{self.key!r}: a parameter id ends with ]")
        self.rank = (3 if bracket else 1) + ("*" not in test) if test else 0
        self._name = _pattern(name) if test else None
        self._param = _pattern(param[:-1]) if bracket else None

    def names(self, name: str) -> bool:
        """Whether the entry names the test *name* of its file, or some of
        its parameters."""
        return self._name is None or bool(self._name.fullmatch(name))

    def matches(self, name: str, param: str | None) -> bool:
        """Whether the entry names the test *name* of its file, with the
        parameter id *param*."""
        if not self.names(name):
            return False
        return self._param is None or (param is not None and bool(self._param.fullmatch(param)))


def load_table(path: Path) -> list[Entry]:
    """The entries of the level table at *path*.

    The table has a ``[levelN.files]`` and a ``[levelN.tests]`` section for
    each level (level 1, the default, has tests only); a key is a test file,
    or ``file::test`` or ``file::test[parameter id]`` with ``*`` for any
    characters (a ``*`` itself cannot be asked for), and its value is the
    reason, in one line.
    """
    entries = []
    for section, content in tomllib.loads(path.read_text()).items():
        if section not in _MARKERS:
            raise ValueError(f"{path.name}: [{section}] is not level0 .. level{LEVELS[-1]}")
        if unknown := set(content) - {"files", "tests"}:
            raise ValueError(f"{path.name}: [{section}.{min(unknown)}] is not files or tests")
        if _MARKERS[section] == DEFAULT_LEVEL and "files" in content:
            raise ValueError(
                f"{path.name}: [{section}.files]: a file is at level {DEFAULT_LEVEL} "
                "without an entry"
            )
        for kind, rows in content.items():
            for key, reason in rows.items():
                if ("::" in key) != (kind == "tests"):
                    raise ValueError(f"{path.name}: {key!r} does not belong in [{section}.{kind}]")
                if not isinstance(reason, str) or not reason.strip() or "\n" in reason:
                    raise ValueError(f"{path.name}: {key!r} needs a one-line reason")
                entries.append(Entry(key, _MARKERS[section], reason))
    return entries


@dataclass(frozen=True)
class Resolved:
    """A collected test, its level, and where the level comes from: a
    marker's scope, ``table`` or ``default``."""

    path: str
    name: str
    param: str | None
    level: int
    source: str
    #: The table entries that name the test, whether or not they set its level.
    entries: tuple[str, ...] = ()

    @property
    def test(self) -> str:
        return f"{self.path}::{self.name}" + ("" if self.param is None else f"[{self.param}]")


class LevelError(Exception):
    """A test with two levels at one scope."""


def _own_marks(cls: type) -> list[pytest.Mark]:
    """The markers written on *cls* itself, not those of its bases."""
    marks = cls.__dict__.get("pytestmark", [])
    marks = marks if isinstance(marks, (list, tuple)) else [marks]
    return [getattr(mark, "mark", mark) for mark in marks]


def _marker_scopes(item: pytest.Item) -> Iterator[tuple[str, list[pytest.Mark]]]:
    """The markers of *item*, scope by scope, the closest first."""
    callspec = getattr(item, "callspec", None)
    on_parameter = list(callspec.marks) if callspec is not None else []
    yield "parameter", on_parameter
    yield "test", [m for m in item.own_markers if not any(m is p for p in on_parameter)]
    for node in reversed(item.listchain()[:-1]):
        if isinstance(node, pytest.Class):
            # pytest gives a class the markers of its bases too: a subclass's
            # own level counts before the level it inherits.
            for cls in node.obj.__mro__:
                yield "class", _own_marks(cls)
        yield type(node).__name__.lower(), node.own_markers


class LevelPlugin:
    """The pytest plugin: resolves each collected test's level, deselects
    above ``--level``, and prints ``--level-report``."""

    def __init__(self, config: pytest.Config, table: Path, root: Path):
        self.config = config
        self.table = table
        self.root = root
        self.entries = load_table(table)
        self._by_path: dict[str, list[Entry]] = {}
        for entry in self.entries:
            self._by_path.setdefault(entry.path, []).append(entry)
        #: Every collected test, before any deselection.
        self.collected: list[Resolved] = []
        #: Test files that skipped themselves when they were imported.
        self.skipped_files: set[str] = set()
        self._level_of: dict[int, Resolved] = {}
        self._relative_paths: dict[Path, str] = {}

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def resolve(self, item: pytest.Item) -> Resolved:
        path = self._relative_paths.get(item.path)
        if path is None:
            path = self._relative_paths[item.path] = self._relative(item.path)
        chain = [node.name for node in item.listchain() if isinstance(node, pytest.Class)]
        name = "::".join([*chain, getattr(item, "originalname", None) or item.name])
        callspec = getattr(item, "callspec", None)
        param = callspec.id if callspec is not None else None
        test = f"{path}::{name}" + ("" if param is None else f"[{param}]")

        named = [e for e in self._by_path.get(path, ()) if e.matches(name, param)]
        closest = [e for e in named if e.rank == max(n.rank for n in named)]
        if len({e.level for e in closest}) > 1:
            raise LevelError(
                f"{test}: the level table gives it "
                + " and ".join(f"level {e.level} ({e.key!r})" for e in closest)
            )
        keys = tuple(e.key for e in named)

        for scope, markers in _marker_scopes(item):
            found = sorted({_MARKERS[m.name] for m in markers if m.name in _MARKERS})
            if len(found) > 1:
                raise LevelError(
                    f"{test}: its {scope} is marked "
                    + " and ".join(f"level{level}" for level in found)
                )
            if found:
                return Resolved(path, name, param, found[0], scope, keys)
        if closest:
            return Resolved(path, name, param, closest[0].level, "table", keys)
        return Resolved(path, name, param, DEFAULT_LEVEL, "default", keys)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped and report.nodeid.endswith(".py"):
            self.skipped_files.add(self._relative(self.config.rootpath / report.nodeid))

    # Before ``-m`` and ``-k`` deselect, so the guards see every collected
    # test, and before the GPU check of tests/conftest.py, so a GPU test
    # above the level asks for no GPU.
    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]):
        resolved, errors = [], []
        for item in items:
            try:
                resolved.append(self.resolve(item))
            except LevelError as error:
                errors.append(str(error))
        if errors:
            shown = "\n  ".join(errors[:20])
            more = f"\n  and {len(errors) - 20} more" if len(errors) > 20 else ""
            raise pytest.UsageError(f"tests with two levels:\n  {shown}{more}")
        self.collected = resolved
        if path := config.getoption("--level-json"):
            Path(path).write_text(dump(resolved, self.skipped_files))
        self._level_of = {id(item): test for item, test in zip(items, resolved, strict=True)}
        limit = config.getoption("--level")
        if limit is None:
            return
        kept, dropped = [], []
        for item, test in zip(items, resolved, strict=True):
            (kept if test.level <= limit else dropped).append(item)
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = kept

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        # Nothing to report when the collection stopped on a usage error.
        if self.config.getoption("--level-report") and self._level_of:
            writer = self.config.get_terminal_writer()
            # An item a later hook added is resolved here.
            selected = [
                self._level_of.get(id(item)) or self.resolve(item) for item in session.items
            ]
            for line in report_lines(selected):
                writer.line(line)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session: pytest.Session):
        # The report is of what a run would select; it runs nothing.
        if self.config.getoption("--level-report") and not session.testsfailed:
            return True
        return None

    def whole_suite(self) -> bool:
        """Whether this session collected every test file under ``tests``."""
        option = self.config.getoption
        if option("ignore") or option("ignore_glob") or option("lf", False):
            return False
        return all("::" not in arg and self._tests_are_under(Path(arg)) for arg in self.config.args)

    def _tests_are_under(self, arg: Path) -> bool:
        path = (self.config.invocation_params.dir / arg).resolve()
        tests = self.root / "tests"
        return path == tests or path in tests.parents


def dump(tests: list[Resolved], skipped_files: set[str]) -> str:
    """*tests* and the files that skipped themselves, as JSON (``load``)."""
    rows = [[t.path, t.name, t.param, t.level, t.source, list(t.entries)] for t in tests]
    return json.dumps({"tests": rows, "skipped_files": sorted(skipped_files)})


def load(text: str) -> tuple[list[Resolved], set[str]]:
    """The tests and skipped files that ``dump`` wrote."""
    data = json.loads(text)
    tests = [Resolved(*row[:5], tuple(row[5])) for row in data["tests"]]
    return tests, set(data["skipped_files"])


def stale_entries(
    entries: list[Entry], tests: list[Resolved], skipped_files: set[str]
) -> list[str]:
    """The table entries that name none of *tests*.

    Two kinds of entry cannot be told and are not listed: one of a file that
    skipped itself, and one for parameters of a test that was collected
    without any (pytest's ``NOTSET`` case), as a test over the examples is
    where the examples are not installed.
    """
    named = {key for test in tests for key in test.entries}
    unset = {
        (t.path, t.name) for t in tests if t.param is not None and t.param.startswith("NOTSET")
    }
    return [
        e.key
        for e in entries
        if e.key not in named
        and e.path not in skipped_files
        and not (e.rank >= 3 and any(path == e.path and e.names(name) for path, name in unset))
    ]


def levelled_twice(tests: list[Resolved]) -> list[str]:
    """The tests that have a level marker and a table entry."""
    return [t.test for t in tests if t.entries and t.source not in ("table", "default")]


def report_lines(tests: list[Resolved]) -> list[str]:
    """The level report: cases and files per level, then each file's cases
    per level and where its levels come from."""
    per_file: dict[str, Counter[int]] = {}
    sources: dict[str, set[str]] = {}
    for test in tests:
        per_file.setdefault(test.path, Counter())[test.level] += 1
        sources.setdefault(test.path, set()).add(
            test.source if test.source in ("table", "default") else "marker"
        )
    lines = [f"test levels of the {len(tests)} selected cases", "", "level   cases   files"]
    for level in LEVELS:
        cases = sum(counts[level] for counts in per_file.values())
        files = sum(1 for counts in per_file.values() if counts[level])
        lines.append(f"{level:>5} {cases:>7} {files:>7}")
    width = max((len(path) for path in per_file), default=4)
    lines += ["", f"{'file':<{width}} " + " ".join(f"{f'L{n}':>6}" for n in LEVELS) + "  from"]
    for path in sorted(per_file):
        counts = per_file[path]
        cells = " ".join(f"{counts[n] or '.':>6}" for n in LEVELS)
        origin = ", ".join(s for s in ("marker", "table", "default") if s in sources[path])
        lines.append(f"{path:<{width}} {cells}  {origin}")
    return lines


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("levels", "test levels")
    group.addoption(
        "--level",
        type=int,
        choices=LEVELS,
        default=None,
        metavar="N",
        help="run the tests of level N and lower (0 contract, 1 behaviour, 2 exhaustive, "
        "3 fine detail and long cases); without it every level runs",
    )
    group.addoption(
        "--level-report",
        action="store_true",
        help="print the selected cases per level and per file, and run nothing",
    )
    group.addoption(
        "--level-json",
        metavar="PATH",
        default=None,
        help="write the level of every collected case, and what set it, to PATH as JSON, "
        "before -m, -k or --level deselect any",
    )


def register(config: pytest.Config, table: Path | None = None, root: Path | None = None) -> None:
    """Register the plugin, with the level table of this repository unless
    *table* (and the directory *root* its paths start from) name another."""
    table = table or _TESTS / "levels.toml"
    try:
        plugin = LevelPlugin(config, table, (root or _TESTS.parent).resolve())
    except (OSError, ValueError) as error:
        raise pytest.UsageError(f"the level table cannot be read: {error}") from error
    config.pluginmanager.register(plugin, PLUGIN)
