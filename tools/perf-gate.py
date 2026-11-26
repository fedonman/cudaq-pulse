#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic performance gate: instruction counts and program sizes.

Runs the cases of ``benchmarks/perf_gate/cases.py`` and compares them with
``benchmarks/baselines/perf_gate.json``:

- program sizes (op count, packed program bytes, export bytes), measured
  natively; any difference fails.
- instructions executed per repetition (Callgrind's ``Ir``), measured under
  Valgrind on Linux x86-64; more than ``REGRESSION`` above the baseline
  fails (a case of ``cases.py`` may declare its own ``tolerance``, which
  the baseline records), and more than ``IMPROVEMENT`` below it fails too,
  until the baseline is updated, so that the gain is kept. Callgrind's
  cache simulation, with a fixed cache geometry, also gives an estimate of
  cycles (``L1 hits + 5 x LL hits + 35 x memory accesses``); it is
  reported, not gated.
- Python-level work per repetition (bytecodes executed, Python calls, C
  calls), counted with ``sys.monitoring`` in two native runs after the
  Callgrind runs, so it does not touch their counts. It does not gate: it
  says what kind of failure a case's instructions are (``failure_kind``).
  With the same work as the baseline's (or, for a slower row compared
  with its base, as the base's) and a change within ``LAYOUT_SHIFT``
  either way, it is ``layout only``: the count moved
  below Python, as when one new type at import renumbers the interpreter's
  type version tags and so its method cache's collisions; the case still
  fails, to be judged before re-baselining, unless its own change moved
  it by layout (below). Otherwise it is a
  ``regression`` or a ``gain`` (to keep by updating the baseline), with
  each counter's change if the work changed, or a note that the change is
  in native code if it did not. A baseline without the work (format 1)
  labels only regressions and gains.

A row is *layout only* when, against its base built in this job, it ran
the same Python work (bytecodes, Python calls and C calls per repetition)
and called every named function as many times as the base did, apart from
the functions whose call counts follow the heap's shape rather than the
code (``LAYOUT_CLASSES``): libc's allocator and its internal steps
(``malloc``, ``free``, ``_int_malloc``, ``malloc_consolidate``,
``unlink_chunk`` and the like), C++'s ``operator new`` and ``delete`` and
LLVM's buffer allocation; libc's memory copies and compares (``memcpy``,
``memmove``, ``memset``, ``memcmp``); the equality and comparison
callbacks of hash tables and ordered containers; the interpreter's own
functions, whose type version tags, method cache and dict probes follow
addresses (what Python runs is compared on its own, as its work); and code
Callgrind cannot attribute to a mapped object. Its base is the commit the
change is applied to (``--base-commit``): for a pull request, the first
parent of its merge commit; for a push, the previous tip of its branch;
for a run dispatched on a branch, the merge base of the branch and main.
A row that fails against the stored baseline and is layout only against
its base is judged against that base instead: it fails when its count is
more than its tolerance (``REGRESSION``, 0.4%, unless the case declares
its own) above the base's, and its failure names the layout classes that
carried the shift and the functions whose cost per call moved most, to
bring to the owner, whose to accept it is; a lower count passes, is
reported, and is not a gain to keep. A row within the baseline's limits
is not judged against its base. A row with any other difference (in
Python work, or in the calls of a function outside those classes) is
judged against the stored baseline, as before. A run with no base (a
change to the gate, its cases, the job or the toolchain pin,
``NATIVE_BASE_REFUSED``; a base that fails to build or measures other
Python work; a baseline run) judges every row against the stored baseline
and says why there was no base. Equal
call counts are evidence of layout, not proof: work added inside a
function, with no new calls, keeps every count; the 0.4% limit against
the base is what catches that. How the base is measured: ``--help``,
"base comparison" and "native base comparison".

The instruction counts repeat exactly only in the environment they were
measured in, so a baseline is versioned by it: the baseline holds the
environment CI measured in (``environment``: the MLIR toolchain, Python,
compiler, libc and Valgrind) and the run it was taken from (``source``).
Counts measured in another environment fail the gate even when they are
within the limits, so a change of pin is followed by a new baseline, taken
on its pull request's branch (``--help``, "taking a baseline"); ``git log
benchmarks/baselines/perf_gate.json`` lists the earlier ones, and
``--baseline`` compares with any of them.

A baseline with ``"bootstrap": true`` and no instruction counts is in
bootstrap mode: the counts are measured and reported but do not fail the
run (sizes still do). Its measurements, taken as the baseline, drop the
field; the field next to any count is an error, so the gate cannot stay in
bootstrap mode once it has counts.

In every mode, the gate fails if the counts cannot be right: a case without
one, a count of zero or one too small for its case, or a heavier case not
measuring well above a lighter one (``implausible``). Such counts are never
written as a baseline; ``--dumps DIR`` keeps the dumps for inspection.

Usage:
    python tools/perf-gate.py                 # compare with the baseline
    python tools/perf-gate.py --update        # rewrite the baseline from this run
    python tools/perf-gate.py --processes 5   # also show the spread of 5 runs
    python tools/perf-gate.py --base-commit SHA   # in CI: compare with the base

``--out FILE`` writes this run's measurements in the baseline's format;
CI uploads it, so a baseline can be taken from a CI run (CONTRIBUTING.md).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import fnmatch
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmarks" / "perf_gate" / "cases.py"
MARKS = ROOT / "benchmarks" / "perf_gate" / "marks.c"
BASELINE = ROOT / "benchmarks" / "baselines" / "perf_gate.json"
# Format 2 adds each case's python_work; a format 1 baseline reads as one
# without it.
FORMAT = 2
READABLE_FORMATS = (1, 2)
# A case's keys in the baseline that are not program sizes.
NOT_SIZES = ("instructions", "estimated_cycles", "python_work", "tolerance", "address_keyed")

LAYOUT_ONLY = (
    "instructions moved with identical Python work, likely layout only (for example "
    "type version tags or allocation layout); judge before re-baselining"
)
# The largest change, either way, that identical Python work labels as
# layout only: the shifts seen were up to 0.5% (one new metaclass at import
# moved specialize_amplitude by +0.44%). Beyond it the change is in native
# code, as when a compile's passes do less (-19% for compile_surface_d5).
LAYOUT_SHIFT = 0.02
NATIVE_ONLY = (
    "instructions moved with identical Python work, by more than layout shifts do: "
    "the change is below Python, in native code"
)

# Fractions of the baseline's instructions per repetition.
REGRESSION = 0.004
IMPROVEMENT = 0.01

# The paths a change may touch and keep its base's measured code: the
# package's Python sources, and files the perf-gate job never reads. Any
# other path (native code, build files, pins, the job in ci.yml, the gate
# and its cases, data files) is measured code. Fails closed.
PYTHON_OR_TEXT = (
    "core/frontend/*.py",
    "core/frontend/*.pyi",
    "tests/*",
    "docs/*",
    "examples/*",
    "*.md",
    "LICENSE",
    "LICENSES/*",
    "NOTICE",
    "THIRD_PARTY_NOTICES",
    "REUSE.toml",
    "CITATION.cff",
    ".github/ISSUE_TEMPLATE/*",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/CODEOWNERS",
    "test/*",
    ".github/workflows/*",
    ".pre-commit-config.yaml",
    "pyrightconfig.json",
    ".clang-format",
    "tools/check-*.py",
    "tools/ci-changes.py",
    "tools/stubtest-allowlist.txt",
    "tools/tutorial-scripts.py",
    "benchmarks/baselines/*",
    "challenges/*",
)
# Read by the perf-gate job, but only through the parts build_inputs() takes.
BUILD_FILES = ("pyproject.toml", "uv.lock")
# The package's sources, which the build stages as links to them.
FRONTEND = "core/frontend"
AS_BASE = (
    "the change keeps its base's measured code; on its base's Python sources, measured on "
    "this build, the case ran the same Python work and the same calls outside the layout "
    "classes, so it is judged against its base: reported, not failed, and not a gain to keep"
)
AS_KEYED = (
    "its count less the change in cost per call of the address-keyed functions (ADDRESS_KEYED: "
    "MLIR lookups whose cost per call follows where a key lands in a hash table), judged against "
    "the baseline's records of them or else its base, is no more than its tolerance above the "
    "stored baseline, and what was taken out is at most 2% of the baseline's count, so it is "
    "reported, not failed, and not a gain to keep"
)
AS_NATIVE_BASE = (
    "the change touches native code; on its base, built in this job, the case ran the same "
    "Python work and the same calls of every named function outside the layout classes, so "
    "it is judged against its base: reported, not failed, and not a gain to keep"
)
# What a change may touch and still be judged against a base this job builds
# (native code and its build files may differ): not the gate, its cases, the
# job or the toolchain pin, which would measure the base differently.
NATIVE_BASE_REFUSED = (
    "tools/perf-gate.py",
    "benchmarks/perf_gate/*",
    ".github/workflows/ci.yml",
    "tools/mlir-toolchain.lock.json",
    "tools/fetch-mlir-toolchain.py",
    "pyproject.toml",
    "uv.lock",
)
# How the job builds the tree, and so the base: the package only, with the
# extension's symbols kept so that its functions have names in the dumps (its
# code is the same; strip only drops the names).
GATE_CMAKE_ARGS = (
    "-G",
    "Ninja",
    "-DCUDAQ_PULSE_BUILD_TESTS=OFF",
    "-DCUDAQ_PULSE_KEEP_SYMBOLS=ON",
)
# A native function whose number of calls may change with layout alone, by
# kind, as (class, object pattern, function pattern); either pattern may be
# None. Builds of identical native code, on the box and on CI's runner, ran
# different numbers of calls only in these kinds of functions (the evidence:
# the call counts of 20 cases on three pairs, and a CI run of a comment-only
# change).
LAYOUT_CLASSES = (
    # A hash table calls its equality callback once per probe, and an ordered
    # container its comparison once per step down its tree: both follow the
    # addresses of the keys, hashed or compared. MLIR's uniquer (types and
    # attributes; FloatAttr's callback compares floats), any DenseMapInfo's
    # isEqual, and the comparisons of std::map and std::set.
    (
        "container probes",
        None,
        r"function_ref<bool \(mlir::StorageUniquer::BaseStorage const\*\)>::callback_fn<"
        r"|^llvm::detail::IEEEFloat::bitwiseIsEqual\("
        r"|::isEqual\("
        r"|^std::common_comparison_category<.*std::operator<=>",
    ),
    # The interpreter's dict and set probes, type version tags and method
    # caches follow addresses and hashes too; what Python runs (bytecodes,
    # Python and C calls) is compared on its own.
    ("interpreter", r"^(python3(\.\d+)?|libpython3[.\d]*\.so.*)$", None),
    # When memory is taken and given back follows when a container grows,
    # which follows its probes: libc's allocator (and its own steps), C++'s
    # new and delete, and LLVM's buffers under its containers.
    (
        "allocation",
        r"^libc\.so",
        r"^(malloc|free|cfree|calloc|realloc|posix_memalign|aligned_alloc|memalign|valloc"
        r"|brk|sbrk|__default_morecore|sysmalloc|_int_\w+|\w*_chunk\w*|_mid_memalign\w*"
        r"|malloc_\w+|alloc_\w+|tcache_\w+|arena_\w+)(\.\w+)*$",
    ),
    (
        "allocation",
        None,
        r"^operator (new|delete)(\[\])?\(|^llvm::(allocate|deallocate)_buffer\(",
    ),
    # A container that grows copies its blocks, and a probe may compare keys
    # byte by byte: how often follows the same growth and probes. libc's own
    # memcpy, memmove, memset and memcmp (by their CPU-specific names).
    ("memory copies", r"^libc\.so", r"^_*(memcpy|memmove|memset|memcmp|bcmp)\w*$"),
    # A libc that does not name its functions (the box's) leaves them as
    # addresses: there they were its allocator's steps and its memory copies,
    # by their call counts against a libc that names them. Named so, as it
    # cannot tell which.
    ("libc, unnamed (allocator or memory copies)", r"^libc\.so", r"^0x[0-9a-f]+$"),
    # Code Callgrind finds in no mapped object ("???"): on the box the code
    # libc enters the kernel through, on CI's runner also the interpreter's
    # own functions. Named so, as it cannot tell which.
    ("unmapped (interpreter or kernel)", r"^\?\?\?$", None),
)

# Functions of the pinned MLIR toolchain whose own instructions per call
# follow where a key lands in a hash table: entered as often, they do the
# same work. A row that fails as a regression against the stored baseline,
# and whose base the job measured, is judged on its count less the change of
# those it entered exactly as often as its base, against the stored baseline
# (apply_address_keyed). Every entry is an anchored
# pattern for one function of MLIR or LLVM, never of this repository or the
# interpreter, with what it keys on and the CI runs that showed it entered
# as often on both sides with its cost per call moved. An entry is added in
# a pull request of its own that changes only this list and its test, and
# the owner approves it.
ADDRESS_KEYED = (
    # Keys on: the storage's TypeID (a DenseMap of TypeID to its parametric
    # uniquer), then the instance's hash in that uniquer's shard.
    # Evidence: CI runs 45638819498 (main) and 45685405489 (a branch): first_compile 984
    # entries each way, 544.6 -> 590.5 instructions per call; specialize_amplitude 740 each
    # way, 190.7 -> 272.0.
    r"^mlir::StorageUniquer::getParametricStorageTypeImpl\(",
    # Keys on: the attribute's TypeID in the context's map of abstract attributes.
    # Evidence: CI runs 45638819498 and 45685405489: first_compile 688 entries each way,
    # 35.4 -> 124.9 instructions per call.
    r"^mlir::detail::AttributeUniquer::initializeAttributeStorage\(",
    # Keys on: the operation's TypeID and name in the context's maps of registered
    # operations.
    # Evidence: CI runs 45638819498 and 45685405489: first_compile 443 entries each way,
    # 911.2 -> 831.0 instructions per call (lower).
    r"^mlir::RegisteredOperationName::insert\(",
)

# Callgrind's cache model, fixed so the estimate does not depend on the
# runner's CPU: 32 KiB 8-way L1 instruction and data caches, 8 MiB 16-way
# last level, 64-byte lines.
CACHES = ("--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64")
CACHE_EVENTS = ("Ir", "Dr", "Dw", "I1mr", "D1mr", "D1mw", "ILmr", "DLmr", "DLmw")


@dataclass
class Measured:
    instructions: list[float] = field(default_factory=list)
    estimated_cycles: list[float] = field(default_factory=list)


def _environment(python: str) -> dict[str, str]:
    """What the instruction counts depend on besides the code."""
    probe = (
        "import json, platform; print(json.dumps({'python': platform.python_version(), "
        "'machine': platform.machine(), 'libc': ' '.join(platform.libc_ver())}))"
    )
    env = json.loads(subprocess.check_output([python, "-c", probe], text=True))
    compiler = shutil.which("c++") or shutil.which("cc")
    if compiler:
        out = subprocess.run([compiler, "--version"], capture_output=True, text=True)
        env["compiler"] = out.stdout.splitlines()[0] if out.stdout else ""
    if shutil.which("valgrind"):
        env["valgrind"] = subprocess.check_output(["valgrind", "--version"], text=True).strip()
    lock = json.loads((ROOT / "tools" / "mlir-toolchain.lock.json").read_text())
    env["mlir_toolchain"] = lock["release_tag"]
    return env


def _run_env(pythonpath: str, home: str) -> dict[str, str]:
    """A fixed environment: the process sees the same variables, and so the
    same stack layout, in every run."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": pythonpath,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        # One MLIR context for the whole run, made by the first compile: a
        # new one is otherwise made every 256 modules, inside whichever
        # case reaches that number.
        "CUDAQ_PULSE_CONTEXT_MAX_MODULES": str(1 << 40),
        "CUDAQ_PULSE_CONTEXT_MAX_WORDS": str(1 << 40),
        # NumPy's OpenBLAS otherwise starts a worker per CPU, whose waiting
        # for work Callgrind counts with the case that runs meanwhile.
        "OPENBLAS_NUM_THREADS": "1",
    }


def _structural(
    python: str, env: dict[str, str]
) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    """Each case's program sizes, and the tolerances the cases declare."""
    out = subprocess.run(
        [python, str(CASES), "--structural"], env=env, capture_output=True, text=True, cwd=ROOT
    )
    if out.returncode:
        sys.exit(f"perf gate: {CASES.name} --structural failed:\n{out.stderr}")
    result = json.loads(out.stdout)
    return result["sizes"], result["tolerances"]


def _byte_compile(python: str, env: dict[str, str], pythonpath: str) -> None:
    """Byte-compiles the package, the benchmarks and the installed packages
    first, so no run compiles a module inside a case."""
    purelib = subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()
    for directory in (pythonpath, str(ROOT / "benchmarks"), purelib):
        subprocess.run(
            [python, "-m", "compileall", "-q", directory],
            env={**env, "PYTHONDONTWRITEBYTECODE": ""},
            check=True,
            stdout=subprocess.DEVNULL,
        )


def _build_marks(directory: Path) -> Path:
    library = directory / "libperf_gate_marks.so"
    compiler = os.environ.get("CC") or shutil.which("cc") or "gcc"
    subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", "-o", str(library), str(MARKS)], check=True
    )
    return library


def _no_aslr() -> list[str]:
    """``setarch -R`` where the kernel lets this process turn off address
    randomization (a container's seccomp profile may not)."""
    setarch = shutil.which("setarch")
    if setarch is None:
        return []
    command = [setarch, platform.machine(), "-R"]
    probe = subprocess.run([*command, "true"], capture_output=True)
    return command if probe.returncode == 0 else []


class GateError(Exception):
    """Measurements that cannot be trusted: the gate fails in every mode."""


@dataclass
class Dump:
    name: str
    events: dict[str, int]
    summary: dict[str, int] | None


# A cost line starts with its positions (a number, "0x" address, or a
# "+n", "-n" or "*" relative to the last one) or, without positions, its
# first event.
_COST_LINE = re.compile(r"[0-9+*-]")


def parse_dump(text: str) -> Dump | None:
    """One Callgrind dump written by a ``perf_gate_stop`` client request, or
    None for any other dump.

    The events are the ``totals:`` line, which Callgrind writes as the sum
    of the dump's self-cost lines: the lines after ``calls=`` (a call's
    inclusive cost) and ``jump=``/``jcnd=`` are not self cost. The sum is
    checked against it. The header's ``summary:`` is kept for comparison."""
    trigger = re.search(r"^desc: Trigger: Client Request: (.+)$", text, re.MULTILINE)
    if trigger is None:
        return None
    name = trigger.group(1).strip()
    names: list[str] | None = None
    positions = 1  # "positions: line" when the header does not say
    summary = totals = None
    self_cost: list[int] = []
    not_self = False

    def costs(fields: list[str]) -> dict[str, int]:
        if names is None or len(fields) > len(names):
            raise GateError(f"dump for {name}: costs {fields} do not match its events: line")
        values = [int(v) for v in fields]
        return dict(zip(names, values + [0] * (len(names) - len(values)), strict=True))

    for line in text.splitlines():
        if line.startswith("positions:"):
            positions = len(line.partition(":")[2].split())
        elif line.startswith("events:"):
            names = line.partition(":")[2].split()
            self_cost = [0] * len(names)
        elif line.startswith("summary:"):
            summary = costs(line.partition(":")[2].split())
        elif line.startswith("totals:"):
            totals = costs(line.partition(":")[2].split())
        elif line.startswith(("calls=", "jump=", "jcnd=")):
            not_self = True
        elif _COST_LINE.match(line):
            if not_self:
                not_self = False
                continue
            for i, value in enumerate(costs(line.split()[positions:]).values()):
                self_cost[i] += value
    if names is None or totals is None:
        raise GateError(f"dump for {name} has no events: or totals: line (truncated?)")
    if list(totals.values()) != self_cost:
        raise GateError(
            f"dump for {name}: totals {list(totals.values())} differ from the sum of its "
            f"cost lines {self_cost}"
        )
    return Dump(name, totals, summary)


def estimated_cycles(events: dict[str, int]) -> float | None:
    """``L1 hits + 5 x LL hits + 35 x memory accesses``, from a dump with
    cache simulation."""
    if not all(name in events for name in CACHE_EVENTS):
        return None
    accesses = events["Ir"] + events["Dr"] + events["Dw"]
    l1_misses = events["I1mr"] + events["D1mr"] + events["D1mw"]
    ll_misses = events["ILmr"] + events["DLmr"] + events["DLmw"]
    return (accesses - l1_misses) + 5 * (l1_misses - ll_misses) + 35 * ll_misses


Difference = tuple[str, dict[str, int]]


def _callgrind(
    python: str, env: dict[str, str], marks: Path, directory: Path, prefix: list[str]
) -> tuple[dict[str, dict[str, float]], list[Difference]]:
    """One Callgrind run of every case: its events per repetition, and how
    far each dump's summary is above its totals. The dumps and Valgrind's
    log stay in *directory*."""
    directory.mkdir(parents=True)
    command = [
        *prefix,
        "valgrind",
        "--tool=callgrind",
        "--instr-atstart=no",
        "--cache-sim=yes",
        *CACHES,
        f"--callgrind-out-file={directory}/callgrind.out",
        python,
        str(CASES),
        "--measure",
        str(marks),
    ]
    out = subprocess.run(command, env=env, capture_output=True, text=True, cwd=ROOT)
    (directory / "valgrind.log").write_text(out.stderr)
    if out.returncode:
        raise GateError(f"Callgrind run failed ({directory.name}):\n{out.stderr[-4000:]}")
    reps = json.loads(out.stdout.strip().splitlines()[-1])
    results: dict[str, dict[str, float]] = {}
    differences: list[Difference] = []
    for path in sorted(directory.glob("callgrind.out*")):
        dump = parse_dump(path.read_text(errors="replace"))
        if dump is None:
            continue
        if dump.name not in reps or dump.name in results:
            raise GateError(f"unexpected Callgrind dump {path.name} for {dump.name}")
        n = reps[dump.name]
        cycles = estimated_cycles(dump.events)
        results[dump.name] = {
            "instructions": dump.events.get("Ir", 0) / n,
            "estimated_cycles": float("nan") if cycles is None else cycles / n,
        }
        if dump.summary is not None:
            above = {k: dump.summary.get(k, 0) - v for k, v in dump.events.items()}
            differences.append((f"{dump.name} ({directory.name}/{path.name})", above))
    missing = set(reps) - set(results)
    if missing:
        raise GateError(f"no Callgrind dump for {', '.join(sorted(missing))} ({directory.name})")
    return results, differences


# The most instructions the marks' own blocks can put between a dump's
# summary and its totals.
MARKS_INSTRUCTIONS = 64


def dump_notes(differences: list[Difference]) -> tuple[list[str], str]:
    """Warnings for the dumps whose summary: differs from their totals: in
    a way the marks do not explain, and a note when every dump differs by
    the same few instructions, which the marks do.

    The summary is the running total since the zeroing request, counted
    instruction by instruction under --cache-sim=yes. The cost lines count
    a block's instructions when the next block starts: they leave out the
    marks' block that ends in the dump request, and take in the one that
    ends in the zeroing request, which the summary leaves out. So the two
    differ by the same number of instructions in every dump, fixed by the
    machine code of marks.c, which cancels out of every comparison."""
    if not differences:
        return [], ""

    def moved(above: dict[str, int]) -> str:
        return ", ".join(f"{k} {v:+d}" for k, v in above.items() if v)

    counted = collections.Counter(tuple(sorted(above.items())) for _, above in differences)
    shared = dict(counted.most_common(1)[0][0])
    if not all(abs(v) <= MARKS_INSTRUCTIONS for v in shared.values()):
        shared = {}
    odd = [(where, above) for where, above in differences if not shared or above != shared]
    warnings = [
        f"{where}: the dump's summary: differs from its totals: by {moved(above)}"
        + (f", not {moved(shared)} like the other dumps" if any(shared.values()) else "")
        for where, above in odd
        if any(above.values())
    ]
    note = ""
    if any(shared.values()):
        note = (
            f"{'every other' if odd else 'every'} dump's summary: is {moved(shared)} from its "
            "totals: (the marks' own blocks); the totals are used"
        )
    return warnings, note


def measure_instructions(
    python: str, env: dict[str, str], processes: int, workdir: Path, dumps: Path
) -> tuple[dict[str, Measured], list[Difference], bool]:
    """Each case's instructions per repetition in *processes* identical
    runs, made side by side (their dumps in *dumps*), how far each dump's
    summary is above its totals, and whether address randomization was
    off."""
    marks = _build_marks(workdir)
    prefix = _no_aslr()
    with concurrent.futures.ThreadPoolExecutor(processes) as pool:
        runs = list(
            pool.map(
                lambda k: _callgrind(python, env, marks, dumps / f"run{k}", prefix),
                range(processes),
            )
        )
    measured: dict[str, Measured] = {}
    differences: list[Difference] = []
    for run, run_differences in runs:
        differences += run_differences
        for name, values in run.items():
            m = measured.setdefault(name, Measured())
            m.instructions.append(values["instructions"])
            m.estimated_cycles.append(values["estimated_cycles"])
    return measured, differences, bool(prefix)


PythonWork = dict[str, dict[str, float]]
# Each case's count measured on the base, and its Python work there.
Base = dict[str, tuple[float, dict[str, float] | None]]


def measure_python_work(python: str, env: dict[str, str]) -> tuple[PythonWork | None, str]:
    """Each case's Python-level work per repetition (bytecodes, Python
    calls, C calls), from two native runs of ``cases.py --python-work``
    made side by side, or None and why it is not used: the counts take
    Python 3.12's sys.monitoring, and two runs must agree."""

    def run(_: int) -> PythonWork:
        out = subprocess.run(
            [python, str(CASES), "--python-work"],
            env=env,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if out.returncode:
            raise GateError(f"{CASES.name} --python-work failed:\n{out.stderr[-4000:]}")
        return json.loads(out.stdout.strip().splitlines()[-1])

    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            first, second = pool.map(run, range(2))
    except GateError as e:
        return None, f"Python work not counted: {e}"
    if not first:
        return None, "Python work not counted: it takes sys.monitoring (Python 3.12+)"
    if first != second:
        differ = sorted(n for n in set(first) | set(second) if first.get(n) != second.get(n))
        return None, (
            f"Python work not used: two identical runs counted it differently ({', '.join(differ)})"
        )
    return first, ""


def _count(x: float) -> str:
    return f"{x:,.3f}".rstrip("0").rstrip(".")


def work_changes(expected: dict[str, float], measured: dict[str, float]) -> list[str]:
    """Each counter of Python work with its change per repetition, or an
    empty list if none changed."""
    keys = [*measured, *(k for k in expected if k not in measured)]
    if all(expected.get(k) == measured.get(k) for k in keys):
        return []
    changes = []
    for k in keys:
        old, new = expected.get(k), measured.get(k)
        if old == new:
            changes.append(f"{k} same")
        elif old is None or new is None:
            changes.append(
                f"{k} {'none' if old is None else _count(old)} -> "
                f"{'none' if new is None else _count(new)}"
            )
        else:
            delta = _count(new - old)
            changes.append(
                f"{k} {_count(old)} -> {_count(new)} ({'' if new < old else '+'}{delta})"
            )
    return changes


# Instructions per repetition below which a count is wrong, not fast (a
# compile is well above a million), and pairs of cases whose first does
# several times the work of the second.
MINIMUM = 200
MINIMUM_COMPILE = 100_000
HEAVIER = (
    ("first_compile", "compile_bell"),
    ("compile_surface_d5", "compile_bell"),
    ("compile_calls_large", "compile_calls_small"),
)


def implausible(measured: dict[str, Measured], cases: set[str]) -> list[str]:
    """Why the instruction counts cannot be right: a case of *cases* without
    a count, a count of zero or one too small for its case, or a heavier
    case measuring less than twice a lighter one. Empty if plausible."""
    problems = [f"{name}: no instruction count measured" for name in sorted(cases - set(measured))]
    for name, m in sorted(measured.items()):
        least = MINIMUM_COMPILE if name.startswith(("compile_", "first_compile")) else MINIMUM
        if not m.instructions or not all(x >= least for x in m.instructions):
            problems.append(
                f"{name}: {', '.join(_num(x) for x in m.instructions) or 'no'} instructions "
                f"per repetition measured; a count below {least:,} is not plausible"
            )
    for heavy, light in HEAVIER:
        if heavy in measured and light in measured:
            h, lo = min(measured[heavy].instructions), min(measured[light].instructions)
            if not h >= 2 * lo or lo <= 0:
                problems.append(
                    f"{heavy} measured {_num(h)} instructions per repetition, not more than "
                    f"twice {light}'s {_num(lo)}: the counts do not follow the work"
                )
    return problems


@dataclass
class Row:
    name: str
    baseline: float | None
    measured: float | None
    spread: float | None
    cycles: float | None
    sizes: str
    verdict: str
    tolerance: float = REGRESSION
    # How the case's Python work compares with the baseline's: None where
    # either has none, else each counter's change (empty if the same).
    work_changes: list[str] | None = None
    # The change of the case's count from its base's, where the base ran
    # the same Python work, and why the row is not compared with its base
    # (None: no base was given; "": it is).
    own: float | None = None
    scope: str | None = None
    # Against a base this job built or measured: the named functions whose
    # calls differed, each with its layout class (None: outside them); None
    # where the calls were not compared.
    calls: list[tuple[str, str | None]] | None = None
    # Whether the row is layout only against its base (the base's Python
    # work, and no call outside the layout classes differs): None where no
    # base judged it. ``base_name`` is the base's short commit.
    on_base: bool | None = None
    base_name: str = ""
    # Where that base ran: built in this job (a change to native code), or
    # its Python sources measured on this build.
    base_where: str = "built in this job"
    # For a layout-only row that fails against its base: what to bring to
    # the owner (shift_detail), from the dumps of the tree and the base.
    shift: str = ""
    # For a row that failed as a regression, with an address-keyed function
    # that moved against the baseline's records or its base: its judgement on
    # the rest.
    keyed: Keyed | None = None

    @property
    def work(self) -> str:
        if self.work_changes is None:
            return ""
        return "changed" if self.work_changes else "same"

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.measured is None or not self.baseline:
            return None
        return self.measured / self.baseline - 1


def compare(
    baseline: dict[str, Any],
    sizes: dict[str, dict[str, int]],
    instructions: dict[str, Measured] | None,
    python_work: PythonWork | None = None,
    base: Base | None = None,
    scope: str | None = None,
    native: NativeBase | None = None,
    tree_calls: dict[str, Calls] | None = None,
    base_calls: dict[str, Calls] | None = None,
    base_name: str = "",
) -> list[Row]:
    """One row per case of the baseline or of this run, with its verdict:
    ``ok``, ``faster`` (within ``IMPROVEMENT``) or a failure: ``slower``,
    ``improved`` (update the baseline), ``sizes``, ``new``, ``removed``,
    ``no baseline``. In bootstrap mode a case without an instruction count
    is ``bootstrap``, which passes. Where this run and the baseline both
    have the case's Python work, the row has its changes. Against a base
    (*base*, the base's Python sources measured on this build, with its
    calls *base_calls*; or *native*, a base built in this job), a row is
    layout only when the base ran the same Python work and every function
    outside the layout classes was called as many times in this tree
    (*tree_calls*) as there; a failing layout-only row is ``layout``, which
    passes, when its count is at most its tolerance above the base's (any
    lower count passes). *base_name* names the base in the reasons."""
    cases = baseline.get("cases", {})
    bootstrap = baseline.get("bootstrap") is True
    rows = []
    for name in sorted(set(cases) | set(sizes)):
        expected = cases.get(name)
        if name not in sizes:
            rows.append(Row(name, None, None, None, None, "", "removed"))
            continue
        if expected is None:
            rows.append(Row(name, None, None, None, None, "", "new"))
            continue
        changed = [
            f"{key} {expected.get(key)} -> {value}"
            for key, value in sorted(sizes[name].items())
            if expected.get(key) != value
        ]
        changed += [
            f"{key} {expected[key]} -> none"
            for key in sorted(expected)
            if key not in NOT_SIZES and key not in sizes[name]
        ]
        tolerance = float(expected.get("tolerance", REGRESSION))
        row = Row(
            name,
            expected.get("instructions"),
            None,
            None,
            None,
            "; ".join(changed) or "same",
            "sizes" if changed else "ok",
            tolerance,
        )
        if instructions is not None:
            m = instructions.get(name)
            if m is not None:
                low, high = min(m.instructions), max(m.instructions)
                row.measured = low
                row.spread = high / low - 1 if low else 0.0
                row.cycles = min(m.estimated_cycles)
            work = (python_work or {}).get(name)
            if work is not None and "python_work" in expected:
                row.work_changes = work_changes(expected["python_work"], work)
            if row.baseline is None:
                if row.verdict == "ok":
                    row.verdict = "bootstrap" if bootstrap else "no baseline"
            elif row.verdict == "ok" and row.delta is not None:
                if row.delta > tolerance:
                    row.verdict = "slower"
                elif row.delta < -IMPROVEMENT:
                    row.verdict = "improved"
                elif row.delta < -tolerance:
                    row.verdict = "faster"
            row.scope = scope
            row.base_name = base_name
            # Its base: built in this job (a change to native code), else
            # the base's Python sources on this build.
            judged: tuple[float, Any, Calls | None] | None = None
            if native is not None and name in native and tree_calls is not None:
                judged = native[name]
            elif base is not None and name in base:
                row.base_where = "measured on this build"
                count, base_work = base[name]
                judged = (
                    count,
                    base_work,
                    None if base_calls is None else base_calls.get(name, {}),
                )
            # A baseline without Python work (format 1) is not compared with a
            # base, and a gain only with the baseline's work, so that a gain
            # from changed work is kept by a new baseline.
            if (
                judged is not None
                and row.measured
                and work is not None
                and row.work_changes is not None
                and (row.verdict != "improved" or row.work_changes == [])
            ):
                count, base_work, calls = judged
                if base_work != work:
                    row.on_base = False
                    if native is not None and name in native:
                        row.scope = "its base, built in this job, ran other Python work"
                elif count:
                    row.own = row.measured / count - 1
                    if calls is not None:
                        row.calls = call_differences((tree_calls or {}).get(name, {}), calls)
                    row.on_base = row.calls is None or all(cls is not None for _, cls in row.calls)
                    # Layout only: judged against its base, at the same limit
                    # above it; lower passes and is not a gain to keep.
                    if (
                        row.on_base
                        and row.verdict in ("slower", "improved")
                        and row.own <= row.tolerance
                    ):
                        row.verdict = "layout"
        rows.append(row)
    return rows


FAILING = {"slower", "improved", "sizes", "new", "removed", "no baseline"}


def failure_kind(r: Row) -> str | None:
    """What a failure on instructions is: ``layout only`` (against its base,
    the base's Python work and no call outside the layout classes differs;
    without a base, the same Python work as the baseline's and a change
    within ``LAYOUT_SHIFT`` either way), else a ``regression`` or a
    ``gain``; None for other failures."""
    if r.verdict not in ("slower", "improved"):
        return None
    if r.on_base is True:
        return "layout only"
    # Calls of its own code differ on its base: not layout.
    if r.on_base is False and r.own is not None:
        return "regression" if r.verdict == "slower" else "gain"
    if may_be_layout(r) and r.work_changes == []:
        return "layout only"
    return "regression" if r.verdict == "slower" else "gain"


def may_be_layout(r: Row) -> bool:
    """Whether its base could judge a failure on instructions: within
    ``LAYOUT_SHIFT`` either way, with the Python work of both the baseline
    and this run. A slower row is judged on its base's work, not the
    baseline's (code merged after the baseline can change a case's work);
    a gain only with the baseline's work, so that a gain from changed work
    is kept by a new baseline."""
    return (
        r.verdict in ("slower", "improved")
        and r.delta is not None
        and abs(r.delta) <= LAYOUT_SHIFT
        and r.work_changes is not None
        and (r.work_changes == [] or r.verdict == "slower")
    )


def _by_class(calls: list[tuple[str, str | None]] | None) -> str:
    """The calls that differed from the base's, counted by layout class."""
    if calls is None:
        return "calls not compared"
    counted = collections.Counter(cls for _, cls in calls if cls is not None)
    if not counted:
        return "no call differs"
    return (
        "calls differ only in layout classes ("
        + ", ".join(f"{cls} {n}" for cls, n in sorted(counted.items()))
        + ")"
    )


def _base(r: Row) -> str:
    return f"its base {r.base_name}" if r.base_name else "its base"


def failure_reason(r: Row) -> str:
    """Why a failing row fails. A failure on instructions says its kind
    (``failure_kind``) and what judged it: its base built in this job (a
    layout-only row, by its change from the base), else the stored
    baseline, with why its base did not judge it. Every kind fails."""
    kind = failure_kind(r)
    reason = {
        "slower": f"{_pct(r.delta)} instructions (limit +{100 * r.tolerance:g}%)",
        "improved": f"{_pct(r.delta)} instructions (limit -{100 * IMPROVEMENT:g}%)",
        "sizes": f"program sizes changed: {r.sizes}",
        "new": "not in the baseline",
        "removed": "in the baseline but no longer a case",
        "no baseline": "no instruction count in the baseline",
    }[r.verdict]
    if kind is None:
        return f"{r.name}: {reason}"
    keyed = (
        f"; judged outside its address-keyed functions: {keyed_text(r.keyed, r.tolerance)}: fails"
        if r.keyed is not None
        else ""
    )
    if r.on_base is True:
        return (
            f"{r.name}: layout only, {_pct(r.own)} from {_base(r)} {r.base_where} "
            f"(limit +{100 * r.tolerance:g}%): fails, the owner's to accept or not; same Python "
            f"work; {_by_class(r.calls)}"
            + (f"; bring to the owner: {r.shift}" if r.shift else "")
            + keyed
        )
    if kind == "gain":
        reason += ": update the baseline to keep the gain"
    if kind == "layout only":
        reason += f"; {LAYOUT_ONLY}"
    elif r.work_changes == []:
        reason += f"; {NATIVE_ONLY}"
    elif r.work_changes:
        reason += f"; Python work changed per repetition: {', '.join(r.work_changes)}"
    if r.on_base is False:
        outside = [what for what, cls in (r.calls or []) if cls is None]
        if r.own is None:
            why = f"{_base(r)} ran other Python work"
        else:
            # Every one, so that a reader can audit the row.
            why = f"calls differ outside the layout classes on {_base(r)}: {', '.join(outside)}"
        reason += f"; judged against the stored baseline: {why}"
    elif r.scope:
        reason += f"; judged against the stored baseline: no base built in this job ({r.scope})"
    return f"{r.name}: {kind}, {reason}{keyed}"


def layout_notes(rows: list[Row]) -> list[str]:
    """One line for each row that passed as layout."""
    return [
        f"{r.name}: layout, {_pct(r.own)} from {_base(r)} {r.base_where} "
        f"(limit +{100 * r.tolerance:g}%): passes, not a gain to keep; {_by_class(r.calls)}"
        for r in rows
        if r.verdict == "layout" and r.own is not None
    ]


def keyed_notes(rows: list[Row]) -> list[str]:
    """One line for each row that passed as address-keyed."""
    return [
        f"{r.name}: address-keyed, {_pct(r.delta)} against the stored baseline"
        + ("" if r.keyed.records else f" and {_pct(r.keyed.whole)} from {_base(r)} {r.base_where}")
        + f"; {keyed_text(r.keyed, r.tolerance)}: passes, not a gain to keep"
        for r in rows
        if r.verdict == "address-keyed" and r.keyed is not None
    ]


def not_python_or_text(files: list[str]) -> list[str]:
    """The paths of *files* that are not Python sources or text (PYTHON_OR_TEXT;
    ci.yml holds the perf-gate job)."""
    return [
        f
        for f in files
        if f == ".github/workflows/ci.yml"
        or not any(fnmatch.fnmatchcase(f, p) for p in PYTHON_OR_TEXT)
    ]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return None if out.returncode else out.stdout


def build_inputs(rev: str) -> str | None:
    """What the perf-gate job reads of pyproject.toml and uv.lock at *rev*:
    the build tables, and the locked packages of the project's dependencies
    and its build group (uv sync --no-default-groups --group build): every
    entry of a package's name, since a lock split by Python version may
    lock a name more than once."""
    try:
        py = tomllib.loads(_git("show", f"{rev}:pyproject.toml") or "")
        lock = tomllib.loads(_git("show", f"{rev}:uv.lock") or "")
    except tomllib.TOMLDecodeError:
        return None
    tool = py.get("tool", {})
    keep: dict[str, Any] = {
        "build-system": py.get("build-system"),
        "build": py.get("dependency-groups", {}).get("build"),
        **{k: tool.get(k) for k in ("uv", "setuptools_scm")},
        # What goes into an sdist does not reach the build this job makes.
        "scikit-build": {k: v for k, v in (tool.get("scikit-build") or {}).items() if k != "sdist"},
    }
    packages: dict[str, list[dict[str, Any]]] = {}
    for p in lock.get("package", []):
        packages.setdefault(p["name"], []).append(p)
    root = [p for p in lock.get("package", []) if p.get("source", {}).get("virtual") == "."]
    todo = [
        d["name"]
        for p in root
        for d in p.get("dependencies", []) + p.get("dev-dependencies", {}).get("build", [])
    ]
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name not in seen and name in packages:
            seen.add(name)
            todo += [d["name"] for p in packages[name] for d in p.get("dependencies", [])]
    keep["locked"] = [p for n in sorted(seen) for p in packages[n]]
    return json.dumps(keep, sort_keys=True, default=str)


def measured_paths(commit: str, head: str = "HEAD") -> list[str] | None:
    """The paths of measured code in which *head* differs from *commit* (a
    rename lists both), or None if git cannot compare them."""
    out = _git("diff", "--name-only", "--no-renames", commit, head, "--")
    if out is None:
        return None
    others = not_python_or_text([line for line in out.splitlines() if line])
    if set(others) & set(BUILD_FILES):
        before, after = build_inputs(commit), build_inputs(head)
        if before is not None and before == after:
            others = [f for f in others if f not in BUILD_FILES]
    return others


def base_scope(commit: str) -> str:
    """Why this tree's rows are not compared with its base *commit*: it is
    not an ancestor of this tree (the commit the change was applied to: a
    pull request's first parent, a branch's merge base with main), or the
    change touches measured code, or git cannot tell; "" when they are."""
    base = _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    if base is None:
        return f"git cannot find the base {commit[:8]}"
    if _git("merge-base", "--is-ancestor", base.strip(), "HEAD") is None:
        return f"the base {commit[:8]} is not an ancestor of this tree"
    others = measured_paths(commit)
    if others is None:
        return f"git cannot compare this tree with its base {commit[:8]}"
    if not others:
        return ""
    more = f" and {len(others) - 3} more" if len(others) > 3 else ""
    return f"the change touches measured code: {', '.join(others[:3])}{more}"


def measure_base(
    commit: str,
    python: str,
    env: dict[str, str],
    pythonpath: str,
    processes: int,
    workdir: Path,
    dumps: Path,
) -> tuple[Base | None, str]:
    """Each case's count and Python work on the base's Python sources, which
    replace the tree's under FRONTEND for the measurement, so that the
    same build, paths and environment run them; or None and why not. The
    tree's sources are restored after it (FRONTEND must have no
    uncommitted change; a file only the base has stays, untracked, and
    the build has no link to it)."""
    if _git("status", "--porcelain", "--", FRONTEND) != "":
        return None, f"{FRONTEND} differs from HEAD's, or git cannot tell"
    if _git("restore", f"--source={commit}", "--worktree", "--", FRONTEND) is None:
        return None, f"git cannot check out the base's {FRONTEND}"
    try:
        _byte_compile(python, env, pythonpath)
        measured, _, _ = measure_instructions(python, env, processes, workdir, dumps)
        work, why = measure_python_work(python, env)
    except (GateError, subprocess.CalledProcessError) as e:
        return None, f"the base could not be measured: {str(e).splitlines()[0]}"
    finally:
        if _git("restore", "--source=HEAD", "--worktree", "--", FRONTEND) is None:
            raise GateError(f"git cannot restore {FRONTEND} after measuring the base")
    if work is None:
        return None, f"on the base, {why}"
    return {n: (min(m.instructions), work.get(n)) for n, m in measured.items()}, ""


# A native function, as (object file name, function name).
Function = tuple[str, str]
Calls = dict[Function, int]
# Each case's count, Python work and calls on a base built in this job.
NativeBase = dict[str, tuple[float, dict[str, float] | None, Calls]]


def dump_calls(text: str) -> Calls:
    """The calls made to each function in one Callgrind dump, from its
    ``calls=`` lines: the callee is the ``cfn=`` before it, in the object
    of the ``cob=`` before that or else of its caller (``ob=``). Names and
    objects are compressed to ``(id)`` after their first use; a recursion
    level's ``'n`` suffix counts with its function."""
    names: dict[str, str] = {}
    objects: dict[str, str] = {}

    def expand(table: dict[str, str], value: str) -> str:
        m = re.match(r"\((\d+)\)(?: (.*))?$", value)
        if m is None:
            return value
        if m.group(2) is not None:
            table[m.group(1)] = m.group(2)
        return table.get(m.group(1), value)

    calls: Calls = {}
    ob = cob = cfn = ""
    for line in text.splitlines():
        if line.startswith("ob="):
            ob = expand(objects, line[3:].strip())
        elif line.startswith("cob="):
            cob = expand(objects, line[4:].strip())
        elif line.startswith("fn="):
            expand(names, line[3:].strip())
            cob = ""
        elif line.startswith("cfn="):
            cfn = expand(names, line[4:].strip())
        elif line.startswith("calls="):
            key = (Path(cob or ob).name, re.sub(r"'\d+$", "", cfn))
            calls[key] = calls.get(key, 0) + int(line[6:].split()[0])
            cob = ""
    return calls


def layout_class(function: Function) -> str | None:
    """The layout class of *function* (LAYOUT_CLASSES), or None."""
    obj, name = function
    for label, obj_pattern, name_pattern in LAYOUT_CLASSES:
        if (obj_pattern is None or re.search(obj_pattern, obj)) and (
            name_pattern is None or re.search(name_pattern, name)
        ):
            return label
    return None


def call_differences(tree: Calls, base: Calls) -> list[tuple[str, str | None]]:
    """The functions called a different number of times in *tree* than in
    *base*, each as "object:name (base -> tree)" with its layout class."""
    differ = []
    for function in sorted(set(tree) | set(base)):
        before, after = base.get(function, 0), tree.get(function, 0)
        if before != after:
            obj, name = function
            differ.append((f"{obj}:{name} ({before} -> {after})", layout_class(function)))
    return differ


def case_texts(dumps: Path) -> dict[str, str]:
    """Each case's dump, from the dumps of one Callgrind run."""
    out: dict[str, str] = {}
    for path in sorted(dumps.glob("callgrind.out*")):
        text = path.read_text(errors="replace")
        trigger = re.search(r"^desc: Trigger: Client Request: (.+)$", text, re.MULTILINE)
        if trigger is not None:
            out[trigger.group(1).strip()] = text
    return out


def case_calls(dumps: Path) -> dict[str, Calls]:
    """Each case's calls, from the dumps of one Callgrind run."""
    return {name: dump_calls(text) for name, text in case_texts(dumps).items()}


def dump_costs(text: str) -> dict[Function, int]:
    """The instructions each function executed itself (exclusive ``Ir``) in
    one Callgrind dump: its cost lines, not those after a ``calls=`` line,
    which are what a call cost in all."""
    names: dict[str, str] = {}
    objects: dict[str, str] = {}

    def expand(table: dict[str, str], value: str) -> str:
        m = re.match(r"\((\d+)\)(?: (.*))?$", value)
        if m is None:
            return value
        if m.group(2) is not None:
            table[m.group(1)] = m.group(2)
        return table.get(m.group(1), value)

    costs: dict[Function, int] = {}
    ob = ""
    fn: Function | None = None
    after_call = False
    for line in text.splitlines():
        if line.startswith("ob="):
            ob = expand(objects, line[3:].strip())
        elif line.startswith("cob="):
            expand(objects, line[4:].strip())
        elif line.startswith("fn="):
            fn = (Path(ob).name, re.sub(r"'\d+$", "", expand(names, line[3:].strip())))
        elif line.startswith("cfn="):
            expand(names, line[4:].strip())
        elif line.startswith("calls="):
            after_call = True
        elif line[:1].isdigit() or line[:1] in "+-*":
            if after_call:
                after_call = False
                continue
            parts = line.split()
            if fn is not None and len(parts) > 1 and parts[1].isdigit():
                costs[fn] = costs.get(fn, 0) + int(parts[1])
    return costs


def _short(name: str, width: int = 90) -> str:
    """A function's name without its template and call arguments, cut to
    *width* characters: enough to find it in the dumps."""
    out, depth = [], 0
    for ch in name:
        if ch in "<(":
            if depth == 0:
                out.append("<...>" if ch == "<" else "(...)")
            depth += 1
        elif ch in ">)" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    short = "".join(out)
    return short if len(short) <= width else short[: width - 3] + "..."


def shift_detail(tree: str, base: str, top: int = 3) -> str:
    """What carried a layout-only row's shift from its base's dump *base*
    to this tree's *tree*: the change of instructions by layout class (and
    outside them), and the *top* functions whose cost moved most, each with
    its calls and its cost per call on both sides."""
    after, before = dump_costs(tree), dump_costs(base)
    tree_calls, base_calls = dump_calls(tree), dump_calls(base)
    moved = {f: after.get(f, 0) - before.get(f, 0) for f in set(after) | set(before)}
    by_class: collections.Counter[str] = collections.Counter()
    for function, d in moved.items():
        by_class[layout_class(function) or "outside the layout classes"] += d
    classes = ", ".join(
        f"{cls} {d:+,}" for cls, d in sorted(by_class.items(), key=lambda kv: -abs(kv[1])) if d
    )
    largest = []
    for function, d in sorted(moved.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:top]:
        if not d:
            break
        n, m = base_calls.get(function, 0), tree_calls.get(function, 0)
        per = (
            f"{before.get(function, 0) / n:,.1f} -> {after.get(function, 0) / m:,.1f} "
            "instructions per call"
            if n and m
            else f"calls {n:,} -> {m:,}"
        )
        largest.append(f"{function[0]}:{_short(function[1])} ({m:,} calls, {per}, {d:+,})")
    return f"the shift by class: {classes or 'none'}; the largest changes: " + (
        "; ".join(largest) or "none"
    )


def address_keyed(function: Function) -> bool:
    """Whether *function* is one of ADDRESS_KEYED."""
    return any(re.search(pattern, function[1]) for pattern in ADDRESS_KEYED)


@dataclass(frozen=True)
class Keyed:
    """A row judged outside its address-keyed functions, against the
    baseline's records of them (*records*) or its base's dump. *whole* and
    *remainder* are its change from that reference, all of it and less the
    listed functions' change in cost per call, each a fraction of the
    reference's count; *excluded* is those functions' change as a
    fraction of the tree's count. *against_baseline* is the rest's change
    from the stored baseline, which judges it (*passes*: at most the row's
    tolerance). *taken_out* names each function taken out, with its calls
    and its cost per call on both sides; *left_in* each listed function
    entered a different number of times."""

    whole: float
    remainder: float
    excluded: float
    taken_out: tuple[str, ...]
    left_in: tuple[str, ...]
    against_baseline: float | None = None
    passes: bool = False
    # Whether its listed functions were judged against the baseline's records
    # of them (address_keyed_records), else against its base's dump.
    records: bool = False
    # Whether what was taken out exceeded LAYOUT_SHIFT of the baseline's count.
    too_much: bool = False


def address_keyed_judgement(tree: str, base: str) -> Keyed | None:
    """The address-keyed functions of a case's dump *tree* against its
    base's dump *base* (judge_address_keyed)."""
    return judge_address_keyed(
        dump_costs(tree), dump_costs(base), dump_calls(tree), dump_calls(base)
    )


# How many more or fewer times than its base a row may enter a listed
# function and still have its change in cost per call taken out: 1% of the
# base's entries, and at least one. With so few new entries the cost per
# call is measured almost wholly on the base's own keys.
ADDRESS_KEYED_ENTRIES = 0.01


def entry_cap(entries: int) -> int:
    """The most entries a listed function may gain or lose against
    *entries* on the base and still be judged by its cost per call."""
    return max(1, int(ADDRESS_KEYED_ENTRIES * entries))


def judge_address_keyed(
    after: dict[Function, int],
    before: dict[Function, int],
    tree_calls: Calls,
    base_calls: Calls,
) -> Keyed | None:
    """From each function's own instructions (*after* in the tree, *before*
    on the base) and its calls: for each ADDRESS_KEYED function entered at
    least once on both sides, and at most entry_cap more or fewer times in
    the tree, its change in cost per call over the tree's entries is taken
    out, and the entries it gained or lost stay in at the base's cost per
    call. None when nothing was taken out, where the rule has nothing to
    judge."""
    total, tree_total = sum(before.values()), sum(after.values())
    if not total or not tree_total:
        return None
    excluded = 0.0
    taken_out, left_in = [], []
    for function in sorted(set(after) | set(before)):
        if not address_keyed(function):
            continue
        n, m = base_calls.get(function, 0), tree_calls.get(function, 0)
        a, b = after.get(function, 0), before.get(function, 0)
        name = f"{function[0]}:{_short(function[1])}" if function[0] else _short(function[1])
        if n and m and abs(m - n) <= entry_cap(n):
            # Its change in cost per call over the tree's entries; the
            # entries it gained or lost stay in the rest at the base's
            # cost per call.
            took = m * (a / m - b / n)
            excluded += took
            if m == n and a != b:
                taken_out.append(
                    f"{name} ({n:,} calls, {b / n:,.1f} -> {a / n:,.1f} per call, {a - b:+,})"
                )
            elif m != n:
                taken_out.append(
                    f"{name} ({n:,} -> {m:,} calls, {b / n:,.1f} -> {a / m:,.1f} per call, "
                    f"taken out {took:+,.0f}, left in for {m - n:+,} calls "
                    f"{(m - n) * b / n:+,.0f})"
                )
        elif a != b or n != m:
            left_in.append(
                f"{name} (calls {n:,} -> {m:,}, more than {entry_cap(n) if n else 0} apart, "
                f"{a - b:+,})"
            )
    if not taken_out:
        return None
    whole = tree_total - total
    return Keyed(
        whole / total,
        (whole - excluded) / total,
        excluded / tree_total,
        tuple(taken_out),
        tuple(left_in),
    )


def apply_address_keyed(
    rows: list[Row],
    trees: dict[str, str],
    bases: dict[str, str],
    records: dict[str, dict[str, list[float]]] | None = None,
    reps: dict[str, int] | None = None,
) -> None:
    """Judge each row that fails as a regression (``slower``) outside its
    address-keyed functions: against the baseline's *records* of them when
    it has some, for the case's *reps* repetitions, else against its base's
    dump in *bases*. Its count less their change, against the stored
    baseline, passes as ``address-keyed`` when at most its tolerance above
    it (judge_row)."""
    for r in rows:
        if r.verdict != "slower" or r.name not in trees:
            continue
        recorded = (records or {}).get(r.name)
        if recorded and r.measured and r.baseline and r.name in (reps or {}):
            judge_row(
                r, judge_against_records(trees[r.name], recorded, r.baseline, (reps or {})[r.name])
            )
        elif r.name in bases:
            judge_row(r, address_keyed_judgement(trees[r.name], bases[r.name]))


def judge_row(r: Row, keyed: Keyed | None) -> None:
    """Judge a ``slower`` row *r* on *keyed*, its address-keyed functions
    against the baseline's records or its base: its count less their
    change, against the stored baseline, passes as ``address-keyed`` when
    at most its tolerance above it and what was taken out is at most
    LAYOUT_SHIFT of the baseline's count."""
    if keyed is None or not r.measured or not r.baseline:
        return
    taken = keyed.excluded * r.measured
    against = (r.measured - taken) / r.baseline - 1
    too_much = abs(taken) > LAYOUT_SHIFT * r.baseline
    r.keyed = replace(
        keyed,
        against_baseline=against,
        passes=against <= r.tolerance and not too_much,
        too_much=too_much,
    )
    if r.keyed.passes:
        r.verdict = "address-keyed"


def address_keyed_records(text: str, reps: int) -> dict[str, list[float]]:
    """Each ADDRESS_KEYED function a case's dump *text* entered: its entries
    and its own instructions per repetition, as a baseline records them."""
    costs, calls = dump_costs(text), dump_calls(text)
    # By name, summed over the objects that hold one.
    totals: dict[str, list[int]] = {}
    for function in set(costs) | set(calls):
        if address_keyed(function):
            total = totals.setdefault(function[1], [0, 0])
            total[0] += calls.get(function, 0)
            total[1] += costs.get(function, 0)
    return {
        name: [round(entries / reps, 3), round(cost / reps, 3)]
        for name, (entries, cost) in sorted(totals.items())
        if entries
    }


def judge_against_records(
    tree: str, records: dict[str, list[float]], baseline: float, reps: int
) -> Keyed | None:
    """judge_records on a case's dump *tree*."""
    return judge_records(dump_costs(tree), dump_calls(tree), records, baseline, reps)


def judge_records(
    after: dict[Function, int],
    tree_calls: Calls,
    records: dict[str, list[float]],
    baseline: float,
    reps: int,
) -> Keyed | None:
    """judge_address_keyed of a case's own instructions *after* and calls
    *tree_calls* against the baseline's *records* of its listed functions
    and its count *baseline*, each per repetition, for *reps* repetitions:
    like for like, without a base."""
    # The listed functions by name, summed over the objects that hold one,
    # as the records keep them.
    merged: dict[Function, int] = {}
    merged_calls: Calls = {}
    for function, cost in after.items():
        key = ("", function[1]) if address_keyed(function) else function
        merged[key] = merged.get(key, 0) + cost
    for function, n in tree_calls.items():
        key = ("", function[1]) if address_keyed(function) else function
        merged_calls[key] = merged_calls.get(key, 0) + n
    after, tree_calls = merged, merged_calls
    before: dict[Function, int] = {}
    base_calls: Calls = {}
    for name, (entries, cost) in records.items():
        before["", name] = round(cost * reps)
        base_calls["", name] = round(entries * reps)
    before["", "(the rest of the baseline)"] = round(baseline * reps) - sum(before.values())
    keyed = judge_address_keyed(after, before, tree_calls, base_calls)
    return None if keyed is None else replace(keyed, records=True)


def keyed_text(k: Keyed, tolerance: float) -> str:
    """What a row's address-keyed judgement says: the rest's change from the
    stored baseline and its limit, its change from the base, and every
    function taken out or left in."""
    where = "the baseline's records" if k.records else "the base"
    text = (
        f"outside the address-keyed functions {_pct(k.against_baseline)} against the stored "
        f"baseline (limit +{100 * tolerance:g}%)"
        + ("" if k.records else f", {_pct(k.remainder)} from the base")
        + f"; taken out, against {where}: "
        + "; ".join(k.taken_out)
    )
    if k.too_much:
        text += (
            f"; what was taken out is more than {100 * LAYOUT_SHIFT:g}% of the baseline's count, "
            "so the whole count is judged"
        )
    if k.left_in:
        text += "; left in, entered a different number of times: " + "; ".join(k.left_in)
    return text


def build_tree(source: Path, build: Path, python: str) -> None:
    """Configures and builds the package in *source* as the perf-gate job
    builds it (GATE_CMAKE_ARGS)."""
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            *GATE_CMAKE_ARGS,
            f"-DPython3_EXECUTABLE={python}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--target", "pulse"], check=True)


def has_symbols(pythonpath: Path) -> bool:
    """Whether the native extension under *pythonpath* keeps its symbols,
    which name its functions in the dumps."""
    native = sorted((pythonpath / "cudaq_pulse" / "_native").glob("_cudaq_pulse_native*"))
    nm = shutil.which("nm")
    if not native or nm is None:
        return False
    out = subprocess.run([nm, str(native[0])], capture_output=True, text=True)
    return bool(out.stdout.strip())


def native_base_scope(commit: str) -> str:
    """Why a change to native code cannot be judged against its base built
    in this job: it touches what would measure the base differently
    (NATIVE_BASE_REFUSED), or git cannot tell; "" when it can."""
    out = _git("diff", "--name-only", "--no-renames", commit, "HEAD", "--")
    if out is None:
        return f"git cannot compare this tree with its base {commit[:8]}"
    refused = [
        f for f in out.splitlines() if any(fnmatch.fnmatchcase(f, p) for p in NATIVE_BASE_REFUSED)
    ]
    if refused and set(refused) <= set(BUILD_FILES):
        before, after = build_inputs(commit), build_inputs("HEAD")
        if before is not None and before == after:
            refused = []
    return f"the change touches {', '.join(refused[:3])}" if refused else ""


def measure_native_base(
    commit: str, python: str, workdir: Path, dumps: Path
) -> tuple[NativeBase | None, str]:
    """Each case's count, Python work and calls on the base, checked out
    beside the tree and built as the job builds it; or None and why not."""
    if "CUDAQ_PULSE_KEEP_SYMBOLS" not in (_git("show", f"{commit}:CMakeLists.txt") or ""):
        return None, "the base cannot keep its extension's symbols (it predates the option)"
    source = workdir / "base-tree"
    if _git("worktree", "add", "--detach", str(source), commit) is None:
        return None, f"git cannot check out the base {commit[:8]}"
    try:
        cache = ROOT / ".cache"
        if cache.exists():
            (source / ".cache").symlink_to(cache)
        build = source / "build" / "dev"
        try:
            build_tree(source, build, python)
        except (OSError, subprocess.CalledProcessError) as e:
            return None, f"the base could not be built: {e}"
        pythonpath = str(build / "python")
        if not has_symbols(build / "python"):
            return None, "the base's extension has no symbols"
        env = _run_env(pythonpath, str(workdir))
        try:
            _byte_compile(python, env, pythonpath)
            measured, _, _ = measure_instructions(python, env, 1, workdir, dumps)
            work, why = measure_python_work(python, env)
        except (GateError, subprocess.CalledProcessError) as e:
            return None, f"the base could not be measured: {str(e).splitlines()[0]}"
        if work is None:
            return None, f"on the base, {why}"
        calls = case_calls(dumps / "run0")
        return {
            n: (min(m.instructions), work.get(n), calls.get(n, {})) for n, m in measured.items()
        }, ""
    finally:
        _git("worktree", "remove", "--force", str(source))


def failure_counts(rows: list[Row]) -> str:
    """The failures on instructions by kind, as "1 regression, 2 gains"."""
    counted = collections.Counter(failure_kind(r) for r in rows if r.verdict in FAILING)
    plural = {"regression": "regressions", "gain": "gains", "layout only": "layout only"}
    return ", ".join(
        f"{counted[k]} {k if counted[k] == 1 else plural[k]}" for k in plural if counted[k]
    )


def baseline_error(baseline: dict[str, Any], frozen: bool = False) -> str | None:
    """Why the baseline is not allowed, or None: an instruction count that
    is not a positive number, a ``bootstrap`` field next to counts, or any
    ``bootstrap`` field on a *frozen* branch (main, which releases are cut
    from; tools/release-meta.py refuses it for a release tag too).

    Bootstrap mode (instruction counts measured and reported, not gated)
    is allowed only while no case has a count: taking a CI run's
    measurements as the baseline drops the field, so the gate cannot stay
    in bootstrap mode once it has counts."""
    form = baseline.get("format", 1)
    if form not in READABLE_FORMATS:
        return (
            f"format {form!r} is not one this tool reads ({', '.join(map(str, READABLE_FORMATS))})"
        )
    cases = baseline.get("cases", {})
    bad = sorted(
        n
        for n, c in cases.items()
        if "instructions" in c
        and not (isinstance(c["instructions"], (int, float)) and c["instructions"] > 0)
    )
    if bad:
        return f"instruction counts must be positive numbers ({', '.join(bad)})"
    flag = baseline.get("bootstrap", False)
    if flag not in (True, False):
        return f"bootstrap must be true or false, not {flag!r}"
    if frozen and "bootstrap" in baseline:
        return (
            '"bootstrap" is not allowed on main: adopt a measured baseline before the '
            "freeze (CONTRIBUTING.md, The perf gate)"
        )
    counted = sorted(n for n, c in cases.items() if "instructions" in c)
    if flag and counted:
        return (
            "bootstrap is true but the baseline has instruction counts "
            f"({', '.join(counted[:3])}{', ...' if len(counted) > 3 else ''}); "
            'remove "bootstrap" so the counts are gated'
        )
    return None


def _pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:+.2f}%"


def _spread(x: float) -> str:
    # Any difference at all shows: identical runs should not differ.
    return "0" if x == 0 else f"{100 * x:.3g}%"


def _num(x: float | None) -> str:
    return "" if x is None or x != x else f"{x:,.0f}"


def render(rows: list[Row], markdown: bool) -> str:
    header = (
        "case",
        "baseline Ir",
        "measured Ir",
        "change",
        "spread",
        "est. cycles",
        "sizes",
        "Python work",
        "verdict",
    )
    table = [
        (
            r.name,
            _num(r.baseline),
            _num(r.measured),
            _pct(r.delta),
            "" if r.spread is None else _spread(r.spread),
            _num(r.cycles),
            r.sizes,
            r.work,
            r.verdict,
        )
        for r in rows
    ]
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(cells) + " |" for cells in table]
        return "\n".join(lines)
    widths = [max(len(str(c)) for c in col) for col in zip(header, *table, strict=True)]
    return "\n".join(
        "  ".join(
            str(c).ljust(w) if i in (0, 6, 7) else str(c).rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ).rstrip()
        for cells in (header, *table)
    )


def tolerance_error(baseline: dict[str, Any], tolerances: dict[str, float]) -> str | None:
    """Why the baseline's tolerances are not the ones the cases declare, or
    None."""
    recorded = {n: c["tolerance"] for n, c in baseline.get("cases", {}).items() if "tolerance" in c}
    if recorded == tolerances:
        return None
    return (
        f"its tolerances {recorded} are not those {CASES.name} declares {tolerances}: "
        "update the baseline (without Valgrind, --update keeps its counts)"
    )


def environment_error(baseline: dict[str, Any], environment: dict[str, str]) -> str | None:
    """Why counts measured in *environment* cannot be compared with the
    baseline's, or None: they were measured in another one."""
    expected = baseline.get("environment", {})
    differ = [
        f"{key} {expected.get(key)!r} -> {value!r}"
        for key, value in environment.items()
        if expected.get(key) != value
    ]
    if not differ:
        return None
    return (
        f"its counts were measured in another environment ({'; '.join(differ)}): "
        "take this run as the baseline (--help, taking a baseline)"
    )


def _source(processes: int) -> dict[str, Any] | None:
    """The CI run a measurement is taken in, or None outside GitHub Actions."""
    run = os.environ.get("GITHUB_RUN_ID")
    if not run:
        return None
    return {
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "run": int(run),
        "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        "event": os.environ.get("GITHUB_EVENT_NAME", ""),
        "ref": os.environ.get("GITHUB_REF", ""),
        "commit": os.environ.get("GITHUB_SHA", ""),
        "processes": processes,
    }


def as_baseline(
    environment: dict[str, str],
    sizes: dict[str, dict[str, int]],
    instructions: dict[str, Measured] | None,
    previous: dict[str, Any],
    tolerances: dict[str, float],
    source: dict[str, Any] | None = None,
    python_work: PythonWork | None = None,
    records: dict[str, dict[str, list[float]]] | None = None,
) -> dict[str, Any]:
    """This run in the baseline's format, with the tolerances the cases
    declare, the run it was measured in and the cases' Python work.
    Without instruction counts the previous baseline's are kept, with their
    Python work, environment and run."""
    old = previous.get("cases", {})
    cases = {}
    for name in sorted(sizes):
        entry: dict[str, Any] = dict(sorted(sizes[name].items()))
        if instructions is not None and name in instructions:
            m = instructions[name]
            entry["instructions"] = round(min(m.instructions))
            cycles = min(m.estimated_cycles)
            if cycles == cycles:
                entry["estimated_cycles"] = round(cycles)
            if python_work and name in python_work:
                entry["python_work"] = python_work[name]
            if records and records.get(name):
                entry["address_keyed"] = records[name]
        else:
            for key in ("instructions", "estimated_cycles", "python_work", "address_keyed"):
                if key in old.get(name, {}):
                    entry[key] = old[name][key]
        if name in tolerances:
            entry["tolerance"] = tolerances[name]
        cases[name] = entry
    out: dict[str, Any] = {"format": FORMAT}
    if instructions is not None:
        out["environment"] = environment
        if source is not None:
            out["source"] = source
    else:
        if previous.get("bootstrap") is True:
            out["bootstrap"] = True
        out["environment"] = previous.get("environment", {})
        if "source" in previous:
            out["source"] = previous["source"]
    out["cases"] = cases
    return out


def _update_command() -> str:
    run = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if run and repo:
        return f"gh run download {run} -R {repo} -n perf-gate-baseline -D benchmarks/baselines"
    return "python tools/perf-gate.py --update   (on Linux x86-64, in CI's environment)"


_REGENERATE = """\
taking a baseline:
  The instruction counts are only taken in CI's environment (the perf-gate
  job of .github/workflows/ci.yml); every run of the job uploads its
  measurements, in the baseline's format, as the perf-gate-baseline
  artifact. To take a new baseline, for a change that moves the counts on
  purpose or for a new toolchain pin, whose counts fail the gate until
  then:

  1. Once the change is on its pull request's branch in this repository,
     run only the perf gate there, with 5 identical Callgrind runs, and
     find the run:
       gh workflow run ci.yml --ref BRANCH -f perf_baseline=true
       gh run list -w ci.yml -b BRANCH -e workflow_dispatch -L 1
  2. In the run's job summary, check that the counts are valid (no "not
     valid" banner) and that no case's identical runs differed by more
     than its tolerance allows (the spread column; no "Identical runs
     differed" banner).
  3. Take its measurements as the baseline and commit them, with no other
     change, on top of the branch, which must still be the commit the run
     measured (the file's source.commit; if the branch has moved, take the
     baseline again):
       gh run download RUN -R OWNER/REPO -n perf-gate-baseline \\
         -D benchmarks/baselines
     The file then records the run, its environment (with the MLIR
     toolchain), the cases' tolerances and their Python work (format 2),
     and has no "bootstrap" field.
  4. The pull request's run of that commit must pass the gate.

regressions, gains and "layout only":
  Every failure on instructions fails the gate, and says its kind; the
  log's "Perf gate failed" line and the summary count them by kind.
  - regression: above the limit. Find the cause, or, if it is intended,
    take a new baseline.
  - gain: more than 1% below. Take a new baseline to keep it, so that a
    later regression cannot hide in it.
  - layout only: the same Python work as the baseline's (its "Python
    work" column says "same"), or as its base's where the row was
    compared with its base, and a change of at most 2% either way. The
    case ran the same bytecodes and calls, so its count moved below
    Python, most likely in the layout of the interpreter's caches and
    memory, which a change elsewhere can shift (one new class at import
    renumbers the type version tags of the types made after it). Judge it
    before taking a new baseline: compare its Callgrind dumps (the
    perf-gate-callgrind-dumps artifact, with callgrind_annotate) with
    those of the baseline's run, and look for the functions whose counts
    moved. If its own change moved it by layout, it passes (base
    comparison, below).
  A regression or gain lists each counter of Python work that changed;
  with the same work but more than 2%, it says the change is in native
  code.

  A case whose identical runs differ declares its own tolerance in
  benchmarks/perf_gate/cases.py, and the next baseline records it. To
  compare with an earlier baseline, such as the last one of the previous
  LLVM version:
       git show COMMIT:benchmarks/baselines/perf_gate.json > old.json
       python tools/perf-gate.py --baseline old.json

base comparison:
  A row's count also moves with the code merged before the change, so CI
  gives the gate the commit the change was applied to (--base-commit: the
  first parent of a pull request's merge commit, a branch's merge base
  with main for a dispatched run, or a push's previous tip; it must be an
  ancestor of the tree). A failure on instructions is then compared with
  its base, if the change keeps the base's measured code: it touches only
  the package's Python sources and files this job never reads
  (pyproject.toml and uv.lock only outside what the job builds), as git
  says. Native code,
  the build, ci.yml, the gate and its cases are measured code; if git
  cannot tell, the limits apply. The gate then checks out the base's
  core/frontend over the tree's, so that the same build, paths and
  environment run it, measures it as it measured the tree, and restores
  the tree's. A row whose case ran the same Python work on the base, and
  called every function outside the layout classes (below) as many times,
  is layout only: it passes as "layout", with a notice, when its count is
  at most 0.4% above the base's, and any lower count passes. The base's
  Python work is compared, not the baseline's: code merged after the
  baseline was taken can change a case's work, and a change on top of it
  ran the same. Hidden native work behind the same calls passes below
  0.4% above the base per change. A gain that passes this way is not
  kept: a later change can give it back. The base is measured only when
  a row fails on instructions, in about half a minute; its dumps are in
  the base directory of the dumps.

native base comparison:
  A change to native code moves the build's layout too, and its base's
  Python sources cannot run on its build. When such a change has a
  failure the base could judge, the gate checks the base out beside the
  tree, builds it as the job builds the tree (--build, with the
  extension's symbols kept: the same code, with names), and runs it once
  under Callgrind (2 to 3 minutes in all). A row is layout only when the
  case ran the same Python work on the base and every named function
  outside the layout classes was called as many times in the tree as on
  the base; it passes as "layout" when its count is at most 0.4% above
  the base's (any lower count passes). The classes are kinds of
  functions whose number of calls follows addresses and not the program:
  the equality and comparison callbacks of hash and ordered containers
  (one per probe of a table of pointers, or per comparison in a tree
  ordered by them: MLIR's uniquer, a DenseMapInfo's isEqual, a
  std::map's comparisons), the interpreter's own functions (its dict
  probes, version tags and caches; what Python runs is compared on its
  own), allocation (libc's allocator, C++'s new and delete, LLVM's
  buffers: when a container grows follows its probes), memory copies
  (libc's memcpy, memmove, memset and memcmp: a growing container copies
  its blocks), and code without a name, in libc or in no mapped object.
  Every other named function, libc's others among them, is compared. The
  row's notice lists every function whose calls differed, with its
  class, to audit. What passes this way: native work that changes no
  named function's number of calls outside those kinds, such as a longer
  loop inside one function or a costlier inline computation, and more
  lookups in a hash or ordered container whose only named trace is its
  equality or comparison callback (the lookups themselves are inlined),
  or more allocations or memory copies (an extra copy of a vector,
  inlined), up to 0.4% above the base per change. A new type or
  attribute changes the calls of the uniquer itself, and new work calls
  the program's own functions: both fail. A change to the gate, its
  cases, the job or the pins is not compared this way, nor a base from
  before the extension's symbols were kept, nor a tree whose extension
  has none; its dumps are in the native-base directory of the dumps.

address-keyed functions:
  A few functions of the pinned MLIR toolchain look a key up in a hash
  table, and the instructions each call takes follow where the key lands
  (ADDRESS_KEYED: the uniquer's parametric storage lookup, an attribute
  storage's initialization, an operation's registration, each keyed on a
  TypeID). A change that moves the build's layout moves their cost per
  call while they do the same work. The baseline records, for each case,
  each listed function's entries and instructions per repetition
  (address_keyed, taken with its count from the same run). A row that
  fails as a regression is also judged against those records: for each
  listed function it entered as often as the baseline's run, or at most
  1% of those entries (and at least one) more or fewer times, its change
  in cost per call over its own entries is taken out; the entries it
  gained or lost stay in the rest at the baseline's cost per call, so
  more work is counted as work at that price, and a function entered
  more or fewer times than that keeps all its change in the rest. When
  that rest is at most its tolerance (0.4% unless the case declares its
  own) above the stored baseline, and what was taken out is at most 2%
  of the baseline's count, it passes as "address-keyed", with a notice
  that gives each listed function's entries and cost per call in the
  baseline and in the row, what was taken out and what was left in; it
  is not a gain to keep and the baseline is not changed. The rest is
  compared like for like with the baseline's, so an address-keyed change
  merged into main does not stay in later rows, and nothing outside the
  listed functions can add up across merges. The baseline's environment
  pins the MLIR toolchain, so the listed functions' code is the same as
  when the baseline was taken. A baseline without these records (taken
  before this rule) is judged the same way against the change's base,
  when this job built or measured one (either way above). A name is
  added to the list in a pull request of its own, with what it keys on
  and the CI runs that showed it entered as often on both sides with its
  cost per call moved, and the owner approves it; only MLIR or LLVM
  functions may be listed.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=_REGENERATE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--base-commit",
        metavar="SHA",
        help="the commit this tree's change was applied to: a failure it could judge is "
        "compared with it (--help, base comparison)",
    )
    parser.add_argument("--update", action="store_true", help="rewrite the baseline")
    parser.add_argument("--out", type=Path, help="write this run in the baseline's format")
    parser.add_argument("--summary", type=Path, help="append a Markdown report to this file")
    parser.add_argument(
        "--processes", type=int, default=2, help="identical Callgrind runs, made side by side"
    )
    parser.add_argument("--pythonpath", default="build/dev/python")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--sizes-only", action="store_true", help="skip the instruction counts (no Valgrind)"
    )
    parser.add_argument(
        "--dumps", type=Path, help="keep Callgrind's dumps and logs in this new directory"
    )
    parser.add_argument(
        "--build",
        type=Path,
        metavar="DIR",
        help="only build the package into DIR as the perf-gate job does (GATE_CMAKE_ARGS), "
        "as it builds a base it compares by calls",
    )
    args = parser.parse_args()
    if args.build:
        build_tree(ROOT, (ROOT / args.build).resolve(), args.python)
        return 0
    if args.dumps and args.dumps.exists() and any(args.dumps.iterdir()):
        parser.error(f"--dumps {args.dumps} is not empty")

    pythonpath = str((ROOT / args.pythonpath).resolve())
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
    environment = _environment(args.python)
    # Instruction counts that cannot be right fail the gate in every mode.
    problems: list[str] = []
    differences: list[Difference] = []

    with tempfile.TemporaryDirectory(prefix="perf-gate-") as tmp:
        workdir = Path(tmp)
        env = _run_env(pythonpath, str(workdir))
        _byte_compile(args.python, env, pythonpath)
        sizes, tolerances = _structural(args.python, env)
        instructions = None
        python_work: PythonWork | None = None
        base: Base | None = None
        native: NativeBase | None = None
        tree_calls: dict[str, Calls] | None = None
        base_calls: dict[str, Calls] | None = None
        scope = None
        note = work_note = ""
        if args.sizes_only:
            note = "instruction counts skipped (--sizes-only)"
        elif shutil.which("valgrind") is None:
            note = "instruction counts skipped: valgrind is not installed"
        elif (platform.system(), platform.machine()) != ("Linux", "x86_64"):
            note = "instruction counts skipped: the baseline is for Linux x86-64"
        else:
            dumps = args.dumps.resolve() if args.dumps else workdir / "callgrind"
            try:
                instructions, differences, no_aslr = measure_instructions(
                    args.python, env, args.processes, workdir, dumps
                )
            except GateError as e:
                problems.append(f"Callgrind: {e}")
                note = "instruction counts not measured"
            else:
                problems += implausible(instructions, set(sizes))
                note = "address randomization " + (
                    "off (setarch -R)" if no_aslr else "on: setarch -R is not permitted here"
                )
                python_work, work_note = measure_python_work(args.python, env)
                # The base is measured only when a row fails on instructions:
                # a layout-only row is judged against it.
                if (
                    args.base_commit
                    and not problems
                    and python_work is not None
                    and any(
                        r.verdict == "slower"
                        and r.work_changes is not None
                        or r.verdict == "improved"
                        and r.work_changes == []
                        for r in compare(baseline, sizes, instructions, python_work)
                    )
                ):
                    scope = base_scope(args.base_commit)
                    if not scope:
                        base, scope = measure_base(
                            args.base_commit,
                            args.python,
                            env,
                            pythonpath,
                            args.processes,
                            workdir,
                            dumps / "base",
                        )
                        if base is not None:
                            base_calls = case_calls(dumps / "base" / "run0")
                            tree_calls = case_calls(dumps / "run0")
                    elif scope.startswith("the change touches measured code"):
                        # A change to native code: judged on its base built
                        # here, by the calls its native functions made.
                        why = native_base_scope(args.base_commit)
                        if not why and not has_symbols(Path(pythonpath)):
                            why = "this tree's extension has no symbols (build it with --build)"
                        if not why:
                            native, why = measure_native_base(
                                args.base_commit, args.python, workdir, dumps / "native-base"
                            )
                            tree_calls = case_calls(dumps / "run0")
                        if why:
                            scope += f"; and not against its base built here: {why}"
        # The dumps the judgements below read, while they exist: without
        # --dumps they go with the work directory.
        tree_texts: dict[str, str] = {}
        base_texts: dict[str, str] = {}
        if instructions is not None:
            tree_texts = case_texts(dumps / "run0")
            if native is not None or base is not None:
                where = dumps / ("native-base" if native is not None else "base") / "run0"
                base_texts = case_texts(where)

    # Each case's repetitions, from run0's dump and its count (run0's is the
    # first of a case's counts).
    reps = {
        name: max(1, round(sum(dump_costs(text).values()) / instructions[name].instructions[0]))
        for name, text in tree_texts.items()
        if instructions is not None
        and name in instructions
        and instructions[name].instructions
        and instructions[name].instructions[0]
    }
    rows = compare(
        baseline,
        sizes,
        instructions,
        python_work,
        base,
        scope,
        native,
        tree_calls,
        base_calls,
        (args.base_commit or "")[:8],
    )
    # A regression: judged outside its address-keyed functions, against the
    # baseline's records of them, else against its base in this job.
    records = {
        name: case["address_keyed"]
        for name, case in baseline.get("cases", {}).items()
        if case.get("address_keyed")
    }
    if any(r.verdict == "slower" for r in rows):
        apply_address_keyed(rows, tree_texts, base_texts, records, reps)
    # A layout-only row that fails against its base: what carried its shift.
    failing = [r for r in rows if r.on_base is True and r.verdict in FAILING]
    for r in failing:
        if r.name in tree_texts and r.name in base_texts:
            r.shift = shift_detail(tree_texts[r.name], base_texts[r.name])
    notes = layout_notes(rows)
    keyed = keyed_notes(rows)
    # Counts that failed the checks are reported but never become a baseline.
    # The listed functions' entries and cost per repetition, from this run's
    # dumps, for the baseline that this run may become.
    recorded: dict[str, dict[str, list[float]]] = {}
    if not problems:
        recorded = {
            name: address_keyed_records(text, reps[name])
            for name, text in tree_texts.items()
            if name in reps
        }
    measured = as_baseline(
        environment,
        sizes,
        None if problems else instructions,
        baseline,
        tolerances,
        _source(args.processes),
        python_work,
        recorded,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(measured, indent=2) + "\n")

    print(render(rows, markdown=False))
    if note:
        print(f"\n{note}")
    gated = any("instructions" in c for c in baseline.get("cases", {}).values())
    env_diff = None
    if instructions is not None and (gated or baseline.get("environment")):
        env_diff = environment_error(baseline, environment)
    if env_diff and not gated:
        print(f"\nNot gated, but {env_diff.removeprefix('its ')}")
    bootstrapped = [r for r in rows if r.verdict == "bootstrap"]
    # A case taken into the baseline must repeat exactly, or within what
    # its own tolerance allows above the default; once gated, a spread of
    # a quarter of its tolerance is worth a warning.
    unstable = [
        r
        for r in rows
        if r.spread is not None
        and (
            r.spread > max(0.0, r.tolerance - REGRESSION)
            if r.verdict == "bootstrap"
            else r.spread > r.tolerance / 4
        )
    ]
    spread_lines = [
        f"{r.name} varied by {_spread(r.spread)} ({r.spread * (r.measured or 0):.3g} "
        f"instructions per repetition) between {args.processes} identical runs"
        + (
            f": give it its own tolerance in {CASES.name}, or a larger one, before it blocks"
            if r.verdict == "bootstrap"
            else ""
        )
        for r in unstable
    ]
    dump_warnings, dump_note = dump_notes(differences)
    if len(dump_warnings) > 3:
        dump_warnings = [*dump_warnings[:3], f"and {len(dump_warnings) - 3} more dumps"]
    if dump_note:
        print(f"\n{dump_note}")
    if python_work and not any("python_work" in c for c in baseline.get("cases", {}).values()):
        work_note = (
            "the baseline has no Python work, so a moved instruction count is not labelled "
            "with it until the next baseline"
        )
    if work_note:
        print(f"\n{work_note}")
    for line in spread_lines + dump_warnings:
        print(f"::warning title=Perf gate::{line}")
    if os.environ.get("GITHUB_ACTIONS"):
        for line in notes:
            why = AS_NATIVE_BASE if native is not None else AS_BASE
            print(f"::notice title=Perf gate::{line}: {why}")
        for line in keyed:
            print(f"::notice title=Perf gate::{line}: {AS_KEYED}")
    if notes:
        why = AS_NATIVE_BASE if native is not None else AS_BASE
        print("\nLayout:\n" + "".join(f"  {line}\n" for line in notes) + f"  ({why})")
    if keyed:
        print("\nAddress-keyed:\n" + "".join(f"  {line}\n" for line in keyed) + f"  ({AS_KEYED})")
    if bootstrapped and not problems:
        print(
            "\nBOOTSTRAP MODE: the baseline has no instruction counts for "
            f"{len(bootstrapped)} case(s); they are measured and reported, not gated."
            f"\nTo gate them, take this run as the baseline and commit it:\n  {_update_command()}"
        )
        if os.environ.get("GITHUB_ACTIONS"):
            print(
                "::warning title=Perf gate::bootstrap mode: instruction counts are not gated "
                "until this run's perf-gate-baseline artifact is committed"
            )

    failures = [r for r in rows if r.verdict in FAILING]
    if args.update:
        if problems:
            print("\nThe instruction counts are not valid; the baseline was not written:")
            for line in problems:
                print(f"  {line}")
            return 1
        args.baseline.write_text(json.dumps(measured, indent=2) + "\n")
        print(f"\nWrote {args.baseline}")
        if instructions is None:
            print("Only the program sizes were updated: " + note)
        return 0

    lines = list(problems)
    lines += [failure_reason(r) for r in failures]
    kinds = failure_counts(rows)
    invalid = baseline_error(baseline, frozen=os.environ.get("GITHUB_REF") == "refs/heads/main")
    if invalid:
        lines.append(f"{args.baseline.name}: {invalid}")
    stale = tolerance_error(baseline, tolerances)
    if stale:
        lines.append(f"{args.baseline.name}: {stale}")
    if env_diff and gated:
        lines.append(f"{args.baseline.name}: {env_diff}")
    if os.environ.get("GITHUB_ACTIONS"):
        dumps_hint = "the perf-gate-callgrind-dumps artifact has each run's dumps and Valgrind log"
    elif args.dumps:
        dumps_hint = f"each run's dumps and Valgrind log are in {args.dumps}"
    else:
        dumps_hint = "keep the dumps with --dumps DIR"
    if os.environ.get("GITHUB_ACTIONS"):
        for line in lines:
            print(f"::error title=Perf gate::{line}")
    if args.summary:
        with args.summary.open("a") as f:
            f.write("## Perf gate: instructions and program sizes\n\n")
            if problems:
                f.write(
                    "> [!CAUTION]\n"
                    "> **The instruction counts are not valid**, so this run fails in every "
                    "mode and its counts are not in its perf-gate-baseline artifact: "
                    + " ".join(f"{line}." for line in problems)
                    + f" To inspect: {dumps_hint}.\n\n"
                )
            elif bootstrapped:
                f.write(
                    "> [!WARNING]\n"
                    f"> **Bootstrap mode.** {len(bootstrapped)} case(s) have no instruction "
                    "count in the baseline: they are measured and reported below but not "
                    "gated. Program sizes are gated as always. To make the gate blocking, "
                    "take this run's measurements as the baseline and commit "
                    "`benchmarks/baselines/perf_gate.json` (this drops `bootstrap`):\n"
                    ">\n"
                    f"> ```\n> {_update_command()}\n> ```\n\n"
                )
            if spread_lines:
                f.write(
                    "> [!CAUTION]\n"
                    "> **Identical runs differed.** "
                    + " ".join(f"{line}." for line in spread_lines)
                    + "\n\n"
                )
            if dump_warnings:
                f.write(
                    "> [!WARNING]\n"
                    "> **Callgrind's dump summaries differ from their totals** more than "
                    "the marks explain (the totals, the sum of the cost lines, are used): "
                    + " ".join(f"{line}." for line in dump_warnings)
                    + f" To inspect: {dumps_hint}.\n\n"
                )
            f.write(render(rows, markdown=True) + "\n\n")
            if notes:
                f.write(
                    "> [!NOTE]\n"
                    f"> **Layout ({len(notes)}).** "
                    + " ".join(f"{line}." for line in notes)
                    + f" In each, {AS_NATIVE_BASE if native is not None else AS_BASE} "
                    + (
                        "(`python tools/perf-gate.py --help`, native base comparison).\n\n"
                        if native is not None
                        else "(`python tools/perf-gate.py --help`, base comparison).\n\n"
                    )
                )
            if keyed:
                f.write(
                    "> [!NOTE]\n"
                    f"> **Address-keyed ({len(keyed)}).** "
                    + " ".join(f"{line}." for line in keyed)
                    + f" In each, {AS_KEYED} "
                    "(`python tools/perf-gate.py --help`, address-keyed functions).\n\n"
                )
            own = ", ".join(f"{n} +{100 * t:g}%" for n, t in sorted(tolerances.items()))
            f.write(
                f"Fails above +{100 * REGRESSION:.1f}% "
                + (f"(own tolerances: {own}) " if own else "")
                + f"or below -{100 * IMPROVEMENT:.1f}% "
                "instructions per repetition, or on any change in program size; a row with "
                "its base's Python work, and no call outside the layout classes that differs "
                "from its base's, is judged against its base built in this job: it passes as "
                f"layout up to +{100 * REGRESSION:.1f}% above it, and any lower count passes "
                "(verdict layout). A regression with a base in this job is also judged on its "
                "count less the change of the address-keyed functions it entered as often as "
                "the base: it "
                "passes as address-keyed when that rest is within its tolerance above the "
                "stored baseline. "
                f"Spread: largest difference between {args.processes} identical runs. "
                "Python work: the bytecodes, Python calls and C calls of a repetition, "
                "counted natively, against the baseline's; it labels a failure, and "
                "does not gate.\n\n"
            )
            if note:
                f.write(f"{note}\n\n")
            if work_note:
                f.write(f"{work_note[0].upper()}{work_note[1:]}.\n\n")
            if dump_note:
                f.write(f"{dump_note[0].upper()}{dump_note[1:]}.\n\n")
            if env_diff and not gated:
                f.write(f"Not gated, but {env_diff.removeprefix('its ')}.\n\n")
            if lines:
                f.write(
                    f"**Failed{f' ({kinds})' if kinds else ''}:**\n\n"
                    + "".join(f"- {line}\n" for line in lines)
                    + "\n"
                )
                if not problems:
                    f.write(
                        "If the change is intended, take this run as the baseline and "
                        f"commit it:\n\n```\n{_update_command()}\n```\n"
                    )
    if lines:
        print(f"\nPerf gate failed{f' ({kinds})' if kinds else ''}:")
        for line in lines:
            print(f"  {line}")
        if problems:
            print(f"\nThe instruction counts are not valid; {dumps_hint}.")
        else:
            print(f"\nIf the change is intended, update the baseline:\n  {_update_command()}")
        return 1
    if bootstrapped:
        print("\nPerf gate passed in bootstrap mode: instruction counts were not gated.")
    else:
        passed = [f"{len(notes)} layout"] * bool(notes) + [f"{len(keyed)} address-keyed"] * bool(
            keyed
        )
        print("\nPerf gate passed" + (f" ({', '.join(passed)})." if passed else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
