# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tools/perf-gate.py: reading Callgrind's dumps, the verdicts, and the
program sizes of its cases, which are the same on every platform."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import cudaq_pulse
import pytest

_ROOT = Path(__file__).resolve().parents[1]

# The wheels' tests run on a copy of tests/ without the tools.
if not (_ROOT / "tools" / "perf-gate.py").exists():
    pytest.skip("tools/perf-gate.py is not in this tree", allow_module_level=True)


def _gate():
    spec = importlib.util.spec_from_file_location("perf_gate", _ROOT / "tools" / "perf-gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


gate = _gate()

# A dump as Callgrind 3.22 writes it with --cache-sim=yes (the format of
# its manual, "Callgrind Format Specification"): the header's summary:, then
# cost lines under ob=/fl=/fn= with positions relative to the last ("+4",
# "-700", "*"), trailing zero events left out, file names given once and
# then referred to by number, a call (calls= and its inclusive cost, which
# is not self cost) and an inlined file (fi=, back with fe=). The totals:
# are the sum of the self-cost lines, added up by hand.
_SUMMARY = "941000 380000 185000 401 1810 600 40 160 60"
_DUMP = f"""\
# callgrind format
version: 1
creator: callgrind-3.22.0
pid: 4242
cmd:  python3 benchmarks/perf_gate/cases.py --measure libperf_gate_marks.so
part: 3


desc: I1 cache: 32768 B, 64 B, 8-way associative
desc: D1 cache: 32768 B, 64 B, 8-way associative
desc: LL cache: 8388608 B, 64 B, 16-way associative
desc: Timerange: Basic block 18263114 - 18402551
desc: Trigger: Client Request: compile_bell

positions: line
events: Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw
summary: {_SUMMARY}


ob=(1) /usr/bin/python3.12
fl=(1) ./Python/ceval.c
fn=(1) _PyEval_EvalFrameDefault
812 600000 250000 120000 300 900 200 40 60 10
+4 200000 80000 30000 100 300
cob=(2) /usr/lib/x86_64-linux-gnu/libc.so.6
cfi=(2) ./string/../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S
cfn=(2) __memmove_avx_unaligned_erms
calls=1500 0
* 90000 30000 30000 0 600 400 0 100 50
fi=(3) ./Include/object.h
-700 50000 20000 5000 0 10
fe=(1)
+700 1000 0 0 1

ob=(2)
fl=(2)
fn=(2)
0 90000 30000 30000 0 600 400 0 100 50

totals: {_SUMMARY}
"""


def test_a_dump_gives_its_case_and_counts():
    dump = gate.parse_dump(_DUMP)
    assert dump.name == "compile_bell"
    assert dump.events["Ir"] == 941_000
    assert dump.events["DLmw"] == 60
    assert dump.summary == dump.events


def test_the_dump_at_exit_is_not_a_case():
    assert (
        gate.parse_dump(_DUMP.replace("Client Request: compile_bell", "Program termination"))
        is None
    )


def test_a_dump_after_instrumentation_stopped_has_a_zero_summary_and_its_totals_are_used():
    # Switching instrumentation off zeroes the running total the summary
    # is taken from, not the cost lines: the first CI run's dumps.
    dump = gate.parse_dump(_DUMP.replace(f"summary: {_SUMMARY}", "summary: 0"))
    assert dump.events["Ir"] == 941_000
    assert dump.summary["Ir"] == 0


def test_a_dump_whose_totals_do_not_add_up_or_are_missing_is_an_error():
    with pytest.raises(gate.GateError, match="differ from the sum"):
        gate.parse_dump(_DUMP.replace("+700 1000 0 0 1", "+700 1001 0 0 1"))
    with pytest.raises(gate.GateError, match="totals"):
        gate.parse_dump(_DUMP.replace(f"totals: {_SUMMARY}\n", ""))


def test_a_dump_may_give_any_number_of_positions():
    instr_line = """\
desc: Trigger: Client Request: submit
positions: instr line
events: Ir
summary: 27
fn=(1) f
0x401000 12 20
+3 * 7
jump=2 +9 *
+9 *
calls=1 0x402000 40
* * 500
totals: 27
"""
    assert gate.parse_dump(instr_line).events == {"Ir": 27}
    no_positions = "desc: Trigger: Client Request: submit\npositions:\nevents: Ir Dr\n"
    no_positions += "fn=(1) f\n20 4\n7\ntotals: 27 4\n"
    assert gate.parse_dump(no_positions).events == {"Ir": 27, "Dr": 4}


def test_the_cycle_estimate_weights_cache_misses():
    events = gate.parse_dump(_DUMP).events
    # 1,506,000 accesses, 2,811 L1 misses of which 260 miss the last level too.
    assert gate.estimated_cycles(events) == (1_506_000 - 2811) + 5 * (2811 - 260) + 35 * 260
    assert gate.estimated_cycles({"Ir": 10}) is None


def _runs(**instructions):
    return {n: gate.Measured([float(x)] * 2, [0.0] * 2) for n, x in instructions.items()}


_PLAUSIBLE = {
    "first_compile": 90e6,
    "compile_bell": 1.2e6,
    "compile_surface_d5": 11e6,
    "compile_calls_small": 1.1e6,
    "compile_calls_large": 24e6,
    "submit": 1500,
}


def test_plausible_counts_pass_the_checks():
    assert gate.implausible(_runs(**_PLAUSIBLE), set(_PLAUSIBLE)) == []


def test_zero_missing_or_too_small_counts_are_errors():
    problems = gate.implausible(_runs(**{**_PLAUSIBLE, "submit": 0}), set(_PLAUSIBLE) | {"x"})
    assert any(p.startswith("submit: 0, 0 instructions") for p in problems)
    assert "x: no instruction count measured" in problems
    small = gate.implausible(_runs(**{**_PLAUSIBLE, "compile_bell": 9e4}), set(_PLAUSIBLE))
    assert any(p.startswith("compile_bell:") for p in small)
    nan = gate.Measured([float("nan")], [0.0])
    assert gate.implausible({"submit": nan}, {"submit"})


def test_a_heavier_case_must_measure_well_above_a_lighter_one():
    problems = gate.implausible(
        _runs(**{**_PLAUSIBLE, "compile_calls_large": 1.5e6}), set(_PLAUSIBLE)
    )
    assert problems == [
        "compile_calls_large measured 1,500,000 instructions per repetition, not more than "
        "twice compile_calls_small's 1,100,000: the counts do not follow the work"
    ]
    names = set(BASELINE["cases"])
    assert {name for pair in gate.HEAVIER for name in pair} <= names


def _baseline(instructions=1_000_000, **sizes):
    return {"cases": {"c": {"instructions": instructions, "op_count": 15, **sizes}}}


def _verdict(measured, baseline=None, sizes=None):
    runs = gate.Measured(instructions=list(measured), estimated_cycles=[0.0] * len(measured))
    rows = gate.compare(baseline or _baseline(), {"c": sizes or {"op_count": 15}}, {"c": runs})
    return rows[0]


@pytest.mark.parametrize(
    ("measured", "verdict"),
    [
        (1_000_000, "ok"),
        (1_003_999, "ok"),
        (1_004_001, "slower"),
        (1_005_000, "slower"),
        (996_001, "ok"),
        (995_000, "faster"),
        (989_000, "improved"),
    ],
)
def test_the_verdict_follows_the_change_in_instructions(measured, verdict):
    assert _verdict([measured]).verdict == verdict


def test_the_lowest_of_identical_runs_is_compared_and_their_spread_reported():
    row = _verdict([1_000_100, 1_000_000])
    assert row.measured == 1_000_000
    assert row.spread == pytest.approx(1e-4)


def test_a_case_may_carry_its_own_tolerance():
    baseline = _baseline()
    baseline["cases"]["c"]["tolerance"] = 0.02
    assert _verdict([1_010_000], baseline).verdict == "ok"


def test_any_change_in_program_size_fails():
    row = _verdict([1_000_000], sizes={"op_count": 16})
    assert row.verdict == "sizes"
    assert row.sizes == "op_count 15 -> 16"


def test_new_removed_and_unmeasured_cases_fail():
    rows = gate.compare(_baseline(), {"d": {}}, {"d": gate.Measured([1.0], [1.0])})
    assert {r.name: r.verdict for r in rows} == {"c": "removed", "d": "new"}
    row = _verdict([1_000_000], {"cases": {"c": {"op_count": 15}}})
    assert row.verdict == "no baseline"
    assert set(gate.FAILING) >= {"slower", "improved", "sizes", "new", "removed", "no baseline"}


def test_without_instruction_counts_only_the_sizes_are_compared():
    rows = gate.compare(_baseline(), {"c": {"op_count": 15}}, None)
    assert rows[0].verdict == "ok"


_SOURCE = {"repository": "o/r", "run": 7, "commit": "a" * 40, "processes": 5}


def test_a_baseline_without_instruction_counts_keeps_the_previous_ones():
    previous = {"environment": {"python": "3.12.15"}, "source": _SOURCE, **_baseline()}
    kept = gate.as_baseline(
        {"python": "other"}, {"c": {"op_count": 16}}, None, previous, {"c": 0.0045}, {"run": 8}
    )
    assert kept["environment"] == {"python": "3.12.15"}
    assert kept["source"] == _SOURCE
    assert kept["cases"]["c"] == {"instructions": 1_000_000, "op_count": 16, "tolerance": 0.0045}
    runs = {"c": gate.Measured([2_000_000.4, 2_000_100.0], [3_000_000.0, 3_000_001.0])}
    taken = gate.as_baseline({"python": "other"}, {"c": {"op_count": 16}}, runs, previous, {})
    assert taken["environment"] == {"python": "other"}
    assert "source" not in taken
    assert taken["cases"]["c"]["instructions"] == 2_000_000
    assert taken["cases"]["c"]["estimated_cycles"] == 3_000_000


def test_a_measured_baseline_records_its_run_and_the_declared_tolerances(monkeypatch):
    for name, value in {
        "GITHUB_RUN_ID": "41000000001",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/perf-baseline",
        "GITHUB_SHA": "b" * 40,
    }.items():
        monkeypatch.setenv(name, value)
    source = gate._source(5)
    assert source == {
        "repository": "o/r",
        "run": 41000000001,
        "attempt": 2,
        "event": "workflow_dispatch",
        "ref": "refs/heads/perf-baseline",
        "commit": "b" * 40,
        "processes": 5,
    }
    previous = {"cases": {"c": {"op_count": 15, "tolerance": 0.02}}}
    runs = {"c": gate.Measured([1_000_000.0] * 5, [2.0] * 5)}
    taken = gate.as_baseline(
        {"python": "3.12.15"}, {"c": {"op_count": 15}}, runs, previous, {}, source
    )
    assert taken["source"] == source
    assert "tolerance" not in taken["cases"]["c"]
    monkeypatch.delenv("GITHUB_RUN_ID")
    assert gate._source(2) is None


def test_the_baseline_must_record_the_tolerances_the_cases_declare():
    baseline = _baseline(tolerance=0.0045)
    assert gate.tolerance_error(baseline, {"c": 0.0045}) is None
    assert "declares" in gate.tolerance_error(baseline, {"c": 0.005})
    assert "declares" in gate.tolerance_error(baseline, {})
    assert "declares" in gate.tolerance_error(_baseline(), {"c": 0.0045})


def test_counts_from_another_environment_are_named():
    baseline = {"environment": {"mlir_toolchain": "mlir-toolchain-22.1.8", "python": "3.12.15"}}
    same = {"mlir_toolchain": "mlir-toolchain-22.1.8", "python": "3.12.15"}
    assert gate.environment_error(baseline, same) is None
    error = gate.environment_error(baseline, {**same, "mlir_toolchain": "mlir-toolchain-23.1.4"})
    assert "'mlir-toolchain-22.1.8' -> 'mlir-toolchain-23.1.4'" in error
    assert "python" not in error
    assert "valgrind None ->" in gate.environment_error(baseline, {**same, "valgrind": "3.22"})


def _differences(*values):
    return [(f"case{i} (run0/callgrind.out.{i})", v) for i, v in enumerate(values)]


def test_a_summary_above_every_total_by_the_same_few_instructions_is_the_marks():
    # The difference CI's dumps showed: the same 4 instructions in every one.
    same = {"Ir": 4, "Dr": 0, "Dw": 0}
    warnings, note = gate.dump_notes(_differences(same, same, dict(same)))
    assert warnings == []
    assert note.startswith("every dump's summary: is Ir +4") and "marks' own blocks" in note
    assert gate.dump_notes(_differences({"Ir": -2}, {"Ir": -2}))[0] == []
    assert gate.dump_notes(_differences({"Ir": 0, "Dr": 0})) == ([], "")
    assert gate.dump_notes([]) == ([], "")


def test_a_summary_that_differs_from_case_to_case_or_by_much_is_a_warning():
    warnings, note = gate.dump_notes(_differences({"Ir": 4}, {"Ir": 5}, {"Ir": 4}))
    assert warnings == [
        "case1 (run0/callgrind.out.1): the dump's summary: differs from its totals: by "
        "Ir +5, not Ir +4 like the other dumps"
    ]
    assert note.startswith("every other dump's summary: is Ir +4")
    assert gate.dump_notes(_differences({"Ir": 0}, {"Ir": 3}, {"Ir": 0}))[1] == ""
    many = {"Ir": gate.MARKS_INSTRUCTIONS + 1}
    assert gate.dump_notes(_differences(many, many)) == (
        [
            "case0 (run0/callgrind.out.0): the dump's summary: differs from its totals: by Ir +65",
            "case1 (run0/callgrind.out.1): the dump's summary: differs from its totals: by Ir +65",
        ],
        "",
    )
    # The first CI run's summaries were 0: below each total by the case.
    zeros = gate.dump_notes(_differences({"Ir": -941_000}, {"Ir": -27}))
    assert len(zeros[0]) == 2 and zeros[1] == ""


_WORK = {"bytecodes": 449.6, "python_calls": 16.0, "c_calls": 17.1}


def _with_work(work=_WORK, measured=1_005_000):
    baseline = _baseline()
    baseline["cases"]["c"]["python_work"] = dict(_WORK)
    runs = {"c": gate.Measured([measured], [0.0])}
    return gate.compare(baseline, {"c": {"op_count": 15}}, runs, {"c": dict(work)})[0]


def test_a_moved_count_with_the_same_python_work_still_fails_and_says_so():
    row = _with_work()
    assert row.verdict == "slower" and row.work == "same"
    assert gate.failure_kind(row) == "layout only"
    assert gate.failure_reason(row) == (
        f"c: layout only, +0.50% instructions (limit +0.4%); {gate.LAYOUT_ONLY}"
    )
    assert "judge before re-baselining" in gate.LAYOUT_ONLY
    # A shift the other way is layout only too, not a gain to keep.
    down = _with_work(measured=985_000)
    assert down.verdict == "improved" and gate.failure_kind(down) == "layout only"
    assert gate.failure_reason(down) == (
        f"c: layout only, -1.50% instructions (limit -1%); {gate.LAYOUT_ONLY}"
    )


def test_a_moved_count_with_other_python_work_shows_each_counter():
    row = _with_work({**_WORK, "bytecodes": 452.6, "c_calls": 16.1})
    assert row.work == "changed" and gate.failure_kind(row) == "regression"
    assert gate.failure_reason(row) == (
        "c: regression, +0.50% instructions (limit +0.4%); Python work changed per "
        "repetition: bytecodes 449.6 -> 452.6 (+3), python_calls same, c_calls 17.1 -> 16.1 (-1)"
    )
    assert gate.work_changes({"a": 1.0}, {"a": 1.0, "b": 2.5}) == ["a same", "b none -> 2.5"]


def test_a_gain_is_reported_as_a_gain_to_keep_not_as_a_regression_or_layout():
    # compile_surface_d5 on 8f67cc3: -19.37%, its passes doing less.
    row = _with_work({**_WORK, "bytecodes": 400.0}, measured=806_300)
    assert gate.failure_kind(row) == "gain"
    assert gate.failure_reason(row) == (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain; "
        "Python work changed per repetition: bytecodes 449.6 -> 400 (-49.6), "
        "python_calls same, c_calls same"
    )
    assert gate.failure_reason(_verdict([806_300])) == (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain"
    )


def test_the_same_python_work_beyond_a_layout_shift_is_a_change_in_native_code():
    gain = _with_work(measured=806_300)
    assert gate.failure_kind(gain) == "gain"
    assert gate.failure_reason(gain) == (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain; "
        f"{gate.NATIVE_ONLY}"
    )
    worse = _with_work(measured=1_000_000 * (1 + gate.LAYOUT_SHIFT) + 1)
    assert gate.failure_kind(worse) == "regression"
    assert gate.failure_reason(worse).endswith(gate.NATIVE_ONLY)
    within = _with_work(measured=1_019_900)
    assert gate.failure_kind(within) == "layout only"


def test_the_failures_are_counted_by_kind():
    rows = [
        _with_work(),
        _with_work(measured=806_300),
        _verdict([806_300]),
        _verdict([1_005_000]),
        _verdict([1_000_000], sizes={"op_count": 16}),
    ]
    assert gate.failure_counts(rows) == "1 regression, 2 gains, 1 layout only"
    assert gate.failure_counts(rows[-1:]) == ""


def test_python_work_labels_only_a_moved_count_and_never_gates():
    assert _with_work(measured=1_000_000).verdict == "ok"
    assert _with_work({**_WORK, "bytecodes": 1.0}, measured=1_000_000).verdict == "ok"
    sizes = gate.compare(
        {"cases": {"c": {"op_count": 15, "python_work": _WORK}}}, {"c": {"op_count": 16}}, None
    )[0]
    assert gate.failure_reason(sizes) == "c: program sizes changed: op_count 15 -> 16"


def test_without_python_work_on_either_side_a_moved_count_is_not_labelled():
    row = _verdict([1_005_000])
    assert row.work_changes is None and row.work == ""
    assert gate.failure_reason(row) == "c: regression, +0.50% instructions (limit +0.4%)"
    baseline = _baseline()
    baseline["cases"]["c"]["python_work"] = _WORK
    runs = {"c": gate.Measured([1_005_000.0], [0.0])}
    unmeasured = gate.compare(baseline, {"c": {"op_count": 15}}, runs, None)[0]
    assert gate.failure_reason(unmeasured) == "c: regression, +0.50% instructions (limit +0.4%)"


def test_a_baseline_records_the_python_work_measured_with_its_counts():
    runs = {"c": gate.Measured([1_000_000.0], [2.0])}
    taken = gate.as_baseline({}, {"c": {"op_count": 15}}, runs, {}, {}, None, {"c": _WORK})
    assert taken["format"] == 2
    assert taken["cases"]["c"]["python_work"] == _WORK
    kept = gate.as_baseline({}, {"c": {"op_count": 15}}, None, taken, {})
    assert kept["cases"]["c"]["python_work"] == _WORK
    unlabelled = gate.as_baseline({}, {"c": {"op_count": 15}}, runs, taken, {}, None, None)
    assert "python_work" not in unlabelled["cases"]["c"]


def test_formats_1_and_2_are_read_and_others_are_errors():
    assert gate.baseline_error({"format": 1, **_baseline()}) is None
    assert gate.baseline_error({"format": 2, **_baseline()}) is None
    assert "format 3" in gate.baseline_error({"format": 3, **_baseline()})


# The fake-Valgrind harness: the gate's main() as CI runs it, with a fake
# valgrind (it writes one dump per case of instructions.json, of one
# repetition), a fake compiler for the marks, and a fake cases.py (its
# sizes, and the Python work of python_work.json, which varies from run to
# run while a file "varies" exists). In a repository with them, the counts
# and work are core/frontend's ir.py and work.py, so a base has its own.
_FAKE_VALGRIND = """\
import json, pathlib, sys
here = pathlib.Path(__file__).parent
if sys.argv[1:] == ["--version"]:
    print("valgrind-3.22.0 (fake)")
    raise SystemExit
out = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--callgrind-out-file="))
counts = json.loads((here / "instructions.json").read_text())
if pathlib.Path("core/frontend/ir.py").exists():
    counts = json.loads(pathlib.Path("core/frontend/ir.py").read_text())
for k, (name, ir) in enumerate(counts.items(), 1):
    pathlib.Path(f"{out}.{k}").write_text(
        "# callgrind format\\nversion: 1\\ncreator: callgrind-3.22.0\\n"
        f"desc: Trigger: Client Request: {name}\\n\\npositions: line\\nevents: Ir\\n"
        f"summary: {ir}\\n\\nfl=(1) a.c\\nfn=(1) f\\n1 {ir}\\n\\ntotals: {ir}\\n"
    )
print(json.dumps({name: 1 for name in counts}))
"""
_FAKE_CC = """\
import pathlib, sys
pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).touch()
"""
_FAKE_CASES = """\
import json, os, pathlib, sys
here = pathlib.Path(__file__).parent
if "--structural" in sys.argv:
    print(json.dumps({"sizes": {"c": {"op_count": 15}}, "tolerances": {}}))
elif "--python-work" in sys.argv:
    work = json.loads((here / "python_work.json").read_text())
    if pathlib.Path("core/frontend/work.py").exists():
        work = json.loads(pathlib.Path("core/frontend/work.py").read_text())
    if (here / "varies").exists():
        work["c"]["bytecodes"] += os.getpid()
    print(json.dumps(work))
"""


@pytest.fixture
def fake_gate(tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("valgrind", _FAKE_VALGRIND), ("cc", _FAKE_CC)):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n{text}")
        path.chmod(0o755)
    (tmp_path / "cases.py").write_text(_FAKE_CASES)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CC", str(bin_dir / "cc"))
    for name in ("GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_REPOSITORY", "GITHUB_REF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gate, "CASES", tmp_path / "cases.py")
    monkeypatch.setattr(gate, "_byte_compile", lambda *args: None)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gate.platform, "machine", lambda: "x86_64")
    environment = gate._environment(sys.executable)

    def run(instructions, work, baseline_work=_WORK, form=2, base_commit=None):
        case = {"op_count": 15, "instructions": 1_000_000}
        if baseline_work is not None:
            case["python_work"] = baseline_work
        baseline = tmp_path / "perf_gate.json"
        baseline.write_text(
            json.dumps({"format": form, "environment": environment, "cases": {"c": case}})
        )
        (bin_dir / "instructions.json").write_text(json.dumps({"c": instructions}))
        (tmp_path / "python_work.json").write_text(json.dumps({"c": work}))
        summary, out = tmp_path / "summary.md", tmp_path / "out.json"
        summary.unlink(missing_ok=True)
        argv = ["perf-gate.py", "--baseline", str(baseline), "--python", sys.executable]
        argv += ["--pythonpath", str(tmp_path), "--summary", str(summary), "--out", str(out)]
        argv += ["--base-commit", base_commit] if base_commit else []
        monkeypatch.setattr(sys, "argv", argv)
        code = gate.main()
        return code, capsys.readouterr().out, summary.read_text(), json.loads(out.read_text())

    run.varies = tmp_path / "varies"
    return run


def test_the_gate_labels_a_failure_whose_python_work_is_the_same(fake_gate):
    code, log, summary, _ = fake_gate(1_005_000, _WORK)
    assert code == 1
    line = f"c: layout only, +0.50% instructions (limit +0.4%); {gate.LAYOUT_ONLY}"
    assert f"  {line}\n" in log
    assert f"- {line}\n" in summary
    assert "| same | slower |" in summary
    assert "Perf gate failed (1 layout only):" in log
    assert "**Failed (1 layout only):**" in summary


def test_the_gate_shows_where_the_python_work_of_a_failure_changed(fake_gate):
    code, log, summary, _ = fake_gate(1_005_000, {**_WORK, "bytecodes": 452.6})
    assert code == 1
    line = (
        "c: regression, +0.50% instructions (limit +0.4%); Python work changed per "
        "repetition: bytecodes 449.6 -> 452.6 (+3), python_calls same, c_calls same"
    )
    assert f"  {line}\n" in log and f"- {line}\n" in summary
    assert gate.LAYOUT_ONLY not in log + summary
    assert "Perf gate failed (1 regression):" in log


def test_the_gate_reports_a_large_gain_as_a_gain_to_keep(fake_gate):
    code, log, summary, _ = fake_gate(806_300, {**_WORK, "bytecodes": 400.0})
    assert code == 1
    line = (
        "c: gain, -19.37% instructions (limit -1%): update the baseline to keep the gain; "
        "Python work changed per repetition: bytecodes 449.6 -> 400 (-49.6), "
        "python_calls same, c_calls same"
    )
    assert f"  {line}\n" in log and f"- {line}\n" in summary
    assert "Perf gate failed (1 gain):" in log and "**Failed (1 gain):**" in summary
    assert gate.LAYOUT_ONLY not in log + summary and "regression" not in log


def test_the_gate_fails_unlabelled_against_a_baseline_without_python_work(fake_gate):
    code, log, summary, taken = fake_gate(1_005_000, _WORK, baseline_work=None, form=1)
    assert code == 1
    assert "  c: regression, +0.50% instructions (limit +0.4%)\n" in log
    assert "- c: regression, +0.50% instructions (limit +0.4%)\n" in summary
    assert gate.LAYOUT_ONLY not in log + summary and "Python work changed" not in log
    assert "the baseline has no Python work" in log
    # The run taken as the next baseline brings it.
    assert taken["format"] == 2 and taken["cases"]["c"]["python_work"] == _WORK


def test_the_gate_passes_and_records_the_python_work_it_measured(fake_gate):
    code, log, summary, taken = fake_gate(1_000_000, _WORK)
    assert code == 0 and "Perf gate passed." in log
    assert taken["cases"]["c"]["python_work"] == _WORK


def test_python_work_that_differs_between_identical_runs_labels_nothing(fake_gate):
    fake_gate.varies.touch()
    code, log, summary, taken = fake_gate(1_005_000, _WORK)
    assert code == 1
    assert "two identical runs counted it differently (c)" in log
    assert "  c: regression, +0.50% instructions (limit +0.4%)\n" in log
    assert "python_work" not in taken["cases"]["c"]


def test_in_bootstrap_mode_a_case_without_a_count_is_reported_and_passes():
    baseline = {"bootstrap": True, "cases": {"c": {"op_count": 15}}}
    row = _verdict([1_000_000, 1_000_000], baseline)
    assert row.verdict == "bootstrap"
    assert row.measured == 1_000_000 and row.spread == 0
    assert "bootstrap" not in gate.FAILING


def test_in_bootstrap_mode_sizes_still_block():
    baseline = {"bootstrap": True, "cases": {"c": {"op_count": 15}}}
    assert _verdict([1_000_000], baseline, sizes={"op_count": 16}).verdict == "sizes"


def test_bootstrap_mode_is_allowed_only_while_no_case_has_a_count():
    assert gate.baseline_error({"bootstrap": True, "cases": {"c": {"op_count": 1}}}) is None
    assert gate.baseline_error(_baseline()) is None
    error = gate.baseline_error({"bootstrap": True, **_baseline()})
    assert error is not None and "remove" in error
    assert gate.baseline_error({"bootstrap": "yes", "cases": {}}) is not None


def test_bootstrap_mode_is_an_error_on_main():
    bootstrapping = {"bootstrap": True, "cases": {"c": {"op_count": 1}}}
    assert "main" in gate.baseline_error(bootstrapping, frozen=True)
    assert gate.baseline_error({"bootstrap": False, **_baseline()}, frozen=True) is not None
    assert gate.baseline_error(_baseline(), frozen=True) is None


def test_a_baseline_count_of_zero_is_an_error():
    assert "positive" in gate.baseline_error(_baseline(instructions=0))
    assert "positive" in gate.baseline_error(_baseline(instructions=None))


def test_taking_a_measured_run_as_the_baseline_ends_bootstrap_mode():
    previous = {"bootstrap": True, "environment": {}, "cases": {"c": {"op_count": 15}}}
    runs = {"c": gate.Measured([1_000_000.0, 1_000_000.0], [2.0, 2.0])}
    taken = gate.as_baseline({"python": "3.12.15"}, {"c": {"op_count": 15}}, runs, previous, {})
    assert "bootstrap" not in taken
    assert gate.baseline_error(taken) is None
    sizes_only = gate.as_baseline({}, {"c": {"op_count": 16}}, None, previous, {})
    assert sizes_only["bootstrap"] is True


BASELINE = json.loads(gate.BASELINE.read_text())


def test_the_checked_in_baseline_is_bootstrapping_or_has_every_count():
    assert gate.baseline_error(BASELINE) is None
    if BASELINE.get("bootstrap") is not True:
        missing = [n for n, c in BASELINE["cases"].items() if "instructions" not in c]
        assert not missing, f"no instruction count for {missing}"


def test_a_measured_baseline_is_of_the_locked_toolchain_and_names_its_run():
    # A new MLIR toolchain moves the counts: its pull request takes a new
    # baseline, and the old one stays in the file's history.
    if BASELINE.get("bootstrap") is True:
        pytest.skip("the baseline has no instruction counts yet")
    lock = json.loads((_ROOT / "tools" / "mlir-toolchain.lock.json").read_text())
    assert BASELINE["environment"]["mlir_toolchain"] == lock["release_tag"]
    assert {"repository", "run", "commit", "processes"} <= set(BASELINE["source"])


def test_a_case_of_its_own_tolerance_still_fails_a_half_percent_regression():
    # The baseline's tolerances are the cases' own (the size test below).
    own = {n: c["tolerance"] for n, c in BASELINE["cases"].items() if "tolerance" in c}
    assert "simulate_setup" in own
    for name, tolerance in own.items():
        assert gate.REGRESSION < tolerance < 0.005, name


def test_every_case_gates_at_least_one_program_size():
    sized = {
        name: [k for k in case if k not in gate.NOT_SIZES]
        for name, case in BASELINE["cases"].items()
    }
    assert not [name for name, keys in sized.items() if not keys]


def test_the_program_sizes_match_the_baseline(tmp_path):
    env = gate._run_env(str(Path(cudaq_pulse.__file__).parents[1]), str(tmp_path))
    # Under the sanitizers the extension needs their runtime preloaded.
    for name in ("LD_PRELOAD", "ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "LSAN_OPTIONS"):
        if name in os.environ:
            env[name] = os.environ[name]
    sizes, tolerances = gate._structural(sys.executable, env)
    assert gate.tolerance_error(BASELINE, tolerances) is None
    expected = {
        name: {k: v for k, v in case.items() if k not in gate.NOT_SIZES}
        for name, case in BASELINE["cases"].items()
    }
    assert sizes == expected, "update with: python tools/perf-gate.py --update"


def test_the_workload_op_counts_agree_with_compile_time_json():
    workloads = json.loads((_ROOT / "benchmarks" / "baselines" / "compile_time.json").read_text())
    for name, entry in workloads.items():
        assert BASELINE["cases"][f"compile_{name}"]["op_count"] == entry["op_count"], name


# Base comparison. The recorded rows of CI runs against the baseline of run
# 44179125553 (commit 933e24f5), for the cases they concern, and the same
# cases on each pull request's base: main's push run of its merge's first
# parent, which stands in for the base the job measures on its own build.
_AMP = {"bytecodes": 439.6, "python_calls": 16.0, "c_calls": 16.1}
_WAIT = {"bytecodes": 640.6, "python_calls": 22.0, "c_calls": 37.1}
_CTRL = {"bytecodes": 1339.0, "python_calls": 29.0, "c_calls": 74.667}
_SAMPLED = {"bytecodes": 5266.0, "python_calls": 40.0, "c_calls": 161.667}
_CTRL_107 = {"bytecodes": 1358.0, "python_calls": 29.0, "c_calls": 75.667}
_SAMPLED_107 = {"bytecodes": 5285.0, "python_calls": 40.0, "c_calls": 162.667}
_B0 = {
    "format": 2,
    "cases": {
        "specialize_amplitude": {"mlir_bytes": 1303, "instructions": 409188, "python_work": _AMP},
        "specialize_wait": {
            "schedule_cycles": 145,
            "schedule_events": 3,
            "instructions": 360385,
            "python_work": _WAIT,
        },
        "export_to_controller": {
            "export_bytes": 87794,
            "instructions": 4725299,
            "python_work": _CTRL,
        },
        "export_to_controller_sampled": {
            "export_bytes": 49449,
            "instructions": 1519377,
            "python_work": _SAMPLED,
        },
    },
}
# PR: its run's counts, its base's counts, and whether the export cases'
# Python work is the one #87 and #100 left (else B0's).
_RECORDED = {
    98: ((411144, 357800, 4734947, 1519008), (408506, 357780, 4736064, 1521018), True),
    63: ((411731, 356647, 4734665, 1517430), (408919, 360001, 4729579, 1518182), False),
    89: ((410644, 358430, 4734348, 1519899), (408019, 357589, 4734175, 1521673), True),
    94: ((406094, 356688, 4731508, 1519372), (408019, 357589, 4734175, 1521673), True),
    106: ((408019, 357410, 4698206, 1512776), (408019, 357589, 4734175, 1521673), True),
    107: ((407971, 357571, 4788958, 1529702), (408019, 357589, 4734175, 1521673), True),
    109: ((409555, 357687, 4739186, 1520159), (408019, 357589, 4734175, 1521673), True),
}
# What each native change touches, as base_scope() says it.
_TOUCHES = {
    106: "core/mlir/include/cudaq-pulse/Analysis/LineTrace.h",
    107: "core/mlir/bindings/ControllerProgram.h, core/mlir/bindings/bindings.cpp",
    109: "core/mlir/bindings/ControllerProgram.h, core/mlir/bindings/bindings.cpp, "
    "core/mlir/conversions/PulseToQOp/PulseToQOp.cpp and 10 more",
}


def _recorded(pr, scope="", amplitude=None, wait=None, amp_work=_AMP, base_work=None):
    """The failures and notices of a recorded run's rows, compared with its
    base (scope "") or not (None); *amplitude* and *wait* replace its
    counts, as fractions of the base's."""
    head, base, export_moved = _RECORDED[pr]
    names = list(_B0["cases"])
    counts = list(head)
    if amplitude:
        counts[0] = base[0] * amplitude
    if wait:
        counts[1] = base[1] * wait
    work = [amp_work, _WAIT, *((_CTRL_107, _SAMPLED_107) if export_moved else (_CTRL, _SAMPLED))]
    sizes = {
        n: {k: v for k, v in c.items() if k not in gate.NOT_SIZES} for n, c in _B0["cases"].items()
    }
    runs = {n: gate.Measured([float(v)], [0.0]) for n, v in zip(names, counts, strict=True)}
    on_base = {
        n: (float(b), (base_work or {}).get(n, w))
        for n, b, w in zip(names, base, [_AMP, *work[1:]], strict=True)
    }
    rows = gate.compare(
        _B0,
        sizes,
        runs,
        dict(zip(names, work, strict=True)),
        on_base if scope == "" else None,
        scope,
    )
    failures = [gate.failure_reason(r) for r in rows if r.verdict in gate.FAILING]
    return failures, gate.layout_notes(rows)


_KEPT = "passes, not a gain to keep; calls not compared"
_FAILS = "fails, the owner's to accept or not; same Python work; calls not compared"


@pytest.mark.parametrize(
    ("pr", "failures", "notes"),
    [
        # Judged against the base at the same limit, +0.4% above it: #98's
        # and #63's amplitude rows, which passed within the 0.95% either way
        # allowed before, fail; any lower count passes.
        pytest.param(
            98, ["specialize_amplitude: layout only, +0.65% from its base measured on this build "
                 "(limit +0.4%): " + _FAILS], [],
            id="#98 passed within 0.95%, fails 0.65% above its base"),
        pytest.param(
            63, ["specialize_amplitude: layout only, +0.69% from its base measured on this build "
                 "(limit +0.4%): " + _FAILS],
                [f"specialize_wait: layout, -0.93% from its base measured on this build (limit "
                 f"+0.4%): {_KEPT}"],
            id="#63 passed within 0.95%, fails 0.69% above its base"),
        pytest.param(
            94, [], [f"specialize_wait: layout, -0.25% from its base measured on this build (limit "
                     f"+0.4%): {_KEPT}"],
            id="#94 passes below its base"),
        pytest.param(89, [], [], id="#89 had no failing row"),
    ],
)  # fmt: skip
def test_the_recorded_layout_shifts_are_judged_against_their_base(pr, failures, notes):
    assert _recorded(pr) == (failures, notes)
    # Without a base, the limits fail every one.
    today, _ = _recorded(pr, scope=None)
    assert len(today) == len(failures) + len(notes)
    assert all(line.split(": ")[1].startswith("layout only,") for line in today)


@pytest.mark.parametrize("pr", [98, 63, 94])
@pytest.mark.parametrize(
    "scope",
    [
        "the change touches measured code: core/mlir/transforms/PulseVerify.cpp",
        "git cannot compare this tree with its base 880c4002",
        "the base 880c4002 is not an ancestor of this tree",
    ],
)
def test_a_recorded_layout_shift_fails_when_its_base_cannot_judge_it(pr, scope):
    failures, notes = _recorded(pr, scope)
    assert failures and not notes
    for line in failures:
        assert line.endswith(
            f"; {gate.LAYOUT_ONLY}; judged against the stored baseline: no base built in this "
            f"job ({scope})"
        )


@pytest.mark.parametrize("pr", [106, 107, 109])
def test_native_changes_are_judged_by_the_limits(pr):
    scope = f"the change touches measured code: {_TOUCHES[pr]}"
    failures, notes = _recorded(pr, scope)
    assert not notes
    why = f"; judged against the stored baseline: no base built in this job ({scope})"
    assert [line.removesuffix(why) for line in failures] == _recorded(pr, scope=None)[0]
    if pr == 107:
        assert failures == [
            "export_to_controller: regression, +1.35% instructions (limit +0.4%); Python work "
            "changed per repetition: bytecodes 1,339 -> 1,358 (+19), python_calls same, "
            "c_calls 74.667 -> 75.667 (+1)" + why,
            "export_to_controller_sampled: regression, +0.68% instructions (limit +0.4%); "
            "Python work changed per repetition: bytecodes 5,266 -> 5,285 (+19), "
            "python_calls same, c_calls 161.667 -> 162.667 (+1)" + why,
        ]
    else:
        assert failures == []


@pytest.mark.parametrize("amplitude", [1.01, 1.0096, None])
def test_a_layout_only_row_more_than_the_limit_above_its_base_fails(amplitude):
    # #98's amplitude row, +0.65% above its base, and a real 1% slowdown
    # behind the same Python work: both more than 0.4% above the base.
    failures, notes = _recorded(98, amplitude=amplitude)
    own = {1.01: "+1.00%", 1.0096: "+0.96%", None: "+0.65%"}[amplitude]
    assert notes == [] and failures == [
        f"specialize_amplitude: layout only, {own} from its base measured on this build "
        "(limit +0.4%): " + _FAILS
    ]


def test_a_layout_only_row_below_its_base_passes_and_is_not_a_gain():
    # #98's wait row made 1% faster than its base: past the gain limit
    # against the baseline, it passes as layout and is not kept.
    failures, notes = _recorded(98, wait=0.99)
    assert notes == [
        "specialize_wait: layout, -1.00% from its base measured on this build (limit +0.4%): "
        + _KEPT
    ]
    assert [line.split(":")[0] for line in failures] == ["specialize_amplitude"]


def _above(own):
    """#98 on a base 0.3% above the baseline, the amplitude row *own* above
    its base."""
    head, base, moved = _RECORDED[98]
    _RECORDED[99] = (head, (409188 * 1.003, *base[1:]), moved)
    try:
        return _recorded(99, amplitude=1 + own)
    finally:
        del _RECORDED[99]


def test_hidden_native_work_passes_only_below_the_limit_above_its_base():
    # The hole that remains: native work behind the same calls passes up to
    # 0.4% above the base; +0.9%, which passed within the 0.95% before, fails.
    failures, notes = _above(0.009)
    assert notes == [] and failures[0].startswith(
        "specialize_amplitude: layout only, +0.90% from its base"
    )
    failures, notes = _above(0.003)
    assert failures == [] and notes == [
        f"specialize_amplitude: layout, +0.30% from its base measured on this build (limit +0.4%): "
        f"{_KEPT}"
    ]


def test_changed_python_work_keeps_the_limits():
    failures, notes = _recorded(98, amp_work={**_AMP, "bytecodes": 442.6})
    assert not notes
    assert failures == [
        "specialize_amplitude: regression, +0.48% instructions (limit +0.4%); Python work "
        "changed per repetition: bytecodes 439.6 -> 442.6 (+3), python_calls same, c_calls same; "
        "judged against the stored baseline: its base ran other Python work"
    ]


def test_the_base_must_run_the_same_python_work():
    other = {"specialize_amplitude": {**_AMP, "bytecodes": 436.6}}
    failures, notes = _recorded(98, base_work=other)
    assert not notes
    assert failures == [
        "specialize_amplitude: layout only, +0.48% instructions (limit +0.4%); "
        f"{gate.LAYOUT_ONLY}; judged against the stored baseline: its base ran other Python work"
    ]


def test_python_work_merged_after_the_baseline_is_judged_on_the_base():
    # Main changed the case's Python work after the baseline was taken; the
    # change ran its base's, and its count moved by layout.
    merged = {**_AMP, "bytecodes": 442.6}
    on_base = {"specialize_amplitude": merged}
    head, base, moved = _RECORDED[98]
    _RECORDED[99] = (head, (409188 * 1.003, *base[1:]), moved)
    try:
        failures, notes = _recorded(99, amplitude=1.003, amp_work=merged, base_work=on_base)
        assert failures == [] and notes == [
            "specialize_amplitude: layout, +0.30% from its base measured on this build (limit "
            f"+0.4%): {_KEPT}"
        ]
        # More than the limit above its base it fails as layout only.
        failures, notes = _recorded(99, amplitude=1.01, amp_work=merged, base_work=on_base)
    finally:
        del _RECORDED[99]
    assert not notes
    assert failures == [
        "specialize_amplitude: layout only, +1.00% from its base measured on this build "
        "(limit +0.4%): " + _FAILS
    ]
    # Without its base it keeps the limits: a regression with its work.
    failures, _ = _recorded(98, scope=None, amp_work=merged)
    assert failures == [
        "specialize_amplitude: regression, +0.48% instructions (limit +0.4%); Python work "
        "changed per repetition: bytecodes 439.6 -> 442.6 (+3), python_calls same, c_calls same"
    ]


def test_a_change_of_python_work_merged_after_the_baseline_and_its_own_fails():
    # Main changed the work, and the change changed it again.
    merged = {**_AMP, "bytecodes": 442.6}
    failures, notes = _recorded(
        98, amp_work={**merged, "bytecodes": 445.6}, base_work={"specialize_amplitude": merged}
    )
    assert not notes and len(failures) == 1
    assert failures[0].startswith("specialize_amplitude: regression, +0.48% instructions")


def test_a_row_beyond_the_layout_shift_from_the_baseline_fails():
    # Main moved +2% and the change +0.5% more: the change is in native code.
    head, base, moved = _RECORDED[98]
    _RECORDED[99] = (head, (409188 * 1.02, *base[1:]), moved)
    try:
        failures, notes = _recorded(99, amplitude=1.005)
    finally:
        del _RECORDED[99]
    # Judged against its base, which ran the same work: +0.50% above it.
    assert not notes and len(failures) == 1
    assert failures[0].startswith(
        "specialize_amplitude: layout only, +0.50% from its base measured on this build"
    )


def _git(repo, *args):
    import subprocess

    command = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    command += ["-c", "commit.gpgsign=false", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


_PYPROJECT = """\
[build-system]
requires = ["scikit-build-core"]
[dependency-groups]
build = ["numpy"]
test = ["pytest"]
"""
_UV_LOCK = """\
version = 1
[[package]]
name = "cudaq-pulse"
source = {{ virtual = "." }}
dependencies = [{{ name = "numpy" }}]
[package.dev-dependencies]
build = [{{ name = "numpy" }}]
test = [{{ name = "pytest" }}]
[[package]]
name = "numpy"
version = "{numpy}"
[[package]]
name = "pytest"
version = "{pytest}"
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository whose first commit is a change's base; commit(**files)
    adds a commit on it (None removes a file) and returns its id."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")

    def commit(**files):
        for name, text in files.items():
            path = root / name.replace("__", "/")
            if text is None:
                _git(root, "rm", "-q", str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "--allow-empty", "-m", "c")
        return _git(root, "rev-parse", "HEAD")

    commit.root = root
    commit.base = commit(
        **{
            "core__frontend__cudaq_pulse__k.py": "x = 1\n",
            "core__mlir__lib__Foo.cpp": "int f();\n",
            "pyproject.toml": _PYPROJECT,
            "uv.lock": _UV_LOCK.format(numpy="2.0", pytest="8.0"),
            "benchmarks__baselines__perf_gate.json": "{}\n",
        }
    )
    # The gate's git runs in it; the environment is still this checkout's.
    environment = gate._environment(sys.executable)
    monkeypatch.setattr(gate, "_environment", lambda python: environment)
    monkeypatch.setattr(gate, "ROOT", root)
    return commit


def test_python_sources_and_files_the_job_never_reads_keep_the_measured_code(repo):
    repo(
        **{
            "core__frontend__cudaq_pulse__k.py": "x = 2\n",
            "tests__test_k.py": "",
            "docs__k.rst": "",
            "CHANGELOG.md": "",
            ".github__workflows__docs.yml": "",
            "tools__check-k.py": "",
            "tools__ci-changes.py": "",
            "challenges__k__score.py": "",
            "benchmarks__baselines__perf_gate.json": '{"a": 1}\n',
            # Dev dependencies, tool settings and the sdist's contents are not built.
            "uv.lock": _UV_LOCK.format(numpy="2.0", pytest="9.0"),
            "pyproject.toml": _PYPROJECT
            + "[tool.ruff]\nline-length = 99\n"
            + '[tool.scikit-build]\nsdist.exclude = ["/challenges"]\n',
        }
    )
    assert gate.base_scope(repo.base) == ""


@pytest.mark.parametrize(
    ("files", "paths"),
    [
        ({"core__mlir__lib__Foo.cpp": "int g();\n"}, "core/mlir/lib/Foo.cpp"),
        ({".github__workflows__ci.yml": ""}, ".github/workflows/ci.yml"),
        ({"tools__perf-gate.py": ""}, "tools/perf-gate.py"),
        ({"benchmarks__perf_gate__cases.py": ""}, "benchmarks/perf_gate/cases.py"),
        ({"core__frontend__cudaq_pulse__data.json": ""}, "core/frontend/cudaq_pulse/data.json"),
        ({"uv.lock": _UV_LOCK.format(numpy="2.1", pytest="8.0")}, "uv.lock"),
        ({"pyproject.toml": _PYPROJECT.replace('"numpy"', '"numpy", "cmake"')}, "pyproject.toml"),
        (
            {"pyproject.toml": _PYPROJECT + '[tool.scikit-build]\ncmake.build-type = "Debug"\n'},
            "pyproject.toml",
        ),
        # A rename into tests/ lists both paths.
        (
            {"core__mlir__lib__Foo.cpp": None, "tests__Foo.cpp": "int f();\n"},
            "core/mlir/lib/Foo.cpp",
        ),
    ],
)
def test_any_other_path_is_measured_code(repo, files, paths):
    repo(**files)
    assert gate.base_scope(repo.base) == f"the change touches measured code: {paths}"


def test_every_entry_of_a_package_locked_twice_is_measured(repo):
    # A lock split by Python version locks numpy twice; either entry counts.
    def lock(old):
        entry = f'[[package]]\nname = "numpy"\nversion = "{old}"\n'
        entry += "resolution-markers = [\"python_full_version < '3.11'\"]\n"
        text = _UV_LOCK.format(numpy="2.0", pytest="8.0")
        return text.replace('[[package]]\nname = "numpy"', entry + '[[package]]\nname = "numpy"')

    split = repo(**{"uv.lock": lock("1.26")})
    repo(**{"uv.lock": lock("1.25")})
    assert gate.base_scope(split) == "the change touches measured code: uv.lock"


def test_the_base_must_be_an_ancestor_of_the_tree_and_git_must_tell(repo, tmp_path, monkeypatch):
    # Any ancestor: a branch's merge base with main, several commits back.
    parent = repo(**{"tests__a.py": ""})
    repo(**{"tests__b.py": ""})
    assert gate.base_scope(repo.base) == ""
    assert gate.base_scope(parent) == ""
    # Not an ancestor: a commit on another branch.
    side = _git(repo.root, "commit-tree", f"{parent}^{{tree}}", "-m", "side").strip()
    assert gate.base_scope(side) == f"the base {side[:8]} is not an ancestor of this tree"
    missing = "0123456789abcdef0123456789abcdef01234567"
    assert gate.base_scope(missing) == "git cannot find the base 01234567"
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    assert gate.base_scope(parent).startswith("git cannot find the base")


def _change(repo, base_ir, head_ir, base_work=_WORK, **files):
    """A change on a base whose Python sources measure *base_ir*, to ones
    that measure *head_ir* (the fake valgrind reads core/frontend/ir.py)."""
    work = json.dumps({"c": base_work})
    base = repo(
        **{"core__frontend__ir.py": json.dumps({"c": base_ir}), "core__frontend__work.py": work}
    )
    repo(
        **{
            "core__frontend__ir.py": json.dumps({"c": head_ir}),
            "core__frontend__work.py": json.dumps({"c": _WORK}),
        },
        **files,
    )
    return base


def test_the_gate_measures_the_base_and_passes_a_layout_shift_of_the_change(fake_gate, repo):
    base = _change(repo, 1_001_000, 1_005_000)
    head = (repo.root / "core/frontend/ir.py").read_text()
    code, log, summary, out = fake_gate(1_005_000, _WORK, base_commit=base)
    note = (
        f"c: layout, +0.40% from its base {base[:8]} measured on this build (limit +0.4%): passes, "
        "not a gain to keep; no call differs"
    )
    assert code == 0 and log.endswith("Perf gate passed (1 layout).\n")
    assert f"Layout:\n  {note}\n  ({gate.AS_BASE})" in log
    assert f"**Layout (1).** {note}. In each, {gate.AS_BASE}" in summary
    assert "| same | layout |" in summary
    # The tree's sources are back, and the artifact holds the tree's count.
    assert (repo.root / "core/frontend/ir.py").read_text() == head
    assert _git(repo.root, "status", "--porcelain") == ""
    assert out["cases"]["c"]["instructions"] == 1_005_000


def test_the_gate_fails_a_change_that_moved_its_count_beyond_the_own_shift(fake_gate, repo):
    base = _change(repo, 995_000, 1_005_000)
    code, log, _, _ = fake_gate(1_005_000, _WORK, base_commit=base)
    assert code == 1 and "Perf gate failed (1 layout only):" in log
    assert (
        f"c: layout only, +1.01% from its base {base[:8]} measured on this build (limit +0.4%): "
        "fails, the owner's to accept or not; same Python work; no call differs"
    ) in log


def test_the_gate_fails_when_the_base_ran_other_python_work(fake_gate, repo):
    base = _change(repo, 1_001_000, 1_005_000, base_work={**_WORK, "bytecodes": 450.6})
    code, log, _, _ = fake_gate(1_005_000, _WORK, base_commit=base)
    assert code == 1 and (
        f"; judged against the stored baseline: its base {base[:8]} ran other Python work" in log
    )


def test_the_gate_measures_the_base_when_main_changed_the_python_work(fake_gate, repo):
    # The baseline predates a merge that changed the case's work; the base
    # and the change run the same work, and the change moved by layout.
    base = _change(repo, 1_001_000, 1_005_000)
    older = {**_WORK, "bytecodes": 440.6}
    code, log, _, _ = fake_gate(1_005_000, _WORK, baseline_work=older, base_commit=base)
    assert code == 0 and log.endswith("Perf gate passed (1 layout).\n")
    # Without the base, the changed work is a regression.
    code, log, _, _ = fake_gate(1_005_000, _WORK, baseline_work=older)
    assert code == 1 and "c: regression, +0.50% instructions" in log


def test_a_baseline_without_python_work_is_not_compared_with_the_base(fake_gate, repo, monkeypatch):
    base = _change(repo, 1_001_000, 1_005_000)
    monkeypatch.setattr(gate, "measure_base", None)
    code, log, _, _ = fake_gate(1_005_000, _WORK, baseline_work=None, form=1, base_commit=base)
    assert code == 1 and "c: regression, +0.50% instructions" in log


def test_a_gain_from_python_work_the_baseline_lacks_is_kept(fake_gate, repo, monkeypatch):
    # Main changed the work since the baseline; a gain on top of it is
    # kept by a new baseline, not passed as layout.
    base = _change(repo, 990_000, 985_000)
    monkeypatch.setattr(gate, "measure_base", None)
    older = {**_WORK, "bytecodes": 440.6}
    code, log, _, _ = fake_gate(985_000, _WORK, baseline_work=older, base_commit=base)
    assert code == 1 and "c: gain, -1.50% instructions" in log


def test_the_gate_does_not_measure_the_base_of_a_change_to_measured_code(fake_gate, repo):
    base = _change(repo, 1_001_000, 1_005_000, **{"core__mlir__lib__Foo.cpp": "int g();\n"})
    code, log, _, _ = fake_gate(1_005_000, _WORK, base_commit=base)
    assert code == 1 and "Perf gate failed (1 layout only):" in log
    assert (
        "; judged against the stored baseline: no base built in this job (the change touches "
        "measured code: core/mlir/lib/Foo.cpp" in log
    )


def test_the_gate_measures_no_base_without_a_row_it_could_pass(fake_gate, repo, monkeypatch):
    base = _change(repo, 1_001_000, 1_002_000)
    monkeypatch.setattr(gate, "measure_base", None)
    code, log, _, _ = fake_gate(1_002_000, _WORK, base_commit=base)
    assert code == 0 and log.endswith("Perf gate passed.\n")


# Native base comparison: the calls each native function received, and the
# classes of functions whose calls follow layout rather than the program.

_CALLS_DUMP = """\
version: 1
creator: callgrind-3.22.0
events: Ir
desc: Trigger: Client Request: case
ob=(1) /usr/lib/x86_64-linux-gnu/libc.so.6
ob=(2) /x/_cudaq_pulse_native.so
fn=(1) build
1 5
cob=(1)
cfn=(2) malloc
calls=3 0
1 30
cfn=(3) helper
calls=2 0
2 20
fn=(3)
cfn=(4) helper'2
calls=4 0
1 8
totals: 63
"""


def test_the_calls_to_each_function_are_counted_with_its_object():
    calls = gate.dump_calls(_CALLS_DUMP)
    # A cob= names the callee's object for one call; a recursion level's
    # 'n counts with its function; a call without cob= stays in its caller's.
    assert calls == {
        ("libc.so.6", "malloc"): 3,
        ("_cudaq_pulse_native.so", "helper"): 6,
    }


@pytest.mark.parametrize(
    ("function", "label"),
    [
        (
            (
                "_cudaq_pulse_native.so",
                "bool llvm::function_ref<bool (mlir::StorageUniquer::BaseStorage const*)>"
                "::callback_fn<mlir::StorageUniquer::get<mlir::detail::FloatAttrStorage>>",
            ),
            "container probes",
        ),
        (
            ("_cudaq_pulse_native.so", "llvm::detail::IEEEFloat::bitwiseIsEqual(x) const"),
            "container probes",
        ),
        (("python3.12", "assign_version_tag"), "interpreter"),
        (("libpython3.12.so.1.0", "set_add"), "interpreter"),
        (("libc.so.6", "0x00000000000a38d0"), "libc, unnamed (allocator or memory copies)"),
        (("libstdc++.so.6.0.33", "operator new(unsigned long)"), "allocation"),
        (("libstdc++.so.6.0.33", "operator delete(void*, std::align_val_t)"), "allocation"),
        (
            ("_cudaq_pulse_native.so", "llvm::allocate_buffer(unsigned long, unsigned long)"),
            "allocation",
        ),
        (
            (
                "_cudaq_pulse_native.so",
                "(anonymous namespace)::Shape::isEqual(mlir::Operation const*)",
            ),
            "container probes",
        ),
        # A tree's erase recurses down its shape, but is the program's call.
        (
            (
                "_cudaq_pulse_native.so",
                "std::_Rb_tree<std::pair<mlir::Operation*, long>, x>::_M_erase(x)",
            ),
            None,
        ),
        (
            (
                "_cudaq_pulse_native.so",
                "std::common_comparison_category<x>::type std::operator<=><mlir::Operation*, x>(x)",
            ),
            "container probes",
        ),
        (("libc.so.6", "malloc"), "allocation"),
        (("libc.so.6", "_int_free_merge_chunk"), "allocation"),
        (("libc.so.6", "__memcpy_avx_unaligned_erms"), "memory copies"),
        (("libc.so.6", "__memcmp_avx2_movbe"), "memory copies"),
        # libc's other functions, and the loader's, are compared.
        (("libc.so.6", "strlen"), None),
        (("libc.so.6", "clock_gettime"), None),
        (("ld-linux-x86-64.so.2", "__tls_get_addr"), None),
        # A tree's inserts are the program's work, not its shape.
        (
            (
                "_cudaq_pulse_native.so",
                "std::_Rb_tree<std::pair<mlir::Operation*, long>, x>::_M_insert_(x)",
            ),
            None,
        ),
        (("???", "0x0000000004972620"), "unmapped (interpreter or kernel)"),
        (("???", "_PyObject_LookupAttr.warm"), "unmapped (interpreter or kernel)"),
        # The uniquer itself, and the program's own functions, are compared.
        (("_cudaq_pulse_native.so", "mlir::StorageUniquer::getParametricStorageTypeImpl"), None),
        (("_cudaq_pulse_native.so", "mlir::ModuleOp mlir::Operation::getParentOfType()"), None),
        (("_cudaq_pulse_native.so", "0x0000000000123456"), None),
    ],
)
def test_the_layout_class_of_a_function(function, label):
    assert gate.layout_class(function) == label


_PAIRS = json.loads((Path(__file__).parent / "perf_gate_call_pairs.json").read_text())


def _judge_pair(label):
    """The rows of one recorded pair, each case made 1% slower than a
    baseline with the tree's Python work, judged against its base by calls."""
    cases = _PAIRS[label]
    baseline = {
        "format": 2,
        "cases": {
            n: {
                "instructions": c["tree"]["instructions"] / 1.01,
                "python_work": c["tree"]["python_work"],
            }
            for n, c in cases.items()
        },
    }
    instructions = {
        n: gate.Measured([float(c["tree"]["instructions"])], [0.0]) for n, c in cases.items()
    }
    work = {n: c["tree"]["python_work"] for n, c in cases.items()}
    native, tree_calls = {}, {}
    for n, c in cases.items():
        base_calls = {(o, f): b for o, f, b, _ in c["calls"]}
        tree_calls[n] = {(o, f): t for o, f, _, t in c["calls"]}
        native[n] = (float(c["base"]["instructions"]), c["base"]["python_work"], base_calls)
    scope = "the change touches measured code: core/mlir/x.cpp"
    rows = gate.compare(
        baseline, {n: {} for n in cases}, instructions, work, None, scope, native, tree_calls
    )
    return {r.name: r for r in rows}


@pytest.mark.parametrize(
    "label",
    [
        "identical code: 813a2b59 / eb6e6619",
        "a comment in a Python source: eb85adf6 / + comment",
        "CI's runner, a comment in native code: run 45077423614",
    ],
)
def test_a_layout_pair_passes_every_row_by_its_calls(label):
    rows = _judge_pair(label)
    assert len(rows) == 20
    # On the box's builds, compile_calls_small erases a std::map ordered by
    # operation addresses (ScheduleAlap's forwardStarts), whose tree, and so
    # its erase's recursion, takes another shape in another layout: the one
    # row a layout pair fails there.
    expected = set() if label.startswith("CI's runner") else {"compile_calls_small"}
    assert {n for n, r in rows.items() if r.verdict != "layout"} == expected
    # The notice lists each function whose calls differed, with its class.
    notes = gate.layout_notes(list(rows.values()))
    assert len(notes) == 20 - len(expected)
    assert any("container probes " in line for line in notes)
    assert any("allocation " in line for line in notes)
    if not expected:
        assert any("memory copies " in line for line in notes)


def test_a_native_change_fails_where_its_code_runs_and_passes_where_it_does_not():
    rows = _judge_pair("a native change, #163: eb6e6619 / eb85adf6")
    passed = {n for n, r in rows.items() if r.verdict == "layout"}
    # #163 adds controller facts: what binds, exports or submits a compiled
    # program does not reach them. Its export_to_emulator row, layout only,
    # is more than 0.4% above its base and fails.
    assert passed == {
        "bind_cached",
        "bind_new_value",
        "export_to_controller",
        "export_to_controller_sampled",
        "submit",
    }
    assert gate.failure_reason(rows["export_to_emulator"]).startswith(
        "export_to_emulator: layout only, "
    )
    reason = gate.failure_reason(rows["specialize_wait"])
    assert "calls differ outside the layout classes on its base" in reason
    assert "getParentOfType" in reason


def test_a_change_to_the_gate_or_the_pins_is_not_judged_against_a_built_base(repo):
    repo(**{"core__mlir__lib__Foo.cpp": "int g();\n"})
    assert gate.native_base_scope(repo.base) == ""
    repo(**{"tools__perf-gate.py": "", "core__mlir__lib__Foo.cpp": "int h();\n"})
    assert gate.native_base_scope(repo.base) == "the change touches tools/perf-gate.py"


# Rule 441, replayed on CI's own dumps: four rows of three runs, each with
# its base's count, Python work and every call that differed from it.
_RULE = json.loads((Path(__file__).parent / "perf_gate_layout_rule.json").read_text())


def _judge_rule(label, case, scope="the change touches measured code: core/mlir/x.cpp", built=True):
    """One recorded row, failing against the baseline its run was judged
    against, judged against its base built in the job (*built*), or with
    no base, when *scope* says why."""
    c = _RULE[label][case]
    # The baseline the run was judged against, with its base's work.
    baseline = {
        "format": 2,
        "cases": {
            case: {
                "instructions": c["baseline"]["instructions"],
                "python_work": c["base"]["python_work"],
            }
        },
    }
    base_calls = {(o, f): b for o, f, b, _ in c["calls"]}
    tree_calls = {(o, f): t for o, f, _, t in c["calls"]}
    native = {case: (float(c["base"]["instructions"]), c["base"]["python_work"], base_calls)}
    rows = gate.compare(
        baseline,
        {case: {}},
        {case: gate.Measured([float(c["tree"]["instructions"])], [0.0])},
        {case: c["tree"]["python_work"]},
        None,
        scope,
        native if built else None,
        {case: tree_calls},
        None,
        "base1234",
    )
    return rows[0]


def test_a_layout_only_row_more_than_the_limit_above_its_base_built_in_the_job_fails():
    # rewrite-work-2's export_to_emulator: the same Python work, and only
    # layout classes' calls differ (the interpreter's type cache missed).
    row = _judge_rule("rewrite-work-2 on 8aedaf87: run 45505053324", "export_to_emulator")
    assert gate.failure_kind(row) == "layout only" and row.verdict == "slower"
    assert gate.failure_reason(row) == (
        "export_to_emulator: layout only, +0.97% from its base base1234 built in this job "
        "(limit +0.4%): fails, the owner's to accept or not; same Python work; calls differ "
        "only in layout classes "
        "(allocation 7, container probes 1, interpreter 1, memory copies 1, "
        "unmapped (interpreter or kernel) 13)"
    )


def test_a_layout_only_row_within_the_limit_above_its_base_passes():
    # The shared-terms branch's first_compile: +0.42% against the stored
    # baseline, +0.18% against its base, only unattributed calls differ.
    row = _judge_rule("shared terms on 1a4bcdaa: run 45499530056", "first_compile")
    assert row.delta == pytest.approx(0.0042, abs=5e-5) and row.verdict == "layout"
    assert gate.layout_notes([row]) == [
        "first_compile: layout, +0.18% from its base base1234 built in this job (limit +0.4%): "
        "passes, not a gain to keep; calls differ only in layout classes "
        "(unmapped (interpreter or kernel) 16)"
    ]


@pytest.mark.parametrize(
    ("label", "case", "outside"),
    [
        ("shared terms on 1a4bcdaa: run 45499530056", "export_to_controller_detuned", 13),
        ("#263 on its base built in the job: run 45515786834", "specialize_wait", 2),
    ],
)
def test_a_row_whose_own_calls_differ_is_judged_against_the_stored_baseline(label, case, outside):
    # Its change calls its own code: not layout only, whatever its count did
    # against its base (+0.65% and +0.21%).
    row = _judge_rule(label, case)
    assert row.on_base is False and row.verdict == "slower"
    assert gate.failure_kind(row) == "regression"
    reason = gate.failure_reason(row)
    assert "; judged against the stored baseline: calls differ outside the layout classes " in (
        reason
    )
    assert reason.count(" -> ") == outside
    if case == "specialize_wait":
        assert "pulse::callsVerified() (0 -> 20)" in reason
        assert "pulse::refuseCalls(mlir::func::FuncOp) (0 -> 20)" in reason


@pytest.mark.parametrize(
    "scope",
    [
        "the change touches measured code: core/mlir/x.cpp; and not against its base built "
        "here: the change touches tools/perf-gate.py",
        "the change touches measured code: core/mlir/x.cpp; and not against its base built "
        "here: the base could not be built: cmake failed",
    ],
)
def test_a_run_with_no_base_judges_every_row_against_the_stored_baseline(scope):
    # A change to the gate (refused scope) or a base that did not build.
    row = _judge_rule(
        "rewrite-work-2 on 8aedaf87: run 45505053324", "export_to_emulator", scope, built=False
    )
    assert row.on_base is None and row.verdict == "slower"
    assert gate.failure_reason(row) == (
        "export_to_emulator: layout only, +0.86% instructions (limit +0.4%); "
        f"{gate.LAYOUT_ONLY}; judged against the stored baseline: no base built in this job "
        f"({scope})"
    )


def test_ci_dispatches_a_branch_with_its_merge_base_and_a_baseline_run_with_none():
    job = (_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    step = job[job.index("- name: Perf gate") :]
    step = step[: step.index("\n\n")]
    assert 'base="$(git merge-base HEAD origin/main)"' in step
    assert '[ "$GITHUB_EVENT_NAME" = workflow_dispatch ]' in step
    assert '[ "$PERF_BASELINE" != true ]' in step
    assert "PERF_BASELINE: ${{ inputs.perf_baseline }}" in step


def test_a_python_only_change_whose_own_calls_differ_is_judged_against_the_baseline():
    # The base's Python sources on this build ran the same Python work, but
    # a function outside the layout classes was called more often.
    baseline = {"format": 2, "cases": {"c": {"instructions": 1_000_000, "python_work": _WORK}}}
    rows = gate.compare(
        baseline,
        {"c": {}},
        {"c": gate.Measured([1_005_000.0], [0.0])},
        {"c": _WORK},
        {"c": (1_002_000.0, _WORK)},
        "",
        None,
        {"c": {("lib.so", "f()"): 3, ("libc.so.6", "malloc"): 9}},
        {"c": {("lib.so", "f()"): 2, ("libc.so.6", "malloc"): 8}},
        "base1234",
    )
    row = rows[0]
    assert row.on_base is False and row.verdict == "slower"
    assert gate.failure_reason(row).endswith(
        "; judged against the stored baseline: calls differ outside the layout classes on its "
        "base base1234: lib.so:f() (2 -> 3)"
    )


def _dump(attr_cost, malloc_cost):
    """A Callgrind dump in which a caller calls PyObject_GenericGetAttr 3
    times and it calls malloc twice; each costs itself what is given."""
    return f"""version: 1
positions: line
events: Ir
ob=(1) /usr/bin/python3.12
fl=(1) x.c
fn=(1) caller
1 10
cfn=(2) PyObject_GenericGetAttr
calls=3 0
1 {attr_cost + malloc_cost}
fn=(2)
1 {attr_cost}
cob=(2) /usr/lib/x86_64-linux-gnu/libc.so.6
cfn=(3) malloc
calls=2 0
1 {malloc_cost}
ob=(2)
fn=(3)
1 {malloc_cost}
"""


def test_the_costs_of_a_dump_are_each_functions_own():
    assert gate.dump_costs(_dump(300, 50)) == {
        ("python3.12", "caller"): 10,
        ("python3.12", "PyObject_GenericGetAttr"): 300,
        ("libc.so.6", "malloc"): 50,
    }


def test_a_failing_layout_shift_names_its_classes_and_costliest_calls():
    # The same calls, each costing more: what to bring to the owner.
    assert gate.shift_detail(_dump(450, 60), _dump(300, 50)) == (
        "the shift by class: interpreter +150, allocation +10; the largest changes: "
        "python3.12:PyObject_GenericGetAttr (3 calls, 100.0 -> 150.0 instructions per call, "
        "+150); libc.so.6:malloc (2 calls, 25.0 -> 30.0 instructions per call, +10)"
    )
    row = gate.Row("c", 1.0, 1.0, 0.0, 0.0, "same", "slower", on_base=True, own=0.01)
    row.calls, row.base_name, row.shift = [], "base1234", "the shift by class: interpreter +150"
    assert gate.failure_reason(row).endswith(
        "fails, the owner's to accept or not; same Python work; no call differs; bring to the "
        "owner: the shift by class: interpreter +150"
    )


def test_a_function_is_named_short_in_the_shift_detail():
    name = (
        "bool llvm::DenseMapBase<llvm::DenseMap<mlir::Value, mlir::Operation*>>::doFind<"
        "mlir::Value>(mlir::Value const&) const"
    )
    assert gate._short(name) == "bool llvm::DenseMapBase<...>::doFind<...>(...) const"
    assert len(gate._short("f" * 200)) == 90


# Rule 537: address-keyed functions, replayed on #274's CI dumps against
# main's (run0 of each): each listed function's own instructions and calls on
# both sides, one of the program's own functions, and the rest of the row.
_KEYED = json.loads((Path(__file__).parent / "perf_gate_address_keyed.json").read_text())
_KEYED_ROWS = next(rows for label, rows in _KEYED.items() if label.startswith("#274"))
# coupling-lines-5: each listed lookup entered twice more than on its base
# (two more ops and a type registered).
_COUPLING = next(rows for label, rows in _KEYED.items() if label.startswith("coupling-lines-5"))


def _keyed_sides(case, rows=_KEYED_ROWS):
    """A recorded row as the judgement reads it: each side's own
    instructions and calls by function."""
    after, before, tree_calls, base_calls = {}, {}, {}, {}
    for obj, name, b, a, n, m in rows[case]["functions"]:
        before[obj, name], after[obj, name] = b, a
        base_calls[obj, name], tree_calls[obj, name] = n, m
    rest_before, rest_after = rows[case]["rest"]
    before["", "(the rest)"], after["", "(the rest)"] = rest_before, rest_after
    return after, before, tree_calls, base_calls


def test_every_address_keyed_entry_names_one_mlir_or_llvm_function_with_its_reasons():
    import re

    source = (_ROOT / "tools" / "perf-gate.py").read_text()
    start = source.index("ADDRESS_KEYED = (")
    block = source[start : source.index("\n)\n", start)]
    entries = re.findall(r"((?:\n    #[^\n]*)+)\n    r\"([^\"]+)\",", block)
    assert [pattern for _, pattern in entries] == list(gate.ADDRESS_KEYED)
    for comments, pattern in entries:
        # One function of MLIR or LLVM, never of this repository or the interpreter.
        assert re.match(r"\^(mlir|llvm)::[\w:]+\\\($", pattern), pattern
        assert "Keys on:" in comments and "Evidence: CI runs" in comments, pattern
    for function in [("x.so", "pulse::DriveOp::build(x)"), ("python3.12", "PyObject_Malloc")]:
        assert not gate.address_keyed(function)


def _keyed_row(case, after, before, tree_calls, base_calls, baseline=None):
    """The recorded row, failing against a stored baseline (the base's count
    unless *baseline*), judged on its address-keyed functions."""
    total = sum(before.values())
    row = gate.Row(
        case,
        baseline or total,
        sum(after.values()),
        0.0,
        None,
        "same",
        "slower",
        base_name="8311de57",
    )
    gate.judge_row(row, gate.judge_address_keyed(after, before, tree_calls, base_calls))
    return row


@pytest.mark.parametrize("case", sorted(_KEYED_ROWS))
def test_274s_rows_pass_outside_their_address_keyed_functions(case):
    expected = _KEYED_ROWS[case]["expected"]
    row = _keyed_row(case, *_keyed_sides(case))
    # Each row failed by more than its tolerance as a whole.
    assert row.delta > gate.REGRESSION
    assert row.keyed.whole == pytest.approx(expected["whole"], abs=1e-6)
    assert row.keyed.remainder == pytest.approx(expected["remainder"], abs=1e-6)
    # Against a baseline equal to the base, the rest is the same change.
    assert row.keyed.against_baseline == pytest.approx(expected["remainder"], abs=1e-6)
    assert row.verdict == "address-keyed" and not row.keyed.left_in


def test_real_work_outside_the_list_still_fails():
    after, before, tree_calls, base_calls = _keyed_sides("first_compile")
    own = next(f for f in after if f[1].startswith("pulse::"))
    after[own] += int(0.003 * sum(before.values()))
    row = _keyed_row("first_compile", after, before, tree_calls, base_calls)
    assert row.verdict == "slower" and row.keyed.against_baseline > gate.REGRESSION


def test_the_rest_is_held_to_the_stored_baseline_not_to_a_base_that_drifted():
    # The base already sits 0.3% above the stored baseline; a change adds
    # 0.35% of real work and moves a listed function by one instruction.
    after, before, tree_calls, base_calls = _keyed_sides("first_compile")
    for f in after:
        after[f] = before[f]
    uniquer = next(f for f in after if "getParametricStorageTypeImpl" in f[1])
    after[uniquer] += 1
    own = next(f for f in after if f[1].startswith("pulse::"))
    total = sum(before.values())
    after[own] = before[own] + int(0.0035 * total)
    row = _keyed_row("first_compile", after, before, tree_calls, base_calls, total / 1.003)
    assert row.keyed.remainder < gate.REGRESSION
    assert row.keyed.against_baseline > gate.REGRESSION
    assert row.verdict == "slower"


def test_a_listed_function_entered_many_more_times_stays_in_the_rest_and_fails():
    after, before, tree_calls, base_calls = _keyed_sides("first_compile")
    uniquer = next(f for f in after if "getParametricStorageTypeImpl" in f[1])
    # 200 more entries, far beyond 1% of 984, at the same cost per call.
    after[uniquer] += 200 * after[uniquer] // tree_calls[uniquer]
    tree_calls[uniquer] += 200
    row = _keyed_row("first_compile", after, before, tree_calls, base_calls)
    assert row.verdict == "slower"
    assert any(
        "getParametricStorageTypeImpl" in s and "calls 984 -> 1,184, more than 9 apart" in s
        for s in row.keyed.left_in
    )


def test_the_coupling_row_passes_with_its_few_extra_entries_counted_as_work():
    after, before, tree_calls, base_calls = _keyed_sides("first_compile", _COUPLING)
    baseline = _COUPLING["first_compile"]["baseline"]
    row = _keyed_row("first_compile", after, before, tree_calls, base_calls, baseline)
    assert row.delta > gate.REGRESSION
    assert row.keyed.against_baseline == pytest.approx(0.00335, abs=5e-5)
    assert row.verdict == "address-keyed"
    insert = next(s for s in row.keyed.taken_out if "RegisteredOperationName::insert" in s)
    assert "443 -> 445 calls, 911.2 -> 928.7 per call, taken out +7,797" in insert
    assert "left in for +2 calls +1,822" in insert


def test_the_entry_cap_is_one_percent_of_the_base_and_at_least_one():
    assert [gate.entry_cap(n) for n in (0, 40, 99, 100, 443, 984)] == [1, 1, 1, 1, 4, 9]
    after, before, tree_calls, base_calls = _keyed_sides("first_compile", _COUPLING)
    insert = next(f for f in after if "RegisteredOperationName::insert" in f[1])
    for extra, taken in ((4, True), (5, False)):
        calls = dict(tree_calls)
        calls[insert] = base_calls[insert] + extra
        keyed = gate.judge_address_keyed(after, before, calls, base_calls)
        assert any("RegisteredOperationName::insert" in s for s in keyed.taken_out) is taken


def test_fewer_entries_credit_the_rest_at_the_bases_cost_per_call():
    after, before, tree_calls, base_calls = _keyed_sides("first_compile")
    uniquer = next(f for f in after if "getParametricStorageTypeImpl" in f[1])
    # Two fewer entries, each at the tree's cost per call.
    per_call = after[uniquer] / tree_calls[uniquer]
    after[uniquer] -= round(2 * per_call)
    tree_calls[uniquer] -= 2
    keyed = gate.judge_address_keyed(after, before, tree_calls, base_calls)
    text = next(s for s in keyed.taken_out if "getParametricStorageTypeImpl" in s)
    assert "984 -> 982 calls" in text and "left in for -2 calls -1,089" in text


def test_a_listed_function_never_called_on_either_side_is_not_taken_out():
    after, before, tree_calls, base_calls = _keyed_sides("first_compile")
    total = sum(before.values())
    ghost = ("x.so", "mlir::RegisteredOperationName::insert(ghost)")
    before[ghost], after[ghost] = 0, int(0.05 * total)
    row = _keyed_row("first_compile", after, before, tree_calls, base_calls)
    assert row.verdict == "slower"
    assert any(s.startswith("x.so:") and "calls 0 -> 0" in s for s in row.keyed.left_in)


def test_a_row_whose_listed_functions_did_not_move_is_not_judged_this_way():
    after, before, tree_calls, base_calls = _keyed_sides("first_compile")
    for f in after:
        if gate.address_keyed(f):
            after[f] = before[f]
    assert gate.judge_address_keyed(after, before, tree_calls, base_calls) is None


def test_only_a_regression_is_judged_this_way():
    for verdict in ("improved", "layout", "ok", "faster"):
        row = gate.Row("first_compile", 1.0, 2.0, 0.0, None, "same", verdict)
        gate.apply_address_keyed([row], {"first_compile": "x"}, {"first_compile": "x"})
        assert row.verdict == verdict and row.keyed is None


def test_an_address_keyed_pass_shows_the_whole_change_and_each_function_per_call():
    row = _keyed_row("first_compile", *_keyed_sides("first_compile"))
    (note,) = gate.keyed_notes([row])
    assert note.startswith(
        "first_compile: address-keyed, +0.85% against the stored baseline and +0.85% from its "
        "base 8311de57"
    )
    assert "outside the address-keyed functions +0.26% against the stored baseline" in note
    assert "984 calls, 544.6 -> 590.5 per call, +45,139" in note
    assert "688 calls, 35.4 -> 124.9 per call, +61,580" in note
    assert "443 calls, 911.2 -> 831.0 per call, -35,537" in note
    assert "address-keyed" not in gate.FAILING


def test_a_regression_still_above_the_limit_says_so_outside_the_list():
    after, before, tree_calls, base_calls = _keyed_sides("first_compile")
    own = next(f for f in after if f[1].startswith("pulse::"))
    after[own] += int(0.003 * sum(before.values()))
    row = _keyed_row("first_compile", after, before, tree_calls, base_calls)
    reason = gate.failure_reason(row)
    assert "regression" in reason
    assert (
        "judged outside its address-keyed functions: "
        "outside the address-keyed functions +0.56% against the stored baseline" in reason
    )
    assert reason.endswith(": fails")
    # A layout-only row that also fails outside the list says both.
    row.on_base, row.own = True, 0.0085
    assert "outside the address-keyed functions +0.56%" in gate.failure_reason(row)


# Rule 550: the listed functions judged like for like against the records the
# baseline keeps of them. walks-without-recursion failed while main stood
# +1.02% above the baseline after #274's address-keyed pass.
_WALKS = next(rows for label, rows in _KEYED.items() if label.startswith("walks-without"))


def _walks_row(case, edit=None):
    """walks' recorded row, failing against the baseline, judged against the
    baseline's records of its listed functions (*edit* changes its own
    instructions and calls first)."""
    c = _WALKS[case]
    after = {(obj, name): ir for obj, name, ir, _ in c["functions"]}
    calls = {(obj, name): n for obj, name, _, n in c["functions"]}
    after["", "(the rest)"] = c["rest"]
    if edit:
        edit(after, calls)
    row = gate.Row(
        case, c["baseline"], sum(after.values()) / c["reps"], 0.0, None, "same", "slower"
    )
    gate.judge_row(row, gate.judge_records(after, calls, c["records"], c["baseline"], c["reps"]))
    return row


def test_a_change_merged_as_address_keyed_does_not_stay_in_later_rows():
    row = _walks_row("first_compile")
    assert row.delta == pytest.approx(0.00608, abs=5e-5)
    assert row.keyed.records and row.keyed.against_baseline == pytest.approx(0.00133, abs=5e-5)
    assert row.verdict == "address-keyed"
    (note,) = gate.keyed_notes([row])
    assert "from its base" not in note and "against the baseline's records" in note
    assert "443 calls, 911.2 -> 931.9 per call" in note


def test_work_outside_the_list_stays_held_to_the_baseline():
    # walks' specialize_wait: main itself stood +0.23% above the baseline
    # outside the listed functions, and walks adds about +0.24% more.
    row = _walks_row("specialize_wait")
    assert row.keyed.against_baseline == pytest.approx(0.00473, abs=5e-5)
    assert row.verdict == "slower"


def test_no_more_than_two_percent_of_the_baseline_is_taken_out():
    def eightfold(after, calls):
        for f in after:
            if gate.address_keyed(f):
                after[f] *= 8

    row = _walks_row("first_compile", eightfold)
    assert row.keyed.too_much and row.verdict == "slower"
    assert "more than 2% of the baseline's count" in gate.failure_reason(row)


def test_a_baseline_records_its_listed_functions_and_they_are_not_a_size():
    records = {"first_compile": _WALKS["first_compile"]["records"]}
    measured = gate.as_baseline(
        {},
        {"first_compile": {"ops": 1}},
        {"first_compile": gate.Measured([12_000_000.0], [1.0])},
        {},
        {},
        records=records,
    )
    entry = measured["cases"]["first_compile"]
    assert entry["address_keyed"] == records["first_compile"]
    (row,) = gate.compare(measured, {"first_compile": {"ops": 1}}, None)
    assert row.verdict == "ok"


def _keyed_dump(entries_a, cost_a, entries_b, cost_b, rest):
    """A small Callgrind dump of one case: a caller that enters the uniquer
    lookup in two objects, and the rest of its own work."""
    name = "mlir::StorageUniquer::getParametricStorageTypeImpl(x)"
    return "\n".join(
        [
            "desc: Trigger: Client Request: case",
            "events: Ir",
            "ob=(1) /x/liba.so",
            "fn=(1) caller",
            f"0 {rest}",
            "cob=(2) /x/libb.so",
            f"cfn=(2) {name}",
            f"calls={entries_b} 0",
            f"0 {cost_b}",
            f"cfn=(3) {name}",
            f"calls={entries_a} 0",
            f"0 {cost_a}",
            "fn=(3)",
            f"0 {cost_a}",
            "ob=(2)",
            "fn=(2)",
            f"0 {cost_b}",
            "",
        ]
    )


def test_records_round_trip_by_name_over_the_objects_that_hold_one():
    text = _keyed_dump(1000, 500_000, 10, 5_000, 2_000_000)
    records = gate.address_keyed_records(text, 10)
    assert records == {"mlir::StorageUniquer::getParametricStorageTypeImpl(x)": [101.0, 50_500.0]}
    baseline = sum(gate.dump_costs(text).values()) / 10
    # The same dump against its own records: nothing moved, nothing to judge.
    assert gate.judge_against_records(text, records, baseline, 10) is None
    # Each call 10% dearer in both objects: all of it taken out, rest unchanged.
    dearer = _keyed_dump(1000, 550_000, 10, 5_500, 2_000_000)
    keyed = gate.judge_against_records(dearer, records, baseline, 10)
    assert keyed.records and keyed.remainder == pytest.approx(0.0, abs=1e-12)
    assert keyed.taken_out == (
        "mlir::StorageUniquer::getParametricStorageTypeImpl(...) "
        "(1,010 calls, 500.0 -> 550.0 per call, +50,500)",
    )


def test_a_listed_function_s_own_cost_without_calls_in_an_object_counts_on_both_sides():
    # Its own instructions in a third object it has no calls into there.
    name = "mlir::StorageUniquer::getParametricStorageTypeImpl(x)"
    extra = f"\nob=(3) /x/libc.so\nfn=(4) {name}\n0 7000\n"
    text = _keyed_dump(1000, 500_000, 10, 5_000, 2_000_000) + extra
    records = gate.address_keyed_records(text, 10)
    baseline = sum(gate.dump_costs(text).values()) / 10
    assert gate.judge_against_records(text, records, baseline, 10) is None


def test_the_gate_judges_a_regression_against_the_records_for_its_repetitions():
    text = _keyed_dump(1000, 500_000, 10, 5_000, 2_000_000)
    records = {"case": gate.address_keyed_records(text, 10)}
    baseline = sum(gate.dump_costs(text).values()) / 10
    # Each call 3% dearer: 0.6% of the row taken out, within the 2% bound.
    dearer = _keyed_dump(1000, 515_000, 10, 5_150, 2_000_000)
    row = gate.Row(
        "case", baseline, sum(gate.dump_costs(dearer).values()) / 10, 0.0, None, "same", "slower"
    )
    gate.apply_address_keyed([row], {"case": dearer}, {}, records, {"case": 10})
    assert row.verdict == "address-keyed" and row.keyed.records
    # Without the case's repetitions, nothing judges it against the records.
    other = gate.Row("case", baseline, row.measured, 0.0, None, "same", "slower")
    gate.apply_address_keyed([other], {"case": dearer}, {}, records, {})
    assert other.verdict == "slower" and other.keyed is None
