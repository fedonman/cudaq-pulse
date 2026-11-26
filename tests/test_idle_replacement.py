# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle replacement: the pass member, its rule, the option that carries it,
and what a compilation without it pays (nothing)."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.compiler import DEFAULT_PASSES, CompileOptions

from tests._handles import Q

_ROOT = str(Path(pulse.__file__).parent.parent)


@pulse.kernel
def echo(qudit, window):
    line = get_drive_line(qudit)
    lead = (window - 40) // 2
    wait(line, lead)
    drive(line, square(40, 0.1))
    wait(line, window - 40 - lead)


@pulse.kernel
def two_lines(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(400, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def ramsey(q, amp: float, tau: float):
    d = get_drive_line(q)
    drive(d, square(40, amp))
    wait(d, tau)
    drive(d, square(40, amp))


def _rule(**settings):
    return pulse.IdleReplacement(echo, **({"threshold": 100 * pulse.ns} | settings))


# -- The method --------------------------------------------------------------


def test_replace_idle_takes_only_a_rule():
    with pytest.raises(pulse.InvalidTypeError, match=r"takes a pulse\.IdleReplacement, got "):
        pulse.compile(two_lines).replace_idle(echo)


def test_replace_idle_leaves_the_compiled_kernel_as_it_is():
    compiled = pulse.compile(two_lines)
    idled = compiled.replace_idle(_rule())
    assert idled is not compiled and isinstance(idled, type(compiled))
    assert compiled.lower(Q[0], Q[1]).idle_windows is None
    assert idled.lower(Q[0], Q[1]).idle_windows is not None
    with pytest.raises(pulse.InvalidValueError, match="replaced already"):
        idled.replace_idle(_rule())


def test_there_is_no_pass_member_or_keyword_for_it():
    assert "idle_replacement" not in {p.value for p in pulse.Pass}
    with pytest.raises(TypeError, match="unexpected keyword argument 'idle_replacement'"):
        pulse.compile(two_lines, **{"idle_replacement": _rule()})


# -- The rule -----------------------------------------------------------------


def test_a_rule_holds_what_it_was_given():
    data = Q[0:2]
    rule = pulse.IdleReplacement(
        echo, threshold=200, qudits=data, leading=True, trailing=1, align="end"
    )
    assert rule.fill is echo
    assert rule.threshold == 200 and type(rule.threshold) is int
    assert rule.qudits == (Q[0], Q[1])
    assert (rule.leading, rule.trailing, rule.align) == (True, True, "end")
    plain = _rule()
    assert plain.threshold == 100 * pulse.ns and isinstance(plain.threshold, pulse.Time)
    assert (plain.qudits, plain.leading, plain.trailing, plain.align) == (
        None,
        False,
        False,
        "center",
    )
    assert pulse.IdleReplacement(None, threshold=0).fill is None


def test_a_rule_is_immutable():
    rule = _rule()
    with pytest.raises(AttributeError, match="immutable"):
        rule.threshold = 5
    with pytest.raises(AttributeError, match="immutable"):
        del rule.fill


def test_rules_compare_by_kernel_and_settings():
    assert _rule() == _rule() and hash(_rule()) == hash(_rule())
    assert _rule() != _rule(threshold=200)
    assert _rule() != _rule(leading=True)
    assert _rule() != _rule(align="start")
    assert _rule() != _rule(qudits=[Q[0]])
    assert _rule() != pulse.IdleReplacement(None, threshold=100 * pulse.ns)
    # A time and a count of cycles are different thresholds on every clock.
    assert _rule(threshold=200) != _rule(threshold=200 * pulse.ns)
    assert _rule() != "rule"


def test_a_rule_prints_its_kernel_by_name_and_only_what_was_set():
    assert repr(_rule(threshold=200)) == f"IdleReplacement({__name__}.echo, threshold=200)"
    shown = repr(_rule(qudits=Q[0:2], trailing=True, align="start"))
    assert shown == (
        f"IdleReplacement({__name__}.echo, threshold={100 * pulse.ns!r}, qudits=(0, 1), "
        "trailing=True, align='start')"
    )


@pytest.mark.parametrize(
    ("threshold", "error", "text"),
    [
        (100.0, pulse.InvalidTypeError, "threshold is a float (100.0)"),
        (True, pulse.InvalidTypeError, "not True"),
        ("100", pulse.InvalidTypeError, "not '100'"),
        (-1, pulse.InvalidValueError, "threshold is -1; it cannot be negative"),
        (-1 * pulse.ns, pulse.InvalidValueError, "it cannot be negative"),
    ],
)
def test_a_threshold_is_cycles_or_a_time(threshold, error, text):
    with pytest.raises(error) as raised:
        pulse.IdleReplacement(echo, threshold=threshold)
    assert text in str(raised.value)


def test_a_float_threshold_shows_both_spellings():
    with pytest.raises(pulse.InvalidTypeError) as raised:
        pulse.IdleReplacement(echo, threshold=100.0)
    assert "    threshold=100\n    threshold=100 * pulse.ns" in str(raised.value)


def test_a_threshold_is_required_by_keyword():
    with pytest.raises(TypeError):
        pulse.IdleReplacement(echo)
    with pytest.raises(TypeError):
        pulse.IdleReplacement(echo, 200)


def test_a_fill_is_a_kernel_of_a_qudit_and_a_window():
    def plain(qudit, window):
        return None

    with pytest.raises(pulse.InvalidTypeError, match=r"fill is a @pulse\.kernel"):
        pulse.IdleReplacement(plain, threshold=200)

    @pulse.kernel
    def one(qudit):
        wait(get_drive_line(qudit), 10)

    @pulse.kernel
    def three(qudit, window, start):
        wait(get_drive_line(qudit), window)

    @pulse.kernel
    def with_float(qudit, window: float):
        wait(get_drive_line(qudit), window)

    @pulse.kernel
    def with_runtime_qudit(qudit: pulse.Qudit, window):
        wait(get_drive_line(qudit), window)

    for fill in (one, three, with_float, with_runtime_qudit):
        with pytest.raises(pulse.InvalidTypeError) as raised:
            pulse.IdleReplacement(fill, threshold=200)
        assert f"fill {fill.__name__}(" in str(raised.value)
        assert "does not take (qudit, window)" in str(raised.value)


def test_align_and_qudits_are_checked():
    with pytest.raises(
        pulse.InvalidValueError, match="unknown align 'centre'. Did you mean 'center'"
    ):
        _rule(align="centre")
    with pytest.raises(pulse.InvalidTypeError, match="qudits are qudit handles"):
        _rule(qudits=[0, 1])
    with pytest.raises(pulse.InvalidValueError, match="qudits is empty"):
        _rule(qudits=[])
    assert _rule(qudits=Q[3]).qudits == (Q[3],)


# -- The options of a kernel whose idle windows are replaced ------------------


def _options(compiled):
    return compiled._dispatch.options


def test_options_with_a_rule_are_apart_from_those_without():
    plain = CompileOptions.make()
    with_rule = _options(pulse.compile(two_lines).replace_idle(_rule()))
    assert type(plain) is CompileOptions
    assert with_rule != plain and with_rule.passes == plain.passes
    assert with_rule == _options(pulse.compile(two_lines).replace_idle(_rule()))
    assert with_rule != _options(pulse.compile(two_lines).replace_idle(_rule(threshold=7)))
    assert with_rule.without_rule() == plain and type(with_rule.without_rule()) is CompileOptions
    assert repr(with_rule) == f"{plain!r} + {_rule()!r}"


def test_a_copy_of_the_options_keeps_the_rule_and_the_unit_warnings():
    rule = _rule()
    options = _options(pulse.compile(two_lines, unit_warnings=True).replace_idle(rule))
    assert options.units is True and options.idle_replacement is rule
    assert options.without_rule().unit_warnings is True
    copy = dataclasses.replace(options, schedule="asap")
    assert copy.idle_replacement is rule and copy.units is True
    assert copy.schedule == "asap" and copy != options


# -- A compilation with a rule ------------------------------------------------


def _compiled(kernel, **settings):
    return pulse.compile(kernel).replace_idle(_rule(**settings))


def test_a_wait_as_long_as_a_float_is_refused_with_the_way_out():
    with pytest.raises(pulse.CompilationError) as raised:
        _compiled(ramsey).lower(Q[0], 0.1, 300.0)
    assert str(raised.value).startswith(
        "idle replacement: the length of a wait in ramsey() is computed from the float "
        "parameter 'tau', so its idle windows differ at every call; annotate tau as int to "
        "compile for each value, or compile without idle replacement"
    )
    assert raised.value.code == pulse.ErrorCode.SYMBOLIC_PARAMETER


def test_refusing_a_float_wait_leaves_the_kernel_as_it_was():
    before = ramsey.lower(Q[0], 0.1, 300.0)
    with pytest.raises(pulse.CompilationError):
        _compiled(ramsey).lower(Q[0], 0.1, 300.0)
    assert ramsey.runtime_parameters == ("amp", "tau")
    after = ramsey.lower(Q[0], 0.1, 500.0)
    assert after.generic.mlir() == before.generic.mlir()
    assert [e.duration_cycles for e in after.schedule] == [40, 500, 40]


def test_qudits_that_each_call_binds_are_not_supported_yet():
    @pulse.kernel
    def anywhere(q: pulse.Qudit):
        drive(get_drive_line(q), square(40, 0.1))

    with pytest.raises(pulse.UnsupportedError, match="does not support yet"):
        _compiled(anywhere).lower(Q[0])


# -- The report of idle windows (a rule without a fill) -----------------------


@pulse.kernel
def host(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(40, 0.1))
    sync(da, db, rb)
    readout(rb, square(600, 0.05))
    sync(da, db, rb)  # ends the window
    drive(da, square(40, 0.1))
    drive(db, square(40, 0.1))


@pulse.kernel
def rounds(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))
        sync(da, db, rb)
        drive(da, square(40, 0.1))


@pulse.kernel
def crossed(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(400, 0.1), tone=da.tone)  # a's tone plays: a is not idle
    wait(db, 300)
    drive(db, square(40, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def own_readout(a):
    da = get_drive_line(a)
    ra = get_readout_line(a)
    drive(da, square(40, 0.1))
    readout(ra, square(300, 0.05))
    wait(ra, 500)
    sync(da, ra)
    drive(da, square(40, 0.1))


@pulse.kernel
def written_wait(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    wait(da, 400)
    drive(db, square(440, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def before_a_branch(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    m = readout(rb, square(600, 0.05))
    if m:
        drive(db, square(40, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def into_a_branch(a, b):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    m = readout(rb, square(600, 0.05))
    if m:
        drive(da, square(40, 0.1))
    wait(da, 400)
    drive(da, square(40, 0.1))


@pulse.kernel
def detuned(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    shift_frequency(da.tone, 1.0e6)
    drive(db, square(600, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def uneven(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(400, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))
        drive(db, square(440, 0.2))


def _windows(kernel, *args, schedule="asap", **settings):
    rule = pulse.IdleReplacement(None, **({"threshold": 100} | settings))
    program = pulse.compile(kernel, schedule=schedule).replace_idle(rule)
    return program.lower(*args).idle_windows


def _spans(windows):
    return [(w.qudit, w.start, w.end, w.rounds) for w in windows]


_OPEN = "not filled: the rule has no fill"


def test_a_gap_the_schedule_made_is_a_window():
    (window,) = _windows(host, Q[0], Q[1])
    line = Path(__file__).read_text().splitlines().index("    sync(da, db, rb)  # ends the window")
    assert window == pulse.compiler.IdleWindow(
        0, 40, 640, 1, f"sync() at line {line + 1} of host()", False, _OPEN
    )
    assert window.cycles == 600


@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp", "alap_rcp"])
def test_the_report_is_the_same_under_every_policy_that_keeps_the_times(schedule):
    assert _spans(_windows(host, Q[0], Q[1], schedule=schedule)) == [(0, 40, 640, 1)]


def test_the_windows_print_as_a_table():
    text = str(_windows(host, Q[0], Q[1]))
    head, row = text.splitlines()
    assert head.split() == ["qudit", "start", "end", "cycles", "rounds", "before", "result"]
    assert row.split()[:5] == ["0", "40", "640", "600", "1"]
    assert row.endswith(_OPEN)
    assert str(_windows(host, Q[0], Q[1], threshold=601)) == "no idle windows"


def test_a_rule_without_a_fill_leaves_the_program_as_it_is():
    rule = pulse.IdleReplacement(None, threshold=100)
    program = pulse.compile(host).replace_idle(rule).lower(Q[0], Q[1])
    plain = pulse.compile(host).lower(Q[0], Q[1])
    assert program.mlir() == plain.mlir()
    assert program._module.serialize() == plain._module.serialize()
    # The program carries the options it was compiled with: the rule is the
    # kernel's (replace_idle), not the program's.
    assert program.options == plain.options
    assert plain.idle_windows is None
    assert program.bind(()).idle_windows is program.idle_windows


@pytest.mark.parametrize(
    ("threshold", "count"),
    [(600, 1), (601, 0), (300 * pulse.ns, 1), (300.25 * pulse.ns, 0)],
)
def test_a_window_is_at_least_the_threshold(threshold, count):
    # The clock runs two cycles a nanosecond: the window lasts 600 cycles.
    assert len(_windows(host, Q[0], Q[1], threshold=threshold)) == count


def test_the_rule_can_name_its_qudits():
    assert _windows(host, Q[0], Q[1], qudits=[Q[1]]) == ()
    assert _spans(_windows(host, Q[0], Q[1], qudits=[Q[0]])) == [(0, 40, 640, 1)]


def test_a_qudit_whose_tone_plays_from_another_line_is_not_idle():
    # a's line is free from 40 to 780; its tone plays on b's line until 440.
    assert _spans(_windows(crossed, Q[0], Q[1])) == [(0, 440, 780, 1)]


@pytest.mark.parametrize(("schedule", "start", "end"), [("asap", 300, 800), ("alap", 300, 760)])
def test_a_qudit_is_not_idle_while_its_own_readout_runs(schedule, start, end):
    # Under alap the readout is written after the gap it runs in.
    settings = {"leading": True} if schedule == "alap" else {}
    windows = _windows(own_readout, Q[0], schedule=schedule, **settings)
    assert _spans(windows) == [(0, start, end, 1)]
    assert windows[0].result == _OPEN


def test_a_written_wait_is_an_operation_and_no_window():
    assert _windows(written_wait, Q[0], Q[1]) == ()


def test_idle_before_the_first_operation_and_after_the_last_is_left_alone_unless_asked():
    (first,) = _windows(host, Q[0], Q[1], schedule="alap", threshold=30)[:1]
    assert _spans([first]) == [(0, 40, 640, 1)]  # nothing leads under alap here
    (lead,) = _windows(before_a_branch, Q[0], Q[1], schedule="alap")
    assert (lead.start, lead.end) == (0, 600)
    assert lead.result == ("left alone: before the qudit's first operation (leading=True fills it)")
    (asked,) = _windows(before_a_branch, Q[0], Q[1], schedule="alap", leading=True)
    assert asked.result == _OPEN
    tail = _windows(uneven, Q[0], Q[1], 1)[-1]
    assert (tail.start, tail.end, tail.before) == (440, 840, None)
    assert tail.result == "left alone: after the qudit's last operation (trailing=True fills it)"
    assert _windows(uneven, Q[0], Q[1], 1, trailing=True)[-1].result == _OPEN


def test_the_gap_before_a_branch_is_a_window_and_one_that_ends_in_an_arm_is_not():
    (window,) = _windows(before_a_branch, Q[0], Q[1])
    assert (window.start, window.end, window.result) == (40, 640, _OPEN)
    (window,) = _windows(into_a_branch, Q[0], Q[1])
    assert (window.start, window.end) == (40, 600)
    assert window.result == "left alone: in or next to an arm of a branch"


@pulse.kernel
def lanes_and_rounds(a, b, c, d, e, f, g, n: int):
    # many()'s lane-bound windows in each round, beside f's uneven rounds.
    lines = [get_drive_line(q) for q in (a, b, c, d)]
    de = get_drive_line(e)
    re = get_readout_line(e)
    df = get_drive_line(f)
    dg = get_drive_line(g)
    for _ in range(n):
        drive(df, square(40, 0.1))
        drive(dg, square(400, 0.1))
        sync(df, dg)
        for k, line in enumerate(lines):
            drive(line, square(40 + 20 * k, 0.1))
        drive(de, square(40, 0.1))
        drive(df, square(40, 0.1))
        drive(dg, square(440, 0.2))
        sync(*lines, de, re, df, dg)
        readout(re, square(600, 0.05))
        sync(*lines, de, re, df, dg)


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_windows_left_alone_for_lanes_keep_the_reasons_of_the_others(schedule):
    # A window the lane fallback never had to try (it was not to be filled
    # anyway) says why, as it would without the fallback, not UNTRIED.
    from cudaq_pulse._idle_windows import LANES, UNTRIED

    program = _filled(lanes_and_rounds, *Q[0:7], 6, fill=pair, schedule=schedule)
    results = [w.result for w in program.idle_windows]
    assert LANES in results
    assert UNTRIED not in results


def test_a_window_after_the_kernel_changes_the_tones_frequency_is_left_alone():
    (window,) = _windows(detuned, Q[0], Q[1])
    assert (window.start, window.end) == (40, 600)
    assert window.result == "left alone: the kernel changed the frequency of the qudit's tone"


@pytest.mark.parametrize("n", [3, 9, 1000])
def test_a_window_in_a_loop_counts_its_rounds(n):
    # Unrolled (3) and rolled (9, 1000): one window, as many rounds.
    (window,) = _windows(rounds, Q[0], Q[1], n)
    assert (window.start, window.end, window.rounds, window.result) == (40, 640, n, _OPEN)


def test_a_window_whose_rounds_differ_only_in_length_is_given_at_its_shortest():
    # The first round idles 360 cycles, the others 760: a fill fits 360.
    window = _windows(uneven, Q[0], Q[1], 50)[0]
    assert (window.start, window.end, window.rounds, window.result) == (40, 400, 50, _OPEN)


def test_windows_whose_rounds_differ_otherwise_are_listed_by_kind_of_round():
    # Under asap a's drive of a round follows its drive of the round before
    # at once in the second round and after the readout in the others.
    windows = _windows(memory, Q[0], Q[1], 20)
    assert [(w.qudit, w.start, w.end, w.rounds, w.result) for w in windows if w.qudit == 0][:1] == [
        (0, 80, 640, 18, "left alone: its rounds differ")
    ]


def test_an_error_of_the_program_is_the_one_it_raises_without_the_rule():
    # The schedule's errors print where the operation is; the operations of
    # a compilation with a rule carry numbers there.
    @pulse.kernel(max_duration_cycles=500)
    def budgeted(a):
        da = get_drive_line(a)
        drive(da, square(400, 0.1))
        wait(da, 200)

    with pytest.raises(RuntimeError) as plain:
        pulse.compile(budgeted).lower(Q[0])
    with pytest.raises(RuntimeError) as ruled:
        pulse.compile(budgeted).replace_idle(pulse.IdleReplacement(None, threshold=100)).lower(Q[0])
    assert "max_duration_cycles=500" in str(plain.value)
    assert str(ruled.value) == str(plain.value)
    assert type(ruled.value) is type(plain.value)
    assert ruled.value.__context__ is None


# -- Fills --------------------------------------------------------------------


@pulse.kernel
def pi(qudit, window):
    line = get_drive_line(qudit)
    drive(line, gaussian(40, 0.2, 10))


@pulse.kernel
def whole(qudit, window):
    line = get_drive_line(qudit)
    drive(line, gaussian(window, 0.01, window / 4))


@pulse.kernel
def pair(qudit, window):
    line = get_drive_line(qudit)
    quarter = (window - 80) // 4
    wait(line, quarter)
    drive(line, gaussian(40, 0.2, 10))
    wait(line, 2 * quarter)
    drive(line, gaussian(40, 0.2, 10))
    wait(line, window - 80 - 3 * quarter)


@pulse.kernel
def train(qudit, window):
    line = get_drive_line(qudit)
    for _ in range(12):
        wait(line, 5)
        drive(line, gaussian(40, 0.2, 10))
        wait(line, 5)


@pulse.kernel
def many(a, b, c, d, e):
    lines = [get_drive_line(q) for q in (a, b, c, d)]
    de = get_drive_line(e)
    re = get_readout_line(e)
    for n, line in enumerate(lines):
        drive(line, square(40 + 20 * n, 0.1))
    drive(de, square(40, 0.1))
    sync(*lines, de, re)
    readout(re, square(600, 0.05))
    sync(*lines, de, re)
    for line in lines:
        drive(line, square(40, 0.1))


@pulse.kernel
def middle_tone(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    wait(db, 200)
    drive(db, square(100, 0.1), tone=da.tone)  # a's tone plays 200..300 of a's gap
    wait(db, 300)
    sync(da, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def into_a_loop(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(600, 0.1))
    for _ in range(n):
        sync(da, db)
        drive(da, square(40, 0.1))
        drive(db, square(100, 0.1))


@pulse.kernel
def waits_in_a_row(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    wait(da, 50)
    wait(da, 50)
    drive(da, square(40, 0.1))
    drive(db, square(600, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))


_POLICIES = ["asap", "alap", "rcp", "alap_rcp"]


def _filled(kernel, *args, fill=pi, schedule="asap", passes=None, **settings):
    rule = pulse.IdleReplacement(fill, **({"threshold": 100} | settings))
    return pulse.compile(kernel, schedule=schedule, passes=passes).replace_idle(rule).lower(*args)


def _events(program):
    return [(e.kind, e.line, e.start, e.end) for e in program.schedule]


def _kept(kernel, *args, schedule="asap", **settings):
    """The program with its windows filled, after checking that it is the
    program without the rule with something written into it."""
    filled = _filled(kernel, *args, schedule=schedule, **settings)
    plain = pulse.compile(kernel, schedule=schedule).lower(*args)
    host = _events(plain)
    assert [event for event in _events(filled) if event in host] == host
    assert filled.duration == plain.duration
    return filled


@pytest.mark.parametrize("schedule", _POLICIES)
def test_a_fill_is_written_into_its_window_and_nothing_else_moves(schedule):
    program = _kept(host, Q[0], Q[1], schedule=schedule)
    assert _events(program) == [
        ("drive", 0, 0, 40),
        ("drive", 1, 0, 40),
        ("readout", 2, 40, 640),
        ("wait", 0, 40, 320),
        ("drive", 0, 320, 360),
        ("wait", 0, 360, 640),
        ("drive", 0, 640, 680),
        ("drive", 1, 640, 680),
    ]
    (window,) = program.idle_windows
    assert (window.qudit, window.start, window.end, window.filled) == (0, 40, 640, True)
    assert window.result == "filled by pi()"


@pytest.mark.parametrize(
    ("align", "written"),
    [
        ("start", [("drive", 0, 40, 80), ("wait", 0, 80, 640)]),
        ("center", [("wait", 0, 40, 320), ("drive", 0, 320, 360), ("wait", 0, 360, 640)]),
        ("end", [("wait", 0, 40, 600), ("drive", 0, 600, 640)]),
    ],
)
@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_fill_shorter_than_its_window_is_placed_as_align_says(schedule, align, written):
    program = _kept(host, Q[0], Q[1], schedule=schedule, align=align)
    assert _events(program)[3:-2] == written


def test_a_fill_as_long_as_its_window_needs_no_wait():
    program = _kept(host, Q[0], Q[1], fill=whole)
    assert _events(program)[3:-2] == [("drive", 0, 40, 640)]


def test_a_fills_pulse_is_not_fused_with_the_kernels_pulse_next_to_it():
    # Square pulses of one amplitude, back to back: the passes would make
    # one pulse of the kernel's two and the fill's between them.
    @pulse.kernel
    def flat(qudit, window):
        drive(get_drive_line(qudit), square(window, 0.1))

    program = _kept(host, Q[0], Q[1], fill=flat)
    assert _events(program)[3:-2] == [("drive", 0, 40, 640)]


def test_a_fill_longer_than_its_window_is_an_error_that_names_both():
    @pulse.kernel
    def long(qudit, window):
        drive(get_drive_line(qudit), gaussian(700, 0.01, 100))

    with pytest.raises(pulse.CompilationError) as raised:
        _filled(host, Q[0], Q[1], fill=long)
    assert str(raised.value).startswith(
        "idle replacement: fill long() takes 700 cycles, 100 more than its window of 600 "
        "cycles on qudit 0; shorten the fill, or raise threshold= so that this window is "
        "left alone"
    )
    assert raised.value.code == pulse.ErrorCode.INVALID_DURATION


def _fill_that(body):
    """A fill whose statements after its line and tone are *body*."""
    scope = {"pulse": pulse, "Q": Q}
    source = (
        "@pulse.kernel\n"
        "def odd(qudit, window):\n"
        "    line = get_drive_line(qudit)\n"
        "    tone = line.tone\n" + "".join(f"    {line}\n" for line in body)
    )
    exec(source, scope)  # noqa: S102
    return scope["odd"]


@pytest.mark.parametrize(
    ("body", "what"),
    [
        (["readout(get_readout_line(qudit), square(100, 0.05))"], "uses get_readout_line"),
        (["sync(line, line)"], "uses sync"),
        (["shift_frequency(tone, 1.0e6)"], "uses shift_frequency"),
        (["set_phase(tone, 0.5)"], "uses set_phase"),
        (
            ["shift_phase(tone, 0.5)", "drive(line, square(40, 0.1))"],
            "leaves its tone's phase shifted by 0.5 rad (its shifts must add up to zero)",
        ),
    ],
)
def test_a_fill_drives_and_waits_on_its_own_line_and_nothing_else(body, what):
    with pytest.raises(pulse.CompilationError) as raised:
        _filled(host, Q[0], Q[1], fill=_fill_that(body))
    assert str(raised.value).startswith(f"idle replacement: fill odd() {what}; in this version")
    assert raised.value.code == pulse.ErrorCode.UNSUPPORTED_CONSTRUCT


def test_a_fill_may_shift_its_tones_phase_by_constants_that_add_up_to_zero():
    fill = _fill_that(
        ["shift_phase(tone, 0.5)", "drive(line, gaussian(40, 0.2, 10))", "shift_phase(tone, -0.5)"]
    )
    program = _kept(host, Q[0], Q[1], fill=fill)
    assert ("drive", 0, 320, 360) in _events(program)


def test_a_fill_with_a_schedule_of_its_own_is_refused():
    @pulse.kernel(schedule="asap")
    def blocked(qudit, window):
        drive(get_drive_line(qudit), gaussian(40, 0.2, 10))

    with pytest.raises(pulse.CompilationError, match="has a schedule= of its own"):
        _filled(host, Q[0], Q[1], fill=blocked)


@pytest.mark.parametrize("schedule", _POLICIES)
def test_the_whole_gap_is_written_around_a_pulse_at_the_qudits_tone(schedule):
    # a idles 40..200 and 300..600; b's line plays a's tone in between.
    program = _kept(middle_tone, Q[0], Q[1], schedule=schedule, threshold=60, leading=True)
    spans = [(w.start, w.end, w.result) for w in program.idle_windows]
    written = [e for e in _events(program) if e[1] == 0 and e[0] == "wait"]
    if schedule in ("asap", "rcp"):
        assert spans == [(40, 200, "filled by pi()"), (300, 600, "filled by pi()")]
        assert written == [
            ("wait", 0, 40, 100),
            ("wait", 0, 140, 200),
            ("wait", 0, 200, 300),  # while the tone plays
            ("wait", 0, 300, 430),
            ("wait", 0, 470, 600),
        ]
    else:
        assert spans == [(0, 160, "filled by pi()"), (300, 600, "filled by pi()")]


def test_a_wait_is_written_while_the_qudits_own_readout_runs():
    program = _kept(own_readout, Q[0], schedule="alap", leading=True)
    assert [e for e in _events(program) if e[1] == 0][:4] == [
        ("wait", 0, 0, 300),
        ("wait", 0, 300, 510),
        ("drive", 0, 510, 550),
        ("wait", 0, 550, 760),
    ]


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_the_windows_of_several_qudits_that_end_at_one_sync_are_all_filled(schedule):
    program = _kept(many, Q[0], Q[1], Q[2], Q[3], Q[4], fill=pair, schedule=schedule)
    assert [(w.qudit, w.start, w.end, w.filled) for w in program.idle_windows] == [
        (qudit, 100, 700, True) for qudit in range(4)
    ]
    drives = [e for e in _events(program) if e[0] == "drive" and 100 <= e[2] < 700]
    assert drives == [("drive", line, at, at + 40) for line in range(4) for at in (230, 530)]


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_fills_that_need_more_drive_lanes_than_there_are_are_left_alone(schedule):
    # Under rcp and alap_rcp a fill's pulses take drive lanes like any
    # pulse: a window whose fill would move an operation of the kernel is
    # left unfilled and says why, and the program compiles (decision 280).
    from cudaq_pulse._idle_windows import LANES, UNTRIED

    program = _filled(many, Q[0], Q[1], Q[2], Q[3], Q[4], fill=pair, schedule=schedule)
    plain = pulse.compile(many, schedule=schedule).lower(Q[0], Q[1], Q[2], Q[3], Q[4])
    results = [w.result for w in program.idle_windows]
    assert LANES in results
    assert all(w.filled or w.result in (LANES, UNTRIED) for w in program.idle_windows)
    assert program.duration == plain.duration
    host = _events(plain)
    assert [event for event in _events(program) if event in host] == host
    # Two qudits fit.
    program = _filled(
        many, Q[0], Q[1], Q[2], Q[3], Q[4], fill=pair, schedule=schedule, qudits=[Q[0], Q[1]]
    )
    assert [w.qudit for w in program.idle_windows] == [0, 1]


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("n", [3, 9, 1000])
def test_a_window_in_a_loop_is_filled_in_every_round_and_the_loop_stays_rolled(schedule, n):
    program = _filled(rounds, Q[0], Q[1], n, schedule=schedule)
    (window,) = program.idle_windows
    assert (window.start, window.end, window.rounds, window.filled) == (40, 640, n, True)
    plain = pulse.compile(rounds, schedule=schedule).lower(Q[0], Q[1], n)
    assert program.duration == plain.duration
    if n > 8:
        # Rolled: as many ops for a thousand rounds as for nine.
        assert program.metrics.op_count == _filled(rounds, Q[0], Q[1], 9).metrics.op_count
    else:
        fills = [e for e in _events(program) if e[0] == "drive" and e[3] - e[2] == 40 and e[1] == 0]
        assert len(fills) == 3 * n


@pulse.kernel
def memory(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))  # a idles until its drive of the next round


@pulse.kernel
def memory_then_gate(a, b, n: int, last: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))
    drive(da, square(last, 0.1))


@pulse.kernel
def tone_first(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(db, square(100, 0.1), tone=da.tone)
        drive(da, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))


@pytest.mark.parametrize("n", [3, 9, 1000])
def test_a_window_across_the_end_of_a_loop_iteration_is_filled_in_every_round(n):
    # Under alap a's drive of each round waits for the round before: a
    # idles from its drive to its drive of the next round, and after the
    # last round as long again, to the program's end.
    program = _filled(memory, Q[0], Q[1], n, schedule="alap")
    (window,) = program.idle_windows
    assert (window.start, window.end, window.rounds, window.filled) == (40, 600, n, True)
    assert window.before.startswith("drive() at line ")
    plain = pulse.compile(memory, schedule="alap").lower(Q[0], Q[1], n)
    assert program.duration == plain.duration
    if n == 3:
        host = _events(plain)
        written = [event for event in _events(program) if event not in host]
        assert written == [
            (kind, 0, 600 * k + a, 600 * k + b)
            for k in range(3)
            for kind, a, b in (("wait", 40, 300), ("drive", 300, 340), ("wait", 340, 600))
        ] + [("wait", 0, 1800, 1840)]
    else:
        assert (
            program.metrics.op_count
            == _filled(memory, Q[0], Q[1], 9, schedule="alap").metrics.op_count
        )


def test_a_window_to_the_programs_end_is_written_after_the_kernels_last_statement():
    # a's last operation is inside the loop; b plays on after it.
    @pulse.kernel
    def tail(a, b, n: int):
        da = get_drive_line(a)
        db = get_drive_line(b)
        for _ in range(n):
            drive(da, square(40, 0.1))
            drive(db, square(400, 0.2))

    program = _kept(tail, Q[0], Q[1], 20, trailing=True)
    (window,) = program.idle_windows
    assert (window.start, window.end, window.before, window.filled) == (800, 8000, None, True)


def test_a_loops_window_needs_as_much_room_after_the_last_round():
    program = _kept(memory_then_gate, Q[0], Q[1], 20, 40, schedule="alap")
    assert [(w.start, w.end, w.rounds, w.filled) for w in program.idle_windows] == [
        (40, 600, 20, True)
    ]
    # A longer gate after the loop starts sooner: the last round has less room.
    program = _kept(memory_then_gate, Q[0], Q[1], 20, 200, schedule="alap")
    window = program.idle_windows[0]
    assert (window.rounds, window.filled) == (19, False)
    assert window.result == "left alone: there is less room for it after the loop's last round"


def test_a_loops_window_with_the_tone_playing_at_the_start_of_the_next_round_is_left_alone():
    program = _kept(tone_first, Q[0], Q[1], 20, schedule="alap", threshold=60)
    assert {w.result for w in program.idle_windows if not w.filled} >= {
        "left alone: its tone plays or changes at the start of the next round"
    }


@pulse.kernel
def barrier_of_pairs(a, b, c):
    la = get_drive_line(a)
    lb = get_drive_line(b)
    lc = get_drive_line(c)
    drive(la, square(40, 0.1))
    drive(lb, square(300, 0.1))
    for one, other in ((la, lb), (lb, lc), (lb, lc), (la, lb)):
        sync(one, other)
    drive(lb, square(60, 0.1))
    drive(la, square(40, 0.1), tone=lc.tone)
    drive(la, square(40, 0.1))


def test_a_gap_is_cut_where_a_sync_of_its_line_meets():
    # a idles from 40 to the barrier at 300; c, with no operation of its
    # own, from the start to the barrier, then its tone plays on a's line.
    program = _kept(barrier_of_pairs, Q[0], Q[1], Q[2], leading=True)
    assert [(w.qudit, w.start, w.end, w.filled) for w in program.idle_windows] == [
        (0, 40, 300, True),
        (2, 0, 300, True),
    ]
    assert [e for e in _events(program) if e[1] == 2] == [
        ("wait", 2, 0, 130),
        ("drive", 2, 130, 170),
        ("wait", 2, 170, 300),
        ("wait", 2, 300, 380),  # the rest of c's gap, to the program's end
    ]


def _surface_code_cycle():
    """The kernel of examples/transmon/transmon_surface17_surface_code_cycle.py
    and its qudits."""
    import contextlib
    import importlib.util
    import io

    path = Path(__file__).parent.parent / "examples/transmon"
    spec = importlib.util.spec_from_file_location(
        "surface_code_cycle_example", path / "transmon_surface17_surface_code_cycle.py"
    )
    example = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(example)
    return example.surface_code_cycle, example.qs.qudits[: example.qudit_count]


@pulse.kernel
def calibrated_echo(qudit, window):
    line = get_drive_line(qudit)
    drive(line, line.get_calibration(pulse.gates.X))


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_the_gaps_of_the_surface_code_cycle_are_written_through_its_barriers(schedule):
    # The example's barriers are chains of syncs of two lines. A sync is no
    # time of its own, so a gap runs from one operation of its line to the
    # next and is written whole: a part before a sync left unwritten would
    # let what is written slide, under alap, towards the operation after.
    kernel, qudits = _surface_code_cycle()
    rule = pulse.IdleReplacement(calibrated_echo, threshold=50 * pulse.ns)
    program = pulse.compile(kernel, schedule=schedule).replace_idle(rule).lower(qudits)
    plain = pulse.compile(kernel, schedule=schedule).lower(qudits)
    host = _events(plain)
    assert [event for event in _events(program) if event in host] == host
    assert program.duration == plain.duration
    windows = program.idle_windows
    assert (len(windows), sum(w.filled for w in windows)) == (88, 46)
    assert {w.cycles for w in windows if w.filled} == {196}


@pulse.kernel(schedule="asap")
def layer(lines):
    for line in lines:
        drive(line, square(40, 0.1))


@pulse.kernel(schedule="alap")
def measured(drive_line, readout_line):
    sync(drive_line, readout_line)
    result = readout(readout_line, square(800, 0.05))
    if result:
        drive(drive_line, square(40, 0.3))
    return result


@pulse.kernel
def cycle(qudits, n: int):
    ancilla = get_drive_line(qudits[0])
    readout_line = get_readout_line(qudits[0])
    lines = [get_drive_line(q) for q in qudits[1:]]
    for _ in range(n):
        layer(lines)
        for line in lines:
            drive(ancilla, square(200, 0.05), tone=line.tone)
        layer(lines)
        measured(ancilla, readout_line)


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("n", [2, 50])
def test_the_data_qudits_of_a_stabilizer_cycle_are_filled_between_its_layers(flatten, n):
    # The layers are kernels with a schedule of their own; each data qudit
    # idles from the pulse at its tone to the next layer, 40 cycles longer
    # in the rounds after the first. The fill takes the shortest and is
    # written after the pulse at the tone, around the calls of the layers.
    rule = pulse.IdleReplacement(pair, threshold=100)
    register = Q[0:4]
    compiled = pulse.compile(cycle, flatten_schedules=flatten).replace_idle(rule)
    program = compiled.lower(register, n)
    plain = pulse.compile(cycle, flatten_schedules=flatten).lower(register, n)
    assert program.duration == plain.duration
    filled = [(w.qudit, w.cycles, w.rounds) for w in program.idle_windows if w.filled]
    assert filled == [
        (1, 1160, n),
        (2, 1000 if flatten else 960, n),
        (3, 800 if flatten else 760, n),
    ]
    if n == 2:
        host = _events(plain)
        assert [event for event in _events(program) if event in host] == host


def test_a_fill_may_hold_a_loop():
    program = _kept(host, Q[0], Q[1], fill=train)
    pulses = [e for e in _events(program) if e[1] == 0 and e[0] == "drive" and 40 <= e[2] < 640]
    assert pulses == [("drive", 0, 45 + 50 * k, 85 + 50 * k) for k in range(12)]


def test_a_gap_that_ends_inside_a_loop_is_written_after_the_operation_before_it():
    program = _kept(into_a_loop, Q[0], Q[1], 20)
    first = program.idle_windows[0]
    assert (first.start, first.end, first.rounds, first.filled) == (40, 600, 1, True)
    assert _events(program)[:5] == [
        ("drive", 0, 0, 40),
        ("wait", 0, 40, 300),
        ("drive", 0, 300, 340),
        ("wait", 0, 340, 600),
        ("drive", 1, 0, 600),
    ]


def test_a_window_before_waits_the_passes_merge_is_written_before_the_first():
    passes = (*DEFAULT_PASSES, pulse.Pass.MERGE_WAITS)
    program = _filled(waits_in_a_row, Q[0], Q[1], schedule="alap", passes=passes, leading=True)
    assert _events(program)[:4] == [
        ("wait", 0, 0, 210),
        ("drive", 0, 210, 250),
        ("wait", 0, 250, 460),
        ("wait", 0, 460, 560),
    ]


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize(
    ("kernel", "args", "settings"),
    [
        (host, (Q[0], Q[1]), {}),
        (crossed, (Q[0], Q[1]), {"leading": True, "trailing": True}),
        (own_readout, (Q[0],), {"leading": True}),
        (before_a_branch, (Q[0], Q[1]), {"leading": True}),
        (middle_tone, (Q[0], Q[1]), {"threshold": 60, "leading": True}),
        (many, (Q[0], Q[1], Q[2], Q[3], Q[4]), {}),
        (rounds, (Q[0], Q[1], 200), {}),
        (into_a_loop, (Q[0], Q[1], 20), {"trailing": True}),
        (memory, (Q[0], Q[1], 20), {"trailing": True}),
        (memory_then_gate, (Q[0], Q[1], 20, 40), {"trailing": True}),
        (memory_then_gate, (Q[0], Q[1], 6, 200), {}),
        (tone_first, (Q[0], Q[1], 20), {"threshold": 60, "leading": True, "trailing": True}),
        (uneven, (Q[0], Q[1], 6), {"leading": True, "trailing": True}),
    ],
)
def test_waits_as_long_as_the_windows_change_no_schedule(
    monkeypatch, schedule, kernel, args, settings
):
    # Each window written as one wait: what finding the windows must get right.
    import cudaq_pulse._idle as idle

    monkeypatch.setattr(idle, "_PADS", True)
    program = _kept(kernel, *args, schedule=schedule, **settings)
    assert all(not w.filled or w.result == "filled by a wait" for w in program.idle_windows)


def test_a_program_with_filled_windows_pickles_as_the_program_it_is():
    import pickle

    program = _filled(host, Q[0], Q[1])
    again = pickle.loads(pickle.dumps(program))
    assert _events(again) == _events(program)
    assert again.idle_windows == program.idle_windows
    # The program carries the options it was compiled with, not the rule.
    assert (
        again.options
        == program.options
        == pulse.compile(host, schedule="asap").lower(Q[0], Q[1]).options
    )


def test_a_program_filled_by_a_kernel_defined_in_a_function_pickles():
    import pickle

    @pulse.kernel
    def local(qudit, window):
        drive(get_drive_line(qudit), gaussian(40, 0.2, 10))

    program = _filled(host, Q[0], Q[1], fill=local)
    again = pickle.loads(pickle.dumps(program))
    assert again.idle_windows[0].result == "filled by local()"


def test_a_rule_pickles_with_its_fill():
    import pickle

    rule = pulse.IdleReplacement(pi, threshold=100, align="end")
    again = pickle.loads(pickle.dumps(rule))
    assert again == rule and again.fill is pi


_PULSE = {"cycles": 40}


@pulse.kernel
def from_a_global(qudit, window):
    drive(get_drive_line(qudit), gaussian(_PULSE["cycles"], 0.2, 10))


def test_a_change_of_what_the_fill_reads_compiles_the_program_again(monkeypatch):
    compiled = pulse.compile(host, schedule="asap").replace_idle(
        pulse.IdleReplacement(from_a_global, threshold=100)
    )
    assert ("drive", 0, 320, 360) in _events(compiled.lower(Q[0], Q[1]))
    monkeypatch.setitem(_PULSE, "cycles", 80)
    assert ("drive", 0, 300, 380) in _events(compiled.lower(Q[0], Q[1]))


def test_a_float_argument_of_the_kernel_stays_an_argument_of_the_filled_program():
    @pulse.kernel
    def scaled(a, b, amp: float):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, square(40, amp))
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, amp))

    compiled = pulse.compile(scaled, schedule="asap").replace_idle(
        pulse.IdleReplacement(pi, threshold=100)
    )
    one, other = compiled.lower(Q[0], Q[1], 0.1), compiled.lower(Q[0], Q[1], 0.2)
    assert one.generic is other.generic
    assert one.generic.parameters == ("amp",)
    assert ("drive", 0, 300, 340) in _events(other)
    assert other.idle_windows[0].filled


def test_a_kernel_that_traces_differently_each_time_is_refused():
    import random

    @pulse.kernel
    def dice(a, b):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, square(40, 0.1 * random.random()))
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))

    with pytest.raises(pulse.UnsupportedError, match="replacing its idle windows traces it twice"):
        _filled(dice, Q[0], Q[1])


# -- What a review of the first version found --------------------------------


def _executions(program):
    """Every execution of every timed op, loops in full: (kind, line, tone,
    start, end), and the lines as (role, qudit)."""
    lines, tree = program._module.site_tree()
    found = []

    def walk(nodes, iterations):
        for node in nodes:
            if node[0] == "for":
                for k in range(node[1]):
                    walk(node[2], (*iterations, k))
            elif node[0] == "if":
                walk(node[1], iterations)
                walk(node[2], iterations)
            elif node[0] not in ("sync", "answer"):
                _, _, line, tone, start, length, shifts, _ = node
                at = start + sum(k * shift for k, shift in zip(iterations, shifts, strict=True))
                found.append((node[0], line, tone, at, at + length))

    walk(tree, ())
    return [(str(role), int(qudit)) for role, qudit in lines], found


def _pulses_on_a_busy_qudit(plain, filled):
    """The pulses *filled* has and *plain* has not that play while their
    qudit's own readout runs or its tone plays from another line."""
    lines, before = _executions(plain)
    _, after = _executions(filled)
    readout_of = {qudit: i for i, (role, qudit) in enumerate(lines) if role == "readout"}
    host = set(before)
    wrong = []
    for kind, line, _, start, end in after:
        if kind != "drive" or (kind, line, line, start, end) in host:
            continue
        qudit = lines[line][1]
        for other, on, tone, a, b in after:
            own_readout = other == "readout" and on == readout_of.get(qudit)
            own_tone = other in ("drive", "readout") and tone == line and on != line
            if (own_readout or own_tone) and start < b and a < end:
                wrong.append((start, end, other, a, b))
    return wrong


@pulse.kernel
def read_every_round(a, b, n: int):
    da = get_drive_line(a)
    ra = get_readout_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    sync(da, ra, db)
    for _ in range(n):
        readout(ra, square(100, 0.05))
        wait(ra, 400)
        drive(db, square(500, 0.1))
    sync(da, ra, db)
    drive(da, square(40, 0.1))


@pulse.kernel
def readouts_behind(a, b, n: int):
    da = get_drive_line(a)
    ra = get_readout_line(a)
    db = get_drive_line(b)
    wait(ra, 2000)
    for _ in range(n):
        drive(da, square(40, 0.1))
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))
        readout(ra, square(300, 0.05))


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_gap_through_a_loop_is_filled_between_the_qudits_readouts_of_every_round(schedule):
    # a's line is free from its first drive to its last, and a is read out
    # in each of the eight rounds in between: every readout is laid over
    # the gap, not only those of the two rounds a loop is read by.
    program = _filled(read_every_round, Q[0], Q[1], 8, fill=pair, schedule=schedule)
    plain = pulse.compile(read_every_round, schedule=schedule).lower(Q[0], Q[1], 8)
    assert program.duration == plain.duration
    assert [(w.start, w.end, w.filled) for w in program.idle_windows] == [
        (140 + 500 * k, 540 + 500 * k, True) for k in range(8)
    ]
    assert _pulses_on_a_busy_qudit(plain, program) == []


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_gap_through_a_loop_of_too_many_readouts_to_list_is_left_alone(schedule):
    program = _filled(read_every_round, Q[0], Q[1], 1000, fill=pair, schedule=schedule)
    plain = pulse.compile(read_every_round, schedule=schedule).lower(Q[0], Q[1], 1000)
    assert program.mlir() == plain.mlir()
    (window,) = program.idle_windows
    assert (window.start, window.end, window.filled) == (499640, 500040, False)
    assert window.result == (
        "left alone: a loop reads the qudit out or plays its tone in every round meanwhile"
    )


@pulse.kernel
def synced_every_round(a, b, n: int):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    for _ in range(n):
        sync(da, rb)
        readout(rb, square(400, 0.05))
        sync(da, rb)
    drive(da, square(40, 0.1))


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("n", [1, 2, 4, 30])
def test_a_window_bounded_by_syncs_in_a_loop_is_filled_in_every_round(schedule, n):
    # a's line has no op in the loop, only syncs with b's readout line:
    # each round's stretch between them is a window of every round, as
    # between ops, and the loop stays rolled.
    program = _filled(synced_every_round, Q[0], Q[1], n, fill=pair, schedule=schedule)
    plain = pulse.compile(synced_every_round, schedule=schedule).lower(Q[0], Q[1], n)
    assert all(w.filled for w in program.idle_windows)
    assert sum(w.rounds for w in program.idle_windows) == n
    assert program.duration == plain.duration
    lines, before = _executions(plain)
    _, after = _executions(program)
    host = set(before)
    assert [e for e in after if e in host] == before
    written = [e for e in after if e not in host]
    assert all(lines[line] == ("drive", 0) for _, line, *_ in written)
    fills = sorted(e for e in written if e[0] == "drive")
    readouts = sorted(
        (start, end) for kind, line, _, start, end in before if lines[line] == ("readout", 1)
    )
    assert len(readouts) == n
    rounds = [
        [(f[3] - start, f[4] - start) for f in fills if start <= f[3] and f[4] <= end]
        for start, end in readouts
    ]
    assert len(fills) == 2 * n
    # Every round, rolled or not, as the single round of an unrolled loop.
    one = _filled(synced_every_round, Q[0], Q[1], 1, fill=pair, schedule=schedule)
    _, once = _executions(one)
    first = min(e[3] for e in once if lines[e[1]] == ("readout", 1))
    alone = sorted((e[3] - first, e[4] - first) for e in once if e[0] == "drive" and e[3] >= first)
    alone = [span for span in alone if span[1] <= 400]
    assert len(alone) == 2
    assert rounds == [alone] * n
    if n == 30:
        # Rolled: as many ops for thirty rounds as for four.
        four = _filled(synced_every_round, Q[0], Q[1], 4, fill=pair, schedule=schedule)
        assert program.metrics.op_count == four.metrics.op_count


def _kept_in_place(plain, program):
    """The executions *program* adds to *plain*'s, which keeps its own in
    order and its length; all of them on the drive line of qudit 0."""
    assert program.duration == plain.duration
    lines, before = _executions(plain)
    _, after = _executions(program)
    host = set(before)
    assert [e for e in after if e in host] == before
    written = [e for e in after if e not in host]
    assert all(lines[line] == ("drive", 0) for _, line, *_ in written)
    return lines, before, sorted(e for e in written if e[0] == "drive")


@pulse.kernel
def synced_to_the_end(a, b, n: int):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    for _ in range(n):
        sync(da, rb)
        readout(rb, square(400, 0.05))
        sync(da, rb)


@pytest.mark.parametrize("n", [1, 4, 30])
def test_windows_bounded_by_syncs_after_the_qudits_last_operation_are_trailing(n):
    program = _filled(synced_to_the_end, Q[0], Q[1], n, fill=pair)
    assert program.idle_windows
    assert {w.result for w in program.idle_windows} == {
        "left alone: after the qudit's last operation (trailing=True fills it)"
    }
    program = _filled(synced_to_the_end, Q[0], Q[1], n, fill=pair, trailing=True)
    assert all(w.filled for w in program.idle_windows)
    plain = pulse.compile(synced_to_the_end).lower(Q[0], Q[1], n)
    _kept_in_place(plain, program)


@pulse.kernel
def synced_with_a_gap_between_rounds(a, b, n: int):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    for _ in range(n):
        sync(da, rb)
        readout(rb, square(400, 0.05))
        sync(da, rb)
        readout(rb, square(200, 0.05))
    sync(da, rb)
    drive(da, square(40, 0.1))


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("n", [2, 4, 30])
def test_a_window_from_a_sync_into_the_next_round_is_filled_and_once_after_the_loop(schedule, n):
    # From the second sync of a round to the first of the next, a idles
    # through b's short readout: across the end of an iteration, written
    # at the end of the body and once more after the loop, before the sync
    # that ends it there.
    program = _filled(synced_with_a_gap_between_rounds, Q[0], Q[1], n, fill=pair, schedule=schedule)
    plain = pulse.compile(synced_with_a_gap_between_rounds, schedule=schedule).lower(Q[0], Q[1], n)
    lines, before, fills = _kept_in_place(plain, program)
    windows = [(w.cycles, w.rounds, w.filled) for w in program.idle_windows]
    if schedule == "alap_rcp":
        from cudaq_pulse._idle_windows import LANES

        assert all(w.filled or w.result == LANES for w in program.idle_windows)
    else:
        assert windows == [(400, n, True), (200, n, True)]
        readouts = [(a, b) for kind, line, _, a, b in before if lines[line] == ("readout", 1)]
        assert len(readouts) == 2 * n
        for start, end in readouts:
            assert len([f for f in fills if start <= f[3] and f[4] <= end]) == 2


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_window_from_a_sync_into_the_next_round_without_room_after_the_loop_says_so(schedule):
    @pulse.kernel
    def no_room(a, b, n: int):
        da = get_drive_line(a)
        rb = get_readout_line(b)
        drive(da, square(40, 0.1))
        for _ in range(n):
            sync(da, rb)
            readout(rb, square(400, 0.05))
            sync(da, rb)
            readout(rb, square(200, 0.05))
        drive(da, square(40, 0.1))

    program = _filled(no_room, Q[0], Q[1], 4, fill=pair, schedule=schedule)
    plain = pulse.compile(no_room, schedule=schedule).lower(Q[0], Q[1], 4)
    _kept_in_place(plain, program)
    results = {(w.cycles, w.rounds): w.result for w in program.idle_windows}
    assert results[(400, 4)] == "filled by pair()"
    assert results[(200, 3)] == "left alone: there is less room for it after the loop's last round"


@pulse.kernel
def a_loop_then_synced_rounds(a, b, n: int):
    da = get_drive_line(a)
    rb = get_readout_line(b)
    for _ in range(n):
        readout(rb, square(400, 0.05))
    drive(da, square(40, 0.1))
    for _ in range(n):
        sync(da, rb)
        readout(rb, square(300, 0.05))
        sync(da, rb)
    drive(da, square(40, 0.1))


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("n", [3, 30])
def test_syncs_in_a_loop_after_another_loop_of_their_lines_keep_the_program(schedule, n):
    program = _filled(a_loop_then_synced_rounds, Q[0], Q[1], n, fill=pair, schedule=schedule)
    plain = pulse.compile(a_loop_then_synced_rounds, schedule=schedule).lower(Q[0], Q[1], n)
    _kept_in_place(plain, program)
    assert any(w.filled and w.rounds == n for w in program.idle_windows)


@pulse.kernel
def read_out_beside_synced_rounds(a, b, n: int):
    da = get_drive_line(a)
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    wait(ra, 1000)
    readout(ra, square(400, 0.05))  # during the third and fourth rounds
    for _ in range(n):
        sync(da, rb)
        readout(rb, square(400, 0.05))
        sync(da, rb)
    drive(da, square(40, 0.1))


@pulse.kernel
def read_out_before_trailing_rounds(a, b, n: int):
    da = get_drive_line(a)
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    wait(ra, 150)
    readout(ra, square(80, 0.05))  # during the first round
    for _ in range(n):
        sync(da, rb)
        readout(rb, square(100, 0.05))
        sync(da, rb)
        readout(rb, square(100, 0.05))


@pulse.kernel
def read_out_then_synced_rounds(a, b, n: int):
    da = get_drive_line(a)
    ra = get_readout_line(a)
    rb = get_readout_line(b)
    drive(da, square(40, 0.1))
    readout(ra, square(400, 0.05))
    sync(da, ra)  # the readout ends before the loop
    wait(da, 100)
    for _ in range(n):
        sync(da, rb)
        readout(rb, square(400, 0.05))
        sync(da, rb)
    drive(da, square(40, 0.1))


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("n", [4, 30])
def test_a_readout_of_the_qudit_before_a_loop_of_syncs_leaves_its_rounds_filled(schedule, n):
    program = _filled(read_out_then_synced_rounds, Q[0], Q[1], n, fill=pair, schedule=schedule)
    plain = pulse.compile(read_out_then_synced_rounds, schedule=schedule).lower(Q[0], Q[1], n)
    _kept_in_place(plain, program)
    assert [(w.cycles, w.rounds, w.filled) for w in program.idle_windows if w.rounds == n] == [
        (400, n, True)
    ]


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("kernel", [read_out_beside_synced_rounds, read_out_before_trailing_rounds])
def test_syncs_in_a_loop_while_the_qudit_is_read_out_outside_it_cut_no_window(schedule, kernel):
    # A window of every round there would be laid over the readout in its
    # first round only, and one before the readout is not after the qudit's
    # last operation in every round: the windows are left alone.
    program = _filled(kernel, Q[0], Q[1], 4, fill=pi, schedule=schedule, threshold=60)
    plain = pulse.compile(kernel, schedule=schedule).lower(Q[0], Q[1], 4)
    assert program.mlir() == plain.mlir()
    assert program.idle_windows
    assert not any(w.filled for w in program.idle_windows)


@pulse.kernel
def own_op_after_the_window(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    for _ in range(n):
        sync(da, db)
        drive(db, square(1000, 0.1))  # a idles here in every round
        sync(da, db)
        drive(da, square(4, 0.1))  # and plays after it


@pulse.kernel
def one(qudit, window):
    drive(get_drive_line(qudit), gaussian(40, 0.2, 10))


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("align", ["center", "start"])
@pytest.mark.parametrize("n", [1, 2, 3, 12])
def test_the_windows_report_the_fills_the_program_plays(schedule, align, n):
    # a's line has an op of its own in the loop, after the window: the
    # syncs before it cut no gap, and each window says what the program
    # does there, in every round.
    program = _filled(
        own_op_after_the_window, Q[0], Q[1], n, fill=one, schedule=schedule, align=align
    )
    plain = pulse.compile(own_op_after_the_window, schedule=schedule).lower(Q[0], Q[1], n)
    _, before = _executions(plain)
    lines, after = _executions(program)
    host = set(before)
    fills = [e for e in after if e not in host and e[0] == "drive" and lines[e[1]] == ("drive", 0)]
    reported = sum(w.rounds for w in program.idle_windows if w.qudit == 0 and w.filled)
    assert len(fills) == reported
    # The window between the two syncs is filled in every round.
    assert reported == n


@pulse.kernel
def own_op_at_the_round_end(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(n):
        sync(da, db)
        drive(da, square(4, 0.1))  # asap: at the round's start; alap: after a idles
        drive(db, square(1004, 0.1))
    sync(da, db)
    drive(da, square(4, 0.1))


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("leading", [False, True])
@pytest.mark.parametrize("n", [2, 3, 12])
def test_a_window_before_the_qudits_op_in_every_round_is_filled_in_every_round(
    schedule, leading, n
):
    # Under alap the qudit's op of a round comes after its window: the
    # window is in the loop's body in every round, the first one too, so it
    # is not the qudit's leading window and is filled as under asap.
    kernel = own_op_at_the_round_end
    program = _filled(kernel, Q[0], Q[1], n, fill=one, schedule=schedule, leading=leading)
    plain = pulse.compile(kernel, schedule=schedule).lower(Q[0], Q[1], n)
    _, _, fills = _kept_in_place(plain, program)
    reported = sum(w.rounds for w in program.idle_windows if w.qudit == 0 and w.filled)
    assert len(fills) == reported == n


@pulse.kernel
def nothing(qudit, window):
    pass


@pulse.kernel
def only_waits(qudit, window):
    wait(get_drive_line(qudit), window)


@pytest.mark.parametrize("schedule", _POLICIES)
@pytest.mark.parametrize("fill", [nothing, only_waits])
def test_a_window_a_fill_plays_no_pulse_in_is_left_alone_and_says_so(schedule, fill):
    program = _filled(rounds, Q[0], Q[1], 3, fill=fill, schedule=schedule)
    plain = pulse.compile(rounds, schedule=schedule).lower(Q[0], Q[1], 3)
    assert program.mlir() == plain.mlir()
    (window,) = program.idle_windows
    assert not window.filled
    assert window.result == (
        f"left alone: {fill.__name__}() plays no pulse in a window of {window.cycles} cycles"
    )


@pulse.kernel
def pulses_every_400(qudit, window):
    line = get_drive_line(qudit)
    for _ in range(window // 400):
        drive(line, gaussian(40, 0.2, 10))


def test_a_fill_whose_pulses_are_in_a_loop_of_no_rounds_plays_no_pulse():
    # In the 600-cycle window the fill's loop runs once; in a 260-cycle one
    # it runs none.
    program = _filled(rounds, Q[0], Q[1], 3, fill=pulses_every_400)
    assert program.idle_windows
    assert all(w.filled for w in program.idle_windows)

    @pulse.kernel
    def short_rounds(a, b, n: int):
        da = get_drive_line(a)
        db = get_drive_line(b)
        for _ in range(n):
            drive(da, square(40, 0.1))
            drive(db, square(300, 0.1))
            sync(da, db)

    program = _filled(short_rounds, Q[0], Q[1], 3, fill=pulses_every_400)
    plain = pulse.compile(short_rounds, schedule="asap").lower(Q[0], Q[1], 3)
    assert program.mlir() == plain.mlir()
    assert {w.result for w in program.idle_windows} == {
        "left alone: pulses_every_400() plays no pulse in a window of 260 cycles"
    }


@pulse.kernel
def first_op_after_a_loop(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    for _ in range(n):
        sync(da, db)
        drive(db, square(1004, 0.1))
        sync(da, db)
    drive(da, square(4, 0.1))


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize("leading", [False, True])
def test_a_loop_the_qudit_sits_out_before_its_first_operation_is_leading(schedule, leading):
    # The qudit has no op of its own in the loop: its windows there come
    # before its first operation in every round, so leading= decides.
    kernel = first_op_after_a_loop
    program = _filled(kernel, Q[0], Q[1], 4, fill=one, schedule=schedule, leading=leading)
    plain = pulse.compile(kernel, schedule=schedule).lower(Q[0], Q[1], 4)
    _, _, fills = _kept_in_place(plain, program)
    loop = [w for w in program.idle_windows if w.cycles >= 1000]
    assert [(w.rounds, w.filled) for w in loop] == [(4, leading)]
    assert len(fills) == (4 if leading else 0)
    if not leading:
        assert loop[0].result == (
            "left alone: before the qudit's first operation (leading=True fills it)"
        )


def test_a_fill_that_runs_other_than_its_windows_report_is_an_internal_error(monkeypatch):
    # The check behind the test above: each written fill runs as many
    # times as the windows it fills report, whatever the kernel.
    from cudaq_pulse._errors import InternalError
    from cudaq_pulse._idle_windows import Plan

    add = Plan.add

    def under(self, where, site, qudit, items, scope, runs=0):
        add(self, where, site, qudit, items, scope, runs and runs + 1)

    monkeypatch.setattr(Plan, "add", under)
    with pytest.raises(InternalError, match="runs 1 times where the windows it fills report 2"):
        _filled(rounds, Q[0], Q[1], 1)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize("n", [3, 9, 12])
def test_no_fill_plays_while_readouts_of_other_rounds_run(schedule, n):
    # a's readouts start late and repeat faster than its drives: the readout
    # that runs during a round's gap is one of another round.
    program = _filled(readouts_behind, Q[0], Q[1], n, schedule=schedule, trailing=True)
    plain = pulse.compile(readouts_behind, schedule=schedule).lower(Q[0], Q[1], n)
    assert _pulses_on_a_busy_qudit(plain, program) == []
    assert program.duration == plain.duration


def test_the_quantum_systems_control_system_is_the_one_compiled_for():
    from cudaq_pulse.controller import DriveLine, ReadoutLine

    from tests._handles import ladder_target

    box = pulse.Controller(
        drive_lines=[DriveLine(q) for q in range(4)],
        readout_lines=[ReadoutLine(q) for q in range(4)],
    )
    qpu = ladder_target(
        4, control_system=pulse.ControlSystem(controllers=[box], max_concurrent_drives=1)
    )
    q = qpu.qudits
    # (A fill under "alap" is not here: with one drive at a time, the plain
    # program's two drives already do not fit under "alap".)
    for fill, schedule in ((None, "alap_rcp"), (None, "rcp")):
        plain = pulse.compile(two_lines, schedule=schedule).lower(q[0], q[1])
        rule = pulse.IdleReplacement(fill, threshold=100, leading=True)
        program = pulse.compile(two_lines, schedule=schedule).replace_idle(rule).lower(q[0], q[1])
        assert program.options.schedule_args() == plain.options.schedule_args()
        assert program.options.schedule_args()[1] == 1
        assert program.options.machine == plain.options.machine
        if fill is None:
            assert program.export_to_controller() == plain.export_to_controller()
    # One drive lane: the fill's pulse does not fit beside b's, so its
    # window is left alone and says why.
    from cudaq_pulse._idle_windows import LANES

    program = (
        pulse.compile(two_lines, schedule="alap_rcp")
        .replace_idle(pulse.IdleReplacement(pi, threshold=100, leading=True))
        .lower(q[0], q[1])
    )
    assert [w.result for w in program.idle_windows] == [LANES]


@pulse.kernel
def three_lines(a, b, c):
    da = get_drive_line(a)
    db = get_drive_line(b)
    dc = get_drive_line(c)
    drive(da, square(40, 0.1))
    sync(da, db, dc)
    drive(db, square(400, 0.1))
    drive(dc, square(400, 0.1))
    sync(da, db, dc)
    drive(da, square(40, 0.1))


@pytest.mark.parametrize("schedule", ["alap", "asap"])
def test_a_fill_past_the_systems_drive_limit_fails_and_names_it(schedule):
    from cudaq_pulse.controller import ControllerLimitError, DriveLine, ReadoutLine

    from tests._handles import ladder_target

    box = pulse.Controller(
        drive_lines=[DriveLine(q) for q in range(4)],
        readout_lines=[ReadoutLine(q) for q in range(4)],
    )
    qpu = ladder_target(
        4, control_system=pulse.ControlSystem(controllers=[box], max_concurrent_drives=2)
    )
    q = qpu.qudits
    rule = pulse.IdleReplacement(pi, threshold=100)
    # Two lines play at once without the fill; its pulse on a's line makes three.
    plain = pulse.compile(three_lines, schedule=schedule).lower(q[0], q[1], q[2])
    assert float(plain.duration) > 0
    with pytest.raises(ControllerLimitError) as raised:
        pulse.compile(three_lines, schedule=schedule).replace_idle(rule).lower(q[0], q[1], q[2])
    text = str(raised.value)
    assert text.startswith(
        "idle replacement: with the idle windows of three_lines() filled by pi(), "
        "the program does not fit the control system:"
    )
    assert "max_concurrent_drives" in text
    assert text.endswith(
        "; fill fewer qudits (qudits=), use a fill with fewer pulses, or raise the "
        "control system's limit"
    )


@pulse.kernel
def three_at_once(a, b, c):
    da = get_drive_line(a)
    db = get_drive_line(b)
    dc = get_drive_line(c)
    drive(da, square(40, 0.1))
    drive(db, square(40, 0.1))
    drive(dc, square(40, 0.1))
    sync(da, db, dc)
    drive(da, square(400, 0.1))
    sync(da, db, dc)
    drive(db, square(40, 0.1))


@pytest.mark.parametrize(("schedule", "placed"), [("asap", "rcp"), ("alap", "alap_rcp")])
def test_a_program_past_the_drive_limit_without_fills_is_filled_as_its_rcp_schedule(
    schedule, placed
):
    from cudaq_pulse.controller import DriveLine, ReadoutLine

    from tests._handles import ladder_target

    box = pulse.Controller(
        drive_lines=[DriveLine(q) for q in range(4)],
        readout_lines=[ReadoutLine(q) for q in range(4)],
    )
    qpu = ladder_target(
        4, control_system=pulse.ControlSystem(controllers=[box], max_concurrent_drives=2)
    )
    q = qpu.qudits
    rule = pulse.IdleReplacement(pi, threshold=100)
    # Three lines play at once without the fill: the program is placed as
    # rcp (alap_rcp) places it, its windows read and filled there: b's
    # window beside a's long pulse takes the free lane, c's has none.
    filled = pulse.compile(three_at_once, schedule=schedule).replace_idle(rule).lower(*q[:3])
    as_placed = pulse.compile(three_at_once, schedule=placed).replace_idle(rule).lower(*q[:3])
    assert filled.options.schedule == placed
    assert [(w.qudit, w.filled) for w in filled.metrics.idle_windows] == [(1, True), (2, False)]
    assert [(e.kind, e.line, e.start_cycles) for e in filled.schedule] == [
        (e.kind, e.line, e.start_cycles) for e in as_placed.schedule
    ]


def test_a_float_parameter_that_sets_a_pulses_length_compiles_at_once():
    # The kernel is traced again with the parameter static: the record of
    # the first trace is not the second's.
    @pulse.kernel
    def widths(a, b, width: float):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, square(width, 0.1))
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))

    with pytest.warns(Warning):
        program = _filled(widths, Q[0], Q[1], 40.0)
    assert [(w.start, w.end, w.filled) for w in program.idle_windows] == [(40, 600, True)]


@pulse.kernel
def virtual_z_rounds(a, b, n: int):
    da = get_drive_line(a)
    db = get_drive_line(b)
    rb = get_readout_line(b)
    for _ in range(n):
        drive(da, square(40, 0.1))
        shift_phase(da.tone, 0.25)
        drive(db, square(40, 0.1))
        sync(da, db, rb)
        readout(rb, square(600, 0.05))
        sync(da, db, rb)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize("n", [3, 20])
def test_a_phase_shift_of_the_kernel_before_a_window_may_fold_into_the_fill(schedule, n):
    # The passes fold a phase shift into the pulse after it, here the
    # fill's first: the fill plays in the shifted frame, as it should.
    program = _filled(virtual_z_rounds, Q[0], Q[1], n, schedule=schedule)
    plain = pulse.compile(virtual_z_rounds, schedule=schedule).lower(Q[0], Q[1], n)
    assert program.duration == plain.duration
    assert any(w.filled for w in program.idle_windows)
    host = [e for e in _events(plain) if e[0] != "shift_phase"]
    assert [e for e in _events(program) if e in host] == host


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize("n", [3, 20])
def test_a_fill_that_holds_a_loop_leaves_the_kernels_later_loops_their_places(schedule, n):
    @pulse.kernel
    def two_stage(a, b, n: int):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, square(40, 0.1))
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))
        for _ in range(n):
            drive(da, square(40, 0.1))
            drive(db, square(600, 0.1))
            sync(da, db)
            drive(da, square(40, 0.1))

    @pulse.kernel
    def ten(qudit, window):
        line = get_drive_line(qudit)
        for _ in range(10):
            wait(line, 5)
            drive(line, gaussian(40, 0.2, 10))
            wait(line, 5)

    program = _filled(two_stage, Q[0], Q[1], n, fill=ten, schedule=schedule, leading=True)
    plain = pulse.compile(two_stage, schedule=schedule).lower(Q[0], Q[1], n)
    assert program.duration == plain.duration
    assert sum(w.filled for w in program.idle_windows) >= 2
    assert _pulses_on_a_busy_qudit(plain, program) == []


def test_two_pulses_of_a_fill_that_fuse_stay_apart_from_the_kernels():
    @pulse.kernel
    def doubled(qudit, window):
        line = get_drive_line(qudit)
        drive(line, square(20, 0.1))
        drive(line, square(20, 0.1))

    program = _kept(host, Q[0], Q[1], fill=doubled, align="start")
    assert _events(program)[3:5] == [("drive", 0, 40, 80), ("wait", 0, 80, 640)]


def test_a_fill_that_calls_a_classical_function_the_kernel_already_calls_is_refused():
    from tests.classical import _schedule_matrix as matrix

    link = next(iter(matrix.LINKS))
    _, tau = matrix.FUNCTIONS[(link, next(iter(matrix.TAKES)))]

    @pulse.kernel
    def calling(a, b):
        da = get_drive_line(a)
        db = get_drive_line(b)
        drive(da, square(40, 0.1))
        future = tau.submit(1.0)
        drive(db, square(600, 0.1))
        sync(da, db)
        drive(da, square(40, 0.1))
        return future

    @pulse.kernel
    def submits(qudit, window):
        tau.submit(2.0)
        drive(get_drive_line(qudit), gaussian(40, 0.2, 10))

    with pytest.raises(pulse.CompilationError) as raised:
        pulse.compile(calling, schedule="asap", control_system=matrix.MACHINE).replace_idle(
            pulse.IdleReplacement(submits, threshold=100)
        ).lower(Q[0], Q[1])
    assert str(raised.value).startswith(
        "idle replacement: fill submits() calls a classical function; in this version"
    )


def test_qudits_of_another_quantum_system_are_refused():
    from tests._handles import ladder_target

    other = ladder_target(8)
    with pytest.raises(pulse.InvalidValueError, match="a qudit of another quantum system"):
        _filled(host, Q[0], Q[1], qudits=[other.qudits[0]])


def test_a_pulse_at_the_tone_written_before_the_gap_it_plays_in_leaves_the_gap_alone():
    # a's line only waits; its tone plays from b's line, written first.
    @pulse.kernel
    def tone_before(a, b):
        da = get_drive_line(a)
        db = get_drive_line(b)
        wait(db, 300)
        drive(db, square(100, 0.1), tone=da.tone)
        wait(da, 40)
        drive(db, square(600, 0.1))
        sync(da, db)
        wait(da, 40)

    program = _kept(tone_before, Q[0], Q[1], threshold=60)
    assert not any(w.filled for w in program.idle_windows if w.qudit == 0)


# -- A compilation without a rule pays nothing --------------------------------


def test_a_compilation_without_a_rule_is_the_program_it_always_was():
    plain = pulse.compile(two_lines).lower(Q[0], Q[1])
    assert type(plain.options) is CompileOptions
    filled = _compiled(two_lines, leading=True).lower(Q[0], Q[1])
    assert len(filled.schedule) == 6
    again = pulse.compile(two_lines).lower(Q[0], Q[1])
    assert again.idle_windows is None
    assert again.mlir() == plain.mlir()
    assert [(e.kind, e.line, e.start, e.end) for e in again.schedule] == [
        ("drive", 0, 360, 400),
        ("drive", 1, 0, 400),
        ("drive", 0, 400, 440),
    ]


_COLD = """
import sys
import cudaq_pulse as p
from cudaq_pulse.quantum_systems import PhysicalQudit

@p.kernel
def k(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    drive(da, square(40, 0.1))
    drive(db, square(400, 0.1))
    sync(da, db)
    drive(da, square(40, 0.1))

qudits = {i: PhysicalQudit(i, 5.0e9 + 1e8 * i, 50.0, 30.0) for i in range(2)}
qs = p.quantum_system(name="cold", qudits=qudits)
p.compile(k, passes=p.compiler.DEFAULT_PASSES).lower(qs.qudits[0], qs.qudits[1]).schedule
k.lower(qs.qudits[0], qs.qudits[1]).mlir()
print("cudaq_pulse._idle" in sys.modules)
"""


def test_a_compilation_without_a_rule_does_not_load_the_module(tmp_path):
    script = tmp_path / "cold.py"
    script.write_text(_COLD)
    env = {**os.environ, "PYTHONPATH": _ROOT, "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_a_pinned_kernel_with_replaced_windows_answers_as_the_compiled_kernel():
    # Pinned to a quantum system, as a compiled kernel is: a view of the
    # program needs what the compiled kernel's needs, and says the same.
    qs = pulse.quantum_system("transmon_surface17")
    compiled = pulse.compile(two_lines, quantum_system=qs, schedule="asap")
    filled = compiled.replace_idle(pulse.IdleReplacement(pi, threshold=100))
    with pytest.raises(TypeError) as plain_error:
        compiled.mlir()
    with pytest.raises(TypeError) as filled_error:
        filled.mlir()
    assert str(filled_error.value) == str(plain_error.value)
    program = filled.lower(qs.qudits[0], qs.qudits[1])
    assert program.idle_windows is not None


_FILL_AMPLITUDE = [0.2]


@pulse.kernel
def drifting(qudit, window):
    # A fill of constant length whose amplitude is drawn at trace time.
    import random

    drive(get_drive_line(qudit), square(40, 0.1 + 0.0 * random.random()))


def test_a_fill_whose_trace_can_differ_is_refused():
    rule = pulse.IdleReplacement(drifting, threshold=100)
    with pytest.raises(pulse.UnsupportedError, match=r"drifting\(\) calls a function"):
        pulse.compile(host).replace_idle(rule).lower(Q[0], Q[1])


@pulse.kernel
def reads_a_global(qudit, window):
    drive(get_drive_line(qudit), square(40, _FILL_AMPLITUDE[0]))


def test_a_program_left_unfilled_is_kept_only_while_the_fills_reads_hold():
    # One drive lane: every window is left alone for lanes. That depends on
    # the fill, so a change to what the fill reads compiles again.
    from cudaq_pulse.controller import DriveLine, ReadoutLine

    from tests._handles import ladder_target

    box = pulse.Controller(
        drive_lines=[DriveLine(q) for q in range(4)],
        readout_lines=[ReadoutLine(q) for q in range(4)],
    )
    qpu = ladder_target(
        4, control_system=pulse.ControlSystem(controllers=[box], max_concurrent_drives=1)
    )
    q = qpu.qudits
    rule = pulse.IdleReplacement(reads_a_global, threshold=100, leading=True)
    filled = pulse.compile(two_lines, schedule="alap_rcp").replace_idle(rule)
    first = filled.lower(q[0], q[1])
    assert not any(w.filled for w in first.idle_windows)
    assert filled.lower(q[0], q[1]) is first
    _FILL_AMPLITUDE[0] = 0.3
    try:
        assert filled.lower(q[0], q[1]) is not first
    finally:
        _FILL_AMPLITUDE[0] = 0.2


def test_an_int_subclass_is_a_threshold_in_cycles():
    import enum

    class Cycles(enum.IntEnum):
        HUNDRED = 100

    plain = _windows(host, Q[0], Q[1], threshold=100)
    assert _windows(host, Q[0], Q[1], threshold=Cycles.HUNDRED) == plain
    assert _windows(host, Q[0], Q[1], threshold=Cycles.HUNDRED + 0) == plain
