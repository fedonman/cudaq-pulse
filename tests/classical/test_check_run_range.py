# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A call in a loop whose executions run past the cycles a schedule counts in.

The classical check counts and orders the calls of a function from their
starts in each iteration, and keeps the calls of a long loop as one run (a
start, a period, a count). A start that does not fit 64 bits was computed
anyway, and a run whose last execution does not fit was written out one
entry per iteration, however many. Such a call is reported instead: the
check computes with times of at most 2**61 cycles either way."""

from __future__ import annotations

import re

from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline

_PAST = "the call of 'f' starts, or lasts, past the cycles a schedule counts in"


def _looped(trips, period, took=None):
    """One call and its answer, 101 cycles later, in a loop; the answer
    moves by *took* cycles an iteration (as the call, if not given)."""
    timing = f"period_cycles = array<i64: {period}>, "
    taking = f"period_cycles = array<i64: {period if took is None else took}>, "
    return f"""module attributes {{
    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>}} {{
  classical.callable @f {{dispatch_key = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}}
  func.func @test() {{
    %l0 = classical.get_link {{device = 0 : i64}} : !classical.link
    %lb = arith.constant 0 : index
    %ub = arith.constant {trips} : index
    %step = arith.constant 1 : index
    %end = scf.for %i = %lb to %ub step %step iter_args(%l = %l0) -> (!classical.link) {{
      %submitted, %h = classical.submit %l, @f()
          <{{duration_cycles = 1 : i64, {timing}start_cycles = 0 : i64}}>
          : () -> (!classical.link, !classical.future)
      %v = classical.wait %h, @f
          <{{duration_cycles = 0 : i64, {taking}start_cycles = 101 : i64}}> : f64
      scf.yield %submitted : !classical.link
    }}
    return
  }}
}}
"""


def _errors(text):
    module = MLIRPipeline().parse_to_module(text)
    try:
        module.run_passes(["pulse-classical-check"])
    except RuntimeError as error:
        return str(error)
    return ""


def test_a_long_loop_of_calls_within_the_range_is_checked_from_one_run():
    # 2**40 iterations, 200 cycles apart: the last starts near 2**48.
    assert _errors(_looped(2**40, 200)) == ""
    # All at one time: the second starts before the first ends, which is
    # found from the run.
    message = _errors(_looped(2**40, 0))
    assert "submission on link 0 starts at 0, before the one before it" in message
    # And every one of them holds a slot of the link at that time.
    assert f"link 0 holds {2**40} requests at once at 0, more than its 64 slots" in message


def test_a_loop_whose_last_call_is_past_64_bits_is_reported_not_written_out():
    # 2**40 iterations, 2**40 cycles apart: the last would start near 2**80.
    message = _errors(_looped(2**40, 2**40))
    assert (
        f"the call of 'f' starts at 0 and {2**40} cycles later in each of {2**40} iterations: "
        "the last is past the cycles a schedule counts in"
    ) in message


def test_a_call_that_starts_past_the_range_in_some_iteration_is_reported():
    # Few iterations, each walked: the second starts at 2**62.
    assert _PAST in _errors(_looped(3, 2**62))
    # The same in a loop kept as one run.
    assert _PAST in _errors(_looped(64, 2**62))
    # Within the range it is checked as any other.
    assert _errors(_looped(3, 2**60)) == ""


def test_an_answer_is_compared_with_its_call_past_64_bits():
    # Twice the period does not fit 64 bits, and the answer is taken early
    # only from the second iteration on: named with its times as they are.
    big = 2**62
    message = _errors(_looped(3, big, took=big - 1))
    found = re.search(r"is taken at (-?\d+), before it is due at (-?\d+)", message)
    assert found and (int(found.group(1)), int(found.group(2))) == (
        101 + 2 * (big - 1),
        101 + 2 * big,
    )
    assert _PAST in message


def _answered(taken):
    """A drive at 120 in a branch on an answer taken at *taken*."""
    line = "!pulse.drive_line, !pulse.tone"
    return f"""module attributes {{
    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>}} {{
  classical.callable @f {{dispatch_key = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}}
  func.func @test(%d0: !pulse.drive_line, %t0: !pulse.tone, %w: !pulse.waveform) {{
    %l = classical.get_link {{device = 0 : i64}} : !classical.link
    %l1, %h = classical.submit %l, @f() <{{duration_cycles = 1 : i64, start_cycles = 0 : i64}}>
        : () -> (!classical.link, !classical.future)
    %v = classical.wait %h, @f <{{duration_cycles = 0 : i64, start_cycles = {taken} : i64}}> : f64
    %half = arith.constant 0.5 : f64
    %c = arith.cmpf ogt, %v, %half : f64
    %d1, %t1 = scf.if %c -> ({line}) {{
      %d2, %t2 = pulse.drive %d0, %w, %t0 <{{duration_cycles = 16 : i64, start_cycles = 120 : i64}}>
          : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {line}
      scf.yield %d2, %t2 : {line}
    }} else {{
      scf.yield %d0, %t0 : {line}
    }}
    return
  }}
}}
"""


def test_an_answer_taken_past_the_range_is_reported_not_passed_over():
    # The drive in the branch starts before the answer is taken.
    early = "an operation in a branch on an answer starts at 120, before the answer is taken at"
    assert f"{early} 150" in _errors(_answered(150))
    assert f"{early} {2**61}" in _errors(_answered(2**61))
    # Past the range the two are not compared, and the answer is reported.
    message = _errors(_answered(2**61 + 10))
    assert "an operation the order of calls depends on starts, or lasts, past" in message
