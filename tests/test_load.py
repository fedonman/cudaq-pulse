# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``pulse.load``: a scheduled function another compiler built."""

import re

import cudaq_pulse as pulse
import pytest

MODULE = """
module {
  func.func private @other(%q: !pulse.qudit) attributes {pulse.schedule_state =
        #pulse.schedule_state<timed>} {
    return
  }
  func.func private @round(%a: !pulse.qudit, %b: !pulse.qudit) -> i1 attributes {made.by =
        "elsewhere", pulse.schedule_state = #pulse.schedule_state<timed>} {
    %da, %ta = pulse.get_drive_line %a {made.channel = 3 : i64} : (!pulse.qudit) ->
          (!pulse.drive_line, !pulse.tone)
    %db, %tb = pulse.get_drive_line %b : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %rb, %rt = pulse.get_readout_line %b : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 16 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %da1, %ta1 = pulse.drive %da, %w, %ta <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %db1, %tb1 = pulse.drive %db, %w, %tb <{duration_cycles = 16 : i64, start_cycles = 16 : i64}>
          : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %m = arith.constant 64 : i64
    %r = pulse.square %m, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %rb1, %rt1, %out = pulse.readout %rb, %r, %rt, iq <{duration_cycles = 64 : i64, start_cycles =
          32 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line,
          !pulse.tone, !pulse.measurement
    %bit = pulse.is_set %out : !pulse.measurement -> i1
    return %bit : i1
  }
}
"""


@pytest.fixture
def system():
    return pulse.quantum_system("transmon_surface17")


def events(program):
    return [
        (e["kind"], e["qudit"], e["start_cycles"], e["duration_cycles"])
        for e in program.export_to_controller()["events"]
    ]


def test_schedule_is_kept_and_qudits_are_bound_in_order(system):
    program = pulse.load(MODULE, entry="round", qudits=[system.qudits[2], system.qudits[0]])
    assert isinstance(program, pulse.ScheduledKernel)
    assert program.qudits == (0, 2)
    assert events(program) == [
        ("drive", 2, 0, 16),
        ("drive", 0, 16, 16),
        ("readout", 0, 32, 64),
    ]


def test_foreign_attributes_and_results_are_dropped(system):
    program = pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    text = program.mlir()
    assert "made." not in text
    assert "func.func @main()" in text
    assert program.metrics.op_count == 3
    assert "3 ops" in repr(program)


def test_lines_get_the_frequencies_of_their_qudits(system):
    program = pulse.load(MODULE, entry="round", qudits=[system.qudits[1], system.qudits[2]])
    exported = program.export_to_controller()
    drives = [e for e in exported["events"] if e["kind"] == "drive"]
    assert [e["frequency_hz"] for e in drives] == [system.frequencies[1], system.frequencies[2]]


def test_bytecode_is_accepted(system):
    from cudaq_pulse._native import _cudaq_pulse_native as native

    bytecode = native.MLIRPipeline().parse_to_module(MODULE).serialize()[0]
    assert isinstance(bytecode, bytes) and bytecode.startswith(b"ML\xefR")
    program = pulse.load(bytecode, entry="round", qudits=system.qudits[:2])
    assert events(program) == events(pulse.load(MODULE, entry="round", qudits=system.qudits[:2]))


def test_a_function_that_selects_at_run_time_is_refused(system):
    text = MODULE.replace(
        ", pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %da,",
        ", pulse.schedule_state = #pulse.schedule_state<timed>, "
        "pulse.selected_values = 1 : i64} {\n    %da,",
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="selects values at run time"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_bytes_are_accepted(system):
    program = pulse.load(MODULE.encode(), entry="round", qudits=system.qudits[:2])
    assert len(events(program)) == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda kw: kw.update(entry="missing"), "no function named missing"),
        (lambda kw: kw.update(qudits=kw["qudits"][:1]), "takes 2 qudit"),
    ],
)
def test_wrong_entry_or_count(system, change, message):
    kw = dict(entry="round", qudits=list(system.qudits[:2]))
    change(kw)
    with pytest.raises(pulse.CompilationError, match=message):
        pulse.load(MODULE, **kw)


def test_unscheduled_function_is_refused(system):
    text = MODULE.replace(
        ", pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %da,", "} {\n    %da,"
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="not scheduled"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_overlap_on_a_line_fails_verification(system):
    text = MODULE.replace("start_cycles = 16 : i64", "start_cycles = 0 : i64")
    assert text != MODULE
    text = text.replace(
        "%db1, %tb1 = pulse.drive %db, %w, %tb", "%db1, %tb1 = pulse.drive %da1, %w, %ta1"
    )
    with pytest.raises(pulse.CompilationError, match="overlaps or precedes"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


def test_qudits_must_be_distinct_and_of_one_system(system):
    with pytest.raises(pulse.InvalidValueError, match="twice"):
        pulse.load(MODULE, entry="round", qudits=[system.qudits[0], system.qudits[0]])
    other = pulse.quantum_system("transmon_surface17")
    if other is not system:
        with pytest.raises(pulse.InvalidValueError, match="more than one"):
            pulse.load(MODULE, entry="round", qudits=[system.qudits[0], other.qudits[1]])


def test_a_stated_clock_must_be_the_systems(system):
    period_fs = round(1e6 / system.clock_ghz)
    stated = (
        "module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = %d>} {"
    )
    same = MODULE.replace("module {", stated % period_fs, 1)
    assert len(events(pulse.load(same, entry="round", qudits=system.qudits[:2]))) == 3
    other = MODULE.replace("module {", stated % (2 * period_fs), 1)
    with pytest.raises(pulse.CompilationError, match="clock"):
        pulse.load(other, entry="round", qudits=system.qudits[:2])


T = "!pulse.drive_line, !pulse.tone"
TIMED = "attributes {pulse.schedule_state = #pulse.schedule_state<timed>}"


def _long(cycles):
    """The attributes of a function a call runs: its length and its schedule."""
    return f"attributes {{pulse.length_cycles = {cycles} : i64, " + TIMED[len("attributes {") :]


def _at(start, cycles, periods=None):
    """A call's timing."""
    period = "" if periods is None else f"{periods}, "
    return f"<{{duration_cycles = {cycles} : i64, {period}start_cycles = {start} : i64}}>"


CALLS = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func private @pair(%a: !pulse.drive_line, %at: !pulse.tone, %b: !pulse.drive_line, %bt:
        !pulse.tone) -> ({T}, {T}) {_long(16)} {{
    %a1, %at1 = pulse.call @x(%a, %at) {_at(0, 8)} : ({T}) -> ({T})
    %b1, %bt1 = pulse.call @x(%b, %bt) {_at(8, 8)} : ({T}) -> ({T})
    return %a1, %at1, %b1, %bt1 : {T}, {T}
  }}
  func.func @shot(%qa: !pulse.qudit, %qb: !pulse.qudit) {TIMED} {{
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %b, %bt = pulse.get_drive_line %qb : (!pulse.qudit) -> ({T})
    %a1, %at1, %b1, %bt1 = pulse.call @pair(%a, %at, %b, %bt) {_at(0, 16)}
        : ({T}, {T}) -> ({T}, {T})
    %b2, %bt2 = pulse.call @x(%b1, %bt1) {_at(40, 8)} : ({T}) -> ({T})
    %b3, %bt3, %a3, %at3 = pulse.call @pair(%b2, %bt2, %a1, %at1) {_at(100, 16)}
        : ({T}, {T}) -> ({T}, {T})
    return
  }}
}}
"""


def test_calls_with_a_stated_start_are_written_in_place(system):
    program = pulse.load(CALLS, entry="shot", qudits=system.qudits[:2])
    assert events(program) == [
        ("drive", 0, 0, 8),
        ("drive", 1, 8, 8),
        ("drive", 1, 40, 8),
        ("drive", 1, 100, 8),
        ("drive", 0, 108, 8),
    ]
    assert "call" not in program.mlir()


@pytest.mark.parametrize(
    "old, new, message",
    [
        (" " + _at(40, 8), "", "without stating when it starts"),
        (_at(0, 16), _at(0, 12), "is 16 cycles long"),
        (_at(40, 8), _at(12, 8), "overlaps or precedes"),
        (
            "pulse.schedule_state = #pulse.schedule_state<timed>} {\n    %n",
            "made.by = 1 : i64} {\n    %n",
            "which is not scheduled",
        ),
    ],
)
def test_calls_that_cannot_be_written_in_place(system, old, new, message):
    text = CALLS.replace(old, new)
    assert text != CALLS
    with pytest.raises(pulse.CompilationError, match=message):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_func_call_is_refused_with_the_call_to_write(system):
    text = CALLS.replace(f"pulse.call @x(%b1, %bt1) {_at(40, 8)}", "func.call @x(%b1, %bt1)")
    assert text != CALLS
    with pytest.raises(
        pulse.CompilationError, match="shot calls a function with func.call.*pulse.call"
    ):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_call_that_lasts_longer_than_its_function_holds_its_lines(system):
    # @x is 8 cycles; given 20 from 40, its line is free again at 60.
    longer = CALLS.replace(_at(40, 8), _at(40, 20))
    program = pulse.load(longer, entry="shot", qudits=system.qudits[:2])
    assert ("drive", 1, 40, 8) in events(program)
    late = longer.replace(_at(100, 16), _at(50, 16))
    with pytest.raises(pulse.CompilationError, match="overlaps or precedes"):
        pulse.load(late, entry="shot", qudits=system.qudits[:2])


@pytest.mark.parametrize(
    "timing",
    [
        "<{duration_cycles = 8 : i64, start_cycles = 9223372036854775807 : i64}>",
        "<{duration_cycles = 8 : i64, start_cycles = -5 : i64}>",
        "<{duration_cycles = 9223372036854775807 : i64, start_cycles = 40 : i64}>",
    ],
)
def test_a_calls_times_are_held_to_the_range_of_every_operation(system, timing):
    text = CALLS.replace(_at(40, 8), timing)
    assert text != CALLS
    with pytest.raises(pulse.CompilationError, match=r"shot calls x with .* outside 0 to 2\^50"):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_the_range_holds_for_nested_calls_periods_and_what_the_calls_write(system):
    far = 2**60
    nested = CALLS.replace(_at(8, 8), _at(far, 8))
    assert nested != CALLS
    with pytest.raises(pulse.CompilationError, match=r"pair calls x with .* outside 0 to 2\^50"):
        pulse.load(nested, entry="shot", qudits=system.qudits[:2])
    period = LOOP.replace("period_cycles = array<i64: 20>", f"period_cycles = array<i64: {far}>", 1)
    assert period != LOOP
    with pytest.raises(pulse.CompilationError, match=r"shot calls x with .* outside 0 to 2\^50"):
        pulse.load(period, entry="shot", qudits=system.qudits[:1])
    # Each start is in range; the drive the two calls write starts at their sum.
    half = 2**49
    summed = (
        CALLS.replace(_at(8, 8), _at(half, 8))
        .replace(_long(16), _long(half + 8))
        .replace(_at(0, 16), _at(0, half + 8))
        .replace(_at(100, 16), _at(2**50 - 8, half + 8))
        # Nothing else is wrong with it: the call between the two is after the first.
        .replace(_at(40, 8), _at(half + 16, 8))
    )
    with pytest.raises(
        pulse.CompilationError, match=r"outside 0 to 2\^50 cycles once its calls are written out"
    ):
        pulse.load(summed, entry="shot", qudits=system.qudits[:2])


def test_a_called_function_is_not_in_the_loaded_module(system):
    # Another compiler's attribute that names the function does not keep it.
    text = CALLS.replace(_long(8), _long(8)[:-1] + ", made.by = @x}")
    assert text != CALLS
    program = pulse.load(text, entry="shot", qudits=system.qudits[:2])
    assert program.mlir().count("func.func") == 1
    assert len(events(program)) == 5


def test_a_called_functions_recorded_length_is_where_its_operations_end(system):
    text = CALLS.replace(_long(8), _long(4))
    assert text != CALLS
    with pytest.raises(pulse.CompilationError, match="its operations end at 8"):
        pulse.load(text, entry="shot", qudits=system.qudits[:2])


def test_a_function_that_calls_itself_is_refused(system):
    text = f"""
module {{
  func.func private @again(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(0)} {{
    %d1, %t1 = pulse.call @again(%d, %t) {_at(0, 0)} : ({T}) -> ({T})
    return %d1, %t1 : {T}
  }}
  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %d1, %t1 = pulse.call @again(%d, %t) {_at(0, 0)} : ({T}) -> ({T})
    return
  }}
}}
"""
    with pytest.raises(pulse.CompilationError, match="calls itself"):
        pulse.load(text, entry="shot", qudits=system.qudits[:1])


# ── Classical functions ──────────────────────────────────────────────────────


def _links():
    from cudaq_pulse.classical import Link

    return pulse.ControlSystem(links={"host": Link(fixed_ns=50, feedback_ns=100)})


@pulse.classical(max_latency_ns=400, link="host")
def decide(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
    return pulse.Bits([bits[0]])


@pulse.kernel
def rounds(a, c):
    ro = get_readout_line(a)  # noqa: F821
    da = get_drive_line(a)  # noqa: F821
    dc = get_drive_line(c)  # noqa: F821
    for _ in range(2):
        drive(da, square(16, 0.5))  # noqa: F821
        sync(da, ro)  # noqa: F821
        m = readout(ro, square(100, 0.05))  # noqa: F821
        fix = decide.submit([m]).result()
        sync(dc, ro)  # noqa: F821
        if fix[0]:
            drive(dc, square(16, 0.5))  # noqa: F821


def _as_written(system, machine, kernel=None):
    """The compiled kernel and its module as another compiler would write it:
    a function of its qudits, with nothing of the target in it."""
    import re

    native = pulse.compile(kernel or rounds, quantum_system=system, control_system=machine).lower(
        system.qudits[0], system.qudits[2]
    )
    text = native.mlir()
    names = []
    for qudit in (0, 2):
        found = re.search(
            r"\n\s*(%\w+) = pulse\.qudit_alloc \{qudit = "
            + str(qudit)
            + r" : i64\} : !pulse\.qudit",
            text,
        )
        assert found is not None
        names.append(found.group(1))
        text = text.replace(found.group(0), "")
    text = text.replace(
        "func.func @main()",
        "func.func @main(" + ", ".join(f"{n}: !pulse.qudit" for n in names) + ")",
    )
    text = re.sub(r" \{frequency_hz = [^}]*\}", "", text)
    # Of its link, the module states the name: the link's cost is the target's.
    text = re.sub(
        r"classical\.links = \{.*?\}\}, ",
        'classical.links = {"0" = {name = "host"}}, ',
        text,
        count=1,
    )
    assert text.count("classical.links") == 1 and "classical.callable" in text
    return native, text


def _calls(program):
    exported = program.export_to_controller()
    return exported["calls"], exported["links"], exported["manifest"]


def test_classical_functions_are_read_from_the_module(system):
    machine = _links()
    native, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert events(loaded) == events(native)
    assert _calls(loaded) == _calls(native)
    assert len(_calls(loaded)[0]) == 2
    assert [fn.implementation for fn in loaded.classical] == [None]


@pulse.classical(max_latency_ns=600, latency_mode="timeout", timeout_ns=250, link="host")
def decide_timed(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
    return pulse.Bits([bits[0]])


@pulse.kernel
def timed_rounds(a, c):
    ro = get_readout_line(a)  # noqa: F821
    dc = get_drive_line(c)  # noqa: F821
    m = readout(ro, square(100, 0.05))  # noqa: F821
    fix = decide_timed.submit([m]).result()
    sync(dc, ro)  # noqa: F821
    if fix[0]:
        drive(dc, square(16, 0.5))  # noqa: F821


def test_a_function_with_a_timeout_is_read_with_it(system):
    machine = _links()
    native, text = _as_written(system, machine, timed_rounds)
    assert "timeout_ns" in text
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert loaded.classical[0].timeout_ns == 250.0
    assert loaded.classical[0].latency_mode == "timeout"
    assert _calls(loaded) == _calls(native)


def test_a_given_classical_function_is_checked_and_its_latency_used(system):
    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(
        text, entry="main", qudits=qudits, control_system=machine, classical=[decide]
    )
    assert loaded.classical[0].implementation is decide.implementation

    @pulse.classical(name="decide", max_latency_ns=900, link="host")
    def slow(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([bits[0]])

    with pytest.raises(pulse.CompilationError, match="before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[slow])

    @pulse.classical(name="decide", max_latency_ns=400, link="host")
    def wide(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 2]:
        return pulse.Bits([bits[0], bits[0]])

    with pytest.raises(pulse.CompilationError, match=r"gives \(bits:1\).*gives \(bits:2\)"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[wide])

    @pulse.classical(max_latency_ns=400, link="host")
    def other(bits: pulse.Bits) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([bits[0]])

    with pytest.raises(pulse.InvalidValueError, match="calls no classical function named other"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[other])
    with pytest.raises(pulse.InvalidTypeError, match="takes classical functions"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=["decide"])


def test_the_links_are_the_control_systems(system):
    from cudaq_pulse.classical import Link

    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    slower = pulse.ControlSystem(links={"host": Link(fixed_ns=50, feedback_ns=900)})
    with pytest.raises(pulse.CompilationError, match="before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=slower)


LOOP = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}> :
          !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func @shot(%qa: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %out:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%l = %a, %lt = %at) -> ({T}) {{
      %l1, %lt1 = pulse.call @x(%l, %lt) {_at(4, 8, "period_cycles = array<i64: 20>")}
          : ({T}) -> ({T})
      %l2, %lt2 = pulse.call @x(%l1, %lt1) {_at(12, 8, "period_cycles = array<i64: 20>")}
          : ({T}) -> ({T})
      scf.yield %l2, %lt2 : {T}
    }}
    return
  }}
}}
"""


def test_calls_in_a_loop_keep_the_loop_and_state_its_period(system):
    program = pulse.load(LOOP, entry="shot", qudits=system.qudits[:1])
    assert [(start, duration) for _, _, start, duration in events(program)] == [
        (4, 8),
        (12, 8),
        (24, 8),
        (32, 8),
        (44, 8),
        (52, 8),
    ]
    text = program.mlir()
    assert text.count("scf.for") == 1 and text.count("pulse.drive ") == 2
    assert "call" not in text


# ── What a review found ──────────────────────────────────────────────────────


def test_a_failed_load_carries_a_documented_code(system):
    with pytest.raises(pulse.CompilationError) as wrong:
        pulse.load(MODULE, entry="missing", qudits=system.qudits[:2])
    assert wrong.value.code == "wrong_argument"
    text = MODULE.replace("start_cycles = 16 : i64", "start_cycles = 0 : i64").replace(
        "%db1, %tb1 = pulse.drive %db, %w, %tb", "%db1, %tb1 = pulse.drive %da1, %w, %ta1"
    )
    with pytest.raises(pulse.CompilationError) as overlap:
        pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert overlap.value.code == "other"


def test_a_function_with_a_qudit_of_its_own_is_refused(system):
    text = MODULE.replace(
        "    %n = arith.constant 16 : i64",
        "    %own = pulse.qudit_alloc {qudit = 5 : i64} : !pulse.qudit\n"
        "    %n = arith.constant 16 : i64",
    )
    assert text != MODULE
    with pytest.raises(pulse.CompilationError, match="allocates a qudit"):
        pulse.load(text, entry="round", qudits=system.qudits[:2])


@pytest.mark.parametrize("limit", [{"max_instructions": 1000}, {"max_program_duration": 1e6}])
def test_program_limits_are_not_checked_yet_and_say_so(system, limit):
    machine = pulse.Controller(**limit)
    with pytest.raises(pulse.UnsupportedError, match="max_instructions or max_program_duration"):
        pulse.load(MODULE, entry="round", qudits=system.qudits[:2], control_system=machine)


def test_a_slower_link_is_refused(system):
    from cudaq_pulse.classical import Link

    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]
    # A link that takes longer to send on, with the same feedback: the answer
    # is still due in time as scheduled, and only the submission is too short.
    slower = pulse.ControlSystem(links={"host": Link(fixed_ns=5000, feedback_ns=100)})
    with pytest.raises(pulse.CompilationError, match=r"is given \d+ cycle\(s\).*it takes \d+"):
        pulse.load(text, entry="main", qudits=qudits, control_system=slower)


def test_every_submission_is_seen_by_the_check(system):
    from cudaq_pulse._load import _check_submissions
    from cudaq_pulse._native import _cudaq_pulse_native as native

    machine = _links()
    _, text = _as_written(system, machine)
    # The loader sees each submission, with the cycles its schedule gives it
    # and the cycles the classical check wrote that its link takes.
    rows = native.PulseModuleBuilder().load_submissions(text.encode())
    assert len(rows) == text.count("classical.submit ") == 1
    assert rows[0][0].startswith("decide") and rows[0][1] == rows[0][2] > 0
    _check_submissions(rows)
    # One the check wrote nothing on was not checked: it is refused.
    bare = re.sub(r"(, )?classical\.submit_cycles = \d+ : i64", "", text)
    assert bare != text
    rows = native.PulseModuleBuilder().load_submissions(bare.encode())
    assert [row[2] for row in rows] == [-1]
    with pytest.raises(pulse.CompilationError, match="no link time to check it against"):
        _check_submissions(rows)
    _check_submissions([("decide", 100, 100, [], False)])
    with pytest.raises(pulse.CompilationError, match="is given 99 cycle"):
        _check_submissions([("decide", 100, 100, [], False), ("decide", 99, 100, [], False)])


def test_a_given_bool_function_keeps_its_own_types(system):
    machine = _links()
    _, text = _as_written(system, machine)
    qudits = [system.qudits[0], system.qudits[2]]

    @pulse.classical(name="decide", max_latency_ns=400, link="host")
    def yes(bits: pulse.Bits) -> bool:
        return bool(bits[0])

    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[yes])
    assert [type(r).__name__ for r in loaded.classical[0].results] == [
        type(r).__name__ for r in yes.specialize(bits=1).results
    ]
    assert loaded.classical[0].base_name == "decide"


def test_a_function_is_given_for_its_versions_only():
    from cudaq_pulse._load import _given_functions

    row = ("decide__fast", 1, 0, ["bits:1"], ["bits:1"], (0.0, 400.0, 400.0), "bounded", "eager")
    with pytest.raises(pulse.InvalidValueError, match="calls no classical function named decide"):
        _given_functions([decide], [row + (None, None, None)])
    version = ("decide__1",) + row[1:] + (None, None, None)
    assert set(_given_functions([decide], [version])) == {"decide__1"}
    lazy = version[:7] + ("lazy",) + version[8:]
    with pytest.raises(pulse.CompilationError, match="submit policy"):
        _given_functions([decide], [lazy])


# ── Links and dispatch keys are the control system's ─────────────────────────


def _host(**link):
    from cudaq_pulse.classical import Link

    return Link(**(link or {"fixed_ns": 50, "feedback_ns": 100}))


def _written(system):
    machine = _links()
    native, text = _as_written(system, machine)
    return native, text, [system.qudits[0], system.qudits[2]]


def test_a_link_is_found_by_its_name_not_its_number(system):
    native, text, qudits = _written(system)
    from cudaq_pulse.classical import Link

    moved = pulse.ControlSystem(links={"fast": Link(), "host": _host()})
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=moved)
    exported = loaded.export_to_controller()
    assert [link["name"] for link in exported["links"].values()] == ["host"]
    assert list(exported["links"]) == ["1"]
    assert [c["due_cycles"] for c in exported["calls"]] == [
        c["due_cycles"] for c in native.export_to_controller()["calls"]
    ]


def test_a_link_the_control_system_does_not_have_is_refused(system):
    _, text, qudits = _written(system)
    with pytest.raises(pulse.CompilationError, match="no link of that name") as none:
        pulse.load(text, entry="main", qudits=qudits)
    assert none.value.code == "control_system"
    other = pulse.ControlSystem(links={"nvqlink": _host()})
    with pytest.raises(pulse.CompilationError, match=r"link 'host'.*its links: 'nvqlink'"):
        pulse.load(text, entry="main", qudits=qudits, control_system=other)


@pytest.mark.parametrize("key", ["host", 0])
def test_links_gives_the_control_systems_name(system, key):
    native, text, qudits = _written(system)
    other = pulse.ControlSystem(links={"nvqlink": _host()})
    loaded = pulse.load(
        text, entry="main", qudits=qudits, control_system=other, links={key: "nvqlink"}
    )
    exported = loaded.export_to_controller()
    assert [link["name"] for link in exported["links"].values()] == ["nvqlink"]
    assert [
        (c["submit_cycles"], c["due_cycles"], c["taken_cycles"]) for c in exported["calls"]
    ] == [
        (c["submit_cycles"], c["due_cycles"], c["taken_cycles"])
        for c in native.export_to_controller()["calls"]
    ]


def test_links_is_checked(system):
    _, text, qudits = _written(system)
    machine = _links()
    with pytest.raises(pulse.InvalidValueError, match="'other', which is not a link"):
        pulse.load(
            text, entry="main", qudits=qudits, control_system=machine, links={"other": "host"}
        )
    with pytest.raises(pulse.InvalidTypeError, match="maps a link's name or id"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, links={"host": 0})
    unnamed = text.replace('classical.links = {"0" = {name = "host"}}, ', "")
    assert unnamed != text
    with pytest.raises(pulse.CompilationError, match="does not name its link 0"):
        pulse.load(unnamed, entry="main", qudits=qudits, control_system=machine)
    assert pulse.load(
        unnamed, entry="main", qudits=qudits, control_system=machine, links={0: "host"}
    ).classical


def test_a_controller_that_fixes_a_key_gives_it(system):
    _, text, qudits = _written(system)
    keys = pulse.Processor(links=["host"], dispatch_keys={"decide": 0x1234})
    fixed = pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[keys])], links={"host": _host()}
    )
    function = pulse.load(
        text, entry="main", qudits=qudits, control_system=fixed
    ).export_to_controller()["manifest"]["functions"][0]
    assert function["dispatch_key"] == 0x1234
    assert "processors[0]" in function["dispatch_key_fixed_by"]
    unlisted = pulse.Processor(links=["host"], dispatch_keys={"other": 5})
    system_of = pulse.ControlSystem(
        controllers=[pulse.Controller(processors=[unlisted])], links={"host": _host()}
    )
    with pytest.raises(pulse.CompilationError, match="dispatch_keys have no 'decide'"):
        pulse.load(text, entry="main", qudits=qudits, control_system=system_of)


def test_a_reserved_dispatch_key_is_refused(system):
    _, text, qudits = _written(system)
    found = re.search(r"dispatch_key = (\d+) : i64", text)
    reserved = text.replace(
        found.group(0), f"dispatch_key = {0x7ED8BE82} : i64, dispatch_key_override"
    )
    with pytest.raises(pulse.CompilationError, match="0x7ed8be82.*rename it"):
        pulse.load(reserved, entry="main", qudits=qudits, control_system=_links())


# ── CUDA-Q QEC's decoder messages ────────────────────────────────────────────


def _decoder_rounds():
    from tests.classical.fake_cudaq_qec import qec

    enqueue = pulse.classical(qec.enqueue_syndromes, link="host")
    corrections = pulse.classical(qec.get_corrections, link="host", max_latency_ns=1000)

    @pulse.kernel
    def decoded(a, c):
        ra = get_readout_line(a)  # noqa: F821
        rc = get_readout_line(c)  # noqa: F821
        dc = get_drive_line(c)  # noqa: F821
        for _ in range(4):
            s0 = readout(ra, square(100, 0.05))  # noqa: F821
            s1 = readout(rc, square(100, 0.05))  # noqa: F821
            enqueue.submit(3, [s0, s1], 7)
            sync(dc, ra, rc)  # noqa: F821
        fix = corrections.submit(3, 1, True).result()
        sync(dc, ra, rc)  # noqa: F821
        if fix[0]:
            drive(dc, square(16, 0.5))  # noqa: F821

    return decoded, enqueue, corrections


def test_a_modules_decoder_messages_are_read_as_cudaq_qecs(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    native, text = _as_written(system, machine, kernel)
    assert "@enqueue_syndromes__3_2" in text and "@get_corrections__3_1" in text
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert events(loaded) == events(native)
    assert _calls(loaded) == _calls(native)
    # Rounds that only submit: four submissions with no answer, then one
    # answer, and the loop stays rolled.
    calls = _calls(loaded)[0]
    assert [c["function"] for c in calls] == ["enqueue_syndromes__3_2"] * 4 + [
        "get_corrections__3_1"
    ]
    assert [c["taken_cycles"] for c in calls[:4]] == [None] * 4
    assert loaded.mlir().count("scf.for") == 1
    # They are the decoding server's messages: its keys, routed by the
    # decoder id, and answered by the configured decoder.
    assert [(fn.name, fn._dispatch_key, fn._key_route) for fn in loaded.classical] == [
        (fn.name, fn._dispatch_key, fn._key_route) for fn in native.classical
    ]
    assert all(fn._key_route == 3 and fn.implementation is not None for fn in loaded.classical)


def test_cudaq_qec_functions_given_for_a_modules_messages_give_their_latency(system):
    from tests.classical.fake_cudaq_qec import qec

    machine = _links()
    kernel, enqueue, _ = _decoder_rounds()
    native, text = _as_written(system, machine, kernel)
    qudits = [system.qudits[0], system.qudits[2]]
    same = pulse.classical(qec.get_corrections, link="host", max_latency_ns=1000)
    loaded = pulse.load(
        text, entry="main", qudits=qudits, control_system=machine, classical=[enqueue, same]
    )
    assert _calls(loaded) == _calls(native)
    slow = pulse.classical(qec.get_corrections, link="host", max_latency_ns=5000)
    with pytest.raises(pulse.CompilationError, match="get_corrections__3_1.*before it is due"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[slow])
    other = pulse.classical(qec.reset_decoder, link="host")
    with pytest.raises(pulse.InvalidValueError, match="no classical function named reset_decoder"):
        pulse.load(text, entry="main", qudits=qudits, control_system=machine, classical=[other])


def test_two_decoders_messages_share_the_servers_key(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    # A second decoder's version of the same message, declared and unused.
    found = re.search(r"\n\s*classical\.callable @get_corrections__3_1 [^\n]*", text)
    text = text.replace(found.group(0), found.group(0) + found.group(0).replace("__3_1", "__4_1"))
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert {fn.name: fn._key_route for fn in loaded.classical} == {
        "enqueue_syndromes__3_2": 3,
        "get_corrections__3_1": 3,
        "get_corrections__4_1": 4,
    }


def _decoder_text(system):
    machine = _links()
    kernel, enqueue, corrections = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    return machine, text, [system.qudits[0], system.qudits[2]], enqueue, corrections


def _one_message(system, name):
    """A module whose entry submits one decoder message, with every
    argument a constant of its own, and waits for it where it answers."""
    from cudaq_pulse.classical import _cudaqx

    from tests.classical.fake_cudaq_qec import qec

    machine = _links()
    declared = pulse.classical(
        getattr(qec, name), link="host", **({"max_latency_ns": 1000} if name[0] == "g" else {})
    )
    arguments = {
        "enqueue_syndromes": "3, [s0, s1], 7",
        "get_corrections": "3, 2, True",
        "reset_decoder": "3",
    }[name]
    source = f"""
@pulse.kernel
def one(a, c):
    ra = get_readout_line(a)
    rc = get_readout_line(c)
    s0 = readout(ra, square(100, 0.05))
    s1 = readout(rc, square(100, 0.05))
    sent = declared.submit({arguments})
    sync(ra, rc)
"""
    scope = {"pulse": pulse, "declared": declared}
    exec(source, scope)  # noqa: S102
    _, text = _as_written(system, machine, scope["one"])
    words = _cudaqx._WORDS[name]
    version = re.search(rf"classical\.submit %\w+, @({name}__\w+)\(([^)]*)\)", text)
    return machine, text, version.group(1), version.group(2).split(", ")[: len(words)], words


def _with_constant(text, old, value):
    """*text* with a new constant where one submission's operand *old* was."""
    other = "%another"
    line = re.search(r"(\n(\s*))(%\w+ = classical\.get_link)", text)
    changed = text.replace(
        line.group(0), f"{line.group(1)}{other} = arith.constant {value} : i64{line.group(0)}", 1
    )
    submit = re.search(r"classical\.submit %\w+, @\w+\(([^)]*)\)", changed)
    operands = submit.group(1).split(", ")
    assert operands.count(old) == 1
    return changed.replace(
        submit.group(0),
        submit.group(0).replace(
            submit.group(1), ", ".join(other if o == old else o for o in operands)
        ),
        1,
    )


RELEASED = ("enqueue_syndromes", "get_corrections", "reset_decoder")


def test_every_decoder_message_states_what_its_submissions_send():
    from cudaq_pulse.classical import _cudaqx

    assert set(_cudaqx._WORDS) == set(_cudaqx._MESSAGES)
    assert {m.name for m in _cudaqx._MESSAGES.values() if m.released} == set(RELEASED)
    for words in _cudaqx._WORDS.values():
        assert words[0] == ("decoder_id", "id")
        assert all(rule in ("id", "length", None, 0) for _, rule in words)


@pytest.mark.parametrize("name", RELEASED)
def test_a_decoder_message_loads_as_compile_sends_it(system, name):
    machine, text, version, operands, words = _one_message(system, name)
    qudits = [system.qudits[0], system.qudits[2]]
    loaded = pulse.load(text, entry="main", qudits=qudits, control_system=machine)
    assert [fn.name for fn in loaded.classical] == [version]
    # compile sends a constant wherever the message's description states one.
    constants = dict(re.findall(r"(%\w+) = arith\.constant (-?\d+) : i64", text))
    numbers = dict(zip(("id", "length"), map(int, version.split("__")[1].split("_")), strict=False))
    for operand, (_, rule) in zip(operands, words, strict=True):
        if rule is not None:
            assert int(constants[operand]) == (numbers[rule] if isinstance(rule, str) else rule)


def _every_word():
    from cudaq_pulse.classical import _cudaqx

    return [(name, at) for name in RELEASED for at in range(len(_cudaqx._WORDS[name]))]


@pytest.mark.parametrize("name, at", _every_word())
def test_every_word_of_a_decoder_message_is_held_to_its_description(system, name, at):

    machine, text, version, operands, words = _one_message(system, name)
    qudits = [system.qudits[0], system.qudits[2]]
    what, rule = words[at]
    changed = _with_constant(text, operands[at], 5)
    if rule is None:
        # The caller's value: any constant loads.
        pulse.load(changed, entry="main", qudits=qudits, control_system=machine)
        return
    said = "its name states" if isinstance(rule, str) else f"{name} sends 0"
    with pytest.raises(
        pulse.CompilationError,
        match=f"a submission to {version} sends the {what} 5, and {said}",
    ):
        pulse.load(changed, entry="main", qudits=qudits, control_system=machine)


def test_a_word_of_a_decoder_message_is_a_constant_64_bit_integer(system):
    machine, text, qudits, _, _ = _decoder_text(system)
    sent = re.search(r"@get_corrections__3_1\((%\w+), [^)]*\) [^\n]*", text)
    line = re.search(r"(\n(\s*))(%\w+ = classical\.get_link)", text)
    # The decoder id as a sum the module computes: not a constant.
    computed = text.replace(
        line.group(0),
        f"{line.group(1)}%three = arith.constant 3 : i64"
        f"{line.group(1)}%sum = arith.addi %three, %three : i64{line.group(0)}",
        1,
    ).replace(sent.group(0), sent.group(0).replace(sent.group(1), "%sum", 1), 1)
    with pytest.raises(
        pulse.CompilationError,
        match="get_corrections__3_1 sends the decoder_id as a value that is not a constant",
    ):
        pulse.load(computed, entry="main", qudits=qudits, control_system=machine)
    # The decoder id as a 32-bit constant: the verifier takes any integer
    # width for an i64 argument, and compile writes 64 bits.
    assert ": (i64, i64, i64) ->" in sent.group(0)
    narrow = text.replace(
        line.group(0), f"{line.group(1)}%three = arith.constant 3 : i32{line.group(0)}", 1
    ).replace(
        sent.group(0),
        sent.group(0)
        .replace(sent.group(1), "%three", 1)
        .replace(": (i64, i64, i64) ->", ": (i32, i64, i64) ->"),
        1,
    )
    with pytest.raises(
        pulse.CompilationError,
        match="get_corrections__3_1 does not send the decoder_id as a 64-bit integer",
    ):
        pulse.load(narrow, entry="main", qudits=qudits, control_system=machine)


def test_an_ordinary_function_is_not_given_for_a_decoder_message(system):
    machine, text, qudits, _, _ = _decoder_text(system)

    def corrections(a: int, b: int, c: int) -> pulse.Length[pulse.Bits, 1]:
        return pulse.Bits([False])

    # A function cannot be declared under a message version's name.
    with pytest.raises(pulse.InvalidValueError, match="can't be named 'get_corrections__3_1'"):
        pulse.classical(corrections, name="get_corrections__3_1", max_latency_ns=10, link="host")
    # One that has the name all the same is not taken for the message.
    ordinary = pulse.classical(corrections, max_latency_ns=10, link="host")
    ordinary.name = "get_corrections__3_1"
    with pytest.raises(
        pulse.InvalidValueError,
        match="get_corrections__3_1 is CUDA-Q QEC's get_corrections; give pulse.classical",
    ):
        pulse.load(
            text,
            entry="main",
            qudits=qudits,
            control_system=machine,
            classical=[ordinary],
        )


def test_a_decoder_messages_name_is_never_an_ordinary_function(system):
    machine, text, qudits, _, corrections = _decoder_text(system)
    # With its own key: neither an ordinary function nor the message.
    own = text.replace(f"dispatch_key = {0x882D5BA1} : i64", "dispatch_key = 77 : i64")
    assert own != text
    for given in ((), (corrections,)):
        with pytest.raises(
            pulse.CompilationError,
            match="get_corrections__3_1 is named as CUDA-Q QEC's get_corrections and has the "
            "dispatch key 0x4d; the message's is 0x882d5ba1",
        ):
            pulse.load(own, entry="main", qudits=qudits, control_system=machine, classical=given)
    # A message that is in no release, whatever its key.
    for key in (0x7ED8BE82, 77):
        mapped = text.replace("enqueue_syndromes__3_2", "enqueue_syndromes_mapped__3_2").replace(
            f"dispatch_key = {0x7ED8BE82} : i64", f"dispatch_key = {key} : i64"
        )
        with pytest.raises(pulse.CompilationError, match="is in no CUDA-Q QEC release"):
            pulse.load(mapped, entry="main", qudits=qudits, control_system=machine)


def test_names_that_are_not_a_versions_name_are_ordinary_functions(system):
    machine, text, qudits, _, _ = _decoder_text(system)
    # None of these is a name compile writes for a message: each is an
    # ordinary function, refused here for the decoding server's key.
    for name in (
        "enqueue_syndromes__3_02",
        "enqueue_syndromes__3",
        "enqueue_syndromes__3_2_1",
        "Enqueue_syndromes__3_2",
        f"enqueue_syndromes__{2**63}_2",
        "enqueue_syndromes__" + "9" * 5000 + "_2",
    ):
        renamed = text.replace("enqueue_syndromes__3_2", name)
        with pytest.raises(pulse.CompilationError, match="rename it"):
            pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)
    # A digit that is not ASCII (here ARABIC-INDIC DIGIT TWO, as a quoted
    # symbol) is not the name compile writes.
    renamed = text.replace("@enqueue_syndromes__3_2", '@"enqueue_syndromes__3_\\D9\\A2"')
    assert renamed != text
    with pytest.raises(pulse.CompilationError, match="rename it"):
        pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)


def test_a_decoder_messages_answer_is_taken_from_its_submissions_own_future(system):
    machine, text, qudits, _, _ = _decoder_text(system)
    # The future goes through a branch before it is waited on.
    found = re.search(
        r"\n(\s*)(%\w+), (%\w+) = classical\.submit (%[\w#]+), @get_corrections__3_1\(([^)]*)\) "
        r"(<\{[^\n]*)\n",
        text,
    )
    pad, link, future, old_link, operands, rest = found.groups()
    arm = f"classical.submit {old_link}, @get_corrections__3_1({operands}) {rest}"
    kinds = "!classical.link, !classical.future"
    through = (
        f"\n{pad}%flag = arith.constant true"
        f"\n{pad}{link}, {future} = scf.if %flag -> ({kinds}) {{"
        f"\n{pad}  %l_a, %f_a = {arm}"
        f"\n{pad}  scf.yield %l_a, %f_a : {kinds}"
        f"\n{pad}}} else {{"
        f"\n{pad}  %l_b, %f_b = {arm}"
        f"\n{pad}  scf.yield %l_b, %f_b : {kinds}"
        f"\n{pad}}}\n"
    )
    changed = text.replace(found.group(0), through, 1)
    refusal = "the future of a submission to get_corrections__3_1 is yielded or passed on"
    with pytest.raises(pulse.CompilationError, match=refusal):
        pulse.load(changed, entry="main", qudits=qudits, control_system=machine)
    # The same when the wait names an ordinary function of the module's.
    callable_ = re.search(r"\n\s*classical\.callable @get_corrections__3_1 [^\n]*", changed)
    mine = (
        callable_.group(0)
        .replace("@get_corrections__3_1", "@mine")
        .replace(
            f"dispatch_key = {0x882D5BA1} : i64, dispatch_key_override",
            "dispatch_key = 2122968502 : i64",  # the hash of its name
        )
    )
    assert "@mine" in mine and "dispatch_key = 2122968502" in mine
    renamed = changed.replace(callable_.group(0), callable_.group(0) + mine, 1)
    renamed = re.sub(
        r"classical\.wait (%\w+), @get_corrections__3_1", r"classical.wait \1, @mine", renamed
    )
    assert "classical.wait" in renamed and renamed != changed
    with pytest.raises(pulse.CompilationError, match=refusal):
        pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)
    # The other direction: an ordinary function's submissions in the arms,
    # and their answer taken under the message's name, or under its own.
    # In a module that calls a decoder message no future is passed on.
    ordinary = "the future of a submission to mine is yielded or passed on; in a module that calls"
    for waited in (changed, renamed):
        reverse = waited.replace(callable_.group(0), callable_.group(0) + mine, 1)
        if "@mine" not in waited:
            waited = reverse
        arms = re.sub(
            r"(= classical\.submit %[\w#]+, )@get_corrections__3_1\(", r"\1@mine(", waited
        )
        assert arms.count("classical.submit") == waited.count("classical.submit")
        assert arms.count("@mine(") == 2 and "@enqueue_syndromes__3_2(" in arms
        with pytest.raises(pulse.CompilationError, match=ordinary):
            pulse.load(arms, entry="main", qudits=qudits, control_system=machine)


def test_a_loaded_programs_decoder_messages_pickle(system):
    import pickle

    machine, text, qudits, enqueue, corrections = _decoder_text(system)
    for given in ((), (enqueue, corrections)):
        loaded = pulse.load(
            text, entry="main", qudits=qudits, control_system=machine, classical=given
        )
        again = pickle.loads(pickle.dumps(loaded))
        assert _calls(again) == _calls(loaded)
        assert [(fn.name, fn.link, fn._dispatch_key, fn._key_route) for fn in again.classical] == [
            (fn.name, fn.link, fn._dispatch_key, fn._key_route) for fn in loaded.classical
        ]


def test_a_decoder_message_with_other_types_is_refused(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    assert 'args = ["i64", "i64", "u8"]' in text
    other = text.replace('args = ["i64", "i64", "u8"]', 'args = ["i64", "i64", "i64"]')
    qudits = [system.qudits[0], system.qudits[2]]
    with pytest.raises(
        pulse.CompilationError,
        match=r"^load: the module's get_corrections__3_1 is named as CUDA-Q QEC's "
        r"get_corrections, which takes \(i64, i64, u8\)",
    ):
        pulse.load(other, entry="main", qudits=qudits, control_system=machine)


def test_a_decoder_messages_key_under_another_name_is_refused(system):
    machine = _links()
    kernel, _, _ = _decoder_rounds()
    _, text = _as_written(system, machine, kernel)
    renamed = text.replace("get_corrections__3_1", "mine__3_1")
    qudits = [system.qudits[0], system.qudits[2]]
    with pytest.raises(pulse.CompilationError, match="mine__3_1 is dispatch key.*rename it"):
        pulse.load(renamed, entry="main", qudits=qudits, control_system=machine)


# ── A loaded module is untrusted input ───────────────────────────────────────

LEAF = f"""
  func.func private @leaf(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}>
        : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
"""


def _shot(cycles):
    return f"""
  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %d1, %t1 = pulse.call @f0(%d, %t) {_at(0, cycles)} : ({T}) -> ({T})
    return
  }}
"""


def _chain(levels, calls):
    """@shot calls @f0; each @f{k} calls @f{k+1} *calls* times, one after
    the other; the last calls @leaf."""
    parts = ["module {", LEAF]
    for level in range(levels):
        callee = f"f{level + 1}" if level + 1 < levels else "leaf"
        length = 8 * calls ** (levels - level - 1)
        body, line, tone = [], "%d", "%t"
        for i in range(calls):
            body.append(
                f"    %d{i}, %t{i} = pulse.call @{callee}({line}, {tone}) "
                f"{_at(i * length, length)} : ({T}) -> ({T})"
            )
            line, tone = f"%d{i}", f"%t{i}"
        parts.append(
            f"  func.func private @f{level}(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) "
            f"{_long(calls * length)} {{\n"
            + "\n".join(body)
            + f"\n    return {line}, {tone} : {T}\n  }}"
        )
    parts += [_shot(8 * calls**levels), "}"]
    return "\n".join(parts)


def test_a_chain_of_calls_loads(system):
    program = pulse.load(_chain(3, 2), entry="shot", qudits=system.qudits[:1])
    assert [(start, duration) for _, _, start, duration in events(program)] == [
        (8 * i, 8) for i in range(8)
    ]


def test_calls_that_would_write_too_much_are_refused_before_any_is_written(system):
    # Each function calls the next twice: 2**21 leaves, written by none.
    with pytest.raises(pulse.CompilationError, match="write more than 1048576 operations"):
        pulse.load(_chain(21, 2), entry="shot", qudits=system.qudits[:1])


def test_calls_nested_too_deep_are_refused(system):
    with pytest.raises(pulse.CompilationError, match="nest deeper than 256"):
        pulse.load(_chain(300, 1), entry="shot", qudits=system.qudits[:1])
    assert pulse.load(_chain(200, 1), entry="shot", qudits=system.qudits[:1]).schedule


def _passes_on(name, callee):
    return (
        f"  func.func private @{name}(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) "
        f"{_long(8)} {{\n"
        f"    %d1, %t1 = pulse.call @{callee}(%d, %t) {_at(0, 8)} : ({T}) -> ({T})\n"
        f"    return %d1, %t1 : {T}\n  }}"
    )


def _two_chains(levels):
    """@shot calls @g0, the first of *levels* functions that each call the
    next, then @h0, the first of *levels* more, the last of which calls
    @g0."""
    parts = ["module {", LEAF]
    for k in range(levels):
        parts.append(_passes_on(f"g{k}", f"g{k + 1}" if k + 1 < levels else "leaf"))
    for k in range(levels):
        parts.append(_passes_on(f"h{k}", f"h{k + 1}" if k + 1 < levels else "g0"))
    parts.append(f"""  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %d1, %t1 = pulse.call @g0(%d, %t) {_at(0, 8)} : ({T}) -> ({T})
    %d2, %t2 = pulse.call @h0(%d1, %t1) {_at(8, 8)} : ({T}) -> ({T})
    return
  }}
}}""")
    return "\n".join(parts)


def test_calls_nested_too_deep_through_a_function_counted_before_are_refused(system):
    # @g0's 150 levels are counted for the first call; the second reaches
    # them below 150 more.
    with pytest.raises(pulse.CompilationError, match="the calls of shot nest deeper than 256"):
        pulse.load(_two_chains(150), entry="shot", qudits=system.qudits[:1])
    program = pulse.load(_two_chains(100), entry="shot", qudits=system.qudits[:1])
    assert [(start, duration) for _, _, start, duration in events(program)] == [(0, 8), (8, 8)]


def _nested_calls(functions, ifs):
    """@f0 .. each calls the next inside *ifs* nested branches on a
    measurement; the last calls @leaf."""
    parts = ["module {", LEAF]
    for k in range(functions):
        last = k + 1 == functions
        callee, args = ("leaf", "%d, %t") if last else (f"f{k + 1}", "%d, %t, %m")
        types = T if last else f"{T}, !pulse.measurement"
        body = [
            f"  func.func private @f{k}(%d: !pulse.drive_line, %t: !pulse.tone, "
            f"%m: !pulse.measurement) -> ({T}) {_long(8)} {{",
            "    %bit = pulse.is_set %m : !pulse.measurement -> i1",
        ]
        body += [f"    %r{i}:2 = scf.if %bit -> ({T}) {{" for i in range(ifs)]
        body.append(f"    %c0, %c1 = pulse.call @{callee}({args}) {_at(0, 8)} : ({types}) -> ({T})")
        body.append(f"    scf.yield %c0, %c1 : {T}")
        for i in reversed(range(ifs)):
            body.append(f"    }} else {{\n      scf.yield %d, %t : {T}\n    }}")
            if i:
                body.append(f"    scf.yield %r{i}#0, %r{i}#1 : {T}")
        body.append(f"    return %r0#0, %r0#1 : {T}\n  }}")
        parts.append("\n".join(body))
    parts.append(f"""  func.func @shot(%q: !pulse.qudit, %a: !pulse.qudit) {TIMED} {{
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %r, %rt = pulse.get_readout_line %a : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %r1, %rt1, %m = pulse.readout %r, %w, %rt, iq
        <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}>
        : !pulse.readout_line, !pulse.waveform, !pulse.tone
          -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %d1, %t1 = pulse.call @f0(%d, %t, %m) {_at(400, 8)} : ({T}, !pulse.measurement) -> ({T})
    return
  }}
}}""")
    return "\n".join(parts)


def test_calls_that_would_nest_regions_too_deep_are_refused_before_any_is_written(system):
    # Each function nests 100 branches, under the 254 a function may; written
    # in place of one another, three of them nest 300.
    with pytest.raises(
        pulse.CompilationError, match="the calls of shot nest regions deeper than 256"
    ):
        pulse.load(_nested_calls(3, 100), entry="shot", qudits=system.qudits[:2])
    program = pulse.load(_nested_calls(2, 100), entry="shot", qudits=system.qudits[:2])
    assert ("drive", 0, 400, 8) in events(program)


def _looped(period):
    """An entry with one drive that starts at 4 in a loop of three trips,
    *period* cycles apart. No call."""
    return f"""module {{
  func.func @shot(%q: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %a, %at = pulse.get_drive_line %q : (!pulse.qudit) -> ({T})
    %out:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%l = %a, %lt = %at) -> ({T}) {{
      %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
      %l1, %lt1 = pulse.drive %l, %w, %lt
          {_at(4, 8, f"period_cycles = array<i64: {period}>")}
          : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
      scf.yield %l1, %lt1 : {T}
    }}
    return
  }}
}}"""


def test_an_operations_own_period_is_held_to_the_range(system):
    program = pulse.load(_looped(20), entry="shot", qudits=system.qudits[:1])
    assert max(e.start_cycles for e in program.schedule) == 44
    for period in (2**60, -20):
        with pytest.raises(
            pulse.CompilationError, match=r"start, duration or period is outside 0 to 2\^50"
        ):
            pulse.load(_looped(period), entry="shot", qudits=system.qudits[:1])


def test_the_last_trip_of_a_loop_starts_within_the_range(system):
    # Three trips of a drive that starts at 4: a period of 2**49 is in the
    # range, and the last trip starts at 2**50 + 4.
    with pytest.raises(
        pulse.CompilationError, match=r"shot has an operation in a loop whose last trip starts"
    ):
        pulse.load(_looped(2**49), entry="shot", qudits=system.qudits[:1])
    program = pulse.load(_looped(2**48), entry="shot", qudits=system.qudits[:1])
    assert max(e.start_cycles for e in program.schedule) == 2**49 + 4
    # The same for the drives calls write, which take the calls' periods.
    calls = LOOP.replace("period_cycles = array<i64: 20>", f"period_cycles = array<i64: {2**49}>")
    assert calls != LOOP
    with pytest.raises(
        pulse.CompilationError,
        match=r"last trip starts past 2\^50 cycles once its calls are written out",
    ):
        pulse.load(calls, entry="shot", qudits=system.qudits[:1])
    # The trips are read from constant bounds: a loop without them is
    # refused, wherever in it the timed operation is.
    summed = _looped(20).replace(
        "    %out:2 = scf.for %i = %c0 to %c3 ",
        "    %c2 = arith.constant 2 : index\n    %sum = arith.addi %c1, %c2 : index\n"
        "    %out:2 = scf.for %i = %c0 to %sum ",
    )
    assert summed != _looped(20)
    with pytest.raises(
        pulse.CompilationError, match="timed operation in a loop whose bounds are not constants"
    ):
        pulse.load(summed, entry="shot", qudits=system.qudits[:1])
    # Bounds whose difference overflows a trip count, and a loop that
    # counts unsigned, give no count to rely on either.
    wide = (
        _looped(20)
        .replace("%c0 = arith.constant 0 : index", f"%c0 = arith.constant {-(2**63)} : index")
        .replace("%c3 = arith.constant 3 : index", f"%c3 = arith.constant {2**63 - 1} : index")
        .replace("%c1 = arith.constant 1 : index", f"%c1 = arith.constant {2**62} : index")
    )
    unsigned = _looped(20).replace("scf.for %i = %c0", "scf.for unsigned %i = %c0")
    typed = re.sub(r"(%c[013] = arith\.constant \d) : index", r"\1 : i64", _looped(20)).replace(
        f"-> ({T}) {{\n      %w", f"-> ({T}) : i64 {{\n      %w"
    )
    for text in (wide, unsigned, typed):
        assert text != _looped(20)
        with pytest.raises(
            pulse.CompilationError, match="in a loop whose bounds are not constants"
        ):
            pulse.load(text, entry="shot", qudits=system.qudits[:1])
    # And a loop that is not an scf.for has no trips to count.
    whiled = _looped(20)
    head = whiled[whiled.index("    %out:2 = scf.for") : whiled.index("      %w = pulse.square")]
    whiled = whiled.replace(
        head,
        f"    %no = arith.constant false\n"
        f"    %out:2 = scf.while (%l = %a, %lt = %at) : ({T}) -> ({T}) {{\n"
        f"      scf.condition(%no) %l, %lt : {T}\n"
        f"    }} do {{\n"
        f"    ^bb0(%l: !pulse.drive_line, %lt: !pulse.tone):\n",
    ).replace("period_cycles = array<i64: 20>, ", "")
    with pytest.raises(pulse.CompilationError, match="neither a branch nor an scf.for loop"):
        pulse.load(whiled, entry="shot", qudits=system.qudits[:1])


def _attribute(value):
    return MODULE.replace("module {", "module attributes {made.by = " + value + "} {", 1)


@pytest.mark.parametrize(
    "value",
    [
        "[" * 300 + "1" + "]" * 300,
        "[" + ", ".join(["affine_set<(d0) : (d0 - 1 >= 0, 9 - d0 >= 0)>"] * 300) + "]",
        "[" + ", ".join(["-1"] * 300) + "]",
        '"((([[[{{{<<< // \\" )))"',
    ],
    ids=["nested-lists", "many-comparisons", "negative-numbers", "brackets-in-a-string"],
)
def test_the_text_is_parsed_as_it_is_given(system, value):
    # Nothing is read before the parser reads it: what parses, loads.
    text = "// " + "-" * 300 + "\n// ((((" * 100 + "\n" + _attribute(value)
    assert events(pulse.load(text, entry="round", qudits=system.qudits[:2])) == events(
        pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    )


def test_a_function_not_valid_as_written_is_refused_with_the_reasons(system, monkeypatch):
    from cudaq_pulse._native import _cudaq_pulse_native as native

    real = native.PulseModuleBuilder

    class Refusing:
        def __init__(self):
            self._real = real()

        def load_callables(self, text):
            return self._real.load_callables(text)

        def load_link_names(self, text):
            return self._real.load_link_names(text)

        def load_function(self, *arguments):
            raise RuntimeError(
                "load: round is not valid as a loaded function\n"
                "  loc(\"-\":3:5): 'pulse.drive' op the reason the verifier gave"
            )

    monkeypatch.setattr(native, "PulseModuleBuilder", Refusing)
    with pytest.raises(pulse.CompilationError) as refused:
        pulse.load(MODULE, entry="round", qudits=system.qudits[:2])
    assert refused.value.code == "wrong_argument"
    assert "is not valid as a loaded function" in str(refused.value)
    assert "the reason the verifier gave" in str(refused.value)


def _nested_ifs(count):
    lines = [
        "module {",
        f"  func.func @deep(%q: !pulse.qudit) {TIMED} {{",
        "    %c = arith.constant true",
    ]
    lines += ["    scf.if %c {"] * count + ["    }"] * count
    lines += ["    return", "  }", "}"]
    return "\n".join(lines)


def test_regions_nested_too_deep_are_refused_after_parsing(system):
    from cudaq_pulse._native import _cudaq_pulse_native as native

    # As bytecode, which the scan before parsing does not read.
    def bytecode(count):
        return native.MLIRPipeline().parse_to_module(_nested_ifs(count)).serialize()[0]

    assert pulse.load(bytecode(200), entry="deep", qudits=system.qudits[:1]) is not None
    assert pulse.load(bytecode(254), entry="deep", qudits=system.qudits[:1]) is not None
    with pytest.raises(pulse.CompilationError, match="nests regions deeper than 256"):
        pulse.load(bytecode(255), entry="deep", qudits=system.qudits[:1])


def test_a_module_inside_the_module_is_refused(system):
    inner = MODULE.strip()
    nested = "module {\n" + inner.replace("module {", "module @inner {", 1) + "\n}\n"
    with pytest.raises(pulse.CompilationError, match="has symbols of its own"):
        pulse.load(nested, entry="round", qudits=system.qudits[:2])


def test_a_module_inside_a_function_is_refused(system):
    inside = MODULE.replace(
        "    %n = arith.constant 16 : i64",
        "    builtin.module {\n    }\n    %n = arith.constant 16 : i64",
    )
    assert inside != MODULE
    with pytest.raises(pulse.CompilationError, match="has symbols of its own"):
        pulse.load(inside, entry="round", qudits=system.qudits[:2])


LOOPS = f"""
module {{
  func.func private @x(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) {_long(8)} {{
    %n = arith.constant 8 : i64
    %amp = arith.constant 0.5 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %n, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %w, %t <{{duration_cycles = 8 : i64, start_cycles = 0 : i64}}>
        : !pulse.drive_line, !pulse.waveform, !pulse.tone -> {T}
    return %d1, %t1 : {T}
  }}
  func.func @shot(%qa: !pulse.qudit) {TIMED} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a, %at = pulse.get_drive_line %qa : (!pulse.qudit) -> ({T})
    %out:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%l = %a, %lt = %at) -> ({T}) {{
      %in:2 = scf.for %j = %c0 to %c3 step %c1 iter_args(%m = %l, %mt = %lt) -> ({T}) {{
        %m1, %mt1 = pulse.call @x(%m, %mt)
            <{{duration_cycles = 8 : i64, PERIODS, start_cycles = 4 : i64}}> : ({T}) -> ({T})
        scf.yield %m1, %mt1 : {T}
      }}
      scf.yield %in#0, %in#1 : {T}
    }}
    return
  }}
}}
"""


def test_a_call_states_one_period_per_loop_around_it(system):
    qudits = system.qudits[:1]
    two = LOOPS.replace("PERIODS", "period_cycles = array<i64: 100, 20>")
    assert [start for _, _, start, _ in events(pulse.load(two, entry="shot", qudits=qudits))] == [
        4,
        24,
        44,
        104,
        124,
        144,
    ]
    one = LOOPS.replace("PERIODS", "period_cycles = array<i64: 20>")
    fix = "one period per scf.for around the call, outermost first"
    with pytest.raises(
        pulse.CompilationError, match=rf"inside 2 scf.for loop\(s\) and states 1 period.*{fix}"
    ):
        pulse.load(one, entry="shot", qudits=qudits)
    outside = CALLS.replace(_at(40, 8), _at(40, 8, "period_cycles = array<i64: 9>"))
    assert outside != CALLS
    with pytest.raises(pulse.CompilationError, match=r"inside 0 scf.for loop\(s\) and states 1"):
        pulse.load(outside, entry="shot", qudits=system.qudits[:2])
    # A called function has no loop: its calls state no period of their own.
    within = CALLS.replace(_at(0, 8), _at(0, 8, "period_cycles = array<i64: 9>"))
    assert within != CALLS
    with pytest.raises(pulse.CompilationError, match="a called function has no loop"):
        pulse.load(within, entry="shot", qudits=system.qudits[:2])


def test_the_entry_is_a_function(system):
    machine = _links()
    _, text = _as_written(system, machine)
    name = re.search(r"classical\.callable @(\w+)", text).group(1)
    with pytest.raises(pulse.CompilationError, match=f"{name} is not a function"):
        pulse.load(text, entry=name, qudits=system.qudits[:2], control_system=machine)


def test_what_a_loaded_function_may_not_hold(system):
    qudits = system.qudits[:2]
    far = re.sub(r"start_cycles =\s+32 :\s+i64", f"start_cycles = {2**60} : i64", MODULE)
    assert far != MODULE
    with pytest.raises(pulse.CompilationError, match="outside 0 to 2\\^50 cycles"):
        pulse.load(far, entry="round", qudits=qudits)
    by_value = MODULE.replace(
        "    %n = arith.constant 16 : i64",
        "    %fn = func.constant @other : (!pulse.qudit) -> ()\n    %n = arith.constant 16 : i64",
    )
    assert by_value != MODULE
    with pytest.raises(pulse.CompilationError, match="takes a function as a value"):
        pulse.load(by_value, entry="round", qudits=qudits)
    selects = _chain(1, 1).replace(
        f"func.func private @leaf(%d: !pulse.drive_line, %t: !pulse.tone) -> ({T}) attributes {{",
        "func.func private @leaf(%d: !pulse.drive_line, %t: !pulse.tone) -> "
        f"({T}) attributes {{pulse.selected_values = 1 : i64, ",
    )
    assert "pulse.selected_values" in selects
    with pytest.raises(pulse.CompilationError, match="leaf selects values at run time"):
        pulse.load(selects, entry="shot", qudits=system.qudits[:1])


def test_attributes_on_results_and_arguments_are_dropped(system):
    text = MODULE.replace("-> i1 attributes", "-> (i1 {made.note = 1 : i64}) attributes").replace(
        "@round(%a: !pulse.qudit,", "@round(%a: !pulse.qudit {made.note = 2 : i64},"
    )
    assert text.count("made.note") == 2
    program = pulse.load(text, entry="round", qudits=system.qudits[:2])
    assert "made." not in program.mlir()
    assert len(events(program)) == 3


def test_a_classical_function_named_main_is_refused(system):
    machine = _links()
    _, text = _as_written(system, machine)
    name = re.search(r"classical\.callable @(\w+)", text).group(1)
    key = re.search(r"dispatch_key = (\d+) : i64", text).group(0)
    # Under another name the function keeps its key, which it then states.
    clash = (
        text.replace(f"@{name}", "@main")
        .replace("func.func @main(", "func.func @entry(")
        .replace(key, key + ", dispatch_key_override")
    )
    with pytest.raises(pulse.CompilationError, match="classical function named main"):
        pulse.load(
            clash,
            entry="entry",
            qudits=[system.qudits[0], system.qudits[2]],
            control_system=machine,
        )


def test_a_link_no_classical_function_travels_on_is_refused(system):
    machine = _links()
    _, text = _as_written(system, machine)
    other = text.replace(
        "classical.get_link {device = 0 : i64}", "classical.get_link {device = 7 : i64}"
    )
    assert other != text
    with pytest.raises(pulse.CompilationError, match="gets link 7, which none"):
        pulse.load(
            other, entry="main", qudits=[system.qudits[0], system.qudits[2]], control_system=machine
        )


def test_a_load_beside_a_failing_load_on_another_thread_keeps_each_ones_reasons(system):
    import threading

    qudits = system.qudits[:2]
    broken = MODULE.replace("pulse.square %n, %amp, %zero", "pulse.square %n, %amp, %missing", 1)
    assert broken != MODULE
    failures, stop = [], threading.Event()

    def fails():
        while not stop.is_set():
            try:
                pulse.load(broken, entry="round", qudits=qudits)
                failures.append("the broken module loaded")
            except pulse.CompilationError as error:
                # Its own reason, with the parser's diagnostic, every time.
                if "cannot read the module" not in str(error) or "undeclared SSA value" not in str(
                    error
                ):
                    failures.append(str(error))

    def loads():
        for _ in range(300):
            if events(pulse.load(MODULE, entry="round", qudits=qudits))[0][0] != "drive":
                failures.append("a good module loaded wrong")

    workers = [threading.Thread(target=fails) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        loads()
    finally:
        stop.set()
        for worker in workers:
            worker.join()
    assert failures == []
