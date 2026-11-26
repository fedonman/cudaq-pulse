# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A list of lines that several names hold is one list to a
measurement-conditioned if.

The if keeps one record of each list of lines its arms may write, however
many local names hold the list, and puts each list back once between its
arms. A kernel with further names of its lists traces to the module of the
kernel without them, and the if hands its arms as many records as that
kernel's does.
"""

from __future__ import annotations

import random

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel import _emitter
from cudaq_pulse.kernel._branch_journal import _put_back_order
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q


class _Held(list):
    """A list subclass, which an if copies and reads through its own methods."""

    reads = 0

    def __getitem__(self, i):
        _Held.reads += 1
        return list.__getitem__(self, i)


_SETUP = """\
    lines = [get_drive_line(q) for q in qudits]
    probes = [get_readout_line(q) for q in qudits]
"""
_BITS = "    bits = [readout(probes[k], square(40, 0.05)) for k in range(4)]\n"

# Kernels, each as (the further names, the rest): the rest names a list
# through {a} and {b}, which are the further names `again` and `more` in the
# kernel with them and the lists' first names in the kernel without.
_NAMED = {
    "many names of one list": (
        "".join(f"    n{i:02d} = lines\n" for i in range(16)) + "    again = lines\n",
        _BITS
        + """\
    for k in range(4):
        if bits[k]:
            drive({a}[k], square(8, 0.1))
        else:
            drive(lines[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "probes"),
    ),
    "names of two lists in turn": (
        "".join(f"    n{i:02d} = {'lines' if i % 2 else 'probes'}\n" for i in range(16))
        + "    again = lines\n    more = probes\n",
        """\
    for k in range(4):
        if readout({b}[k], square(40, 0.05)):
            drive({a}[k], square(8, 0.1))
        else:
            wait(probes[k], 8)
            drive(lines[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "probes"),
    ),
    "names bound between ifs": (
        "",
        _BITS
        + """\
    for k in range(4):
        if k == 1:
            {a} = lines
        if k == 2:
            {b} = probes
        if k == 3:
            {a} = lines
        if bits[k]:
            drive({a}[k] if k else lines[k], square(8, 0.1))
        else:
            wait({b}[k] if k > 1 else probes[k], 8)
    drive(lines[0], square(8, 0.3))
""",
        ("lines", "probes"),
    ),
    "a list and a subclass, their names in turn": (
        "    again = lines\n    more = held\n    zz = lines\n    zzz = held\n",
        _BITS
        + """\
    for k in range(4):
        if bits[k]:
            drive({a}[k], square(8, 0.1))
            drive({b}[k], square(8, 0.1))
        else:
            drive(held[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
    drive({b}[1], square(8, 0.3))
""",
        ("lines", "held"),
    ),
    "a list that grows in both arms": (
        "    again = lines\n    more = lines\n",
        _BITS
        + """\
    spare = [get_drive_line(q) for q in qudits[4:8]]
    for k in range(4):
        if bits[k]:
            {a}.append(spare[k])
            drive({a}[-1], square(8, 0.1))
        else:
            lines.append(spare[k])
            drive({b}[k], square(8, 0.2))
    drive({a}[5], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
    "a list that grows in one arm": (
        "    again = lines\n",
        _BITS
        + """\
    for k in range(4):
        if bits[k]:
            {a} += [lines[k]]
            drive({a}[-1], square(8, 0.1))
    drive(lines[0], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
    "a list changed and put back in an arm": (
        "    again = lines\n    more = lines\n",
        _BITS
        + """\
    for k in range(4):
        if bits[k]:
            {a}[0], {b}[3] = lines[3], lines[0]
            drive({a}[0], square(8, 0.1))
            {b}[0], {a}[3] = lines[3], lines[0]
        else:
            drive({b}[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
    "a name bound again in one arm, and back": (
        "    again = lines\n",
        _BITS
        + """\
    spare = [get_drive_line(q) for q in qudits[4:8]]
    for k in range(4):
        if bits[k]:
            drive({a}[k], square(8, 0.1))
            again = spare
            drive(again[k], square(8, 0.1))
            again = lines
        else:
            drive({a}[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
    "nested ifs": (
        "    again = lines\n    more = probes\n    zz = lines\n",
        _BITS
        + """\
    for k in range(3):
        if bits[k]:
            drive({a}[k], square(8, 0.1))
            if bits[k + 1]:
                drive(lines[k + 1], square(8, 0.1))
                wait({b}[k], 8)
            else:
                drive({a}[k], square(16, 0.1))
            drive(lines[k], square(8, 0.1))
        else:
            if bits[k + 1]:
                wait({b}[k + 1], 8)
            drive({a}[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "probes"),
    ),
    "an if in a loop kept as a loop": (
        "    again = lines\n    more = probes\n    zz = lines\n",
        """\
    for _ in range(200):
        if readout({b}[0], square(40, 0.05)):
            drive({a}[0], square(8, 0.1))
        else:
            drive(lines[1], square(8, 0.2))
        sync(lines[0], lines[1], probes[0])
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "probes"),
    ),
    "names held in containers": (
        "    again = lines\n    more = lines\n"
        '    keyed = {{"a": lines, "b": lines}}\n    pair = (lines, lines)\n'
        "    nest = [lines, lines]\n",
        _BITS
        + """\
    for k in range(4):
        if bits[k]:
            drive({a}[k], square(8, 0.1))
        else:
            drive({b}[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
    "a list read through containers that hold it": (
        '    keyed = {{"a": lines, "b": lines}}\n    pair = (lines, lines)\n'
        "    nest = [lines, lines]\n    again = lines\n",
        _BITS
        + """\
    for k in range(4):
        if bits[k]:
            drive({c}[k], square(8, 0.1))
            drive({d}[k], square(8, 0.1))
        else:
            drive({e}[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
    "a comprehension in an arm reads a further name": (
        "    again = lines\n    more = lines\n",
        _BITS
        + """\
    for k in range(3):
        if bits[k]:
            sync(*[{a}[j] for j in range(k, k + 2)])
            drive({b}[k], square(8, 0.1))
        else:
            drive({b}[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
    "a list that grows and shrinks in an arm": (
        "    again = lines\n    more = lines\n",
        _BITS
        + """\
    for k in range(4):
        if bits[k]:
            {a}.append(lines[k])
            drive({b}[-1], square(8, 0.1))
            {b}.pop()
        else:
            drive({b}[k], square(8, 0.2))
    drive({a}[0], square(8, 0.3))
""",
        ("lines", "lines"),
    ),
}
_THROUGH = {
    True: {"c": 'keyed["b"]', "d": "pair[1]", "e": "nest[0]"},
    False: {"c": "lines", "d": "lines", "e": "lines"},
}


def _kernel(case: str, named: bool, **options):
    names, rest, first = _NAMED[case]
    through = {"a": "again", "b": "more"} if named else {"a": first[0], "b": first[1]}
    source = (
        "def k(qudits):\n"
        + _SETUP
        + "    held = Held(get_drive_line(q) for q in qudits[8:12])\n"
        + (names.format() if named else "")
        + rest.format(**through, **_THROUGH[named])
    )
    scope = vars(pulse) | {"Held": _Held}
    exec(compile(source, f"<{case}>", "exec"), scope)
    return pulse.kernel(scope["k"], **options)


def _traced(kernel, monkeypatch=None) -> tuple[str, list[int]]:
    """The module *kernel* compiles to on twelve qudits, or its error, and
    the number of container records each of its measured ifs hands its
    arms."""
    records: list[int] = []
    trace_arm = _emitter.PulseIREmitter._trace_arm
    opened = [False]
    branch = _emitter.PulseIREmitter._runtime_branch

    def counted_branch(self, *args):
        opened[0] = True
        return branch(self, *args)

    def counted_arm(self, view, at, join, start, snapshot):
        if opened[0]:
            opened[0] = False
            records.append(len(snapshot[0]))
        return trace_arm(self, view, at, join, start, snapshot)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_emitter.PulseIREmitter, "_runtime_branch", counted_branch)
        patch.setattr(_emitter.PulseIREmitter, "_trace_arm", counted_arm)
        try:
            module = pulse.compile(kernel).lower(Q[0:12]).mlir()
        except (CompilationError, RuntimeError) as e:
            module = f"error: {type(e).__name__}: {e}"
    return module, records


# The kernels above an arm of which adds to a list of lines, which an if
# does not take; the others compile.
_REFUSED = {"a list that grows in both arms", "a list that grows in one arm"}


@pytest.mark.parametrize("case", list(_NAMED))
def test_further_names_of_a_list_of_lines_leave_the_module_and_the_ifs_records(case):
    options = {"unroll_threshold": 16} if "kept as a loop" in case else {}
    plain, plain_records = _traced(_kernel(case, False, **options))
    named, named_records = _traced(_kernel(case, True, **options))
    assert plain.startswith("error") == (case in _REFUSED), plain
    assert named == plain
    assert plain_records
    # A subclass keeps a record for each of its names (the next test).
    if "subclass" not in case:
        assert named_records == plain_records


def test_a_subclass_of_list_is_copied_and_read_for_each_name_as_before():
    # A subclass keeps a record for each of its names (two more here), and
    # its own methods run as often as they did: 31 reads of its items
    # without the names, 77 with them.
    case = "a list and a subclass, their names in turn"
    _Held.reads = 0
    _, plain_records = _traced(_kernel(case, False))
    plain_reads = _Held.reads
    _Held.reads = 0
    _, named_records = _traced(_kernel(case, True))
    assert [n - p for n, p in zip(named_records, plain_records, strict=True)] == [2] * 4
    assert (plain_reads, _Held.reads) == (31, 77)


def _helper(twice: bool):
    """A kernel whose ifs are in a kernel it calls with its list of lines
    as two arguments, which names it once more; or with the ifs written
    where the call is."""
    scope = dict(vars(pulse))
    exec(
        compile(
            """def either(first, second, bits, k):
    third = first
    if bits[k]:
        drive(third[k], square(8, 0.1))
    else:
        drive(second[k], square(8, 0.2))
""",
            "<a helper>",
            "exec",
        ),
        scope,
    )
    scope["either"] = pulse.kernel(scope["either"])
    call = (
        "        either(lines, lines, bits, k)\n"
        if twice
        else """        if bits[k]:
            drive(lines[k], square(8, 0.1))
        else:
            drive(lines[k], square(8, 0.2))
"""
    )
    source = "def k(qudits):\n" + _SETUP + _BITS + "    for k in range(4):\n" + call
    exec(compile(source, "<a helper given the list twice>", "exec"), scope)
    return pulse.kernel(scope["k"])


def test_a_called_kernel_given_one_list_as_two_arguments_keeps_one_record():
    inline, _ = _traced(_helper(False))
    called, records = _traced(_helper(True))
    assert not inline.startswith("error"), inline
    assert called == inline
    assert records == [1] * 4


# A line that two lists hold is recorded, when an if puts its lists back, as
# the list's that it puts back last. Further names of the lists, wherever
# they come among the kernel's names, leave the module of the kernel
# without them.
_SHARED = {
    "a later name of the first list": "    c = a\n",
    "an earlier name of the first list": "    aa = a\n",
    "a subclass between two names of a list": "    c = a\n",
    "names of both in turn": "    c = a\n    d = b\n    e = a\n    f = b\n",
    "names of both, the first list's last": "    aa = b\n    c = b\n    d = a\n",
}


def _shared(case: str, named: bool):
    second = "Held([d1, d0])" if "subclass" in case else "[d1, d0]"
    source = (
        "def k(q0, q1):\n"
        "    probe = get_readout_line(q0)\n"
        "    d0 = get_drive_line(q0)\n"
        "    d1 = get_drive_line(q1)\n"
        "    a = [d0, d1]\n"
        f"    b = {second}\n"
        + (_SHARED[case] if named else "")
        + """\
    del d0, d1
    for _ in range(3):
        if readout(probe, square(40, 0.05)):
            drive(a[0], square(8, 0.1))
        else:
            drive(b[0], square(8, 0.2))
        if readout(probe, square(40, 0.05)):
            drive(b[1], square(8, 0.1))
    drive(a[0], square(8, 0.3))
    drive(b[0], square(8, 0.3))
"""
    )
    scope = vars(pulse) | {"Held": _Held}
    exec(compile(source, f"<{case}>", "exec"), scope)
    return pulse.kernel(scope["k"])


def _compiled(kernel) -> str:
    return pulse.compile(kernel).lower(Q[0], Q[1]).mlir()


@pytest.mark.parametrize("case", list(_SHARED))
def test_further_names_of_lists_that_share_a_line_leave_the_module(case):
    assert _compiled(_shared(case, True)) == _compiled(_shared(case, False))


def _last_name(named: bool, rounds: int = 1):
    """Two containers that hold line 2, a list and a subclass, at ifs whose
    arms reach it by name and through the list: with the list bound first
    and named twice more after the subclass, or bound after the subclass
    and not named again. With more *rounds*, the ifs after the first record
    only what changed since the one before."""
    one = "    one = [x0, x2]\n"
    held = "    held = Held([x1, x1, x2])\n    too = held\n"
    through = {"a": "again", "b": "more"} if named else {"a": "one", "b": "one"}
    source = (
        "def k(q0, q1, q2):\n"
        "    probe = get_readout_line(q0)\n"
        "    x0 = get_drive_line(q0)\n"
        "    x1 = get_drive_line(q1)\n"
        "    x2 = get_drive_line(q2)\n"
        + (one + held + "    again = one\n    more = one\n" if named else held + one)
        + """\
    drive({a}[0], square(8, 0.2))
    if readout(probe, square(40, 0.05)):
        drive({b}[1], square(8, 0.3))
    else:
        wait(x2, 8)
        {b}[0], {b}[1] = {b}[1], {b}[0]
    drive(too[-3], square(8, 0.1))
    drive(x2, square(8, 0.4))
""".format(**through)
        * rounds
    )
    scope = vars(pulse) | {"Held": _Held}
    exec(compile(source, "<the last name>", "exec"), scope)
    return pulse.kernel(scope["k"])


def _module_or_error(kernel) -> str:
    try:
        return pulse.compile(kernel).lower(Q[0], Q[1], Q[2]).mlir()
    except CompilationError as e:
        return f"error: {e}"


@pytest.mark.parametrize("rounds", [1, 2])
def test_a_list_named_again_is_put_back_as_one_bound_where_its_last_name_comes(rounds):
    # The if puts its containers back in the order of the names that hold
    # them, a list of lines once, where the last of its names comes: here
    # after the subclass, as the list bound after it is. One round compiles;
    # the second if of two is refused, with the same words either way.
    last = _module_or_error(_last_name(False, rounds))
    assert last.startswith("error") == (rounds == 2), last
    assert _module_or_error(_last_name(True, rounds)) == last


def _raises_in_an_arm(named: bool, marks: list[int]):
    source = (
        "def k(qudits):\n"
        + _SETUP
        + ("    again = lines\n    more = lines\n" if named else "")
        + _BITS
        + f"""\
    for k in range(4):
        if bits[k]:
            drive({"again" if named else "lines"}[k], square(8, 0.1))
            marks.append(k)
            lines[0], lines[1] = lines[1], lines[0]
            drive(lines[k], square(8, 0.1), tone=3)
        else:
            drive(lines[k], square(8, 0.2))
"""
    )
    scope = vars(pulse) | {"marks": marks}
    exec(compile(source, "<an error in an arm>", "exec"), scope)
    return pulse.kernel(scope["k"])


def test_an_error_in_an_arm_is_the_error_without_the_names():
    # The trace stops at the drive on no tone in the then arm of the first
    # if: the error, what the arm did before it to a list outside the
    # kernel, and a compile after it are the same with further names of
    # the list of lines.
    left = {}
    for named in (False, True):
        marks: list[int] = []
        error, records = _traced(_raises_in_an_arm(named, marks))
        assert error.startswith("error: ") and "must be a tone" in error, error
        left[named] = (error, marks, records)
        case = "many names of one list"
        assert _traced(_kernel(case, named))[0] == _traced(_kernel(case, False))[0]
    assert left[True] == left[False]
    assert left[False][1] == [0]


def _put_back_by_name(names: list[tuple[str, bool]]) -> tuple[list[str], list[str]]:
    """The containers of an if whose names hold, in order, the lists
    *names* (each with whether it is put back from the slot log), as the
    if collects them, and the order ``_put_back_order`` gives."""
    lists: dict[str, list[str]] = {}
    containers: list[tuple[list[str], list[str] | None]] = []
    repeats: list[tuple[int, list[str]]] = []
    seen: set[str] = set()
    for name, logged in names:
        obj = lists.setdefault(name, [name])
        if not logged:
            containers.append((obj, [name]))
        elif name in seen:
            repeats.append((len(containers), obj))
        else:
            seen.add(name)
            containers.append((obj, None))
    first = [obj[0] for obj, _ in containers]
    # What a comprehension around the if builds joins the containers in an
    # arm, after the names: it is put back last.
    containers.append((["built"], ["built"]))
    back = _put_back_order(containers, repeats) if repeats else containers
    assert back[-1] is containers[-1]
    return first, [obj[0] for obj, _ in back[:-1]]


def test_a_list_is_put_back_where_its_last_name_comes():
    assert _put_back_by_name([("L", True), ("M", True), ("L", True)]) == (["L", "M"], ["M", "L"])
    assert _put_back_by_name([("L", True), ("L", True), ("M", True)]) == (["L", "M"], ["L", "M"])
    assert _put_back_by_name([("L", True), ("s", False), ("L", True), ("s", False)]) == (
        ["L", "s", "s"],
        ["s", "L", "s"],
    )
    rng = random.Random(7)
    for _ in range(500):
        names = [(name, name.isupper()) for name in rng.choices("LMNst", k=rng.randrange(1, 12))]
        first, back = _put_back_by_name(names)
        # One record for each list put back from the slot log, at its first
        # name, and for each name of any other container.
        expected_first = []
        for name, logged in names:
            if not logged or name not in expected_first:
                expected_first.append(name)
        assert first == expected_first
        # Put back in the order of the names, each logged list at its last.
        last = {name: at for at, (name, logged) in enumerate(names) if logged}
        assert back == [
            name for at, (name, logged) in enumerate(names) if not logged or last[name] == at
        ]
        assert len(back) == len(first)
