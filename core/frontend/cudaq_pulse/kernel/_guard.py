# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards that keep a cached kernel compilation faithful to Python semantics.

A kernel reads module globals, closure cells and attribute chains of either
(``cfg.amp``, ``math.pi``) while it is traced, and the compiled IR bakes the
values in. The guard records those reads, found statically in the decoded
bytecode of the kernel and of every Python helper it can reach, as a list of
steps that re-read them, in order, and compare with the values seen at trace
time; one loop runs the steps, so building a guard compiles no code. A cache
entry is used only while its guard holds, so rebinding a global or mutating a
list a kernel reads is seen on the next call.

A mutable container is compared by its contents, copied when the guard is
built. A dict the kernel only subscripts (``TABLE[key]`` at every read, and
never iterated or tested with ``in``) is instead compared at the items the
trace read, so the cost of a guard does not grow with the dict's size.
"""

from __future__ import annotations

import builtins
import copy
import keyword
import types
import weakref
from itertools import compress
from operator import itemgetter
from typing import TYPE_CHECKING

from . import _bytecode_normalize as bc

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable, Collection, Iterable, Mapping
    from typing import Any

    _Read = tuple[str, str, tuple[str, ...], bool]
    # Bound by _load (below).
    _generated: Callable[[tuple[tuple[Any, ...], ...]], Callable[[], bool]]
    _CODE: dict[str, Callable[..., Callable[[], bool]]]
    _CODE_LIMIT: int
    _CODE_LOCK: threading.Lock

_IMMUTABLE = (int, float, complex, str, bytes, bool, types.NoneType, range)
_SNAPSHOT = (list, dict, set, bytearray)
_MAX_HELPER_DEPTH = 8


def _always_false() -> bool:
    return False


def _immutable(value: Any) -> bool:
    if isinstance(value, _IMMUTABLE):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable(v) for v in value)
    return False


# A copy to compare a mutable container's contents against, or None.
def _snapshot(value: Any) -> Any | None:
    if isinstance(value, _SNAPSHOT):
        try:
            return copy.deepcopy(value)
        except Exception:
            return None
    if type(value).__module__ == "numpy" and type(value).__name__ == "ndarray":
        return value.copy()
    return None


# A copy of a dict item to compare its contents against, or None when
# a copy does not compare equal to it (an object without ``__eq__``).
def _item_snapshot(value: Any) -> Any | None:
    snap = _snapshot(value)
    if snap is not None:
        return snap
    try:
        snap = copy.deepcopy(value)
    except Exception:
        return None
    return snap if _same_contents(value, snap) else None


def _same_contents(value: Any, snapshot: Any) -> bool:
    if type(value) is not type(snapshot):
        return False
    if type(value).__name__ == "ndarray":
        if value.shape != snapshot.shape or value.dtype != snapshot.dtype:
            return False
        if (value == snapshot).all():
            return True
        # NaN is not equal to itself: NaN in the same places is the same.
        if value.dtype.kind not in "fc":
            return False
        import numpy

        return bool(numpy.array_equal(value, snapshot, equal_nan=True))
    try:
        return bool(value == snapshot)
    except Exception:
        return False


# What finding reads and generating checks use is made when the first guard
# is built, not at import: _reads is until then a stub that calls _load, which
# binds _reads, _generated and the code cache. A guard upgrades only after it
# was built, so _generated is always bound by then.
def _load() -> None:
    global _reads, _generated, _CODE, _CODE_LIMIT, _CODE_LOCK
    import threading

    # A code's canonical opcodes are read as bytes: 1 in is_read at a global
    # or closure read; end_mark follows the last instruction.
    is_read = bytearray(256)
    is_read[bc.LOAD_GLOBAL] = is_read[bc.LOAD_DEREF] = 1
    end_mark = b"\xff\xff"
    # Instructions that push one value and read nothing from the stack.
    push_one = frozenset({bc.LOAD_CONST, bc.LOAD_FAST, bc.LOAD_DEREF, bc.LOAD_GLOBAL})
    op_of = itemgetter(0)
    arg_of = itemgetter(1)
    # The reads of each code object; equal code objects read the same names.
    cache: dict[types.CodeType, list[_Read]] = {}
    limit = 4096

    def reads(code: types.CodeType) -> list[_Read]:
        """``(kind, name, attribute chain, indexed)`` for each global or closure
        read; *indexed* when the value read is subscripted at once with a key
        pushed by one instruction, so nothing else ever holds it."""
        if code in cache:
            return cache[code]
        try:
            prog = bc.decode(code)
        except Exception:
            return []
        instrs = prog.instrs
        ops = bytes(map(op_of, instrs)) + end_mark
        cellvars = code.co_cellvars
        found: list[_Read] = []
        for start in compress(range(len(instrs)), ops.translate(is_read)):
            op, arg = instrs[start]
            if op == bc.LOAD_GLOBAL:
                kind = "g"
            elif arg not in cellvars:
                kind = "c"
            else:
                continue
            end = start + 1
            while ops[end] == bc.LOAD_ATTR:
                end += 1
            chain = () if end == start + 1 else tuple(map(arg_of, instrs[start + 1 : end]))
            found.append((kind, arg, chain, ops[end] in push_one and ops[end + 1] == bc.SUBSCR))
        if len(cache) >= limit:
            cache.clear()
        cache[code] = found
        return found

    def generated(plan: tuple[tuple[Any, ...], ...]) -> Callable[[], bool]:
        """The guard's check as one generated function, its code compiled once
        per source (a bounded cache, safe to share between threads). The
        guards of one source share its globals too: what differs between them
        is in their closures, so a guard run after another of its source
        finds the code's global reads still specialized for it."""
        src, values = _source(plan)
        make = _CODE.get(src)
        if make is None:
            namespace: dict[str, Any] = {"_B": vars(builtins), "_same": _same_contents}
            exec(compile(src, "<pulse guard>", "exec"), namespace)
            with _CODE_LOCK:
                if len(_CODE) >= _CODE_LIMIT:
                    _CODE.clear()
                make = _CODE.setdefault(src, namespace["_make"])
        check: Callable[[], bool] = make(*values)
        return check

    _CODE, _CODE_LIMIT, _CODE_LOCK = {}, 256, threading.Lock()
    _reads, _generated = reads, generated


def _reads(code: types.CodeType) -> list[_Read]:
    _load()
    return _reads(code)


# Where a step reads its value from: a module global, a global that falls back
# to the builtins, a closure cell, an attribute or an item of an earlier
# step's value.
_GLOBAL, _OR_BUILTIN, _CELL, _ATTR, _ITEM = 0, 1, 2, 3, 4
# How a step compares the value it reads with the one seen at trace time.
_SAME_OBJECT, _EQUAL_VALUE, _EQUAL_METHOD = 0, 1, 2


# A guard runs as closures over its steps until it holds this many times;
# then it runs as one generated function (as fast as the reads written out),
# whose compiled code is shared by every guard of the same source.
UPGRADE_AFTER = 2


# The guard's check as closures: runs *plan*, true while no step fails.
#
# A step is ``(None, keys, names, -1, 0, globals, None)``, failing when a
# name is bound in *globals* (whose ``keys()`` these are), or ``(source, a,
# b, register, how, expected, snapshot)``: it reads a value (``a[b]``,
# ``a[b]`` else the builtin ``b``, ``a.cell_contents``, or the attribute or
# item ``b`` of the value in register ``a``), compares it with *expected*
# (and its contents with *snapshot*) and keeps it in *register* (``-1``: no
# later step reads it). Registers live in the call, so threads may check
# one guard at once. With *upgrade*, the check counts its calls and, at
# UPGRADE_AFTER, runs (and from then on is) the check *upgrade* returns.
def _closures(
    plan: tuple[tuple[Any, ...], ...],
    registers: int,
    upgrade: Callable[[], Callable[[], bool]] | None = None,
) -> Callable[[], bool]:
    builtin = vars(builtins)
    same = _same_contents
    calls = 0
    upgraded: Callable[[], bool] | None = None

    def check() -> bool:
        nonlocal calls, upgraded
        if upgrade is not None:
            if upgraded is not None:
                # Called through a reference taken before the upgrade.
                return upgraded()
            calls += 1
            if calls >= UPGRADE_AFTER:
                upgraded = upgrade()
                return upgraded()
        values: list[Any] = [None] * registers
        try:
            for source, a, b, register, how, expected, snapshot in plan:
                if source is None:
                    if not a.isdisjoint(b):
                        return False
                    continue
                if source == _GLOBAL:
                    value = a[b]
                elif source == _ATTR:
                    value = getattr(values[a], b)
                elif source == _CELL:
                    value = a.cell_contents
                elif source == _ITEM:
                    value = values[a][b]
                else:
                    value = a[b] if b in a else builtin[b]
                if value is not expected:
                    if how == _SAME_OBJECT:
                        return False
                    if how == _EQUAL_VALUE:
                        if value.__class__ is not expected.__class__ or value != expected:
                            return False
                    elif value != expected:
                        return False
                if snapshot is not None and not same(value, snapshot):
                    return False
                if register >= 0:
                    values[register] = value
        except Exception:
            return False
        return True

    return check


# The generated form of *plan*: the same reads, in the same order, with
# the same comparisons, as the source of ``_make`` and the values it takes.
# The source holds only the plan's shape and names, so guards of one kernel
# share it; each guard's values are its closure's.
def _source(plan: tuple[tuple[Any, ...], ...]) -> tuple[str, list[Any]]:
    names: list[str] = []
    values: list[Any] = []

    def take(name: str, value: Any) -> str:
        names.append(name)
        values.append(value)
        return name

    lines: list[str] = []
    for k, (source, a, b, register, how, expected, snapshot) in enumerate(plan):
        if source is None:
            g = take(f"_g{k}", expected)
            tests = " or ".join(f"{name!r} in {g}" for name in sorted(b))
            lines.append(f"if {tests}: return False")
            continue
        if source == _GLOBAL:
            expr = f"{take(f'_a{k}', a)}[{b!r}]"
        elif source == _OR_BUILTIN:
            m = take(f"_a{k}", a)
            expr = f"({m}[{b!r}] if {b!r} in {m} else _B[{b!r}])"
        elif source == _CELL:
            expr = f"{take(f'_a{k}', a)}.cell_contents"
        elif source == _ATTR:
            expr = f"r{a}.{b}"
        else:
            expr = f"r{a}[{take(f'_b{k}', b)}]"
        var = f"r{register}" if register >= 0 else "v"
        take(f"_e{k}", expected)
        lines.append(f"{var} = {expr}")
        if how == _EQUAL_VALUE:
            lines.append(
                f"if {var} is not _e{k} and ({var}.__class__ is not _e{k}.__class__"
                f" or {var} != _e{k}): return False"
            )
        elif how == _EQUAL_METHOD:
            lines.append(f"if {var} is not _e{k} and {var} != _e{k}: return False")
        else:
            lines.append(f"if {var} is not _e{k}: return False")
        if snapshot is not None:
            lines.append(f"if not _same({var}, {take(f'_s{k}', snapshot)}): return False")
    body = "".join(f"            {line}\n" for line in lines) or "            pass\n"
    src = (
        f"def _make({', '.join(names)}):\n"
        "    def _guard():\n"
        "        try:\n"
        f"{body}"
        "        except Exception:\n"
        "            return False\n"
        "        return True\n"
        "    return _guard\n"
    )
    return src, values


class Guard:
    """Re-reads a kernel's globals and closures; true while none changed.

    ``check`` starts as closures over the guard's steps and becomes generated
    code once the guard has held UPGRADE_AFTER times.
    """

    __slots__ = ("check", "reads", "impure", "plan", "registers", "__weakref__")

    def __init__(
        self,
        check: Callable[[], bool],
        reads: int,
        impure: bool,
        plan: tuple[tuple[Any, ...], ...],
        registers: int,
    ) -> None:
        self.check = check
        self.reads = reads
        self.impure = impure
        self.plan = plan
        self.registers = registers


def _attributes(value: Any) -> list[str]:
    """The attributes to compare of an object a kernel was passed and is
    cached for by identity: a dataclass's fields, else what its ``__dict__``
    holds; none for a class, module or function, or cudaq_pulse's own."""
    cls = type(value)
    code = (type, types.ModuleType, types.FunctionType, types.MethodType, types.BuiltinFunctionType)
    if isinstance(value, code) or cls.__module__.partition(".")[0] == "cudaq_pulse":
        return []
    fields = getattr(cls, "__dataclass_fields__", None)
    if fields is None:
        fields = getattr(value, "__dict__", None)
        if not isinstance(fields, dict):
            return []
    return [name for name in fields if name.isidentifier() and not keyword.iskeyword(name)]


def _argument_reads(code: types.CodeType, name: str) -> list[tuple[str, ...]] | None:
    """The attribute chains *code* reads off its parameter *name*
    (``p.amplitude``), or None when it uses the parameter otherwise: passes
    it on, stores it or keeps it in a closure."""
    if name in code.co_cellvars:
        return None
    try:
        instrs = bc.decode(code).instrs
    except Exception:
        return None
    other = (
        bc.STORE_FAST,
        bc.STORE_FAST_MAYBE,
        bc.DELETE_FAST,
        bc.LOAD_FAST_AND_CLEAR,
        bc.LOAD_DEREF,
        bc.STORE_DEREF,
    )
    chains: list[tuple[str, ...]] = []
    for start, (op, arg) in enumerate(instrs):
        if op == bc.LOAD_FAST and arg == name:
            end = start + 1
            while end < len(instrs) and instrs[end][0] == bc.LOAD_ATTR:
                end += 1
            if end == start + 1:
                return None
            chains.append(tuple(attr for _, attr in instrs[start + 1 : end]))
        elif op in other and arg == name:
            return None
    return chains


# Whether a read of *chain* off *value* meets a method, which may read any
# attribute of its object.
def _meets_method(value: Any, chain: tuple[str, ...]) -> bool:
    for attr in chain:
        try:
            value = getattr(value, attr)
        except Exception:
            return False
        if isinstance(value, (types.MethodType, types.BuiltinMethodType)):
            return True
    return False


def pin_arguments(
    guard: Guard, fn: types.FunctionType, pinned: Iterable[tuple[str | None, Any]]
) -> Guard:
    """*guard*, extended to compare the static arguments *pinned* that the
    cache keys by the object itself, so a change in place is seen:
    ``(parameter, value)``, or ``(None, value)`` for a value in a tuple, list
    or dict argument. An array or a container is compared by a copy of its
    contents, as a global is; an object by the attribute chains *fn* reads
    off its parameter (``p.amplitude``), or by all its attributes
    (``_attributes``) when *fn* uses it otherwise or calls a method of it.
    """
    if guard.impure:
        return guard
    plan = list(guard.plan)
    registers = guard.registers
    seen: set[int] = set()

    def whole(value: Any, source: int, a: Any, b: Any, depth: int) -> None:
        nonlocal registers
        if _immutable(value):
            plan.append((source, a, b, -1, _EQUAL_VALUE, value, None))
            return
        if isinstance(value, (types.MethodType, types.BuiltinMethodType)):
            plan.append((source, a, b, -1, _EQUAL_METHOD, value, None))
            return
        snap = _snapshot(value)
        names = (
            []
            if snap is not None or id(value) in seen or depth > _MAX_HELPER_DEPTH
            else _attributes(value)
        )
        if snap is None and not names and source == _GLOBAL:
            return  # Only its identity, which the cache key holds.
        register = -1
        if names:
            seen.add(id(value))
            register = registers
            registers += 1
        plan.append((source, a, b, register, _SAME_OBJECT, value, snap))
        for name in names:
            try:
                attribute = getattr(value, name)
            except Exception:
                continue
            whole(attribute, _ATTR, register, name, depth + 1)

    for name, value in pinned:
        reads = None
        if name is not None and _attributes(value):
            reads = _argument_reads(fn.__code__, name)
        if reads is None or any(_meets_method(value, chain) for chain in reads):
            whole(value, _GLOBAL, (value,), 0, 0)
            continue
        # The registers of the argument and of each attribute read off it
        # that a longer chain reads on from.
        parents: dict[tuple[str, ...], int] = {(): registers}
        plan.append((_GLOBAL, (value,), 0, registers, _SAME_OBJECT, value, None))
        registers += 1
        for chain in dict.fromkeys(reads):
            parent = value
            for n, attr in enumerate(chain, start=1):
                try:
                    child = getattr(parent, attr)
                except Exception:
                    break
                if n == len(chain):
                    whole(child, _ATTR, parents[chain[: n - 1]], attr, 1)
                elif chain[:n] not in parents:
                    parents[chain[:n]] = registers
                    registers += 1
                    step = (_ATTR, parents[chain[: n - 1]], attr, parents[chain[:n]])
                    plan.append((*step, _SAME_OBJECT, child, _snapshot(child)))
                parent = child
    pinned_guard = Guard(_always_false, guard.reads, False, tuple(plan), registers)
    steps = pinned_guard.plan
    ref = weakref.ref(pinned_guard)

    def upgrade() -> Callable[[], bool]:
        check = _generated(steps)
        held = ref()
        if held is not None:
            held.check = check
        return check

    pinned_guard.check = _closures(steps, registers, upgrade)
    return pinned_guard


def build_guard(
    fn: types.FunctionType,
    *,
    impure: bool = False,
    dict_items: Mapping[tuple[int, Any], tuple[dict[Any, Any], Any, Any]] | None = None,
    dict_whole: Collection[int] = (),
) -> Guard:
    """The guard for *fn*, from the values its reads have right now.

    *impure* marks a trace that evaluated a host call whose result may change
    between calls (``random.random()``); its guard never holds. *dict_items*
    are the dict items the trace read, ``(id(dict), key) -> (dict, key,
    value)``, and *dict_whole* the ids of the dicts it iterated or tested
    with ``in``; without *dict_items* every dict is compared whole.
    """
    if impure:
        return Guard(_always_false, 0, True, (), 0)

    # [register, source, a, b, value, item] for each value to compare,
    # planned once every read is known; [None, globals, name] for a name
    # unbound at trace time.
    lines: list[list[Any]] = []
    slots: dict[tuple[Any, ...], int] = {}
    visited: set[int] = set()
    # Values some read uses other than by an immediate subscript.
    escaped: set[int] = set()
    # Registers a later step reads an attribute or item of.
    parents: set[int] = set()

    def bind(
        value: Any, source: int, a: Any, b: Any, key: tuple[Any, ...], indexed: bool = False
    ) -> int:
        if not indexed:
            escaped.add(id(value))
        register = slots.get(key)
        if register is not None:
            return register
        register = len(slots)
        slots[key] = register
        if source in (_ATTR, _ITEM):
            parents.add(a)
        lines.append([register, source, a, b, value, False])
        return register

    def walk(func: types.FunctionType, depth: int) -> None:
        if id(func.__code__) in visited or depth > _MAX_HELPER_DEPTH:
            return
        visited.add(id(func.__code__))
        fn_globals = func.__globals__
        freevars = func.__code__.co_freevars
        cells = func.__closure__ or ()
        for kind, name, chain, indexed in _reads(func.__code__):
            if kind == "g":
                if name in fn_globals:
                    value = fn_globals[name]
                    source = _GLOBAL
                elif name in vars(builtins):
                    value = vars(builtins)[name]
                    source = _OR_BUILTIN
                else:
                    # Unresolved at trace time: a later binding must retrace.
                    lines.append([None, fn_globals, name])
                    continue
                key: tuple[Any, ...] = ("g", id(fn_globals), name)
                register = bind(value, source, fn_globals, name, key, indexed and not chain)
            else:
                if name not in freevars:
                    continue
                cell = cells[freevars.index(name)]
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                key = ("c", id(cell))
                register = bind(value, _CELL, cell, None, key, indexed and not chain)
            follow(value, depth)
            for n, attr in enumerate(chain, start=1):
                try:
                    value = getattr(value, attr)
                except Exception:
                    break
                key = (*key, attr)
                register = bind(value, _ATTR, register, attr, key, indexed and n == len(chain))
                follow(value, depth)

    def follow(value: Any, depth: int) -> None:
        if isinstance(value, types.FunctionType):
            module = getattr(value, "__module__", "") or ""
            if not module.startswith("cudaq_pulse."):
                walk(value, depth + 1)
        inner = getattr(value, "__pulse_kernel_fn__", None)
        if isinstance(inner, types.FunctionType):
            walk(inner, depth + 1)

    walk(fn, 0)
    items: dict[int, list[tuple[Any, Any]]] = {}
    if dict_items:
        for container, index, value in dict_items.values():
            items.setdefault(id(container), []).append((index, value))
    plan: list[tuple[Any, ...]] = []
    # The globals of the step that tests unbound names, while it is the last.
    unbound_in: dict[str, Any] | None = None
    # The items of a dict compared at the items read are appended to *lines*
    # as it is planned, and so planned after it by this same loop.
    for line in lines:
        if line[0] is None:
            _, fn_globals, name = line
            # Unbound names in a row are tested together, each once.
            if unbound_in is fn_globals:
                _, keys, names, *rest = plan[-1]
                plan[-1] = (None, keys, names | {name}, *rest)
            else:
                plan.append((None, fn_globals.keys(), frozenset((name,)), -1, 0, fn_globals, None))
                unbound_in = fn_globals
            continue
        unbound_in = None
        register, source, a, b, value, item = line
        kept = register if register in parents else -1
        if _immutable(value):
            plan.append((source, a, b, kept, _EQUAL_VALUE, value, None))
        elif isinstance(value, (types.MethodType, types.BuiltinMethodType)):
            # A bound method is made anew on every attribute read; two are
            # equal when they bind the same function to the same object.
            plan.append((source, a, b, kept, _EQUAL_METHOD, value, None))
        elif (
            value.__class__ is dict
            and dict_items is not None
            and id(value) not in escaped
            and id(value) not in dict_whole
        ):
            # Only subscripted: the items read are all the trace used.
            read_items = items.get(id(value), ())
            if read_items:
                parents.add(register)
                kept = register
            plan.append((source, a, b, kept, _SAME_OBJECT, value, None))
            for index, read in read_items:
                # What the trace did with an item is not known: compare it whole.
                escaped.add(id(read))
                lines.append([-1, _ITEM, register, index, read, True])
        else:
            snap = _item_snapshot(value) if item else _snapshot(value)
            plan.append((source, a, b, kept, _SAME_OBJECT, value, snap))
    guard = Guard(_always_false, len(slots), False, tuple(plan), max(parents, default=-1) + 1)
    steps = guard.plan
    ref = weakref.ref(guard)

    def upgrade() -> Callable[[], bool]:
        # Threads that upgrade one guard at once each set an equal check.
        check = _generated(steps)
        held = ref()
        if held is not None:
            held.check = check
        return check

    guard.check = _closures(steps, guard.registers, upgrade)
    return guard
