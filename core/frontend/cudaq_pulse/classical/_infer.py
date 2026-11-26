# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A version's result lengths, inferred from the function's bytecode.

The kernel's decoder (``kernel._bytecode_normalize``) turns the function's
code object into canonical instructions once; a version runs them
abstractly, with its input lengths as numbers. Every list gets a length:
a number, or a sum over symbols with bounds where values decide it (how
many items pass a comprehension's ``if``, how many bits are set), kept with
the reason it isn't a number. A result needs a number; otherwise the
version is rejected with the line and the reason.

Paths split at a test the lengths don't decide and merge where they meet;
a ``try`` or ``with`` body also reaches its handler from wherever it can
raise. A ``for`` loop runs its body once from the entry state, to see what
changes, then again with those values unknown: whatever moves by the same
amount each pass ends at entry + passes * step, and anything else that
moves has no length after the loop. A ``while`` loop runs pass by pass
while its test is decided. A list something unseen could change (passed
to a call the analysis can't follow, captured by a nested function, put
in a container such a call could reach) loses its length. Compiled code
doesn't check an answer's length, so inference is sound by rejection: it
never guesses. ``pulse.Bits()`` and ``list[float]()`` are empty lists.
Source text is read only to word an error.

A run's input lengths are ints that record what is made of them: each
test of one is kept with its outcome, and each result length as a formula
of the input lengths. ``LengthRules`` keeps a function's runs; a later
version whose lengths pass the same tests, and that finds the same values
outside the body, takes the formulas evaluated instead of running the
body. The instructions few bodies use and the wording of an error are in
``_infer_rare``, imported when first needed.
"""

# Contents, in file order. Mostly bottom-up; the entry point section is last.
#
#   Errors and reasons               InferenceError; why a length isn't a number
#   Numbers: ints, or k + sum(c * s) over symbols with bounds
#   Numbers the input lengths make   _Trace, _Num: ints that record how they are used
#   Abstract values                  Seq, Const, Tup, Opq, Obj, Attr, It, Fn, Map, Slc
#   Builtins the analysis knows      tables of builtins and operators
#   Canonical opcodes                opcode sets and limits
#   States                           _State: the stack and names of one path
#   The analysis                     _Run, one run of a body:
#     Symbols and bounds               lengths that values decide
#     Values                           lengths and truth of abstract values
#     Running the instructions         sweep: the loop over instructions
#     Control flow                     where paths meet
#     Loops                            for and while
#     Instructions                     what sweep's arms call
#     Calls                            helpers, builtins, methods, generators
#   The part loaded when first needed  _rare: error wording, rare instructions
#   Reads from outside the function  globals, attributes, closure cells
#   Entry point                      LengthRules, infer_lengths

from __future__ import annotations

import builtins
import math
import operator as _op
import sys
import types
from typing import Any

from .._errors import PulseError
from ..kernel import _bytecode_normalize as _bc
from . import Bits

__all__ = ["InferenceError", "LengthRules", "infer_lengths"]

# ── Errors and reasons ──────────────────────────────────────────────────────


class InferenceError(TypeError):
    """A result's length can't be told from the body."""


class _NoLengthError(Exception):
    def __init__(self, why: Any) -> None:
        super().__init__()
        self.why = why


# A reason is ``(line, template, parts)``, worded only for an error: each
# part is text, a number, another reason, or ``(kind, decoded, index, ...)``
# naming source by an instruction (see ``_show``).


def _why(line: int, template: str, *parts: Any) -> tuple[int, str, tuple[Any, ...]]:
    return (line, template, parts)


def _line(dec: Any, i: int) -> int:
    pos = dec.positions[i] if 0 <= i < len(dec.positions) else None
    return int(pos.lineno or 0) if pos is not None and pos.lineno is not None else 0


# ── Numbers: ints, or k + sum(c * s) over symbols with bounds ───────────────


class Lin:
    __slots__ = ("k", "t")

    def __init__(self, k: int, t: dict[int, int]) -> None:
        self.k, self.t = k, t

    def __eq__(self, other: object) -> bool:
        return other.__class__ is Lin and self.k == other.k and self.t == other.t

    __hash__ = None  # type: ignore[assignment]


_NONE_T: dict[int, int] = {}


def _add(a: Any, b: Any, c: int = 1) -> Any:
    """``a + c * b``."""
    if a.__class__ in _INT and b.__class__ in _INT:
        return a + c * b
    ak, at = (a, _NONE_T) if a.__class__ in _INT else (a.k, a.t)
    bk, bt = (b, _NONE_T) if b.__class__ in _INT else (b.k, b.t)
    t = dict(at)
    for s, x in bt.items():
        y = t.get(s, 0) + c * x
        if y:
            t[s] = y
        else:
            del t[s]
    return Lin(ak + c * bk, t) if t else ak + c * bk


def _mul(a: Any, b: Any) -> Any:
    """``a * b``, or None when neither is an int."""
    if a.__class__ not in _INT:
        a, b = b, a
    if a.__class__ not in _INT:
        return None
    if b.__class__ in _INT:
        return a * b
    if not a:
        return 0
    return Lin(b.k * a, {s: x * a for s, x in b.t.items()})


def _fmt(n: Any) -> str:
    return str(n) if n.__class__ in _INT else "?"


# ── Numbers the input lengths make ──────────────────────────────────────────
#
# A run gives each input length as a _Num: an int, so the run goes exactly
# as it would with the plain number, that also says how it was made from
# the inputs (*e*, a node: ``(None, name)`` for an input, else ``(f, *args)``
# with nodes or plain values as arguments). Each test of one is recorded in
# the run's _Trace with its outcome; a later version whose lengths give the
# same outcomes runs the same way, so its results are the nodes evaluated.

_RAISES: Any = object()  # a node's outcome when computing it raises


class _Trace:
    """The tests a run made of its input lengths (node -> outcome, keyed
    so a repeated test is kept once), and what it read from outside the
    function."""

    __slots__ = ("guards", "reads")

    def __init__(self) -> None:
        self.guards: dict[Any, tuple[Any, Any]] = {}
        self.reads: dict[Any, tuple[Any, ...]] = {}

    def test(self, node: tuple[Any, ...], outcome: Any) -> None:
        key = (node[0], *[id(a) if a.__class__ is tuple else a for a in node[1:]])
        self.guards[key] = (node, outcome)


def _num(v: int, e: Any, tr: _Trace) -> Any:
    n = int.__new__(_Num, v)
    n.v, n.e, n.tr = v, e, tr
    return n


def _parts(a: Any, b: Any) -> tuple[Any, Any, Any, Any, _Trace]:
    """Plain values and nodes of two operands, one of them a _Num."""
    if a.__class__ is _Num:
        x, ea, tr = a.v, a.e, a.tr
    else:
        x = ea = int(a)
    if b.__class__ is _Num:
        y, eb, tr = b.v, b.e, b.tr
    else:
        y = eb = int(b)
    return x, y, ea, eb, tr


def _made(f: Any, x: Any, y: Any, node: tuple[Any, ...], tr: _Trace) -> Any:
    try:
        v = f(x, y)
    except (ArithmeticError, ValueError):
        tr.test(node, _RAISES)
        raise
    if v.__class__ in _INT:
        return _num(v, node, tr)
    tr.test(node, v)
    return v


def _arith(f: Any) -> tuple[Any, Any]:
    def left(a: Any, b: Any) -> Any:
        if not isinstance(b, int):
            return NotImplemented
        x, y, ea, eb, tr = _parts(a, b)
        return _made(f, x, y, (f, ea, eb), tr)

    def right(a: Any, b: Any) -> Any:
        if not isinstance(b, int):
            return NotImplemented
        y, x, eb, ea, tr = _parts(b, a)
        return _made(f, y, x, (f, eb, ea), tr)

    return left, right


def _compare(f: Any) -> Any:
    def test(a: Any, b: Any) -> Any:
        if not isinstance(b, int):
            return NotImplemented
        x, y, ea, eb, tr = _parts(a, b)
        out = f(x, y)
        tr.test((f, ea, eb), out)
        return out

    return test


def _unary(f: Any) -> Any:
    def run(a: Any) -> Any:
        return _num(f(a.v), (f, a.e), a.tr)

    return run


def _exact(a: Any) -> int:
    """A use of the value the analysis doesn't follow: only this value
    gives the same run."""
    a.tr.test((int, a.e), a.v)
    v: int = a.v
    return v


class _Num(int):
    """A length (or a number made from lengths) of one version; see above.
    Its attributes are set by ``_num``: int subclasses can't have slots."""

    v: int
    e: Any
    tr: _Trace

    __add__, __radd__ = _arith(_op.add)
    __sub__, __rsub__ = _arith(_op.sub)
    __mul__, __rmul__ = _arith(_op.mul)
    __floordiv__, __rfloordiv__ = _arith(_op.floordiv)
    __truediv__, __rtruediv__ = _arith(_op.truediv)
    __mod__, __rmod__ = _arith(_op.mod)
    __pow__, __rpow__ = _arith(_op.pow)
    __lshift__, __rlshift__ = _arith(_op.lshift)
    __rshift__, __rrshift__ = _arith(_op.rshift)
    __and__, __rand__ = _arith(_op.and_)
    __or__, __ror__ = _arith(_op.or_)
    __xor__, __rxor__ = _arith(_op.xor)
    __lt__ = _compare(_op.lt)
    __le__ = _compare(_op.le)
    __gt__ = _compare(_op.gt)
    __ge__ = _compare(_op.ge)
    __eq__ = _compare(_op.eq)
    __ne__ = _compare(_op.ne)
    __neg__ = _unary(_op.neg)
    __pos__ = _unary(_op.pos)
    __abs__ = _unary(abs)
    __invert__ = _unary(_op.invert)
    __hash__ = int.__hash__

    def __bool__(self) -> bool:
        out = self.v != 0
        self.tr.test((bool, self.e), out)
        return out

    __index__ = __int__ = __trunc__ = __floor__ = __ceil__ = _exact

    def __float__(self) -> float:
        return float(_exact(self))

    def __round__(self, ndigits: Any = None) -> Any:
        return round(_exact(self), ndigits)

    def __divmod__(self, other: Any) -> Any:
        return divmod(_exact(self), other)

    def __rdivmod__(self, other: Any) -> Any:
        return divmod(other, _exact(self))


_INT = (int, _Num)


def _derive(f: Any, *args: Any) -> Any:
    """``f(*args)``, recorded as a node when an argument is a _Num."""
    tr = None
    for a in args:
        if a.__class__ is _Num:
            tr = a.tr
            break
    if tr is None:
        return f(*args)
    vals = [a.v if a.__class__ is _Num else a for a in args]
    node = (f, *[a.e if a.__class__ is _Num else a for a in args])
    try:
        v = f(*vals)
    except (ArithmeticError, ValueError, TypeError):
        tr.test(node, _RAISES)
        raise
    if v.__class__ in _INT:
        return _num(v, node, tr)
    tr.test(node, v)
    return v


def _eval(e: Any, env: dict[str, int], memo: dict[int, Any]) -> Any:
    """A node's value for *env*'s input lengths; _RAISES if computing it,
    or an argument, raises."""
    if e.__class__ is not tuple:
        return e
    r = memo.get(id(e), _MISSING)
    if r is _MISSING:
        f = e[0]
        if f is None:
            r = env[e[1]]
        else:
            args = [_eval(a, env, memo) for a in e[1:]]
            if _RAISES in args:
                r = _RAISES
            else:
                try:
                    r = f(*args)
                except (ArithmeticError, ValueError, TypeError):
                    r = _RAISES
        memo[id(e)] = r
    return r


def _range_len(*nums: Any) -> int:
    return len(range(*nums))


def _slice_len(n: int, start: Any, stop: Any, step: Any) -> int:
    return len(range(n)[slice(start, stop, step)])


# ── Abstract values ─────────────────────────────────────────────────────────


class Seq:
    """A list, tuple or string: *n* is its length (an int, a Lin, or None)
    with *why* when it isn't an int. Mutable on purpose, so names bound to
    one list share one box. *frozen* names a module-level list, *src* an
    input whose items it still holds; *elem* says what its items unpack
    to; *esc* marks a list put where an unseen call could reach it."""

    __slots__ = ("n", "why", "frozen", "src", "elem", "esc")

    def __init__(
        self, n: Any, why: Any = None, frozen: Any = None, src: Any = None, elem: Any = None
    ) -> None:
        self.n, self.why, self.frozen, self.src, self.elem = n, why, frozen, src, elem
        self.esc = False

    def lose(self, why: Any) -> None:
        if self.n is not None:
            if self.n.__class__ in _INT or self.why is None:
                self.why = why
            self.n, self.src = None, None


class Const:
    __slots__ = ("v", "why")

    def __init__(self, v: Any, why: Any = None) -> None:
        self.v, self.why = v, why


class Tup:
    __slots__ = ("items",)

    def __init__(self, items: list[Any]) -> None:
        self.items = items


class Opq:
    """A value the analysis doesn't follow; *why* says what it is, for when
    a length is asked of it. *truth* is an item's truth as a 0/1 symbol."""

    __slots__ = ("why", "truth")

    def __init__(self, why: Any = None, truth: Any = None) -> None:
        self.why, self.truth = why, truth


class Obj:
    """An object from the function's globals, builtins or an import."""

    __slots__ = ("obj", "name", "line")

    def __init__(self, obj: Any, name: str, line: int = 0) -> None:
        self.obj, self.name, self.line = obj, name, line


class Attr:
    """``value.name`` of a value that isn't a global."""

    __slots__ = ("recv", "name", "dec", "i")

    def __init__(self, recv: Any, name: str, dec: Any, i: int) -> None:
        self.recv, self.name, self.dec, self.i = recv, name, dec, i


class It:
    """An iterator: *n* passes (int, Lin or None with *why*)."""

    __slots__ = ("n", "why", "src", "elem")

    def __init__(self, n: Any, why: Any, src: Any = None, elem: Any = None) -> None:
        self.n, self.why, self.src, self.elem = n, why, src, elem


class Fn:
    """A function made in the body; *gen* for a generator expression."""

    __slots__ = ("code", "gen", "dec", "i")

    def __init__(self, code: Any, gen: bool, dec: Any, i: int) -> None:
        self.code, self.gen, self.dec, self.i = code, gen, dec, i


class Map:
    """A module-level dict whose values are lists of one length."""

    __slots__ = ("n", "count", "name")

    def __init__(self, n: int, count: int, name: str) -> None:
        self.n, self.count, self.name = n, count, name


class Slc:
    __slots__ = ("parts",)

    def __init__(self, parts: list[Any]) -> None:
        self.parts = parts


SCALAR = Opq()
NONE = Obj(None, "None")
_UNBOUND = Opq()  # what LOAD_FAST_AND_CLEAR saves for an unbound name
_SELF = Opq()  # the receiver slot of a 3.14 ``with``'s special method
_EXIT = Opq()  # 3.12-3.13 ``with``: its __exit__

_SEQUENCES = (Seq, Tup)

# ── Builtins the analysis knows ─────────────────────────────────────────────

_LEN, _RANGE, _SAME, _ENUM, _ZIP, _MAP, _FILTER, _FOLD, _SUM, _PURE = range(10)
_KNOWN: dict[Any, int] = {
    len: _LEN,
    range: _RANGE,
    list: _SAME,
    tuple: _SAME,
    reversed: _SAME,
    sorted: _SAME,
    enumerate: _ENUM,
    zip: _ZIP,
    map: _MAP,
    filter: _FILTER,
    min: _FOLD,
    max: _FOLD,
    abs: _FOLD,
    int: _FOLD,
    sum: _SUM,
    Bits: _SAME,
}
for _f in (float, bool, round, any, all, complex, str, isinstance, divmod, pow, hash):
    _KNOWN[_f] = _PURE
del _f

_ARITH: dict[Any, Any] = {}
for _a, _b in (
    ("add", "iadd"),
    ("sub", "isub"),
    ("mul", "imul"),
    ("floordiv", "ifloordiv"),
    ("mod", "imod"),
    ("pow", "ipow"),
    ("lshift", "ilshift"),
    ("rshift", "irshift"),
    ("and_", "iand"),
    ("or_", "ior"),
    ("xor", "ixor"),
):
    _ARITH[getattr(_op, _a)] = _ARITH[getattr(_op, _b)] = getattr(_op, _a)
del _a, _b
_INPLACE = frozenset(
    getattr(_op, n)
    for n in (
        "iadd isub imul ifloordiv imod ipow ilshift irshift iand ior ixor imatmul itruediv"
    ).split()
)
_ADD, _SUB, _MULT = _op.add, _op.sub, _op.mul
_CMP = frozenset((_op.lt, _op.le, _op.eq, _op.ne, _op.gt, _op.ge))
_MUTATORS = {"append": 1, "insert": 1, "pop": -1, "remove": -1}

# ── Canonical opcodes ───────────────────────────────────────────────────────

_ITER = _bc.ITER_SLOTS
_QUIET = frozenset(
    (
        _bc.STORE_FAST,
        _bc.LOAD_CONST,
        _bc.POP_TOP,
        _bc.COPY,
        _bc.SWAP,
        _bc.JUMP,
        _bc.STORE_FAST_MAYBE,
        _bc.LOAD_FAST_AND_CLEAR,
        _bc.BUILD_TUPLE,
        _bc.BUILD_SLICE,
        _bc.RETURN,
        _bc.NOTE_INPLACE,
    )
)
_CALLS = frozenset((_bc.CALL, _bc.CALL_KW, _bc.CALL_EX))
_CO_ASYNC = 0x80 | 0x100 | 0x200
_CO_GENERATOR = 0x20
_CO_STARRED = 0x04 | 0x08  # *args, **kwargs
_HELPER_DEPTH = 16  # helpers inside helpers the analysis follows
_ITSELF = object()  # a helper that calls itself
_UNROLL = 4096  # while-loop passes run one by one
_BUDGET = 200_000  # instructions per version


class _GiveUpError(Exception):
    pass


# ── States ──────────────────────────────────────────────────────────────────


class _State:
    """One path: the value stack, the locals, and the undecided tests it
    passed, as ``(index, taken, truth)``."""

    __slots__ = ("stack", "env", "path")

    def __init__(self, stack: list[Any], env: dict[str, Any], path: tuple[Any, ...]) -> None:
        self.stack, self.env, self.path = stack, env, path


def _cp(v: Any, memo: dict[int, Any]) -> Any:
    c = v.__class__
    if c is Seq:
        r = memo.get(id(v))
        if r is None:
            r = memo[id(v)] = Seq(v.n, v.why, v.frozen, v.src, v.elem)
            r.esc = v.esc
        return r
    if c is Tup:
        return Tup([_cp(x, memo) for x in v.items])
    return v


def _fork(st: _State) -> _State:
    memo: dict[int, Any] = {}
    return _State(
        [_cp(v, memo) for v in st.stack], {k: _cp(v, memo) for k, v in st.env.items()}, st.path
    )


def _lkey(n: Any) -> Any:
    return n if n.__class__ is not Lin else ("lin", n.k, tuple(sorted(n.t.items())))


def _sig(v: Any) -> Any:
    if v is None:
        return ("u",)
    c = v.__class__
    if c is Seq:
        return ("s", _lkey(v.n) if v.n is not None else None)
    if c is Const:
        return ("c", _lkey(v.v))
    if c is Tup:
        return ("t", tuple(_sig(x) for x in v.items))
    if c is Opq:
        return ("o",) if v is not _UNBOUND else ("ub",)
    if c is It:
        return ("i", _lkey(v.n) if v.n is not None else None)
    return ("x", id(v))


def _whys(values: list[Any]) -> list[Any]:
    out = []
    for v in values:
        c = v.__class__
        if c is Tup:
            out.extend(_whys(v.items))
        elif c in (Seq, Const, Opq, It) and v.why is not None:
            out.append(v.why)
    return out


def _rewhy(v: Any, within: Any) -> Any:
    """*v* with the reasons it gives for an unknown length or value passed
    through *within*."""
    c = v.__class__
    if c is Tup:
        return Tup([_rewhy(x, within) for x in v.items])
    if c is Seq:
        v.why = within(v.why)
    elif c is Const and v.why is not None:
        return Const(v.v, within(v.why))
    elif c is Opq and v.why is not None:
        return Opq(within(v.why), v.truth)
    elif c is It and v.why is not None:
        v.why = within(v.why)
    return v


def _boxes(v: Any, out: list[Seq] | None = None) -> list[Seq]:
    if out is None:
        out = []
    if v.__class__ is Seq:
        out.append(v)
    elif v.__class__ is Tup:
        for x in v.items:
            _boxes(x, out)
    return out


def _slot(st: _State, k: Any) -> Any:
    return st.env.get(k) if k.__class__ is str else st.stack[k]


def _set_slot(st: _State, k: Any, v: Any) -> None:
    if k.__class__ is str:
        st.env[k] = v
    else:
        st.stack[k] = v


# ── The analysis ────────────────────────────────────────────────────────────


class _Run:
    def __init__(
        self,
        fn: Any,
        code: Any,
        lengths: dict[str, Any],
        bits: frozenset[str],
        forced: set[str],
        trace: _Trace,
    ) -> None:
        self.g: dict[str, Any] = getattr(fn, "__globals__", {})
        self.cells: dict[str, Any] = dict(
            zip(code.co_freevars, getattr(fn, "__closure__", None) or (), strict=False)
        )
        self.lengths, self.bits, self.forced = lengths, bits, forced
        self.trace = trace
        self.changed: set[str] = set()  # globals the function changes
        self.lo: dict[int, Any] = {}
        self.hi: dict[int, Any] = {}
        self.iota: dict[int, str] = {}  # an item's truth symbol -> its input
        self.counts: dict[str, int] = {}  # input -> its set-items symbol
        self.escaped = 0
        self.steps = 0
        self.placeholders: set[int] = set()
        self.active: list[Any] = [code]  # the bodies running, outermost first

    # ── Symbols and bounds ──────────────────────────────────────────────────

    def sym(self, lo: Any, hi: Any) -> int:
        s = len(self.lo) + 1
        self.lo[s], self.hi[s] = lo, hi
        return s

    def bounds(self, n: Any) -> tuple[Any, Any]:
        if n.__class__ in _INT:
            return n, n
        lo = hi = n.k
        for s, c in n.t.items():
            a, b = self.lo[s], self.hi[s]
            if c < 0:
                a, b = b, a
            lo = None if lo is None or a is None else lo + c * a
            hi = None if hi is None or b is None else hi + c * b
        return lo, hi

    def count_of(self, src: str) -> Lin:
        """How many items of input *src* are true."""
        s = self.counts.get(src)
        if s is None:
            s = self.counts[src] = self.sym(0, self.lengths[src])
        return Lin(0, {s: 1})

    def max0(self, n: Any) -> Any:
        if n.__class__ in _INT:
            return max(0, n)
        lo, hi = self.bounds(n)
        if lo is not None and lo >= 0:
            return n
        if hi is not None and hi <= 0:
            return 0
        return None

    def decide(self, n: Any) -> bool | None:
        """Whether *n* is nonzero, if its bounds say."""
        if n.__class__ in _INT:
            return bool(n)
        lo, hi = self.bounds(n)
        if (lo is not None and lo > 0) or (hi is not None and hi < 0):
            return True
        if lo == 0 and hi == 0:
            return False
        return None

    # ── Values ──────────────────────────────────────────────────────────────

    def from_object(self, obj: Any, name: str, line: int) -> Any:
        c = obj.__class__
        if c is int:
            return Const(obj)
        if c is bool:
            return Const(int(obj))
        if c is list or c is tuple:
            return Seq(len(obj), frozen=name)  # read when the version is built
        if obj is None:
            return NONE
        if c is dict and obj:
            sizes = {len(v) if v.__class__ in (list, tuple) else -1 for v in obj.values()}
            if len(sizes) == 1 and -1 not in sizes:
                return Map(sizes.pop(), len(obj), name)
        return Obj(obj, name, line)

    def length(self, v: Any, use: Any) -> tuple[Any, Any]:
        """``(n, why)``: *v*'s length, an int or a Lin explained by *why*."""
        c = v.__class__
        if c is Seq:
            if v.n is None:
                raise _NoLengthError(v.why or _why(0, "`{}` has no fixed length", use))
            return v.n, v.why
        if c is Tup:
            return len(v.items), None
        if c is It:
            if v.n is None:
                raise _NoLengthError(v.why or _why(0, "`{}` has no fixed length", use))
            return v.n, v.why
        if c is Opq and v.why:
            raise _NoLengthError(v.why)
        if c is Obj and v is not NONE:
            raise _NoLengthError(_why(v.line, "`{}` is not a list the compiler can follow", v.name))
        if c is Attr:
            raise _NoLengthError(_why(_line(v.dec, v.i), "`{}` is not followed", ("x", v.dec, v.i)))
        raise _NoLengthError(_why(0, "`{}` is not a list the compiler can follow", use))

    def number(self, v: Any, use: Any) -> tuple[Any, Any]:
        if v.__class__ is Const:
            return v.v, v.why
        if v.__class__ is Opq and v.why:
            raise _NoLengthError(v.why)
        raise _NoLengthError(_why(0, "`{}` depends on values, not only on input lengths", use))

    def truth(self, v: Any) -> bool | None:
        c = v.__class__
        if c is Const:
            return self.decide(v.v)
        if c is Seq:
            if v.n is None:
                return None
            return self.decide(v.n) if v.n.__class__ not in _INT else v.n > 0
        if c is Tup:
            return bool(v.items)
        if v is NONE:
            return False
        if c is Obj:
            o = v.obj
            if callable(o) or o.__class__ is type(sys):
                return True
            return None
        if c is It or c is Fn or c is Map:
            return True
        return None

    def lose_all(self, v: Any, why: Any) -> None:
        for box in _boxes(v):
            if box.frozen is not None:
                self.changed.add(box.frozen)
            box.lose(why)

    def escape(self, v: Any) -> None:
        for box in _boxes(v):
            box.esc = True
            self.escaped += 1

    def lose_escaped(self, st: _State, why: Any) -> None:
        if not self.escaped:
            return
        for v in [*st.stack, *st.env.values()]:
            for box in _boxes(v):
                if box.esc:
                    box.lose(why)

    def mutate(self, box: Seq, dec: Any, i: int) -> bool:
        """A list the body changes; False for a module-level one."""
        if box.frozen is not None:
            self.changed.add(box.frozen)
            box.lose(
                _why(
                    _line(dec, i), "`{}` is a module-level list changed by the function", box.frozen
                )
            )
            return False
        box.src = None
        return box.n is not None

    # ── Running the instructions ────────────────────────────────────────────

    def sweep(
        self, dec: Any, st: _State | None, lo: int, hi: int, head: int, comp: Any, rets: list[Any]
    ) -> tuple[list[_State], list[tuple[int, _State]]]:
        """Run instructions ``lo..hi-1`` from *st* at *lo*. Paths meet where
        they join; a jump to *head* ends a pass (the states are returned
        first), and a jump out of the range is returned with its state."""
        instrs = dec.instrs
        notes = dec.notes
        hand = self.handlers(dec) if notes is not None and "handlers" in notes else None
        whiles = self.whiles(dec) if notes is not None else None
        pending: dict[int, list[_State]] = {}
        backs: list[_State] = []
        exits: list[tuple[int, _State]] = []
        i = lo

        def go(t: int, s: _State) -> None:
            if t == head:
                backs.append(s)
            elif lo <= t < hi:
                if t <= i:
                    raise _GiveUpError
                pending.setdefault(t, []).append(s)
            else:
                exits.append((t, s))

        def throw(s: _State) -> None:
            if hand is not None and hand[i] is not None:
                start, t, depth, lasti = hand[i]
                s = _fork(s)
                del s.stack[depth:]
                if lasti:
                    s.stack.append(SCALAR)
                s.stack.append(SCALAR)
                s.path = (*s.path, (start, None, None))
                go(t, s)

        def branch(s: _State, at: int) -> _State | None:
            """The conditional jump at *at*: send the path that jumps, return
            the one that falls through."""
            op, target = instrs[at]
            v = s.stack.pop()
            t = self.truth(v)
            jump_if = op == _bc.JUMP_IF_TRUE
            same = v if v.__class__ is Opq and v is not SCALAR else None
            if t is None and same is not None:
                for e in s.path:
                    if len(e) > 3 and e[3] is v:
                        t = e[1]  # this path has tested it already
            if t is None:
                truth = v.truth if v.__class__ is Opq else None
                other = _fork(s)
                other.path = (*s.path, (at, jump_if, truth, same))
                s.path = (*s.path, (at, not jump_if, truth, same))
                go(target, other)
                return s
            if t == jump_if:
                go(target, s)
                return None
            return s

        def meet(at: int, states: list[_State]) -> _State | None:
            if len(states) == 1:
                return states[0]
            if (
                instrs[at] == (_bc.COPY, 1)
                and at + 1 < len(instrs)
                and instrs[at + 1][0] in (_bc.JUMP_IF_FALSE, _bc.JUMP_IF_TRUE)
            ):
                # An `and`/`or` tests the value each path brings: test each
                # before they meet, so a value known on one path stays known.
                for s in states:
                    s.stack.append(s.stack[-1])
                    falls = branch(s, at + 1)
                    if falls is not None:
                        go(at + 2, falls)
                return None
            return self.merge(dec, states, comp)

        while True:
            if st is None:
                if not pending:
                    return backs, exits
                i = min(pending)
                st = meet(i, pending.pop(i))
                if st is None:
                    continue
            elif i in pending:
                states = pending.pop(i)
                states.append(st)
                st = meet(i, states)
                if st is None:
                    continue
            if i >= hi:
                exits.append((i, st))
                st = None
                continue
            self.steps += 1
            if self.steps > _BUDGET:
                raise _NoLengthError(
                    _why(_line(dec, i), "it takes too long to follow for this version")
                )
            if whiles is not None and i != head and i in whiles:
                after, out = self.while_loop(dec, st, i, whiles[i], comp, rets)
                for t, s in out:
                    go(t, s)
                st = after
                continue
            op, arg = instrs[i]
            if hand is not None and hand[i] is not None and op not in _QUIET and op not in _CALLS:
                throw(st)
            stack = st.stack
            env = st.env

            # The arms, by subject (search for the name):
            #   names         LOAD_FAST STORE_FAST STORE_FAST_MAYBE DELETE_FAST
            #                 LOAD_FAST_AND_CLEAR LOAD_DEREF STORE_DEREF
            #                 LOAD_GLOBAL LOAD_CONST LOAD_ATTR
            #   stack         POP_TOP COPY SWAP
            #   operators     BINARY UNARY NOT COMPARE IS CONTAINS
            #   calls         CALL CALL_KW CALL_EX RETURN GEN_YIELD
            #                 MAKE_GENEXPR SET_FUNCATTR
            #   jumps         JUMP JUMP_IF_FALSE JUMP_IF_TRUE JUMP_IF_NONE
            #                 JUMP_IF_NOT_NONE RAISE
            #   loops         GET_ITER FOR_ITER
            #   sequences     BUILD_LIST BUILD_TUPLE BUILD_SLICE LIST_APPEND
            #                 LIST_EXTEND UNPACK_SEQUENCE UNPACK_EX SUBSCR
            #                 STORE_SUBSCR DELETE_SUBSCR
            #   dicts, sets   BUILD_MAP BUILD_CONST_KEY_MAP MAP_ADD DICT_MERGE
            #                 BUILD_SET SET_ADD SET_UPDATE
            #   strings       BUILD_STRING CONVERT FORMAT
            #   imports       IMPORT_NAME IMPORT_FROM
            #   other         NOTE_INPLACE UNSUPPORTED (a ``while``, a ``try``,
            #                 or an instruction ``_infer_rare`` runs)
            if op == _bc.LOAD_FAST:
                v = env.get(arg)
                if v is None:
                    throw(st)
                    st = None
                    continue
                stack.append(v)
            elif op == _bc.STORE_FAST:
                env[arg] = stack.pop()
            elif op == _bc.LOAD_CONST:
                stack.append(self.const(arg, dec, i))
            elif op == _bc.LOAD_GLOBAL:
                stack.append(self.load_global(arg, dec, i))
            elif op == _bc.CALL or op == _bc.CALL_KW:
                names = len(stack.pop().items) if op == _bc.CALL_KW else 0
                args = stack[len(stack) - arg :]
                del stack[len(stack) - arg :]
                f = stack.pop()
                if f is _SELF or (f is NONE and stack and stack[-1] is _EXIT):
                    f = stack.pop()
                stack.append(self.call(st, f, args, names, dec, i))
                if hand is not None and hand[i] is not None:
                    throw(st)
            elif op == _bc.LOAD_ATTR:
                stack.append(self.attr(stack.pop(), arg, dec, i))
            elif op == _bc.BINARY:
                b = stack.pop()
                stack.append(self.binary(st, arg, stack.pop(), b, dec, i))
            elif op == _bc.SUBSCR:
                ix = stack.pop()
                stack.append(self.subscr(stack.pop(), ix, dec, i))
            elif op == _bc.POP_TOP:
                stack.pop()
            elif op == _bc.JUMP:
                go(arg, st)
                st = None
                continue
            elif op == _bc.JUMP_IF_FALSE or op == _bc.JUMP_IF_TRUE:
                st = branch(st, i)
                if st is None:
                    continue
            elif op == _bc.FOR_ITER:
                exit_ = arg[0]
                after, out = self.for_loop(dec, st, i, comp, rets)
                for t, s in out:
                    go(t, s)
                st = after
                i = exit_
                continue
            elif op == _bc.GET_ITER:
                stack.append(self.iterate(stack.pop(), dec, i))
                if _ITER > 1:
                    stack.extend([SCALAR] * (_ITER - 1))
            elif op == _bc.RETURN:
                rets.append((stack.pop(), dec, i, st))
                st = None
                continue
            elif op == _bc.COMPARE:
                b = stack.pop()
                stack.append(self.compare(arg, stack.pop(), b))
            elif op == _bc.BUILD_LIST:
                items = stack[len(stack) - arg :]
                del stack[len(stack) - arg :]
                for x in items:
                    self.escape(x)
                stack.append(Seq(arg))
            elif op == _bc.BUILD_TUPLE:
                items = stack[len(stack) - arg :]
                del stack[len(stack) - arg :]
                stack.append(Tup(items))
            elif op == _bc.LIST_APPEND:
                v = stack.pop()
                self.escape(v)
                box = stack[-arg]
                if box.__class__ is Seq and box.n is not None:
                    box.n = _add(box.n, 1)
            elif op == _bc.LIST_EXTEND:
                v = stack.pop()
                box = stack[-arg]
                for x in v.items if v.__class__ is Tup else ():
                    self.escape(x)
                if box.__class__ is Seq and box.n is not None:
                    try:
                        n, why = self.length(v, ("x", dec, i))
                        box.n = _add(box.n, n)
                        box.why = box.why or why
                    except _NoLengthError as c:
                        box.lose(c.why)
            elif op == _bc.UNPACK_SEQUENCE:
                v = stack.pop()
                if v.__class__ is Tup:
                    if len(v.items) != arg:
                        throw(st)
                        st = None
                        continue
                    stack.extend(reversed(v.items))
                else:
                    stack.extend([SCALAR] * arg)
            elif op == _bc.COPY:
                stack.append(stack[-arg])
            elif op == _bc.SWAP:
                stack[-1], stack[-arg] = stack[-arg], stack[-1]
            elif op == _bc.STORE_FAST_MAYBE:
                v = stack.pop()
                if v is _UNBOUND:
                    env.pop(arg, None)
                else:
                    env[arg] = v
            elif op == _bc.LOAD_FAST_AND_CLEAR:
                stack.append(env.pop(arg, _UNBOUND))
            elif op == _bc.BUILD_SLICE:
                parts = stack[len(stack) - arg :]
                del stack[len(stack) - arg :]
                stack.append(Slc(parts))
            elif op == _bc.LOAD_DEREF:
                v = env.get(arg)
                if v is None:
                    cell = self.cells.get(arg)
                    value = _MISSING if cell is None else self.read(_CELL, cell, arg)
                    if value is not _MISSING:
                        v = self.from_object(value, arg, _line(dec, i))
                    else:
                        throw(st)
                        st = None
                        continue
                stack.append(v)
            elif op == _bc.STORE_DEREF:
                env[arg] = stack.pop()
            elif op == _bc.DELETE_FAST:
                env.pop(arg, None)
            elif op == _bc.UNARY:
                stack.append(self.unary(arg, stack.pop()))
            elif op == _bc.NOT:
                v = stack.pop()
                known = self.truth(v)
                if known is not None:
                    stack.append(Const(int(not known)))
                elif v.__class__ is Opq and v.truth is not None:
                    stack.append(Opq(None, _add(1, v.truth, -1)))
                else:
                    stack.append(SCALAR)
            elif op == _bc.IS:
                b = stack.pop()
                a = stack.pop()
                known = self.is_none(a) if b is NONE else self.is_none(b) if a is NONE else None
                stack.append(SCALAR if known is None else Const(int(known != arg)))
            elif op == _bc.JUMP_IF_NONE or op == _bc.JUMP_IF_NOT_NONE:
                known = self.is_none(stack.pop())
                jump_if = op == _bc.JUMP_IF_NONE
                if known is None:
                    other = _fork(st)
                    other.path = (*st.path, (i, jump_if, None))
                    st.path = (*st.path, (i, not jump_if, None))
                    go(arg, other)
                elif known == jump_if:
                    go(arg, st)
                    st = None
                    continue
            elif op == _bc.CONTAINS:
                del stack[-2:]
                stack.append(SCALAR)
            elif op == _bc.STORE_SUBSCR:
                ix = stack.pop()
                obj = stack.pop()
                self.store_subscr(st, stack.pop(), obj, ix, dec, i)
            elif op == _bc.DELETE_SUBSCR:
                ix = stack.pop()
                _rare().delete_subscr(self, st, stack.pop(), ix, dec, i)
            elif op == _bc.BUILD_MAP or op == _bc.BUILD_CONST_KEY_MAP:
                k = 2 * arg if op == _bc.BUILD_MAP else arg + 1
                for x in stack[len(stack) - k :]:
                    self.escape(x)
                del stack[len(stack) - k :]
                stack.append(SCALAR)
            elif op == _bc.MAP_ADD:
                self.escape(stack.pop())
                stack.pop()
            elif op == _bc.DICT_MERGE:
                stack.pop()
            elif op == _bc.BUILD_SET:
                del stack[len(stack) - arg :]
                stack.append(
                    Seq(
                        None,
                        _why(
                            _line(dec, i),
                            "a set drops repeated values, so its length depends on them",
                        ),
                    )
                )
            elif op == _bc.SET_ADD:
                stack.pop()
                box = stack[-arg]
                if box.__class__ is Seq:
                    box.why = _why(
                        _line(dec, i),
                        "a set comprehension drops repeated values, so its length depends on them",
                    )
            elif op == _bc.SET_UPDATE:
                stack.pop()
            elif op == _bc.UNPACK_EX:
                before, after_n = arg
                v = stack.pop()
                if v.__class__ is Tup and len(v.items) >= before + after_n:
                    rest = v.items[before : len(v.items) - after_n]
                    items = [*v.items[:before], Tup(rest), *v.items[len(v.items) - after_n :]]
                else:
                    try:
                        n, why = self.length(v, ("x", dec, i))
                        rest_n: Any = _add(n, -(before + after_n))
                    except _NoLengthError:
                        rest_n, why = None, None
                    if rest_n is None:
                        why = _why(_line(dec, i), "unpacked from a list of unknown length")
                    star = Seq(rest_n, why if rest_n.__class__ not in _INT else None)
                    items = [*[SCALAR] * before, star, *[SCALAR] * after_n]
                stack.extend(reversed(items))
            elif op == _bc.BUILD_STRING:
                del stack[len(stack) - arg :]
                stack.append(SCALAR)
            elif op == _bc.CONVERT:
                stack[-1] = SCALAR
            elif op == _bc.FORMAT:
                if arg:
                    stack.pop()
                stack[-1] = SCALAR
            elif op == _bc.RAISE:
                del stack[len(stack) - arg :]
                st = None
                continue
            elif op == _bc.CALL_EX:
                kw = stack.pop() if arg else None
                star = stack.pop()
                f = stack.pop()
                if star.__class__ is Tup and kw is None:
                    stack.append(self.call(st, f, star.items, 0, dec, i))
                else:
                    stack.append(self.call(st, f, [star] + ([kw] if kw else []), -1, dec, i))
                if hand is not None and hand[i] is not None:
                    throw(st)
            elif op == _bc.IMPORT_NAME:
                fromlist = stack.pop()
                level = stack.pop()
                stack.append(_rare().import_name(self, arg, fromlist, level, dec, i))
            elif op == _bc.IMPORT_FROM:
                top = stack[-1]
                value = (
                    self.read(_ATTR, top.obj, arg)
                    if top.__class__ is Obj and top.obj is not None
                    else _MISSING
                )
                if value is not _MISSING:
                    stack.append(self.from_object(value, arg, _line(dec, i)))
                else:
                    stack.append(SCALAR)
            elif op == _bc.MAKE_GENEXPR:
                code = stack.pop()
                if arg:
                    stack.pop()
                stack.append(Fn(code.code, True, code.dec, code.i))
            elif op == _bc.SET_FUNCATTR:
                fn = stack.pop()
                self.capture_values(stack.pop(), dec, i)
                stack.append(fn)
            elif op == _bc.GEN_YIELD:
                stack.pop()
                box = env.get(".yielded")
                if box is not None and box.n is not None:
                    box.n = _add(box.n, 1)
                stack.append(NONE)
            elif op == _bc.NOTE_INPLACE:
                pass
            elif op == _bc.UNSUPPORTED:
                note = notes.get(i) if notes is not None else None
                if note is None:
                    raise _GiveUpError
                if note[0] == "while":
                    go(note[1], st)
                    st = None
                    continue
                if note[0] != "try":
                    st = _rare().raw(self, st, note[0], note[1], dec, i)
                    if st is None:
                        continue
            else:
                raise _GiveUpError
            i += 1

    # ── Control flow ────────────────────────────────────────────────────────

    def handlers(self, dec: Any) -> list[Any]:
        hand: list[Any] = [None] * len(dec.instrs)
        for start, end, target, depth, lasti in dec.notes["handlers"]:
            for j in range(start, min(end, len(hand))):
                hand[j] = (start, target, depth, lasti)
        return hand

    def whiles(self, dec: Any) -> dict[int, int] | None:
        out: dict[int, int] = {}
        for i, note in dec.notes.items():
            if i.__class__ in _INT and note[0] == "while":
                out[note[1]] = max(i, out.get(note[1], i))
        return out or None

    def merge(self, dec: Any, states: list[_State], comp: Any, k: int = 0) -> _State:
        """One state for paths that meet: a value both give stays; one that
        differs gets a length up to a symbol, with why."""
        if len(states) == 1:
            return states[0]
        first = states[0].path
        d = k
        while all(len(s.path) > d and s.path[d][:2] == first[d][:2] for s in states):
            d += 1
        groups: dict[Any, list[_State]] = {}
        for s in states:
            groups.setdefault(s.path[d][:2] if len(s.path) > d else None, []).append(s)
        if len(groups) == 1:
            # Paths no test told apart: a loop's end and its early exits, or
            # the raises a try's handler catches.
            fork = first[d - 1] if d and first[d - 1][1] is None else (-1, "exit", None)
            out = states[0]
            for other in states[1:]:
                out = self.join_states(dec, out, other, fork, comp)
            out.path = first[:d]
            return out
        order = sorted(groups, key=lambda e: (e is None, e is not None and e[1] is not True))
        merged = [self.merge(dec, groups[e], comp, d if e is None else d + 1) for e in order]
        fork = next(groups[e][0].path[d] for e in order if e is not None)
        out = merged[0]
        for other in merged[1:]:
            out = self.join_states(dec, out, other, fork, comp)
        out.path = first[:d]
        return out

    def join_states(self, dec: Any, a: _State, b: _State, fork: Any, comp: Any) -> _State:
        if len(a.stack) != len(b.stack):
            raise _GiveUpError
        memo: dict[Any, Any] = {}
        stack = [
            self.join(x, y, None, fork, dec, comp, memo)
            for x, y in zip(a.stack, b.stack, strict=False)
        ]
        env = dict(b.env)
        for name, x in a.env.items():
            y = b.env.get(name)
            # A name one path leaves unbound fails to read there: the other's value stands.
            env[name] = x if y is None else self.join(x, y, name, fork, dec, comp, memo)
        return _State(stack, env, a.path)

    def fork_reason(self, what: str, name: Any, sizes: str, fork: Any, dec: Any, comp: Any) -> Any:
        index = fork[0]
        line = _line(dec, index)
        if comp is not None:
            return _why(
                _line(comp[1], comp[2]),
                "`{}` is a comprehension with a condition (`if {}`), so its length depends on"
                " the values",
                comp,
                ("compif", *comp[1:]),
            )
        place = (
            ("try", dec, index)
            if fork[1] is None
            else ("exit", dec, index)
            if fork[1] == "exit"
            else ("fork", dec, index)
        )
        if name is None:
            return _why(
                line, "`{}` has " + what + " {} depending on its test", ("ifexp", dec, index), sizes
            )
        return _why(line, "`{}` has " + what + " {} depending on {}", name, sizes, place)

    def join_n(self, a: Any, b: Any, fork: Any) -> Any:
        if a is None or b is None:
            return None
        d = _add(a, b, -1)
        if d.__class__ not in _INT:
            return None
        truth = fork[2]
        if truth is not None and fork[1].__class__ is bool:
            return _add(b, _mul(d, truth))
        return _add(b, Lin(0, {self.sym(min(0, d), max(0, d)): 1}))

    def join(
        self, x: Any, y: Any, name: Any, fork: Any, dec: Any, comp: Any, memo: dict[Any, Any]
    ) -> Any:
        if x is y:
            return x
        cx, cy = x.__class__, y.__class__
        if cx is Seq and cy is Seq:
            key = (id(x), id(y))
            r = memo.get(key)
            if r is not None:
                return r
            clash = [memo[memo[j]] for j in (id(x), id(y)) if memo.get(j, key) != key]
            if clash:
                # One list on one path is two on the other: neither length holds.
                why = self.fork_reason("length", name, "?", fork, dec, comp)
                for other in clash:
                    other.lose(why)
                r = memo[key] = Seq(None, why)
                return r
            memo[id(x)] = memo[id(y)] = key
            if x.n == y.n:
                r = x
                if x.src != y.src:
                    r.src = None
                if x.frozen != y.frozen:
                    r.frozen = None
                r.esc = x.esc or y.esc
                r.why = x.why or y.why
            else:
                n = self.join_n(x.n, y.n, fork)
                why = x.why or y.why
                if why is None:
                    a_n = _fmt(x.n) if x.n is not None else "?"
                    b_n = _fmt(y.n) if y.n is not None else "?"
                    sizes = f"{a_n} or {b_n}"
                    why = self.fork_reason("length", name, sizes, fork, dec, comp)
                r = Seq(n, why)
                r.esc = x.esc or y.esc
            memo[key] = r
            return r
        if cx is Const and cy is Const:
            if x.v == y.v:
                return x
            n = self.join_n(x.v, y.v, fork)
            why = (
                x.why
                or y.why
                or self.fork_reason("value", name, f"{_fmt(x.v)} or {_fmt(y.v)}", fork, dec, comp)
            )
            return Const(n, why) if n is not None else Opq(why)
        if cx is Tup and cy is Tup and len(x.items) == len(y.items):
            return Tup(
                [
                    self.join(p, q, name, fork, dec, comp, memo)
                    for p, q in zip(x.items, y.items, strict=False)
                ]
            )
        if cx is Opq and cy is Opq:
            return x if x.why else y
        if cx is It and cy is It:
            if x.n == y.n:
                return x
            n = self.join_n(x.n, y.n, fork)
            return It(n, x.why or y.why or self.fork_reason("length", name, "?", fork, dec, comp))
        if cx is cy and cx in (Obj, Fn, Map) and (getattr(x, "obj", x) is getattr(y, "obj", y)):
            return x
        if cx in (Obj, Fn, Attr) and cy in (Obj, Fn, Attr):
            return SCALAR
        return Seq(None, self.fork_reason("length", name, "of different kinds", fork, dec, comp))

    # ── Loops ───────────────────────────────────────────────────────────────

    def comp_of(self, dec: Any, h: int, comp: Any) -> Any:
        """The comprehension a loop builds: ``("comp", decoded, index)``."""
        instrs = dec.instrs
        nxt = instrs[h + 1][0] if h + 1 < len(instrs) else None
        if nxt in (_bc.UNPACK_SEQUENCE, _bc.UNPACK_EX):
            j = h + 2
            while j < len(instrs) and instrs[j][0] == _bc.STORE_FAST_MAYBE:
                j += 1
            nxt = _bc.STORE_FAST_MAYBE if j > h + 2 else nxt
        if dec.code.co_name == "<genexpr>":
            return comp if comp is not None else self.gen_span
        if nxt != _bc.STORE_FAST_MAYBE:
            return None
        if comp is not None:
            return comp
        for j in range(h - 1, max(-1, h - 6), -1):
            op, arg = instrs[j]
            if op in (_bc.BUILD_LIST, _bc.BUILD_SET, _bc.BUILD_MAP) and arg == 0:
                return ("comp", dec, j)
        return None

    gen_span: Any = None

    def for_loop(
        self, dec: Any, st: _State, h: int, comp: Any, rets: list[Any]
    ) -> tuple[_State | None, list[tuple[int, _State]]]:
        exit_ = dec.instrs[h][1][0]
        it = st.stack[-_ITER]
        if it.__class__ is not It:
            it = self.iterate(it, dec, h)
        src, elem = it.src, it.elem
        line = _line(dec, h)

        def item() -> Any:
            truth = None
            if src is not None:
                s = self.sym(0, 1)
                self.iota[s] = src
                truth = Lin(0, {s: 1})
            one = Opq(None, truth) if truth is not None else SCALAR
            if elem == "enumerate":
                return Tup([SCALAR, one])
            if elem.__class__ in _INT:
                return Tup([SCALAR] * elem)
            return one

        inner = self.comp_of(dec, h, comp)
        return self.loop(
            dec, st, h, h + 1, exit_, it.n, it.why, item, inner, rets, "`for`", line, src
        )

    def while_loop(
        self, dec: Any, st: _State, head: int, back: int, comp: Any, rets: list[Any]
    ) -> tuple[_State | None, list[tuple[int, _State]]]:
        s = st
        for _ in range(_UNROLL):
            found: list[Any] = []
            backs, exits = self.sweep(dec, _fork(s), head, back + 1, head, comp, found)
            if not backs:
                rets.extend(found)
                return None, exits
            if exits:
                break
            rets.extend(found)
            s = backs[0] if len(backs) == 1 else self.merge(dec, backs, comp)
        line = _line(dec, back)
        return self.loop(
            dec, s, head, head, back + 1, None, None, None, comp, rets, "`while`", line, None
        )

    def loop(
        self,
        dec: Any,
        st: _State,
        h: int,
        lo: int,
        hi: int,
        trips: Any,
        trips_why: Any,
        item: Any,
        comp: Any,
        rets: list[Any],
        kind: str,
        line: int,
        src: Any,
    ) -> tuple[_State | None, list[tuple[int, _State]]]:
        """A loop's state after it, and the jumps out of it. ``trips`` is
        None for a while loop's (only its test decides)."""
        endless = item is None
        pops = 0 if endless else _ITER

        def after_none() -> _State:
            s = _fork(st)
            del s.stack[len(s.stack) - pops :]
            return s

        if trips == 0:
            return after_none(), []
        least = None if trips is None else self.bounds(trips)[0]
        may_skip = least is None or least < 1

        def how() -> str:
            if endless:
                return "which runs a number of times only the values decide"
            if trips is None:
                if trips_why is None:
                    return "whose number of passes isn't known"
                return "whose number of passes isn't known ({})"
            return "by a different amount each pass"

        def reason(k: Any, early: bool = False) -> Any:
            if comp is not None and k.__class__ is not str:
                return self.fork_reason("length", None, "?", (h, True, None), dec, comp)
            name = k if k.__class__ is str else "a list"
            if comp is not None:
                return _why(line, "`{}` changes inside the comprehension at line {}", name, line)
            words = (
                "which can exit early (break/continue/return)" if early and not endless else how()
            )
            parts: tuple[Any, ...] = (trips_why,) if "{}" in words else ()
            return _why(
                line,
                "`{}` changes inside the {} loop at line {}, " + words,
                name,
                kind,
                line,
                *parts,
            )

        # One pass from the entry: what changes.
        s = _fork(st)
        mark = len(self.lo)
        if item is not None:
            s.stack.append(item())
        first: list[Any] = []
        backs, exits = self.sweep(dec, s, lo, hi, h, comp, first)
        if not backs:
            # The body never comes back: at most one pass, from the entry.
            for _, out in exits:
                self.settle(out, {}, {}, mark, set())
            for value, *_ in first:
                self.settle(_State([value], {}, ()), {}, {}, mark, set())
            rets.extend(first)
            return (after_none() if may_skip and not endless else None), exits
        one = self.merge(dec, backs, comp)
        changed = self.changes(st, one)

        # Then with what changes unknown, until nothing new changes.
        for _ in range(8):
            base = _fork(st)
            syms, start = self.symbolize(base, changed, reason)
            snap = _fork(base)
            mark = len(self.lo)
            s = _fork(base)
            if item is not None:
                s.stack.append(item())
            found: list[Any] = []
            backs, exits = self.sweep(dec, s, lo, hi, h, comp, found)
            two = self.merge(dec, backs, comp) if backs else None
            if two is None:
                break
            new = [k for k in self.changes(snap, two) if k not in changed]
            if not new:
                break
            changed.extend(new)
        else:
            raise _GiveUpError

        loop_syms = set(syms.values())
        lost: dict[int, Any] = {s_: reason(k, True) for k, s_ in syms.items()}
        for _, out in exits:
            self.settle(out, {}, lost, mark, loop_syms)
        for value, *_ in found:
            self.settle(_State([value], {}, ()), {}, lost, mark, loop_syms)
        rets.extend(found)
        if two is None or endless:
            return None, exits

        # What each changed value does per pass.
        subst: dict[int, Any] = {}
        lost = {}
        finals: dict[Any, Any] = {}
        for k, s_ in syms.items():
            v = _slot(two, k)
            n = v.n if v.__class__ is Seq else v.v if v.__class__ is Const else None
            step = None if n is None else _add(n, Lin(0, {s_: 1}), -1)
            terms = () if step is None or step.__class__ in _INT else step.t
            if step is None or any(t in loop_syms for t in terms) or trips is None:
                lost[s_] = reason(k)
                continue
            if not any(t > mark for t in terms):
                times = _mul(_add(trips, -1), step)
                if times is not None:
                    subst[s_] = _add(start[s_], times)
                    continue
            elif trips.__class__ in _INT:
                finals[k] = _add(start[s_], self.passes(step, trips, mark, src))
            lost[s_] = reason(k)
        del two.stack[len(two.stack) - pops :]
        for k, n in finals.items():
            v = _slot(two, k)
            seen = _slot(one, k)
            why = getattr(seen, "why", None) or reason(k)
            if v.__class__ is Seq:
                v.n, v.why = n, (why if n.__class__ not in _INT else None)
            else:
                _set_slot(two, k, Const(n, why if n.__class__ not in _INT else None))
        for k, s_ in syms.items():
            if s_ in lost and k not in finals:
                seen = _slot(one, k)
                if seen.__class__ in (Seq, Const) and seen.why is not None:
                    lost[s_] = seen.why
        self.settle(two, subst, lost, mark, loop_syms)
        if may_skip:
            zero = after_none()
            zero.path = two.path
            return self.join_states(dec, two, zero, (h, "exit", None), comp), exits
        return two, exits

    def changes(self, a: _State, b: _State) -> list[Any]:
        if len(a.stack) != len(b.stack):
            raise _GiveUpError
        out: list[Any] = [
            k for k in a.env.keys() | b.env.keys() if _sig(a.env.get(k)) != _sig(b.env.get(k))
        ]
        out.sort()
        out += [
            j for j, (x, y) in enumerate(zip(a.stack, b.stack, strict=False)) if _sig(x) != _sig(y)
        ]
        return out

    def symbolize(
        self, st: _State, changed: list[Any], reason: Any
    ) -> tuple[dict[Any, int], dict[int, Any]]:
        syms: dict[Any, int] = {}
        start: dict[int, Any] = {}
        seen: dict[int, int] = {}
        for k in changed:
            v = _slot(st, k)
            if v is None:
                continue
            c = v.__class__
            if c is Seq and v.n is not None:
                s = seen.get(id(v))
                if s is None:
                    s = seen[id(v)] = self.sym(None, None)
                    start[s] = v.n
                    v.n, v.src = Lin(0, {s: 1}), None
                    v.why = reason(k)
                    self.placeholders.add(id(v.why))
                syms[k] = s
            elif c is Const:
                s = self.sym(None, None)
                start[s] = v.v
                why = reason(k)
                self.placeholders.add(id(why))
                _set_slot(st, k, Const(Lin(0, {s: 1}), why))
                syms[k] = s
            elif c in (Tup, It, Obj, Map, Fn):
                _set_slot(st, k, Opq(reason(k)))
        return syms, start

    def passes(self, step: Any, trips: int, mark: int, src: Any) -> Any:
        """The sum of *step* over *trips* passes, each with its own
        pass-local symbols."""
        total: Any = step.k * trips
        for s, c in step.t.items():
            if s <= mark:
                total = _add(total, Lin(0, {s: c * trips}))
                continue
            if s in self.iota and self.iota[s] == src:
                each: Any = self.count_of(src)
            else:
                lo, hi = self.lo[s], self.hi[s]
                each = Lin(
                    0,
                    {
                        self.sym(
                            None if lo is None else lo * trips, None if hi is None else hi * trips
                        ): 1
                    },
                )
            total = _add(total, each, c)
        return total

    def settle(
        self,
        st: _State,
        subst: dict[int, Any],
        lost: dict[int, Any],
        mark: int,
        loop_syms: set[int],
    ) -> None:
        """Rewrite a state leaving a loop: per-pass symbols become what they
        add up to (*subst*), or unknowns with the reason (*lost*)."""
        fresh: dict[int, int] = {}
        memo: dict[int, Any] = {}

        def num(n: Any) -> tuple[Any, Any]:
            if n.__class__ is not Lin:
                return n, None
            why = None
            out: Any = n.k
            for s, c in n.t.items():
                if s in subst:
                    out = _add(out, _mul(c, subst[s]))
                    continue
                if s in loop_syms or s > mark:
                    if s in lost and why is None:
                        why = lost[s]
                    if s not in fresh:
                        fresh[s] = self.sym(self.lo[s], self.hi[s])
                    s = fresh[s]
                out = _add(out, Lin(0, {s: c}))
            return out, why

        def fix(v: Any) -> Any:
            c = v.__class__
            if c is Seq:
                if id(v) in memo:
                    return v
                memo[id(v)] = v
                if v.n is not None:
                    v.n, why = num(v.n)
                    if v.n.__class__ in _INT:
                        v.why = None
                    elif why is not None and (v.why is None or id(v.why) in self.placeholders):
                        v.why = why
                return v
            if c is Const:
                n, why = num(v.v)
                if n == v.v:
                    return v
                if n.__class__ in _INT:
                    return Const(n)
                return Const(n, why if v.why is None or id(v.why) in self.placeholders else v.why)
            if c is Tup:
                v.items = [fix(x) for x in v.items]
                return v
            if c is Opq and v.truth is not None:
                return SCALAR
            return v

        st.stack[:] = [fix(v) for v in st.stack]
        for k in list(st.env):
            st.env[k] = fix(st.env[k])

    # ── Instructions ────────────────────────────────────────────────────────

    def const(self, value: Any, dec: Any, i: int) -> Any:
        c = value.__class__
        if c is int:
            return Const(value)
        if c is bool:
            return Const(int(value))
        if c is str or c is bytes:
            return Seq(len(value), elem="str")
        if value is None:
            return NONE
        if c is tuple:
            return Tup([self.const(x, dec, i) for x in value])
        if c is slice:
            return Slc([self.const(x, dec, i) for x in (value.start, value.stop, value.step)])
        if c is type(self.const.__code__):
            return Fn(value, False, dec, i)
        if c is type:
            return Obj(value, value.__name__, _line(dec, i))
        return SCALAR

    def read(self, kind: int, where: Any, key: str) -> Any:
        """What the function reads from outside it, kept so a later version
        reads the same before it reuses this run."""
        value = _read(kind, where, key)
        reads = self.trace.reads
        k = (kind, id(where), key)
        if k not in reads:
            reads[k] = (kind, where, key, value, _read_sig(value))
        return value

    def load_global(self, name: str, dec: Any, i: int) -> Any:
        line = _line(dec, i)
        if name in self.forced:
            return Opq(_why(line, "`{}` is a module-level variable the function changes", name))
        value = self.read(_GLOBAL, self.g, name)
        if value is not _MISSING:
            return self.from_object(value, name, line)
        value = self.read(_ATTR, builtins, name)
        if value is not _MISSING:
            return self.from_object(value, name, line)
        return SCALAR

    def attr(self, obj: Any, name: str, dec: Any, i: int) -> Any:
        if obj.__class__ is Obj and obj is not NONE:
            value = self.read(_ATTR, obj.obj, name)
            if value is not _MISSING:
                return self.from_object(value, f"{obj.name}.{name}", _line(dec, i))
        return Attr(obj, name, dec, i)

    def iterate(self, v: Any, dec: Any, i: int) -> It:
        c = v.__class__
        if c is It:
            it: It = v
            return it
        if c is Seq:
            return It(v.n, v.why, v.src, v.elem if v.elem != "str" else None)
        if c is Tup:
            return It(len(v.items), None)
        if c is Map:
            return It(v.count, None)
        try:
            n, why = self.length(v, ("x", dec, i))
            return It(n, why)
        except _NoLengthError as c_:
            return It(None, c_.why)

    def unary(self, f: Any, v: Any) -> Any:
        if f is tuple:
            if v.__class__ is Seq:
                return Seq(v.n, v.why, src=v.src)
            return v if v.__class__ is Tup else SCALAR
        if v.__class__ is Const:
            if f is _op.neg:
                return Const(_mul(-1, v.v), v.why)
            if f is _op.pos:
                return v
            if f is _op.invert:
                return Const(_add(-1, v.v, -1), v.why)
        return SCALAR

    def is_none(self, v: Any) -> bool | None:
        if v is NONE:
            return True
        if v.__class__ in (Seq, Const, Tup, It, Fn, Map):
            return False
        if v.__class__ is Obj:
            return v.obj is None
        return None

    def bit(self, v: Any) -> Any:
        """An item of a Bits input is its truth, 0 or 1."""
        if v.__class__ is Opq and v.truth is not None:
            (s,) = v.truth.t
            if self.iota.get(s) in self.bits:
                return Const(v.truth)
        return v

    def compare(self, f: Any, a: Any, b: Any) -> Any:
        a, b = self.bit(a), self.bit(b)
        if a.__class__ is Const and b.__class__ is Const and f in _CMP:
            x, y = a.v, b.v
            if x.__class__ in _INT and y.__class__ in _INT:
                return Const(int(f(x, y)))
            lo, hi = self.bounds(_add(x, y, -1))
            if lo is not None and hi is not None and lo == hi:
                return Const(int(f(lo, 0)))
            if f is _op.lt or f is _op.ge:
                t = (
                    True
                    if hi is not None and hi < 0
                    else False
                    if lo is not None and lo >= 0
                    else None
                )
                if t is not None:
                    return Const(int(t if f is _op.lt else not t))
            if f is _op.gt or f is _op.le:
                t = (
                    True
                    if lo is not None and lo > 0
                    else False
                    if hi is not None and hi <= 0
                    else None
                )
                if t is not None:
                    return Const(int(t if f is _op.gt else not t))
            if f is _op.eq or f is _op.ne:
                if (lo is not None and lo > 0) or (hi is not None and hi < 0):
                    return Const(int(f is _op.ne))
        return SCALAR

    def binary(self, st: _State, f: Any, a: Any, b: Any, dec: Any, i: int) -> Any:
        ca, cb = a.__class__, b.__class__
        op = _ARITH.get(f)
        if ca is Const and cb is Const:
            x, y = a.v, b.v
            if x.__class__ in _INT and y.__class__ in _INT:
                if op is None:
                    return SCALAR
                try:
                    r = op(x, y)
                    return Const(r if r.__class__ is _Num else int(r))
                except (ArithmeticError, ValueError, TypeError):
                    return SCALAR
            why = a.why or b.why
            if op is _ADD:
                return Const(_add(x, y), why)
            if op is _SUB:
                return Const(_add(x, y, -1), why)
            if op is _MULT:
                n = _mul(x, y)
                if n is not None:
                    return Const(n, why)
            return Opq(why)
        inplace = f in _INPLACE
        if op is _ADD and (ca in _SEQUENCES or cb in _SEQUENCES):
            if ca is Tup and cb is Tup:
                return Tup(a.items + b.items)
            try:
                n, why = self.length(a, ("left", dec, i))
                m, why2 = self.length(b, ("right", dec, i))
                n = _add(n, m)
                why = why or why2
            except _NoLengthError as c:
                n, why = None, c.why
            if inplace and ca is Seq:
                if self.mutate(a, dec, i):
                    if n is None:
                        a.lose(why)
                    else:
                        a.n, a.why = n, (why if n.__class__ not in _INT else None)
                return a
            return Seq(n, why if n.__class__ not in _INT else None)
        if op is _MULT and (ca in _SEQUENCES or cb in _SEQUENCES):
            seq, k, side = (a, b, "right") if ca in _SEQUENCES else (b, a, "left")
            try:
                n, why = self.length(seq, ("left" if side == "right" else "right", dec, i))
                times, why2 = self.number(k, (side, dec, i))
                times = self.max0(times)
                n = None if times is None else _mul(n, times)
                why = why or why2
                if n is None:
                    why = (
                        why2
                        or why
                        or _why(_line(dec, i), "`{}` has no fixed length", ("x", dec, i))
                    )
            except _NoLengthError as c:
                n, why = None, c.why
            if inplace and ca is Seq:
                if self.mutate(a, dec, i):
                    if n is None:
                        a.lose(why)
                    else:
                        a.n, a.why = n, (why if n.__class__ not in _INT else None)
                return a
            return Seq(n, why if n.__class__ not in _INT else None)
        if inplace and ca not in (Const, Seq, Tup):
            self.lose_escaped(
                st, _why(_line(dec, i), "`{}` could change it in place", ("x", dec, i))
            )
        return SCALAR

    def subscr(self, base: Any, ix: Any, dec: Any, i: int) -> Any:
        cb = base.__class__
        if ix.__class__ is Slc:
            parts = (ix.parts + [NONE, NONE])[:3]
            if cb is Tup and all(
                p is NONE or (p.__class__ is Const and p.v.__class__ in _INT) for p in parts
            ):
                return Tup(base.items[slice(*[None if p is NONE else p.v for p in parts])])
            try:
                n, why = self.length(base, ("value", dec, i))
                bounds: list[Any] = []
                for k, p in enumerate(parts):
                    if p is NONE:
                        bounds.append(None)
                    else:
                        v, w = self.number(p, ("slice", dec, i, k))
                        bounds.append(v)
                        why = why or w
                return self.slice_length(n, bounds, why, dec, i, base)
            except _NoLengthError as c:
                return Seq(None, c.why)
        if cb is Tup and ix.__class__ is Const and ix.v.__class__ in _INT:
            if -len(base.items) <= ix.v < len(base.items):
                return base.items[ix.v]
        if cb is Map:
            return Seq(base.n, frozen=base.name)
        return SCALAR

    def slice_length(self, n: Any, bounds: list[Any], why: Any, dec: Any, i: int, base: Any) -> Any:
        start, stop, step = bounds
        if all(x is None or x.__class__ in _INT for x in (n, start, stop, step)):
            try:
                out = _derive(_slice_len, n, start, stop, step)
            except ValueError:
                return Seq(None, _why(_line(dec, i), "`{}` has a zero step", ("x", dec, i)))
            return Seq(out, src=None, elem=base.elem if base.__class__ is Seq else None)
        if step not in (None, 1):
            return Seq(None, why or _why(_line(dec, i), "`{}` has no fixed length", ("x", dec, i)))
        a = 0 if start is None else start
        b = n if stop is None else stop
        ok = True
        for x in (a, b):
            lo, hi = self.bounds(x)
            if lo is None or lo < 0:
                ok = False
            hi2 = self.bounds(_add(n, x, -1))[0]
            if hi2 is None or hi2 < 0:
                ok = False
        width = self.max0(_add(b, a, -1)) if ok else None
        if width is None:
            return Seq(None, why or _why(_line(dec, i), "`{}` has no fixed length", ("x", dec, i)))
        return Seq(width, why if width.__class__ not in _INT else None)

    def store_subscr(self, st: _State, value: Any, obj: Any, ix: Any, dec: Any, i: int) -> None:
        c = obj.__class__
        if c is Seq:
            if ix.__class__ is Slc:
                if self.mutate(obj, dec, i):
                    obj.lose(
                        _why(
                            _line(dec, i),
                            "`{}` is assigned a slice, which can change its length",
                            ("x", dec, i),
                        )
                    )
            else:
                obj.src = None
            self.escape(value)
            return
        if c is Map:
            self.changed.add(obj.name)
            return
        self.escape(value)
        self.lose_escaped(st, _why(_line(dec, i), "`{}` could change it", ("x", dec, i)))

    def capture_values(self, v: Any, dec: Any, i: int) -> None:
        line = _line(dec, i)
        self.lose_all(
            v, _why(line, "it is used by a nested function at line {}, which could change it", line)
        )

    # ── Calls ───────────────────────────────────────────────────────────────

    def call(self, st: _State, f: Any, args: list[Any], names: int, dec: Any, i: int) -> Any:
        c = f.__class__
        recursive = False
        if c is Obj:
            o = f.obj
            try:
                kind = _KNOWN.get(o)
            except TypeError:
                kind = None
            if kind is None and o.__class__ is types.FunctionType and names == 0:
                got = self.helper(st, o, f.name, args, dec, i)
                if got is _ITSELF:
                    recursive = True
                elif got is not None:
                    return got
            if kind is None and o.__class__ is types.GenericAlias:
                kind = _KNOWN.get(o.__origin__)  # list[float]()
            if kind is not None and names >= 0:
                return self.builtin(kind, o, args[: len(args) - names], dec, i)
            if o is math:
                return SCALAR
            cmath = self.read(_GLOBAL, sys.modules, "cmath")
            if o is (None if cmath is _MISSING else cmath) or (
                callable(o) and getattr(o, "__module__", None) in ("math", "cmath")
            ):
                return SCALAR
        elif c is Attr:
            recv = f.recv
            if recv.__class__ is Seq:
                return self.method(st, recv, f.name, args, names, dec, i)
            if recv.__class__ is Tup and f.name in ("index", "count"):
                return SCALAR
            if recv.__class__ is Map and f.name in ("get", "keys", "values", "items", "copy"):
                return SCALAR
            if recv.__class__ is Map:
                self.changed.add(recv.name)
        elif c is Fn and f.gen and len(args) == 1:
            return self.genexpr(st, f, args[0])
        # A call the analysis can't see into: any list it can reach may change.
        line = _line(dec, i)
        for k, a in enumerate(args):
            self.lose_all(
                a,
                _why(
                    line,
                    "`{}` is passed to `{}()` at line {}, which could change its length",
                    ("arg", dec, i, k) if names >= 0 else ("x", dec, i),
                    ("fn", dec, i),
                    line,
                ),
            )
        self.lose_escaped(st, _why(line, "`{}()` at line {} could change it", ("fn", dec, i), line))
        if recursive:
            return Opq(
                _why(line, "`{}` calls itself, and the compiler doesn't follow that", ("x", dec, i))
            )
        return Opq(
            _why(line, "`{}` returns something whose length the compiler can't see", ("x", dec, i))
        )

    def helper(
        self, st: _State, fn: types.FunctionType, name: str, args: list[Any], dec: Any, i: int
    ) -> Any:
        """A call of a plain Python function: its body run on *args*, or None
        when the analysis doesn't follow it (it calls itself, takes keyword
        or starred arguments, or does something the analysis can't follow).
        The caller's values ride at the bottom of the body's stack, so a list
        it changes on any of its paths comes back joined."""
        code = self.read(_ATTR, fn, "__code__")
        if code in self.active:
            return _ITSELF
        if (
            len(self.active) > _HELPER_DEPTH
            or code.co_flags & (_CO_ASYNC | _CO_GENERATOR | _CO_STARRED)
            or code.co_kwonlyargcount
            or len(args) > code.co_argcount
        ):
            return None
        defaults = self.read(_ATTR, fn, "__defaults__") or ()
        missing = code.co_argcount - len(args)
        if missing > len(defaults):
            return None
        try:
            body = _bc.decode(code)
        except PulseError:
            return None
        line = _line(dec, i)
        env = dict(zip(code.co_varnames, args, strict=False))
        for k, value in enumerate(defaults[len(defaults) - missing :] if missing else ()):
            env[code.co_varnames[len(args) + k]] = self.from_object(value, name, line)
        carried = [*st.stack, *st.env.values(), *args]
        known = {id(why) for why in _whys(carried)}
        before = (self.g, self.cells)
        self.g = self.read(_ATTR, fn, "__globals__")
        self.cells = dict(zip(code.co_freevars, fn.__closure__ or (), strict=False))
        self.active.append(code)
        rets: list[Any] = []
        try:
            self.sweep(
                body, _State(list(carried), env, st.path), 0, len(body.instrs), -1, None, rets
            )
        except (_GiveUpError, RecursionError):
            return None
        finally:
            self.active.pop()
            self.g, self.cells = before
        ends = [
            _State([value, *s.stack], {}, s.path)
            for value, _, _, s in rets
            if len(s.stack) == len(carried)
        ]
        if not ends or len(ends) != len(rets):
            return None
        try:
            out = self.merge(body, ends, None, len(st.path)).stack
        except _GiveUpError:
            return None
        boxes, joined = _boxes(Tup(carried)), _boxes(Tup(out[1:]))
        if len(boxes) != len(joined):
            return None

        def within(why: Any) -> Any:
            if why is None or id(why) in known:
                return why
            return _why(line, "in `{}`, {}", ("x", dec, i), why)

        for box, now in zip(boxes, joined, strict=True):
            if now is not box:
                box.n, box.why, box.src, box.esc = now.n, within(now.why), now.src, now.esc
        result = out[0]
        for box, now in zip(boxes, joined, strict=True):
            if result is now:
                return box
        return _rewhy(result, within)

    def builtin(self, kind: int, o: Any, args: list[Any], dec: Any, i: int) -> Any:
        line = _line(dec, i)
        if kind == _LEN and len(args) == 1:
            try:
                n, why = self.length(args[0], ("arg", dec, i, 0))
                return Const(n, why)
            except _NoLengthError as c:
                return Opq(c.why)
        if kind == _RANGE:
            try:
                bounds = [self.number(a, ("arg", dec, i, k)) for k, a in enumerate(args)]
            except _NoLengthError as c:
                return It(None, c.why)
            why = next((w for _, w in bounds if w is not None), None)
            nums = [b for b, _ in bounds]
            if all(b.__class__ in _INT for b in nums):
                try:
                    return It(_derive(_range_len, *nums), None)
                except (TypeError, ValueError):
                    pass
            elif len(nums) in (1, 2) or (len(nums) == 3 and nums[2] == 1):
                width = self.max0(nums[0] if len(nums) == 1 else _add(nums[1], nums[0], -1))
                if width is not None:
                    return It(width, why)
            return It(None, why or _why(line, "`{}` is not a range of known length", ("x", dec, i)))
        if kind == _SAME or kind == _ENUM:
            if not args:
                return Seq(0)
            try:
                n, why = self.length(args[0], ("arg", dec, i, 0))
            except _NoLengthError as c:
                return Seq(None, c.why)
            a = args[0]
            src = a.src if a.__class__ in (Seq, It) else None
            return Seq(n, why, src=src, elem="enumerate" if kind == _ENUM else None)
        if kind == _ZIP or kind == _MAP:
            seqs = list(enumerate(args))[1:] if kind == _MAP else list(enumerate(args))
            try:
                sizes = [self.length(a, ("arg", dec, i, k)) for k, a in seqs]
            except _NoLengthError as c:
                return Seq(None, c.why)
            if not sizes:
                return Seq(0)
            n, why = sizes[0]
            for m, w in sizes[1:]:
                d = _add(n, m, -1)
                if d.__class__ in _INT:
                    n = m if d > 0 else n
                else:
                    lo, hi = self.bounds(d)
                    if lo is not None and lo >= 0:
                        n = m
                    elif not (hi is not None and hi <= 0):
                        return Seq(
                            None, why or w or _why(line, "`{}` has no fixed length", ("x", dec, i))
                        )
                why = why or w
            return Seq(
                n,
                why if n.__class__ not in _INT else None,
                elem=len(args) if kind == _ZIP else None,
            )
        if kind == _FILTER:
            return Seq(
                None,
                _why(
                    line,
                    "`{}` keeps only some items, so its length depends on the values",
                    ("x", dec, i),
                ),
            )
        if kind == _FOLD and args and all(a.__class__ is Const for a in args):
            vals = [a.v for a in args]
            why = next((a.why for a in args if a.why is not None), None)
            if all(v.__class__ in _INT for v in vals):
                if o is int and len(vals) == 1:
                    return Const(vals[0])
                try:
                    return Const(o(*vals))
                except (TypeError, ValueError):
                    return SCALAR
            if o is int:
                return Const(vals[0], why) if len(vals) == 1 else SCALAR
            if o is abs and len(vals) == 1:
                lo, hi = self.bounds(vals[0])
                if lo is not None and lo >= 0:
                    return Const(vals[0], why)
                if hi is not None and hi <= 0:
                    return Const(_mul(-1, vals[0]), why)
                return Opq(why)
            if o in (min, max) and len(vals) >= 2:
                best = vals[0]
                for v in vals[1:]:
                    d = _add(v, best, -1)
                    lo, hi = self.bounds(d)
                    bigger = (
                        True
                        if lo is not None and lo >= 0
                        else False
                        if hi is not None and hi <= 0
                        else None
                    )
                    if bigger is None:
                        return Opq(why)
                    if bigger == (o is max):
                        best = v
                return Const(best, why)
            return Opq(why)
        if kind == _SUM and len(args) == 1:
            a = args[0]
            if a.__class__ is Seq and a.src in self.bits and a.n is not None:
                return Const(
                    self.count_of(a.src),
                    _why(line, "`{}` depends on values, not only on input lengths", ("x", dec, i)),
                )
        return SCALAR

    def method(
        self, st: _State, box: Seq, name: str, args: list[Any], names: int, dec: Any, i: int
    ) -> Any:
        if box.elem == "str":
            return SCALAR
        pos = args[: len(args) - names] if names >= 0 else args
        if name == "copy":
            return Seq(box.n, box.why, src=box.src)
        if name in ("index", "count", "sort", "reverse"):
            return SCALAR
        if name in _MUTATORS or name in ("extend", "clear"):
            for a in pos:
                self.escape(a)
            if self.mutate(box, dec, i):
                if name == "extend":
                    try:
                        n, why = self.length(pos[0], ("arg", dec, i, 0))
                        box.n = _add(box.n, n)
                        box.why = box.why or why
                    except (_NoLengthError, IndexError) as c:
                        box.lose(
                            getattr(c, "why", None) or _why(_line(dec, i), "extend() of nothing")
                        )
                elif name == "clear":
                    box.n, box.why = 0, None
                elif box.n.__class__ in _INT:
                    box.n = max(0, box.n + _MUTATORS[name])
                else:
                    box.n = _add(box.n, _MUTATORS[name])
            return SCALAR
        line = _line(dec, i)
        why = _why(line, "`{}` is not followed", ("x", dec, i))
        if self.mutate(box, dec, i):
            box.lose(why)
        for a in args:
            self.lose_all(a, why)
        self.lose_escaped(st, why)
        return Opq(why)

    def genexpr(self, st: _State, fn: Fn, iterable: Any) -> Any:
        """Run a generator expression's code: as many items as it yields."""
        dec = _bc.decode(fn.code)
        env = {name: st.env[name] for name in fn.code.co_freevars if name in st.env}
        env[".0"] = iterable
        env[".yielded"] = Seq(0)
        saved = self.gen_span
        self.gen_span = ("comp", fn.dec, fn.i)
        found: list[Any] = []
        try:
            self.sweep(dec, _State([], env, ()), 0, len(dec.instrs), -1, None, found)
        finally:
            self.gen_span = saved
        sizes = [s.env.get(".yielded") for *_, s in found if s is not None]
        if not sizes:
            return Seq(0)
        box = sizes[0]
        for other in sizes[1:]:
            if other.n != box.n:
                return Seq(None, box.why or other.why)
        return Seq(box.n, box.why)


_MISSING: Any = object()

# ── The part loaded when first needed ───────────────────────────────────────


def _rare() -> Any:
    """``_infer_rare``: error wording and the instructions few bodies use,
    loaded when first needed."""
    from . import _infer_rare

    return _infer_rare


# ── Reads from outside the function ─────────────────────────────────────────

_GLOBAL, _ATTR, _CELL = range(3)


def _read(kind: int, where: Any, key: str) -> Any:
    if kind == _GLOBAL:
        return where.get(key, _MISSING)
    if kind == _ATTR:
        return getattr(where, key, _MISSING)
    try:
        return where.cell_contents
    except ValueError:
        return _MISSING


def _read_sig(value: Any) -> Any:
    """What ``from_object`` reads of a value besides its identity."""
    c = value.__class__
    if c is list or c is tuple:
        return len(value)
    if c is dict:
        return len(value), frozenset(
            len(v) if v.__class__ in (list, tuple) else -1 for v in value.values()
        )
    return None


# ── Entry point ─────────────────────────────────────────────────────────────


class _Rule:
    """One run's results as nodes of the input lengths, and what decides
    that another version runs the same: the outcome of each test the run
    made of its lengths, and the values it read from outside the body."""

    __slots__ = ("code", "guards", "reads", "results")

    def __init__(self, code: Any, trace: _Trace, results: list[Any]) -> None:
        self.code, self.results = code, results
        self.guards = list(trace.guards.values())
        self.reads = list(trace.reads.values())

    def apply(self, fn: Any, lengths: dict[str, int]) -> tuple[int, ...] | None:
        """The results for *lengths*, or None when this run doesn't hold."""
        if getattr(fn, "__code__", None) is not self.code:
            return None
        memo: dict[int, Any] = {}
        for node, outcome in self.guards:
            r = _eval(node, lengths, memo)
            if r is not outcome and (r is _RAISES or outcome is _RAISES or r != outcome):
                return None
        for kind, where, key, value, sig in self.reads:
            now = _read(kind, where, key)
            if now is not value and not (
                now.__class__ is value.__class__ and now.__class__ in (int, bool) and now == value
            ):
                return None
            if sig is not None and _read_sig(now) != sig:
                return None
        out = tuple(_eval(e, lengths, memo) for e in self.results)
        return None if _RAISES in out else out


_RULES_KEPT = 8  # runs a function keeps; a version none holds for runs again


class LengthRules:
    """A function's list-result lengths for each version. The first
    version's run gives them as formulas of the input lengths, held for
    every version whose lengths pass the same tests of them (most do: a
    test like ``len(x) > 0`` holds for every length but 0); such a version
    takes the formulas evaluated, a few integer operations. A version no
    kept run holds for runs the body itself, and its run is kept too.

    *list_results* are the list results' positions with their type names,
    *result_count* how many results there are, and *bits* the inputs whose
    items are bits, so their sum counts the set ones."""

    __slots__ = ("fn", "names", "list_results", "result_count", "bits", "rules", "runs")

    def __init__(
        self,
        fn: Any,
        names: tuple[str, ...],
        list_results: list[tuple[int, str]],
        result_count: int,
        bits: frozenset[str] = frozenset(),
    ) -> None:
        self.fn, self.names, self.list_results = fn, names, list_results
        self.result_count, self.bits = result_count, bits
        self.rules: list[_Rule] = []
        self.runs = 0  # how many versions ran the body

    def lengths(self, where: str, input_lengths: dict[str, int]) -> tuple[int, ...]:
        """One version's list-result lengths; *where* names it in errors."""
        for rule in self.rules:
            out = rule.apply(self.fn, input_lengths)
            if out is not None:
                return out
        self.runs += 1
        out, rule = _run(
            self.fn,
            where,
            self.names,
            input_lengths,
            self.list_results,
            self.result_count,
            self.bits,
        )
        if len(self.rules) < _RULES_KEPT:
            self.rules.append(rule)
        return out


def infer_lengths(
    fn: Any,
    where: str,
    names: tuple[str, ...],
    input_lengths: dict[str, int],
    list_results: list[tuple[int, str]],
    result_count: int,
    bits: frozenset[str] = frozenset(),
) -> tuple[int, ...]:
    """The lengths of one version's list results, from a run of the body
    (see ``LengthRules`` for the arguments)."""
    return _run(fn, where, names, input_lengths, list_results, result_count, bits)[0]


def _run(
    fn: Any,
    where: str,
    names: tuple[str, ...],
    input_lengths: dict[str, int],
    list_results: list[tuple[int, str]],
    result_count: int,
    bits: frozenset[str],
) -> tuple[tuple[int, ...], _Rule]:
    position, type_name = list_results[0]
    try:
        code = getattr(fn, "__code__", None)
        if code is None:
            raise _NoLengthError(_why(0, "its body is not Python code the compiler can read"))
        if code.co_flags & _CO_ASYNC:
            raise _NoLengthError(_why(code.co_firstlineno, "it is an async function"))
        if code.co_flags & _CO_GENERATOR:
            raise _NoLengthError(_why(code.co_firstlineno, "it is a generator function"))
        try:
            dec = _bc.decode(code)
        except PulseError as error:
            raise _NoLengthError(
                _why(0, str(error).replace("{", "{{").replace("}", "}}"))
            ) from None
        trace = _Trace()
        nums = {name: _num(n, (None, name), trace) for name, n in input_lengths.items()}
        forced: set[str] = set()
        while True:
            run = _Run(fn, code, nums, bits, forced, trace)
            for name in nums:
                run.count_of(name)  # before any loop, so no pass takes it for its own
            env: dict[str, Any] = {}
            for name in names:
                if name in nums:
                    env[name] = Seq(nums[name], src=name)
                else:
                    env[name] = Opq(
                        _why(
                            0, "`{}` is a number parameter, known only when the function runs", name
                        )
                    )
            rets: list[Any] = []
            try:
                run.sweep(dec, _State([], env, ()), 0, len(dec.instrs), -1, None, rets)
            except (_GiveUpError, RecursionError):
                raise _NoLengthError(
                    _why(code.co_firstlineno, "the compiler can't follow its control flow")
                ) from None
            if run.changed <= forced:
                break
            forced |= run.changed
        got = _results(run, rets, list_results, result_count)
        made = _Rule(code, trace, [n.e if n.__class__ is _Num else n for n in got])
        return tuple(n.v if n.__class__ is _Num else n for n in got), made
    except _NoLengthError as c:
        line = c.why[0] if c.why else 0
        which = (
            f"its {type_name} result" if result_count == 1 else f"result {position} ({type_name})"
        )
        at = f" (line {line})" if line else ""
        raise InferenceError(
            f"{where}: can't tell the length of {which} from the body{at}: {_rare().say(c.why)}."
        ) from None


def _results(
    run: _Run, rets: list[Any], list_results: list[tuple[int, str]], result_count: int
) -> tuple[int, ...]:
    if not rets:
        raise _NoLengthError(_why(0, "it always raises an exception"))
    rets = sorted(rets, key=lambda r: r[2])
    out = []
    for position, _type_name in list_results:
        found: list[tuple[int, int]] = []
        for value, dec, i, _ in rets:
            line = _line(dec, i)
            if value is NONE:
                raise _NoLengthError(_why(line, "{}", ("noreturn", dec, i)))
            item = value
            if result_count > 1:
                if value.__class__ is not Tup or len(value.items) != result_count:
                    raise _NoLengthError(
                        _why(
                            line,
                            "`return {}` isn't a tuple of {} items",
                            ("ret", dec, i),
                            result_count,
                        )
                    )
                item = value.items[position]
            try:
                n, why = run.length(item, ("ret", dec, i))
            except _NoLengthError as c:
                if c.why[0] == 0:
                    raise _NoLengthError((line, c.why[1], c.why[2])) from None
                raise
            if n.__class__ not in _INT:
                raise _NoLengthError(why or _why(line, "`{}` has no fixed length", ("ret", dec, i)))
            if (n, line) not in found:
                found.append((n, line))
        sizes = {n for n, _ in found}
        if len(sizes) > 1:
            detail = ", ".join(f"{n} at line {line}" for n, line in found)
            raise _NoLengthError(
                _why(found[0][1], "its returns give different lengths ({})", detail)
            )
        out.append(found[0][0])
    return tuple(out)
