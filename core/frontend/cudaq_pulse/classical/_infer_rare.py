# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The parts of ``_infer`` few versions need, imported on first use: the
instructions few classical functions contain (``try``, ``with``,
``match``, nested functions, ``del``, ``import``, global stores) and the
wording of an error. The functions taking *self* are ``_Run`` methods.
"""

from __future__ import annotations

import sys
from typing import Any

from ..kernel import _bytecode_normalize as _bc
from ._infer import (
    _EXIT,
    _GLOBAL,
    _INT,
    _MISSING,
    _SELF,
    SCALAR,
    Const,
    Fn,
    Map,
    Obj,
    Opq,
    Seq,
    Slc,
    Tup,
    _add,
    _GiveUpError,
    _line,
    _NoLengthError,
    _Run,
    _State,
    _why,
)

# ── Instructions ────────────────────────────────────────────────────────────


def raw(self: _Run, st: _State, name: str, arg: int, dec: Any, i: int) -> _State | None:
    """An instruction the kernel tracer doesn't run, by its raw opname."""
    stack = st.stack
    line = _line(dec, i)
    if name == "MAKE_FUNCTION":
        code = stack.pop()
        fn = Fn(getattr(code, "code", None), False, dec, i)
        _make_function(self, st, fn, dec, i)
        if _bc._PY_MINOR == 12:
            for _ in range(bin(arg & 0x0F).count("1")):
                self.capture_values(stack.pop(), dec, i)
        stack.append(fn)
    elif name == "BEFORE_WITH":
        stack.pop()
        self.lose_escaped(st, _why(line, "the `with` at line {} could change it", line))
        stack.extend((_EXIT, SCALAR))
    elif name == "LOAD_SPECIAL":
        stack.pop()
        stack.extend((SCALAR, _SELF))
    elif name == "PUSH_EXC_INFO":
        exc = stack.pop()
        stack.extend((SCALAR, exc))
    elif name == "POP_EXCEPT":
        stack.pop()
    elif name == "CHECK_EXC_MATCH":
        stack[-1] = SCALAR
    elif name == "CHECK_EG_MATCH":
        stack[-2:] = [SCALAR, SCALAR]
    elif name == "WITH_EXCEPT_START":
        self.lose_escaped(st, _why(line, "the `with` at line {} could change it", line))
        stack.append(SCALAR)
    elif name == "RERAISE":
        return None  # its handler, if any, was given the state before it
    elif name == "GET_LEN":
        try:
            n, why = self.length(stack[-1], ("x", dec, i))
            stack.append(Const(n, why))
        except _NoLengthError as c:
            stack.append(Opq(c.why))
    elif name in ("MATCH_SEQUENCE", "MATCH_MAPPING", "MATCH_KEYS"):
        stack.append(SCALAR)
    elif name == "MATCH_CLASS":
        del stack[-3:]
        stack.append(SCALAR)
    elif name in ("STORE_GLOBAL", "DELETE_GLOBAL"):
        gname = dec.code.co_names[arg]
        if name == "STORE_GLOBAL":
            self.lose_all(stack.pop(), _why(line, "stored in `{}` at line {}", gname, line))
        self.changed.add(gname)
    elif name == "STORE_ATTR":
        stack.pop()
        self.lose_all(stack.pop(), _why(line, "stored in `{}` at line {}", ("x", dec, i), line))
        self.lose_escaped(st, _why(line, "`{}` could change it", ("x", dec, i)))
    elif name == "DELETE_ATTR":
        stack.pop()
    elif name == "DELETE_DEREF":
        st.env.pop(dec.code._varname_from_oparg(arg), None)
    else:
        import dis

        code = dis.opmap.get(name)
        if code is None:
            raise _GiveUpError
        try:
            effect = dis.stack_effect(code, arg if code >= dis.HAVE_ARGUMENT else None)
        except (TypeError, ValueError):
            raise _GiveUpError from None
        why = _why(line, "`{}` is not followed", ("x", dec, i))
        k = max(0, -effect) if effect < 0 else 0
        for v in stack[len(stack) - k :]:
            self.lose_all(v, why)
        del stack[len(stack) - k :]
        stack.extend([SCALAR] * max(0, effect))
        self.lose_escaped(st, why)
    return st


def _make_function(self: _Run, st: _State, fn: Any, dec: Any, i: int) -> None:
    """A nested function or lambda can change the lists it closes over,
    and rebind the numbers it declares nonlocal."""
    line = _line(dec, i)
    code = fn.code
    if code is None:
        return
    stores = _cell_stores(code)
    for name in code.co_freevars:
        v = st.env.get(name)
        if v is None:
            continue
        why = _why(
            line,
            "`{}` is used by a nested function at line {}, which could change it",
            name,
            line,
        )
        self.lose_all(v, why)
        if name in stores and v.__class__ is not Seq:
            st.env[name] = Opq(why)


def _cell_stores(code: Any) -> set[str]:
    """Names a nested function (or one nested in it) rebinds as nonlocal."""
    out: set[str] = set()
    dec = _bc.decode(code)
    for op, arg in dec.instrs:
        if op == _bc.STORE_DEREF:
            out.add(arg)
    for c in code.co_consts:
        if c.__class__ is code.__class__:
            out |= _cell_stores(c)
    return out


def delete_subscr(self: _Run, st: _State, obj: Any, ix: Any, dec: Any, i: int) -> None:
    c = obj.__class__
    if c is Seq:
        if not self.mutate(obj, dec, i):
            return
        if ix.__class__ is Slc:
            cut = self.subscr(obj, ix, dec, i)
            if cut.n is None or obj.n is None:
                obj.lose(cut.why)
            else:
                obj.n = _add(obj.n, cut.n, -1)
                obj.why = obj.why or cut.why
        elif obj.n.__class__ in _INT:
            obj.n = max(0, obj.n - 1)
        else:
            obj.n = _add(obj.n, -1)
        return
    if c is Map:
        self.changed.add(obj.name)
        return
    self.lose_escaped(st, _why(_line(dec, i), "`{}` could change it", ("x", dec, i)))


def import_name(self: _Run, name: str, fromlist: Any, level: Any, dec: Any, i: int) -> Any:
    if level.__class__ is Const and level.v == 0:
        wanted = fromlist.__class__ is Tup and fromlist.items
        module = self.read(_GLOBAL, sys.modules, name if wanted else name.partition(".")[0])
        if module is not _MISSING and module is not None:
            return Obj(module, name, _line(dec, i))
    return Opq(_why(_line(dec, i), "an imported module"))


# ── Wording an error (reads source; only when a version is rejected) ───────


def say(why: Any) -> str:
    line, template, parts = why
    place = parts[-1] if parts else None
    if (
        place.__class__ is tuple
        and place[:1] == ("fork",)
        and template.endswith("depending on {}")
        and _word(place[1], line) == "while"
    ):
        # A while's test copied above its loop: the loop decides, as at its bottom.
        return (
            f"`{_show(parts[0])}` changes inside the `while` loop at line {line}, "
            "which runs a number of times only the values decide"
        )
    return str(template.format(*[_show(p) for p in parts]))


def _word(dec: Any, line: int) -> str:
    """The keyword a source line starts with."""
    word = ""
    for ch in _segment_line(dec, line):
        if not (ch.isalpha() or ch == "_"):
            break
        word += ch
    return word


def _show(p: Any) -> str:
    if p.__class__ is tuple:
        if p and p[0].__class__ in _INT:
            return say(p)
        try:
            return _text(p)
        except Exception:  # wording falls back, never fails the error
            return "..."
    return str(p)


def _short(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= 60 else s[:57] + "..."


def _segment(dec: Any, i: int) -> str | None:
    import linecache

    pos = dec.positions[i] if 0 <= i < len(dec.positions) else None
    if pos is None or pos.lineno is None:
        return None
    lines = linecache.getlines(dec.code.co_filename)
    if not lines or pos.lineno > len(lines):
        return None
    end = pos.end_lineno or pos.lineno
    chunk = [line.encode() for line in lines[pos.lineno - 1 : end]]
    if pos.col_offset is None or pos.end_col_offset is None:
        return b"".join(chunk).decode(errors="replace").strip()
    if len(chunk) == 1:
        text = chunk[0][pos.col_offset : pos.end_col_offset]
    else:
        text = chunk[0][pos.col_offset :] + b"".join(chunk[1:-1]) + chunk[-1][: pos.end_col_offset]
    return text.decode(errors="replace")


def _node(text: str | None) -> Any:
    import ast

    if not text:
        return None
    text = text.strip()
    for mode, src in (("eval", text), ("eval", f"({text})"), ("exec", text)):
        try:
            tree = ast.parse(src, mode=mode)
        except SyntaxError:
            continue
        if mode == "eval":
            return tree.body  # type: ignore[attr-defined]
        return tree.body[0] if tree.body else None  # type: ignore[attr-defined]
    return None


def _text(p: tuple[Any, ...]) -> str:
    import ast

    kind, dec, i = p[0], p[1], p[2]
    line = _line(dec, i)
    if kind == "line":
        return str(line)
    src = _segment(dec, i)
    if kind == "fork":
        word = _word(dec, line)
        if word in ("if", "elif"):
            return f"the if at line {line}"
        if word in ("while", "case", "for", "assert"):
            return f"the `{word}` at line {line}"
        return f"the test at line {line}"
    if kind == "try":
        return f"whether the block at line {line} raises"
    if kind == "exit":
        return "where a loop stops"
    if kind == "noreturn":
        stripped = (src or "").strip()
        if stripped.startswith("return") and stripped[6:].strip():
            return "`" + _short(stripped[6:].strip()) + "` is not a list the compiler can follow"
        return "it can finish without returning a value"
    node = _node(src)
    target: Any = node
    if kind == "ret":
        target = getattr(node, "value", None) if isinstance(node, ast.Return) else None
    elif kind in ("arg", "fn"):
        call = node.value if isinstance(node, ast.Expr) else node
        if not isinstance(call, ast.Call):
            target = None
        elif kind == "fn":
            target = call.func
        else:
            k = p[3]
            pos = [a.value if isinstance(a, ast.Starred) else a for a in call.args]
            target = pos[k] if k < len(pos) else call.keywords[k - len(pos)].value
    elif kind in ("left", "right"):
        if isinstance(node, ast.BinOp):
            target = node.left if kind == "left" else node.right
        elif isinstance(node, ast.AugAssign):
            target = node.target if kind == "left" else node.value
        else:
            target = None
    elif kind in ("value", "slice"):
        if isinstance(node, ast.Subscript):
            if kind == "value":
                target = node.value
            else:
                s = node.slice
                target = getattr(s, ("lower", "upper", "step")[p[3]], None)
        else:
            target = None
    elif kind == "compif":
        gens = getattr(node, "generators", ())
        target = next((g.ifs[0] for g in gens if g.ifs), None)
    elif kind == "ifexp":
        target = _enclosing_test(dec, i)
    if target is None:
        return _short(src) if src and kind == "x" else "..."
    return _short(ast.unparse(target))


def _segment_line(dec: Any, line: int) -> str:
    import linecache

    return linecache.getline(dec.code.co_filename, line).strip() if line else ""


def _enclosing_test(dec: Any, i: int) -> Any:
    import ast
    import linecache

    pos = dec.positions[i]
    lines = linecache.getlines(dec.code.co_filename)
    try:
        tree = ast.parse("".join(lines))
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        tests = (
            [node.test]
            if isinstance(node, ast.IfExp)
            else node.values
            if isinstance(node, ast.BoolOp)
            else ()
        )
        for t in tests:
            if t.lineno == pos.lineno and t.col_offset == pos.col_offset:
                return node
    return None
