#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the packed op stream is written and read with one schema.

Usage:
    python tools/check-packed-schema.py [--root DIR] [--unchecked]

The kernel builder writes a program as a stream of records (a header word
with the opcode, the payload length and a mask of the slots that hold value
ids, then the payload) and the native module reads it. The opcodes, the
payload lengths, the kinds of the waveform slots and the codes inside a
payload are written by hand on both sides. This reads both sides from their
source text, without importing or building anything, and exits 1 naming each
opcode or field on which they disagree:

- the opcodes of ``kernel/packed_ir_builder.py`` and the ``kOp*`` constants
  of ``bindings.cpp``: the same names with the same values, each one a case
  of the decoder;
- the header: where the payload length and the slot mask sit in the word,
  and the length that says the next word holds the length;
- the payload length of each record that has a fixed one, at every place the
  package writes it, against the decoder's ``requirePayload``; and that such
  a record is written with as many words as its header says;
- each waveform's slots: integer or float, in order, the slots a mask bit
  may mark, and the name each one has in an error (``_WF_FIELDS``, which
  must also be the attribute names of ``kernel/ir_builder.py``'s
  ``OP_TABLE``); the two helpers every such record goes through,
  ``_emit_wf`` and ``_record``, must still have the statements that write
  the header, the mask and the length (the statements are pinned, not what
  the rest of each helper does);
- the codes inside a payload: numeric types (in every record that carries
  one), binary operations, compare predicates, schedule-block policies, line
  types, the unscheduled mark of a drive's times and the kinds of a site
  record;
- ``controller.py``'s waveform operand names and amplitude operands against
  the Pulse dialect's operands and the amplitude operands the native module
  lists, and that ``_clocks.py``'s time parameters are among those names.

A table this cannot find where it reads it is a failure that names the file
and the table, so a rewrite of either side is noticed.

Not checked, since the sources do not say it in a form to compare:

- which payload word holds which value id, and whether a literal slot of a
  record that is not a waveform (a wait's length, a phase, a frequency, a
  tone's) is an integer or a float's bits;
- the layout of the records whose length varies (loop and if ends, sync,
  schedule blocks, custom waveforms and samples, the classical records)
  beyond their opcode, and the sizes that follow from the header's
  (``MAX_CUSTOM_SAMPLES``, the place of the longest program's word, the
  room the builder makes before it writes a drive);
- the latency modes and flags of a classical function, and the marks inside
  a variable wait;
- the error names of the slots that have no ``_WF_FIELDS`` entry (the
  builder writes those messages one by one);
- which of ``_clocks.py``'s time parameters the builder converts.

``--unchecked`` lists the places it could not compare: a record written with
a length or an opcode set at run time, a header whose words it could not
count, an opcode no record is written with by name, and a slot name of the
decoder without a ``_WF_FIELDS`` entry.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE = "core/frontend/cudaq_pulse"
BUILDER = f"{PACKAGE}/kernel/packed_ir_builder.py"
IR_BUILDER = f"{PACKAGE}/kernel/ir_builder.py"
CONTROLLER = f"{PACKAGE}/controller.py"
CLOCKS = f"{PACKAGE}/_clocks.py"
IDLE = f"{PACKAGE}/_idle_windows.py"
BINDINGS = "core/mlir/bindings/bindings.cpp"
DIALECT = "core/mlir/include/cudaq-pulse/Dialect/Pulse/PulseOps.td"

# The line above the builder's opcodes.
_MARKER = "# OpCodes"
# The arith op the decoder builds for each binary operation, less its
# I, SI or F.
_BINARY_STEMS = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "floordiv": "FloorDiv",
    "mod": "Rem",
}
# The binary operations of integers alone.
_INTEGER_ONLY = ("floordiv", "mod")
_INT_PREDICATES = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}
_FLOAT_PREDICATES = {"==": "OEQ", "!=": "UNE", "<": "OLT", "<=": "OLE", ">": "OGT", ">=": "OGE"}
# The export's names of the dialect's operands, where they differ.
_EXPORT_NAMES = {
    ("square", "amp_real"): "amplitude",
    ("square", "amp_imag"): "amplitude_imag",
    ("scale", "scale"): "factor",
}
# Amplitude operands only the native module lists: ``pulse.scale`` is an op
# on a waveform, not a shape of the export's table.
_NATIVE_ONLY_AMPLITUDES = {"scale": (0,)}


class UnreadableError(Exception):
    """A table is not where, or not written as, this tool reads it."""


# The files read by the check that is running, by path.
_read: dict[tuple[Path, str], tuple[str, ast.Module | None]] = {}


def _text(root: Path, path: str) -> str:
    if (root, path) not in _read:
        try:
            _read[root, path] = ((root / path).read_text(encoding="utf-8"), None)
        except (OSError, UnicodeDecodeError) as error:
            why = getattr(error, "strerror", None) or type(error).__name__
            raise UnreadableError(f"{path}: cannot be read ({why})") from None
    return _read[root, path][0]


def _tree(root: Path, path: str) -> ast.Module:
    text = _text(root, path)
    tree = _read[root, path][1]
    if tree is None:
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            raise UnreadableError(f"{path}: does not parse (line {error.lineno})") from None
        _read[root, path] = (text, tree)
    return tree


def _snake(camel: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).upper()


def _int(node: ast.AST) -> int | None:
    """*node* as an integer literal (``-1`` too), else ``None``."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _int(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _literal(node: ast.AST) -> Any:
    """*node* as a literal; a ``MappingProxyType(...)`` as its argument."""
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        node = node.args[0]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _assignments(tree: ast.Module) -> dict[str, ast.expr]:
    """The module-level ``name = value`` and ``name: type = value``."""
    found: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target: ast.expr = statement.targets[0]
            value: ast.expr | None = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name) and value is not None:
            found[target.id] = value
    return found


def _table(root: Path, path: str, name: str) -> Any:
    value = _assignments(_tree(root, path)).get(name)
    table = None if value is None else _literal(value)
    if not isinstance(table, dict) or not table:
        raise UnreadableError(f"{path}: {name} is not a table of literals")
    return table


@dataclass
class Site:
    """A place that writes a record of *opcode*."""

    opcode: str
    where: str
    length: int | None = None
    kinds: tuple[str, ...] | None = None
    masked: int | None = None


@dataclass
class Written:
    """What the package writes, as its source says it."""

    opcodes: dict[str, int] = field(default_factory=dict)
    shifts: dict[str, int] = field(default_factory=dict)
    extended: int | None = None
    unscheduled: int | None = None
    codes: dict[str, dict[str, int]] = field(default_factory=dict)
    waveforms: dict[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    sites: list[Site] = field(default_factory=list)
    unscheduled_at: list[int] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)


def _opcodes(source: str, tree: ast.Module) -> dict[str, int]:
    """The builder's opcodes: the integers assigned below the marker, up to
    the first statement that assigns neither an integer nor a table."""
    lines = source.splitlines()
    marker = next((n for n, line in enumerate(lines, 1) if line.startswith(_MARKER)), None)
    if marker is None:
        raise UnreadableError(f"{BUILDER}: no '{_MARKER}' comment above the opcodes")
    opcodes: dict[str, int] = {}
    for statement in tree.body:
        if statement.lineno < marker:
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target: ast.expr = statement.targets[0]
            assigned: ast.expr | None = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, assigned = statement.target, statement.value
        else:
            break
        if not isinstance(target, ast.Name) or assigned is None:
            break
        value = _int(assigned)
        if value is None:
            if isinstance(assigned, ast.Dict):
                continue
            break
        if not target.id.startswith("_"):
            break
        opcodes[target.id[1:]] = value
    if not opcodes:
        raise UnreadableError(f"{BUILDER}: no opcodes below the '{_MARKER}' comment")
    return opcodes


def _shifts(tree: ast.Module) -> dict[str, int]:
    """The bit each argument of ``_header`` starts at."""
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef) and statement.name == "_header":
            body = statement.body[-1]
            if not isinstance(body, ast.Return) or body.value is None:
                break
            shifts: dict[str, int] = {}
            parts = [body.value]
            while parts:
                part = parts.pop()
                if isinstance(part, ast.BinOp) and isinstance(part.op, ast.BitOr):
                    parts += [part.left, part.right]
                elif isinstance(part, ast.Name):
                    shifts[part.id] = 0
                elif (
                    isinstance(part, ast.BinOp)
                    and isinstance(part.op, ast.LShift)
                    and isinstance(part.left, ast.Name)
                    and _int(part.right) is not None
                ):
                    shifts[part.left.id] = _int(part.right)  # type: ignore[assignment]
                else:
                    shifts.clear()
                    break
            if set(shifts) == {"opcode", "payload_len", "param_mask"}:
                return shifts
            break
    raise UnreadableError(
        f"{BUILDER}: _header is not opcode | (payload_len << n) | (param_mask << m)"
    )


def _mask_slot(node: ast.AST) -> int | None:
    """``k`` of ``1 << k``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift) and _int(node.left) == 1:
        return _int(node.right)
    return None


# What the two helpers must still do for their calls to be read as records:
# the statements of each, as ``ast.unparse`` writes them.
_HELPERS = {
    "_emit_wf": (
        "for i, (v, dtype) in enumerate(zip(vals, dtypes, strict=False)):",
        "param_mask |= 1 << i + 1",
        "payload_len = 1 + len(encoded)",
        "self._put([_header(wf_code, payload_len, param_mask), rv, *encoded])",
    ),
    "_record": (
        "n = len(payload)",
        "if n >= _MAX_PAYLOAD:",
        "self._put([_header(opcode, _MAX_PAYLOAD), n, *payload])",
        "self._put([_header(opcode, n), *payload])",
    ),
}
_KINDS = ("i64", "f64", "t64")


def _helpers(tree: ast.Module) -> None:
    """``_emit_wf(code, rv, vals, kinds)`` writes a header, the result and
    one word per kind, marking slot i + 1 for value i; ``_record(code,
    payload)`` writes a header for the payload's length and the payload."""
    bodies = {
        node.name: [line.strip() for line in ast.unparse(node).splitlines()]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
    }
    for name, statements in _HELPERS.items():
        lines = bodies.get(name, [])
        for statement in statements:
            if statement not in lines:
                raise UnreadableError(
                    f"{BUILDER}: {name} no longer has `{statement}`, by which this "
                    f"tool reads the records written through it"
                )
        # Each word of a record is written by one of the statements above.
        writes = sum(line.startswith("self._put(") for line in statements)
        if sum("self._put(" in line or "self._w(" in line for line in lines) != writes:
            raise UnreadableError(
                f"{BUILDER}: {name} no longer writes its record with {writes} "
                f"self._put, as this tool reads the records written through it"
            )


class _Sites(ast.NodeVisitor):
    """The records one file writes: ``_header(...)``, ``_record(...)`` and
    ``_emit_wf(...)`` calls, and how many words follow a header."""

    def __init__(
        self, path: str, written: Written, headers: dict[str, Site], constants: set[str]
    ) -> None:
        self.path = path
        self.written = written
        self.headers = headers
        self.constants = constants
        self.inside: list[str] = []
        # The header calls of a fixed length, and those whose words are counted.
        self.fixed: dict[int, Site] = {}
        self.counted: set[int] = set()

    def _where(self, node: ast.AST) -> str:
        return f"{self.path}:{getattr(node, 'lineno', 0)}"

    def _opcode(self, node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Name)
            and node.id[:1] == "_"
            and node.id[1:] in self.written.opcodes
        ):
            return node.id[1:]
        return None

    def _header(self, node: ast.AST) -> Site | None:
        """The site of a header: a ``_header`` call or a named one."""
        if isinstance(node, ast.Name):
            return self.headers.get(node.id)
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_header"
            and len(node.args) >= 2
        ):
            return None
        opcode = self._opcode(node.args[0])
        if opcode is None:
            return None
        site = Site(opcode, self._where(node), _int(node.args[1]))
        if len(node.args) > 2:
            site.masked = _mask_slot(node.args[2])
            if site.masked is None:
                self.written.unchecked.append(f"{site.where}: the mask of {site.opcode}")
        return site

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.inside.append(node.name)
        self.generic_visit(node)
        self.inside.pop()

    def generic_visit(self, node: ast.AST) -> None:
        super().generic_visit(node)
        for name in ("body", "orelse", "finalbody"):
            body = getattr(node, name, None)
            if isinstance(body, list):
                self._runs(body)

    def _runs(self, body: list[ast.stmt]) -> None:
        """A header written with ``self._w`` is followed by its payload,
        one ``self._w`` each."""

        def word(statement: ast.stmt) -> ast.expr | None:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_w"
                and len(call.args) == 1
            ):
                return call.args[0]
            return None

        for at, statement in enumerate(body):
            first = word(statement)
            site = None if first is None else self._header(first)
            if site is None or site.length is None:
                continue
            self.counted.add(id(first))
            count = 0
            for later in body[at + 1 :]:
                following = word(later)
                if following is None or self._header(following) is not None:
                    break
                count += 1
            if count != site.length:
                self.written.problems.append(
                    f"opcode {site.opcode}: {site.where} writes a header for "
                    f"{site.length} payload words and then {count}"
                )

    def _sequence(self, node: ast.Tuple | ast.List) -> None:
        if not node.elts or any(isinstance(e, ast.Starred) for e in node.elts):
            return
        site = self._header(node.elts[0])
        if site is None or site.length is None:
            return
        self.counted.add(id(node.elts[0]))
        if len(node.elts) != site.length + 1:
            self.written.problems.append(
                f"opcode {site.opcode}: {self._where(node)} writes a header for "
                f"{site.length} payload words and then {len(node.elts) - 1}"
            )
        if site.opcode == "DRIVE":
            self.written.unscheduled_at = [
                at - 1
                for at, e in enumerate(node.elts)
                if isinstance(e, ast.Name) and e.id == "_UNSCHEDULED"
            ]

    def visit_Tuple(self, node: ast.Tuple) -> None:
        self._sequence(node)
        self.generic_visit(node)

    def visit_List(self, node: ast.List) -> None:
        self._sequence(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if name not in ("_header", "_record", "_emit_wf") or not node.args:
            return
        where = self._where(node)
        opcode = self._opcode(node.args[0])
        if opcode is None:
            given = node.args[0]
            if isinstance(given, ast.Name) and given.id in self.constants:
                self.written.problems.append(
                    f"opcode {given.id.lstrip('_')}: {where} writes it, and it is not "
                    f"among the opcodes below the '{_MARKER}' comment of {BUILDER}"
                )
            # The helpers themselves pass on the opcode they are given.
            elif not (self.inside and self.inside[-1] in _HELPERS):
                self.written.unchecked.append(
                    f"{where}: {name}({ast.unparse(given)}, ...), an opcode chosen at run time"
                )
            return
        site = Site(opcode, where)
        if name == "_header":
            site = self._header(node) or site
            if site.length is not None:
                self.fixed[id(node)] = site
        elif name == "_record" and len(node.args) == 2 and isinstance(node.args[1], ast.List):
            if not any(isinstance(e, ast.Starred) for e in node.args[1].elts):
                site.length = len(node.args[1].elts)
        elif name == "_emit_wf" and len(node.args) == 4:
            kinds = _literal(node.args[3])
            if isinstance(kinds, list) and all(isinstance(k, str) for k in kinds):
                site.kinds = tuple(kinds)
                site.length = 1 + len(kinds)
                for kind in kinds:
                    if kind not in _KINDS:
                        self.written.problems.append(
                            f"opcode {opcode}: {where} writes a slot of kind {kind!r}, "
                            f"not one of {', '.join(_KINDS)}"
                        )
        if site.length is None:
            self.written.unchecked.append(
                f"{site.where}: a {site.opcode} record of a length set at run time"
            )
        self.written.sites.append(site)

    def uncounted(self) -> None:
        """The headers of a fixed length whose words were not counted: not
        the first of a tuple or list written whole, nor of a run of
        ``self._w``, nor a header kept by name."""
        named = {site.where for site in self.headers.values()}
        for at, site in self.fixed.items():
            if at not in self.counted and site.where not in named:
                self.written.unchecked.append(
                    f"{site.where}: the words written after the header of {site.opcode}"
                )


def written(root: Path) -> Written:
    """The schema the Python sources write."""
    out = Written()
    source = _text(root, BUILDER)
    tree = _tree(root, BUILDER)
    out.opcodes = _opcodes(source, tree)
    out.shifts = _shifts(tree)
    _helpers(tree)
    names = _assignments(tree)
    constants = {name for name, value in names.items() if _int(value) is not None}
    for name, into in (("_MAX_PAYLOAD", "extended"), ("_UNSCHEDULED", "unscheduled")):
        number = _int(names[name]) if name in names else None
        if number is None:
            raise UnreadableError(f"{BUILDER}: {name} is not an integer literal")
        setattr(out, into, number)
    for name in ("_BLOCK_POLICY", "_COMPARE", "_NUMERIC_TYPE", "_BINARY_OP", "_LINE_TYPE_CODES"):
        out.codes[name] = _table(root, BUILDER, name)
    out.codes["_SITE_KINDS"] = _table(root, IDLE, "_SITE_KINDS")
    fields = names.get("_WF_FIELDS")
    if not isinstance(fields, ast.Dict):
        raise UnreadableError(f"{BUILDER}: _WF_FIELDS is not a table")
    for key, value in zip(fields.keys, fields.values, strict=True):
        entry = _literal(value)
        if not (
            isinstance(key, ast.Name)
            and isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], tuple)
        ):
            raise UnreadableError(
                f"{BUILDER}: an entry of _WF_FIELDS is not opcode: (name, fields)"
            )
        out.waveforms[key.id.removeprefix("_")] = (entry[0], entry[1])
    # The headers made once, by name.
    finder = _Sites(BUILDER, out, {}, constants)
    headers = {
        name: site
        for name, value in names.items()
        if (site := finder._header(value)) is not None and site.length is not None
    }
    out.unchecked.clear()
    for path in writers(root):
        tree = _tree(root, path)
        own = {name for name, value in _assignments(tree).items() if _int(value) is not None}
        sites = _Sites(path, out, headers if path == BUILDER else {}, constants | own)
        sites.visit(tree)
        sites.uncounted()
    out.unchecked = list(dict.fromkeys(out.unchecked))
    for name in out.opcodes:
        if not any(site.opcode == name for site in out.sites):
            out.unchecked.append(f"{name}: no record of it is written with its opcode by name")
    return out


def writers(root: Path) -> list[str]:
    """The files of the package that may write a record."""
    found = []
    for file in sorted((root / PACKAGE).rglob("*.py")):
        path = file.relative_to(root).as_posix()
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(call in text for call in ("_header(", "._record(", "._emit_wf(")):
            found.append(path)
    return found


def sources(root: Path) -> list[str]:
    """Every file the check reads."""
    fixed = [BUILDER, IR_BUILDER, CONTROLLER, CLOCKS, IDLE, BINDINGS, DIALECT]
    return sorted({*fixed, *writers(root)})


@dataclass
class Case:
    """One opcode's case of the decoder."""

    length: int | None = None
    kinds: dict[int, str] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    masked: set[int] = field(default_factory=set)
    body: str = ""


@dataclass
class Read:
    """What the native module reads, as its source says it."""

    opcodes: dict[str, int] = field(default_factory=dict)
    cases: dict[str, Case] = field(default_factory=dict)
    shifts: dict[str, int] = field(default_factory=dict)
    widths: dict[str, int] = field(default_factory=dict)
    extended: int | None = None
    problems: list[str] = field(default_factory=list)


def _one(pattern: str, text: str, what: str, flags: int = 0) -> re.Match[str]:
    found = re.search(pattern, text, flags)
    if found is None:
        raise UnreadableError(f"{BINDINGS}: cannot find {what}")
    return found


def _number(text: str, what: str) -> int:
    try:
        return int(text, 0)
    except ValueError:
        raise UnreadableError(f"{BINDINGS}: {what} is {text}, not a number") from None


def read(root: Path) -> Read:
    """The schema the native decoder reads."""
    out = Read()
    text = _text(root, BINDINGS)
    for name, value in re.findall(r"^constexpr int kOp(\w+) = ([^;]+);", text, re.M):
        if _snake(name) in out.opcodes:
            out.problems.append(f"opcode {_snake(name)}: {BINDINGS} defines kOp{name} twice")
        out.opcodes[_snake(name)] = _number(value, f"kOp{name}")
    if not out.opcodes:
        raise UnreadableError(f"{BINDINGS}: no 'constexpr int kOp... = n;' opcodes")
    out.extended = _number(
        _one(r"constexpr int kExtendedPayload = (\w+);", text, "kExtendedPayload")[1],
        "kExtendedPayload",
    )
    opcode = _one(r"int opcode = static_cast<int>\(hdr & (\w+)\);", text, "the header's opcode")
    length = _one(r"int64_t plen = \(hdr >> (\d+)\) & (\w+);", text, "the header's length")
    mask = _one(
        r"int pmask = static_cast<int>\(\(hdr >> (\d+)\) & (\w+)\);", text, "the header's mask"
    )
    out.shifts = {"opcode": 0, "payload_len": int(length[1]), "param_mask": int(mask[1])}
    out.widths = {
        "opcode": _number(opcode[1], "the header's opcode mask").bit_length(),
        "payload_len": _number(length[2], "the header's length mask").bit_length(),
        "param_mask": _number(mask[2], "the header's slot mask").bit_length(),
    }
    # The decoder's switch, from the header's words to its default.
    start = text.find("switch (opcode) {", mask.end())
    if start < 0:
        raise UnreadableError(f"{BINDINGS}: cannot find the decoder's switch (opcode)")
    first = re.compile(r"^( *)case kOp\w+:", re.M).search(text, start)
    if first is None:
        raise UnreadableError(f"{BINDINGS}: the decoder's switch has no case kOp...")
    indent = first[1]
    end = re.compile(rf"^{indent}default:", re.M).search(text, start)
    if end is None:
        raise UnreadableError(f"{BINDINGS}: the decoder's switch has no default")
    # Without its comments, which may name what the code does not do.
    switch = re.sub(
        r'("(?:\\.|[^"\\\n])*")|//[^\n]*|/\*.*?\*/',
        lambda found: found[1] or "",
        text[first.start() : end.start()],
        flags=re.S,
    )
    labels = list(re.finditer(rf"^{indent}case kOp(\w+):", switch, re.M))
    pending: list[str] = []
    for label, following in zip(labels, [*labels[1:], None], strict=True):
        body = switch[label.end() : len(switch) if following is None else following.start()]
        name = _snake(label[1])
        if name in out.cases or name in pending:
            out.problems.append(f"opcode {name}: {BINDINGS} has two cases for kOp{label[1]}")
        pending.append(name)
        if not body.strip():
            continue  # falls through to the next label
        case = Case(body=body)
        lengths = {int(n) for n in re.findall(r"requirePayload\((\d+)\)", body)}
        if len(lengths) > 1:
            out.problems.append(
                f"opcode {name}: {BINDINGS} requires payloads of {sorted(lengths)} words"
            )
        case.length = lengths.pop() if len(lengths) == 1 else -1 if lengths else None
        for slot in re.findall(r"getI64Arg\((\d+)\)", body):
            case.kinds[int(slot)] = "i64"
        for slot in re.findall(r"getF64Arg\((\d+)\)", body):
            case.kinds[int(slot)] = "f64"
        for _, slot, quoted in re.findall(
            r'\b(finite|positive)\(\s*(\d+),\s*((?:"[^"]*"\s*)+)\)', body
        ):
            case.kinds[int(slot)] = "f64"
            case.names[int(slot)] = "".join(re.findall(r'"([^"]*)"', quoted))
        case.masked = {int(k) for k in re.findall(r"pmask & \(1 << (\d+)\)", body)}
        case.masked |= set(case.kinds)
        for each in pending:
            out.cases[each] = case
        pending = []
    return out


# The letter of a site record's kind for the op the decoder tests for.
_SITE_OPS = {"Drive": "d", "Readout": "r", "Wait": "w", "Sync": "s"}


def _guarded(out: list[str], what: str, part: Callable[[], None]) -> None:
    """Run *part*; a table it cannot read, or one whose values are not of
    the kind it reads (a number where a tuple is, say), is a problem."""
    try:
        part()
    except UnreadableError as error:
        out.append(str(error))
    except (TypeError, ValueError, IndexError, KeyError, AttributeError) as error:
        out.append(f"{what} is not written as this tool reads it ({type(error).__name__}: {error})")


def _codes(found: Read, out: list[str], sent: Written) -> None:
    """The codes inside a payload, table by table: one that cannot be read
    does not hide the next."""

    def body(opcode: str) -> str:
        case = found.cases.get(opcode)
        if case is None:
            raise UnreadableError(f"{BINDINGS}: no case for {opcode}")
        return case.body

    def differs(table: str, name: str, ours: object, theirs: object, where: str) -> None:
        if ours != theirs:
            out.append(
                f"{table}[{name!r}]: {BUILDER if table != '_SITE_KINDS' else IDLE} says "
                f"{ours}, {BINDINGS} ({where}) says {theirs}"
            )

    numeric = sent.codes["_NUMERIC_TYPE"]

    def types(where: str, codes: dict[str, int]) -> None:
        if set(codes) != {"i64", "f64"}:
            raise UnreadableError(f"{BINDINGS}: cannot read the numeric types of {where}")
        for kind, code in sorted(codes.items()):
            differs("_NUMERIC_TYPE", kind, numeric.get(kind), code, where)

    def numbers() -> None:
        made = {"Int": "i64", "Float": "f64", "SubI": "i64", "NegF": "f64"}
        for opcode, where, ops in (
            ("NUM_CONST", "kOpNumConst", "ConstantInt|ConstantFloat"),
            ("NUM_NEG", "kOpNumNeg", "SubI|NegF"),
        ):
            coded = re.findall(rf"p\[1\] == (\d+)\) \{{[^{{}}]*?arith::({ops})Op", body(opcode))
            types(where, {made[op.removeprefix("Constant")]: int(code) for code, op in coded})
        casts = re.findall(
            r"p\[1\] == (\d+) && p\[2\] == (\d+)\)\s*bind\(p\[0\],\s*"
            r"mlir::arith::(SIToFP|FPToSI)Op",
            body("NUM_CAST"),
        )
        if sorted(op for _, _, op in casts) != ["FPToSI", "SIToFP"]:
            raise UnreadableError(f"{BINDINGS}: cannot read the numeric types of kOpNumCast")
        for source, target, op in casts:
            kinds = ("i64", "f64") if op == "SIToFP" else ("f64", "i64")
            types(f"kOpNumCast, {op}", {kinds[0]: int(source), kinds[1]: int(target)})

    def binary() -> None:
        table = sent.codes["_BINARY_OP"]
        parts = re.split(r"p\[1\] == (\d+)\)", body("NUM_BINARY"))
        codes, branches = parts[1::2], parts[2::2]
        if len(branches) != 2:
            raise UnreadableError(f"{BINDINGS}: cannot read the two type branches of kOpNumBinary")
        kinds = {}
        for code, branch in zip(codes, branches, strict=True):
            made = {
                int(n): op
                for n, op in re.findall(r"case (\d+):\s*result =\s*mlir::arith::(\w+)Op::", branch)
            }
            if not made or len({op[-1] for op in made.values()}) != 1:
                raise UnreadableError(f"{BINDINGS}: cannot read the operations of kOpNumBinary")
            kind = "f64" if next(iter(made.values())).endswith("F") else "i64"
            kinds[kind] = int(code)
            suffixes = ("F",) if kind == "f64" else ("I", "SI")
            for name, n in table.items():
                stem = _BINARY_STEMS.get(name)
                if stem is None:
                    out.append(f"_BINARY_OP[{name!r}]: this tool knows no arith op for it")
                elif n not in made:
                    if kind == "i64" or name not in _INTEGER_ONLY:
                        out.append(
                            f"_BINARY_OP[{name!r}]: {BUILDER} says {n}, which "
                            f"{BINDINGS} (kOpNumBinary, {kind}) does not read"
                        )
                elif made[n] not in [stem + suffix for suffix in suffixes]:
                    out.append(
                        f"_BINARY_OP[{name!r}]: {BUILDER} says {n}, where {BINDINGS} "
                        f"(kOpNumBinary, {kind}) builds arith.{made[n]}"
                    )
            for n in sorted(set(made) - set(table.values())):
                out.append(
                    f"_BINARY_OP: {BINDINGS} (kOpNumBinary, {kind}) reads {n} "
                    f"(arith.{made[n]}), which {BUILDER} does not write"
                )
        types("kOpNumBinary", kinds)

    def compare() -> None:
        table = sent.codes["_COMPARE"]
        text = body("CLASSICAL_COMPARE")
        orders = []
        for array, enum, expected in (
            ("ip", "CmpIPredicate", _INT_PREDICATES),
            ("fp", "CmpFPredicate", _FLOAT_PREDICATES),
        ):
            listed = re.search(rf"{enum} {array}\[\] = \{{([^}}]*)\}}", text)
            if listed is None:
                raise UnreadableError(f"{BINDINGS}: cannot read {array}[] of kOpClassicalCompare")
            order = re.findall(rf"{enum}::(\w+)", listed[1])
            orders.append(len(order))
            for name, predicate in expected.items():
                at = table.get(name)
                theirs = order[at] if at is not None and 0 <= at < len(order) else None
                if theirs != predicate:
                    out.append(
                        f"_COMPARE[{name!r}]: {BUILDER} says {at}, where {BINDINGS} "
                        f"(kOpClassicalCompare, {array}[]) has {theirs}, not {predicate}"
                    )
        both = re.search(r"p\[1\] == (\d+)\)\s*result = mlir::arith::(And|Or)IOp", text)
        highest = re.search(r"p\[1\] > (\d+) \|\|", text)
        logical = re.search(r"if \(p\[1\] > (\d+)\) \{", text)
        floats = re.search(r"p\[2\] == (\d+)\) \{\s*static const mlir::arith::CmpFPredicate", text)
        if both is None or highest is None or logical is None or floats is None:
            raise UnreadableError(f"{BINDINGS}: cannot read the branches of kOpClassicalCompare")
        first, second = ("and", "or") if both[2] == "And" else ("or", "and")
        differs("_COMPARE", first, table.get(first), int(both[1]), "kOpClassicalCompare")
        differs("_COMPARE", second, table.get(second), int(highest[1]), "kOpClassicalCompare")
        if max(table.values()) != int(highest[1]):
            out.append(
                f"_COMPARE: {BUILDER} writes up to {max(table.values())}, {BINDINGS} "
                f"(kOpClassicalCompare) reads up to {highest[1]}"
            )
        if {int(logical[1]) + 1} != set(orders):
            out.append(
                f"_COMPARE: {BINDINGS} (kOpClassicalCompare) reads the codes above "
                f"{logical[1]} as and/or, and has {orders} predicates below them"
            )
        # The type word is 1 for a float, as the builder's int(is_float).
        differs("_NUMERIC_TYPE", "f64", numeric.get("f64"), int(floats[1]), "kOpClassicalCompare")

    def policies() -> None:
        text = body("BLOCK_BEGIN")
        policy = re.search(r'p\[0\] == (\d+) \? "(\w+)" : "(\w+)"', text)
        limit = re.search(r"p\[0\] > (\d+)", text)
        if policy is None or limit is None:
            raise UnreadableError(f"{BINDINGS}: cannot read the policies of kOpBlockBegin")
        table = sent.codes["_BLOCK_POLICY"]
        differs("_BLOCK_POLICY", policy[2], table.get(policy[2]), int(policy[1]), "kOpBlockBegin")
        if table.get(policy[3]) in (None, int(policy[1])) or set(table) != {policy[2], policy[3]}:
            out.append(
                f"_BLOCK_POLICY: {BUILDER} has {table}, {BINDINGS} (kOpBlockBegin) "
                f"reads {policy[1]} as {policy[2]} and any other as {policy[3]}"
            )
        if max(table.values()) != int(limit[1]):
            out.append(
                f"_BLOCK_POLICY: {BUILDER} writes up to {max(table.values())}, "
                f"{BINDINGS} (kOpBlockBegin) reads up to {limit[1]}"
            )

    def lines() -> None:
        line = re.search(
            r"vtype == (\d+)\s*\? mlir::Type\((\w+)Ty\)\s*:\s*mlir::Type\((\w+)Ty\)", body("SYNC")
        )
        if line is None:
            raise UnreadableError(f"{BINDINGS}: cannot read the line types of kOpSync")
        table = sent.codes["_LINE_TYPE_CODES"]
        name = f"{line[2]}_line"
        differs("_LINE_TYPE_CODES", name, table.get(name), int(line[1]), "kOpSync")
        if table.get(f"{line[3]}_line") in (None, int(line[1])) or len(table) != 2:
            out.append(
                f"_LINE_TYPE_CODES: {BUILDER} has {table}, {BINDINGS} (kOpSync) reads "
                f"{line[1]} as a {line[2]} line and any other as a {line[3]} line"
            )

    def drive() -> None:
        text = body("DRIVE")
        times = re.findall(r"p\[(\d+)\] != (-?\d+)\s*\?", text)
        marks = {int(mark) for _, mark in times}
        if len(marks) != 1 or len(times) != len(re.findall(r"std::optional<int64_t>\(p\[", text)):
            raise UnreadableError(f"{BINDINGS}: cannot read the unscheduled mark of kOpDrive")
        slots = sorted(int(slot) for slot, _ in times)
        if marks != {sent.unscheduled}:
            out.append(
                f"_UNSCHEDULED: {BUILDER} says {sent.unscheduled}, {BINDINGS} "
                f"(kOpDrive) says {marks.pop()}"
            )
        if sent.unscheduled_at != slots:
            out.append(
                f"opcode DRIVE: {BUILDER} writes _UNSCHEDULED in payload words "
                f"{sent.unscheduled_at}, {BINDINGS} reads the times from words {slots}"
            )

    def sites() -> None:
        text = body("SITE")
        single = re.findall(r"isa<pulse::(\w+)Op>\(op\)\s*\? (\d+)", text)
        frame = re.search(r"isa<pulse::\w+Op,[^>]*>\(op\)\s*\? (\d+)", text)
        if sorted(op for op, _ in single) != sorted(_SITE_OPS) or frame is None:
            raise UnreadableError(f"{BINDINGS}: cannot read the kinds of kOpSite")
        kinds = {_SITE_OPS[op]: int(code) for op, code in single}
        kinds["f"] = int(frame[1])
        table = sent.codes["_SITE_KINDS"]
        for letter in sorted(set(kinds) | set(table)):
            differs("_SITE_KINDS", letter, table.get(letter), kinds.get(letter), "kOpSite")

    for part in (numbers, binary, compare, policies, lines, drive, sites):
        _guarded(out, f"{BUILDER} or {IDLE}: a table of codes", part)


def _operands(root: Path, out: list[str]) -> None:
    """The export's waveform operand tables against the dialect and the
    native module's amplitude operands."""
    parameters = _table(root, CONTROLLER, "_PARAMETERS")
    amplitudes = _table(root, CONTROLLER, "_AMPLITUDES")
    times = _table(root, CLOCKS, "_TIMES_OF")
    dialect = _text(root, DIALECT)
    declared: dict[str, list[tuple[str, str]]] = {}
    for op in re.finditer(r'^def \w+ : PulseOp<"(\w+)"', dialect, re.M):
        following = re.compile(r"^def ", re.M).search(dialect, op.end())
        text = dialect[op.end() : following.start() if following else len(dialect)]
        arguments = re.search(r"let arguments = \(ins(.*?)\);", text, re.S)
        if arguments is not None:
            declared[op[1]] = re.findall(r"(\w+(?:<[^$]*>)?):\$(\w+)", arguments[1])
    if not declared:
        raise UnreadableError(f"{DIALECT}: no 'def ... : PulseOp<\"name\"' with arguments")
    numbers: dict[str, list[tuple[str, str]]] = {}
    for shape, names in parameters.items():
        if shape not in declared:
            out.append(f"_PARAMETERS[{shape!r}]: {DIALECT} has no pulse.{shape}")
            continue
        # The operands that are numbers, as the export counts them.
        numbers[shape] = [
            (kind, name)
            for kind, name in declared[shape]
            if "Waveform" not in kind and "Attr" not in kind
        ]
        theirs = tuple(_EXPORT_NAMES.get((shape, name), name) for _, name in numbers[shape])
        if tuple(names) != theirs:
            out.append(
                f"_PARAMETERS[{shape!r}]: {CONTROLLER} says {tuple(names)}, the "
                f"operands of pulse.{shape} in {DIALECT} are {theirs}"
            )
    for shape, at in amplitudes.items():
        for index in at:
            kind = numbers[shape][index][0] if index < len(numbers.get(shape, ())) else None
            if kind is None or "Float" not in kind:
                out.append(
                    f"_AMPLITUDES[{shape!r}]: {CONTROLLER} says operand {index}, which "
                    f"is {'no operand' if kind is None else kind} of pulse.{shape} in {DIALECT}"
                )
    listed = _one(
        r"std::map<std::string, std::set<unsigned>> amplitudes = \{(.*?)\};",
        _text(root, BINDINGS),
        "the amplitudes of parameter_roles",
        re.S,
    )
    native = {
        name: tuple(sorted(int(n) for n in re.findall(r"\d+", at)))
        for name, at in re.findall(r'\{"pulse\.(\w+)",\s*\{([\d,\s]*)\}\}', listed[1])
    }
    if not native:
        raise UnreadableError(f"{BINDINGS}: cannot read the amplitudes of parameter_roles")
    ours = {shape: tuple(sorted(at)) for shape, at in amplitudes.items()}
    for shape in sorted(set(ours) | set(native) | set(_NATIVE_ONLY_AMPLITUDES)):
        if shape in _NATIVE_ONLY_AMPLITUDES and shape not in ours:
            if native.get(shape) != _NATIVE_ONLY_AMPLITUDES[shape]:
                out.append(
                    f"_AMPLITUDES[{shape!r}]: {BINDINGS} (parameter_roles) says "
                    f"{native.get(shape)}, this tool expects {_NATIVE_ONLY_AMPLITUDES[shape]} "
                    f"of an op that is not a shape of {CONTROLLER}"
                )
        elif ours.get(shape) != native.get(shape):
            out.append(
                f"_AMPLITUDES[{shape!r}]: {CONTROLLER} says {ours.get(shape)}, "
                f"{BINDINGS} (parameter_roles) says {native.get(shape)}"
            )
    for shape, names in times.items():
        unknown = [name for name in names if name not in parameters.get(shape, ())]
        if unknown:
            out.append(
                f"_TIMES_OF[{shape!r}]: {CLOCKS} names {unknown}, which "
                f"_PARAMETERS[{shape!r}] in {CONTROLLER} does not"
            )


def check(root: Path) -> tuple[list[str], list[str]]:
    """The disagreements under *root*, and the places not checked."""
    out: list[str] = []
    _read.clear()
    try:
        sent = written(root)
        found = read(root)
    except UnreadableError as error:
        return [str(error)], []
    out += sent.problems + found.problems

    for name in sorted(
        set(sent.opcodes) | set(found.opcodes),
        key=lambda n: (sent.opcodes.get(n, found.opcodes.get(n, 0)), n),
    ):
        ours, theirs = sent.opcodes.get(name), found.opcodes.get(name)
        if ours is None:
            out.append(f"opcode {name}: {BINDINGS} says {theirs}, {BUILDER} has no _{name}")
        elif theirs is None:
            out.append(f"opcode {name}: {BUILDER} says {ours}, {BINDINGS} has no kOp for it")
        elif ours != theirs:
            out.append(f"opcode {name}: {BUILDER} says {ours}, {BINDINGS} says {theirs}")
        if theirs is not None and name not in found.cases:
            out.append(f"opcode {name}: the decoder in {BINDINGS} has no case for it")
    for side, opcodes in ((BUILDER, sent.opcodes), (BINDINGS, found.opcodes)):
        seen: dict[int, str] = {}
        for name, value in opcodes.items():
            if value in seen:
                out.append(f"opcode {name}: {side} gives it {value}, as {seen[value]}")
            seen[value] = name
            if value >> found.widths["opcode"]:
                out.append(f"opcode {name}: {value} does not fit the header's opcode bits")

    for part in ("payload_len", "param_mask"):
        if sent.shifts[part] != found.shifts[part]:
            out.append(
                f"header {part}: {BUILDER} writes it at bit {sent.shifts[part]}, "
                f"{BINDINGS} reads it at bit {found.shifts[part]}"
            )
    if found.widths["opcode"] != found.shifts["payload_len"] or (
        found.shifts["payload_len"] + found.widths["payload_len"] != found.shifts["param_mask"]
    ):
        out.append(f"header: the fields {BINDINGS} reads overlap or leave a gap")
    if sent.extended != found.extended:
        out.append(
            f"header extended length: {BUILDER} (_MAX_PAYLOAD) says {sent.extended}, "
            f"{BINDINGS} (kExtendedPayload) says {found.extended}"
        )
    if found.extended != (1 << found.widths["payload_len"]) - 1:
        out.append(f"header extended length: {found.extended} is not the length field's highest")

    for site in sent.sites:
        case = found.cases.get(site.opcode)
        if case is None:
            continue
        if site.length is not None and site.length != case.length and case.length != -1:
            out.append(
                f"opcode {site.opcode}: {site.where} writes {site.length} payload "
                f"words, {BINDINGS} "
                + (
                    "has no requirePayload for it"
                    if case.length is None
                    else f"requires {case.length}"
                )
            )
        marked = [site.masked] if site.kinds is None else list(range(1, len(site.kinds) + 1))
        if any(slot is not None and slot >= found.widths["param_mask"] for slot in marked):
            out.append(
                f"opcode {site.opcode}: {site.where} marks a slot beyond the "
                f"{found.widths['param_mask']} bits of the mask {BINDINGS} reads"
            )
        if site.masked is not None and site.masked not in case.masked:
            out.append(
                f"opcode {site.opcode}: {site.where} marks slot {site.masked} as a "
                f"value id, which {BINDINGS} reads as a literal"
            )
        if site.kinds is not None:
            # "t64" is a time written as a float.
            kinds = {at: "i64" if k == "i64" else "f64" for at, k in enumerate(site.kinds, 1)}
            for slot in sorted(set(kinds) | set(case.kinds)):
                if kinds.get(slot) != case.kinds.get(slot):
                    out.append(
                        f"opcode {site.opcode}, slot {slot}: {site.where} writes "
                        f"{kinds.get(slot)}, {BINDINGS} reads {case.kinds.get(slot)}"
                    )

    try:
        table = _table(root, IR_BUILDER, "OP_TABLE")
    except UnreadableError as error:
        out.append(str(error))
        table = {}
    for opcode, (op, fields) in sent.waveforms.items():
        case = found.cases.get(opcode)
        if case is None:
            out.append(f"_WF_FIELDS: {BUILDER} names {opcode}, which {BINDINGS} does not read")
            continue
        for slot, name in sorted(case.names.items()):
            named = f"{op} {fields[slot - 1]}" if slot - 1 < len(fields) else None
            if named != name:
                out.append(
                    f"opcode {opcode}, slot {slot}: {BUILDER} (_WF_FIELDS) names it "
                    f"{named!r}, {BINDINGS} names it {name!r}"
                )
        entry = table.get(op)
        attributes = entry[1] if isinstance(entry, tuple) and len(entry) > 1 else None
        if attributes != fields:
            out.append(
                f"_WF_FIELDS[{opcode}]: {BUILDER} says {fields}, OP_TABLE[{op!r}] in "
                f"{IR_BUILDER} says {attributes}"
            )

    unnamed = [
        f"{name}: the decoder's name of slot {slot} ({text!r}) has no _WF_FIELDS entry"
        for name, case in found.cases.items()
        if name not in sent.waveforms
        for slot, text in sorted(case.names.items())
    ]
    _codes(found, out, sent)
    _guarded(
        out,
        f"{CONTROLLER} or {CLOCKS}: _PARAMETERS, _AMPLITUDES or _TIMES_OF",
        lambda: _operands(root, out),
    )
    return out, sent.unchecked + unnamed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--unchecked", action="store_true", help="list the records this does not check"
    )
    args = parser.parse_args(argv)
    problems, unchecked = check(args.root)
    if args.unchecked:
        for line in unchecked:
            print(f"not checked: {line}")
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        print(f"{len(problems)} disagreement(s) in the packed schema", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
