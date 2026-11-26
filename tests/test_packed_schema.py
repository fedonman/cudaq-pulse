# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tools/check-packed-schema.py: the packed op stream's opcodes, payload
lengths, slot kinds and codes are the same where the package writes them
and where the native module reads them, as their sources say."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "check-packed-schema.py"
# Without the sources (an installed package's tests) there is nothing to
# read; with them, a file the tool does not find is its failure, not a skip.
if not _TOOL.exists() or not (_ROOT / "core" / "mlir").is_dir():
    pytest.skip("the tool and the sources it reads are not in this tree", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("check_packed_schema", _TOOL)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
# The tool's dataclasses look their module up by name.
sys.modules[_spec.name] = check
_spec.loader.exec_module(check)


def test_the_sources_agree():
    problems, unchecked = check.check(_ROOT)
    assert problems == []
    # What the check reads: every opcode on both sides, and a payload length
    # for most of them.
    written, read = check.written(_ROOT), check.read(_ROOT)
    assert written.opcodes == read.opcodes and len(read.opcodes) >= 44
    assert set(read.cases) == set(read.opcodes)
    fixed = {site.opcode for site in written.sites if site.length is not None}
    assert len(fixed) >= 30 and {"DRIVE", "READOUT", "WAIT", "SELECT", "SITE"} <= fixed
    # The records it cannot read are listed, not passed over.
    assert any("SYNC" in line for line in unchecked)
    assert any("an opcode chosen at run time" in line for line in unchecked)
    assert any("'set_phase phase'" in line for line in unchecked)


def test_the_tool_exits_0_on_the_tree(capsys):
    assert check.main(["--root", str(_ROOT), "--unchecked"]) == 0
    assert "not checked:" in capsys.readouterr().out


@pytest.fixture(scope="module")
def sources():
    return {path: (_ROOT / path).read_text(encoding="utf-8") for path in check.sources(_ROOT)}


def _copy(sources, tmp_path, path, old, new, count=1):
    """The sources under *tmp_path*, with *old* replaced by *new* in *path*."""
    assert sources[path].count(old) == count, f"{old!r} is not {count} time(s) in {path}"
    for name, text in sources.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.replace(old, new) if name == path else text, encoding="utf-8")
    return tmp_path


_BUILDER, _BINDINGS = check.BUILDER, check.BINDINGS
# (file, its text, the text in its place, what the check then names)
_BROKEN = {
    "an opcode's value in the decoder": (
        _BINDINGS,
        "constexpr int kOpWfDrag = 5;",
        "constexpr int kOpWfDrag = 6;",
        [f"opcode WF_DRAG: {_BUILDER} says 5, {_BINDINGS} says 6"],
    ),
    "an opcode's value in the builder": (
        _BUILDER,
        "_SELECT = 41\n",
        "_SELECT = 44\n",
        [f"opcode SELECT: {_BUILDER} says 44, {_BINDINGS} says 41"],
    ),
    "an opcode the decoder does not define": (
        _BUILDER,
        "_SITE = 43\n",
        "_SITE = 43\n_MARK = 44\n",
        [f"opcode MARK: {_BUILDER} says 44, {_BINDINGS} has no kOp for it"],
    ),
    "an opcode the builder does not define": (
        _BINDINGS,
        "constexpr int kOpSite = 43;\n",
        "constexpr int kOpSite = 43;\nconstexpr int kOpMark = 44;\n",
        [
            f"opcode MARK: {_BINDINGS} says 44, {_BUILDER} has no _MARK",
            f"opcode MARK: the decoder in {_BINDINGS} has no case for it",
        ],
    ),
    "the payload length the decoder requires": (
        _BINDINGS,
        "case kOpDrive: {\n        requirePayload(7);",
        "case kOpDrive: {\n        requirePayload(8);",
        [f"opcode DRIVE: {_BUILDER}:", f" writes 7 payload words, {_BINDINGS} requires 8"],
    ),
    "the payload length the builder writes": (
        _BUILDER,
        "_header(_READOUT, 6)",
        "_header(_READOUT, 5)",
        [
            "opcode READOUT: ",
            "writes a header for 5 payload words and then 6",
            f"writes 5 payload words, {_BINDINGS} requires 6",
        ],
    ),
    "the payload length another module writes": (
        f"{check.PACKAGE}/kernel/_conditions.py",
        "_header(_SELECT, 4)",
        "_header(_SELECT, 3)",
        ["opcode SELECT: ", f"writes 3 payload words, {_BINDINGS} requires 4"],
    ),
    "a word of a record written word by word": (
        _BUILDER,
        "        self._w(highest)\n",
        "",
        ["opcode VARIABLE_WAIT: ", "writes a header for 6 payload words and then 5"],
    ),
    "a waveform slot's kind": (
        _BUILDER,
        'self._emit_wf(_WF_COSINE, rv, vals, ["i64", "f64"])',
        'self._emit_wf(_WF_COSINE, rv, vals, ["f64", "f64"])',
        ["opcode WF_COSINE, slot 1: ", f"writes f64, {_BINDINGS} reads i64"],
    ),
    "a waveform slot more than the decoder reads": (
        _BUILDER,
        'self._emit_wf(_WF_COSINE, rv, vals, ["i64", "f64"])',
        'self._emit_wf(_WF_COSINE, rv, vals, ["i64", "f64", "f64"])',
        [
            f"writes 4 payload words, {_BINDINGS} requires 3",
            "opcode WF_COSINE, slot 3: ",
            f"writes f64, {_BINDINGS} reads None",
        ],
    ),
    "a waveform slot's name": (
        _BINDINGS,
        'positive(3, "gaussian sigma")',
        'positive(3, "gaussian width")',
        [
            f"opcode WF_GAUSSIAN, slot 3: {_BUILDER} (_WF_FIELDS) names it "
            f"'gaussian sigma', {_BINDINGS} names it 'gaussian width'"
        ],
    ),
    "a waveform's fields against the op table": (
        check.IR_BUILDER,
        '"drag": (0, ("duration", "amplitude", "sigma", "beta"), ("waveform",)),',
        '"drag": (0, ("duration", "amplitude", "beta", "sigma"), ("waveform",)),',
        ["_WF_FIELDS[WF_DRAG]: ", "OP_TABLE['drag']"],
    ),
    "the slot a mask bit marks": (
        _BUILDER,
        "self._put((_header(_SET_PHASE, 3, 1 << 2), tv, rtv, pvid))",
        "self._put((_header(_SET_PHASE, 3, 1 << 1), tv, rtv, pvid))",
        ["opcode SET_PHASE: ", "marks slot 1 as a value id"],
    ),
    "where the header holds the mask": (
        _BUILDER,
        "(param_mask << 16)",
        "(param_mask << 24)",
        [f"header param_mask: {_BUILDER} writes it at bit 24, {_BINDINGS} reads it at bit 16"],
    ),
    "where the decoder reads the length": (
        _BINDINGS,
        "int64_t plen = (hdr >> 8) & 0xFF;",
        "int64_t plen = (hdr >> 9) & 0xFF;",
        ["header payload_len: ", "reads it at bit 9", "overlap or leave a gap"],
    ),
    "the extended length": (
        _BUILDER,
        "_MAX_PAYLOAD = 0xFF\n",
        "_MAX_PAYLOAD = 0xFE\n",
        ["header extended length: ", "(_MAX_PAYLOAD) says 254", "(kExtendedPayload) says 255"],
    ),
    "a numeric type": (
        _BUILDER,
        '_NUMERIC_TYPE = {"i64": 0, "f64": 1}',
        '_NUMERIC_TYPE = {"i64": 1, "f64": 0}',
        ["_NUMERIC_TYPE['i64']: ", "_NUMERIC_TYPE['f64']: "],
    ),
    "a binary operation": (
        _BUILDER,
        '"floordiv": 4,\n    "mod": 5,',
        '"floordiv": 5,\n    "mod": 4,',
        ["_BINARY_OP['floordiv']: ", "builds arith.RemSI", "_BINARY_OP['mod']: "],
    ),
    "a compare predicate": (
        _BUILDER,
        '"<": 2, "<=": 3,',
        '"<": 3, "<=": 2,',
        ["_COMPARE['<']: ", "has sle, not slt", "has OLE, not OLT"],
    ),
    "the decoder's predicates": (
        _BINDINGS,
        "mlir::arith::CmpIPredicate::sgt, mlir::arith::CmpIPredicate::sge}",
        "mlir::arith::CmpIPredicate::sge, mlir::arith::CmpIPredicate::sgt}",
        ["_COMPARE['>']: ", "has sge, not sgt"],
    ),
    "a schedule-block policy": (
        _BUILDER,
        '_BLOCK_POLICY = {"asap": 0, "alap": 1}',
        '_BLOCK_POLICY = {"asap": 1, "alap": 0}',
        ["_BLOCK_POLICY['asap']: ", "(kOpBlockBegin) says 0"],
    ),
    "a line type": (
        _BUILDER,
        '_LINE_TYPE_CODES = {"drive_line": 0, "readout_line": 1}',
        '_LINE_TYPE_CODES = {"drive_line": 1, "readout_line": 0}',
        ["_LINE_TYPE_CODES['readout_line']: ", "(kOpSync) says 1"],
    ),
    "the unscheduled mark": (
        _BUILDER,
        "_UNSCHEDULED = -1\n",
        "_UNSCHEDULED = -2\n",
        [f"_UNSCHEDULED: {_BUILDER} says -2, {_BINDINGS} (kOpDrive) says -1"],
    ),
    "where a drive's times are": (
        _BUILDER,
        "(_DRIVE_HEADER, lv, wv, tv, rlv, rtv, _UNSCHEDULED, _UNSCHEDULED)",
        "(_DRIVE_HEADER, lv, wv, tv, rlv, _UNSCHEDULED, _UNSCHEDULED, rtv)",
        ["opcode DRIVE: ", "in payload words [4, 5]", "from words [5, 6]"],
    ),
    "a site kind": (
        check.IDLE,
        '_SITE_KINDS = {"d": 0, "r": 1, "w": 2, "s": 3, "f": 4}',
        '_SITE_KINDS = {"d": 0, "r": 1, "w": 3, "s": 2, "f": 4}',
        ["_SITE_KINDS['s']: ", "_SITE_KINDS['w']: ", "(kOpSite) says 3"],
    ),
    "an amplitude operand of the export": (
        check.CONTROLLER,
        '"drag": (1,),',
        '"drag": (2,),',
        ["_AMPLITUDES['drag']: ", "says (2,)", "(parameter_roles) says (1,)"],
    ),
    "an amplitude operand of the native module": (
        _BINDINGS,
        '{"pulse.cosine", {1}}',
        '{"pulse.cosine", {0}}',
        ["_AMPLITUDES['cosine']: ", "(parameter_roles) says (0,)"],
    ),
    "an operand name of the export": (
        check.CONTROLLER,
        '"gaussian_square": ("duration", "amplitude", "sigma", "risefall"),',
        '"gaussian_square": ("duration", "amplitude", "sigma", "width"),',
        ["_PARAMETERS['gaussian_square']: ", "'risefall'"],
    ),
    "an operand of the dialect": (
        check.DIALECT,
        "    AnyFloat:$sigma,\n    AnyFloat:$beta\n",
        "    AnyFloat:$beta,\n    AnyFloat:$sigma\n",
        ["_PARAMETERS['drag']: ", "the operands of pulse.drag"],
    ),
    "a time parameter": (
        check.CLOCKS,
        '"gaussian_square": ("duration", "sigma", "risefall"),',
        '"gaussian_square": ("duration", "sigma", "width"),',
        ["_TIMES_OF['gaussian_square']: ", "['width']"],
    ),
    "an opcode written from outside the table": (
        _BUILDER,
        '_NUMERIC_TYPE = {"i64": 0, "f64": 1}\n',
        '_NUMERIC_TYPE = {"i64": 0, "f64": 1}\n_MARK = 44\n_MARKED = _header(_MARK, 1)\n',
        ["opcode MARK: ", "is not among the opcodes below the '# OpCodes' comment"],
    ),
    "a record the decoder no longer requires a length of": (
        _BINDINGS,
        "case kOpDrive: {\n        requirePayload(7);",
        'case kOpDrive: {\n        if (plen != 8) throw std::invalid_argument("packed");',
        ["opcode DRIVE: ", f"writes 7 payload words, {_BINDINGS} has no requirePayload for it"],
    ),
    "the slot the waveform helper marks": (
        _BUILDER,
        "param_mask |= 1 << (i + 1)",
        "param_mask |= 1 << i",
        [f"{_BUILDER}: _emit_wf no longer has `param_mask |= 1 << i + 1`"],
    ),
    "the length the record helper writes": (
        _BUILDER,
        "self._put([_header(opcode, n), *payload])",
        "self._put([_header(opcode, n + 1), *payload])",
        [f"{_BUILDER}: _record no longer has `self._put([_header(opcode, n), *payload])`"],
    ),
    "a numeric type of a negation": (
        _BINDINGS,
        "auto operand = look(p[2]);\n        if (p[1] == 0) {",
        "auto operand = look(p[2]);\n        if (p[1] == 2) {",
        ["_NUMERIC_TYPE['i64']: ", "(kOpNumNeg) says 2"],
    ),
    "a numeric type of a binary operation": (
        _BINDINGS,
        "mlir::Value result;\n        if (p[1] == 0) {\n          switch (p[2]) {",
        "mlir::Value result;\n        if (p[1] == 7) {\n          switch (p[2]) {",
        ["_NUMERIC_TYPE['i64']: ", "(kOpNumBinary) says 7"],
    ),
    "the numeric types of a cast": (
        _BINDINGS,
        "if (p[1] == 0 && p[2] == 1)\n          bind(p[0],\n               mlir::arith::SIToFPOp",
        "if (p[1] == 1 && p[2] == 0)\n          bind(p[0],\n               mlir::arith::SIToFPOp",
        ["_NUMERIC_TYPE['i64']: ", "(kOpNumCast, SIToFP) says 1"],
    ),
    "the type word of a compare": (
        _BINDINGS,
        "} else if (p[2] == 1) {\n          static const mlir::arith::CmpFPredicate",
        "} else if (p[2] == 0) {\n          static const mlir::arith::CmpFPredicate",
        ["_NUMERIC_TYPE['f64']: ", "(kOpClassicalCompare) says 0"],
    ),
    "where a compare's and/or begin": (
        _BINDINGS,
        "        if (p[1] > 5) {\n",
        "        if (p[1] > 4) {\n",
        ["_COMPARE: ", "reads the codes above 4 as and/or"],
    ),
    "the mask's bits": (
        _BINDINGS,
        "int pmask = static_cast<int>((hdr >> 16) & 0xFFFF);",
        "int pmask = static_cast<int>((hdr >> 16) & 0x7);",
        ["opcode WF_DRAG: ", "marks a slot beyond the 3 bits of the mask"],
    ),
    "a header mask by name": (
        _BINDINGS,
        "int opcode = static_cast<int>(hdr & 0xFF);",
        "int opcode = static_cast<int>(hdr & kOpcodeMask);",
        [f"{_BINDINGS}: the header's opcode mask is kOpcodeMask, not a number"],
    ),
    "a drive's times read another way": (
        _BINDINGS,
        "p[5] != -1 ? std::optional<int64_t>(p[5]) : std::nullopt,",
        "p[5] >= 0 ? std::optional<int64_t>(p[5]) : std::nullopt,",
        [f"{_BINDINGS}: cannot read the unscheduled mark of kOpDrive"],
    ),
    "a site kind tested for another op": (
        _BINDINGS,
        "mlir::isa<pulse::SyncOp>(op)    ? 3",
        "mlir::isa<pulse::SetPhaseOp>(op) ? 3",
        [f"{_BINDINGS}: cannot read the kinds of kOpSite"],
    ),
    "a table entry that is not a tuple": (
        check.CONTROLLER,
        '"drag": (1,),',
        '"drag": (1),',
        [f"{check.CONTROLLER} or {check.CLOCKS}: _PARAMETERS, _AMPLITUDES or _TIMES_OF is not"],
    ),
    "a float operation the decoder no longer reads": (
        _BINDINGS,
        "          case 3:\n"
        "            result = mlir::arith::DivFOp::create(*builder_, loc, lhs, rhs);\n"
        "            break;\n",
        "",
        ["_BINARY_OP['div']: ", "(kOpNumBinary, f64) does not read"],
    ),
    "the length at which the record helper extends": (
        _BUILDER,
        "        if n >= _MAX_PAYLOAD:\n            self._put([_header(opcode, _MAX_PAYLOAD)",
        "        if n > _MAX_PAYLOAD:\n            self._put([_header(opcode, _MAX_PAYLOAD)",
        [f"{_BUILDER}: _record no longer has `if n >= _MAX_PAYLOAD:`"],
    ),
    "a word more in the waveform helper": (
        _BUILDER,
        "        self._put([_header(wf_code, payload_len, param_mask), rv, *encoded])\n",
        "        self._put([_header(wf_code, payload_len, param_mask), rv, *encoded])\n"
        "        self._put([0])\n",
        [f"{_BUILDER}: _emit_wf no longer writes its record with 1 self._put"],
    ),
    "the builder's opcodes moved": (
        _BUILDER,
        "# OpCodes -- must match",
        "# The codes -- must match",
        [f"{_BUILDER}: no '# OpCodes' comment above the opcodes"],
    ),
    "the decoder's cases rewritten": (
        _BINDINGS,
        "      switch (opcode) {\n",
        "      switch (code) {\n",
        [f"{_BINDINGS}: cannot find the decoder's switch (opcode)"],
    ),
}


@pytest.mark.parametrize("change", _BROKEN)
def test_a_disagreement_is_named(change, sources, tmp_path, capsys):
    path, old, new, named = _BROKEN[change]
    root = _copy(sources, tmp_path, path, old, new)
    assert check.main(["--root", str(root)]) == 1
    text = capsys.readouterr().err
    for expected in named:
        assert expected in text, text
    assert "in the packed schema" in text


# Changes that keep the schema: the check still passes.
_HARMLESS = {
    "nothing": (_BUILDER, "_SITE = 43\n", "_SITE = 43\n"),
    "a comment that names another length": (
        _BINDINGS,
        "        // [the controller's longest program, in cycles]\n",
        "        // [the longest program]; a site record is requirePayload(2)\n",
    ),
    "a drive's times wrapped onto another line": (
        _BINDINGS,
        "p[5] != -1 ? std::optional<int64_t>(p[5]) : std::nullopt,",
        "p[5] != -1\n                ? std::optional<int64_t>(p[5])\n"
        "                : std::nullopt,",
    ),
    "a block comment that names another length": (
        _BINDINGS,
        "        // [the controller's longest program, in cycles]\n",
        "        /* was requirePayload(2) */\n",
    ),
    "a slot name with two slashes": (
        _BINDINGS,
        'finite(2, "square amplitude\'s imaginary part")'.replace("2", "3"),
        'finite(3, "square amplitude\'s imaginary // part")',
    ),
    "an opcode with its type": (_BUILDER, "_WF_DRAG = 5\n", "_WF_DRAG: int = 5\n"),
    "amplitude operands in another order": (
        check.CONTROLLER,
        '"square": (1, 2),',
        '"square": (2, 1),',
    ),
}


@pytest.mark.parametrize("change", _HARMLESS)
def test_a_change_that_keeps_the_schema_passes(change, sources, tmp_path):
    root = _copy(sources, tmp_path, *_HARMLESS[change])
    problems, unchecked = check.check(root)
    assert problems == []
    assert len(unchecked) == len(check.check(_ROOT)[1])


def test_a_header_whose_words_are_not_counted_is_listed(sources, tmp_path):
    root = _copy(
        sources,
        tmp_path,
        _BUILDER,
        "self._put((_header(_READOUT, 6), lv, wv, tv, rlv, rtv, mv))",
        "head = _header(_READOUT, 6)\n            self._put((head, lv, wv, tv, rlv, rtv))",
    )
    problems, unchecked = check.check(root)
    assert problems == []
    assert any("the words written after the header of READOUT" in line for line in unchecked)
