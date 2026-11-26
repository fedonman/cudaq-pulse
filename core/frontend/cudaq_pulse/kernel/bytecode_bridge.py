# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bytecode-based kernel capture.

Uses ``fn.__code__`` directly instead of ``inspect.getsource`` + ``ast.parse``.
All version-specific bytecode handling is in ``_bytecode_normalize.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .._device import Qudit, Register
from ._bytecode_normalize import DecodedCode, decode
from ._emitter import PulseIREmitter, _extract_closures, add_source_note
from ._unroll import resolve_unroll_threshold
from .ir_builder import CompilationError, PythonIRBuilder


def compile_kernel_bytecode(
    fn: Callable[..., Any], *, unroll_threshold: int | None = None
) -> Callable[..., PythonIRBuilder]:
    """Compile a pulse kernel function via bytecode analysis.

    Returns a callable that, given concrete arguments, produces a
    populated ``PythonIRBuilder``. Captured variables are read on every call.
    *unroll_threshold* is that of ``cudaq_pulse.compile``.
    """
    code = fn.__code__
    prog = decode(code)
    params = code.co_varnames[: code.co_argcount]
    unroll = resolve_unroll_threshold(unroll_threshold)

    def _emit(*args: Any, **kwargs: Any) -> PythonIRBuilder:
        if kwargs:
            raise CompilationError(
                "keyword arguments not supported in pulse kernels", code="wrong_argument"
            )
        builder = PythonIRBuilder(name=fn.__name__)
        builder.unroll_threshold = unroll
        _trace(fn, builder, args, "pulse.qudit_arg", prog, params)
        return builder

    return _emit


# Re-trace *fn* using *builder* (any object with an ``emit()`` method).
#
# *args* gives every parameter, positional then keyword-only, in signature
# order. Returns the kernel's traced return value. *emitter_type*, when
# given, is the tracer, else the one *builder*'s sample rates take.
def _trace_kernel_with_builder(
    fn: Callable[..., Any],
    builder: Any,
    args: Sequence[Any],
    emitter_type: type[PulseIREmitter] | None = None,
) -> Any:
    code = fn.__code__
    names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    return _trace(fn, builder, args, "pulse.qudit_alloc", decode(code), names, emitter_type)


def _trace(
    fn: Any,
    builder: Any,
    args: Sequence[Any],
    qudit_op: str,
    prog: DecodedCode,
    params: tuple[str, ...],
    emitter_type: type[PulseIREmitter] | None = None,
) -> Any:
    if len(args) != len(params):
        raise CompilationError(
            f"{fn.__name__}: expected {len(params)} args, got {len(args)}",
            code="wrong_argument",
        )

    closures = _extract_closures(fn) if fn.__closure__ else {}
    if emitter_type is not None:
        emitter = emitter_type(builder, fn.__globals__, closures)
    elif builder.tracer is None:
        emitter = PulseIREmitter(builder, fn.__globals__, closures)
    else:
        emitter = builder.tracer(builder, fn.__globals__, closures)
    local_vars = emitter.locals
    for name, val in zip(params, args, strict=False):
        if type(val) is Qudit:
            (ir_val,) = builder.emit(qudit_op, (), ("qref",), {"index": val.index})
            local_vars[name] = ir_val
        elif type(val) is Register:
            local_vars[name] = [
                builder.emit(qudit_op, (), ("qref",), {"index": q.index})[0] for q in val
            ]
        else:
            local_vars[name] = val

    try:
        return emitter.trace(prog)
    except Exception as e:
        add_source_note(e)
        raise
