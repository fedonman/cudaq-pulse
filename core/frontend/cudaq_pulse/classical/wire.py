# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wire format of a call: what a request and an answer look like.

A request is a 24-byte header (magic ``'CUQR'``, dispatch key, argument
length, request id, timestamp) followed by the arguments, tightly packed and
little-endian in the order of the function's argument types; an answer is a
24-byte header (magic ``'CUQS'``, status, result length, request id,
timestamp) followed by the results. This is the layout of the CUDA-Q Realtime
messaging protocol, declared here so cudaq-pulse needs nothing from CUDA-Q;
``core/mlir/include/cudaq-pulse/Runtime/WireFormat.h`` declares the same for
C++.

``native(library, symbol, function)`` wraps a host handler written in C with
the signature ``void handler(const void *request, void *answer, size_t
slot_bytes)``, the shape of a CUDA-Q Realtime host-call handler, as a Python
handler for simulation.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from .._errors import ExecutionError, InvalidTypeError, InvalidValueError
from ..compiler import _dispatch_keys, _MachineModel
from . import (
    ClassicalFunction,
    ValueType,
    _LazyTypes,
    _resolve,
    _unsized_names,
    _wrong_length,
)

if TYPE_CHECKING:
    from .. import _clocks
    from .._classical_report import ClassicalCall

__all__ = [
    "HEADER_BYTES",
    "REQUEST_MAGIC",
    "RESPONSE_MAGIC",
    "decode_answer",
    "decode_request",
    "encode_answer",
    "encode_request",
    "native",
    "schema",
]

#: The size of a request or response header.
HEADER_BYTES = 24
#: The first word of a request header, ``'CUQR'``.
REQUEST_MAGIC = 0x43555152
#: The first word of a response header, ``'CUQS'``.
RESPONSE_MAGIC = 0x43555153
_REQUEST = struct.Struct("<IIIIQ")
_RESPONSE = struct.Struct("<IiIIQ")

# Payload type ids, and the element format of each argument descriptor.
_TYPES = {
    "u8": (0x10, "B"),
    "i32": (0x11, "i"),
    "i64": (0x12, "q"),
    "f32": (0x13, "f"),
    "f64": (0x14, "d"),
}
_ARRAYS = {"u8": (0x20, "B"), "i32": (0x21, "i"), "f32": (0x22, "f"), "f64": (0x23, "d")}
_BIT_PACKED = 0x30


def _layout(spec: ValueType) -> tuple[int, int, int, str]:
    """(type id, bytes, wire elements, element format) of one type."""
    d = spec.descriptor
    if d.startswith("bits:"):
        return _BIT_PACKED, spec.wire_bytes, spec.count, ""
    if d.startswith("levels:"):
        return _ARRAYS["u8"][0], spec.count, spec.count, "B"
    if d.startswith("iq:"):
        return _ARRAYS["f32"][0], 8 * spec.count, 2 * spec.count, "f"
    if "[" in d:
        element = d[: d.index("[")]
        type_id, fmt = _ARRAYS[element]
        return type_id, spec.wire_bytes, spec.count, fmt
    type_id, fmt = _TYPES[d]
    return type_id, spec.wire_bytes, 1, fmt


def schema(function: ClassicalFunction[..., Any]) -> dict[str, Any]:
    """The function's handler schema: per argument and result its payload
    type id, size in bytes and element count, as a dispatcher function table
    entry declares them. A function whose list lengths come from its calls
    has a schema per version: pass ``fn.specialize(...)``."""
    if function._shape is not None or function.args.__class__ is _LazyTypes:  # type: ignore[comparison-overlap]
        _one_schema(function, "wire.schema")

    def describe(specs: Sequence[ValueType]) -> list[dict[str, int]]:
        out = []
        for spec in specs:
            type_id, size, elements, _ = _layout(spec)
            out.append({"type_id": type_id, "size_bytes": size, "num_elements": elements})
        return out

    return {
        "args": describe(function.args),
        "results": describe(function.results),
    }


def _one_schema(function: ClassicalFunction[..., Any], what: str) -> None:
    _resolve(function)
    if function._shape is not None:
        names = _unsized_names(function)
        example = ", ".join(f"{n}=..." for n in names)
        raise InvalidTypeError(
            f"{function.name}'s list lengths come from its calls ({', '.join(names)}), so it has"
            f" a schema per version; for one version use {what}("
            f"{function.name}.specialize({example}))"
        )


def _pack(
    specs: Sequence[ValueType],
    values: Sequence[Any],
    what: str,
    answering: ClassicalFunction[..., Any] | None = None,
) -> bytes:
    if len(values) != len(specs):
        raise InvalidValueError(f"{len(specs)} {what} expected, got {len(values)}")
    out = bytearray()
    for spec, value in zip(specs, values, strict=False):
        _, size, _, fmt = _layout(spec)
        if spec.is_scalar:
            items = [value]
        else:
            items = list(value)
            if answering is not None and len(items) != spec.count:
                position = [s is spec for s in specs].index(True)
                raise InvalidValueError(_wrong_length(answering, position, len(items)))
        if spec.descriptor.startswith("bits:"):
            packed = bytearray(size)
            for i, bit in enumerate(items):
                if bit:
                    packed[i // 8] |= 1 << (i % 8)
            out += packed
        else:
            if spec.descriptor.startswith("iq:") and len(items) == spec.count:
                items = [c for point in items for c in _iq_pair(point)]
            out += struct.pack("<" + fmt * len(items), *items)
    return bytes(out)


def _iq_pair(point: Any) -> tuple[float, float]:
    """An IQ point's (I, Q): a complex number, or an (i, q) pair."""
    if isinstance(point, complex):
        return (point.real, point.imag)
    i, q = point
    return (i, q)


def _unpack(specs: Sequence[ValueType], data: bytes) -> list[Any]:
    values: list[Any] = []
    cursor = 0
    for spec in specs:
        _, size, elements, fmt = _layout(spec)
        chunk = data[cursor : cursor + size]
        cursor += size
        if len(chunk) != size:
            raise ExecutionError("payload shorter than its schema")
        if spec.descriptor.startswith("bits:"):
            items: list[Any] = [(chunk[i // 8] >> (i % 8)) & 1 for i in range(spec.count)]
        else:
            items = list(struct.unpack("<" + fmt * elements, chunk))
            if spec.descriptor.startswith("iq:"):
                items = [complex(i, q) for i, q in zip(items[0::2], items[1::2], strict=True)]
        values.append(items[0] if spec.is_scalar else items)
    return values


def encode_request(
    function: ClassicalFunction[..., Any],
    args: Sequence[Any],
    request_id: int = 0,
    timestamp: int = 0,
    dispatch_key: int | None = None,
) -> bytes:
    """A request to *function* with *args* (as a handler receives them),
    sent to *dispatch_key*: the program's (``call.dispatch_key``), by default
    the hash of the function's compiled name."""
    payload = _pack(function.args, args, "arguments")
    key = function._dispatch_key if dispatch_key is None else dispatch_key
    header = _REQUEST.pack(REQUEST_MAGIC, key, len(payload), request_id, timestamp)
    return header + payload


def decode_request(
    function: ClassicalFunction[..., Any], data: bytes, dispatch_key: int | None = None
) -> tuple[int, int, list[Any]]:
    """``(request id, timestamp, arguments)`` of a request to *function* at
    *dispatch_key* (as for ``encode_request``)."""
    magic, key, length, request_id, timestamp = _REQUEST.unpack_from(data)
    if magic != REQUEST_MAGIC:
        raise ExecutionError(f"not a request: magic {magic:#x}")
    expected = function._dispatch_key if dispatch_key is None else dispatch_key
    if key != expected:
        raise ExecutionError(
            f"request for dispatch key {key:#x}, not {function.name}'s ({expected:#x})"
        )
    return request_id, timestamp, _unpack(function.args, data[HEADER_BYTES : HEADER_BYTES + length])


def encode_answer(
    function: ClassicalFunction[..., Any],
    answer: Any,
    request_id: int = 0,
    timestamp: int = 0,
    status: int = 0,
) -> bytes:
    """The answer of *function* (as a handler returns it) to one request."""
    results = function.results
    values = list(answer) if len(results) > 1 else ([answer] if results else [])
    payload = _pack(results, values, "results", function)
    return _RESPONSE.pack(RESPONSE_MAGIC, status, len(payload), request_id, timestamp) + payload


def _status_answer(request_id: int, timestamp: int, status: int) -> bytes:
    """An answer with *status* and no results, as a server answers a request
    it could not serve."""
    return _RESPONSE.pack(RESPONSE_MAGIC, status, 0, request_id, timestamp)


def call_failed(name: str, request_id: int, status: int) -> ExecutionError:
    """The error for an answer whose status is not 0."""
    return ExecutionError(
        f"classical function {name!r} failed on its host: request {request_id} was answered "
        f"with status {status} (0 means success); see the handler's log"
    )


def decode_answer(function: ClassicalFunction[..., Any], data: bytes) -> tuple[int, int, Any]:
    """``(status, request id, answer)`` of an answer from *function*."""
    magic, status, length, request_id, _ = _RESPONSE.unpack_from(data)
    if magic != RESPONSE_MAGIC:
        raise ExecutionError(f"not an answer: magic {magic:#x}")
    results = function.results
    if status != 0 or not results:
        return status, request_id, None
    values = _unpack(results, data[HEADER_BYTES : HEADER_BYTES + length])
    return status, request_id, values[0] if len(values) == 1 else tuple(values)


class _Native:
    """A C host handler, called through its request and answer slots."""

    def __init__(
        self, handler: Any, function: ClassicalFunction[..., Any], slot_bytes: int
    ) -> None:
        import ctypes

        self._handler = handler
        self._function = function
        self._slot_bytes = slot_bytes
        self._request = ctypes.create_string_buffer(slot_bytes)
        self._answer = ctypes.create_string_buffer(slot_bytes)
        self._memmove = ctypes.memmove
        self._next_id = 0

    def __call__(self, *args: Any) -> Any:
        request = encode_request(self._function, args, self._next_id)
        if len(request) > self._slot_bytes:
            raise ExecutionError(
                f"{self._function.name}: request of {len(request)} bytes exceeds the slot"
            )
        self._memmove(self._request, request, len(request))
        self._handler(self._request, self._answer, self._slot_bytes)
        status, request_id, answer = decode_answer(self._function, self._answer.raw)
        if request_id != self._next_id:
            raise ExecutionError(
                f"{self._function.name}: answer to request {request_id}, expected {self._next_id}"
            )
        self._next_id = (self._next_id + 1) & 0xFFFFFFFF
        if status != 0:
            raise call_failed(self._function.name, request_id, status)
        return answer


def native(
    library: str, symbol: str, function: ClassicalFunction[..., Any], slot_bytes: int = 1024
) -> Any:
    """A handler for *function* that calls the C function *symbol* in the
    shared library *library* with a request and an answer slot."""
    import ctypes

    _one_schema(function, "wire.native")
    handler = getattr(ctypes.CDLL(library), symbol)
    handler.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    handler.restype = None
    return _Native(handler, function, slot_bytes)


# The emulator export and its manifest, which cudaq_pulse.classical.emulation
# exposes: they live here, where an export and an emulation both look, so an
# export loads none of the emulator.


def _first_report(lowered: Any) -> list[ClassicalCall]:
    # The report's dataclass is built by the first export of a program with
    # calls; later exports call the report directly.
    global _report
    from .._classical_report import report

    _report = report
    return report(lowered)


_report: Callable[[Any], list[ClassicalCall]] = _first_report


_EMULATION_FORMAT = "cudaq-pulse.emulation/1"
_MANIFEST_FORMAT = "cudaq-pulse.call-manifest/1"


def _manifest(lowered: Any) -> dict[str, Any]:
    """The classical functions a compiled kernel calls, for a dispatcher."""
    machine = lowered.options.machine
    functions = []
    for fn in lowered.classical:
        link = machine.link(fn.link)
        link_id = machine.link_id(fn.link)
        key, fixed_by = _dispatch_keys(lowered)[fn.name]
        entry = {"name": fn.name, "dispatch_key": key}
        if fixed_by is not None:
            entry["dispatch_key_fixed_by"] = fixed_by
        functions.append(
            {
                **entry,
                "random": fn.uniform,
                "link": link_id,
                "link_name": machine.link_name(link_id),
                "schema": schema(fn),
                "argument_types": [a.descriptor for a in fn.args],
                "result_types": [r.descriptor for r in fn.results],
                "latency_ns": {
                    "min": fn.min_latency_ns,
                    "nominal": fn.nominal_latency_ns,
                    "max": fn.max_latency_ns,
                },
                "latency_mode": fn.latency_mode,
                "timeout_ns": fn.timeout_ns,
                "deadline_ns": fn.deadline_ns,
                "submit": fn.submit_policy,
                "request_bytes": HEADER_BYTES + sum(a.wire_bytes for a in fn.args),
                "answer_bytes": HEADER_BYTES + sum(r.wire_bytes for r in fn.results),
                "slot_bytes": link.slot_bytes,
                "slots": link.slots,
            }
        )
    return {"format": _MANIFEST_FORMAT, "functions": functions}


def _emulation_export(lowered: Any, controller: int | None = None) -> dict[str, Any]:
    """A compiled kernel for a controller emulator: schedule, segments, calls,
    links and manifest, as JSON-serializable data. Times are in clock cycles
    of the controller, at ``clock_ghz`` (with ``program_clock_ghz`` when that
    is not the program clock, as in ``cudaq_pulse.controller``); each event's
    ``segment_start_cycles`` is the static start of its segment, which moves at
    run time by the lengths of the run-time waits before it on its lines. An
    segment's shift adds to that of the segment it follows, the one before it
    unless it names another in ``after``. On a control system of several
    controllers, the part of the one *controller* indexes, with only its
    events, calls, links and functions, and ``controller`` its index."""
    machine = lowered.options.machine
    if machine._system is not None and machine.controller is None:
        return machine._emulation_part(lowered, controller)  # type: ignore[no-any-return]
    if controller is not None:
        from ..controller import _controller_index

        _controller_index(controller, 1)
    # The model's: that of a control system with a broadcast drive line
    # writes the line's qudits on its drives (_broadcast_lines).
    return machine._emulation(lowered)  # type: ignore[no-any-return]


_Part = tuple[int, "_clocks.ControllerClock | None", frozenset[str], Sequence[bool]]


def _export_part(lowered: Any, controller: int | None) -> dict[str, Any]:
    """``export`` on a control system of several controllers: the part of
    one (as ``cudaq_pulse.controller.export`` chooses it), in its cycles."""
    from ..controller import _part

    _, events, _, parts, index, _ = _part(lowered, controller)
    functions = frozenset(
        event["function"]
        for event, part in zip(events, parts, strict=True)
        if index in part and event["kind"] in ("submit", "answer")
    )
    # The schedule holds the controller events but their syncs, in order.
    takes_part = [
        index in part for event, part in zip(events, parts, strict=True) if event["kind"] != "sync"
    ]
    clocks = (lowered._generic or lowered).clocks
    own = None if clocks is None else clocks.controllers[index]
    return _export(lowered, True, (index, own, functions, takes_part))


def _export(
    lowered: Any, with_events: bool = True, part: _Part | None = None, program: Any = None
) -> dict[str, Any]:
    """``export``; without *with_events*, its ``events`` are left empty. With
    *part* (a controller's index and clock, the functions it calls or takes
    answers of, and whether it takes part in each scheduled event), only
    that controller's part, in its cycles. *program* is the program's
    ``_segments()`` and its calls (``_program``), when the caller has them."""
    segments, after = lowered._segments() if program is None else program[0]
    starts = [e.start_cycles for e in segments]
    takes = None
    if with_events:
        segment_of = lowered._event_segments(syncs=False)
        # Read once: each read of ``schedule`` builds it anew.
        schedule = lowered.schedule
        if part is not None:
            takes = part[3]
            assert len(takes) == len(schedule)
        events = [
            {
                "kind": event.kind,
                "line": event.line,
                "link": event.link,
                "qudit": event.qudit,
                "start_cycles": event.start_cycles,
                "duration_cycles": event.duration_cycles,
                "segment": (
                    index := segment_of[position]
                    if segment_of and event.start_cycles is not None
                    else 0
                ),
                "segment_start_cycles": starts[index],
            }
            for position, event in enumerate(schedule)
            if takes is None or takes[position]
        ]
    else:
        events = []
    machine = lowered.options.machine
    data = {
        "format": _EMULATION_FORMAT,
        "clock_ghz": lowered.quantum_system.clock_ghz,
        "links": {
            str(machine.link_id(fn.link)): {
                **machine.link(fn.link).cost(),
                "name": machine.link_name(machine.link_id(fn.link)),
            }
            for fn in lowered.classical
            if part is None or fn.name in part[2]
        },
        "segments": [
            {**asdict(e), "after": a} if k and a != k - 1 else asdict(e)
            for k, (e, a) in enumerate(zip(segments, after, strict=False))
        ],
        "events": events,
        "calls": [
            _call_data(c)
            for c in (_calls(lowered) if program is None else program[1])
            if part is None or (c.compiled_name in part[2] and c.controller in (None, part[0]))
        ],
        "manifest": _manifest(lowered),
    }
    if machine._plain:
        return data
    if part is not None:
        data["manifest"]["functions"] = [
            fn for fn in data["manifest"]["functions"] if fn["name"] in part[2]
        ]
        data["controller"] = part[0]
    clocks = (lowered._generic or lowered).clocks
    if clocks is None:
        return data
    own = clocks.controllers[0] if part is None or part[1] is None else part[1]
    return _in_own_clock(data, events, clocks, own)


def _program(lowered: Any) -> tuple[Any, Sequence[Any]]:
    """What ``_export`` reads of the whole program, whichever part it
    exports: its segments with the segment each follows (``_segments``), and
    its calls. ``_export`` only reads them."""
    return lowered._segments(), _calls(lowered)


def _calls(lowered: Any) -> Sequence[Any]:
    return _report(lowered) if lowered.classical else ()


def _call_data(call: Any) -> dict[str, Any]:
    """A ``ClassicalCall`` as export data: its fields, in order (the
    ``asdict`` of one, made directly); ``controller`` only for a call over a
    per-controller link. An emulator serves the compiled name:
    ``"function"`` names it, as the manifest's ``"name"`` does, and the
    manifest gives its dispatch key."""
    data = {
        "function": call.compiled_name,
        "link": call.link,
        "iteration": list(call.iteration),
        "submit_cycles": call.submit_cycles,
        "submit_duration_cycles": call.submit_duration_cycles,
        "due_cycles": call.due_cycles,
        "taken_cycles": call.taken_cycles,
        "first_use_cycles": call.first_use_cycles,
        "deadline_cycles": call.deadline_cycles,
        "segment": call.segment,
        "link_name": call.link_name,
        "exposed_cycles": call.exposed_cycles,
    }
    if call.controller is not None:
        data["controller"] = call.controller
    return data


def _in_own_clock(data: dict[str, Any], events: list[Any], clocks: Any, own: Any) -> dict[str, Any]:
    """*data*, an export of a program on *clocks*, in the clock *own* of its
    controller."""
    from .. import _clocks

    if own.ratio != 1:
        _clocks.to_own(events, _clocks.EMULATION_EVENT_TIMES, own)
        _clocks.to_own(data["segments"], _clocks.SEGMENT_TIMES, own, _clocks.SEGMENT_LOWER_BOUNDS)
        _clocks.to_own(data["calls"], _clocks.CALL_TIMES, own)
        data["clock_ghz"] = own.clock_ghz
    return {
        "format": _EMULATION_FORMAT,
        "clock_ghz": data["clock_ghz"],
        **_clocks.clock_keys(clocks, own),
        **data,
    }


# The export of a program on one controller or none is the model's own, so
# that no export asks whether its control system has a broadcast line.
_MachineModel._emulation = staticmethod(_export)
_MachineModel._emulation_part = staticmethod(_export_part)
