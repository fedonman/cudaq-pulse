# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emulating a controller that calls classical services, in virtual time.

``export(lowered)`` writes what a controller needs to run a compiled kernel:
its static schedule with each event's segment, the segment boundaries, its calls,
and the **manifest** of the classical functions it calls (``manifest``): per
function its dispatch key, link, handler schema and latency budget, which is
what a CUDA-Q Realtime dispatcher's function table holds.

``emulate(lowered, ...)`` runs the program shot by shot in virtual time,
without physics: measurement outcomes come from a source, each call is sent
as a real request (``wire``) over a channel to a service, and the answer
comes back after the modelled transport delay plus the service's measured
time. Its budget is what the compiled program allows: it is in time if it
arrives the link's feedback time before it is due. A late answer, or a
decoder's NOT_READY, takes the timeout arm of a wait with a timeout; a late
answer is recorded as late otherwise. Answers are bound to the shot as in
simulation, so variable waits get their real lengths and later segments shift
accordingly.
"""

from __future__ import annotations

import math
import struct
import sys
import threading
import time
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._errors import (
    ExecutionError,
    InvalidValueError,
    NotFoundError,
    UnsupportedError,
    _CallTimeoutError,
    _positive_int,
)
from ..compiler import _dispatch_keys
from ..runtime._calls import IQModel, _answer_values, _call_inputs, takes_iq
from . import (
    _NOT_READY,
    _RESERVED_KEYS,
    ClassicalFunction,
    _handlers_for,
    _NotReadyError,
    wire,
)

if TYPE_CHECKING:
    import numpy as np

    from .wire import _EMULATION_FORMAT as FORMAT  # noqa: F401
    from .wire import _MANIFEST_FORMAT as MANIFEST_FORMAT  # noqa: F401
    from .wire import _emulation_export as export
    from .wire import _export_part as export_part  # noqa: F401
    from .wire import _manifest as manifest

__all__ = [
    "CallTrace",
    "RealtimeChannel",
    "device_call_text",
    "EmulationResult",
    "InProcess",
    "Transport",
    "UdpChannel",
    "emulate",
    "export",
    "manifest",
    "serve_udp",
]

# ── Transport and channels ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Transport:
    """The delay of one direction of a link: ``fixed_ns + bytes /
    bytes_per_ns`` plus a jitter drawn from an exponential with mean
    ``jitter_ns`` (0 for none)."""

    fixed_ns: float = 0.0
    bytes_per_ns: float = 0.0
    jitter_ns: float = 0.0

    def delay_ns(self, byte_count: int, rng: np.random.Generator) -> float:
        """The delay in ns of *byte_count* bytes in this direction: ``fixed_ns``,
        plus ``byte_count / bytes_per_ns`` if ``bytes_per_ns`` is positive, plus
        a jitter drawn from *rng* if ``jitter_ns`` is."""
        delay = self.fixed_ns
        if self.bytes_per_ns > 0:
            delay += byte_count / self.bytes_per_ns
        if self.jitter_ns > 0:
            delay += float(rng.exponential(self.jitter_ns))
        return delay


class InProcess:
    """A channel to services in this process: handlers by function name,
    called on the decoded request. A handler named for a function whose
    list lengths come from its calls serves each of its versions.
    *dispatch_keys* gives the program's key per compiled name (``emulate``
    passes them); a function not in it is at the hash of its name."""

    def __init__(
        self,
        functions: Sequence[ClassicalFunction[..., Any]],
        handlers: Mapping[str, Callable[..., Any]],
        dispatch_keys: Mapping[str, int] | None = None,
    ):
        self._handlers = dict(handlers)
        self._keys = dict(dispatch_keys or {})
        # Handlers that keep state per path, as a QEC decoder's do: emulate
        # passes them the shot's calls, as simulation passes a path's.
        self._stateful = {
            name: self._handlers.pop(name)
            for name, handler in handlers.items()
            if getattr(handler, "takes_history", False)
        }

    def __repr__(self) -> str:
        names = sorted(self._handlers.keys() | self._stateful.keys())
        return f"<InProcess serving {', '.join(names) or 'nothing'}>"

    def call(self, function: ClassicalFunction[..., Any], request: bytes) -> tuple[bytes, float]:
        """Deliver *request*; returns the answer bytes and the service's time in ns."""
        handler = self._handlers.get(function.name) or self._handlers.get(function.base_name)
        if handler is None:
            return self._call(function, request, None)
        key = self._keys.get(function.name)
        request_id, timestamp, args = wire.decode_request(function, request, key)
        t0 = time.perf_counter_ns()
        answer = handler(*args)
        service_ns = float(time.perf_counter_ns() - t0)
        return wire.encode_answer(function, answer, request_id, timestamp), service_ns

    def _call(
        self, function: ClassicalFunction[..., Any], request: bytes, history: Sequence[Any] | None
    ) -> tuple[bytes, float]:
        """``call``, giving a handler that keeps state per path *history*,
        the calls the shot made before this one (``None``: its own state)."""
        handler = self._stateful.get(function.name) or self._stateful.get(function.base_name)
        if handler is None:
            if history is None:
                raise NotFoundError(f"no service for {function.name!r}")
            return self.call(function, request)
        key = self._keys.get(function.name)
        request_id, timestamp, args = wire.decode_request(function, request, key)
        t0 = time.perf_counter_ns()
        try:
            answer = handler(*args) if history is None else handler(*args, history=history)
        except _NotReadyError:
            reply = wire._status_answer(request_id, timestamp, _NOT_READY)
            return reply, float(time.perf_counter_ns() - t0)
        service_ns = float(time.perf_counter_ns() - t0)
        return wire.encode_answer(function, answer, request_id, timestamp), service_ns

    def close(self) -> None:
        """Nothing to release: the services run in this process. It is here so
        that every channel can be closed alike."""


def _reply(
    function: ClassicalFunction[..., Any],
    handler: Callable[..., Any],
    args: Sequence[Any],
    request_id: int,
    timestamp: int,
) -> tuple[bytes, int]:
    """*handler*'s answer to one request, encoded, and its time in ns. A
    decoder with nothing decoded answers NOT_READY, as a CUDA-QX server does."""
    t0 = time.perf_counter_ns()
    try:
        answer = handler(*args)
    except _NotReadyError:
        return wire._status_answer(request_id, timestamp, _NOT_READY), time.perf_counter_ns() - t0
    service_ns = time.perf_counter_ns() - t0
    return wire.encode_answer(function, answer, request_id, timestamp), service_ns


def serve_udp(
    functions: Sequence[ClassicalFunction[..., Any]],
    handlers: Mapping[str, Callable[..., Any]],
    host: str = "127.0.0.1",
    port: int = 0,
    slot_bytes: int = 65536,
    dispatch_keys: Mapping[str, int] | None = None,
) -> tuple[threading.Thread, tuple[str, int], Callable[[], None]]:
    """Serve *handlers* over UDP: one datagram per request and per answer, in
    the wire format. Returns the server thread, its address and a stop
    function. Each answer is followed by the service's own time: the 4-byte
    tag ``'CUQT'`` and the time in ns as 8 bytes, which ``UdpChannel`` reads.
    A request goes to the function at its dispatch key (*dispatch_keys* by
    compiled name, else the hash of the name). Built-in functions that share
    a key, such as QEC decoders, are told apart by their payload: its length,
    then its first argument (the decoder id)."""
    import socket

    by_key = _by_dispatch_key(functions, dispatch_keys or {})
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.05)
    running = threading.Event()
    running.set()

    def loop() -> None:
        while running.is_set():
            try:
                data, peer = sock.recvfrom(slot_bytes)
            except TimeoutError:
                continue
            except OSError:
                return
            key, length = struct.unpack_from("<II", data, 4)
            fn = _served(by_key, key, length, data)
            request_id, timestamp, args = wire.decode_request(fn, data, key)
            handler = handlers.get(fn.name) or handlers[fn.base_name]
            reply, service_ns = _reply(fn, handler, args, request_id, timestamp)
            sock.sendto(reply + _TIMING.pack(_TIMING_TAG, service_ns), peer)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    def stop() -> None:
        running.clear()
        thread.join(timeout=1.0)
        sock.close()

    return thread, sock.getsockname(), stop


def _by_dispatch_key(
    functions: Sequence[ClassicalFunction[..., Any]], keys: Mapping[str, int]
) -> dict[int, dict[int, list[ClassicalFunction[..., Any]]]]:
    """Per dispatch key, per request payload length, the functions there."""
    by_key: dict[int, dict[int, list[ClassicalFunction[..., Any]]]] = {}
    for fn in functions:
        key = keys.get(fn.name, fn._dispatch_key)
        length = sum(a.wire_bytes for a in fn.args)
        there = by_key.setdefault(key, {}).setdefault(length, [])
        if there and (fn._key_route is None or there[0]._key_route is None):
            raise InvalidValueError(
                f"{there[0].name} and {fn.name} are both at dispatch key {key:#x}; a server"
                " tells them apart only by key"
            )
        there.append(fn)
    return by_key


def _served(
    by_key: dict[int, dict[int, list[ClassicalFunction[..., Any]]]],
    key: int,
    length: int,
    data: bytes,
) -> ClassicalFunction[..., Any]:
    """The function a request at *key* with a payload of *length* bytes is for."""
    there = by_key.get(key, {}).get(length)
    if not there:
        raise NotFoundError(f"no function served at dispatch key {key:#x} takes {length} bytes")
    if len(there) == 1:
        return there[0]
    route = wire.decode_request(there[0], data, key)[2][0]
    for fn in there:
        if fn._key_route == route:
            return fn
    raise NotFoundError(f"no function served at dispatch key {key:#x} for decoder {route}")


class RealtimeChannel:
    """A channel through the CUDA-Q Realtime host library, which dispatches
    requests to registered GPU or host services. Not available yet: use
    ``InProcess`` (with ``wire.native`` handlers for compiled code) or
    ``UdpChannel``. The requests and answers are already in its wire
    format, so a channel built on it drops in here."""

    def __init__(self, library: str, *args: Any, **kwargs: Any) -> None:
        raise UnsupportedError(
            "a channel through the CUDA-Q Realtime host library is not available yet; "
            "emulate with InProcess (wire.native handlers run compiled code) or UdpChannel"
        )


def device_call_text(lowered: Any) -> str:
    """CUDA-Q kernel text that makes a compiled program's calls as
    ``cc.device_call`` operations. Not available yet: export the program
    with ``export`` (or ``ScheduledKernel.export_to_emulator()``), whose manifest holds
    each function's dispatch key, payload types and link."""
    raise UnsupportedError(
        "exporting calls as CUDA-Q cc.device_call operations is not available yet; "
        "use ScheduledKernel.export_to_emulator(), whose manifest lists every call"
    )


# What serve_udp sends after an answer: the tag 'CUQT' and the service's time in ns.
_TIMING = struct.Struct("<IQ")
_TIMING_TAG = 0x43555154


class UdpChannel:
    """A channel to a service reached over UDP: ``serve_udp``, or any server
    that answers in the wire format, such as a CUDA-QX decoding server.

    An answer ends where its header's result length says; whatever follows,
    such as the padding of a server that sends whole slots, is not part of
    it. The service's time is the one ``serve_udp`` sends after the answer,
    or, from a server that sends none, the round trip measured here."""

    def __init__(self, address: tuple[str, int], slot_bytes: int = 65536, timeout_s: float = 2.0):
        self._address = address
        import socket

        self._slot_bytes = slot_bytes
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout_s)

    def __repr__(self) -> str:
        return (
            f"UdpChannel({self._address!r}, slot_bytes={self._slot_bytes!r}, "
            f"timeout_s={self._sock.gettimeout()!r})"
        )

    def call(self, function: ClassicalFunction[..., Any], request: bytes) -> tuple[bytes, float]:
        """Send *request* and wait for the answer; returns the answer bytes and
        the service's time in ns (the measured round trip if the server sends
        none). No answer within the channel's timeout raises an error that
        names the function and the address."""
        t0 = time.perf_counter_ns()
        self._sock.sendto(request, self._address)
        try:
            data, _ = self._sock.recvfrom(self._slot_bytes)
            round_trip_ns = time.perf_counter_ns() - t0
        except TimeoutError as error:
            host, port = self._address
            raise _CallTimeoutError(
                f"classical function {function.name!r} got no answer from {host}:{port} within "
                f"{self._sock.gettimeout()} s; check that serve_udp is running there, or raise "
                "UdpChannel(timeout_s=...)"
            ) from error
        if len(data) < wire.HEADER_BYTES:
            return data, float(round_trip_ns)
        end = wire.HEADER_BYTES + struct.unpack_from("<I", data, 8)[0]
        if len(data) >= end + _TIMING.size:
            tag, service_ns = _TIMING.unpack_from(data, end)
            if tag == _TIMING_TAG:
                return data[:end], float(service_ns)
        return data[:end], float(round_trip_ns)

    def close(self) -> None:
        """Close the channel's socket."""
        self._sock.close()


# ── The virtual-time driver ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CallTrace:
    """One call in an emulated shot. Times in ns of the shot's timeline.

    ``budget_ns`` is how long after ``submit_ns`` the answer may arrive: until
    it is due, less the link's feedback time. ``in_time`` is whether
    ``latency_ns`` is within it (false for a decoder's NOT_READY)."""

    function: str
    submit_ns: float
    latency_ns: float
    budget_ns: float
    in_time: bool
    inputs: tuple[Any, ...]
    answer: Any


@dataclass
class EmulationResult:
    """Per shot, the measurement outcomes (in time order) and the calls."""

    outcomes: list[list[int]] = field(default_factory=list)
    calls: list[list[CallTrace]] = field(default_factory=list)
    ends_ns: list[float] = field(default_factory=list)
    wall_s: float = 0.0
    # (shot, call) of each call the program sends without waiting on its answer.
    _sent: list[tuple[int, int]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    @property
    def late_fraction(self) -> float:
        """The fraction of the answers the program waits on that came late
        (a decoder's NOT_READY included). A call it sends without taking an
        answer, such as an enqueue of syndromes, counts in neither part, though
        its trace says whether it came back in time."""
        if self._sent:
            sent = set(self._sent)
            traces = [
                c
                for s, shot in enumerate(self.calls)
                for k, c in enumerate(shot)
                if (s, k) not in sent
            ]
        else:
            traces = [c for shot in self.calls for c in shot]
        return sum(not c.in_time for c in traces) / len(traces) if traces else 0.0

    @property
    def calls_per_second(self) -> float:
        """The calls of every shot per second of the run's wall-clock time;
        infinite when no wall time was measured."""
        n = sum(len(shot) for shot in self.calls)
        return n / self.wall_s if self.wall_s > 0 else math.inf


def _not_ready(callee: str, request_id: int) -> ExecutionError:
    return ExecutionError(
        f"decoder call {callee!r} was answered NOT_READY (status {_NOT_READY}) to request"
        f" {request_id}, and the call has no timeout to fall back on; take it with"
        " .result(timeout=...) and branch on the flag"
    )


def _check_transport(transport: Any, machine: Any) -> None:
    """Refuse a *transport* mapping that names a link the control system
    does not have, or holds anything but a ``Transport``."""
    from .._errors import InvalidTypeError, InvalidValueError, _closest

    if not isinstance(transport, Mapping):
        raise InvalidTypeError(
            "transport is a Transport, or a mapping of link names or ids to Transports; "
            f"got {type(transport).__name__}"
        )
    names = list(machine.link_names)
    for key, value in transport.items():
        if isinstance(key, str) and key not in names:
            close = _closest(key, names)
            hint = f"did you mean {close!r}? " if close else ""
            known = f"its links are {', '.join(names)}" if names else "it has no named links"
            raise InvalidValueError(
                f"transport names the link {key!r}, which the control system does not "
                f"have; {hint}{known}"
            )
        if not isinstance(value, Transport):
            raise InvalidTypeError(
                f"transport[{key!r}] must be a Transport, got {type(value).__name__}"
            )


def emulate(
    lowered: Any,
    *,
    shots_count: int = 1,
    channel: Any = None,
    handlers: Mapping[str, Callable[..., Any]] | None = None,
    transport: Transport | Mapping[int | str, Transport] | None = None,
    outcomes: Callable[[int, int, np.random.Generator], int] | None = None,
    service_time: str = "measured",
    seed: int | None = None,
) -> EmulationResult:
    """Run *lowered* for *shots_count* in virtual time against classical services.

    *channel* reaches the services (``InProcess`` of *handlers* by default,
    the functions' bodies unless given). *transport* models each link's
    delay in each direction, per link (name or id) or for all. *outcomes* gives the
    level a readout reports, from its qudit and the shot's generator
    (uniform 0/1 by default); a call that takes IQ points gets them drawn
    at those levels from the quantum system's dispersive readout model
    (``IQModel``). The answer's latency is the request's
    transport delay, plus the service's time (``"measured"`` wall-clock, or
    ``"none"``), plus the answer's transport delay. *shots_count* is a positive int.
    """
    if shots_count.__class__ is not int or shots_count < 1:
        shots_count = _positive_int("shots_count", shots_count)
    functions = tuple(lowered.classical)
    by_name = {fn.name: fn for fn in functions}
    chosen = _handlers_for(functions, handlers)
    keys = {name: key for name, (key, _) in _dispatch_keys(lowered).items()}
    channel = channel or InProcess(functions, chosen, keys)
    call_service = channel.call
    if isinstance(channel, InProcess) and channel._stateful:
        # Handlers that keep state per path get the shot's calls so far.
        def call_service(fn: ClassicalFunction[..., Any], request: bytes) -> tuple[bytes, float]:
            return channel._call(fn, request, traces)

    import numpy as np

    rng = np.random.default_rng(seed)
    machine = lowered.options.machine
    if transport is not None and not isinstance(transport, Transport):
        _check_transport(transport, machine)
    from .._programs import _schedule_args

    schedule_args = _schedule_args(lowered.options, lowered.clocks)
    base = lowered._concrete().simulation_module(*schedule_args)
    due_ns = base.call_due_ns()
    clock = lowered.quantum_system.clock_ghz
    iq_model = (
        IQModel(lowered.quantum_system._description, lowered.qudits, decay=False)
        if any(takes_iq(fn) for fn in functions)
        else None
    )

    def transport_of(link: int) -> Transport:
        if transport is None:
            return Transport()
        if isinstance(transport, Transport):
            return transport
        found = transport.get(link)
        if found is None:
            name = machine.link_name(link)
            found = transport.get(name) if name is not None else None
        return found or Transport()

    def outcome_of(qudit: int) -> int:
        if outcomes is None:
            return int(rng.integers(0, 2))
        return int(outcomes(qudit, 0, rng))

    result = EmulationResult()
    t_wall = time.perf_counter()
    # Unique over the run, so that an answer to an earlier shot's request
    # is not taken for one to this shot's.
    request_id = 0
    for _ in range(shots_count):
        module = base
        path: list[tuple[tuple[int, tuple[int, ...]], int]] = []
        shot_outcomes: list[int] = []
        traces: list[CallTrace] = []
        seen: dict[tuple[int, tuple[int, ...]], tuple[int, float]] = {}
        while True:
            entries = [[index, *iteration, k] for (index, iteration), k in path]
            found = module.next_measurement(entries)
            call = module.next_call(entries)
            measure_at = math.inf if found is None else found[4]
            call_at = math.inf if call is None else call[1]
            if call is not None and call_at < measure_at - 1e-9:
                index, start, callee, arguments, answered, timeout = call
                fn = by_name[callee]
                link = machine.link(fn.link)
                # The answer is in time if it arrives the link's feedback
                # time before it is due, so the pulses that use it can.
                budget = due_ns[index] - link.feedback_ns
                known = {key: k for key, k in path}
                point = None
                if iq_model is not None and takes_iq(fn):
                    model = iq_model
                    drives = {
                        (i, tuple(it)): (f, a) for i, it, f, a in module.readout_drives(entries)
                    }

                    def point(key: tuple[int, tuple[int, ...]]) -> tuple[float, float]:
                        qudit, duration = seen[key]  # noqa: B023
                        frequency, amplitude = drives.get(key, (math.nan, math.nan))  # noqa: B023
                        return model.sample(rng, qudit, known[key], frequency, amplitude, duration)  # noqa: B023

                inputs = _call_inputs(fn, arguments, known, point)
                if fn.uniform is not None:
                    # An on-device source: the controller answers, not a link.
                    value = int(rng.integers(fn.uniform))
                    traces.append(
                        CallTrace(callee, start, fn.nominal_latency_ns, budget, True, (), value)
                    )
                    module = module.bind_call(index, [float(value)], *schedule_args)
                    continue
                request = wire.encode_request(fn, inputs, request_id, dispatch_key=keys[callee])
                answer_bytes, service_ns = call_service(fn, request)
                status, answered_id, answer = wire.decode_answer(fn, answer_bytes)
                # A decoder not done decoding has no answer: it is late.
                ready = status == 0
                if not ready and (status != _NOT_READY or fn._dispatch_key not in _RESERVED_KEYS):
                    raise wire.call_failed(callee, answered_id, status)
                path_transport = transport_of(machine.link_id(fn.link))
                if answered_id != request_id:
                    from .._errors import ExecutionError

                    raise ExecutionError(
                        f"classical function {callee!r} sent request {request_id} and got the "
                        f"answer to request {answered_id}: a late or repeated answer to an "
                        "earlier request, or one meant for another client; a channel must "
                        "return the answer to the request it was given"
                    )
                request_id += 1
                latency = path_transport.delay_ns(len(request), rng) + path_transport.delay_ns(
                    len(answer_bytes), rng
                )
                if service_time == "measured":
                    latency += service_ns
                in_time = ready and latency <= budget
                traces.append(CallTrace(callee, start, latency, budget, in_time, inputs, answer))
                if not answered:
                    result._sent.append((len(result.calls), len(traces) - 1))
                    module = module.bind_call(index, [], *schedule_args)
                    continue
                has_timeout = timeout >= 0 or fn.timeout_ns is not None
                if not ready:
                    if not has_timeout:
                        raise _not_ready(callee, answered_id)
                    values = [0.0] * (1 + sum(r.answer_count for r in fn.results))
                else:
                    values = _answer_values(fn, answer)
                    if has_timeout:
                        values = [1.0, *values] if in_time else [0.0] * (1 + len(values))
                module = module.bind_call(index, values, *schedule_args)
                continue
            if found is None:
                break
            ridx, iteration, qudit, begin, finish = found
            level = outcome_of(qudit)
            seen[(ridx, tuple(iteration))] = (qudit, finish - begin)
            path.append(((ridx, tuple(iteration)), level))
            shot_outcomes.append(level)
        ends = [s + d for _, _, _, s, d in module.events() if s is not None and d is not None]
        result.outcomes.append(shot_outcomes)
        result.calls.append(traces)
        result.ends_ns.append(max(ends, default=0) / clock)
    result.wall_s = time.perf_counter() - t_wall
    return result


# ── The export, named here ───────────────────────────────────────────────────


class _Exported(tuple[str, str]):
    """A name of this module and that of what it names in ``wire``, looked up
    on first use and then kept in this module's namespace, where later reads
    find it as a plain attribute."""

    def __get__(self, instance: types.ModuleType | None, owner: type) -> Any:
        if instance is None:
            return self
        name, home = self
        value = instance.__dict__[name] = getattr(wire, home)
        return value


class _EmulationModule(types.ModuleType):
    # Emulating needs none of the export, and exporting loads only the wire
    # format, where the export lives.
    FORMAT = _Exported(("FORMAT", "_EMULATION_FORMAT"))
    MANIFEST_FORMAT = _Exported(("MANIFEST_FORMAT", "_MANIFEST_FORMAT"))
    export = _Exported(("export", "_emulation_export"))
    export_part = _Exported(("export_part", "_export_part"))
    manifest = _Exported(("manifest", "_manifest"))

    def __dir__(self) -> list[str]:
        exported = [
            name for name, value in vars(_EmulationModule).items() if isinstance(value, _Exported)
        ]
        return sorted({*super().__dir__(), *exported})


sys.modules[__name__].__class__ = _EmulationModule
