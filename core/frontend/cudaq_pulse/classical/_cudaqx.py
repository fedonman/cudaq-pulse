# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-Q QEC's real-time decoding functions, as pulse kernels call them.

``pulse.classical(cudaq_qec.qec.get_corrections, ...)`` gives a classical
function that sends CUDA-Q QEC's own message: the decoders are defined and
configured in CUDA-Q QEC (``cudaq_qec.configure_decoders...``) and a pulse
program calls them as a CUDA-Q kernel would. The functions are recognized
by their module and name, so nothing here imports ``cudaq_qec`` until a
decoder has to answer in simulation or emulation.

A call sends the decoding server's message (its ``MESSAGE_FORMAT.md``):

- ``enqueue_syndromes(decoder_id, syndromes, tag)``: ``{decoder id, tag,
  syndrome mapping id 0, count}`` and the syndrome bits; nothing comes back.
- ``get_corrections(decoder_id, return_size, reset)``: ``{decoder id,
  return size, reset}``; ``return_size`` bits come back.
- ``reset_decoder(decoder_id)``: ``{decoder id}``.

Each message's key is its name's hash, shared by every decoder; the server
tells decoders apart by the id. So the decoder id and the return size must
be known when the kernel is compiled: each pair is one version
(``get_corrections__0_3``: decoder 0, three bits).

Simulation and emulation run the configured decoders through CUDA-Q QEC's
simulation library, which a CUDA-Q kernel on a simulator also calls: the
corrections of every enqueued round accumulate until a reset. They answer
not ready, as the decoding server does, when nothing was enqueued since the
decoder was reset (by ``reset_decoder`` or ``get_corrections(..., True)``);
the simulation library itself answers zeros there. The library keeps one
state per decoder for the process, so each answer replays the path's (or
shot's) earlier calls of that decoder after a reset, under a lock.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Sequence
from typing import Any

from .._errors import ExecutionError, InvalidTypeError
from . import (
    ClassicalFunction,
    _Bits,
    _Byte,
    _Integer,
    _NotReadyError,
    dispatch_key,
)

# The module CUDA-Q QEC's realtime functions live in (cudaq_qec.qec), from
# cudaq-qec 0.6.0 (pybind11) to 0.8.1 (nanobind).
_MODULE = "cudaq_qec._pycudaqx_qec_the_suffix_matters_cudaq_qec.qec"

_LIBRARY = "libcudaq-qec-realtime-decoding-simulation.so"
# Those a CUDA-Q kernel on a simulator calls, in every release from 0.6.0.
_SYMBOLS = {
    "reset": "_ZN5cudaq3qec8decoding10simulation13reset_decoderEm",
    "enqueue": "simulation_enqueue_syndromes",
    "get": "simulation_get_corrections",
}


class _Message:
    """One decoding-server message: the CUDA-Q QEC function that sends it,
    its parameters there, and the key the server registers it under."""

    __slots__ = ("name", "parameters", "key", "answers", "released")

    def __init__(
        self, name: str, parameters: tuple[str, ...], key: str, answers: bool, released: bool
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.key = dispatch_key(key)
        self.answers = answers
        # In a cudaq-qec release: the others are on its main branch only.
        self.released = released

    def __reduce__(self) -> Any:
        return (_message, (self.name,))


_MESSAGES = {
    m.name: m
    for m in (
        _Message(
            "enqueue_syndromes",
            ("decoder_id", "syndromes", "tag"),
            "enqueue_syndromes",
            answers=False,
            released=True,
        ),
        _Message(
            "enqueue_syndromes_mapped",
            ("decoder_id", "syndromes", "tag", "syndrome_mapping_id"),
            "enqueue_syndromes",
            answers=False,
            released=False,
        ),
        _Message(
            "get_corrections",
            ("decoder_id", "return_size", "reset"),
            "get_corrections",
            answers=True,
            released=True,
        ),
        _Message(
            "get_logical_measurements",
            ("decoder_id", "first", "return_size", "reset"),
            "get_logical_measurements",
            answers=True,
            released=False,
        ),
        _Message("reset_decoder", ("decoder_id",), "reset_decoder", answers=False, released=True),
    )
}

# What a submission of each message sends before its measurements, word by
# word: the decoder id or the length its version is named after ("id",
# "length"), a constant, or a value of the caller's (None). ``_Calls._wire``
# sends these, and a module loaded from another compiler is held to them.
_WORDS: dict[str, tuple[tuple[str, Any], ...]] = {
    "enqueue_syndromes": (
        ("decoder_id", "id"),
        ("tag", None),
        ("syndrome_mapping_id", 0),
        ("number of syndromes", "length"),
    ),
    "enqueue_syndromes_mapped": (
        ("decoder_id", "id"),
        ("tag", None),
        ("syndrome_mapping_id", None),
        ("number of syndromes", "length"),
    ),
    "get_corrections": (("decoder_id", "id"), ("return_size", "length"), ("reset", None)),
    "get_logical_measurements": (
        ("decoder_id", "id"),
        ("first", None),
        ("return_size", "length"),
        ("reset", None),
    ),
    "reset_decoder": (("decoder_id", "id"),),
}


def _message_version(name: str) -> tuple[Any, tuple[int, ...]] | None:
    """The CUDA-Q QEC decoder message *name* is a version of, and the
    version's numbers (the decoder id, then a length), as ``compile`` names
    one: ``get_corrections__3_1``."""
    import re

    # ASCII digits only: the name must be the one compile writes.
    found = re.fullmatch(r"([a-z_]+?)__(0|[1-9][0-9]{0,18})(?:_([1-9][0-9]{0,18}))?", name)
    if found is None:
        return None
    message = _MESSAGES.get(found[1])
    numbers = tuple(int(n) for n in found.groups()[1:] if n is not None)
    # With numbers a message's word holds, as many as its name has.
    if (
        message is None
        or len(numbers) != (1 if message.name == "reset_decoder" else 2)
        or any(n >= 2**63 for n in numbers)
    ):
        return None
    return message, numbers


_RELEASED = ", ".join(f"cudaq_qec.qec.{m.name}" for m in _MESSAGES.values() if m.released)


def _message(name: str) -> _Message:
    return _MESSAGES[name]


def is_cudaq_qec(fn: Any) -> bool:
    """Whether *fn* comes from CUDA-Q QEC (by its module; nothing is imported)."""
    module = getattr(fn, "__module__", None)
    return isinstance(module, str) and (module == "cudaq_qec" or module.startswith("cudaq_qec."))


def declare(fn: Any, options: dict[str, Any]) -> ClassicalFunction[..., Any]:
    """``pulse.classical(fn, **options)`` for a function of CUDA-Q QEC: a
    classical function of no schema of its own, whose calls (a kernel's
    ``.submit(...)``, a call outside one) its ``_Calls`` routes to the
    version for their decoder id and lengths."""
    name = getattr(fn, "__name__", None)
    message = _MESSAGES.get(name) if getattr(fn, "__module__", None) == _MODULE else None  # type: ignore[arg-type]
    if message is None:
        raise InvalidTypeError(
            f"pulse.classical takes CUDA-Q QEC's real-time decoding functions ({_RELEASED});"
            f" {getattr(fn, '__module__', '?')}.{name} is not one of them"
        )
    if not message.released:
        raise InvalidTypeError(
            f"cudaq_qec.qec.{message.name} is not in a cudaq-qec release yet, and cudaq-pulse"
            f" calls only released ones ({_RELEASED})"
        )
    if "name" in options:
        raise InvalidTypeError(
            f"cudaq_qec.qec.{message.name}: name= is the decoding server's; pulse.classical takes"
            " the other options"
        )
    options = dict(options)
    if message.answers:
        if "max_latency_ns" not in options:
            raise InvalidTypeError(
                f"pulse.classical(cudaq_qec.qec.{message.name}, ...) needs max_latency_ns, the"
                " longest the decoder may take to answer"
            )
    else:
        options.setdefault("max_latency_ns", 0.0)
    return _Calls(message, options, getattr(fn, "__doc__", None)).function


class _Calls:
    """The calls of one ``pulse.classical(cudaq_qec.qec.<name>, ...)``: the
    function's implementation, which takes CUDA-Q QEC's arguments and picks
    the version (decoder id and length) they send. A kernel's submit and a
    call outside one reach it because the function has no schema of its own
    (``args == ()``)."""

    __slots__ = ("message", "options", "versions", "function")

    def __init__(self, message: _Message, options: dict[str, Any], doc: str | None) -> None:
        self.message = message
        self.options = options
        self.versions: dict[tuple[int, ...], ClassicalFunction[..., Any]] = {}
        function = ClassicalFunction._declared(
            self, (), (), name=message.name, _dispatch_key=message.key, **options
        )
        function.__doc__ = doc
        function.__wrapped__ = None
        self.function = function

    def _route(self, args: Sequence[Any]) -> tuple[ClassicalFunction[..., Any], tuple[Any, ...]]:
        """The version a kernel's ``.submit(*args)`` sends, and its wire arguments."""
        from ..kernel.ir_builder import CompilationError

        return self._wire(args, CompilationError)

    def _wire(
        self, args: Sequence[Any], error: type[Exception]
    ) -> tuple[ClassicalFunction[..., Any], tuple[Any, ...]]:
        message = self.message
        name = message.name
        if len(args) != len(message.parameters):
            count = len(message.parameters)
            raise error(
                f"{name}.submit() takes {count} argument{'' if count == 1 else 's'}"
                f" ({', '.join(message.parameters)}), got {len(args)}"
            )
        decoder_id = _known(name, "decoder_id", args[0], error, 0)
        if name == "reset_decoder":
            return self._version((decoder_id,)), (decoder_id,)
        if name.startswith("enqueue_syndromes"):
            syndromes = args[1]
            if syndromes.__class__ not in (list, tuple) or not syndromes:
                raise error(
                    f"{name}.submit(): syndromes takes a list of measurements, got {syndromes!r}"
                )
            mapping = args[3] if len(args) > 3 else 0
            version = self._version((decoder_id, len(syndromes)))
            return version, (decoder_id, args[2], mapping, len(syndromes), list(syndromes))
        if name == "get_corrections":
            size = _known(name, "return_size", args[1], error, 1)
            reset = args[2]
            return self._version((decoder_id, size)), (decoder_id, size, _byte(reset))
        size = _known(name, "return_size", args[2], error, 1)
        return self._version((decoder_id, size)), (decoder_id, args[1], size, _byte(args[3]))

    def _version(self, key: tuple[int, ...]) -> ClassicalFunction[..., Any]:
        version = self.versions.get(key)
        if version is None:
            version = self.versions[key] = _make(self, key)
        return version

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """A call outside a kernel: the configured decoder answers now
        (CUDA-Q QEC's simulation library keeps its state)."""
        name = self.message.name
        if kwargs:
            raise InvalidTypeError(f"{name}() takes positional arguments, as CUDA-Q QEC's does")
        version, wired = self._wire(args, InvalidTypeError)
        answer = version.implementation
        assert answer is not None
        return answer(*wired)

    def _pickle_function(self, function: ClassicalFunction[..., Any]) -> Any:
        return (_unpickled_call, (self.message.name, self.options))


def _known(name: str, what: str, value: Any, error: type[Exception], least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(
            f"{name}.submit(): {what} must be known when the kernel is compiled (a number, or a"
            f" kernel argument given to lower()), because it picks the message's version; got"
            f" {value!r}"
        )
    if not least <= value < 2**63:
        raise error(f"{name}.submit(): {what} must be at least {least}, got {value}")
    return value


def _byte(value: Any) -> Any:
    return int(value) if value.__class__ is bool else value


def _make(call: _Calls, key: tuple[int, ...]) -> ClassicalFunction[..., Any]:
    """The version of *call* for *key* (decoder id, then a length): one
    fixed wire schema, routed by the decoder id."""
    message = call.message
    decoder_id = key[0]
    i64 = _Integer()
    results: tuple[Any, ...] = ()
    if message.name == "reset_decoder":
        args: tuple[Any, ...] = (i64,)
    elif message.name.startswith("enqueue_syndromes"):
        args = (i64, i64, i64, i64, _Bits(key[1]))
    elif message.name == "get_corrections":
        args, results = (i64, i64, _Byte()), (_Bits(key[1]),)
    else:
        args, results = (i64, i64, i64, _Byte()), (_Bits(key[1]),)
    answer = _Answer(message, decoder_id, key[1] if len(key) > 1 else 0, call)
    version = ClassicalFunction._declared(
        answer,
        args,
        results,
        name=f"{message.name}__{'_'.join(map(str, key))}",
        _dispatch_key=message.key,
        _key_route=decoder_id,
        **call.options,
    )
    version.base_name = message.name
    version.__doc__ = call.function.__doc__
    version.__wrapped__ = None
    _ANSWERS[version.name] = answer
    return version


# Per version name, what answers it: the replay of a path's calls finds
# them by their compiled names.
_ANSWERS: dict[str, _Answer] = {}


class _Answer:
    """What answers a version: the configured CUDA-Q QEC decoder."""

    __slots__ = ("message", "decoder_id", "size", "call")

    # Simulation and emulation pass the path's (or shot's) calls.
    takes_history = True

    def __init__(self, message: _Message, decoder_id: int, size: int, call: _Calls) -> None:
        self.message = message
        self.decoder_id = decoder_id
        self.size = size
        self.call = call

    def __call__(self, *inputs: Any, history: Sequence[Any] | None = None) -> Any:
        name = self.message.name
        decoder_id = self.decoder_id
        if history is not None and name != "get_corrections":
            return None  # the get that needs the state replays it
        with _LOCK:
            decoders = _decoders(name)
            if history is None:
                return _live(decoders, self, inputs)
            pending = _replay(decoders, decoder_id, history)
            if not pending:
                raise _NotReadyError
            return decoders.get(decoder_id, self.size, bool(inputs[2]))

    def _key(self) -> tuple[int, ...]:
        if self.message.name == "reset_decoder":
            return (self.decoder_id,)
        return (self.decoder_id, self.size)

    def __reduce__(self) -> Any:
        return (_answer_of, (self.call.function, self._key()))

    def _pickle_function(self, version: ClassicalFunction[..., Any]) -> Any:
        return (_unpickled_version, (self.call.function, self._key()))


_LOCK = threading.Lock()
# Outside simulation and emulation: per decoder, whether something was
# enqueued since its last reset.
_PENDING: dict[int, bool] = {}


def _live(decoders: Any, answer: _Answer, inputs: Sequence[Any]) -> Any:
    name = answer.message.name
    decoder_id = answer.decoder_id
    if name == "reset_decoder":
        decoders.reset(decoder_id)
        _PENDING[decoder_id] = False
        return None
    if name == "enqueue_syndromes":
        decoders.enqueue(decoder_id, inputs[4], int(inputs[1]))
        _PENDING[decoder_id] = True
        return None
    if not _PENDING.get(decoder_id, False):
        raise _NotReadyError
    reset = bool(inputs[2])
    corrections = decoders.get(decoder_id, answer.size, reset)
    if reset:
        _PENDING[decoder_id] = False
    return corrections


def _replay(decoders: Any, decoder_id: int, history: Sequence[Any]) -> bool:
    """Reset the decoder and make the path's earlier calls of it again;
    whether something is enqueued since its last reset."""
    decoders.reset(decoder_id)
    pending = False
    for record in history:
        answer = _ANSWERS.get(getattr(record, "compiled_name", record.function))
        if answer is None or answer.decoder_id != decoder_id:
            continue
        name = answer.message.name
        inputs = record.inputs
        if name == "enqueue_syndromes":
            decoders.enqueue(decoder_id, inputs[4], int(inputs[1]))
            pending = True
        elif name == "reset_decoder":
            decoders.reset(decoder_id)
            pending = False
        elif pending and inputs[2]:
            decoders.get(decoder_id, answer.size, True)
            pending = False
    return pending


# The decoders: CUDA-Q QEC's simulation library, loaded at the first answer.
# Tests put a stand-in here.
_backend: Any = None


def _decoders(name: str) -> Any:
    global _backend
    if _backend is None:
        _backend = _SimulationLibrary(name)
    return _backend


class _SimulationLibrary:
    """CUDA-Q QEC's simulation decoders: the decoders the process configured
    with ``cudaq_qec.configure_decoders...``."""

    def __init__(self, name: str) -> None:
        import ctypes
        import os

        what = f"simulating cudaq_qec.qec.{name} runs the decoders configured in CUDA-Q QEC"
        if sys.platform == "darwin":
            raise ExecutionError(
                f"{what}, and CUDA-Q QEC has no macOS build: simulate it on Linux, or give the"
                f" call a handler (handlers={{{name!r}: ...}})"
            )
        try:
            cudaq_qec = importlib.import_module("cudaq_qec")
        except ImportError as error:
            raise ExecutionError(
                f"{what}, and cudaq_qec does not import ({error}): pip install cudaq-qec"
                " (Linux, Python 3.11-3.13), or give the call a handler"
            ) from error
        package_file = cudaq_qec.__file__
        if package_file is None:
            raise ExecutionError(f"{what}, and this cudaq_qec has no {_LIBRARY}")
        path = os.path.join(os.path.dirname(package_file), "lib", _LIBRARY)
        if not os.path.exists(path):
            raise ExecutionError(f"{what}, and this cudaq_qec has no {_LIBRARY}")
        load = getattr(getattr(cudaq_qec, "qecrt", None), "load_simulation_realtime_decoding", None)
        if load is not None:
            load()
        library = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        self._addresses: dict[str, int] = {}
        for role, symbol in _SYMBOLS.items():
            function = getattr(library, symbol, None)
            address = None if function is None else ctypes.cast(function, ctypes.c_void_p).value
            if not address:
                raise ExecutionError(f"{what}, and its {_LIBRARY} has no {symbol}")
            self._addresses[role] = address

    def reset(self, decoder_id: int) -> None:
        from .._native import decoder_reset

        _checked(decoder_reset, decoder_id, self._addresses["reset"], decoder_id)

    def enqueue(self, decoder_id: int, syndromes: Sequence[Any], tag: int) -> None:
        from .._native import decoder_enqueue

        bits = bytes(1 if s else 0 for s in syndromes)
        _checked(decoder_enqueue, decoder_id, self._addresses["enqueue"], decoder_id, bits, tag)

    def get(self, decoder_id: int, size: int, reset: bool) -> list[int]:
        from .._native import decoder_get

        bits = _checked(decoder_get, decoder_id, self._addresses["get"], decoder_id, size, reset)
        return list(bits)


def _checked(call: Any, decoder_id: int, *args: Any) -> Any:
    try:
        return call(*args)
    except RuntimeError as error:
        raise ExecutionError(
            f"CUDA-Q QEC decoder {decoder_id}: {error} (decoders are configured in this process"
            " with cudaq_qec.configure_decoders...)"
        ) from None


def _use_backend(backend: Any) -> Any:
    """Answer with *backend* (``reset``, ``enqueue``, ``get``) instead of
    CUDA-Q QEC's library; returns the one before. For tests."""
    global _backend
    with _LOCK:
        before, _backend = _backend, backend
        _PENDING.clear()
    return before


def _unpickled_call(name: str, options: dict[str, Any]) -> ClassicalFunction[..., Any]:
    return _Calls(_MESSAGES[name], options, None).function


def _calls(function: ClassicalFunction[..., Any]) -> _Calls:
    calls = function.implementation
    assert isinstance(calls, _Calls)
    return calls


def _answer_of(function: ClassicalFunction[..., Any], key: tuple[int, ...]) -> _Answer:
    answer = _calls(function)._version(key).implementation
    assert isinstance(answer, _Answer)
    return answer


def _unpickled_version(
    function: ClassicalFunction[..., Any], key: tuple[int, ...]
) -> ClassicalFunction[..., Any]:
    return _calls(function)._version(key)
