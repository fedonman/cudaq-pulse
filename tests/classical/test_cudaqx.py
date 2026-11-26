# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-Q QEC's real-time decoding functions called from pulse kernels.

A stand-in (``fake_cudaq_qec``) takes the place of ``cudaq_qec`` and of the
decoders it configures, so these run where CUDA-Q QEC does not install;
``test_cudaqx_wheel.py`` runs them on the real package.
"""

from __future__ import annotations

import json
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import classical
from cudaq_pulse.classical import _cudaqx, emulation, wire

from tests.classical.fake_cudaq_qec import Decoders, answering, qec, qec_060, repetition
from tests.classical.test_simulation import PI_AMP, _device

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "realtime_wire.json").read_text())
CUDAQX = FIXTURE["cudaqx_decoder"]

enqueue_syndromes = pulse.classical(qec.enqueue_syndromes)
get_corrections = pulse.classical(qec.get_corrections, max_latency_ns=100)
reset_decoder = pulse.classical(qec.reset_decoder)


def _key(request):
    return struct.unpack_from("<I", request, 4)[0]


def _parity(bits):
    return [sum(bits) % 2]


# ── Declaring them ───────────────────────────────────────────────────────────


def test_pulse_classical_takes_cudaq_qec_s_functions():
    assert isinstance(enqueue_syndromes, classical.ClassicalFunction)
    assert enqueue_syndromes.name == "enqueue_syndromes" and enqueue_syndromes.link == 0
    assert get_corrections.base_name == "get_corrections" and get_corrections.args == ()
    # Fire and forget: no latency budget unless one is given.
    assert enqueue_syndromes.max_latency_ns == 0.0 and reset_decoder.max_latency_ns == 0.0
    on_gpu = pulse.classical(qec.enqueue_syndromes, link="gpu", max_latency_ns=50)
    assert on_gpu.link == "gpu" and on_gpu.max_latency_ns == 50.0
    # cudaq-qec 0.6.0's functions (pybind11) are taken by the same module and name.
    assert pulse.classical(qec_060.reset_decoder).name == "reset_decoder"
    # As a decorator's argument too.
    assert pulse.classical(max_latency_ns=10)(qec.get_corrections).max_latency_ns == 10.0


def test_importing_and_declaring_import_neither_cudaq_qec_nor_cudaq():
    code = (
        "import sys, cudaq_pulse as pulse\n"
        "from tests.classical.fake_cudaq_qec import qec\n"
        "pulse.classical(qec.get_corrections, max_latency_ns=1)\n"
        "print(sorted(m for m in ('cudaq', 'cudaq_qec') if m in sys.modules))\n"
    )
    path = [str(Path(pulse.__file__).parents[1]), str(Path(__file__).resolve().parents[2])]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(path))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_a_function_that_is_not_a_realtime_one_is_an_error():
    with pytest.raises(pulse.InvalidTypeError, match=r"enqueue_syndromes_test is not one of them"):
        pulse.classical(qec.enqueue_syndromes_test)
    other = type(qec.reset_decoder)("get_decoder", module="cudaq_qec")
    with pytest.raises(pulse.InvalidTypeError, match=r"cudaq_qec.get_decoder is not one of them"):
        pulse.classical(other)
    with pytest.raises(pulse.InvalidTypeError, match="function defined with def"):
        pulse.classical(len)


def test_a_function_no_release_has_yet_is_an_error():
    for fn in (qec.enqueue_syndromes_mapped, qec.get_logical_measurements):
        with pytest.raises(pulse.InvalidTypeError, match="not in a cudaq-qec release yet"):
            pulse.classical(fn, max_latency_ns=10)


def test_its_options_are_those_of_pulse_classical_but_name():
    with pytest.raises(pulse.InvalidTypeError, match="needs max_latency_ns"):
        pulse.classical(qec.get_corrections)
    with pytest.raises(pulse.InvalidTypeError, match="name= is the decoding server's"):
        pulse.classical(qec.reset_decoder, name="reset")
    with pytest.raises(pulse.InvalidValueError, match="latencies must satisfy"):
        pulse.classical(qec.get_corrections, max_latency_ns=10, min_latency_ns=20)


# ── The messages ─────────────────────────────────────────────────────────────


def test_calls_carry_the_cudaqx_ids_and_payloads():
    bits = [1, 0] * 6
    version, wired = _cudaqx._calls(enqueue_syndromes)._wire((3, bits, 5), TypeError)
    assert version.name == "enqueue_syndromes__3_12" and wired == (3, 5, 0, 12, bits)
    enqueue = wire.encode_request(version, wired)
    assert _key(enqueue) == CUDAQX["function_ids"]["enqueue_syndromes"]
    header = CUDAQX["payload_bytes"]["enqueue_syndromes_header"]
    assert len(enqueue) - 24 == header + 2
    assert struct.unpack_from("<qqqq", enqueue, 24) == (3, 5, 0, 12)
    version, wired = _cudaqx._calls(get_corrections)._wire((3, 2, True), TypeError)
    assert version.name == "get_corrections__3_2" and wired == (3, 2, 1)
    get = wire.encode_request(version, wired)
    assert _key(get) == CUDAQX["function_ids"]["get_corrections"]
    assert len(get) - 24 == CUDAQX["payload_bytes"]["get_corrections"]
    assert [r.descriptor for r in version.results] == ["bits:2"]
    version, wired = _cudaqx._calls(reset_decoder)._wire((3,), TypeError)
    reset = wire.encode_request(version, wired)
    assert _key(reset) == CUDAQX["function_ids"]["reset_decoder"]
    assert len(reset) - 24 == CUDAQX["payload_bytes"]["reset_decoder"]


def test_the_message_table_has_the_server_s_keys():
    for name, message in _cudaqx._MESSAGES.items():
        server = "enqueue_syndromes" if name.startswith("enqueue") else name
        assert message.key == CUDAQX["function_ids"][server]


def test_decoders_share_the_cudaqx_keys_in_one_program():
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        enqueue_syndromes.submit(0, [m], 0)
        get_corrections.submit(0, 1, False).result()
        enqueue_syndromes.submit(1, [m, m], 0)
        get_corrections.submit(1, 2, False).result()

    dev = _device()
    low = pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])
    assert [f.name for f in low.classical] == [
        "enqueue_syndromes__0_1",
        "get_corrections__0_1",
        "enqueue_syndromes__1_2",
        "get_corrections__1_2",
    ]
    report = low.classical_report()
    assert {c.function for c in report} == {"enqueue_syndromes", "get_corrections"}
    keys = {c.dispatch_key for c in report}
    assert keys == {CUDAQX["function_ids"][n] for n in ("enqueue_syndromes", "get_corrections")}


def test_the_decoder_id_and_return_size_are_known_when_compiled():
    @pulse.kernel
    def by_argument(q, decoder_id: int, size: int):
        r = get_readout_line(q)
        enqueue_syndromes.submit(decoder_id, [readout(r, square(100, 0.05))], 0)
        get_corrections.submit(decoder_id, size, False).result()

    dev = _device()
    low = pulse.compile(by_argument, quantum_system=dev).lower(dev.qudits[0], 2, 4)
    assert [f.name for f in low.classical] == ["enqueue_syndromes__2_1", "get_corrections__2_4"]

    @pulse.kernel
    def measured(q):
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        get_corrections.submit(0, m, False).result()

    with pytest.raises(pulse.CompilationError, match="return_size must be known when the kernel"):
        pulse.compile(measured, quantum_system=dev).lower(dev.qudits[0])


@pulse.kernel
def _too_few(q):
    m = readout(get_readout_line(q), square(100, 0.05))
    enqueue_syndromes.submit(0, [m])


@pulse.kernel
def _not_a_list(q):
    m = readout(get_readout_line(q), square(100, 0.05))
    enqueue_syndromes.submit(0, m, 0)


@pulse.kernel
def _no_bits(q):
    get_corrections.submit(0, 0, False).result()


@pulse.kernel
def _negative_id(q):
    get_corrections.submit(-1, 1, False).result()


@pulse.kernel
def _by_keyword(q):
    get_corrections.submit(0, return_size=1, reset=False).result()


@pulse.kernel
def _no_arguments(q):
    reset_decoder.submit()


@pulse.kernel
def _called(q):
    reset_decoder(0)


@pytest.mark.parametrize(
    "kernel, match",
    [
        (_too_few, r"takes 3 arguments \(decoder_id, syndromes, tag\), got 2"),
        (_not_a_list, "syndromes takes a list of measurements"),
        (_no_bits, "return_size must be at least 1"),
        (_negative_id, "decoder_id must be at least 0"),
        (_by_keyword, "takes positional arguments only"),
        (_no_arguments, r"takes 1 argument \(decoder_id\), got 0"),
        (_called, r"call reset_decoder.submit\(...\)"),
    ],
)
def test_a_wrong_call_says_what_is_wrong(kernel, match):
    dev = _device()
    with pytest.raises(pulse.CompilationError, match=match):
        pulse.compile(kernel, quantum_system=dev).lower(dev.qudits[0])


# ── Names of the versions ────────────────────────────────────────────────────


def test_a_function_of_the_users_cannot_take_the_name_of_a_message_version():
    # A program's functions, and the replay of a decoder's calls in a
    # simulation, are found by name: under the name of a version, a call
    # with one argument failed the simulation with a bare IndexError, and
    # one with five arguments of fitting kinds was enqueued into the decoder.
    def one(bits: pulse.Length[pulse.Bits, 1]) -> None:
        return None

    def five(
        decoder: int, tag: int, mapping: int, count: int, bits: pulse.Length[pulse.Bits, 1]
    ) -> None:
        return None

    for body in (one, five):
        for name, said in (
            ("enqueue_syndromes__0_1", "enqueue_syndromes for decoder 0 and length 1"),
            ("get_corrections__3_1", "get_corrections for decoder 3 and length 1"),
            ("reset_decoder__7", "reset_decoder for decoder 7"),
        ):
            with pytest.raises(pulse.InvalidValueError) as refused:
                pulse.classical(body, name=name, max_latency_ns=0)
            message = str(refused.value)
            assert f"can't be named {name!r}" in message
            assert f"the CUDA-Q QEC decoder message {said}" in message
            assert "give it another name" in message
    body = one
    body.__name__ = "enqueue_syndromes__0_1"
    with pytest.raises(pulse.InvalidValueError, match="can't be named 'enqueue_syndromes__0_1'"):
        pulse.classical(max_latency_ns=0)(body)
    # Names that are no version's are the user's to give.
    for name in (
        "enqueue_syndromes",
        "enqueue_syndromes__0",
        "enqueue_syndromes__01_1",
        "reset_decoder__7_1",
        "my_enqueue_syndromes__0_1",
        "decode__0_1",
    ):
        assert pulse.classical(one, name=name, max_latency_ns=0).name == name


def test_a_version_of_a_function_of_the_users_cannot_take_the_name_of_a_message_version():
    @pulse.classical(max_latency_ns=0)
    def reset_decoder(bits: pulse.Bits) -> None:  # noqa: ARG001
        return None

    @pulse.kernel
    def k(a):
        reset_decoder.submit([readout(get_readout_line(a), square(100, 0.05))])

    dev = _device()
    with pytest.raises(pulse.CompilationError) as refused:
        pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])
    message = str(refused.value)
    assert "the version of reset_decoder for bits=1 would be named 'reset_decoder__1'" in message
    assert "the CUDA-Q QEC decoder message reset_decoder for decoder 1" in message
    assert "rename reset_decoder" in message
    # Outside a kernel the same refusal is of a value.
    with pytest.raises(pulse.InvalidValueError, match="would be named 'reset_decoder__2'"):
        reset_decoder.specialize(bits=2)


def test_the_versions_of_the_messages_keep_their_names():
    @pulse.kernel
    def k(a):
        reset_decoder.submit(3)
        enqueue_syndromes.submit(3, [readout(get_readout_line(a), square(100, 0.05))], 0)
        get_corrections.submit(3, 1, False).result()

    dev = _device()
    low = pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])
    names = [f["name"] for f in low.export_to_controller()["manifest"]["functions"]]
    assert names == ["reset_decoder__3", "enqueue_syndromes__3_1", "get_corrections__3_1"]


# ── Answers: CUDA-Q QEC's decoders ───────────────────────────────────────────


def test_each_path_keeps_its_own_decoder_state():
    @pulse.kernel
    def k(a, b, flag):
        reset_decoder.submit(0)
        for q in (a, b):
            r = get_readout_line(q)
            enqueue_syndromes.submit(0, [readout(r, square(100, 0.05))], 0)
        f = get_drive_line(flag)
        if get_corrections.submit(0, 1, False).result()[0]:
            drive(f, square(40, PI_AMP))

    dev = _device(n=3)
    low = pulse.compile(k, quantum_system=dev).lower(*dev.qudits[:3])
    plus = np.array([1, 1]) / np.sqrt(2)
    state = np.kron(np.array([1, 0]), np.kron(plus, plus))  # a, b in |+>, flag in |0>
    with answering(Decoders(d0=lambda bits: [bits[0]])):
        result = pulse.simulate(low, initial_state=state, levels=2)
    assert len(result.branches) == 4
    # Each round's correction accumulates: the answer is the parity of a and b.
    flag = sum(result.density_matrix[i, i].real for i in range(8) if i & 4)
    assert flag == pytest.approx(0.5, abs=1e-6)
    for branch in result.branches:
        assert [c.function for c in branch.calls] == [
            "reset_decoder",
            "enqueue_syndromes",
            "enqueue_syndromes",
            "get_corrections",
        ]
        assert branch.calls[-1].answer == [(branch.levels[0] + branch.levels[1]) % 2]


def test_emulation_decodes_what_each_shot_enqueued():
    @pulse.kernel
    def k(a, b):
        ra, rb = get_readout_line(a), get_readout_line(b)
        reset_decoder.submit(0)
        for r in range(2):
            syndromes = [readout(ra, square(100, 0.05)), readout(rb, square(100, 0.05))]
            enqueue_syndromes.submit(0, syndromes, r)
        get_corrections.submit(0, 1, False).result()

    dev = _device(n=2)
    low = pulse.compile(k, quantum_system=dev).lower(*dev.qudits[:2])
    with answering(Decoders(d0=_parity)) as decoders:
        run = emulation.emulate(low, shots_count=40, seed=3, service_time="none")
    parities = set()
    for outcomes, calls in zip(run.outcomes, run.calls, strict=True):
        assert calls[-1].answer == [sum(outcomes) % 2]
        parities.add(sum(outcomes) % 2)
    assert parities == {0, 1}
    # Each answer replays its shot's rounds, with their tags, after a reset.
    assert [c[0] for c in decoders.calls[:4]] == ["reset", "reset", "enqueue", "enqueue"]
    assert [c[3] for c in decoders.calls[2:4]] == [0, 1]


def _server(decoders):
    functions = [
        _cudaqx._calls(enqueue_syndromes)._version((0, 2)),
        _cudaqx._calls(get_corrections)._version((0, 1)),
        _cudaqx._calls(reset_decoder)._version((0,)),
    ]
    handlers = {f.name: f.implementation for f in functions}
    _, address, stop = emulation.serve_udp(functions, handlers)
    return functions, emulation.UdpChannel(address), stop


def test_serve_udp_keeps_a_decoders_state_as_the_server_does():
    with answering(Decoders(d0=_parity)):
        (enqueue, get, reset), channel, stop = _server(None)
        request_id = iter(range(100))

        def call(fn, *args):
            answer, _ = channel.call(fn, wire.encode_request(fn, args, next(request_id)))
            status, _, value = wire.decode_answer(fn, answer)
            return status, value

        not_ready = CUDAQX["status"]["NOT_READY"]
        try:
            assert call(reset, 0) == (0, None)
            assert call(get, 0, 1, 0) == (not_ready, None)
            assert call(enqueue, 0, 0, 0, 2, [1, 0]) == (0, None)
            assert call(get, 0, 1, 0) == (0, [1])
            # Corrections accumulate over rounds, as CUDA-Q QEC's do.
            assert call(enqueue, 0, 1, 0, 2, [1, 1]) == (0, None)
            assert call(get, 0, 1, 0) == (0, [1])
            assert call(enqueue, 0, 2, 0, 2, [0, 1]) == (0, None)
            assert call(get, 0, 1, 1) == (0, [0])  # then reset
            assert call(get, 0, 1, 0) == (not_ready, None)
            call(enqueue, 0, 3, 0, 2, [0, 1])
            assert call(get, 0, 1, 0) == (0, [1])
            call(reset, 0)
            assert call(get, 0, 1, 0) == (not_ready, None)
        finally:
            channel.close()
            stop()


def test_a_udp_server_tells_decoders_that_share_a_key_apart():
    functions = [_cudaqx._calls(get_corrections)._version((d, 1)) for d in (0, 1, 2)]
    served = {f.name: (lambda *args, d=f._key_route: [d % 2]) for f in functions}
    _, address, stop = emulation.serve_udp(functions, served)
    channel = emulation.UdpChannel(address)
    try:
        for fn in functions:
            answer, _ = channel.call(fn, wire.encode_request(fn, (fn._key_route, 1, 0)))
            assert wire.decode_answer(fn, answer)[2] == [fn._key_route % 2]
    finally:
        channel.close()
        stop()


def _nothing_enqueued(timeout):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        reset_decoder.submit(0)
        if timeout:
            ok, fix = get_corrections.submit(0, 1, False).result(timeout=200 * pulse.ns)
            if ok:
                drive(d, square(40, PI_AMP))
        elif get_corrections.submit(0, 1, False).result()[0]:
            drive(d, square(40, PI_AMP))

    dev = _device()
    return pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])


def test_a_get_with_nothing_enqueued_is_late():
    with answering(Decoders(d0=_parity)):
        low = _nothing_enqueued(timeout=True)
        (branch,) = pulse.simulate(low, levels=2).branches
        assert not branch.calls[-1].in_time and branch.calls[-1].answer is None
        run = emulation.emulate(low, shots_count=2, service_time="none")
        assert [shot[-1].in_time for shot in run.calls] == [False, False]
        low = _nothing_enqueued(timeout=False)
        with pytest.raises(pulse.ExecutionError, match=r"nothing to decode \(NOT_READY"):
            pulse.simulate(low, levels=2)
        with pytest.raises(pulse.ExecutionError, match=r"NOT_READY \(status 4\)"):
            emulation.emulate(low, service_time="none")


def test_a_get_with_reset_starts_over():
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        enqueue_syndromes.submit(0, [readout(r, square(100, 0.05))], 0)
        get_corrections.submit(0, 1, True).result()
        enqueue_syndromes.submit(0, [readout(r, square(100, 0.05))], 1)
        get_corrections.submit(0, 1, False).result()

    dev = _device()
    low = pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])
    with answering(Decoders(d0=lambda bits: [1])):
        for branch in pulse.simulate(low, levels=2).branches:
            assert [c.answer for c in branch.calls if c.function == "get_corrections"] == [[1], [1]]
        for shot in emulation.emulate(low, shots_count=4, seed=1, service_time="none").calls:
            assert [c.answer for c in shot if c.function.startswith("get_corrections")] == [
                [1],
                [1],
            ]


def test_the_repetition_code_s_corrections():
    links = pulse.ControlSystem(links={"gpu": pulse.Link(feedback_ns=50)})
    enqueue = pulse.classical(qec.enqueue_syndromes, link="gpu")
    get = pulse.classical(qec.get_corrections, link="gpu", max_latency_ns=500)

    @pulse.kernel
    def k(a0, a1):
        s0 = readout(get_readout_line(a0), square(100, 0.05))
        s1 = readout(get_readout_line(a1), square(100, 0.05))
        enqueue.submit(0, [s0, s1], 0)
        get.submit(0, 3, False).result()

    dev = _device(n=2)
    low = pulse.compile(k, quantum_system=dev, control_system=links).lower(*dev.qudits[:2])
    plus = np.array([1, 1]) / np.sqrt(2)
    with answering(Decoders(d0=repetition)):
        result = pulse.simulate(low, initial_state=np.kron(plus, plus), levels=2)
    answers = {tuple(b.levels): b.calls[-1].answer for b in result.branches}
    assert answers == {
        (0, 0): [0, 0, 0],
        (1, 0): [1, 0, 0],
        (1, 1): [0, 1, 0],
        (0, 1): [0, 0, 1],
    }


def test_outside_a_kernel_a_call_asks_the_decoder_now():
    with answering(Decoders(d0=_parity)) as decoders:
        reset_decoder(0)
        with pytest.raises(classical._NotReadyError):
            get_corrections(0, 1, False)
        enqueue_syndromes(0, [1, 1, 1], 4)
        assert get_corrections.submit(0, 1, False).result() == [1]
        assert decoders.calls[-2:] == [("enqueue", 0, (1, 1, 1), 4), ("get", 0, 1, False)]
    with pytest.raises(pulse.InvalidTypeError, match="takes 3 arguments"):
        get_corrections(0, 1)
    with pytest.raises(pulse.InvalidTypeError, match="takes positional arguments"):
        get_corrections.submit(0, return_size=1, reset=False)


def test_a_decoder_s_error_is_an_execution_error():
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        enqueue_syndromes.submit(7, [readout(r, square(100, 0.05))], 0)
        get_corrections.submit(7, 1, False).result()

    dev = _device()
    low = pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])
    with answering(Decoders(d0=_parity)), pytest.raises(pulse.ExecutionError, match="decoder 7"):
        pulse.simulate(low, levels=2)


def test_a_handler_answers_instead_of_the_decoder():
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        enqueue_syndromes.submit(0, [readout(r, square(100, 0.05))], 0)
        get_corrections.submit(0, 2, False).result()

    dev = _device()
    low = pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])
    seen = []
    handlers = {
        "enqueue_syndromes": lambda *wired: seen.append(wired),
        "get_corrections": lambda decoder_id, size, reset: [1] * size,
    }
    (branch, *_) = pulse.simulate(low, levels=2, handlers=handlers).branches
    assert branch.calls[-1].answer == [1, 1]
    assert seen[0][:4] == (0, 0, 0, 1)  # decoder id, tag, mapping row, count


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_without_cudaq_qec_simulating_a_decoder_call_says_so(platform, monkeypatch):
    low = _nothing_enqueued(timeout=True)
    monkeypatch.setattr(_cudaqx, "_backend", None)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setitem(sys.modules, "cudaq_qec", None)
    match = "no macOS build" if platform == "darwin" else "cudaq_qec does not import"
    with pytest.raises(pulse.ExecutionError, match=match):
        pulse.simulate(low, levels=2)
    assert _cudaqx._backend is None


def test_they_pickle():
    @pulse.kernel
    def k(q):
        r = get_readout_line(q)
        enqueue_syndromes.submit(0, [readout(r, square(100, 0.05))], 0)
        get_corrections.submit(0, 1, False).result()

    dev = _device()
    low = pulse.compile(k, quantum_system=dev).lower(dev.qudits[0])
    again = pickle.loads(pickle.dumps(low))
    assert [f.name for f in again.classical] == [f.name for f in low.classical]
    assert [f._dispatch_key for f in again.classical] == [f._dispatch_key for f in low.classical]
    call = pickle.loads(pickle.dumps(get_corrections))
    assert call.max_latency_ns == 100.0
    assert _cudaqx._calls(call)._version((0, 1)).name == "get_corrections__0_1"
    with answering(Decoders(d0=_parity)):
        (branch,) = pulse.simulate(again, levels=2).branches
        assert branch.calls[-1].answer == [0]
