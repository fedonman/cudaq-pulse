# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact propagation of idle stretches.

Each qudit evolves in the frame of its own frequency, so where no drive
plays the Lindbladian depends on time only through the exchange couplings of
detuned qudits: in those frames a coupling g (a^+ b + h.c.) is
g (e^{i D t} a^+ b + h.c.), D the qudits' detuning, and the lowering gives it
as terms that play throughout at the rate D. All of them turn with one
number rotation,

    H_c(t) = R(t) H_c(0) R(t)^+,    R(t) = exp(i t sum_m w_m n_m),

with w_a - w_b = D for each coupled pair. The static terms commute with R,
and R turns each collapse operator c by a phase (R c R^+ = e^{i r t} c), which
the dissipator does not see. In the frame that turns with R the Lindbladian
of an idle stretch is therefore constant, with the Hamiltonian

    H~ = sum_m w_m n_m + S + H_c(0),

and a stretch from t_a to t_b is, exactly,

    rho(t_b) = R(t_b) [exp((t_b - t_a) L~) (R(t_a)^+ rho(t_a) R(t_a))] R(t_b)^+.

The rates w are solved from the terms and every assumption above is checked
on the operators the lowering gave (the terms' rotation against their own
callbacks); a register that fails a check is stepped as before.

Without collapse operators exp(-i H~ t) comes from the eigenvectors of H~,
for any t. With them the superoperator of L~ splits into the blocks of
density-matrix entries it couples (H~ and the jumps conserve excitation
differences), each exponentiated once for a base step and squared: a stretch
of q base steps and a remainder r is exp(r L~) times the kept powers of the
bits of q, a few products of the blocks with the state.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .._native._cudaq_pulse_native import evaluate_callback
from . import _propagate

if TYPE_CHECKING:
    import numpy.typing as npt

    from .simulate import _GpuBackend, _Stepper

try:
    from _blake2 import blake2b as _blake2b
except ImportError:  # pragma: no cover - CPython builds without the _blake2 module
    from hashlib import blake2b as _blake2b

# The lowering gives a term that plays throughout the largest double as its
# duration; a pulse lasts less than this.
_THROUGHOUT_NS = 1e300
# A register with more levels than this keeps no superoperator: its
# (levels)^4 entries would take more memory than stepping saves.
MAX_DISSIPATIVE_DIM = 40
# Relative size below which an entry of a sum of terms is a cancellation.
_ZERO = 1e-13
# Relative disagreement that fails a check of the frame.
_CHECK = 1e-8
# A remainder exponential kept per idle generator for each stretch length.
_REMAINDERS = 64


def plays_throughout(kind: int, params: Sequence[float]) -> bool:
    """Whether a callback term is a constant envelope that never ends,
    turning at its rate: the exchange coupling of detuned qudits."""
    return kind == 1 and len(params) == 10 and params[1] >= _THROUGHOUT_NS


def step_through(
    stepper: _Stepper,
    dims: tuple[int, ...],
    rho: npt.NDArray[np.complex128],
    t0: float,
    steps: int,
    base_ns: float,
) -> npt.NDArray[np.complex128] | None:
    """*rho* evolved over the *steps* of *stepper* from *t0*: drives stepped,
    idle stretches propagated exactly where that costs less than stepping
    them; None, *stepper* untouched, if the register's idle Lindbladian is
    not constant in any frame."""
    from .simulate import _active_ranges

    frame = [d for d in stepper.driven if plays_throughout(d[0], d[1])]
    idle = propagator(dims, stepper.static, frame, stepper.jumps, base_ns)
    if idle is None or idle.start > t0:
        return None
    drives = [d for d in stepper.driven if not plays_throughout(d[0], d[1])]
    h = stepper.h
    for lo, hi in [*_active_ranges(drives, t0, h, steps), (steps, steps)]:
        n = lo - stepper.done
        if n > 0 and idle.cost(n * h) < n * stepper._rk4_cost:
            rho = idle.advance(rho, t0 + stepper.done * h, n * h)
            stepper.done = lo
        if hi > stepper.done:
            rho = stepper.play(rho, t0, stepper.done, hi)
    return rho


# The fewest steps of an idle stretch that the GPU path propagates on the
# host: each part the stretch splits the rest into is lowered and compiled
# on its own (tens of milliseconds), and a step costs about a millisecond.
GPU_IDLE_STEPS = 1024


def through_gpu(
    backend: _GpuBackend,
    states: list[Any],
    entries: list[list[int]],
    t0: float,
    t1: float,
    steps: int,
) -> list[Any] | None:
    """*states*, which evolve under the same operators along *entries*,
    evolved from *t0* to *t1* in *steps* steps by *backend*, the GPU path:
    their idle stretches of at least ``GPU_IDLE_STEPS`` steps propagated
    exactly on the host, state by state, every other stretch run on the GPU
    as one batch with the steps it has in the whole; None if there is no
    such stretch or the register's idle Lindbladian is not constant in any
    frame."""
    from .simulate import (
        _active_ranges,
        _bounded,
        _DivergedError,
        _full_operator,
        _interval_attributes,
    )

    with backend.lock:
        backend.module.set_attributes(backend.base, _interval_attributes(entries, t0, t1, steps))
        _, hamiltonian, collapse, *_ = backend.module.operator_program()
    # What ``propagator`` declines for its size, before any operator is built.
    if collapse and backend.dim > MAX_DISSIPATIVE_DIM:
        return None
    # The stretches follow from the terms' supports alone, so the operators
    # on the whole register are built only once a stretch is long enough,
    # and those of the drives not at all.
    drives: list[tuple[int, Any, Any]] = []
    frame_terms: list[tuple[int, Any, Any]] = []
    static_terms: list[tuple[float, float, Any]] = []
    for re, im, kind, params, factors in hamiltonian:
        if kind:
            start, end = _propagate.support(params)
            if end > t0 and start < t1:
                terms = frame_terms if plays_throughout(kind, params) else drives
                terms.append((kind, params, factors))
        else:
            static_terms.append((re, im, factors))
    h = (t1 - t0) / steps
    parts = []
    pending = done = 0
    for lo, hi in [*_active_ranges(drives, t0, h, steps), (steps, steps)]:
        if lo - done >= GPU_IDLE_STEPS:
            if done > pending:
                parts.append((pending, done, False))
            parts.append((done, lo, True))
            pending = lo
        done = max(done, hi)
    if not any(idle for _, _, idle in parts):
        return None
    dims = backend.dims
    static = np.zeros((backend.dim, backend.dim), dtype=np.complex128)
    for re, im, factors in static_terms:
        static += (re + 1j * im) * _full_operator(factors, dims)
    frame = [(kind, params, _full_operator(factors, dims)) for kind, params, factors in frame_terms]
    jumps = [(re + 1j * im) * _full_operator(f, dims) for re, im, _, _, f in collapse]
    idle = propagator(dims, static, frame, jumps, backend.dt_ns)
    if idle is None or idle.start > t0:
        return None
    if steps > pending:
        parts.append((pending, steps, False))
    for lo, hi, quiet in parts:
        start, end = t0 + lo * h, t1 if hi == steps else t0 + hi * h
        if quiet:
            states = [idle.advance(state, start, end - start) for state in states]
            continue
        mixed = states[0].ndim == 2
        attributes = _interval_attributes(entries, start, end, hi - lo)[:-1]
        if mixed and not backend.dissipative:
            attributes += ", qop.mixed_state"
        with backend.lock:
            backend.module.set_attributes(backend.base, attributes + "}")
            text = backend.module.print()
        results = backend._execute(text, states, mixed)
        for result, state in zip(results, states, strict=True):
            if not _bounded(result, state):
                raise _DivergedError(h)
        states = results
    return states


def propagator(
    dims: tuple[int, ...],
    static: npt.NDArray[np.complex128],
    frame: list[tuple[int, Any, npt.NDArray[np.complex128]]],
    jumps: list[npt.NDArray[np.complex128]],
    base_ns: float,
) -> Idle | None:
    """The idle propagator of a register whose static Hamiltonian is
    *static*, whose terms that play throughout are *frame* (kind, params,
    operator) and whose collapse operators are *jumps*; None where the
    frame does not make it constant or the register is too large."""
    dim = static.shape[0]
    if jumps and dim > MAX_DISSIPATIVE_DIM:
        return None
    digest = _blake2b(digest_size=16)
    digest.update(np.asarray(dims, dtype=np.int64).tobytes())
    digest.update(static.tobytes())
    for _, params, op in frame:
        digest.update(np.asarray(params, dtype=float).tobytes())
        digest.update(op.tobytes())
    for c in jumps:
        digest.update(c.tobytes())
    key = (base_ns, digest.digest())
    with _kept_lock:
        if key in _kept:
            return _kept[key]
    made = _build(dims, static, frame, jumps, base_ns)
    with _kept_lock:
        if len(_kept) >= _KEPT:
            del _kept[next(iter(_kept))]
        _kept[key] = made
    return made


# The idle propagators (or None: no frame) of the registers simulated last.
_KEPT = 8
_kept: dict[Any, Idle | None] = {}
_kept_lock = threading.Lock()


def forget() -> None:
    """Forget every kept idle propagator."""
    with _kept_lock:
        _kept.clear()


def _levels(dims: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """The level of each mode (rows) in each basis state (columns); mode 0
    varies fastest."""
    index = np.arange(math.prod(dims))
    rows = []
    stride = 1
    for d in dims:
        rows.append((index // stride) % d)
        stride *= d
    return np.array(rows, dtype=np.int64)


def _build(
    dims: tuple[int, ...],
    static: npt.NDArray[np.complex128],
    frame: list[tuple[int, Any, npt.NDArray[np.complex128]]],
    jumps: list[npt.NDArray[np.complex128]],
    base_ns: float,
) -> Idle | None:
    dim = static.shape[0]
    # Each term's coefficient, (Re or Im of E e^{i(phase + D (t - start))}) / 2,
    # is a e^{iDt} + conj(a) e^{-iDt}: the parts of H_c by rate.
    parts: dict[float, npt.NDArray[np.complex128]] = {}
    for _, params, op in frame:
        e = complex(params[2], params[3]) * complex(
            math.cos(params[7] - params[9] * params[0]), math.sin(params[7] - params[9] * params[0])
        )
        a = 0.25 * e if params[8] == 0.0 else e / 4j
        rate = float(params[9])
        parts[rate] = parts.get(rate, 0) + a * op
        parts[-rate] = parts.get(-rate, 0) + a.conjugate() * op
    coupled = sum(parts.values(), np.zeros((dim, dim), dtype=np.complex128))
    scale = max(float(np.abs(coupled).max()), float(np.abs(static).max()), 1e-300)
    levels = _levels(dims)
    rate_of = np.full((dim, dim), np.nan)
    for rate, part in parts.items():
        on = np.abs(part) > _ZERO * scale
        if np.any(on & ~np.isnan(rate_of) & (rate_of != rate)):
            return None
        rate_of[on] = rate
    off = ~np.eye(dim, dtype=bool)
    rate_of[np.isnan(rate_of) & (np.abs(static) > _ZERO * scale) & off] = 0.0
    rows, columns = np.nonzero(~np.isnan(rate_of) & off)
    rates = rate_of[rows, columns]
    if rows.size:
        differences = (levels[:, rows] - levels[:, columns]).T.astype(float)
        w = np.linalg.lstsq(differences, rates, rcond=None)[0]
        fastest = max(1.0, float(np.abs(rates).max()))
        if float(np.abs(differences @ w - rates).max()) > _CHECK * fastest:
            return None
    else:
        w = np.zeros(len(dims))
    phases = w @ levels
    turn = phases[:, None] - phases[None, :]
    tolerance = _CHECK * max(1.0, float(np.abs(w).max()))
    for c in jumps:
        on = np.abs(c) > _ZERO * float(np.abs(c).max())
        if on.any() and np.ptp(turn[on]) > tolerance:
            return None
    coupled[np.abs(coupled) <= _ZERO * scale] = 0.0
    # The frame against the terms' own callbacks, at two times well apart.
    fastest = max(float(np.abs(w).max()), 1e-3)
    start = max((float(params[0]) for _, params, _ in frame), default=-math.inf)
    for t in (start + 1.3 / fastest, start + 977.0 / fastest) if frame else ():
        played = np.zeros((dim, dim), dtype=np.complex128)
        for kind, params, op in frame:
            played += evaluate_callback(kind, params, [t])[0] * op
        if float(np.abs(played - coupled * np.exp(1j * turn * t)).max()) > _CHECK * scale:
            return None
    hamiltonian = np.diag(phases).astype(np.complex128) + static + coupled
    return Idle(hamiltonian, phases, jumps, base_ns, start)


def _blocks(adjacent: npt.NDArray[np.bool_]) -> list[npt.NDArray[np.int64]]:
    """The connected components of a symmetric adjacency matrix."""
    n = adjacent.shape[0]
    label = np.full(n, -1)
    blocks: list[npt.NDArray[np.int64]] = []
    for seed in range(n):
        if label[seed] >= 0:
            continue
        label[seed] = len(blocks)
        members = [np.array([seed])]
        frontier = members[0]
        while frontier.size:
            frontier = np.flatnonzero(adjacent[frontier].any(axis=0) & (label < 0))
            label[frontier] = len(blocks)
            members.append(frontier)
        blocks.append(np.sort(np.concatenate(members)))
    return blocks


class Idle:
    """The exact propagator of an idle stretch (see the module docstring)."""

    def __init__(
        self,
        hamiltonian: npt.NDArray[np.complex128],
        phases: npt.NDArray[np.float64],
        jumps: list[npt.NDArray[np.complex128]],
        base_ns: float,
        start: float,
    ) -> None:
        self.dim = hamiltonian.shape[0]
        self.phases = phases
        self.turn = phases[:, None] - phases[None, :]
        self.start = start
        self.base_ns = base_ns
        self.dissipative = bool(jumps)
        self._lock = threading.Lock()
        if not jumps:
            # exp(-i H~ t) = V e^{-i E t} V^+, from the eigenvectors of each
            # block of H~.
            self.vectors = np.zeros((self.dim, self.dim), dtype=np.complex128)
            self.energies = np.zeros(self.dim)
            for block in _blocks(hamiltonian != 0):
                values, vectors = np.linalg.eigh(hamiltonian[np.ix_(block, block)])
                self.energies[block] = values
                self.vectors[np.ix_(block, block)] = vectors
            return
        generator = _propagate.liouvillian(
            hamiltonian,
            _propagate.dissipator(jumps, self.dim, b"".join(c.tobytes() for c in jumps)),
        )
        self.blocks = _blocks((generator != 0) | (generator != 0).T)
        self.generators = [generator[np.ix_(b, b)] for b in self.blocks]
        # powers[j][k]: block k of exp(2^j base_ns L~).
        self.powers: list[list[npt.NDArray[np.complex128]]] = []
        self.remainders: dict[float, list[npt.NDArray[np.complex128]]] = {}
        self._entries = sum(len(b) ** 2 for b in self.blocks)
        self._cubes = float(sum(len(b) ** 3 for b in self.blocks))

    def cost(self, duration_ns: float) -> float:
        """An estimate, in the units of ``_Stepper``'s, of advancing over
        *duration_ns*."""
        if not self.dissipative:
            return 8.0 * float(self.dim) ** 3 + 5e4
        q = int(duration_ns // self.base_ns)
        missing = max(0, q.bit_length() - len(self.powers))
        cost = missing * self._cubes + bin(q).count("1") * self._entries + 5e4
        if not self.powers:
            cost += 14 * self._cubes
        if duration_ns - q * self.base_ns not in self.remainders:
            cost += 14 * self._cubes
        return cost

    def advance(self, state: Any, t: float, duration_ns: float) -> Any:
        """*state* at *t*, a density matrix or (without collapse operators)
        a vector, evolved over the idle stretch of *duration_ns* that
        starts there."""
        end = t + duration_ns
        if state.ndim == 1:
            psi = np.exp(-1j * self.phases * t) * state
            psi = self.vectors @ (
                np.exp(-1j * self.energies * duration_ns) * (self.vectors.conj().T @ psi)
            )
            return np.exp(1j * self.phases * end) * psi
        rho = state * np.exp(-1j * self.turn * t)
        if not self.dissipative:
            inner = self.vectors.conj().T @ rho @ self.vectors
            inner *= np.exp(-1j * (self.energies[:, None] - self.energies[None, :]) * duration_ns)
            rho = self.vectors @ inner @ self.vectors.conj().T
        else:
            rho = self._dissipate(rho, duration_ns)
        return rho * np.exp(1j * self.turn * end)

    def _dissipate(self, rho: Any, duration_ns: float) -> Any:
        q = int(duration_ns // self.base_ns)
        remainder = duration_ns - q * self.base_ns
        with self._lock:
            while len(self.powers) < max(1, q.bit_length()):
                if not self.powers:
                    self.powers.append([_propagate.expm(g * self.base_ns) for g in self.generators])
                else:
                    self.powers.append([p @ p for p in self.powers[-1]])
            first = self.remainders.get(remainder)
            if first is None and remainder > 0:
                if len(self.remainders) >= _REMAINDERS:
                    self.remainders.clear()
                first = [_propagate.expm(g * remainder) for g in self.generators]
                self.remainders[remainder] = first
            powers = [self.powers[j] for j in range(q.bit_length()) if q >> j & 1]
        flat = rho.reshape(-1)
        out = np.empty_like(flat)
        for k, block in enumerate(self.blocks):
            x = flat[block]
            if first is not None:
                x = first[k] @ x
            for p in powers:
                x = p[k] @ x
            out[block] = x
        return out.reshape(self.dim, self.dim)
