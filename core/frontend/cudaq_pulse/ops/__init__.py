# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel DSL operations for pulse programming.

These functions form the "language extension" vocabulary used as bare
names inside ``@pulse.kernel`` decorated functions.  During bytecode
tracing the decorator intercepts calls to these names and lowers them
to MLIR ops.  Called outside a kernel, each raises ``CompilationError``
naming it.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

from .._errors import CompilationError, InvalidTypeError
from ..kernel.ir_builder import Parameter

if TYPE_CHECKING:
    from .._units import Time
    from ..gates import Gate

__all__ = [
    "cosine",
    "custom",
    "custom_samples",
    "drag",
    "drive",
    "gaussian",
    "gaussian_square",
    "get_drive_line",
    "get_readout_line",
    "readout",
    "set_frequency",
    "set_phase",
    "shift_frequency",
    "shift_phase",
    "square",
    "sync",
    "tanh_ramp",
    "wait",
    "wf_add",
    "wf_mul",
    "wf_neg",
    "wf_scale",
    "wf_sub",
]

Numeric = int | float | Parameter

# ── Opaque type aliases (for IDE hover-docs, not enforced at runtime) ────


class Waveform:
    """Opaque waveform handle returned by envelope constructors."""


class Tone:
    """Opaque tone handle for phase and frequency operations."""


class Line:
    """A qudit's drive line, from ``get_drive_line()``, or its readout line,
    from ``get_readout_line()``. A line is one value, not a pair: it is not
    iterable or unpackable."""

    @property
    def tone(self) -> Tone:
        """The line's own tone, at its qudit's frequency: the tone of the
        default transition of the system's model. ``drive(line, waveform)``
        and ``readout(line, waveform)`` play on it, and the phase and
        frequency ops change it, as in ``shift_phase(line.tone, phase)``."""
        return _require_context("Line.tone")

    @property
    def tones(self) -> Mapping[tuple[int, int], Tone]:
        """The tone of each transition of the line's qudit, by its
        ``(lower, upper)`` levels, as the system's model numbers them: read
        only. The default transition's is ``line.tone``; another's starts
        at that transition's frequency with phase 0, as in
        ``drive(line, waveform, tone=line.tones[(1, 2)])``. A readout line
        has only its own tone. A transition the qudit does not have raises
        a ``KeyError`` that lists those it has."""
        return _require_context("Line.tones")

    def get_calibration(self, gate: Gate, transition: tuple[int, int] = (0, 1)) -> Any:
        """The calibrated pulse of *gate* (a ``pulse.gates.Gate``) on the
        line's qudit, from the quantum system the kernel compiles for: a
        ``DragPulse`` for ``X``, ``X90``, ``Y`` and ``Y90`` and a
        ``VirtualZ`` for ``Z``, ``S`` and ``T`` on a drive line, a
        ``ReadoutPulse`` for ``READOUT`` on a readout line, derived as
        ``pulse.gates`` describes where not published. All play:
        ``drive(line, line.get_calibration(pulse.gates.X))``, and
        ``drive(line, line.get_calibration(pulse.gates.T))`` shifts the
        phase of the line's tone by pi / 4, with no pulse; their fields are
        numbers known while the kernel compiles. *transition*
        names another transition's calibration, which must be published;
        play it at that transition's tone, ``line.tones[transition]``."""
        return _require_context("Line.get_calibration")


class MeasurementResult(int):
    """One measurement from ``readout()``: its value is the outcome index
    (0 or 1 for a qubit, 0 to d-1 for a qudit). Kernels pass it to classical
    functions as ``pulse.Bits``, ``list[int]`` or ``complex``."""


class RoundDirection(enum.StrEnum):
    """Which way ``round_duration`` rounds a length onto its line's grid; the
    strings ``"up"``, ``"down"`` and ``"nearest"`` mean the same."""

    UP = "up"
    DOWN = "down"
    NEAREST = "nearest"


def round_duration(
    line: Any, duration: float, direction: RoundDirection | str = RoundDirection.UP
) -> int:
    """*duration*, an int number of program cycles or a time
    (``10.5 * pulse.ns``), rounded to whole timesteps of *line*: the nearest
    length (or start) its controller can time, ``"up"`` (default),
    ``"down"`` or ``"nearest"``, in program cycles.

    Call it in a kernel with a line from ``get_drive_line`` or
    ``get_readout_line``; *duration* must be known while the kernel compiles.
    A line whose timestep is one cycle takes every whole cycle, so
    ``round_duration(line, n)`` is ``ceil(n)``.
    """
    # In a kernel, a call whose line is a line never runs this; one whose
    # line is not reaches it as a plain call and fails like a bad argument.
    raise InvalidTypeError(
        "round_duration() takes a line from get_drive_line or get_readout_line, inside a "
        f"@cudaq_pulse.kernel function; got {line!r}"
    )


def _require_context(op_name: str) -> NoReturn:
    raise CompilationError(
        f"{op_name}() must be called inside a @cudaq_pulse.kernel function",
        code="unsupported_construct",
    )


# ── Channel access ───────────────────────────────────────────────────────


def get_drive_line(qudit: Any, *others: Any) -> Line:
    """The drive line of a qudit.

    A qudit has one drive line for the whole program: later calls, also
    from called kernels, return it as it now is, so pulses on it play in
    sequence. Its tone is ``line.tone``.

    On a control system that declares a broadcast drive line,
    ``DriveLine([0, 1, 2], broadcast=True)``, the line of any of its qudits
    is that one line, whose one tone reaches all of them, and a drive on it
    drives every one. ``get_drive_line(a, b, c)``, with all the line's
    qudits, gives the same line and says what it reaches.

    Args:
        qudit: The qudit whose drive line to use.
        others: The other qudits of a broadcast drive line, when the line
            is named by all its qudits.
    """
    return _require_context("get_drive_line")


def get_readout_line(qudit: Any) -> Line:
    """The readout line of a qudit.

    Like the drive line, a qudit has one readout line for the whole
    program. Its tone is ``line.tone``, at the frequency its readout gives
    (a ``DispersiveReadout``'s ``frequency_hz``), else at its 0-1
    frequency.

    Args:
        qudit: The qudit whose readout line to use.
    """
    return _require_context("get_readout_line")


# ── Scheduling ops ──────────────────────────────────────────────────────


def drive(line: Line, waveform: Waveform, *, tone: Tone | None = None) -> None:
    """Play a waveform on a drive line.

    Args:
        line: Drive line from ``get_drive_line()``.
        waveform: Waveform envelope to play (e.g. from ``gaussian()``).
        tone: The tone to play it at; by default the line's own,
            ``line.tone``. Another transition's, ``line.tones[(1, 2)]``,
            or another line's tone retunes the drive.
    """
    _require_context("drive")


def readout(line: Line, waveform: Waveform, *, tone: Tone | None = None) -> MeasurementResult:
    """Acquire a measurement through a readout line.

    Args:
        line: Readout line from ``get_readout_line()``.
        waveform: Readout waveform envelope.
        tone: The tone to play it at; by default the line's own,
            ``line.tone``.

    Returns:
        Measurement result handle.
    """
    _require_context("readout")


def wait(
    target: Line, duration: Numeric, bounds: tuple[int | Time, int | Time | None] | None = None
) -> None:
    """Insert an idle delay on a line.

    Args:
        target: Drive or readout line.
        duration: Wait duration: an int number of clock cycles, or a time
            (``20 * pulse.ns``) that is a whole number of them; a plain float
            is an error. It may be computed from ``float`` kernel
            parameters with ``+``, ``-``, ``*`` and ``/``; the kernel then
            compiles once, and each call rounds the length to the nearest
            cycle (ties up) and schedules with it. When it is computed from
            the answer of a classical call, the wait is a variable region:
            every line syncs before it, and everything after it shifts with
            its length, which is known only when the program runs; an
            answer in nanoseconds is ``f.result() * pulse.ns``, converted
            with one multiply by a constant.
        bounds: ``(lowest, highest)``, each in clock cycles or a time;
            ``highest`` may be ``None``. A call whose parameter-derived
            length falls outside them raises ``ValueError``; a length from
            an answer is rounded and clamped to them. Without bounds the
            length must not be negative. Scheduling without the length puts
            the wait at ``lowest``.
    """
    _require_context("wait")


def sync(*targets: Line) -> None:
    """Synchronize multiple lines to a common time point.

    All lines are padded to the latest time among them before
    subsequent operations proceed. When their controllers start operations
    only every few program cycles, they meet at the next time that is a
    whole number of every line's timesteps.

    Args:
        targets: Two or more drive/readout lines to synchronize.
    """
    _require_context("sync")


# ── Phase / frequency ops ───────────────────────────────────────────────


def shift_phase(tone: Tone, phase: Numeric) -> Tone:
    """Advance a tone's frame phase: a virtual Z rotation by *phase*.

    Following McKay et al., PRA 96, 022330 (2017), a pulse played after it
    turns about an axis rotated by ``-phase``.

    Args:
        tone: Tone handle, such as ``line.tone``.
        phase: Phase increment in radians.
    """
    _require_context("shift_phase")


def set_phase(tone: Tone, phase: Numeric) -> Tone:
    """Set the frame phase of a tone, with the sign of ``shift_phase``.

    Args:
        tone: Tone handle.
        phase: Absolute phase in radians.
    """
    _require_context("set_phase")


def shift_frequency(tone: Tone, frequency: Numeric) -> Tone:
    """Add a relative frequency offset to a tone.

    Args:
        tone: Tone handle.
        frequency: Frequency offset in Hz.
    """
    _require_context("shift_frequency")


def set_frequency(tone: Tone, frequency: Numeric) -> Tone:
    """Set the absolute frequency of a tone.

    Args:
        tone: Tone handle.
        frequency: Absolute frequency in Hz.
    """
    _require_context("set_frequency")


# ── Waveform constructors ───────────────────────────────────────────────


def gaussian(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a Gaussian envelope waveform.

    The envelope is ``amplitude * exp(-(t - duration / 2)**2 / (2 * sigma**2))``
    at time ``t`` from the start of the pulse. It is cut off at the ends of
    *duration*, with no offset subtracted: with ``duration = 4 * sigma`` it
    starts and ends at about 14% of *amplitude*.

    Args:
        duration: Pulse duration: an int number of clock cycles, or a time
            (``20 * pulse.ns``).
        amplitude: Peak amplitude, with no limit but the controller's
            (``Controller.max_amplitude``).
        sigma: Standard deviation in clock cycles, or a time.

    Returns:
        Waveform value for use with ``drive()`` or waveform arithmetic.
    """
    return _require_context("gaussian")


def square(duration: Numeric, amplitude: Numeric | complex | Sequence[float]) -> Waveform:
    """Create a flat-top (square) envelope waveform.

    Args:
        duration: Pulse duration: an int number of clock cycles, or a time
            (``20 * pulse.ns``).
        amplitude: Constant amplitude: a real number, a complex number
            ``re + im*1j``, or ``[re, im]``, with no limit but the
            controller's (``Controller.max_amplitude``).

    Returns:
        Waveform value.
    """
    return _require_context("square")


def drag(duration: Numeric, amplitude: Numeric, sigma: Numeric, beta: Numeric) -> Waveform:
    """Create a DRAG (Derivative Removal by Adiabatic Gate) waveform.

    The envelope is ``G(t) + i beta dG/dt`` with ``G`` the Gaussian (Motzoi
    et al., PRL 103, 110501 (2009)). Leakage to the second excited level of
    a transmon with anharmonicity ``alpha`` is smallest near
    ``beta = -1 / alpha``, with ``alpha`` in radians per clock cycle.

    Args:
        duration: Pulse duration: an int number of clock cycles, or a time
            (``20 * pulse.ns``).
        amplitude: Peak amplitude, with no limit but the controller's
            (``Controller.max_amplitude``).
        sigma: Gaussian standard deviation in clock cycles, or a time.
        beta: DRAG coefficient in clock cycles, or a time.

    Returns:
        Waveform value.
    """
    return _require_context("drag")


def cosine(duration: Numeric, amplitude: Numeric) -> Waveform:
    """Create a raised-cosine envelope waveform.

    Args:
        duration: Pulse duration: an int number of clock cycles, or a time
            (``20 * pulse.ns``).
        amplitude: Peak amplitude, with no limit but the controller's
            (``Controller.max_amplitude``).

    Returns:
        Waveform value.
    """
    return _require_context("cosine")


def tanh_ramp(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a hyperbolic-tangent ramp waveform.

    Args:
        duration: Pulse duration: an int number of clock cycles, or a time
            (``20 * pulse.ns``).
        amplitude: Peak amplitude, with no limit but the controller's
            (``Controller.max_amplitude``).
        sigma: Rise/fall steepness in clock cycles, or a time.

    Returns:
        Waveform value.
    """
    return _require_context("tanh_ramp")


def gaussian_square(
    duration: Numeric,
    amplitude: Numeric,
    sigma: Numeric,
    width: Numeric,
) -> Waveform:
    """Create a Gaussian-square (flat-top Gaussian) waveform.

    A square pulse with Gaussian rise and fall edges.

    Args:
        duration: Total pulse duration: an int number of clock cycles, or
            a time (``20 * pulse.ns``).
        amplitude: Peak amplitude, with no limit but the controller's
            (``Controller.max_amplitude``).
        sigma: Gaussian edge standard deviation in clock cycles, or a time.
        width: Flat-top width: an int number of clock cycles, or a time.

    Returns:
        Waveform value.
    """
    return _require_context("gaussian_square")


def custom(duration: int | Time, envelope_fn: Callable[[float], float]) -> Waveform:
    """Create a waveform by sampling a Python function while the kernel is traced.

    *envelope_fn* is called once per sample of the lines it plays on, at the
    middle of each: ``t = 0.5, 1.5, ..., duration - 0.5``, in clock cycles
    from the start of the pulse, when a line plays one sample per clock
    cycle. A controller with its own clock or several samples per cycle
    plays at its own rate, and the envelope is sampled at that rate instead
    (every 2 clock cycles at half the clock: ``t = 1, 3, ...``). Each value
    is held until the next, like ``custom_samples`` of the values (which is
    what the kernel's IR records), so the waveform compiles, schedules and
    simulates like the built-in envelopes. A function that matches a
    built-in envelope, such as
    ``lambda t: a * math.exp(-((t - duration / 2) ** 2) / (2 * sigma**2))``
    for ``gaussian(duration, a, sigma)``, gives the same waveform up to that
    sampling.

    The duration must be known while tracing (an ``int`` or a time, as an
    argument or a constant) and at most 252 samples (250 at a controller's own rate);
    play a longer envelope as consecutive drives. The function must return
    real numbers that do not depend on kernel parameters: scale the waveform
    by a parameter with
    ``wf_scale(custom(duration, f), amplitude)``. Define the function outside
    the kernel: a lambda or ``def`` in a kernel body raises
    ``CompilationError``.

    Args:
        duration: Pulse duration: an int number of clock cycles, or a time
            (``20 * pulse.ns``).
        envelope_fn: A function ``f(t) -> float`` of the time in clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("custom")


def custom_samples(samples: Sequence[float], sample_rate_ghz: float | None = None) -> Waveform:
    """Create a waveform from pre-computed sample data.

    Each sample is held until the next. The samples are made for one rate,
    and play only on lines at that rate: by default one per clock cycle of
    the quantum system (``QuantumSystem.clock_ghz``), or *sample_rate_ghz*. A line plays its
    controller's rate, its ``sample_rate_ghz``; samples made for another rate are a
    ``CompilationError``, never a change in how long they last. Resample
    them for the line, or use ``custom``, which samples at the line's rate.

    Args:
        samples: Non-empty array-like of real envelope sample values.
        sample_rate_ghz: The rate the samples were made for, in GHz; they
            must last a whole number of clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("custom_samples")


# ── Waveform arithmetic ─────────────────────────────────────────────────


def wf_add(left: Waveform, right: Waveform) -> Waveform:
    """Add two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform (must have same duration as *left*).

    Returns:
        Combined waveform ``left + right``.
    """
    return _require_context("wf_add")


def wf_sub(left: Waveform, right: Waveform) -> Waveform:
    """Subtract two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform.

    Returns:
        Difference waveform ``left - right``.
    """
    return _require_context("wf_sub")


def wf_mul(left: Waveform, right: Waveform) -> Waveform:
    """Multiply two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform.

    Returns:
        Product waveform ``left * right``.
    """
    return _require_context("wf_mul")


def wf_scale(waveform: Waveform, scalar: Numeric, /) -> Waveform:
    """Scale a waveform by a constant factor.

    Args:
        waveform: Waveform to scale.
        scalar: Real scaling factor.

    Returns:
        Scaled waveform ``scalar * waveform``.
    """
    return _require_context("wf_scale")


def wf_neg(waveform: Waveform) -> Waveform:
    """Negate a waveform (flip sign of all samples).

    Args:
        waveform: Waveform to negate.

    Returns:
        Negated waveform ``-waveform``.
    """
    return _require_context("wf_neg")
