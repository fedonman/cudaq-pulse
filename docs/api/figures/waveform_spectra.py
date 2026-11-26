# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the waveform figure of the pulse operations reference.

Usage: python docs/api/figures/waveform_spectra.py

Needs matplotlib (the ``[viz]`` extra). Writes ``waveform_spectra.svg`` and
its ``-dark`` twin next to this file: four waveform constructors, each
compiled in a one-pulse kernel for ``transmon_surface17`` and sampled with
``cudaq_pulse.controller.sample_waveform``, above the spectrum of those
samples. The reference page has no code of its own, so the four calls are
the ones below; the amplitude, the length and the band each panel shades
are read from the samples.
"""

import sys
from pathlib import Path

import cudaq_pulse as pulse
import matplotlib.pyplot as plt
import numpy as np
from cudaq_pulse.controller import sample_waveform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials" / "figures"))
from _style import WIDTH_IN, save_themed  # noqa: E402

# Four of the calls of the operations guide's waveform figure: the envelopes with no Q.
CALLS = [
    "square(40, 0.3)",
    "gaussian_square(40, 0.3, 4, 20)",
    "gaussian(40, 0.3, 10)",
    "cosine(40, 0.3)",
]
SHARE = 0.99
SPAN_MHZ = 500.0
FLOOR_DB = -60.0


def envelope(call: str, qs: pulse.QuantumSystem) -> np.ndarray:
    """The samples ``drive(line, <call>)`` puts on a line, one per clock cycle."""
    namespace: dict = {}
    source = f"def one_pulse(q):\n    drive(get_drive_line(q), {call})\n"
    exec(compile(source, "<figure>", "exec"), namespace)
    program = pulse.compile(pulse.kernel(namespace["one_pulse"]), quantum_system=qs)
    (event,) = program.lower(qs.qudits[0]).export_to_controller()["events"]
    return sample_waveform(event["waveform"], event["duration_cycles"])


def spectrum(samples: np.ndarray, clock_ghz: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The frequencies from the tone (MHz) and the squared magnitude of the
    samples' Fourier transform there, at *n* frequencies up to half the clock."""
    mhz = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / (clock_ghz * 1e3)))
    power = np.abs(np.fft.fftshift(np.fft.fft(samples, n))) ** 2
    return mhz, power


def band_mhz(mhz: np.ndarray, power: np.ndarray, share: float = SHARE) -> float:
    """The smallest distance from the tone within which *share* of the energy lies."""
    order = np.argsort(np.abs(mhz), kind="stable")
    inside = np.cumsum(power[order]) / power.sum()
    return float(np.abs(mhz[order][np.searchsorted(inside, share)]))


def waveform_spectra() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    shapes = []
    for call in CALLS:
        samples = envelope(call, qs)
        assert np.max(np.abs(samples.imag)) == 0.0, f"{call} has a Q quadrature to draw"
        # The band is measured on a fine grid; the curve is drawn on a coarser one.
        band = band_mhz(*spectrum(samples, qs.clock_ghz, 1 << 15))
        shapes.append((call, samples.real, *spectrum(samples, qs.clock_ghz, 1 << 11), band))
    (cycles,) = {len(samples) for _, samples, *_ in shapes}
    (amplitude,) = {round(float(samples.max()), 6) for _, samples, *_ in shapes}
    length_ns = cycles / qs.clock_ghz

    def figure(p):
        fig, grid = plt.subplots(
            2,
            len(shapes),
            figsize=(WIDTH_IN, 3.7),
            sharey="row",
            gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.7, "wspace": 0.2},
        )
        for (top, low), (call, samples, mhz, power, band) in zip(grid.T, shapes, strict=True):
            # Each sample is held for its clock cycle.
            edges = np.arange(len(samples) + 1) / qs.clock_ghz
            held = np.append(samples, samples[-1])
            top.fill_between(edges, held, step="post", color=p.drive, alpha=p.fill_alpha, lw=0)
            top.stairs(samples, edges, color=p.drive, lw=1.4)
            top.set_title(call.split("(")[0], color=p.text, pad=5)
            top.set_xlim(0, length_ns)
            top.set_xticks([0, length_ns / 2, length_ns])
            top.set_ylim(0, amplitude * 1.13)
            top.set_yticks([0, amplitude])
            top.spines["bottom"].set_color(p.rule)

            keep = np.abs(mhz) <= SPAN_MHZ
            db = 10 * np.log10(np.maximum(power / power.max(), 1e-12))
            low.axvspan(-band, band, color=p.muted, alpha=0.13, lw=0)
            low.plot(mhz[keep], np.maximum(db[keep], FLOOR_DB - 5), color=p.drive, lw=1.2)
            low.set_xlim(-SPAN_MHZ, SPAN_MHZ)
            low.set_xticks([-400, 0, 400])
            low.set_ylim(FLOOR_DB, 4)
            low.set_yticks([0, -20, -40, -60])
            low.spines["bottom"].set_color(p.rule)
            low.set_title(f"{SHARE:.0%} within ±{band:.0f} MHz", color=p.muted, fontsize=8.5, pad=4)
        grid[0, 0].set_ylabel("amplitude")
        grid[1, 0].set_ylabel("spectrum (dB)")
        grid[0, -1].set_xlabel("time (ns)", loc="right", labelpad=1)
        grid[1, -1].set_xlabel("frequency from the tone (MHz)", loc="right", labelpad=1)
        return fig

    # What the band leaves out: how high each spectrum's skirt still peaks at the edge of
    # the plot, and from where on it stays under the floor.
    skirt_db, quiet_mhz = {}, {}
    for call, _, mhz, power, _ in shapes:
        db = 10 * np.log10(np.maximum(power / power.max(), 1e-12))
        name = call.split("(")[0]
        skirt_db[name] = float(
            db[(np.abs(mhz) >= 0.8 * SPAN_MHZ) & (np.abs(mhz) <= SPAN_MHZ)].max()
        )
        above = np.abs(mhz)[(db > FLOOR_DB) & (np.abs(mhz) <= SPAN_MHZ)]
        quiet_mhz[name] = float(above.max())
    assert skirt_db["gaussian"] > FLOOR_DB + 10 and quiet_mhz["cosine"] < 0.8 * SPAN_MHZ
    # What the page's caption says of the gaussian: the narrowest band of the four.
    assert min(shapes, key=lambda shape: shape[-1])[0].startswith("gaussian(")
    bands = ", ".join(f"{call.split('(')[0]} ±{band:.0f} MHz" for call, *_, band in shapes)
    alt = (
        f"Four waveforms, {', '.join(CALLS[:-1])} and {CALLS[-1]}, each {length_ns:g} ns long "
        f"with amplitude {amplitude:g}, above the spectrum of their samples from "
        f"{FLOOR_DB:g} to 0 dB within {SPAN_MHZ:g} MHz of the "
        f"tone. A shaded band marks where {SHARE:.0%} of each spectrum's energy up to half the "
        f"clock lies: {bands}. Outside its band the gaussian's spectrum still peaks at "
        f"{skirt_db['gaussian']:.0f} dB between {0.8 * SPAN_MHZ:g} and {SPAN_MHZ:g} MHz, and the "
        f"cosine's is below {FLOOR_DB:g} dB from {quiet_mhz['cosine']:.0f} MHz on."
    )
    save_themed(
        figure,
        "waveform_spectra",
        title="Four waveforms and the spectra of their samples",
        desc=alt,
        directory=HERE,
    )


def main() -> None:
    waveform_spectra()


if __name__ == "__main__":
    main()
