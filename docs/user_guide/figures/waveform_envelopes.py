# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the waveform figure of the operations guide.

Usage: python docs/user_guide/figures/waveform_envelopes.py

Needs matplotlib (the ``[viz]`` extra). Writes ``waveform_envelopes.svg``
and its ``-dark`` twin next to this file: the I and Q of each parametric
waveform at the waveform explorer's defaults, compiled and sampled with
``cudaq_pulse.controller.sample_waveform``. The page shows it where the
explorer does not run; tests/test_waveform_explorer.py checks that its
calls are the explorer's defaults.
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

CALLS = [
    "gaussian(40, 0.3, 10)",
    "drag(40, 0.3, 10, 0.5)",
    "cosine(40, 0.3)",
    "tanh_ramp(40, 0.3, 4)",
    "gaussian_square(40, 0.3, 4, 20)",
    "square(40, 0.3)",
]


def envelope(call: str, qs: pulse.QuantumSystem) -> np.ndarray:
    """The samples ``drive(d, <call>)`` puts on a line, one per clock cycle."""
    namespace: dict = {}
    source = f"def one_pulse(q):\n    drive(get_drive_line(q), {call})\n"
    exec(compile(source, "<figure>", "exec"), namespace)
    program = pulse.compile(pulse.kernel(namespace["one_pulse"]), quantum_system=qs)
    (event,) = program.lower(qs.qudits[0]).export_to_controller()["events"]
    return sample_waveform(event["waveform"], event["duration_cycles"])


def main() -> None:
    qs = pulse.quantum_system("transmon_surface17")
    shapes = [(call, envelope(call, qs)) for call in CALLS]
    (cycles,) = {len(samples) for _, samples in shapes}
    names = [call.split("(")[0] for call in CALLS]
    # The one shape with a Q large enough to label.
    labelled, largest = max(shapes, key=lambda shape: float(np.abs(shape[1].imag).max()))
    assert np.abs(largest.imag).max() > 0

    def figure(p):
        fig, grid = plt.subplots(
            2,
            3,
            figsize=(WIDTH_IN, 3.6),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.42, "wspace": 0.1},
        )
        for ax, (call, samples) in zip(grid.flat, shapes, strict=True):
            t = np.arange(len(samples))
            ax.axhline(0, color=p.rule, linewidth=0.6)
            ax.plot(t, samples.real, "-", color=p.drive, linewidth=1.6)
            ax.plot(t, samples.imag, color=p.q, linewidth=1.4, ls=(0, (2.4, 1.6)))
            ax.set_title(call, fontsize=8, color=p.text, pad=4)
            ax.spines["bottom"].set_color(p.rule)
            if call == labelled:
                peak = int(np.argmax(samples.real))
                ax.annotate(
                    "I (real)",
                    (t[peak], samples.real[peak]),
                    xytext=(0, -7),
                    textcoords="offset points",
                    color=p.drive,
                    fontsize=8.5,
                    ha="center",
                    va="top",
                )
                low = int(np.argmin(samples.imag))
                ax.annotate(
                    "Q (imag)",
                    (t[low], samples.imag[low]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    color=p.q,
                    fontsize=8.5,
                    ha="center",
                    va="bottom",
                )
        grid[1, 2].set_xlabel(
            f"time (clock cycles at {qs.clock_ghz:g} GHz)", loc="right", labelpad=1
        )
        for ax in grid[:, 0]:
            ax.set_ylabel("amplitude")
        return fig

    save_themed(
        figure,
        "waveform_envelopes",
        title="The envelopes of the parametric waveforms",
        desc=(
            f"The I and Q envelopes of {', '.join(names[:-1])} and {names[-1]}, "
            f"each {cycles} clock cycles long."
        ),
        directory=HERE,
    )


if __name__ == "__main__":
    main()
