# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the pipeline figure of the architecture pages.

Usage: python docs/architecture/figures/pipeline.py

Needs matplotlib (the ``[viz]`` extra). Writes ``pipeline.svg`` and its
``-dark`` twin next to this file: the stages of the compilation pipeline
with one kernel followed through them. The kernel is the Hello World of
``docs/getting_started.rst``, run as that page has it; the operations
counted at each level are those of ``scheduled.mlir(level)``, and the
length is the scheduled program's.
"""

import contextlib
import io
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tutorials" / "figures"))
from _style import WIDTH_IN, _page_blocks, save_themed  # noqa: E402

PAGE = HERE.parents[1] / "getting_started.rst"
# Each IR level of ``mlir()``, the dialect its operations are in, and its name.
LEVELS = (
    ("pulse", "pulse", "scheduled Pulse dialect"),
    ("qop", "qop", "QOp dialect"),
    ("cudm", "cudm", "CuDensityMat dialect"),
    ("llvm", "llvm", "LLVM dialect"),
)


def page_namespace() -> dict:
    """The names of the page's Hello World block, run as the page has it."""
    (code,) = (code for _, code, _ in _page_blocks()(PAGE) if "def rabi_oscillation(" in code)
    namespace: dict = {"__name__": "guide"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)
    return namespace


def operations(text: str, dialect: str) -> int:
    """How many operations of *dialect* the MLIR *text* holds: the lines that an
    operation of the dialect starts, after its results if it has any."""
    return len(re.findall(rf'^\s*(?:[%\w:#, ]+ = )?"?{dialect}\.[a-z_][\w.]*', text, re.M))


def pipeline(page) -> None:
    scheduled = page["scheduled"]
    name = "rabi_oscillation"
    assert f"module @{name} " in scheduled.mlir()
    counts = {level: operations(scheduled.mlir(level), dialect) for level, dialect, _ in LEVELS}
    assert all(counts.values()), counts
    length = scheduled.duration
    # Each row: the step that leads to it, what exists after it, and the page's kernel there.
    rows = [
        (None, "@pulse.kernel function", f"{name}()"),
        ("1. bytecode tracing", "packed int64 buffer", "a numpy array"),
        ("2. one zero-copy FFI call", "Pulse dialect, in memory", ""),
        ("3. the passes of passes= (DEFAULT_PASSES by default)", "optimized Pulse dialect", ""),
        (
            "4. the schedule of the policy (asap, alap, rcp, alap_rcp)",
            LEVELS[0][2],
            f"{counts['pulse']} pulse operations, {length}",
        ),
        ("5. Pulse to QOp", LEVELS[1][2], f"{counts['qop']} qop operations"),
        ("5. QOp to CuDensityMat", LEVELS[2][2], f"{counts['cudm']} cudm operations"),
        ("5. CuDensityMat to LLVM", LEVELS[3][2], f"{counts['llvm']} llvm operations"),
        ("JIT compile and run", "simulation on the GPU", ""),
    ]
    got = {
        4: "kernel.lower(...)",
        5: 'scheduled.mlir("qop")',
        6: 'scheduled.mlir("cudm")',
        7: 'scheduled.mlir("llvm")',
        8: 'pulse.simulate(backend="gpu")',
    }
    optional_from = 5
    assert len(rows) == max(got) + 1

    def figure(p):
        step, box_h, box_w, x = 0.62, 0.36, 2.0, 3.35
        height = step * len(rows) + 0.1
        fig, ax = plt.subplots(figsize=(WIDTH_IN, height))
        ax.axis("off")
        ax.set_xlim(0, WIDTH_IN)
        ax.set_ylim(0, height)
        for k, (via, what, here) in enumerate(rows):
            y = height - 0.05 - step * k - box_h
            lowering = k >= optional_from
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    box_w,
                    box_h,
                    boxstyle="round,pad=0.0,rounding_size=0.06",
                    fc="none",
                    ec=p.q if lowering else p.drive,
                    lw=1.2,
                    ls=(0, (4, 2)) if lowering else "solid",
                )
            )
            ax.text(x + box_w / 2, y + box_h / 2, what, ha="center", va="center", color=p.text)
            both = bool(here) and k in got
            if here:
                ax.text(
                    x + box_w + 0.14,
                    y + box_h / 2 + (0.08 if both else 0),
                    here,
                    va="center",
                    fontsize=8,
                    color=p.muted,
                )
            if k in got:
                ax.text(
                    x + box_w + 0.14,
                    y + box_h / 2 - (0.09 if both else 0),
                    got[k],
                    va="center",
                    fontsize=8,
                    color=p.text,
                )
            if via:
                ax.annotate(
                    "",
                    (x + box_w / 2, y + box_h + 0.01),
                    (x + box_w / 2, y + step - 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=p.muted, lw=1.0),
                )
                ax.text(
                    x - 0.12,
                    y + (box_h + step) / 2,
                    via,
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=p.muted,
                )
        return fig

    save_themed(
        figure,
        "pipeline",
        title="The compilation pipeline, with one kernel followed through it",
        desc=(
            f"The pipeline as a column of stages, with the kernel {name} of Getting Started "
            "followed through it, the steps numbered as the page's stages. A @pulse.kernel "
            "function is traced from its bytecode (1) into a packed int64 buffer, a numpy "
            "array, which one zero-copy FFI call (2) builds into the Pulse dialect in memory. "
            "The passes of passes=, DEFAULT_PASSES by default (3), give the optimized Pulse "
            "dialect, and the schedule of the policy, asap, alap, rcp or alap_rcp (4), the "
            f"scheduled Pulse dialect: {counts['pulse']} pulse operations, {length}, what "
            "kernel.lower(...) returns. Four optional steps, outlined dashed, follow: the "
            f"lowerings (5) to the QOp dialect ({counts['qop']} qop operations), to the "
            f"CuDensityMat dialect ({counts['cudm']} cudm operations) and to the LLVM dialect "
            f"({counts['llvm']} llvm operations), each read with scheduled.mlir of its level, "
            'and the simulation on the GPU, which pulse.simulate with backend="gpu" compiles '
            "and runs."
        ),
        directory=HERE,
    )


def main() -> None:
    pipeline(page_namespace())


if __name__ == "__main__":
    main()
