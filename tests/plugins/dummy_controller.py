# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dummy controller backend, installed by tests/test_backends.py as the
entry point ``dummy = dummy_controller:DummyBackend``.

Its controller runs at 1 GHz, issues one instruction every 2 cycles and has
one detector per pair of qudits. Its program is one tuple per drive and
readout, ``(start, line, kind, shape)``; running it returns them in time
order with an outcome of 0 for each readout of each shot.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.controller import FORMAT


class DummyBackend:
    name = "dummy"
    formats = (FORMAT,)

    def control_system(self) -> ControlSystem:
        limits = Controller(
            clock_ghz=1.0,
            instruction_cycles=2,
            min_idle_cycles=2,
            detectors={q: q // 2 for q in range(8)},
            detector_recovery_cycles=10,
        )
        return ControlSystem(controllers=[limits], max_concurrent_readouts=1)

    def emit(self, program: Mapping[str, Any]) -> list[tuple[int, int, str, str]]:
        if program.get("format") != FORMAT:
            raise ValueError(f"expected a {FORMAT} program, got {program.get('format')!r}")
        return sorted(
            (e["start_cycles"], e["line"], e["kind"], e["waveform"]["shape"])
            for e in program["events"]
            if e["kind"] in ("drive", "readout")
        )

    def run(self, artifact: Any, *, shots_count: int = 1) -> dict[str, Any]:
        readouts = sum(1 for _, _, kind, _ in artifact if kind == "readout")
        return {"played": list(artifact), "outcomes": [[0] * readouts for _ in range(shots_count)]}


class OldBackend(DummyBackend):
    """A backend written for an export format this cudaq-pulse does not write."""

    formats = ("cudaq-pulse.controller/0",)
