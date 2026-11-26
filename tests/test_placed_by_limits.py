# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Under ``asap`` and ``alap``, a program placed past a concurrency limit its
control system declares is placed as ``rcp`` (``alap_rcp``) places it, when
that keeps to every limit; a program placed within its limits is untouched,
and one the rcp placement cannot fit either fails as before."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import controller
from cudaq_pulse.controller import ControllerLimitError

_PLACED = {"asap": "rcp", "alap": "alap_rcp"}


@pulse.kernel
def drives_only(qs: list):
    for q in qs:
        drive(get_drive_line(q), gaussian(80, 0.5, 20))


@pulse.kernel
def drives_again(qs: list):
    for q in qs:
        drive(get_drive_line(q), gaussian(80, 0.5, 20))


@pulse.kernel
def drives_then_readouts(qs: list):
    for q in qs:
        drive(get_drive_line(q), gaussian(80, 0.5, 20))
    for q in qs:
        readout(get_readout_line(q), square(400, 0.1))


@pulse.kernel
def loud(qs: list):
    for q in qs:
        drive(get_drive_line(q), square(40, 0.9))


def _system(controller_limits=None, **limits):
    controllers = [
        pulse.Controller(
            drive_lines=range(17), readout_lines=range(17), **(controller_limits or {})
        )
    ]
    control = pulse.ControlSystem(controllers=controllers, **limits)
    return pulse.quantum_system("transmon_surface17", control_system=control)


def _lower(kernel, system, schedule, *args):
    return pulse.compile(kernel, quantum_system=system, schedule=schedule).lower(*args)


def _events(lowered):
    return [(e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule]


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_program_past_a_limit_is_placed_as_its_rcp_schedule(schedule):
    system = _system(max_concurrent_drives=6, max_concurrent_readouts=4)
    qudits = system.qudits[:13]
    lowered = _lower(drives_then_readouts, system, schedule, qudits)
    placed = _lower(drives_then_readouts, system, _PLACED[schedule], qudits)
    assert lowered.options.schedule == _PLACED[schedule]
    assert _events(lowered) == _events(placed)
    assert lowered.mlir() == placed.mlir()
    assert controller.check(lowered) == []


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_program_within_its_limits_keeps_its_schedule(schedule, monkeypatch):
    from cudaq_pulse import compiler

    def entered(*args, **kwargs):
        raise AssertionError("a program within its limits was placed again")

    monkeypatch.setattr(compiler, "_placed_by_limits", entered)
    limited = _system(max_concurrent_drives=13, max_concurrent_readouts=13)
    free = _system()
    qudits = limited.qudits[:13]
    lowered = _lower(drives_then_readouts, limited, schedule, qudits)
    assert lowered.options.schedule == schedule
    unlimited = _lower(drives_then_readouts, free, schedule, free.qudits[:13])
    assert _events(lowered) == _events(unlimited)


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_the_rcp_schedules_place_as_they_did(schedule):
    system = _system(max_concurrent_drives=6)
    lowered = _lower(drives_only, system, schedule, system.qudits[:13])
    assert lowered.options.schedule == schedule


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_program_the_rcp_placement_cannot_fit_fails_naming_the_limit_it_is_over(schedule):
    # Over the drive limit and the amplitude limit: rcp fixes the first only,
    # so the program fails with the error asap or alap gave.
    system = _system({"max_amplitude": 0.5}, max_concurrent_drives=2)
    with pytest.raises(ControllerLimitError) as caught:
        _lower(loud, system, schedule, system.qudits[:3])
    assert any("max_concurrent_drives" in v for v in caught.value.violations)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_without_a_concurrency_limit_another_limit_fails_as_before(schedule):
    system = _system({"max_amplitude": 0.5})
    with pytest.raises(ControllerLimitError) as caught:
        _lower(loud, system, schedule, system.qudits[:3])
    assert not any("max_concurrent" in v for v in caught.value.violations)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_program_with_a_float_parameter_is_placed_as_its_rcp_schedule(schedule):
    @pulse.kernel
    def scaled(qs: list, amplitude: float):
        for q in qs:
            drive(get_drive_line(q), square(80, amplitude))

    system = _system(max_concurrent_drives=2)
    qudits = system.qudits[:5]
    lowered = pulse.compile(scaled, quantum_system=system, schedule=schedule).lower(qudits, 0.2)
    placed = pulse.compile(scaled, quantum_system=system, schedule=_PLACED[schedule]).lower(
        qudits, 0.2
    )
    assert lowered.options.schedule == _PLACED[schedule]
    assert _events(lowered) == _events(placed)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_line_delays_of_another_system_place_the_program_as_a_full_compile_does(schedule):
    from tests._handles import with_line_delays

    def delayed(delays):
        controllers = with_line_delays(
            [pulse.Controller(drive_lines=range(17), readout_lines=range(17))], delays
        )
        control = pulse.ControlSystem(controllers=controllers, max_concurrent_drives=6)
        return pulse.quantum_system("transmon_surface17", control_system=control)

    first = delayed({("drive", 0): 4.0})
    second = delayed({("drive", 1): 8.0})
    # The second compile reuses the first's trace with its line delays.
    a = _lower(drives_only, first, schedule, first.qudits[:13])
    b = _lower(drives_only, second, schedule, second.qudits[:13])
    assert (a.options.schedule, b.options.schedule) == (_PLACED[schedule],) * 2
    # A kernel of its own: compiled in full, not from the first's trace.
    fresh = _lower(drives_again, second, _PLACED[schedule], second.qudits[:13])
    assert _events(b) == _events(fresh)
    assert controller.check(b) == []
