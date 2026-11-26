# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A qudit's drive and readout: components of their modality, holding its physics."""

from __future__ import annotations

import copy
import math
import pickle
import sys

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.quantum_systems import (
    DispersiveReadout,
    DragPulse,
    Drive,
    MicrowaveDrive,
    PhysicalQudit,
    Readout,
    SimulationModel,
    TransmonModel,
    transmon_surface17,
)
from cudaq_pulse.runtime._call_tree import ReadoutModel
from cudaq_pulse.runtime._calls import IQModel


class _Fluorescence(Readout):
    """A readout no model knows."""

    __slots__ = ()
    _fields = ()

    def __init__(self) -> None:
        object.__setattr__(self, "_values", ())


class _Optical(Drive):
    """A drive no model knows."""

    __slots__ = ()
    _fields = ()

    def __init__(self) -> None:
        object.__setattr__(self, "_values", ())


def _target(**members):
    return pulse.quantum_system(
        name="c", qudits={0: PhysicalQudit(0, 5e9, 10.0, 5.0, anharmonicity_hz=-2e8, **members)}
    )


def test_a_component_is_a_read_only_value():
    a = DispersiveReadout(frequency_hz=7e9, linewidth_hz=1e6)
    b = DispersiveReadout(linewidth_hz=1e6, frequency_hz=7e9)
    assert a == b and hash(a) == hash(b)
    assert a != DispersiveReadout(frequency_hz=7e9) and a != MicrowaveDrive()
    assert repr(a) == "DispersiveReadout(frequency_hz=7000000000.0, linewidth_hz=1000000.0)"
    assert (a.frequency_hz, a.efficiency, a.error_1to0, a.depletion_ns) == (7e9, 0.5, 0.0, None)
    with pytest.raises(AttributeError, match="is read-only"):
        a.linewidth_hz = 2e6  # type: ignore[misc]
    with pytest.raises(AttributeError):
        a.kappa = 2e6  # type: ignore[attr-defined]
    changed = a.replace(linewidth_hz=2e6)
    assert changed.linewidth_hz == 2e6 and changed.frequency_hz == 7e9 and a.linewidth_hz == 1e6
    with pytest.raises(TypeError, match="kapa_hz"):
        a.replace(kapa_hz=2e6)
    with pytest.raises(TypeError):
        DispersiveReadout(7e9)  # type: ignore[call-arg]
    for back in (pickle.loads(pickle.dumps(a)), copy.deepcopy(a)):
        assert back == a and type(back) is DispersiveReadout
    if sys.version_info >= (3, 13):
        assert copy.replace(a, efficiency=0.9).efficiency == 0.9  # type: ignore[attr-defined]
    drive = MicrowaveDrive(amplitude_scale_rad_per_ns=0.2)
    assert pickle.loads(pickle.dumps(drive)) == drive
    assert MicrowaveDrive().amplitude_scale_rad_per_ns is None


def test_the_components_hold_physics_and_not_tuned_pulses():
    assert MicrowaveDrive._fields == ("amplitude_scale_rad_per_ns",)
    assert DispersiveReadout._fields == (
        "frequency_hz",
        "dispersive_shift_hz",
        "linewidth_hz",
        "efficiency",
        "error_1to0",
        "error_0to1",
        "depletion_ns",
    )
    for tuned in ("x_amplitude", "x_duration_ns", "x_sigma_ns"):
        with pytest.raises(TypeError):
            MicrowaveDrive(**{tuned: 1.0})
    for tuned in ("amplitude", "duration_ns", "sigma_ns", "integration_ns", "photon_number"):
        with pytest.raises(TypeError):
            DispersiveReadout(**{tuned: 1.0})


def test_the_anharmonicity_is_a_keyword_a_qudit_may_leave_out():
    with pytest.raises(TypeError):
        PhysicalQudit(0, 5e9, -2e8, 50.0, 30.0)  # type: ignore[misc]
    qudit = PhysicalQudit(0, 5e9, 50.0, 30.0)
    assert (qudit.t1_us, qudit.t2_star_us, qudit.anharmonicity_hz) == (50.0, 30.0, None)
    assert pickle.loads(pickle.dumps(qudit)) == qudit == copy.deepcopy(qudit)
    two_levels = TransmonModel(levels={0: 2})
    assert (
        pulse.quantum_system(name="two", qudits={0: qudit}, model=two_levels).qudit_freq[0] == 5e9
    )
    evenly = pulse.quantum_system(
        name="even", qudits={0: qudit}, model=SimulationModel(levels={0: 3})
    )
    assert evenly.transition_freq[0, 1, 2] == 5e9
    three_levels = pulse.quantum_system(name="three", qudits={0: qudit})
    with pytest.raises(
        pulse.InvalidValueError,
        match=r"qudit 0 gives no anharmonicity_hz, which a transmon of 3 levels needs: .*"
        r"TransmonModel\(levels=\{0: 2\}\)",
    ):
        _ = three_levels.transition_freq


def test_the_depletion_comes_from_the_readout():
    explicit = PhysicalQudit(
        0, 5e9, 1, 1, anharmonicity_hz=-2e8, readout=DispersiveReadout(depletion_ns=250.0)
    )
    from_kappa = PhysicalQudit(
        0, 5e9, 1, 1, anharmonicity_hz=-2e8, readout=DispersiveReadout(linewidth_hz=1e6)
    )
    assert explicit.depletion_ns == 250.0
    assert from_kappa.depletion_ns == pytest.approx(math.log(100) / (2 * math.pi * 1e-3))
    assert PhysicalQudit(0, 5e9, 1, 1, anharmonicity_hz=-2e8).depletion_ns == 0.0
    assert (
        PhysicalQudit(0, 5e9, 1, 1, anharmonicity_hz=-2e8, readout=_Fluorescence()).depletion_ns
        == 0.0
    )


def test_the_shipped_target_describes_its_drive_and_readout():
    qudit = transmon_surface17().physical_qudits[0]
    assert isinstance(qudit.drive, MicrowaveDrive) and isinstance(qudit.readout, DispersiveReadout)
    x = qudit.get_calibration(pulse.gates.X)
    # No amplitude is published: the one read is the pi rotation of the pulse's area.
    assert (x.duration_ns, x.sigma_ns) == (40.0, 10.0) and x.amplitude > 0
    assert qudit.readout.frequency_hz > 6e9 and qudit.readout.linewidth_hz > 0


def test_the_drive_strength_comes_from_the_drive():
    pi = {pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0, amplitude=0.5)}
    area = 10.0 * math.sqrt(2 * math.pi) * math.erf(40.0 / (2 * math.sqrt(2) * 10.0))
    published = _target(drive=MicrowaveDrive(), calibrations=pi)
    assert published.drive_amplitude_scale(0) == pytest.approx(math.pi / (0.5 * area))
    unpublished = {pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0)}
    assert _target(drive=MicrowaveDrive(), calibrations=unpublished).drive_amplitude_scale(0) == 1.0
    scaled = _target(drive=MicrowaveDrive(amplitude_scale_rad_per_ns=2.0))
    assert scaled.drive_amplitude_scale(0) == 2.0
    assert _target().drive_amplitude_scale(0) == 1.0
    with pytest.raises(pulse.InvalidValueError, match="must be positive"):
        _target(drive=MicrowaveDrive(amplitude_scale_rad_per_ns=0.0)).drive_amplitude_scale(0)
    with pytest.raises(pulse.UnsupportedError, match="no drive model for _Optical"):
        _target(drive=_Optical()).drive_amplitude_scale(0)


def test_a_drive_of_no_stated_modality_gives_its_scale():
    drive = Drive(amplitude_scale_rad_per_ns=0.2)
    assert drive.amplitude_scale_rad_per_ns == 0.2 and Drive().amplitude_scale_rad_per_ns is None
    assert Drive._fields == MicrowaveDrive._fields == ("amplitude_scale_rad_per_ns",)
    assert repr(drive) == "Drive(amplitude_scale_rad_per_ns=0.2)" and repr(Drive()) == "Drive()"
    assert repr(MicrowaveDrive(amplitude_scale_rad_per_ns=0.2)) == (
        "MicrowaveDrive(amplitude_scale_rad_per_ns=0.2)"
    )
    # The two are different components, equal only to their own kind.
    assert drive == Drive(amplitude_scale_rad_per_ns=0.2) and hash(drive) == hash(drive.replace())
    microwave = MicrowaveDrive(amplitude_scale_rad_per_ns=0.2)
    assert drive != microwave and microwave != drive and hash(drive) != hash(microwave)
    assert len({drive, microwave, Drive(amplitude_scale_rad_per_ns=0.2)}) == 2
    assert drive.replace(amplitude_scale_rad_per_ns=0.3) == Drive(amplitude_scale_rad_per_ns=0.3)
    for back in (pickle.loads(pickle.dumps(drive)), copy.deepcopy(drive)):
        assert back == drive and type(back) is Drive
    with pytest.raises(TypeError):
        Drive(0.2)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Drive(x_amplitude=0.2)  # type: ignore[call-arg]
    with pytest.raises(AttributeError, match="is read-only"):
        drive.amplitude_scale_rad_per_ns = 0.3  # type: ignore[misc]


def test_the_drive_strength_comes_from_a_drive_of_no_stated_modality():
    assert _target(drive=Drive(amplitude_scale_rad_per_ns=2.0)).drive_amplitude_scale(0) == 2.0
    pi = {pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0, amplitude=0.5)}
    area = 10.0 * math.sqrt(2 * math.pi) * math.erf(40.0 / (2 * math.sqrt(2) * 10.0))
    for drive in (Drive(), MicrowaveDrive()):
        published = _target(drive=drive, calibrations=pi)
        assert published.drive_amplitude_scale(0) == pytest.approx(math.pi / (0.5 * area))
    assert _target(drive=Drive()).drive_amplitude_scale(0) == 1.0
    with pytest.raises(pulse.InvalidValueError, match="must be positive"):
        _target(drive=Drive(amplitude_scale_rad_per_ns=-1.0)).drive_amplitude_scale(0)
    # A pi pulse no amplitude is published for is derived from either's scale.
    unpublished = {pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0)}
    amplitudes = [
        PhysicalQudit(0, 5e9, 10.0, 5.0, drive=drive, calibrations=unpublished)
        .get_calibration(pulse.gates.X)
        .amplitude
        for drive in (
            Drive(amplitude_scale_rad_per_ns=2.0),
            MicrowaveDrive(amplitude_scale_rad_per_ns=2.0),
            Drive(amplitude_scale_rad_per_ns=1.0),
        )
    ]
    assert amplitudes[0] == amplitudes[1] == pytest.approx(amplitudes[2] / 2)


@pytest.mark.parametrize("made", [Drive, MicrowaveDrive])
def test_a_square_pulse_turns_the_qudit_by_its_scale(made):
    qudit = PhysicalQudit(0, 5e9, 0.0, 0.0, drive=made(amplitude_scale_rad_per_ns=0.5))
    qs = pulse.quantum_system(name="turn", qudits={0: qudit}, model=TransmonModel(levels={0: 2}))

    @pulse.kernel
    def turn(q):
        drive(get_drive_line(q), square(20 * pulse.ns, math.pi / 20))  # noqa: F821

    # 20 ns at pi / 20 with a scale of 0.5 is a quarter turn.
    result = pulse.simulate(turn, qs.qudits[0], quantum_system=qs)
    assert result.level_population(1) == pytest.approx(0.5, abs=1e-3)


def test_a_readout_no_model_knows_is_refused():
    target = _target(readout=_Fluorescence())
    with pytest.raises(pulse.UnsupportedError, match="no readout model for _Fluorescence"):
        IQModel(target._description, (0,), decay=False)
    with pytest.raises(pulse.UnsupportedError, match="no readout model for _Fluorescence"):
        ReadoutModel(target._description, (0,), (2,), decay=False)
    # Without a readout, ideal reports, and no IQ points.
    assert ReadoutModel(_target()._description, (0,), (2,), decay=True).reports(0, 1) == [(1, 1.0)]
    with pytest.raises(
        pulse.UnsupportedError, match="qudit 0 has no readout: IQ points are synthesized"
    ):
        IQModel(_target()._description, (0,), decay=False)


def test_the_iq_model_needs_the_dispersive_shift_and_the_linewidth():
    given = DispersiveReadout(frequency_hz=7e9, dispersive_shift_hz=1e6, linewidth_hz=2e6)
    IQModel(_target(readout=given)._description, (0,), decay=False)
    message = r"qudit 0 need its DispersiveReadout's linewidth_hz, which it does not give"
    with pytest.raises(pulse.UnsupportedError, match=message):
        IQModel(_target(readout=given.replace(linewidth_hz=None))._description, (0,), decay=False)
    both = r"dispersive_shift_hz and linewidth_hz"
    with pytest.raises(pulse.UnsupportedError, match=both):
        IQModel(
            _target(readout=DispersiveReadout(frequency_hz=7e9))._description, (0,), decay=False
        )
