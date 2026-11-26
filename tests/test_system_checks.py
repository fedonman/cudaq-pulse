# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A quantum system's couplings, their CZ calibrations and residual ZZ entries.

A field left out in Python is Python's own TypeError at the constructor; a
system file or config reports each missing key, at its path and (for a file)
its line, under the file's key names.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import QuantumSystemDefinitionError
from cudaq_pulse.quantum_systems import (
    Coupling,
    FluxCZ,
    PhysicalQudit,
    ResidualZZ,
    SimulationModel,
    TransmonModel,
)
from cudaq_pulse.quantum_systems.base import _COUPLING_OPERATORS, _coupling_operator, _Description


def _qudits(n=3):
    return {
        i: PhysicalQudit(i, 5.0e9 + 1.0e8 * i, 50.0, 40.0, anharmonicity_hz=-2.5e8)
        for i in range(n)
    }


def _system(**fields):
    return pulse.quantum_system(name="checked", qudits=_qudits(), **fields)


def _cz(duration_ns, buffer_ns=None, interaction_frequency_hz=None):
    """The calibrations of a coupling with a CZ."""
    cz = FluxCZ(
        duration_ns=duration_ns,
        buffer_ns=buffer_ns,
        interaction_frequency_hz=interaction_frequency_hz,
    )
    return {pulse.gates.CZ: cz}


def _file_problems(tmp_path, data):
    path = tmp_path / "qpu.json"
    text = json.dumps(data, indent=2)
    path.write_text(text)
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.load(path)
    assert str(raised.value).startswith(f"{len(raised.value.problems)} problem")
    return raised.value.problems, text.splitlines()


def _system_data(**changes):
    data = _system().as_dict()
    data.update(changes)
    return data


def _line_of(lines, needle, after=0):
    return next(i for i, line in enumerate(lines, start=1) if i > after and needle in line)


# ── The coupling operator comes from the model ─────────────────────────────


def test_a_system_names_no_architecture():
    assert not hasattr(pulse.quantum_systems, "Architecture")
    assert "architecture" not in _system_data()


def test_a_system_gives_no_hamiltonian_term_dicts():
    assert not hasattr(_Description, "hamiltonian_terms")
    assert not hasattr(_Description, "dissipator_terms")


def test_the_model_decides_the_coupling_operator():
    assert dict(_COUPLING_OPERATORS) == {TransmonModel: "exchange"}
    assert _coupling_operator(TransmonModel()) == "exchange"
    assert _coupling_operator(TransmonModel(levels={0: 4})) == "exchange"

    class Tuned(TransmonModel):
        __slots__ = ()

    assert _coupling_operator(Tuned()) == "exchange"
    assert _coupling_operator(SimulationModel()) is None
    assert _coupling_operator(None) is None
    assert _system().model == TransmonModel()


# ── Couplings, CZ calibrations and residual ZZ ─────────────────────────────


@pulse.kernel
def _pair_program(q0, q1):
    d0 = get_drive_line(q0)
    d1 = get_drive_line(q1)
    drive(d0, square(8, 0.1))
    sync(d0, d1)
    drive(d1, square(8, 0.1))


def _unmeasured_system():
    return _system(
        couplings=[Coupling((0, 1), None), Coupling((1, 2), 3.0e6), Coupling((2, 0), None)]
    )


def test_a_coupling_not_measured_yet_compiles_schedules_and_exports():
    dev = _unmeasured_system()
    lowered = pulse.compile(_pair_program).lower(*dev.qudits[:2])
    assert [e.start_cycles for e in lowered.schedule] == [0, 8]
    assert lowered.export_to_controller() and lowered.export_to_emulator()
    assert "pulse.coupling" not in lowered.mlir()
    data = dev.as_dict()
    assert [c["strength_hz"] for c in data["couplings"]] == [None, 3.0e6, None]
    assert pulse.QuantumSystem.from_dict(json.loads(json.dumps(data))) == dev


def test_simulate_refuses_a_coupling_not_measured_yet_by_name():
    dev = _unmeasured_system()
    lowered = pulse.compile(_pair_program).lower(*dev.qudits[[2, 0]])
    with pytest.raises(pulse.InvalidValueError) as raised:
        pulse.simulate(lowered, levels=2)
    assert str(raised.value) == (
        "simulating needs the strength of every coupling of the program's qudits; "
        "couplings[2] (qudits 2 and 0) of 'checked' has strength_hz None, not measured yet: "
        "give the strength in Hz, or leave those qudits out of the program"
    )
    everything = pulse.compile(_three_qudits).lower(*dev.qudits)
    with pytest.raises(pulse.InvalidValueError, match=r"couplings\[0\] \(qudits 0 and 1\) and "):
        pulse.simulate(everything, levels=2)
    # A program on measured couplings only simulates.
    pulse.simulate(pulse.compile(_pair_program).lower(*dev.qudits[1:]), levels=2)


@pulse.kernel
def _three_qudits(q0, q1, q2):
    d0 = get_drive_line(q0)
    get_drive_line(q1)
    get_drive_line(q2)
    drive(d0, square(8, 0.1))


def test_residual_zz_is_a_static_zz_term():
    from cudaq_pulse.runtime._attributes import _target_attributes

    system = _system(residual_zz=[ResidualZZ((2, 0), -4.5e4), ResidualZZ((1, 2), 1.0e3)])
    attrs = _target_attributes(system._description, (0, 2), integrator="rk4")
    assert "crosstalk" not in attrs
    assert attrs.endswith(
        "static_terms = [#pulse.static_term<zz, qudits = [2, 0], "
        "rate_hz = -4.50000000000000000e+04>]>"
    )


def test_a_cz_calibration_belongs_to_its_coupling_and_is_read_only():
    calibrations = _cz(98.0, 15.0, 4.4e9)
    coupling = Coupling([3, 1], 5.0e6, calibrations=calibrations)
    assert coupling.qudits == (3, 1)
    assert coupling == Coupling((3, 1), 5.0e6, calibrations=_cz(98.0, 15.0, 4.4e9))
    assert hash(coupling) == hash(Coupling((3, 1), 5.0e6, calibrations=calibrations))
    cz = coupling.get_calibration(pulse.gates.CZ)
    assert cz == FluxCZ(duration_ns=98.0, buffer_ns=15.0, interaction_frequency_hz=4.4e9)
    with pytest.raises(AttributeError):
        cz.duration_ns = 4.5e9  # type: ignore[misc]
    assert coupling.calibrations == {pulse.gates.CZ: cz}
    with pytest.raises(AttributeError, match="cannot assign to field 'strength_hz'"):
        coupling.strength_hz = 1.0  # type: ignore[misc]
    assert not hasattr(cz, "fidelity") and not hasattr(coupling, "gate")
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'duration_ns'"):
        FluxCZ()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'strength_hz'"):
        Coupling((0, 1))  # type: ignore[call-arg]
    assert list(inspect.signature(Coupling).parameters) == ["qudits", "strength_hz", "calibrations"]


def test_a_coupling_has_only_a_couplings_calibrations():
    with pytest.raises(pulse.InvalidValueError, match="pulse.gates.X is a qudit's calibration"):
        Coupling((0, 1), 5.0e6, calibrations={pulse.gates.X: FluxCZ(duration_ns=1.0)})
    with pytest.raises(pulse.InvalidTypeError, match="CZ calibration .* must be a FluxCZ"):
        Coupling((0, 1), 5.0e6, calibrations={pulse.gates.CZ: 98.0})
    with pytest.raises(pulse.InvalidValueError, match="did you mean pulse.gates.CZ"):
        Coupling((0, 1), 5.0e6, calibrations={"czz": FluxCZ(duration_ns=1.0)})
    plain = Coupling((0, 1), 5.0e6)
    with pytest.raises(pulse.NotFoundError, match="qudits 0 and 1 publishes no pulse.gates.CZ"):
        plain.get_calibration(pulse.gates.CZ)
    with pytest.raises(pulse.InvalidValueError, match="calibrated on a qudit"):
        plain.get_calibration(pulse.gates.X)


def test_residual_zz_is_its_qudits_and_rate():
    entry = ResidualZZ([0, 1], -2.0e4)
    assert (entry.qudits, entry.rate_hz) == ((0, 1), -2.0e4)
    assert [f.name for f in dataclasses.fields(ResidualZZ)] == ["qudits", "rate_hz"]
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'rate_hz'"):
        ResidualZZ((0, 1))  # type: ignore[call-arg]
    assert _system(residual_zz=[entry]).residual_zz == (entry,)


def test_a_system_file_holds_couplings_gates_and_residual_zz():
    dev = _system(
        couplings=[
            Coupling((1, 0), 5.7e6, calibrations=_cz(98.0, 15.0, 4.4e9)),
            Coupling((1, 2), None),
        ],
        residual_zz=[ResidualZZ((0, 2), 1.5e4)],
    )
    system = dev.as_dict()
    assert system["couplings"] == [
        {
            "qudits": [1, 0],
            "strength_hz": 5.7e6,
            "calibrations": [
                {
                    "gate": "cz",
                    "duration_ns": 98.0,
                    "buffer_ns": 15.0,
                    "interaction_frequency_hz": 4.4e9,
                }
            ],
        },
        {"qudits": [1, 2], "strength_hz": None},
    ]
    assert system["residual_zz"] == [{"qudits": [0, 2], "rate_hz": 1.5e4}]
    assert "crosstalk" not in system
    back = pulse.QuantumSystem.from_dict(json.loads(json.dumps(dev.as_dict())))
    assert back == dev and back._description == dev._description
    assert back.couplings[0].qudits == (1, 0)


def test_a_key_that_is_no_field_is_an_unknown_key(tmp_path):
    cz = {"gate": "cz", "duration_ns": 98.0, "fidelity": 0.99, "params": {"cz_step": 1.0}}
    data = _system_data(
        couplings=[{"qudits": [0, 1], "strength_hz": 1.0e6, "calibrations": [cz], "gate": cz}],
        crosstalk=[{"qudits": [0, 1], "static_zz_hz": 1.0e4}],
        residual_zz=[{"qudits": [0, 2], "static_zz_hz": 1.0e4}],
    )
    problems, _ = _file_problems(tmp_path, data)
    assert sorted((p.path, p.message) for p in problems) == [
        ("couplings[0].calibrations[0].fidelity", "is not a known key"),
        ("couplings[0].calibrations[0].params", "is not a known key"),
        ("couplings[0].gate", "is not a known key"),
        ("crosstalk", "is not a key of a system file"),
        ("residual_zz[0].rate_hz", "is missing"),
        ("residual_zz[0].static_zz_hz", "is not a known key"),
    ]


def test_the_fingerprint_holds_the_whole_cz_calibration():
    def fingerprint(calibrations):
        return _system(couplings=[Coupling((0, 1), 5e6, calibrations)]).fingerprint

    prints = {
        fingerprint({}),
        fingerprint(_cz(98.0)),
        fingerprint(_cz(99.0)),
        fingerprint(_cz(98.0, buffer_ns=15.0)),
        fingerprint(_cz(98.0, interaction_frequency_hz=4.4e9)),
    }
    assert len(prints) == 5


def test_a_calibration_matches_a_pair_named_in_either_order():
    from cudaq_pulse.quantum_systems import transmon_surface17
    from cudaq_pulse.quantum_systems.calibration import shipped_calibration

    published = transmon_surface17(calibration=None)
    calibration = shipped_calibration("transmon_surface17")
    reversed_ = published.replace(
        couplings=[c.replace(qudits=c.qudits[::-1]) for c in published.couplings]
    )
    forward = calibration._apply(published._description).couplings
    backward = calibration._apply(reversed_._description).couplings
    assert [c.strength_hz for c in backward] == [c.strength_hz for c in forward]
    assert [c.qudits for c in backward] == [c.qudits[::-1] for c in forward]
    assert {c.strength_hz for c in forward}.isdisjoint(c.strength_hz for c in published.couplings)


# ── Value checks ────────────────────────────────────────────────────────────


def _problems(**fields):
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        _system(**fields)
    return [str(p) for p in raised.value.problems]


def test_every_problem_of_every_coupling_is_listed_at_its_path():
    problems = _problems(
        couplings=[
            Coupling((0, 7), 1.0e6),
            Coupling((1, 1), 1.0e6),
            Coupling((0, 1), math.nan),
            Coupling((2, True), math.inf),  # type: ignore[arg-type]
        ]
    )
    assert problems == [
        "couplings[0].qudits[1]: 7 is not a qudit of the quantum system; its qudits are 0..2",
        "couplings[1].qudits: names qudit 1 twice; a coupling is between two distinct qudits",
        "couplings[2].strength_hz: nan is not a finite number of Hz; give the strength in Hz, "
        "or None (null in a file) if it is not measured yet",
        "couplings[3].qudits[1]: must be a qudit index (an int), got bool",
        "couplings[3].strength_hz: inf is not a finite number of Hz; give the strength in Hz, "
        "or None (null in a file) if it is not measured yet",
    ]


def test_zero_negative_and_unmeasured_strengths_are_allowed():
    target = _system(
        couplings=[Coupling((0, 1), 0.0), Coupling((1, 2), -2.5e6), Coupling((0, 2), None)]
    )
    assert [c.strength_hz for c in target.couplings] == [0.0, -2.5e6, None]
    assert _system(couplings=[Coupling((0, 1), 3)]).couplings[0].strength_hz == 3


def test_a_pair_has_one_coupling_in_either_order():
    problems = _problems(
        couplings=[Coupling((0, 1), 1.0e6), Coupling((0, 1), 2.0e6), Coupling((1, 0), 3.0e6)]
    )
    assert problems == [
        "couplings[1].qudits: names the same pair of qudits as couplings[0]; give each pair "
        "one coupling; (0, 1) and (1, 0) are the same pair",
        "couplings[2].qudits: names the same pair of qudits as couplings[0]; give each pair "
        "one coupling; (1, 0) and (0, 1) are the same pair",
    ]
    # A coupling and a residual ZZ entry of one pair are two different terms.
    _system(couplings=[Coupling((0, 1), 1.0e6)], residual_zz=[ResidualZZ((1, 0), 1.0e4)])


def test_a_coupling_is_between_two_qudits_in_this_release():
    problems = _problems(
        couplings=[Coupling((0, 1, 2), 1.0e6), Coupling((0,), 1.0e6), Coupling(0, 1.0e6)]  # type: ignore[arg-type]
    )
    assert problems == [
        "couplings[0].qudits: names 3 qudits; a coupling is between two qudits in this "
        "release; give the indices of its two qudits, as (0, 1)",
        "couplings[1].qudits: names 1 qudit; a coupling is between two qudits in this "
        "release; give the indices of its two qudits, as (0, 1)",
        "couplings[2].qudits: must be a tuple of qudit indices, got int; give the indices of "
        "its two qudits, as (0, 1)",
    ]


def test_a_cz_calibration_is_checked_with_its_coupling():
    problems = _problems(
        couplings=[
            Coupling((0, 1), 1.0e6, calibrations=_cz(-1.0, buffer_ns=math.nan)),
            Coupling((1, 2), 1.0e6, calibrations=_cz(98.0, interaction_frequency_hz="4.4e9")),
        ]
    )
    assert problems == [
        "couplings[0].calibrations[CZ].duration_ns: -1.0 is not a finite time of 0 ns or more; "
        "give it in ns",
        "couplings[0].calibrations[CZ].buffer_ns: nan is not a finite time of 0 ns or more; "
        "give it in ns, or None",
        "couplings[1].calibrations[CZ].interaction_frequency_hz: '4.4e9' is not a finite "
        "frequency above 0 Hz; give it in Hz, or None",
    ]
    assert _problems(couplings=[Coupling((0, 1), 1.0e6, calibrations=_cz(98.0, None, 0.0))]) == [
        "couplings[0].calibrations[CZ].interaction_frequency_hz: 0.0 is not a finite frequency "
        "above 0 Hz; give it in Hz, or None",
    ]
    checked = _system(couplings=[Coupling((0, 1), 1.0e6, calibrations=_cz(0, 0, 4))])
    cz = checked.couplings[0].get_calibration(pulse.gates.CZ)
    assert (cz.duration_ns, cz.interaction_frequency_hz) == (0, 4)


def test_residual_zz_has_the_qudit_checks_of_couplings():
    problems = _problems(
        residual_zz=[
            ResidualZZ((0, 1), 1.0e4),
            ResidualZZ((1, 0), math.inf),
            ResidualZZ((2, 2), 1.0e4),
            Coupling((1, 2), 1.0e6),
        ]
    )
    assert problems == [
        "residual_zz[1].qudits: names the same pair of qudits as residual_zz[0]; give each pair "
        "one residual ZZ entry; (1, 0) and (0, 1) are the same pair",
        "residual_zz[1].rate_hz: inf is not a finite number of Hz; give the ZZ rate in Hz",
        "residual_zz[2].qudits: names qudit 2 twice; a residual ZZ entry is between two "
        "distinct qudits",
        "residual_zz[3]: must be a ResidualZZ, got Coupling",
    ]


def test_coupling_and_residual_zz_problems_come_in_one_error():
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        _system(couplings=[Coupling((0, 9), 1.0e6)], residual_zz=[ResidualZZ((1, 1), 1.0e4)])
    assert str(raised.value).splitlines() == [
        "2 problems in the quantum system 'checked':",
        "  couplings[0].qudits[1]: 9 is not a qudit of the quantum system; its qudits are 0..2",
        "  residual_zz[0].qudits: names qudit 1 twice; a residual ZZ entry is between two "
        "distinct qudits",
    ]


def test_a_calibration_keeps_the_checks():
    from cudaq_pulse.quantum_systems import transmon_surface17

    target = transmon_surface17()
    assert all(c.get_calibration(pulse.gates.CZ) for c in target.couplings)
    with pytest.raises(
        QuantumSystemDefinitionError, match=r"couplings\[1\].qudits: names the same pair"
    ):
        target.replace(couplings=[target.couplings[0]] * 2)


def _qudit_problems(qudits):
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.quantum_system(name="checked", qudits=qudits)
    return [str(p) for p in raised.value.problems]


def test_every_problem_of_every_qudit_is_listed_at_its_path():
    qudits = _qudits()
    qudits[0] = PhysicalQudit(0, -5.0e9, -10.0, -3.0, thermal_population=2.0)
    qudits[2] = PhysicalQudit(2, math.nan, 50.0, math.inf, t2_echo_us=-1.0)
    assert _qudit_problems(qudits) == [
        "qudits[0].frequency_hz: -5000000000.0 is not a positive, finite number of Hz; "
        "give the 0-1 frequency in Hz, more than 0; or, for a system that does not state it, "
        "Offset(hz) from a reference its qudits share, or None (null in a file) for each "
        "qudit's own carrier",
        "qudits[0].t1_us: -10.0 is not a finite number of microseconds, 0 or more; "
        "give T1 in microseconds, 0 or more (0 is no decay)",
        "qudits[0].t2_star_us: -3.0 is not a finite number of microseconds, 0 or more; "
        "give T2* in microseconds, 0 or more (0 is no decay)",
        "qudits[0].thermal_population: 2.0 is not a finite number in [0, 1); "
        "give the excited-state population as a fraction, 0 or more and less than 1",
        "qudits[2].frequency_hz: nan is not a positive, finite number of Hz; "
        "give the 0-1 frequency in Hz, more than 0; or, for a system that does not state it, "
        "Offset(hz) from a reference its qudits share, or None (null in a file) for each "
        "qudit's own carrier",
        "qudits[2].t2_star_us: inf is not a finite number of microseconds, 0 or more; "
        "give T2* in microseconds, 0 or more (0 is no decay)",
        "qudits[2].t2_echo_us: -1.0 is not a finite number of microseconds, 0 or more; "
        "give the Hahn-echo T2 in microseconds, 0 or more, or None (null in a file) if it "
        "is not measured",
    ]


def test_a_problem_names_the_qudit_by_its_index_and_holds_the_value():
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.quantum_system(name="checked", qudits={5: PhysicalQudit(5, 5.0e9, -50.0, 40.0)})
    [problem] = raised.value.problems
    assert (problem.path, problem.value) == ("qudits[5].t1_us", -50.0)


@pytest.mark.parametrize("thermal_population", [1.0, -0.01, math.nan, True])
def test_a_thermal_population_is_a_fraction_below_one(thermal_population):
    qudits = {0: PhysicalQudit(0, 5.0e9, 50.0, 40.0, thermal_population=thermal_population)}
    [problem] = _qudit_problems(qudits)
    assert problem.startswith("qudits[0].thermal_population: ")


def test_zero_decay_unmeasured_echo_and_int_values_are_allowed():
    target = pulse.quantum_system(
        name="checked",
        qudits={
            0: PhysicalQudit(0, 5.0e9, 0.0, 0.0, t2_echo_us=0.0),
            1: PhysicalQudit(1, 5_100_000_000, 50, 40, t2_echo_us=None, thermal_population=0),
            2: PhysicalQudit(2, 5.2e9, 50.0, 40.0, thermal_population=0.99),
        },
    )
    assert target.t1_times == {0: 0.0, 1: 50, 2: 50.0}


def test_qudit_and_coupling_problems_come_in_one_error():
    qudits = _qudits()
    qudits[1] = PhysicalQudit(1, 5.1e9, -50.0, 40.0)
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.quantum_system(name="checked", qudits=qudits, couplings=[Coupling((0, 7), 1.0e6)])
    assert [p.path for p in raised.value.problems] == ["qudits[1].t1_us", "couplings[0].qudits[1]"]


def test_a_system_from_data_reports_a_bad_qudit_value():
    data = _system_data()
    data["qudits"][1]["t2_star_us"] = -40.0
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.from_dict(data)
    assert [(p.path, p.value) for p in raised.value.problems] == [("qudits[1].t2_star_us", -40.0)]


def test_a_qudit_problem_in_a_file_is_at_the_qudit_in_the_list(tmp_path):
    data = _system_data()
    # The file lists the qudits out of index order, so a problem's path is
    # where its qudit is in the list, not the qudit's index.
    data["qudits"].reverse()
    data["qudits"][0]["thermal_population"] = 1.5
    problems, lines = _file_problems(tmp_path, data)
    assert [(p.path, p.value) for p in problems] == [("qudits[0].thermal_population", 1.5)]
    assert problems[0].line == _line_of(lines, '"thermal_population": 1.5')


# ── System files: missing keys and value problems at their lines ───────────


def _coupled_data():
    target = _system(
        couplings=[Coupling((0, 1), 5.0e6, _cz(98.0)), Coupling((1, 2), 4.0e6)],
        residual_zz=[ResidualZZ((0, 2), 1.0e4)],
    )
    return target.as_dict()


def test_a_system_file_names_each_missing_key_by_its_file_name_and_line(tmp_path):
    system = _coupled_data()
    del system["couplings"][0]["strength_hz"]
    del system["couplings"][0]["calibrations"][0]["duration_ns"]
    del system["couplings"][1]["qudits"]
    del system["residual_zz"][0]["rate_hz"]
    problems, lines = _file_problems(tmp_path, system)
    assert [(p.path, p.message, p.fix) for p in problems] == [
        (
            "couplings[0].strength_hz",
            "is missing",
            "give the strength in Hz, or null if it is not measured yet",
        ),
        (
            "couplings[1].qudits",
            "is missing",
            "give the indices of the two qudits, as [0, 1]",
        ),
        ("residual_zz[0].rate_hz", "is missing", "give the ZZ rate in Hz"),
    ]
    first = _line_of(lines, '"couplings"')
    assert problems[0].line == _line_of(lines, "{", after=first)
    assert problems[2].line == _line_of(lines, "{", after=_line_of(lines, '"residual_zz"'))
    assert all(p.file is not None and p.file.endswith("qpu.json") for p in problems)


def test_a_config_mapping_lists_every_missing_key_and_value_problem():
    system = _coupled_data()
    del system["couplings"][0]["calibrations"][0]["gate"]
    system["couplings"][1]["strength_hz"] = math.nan
    system["residual_zz"][0]["qudits"] = [2, 2]
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        pulse.QuantumSystem.from_dict(system)
    assert [(p.path, p.message, p.file) for p in raised.value.problems] == [
        ("couplings[0].calibrations[0].gate", "is missing", None),
        ("couplings[1].strength_hz", "nan is not a finite number of Hz", None),
        ("residual_zz[0].qudits", "names qudit 2 twice", None),
    ]


def test_a_missing_calibration_key_is_named_inside_the_calibration(tmp_path):
    data = _coupled_data()
    del data["couplings"][0]["calibrations"][0]["duration_ns"]
    data["couplings"][0]["calibrations"][0]["interaction_frequency"] = 4.4e9
    problems, lines = _file_problems(tmp_path, data)
    assert [str(p).split(": ", 1)[1] for p in problems] == [
        "couplings[0].calibrations[0].interaction_frequency: is not a known key; "
        "did you mean 'interaction_frequency_hz'?",
        "couplings[0].calibrations[0]: FluxCZ.__init__() missing 1 required "
        "keyword-only argument: 'duration_ns'",
    ]
    assert problems[1].line == _line_of(lines, "{", after=_line_of(lines, '"calibrations"'))


def test_a_coupling_of_unknown_keys_names_each_and_its_missing_qudits(tmp_path):
    data = _coupled_data()
    data["couplings"][1] = {"a": 1, "b": 2, "strength_hz": 4.0e6}
    problems, _ = _file_problems(tmp_path, data)
    assert [(p.path, p.message) for p in problems] == [
        ("couplings[1].a", "is not a known key"),
        ("couplings[1].b", "is not a known key"),
        ("couplings[1].qudits", "is missing"),
    ]


def test_a_value_problem_in_a_file_has_its_file_path_and_line(tmp_path):
    system = _coupled_data()
    del system["couplings"][0]["strength_hz"]
    system["couplings"][1]["qudits"] = [1, 9]
    system["residual_zz"][0]["rate_hz"] = "1e4"
    problems, lines = _file_problems(tmp_path, system)
    assert [(p.path, p.message) for p in problems] == [
        ("couplings[0].strength_hz", "is missing"),
        ("couplings[1].qudits[1]", "9 is not a qudit of the quantum system"),
        ("residual_zz[0].rate_hz", "'1e4' is not a finite number of Hz"),
    ]
    first = _line_of(lines, '"qudits"', after=_line_of(lines, '"couplings"'))
    assert problems[1].line == _line_of(lines, "9", after=_line_of(lines, '"qudits"', after=first))
    assert problems[2].line == _line_of(lines, '"rate_hz"')


def test_a_cz_calibration_is_read_only_and_pickles_by_content():
    import pickle

    cz = FluxCZ(duration_ns=98.0, buffer_ns=15.0, interaction_frequency_hz=4.4e9)
    with pytest.raises(AttributeError):
        cz.buffer_ns = 1.0  # type: ignore[misc]
    assert repr(cz) == (
        "FluxCZ(duration_ns=98.0, buffer_ns=15.0, interaction_frequency_hz=4400000000.0)"
    )
    back = pickle.loads(pickle.dumps(cz))
    assert back == cz and hash(back) == hash(cz)
    assert cz != FluxCZ(duration_ns=98.0, buffer_ns=15.0) and cz != (98.0, 15.0, 4.4e9)
    coupling = Coupling((0, 1), 1.0e6, _cz(98.0))
    back = pickle.loads(pickle.dumps(coupling))
    assert back == coupling and repr(back) == repr(coupling)
    assert back.get_calibration(pulse.gates.CZ) == coupling.get_calibration(pulse.gates.CZ)


def test_a_system_made_from_a_checked_one_is_checked_for_what_changed():
    checked = _system(
        couplings=[Coupling((0, 1), 1.0e6, _cz(98.0)), Coupling((1, 2), 2.0e6)],
        residual_zz=[ResidualZZ((0, 2), 1.0e4)],
    )
    described = checked._description
    assert described.replace(clock_ghz=1.0).couplings is described.couplings
    fewer = {i: q for i, q in described.qudits.items() if i < 2}
    with pytest.raises(QuantumSystemDefinitionError, match=r"couplings\[1\].qudits\[1\]: 2 is not"):
        described.replace(qudits=fewer)
    restrengthened = [c.replace(strength_hz=math.nan) for c in checked.couplings]
    problems = [str(p) for p in _problems_of(checked, restrengthened)]
    assert [p.split(":")[0] for p in problems] == [
        "couplings[0].strength_hz",
        "couplings[1].strength_hz",
    ]
    recalibrated = [Coupling(c.qudits, c.strength_hz, _cz(-1.0)) for c in checked.couplings]
    problems = [str(p) for p in _problems_of(checked, recalibrated)]
    assert [p.split(":")[0] for p in problems] == [
        "couplings[0].calibrations[CZ].duration_ns",
        "couplings[1].calibrations[CZ].duration_ns",
    ]


def _problems_of(target, couplings):
    with pytest.raises(QuantumSystemDefinitionError) as raised:
        target.replace(couplings=couplings)
    return raised.value.problems


def test_values_that_are_not_floats_take_the_complete_checks():
    target = _system(
        couplings=[
            Coupling((0, 1), 5, _cz(98, buffer_ns=0, interaction_frequency_hz=4)),
            Coupling((1, 2), 0.0, _cz(0.0, buffer_ns=0.0)),
        ],
        residual_zz=[ResidualZZ((2, 0), 0)],
    )
    assert [c.strength_hz for c in target.couplings] == [5, 0.0]
    with pytest.raises(QuantumSystemDefinitionError, match="names the same pair of qudits"):
        target.replace(couplings=[*target.couplings, Coupling((2, 1), 7)])
