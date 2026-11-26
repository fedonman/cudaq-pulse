# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dressed frames: in a system with couplings, each qudit's drive tone plays
at its dressed 0-1 frequency, the bare frequency_hz moved by its couplings,
which the Hamiltonian keeps bare."""

from __future__ import annotations

import math
import warnings

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems.base import (
    _exchange_pairs,
    _single_excitation,
    dressed_frequencies,
    layout_frequencies,
)


@pytest.fixture(scope="module")
def surface17():
    return pulse.quantum_system("transmon_surface17")


def _shift_khz(qs, label):
    q = qs.qudit(label)
    return (q.dressed_frequency_hz - qs._description.qudits[q.index].frequency_hz) / 1e3


@pytest.mark.parametrize(
    ("label", "khz"),
    [("D1", -31.1), ("D5", -63.0), ("D9", -38.5), ("X3", +66.5), ("Z2", +64.0), ("Z4", +33.5)],
)
def test_a_coupled_qudit_is_dressed_by_all_its_couplings(surface17, label, khz):
    assert _shift_khz(surface17, label) == pytest.approx(khz, abs=0.05)


def test_the_dressed_frequency_is_the_exact_single_excitation_eigenvalue():
    # Two transmons and their exchange, diagonalised in full at 6 levels each:
    # D5's dressed 0-1 frequency is E(1, 0) - E(0, 0), whatever the levels.
    full = pulse.quantum_system("transmon_surface17")._description
    d, z = full.qudits[4], full.qudits[14]
    (pair,) = [c for c in full.couplings if set(c.qudits) == {4, 14}]
    g, levels = pair.strength_hz, 6
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    eye = np.eye(levels)

    def duffing(p):
        n = np.arange(levels)
        return np.diag(n * p.frequency_hz + n * (n - 1) / 2 * p.anharmonicity_hz)

    h = np.kron(duffing(d), eye) + np.kron(eye, duffing(z))
    h += g * (np.kron(a.T, a) + np.kron(a, a.T))
    values, vectors = np.linalg.eigh(h)

    def energy(i, j):
        bare = np.zeros(levels * levels)
        bare[i * levels + j] = 1.0
        return values[np.argmax((vectors.T @ bare) ** 2)]

    reference = energy(1, 0) - energy(0, 0)
    alone = pulse.quantum_system(name="pair", qudits={4: d, 14: z}, couplings=[pair])
    assert alone.qudits[0].dressed_frequency_hz == pytest.approx(reference, abs=0.01)
    assert reference - d.frequency_hz == pytest.approx(-15_151.3, abs=0.1)
    # Second order: J^2 / Delta.
    assert reference - d.frequency_hz == pytest.approx(
        g**2 / (d.frequency_hz - z.frequency_hz), rel=1e-3
    )


def test_a_system_without_couplings_is_left_as_it_is():
    qs = pulse.quantum_system("transmon_lattice:1")
    description = qs._description
    assert not description.couplings
    assert qs._dressed is None and "_dressed" not in vars(qs)
    # Its line tones are the class's own method, not a dressed one.
    assert "_line_tones" not in vars(qs)
    assert qs.frequencies == description.frequencies
    assert qs.qudits[0].dressed_frequency_hz == description.qudits[0].frequency_hz


def test_a_drive_tone_plays_at_the_dressed_frequency(surface17):
    d5 = surface17.qudit("D5")
    default, tones = surface17._line_tones(d5.index, False)
    assert default == (0, 1)
    assert tones[0, 1] == d5.dressed_frequency_hz
    assert surface17.frequencies[d5.index] == d5.dressed_frequency_hz
    # Its readout tone is the resonator's, not moved.
    _, readout = surface17._line_tones(d5.index, True)
    assert readout[0, 1] == surface17._description.qudits[d5.index].readout.frequency_hz


def test_the_dressed_table_is_computed_once_per_system(surface17):
    description = surface17._description
    assert dressed_frequencies(description) is dressed_frequencies(description)


def test_a_simulated_part_of_the_system_is_dressed_as_on_the_whole(surface17):
    # D5 and Z2 alone keep their one coupling: given these frequencies, that
    # coupling dresses them to their frequencies on the whole system.
    description = surface17._description
    given = layout_frequencies(description, (4, 14))
    within = {**{q: description.qudits[q].frequency_hz for q in (4, 14)}, **given}
    pairs = [(q, s) for q, s in _exchange_pairs(description) if set(q) <= {4, 14}]
    dressed = dressed_frequencies(description)
    for q, hz in _single_excitation(within, pairs).items():
        assert hz == pytest.approx(dressed[q], abs=1e-3)
    # The whole system, simulated, needs no moving.
    assert layout_frequencies(description, tuple(description.qudits)) == {}


@pulse.kernel
def _ramsey(d, z, t: int, quarter: int):
    dl = get_drive_line(d)
    zl = get_drive_line(z)
    # Z2 is in the program, so the simulation keeps it and its coupling.
    sync(dl, zl)
    drive(dl, dl.get_calibration(pulse.gates.X90))
    wait(dl, t)
    for _ in range(quarter):
        shift_phase(dl.tone, math.pi / 2)
    drive(dl, dl.get_calibration(pulse.gates.X90))


def test_a_ramsey_on_a_coupled_qudit_shows_no_offset(surface17):
    # D5 beside Z2 (in its ground level), the rest of surface17 left out, no
    # decoherence: at its default frame D5 no longer precesses at its dressed
    # shift (-63 kHz from all its neighbours; with bare frames, as before, a
    # simulation of D5 and Z2 alone showed Z2's -15 kHz).
    d5, z2 = surface17.qudit("D5").index, surface17.qudit("Z2").index
    base = pulse.quantum_system("transmon_surface17", layout=[d5, z2])
    quiet = {
        i: q.replace(t1_us=0, t2_star_us=0, t2_echo_us=0) for i, q in base.physical_qudits.items()
    }
    qs = base.replace(qudits=quiet)
    d, z = qs.qudit("D5"), qs.qudit("Z2")
    cycle_ns = 1 / qs.clock_ghz

    def phase(t_ns):
        population = []
        for quarter in (0, 1):
            program = pulse.compile(_ramsey, quantum_system=qs).lower(
                d, z, round(t_ns / cycle_ns), quarter
            )
            result = pulse.simulate(program, quantum_system=qs, dt_ns=1.0, levels=2)
            population.append(result.level_population(1, qudit=d.index))
        return math.atan2(2 * population[1] - 1, 2 * population[0] - 1)

    turned = math.remainder(phase(12_000) - phase(2_000), 2 * math.pi)
    assert abs(turned / (2 * math.pi * 10e-6)) < 100


def test_bare_frames_keep_the_drive_tones_at_frequency_hz():
    bare = pulse.quantum_system("transmon_surface17", frames="bare")
    dressed = pulse.quantum_system("transmon_surface17")
    d5 = bare.qudit("D5")
    assert bare._dressed is None and "_line_tones" not in vars(bare)
    assert d5.dressed_frequency_hz == bare.physical_qudits[d5.index].frequency_hz
    assert bare._line_tones(d5.index, False)[1][0, 1] == d5.dressed_frequency_hz
    # A different system: another fingerprint, as its programs differ.
    assert bare.fingerprint != dressed.fingerprint and bare != dressed
    # The registered name stays, and a system file says which frames.
    data = bare.as_dict()
    assert data["name"] == "transmon_surface17" and data["frames"] == "bare"
    assert "frames" not in dressed.as_dict()
    assert pulse.QuantumSystem.from_dict(data) == bare


def test_frames_is_a_keyword_of_a_built_system_and_of_replace():
    surface = pulse.quantum_system("transmon_surface17")
    built = pulse.quantum_system(
        name="pair",
        qudits={4: surface.physical_qudits[4], 14: surface.physical_qudits[14]},
        couplings=[c for c in surface.couplings if set(c.qudits) == {4, 14}],
        frames="bare",
    )
    assert built._dressed is None
    dressed = built.replace(frames="dressed")
    assert dressed.qudits[0].dressed_frequency_hz != dressed.physical_qudits[4].frequency_hz
    for wrong in ("lab", None):
        with pytest.raises(pulse.QuantumSystemDefinitionError, match="frames"):
            pulse.quantum_system(name="x", qudits=dict(built.physical_qudits), frames=wrong)


def test_dressed_frames_keep_the_fingerprints_systems_had():
    # The default adds nothing to what a fingerprint hashes.
    plain = pulse.quantum_system("transmon_lattice:1")
    assert (
        plain.fingerprint == pulse.quantum_system("transmon_lattice:1", frames="bare").fingerprint
    )


def test_a_variant_plays_dressed_tones_whatever_it_is_first_asked():
    # A variant makes its tone table when first read; its line tones, read
    # first, still make it and are dressed.
    variant = pulse.quantum_system("transmon_surface17").replace(layout=[4, 14])
    assert "fingerprint" not in vars(variant)
    index = variant.layout[0]
    _, tones = variant._line_tones(index, False)
    assert tones[0, 1] == dressed_frequencies(variant._description)[index]
    assert tones[0, 1] == variant.qudits[0].dressed_frequency_hz


def test_a_dressed_system_keeps_the_fingerprint_it_had():
    # Main's fingerprints before dressed frames: the default adds nothing.
    assert pulse.quantum_system("transmon_surface17").fingerprint == (
        "c206f28b5e1c7f2351cc47c17f36449b"
    )
    assert pulse.quantum_system("transmon_lattice:4").fingerprint == (
        "c3f8edd5695e3391f1501b05425e3313"
    )


def test_the_dressed_table_is_the_eigenvalues_of_the_whole_system(surface17):
    description = surface17._description
    qudits = sorted(description.qudits)
    matrix = np.diag([description.qudits[q].frequency_hz for q in qudits])
    row = {q: i for i, q in enumerate(qudits)}
    for (a, b), g in _exchange_pairs(description):
        matrix[row[a], row[b]] = matrix[row[b], row[a]] = g
    assert sorted(dressed_frequencies(description).values()) == pytest.approx(
        sorted(np.linalg.eigvalsh(matrix)), abs=1e-3
    )


@pytest.mark.parametrize("detuning_hz", [100e6, 20e6, 5e6, 1e6])
def test_a_simulated_pair_near_resonance_is_dressed_exactly(detuning_hz):
    # Two kept qudits within a few couplings of each other, each with a
    # dropped neighbour: their given frequencies dress, with the kept
    # coupling alone, to their frequencies on the whole system.
    base = pulse.quantum_system("transmon_surface17")._description
    q = {i: base.qudits[k].replace(index=i) for i, k in enumerate((4, 14, 13, 3))}
    q[0] = q[0].replace(frequency_hz=5.0e9)
    q[1] = q[1].replace(frequency_hz=5.0e9 + detuning_hz)
    q[2] = q[2].replace(frequency_hz=5.015e9)
    q[3] = q[3].replace(frequency_hz=5.0e9 + detuning_hz + 15e6)
    coupling = pulse.quantum_systems.Coupling
    system = pulse.quantum_system(
        name="near",
        qudits=q,
        couplings=[coupling((0, 1), 5e6), coupling((0, 2), 5e6), coupling((1, 3), 5e6)],
    )
    description = system._description
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        given = layout_frequencies(description, (0, 1))
    within = {k: given.get(k, description.qudits[k].frequency_hz) for k in (0, 1)}
    dressed = dressed_frequencies(description)
    # The pair has the whole system's spectrum of them (near resonance which
    # qudit a mixed state counts as may differ between the two)...
    got = sorted(_single_excitation(within, [((0, 1), 5e6)]).values())
    want = sorted(dressed[k] for k in (0, 1))
    if want[1] - want[0] >= 2 * 5e6:
        assert got == pytest.approx(want, abs=0.01)
        assert caught == []
    else:
        # ...unless the dropped neighbours narrow the pair's splitting below
        # twice its coupling, which no frequencies of the pair alone give:
        # then the closest, the splitting at its least.
        least = (2 * 5e6 - (want[1] - want[0])) / 2
        (warned,) = caught
        assert issubclass(warned.category, pulse.PulseWarning)
        assert str(warned.message).startswith(
            "simulate: qudits 0, 1 are so near resonance that, simulated without qudits 2, 3,"
        )
        assert max(abs(g - w) for g, w in zip(got, want, strict=True)) == pytest.approx(
            least, rel=0.02
        )


@pytest.mark.parametrize("eps_hz", [0.0, 1.0, 1e3])
def test_a_simulated_pair_at_resonance_still_gets_the_closest(eps_hz):
    # Two qubits at resonance, coupled at 5 MHz, each pulled 4.5 MHz by a
    # coupler left out: the step is nearly singular, and they still move.
    base = pulse.quantum_system("transmon_surface17")._description
    q = {i: base.qudits[k].replace(index=i) for i, k in enumerate((4, 14, 13, 3))}
    for i, hz in enumerate((5.0e9, 5.0e9 + eps_hz, 5.5e9, 5.6e9)):
        q[i] = q[i].replace(frequency_hz=hz)
    coupling = pulse.quantum_systems.Coupling
    system = pulse.quantum_system(
        name="resonant",
        qudits=q,
        couplings=[coupling((0, 1), 5e6), coupling((0, 2), 50e6), coupling((1, 3), 50e6)],
    )
    description = system._description
    given = layout_frequencies(description, (0, 1))
    assert set(given) == {0, 1}
    within = {k: given[k] for k in (0, 1)}
    got = sorted(_single_excitation(within, [((0, 1), 5e6)]).values())
    want = sorted(dressed_frequencies(description)[k] for k in (0, 1))
    assert max(abs(g - w) for g, w in zip(got, want, strict=True)) < 50e3


def _pair_alone(name="pair"):
    full = pulse.quantum_system("transmon_surface17")._description
    d, z = full.qudits[4], full.qudits[14]
    (pair,) = [c for c in full.couplings if set(c.qudits) == {4, 14}]
    return pulse.quantum_system(name=name, qudits={4: d, 14: z}, couplings=[pair]), d, z, pair


def test_the_dressed_1_2_tone_is_the_exact_two_excitation_energy():
    # D5 beside Z2 alone, diagonalised in full at 6 levels each: its dressed
    # 1-2 transition is E(2, 0) - E(1, 0).
    alone, d, z, pair = _pair_alone()
    levels = 6
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)
    eye = np.eye(levels)

    def duffing(p):
        n = np.arange(levels)
        return np.diag(n * p.frequency_hz + n * (n - 1) / 2 * p.anharmonicity_hz)

    h = np.kron(duffing(d), eye) + np.kron(eye, duffing(z))
    h += pair.strength_hz * (np.kron(a.T, a) + np.kron(a, a.T))
    values, vectors = np.linalg.eigh(h)

    def energy(i, j):
        bare = np.zeros(levels * levels)
        bare[i * levels + j] = 1.0
        return values[np.argmax((vectors.T @ bare) ** 2)]

    _, tones = alone._line_tones(alone.qudits[0].index, False)
    assert tones[1, 2] == pytest.approx(energy(2, 0) - energy(1, 0), abs=0.01)
    assert tones[0, 1] == pytest.approx(energy(1, 0) - energy(0, 0), abs=0.01)
    assert tones[0, 2] == pytest.approx(energy(2, 0) - energy(0, 0), abs=0.01)


def test_a_simulated_part_keeps_its_dropped_neighbours_share_of_the_1_2_tone():
    from cudaq_pulse.quantum_systems.base import _level, layout_anharmonicities

    qs = pulse.quantum_system("transmon_surface17")
    description = qs._description
    kept = (4, 14)
    # Every kept qudit whose group reaches a dropped one, whatever a program
    # plays: D5 and Z2 both have neighbours left out.
    alphas = layout_anharmonicities(description, kept)
    assert set(alphas) == {4, 14}
    assert layout_anharmonicities(description, kept) is alphas
    moved = layout_frequencies(description, kept)
    f = {q: moved.get(q, description.qudits[q].frequency_hz) for q in kept}
    alpha = {q: alphas.get(q, description.qudits[q].anharmonicity_hz) for q in kept}
    energies = {q: [0.0, f[q], 2 * f[q] + alpha[q]] for q in kept}
    pairs = [(q, s) for q, s in _exchange_pairs(description) if set(q) <= set(kept)]
    # Each kept qudit's simulated 1-2 transition is the tone a program plays.
    for q in kept:
        _, tones = qs._line_tones(q, False)
        simulated = _level(energies, pairs, q, 2)[0] - _level(energies, pairs, q, 1)[0]
        assert simulated == pytest.approx(tones[1, 2], abs=0.01)


def test_a_0_1_program_on_a_coupled_system_loads_no_numpy(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "probe.py"
    script.write_text(
        "import sys, cudaq_pulse as p\n"
        "qs = p.quantum_system('transmon_surface17')\n"
        "_, tones = qs._line_tones(4, False)\n"
        "tones[0, 1]\n"
        "print('numpy' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_asking_whether_a_tone_exists_works_nothing_out():
    from cudaq_pulse.quantum_systems.base import _LEVELS

    # A system of its own (descriptions are equal by value, so a name no
    # other test uses), so that no earlier test has worked out its levels.
    alone, _, _, _ = _pair_alone("asked")
    qudit = alone.qudits[0].index
    _, tones = alone._line_tones(qudit, False)
    assert (1, 2) in tones and (5, 6) not in tones
    assert not any(key[0] == qudit for key in _LEVELS.get(alone._description, {}))


def test_a_kept_qudit_two_couplings_from_a_dropped_one_is_corrected():
    from cudaq_pulse.quantum_systems.base import (
        _level,
        dressed_level_energy,
        layout_anharmonicities,
    )

    base = pulse.quantum_system("transmon_surface17")._description
    q = {i: base.qudits[k].replace(index=i) for i, k in enumerate((4, 14, 13))}
    for i, hz in enumerate((5.0e9, 5.3e9, 5.25e9)):
        q[i] = q[i].replace(frequency_hz=hz, anharmonicity_hz=-200e6)
    coupling = pulse.quantum_systems.Coupling
    chain = pulse.quantum_system(
        name="chain", qudits=q, couplings=[coupling((0, 1), 30e6), coupling((1, 2), 30e6)]
    )
    description = chain._description
    kept = (0, 1)
    alphas = layout_anharmonicities(description, kept)
    assert 0 in alphas
    moved = layout_frequencies(description, kept)
    f = {k: moved.get(k, description.qudits[k].frequency_hz) for k in kept}
    alpha = {k: alphas.get(k, description.qudits[k].anharmonicity_hz) for k in kept}
    energies = {k: [0.0, f[k], 2 * f[k] + alpha[k]] for k in kept}
    pairs = [((0, 1), 30e6)]
    assert _level(energies, pairs, 0, 2)[0] == pytest.approx(
        dressed_level_energy(description, 0, 2), abs=0.01
    )


def _crowded_chain():
    # Five qudits near resonance, one coupling apart: qudit 0's level 1
    # among the qudits within three couplings is MHz from the whole chain's.
    base = pulse.quantum_system("transmon_surface17")._description
    q = {
        i: base.qudits[4].replace(index=i, frequency_hz=hz, anharmonicity_hz=-200e6)
        for i, hz in enumerate((5.0e9, 5.08e9, 5.0e9, 5.08e9, 5.0e9))
    }
    coupling = pulse.quantum_systems.Coupling
    return pulse.quantum_system(
        name="crowded", qudits=q, couplings=[coupling((i, i + 1), 30e6) for i in range(4)]
    )


def test_the_0_2_tone_near_resonance_is_the_whole_chains_level_2():
    from cudaq_pulse.quantum_systems.base import _level

    chain = _crowded_chain()
    description = chain._description
    f = {q: description.qudits[q].frequency_hz for q in range(5)}
    energies = {q: [0.0, f[q], 2 * f[q] - 200e6] for q in range(5)}
    pairs = _exchange_pairs(description)
    for q in range(5):
        _, tones = chain._line_tones(q, False)
        # Within the share of the qudits beyond three couplings.
        assert tones[0, 2] == pytest.approx(_level(energies, pairs, q, 2)[0], abs=50e3)


def test_the_1_2_tone_near_resonance_drives_a_transition_out_of_the_dressed_state():
    # The whole chain at three levels each: the 1-2 tone is a transition of
    # weight from the state the 0-1 tone reaches.
    chain = _crowded_chain()
    description = chain._description
    n, levels = 5, 3
    a = np.diag(np.sqrt(np.arange(1, levels)), 1)

    def at(op, q):
        out = np.eye(1)
        for r in range(n):
            out = np.kron(out, op if r == q else np.eye(levels))
        return out

    h = sum(
        at(np.diag([0.0, f, 2 * f - 200e6]), q)
        for q, f in ((q, description.qudits[q].frequency_hz) for q in range(n))
    )
    for (p, q), g in _exchange_pairs(description):
        h = h + g * (at(a.T, p) @ at(a, q) + at(a, p) @ at(a.T, q))
    values, vectors = np.linalg.eigh(h)
    for q in range(n):
        _, tones = chain._line_tones(q, False)
        first = np.argmin(abs(values - tones[0, 1]))
        assert values[first] == pytest.approx(tones[0, 1], abs=1.0)
        raised = at(a.T, q) @ vectors[:, first]
        strength = (vectors.T @ raised) ** 2
        lines = values[strength > 0.1] - values[first]
        assert min(abs(lines - tones[1, 2])) < 50e3


def test_a_part_simulated_at_two_levels_is_not_corrected():
    from cudaq_pulse.quantum_systems.base import layout_anharmonicities

    description = pulse.quantum_system("transmon_surface17")._description
    assert set(layout_anharmonicities(description, (4, 14), (2, 3))) == {14}
    assert set(layout_anharmonicities(description, (14, 4), (2, 3))) == {4}
    assert not layout_anharmonicities(description, (4, 14), (2, 2))
    assert set(layout_anharmonicities(description, (4, 14), (3, 3))) == {4, 14}


def test_a_tone_the_line_does_not_have_is_not_found():
    alone, _, _, _ = _pair_alone("missing")
    _, tones = alone._line_tones(alone.qudits[0].index, False)
    with pytest.raises(pulse.NotFoundError, match=r"no tone \(5, 6\); it has \(0, 1\)"):
        tones[5, 6]
    assert tones.get((5, 6)) is None


def _offset_pair(reference=5e9, frames="dressed"):
    # D5 beside Z2, their frequencies as offsets from *reference*.
    from cudaq_pulse.quantum_systems.base import Offset

    alone, d, z, pair = _pair_alone(f"offsets {reference}")
    qudits = {
        i: q.replace(frequency_hz=Offset(q.frequency_hz - reference), readout=None)
        for i, q in ((4, d), (14, z))
    }
    return pulse.quantum_system(
        name=f"offsets {reference}", qudits=qudits, couplings=[pair]
    ).replace(frames=frames)


def test_offsets_from_a_shared_reference_dress_as_the_frequencies_do():
    from cudaq_pulse.quantum_systems.base import Offset

    reference = 5e9
    absolute, _, _, _ = _pair_alone("absolute")
    offsets = _offset_pair(reference)
    assert offsets.frequency_frame == "shared_reference"
    for index in (4, 14):
        dressed = offsets.qudits[
            [q.index for q in offsets.qudits].index(index)
        ].dressed_frequency_hz
        whole = absolute.qudits[
            [q.index for q in absolute.qudits].index(index)
        ].dressed_frequency_hz
        # The exchange keeps the number of excitations: every level of k
        # excitations moves by k times the reference.
        assert dressed.__class__ is Offset
        assert float(dressed) == pytest.approx(whole - reference, abs=1e-3)
        _, tones = offsets._line_tones(index, False)
        _, tones_absolute = absolute._line_tones(index, False)
        assert tones[0, 1] == pytest.approx(tones_absolute[0, 1] - reference, abs=1e-3)
        # A difference of a 2-excitation and a 1-excitation level: 2R - R.
        assert tones[1, 2] == pytest.approx(tones_absolute[1, 2] - reference, abs=1e-3)
        assert tones[0, 2] == pytest.approx(tones_absolute[0, 2] - 2 * reference, abs=1e-3)


def test_bare_frames_on_offsets_keep_both():
    bare = _offset_pair(frames="bare")
    assert bare.frequency_frame == "shared_reference"
    assert bare._description.frames == "bare"
    d5 = bare.qudits[0]
    assert d5.dressed_frequency_hz == bare.physical_qudits[d5.index].frequency_hz
    dressed = _offset_pair()
    assert bare.fingerprint != dressed.fingerprint
    assert bare.replace(frames="dressed").fingerprint == dressed.replace().fingerprint


def test_a_system_without_frequencies_has_nothing_to_dress():
    full = pulse.quantum_system("transmon_surface17")._description
    qudits = {i: full.qudits[i].replace(frequency_hz=None, readout=None) for i in (4, 14)}
    qs = pulse.quantum_system(
        name="own carriers",
        qudits=qudits,
        residual_zz=[pulse.quantum_systems.ResidualZZ((4, 14), -50e3)],
    )
    assert qs.frequency_frame == "own_carrier"
    assert [q.dressed_frequency_hz for q in qs.qudits] == [None, None]
    assert "_line_tones" not in vars(qs)


def test_transition_freq_keeps_the_bare_signed_transitions():
    alone, d, _, _ = _pair_alone("signed")
    q = [h.index for h in alone.qudits].index(4)
    table = alone.transition_freq
    alpha = d.anharmonicity_hz
    assert table[q, 0, 1] == pytest.approx(d.frequency_hz)
    assert table[q, 1, 2] == pytest.approx(d.frequency_hz + alpha)
    # Signed: the upper level's energy less the lower's, either way round.
    assert table[q, 2, 1] == pytest.approx(d.frequency_hz + alpha)
    # The drive line plays the dressed tones, which the table does not hold.
    _, tones = alone._line_tones(4, False)
    assert tones[1, 2] != pytest.approx(table[q, 1, 2], abs=1.0)
