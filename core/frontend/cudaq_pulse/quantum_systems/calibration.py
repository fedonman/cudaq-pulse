# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantum system calibrations: fitted circuit values, kept apart from published systems.

A calibration file is JSON whose leaves are ``{value, units, kind, source,
uncertainty, measurement_run[, method]}`` objects: ``kind`` is ``published``,
``digitized`` or ``fitted``, ``source`` cites where the value comes from and
``measurement_run`` names one of the file's ``measurement_runs``. ``Calibration.load``
reads one, a shipped name (``"transmon_surface17"``) or a path, for one measurement run.

A calibration gives each flux-tunable qubit its circuit (E_C, E_J,max and
SQUID asymmetry d, so flux in flux quanta converts to frequency and back)
and each coupled pair its charge coupling g_c, from which the exchange
J(w_a, w_d) and the idle ZZ follow at any operating point. It may also give
each qubit's single-qubit pulses (the DRAG coefficient of its ``X`` and an
``X`` of the 1-2 transition, ``single_qubit_gates``) and its readout
assignment errors (``readout.assignment``).
"""

from __future__ import annotations

import copy
import os
import threading
from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any

from .._errors import InvalidValueError, NotFoundError

if TYPE_CHECKING:
    from .transmon_model import ChargeCoupledPair, TransmonCircuit

# A path string, not a pathlib.Path: pathlib (with urllib.parse) would add a
# tenth to the time ``import cudaq_pulse`` takes.
_SHIPPED = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibrations")


def _shipped_file(name: str) -> str:
    return os.path.join(_SHIPPED, name + ".json")


def _read(path: str) -> Any:
    import json

    with open(path, encoding="utf-8") as file:
        return json.load(file)


# The flux at which a qubit idles, by the sweet spot it idles at.
_SWEET_SPOT_FLUX = {"upper": 0.0, "lower": 0.5}


class Calibration:
    """One measurement run of a calibration file; see the module docstring.

    Qubits and pairs are named by their labels (``"D4"``, ``"X2-D4"`` with
    the auxiliary first).
    """

    def __init__(self, data: Mapping[str, Any], *, name: str, measurement_run: str | None = None):
        self._name(data, name, measurement_run)
        self._data = copy.deepcopy(dict(data))

    def _name(self, data: Mapping[str, Any], name: str, measurement_run: str | None) -> None:
        schema = data.get("_schema", {})
        measurement_run = (
            schema.get("default_measurement_run") if measurement_run is None else measurement_run
        )
        measurement_runs = data.get("measurement_runs", {})
        if measurement_run not in measurement_runs:
            raise InvalidValueError(
                f"calibration {name!r} has no measurement run {measurement_run!r}; "
                f"it has {', '.join(sorted(measurement_runs))}"
            )
        self.name = name
        self.measurement_run: str = measurement_run
        # The digest of the shipped file this was loaded from, if it was.
        self._shipped: str | None = None

    @cached_property
    def _data(self) -> dict[str, Any]:
        # Set by __init__; a shipped calibration that was unpickled reads its
        # file when its values are first used.
        data: dict[str, Any] = _read(_shipped_file(self.name))
        return data

    @cached_property
    def quantum_system(self) -> str | None:
        """The quantum system the calibration was made for, if its file names one."""
        target: str | None = self._data.get("_schema", {}).get("quantum_system")
        return target

    @classmethod
    def load(
        cls,
        source: str | os.PathLike[str] = "transmon_surface17",
        *,
        measurement_run: str | None = None,
    ) -> Calibration:
        """A shipped calibration by name, or a calibration file by path."""
        path = name = os.fspath(source)
        stem, suffix = os.path.splitext(os.path.basename(path))
        digest = None
        if not suffix and os.path.isfile(_shipped_file(name)):
            from .._source_stamp import CALIBRATIONS

            path, digest = _shipped_file(name), CALIBRATIONS.get(name)
        elif not os.path.isfile(path):
            shipped = sorted(f[:-5] for f in os.listdir(_SHIPPED) if f.endswith(".json"))
            raise NotFoundError(
                f"no calibration {name!r}: not a file, and the shipped ones are "
                + ", ".join(shipped)
            )
        else:
            name = stem
        data = _read(path)
        loaded = cls.__new__(cls)
        loaded._name(data, name, measurement_run)
        # Not copied: parsed here, it is held by this calibration alone, which
        # only reads it and hands out copies (entry, value, measurement_runs).
        loaded._data = data
        loaded._shipped = digest
        return loaded

    def measurement_runs(self) -> Mapping[str, str]:
        """Each measurement run of the file and what it is."""
        return dict(self._data["measurement_runs"])

    def entry(self, *path: str) -> Mapping[str, Any]:
        """The leaf at *path* with its provenance, e.g. ``entry("qudits", "D4", "ec")``."""
        node: Any = self._data
        for key in path:
            node = node[key]
        if not (isinstance(node, Mapping) and "value" in node):
            raise NotFoundError(f"{'/'.join(path)} is not a calibrated value")
        return copy.deepcopy(dict(node))

    def value(self, *path: str) -> Any:
        """The value of the leaf at *path*."""
        return self.entry(*path)["value"]

    def __repr__(self) -> str:
        return f"<Calibration {self.name} ({self.measurement_run})>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Calibration) and other.fingerprint == self.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    # Read-only: a copy is the calibration itself.
    def __copy__(self) -> Calibration:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Calibration:
        return self

    def __reduce_ex__(self, protocol: Any) -> Any:
        # A shipped calibration pickles as its name, not its values: the
        # package that unpickles it ships it, checked by the file's digest.
        if self._shipped is None or type(self) is not Calibration:
            return super().__reduce_ex__(protocol)
        from .._build_config import PACKAGE_VERSION
        from .._pickling import rebuilt_shipped_calibration

        return (
            rebuilt_shipped_calibration,
            (self.name, self.measurement_run, self._shipped, PACKAGE_VERSION),
        )

    @cached_property
    def fingerprint(self) -> str:
        import hashlib
        import json

        content = json.dumps(self._data, sort_keys=True).encode() + self.measurement_run.encode()
        return hashlib.blake2b(content, digest_size=16).hexdigest()

    # What a quantum system's fingerprint takes of it: a shipped calibration's
    # name, measurement run and the file's digest, recorded when the package was
    # built; any other's ``fingerprint``, a hash of its contents.
    @property
    def _identity(self) -> tuple[str, ...]:
        if self._shipped is not None:
            return ("shipped", self.name, self.measurement_run, self._shipped)
        return ("contents", self.fingerprint)

    # ── Circuits and couplings ──────────────────────────────────────────

    @property
    def flux_qudits(self) -> tuple[str, ...]:
        """The qudits whose circuit, including d, is calibrated."""
        return tuple(
            label for label, q in self._data["qudits"].items() if q.get("flux_quanta_capable")
        )

    def circuit(self, qudit: str) -> TransmonCircuit:
        """The circuit of *qudit*, for flux in flux quanta."""
        from .transmon_model import TransmonCircuit

        if qudit not in self.flux_qudits:
            raise NotFoundError(f"calibration {self.name!r} has no circuit for qudit {qudit!r}")
        q = self._data["qudits"][qudit]
        return TransmonCircuit(
            q["ec"]["value"], q["ej_max"]["value"], q["asymmetry_d"]["value"], qudit
        )

    def idle_flux(self, qudit: str) -> float:
        """The flux at which *qudit* idles: its calibrated sweet spot."""
        return _SWEET_SPOT_FLUX[self._data["qudits"][qudit]["sweet_spot_idle"]]

    @property
    def pairs(self) -> tuple[str, ...]:
        """The coupled pairs, auxiliary first."""
        return tuple(self._couplings()["idle_exchange"])

    def charge_coupling_hz(self, pair: str) -> float:
        """g_c of *pair* (``"X2-D4"``): the system-wide charge coupling."""
        if pair not in self._couplings()["idle_exchange"]:
            raise NotFoundError(f"calibration {self.name!r} has no coupling for pair {pair!r}")
        couplings = self._couplings()
        value: float = couplings[couplings["used"]]["value"]
        return value

    def pair(self, pair: str, level_count: int = 5) -> ChargeCoupledPair:
        """The charge-coupled pair, data transmon first (so ``|02>`` is the
        auxiliary's second excited state)."""
        from .transmon_model import ChargeCoupledPair

        auxiliary, data = pair.split("-")
        return ChargeCoupledPair(
            self.circuit(data), self.circuit(auxiliary), self.charge_coupling_hz(pair), level_count
        )

    def exchange_hz(
        self, pair: str, auxiliary_hz: float | None = None, data_hz: float | None = None
    ) -> float:
        """The 0-1 exchange J_qq of *pair* with its qubits at these 0-1
        frequencies; at their idle points by default."""
        auxiliary, data = pair.split("-")
        coupled = self.pair(pair)
        flux_a = (
            self.idle_flux(auxiliary)
            if auxiliary_hz is None
            else coupled.b.flux_for_f01(auxiliary_hz)
        )
        flux_d = self.idle_flux(data) if data_hz is None else coupled.a.flux_for_f01(data_hz)
        return coupled.j_qq_hz(flux_d, flux_a)

    def idle_zz_hz(self, pair: str) -> float:
        """The static ZZ of *pair* at its idle points, from the dressed
        levels of the charge-coupled pair."""
        auxiliary, data = pair.split("-")
        return self.pair(pair).idle_zz_hz(self.idle_flux(data), self.idle_flux(auxiliary))

    # ── Flux quanta ─────────────────────────────────────────────────────

    def detuning_for_flux(self, qudit: str, flux: float) -> float:
        """The 0-1 detuning (Hz) from the idle point that a flux of *flux*
        flux quanta tunes *qudit* to."""
        circuit = self.circuit(qudit)
        return circuit.f01_hz(flux) - circuit.f01_hz(self.idle_flux(qudit))

    def flux_for_detuning(self, qudit: str, detuning_hz: float) -> float:
        """The flux (flux quanta, in [0, 1/2]) that detunes *qudit* by
        *detuning_hz* from its idle point."""
        circuit = self.circuit(qudit)
        return circuit.flux_for_f01(circuit.f01_hz(self.idle_flux(qudit)) + detuning_hz)

    # ── Targets ─────────────────────────────────────────────────────────

    # *target* with each calibrated pair's coupling set to its exchange
    # at the qubits' frequencies in the target (``_exchange``), and each
    # calibrated qudit's pulses (``_calibrated_pulses``) and readout errors
    # set. A qudit matches its label.
    def _apply(self, target: Any) -> Any:
        from .base import Coupling, DispersiveReadout
        from .transmon import _calibrated_pulses

        qudits = target.qudits
        idle_exchange = self._idle_exchange
        couplings = []
        for c in target.couplings:
            a, b = qudits[c.qudits[0]], qudits[c.qudits[1]]
            strength = self._exchange(
                idle_exchange, f"{a.label}-{b.label}", a._circuit_hz, b._circuit_hz
            )
            if strength is None:
                strength = self._exchange(
                    idle_exchange, f"{b.label}-{a.label}", b._circuit_hz, a._circuit_hz
                )
            if strength is not None:
                if c.__class__ is Coupling:
                    c = Coupling(c.qudits, strength, c._calibrations)
                else:
                    c = c.replace(strength_hz=strength)
            couplings.append(c)
        pulses = self._qudit_pulses()
        if not pulses:
            return target.replace(couplings=couplings)
        made: dict[Any, Any] = {}
        calibrated = {}
        for i, q in qudits.items():
            values = pulses.get(q.label)
            if values is not None:
                beta, pulse, errors = values
                changes: dict[str, Any] = {}
                if beta is not None or pulse is not None:
                    changes["calibrations"] = _calibrated_pulses(q._calibrations, beta, pulse, made)
                readout = q.readout
                if errors is not None and readout.__class__ is DispersiveReadout:
                    changes["readout"] = readout.replace(error_0to1=errors[0], error_1to0=errors[1])
                if changes:
                    q = q.replace(**changes)
            calibrated[i] = q
        return target.replace(qudits=calibrated, couplings=couplings)

    # The exchange of *pair* (``"X2-D4"``) at its qubits' frequencies
    # *at_a* and *at_b*, or None if the calibration does not calibrate it.
    # A pair whose qubits sit at the idle frequencies its entry records
    # takes the exchange stored with it, computed when the file was built.
    def _exchange(
        self, idle_exchange: Mapping[str, tuple[float, Any]], pair: str, at_a: float, at_b: float
    ) -> float | None:
        found = idle_exchange.get(pair)
        if found is None:
            return None
        value, stored = found
        if stored is not None and abs(at_a - stored[0]) < 1e3 and abs(at_b - stored[1]) < 1e3:
            return value
        return self.exchange_hz(pair, at_a, at_b)

    # Each calibrated qudit's values by label, in the file's SI units:
    # (the DRAG coefficient of its ``X``, its 1-2 ``X`` as (duration,
    # sigma, amplitude, beta), its (error_0to1, error_1to0)), each None
    # where the file gives none. Read once; preset for an indexed shipped
    # calibration.
    def _qudit_pulses(self) -> Mapping[str, tuple[Any, Any, Any]]:
        found: dict[str, tuple[Any, Any, Any]] | None = self.__dict__.get("_pulses")
        if found is not None:
            return found
        data = self._data
        gates = data.get("single_qubit_gates", {})
        x, x12 = gates.get("x", {}), gates.get("x12", {})
        assignment = data.get("readout", {}).get("assignment", {})
        found = {}
        for label in data.get("qudits", {}):
            beta = x[label]["beta"]["value"] if label in x else None
            pulse = None
            if label in x12:
                pulse = (
                    x12["duration"]["value"],
                    x12["sigma"]["value"],
                    x12[label]["amplitude"]["value"],
                    x12[label]["beta"]["value"],
                )
            errors = None
            if label in assignment:
                errors = (
                    assignment[label]["error_0to1"]["value"],
                    assignment[label]["error_1to0"]["value"],
                )
            if (beta, pulse, errors) != (None, None, None):
                found[label] = (beta, pulse, errors)
        self._pulses = found
        return found

    # Each pair's stored idle exchange and the idle frequencies it was
    # computed at (None if not recorded); preset for an indexed shipped
    # calibration (see ``shipped_calibration``).
    @cached_property
    def _idle_exchange(self) -> Mapping[str, tuple[float, Any]]:
        return {
            pair: (entry["value"], entry.get("idle_frequencies_hz"))
            for pair, entry in self._couplings()["idle_exchange"].items()
        }

    def _couplings(self) -> Mapping[str, Any]:
        couplings: Mapping[str, Any] = self._data["coupling"]["charge_coupling_g_c"]
        return couplings


_LOADED: dict[tuple[str, str | None], Calibration] = {}
_LOADED_LOCK = threading.Lock()


def shipped_calibration(name: str, measurement_run: str | None = None) -> Calibration:
    """A shipped calibration, loaded once: its default measurement run asked for by
    name or as ``None`` is one calibration. One in ``_shipped_calibrations``
    is not read until a value beyond what a quantum system takes is used."""
    key = (name, measurement_run)
    loaded = _LOADED.get(key)
    if loaded is None:
        with _LOADED_LOCK:
            loaded = _LOADED.get(key)
            if loaded is None:
                loaded = _indexed(name, measurement_run) or Calibration.load(
                    name, measurement_run=measurement_run
                )
                loaded = _LOADED.setdefault((name, loaded.measurement_run), loaded)
                _LOADED[key] = loaded
    return loaded


# The shipped calibration *name* (*measurement_run*), unread, from
# ``_shipped_calibrations`` if its entry is of the file this package ships;
# None otherwise, and for a measurement run the file does not have.
def _indexed(name: str, measurement_run: str | None) -> Calibration | None:
    from .._source_stamp import CALIBRATIONS
    from ._shipped_calibrations import SHIPPED

    entry = SHIPPED.get(name)
    if entry is None or CALIBRATIONS.get(name) != entry[0]:
        return None
    digest, default, measurement_runs, idle_exchange, qudit_pulses = entry
    measurement_run = default if measurement_run is None else measurement_run
    if measurement_run not in measurement_runs:
        return None
    unread = Calibration.__new__(Calibration)
    unread.name, unread.measurement_run, unread._shipped = name, measurement_run, digest
    unread._idle_exchange = idle_exchange
    unread._pulses = dict(qudit_pulses)
    return unread


# The shipped calibration *name* (*measurement_run*), if this package ships the
# file of that *digest*, without reading the file until its values are used.
def _shipped_unread(name: str, measurement_run: str, digest: str) -> Calibration | None:
    from .._source_stamp import CALIBRATIONS

    if CALIBRATIONS.get(name) != digest:
        return None
    found = _LOADED.get((name, measurement_run))
    if found is None:
        unread = Calibration.__new__(Calibration)
        unread.name, unread.measurement_run, unread._shipped = name, measurement_run, digest
        with _LOADED_LOCK:
            found = _LOADED.setdefault((name, measurement_run), unread)
    return found
