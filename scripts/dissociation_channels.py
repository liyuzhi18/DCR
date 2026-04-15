#!/usr/bin/env python3
"""
Channel-resolved H2 dissociation postprocessing.

This module decomposes the existing total MCCC neutral-fragment dissociation
cross section into:
  - DE from all available local singlet-state MCCC fits
  - explicit ERDD / PD for states with supplied decay data
  - an effective bound-state dissociation remainder for the other bound singlets

The implementation is intentionally postprocessing-only and leaves the runtime
solver dissociation model unchanged.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BOHR_CM = 5.29177210903e-9
A0_CM2 = BOHR_CM * BOHR_CM
GROUND_STATE = "X1Sg"

EXPLICIT_STATES = ("B1Su", "Bp1Su", "C1Pu", "D1Pu", "EF1Sg")
REMAINDER_STATES = ("GK1Sg", "H1Sg", "I1Pg", "J1Dg")
ALL_STATES = EXPLICIT_STATES + REMAINDER_STATES
TRIPLET_DE_STATES = ("a3Sg", "b3Su", "c3Pu")

TEMPLATE_SCHEMAS = {
    "radiative_bound.csv": ("upper_state", "upper_v", "lower_state", "lower_v", "A_s_inv"),
    "radiative_continuum.csv": ("upper_state", "upper_v", "target_state", "A_cont_s_inv"),
    "pd_fraction.csv": ("state", "vi", "F_pd"),
    "cascade_map.csv": ("upper_state", "upper_v", "lower_state", "lower_v_or_continuum"),
    "nonadiabatic_sigma_g_levels.csv": ("k", "energy_cm_inv", "A_tot_s_inv"),
    "nonadiabatic_sigma_g_eigenvectors.csv": ("k", "upper_state", "upper_v", "coeff"),
    "nonadiabatic_sigma_g_bound.csv": ("k", "lower_state", "lower_v", "A_s_inv"),
    "state_decay_summary.csv": (
        "state",
        "upper_v",
        "A_cont_s_inv",
        "A_bound_s_inv",
        "A_tot_s_inv",
        "tau_s",
        "f_cont",
    ),
}


class DissociationChannelError(RuntimeError):
    """Raised for invalid reference data or unsupported bookkeeping."""


@dataclass(frozen=True)
class FitRow:
    label: str
    vi: int
    threshold_eV: float
    coeffs: tuple[float, ...]
    is_de: bool


@dataclass
class StateFitData:
    state: str
    vi: int
    bound_rows: dict[int, FitRow]
    de_row: FitRow | None


@dataclass(frozen=True)
class BoundTransition:
    upper_state: str
    upper_v: int
    lower_state: str
    lower_v: int
    a_s_inv: float


@dataclass(frozen=True)
class ContinuumTransition:
    upper_state: str
    upper_v: int
    target_state: str
    a_cont_s_inv: float


@dataclass(frozen=True)
class CascadeEdge:
    upper_state: str
    upper_v: int
    lower_state: str
    lower_v_or_continuum: str


@dataclass
class ReferenceTables:
    bound: dict[tuple[str, int], list[BoundTransition]]
    continuum: dict[tuple[str, int], list[ContinuumTransition]]
    pd_fraction: dict[tuple[str, int], float]
    cascade_edges: set[CascadeEdge]


@dataclass
class ViChannelBreakdown:
    vi: int
    energy_eV: np.ndarray
    sigma_total_mccc_cm2: np.ndarray
    raw_channels: dict[str, np.ndarray]
    norm_channels: dict[str, np.ndarray]
    raw_sum_cm2: np.ndarray
    norm_sum_cm2: np.ndarray
    renorm_scale: np.ndarray
    renorm_support: np.ndarray
    f_eff: np.ndarray


def _sanitize_state_key(value: str) -> str:
    text = value.strip()
    text = text.replace("′", "'")
    text = text.replace("Σ", "SIGMA")
    text = text.replace("Π", "PI")
    text = text.replace("\\Sigma", "SIGMA")
    text = text.replace("\\Pi", "PI")
    text = text.replace(" ", "")
    text = text.replace(",", "")
    text = text.replace("^", "")
    text = text.replace("_", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("+", "")
    text = text.replace("'", "P")
    return text.upper()


def normalize_state_label(value: str) -> str:
    key = _sanitize_state_key(value)
    alias_map = {
        "B1SIGMAU": "B1Su",
        "B1SU": "B1Su",
        "BP1SIGMAU": "Bp1Su",
        "BP1SU": "Bp1Su",
        "C1PIU": "C1Pu",
        "C1PU": "C1Pu",
        "D1PIU": "D1Pu",
        "D1PU": "D1Pu",
        "EF1SIGMAG": "EF1Sg",
        "EF1SG": "EF1Sg",
        "EFSIGMAG": "EF1Sg",
        "GK1SIGMAG": "GK1Sg",
        "GK1SG": "GK1Sg",
        "H1SIGMAG": "H1Sg",
        "H1SG": "H1Sg",
        "I1PIG": "I1Pg",
        "I1PG": "I1Pg",
        "J1DELTAG": "J1Dg",
        "J1DG": "J1Dg",
        "X1SIGMAG": "X1Sg",
        "X1SG": "X1Sg",
        "X": "X1Sg",
    }
    normalized = alias_map.get(key)
    if normalized is None:
        raise DissociationChannelError(f"Unknown state label '{value}'")
    return normalized


def write_template_tables(table_dir: Path) -> list[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, header in TEMPLATE_SCHEMAS.items():
        path = table_dir / name
        if not path.exists():
            path.write_text(",".join(header) + "\n", encoding="utf-8")
        written.append(path)
    return written


def _csv_row_iter(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        filtered = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
    if not filtered:
        return []
    reader = csv.DictReader(filtered)
    return list(reader)


def _require_columns(path: Path, rows: list[dict[str, str]], expected: tuple[str, ...]) -> None:
    if not rows:
        with path.open(newline="", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",") if fh.readable() else []
        if tuple(header) != expected:
            raise DissociationChannelError(
                f"{path} has header {header}, expected {list(expected)}"
            )
        return
    actual = tuple(rows[0].keys())
    if actual != expected:
        raise DissociationChannelError(f"{path} has columns {list(actual)}, expected {list(expected)}")


def _load_table_rows(table_dir: Path, filename: str) -> list[dict[str, str]]:
    path = table_dir / filename
    if not path.exists():
        raise DissociationChannelError(f"Missing required table: {path}")
    rows = _csv_row_iter(path)
    _require_columns(path, rows, TEMPLATE_SCHEMAS[filename])
    return rows


def load_reference_tables(table_dir: Path) -> ReferenceTables:
    write_template_tables(table_dir)

    bound_rows = _load_table_rows(table_dir, "radiative_bound.csv")
    cont_rows = _load_table_rows(table_dir, "radiative_continuum.csv")
    pd_rows = _load_table_rows(table_dir, "pd_fraction.csv")
    cascade_rows = _load_table_rows(table_dir, "cascade_map.csv")
    nad_levels_rows = _load_table_rows(table_dir, "nonadiabatic_sigma_g_levels.csv")
    nad_evec_rows = _load_table_rows(table_dir, "nonadiabatic_sigma_g_eigenvectors.csv")
    nad_bound_rows = _load_table_rows(table_dir, "nonadiabatic_sigma_g_bound.csv")
    summary_rows = _load_table_rows(table_dir, "state_decay_summary.csv")

    bound: dict[tuple[str, int], list[BoundTransition]] = {}
    continuum: dict[tuple[str, int], list[ContinuumTransition]] = {}
    pd_fraction: dict[tuple[str, int], float] = {}
    cascade_edges: set[CascadeEdge] = set()

    seen_bound = set()
    for row in bound_rows:
        upper_state = normalize_state_label(row["upper_state"])
        lower_state = normalize_state_label(row["lower_state"])
        upper_v = int(row["upper_v"])
        lower_v = int(row["lower_v"])
        a_s_inv = float(row["A_s_inv"])
        if a_s_inv <= 0.0:
            raise DissociationChannelError(f"Non-positive A_s_inv for {row}")
        key = (upper_state, upper_v, lower_state, lower_v)
        if key in seen_bound:
            raise DissociationChannelError(f"Duplicate radiative_bound row for {key}")
        seen_bound.add(key)
        bound.setdefault((upper_state, upper_v), []).append(
            BoundTransition(upper_state, upper_v, lower_state, lower_v, a_s_inv)
        )

    seen_cont = set()
    for row in cont_rows:
        upper_state = normalize_state_label(row["upper_state"])
        target_state = normalize_state_label(row["target_state"])
        upper_v = int(row["upper_v"])
        a_cont = float(row["A_cont_s_inv"])
        if a_cont <= 0.0:
            raise DissociationChannelError(f"Non-positive A_cont_s_inv for {row}")
        key = (upper_state, upper_v, target_state)
        if key in seen_cont:
            raise DissociationChannelError(f"Duplicate radiative_continuum row for {key}")
        seen_cont.add(key)
        continuum.setdefault((upper_state, upper_v), []).append(
            ContinuumTransition(upper_state, upper_v, target_state, a_cont)
        )

    for row in summary_rows:
        state = normalize_state_label(row["state"])
        upper_v = int(row["upper_v"])
        a_cont = float(row["A_cont_s_inv"])
        a_bound = float(row["A_bound_s_inv"])
        a_tot = float(row["A_tot_s_inv"])
        tau_s = float(row["tau_s"])
        f_cont = float(row["f_cont"])

        if a_cont < 0.0 or a_bound < 0.0 or a_tot <= 0.0 or tau_s <= 0.0:
            raise DissociationChannelError(f"Invalid state_decay_summary values for {row}")
        if not (0.0 <= f_cont <= 1.0):
            raise DissociationChannelError(f"f_cont outside [0,1] for {row}")

        sum_rel = abs((a_cont + a_bound) - a_tot) / a_tot
        frac_rel = abs((a_cont / a_tot) - f_cont)
        if sum_rel > 5.0e-3:
            raise DissociationChannelError(
                f"A_cont + A_bound inconsistent with A_tot for {state} v={upper_v}"
            )
        if frac_rel > 5.0e-3:
            raise DissociationChannelError(
                f"A_cont / A_tot inconsistent with f_cont for {state} v={upper_v}"
            )

        if a_bound > 0.0:
            bound_key = (state, upper_v, GROUND_STATE, -1)
            if bound_key in seen_bound:
                raise DissociationChannelError(f"Duplicate bound decay data for {state} v={upper_v}")
            seen_bound.add(bound_key)
            bound.setdefault((state, upper_v), []).append(
                BoundTransition(state, upper_v, GROUND_STATE, -1, a_bound)
            )

        if a_cont > 0.0:
            cont_key = (state, upper_v, GROUND_STATE)
            if cont_key in seen_cont:
                raise DissociationChannelError(f"Duplicate continuum decay data for {state} v={upper_v}")
            seen_cont.add(cont_key)
            continuum.setdefault((state, upper_v), []).append(
                ContinuumTransition(state, upper_v, GROUND_STATE, a_cont)
            )

    nad_level_totals: dict[int, float] = {}
    for row in nad_levels_rows:
        k = int(row["k"])
        _ = float(row["energy_cm_inv"])
        a_tot = float(row["A_tot_s_inv"])
        if k <= 0 or a_tot <= 0.0:
            raise DissociationChannelError(f"Invalid nonadiabatic_sigma_g_levels row: {row}")
        if k in nad_level_totals:
            raise DissociationChannelError(f"Duplicate nonadiabatic_sigma_g_levels row for k={k}")
        nad_level_totals[k] = a_tot

    nad_weights: dict[tuple[str, int], dict[int, float]] = {}
    for row in nad_evec_rows:
        k = int(row["k"])
        state = normalize_state_label(row["upper_state"])
        upper_v = int(row["upper_v"])
        coeff = float(row["coeff"])
        if k <= 0:
            raise DissociationChannelError(f"Invalid nonadiabatic_sigma_g_eigenvectors row: {row}")
        nad_weights.setdefault((state, upper_v), {})
        if k in nad_weights[(state, upper_v)]:
            raise DissociationChannelError(
                f"Duplicate nonadiabatic eigenvector coefficient for {state} v={upper_v}, k={k}"
            )
        nad_weights[(state, upper_v)][k] = coeff * coeff

    nad_bound: dict[int, list[BoundTransition]] = {}
    for row in nad_bound_rows:
        k = int(row["k"])
        lower_state = normalize_state_label(row["lower_state"])
        lower_v = int(row["lower_v"])
        a_s_inv = float(row["A_s_inv"])
        if k <= 0 or a_s_inv < 0.0:
            raise DissociationChannelError(f"Invalid nonadiabatic_sigma_g_bound row: {row}")
        nad_bound.setdefault(k, []).append(BoundTransition("__NONADIABATIC__", -1, lower_state, lower_v, a_s_inv))

    if nad_weights or nad_bound or nad_level_totals:
        if not nad_weights:
            raise DissociationChannelError(
                "Nonadiabatic Sigma_g tables are incomplete: missing nonadiabatic_sigma_g_eigenvectors.csv entries"
            )
        if not nad_bound:
            raise DissociationChannelError(
                "Nonadiabatic Sigma_g tables are incomplete: missing nonadiabatic_sigma_g_bound.csv entries"
            )

        for k, transitions in nad_bound.items():
            if k not in nad_level_totals:
                raise DissociationChannelError(f"Missing nonadiabatic level total for k={k}")
            a_sum = sum(entry.a_s_inv for entry in transitions)
            a_tot = nad_level_totals[k]
            if a_sum <= 0.0:
                raise DissociationChannelError(f"Zero transition sum for nonadiabatic level k={k}")
            rel = abs(a_sum - a_tot) / a_tot
            if rel > 5.0e-2:
                raise DissociationChannelError(
                    f"Nonadiabatic transition sum inconsistent with A_tot for k={k}: "
                    f"sum={a_sum:.6e}, A_tot={a_tot:.6e}"
                )

        for (state, upper_v), weights in nad_weights.items():
            total_weight = sum(weights.values())
            if total_weight <= 0.0:
                raise DissociationChannelError(
                    f"Zero nonadiabatic projection weight for {state} v={upper_v}"
                )

            derived: dict[tuple[str, int], float] = {}
            for k, raw_weight in weights.items():
                if k not in nad_bound:
                    continue
                weight = raw_weight / total_weight
                for entry in nad_bound[k]:
                    key = (entry.lower_state, entry.lower_v)
                    derived[key] = derived.get(key, 0.0) + weight * entry.a_s_inv

            if not derived:
                continue

            for (lower_state, lower_v), a_s_inv in derived.items():
                bound_key = (state, upper_v, lower_state, lower_v)
                if bound_key in seen_bound:
                    raise DissociationChannelError(
                        f"Duplicate derived nonadiabatic bound decay for {state} v={upper_v} "
                        f"-> {lower_state} v={lower_v}"
                    )
                seen_bound.add(bound_key)
                bound.setdefault((state, upper_v), []).append(
                    BoundTransition(state, upper_v, lower_state, lower_v, a_s_inv)
                )
                if lower_state != GROUND_STATE:
                    cascade_edges.add(CascadeEdge(state, upper_v, lower_state, str(lower_v)))

    for row in pd_rows:
        state = normalize_state_label(row["state"])
        vi = int(row["vi"])
        frac = float(row["F_pd"])
        if not (0.0 <= frac <= 1.0):
            raise DissociationChannelError(f"PD fraction outside [0,1] for {row}")
        key = (state, vi)
        if key in pd_fraction:
            raise DissociationChannelError(f"Duplicate pd_fraction row for {key}")
        pd_fraction[key] = frac

    for row in cascade_rows:
        upper_state = normalize_state_label(row["upper_state"])
        lower_state = normalize_state_label(row["lower_state"])
        upper_v = int(row["upper_v"])
        lower = row["lower_v_or_continuum"].strip()
        if lower.lower() != "continuum":
            lower = str(int(lower))
        edge = CascadeEdge(upper_state, upper_v, lower_state, lower)
        if edge in cascade_edges:
            raise DissociationChannelError(f"Duplicate cascade_map row for {row}")
        cascade_edges.add(edge)

    return ReferenceTables(
        bound=bound,
        continuum=continuum,
        pd_fraction=pd_fraction,
        cascade_edges=cascade_edges,
    )


def _fit_file_state(path: Path) -> str:
    match = re.search(r"MCCC-el-H2-([^.]+)\.X1Sg_vi=", path.name)
    if not match:
        raise DissociationChannelError(f"Could not parse state from fit filename: {path.name}")
    return match.group(1)


def _fit_file_vi(path: Path) -> int:
    match = re.search(r"_vi=(\d+)_fit\.txt$", path.name)
    if not match:
        raise DissociationChannelError(f"Could not parse vi from fit filename: {path.name}")
    return int(match.group(1))


def discover_fit_files(fits_dir: Path, vi: int) -> dict[str, Path]:
    vi_dir = fits_dir / f"vi={vi}"
    files = {}
    for path in sorted(vi_dir.glob("MCCC-el-H2-*.X1Sg_vi=*_fit.txt")):
        files[_fit_file_state(path)] = path
    if not files:
        raise DissociationChannelError(f"No fit files found under {vi_dir}")
    return files


def parse_state_fit_file(path: Path) -> StateFitData:
    state = _fit_file_state(path)
    vi = _fit_file_vi(path)
    bound_rows: dict[int, FitRow] = {}
    de_row: FitRow | None = None

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5 or parts[1] != "<-":
                continue
            label = parts[0]
            row_vi = int(parts[2])
            threshold = float(parts[3])
            coeffs = tuple(float(x) for x in parts[4:])
            is_de = label.upper() == "DE"
            fit_row = FitRow(label=label, vi=row_vi, threshold_eV=threshold, coeffs=coeffs, is_de=is_de)
            if is_de:
                de_row = fit_row
            else:
                bound_rows[int(label)] = fit_row

    if de_row is None:
        raise DissociationChannelError(f"Missing DE row in {path}")
    if not bound_rows:
        raise DissociationChannelError(f"No bound rows found in {path}")
    return StateFitData(state=state, vi=vi, bound_rows=bound_rows, de_row=de_row)


def evaluate_fit_row(row: FitRow, energy_eV: np.ndarray) -> np.ndarray:
    sigma = np.zeros_like(energy_eV, dtype=float)
    mask = energy_eV > row.threshold_eV
    if not np.any(mask):
        return sigma

    x = energy_eV[mask] / row.threshold_eV
    log_term = (row.coeffs[0] ** 2) * np.log(x) / x
    poly_term = np.zeros_like(x)
    for power, coeff in enumerate(row.coeffs[1:], start=1):
        poly_term += coeff / np.power(x, power)
    sigma[mask] = np.abs(((x - 1.0) / x) * (log_term + poly_term)) * A0_CM2
    return sigma


def load_total_mccc_dissociation(mccc_dir: Path, vi: int) -> tuple[np.ndarray, np.ndarray]:
    path = mccc_dir / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt"
    if not path.exists():
        raise DissociationChannelError(f"Missing total dissociation file: {path}")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise DissociationChannelError(f"Unexpected total dissociation format in {path}")
    return data[:, 0], data[:, 1] * A0_CM2


def load_optional_triplet_de_channels(
    mccc_dir: Path, vi: int, target_energy_eV: np.ndarray
) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for state in TRIPLET_DE_STATES:
        path = mccc_dir / f"MCCC-el-H2-{state}_DE.X1Sg_vi={vi}.txt"
        if not path.exists():
            channels[state] = np.zeros_like(target_energy_eV)
            continue
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise DissociationChannelError(f"Unexpected {state} dissociation format in {path}")
        energy = data[:, 0]
        sigma_cm2 = data[:, 1] * A0_CM2
        channels[state] = np.interp(target_energy_eV, energy, sigma_cm2, left=0.0, right=0.0)
    return channels


def available_vi(mccc_dir: Path) -> list[int]:
    vis = []
    for path in sorted(mccc_dir.glob("MCCC-el-H2-DISS.X1Sg_vi=*.txt")):
        match = re.search(r"vi=(\d+)", path.name)
        if match:
            vis.append(int(match.group(1)))
    return vis


def _bound_nodes_from_fit(fits_by_state: dict[str, StateFitData]) -> set[tuple[str, int]]:
    nodes = set()
    for state in EXPLICIT_STATES:
        if state not in fits_by_state:
            raise DissociationChannelError(f"Missing fit data for explicit state {state}")
        for v in fits_by_state[state].bound_rows:
            nodes.add((state, v))
    return nodes


def validate_reference_completeness(
    fits_by_state: dict[str, StateFitData],
    tables: ReferenceTables,
    vi: int,
) -> dict[str, list[int]]:
    explicit_nodes = _bound_nodes_from_fit(fits_by_state)
    errors = []
    missing_by_state: dict[str, list[int]] = {}

    for node in sorted(explicit_nodes):
        bound = tables.bound.get(node, [])
        cont = tables.continuum.get(node, [])
        if not bound and not cont:
            missing_by_state.setdefault(node[0], []).append(node[1])
            continue
        for entry in bound:
            if entry.lower_state != GROUND_STATE:
                edge = CascadeEdge(entry.upper_state, entry.upper_v, entry.lower_state, str(entry.lower_v))
                if tables.cascade_edges and edge not in tables.cascade_edges:
                    errors.append(
                        f"Missing cascade_map entry for {entry.upper_state} v={entry.upper_v} "
                        f"-> {entry.lower_state} v={entry.lower_v}"
                    )
                if (entry.lower_state, entry.lower_v) not in explicit_nodes:
                    errors.append(
                        f"Cascade target {entry.lower_state} v={entry.lower_v} has no explicit decay data"
                    )

    for (state, vi_key), frac in tables.pd_fraction.items():
        if state not in EXPLICIT_STATES:
            errors.append(f"PD fraction provided for non-explicit state {state}")
        if frac and vi_key != vi:
            # Other vi rows are allowed in the table; they are not an error.
            continue

    for state, levels in sorted(missing_by_state.items()):
        # Missing upper-level decay data are allowed. These levels are omitted from
        # the explicit ERDD/PD bookkeeping rather than treated as fatal.
        _ = levels

    if errors:
        raise DissociationChannelError("Reference table validation failed:\n- " + "\n- ".join(errors))
    return missing_by_state


def _pd_fraction_for_state(tables: ReferenceTables, state: str, vi: int) -> float:
    return tables.pd_fraction.get((state, vi), 0.0)


def _node_probabilities(
    state: str,
    upper_v: int,
    vi: int,
    tables: ReferenceTables,
    memo: dict[tuple[str, int, int], tuple[float, float]],
    visiting: set[tuple[str, int, int]],
) -> tuple[float, float]:
    key = (state, upper_v, vi)
    if key in memo:
        return memo[key]
    if key in visiting:
        raise DissociationChannelError(f"Cascade cycle detected at {state} v={upper_v}")
    visiting.add(key)

    bound_rows = tables.bound.get((state, upper_v), [])
    cont_rows = tables.continuum.get((state, upper_v), [])
    total_a = sum(row.a_s_inv for row in bound_rows) + sum(row.a_cont_s_inv for row in cont_rows)
    if total_a <= 0.0:
        raise DissociationChannelError(f"No outgoing decay probability for {state} v={upper_v}")

    pd_prob = _pd_fraction_for_state(tables, state, vi)
    radiative_survival = 1.0 - pd_prob

    erdd_prob = radiative_survival * sum(row.a_cont_s_inv for row in cont_rows) / total_a

    for row in bound_rows:
        if row.lower_state == GROUND_STATE:
            continue
        child_pd, child_erdd = _node_probabilities(
            row.lower_state, row.lower_v, vi, tables, memo, visiting
        )
        erdd_prob += radiative_survival * (row.a_s_inv / total_a) * (child_pd + child_erdd)

    visiting.remove(key)
    memo[key] = (pd_prob, erdd_prob)
    return memo[key]


def compute_effective_probabilities(
    fits_by_state: dict[str, StateFitData],
    tables: ReferenceTables,
    vi: int,
) -> tuple[dict[tuple[str, int], tuple[float, float]], dict[str, list[int]]]:
    missing_by_state = validate_reference_completeness(fits_by_state, tables, vi)
    memo: dict[tuple[str, int, int], tuple[float, float]] = {}
    out: dict[tuple[str, int], tuple[float, float]] = {}
    for state in EXPLICIT_STATES:
        for upper_v in fits_by_state[state].bound_rows:
            if upper_v in missing_by_state.get(state, []):
                continue
            out[(state, upper_v)] = _node_probabilities(state, upper_v, vi, tables, memo, set())
    return out, missing_by_state


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=float)
    mask = den > 0.0
    out[mask] = num[mask] / den[mask]
    return out


def compute_vi_breakdown(
    vi: int,
    mccc_dir: Path,
    fits_dir: Path,
    table_dir: Path,
) -> ViChannelBreakdown:
    energy_eV, total_sigma_cm2 = load_total_mccc_dissociation(mccc_dir, vi)
    fit_files = discover_fit_files(fits_dir, vi)
    fits_by_state = {state: parse_state_fit_file(path) for state, path in fit_files.items() if state in ALL_STATES}

    missing_states = [state for state in ALL_STATES if state not in fits_by_state]
    if missing_states:
        raise DissociationChannelError(f"Missing fit files for states: {missing_states}")

    tables = load_reference_tables(table_dir)
    probabilities, missing_by_state = compute_effective_probabilities(fits_by_state, tables, vi)

    raw_channels: dict[str, np.ndarray] = {}
    triplet_de = load_optional_triplet_de_channels(mccc_dir, vi, energy_eV)
    triplet_total = np.zeros_like(energy_eV)
    for state in TRIPLET_DE_STATES:
        sigma = triplet_de[state]
        raw_channels[f"DE_{state}_raw"] = sigma
        triplet_total += sigma
    raw_channels["DE_triplet_total_raw"] = triplet_total
    raw_channels["TRIPLET_total_raw"] = triplet_total

    de_total = np.zeros_like(energy_eV)
    de_explicit_total = np.zeros_like(energy_eV)
    de_remainder_total = np.zeros_like(energy_eV)
    for state in ALL_STATES:
        sigma = evaluate_fit_row(fits_by_state[state].de_row, energy_eV)
        raw_channels[f"DE_{state}_raw"] = sigma
        de_total += sigma
        if state in EXPLICIT_STATES:
            de_explicit_total += sigma
        else:
            de_remainder_total += sigma
    raw_channels["DE_singlet_total_raw"] = de_total
    raw_channels["DE_total_raw"] = de_total + triplet_total
    raw_channels["DE_explicit_total_raw"] = de_explicit_total
    raw_channels["DE_remainder_total_raw"] = de_remainder_total

    explicit_erdd_total = np.zeros_like(energy_eV)
    explicit_pd_total = np.zeros_like(energy_eV)
    explicit_bound_exc_total = np.zeros_like(energy_eV)
    explicit_bound_missing_total = np.zeros_like(energy_eV)
    for state in EXPLICIT_STATES:
        sigma_erdd = np.zeros_like(energy_eV)
        sigma_pd = np.zeros_like(energy_eV)
        sigma_bound = np.zeros_like(energy_eV)
        sigma_missing = np.zeros_like(energy_eV)
        for upper_v, row in fits_by_state[state].bound_rows.items():
            sigma = evaluate_fit_row(row, energy_eV)
            if upper_v in missing_by_state.get(state, []):
                sigma_missing += sigma
                continue
            sigma_bound += sigma
            pd_prob, erdd_prob = probabilities[(state, upper_v)]
            sigma_pd += pd_prob * sigma
            sigma_erdd += erdd_prob * sigma
        raw_channels[f"ERDD_{state}_raw"] = sigma_erdd
        raw_channels[f"PD_{state}_raw"] = sigma_pd
        raw_channels[f"BOUND_{state}_raw"] = sigma_bound
        raw_channels[f"BOUND_{state}_missing_raw"] = sigma_missing
        explicit_erdd_total += sigma_erdd
        explicit_pd_total += sigma_pd
        explicit_bound_exc_total += sigma_bound
        explicit_bound_missing_total += sigma_missing

    raw_channels["ERDD_explicit_total_raw"] = explicit_erdd_total
    raw_channels["PD_explicit_total_raw"] = explicit_pd_total
    raw_channels["BOUND_explicit_total_raw"] = explicit_bound_exc_total
    raw_channels["BOUND_explicit_missing_total_raw"] = explicit_bound_missing_total

    explicit_diss_total = de_explicit_total + explicit_erdd_total + explicit_pd_total
    explicit_excitation_total = de_explicit_total + explicit_bound_exc_total
    raw_channels["DISS_explicit_total_raw"] = explicit_diss_total
    raw_channels["EXCITATION_explicit_total_raw"] = explicit_excitation_total

    f_eff = _safe_divide(explicit_diss_total, explicit_excitation_total)
    f_eff_applied = np.where(np.isfinite(f_eff), np.clip(f_eff, 0.0, 1.0), 0.0)
    raw_channels["F_eff_raw"] = f_eff
    raw_channels["F_eff_applied"] = f_eff_applied

    effective_total = np.zeros_like(energy_eV)
    remainder_excitation_total = np.zeros_like(energy_eV)
    for state in REMAINDER_STATES:
        sigma_bound = np.zeros_like(energy_eV)
        for row in fits_by_state[state].bound_rows.values():
            sigma_bound += evaluate_fit_row(row, energy_eV)
        sigma_exc = sigma_bound + raw_channels[f"DE_{state}_raw"]
        sigma_eff = f_eff_applied * sigma_exc
        raw_channels[f"BOUND_{state}_raw"] = sigma_bound
        raw_channels[f"EXCITATION_{state}_raw"] = sigma_exc
        raw_channels[f"EFFECTIVE_{state}_raw"] = sigma_eff
        effective_total += sigma_eff
        remainder_excitation_total += sigma_exc
    raw_channels["EFFECTIVE_total_raw"] = effective_total
    raw_channels["EXCITATION_remainder_total_raw"] = remainder_excitation_total

    raw_sum = triplet_total + explicit_diss_total + effective_total
    raw_channels["sigma_raw_sum"] = raw_sum
    raw_channels["sigma_total_mccc"] = total_sigma_cm2

    support_mask = raw_sum > 0.0
    renorm_scale = np.full_like(raw_sum, np.nan)
    renorm_scale[support_mask] = total_sigma_cm2[support_mask] / raw_sum[support_mask]
    norm_channels = {}
    for name, values in raw_channels.items():
        if name in {"sigma_total_mccc", "sigma_raw_sum"}:
            continue
        if name == "F_eff_raw":
            continue
        if name == "F_eff_applied":
            norm_channels[name.replace("_raw", "_norm")] = values.copy()
            continue
        norm_channels[name.replace("_raw", "_norm")] = values * renorm_scale
    norm_sum = raw_sum * renorm_scale
    norm_channels["sigma_norm_sum"] = norm_sum
    norm_channels["renorm_scale"] = renorm_scale

    return ViChannelBreakdown(
        vi=vi,
        energy_eV=energy_eV,
        sigma_total_mccc_cm2=total_sigma_cm2,
        raw_channels=raw_channels,
        norm_channels=norm_channels,
        raw_sum_cm2=raw_sum,
        norm_sum_cm2=norm_sum,
        renorm_scale=renorm_scale,
        renorm_support=support_mask,
        f_eff=f_eff,
    )


def _plottable(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def write_breakdown_csv(result: ViChannelBreakdown, outpath: Path) -> None:
    columns: list[tuple[str, np.ndarray]] = [
        ("E_eV", result.energy_eV),
        ("sigma_total_mccc_cm2", result.sigma_total_mccc_cm2),
        ("sigma_raw_sum_cm2", result.raw_sum_cm2),
        ("sigma_norm_sum_cm2", result.norm_sum_cm2),
        ("renorm_scale", result.renorm_scale),
        ("renorm_support", result.renorm_support.astype(int)),
    ]
    for key in sorted(result.raw_channels):
        if key in {"sigma_total_mccc", "sigma_raw_sum"}:
            continue
        columns.append((key, result.raw_channels[key]))
    for key in sorted(result.norm_channels):
        if key in {"sigma_norm_sum", "renorm_scale"}:
            continue
        columns.append((key, result.norm_channels[key]))

    header = ",".join(name for name, _ in columns)
    data = np.column_stack([values for _, values in columns])
    np.savetxt(outpath, data, delimiter=",", header=header, comments="")


def write_summary_csv(results: list[ViChannelBreakdown], outpath: Path) -> None:
    with outpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "vi",
                "n_energy",
                "support_points",
                "support_fraction",
                "raw_max_rel_gap_on_support",
                "raw_mean_rel_gap_on_support",
                "f_eff_min",
                "f_eff_max",
            ]
        )
        for result in results:
            support = result.renorm_support
            if np.any(support):
                rel = np.abs(result.raw_sum_cm2[support] - result.sigma_total_mccc_cm2[support]) / np.maximum(
                    result.sigma_total_mccc_cm2[support], 1.0e-300
                )
                raw_max = float(np.nanmax(rel))
                raw_mean = float(np.nanmean(rel))
            else:
                raw_max = math.nan
                raw_mean = math.nan

            finite_f = result.f_eff[np.isfinite(result.f_eff)]
            writer.writerow(
                [
                    result.vi,
                    result.energy_eV.size,
                    int(np.count_nonzero(support)),
                    float(np.count_nonzero(support) / result.energy_eV.size),
                    raw_max,
                    raw_mean,
                    float(np.nanmin(finite_f)) if finite_f.size else math.nan,
                    float(np.nanmax(finite_f)) if finite_f.size else math.nan,
                ]
            )


def plot_breakdown(result: ViChannelBreakdown, outpath: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(outpath) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 10.0), constrained_layout=True)
        ax = axes[0]
        ax.plot(result.energy_eV, _plottable(result.sigma_total_mccc_cm2), color="black", lw=2.2, label="Total MCCC")
        ax.plot(result.energy_eV, _plottable(result.raw_sum_cm2), color="#1f77b4", lw=1.8, label="Raw sum")
        ax.plot(result.energy_eV, _plottable(result.norm_sum_cm2), "--", color="#d62728", lw=1.8, label="Renormalized sum")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Incident energy $E$ [eV]")
        ax.set_ylabel(r"$\sigma(E)$ [cm$^2$]")
        ax.set_title(f"H2 dissociation closure comparison (vi={result.vi})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.plot(result.energy_eV, _plottable(result.raw_channels["DE_total_raw"]), lw=1.8, label="DE total")
        ax.plot(result.energy_eV, _plottable(result.raw_channels["ERDD_explicit_total_raw"]), lw=1.8, label="ERDD explicit")
        ax.plot(result.energy_eV, _plottable(result.raw_channels["PD_explicit_total_raw"]), lw=1.8, label="PD explicit")
        ax.plot(result.energy_eV, _plottable(result.raw_channels["EFFECTIVE_total_raw"]), lw=1.8, label="Effective remainder")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Incident energy $E$ [eV]")
        ax.set_ylabel(r"$\sigma(E)$ [cm$^2$]")
        ax.set_title(f"Mechanism totals (raw) (vi={result.vi})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10.0, 10.0), constrained_layout=True)
        ax = axes[0]
        for state in ALL_STATES + TRIPLET_DE_STATES:
            if f"DE_{state}_raw" not in result.raw_channels:
                continue
            ax.plot(result.energy_eV, _plottable(result.raw_channels[f"DE_{state}_raw"]), lw=1.4, label=state)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Incident energy $E$ [eV]")
        ax.set_ylabel(r"$\sigma^{DE}_s(E)$ [cm$^2$]")
        ax.set_title(f"State-resolved DE channels (vi={result.vi})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=3)

        ax = axes[1]
        for state in EXPLICIT_STATES:
            ax.plot(result.energy_eV, _plottable(result.raw_channels[f"ERDD_{state}_raw"]), lw=1.4, label=f"ERDD {state}")
        for state in EXPLICIT_STATES:
            ax.plot(result.energy_eV, _plottable(result.raw_channels[f"PD_{state}_raw"]), "--", lw=1.2, label=f"PD {state}")
        for state in REMAINDER_STATES:
            ax.plot(result.energy_eV, _plottable(result.raw_channels[f"EFFECTIVE_{state}_raw"]), ":", lw=1.6, label=f"Eff {state}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Incident energy $E$ [eV]")
        ax.set_ylabel(r"$\sigma(E)$ [cm$^2$]")
        ax.set_title(f"Explicit ERDD / PD and effective remainder (vi={result.vi})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        pdf.savefig(fig)
        plt.close(fig)
