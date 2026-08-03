#!/usr/bin/env python3
"""
Post-process DCR HDF5 output from command line.

This script generates journal-style figures for:
1) Grouped populations vs x:
   background {H+, H-, H, H2, H2+} and flow totals {A(H), M(H2)}.
2) Detailed flow-state evolution vs x:
   per-state curves for A and M flow blocks.
3) Detailed local-state composition vs x:
   per-state curves for local atomic H and local molecular H2 background blocks.

Examples:
  python scripts/postprocess_dcr.py groups --input output/dcr_results.h5
  python scripts/postprocess_dcr.py flow-detail --input output/dcr_results.h5
  python scripts/postprocess_dcr.py background-detail --input output/dcr_results.h5
  python scripts/postprocess_dcr.py all --input output/dcr_results.h5
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: h5py (pip install h5py)") from exc

try:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib (pip install matplotlib)") from exc

try:
    from scipy.linalg import expm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: scipy (pip install scipy)") from exc


@dataclass
class DCRData:
    x_cm: np.ndarray
    labels: List[str]
    p_indices: np.ndarray
    a_indices: np.ndarray
    m_indices: np.ndarray
    background_full: np.ndarray
    flow_a: np.ndarray
    flow_m: np.ndarray
    type_id: Optional[np.ndarray] = None
    charge: Optional[np.ndarray] = None
    atomicity: Optional[np.ndarray] = None
    internal_id: Optional[np.ndarray] = None
    qss_relaxation_length_cm: Optional[np.ndarray] = None
    qss_first_excited_loss_frequency_s: Optional[np.ndarray] = None
    qss_first_excited_index: Optional[np.ndarray] = None
    qss_local_transient_atomic: Optional[np.ndarray] = None
    qss_flow_transient_atomic: Optional[np.ndarray] = None
    qss_local_transient_atomic_indices: Optional[np.ndarray] = None
    qss_flow_transient_atomic_indices: Optional[np.ndarray] = None
    electron_density_cm3: Optional[np.ndarray] = None
    atomic_scd_cm3_s: Optional[np.ndarray] = None
    atomic_effective_eir_rate_cm3_s: Optional[np.ndarray] = None
    atomic_mar_h_source_rate_cm3_s: Optional[np.ndarray] = None
    atomic_flow_h_source_rate_cm3_s: Optional[np.ndarray] = None
    molecular_flow_ionization_rate_cm3_s: Optional[np.ndarray] = None
    molecular_flow_charge_exchange_rate_cm3_s: Optional[np.ndarray] = None
    molecular_dissociation_rate_cm3_s: Optional[np.ndarray] = None
    full_ion_nuclei_source_cm3_s: Optional[np.ndarray] = None
    full_ion_nuclei_sink_cm3_s: Optional[np.ndarray] = None
    molecular_flow_lhs_transport_rate_cm3_s: Optional[np.ndarray] = None
    electron_temperature_eV: Optional[np.ndarray] = None
    ion_temperature_eV: Optional[np.ndarray] = None
    power_loss_atomic_ionization_W_cm3: Optional[np.ndarray] = None
    power_loss_molecular_ionization_W_cm3: Optional[np.ndarray] = None
    power_loss_molecular_dissociation_W_cm3: Optional[np.ndarray] = None
    power_loss_total_electron_inelastic_W_cm3: Optional[np.ndarray] = None
    grouped_source_H_plus_cm3_s: Optional[np.ndarray] = None
    grouped_source_H_cm3_s: Optional[np.ndarray] = None
    grouped_source_H2_cm3_s: Optional[np.ndarray] = None
    grouped_source_H_minus_cm3_s: Optional[np.ndarray] = None
    grouped_source_H2_plus_cm3_s: Optional[np.ndarray] = None
    flow_to_local_source_H_plus_cm3_s: Optional[np.ndarray] = None
    flow_to_local_source_H_cm3_s: Optional[np.ndarray] = None
    flow_to_local_source_H2_cm3_s: Optional[np.ndarray] = None
    flow_to_local_source_H_minus_cm3_s: Optional[np.ndarray] = None
    flow_to_local_source_H2_plus_cm3_s: Optional[np.ndarray] = None
    h2plus_indices: Optional[np.ndarray] = None
    h2plus_mcx_production_cm3_s: Optional[np.ndarray] = None
    h2plus_mi_production_cm3_s: Optional[np.ndarray] = None
    h2plus_dr_h_source_frequency_s: Optional[np.ndarray] = None
    h2plus_target_velocity_cm_s: Optional[np.ndarray] = None
    h2plus_generator_s: Optional[np.ndarray] = None
    mar_simple_branching_dr_h_source_cm3_s: Optional[np.ndarray] = None


def configure_journal_style(dpi: int, font_scale: float) -> None:
    """
    Apply a clean, journal-oriented plotting style:
    serif fonts, inward ticks, subtle grids, and consistent line widths.
    """
    fs_base = 10.0 * font_scale
    mpl.rcParams.update(
        {
            "savefig.dpi": dpi,
            "figure.dpi": dpi,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "axes.labelsize": fs_base + 1.0,
            "axes.titlesize": fs_base + 2.0,
            "xtick.labelsize": fs_base,
            "ytick.labelsize": fs_base,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "legend.fontsize": fs_base - 0.5,
            "lines.linewidth": 1.9,
            "lines.markersize": 4.5,
            "axes.prop_cycle": mpl.cycler(
                color=[
                    "#1f77b4",  # blue
                    "#d62728",  # red
                    "#2ca02c",  # green
                    "#ff7f0e",  # orange
                    "#17becf",  # cyan
                    "#4d4d4d",  # gray
                    "#8c564b",  # brown
                    "#7f7f7f",  # secondary gray
                ]
            ),
        }
    )


def decode_labels(raw: Sequence[object]) -> List[str]:
    labels: List[str] = []
    for x in raw:
        if isinstance(x, bytes):
            labels.append(x.decode("utf-8"))
        else:
            labels.append(str(x))
    return labels


def load_hdf5(path: Path) -> DCRData:
    with h5py.File(path, "r") as f:
        x_cm = np.asarray(f["/grid/x_cm"][:], dtype=float)
        labels = decode_labels(f["/states/labels"][:])
        p_indices = np.asarray(f["/states/P_indices"][:], dtype=int)
        a_indices = np.asarray(f["/states/A_indices"][:], dtype=int)
        m_indices = np.asarray(f["/states/M_indices"][:], dtype=int)
        background_full = np.asarray(f["/population/background_full"][:], dtype=float)
        flow_a = np.asarray(f["/population/flowA"][:], dtype=float)
        flow_m = np.asarray(f["/population/flowM"][:], dtype=float)

        type_id = np.asarray(f["/states/type_id"][:], dtype=int) if "/states/type_id" in f else None
        charge = np.asarray(f["/states/charge"][:], dtype=int) if "/states/charge" in f else None
        atomicity = np.asarray(f["/states/atomicity"][:], dtype=int) if "/states/atomicity" in f else None
        internal_id = np.asarray(f["/states/internal_id"][:], dtype=int) if "/states/internal_id" in f else None
        qss_relaxation_length_cm = (
            np.asarray(f["/rates/atomic_qss_relaxation_length_cm"][:], dtype=float)
            if "/rates/atomic_qss_relaxation_length_cm" in f else None
        )
        qss_first_excited_loss_frequency_s = (
            np.asarray(f["/rates/atomic_qss_first_excited_loss_frequency_s"][:], dtype=float)
            if "/rates/atomic_qss_first_excited_loss_frequency_s" in f else None
        )
        qss_first_excited_index = (
            np.asarray(f["/rates/atomic_qss_first_excited_index"][:], dtype=int)
            if "/rates/atomic_qss_first_excited_index" in f else None
        )
        qss_local_transient_atomic = (
            np.asarray(f["/population/qss_local_transient_atomic"][:], dtype=float)
            if "/population/qss_local_transient_atomic" in f else None
        )
        qss_flow_transient_atomic = (
            np.asarray(f["/population/qss_flow_transient_atomic"][:], dtype=float)
            if "/population/qss_flow_transient_atomic" in f else None
        )
        qss_local_transient_atomic_indices = (
            np.asarray(f["/states/qss_local_transient_atomic_indices"][:], dtype=int)
            if "/states/qss_local_transient_atomic_indices" in f else None
        )
        qss_flow_transient_atomic_indices = (
            np.asarray(f["/states/qss_flow_transient_atomic_indices"][:], dtype=int)
            if "/states/qss_flow_transient_atomic_indices" in f else None
        )
        electron_density_cm3 = (
            np.asarray(f["/rates/ne_cm3"][:], dtype=float)
            if "/rates/ne_cm3" in f else None
        )
        electron_temperature_eV = (
            np.asarray(f["/rates/Te_eV"][:], dtype=float)
            if "/rates/Te_eV" in f else None
        )
        ion_temperature_eV = (
            np.asarray(f["/rates/Ti_eV"][:], dtype=float)
            if "/rates/Ti_eV" in f else None
        )
        atomic_scd_cm3_s = (
            np.asarray(f["/rates/atomic_scd_cm3_s"][:], dtype=float)
            if "/rates/atomic_scd_cm3_s" in f else None
        )
        atomic_effective_eir_rate_cm3_s = (
            np.asarray(f["/rates/atomic_effective_eir_rate_cm3_s"][:], dtype=float)
            if "/rates/atomic_effective_eir_rate_cm3_s" in f else None
        )
        atomic_mar_h_source_rate_cm3_s = (
            np.asarray(f["/rates/atomic_mar_h_source_rate_cm3_s"][:], dtype=float)
            if "/rates/atomic_mar_h_source_rate_cm3_s" in f else None
        )
        atomic_flow_h_source_rate_cm3_s = (
            np.asarray(f["/rates/atomic_flow_h_source_rate_cm3_s"][:], dtype=float)
            if "/rates/atomic_flow_h_source_rate_cm3_s" in f else None
        )
        molecular_flow_ionization_rate_cm3_s = (
            np.asarray(f["/rates/molecular_flow_ionization_rate_cm3_s"][:], dtype=float)
            if "/rates/molecular_flow_ionization_rate_cm3_s" in f else None
        )
        molecular_flow_charge_exchange_rate_cm3_s = (
            np.asarray(f["/rates/molecular_flow_charge_exchange_rate_cm3_s"][:], dtype=float)
            if "/rates/molecular_flow_charge_exchange_rate_cm3_s" in f else None
        )
        molecular_dissociation_rate_cm3_s = (
            np.asarray(f["/rates/molecular_dissociation_rate_cm3_s"][:], dtype=float)
            if "/rates/molecular_dissociation_rate_cm3_s" in f else None
        )
        full_ion_nuclei_source_cm3_s = (
            np.asarray(f["/rates/full_ion_nuclei_source_cm3_s"][:], dtype=float)
            if "/rates/full_ion_nuclei_source_cm3_s" in f else None
        )
        full_ion_nuclei_sink_cm3_s = (
            np.asarray(f["/rates/full_ion_nuclei_sink_cm3_s"][:], dtype=float)
            if "/rates/full_ion_nuclei_sink_cm3_s" in f else None
        )
        molecular_flow_lhs_transport_rate_cm3_s = (
            np.asarray(f["/rates/molecular_flow_lhs_transport_rate_cm3_s"][:], dtype=float)
            if "/rates/molecular_flow_lhs_transport_rate_cm3_s" in f else None
        )
        power_loss_atomic_ionization_W_cm3 = (
            np.asarray(f["/power_loss/atomic_ionization_W_cm3"][:], dtype=float)
            if "/power_loss/atomic_ionization_W_cm3" in f else None
        )
        power_loss_molecular_ionization_W_cm3 = (
            np.asarray(f["/power_loss/molecular_ionization_W_cm3"][:], dtype=float)
            if "/power_loss/molecular_ionization_W_cm3" in f else None
        )
        power_loss_molecular_dissociation_W_cm3 = (
            np.asarray(f["/power_loss/molecular_dissociation_W_cm3"][:], dtype=float)
            if "/power_loss/molecular_dissociation_W_cm3" in f else None
        )
        power_loss_total_electron_inelastic_W_cm3 = (
            np.asarray(f["/power_loss/total_electron_inelastic_W_cm3"][:], dtype=float)
            if "/power_loss/total_electron_inelastic_W_cm3" in f else None
        )
        grouped_source_H_plus_cm3_s = (
            np.asarray(f["/rates/grouped_source_H_plus_cm3_s"][:], dtype=float)
            if "/rates/grouped_source_H_plus_cm3_s" in f else None
        )
        grouped_source_H_cm3_s = (
            np.asarray(f["/rates/grouped_source_H_cm3_s"][:], dtype=float)
            if "/rates/grouped_source_H_cm3_s" in f else None
        )
        grouped_source_H2_cm3_s = (
            np.asarray(f["/rates/grouped_source_H2_cm3_s"][:], dtype=float)
            if "/rates/grouped_source_H2_cm3_s" in f else None
        )
        grouped_source_H_minus_cm3_s = (
            np.asarray(f["/rates/grouped_source_H_minus_cm3_s"][:], dtype=float)
            if "/rates/grouped_source_H_minus_cm3_s" in f else None
        )
        grouped_source_H2_plus_cm3_s = (
            np.asarray(f["/rates/grouped_source_H2_plus_cm3_s"][:], dtype=float)
            if "/rates/grouped_source_H2_plus_cm3_s" in f else None
        )
        flow_to_local_source_H_plus_cm3_s = (
            np.asarray(f["/rates/flow_to_local_source_H_plus_cm3_s"][:], dtype=float)
            if "/rates/flow_to_local_source_H_plus_cm3_s" in f else None
        )
        flow_to_local_source_H_cm3_s = (
            np.asarray(f["/rates/flow_to_local_source_H_cm3_s"][:], dtype=float)
            if "/rates/flow_to_local_source_H_cm3_s" in f else None
        )
        flow_to_local_source_H2_cm3_s = (
            np.asarray(f["/rates/flow_to_local_source_H2_cm3_s"][:], dtype=float)
            if "/rates/flow_to_local_source_H2_cm3_s" in f else None
        )
        flow_to_local_source_H_minus_cm3_s = (
            np.asarray(f["/rates/flow_to_local_source_H_minus_cm3_s"][:], dtype=float)
            if "/rates/flow_to_local_source_H_minus_cm3_s" in f else None
        )
        flow_to_local_source_H2_plus_cm3_s = (
            np.asarray(f["/rates/flow_to_local_source_H2_plus_cm3_s"][:], dtype=float)
            if "/rates/flow_to_local_source_H2_plus_cm3_s" in f else None
        )
        h2plus_indices = (
            np.asarray(f["/states/H2_plus_indices"][:], dtype=int)
            if "/states/H2_plus_indices" in f else None
        )
        h2plus_mcx_production_cm3_s = (
            np.asarray(f["/rates/H2_plus_mcx_production_cm3_s"][:], dtype=float)
            if "/rates/H2_plus_mcx_production_cm3_s" in f else None
        )
        h2plus_mi_production_cm3_s = (
            np.asarray(f["/rates/H2_plus_mi_production_cm3_s"][:], dtype=float)
            if "/rates/H2_plus_mi_production_cm3_s" in f else None
        )
        h2plus_dr_h_source_frequency_s = (
            np.asarray(f["/rates/H2_plus_dr_h_source_frequency_s"][:], dtype=float)
            if "/rates/H2_plus_dr_h_source_frequency_s" in f else None
        )
        h2plus_target_velocity_cm_s = (
            np.asarray(f["/rates/H2_plus_target_velocity_cm_s"][:], dtype=float)
            if "/rates/H2_plus_target_velocity_cm_s" in f else None
        )
        h2plus_generator_s = (
            np.asarray(f["/rates/H2_plus_generator_s"][:], dtype=float)
            if "/rates/H2_plus_generator_s" in f else None
        )
        mar_simple_branching_dr_h_source_cm3_s = (
            np.asarray(f["/rates/mar_simple_branching_dr_h_source_cm3_s"][:], dtype=float)
            if "/rates/mar_simple_branching_dr_h_source_cm3_s" in f else None
        )

    return DCRData(
        x_cm=x_cm,
        labels=labels,
        p_indices=p_indices,
        a_indices=a_indices,
        m_indices=m_indices,
        background_full=background_full,
        flow_a=flow_a,
        flow_m=flow_m,
        type_id=type_id,
        charge=charge,
        atomicity=atomicity,
        internal_id=internal_id,
        qss_relaxation_length_cm=qss_relaxation_length_cm,
        qss_first_excited_loss_frequency_s=qss_first_excited_loss_frequency_s,
        qss_first_excited_index=qss_first_excited_index,
        qss_local_transient_atomic=qss_local_transient_atomic,
        qss_flow_transient_atomic=qss_flow_transient_atomic,
        qss_local_transient_atomic_indices=qss_local_transient_atomic_indices,
        qss_flow_transient_atomic_indices=qss_flow_transient_atomic_indices,
        electron_density_cm3=electron_density_cm3,
        atomic_scd_cm3_s=atomic_scd_cm3_s,
        atomic_effective_eir_rate_cm3_s=atomic_effective_eir_rate_cm3_s,
        atomic_mar_h_source_rate_cm3_s=atomic_mar_h_source_rate_cm3_s,
        atomic_flow_h_source_rate_cm3_s=atomic_flow_h_source_rate_cm3_s,
        molecular_flow_ionization_rate_cm3_s=molecular_flow_ionization_rate_cm3_s,
        molecular_flow_charge_exchange_rate_cm3_s=molecular_flow_charge_exchange_rate_cm3_s,
        molecular_dissociation_rate_cm3_s=molecular_dissociation_rate_cm3_s,
        full_ion_nuclei_source_cm3_s=full_ion_nuclei_source_cm3_s,
        full_ion_nuclei_sink_cm3_s=full_ion_nuclei_sink_cm3_s,
        molecular_flow_lhs_transport_rate_cm3_s=molecular_flow_lhs_transport_rate_cm3_s,
        electron_temperature_eV=electron_temperature_eV,
        ion_temperature_eV=ion_temperature_eV,
        power_loss_atomic_ionization_W_cm3=power_loss_atomic_ionization_W_cm3,
        power_loss_molecular_ionization_W_cm3=power_loss_molecular_ionization_W_cm3,
        power_loss_molecular_dissociation_W_cm3=power_loss_molecular_dissociation_W_cm3,
        power_loss_total_electron_inelastic_W_cm3=power_loss_total_electron_inelastic_W_cm3,
        grouped_source_H_plus_cm3_s=grouped_source_H_plus_cm3_s,
        grouped_source_H_cm3_s=grouped_source_H_cm3_s,
        grouped_source_H2_cm3_s=grouped_source_H2_cm3_s,
        grouped_source_H_minus_cm3_s=grouped_source_H_minus_cm3_s,
        grouped_source_H2_plus_cm3_s=grouped_source_H2_plus_cm3_s,
        flow_to_local_source_H_plus_cm3_s=flow_to_local_source_H_plus_cm3_s,
        flow_to_local_source_H_cm3_s=flow_to_local_source_H_cm3_s,
        flow_to_local_source_H2_cm3_s=flow_to_local_source_H2_cm3_s,
        flow_to_local_source_H_minus_cm3_s=flow_to_local_source_H_minus_cm3_s,
        flow_to_local_source_H2_plus_cm3_s=flow_to_local_source_H2_plus_cm3_s,
        h2plus_indices=h2plus_indices,
        h2plus_mcx_production_cm3_s=h2plus_mcx_production_cm3_s,
        h2plus_mi_production_cm3_s=h2plus_mi_production_cm3_s,
        h2plus_dr_h_source_frequency_s=h2plus_dr_h_source_frequency_s,
        h2plus_target_velocity_cm_s=h2plus_target_velocity_cm_s,
        h2plus_generator_s=h2plus_generator_s,
        mar_simple_branching_dr_h_source_cm3_s=mar_simple_branching_dr_h_source_cm3_s,
    )


def convert_x(x_cm: np.ndarray, x_unit: str) -> Tuple[np.ndarray, str]:
    if x_unit == "m":
        return 1.0e-2 * x_cm, "x (m)"
    return x_cm, "x (cm)"


def infer_u_floor_fraction_from_path(path: Path) -> Optional[float]:
    """Infer common velocity-floor tags such as ufloor001, ufloor003, ufloor01."""
    text = str(path)
    match = re.search(r"ufloor(\d+)", text)
    if match is None:
        return None
    token = match.group(1)
    if token.startswith("00"):
        return float(int(token)) / 100.0
    if token.startswith("0"):
        return float(int(token)) / 10.0
    return float(int(token))


def resolve_u_floor_fraction(args: argparse.Namespace) -> float:
    value = getattr(args, "u_floor_fraction", None)
    if value is not None:
        return float(value)
    inferred = infer_u_floor_fraction_from_path(args.input)
    if inferred is not None:
        return inferred
    raise SystemExit(
        "Could not infer u-floor from the input path. Use -u/--ufloor, e.g. -u 0.03."
    )


def safe_log_series(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[out <= 0.0] = np.nan
    return out


def positive_x_mask(x: np.ndarray) -> np.ndarray:
    # Log-x plotting cannot represent x=0 (boundary node), so skip non-positive locations.
    return x > 0.0


def apply_x_max_mask(x_cm: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    x_max_cm = getattr(args, "x_max_cm", None)
    if x_max_cm is None:
        return mask
    return mask & (x_cm <= float(x_max_cm))


def state_atomicity(data: DCRData, gi: int) -> float:
    if data.atomicity is not None and 0 <= gi < data.atomicity.size:
        return float(data.atomicity[gi])
    if 0 <= gi < len(data.labels):
        lab = data.labels[gi].lower()
        if "h2_" in lab or "h2p_" in lab:
            return 2.0
    return 1.0


def atomic_neutral_ground_index(data: DCRData) -> int:
    candidates: List[int] = []
    if data.type_id is not None and data.charge is not None and data.atomicity is not None:
        n = min(data.type_id.size, data.charge.size, data.atomicity.size)
        for gi in range(n):
            if data.type_id[gi] == 0 and data.charge[gi] == 0 and data.atomicity[gi] == 1:
                candidates.append(gi)
    else:
        for gi, label in enumerate(data.labels):
            low = label.lower()
            if "h_atom" in low and "barenucl" not in low:
                candidates.append(gi)

    if not candidates:
        raise SystemExit("Could not identify neutral atomic H ground state in HDF5 metadata.")
    if data.internal_id is not None:
        return min(candidates, key=lambda gi: int(data.internal_id[gi]))
    return candidates[0]


def total_nuclei_profile(data: DCRData) -> np.ndarray:
    """Total hydrogen nuclei density profile: background + A-flow + M-flow."""
    n_nodes = data.background_full.shape[0]
    total = np.zeros(n_nodes)

    for gi in range(data.background_full.shape[1]):
        total += np.maximum(data.background_full[:, gi], 0.0) * state_atomicity(data, gi)

    for j, gi in enumerate(data.a_indices):
        if j >= data.flow_a.shape[1]:
            break
        total += np.maximum(data.flow_a[:, j], 0.0) * state_atomicity(data, int(gi))

    for j, gi in enumerate(data.m_indices):
        if j >= data.flow_m.shape[1]:
            break
        total += np.maximum(data.flow_m[:, j], 0.0) * state_atomicity(data, int(gi))

    return total


def flow_block_density_profiles(data: DCRData) -> Dict[str, np.ndarray]:
    n_nodes = data.x_cm.size
    flow_a = np.zeros(n_nodes)
    flow_m = np.zeros(n_nodes)
    if data.flow_a.size > 0:
        flow_a = np.maximum(data.flow_a, 0.0).sum(axis=1)
    if data.flow_m.size > 0:
        flow_m = np.maximum(data.flow_m, 0.0).sum(axis=1)
    return {"A(H)": flow_a, "M(H2)": flow_m}


def grouped_local_particle_densities(data: DCRData) -> Dict[str, np.ndarray]:
    curves = {
        "H+": np.zeros(data.x_cm.size),
        "H-": np.zeros(data.x_cm.size),
        "H": np.zeros(data.x_cm.size),
        "H2": np.zeros(data.x_cm.size),
        "H2+": np.zeros(data.x_cm.size),
    }
    for gi_raw in data.p_indices:
        gi = int(gi_raw)
        if gi < 0 or gi >= data.background_full.shape[1]:
            continue
        group = classify_state(data, gi)
        if group not in curves:
            continue
        curves[group] += np.maximum(data.background_full[:, gi], 0.0)
    return curves


def classify_state(data: DCRData, gi: int) -> Optional[str]:
    """
    Group background states into: H+, H-, H, H2, H2+.
    Uses explicit metadata when available, with a label fallback.
    """
    if data.type_id is not None and data.charge is not None and data.atomicity is not None:
        if gi < 0 or gi >= data.type_id.size:
            return None
        t = int(data.type_id[gi])   # 0=Atom, 1=Molecule, 2=Ion
        z = int(data.charge[gi])
        nu = int(data.atomicity[gi])
        if t == 2:  # ion
            if z > 0:
                return "H2+" if nu >= 2 else "H+"
            if z < 0:
                return "H-"
            return None
        if t == 0 and z == 0:
            return "H"
        if t == 1:
            if z > 0:
                return "H2+"
            if z == 0:
                return "H2"
        return None

    if gi < 0 or gi >= len(data.labels):
        return None
    lab = data.labels[gi].lower()
    if "barenucl" in lab:
        return "H+"
    if "h_200001" in lab:
        return "H-"
    if "h2p_" in lab:
        return "H2+"
    if "h2_" in lab:
        return "H2"
    if "_atom " in lab:
        return "H"
    return None


def grouped_curves_nuclei_density(data: DCRData) -> Dict[str, np.ndarray]:
    n_nodes = data.background_full.shape[0]
    curves = {
        "H+": np.zeros(n_nodes),
        "H-": np.zeros(n_nodes),
        "H": np.zeros(n_nodes),
        "H2": np.zeros(n_nodes),
        "H2+": np.zeros(n_nodes),
        "A(H)": np.zeros(n_nodes),
        "M(H2)": np.zeros(n_nodes),
    }

    # Background subgroups (only over P block).
    for gi in data.p_indices:
        if gi < 0 or gi >= data.background_full.shape[1]:
            continue
        grp = classify_state(data, int(gi))
        if grp is None:
            continue
        y = np.maximum(data.background_full[:, int(gi)], 0.0) * state_atomicity(data, int(gi))
        curves[grp] += y

    # Flow totals.
    if data.flow_a.size > 0:
        flow_a = np.maximum(data.flow_a, 0.0).copy()
        for j, gi in enumerate(data.a_indices):
            if j >= flow_a.shape[1]:
                break
            flow_a[:, j] *= state_atomicity(data, int(gi))
        curves["A(H)"] = flow_a.sum(axis=1)

    if data.flow_m.size > 0:
        flow_m = np.maximum(data.flow_m, 0.0).copy()
        for j, gi in enumerate(data.m_indices):
            if j >= flow_m.shape[1]:
                break
            flow_m[:, j] *= state_atomicity(data, int(gi))
        curves["M(H2)"] = flow_m.sum(axis=1)

    return curves


def grouped_curves_fraction(data: DCRData) -> Dict[str, np.ndarray]:
    curves = grouped_curves_nuclei_density(data)
    total = total_nuclei_profile(data)
    denom = np.where(total > 0.0, total, np.nan)
    return {key: values / denom for key, values in curves.items()}


def flow_state_display_label(data: DCRData, gi: int, label: str, block_kind: str) -> str:
    """
    Render compact scientific legend labels for atomic and molecular state blocks.
    A: H(n=...)
    M: H2(v=...)
    """
    token = label.split()[-1] if label.split() else label
    t = token.lower()

    if block_kind == "A":
        if data.internal_id is not None and 0 <= gi < data.internal_id.size:
            return rf"$\mathrm{{H}}(n={int(data.internal_id[gi])})$"
        m = re.search(r"h_(\d+)$", t)
        if m:
            digits = m.group(1)
            n = int(digits[-3:]) if len(digits) >= 3 else int(digits)
            if n > 50 and len(digits) >= 2:
                n = int(digits[-2:])
            return rf"$\mathrm{{H}}(n={n})$"
        return r"$\mathrm{H}$"

    # M flow: vibrational molecular states.
    if data.internal_id is not None and 0 <= gi < data.internal_id.size:
        return rf"$\mathrm{{H}}_2(v={max(0, int(data.internal_id[gi]) - 1)})$"
    m = re.search(r"h2_v(\d+)$", t)
    if m:
        return rf"$\mathrm{{H}}_2(v={max(0, int(m.group(1)) - 1)})$"
    return r"$\mathrm{H}_2$"


def molecular_display_v(data: DCRData, gi: int, label: str) -> Optional[int]:
    if data.internal_id is not None and 0 <= gi < data.internal_id.size:
        return max(0, int(data.internal_id[gi]) - 1)
    token = label.split()[-1] if label.split() else label
    m = re.search(r"h2_v(\d+)$", token.lower())
    if m:
        return max(0, int(m.group(1)) - 1)
    return None


def save_group_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")
    x_plot = x[mask]
    total_nuclei = total_nuclei_profile(data)
    n_nuc0 = float(total_nuclei[0])
    if not np.isfinite(n_nuc0) or n_nuc0 <= 0.0:
        raise SystemExit("Initial total nuclei density must be positive for normalization.")
    curves = {
        key: values / n_nuc0
        for key, values in grouped_curves_nuclei_density(data).items()
    }
    curves["n_nuc"] = total_nuclei / n_nuc0

    colors = {
        "H+": "#1f77b4",
        "H-": "#17becf",
        "H": "#2ca02c",
        "H2": "#ff7f0e",
        "H2+": "#d62728",
        "A(H)": "#4d4d4d",
        "M(H2)": "#8c564b",
        "n_nuc": "#000000",
    }
    styles = {
        "H+": "-",
        "H-": "-",
        "H": "-",
        "H2": "-",
        "H2+": "-",
        "A(H)": "--",
        "M(H2)": "--",
        "n_nuc": "-",
    }
    order = ["H+", "H-", "H", "H2", "H2+", "n_nuc", "A(H)", "M(H2)"]
    legend = {
        "H+": r"$\mathrm{H}^{+}$",
        "H-": r"$\mathrm{H}^{-}$",
        "H": r"$\mathrm{H}$",
        "H2": r"$\mathrm{H}_2$",
        "H2+": r"$\mathrm{H}_2^{+}$",
        "A(H)": r"recycled flow $A(\mathrm{H})$",
        "M(H2)": r"recycled flow $M(\mathrm{H}_2)$",
        "n_nuc": r"$n_{\rm nuc}(x)$",
    }

    standalone_csv = getattr(args, "standalone_cr_csv", None)
    standalone = None
    if standalone_csv is not None:
        if not standalone_csv.exists():
            raise SystemExit(f"Standalone CR CSV not found: {standalone_csv}")
        standalone = np.genfromtxt(standalone_csv, delimiter=",", names=True)

    fig, (ax_hi, ax_lo) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    flow_zone_end, _ = convert_x(np.asarray([15.0]), args.x_unit)
    shade_end = min(float(flow_zone_end[0]), float(np.max(x_plot)))
    for ax in (ax_hi, ax_lo):
        if shade_end > float(np.min(x_plot)):
            ax.axvspan(
                float(np.min(x_plot)),
                shade_end,
                color="#d9d9d9",
                alpha=0.35,
                linewidth=0.0,
                zorder=0,
            )
        for key in order:
            if ax is ax_lo and key == "H2+":
                continue
            y = curves[key][mask]
            y_plot = safe_log_series(y)
            linewidth = 1.15 if key == "n_nuc" else None
            ax.plot(
                x_plot,
                y_plot,
                linestyle=styles[key],
                color=colors[key],
                linewidth=linewidth,
                label=legend[key],
            )
        if standalone is not None:
            x_standalone = np.asarray(standalone["x_cm"], dtype=float)
            x_standalone, _ = convert_x(x_standalone, args.x_unit)
            valid = np.isfinite(x_standalone)
            if args.x_max_cm is not None:
                xmax_plot, _ = convert_x(np.asarray([args.x_max_cm]), args.x_unit)
                valid &= x_standalone <= float(xmax_plot[0])
            overlays = [
                ("standalone_Hplus", 1.0, "H+", r"standalone CR $\mathrm{H}^{+}$"),
                ("standalone_H", 1.0, "H", r"standalone CR $\mathrm{H}$"),
            ]
            if not getattr(args, "hide_standalone_h2", False):
                overlays.append(
                    ("standalone_H2", 2.0, "H2", r"standalone CR $\mathrm{H}_2$")
                )
            for col, atomicity, key, label in overlays:
                y = np.asarray(standalone[col], dtype=float) * atomicity / n_nuc0
                ax.plot(
                    x_standalone[valid],
                    safe_log_series(y[valid]),
                    linestyle=":",
                    marker="o",
                    color=colors[key],
                    linewidth=2.2,
                    label=label,
                )
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

    # Adapt the broken-axis limits to retain both the local H branch and the
    # trace H2 branch across temperatures while preserving the same layout.
    h_positive = curves["H"][mask]
    h_positive = h_positive[np.isfinite(h_positive) & (h_positive > 0.0)]
    trace_positive = np.concatenate((curves["H-"][mask], curves["H2"][mask]))
    trace_positive = trace_positive[np.isfinite(trace_positive) & (trace_positive > 0.0)]
    if standalone is not None:
        standalone_h = np.asarray(standalone["standalone_H"], dtype=float) / n_nuc0
        h_positive = np.concatenate((h_positive, standalone_h[standalone_h > 0.0]))
        if not getattr(args, "hide_standalone_h2", False):
            standalone_h2 = 2.0 * np.asarray(standalone["standalone_H2"], dtype=float) / n_nuc0
            trace_positive = np.concatenate((trace_positive, standalone_h2[standalone_h2 > 0.0]))
    hi_bottom = min(5.0e-4, float(np.min(h_positive)) / 3.0) if h_positive.size else 5.0e-4
    lo_bottom = min(1.0e-12, float(np.min(trace_positive)) / 3.0) if trace_positive.size else 1.0e-12
    if getattr(args, "group_ymin", None) is not None:
        lo_bottom = float(args.group_ymin)
    ax_hi.set_ylim(max(hi_bottom, 1.0e-18), 8.0)
    ax_lo.set_ylim(max(lo_bottom, 1.0e-22), 3.0e-8)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(labelbottom=False)
    ax_lo.set_xlabel(xlabel)
    fig.supylabel(r"$\mu_s n_s / n_{\rm nuc,0}$")
    legend_x = 0.68 if hi_bottom < 1.0e-5 else 0.99
    legend_y = 0.30 if hi_bottom < 1.0e-5 else 0.57
    ax_hi.legend(
        ncol=2,
        loc="center right",
        bbox_to_anchor=(legend_x, legend_y),
        columnspacing=1.0,
        handlelength=2.1,
        handletextpad=0.6,
        labelspacing=0.3,
        fontsize=10.5,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )

    d = 0.012
    kwargs = dict(transform=ax_hi.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_hi.plot((-d, +d), (-d, +d), **kwargs)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_lo.transAxes)
    ax_lo.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_lo.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    out = outdir / "bg_A_M_groups_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_proposal_summary_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")
    x_plot = x[mask]
    curves = grouped_curves_fraction(data)

    series = [
        ("H+", r"local $\mathrm{H}^{+}$", "#1f77b4", "-"),
        ("H", r"local $\mathrm{H}$", "#2ca02c", "-"),
        ("H2+", r"local $\mathrm{H}_2^{+}$", "#d62728", "-"),
        ("A(H)", r"recycled flow $A(\mathrm{H})$", "#4d4d4d", "--"),
        ("M(H2)", r"recycled flow $M(\mathrm{H}_2)$", "#8c564b", "--"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for key, label, color, style in series:
        y_plot = safe_log_series(curves[key][mask])
        lw = 2.3 if style == "--" else 2.0
        ax.plot(x_plot, y_plot, linestyle=style, color=color, linewidth=lw, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of total nuclei")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.32)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.18)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_ylim(bottom=1.0e-3)

    out = outdir / "dcr_transport_recycling_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_atom_source_rates_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    required = [
        data.atomic_effective_eir_rate_cm3_s,
        data.atomic_mar_h_source_rate_cm3_s,
        data.atomic_flow_h_source_rate_cm3_s,
    ]
    if any(v is None for v in required):
        raise SystemExit(
            "Missing atomic source-rate diagnostics in HDF5. Re-run DCR_Main after rebuilding."
        )

    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    series = [
        (
            data.atomic_effective_eir_rate_cm3_s,
            r"effective EIR, $n_e\alpha_{\mathrm{eff}}n_{\mathrm{H}^{+}}$",
            "#1f77b4",
            "-",
        ),
        (
            data.atomic_mar_h_source_rate_cm3_s,
            r"MAR source into $\mathrm{H}$",
            "#d62728",
            "-",
        ),
        (
            data.atomic_flow_h_source_rate_cm3_s,
            r"recycling-flow source into $\mathrm{H}$",
            "#4d4d4d",
            "--",
        ),
    ]
    for values, label, color, linestyle in series:
        ax.plot(
            x[mask],
            safe_log_series(np.asarray(values, dtype=float)[mask]),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Volumetric rate ($\mathrm{cm^{-3}\,s^{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(loc="best")

    out = outdir / "atomic_source_rates_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_ionization_vs_h_pump_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    required = [
        data.electron_density_cm3,
        data.atomic_scd_cm3_s,
        data.atomic_flow_h_source_rate_cm3_s,
    ]
    if any(v is None for v in required):
        raise SystemExit(
            "Missing effective ionization coefficient, electron density, or H neutral pump diagnostics in HDF5."
        )

    h_ground_index = atomic_neutral_ground_index(data)
    h_ground_density = np.maximum(data.background_full[:, h_ground_index], 0.0)
    effective_ionization_rate = (
        np.asarray(data.electron_density_cm3, dtype=float)
        * np.asarray(data.atomic_scd_cm3_s, dtype=float)
        * h_ground_density
    )

    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    series = [
        (
            effective_ionization_rate,
            r"effective ionization, $n_e S_{\mathrm{eff}} n_{\mathrm{H}}$",
            "#1f77b4",
            "-",
        ),
        (
            data.atomic_flow_h_source_rate_cm3_s,
            r"neutral H pump/recycling source",
            "#4d4d4d",
            "--",
        ),
    ]
    for values, label, color, linestyle in series:
        ax.plot(
            x[mask],
            safe_log_series(np.asarray(values, dtype=float)[mask]),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Volumetric rate ($\mathrm{cm^{-3}\,s^{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(ncol=1, loc="best")

    out = outdir / "effective_ionization_vs_h_neutral_pump.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_molecular_flow_rates_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    required = [
        data.molecular_flow_ionization_rate_cm3_s,
        data.molecular_flow_charge_exchange_rate_cm3_s,
    ]
    if any(v is None for v in required):
        raise SystemExit(
            "Missing molecular-flow rate diagnostics in HDF5. Re-run DCR_Main after rebuilding."
        )

    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    series = [
        (
            data.molecular_flow_ionization_rate_cm3_s,
            r"molecular ionization from $M(\mathrm{H}_2)$",
            "#d62728",
            "-",
        ),
        (
            data.molecular_flow_charge_exchange_rate_cm3_s,
            r"molecular charge exchange from $M(\mathrm{H}_2)$",
            "#1f77b4",
            "--",
        ),
    ]
    if data.molecular_flow_lhs_transport_rate_cm3_s is not None:
        series.append(
            (
                data.molecular_flow_lhs_transport_rate_cm3_s,
                r"$\nabla\cdot\Gamma^{\mathrm{H}_2^+}$",
                "#4d4d4d",
                ":",
            )
        )
    for values, label, color, linestyle in series:
        ax.plot(
            x[mask],
            safe_log_series(np.asarray(values, dtype=float)[mask]),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Volumetric rate ($\mathrm{cm^{-3}\,s^{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(loc="best")

    out = outdir / "molecular_flow_ion_sources_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_molecular_dissociation_vs_ionization_plot(
    data: DCRData, args: argparse.Namespace, outdir: Path
) -> Path:
    required = [
        data.molecular_dissociation_rate_cm3_s,
        data.molecular_flow_ionization_rate_cm3_s,
    ]
    if any(v is None for v in required):
        raise SystemExit(
            "Missing molecular dissociation or ionization event-rate diagnostics in HDF5."
        )

    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    series = [
        (
            data.molecular_dissociation_rate_cm3_s,
            r"molecular dissociation",
            "#2ca02c",
        ),
        (
            data.molecular_flow_ionization_rate_cm3_s,
            r"molecular ionization",
            "#d62728",
        ),
    ]
    for values, label, color in series:
        ax.plot(
            x[mask],
            safe_log_series(np.asarray(values, dtype=float)[mask]),
            label=label,
            color=color,
            linewidth=2.1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Volumetric event rate ($\mathrm{cm^{-3}\,s^{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(loc="best")

    out = outdir / "molecular_dissociation_vs_ionization_rate.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def compute_nonlocal_state_resolved_mar(data: DCRData) -> Tuple[np.ndarray, np.ndarray]:
    required = [
        data.h2plus_mcx_production_cm3_s,
        data.h2plus_dr_h_source_frequency_s,
        data.h2plus_target_velocity_cm_s,
        data.h2plus_generator_s,
    ]
    if any(value is None for value in required):
        raise SystemExit("Missing state-resolved H2+ transport diagnostics in HDF5.")

    x = np.asarray(data.x_cm, dtype=float)
    production = np.asarray(data.h2plus_mcx_production_cm3_s, dtype=float)
    dr_h_frequency = np.asarray(data.h2plus_dr_h_source_frequency_s, dtype=float)
    velocity = np.asarray(data.h2plus_target_velocity_cm_s, dtype=float)
    generator = np.asarray(data.h2plus_generator_s, dtype=float)
    if production.ndim != 2:
        raise SystemExit("Expected state-resolved MCX production with shape [x, state].")
    n_nodes, n_states = production.shape
    expected = {
        "DR H-source frequency": (n_nodes, n_states),
        "H2+ velocity": (n_nodes, n_states),
        "H2+ generator": (n_nodes, n_states, n_states),
    }
    actual = {
        "DR H-source frequency": dr_h_frequency.shape,
        "H2+ velocity": velocity.shape,
        "H2+ generator": generator.shape,
    }
    for label, shape in expected.items():
        if actual[label] != shape:
            raise SystemExit(f"Unexpected {label} shape: {actual[label]}, expected {shape}.")
    if x.size != n_nodes or np.any(np.diff(x) <= 0.0):
        raise SystemExit("State-resolved MAR transport requires a strictly increasing x grid.")
    if np.any(~np.isfinite(production)) or np.any(~np.isfinite(generator)):
        raise SystemExit("Non-finite state-resolved MAR transport coefficients.")
    if np.any(~np.isfinite(velocity)) or np.any(velocity <= 0.0):
        raise SystemExit("State-resolved MAR transport requires positive finite H2+ velocities.")

    tagged_population = np.zeros((n_nodes, n_states), dtype=float)
    tagged_flux = np.zeros(n_states, dtype=float)

    # No MCX-tagged molecular-ion flux enters at the upstream boundary x=L.
    for lower in range(n_nodes - 2, -1, -1):
        upper = lower + 1
        dx = float(x[upper] - x[lower])
        p_mid = 0.5 * (production[lower] + production[upper])
        q_mid = 0.5 * (generator[lower] + generator[upper])
        u_mid = 0.5 * (velocity[lower] + velocity[upper])
        augmented = np.zeros((n_states + 1, n_states + 1), dtype=float)
        augmented[:n_states, :n_states] = q_mid / u_mid[np.newaxis, :]
        augmented[:n_states, n_states] = p_mid
        propagated = expm(dx * augmented) @ np.append(tagged_flux, 1.0)
        tagged_flux = propagated[:n_states]
        scale = max(float(np.max(np.abs(tagged_flux))), 1.0)
        tagged_flux[np.abs(tagged_flux) < 1.0e-13 * scale] = 0.0
        if np.min(tagged_flux) < -1.0e-10 * scale:
            raise SystemExit(
                f"State-resolved MAR propagation produced a negative flux at x={x[lower]:.6g} cm."
            )
        tagged_flux = np.maximum(tagged_flux, 0.0)
        tagged_population[lower] = tagged_flux / velocity[lower]

    mar_h_source = np.sum(dr_h_frequency * tagged_population, axis=1)
    return mar_h_source, tagged_population


def compute_local_dr_weighted_mar(data: DCRData) -> Tuple[np.ndarray, np.ndarray]:
    required = [
        data.h2plus_indices,
        data.h2plus_mcx_production_cm3_s,
        data.h2plus_mi_production_cm3_s,
        data.h2plus_dr_h_source_frequency_s,
        data.h2plus_generator_s,
    ]
    if any(value is None for value in required):
        raise SystemExit("Missing state-resolved diagnostics for local DR-weighted branching.")

    indices = np.asarray(data.h2plus_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= data.background_full.shape[1]):
        raise SystemExit("Invalid H2+ state indices for local DR-weighted branching.")
    mcx = np.asarray(data.h2plus_mcx_production_cm3_s, dtype=float)
    mi = np.asarray(data.h2plus_mi_production_cm3_s, dtype=float)
    dr_h_frequency = np.asarray(data.h2plus_dr_h_source_frequency_s, dtype=float)
    generator = np.asarray(data.h2plus_generator_s, dtype=float)
    expected_2d = (data.x_cm.size, indices.size)
    if mcx.shape != expected_2d or mi.shape != expected_2d or dr_h_frequency.shape != expected_2d:
        raise SystemExit("Unexpected state-resolved production or DR-frequency shape.")
    if generator.shape != (data.x_cm.size, indices.size, indices.size):
        raise SystemExit("Unexpected H2+ generator shape for local DR-weighted branching.")

    # Stored DR frequency produces two H atoms, so ne*K_DR is half this value.
    dr_event_frequency = 0.5 * np.maximum(dr_h_frequency, 0.0)
    loss_frequency = np.maximum(-np.diagonal(generator, axis1=1, axis2=2), 0.0)
    weights = np.divide(
        dr_event_frequency,
        loss_frequency,
        out=np.zeros_like(dr_event_frequency),
        where=loss_frequency > 0.0,
    )
    numerator = np.sum(weights * np.maximum(mcx, 0.0), axis=1)
    denominator = np.sum(weights * (np.maximum(mcx, 0.0) + np.maximum(mi, 0.0)), axis=1)
    weighted_fraction = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )

    solved_h2plus = np.maximum(data.background_full[:, indices], 0.0)
    total_dr_h_source = np.sum(dr_h_frequency * solved_h2plus, axis=1)
    return weighted_fraction * total_dr_h_source, weighted_fraction


def save_mar_transport_comparison_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    if data.mar_simple_branching_dr_h_source_cm3_s is None:
        raise SystemExit("Missing simple-branching DR MAR diagnostic in HDF5.")
    nonlocal_mar, tagged_population = compute_nonlocal_state_resolved_mar(data)
    weighted_local_mar, weighted_fraction = compute_local_dr_weighted_mar(data)
    simple_mar = np.asarray(data.mar_simple_branching_dr_h_source_cm3_s, dtype=float)
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    if data.h2plus_indices is not None:
        indices = np.asarray(data.h2plus_indices, dtype=int)
        valid = indices[(indices >= 0) & (indices < data.background_full.shape[1])]
        if valid.size == tagged_population.shape[1]:
            solved_h2plus = np.maximum(data.background_full[:, valid], 0.0)
            solved_total = np.sum(solved_h2plus, axis=1)
            tagged_total = np.sum(tagged_population, axis=1)
            total_ratio = np.divide(
                tagged_total,
                solved_total,
                out=np.zeros_like(tagged_total),
                where=solved_total > 0.0,
            )
            max_total_ratio = float(np.max(total_ratio))
            print(
                "[postprocess] Max total MCX-tagged/solved H2+ population ratio: "
                f"{max_total_ratio:.6g}"
            )
            significant = (
                np.maximum(tagged_population, solved_h2plus)
                > 1.0e-5 * solved_total[:, np.newaxis]
            )
            state_ratio = np.divide(
                tagged_population,
                solved_h2plus,
                out=np.zeros_like(tagged_population),
                where=(solved_h2plus > 0.0) & significant,
            )
            max_state_ratio = float(np.max(state_ratio))
            if max_total_ratio > 1.05 or max_state_ratio > 1.05:
                print(
                    "[postprocess] Warning: materially populated passive MCX-tagged H2+ "
                    "exceeds the solved H2+ population "
                    f"(total={max_total_ratio:.3g}, state={max_state_ratio:.3g})."
                )

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    ax.plot(
        x[mask],
        safe_log_series(nonlocal_mar[mask]),
        color="#1f77b4",
        linewidth=2.2,
        label=r"nonlocal state-resolved $\mathrm{MCX}\!\rightarrow\!\mathrm{DR}$",
    )
    ax.plot(
        x[mask],
        safe_log_series(weighted_local_mar[mask]),
        color="#2ca02c",
        linestyle="-.",
        linewidth=2.1,
        label=r"local DR-weighted branching",
    )
    ax.plot(
        x[mask],
        safe_log_series(simple_mar[mask]),
        color="#d62728",
        linestyle="--",
        linewidth=2.1,
        label=r"local simple branching",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"MAR H-source rate ($\mathrm{cm^{-3}\,s^{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(loc="best")

    finite_fraction = weighted_fraction[np.isfinite(weighted_fraction)]
    if finite_fraction.size:
        print(
            "[postprocess] Local DR-weighted MCX fraction range: "
            f"{np.min(finite_fraction):.6g} to {np.max(finite_fraction):.6g}"
        )

    out = outdir / "mar_nonlocal_state_resolved_vs_simple_branching.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if y.size > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def save_recombination_degree_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    if (
        data.full_ion_nuclei_source_cm3_s is None
        or data.full_ion_nuclei_sink_cm3_s is None
    ):
        raise SystemExit("Missing full ion nuclei source or sink diagnostics in HDF5.")

    x_cm = np.asarray(data.x_cm, dtype=float)
    full_source = np.maximum(
        np.asarray(data.full_ion_nuclei_source_cm3_s, dtype=float), 0.0)
    full_sink = np.maximum(
        np.asarray(data.full_ion_nuclei_sink_cm3_s, dtype=float), 0.0)
    full_source_integral = cumulative_trapezoid(full_source, x_cm)
    full_sink_integral = cumulative_trapezoid(full_sink, x_cm)
    full_degree = np.divide(
        full_sink_integral,
        full_source_integral,
        out=np.full_like(full_sink_integral, np.nan),
        where=full_source_integral > 0.0,
    )

    x, xlabel = convert_x(x_cm, args.x_unit)
    mask = apply_x_max_mask(x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for recombination-degree plotting.")

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    label = r"all-channel ion nuclei balance"
    ax.plot(
        x[mask],
        safe_log_series(full_degree[mask]),
        color="#000000",
        linewidth=2.3,
        label=label,
    )
    finite = np.flatnonzero(mask & np.isfinite(full_degree) & (full_degree > 0.0))
    if finite.size:
        endpoint = finite[-1]
        print(
            f"[postprocess] D_rec({x_cm[endpoint]:.6g} cm), {label}: "
            f"{full_degree[endpoint]:.6e}"
        )

    ax.axhline(1.0, color="#4d4d4d", linestyle=":", linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Cumulative recombination degree, $\mathcal{D}_{\rm rec}(x)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(loc="best")

    out = outdir / "cumulative_recombination_degree_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_power_loss_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    required = [
        data.power_loss_atomic_ionization_W_cm3,
        data.power_loss_molecular_ionization_W_cm3,
        data.power_loss_molecular_dissociation_W_cm3,
        data.power_loss_total_electron_inelastic_W_cm3,
    ]
    if any(v is None for v in required):
        raise SystemExit(
            "Missing /power_loss diagnostics in HDF5. Re-run DCR_Main after rebuilding."
        )

    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    series = [
        (
            data.power_loss_atomic_ionization_W_cm3,
            r"atomic ionization",
            "#1f77b4",
            "-",
        ),
        (
            data.power_loss_molecular_ionization_W_cm3,
            r"molecular ionization",
            "#d62728",
            "-",
        ),
        (
            data.power_loss_molecular_dissociation_W_cm3,
            r"molecular dissociation",
            "#2ca02c",
            "-",
        ),
        (
            data.power_loss_total_electron_inelastic_W_cm3,
            r"total electron inelastic",
            "#4d4d4d",
            "--",
        ),
    ]
    for values, label, color, linestyle in series:
        ax.plot(
            x[mask],
            safe_log_series(np.asarray(values, dtype=float)[mask]),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Power loss ($\mathrm{W\,cm^{-3}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(ncol=1, loc="best")

    out = outdir / "electron_inelastic_power_loss_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_normalized_flow_density_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    profiles = flow_block_density_profiles(data)
    denom_a = profiles["A(H)"][0] if profiles["A(H)"].size > 0 else 0.0
    denom_m = profiles["M(H2)"][0] if profiles["M(H2)"].size > 0 else 0.0
    if denom_a <= 0.0 and denom_m <= 0.0:
        raise SystemExit("No nonzero flow densities available for normalized-flow plot.")

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    if denom_a > 0.0 and data.flow_a.size > 0:
        ax = axes[0]
        for j, gi_raw in enumerate(data.a_indices):
            if j >= data.flow_a.shape[1]:
                break
            gi = int(gi_raw)
            label = data.labels[gi] if 0 <= gi < len(data.labels) else f"state_{gi}"
            y = np.maximum(data.flow_a[:, j], 0.0) / denom_a
            ax.plot(x[mask], safe_log_series(y[mask]), color=cmap(j % 20), label=flow_state_display_label(data, gi, label, "A"))

    if denom_m > 0.0 and data.flow_m.size > 0:
        ax = axes[1]
        for j, gi_raw in enumerate(data.m_indices):
            if j >= data.flow_m.shape[1]:
                break
            gi = int(gi_raw)
            label = data.labels[gi] if 0 <= gi < len(data.labels) else f"state_{gi}"
            y = np.maximum(data.flow_m[:, j], 0.0) / denom_m
            ax.plot(x[mask], safe_log_series(y[mask]), color=cmap(j % 20), label=flow_state_display_label(data, gi, label, "M"))

    for ax, title in zip(axes, ["Atomic Flow", "Molecular Flow"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_title(title)

    for ax in axes:
        ax.set_ylabel("Normalized population")
    axes[1].set_xlabel(xlabel)

    out = outdir / "flow_density_normalized_to_initial_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_source_to_local_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    required = [
        data.flow_to_local_source_H_plus_cm3_s,
        data.flow_to_local_source_H_cm3_s,
        data.flow_to_local_source_H2_cm3_s,
        data.flow_to_local_source_H_minus_cm3_s,
        data.flow_to_local_source_H2_plus_cm3_s,
    ]
    if any(v is None for v in required):
        raise SystemExit("Missing flow-to-local source diagnostics in HDF5. Re-run DCR_Main after rebuilding.")

    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    series = [
        (data.flow_to_local_source_H_plus_cm3_s, r"$\mathrm{H}^{+}$", "#1f77b4"),
        (data.flow_to_local_source_H_cm3_s, r"$\mathrm{H}$", "#2ca02c"),
        (data.flow_to_local_source_H2_cm3_s, r"$\mathrm{H}_2$", "#ff7f0e"),
        (data.flow_to_local_source_H_minus_cm3_s, r"$\mathrm{H}^{-}$", "#17becf"),
        (data.flow_to_local_source_H2_plus_cm3_s, r"$\mathrm{H}_2^{+}$", "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for values, label, color in series:
        ax.plot(
            x[mask],
            safe_log_series(np.asarray(values, dtype=float)[mask]),
            label=label,
            color=color,
            linewidth=2.0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Flow-to-local source ($\mathrm{cm^{-3}\,s^{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))

    out = outdir / "flow_to_local_source_rate_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def grouped_local_nuclei_densities(data: DCRData) -> Dict[str, np.ndarray]:
    curves = {
        "H+": np.zeros(data.x_cm.size),
        "H-": np.zeros(data.x_cm.size),
        "H": np.zeros(data.x_cm.size),
        "H2": np.zeros(data.x_cm.size),
        "H2+": np.zeros(data.x_cm.size),
    }
    for gi_raw in data.p_indices:
        gi = int(gi_raw)
        if gi < 0 or gi >= data.background_full.shape[1]:
            continue
        group = classify_state(data, gi)
        if group not in curves:
            continue
        curves[group] += np.maximum(data.background_full[:, gi], 0.0) * state_atomicity(data, gi)
    return curves


def compute_ion_nuclei_fluxes(
    data: DCRData,
    Ls_cm: float,
    u_floor_fraction: float,
    density_cutoff_cm3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if data.type_id is None or data.charge is None or data.atomicity is None:
        raise SystemExit("Missing state metadata needed to identify ion groups.")

    if data.electron_temperature_eV is None or data.ion_temperature_eV is None:
        raise SystemExit("Missing /rates/Te_eV or /rates/Ti_eV needed to compute Bohm speeds.")

    eV_to_erg = 1.602176634e-12
    amu_g = 1.6726219e-24
    x = np.asarray(data.x_cm, dtype=float)
    Te_eV = np.asarray(data.electron_temperature_eV, dtype=float)
    Ti_eV = np.asarray(data.ion_temperature_eV, dtype=float)
    shape = np.where(x <= Ls_cm, np.maximum(u_floor_fraction, 1.0 - x / Ls_cm), 0.0)
    flux_Hp = np.zeros_like(x)
    flux_H2p = np.zeros_like(x)

    for gi_raw in data.p_indices:
        gi = int(gi_raw)
        if gi < 0 or gi >= data.background_full.shape[1]:
            continue
        if int(data.type_id[gi]) != 2 or int(data.charge[gi]) <= 0:
            continue

        mu = max(1.0, float(data.atomicity[gi]))
        density = np.maximum(data.background_full[:, gi], 0.0).copy()
        density[density < density_cutoff_cm3] = 0.0
        mass_amu = mu
        u_bohm = np.sqrt(
            np.maximum(Te_eV + 3.0 * Ti_eV, 0.0) * eV_to_erg / (mass_amu * amu_g)
        )
        flux = mu * density * u_bohm * shape
        if mu >= 2.0:
            flux_H2p += flux
        else:
            flux_Hp += flux

    return flux_Hp, flux_H2p


def compute_ion_nuclei_flux_divergences(
    data: DCRData,
    Ls_cm: float,
    u_floor_fraction: float,
    density_cutoff_cm3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    flux_Hp, flux_H2p = compute_ion_nuclei_fluxes(
        data,
        Ls_cm,
        u_floor_fraction,
        density_cutoff_cm3,
    )
    x = np.asarray(data.x_cm, dtype=float)
    return np.abs(np.gradient(flux_Hp, x)), np.abs(np.gradient(flux_H2p, x))


def save_ion_flux_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, np.isfinite(x), args)
    if args.exclude_last_point and mask.size > 0:
        mask[-1] = False
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two finite x points for ion-flux plotting.")

    flux_Hp, flux_H2p = compute_ion_nuclei_fluxes(
        data,
        args.Ls_cm if args.Ls_cm is not None else float(data.x_cm[-1]),
        resolve_u_floor_fraction(args),
        args.ion_density_cutoff_cm3,
    )
    flux_total = flux_Hp + 0.5 * flux_H2p
    flux_Hp[flux_Hp <= 0.0] = np.nan
    flux_H2p[flux_H2p <= 0.0] = np.nan
    flux_total[flux_total <= 0.0] = np.nan

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    if args.total_ion_flux_only:
        ax.plot(
            x[mask],
            flux_total[mask],
            color="#1f77b4",
            linewidth=2.3,
            label=r"total ion flux",
        )
    else:
        ax.plot(x[mask], flux_Hp[mask], color="#1f77b4", label=r"$\mathrm{H}^{+}$")
        ax.plot(x[mask], flux_H2p[mask], color="#d62728", label=r"$\mathrm{H}_2^{+}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        r"Target-directed total ion flux ($\mathrm{cm^{-2}\,s^{-1}}$)"
        if args.total_ion_flux_only
        else r"Target-directed ion nuclei flux ($\mathrm{cm^{-2}\,s^{-1}}$)"
    )
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(ncol=1, loc="best")

    out = outdir / (
        "total_ion_flux_vs_x.pdf"
        if args.total_ion_flux_only
        else "ion_and_molecular_ion_flux_vs_x.pdf"
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_ion_divergence_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if args.exclude_last_point and mask.size > 0:
        mask[-1] = False
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")

    div_Hp, div_H2p = compute_ion_nuclei_flux_divergences(
        data,
        args.Ls_cm if args.Ls_cm is not None else float(data.x_cm[-1]),
        resolve_u_floor_fraction(args),
        args.ion_density_cutoff_cm3,
    )
    div_Hp[div_Hp <= 0.0] = np.nan
    div_H2p[div_H2p <= 0.0] = np.nan

    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    ax.plot(x[mask], div_Hp[mask], color="#1f77b4", label=r"$\mathrm{H}^{+}$")
    ax.plot(x[mask], div_H2p[mask], color="#d62728", label=r"$\mathrm{H}_2^{+}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$|d\Gamma_i/dx|$ nuclei divergence ($\mathrm{cm^{-3}\,s^{-1}}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(ncol=1, loc="best")

    out = outdir / "ion_and_molecular_ion_divergence_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_local_flow_total_nuclei_density_plot(data: DCRData, args: argparse.Namespace, outdir: Path) -> Path:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    mask = apply_x_max_mask(data.x_cm, np.isfinite(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two finite x points for local-density plotting.")

    local = grouped_local_nuclei_densities(data)
    flow_a = np.zeros(data.x_cm.size)
    flow_m = np.zeros(data.x_cm.size)
    for j, gi_raw in enumerate(data.a_indices):
        if j >= data.flow_a.shape[1]:
            break
        flow_a += np.maximum(data.flow_a[:, j], 0.0) * state_atomicity(data, int(gi_raw))
    for j, gi_raw in enumerate(data.m_indices):
        if j >= data.flow_m.shape[1]:
            break
        flow_m += np.maximum(data.flow_m[:, j], 0.0) * state_atomicity(data, int(gi_raw))
    total = total_nuclei_profile(data)

    standalone_csv = getattr(args, "standalone_cr_csv", None)
    standalone = None
    if standalone_csv is not None:
        if not standalone_csv.exists():
            raise SystemExit(f"Standalone CR CSV not found: {standalone_csv}")
        standalone = np.genfromtxt(standalone_csv, delimiter=",", names=True)

    colors = {
        "H+": "#1f77b4",
        "H-": "#17becf",
        "H": "#2ca02c",
        "H2": "#ff7f0e",
        "H2+": "#d62728",
    }
    labels = {
        "H+": r"$\mathrm{H}^{+}$",
        "H-": r"$\mathrm{H}^{-}$",
        "H": r"$\mathrm{H}$",
        "H2": r"$\mathrm{H}_2$",
        "H2+": r"$\mathrm{H}_2^{+}$",
    }

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    ax_major, ax_minor = axes

    ax_major.plot(x[mask], safe_log_series(local["H+"][mask]), color=colors["H+"], label=labels["H+"])
    ax_major.plot(x[mask], safe_log_series(local["H"][mask]), color=colors["H"], label=labels["H"])
    ax_major.plot(x[mask], safe_log_series(local["H2+"][mask]), color=colors["H2+"], label=labels["H2+"])
    ax_major.plot(x[mask], safe_log_series(flow_a[mask]), color="#4d4d4d", linestyle="--", label=r"flow $A(\mathrm{H})$")
    ax_major.plot(x[mask], safe_log_series(flow_m[mask]), color="#8c564b", linestyle="--", label=r"flow $M(\mathrm{H}_2)$")
    ax_major.plot(x[mask], safe_log_series(total[mask]), color="#000000", linewidth=2.3, label=r"$n_{\rm nuc}$")

    ax_minor.plot(x[mask], safe_log_series(local["H2"][mask]), color=colors["H2"], label=labels["H2"])
    ax_minor.plot(x[mask], safe_log_series(local["H-"][mask]), color=colors["H-"], label=labels["H-"])

    if standalone is not None:
        x_standalone = np.asarray(standalone["x_cm"], dtype=float)
        x_standalone, _ = convert_x(x_standalone, args.x_unit)
        valid = np.isfinite(x_standalone)
        if args.x_max_cm is not None:
            xmax_plot, _ = convert_x(np.asarray([args.x_max_cm]), args.x_unit)
            valid &= x_standalone <= float(xmax_plot[0])
        ax_major.plot(
            x_standalone[valid],
            safe_log_series(np.asarray(standalone["standalone_Hplus"], dtype=float)[valid]),
            color=colors["H+"],
            linestyle=":",
            marker="o",
            linewidth=2.2,
            label=r"standalone CR $\mathrm{H}^{+}$",
        )
        ax_major.plot(
            x_standalone[valid],
            safe_log_series(np.asarray(standalone["standalone_H"], dtype=float)[valid]),
            color=colors["H"],
            linestyle=":",
            marker="o",
            linewidth=2.2,
            label=r"standalone CR $\mathrm{H}$",
        )
        ax_minor.plot(
            x_standalone[valid],
            safe_log_series(2.0 * np.asarray(standalone["standalone_H2"], dtype=float)[valid]),
            color=colors["H2"],
            linestyle=":",
            marker="o",
            linewidth=2.2,
            label=r"standalone CR $\mathrm{H}_2$",
        )

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.set_yscale("log")
        if args.local_density_ymin_cm3 > 0.0:
            ax.set_ylim(bottom=args.local_density_ymin_cm3)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.text(0.02, 0.93, label, transform=ax.transAxes, ha="left", va="top")
        ax.legend(ncol=1, loc="best")

    ax_major.set_ylim(bottom=1.0e9)
    if standalone is not None:
        ax_minor.set_ylim(bottom=min(args.local_density_ymin_cm3, 1.0e3))
    ax_minor.set_xlabel(xlabel)
    fig.supylabel(r"Density ($\mathrm{cm^{-3}}$)")

    out = outdir / "local_flow_total_nuclei_densities_vs_x.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_flow_detail_plot(
    data: DCRData,
    x: np.ndarray,
    xlabel: str,
    matrix: np.ndarray,
    labels: List[str],
    state_indices: List[int],
    flow_kind: str,
    args: argparse.Namespace,
    out_path: Path,
    title: str,
    ylabel: str,
) -> Optional[Path]:
    if matrix.size == 0 or matrix.shape[1] == 0:
        return None

    mask = apply_x_max_mask(data.x_cm, positive_x_mask(x), args)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")
    x_plot = x[mask]
    matrix = matrix[mask, :]

    m_sel, l_sel = matrix, labels
    idx_sel = state_indices
    if flow_kind == "M":
        wanted_v = {0, 1, 2, 3, 4, 5, 10, 14}
        keep = [
            j for j, gi in enumerate(state_indices)
            if molecular_display_v(data, gi, labels[j]) in wanted_v
        ]
        m_sel = matrix[:, keep]
        l_sel = [labels[j] for j in keep]
        idx_sel = [state_indices[j] for j in keep]

    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)

    cmap = plt.get_cmap("tab20")
    for j in range(m_sel.shape[1]):
        y = np.maximum(m_sel[:, j], 0.0)
        y_plot = safe_log_series(y)
        gi = idx_sel[j] if j < len(idx_sel) else -1
        ax.plot(
            x_plot,
            y_plot,
            color=cmap(j % 20),
            label=flow_state_display_label(data, gi, l_sel[j], flow_kind),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def prepare_flow_matrix_and_labels_fraction(
    data: DCRData, flow_kind: str
) -> Tuple[np.ndarray, List[str], List[int]]:
    if flow_kind == "A":
        mat = np.array(data.flow_a, copy=True)
        idx = data.a_indices
    else:
        mat = np.array(data.flow_m, copy=True)
        idx = data.m_indices

    labels: List[str] = []
    state_indices: List[int] = []
    for gi in idx:
        if 0 <= gi < len(data.labels):
            labels.append(data.labels[int(gi)])
        else:
            labels.append(f"state_{int(gi)}")
        state_indices.append(int(gi))

    # For flow-detail plots, show the internal composition of each flow block.
    # Normalize each row by that block's own total density so the plotted state
    # fractions sum to 1 at each x.
    mat = np.maximum(mat, 0.0).copy()
    denom = np.sum(mat, axis=1)
    denom = np.where(denom > 0.0, denom, np.nan)
    mat = mat / denom[:, None]
    return mat, labels, state_indices


def prepare_background_matrix_and_labels_fraction(
    data: DCRData, block_kind: str
) -> Tuple[np.ndarray, List[str], List[int]]:
    if block_kind not in {"A", "M"}:
        raise ValueError(f"Unsupported background block kind: {block_kind}")

    wanted_group = "H" if block_kind == "A" else "H2"
    state_indices: List[int] = []
    labels: List[str] = []

    for gi in data.p_indices:
        gi_int = int(gi)
        if classify_state(data, gi_int) != wanted_group:
            continue
        state_indices.append(gi_int)
        if 0 <= gi_int < len(data.labels):
            labels.append(data.labels[gi_int])
        else:
            labels.append(f"state_{gi_int}")

    if not state_indices:
        return np.zeros((data.background_full.shape[0], 0)), labels, state_indices

    mat = np.maximum(data.background_full[:, state_indices], 0.0).copy()
    denom = np.sum(mat, axis=1)
    denom = np.where(denom > 0.0, denom, np.nan)
    mat = mat / denom[:, None]
    return mat, labels, state_indices


def run_groups(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_group_plot(data, args, outdir)
    return [out]


def run_proposal_summary(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_proposal_summary_plot(data, args, outdir)
    return [out]


def run_atom_source_rates(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_atom_source_rates_plot(data, args, outdir)
    return [out]


def run_ionization_vs_h_pump(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_ionization_vs_h_pump_plot(data, args, outdir)
    return [out]


def run_molecular_flow_rates(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_molecular_flow_rates_plot(data, args, outdir)
    return [out]


def run_molecular_dissociation_vs_ionization(
    data: DCRData, args: argparse.Namespace, outdir: Path
) -> List[Path]:
    return [save_molecular_dissociation_vs_ionization_plot(data, args, outdir)]


def run_mar_transport_comparison(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    return [save_mar_transport_comparison_plot(data, args, outdir)]


def run_recombination_degree(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    return [save_recombination_degree_plot(data, args, outdir)]


def run_power_loss(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_power_loss_plot(data, args, outdir)
    return [out]


def run_normalized_flow(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_normalized_flow_density_plot(data, args, outdir)
    return [out]


def run_source_to_local(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    out = save_source_to_local_plot(data, args, outdir)
    return [out]


def run_variable_diagnostics(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    return [
        save_ion_flux_plot(data, args, outdir),
        save_ion_divergence_plot(data, args, outdir),
        save_local_flow_total_nuclei_density_plot(data, args, outdir),
    ]


def run_flow_detail(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    written: List[Path] = []

    mat_a, labels_a, idx_a = prepare_flow_matrix_and_labels_fraction(data, "A")
    out_a = save_flow_detail_plot(
        data=data,
        x=x,
        xlabel=xlabel,
        matrix=mat_a,
        labels=labels_a,
        state_indices=idx_a,
        flow_kind="A",
        args=args,
        out_path=outdir / "flow_A_states_vs_x.pdf",
        title="Flow A(H) State Composition",
        ylabel="Normalized Flow Population",
    )
    if out_a is not None:
        written.append(out_a)

    mat_m, labels_m, idx_m = prepare_flow_matrix_and_labels_fraction(data, "M")
    out_m = save_flow_detail_plot(
        data=data,
        x=x,
        xlabel=xlabel,
        matrix=mat_m,
        labels=labels_m,
        state_indices=idx_m,
        flow_kind="M",
        args=args,
        out_path=outdir / "flow_M_states_vs_x.pdf",
        title="Flow M(H2) State Composition",
        ylabel="Normalized Flow Population",
    )
    if out_m is not None:
        written.append(out_m)

    return written


def run_background_detail(data: DCRData, args: argparse.Namespace, outdir: Path) -> List[Path]:
    x, xlabel = convert_x(data.x_cm, args.x_unit)
    written: List[Path] = []

    mat_a, labels_a, idx_a = prepare_background_matrix_and_labels_fraction(data, "A")
    out_a = save_flow_detail_plot(
        data=data,
        x=x,
        xlabel=xlabel,
        matrix=mat_a,
        labels=labels_a,
        state_indices=idx_a,
        flow_kind="A",
        args=args,
        out_path=outdir / "bg_atomic_states_vs_x.pdf",
        title="Local Atomic H State Composition",
        ylabel="Normalized Local Population",
    )
    if out_a is not None:
        written.append(out_a)

    mat_m, labels_m, idx_m = prepare_background_matrix_and_labels_fraction(data, "M")
    out_m = save_flow_detail_plot(
        data=data,
        x=x,
        xlabel=xlabel,
        matrix=mat_m,
        labels=labels_m,
        state_indices=idx_m,
        flow_kind="M",
        args=args,
        out_path=outdir / "bg_molecular_states_vs_x.pdf",
        title="Local Molecular H2 State Composition",
        ylabel="Normalized Local Population",
    )
    if out_m is not None:
        written.append(out_m)

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process DCR HDF5 output and generate journal-style plots."
    )

    def add_common_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", required=True, type=Path, help="Path to dcr_results.h5")
        p.add_argument(
            "--outdir",
            type=Path,
            default=None,
            help="Output directory for plots (default: <input_dir>/plots_journal)",
        )
        p.add_argument("--x-unit", choices=["cm", "m"], default="cm")
        p.add_argument("--x-max-cm", type=float, default=None, help="Upper x limit in cm for plotted data")
        p.add_argument("--dpi", type=int, default=300, help="Figure DPI")
        p.add_argument("--font-scale", type=float, default=1.4, help="Global font scaling factor")

    sub = parser.add_subparsers(dest="command", required=True)
    p_groups = sub.add_parser("groups", help="Plot grouped background and flow totals vs x")
    add_common_options(p_groups)
    p_groups.add_argument("--standalone-cr-csv", type=Path, default=None, help="Overlay standalone local-CR H+, H, and H2 curves from CSV")
    p_groups.add_argument("--hide-standalone-h2", action="store_true", help="Omit the standalone H2 comparison curve")
    p_groups.add_argument("--group-ymin", type=float, default=None, help="Override the grouped plot lower y-axis limit")

    p_proposal = sub.add_parser("proposal-summary", help="Plot a clean DCR transport/recycling summary figure")
    add_common_options(p_proposal)

    p_sources = sub.add_parser("atom-source-rates", help="Plot effective EIR, MAR, and atomic recycling-flow source rates")
    add_common_options(p_sources)

    p_ion_pump = sub.add_parser("ionization-vs-h-pump", help="Plot effective ionization against the neutral H pump rate")
    add_common_options(p_ion_pump)

    p_mol_flow = sub.add_parser("molecular-flow-rates", help="Plot molecular-flow ionization and charge-exchange ion sources")
    add_common_options(p_mol_flow)

    p_mol_compare = sub.add_parser(
        "molecular-dissociation-vs-ionization",
        help="Plot molecular dissociation and ionization event rates",
    )
    add_common_options(p_mol_compare)

    p_mar_transport = sub.add_parser(
        "mar-transport-comparison",
        help="Compare nonlocal state-resolved MAR against local simple branching",
    )
    add_common_options(p_mar_transport)

    p_recombination_degree = sub.add_parser(
        "recombination-degree",
        help="Plot cumulative all-channel positive-ion nuclei sinks divided by sources",
    )
    add_common_options(p_recombination_degree)

    p_power_loss = sub.add_parser("power-loss", help="Plot electron inelastic power-loss diagnostics")
    add_common_options(p_power_loss)

    p_norm_flow = sub.add_parser("normalized-flow", help="Plot flow densities normalized to boundary values")
    add_common_options(p_norm_flow)

    p_source_local = sub.add_parser("source-to-local", help="Plot routed flow-to-local source rates")
    add_common_options(p_source_local)

    p_variable = sub.add_parser(
        "variable-diagnostics",
        aliases=["vd"],
        help="Plot variable-density ion divergence and absolute nuclei densities",
    )
    add_common_options(p_variable)
    p_variable.add_argument("--Ls-cm", "--Ls", dest="Ls_cm", type=float, default=None, help="Ion velocity length scale used in the run (default: grid end)")
    p_variable.add_argument("-u", "--ufloor", "--u-floor", "--u-floor-fraction", dest="u_floor_fraction", type=float, default=None, help="Ion velocity floor as a fraction of u_B; inferred from ufloor001/003/01 path tags when omitted")
    p_variable.add_argument("--ion-density-cutoff-cm3", type=float, default=1.0e1, help="Set ion densities below this value to zero before divergence")
    p_variable.add_argument("--local-density-ymin-cm3", type=float, default=1.0e5, help="Lower y-axis limit for local/flow/total density plot")
    p_variable.add_argument("--total-ion-flux-only", action="store_true", help="Plot only the total positive-ion particle flux")
    p_variable.add_argument("--standalone-cr-csv", type=Path, default=None, help="Overlay standalone local-CR H+, H, and H2 curves on local density plot")
    p_variable.add_argument("--exclude-last-point", action="store_true", default=True, help="Exclude the last grid point from ion-divergence plot")
    p_variable.add_argument("--include-last-point", dest="exclude_last_point", action="store_false", help="Include the last grid point in ion-divergence plot")

    p_flow = sub.add_parser("flow-detail", help="Plot detailed flow-state evolution vs x")
    add_common_options(p_flow)

    p_bg = sub.add_parser("background-detail", help="Plot detailed local atomic and molecular state composition vs x")
    add_common_options(p_bg)

    p_all = sub.add_parser("all", help="Generate grouped, flow-detail, and local background-detail plots")
    add_common_options(p_all)
    p_all.add_argument("--standalone-cr-csv", type=Path, default=None, help="Overlay standalone local-CR H+, H, and H2 curves on grouped density plot")
    p_all.add_argument("--hide-standalone-h2", action="store_true", help="Omit the standalone H2 comparison curve")
    p_all.add_argument("--group-ymin", type=float, default=None, help="Override the grouped plot lower y-axis limit")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    outdir = args.outdir if args.outdir is not None else (args.input.parent / "plots_journal")
    outdir.mkdir(parents=True, exist_ok=True)

    configure_journal_style(args.dpi, args.font_scale)
    data = load_hdf5(args.input)

    written: List[Path] = []
    if args.command == "groups":
        written.extend(run_groups(data, args, outdir))
    elif args.command == "proposal-summary":
        written.extend(run_proposal_summary(data, args, outdir))
    elif args.command == "atom-source-rates":
        written.extend(run_atom_source_rates(data, args, outdir))
    elif args.command == "ionization-vs-h-pump":
        written.extend(run_ionization_vs_h_pump(data, args, outdir))
    elif args.command == "molecular-flow-rates":
        written.extend(run_molecular_flow_rates(data, args, outdir))
    elif args.command == "molecular-dissociation-vs-ionization":
        written.extend(run_molecular_dissociation_vs_ionization(data, args, outdir))
    elif args.command == "mar-transport-comparison":
        written.extend(run_mar_transport_comparison(data, args, outdir))
    elif args.command == "recombination-degree":
        written.extend(run_recombination_degree(data, args, outdir))
    elif args.command == "power-loss":
        written.extend(run_power_loss(data, args, outdir))
    elif args.command == "normalized-flow":
        written.extend(run_normalized_flow(data, args, outdir))
    elif args.command == "source-to-local":
        written.extend(run_source_to_local(data, args, outdir))
    elif args.command in {"variable-diagnostics", "vd"}:
        written.extend(run_variable_diagnostics(data, args, outdir))
    elif args.command == "flow-detail":
        written.extend(run_flow_detail(data, args, outdir))
    elif args.command == "background-detail":
        written.extend(run_background_detail(data, args, outdir))
    elif args.command == "all":
        written.extend(run_groups(data, args, outdir))
        written.extend(run_flow_detail(data, args, outdir))
        written.extend(run_background_detail(data, args, outdir))

    if not written:
        print("[postprocess] No plots were generated (empty flow blocks?).")
    else:
        print("[postprocess] Wrote:")
        for p in written:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
