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
    molecular_flow_lhs_transport_rate_cm3_s: Optional[np.ndarray] = None


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
        molecular_flow_lhs_transport_rate_cm3_s = (
            np.asarray(f["/rates/molecular_flow_lhs_transport_rate_cm3_s"][:], dtype=float)
            if "/rates/molecular_flow_lhs_transport_rate_cm3_s" in f else None
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
        molecular_flow_lhs_transport_rate_cm3_s=molecular_flow_lhs_transport_rate_cm3_s,
    )


def convert_x(x_cm: np.ndarray, x_unit: str) -> Tuple[np.ndarray, str]:
    if x_unit == "m":
        return 1.0e-2 * x_cm, "x (m)"
    return x_cm, "x (cm)"


def safe_log_series(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[out <= 0.0] = np.nan
    return out


def positive_x_mask(x: np.ndarray) -> np.ndarray:
    # Log-x plotting cannot represent x=0 (boundary node), so skip non-positive locations.
    return x > 0.0


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


def grouped_curves_fraction(data: DCRData) -> Dict[str, np.ndarray]:
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

    total = total_nuclei_profile(data)
    denom = np.where(total > 0.0, total, np.nan)
    out: Dict[str, np.ndarray] = {}
    for k, v in curves.items():
        out[k] = v / denom
    return out


def state_display_label(data: DCRData, gi: int, label: str, block_kind: str) -> str:
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
    mask = positive_x_mask(x)
    if int(np.count_nonzero(mask)) < 2:
        raise SystemExit("Need at least two positive x points for log-x plotting.")
    x_plot = x[mask]
    curves = grouped_curves_fraction(data)

    colors = {
        "H+": "#1f77b4",
        "H-": "#17becf",
        "H": "#2ca02c",
        "H2": "#ff7f0e",
        "H2+": "#d62728",
        "A(H)": "#4d4d4d",
        "M(H2)": "#8c564b",
    }
    styles = {
        "H+": "-",
        "H-": "-",
        "H": "-",
        "H2": "-",
        "H2+": "-",
        "A(H)": "--",
        "M(H2)": "--",
    }
    order = ["H+", "H-", "H", "H2", "H2+", "A(H)", "M(H2)"]
    legend = {
        "H+": r"$\mathrm{H}^{+}$",
        "H-": r"$\mathrm{H}^{-}$",
        "H": r"$\mathrm{H}$",
        "H2": r"$\mathrm{H}_2$",
        "H2+": r"$\mathrm{H}_2^{+}$",
        "A(H)": r"recycled flow $A(\mathrm{H})$",
        "M(H2)": r"recycled flow $M(\mathrm{H}_2)$",
    }

    fig, (ax_hi, ax_lo) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    for ax in (ax_hi, ax_lo):
        for key in order:
            y = curves[key][mask]
            y_plot = safe_log_series(y)
            ax.plot(x_plot, y_plot, linestyle=styles[key], color=colors[key], label=legend[key])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)

    # Broken y-axis: hide the empty middle decade range so both abundant and
    # trace local species are readable in the same figure.
    ax_hi.set_ylim(5.0e-4, 2.0)
    ax_lo.set_ylim(1.0e-10, 3.0e-8)
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(labelbottom=False)
    ax_lo.set_xlabel(xlabel)
    fig.supylabel("Fractional Population")
    ax_hi.legend(ncol=2, loc="best")

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
    mask = positive_x_mask(x)
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
    mask = positive_x_mask(x)
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
    mask = positive_x_mask(x)
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
    mask = positive_x_mask(x)
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

    mask = positive_x_mask(x)
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
            label=state_display_label(data, gi, l_sel[j], flow_kind),
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
        p.add_argument("--dpi", type=int, default=300, help="Figure DPI")
        p.add_argument("--font-scale", type=float, default=1.4, help="Global font scaling factor")

    sub = parser.add_subparsers(dest="command", required=True)
    p_groups = sub.add_parser("groups", help="Plot grouped background and flow totals vs x")
    add_common_options(p_groups)

    p_proposal = sub.add_parser("proposal-summary", help="Plot a clean DCR transport/recycling summary figure")
    add_common_options(p_proposal)

    p_sources = sub.add_parser("atom-source-rates", help="Plot effective EIR, MAR, and atomic recycling-flow source rates")
    add_common_options(p_sources)

    p_ion_pump = sub.add_parser("ionization-vs-h-pump", help="Plot effective ionization against the neutral H pump rate")
    add_common_options(p_ion_pump)

    p_mol_flow = sub.add_parser("molecular-flow-rates", help="Plot molecular-flow ionization and charge-exchange ion sources")
    add_common_options(p_mol_flow)

    p_flow = sub.add_parser("flow-detail", help="Plot detailed flow-state evolution vs x")
    add_common_options(p_flow)

    p_bg = sub.add_parser("background-detail", help="Plot detailed local atomic and molecular state composition vs x")
    add_common_options(p_bg)

    p_all = sub.add_parser("all", help="Generate grouped, flow-detail, and local background-detail plots")
    add_common_options(p_all)

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
    elif args.command == "flow-detail":
        written.extend(run_flow_detail(data, args, outdir))
    elif args.command == "background-detail":
        written.extend(run_background_detail(data, args, outdir))
    elif args.command == "all":
        written.extend(run_groups(data, args, outdir))
        written.extend(run_atom_source_rates(data, args, outdir))
        written.extend(run_molecular_flow_rates(data, args, outdir))
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
