#!/usr/bin/env python3
"""
Compute and plot Radiated Power Loss (RPL) from DCR solver HDF5 output.

Reads background_full plus atomic excited flow populations from DCR HDF5 output,
parses oscillator strengths from the H-atom atomic data file (sclit.01), and computes:

    P_rad(x) [W/cm^3] = sum_{n>n'} n_total(H,n,x) * A(n->n') * delta_E(n->n') * eV_to_J

where n_total(H,n,x) includes:
  - local/background excited H populations from /population/background_full
  - excited atomic flow populations from /population/flowA for full_dcr outputs
  - reconstructed excited atomic flow populations from /population/qss_flow_transient_atomic
    for qss_dcr outputs, when present

Einstein A coefficient from oscillator strength (SE_RATE_CONST verified against sclit.01):
    A_ji [s^-1] = SE_RATE_CONST * (g_lower/g_upper) * f_ij * (delta_E_eV)^2
    SE_RATE_CONST = 4.3376e7  [s^-1 eV^-2]

Usage:
    # Single run plot
    python scripts/compute_rpl.py output/test_run_qss/5eV/dcr_results.h5

    # Comparison between two runs
    python scripts/compute_rpl.py \
        output/test_run_Tx_profile/dcr_results.h5 \
        output/adas_compare/no_h2plus_dr/dcr_results.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit("Missing h5py: pip install h5py") from exc

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc


# ── Constants ─────────────────────────────────────────────────────────────────

# A_ji [s^-1] = SE_RATE_CONST * (g_l/g_u) * f_ij * (delta_E_eV)^2
# Derived from CGS formula: (8 pi^2 e^2 nu^2)/(m_e c^3) * (g_l/g_u) * f_lu
# Verified: A_avg(n=2) = 4.70e8 s^-1  (matches sclit.01 level 2 decay rate)
SE_RATE_CONST = 4.3376e7   # s^-1 eV^-2

EV_TO_J = 1.602176634e-19  # J per eV
DEFAULT_ATOMIC = Path("atomic_data/sclit.01")
DEFAULT_LW = Path("atomic_data/lw_rates.npz")


# ── Atomic data parser ────────────────────────────────────────────────────────

class HAtomData:
    """
    Parses sclit.01 (or equivalent) for H-atom level energies, degeneracies,
    and bound-bound oscillator strengths.
    """

    def __init__(self, path: Path) -> None:
        self.levels: Dict[int, Tuple[float, float]] = {}  # n -> (energy_eV, g)
        self.transitions: List[Tuple[int, int, float]] = []  # (n_lower, n_upper, f_ij)
        self._parse(path)

    def _parse(self, path: Path) -> None:
        """
        Level lines start with an integer charge block indicator.
        Bound-bound lines start with 'b'.
        Format for level (neutral H, charge=1 in Z-notation of file):
          Z  local_idx  global_idx  label  excitation_eV  g  ...
        Format for bound-bound:
          b  Z_l  n_l  Z_u  n_u  flag  p0 p1 p2 p3 p4  f_ij  f_ij_dup
        """
        in_neutral = False
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if not parts:
                    continue

                # Bound-bound transition
                if parts[0] == "b":
                    # b  Z_lower  n_lower  Z_upper  n_upper  flag  [5 values]  f_ij  f_ij
                    if len(parts) < 13:
                        continue
                    try:
                        n_lower = int(parts[2])
                        n_upper = int(parts[4])
                        f_ij = float(parts[-1])  # last column
                    except (ValueError, IndexError):
                        continue
                    if n_lower >= 1 and n_upper > n_lower and f_ij > 0.0:
                        self.transitions.append((n_lower, n_upper, f_ij))
                    continue

                # Skip non-level lines
                if parts[0] in ("f", "c", "r", "a"):
                    continue

                # Level line: first token is Z (integer), second is local index
                try:
                    z = int(parts[0])
                    n_local = int(parts[1])
                    energy_eV = float(parts[4])
                    g = float(parts[5])
                except (ValueError, IndexError):
                    continue

                # Only neutral H excited levels (z=1 in file encoding, not bare nucleus)
                if z == 1 and n_local >= 1:
                    self.levels[n_local] = (energy_eV, g)

    def einstein_a(self, n_lower: int, n_upper: int, f_ij: float) -> Tuple[float, float]:
        """
        Returns (A_ji [s^-1], delta_E [eV]) for transition n_upper -> n_lower.
        """
        if n_lower not in self.levels or n_upper not in self.levels:
            return 0.0, 0.0
        e_lower, g_lower = self.levels[n_lower]
        e_upper, g_upper = self.levels[n_upper]
        delta_e = e_upper - e_lower
        if delta_e <= 0.0 or g_upper <= 0.0:
            return 0.0, 0.0
        a_ji = SE_RATE_CONST * (g_lower / g_upper) * f_ij * delta_e ** 2
        return a_ji, delta_e


# ── HDF5 loader ───────────────────────────────────────────────────────────────

def load_dcr(path: Path):
    """Load DCR results and return relevant arrays."""
    with h5py.File(path, "r") as f:
        x_cm = np.asarray(f["/grid/x_cm"][:], dtype=float)
        bg = np.asarray(f["/population/background_full"][:], dtype=float)
        flow_a = np.asarray(f["/population/flowA"][:], dtype=float)
        flow_m = np.asarray(f["/population/flowM"][:], dtype=float) if "/population/flowM" in f else None
        a_indices = np.asarray(f["/states/A_indices"][:], dtype=int)
        m_indices = np.asarray(f["/states/M_indices"][:], dtype=int) if "/states/M_indices" in f else None
        type_id = np.asarray(f["/states/type_id"][:], dtype=int)
        charge = np.asarray(f["/states/charge"][:], dtype=int)
        internal_id = np.asarray(f["/states/internal_id"][:], dtype=int)
        atomicity = np.asarray(f["/states/atomicity"][:], dtype=int)
        qss_flow_transient_atomic = (
            np.asarray(f["/population/qss_flow_transient_atomic"][:], dtype=float)
            if "/population/qss_flow_transient_atomic" in f else None
        )
        qss_flow_transient_atomic_indices = (
            np.asarray(f["/states/qss_flow_transient_atomic_indices"][:], dtype=int)
            if "/states/qss_flow_transient_atomic_indices" in f else None
        )
        # Plasma profiles (may have n_cells rows vs n_nodes for populations)
        Te_eV = np.asarray(f["/rates/Te_eV"][:], dtype=float) if "/rates/Te_eV" in f else None
        ne_cm3 = np.asarray(f["/rates/ne_cm3"][:], dtype=float) if "/rates/ne_cm3" in f else None
    return (
        x_cm,
        bg,
        flow_a,
        flow_m,
        a_indices,
        m_indices,
        qss_flow_transient_atomic,
        qss_flow_transient_atomic_indices,
        type_id,
        charge,
        internal_id,
        atomicity,
        Te_eV,
        ne_cm3,
    )


# ── RPL computation ───────────────────────────────────────────────────────────

def compute_rpl(
    bg: np.ndarray,
    flow_a: np.ndarray,
    a_indices: np.ndarray,
    qss_flow_transient_atomic: Optional[np.ndarray],
    qss_flow_transient_atomic_indices: Optional[np.ndarray],
    type_id: np.ndarray,
    charge: np.ndarray,
    internal_id: np.ndarray,
    atom_data: HAtomData,
) -> np.ndarray:
    """
    Compute radiated power loss P_rad(x) [W/cm^3] at each spatial node.

    P_rad(x) = sum_{n>n'} n(H,n,x) [cm^-3] * A(n->n') [s^-1] * delta_E(n->n') [J]
    """
    n_nodes, n_states = bg.shape

    # Map n (quantum number) -> total excited H population profile (background + flow).
    n_to_population: Dict[int, np.ndarray] = {}
    for gi in range(n_states):
        if (gi < type_id.size and gi < charge.size and gi < internal_id.size):
            if int(type_id[gi]) == 0 and int(charge[gi]) == 0:  # neutral atom
                n = int(internal_id[gi])
                if n >= 2:
                    n_to_population[n] = np.maximum(bg[:, gi], 0.0).copy()

    for j, gi in enumerate(a_indices):
        if j >= flow_a.shape[1]:
            break
        if gi < 0 or gi >= type_id.size or gi >= charge.size or gi >= internal_id.size:
            continue
        if int(type_id[gi]) != 0 or int(charge[gi]) != 0:
            continue
        n = int(internal_id[gi])
        if n < 2:
            continue
        n_to_population.setdefault(n, np.zeros(n_nodes))
        n_to_population[n] += np.maximum(flow_a[:, j], 0.0)

    if qss_flow_transient_atomic is not None and qss_flow_transient_atomic_indices is not None:
        for j, gi in enumerate(qss_flow_transient_atomic_indices):
            if j >= qss_flow_transient_atomic.shape[1]:
                break
            if gi < 0 or gi >= type_id.size or gi >= charge.size or gi >= internal_id.size:
                continue
            if int(type_id[gi]) != 0 or int(charge[gi]) != 0:
                continue
            n = int(internal_id[gi])
            if n < 2:
                continue
            n_to_population.setdefault(n, np.zeros(n_nodes))
            n_to_population[n] += np.maximum(qss_flow_transient_atomic[:, j], 0.0)

    if not n_to_population:
        raise RuntimeError("No excited H states found in background_full. "
                           "Check type_id/charge/internal_id metadata.")

    p_rad = np.zeros(n_nodes)

    for n_lower, n_upper, f_ij in atom_data.transitions:
        if n_upper not in n_to_population:
            continue  # upper level not in output (below n=2)

        a_ji, delta_e = atom_data.einstein_a(n_lower, n_upper, f_ij)
        if a_ji <= 0.0:
            continue

        n_upper_arr = n_to_population[n_upper]
        p_rad += n_upper_arr * a_ji * delta_e * EV_TO_J  # W/cm^3

    return p_rad


def compute_rpl_components(
    bg: np.ndarray,
    flow_a: np.ndarray,
    a_indices: np.ndarray,
    qss_flow_transient_atomic: Optional[np.ndarray],
    qss_flow_transient_atomic_indices: Optional[np.ndarray],
    type_id: np.ndarray,
    charge: np.ndarray,
    internal_id: np.ndarray,
    atom_data: HAtomData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute total/background/flow atomic line radiation [W/cm^3].
    """
    n_nodes, n_states = bg.shape
    n_to_background: Dict[int, np.ndarray] = {}
    n_to_flow: Dict[int, np.ndarray] = {}

    for gi in range(n_states):
        if gi < type_id.size and gi < charge.size and gi < internal_id.size:
            if int(type_id[gi]) == 0 and int(charge[gi]) == 0:
                n = int(internal_id[gi])
                if n >= 2:
                    n_to_background[n] = np.maximum(bg[:, gi], 0.0).copy()

    for j, gi in enumerate(a_indices):
        if j >= flow_a.shape[1]:
            break
        if gi < 0 or gi >= type_id.size or gi >= charge.size or gi >= internal_id.size:
            continue
        if int(type_id[gi]) != 0 or int(charge[gi]) != 0:
            continue
        n = int(internal_id[gi])
        if n < 2:
            continue
        n_to_flow.setdefault(n, np.zeros(n_nodes))
        n_to_flow[n] += np.maximum(flow_a[:, j], 0.0)

    if qss_flow_transient_atomic is not None and qss_flow_transient_atomic_indices is not None:
        for j, gi in enumerate(qss_flow_transient_atomic_indices):
            if j >= qss_flow_transient_atomic.shape[1]:
                break
            if gi < 0 or gi >= type_id.size or gi >= charge.size or gi >= internal_id.size:
                continue
            if int(type_id[gi]) != 0 or int(charge[gi]) != 0:
                continue
            n = int(internal_id[gi])
            if n < 2:
                continue
            n_to_flow.setdefault(n, np.zeros(n_nodes))
            n_to_flow[n] += np.maximum(qss_flow_transient_atomic[:, j], 0.0)

    if not n_to_background and not n_to_flow:
        raise RuntimeError("No excited H states found in background_full or excited flow datasets.")

    p_background = np.zeros(n_nodes)
    p_flow = np.zeros(n_nodes)
    for n_lower, n_upper, f_ij in atom_data.transitions:
        a_ji, delta_e = atom_data.einstein_a(n_lower, n_upper, f_ij)
        if a_ji <= 0.0:
            continue
        if n_upper in n_to_background:
            p_background += n_to_background[n_upper] * a_ji * delta_e * EV_TO_J
        if n_upper in n_to_flow:
            p_flow += n_to_flow[n_upper] * a_ji * delta_e * EV_TO_J

    return p_background + p_flow, p_background, p_flow


# ── Molecular electronic excitation RPL ──────────────────────────────────────

def compute_lw_rpl(
    bg: np.ndarray,
    flow_m: Optional[np.ndarray],
    m_indices: Optional[np.ndarray],
    charge: np.ndarray,
    atomicity: np.ndarray,
    internal_id: np.ndarray,
    Te_eV_profile: np.ndarray,
    ne_cm3_profile: np.ndarray,
    lw_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Lyman-Werner molecular RPL [W/cm^3] at each spatial node.

    P_LW(x) = ne(x) * sum_{vi,s} n_H2(vi,x) * k_s(vi,Te(x)) * E_photon(s) [W/cm^3]

    Includes both background H2 (from background_full) and flow H2 (from flowM).
    Returns (p_lw_total, p_lw_bg, p_lw_flow).

    Uses the state-resolved k_tot(state, vi, Te) table over the 4 singlet ungerade
    states (B¹Σu⁺, B'¹Σu⁺, C¹Πu, D¹Πu) and weights each by its own photon energy.
    lw_path points to atomic_data/lw_rates.npz produced by precompute_lw_rates.py.
    """
    lw = np.load(lw_path)
    Te_grid  = lw["Te_eV"]                  # (N_Te,)
    k_table  = lw["k_tot"]                  # (4, 15, N_Te)
    E_photon = lw["E_photon_eV"]            # (4,)

    n_nodes = bg.shape[0]

    if Te_eV_profile is None or ne_cm3_profile is None:
        zero = np.zeros(n_nodes)
        return zero, zero, zero

    # Interpolate Te and ne onto node grid if shapes differ
    node_idx = np.arange(n_nodes, dtype=float)
    rate_idx  = np.linspace(0, n_nodes - 1, len(Te_eV_profile))
    Te_nodes = np.interp(node_idx, rate_idx, Te_eV_profile)
    ne_nodes = np.interp(node_idx, rate_idx, ne_cm3_profile)

    p_bg   = np.zeros(n_nodes)
    p_flow = np.zeros(n_nodes)

    # --- background H2 from background_full ---
    for gi in range(bg.shape[1]):
        if gi >= len(charge) or gi >= len(atomicity) or gi >= len(internal_id):
            continue
        if int(charge[gi]) != 0 or int(atomicity[gi]) != 2:
            continue
        vi = int(internal_id[gi])
        if vi < 0 or vi >= k_table.shape[1]:
            continue
        emissivity_vi = np.zeros(n_nodes)
        for si in range(k_table.shape[0]):
            k_vi = np.interp(
                Te_nodes, Te_grid, k_table[si, vi], left=0.0, right=k_table[si, vi, -1]
            )
            emissivity_vi += k_vi * E_photon[si] * EV_TO_J
        p_bg += ne_nodes * np.maximum(bg[:, gi], 0.0) * emissivity_vi

    # --- flow H2 from flowM ---
    if flow_m is not None and m_indices is not None:
        for j, gi in enumerate(m_indices):
            if j >= flow_m.shape[1]:
                break
            if gi < 0 or gi >= len(charge) or gi >= len(atomicity) or gi >= len(internal_id):
                continue
            if int(charge[gi]) != 0 or int(atomicity[gi]) != 2:
                continue
            vi = int(internal_id[gi])
            if vi < 0 or vi >= k_table.shape[1]:
                continue
            emissivity_vi = np.zeros(n_nodes)
            for si in range(k_table.shape[0]):
                k_vi = np.interp(
                    Te_nodes, Te_grid, k_table[si, vi], left=0.0, right=k_table[si, vi, -1]
                )
                emissivity_vi += k_vi * E_photon[si] * EV_TO_J
            p_flow += ne_nodes * np.maximum(flow_m[:, j], 0.0) * emissivity_vi

    return p_bg + p_flow, p_bg, p_flow


# ── Plotting ──────────────────────────────────────────────────────────────────

def configure_style(dpi: int, font_scale: float) -> None:
    fs = 10.0 * font_scale
    mpl.rcParams.update({
        "savefig.dpi": dpi, "figure.dpi": dpi,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "axes.labelsize": fs + 1, "axes.titlesize": fs + 2,
        "xtick.labelsize": fs, "ytick.labelsize": fs,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": False, "legend.fontsize": fs - 0.5,
        "lines.linewidth": 1.9,
    })


def default_label_for_path(path: Path) -> str:
    text = path.as_posix().lower()
    if "qss" in text:
        return "QSS"
    if "full" in text:
        return "Full"
    if path.parent.name and path.parent.name != ".":
        return path.parent.name
    return path.stem


def plot_rpl(
    results: List[Tuple[np.ndarray, np.ndarray, str]],  # (x_cm, p_rad, label)
    outpath: Path,
    x_unit: str = "cm",
) -> None:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    xlabel = "x (m)" if x_unit == "m" else "x (cm)"
    for i, (x_cm, p_rad, label) in enumerate(results):
        x = x_cm * 1e-2 if x_unit == "m" else x_cm
        mask = x > 0.0
        xp = x[mask]
        yp = np.where(p_rad[mask] > 0, p_rad[mask], np.nan)
        ax.plot(xp, yp, color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)], label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$P_\mathrm{rad}$ (W cm$^{-3}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend()
    ax.set_title("Radiated Power Loss Profile")

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[compute_rpl] Wrote: {outpath}")


def plot_rpl_components(
    results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    outpath: Path,
    x_unit: str = "cm",
) -> None:
    """
    results: list of (x_cm, p_total, p_atomic_bg, p_atomic_flow, p_mol_bg, p_mol_flow, label)

    2×2 panels: top row = atomic (bg | flow), bottom row = mol. excitation (bg | flow).
    Runs are distinguished by color; linestyle encodes solver: solid = run[0] (full DCR),
    dashed = run[1] (QSS), dotted = run[2], etc.
    """
    colors    = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    linestyles = ["-", "--", ":", "-."]
    xlabel = "x (m)" if x_unit == "m" else "x (cm)"

    has_mol = any(p_mol_bg.max() + p_mol_flow.max() > 0
                  for _, _, _, _, p_mol_bg, p_mol_flow, _ in results)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)

    panel_data = [
        (axes[0, 0], "Atomic RPL — Background",          2),  # p_atomic_bg
        (axes[0, 1], "Atomic RPL — Flow",                3),  # p_atomic_flow
        (axes[1, 0], "Mol. Excitation RPL — Background", 4),  # p_mol_bg
        (axes[1, 1], "Mol. Excitation RPL — Flow",       5),  # p_mol_flow
    ]

    for ax, title, col_idx in panel_data:
        for i, row in enumerate(results):
            x_cm  = row[0]
            label = row[6]
            y     = row[col_idx]
            x     = x_cm * 1e-2 if x_unit == "m" else x_cm
            mask  = x > 0.0
            xp    = x[mask]
            yp    = np.where(y[mask] > 0, y[mask], np.nan)
            ax.plot(xp, yp,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$P_{\mathrm{rad}}$ (W cm$^{-3}$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.legend(fontsize=8)

    if not has_mol:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[compute_rpl] Wrote: {outpath}")


def plot_atomic_vs_molecular(
    results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    outpath: Path,
    x_unit: str = "cm",
) -> None:
    """
    results: list of (x_cm, p_total, p_atomic_bg, p_atomic_flow, p_mol_bg, p_mol_flow, label)

    One panel per run showing total atomic, total molecular, and total RPL.
    """
    xlabel = "x (m)" if x_unit == "m" else "x (cm)"
    nrows = len(results)
    fig, axes = plt.subplots(nrows, 1, figsize=(10.0, 3.8 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    for ax, row in zip(axes, results):
        x_cm, p_total, p_atomic_bg, p_atomic_flow, p_mol_bg, p_mol_flow, label = row
        x = x_cm * 1e-2 if x_unit == "m" else x_cm
        p_atomic = p_atomic_bg + p_atomic_flow
        p_molecular = p_mol_bg + p_mol_flow
        mask = x > 0.0
        xp = x[mask]

        ax.plot(xp, np.where(p_atomic[mask] > 0, p_atomic[mask], np.nan),
                color="#1f77b4", linestyle="-", label="Atomic total")
        ax.plot(xp, np.where(p_molecular[mask] > 0, p_molecular[mask], np.nan),
                color="#d62728", linestyle="--", label="Molecular total")
        ax.plot(xp, np.where(p_total[mask] > 0, p_total[mask], np.nan),
                color="#2ca02c", linestyle="-.", label="Total")

        ax.set_title(f"Atomic vs Molecular RPL — {label}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$P_{\mathrm{rad}}$ (W cm$^{-3}$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.legend()

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[compute_rpl] Wrote: {outpath}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", type=Path,
                   help="One or two dcr_results.h5 files")
    p.add_argument("--input", default=None, type=Path,
                   help="Primary dcr_results.h5")
    p.add_argument("--label", default="Run 1",
                   help="Legend label for primary run")
    p.add_argument("--input2", type=Path, default=None,
                   help="Secondary dcr_results.h5 for comparison")
    p.add_argument("--label2", default="Run 2",
                   help="Legend label for secondary run")
    p.add_argument("--atomic", default=DEFAULT_ATOMIC, type=Path,
                   help="Path to H-atom data file (e.g. atomic_data/sclit.01)")
    p.add_argument("--lw-rates", type=Path, default=None,
                   help="Path to lw_rates.npz (from precompute_lw_rates.py); "
                        "enables H2 electronic excitation (B/B'/C/D singlet) RPL")
    p.add_argument("--no-lw", action="store_true",
                   help="Disable molecular LW contribution even if the default table exists")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output directory (default: directory of --input)")
    p.add_argument("--x-unit", choices=["cm", "m"], default="cm")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--font-scale", type=float, default=1.4)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.paths:
        if len(args.paths) > 2:
            raise SystemExit("Pass at most two HDF5 paths")
        if args.input != parser.get_default("input"):
            raise SystemExit("Use either positional paths or --input/--input2, not both")
        if args.input2 is not None:
            raise SystemExit("Use either positional paths or --input/--input2, not both")
        args.input = args.paths[0]
        args.input2 = args.paths[1] if len(args.paths) == 2 else None

    if args.input is None:
        raise SystemExit("Provide one or two HDF5 paths")
    if args.atomic == parser.get_default("atomic"):
        args.atomic = DEFAULT_ATOMIC
    if args.lw_rates is None and not args.no_lw and DEFAULT_LW.exists():
        args.lw_rates = DEFAULT_LW
    if args.label == parser.get_default("label"):
        args.label = default_label_for_path(args.input)
    if args.input2 is not None and args.label2 == parser.get_default("label2"):
        args.label2 = default_label_for_path(args.input2)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.atomic.exists():
        raise SystemExit(f"Atomic data not found: {args.atomic}")

    configure_style(args.dpi, args.font_scale)
    atom_data = HAtomData(args.atomic)
    print(f"[compute_rpl] Loaded {len(atom_data.levels)} levels, "
          f"{len(atom_data.transitions)} transitions from {args.atomic.name}")

    outdir = args.outdir if args.outdir else args.input.parent / "plots_journal"
    outdir.mkdir(parents=True, exist_ok=True)

    lw_path = None if args.no_lw else args.lw_rates
    if lw_path is not None and not lw_path.exists():
        raise SystemExit(f"LW rates not found: {lw_path}")

    def load_and_compute(path, label):
        x, bg, flow_a, flow_m, a_idx, m_idx, qss_flow, qss_flow_idx, tid, chg, iid, atom, Te, ne = load_dcr(path)
        p_total, p_bg, p_flow = compute_rpl_components(
            bg, flow_a, a_idx, qss_flow, qss_flow_idx, tid, chg, iid, atom_data
        )
        p_lw = p_lw_bg = p_lw_flow = np.zeros(bg.shape[0])
        if lw_path is not None:
            p_lw, p_lw_bg, p_lw_flow = compute_lw_rpl(
                bg, flow_m, m_idx, chg, atom, iid, Te, ne, lw_path)
            print(f"[compute_rpl] {label}: max P_LW (bg)   = {p_lw_bg.max():.3e} W/cm^3")
            print(f"[compute_rpl] {label}: max P_LW (flow) = {p_lw_flow.max():.3e} W/cm^3")
            p_total = p_total + p_lw
        print(f"[compute_rpl] {label}: max P_rad = {p_total.max():.3e} W/cm^3")
        return x, p_total, p_bg, p_flow, p_lw_bg, p_lw_flow

    x1, p1, p1_bg, p1_flow, p1_lw_bg, p1_lw_flow = load_and_compute(args.input, args.label)
    component_results = [(x1, p1, p1_bg, p1_flow, p1_lw_bg, p1_lw_flow, args.label)]

    if args.input2 is not None:
        if not args.input2.exists():
            raise SystemExit(f"Input2 not found: {args.input2}")
        x2, p2, p2_bg, p2_flow, p2_lw_bg, p2_lw_flow = load_and_compute(args.input2, args.label2)
        component_results.append((x2, p2, p2_bg, p2_flow, p2_lw_bg, p2_lw_flow, args.label2))

    outpath = outdir / "rpl_vs_x.pdf"
    plot_rpl_components(component_results, outpath, x_unit=args.x_unit)
    if lw_path is not None:
        outpath = outdir / "rpl_atomic_vs_molecular.pdf"
        plot_atomic_vs_molecular(component_results, outpath, x_unit=args.x_unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
