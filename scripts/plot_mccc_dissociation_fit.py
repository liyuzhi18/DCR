#!/usr/bin/env python3
"""
Plot one MCCC H2 dissociation channel and its fitted Maxwellian rate.

Left panel:
  raw dissociation cross section sigma(E) from atomic_data/mccc-dissociation

Right panel:
  Maxwellian-integrated rate data k(Te) from the raw cross section
  versus the analytic fit stored in atomic_data/mccc_dissociation_rate_fit.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc


DATA_DIR = Path("atomic_data/mccc-dissociation")
FIT_PATH = Path("atomic_data/mccc_dissociation_rate_fit.npz")
DEFAULT_OUTDIR = Path("output/mccc_dissociation_fit")

EV_ERG = 1.60218e-12
ME_G = 9.10938e-28
A0_CM = 5.29177210903e-9
A0_CM2 = A0_CM * A0_CM


def parse_vi(path: Path) -> int:
    match = re.search(r"vi=(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse vi from {path}")
    return int(match.group(1))


def load_cross_section(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    threshold_ev = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("# Threshold:"):
                threshold_ev = float(line.split()[2].replace("E", "e"))
                break
    if threshold_ev is None:
        raise ValueError(f"Threshold not found in {path}")
    data = np.loadtxt(path)
    return threshold_ev, data[:, 0], data[:, 1] * A0_CM2


def maxwellian_rate(E_eV: np.ndarray, sigma_cm2: np.ndarray, Te_eV: float) -> float:
    v = np.sqrt(2.0 * E_eV * EV_ERG / ME_G)
    F = (2.0 / Te_eV) * np.sqrt(E_eV / (np.pi * Te_eV)) * np.exp(-E_eV / Te_eV)
    return np.trapezoid(v * F * sigma_cm2, E_eV)


def compute_rate_profile(E_eV: np.ndarray, sigma_cm2: np.ndarray, Te_grid: np.ndarray) -> np.ndarray:
    return np.array([maxwellian_rate(E_eV, sigma_cm2, Te) for Te in Te_grid])


def evaluate_fit(threshold_ev: float, coeffs: np.ndarray, Te_eV: np.ndarray) -> np.ndarray:
    return np.exp(np.polyval(coeffs, np.log(Te_eV)) - threshold_ev / Te_eV)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vi", type=int, default=0, help="Initial H2 vibrational level to plot")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    return p


def main() -> int:
    args = build_parser().parse_args()
    path = DATA_DIR / f"MCCC-el-H2-DISS.X1Sg_vi={args.vi}.txt"
    if not path.exists():
        raise SystemExit(f"Cross section file not found: {path}")
    if not FIT_PATH.exists():
        raise SystemExit(f"Fit table not found: {FIT_PATH}. Run scripts/fit_mccc_dissociation.py first.")

    fit = np.load(FIT_PATH, allow_pickle=True)
    coeffs = fit["coeffs"]
    thresholds = fit["threshold_eV"]
    te_grid = fit["te_grid_eV"]

    if args.vi < 0 or args.vi >= coeffs.shape[0]:
        raise SystemExit(f"vi out of range: {args.vi}")

    threshold_ev, E_eV, sigma_cm2 = load_cross_section(path)
    k_data = compute_rate_profile(E_eV, sigma_cm2, te_grid)
    k_fit = evaluate_fit(thresholds[args.vi], coeffs[args.vi], te_grid)
    rel = np.abs(k_fit - k_data) / np.maximum(k_data, 1.0e-300)

    args.outdir.mkdir(parents=True, exist_ok=True)
    outpath = args.outdir / f"mccc_dissociation_fit_vi{args.vi}.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    axes[0].plot(E_eV, np.where(sigma_cm2 > 0.0, sigma_cm2, np.nan), color="#1f77b4")
    axes[0].axvline(threshold_ev, color="k", linestyle=":", linewidth=0.9, label=f"Eth={threshold_ev:.3f} eV")
    axes[0].set_title(f"Raw MCCC Dissociation Cross Section (vi={args.vi})")
    axes[0].set_xlabel("Electron energy (eV)")
    axes[0].set_ylabel(r"$\sigma_{\mathrm{diss}}$ (cm$^2$)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    axes[0].legend()

    axes[1].plot(te_grid, k_data, color="#1f77b4", linestyle="-", label="Integrated data")
    axes[1].plot(te_grid, k_fit, color="#d62728", linestyle="--", label="Analytic fit")
    axes[1].set_title(f"Rate Fit from Same Channel (max rel={rel.max():.3e})")
    axes[1].set_xlabel(r"$T_e$ (eV)")
    axes[1].set_ylabel(r"$k_{\mathrm{diss}}$ (cm$^3$ s$^{-1}$)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[1].grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    axes[1].legend()

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
