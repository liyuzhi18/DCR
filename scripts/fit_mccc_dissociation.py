#!/usr/bin/env python3
"""
Fit MCCC H2 dissociation data to a threshold-aware analytic rate formula.

The raw files in atomic_data/mccc-dissociation provide electron-impact
dissociation cross sections for H2(X1Sigma_g, vi=0..14). This script:

1. Maxwellian-integrates each cross section to k_diss(Te) [cm^3/s]
2. Fits the rate to

    k_vi(Te) = exp(-Eth_vi / Te + sum_{m=0}^N a_{vi,m} [ln Te]^m)

   where Te and Eth are in eV.

The threshold factor captures the strong low-Te suppression for low-v levels,
while the polynomial in ln(Te) fits the remaining smooth part.
"""

from __future__ import annotations

from pathlib import Path
import csv
import re

import numpy as np


DATA_DIR = Path("atomic_data/mccc-dissociation")
OUT_NPZ = Path("atomic_data/mccc_dissociation_rate_fit.npz")
OUT_CSV = Path("atomic_data/mccc_dissociation_rate_fit.csv")
POLY_DEGREE = 9
TE_GRID = np.logspace(np.log10(0.1), np.log10(20.0), 80)

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


def compute_rate_profile(E_eV: np.ndarray, sigma_cm2: np.ndarray) -> np.ndarray:
    return np.array([maxwellian_rate(E_eV, sigma_cm2, Te) for Te in TE_GRID])


def fit_rate_profile(threshold_ev: float, k_cm3_s: np.ndarray) -> np.ndarray:
    z = np.log(TE_GRID)
    y = np.log(k_cm3_s) + threshold_ev / TE_GRID
    return np.polyfit(z, y, POLY_DEGREE)


def evaluate_fit(threshold_ev: float, coeffs: np.ndarray, Te_eV: np.ndarray) -> np.ndarray:
    return np.exp(np.polyval(coeffs, np.log(Te_eV)) - threshold_ev / Te_eV)


def main() -> int:
    paths = sorted(DATA_DIR.glob("MCCC-el-H2-DISS.X1Sg_vi=*.txt"), key=parse_vi)
    if not paths:
        raise SystemExit(f"No dissociation files found in {DATA_DIR}")

    thresholds = []
    coeffs = []
    max_rel = []
    rms_log = []

    for path in paths:
        vi = parse_vi(path)
        threshold_ev, E_eV, sigma_cm2 = load_cross_section(path)
        k_cm3_s = compute_rate_profile(E_eV, sigma_cm2)
        fit_coeffs = fit_rate_profile(threshold_ev, k_cm3_s)
        k_fit = evaluate_fit(threshold_ev, fit_coeffs, TE_GRID)
        rel = np.abs(k_fit - k_cm3_s) / np.maximum(k_cm3_s, 1.0e-300)
        thresholds.append(threshold_ev)
        coeffs.append(fit_coeffs)
        max_rel.append(float(rel.max()))
        rms_log.append(float(np.sqrt(np.mean((np.log(k_fit) - np.log(k_cm3_s)) ** 2))))
        print(
            f"vi={vi:2d} Eth={threshold_ev:8.5f} eV "
            f"max_rel={rel.max():8.3e} rms_log={rms_log[-1]:8.3e}"
        )

    thresholds = np.asarray(thresholds, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    max_rel = np.asarray(max_rel, dtype=float)
    rms_log = np.asarray(rms_log, dtype=float)

    np.savez(
        OUT_NPZ,
        formula="k(Te)=exp(-Eth/Te + sum_m a_m [ln Te]^m)",
        te_grid_eV=TE_GRID,
        threshold_eV=thresholds,
        poly_degree=POLY_DEGREE,
        coeffs=coeffs,
        max_relative_error=max_rel,
        rms_log_error=rms_log,
    )

    header = ["vi", "threshold_eV"] + [f"a{j}" for j in range(POLY_DEGREE, -1, -1)] + [
        "max_relative_error",
        "rms_log_error",
    ]
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for vi, (Eth, c, mr, rl) in enumerate(zip(thresholds, coeffs, max_rel, rms_log)):
            writer.writerow([vi, Eth, *c.tolist(), mr, rl])

    print(f"Saved: {OUT_NPZ}")
    print(f"Saved: {OUT_CSV}")
    print(f"Worst max relative error: {max_rel.max():.3e} at vi={int(np.argmax(max_rel))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
