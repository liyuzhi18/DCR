#!/usr/bin/env python3
"""
Fit and plot one raw MCCC H2 dissociation cross section sigma(E).

Compares two forms:

1. Log-polynomial threshold form
   sigma(E) = exp( a_t * ln(1 - Eth/E) + sum_m a_m [ln(E/Eth)]^m )

2. MCCC-style threshold/rational form
   sigma(E) = | (1 - Eth/E)^p * ( b0 ln(x)/x + sum_m b_m / x^m ) |, x = E / Eth

with sigma in cm^2 and E, Eth in eV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc


DATA_DIR = Path("atomic_data/mccc-dissociation")
OUTDIR = Path("output/mccc_dissociation_xsec_fit")
A0_CM2 = (5.29177210903e-9) ** 2


def load_cross_section(vi: int):
    path = DATA_DIR / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt"
    if not path.exists():
        raise SystemExit(f"Cross section file not found: {path}")
    data = np.loadtxt(path)
    E_eV = data[:, 0]
    sigma_cm2 = data[:, 1] * A0_CM2
    return path, E_eV, sigma_cm2


def fit_sigma(E_eV: np.ndarray, sigma_cm2: np.ndarray, degree: int):
    Eth = E_eV[0]
    mask = sigma_cm2 > 0.0
    E = E_eV[mask]
    sigma = sigma_cm2[mask]
    x = np.log(E / Eth)
    t = np.log(np.clip(1.0 - Eth / E, 1.0e-14, None))
    design = np.column_stack([t] + [x ** k for k in range(degree + 1)])
    coeffs, *_ = np.linalg.lstsq(design, np.log(sigma), rcond=None)
    sigma_fit = np.exp(design @ coeffs)
    rel = np.abs(sigma_fit - sigma) / sigma
    rms = np.sqrt(np.mean((np.log(sigma_fit) - np.log(sigma)) ** 2))
    return Eth, coeffs, E, sigma, sigma_fit, float(rel.max()), float(rms)


def select_best_fit(E_eV: np.ndarray, sigma_cm2: np.ndarray):
    best = None
    for degree in range(5, 14):
        fit = fit_sigma(E_eV, sigma_cm2, degree)
        max_rel = fit[5]
        rms = fit[6]
        score = (max_rel, rms)
        if best is None or score < best[0]:
            best = (score, degree, fit)
    return best[1], best[2]


def fit_sigma_mccc(
    E_eV: np.ndarray,
    sigma_cm2: np.ndarray,
    n_terms: int,
    threshold_power: float,
    weight_power: float,
):
    Eth = E_eV[0]
    mask = sigma_cm2 > 0.0
    E = E_eV[mask]
    sigma = sigma_cm2[mask]
    x = E / Eth
    threshold = np.clip(1.0 - 1.0 / x, 1.0e-14, None) ** threshold_power
    design = np.column_stack(
        [threshold * np.log(x) / x] +
        [threshold / (x ** k) for k in range(1, n_terms + 1)]
    )
    weights = 1.0 / np.maximum(sigma, 1.0e-300) ** weight_power
    coeffs, *_ = np.linalg.lstsq(design * weights[:, None], sigma * weights, rcond=None)
    sigma_fit = np.abs(design @ coeffs)
    rel = np.abs(sigma_fit - sigma) / sigma
    rms = np.sqrt(np.mean((np.log(np.maximum(sigma_fit, 1.0e-300)) - np.log(sigma)) ** 2))
    return Eth, coeffs, E, sigma, sigma_fit, float(rel.max()), float(rms)


def select_best_mccc(E_eV: np.ndarray, sigma_cm2: np.ndarray):
    best = None
    for n_terms in range(8, 15, 2):
        for weight_power in (0.5, 0.75, 1.0):
            for threshold_power in np.linspace(0.2, 4.0, 77):
                fit = fit_sigma_mccc(E_eV, sigma_cm2, n_terms, threshold_power, weight_power)
                max_rel = fit[5]
                rms = fit[6]
                score = (max_rel, rms)
                if best is None or score < best[0]:
                    best = (score, n_terms, threshold_power, weight_power, fit)
    return best[1], best[2], best[3], best[4]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, default=0, help="Initial H2 vibrational level")
    parser.add_argument("--degree", type=int, default=None, help="Polynomial degree in ln(E/Eth)")
    parser.add_argument("--outdir", type=Path, default=OUTDIR, help="Output directory")
    args = parser.parse_args()

    path, E_eV, sigma_cm2 = load_cross_section(args.vi)
    if args.degree is None:
        degree, fit = select_best_fit(E_eV, sigma_cm2)
    else:
        degree = args.degree
        fit = fit_sigma(E_eV, sigma_cm2, degree)
    Eth, coeffs, E_fit, sigma_data, sigma_fit, max_rel, rms = fit
    mccc_n_terms, mccc_p, mccc_weight, mccc_fit = select_best_mccc(E_eV, sigma_cm2)
    _, mccc_coeffs, E_mccc, sigma_mccc_data, sigma_mccc_fit, mccc_max_rel, mccc_rms = mccc_fit

    args.outdir.mkdir(parents=True, exist_ok=True)
    outpath = args.outdir / f"mccc_dissociation_xsec_fit_vi{args.vi}.pdf"

    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 6.4), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.4]}
    )

    axes[0].plot(E_eV, np.where(sigma_cm2 > 0.0, sigma_cm2, np.nan), "o-",
                 color="#1f77b4", label="MCCC data")
    axes[0].plot(E_fit, sigma_fit, "--", color="#d62728", label=f"Log-poly fit (deg={degree})")
    axes[0].plot(E_mccc, sigma_mccc_fit, "-.", color="#2ca02c",
                 label=f"MCCC-style fit (n={mccc_n_terms}, p={mccc_p:.2f})")
    axes[0].axvline(Eth, color="k", linestyle=":", linewidth=0.9, label=f"Eth={Eth:.4f} eV")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\sigma_{\mathrm{diss}}$ (cm$^2$)")
    axes[0].set_title(f"H2 Dissociation Cross Section Fit (vi={args.vi})")
    axes[0].grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    axes[0].legend()

    rel = np.abs(sigma_fit - sigma_data) / sigma_data
    rel_mccc = np.abs(sigma_mccc_fit - sigma_mccc_data) / sigma_mccc_data
    axes[1].plot(E_fit, rel, "o-", color="#d62728", label=f"Log-poly max={max_rel:.3f}")
    axes[1].plot(E_mccc, rel_mccc, "s-", color="#2ca02c",
                 label=f"MCCC-style max={mccc_max_rel:.3f}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Electron energy (eV)")
    axes[1].set_ylabel(r"$|\Delta \sigma| / \sigma$")
    axes[1].grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[1].grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    axes[1].legend()

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    print(f"file: {path}")
    print("logpoly: sigma(E)=exp(a_t*ln(1-Eth/E)+sum_{m=0}^N a_m[ln(E/Eth)]^m), E>Eth")
    print(f"logpoly: vi={args.vi} degree={degree} Eth={Eth:.8g} eV max_rel={max_rel:.6g} rms_log={rms:.6g}")
    print(f"a_t={coeffs[0]:.16e}")
    for m, coeff in enumerate(coeffs[1:]):
        print(f"a_{m}={coeff:.16e}")
    print("mccc_style: sigma(E)=|(1-Eth/E)^p * (b0*ln(x)/x + sum_m b_m/x^m)|, x=E/Eth")
    print(
        f"mccc_style: vi={args.vi} n_terms={mccc_n_terms} p={mccc_p:.8g} "
        f"weight_power={mccc_weight:.8g} max_rel={mccc_max_rel:.6g} rms_log={mccc_rms:.6g}"
    )
    for m, coeff in enumerate(mccc_coeffs):
        print(f"b_{m}={coeff:.16e}")
    print(f"Wrote: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
