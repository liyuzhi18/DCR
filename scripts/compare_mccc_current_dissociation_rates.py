#!/usr/bin/env python3
"""
Compare H2 dissociation rates from raw MCCC cross sections against the original
published "de" fit form associated with the parameters in atomic_data/sclit_M.01.

The comparison uses the equation shown in the figure (Eq. 128):

    T' = max(Te_eV * 11606 / 1000, 1.0)
    ln k = a1 / T'^a2 + a3 / T'^a4 + a5 / T'^(2 a6)

so

    k = exp(a1 * T'^(-a2) + a3 * T'^(-a4) + a5 * T'^(-2 a6))

Multiple "de" entries for the same H2(v) are summed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc


MCCC_DIR = Path("atomic_data/mccc-dissociation")
MOLECULAR_FILE = Path("atomic_data/sclit_M.01")
OUTDIR = Path("output/mccc_current_dissociation_compare")

EV_ERG = 1.60218e-12
ME_G = 9.10938e-28
A0_CM2 = (5.29177210903e-9) ** 2
TE_GRID = np.logspace(np.log10(0.1), np.log10(20.0), 160)


def maxwellian_rate(E_eV: np.ndarray, sigma_cm2: np.ndarray, Te_eV: float) -> float:
    v = np.sqrt(2.0 * E_eV * EV_ERG / ME_G)
    F = (2.0 / Te_eV) * np.sqrt(E_eV / (np.pi * Te_eV)) * np.exp(-E_eV / Te_eV)
    return np.trapezoid(v * F * sigma_cm2, E_eV)


def load_mccc_rates() -> dict[int, np.ndarray]:
    rates = {}
    for path in sorted(MCCC_DIR.glob("MCCC-el-H2-DISS.X1Sg_vi=*.txt")):
        match = re.search(r"vi=(\d+)", path.name)
        if not match:
            continue
        vi = int(match.group(1))
        data = np.loadtxt(path)
        E_eV = data[:, 0]
        sigma_cm2 = data[:, 1] * A0_CM2
        rates[vi] = np.array([maxwellian_rate(E_eV, sigma_cm2, Te) for Te in TE_GRID])
    return rates


def parse_h2_local_to_vi() -> dict[int, int]:
    mapping = {}
    pattern = re.compile(r"^\s*4\s+(\d+)\s+\d+\s+h2_v(\d+)\b")
    with open(MOLECULAR_FILE) as fh:
        for line in fh:
            match = pattern.match(line)
            if match:
                mapping[int(match.group(1))] = int(match.group(2))
    return mapping


def parse_current_de_params() -> dict[int, list[np.ndarray]]:
    local_to_vi = parse_h2_local_to_vi()
    params_by_vi: dict[int, list[np.ndarray]] = defaultdict(list)
    with open(MOLECULAR_FILE) as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] != "de":
                continue
            if len(parts) < 14:
                continue
            reactant_species = int(parts[2])
            reactant_local = int(parts[3])
            if reactant_species != 4:
                continue
            vi = local_to_vi.get(reactant_local)
            if vi is None:
                continue
            params = np.array([float(x) for x in parts[8:14]], dtype=float)
            params_by_vi[vi].append(params)
    return params_by_vi


def current_de_rate_single(params: np.ndarray, Te_eV: np.ndarray) -> np.ndarray:
    Te_fit = np.maximum(Te_eV / 1000.0 * 11606.0, 1.0)
    exponent = (
        params[0] * np.power(Te_fit, -params[1])
        + params[2] * np.power(Te_fit, -params[3])
        + params[4] * np.power(Te_fit, -2.0 * params[5])
    )
    return np.exp(exponent)


def current_de_rate_total(params_list: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(TE_GRID)
    for params in params_list:
        total += current_de_rate_single(params, TE_GRID)
    return total


def plot_single_vi(vi: int, k_mccc: np.ndarray, k_current: np.ndarray, outdir: Path) -> None:
    outpath = outdir / f"compare_mccc_current_dissociation_rate_vi{vi}.pdf"
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(TE_GRID, k_mccc, color="#1f77b4", label="MCCC integrated")
    ax.plot(TE_GRID, k_current, "--", color="#d62728", label="Janev")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_e$ (eV)")
    ax.set_ylabel(r"$k_{\mathrm{diss}}$ (cm$^3$ s$^{-1}$)")
    ax.set_title(f"H2 Dissociation Rate Comparison (vi={vi})")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend()

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


def plot_all(vi_list: list[int], mccc: dict[int, np.ndarray], current: dict[int, np.ndarray], outdir: Path) -> None:
    n = len(vi_list)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.7 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, vi in zip(axes, vi_list):
        ax.plot(TE_GRID, mccc[vi], color="#1f77b4", label="MCCC")
        ax.plot(TE_GRID, current[vi], "--", color="#d62728", label="Janev")
        ax.set_title(f"vi={vi}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_xlabel(r"$T_e$ (eV)")
        ax.set_ylabel(r"$k$ (cm$^3$ s$^{-1}$)")
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend()
    outpath = outdir / "compare_mccc_current_dissociation_rates_all.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {outpath}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, default=None, help="Single vibrational level to plot")
    parser.add_argument("--outdir", type=Path, default=OUTDIR, help="Output directory")
    args = parser.parse_args()

    mccc = load_mccc_rates()
    current_params = parse_current_de_params()
    common_vi = sorted(set(mccc.keys()) & set(current_params.keys()))
    if not common_vi:
        raise SystemExit("No overlapping vibrational levels between MCCC and current de fits")

    current = {vi: current_de_rate_total(current_params[vi]) for vi in common_vi}
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.vi is not None:
        if args.vi not in current:
            raise SystemExit(f"No current de fit found for vi={args.vi}")
        plot_single_vi(args.vi, mccc[args.vi], current[args.vi], args.outdir)
        rel = np.abs(current[args.vi] - mccc[args.vi]) / np.maximum(mccc[args.vi], 1.0e-300)
        print(
            f"vi={args.vi}: max(Current/MCCC)={np.max(current[args.vi]/np.maximum(mccc[args.vi],1.0e-300)):.3e} "
            f"max_rel={rel.max():.3e}"
        )
    else:
        plot_all(common_vi, mccc, current, args.outdir)
        for vi in common_vi:
            rel = np.abs(current[vi] - mccc[vi]) / np.maximum(mccc[vi], 1.0e-300)
            print(f"vi={vi:2d} max_rel={rel.max():.3e}")

    missing_current = sorted(set(mccc.keys()) - set(current_params.keys()))
    if missing_current:
        print(f"No current de fit for vi={missing_current}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
