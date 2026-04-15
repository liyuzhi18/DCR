#!/usr/bin/env python3
"""
Fit reconstructed H2 dissociation rates to a vibrationally resolved analytic law.

This script takes the reconstructed total dissociation cross section already
built from:

  - explicit triplet dissociation channels in atomic_data/mccc-dissociation
  - reconstructed singlet dissociation channels from local fits + decay tables

and turns it into a Maxwellian rate-coefficient fit that is directly compatible
with the existing DCR dissociation-fit CSV loader.

For each initial vibrational level v_i we:

1. Reconstruct the total dissociation rate coefficient k_vi(T_e)
   from the resolved channels.
2. Choose a physical threshold for that v_i as the minimum threshold among the
   resolved contributing channels.
3. Fit the total rate to

       k_vi(T_e) = exp(-Eth_vi / T_e + sum_{m=0}^N a_{vi,m} [ln T_e]^m)

   where T_e and Eth are in eV.

The output CSV intentionally uses the same column convention as the dormant
MCCC dissociation-fit loader in the C++ code:

    vi, threshold_eV, aN, ..., a0, ...

so it can be consumed by the main code without inventing another runtime format.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dissociation_channels import DissociationChannelError  # pylint: disable=import-error
from scripts.postprocess_dissociation_energy_loss import (  # pylint: disable=import-error
    DEFAULT_FITS_DIR,
    DEFAULT_MCCC_DIR,
    DEFAULT_TABLES_DIR,
    compute_vi_energy_loss,
)


DEFAULT_OUT_CSV = Path("atomic_data/reconstructed_dissociation_rate_fit.csv")
DEFAULT_OUT_NPZ = Path("atomic_data/reconstructed_dissociation_rate_fit.npz")
DEFAULT_DIAG_DIR = Path("output/reconstructed_dissociation_rate_fit")
DEFAULT_TE_GRID = np.logspace(np.log10(0.1), np.log10(20.0), 80)
DEFAULT_POLY_DEGREE = 9


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, nargs="*", help="Initial vibrational levels to fit")
    parser.add_argument("--all", action="store_true", help="Fit all v_i = 0..14")
    parser.add_argument("--mccc-dir", type=Path, default=DEFAULT_MCCC_DIR, help="Directory with triplet dissociation files")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Directory with singlet excitation fits")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Directory with stored decay-reference tables")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Output CSV fit table")
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ, help="Output NPZ fit table")
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_DIR, help="Diagnostic output directory")
    parser.add_argument("--poly-degree", type=int, default=DEFAULT_POLY_DEGREE, help="Degree of the log-polynomial fit")
    parser.add_argument("--Te-min", type=float, default=float(DEFAULT_TE_GRID[0]), help="Minimum fitting temperature in eV")
    parser.add_argument("--Te-max", type=float, default=float(DEFAULT_TE_GRID[-1]), help="Maximum fitting temperature in eV")
    parser.add_argument("--n-Te", type=int, default=DEFAULT_TE_GRID.size, help="Number of fitting temperatures")
    parser.add_argument("--n-energy", type=int, default=4000, help="Number of energy-grid points for Maxwellian integration")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-v_i fit diagnostic plots")
    return parser.parse_args()


def determine_vi_list(args: argparse.Namespace) -> list[int]:
    """Resolve which vibrational levels to process."""
    if args.all:
        return list(range(15))
    if args.vi:
        return sorted(set(args.vi))
    return [0]


def fit_rate_profile(
    threshold_eV: float,
    temperature_eV: np.ndarray,
    rate_cm3_s: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Fit one vibrationally resolved total dissociation rate profile.

    With the threshold factor factored out, the remaining dependence on ln(T_e)
    is smooth and fits well with a polynomial.
    """
    safe_rate = np.maximum(rate_cm3_s, 1.0e-300)
    z = np.log(temperature_eV)
    y = np.log(safe_rate) + threshold_eV / temperature_eV
    return np.polyfit(z, y, degree)


def evaluate_fit(
    threshold_eV: float,
    coeffs: np.ndarray,
    temperature_eV: np.ndarray,
) -> np.ndarray:
    """Evaluate the fitted threshold-aware log-polynomial rate law."""
    return np.exp(np.polyval(coeffs, np.log(temperature_eV)) - threshold_eV / temperature_eV)


def select_threshold_eV(resolved_channels) -> float:
    """
    Choose the fit threshold for one vibrational level.

    We use the minimum threshold among the resolved contributing channels,
    because that is the physical onset of the reconstructed total dissociation.
    """
    active_thresholds = [
        channel.threshold_eV
        for channel in resolved_channels
        if np.nanmax(channel.sigma_cm2) > 0.0
    ]
    if not active_thresholds:
        raise DissociationChannelError("No active resolved dissociation channels were reconstructed")
    return float(min(active_thresholds))


def write_diagnostic_csv(
    vi: int,
    temperature_eV: np.ndarray,
    raw_rate_cm3_s: np.ndarray,
    fit_rate_cm3_s: np.ndarray,
    outpath: Path,
) -> None:
    """Write one per-v_i diagnostic table with the raw and fitted rates."""
    header = "Te_eV,k_reconstructed_cm3_s,k_fit_cm3_s,fit_over_raw"
    data = np.column_stack(
        [
            temperature_eV,
            raw_rate_cm3_s,
            fit_rate_cm3_s,
            np.divide(
                fit_rate_cm3_s,
                raw_rate_cm3_s,
                out=np.full_like(raw_rate_cm3_s, np.nan),
                where=raw_rate_cm3_s > 0.0,
            ),
        ]
    )
    np.savetxt(outpath, data, delimiter=",", header=header, comments="")


def plot_fit_diagnostic(
    vi: int,
    threshold_eV: float,
    temperature_eV: np.ndarray,
    raw_rate_cm3_s: np.ndarray,
    fit_rate_cm3_s: np.ndarray,
    outpath: Path,
) -> None:
    """Make one diagnostic plot showing the reconstructed rate and its fit."""
    import matplotlib.pyplot as plt

    ratio = np.divide(
        fit_rate_cm3_s,
        raw_rate_cm3_s,
        out=np.full_like(raw_rate_cm3_s, np.nan),
        where=raw_rate_cm3_s > 0.0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(temperature_eV, raw_rate_cm3_s, color="black", lw=2.0, label="Reconstructed")
    ax.plot(temperature_eV, fit_rate_cm3_s, color="#d62728", lw=1.8, ls="--", label="Fit")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_e$ [eV]")
    ax.set_ylabel(r"$k_{\mathrm{diss}}$ [cm$^3$/s]")
    ax.set_title(f"Reconstructed dissociation rate fit (vi={vi}, Eth={threshold_eV:.4f} eV)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(temperature_eV, ratio, color="#1f77b4", lw=1.8)
    ax.axhline(1.0, color="black", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_e$ [eV]")
    ax.set_ylabel("Fit / reconstructed")
    ax.grid(True, which="both", alpha=0.25)

    fig.savefig(outpath)
    plt.close(fig)


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    vi_list = determine_vi_list(args)
    temperature_eV = np.logspace(np.log10(args.Te_min), np.log10(args.Te_max), args.n_Te)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.diag_dir.mkdir(parents=True, exist_ok=True)

    thresholds = []
    coeff_rows = []
    max_rel_errors = []
    rms_log_errors = []
    n_channels_used = []

    for vi in vi_list:
        result = compute_vi_energy_loss(
            vi=vi,
            mccc_dir=args.mccc_dir,
            fits_dir=args.fits_dir,
            tables_dir=args.tables_dir,
            temperature_eV=temperature_eV,
            n_energy=args.n_energy,
        )
        threshold_eV = select_threshold_eV(result.resolved_channels)
        coeffs = fit_rate_profile(threshold_eV, temperature_eV, result.total_rate_cm3_s, args.poly_degree)
        fit_rate_cm3_s = evaluate_fit(threshold_eV, coeffs, temperature_eV)

        rel = np.abs(fit_rate_cm3_s - result.total_rate_cm3_s) / np.maximum(result.total_rate_cm3_s, 1.0e-300)
        rms_log = np.sqrt(np.mean((np.log(np.maximum(fit_rate_cm3_s, 1.0e-300)) - np.log(np.maximum(result.total_rate_cm3_s, 1.0e-300))) ** 2))

        thresholds.append(threshold_eV)
        coeff_rows.append(coeffs)
        max_rel_errors.append(float(rel.max()))
        rms_log_errors.append(float(rms_log))
        n_channels_used.append(len(result.resolved_channels))

        diag_csv = args.diag_dir / f"reconstructed_dissociation_rate_fit_vi={vi}.csv"
        write_diagnostic_csv(vi, temperature_eV, result.total_rate_cm3_s, fit_rate_cm3_s, diag_csv)
        if not args.no_plots:
            diag_pdf = args.diag_dir / f"reconstructed_dissociation_rate_fit_vi={vi}.pdf"
            plot_fit_diagnostic(vi, threshold_eV, temperature_eV, result.total_rate_cm3_s, fit_rate_cm3_s, diag_pdf)

        print(
            f"vi={vi:2d} Eth={threshold_eV:8.5f} eV "
            f"max_rel={rel.max():8.3e} rms_log={rms_log:8.3e} "
            f"n_channels={len(result.resolved_channels):3d}"
        )

    thresholds_arr = np.asarray(thresholds, dtype=float)
    coeffs_arr = np.asarray(coeff_rows, dtype=float)
    max_rel_arr = np.asarray(max_rel_errors, dtype=float)
    rms_log_arr = np.asarray(rms_log_errors, dtype=float)
    n_channels_arr = np.asarray(n_channels_used, dtype=int)

    np.savez(
        args.out_npz,
        formula="k(Te)=exp(-Eth/Te + sum_m a_m [ln Te]^m)",
        source="reconstructed total dissociation from local singlet+triplet channels",
        te_grid_eV=temperature_eV,
        threshold_eV=thresholds_arr,
        poly_degree=args.poly_degree,
        coeffs=coeffs_arr,
        max_relative_error=max_rel_arr,
        rms_log_error=rms_log_arr,
        n_resolved_channels=n_channels_arr,
        vi=np.asarray(vi_list, dtype=int),
    )

    header = ["vi", "threshold_eV"] + [f"a{j}" for j in range(args.poly_degree, -1, -1)] + [
        "max_relative_error",
        "rms_log_error",
        "n_resolved_channels",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for vi, Eth, coeffs, max_rel, rms_log, n_used in zip(
            vi_list, thresholds_arr, coeffs_arr, max_rel_arr, rms_log_arr, n_channels_arr
        ):
            writer.writerow([vi, Eth, *coeffs.tolist(), max_rel, rms_log, n_used])

    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_npz}")
    print(f"Diagnostics: {args.diag_dir}")
    if max_rel_arr.size:
        worst_idx = int(np.argmax(max_rel_arr))
        print(
            f"Worst max relative error: {max_rel_arr[worst_idx]:.3e} "
            f"at vi={vi_list[worst_idx]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
