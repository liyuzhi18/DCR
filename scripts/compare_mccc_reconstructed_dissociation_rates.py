#!/usr/bin/env python3
"""
Compare raw MCCC total dissociation rates against reconstructed total rates.

This script answers a specific question:

  "If we integrate the published total MCCC dissociation cross section directly,
   and we also reconstruct the total dissociation by summing all explicit
   singlet and triplet intermediate channels, how different are the resulting
   Maxwellian rate coefficients?"

For each initial vibrational level v_i:

1. Load the raw total MCCC dissociation cross section sigma_DISS(E).
2. Maxwellian-integrate it to get k_MCCC(T_e).
3. Reconstruct the total dissociation from all resolved channels already used in
   the postprocessor and sum them to get k_reconstructed(T_e).
4. Plot both rates versus T_e.

Default behavior is PDF-only, matching the recent output convention requested in
the thread.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import csv

import numpy as np

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dissociation_channels import A0_CM2, DissociationChannelError  # pylint: disable=import-error
from scripts.postprocess_dissociation_energy_loss import (  # pylint: disable=import-error
    DEFAULT_FITS_DIR,
    DEFAULT_MCCC_DIR,
    DEFAULT_TABLES_DIR,
    compute_vi_energy_loss,
)


DEFAULT_OUTDIR = Path("output/compare_mccc_reconstructed_dissociation_rates")
DEFAULT_MCCC_FIT_CSV = Path("atomic_data/mccc_dissociation_rate_fit.csv")
EV_ERG = 1.60218e-12
ME_G = 9.10938e-28


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, nargs="*", help="Initial vibrational levels to process")
    parser.add_argument("--all", action="store_true", help="Process all v_i = 0..14")
    parser.add_argument("--mccc-dir", type=Path, default=DEFAULT_MCCC_DIR, help="Directory with MCCC dissociation files")
    parser.add_argument("--mccc-fit-csv", type=Path, default=DEFAULT_MCCC_FIT_CSV, help="Fallback MCCC total-rate fit CSV if raw DISS files are absent")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Directory with singlet excitation fits")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Directory with stored decay-reference tables")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--Te-min", type=float, default=0.1, help="Minimum electron temperature in eV")
    parser.add_argument("--Te-max", type=float, default=20.0, help="Maximum electron temperature in eV")
    parser.add_argument("--n-Te", type=int, default=120, help="Number of electron-temperature points")
    parser.add_argument("--n-energy", type=int, default=4000, help="Number of energy-grid points for reconstructed-rate integration")
    parser.add_argument("--with-csv", action="store_true", help="Also write CSV summaries")
    return parser.parse_args()


def determine_vi_list(args: argparse.Namespace) -> list[int]:
    """Resolve which vibrational levels to process."""
    if args.all:
        return list(range(15))
    if args.vi:
        return sorted(set(args.vi))
    return [0]


def _find_total_diss_file(mccc_dir: Path, vi: int) -> Path | None:
    """
    Find the total MCCC dissociation file for one vibrational level.

    The local folder layout is inconsistent between vi=0 and vi>=1, so the
    helper checks both known locations.
    """
    candidates = [
        mccc_dir / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt",
        mccc_dir / f"vi={vi}" / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def maxwellian_rate(E_eV: np.ndarray, sigma_cm2: np.ndarray, Te_eV: float) -> float:
    """Integrate one cross section over a Maxwellian EEDF in energy space."""
    v = np.sqrt(2.0 * E_eV * EV_ERG / ME_G)
    F = (2.0 / Te_eV) * np.sqrt(E_eV / (np.pi * Te_eV)) * np.exp(-E_eV / Te_eV)
    return np.trapezoid(v * F * sigma_cm2, E_eV)


def load_mccc_total_rate(
    mccc_dir: Path,
    mccc_fit_csv: Path,
    vi: int,
    temperature_eV: np.ndarray,
) -> np.ndarray:
    """
    Load the MCCC total dissociation rate for one vi.

    Preferred source:
      - raw MCCC DISS cross section, integrated directly

    Fallback source:
      - pre-fitted MCCC total dissociation rate CSV already stored in atomic_data

    The fallback is necessary because the current local mccc-dissociation folder
    contains explicit channel files but no longer contains the total DISS tables.
    """
    path = _find_total_diss_file(mccc_dir, vi)
    if path is not None:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise DissociationChannelError(f"Unexpected total MCCC dissociation format in {path}")
        energy_eV = data[:, 0]
        sigma_cm2 = data[:, 1] * A0_CM2
        return np.array([maxwellian_rate(energy_eV, sigma_cm2, Te) for Te in temperature_eV])

    if not mccc_fit_csv.exists():
        raise DissociationChannelError(
            f"Missing total MCCC dissociation source for vi={vi}: "
            f"no DISS file and no fallback fit CSV at {mccc_fit_csv}"
        )

    with mccc_fit_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["vi"]) != vi:
                continue
            threshold_eV = float(row["threshold_eV"])
            coeff_names = sorted(
                (name for name in row if name.startswith("a")),
                key=lambda name: int(name[1:]),
                reverse=True,
            )
            coeffs = np.array([float(row[name]) for name in coeff_names], dtype=float)
            return np.exp(np.polyval(coeffs, np.log(temperature_eV)) - threshold_eV / temperature_eV)

    raise DissociationChannelError(f"No row for vi={vi} in fallback MCCC fit CSV {mccc_fit_csv}")


def write_summary_csv(rows: list[tuple[int, float, float, float]], outpath: Path) -> None:
    """Write a compact one-row-per-vi fit summary."""
    import csv

    with outpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["vi", "max_rel", "min_reconstructed_over_mccc", "max_reconstructed_over_mccc"])
        for row in rows:
            writer.writerow(list(row))


def write_per_vi_csv(
    temperature_eV: np.ndarray,
    mccc_rate_cm3_s: np.ndarray,
    reconstructed_rate_cm3_s: np.ndarray,
    outpath: Path,
) -> None:
    """Write the detailed comparison for one vi."""
    header = "Te_eV,k_mccc_cm3_s,k_reconstructed_cm3_s,reconstructed_over_mccc"
    ratio = np.divide(
        reconstructed_rate_cm3_s,
        mccc_rate_cm3_s,
        out=np.full_like(mccc_rate_cm3_s, np.nan),
        where=mccc_rate_cm3_s > 0.0,
    )
    data = np.column_stack([temperature_eV, mccc_rate_cm3_s, reconstructed_rate_cm3_s, ratio])
    np.savetxt(outpath, data, delimiter=",", header=header, comments="")


def plot_single_vi(
    vi: int,
    temperature_eV: np.ndarray,
    mccc_rate_cm3_s: np.ndarray,
    reconstructed_rate_cm3_s: np.ndarray,
    outpath: Path,
) -> None:
    """
    Plot one vi comparison:
      - raw MCCC vs reconstructed rate
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0), constrained_layout=True)
    ax.plot(temperature_eV, mccc_rate_cm3_s, color="black", lw=2.0, label="MCCC total")
    ax.plot(temperature_eV, reconstructed_rate_cm3_s, color="#d62728", lw=1.8, ls="--", label="Reconstructed total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_e$ [eV]")
    ax.set_ylabel(r"$k_{\mathrm{diss}}$ [cm$^3$/s]")
    ax.set_title(f"MCCC vs reconstructed dissociation rate (vi={vi})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.savefig(outpath)
    plt.close(fig)


def plot_all_vi(
    vi_list: list[int],
    temperature_eV: np.ndarray,
    mccc_by_vi: dict[int, np.ndarray],
    reconstructed_by_vi: dict[int, np.ndarray],
    outpath: Path,
) -> None:
    """
    Multi-panel summary over all requested vibrational levels.
    """
    import matplotlib.pyplot as plt

    n = len(vi_list)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.9 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, vi in zip(axes, vi_list):
        ax.plot(temperature_eV, mccc_by_vi[vi], color="black", lw=1.8, label="MCCC")
        ax.plot(temperature_eV, reconstructed_by_vi[vi], color="#d62728", lw=1.5, ls="--", label="Reconstructed")
        ax.set_title(f"vi={vi}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$T_e$ [eV]")
        ax.set_ylabel(r"$k_{\mathrm{diss}}$ [cm$^3$/s]")
        ax.grid(True, which="both", alpha=0.20)

    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right")

    fig.savefig(outpath)
    plt.close(fig)


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    vi_list = determine_vi_list(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    temperature_eV = np.logspace(np.log10(args.Te_min), np.log10(args.Te_max), args.n_Te)

    mccc_by_vi: dict[int, np.ndarray] = {}
    reconstructed_by_vi: dict[int, np.ndarray] = {}
    summary_rows: list[tuple[int, float, float, float]] = []

    try:
        for vi in vi_list:
            mccc_rate = load_mccc_total_rate(args.mccc_dir, args.mccc_fit_csv, vi, temperature_eV)
            reconstructed = compute_vi_energy_loss(
                vi=vi,
                mccc_dir=args.mccc_dir,
                fits_dir=args.fits_dir,
                tables_dir=args.tables_dir,
                temperature_eV=temperature_eV,
                n_energy=args.n_energy,
            )

            reconstructed_rate = reconstructed.total_rate_cm3_s
            ratio = np.divide(
                reconstructed_rate,
                mccc_rate,
                out=np.full_like(mccc_rate, np.nan),
                where=mccc_rate > 0.0,
            )
            rel = np.abs(reconstructed_rate - mccc_rate) / np.maximum(mccc_rate, 1.0e-300)

            mccc_by_vi[vi] = mccc_rate
            reconstructed_by_vi[vi] = reconstructed_rate
            summary_rows.append((vi, float(np.nanmax(rel)), float(np.nanmin(ratio)), float(np.nanmax(ratio))))

            single_pdf = args.outdir / f"compare_mccc_reconstructed_dissociation_rate_vi={vi}.pdf"
            plot_single_vi(vi, temperature_eV, mccc_rate, reconstructed_rate, single_pdf)
            print(f"Wrote: {single_pdf}")

            if args.with_csv:
                single_csv = args.outdir / f"compare_mccc_reconstructed_dissociation_rate_vi={vi}.csv"
                write_per_vi_csv(temperature_eV, mccc_rate, reconstructed_rate, single_csv)
                print(f"Wrote: {single_csv}")

            print(
                f"vi={vi:2d} max_rel={np.nanmax(rel):8.3e} "
                f"min(recon/mccc)={np.nanmin(ratio):8.3e} "
                f"max(recon/mccc)={np.nanmax(ratio):8.3e}"
            )

        all_pdf = args.outdir / "compare_mccc_reconstructed_dissociation_rates_all.pdf"
        plot_all_vi(vi_list, temperature_eV, mccc_by_vi, reconstructed_by_vi, all_pdf)
        print(f"Wrote: {all_pdf}")

        if args.with_csv:
            summary_csv = args.outdir / "summary.csv"
            write_summary_csv(summary_rows, summary_csv)
            print(f"Wrote: {summary_csv}")
    except DissociationChannelError as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
