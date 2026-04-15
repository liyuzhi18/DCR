#!/usr/bin/env python3
"""
Compare raw MCCC total dissociation cross sections against reconstructed totals.

For each initial vibrational level v_i:

1. Load the raw total MCCC dissociation cross section sigma_MCCC(E).
2. Reconstruct the total dissociation cross section by summing all resolved
   singlet and triplet intermediate channels.
3. Plot both totals and the ratio

       sigma_reconstructed(E) / sigma_MCCC(E)

   as a function of incident electron energy.

This script requires the raw total DISS files. It does not use the fitted
rate-table fallback because the user asked for cross-section comparison.
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
from scripts.reconstruct_dissociation_channels import (  # pylint: disable=import-error
    DEFAULT_FITS_DIR,
    DEFAULT_MCCC_DIR,
    DEFAULT_TABLES_DIR,
    reconstruct_vi,
)


DEFAULT_OUTDIR = Path("output/compare_mccc_reconstructed_dissociation_xsec")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, nargs="*", help="Initial vibrational levels to process")
    parser.add_argument("--all", action="store_true", help="Process all v_i = 0..14")
    parser.add_argument("--mccc-dir", type=Path, default=DEFAULT_MCCC_DIR, help="Directory with raw MCCC dissociation files")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Directory with singlet excitation fits")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Directory with stored decay-reference tables")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--with-csv", action="store_true", help="Also write CSV summaries")
    return parser.parse_args()


def determine_vi_list(args: argparse.Namespace) -> list[int]:
    """Resolve the vibrational levels to process."""
    if args.all:
        return list(range(15))
    if args.vi:
        return sorted(set(args.vi))
    return [0]


def write_summary_csv(rows: list[tuple[int, float, float, float]], outpath: Path) -> None:
    """Write a compact one-row-per-vi comparison summary."""
    with outpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["vi", "max_rel", "min_reconstructed_over_mccc", "max_reconstructed_over_mccc"])
        for row in rows:
            writer.writerow(list(row))


def write_per_vi_csv(
    energy_eV: np.ndarray,
    mccc_sigma_cm2: np.ndarray,
    reconstructed_sigma_cm2: np.ndarray,
    outpath: Path,
) -> None:
    """Write the detailed cross-section comparison for one vi."""
    ratio = np.divide(
        reconstructed_sigma_cm2,
        mccc_sigma_cm2,
        out=np.full_like(mccc_sigma_cm2, np.nan),
        where=mccc_sigma_cm2 > 0.0,
    )
    header = "E_eV,sigma_mccc_cm2,sigma_reconstructed_cm2,reconstructed_over_mccc"
    data = np.column_stack([energy_eV, mccc_sigma_cm2, reconstructed_sigma_cm2, ratio])
    np.savetxt(outpath, data, delimiter=",", header=header, comments="")


def plot_single_vi(
    vi: int,
    energy_eV: np.ndarray,
    mccc_sigma_cm2: np.ndarray,
    reconstructed_sigma_cm2: np.ndarray,
    outpath: Path,
) -> None:
    """Plot one vi cross-section comparison."""
    import matplotlib.pyplot as plt

    ratio = np.divide(
        reconstructed_sigma_cm2,
        mccc_sigma_cm2,
        out=np.full_like(mccc_sigma_cm2, np.nan),
        where=mccc_sigma_cm2 > 0.0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(energy_eV, mccc_sigma_cm2, color="black", lw=2.0, label="MCCC total")
    ax.plot(energy_eV, reconstructed_sigma_cm2, color="#d62728", lw=1.8, ls="--", label="Reconstructed total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Incident energy $E$ [eV]")
    ax.set_ylabel(r"$\sigma_{\mathrm{diss}}(E)$ [cm$^2$]")
    ax.set_title(f"MCCC vs reconstructed dissociation cross section (vi={vi})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(energy_eV, ratio, color="#1f77b4", lw=1.8)
    ax.axhline(1.0, color="black", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Incident energy $E$ [eV]")
    ax.set_ylabel("Reconstructed / MCCC")
    ax.grid(True, which="both", alpha=0.25)

    fig.savefig(outpath)
    plt.close(fig)


def plot_all_vi(
    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    outpath: Path,
) -> None:
    """Multi-panel cross-section comparison over all requested vi."""
    import matplotlib.pyplot as plt

    vi_list = sorted(results)
    n = len(vi_list)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.9 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, vi in zip(axes, vi_list):
        energy_eV, mccc_sigma_cm2, reconstructed_sigma_cm2 = results[vi]
        ratio = np.divide(
            reconstructed_sigma_cm2,
            mccc_sigma_cm2,
            out=np.full_like(mccc_sigma_cm2, np.nan),
            where=mccc_sigma_cm2 > 0.0,
        )
        ax.plot(energy_eV, mccc_sigma_cm2, color="black", lw=1.8, label="MCCC")
        ax.plot(energy_eV, reconstructed_sigma_cm2, color="#d62728", lw=1.5, ls="--", label="Reconstructed")
        ax2 = ax.twinx()
        ax2.plot(energy_eV, ratio, color="#1f77b4", lw=1.1, alpha=0.9)
        ax.set_title(f"vi={vi}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax.set_xlabel(r"$E$ [eV]")
        ax.set_ylabel(r"$\sigma$ [cm$^2$]")
        ax2.set_ylabel("Recon/MCCC")
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

    panel_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_rows: list[tuple[int, float, float, float]] = []

    try:
        for vi in vi_list:
            result = reconstruct_vi(vi, args.mccc_dir, args.fits_dir, args.tables_dir)
            if result.benchmark_total_cm2 is None:
                raise DissociationChannelError(
                    f"Missing raw total MCCC DISS cross section for vi={vi}. "
                    "Cross-section comparison requires the MCCC-el-H2-DISS.X1Sg_vi=<vi>.txt files."
                )

            mccc_sigma_cm2 = result.benchmark_total_cm2
            reconstructed_sigma_cm2 = result.reconstructed_total_cm2
            ratio = np.divide(
                reconstructed_sigma_cm2,
                mccc_sigma_cm2,
                out=np.full_like(mccc_sigma_cm2, np.nan),
                where=mccc_sigma_cm2 > 0.0,
            )
            rel = np.abs(reconstructed_sigma_cm2 - mccc_sigma_cm2) / np.maximum(mccc_sigma_cm2, 1.0e-300)

            panel_data[vi] = (result.energy_eV, mccc_sigma_cm2, reconstructed_sigma_cm2)
            summary_rows.append((vi, float(np.nanmax(rel)), float(np.nanmin(ratio)), float(np.nanmax(ratio))))

            single_pdf = args.outdir / f"compare_mccc_reconstructed_dissociation_xsec_vi={vi}.pdf"
            plot_single_vi(vi, result.energy_eV, mccc_sigma_cm2, reconstructed_sigma_cm2, single_pdf)
            print(f"Wrote: {single_pdf}")

            if args.with_csv:
                single_csv = args.outdir / f"compare_mccc_reconstructed_dissociation_xsec_vi={vi}.csv"
                write_per_vi_csv(result.energy_eV, mccc_sigma_cm2, reconstructed_sigma_cm2, single_csv)
                print(f"Wrote: {single_csv}")

            print(
                f"vi={vi:2d} max_rel={np.nanmax(rel):8.3e} "
                f"min(recon/mccc)={np.nanmin(ratio):8.3e} "
                f"max(recon/mccc)={np.nanmax(ratio):8.3e}"
            )

        all_pdf = args.outdir / "compare_mccc_reconstructed_dissociation_xsec_all.pdf"
        plot_all_vi(panel_data, all_pdf)
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
