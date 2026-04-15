#!/usr/bin/env python3
"""
Compute channel-resolved H2 dissociation cross sections from local MCCC fits and
reference decay tables.

The calculator is postprocessing-only: it does not alter the runtime solver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dissociation_channels import (  # pylint: disable=import-error
    DissociationChannelError,
    available_vi,
    compute_vi_breakdown,
    plot_breakdown,
    write_breakdown_csv,
    write_summary_csv,
    write_template_tables,
)


DEFAULT_MCCC_DIR = Path("atomic_data/mccc-dissociation")
DEFAULT_FITS_DIR = Path("atomic_data/vibex_channel/EIE_Xsec2RC/X1Sg/fits")
DEFAULT_TABLES_DIR = Path("atomic_data/dissociation_channels")
DEFAULT_OUTDIR = Path("output/dissociation_channels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, nargs="*", default=None, help="Specific vi values to process")
    parser.add_argument("--mccc-dir", type=Path, default=DEFAULT_MCCC_DIR, help="Directory with total dissociation cross sections")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Directory with state-resolved excitation fits")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Directory with decay-reference CSV tables")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--summary-name", default="summary.csv", help="Summary CSV filename")
    parser.add_argument("--no-plots", action="store_true", help="Skip PDF generation")
    parser.add_argument("--write-templates", action="store_true", help="Create empty CSV templates and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_templates:
        for path in write_template_tables(args.tables_dir):
            print(f"Wrote template: {path}")
        return 0

    vis = args.vi if args.vi else available_vi(args.mccc_dir)
    if not vis:
        raise SystemExit("No vi values found to process")

    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for vi in vis:
            result = compute_vi_breakdown(vi, args.mccc_dir, args.fits_dir, args.tables_dir)
            results.append(result)
            csv_path = args.outdir / f"channel_breakdown_vi={vi}.csv"
            write_breakdown_csv(result, csv_path)
            print(f"Wrote: {csv_path}")
            if not args.no_plots:
                pdf_path = args.outdir / f"channel_breakdown_vi={vi}.pdf"
                plot_breakdown(result, pdf_path)
                print(f"Wrote: {pdf_path}")
        summary_path = args.outdir / args.summary_name
        write_summary_csv(results, summary_path)
        print(f"Wrote: {summary_path}")
    except DissociationChannelError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

