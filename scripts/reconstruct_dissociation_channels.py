#!/usr/bin/env python3
"""
Reconstruct channel-resolved H2 dissociation cross sections from the local data.

This script implements the reduced model the user requested:

1. Singlet dissociation is reconstructed from the local MCCC excitation fits and
   the stored decay-reference tables.
2. Triplet dissociation is taken directly from every *_DE file available under
   atomic_data/mccc-dissociation for the requested vibrational level vi.
3. The final output is the total reconstructed dissociation cross section and
   the pointwise channel ratios

       R_c(E) = sigma_c(E) / sum_c sigma_c(E).

Important scope note:
  - This script intentionally does NOT add the Figure-2 "remaining bound
    triplet spectrum" from the 2019 paper.
  - It only uses channels that are already explicit dissociation channels in the
    local data, because that is the model the user requested in this turn.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise SystemExit("Missing matplotlib: pip install matplotlib") from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dissociation_channels import (  # pylint: disable=import-error
    A0_CM2,
    DissociationChannelError,
    EXPLICIT_STATES,
    REMAINDER_STATES,
    compute_effective_probabilities,
    discover_fit_files,
    evaluate_fit_row,
    load_reference_tables,
    parse_state_fit_file,
)


DEFAULT_MCCC_DIR = Path("atomic_data/mccc-dissociation")
DEFAULT_FITS_DIR = Path("atomic_data/vibex_channel/EIE_Xsec2RC/X1Sg/fits")
DEFAULT_TABLES_DIR = Path("atomic_data/dissociation_channels")
DEFAULT_OUTDIR = Path("output/reconstructed_dissociation_channels")


@dataclass
class ReconstructionResult:
    """Container for one vi reconstruction result."""

    vi: int
    energy_eV: np.ndarray
    channel_sigma_cm2: dict[str, np.ndarray]
    channel_ratio: dict[str, np.ndarray]
    singlet_total_cm2: np.ndarray
    triplet_total_cm2: np.ndarray
    reconstructed_total_cm2: np.ndarray
    benchmark_total_cm2: np.ndarray | None
    f_eff_raw: np.ndarray
    f_eff_applied: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, nargs="*", help="Initial vibrational levels to process")
    parser.add_argument("--all", action="store_true", help="Process all v_i = 0..14")
    parser.add_argument("--mccc-dir", type=Path, default=DEFAULT_MCCC_DIR, help="Directory with MCCC dissociation files")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Directory with MCCC excitation fits")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Directory with stored decay-reference tables")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip PDF plot generation")
    parser.add_argument("--with-csv", action="store_true", help="Also write CSV outputs in addition to the PDFs")
    return parser.parse_args()


def determine_vi_list(args: argparse.Namespace) -> list[int]:
    """Resolve the vibrational levels to process."""
    if args.all:
        return list(range(15))
    if args.vi:
        return sorted(set(args.vi))
    return [0]


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """
    Divide arrays elementwise while keeping undefined points as NaN.

    For channel ratios this is safer than a raw division because there can be
    low-energy points where the reconstructed total is exactly zero.
    """
    out = np.full_like(num, np.nan, dtype=float)
    mask = den > 0.0
    out[mask] = num[mask] / den[mask]
    return out


def _find_total_dissociation_file(mccc_dir: Path, vi: int) -> Path | None:
    """
    Find the total dissociation benchmark file for one vi.

    The local directory layout is inconsistent:
      - vi=0 has the DISS file at the top level
      - vi>=1 store the DISS files under vi=<n> subdirectories

    This helper hides that detail from the reconstruction logic.
    """
    candidates = [
        mccc_dir / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt",
        mccc_dir / f"vi={vi}" / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_total_dissociation_benchmark(mccc_dir: Path, vi: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load the benchmark total dissociation cross section if it exists.

    The reconstructed total in this script is built only from the channels the
    user asked to include. The benchmark is therefore written for comparison,
    not treated as a closure requirement.
    """
    path = _find_total_dissociation_file(mccc_dir, vi)
    if path is None:
        return None
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise DissociationChannelError(f"Unexpected total dissociation format in {path}")
    return data[:, 0], data[:, 1] * A0_CM2


def _find_triplet_de_files(mccc_dir: Path, vi: int) -> dict[str, Path]:
    """
    Discover all triplet dissociative-excitation files for one vi.

    We do not hardcode the list of triplet states here. The folder already tells
    us which explicit triplet DE channels are available, and this function turns
    that folder contents into a channel map.
    """
    state_to_path: dict[str, Path] = {}
    pattern = re.compile(rf"MCCC-el-H2-(.+)_DE\.X1Sg_vi={vi}\.txt$")
    search_roots = [mccc_dir / f"vi={vi}", mccc_dir]
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(f"MCCC-el-H2-*_DE.X1Sg_vi={vi}.txt")):
            match = pattern.match(path.name)
            if not match:
                continue
            state_to_path.setdefault(match.group(1), path)
    return state_to_path


def _looks_like_html_error(path: Path) -> bool:
    """
    Detect the common failed-download case in the local triplet folder.

    Several files under atomic_data/mccc-dissociation are not numeric tables at
    all; they are HTML error pages saved with a .txt suffix.  NumPy then fails
    with a low-level "could not convert string '<!DOCTYPE'" error, which is not
    useful for the user.  This helper turns that failure mode into a clear,
    physics-aware message and lets the reconstruction continue with the valid
    channels that do exist.
    """
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:512].lstrip()
    except OSError:
        return False
    lowered = head.lower()
    return lowered.startswith("<!doctype") or lowered.startswith("<html")


def _load_numeric_cross_section_table(path: Path) -> np.ndarray | None:
    """
    Load one cross-section table if it is valid numeric content.

    Returns:
      - the loaded numeric array for a valid table
      - None for a known bad HTML placeholder / failed download

    This keeps the reconstruction robust when the folder contains a mixture of
    valid triplet channels and failed downloads.
    """
    if _looks_like_html_error(path):
        return None
    try:
        data = np.loadtxt(path)
    except ValueError:
        # Treat parse failures the same way as the explicit HTML case.  The
        # local triplet directory is known to contain malformed downloads, and
        # the reconstruction should skip those files cleanly.
        return None
    return data


def load_triplet_dissociation_channels(
    mccc_dir: Path,
    vi: int,
    target_energy_eV: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Load and interpolate every explicit triplet dissociation channel onto a
    common target energy grid.

    The interpolation step is necessary because the triplet files and the
    singlet reconstruction do not all share the same energy grid.
    """
    state_to_path = _find_triplet_de_files(mccc_dir, vi)
    if not state_to_path:
        raise DissociationChannelError(
            f"No triplet *_DE files were found in {mccc_dir} for vi={vi}"
        )

    channels: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for state, path in sorted(state_to_path.items()):
        data = _load_numeric_cross_section_table(path)
        if data is None:
            skipped.append(f"{state} ({path})")
            continue
        if data.ndim != 2 or data.shape[1] < 2:
            raise DissociationChannelError(f"Unexpected triplet DE format in {path}")
        energy_eV = data[:, 0]
        sigma_cm2 = data[:, 1] * A0_CM2
        channels[state] = np.interp(target_energy_eV, energy_eV, sigma_cm2, left=0.0, right=0.0)
    if skipped:
        print(
            "Warning: skipped invalid triplet DE files:\n  - " + "\n  - ".join(skipped),
            file=sys.stderr,
        )
    if not channels:
        raise DissociationChannelError(
            f"No valid numeric triplet *_DE files were found in {mccc_dir} for vi={vi}"
        )
    return channels


def _load_singlet_fit_data(fits_dir: Path, vi: int) -> dict[str, object]:
    """
    Load the state-resolved MCCC singlet excitation fits for one vi.

    The local fit database covers more states than the explicit paper set.
    We keep the explicit states for DE+ERDD+PD reconstruction, and we keep the
    remaining singlet states for the effective-remainder bucket.
    """
    fit_files = discover_fit_files(fits_dir, vi)
    needed_states = tuple(EXPLICIT_STATES) + tuple(REMAINDER_STATES)
    fits_by_state = {
        state: parse_state_fit_file(path)
        for state, path in fit_files.items()
        if state in needed_states
    }
    missing_states = [state for state in needed_states if state not in fits_by_state]
    if missing_states:
        raise DissociationChannelError(f"Missing fit files for states: {missing_states}")
    return fits_by_state


def compute_singlet_dissociation_channels(
    vi: int,
    energy_eV: np.ndarray,
    fits_dir: Path,
    tables_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Reconstruct singlet dissociation channels on a common energy grid.

    Channel definitions in this reduced model:
      - Explicit singlet states: each channel is

            sigma_s = sigma_s^DE + sigma_s^ERDD + sigma_s^PD

      - Remaining singlets: each state is kept as a separate effective channel

            sigma_r = F_eff * (sigma_r^DE + sigma_r^bound_exc)

    The effective fraction is computed from the explicit states only:

        F_eff = sigma_explicit_diss / sigma_explicit_excitation

    This mirrors the bookkeeping already used in the existing postprocessor.
    """
    fits_by_state = _load_singlet_fit_data(fits_dir, vi)
    tables = load_reference_tables(tables_dir)
    probabilities, missing_by_state = compute_effective_probabilities(fits_by_state, tables, vi)

    channel_sigma: dict[str, np.ndarray] = {}
    explicit_diss_total = np.zeros_like(energy_eV)
    explicit_exc_total = np.zeros_like(energy_eV)

    # First build the explicit singlet state channels one by one.
    for state in EXPLICIT_STATES:
        sigma_de = evaluate_fit_row(fits_by_state[state].de_row, energy_eV)
        sigma_erdd = np.zeros_like(energy_eV)
        sigma_pd = np.zeros_like(energy_eV)
        sigma_bound_exc = np.zeros_like(energy_eV)

        for upper_v, fit_row in fits_by_state[state].bound_rows.items():
            sigma_exc = evaluate_fit_row(fit_row, energy_eV)
            if upper_v in missing_by_state.get(state, []):
                # Missing upper-level decay data are intentionally omitted.
                # The user already decided those missing high-v rows are not the
                # dominant issue, so we keep the omission explicit here.
                continue
            sigma_bound_exc += sigma_exc
            pd_prob, erdd_prob = probabilities[(state, upper_v)]
            sigma_pd += pd_prob * sigma_exc
            sigma_erdd += erdd_prob * sigma_exc

        channel_sigma[state] = sigma_de + sigma_erdd + sigma_pd
        explicit_diss_total += channel_sigma[state]
        explicit_exc_total += sigma_de + sigma_bound_exc

    # The explicit states define the effective dissociation fraction used for
    # the remaining singlet states.
    f_eff_raw = _safe_divide(explicit_diss_total, explicit_exc_total)
    f_eff_applied = np.where(np.isfinite(f_eff_raw), np.clip(f_eff_raw, 0.0, 1.0), 0.0)

    for state in REMAINDER_STATES:
        sigma_de = evaluate_fit_row(fits_by_state[state].de_row, energy_eV)
        sigma_bound_exc = np.zeros_like(energy_eV)
        for fit_row in fits_by_state[state].bound_rows.values():
            sigma_bound_exc += evaluate_fit_row(fit_row, energy_eV)
        channel_sigma[f"{state}_effective"] = f_eff_applied * (sigma_de + sigma_bound_exc)

    return channel_sigma, f_eff_raw, f_eff_applied


def reconstruct_vi(
    vi: int,
    mccc_dir: Path,
    fits_dir: Path,
    tables_dir: Path,
) -> ReconstructionResult:
    """
    Reconstruct one vi channel set.

    Procedure:
      1. Choose a common energy grid. We prefer the benchmark total DISS grid
         when it exists because it is a convenient reference grid.
      2. Reconstruct the singlet channels on that grid.
      3. Interpolate all explicit triplet DE channels onto that same grid.
      4. Sum all channels to obtain the reconstructed total.
      5. Convert the state-resolved cross sections into channel ratios.
    """
    benchmark = load_total_dissociation_benchmark(mccc_dir, vi)
    if benchmark is not None:
        energy_eV, benchmark_total_cm2 = benchmark
    else:
        # Fallback: if no total benchmark file exists, use the vi-specific b3Su
        # grid. This keeps the reconstruction runnable even without DISS files.
        b_path = _find_triplet_de_files(mccc_dir, vi).get("b3Su")
        if b_path is None:
            raise DissociationChannelError(f"No benchmark DISS file and no b3Su file found for vi={vi}")
        b_data = np.loadtxt(b_path)
        energy_eV = b_data[:, 0]
        benchmark_total_cm2 = None

    singlet_channels, f_eff_raw, f_eff_applied = compute_singlet_dissociation_channels(
        vi, energy_eV, fits_dir, tables_dir
    )
    triplet_channels = load_triplet_dissociation_channels(mccc_dir, vi, energy_eV)

    # Keep an explicit channel map so the output stays easy to inspect.
    channel_sigma: dict[str, np.ndarray] = {}
    for state, sigma in sorted(triplet_channels.items()):
        channel_sigma[state] = sigma
    for state, sigma in singlet_channels.items():
        channel_sigma[state] = sigma

    triplet_total = np.zeros_like(energy_eV)
    for sigma in triplet_channels.values():
        triplet_total += sigma

    singlet_total = np.zeros_like(energy_eV)
    for sigma in singlet_channels.values():
        singlet_total += sigma

    reconstructed_total = singlet_total + triplet_total
    channel_ratio = {
        name: _safe_divide(values, reconstructed_total)
        for name, values in channel_sigma.items()
    }

    return ReconstructionResult(
        vi=vi,
        energy_eV=energy_eV,
        channel_sigma_cm2=channel_sigma,
        channel_ratio=channel_ratio,
        singlet_total_cm2=singlet_total,
        triplet_total_cm2=triplet_total,
        reconstructed_total_cm2=reconstructed_total,
        benchmark_total_cm2=benchmark_total_cm2,
        f_eff_raw=f_eff_raw,
        f_eff_applied=f_eff_applied,
    )


def write_result_csv(result: ReconstructionResult, outpath: Path) -> None:
    """
    Write one CSV with cross sections and ratios for every reconstructed channel.

    The CSV is intended for inspection, plotting, and later reuse in other tools.
    """
    columns: list[tuple[str, np.ndarray]] = [
        ("E_eV", result.energy_eV),
        ("sigma_reconstructed_total_cm2", result.reconstructed_total_cm2),
        ("sigma_triplet_total_cm2", result.triplet_total_cm2),
        ("sigma_singlet_total_cm2", result.singlet_total_cm2),
        ("F_eff_raw", result.f_eff_raw),
        ("F_eff_applied", result.f_eff_applied),
    ]
    if result.benchmark_total_cm2 is not None:
        columns.append(("sigma_mccc_total_cm2", result.benchmark_total_cm2))
        columns.append(
            (
                "reconstructed_over_mccc",
                _safe_divide(result.reconstructed_total_cm2, result.benchmark_total_cm2),
            )
        )

    for name in sorted(result.channel_sigma_cm2):
        columns.append((f"sigma_{name}_cm2", result.channel_sigma_cm2[name]))
    for name in sorted(result.channel_ratio):
        columns.append((f"ratio_{name}", result.channel_ratio[name]))

    header = ",".join(name for name, _ in columns)
    data = np.column_stack([values for _, values in columns])
    np.savetxt(outpath, data, delimiter=",", header=header, comments="")


def plot_result(result: ReconstructionResult, outpath: Path) -> None:
    """
    Make one PDF with:
      - reconstructed channel cross sections
      - channel ratios

    The plot focuses on the total channel contributions only. It does not
    separate DE/ERDD/PD inside one explicit singlet state because the user
    asked for total channels.
    """
    import matplotlib.pyplot as plt

    def _plottable(values: np.ndarray) -> np.ndarray:
        return np.where(values > 0.0, values, np.nan)

    ordered_channels = (
        "a3Sg",
        "b3Su",
        "c3Pu",
        "d3Pu",
        "e3Su",
        "e3Pu",
        "g3Sg",
        "h3Sg",
        "i3Pg",
        "j3Dg",
        "B1Su",
        "Bp1Su",
        "C1Pu",
        "D1Pu",
        "EF1Sg",
        "GK1Sg_effective",
        "H1Sg_effective",
        "I1Pg_effective",
        "J1Dg_effective",
    )
    channel_names = [name for name in ordered_channels if name in result.channel_sigma_cm2]

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 10.0), constrained_layout=True)

    ax = axes[0]
    if result.benchmark_total_cm2 is not None:
        ax.plot(
            result.energy_eV,
            _plottable(result.benchmark_total_cm2),
            color="black",
            lw=2.0,
            label="MCCC total",
        )
    ax.plot(
        result.energy_eV,
        _plottable(result.reconstructed_total_cm2),
        color="#d62728",
        lw=2.0,
        label="Reconstructed total",
    )
    for name in channel_names:
        ax.plot(result.energy_eV, _plottable(result.channel_sigma_cm2[name]), lw=1.3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Incident energy $E$ [eV]")
    ax.set_ylabel(r"$\sigma(E)$ [cm$^2$]")
    ax.set_title(f"Dissociation channels (vi={result.vi})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)

    ax = axes[1]
    for name in channel_names:
        ax.plot(result.energy_eV, _plottable(result.channel_ratio[name]), lw=1.3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Incident energy $E$ [eV]")
    ax.set_ylabel(r"Channel ratio $R_c(E)$")
    ax.set_title(f"Channel ratios (vi={result.vi})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)

    fig.savefig(outpath)
    plt.close(fig)


def write_summary_csv(results: list[ReconstructionResult], outpath: Path) -> None:
    """Write one compact summary row per vi."""
    with outpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "vi",
                "n_energy",
                "triplet_peak_cm2",
                "singlet_peak_cm2",
                "reconstructed_peak_cm2",
                "benchmark_peak_cm2",
                "f_eff_min",
                "f_eff_max",
            ]
        )
        for result in results:
            finite_f = result.f_eff_raw[np.isfinite(result.f_eff_raw)]
            writer.writerow(
                [
                    result.vi,
                    result.energy_eV.size,
                    float(np.nanmax(result.triplet_total_cm2)),
                    float(np.nanmax(result.singlet_total_cm2)),
                    float(np.nanmax(result.reconstructed_total_cm2)),
                    float(np.nanmax(result.benchmark_total_cm2))
                    if result.benchmark_total_cm2 is not None
                    else np.nan,
                    float(np.nanmin(finite_f)) if finite_f.size else np.nan,
                    float(np.nanmax(finite_f)) if finite_f.size else np.nan,
                ]
            )


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    vi_list = determine_vi_list(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    results: list[ReconstructionResult] = []
    try:
        for vi in vi_list:
            result = reconstruct_vi(vi, args.mccc_dir, args.fits_dir, args.tables_dir)
            results.append(result)

            if args.with_csv:
                csv_path = args.outdir / f"dissociation_channel_ratios_vi={vi}.csv"
                write_result_csv(result, csv_path)
                print(f"Wrote: {csv_path}")

            if not args.no_plots:
                pdf_path = args.outdir / f"dissociation_channel_ratios_vi={vi}.pdf"
                plot_result(result, pdf_path)
                print(f"Wrote: {pdf_path}")

        if args.with_csv:
            summary_path = args.outdir / "summary.csv"
            write_summary_csv(results, summary_path)
            print(f"Wrote: {summary_path}")
    except DissociationChannelError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
