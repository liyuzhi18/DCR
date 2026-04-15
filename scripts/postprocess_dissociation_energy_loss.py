#!/usr/bin/env python3
"""
Postprocess reconstructed H2 dissociation channels into Maxwellian rates and
threshold-based electron-energy-loss coefficients.

This script follows the approximation agreed in the thread:

1. Reconstruct the dissociation cross section for each resolved channel.
2. Use the channel threshold energy as the electron energy loss of that channel.
3. Integrate the channel cross section over a Maxwellian EEDF to obtain a rate
   coefficient k(T_e).
4. Multiply the rate coefficient by the threshold energy to obtain the
   electron-energy-loss coefficient:

       L(T_e) = E_th * k(T_e)

The resulting coefficients have units:
  - k(T_e): cm^3 / s
  - L(T_e): eV cm^3 / s

No densities are needed in this postprocessing step.  If the user later wants
volumetric power loss, it is recovered from:

    P_e(T_e) = n_e * n_H2 * e * L(T_e)

with e the elementary charge in J/eV.
"""

from __future__ import annotations

import argparse
import csv
import math
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
DEFAULT_OUTDIR = Path("output/dissociation_energy_loss")

# Constants for Maxwellian rate integration in energy space.
EV_TO_ERG = 1.0 / 6.2415e11
M_E_G = 9.10938e-28
VELOCITY_CONST = math.sqrt(2.0 * EV_TO_ERG / M_E_G)


@dataclass(frozen=True)
class ResolvedThresholdChannel:
    """
    One resolved dissociation contribution.

    aggregate:
      The total state-level channel name used in the final output, e.g. B1Su or
      b3Su.

    component:
      A more specific label for the resolved contribution, e.g. B1Su_ERDD_v=5.
      These resolved labels are used only in the threshold-inspection file.

    threshold_eV:
      The electron-energy-loss proxy for this resolved contribution.  In this
      script it is always the first-step threshold of the resolved channel.

    sigma_cm2:
      Cross section of this resolved contribution on the common integration grid.
    """

    aggregate: str
    component: str
    threshold_eV: float
    sigma_cm2: np.ndarray


@dataclass
class EnergyLossResult:
    """Output container for one vi."""

    vi: int
    temperature_eV: np.ndarray
    aggregate_rate_cm3_s: dict[str, np.ndarray]
    aggregate_loss_eV_cm3_s: dict[str, np.ndarray]
    total_rate_cm3_s: np.ndarray
    total_loss_eV_cm3_s: np.ndarray
    f_eff_raw: np.ndarray
    f_eff_applied: np.ndarray
    integration_energy_eV: np.ndarray
    resolved_channels: list[ResolvedThresholdChannel]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi", type=int, nargs="*", help="Initial vibrational levels to process")
    parser.add_argument("--all", action="store_true", help="Process all v_i = 0..14")
    parser.add_argument("--mccc-dir", type=Path, default=DEFAULT_MCCC_DIR, help="Directory with triplet dissociation files")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Directory with singlet excitation fits")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR, help="Directory with stored decay-reference tables")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--Te-min", type=float, default=0.2, help="Minimum Maxwellian temperature in eV")
    parser.add_argument("--Te-max", type=float, default=20.0, help="Maximum Maxwellian temperature in eV")
    parser.add_argument("--n-Te", type=int, default=80, help="Number of Maxwellian temperatures")
    parser.add_argument("--n-energy", type=int, default=4000, help="Number of integration-grid points")
    parser.add_argument("--no-plots", action="store_true", help="Skip PDF generation")
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
    """Elementwise divide while leaving undefined points as NaN."""
    out = np.full_like(num, np.nan, dtype=float)
    mask = den > 0.0
    out[mask] = num[mask] / den[mask]
    return out


def _looks_like_html_error(path: Path) -> bool:
    """Detect the failed-download HTML pages saved with a .txt suffix."""
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:512].lstrip()
    except OSError:
        return False
    lowered = head.lower()
    return lowered.startswith("<!doctype") or lowered.startswith("<html")


def _load_numeric_cross_section_table(path: Path) -> np.ndarray | None:
    """
    Load a numeric cross-section table.

    Returns None when the local file is an HTML placeholder rather than a data
    table.  The caller can then decide whether to skip the state or fail.
    """
    if _looks_like_html_error(path):
        return None
    try:
        return np.loadtxt(path)
    except ValueError:
        return None


def _find_total_diss_file(mccc_dir: Path, vi: int) -> Path | None:
    """Find the total neutral-fragment dissociation file if it exists."""
    candidates = [
        mccc_dir / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt",
        mccc_dir / f"vi={vi}" / f"MCCC-el-H2-DISS.X1Sg_vi={vi}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_triplet_de_files(mccc_dir: Path, vi: int) -> dict[str, Path]:
    """Discover every explicit triplet *_DE file for one vi."""
    pattern = re.compile(rf"MCCC-el-H2-(.+)_DE\.X1Sg_vi={vi}\.txt$")
    files: dict[str, Path] = {}
    for root in (mccc_dir / f"vi={vi}", mccc_dir):
        if not root.exists():
            continue
        for path in sorted(root.glob(f"MCCC-el-H2-*_DE.X1Sg_vi={vi}.txt")):
            match = pattern.match(path.name)
            if match:
                files.setdefault(match.group(1), path)
    return files


def _parse_threshold_from_header(path: Path) -> float:
    """
    Read the threshold energy from a text-file header.

    The local triplet files store a line like:

        # Threshold: 4.47713E+00 eV
    """
    threshold_re = re.compile(r"Threshold:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*eV")
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = threshold_re.search(line)
            if match:
                return float(match.group(1))
    raise DissociationChannelError(f"Could not parse threshold from {path}")


def _maxwellian_energy_pdf(energy_eV: np.ndarray, temperature_eV: float) -> np.ndarray:
    """
    Maxwellian electron energy distribution in energy space.

    This is the same convention used in the older local helper code:

        f_M(E; T_e) = (2/T_e) * sqrt(E / (pi T_e)) * exp(-E/T_e)

    and it satisfies:

        integral_0^inf f_M(E; T_e) dE = 1
    """
    return (2.0 / temperature_eV) * np.sqrt(energy_eV / (np.pi * temperature_eV)) * np.exp(
        -energy_eV / temperature_eV
    )


def _electron_speed_cm_s(energy_eV: np.ndarray) -> np.ndarray:
    """Electron speed corresponding to kinetic energy E in eV."""
    return VELOCITY_CONST * np.sqrt(energy_eV)


def _build_integration_grid(
    vi: int,
    mccc_dir: Path,
    fits_dir: Path,
    n_energy: int,
) -> np.ndarray:
    """
    Build a dense common energy grid for all integrations.

    Why we do this:
      - triplet DE channels are tabulated numerically on their own grids
      - singlet channels are analytic fits and can be evaluated anywhere
      - a dense common grid makes the subsequent Maxwellian integration simple

    The grid spans the smallest available threshold up to the largest available
    energy found in the local dissociation data.
    """
    fit_files = discover_fit_files(fits_dir, vi)
    fit_thresholds = []
    for path in fit_files.values():
        fit_data = parse_state_fit_file(path)
        fit_thresholds.append(fit_data.de_row.threshold_eV)
        fit_thresholds.extend(row.threshold_eV for row in fit_data.bound_rows.values())

    triplet_paths = _find_triplet_de_files(mccc_dir, vi)
    triplet_thresholds = []
    triplet_max_energies = []
    for path in triplet_paths.values():
        data = _load_numeric_cross_section_table(path)
        if data is None:
            continue
        triplet_thresholds.append(_parse_threshold_from_header(path))
        triplet_max_energies.append(float(data[-1, 0]))

    diss_path = _find_total_diss_file(mccc_dir, vi)
    benchmark_max = None
    if diss_path is not None:
        data = np.loadtxt(diss_path)
        benchmark_max = float(data[-1, 0])

    thresholds = fit_thresholds + triplet_thresholds
    if not thresholds:
        raise DissociationChannelError(f"No thresholds found for vi={vi}")

    emin = min(thresholds)
    emax = max(triplet_max_energies + ([benchmark_max] if benchmark_max is not None else []))
    if emax <= emin:
        emax = emin * 5.0

    dense = np.geomspace(max(emin, 1.0e-6), emax, n_energy)
    # Include exact threshold points so the onset of each channel is represented
    # explicitly in the integration grid.
    grid = np.unique(np.concatenate([dense, np.asarray(thresholds, dtype=float)]))
    return grid


def _load_singlet_fit_data(fits_dir: Path, vi: int) -> dict[str, object]:
    """Load all singlet fit files needed by the local reconstruction."""
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


def build_resolved_singlet_channels(
    vi: int,
    energy_eV: np.ndarray,
    fits_dir: Path,
    tables_dir: Path,
) -> tuple[list[ResolvedThresholdChannel], np.ndarray, np.ndarray]:
    """
    Build the resolved singlet dissociation contributions.

    This function mirrors the current singlet reconstruction logic, but it keeps
    the resolved subchannels separate so the threshold-based energy loss can be
    applied correctly.

    Explicit states:
      - one DE subchannel with the DE threshold
      - one ERDD subchannel per bound upper level v'
      - one PD subchannel per bound upper level v'

    Remaining states:
      - the same state-level effective dissociation fraction is applied to
        each resolved DE or bound-excitation subchannel
      - each resolved piece keeps its own threshold
    """
    fits_by_state = _load_singlet_fit_data(fits_dir, vi)
    tables = load_reference_tables(tables_dir)
    probabilities, missing_by_state = compute_effective_probabilities(fits_by_state, tables, vi)

    resolved: list[ResolvedThresholdChannel] = []
    explicit_diss_total = np.zeros_like(energy_eV)
    explicit_exc_total = np.zeros_like(energy_eV)

    # Build explicit singlet states first so we can compute F_eff from them.
    for state in EXPLICIT_STATES:
        de_row = fits_by_state[state].de_row
        sigma_de = evaluate_fit_row(de_row, energy_eV)
        resolved.append(
            ResolvedThresholdChannel(
                aggregate=state,
                component=f"{state}_DE",
                threshold_eV=de_row.threshold_eV,
                sigma_cm2=sigma_de,
            )
        )
        explicit_diss_total += sigma_de
        explicit_exc_total += sigma_de

        for upper_v, fit_row in fits_by_state[state].bound_rows.items():
            sigma_exc = evaluate_fit_row(fit_row, energy_eV)
            if upper_v in missing_by_state.get(state, []):
                continue
            explicit_exc_total += sigma_exc

            pd_prob, erdd_prob = probabilities[(state, upper_v)]
            if erdd_prob > 0.0:
                sigma_erdd = erdd_prob * sigma_exc
                resolved.append(
                    ResolvedThresholdChannel(
                        aggregate=state,
                        component=f"{state}_ERDD_v={upper_v}",
                        threshold_eV=fit_row.threshold_eV,
                        sigma_cm2=sigma_erdd,
                    )
                )
                explicit_diss_total += sigma_erdd
            if pd_prob > 0.0:
                sigma_pd = pd_prob * sigma_exc
                resolved.append(
                    ResolvedThresholdChannel(
                        aggregate=state,
                        component=f"{state}_PD_v={upper_v}",
                        threshold_eV=fit_row.threshold_eV,
                        sigma_cm2=sigma_pd,
                    )
                )
                explicit_diss_total += sigma_pd

    f_eff_raw = _safe_divide(explicit_diss_total, explicit_exc_total)
    f_eff_applied = np.where(np.isfinite(f_eff_raw), np.clip(f_eff_raw, 0.0, 1.0), 0.0)

    # Apply the explicit-state effective dissociation fraction to the remaining
    # singlet states, but keep each resolved threshold separately.
    for state in REMAINDER_STATES:
        de_row = fits_by_state[state].de_row
        sigma_de = f_eff_applied * evaluate_fit_row(de_row, energy_eV)
        resolved.append(
            ResolvedThresholdChannel(
                aggregate=f"{state}_effective",
                component=f"{state}_effective_DE",
                threshold_eV=de_row.threshold_eV,
                sigma_cm2=sigma_de,
            )
        )
        for upper_v, fit_row in fits_by_state[state].bound_rows.items():
            sigma_exc = f_eff_applied * evaluate_fit_row(fit_row, energy_eV)
            resolved.append(
                ResolvedThresholdChannel(
                    aggregate=f"{state}_effective",
                    component=f"{state}_effective_v={upper_v}",
                    threshold_eV=fit_row.threshold_eV,
                    sigma_cm2=sigma_exc,
                )
            )

    return resolved, f_eff_raw, f_eff_applied


def build_resolved_triplet_channels(
    vi: int,
    energy_eV: np.ndarray,
    mccc_dir: Path,
) -> list[ResolvedThresholdChannel]:
    """
    Build the resolved triplet dissociation contributions.

    Every valid local *_DE file is treated as one explicit dissociation channel.
    The threshold is read from the file header and used later as the
    electron-energy-loss proxy for that channel.
    """
    files = _find_triplet_de_files(mccc_dir, vi)
    if not files:
        raise DissociationChannelError(f"No triplet *_DE files found for vi={vi}")

    resolved: list[ResolvedThresholdChannel] = []
    skipped: list[str] = []
    for state, path in sorted(files.items()):
        data = _load_numeric_cross_section_table(path)
        if data is None:
            skipped.append(f"{state} ({path})")
            continue
        if data.ndim != 2 or data.shape[1] < 2:
            raise DissociationChannelError(f"Unexpected triplet DE format in {path}")
        threshold_eV = _parse_threshold_from_header(path)
        sigma_cm2 = np.interp(energy_eV, data[:, 0], data[:, 1] * A0_CM2, left=0.0, right=0.0)
        resolved.append(
            ResolvedThresholdChannel(
                aggregate=state,
                component=state,
                threshold_eV=threshold_eV,
                sigma_cm2=sigma_cm2,
            )
        )
    if skipped:
        print("Warning: skipped invalid triplet files:\n  - " + "\n  - ".join(skipped), file=sys.stderr)
    if not resolved:
        raise DissociationChannelError(f"No valid triplet *_DE files found for vi={vi}")
    return resolved


def integrate_resolved_channels(
    resolved_channels: list[ResolvedThresholdChannel],
    energy_eV: np.ndarray,
    temperature_eV: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Integrate all resolved channels over a Maxwellian energy distribution.

    We integrate in energy space using a dense common grid and the trapezoidal
    rule.  For each resolved channel c:

        k_c(T_e) = integral sigma_c(E) * v(E) * f_M(E; T_e) dE

    and with the threshold-energy approximation:

        L_c(T_e) = E_th,c * k_c(T_e)

    After computing the resolved contributions, we sum them by aggregate channel
    name so the output stays compact.
    """
    speed_cm_s = _electron_speed_cm_s(energy_eV)
    aggregate_rate: dict[str, np.ndarray] = {}
    aggregate_loss: dict[str, np.ndarray] = {}

    for channel in resolved_channels:
        k_vs_te = np.zeros_like(temperature_eV)
        for idx, temp in enumerate(temperature_eV):
            maxwellian = _maxwellian_energy_pdf(energy_eV, temp)
            integrand = channel.sigma_cm2 * speed_cm_s * maxwellian
            k_vs_te[idx] = np.trapezoid(integrand, energy_eV)

        aggregate_rate.setdefault(channel.aggregate, np.zeros_like(temperature_eV))
        aggregate_loss.setdefault(channel.aggregate, np.zeros_like(temperature_eV))
        aggregate_rate[channel.aggregate] += k_vs_te
        aggregate_loss[channel.aggregate] += channel.threshold_eV * k_vs_te

    return aggregate_rate, aggregate_loss


def compute_vi_energy_loss(
    vi: int,
    mccc_dir: Path,
    fits_dir: Path,
    tables_dir: Path,
    temperature_eV: np.ndarray,
    n_energy: int,
) -> EnergyLossResult:
    """
    Full reconstruction for one vi:

      1. Build a common integration-energy grid.
      2. Construct all resolved singlet and triplet channels.
      3. Integrate every resolved channel over the Maxwellian temperature grid.
      4. Sum the resolved contributions into aggregate state channels.
    """
    energy_eV = _build_integration_grid(vi, mccc_dir, fits_dir, n_energy)
    singlet_resolved, f_eff_raw, f_eff_applied = build_resolved_singlet_channels(
        vi, energy_eV, fits_dir, tables_dir
    )
    triplet_resolved = build_resolved_triplet_channels(vi, energy_eV, mccc_dir)
    resolved = triplet_resolved + singlet_resolved

    aggregate_rate, aggregate_loss = integrate_resolved_channels(resolved, energy_eV, temperature_eV)

    total_rate = np.zeros_like(temperature_eV)
    total_loss = np.zeros_like(temperature_eV)
    for values in aggregate_rate.values():
        total_rate += values
    for values in aggregate_loss.values():
        total_loss += values

    return EnergyLossResult(
        vi=vi,
        temperature_eV=temperature_eV,
        aggregate_rate_cm3_s=aggregate_rate,
        aggregate_loss_eV_cm3_s=aggregate_loss,
        total_rate_cm3_s=total_rate,
        total_loss_eV_cm3_s=total_loss,
        f_eff_raw=f_eff_raw,
        f_eff_applied=f_eff_applied,
        integration_energy_eV=energy_eV,
        resolved_channels=resolved,
    )


def write_rates_csv(result: EnergyLossResult, outpath: Path) -> None:
    """
    Write the Maxwellian rates and threshold-based energy-loss coefficients.

    Each row corresponds to one electron temperature.
    """
    columns: list[tuple[str, np.ndarray]] = [
        ("Te_eV", result.temperature_eV),
        ("k_total_cm3_s", result.total_rate_cm3_s),
        ("L_total_eV_cm3_s", result.total_loss_eV_cm3_s),
    ]
    for name in sorted(result.aggregate_rate_cm3_s):
        columns.append((f"k_{name}_cm3_s", result.aggregate_rate_cm3_s[name]))
    for name in sorted(result.aggregate_loss_eV_cm3_s):
        columns.append((f"L_{name}_eV_cm3_s", result.aggregate_loss_eV_cm3_s[name]))
    data = np.column_stack([values for _, values in columns])
    header = ",".join(name for name, _ in columns)
    np.savetxt(outpath, data, delimiter=",", header=header, comments="")


def write_thresholds_csv(result: EnergyLossResult, outpath: Path) -> None:
    """
    Write a transparent list of the resolved thresholds used in the calculation.

    This file makes it easy to audit how the threshold-energy approximation was
    applied to each resolved contribution.
    """
    with outpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["aggregate_channel", "resolved_component", "threshold_eV", "peak_sigma_cm2"])
        for channel in result.resolved_channels:
            writer.writerow(
                [
                    channel.aggregate,
                    channel.component,
                    channel.threshold_eV,
                    float(np.nanmax(channel.sigma_cm2)),
                ]
            )


def plot_result(result: EnergyLossResult, outpath: Path) -> None:
    """
    Plot the aggregate channel rate coefficients and energy-loss coefficients.

    The top panel shows:
      k_c(T_e)

    The bottom panel shows:
      L_c(T_e) = E_th,c * k_c(T_e)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 10.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(result.temperature_eV, result.total_rate_cm3_s, color="black", lw=2.0, label="Total")
    for name, values in sorted(result.aggregate_rate_cm3_s.items()):
        ax.plot(result.temperature_eV, values, lw=1.4, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Electron temperature $T_e$ [eV]")
    ax.set_ylabel(r"$k(T_e)$ [cm$^3$/s]")
    ax.set_title(f"Dissociation rate coefficients (vi={result.vi})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)

    ax = axes[1]
    ax.plot(result.temperature_eV, result.total_loss_eV_cm3_s, color="black", lw=2.0, label="Total")
    for name, values in sorted(result.aggregate_loss_eV_cm3_s.items()):
        ax.plot(result.temperature_eV, values, lw=1.4, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Electron temperature $T_e$ [eV]")
    ax.set_ylabel(r"$L(T_e)$ [eV cm$^3$/s]")
    ax.set_title(f"Threshold-based electron energy loss (vi={result.vi})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)

    fig.savefig(outpath)
    plt.close(fig)


def plot_total_loss_all_vi(results: list[EnergyLossResult], outpath: Path) -> None:
    """
    Plot the total threshold-based electron energy-loss coefficient for all vi.

    This produces the compact comparison the user asked for:

        L_total(Te; vi)

    with one curve per initial vibrational level.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)

    for result in sorted(results, key=lambda item: item.vi):
        ax.plot(
            result.temperature_eV,
            result.total_loss_eV_cm3_s,
            lw=1.8,
            label=fr"$v_i={result.vi}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Electron temperature $T_e$ [eV]")
    ax.set_ylabel(r"$L_{\mathrm{diss,tot}}(T_e)$ [eV cm$^3$/s]")
    ax.set_title(r"Total dissociation electron energy loss for all $v_i$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)

    fig.savefig(outpath)
    plt.close(fig)


def write_summary_csv(results: list[EnergyLossResult], outpath: Path) -> None:
    """Write one compact summary row per vi."""
    with outpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "vi",
                "n_temperature",
                "temperature_min_eV",
                "temperature_max_eV",
                "k_total_peak_cm3_s",
                "L_total_peak_eV_cm3_s",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.vi,
                    result.temperature_eV.size,
                    float(result.temperature_eV[0]),
                    float(result.temperature_eV[-1]),
                    float(np.nanmax(result.total_rate_cm3_s)),
                    float(np.nanmax(result.total_loss_eV_cm3_s)),
                ]
            )


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    vi_list = determine_vi_list(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    temperature_eV = np.geomspace(args.Te_min, args.Te_max, args.n_Te)

    results: list[EnergyLossResult] = []
    try:
        for vi in vi_list:
            result = compute_vi_energy_loss(
                vi=vi,
                mccc_dir=args.mccc_dir,
                fits_dir=args.fits_dir,
                tables_dir=args.tables_dir,
                temperature_eV=temperature_eV,
                n_energy=args.n_energy,
            )
            results.append(result)

            if args.with_csv:
                rates_path = args.outdir / f"dissociation_energy_loss_rates_vi={vi}.csv"
                thresholds_path = args.outdir / f"dissociation_energy_loss_thresholds_vi={vi}.csv"
                write_rates_csv(result, rates_path)
                write_thresholds_csv(result, thresholds_path)
                print(f"Wrote: {rates_path}")
                print(f"Wrote: {thresholds_path}")

            if not args.no_plots:
                pdf_path = args.outdir / f"dissociation_energy_loss_vi={vi}.pdf"
                plot_result(result, pdf_path)
                print(f"Wrote: {pdf_path}")

        if not args.no_plots and results:
            combined_pdf = args.outdir / "dissociation_total_energy_loss_all_vi.pdf"
            plot_total_loss_all_vi(results, combined_pdf)
            print(f"Wrote: {combined_pdf}")

        if args.with_csv:
            summary_path = args.outdir / "summary.csv"
            write_summary_csv(results, summary_path)
            print(f"Wrote: {summary_path}")
    except DissociationChannelError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
