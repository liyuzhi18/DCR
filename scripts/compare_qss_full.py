#!/usr/bin/env python3
"""
Compare QSS and full DCR runs from two HDF5 output files.

Outputs:
1) Background species density comparison: H+, H-, H, H2, H2+
2) Total recycling flow density comparison: A(H), M(H2)
3) Local reconstructed excited-state comparison, when available
4) Flow reconstructed excited-state comparison, when available
5) Boundary QSS first-excited relaxation length summary, when available
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib (pip install matplotlib)") from exc

from postprocess_dcr import (
    DCRData,
    classify_state,
    configure_journal_style,
    convert_x,
    load_hdf5,
    positive_x_mask,
    safe_log_series,
    state_atomicity,
)


def grouped_background_density(data: DCRData) -> Dict[str, np.ndarray]:
    curves = {key: np.zeros(data.background_full.shape[0]) for key in ("H+", "H-", "H", "H2", "H2+")}
    for gi in data.p_indices:
        if gi < 0 or gi >= data.background_full.shape[1]:
            continue
        group = classify_state(data, int(gi))
        if group not in curves:
            continue
        curves[group] += np.maximum(data.background_full[:, int(gi)], 0.0) * state_atomicity(data, int(gi))
    return curves


def grouped_flow_density(data: DCRData) -> Dict[str, np.ndarray]:
    curves = {
        "A(H)": np.zeros(data.flow_a.shape[0]),
        "M(H2)": np.zeros(data.flow_m.shape[0]),
    }
    if data.flow_a.size > 0:
        for j, gi in enumerate(data.a_indices):
            if j >= data.flow_a.shape[1]:
                break
            curves["A(H)"] += np.maximum(data.flow_a[:, j], 0.0) * state_atomicity(data, int(gi))
    if data.flow_m.size > 0:
        for j, gi in enumerate(data.m_indices):
            if j >= data.flow_m.shape[1]:
                break
            curves["M(H2)"] += np.maximum(data.flow_m[:, j], 0.0) * state_atomicity(data, int(gi))
    return curves


def save_bg_comparison(full: DCRData, qss: DCRData, x_unit: str, out_path: Path) -> None:
    x_full, xlabel = convert_x(full.x_cm, x_unit)
    x_qss, _ = convert_x(qss.x_cm, x_unit)
    mask_full = positive_x_mask(x_full)
    mask_qss = positive_x_mask(x_qss)
    bg_full = grouped_background_density(full)
    bg_qss = grouped_background_density(qss)
    order = ("H+", "H-", "H", "H2", "H2+")
    colors = {
        "H+": "#1f77b4",
        "H-": "#17becf",
        "H": "#2ca02c",
        "H2": "#ff7f0e",
        "H2+": "#d62728",
    }

    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.2), constrained_layout=True)
    flat_axes = list(axes.flat)
    for idx, key in enumerate(order):
        ax = flat_axes[idx]
        ax.plot(
            x_full[mask_full],
            safe_log_series(bg_full[key][mask_full]),
            color=colors[key],
            linestyle="-",
            label="full",
        )
        ax.plot(
            x_qss[mask_qss],
            safe_log_series(bg_qss[key][mask_qss]),
            color=colors[key],
            linestyle="--",
            label="qss",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Density (cm$^{-3}$ nuclei)")
        ax.legend(loc="best")
    flat_axes[-1].axis("off")
    fig.suptitle("Background Species: Full vs QSS")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_flow_comparison(full: DCRData, qss: DCRData, x_unit: str, out_path: Path) -> None:
    x_full, xlabel = convert_x(full.x_cm, x_unit)
    x_qss, _ = convert_x(qss.x_cm, x_unit)
    mask_full = positive_x_mask(x_full)
    mask_qss = positive_x_mask(x_qss)
    flow_full = grouped_flow_density(full)
    flow_qss = grouped_flow_density(qss)
    colors = {"A(H)": "#4d4d4d", "M(H2)": "#8c564b"}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    for ax, key in zip(axes, ("A(H)", "M(H2)")):
        ax.plot(
            x_full[mask_full],
            safe_log_series(flow_full[key][mask_full]),
            color=colors[key],
            linestyle="-",
            label="full",
        )
        ax.plot(
            x_qss[mask_qss],
            safe_log_series(flow_qss[key][mask_qss]),
            color=colors[key],
            linestyle="--",
            label="qss",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Density (cm$^{-3}$ nuclei)")
        ax.legend(loc="best")
    fig.suptitle("Total Recycling Flow: Full vs QSS")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def extract_background_columns(data: DCRData, indices: Sequence[int]) -> np.ndarray:
    out = np.zeros((data.background_full.shape[0], len(indices)))
    for j, gi in enumerate(indices):
        if 0 <= gi < data.background_full.shape[1]:
            out[:, j] = data.background_full[:, gi]
    return out


def extract_flow_a_columns(data: DCRData, indices: Sequence[int]) -> np.ndarray:
    out = np.zeros((data.flow_a.shape[0], len(indices)))
    col_of = {int(gi): j for j, gi in enumerate(data.a_indices) if j < data.flow_a.shape[1]}
    for j, gi in enumerate(indices):
        col = col_of.get(int(gi))
        if col is not None:
            out[:, j] = data.flow_a[:, col]
    return out


def transient_labels(data: DCRData, indices: Sequence[int]) -> list[str]:
    labels: list[str] = []
    for gi in indices:
        if 0 <= int(gi) < len(data.labels):
            labels.append(data.labels[int(gi)])
        else:
            labels.append(f"state {gi}")
    return labels


def save_transient_comparison(full: DCRData,
                              qss: DCRData,
                              x_unit: str,
                              out_path: Path,
                              *,
                              title: str,
                              labels: Sequence[str],
                              full_values: np.ndarray,
                              qss_values: np.ndarray) -> bool:
    if full_values.size == 0 or qss_values.size == 0 or len(labels) == 0:
        return False
    n_states = min(full_values.shape[1], qss_values.shape[1], len(labels))
    if n_states == 0:
        return False

    x_full, xlabel = convert_x(full.x_cm, x_unit)
    x_qss, _ = convert_x(qss.x_cm, x_unit)
    mask_full = positive_x_mask(x_full)
    mask_qss = positive_x_mask(x_qss)
    if not np.any(mask_full):
        mask_full = np.isfinite(x_full)
    if not np.any(mask_qss):
        mask_qss = np.isfinite(x_qss)
    if not np.any(mask_full) or not np.any(mask_qss):
        return False

    ncols = 2
    nrows = int(math.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 2.6 * nrows), constrained_layout=True)
    flat_axes = list(np.atleast_1d(axes).flat)
    for idx in range(n_states):
        ax = flat_axes[idx]
        ax.plot(
            x_full[mask_full],
            safe_log_series(full_values[mask_full, idx]),
            color="#1f77b4",
            linestyle="-",
            label="full",
        )
        ax.plot(
            x_qss[mask_qss],
            safe_log_series(qss_values[mask_qss, idx]),
            color="#d62728",
            linestyle="--",
            label="qss",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_title(labels[idx])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Density (cm$^{-3}$)")
        ax.legend(loc="best")
    for idx in range(n_states, len(flat_axes)):
        flat_axes[idx].axis("off")
    fig.suptitle(title)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_transient_relative_error_comparison(full: DCRData,
                                             qss: DCRData,
                                             x_unit: str,
                                             out_path: Path,
                                             *,
                                             title: str,
                                             labels: Sequence[str],
                                             full_values: np.ndarray,
                                             qss_values: np.ndarray) -> bool:
    if full_values.size == 0 or qss_values.size == 0 or len(labels) == 0:
        return False
    n_states = min(full_values.shape[1], qss_values.shape[1], len(labels))
    if n_states == 0:
        return False

    x_full, xlabel = convert_x(full.x_cm, x_unit)
    mask_full = positive_x_mask(x_full)
    if not np.any(mask_full):
        mask_full = np.isfinite(x_full)
    if not np.any(mask_full):
        return False

    ncols = 2
    nrows = int(math.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 2.6 * nrows), constrained_layout=True)
    flat_axes = list(np.atleast_1d(axes).flat)
    for idx in range(n_states):
        ax = flat_axes[idx]
        pct = np.maximum(
            np.abs(relative_error(full_values[:, idx], qss_values[:, idx]))[mask_full],
            1.0e-30,
        )
        ax.plot(x_full[mask_full], pct, color="#d62728", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_title(labels[idx])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"|QSS - Full| / |Full|")
    for idx in range(n_states, len(flat_axes)):
        flat_axes[idx].axis("off")
    fig.suptitle(title)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_local_transient_comparison(full: DCRData, qss: DCRData, x_unit: str, out_path: Path) -> bool:
    if qss.qss_local_transient_atomic is None or qss.qss_local_transient_atomic_indices is None:
        return False
    indices = [int(gi) for gi in qss.qss_local_transient_atomic_indices]
    return save_transient_comparison(
        full,
        qss,
        x_unit,
        out_path,
        title="Local Excited Atomic States: Full vs QSS Reconstruction",
        labels=transient_labels(qss, indices),
        full_values=extract_background_columns(full, indices),
        qss_values=np.asarray(qss.qss_local_transient_atomic, dtype=float),
    )


def save_local_transient_relative_error_comparison(full: DCRData,
                                                   qss: DCRData,
                                                   x_unit: str,
                                                   out_path: Path) -> bool:
    if qss.qss_local_transient_atomic is None or qss.qss_local_transient_atomic_indices is None:
        return False
    indices = [int(gi) for gi in qss.qss_local_transient_atomic_indices]
    return save_transient_relative_error_comparison(
        full,
        qss,
        x_unit,
        out_path,
        title="Local Excited Atomic Relative Error: Full vs QSS Reconstruction",
        labels=transient_labels(qss, indices),
        full_values=extract_background_columns(full, indices),
        qss_values=np.asarray(qss.qss_local_transient_atomic, dtype=float),
    )


def save_flow_transient_comparison(full: DCRData, qss: DCRData, x_unit: str, out_path: Path) -> bool:
    if qss.qss_flow_transient_atomic is None or qss.qss_flow_transient_atomic_indices is None:
        return False
    indices = [int(gi) for gi in qss.qss_flow_transient_atomic_indices]
    return save_transient_comparison(
        full,
        qss,
        x_unit,
        out_path,
        title="Excited Atomic Flow States: Full vs QSS Reconstruction",
        labels=transient_labels(qss, indices),
        full_values=extract_flow_a_columns(full, indices),
        qss_values=np.asarray(qss.qss_flow_transient_atomic, dtype=float),
    )


def save_flow_transient_relative_error_comparison(full: DCRData,
                                                  qss: DCRData,
                                                  x_unit: str,
                                                  out_path: Path) -> bool:
    if qss.qss_flow_transient_atomic is None or qss.qss_flow_transient_atomic_indices is None:
        return False
    indices = [int(gi) for gi in qss.qss_flow_transient_atomic_indices]
    return save_transient_relative_error_comparison(
        full,
        qss,
        x_unit,
        out_path,
        title="Excited Atomic Flow Relative Error: Full vs QSS Reconstruction",
        labels=transient_labels(qss, indices),
        full_values=extract_flow_a_columns(full, indices),
        qss_values=np.asarray(qss.qss_flow_transient_atomic, dtype=float),
    )


def relative_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(reference), 1.0e-300)
    return (candidate - reference) / denom


def save_relative_error_comparison(full: DCRData, qss: DCRData, x_unit: str, out_path: Path) -> None:
    x_full, xlabel = convert_x(full.x_cm, x_unit)
    mask_full = positive_x_mask(x_full)
    bg_full = grouped_background_density(full)
    bg_qss = grouped_background_density(qss)
    flow_full = grouped_flow_density(full)
    flow_qss = grouped_flow_density(qss)
    curves = {
        "H+": relative_error(bg_full["H+"], bg_qss["H+"]),
        "H-": relative_error(bg_full["H-"], bg_qss["H-"]),
        "H": relative_error(bg_full["H"], bg_qss["H"]),
        "H2": relative_error(bg_full["H2"], bg_qss["H2"]),
        "H2+": relative_error(bg_full["H2+"], bg_qss["H2+"]),
        "A(H)": relative_error(flow_full["A(H)"], flow_qss["A(H)"]),
        "M(H2)": relative_error(flow_full["M(H2)"], flow_qss["M(H2)"]),
    }
    order = ("H+", "H-", "H", "H2", "H2+", "A(H)", "M(H2)")

    fig, axes = plt.subplots(4, 2, figsize=(9.0, 10.0), constrained_layout=True)
    flat_axes = list(axes.flat)
    for idx, key in enumerate(order):
        ax = flat_axes[idx]
        abs_error = np.maximum(np.abs(curves[key][mask_full]), 1.0e-30)
        ax.plot(x_full[mask_full], abs_error, color="#1f77b4", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"|QSS - Full| / |Full|")
    flat_axes[-1].axis("off")
    fig.suptitle("Absolute Relative Error: QSS vs Full")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_percentage_profile_comparison(full: DCRData, qss: DCRData, x_unit: str, out_path: Path) -> None:
    x_full, xlabel = convert_x(full.x_cm, x_unit)
    mask_full = positive_x_mask(x_full)
    bg_full = grouped_background_density(full)
    bg_qss = grouped_background_density(qss)
    flow_full = grouped_flow_density(full)
    flow_qss = grouped_flow_density(qss)
    curves = {
        "H+": np.abs(relative_error(bg_full["H+"], bg_qss["H+"])),
        "H-": np.abs(relative_error(bg_full["H-"], bg_qss["H-"])),
        "H": np.abs(relative_error(bg_full["H"], bg_qss["H"])),
        "H2": np.abs(relative_error(bg_full["H2"], bg_qss["H2"])),
        "H2+": np.abs(relative_error(bg_full["H2+"], bg_qss["H2+"])),
        "A(H)": np.abs(relative_error(flow_full["A(H)"], flow_qss["A(H)"])),
        "M(H2)": np.abs(relative_error(flow_full["M(H2)"], flow_qss["M(H2)"])),
    }
    order = ("H+", "H-", "H", "H2", "H2+", "A(H)", "M(H2)")

    fig, axes = plt.subplots(4, 2, figsize=(9.0, 10.0), constrained_layout=True)
    flat_axes = list(axes.flat)
    for idx, key in enumerate(order):
        ax = flat_axes[idx]
        pct = np.maximum(curves[key][mask_full], 1.0e-30)
        ax.plot(x_full[mask_full], pct, color="#d62728", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"|QSS - Full| / |Full|")
    flat_axes[-1].axis("off")
    fig.suptitle("Absolute Relative Error vs x: QSS vs Full")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def boundary_value(values: np.ndarray | None) -> float | None:
    if values is None or len(values) == 0:
        return None
    value = float(values[0])
    return value if np.isfinite(value) else None


def write_boundary_relaxation_summary(full: DCRData, qss: DCRData, out_path: Path) -> bool:
    full_length = boundary_value(full.qss_relaxation_length_cm)
    qss_length = boundary_value(qss.qss_relaxation_length_cm)
    full_loss = boundary_value(full.qss_first_excited_loss_frequency_s)
    qss_loss = boundary_value(qss.qss_first_excited_loss_frequency_s)

    full_index = None if full.qss_first_excited_index is None or len(full.qss_first_excited_index) == 0 else int(full.qss_first_excited_index[0])
    qss_index = None if qss.qss_first_excited_index is None or len(qss.qss_first_excited_index) == 0 else int(qss.qss_first_excited_index[0])

    if full_length is None and qss_length is None and full_loss is None and qss_loss is None:
        return False

    lines = [
        "# Boundary QSS first-excited relaxation diagnostic",
        "# columns: case first_excited_index relaxation_length_cm loss_frequency_s",
    ]
    if full_length is not None or full_loss is not None:
        lines.append(
            f"full {full_index if full_index is not None else -1} "
            f"{full_length if full_length is not None else 'nan'} "
            f"{full_loss if full_loss is not None else 'nan'}"
        )
    if qss_length is not None or qss_loss is not None:
        lines.append(
            f"qss {qss_index if qss_index is not None else -1} "
            f"{qss_length if qss_length is not None else 'nan'} "
            f"{qss_loss if qss_loss is not None else 'nan'}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def abs_relative_percent(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(reference), 1.0e-300)
    return 100.0 * np.abs(candidate - reference) / denom


def summarize_series(lines: list[str],
                     section: str,
                     names: Sequence[str],
                     reference: Dict[str, np.ndarray],
                     candidate: Dict[str, np.ndarray]) -> None:
    lines.append(f"[{section}]")
    lines.append("name mean_abs_pct max_abs_pct boundary_abs_pct end_abs_pct")
    for name in names:
        pct = abs_relative_percent(reference[name], candidate[name])
        lines.append(
            f"{name} "
            f"{float(np.mean(pct)):.6g} "
            f"{float(np.max(pct)):.6g} "
            f"{float(pct[0]):.6g} "
            f"{float(pct[-1]):.6g}"
        )
    lines.append("")


def summarize_matrix(lines: list[str],
                     section: str,
                     labels: Sequence[str],
                     reference: np.ndarray,
                     candidate: np.ndarray) -> None:
    if reference.size == 0 or candidate.size == 0 or len(labels) == 0:
        return
    n_states = min(reference.shape[1], candidate.shape[1], len(labels))
    if n_states == 0:
        return
    lines.append(f"[{section}]")
    lines.append("name mean_abs_pct max_abs_pct boundary_abs_pct end_abs_pct")
    for idx in range(n_states):
        pct = abs_relative_percent(reference[:, idx], candidate[:, idx])
        lines.append(
            f"{labels[idx]} "
            f"{float(np.mean(pct)):.6g} "
            f"{float(np.max(pct)):.6g} "
            f"{float(pct[0]):.6g} "
            f"{float(pct[-1]):.6g}"
        )
    lines.append("")


def write_percentage_summary(full: DCRData, qss: DCRData, out_path: Path) -> None:
    bg_full = grouped_background_density(full)
    bg_qss = grouped_background_density(qss)
    flow_full = grouped_flow_density(full)
    flow_qss = grouped_flow_density(qss)

    lines = [
        "# Absolute percentage difference summary",
        "# definition: 100 * |QSS - Full| / |Full|",
        "",
    ]
    summarize_series(lines, "background_species", ("H+", "H-", "H", "H2", "H2+"), bg_full, bg_qss)
    summarize_series(lines, "flow_totals", ("A(H)", "M(H2)"), flow_full, flow_qss)

    if qss.qss_local_transient_atomic is not None and qss.qss_local_transient_atomic_indices is not None:
        local_indices = [int(gi) for gi in qss.qss_local_transient_atomic_indices]
        summarize_matrix(
            lines,
            "local_transient_atomic",
            transient_labels(qss, local_indices),
            extract_background_columns(full, local_indices),
            np.asarray(qss.qss_local_transient_atomic, dtype=float),
        )

    if qss.qss_flow_transient_atomic is not None and qss.qss_flow_transient_atomic_indices is not None:
        flow_indices = [int(gi) for gi in qss.qss_flow_transient_atomic_indices]
        summarize_matrix(
            lines,
            "flow_transient_atomic",
            transient_labels(qss, flow_indices),
            extract_flow_a_columns(full, flow_indices),
            np.asarray(qss.qss_flow_transient_atomic, dtype=float),
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QSS and full DCR HDF5 outputs.")
    parser.add_argument("--full", required=True, type=Path, help="Full DCR dcr_results.h5")
    parser.add_argument("--qss", required=True, type=Path, help="QSS DCR dcr_results.h5")
    parser.add_argument("--output-dir", type=Path, default=Path("output/comparison_qss_full"))
    parser.add_argument("--x-unit", choices=("cm", "m"), default="cm")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--font-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.full.exists():
        raise SystemExit(f"Missing full input: {args.full}")
    if not args.qss.exists():
        raise SystemExit(f"Missing QSS input: {args.qss}")

    configure_journal_style(args.dpi, args.font_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = load_hdf5(args.full)
    qss = load_hdf5(args.qss)

    save_bg_comparison(full, qss, args.x_unit, args.output_dir / "compare_bg_species_full_vs_qss.pdf")
    save_flow_comparison(full, qss, args.x_unit, args.output_dir / "compare_flow_totals_full_vs_qss.pdf")
    save_relative_error_comparison(
        full,
        qss,
        args.x_unit,
        args.output_dir / "compare_relative_error_full_vs_qss.pdf",
    )
    save_percentage_profile_comparison(
        full,
        qss,
        args.x_unit,
        args.output_dir / "compare_percentage_profile_full_vs_qss.pdf",
    )
    wrote_local_nt = save_local_transient_comparison(
        full,
        qss,
        args.x_unit,
        args.output_dir / "compare_local_nt_full_vs_qss.pdf",
    )
    wrote_flow_nt = save_flow_transient_comparison(
        full,
        qss,
        args.x_unit,
        args.output_dir / "compare_flow_nt_full_vs_qss.pdf",
    )
    wrote_local_nt_rel = save_local_transient_relative_error_comparison(
        full,
        qss,
        args.x_unit,
        args.output_dir / "compare_local_nt_relative_error_full_vs_qss.pdf",
    )
    wrote_flow_nt_rel = save_flow_transient_relative_error_comparison(
        full,
        qss,
        args.x_unit,
        args.output_dir / "compare_flow_nt_relative_error_full_vs_qss.pdf",
    )
    wrote_relax = write_boundary_relaxation_summary(
        full,
        qss,
        args.output_dir / "boundary_qss_relaxation_length.txt",
    )
    write_percentage_summary(
        full,
        qss,
        args.output_dir / "percentage_difference_full_vs_qss.txt",
    )

    print(f"Wrote {args.output_dir / 'compare_bg_species_full_vs_qss.pdf'}")
    print(f"Wrote {args.output_dir / 'compare_flow_totals_full_vs_qss.pdf'}")
    print(f"Wrote {args.output_dir / 'compare_relative_error_full_vs_qss.pdf'}")
    print(f"Wrote {args.output_dir / 'compare_percentage_profile_full_vs_qss.pdf'}")
    print(f"Wrote {args.output_dir / 'percentage_difference_full_vs_qss.txt'}")
    if wrote_local_nt:
        print(f"Wrote {args.output_dir / 'compare_local_nt_full_vs_qss.pdf'}")
    else:
        print("Skipped compare_local_nt_full_vs_qss.pdf (QSS transient dataset missing)")
    if wrote_local_nt_rel:
        print(f"Wrote {args.output_dir / 'compare_local_nt_relative_error_full_vs_qss.pdf'}")
    else:
        print("Skipped compare_local_nt_relative_error_full_vs_qss.pdf (QSS transient dataset missing)")
    if wrote_flow_nt:
        print(f"Wrote {args.output_dir / 'compare_flow_nt_full_vs_qss.pdf'}")
    else:
        print("Skipped compare_flow_nt_full_vs_qss.pdf (QSS transient flow dataset missing)")
    if wrote_flow_nt_rel:
        print(f"Wrote {args.output_dir / 'compare_flow_nt_relative_error_full_vs_qss.pdf'}")
    else:
        print("Skipped compare_flow_nt_relative_error_full_vs_qss.pdf (QSS transient flow dataset missing)")
    if wrote_relax:
        print(f"Wrote {args.output_dir / 'boundary_qss_relaxation_length.txt'}")
    else:
        print("Skipped boundary_qss_relaxation_length.txt (dataset missing in both inputs)")


if __name__ == "__main__":
    main()
