#!/usr/bin/env python3
"""
Run a reversible length-continuation sequence for DCR.

This helper does not change the default solver behavior. Instead, it generates
temporary stage configs that optionally point `io.restart_profile_h5` at the
previous stage's `dcr_results.h5`. The marching solver then interpolates that
previous profile to build better per-cell initial guesses.

Typical use:
    python3 scripts/run_dcr_continuation.py \
        --base-config config/hydro_am_T2_recon_march.yaml

Default stages are chosen to ease a stiff 1 cm, 100-cell run into place:
    0.05 cm / 20 cells
    0.10 cm / 30 cells
    0.20 cm / 40 cells
    0.50 cm / 70 cells
    1.00 cm / 100 cells
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_STAGES = [
    (0.05, 20),
    (0.10, 30),
    (0.20, 40),
    (0.50, 70),
    (1.00, 100),
]


def parse_stage(text: str) -> tuple[float, int]:
    try:
        length_s, cells_s = text.split(":")
        length_cm = float(length_s)
        num_cells = int(cells_s)
    except Exception as exc:  # pragma: no cover - defensive parse error
        raise argparse.ArgumentTypeError(
            f"Invalid stage '{text}'. Expected format LENGTH_CM:NUM_CELLS."
        ) from exc
    if length_cm <= 0.0:
        raise argparse.ArgumentTypeError("Stage length must be positive.")
    if num_cells < 1:
        raise argparse.ArgumentTypeError("Stage num_cells must be >= 1.")
    return (length_cm, num_cells)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Config is not a mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def ensure_mapping(root: dict, key: str) -> dict:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


def run_stage(
    exe: Path,
    repo_root: Path,
    config_path: Path,
    log_path: Path,
    stream_output: bool,
) -> None:
    env = os.environ.copy()
    env["DCR_CONFIG"] = str(config_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if stream_output:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [str(exe)],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
            returncode = proc.returncode
    else:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [str(exe)],
                cwd=str(repo_root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = proc.returncode

    if returncode != 0:
        raise SystemExit(
            f"Stage failed for config {config_path} with exit code {returncode}. "
            f"See log: {log_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run staged DCR continuation using previous dcr_results.h5 as marching seed."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/hydro_am_T2_recon_march.yaml"),
        help="Base YAML config to clone for each stage.",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path("build/bin/DCR_Main"),
        help="Path to DCR_Main executable.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        type=parse_stage,
        dest="stages",
        help="Stage as LENGTH_CM:NUM_CELLS. Repeat to override defaults.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/continuation_T2_recon"),
        help="Root directory for generated stage configs, logs, and outputs.",
    )
    parser.add_argument(
        "--verbose-stage-configs",
        action="store_true",
        help="Force verbose_logging: true in generated stage configs.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not mirror DCR_Main output to the terminal; write only run.log.",
    )
    args = parser.parse_args()

    base_config = args.base_config.resolve()
    exe = args.exe.resolve()
    if not base_config.exists():
        raise SystemExit(f"Missing base config: {base_config}")
    if not exe.exists():
        raise SystemExit(f"Missing executable: {exe}")
    repo_root = Path(__file__).resolve().parents[1]

    stages = args.stages if args.stages else DEFAULT_STAGES
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    previous_h5: Path | None = None
    for idx, (length_cm, num_cells) in enumerate(stages, start=1):
        cfg = load_yaml(base_config)
        io_cfg = ensure_mapping(cfg, "io")
        grid_cfg = ensure_mapping(cfg, "grid")

        stage_name = f"stage_{idx:02d}_{length_cm:.2f}cm_{num_cells}cells"
        stage_root = output_root / stage_name
        stage_output = stage_root / "run_output"
        stage_config = stage_root / "config.yaml"
        stage_log = stage_root / "run.log"

        io_cfg["output_dir"] = str(stage_output)
        if args.verbose_stage_configs:
            io_cfg["verbose_logging"] = True
        io_cfg["restart_profile_h5"] = str(previous_h5) if previous_h5 else ""
        grid_cfg["length_cm"] = float(length_cm)
        grid_cfg["num_cells"] = int(num_cells)

        dump_yaml(stage_config, cfg)
        print(
            f"[continuation] stage {idx}/{len(stages)}: "
            f"L={length_cm:.2f} cm, cells={num_cells}, "
            f"restart={'none' if previous_h5 is None else previous_h5}"
        )
        run_stage(exe, repo_root, stage_config, stage_log, stream_output=not args.no_stream)

        h5_path = stage_output / "dcr_results.h5"
        if not h5_path.exists():
            raise SystemExit(
                f"Stage completed without dcr_results.h5: {h5_path}. "
                f"See log: {stage_log}"
            )
        previous_h5 = h5_path

    print(f"[continuation] completed all {len(stages)} stages under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
