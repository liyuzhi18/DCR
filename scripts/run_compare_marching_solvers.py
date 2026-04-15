#!/usr/bin/env python3
"""
Run two DCR cases from one base config, differing only in marching solver.

This is intentionally narrow:
- keep the same boundary solver and all other physics/numerics
- write two derived configs
- launch Picard and Newton-Krylov/PTC cases with separate output dirs/logs

Default behavior is parallel execution because that is the main use case for
quick solver comparison on separate CPU cores.
"""

from __future__ import annotations

import argparse
import copy
import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

import yaml


SOLVERS = ("picard", "newton_krylov_ptc")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not load as a YAML mapping.")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def derived_config(base: dict, solver: str, output_root: Path) -> tuple[dict, Path, Path]:
    data = copy.deepcopy(base)
    data.setdefault("numerics", {})
    data.setdefault("io", {})

    stage_dir = output_root / solver
    run_output_dir = stage_dir / "run_output"

    data["numerics"]["marching_solver"] = solver
    data["io"]["output_dir"] = str(run_output_dir)

    # Keep boundary behavior unchanged. Only the Newton marching comparison case
    # gets a hard marching-only cap plus immediate abort on nonconvergence.
    if solver == "newton_krylov_ptc":
        data["numerics"]["marching_max_iterations"] = 50
        data["numerics"]["abort_on_marching_nonconvergence"] = True

    config_path = stage_dir / "config.yaml"
    log_path = stage_dir / "run.log"
    return data, config_path, log_path


def format_elapsed(seconds: float) -> str:
    return f"{seconds:8.3f}s"


def launch_case(repo_root: Path, config_path: Path, log_path: Path) -> tuple[subprocess.Popen, int, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["DCR_CONFIG"] = str(config_path)
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        ["./build/bin/DCR_Main"],
        cwd=repo_root,
        env=env,
        stdout=slave_fd,
        stderr=slave_fd,
        text=True,
        close_fds=True,
    )
    os.close(slave_fd)
    return proc, master_fd, log_handle


def stream_case_output(solver: str,
                       master_fd: int,
                       log_handle,
                       start_time: float,
                       echo: bool = True) -> None:
    buffer = ""
    while True:
        ready, _, _ = select.select([master_fd], [], [], 0.1)
        if not ready:
            break
        chunk = os.read(master_fd, 4096)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            prefix = f"[{solver} +{format_elapsed(time.perf_counter() - start_time)}] "
            rendered = prefix + line
            log_handle.write(rendered + "\n")
            log_handle.flush()
            if echo:
                print(rendered)
    if buffer:
        prefix = f"[{solver} +{format_elapsed(time.perf_counter() - start_time)}] "
        rendered = prefix + buffer
        log_handle.write(rendered + "\n")
        log_handle.flush()
        if echo:
            print(rendered)


def summarize_log(log_path: Path) -> str:
    if not log_path.exists():
        return "no log"
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines):
        if "Marching cell" in line and "converged" in line:
            return line
    for line in reversed(lines):
        if "Boundary converged" in line:
            return line
    if lines:
        return lines[-1]
    return "empty log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Picard and Newton marching cases from one base DCR config."
    )
    parser.add_argument(
        "--base-config",
        default="config/hydro_am_T2_recon.yaml",
        help="Base YAML config to clone.",
    )
    parser.add_argument(
        "--output-root",
        default="output/compare_marching_solvers",
        help="Directory where derived configs and logs will be written.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run the two cases sequentially instead of in parallel.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not mirror solver logs to the terminal; still write per-case run.log files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (repo_root / args.base_config).resolve()
    output_root = (repo_root / args.output_root).resolve()

    if not base_config_path.exists():
        raise SystemExit(f"Base config not found: {base_config_path}")

    base = load_yaml(base_config_path)
    prepared: list[tuple[str, Path, Path]] = []

    for solver in SOLVERS:
        data, config_path, log_path = derived_config(base, solver, output_root)
        dump_yaml(config_path, data)
        prepared.append((solver, config_path, log_path))

    print(f"[compare] base config: {base_config_path}")
    print(f"[compare] output root: {output_root}")
    print(f"[compare] mode: {'serial' if args.serial else 'parallel'}")

    procs: list[tuple[str, Path, subprocess.Popen, int, object, float]] = []

    if args.serial:
        for solver, config_path, log_path in prepared:
            print(f"[compare] starting {solver}")
            start_time = time.perf_counter()
            proc, master_fd, log_handle = launch_case(repo_root, config_path, log_path)
            while proc.poll() is None:
                stream_case_output(solver, master_fd, log_handle, start_time, echo=not args.no_stream)
            stream_case_output(solver, master_fd, log_handle, start_time, echo=not args.no_stream)
            os.close(master_fd)
            log_handle.close()
            rc = proc.wait()
            elapsed = time.perf_counter() - start_time
            print(f"[compare] finished {solver} rc={rc} wall={elapsed:.3f}s")
            print(f"[compare] {solver} summary: {summarize_log(log_path)}")
            if rc != 0:
                return rc
        return 0

    for solver, config_path, log_path in prepared:
        print(f"[compare] starting {solver}")
        start_time = time.perf_counter()
        proc, master_fd, log_handle = launch_case(repo_root, config_path, log_path)
        procs.append((solver, log_path, proc, master_fd, log_handle, start_time))

    exit_code = 0
    active = list(procs)
    while active:
        next_active: list[tuple[str, Path, subprocess.Popen, int, object, float]] = []
        for solver, log_path, proc, master_fd, log_handle, start_time in active:
            stream_case_output(solver, master_fd, log_handle, start_time, echo=not args.no_stream)
            rc = proc.poll()
            if rc is None:
                next_active.append((solver, log_path, proc, master_fd, log_handle, start_time))
                continue
            stream_case_output(solver, master_fd, log_handle, start_time, echo=not args.no_stream)
            os.close(master_fd)
            log_handle.close()
            elapsed = time.perf_counter() - start_time
            print(f"[compare] finished {solver} rc={rc} wall={elapsed:.3f}s")
            print(f"[compare] {solver} summary: {summarize_log(log_path)}")
            if rc != 0 and exit_code == 0:
                exit_code = rc
        active = next_active
        if active:
            time.sleep(0.05)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
