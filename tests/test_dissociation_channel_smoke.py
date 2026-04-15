#!/usr/bin/env python3
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


HEADER = """# Analytic fits to MCCC calculations of electron scattering on molecular hydrogen
# Adiabatic nuclei calculations performed with the spheroidal MCCC(210) model
# Reference: synthetic
# This file: synthetic
#
# Fitting function: |(x-1)/x * (a0^2*log(x)/x + a1/x + a2/x^2 + a3/x^3 + a4/x^4 + a5/x^5)|
#                   where x = energy / threshold_energy
# Yields integrated cross section in atomic units

# vf    vi  threshold (eV)   a0            a1            a2            a3            a4            a5
"""


ALL_STATES = ["B1Su", "Bp1Su", "C1Pu", "D1Pu", "EF1Sg", "GK1Sg", "H1Sg", "I1Pg", "J1Dg"]


def _write_fit_file(path: Path, vi: int, bound_threshold: float, de_threshold: float, scale: float) -> None:
    rows = [
        f"   0 <-  {vi}  {bound_threshold: .5E}    {0.25 * scale: .5E}   {0.05 * scale: .5E}  {-0.01 * scale: .5E}   0.00000E+00   0.00000E+00   0.00000E+00",
        f"  DE <-  {vi}  {de_threshold: .5E}    {0.18 * scale: .5E}   {0.03 * scale: .5E}  {-0.005 * scale: .5E}   0.00000E+00   0.00000E+00   0.00000E+00",
    ]
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "calc_dissociation_channels.py"
    assert script.exists(), f"Missing calculator script: {script}"

    with tempfile.TemporaryDirectory(prefix="diss_channels_") as td:
        root = Path(td)
        mccc_dir = root / "mccc"
        fits_dir = root / "fits" / "X1Sg" / "fits" / "vi=0"
        table_dir = root / "tables"
        outdir = root / "out"
        mccc_dir.mkdir(parents=True, exist_ok=True)
        fits_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        outdir.mkdir(parents=True, exist_ok=True)

        for i, state in enumerate(ALL_STATES, start=1):
            _write_fit_file(fits_dir / f"MCCC-el-H2-{state}.X1Sg_vi=0_fit.txt", 0, 6.0, 6.5, 1.0 + 0.1 * i)

        np.savetxt(
            mccc_dir / "MCCC-el-H2-DISS.X1Sg_vi=0.txt",
            np.array(
                [
                    [6.0, 1.0e-6],
                    [7.0, 2.0e-6],
                    [8.0, 3.0e-6],
                    [10.0, 5.0e-6],
                ]
            ),
            fmt="%.8e",
            header="Energy (eV) CS (a.u.)",
            comments="# ",
        )

        (table_dir / "radiative_bound.csv").write_text(
            "\n".join(
                [
                    "upper_state,upper_v,lower_state,lower_v,A_s_inv",
                    "B1Su,0,X1Sg,0,2.0",
                    "Bp1Su,0,X1Sg,0,3.0",
                    "D1Pu,0,X1Sg,0,1.0",
                    "EF1Sg,0,B1Su,0,1.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (table_dir / "radiative_continuum.csv").write_text(
            "\n".join(
                [
                    "upper_state,upper_v,target_state,A_cont_s_inv",
                    "B1Su,0,X1Sg,1.0",
                    "Bp1Su,0,X1Sg,1.0",
                    "C1Pu,0,X1Sg,2.0",
                    "D1Pu,0,X1Sg,1.0",
                    "EF1Sg,0,X1Sg,1.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (table_dir / "pd_fraction.csv").write_text(
            "\n".join(
                [
                    "state,vi,F_pd",
                    "D1Pu,0,0.25",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (table_dir / "cascade_map.csv").write_text(
            "\n".join(
                [
                    "upper_state,upper_v,lower_state,lower_v_or_continuum",
                    "EF1Sg,0,B1Su,0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            str(script),
            "--vi",
            "0",
            "--mccc-dir",
            str(mccc_dir),
            "--fits-dir",
            str(fits_dir.parent),
            "--tables-dir",
            str(table_dir),
            "--outdir",
            str(outdir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
        assert proc.returncode == 0, "dissociation-channel calculator failed"

        csv_path = outdir / "channel_breakdown_vi=0.csv"
        pdf_path = outdir / "channel_breakdown_vi=0.pdf"
        summary_path = outdir / "summary.csv"
        assert csv_path.exists() and csv_path.stat().st_size > 0
        assert pdf_path.exists() and pdf_path.stat().st_size > 0
        assert summary_path.exists() and summary_path.stat().st_size > 0

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows, "output CSV is empty"
        total = np.array([float(row["sigma_total_mccc_cm2"]) for row in rows])
        norm_sum = np.array([float(row["sigma_norm_sum_cm2"]) for row in rows])
        support = np.array([float(row["renorm_support"]) > 0.5 for row in rows], dtype=bool)
        np.testing.assert_allclose(norm_sum[support], total[support], rtol=1e-12, atol=0.0)
        assert np.any(np.array([float(row["EFFECTIVE_total_raw"]) for row in rows]) > 0.0)

    print("[PASS] Dissociation-channel smoke checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
