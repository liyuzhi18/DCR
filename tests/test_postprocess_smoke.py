#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "postprocess_dcr.py"
    assert script.exists(), f"Missing postprocess script: {script}"

    with tempfile.TemporaryDirectory(prefix="dcr_post_smoke_") as td:
        tdp = Path(td)
        h5_path = tdp / "mini_dcr.h5"
        out_dir = tdp / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        x = np.array([0.0, 1.0e-3, 1.0e-2, 1.0e-1], dtype=float)
        labels = np.array(
            [
                "H_Atom barenucl",
                "H_Atom h_100001",
                "H2_Molecule h_200001",
                "H2_Molecule h2_v0",
                "H2_Molecule h2p_v0",
                "H_Atom h_010002",
            ],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        type_id = np.array([2, 0, 2, 1, 2, 0], dtype=np.int32)
        charge = np.array([1, 0, -1, 0, 1, 0], dtype=np.int32)
        atomicity = np.array([1, 1, 1, 2, 2, 1], dtype=np.int32)
        internal_id = np.array([0, 1, 0, 0, 0, 2], dtype=np.int32)
        p_idx = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        a_idx = np.array([1, 5], dtype=np.int32)
        m_idx = np.array([3], dtype=np.int32)

        bg = np.array(
            [
                [5.0e12, 2.0e12, 1.0e11, 3.0e11, 4.0e11, 0.0],
                [4.6e12, 2.2e12, 1.2e11, 3.5e11, 4.2e11, 0.0],
                [4.0e12, 2.5e12, 1.5e11, 4.0e11, 4.6e11, 0.0],
                [3.6e12, 2.8e12, 1.6e11, 4.4e11, 5.0e11, 0.0],
            ],
            dtype=float,
        )
        flow_a = np.array(
            [
                [8.0e11, 2.0e11],
                [7.5e11, 2.5e11],
                [7.0e11, 3.0e11],
                [6.8e11, 3.2e11],
            ],
            dtype=float,
        )
        flow_m = np.array([[4.0e11], [3.8e11], [3.6e11], [3.4e11]], dtype=float)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("/grid/x_cm", data=x)
            f.create_dataset("/states/labels", data=labels)
            f.create_dataset("/states/type_id", data=type_id)
            f.create_dataset("/states/charge", data=charge)
            f.create_dataset("/states/atomicity", data=atomicity)
            f.create_dataset("/states/internal_id", data=internal_id)
            f.create_dataset("/states/P_indices", data=p_idx)
            f.create_dataset("/states/A_indices", data=a_idx)
            f.create_dataset("/states/M_indices", data=m_idx)
            f.create_dataset("/population/background_full", data=bg)
            f.create_dataset("/population/flowA", data=flow_a)
            f.create_dataset("/population/flowM", data=flow_m)

        cmd = [sys.executable, str(script), "all", "--input", str(h5_path), "--outdir", str(out_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
        assert proc.returncode == 0, "postprocess script failed"

        expected = [
            out_dir / "bg_A_M_groups_vs_x.pdf",
            out_dir / "flow_A_states_vs_x.pdf",
            out_dir / "flow_M_states_vs_x.pdf",
        ]
        for p in expected:
            assert p.exists(), f"Missing output plot: {p}"
            assert p.stat().st_size > 0, f"Empty output plot: {p}"

    print("[PASS] Postprocess smoke checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
