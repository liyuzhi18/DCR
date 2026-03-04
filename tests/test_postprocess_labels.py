#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from scripts.postprocess_dcr import (  # pylint: disable=import-error
        DCRData,
        flow_state_display_label,
        grouped_curves_fraction,
        total_nuclei_profile,
    )

    labels = [
        "H_Atom barenucl",    # H+
        "H_Atom h_100001",    # H(n=1)
        "H2_Molecule h_200001",  # H-
        "H2_Molecule h2_v0",  # H2(v=0)
        "H2_Molecule h2p_v0", # H2+
    ]
    type_id = np.array([2, 0, 2, 1, 2], dtype=int)
    charge = np.array([1, 0, -1, 0, 1], dtype=int)
    atomicity = np.array([1, 1, 1, 2, 2], dtype=int)
    internal_id = np.array([0, 1, 0, 0, 0], dtype=int)

    bg = np.array(
        [
            [50.0, 20.0, 5.0, 10.0, 3.0],
            [45.0, 18.0, 4.0, 12.0, 4.0],
            [42.0, 16.0, 3.0, 15.0, 5.0],
        ],
        dtype=float,
    )
    flow_a = np.array([[8.0], [7.0], [6.0]], dtype=float)   # H(n=1)
    flow_m = np.array([[4.0], [4.0], [4.0]], dtype=float)   # H2(v=0)

    data = DCRData(
        x_cm=np.array([0.0, 1.0e-3, 2.0e-3], dtype=float),
        labels=labels,
        p_indices=np.array([0, 1, 2, 3, 4], dtype=int),
        a_indices=np.array([1], dtype=int),
        m_indices=np.array([3], dtype=int),
        background_full=bg,
        flow_a=flow_a,
        flow_m=flow_m,
        type_id=type_id,
        charge=charge,
        atomicity=atomicity,
        internal_id=internal_id,
    )

    assert flow_state_display_label(data, 1, labels[1], "A") == r"$\mathrm{H}(n=1)$"
    assert flow_state_display_label(data, 3, labels[3], "M") == r"$\mathrm{H}_2(v=0)$"

    curves = grouped_curves_fraction(data)
    total = total_nuclei_profile(data)
    np.testing.assert_allclose(curves["H+"], bg[:, 0] / total, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curves["H-"], bg[:, 2] / total, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curves["H"], bg[:, 1] / total, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curves["H2"], (2.0 * bg[:, 3]) / total, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curves["H2+"], (2.0 * bg[:, 4]) / total, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curves["A(H)"], flow_a[:, 0] / total, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(curves["M(H2)"], (2.0 * flow_m[:, 0]) / total, rtol=1e-12, atol=0.0)

    print("[PASS] Postprocess label/fraction checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
