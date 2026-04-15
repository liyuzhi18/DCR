#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from scripts.dissociation_channels import (  # pylint: disable=import-error
        load_reference_tables,
        write_template_tables,
    )

    with tempfile.TemporaryDirectory(prefix="diss_nonadiabatic_") as td:
        table_dir = Path(td)
        write_template_tables(table_dir)

        (table_dir / "state_decay_summary.csv").write_text(
            "\n".join(
                [
                    "state,upper_v,A_cont_s_inv,A_bound_s_inv,A_tot_s_inv,tau_s,f_cont",
                    "B1Su,0,2.0,8.0,10.0,1.0e-1,0.2",
                    "C1Pu,0,3.0,7.0,10.0,1.0e-1,0.3",
                    "Bp1Su,0,1.0,9.0,10.0,1.0e-1,0.1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        (table_dir / "nonadiabatic_sigma_g_levels.csv").write_text(
            "\n".join(
                [
                    "k,energy_cm_inv,A_tot_s_inv",
                    "1,100000.0,10.0",
                    "2,101000.0,20.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (table_dir / "nonadiabatic_sigma_g_eigenvectors.csv").write_text(
            "\n".join(
                [
                    "k,upper_state,upper_v,coeff",
                    "1,EF1Sg,0,0.8",
                    "2,EF1Sg,0,0.6",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (table_dir / "nonadiabatic_sigma_g_bound.csv").write_text(
            "\n".join(
                [
                    "k,lower_state,lower_v,A_s_inv",
                    "1,B1Su,0,4.0",
                    "1,C1Pu,0,6.0",
                    "2,Bp1Su,0,5.0",
                    "2,C1Pu,0,15.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        tables = load_reference_tables(table_dir)
        derived = tables.bound[("EF1Sg", 0)]
        by_target = {(entry.lower_state, entry.lower_v): entry.a_s_inv for entry in derived}

        # Normalized squared weights are 0.64 and 0.36.
        assert abs(by_target[("B1Su", 0)] - (0.64 * 4.0)) < 1.0e-12
        assert abs(by_target[("Bp1Su", 0)] - (0.36 * 5.0)) < 1.0e-12
        assert abs(by_target[("C1Pu", 0)] - (0.64 * 6.0 + 0.36 * 15.0)) < 1.0e-12

    print("[PASS] Dissociation-channel nonadiabatic-table checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
