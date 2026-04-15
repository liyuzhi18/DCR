#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from scripts.dissociation_channels import (  # pylint: disable=import-error
        GROUND_STATE,
        load_reference_tables,
        write_template_tables,
    )

    with tempfile.TemporaryDirectory(prefix="diss_summary_") as td:
        table_dir = Path(td)
        write_template_tables(table_dir)
        (table_dir / "state_decay_summary.csv").write_text(
            "\n".join(
                [
                    "state,upper_v,A_cont_s_inv,A_bound_s_inv,A_tot_s_inv,tau_s,f_cont",
                    "B1Su,0,1.0,9.0,10.0,1.0e-9,0.1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        tables = load_reference_tables(table_dir)
        bound = tables.bound[("B1Su", 0)]
        cont = tables.continuum[("B1Su", 0)]
        assert len(bound) == 1
        assert len(cont) == 1
        assert bound[0].lower_state == GROUND_STATE
        assert bound[0].lower_v == -1
        assert abs(bound[0].a_s_inv - 9.0) < 1e-12
        assert abs(cont[0].a_cont_s_inv - 1.0) < 1e-12

    print("[PASS] Dissociation-channel summary-table checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
