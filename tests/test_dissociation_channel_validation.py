#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from scripts.dissociation_channels import (  # pylint: disable=import-error
        DissociationChannelError,
        load_reference_tables,
        write_template_tables,
    )

    with tempfile.TemporaryDirectory(prefix="diss_validation_") as td:
        table_dir = Path(td)
        write_template_tables(table_dir)
        (table_dir / "radiative_bound.csv").write_text(
            "upper_state,upper_v,lower_state,lower_v,A_s_inv\n"
            "NOT_A_STATE,0,X1Sg,0,1.0\n",
            encoding="utf-8",
        )
        try:
            load_reference_tables(table_dir)
        except DissociationChannelError as exc:
            assert "Unknown state label" in str(exc)
        else:
            raise AssertionError("Expected reference-table validation to fail")

    print("[PASS] Dissociation-channel validation checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

