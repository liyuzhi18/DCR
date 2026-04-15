#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from scripts.dissociation_channels import (  # pylint: disable=import-error
        parse_state_fit_file,
    )

    fit_path = repo / "atomic_data" / "vibex_channel" / "EIE_Xsec2RC" / "X1Sg" / "fits" / "vi=0" / "MCCC-el-H2-B1Su.X1Sg_vi=0_fit.txt"
    parsed = parse_state_fit_file(fit_path)

    assert parsed.state == "B1Su"
    assert parsed.vi == 0
    assert len(parsed.bound_rows) == 40
    assert parsed.de_row is not None
    assert parsed.de_row.label == "DE"
    assert parsed.de_row.is_de
    assert 0 in parsed.bound_rows
    assert 39 in parsed.bound_rows

    print("[PASS] Dissociation-channel fit parser checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

