#!/usr/bin/env python3
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    if shutil.which("tesseract") is None:
        print("[SKIP] tesseract not available.")
        return 0

    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "ocr_reference_table.py"
    assert script.exists(), f"Missing OCR helper: {script}"

    with tempfile.TemporaryDirectory(prefix="ocr_ref_") as td:
        root = Path(td)
        image_path = root / "table.1.png"
        outdir = root / "out"

        image = Image.new("RGB", (900, 180), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), "upper_state upper_v lower_state lower_v A_s_inv", fill="black")
        draw.text((20, 80), "B1Su 0 X1Sg 0 1.23E+08", fill="black")
        image.save(image_path)

        proc = subprocess.run(
            [sys.executable, str(script), str(image_path), "--outdir", str(outdir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
        assert proc.returncode == 0, "OCR helper failed"
        txt_path = outdir / "table.1.txt"
        assert txt_path.exists(), "Missing OCR output text file"
        assert txt_path.read_text(encoding="utf-8").strip(), "OCR output is empty"

    print("[PASS] OCR helper smoke checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
