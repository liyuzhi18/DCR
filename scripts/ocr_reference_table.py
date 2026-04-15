#!/usr/bin/env python3
"""
Extract draft text from reference screenshots using tesseract.

The output is a human-reviewable text draft only. It is not used directly by the
dissociation-channel calculator.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Reference screenshot image(s)")
    parser.add_argument("--outdir", type=Path, default=Path("output/ocr_reference_tables"), help="Output directory")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--lang", default="eng", help="OCR language")
    return parser.parse_args()


def run_tesseract(image: Path, outdir: Path, psm: int, lang: str) -> Path:
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        raise SystemExit("tesseract was not found in PATH")
    if not image.exists():
        raise SystemExit(f"Missing image: {image}")

    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / image.stem
    cmd = [tesseract, str(image), str(base), "--psm", str(psm), "-l", lang]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or f"tesseract failed on {image}")
    # `image.stem` can itself contain dots (for example macOS screenshot names).
    # `with_suffix()` would drop the trailing dotted portion, so build the output
    # path explicitly from the Tesseract basename instead.
    txt_path = Path(f"{base}.txt")
    if not txt_path.exists():
        raise SystemExit(f"Expected OCR output missing: {txt_path}")
    return txt_path


def main() -> int:
    args = parse_args()
    for image in args.images:
        txt = run_tesseract(image, args.outdir, args.psm, args.lang)
        print(f"Wrote draft OCR text: {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
