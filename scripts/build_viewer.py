"""Regenerates the shipped viewer asset from its sources.

    viewer/src.html            — viewer source (edit this)
    viewer/three_r128_inline.html — vendored three.js r128, pre-wrapped in
                                    <script>...</script> (do not edit)
        -> torchspace/assets/viewer.html   (packaged, three.js inlined)

Run after every change to viewer/src.html:

    python scripts/build_viewer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "<!--THREE_INLINE-->"

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "viewer" / "src.html"
THREE = ROOT / "viewer" / "three_r128_inline.html"
OUT = ROOT / "torchspace" / "assets" / "viewer.html"


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    if MARKER not in src:
        print(f"error: {SRC} does not contain the {MARKER} marker",
              file=sys.stderr)
        return 1
    three = THREE.read_text(encoding="utf-8")
    if "<script>" not in three:
        print(f"error: {THREE} does not look like a wrapped script block",
              file=sys.stderr)
        return 1
    out = src.replace(MARKER, three.rstrip("\n"))
    OUT.write_text(out, encoding="utf-8")
    print(f"built {OUT} ({OUT.stat().st_size / 1024:.0f} KB) "
          f"from {SRC.name} + {THREE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
