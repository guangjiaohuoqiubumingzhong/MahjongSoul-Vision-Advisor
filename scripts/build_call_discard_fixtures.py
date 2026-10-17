"""Build 1920x1080 client-area fixtures from the whole-window screenshots.

The user's screenshots show the desktop: the game window plus the advice panel.
The game client is cropped out and resampled to the 16:9 client size the app
captures, so the fixtures behave like the real capture path.

Usage: python -m scripts.build_call_discard_fixtures
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/fixtures/no-draw-call-discard"
# text label -> source file, client crop (left, top, right, bottom) in source px
FIXTURES = {
    "chi-buttons-1920": ("fig2-chi-buttons.png", (0, 31, 1920, 1112)),
    "ron-buttons-1920": ("fig3-ron-buttons.png", (0, 31, 1918, 1111)),
    "after-chi-no-advice-1920": ("fig1-after-chi-no-advice.png", (0, 31, 1918, 1110)),
}


def find_border(grey: np.ndarray):
    """Client area inside the game window's black frame."""
    height, width = grey.shape[:2]
    dark = grey < 45
    columns = [i for i in range(width) if dark[:, i].mean() > .55]
    rows = [i for i in range(height) if dark[i, :columns[-1]].mean() > .5]
    return columns[-1], rows[0]


def main() -> int:
    manifest = {}
    for name, (filename, crop) in FIXTURES.items():
        pixels = cv2.imread(str(SOURCE / filename))
        grey = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        right_border, top_border = find_border(grey)
        left, top, right, bottom = crop
        right = min(right, right_border)
        client = pixels[top:bottom, left:right]
        resized = cv2.resize(client, (1920, 1080), interpolation=cv2.INTER_AREA)
        target = SOURCE / f"{name}.png"
        cv2.imwrite(str(target), resized)
        manifest[name] = {"source": filename, "crop": [left, top, right, bottom],
                          "client": [right - left, bottom - top], "output": target.name}
        print(f"{name}: {client.shape[1]}x{client.shape[0]} -> 1920x1080 ({target})")
    (SOURCE / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
