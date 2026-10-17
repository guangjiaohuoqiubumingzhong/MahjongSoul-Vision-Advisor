"""Locate the reaction glow and the discard it marks in a raw screenshot.

The game outlines the discard waiting for a call with a pulsing green halo.
This prints every connected glow region with its size and position so the
detector's assumptions (final tile of a river, minimum sides lined) can be
checked against the real pixels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.capture.frames import load_image
from app.vision.actions import GLOW_MARGIN, glow_mask, tile_glow

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    for path in args.images:
        pixels = load_image(path)
        mask = glow_mask(pixels).astype(np.uint8)
        joined = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(joined)
        pieces = sorted(((int(stats[i][4]), [int(v) for v in stats[i][:4]], [round(float(v), 1) for v in centroids[i]])
                         for i in range(1, count)), reverse=True)
        print(f"== {Path(path).name} {pixels.shape[1]}x{pixels.shape[0]} glow components={len(pieces)}")
        for area, box, centre in pieces[:12]:
            print(f"   area={area:6d} box={box} centre={centre}")
        # Visual: glow pixels in magenta over the region right of x=900 and
        # below y=350, where the right and top rivers hold the called discard.
        canvas = pixels.copy()
        canvas[mask.astype(bool)] = (255, 0, 255)
        out = ROOT / "reports/no-draw-call-discard"
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{Path(path).stem}-glowoverlay.png"), canvas[200:800, 900:1500])
        cv2.imwrite(str(out / f"{Path(path).stem}-glow.png"), (mask * 255))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
