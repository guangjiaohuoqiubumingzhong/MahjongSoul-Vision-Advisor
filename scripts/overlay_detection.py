"""Draw region detection boxes and the glow mask on a raw screenshot.

Usage: python -m scripts.overlay_detection <image> <region> [--crop l,t,r,b]

Shows whether the calibrated region rectangles line up with the game and
which pixels the reaction-glow detector sees.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.capture.frames import load_image
from app.vision.actions import glow_mask
from app.vision.config import load_config, pixel_rect
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("regions", nargs="+")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--crop", default="0,26,1912,1092")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/no-draw-call-discard")
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    pixels = load_image(args.image)
    l, t, r, b = (int(v) for v in args.crop.split(","))
    client = pixels[t:b, l:r]
    observation = recognizer.recognize(client, {})
    glow = glow_mask(client)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.regions:
        region = observation["regions"][name]
        x, y, w, h = region["bbox"]
        canvas = client.copy()
        canvas[glow] = (0, 255, 255)
        for item in region.get("items", []):
            ix, iy, iw, ih = item["bbox"]
            cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), (0, 0, 255), 2)
            cv2.putText(canvas, item["label"], (ix, max(14, iy - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, (0, 0, 255), 2)
        for group in region.get("groups") or []:
            for item in group.get("items", []):
                ix, iy, iw, ih = item["bbox"]
                cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), (255, 0, 255), 2)
        path = args.out / f"{Path(args.image).stem}-{name}-boxes.png"
        cv2.imwrite(str(path), canvas[y:y + h, x:x + w])
        print(f"{name}: bbox={region['bbox']} complete={region['complete']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
