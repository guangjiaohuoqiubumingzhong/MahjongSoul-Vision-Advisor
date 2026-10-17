"""Find the client-area crop of a whole-window screenshot by landmark score.

The supplied screenshots are the desktop (game client plus the advice panel),
so the calibrated region rectangles only line up after cropping to the client
area. The action-timer digits and the round text are stable landmarks: a crop
whose scale and offset are right reads them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.capture.frames import load_image
from app.vision.config import load_config, pixel_rect
from app.vision.text import TextReader
from app.vision.turn import detect_timer

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--expect-round", default="東1局")
    args = parser.parse_args()
    config = load_config(args.config)
    reader = TextReader()
    pixels = load_image(args.image)
    height, width = pixels.shape[:2]
    timer_rect = config["regions"]["buttons"]["rect"]  # not used; kept for parity
    results = []
    for top in range(26, 40):
        for left in range(0, 6):
            for right in range(1912, 1928):
                for bottom in range(1060, 1105):
                    crop = pixels[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    ch, cw = crop.shape[:2]
                    if abs((cw / ch) / (1366 / 768) - 1) > .01:
                        continue
                    reference = cv2.resize(crop, (1366, 768), interpolation=cv2.INTER_AREA)
                    timer = detect_timer(reference, config["action_timer"], reader)
                    if not timer.get("active"):
                        continue
                    results.append({"crop": [left, top, cw, ch], "timer": timer})
    print(f"candidates: {len(results)}")
    for item in sorted(results, key=lambda v: -v["timer"]["score"])[:15]:
        print(item["crop"], item["timer"]["raw_text"], round(item["timer"]["score"], 4))
    Path(ROOT / "reports/no-draw-call-discard/timer-sweep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
