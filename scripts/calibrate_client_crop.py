"""Recover the client-area crop of a whole-window screenshot from the board.

The screenshot shows the game window plus the advice panel, and the client
size is unknown, so the crop is scored by what the recognizer reads: a full
13-tile hand for the response screenshots is the strongest signal.

Usage: python -m scripts.calibrate_client_crop <image> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.capture.frames import load_image
from app.vision.config import load_config
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]


def hand_band(pixels: np.ndarray):
    """Rows of the bottom hand row from its bright, low-saturation faces."""
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    face = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 185))[:, :1920]
    rows = np.nonzero(face.mean(axis=1) > .30)[0]
    return (int(rows[0]), int(rows[-1])) if rows.size else (None, None)


def score(observation, expected_hand):
    regions = observation["regions"]
    hand = [i["label"] for i in regions["hand"]["items"]]
    return (10 * int(regions["hand"]["complete"] and len(hand) == expected_hand)
            + 2 * len(hand)
            + 4 * int(regions["draw"]["complete"])
            + 4 * int(regions["dora"]["complete"])
            + 3 * int(bool(regions["buttons"]["items"]))
            + 2 * int(regions["river_self"]["complete"])
            + 2 * int((regions["remaining"].get("text") or {}).get("value") is not None)
            + 2 * int((regions["round"].get("text") or {}).get("value") is not None))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--expect-hand", type=int, default=13)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    pixels = load_image(args.image)
    print("hand band:", hand_band(pixels))
    results = []
    for left in range(0, 4):
        for right in range(1898, 1932):
            for bottom in range(1046, min(1112, pixels.shape[0] + 1)):
                height = bottom - 31
                width = right - left
                if width < 1366 or height < 768:
                    continue
                if abs((width / height) / (1366 / 768) - 1) > .012:
                    continue
                client = pixels[31:bottom, left:right]
                try:
                    observation = recognizer.recognize(client, {})
                except Exception:
                    continue
                regions = observation["regions"]
                hand = [i["label"] for i in regions["hand"]["items"]]
                results.append({"crop": [left, 31, right, bottom], "score": score(observation, args.expect_hand),
                                "hand": hand, "hand_complete": regions["hand"]["complete"],
                                "draw": [i["label"] for i in regions["draw"]["items"]],
                                "buttons": [i["label"] for i in regions["buttons"]["items"]],
                                "remaining": (regions["remaining"].get("text") or {}).get("value")})
    results.sort(key=lambda v: -v["score"])
    for item in results[:8]:
        print(json.dumps(item, ensure_ascii=False))
    print("BEST:", json.dumps(results[0], ensure_ascii=False) if results else None)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results[:40], ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
