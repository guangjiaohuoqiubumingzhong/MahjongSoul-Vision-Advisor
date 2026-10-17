"""Find the client-area crop of a whole-window screenshot by landmark score.

The supplied screenshots show the game window plus the advice panel, so the
calibrated region rectangles only line up after cropping to the game window's
client area. The client size is unknown (the screenshot may be scaled), so the
crop is recovered from two landmarks: the action-timer digits, whose position
and glyph size follow the crop height, and the hand row, which must read 13
tiles for the response screenshots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.capture.frames import load_image
from app.vision.config import load_config, pixel_rect
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader
from app.vision.turn import detect_timer

ROOT = Path(__file__).resolve().parents[1]


def candidates(pixels):
    boxes = []
    for top in (25, 26, 27, 30, 31, 32, 33):
        for left in (0, 1, 2):
            for right in range(1900, 1940):
                for bottom in range(1040, min(1112, pixels.shape[0] + 1)):
                    if (right - left) < 1366 or (bottom - top) < 768:
                        continue
                    if abs(((right - left) / (bottom - top)) / (1366 / 768) - 1) > .01:
                        continue
                    boxes.append((left, top, right, bottom))
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--expect-round", default="東1局")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/no-draw-call-discard/alignment.json")
    args = parser.parse_args()
    config = load_config(args.config)
    import cv2

    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    reader = recognizer.text_reader
    pixels = load_image(args.image)
    landmarks = []
    for box in candidates(pixels):
        l, t, r, b = box
        if b > pixels.shape[0] or r > pixels.shape[1]:
            continue
        client = pixels[t:b, l:r]
        reference = cv2.resize(client, tuple(config["client_size"]), interpolation=cv2.INTER_AREA)
        timer = detect_timer(reference, config["action_timer"], reader)
        if not timer.get("active"):
            continue
        landmarks.append({"crop": list(box), "timer": timer.get("raw_text"), "timer_score": round(float(timer["score"]), 4)})
    landmarks.sort(key=lambda v: -v["timer_score"])
    print(f"timer landmarks: {len(landmarks)}")
    for item in landmarks[:10]:
        print("  ", item)
    results = []
    for item in landmarks[:24]:
        l, t, r, b = item["crop"]
        client = pixels[t:b, l:r]
        try:
            observation = recognizer.recognize(client, {})
        except Exception as exc:
            continue
        regions = observation["regions"]
        hand = [i["label"] for i in regions["hand"]["items"]]
        buttons = [i["label"] for i in regions["buttons"]["items"]]
        scores = [(regions[n].get("text") or {}).get("value")
                  for n in ("score_self", "score_right", "score_top", "score_left")]
        round_value = (regions["round"].get("text") or {}).get("value")
        remaining = (regions["remaining"].get("text") or {}).get("value")
        quality = (2 * len(hand) + 8 * int(regions["hand"]["complete"])
                   + 4 * sum(1 for v in scores if v is not None)
                   + 4 * int(round_value is not None) + 2 * int(remaining is not None)
                   + 3 * len(buttons) + 4 * int(regions["dora"]["complete"])
                   + 2 * int(regions["river_self"]["complete"]))
        results.append({"crop": list(item["crop"]), "timer": item["timer"], "quality": round(quality, 2),
                        "hand": hand, "hand_complete": regions["hand"]["complete"],
                        "dora": [i["label"] for i in regions["dora"]["items"]],
                        "buttons": buttons, "scores": scores, "round": round_value,
                        "remaining": remaining})
        print(json.dumps(results[-1], ensure_ascii=False))
    results.sort(key=lambda v: -v["quality"])
    print("BEST:", json.dumps(results[0], ensure_ascii=False) if results else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
