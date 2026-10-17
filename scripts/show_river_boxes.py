"""Crop a river region and draw the detected tile boxes on it.

Usage: python -m scripts.show_river_boxes <fixture> <region-name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.capture.frames import load_image
from app.vision.config import load_config
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("region")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/no-draw-call-discard")
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    pixels = load_image(args.image)
    observation = recognizer.recognize(pixels, {})
    region = observation["regions"][args.region]
    x, y, w, h = region["bbox"]
    crop = pixels[y:y + h, x:x + w].copy()
    for index, item in enumerate(region["items"]):
        bx, by, bw, bh = item["bbox"]
        cv2.rectangle(crop, (bx - x, by - y), (bx - x + bw, by - y + bh), (0, 0, 255), 2)
        label = f"{index}:{item['label']}"
        if "slot" in item:
            label += f"@{item['slot']}"
        cv2.putText(crop, label, (bx - x, max(12, by - y - 3)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
    for part in region.get("unresolved_components", []):
        bx, by, bw, bh = part["bbox"]
        cv2.rectangle(crop, (bx - x, by - y), (bx - x + bw, by - y + bh), (255, 0, 255), 2)
    zoom = cv2.resize(crop, (int(w * args.scale), int(h * args.scale)), interpolation=cv2.INTER_NEAREST)
    target = args.out / f"{Path(args.image).stem}-{args.region}-boxes.png"
    cv2.imwrite(str(target), zoom)
    print("region bbox", region["bbox"], "items", [(i["label"], i.get("slot")) for i in region["items"]],
          "unresolved", region.get("unresolved_components"), "->", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
