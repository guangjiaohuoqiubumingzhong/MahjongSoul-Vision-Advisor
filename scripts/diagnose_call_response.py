"""Align a raw whole-window screenshot to the calibrated client area and dump evidence.

Usage: python -m scripts.diagnose_call_response <image> [--crop l,t,r,b]

The supplied screenshots show the desktop: the game window plus the advice
panel to its right. Region rectangles only line up after cropping to the game
window's client area; without ``--crop`` the black window border is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.capture.frames import load_image
from app.vision.actions import call_banners, enabled_action_buttons, glow_mask, highlighted_discard
from app.vision.config import load_config, to_reference
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]


def find_client(pixels: np.ndarray) -> list[int]:
    """Client area of the game window: inside the pure-black window border.

    The advice panel is to the right of the game window and shares no colour
    with its frame, so the rightmost long dark column is the window's edge.
    """
    grey = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape[:2]
    dark = grey < 45
    columns = [i for i in range(width) if dark[:, i].mean() > .55]
    if not columns:
        raise SystemExit("no window border found")
    x1 = columns[-1]
    rows = [i for i in range(height) if dark[i, :x1].mean() > .5]
    y0 = rows[0] if rows else 0
    return [0, int(y0), int(x1), height - int(y0)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--crop", default=None, help="left,top,right,bottom in source pixels")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/no-draw-call-discard")
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for path in args.images:
        pixels = load_image(path)
        if args.crop:
            l, t, r, b = (int(v) for v in args.crop.split(","))
        else:
            l, t, r, b = find_client(pixels)
        client = pixels[t:b, l:r]
        reference, native, scale = to_reference(client, config)
        print(f"== {Path(path).name} source={pixels.shape[1]}x{pixels.shape[0]} "
              f"crop=({l},{t},{r},{b}) client={client.shape[1]}x{client.shape[0]} scale={scale:.4f}")
        observation = recognizer.recognize(client, {"image_path": str(Path(path).resolve())})
        result = {"image": str(path), "crop": [l, t, r, b], "client_size": list(client.shape[:2][::-1]),
                  "scale": scale, "status": observation["status"], "issues": observation["issues"],
                  "action_timer": observation.get("action_timer"),
                  "reaction_discard": observation.get("reaction_discard"),
                  "self_wind": observation.get("self_wind"),
                  "regions": {name: {"kind": r["kind"], "complete": r["complete"], "player": r.get("player"),
                                     "items": [i["label"] for i in r.get("items", [])],
                                     "groups": r.get("groups"),
                                     "text": (r.get("text") or {}).get("raw_text")}
                              for name, r in observation["regions"].items()}}
        print(json.dumps({"status": result["status"], "issues": result["issues"],
                          "timer": result["action_timer"], "reaction": result["reaction_discard"],
                          "regions": result["regions"]}, ensure_ascii=False, indent=1))
        stem = Path(path).stem
        cv2.imwrite(str(args.out / f"{stem}-aligned.png"), reference)
        rect = config["regions"]["buttons"]["banner_rect"]
        bx, by, bw, bh = (round(rect[0] * reference.shape[1]), round(rect[1] * reference.shape[0]),
                          round(rect[2] * reference.shape[1]), round(rect[3] * reference.shape[0]))
        crop = reference[by:by + bh, bx:bx + bw]
        cv2.imwrite(str(args.out / f"{stem}-banners.png"), crop)
        result["banners"] = call_banners(crop, recognizer.text_reader)
        print("banners:", json.dumps(result["banners"], ensure_ascii=False))
        cv2.imwrite(str(args.out / f"{stem}-glowmask.png"),
                    (glow_mask(reference).astype(np.uint8) * 255))
        report.append(result)
    (args.out / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
