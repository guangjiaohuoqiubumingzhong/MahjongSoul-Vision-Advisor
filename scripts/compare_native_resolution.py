"""Does a larger game window help tile classification?

For every captured frame larger than the calibrated 1366x768 (same 16:9
aspect), the recognizer runs twice on the *same* frame:

* ``native``     - layout at 1366x768, classifier crops from the native frame
* ``calibrated`` - the frame downscaled to 1366x768 only (what a 1366 window gives)

Tiles are paired by region and box. The report counts accepted / unknown per
region, compares scores, and lists every tile whose label differs. An atlas of
those tiles (calibrated crop | native crop | labels) is written for review, so
the differences can be checked by eye even without annotations.

    .venv\\Scripts\\python scripts\\compare_native_resolution.py
    .venv\\Scripts\\python scripts\\compare_native_resolution.py --images "data/screenshots/manual-captures/**/*.png"
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, native_box, scaled_client, to_reference, write_json
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

DEFAULT_PATTERNS = ("data/screenshots/**/*.png", "reports/screenshot-advice/**/*.png")
TILE_KINDS = {"hand", "draw", "dora", "river", "meld"}


def native_frames(patterns, config, stride=1):
    seen, index = set(), 0
    for pattern in patterns:
        for name in sorted(glob.glob(pattern, recursive=True)):
            path = Path(name)
            if path in seen or ".overlay" in path.name:
                continue
            seen.add(path)
            index += 1
            if (index - 1) % stride:
                continue
            pixels = load_image(path)
            height, width = pixels.shape[:2]
            scale = scaled_client(width, height, config)
            if scale is not None and scale > 1.0:
                yield path, pixels


def tiles(observation):
    for name, region in observation["regions"].items():
        if region["kind"] not in TILE_KINDS:
            continue
        for item in region["items"]:
            yield (name, tuple(item["bbox"])), item


def score(item):
    return (item.get("pjura_evidence") or {}).get("score", item.get("score"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--images", action="append", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/native-resolution-comparison.json"))
    parser.add_argument("--stride", type=int, default=1, help="use every Nth image")
    parser.add_argument("--atlas", type=Path, default=Path("reports/vision/native-resolution-differences.png"))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    recogniser = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")),
                                  TextReader())
    totals = {mode: defaultdict(lambda: {"tiles": 0, "accepted": 0, "unknown": 0, "scores": []})
              for mode in ("native", "calibrated")}
    differences, frames, cells = [], [], []
    for path, native in native_frames(args.images or DEFAULT_PATTERNS, config, args.stride):
        reference, _, scale = to_reference(native, config)
        runs = {"native": recogniser.recognize(native, {"image_path": str(path)}),
                "calibrated": recogniser.recognize(reference, {"image_path": str(path)})}
        paired = {mode: dict(tiles(observation)) for mode, observation in runs.items()}
        frames.append({"image": str(path), "size": [native.shape[1], native.shape[0]], "scale": scale,
                       "native_tiles": len(paired["native"]), "calibrated_tiles": len(paired["calibrated"])})
        for mode, items in paired.items():
            for (region, _), item in items.items():
                bucket = totals[mode][region]
                bucket["tiles"] += 1
                bucket["accepted"] += item["label"] != "?"
                bucket["unknown"] += item["label"] == "?"
                if score(item) is not None:
                    bucket["scores"].append(score(item))
        for key in paired["native"].keys() & paired["calibrated"].keys():
            high, low = paired["native"][key], paired["calibrated"][key]
            if high["label"] == low["label"]:
                continue
            region, box = key
            differences.append({"image": str(path), "region": region, "bbox": list(box),
                                "calibrated": [low["label"], score(low), low.get("reason")],
                                "native": [high["label"], score(high), high.get("reason")]})
            x, y, w, h = box
            nx, ny, nw, nh = native_box(box, scale, native.shape)
            panel = np.full((150, 230, 3), 30, np.uint8)
            for left, crop in ((4, reference[y:y+h, x:x+w]), (116, native[ny:ny+nh, nx:nx+nw])):
                ratio = min(108 / crop.shape[1], 110 / crop.shape[0])
                crop = cv2.resize(crop, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
                panel[34:34+crop.shape[0], left:left+crop.shape[1]] = crop
            cv2.putText(panel, region[:14], (4, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1)
            cv2.putText(panel, f"1366:{low['label']} {score(low) or 0:.2f}", (4, 28), cv2.FONT_HERSHEY_SIMPLEX, .38,
                        (200, 200, 255), 1)
            cv2.putText(panel, f"native:{high['label']} {score(high) or 0:.2f}", (116, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (200, 255, 200), 1)
            cells.append(panel)
        print(f"{path}: {native.shape[1]}x{native.shape[0]} differences so far {len(differences)}", flush=True)

    summary = {}
    for mode, regions in totals.items():
        summary[mode] = {}
        everything = {"tiles": 0, "accepted": 0, "unknown": 0, "scores": []}
        for region, bucket in sorted(regions.items()):
            for key in ("tiles", "accepted", "unknown"):
                everything[key] += bucket[key]
            everything["scores"] += bucket["scores"]
            summary[mode][region] = {**{k: bucket[k] for k in ("tiles", "accepted", "unknown")},
                                     "score_median": float(np.median(bucket["scores"])) if bucket["scores"] else None,
                                     "score_p10": float(np.percentile(bucket["scores"], 10)) if bucket["scores"] else None}
        summary[mode]["all"] = {**{k: everything[k] for k in ("tiles", "accepted", "unknown")},
                                "score_median": float(np.median(everything["scores"])) if everything["scores"] else None,
                                "score_p10": float(np.percentile(everything["scores"], 10)) if everything["scores"] else None}
    report = {"scope": "same_frames_native_vs_downscaled_no_ground_truth", "config_version": config["version"],
              "frames": frames, "summary": summary, "label_differences": differences}
    write_json(args.output, report)
    if cells:
        while len(cells) % 6:
            cells.append(np.full((150, 230, 3), 30, np.uint8))
        atlas = np.vstack([np.hstack(cells[i:i+6]) for i in range(0, len(cells), 6)])
        args.atlas.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", atlas)[1].tofile(str(args.atlas))
    if not frames:
        print("没有找到大于 1366×768 的同比例截图。请先把游戏窗口调到 1920×1080 并采集几张。")
    print(json.dumps({mode: summary[mode].get("all") for mode in summary}, ensure_ascii=False, indent=2))
    print(f"frames={len(frames)} label_differences={len(differences)} report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
