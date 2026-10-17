"""Measure tile classification under controlled plane tilt, with and without correction.

Labeled hand tiles from reviewed annotations supply real game pixels with a known
identity. Each tile is re-projected as if it lay flat on the table and were seen
from an angle: the projected outline gains a trapezoid shape (skew) and, when the
viewing direction is compressed, a shorter long axis (foreshortening, reported as
the outline's long/short ratio). Those are exactly the two geometric features the
detector's axis-aligned box cannot undo, and real river tiles in the recorded
frames were measured to span ratios from 1.50 down to about 1.05.

For every case the tile is classified twice:

``bbox``       the current pipeline, resizing the axis-aligned box of the
               projected tile to the network input;
``corrected``  the tile's fitted outline warped back to the canonical upright
               face before classification.

Development evidence for the correction, not an acceptance result: the tiles are
re-projections of samples that are already used elsewhere in the project.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, write_json
from app.vision.recognizer import StaticRecognizer
from app.vision.rectify import min_area_quad, rectify_tile, rectify_tile_aspect
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

TILE_RATIO = 1.5  # a tile face is 2:3, so its long axis is 1.5 short axes


def project(patch, ratio=TILE_RATIO, skew=0.0, margin=0.10):
    """Re-project an upright tile onto a plane seen at an angle.

    ``ratio`` is the projected long/short edge ratio; ``skew`` compresses one
    end of the tile relative to the other, producing the trapezoid an off-axis
    camera yields. The result is cropped to the projected outline plus
    ``margin``, which is how the detector's box sits around a real tile.
    """
    height, width = patch.shape[:2]
    long_side, short_side = height, width
    ratio = max(float(ratio), 1.0)
    far = short_side * (1 - skew) / 2
    near = short_side * (1 + skew) / 2
    # The projected long axis is the true long axis shortened by the viewing
    # direction, so the ratio fixes the projected height directly.
    target_height = long_side * ratio / TILE_RATIO
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    average_width = (far + near) / 2
    target = np.array([[short_side / 2 - far, 0], [short_side / 2 + far, 0],
                       [short_side / 2 + near, target_height - 1],
                       [short_side / 2 - near, target_height - 1]], np.float32)
    target[:, 0] += (short_side / 2 - short_side / 2)
    target[:, 0] += (width - short_side) / 2
    minimum = target.min(axis=0) - 2
    target = target - minimum
    size = np.ceil(target.max(axis=0) + 2).astype(int)
    transform = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(patch, transform, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
    pad_y = max(1, int(round(warped.shape[0] * margin)))
    pad_x = max(1, int(round(warped.shape[1] * margin)))
    return cv2.copyMakeBorder(warped, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def classify(bank, patch, role="hand"):
    match = bank.classify(patch, role, 0, 0, (0, 90, 180, 270), kinds=("tile",))
    return {"accepted": match.get("label"), "best": match.get("best_label"),
            "score": (match.get("pjura_evidence") or {}).get("score", match.get("score")),
            "rotation": match.get("identity_rotation"),
            "reason": match.get("reason")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.5, 1.3, 1.15, 1.0])
    parser.add_argument("--skews", type=float, nargs="+", default=[0.0, 0.12, 0.25, 0.5])
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/tile-perspective-benchmark.json"))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    bank = recognizer.bank
    tiles = []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if not annotation.get("reviewed"):
            continue
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            continue
        pixels = load_image(image)
        if list(pixels.shape[1::-1]) != config["client_size"]:
            continue
        for target in annotation["targets"]:
            if (target["kind"] != "tile" or target["label"] in {"?", "back"}
                    or target["region"] not in {"hand", "draw"} or target.get("rotation", 0) != 0):
                continue
            x, y, w, h = target["bbox"]
            tiles.append({"label": target["label"], "patch": pixels[y:y+h, x:x+w]})
        if len(tiles) >= args.limit:
            break
    tiles = tiles[:args.limit]

    records, summary = [], {}
    for ratio in args.ratios:
        for skew in args.skews:
            subset = []
            for tile in tiles:
                patch = project(tile["patch"], ratio, skew)
                baseline = classify(bank, patch)
                # Control: the same crop resized to the warped variants' input
                # resolution, so any difference cannot come from pixel count.
                resized = cv2.resize(patch, (120, 180), interpolation=cv2.INTER_LINEAR)
                control = classify(bank, resized)
                geometry = min_area_quad(patch)
                canonical = rectify_tile(patch, geometry=geometry) if geometry is not None else None
                aspect = rectify_tile_aspect(patch, geometry=geometry) if geometry is not None else None
                subset.append({"ground_truth": tile["label"], "bbox": baseline, "upscaled": control,
                               "corrected": classify(bank, canonical.pixels) if canonical is not None else None,
                               "kept_aspect": classify(bank, aspect.pixels) if aspect is not None else None,
                               "outline_aspect": geometry.aspect if geometry else None,
                               "outline_area": geometry.area_fraction if geometry else None})
            key = f"ratio={ratio},skew={skew}"
            entry = {"tiles": len(subset), "with_outline": sum(1 for r in subset if r["outline_aspect"]),
                     "outline_aspect_median": statistics.median([r["outline_aspect"] for r in subset if r["outline_aspect"]])
                     if any(r["outline_aspect"] for r in subset) else None}
            for name in ("bbox", "upscaled", "corrected", "kept_aspect"):
                evaluated = [r for r in subset if r[name] is not None]
                correct = sum(r[name]["accepted"] == r["ground_truth"] for r in evaluated)
                unknown = sum(r[name]["accepted"] == "?" for r in evaluated)
                scores = [r[name]["score"] for r in evaluated if r[name]["score"] is not None]
                entry[name] = {"evaluated": len(evaluated), "correct": correct, "unknown": unknown,
                               "wrong": len(evaluated) - correct - unknown,
                               "accuracy": correct / len(evaluated) if evaluated else None,
                               "accepted_rate": (len(evaluated) - unknown) / len(subset) if subset else None,
                               "score_median": statistics.median(scores) if scores else None,
                               "score_p10": sorted(scores)[max(0, len(scores)//10-1)] if scores else None,
                               "misses": [{"ground_truth": r["ground_truth"], "accepted": r[name]["accepted"],
                                           "score": r[name]["score"]}
                                          for r in subset if r[name] is not None
                                          and r[name]["accepted"] != r["ground_truth"]][:6]}
            summary[key] = entry
            records.extend({**record, "ratio": ratio, "skew": skew} for record in subset)
            print(f"{key} done", flush=True)

    report = {"scope": "controlled_development_benchmark_not_acceptance",
              "config_version": config["version"], "tile_classifier": getattr(bank, "tile_classifier_info", None),
              "tilt_model": "tile plane re-projection: ratio = projected long/short edge, skew = far/near compression",
              "summary": summary, "records": records}
    write_json(args.output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
