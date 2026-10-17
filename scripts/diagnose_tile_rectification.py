"""Measure how tile pose (skew, foreshortening) affects classifier confidence.

For every detected tile in the given captured frames this script compares the
classifier on three crops of the same tile:

``bbox``     the current pipeline: the detector's axis-aligned box, resized to
             the classifier's fixed size, so a skewed tile stays skewed;
``silhouette`` the tile's outer silhouette fitted to a quadrilateral and
             perspective-warped to a canonical upright rectangle;
``face``     the tile's white face fitted to a quadrilateral and perspective
             warped, then expanded to the tile's known aspect ratio.

It reports, per region and per pose bucket, the confidence and whether the
top-1 label is stable across the three crops. Frames carry no per-tile ground
truth here, so the report describes geometry and stability, not accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, write_json
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank, rotate
from app.vision.text import TextReader

TILE_REGIONS = {"hand", "draw", "river", "dora", "meld"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/tile-pose-geometry-probe.json"))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    from app.vision import rectify as rectify_module

    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    bank = recognizer.bank
    images = []
    for root in args.root:
        images.extend(sorted(root.rglob("*.png")))
    step = max(1, len(images)//args.limit)
    images = images[::step][:args.limit]
    records = []
    for index, image in enumerate(images, 1):
        pixels = load_image(image)
        if list(pixels.shape[1::-1]) != config["client_size"]:
            continue
        observation = recognizer.recognize(pixels, {"image_path": str(image.resolve())})
        for name, region in observation["regions"].items():
            if region.get("kind") not in TILE_REGIONS:
                continue
            if name not in recognizer.config["regions"]:
                continue
            source = recognizer.config["regions"][name]
            x, y, w, h = region["bbox"]
            degrees = region["rotation"]
            crop = rotate(pixels[y:y+h, x:x+w], degrees)
            for item in region["items"]:
                box = item["canonical_bbox"]
                patch = crop[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                if patch.size == 0:
                    continue
                shapes = rectify_module.probe_shapes(patch, source)
                record = {"image": str(image), "region": name, "label": item["label"],
                          "identity_rotation": item.get("identity_rotation"),
                          "bbox_score": (item.get("pjura_evidence") or {}).get("score")}
                variants = rectify_module.probe_variants(patch, source, bank)
                record.update(variants)
                record["shapes"] = shapes
                records.append(record)
        print(f"{index}/{len(images)} {image.name} records={len(records)}", flush=True)

    summary = defaultdict(Counter)
    scores = defaultdict(list)
    for record in records:
        key = (record["region"], "horizontal" if record.get("identity_rotation") in (90, 270) else "upright")
        summary[key]["count"] += 1
        for variant in ("bbox", "silhouette", "face"):
            value = (record.get(variant) or {})
            if value.get("score") is not None:
                scores[(key[1], variant)].append(value["score"])
            if value.get("label") != record.get("label"):
                summary[key][f"{variant}_label_changed"] += 1
        if record.get("shapes", {}).get("silhouette") is None:
            summary[key]["no_silhouette_quad"] += 1
        if record.get("shapes", {}).get("face") is None:
            summary[key]["no_face_quad"] += 1
    report = {"scope": "geometry_and_stability_probe_not_accuracy",
              "config_version": config["version"], "images": [str(p) for p in images],
              "by_pose": {f"{a}/{b}": dict(value) for (a, b), value in sorted(summary.items())},
              "scores": {f"{b}/{v}": {"n": len(s), "median": statistics.median(s) if s else None,
                                      "mean": statistics.mean(s) if s else None,
                                      "min": min(s) if s else None,
                                      "p10": sorted(s)[max(0, len(s)//10-1)] if s else None}
                         for (b, v), s in sorted(scores.items())},
              "records": records}
    write_json(args.output, report)
    print(json.dumps({"by_pose": report["by_pose"], "scores": report["scores"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
