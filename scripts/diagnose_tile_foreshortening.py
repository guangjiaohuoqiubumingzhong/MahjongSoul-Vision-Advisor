"""Report how often detected tiles lie at an angle, and how much they are skewed.

For every detected tile in captured frames this prints the fitted outline's
aspect ratio and area fraction, the foreshortening relative to the tile's true
proportions, and whether the correction would be applied. It also writes the
histogram, so a threshold can be chosen from measured geometry instead of a
guess. No config, template or model is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, write_json
from app.vision.recognizer import StaticRecognizer
from app.vision.rectify import min_area_quad, face_quad, quad_skew
from app.vision.templates import TemplateBank, rotate
from app.vision.text import TextReader

TILE_REGIONS = {"hand", "draw", "river", "dora", "meld"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/tile-pose-foreshortening.json"))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    from app.vision import pjura as pjura_module

    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
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
            source = recognizer.config["regions"].get(name)
            if source is None:
                continue
            x, y, w, h = region["bbox"]
            crop = rotate(pixels[y:y+h, x:x+w], region["rotation"])
            for item in region["items"]:
                box = item["canonical_bbox"]
                patch = crop[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                if patch.size == 0:
                    continue
                geometry = min_area_quad(patch)
                face = face_quad(patch)
                records.append({"image": str(image), "region": name, "label": item["label"],
                                "identity_rotation": item.get("identity_rotation"),
                                "silhouette_aspect": geometry.aspect if geometry else None,
                                "silhouette_area": geometry.area_fraction if geometry else None,
                                "silhouette_skew": quad_skew(geometry.quad) if geometry else None,
                                "face_aspect": face.aspect if face else None,
                                "face_area": face.area_fraction if face else None,
                                "face_skew": quad_skew(face.quad) if face else None,
                                "foreshortening": (pjura_module.outline_foreshortening(geometry.quad)
                                                   if geometry else None),
                                "fires": geometry is not None and pjura_module.rectification_needed(patch) is not None})
        print(f"{index}/{len(images)} {image.name} records={len(records)}", flush=True)

    values = [r["foreshortening"] for r in records if r["foreshortening"] is not None]
    skews = [r["silhouette_skew"] for r in records if r["silhouette_skew"] is not None]
    hand_skews = [r["silhouette_skew"] for r in records
                  if r["silhouette_skew"] is not None and r["region"] in {"hand", "draw"}]
    buckets = Counter(f"{min(int(v*10)/10, 1.0):.1f}" for v in values)
    skew_buckets = Counter(f"{min(int(v*20)/20, 1.0):.2f}" for v in skews)
    by_region = Counter(r["region"] for r in records if r["fires"])
    report = {"scope": "measured_geometry_only",
              "count": len(records), "with_outline": len(values), "fires": sum(1 for r in records if r["fires"]),
              "foreshortening": {"median": statistics.median(values) if values else None,
                                 "mean": statistics.mean(values) if values else None,
                                 "p90": sorted(values)[int(.9*len(values))] if values else None,
                                 "histogram": dict(sorted(buckets.items()))},
              "skew": {"median": statistics.median(skews) if skews else None,
                       "p75": sorted(skews)[int(.75*len(skews))] if skews else None,
                       "p90": sorted(skews)[int(.9*len(skews))] if skews else None,
                       "hand_median": statistics.median(hand_skews) if hand_skews else None,
                       "hand_p90": sorted(hand_skews)[int(.9*len(hand_skews))] if hand_skews else None,
                       "histogram": dict(sorted(skew_buckets.items()))},
              "fires_by_region": dict(by_region),
              "records": records}
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("count", "with_outline", "fires", "foreshortening", "skew",
                                                   "fires_by_region")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
