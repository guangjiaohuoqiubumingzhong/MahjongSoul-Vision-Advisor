"""Report the distribution of detected tile silhouette proportions.

The correction needs to know what an ordinary tile face measures in this
pipeline, because the outline the detector supplies is the visible body and not
the printed face. This prints the fitted long/short ratio over a sample of
captured frames so that reference and tolerance come from measurement.
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
from app.vision.rectify import min_area_quad, quad_ratio
from app.vision.templates import TemplateBank, rotate
from app.vision.text import TextReader

TILE_REGIONS = {"hand", "draw", "river", "dora", "meld"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/tile-outline-ratio.json"))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    images = []
    for root in args.root:
        images.extend(sorted(root.rglob("*.png")))
    step = max(1, len(images)//args.limit)
    images = images[::step][:args.limit]
    ratios, by_region = [], {}
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
                geometry = min_area_quad(patch) if patch.size else None
                if geometry is None:
                    continue
                ratios.append(geometry.aspect)
                by_region.setdefault(name, []).append(geometry.aspect)
        print(f"{index}/{len(images)} {image.name} n={len(ratios)}", flush=True)

    ordered = sorted(ratios)
    report = {"scope": "measured_geometry_only", "count": len(ordered),
              "median": statistics.median(ordered) if ordered else None,
              "mean": statistics.mean(ordered) if ordered else None,
              "p05": ordered[int(.05*len(ordered))] if ordered else None,
              "p10": ordered[int(.10*len(ordered))] if ordered else None,
              "p90": ordered[int(.90*len(ordered))] if ordered else None,
              "p95": ordered[min(len(ordered)-1, int(.95*len(ordered)))] if ordered else None,
              "deciles": {f"{index/10:.1f}": ordered[min(len(ordered)-1, int(index/10*len(ordered)))]
                          for index in range(1, 10)} if ordered else {},
              "by_region_median": {name: statistics.median(values) for name, values in sorted(by_region.items())},
              "histogram": dict(sorted(Counter(f"{min(int(value*10)/10, 4.0):.1f}" for value in ordered).items())),
              "images": [str(path) for path in images]}
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("count", "median", "p05", "p10", "p90", "p95", "deciles",
                                                  "by_region_median", "histogram")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
