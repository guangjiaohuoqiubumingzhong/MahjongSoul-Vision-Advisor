"""Score tile classification on unlabeled real gameplay frames.

Unlabeled frames have no per-tile ground truth, so this measures the
distribution of acceptance outcomes and scores instead of accuracy: how often a
visible tile is rejected as unknown, and how the score distribution differs
between tiles the classifier sees upright and tiles it sees sideways.

Only frozen captured pixels are read; nothing is written except the report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, write_json
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

TILE_REGIONS = {"hand", "draw", "river", "dora", "meld", "choices"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--root", type=Path, action="append", required=True,
                        help="directory searched recursively for *.png")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/tile-pose-live-probe.json"))
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

    records, summary = [], defaultdict(Counter)
    for index, image in enumerate(images, 1):
        pixels = load_image(image)
        if list(pixels.shape[1::-1]) != config["client_size"]:
            continue
        observation = recognizer.recognize(pixels, {"image_path": str(image.resolve())})
        for name, region in observation["regions"].items():
            if region.get("kind") not in TILE_REGIONS:
                continue
            summary[name]["items"] += len(region["items"])
            for item in region["items"]:
                horizontal = item.get("orientation_status") == "confirmed_by_template" and item.get("horizontal")
                bucket = "unknown" if item["label"] == "?" else ("horizontal" if horizontal else "upright")
                summary[name][bucket] += 1
                score = (item.get("pjura_evidence") or {}).get("score")
                records.append({"image": str(image), "region": name, "label": item["label"],
                                "bucket": bucket, "score": score,
                                "reason": item.get("reason"),
                                "orientation_status": item.get("orientation_status"),
                                "identity_rotation": item.get("identity_rotation"),
                                "face_rotation": item.get("face_rotation")})
        print(f"{index}/{len(images)} {image.name} items={sum(len(r['items']) for r in observation['regions'].values())}", flush=True)

    by_bucket = {}
    for bucket in ("upright", "horizontal", "unknown"):
        scores = [r["score"] for r in records if r["bucket"] == bucket and r["score"] is not None]
        by_bucket[bucket] = {"count": sum(1 for r in records if r["bucket"] == bucket), "scored": len(scores),
                             "median": statistics.median(scores) if scores else None,
                             "mean": statistics.mean(scores) if scores else None,
                             "min": min(scores) if scores else None,
                             "p10": (sorted(scores)[max(0, len(scores)//10-1)] if scores else None)}
    report = {"scope": "unlabeled_live_frames_not_accuracy_evidence",
              "config_version": config["version"], "images": [str(p) for p in images],
              "by_bucket": by_bucket,
              "by_region": {name: dict(value) for name, value in sorted(summary.items())},
              "reasons": dict(Counter(r["reason"] for r in records if r["label"] == "?" )),
              "records": records}
    write_json(args.output, report)
    print(json.dumps({"by_bucket": by_bucket, "by_region": report["by_region"], "reasons": report["reasons"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
