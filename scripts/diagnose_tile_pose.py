"""Diagnose sideways/perspective tile classification without tuning anything.

For every reviewed test annotation this script records, per visible tile:
the ground-truth label and rotation, the detector box, whether the region was
rotated, the accepted label, and the classifier scores with and without the
90/270 rotation candidates. It writes a report and prints a compact summary.

The script only reads frozen test games; it never modifies configs or models.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, write_json
from app.vision.evaluate import iou
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader


def reviewed_annotations():
    paths = []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if value.get("reviewed"):
            paths.append(path)
    return paths


def classifier_evidence(item):
    legacy = item.get("legacy_evidence") or {}
    pjura = item.get("pjura_evidence") or {}
    if not pjura and "pjura_evidence" not in item:
        legacy = item if "best_label" in item else legacy
    return {"accepted": item.get("label"),
            "reason": item.get("reason"),
            "identity_rotation": item.get("identity_rotation"),
            "face_rotation": item.get("face_rotation"),
            "orientation_confirmed": item.get("orientation_confirmed"),
            "pjura": {key: pjura.get(key) for key in ("label", "score", "margin", "alternatives")},
            "legacy": {key: legacy.get(key) for key in ("label", "best_label", "score", "margin", "face_rotation",
                                                        "reason", "feature_scope", "alternatives")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/vision/tile-pose-diagnosis.json"))
    parser.add_argument("--only-region", default="")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    records, totals = [], Counter()
    for path in reviewed_annotations():
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if annotation.get("split") != "test" or annotation.get("frame_quality") != "stable":
            continue
        if not annotation.get("count_as_distinct_scene", True):
            continue
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent / image
        pixels = load_image(image)
        observation = recognizer.recognize(pixels, {"image_path": str(image.resolve())})
        for target in annotation["targets"]:
            if target["kind"] != "tile" or target.get("rotation", 0) in (180,):
                continue
            region_name = target["region"]
            if args.only_region and args.only_region not in region_name:
                continue
            region = observation["regions"].get(region_name)
            if region is None:
                continue
            best, best_iou = None, 0.0
            for item in region["items"]:
                value = iou(target["bbox"], item["bbox"])
                if value > best_iou:
                    best, best_iou = item, value
            totals["expected"] += 1
            record = {"annotation": str(path), "region": region_name,
                      "region_rotation": region.get("rotation"),
                      "ground_truth": target["label"], "target_rotation": target.get("rotation", 0),
                      "matched": best_iou >= .5, "iou": round(best_iou, 3),
                      "detected_count": len(region["items"])}
            if best is not None and best_iou >= .5:
                record.update(classifier_evidence(best))
                totals["matched"] += 1
                totals["correct" if record["accepted"] == target["label"] else "wrong"] += 1
                if record["accepted"] == "?":
                    totals["unknown"] += 1
            records.append(record)
    by_region = defaultdict(Counter)
    for record in records:
        by_region[record["region"]].update({("correct" if record.get("accepted") == record["ground_truth"]
                                            else "unknown" if record.get("accepted") == "?"
                                            else "wrong" if record["matched"] else "unmatched"): 1})
    report = {"scope": "diagnosis_only_not_acceptance", "config_version": config["version"],
              "tile_classifier": getattr(recognizer.bank, "tile_classifier_info", None),
              "totals": dict(totals),
              "by_region": {name: dict(value) for name, value in sorted(by_region.items())},
              "records": records}
    write_json(args.output, report)
    print(json.dumps({"totals": report["totals"], "by_region": report["by_region"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
