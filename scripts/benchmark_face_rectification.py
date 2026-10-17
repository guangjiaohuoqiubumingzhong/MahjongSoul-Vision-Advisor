"""Compare pjura on original crops vs perspective-corrected top faces.

Every reviewed annotation's tile boxes are cropped exactly as the recognizer
does (region crop rotated by the region's quarter turn), then classified with
``PjuraBank`` in the three rectification modes:

* ``off``       - original crops only (behaviour before the correction)
* ``rectified`` - corrected top face when it can be fitted
* ``both``      - both views, best wins, confident disagreement -> unknown

The detector is bypassed (annotation boxes are used) so the numbers isolate
the classifier input. Accepted labels must match exactly, so a red five read
as an ordinary five (or the reverse) counts as wrong; ``top1`` compares the
network's base tile only.

    .venv\\Scripts\\python scripts\\benchmark_face_rectification.py
    .venv\\Scripts\\python scripts\\benchmark_face_rectification.py --split test --atlas reports\\vision\\face-rectification-atlas.png
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.state.events import base
from app.vision.config import load_config, write_json
from app.vision.pjura import PjuraBank, PjuraModel
from app.vision.rectify import top_face_outline, warp_top_face
from app.vision.templates import rotate

MODES = ("off", "rectified", "both")


class NoLegacy:
    """Stand-in template bank: never claims a back or a colour."""

    sha256 = None

    @staticmethod
    def classify(*args, **kwargs):
        return {"label": "?", "score": None, "margin": None, "face_rotation": 0}


def annotations(split):
    for path in sorted(Path("data").rglob("*.annotation.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not value.get("reviewed") or value.get("frame_quality") != "stable":
            continue
        if split != "all" and value.get("split") != split:
            continue
        image = Path(value["image"])
        if not image.is_file():
            image = path.with_name(path.name.replace(".annotation.json", ".png"))
        if image.is_file():
            yield path, value, image


def tile_crops(config, split):
    for path, annotation, image_path in annotations(split):
        pixels = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            continue
        for target in annotation["targets"]:
            region = config["regions"].get(target.get("region"))
            if target.get("kind") != "tile" or region is None or target["label"] == "?":
                continue
            x, y, w, h = target["bbox"]
            crop = pixels[max(0, y):y+h, max(0, x):x+w]
            if crop.size == 0:
                continue
            degrees = region.get("rotation", 0)
            rotations = (0, 90, 270) if region["kind"] in {"river", "meld", "choices"} else (0,)
            yield {"annotation": str(path), "id": target.get("id"), "region": target["region"],
                   "kind": region["kind"], "label": target["label"], "horizontal": target.get("horizontal"),
                   "crop": rotate(crop, degrees), "rotations": rotations}


def judge(result, truth):
    evidence = result.get("pjura_evidence") or {}
    best = result.get("best_label")
    accepted = result["label"] != "?"
    return {"best": best, "label": result["label"], "score": result.get("score"), "reason": result["reason"],
            "view": evidence.get("view"), "views": evidence.get("views"),
            "top1_correct": best is not None and base(best) == base(truth),
            "accepted": accepted, "accepted_correct": accepted and result["label"] == truth,
            "accepted_wrong": accepted and result["label"] != truth}


def summarise(records):
    groups = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[record["region"]].append(record)
        groups["table" if record["kind"] in {"river", "meld"} else record["kind"]].append(record)
    summary = {}
    for name, values in sorted(groups.items()):
        scores = sorted(record["score"] for record in values if record["score"] is not None)
        summary[name] = {
            "tiles": len(values),
            "top1_accuracy": sum(record["top1_correct"] for record in values) / len(values),
            "accepted_correct": sum(record["accepted_correct"] for record in values),
            "unknown": sum(not record["accepted"] for record in values),
            "accepted_wrong": sum(record["accepted_wrong"] for record in values),
            "score_median": scores[len(scores)//2] if scores else None,
            "score_p10": scores[max(0, len(scores)//10 - 1)] if scores else None,
            "face_fit": sum(record.get("fitted", False) for record in values),
        }
    return summary


def write_atlas(path, tiles, limit=160):
    cells = []
    for tile in tiles[:limit]:
        crop = tile["crop"]
        geometry = top_face_outline(crop)
        turns = (90,) if tile.get("horizontal") else (0,)
        panel = [crop] + ([warp_top_face(crop, geometry, turns[0]).pixels] if geometry else [])
        cell = np.full((112, 150, 3), 32, np.uint8)
        left = 2
        for image in panel:
            scale = min(70 / image.shape[1], 96 / image.shape[0])
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            cell[14:14+resized.shape[0], left:left+resized.shape[1]] = resized
            left += 74
        cv2.putText(cell, f"{tile['region'][:10]} {tile['label']}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (0, 255, 255), 1)
        cells.append(cell)
    while len(cells) % 8:
        cells.append(np.full((112, 150, 3), 32, np.uint8))
    rows = [np.hstack(cells[i:i+8]) for i in range(0, len(cells), 8)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-pjura.json"))
    parser.add_argument("--split", default="all", choices=("all", "train", "test"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/face-rectification-benchmark.json"))
    parser.add_argument("--atlas", type=Path, default=None)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    classifier = dict(config["tile_classifier"])
    if args.provider:
        classifier["provider"] = args.provider
    model = PjuraModel(classifier["manifest"], provider=classifier.get("provider", "CUDAExecutionProvider"),
                       batch_size=classifier.get("batch_size", 16))
    tiles = list(tile_crops(config, args.split))
    for tile in tiles:
        tile["fitted"] = top_face_outline(tile["crop"]) is not None
    report = {"scope": "annotation_boxes_classifier_only", "split": args.split,
              "config_version": config["version"], "tiles": len(tiles), "modes": {}, "changed": []}
    by_mode = {}
    for mode in MODES:
        bank = PjuraBank(NoLegacy(), {**classifier, "rectification": mode}, model=model)
        records = []
        for tile in tiles:
            result = bank.classify_many([tile["crop"]], tile["region"], 0, 0, tile["rotations"],
                                        kinds=("tile",))[0]
            records.append({"region": tile["region"], "kind": tile["kind"], "label": tile["label"],
                            "id": tile["id"], "annotation": tile["annotation"], "fitted": tile["fitted"],
                            **judge(result, tile["label"])})
        by_mode[mode] = records
        report["modes"][mode] = summarise(records)
        overall = report["modes"][mode]["all"]
        print(f"{mode:9s} top1={overall['top1_accuracy']:.3f} accepted_correct={overall['accepted_correct']} "
              f"unknown={overall['unknown']} accepted_wrong={overall['accepted_wrong']} "
              f"median={overall['score_median']}", flush=True)
    for old, new in zip(by_mode["off"], by_mode["both"]):
        if (old["best"], old["accepted"]) != (new["best"], new["accepted"]):
            report["changed"].append({"annotation": old["annotation"], "id": old["id"], "region": old["region"],
                                      "truth": old["label"], "off": [old["best"], old["score"], old["reason"]],
                                      "both": [new["best"], new["score"], new["reason"], new["view"]]})
    write_json(args.output, report)
    if args.atlas:
        write_atlas(args.atlas, [tile for tile in tiles if tile["kind"] != "hand"])
    print(json.dumps({mode: {key: report["modes"][mode][key] for key in ("table", "all") if key in report["modes"][mode]}
                      for mode in MODES}, ensure_ascii=False, indent=2))
    print(f"changed tiles off->both: {len(report['changed'])}; report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
