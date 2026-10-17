"""Report which frozen-test tiles change when the top-face correction is enabled.

Runs the full recognizer twice on the reviewed test games: tile classifier
rectification ``off`` (original crops) and ``both`` (original + perspective
corrected top face), on the same reviewed, ground-truthed test games the
acceptance report uses.

Reports are written to dedicated files so the project's acceptance reports are
left alone.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, write_json
from app.vision.evaluate import iou
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader


def test_annotations():
    paths = []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if value.get("reviewed") and value.get("split") == "test" and value.get("frame_quality") == "stable":
            paths.append(path)
    return paths


def region_pose(config):
    """Region name to its quarter-turn rotation, so results can be grouped."""
    return {name: region.get("rotation", 0) for name, region in config["regions"].items()}


def by_group(records, pose):
    groups = {}
    for record in records:
        if record["ground_truth"] == "?":
            continue
        key = (record["region"], f"rotation={pose.get(record['region'], 0)}",
               "horizontal" if record.get("identity_rotation") in (90, 270) else "upright")
        groups.setdefault(key, []).append(record)
    result = {}
    for key, values in sorted(groups.items()):
        correct = sum(record["accepted"] == record["ground_truth"] for record in values)
        unknown = sum(record["accepted"] == "?" for record in values)
        result["/".join(key)] = {"tiles": len(values), "correct": correct, "unknown": unknown,
                                 "wrong": len(values) - correct - unknown,
                                 "corrected": sum(record["rectification"] is not None for record in values)}
    return result


def recognised_tiles(recogniser, paths):
    """Map every matched tile to its ground truth and accepted label."""
    found = []
    for path in paths:
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent / image
        observation = recogniser.recognize(load_image(image), {"image_path": str(image.resolve())})
        for target in annotation["targets"]:
            if target["kind"] != "tile":
                continue
            region = observation["regions"].get(target["region"])
            if region is None:
                continue
            best, best_iou = None, 0.0
            for item in region["items"]:
                value = iou(target["bbox"], item["bbox"])
                if value > best_iou:
                    best, best_iou = item, value
            if best is None or best_iou < .5:
                continue
            found.append({"annotation": str(path), "region": target["region"],
                          "ground_truth": target["label"], "accepted": best["label"],
                          "score": (best.get("pjura_evidence") or {}).get("score"),
                          "rectification": (best.get("pjura_evidence") or {}).get("rectification"),
                          "view": (best.get("pjura_evidence") or {}).get("view"),
                          "identity_rotation": best.get("identity_rotation")})
    return found


def metrics(records):
    tiles = [record for record in records if record["ground_truth"] != "?"]
    correct = sum(record["accepted"] == record["ground_truth"] for record in tiles)
    unknown = sum(record["accepted"] == "?" for record in tiles)
    scores = [record["score"] for record in tiles if record["score"] is not None]
    return {"tiles": len(tiles), "correct": correct, "unknown": unknown,
            "wrong": len(tiles) - correct - unknown,
            "accuracy": correct / len(tiles) if tiles else None,
            "scores_median": sorted(scores)[len(scores)//2] if scores else None,
            "scores_p10": sorted(scores)[max(0, len(scores)//10 - 1)] if scores else None,
            "corrected": sum(record["rectification"] is not None for record in tiles)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/vision/top-face-corpus-evidence.json"))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    paths = test_annotations()
    config = load_config(args.config)
    reader = TextReader()
    result = {"scope": "frozen_reviewed_test_games_only", "config_version": config["version"],
              "annotations": [str(path) for path in paths], "runs": {}, "changed_tiles": {}}
    runs = {}
    for name in ("off", "both"):
        variant = deepcopy(config)
        variant["tile_classifier"] = {**variant["tile_classifier"], "rectification": name}
        recogniser = StaticRecognizer(variant, TemplateBank(variant["templates"],
                                                           variant.get("classifier_manifest")), reader)
        runs[name] = recognised_tiles(recogniser, paths)
        result["runs"][name] = {**metrics(runs[name]), "rectification": name,
                                "by_region_pose": by_group(runs[name], region_pose(config))}
        print(f"{name}: {result['runs'][name]['accuracy']} unknown={result['runs'][name]['unknown']} "
              f"corrected={result['runs'][name]['corrected']}", flush=True)
    by_key = {(record["annotation"], record["region"], record["ground_truth"]): index
              for index, record in enumerate(runs["off"])}
    changed = []
    for index, record in enumerate(runs["both"]):
        key = (record["annotation"], record["region"], record["ground_truth"])
        if key not in by_key:
            continue
        baseline = runs["off"][by_key[key]]
        if baseline["accepted"] != record["accepted"]:
            changed.append({"region": record["region"], "ground_truth": record["ground_truth"],
                            "off_accepted": baseline["accepted"], "off_score": baseline["score"],
                            "both_accepted": record["accepted"], "both_score": record["score"]})
    result["changed_tiles"] = {"count": len(changed), "records": changed[:40]}
    write_json(args.output, result)
    print(json.dumps({"runs": result["runs"], "changed": len(changed)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
