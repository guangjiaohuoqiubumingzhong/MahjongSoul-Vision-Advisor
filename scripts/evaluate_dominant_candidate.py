"""Replay the majority/runner-up rule on fixed public crops and user frames."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2

from app.advice import ScreenshotAdvice, load_screenshot
from app.capture.frames import load_image
from app.vision.acceptance import candidate_acceptance
from app.vision.config import load_config, write_json
from app.vision.pjura import PjuraModel, file_sha256
from scripts.evaluate_meld_corpus import evaluate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/dominant-candidate"


def crop_comparison(bank, config):
    dominant = config["tile_classifier"]["dominant_candidate"]
    original = PjuraModel(ROOT / "models/pjura-vision/manifest.json", dominant_candidate=dominant)
    original.warmup()
    models = {"original_fp32": original, "accelerated_guarded": bank.model}
    records = [json.loads(line) for line in (ROOT / "data/public-mahjong/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r["kind"] == "tile_crop" and r.get("label_mjai") and not r["duplicate_of"]]
    metrics = {name: {"samples": 0, "top1_correct": 0, "old_accepted": 0, "new_accepted": 0,
                     "old_wrong": 0, "new_wrong": 0, "ms": 0.} for name in models}
    differences, wrong, rows = [], [], []
    for start in range(0, len(records), 64):
        group = records[start:start+64]
        crops = [load_image(ROOT / "data/public-mahjong" / r["path"]) for r in group]
        predictions = {}
        for name, model in models.items():
            started = time.perf_counter()
            predictions[name] = model.predict_many(crops)
            metric = metrics[name]
            metric["ms"] += (time.perf_counter()-started)*1000
            for record, p in zip(group, predictions[name]):
                old = candidate_acceptance(p, .9, .04) is not None
                new = candidate_acceptance(p, .9, .04, dominant) is not None
                correct = p["label"] == record["label_mjai"]
                metric["samples"] += 1
                metric["top1_correct"] += correct
                metric["old_accepted"] += old
                metric["new_accepted"] += new
                metric["old_wrong"] += old and not correct
                metric["new_wrong"] += new and not correct
                row = {"path": record["path"], "expected": record["label_mjai"], "model": name,
                       "old_accepted": old, "new_accepted": new, **p}
                rows.append(row)
                if new and not correct: wrong.append(row)
        for record, old, new in zip(group, predictions["original_fp32"], predictions["accelerated_guarded"]):
            if (old["label"] != new["label"] or
                bool(candidate_acceptance(old, .9, .04, dominant)) != bool(candidate_acceptance(new, .9, .04, dominant))):
                differences.append({"path": record["path"], "original": old, "accelerated": new})
        if start % 512 == 0: print("CROPS", min(start+64, len(records)), len(records), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    return {"scope": "Direct tile-crop classification against upstream labels; not an independent holdout, no geometry/colour checks.",
            "models": {name: model.manifest["model_sha256"] for name, model in models.items()},
            "metrics": metrics, "precision_differences": differences, "accepted_wrong": wrong,
            "fp32_rechecks": bank.model.precision_rechecks}


def clear(service):
    service.clear()
    service.recognizer.river_memory.clear()
    bank = service.recognizer.bank
    bank._cache.clear()
    bank._classification_cache.clear()
    service.recognizer.text_reader._read_cache.clear()


def scene_comparison(service, config):
    bank = service.recognizer.bank
    dominant = deepcopy(bank.dominant_candidate)
    paths = [ROOT / "data/fixtures/dominant-candidate/client-1920.png",
             ROOT / "data/fixtures/kan-recognition/top-C-pon-arrow-1920.png",
             ROOT / "data/fixtures/river-left-late/client-1920.png"]
    rows = []
    for path in paths:
        original, metadata = load_screenshot(path, config)
        for size in ((1920, 1080), (1366, 768)):
            pixels = original if size == (1920, 1080) else cv2.resize(original, size, interpolation=cv2.INTER_AREA)
            result_row = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path), "size": size}
            for name, policy in (("old", None), ("new", dominant)):
                bank.dominant_candidate = policy
                if policy is None: bank.tile_classifier_info.pop("dominant_candidate", None)
                else: bank.tile_classifier_info["dominant_candidate"] = deepcopy(policy)
                clear(service)
                result = service.process(pixels, metadata)
                destination = f"{path.parent.name}-{path.stem}-{size[0]}-{name}.json"
                write_json(OUT / destination, result)
                result_row[name] = {"status": result["status"], "reason": result["reason"],
                    "ms": result["total_ms"], "recommendation": (result.get("recommendation") or {}).get("event"),
                    "corrections": result["corrections"], "context": result["context"],
                    "incomplete_regions": [n for n, r in result["observation"]["regions"].items() if not r["complete"]]}
            rows.append(result_row)
            print("SCENE", path.name, size, result_row["old"]["status"], result_row["new"]["status"], flush=True)
    bank.dominant_candidate = dominant
    bank.tile_classifier_info["dominant_candidate"] = deepcopy(dominant)
    return rows


def corpus_comparison(bank, config):
    manifest_path = ROOT / "data/meld-orientation/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dominant = deepcopy(bank.dominant_candidate)
    rows = []
    for case in manifest["cases"]:
        row = {"name": case["name"]}
        for name, policy in (("old", None), ("new", dominant)):
            bank.dominant_candidate = policy
            bank._classification_cache.clear()
            evaluated, _, _ = evaluate(config, bank, case)
            row[name] = evaluated
        rows.append(row)
    bank.dominant_candidate = dominant
    summary = {}
    for name in ("old", "new"):
        values = [row[name] for row in rows]
        summary[name] = {"correct_groups": sum(r["groups_correct"] for r in values),
            "correct_positive_regions": sum(r["correct"] for r in values if r["expected_groups"]),
            "wrong_confirmations": sum(r["wrong_confirmation"] for r in values),
            "safe_negative_regions": sum(r["safe_negative"] for r in values if not r["expected_groups"])}
    return {"manifest_sha256": file_sha256(manifest_path), "scope": manifest["scope"], "summary": summary, "rows": rows}


def main():
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    service = ScreenshotAdvice(config)
    service.prepare()
    report = {"measured_at": datetime.now(timezone.utc).isoformat(), "version": config["version"],
        "rule": config["tile_classifier"]["dominant_candidate"],
        "source_hashes": {p: file_sha256(ROOT / p) for p in ("app/vision/actions.py", "app/vision/acceptance.py", "app/vision/pjura.py")}}
    report["scenes"] = scene_comparison(service, config)
    report["public_crops"] = crop_comparison(service.recognizer.bank, config)
    report["meld_corpus"] = corpus_comparison(service.recognizer.bank, config)
    write_json(OUT / "report.json", report)
    print("PUBLIC", report["public_crops"]["metrics"], flush=True)
    print("MELDS", report["meld_corpus"]["summary"], flush=True)
    print("PRECISION_DIFFERENCES", len(report["public_crops"]["precision_differences"]), flush=True)


if __name__ == "__main__": main()
