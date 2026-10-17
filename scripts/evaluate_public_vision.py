"""Measure the current network on deduplicated public tiles and scene images."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

from app.advice import load_screenshot
from app.capture.frames import load_image, save_image
from app.vision.config import load_config, write_json
from app.vision.pjura import PjuraModel
from app.vision.recognizer import StaticRecognizer, overlay
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public-mahjong"


def metric(rows):
    accepted = [r for r in rows if r["accepted"]]
    correct = sum(r["prediction"]["label"] == r["expected"] for r in rows)
    return {"samples": len(rows), "correct_top1": correct,
            "top1_accuracy": correct / len(rows) if rows else None,
            "accepted": len(accepted), "rejected": len(rows)-len(accepted),
            "accepted_wrong": sum(r["prediction"]["label"] != r["expected"] for r in accepted),
            "accepted_accuracy": sum(r["prediction"]["label"] == r["expected"] for r in accepted)/len(accepted) if accepted else None}


def crops(config, records, output):
    classifier = config["tile_classifier"]
    model = PjuraModel(classifier["manifest"], provider=classifier.get("provider", "CUDAExecutionProvider"),
                       batch_size=classifier.get("batch_size", 16))
    rows = []
    samples = [r for r in records if r["kind"] == "tile_crop" and r.get("label_mjai") and not r["duplicate_of"]]
    for start in range(0, len(samples), 64):
        group = samples[start:start+64]
        predictions = model.predict_many([load_image(DATA/r["path"]) for r in group])
        for record, prediction in zip(group, predictions):
            accepted = prediction["score"] >= classifier["confidence"] and prediction["margin"] >= classifier["margin"]
            rows.append({"path": record["path"], "source": record["source"], "expected": record["label_mjai"],
                         "upstream_label": record["upstream_label"], "prediction": prediction, "accepted": accepted})
        if start % 512 == 0:
            print(f"Network crops: {min(start+64,len(samples))}/{len(samples)}", flush=True)
    summary = {"scope": "Against upstream labels; same-author pjura samples are not an independent model holdout",
               "overall": metric(rows), "by_source": {s: metric([r for r in rows if r["source"]==s]) for s in sorted({r["source"] for r in rows})},
               "confusions": [{"expected": e,"predicted":p,"count":n} for (e,p),n in Counter(
                   (r["expected"],r["prediction"]["label"]) for r in rows if r["expected"]!=r["prediction"]["label"]).most_common()],
               "model_sha256": model.manifest["model_sha256"], "threshold": classifier["confidence"], "margin": classifier["margin"]}
    write_json(output/"crops-summary.json", summary)
    (output/"crops.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
    print(json.dumps(summary["by_source"],ensure_ascii=False),flush=True)


def scenes(config, records, output, limit=None):
    bank = TemplateBank(config["templates"], config.get("classifier_manifest"))
    recognizer = StaticRecognizer(config, bank, TextReader())
    samples = [r for r in records if r.get("scene") == "board" and not r["duplicate_of"]]
    if limit:
        samples = samples[:limit]
    rows = []
    for i, record in enumerate(samples, 1):
        name = f"{i:03d}-"+Path(record["path"]).stem
        try:
            pixels, metadata = load_screenshot(DATA/record["path"], config)
            recognizer.river_memory.clear()
            observation = recognizer.recognize(pixels, metadata)
            write_json(output/"scenes"/name/"observation.json",observation)
            save_image(output/"scenes"/name/"overlay.png",overlay(pixels,observation))
            region_rows = {n:{"kind":r["kind"],"complete":r["complete"],"items":len(r.get("items",[])),
                              "unknown":sum(t.get("label")=="?" for t in r.get("items",[]))} for n,r in observation["regions"].items()}
            row = {"path":record["path"],"scene_tags":record.get("scene_tags",[]),"regions":region_rows,
                   "issues":observation["issues"],"output":str((output/"scenes"/name/"observation.json").relative_to(ROOT))}
        except Exception as exc:
            row = {"path":record["path"],"error":str(exc)}
        rows.append(row)
        if i%5 == 0 or i==len(samples):
            print(f"Scene recognizer: {i}/{len(samples)}",flush=True)
    summary = {"scope":"Unlabelled public scenes: completeness diagnostics, not recognition accuracy",
               "samples":len(rows),"input_errors":sum("error" in r for r in rows),
               "incomplete_regions":dict(Counter(n for row in rows for n,r in row.get("regions",{}).items() if not r["complete"])),
               "rows":rows}
    write_json(output/"scenes-summary.json",summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--parts",nargs="+",choices=["crops","scenes"],default=["crops","scenes"])
    parser.add_argument("--scene-limit",type=int)
    args=parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    config=load_config(ROOT/"configs/window-steam-1366-v2.json")
    records=[json.loads(s) for s in (DATA/"manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    code={}
    for name in ("app/vision/pjura.py","app/vision/rectify.py","app/vision/recognizer.py","app/vision/detector.py",
                 "app/vision/config.py","app/vision/actions.py","app/advice.py","app/vision/sideways.py",
                 "models/sideways-glyph/manifest.json","configs/window-steam-1366-v2.json"):
        src=ROOT/name; dst=args.output/"code"/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        code[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    write_json(args.output/"run.json",{"config_version":config["version"],"code":code,
               "corpus_manifest_sha256":hashlib.sha256((DATA/"manifest.jsonl").read_bytes()).hexdigest()})
    started=time.perf_counter()
    if "crops" in args.parts:crops(config,records,args.output)
    if "scenes" in args.parts:scenes(config,records,args.output,args.scene_limit)
    print(f"Completed in {time.perf_counter()-started:.1f}s",flush=True)


if __name__=="__main__":
    main()
