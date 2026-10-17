"""Measure repeated offline recognition; this is not end-to-end acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform

import cv2
import numpy as np
import onnxruntime as ort

from app.capture.frames import config_digest, load_image
from app.vision.config import load_config, write_json
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/static-timing.json"))
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"]), TextReader())
    loaded = [(path, load_image(path)) for path in args.images]
    signatures = {}
    warmups = []
    for path, pixels in loaded:
        result = recognizer.recognize(pixels)
        signatures[str(path)] = signature(result)
        warmups.append({"image":str(path.resolve()), "recognition_ms":result["recognition_ms"]})
    samples = []
    for run in range(args.runs):
        for path, pixels in loaded:
            result = recognizer.recognize(pixels)
            samples.append({"run":run, "image":str(path.resolve()),
                            "recognition_ms":result["recognition_ms"],
                            "same_classification_as_warmup":signature(result)==signatures[str(path)]})
    times = [sample["recognition_ms"] for sample in samples]
    report = {"scope":"repeated_offline_static_recognition_only_not_end_to_end_or_long_run_acceptance",
              "measured_at":datetime.now(timezone.utc).isoformat(),
              "config_sha256":config_digest(config), "template_manifest_sha256":recognizer.bank.sha256,
              "feature_version":recognizer.bank.feature_version,
              "runtime":{"python":platform.python_version(),"numpy":np.__version__,
                         "opencv":cv2.__version__,"onnxruntime":ort.__version__},
              "images":[{"path":str(path.resolve()),"file_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path,_ in loaded],
              "warmups":warmups, "samples":samples,
              "median_ms":float(np.median(times)),"p95_ms":float(np.quantile(times,.95)),
              "min_ms":min(times),"max_ms":max(times),
              "classification_consistent":all(sample["same_classification_as_warmup"] for sample in samples)}
    write_json(args.output, report)
    print({key:report[key] for key in ("median_ms","p95_ms","min_ms","max_ms","classification_consistent")})


def signature(result):
    """Ignore timing and tiny numeric scores; compare all accepted state fields."""
    return {name:{"complete":region["complete"],
                  "items":[(item["label"],item.get("face_rotation"),item.get("slot")) for item in region["items"]],
                  "text":region.get("text",{}).get("value"),
                  "groups":[(group["labels"],group["type"],group["target"]) for group in region.get("groups",[])]}
            for name,region in result["regions"].items()}


if __name__ == "__main__":
    main()
