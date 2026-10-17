"""Evaluate a fixed optional SVM candidate without changing the CLI default."""
from pathlib import Path
import argparse
import json

from app.vision.config import load_config, write_json
from app.vision.evaluate import evaluate
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--back-bank",action="store_true")
    parser.add_argument("--highlight-fallback",action="store_true")
    args = parser.parse_args()
    model_report = "test-svm-highlight-prototype.json" if args.highlight_fallback else "test-svm-prototype.json"
    fixed = json.loads((Path("reports/vision")/model_report).read_text(encoding="utf-8"))["manifest_sha256"]
    manifest = Path(f"models/vision-svm-prototype/manifest-{fixed}.json")
    baseline_path = "configs/history/steam-1366x768-default-draft-6-be1d5f4ccfa56d629411aefe1c7935493cb33ce117466db0a96507b19186fb9d.json"
    config = load_config(baseline_path)
    annotations = []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if value.get("reviewed"):
            annotations.append(path)
    reader = TextReader()
    baseline = evaluate(StaticRecognizer(config,TemplateBank(config["templates"]),reader),annotations)
    write_json("reports/vision/draft6-baseline-acceptance.json",baseline)
    if args.highlight_fallback:
        name = "svm-highlight-candidate"
        config["version"] = "steam-1366x768-svm-highlight-candidate-3"
        config["templates"] = "assets/templates/steam-highlight-candidate/manifest.json"
    else:
        name = "svm-backs-candidate" if args.back_bank else "svm-candidate"
        config["version"] = "steam-1366x768-svm-backs-candidate-2" if args.back_bank else "steam-1366x768-svm-candidate-1"
    if args.back_bank and not args.highlight_fallback:
        config["templates"] = "assets/templates/steam-backs-candidate/manifest.json"
    config["classifier_manifest"] = str(manifest)
    write_json(f"configs/window-steam-1366-{name}.json",config)
    bank = TemplateBank(config["templates"],manifest)
    candidate = evaluate(StaticRecognizer(config,bank,reader),annotations)
    write_json(f"reports/vision/{name}-acceptance.json",candidate)
    for label,result in [("baseline",baseline),("candidate",candidate)]:
        print(label,{"dataset":result["dataset_images"],"tiles":result["dataset_known_tiles"],
                     "classes":len(result["test_class_coverage"]),"regions":result["regions"],
                     "text":result["text"],"melds":result["melds"],"uncertain":result["uncertain_frames"],
                     "whole":result["whole_scene_accuracy"],"gate_passed":result["gate_passed"]})


if __name__ == "__main__":
    main()
