"""Report the enabled pjura path against the frozen template baseline; no tuning."""
from pathlib import Path
import argparse
import json

from app.capture.frames import load_image, save_image
from app.vision.config import load_config, write_json
from app.vision.evaluate import evaluate
from app.vision.recognizer import StaticRecognizer, overlay
from app.vision.templates import TemplateBank
from app.vision.text import TextReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, required=True)
    args = parser.parse_args()
    annotations = []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if value.get("reviewed"):
            annotations.append(path)
    reader = TextReader()
    baseline_config = load_config(args.baseline_config)
    baseline = StaticRecognizer(baseline_config, TemplateBank(baseline_config["templates"], baseline_config.get("classifier_manifest")), reader)
    baseline_report = evaluate(baseline, annotations)
    write_json("reports/vision/pjura-template-baseline.json", baseline_report)
    config = load_config("configs/window-steam-1366-v2.json")
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), reader)
    report = evaluate(recognizer, annotations)
    report.update(integration_policy="enabled_by_explicit_user_request_regardless_of_test_accuracy",
                  evaluation_scope="local_test_games_held_out_from_local_training; upstream_training_game_overlap_unknown",
                  local_test_used_for_training=False)
    write_json("reports/vision/pjura-acceptance.json", report)
    write_json("reports/stage2-acceptance.json", report)
    examples = []
    for path in annotations:
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if annotation.get("split") != "test":
            continue
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent / image
        pixels = load_image(image)
        frame_path = image.with_suffix(".frame.json")
        metadata = json.loads(frame_path.read_text(encoding="utf-8-sig")) if frame_path.exists() else {}
        metadata["image_path"] = str(image.resolve())
        observation = recognizer.recognize(pixels, metadata)
        output = Path("reports/vision/pjura-test") / f"{len(examples)+1:02d}" / "observation.json"
        write_json(output, observation)
        save_image(output.with_suffix(".overlay.png"), overlay(pixels, observation))
        examples.append({"annotation": str(path), "image": str(image), "output": str(output)})
    write_json("reports/vision/pjura-test/index.json", examples)
    summary = {"enabled": True, "config_version": config["version"],
               "tile_classifier": recognizer.bank.tile_classifier_info,
               "baseline": {key: baseline_report[key] for key in ("regions", "melds", "text", "whole_scene_accuracy", "gate_passed")},
               "pjura": {key: report[key] for key in ("regions", "melds", "text", "whole_scene_accuracy", "gate_passed")},
               "test_images": len(examples), "stable_test_images": report["stable_test_frames"],
               "accepted_wrong_labels": sum(error.get("category") == "tile_classification_or_slot"
                                            and error["actual"]["label"] not in {"?", error["expected"]["label"]}
                                            for error in report["errors"]),
               "test_used_for_training": False,
               "limitations": "small local held-out sample; upstream training overlap unknown; orientation correction followed first-pass diagnosis"}
    write_json("reports/vision/pjura-comparison.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
