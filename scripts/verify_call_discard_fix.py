"""Verify the reported call/response cases against the real screenshots.

Run: python -m scripts.verify_call_discard_fix [--backend]

Uses the client-area fixtures built from the user's whole-window screenshots by
scripts/build_call_discard_fixtures.py. The self-wind glyph is not readable on
these captures (the panel lets the user set the dealer), so the dealer is
supplied here explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.capture.frames import load_image
from app.state.snapshot import build_snapshot, SnapshotError
from app.vision.config import load_config
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/fixtures/no-draw-call-discard"
CASES = ["chi-buttons-1920", "ron-buttons-1920", "after-chi-no-advice-1920"]
CONTEXT = {"oya": 2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--backend", action="store_true")
    parser.add_argument("--json", type=Path, default=ROOT / "reports/no-draw-call-discard/verify.json")
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    backend = None
    if args.backend:
        from app.engine.runtime import OnnxRuntime
        from app.engine.snapshot import SnapshotBackend
        backend = SnapshotBackend(OnnxRuntime())
    report = []
    for name in CASES:
        pixels = load_image(FIXTURES / f"{name}.png")
        observation = recognizer.recognize(pixels, {})
        regions = observation["regions"]
        entry = {"fixture": name, "recognition": observation["status"],
                 "timer": (observation.get("action_timer") or {}).get("raw_text"),
                 "buttons": [item["label"] for item in regions["buttons"]["items"]],
                 "hand": [item["label"] for item in regions["hand"]["items"]],
                 "draw": [item["label"] for item in regions["draw"]["items"]],
                 "reaction_discard": {k: v for k, v in observation["reaction_discard"].items()
                                      if k != "glowing_tiles"}}
        try:
            snapshot, allowed = build_snapshot(observation, config, {}, CONTEXT)
            entry["phase"] = snapshot["phase"]
            entry["target"] = snapshot["target"]
            entry["reaction_tile"] = snapshot["reaction_tile"]
            entry["last_call"] = snapshot.get("last_call")
            entry["allowed"] = allowed
            if backend is not None:
                decision = backend.decide(snapshot, allowed)
                entry["recommendation"] = decision.get("recommendation")
        except SnapshotError as exc:
            entry["blocked"] = str(exc)
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
