"""Run the three reported screenshots through the review path to a decision.

The user-facing fallback for an unreadable highlight is the review dialog: it
sets the responder, the allowed calls and, after a call, which meld was just
completed. This runs the same context through the whole chain (recognition ->
snapshot -> Mortal) so the fallback is known to reach a recommendation.

Run: python -m scripts.verify_call_review_path
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
# oya: the dealer; target: the seat whose discard is being answered.
CHI_CONTEXT = {"oya": 2, "phase": "reaction", "target": 3, "allowed_actions": ["pass", "chi"]}
RON_CONTEXT = {"oya": 2, "phase": "reaction", "target": 1, "allowed_actions": ["pass", "pon"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--json", type=Path, default=ROOT / "reports/no-draw-call-discard/review-path.json")
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    from app.engine.runtime import OnnxRuntime
    from app.engine.snapshot import SnapshotBackend
    backend = SnapshotBackend(OnnxRuntime())
    cases = [("chi-buttons-1920", CHI_CONTEXT), ("ron-buttons-1920", RON_CONTEXT)]
    report = []
    for name, context in cases:
        pixels = load_image(FIXTURES / f"{name}.png")
        observation = recognizer.recognize(pixels, {})
        entry = {"fixture": name, "context": context}
        try:
            snapshot, allowed = build_snapshot(observation, config, {}, context)
            decision = backend.decide(snapshot, allowed)
            entry.update(phase=snapshot["phase"], target=snapshot["target"],
                         reaction_tile=snapshot["reaction_tile"], allowed=allowed,
                         recommendation=decision.get("recommendation"),
                         legal=[c["semantic"] for c in decision.get("legal_candidates", [])])
        except SnapshotError as exc:
            entry["blocked"] = str(exc)
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
