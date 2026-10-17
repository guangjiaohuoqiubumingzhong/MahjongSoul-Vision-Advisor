"""Reproduce the three reported call/response screenshots end to end.

Each fixture is a whole-window screenshot; the client crop below aligns the
calibrated region rectangles with the game. For every fixture the script
prints the recognition summary, the snapshot result and, when the backend is
available, the advice.

Usage: python -m scripts.repro_call_discard [--backend] [--json reports/...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.capture.frames import load_image
from app.state.snapshot import build_snapshot, SnapshotError
from app.vision.config import load_config
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]

# Image -> client crop (left, top, right, bottom) in source pixels. The game
# window's client area starts below its title bar; the advice panel to the
# right is excluded.
# Whole-window screenshots are cropped to the client area and resampled by
# scripts/build_call_discard_fixtures.py; the *-1920.png files are ready to
# analyse directly.
FIXTURES = [
    ("chi-buttons-1920", None),
    ("ron-buttons-1920", None),
    ("after-chi-no-advice-1920", None),
    ("fig1-after-chi-no-advice", (0, 26, 1912, 1092)),
    ("fig2-chi-buttons", (0, 26, 1912, 1092)),
    ("fig3-ron-buttons", (0, 26, 1912, 1092)),
]


def summarise(observation):
    return {name: {"kind": r["kind"], "complete": r["complete"], "player": r.get("player"),
                   "items": [i["label"] for i in r.get("items", [])],
                   "groups": [{k: g.get(k) for k in ("labels", "type", "target", "called_positions", "status")}
                              for g in (r.get("groups") or [])],
                   "text": (r.get("text") or {}).get("raw_text")}
            for name, r in observation["regions"].items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "data/fixtures/no-draw-call-discard")
    parser.add_argument("--backend", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    backend = None
    if args.backend:
        from app.engine.runtime import OnnxRuntime
        from app.engine.snapshot import SnapshotBackend
        backend = SnapshotBackend(OnnxRuntime())
    report = []
    for stem, crop in FIXTURES:
        if args.only and args.only not in stem:
            continue
        path = args.fixtures / f"{stem}.png"
        pixels = load_image(path)
        if crop is not None:
            l, t, r, b = crop
            client = pixels[t:b, l:r]
        else:
            client = pixels
        observation = recognizer.recognize(client, {"image_path": str(path.resolve())})
        entry = {"fixture": stem, "crop": list(crop) if crop else None, "status": observation["status"],
                 "issues": observation["issues"], "regions": summarise(observation),
                 "reaction_discard": observation.get("reaction_discard"),
                 "timer": observation.get("action_timer")}
        print(f"== {stem}: recognition {observation['status']}")
        print("   reaction:", json.dumps(entry["reaction_discard"], ensure_ascii=False))
        print("   buttons:", [i["label"] for i in observation["regions"]["buttons"]["items"]])
        print("   issues:", json.dumps(observation["issues"], ensure_ascii=False))
        try:
            snapshot, allowed = build_snapshot(observation, config)
            entry["snapshot"] = snapshot
            entry["allowed"] = allowed
            print("   snapshot phase:", snapshot["phase"], "hand", len(snapshot["hand"]),
                  "draw", snapshot["draw"], "target", snapshot["target"],
                  "melds", [[g.get("type") for g in m] for m in snapshot["melds"]])
            if backend is not None:
                decision = backend.decide(snapshot, allowed)
                entry["decision"] = decision
                print("   advice:", json.dumps(decision.get("recommendation"), ensure_ascii=False))
        except SnapshotError as exc:
            entry["snapshot_error"] = str(exc)
            print("   snapshot error:", exc)
        report.append(entry)
        cv2.imwrite(str(ROOT / f"reports/no-draw-call-discard/{stem}-client.png"), client)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
