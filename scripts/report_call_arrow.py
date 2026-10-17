"""Report the call-arrow detection on every real fixture."""
from __future__ import annotations

import json
from pathlib import Path

from app.capture.frames import load_image
from app.vision.actions import arrow_target, call_arrow, glow_mask
from app.vision.config import load_config
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = [
    "screenshot-response/chi.png",
    "screenshot-response/ron.png",
    "white-meld/turn-58.png",
    "river-fragments/left-7p-1920.png",
    "no-draw-call-discard/chi-buttons-1920.png",
    "no-draw-call-discard/ron-buttons-1920.png",
    "no-draw-call-discard/after-chi-no-advice-1920.png",
]


def main() -> int:
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    for name in FIXTURES:
        pixels = load_image(ROOT / "data/fixtures" / name)
        observation = recognizer.recognize(pixels, {})
        mask = glow_mask(pixels)
        arrow = call_arrow(mask)
        target = arrow_target(arrow, observation["regions"], 0) if arrow else None
        print("==", name)
        print("   arrow:", json.dumps(arrow, ensure_ascii=False))
        print("   target:", json.dumps(target, ensure_ascii=False))
        print("   reaction_discard:", json.dumps(
            {k: v for k, v in observation["reaction_discard"].items() if k != "glowing_tiles"},
            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
