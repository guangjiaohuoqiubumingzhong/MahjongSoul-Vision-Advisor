"""Report the glow halo of every river tile in the real fixtures."""
from __future__ import annotations

import json
from pathlib import Path

from app.capture.frames import load_image
from app.vision.actions import glow_halo, glow_mask, highlighted_discard
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
        path = ROOT / "data/fixtures" / name
        pixels = load_image(path)
        observation = recognizer.recognize(pixels, {})
        regions = observation["regions"]
        mask = glow_mask(pixels)
        boxes = [item["bbox"] for region in regions.values() if region["kind"] == "river"
                 for item in region["items"]]
        print("==", name, observation["status"])
        for region_name, region in regions.items():
            if region["kind"] != "river":
                continue
            for index, tile in enumerate(region["items"]):
                others = [box for box in boxes if box is not tile["bbox"]]
                halo_pixels, coverage, sides = glow_halo(mask, tile["bbox"], others)
                if halo_pixels:
                    print(f"   {region_name}[{index}] {tile['label']} halo_px={halo_pixels} "
                          f"cov={coverage:.3f} sides={sides} last={index == len(region['items']) - 1}")
        print("   =>", json.dumps(highlighted_discard(pixels, regions, 0), ensure_ascii=False)[:260])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
