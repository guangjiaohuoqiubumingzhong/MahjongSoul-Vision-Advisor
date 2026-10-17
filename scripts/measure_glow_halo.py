"""Measure the reaction-glow halo metric on real river tiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.capture.frames import load_image
from app.vision.actions import glow_halo, glow_mask, highlighted_discard, tile_glow
from app.vision.config import load_config
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/window-steam-1366-v2.json")
    parser.add_argument("--crop", default="0,26,1912,1092")
    args = parser.parse_args()
    config = load_config(args.config)
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader())
    pixels = load_image(args.image)
    l, t, r, b = (int(v) for v in args.crop.split(","))
    client = pixels[t:b, l:r]
    observation = recognizer.recognize(client, {})
    mask = glow_mask(client)
    regions = observation["regions"]
    boxes = [item["bbox"] for region in regions.values() if region["kind"] == "river"
             for item in region["items"]]
    print("river tiles with halo measurements:")
    for name, region in regions.items():
        if region["kind"] != "river":
            continue
        for index, tile in enumerate(region["items"]):
            others = [box for box in boxes if box is not tile["bbox"]]
            count, density, sides = tile_glow(mask, tile["bbox"], others)
            halo_pixels, coverage, halo_sides = glow_halo(mask, tile["bbox"], others)
            print(f"  {name}[{index}] {tile['label']:>3} bbox={tile['bbox']} "
                  f"side_px={count:4d} density={density:.3f} sides={sides} "
                  f"halo_px={halo_pixels:4d} coverage={coverage:.3f} halo_sides={halo_sides}")
    print("highlighted_discard:", json.dumps(highlighted_discard(client, regions, config["seat_mapping"]["self"]),
                                             ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
