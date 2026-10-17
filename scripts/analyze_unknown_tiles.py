"""Why are tiles unknown at native resolution?

Runs the recognizer on native-resolution frames (e.g. 1920x1080) and, for every
tile that ends up "?" or never becomes a tile item, records:

* reason      - pjura_low_confidence / pjura_ambiguous_classes /
                pjura_rectification_disagrees / pjura_five_colour_unresolved /
                river_slot_ambiguous / duplicate_river_slot / ...
* best guess  - top-3 labels and scores, which view (original / top_face) won,
                each view's best reading, identity rotation
* detection   - unresolved foreground components and clipped boundary
                fragments per region (things never classified at all)

Identical physical tiles recur in many frames, so tiles are grouped by region,
rounded box and best label. Outputs a JSON report, an atlas of unique unknown
tiles (calibrated crop | native crop | rectified face) and a few full-frame
overlays for checking missed detections by eye.

    .venv\\Scripts\\python scripts\\analyze_unknown_tiles.py --stride 3
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.config import load_config, native_box, scaled_client, to_reference, write_json
from app.vision.recognizer import StaticRecognizer, overlay
from app.vision.rectify import top_face_outline, warp_top_face
from app.vision.templates import TemplateBank, rotate
from app.vision.text import TextReader

TABLE = {"river", "meld", "dora", "hand", "draw"}


def frames(pattern, config, stride):
    index = 0
    for name in sorted(glob.glob(pattern, recursive=True)):
        if ".overlay" in name:
            continue
        index += 1
        if (index - 1) % stride:
            continue
        pixels = load_image(name)
        scale = scaled_client(pixels.shape[1], pixels.shape[0], config)
        if scale is not None and scale > 1.0:
            yield Path(name), pixels


def tile_crops(native, reference, scale, bbox, degrees):
    x, y, w, h = bbox
    small = reference[y:y+h, x:x+w]
    nx, ny, nw, nh = native_box(bbox, scale, native.shape)
    large = rotate(native[ny:ny+nh, nx:nx+nw], degrees)
    return small, large


def panel(small, large, rectified, lines):
    cell = np.full((170, 330, 3), 28, np.uint8)
    left = 4
    for image in (small, large, rectified):
        if image is None or image.size == 0:
            left += 108
            continue
        ratio = min(100 / image.shape[1], 118 / image.shape[0])
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        cell[48:48+image.shape[0], left:left+image.shape[1]] = image
        left += 108
    for row, (text, colour) in enumerate(lines):
        cv2.putText(cell, text[:52], (4, 13 + 14 * row), cv2.FONT_HERSHEY_SIMPLEX, .38, colour, 1)
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--images", default="data/screenshots/native-series/**/*.png")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/vision/unknown-tiles-1080p.json"))
    parser.add_argument("--atlas", type=Path, default=Path("reports/vision/unknown-tiles-1080p.png"))
    parser.add_argument("--overlays", type=Path, default=Path("reports/vision/unknown-tiles-1080p-frames"))
    parser.add_argument("--overlay-count", type=int, default=6)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    regions = config["regions"]
    recogniser = StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")),
                                  TextReader())
    groups = {}
    totals = defaultdict(Counter)
    detection = defaultdict(Counter)
    frame_list = list(frames(args.images, config, args.stride))
    overlay_every = max(1, len(frame_list) // max(1, args.overlay_count))
    args.overlays.mkdir(parents=True, exist_ok=True)
    for number, (path, native) in enumerate(frame_list):
        reference, _, scale = to_reference(native, config)
        observation = recogniser.recognize(native, {"image_path": str(path)})
        if number % overlay_every == 0:
            image = overlay(native, observation)
            cv2.imencode(".png", image)[1].tofile(str(args.overlays / f"{path.parent.name}-{path.stem}.png"))
        for name, region in observation["regions"].items():
            if region["kind"] not in TABLE:
                continue
            degrees = regions[name].get("rotation", 0)
            for item in region["items"]:
                totals[name]["tiles"] += 1
                if item["label"] != "?":
                    totals[name]["accepted"] += 1
                    continue
                totals[name]["unknown"] += 1
                totals[name][item.get("reason") or "unknown"] += 1
                evidence = item.get("pjura_evidence") or {}
                key = (name, tuple(int(round(v / 6)) for v in item["bbox"]), item.get("best_label"))
                group = groups.get(key)
                if group is None:
                    small, large = tile_crops(native, reference, scale, item["bbox"], degrees)
                    geometry = top_face_outline(large) if large.size else None
                    turn = item.get("identity_rotation") or 0
                    rectified = warp_top_face(large, geometry, turn).pixels if geometry is not None else None
                    group = groups[key] = {
                        "region": name, "bbox": item["bbox"], "frames": 0, "first_image": str(path),
                        "reason": item.get("reason"), "best_label": item.get("best_label"),
                        "score": item.get("score"), "margin": item.get("margin"),
                        "alternatives": item.get("alternatives"), "view": evidence.get("view"),
                        "views": evidence.get("views"), "identity_rotation": item.get("identity_rotation"),
                        "legacy_label": (item.get("legacy_evidence") or {}).get("label"),
                        "face_fitted": geometry is not None, "slot": item.get("slot"),
                        "_crops": (small, large, rectified)}
                group["frames"] += 1
            for kind in ("unresolved_components", "boundary_fragments", "backs"):
                detection[name][kind] += len(region.get(kind, []))
        print(f"{path.name}: unique unknown tiles {len(groups)}", flush=True)

    unique = sorted(groups.values(), key=lambda value: (value["region"], -(value["frames"])))
    reasons = Counter(value["reason"] for value in unique)
    near = [value for value in unique if value["reason"] == "pjura_low_confidence" and (value["score"] or 0) >= .8]
    cells = []
    for value in unique:
        small, large, rectified = value.pop("_crops")
        alternatives = ", ".join(f"{a['label']} {a['score']:.2f}" for a in (value["alternatives"] or [])[:3])
        views = value.get("views") or {}
        view_text = " ".join(f"{k[:4]}:{v['label']} {v['score']:.2f}" for k, v in views.items())
        cells.append(panel(small, large, rectified, [
            (f"{value['region']} x{value['frames']} {value['reason'] or ''}", (0, 230, 255)),
            (f"top3: {alternatives}", (200, 255, 200)),
            (view_text, (220, 220, 220))]))
    if cells:
        while len(cells) % 4:
            cells.append(np.full((170, 330, 3), 28, np.uint8))
        atlas = np.vstack([np.hstack(cells[i:i+4]) for i in range(0, len(cells), 4)])
        cv2.imencode(".png", atlas)[1].tofile(str(args.atlas))
    report = {"scope": "native_resolution_frames_no_ground_truth", "config_version": config["version"],
              "frames": len(frame_list), "per_region": {k: dict(v) for k, v in sorted(totals.items())},
              "detection_issues": {k: dict(v) for k, v in sorted(detection.items())},
              "unique_unknown_tiles": len(unique), "unique_by_reason": dict(reasons),
              "unique_low_confidence_score_at_least_0_8": len(near), "tiles": unique}
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("frames", "per_region", "detection_issues", "unique_unknown_tiles",
                                             "unique_by_reason", "unique_low_confidence_score_at_least_0_8")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
