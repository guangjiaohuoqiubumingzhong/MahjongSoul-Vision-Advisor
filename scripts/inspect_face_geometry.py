"""Diagnose frozen face geometry on reviewed crops without changing labels."""
from pathlib import Path
import json

import cv2
import numpy as np

from app.capture.frames import load_image, save_image
from app.vision.config import write_json
from app.vision.faces import FACE_PARAMETERS, tile_face
from app.vision.templates import rotate


def main():
    rows, records = [], []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if not annotation.get("reviewed") or annotation.get("frame_quality") != "stable":
            continue
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent / image
        pixels = load_image(image)
        for target in annotation["targets"]:
            if target.get("kind") != "tile":
                continue
            x, y, width, height = target["bbox"]
            patch = rotate(pixels[y:y+height, x:x+width], target.get("rotation", 0))
            if tile_face(patch) is not None:
                continue
            background = np.percentile(patch, FACE_PARAMETERS["background_percentile"], axis=(0, 1))
            normalized = np.clip(patch.astype(np.float32) / np.maximum(background, FACE_PARAMETERS["minimum_background"]), 0, 1)
            mask = (normalized.min(axis=2) >= FACE_PARAMETERS["white_min_channel"]).astype(np.uint8)*255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(max(contours, key=cv2.contourArea)) if contours else None
            polygon = cv2.approxPolyDP(hull, FACE_PARAMETERS["polygon_epsilon"]*cv2.arcLength(hull, True), True) if hull is not None else None
            record = {"annotation":str(path), "split":annotation["split"], "id":target["id"],
                      "label":target["label"], "region":target["region"],
                      "shape":list(patch.shape), "background":background.tolist(),
                      "polygon":polygon.reshape(-1,2).tolist() if polygon is not None else [],
                      "area_fraction":cv2.contourArea(polygon)/(patch.shape[0]*patch.shape[1]) if polygon is not None else 0}
            records.append(record)
            marked = patch.copy()
            if polygon is not None:
                cv2.polylines(marked, [polygon], True, (0,255,0), 1)
            row = np.full((195,550,3),45,np.uint8)
            cv2.putText(row, f"{annotation['split']} {target['region']} {target['label']} {len(record['polygon'])} corners", (5,20),0,.48,(255,255,255),1)
            for index, view in enumerate([patch, cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR), marked]):
                row[30:190,5+180*index:165+180*index] = cv2.resize(view,(160,160),interpolation=cv2.INTER_NEAREST)
            rows.append(row)
    write_json("reports/vision/face-geometry-failures.json", {"scope":"diagnosis_only_not_training_or_acceptance", "parameters":FACE_PARAMETERS,"failures":records})
    if rows:
        save_image("reports/vision/face-geometry-failures.png",np.vstack(rows))
    print(json.dumps(records,ensure_ascii=True))


if __name__ == "__main__":
    main()
