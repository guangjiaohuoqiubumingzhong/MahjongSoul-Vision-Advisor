"""Reviewed blue-back negatives from an existing train-only opening image.

Large connected backs are negative evidence, never an opponent tile identity.
All coordinates come from the shown original 250819 training scene and were
visually checked to contain backs; no held-out pixels are used as templates.
"""
from pathlib import Path
import json

from app.capture.frames import load_image
from scripts.label_public_meld_samples import target, save_reviewed


def main():
    image = Path("data/screenshots/user-250819-train/kyoku04/00000001.png")
    load_image(image)
    path = image.with_suffix(".annotation.json")
    annotation = json.loads(path.read_text(encoding="utf-8"))
    if annotation.get("split") != "train" or annotation.get("game_id") != "user-250819-2005":
        raise ValueError("Only this reviewed train game can supply back negatives")
    targets = [target("meld_right",200+i,box,"back",kind="back") for i,box in enumerate(
        [[82,85,137,61],[27,123,56,40],[54,120,56,35],[0,148,27,32],[0,132,56,40],[38,131,45,32]])]
    for item in targets:
        item["negative_scope"] = "visible_blue_back_component_no_hidden_identity_or_physical_tile_count"
    new_ids = {item["id"] for item in targets}
    annotation["targets"] = [item for item in annotation["targets"] if item["id"] not in new_ids]+targets
    annotation.setdefault("review_evidence",[])
    if str(image) not in annotation["review_evidence"]:
        annotation["review_evidence"].append(str(image))
    save_reviewed(path,annotation)
    image = Path("data/screenshots/user-250819-train/kyoku05/00000001.png")
    load_image(image)
    path = image.with_suffix(".annotation.json")
    annotation = json.loads(path.read_text(encoding="utf-8"))
    if annotation.get("split") != "train" or annotation.get("game_id") != "user-250819-2005":
        raise ValueError("Only this reviewed train game can supply back negatives")
    item = target("meld_top",200,[14,29,35,44],"back",kind="back")
    item["negative_scope"] = "separate_visible_blue_back_no_hidden_tile_identity"
    annotation["targets"] = [old for old in annotation["targets"] if old["id"] != item["id"]]+[item]
    annotation.setdefault("review_evidence",[])
    evidence = "reports/vision/training-back-components-review.png"
    if evidence not in annotation["review_evidence"]:
        annotation["review_evidence"].append(evidence)
    save_reviewed(path,annotation)


if __name__ == "__main__":
    main()
