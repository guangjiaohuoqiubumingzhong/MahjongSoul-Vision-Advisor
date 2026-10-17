"""Reproduce labels visually transcribed from enlarged original training frames.

Contour boxes only locate reviewed pixels; recognition labels are never truth.
"""
from pathlib import Path

from app.capture.frames import load_image
from app.vision.config import load_config, pixel_rect
from app.vision.detector import global_box, river_slot, tile_boxes
from app.vision.templates import rotate
from scripts.label_public_meld_samples import save_reviewed
from scripts.label_reviewed_samples import hand_targets, text_targets

ROOT = Path(__file__).resolve().parents[1]


def main():
    image = ROOT/"data/replays/user-251228-train-recording-01/screenshots/00001362.png"
    pixels = load_image(image)
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")
    rows = {"river_self":["F","1m","5m","P","8p","7s"],
            "river_right":["1p","W","S","9m","P","8p"],
            "river_top":["W","N","F","F","E","9s"],
            "river_left":["1m","N","S","9s","7s","8p"]}
    targets = hand_targets(["3m","2p","2p","3p","5p","6p","1s","2s","3s","4s","4s","5sr","C"])
    for name, labels in rows.items():
        region = config["regions"][name]
        x,y,w,h = pixel_rect(region["rect"],1366,768)
        crop = rotate(pixels[y:y+h,x:x+w],region["rotation"])
        seen = set()
        for box in tile_boxes(crop,region):
            slot,residual = river_slot(box,region["grid"])
            row,column = slot
            if residual>13 or row!=0 or not 0<=column<len(labels) or tuple(slot) in seen:
                raise ValueError(f"Reviewed training layout changed: {name} {box} {slot}")
            seen.add(tuple(slot))
            targets.append({"id":f"{name}-0-{column}","kind":"tile","label":labels[column],
                            "bbox":global_box(box,(x,y),(w,h),region["rotation"]),"region":name,
                            "roles":[name],"rotation":region["rotation"],"player":region["player"],
                            "slot":slot,"horizontal":False,"source":"manual_review_of_enlarged_original_pixels"})
        assert len(seen)==len(labels), name
    targets.append({"id":"dora-0","kind":"tile","label":"2m","bbox":[20,33,39,56],
                    "region":"dora","roles":["dora"],"rotation":0})
    values = [(name,"25000",25000) for name in ("score_self","score_right","score_top","score_left")]
    values += [("round","東1局",{"bakaze":"E","kyoku":1}),("remaining","余45",45),
               ("honba","0",0),("kyotaku","0",0)]
    targets += text_targets(values)
    save_reviewed(image.with_suffix(".annotation.json"),{
        "schema_version":1,"image":str(image),"game_id":"user-251228-f82d20f0","split":"train",
        "reviewed":True,"reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable",
        "targets":targets,"complete_regions":[name for name in config["regions"] if name not in {"buttons","choices"}],
        "coverage_tags":["red_sou"],"information_scope":"self_hand_and_public_information_only",
        "manual_overlay_review":"Replay pointer is on toolbar; enlarged reviewed tile/text fields show no overlay.",
        "review_evidence":[str(image),"reports/vision/train-turn06-rivers-review.png"],
        "notes":"38 known visible tiles: thirteen concealed tiles, no separate draw, twenty-four river tiles, one 2m dora indicator. All four meld areas are empty; visible concealed-hand blue backs supply no identity. Train only."})
    print("Saved one manually reviewed training frame with 38 known visible tiles.")


if __name__ == "__main__":
    main()
