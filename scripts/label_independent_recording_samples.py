"""Reproduce a visually reviewed stable frame and genuine settlement occlusion.

All frames of this replay remain test-only, never a template source.
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
    image = ROOT/"data/replays/user-250821-test-recording-01/screenshots/00000546.png"
    pixels = load_image(image)
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")
    rows = {"river_self":["S","1p","1m"],"river_right":["W","9s","1m"],
            "river_top":["2s","W","N"],"river_left":["W","S","9m"]}
    targets = hand_targets(["2m","2m","3p","4p","7p","8p","8p","3s","4s","7s","7s","8s","8s"])
    for name,labels in rows.items():
        region = config["regions"][name]
        x,y,w,h = pixel_rect(region["rect"],1366,768)
        crop = rotate(pixels[y:y+h,x:x+w],region["rotation"])
        seen = set()
        for box in tile_boxes(crop,region):
            slot,residual = river_slot(box,region["grid"])
            row,column = slot
            if residual>13 or row!=0 or not 0<=column<len(labels) or tuple(slot) in seen:
                raise ValueError(f"Reviewed independent layout changed: {name} {box} {slot}")
            seen.add(tuple(slot))
            targets.append({"id":f"{name}-0-{column}","kind":"tile","label":labels[column],
                            "bbox":global_box(box,(x,y),(w,h),region["rotation"]),"region":name,"roles":[name],
                            "rotation":region["rotation"],"player":region["player"],"slot":slot,"horizontal":False,
                            "source":"manual_review_of_enlarged_original_pixels"})
        assert len(seen)==len(labels), name
    targets.append({"id":"dora-0","kind":"tile","label":"P","bbox":[20,33,39,56],"region":"dora","rotation":0})
    values = [(name,"25000",25000) for name in ("score_self","score_right","score_top","score_left")]
    values += [("round","東1局",{"bakaze":"E","kyoku":1}),("remaining","余57",57),
               ("honba","0",0),("kyotaku","0",0)]
    targets += text_targets(values)
    save_reviewed(image.with_suffix(".annotation.json"),{
        "schema_version":1,"image":str(image),"game_id":"user-250821-1613","split":"test","reviewed":True,
        "reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable","targets":targets,
        "complete_regions":[name for name in config["regions"] if name not in {"buttons","choices"}],
        "coverage_tags":[],"information_scope":"self_hand_and_public_information_only",
        "manual_overlay_review":"Replay pointer on toolbar, outside enlarged reviewed tile/text fields.",
        "review_evidence":[str(image),"reports/vision/independent-turn03-rivers-review.png"],
        "notes":"26 known visible tiles: thirteen concealed, twelve river tiles and one white dora indicator. No separate draw or public melds. All opponents' hands hidden. Test only, never used for templates."})
    image = ROOT/"data/replays/user-250821-test-recording-01/screenshots/00001360.png"
    load_image(image)
    save_reviewed(image.with_suffix(".annotation.json"),{
        "schema_version":1,"image":str(image),"game_id":"user-250821-1613","split":"test","reviewed":True,
        "reviewer":"manual_visual_review_by_developer","frame_quality":"occluded","targets":[],
        "complete_regions":[],"expected_pause_regions":["hand","score_self","remaining"],
        "coverage_tags":["occlusion"],"review_evidence":[str(image)],
        "information_scope":"settlement_panel_occluding_game_table_not_a_decision_state",
        "manual_overlay_review":"Settlement panel itself is the reviewed occluder; cursor on panel adds no truth.",
        "notes":"Genuine recorded settlement panel covers the own hand, own score and remaining count. No panel tile/score is ground-truthed as an active-table target. Pause must have evidence from these occluded regions, not just an uncalibrated configuration or missing button templates."})
    for number, regions, note in [
        (637,["hand","score_self"],"A genuine discard-arm animation covers one regular hand slot and the self score."),
        (683,["meld_left"],"A genuine pon character animation occludes the left meld area with an oversized bright foreground; it cannot be reported as confirmed empty.")]:
        image = ROOT/f"data/replays/user-250821-test-recording-01/screenshots/{number:08d}.png"
        load_image(image)
        save_reviewed(image.with_suffix(".annotation.json"),{
            "schema_version":1,"image":str(image),"game_id":"user-250821-1613","split":"test","reviewed":True,
            "reviewer":"manual_visual_review_by_developer","frame_quality":"animation","targets":[],
            "complete_regions":[],"expected_pause_regions":regions,"coverage_tags":["animation","occlusion"],
            "review_evidence":[str(image)],"information_scope":"occluding_animation_not_a_decision_state",
            "notes":note+" No animation pixels are labeled as active-table tiles. Requires explicit region failure evidence."})
    print("Saved one stable test frame with 26 known visible tiles, one settlement occlusion and two occluding animations.")


if __name__ == "__main__":
    main()
