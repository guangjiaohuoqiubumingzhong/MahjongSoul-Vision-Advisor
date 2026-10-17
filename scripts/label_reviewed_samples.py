"""Reproduce manually transcribed labels for two saved client frames.

The transcription below was reviewed against enlarged original crops. The late
251228 frame is train only; the 250821 starting frame is independent test only.
No opponent concealed hands or reference events are used.
"""
from pathlib import Path

from app.capture.frames import load_image
from app.vision.config import load_config, pixel_rect, write_json
from app.vision.detector import global_box, river_slot, tile_boxes
from app.vision.templates import rotate

ROOT = Path(__file__).resolve().parents[1]


def text_targets(values):
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")
    return [{"id": name, "kind": "text", "label": label, "value": value,
             "bbox": list(pixel_rect(config["regions"][name]["rect"],1366,768)),
             "region":name, "field":config["regions"][name]["field"],
             "rotation":config["regions"][name].get("rotation",0)}
            for name, label, value in values]


def hand_targets(labels):
    edges = [158,225,292,359,427,494,562,629,697,764,832,899,967,1035]
    return [{"id":f"hand-{i}","kind":"tile","label":label,"bbox":[edges[i],655,edges[i+1]-edges[i],108],
             "region":"hand","roles":["hand","draw"],"rotation":0,"player":0,"slot_index":i}
            for i,label in enumerate(labels)]


def label_additional_training_starts():
    # Reviewed from reports/vision/additional-train-hands.png and original scenes.
    samples = [
        (["3m","9m","2p","3p","9p","1s","2s","2s","4s","6s","7s","S","N"],"C","3p",1,0,0,[25000,25000,25000,25000]),
        (["4m","4m","5mr","5m","5m","2p","5pr","7p","1s","8s","9s","E","F"],"C","6s",1,1,0,[37000,13000,25000,25000]),
        (["9m","1p","3p","5pr","6p","9p","1s","5s","6s","E","W","N","N"],"P","2s",1,2,0,[43100,13000,18900,25000]),
        (["4m","5m","7m","5p","7p","9p","3s","3s","6s","7s","8s","9s","S"],None,"4p",2,0,0,[42200,15100,18300,24400]),
        (["2m","2m","8m","1p","2p","6p","1s","1s","2s","5sr","6s","7s","9s"],None,"6p",3,0,0,[34200,15100,26300,24400]),
        (["1p","2p","2p","3p","5sr","5s","7s","8s","9s","E","E","S","P"],None,"8p",4,1,1,[34700,16600,24800,22900]),
    ]
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")
    for index,(labels,draw,dora,kyoku,honba,kyotaku,scores) in enumerate(samples,1):
        image = ROOT/f"data/screenshots/user-250819-train/kyoku{index:02d}/00000001.png"
        load_image(image)  # Require the actual source evidence; never create constructed data.
        targets = hand_targets(labels)
        if draw:
            targets.append({"id":"draw-0","kind":"tile","label":draw,"bbox":[1057,655,67,108],
                            "region":"draw","roles":["hand","draw"],"rotation":0,"player":0})
        targets.append({"id":"dora-0","kind":"tile","label":dora,"bbox":[20,33,39,56],
                        "region":"dora","roles":["dora"],"rotation":0})
        values = [(name,str(value),value) for name,value in zip(
            ["score_self","score_right","score_top","score_left"],scores)]
        values += [("round",f"東{kyoku}局",{"bakaze":"E","kyoku":kyoku}),
                   ("remaining","余69",69),("honba",str(honba),honba),("kyotaku",str(kyotaku),kyotaku)]
        targets += text_targets(values)
        write_json(image.with_suffix(".annotation.json"),{
            "schema_version":1,"image":str(image),"game_id":"user-250819-2005","split":"train",
            "reviewed":True,"reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable",
            "manual_overlay_review":"Cursor on replay toolbar; labelled tile and text areas are unobstructed.",
            "targets":targets,"complete_regions":[name for name in config["regions"] if name not in {"buttons","choices"}],
            "coverage_tags":["red_man","red_pin","red_sou"],
            "information_scope":"selected_self_hand_and_public_information_only",
            "notes":"Opening scenes with empty rivers/melds; only train, never used for independent accuracy."})
    image = ROOT/"data/screenshots/user-250819-train/alternate-top-kyoku02/00000001.png"
    load_image(image)
    targets = hand_targets(["6m","7m","4p","7p","9p","2s","E","S","W","W","N","P","P"])
    targets.append({"id":"dora-0","kind":"tile","label":"6s","bbox":[20,33,39,56],
                    "region":"dora","roles":["dora"],"rotation":0})
    values = [("score_self","25000",25000),("score_right","25000",25000),("score_top","37000",37000),
              ("score_left","13000",13000),("round","東1局",{"bakaze":"E","kyoku":1}),
              ("remaining","余69",69),("honba","1",1),("kyotaku","0",0)]
    targets += text_targets(values)
    write_json(image.with_suffix(".annotation.json"),{
        "schema_version":1,"image":str(image),"game_id":"user-250819-2005","split":"train",
        "reviewed":True,"reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable",
        "perspective":"original_top_player_selected_as_bottom_in_replay",
        "use":"offline_template_collection_only_not_a_user_decision_state",
        "manual_overlay_review":"Cursor on replay toolbar, outside labelled fields.","targets":targets,
        "complete_regions":[name for name in config["regions"] if name not in {"buttons","choices"}],
        "coverage_tags":[],"information_scope":"selected_visible_hand_only_other_three_hands_hidden",
        "notes":"Replay viewpoint changed through its camera control. The visible selected hand provides 6m and 4p templates; never passed to the user's decision session."})


def main():
    image = ROOT/"data/replays/user-251228-train-recording-01/screenshots/00002269.png"
    pixels = load_image(image)
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")
    # Rows follow the owning player's reading direction, independently of contour order.
    rows = {
        "river_self":[["F","1m","5m","P","8p","7s"],["1s","C","3p","C","7m","8p"]],
        "river_right":[["1p","W","S","9m","P","8p"],["7s","4p","8s","C","2s","4p"],["P","4m"]],
        "river_top":[["W","N","F","F","E","9s"],["N","E","8s","5s","9p","7p"],["9s"]],
        "river_left":[["1m","N","S","9s","7s","8p"],["9m","2m","7s","C","2s","E"],["3m"]],
    }
    targets = hand_targets(["2m","3m","2p","2p","5p","5p","5p","6p","2s","3s","4s","4s","5sr"])
    for i in (2,3,4,5,6,10,11):
        targets[i]["highlight"] = "pink"
    for name, labels in rows.items():
        region = config["regions"][name]
        x,y,w,h = pixel_rect(region["rect"],1366,768)
        crop = rotate(pixels[y:y+h,x:x+w],region["rotation"])
        seen = set()
        for box in tile_boxes(crop,region):
            slot, residual = river_slot(box,region["grid"])
            row,column = slot
            if residual > 13 or row >= len(labels) or column >= len(labels[row]) or tuple(slot) in seen:
                raise ValueError(f"Reviewed layout differs from candidate crop: {name} {box} {slot}")
            seen.add(tuple(slot))
            horizontal = name == "river_left" and slot == [1,4]
            targets.append({"id":f"{name}-{row}-{column}","kind":"tile","label":labels[row][column],
                            "bbox":global_box(box,(x,y),(w,h),region["rotation"]),"region":name,"roles":[name],
                            "rotation":(region["rotation"]+90*horizontal)%360,"player":region["player"],
                            "slot":slot,"horizontal":horizontal,"source":"manually_reviewed_saved_crop"})
        assert len(seen) == sum(map(len,labels)), name
    for i,label in enumerate(["2m","7p"]):
        targets.append({"id":f"dora-{i}","kind":"tile","label":label,"bbox":[20+40*i,33,39,56],
                        "region":"dora","roles":["dora"],"rotation":0})
    values = [("score_self","25000",25000),("score_right","25000",25000),("score_top","25000",25000),
              ("score_left","24000",24000),("round","東1局",{"bakaze":"E","kyoku":1}),
              ("remaining","余16",16),("honba","0",0),("kyotaku","1",1)]
    targets += text_targets(values)
    write_json(image.with_suffix(".annotation.json"),{
        "schema_version":1,"image":str(image),"game_id":"user-251228-f82d20f0","split":"train",
        "reviewed":True,"reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable",
        "manual_overlay_review":"Cursor over replay toolbar; reviewed labelled regions are unobstructed.",
        "targets":targets,"complete_regions":["hand","draw","dora",*rows,*[v[0] for v in values]],
        "coverage_tags":["highlight","riichi_sideways"],
        "information_scope":"self_hand_and_public_information_only",
        "notes":"67 visible tile instances. Melds remain unlabelled; not a complete-scene ground truth or a test sample."})

    image = ROOT/"data/screenshots/user-250821-independent-start/00000001.png"
    targets = hand_targets(["1m","2m","2m","1p","3p","4p","7p","8p","3s","7s","7s","8s","S"])
    targets.append({"id":"dora-0","kind":"tile","label":"P","bbox":[20,33,39,56],
                    "region":"dora","rotation":0})
    targets.extend(text_targets([(name,"25000",25000) for name in ("score_self","score_right","score_top","score_left")]
        +[("round","東1局",{"bakaze":"E","kyoku":1}),("remaining","余69",69),("honba","0",0),("kyotaku","0",0)]))
    write_json(image.with_suffix(".annotation.json"),{
        "schema_version":1,"image":str(image),"game_id":"user-250821-1613","split":"test",
        "reviewed":True,"reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable",
        "manual_overlay_review":"Cursor at right player portrait; all labelled visible fields are unobstructed.",
        "targets":targets,"complete_regions":[name for name in config["regions"] if name not in {"buttons","choices"}],
        "coverage_tags":[],"information_scope":"self_hand_and_public_information_only",
        "notes":"Independent starting scene. All four rivers and meld areas are visibly empty. Replay controls are not gameplay action buttons."})
    label_additional_training_starts()
    print("Saved reviewed labels for nine train frames and one independent test frame.")
    from scripts.label_public_meld_samples import main as label_public_meld_samples
    label_public_meld_samples()


if __name__ == "__main__":
    main()
