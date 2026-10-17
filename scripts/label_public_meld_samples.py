"""Labels manually reviewed in enlarged original public-tile crops.

No recognition labels are accepted automatically. Candidate rectangles locate
the saved pixels only; the identities below were visually transcribed. Replay
250821 remains test-only, including all viewpoints and rounds of that game.
"""
from pathlib import Path
import json
from datetime import datetime, timezone

from app.capture.frames import load_image
from app.vision.config import write_json
from app.vision.detector import global_box
from scripts.label_reviewed_samples import hand_targets, text_targets

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = {"meld_right":(1038,0,180,342,270), "meld_left":(30,463,124,250,90),"meld_top":(240,0,277,78,180),
            "meld_self":(600,635,766,133,0),
            "river_self":(540,380,278,130,0),"river_right":(806,203,202,191,270),
            "river_top":(562,108,240,112,180),"river_left":(387,204,170,197,90)}
PLAYERS = {"self":0,"right":1,"top":2,"left":3}


def target(region, index, box, label, *, sideways=False, group=None, meld_type=None, source=None, slot=None, kind="tile"):
    x,y,w,h,rotation = GEOMETRY[region]
    value = {"id":f"{region}-{index}","kind":kind,"label":label,
             "bbox":global_box(box,(x,y),(w,h),rotation),"region":region,
             "roles":["meld_self","meld_right","meld_top","meld_left"] if region.startswith("meld_") and kind=="tile" else [region],
             "rotation":(rotation+270*sideways)%360,"horizontal":sideways,
             "player":PLAYERS[region.split('_')[1]],"source":"manual_review_of_enlarged_original_pixels"}
    if group is not None:
        value.update(group_id=group,meld_type=meld_type,target=source)
    if slot is not None:
        value["slot"] = slot
    return value


def save_reviewed(path, annotation, *, reason="Manual review of full public tile bounds and shared meld template roles"):
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        old = {t["id"]:t for t in previous["targets"]}
        corrections = [{"at":datetime.now(timezone.utc).isoformat(),"before":old[t["id"]],"after":t,
                        "reason":reason}
                       for t in annotation["targets"] if t["id"] in old and old[t["id"]] != t]
        if corrections:
            with path.with_suffix(".corrections.jsonl").open("a",encoding="utf-8") as stream:
                for correction in corrections:
                    stream.write(json.dumps(correction,ensure_ascii=False)+"\n")
    write_json(path,annotation)


def main():
    image = ROOT/"data/replays/user-251228-train-recording-01/screenshots/00002269.png"
    annotation_path = image.with_suffix(".annotation.json")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["targets"] = [t for t in annotation["targets"] if t["region"] != "meld_right"]
    boxes = [(46,97,40,59),(87,99,33,48),(107,80,39,56),(139,72,31,56),
             (163,66,30,54),(187,59,29,53),(217,61,28,43),
             (238,53,34,42),(259,36,34,50),(286,30,26,49)]
    labels = ["6s"]*4+["8m"]*3+["1s"]*3
    for i,(box,label) in enumerate(zip(boxes,labels)):
        group = "six-sou-four-inline" if i<4 else "eight-man-pon" if i<7 else "one-sou-pon"
        annotation["targets"].append(target("meld_right",i,box,label,sideways=i in {1,6,7},
            group=group,meld_type=None if i<4 else "pon"))
    annotation["notes"] = "77 known visible tiles. The right meld area has ten tiles, not twelve: 6s x4, 8m x3, 1s x3. Called source and exact kan type remain unconfirmed; no full-scene or independent accuracy claim."
    annotation["review_evidence"] = ["reports/vision/train-meld-tile-crops.png","reports/vision/hand-comparison.png","reports/vision/river-regions-review.png"]
    if "meld_right" not in annotation["complete_regions"]:
        annotation["complete_regions"].append("meld_right")
    save_reviewed(annotation_path,annotation)

    image = ROOT/"data/screenshots/user-250819-train/kyoku02-turn06/00000001.png"
    load_image(image)
    targets = hand_targets(["3m","4m","4m","5mr","5m","5m","E","C"])
    targets.append({"id":"dora-0","kind":"tile","label":"6s","bbox":[20,33,39,56],"region":"dora","roles":["dora"],"rotation":0})
    melds = {
        "meld_self":[([351,52,71,54],"6p",True,"a","chi",3),([416,38,58,68],"5pr",False,"a","chi",3),
                     ([467,38,58,68],"7p",False,"a","chi",3),([521,52,72,54],"4p",True,"b","chi",3),
                     ([582,38,63,68],"2p",False,"b","chi",3),([634,38,63,68],"3p",False,"b","chi",3)],
        "meld_right":[([231,53,41,42],"1s",True,"c","chi",0),([259,36,34,50],"2s",False,"c","chi",0),([286,30,26,49],"3s",False,"c","chi",0)],
        "meld_left":[([79,3,47,73],"8s",False,"d","pon",0),([117,14,50,75],"8s",False,"d","pon",0),([157,41,65,66],"8s",True,"d","pon",0)]}
    for region,items in melds.items():
        targets += [target(region,i,box,label,sideways=sideways,group=group,meld_type=kind,source=source)
                    for i,(box,label,sideways,group,kind,source) in enumerate(items)]
    # Visible blue backs are negative examples, never a concealed tile identity.
    for i,box in enumerate([[27,123,56,40],[54,120,56,35],[0,148,27,32],[0,132,56,40]],100):
        targets.append(target("meld_right",i,box,"back",kind="back"))
    rivers = {
        "river_self":[([7,7,44,55],"1s"),([52,6,41,55],"9s"),([96,6,41,55],"F")],
        "river_right":[([3,14,43,56],"7p"),([48,14,28,54],"1m"),([77,11,27,53],"4p"),([105,8,28,53],"S")],
        "river_top":[([5,3,36,46],"2s"),([43,3,37,45],"9p"),([81,2,36,44],"6s"),([119,2,36,45],"4s")],
        "river_left":[([3,5,24,51],"C"),([29,6,26,52],"3m"),([55,9,41,53],"7m")]}
    for region,items in rivers.items():
        targets += [target(region,i,box,label,slot=[0,i]) for i,(box,label) in enumerate(items)]
    values = [("score_self","37000",37000),("score_right","13000",13000),("score_top","25000",25000),
              ("score_left","25000",25000),("round","東1局",{"bakaze":"E","kyoku":1}),
              ("remaining","余55",55),("honba","1",1),("kyotaku","0",0)]
    targets += text_targets(values)
    save_reviewed(image.with_suffix(".annotation.json"),{
        "schema_version":1,"image":str(image),"game_id":"user-250819-2005","split":"train","reviewed":True,
        "reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable","action_phase":"post_chi_before_discard",
        "targets":targets,"complete_regions":["hand","draw","dora",*rivers,*melds,"meld_top",*[v[0] for v in values]],
        "coverage_tags":["chi","pon","red_in_meld"],"information_scope":"self_hand_and_public_information_only",
        "manual_overlay_review":"Replay cursor and colour-key overlay are on the toolbar, outside reviewed tile/text fields.",
        "review_evidence":["reports/vision/native-chi-pon-review.png","reports/vision/native-chi-rivers-review.png"],
        "meld_source_evidence":"Left pon has its rightmost tile sideways in the normalized player view: source is self (0). This follows the public display rule, and its 8s is absent from the self opening hand and visible discards after the call; no concealed opponent identity is read.",
        "notes":"35 known visible tiles; two self chi groups, one right chi and one left pon. Chi sources and the left pon source are confirmed by calibrated public layout. Exact kan structure is not covered. Blue backs have no tile identity."})

    for number,draw in [(1045,"6m"),(1135,None)]:
        image = ROOT/f"data/replays/user-250821-test-recording-01/screenshots/{number:08d}.png"
        load_image(image)
        targets = hand_targets(["3p","4p","8p","8p","3s","4s","5s","7s","8s","8s"])
        if draw:
            targets.append({"id":"draw-0","kind":"tile","label":draw,"bbox":[854,655,68,108],"region":"draw","rotation":0,"player":0})
        annotation = {
            "schema_version":1,"image":str(image),"game_id":"user-250821-1613","split":"test","reviewed":True,
            "reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable","targets":targets,
            "complete_regions":["hand","draw"],"coverage_tags":["shortened_concealed_hand"],
            "review_evidence":["reports/vision/independent-post-pon-hand-review.png",str(image)],
            "manual_overlay_review":"Cursor is on replay controls, outside hand and draw.",
            "information_scope":"visible_self_hand_only","notes":"Normal 5s has a red central stick; normal 7s has a red top stick. Neither is a red five. Public areas not ground-truthed in this partial sample; never used for templates."}
        if number == 1135:
            public = {
                "meld_self":[([514,38,63,68],"2m",False),([572,52,73,54],"2m",True),([632,38,65,68],"2m",False)],
                "meld_right":[([231,53,41,42],"6m",True),([259,36,34,50],"4m",False),([286,30,26,49],"5m",False)],
                "meld_top":[([120,14,38,39],"5mr",False),([156,20,45,33],"5m",True),([200,14,37,39],"5m",False)],
                "meld_left":[([79,3,49,73],"3m",False),([117,29,52,64],"3m",True),([170,29,53,78],"3m",False)]}
            for region,items in public.items():
                targets += [target(region,i,box,label,sideways=sideways,group=region,
                                   meld_type="chi" if region=="meld_right" else "pon",
                                   source={"meld_self":2,"meld_right":0,"meld_top":0,"meld_left":1}[region])
                            for i,(box,label,sideways) in enumerate(items)]
            rows = {
                "river_self":[([6,8,44,54],"S",[0,0]),([52,7,41,54],"1p",[0,1]),([96,6,41,54],"1m",[0,2]),
                              ([139,6,41,54],"7p",[0,3]),([183,6,43,55],"W",[0,4]),([227,7,45,55],"7s",[0,5])],
                "river_right":[([4,16,43,55],"W",[0,0]),([48,14,30,54],"9s",[0,1]),([72,9,35,54],"1m",[0,2]),
                               ([108,8,27,51],"2s",[0,3]),([135,6,26,51],"1m",[0,4]),([161,2,26,51],"9p",[0,5]),([4,73,43,57],"E",[1,0])],
                "river_top":[([3,2,38,45],"2s",[0,0]),([42,2,36,45],"W",[0,1]),([80,3,36,44],"N",[0,2]),
                             ([118,3,36,46],"S",[0,3]),([156,3,37,45],"P",[0,4]),([194,3,38,45],"9m",[0,5]),([6,49,37,29],"2m",[1,0])],
                "river_left":[([3,6,26,49],"W",[0,0]),([29,6,28,52],"S",[0,1]),([56,10,27,53],"9m",[0,2]),
                              ([83,13,28,53],"C",[0,3]),([112,14,31,54],"4m",[0,4]),([142,18,43,56],"9s",[0,5]),([3,57,38,52],"1p",[1,0])]}
            for region,items in rows.items():
                targets += [target(region,i,box,label,slot=slot) for i,(box,label,slot) in enumerate(items)]
            targets.append({"id":"dora-0","kind":"tile","label":"P","bbox":[20,33,39,56],"region":"dora","rotation":0})
            values = [(name,"25000",25000) for name in ["score_self","score_right","score_top","score_left"]]
            values += [("round","東1局",{"bakaze":"E","kyoku":1}),("remaining","余43",43),("honba","0",0),("kyotaku","0",0)]
            targets += text_targets(values)
            annotation["complete_regions"] += ["dora",*public,*rows,*[v[0] for v in values]]
            annotation["coverage_tags"] += ["chi","pon","red_in_meld"]
            annotation["review_evidence"].append("reports/vision/independent-public-review.png")
            annotation["review_evidence"] += ["reports/vision/independent-meld-orientation-review.png",
                                               "reports/vision/independent-call-sequence-review.png"]
            annotation["meld_source_evidence"] = "Public call sequence confirms left 3m pon from right (frames 683/729), top 5m pon from self (864/909), self 2m pon from top (909/954). Each has the middle tile sideways after player-view normalization. Right chi source is self by the four-player chi rule. See reports/vision/meld-source-review.json; never use concealed opponents' tiles."
            annotation["information_scope"] = "self_hand_and_public_information_only"
            annotation["notes"] = "50 known visible tiles, manually reviewed in enlarged originals. Right meld is 6m/4m/5m chi; its called six was originally mistranscribed as three and corrected after enlarged-pixel comparison. Top is a red-five pon. The three pon sources are verified from public call sequences and sideways placement; button ground truth is incomplete. Never used for templates. Normal 5s/7s red accents are not red fives."
        save_reviewed(image.with_suffix(".annotation.json"),annotation)
    print("Added reviewed public meld/rivers and two independent shortened-hand frames.")


if __name__ == "__main__":
    main()
