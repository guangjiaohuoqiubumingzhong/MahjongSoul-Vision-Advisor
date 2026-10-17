"""Manually transcribed labels from the first user-client calibration frame.

This is a train sample, never an independent accuracy claim.
"""
import json
from pathlib import Path
from app.vision.config import write_json

root = Path(__file__).resolve().parents[1]
folder = root / "data/screenshots/user-251228-calibration"
image = folder / "00000001.png"
labels = ["1m", "5m", "2p", "2p", "3p", "6p", "8p", "1s", "2s", "3s", "4s", "4s", "5sr"]
edges = [158, 225, 292, 359, 427, 494, 562, 629, 697, 764, 832, 899, 967, 1035]
targets = [{"id": f"hand-{i}", "kind": "tile", "label": label, "bbox": [edges[i], 655, edges[i+1]-edges[i], 108],
            "region": "hand", "roles": ["hand", "draw"], "rotation": 0, "player": 0} for i,label in enumerate(labels)]
targets.append({"id": "dora-0", "kind": "tile", "label": "2m", "bbox": [20,33,39,56], "region": "dora", "roles": ["dora"], "rotation": 0})
targets.append({"id": "dora-back", "kind": "back", "label": "back", "bbox": [60,33,38,56], "region": "dora", "roles": ["dora"], "rotation": 0})
text_targets = [("score_self", "25000",25000,[650,332,66,22]), ("score_top","25000",25000,[650,249,66,22]),
                ("score_right","25000",25000,[733,276,25,76]), ("score_left","25000",25000,[610,276,25,76]),
                ("round", "東1局", {"bakaze":"E","kyoku":1},[650,274,70,28]),
                ("remaining","余69",69,[650,306,70,25]), ("honba","0",0,[181,94,27,35]), ("kyotaku","0",0,[80,94,27,35])]
for region,label,value,box in text_targets:
    targets.append({"id":region,"kind":"text","label":label,"value":value,"bbox":box,"region":region,"rotation":0})
annotation = {"schema_version":1,"image":str(image),"game_id":"user-251228-f82d20f0","split":"train","reviewed":True,
              "reviewer":"manual_visual_transcription_by_developer", "frame_quality":"stable", "targets":targets,
              "complete_regions":["hand","draw","dora", *[value[0] for value in text_targets]],
              "information_scope":"self_hand_and_public_information_only",
              "notes":"Opponent concealed hands visible in this calibration original are excluded from every annotation and recognition region. No test split labels are derived from the templates."}
write_json(image.with_suffix(".annotation.json"), annotation)
config_path = root / "configs/window-steam-1366.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["regions"]["hand"]["tile_width"] = 67.5
config["regions"]["dora"]["tile_width"] = 39.5
write_json(config_path, config)
print("Seeded 14 known visible tile instances, one back, eight text fields; train only.")
