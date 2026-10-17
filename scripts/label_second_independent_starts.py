"""Reproduce manually reviewed opening labels for the second test-only game.

Only the selected user's visible hand and public fields are transcribed. No
recognition result, opponent concealed hand or reference event supplies truth.
"""
from pathlib import Path

from app.capture.frames import load_image
from app.vision.config import load_config
from scripts.label_public_meld_samples import save_reviewed
from scripts.label_reviewed_samples import hand_targets, text_targets

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    samples = [
        (1,["1m","3m","6p","6p","1s","4s","6s","7s","9s","E","N","C","C"],"2s",[25000]*4),
        (2,["2m","7m","3p","4p","5p","2s","3s","E","S","W","N","P","C"],"8p",[23000,25000,25000,27000]),
        (3,["1m","4m","4m","8m","9m","9p","1s","5sr","6s","6s","E","N","F"],"9s",[22000,24000,23000,31000]),
        (4,["1m","2m","6m","7m","1p","2p","2s","3s","5sr","5s","E","W","P"],"8m",[21500,26000,22500,30000]),
        (5,["1m","1m","5mr","7m","8m","9m","2p","4p","9p","7s","8s","S","W"],"E",[21500,27000,22500,29000]),
    ]
    for kyoku,labels,dora,scores in samples:
        folder = "south01-start" if kyoku==5 else f"kyoku{kyoku:02d}-start"
        image = ROOT / f"data/screenshots/user-251228-2240-test/{folder}/00000001.png"
        load_image(image)
        targets = hand_targets(labels)
        if kyoku == 4:
            targets.append({"id":"draw-0","kind":"tile","label":"C","bbox":[1057,655,67,108],
                            "region":"draw","rotation":0,"player":0})
        targets.append({"id":"dora-0","kind":"tile","label":dora,"bbox":[20,33,39,56],
                        "region":"dora","rotation":0})
        values = [(name,str(score),score) for name,score in zip(("score_self","score_right","score_top","score_left"),scores)]
        wind,number = ("S",1) if kyoku==5 else ("E",kyoku)
        values += [("round",f"{'南' if wind=='S' else '東'}{number}局",{"bakaze":wind,"kyoku":number}), ("remaining","余69",69),
                   ("honba","0",0),("kyotaku","0",0)]
        targets += text_targets(values)
        save_reviewed(image.with_suffix(".annotation.json"),{
            "schema_version":1,"image":str(image),"game_id":"user-251228-2240","split":"test",
            "reviewed":True,"reviewer":"manual_visual_transcription_by_developer","frame_quality":"stable",
            "targets":targets,"complete_regions":[name for name in config["regions"] if name not in {"buttons","choices"}],
            "coverage_tags":[],"information_scope":"self_hand_and_public_information_only",
            "observed_match_mode":"four_player_east","observed_room_label":"银之间 · 四人东",
            "manual_overlay_review":"Other hands are blue backs. Cursor is outside every transcribed target; replay overlay is confined to controls.",
            "review_evidence":[str(image)],
            "notes":"Second independent game, 2025-12-28 22:40. The room label still says four-player East while the fifth sampled opening says South 1 at the centre. Match mode and current wind remain separate fields. Test-only for every round and viewpoint. Rivers and melds are visibly empty; only the fourth opening has a separate drawn red dragon."})


if __name__ == "__main__":
    main()
