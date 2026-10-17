"""Compare training hand glyphs against training public poses at several scales.

This development experiment reads train only and never modifies the active bank.
Derived downsampled prototypes are not additional real samples.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from app.capture.frames import load_image
from app.vision.config import write_json
from app.vision.faces import FaceGallery, tile_face
from app.vision.templates import rotate


def main():
    hands,queries = {},[]
    for path in sorted(Path("data").rglob("*.annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        if annotation.get("split") != "train" or not annotation.get("reviewed"):
            continue
        pixels = load_image(annotation["image"])
        for target in annotation["targets"]:
            if target.get("kind") != "tile" or target["label"] == "?":
                continue
            x,y,w,h = target["bbox"]
            patch = rotate(pixels[y:y+h,x:x+w],target.get("rotation",0))
            if target["region"] in {"hand","draw"}:
                hands.setdefault(target["label"],patch)
            else:
                queries.append((str(path),target,patch))
    variants = []
    for scales in [(1,),(1,.75,.5),(1,.75,.5,.375),(1,.75,.5,.375,.25)]:
        labels,features = [],[]
        for label,patch in hands.items():
            for scale in scales:
                candidate = patch if scale==1 else cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                face = tile_face(candidate)
                if face is not None:
                    labels.append(label);features.append(face.feature)
        gallery = FaceGallery(np.stack(features))
        argmax, accepted, wrong, missing = 0,0,[],[]
        for path,target,patch in queries:
            face = tile_face(patch)
            if face is None:
                missing.append({"annotation":path,"id":target["id"]});continue
            scores = {}
            for label,value in zip(labels,gallery.scores(face.feature)):
                if label=="P" and not face.blank:
                    continue
                scores[label] = max(scores.get(label,-1),float(value))
            ranked = sorted(scores.items(),key=lambda pair:pair[1],reverse=True)
            if not ranked:
                missing.append({"annotation":path,"id":target["id"]});continue
            best,score = ranked[0];margin = score-ranked[1][1] if len(ranked)>1 else 0
            correct = best==target["label"]
            argmax += correct
            known = score>=.9 and margin>=.04
            accepted += known
            if known and not correct:
                wrong.append({"annotation":path,"id":target["id"],"expected":target["label"],
                              "actual":best,"score":score,"margin":margin})
        variants.append({"scales":list(scales),"prototype_count":len(features),"query_count":len(queries),
                         "argmax_correct":argmax,"accepted":accepted,"accepted_wrong":wrong,"unrectifiable":missing})
    report = {"scope":"train_only_cross_role_development_comparison_not_independent_acceptance",
              "classes":sorted(hands),"tile_threshold":.9,"margin_threshold":.04,"variants":variants}
    write_json("reports/vision/training-glyph-resolution-prototype.json",report)
    for value in variants:
        print({k:v for k,v in value.items() if k not in {"accepted_wrong","unrectifiable"}},
              "wrong",len(value["accepted_wrong"]),"unrectifiable",len(value["unrectifiable"]))


if __name__ == "__main__":
    main()
