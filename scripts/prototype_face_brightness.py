"""Train-only comparison of face masks under specular highlights.

Parameters are changed only inside this diagnostic process. The CLI defaults,
template manifests, annotations and original images are left unchanged.
"""
from pathlib import Path
import json

import numpy as np

from app.capture.frames import load_image
from app.vision.config import write_json
from app.vision.faces import FACE_PARAMETERS, FaceGallery, tile_face
from app.vision.templates import rotate


def main():
    hands, public, backs = [], [], []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if annotation.get("split") != "train" or not annotation.get("reviewed"):
            continue
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent/image
        pixels = load_image(image)
        for target in annotation["targets"]:
            if target.get("kind") not in {"tile","back"}:
                continue
            x,y,w,h = target["bbox"]
            record = (str(path),target,rotate(pixels[y:y+h,x:x+w],target.get("rotation",0)))
            if target.get("kind") == "back":
                backs.append(record)
            else:
                (hands if target["region"] in {"hand","draw"} else public).append(record)
    variants = []
    for threshold in [.94,.90,.88,.86,.84,.82,.80]:
        FACE_PARAMETERS["white_min_channel"] = threshold
        entries = [(target,tile_face(patch)) for _,target,patch in hands]
        entries = [(target,face) for target,face in entries if face is not None]
        gallery = FaceGallery(np.stack([face.feature for _,face in entries]))
        results = []
        for path,target,patch in public+backs:
            face = tile_face(patch)
            if face is None:
                results.append({"annotation":path,"id":target["id"],"expected":target["label"],"actual":"?","reason":"unrectifiable"})
                continue
            scores = {}
            for (reference,_),score in zip(entries,gallery.scores(face.feature)):
                label = reference["label"]
                if label == "P" and not face.blank:
                    continue
                scores[label] = max(scores.get(label,-1),float(score))
            ranked = sorted(scores.items(),key=lambda row:row[1],reverse=True)
            label,score = ranked[0]
            accepted = score>=.9 and len(ranked)>1 and score-ranked[1][1]>=.04
            results.append({"annotation":path,"id":target["id"],"expected":target["label"],"actual":label if accepted else "?","best_label":label,"score":score,"area_fraction":face.area_fraction})
        actual_public = results[:len(public)]
        variant = {"white_min_channel":threshold,"hand_rectified":len(entries),"hand_total":len(hands),
                   "public_total":len(public),"public_correct":sum(r["actual"]==r["expected"] for r in actual_public),
                   "public_unknown":sum(r["actual"]=="?" for r in actual_public),
                   "public_wrong":[r for r in actual_public if r["actual"] not in {"?",r["expected"]}],
                   "public_unrectifiable":sum(r.get("reason")=="unrectifiable" for r in actual_public),
                   "back_false_tile_proposals":[r for r in results[len(public):] if r["actual"]!="?"]}
        variants.append(variant)
        print({key:value for key,value in variant.items() if key not in {"public_wrong","back_false_tile_proposals"}} | {"wrong":len(variant["public_wrong"]),"back_false":len(variant["back_false_tile_proposals"])},flush=True)
    write_json("reports/vision/training-face-brightness-prototype.json", {"scope":"train_only_mask_development_not_independent_acceptance","variants":variants})


if __name__ == "__main__":
    main()
