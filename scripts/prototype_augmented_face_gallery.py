"""Train-only scale gallery; optional frozen held-out crop diagnosis.

Source images and the active recognizer are never changed. Derived features
are not new real samples, and crop results are not end-to-end acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np

from app.capture.frames import load_image
from app.vision.config import load_config,write_json
from app.vision.faces import FaceGallery,tile_face
from app.vision.templates import TemplateBank,rotate

SCALES=(1,.75,.5,.375,.25,.20,.167)


def score(gallery,labels,face):
    values={}
    for label,value in zip(labels,gallery.scores(face.feature)):
        if label=="P" and not face.blank:
            continue
        values[label]=max(values.get(label,-1),float(value))
    ranked=sorted(values.items(),key=lambda pair:pair[1],reverse=True)
    best,value=ranked[0]
    margin=value-ranked[1][1] if len(ranked)>1 else 0
    return {"best_label":best,"score":value,"margin":margin,
            "label":best if value>=.9 and margin>=.04 else "?"}


def records(split):
    for path in sorted(Path("data").rglob("*.annotation.json")):
        annotation=json.loads(path.read_text(encoding="utf-8-sig"))
        if annotation.get("split")!=split or not annotation.get("reviewed") or annotation.get("frame_quality")!="stable":
            continue
        image=Path(annotation["image"])
        if not image.is_absolute():
            image=path.parent/image
        pixels=load_image(image)
        for target in annotation["targets"]:
            if target.get("kind") not in {"tile","back"} or target["label"]=="?":
                continue
            x,y,w,h=target["bbox"]
            yield path,annotation,target,rotate(pixels[y:y+h,x:x+w],target.get("rotation",0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--validate-test",action="store_true")
    args=parser.parse_args()
    config=load_config("configs/window-steam-1366-v2.json")
    bank=TemplateBank(config["templates"],config.get("classifier_manifest"))
    training=list(records("train"))
    rows,features,labels,sources=[],[],[],[]
    for path,annotation,target,patch in training:
        if target.get("kind")!="tile" or target["region"] not in {"hand","draw"}:
            continue
        sources.append({"annotation":str(path),"game_id":annotation["game_id"],"id":target["id"],
                        "image_sha256":hashlib.sha256(Path(annotation["image"]).read_bytes()).hexdigest()})
        for scale in SCALES:
            candidate=patch if scale==1 else cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            face=tile_face(candidate,bank.face_parameters)
            if face is not None:
                labels.append(target["label"])
                features.append(face.feature)
    values=np.stack(features)
    digest=hashlib.sha256(values.tobytes()+json.dumps(labels).encode()).hexdigest()
    gallery=FaceGallery(values)
    for path,annotation,target,patch in training:
        if target["region"] in {"hand","draw"}:
            continue
        face=tile_face(patch,bank.face_parameters)
        result=score(gallery,labels,face) if face is not None else {"label":"?","reason":"unrectifiable"}
        rows.append({"annotation":str(path),"id":target["id"],"expected":target["label"],"kind":target["kind"],**result})
    def metrics(rows):
        known=[r for r in rows if r["kind"]=="tile"]
        return {"tiles":len(known),"correct":sum(r["label"]==r["expected"] for r in known),
                "unknown":sum(r["label"]=="?" for r in known),
                "wrong":[r for r in known if r["label"] not in {"?",r["expected"]}],
                "back_false_tiles":[r for r in rows if r["kind"]=="back" and r["label"]!="?"]}
    training_report={"scope":"train_only_cross_role_experiment_not_acceptance","scales":SCALES,
                     "derived_features":len(features),"feature_sha256":digest,"sources":sources,
                     "face_parameters":bank.face_parameters,"metrics":metrics(rows),"rows":rows}
    write_json("reports/vision/training-augmented-face-gallery.json",training_report)
    print("training",{k:v if not isinstance(v,list) else len(v) for k,v in metrics(rows).items()})
    if args.validate_test:
        rows=[]
        for path,annotation,target,patch in records("test"):
            face=tile_face(patch,bank.face_parameters)
            result=score(gallery,labels,face) if face is not None else {"label":"?","reason":"unrectifiable"}
            rows.append({"annotation":str(path),"id":target["id"],"expected":target["label"],"kind":target["kind"],"region":target["region"],**result})
        write_json("reports/vision/test-augmented-face-gallery.json",{
            "scope":"frozen_train_features_on_manually_located_test_crops_not_end_to_end_acceptance",
            "feature_sha256":digest,"metrics":metrics(rows),"rows":rows})
        print("test",{k:v if not isinstance(v,list) else len(v) for k,v in metrics(rows).items()})


if __name__=="__main__":
    main()
