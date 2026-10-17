"""Display current unknown public tiles and their frozen visual evidence."""
from pathlib import Path
import json

import cv2
import numpy as np

from app.capture.frames import load_image,save_image
from app.vision.config import load_config,write_json,pixel_rect
from app.vision.faces import tile_face
from app.vision.svm import descriptor
from app.vision.templates import TemplateBank,rotate


def main():
    config=load_config("configs/window-steam-1366-v2.json")
    bank=TemplateBank(config["templates"],config.get("classifier_manifest"))
    report=json.loads(Path("reports/stage2-acceptance.json").read_text(encoding="utf-8"))
    rows,records,seen=[],[],set()
    for error in report["errors"]:
        if error["category"]!="tile_classification_or_slot":
            continue
        annotation=json.loads(Path(error["annotation"]).read_text(encoding="utf-8-sig"))
        image=Path(annotation["image"])
        if not image.is_absolute():
            image=Path(error["annotation"]).parent/image
        actual=error["actual"]
        key=(tuple(actual["bbox"]),error["expected"]["label"])
        if key in seen:
            continue
        seen.add(key)
        pixels=load_image(image)
        region=config["regions"][error["region"]]
        rx,ry,rw,rh=pixel_rect(region["rect"],*config["client_size"])
        canonical=rotate(pixels[ry:ry+rh,rx:rx+rw],region.get("rotation",0))
        x,y,w,h=actual["canonical_bbox"]
        patch=rotate(canonical[y:y+h,x:x+w],actual.get("face_rotation",0))
        face=tile_face(patch,bank.face_parameters)
        feature=descriptor(patch,bank.verifier.feature_parameters["face"])
        votes=[bank.verifier.classes[int(model.predict(feature[None])[1].ravel()[0])] for model in bank.verifier.models] if feature is not None else []
        record={"annotation":error["annotation"],"expected":error["expected"]["label"],"region":error["region"],
                "bbox":actual["bbox"],"best_label":actual.get("best_label"),"score":actual.get("score"),
                "diagnostic_votes_without_acceptance":votes,"quad":face.quad.tolist() if face is not None else None}
        records.append(record)
        canvas=np.full((185,760,3),35,np.uint8)
        title=f"{error['region']} GT:{record['expected']} best:{record['best_label']} {record['score']:.3f} votes:{','.join(votes)}"
        cv2.putText(canvas,title,(5,18),0,.43,(255,255,255),1)
        marked=patch.copy()
        if face is not None:
            cv2.polylines(marked,[face.quad.astype(np.int32)],True,(0,255,0),1)
        views=[patch,marked,(face.feature*255).astype(np.uint8) if face is not None else np.zeros((60,40,3),np.uint8)]
        identifier=actual.get("template_id")
        reference=next((entry for entry,_ in bank.entries if entry["id"]==identifier),None)
        views.append(load_image(bank.path.parent/reference["file"]) if reference else np.zeros((60,40,3),np.uint8))
        for index,view in enumerate(views):
            canvas[25:180,5+185*index:175+185*index]=cv2.resize(view,(170,155),interpolation=cv2.INTER_NEAREST)
        rows.append(canvas)
    write_json("reports/vision/public-failure-inspection.json",{"scope":"diagnosis_only_not_new_training_or_acceptance","records":records})
    if rows:
        save_image("reports/vision/public-failure-inspection.png",np.vstack(rows))
    print(json.dumps(records,ensure_ascii=True))


if __name__=="__main__":
    main()
