"""Train-only HOG/colour SVM experiment; never modifies the active recognizer.

Hand/draw images supply training pixels. Public train tiles supply a cross-role
development check. Optional test diagnosis runs only after freezing artifacts.
Synthetic augmentation is recorded separately and does not add real samples.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np

from app.capture.frames import load_image
from app.vision.config import write_json
from app.vision.faces import tile_face
from app.vision.templates import rotate
from app.vision.svm import (descriptor, FEATURE_VERSION, FEATURE_PARAMETERS, POLICY,
                           HIGHLIGHT_FEATURE_VERSION,HIGHLIGHT_FEATURE_PARAMETERS)

def augment(patch,rng):
    height,width = patch.shape[:2]
    # Includes small rendered sizes and slightly slanted visible faces. Each
    # derived patch retains its single manually reviewed source identity.
    scale = float(rng.choice([1,.75,.5,.375,.25]))
    small = cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    h,w = small.shape[:2]
    source = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    destination = source + rng.normal(0,.018,(4,2)).astype(np.float32)*[w,h]
    warped = cv2.warpPerspective(small,cv2.getPerspectiveTransform(source,destination.astype(np.float32)),
                                 (w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    tint = rng.uniform(.88,1.1,3)
    return np.clip(warped.astype(np.float32)*tint,0,255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-test",action="store_true",help="Frozen train-only candidates on reviewed test crops; diagnostic only")
    parser.add_argument("--highlight-fallback",action="store_true",help="Use the v4 geometry fallback only when the original bright mask fails")
    args = parser.parse_args()
    feature_version = HIGHLIGHT_FEATURE_VERSION if args.highlight_fallback else FEATURE_VERSION
    feature_parameters = HIGHLIGHT_FEATURE_PARAMETERS if args.highlight_fallback else FEATURE_PARAMETERS
    face_parameters = feature_parameters["face"]
    report_name = "svm-highlight-prototype" if args.highlight_fallback else "svm-prototype"
    started = time.perf_counter()
    rng = np.random.default_rng(20261009)
    hands,queries,sources = [],[],[]
    for path in sorted(Path("data").rglob("*.annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if annotation.get("split") != "train" or not annotation.get("reviewed"):
            continue
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent/image
        pixels = load_image(image)
        sources.append({"annotation":str(path),"game_id":annotation["game_id"],
                        "split":"train","reviewed":True,
                        "annotation_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "image_sha256":hashlib.sha256(image.read_bytes()).hexdigest()})
        for target in annotation["targets"]:
            if target.get("kind") != "tile" or target["label"] == "?":
                continue
            x,y,w,h = target["bbox"]
            patch = rotate(pixels[y:y+h,x:x+w],target.get("rotation",0))
            record = (str(path),target,patch)
            (hands if target["region"] in {"hand","draw"} else queries).append(record)
    classes = sorted({target["label"] for _,target,_ in hands})
    if len(classes) != 37:
        raise ValueError("Need all 37 manually reviewed training hand classes")
    features,responses = [],[]
    for _,target,patch in hands:
        for index in range(21):
            feature = descriptor(patch if index == 0 else augment(patch,rng),face_parameters)
            if feature is not None:
                features.append(feature)
                responses.append(classes.index(target["label"]))
    train = np.stack(features)
    labels = np.asarray(responses,np.int32)
    rows = [(path,target,descriptor(patch,face_parameters)) for path,target,patch in queries]
    valid = [(path,target,feature) for path,target,feature in rows if feature is not None]
    query = np.stack([feature for _,_,feature in valid])
    variants,models,artifacts = [],[],[]
    artifact_root = Path("models/vision-svm-prototype")
    artifact_root.mkdir(parents=True,exist_ok=True)
    for c,gamma in [(1,.01),(10,.01),(10,.05),(10,.1),(100,.05)]:
        model = cv2.ml.SVM_create()
        model.setType(cv2.ml.SVM_C_SVC)
        model.setKernel(cv2.ml.SVM_RBF)
        model.setC(c)
        model.setGamma(gamma)
        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,3000,1e-6))
        model.train(train,cv2.ml.ROW_SAMPLE,labels)
        _,predictions = model.predict(query)
        wrong = [{"annotation":path,"id":target["id"],"expected":target["label"],"actual":classes[int(prediction)]}
                 for (path,target,_),prediction in zip(valid,predictions.ravel())
                 if target["label"] != classes[int(prediction)]]
        variants.append({"C":c,"gamma":gamma,"argmax_correct":len(valid)-len(wrong),
                         "query_count":len(queries),"unrectifiable":len(rows)-len(valid),"wrong":wrong})
        print({k:v for k,v in variants[-1].items() if k!="wrong"},flush=True)
        if c==10:
            temporary = artifact_root/f"C{c}-gamma{gamma}.xml.tmp.xml"
            model.save(str(temporary))
            payload = temporary.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            output = artifact_root/f"svm-{digest}.xml"
            if output.exists() and output.read_bytes()!=payload:
                raise ValueError("Existing content-addressed model changed")
            if not output.exists():
                output.write_bytes(payload)
            temporary.unlink()
            models.append(model)
            artifacts.append({"C":c,"gamma":gamma,"path":output.name,"sha256":digest})
    manifest = {"schema_version":1,"scope":"development_verifier_not_independent_acceptance",
                "opencv_version":cv2.__version__,"classes":classes,"sources":sources,
                "seed":20261009,"source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "selection":"C=10, gamma=.01/.05/.1 fixed from train-only cross-role results; no test selection",
                "feature_version":feature_version,"feature_parameters":feature_parameters,"policy":POLICY,
                "models":artifacts}
    digest = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    write_json(artifact_root/f"manifest-{digest}.json",manifest)
    write_json(f"reports/vision/training-{report_name}.json",{
        "scope":"train_only_cross_role_development_experiment_not_independent_acceptance",
        "runtime":"OpenCV SVM on CPU; decision backend unchanged",
        "opencv_version":cv2.__version__,"seed":20261009,"hand_source_tiles":len(hands),
        "derived_training_rows":len(features),"classes":classes,"sources":sources,
        "feature_version":feature_version,"feature_parameters":feature_parameters,
        "augmentation":"20 deterministic sampled render scales, perspective jitter and tints per real hand tile",
        "acceptance_policy":"No confidence calibration; all predictions are development argmax only",
        "variants":variants,"elapsed_s":time.perf_counter()-started})
    if args.validate_test:
        validate_test(models,classes,digest,face_parameters,report_name)


def validate_test(models,classes,manifest_sha256,face_parameters,report_name):
    results = []
    for path in sorted(Path("data").rglob("*.annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if annotation.get("split") != "test" or not annotation.get("reviewed") or annotation.get("frame_quality") != "stable":
            continue
        image = Path(annotation["image"])
        if not image.is_absolute():
            image = path.parent/image
        pixels = load_image(image)
        for target in annotation["targets"]:
            if target.get("kind") != "tile" or target["label"] == "?":
                continue
            x,y,w,h = target["bbox"]
            feature = descriptor(rotate(pixels[y:y+h,x:x+w],target.get("rotation",0)),face_parameters)
            votes = [classes[int(model.predict(feature[None])[1].ravel()[0])] for model in models] if feature is not None else []
            proposal = votes[0] if votes and len(set(votes))==1 else "?"
            results.append({"annotation":str(path),"game_id":annotation["game_id"],"id":target["id"],
                            "region":target["region"],"expected":target["label"],"votes":votes,"proposal":proposal})
    regions = {}
    for row in results:
        counter = regions.setdefault(row["region"],{"total":0,"correct":0,"unknown":0,"wrong":0})
        counter["total"] += 1
        counter["correct"] += row["proposal"]==row["expected"]
        counter["unknown"] += row["proposal"]=="?"
        counter["wrong"] += row["proposal"] not in {"?",row["expected"]}
    write_json(f"reports/vision/test-{report_name}.json",{
        "scope":"frozen_train_candidate_on_manually_located_test_crops_not_end_to_end_acceptance",
        "manifest_sha256":manifest_sha256,"test_games":sorted({row["game_id"] for row in results}),
        "policy":"Three RBF SVMs must agree; no calibrated rejection policy or active runtime integration",
        "regions":regions,"results":results})
    print("test crop diagnostic",regions,flush=True)


if __name__ == "__main__":
    main()
