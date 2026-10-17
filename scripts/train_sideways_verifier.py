"""Build a frozen auxiliary glyph verifier; keep public test/external images out."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.capture.frames import load_image
from app.state.events import BASE
from app.vision.config import write_json
from app.vision.faces import HIGHLIGHT_FACE_PARAMETERS
from app.vision.sideways import POLICY,digest
from app.vision.svm import descriptor,HIGHLIGHT_FEATURE_PARAMETERS,HIGHLIGHT_FEATURE_VERSION

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/public-mahjong"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"models/sideways-glyph")
    args = parser.parse_args()
    records = [json.loads(s) for s in (DATA/"manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    heldout = {r["sha256"] for r in records if r["source"]!="pjura" or "/test/" in r["path"]}
    training = [r for r in records if r["source"]=="pjura" and "/train/" in r["path"]
                and r.get("label_mjai") and not r["duplicate_of"] and r["sha256"] not in heldout]
    classes = sorted(BASE); sources = []; features = []; labels = []
    for record in training:
        feature = descriptor(load_image(DATA/record["path"]),HIGHLIGHT_FACE_PARAMETERS)
        if feature is None:
            continue
        features.append(feature); labels.append(classes.index(record["label_mjai"]))
        sources.append({"path":record["path"],"sha256":record["sha256"],"label":record["label_mjai"],
                        "source":"pjura","split":"train","label_basis":"published_upstream_label"})
    features = np.asarray(features,np.float32); labels = np.asarray(labels,np.int32)
    if set(labels.tolist()) != set(range(len(classes))):
        raise ValueError("Training sources do not cover all 34 classes")
    args.output.mkdir(parents=True,exist_ok=True)
    archive = args.output/"features.npz"
    np.savez_compressed(archive,features=features,labels=labels)
    entries = []
    for gamma in POLICY["svm_gammas"]:
        model = cv2.ml.SVM_create(); model.setType(cv2.ml.SVM_C_SVC); model.setKernel(cv2.ml.SVM_RBF)
        model.setC(POLICY["svm_c"]); model.setGamma(gamma)
        model.train(features,cv2.ml.ROW_SAMPLE,labels)
        path = args.output/f"svm-{gamma}.xml"; model.save(str(path))
        entries.append({"path":path.name,"sha256":digest(path),"gamma":gamma})
    write_json(args.output/"manifest.json",{"schema_version":1,"classes":classes,"policy":POLICY,
               "feature_version":HIGHLIGHT_FEATURE_VERSION,
               "feature_parameters":HIGHLIGHT_FEATURE_PARAMETERS,"opencv_version":cv2.__version__,
               "features":{"path":archive.name,"sha256":digest(archive)},"models":entries,"sources":sources,
               "corpus_manifest_sha256":digest(DATA/"manifest.jsonl"),"heldout_hashes_excluded":len(heldout),
               "training_scope":"Deduplicated pjura published train only; all public test and external source hashes excluded"})
    print(f"Frozen glyph verifier: {len(sources)} training images, {len(classes)} classes",flush=True)


if __name__=="__main__":main()
