"""Compare the sideways fallback on reprojected external tiles, with no retraining."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.capture.frames import load_image
from app.state.events import base
from app.vision.config import load_config,write_json
from app.vision.pjura import PjuraBank
from app.vision.sideways import digest
from app.vision.templates import rotate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/public-mahjong"


def project(pixels,degrees,squeeze):
    upright = rotate(pixels,90) if pixels.shape[1] > 1.25*pixels.shape[0] else pixels
    face = rotate(cv2.resize(upright,(64,94),interpolation=cv2.INTER_AREA),degrees)
    h = 64*squeeze
    quad = np.float32([[4,5],[98,2],[94,h+3],[1,h+6]])
    size = (103,int(np.ceil(h))+18)
    canvas = np.full((size[1],size[0],3),(80,60,40),np.uint8)
    side = np.float32([quad[3],quad[2],quad[2]+[0,9],quad[3]+[0,9]])
    cv2.fillPoly(canvas,[np.rint(side).astype(np.int32)],(175,178,181))
    source = np.float32([[0,0],[93,0],[93,63],[0,63]])
    transform = cv2.getPerspectiveTransform(source,quad)
    projected = cv2.warpPerspective(face,transform,size,flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((64,94),np.uint8)*255,transform,size,flags=cv2.INTER_NEAREST)
    canvas[mask>0] = projected[mask>0]
    return canvas


def measure(rows,key):
    labels = [base(r[key]["label"]) for r in rows]
    return {"samples":len(rows),"correct":sum(v==r["expected"] for v,r in zip(labels,rows)),
            "unknown":labels.count("?"),"wrong":sum(v not in ("?",r["expected"]) for v,r in zip(labels,rows))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"reports/public-sideways-fix/external-poses")
    args = parser.parse_args()
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")["tile_classifier"]
    legacy = SimpleNamespace(sha256="no-template-control",classify=lambda *a,**k:{"label":"?","score":0,"margin":0})
    after = PjuraBank(legacy,config)
    primary = deepcopy(config); primary.pop("sideways_verifier_manifest")
    before = PjuraBank(legacy,primary,model=after.model)
    before._cache,before.lock = after._cache,after.lock
    records = [json.loads(s) for s in (DATA/"manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r["source"]=="cvmaj" and r.get("label_mjai") and not r["duplicate_of"]]
    jobs = [(r,d,s) for r in records for d in (90,270) for s in (.5,.75)]
    rows = []
    for start in range(0,len(jobs),32):
        group = jobs[start:start+32]
        inputs = [project(load_image(DATA/r["path"]),d,s) for r,d,s in group]
        old = before.classify_many(inputs,"river",.9,.04,(0,90,270))
        new = after.classify_many(inputs,"river",.9,.04,(0,90,270))
        for (record,degrees,squeeze),b,a in zip(group,old,new):
            rows.append({"path":record["path"],"expected":record["label_mjai"],"rotation":degrees,
                         "squeeze":squeeze,"before":b,"after":a,
                         "glyph_used":a["acceptance_policy"].startswith("sideways_glyph")})
        if start%256==0:print(f"External sideways poses: {min(start+32,len(jobs))}/{len(jobs)}",flush=True)
    used = [r for r in rows if r["glyph_used"]]
    summary = {"scope":"Controlled re-projections of external cvmaj images; source hashes absent from auxiliary training. Identity only, red-five GT not supplied upstream. This is not real-scene accuracy.",
               "before":measure(rows,"before"),"after":measure(rows,"after"),
               "glyph_used":measure(used,"after"),
               "svm_manifest_sha256":after.sideways_verifier.sha256,"model_sha256":after.model.manifest["model_sha256"],
               "corpus_manifest_sha256":digest(DATA/"manifest.jsonl"),
               "transforms":{"rotations":[90,270],"short_axis_scales":[.5,.75],"added_grey_side_pixels":9},
               "changed":[{"path":r["path"],"rotation":r["rotation"],"squeeze":r["squeeze"],
                            "expected":r["expected"],"before":r["before"]["label"],"after":r["after"]["label"]}
                           for r in rows if r["before"]["label"]!=r["after"]["label"]]}
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"summary.json",summary)
    (args.output/"predictions.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("before","after","glyph_used")}),flush=True)


if __name__=="__main__":main()
