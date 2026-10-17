"""Compare the integrated polygon reader with the unchanged default config."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np

from app.advice import ScreenshotAdvice, load_screenshot
from app.capture.frames import load_image
from app.vision.config import load_config, write_json
from app.vision.pjura import PjuraBank, file_sha256
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader
from scripts.evaluate_river_polygons import annotations, fixture_score, score, RIVERS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/river-polygons"


def clear(service):
    service.clear();service.recognizer.river_memory.clear()
    banks=[service.recognizer.bank]
    if service.recognizer.river_polygons is not None:banks.append(service.recognizer.river_polygons.fixed)
    for bank in banks:bank._cache.clear();bank._classification_cache.clear()
    if service.recognizer.text_reader:service.recognizer.text_reader._read_cache.clear()


def state(result):
    obs=result["observation"]
    return {"status":result["status"],"reason":result["reason"],"snapshot":result["snapshot"],
        "action":(result.get("recommendation") or {}).get("event"),
        "mask":(result.get("decision") or {}).get("action_mask"),
        "scores":[c["model_score"] for c in (result.get("decision") or {}).get("legal_candidates",[])],
        "rivers":{n:{"complete":obs["regions"][n]["complete"],
            "items":[[i["label"],i.get("slot"),i.get("horizontal")] for i in obs["regions"][n]["items"]]} for n in RIVERS}}


def verified_pose_correction(row,case,config):
    """Allow only a snapshot change explicitly supported by physical truth."""
    b,a=row["before"],row["after"]
    if b["snapshot"]==a["snapshot"] or any(b[k]!=a[k] for k in ("status","action","mask")):return None
    expected=deepcopy(b["snapshot"])
    corrected=[]
    for name,labels in case["regions"].items():
        player=config["regions"][name]["player"]
        river=expected["rivers"][player]
        if len(river)!=len(labels) or [i["tile"] for i in river]!=labels:return None
        for i,item in enumerate(river):
            horizontal=list(divmod(i,6)) in case["horizontal_slots"].get(name,[])
            if item["riichi"]!=horizontal:
                corrected.append({"region":name,"slot":list(divmod(i,6)),"label":item["tile"],
                                  "old":item["riichi"],"truth":horizontal})
                item["riichi"]=horizontal
    return {"source":"data/fixtures/river-polygons/manifest.json","corrections":corrected} if corrected and expected==a["snapshot"] else None


def main():
    baseline=load_config(OUT/"baseline/configs/window-steam-1366-v2.json")
    candidate=load_config(OUT/"candidate-config.json")
    bank=PjuraBank(TemplateBank(baseline["templates"],baseline.get("classifier_manifest")),baseline["tile_classifier"])
    reader=TextReader()
    services={"before":ScreenshotAdvice(baseline,recognizer=StaticRecognizer(baseline,bank,reader)),
              "after":ScreenshotAdvice(candidate,recognizer=StaticRecognizer(candidate,bank,reader))}
    services["before"].prepare();services["after"].backend=services["before"].backend
    report={"measured_at":datetime.now(timezone.utc).isoformat(),"default_changed":False,
        "scope":"Existing reviewed games and regression screenshots, not a new blind holdout. Geometry calibrated only on train. Public unlabelled boards are diagnostics, never scored as accurate.",
        "source_sha256":{p:file_sha256(ROOT/p) for p in ("app/vision/river_polygons.py","app/vision/recognizer.py","app/vision/actions.py","configs/river-polygons.json")}}
    metrics=defaultdict(lambda:defaultdict(Counter));rows=[]
    for index,(path,a,image) in enumerate(annotations()):
        row={"annotation":path.relative_to(ROOT).as_posix(),"split":a["split"],"distinct":a.get("count_as_distinct_scene",True),"regions":{}}
        observed={}
        for mode,service in services.items():
            clear(service);observed[mode]=service.recognizer.recognize(load_image(image))
        for n in RIVERS:
            if n not in a.get("complete_regions",[]):continue
            expected=[t for t in a["targets"] if t.get("kind")=="tile" and t.get("region")==n]
            row["regions"][n]={}
            for mode,obs in observed.items():
                values=score(expected,obs["regions"][n]["items"])
                row["regions"][n][mode]=values
                if row["distinct"]:metrics[a["split"]][mode].update(values)
        rows.append(row)
        print("ANNOTATION",index,a["split"],flush=True)
    report["annotations"]={"metrics":{s:{m:dict(v) for m,v in modes.items()} for s,modes in metrics.items()},"rows":rows}
    manifest=json.loads((ROOT/"data/fixtures/river-polygons/manifest.json").read_text(encoding="utf-8"))
    scenes=[];fixture_metrics=defaultdict(lambda:defaultdict(Counter))
    paths=[ROOT/c["path"] for c in manifest["cases"]]
    extra=[ROOT/"data/fixtures"/p for p in ("kan-recognition/top-3p-ankan-1920.png",
        "kan-all-seats/user-self-F-kakan-1920.png","no-draw-call-discard/chi-buttons-1920.png",
        "no-draw-call-discard/ron-buttons-1920.png")]
    for i,path in enumerate(paths+extra):
        original,metadata=load_screenshot(path,baseline)
        for size in ((1920,1080),(1366,768)):
            pixels=original if size==(1920,1080) else cv2.resize(original,size,interpolation=cv2.INTER_AREA)
            row={"path":path.relative_to(ROOT).as_posix(),"size":size,"sha256":file_sha256(path)}
            for mode,service in services.items():
                clear(service)
                result=service.process(pixels,metadata)
                row[mode]={"ms":result["total_ms"],**state(result)}
                if i<len(paths):
                    case=manifest["cases"][i]
                    row[mode]["truth"]={}
                    for n,labels in case["regions"].items():
                        values=fixture_score(labels,case["horizontal_slots"].get(n,[]),result["observation"]["regions"][n]["items"])
                        row[mode]["truth"][n]=values;fixture_metrics[str(size[0])][mode].update(values)
                write_json(OUT/"integrated-scenes"/(f"{i:02d}-{size[0]}-{mode}.json"),result)
            row["changes"]=[k for k in ("status","reason","snapshot","action","mask","scores","rivers") if row["before"][k]!=row["after"][k]]
            if i<len(paths):row["verified_pose_correction"]=verified_pose_correction(row,manifest["cases"][i],baseline)
            scenes.append(row)
            print("SCENE",i,size,row["before"]["status"],row["after"]["status"],row["changes"],flush=True)
    report["scenes"]={"fixture_metrics":{s:{m:dict(v) for m,v in modes.items()} for s,modes in fixture_metrics.items()},"rows":scenes}
    # Full public boards: quantify changes and preserve every record, including
    # unsupported dimensions. No labels are inferred from a previous model.
    public=[];public_truth=defaultdict(Counter)
    partial=json.loads((ROOT/"data/public-mahjong/manual-regressions.json").read_text(encoding="utf-8"))
    reviewed=json.loads((ROOT/"data/fixtures/river-polygons/public-review.json").read_text(encoding="utf-8"))
    records=[json.loads(s) for s in (ROOT/"data/public-mahjong/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    records=[r for r in records if r.get("scene")=="board" and not r["duplicate_of"]]
    for index,record in enumerate(records):
        path=ROOT/"data/public-mahjong"/record["path"];row={"path":path.relative_to(ROOT).as_posix()}
        for mode,service in services.items():
            clear(service)
            try:
                pixels,metadata=load_screenshot(path,service.config)
                obs=service.recognizer.recognize(pixels,metadata)
                row[mode]={n:{"complete":obs["regions"][n]["complete"],
                    "items":[{k:i.get(k) for k in ("label","slot","horizontal","bbox","score","fixed_polygon_fallback")} for i in obs["regions"][n]["items"]]} for n in RIVERS}
                for case in partial["cases"]:
                    if case["path"]!=row["path"]:continue
                    for n,part in case["regions"].items():
                        if n not in RIVERS:continue
                        values=score(part["targets"],obs["regions"][n]["items"])
                        # The old public review has boxes but no slot labels.
                        values.pop("wrong_slot",None);public_truth[mode].update(values)
                for case in reviewed["cases"]:
                    if case["path"]!=row["path"]:continue
                    if file_sha256(path)!=case["sha256"]:raise ValueError("Reviewed public pixels changed")
                    item=next((i for i in obs["regions"][case["region"]]["items"] if i.get("slot")==case["slot"]),None)
                    key="missed" if item is None else "unknown" if item["label"]=="?" else "correct" if item["label"]==case["label"] else "wrong"
                    public_truth[mode]["changed_case_"+key]+=1
            except ValueError as exc:row[mode]={"error":str(exc)}
        public.append(row)
        if index%10==0:print("PUBLIC",index+1,len(records),flush=True)
    report["public_boards"]={"scope":"Unlabelled diagnostics, not accuracy. Standalone tile crops have no river coordinates and are not applicable.",
        "reviewed_metrics":{m:dict(v) for m,v in public_truth.items()},"rows":public}
    # Accurate labels must not be lost or replaced with wrong/extra accepted
    # tiles. This is a necessary gate; final tests and full-flow comparison
    # remain necessary before changing the default.
    blockers=[]
    for scope,modes in [("annotation-"+s,m) for s,m in report["annotations"]["metrics"].items()]+[("fixture-"+s,m) for s,m in report["scenes"]["fixture_metrics"].items()]:
        b,a=modes["before"],modes["after"]
        if a["correct"]<b["correct"] or any(a[k]>b[k] for k in ("wrong","missed","extra_accepted")):
            blockers.append(scope)
    ready_regressions=[r["path"]+"-"+str(r["size"][0]) for r in scenes if r["before"]["status"]=="ready" and not r.get("verified_pose_correction") and
                       any(k in r["changes"] for k in ("status","snapshot","action","mask","scores"))]
    times={mode:[r[mode]["ms"] for r in scenes] for mode in services}
    report["gate"]={"label_blockers":blockers,"ready_state_changes":ready_regressions,
        "full_process_median_ms":{m:float(np.median(t)) for m,t in times.items()},
        "verified_pose_corrections":[r["verified_pose_correction"] for r in scenes if r.get("verified_pose_correction")],
        "passed":not blockers and not ready_regressions}
    write_json(OUT/"integrated.json",report)
    print("LABELS",report["annotations"]["metrics"],flush=True)
    print("FIXTURES",report["scenes"]["fixture_metrics"],flush=True)
    print("GATE",report["gate"],flush=True)


if __name__=="__main__":main()
