"""Calibrate river face polygons on train only, then compare fixed-slot scanning.

This experiment does not enable or modify the default recognizer. Evaluation
annotations are used only after inference, never to select occupied slots.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from app.advice import load_screenshot
from app.capture.frames import load_image, save_image
from app.vision.config import load_config, pixel_rect, to_reference, write_json
from app.vision.detector import global_box, river_slots, tile_candidates
from app.vision.evaluate import match_boxes
from app.vision.pjura import PjuraBank, file_sha256
from app.vision.recognizer import StaticRecognizer, mark_sideways_by_shape
from app.vision.rectify import order_image_quad, top_face_outline
from app.vision.templates import TemplateBank, rotate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/river-polygons"
RIVERS = ("river_self", "river_right", "river_top", "river_left")


def annotations():
    rows = []
    for path in sorted((ROOT / "data").rglob("*.annotation.json")):
        a = json.loads(path.read_text(encoding="utf-8-sig"))
        if not a.get("reviewed") or a.get("frame_quality") != "stable": continue
        image = Path(a["image"])
        if not image.is_absolute(): image = path.parent / image
        if not image.is_file(): continue
        rows.append((path, a, image))
    return rows


def local_box(box, region, config):
    x,y,w,h = pixel_rect(region["rect"], *config["client_size"])
    bx,by,bw,bh = box
    size = (h,w) if region.get("rotation",0) in (90,270) else (w,h)
    return global_box([bx-x,by-y,bw,bh],(0,0),size,(-region.get("rotation",0))%360)


def features(c, r):
    return np.array([1., c, r, c*r])


def calibrate(config, rows):
    found = defaultdict(lambda: defaultdict(list))
    sources = []
    for path,a,image in rows:
        if a["split"] != "train": continue
        reference,_,_ = to_reference(load_image(image),config)
        sources.append({"annotation":path.relative_to(ROOT).as_posix(),"image":image.relative_to(ROOT).as_posix(),
                        "sha256":file_sha256(image),"game_id":a["game_id"]})
        for name in RIVERS:
            region=config["regions"][name]
            x,y,w,h=pixel_rect(region["rect"],*config["client_size"])
            crop=rotate(reference[y:y+h,x:x+w],region.get("rotation",0))
            targets=[t for t in a["targets"] if t.get("kind")=="tile" and t.get("region")==name and "slot" in t]
            horizontal={tuple(t["slot"]) for t in targets if t.get("horizontal") is True}
            for target in targets:
                r,c=target["slot"]
                if any(hr==r and hc<=c for hr,hc in horizontal): continue
                bx,by,bw,bh=local_box(target["bbox"],region,config)
                patch=crop[max(0,by):by+bh,max(0,bx):bx+bw]
                if not patch.size:continue
                geometry=top_face_outline(patch)
                if geometry is None:continue
                quad=order_image_quad(geometry.quad)+[bx,by]
                # Upright is defined by the reviewed physical orientation;
                # foreshortening may make its image wider than it is tall.
                found[name][(r,c)].append(quad)
    model={"reference_size":config["client_size"],"sources":sources,"regions":{},
           "method":"robust per-corner bilinear fit from normal train faces; frozen polygons for every slot"}
    for name in RIVERS:
        observed=found[name]
        coordinates=sorted(observed)
        X=np.stack([features(c,r) for r,c in coordinates])
        Y=np.stack([np.median(observed[s],axis=0).reshape(-1) for s in coordinates])
        weights=np.ones(len(X))
        for _ in range(6):
            coef=np.linalg.lstsq(X*weights[:,None],Y*weights[:,None],rcond=None)[0]
            errors=np.max(np.linalg.norm((X@coef-Y).reshape(-1,4,2),axis=2),axis=1)
            weights=np.minimum(1.,3./np.maximum(errors,.01))
        region=config["regions"][name];grid=region["grid"]
        polygons=[{"slot":[r,c],"quad":(features(c,r)@coef).reshape(4,2).tolist(),
                   "calibration_samples":len(observed.get((r,c),[]))}
                  for r in range(grid["rows"]) for c in range(grid["columns"])]
        model["regions"][name]={"coefficients":coef.tolist(),"polygons":polygons,
            "observed_slots":len(coordinates),"geometry_max_residual":float(errors.max()),
            "geometry_median_residual":float(np.median(errors))}
    write_json(OUT/"calibration.json",model)
    return model


def quad_at(model,name,r,c,horizontal=False):
    coef=np.array(model["regions"][name]["coefficients"])
    quad=(features(c,r)@coef).reshape(4,2)
    if horizontal:
        # A physical 2:3 face becomes 3:2 when turned. Keep the leading edge
        # position and centre line, increasing the following slots by 1/2 cell.
        centre=quad.mean(axis=0)
        u=((quad[1]-quad[0])+(quad[2]-quad[3]))/2
        v=((quad[3]-quad[0])+(quad[2]-quad[1]))/2
        centre=centre+.25*u
        quad=np.array([centre-.75*u-v/3,centre+.75*u-v/3,
                       centre+.75*u+v/3,centre-.75*u+v/3])
    return quad.astype(np.float32)


def warp(crop,quad,horizontal=False):
    size=(94,64) if horizontal else (64,94)
    w,h=size
    dx,dy=.03*w,.03*h
    target=np.array([[dx,dy],[w-1-dx,dy],[w-1-dx,h-1-dy],[dx,h-1-dy]],np.float32)
    return cv2.warpPerspective(crop,cv2.getPerspectiveTransform(quad.astype(np.float32),target),size,
        flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))


def presence(face):
    face=cv2.resize(face,(32,48),interpolation=cv2.INTER_AREA)
    bgr=face.astype(np.int16)
    white=(bgr.min(axis=2)>=145)&(bgr.max(axis=2)-bgr.min(axis=2)<=40)
    inner=white[3:-3,3:-3]
    border=np.concatenate((white[3:8,3:-3].ravel(),white[-8:-3,3:-3].ravel(),
                           white[8:-8,3:7].ravel(),white[8:-8,-7:-3].ravel()))
    edges=[white[3:8,3:-3].mean(),white[-8:-3,3:-3].mean(),
           white[8:-8,3:7].mean(),white[8:-8,-7:-3].mean()]
    # A neighbour's grey thickness fills only one part of an empty slot. It
    # lacks a whole edge; dense glyphs still leave some white on all four.
    return {"white":float(inner.mean()),"border":float(border.mean()),"minimum_edge":float(min(edges)),
            "occupied":bool(inner.mean()>=.25 and border.mean()>=.4 and min(edges)>.04)}


def sideways_layout(crop,region):
    """Retain the existing independent body-shape check for riichi exceptions.

    No glyph labels or evaluation annotations are used to locate a sideways
    tile. Normal tile faces still use only frozen calibration polygons.
    """
    boxes=tile_candidates(crop,region)["boxes"]
    first=river_slots(boxes,region["grid"])
    items=[{"slot":slot,"canonical_bbox":box,"horizontal":False} for box,(slot,_) in zip(boxes,first)]
    mark_sideways_by_shape(items,region["grid"])
    slots=river_slots(boxes,region["grid"],[i["horizontal"] for i in items])
    counts=Counter(tuple(slot) for slot,_ in slots)
    return {tuple(slot):i for i,(slot,residual) in zip(items,slots)
            if counts[tuple(slot)]==1 and residual<=region["grid"]["tolerance"]}


def scan(pixels, config, model, bank, original_bank=None, *, fallback=False):
    reference,native,scale=to_reference(pixels,config)
    native=pixels if native is None else native
    results={}
    for name in RIVERS:
        region=config["regions"][name];degrees=region.get("rotation",0)
        x,y,w,h=pixel_rect(region["rect"],*config["client_size"])
        # Warps read original-resolution pixels; geometry stays in the frozen
        # 1366x768 coordinate system.
        nx,ny,nw,nh=pixel_rect(region["rect"],native.shape[1],native.shape[0])
        crop=rotate(native[ny:ny+nh,nx:nx+nw],degrees)
        reference_crop=rotate(reference[y:y+h,x:x+w],degrees)
        sx,sy=crop.shape[1]/reference_crop.shape[1],crop.shape[0]/reference_crop.shape[0]
        layout=sideways_layout(reference_crop,region)
        sideways={s:i for s,i in layout.items() if i["horizontal"]}
        items=[];empty=[];faces=[];plans=[]
        for r in range(region["grid"]["rows"]):
            shift=0.
            for c in range(region["grid"]["columns"]):
                slot=[r,c]
                horizontal=(r,c) in sideways
                quad=quad_at(model,name,r,c+shift,horizontal)
                face=warp(crop,quad*[sx,sy],horizontal)
                evidence=presence(face)
                if not evidence["occupied"]:
                    empty.append({"slot":slot,**evidence})
                    continue
                if horizontal:
                    pitch=float(np.linalg.norm(region["grid"]["basis"][0]))
                    shift+=max(0.,sideways[(r,c)]["canonical_bbox"][2]/pitch-1)
                low=np.floor(quad.min(axis=0)).astype(int);high=np.ceil(quad.max(axis=0)).astype(int)
                box=[int(low[0]),int(low[1]),int(high[0]-low[0]),int(high[1]-low[1])]
                item={"slot":slot,"canonical_bbox":box,"bbox":global_box(box,(x,y),(w,h),degrees),
                    "quad":quad.tolist(),"presence":evidence,"horizontal":horizontal}
                items.append(item)
                if horizontal and original_bank is not None:
                    # Reuse the validated existing horizontal path; the new
                    # experiment replaces only normal, stationary slot crops.
                    bx,by,bw,bh=sideways[(r,c)]["canonical_bbox"]
                    nx,ny,nw,nh=np.rint(np.array([bx,by,bw,bh])*[sx,sy,sx,sy]).astype(int)
                    result=original_bank.classify_many([reference_crop[by:by+bh,bx:bx+bw]],name,.9,.04,(0,90,270),
                        model_crops=[crop[ny:ny+nh,nx:nx+nw]])[0]
                    item.update(result)
                    plans.append(None)
                elif horizontal:
                    faces.extend([rotate(face,90),rotate(face,270)])
                    plans.append((len(faces)-2,len(faces)))
                else:
                    faces.append(face);plans.append((len(faces)-1,len(faces)))
        matches=bank.classify_many(faces,name,.9,.04,(0,),kinds=("tile",)) if faces else []
        for item,plan in zip(items,plans):
            if plan is None:continue
            first,last=plan
            choices=matches[first:last]
            selected=max(choices,key=lambda m:(m["score"],m["margin"]))
            if fallback and selected["label"]=="?" and original_bank is not None and tuple(item["slot"]) in layout:
                bx,by,bw,bh=layout[tuple(item["slot"])]["canonical_bbox"]
                nx,ny,nw,nh=np.rint(np.array([bx,by,bw,bh])*[sx,sy,sx,sy]).astype(int)
                original=original_bank.classify_many([reference_crop[by:by+bh,bx:bx+bw]],name,.9,.04,(0,90,270),
                    model_crops=[crop[ny:ny+nh,nx:nx+nw]])[0]
                selected={**original,"fixed_polygon_reading":selected,"fallback":"fixed_polygon_unknown"}
            item.update(selected)
        results[name]={"items":items,"empty":empty,"complete":all(i["label"]!="?" for i in items)}
    return results


def score(expected, actual):
    counts=Counter(expected=len(expected),correct=0,wrong=0,unknown=0,missed=0,extra=0,extra_accepted=0,wrong_slot=0,wrong_horizontal=0)
    matches,missing,extra=match_boxes(expected,actual,minimum_iou=.2)
    for ei,ai in matches:
        e,a=expected[ei],actual[ai]
        counts["correct" if a["label"]==e["label"] else "unknown" if a["label"]=="?" else "wrong"]+=1
        counts["wrong_slot"]+=e.get("slot")!=a.get("slot")
        counts["wrong_horizontal"]+=a.get("horizontal") is not None and e.get("horizontal",False)!=a.get("horizontal")
    counts["missed"]+=len(missing);counts["extra"]+=len(extra)
    counts["extra_accepted"]+=sum(actual[i]["label"]!="?" for i in extra)
    return dict(counts)


def fixture_score(labels, horizontal, items):
    expected={tuple(divmod(i,6)):label for i,label in enumerate(labels)}
    by_slot={tuple(i["slot"]):i for i in items}
    counts=Counter(expected=len(labels),correct=0,wrong=0,unknown=0,missed=0,extra=0,extra_accepted=0,wrong_horizontal=0)
    for slot,label in expected.items():
        item=by_slot.get(slot)
        if item is None:counts["missed"]+=1;continue
        counts["correct" if item["label"]==label else "unknown" if item["label"]=="?" else "wrong"]+=1
        counts["wrong_horizontal"]+=item.get("horizontal") is not None and item["horizontal"]!=(list(slot) in horizontal)
    for slot,item in by_slot.items():
        if slot not in expected:
            counts["extra"]+=1;counts["extra_accepted"]+=item["label"]!="?"
    return dict(counts)


def fixtures(config, model, bank, fixed_bank, baseline):
    manifest=json.loads((ROOT/"data/fixtures/river-polygons/manifest.json").read_text(encoding="utf-8"))
    rows=[];metrics=defaultdict(lambda:defaultdict(Counter))
    for case in manifest["cases"]:
        original=load_image(ROOT/case["path"])
        if case.get("sha256") and file_sha256(ROOT/case["path"])!=case["sha256"]:raise ValueError("Fixture pixels changed")
        for size in ((1920,1080),(1366,768)):
            pixels=original if size==(1920,1080) else cv2.resize(original,size,interpolation=cv2.INTER_AREA)
            observations={};timings={}
            for mode in ("before","after","guarded"):
                baseline.river_memory.clear()
                for b in (bank,fixed_bank):b._cache.clear();b._classification_cache.clear()
                started=time.perf_counter()
                observations[mode]=(baseline.recognize(pixels)["regions"] if mode=="before" else
                    scan(pixels,config,model,fixed_bank,bank,fallback=mode=="guarded"))
                timings[mode]=(time.perf_counter()-started)*1000
            row={"name":case["name"],"path":case["path"],"sha256":file_sha256(ROOT/case["path"]),
                 "size":size,"timings":timings,"regions":{}}
            for name,labels in case["regions"].items():
                row["regions"][name]={}
                for mode,obs in observations.items():
                    value=fixture_score(labels,case["horizontal_slots"].get(name,[]),obs[name]["items"])
                    row["regions"][name][mode]=value
                    row["regions"][name][mode+"_items"]=obs[name]["items"]
                    metrics[str(size[0])][mode].update(value)
            rows.append(row)
            write_json(OUT/"fixture-frames"/(case["name"]+"-"+str(size[0])+".json"),row)
            print("FIXTURE",case["name"],size,round(timings["before"]),round(timings["guarded"]),flush=True)
    return {"scope":manifest["scope"],"manifest_sha256":file_sha256(ROOT/"data/fixtures/river-polygons/manifest.json"),
            "rows":rows,"metrics":{s:{k:dict(v) for k,v in m.items()} for s,m in metrics.items()}}


def main():
    # Keep the comparison baseline fixed even after adopting the new reader.
    baseline_path=OUT/"baseline/configs/window-steam-1366-v2.json"
    config=load_config(baseline_path if baseline_path.is_file() else ROOT/"configs/window-steam-1366-v2.json")
    config.pop("river_polygons",None)
    rows=annotations()
    model=calibrate(config,rows)
    bank=PjuraBank(TemplateBank(config["templates"],config.get("classifier_manifest")),config["tile_classifier"])
    bank.model.warmup()
    fixed_bank=PjuraBank(bank.legacy,{**config["tile_classifier"],"rectification":"off"},model=bank.model)
    # Shared model, separate caches. No evaluation label reaches either classifier.
    river_config={**config,"regions":{n:config["regions"][n] for n in RIVERS}}
    baseline=StaticRecognizer(river_config,bank)
    metrics=defaultdict(lambda:defaultdict(Counter));records=[]
    for index,(path,a,image) in enumerate(rows):
        pixels=load_image(image)
        baseline.river_memory.clear();bank._classification_cache.clear();bank._cache.clear()
        started=time.perf_counter();before=baseline.recognize(pixels)
        before_ms=(time.perf_counter()-started)*1000
        fixed_bank._classification_cache.clear();fixed_bank._cache.clear()
        bank._classification_cache.clear();bank._cache.clear()
        started=time.perf_counter();after=scan(pixels,config,model,fixed_bank,bank)
        after_ms=(time.perf_counter()-started)*1000
        for b in (bank,fixed_bank):b._cache.clear();b._classification_cache.clear()
        started=time.perf_counter();guarded=scan(pixels,config,model,fixed_bank,bank,fallback=True)
        guarded_ms=(time.perf_counter()-started)*1000
        frame={"annotation":path.relative_to(ROOT).as_posix(),"image":image.relative_to(ROOT).as_posix(),
            "sha256":file_sha256(image),"split":a["split"],"game_id":a["game_id"],
            "distinct":a.get("count_as_distinct_scene",True),"before_ms":before_ms,"after_ms":after_ms,"guarded_ms":guarded_ms,"regions":{}}
        for name in RIVERS:
            if name not in a.get("complete_regions",[]):continue
            expected=[t for t in a["targets"] if t.get("kind")=="tile" and t.get("region")==name]
            part={"expected":expected,"before":score(expected,before["regions"][name]["items"]),
                  "after":score(expected,after[name]["items"]),"guarded":score(expected,guarded[name]["items"]),
                  "after_items":after[name]["items"],"guarded_items":guarded[name]["items"],
                  "before_items":before["regions"][name]["items"]}
            frame["regions"][name]=part
            if frame["distinct"]:
                for stage in ("before","after","guarded"):metrics[a["split"]][stage].update(part[stage])
        records.append(frame)
        write_json(OUT/"frames"/(f"{index:03d}.json"),frame)
        print(index,a["split"],image.parent.name,round(before_ms),round(after_ms),round(guarded_ms),flush=True)
    report={"measured_at":datetime.now(timezone.utc).isoformat(),"config_version":config["version"],
        "scope":"Fixed slot polygons, trained geometry on reviewed train frames only; test games previously used in project development, not a new blind holdout. Empty slots scanned independently. Distinct-scene metrics exclude duplicate render.",
        "calibration_sha256":file_sha256(OUT/"calibration.json"),"metrics":{s:{k:dict(v) for k,v in m.items()} for s,m in metrics.items()},
        "records":records,"default_changed":False,
        "source_hashes":{p:file_sha256(ROOT/p) for p in ("app/vision/actions.py","app/vision/recognizer.py","app/vision/pjura.py","configs/window-steam-1366-v2.json","scripts/evaluate_river_polygons.py")}}
    report["fixtures"]=fixtures(config,model,bank,fixed_bank,baseline)
    write_json(OUT/"experiment.json",report)
    print("METRICS",report["metrics"],flush=True)
    print("FIXTURE_METRICS",report["fixtures"]["metrics"],flush=True)


if __name__=="__main__": main()
