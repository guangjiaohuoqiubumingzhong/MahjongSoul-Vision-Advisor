"""Frozen river face corners; body/slot checks and the original reader remain.

Calibration contains geometry from train frames only. A polygon supplies a
normal face crop, never a tile identity or evidence that an empty slot is full.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .config import VisionError
from .detector import river_slots
from .pjura import PjuraBank


def warp_face(pixels, quad):
    target=np.array([[1.92,2.82],[61.08,2.82],[61.08,90.18],[1.92,90.18]],np.float32)
    return cv2.warpPerspective(pixels,cv2.getPerspectiveTransform(np.asarray(quad,np.float32),target),
                              (64,94),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)


def face_presence(face):
    pixels=cv2.resize(face,(32,48),interpolation=cv2.INTER_AREA).astype(np.int16)
    white=(pixels.min(axis=2)>=145)&(pixels.max(axis=2)-pixels.min(axis=2)<=40)
    inner=white[3:-3,3:-3]
    edges=[white[3:8,3:-3].mean(),white[-8:-3,3:-3].mean(),
           white[8:-8,3:7].mean(),white[8:-8,-7:-3].mean()]
    border=np.concatenate((white[3:8,3:-3].ravel(),white[-8:-3,3:-3].ravel(),
                           white[8:-8,3:7].ravel(),white[8:-8,-7:-3].ravel()))
    return {"white":float(inner.mean()),"border":float(border.mean()),"minimum_edge":float(min(edges)),
            "occupied":bool(inner.mean()>=.25 and border.mean()>=.4 and min(edges)>.04)}


def layout_signature(config):
    return {n:{k:r[k] for k in ("rect","rotation","grid")} for n,r in config["regions"].items()
            if r["kind"]=="river" and all(k in r for k in ("rect","rotation","grid"))}


class RiverPolygonReader:
    def __init__(self, config, bank):
        value=config["river_polygons"]
        path=Path(value["manifest"])
        try:
            raw=path.read_bytes()
            self.sha256=hashlib.sha256(raw).hexdigest()
            if self.sha256!=value["sha256"]:raise ValueError("标定文件校验失败")
            self.model=json.loads(raw)
            if self.model["reference_size"]!=config["client_size"]:raise ValueError("参考尺寸不匹配")
            layout=layout_signature(config)
            if any(self.model["layout"].get(n)!=r for n,r in layout.items()):raise ValueError("牌河布局与标定不匹配")
            for name,region in self.model["regions"].items():
                coef=np.asarray(region["coefficients"],np.float64)
                if coef.shape!=(4,8) or not np.isfinite(coef).all():raise ValueError("标定角点无效")
                grid=self.model["layout"][name]["grid"]
                if len(region["polygons"])!=grid["rows"]*grid["columns"]:raise ValueError("标定牌位数量无效")
                for index,polygon in enumerate(region["polygons"]):
                    quad=np.asarray(polygon["quad"],np.float32)
                    if (polygon["slot"]!=list(divmod(index,grid["columns"])) or quad.shape!=(4,2)
                            or not np.isfinite(quad).all() or not cv2.isContourConvex(quad)
                            or abs(cv2.contourArea(quad))<4):raise ValueError("标定牌位四角无效")
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise VisionError("固定牌河多边形配置无效："+str(exc)) from exc
        self.original=bank
        # A new crop must pass the primary confidence threshold before taking
        # the fast path. Less certain warps use the original reader, including
        # its configured majority/dominance rule and independent verifiers.
        self.fixed=PjuraBank(bank.legacy,{**config["tile_classifier"],"rectification":"off",
                                         "dominant_candidate":None},model=bank.model)

    def classify(self, name, region, reference, native, boxes, threshold, margin, *, raw_model_crops=None):
        from .recognizer import mark_sideways_by_shape
        slots=river_slots(boxes,region["grid"])
        poses=[{"slot":slot,"canonical_bbox":box,"horizontal":False} for box,(slot,_) in zip(boxes,slots)]
        mark_sideways_by_shape(poses,region["grid"])
        slots=river_slots(boxes,region["grid"],[p["horizontal"] for p in poses])
        multiplicities=Counter(tuple(slot) for slot,_ in slots)
        pitch=float(np.linalg.norm(region["grid"]["basis"][0]))
        coefficients=np.asarray(self.model["regions"][name]["coefficients"])
        scale=np.array([native.shape[1]/reference.shape[1],native.shape[0]/reference.shape[0]])
        warped=[];legacy=[];indices=[];evidence={};results=[None]*len(boxes)
        for i,(box,pose,(slot,residual)) in enumerate(zip(boxes,poses,slots)):
            r,c=slot
            if (pose["horizontal"] or min(slot)<0 or r>=region["grid"]["rows"] or c>=region["grid"]["columns"]
                    or residual>region["grid"]["tolerance"] or multiplicities[tuple(slot)]!=1):continue
            shift=sum(max(0.,p["canonical_bbox"][2]/pitch-1) for p,(s,_) in zip(poses,slots)
                      if p["horizontal"] and s[0]==r and s[1]<c)
            stored=self.model["regions"][name]["polygons"][r*region["grid"]["columns"]+c]
            quad=np.asarray(stored["quad"],np.float32)
            if shift:
                quad=quad+(np.array([0.,shift,0.,shift*r])@coefficients).reshape(4,2).astype(np.float32)
            bx,by,bw,bh=box
            # Reject a polygon extending substantially outside the independently
            # observed body. This also protects against transient misalignment.
            polygon_area=abs(cv2.contourArea(quad))
            body=np.array([[bx,by],[bx+bw,by],[bx+bw,by+bh],[bx,by+bh]],np.float32)
            intersection,_=cv2.intersectConvexConvex(quad,body)
            if polygon_area<=0 or intersection/polygon_area<.85:continue
            face=warp_face(native,quad*scale)
            presence=face_presence(face)
            if not presence["occupied"]:continue
            indices.append(i);warped.append(face);legacy.append(reference[by:by+bh,bx:bx+bw])
            evidence[i]={"manifest_sha256":self.sha256,"quad_local":quad.tolist(),"slot":slot,
                         "row_shift_cells":shift,"presence":presence,"model_input_size":[64,94]}
        if indices:
            matches=self.fixed.classify_many(legacy,name,threshold,margin,(0,),model_crops=warped)
            for i,match in zip(indices,matches):
                if match["label"]=="?":continue
                results[i]={**match,"face_rotation":0,"identity_rotation":0,"orientation_confirmed":True,
                            "orientation_source":"calibrated_river_slot","fixed_polygon":evidence[i]}
        fallback=[i for i,result in enumerate(results) if result is None]
        if fallback:
            ref_crops=[];native_crops=[]
            for i in fallback:
                bx,by,bw,bh=boxes[i]
                ref_crops.append(reference[by:by+bh,bx:bx+bw])
                left,top=np.rint(np.array([bx,by])*scale).astype(int)
                right,bottom=np.rint(np.array([bx+bw,by+bh])*scale).astype(int)
                native_crops.append(raw_model_crops[i] if raw_model_crops is not None else native[top:bottom,left:right])
            matches=self.original.classify_many(ref_crops,name,threshold,margin,(0,90,270),model_crops=native_crops)
            for i,match in zip(fallback,matches):
                results[i]={**match,"fixed_polygon_fallback":"uncertain_or_special_geometry"}
        return results
