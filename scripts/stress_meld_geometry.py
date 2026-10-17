"""Controlled projective poses: physical truth precedes corner-order perturbation.

These are synthetic geometry cases, not additional real photographs or network
accuracy samples. The model, glyph predictions and primary thresholds are not
involved. Measured side-view compression is exercised at all four seats.
"""
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import cv2
import numpy as np

from app.vision.config import load_config,write_json
from app.vision.recognizer import group_melds

ROOT = Path(__file__).resolve().parents[1]
POSES = ((1,1,0,0),(.55,1,.12,-.25),(1,.75,-.12,0),(.7,1,-.12,.25))


def controlled_items(actor, kind, called, shift, perspective, blank):
    sx,sy,kx,ky = POSES[actor]
    matrix = np.asarray([[sx,kx,80],[ky,sy,90],[perspective,0,1]],np.float32)
    widths = [42 if i==called else 28 for i in range(4 if kind=="daiminkan" else 3)]
    items = []
    x = 0
    for i,w in enumerate(widths):
        h = 28 if i==called else 42
        points = np.asarray([[x,42-h],[x+w,42-h],[x+w,42],[x,42]],np.float32)
        points = cv2.perspectiveTransform(points[None],matrix)[0]
        lo = np.floor(points.min(axis=0))
        hi = np.ceil(points.max(axis=0))
        local = points-lo
        labels = ["P"]*len(widths) if blank else ["5pr"]+["5p"]*(len(widths)-1)
        items.append({"label":labels[i],"horizontal":None,
                      "canonical_bbox":[float(lo[0]),float(lo[1]),float(hi[0]-lo[0]),float(hi[1]-lo[1])],
                      "legacy_evidence":{"blank_face":blank,
                                         "face_quad_local":np.roll(local,shift,axis=0).tolist()}})
        x += w+3
    return items


def baseline():
    folder = ROOT/"reports/meld-dataset/baseline"
    hashes = json.loads((folder/"source.json").read_text(encoding="utf-8"))
    modules = []
    for name in ("recognizer","kan"):
        path = folder/"code/app/vision"/(name+".py")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes["app/vision/"+name+".py"]:
            raise ValueError("Baseline code changed")
        spec = importlib.util.spec_from_file_location("app.vision._meld_baseline_"+name,path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def main():
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")
    old_recognizer,old_kan = baseline()
    rows = []
    timings = []
    for actor,kind,shift,perspective,blank in itertools.product(range(4),("pon","daiminkan"),range(4),(-.0004,0,.0004),(False,True)):
        seat = ("self","right","top","left")[actor]
        region = config["regions"]["meld_"+seat]
        for called in range(4 if kind=="daiminkan" else 3):
            items = controlled_items(actor,kind,called,shift,perspective,blank)
            # Independent public layout rule, not the config lookup under test.
            offset = (3,2,2,1)[called] if kind=="daiminkan" else (3,2,1)[called]
            target = (actor+offset)%4
            # The archived triplet function imports .kan at call time. Pin its
            # row check too, so current code cannot leak into the baseline.
            import app.vision.kan as live_kan
            current_row = live_kan.row_geometry
            live_kan.row_geometry = old_kan.row_geometry
            try:
                before = old_kan.kan_structure(deepcopy(items),region) if kind=="daiminkan" else old_recognizer.group_melds(deepcopy(items),region)[0]
            finally:
                live_kan.row_geometry = current_row
            started = time.perf_counter_ns()
            after = group_melds(deepcopy(items),region)[0]
            timings.append((time.perf_counter_ns()-started)/1e6)
            def correct(g):
                return bool(g and g.get("type")==kind and g.get("target")==target and g.get("called_positions")==[called])
            rows.append({"seat":seat,"type":kind,"called":called,"corner_shift":shift,"perspective":perspective,
                         "blank":blank,"before_correct":correct(before),"after_correct":correct(after)})
    report = {"scope":"Synthetic projective geometry only; no real images or model inference. Repeated corner-order variants are not independent samples.",
              "cases":len(rows),"before_correct":sum(r["before_correct"] for r in rows),
              "after_correct":sum(r["after_correct"] for r in rows),
              "geometry_median_ms":float(np.median(timings)),"geometry_p95_ms":float(np.quantile(timings,.95)),"rows":rows}
    write_json(ROOT/"reports/meld-dataset/projective-stress.json",report)
    print({k:v for k,v in report.items() if k!="rows"})


if __name__=="__main__":
    main()
