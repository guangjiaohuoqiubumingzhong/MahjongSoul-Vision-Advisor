"""Replay a frozen corpus: pixels/ROI enter recognition, truth enters scoring only."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from app.capture.frames import load_image, save_image
from app.vision.config import load_config, write_json
from app.vision.pjura import PjuraBank
from app.vision.recognizer import StaticRecognizer, overlay
from app.vision.templates import TemplateBank

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/meld-orientation/manifest.json"


def component_config(config, case, width, height):
    result = deepcopy(config)
    name = "meld_"+case["seat"]
    region = deepcopy(config["regions"][name])
    # The nonstandard article screenshots need a newly located ROI. Standard
    # layout captures retain their existing global UI/back exclusions, as live
    # recognition does; dropping them would manufacture foreground failures.
    if case["category"] == "kan_and_pon":
        for key in ("concealed_hand_exclusions", "exclude_polygons"):
            region.pop(key,None)
    x,y,w,h = case["roi_xywh"]
    region["rect"] = [x/width,y/height,w/width,h/height]
    result["client_size"] = [round(width*768/height),768]
    result["regions"] = {name:region}
    result.pop("action_timer",None)
    result.pop("self_wind",None)
    return result


def group_matches(actual, expected):
    return (actual.get("status")=="confirmed" and actual.get("type")==expected["type"]
            and sorted(actual["labels"])==sorted(expected["labels"])
            and (expected.get("target") is None or actual.get("target")==expected["target"])
            and ("called_positions" not in expected or actual.get("called_positions")==expected["called_positions"]))


def score(region, expected):
    actual = region.get("groups",[])
    # Unknown foreground before a real group must not shift the scoring pair.
    # Match confirmed groups one-to-one; extra unknowns still block completeness.
    unmatched = list(expected)
    matched, wrong = 0, False
    for found in actual:
        if found["status"] != "confirmed":
            continue
        index = next((i for i,e in enumerate(unmatched) if group_matches(found,e)),None)
        if index is None:
            wrong = True
        else:
            matched += 1
            unmatched.pop(index)
    correct = not wrong and not unmatched and len(actual)==len(expected)
    return {"correct":correct and region["complete"], "wrong_confirmation":wrong,
            "safe_negative":not expected and not any(a["status"]=="confirmed" for a in actual),
            "groups_correct":matched}


def evaluate(config, bank, case, recognizer_type=StaticRecognizer):
    path = ROOT / case["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=case["sha256"]:
        raise ValueError("Original image changed: "+str(path))
    pixels = load_image(path)
    height,width = pixels.shape[:2]
    recognizer = recognizer_type(component_config(config,case,width,height),bank)
    started = time.perf_counter()
    observation = recognizer.recognize(pixels,{"evaluation_mode":"native_meld_component_only"})
    elapsed = (time.perf_counter()-started)*1000
    region = observation["regions"]["meld_"+case["seat"]]
    return {"name":case["name"],"family":case["source_family"],"seat":case["seat"],"ms":elapsed,
            "expected_groups":case["expected_groups"],"category":case["category"],
            **score(region,case["expected_groups"]),"complete":region["complete"],
            "groups":[{k:g.get(k) for k in ("type","labels","target","called_positions","called_position_evidence","status")} for g in region["groups"]]},observation,pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",default="after")
    parser.add_argument("--baseline",action="store_true")
    args = parser.parse_args()
    manifest = json.loads(DATA.read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    bank = PjuraBank(TemplateBank(config["templates"],config.get("classifier_manifest")),config["tile_classifier"])
    bank.model.warmup()
    out = ROOT / "reports/meld-dataset" / args.stage
    recognizer_type = StaticRecognizer
    if args.baseline:
        import sys
        from scripts.stress_meld_geometry import baseline
        old_recognizer,old_kan = baseline()
        recognizer_type = old_recognizer.StaticRecognizer
        sys.modules["app.vision.kan"] = old_kan
    rows = []
    for case in manifest["cases"]:
        row,observation,pixels = evaluate(config,bank,case,recognizer_type)
        rows.append(row)
        write_json(out/(case["name"]+".json"),observation)
        save_image(out/(case["name"]+"-overlay.png"),overlay(pixels,observation))
        x,y,w,h = case["roi_xywh"]
        save_image(out/(case["name"]+"-original-roi.png"),pixels[y:y+h,x:x+w])
        print(row["name"],"WRONG" if row["wrong_confirmation"] else "correct" if row["correct"] else "safe_negative" if row["safe_negative"] else "unresolved",row["groups_correct"],flush=True)
    positive = [r for r in rows if r["expected_groups"]]
    report = {"scope":manifest["scope"],"manifest_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
              "positive_regions":len(positive),"correct_positive_regions":sum(r["correct"] for r in positive),
              "expected_groups":sum(len(r["expected_groups"]) for r in positive),"correct_groups":sum(r["groups_correct"] for r in positive),
              "wrong_confirmations":sum(r["wrong_confirmation"] for r in rows),
              "negative_regions":len(rows)-len(positive),"safe_negatives":sum(r["safe_negative"] for r in rows),
              "rows":rows,"code_sha256":{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                    ("app/vision/recognizer.py","app/vision/kan.py","app/vision/detector.py")}}
    if args.baseline:
        report["code_sha256"] = json.loads((ROOT/"reports/meld-dataset/baseline/source.json").read_text(encoding="utf-8"))
    write_json(out/"report.json",report)
    if args.baseline:
        write_json(DATA.parent/"baseline.json",{
            "manifest_sha256":report["manifest_sha256"],"scope":"Archived draft-22 source replay; original labels remain in manifest.json.",
            "rows":[{k:r[k] for k in ("name","correct","groups_correct")} for r in rows]})
    print({k:v for k,v in report.items() if k not in ("rows","code_sha256","scope")},flush=True)


if __name__=="__main__":
    main()
