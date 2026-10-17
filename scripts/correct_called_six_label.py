"""Preserve the audit trail for one manually verified held-out label correction."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json

from app.vision.config import write_json
from scripts.label_public_meld_samples import save_reviewed


def main():
    path=Path("data/replays/user-250821-test-recording-01/screenshots/00001135.annotation.json")
    annotation=json.loads(path.read_text(encoding="utf-8"))
    before=deepcopy(annotation)
    target=next(t for t in annotation["targets"] if t["id"]=="meld_right-0")
    assert target["label"] in {"3m","6m"} and target["bbox"]==[1091,70,42,41]
    target["label"]="6m"
    annotation["notes"]=annotation["notes"].replace("Right meld is 3m/4m/5m chi","Right meld is 6m/4m/5m chi")
    evidence=["reports/vision/called-number-label-audit.png","reports/vision/right-meld-original-enlarged.png"]
    annotation["review_evidence"]=list(dict.fromkeys(annotation.get("review_evidence",[])+evidence))
    reason="Enlarged original called tile has the dot, cross-stroke and two lower strokes of six, matching existing train 6m rather than three horizontal strokes. Right public meld is 4m/5m/6m chi. Correct test truth only; model votes are not the ground truth."
    save_reviewed(path,annotation,reason=reason)
    write_json("reports/vision/called-six-label-correction.json",{
        "scope":"manual_test_truth_correction_not_model_improvement","annotation":str(path),
        "image_sha256":hashlib.sha256(Path(annotation["image"]).read_bytes()).hexdigest(),
        "before":next(t for t in before["targets"] if t["id"]=="meld_right-0"),"after":target,
        "reason":reason,"evidence":evidence,"training_data_changed":False,"gate_passed":False})
    print("Confirmed held-out called 6m; preserved annotation correction trail")


if __name__=="__main__":
    main()
