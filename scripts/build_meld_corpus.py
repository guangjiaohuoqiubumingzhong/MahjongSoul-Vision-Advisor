"""Freeze manually reviewed meld structure labels and original image provenance.

No recognizer output supplies truth. Coordinates only locate reviewed pixels;
expected labels/types/sources are consumed exclusively by the scorer.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from app.capture.frames import load_image
from app.vision.config import load_config, pixel_rect, write_json
from app.vision.detector import global_box

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/meld-orientation"


def group(kind, labels, target, called=None):
    result = {"type":kind, "labels":labels, "target":target}
    if called is not None:
        result["called_positions"] = called
    return result


def main():
    cfg = load_config(ROOT / "configs/window-steam-1366-v2.json")
    records = [json.loads(line) for line in (ROOT / "data/public-mahjong/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    sources = {row["path"]:row for row in records}
    cases = []

    def add(name, path, seat, expected, *, roi=None, canonical=None, source=None, family=None, category="structure"):
        path = Path(path)
        pixels = load_image(ROOT / path)
        height, width = pixels.shape[:2]
        if roi is None:
            region = cfg["regions"]["meld_"+seat]
            x,y,w,h = pixel_rect(region["rect"],1366,768)
            if canonical:
                x,y,w,h = global_box(canonical,(x,y),(w,h),region.get("rotation",0))
            roi = [round(x*width/1366),round(y*height/768),round(w*width/1366),round(h*height/768)]
        cases.append({"name":name, "path":path.as_posix(), "sha256":hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                      "seat":seat, "roi_xywh":roi, "expected_groups":expected,
                      "source_page":source, "source_family":family or str(path.parent), "category":category,
                      "review_method":"Visual review of enlarged original pixels; inherited labels cite their original reviewed annotation.",
                      "split":"regression"})

    # New structure annotations, reviewed before changing the draft-22 code.
    specs = [
        ("cvmaj-top-west-added-east-pon","cvmaj/examples/img/n.png","top",
         [group("kakan",["W"]*4,3,[2,3]),group("pon",["E"]*3,0,[1])],None),
        ("cvmaj-top-dot-chi-eight-pon","cvmaj/examples/templates/multiple/ingame03.png","top",
         [group("chi",["5p","6p","7p"],1,[0]),group("pon",["8m"]*3,0,[1])],None),
        ("steam-right-white-four-pons","steam-community/images/3799875797.jpg","right",
         [group("pon",["P"]*3,3,[1]),group("pon",["9m"]*3,3,[1])],None),
        ("steam-right-two-sou-pon","steam-community/images/3800908090.jpg","right",
         [group("pon",["2s"]*3,2,[2])],None),
        ("steam-top-white-called-first","steam-community/images/3800908090.jpg","top",
         [group("pon",["P"]*3,1,[0])],None),
        ("steam-top-two-sou-chis","steam-community/images/3803049411.jpg","top",
         [group("chi",["5s","6s","7s"],1,[0]),group("chi",["4s","2s","3s"],1,[0])],None),
        ("steam-self-nine-dot-chi","steam-community/images/3803922882.jpg","self",
         [group("chi",["9p","7p","8p"],3,[0])],[500,30,240,90]),
        ("steam-top-two-man-pon-north-closed","steam-community/images/3803922882.jpg","top",
         [group("pon",["2m"]*3,0,[1]),group("ankan",["N"]*4,None,[])],[-25,0,280,78]),
        ("steam-right-two-sou-bird-pons","steam-community/images/3805041403.jpg","right",
         [group("pon",["2s"]*3,0,[0]),group("pon",["1s"]*3,3,[1])],None),
        ("steam-top-two-dot-called-last","steam-community/images/3805041403.jpg","top",
         [group("pon",["2p"]*3,3,[2])],None),
        ("steam-right-three-dot-kan-seven-pon","steam-community/images/3808714033.jpg","right",
         [group("daiminkan",["3p"]*4,0,[0]),group("pon",["7p"]*3,3,[1])],None),
        ("steam-top-sou-chi-bird-pon","steam-community/images/3808714033.jpg","top",
         [group("chi",["3s","4s","5s"],1,[0]),group("pon",["F"]*3,1,[0])],None),
        ("steam-top-two-one-three-sou","steam-community/images/3817141720.jpg","top",
         [group("chi",["2s","1s","3s"],1,[0])],None),
    ]
    for name,relative,seat,expected,canonical in specs:
        source = sources[relative]
        add(name,"data/public-mahjong/"+relative,seat,expected,canonical=canonical,
            source=source.get("source_page") or "https://github.com/xdedss/cvmaj",
            family="cvmaj" if relative.startswith("cvmaj/") else "steam-"+Path(relative).stem)
    # Existing public kan cases include dark/custom-back failures; retain them.
    public = json.loads((ROOT/"data/public-mahjong/kan-scenes/regressions.json").read_text(encoding="utf-8"))
    for case in public["cases"]:
        add("public-kan-"+case["name"],"data/public-mahjong/"+case["path"],case["seat"],[case["expected"]],
            roi=case["roi_xywh"],source=case["source_page"],family=case["source_page"],category="kan_and_pon")
    # Two separately recorded games with previously reviewed full meld groups.
    annotations = ["data/screenshots/user-250819-train/kyoku02-turn06/00000001.annotation.json",
                   "data/replays/user-250821-test-recording-01/screenshots/00001135.annotation.json"]
    for annotation in annotations:
        a = json.loads((ROOT/annotation).read_text(encoding="utf-8"))
        for seat in ("self","right","top","left"):
            targets = [t for t in a["targets"] if t.get("region")=="meld_"+seat and t.get("group_id")]
            if not targets:
                continue
            groups = {}
            for target in targets:
                groups.setdefault(target["group_id"],[]).append(target)
            expected = []
            for tiles in groups.values():
                expected.append(group(tiles[0]["meld_type"],[t["label"] for t in tiles],tiles[0].get("target"),
                                      [i for i,t in enumerate(tiles) if t.get("horizontal")]))
            canonical = [300,30,440,90] if seat=="self" else None
            add(a["game_id"]+"-"+seat,Path(annotation).with_suffix("").with_suffix(".png"),seat,expected,
                canonical=canonical,source=annotation,family=a["game_id"],category="recorded_game")
    for turn in (293,300):
        add("user-red-chi-"+str(turn),f"data/fixtures/right-chi-perspective/turn-{turn}.png","right",
            [group("chi",["5mr","4m","6m"],0,[0])],family="user-20261015-east3",category="user_regression")
    for turn in (62,58):
        add("user-white-pon-"+str(turn),f"data/fixtures/white-meld/turn-{turn}.png","right",
            [group("pon",["P"]*3,3,[1])],family="user-white-meld",category="user_regression")
    # Reviewed negative regions: no exposed meld, only backs or table art.
    for relative in ("cvmaj/examples/templates/multiple/ingame01.png","cvmaj/examples/templates/multiple/ingame02.png",
                     "steam-community/images/3801772233.jpg","steam-community/images/3805320035.jpg",
                     "steam-community/images/3812161238.jpg"):
        for seat in ("right","top"):
            add("empty-"+Path(relative).stem+"-"+seat,"data/public-mahjong/"+relative,seat,[],
                source=sources[relative].get("source_page"),family="cvmaj" if relative.startswith("cvmaj/") else "steam-"+Path(relative).stem,
                category="negative_backs_or_table")
    families = sorted({c["source_family"] for c in cases})
    for case in cases:
        case["family_id"] = families.index(case["source_family"])
    manifest = {"schema_version":1,"created":"2026-10-15", "cases":cases,
                "scope":"Frozen targeted debugging/regression corpus, not a population accuracy estimate or a blind holdout. Repeated views and captures share source_family. No model training.",
                "sources":{"pjura":"https://huggingface.co/datasets/pjura/mahjong_souls_tiles",
                           "cvmaj":"https://github.com/xdedss/cvmaj",
                           "roboflow_screen":"https://universe.roboflow.com/will-fuogp/mahjongsoul-tile-detection",
                           "roboflow_tiles":"https://universe.roboflow.com/majsoul/majsoul-6igdu",
                           "sightread":"https://huggingface.co/datasets/sightread-bench/mahjong-v0.1"},
                "counts":{"cases":len(cases),"unique_images":len({c['sha256'] for c in cases}),
                          "source_families":len(families),"positive_groups":sum(len(c['expected_groups']) for c in cases),
                          "negative_regions":sum(not c['expected_groups'] for c in cases),
                          "seats":dict(Counter(c['seat'] for c in cases))}}
    write_json(OUT/"manifest.json",manifest)
    print(manifest["counts"])


if __name__ == "__main__":
    main()
