"""Audit production rotation/perspective selection independently of templates."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from app.capture.frames import load_image
from app.state.events import base
from app.vision.config import load_config, write_json
from app.vision.pjura import PjuraBank, PjuraModel

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/public-mahjong"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=load_config(ROOT/"configs/window-steam-1366-v2.json")["tile_classifier"]
    model=PjuraModel(config["manifest"],provider=config["provider"],batch_size=config["batch_size"])
    legacy=SimpleNamespace(classify=lambda *a,**k:{"label":"?","score":0,"margin":0,"face_rotation":0},sha256="no-template-probe")
    bank=PjuraBank(legacy,config,model=model)
    samples=[json.loads(line) for line in (DATA/"manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    samples=[r for r in samples if r["kind"]=="tile_crop" and r.get("label_mjai") and not r["duplicate_of"]]
    rows=[]
    for start in range(0,len(samples),64):
        group=samples[start:start+64]
        results=bank.classify_many([load_image(DATA/r["path"]) for r in group],"river",.9,.04,(0,90,270))
        for r,p in zip(group,results):
            glyph=p.get("pjura_evidence",{}).get("sideways_glyph",{})
            identity_accepted=(glyph.get("accepted",False) or p["score"]>=bank.threshold and p["margin"]>=bank.margin_threshold
                               and p.get("reason")!="pjura_rectification_disagrees")
            rows.append({"path":r["path"],"source":r["source"],"expected":r["label_mjai"],
                         "accepted":identity_accepted,"prediction":p})
        if start%512==0:print(f"Production view audit: {min(start+64,len(samples))}/{len(samples)}",flush=True)
    summary={}
    for source in ("all","cvmaj","pjura"):
        group=[r for r in rows if source=="all" or r["source"]==source]
        accepted=[r for r in group if r["accepted"]]
        summary[source]={"samples":len(group),"accepted":len(accepted),
                         "accepted_wrong":sum(r["prediction"]["best_label"]!=r["expected"] for r in accepted),
                         "best_label_correct":sum(r["prediction"]["best_label"]==r["expected"] for r in group),
                         "reasons":dict(Counter(r["prediction"].get("reason") or "accepted" for r in group)),
                         "confusions":Counter(f'{r["expected"]}->{r["prediction"]["best_label"]}' for r in accepted if r["prediction"]["best_label"]!=r["expected"])}
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"summary.json",summary)
    (args.output/"predictions.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
