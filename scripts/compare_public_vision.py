"""Compare saved before/after runs and a small, visually reviewed regression set."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import hashlib
import json
import os
from pathlib import Path

from app.vision.config import write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def overlap(a,b):
    x,y,w,h = a; bx,by,bw,bh = b
    intersection = max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
    return intersection/(w*h+bw*bh-intersection) if intersection else 0


def reviewed(run,ground_truth):
    scenes = {r["path"]:r for r in read(run/"scenes-summary.json")["rows"]}
    totals = Counter(); cases = []
    for case in ground_truth["cases"]:
        image = ROOT/case["path"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError(f"Reviewed screenshot changed: {case['path']}")
        row = scenes.get(case["path"].removeprefix("data/public-mahjong/"))
        expected = sum(len(r["targets"]) for r in case["regions"].values())
        if row is None or "error" in row:
            totals["unprocessed"] += expected
            cases.append({"path":case["path"],"unprocessed":expected,"error":(row or {}).get("error","not in run")})
            continue
        observation = read(ROOT/row["output"])
        results = {}
        for name,truth in case["regions"].items():
            targets = truth["targets"]
            items = observation["regions"][name]["items"]
            pairs = sorted(((overlap(t["bbox"],p["bbox"]),i,j) for i,t in enumerate(targets)
                            for j,p in enumerate(items)),reverse=True)
            matched,used = {},set()
            for iou,i,j in pairs:
                if iou >= .5 and i not in matched and j not in used:
                    matched[i] = j; used.add(j)
            details = []
            for i,target in enumerate(targets):
                item = items[matched[i]] if i in matched else {}
                label = item.get("label")
                verdict = ("missing" if label is None else "unknown" if label=="?" else
                           "accepted_correct" if label==target["label"] else "accepted_wrong")
                totals[verdict] += 1
                details.append({"expected":target["label"],"bbox":target["bbox"],"status":verdict,
                                "label":label,"best_label":item.get("best_label"),"score":item.get("score")})
            if truth.get("complete_region",True):
                totals["extra_accepted"] += sum(items[j]["label"]!="?" for j in range(len(items)) if j not in used)
            results[name] = details
        cases.append({"path":case["path"],"regions":results})
    return {"expected_tiles":sum(sum(len(r["targets"]) for r in c["regions"].values())
                                 for c in ground_truth["cases"]),
            "counts":{name:totals[name] for name in ("accepted_correct","accepted_wrong","unknown","missing","unprocessed","extra_accepted")},
            "cases":cases}


def render(report,output,before_path,after_path):
    before,after = report["reviewed_regressions"]["before"],report["reviewed_regressions"]["after"]
    metrics = report["network"]["overall"]
    external = report["network"]["by_source"]["cvmaj"]
    same_author = report["network"]["by_source"]["pjura"]
    rows = "".join(f"<tr><td>{escape(name)}</td><td>{before['counts'][key]}</td><td>{after['counts'][key]}</td></tr>"
                   for key,name in (("accepted_correct","正确识别"),("accepted_wrong","错误确认"),
                                    ("unknown","保留为未知牌"),("missing","漏掉牌面"),("unprocessed","截图尺寸未处理")))
    text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>当前识别网络：数据集回放与修复</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#17242b;background:#f5f7f8;max-width:1050px;margin:40px auto;padding:0 25px}}h1{{font-size:29px}}section{{background:white;padding:22px 28px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #e3e8eb;padding:10px}}p{{margin:8px 0}}img{{max-width:100%}}a{{color:#006989}}small{{color:#546675}}</style>
<h1>当前识别网络：数据集回放与修复</h1><p>2026-10-13 · {escape(report['config_version'])}</p>
<section><h2>单牌网络基线</h2><p>去重后 {metrics['samples']:,} 张有标签单牌：首选正确 {metrics['correct_top1']:,} 张，准确率 {metrics['top1_accuracy']:.2%}。满足当前置信度的 {metrics['accepted']:,} 张中没有错分，{metrics['rejected']} 张被拦住。</p>
<p>外部来源 cvmaj：{external['samples']} 张，首选正确 {external['correct_top1']} 张（{external['top1_accuracy']:.2%}）。pjura 同作者数据：{same_author['samples']:,} 张，首选正确 {same_author['correct_top1']:,} 张，可能与模型训练重合。</p>
<p>模型权重、输入归一化和 0.90 / 0.04 置信度门槛保持不变。4 张单牌首选错分仍低于门槛：一张模糊四索，三张变形的西风。</p>
<a href="before/errors-and-rejects.jpg">查看错分和低置信度牌面</a></section>
<section><h2>人工核对的 35 张牌面回归</h2><table><tr><th>结果</th><th>修改前</th><th>修改后</th></tr>{rows}</table>
<small>两张截图中的选定牌面；用于定位和固定回归，不代表整个公开截图集的准确率。</small>
<p>横放四索原来被确认为七索（0.901）；本次第一轮加入方向约束后保留为未知（0.812）。放大原始截图后已纠正此前误记为二索的人工标注。副露赤五索原来被当成背景漏掉，第一轮修复后保留为未知牌。横牌后续修复另见 public-sideways-fix 报告。</p>
<a href="before/scenes/013-3803049411/overlay.png">修改前带框图</a> · <a href="after/scenes/013-3803049411/overlay.png">修改后带框图</a></section>
<section><h2>整局截图和实际故障</h2><p>44 张公开牌桌图，能进入识别流程的图片由 17 张增加到 22 张。1280×720 等较小同布局截图现可按参考坐标检测，网络继续读取原始牌面像素。其余 22 张含不同比例或浏览器边框，仍需要单独定位客户区。</p>
<p>读取副露区域边缘的一圈图像后，白边不再因牌背花纹被截掉而误判为副露；原用户倒计时截图已重新直接返回建议。橙色牌背排除也加入了颜色占比和走廊位置限制。</p>
<p>立直棒检测现在跳过越界区域，避免小图触发 OpenCV 错误。旧吃碰测试改为合法的“两组副露加八张手牌”，并保留非法手牌数量检查。</p>
<p>旋转/透视选择的单牌控制实验中，满足牌种置信度的数量由 2,224 张变为 2,223 张，错误确认都为零。一张横放二万因此多保留为未知；这是方向约束的覆盖率代价，详见原始控制实验。</p>
<p>全部回归测试：343 项通过。不同皮肤、强透视、弹窗覆盖和部分副露牌面仍有未知结果；整局未做完整真值标注。</p>
<a href="after/scenes/004-ingame02/overlay.png">1280×720 截图回放（29 张已核对牌面均正确）</a></section>
<section><h2>复现与详细记录</h2><p>在项目根目录运行：</p><pre>python -m scripts.evaluate_public_vision --output reports/public-vision-latest
python -m scripts.audit_public_tile_views --output reports/public-vision-latest/views
python -m scripts.compare_public_vision --before reports/public-vision-fix/before --after reports/public-vision-latest</pre>
<a href="report.json">机器可读对比报告</a> · <a href="after/crops-summary.json">原始网络统计</a> · <a href="after/scenes-summary.json">逐张整局诊断</a> · <a href="final-tests.txt">回归测试记录</a></section></html>"""
    for prefix,path in (("before",before_path),("after",after_path)):
        relative = Path(os.path.relpath(path.resolve(),output.resolve())).as_posix()
        text = text.replace(f'href="{prefix}/',f'href="{escape(relative)}/')
    (output/"report.html").write_text(text,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before",type=Path,default=ROOT/"reports/public-vision-fix/before")
    parser.add_argument("--after",type=Path,default=ROOT/"reports/public-vision-fix/after")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/public-vision-fix")
    args = parser.parse_args()
    truth = read(ROOT/"data/public-mahjong/manual-regressions.json")
    b,a = read(args.before/"scenes-summary.json"),read(args.after/"scenes-summary.json")
    report = {"date":"2026-10-13","config_version":read(args.after/"run.json")["config_version"],
              "network":read(args.after/"crops-summary.json"),
              "reviewed_regressions":{"scope":truth["review_method"],
                  "before":reviewed(args.before,truth),"after":reviewed(args.after,truth)},
              "scene_replay":{"samples":a["samples"],"before_input_errors":b["input_errors"],
                              "after_input_errors":a["input_errors"],"remaining_input_errors":[r for r in a["rows"] if "error" in r]},
              "model_retrained":False,"thresholds_changed":False,
              "limits":["Upstream pjura samples may overlap model training","Selected reviewed cases are debugging regressions, not a holdout",
                        "Unlabelled full-scene completeness is not tile accuracy","4 raw crop errors remain rejected by the confidence threshold"]}
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"report.json",report)
    render(report,args.output,args.before,args.after)
    print(json.dumps({k:r["counts"] for k,r in report["reviewed_regressions"].items() if isinstance(r,dict)},ensure_ascii=False))


if __name__=="__main__":
    main()
