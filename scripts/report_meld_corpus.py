"""Build an auditable comparison, including every unresolved original ROI."""
import hashlib
from html import escape
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from app.vision.config import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"reports/meld-dataset"


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def main():
    before,after,stress,latency = (read(n) for n in ("before/report.json","after/report.json","projective-stress.json","advice-latency.json"))
    assert before["manifest_sha256"] == after["manifest_sha256"]
    original = {r["name"]:r for r in before["rows"]}
    regressions = [r["name"] for r in after["rows"] if original[r["name"]]["correct"] and not r["correct"]]
    suite = ET.parse(OUT/"tests.xml").getroot().find("testsuite")
    metrics = ("positive_regions","correct_positive_regions","expected_groups","correct_groups","wrong_confirmations","negative_regions","safe_negatives")
    report = {"version":"draft-23","before":{k:before[k] for k in metrics},"after":{k:after[k] for k in metrics},
              "manifest_sha256":after["manifest_sha256"],"regressions":regressions,
              "projective_geometry":{k:v for k,v in stress.items() if k!="rows"},"stored_evidence":read("stored-evidence.json"),
              "tests":dict(suite.attrib),"latency":{k:latency[k] for k in ("run_0","pipeline","pipeline_scope","comparison")},
              "unresolved":[r for r in after["rows"] if r["expected_groups"] and not r["correct"]],
              "scope":"Targeted debugging/regression, not independent population accuracy. Synthetic geometry does not measure image recognition. Known failures retained; no universal success claim.",
              "source_sha256":{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                   ("app/vision/meld_geometry.py","app/vision/kan.py","app/vision/recognizer.py","tests/test_meld_corpus.py",
                    "configs/window-steam-1366-v2.json","configs/window-steam-1366-pjura.json")}}
    checked = OUT/"final-annotation-tests.xml"
    if checked.exists():
        report["final_annotation_tests"] = dict(ET.parse(checked).getroot().find("testsuite").attrib)
    write_json(OUT/"report.json",report)
    rows = []
    words = {"chi":"吃","pon":"碰","daiminkan":"明杠","ankan":"暗杠","kakan":"加杠"}
    for row in after["rows"]:
        name = escape(row["name"])
        verdict = "错误确认" if row["wrong_confirmation"] else "已完整确认" if row["correct"] else "反例未误认" if row["safe_negative"] else "仍未完整确认"
        expected = "；".join(words[g["type"]]+" "+" ".join(g["labels"]) for g in row["expected_groups"]) or "没有副露"
        rows.append(f'<tr data-unresolved="{str(bool(row["expected_groups"] and not row["correct"])).lower()}"><td><a href="after/{name}.json">{name}</a><br>{escape(row["seat"])}</td><td><a href="after/{name}-original-roi.png"><img loading="lazy" src="after/{name}-original-roi.png" alt="原生截图局部"></a></td><td>{escape(expected)}</td><td>{original[row["name"]]["groups_correct"]} → {row["groups_correct"]}<br>{verdict}</td></tr>')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>副露透视识别：数据与修复</title>
<style>body{{font:16px/1.7 system-ui;margin:32px;color:#182235;max-width:1400px}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border:1px solid #cad3df;padding:12px;vertical-align:top}}img{{max-width:360px;max-height:200px;background:#182235}}a{{color:#185abc}}button{{padding:8px 14px;margin:10px 8px 20px 0}}.note{{background:#eef3f8;padding:16px;border-radius:8px}}</style>
<h1>副露透视识别：数据与修复（draft-23）</h1>
<p>47 个真实局部区域，32 张去重原图，21 个来源家族；46 组副露及 10 个牌背/空桌面反例。覆盖四个座位，包含赤牌、白板、相邻多组、明杠、暗杠、加杠。<a href="../../data/meld-orientation/README.md">数据选择与标注说明</a> · <a href="research.json">联网来源</a></p>
<table><tr><th>真实截图固定集</th><th>修改前</th><th>修改后</th></tr><tr><td>正确确认的副露组</td><td>{before['correct_groups']} / {before['expected_groups']}</td><td>{after['correct_groups']} / {after['expected_groups']}</td></tr><tr><td>完整通过的副露区域</td><td>{before['correct_positive_regions']} / {before['positive_regions']}</td><td>{after['correct_positive_regions']} / {after['positive_regions']}</td></tr><tr><td>错误确认</td><td>{before['wrong_confirmations']}</td><td>{after['wrong_confirmations']}</td></tr><tr><td>反例未误认</td><td>{before['safe_negatives']} / 10</td><td>{after['safe_negatives']} / 10</td></tr></table>
<p>受控投影与角点顺序检查：{stress['before_correct']} → {stress['after_correct']} / {stress['cases']}；几何处理本次中位数 {stress['geometry_median_ms']:.3f} 毫秒。它们是合成几何变体，不是额外真实图片或网络准确率。</p>
<p class="note">仍有 {after['expected_groups']-after['correct_groups']} 组未确认，涉及低清字形、辅助文字、特殊牌背以及额外前景。全部保留在下表，未降低主网络阈值强行确认。这是定向回归集，不能推成所有对局的成功率。</p>
<p>完整回归 {suite.attrib['tests']} 项，失败 {suite.attrib['failures']}、错误 {suite.attrib['errors']}、跳过 {suite.attrib['skipped']}。<a href="tests.xml">测试记录</a> · <a href="report.json">汇总和源码校验值</a> · <a href="advice-latency.json">完整时延测量</a></p>
<button onclick="document.querySelectorAll('tbody tr').forEach(r=>r.hidden=false)">全部案例</button><button onclick="document.querySelectorAll('tbody tr').forEach(r=>r.hidden=r.dataset.unresolved!=='true')">只看待解决案例</button>
<table><thead><tr><th>案例</th><th>原始图片局部</th><th>人工核对预期</th><th>正确组数与状态</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>'''
    (OUT/"index.html").write_text(html,encoding="utf-8")
    text = f'''# 副露透视识别：数据驱动修复（draft-23）

已联网检索并选取完整雀魂截图，建立 47 个真实局部案例：32 张去重原图、21 个来源家族、46 组副露和 10 个反例，覆盖四个座位。单牌来源为 [pjura](https://huggingface.co/datasets/pjura/mahjong_souls_tiles) 与 [cvmaj](https://github.com/xdedss/cvmaj)，完整场景来自本地已留存的公开截图及用户帧；本次没有重新训练主网络。细节见 [数据集说明](../../data/meld-orientation/README.md)及 [联网检索记录](research.json)。

修复：三张副露和杠牌统一角点的画面坐标顺序；旧模板轮廓缺失时使用现有网络几何，完整来源矛盾时不确认；白板不依赖字形角度；行几何允许透视倾斜并保留直线残差和相邻牌体约束。相邻多组副露可以由强候选确定唯一边界，但牌种确认沿用原规则，不能因此把两个未知牌都补成已确认。

真实固定集正确组数 **{before['correct_groups']}→{after['correct_groups']} / {after['expected_groups']}**，完整区域 **{before['correct_positive_regions']}→{after['correct_positive_regions']} / {after['positive_regions']}**，错误确认 **{after['wrong_confirmations']}**；10 个反例均未误认。受控几何变体 **{stress['before_correct']}→{stress['after_correct']} / {stress['cases']}**，不是独立真实图片或主网络准确率。此前 69 组已确认副露的存档证据无变化。

完整回归 **{suite.attrib['tests']} 项通过**，失败/错误/跳过均为 0。Tk 测试在本机显式指定现有 Tcl/Tk 目录，并在加载大量 GPU 会话前初始化；未修改用户的 Python 安装。建议工作流回放约 2.43–2.74 秒，另一个未确认响应约 1.87 秒，含 0.3 秒等待和界面轮询预留，不含真实窗口采集及首次预热。九张首次处理本次中位数约 1.54 秒，机器负载有波动；原九张图的局面、动作和 Q 分数保持一致。

**尚有 {after['expected_groups']-after['correct_groups']} 组未确认**：低分辨率字形、辅助文字牌面、特殊红/绿牌背或额外前景仍会阻止完整建议。未把未知算成功，也不声称所有场景已彻底解决。完整成功、失败及原像素见 [可筛选对照](index.html)。

[逐图修改前](before/report.json)、[逐图修改后](after/report.json)、[受控投影](projective-stress.json)、[存档证据重放](stored-evidence.json)、[时延](advice-latency.json)、[测试](tests.xml)、[汇总与校验值](report.json)。重启面板使用 draft-23。
'''
    (OUT/"README.md").write_text(text,encoding="utf-8")
    print("Report:",report["after"],"tests",suite.attrib["tests"],"regressions",regressions)


if __name__=="__main__":
    main()
