"""Summarize real screenshot repairs and the existing public regressions."""
from html import escape
import hashlib
import json
from pathlib import Path
import re

from app.capture.frames import load_image, save_image
from app.vision.config import write_json
from scripts.compare_public_vision import read, reviewed, overlap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/kan-recognition-fix'


def main():
    run = read(OUT / 'after/summary.json')
    for name,expected in run['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    public = OUT / 'public-scenes'
    provenance = read(public / 'run.json')
    for name,expected in provenance['code'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    tests = (OUT / 'tests.txt').read_text(encoding='utf-8-sig')
    match = re.search(r'(\d+) passed',tests)
    assert match and not re.search(r'\d+ failed',tests)
    regression = reviewed(public,read(ROOT / 'data/public-mahjong/manual-regressions.json'))
    sideways = reviewed(public,read(ROOT / 'data/public-mahjong/sideways-regressions.json'))
    assert regression['counts'] == {'accepted_correct':32,'accepted_wrong':0,'unknown':3,
                                    'missing':0,'unprocessed':0,'extra_accepted':0}
    assert sideways['counts']['accepted_correct'] == 2 and sideways['counts']['accepted_wrong'] == 0
    scenes = read(public / 'scenes-summary.json')
    mapped = {r['path']:r for r in scenes['rows'] if 'output' in r}
    negatives = []
    for case in read(ROOT / 'data/public-mahjong/sideways-regressions.json')['non_tile_bodies']:
        row = mapped[case['path'].removeprefix('data/public-mahjong/')]
        observation = read(ROOT / row['output'])
        labels = [i['label'] for i in observation['regions'][case['region']]['items']
                  if overlap(i['bbox'],case['bbox']) >= .5]
        assert labels and all(t == '?' for t in labels)
        negatives.append({'path':case['path'],'labels':labels})
    before = {name:read(OUT/'before'/f'{name}.json') for name in ('top-3p-ankan','top-C-pon-arrow')}
    after = {name:read(OUT/'after'/f'{name}.json') for name in before}
    for name in before:
        assert before[name]['native_image_sha256'] == after[name]['native_image_sha256']
    for stage in ('before','after'):
        image = load_image(OUT / stage / 'top-3p-ankan-overlay.png')
        save_image(OUT / f'{stage}-ankan.png',image[:100,250:440])
    report = {'date':'2026-10-13','config_version':run['config_version'],'tests_passed':int(match[1]),
              'actual_screenshots':run['rows'],'public_scene_inputs':scenes['samples'],
              'public_scene_wrong_aspect_inputs':scenes['input_errors'],'manual_regressions':regression,
              'sideways_regressions':sideways,'back_edge_negative_cases':negatives,
              'limitations':['open_and_added_kan_still_require_independent_type_and_source_geometry',
                             'downsampled_second_scene_has_other_unmarked_low_confidence_river_tiles',
                             'fixed_four_player_one_red_five_per_suit_rules'],
              'code_sha256':run['code_sha256']}
    write_json(OUT / 'report.json',report)
    write_json(public / 'reviewed.json',{'manual':regression,'sideways':sideways,'negatives':negatives})
    rows = []
    for row in run['rows']:
        name,size = row['name'],row['size']
        stem = name if size[0] == 1920 else name+'-1366'
        title = '三筒暗杠' if name == 'top-3p-ankan' else '箭头遮挡一筒'
        event = row.get('recommendation') or {}
        outcome = ('打出北' if event.get('pai') == 'N' else '跳过' if event.get('type') == 'none'
                   else row['reason'] or row['status'])
        rows.append(f'<tr><td>{title}</td><td>{size[0]}×{size[1]}</td><td>{escape(outcome)}</td>'
                    f'<td><a href="after/{stem}-advice.json">完整建议记录</a> · <a href="after/{stem}-overlay.png">识别图</a></td></tr>')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>暗杠与箭头遮挡修复</title><style>
body{{font:16px/1.75 system-ui,sans-serif;max-width:1080px;margin:36px auto;padding:0 24px;color:#23313a;background:#f5f7f8}}
section{{background:white;padding:22px 28px;margin:20px 0;border-radius:12px}}h1{{font-size:30px}}h2{{font-size:22px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}
img{{max-width:100%;image-rendering:auto}}.pair{{display:flex;gap:24px;flex-wrap:wrap}}.pair img{{width:380px}}
a{{color:#006b81}}code{{background:#eef2f4;padding:2px 5px}}</style>
<h1>暗杠与箭头遮挡修复</h1><p>2026-10-13 · {escape(run['config_version'])} · 原始截图与识别证据保留</p>
<section><h2>三筒暗杠为什么未识别完整</h2>
<p>两张三筒都读对了；蓝纹牌背的旧模板不适合对家的透视姿态，两端被当成背景，副露只剩两张三筒。截图转局面也只自动接收吃、碰，暗杠无法进入后端。</p>
<p>现在用完整牌体内的蓝纹证据保留两端牌背，结合“牌背—同牌—同牌—牌背”的排列、尺寸、间隔和同排位置确认暗杠。原始识别保留 back，仅副露组推导四张 consumed，并参与实际数量检查。</p>
<div class="pair"><div><p>修改前：只剩两张三筒</p><img src="before-ankan.png" alt="修改前三筒暗杠识别"></div>
<div><p>修改后：两端牌背与四张暗杠结构</p><img src="after-ankan.png" alt="修改后三筒暗杠识别"></div></div>
<p>有一张明牌略低于主网络阈值时，另一张必须确定、弱候选必须指向同牌且满足既有副露补判分数与差距要求。两张未知、牌背缺失、被裁切、不同牌面或不符合排列仍不确认。五的暗杠按当前每色一枚赤五的固定规则保留赤牌。</p></section>
<section><h2>第二张图的问题</h2>
<p>面板停在 river_left，右侧是白板碰和红中碰。提示箭头盖住了一筒，主网络置信度 0.684；原字形模板提案与三个既有辅助分类器都认成一筒。</p>
<p>仅当检测到箭头、至少半张牌未被遮住、这些字形证据与原网络清晰候选一致时恢复。模板相似度与原网络读数分别保留；空白白板、五的颜色歧义、无箭头或证据冲突不会补判。计时器装饰误入自己副露区的宽碎块，也根据缺少牌面轮廓和颜色证据排除。</p>
<img src="../../data/fixtures/kan-recognition/arrow-detail.png" width="240" alt="提示箭头盖住一筒原图"></section>
<section><h2>实际截图完整流程</h2><table><tr><th>截图</th><th>输入尺寸</th><th>结果</th><th>证据</th></tr>{''.join(rows)}</table>
<p>均从像素读取，没有人工修正或补录局况，缓存清空后独立处理。1366×768 是原始客户区缩小的测试图，第二张仍有其他小尺寸牌河牌低于主网络阈值，保留核对；不能宣称所有低分辨率牌面已经恢复。</p></section>
<section><h2>回归结果与边界</h2><p>完整测试 {match[1]} 项通过。44 张公开图回放：22 张布局尺寸可处理，22 张不符合当前 16:9 校准比例。已有 35 张人工核对牌面仍为 32 张正确、3 张未知、0 张错误、0 张漏牌；横放四索、八索两例保持正确，三个牌背白边反例仍未知。</p>
<p>主网络及横放字形模型没有重新训练，全局主网络阈值仍为 0.90。明杠、加杠缺少独立结构与来源证据时仍需核对；本次自动恢复的是图中两端蓝纹牌背的暗杠结构。</p>
<p><a href="report.json">完整统计与代码哈希</a> · <a href="tests.txt">测试记录</a> · <a href="../../data/fixtures/kan-recognition/sources.json">原图哈希与裁切记录</a></p></section></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'tests_passed':int(match[1]),'screenshots':[r['status'] for r in run['rows']],
                      'public':regression['counts'],'negative_back_edges':len(negatives)},ensure_ascii=False))


if __name__ == '__main__':
    main()
