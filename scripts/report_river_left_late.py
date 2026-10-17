"""Summarize the riichi-body repair using real screenshots and old regressions."""
from html import escape
import hashlib
import json
from pathlib import Path
import re
import shutil

from app.capture.frames import load_image,save_image
from app.vision.config import write_json
from scripts.compare_public_vision import read,reviewed,overlap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/river-left-late-fix'


def main():
    before=read(OUT/'before/summary.json');after=read(OUT/'after/summary.json')
    for old,new in zip(before['rows'],after['rows']):
        assert old['image_sha256']==new['image_sha256']
        assert old['status']=='needs_confirmation' and old['river_items']==17
        assert new['status']=='ready' and new['river_items']==16 and new['river_complete']
        assert new['corrections']=={} and new['context']=={} and new['riichi']==[False,False,False,True]
    public=OUT/'public-scenes'
    manual=reviewed(public,read(ROOT/'data/public-mahjong/manual-regressions.json'))
    sideways=reviewed(public,read(ROOT/'data/public-mahjong/sideways-regressions.json'))
    assert manual['counts']=={'accepted_correct':32,'accepted_wrong':0,'unknown':3,'missing':0,'unprocessed':0,'extra_accepted':0}
    assert sideways['counts']['accepted_correct']==2 and sideways['counts']['accepted_wrong']==0
    scenes=read(public/'scenes-summary.json');mapping={r['path']:r for r in scenes['rows'] if 'output' in r}
    negatives=[]
    for case in read(ROOT/'data/public-mahjong/sideways-regressions.json')['non_tile_bodies']:
        observation=read(ROOT/mapping[case['path'].removeprefix('data/public-mahjong/')]['output'])
        labels=[i['label'] for i in observation['regions'][case['region']]['items'] if overlap(i['bbox'],case['bbox'])>=.5]
        assert labels and all(t=='?' for t in labels)
        negatives.append({'path':case['path'],'labels':labels})
    tests=(OUT/'tests.txt').read_text(encoding='utf-8-sig');matched=re.search(r'(\d+) passed',tests)
    assert matched and not re.search(r'\d+ failed',tests)
    code={}
    for name in ['app/vision/detector.py','app/vision/recognizer.py','configs/window-steam-1366-v2.json',
                 'configs/window-steam-1366-pjura.json']:
        source=ROOT/name;code[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        target=OUT/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for stage in ['before','after']:
        observation=read(OUT/stage/'client-1920-advice.json')['observation']
        x,y,w,h=observation['regions']['river_left']['bbox']
        image=load_image(OUT/stage/'client-1920-overlay.png')
        save_image(OUT/(stage+'-river.png'),image[max(0,y-12):y+h,x:x+w])
    report={'date':'2026-10-15','config_version':after['config_version'],'tests_passed':int(matched[1]),
            'before':before['rows'],'after':after['rows'],'manual_public':manual,'sideways_public':sideways,
            'back_edge_negatives':negatives,
            'public_scenes':{k:v for k,v in scenes.items() if k!='rows'},'code_sha256':code,
            'model_sha256':read(ROOT/'models/pjura-vision/manifest.json')['model_sha256'],
            'network_confidence':.9,'network_margin':.04,
            'scope':'One native user capture, plus its derived 1366x768 resize. Public scenes without full ground truth are layout diagnostics, not an accuracy benchmark.'}
    write_json(OUT/'report.json',report)
    table=[]
    for old,new in zip(before['rows'],after['rows']):
        size=new['size'];table.append(f'<tr><td>{size[0]}×{size[1]}</td><td>{old["river_items"]} 张，待核对</td>'
            f'<td>{new["river_items"]} 张，正常建议</td><td><a href="after/client-{size[0]}-advice.json">识别及建议</a></td></tr>')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>上家立直牌河误拆与邻牌裁切修复</title><style>body{{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;color:#23334a;background:#f3f6fa;max-width:1000px;margin:32px auto;padding:0 24px}}section{{background:white;border-radius:12px;padding:24px;margin:20px 0}}h1{{font-size:28px}}h2{{font-size:22px}}.pair{{display:flex;gap:24px;flex-wrap:wrap}}img{{max-width:100%;width:380px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#1762a5}}</style>
<h1>上家立直牌河识别修复</h1><p>2026-10-15 · {escape(after['config_version'])} · 完整测试 {matched[1]} 项通过</p>
<section><h2>为何卡在 river_left</h2><p>横放的立直东牌包含较厚的灰色侧面，检测框为 51×44，旧规则将它当成两张直立牌切开，多出一张未知牌。前一张七筒的框还包含了部分东牌，网络分数 0.893，低于原有 0.90 阈值。</p>
<p>现在侧边牌河拆列时要求直立牌达到接近整排高度，保留厚侧面的横牌。仅当同排相邻格子有明确横牌形状、重叠较小且位置符合网格时，裁去前一张框里的邻牌部分，原始框和裁切依据仍保留。七筒重新读取的分数为 0.963，模型及阈值均未改变。</p>
<div class="pair"><div><p>修改前：17 个框，末张东牌被拆开</p><img src="before-river.png" alt="修改前上家牌河"></div><div><p>修改后：16 张，横放东牌与七筒均保留</p><img src="after-river.png" alt="修改后上家牌河"></div></div></section>
<section><h2>实际截图完整流程</h2><table><tr><th>输入尺寸</th><th>修改前</th><th>修改后</th><th>证据</th></tr>{''.join(table)}</table>
<p>两种尺寸均无人工修正返回合法建议“打出南”；后端保留上家的立直状态。1366×768 是该原图的缩小派生测试，不是第二张独立采集的截图。</p>
<p><a href="../../data/fixtures/river-left-late/source.json">原图哈希及客户区裁切</a> · <a href="../../data/fixtures/river-left-late/labels.json">上家 16 张牌的局部真值</a></p></section>
<section><h2>回归和边界</h2><p>完整 {matched[1]} 项测试通过，包括四家杠牌与提前操作、手动截图的已有测试。公开样本中 35 张核对牌面仍为 32 正确、3 未知、0 错误、0 漏牌；横放四索、八索均正确，三个绿色牌背白边反例继续未知。</p>
<p>另对 {scenes['samples']} 张去重公开牌桌回放；其中 {scenes['input_errors']} 张因截图比例等输入条件未处理。无全图真值的样本不统计整局准确率。大面积遮挡、位置偏离网格、邻牌方向不明时不执行这项裁切，保留核对。</p>
<p><a href="tests.txt">完整测试记录</a> · <a href="report.json">统计和代码哈希</a> · <a href="public-scenes/scenes-summary.json">公开牌桌诊断</a></p></section></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'tests_passed':int(matched[1]),'manual':manual['counts'],'public_scenes':scenes['samples'],
                      'input_errors':scenes['input_errors'],'user_statuses':[r['status'] for r in after['rows']]},ensure_ascii=False))


if __name__=='__main__':main()
