"""Publish local kan evidence, reviewed pixels and explicit coverage limits."""
from html import escape
import hashlib
import json
from pathlib import Path
import re
import shutil

from app.vision.config import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/kan-all-seats-fix'
BASE=ROOT/'data/public-mahjong'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    before=read(OUT/'before/public-kan-summary.json');after=read(OUT/'after/public-kan-summary.json')
    for name,expected in after['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    user=read(OUT/'after/user-summary.json');old=read(OUT/'existing-reviewed.json')
    newscenes=read(OUT/'new-public-scenes/scenes-summary.json')
    assert after['correct']==11 and after['wrong_confirmation']==0 and after['unresolved']==2
    assert all(r['status']=='ready' and r['corrections']=={} and r['context']=={} for r in user['rows'])
    assert old['manual']['counts']['accepted_correct']==32 and old['manual']['counts']['accepted_wrong']==0
    assert old['sideways']['counts']['accepted_correct']==2
    assert all(r['labels']==['?'] for r in old['negative_back_edges'])
    tests=(OUT/'tests.txt').read_text(encoding='utf-8-sig');matched=re.search(r'(\d+) passed',tests)
    assert matched and not re.search(r'\d+ failed',tests)
    manifest=[json.loads(s) for s in (BASE/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    new=[r for r in manifest if r['path'].startswith('kan-scenes/')]
    original=[r for r in manifest if not r['path'].startswith('kan-scenes/')]
    addition=len({r['sha256'] for r in manifest})-len({r['sha256'] for r in original})
    code={}
    for name in ['app/vision/kan.py','app/vision/detector.py','app/vision/recognizer.py','app/vision/config.py',
                 'app/state/snapshot.py','configs/window-steam-1366-v2.json','configs/window-steam-1366-pjura.json']:
        path=ROOT/name;code[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        target=OUT/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    summary={'date':'2026-10-13','config_version':after['config_version'],'tests_passed':int(matched[1]),
             'user_screenshot':user,'downloaded_files':len(new),'additional_unique_images':addition,
             'dataset_validation':read(BASE/'validation.json'),
             'public_reviewed_component_before':{k:before[k] for k in ['cases','correct','wrong_confirmation','unresolved']},
             'public_reviewed_component_after':{k:after[k] for k in ['cases','correct','wrong_confirmation','unresolved']},
             'seat_coverage':{seat:sorted({r['expected']['type'] for r in after['rows'] if r['seat']==seat})
                              for seat in ['self','right','top','left']},
             'passed_kan_types_by_seat':{seat:sorted({r['expected']['type'] for r in after['rows']
                                          if r['seat']==seat and r['correct'] and r['expected']['type']!='pon'})
                                       for seat in ['self','right','top','left']},
             'public_corpus_full_layout_diagnostics':{k:newscenes[k] for k in ['samples','input_errors','incomplete_regions']},
             'previous_public_regressions':old,'code_sha256':code,
             'model_sha256':read(ROOT/'models/pjura-vision/manifest.json')['model_sha256'],
             'global_confidence':.9,'global_margin':.04,
             'limitations':['13 hand-selected, partially labelled component regressions are not a held-out overall accuracy estimate',
                            'public ROI calibrations preserve native aspect/pixels; they do not prove default window layouts or four-player backend compatibility',
                            'not all 12 seat/type combinations have real public examples; independent unit cases exercise every actor and called source position',
                            'darkened top-E result frame and custom-red-back top-5p closed kan remain unresolved',
                            'self-P late ROI includes an unlabelled clipped neighbour; group correctness does not imply entire ROI completeness']}
    write_json(OUT/'report.json',summary)
    seats={'self':'自己','right':'下家','top':'对家','left':'上家'}
    types={'kakan':'加杠','daiminkan':'大明杠','ankan':'暗杠','pon':'碰牌反例'}
    tiles={'P':'白板','C':'红中','9s':'九索','E':'东','6p':'六筒','1p':'一筒','5p':'五筒'}
    previous={r['name']:r for r in before['rows']}
    cards=[]
    for row in after['rows']:
        name=row['name'];expected=row['expected'];tile=tiles[expected['labels'][0]]
        title=seats[row['seat']]+' · '+tile+' '+types[expected['type']]
        oldrow=previous[name]
        outcome='已识别' if row['correct'] else '仍需核对'
        earlier='已识别' if oldrow['correct'] else '误认成碰' if oldrow['wrong_confirmation'] else '未确认'
        cards.append(f'<article><h3>{escape(title)}</h3><a href="after/{name}-overlay.png"><img src="after/{name}-roi.png" alt="{escape(title)}原图局部"></a>'
                     f'<p>{earlier} → <b>{outcome}</b></p><p class="small">{escape(row["scene_rules"])} · 原图局部标注 · 区域完整：{row["complete"]}</p>'
                     f'<a href="{escape(row["source_page"])}">作者原文</a> · <a href="after/{name}.json">识别证据</a></article>')
    wrc='https://static1.squarespace.com/static/634a7884c297a25f06589b79/t/6834d67360e19c1da6c0d12c/1748293243651/WRC%2BRules%2B2025.pdf'
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>四家杠牌识别与公开样本回归</title><style>body{{background:#f2f5fa;color:#22334b;font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif;max-width:1140px;margin:30px auto;padding:0 22px}}section,article{{background:white;border:1px solid #dae3ef;border-radius:12px;padding:22px;margin:18px 0}}h1{{font-size:29px}}h2{{font-size:23px}}h3{{font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}.grid article{{margin:0}}img{{max-width:100%;width:100%;object-fit:contain;max-height:260px;background:#eef2f6}}.small{{font-size:13px;color:#667589}}a{{color:#1765ab}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}</style>
<h1>四家杠牌识别与公开样本回归</h1><p>2026-10-13 · 配置 draft-19 · {matched[1]} 项完整测试通过</p>
<section><h2>这张发财加杠为什么没提示</h2><p>四张发财原本都识别正确，暂停来自副露类型判断：旧逻辑将四张同牌同时保留为大明杠、暗杠和加杠，没有判断第四张叠牌的位置。现在用相邻横牌、叠放位置以及剩余三张碰牌的排列确认加杠，并保留实际追加的那张牌。</p>
<p>用户原始 1920×1080 客户区及其 1366×768 缩小图均直接返回合法建议“打出南”，无人工修正。</p><a href="after/user-self-F-kakan-overlay.png"><img src="after/user-self-F-kakan-overlay.png" alt="发财加杠完整识别"></a><p><a href="after/user-self-F-kakan-advice.json">1920 建议记录</a> · <a href="after/user-self-F-kakan-1366-advice.json">1366 建议记录</a></p></section>
<section><h2>新增公开资料与本地数据</h2><p>新增 {len(new)} 个原始图片文件，去重后新增 {addition} 张。保留作者页面、图片地址、原尺寸、哈希及重复关系；包括四家杠牌、三人场、复盘、结算及其他检索结果。图库中的场景分类不是全部牌面的真值。当前没有用这些图重新训练网络。</p>
<p><a href="../../data/public-mahjong/kan-scenes/index.html">打开杠牌专项样本</a> · <a href="../../data/public-mahjong/index.html">完整本地图库</a> · <a href="../../data/public-mahjong/kan-scenes/regressions.json">人工核对标注</a></p>
<p class="small">原作者与游戏方版权保留，公开图片作为本地调试资料；没有统一开放再分发许可。</p></section>
<section><h2>修改与测试结果</h2><p>明杠按四张一排及唯一横牌确认来源；加杠按叠放的两张横牌与三张底牌确认。对面较薄的上层牌不再漏检；两张粘连的副露要求相邻牌宽度和两块完整牌面轮廓，才能分开。橙色暗杠牌背独立检测，不能吞并旁边的白色牌面；斜向排列也必须符合连续同排位置。</p>
<p>结构补判只采用网络和已接受模板一致的候选，原分数与未知读数保留；全局网络阈值仍为 0.90。赤五与普通五的追加层无法区分时保留核对。摆放规则参考 <a href="{wrc}">WRC 2025 §8.12</a>，仅核对副露摆法。</p>
<table><tr><th>13 个局部人工核对样本</th><th>修改前</th><th>修改后</th></tr><tr><td>正确</td><td>2</td><td>11</td></tr><tr><td>错误确认</td><td>1</td><td>0</td></tr><tr><td>未确认</td><td>10</td><td>2</td></tr></table>
<p>自己、下家、对家、上家均有真实杠样本通过；尚未收齐四家各三类杠的全部 12 种实图组合。每家明杠来源位置、加杠层、赤牌歧义另用几何单元测试覆盖，不能算作真实截图准确率。</p>
<p>既有 35 张核对牌面仍为 32 正确、3 未知、0 错误、0 漏牌；横放四索、八索及三个牌背白边反例没有回退。</p></section>
<section><h2>每个核对样本</h2><div class="grid">{''.join(cards)}</div></section>
<section><h2>测试边界和未解决项</h2><p>13 个样本采用人工标出的副露区域和原始像素，三人场、修罗场及不同比例图只作视觉局部测试。局部判断正确不意味着默认窗口布局或整局后端已支持该图；白板加杠后期局部还包含一块未标注的相邻牌边。两张暗色图（结算时的东加杠和特殊红背五筒暗杠）仍未确认，已保留原图和失败记录。</p>
<p>新增 {newscenes['samples']} 张去重牌桌图已走默认完整布局诊断，其中 {newscenes['input_errors']} 张因尺寸比例等输入条件未通过；这部分没有整局牌面真值，不报告总体准确率。原有 44 张牌桌也已回放。</p>
<p><a href="report.json">完整报告与代码哈希</a> · <a href="tests.txt">完整测试记录</a> · <a href="before/public-kan-summary.json">修改前局部结果</a> · <a href="after/public-kan-summary.json">修改后局部结果</a> · <a href="new-public-scenes/scenes-summary.json">新增牌桌诊断</a></p></section></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    # The dedicated data entry links to the complete report; all embedded crops
    # remain diagnostics and originals stay in the dataset untouched.
    (BASE/'kan-scenes/index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>杠牌专项样本</title><style>body{font:17px/1.8 system-ui;max-width:1000px;margin:40px auto;padding:20px}img{max-width:300px}article{display:inline-block;vertical-align:top;width:300px;margin:12px}</style><h1>杠牌专项样本</h1><p>原图、作者来源和核对标注已保留；只标注列出的副露区域。<a href="../../../reports/kan-all-seats-fix/report.html">完整对照报告</a> · <a href="../index.html">返回全部图库</a></p>'+''.join(cards).replace('href="after/','href="../../../reports/kan-all-seats-fix/after/').replace('src="after/','src="../../../reports/kan-all-seats-fix/after/'),encoding='utf-8')
    print({k:summary[k] for k in ('tests_passed','downloaded_files','additional_unique_images')})


if __name__=='__main__':main()
