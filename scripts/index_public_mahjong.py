"""Build a local, searchable image browser and verify downloaded originals."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/public-mahjong"
LABELS = {f"{n}{s}": f"{n}{mapped}" for n in range(1, 10) for s, mapped in (("b", "s"), ("n", "m"), ("p", "p"))}
LABELS.update(ew="E", sw="S", ww="W", nw="N", wd="P", gd="F", rd="C")
HONORS = dict(zip([f"{n}z" for n in range(1, 8)], "ESWNPFC"))


def main():
    records = [json.loads(line) for line in (BASE / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    scene_path = BASE / "scene-review.json"
    scenes = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.exists() else {}
    browse = BASE / "browse"
    browse.joinpath("thumbnails").mkdir(parents=True, exist_ok=True)
    screenshots, failures = [], []
    for record in records:
        path = BASE / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append({"path": record["path"], "error": "SHA-256 mismatch"})
            continue
        with Image.open(path) as image:
            image.verify()
        label = record.get("upstream_label")
        if record["source"] == "pjura":
            record["label_mjai"] = LABELS.get(label)
        elif record["source"] == "cvmaj" and label:
            record["label_mjai"] = HONORS.get(label, label if label in LABELS.values() else None)
            # Keep 5p- and x as upstream labels unless separately reviewed.
        if record["kind"] != "tile_crop":
            record["scene"] = scenes.get(record["path"], {}).get("scene", "unreviewed")
            record["scene_tags"] = scenes.get(record["path"], {}).get("tags", [])
            record["scene_reviewed"] = record["path"] in scenes
            thumb = browse / "thumbnails" / (record["sha256"][:20] + ".jpg")
            if not thumb.exists():
                with Image.open(path) as image:
                    ImageOps.contain(image.convert("RGB"), (480, 300)).save(thumb, quality=82)
            record["thumbnail"] = thumb.relative_to(BASE).as_posix()
            screenshots.append(record)
    if failures:
        raise ValueError(f"Corrupt or changed originals: {failures}")
    records.sort(key=lambda r: r["path"])
    (BASE / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    contact = browse / "contact-sheets"
    contact.mkdir(exist_ok=True)
    for start in range(0, len(screenshots), 40):
        sheet = Image.new("RGB", (1280, 164 * 8), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(screenshots[start:start + 40]):
            with Image.open(BASE / record["path"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (252, 142))
            x, y = i % 5 * 256, i // 5 * 164
            sheet.paste(thumb, (x + (256 - thumb.width) // 2, y))
            draw.text((x + 3, y + 144), f"{start+i+1:03d} {Path(record['path']).stem}", fill="#222222")
        sheet.save(contact / f"all-{start//40+1:02d}.jpg", quality=88)
    compact = [{k: r.get(k) for k in ("path", "source", "kind", "width", "height", "upstream_label", "label_mjai",
               "scene", "scene_tags", "source_page", "thumbnail", "duplicate_of")} for r in records]
    data = json.dumps(compact, ensure_ascii=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>雀魂公开截图与牌面样本</title><style>
body{margin:0;background:#f4f6fa;color:#202e40;font:16px/1.6 'Segoe UI','Microsoft YaHei',sans-serif}header,main{max-width:1500px;margin:auto;padding:24px}h1{font-size:27px;margin:0}.note{color:#627086}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}input,select,button{padding:10px 12px;border:1px solid #ccd3df;border-radius:7px;background:white;font:inherit}input{flex:1;min-width:200px}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}.card{background:white;border-radius:10px;overflow:hidden;border:1px solid #dfe5ed}.card img{width:100%;height:190px;object-fit:contain;background:#edf0f5}.info{padding:12px;font-size:13px;overflow-wrap:anywhere}.meta{color:#627086}.badge{display:inline-block;background:#e9f0ff;padding:2px 7px;border-radius:4px}a{color:#2054b5}#count{margin:16px 0}#more{display:block;margin:28px auto}.tiles .card img{height:160px}.tiles{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))!important}</style>
<header><h1>雀魂公开截图与牌面样本</h1><div class="note">原图可点开查看。场景分类只描述画面类型，牌面内容尚未逐张核对；公开来源和标签保留在清单中。</div>
<nav><select id="kind"><option value="screens">截图与场景</option><option value="tile_crop">带标签单牌</option><option value="all">全部</option></select><select id="source"><option value="all">全部来源</option><option value="steam-community">Steam 社区截图</option><option value="pjura">pjura 单牌数据</option><option value="cvmaj">cvmaj 样本</option><option value="kan-note">杠牌 · 作者复盘</option><option value="kan-reddit">杠牌 · Reddit 原图</option><option value="steam-kan-research">杠牌检索 · Steam</option></select><select id="scene"><option value="all">全部场景</option><option value="board">牌桌场景</option><option value="win_result">和牌 / 结算 / 特效</option><option value="hand_closeup">牌面局部</option><option value="lobby">大厅 / 角色 / 其他界面</option><option value="processed">处理结果 / 示意图</option><option value="unreviewed">待分类</option></select><input id="search" placeholder="搜索牌种、标签、文件名，如 7p、riichi、meld"><label><input id="dedup" type="checkbox" checked>隐藏重复</label></nav>
<div id="count"></div><a href="README.md">来源与使用说明</a> · <a href="manifest.jsonl">完整图片清单</a> · <a href="kan-scenes/index.html">杠牌专项样本与测试</a></header><main><div id="grid"></div><button id="more">继续显示</button></main>
<script>const DATA=__DATA__;let limit=80;const $=id=>document.getElementById(id);const names={board:'牌桌场景',win_result:'和牌 / 结算 / 特效',hand_closeup:'牌面局部',lobby:'大厅 / 角色 / 其他界面',processed:'处理结果 / 示意图',unreviewed:'待分类'};
function render(){const q=$('search').value.trim().toLowerCase();const items=DATA.filter(r=>($('kind').value==='all'||($('kind').value==='screens'?r.kind!=='tile_crop':r.kind==='tile_crop'))&&($('source').value==='all'||r.source===$('source').value)&&($('scene').value==='all'||r.scene===$('scene').value)&&(!$('dedup').checked||!r.duplicate_of)&&(!q||[r.path,r.upstream_label,r.label_mjai,...(r.scene_tags||[])].join(' ').toLowerCase().includes(q)));$('count').textContent=`共 ${items.length} 张匹配图片；原图总数 ${DATA.length} 张。`;$('grid').replaceChildren();$('grid').classList.toggle('tiles',$('kind').value==='tile_crop');for(const r of items.slice(0,limit)){const card=document.createElement('article');card.className='card';const a=document.createElement('a');a.href=encodeURI(r.path);a.target='_blank';a.rel='noopener';const img=document.createElement('img');img.loading='lazy';img.src=encodeURI(r.thumbnail||r.path);img.alt=r.path;a.append(img);card.append(a);const info=document.createElement('div');info.className='info';const badge=document.createElement('span');badge.className='badge';badge.textContent=r.kind==='tile_crop'?`标签 ${r.upstream_label}${r.label_mjai?' → '+r.label_mjai:''}`:(names[r.scene]||r.scene);info.append(badge);const path=document.createElement('div');path.textContent=r.path;info.append(path);const meta=document.createElement('div');meta.className='meta';meta.textContent=`${r.width} × ${r.height} · ${r.source} ${(r.scene_tags||[]).join(', ')}`;info.append(meta);const source=document.createElement('a');source.href=r.source_page;source.target='_blank';source.rel='noopener';source.textContent='查看原始来源';info.append(source);card.append(info);$('grid').append(card)}$('more').hidden=items.length<=limit}
for(const id of ['kind','source','scene','search','dedup'])$(id).addEventListener('input',()=>{limit=80;render()});$('more').onclick=()=>{limit+=80;render()};render();</script></html>""".replace("__DATA__", data)
    (BASE / "index.html").write_text(page, encoding="utf-8")
    board_paths = [r["path"] for r in records if r.get("scene") == "board" and not r["duplicate_of"]]
    (BASE / "board-images.txt").write_text("\n".join(board_paths) + ("\n" if board_paths else ""), encoding="utf-8")
    clear_paths = [r["path"] for r in records if r.get("scene") == "board" and
                   "clear_board" in r.get("scene_tags", []) and not r["duplicate_of"]]
    (BASE / "clear-board-images.txt").write_text("\n".join(clear_paths) + ("\n" if clear_paths else ""), encoding="utf-8")
    report = {"valid_images": len(records), "unique_images": len({r["sha256"] for r in records}),
              "screenshot_examples": len(screenshots), "unique_screenshot_examples": sum(not r["duplicate_of"] for r in screenshots),
              "by_source": dict(Counter(r["source"] for r in records)),
              "scene_counts": dict(Counter(r.get("scene", "tile_crop") for r in records)),
              "unique_board_images": len(board_paths), "clear_board_images": len(clear_paths),
              "sha256_verified": True, "decode_verified": True}
    (BASE / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
