"""Collect public kan-related screenshots, retaining sources and original bytes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from html import unescape
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

from scripts.collect_public_mahjong import ScreenshotCards,inspect_image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/public-mahjong'
OUT=BASE/'kan-scenes'
PAGES={
    'note-open':'https://note.com/okuohouko/n/n66c05a0ed410',
    'note-added':'https://note.com/shinux217/n/n42348a994488',
    'note-top-closed':'https://note.com/cos13/n/n7a417fe34fd2',
    'note-left-closed':'https://note.com/amutan_mj/n/n058423d1de4c',
    'note-right-kan':'https://note.com/raccoondogs/n/n987c367f73b1',
    'note-right-closed':'https://note.com/fine_koala1007/n/n1a02f9d306b2',
    'reddit-four-quads':'https://www.reddit.com/r/mahjongsoul/comments/1fwm36g/ever_seen_someone_kan_4_times_in_a_row_into_four/',
    'reddit-fourth-kan':'https://www.reddit.com/r/mahjongsoul/comments/17ebm8l/tried_to_draw_the_game_with_a_fourth_kan_and/',
}
ALLOWED={'note.com','assets.st-note.com','steamcommunity.com','images.steamusercontent.com',
         'steamuserimages-a.akamaihd.net','www.reddit.com','i.redd.it','preview.redd.it'}

def read_public(url,limit):
    if urllib.parse.urlsplit(url).hostname not in ALLOWED:
        raise ValueError('Unapproved source host')
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (Majan public kan screenshot research)'})
    with urllib.request.urlopen(request,timeout=15) as response:
        data=response.read(limit+1)
    if len(data)>limit:raise ValueError('File size limit exceeded')
    return data

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--steam-pages',type=int,default=3)
    args=parser.parse_args()
    assert 0<=args.steam_pages<=8
    OUT.joinpath('source').mkdir(parents=True,exist_ok=True)
    OUT.joinpath('images').mkdir(exist_ok=True)
    candidates=[];errors=[]
    pages={**PAGES,**{f'steam-top-{i}':f'https://steamcommunity.com/app/2739990/screenshots/?browsefilter=toprated&p={i}'
                      for i in range(1,args.steam_pages+1)}}
    for name,url in pages.items():
        target=OUT/'source'/f'{name}.html'
        try:
            if not target.exists():target.write_bytes(read_public(url,4000000))
            html=unescape(target.read_text(encoding='utf-8'))
            if name.startswith('steam'):
                cards=ScreenshotCards();cards.feed(html)
                candidates.extend({**c,'source':'steam-kan-research','gallery_page':url} for c in cards.cards)
            else:
                # Only article content/CDN screenshots; avatars and badges are
                # removed by the image dimension check after download.
                html=html.replace('\\/','/')
                if name.startswith('note'):
                    urls=re.findall(r'https://assets\.st-note\.com/[^"<>\s\\]+',html)
                    urls=[u.split('?')[0] for u in urls if re.search(r'\.(?:png|jpe?g|webp)(?:\?|$)',u,re.I)]
                else:
                    urls=re.findall(r'https://(?:i|preview)\.redd\.it/[^"<>\s\\]+',html)
                    urls=[u.split('?')[0] for u in urls if re.search(r'\.(?:png|jpe?g)(?:\?|$)',u,re.I)
                          and 'avatar' not in u]
                candidates.extend({'id':name+'-'+str(i),'image_url':u,'source_page':url,'source':'kan-'+name.split('-')[0]}
                                  for i,u in enumerate(dict.fromkeys(urls)))
            print(name,'sources discovered',len(candidates),flush=True)
        except Exception as exc:
            errors.append({'url':url,'error':str(exc)})
            print(name,'unavailable',str(exc),flush=True)
    # A public image link observed on the post when a page fetch is blocked.
    candidates.append({'id':'reddit-four-quads-direct','source':'kan-reddit','source_page':PAGES['reddit-four-quads'],
                       'image_url':'https://i.redd.it/r609jfauiwsd1.png'})
    candidates={c['image_url']:c for c in candidates}
    def download(c):
        suffix=Path(urllib.parse.urlsplit(c['image_url']).path).suffix.lower()
        if suffix not in {'.jpg','.jpeg','.png','.webp'}:suffix='.jpg'
        path=OUT/'images'/(c['id']+suffix)
        if not path.exists():path.write_bytes(read_public(c['image_url'],20*1024*1024))
        info=inspect_image(path)
        if info['width']<500 or info['height']<250:return None
        return {**c,**info,'path':path.relative_to(BASE).as_posix(),'kind':'screenshot_candidate',
                'upstream_label':None,'label_mjai':None,'scene':'unreviewed','scene_tags':['kan_search_candidate'],
                'ground_truth_reviewed':False,'license':'Public author/game image; rights retained; local debugging only',
                'downloaded_at':datetime.now(timezone.utc).isoformat()}
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(download,c):c for c in candidates.values()}
        for future in as_completed(jobs):
            try:
                row=future.result()
                if row:rows.append(row)
            except Exception as exc:errors.append({'url':jobs[future]['image_url'],'error':str(exc)})
    manifest=BASE/'manifest.jsonl'
    existing=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    paths={r['path'] for r in existing};hashes={r['sha256']:r['path'] for r in existing if not r.get('duplicate_of')}
    previous={r['path']:r for r in existing}
    added=0
    for row in sorted(rows,key=lambda r:r['path']):
        row['duplicate_of']=previous[row['path']].get('duplicate_of') if row['path'] in previous else hashes.get(row['sha256'])
        if row['path'] not in paths:
            existing.append(row);paths.add(row['path']);hashes.setdefault(row['sha256'],row['path']);added+=1
    manifest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in existing)+'\n',encoding='utf-8')
    (OUT/'collection.json').write_text(json.dumps({'records':rows,'errors':errors},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'saved_images':len(rows),'added_files':added,'unique_collected':sum(r['duplicate_of'] is None for r in rows),'errors':len(errors)},ensure_ascii=False))

if __name__=='__main__':main()
