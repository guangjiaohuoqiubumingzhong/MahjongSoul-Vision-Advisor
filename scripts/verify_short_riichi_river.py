"""Replay the short side-river riichi failure and its saved continuation."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2

from app.advice import ScreenshotAdvice, format_advice
from app.capture.frames import load_image
from app.vision.config import load_config, write_json


ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'data/fixtures/short-riichi-river'
OUT=ROOT/'reports/short-riichi-river'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['before','after'],required=True)
    args=parser.parse_args()
    config=load_config(ROOT/'configs/window-steam-1366-v2.json')
    service=ScreenshotAdvice(config);service.prepare()
    manifest=json.loads((FIXTURES/'manifest.json').read_text(encoding='utf-8'))
    report={'stage':args.stage,'scope':manifest['scope'],'cases':[],
            'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ['app/vision/recognizer.py','app/vision/river_polygons.py','app/vision/detector.py',
                                    'configs/window-steam-1366-v2.json']}}
    for case in manifest['cases']:
        path=ROOT/case['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
        original=load_image(path)
        for width,height in ((1920,1080),(1366,768)):
            service.clear();service.recognizer.river_memory.clear()
            for bank in [service.recognizer.bank,service.recognizer.river_polygons.fixed]:
                for name in ('_cache','_classification_cache'):
                    getattr(bank,name,{}).clear()
            service.recognizer.text_reader._read_cache.clear()
            pixels=original if width==1920 else cv2.resize(original,(width,height),interpolation=cv2.INTER_AREA)
            result=service.process(pixels,{'image_path':str(path),'capture_quality':'window_content'})
            river=result['observation']['regions']['river_right']
            labels=[i['label'] for i in river['items']]
            slots=[i['slot'] for i in river['items']]
            horizontal=[i['slot'] for i in river['items'] if i.get('horizontal') is True]
            truth={'labels':labels==case['labels'],
                   'slots':slots==[list(divmod(i,6)) for i in range(len(case['labels']))],
                   'riichi':horizontal==case['horizontal_slots']}
            row={'id':case['id'],'size':[width,height],'status':result['status'],'reason':result.get('reason'),
                 'message':format_advice(result),'ms':result['total_ms'],'river_complete':river['complete'],
                 'labels':labels,'slots':slots,'horizontal_slots':horizontal,'truth':truth,
                 'riichi':(result.get('snapshot') or {}).get('riichi'),'candidates':[
                     {k:i.get(k) for k in ('label','score','canonical_bbox','slot','horizontal','orientation_source','fixed_polygon_fallback','reason')}
                     for i in river['items']]}
            report['cases'].append(row)
            write_json(OUT/args.stage/f"{case['id']}-{width}.json",result)
            print(json.dumps({k:row[k] for k in ('id','size','status','truth','horizontal_slots','ms')},ensure_ascii=False),flush=True)
    write_json(OUT/(args.stage+'.json'),report)
    if args.stage=='after':
        assert all(row['status']=='ready' and row['river_complete'] and all(row['truth'].values()) for row in report['cases'])


if __name__=='__main__':main()
