"""Replay the original riichi-overlap failure against frozen/current code."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import cv2

from app.advice import ScreenshotAdvice
from app.capture.frames import load_image,save_image
from app.engine.runtime import OnnxRuntime
from app.engine.snapshot import SnapshotBackend
from app.vision.config import load_config,write_json
from app.vision.recognizer import StaticRecognizer,overlay
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/fixtures/river-left-late'
OUT=ROOT/'reports/river-left-late-fix'


def baseline():
    import json
    report=ROOT/'reports/kan-all-seats-fix'
    hashes=json.loads((report/'report.json').read_text(encoding='utf-8'))['code_sha256']
    modules={}
    for name in ('detector','recognizer'):
        relative='app/vision/'+name+'.py';path=report/'code'/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative]
        spec=importlib.util.spec_from_file_location('app.vision._river_baseline_'+name,path)
        module=importlib.util.module_from_spec(spec)
        if name=='recognizer':
            current=sys.modules['app.vision.detector'];sys.modules['app.vision.detector']=modules['detector']
            try:spec.loader.exec_module(module)
            finally:sys.modules['app.vision.detector']=current
        else:spec.loader.exec_module(module)
        modules[name]=module
    config_path=report/'code/configs/window-steam-1366-v2.json'
    assert hashlib.sha256(config_path.read_bytes()).hexdigest()==hashes['configs/window-steam-1366-v2.json']
    return modules['recognizer'].StaticRecognizer,load_config(config_path)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['before','after'],default='after')
    args=parser.parse_args()
    kind,config=baseline() if args.stage=='before' else (StaticRecognizer,load_config(ROOT/'configs/window-steam-1366-v2.json'))
    recognizer=kind(config,TemplateBank(config['templates'],config.get('classifier_manifest')),TextReader())
    advice=ScreenshotAdvice(config,recognizer=recognizer,backend=SnapshotBackend(OnnxRuntime()))
    pixels=load_image(DATA/'client-1920.png');out=OUT/args.stage;out.mkdir(parents=True,exist_ok=True);rows=[]
    for size in [(1920,1080),(1366,768)]:
        frame=pixels if size[0]==1920 else cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
        advice.clear();recognizer.river_memory.clear();result=advice.process(frame)
        stem='client-'+str(size[0]);write_json(out/(stem+'-advice.json'),result)
        save_image(out/(stem+'-overlay.png'),overlay(frame,result['observation']))
        river=result['observation']['regions']['river_left']
        row={'size':list(size),'status':result['status'],'reason':result['reason'],
             'corrections':result['corrections'],'context':result['context'],
             'river_items':len(river['items']),'river_complete':river['complete'],
             'labels':[i['label'] for i in river['items']],
             'unknown':[{k:i.get(k) for k in ('slot','label','best_label','score','reason')} for i in river['items'] if i['label']=='?'],
             'neighbour_crops':river.get('neighbour_crops',[]),
             'native_image_sha256':result['observation'].get('native_image_sha256'),
             'image_sha256':result['observation']['image_sha256'],
             'recommendation':result['decision']['recommendation']['event'] if result['decision'] else None,
             'riichi':result['snapshot']['riichi'] if result['snapshot'] else None}
        rows.append(row);print(size,row['status'],row['reason'] or row['recommendation'],flush=True)
    write_json(out/'summary.json',{'config_version':config['version'],'rows':rows})


if __name__=='__main__':main()
