"""Compare the selected optimization on deduplicated public crops and scenes."""
import argparse
from copy import copy,deepcopy
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from app.advice import ScreenshotAdvice,load_screenshot
from app.capture.frames import load_image
from app.vision.config import load_config,write_json
from app.vision.pjura import PjuraModel,file_sha256
from scripts.benchmark_pjura_candidates import session
from scripts.benchmark_static_vision import signature

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pjura-acceleration'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',default='fused-fp16')
    parser.add_argument('--guard',action='store_true')
    parser.add_argument('--installed',action='store_true')
    args=parser.parse_args()
    cfg=load_config(ROOT/'configs/window-steam-1366-v2.json')
    service=ScreenshotAdvice(cfg);service.prepare()
    bank=service.recognizer.bank;baseline=PjuraModel(ROOT/'models/pjura-vision/manifest.json')
    candidate_path=ROOT/'.tools/pjura-acceleration'/(args.candidate+'.onnx')
    candidate=bank.model if args.installed else copy(baseline)
    if not args.installed:
        candidate.session=session(candidate_path);candidate.lock=threading.RLock()
        candidate.manifest={**baseline.manifest,'model_sha256':file_sha256(candidate_path)}
    candidate._warmed=False;candidate.warmup()
    guard_count=[0]
    if args.guard and not args.installed:
        predict=candidate.predict_many
        def guarded(crops):
            predictions=predict(crops)
            indices=[i for i,p in enumerate(predictions)
                     if any(abs(p['score']-t)<=.005 for t in (.9,.5))
                     or any(abs(p['margin']-t)<=.005 for t in (.04,.3))]
            if indices:
                for i,precise in zip(indices,baseline.predict_many([crops[i] for i in indices])):
                    predictions[i]=precise
                guard_count[0]+=len(indices)
            return predictions
        candidate.predict_many=guarded
    records=[json.loads(line) for line in (ROOT/'data/public-mahjong/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    records=[r for r in records if r['kind']=='tile_crop' and r.get('label_mjai') and not r['duplicate_of']]
    metrics={name:{'samples':0,'top1_correct':0,'accepted':0,'accepted_wrong':0,'total_ms':0} for name in ('baseline','candidate')}
    changes=[];max_score_error=0.0
    for start in range(0,len(records),64):
        group=records[start:start+64];crops=[load_image(ROOT/'data/public-mahjong'/r['path']) for r in group]
        predictions={}
        for name,model in [('baseline',baseline),('candidate',candidate)]:
            started=time.perf_counter();predictions[name]=model.predict_many(crops)
            metric=metrics[name];metric['total_ms']+=(time.perf_counter()-started)*1000
            for record,prediction in zip(group,predictions[name]):
                accepted=prediction['score']>=.9 and prediction['margin']>=.04
                metric['samples']+=1;metric['top1_correct']+=prediction['label']==record['label_mjai']
                metric['accepted']+=accepted;metric['accepted_wrong']+=accepted and prediction['label']!=record['label_mjai']
        for record,old,new in zip(group,predictions['baseline'],predictions['candidate']):
            old_accepted=old['score']>=.9 and old['margin']>=.04
            new_accepted=new['score']>=.9 and new['margin']>=.04
            max_score_error=max(max_score_error,abs(old['score']-new['score']))
            if old['label']!=new['label'] or old_accepted!=new_accepted:
                changes.append({'path':record['path'],'expected':record['label_mjai'],'old':old,'new':new})
        if start%512==0:print('PUBLIC',min(start+64,len(records)),len(records),flush=True)
    paths=[ROOT/'data/fixtures'/name for name in (
        'river-left-late/client-1920.png','kan-recognition/top-3p-ankan-1920.png',
        'kan-recognition/top-C-pon-arrow-1920.png','kan-all-seats/user-self-F-kakan-1920.png',
        'no-draw-call-discard/chi-buttons-1920.png','no-draw-call-discard/ron-buttons-1920.png')]
    for action in ('0093','0094','0112','0113'):
        paths.append(next((ROOT/f'reports/screenshot-advice/run-20261015T083403-4/action-{action}').glob('[0-9]*.png')))
    scenes=[]
    for path in paths:
        original,metadata=load_screenshot(path,cfg)
        for size in ((1920,1080),(1366,768)):
            pixels=original if size[0]==1920 else cv2.resize(original,size,interpolation=cv2.INTER_AREA)
            outputs={}
            for name,model in [('baseline',baseline),('candidate',candidate)]:
                service.clear();service.recognizer.river_memory.clear()
                bank._cache.clear();bank._classification_cache.clear()
                service.recognizer.text_reader._read_cache.clear();bank.model=model
                started=time.perf_counter();result=service.process(pixels,metadata)
                outputs[name]={'ms':(time.perf_counter()-started)*1000,'status':result['status'],
                    'reason':result['reason'],'signature':json.loads(json.dumps(signature(result['observation']))),
                    'snapshot':result['snapshot'],'recommendation':(result.get('recommendation') or {}).get('event'),
                    'mask':(result.get('decision') or {}).get('action_mask'),
                    'observation_sha256':(result.get('decision') or {}).get('observation_sha256')}
                # Blank glyphs and unconfirmed network turns are diagnostic;
                # the detector's physical horizontal/slot evidence drives play.
                for region_name,region in result['observation']['regions'].items():
                    outputs[name]['signature'][region_name]['items']=[
                        [item['label'],item.get('horizontal'),item.get('slot')] for item in region['items']]
            differences=[key for key in ('status','reason','signature','snapshot','recommendation','mask','observation_sha256')
                         if outputs['baseline'][key]!=outputs['candidate'][key]]
            row={'path':path.relative_to(ROOT).as_posix(),'size':size,'differences':differences,**outputs}
            scenes.append(row);print('SCENE',path.name,size,round(outputs['baseline']['ms']),round(outputs['candidate']['ms']),differences,flush=True)
    report={'candidate':args.candidate,'candidate_sha256':file_sha256(candidate_path),
            'baseline_sha256':baseline.manifest['model_sha256'],'public':metrics,'public_changes':changes,
            'max_public_score_error':max_score_error,'scenes':scenes,
            'precision_guard':{'enabled':args.guard or args.installed,'confidence':.9,'margin':.04,'band':.005,
                               'score_boundaries':[.9,.5],'margin_boundaries':[.04,.3],
                               'rechecked_inputs':candidate.precision_rechecks if args.installed else guard_count[0]},
            'passed':not changes and not any(row['differences'] for row in scenes),
            'scope':'Optimization parity, no training. Public upstream labels are not an independent holdout; resized scenes are derived variants, not additional real captures.'}
    suffix='-installed' if args.installed else '-guarded' if args.guard else ''
    write_json(OUT/(args.candidate+suffix+'-validation.json'),report)
    print('RESULT',report['passed'],'public_changes',len(changes),'scene_changes',sum(bool(r['differences']) for r in scenes),flush=True)


if __name__=='__main__':main()
