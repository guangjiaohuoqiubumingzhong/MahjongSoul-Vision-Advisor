"""Replay original captures; report cold setup and uncached/changing-frame latency."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from app.advice import ScreenshotAdvice, load_screenshot
from app.capture.frames import Frame
from app.ui.advice import CAPTURE_DELAY_SECONDS
from app.vision.config import load_config, pixel_rect, to_reference, write_json
from scripts.benchmark_static_vision import signature

ROOT=Path(__file__).resolve().parents[1]


def state_signature(value):
    value=deepcopy(value)
    for region in value.values():
        region['items']=[[item[0],item[2]] for item in region['items']]
    return value


def pipeline_replay(service,paths,output):
    """Exercise the actual worker with supplied frames; never controls a game."""
    import app.ui.advice as ui
    rows=[];original_capture=ui.WindowCapture
    try:
        for index,path in enumerate(paths):
            pixels,metadata=load_screenshot(path,service.config)
            reference,_,_=to_reference(pixels,service.config)
            stop=threading.Event();published=[]
            service.clear();service.recognizer.river_memory.clear()
            # A new turn's first image, not a timing of a cached whole screenshot.
            service.recognizer.bank._cache.clear()
            service.recognizer.bank._classification_cache.clear()
            service.recognizer.text_reader._read_cache.clear()
            class Capture:
                frame_id=0
                def grab(self):
                    self.frame_id+=1
                    return Frame(pixels.copy(),{**metadata,'frame_id':self.frame_id,
                        'source':'offline_latency_replay'})
                def grab_region(self,rect):
                    x,y,w,h=pixel_rect(rect,*service.config['client_size'])
                    return SimpleNamespace(pixels=reference[y:y+h,x:x+w])
                def close(self):pass
            ui.WindowCapture=lambda *a:Capture()
            panel=ui.AdvicePanel.__new__(ui.AdvicePanel)
            panel.config,panel.output=service.config,output
            panel.get_service=lambda count:service
            def publish(generation,result,image):
                if 'latency' in result:
                    published.append(deepcopy(result));stop.set()
            panel.publish=publish
            started=time.perf_counter();panel.live_worker(index,stop,0)
            worker_ms=(time.perf_counter()-started)*1000
            result=published[-1]
            rows.append({'path':path.relative_to(ROOT).as_posix(),'status':result['status'],
                         'latency':result['latency'],'worker_including_disk_ms':worker_ms,
                         'estimated_display_ms':result['latency']['trigger_to_result_ms']+100,
                         'recaptures':result.get('recaptures',0),
                         'recapture_budget_exhausted':result.get('recapture_budget_exhausted',False)})
            print('PIPELINE',rows[-1]['path'],result['status'],round(rows[-1]['estimated_display_ms']),flush=True)
    finally:ui.WindowCapture=original_capture
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path)
    parser.add_argument('--runs',type=int,default=2)
    parser.add_argument('--pipeline',action='store_true')
    args=parser.parse_args()
    paths=[ROOT/'data/fixtures'/name for name in (
        'river-left-late/client-1920.png',
        'kan-recognition/top-3p-ankan-1920.png',
        'kan-all-seats/user-self-F-kakan-1920.png',
        'no-draw-call-discard/chi-buttons-1920.png',
        'no-draw-call-discard/ron-buttons-1920.png')]
    for action in ('0093','0094','0112','0113'):
        directory=ROOT/f'reports/screenshot-advice/run-20261015T083403-4/action-{action}'
        paths.append(next(directory.glob('[0-9]*.png')))
    config=load_config(ROOT/'configs/window-steam-1366-v2.json')
    started=time.perf_counter()
    service=ScreenshotAdvice(config)
    vision_setup_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter();service.prepare()
    decision_setup_ms=(time.perf_counter()-started)*1000
    counts=defaultdict(lambda:{'calls':0,'ms':0,'inputs':0})
    def instrument(target,name,label):
        original=getattr(target,name)
        def measured(*a,**k):
            started=time.perf_counter()
            try:return original(*a,**k)
            finally:
                entry=counts[label];entry['calls']+=1
                entry['ms']+=(time.perf_counter()-started)*1000
                if label=='network':entry['inputs']+=len(a[0])
        setattr(target,name,measured)
    bank=service.recognizer.bank
    instrument(bank.model,'predict_many','network')
    instrument(bank.legacy,'classify','templates')
    instrument(service.recognizer.text_reader,'_infer','ocr')
    rows=[]
    for run in range(args.runs):
        for path in paths:
            pixels,metadata=load_screenshot(path,config)
            service.clear();service.recognizer.river_memory.clear();counts.clear()
            started=time.perf_counter();result=service.process(pixels,metadata)
            elapsed=(time.perf_counter()-started)*1000
            observation=result['observation'];decision=result.get('decision') or {}
            row={'run':run,'path':path.relative_to(ROOT).as_posix(),
                 'image_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                 'process_ms':elapsed,'recognition_ms':observation['recognition_ms'],
                 'decision_ms':decision.get('total_ms'),
                 'stages':deepcopy(dict(counts)),
                 'status':result['status'],'reason':result['reason'],
                 'recognition':json.loads(json.dumps(signature(observation))),
                 'snapshot_sha256':decision.get('snapshot_sha256'),
                 'action':(result.get('recommendation') or {}).get('event'),
                 'scores':[c['model_score'] for c in decision.get('legal_candidates',[])]}
            rows.append(row)
            print(run,row['path'],row['status'],round(elapsed),flush=True)
    report={'scope':'Offline original captures, excluding live window acquisition, animation/retry waits, UI polling and disk writes. Runs contain distinct changing frames, not repeated one-frame timings.',
            'vision_setup_ms':vision_setup_ms,'decision_setup_ms':decision_setup_ms,
            'fixed_animation_wait_ms':CAPTURE_DELAY_SECONDS*1000,'ui_poll_interval_ms':100,'samples':rows}
    for run in range(args.runs):
        times=[r['process_ms'] for r in rows if r['run']==run]
        report[f'run_{run}']={'median_ms':float(np.median(times)),
                             'p95_ms':float(np.quantile(times,.95)),'max_ms':max(times),
                             'max_with_animation_and_ui_ms':max(times)+CAPTURE_DELAY_SECONDS*1000+100}
    if args.compare:
        baseline=json.loads(args.compare.read_text(encoding='utf-8'))
        old={(r['run'],r['path']):r for r in baseline['samples']}
        differences=[];diagnostic_changes=[]
        for row in rows:
            before=old[(row['run'],row['path'])]
            for key in ('image_sha256','status','reason','recognition','snapshot_sha256','action'):
                if key=='recognition':
                    if state_signature(before[key])!=state_signature(row[key]):
                        differences.append({'run':row['run'],'path':row['path'],'field':key})
                    elif before[key]!=row[key]:
                        diagnostic_changes.append({'run':row['run'],'path':row['path'],'field':'classifier_quarter_turn'})
                    continue
                if before[key]!=row[key]:differences.append({'run':row['run'],'path':row['path'],'field':key})
            if not np.allclose(before['scores'],row['scores'],atol=1e-6,rtol=0):
                differences.append({'run':row['run'],'path':row['path'],'field':'scores'})
        report['comparison']={'baseline':str(args.compare),'differences':differences,
                              'unchanged':not differences,'diagnostic_changes':diagnostic_changes,
                              'comparison_scope':'Labels/slots, region completeness, text, melds, snapshot, action and Q scores; raw classifier quarter turns are diagnostic, not physical orientation. Full physical orientation parity is covered by the scene validation report.'}
    if args.pipeline:
        report['pipeline_scope']=f'Actual automatic worker, original supplied frames, includes {CAPTURE_DELAY_SECONDS:g}s animation wait, countdown/hand checks and retries; adds worst-case 0.1s UI polling. Native window capture latency and real animation changes are excluded.'
        report['pipeline']=pipeline_replay(service,[paths[0],paths[3],paths[6]],args.output.parent/'pipeline')
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','scope')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
