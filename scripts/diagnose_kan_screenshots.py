"""Replay the two user kan screenshots; retain untouched sources and observations."""
from pathlib import Path
import json
import shutil
import cv2
import hashlib
from app.advice import ScreenshotAdvice
from app.capture.frames import load_image, save_image
from app.vision.config import load_config, write_json
from app.vision.recognizer import StaticRecognizer, overlay
from app.vision.templates import TemplateBank
from app.vision.text import TextReader
from app.state.snapshot import build_snapshot, SnapshotError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/fixtures/kan-recognition'
SOURCES = {
    'top-3p-ankan': 'codex-clipboard-dd3ad678-53e5-4b6b-83e6-603aba966e8d.png',
    'top-C-pon-arrow': 'codex-clipboard-aad72aa5-1c99-4d44-b1f5-ebcb2e946133.png',
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='after')
    parser.add_argument('--inference', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'reports/kan-recognition-fix' / args.stage
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / 'configs/window-steam-1366-v2.json')
    recognizer = StaticRecognizer(config, TemplateBank(config['templates'], config.get('classifier_manifest')), TextReader())
    advice = None
    if args.inference:
        from app.engine.runtime import OnnxRuntime
        from app.engine.snapshot import SnapshotBackend
        advice = ScreenshotAdvice(config, recognizer=recognizer, backend=SnapshotBackend(OnnxRuntime()))
    rows, sources = [], {}
    for name, filename in SOURCES.items():
        target = DATA / (name + '-desktop.png')
        if not target.exists():
            shutil.copyfile(Path('C:/Users/16bits_love/AppData/Local/Temp') / filename, target)
        pixels = load_image(target)
        # The native desktop sources contain the 1920x1080 client at x=1,y=31.
        client = pixels[31:1111, 1:1921].copy()
        save_image(DATA / (name + '-1920.png'), client)
        sources[name] = {'source':target.name, 'source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                         'source_size':[pixels.shape[1],pixels.shape[0]], 'client_crop_xywh':[1,31,1920,1080],
                         'native_client':name+'-1920.png', 'source_kind':'user_desktop_attachment'}
        for size in ((1920,1080),(1366,768)):
            stem = name if size == (1920,1080) else name+'-1366'
            frame = client if size == (1920,1080) else cv2.resize(client,size,interpolation=cv2.INTER_AREA)
            recognizer.river_memory.clear()
            if advice:
                advice.clear()
                response = advice.process(frame)
                observation = response['observation']
                write_json(output / (stem+'-advice.json'),response)
                row = {'name':name,'size':list(size),'status':response['status'],'reason':response['reason'],
                       'corrections':response['corrections'],'context':response['context']}
                if response['snapshot']:
                    row['melds'] = response['snapshot']['melds']
                if response['decision']:
                    row['recommendation'] = response['decision']['recommendation']['event']
            else:
                observation = recognizer.recognize(frame)
                row = {'name':name,'size':list(size),'status':'needs_confirmation','reason':None}
                try:
                    snapshot, allowed = build_snapshot(observation, config)
                    write_json(output / (stem+'-snapshot.json'), {'snapshot':snapshot,'allowed':allowed})
                    row.update(status='snapshot_complete',melds=snapshot['melds'])
                except SnapshotError as exc:
                    row['reason'] = str(exc)
            row['kan_groups'] = [{k:g.get(k) for k in ('labels','observed_labels','type','status','structure_evidence')}
                                 for g in observation['regions']['meld_top']['groups']]
            row['unknown_tiles'] = {n:[i.get('best_label') for i in r['items'] if i['label']=='?']
                                    for n,r in observation['regions'].items()
                                    if r['kind'] not in {'text','buttons','choices'} and any(i['label']=='?' for i in r['items'])}
            rows.append(row)
            write_json(output / (stem+'.json'),observation)
            save_image(output / (stem+'-overlay.png'),overlay(frame,observation))
            print(stem,row['status'],row['reason'] or row.get('recommendation',''),flush=True)
    write_json(DATA / 'sources.json',sources)
    code = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'app/vision/recognizer.py','app/vision/detector.py','app/state/snapshot.py',
        'configs/window-steam-1366-v2.json','configs/window-steam-1366-pjura.json')}
    write_json(output / 'summary.json',{'config_version':config['version'],'code_sha256':code,'rows':rows})

if __name__ == '__main__':
    main()
