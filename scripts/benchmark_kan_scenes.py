"""Replay manually reviewed public kan ROIs without altering native pixels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from app.capture.frames import load_image,save_image
from app.vision.config import load_config,write_json
from app.vision.recognizer import StaticRecognizer,overlay
from app.vision.templates import TemplateBank

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/public-mahjong'


def component_config(config,case,width,height):
    """Only ROI location/pose is calibrated; expected labels never enter inference."""
    config=deepcopy(config)
    name='meld_'+case['seat']
    region=deepcopy(config['regions'][name])
    for key in ('concealed_hand_exclusions','exclude_polygons'):
        region.pop(key,None)
    x,y,w,h=case['roi_xywh']
    region['rect']=[x/width,y/height,w/width,h/height]
    # Keep every screenshot's aspect and native classifier inputs; no stretch
    # into a 16:9 client. Detection alone uses the usual 768px reference height.
    config['client_size']=[round(width*768/height),768]
    config['regions']={name:region}
    config.pop('action_timer',None);config.pop('self_wind',None)
    return config


def evaluate(case,config,bank,recognizer_type=StaticRecognizer):
    source=BASE/case['path']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=case['sha256']:
        raise ValueError('Reviewed original changed: '+case['path'])
    pixels=load_image(source);height,width=pixels.shape[:2]
    configured=component_config(config,case,width,height)
    recognizer=recognizer_type(configured,bank)
    observation=recognizer.recognize(pixels,{'evaluation_mode':'native_public_meld_component'})
    region=observation['regions']['meld_'+case['seat']]
    expected=case['expected']
    # Multiple groups in the labelled ROI are not quietly reduced to a match.
    groups=region['groups']
    group=groups[0] if len(groups)==1 else None
    correct=bool(group and group['status']=='confirmed'
                 and group['type']==expected['type'] and sorted(group['labels'])==sorted(expected['labels'])
                 and (expected['target'] is None or group.get('target')==expected['target']))
    wrong=bool(group and group['status']=='confirmed' and not correct)
    result={**case,'correct':correct,'wrong_confirmation':wrong,'complete':region['complete'],
            'groups':[{k:g.get(k) for k in ('type','status','labels','called_positions','target','added_position',
                         'called_position_evidence','structure_evidence')} for g in groups],
            'items':[{k:i.get(k) for k in ('label','best_label','score','reason','horizontal','canonical_bbox',
                                             'structural_reading')} for i in region['items']],
            'back_count':len(region.get('backs',[])),
            'blocks':{k:region[k] for k in ('unresolved_components','boundary_fragments') if region.get(k)}}
    return result,observation,pixels


def baseline_recognizer():
    """Verified local source snapshot from the preceding draft-18 replay."""
    import importlib.util
    import sys
    base=ROOT/'reports/kan-recognition-fix/public-scenes'
    hashes=json.loads((base/'run.json').read_text(encoding='utf-8'))['code']
    modules={}
    for name in ('detector','recognizer'):
        relative='app/vision/'+name+'.py';path=base/'code'/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative]
        spec=importlib.util.spec_from_file_location('app.vision._kan_baseline_'+name,path)
        module=importlib.util.module_from_spec(spec)
        if name=='recognizer':
            current=sys.modules['app.vision.detector']
            sys.modules['app.vision.detector']=modules['detector']
            try:spec.loader.exec_module(module)
            finally:sys.modules['app.vision.detector']=current
        else:spec.loader.exec_module(module)
        modules[name]=module
    config=load_config(base/'code/configs/window-steam-1366-v2.json')
    return modules['recognizer'].StaticRecognizer,config,hashes


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='after')
    args=parser.parse_args()
    cases=json.loads((BASE/'kan-scenes/regressions.json').read_text(encoding='utf-8'))['cases']
    config=load_config(ROOT/'configs/window-steam-1366-v2.json')
    recognizer_type=StaticRecognizer;baseline_hashes=None
    if args.stage=='before':recognizer_type,config,baseline_hashes=baseline_recognizer()
    bank=TemplateBank(config['templates'],config.get('classifier_manifest'))
    # One resident network, as in the UI. Each recognizer still has fresh state.
    from app.vision.pjura import PjuraBank
    bank=PjuraBank(bank,config['tile_classifier'])
    out=ROOT/'reports/kan-all-seats-fix'/args.stage;out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in cases:
        row,observation,pixels=evaluate(case,config,bank,recognizer_type);rows.append(row)
        write_json(out/(case['name']+'.json'),observation)
        save_image(out/(case['name']+'-overlay.png'),overlay(pixels,observation))
        x,y,w,h=case['roi_xywh'];save_image(out/(case['name']+'-roi.png'),pixels[y:y+h,x:x+w])
        print(case['name'],'correct' if row['correct'] else 'WRONG' if row['wrong_confirmation'] else 'unresolved',row['groups'],flush=True)
    summary={'evaluation_mode':'native_public_meld_component','config_version':config['version'],
             'cases':len(rows),'correct':sum(r['correct'] for r in rows),
             'wrong_confirmation':sum(r['wrong_confirmation'] for r in rows),
             'unresolved':sum(not r['correct'] and not r['wrong_confirmation'] for r in rows),'rows':rows}
    summary['baseline_source_hashes']=baseline_hashes
    if args.stage!='before':
        summary['code_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                               ['app/vision/kan.py','app/vision/detector.py','app/vision/recognizer.py',
                                'app/vision/config.py','configs/window-steam-1366-v2.json']}
    write_json(out/'public-kan-summary.json',summary)
    print({k:v for k,v in summary.items() if k!='rows'})


if __name__=='__main__':main()
