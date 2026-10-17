"""Evaluate the real supplied countdown and existing independent negative frames."""
import json
from pathlib import Path
import sys
import cv2

from app.advice import ScreenshotAdvice, load_screenshot, save_advice, format_advice
from app.engine.runtime import OnnxRuntime
from app.engine.snapshot import SnapshotBackend, ASSUMPTIONS
from app.vision.config import load_config, write_json
from app.vision.turn import detect_timer


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/window-steam-1366-v2.json')
    service=ScreenshotAdvice(cfg,backend=SnapshotBackend(OnnxRuntime()))
    folder=root/'reports/snapshot-advice'
    pixels,metadata=load_screenshot(root/'data/screenshots/user-countdown-example/window.png',cfg)
    positive=service.process(pixels,metadata)
    save_advice(folder/'countdown-advice.json',positive,pixels)
    assert positive['status']=='ready',positive.get('reason')
    assert positive['corrections']=={} and positive['context']=={}
    # This second user image is the advice panel containing a reduced preview,
    # so verify the timer only; it is not a full-resolution board benchmark.
    panel=cv2.imread(str(root/'data/screenshots/user-single-countdown-example/panel.png'))
    preview=cv2.resize(panel[261:553,21:541],(1366,768))
    single=detect_timer(preview,cfg['action_timer'],service.recognizer.text_reader)
    assert single['active'] and single['value']=={'fixed_seconds':15,'reserve_seconds':None}
    panel=cv2.imread(str(root/'data/screenshots/user-three-digit-countdown-example/panel.png'))
    preview=cv2.resize(panel[261:553,21:541],(1366,768))
    triple=detect_timer(preview,cfg['action_timer'],service.recognizer.text_reader)
    assert triple['active'] and triple['value']=={'fixed_seconds':293,'reserve_seconds':None}
    negatives=[]
    index=json.loads((root/'reports/vision/pjura-test/index.json').read_text(encoding='utf-8'))
    for entry in index:
        p,m=load_screenshot(entry['image'],cfg)
        result=service.process(p,m)
        assert result['status']=='waiting_turn' and result['recommendation'] is None
        negatives.append({'image':entry['image'],'status':result['status'],
                          'timer':result['observation']['action_timer']})
    report={'status':'passed','mode':'single_current_screenshot_no_history',
            'positive_real_images':1,'positive_ready_without_corrections':1,
            'single_countdown_panel_preview':single,
            'three_digit_countdown_panel_preview':triple,
            'negative_independent_images':len(negatives),'negative_false_triggers':0,
            'screenshot_has_user_red_circle':True,'raw_window_crop':[1,31,1366,768],
            'message':format_advice(positive),'timer':positive['observation']['action_timer'],
            'self_wind':positive['observation']['self_wind'],
            'recognition_ms':positive['observation']['recognition_ms'],
            'decision_ms':positive['decision']['total_ms'],'total_ms':positive['total_ms'],
            'execution_provider':positive['decision']['execution_provider'],
            'model_version':positive['decision']['model_version'],'assumptions':ASSUMPTIONS,
            'positive_report':str((folder/'countdown-advice.json').resolve()),'negatives':negatives,
            'limits':['One positive real countdown screenshot; live game monitoring not yet empirically accepted',
                      'Reaction source and ambiguous melds require current-frame confirmation',
                      'Fixed Mortal four-player red-five hanchan model; does not infer room rule variations']}
    write_json(folder/'acceptance.json',report)
    print(json.dumps({k:report[k] for k in ['status','message','positive_ready_without_corrections',
                                          'negative_independent_images','negative_false_triggers','total_ms']},ensure_ascii=False))


if __name__=='__main__':main()
