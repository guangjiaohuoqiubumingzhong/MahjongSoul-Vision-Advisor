"""Build only from reviewed train images and compare with fixed draft 7."""
from pathlib import Path
import json

from app.vision.config import load_config,write_json
from app.vision.templates import TemplateBank,build_templates,HIGHLIGHT_FEATURE_VERSION
from app.vision.recognizer import StaticRecognizer
from app.vision.text import TextReader
from app.vision.evaluate import evaluate
from app.capture.frames import config_digest

BASELINE=Path('configs/history/steam-1366x768-default-draft-7-39d24649ceb67d8263ad10b3f431dd4d6155d543a5ee37c2aea74d989fb3f158.json')


def main():
    reviewed=[]
    training=[]
    for path in sorted(Path('data').rglob('*.annotation.json')):
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        if value.get('reviewed'):
            reviewed.append(path)
            if value['split']=='train':
                training.append(path)
    baseline=load_config(BASELINE)
    reader=TextReader()
    baseline_result=evaluate(StaticRecognizer(baseline,TemplateBank(baseline['templates'],baseline.get('classifier_manifest')),reader),reviewed)
    if config_digest(load_config('configs/window-steam-1366-v2.json'))==config_digest(baseline):
        write_json('reports/stage2-acceptance.json',baseline_result)
    write_json('reports/vision/draft7-before-manual-samples-acceptance.json',baseline_result)
    config={**baseline,'version':'steam-1366x768-manual-public-candidate-4',
            'templates':'assets/templates/steam-manual-public-candidate/manifest.json'}
    build_templates(training,config['templates'],feature_version=HIGHLIGHT_FEATURE_VERSION)
    write_json('configs/window-steam-1366-manual-public-candidate.json',config)
    candidate_result=evaluate(StaticRecognizer(config,TemplateBank(config['templates'],config.get('classifier_manifest')),reader),reviewed)
    write_json('reports/vision/manual-public-candidate-acceptance.json',candidate_result)
    for name,r in [('baseline',baseline_result),('candidate',candidate_result)]:
        print(name,{key:r[key] for key in ['dataset_images','dataset_known_tiles','dataset_distinct_scenes','dataset_distinct_scene_known_tiles','regions','melds','text','gate_passed']})


if __name__=='__main__':
    main()
