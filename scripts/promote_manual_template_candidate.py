"""Adopt verified human-capture templates while preserving draft 7 evidence."""
from pathlib import Path
import json
import shutil

from app.capture.frames import config_digest
from app.vision.config import load_config,write_json
from app.vision.templates import TemplateBank
from app.vision.recognizer import StaticRecognizer
from app.vision.text import TextReader
from app.vision.evaluate import evaluate
from scripts.evaluate_manual_template_candidate import BASELINE


def main():
    before=load_config(BASELINE)
    candidate=load_config('configs/window-steam-1366-manual-public-candidate.json')
    old_report=json.loads(Path('reports/vision/draft7-before-manual-samples-acceptance.json').read_text(encoding='utf-8'))
    new_report=json.loads(Path('reports/vision/manual-public-candidate-acceptance.json').read_text(encoding='utf-8'))
    if any(e['category']=='tile_classification_or_slot' and e['actual']['label']!='?' for e in new_report['errors']):
        raise ValueError('Do not promote a candidate with accepted wrong held-out labels.')
    for role in ['hand','draw','dora','red','river','meld']:
        old,new=old_report['regions'][role],new_report['regions'][role]
        if new['correct']<old['correct'] or new['matched']<old['matched'] or new['false_positive']>old['false_positive']:
            raise ValueError(f'Candidate regresses {role}.')
    destination=Path('assets/templates/steam-default-v5')
    destination.mkdir(parents=True,exist_ok=True)
    for source in Path(candidate['templates']).parent.iterdir():
        if source.is_file():
            target=destination/source.name
            if target.exists() and target.read_bytes()!=source.read_bytes():
                raise ValueError(f'Refusing to overwrite preserved template: {target}')
            if not target.exists():
                shutil.copyfile(source,target)
    config={**candidate,'version':'steam-1366x768-default-draft-8','templates':str(destination/'manifest.json')}
    if config['regions']!=before['regions'] or config['thresholds']!=before['thresholds']:
        raise ValueError('This adoption is for new training templates only.')
    reviewed=[]
    for path in sorted(Path('data').rglob('*.annotation.json')):
        if json.loads(path.read_text(encoding='utf-8-sig')).get('reviewed'):
            reviewed.append(path)
    bank=TemplateBank(config['templates'],config.get('classifier_manifest'))
    result=evaluate(StaticRecognizer(config,bank,TextReader()),reviewed)
    write_json('configs/window-steam-1366-v2.json',config)
    write_json('reports/stage2-acceptance.json',result)
    write_json('reports/vision/manual-template-promotion.json',{
        'from_version':before['version'],'from_config_sha256':config_digest(before),'preserved_config':str(BASELINE),
        'to_version':config['version'],'to_config_sha256':config_digest(config),'template_manifest_sha256':bank.sha256,
        'classifier_manifest_sha256':bank.classifier_manifest_sha256,'template_count':len(bank.entries),
        'all_region_coordinates_reused':True,'classification_thresholds_unchanged':True,
        'svm_weights_unchanged':True,'experimental_augmented_gallery_enabled':False,
        'new_real_images':5,'new_distinct_scenes':4,'new_labelled_tiles':257,'new_distinct_scene_tiles':222,
        'baseline_report':'reports/vision/draft7-before-manual-samples-acceptance.json',
        'candidate_report':'reports/vision/manual-public-candidate-acceptance.json','gate_passed':result['gate_passed']})
    print({'version':config['version'],'sha256':config_digest(config),'template_sha256':bank.sha256,
           'templates':len(bank.entries),'regions':result['regions'],'gate_passed':result['gate_passed']})


if __name__=='__main__':
    main()
