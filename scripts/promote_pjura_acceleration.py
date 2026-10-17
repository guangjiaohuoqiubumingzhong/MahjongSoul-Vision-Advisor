"""Install a parity-validated derivative alongside the immutable FP32 source."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import onnxruntime as ort

from app.vision.config import write_json
from app.vision.pjura import FP32_MODEL_SHA256,file_sha256,verify_acceleration

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate',action='store_true')
    parser.add_argument('--validation',type=Path,default=ROOT/'reports/pjura-acceleration/fused-fp16-guarded-validation.json')
    args=parser.parse_args()
    source=ROOT/'models/pjura-vision/manifest.json'
    original=json.loads(source.read_text(encoding='utf-8'))
    evidence=args.validation
    checked=json.loads(evidence.read_text(encoding='utf-8'))
    candidate=ROOT/'.tools/pjura-acceleration/fused-fp16.onnx'
    assert original['model_sha256']==checked['baseline_sha256']==FP32_MODEL_SHA256
    assert checked['passed'] is True and checked['precision_guard']['enabled'] is True
    assert file_sha256(candidate)==checked['candidate_sha256']
    folder=ROOT/'models/pjura-vision/accelerated';folder.mkdir(parents=True,exist_ok=True)
    model_name='vit-fused-fp16-'+checked['candidate_sha256'][:12]+'.onnx'
    report_name='validation-'+file_sha256(evidence)[:12]+'.json'
    shutil.copyfile(candidate,folder/model_name);shutil.copyfile(evidence,folder/report_name)
    manifest=deepcopy(original)
    manifest['source_export']=manifest.pop('export')
    manifest.update(model=model_name,model_sha256=checked['candidate_sha256'])
    manifest['optimization']={
        'method':'ort-vit-fused-fp16','precision':'mixed-fp16-fp32-io',
        'onnxruntime':ort.__version__,'source_model_sha256':FP32_MODEL_SHA256,
        'source_manifest':'../manifest.json','source_manifest_sha256':file_sha256(source),
        'validation_file':report_name,'validation_sha256':file_sha256(folder/report_name),
        'fp32_guard_band':.005,
        'sources':['https://onnxruntime.ai/docs/performance/transformers-optimization.html',
                   'https://onnxruntime.ai/docs/performance/model-optimizations/float16.html']}
    manifest['validation']={'status':'passed','public_crops':checked['public']['candidate']['samples'],
        'public_top1_and_acceptance_unchanged':True,'scene_variants':len(checked['scenes']),
        'scene_state_and_recommendations_unchanged':True,
        'scope':'Numerical/recognition parity with the pinned FP32 model using threshold-boundary rechecks; not new training or a new held-out accuracy estimate.'}
    verify_acceleration(folder/'manifest.json',manifest,'CUDAExecutionProvider')
    write_json(folder/'manifest.json',manifest)
    if args.activate:
        for name in ('window-steam-1366-v2.json','window-steam-1366-pjura.json'):
            path=ROOT/'configs'/name;config=json.loads(path.read_text(encoding='utf-8'))
            config['version']='steam-1366x768-pjura-snapshot-draft-21'
            config['tile_classifier']['manifest']='models/pjura-vision/accelerated/manifest.json'
            write_json(path,config)
    print(json.dumps({'model':model_name,'manifest':str(folder/'manifest.json'),'activated':args.activate}))


if __name__=='__main__':main()
