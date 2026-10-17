from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.config import VisionError
from app.vision.pjura import FP32_MODEL_SHA256,PjuraModel,file_sha256,verify_acceleration


@pytest.fixture
def derivative(tmp_path):
    folder=tmp_path/'accelerated';folder.mkdir()
    source=tmp_path/'manifest.json';source.write_text(json.dumps({'model_sha256':FP32_MODEL_SHA256}))
    evidence=folder/'validation.json'
    evidence.write_text(json.dumps({'passed':True,'candidate_sha256':'derived',
        'baseline_sha256':FP32_MODEL_SHA256,'precision_guard':{'enabled':True,'band':.005}}))
    manifest={'model_sha256':'derived','optimization':{'method':'ort-vit-fused-fp16',
        'source_model_sha256':FP32_MODEL_SHA256,'fp32_guard_band':.005,
        'source_manifest':'../manifest.json','source_manifest_sha256':file_sha256(source),
        'validation_file':'validation.json','validation_sha256':file_sha256(evidence)}}
    return folder/'manifest.json',manifest,source,evidence


def test_derived_model_keeps_a_verified_reference_and_requires_cuda(derivative):
    path,manifest,source,_=derivative
    assert verify_acceleration(path,manifest,'CUDAExecutionProvider')==source
    with pytest.raises(VisionError,match='CUDA'):
        verify_acceleration(path,manifest,'CPUExecutionProvider')
    assert verify_acceleration(path,{},'CPUExecutionProvider') is None


@pytest.mark.parametrize('mutation',['source','guard','evidence_hash','outside','not_passed','different_model'])
def test_unverified_or_changed_optimization_is_rejected(derivative,mutation):
    path,manifest,_,evidence=derivative
    if mutation=='source':manifest['optimization']['source_model_sha256']='other'
    elif mutation=='guard':manifest['optimization']['fp32_guard_band']=.01
    elif mutation=='evidence_hash':manifest['optimization']['validation_sha256']='other'
    elif mutation=='outside':manifest['optimization']['validation_file']='../validation.json'
    else:
        checked=json.loads(evidence.read_text())
        checked['passed' if mutation=='not_passed' else 'candidate_sha256']=False if mutation=='not_passed' else 'other'
        evidence.write_text(json.dumps(checked))
        manifest['optimization']['validation_sha256']=file_sha256(evidence)
    with pytest.raises(VisionError):verify_acceleration(path,manifest,'CUDAExecutionProvider')


def test_only_confidence_and_margin_boundary_inputs_use_original_precision():
    model=PjuraModel.__new__(PjuraModel)
    model._reference_manifest='source';model.confidence,model.margin=.9,.04
    model.manifest={'optimization':{'fp32_guard_band':.005}};model.precision_rechecks=0
    calls=[]
    def exact(crops):
        calls.extend(crops)
        return [{'label':'3s','score':.90015,'margin':.88},{'label':'8s','score':.99,'margin':.0399},
                {'label':'6p','score':.5001,'margin':.2},{'label':'9s','score':.7,'margin':.2999}]
    model._reference_model=SimpleNamespace(predict_many=exact)
    crops=[np.full((3,3,3),i,np.uint8) for i in range(6)]
    predictions=[{'label':'1m','score':.99,'margin':.9},{'label':'3s','score':.89994,'margin':.88},
                 {'label':'8s','score':.99,'margin':.0401},{'label':'2m','score':.6,'margin':.15},
                 {'label':'6p','score':.4999,'margin':.2},{'label':'9s','score':.7,'margin':.3001}]
    before=deepcopy(predictions)
    result=model._precision_guard(crops,predictions)
    assert [int(c[0,0,0]) for c in calls]==[1,2,4,5] and model.precision_rechecks==4
    assert result[0]==before[0] and result[3]==before[3]
    assert result[1]['score']>.9 and result[2]['margin']<.04
    assert all(result[i]['precision_recheck']=='fp32_threshold_boundary' for i in (1,2,4,5))
