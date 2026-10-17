"""Compare CUDA graph/precision candidates without changing the running config."""
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from app.vision.config import write_json
from app.vision.pjura import PjuraModel,file_sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pjura-acceleration'


def session(path,tf32=0):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.log_severity_level=3
    value=ort.InferenceSession(str(path),sess_options=options,
        providers=[('CUDAExecutionProvider',{'device_id':'0','use_tf32':str(tf32)})])
    value.disable_fallback()
    assert value.get_providers()[0]=='CUDAExecutionProvider'
    return value


def softmax(logits):
    probabilities=np.exp(logits-logits.max(axis=1,keepdims=True))
    return probabilities/probabilities.sum(axis=1,keepdims=True)


def main():
    baseline=PjuraModel(ROOT/'models/pjura-vision/manifest.json')
    inputs=np.load(ROOT/'models/pjura-vision/export-reference.npz',allow_pickle=False)['pixels']
    reference=baseline.session.run(['logits'],{'pixel_values':inputs})[0]
    rows=[]
    candidates=[('original-fp32',ROOT/'models/pjura-vision/model.onnx',0),
                ('original-tf32',ROOT/'models/pjura-vision/model.onnx',1)]
    candidates += [(name,ROOT/'.tools/pjura-acceleration'/(name+'.onnx'),0)
                   for name in ('fused-fp32','fused-fp16','fp16')]
    for name,path,tf32 in candidates:
        try:
            engine=baseline.session if name=='original-fp32' else session(path,tf32)
            logits=engine.run(['logits'],{'pixel_values':inputs})[0]
            assert np.isfinite(logits).all()
            row={'name':name,'path':str(path.relative_to(ROOT)),'model_sha256':file_sha256(path),
                 'tf32':tf32,'reference_top1_same':bool((logits.argmax(1)==reference.argmax(1)).all()),
                 'max_logit_error':float(np.abs(logits-reference).max()),
                 'max_probability_error':float(np.abs(softmax(logits)-softmax(reference)).max()),'batches':{}}
            for batch in (1,16):
                tensor=np.ascontiguousarray(np.tile(inputs,(int(np.ceil(batch/3)),1,1,1))[:batch])
                for _ in range(2):engine.run(['logits'],{'pixel_values':tensor})
                times=[]
                for _ in range(8):
                    started=time.perf_counter();engine.run(['logits'],{'pixel_values':tensor})
                    times.append((time.perf_counter()-started)*1000)
                row['batches'][str(batch)]={'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95))}
            rows.append(row);print(json.dumps(row),flush=True)
            del engine
        except Exception as exc:
            rows.append({'name':name,'error':str(exc)});print(name,type(exc).__name__,str(exc),flush=True)
    write_json(OUT/'microbench.json',{'scope':'Three training export references; CUDA inference incl. host input/output copies. Not accuracy or end-to-end acceptance.','rows':rows})


if __name__=='__main__':main()
