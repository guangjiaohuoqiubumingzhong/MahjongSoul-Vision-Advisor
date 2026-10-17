"""Build reversible optimization candidates from the pinned, verified ONNX file."""
from collections import Counter
import json
from pathlib import Path
import time
import tempfile

import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel

from app.vision.pjura import file_sha256

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/pjura-vision'
OUT=ROOT/'.tools/pjura-acceleration'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(OUT.resolve())
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    source=SOURCE/manifest['model']
    assert file_sha256(source)==manifest['model_sha256']
    basic=OUT/'basic-source.onnx'
    if not basic.exists():
        options=ort.SessionOptions();options.intra_op_num_threads=2
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath=str(basic)
        session=ort.InferenceSession(str(source),sess_options=options,providers=['CPUExecutionProvider'])
        del session
    rows=[]
    for name in ('fused-fp32','fused-fp16','fp16'):
        path=OUT/(name+'.onnx');started=time.perf_counter()
        if not path.exists():
            if name.startswith('fused'):
                model=optimizer.optimize_model(str(basic),model_type='vit',num_heads=12,hidden_size=768,
                                               opt_level=0,use_gpu=False)
                print(name,model.get_fused_operator_statistics(),flush=True)
                if name=='fused-fp16':model.convert_float_to_float16(keep_io_types=True)
                model.save_model_to_file(str(path))
            else:
                model=convert_float_to_float16(onnx.load(source),keep_io_types=True,
                                               op_block_list=['LayerNormalization','Softmax'])
                onnx.save_model(model,path)
        try:onnx.checker.check_model(str(path))
        except onnx.checker.ValidationError:
            model=OnnxModel(onnx.load(path));model.topological_sort()
            onnx.save_model(model.model,path)
            onnx.checker.check_model(str(path))
        graph=onnx.load(path)
        row={'name':name,'path':str(path.relative_to(ROOT)),'sha256':file_sha256(path),
             'bytes':path.stat().st_size,'nodes':len(graph.graph.node),
             'operators':dict(Counter(n.op_type for n in graph.graph.node)),
             'build_seconds':time.perf_counter()-started}
        rows.append(row);print(row['name'],row['nodes'],row['bytes'],round(row['build_seconds'],2),flush=True)
    report={'source_model_sha256':manifest['model_sha256'],'candidates':rows}
    (OUT/'build.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
