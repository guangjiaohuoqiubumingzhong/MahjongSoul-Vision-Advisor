"""Offline ONNX integrity/precision inspection, without a training runtime."""
from collections import Counter
import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "models/mortal.onnx"
onnx.checker.check_model(path)
model = onnx.load(path)
types = Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer)
assert not set(types) & {"FLOAT16", "BFLOAT16", "DOUBLE"}, "Only FP32 floating initializers are allowed"
for node in model.graph.node:
    for attr in node.attribute:
        if node.op_type == "Cast" and attr.name == "to":
            assert attr.i not in (onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16), "FP16 cast found"
from hashlib import sha256
result = {"status": "passed", "sha256": sha256(path.read_bytes()).hexdigest(),
          "ir_version": model.ir_version, "opsets": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
          "initializer_types": dict(types), "nodes": len(model.graph.node), "floating_precision": "FP32",
          "remote_lfs_sha256_verified": "ce6c44cb25f9381cb31065a035556df2ea5e86107b44151a8eff8c266e40971e"}
assert result["sha256"] == result["remote_lfs_sha256_verified"], "Remote pinned LFS digest mismatch"
(ROOT / "reports/model-inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
