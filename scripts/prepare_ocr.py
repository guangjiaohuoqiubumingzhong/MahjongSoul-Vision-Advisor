"""Extract one pinned OCR model; never install the wheel's CPU ORT dependency."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import onnx

root = Path(__file__).resolve().parents[1]
wheel = root / ".tools/downloads/rapidocr_onnxruntime-1.4.4-py3-none-any.whl"
expected_wheel = "971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf"
if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_wheel:
    raise ValueError("RapidOCR wheel SHA256 differs from official PyPI release metadata")
folder = root / "assets/ocr"
folder.mkdir(parents=True, exist_ok=True)
with ZipFile(wheel) as archive:
    member = "rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx"
    model = folder / "ch_PP-OCRv4_rec_infer.onnx"
    model.write_bytes(archive.read(member))
if hashlib.sha256(model.read_bytes()).hexdigest() != "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b":
    raise ValueError("OCR model hash differs from the pinned wheel")
metadata = {item.key: item.value for item in onnx.load(model).metadata_props}
assert "character" in metadata
manifest = {"source": "https://pypi.org/project/rapidocr-onnxruntime/1.4.4/", "upstream": "https://github.com/RapidAI/RapidOCR/tree/v1.4.4",
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "model": model.name,
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "license": "Apache-2.0",
            "engine": "ONNX Runtime CPUExecutionProvider for fixed text crops only",
            "decision_engine": "Mortal remains CUDAExecutionProvider only; no CPU ORT distribution installed",
            "input": [{"name": v.name, "shape": str(v.type)} for v in onnx.load(model).graph.input]}
(folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: manifest[key] for key in ("model_sha256", "wheel_sha256", "license")}, indent=2))
