"""Export verified upstream weights in an isolated CPU Torch environment."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".tools/pjura-export/hf-cache")

import numpy as np
import onnx
from PIL import Image
import torch
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

FOLDER = ROOT / "models/pjura-vision"
UPSTREAM = FOLDER / "upstream"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    provenance = json.loads((UPSTREAM / "download.json").read_text(encoding="utf-8"))
    for filename, expected in provenance["files"].items():
        if digest(UPSTREAM / filename) != expected:
            raise ValueError(f"Upstream checksum mismatch: {filename}")
    torch.set_num_threads(2)
    config = ViTConfig.from_json_file(str(UPSTREAM / "config.json"))
    config._attn_implementation = "eager"
    model = ViTForImageClassification(config)
    # Never unpickle unrestricted upstream code or training_args.bin.
    weights = torch.load(UPSTREAM / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(str(UPSTREAM), local_files_only=True)
    crops = []
    references = []
    for path in sorted((ROOT / "data").rglob("*.annotation.json")):
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        if not annotation.get("reviewed") or annotation.get("split") != "train":
            continue
        image_path = Path(annotation["image"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        with Image.open(image_path) as image:
            for target in annotation["targets"]:
                if target.get("kind") != "tile" or target.get("region") not in {"hand", "draw"}:
                    continue
                x, y, w, h = target["bbox"]
                crops.append(image.convert("RGB").crop((x, y, x+w, y+h)))
                references.append({"annotation": str(path.relative_to(ROOT)), "id": target["id"]})
                if len(crops) == 3:
                    break
        if len(crops) == 3:
            break
    if len(crops) != 3:
        raise ValueError("Need three reviewed training crops for export validation")
    pixels = processor(images=crops, return_tensors="pt")["pixel_values"]

    class Logits(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, pixel_values):
            return self.network(pixel_values=pixel_values).logits

    wrapper = Logits(model).eval()
    with torch.no_grad():
        expected = wrapper(pixels).numpy()
        expected_single = wrapper(pixels[:1]).numpy()
        torch.onnx.export(wrapper, pixels[:1], str(FOLDER / "model.onnx"),
                          input_names=["pixel_values"], output_names=["logits"],
                          dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(FOLDER / "model.onnx"))
    np.savez_compressed(FOLDER / "export-reference.npz", pixels=pixels.numpy(),
                        logits=expected, logits_single=expected_single,
                        rgb_crops=np.array([np.asarray(crop.resize((224, 224), Image.Resampling.BILINEAR))
                                            for crop in crops]))
    manifest = {"schema_version": 1, **provenance, "model": "model.onnx",
                "model_sha256": digest(FOLDER / "model.onnx"),
                "id2label": config.id2label,
                "preprocessor": json.loads((UPSTREAM / "preprocessor_config.json").read_text()),
                "export": {"opset": 17, "weights_only": True, "reference_crops": references,
                           "versions": {name: importlib.metadata.version(name)
                                        for name in ("torch", "transformers", "onnx", "numpy", "Pillow")}},
                "scope": "34 base tile classes only; no red-five, back, text, button or detector classes",
                "validation": "pending"}
    (FOLDER / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"model_sha256": manifest["model_sha256"], "classes": len(config.id2label)}))


if __name__ == "__main__":
    main()
