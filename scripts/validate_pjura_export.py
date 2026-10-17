"""Verify ONNX logits against Torch reference batches, in the application environment."""
from pathlib import Path
import json

import numpy as np

from app.vision.config import write_json
from app.vision.pjura import PjuraModel, file_sha256, preprocess

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "models/pjura-vision"


def main():
    model = PjuraModel(FOLDER / "manifest.json", allow_unvalidated=True)
    reference = np.load(FOLDER / "export-reference.npz", allow_pickle=False)
    preprocessing_error = float(np.max(np.abs(
        np.stack([preprocess(crop[:, :, ::-1].copy()) for crop in reference["rgb_crops"]])
        - reference["pixels"])))
    # rgb_crops are already resized; separately comparing the original crops
    # below checks PIL resizing as well as the channel/normalization contract.
    from PIL import Image
    manifest = model.manifest
    raw = []
    for record in manifest["export"]["reference_crops"]:
        path = ROOT / record["annotation"]
        annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        target = next(t for t in annotation["targets"] if t["id"] == record["id"])
        image_path = Path(annotation["image"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        x, y, w, h = target["bbox"]
        with Image.open(image_path) as image:
            raw.append(np.asarray(image.convert("RGB").crop((x, y, x+w, y+h)))[:, :, ::-1].copy())
    pixels = np.stack([preprocess(crop) for crop in raw])
    original_preprocessing_error = float(np.max(np.abs(pixels - reference["pixels"])))
    results = []
    for size, expected in ((1, reference["logits_single"]), (3, reference["logits"])):
        actual = model.session.run(["logits"], {"pixel_values": pixels[:size]})[0]
        same_top1 = bool(np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1)))
        error = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        assert same_top1
        results.append({"batch_size": size, "max_absolute_logit_error": error, "same_top1": same_top1})
    assert preprocessing_error <= 1e-6 and original_preprocessing_error <= 1e-6
    validation = {"status": "passed", "provider": model.provider,
                  "reference_sha256": file_sha256(FOLDER / "export-reference.npz"),
                  "preprocessing_max_absolute_error": original_preprocessing_error,
                  "cases": results, "scope": "export parity on three training crops, not recognition accuracy"}
    manifest["validation"] = validation
    write_json(FOLDER / "manifest.json", manifest)
    write_json(ROOT / "reports/vision/pjura-export-validation.json", validation)
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
