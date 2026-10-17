from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import re

import cv2
import numpy as np

from .config import VisionError


def parse_text(text: str, field: str):
    clean = re.sub(r"\s", "", text).replace("−", "-")
    if field == "round":
        match = re.fullmatch(r"([東东南西北])([1-4一二三四])(?:局)?", clean)
        if not match:
            return None
        wind = {"東": "E", "东": "E", "南": "S", "西": "W", "北": "N"}[match[1]]
        number = int(match[2]) if match[2].isdigit() else "一二三四".index(match[2]) + 1
        return {"bakaze": wind, "kyoku": number}
    prefixes = {"remaining": r"(?:余|餘|剩)?", "honba": r"", "kyotaku": r"", "score": r""}
    if field not in prefixes or not re.fullmatch(prefixes[field] + r"-?\d+", clean):
        return None
    value = int(re.search(r"-?\d+$", clean)[0])
    limits = {"score": (-100000, 200000), "remaining": (0, 70), "honba": (0, 99), "kyotaku": (0, 99)}
    lower, upper = limits[field]
    if not lower <= value <= upper or (field == "score" and value % 100):
        return None
    return value


class TextReader:
    def __init__(self, manifest: str | Path = "assets/ocr/manifest.json", *, cache_size: int = 64):
        if cache_size < 0:
            raise ValueError("文字缓存数量不能为负数。")
        self.cache_size = cache_size
        self._read_cache = OrderedDict()
        import onnxruntime as ort
        path = Path(manifest)
        if not path.is_file():
            raise VisionError("文字模型缺失；请执行 scripts/prepare_ocr.py。")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        model = path.parent / metadata["model"]
        if hashlib.sha256(model.read_bytes()).hexdigest() != metadata["model_sha256"]:
            raise VisionError("文字模型校验失败。")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        chars = self.session.get_modelmeta().custom_metadata_map["character"].splitlines()
        self.characters = ["blank", *chars, " "]
        self._warmed=False

    def warmup(self):
        if not self._warmed:
            self._infer(np.full((48,320,3),127,np.uint8))
            self._warmed=True

    def read(self, pixels: np.ndarray, field: str, threshold: float) -> dict:
        # Only identical pixels/field/threshold from this model instance reuse
        # OCR. A changed score or counter is always evaluated again. The bounded
        # cache stores independent dictionaries so consumer edits cannot leak.
        key = (field, threshold, pixels.shape, pixels.dtype.str,
               hashlib.sha256(pixels.tobytes()).digest())
        if key in self._read_cache:
            self._read_cache.move_to_end(key)
            return deepcopy(self._read_cache[key])
        result = self._read(pixels, field, threshold)
        if self.cache_size:
            self._read_cache[key] = deepcopy(result)
            while len(self._read_cache) > self.cache_size:
                self._read_cache.popitem(last=False)
        return result

    def _read(self, pixels: np.ndarray, field: str, threshold: float) -> dict:
        candidates = [self._infer(pixels)]
        if candidates[0][1] < threshold or parse_text(candidates[0][0], field) is None:
            padded = cv2.copyMakeBorder(pixels, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(0,0,0))
            candidates.append(self._infer(padded))
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            inverted = cv2.cvtColor(255-mask, cv2.COLOR_GRAY2BGR)
            padded = cv2.copyMakeBorder(inverted, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255,255,255))
            candidates.append(self._infer(padded))
        accepted = [(text, score, parse_text(text,field)) for text,score in candidates if score >= threshold and parse_text(text,field) is not None]
        values = {json.dumps(value, sort_keys=True) for _,_,value in accepted}
        text, score = max(candidates, key=lambda item: item[1])
        value = None
        reason = "low_ocr_score_or_invalid_value"
        if len(values) == 1:
            text, score, value = max(accepted, key=lambda item:item[1])
            reason = None
        elif len(values) > 1:
            reason = "conflicting_ocr_variants"
        return {"raw_text": text, "score": score, "value": value, "reason": reason,
                "variants": [{"text": text, "score": score} for text,score in candidates],
                "score_kind": "ocr_recognition_score_not_probability"}

    def _infer(self, pixels: np.ndarray) -> tuple[str, float]:
        height, width = pixels.shape[:2]
        resized_width = math.ceil(48*width/height)
        padded_width = max(320, resized_width)
        normalized = cv2.resize(pixels, (resized_width, 48)).astype(np.float32).transpose(2, 0, 1)/127.5 - 1
        tensor = np.zeros((1, 3, 48, padded_width), dtype=np.float32)
        tensor[0, :, :, :resized_width] = normalized
        output = self.session.run(None, {self.input: tensor})[0][0]
        indices, scores = output.argmax(axis=1), output.max(axis=1)
        selected = [i for i, value in enumerate(indices) if value != 0 and (i == 0 or value != indices[i-1])]
        if output.shape[1] != len(self.characters):
            raise VisionError("文字字典和模型输出不一致。")
        text = "".join(self.characters[indices[i]] for i in selected)
        score = float(np.mean(scores[selected])) if selected else 0.0
        self._warmed=True
        return text, score
