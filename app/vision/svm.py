"""Optional train-only SVM verification of an otherwise unknown tile proposal."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from app.state.events import TILES
from .config import VisionError
from .faces import FACE_PARAMETERS, HIGHLIGHT_FACE_PARAMETERS, tile_face

FEATURE_VERSION = "rectified-glyph-hog-colour-projection-v1"
FEATURE_PARAMETERS = {"face":FACE_PARAMETERS,"hog":{"win":[32,48],"block":[16,16],
                      "stride":[8,8],"cell":[8,8],"bins":9},"coarse_colour":[4,6],
                      "coarse_weight":.5,"projection_weight":.5}
HIGHLIGHT_FEATURE_VERSION = "rectified-glyph-hog-colour-projection-v2"
HIGHLIGHT_FEATURE_PARAMETERS = {**FEATURE_PARAMETERS,"face":HIGHLIGHT_FACE_PARAMETERS}
FEATURE_RECIPES = {FEATURE_VERSION:FEATURE_PARAMETERS,HIGHLIGHT_FEATURE_VERSION:HIGHLIGHT_FEATURE_PARAMETERS}
POLICY = {"model_count":3,"minimum_template_proposal_score":.75,
          "required_agreement":"all_models_and_existing_template_proposal",
          "blank_white_dragon_requires_blank_face":True}
_HOG = cv2.HOGDescriptor((32,48),(16,16),(8,8),(8,8),9)


def descriptor(patch, face_parameters=None):
    face = tile_face(patch,face_parameters)
    if face is None:
        return None
    colour = cv2.resize(face.feature,(32,48),interpolation=cv2.INTER_AREA)
    ink = 1-colour.min(axis=2)
    hog = _HOG.compute(np.clip(ink*255,0,255).astype(np.uint8)).ravel()
    coarse = cv2.resize(colour,(4,6),interpolation=cv2.INTER_AREA).ravel()
    projection = np.r_[ink.mean(axis=0),ink.mean(axis=1)]
    return np.r_[hog,coarse*.5,projection*.5].astype(np.float32)


class SvmVerifier:
    def __init__(self,path):
        self.path = Path(path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8-sig"))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        value = self.manifest
        self.feature_version = value.get("feature_version")
        self.feature_parameters = FEATURE_RECIPES.get(self.feature_version)
        if (self.feature_parameters is None or value.get("feature_parameters") != self.feature_parameters or
                value.get("policy") != POLICY or value.get("opencv_version") != cv2.__version__):
            raise VisionError("视觉 SVM 特征、策略或 OpenCV 版本不兼容，停止加载。")
        self.classes = value.get("classes",[])
        if self.classes != sorted(TILES):
            raise VisionError("视觉 SVM 必须区分全部 37 类。")
        sources = value.get("sources",[])
        if not sources or any(source.get("split") != "train" or not source.get("reviewed") or not source.get("game_id") for source in sources):
            raise VisionError("视觉 SVM 来源只能是已核对的 train 对局。")
        self.source_games = {source["game_id"] for source in sources}
        entries = value.get("models",[])
        if len(entries) != POLICY["model_count"] or {(entry.get("C"),entry.get("gamma")) for entry in entries} != {(10,.01),(10,.05),(10,.1)}:
            raise VisionError("视觉 SVM 一致投票模型配置不完整。")
        self.models = []
        for entry in entries:
            model_path = Path(entry["path"])
            if not model_path.is_absolute():
                model_path = self.path.parent/model_path
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != entry.get("sha256"):
                raise VisionError(f"视觉 SVM 模型校验失败：{model_path}")
            model = cv2.ml.SVM_load(str(model_path))
            if not model.isTrained() or model.getVarCount() != 692:
                raise VisionError("视觉 SVM 模型特征维度不符。")
            if abs(model.getC()-entry["C"]) > 1e-10 or abs(model.getGamma()-entry["gamma"]) > 1e-10:
                raise VisionError("视觉 SVM 模型参数与清单不符。")
            self.models.append(model)

    def verify(self,pixels,proposal):
        evidence = {"manifest_sha256":self.sha256,"feature_version":getattr(self,"feature_version",FEATURE_VERSION),
                    "score_kind":"unanimous_model_votes_not_probability","votes":[],"accepted":False}
        if (proposal.get("best_label") not in TILES or
                (proposal.get("score") or -1) < POLICY["minimum_template_proposal_score"] or
                proposal.get("best_label") == "P" and not proposal.get("blank_face")):
            return {**evidence,"reason":"template_proposal_not_supported"}
        feature = descriptor(pixels,getattr(self,"feature_parameters",FEATURE_PARAMETERS)["face"])
        if feature is None:
            return {**evidence,"reason":"unrectifiable_face"}
        votes = [self.classes[int(model.predict(feature[None])[1].ravel()[0])] for model in self.models]
        evidence["votes"] = votes
        accepted = len(set(votes))==1 and votes[0]==proposal["best_label"]
        return {**evidence,"accepted":accepted,"reason":None if accepted else "classifier_or_template_disagreement"}
