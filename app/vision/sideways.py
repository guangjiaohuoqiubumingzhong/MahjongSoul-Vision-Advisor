"""Optional, train-only glyph verification for uncertain sideways tile identities."""
from pathlib import Path
import hashlib
import json
import threading

import cv2
import numpy as np

from app.state.events import BASE
from .config import VisionError
from .faces import HIGHLIGHT_FACE_PARAMETERS
from .svm import descriptor,HIGHLIGHT_FEATURE_PARAMETERS,HIGHLIGHT_FEATURE_VERSION
from .templates import rotate


POLICY = {"minimum_similarity":.94,"minimum_class_gap":.01,"nearest_votes":3,
          "svm_gammas":[.01,.05,.1],"svm_c":10,"rotations":[90,270],
          "excluded_labels":["P"],
          "agreement":"both_rotations_all_svms_and_nearest_references"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SidewaysVerifier:
    def __init__(self,path):
        self.path = Path(path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.sha256 = digest(self.path)
        value = self.manifest
        if (value.get("schema_version") != 1 or value.get("policy") != POLICY
                or value.get("feature_version") != HIGHLIGHT_FEATURE_VERSION
                or value.get("feature_parameters") != HIGHLIGHT_FEATURE_PARAMETERS
                or value.get("opencv_version") != cv2.__version__
                or value.get("classes") != sorted(BASE)):
            raise VisionError("横牌字形模型的特征、类别或策略不兼容。")
        self.classes = value["classes"]
        sources = value.get("sources",[])
        if not sources or any(s.get("split")!="train" or s.get("source")!="pjura" for s in sources):
            raise VisionError("横牌字形模型只能使用声明的训练样本。")
        archive = self.path.parent/value["features"]["path"]
        if digest(archive) != value["features"]["sha256"]:
            raise VisionError("横牌参考字形校验失败。")
        with np.load(archive,allow_pickle=False) as data:
            self.features,self.labels = data["features"],data["labels"]
        if (self.features.shape != (len(sources),692) or self.labels.shape != (len(sources),)
                or not np.isfinite(self.features).all()
                or not np.array_equal(self.labels,[self.classes.index(s["label"]) for s in sources])):
            raise VisionError("横牌参考字形与来源记录不一致。")
        norms = np.linalg.norm(self.features,axis=1,keepdims=True)
        self.normalized = self.features/np.maximum(norms,1e-9)
        self.models = []
        for entry in value["models"]:
            model_path = self.path.parent/entry["path"]
            if digest(model_path) != entry["sha256"]:
                raise VisionError("横牌字形模型校验失败。")
            model = cv2.ml.SVM_load(str(model_path))
            if (not model.isTrained() or model.getVarCount()!=692 or model.getC()!=POLICY["svm_c"]
                    or model.getGamma()!=entry["gamma"]):
                raise VisionError("横牌字形模型参数无效。")
            self.models.append(model)
        if [m.getGamma() for m in self.models] != POLICY["svm_gammas"]:
            raise VisionError("横牌字形投票模型不完整。")
        self.lock = threading.RLock()

    def verify(self,pixels):
        result = {"accepted":False,"manifest_sha256":self.sha256,
                  "score_kind":"glyph_similarity_and_unanimous_votes_not_probability","readings":[]}
        with self.lock:
            for degrees in POLICY["rotations"]:
                feature = descriptor(rotate(pixels,degrees),HIGHLIGHT_FACE_PARAMETERS)
                if feature is None:
                    return {**result,"reason":"missing_glyph_outline"}
                similarity = self.normalized@(feature/max(1e-9,float(np.linalg.norm(feature))))
                best = np.argsort(-similarity,kind="stable")[:POLICY["nearest_votes"]]
                label_id = int(self.labels[best[0]])
                gap = float(similarity[best[0]]-similarity[self.labels!=label_id].max())
                votes = [int(model.predict(feature[None])[1].ravel()[0]) for model in self.models]
                reading = {"rotation":degrees,"label":self.classes[label_id],
                           "similarity":float(similarity[best[0]]),"class_gap":gap,
                           "votes":[self.classes[v] for v in votes],
                           "references":[self.manifest["sources"][int(index)]["path"] for index in best]}
                result["readings"].append(reading)
                # A blank white face is also what the side of a concealed tile
                # looks like. Shape votes cannot supply independent identity.
                if reading["label"] in POLICY["excluded_labels"]:
                    return {**result,"reason":"blank_has_no_independent_glyph"}
                if (reading["similarity"] < POLICY["minimum_similarity"] or gap < POLICY["minimum_class_gap"]
                        or any(v!=label_id for v in votes) or np.any(self.labels[best]!=label_id)):
                    return {**result,"reason":"glyph_evidence_insufficient"}
        if len({r["label"] for r in result["readings"]}) != 1:
            return {**result,"reason":"glyph_rotations_disagree"}
        return {**result,"accepted":True,"label":result["readings"][0]["label"],"reason":None}
