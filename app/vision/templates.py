from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.capture.frames import load_image
from app.state.events import TILES
from .config import VisionError, write_json
from .faces import FACE_PARAMETERS, HIGHLIGHT_FACE_PARAMETERS, FaceGallery, tile_face, unrotate_quad
from .text import parse_text

FEATURE_VERSION = "bgr-body-v2-and-rectified-colour-glyph-v3"
HIGHLIGHT_FEATURE_VERSION = "bgr-body-v2-and-rectified-colour-glyph-v4"
FACE_RECIPES = {FEATURE_VERSION:FACE_PARAMETERS,HIGHLIGHT_FEATURE_VERSION:HIGHLIGHT_FACE_PARAMETERS}


def rotate(pixels: np.ndarray, degrees: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(pixels, (degrees // 90) % 4))


def canonical(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        raise VisionError("模板裁切为空。")
    # BGR colour remains present: greyscale alone loses the three red fives.
    resized = cv2.resize(pixels, (40, 60), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Highlighting changes the background tint, not the identity of the tile.
    # Correct each BGR channel using bright background pixels, retaining the
    # coloured glyphs needed to distinguish ordinary and red fives.
    background = np.percentile(resized, 85, axis=(0,1)).astype(np.float32)
    normalized = np.clip(resized / np.maximum(background, 64), 0, 1)
    return normalized[3:-3, 3:-3]


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left - right) ** 2))
    a, b = left.reshape(-1), right.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    correlation = float(a @ b / denom) if denom > 1e-8 else 1 - mse * 5
    return float(np.clip(0.65 * correlation + 0.35 * (1 - min(1, mse * 6)), -1, 1))


class TemplateBank:
    def __init__(self, manifest: str | Path, classifier_manifest: str | Path | None = None):
        self.path = Path(manifest)
        if not self.path.is_file():
            raise VisionError(f"模板库缺失：{self.path}；请先标注并构建模板。")
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.entries = []
        self.feature_version = self.manifest.get("feature_version")
        if self.feature_version not in (None,"bgr-background-normalized-v2",*FACE_RECIPES):
            raise VisionError("模板特征版本不兼容，停止加载。")
        self.face_parameters = FACE_RECIPES.get(self.feature_version)
        if self.face_parameters is not None and self.manifest.get("face_parameters") != self.face_parameters:
            raise VisionError("模板牌面特征参数不兼容，停止加载；请从 train 标注重新构建。")
        self.face_entries = []
        for entry in self.manifest.get("templates", []):
            path = self.path.parent / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise VisionError(f"模板校验失败：{path}")
            pixels = load_image(path)
            self.entries.append((entry, canonical(pixels)))
            if self.face_parameters is not None and entry.get("kind","tile") in {"tile","back"}:
                face = tile_face(pixels,self.face_parameters)
                if face is not None:
                    self.face_entries.append((entry,face.feature))
        if not self.entries:
            raise VisionError("模板库为空。")
        self.face_classes = {entry["label"] for entry,_ in self.face_entries if entry.get("kind","tile")=="tile"}
        self.face_gallery = FaceGallery(np.stack([feature for _,feature in self.face_entries])) if self.face_entries else None
        self.body_galleries = {}
        self.verifier = None
        self.training_games = set(self.manifest.get("source_games",[]))
        if classifier_manifest is not None:
            from .svm import SvmVerifier
            if self.face_parameters is None or self.face_classes != set(TILES):
                raise VisionError("视觉 SVM 需与完整牌面模板共同使用。")
            self.verifier = SvmVerifier(classifier_manifest)
            if self.verifier.feature_parameters["face"] != self.face_parameters:
                raise VisionError("视觉 SVM 与模板的牌面参数不一致，停止加载。")
            self.training_games.update(self.verifier.source_games)
        self.classifier_manifest_sha256 = self.verifier.sha256 if self.verifier else None

    def classify(self, pixels: np.ndarray, role: str, threshold: float, margin: float, rotations=(0,), *, kinds=("tile","back")) -> dict:
        body = self._classify_body(pixels,role,threshold,margin,rotations,kinds=kinds)
        if (not set(kinds)&{"tile","back"} or not self.face_entries or
                role == "choices" and body.get("reason") == "missing_role_templates"):
            return body
        face = self._classify_face(pixels,role,threshold,margin,rotations,kinds)
        if body["label"] == "P" and (face is None or not face.get("blank_face")):
            body = {**body,"label":"?","reason":"blank_face_unconfirmed"}
        if face is None:
            return body
        if body["label"] != "?" and face["label"] != "?" and body["label"] != face["label"]:
            return {**body,"label":"?","reason":"conflicting_tile_features","face_evidence":face}
        if body["label"] != "?":
            return {**body,"feature_scope":"region_body","face_evidence":face}
        if face["label"] != "?":
            return {**face,"body_evidence":body}
        if face["score"] > (body.get("score") if body.get("score") is not None else -1):
            proposed = {**face,"body_evidence":body}
        else:
            proposed = {**body,"face_evidence":face,"blank_face":face.get("blank_face",False)}
        if self.verifier is not None and proposed.get("best_label") in TILES:
            verifier = self.verifier.verify(rotate(pixels,proposed.get("face_rotation",0)),proposed)
            proposed["classifier_evidence"] = verifier
            if verifier["accepted"]:
                proposed.update(label=proposed["best_label"],reason=None,
                                acceptance_policy="unanimous_train_svm_and_existing_template_proposal")
        return proposed

    def _classify_body(self, pixels, role, threshold, margin, rotations, *, kinds):
        key = (role, tuple(kinds))
        if key not in self.body_galleries:
            selected = [(entry, template) for entry,template in self.entries
                        if role in entry["roles"] and entry.get("kind","tile") in kinds]
            self.body_galleries[key] = (selected, FaceGallery(np.stack([value for _,value in selected]))
                                       if selected else None)
        entries, gallery = self.body_galleries[key]
        if gallery is None:
            return {"label": "?", "score": None, "margin": None, "reason": "missing_role_templates"}
        scores = defaultdict(lambda: (-1.0, None, 0))
        for degrees in rotations:
            face = canonical(rotate(pixels, degrees))
            values = gallery.scores(face)
            for (entry, _), value in zip(entries, values):
                score = float(value)
                label = entry["label"]
                if score > scores[label][0]:
                    scores[label] = (score, entry["id"], degrees)
        ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
        if not ranked:
            return {"label": "?", "score": None, "margin": None, "reason": "missing_role_templates"}
        label, (score, template_id, degrees) = ranked[0]
        difference = score - ranked[1][1][0] if len(ranked) > 1 else 0.0
        accepted = score >= threshold and difference >= margin
        return {"label": label if accepted else "?", "best_label": label, "score": score,
                "margin": difference, "template_id": template_id, "face_rotation": degrees,
                "reason": None if accepted else "low_score_or_ambiguous",
                "alternatives": [{"label": label, "score": value[0]} for label, value in ranked[:3]]}

    def _classify_face(self,pixels,role,threshold,margin,rotations,kinds):
        shared = self.face_classes == set(TILES)
        indices = [index for index,(entry,_) in enumerate(self.face_entries)
                   if entry.get("kind","tile") in kinds and
                   (role in entry["roles"] or shared and entry.get("kind","tile")=="tile")]
        if not indices:
            return None
        scores = {}
        for degrees in rotations:
            face = tile_face(rotate(pixels,degrees),self.face_parameters)
            if face is None:
                continue
            values = self.face_gallery.scores(face.feature)
            for index in indices:
                entry = self.face_entries[index][0]
                label = entry["label"]
                # Faint visible marks cannot become a blank white dragon merely
                # because colour quantisation removed them from the glyph mask.
                if label == "P" and not face.blank:
                    continue
                score = float(values[index])
                if label not in scores or score > scores[label][0]:
                    scores[label] = (score,entry["id"],degrees,face)
        if not scores:
            return None
        ranked = sorted(scores.items(),key=lambda item:item[1][0],reverse=True)
        label,(score,identifier,degrees,face) = ranked[0]
        difference = score-ranked[1][1][0] if len(ranked)>1 else 0.0
        accepted = score>=threshold and difference>=margin
        return {"label":label if accepted else "?","best_label":label,"score":score,"margin":difference,
                "template_id":identifier,"face_rotation":degrees,"blank_face":face.blank,
                "face_ink_fraction":face.ink_fraction,"face_area_fraction":face.area_fraction,
                "face_quad_local":unrotate_quad(face.quad,pixels.shape[1],pixels.shape[0],degrees),
                "feature_scope":"shared_rectified_glyph" if shared else "region_rectified_glyph",
                "reason":None if accepted else "low_face_score_or_ambiguous",
                "alternatives":[{"label":key,"score":value[0]} for key,value in ranked[:3]]}

    def text_match(self, pixels: np.ndarray, role: str, field: str, threshold: float, margin: float) -> dict:
        matched = self.classify(pixels, role, threshold, margin, kinds=("text",))
        value = parse_text(matched["label"], field) if matched["label"] != "?" else None
        return {**matched,"raw_text":matched.get("best_label", ""),"value":value,
                "score_kind":"text_template_similarity_not_probability"}

    def button_matches(self, pixels: np.ndarray, threshold: float):
        found = []
        for entry, _ in self.entries:
            if entry.get("kind") != "button":
                continue
            template = load_image(self.path.parent / entry["file"])
            if template.shape[0] > pixels.shape[0] or template.shape[1] > pixels.shape[1]:
                continue
            values = cv2.matchTemplate(pixels, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(values)
            if score >= threshold:
                found.append({"label": entry["label"], "state": entry.get("state", "unknown"),
                              "score": score, "bbox": [x, y, template.shape[1], template.shape[0]],
                              "template_id": entry["id"]})
        return found


def build_templates(annotation_paths: list[Path], output: str | Path, *, feature_version: str = FEATURE_VERSION):
    if feature_version not in FACE_RECIPES:
        raise VisionError("不能构建未知牌面特征版本。")
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        _snapshot_manifest(output)
    entries, source_games = [], set()
    for annotation_path in annotation_paths:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        if annotation.get("split") != "train":
            raise VisionError("模板只能使用 train 对局，禁止从 test/debug 验收图取模板。")
        if not annotation.get("game_id") or not annotation.get("reviewed"):
            raise VisionError("模板来源必须有对局编号并经人工核对。")
        source_games.add(annotation["game_id"])
        image_path = Path(annotation["image"])
        if not image_path.is_absolute():
            image_path = annotation_path.parent / image_path
        pixels = load_image(image_path)
        for index, target in enumerate(annotation["targets"]):
            kind = target.get("kind", "tile")
            if target["label"] == "?" or kind not in ("tile", "button", "back", "text"):
                continue
            if kind == "tile" and target["label"] not in TILES:
                raise VisionError(f"无效牌种：{target['label']}")
            if kind == "text":
                field = target.get("field", target["region"].split("_")[0])
                if target.get("value") is None or parse_text(target["label"],field) != target["value"]:
                    raise VisionError("文字模板标签与结构化值不符。")
            x, y, w, h = target["bbox"]
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > pixels.shape[1] or y+h > pixels.shape[0]:
                raise VisionError("标注牌框越界。")
            face = rotate(pixels[y:y+h, x:x+w], target.get("rotation", 0))
            identifier = hashlib.sha256((str(image_path.resolve()) + str(index) + target["label"]).encode()).hexdigest()[:16]
            ok,encoded = cv2.imencode(".png",face)
            if not ok:
                raise VisionError("模板 PNG 编码失败。")
            payload = encoded.tobytes()
            digest = hashlib.sha256(payload).hexdigest()
            file = f"{identifier}-{digest[:16]}.png"
            template_path = output.parent/file
            if template_path.exists():
                if hashlib.sha256(template_path.read_bytes()).hexdigest() != digest:
                    raise VisionError("已有模板文件校验失败，不能覆盖。")
            else:
                with template_path.open("xb") as stream:
                    stream.write(payload)
            entries.append({"id": identifier, "file": file, "sha256": digest,
                            "label": target["label"], "kind": kind, "roles": target.get("roles", [target["region"]]),
                            "state": target.get("state", "unknown"), "source_game": annotation["game_id"],
                            "source_annotation": str(annotation_path.resolve()), "source_target": index})
    write_json(output, {"schema_version": 1,"feature_version":feature_version,"face_parameters":FACE_RECIPES[feature_version],
                        "source_games": sorted(source_games), "templates": entries,
                        "score_kind": "template_similarity_not_probability"})
    _snapshot_manifest(output)
    return {"templates": len(entries), "source_games": sorted(source_games)}


def _snapshot_manifest(path: Path) -> Path:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    snapshot = path.with_name(f"{path.stem}-{digest}.json")
    if snapshot.exists():
        if snapshot.read_bytes() != payload:
            raise VisionError("已有模板清单副本校验失败，不能覆盖。")
    else:
        with snapshot.open("xb") as stream:
            stream.write(payload)
    return snapshot
