"""Pinned pjura ViT tile recognition using the existing ONNX Runtime package."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image

from app.state.events import BASE, base
from .config import VisionError
from .acceptance import candidate_acceptance, dominant_candidate_config
from .rectify import (RECTIFY_PARAMETERS, face_tint, five_colour, five_verdict, min_area_quad, quad_ratio,
                      quad_skew, top_face_outline, warp_top_face)
from .templates import rotate

UPSTREAM_REPO = "pjura/mahjong_soul_vision"
UPSTREAM_REVISION = "ea01ca0bcb9be0fc10a5fa09ee8356770c25adc5"
UPSTREAM_WEIGHTS_SHA256 = "a3cc566c8011b4496239a5228de3985e37ba2c2dfceefd52f67bb34c5f7e49c0"
FP32_MODEL_SHA256 = "f34e4537a7b9ff41ea12fdb4843cf0ec8a6109def22ca3eabb51045b492fc75c"
LABELS = {f"{number}{suit}": f"{number}{mapped}"
          for number in range(1, 10) for suit, mapped in (("n", "m"), ("p", "p"), ("b", "s"))}
LABELS.update(ew="E", sw="S", ww="W", nw="N", wd="P", gd="F", rd="C")
PREPROCESSOR = {"do_normalize": True, "do_rescale": True, "do_resize": True,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "image_processor_type": "ViTImageProcessor", "resample": 2,
                "rescale_factor": 1 / 255, "size": {"height": 224, "width": 224}}
# The upstream normalization has only 256 possible inputs. Preserve its exact
# float64 -> float32 rounding once, instead of allocating large temporaries for
# every rotated/rectified view of every tile.
_NORMALIZED_UINT8 = (np.arange(256, dtype=np.float64) * (1 / 255)).astype(np.float32)
_NORMALIZED_UINT8 = (_NORMALIZED_UINT8 - np.float32(.5)) / np.float32(.5)


# How tile crops reach the network.
#   "both"      - every rotation is tried on the original crop and on the
#                 perspective-corrected top face; the best view wins, and two
#                 confident views that disagree make the tile unknown.
#   "rectified" - only the corrected face when it can be fitted, else original.
#   "off"       - original crops only (behaviour before the correction).
RECTIFICATION_MODES = ("both", "rectified", "off")
DEFAULT_RECTIFICATION = "both"
# A rejected candidate this unsure, without any tile-face outline, is table art.
BACKGROUND_SCORE_MAX = .5
# Frontal hand/draw tiles are already read reliably and carry no perspective.
DEFAULT_RECTIFICATION_SKIP_ROLES = ("hand", "draw")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_acceleration(manifest_path,manifest,provider):
    """Derived models need matching source, parity evidence and a CUDA backend."""
    optimization=manifest.get('optimization')
    if optimization is None:return None
    folder=Path(manifest_path).resolve().parent
    if (not isinstance(optimization,dict) or optimization.get('method')!='ort-vit-fused-fp16'
            or optimization.get('source_model_sha256')!=FP32_MODEL_SHA256
            or optimization.get('fp32_guard_band')!=.005):
        raise VisionError('认牌加速模型的来源或精度复核策略无效。')
    if provider!='CUDAExecutionProvider':
        raise VisionError('认牌加速模型需要 CUDA；CPU 调试请使用原始 FP32 配置。')
    evidence=(folder/optimization.get('validation_file','')).resolve()
    source=(folder/optimization.get('source_manifest','')).resolve()
    if (not evidence.is_relative_to(folder) or not evidence.is_file()
            or not source.is_relative_to(folder.parent) or not source.is_file()
            or file_sha256(evidence)!=optimization.get('validation_sha256')
            or file_sha256(source)!=optimization.get('source_manifest_sha256')):
        raise VisionError('认牌加速模型的来源或验证记录校验失败。')
    checked=json.loads(evidence.read_text(encoding='utf-8'))
    original=json.loads(source.read_text(encoding='utf-8'))
    guard=checked.get('precision_guard') or {}
    if (checked.get('passed') is not True or checked.get('candidate_sha256')!=manifest['model_sha256']
            or checked.get('baseline_sha256')!=FP32_MODEL_SHA256
            or guard.get('enabled') is not True or guard.get('band')!=.005
            or original.get('optimization') is not None or original.get('model_sha256')!=FP32_MODEL_SHA256):
        raise VisionError('认牌加速模型尚未通过对应的原精度对照验证。')
    return source


# Legacy silhouette gate, kept for the earlier diagnosis scripts. The classifier
# no longer uses it: the silhouette includes the tile's side face, so the
# top-face correction below replaced it.
# A tile merely turned inside its detector box is still a rectangle, so it needs
# no correction; only a real taper means the axis-aligned resize is squashing the
# glyph. Reviewed-tile re-projections stayed entirely correct while the taper was
# mild and collapsed once it grew, and the taper measured on captured frames is
# small for upright tiles (about 0.02) and large only for tiles lying away from
# the camera. Outlines that do not describe the crop are never corrected.
RECTIFY_SKEW_TOLERANCE = .12
# A tile turned inside its detector box always leaves background in the box's
# corners, and real detections were measured to keep about 0.44 of the box
# corners filled. The floor only rejects an outline that does not describe the
# crop at all.
RECTIFY_MINIMUM_AREA_FRACTION = .40


def rail_aspect(quad):
    """Long-edge to short-edge ratio of a fitted outline, whatever its turn."""
    return quad_ratio(quad)


def outline_skew(quad):
    """Relative taper between opposite edges of a fitted outline."""
    return quad_skew(quad)


def outline_foreshortening(quad, reference=None):
    """Relative departure of a fitted outline from the tile's true proportions."""
    if reference is None:
        width, height = RECTIFY_PARAMETERS["warp_size"]
        reference = height / width
    aspect = quad_ratio(quad)
    if aspect <= 0:
        return 1.0
    return abs(aspect - reference) / reference


def rectification_needed(patch, parameters=None, *, skew_tolerance=None):
    """Return the fitted outline when this crop demonstrably lies at an angle.

    ``None`` means the crop is already an undistorted outline, or its outline
    could not be established, and the caller must keep the original pixels
    untouched. ``skew_tolerance`` overrides the module default; passing 0
    applies the correction to every crop that has a usable outline.
    """
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    geometry = min_area_quad(patch, parameters)
    if geometry is None:
        return None
    if geometry.area_fraction < max(RECTIFY_MINIMUM_AREA_FRACTION, parameters["minimum_silhouette_area_fraction"]):
        return None
    tolerance = RECTIFY_SKEW_TOLERANCE if skew_tolerance is None else skew_tolerance
    return geometry if quad_skew(geometry.quad) >= tolerance else None


def preprocess(pixels):
    """Match upstream ViTImageProcessor: BGR -> RGB, PIL bilinear, [0,1], normalize."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.size == 0:
        raise VisionError("pjura 需要非空 uint8 BGR 牌面裁切。")
    image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    rgb = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(_NORMALIZED_UINT8[rgb].transpose(2, 0, 1))


class PjuraModel:
    def __init__(self, manifest_path, *, provider="CUDAExecutionProvider", batch_size=16, allow_unvalidated=False,
                 confidence=.9, margin=.04, dominant_candidate=None):
        import onnxruntime as ort
        self.path = Path(manifest_path)
        if not self.path.is_file():
            raise VisionError("pjura 模型缺失；请先下载并执行 scripts/export_pjura.py。")
        self.manifest_sha256 = file_sha256(self.path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        manifest = self.manifest
        if (manifest.get("repo") != UPSTREAM_REPO or manifest.get("revision") != UPSTREAM_REVISION
                or manifest.get("files", {}).get("pytorch_model.bin") != UPSTREAM_WEIGHTS_SHA256):
            raise VisionError("pjura 模型来源不符合固定版本。")
        if not allow_unvalidated and (not isinstance(manifest.get("validation"), dict)
                                     or manifest["validation"].get("status") != "passed"):
            raise VisionError("pjura ONNX 尚未通过原模型一致性验证。")
        if manifest.get("preprocessor") != PREPROCESSOR:
            raise VisionError("pjura 预处理参数与上游不一致。")
        mapping = manifest.get("id2label", {})
        if mapping != {str(i): label for i, label in enumerate(sorted(LABELS))}:
            raise VisionError("pjura 分类标签顺序必须与固定上游模型一致。")
        self.labels = [LABELS[mapping[str(i)]] for i in range(34)]
        if set(self.labels) != set(BASE):
            raise VisionError("pjura 牌种映射不完整。")
        model = self.path.parent / manifest["model"]
        if not model.resolve().is_relative_to(self.path.parent.resolve()):
            raise VisionError("pjura 模型路径越界。")
        if file_sha256(model) != manifest["model_sha256"]:
            raise VisionError("pjura ONNX 模型 SHA256 校验失败。")
        self._reference_manifest=verify_acceleration(self.path,manifest,provider)
        self._reference_model=None
        self.confidence,self.margin=confidence,margin
        self.dominant_candidate=dominant_candidate_config(dominant_candidate)
        self.precision_rechecks=0
        if provider not in {"CUDAExecutionProvider", "CPUExecutionProvider"}:
            raise VisionError("pjura 推理设备无效。")
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise VisionError("pjura 批量大小应在 1–64 之间。")
        self.batch_size = batch_size
        self.lock = threading.RLock()
        self.dll_handles = []
        if provider == "CUDAExecutionProvider":
            if os.name == "nt":
                directories = sorted((Path(ort.__file__).parent.parent / "nvidia").glob("*/bin"))
                self.dll_handles = [os.add_dll_directory(str(directory)) for directory in directories]
                os.environ["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
            ort.preload_dlls(directory="")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        providers = [(provider, {"device_id": "0", "use_tf32": "0"})] if provider.startswith("CUDA") else [provider]
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
        self.session.disable_fallback()
        if self.session.get_providers()[0] != provider:
            raise VisionError("pjura 未使用指定推理设备；停止而非静默替换模型。")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "pixel_values" or inputs[0].shape[1:] != [3, 224, 224]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1 or outputs[0].name != "logits"
                or outputs[0].shape[-1] != 34):
            raise VisionError("pjura ONNX 输入输出不兼容。")
        self.provider = provider
        self._warmed = False

    def warmup(self):
        with self.lock:
            if not self._warmed:
                self.predict_many([np.full((32, 24, 3), 127, np.uint8)]*self.batch_size)
            if self._reference_manifest is not None:self._reference().warmup()

    def _reference(self):
        if self._reference_model is None:
            self._reference_model=PjuraModel(self._reference_manifest,provider=self.provider,
                batch_size=self.batch_size,confidence=self.confidence,margin=self.margin,
                dominant_candidate=self.dominant_candidate)
        return self._reference_model

    def _precision_guard(self,crops,predictions):
        if self._reference_manifest is None:return predictions
        band=self.manifest['optimization']['fp32_guard_band']
        dominant=getattr(self,'dominant_candidate',None)
        def dominant_boundary(p):
            if dominant is None:return False
            ratio=dominant['minimum_ratio']
            second=max(0.,p['score']-p['margin'])
            return (abs(p['score']-dominant['minimum_confidence'])<=band
                    or abs(p['score']-ratio*second)<=band*(1+ratio))
        # .5/.3 also gate recovery of occluded tiles and partially read melds.
        indices=[i for i,p in enumerate(predictions)
                 if any(abs(p['score']-t)<=band for t in (self.confidence,.5))
                 or any(abs(p['margin']-t)<=band for t in (self.margin,.3))
                 or dominant_boundary(p)]
        if indices:
            checked=self._reference().predict_many([crops[i] for i in indices])
            for i,prediction in zip(indices,checked):
                predictions[i]={**prediction,'precision_recheck':'fp32_threshold_boundary'}
            self.precision_rechecks+=len(indices)
        return predictions

    def predict_many(self, crops):
        results = []
        with self.lock:
            for start in range(0, len(crops), self.batch_size):
                group=crops[start:start+self.batch_size]
                inputs = np.stack([preprocess(crop) for crop in group])
                logits = self.session.run(["logits"], {"pixel_values": inputs})[0]
                if logits.shape != (len(inputs), 34) or not np.isfinite(logits).all():
                    raise VisionError("pjura 推理返回无效分类结果。")
                probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                predictions=[]
                for row in probabilities:
                    ranked = np.argsort(-row, kind="stable")[:3]
                    top = [{"label": self.labels[int(i)], "score": float(row[i])} for i in ranked]
                    predictions.append({"label": top[0]["label"], "score": top[0]["score"],
                                    "margin": top[0]["score"] - top[1]["score"], "alternatives": top,
                                    "score_kind": "softmax_not_calibrated_probability"})
                results.extend(self._precision_guard(group,predictions))
            if crops:self._warmed = True
        return results


class PjuraBank:
    """Use ViT for tile identity; retain legacy text/buttons/backs and five colour evidence."""
    def __init__(self, bank, config, *, model=None):
        self.legacy = bank
        self.dominant_candidate = dominant_candidate_config(config.get("dominant_candidate"))
        self.model = model if model is not None else PjuraModel(
            config["manifest"], provider=config.get("provider", "CUDAExecutionProvider"),
            batch_size=config.get("batch_size", 16),confidence=config.get('confidence',.9),
            margin=config.get('margin',.04),dominant_candidate=self.dominant_candidate)
        self.threshold = config.get("confidence", 0.9)
        self.margin_threshold = config.get("margin", 0.04)
        self.rectification = config.get("rectification", DEFAULT_RECTIFICATION)
        if self.rectification not in RECTIFICATION_MODES:
            raise VisionError("pjura 透视校正模式无效。")
        self.rectification_skip_roles = tuple(config.get("rectification_skip_roles",
                                                         DEFAULT_RECTIFICATION_SKIP_ROLES))
        self.sideways_verifier = None
        if config.get("sideways_verifier_manifest"):
            from .sideways import SidewaysVerifier
            self.sideways_verifier = SidewaysVerifier(config["sideways_verifier_manifest"])
        self._cache = OrderedDict()
        self._classification_cache = OrderedDict()
        self.classification_cache_size = 1024
        self.lock = threading.RLock()
        self.tile_classifier_info = {"name": "pjura", "repo": UPSTREAM_REPO,
                                     "revision": UPSTREAM_REVISION,
                                     "manifest_sha256": self.model.manifest_sha256,
                                     "model_sha256": self.model.manifest["model_sha256"],
                                     "provider": self.model.provider, "base_classes": 34,
                                     "red_five_policy": "matching_legacy_colour_template_or_unknown",
                                     "orientation_policy": "matching_legacy_template_or_unconfirmed",
                                     "confidence": self.threshold, "margin": self.margin_threshold,
                                     "rectification": self.rectification,
                                     "rectification_skip_roles": list(self.rectification_skip_roles)}
        if self.dominant_candidate is not None:
            self.tile_classifier_info["dominant_candidate"]=deepcopy(self.dominant_candidate)
        if self.model.manifest.get('optimization'):
            self.tile_classifier_info['optimization']=deepcopy(self.model.manifest['optimization'])
        if self.sideways_verifier is not None:
            self.tile_classifier_info["sideways_verifier_sha256"] = self.sideways_verifier.sha256
            self.tile_classifier_info["orientation_policy"] = "matching_legacy_template_or_verified_sideways_geometry"

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def _predictions(self, crops):
        with self.lock:
            keys = [(crop.shape, hashlib.sha256(crop.tobytes()).digest()) for crop in crops]
            missing = {}
            for key, crop in zip(keys, crops):
                if key not in self._cache:
                    missing.setdefault(key, crop)
            predictions = self.model.predict_many(list(missing.values())) if missing else []
            for key, prediction in zip(missing, predictions):
                self._cache[key] = prediction
            results = [deepcopy(self._cache[key]) for key in keys]
            for key in keys:
                self._cache.move_to_end(key)
            while len(self._cache) > 512:
                self._cache.popitem(last=False)
            return results

    def classify(self, pixels, role, threshold, margin, rotations=(0,), *, kinds=("tile", "back")):
        return self.classify_many([pixels], role, threshold, margin, rotations, kinds=kinds)[0]

    def _views(self, crop, role, rotations):
        """Candidate network inputs for one crop: (view, degrees, pixels)."""
        geometry = None
        if self.rectification != "off" and role not in self.rectification_skip_roles:
            geometry = top_face_outline(crop)
        if (geometry is not None and geometry.aspect >= 1.6
                and crop.shape[1] >= 1.25*crop.shape[0] and any(d in (90,270) for d in rotations)):
            # A clearly landscape face is a sideways tile. Trying its stretched
            # portrait hypothesis can turn two horizontal bamboos into seven.
            rotations = tuple(d for d in rotations if d in (90,270))
        views = []
        if self.rectification != "rectified" or geometry is None:
            views += [("original", degrees, rotate(crop, degrees)) for degrees in rotations]
        if geometry is not None:
            views += [("top_face", degrees, warp_top_face(crop, geometry, degrees).pixels)
                      for degrees in rotations]
        return views, geometry

    def classify_many(self, crops, role, threshold, margin, rotations=(0,), *, kinds=("tile", "back"),
                      model_crops=None):
        """Reuse complete evidence only when every source pixel and option matches."""
        if model_crops is not None and len(model_crops) != len(crops):
            raise VisionError("高分辨率裁切数量与检测框不一致。")
        def pixel_key(pixels):
            return (pixels.shape, pixels.dtype.str, hashlib.sha256(pixels.tobytes()).digest())
        policy=(role,threshold,margin,tuple(rotations),tuple(kinds),self.threshold,
                self.margin_threshold,self.rectification,self.rectification_skip_roles,
                id(self.legacy),id(self.model),id(self.sideways_verifier),
                tuple(sorted(self.dominant_candidate.items())) if self.dominant_candidate is not None else None)
        keys=[(policy,pixel_key(crop),pixel_key(source))
              for crop,source in zip(crops,model_crops if model_crops is not None else crops)]
        with self.lock:
            missing={}
            for index,key in enumerate(keys):
                if key not in self._classification_cache:missing.setdefault(key,index)
            indices=list(missing.values())
            if indices:
                matches=self._classify_many_uncached([crops[i] for i in indices],role,threshold,margin,
                    rotations,kinds=kinds,model_crops=[model_crops[i] for i in indices] if model_crops is not None else None)
                for key,match in zip(missing,matches):self._classification_cache[key]=deepcopy(match)
            results=[deepcopy(self._classification_cache[key]) for key in keys]
            for key in keys:self._classification_cache.move_to_end(key)
            while len(self._classification_cache)>self.classification_cache_size:
                self._classification_cache.popitem(last=False)
            return results

    def _classify_many_uncached(self, crops, role, threshold, margin, rotations=(0,), *, kinds=("tile", "back"),
                                model_crops=None):
        # ``model_crops`` are the same tiles cut from a higher-resolution native
        # frame; only the network reads them.
        if model_crops is not None and len(model_crops) != len(crops):
            raise VisionError("高分辨率裁切数量与检测框不一致。")
        # The template bank keeps the original pixels: its whole-body and
        # rectified-glyph templates were captured from unrectified crops, and its
        # five-colour evidence separates a red five from an ordinary one by
        # matching exactly those templates. Orientation confirmation also comes
        # from that unchanged reading.
        legacy = [self.legacy.classify(crop, role, threshold, margin, rotations, kinds=kinds)
                  for crop in crops]
        indices = [i for i, old in enumerate(legacy) if old["label"] != "back" and "tile" in kinds]
        # A tile on the table is seen at an angle: its white top face is a
        # sheared/foreshortened quadrilateral and the box also holds the tile's
        # side. Resizing that box to 224x224 distorts the glyph, so the top face
        # is warped upright (per rotation hypothesis) before classification.
        sources = model_crops if model_crops is not None else crops
        plans = {index: self._views(sources[index], role, rotations) for index in indices}
        flat = [pixels for index in indices for _, _, pixels in plans[index][0]]
        predictions = self._predictions(flat) if flat else []
        results = list(legacy)
        offset = 0
        for index in indices:
            views, geometry = plans[index]
            choices = predictions[offset:offset+len(views)]
            offset += len(views)
            selected = max(range(len(choices)), key=lambda j: (choices[j]["score"], choices[j]["margin"]))
            prediction, old = choices[selected], legacy[index]
            view, degrees = views[selected][0], views[selected][1]
            evidence = {**prediction, "face_rotation": degrees, "view": view,
                        "input_size": [int(sources[index].shape[1]), int(sources[index].shape[0])],
                        "model_sha256": self.tile_classifier_info["model_sha256"]}
            best_by_view = {}
            for (name, turn, _), choice in zip(views, choices):
                current = best_by_view.get(name)
                if current is None or (choice["score"], choice["margin"]) > (current["score"], current["margin"]):
                    best_by_view[name] = {"label": choice["label"], "score": choice["score"],
                                          "margin": choice["margin"], "face_rotation": turn}
            evidence["views"] = best_by_view
            if geometry is not None:
                evidence["rectified_quad"] = geometry.quad.astype(float).tolist()
                evidence["rectification"] = "top_face_perspective"
                evidence["rectification_geometry"] = geometry.as_dict()
            source = sources[index]
            outline = geometry if geometry is not None else top_face_outline(source)
            tint = face_tint(source, outline)
            evidence["face_tint"] = tint
            evidence["face_fitted"] = outline is not None
            rule = candidate_acceptance(prediction,self.threshold,self.margin_threshold,self.dominant_candidate)
            # Keep the original cross-view conflict threshold. Dominance is a
            # comparison between classes of the selected input, not between
            # scores from different perspective/rotation hypotheses.
            confident = [value for value in best_by_view.values()
                         if candidate_acceptance(value,self.threshold,self.margin_threshold)]
            if self.dominant_candidate is not None:
                second=max(0.,prediction["score"]-prediction["margin"])
                evidence["class_acceptance"]={"rule":rule,"second_score":second,
                    "top1_to_second_ratio":prediction["score"]/second if second else None,
                    **self.dominant_candidate}
            label, reason = prediction["label"], None
            sideways = None
            if prediction["score"] < self.threshold and rule is None:
                label, reason = "?", "pjura_low_confidence"
            elif prediction["margin"] < self.margin_threshold:
                label, reason = "?", "pjura_ambiguous_classes"
            elif len({value["label"] for value in confident}) > 1:
                # Two confident readings of the same tile disagree; neither the
                # correction nor the original crop is trusted over the other.
                label, reason = "?", "pjura_rectification_disagrees"
            if (reason in (None,"pjura_low_confidence","pjura_ambiguous_classes")
                    and (prediction["score"] < self.threshold or prediction["margin"] < self.margin_threshold)
                    and self.sideways_verifier is not None and tint["neutral"]
                    and (role == "river" or role == "meld" or role.startswith(("river_","meld_")))
                    and geometry is not None and geometry.aspect >= 1.6
                    and source.shape[1] >= 1.25*source.shape[0] and {90,270}.issubset(rotations)):
                sideways = self.sideways_verifier.verify(source)
                evidence["sideways_glyph"] = sideways
                if sideways["accepted"]:
                    label,reason = sideways["label"],None
            if label != "?" and label.startswith("5"):
                colour = five_colour(source, outline)
                verdict = five_verdict(label, colour)
                evidence["five_colour"] = {**colour, "verdict": verdict}
                legacy_ok = old["label"] != "?" and base(old["label"]) == label
                legacy_red = legacy_ok and old["label"] != label
                if legacy_ok and verdict is not None and (verdict == "red") != legacy_red:
                    label, reason = "?", "pjura_five_colour_conflict"
                elif legacy_ok:
                    label = old["label"]
                    evidence["colour_evidence"] = deepcopy(old)
                elif verdict is not None:
                    # Template colour evidence was not available (angled or
                    # high-resolution face); the stroke colours decide.
                    label = label + "r" if verdict == "red" else label
                    evidence["colour_evidence"] = {"source": "stroke_red_share", **evidence["five_colour"]}
                else:
                    label, reason = "?", "pjura_five_colour_unresolved"
            background = label == "?" and (not tint["neutral"] or
                                           (outline is None and prediction["score"] < BACKGROUND_SCORE_MAX))
            # Geometry, backs and unknown components stay the responsibility of
            # the detector. A 34-class network cannot certify hidden information.
            # A blank white dragon has identical glyphs at every quarter turn.
            # Its identity can be certain while the template's winning turn is
            # arbitrary. Called-tile orientation must come from face geometry.
            orientation_confirmed = label not in ("?", "P") and old["label"] != "?" and base(old["label"]) == base(label)
            verified = bool(sideways and sideways["accepted"] and label!="?")
            score = min(r["similarity"] for r in sideways["readings"]) if verified else prediction["score"]
            gap = min(r["class_gap"] for r in sideways["readings"]) if verified else prediction["margin"]
            results[index] = {"label": label, "best_label": base(label) if verified else prediction["label"],
                              "score": score, "margin": gap,
                              "score_kind": "glyph_cosine_not_probability" if verified else prediction.get("score_kind","softmax_not_calibrated_probability"),
                              "alternatives": [{"label":base(label),"score":score}] if verified else prediction["alternatives"], "reason": reason,
                              "face_rotation": degrees if verified else old.get("face_rotation", 0) if orientation_confirmed else degrees,
                              "identity_rotation": degrees,
                              "orientation_confirmed": verified or orientation_confirmed,
                              "orientation_source": "sideways_face_geometry" if verified else "legacy_template" if orientation_confirmed else "unconfirmed",
                              "feature_scope": "pjura_vit_rgb",
                              "acceptance_policy": "sideways_glyph_unanimity_with_five_colour_evidence" if verified else
                                  "pjura_dominant_candidate_with_five_colour_evidence" if rule=="dominant_candidate" else
                                  "pjura_softmax_with_five_colour_evidence",
                              "pjura_evidence": evidence, "legacy_evidence": old,
                              "background_candidate": bool(background)}
        return results
