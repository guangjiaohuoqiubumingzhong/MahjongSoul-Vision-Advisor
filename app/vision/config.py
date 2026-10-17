from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


class VisionError(ValueError):
    pass


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def validate_config(config: dict):
    polygons=config.get("river_polygons")
    if polygons is not None:
        if (not isinstance(polygons,dict) or not isinstance(polygons.get("manifest"),str) or not polygons["manifest"]
                or not isinstance(polygons.get("sha256"),str) or len(polygons["sha256"])!=64
                or any(c not in "0123456789abcdef" for c in polygons["sha256"])
                or (config.get("tile_classifier") or {}).get("name")!="pjura"):
            raise VisionError("固定牌河多边形配置无效。")
    for auxiliary in ("action_timer", "self_wind"):
        value = config.get(auxiliary)
        if value is not None:
            if not isinstance(value,dict):
                raise VisionError("行动计时器/自风配置无效。")
            pixel_rect(value.get("rect",[]), *config["client_size"])
            threshold = value.get("threshold",.95)
            if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
                raise VisionError("行动计时器/自风识别阈值无效。")
            if auxiliary == "self_wind" and not value.get("manifest"):
                raise VisionError("缺少自风模板。")
    classifier = config.get("tile_classifier")
    if classifier is not None:
        if (not isinstance(classifier, dict) or classifier.get("name") != "pjura"
                or not isinstance(classifier.get("manifest"), str) or not classifier["manifest"]):
            raise VisionError("牌面识别器配置无效。")
        if classifier.get("provider", "CUDAExecutionProvider") not in {"CUDAExecutionProvider", "CPUExecutionProvider"}:
            raise VisionError("pjura 推理设备无效。")
        for key, default in (("confidence", .9), ("margin", .04)):
            value = classifier.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise VisionError("pjura 分类阈值应在 0–1 范围。")
        from .acceptance import dominant_candidate_config
        try:
            dominant_candidate_config(classifier.get("dominant_candidate"))
        except ValueError as exc:
            raise VisionError(str(exc)) from exc
        batch_size = classifier.get("batch_size", 16)
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise VisionError("pjura 批量大小应在 1–64 之间。")
        if classifier.get("rectification", "both") not in {"both", "rectified", "off"}:
            raise VisionError("pjura 透视校正模式无效。")
        skip = classifier.get("rectification_skip_roles", ["hand", "draw"])
        if not isinstance(skip, list) or not all(isinstance(value, str) for value in skip):
            raise VisionError("pjura 透视校正跳过区域应为名称列表。")
        sideways = classifier.get("sideways_verifier_manifest")
        if sideways is not None and (not isinstance(sideways,str) or not sideways):
            raise VisionError("横牌字形模型配置无效。")
    if not isinstance(config.get("version"), str) or not config["version"]:
        raise VisionError("区域配置缺少版本。")
    size = config.get("client_size", [])
    if len(size) != 2 or any(type(v) is not int or v < 64 for v in size):
        raise VisionError("客户区尺寸无效。")
    seats = config.get("seat_mapping", {})
    if sorted(seats.values()) != [0, 1, 2, 3] or set(seats) != {"self", "right", "top", "left"}:
        raise VisionError("座位映射必须是一一对应的四家编号。")
    if not config.get("regions"):
        raise VisionError("区域配置为空。")
    for name, region in config["regions"].items():
        if 'split_touching_meld_faces' in region and (region['kind'] != 'meld' or
                type(region['split_touching_meld_faces']) is not bool):
            raise VisionError("副露牌面分离配置无效。")
        if "kan_back_colours" in region and (region["kind"] != "meld" or
                region["kan_back_colours"] != ["orange"]):
            raise VisionError("杠牌背颜色配置无效。")
        for polygon in region.get("concealed_hand_exclusions",[]):
            if (region.get("kind") != "meld" or len(polygon) < 3 or any(len(p) != 2 or any(
                    type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in polygon)):
                raise VisionError("对手暗手排除区域无效。")
        pixel_rect(region["rect"], *size)
        if region.get("rotation", 0) not in (0, 90, 180, 270):
            raise VisionError(f"{name} 的方向必须是直角。")
        if region["kind"] not in {"hand", "draw", "river", "dora", "meld", "text", "buttons", "choices"}:
            raise VisionError(f"{name} 的区域类型无效。")
        if "player" in region and region["player"] not in range(4):
            raise VisionError(f"{name} 的玩家编号无效。")
        if "min_fill" in region and not 0 < region["min_fill"] <= 1:
            raise VisionError(f"{name} 的牌体填充阈值无效。")
        if "body_saturation_max" in region and not 0 <= region["body_saturation_max"] <= 255:
            raise VisionError(f"{name} 的牌体饱和度阈值无效。")
        if "body_value_min" in region and not 0 <= region["body_value_min"] <= 255:
            raise VisionError(f"{name} 的牌体亮度阈值无效。")
        for mapping_name in ("source_by_called_position","chi_source_by_called_position","pon_source_by_called_position","kan_source_by_called_position"):
            mapping = region.get(mapping_name, {})
            if (not isinstance(mapping,dict) or any(key not in {"0","1","2","3"} or
                    type(value) is not int or value not in range(4) or value == region.get("player")
                    for key,value in mapping.items())):
                raise VisionError(f"{name} 的副露来源映射无效。")
            if mapping_name in {"chi_source_by_called_position","pon_source_by_called_position"} and "3" in mapping:
                raise VisionError(f"{name} 的吃碰来源位置必须在三张牌内。")
        for band in region.get("body_presence_bands", []):
            if len(band) != 2 or not 0 <= band[0] < band[1] <= 1:
                raise VisionError(f"{name} 的牌体检测带无效。")
        if "hand_layout" in region:
            layout = region["hand_layout"]
            draw = config["regions"].get(layout.get("draw_region"), {})
            rw,rh = pixel_rect(region["rect"], *size)[2:]
            slots = region.get("slots", [])
            gap = layout.get("draw_gap")
            counts = layout.get("concealed_counts", [1,4,7,10,13])
            if (region["kind"] != "hand" or region.get("rotation",0) != 0 or
                    draw.get("kind") != "draw" or draw.get("player") != region.get("player") or
                    type(gap) is not int or gap <= 6 or not slots or
                    any(type(count) is not int or not 1 <= count <= len(slots) for count in counts) or
                    not 0 <= layout.get("gap_fill_max",.1) <= 1):
                raise VisionError(f"{name} 的手牌/摸牌布局配置无效。")
            for bx,by,bw,bh in slots:
                if min(bx,by) < 0 or min(bw,bh) <= 0 or bx+2*bw+gap > rw or by+bh > rh:
                    raise VisionError(f"{name} 的摸牌槽越出校准区域。")
        if "grid" in region:
            grid = region["grid"]
            origin = grid.get("origin", [])
            if len(origin) != 2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in origin):
                raise VisionError(f"{name} 的牌河网格原点无效。")
            basis = grid.get("basis")
            if basis is not None:
                if (len(basis) != 2 or any(len(v) != 2 for v in basis) or
                        not all(isinstance(v,(int,float)) and math.isfinite(v) for row in basis for v in row) or
                        abs(basis[0][0]*basis[1][1]-basis[0][1]*basis[1][0]) < 1e-6):
                    raise VisionError(f"{name} 的牌河网格基向量无效。")
            else:
                pitch = grid.get("pitch", [])
                if len(pitch) != 2 or not all(isinstance(v,(int,float)) and math.isfinite(v) and v > 0 for v in pitch):
                    raise VisionError(f"{name} 的牌河网格间距无效。")
            if grid.get("anchor", "top_left") not in {"top_left", "center"}:
                raise VisionError(f"{name} 的牌河网格锚点无效。")
    for threshold in config.get("thresholds", {}).values():
        if not 0 <= threshold <= 1:
            raise VisionError("识别阈值应在 0–1 范围。")


# A client larger than the calibrated one but with the same aspect ratio (e.g.
# 1920x1080 for a 1366x768 calibration) is analysed at the calibrated size for
# layout, detection, OCR and templates; only the tile classifier reads the
# native pixels. Mahjong Soul keeps a fixed 16:9 layout that scales uniformly.
ASPECT_TOLERANCE = .01


def scaled_client(width: int, height: int, config: dict, *, allow_smaller: bool = False) -> float | None:
    """Native/calibrated scale for a compatible client size, else ``None``."""
    reference_width, reference_height = config["client_size"]
    if min(width,height) <= 0:
        return None
    if [width, height] == [reference_width, reference_height]:
        return 1.0
    if not allow_smaller and (width < reference_width or height < reference_height):
        return None
    if abs((width / height) / (reference_width / reference_height) - 1) > ASPECT_TOLERANCE:
        return None
    return width / reference_width


def to_reference(pixels: np.ndarray, config: dict):
    """Return ``(calibrated_pixels, native_pixels_or_None, scale)``."""
    height, width = pixels.shape[:2]
    scale = scaled_client(width, height, config, allow_smaller=True)
    if scale is None:
        raise VisionError(f"截图尺寸 {width}×{height} 与窗口配置 {config['client_size'][0]}×"
                          f"{config['client_size'][1]} 不同比例，暂停识别。")
    if scale == 1.0:
        return pixels, None, 1.0
    interpolation = cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR
    reference = cv2.resize(pixels, tuple(config["client_size"]), interpolation=interpolation)
    return reference, pixels, scale


def native_box(box, scale: float, native_shape) -> list[int]:
    """Scale a calibrated-frame ``[x, y, w, h]`` to native pixels, clipped."""
    x, y, w, h = box
    height, width = native_shape[:2]
    x1, y1 = max(0, int(round(x * scale))), max(0, int(round(y * scale)))
    x2, y2 = min(width, int(round((x + w) * scale))), min(height, int(round((y + h) * scale)))
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def pixel_rect(rect, width: int, height: int) -> tuple[int, int, int, int]:
    if len(rect) != 4:
        raise VisionError("区域必须包含 x、y、宽、高。")
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
        raise VisionError(f"区域越出客户区：{rect}")
    x1, y1 = round(x * width), round(y * height)
    x2, y2 = min(width, round((x + w) * width)), min(height, round((y + h) * height))
    if x2 <= x1 or y2 <= y1:
        raise VisionError("区域缩放后为空。")
    return x1, y1, x2 - x1, y2 - y1


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
