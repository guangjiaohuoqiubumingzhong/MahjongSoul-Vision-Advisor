from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from .window import (CaptureError, OccludedError, client_area, ensure_unoccluded, is_minimized,
                     print_window, restore_window, window_label)
from .graphics import GraphicsCapture


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    pixels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise CaptureError(f"无法读取图像：{path}")
    return pixels


def save_image(path: str | Path, pixels: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", pixels)
    if not ok:
        raise CaptureError("图像编码失败。")
    encoded.tofile(path)


def config_digest(config: dict) -> str:
    encoded = json.dumps(config,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_capture_config(directory: Path, config: dict) -> Path:
    directory.mkdir(parents=True,exist_ok=True)
    digest = config_digest(config)
    path = directory/f"capture-config-{digest}.json"
    if path.exists():
        try:
            if config_digest(json.loads(path.read_text(encoding="utf-8"))) != digest:
                raise CaptureError("采集配置副本已被修改。")
        except (ValueError,OSError) as exc:
            raise CaptureError(f"采集配置副本无效：{path}") from exc
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(path)
    return path


@dataclass
class Frame:
    pixels: np.ndarray
    metadata: dict
    configuration: dict | None = None

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        path = directory / f'{self.metadata["frame_id"]:08d}.png'
        if path.exists():
            raise CaptureError(f"帧文件已存在，不能覆盖原始证据：{path}")
        if self.configuration is not None:
            digest = config_digest(self.configuration)
            if self.metadata.get("region_config_sha256",digest) != digest:
                raise CaptureError("帧对应的采集配置发生变化，拒绝写入错误证据。")
            snapshot = save_capture_config(directory,self.configuration)
            self.metadata.update(region_config_sha256=digest,region_config_snapshot=str(snapshot.resolve()))
        save_image(path, self.pixels)
        self.metadata["image_path"] = str(path.resolve())
        path.with_suffix(".frame.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class WindowCapture:
    """Create in the capture thread; MSS owns thread-local handles."""
    def __init__(self, hwnd: int, config: dict, annotation_overlays: bool = False):
        from mss import mss
        self.hwnd, self.config = hwnd, deepcopy(config)
        self.config_sha256 = config_digest(self.config)
        self.annotation_overlays = annotation_overlays
        self.sct = mss()
        self.frame_id = 0
        try:
            self.initial = client_area(hwnd)
            self._validate_geometry(self.initial)
        except Exception:
            self.sct.close()
            raise

    def _validate_geometry(self, area):
        from app.vision.config import scaled_client
        size = self.config["client_size"]
        if scaled_client(area.width, area.height, self.config) is None:
            raise CaptureError(f"窗口尺寸不支持：需要 {size[0]}×{size[1]} 或同比例更大的客户区（如 1920×1080），"
                               f"实际 {area.width}×{area.height}。")
        if (area.width, area.height) != (self.initial.width, self.initial.height):
            raise CaptureError(f"窗口尺寸发生变化：{self.initial.width}×{self.initial.height} → {area.width}×{area.height}，请重新开始。")
        if self.config.get("dpi") not in (None, area.dpi):
            raise CaptureError("显示 DPI 变化，请重新校准。")
        # Every segment must lie on a real display, including multi-monitor gaps.
        monitors = self.sct.monitors[1:]
        xs = sorted({area.left, area.left+area.width, *[v for m in monitors for v in (m["left"], m["left"]+m["width"]) if area.left < v < area.left+area.width]})
        ys = sorted({area.top, area.top+area.height, *[v for m in monitors for v in (m["top"], m["top"]+m["height"]) if area.top < v < area.top+area.height]})
        for left, right in zip(xs, xs[1:]):
            for top, bottom in zip(ys, ys[1:]):
                if not any(m["left"] <= left and right <= m["left"]+m["width"] and m["top"] <= top and bottom <= m["top"]+m["height"] for m in monitors):
                    raise CaptureError("客户区部分移出可采集显示器，暂停截图。")

    def grab(self) -> Frame:
        return self._grab()

    def grab_region(self, rect) -> Frame:
        """Return only a normalized client region to the timer/text detectors.

        On a larger same-aspect client the crop is resized to the calibrated
        size, so timer/text detectors keep their calibrated pixel parameters.
        WGC streams the window internally; only this crop reaches the caller.
        """
        from app.vision.config import pixel_rect
        native = pixel_rect(rect, self.initial.width, self.initial.height)
        frame = self._grab(native)
        _, _, width, height = pixel_rect(rect, *self.config['client_size'])
        if frame.pixels.shape[1] != width or frame.pixels.shape[0] != height:
            frame.pixels = cv2.resize(frame.pixels, (width, height), interpolation=cv2.INTER_AREA)
            frame.metadata["native_capture_rect"] = list(native)
        return frame

    # Live advice reads the selected HWND first, without raising it or changing
    # focus. Evidence capture keeps the strict screen-grab behaviour.
    window_content_fallback = False
    restore_minimized = False
    _graphics_capture = None
    RESTORE_WAIT_SECONDS = .4

    def _client(self):
        try:
            return client_area(self.hwnd)
        except CaptureError:
            # A minimised window may stop rendering; discard the old stream so
            # restoring it cannot reuse a cached countdown from the last turn.
            if is_minimized(self.hwnd) and self._graphics_capture is not None:
                self._graphics_capture.close()
                self._graphics_capture = None
            if not (self.restore_minimized and is_minimized(self.hwnd)):
                raise
            restore_window(self.hwnd)
            time.sleep(self.RESTORE_WAIT_SECONDS)
            return client_area(self.hwnd)

    def _grab_window_content(self, rect):
        """Read WGC first, with PrintWindow as a compatibility fallback."""
        started = time.perf_counter_ns()
        area = self._client()
        if area.pid != self.initial.pid:
            raise CaptureError("窗口句柄被其他进程复用，需重新选择窗口。")
        self._validate_geometry(area)
        details = {}
        source, api = "window_graphics_capture", "Windows.Graphics.Capture"
        try:
            if self._graphics_capture is None:
                self._graphics_capture = GraphicsCapture(self.hwnd)
            pixels, area, details = self._graphics_capture.read()
        except CaptureError as exc:
            if self._graphics_capture is not None:
                self._graphics_capture.close()
                self._graphics_capture = None
            pixels, area = print_window(self.hwnd)
            source, api = "window_content", "PrintWindow"
            details["graphics_capture_error"] = str(exc)
        if area.pid != self.initial.pid:
            raise CaptureError("窗口句柄被其他进程复用，需重新选择窗口。")
        self._validate_geometry(area)
        if client_area(self.hwnd) != area:
            raise CaptureError("采集期间窗口位置/尺寸变化，丢弃本帧。")
        x,y,w,h = rect if rect is not None else (0,0,area.width,area.height)
        pixels = np.ascontiguousarray(pixels[y:y+h, x:x+w])
        self.frame_id += 1
        frame = Frame(pixels, {"frame_id": self.frame_id, "captured_at": datetime.now(timezone.utc).isoformat(),
                              "capture_monotonic_ns": started, "capture_clock": "perf_counter_ns",
                              "capture_ms": (time.perf_counter_ns() - started) / 1e6,
                              "client": area.asdict(), "client_size": [area.width, area.height],
                              "calibrated_size": list(self.config["client_size"]),
                              "region_config_version": self.config["version"],
                              "region_config_sha256": self.config_sha256, "source": source,
                              "capture_api": api, "keyed_overlays": [], "capture_quality": "window_content",
                              **details},
                      self.config)
        if rect is not None:
            frame.metadata.update(source=source+"_region", capture_rect=[x,y,w,h])
        return frame

    def _grab(self, rect=None) -> Frame:
        if not self.window_content_fallback:
            return self._grab_screen(rect)
        try:
            return self._grab_window_content(rect)
        except CaptureError as content_error:
            # Screen pixels are usable only when unobscured. Never raise the
            # game to force this fallback; leave the user's foreground alone.
            try:
                frame = self._grab_screen(rect)
            except OccludedError as exc:
                raise OccludedError(f"{exc}窗口内容采集失败：{content_error}") from exc
            frame.metadata["window_content_error"] = str(content_error)
            return frame

    def _grab_screen(self, rect=None) -> Frame:
        """Screen grab; raises OccludedError when anything covers the client."""
        before = self._client()
        if before.pid != self.initial.pid:
            raise CaptureError("窗口句柄被其他进程复用，需重新选择窗口。")
        self._validate_geometry(before)
        overlays = ensure_unoccluded(before)
        if overlays and not self.annotation_overlays:
            if self.window_content_fallback:
                raise OccludedError("检测到透明叠加窗口（" + "、".join(
                    window_label(item["hwnd"]) for item in overlays) + "）。")
            raise CaptureError("检测到透明叠加窗口；生产采集暂停。人工标注采集可显式保留带警告的原图。")
        started = time.perf_counter_ns()
        x,y,w,h = rect if rect is not None else (0,0,before.width,before.height)
        pixels = np.asarray(self.sct.grab({"left": before.left+x, "top": before.top+y,
                                          "width": w, "height": h}))[:, :, :3].copy()
        after = client_area(self.hwnd)
        if before != after:
            raise CaptureError("采集期间窗口位置/尺寸变化，丢弃本帧。")
        later_overlays = ensure_unoccluded(after)
        if later_overlays and not self.annotation_overlays:
            if self.window_content_fallback:
                raise OccludedError("采集中出现透明叠加窗口。")
            raise CaptureError("采集中出现透明叠加窗口，丢弃本帧。")
        self.frame_id += 1
        frame = Frame(pixels, {"frame_id": self.frame_id, "captured_at": datetime.now(timezone.utc).isoformat(),
                              "capture_monotonic_ns": started, "capture_clock": "perf_counter_ns", "capture_ms": (time.perf_counter_ns() - started) / 1e6,
                              "client": before.asdict(), "client_size": [before.width, before.height],
                              "calibrated_size": list(self.config["client_size"]),
                              "region_config_version": self.config["version"],
                              "region_config_sha256":self.config_sha256,"source": "window_mss"},self.config)
        frame.metadata["keyed_overlays"] = overlays + [item for item in later_overlays if item not in overlays]
        frame.metadata["capture_quality"] = "requires_manual_overlay_review" if frame.metadata["keyed_overlays"] else "unoccluded"
        if rect is not None:
            frame.metadata.update(source="window_mss_region",capture_rect=[x,y,w,h])
        return frame

    def close(self):
        if self._graphics_capture is not None:
            self._graphics_capture.close()
            self._graphics_capture = None
        self.sct.close()


def video_frames(path: str | Path, config: dict, sample_fps: float = 15):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise CaptureError("录像无法打开。")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or sample_fps <= 0:
        cap.release()
        raise CaptureError("录像帧率无效。")
    step = max(1, round(fps / sample_fps))
    frame_id = 0
    try:
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            frame_id += 1
            if (frame_id - 1) % step:
                continue
            from app.vision.config import scaled_client
            if scaled_client(pixels.shape[1], pixels.shape[0], config) is None:
                raise CaptureError("录像尺寸与校准配置不同比例。")
            yield Frame(pixels, {"frame_id": frame_id, "captured_at": None, "video_time_ms": (frame_id - 1) / fps * 1000,
                                 "client_size": config["client_size"], "region_config_version": config["version"],
                                 "source": "video", "image_path": str(Path(path).resolve())})
    finally:
        cap.release()
