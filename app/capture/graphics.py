"""Capture one HWND through Windows.Graphics.Capture, without changing focus."""
from __future__ import annotations

from threading import Condition
import time

import numpy as np

from .window import CaptureError, client_area, graphics_capture_bounds


def crop_client(pixels, area, bounds):
    """WGC omits invisible resize borders; choose the matching window bounds."""
    height, width = pixels.shape[:2]
    for left, top, right, bottom in bounds:
        if (right - left, bottom - top) != (width, height):
            continue
        x, y = area.left - left, area.top - top
        if x < 0 or y < 0 or x + area.width > width or y + area.height > height:
            continue
        client = np.ascontiguousarray(pixels[y:y + area.height, x:x + area.width, :3])
        if int(client.max()) < 16:
            raise CaptureError("窗口采集得到空白画面，正在等待新的游戏画面。")
        return client
    raise CaptureError("窗口采集尺寸与客户区不一致，丢弃本帧。")


class GraphicsCapture:
    """Keep just the latest owned frame; never save the background stream."""
    MAX_FRAME_AGE_SECONDS = 2.0

    def __init__(self, hwnd):
        try:
            from windows_capture import WindowsCapture
        except ImportError as exc:
            raise CaptureError("缺少窗口采集依赖，请安装项目 requirements.lock。") from exc
        self.hwnd = hwnd
        self._condition = Condition()
        self._latest = None
        self._closed = False
        self._control = None
        try:
            self._capture = WindowsCapture(window_hwnd=hwnd, cursor_capture=False,
                                           draw_border=False, minimum_update_interval=100)

            @self._capture.event
            def on_frame_arrived(frame, control):
                # Native surfaces are borrowed: retain an independent BGR copy.
                pixels = frame.frame_buffer[:, :, :3].copy(order="C")
                received = time.perf_counter_ns()
                with self._condition:
                    if self._closed:
                        control.stop()
                        return
                    self._latest = (pixels, received, frame.timespan)
                    self._condition.notify_all()

            @self._capture.event
            def on_closed():
                with self._condition:
                    self._closed = True
                    self._latest = None
                    self._condition.notify_all()

            self._control = self._capture.start_free_threaded()
        except Exception as exc:
            self.close()
            raise CaptureError(f"无法启动窗口图像采集：{exc}") from exc

    def read(self):
        before = client_area(self.hwnd)
        deadline = time.monotonic() + (2.0 if self._latest is None else .3)
        with self._condition:
            while True:
                if self._closed or self._control is None or self._control.is_finished():
                    raise CaptureError("窗口图像采集已停止，正在重新连接。")
                latest = self._latest
                if latest and (time.perf_counter_ns() - latest[1]) / 1e9 <= self.MAX_FRAME_AGE_SECONDS:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CaptureError("窗口未提供新的画面，已丢弃过期截图。")
                self._condition.wait(min(remaining, .1))
        pixels, received, timespan = latest
        client = crop_client(pixels, before, graphics_capture_bounds(self.hwnd))
        if client_area(self.hwnd) != before:
            raise CaptureError("采集期间窗口位置/尺寸变化，丢弃本帧。")
        return client, before, {"capture_monotonic_ns": received, "graphics_timespan": timespan,
                                "frame_age_ms": (time.perf_counter_ns() - received) / 1e6}

    def close(self):
        with self._condition:
            self._closed = True
            self._latest = None
            self._condition.notify_all()
        control, self._control = self._control, None
        if control is not None:
            try:
                control.stop()
            except RuntimeError:
                # A failed native session must not prevent fallback/cleanup.
                pass
