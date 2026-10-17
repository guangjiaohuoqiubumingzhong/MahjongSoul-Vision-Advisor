"""Display scaling must not be confused with the client's real pixel size."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app.capture import frames
from app.capture.window import CaptureError, ClientArea, initialize_dpi_awareness
from app.vision.config import pixel_rect


def dpi_api(start=0, process_allowed=True, thread_allowed=True):
    state = [start]
    def process(_):
        if process_allowed: state[0] = 2
        return process_allowed
    def thread(_):
        old = state[0]
        if thread_allowed: state[0] = 2
        return -1 if thread_allowed else 0
    return SimpleNamespace(
        GetThreadDpiAwarenessContext=Mock(side_effect=lambda: state[0]),
        GetAwarenessFromDpiAwarenessContext=Mock(side_effect=lambda context: context),
        SetProcessDpiAwarenessContext=Mock(side_effect=process),
        SetThreadDpiAwarenessContext=Mock(side_effect=thread)), state


def test_initialize_physical_coordinates_for_a_new_process():
    user, state = dpi_api()
    assert initialize_dpi_awareness(user) == "per-monitor" and state[0] == 2
    user.SetProcessDpiAwarenessContext.assert_called_once()
    user.SetThreadDpiAwarenessContext.assert_not_called()


def test_existing_per_monitor_mode_is_preserved_without_repeated_changes():
    user, _ = dpi_api(start=2)
    assert initialize_dpi_awareness(user) == "per-monitor"
    user.SetProcessDpiAwarenessContext.assert_not_called()
    user.SetThreadDpiAwarenessContext.assert_not_called()


def test_capture_thread_can_use_physical_pixels_when_process_mode_was_already_set():
    user, state = dpi_api(start=1, process_allowed=False)
    assert initialize_dpi_awareness(user) == "per-monitor-thread" and state[0] == 2
    user.SetThreadDpiAwarenessContext.assert_called_once()


def test_failure_to_enable_physical_coordinates_is_not_silently_ignored():
    user, _ = dpi_api(process_allowed=False, thread_allowed=False)
    with pytest.raises(CaptureError, match="实际像素坐标"):
        initialize_dpi_awareness(user)


def capture(dpi, size=(1920, 1080)):
    area = ClientArea(123, "雀魂麻将", 456, 8, 39, *size, dpi)
    value = frames.WindowCapture.__new__(frames.WindowCapture)
    value.hwnd, value.initial, value.frame_id = area.hwnd, area, 0
    value.config = {"version": "pixel-layout", "client_size": [1366, 768], "dpi": 96}
    value.config_sha256 = frames.config_digest(value.config)
    value.sct = SimpleNamespace(monitors=[None, {"left": 0, "top": 0, "width": 3840, "height": 2160}])
    return value, area


@pytest.mark.parametrize("dpi", [0, 96, 120, 144, 192])
@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)])
def test_client_with_supported_pixels_is_accepted_at_different_display_scales(dpi, size):
    value, area = capture(dpi, size)
    value._validate_geometry(area)
    assert value.config["dpi"] == 96  # Preserve the original calibration provenance.


@pytest.mark.parametrize("dpi", [120, 144, 192])
def test_timer_uses_real_pixels_and_keeps_its_reference_size_at_high_dpi(monkeypatch, dpi):
    value, area = capture(dpi)
    pixels = np.random.default_rng(1).integers(20, 255, (1080, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr(frames, "client_area", lambda _: area)
    value._graphics_capture = SimpleNamespace(read=lambda: (pixels, area, {}))
    value.window_content_fallback = True
    rect = [1000/1366, 543/768, 230/1366, 88/768]
    frame = value.grab_region(rect)
    x, y, w, h = pixel_rect(rect, 1920, 1080)
    expected = cv2.resize(pixels[y:y+h, x:x+w], (230, 88), interpolation=cv2.INTER_AREA)
    assert np.array_equal(frame.pixels, expected)
    assert frame.metadata["client"]["dpi"] == dpi
    assert frame.metadata["native_capture_rect"] == [x, y, w, h]


def test_high_dpi_does_not_allow_an_unsupported_client_or_offscreen_capture():
    value, area = capture(144, (560, 640))
    with pytest.raises(CaptureError, match="窗口尺寸不支持"):
        value._validate_geometry(area)
    value, area = capture(144)
    with pytest.raises(CaptureError, match="移出"):
        value._validate_geometry(replace(area, left=3000))


@pytest.mark.parametrize("dpi", [0, 96, 120, 144, 192])
def test_display_scale_changes_do_not_interrupt_a_session_with_the_same_pixels(dpi):
    value, area = capture(120)
    value._validate_geometry(replace(area, dpi=dpi))


def test_real_client_resize_is_still_rejected_when_display_scale_also_changes():
    value, area = capture(120, (1366, 768))
    with pytest.raises(CaptureError, match="窗口尺寸发生变化"):
        value._validate_geometry(replace(area, width=1920, height=1080, dpi=144))


def test_window_content_keeps_the_frame_when_only_dpi_changes_during_capture(monkeypatch):
    value, area = capture(96, (1366, 768))
    pixels = np.full((768, 1366, 3), 90, np.uint8)
    sequence = iter([replace(area, dpi=120), replace(area, dpi=192)])
    monkeypatch.setattr(frames, "client_area", lambda _: next(sequence))
    value._graphics_capture = SimpleNamespace(read=lambda: (pixels, replace(area, dpi=144), {}))
    value.window_content_fallback = True
    frame = value.grab()
    assert np.array_equal(frame.pixels, pixels)
    assert frame.metadata["client"]["dpi"] == 144
    assert frame.metadata["source"] == "window_graphics_capture"


def test_screen_capture_keeps_the_frame_when_only_dpi_changes_during_capture(monkeypatch):
    value, area = capture(120, (1366, 768))
    pixels = np.full((768, 1366, 4), 90, np.uint8)
    value.sct.grab = Mock(return_value=pixels)
    sequence = iter([replace(area, dpi=144), replace(area, dpi=192)])
    monkeypatch.setattr(frames, "client_area", lambda _: next(sequence))
    monkeypatch.setattr(frames, "ensure_unoccluded", lambda _: [])
    frame = value.grab()
    value.sct.grab.assert_called_once_with({"left": 8, "top": 39, "width": 1366, "height": 768})
    assert np.array_equal(frame.pixels, pixels[:, :, :3])
    assert frame.metadata["client"]["dpi"] == 144
