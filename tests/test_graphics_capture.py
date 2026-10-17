from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.capture import graphics
from app.capture.window import CaptureError, ClientArea


def area(width=1366, height=768):
    return ClientArea(123, "雀魂", 7, 101, 131, width, height, 96)


@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)])
def test_client_crop_excludes_title_and_invisible_resize_border(size):
    width, height = size
    client = area(width, height)
    # WGC excludes the seven-pixel invisible resize border of GetWindowRect.
    image = np.zeros((height + 32, width + 2, 4), np.uint8)
    image[31:31 + height, 1:1 + width, :3] = [20, 80, 150]
    crop = graphics.crop_client(image, client, [
        (93, 93, 109 + width, 140 + height), (100, 100, 102 + width, 132 + height)])
    assert crop.shape == (height, width, 3)
    assert np.all(crop == [20, 80, 150]) and crop.flags.c_contiguous


def test_invalid_crop_or_blank_client_is_rejected():
    with pytest.raises(CaptureError, match="尺寸"):
        graphics.crop_client(np.ones((4, 4, 3), np.uint8), area(), [(0, 0, 3, 3)])
    image = np.zeros((800, 1368, 3), np.uint8)
    image[:31] = 255  # A bright title bar cannot hide an empty client.
    with pytest.raises(CaptureError, match="空白"):
        graphics.crop_client(image, area(), [(100, 100, 1468, 900)])


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(arguments=None, stopped=0, frames=np.full((800, 1368, 4), 50, np.uint8))

    class Control:
        def is_finished(self):
            return False

        def stop(self):
            state.stopped += 1

    class Capture:
        def __init__(self, **kwargs):
            state.arguments = kwargs
            state.capture = self

        def event(self, handler):
            setattr(self, handler.__name__, handler)
            return handler

        def start_free_threaded(self):
            self.on_frame_arrived(SimpleNamespace(frame_buffer=state.frames, timespan=1000), Control())
            return Control()

    monkeypatch.setitem(sys.modules, "windows_capture", SimpleNamespace(WindowsCapture=Capture))
    monkeypatch.setattr(graphics, "client_area", lambda hwnd: area())
    monkeypatch.setattr(graphics, "graphics_capture_bounds", lambda hwnd: [(100, 100, 1468, 900)])
    return state


def test_hwnd_target_and_owned_frame_copy(native):
    capture = graphics.GraphicsCapture(123)
    native.frames[:] = 99
    pixels, client, details = capture.read()
    assert native.arguments == {"window_hwnd": 123, "cursor_capture": False,
                                "draw_border": False, "minimum_update_interval": 100}
    assert np.all(pixels == 50) and client == area()
    assert details["graphics_timespan"] == 1000
    assert 0 <= details["frame_age_ms"] < 2000
    capture.close()
    capture.close()
    assert native.stopped == 1 and capture._latest is None


def test_closed_session_cannot_return_cached_pixels(native):
    capture = graphics.GraphicsCapture(123)
    native.capture.on_closed()
    with pytest.raises(CaptureError, match="停止"):
        capture.read()
    assert capture._latest is None
    capture.close()


def test_stale_frames_are_not_given_a_new_timestamp(native, monkeypatch):
    capture = graphics.GraphicsCapture(123)
    pixels, _, timespan = capture._latest
    capture._latest = (pixels, 0, timespan)
    monkeypatch.setattr(graphics.time, "perf_counter_ns", lambda: 3_000_000_000)
    ticks = iter([0, 1])
    monkeypatch.setattr(graphics.time, "monotonic", lambda: next(ticks))
    with pytest.raises(CaptureError, match="过期"):
        capture.read()
    capture.close()


def test_dpi_change_during_read_preserves_the_same_client_pixels(native, monkeypatch):
    capture = graphics.GraphicsCapture(123)
    sequence = iter([area(), replace(area(), dpi=144)])
    monkeypatch.setattr(graphics, "client_area", lambda hwnd: next(sequence))
    pixels, client, _ = capture.read()
    assert pixels.shape == (768, 1366, 3) and np.all(pixels == 50)
    assert client.asdict()["dpi"] == 96
    capture.close()


@pytest.mark.parametrize("change", [{"left": 102}, {"width": 1920, "height": 1080}])
def test_window_move_or_resize_during_read_discards_the_frame(native, monkeypatch, change):
    capture = graphics.GraphicsCapture(123)
    moved = replace(area(), **change)
    sequence = iter([area(), moved])
    monkeypatch.setattr(graphics, "client_area", lambda hwnd: next(sequence))
    with pytest.raises(CaptureError, match="位置/尺寸变化"):
        capture.read()
    capture.close()
