"""Offline checks: these mocks never enumerate or capture the desktop."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from app.capture.frames import Frame
from app.capture.window import CaptureError
from app.ui import manual_capture as manual


def test_window_filter_requires_exact_executable_and_reports_access_failure(monkeypatch):
    area=SimpleNamespace(hwnd=10,pid=20)
    monkeypatch.setattr(manual,"windows",lambda title:[area])
    monkeypatch.setattr(manual,"process_exe",lambda _:Path("other.exe"))
    assert manual.game_windows(Path("game.exe"))==[]
    monkeypatch.setattr(manual,"process_exe",lambda _:Path("game.exe"))
    assert manual.game_windows(Path("GAME.exe"))==[area]
    def denied(_):
        raise manual.psutil.AccessDenied(20)
    monkeypatch.setattr(manual,"process_exe",denied)
    with pytest.raises(CaptureError,match="无权限"):
        manual.game_windows(Path("game.exe"))


def test_split_cannot_reuse_a_test_game_for_training(tmp_path,monkeypatch):
    monkeypatch.setattr(manual,"WORKSPACE",tmp_path)
    (tmp_path/"data").mkdir()
    (tmp_path/"data/frame.annotation.json").write_text(
        json.dumps({"game_id":"held-out-game","split":"test","reviewed":False}))
    with pytest.raises(ValueError,match="同一对局"):
        manual.validate_split("held-out-game","train")
    manual.validate_split("held-out-game","test")


@pytest.mark.parametrize("pid",[None,99])
def test_reused_or_missing_window_never_starts_capture(tmp_path,monkeypatch,pid):
    monkeypatch.setattr(manual,"game_windows",lambda _:[] if pid is None else [SimpleNamespace(hwnd=10,pid=pid)])
    def forbidden(*_):
        pytest.fail("Must not open a capture handle for a replaced process")
    monkeypatch.setattr(manual,"WindowCapture",forbidden)
    request=manual.CaptureRequest(10,20,"training-game","train",tmp_path,Path("game.exe"))
    with pytest.raises(CaptureError):
        manual.capture_one(request,{"version":"unit"})


def test_human_capture_preserves_pixels_and_starts_unreviewed(tmp_path,monkeypatch):
    monkeypatch.setattr(manual,"WORKSPACE",tmp_path)
    monkeypatch.setattr(manual,"game_windows",lambda _:[SimpleNamespace(hwnd=10,pid=20)])
    pixels=np.full((3,4,3),127,dtype=np.uint8)
    config={"version":"unit","client_size":[4,3]}
    closed=[]
    class FakeCapture:
        def __init__(self,*_):
            self.initial=SimpleNamespace(pid=20)
        def grab(self):
            return Frame(pixels,{"frame_id":1},config)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(manual,"WindowCapture",FakeCapture)
    request=manual.CaptureRequest(10,20,"training-game","train",tmp_path/"data/capture",Path("game.exe"))
    image=manual.capture_one(request,config)
    from app.capture.frames import load_image
    assert np.array_equal(load_image(image),pixels)
    annotation=json.loads(image.with_suffix(".annotation.json").read_text(encoding="utf-8"))
    assert annotation["reviewed"] is False and annotation["targets"]==[]
    assert annotation["split"]=="train" and annotation["game_id"]=="training-game"
    metadata=json.loads(image.with_suffix(".frame.json").read_text(encoding="utf-8"))
    assert metadata["game_inputs_sent"]==0
    assert metadata["capture_trigger"]=="human_button_click_after_countdown"
    assert Path(metadata["region_config_snapshot"]).is_file()
    with pytest.raises(CaptureError,match="不能覆盖"):
        manual.capture_one(request,config)
    assert closed==[True,True]


def test_capture_initialization_failure_closes_handle(monkeypatch):
    import mss
    from app.capture import frames
    closed=[]
    monkeypatch.setattr(mss,'mss',lambda:SimpleNamespace(close=lambda:closed.append(True)))
    def missing(_):
        raise CaptureError('Window closed before initialization')
    monkeypatch.setattr(frames,'client_area',missing)
    with pytest.raises(CaptureError):
        frames.WindowCapture(10,{})
    assert closed==[True]
