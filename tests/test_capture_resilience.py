"""Live capture keeps going: incomplete boards are recaptured, covered or
minimised game windows are switched back (or read from their own rendering)."""
from types import SimpleNamespace

import numpy as np
import pytest

import app.capture.frames as frames
from app.advice import recapture_until_complete
from app.capture.window import CaptureError, ClientArea, OccludedError
from app.vision.config import load_config
from conftest import ROOT


def board(*incomplete, status="needs_confirmation"):
    regions = {name: {"kind": kind, "complete": name not in incomplete}
               for name, kind in [("hand", "hand"), ("river_left", "river"), ("meld_self", "meld"),
                                  ("score_self", "text")]}
    return {"status": status, "observation": {"regions": regions}}


def run(results, *, timer=lambda: True, stop=lambda d: False, first=None):
    frames_taken, waits = [], []
    sequence = iter(results)

    def grab():
        frames_taken.append(len(frames_taken) + 1)
        return f"frame-{len(frames_taken)}"

    def wait(delay):
        waits.append(delay)
        return stop(delay)

    frame, result, attempts = recapture_until_complete(
        "frame-0", first or board("river_left"), grab=grab, process=lambda shot: next(sequence),
        timer_active=timer, wait=wait)
    return frame, result, attempts, waits


def test_incomplete_board_is_recaptured_until_ready():
    frame, result, attempts, waits = run([board("river_left"), board(status="ready")])
    assert (frame, result["status"], attempts, waits) == ("frame-2", "ready", 2, [.4, .4])
    assert result["recaptures"] == 2


def test_recapture_keeps_the_more_complete_frame_and_stops_at_the_limit():
    first = board("river_left", "meld_self")
    frame, result, attempts, _ = run([board("river_left"), board("river_left", "meld_self"),
                                      board("river_left", "hand")], first=first)
    assert attempts == 3 and frame == "frame-1"
    assert result["observation"]["regions"]["meld_self"]["complete"]


def test_no_recapture_after_the_turn_or_on_stop_or_for_text_only_gaps():
    assert run([], timer=lambda: False)[2] == 0
    assert run([], stop=lambda d: True)[2] == 0
    text_only = board("score_self")
    frame, result, attempts, waits = run([], first=text_only)
    assert (frame, attempts, waits) == ("frame-0", 0, []) and "recaptures" not in result
    ready = board(status="ready")
    assert run([], first=ready)[2] == 0


def test_retry_does_not_start_when_its_measured_cost_would_exceed_the_budget():
    first=board('river_left');first['total_ms']=2200
    unexpected=lambda:pytest.fail('不应在时间预算不足时再截图')
    _,result,attempts=recapture_until_complete('first',first,grab=unexpected,process=unexpected,
        timer_active=lambda:True,wait=lambda _:False,deadline=5,clock=lambda:3)
    assert attempts==0 and result['status']=='needs_confirmation'
    assert result['recapture_budget_exhausted'] and result['observation']['regions']['river_left']['complete'] is False


def test_retry_rechecks_its_budget_after_waiting_and_after_an_action_probe():
    first=board('river_left');first['total_ms']=500
    now=[0.0]
    def wait(delay):now[0]=4.7;return False
    unexpected=lambda:pytest.fail('检查期间已用完预算，不应再截图')
    _,result,attempts=recapture_until_complete('first',first,grab=unexpected,process=unexpected,
        timer_active=lambda:True,wait=wait,deadline=5,clock=lambda:now[0])
    assert attempts==0 and result['recapture_budget_exhausted']


def area():
    return ClientArea(hwnd=1, title="雀魂", pid=7, left=0, top=0, width=1366, height=768, dpi=96)


@pytest.fixture
def capture(monkeypatch):
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    value = frames.WindowCapture.__new__(frames.WindowCapture)
    value.hwnd, value.config, value.config_sha256 = 1, config, "test"
    value.annotation_overlays, value.frame_id, value.initial = False, 0, area()
    value._validate_geometry = lambda _: None
    value.sct = SimpleNamespace(grab=lambda rect: np.full((rect["height"], rect["width"], 4), 90, np.uint8))
    value.RESTORE_WAIT_SECONDS = 0
    state = SimpleNamespace(covered=True, fronted=[], minimized=False, graphics_reads=0, graphics_closed=0)
    monkeypatch.setattr(frames, "client_area", lambda hwnd: (_ for _ in ()).throw(
        CaptureError("目标窗口不存在、不可见或已最小化。")) if state.minimized else area())

    def unoccluded(_):
        if state.covered:
            raise OccludedError("目标客户区被其他窗口遮挡（通知）。")
        return []

    def front(hwnd):
        state.fronted.append(hwnd)
        state.minimized = False
        if state.uncover_on_front:
            state.covered = False

    state.uncover_on_front = True
    monkeypatch.setattr(frames, "ensure_unoccluded", unoccluded)
    monkeypatch.setattr(frames, "restore_window", front)
    monkeypatch.setattr(frames, "is_minimized", lambda hwnd: state.minimized)
    monkeypatch.setattr(frames, "print_window", lambda hwnd: (np.full((768, 1366, 3), 50, np.uint8), area()))

    class Graphics:
        def __init__(self, hwnd):
            assert hwnd == 1

        def read(self):
            state.graphics_reads += 1
            return np.full((768, 1366, 3), 60, np.uint8), area(), {"capture_monotonic_ns": 1}

        def close(self):
            state.graphics_closed += 1

    monkeypatch.setattr(frames, "GraphicsCapture", Graphics)
    return value, state


def test_strict_capture_still_refuses_a_covered_window(capture):
    value, state = capture
    with pytest.raises(OccludedError):
        value.grab()
    assert state.fronted == []


@pytest.mark.parametrize("covered", [True, False])
def test_window_graphics_are_primary_without_changing_focus(capture, covered):
    value, state = capture
    state.covered = covered
    value.window_content_fallback = True
    frame = value.grab()
    assert state.fronted == [] and frame.metadata["source"] == "window_graphics_capture"
    assert frame.metadata["capture_api"] == "Windows.Graphics.Capture"
    assert frame.pixels.shape == (768, 1366, 3)
    assert frame.metadata["capture_monotonic_ns"] == 1


def test_timer_probe_uses_the_same_window_stream_when_covered(capture):
    value, state = capture
    value.window_content_fallback = True
    first = value.grab()
    assert state.fronted == [] and first.metadata["capture_quality"] == "window_content"
    probe = value.grab_region(value.config["action_timer"]["rect"])
    assert state.fronted == [] and probe.metadata["source"] == "window_graphics_capture_region"
    assert probe.pixels.shape[:2] == (88, 230)
    assert state.graphics_reads == 2


def fail_graphics(monkeypatch):
    def unavailable(hwnd):
        raise CaptureError("窗口图像采集不可用。")
    monkeypatch.setattr(frames, "GraphicsCapture", unavailable)


def test_print_window_compatibility_fallback_does_not_change_focus(capture, monkeypatch):
    value, state = capture
    value.window_content_fallback = True
    fail_graphics(monkeypatch)
    frame = value.grab()
    assert state.fronted == [] and frame.metadata["source"] == "window_content"
    assert frame.metadata["capture_api"] == "PrintWindow"
    assert frame.metadata["graphics_capture_error"] == "窗口图像采集不可用。"


def test_blank_window_content_is_reported_as_occlusion_for_a_retry(capture, monkeypatch):
    value, state = capture
    value.window_content_fallback = True
    fail_graphics(monkeypatch)
    monkeypatch.setattr(frames, "print_window", lambda hwnd: (_ for _ in ()).throw(
        CaptureError("游戏窗口自身画面为空，无法补截。")))
    with pytest.raises(OccludedError, match="通知"):
        value.grab()
    assert state.fronted == []


def test_screen_fallback_requires_an_uncovered_window(capture, monkeypatch):
    value, state = capture
    value.window_content_fallback = True
    state.covered = False
    fail_graphics(monkeypatch)
    monkeypatch.setattr(frames, "print_window", lambda hwnd: (_ for _ in ()).throw(CaptureError("空白画面")))
    frame = value.grab()
    assert frame.metadata["source"] == "window_mss"
    assert frame.metadata["window_content_error"] == "空白画面"
    assert state.fronted == []


def test_background_capture_does_not_restore_a_minimised_game(capture):
    value, state = capture
    value.window_content_fallback = True
    value.grab()
    state.minimized = True
    with pytest.raises(CaptureError, match="最小化"):
        value.grab()
    assert state.fronted == [] and state.graphics_closed == 1
    state.minimized = False
    assert value.grab().metadata["source"] == "window_graphics_capture"


def test_minimised_game_is_restored_before_capture(capture):
    value, state = capture
    value.restore_minimized = True
    state.covered, state.minimized = False, True
    frame = value.grab()
    assert state.fronted == [1] and frame.metadata["source"] == "window_mss"
    state.minimized = True
    value.restore_minimized = False
    with pytest.raises(CaptureError, match="最小化"):
        value.grab()


def test_call_buttons_without_the_glowing_discard_are_recaptured():
    def reaction(confirmed, status="needs_confirmation"):
        value = board(status=status)
        value["observation"]["regions"]["buttons"] = {"kind": "buttons", "items": [
            {"label": "pon", "clickable": True}]}
        value["observation"]["reaction_discard"] = {"confirmed": confirmed}
        return value
    frame, result, attempts, _ = run([reaction(False), reaction(True, "ready")], first=reaction(False))
    assert (frame, result["status"], attempts) == ("frame-2", "ready", 2)
    # Source already known: nothing a new frame could add.
    assert run([], first=reaction(True))[2] == 0


def test_live_advice_starts_a_fresh_analysis_after_capture_recovers(tmp_path, monkeypatch):
    import threading
    from app.capture.frames import Frame
    import app.ui.advice as ui

    stop = threading.Event()
    stop.wait = lambda delay: stop.is_set()
    state = SimpleNamespace(probes=0, boards=0, closed=False)

    def probe(rect):
        state.probes += 1
        if state.probes == 4:
            raise CaptureError("目标窗口已最小化。")
        return SimpleNamespace(pixels=np.ones(1))

    def board_frame():
        state.boards += 1
        return Frame(np.full((2, 2, 3), 50, np.uint8), {"frame_id": state.boards})

    capture = SimpleNamespace(grab_region=probe, grab=board_frame,
                              close=lambda: setattr(state, "closed", True))
    service = SimpleNamespace(clear=lambda: None, timer_active=lambda *a, **k: True,
                              process=lambda *a: {"status": "ready", "observation": None,
                                                  "recommendation": {"semantic": "discard"}})
    monkeypatch.setattr(ui, "WindowCapture", lambda hwnd, config: capture)
    panel = ui.AdvicePanel.__new__(ui.AdvicePanel)
    panel.output, panel.config = tmp_path, {"action_timer": {"rect": [0, 0, 1, 1]}}
    panel.get_service = lambda count: service
    published = []

    def publish(generation, result, pixels):
        published.append(result["status"])
        if published.count("ready") == 2:
            stop.set()

    panel.publish = publish
    panel.live_worker(1, stop, 123)
    assert published == ["confirming_frame", "ready", "waiting_turn", "confirming_frame", "ready"]
    assert state.boards == 2 and state.closed
    assert not getattr(capture, "restore_minimized", False)


def test_countdown_restart_without_disappearing_is_a_new_action():
    """Skipping a chi on the upper player's discard runs straight into the own draw."""
    from app.vision.trigger import TimerEdge
    edge = TimerEdge()
    assert edge.observe(True, 299) == "start"
    assert [edge.observe(True, s) for s in (290, 280, 277)] == [None, None, None]
    assert edge.observe(True, 300) is None          # one read could be OCR noise
    assert edge.observe(True, 300) == "start"       # confirmed restart
    assert [edge.observe(True, s) for s in (299, 298, 299, 297)] == [None] * 4   # 1 s jitter
    assert edge.observe(True, 310) is None and edge.observe(True, 296) is None   # single misread
    assert edge.observe(True, None) is None         # unread value never restarts
    assert edge.observe(False) is None and edge.observe(False) == "end"
    assert edge.observe(True, 300) == "start"


def test_recapture_reasons_name_what_is_still_moving():
    from app.advice import recapture_reasons
    assert recapture_reasons(board("river_left", "meld_self")) == ["牌河", "副露"]
    value = board(status="needs_confirmation")
    value["observation"]["regions"]["buttons"] = {"kind": "buttons", "items": [{"label": "pon", "clickable": True}]}
    value["observation"]["reaction_discard"] = {"confirmed": False}
    assert recapture_reasons(value) == ["出牌来源"]
