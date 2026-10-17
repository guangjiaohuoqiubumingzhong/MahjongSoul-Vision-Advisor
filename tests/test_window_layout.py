from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.capture import layout
from app.capture.window import CaptureError, ClientArea


@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)])
def test_presets_share_the_same_game_position_and_leave_panel_outside(size):
    result = layout.plan_layout((0, 0, 2560, 1400), size, (16, 39))
    gx, gy, gw, gh = result.game
    px, py, pw, ph = result.panel
    assert (gx, gy) == (8, 8)
    assert (gw-16, gh-39) == size
    assert px > gx+gw and px+pw+16 <= 2560 and py+ph+39 <= 1400


def test_small_display_narrows_panel_and_negative_monitor_origin_is_preserved():
    result = layout.plan_layout((-1920, 0, 0, 1040), (1366, 768), (16, 39))
    assert result.game[0] == -1912
    assert 450 <= result.panel[2] < 560
    assert result.panel[0]+result.panel[2]+16 <= 0
    with pytest.raises(CaptureError, match="放不下"):
        layout.plan_layout((0, 0, 1920, 1040), (1920, 1080), (16, 39))


def fake_window(monkeypatch, work=(0, 0, 2560, 1400)):
    state = [ClientArea(123, "雀魂", 456, 150, 250, 1366, 768, 96)]
    calls = []

    def outer(_, pointer):
        area = state[0]
        pointer._obj.left, pointer._obj.top = area.left-8, area.top-31
        pointer._obj.right, pointer._obj.bottom = area.left+area.width+8, area.top+area.height+8
        return True

    def monitor(_, pointer):
        rect = pointer._obj.work
        rect.left, rect.top, rect.right, rect.bottom = work
        return True

    def move(hwnd, _, x, y, width, height, flags):
        calls.append((hwnd, x, y, width, height, flags))
        state[0] = replace(state[0], left=x+8, top=y+31, width=width-16, height=height-39)
        return True

    user = SimpleNamespace(GetWindowLongPtrW=lambda *_: 0x00C00000, IsZoomed=lambda _: False,
                           GetWindowRect=outer, MonitorFromWindow=lambda *_: 1,
                           GetMonitorInfoW=monitor, SetWindowPos=move)
    monkeypatch.setattr(layout, "_layout_api", lambda: user)
    monkeypatch.setattr(layout, "client_area", lambda _: state[0])
    return state, calls


@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)])
def test_resizing_uses_client_dimensions_and_real_frame_insets(monkeypatch, size):
    _, calls = fake_window(monkeypatch)
    area, result = layout.arrange_client(123, size, expected_pid=456)
    assert calls == [(123, 8, 8, size[0]+16, size[1]+39, 0x10)]
    assert (area.left, area.top, area.width, area.height) == (16, 39, *size)
    assert result.panel[0] > area.left+area.width


def test_reused_handle_and_unfitting_size_do_not_move_a_window(monkeypatch):
    _, calls = fake_window(monkeypatch, (0, 0, 1920, 1040))
    with pytest.raises(CaptureError, match="窗口已变化"):
        layout.arrange_client(123, (1366, 768), expected_pid=999)
    with pytest.raises(CaptureError, match="放不下"):
        layout.arrange_client(123, (1920, 1080), expected_pid=456)
    assert not calls
