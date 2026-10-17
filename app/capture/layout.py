"""Arrange a window by client size, accounting for its real frame and monitor."""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from .window import CaptureError, _api, client_area


CLIENT_PRESETS = {"1366×768": (1366, 768), "1920×1080": (1920, 1080)}


class MonitorInfo(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("monitor", wintypes.RECT),
                ("work", wintypes.RECT), ("flags", wintypes.DWORD)]


@dataclass(frozen=True)
class WindowLayout:
    game: tuple[int, int, int, int]
    panel: tuple[int, int, int, int]


def plan_layout(work, size, frame, panel_size=(560, 640), panel_frame=(16, 39)):
    left, top, right, bottom = work
    width, height = size[0] + frame[0], size[1] + frame[1]
    margin, gap = 8, 12
    available_width, available_height = right-left-2*margin, bottom-top-2*margin
    if width > available_width or height > available_height:
        raise CaptureError(f"当前屏幕放不下 {size[0]}×{size[1]} 的游戏窗口，请选择较小画面。")
    game = (left+margin, top+margin, width, height)
    panel_width, panel_height = panel_size
    # A 1920-wide display can fit the smaller game with a slightly narrower panel.
    beside = min(panel_width, available_width-width-gap-panel_frame[0])
    if beside >= 450 and panel_height+panel_frame[1] <= available_height:
        panel = (game[0]+width+gap, game[1], beside, panel_height)
    elif (panel_width+panel_frame[0] <= available_width
          and height+gap+panel_height+panel_frame[1] <= available_height):
        panel = (game[0], game[1]+height+gap, panel_width, panel_height)
    else:
        raise CaptureError("当前屏幕无法同时放下游戏和建议面板，请选择较小画面。")
    return WindowLayout(game, panel)


def _layout_api():
    user = _api()
    user.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user.IsZoomed.argtypes = [wintypes.HWND]
    user.IsZoomed.restype = wintypes.BOOL
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.ShowWindow.restype = wintypes.BOOL
    user.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user.MonitorFromWindow.restype = wintypes.HANDLE
    user.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
    user.GetMonitorInfoW.restype = wintypes.BOOL
    user.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user.SetWindowPos.restype = wintypes.BOOL
    return user


def position_panel(root, layout):
    x, y, width, height = layout.panel
    root.geometry(f"{width}x{height}")
    root.update_idletasks()
    user = _layout_api()
    hwnd = user.GetAncestor(root.winfo_id(), 2)
    # Native coordinates also handle monitors left/above the primary screen;
    # Tk's negative geometry offsets instead anchor to the screen's far edge.
    if not user.SetWindowPos(hwnd, None, x, y, 0, 0, 0x15):
        raise CaptureError("无法将建议面板移到游戏旁边。")


def arrange_client(hwnd, size, *, expected_pid=None):
    if tuple(size) not in CLIENT_PRESETS.values():
        raise CaptureError("请选择 1366×768 或 1920×1080。")
    user = _layout_api()
    area = client_area(hwnd)
    if expected_pid is not None and area.pid != expected_pid:
        raise CaptureError("游戏窗口已变化，请刷新窗口列表。")
    if not user.GetWindowLongPtrW(hwnd, -16) & 0x00C00000:  # WS_CAPTION
        raise CaptureError("请先在游戏设置中切换到窗口模式，再整理窗口。")
    if user.IsZoomed(hwnd):
        user.ShowWindow(hwnd, 9)  # SW_RESTORE
        area = client_area(hwnd)
    outer = wintypes.RECT()
    if not user.GetWindowRect(hwnd, ctypes.byref(outer)):
        raise CaptureError("无法读取游戏窗口边框。")
    frame = (outer.right-outer.left-area.width, outer.bottom-outer.top-area.height)
    info = MonitorInfo()
    info.size = ctypes.sizeof(info)
    monitor = user.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
    if not monitor or not user.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise CaptureError("无法读取游戏所在屏幕的可用区域。")
    work = (info.work.left, info.work.top, info.work.right, info.work.bottom)
    layout = plan_layout(work, size, frame)
    x, y, width, height = layout.game
    # Resize/move only the chosen game window; do not send any game operation.
    if not user.SetWindowPos(hwnd, None, x, y, width, height, 0x10):  # SWP_NOACTIVATE
        raise CaptureError("自动整理游戏窗口失败。")
    arranged = client_area(hwnd)
    if arranged.pid != area.pid:
        raise CaptureError("游戏窗口已变化，请重新选择。")
    if (arranged.width, arranged.height) != tuple(size):
        raise CaptureError(f"游戏未应用画面大小：实际 {arranged.width}×{arranged.height}。")
    return arranged, layout
