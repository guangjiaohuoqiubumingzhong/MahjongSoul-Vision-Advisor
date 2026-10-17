from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
import os

import numpy as np


class CaptureError(RuntimeError):
    pass


class OccludedError(CaptureError):
    """Another window covers the client area; its own content is still readable."""


def _api():
    if os.name != "nt":
        raise CaptureError("窗口采集只支持 Windows。")
    user = ctypes.WinDLL("user32", use_last_error=True)
    signatures = {
        "IsWindow": ([wintypes.HWND], wintypes.BOOL),
        "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
        "IsIconic": ([wintypes.HWND], wintypes.BOOL),
        "GetWindowTextLengthW": ([wintypes.HWND], ctypes.c_int),
        "GetWindowTextW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        "GetClientRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        "ClientToScreen": ([wintypes.HWND, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
        "GetDpiForWindow": ([wintypes.HWND], wintypes.UINT),
        "WindowFromPoint": ([wintypes.POINT], wintypes.HWND),
        "GetAncestor": ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
        "GetWindow": ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
        "GetWindowRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        "SetProcessDpiAwarenessContext": ([wintypes.HANDLE], wintypes.BOOL),
        "GetWindowThreadProcessId": ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
    }
    for name, (args, result) in signatures.items():
        getattr(user, name).argtypes = args
        getattr(user, name).restype = result
    user.SetProcessDpiAwarenessContext(wintypes.HANDLE(-4))
    return user


@dataclass(frozen=True)
class ClientArea:
    hwnd: int
    title: str
    pid: int
    left: int
    top: int
    width: int
    height: int
    dpi: int

    def asdict(self):
        return asdict(self)


def client_area(hwnd: int) -> ClientArea:
    user = _api()
    if not user.IsWindow(hwnd) or not user.IsWindowVisible(hwnd) or user.IsIconic(hwnd):
        raise CaptureError("目标窗口不存在、不可见或已最小化。")
    rect, origin, pid = wintypes.RECT(), wintypes.POINT(0, 0), wintypes.DWORD()
    if not user.GetClientRect(hwnd, ctypes.byref(rect)) or not user.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise CaptureError("无法读取窗口客户区。")
    text = ctypes.create_unicode_buffer(user.GetWindowTextLengthW(hwnd) + 1)
    user.GetWindowTextW(hwnd, text, len(text))
    user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        raise CaptureError("客户区尺寸无效。")
    return ClientArea(hwnd, text.value, pid.value, origin.x, origin.y, width, height, user.GetDpiForWindow(hwnd))


def windows(title_contains: str = "") -> list[ClientArea]:
    user = _api()
    results = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def visit(hwnd, _):
        try:
            area = client_area(hwnd)
            if area.title and title_contains.casefold() in area.title.casefold():
                results.append(area)
        except CaptureError:
            pass
        return True

    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.EnumWindows.restype = wintypes.BOOL
    if not user.EnumWindows(visit, 0):
        raise CaptureError("枚举窗口失败。")
    return results


def window_label(hwnd) -> str:
    """Title (or class name) of a window, to tell the user what is in the way."""
    user = _api()
    try:
        length = user.GetWindowTextLengthW(hwnd)
        text = ctypes.create_unicode_buffer(length + 1)
        user.GetWindowTextW(hwnd, text, len(text))
        if text.value.strip():
            return text.value.strip()
        user.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        name = ctypes.create_unicode_buffer(256)
        user.GetClassNameW(hwnd, name, len(name))
        return name.value or "未命名窗口"
    except Exception:
        return "未知窗口"


def is_minimized(hwnd: int) -> bool:
    user = _api()
    return bool(user.IsWindow(hwnd) and user.IsIconic(hwnd))


def restore_window(hwnd: int):
    """Bring the game window back to the front (restoring it if minimised).

    A maximised or normal window keeps its size: SW_RESTORE is used only when
    the window is iconic, otherwise it would un-maximise the game.
    """
    user = _api()
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.SetForegroundWindow.argtypes = [wintypes.HWND]
    user.BringWindowToTop.argtypes = [wintypes.HWND]
    if user.IsIconic(hwnd):
        user.ShowWindow(hwnd, 9)      # SW_RESTORE
    if not user.SetForegroundWindow(hwnd):
        # Foreground lock (another app has focus): at least raise it in z-order.
        user.BringWindowToTop(hwnd)


def print_window(hwnd: int) -> tuple[np.ndarray, ClientArea]:
    """Client-area BGR pixels from the window's own rendering (PrintWindow).

    Windows drawn above the game (overlays, notifications, other apps) are not
    part of this image, so it stays valid while the client area is covered.
    """
    user, gdi = _api(), ctypes.WinDLL("gdi32")
    area = client_area(hwnd)
    handle = ctypes.c_void_p
    user.GetWindowDC.restype, user.GetWindowDC.argtypes = handle, [wintypes.HWND]
    user.ReleaseDC.argtypes = [wintypes.HWND, handle]
    user.PrintWindow.argtypes = [wintypes.HWND, handle, wintypes.UINT]
    user.PrintWindow.restype = wintypes.BOOL
    gdi.CreateCompatibleDC.restype, gdi.CreateCompatibleDC.argtypes = handle, [handle]
    gdi.CreateCompatibleBitmap.restype = handle
    gdi.CreateCompatibleBitmap.argtypes = [handle, ctypes.c_int, ctypes.c_int]
    gdi.SelectObject.restype, gdi.SelectObject.argtypes = handle, [handle, handle]
    gdi.DeleteObject.argtypes = [handle]
    gdi.DeleteDC.argtypes = [handle]
    gdi.GetDIBits.argtypes = [handle, handle, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p,
                              wintypes.UINT]
    window = wintypes.RECT()
    user.GetWindowRect(hwnd, ctypes.byref(window))
    width, height = window.right - window.left, window.bottom - window.top
    source = user.GetWindowDC(hwnd)
    memory = gdi.CreateCompatibleDC(source)
    bitmap = gdi.CreateCompatibleBitmap(source, width, height)
    old = gdi.SelectObject(memory, bitmap)
    try:
        if not user.PrintWindow(hwnd, memory, 2):          # PW_RENDERFULLCONTENT
            raise CaptureError("无法读取游戏窗口自身画面。")

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

        header = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        buffer = (ctypes.c_ubyte * (width * height * 4))()
        if not gdi.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(header), 0):
            raise CaptureError("无法读取游戏窗口自身画面。")
        image = np.frombuffer(buffer, np.uint8).reshape(height, width, 4)[:, :, :3]
    finally:
        gdi.SelectObject(memory, old)
        gdi.DeleteObject(bitmap)
        gdi.DeleteDC(memory)
        user.ReleaseDC(hwnd, source)
    left, top = area.left - window.left, area.top - window.top
    pixels = np.ascontiguousarray(image[top:top + area.height, left:left + area.width])
    if pixels.shape[:2] != (area.height, area.width) or int(pixels.max()) < 16:
        raise CaptureError("游戏窗口自身画面为空，无法补截。")
    return pixels, area


def graphics_capture_bounds(hwnd: int):
    """Visible and full window bounds; WGC normally uses the former."""
    user = _api()
    rect = wintypes.RECT()
    if not user.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise CaptureError("无法读取窗口边界。")
    bounds = [(rect.left, rect.top, rect.right, rect.bottom)]
    dwm = ctypes.WinDLL("dwmapi")
    dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dwm.DwmGetWindowAttribute.restype = ctypes.c_long
    visible = wintypes.RECT()
    if dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(visible), ctypes.sizeof(visible)) == 0:
        bounds.insert(0, (visible.left, visible.top, visible.right, visible.bottom))
    return bounds


def ensure_unoccluded(area: ClientArea):
    user = _api()
    # Enumerate all windows above the target: sparse point sampling alone misses
    # small tooltips and panels between the samples. Conservative rectangles may
    # pause for transparent decoration, which is preferable to corrupted pixels.
    above = user.GetWindow(area.hwnd, 3)  # GW_HWNDPREV, towards the top of z-order
    dwm = ctypes.WinDLL("dwmapi")
    keyed_overlays = []
    user.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user.GetLayeredWindowAttributes.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(wintypes.DWORD)]
    while above:
        rect, cloaked = wintypes.RECT(), wintypes.DWORD()
        dwm.DwmGetWindowAttribute(wintypes.HWND(above), 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        if user.IsWindowVisible(above) and not user.IsIconic(above) and not cloaked.value and user.GetWindowRect(above, ctypes.byref(rect)):
            if rect.left < area.left+area.width and rect.right > area.left and rect.top < area.top+area.height and rect.bottom > area.top:
                style = user.GetWindowLongPtrW(above, -20)
                color, alpha, flags = wintypes.DWORD(), ctypes.c_ubyte(), wintypes.DWORD()
                keyed = user.GetLayeredWindowAttributes(above, ctypes.byref(color), ctypes.byref(alpha), ctypes.byref(flags))
                if style & 0x20 and keyed and flags.value & 1:
                    keyed_overlays.append({"hwnd": above, "screen_rect": [rect.left, rect.top, rect.right, rect.bottom], "color_key": color.value})
                else:
                    raise OccludedError(f"目标客户区被其他窗口遮挡（{window_label(above)}）。")
        above = user.GetWindow(above, 3)
    # Child/owned-window and unusual compositor cases also get point checks.
    for x_fraction in (0.04, 0.25, 0.5, 0.75, 0.96):
        for y_fraction in (0.04, 0.25, 0.5, 0.75, 0.96):
            point = wintypes.POINT(area.left + int(area.width * x_fraction), area.top + int(area.height * y_fraction))
            under = user.WindowFromPoint(point)
            if user.GetAncestor(under, 2) != area.hwnd:
                raise OccludedError(f"目标客户区被其他窗口遮挡（{window_label(user.GetAncestor(under, 2))}）。")
    return keyed_overlays
