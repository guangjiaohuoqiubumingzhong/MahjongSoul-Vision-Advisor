"""Exercise WGC against an opaque, fully covered test window without input."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
from threading import Event, Thread
import time

import numpy as np

from app.capture.frames import WindowCapture
from app.capture.window import _api, CaptureError, client_area, ensure_unoccluded
from app.vision.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PALETTES = [[(20, 80, 150), (100, 180, 30), (180, 40, 100), (30, 120, 210)],
            [(210, 120, 30), (30, 180, 100), (100, 40, 180), (150, 80, 20)]]


def test_windows(state, ready):
    """Own the fixture windows on a message thread; neither window activates."""
    user, gdi = _api(), ctypes.WinDLL("gdi32")
    procedure = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                  wintypes.WPARAM, wintypes.LPARAM)

    class WindowClass(ctypes.Structure):
        _fields_ = [("style", wintypes.UINT), ("procedure", procedure), ("class_extra", ctypes.c_int),
                    ("window_extra", ctypes.c_int), ("instance", wintypes.HINSTANCE),
                    ("icon", wintypes.HICON), ("cursor", wintypes.HANDLE), ("brush", wintypes.HBRUSH),
                    ("menu", wintypes.LPCWSTR), ("name", wintypes.LPCWSTR)]

    class Paint(ctypes.Structure):
        _fields_ = [("dc", wintypes.HDC), ("erase", wintypes.BOOL), ("rect", wintypes.RECT),
                    ("restore", wintypes.BOOL), ("update", wintypes.BOOL), ("reserved", ctypes.c_byte * 32)]

    user.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user.DefWindowProcW.restype = ctypes.c_ssize_t
    user.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(Paint)]
    user.BeginPaint.restype = wintypes.HDC
    user.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(Paint)]
    user.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH]
    user.InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.BOOL]
    user.DestroyWindow.argtypes = [wintypes.HWND]
    user.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user.PostQuitMessage.argtypes = [ctypes.c_int]
    gdi.CreateSolidBrush.argtypes = [wintypes.DWORD]
    gdi.CreateSolidBrush.restype = wintypes.HBRUSH
    gdi.DeleteObject.argtypes = [wintypes.HANDLE]

    @procedure
    def paint(hwnd, message, wp, lp):
        if message == 0x8001:
            state["phase"] = 1
            user.InvalidateRect(hwnd, None, True)
            return 0
        if message == 0x000F:  # WM_PAINT
            ps = Paint()
            dc = user.BeginPaint(hwnd, ctypes.byref(ps))
            colors = PALETTES[state["phase"]] if hwnd == state.get("target") else [(255, 0, 255)] * 4
            for i, (b, g, r) in enumerate(colors):
                rect = wintypes.RECT((i % 2) * 683, (i // 2) * 384,
                                    (i % 2 + 1) * 683, (i // 2 + 1) * 384)
                brush = gdi.CreateSolidBrush(r | (g << 8) | (b << 16))
                user.FillRect(dc, ctypes.byref(rect), brush)
                gdi.DeleteObject(brush)
            user.EndPaint(hwnd, ctypes.byref(ps))
            return 0
        if message == 0x0010:
            user.DestroyWindow(hwnd)
            return 0
        if message == 0x0002 and hwnd == state.get("target"):
            user.PostQuitMessage(0)
            return 0
        return user.DefWindowProcW(hwnd, message, wp, lp)

    class_name = "MajanWindowCaptureVerification"
    wc = WindowClass(procedure=paint, name=class_name)
    user.RegisterClassW.argtypes = [ctypes.POINTER(WindowClass)]
    user.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
    user.CreateWindowExW.restype = wintypes.HWND
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.RegisterClassW(ctypes.byref(wc))
    # WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, borderless WS_POPUP.
    for name in ("target", "cover"):
        state[name] = user.CreateWindowExW(0x08000080, class_name, "Majan capture " + name,
                                          0x80000000, 16, 39, 1366, 768, None, None, None, None)
        if not state[name]:
            state["error"] = "测试窗口创建失败"
            ready.set()
            return
    user.ShowWindow(state["target"], 4)  # SW_SHOWNOACTIVATE
    ready.set()
    message = wintypes.MSG()
    while user.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
        user.TranslateMessage(ctypes.byref(message))
        user.DispatchMessageW(ctypes.byref(message))


def samples(pixels):
    return [pixels[y, x].tolist() for y, x in [(192, 341), (192, 1024), (576, 341), (576, 1024)]]


def capture_item_diagnostic(hwnd):
    """Expose the HRESULT hidden by the native capture library on failure."""
    from uuid import UUID
    api = ctypes.WinDLL("combase")
    pointer = ctypes.c_void_p
    api.WindowsCreateString.argtypes = [wintypes.LPCWSTR, ctypes.c_uint, ctypes.POINTER(pointer)]
    api.RoGetActivationFactory.argtypes = [pointer, pointer, ctypes.POINTER(pointer)]
    api.RoInitialize.argtypes = [ctypes.c_uint]
    name, factory, item = pointer(), pointer(), pointer()
    result = {"initialize": hex(api.RoInitialize(1) & 0xffffffff)}
    text = "Windows.Graphics.Capture.GraphicsCaptureItem"
    api.WindowsCreateString(text, len(text), ctypes.byref(name))
    interop = (ctypes.c_ubyte * 16).from_buffer_copy(UUID("3628e81b-3cac-4c60-b7f4-23ce0e0c3356").bytes_le)
    item_iid = (ctypes.c_ubyte * 16).from_buffer_copy(UUID("79c3f95b-31f7-4ec2-a464-632ef5d30760").bytes_le)
    hr = api.RoGetActivationFactory(name, ctypes.byref(interop), ctypes.byref(factory))
    result["factory"] = hex(hr & 0xffffffff)
    if hr >= 0:
        table = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(pointer))).contents
        create = ctypes.WINFUNCTYPE(ctypes.c_long, pointer, wintypes.HWND, pointer, ctypes.POINTER(pointer))(table[3])
        result["create_for_window"] = hex(create(factory, hwnd, ctypes.byref(item_iid), ctypes.byref(item)) & 0xffffffff)
        if item:
            item_table = ctypes.cast(item, ctypes.POINTER(ctypes.POINTER(pointer))).contents
            ctypes.WINFUNCTYPE(ctypes.c_ulong, pointer)(item_table[2])(item)
        ctypes.WINFUNCTYPE(ctypes.c_ulong, pointer)(table[2])(factory)
    api.WindowsDeleteString.argtypes = [pointer]
    api.WindowsDeleteString(name)
    return result


def main(output=None, config_path=None, require_wgc=True):
    output = Path(output) if output is not None else ROOT / "reports/window-content-capture"
    config_path = Path(config_path) if config_path is not None else ROOT / "configs/window-steam-1366-v2.json"
    output.mkdir(parents=True, exist_ok=True)
    user = _api()
    user.GetForegroundWindow.restype = wintypes.HWND
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    foreground = user.GetForegroundWindow()
    ready, state, capture = Event(), {"phase": 0}, None
    thread = Thread(target=test_windows, args=(state, ready), daemon=True)
    thread.start()
    try:
        if not ready.wait(3) or state.get("error"):
            raise RuntimeError(state.get("error", "测试窗口启动超时"))
        capture = WindowCapture(state["target"], load_config(config_path))
        capture.window_content_fallback = True
        before = capture.grab()
        if require_wgc and before.metadata["source"] != "window_graphics_capture":
            raise RuntimeError(str({"frame": before.metadata, "winrt": capture_item_diagnostic(state["target"])}))
        assert samples(before.pixels) == [list(c) for c in PALETTES[0]]
        before.save(output / "before-cover")
        user.ShowWindow(state["cover"], 4)
        occlusion = None
        try:
            ensure_unoccluded(client_area(state["target"]))
        except CaptureError as exc:
            occlusion = str(exc)
        assert occlusion, "测试遮挡窗口未盖住目标"
        # Change the target while completely covered: a cached pre-cover image
        # cannot satisfy this assertion, nor can pixels of the magenta cover.
        user.PostMessageW(state["target"], 0x8001, 0, 0)
        deadline = time.monotonic() + 3
        while True:
            after = capture.grab()
            if samples(after.pixels) == [list(c) for c in PALETTES[1]]:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("完全遮挡后未读到目标的新画面")
            time.sleep(.1)
        assert after.metadata["source"] in ({"window_graphics_capture"} if require_wgc else {"window_graphics_capture", "window_content"}), after.metadata
        after.save(output / "covered-and-updated")
        probe = capture.grab_region(capture.config["action_timer"]["rect"])
        assert probe.pixels.shape == (88, 230, 3)
        assert probe.metadata["source"] in ({"window_graphics_capture_region"} if require_wgc else {"window_graphics_capture_region", "window_content_region"})
        summary = {"status": "passed", "fully_occluded": True, "changed_while_covered": True,
                   "occlusion": occlusion, "focus_unchanged": foreground == user.GetForegroundWindow(),
                   "client_size": [1366, 768], "probe_size": [230, 88],
                   "before_samples": samples(before.pixels), "after_samples": samples(after.pixels),
                   "capture_api": after.metadata["capture_api"], "frame_age_ms": after.metadata.get("frame_age_ms")}
        assert summary["focus_unchanged"], summary
        (output / "occlusion-check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        if capture is not None:
            capture.close()
        for name in ("cover", "target"):
            if state.get(name):
                user.PostMessageW(state[name], 0x0010, 0, 0)
        thread.join(2)


if __name__ == "__main__":
    main()
