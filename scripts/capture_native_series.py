"""Capture native-resolution 雀魂 frames in the background for evaluation.

Frames are read with PrintWindow (the window's own rendered content), so
overlays drawn above the game - e.g. a remote-control indicator - neither block
capture nor end up in the pixels. Production capture keeps its screen-grab and
occlusion checks; this script is evaluation-only.

Saves a lossless PNG (+ .frame.json) every ``--every`` seconds while the board
changes, for ``--seconds`` seconds or until ``stop.request`` appears in the
output folder. Only reads the screen; never sends input to the game. Frames are
for comparing native vs calibrated crops (scripts/compare_native_resolution.py).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import save_image
from app.capture.window import client_area, windows
from app.vision.config import load_config, scaled_client


def print_window(hwnd):
    """Client-area BGR pixels of ``hwnd`` via PrintWindow(PW_RENDERFULLCONTENT)."""
    user, gdi = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
    handle = ctypes.c_void_p
    user.GetWindowDC.restype, user.GetWindowDC.argtypes = handle, [wintypes.HWND]
    user.ReleaseDC.argtypes = [wintypes.HWND, handle]
    user.PrintWindow.argtypes = [wintypes.HWND, handle, wintypes.UINT]
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    gdi.CreateCompatibleDC.restype, gdi.CreateCompatibleDC.argtypes = handle, [handle]
    gdi.CreateCompatibleBitmap.restype = handle
    gdi.CreateCompatibleBitmap.argtypes = [handle, ctypes.c_int, ctypes.c_int]
    gdi.SelectObject.restype, gdi.SelectObject.argtypes = handle, [handle, handle]
    gdi.DeleteObject.argtypes = [handle]
    gdi.DeleteDC.argtypes = [handle]
    gdi.GetDIBits.argtypes = [handle, handle, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p,
                              wintypes.UINT]
    area = client_area(hwnd)
    window = wintypes.RECT()
    user.GetWindowRect(hwnd, ctypes.byref(window))
    width, height = window.right - window.left, window.bottom - window.top
    source = user.GetWindowDC(hwnd)
    memory = gdi.CreateCompatibleDC(source)
    bitmap = gdi.CreateCompatibleBitmap(source, width, height)
    old = gdi.SelectObject(memory, bitmap)
    try:
        if not user.PrintWindow(hwnd, memory, 2):
            raise RuntimeError("PrintWindow failed")

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

        header = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        buffer = (ctypes.c_ubyte * (width * height * 4))()
        if not gdi.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(header), 0):
            raise RuntimeError("GetDIBits failed")
        image = np.frombuffer(buffer, np.uint8).reshape(height, width, 4)[:, :, :3]
    finally:
        gdi.SelectObject(memory, old)
        gdi.DeleteObject(bitmap)
        gdi.DeleteDC(memory)
        user.ReleaseDC(hwnd, source)
    left, top = area.left - window.left, area.top - window.top
    pixels = np.ascontiguousarray(image[top:top+area.height, left:left+area.width])
    if pixels.shape[:2] != (area.height, area.width) or pixels.max() < 16:
        raise RuntimeError("PrintWindow returned an empty or cropped image")
    return pixels, area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--title", default="雀魂")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--every", type=float, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path("data/screenshots/native-series") / datetime.now().strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--log", type=Path, default=Path("reports/vision/native-capture.log"))
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("w", encoding="utf-8")
    sys.stdout = sys.stderr = log
    config = deepcopy(load_config(args.config))
    # Evaluation capture only: record DPI in metadata instead of refusing.
    config["dpi"] = None
    areas = [area for area in windows(args.title) if scaled_client(area.width, area.height, config)]
    if not areas:
        print(f"找不到标题含“{args.title}”且为 16:9 的窗口。", flush=True)
        return 1
    area = max(areas, key=lambda value: value.width * value.height)
    print(f"capturing {area.title} {area.width}x{area.height} dpi={area.dpi} -> {args.output}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    started, last, saved = time.perf_counter(), None, 0
    while time.perf_counter() - started < args.seconds and not (args.output / "stop.request").exists():
        try:
            pixels, current = print_window(area.hwnd)
            small = cv2.resize(pixels, (96, 54), interpolation=cv2.INTER_AREA) // 8
            key = hashlib.sha256(small.tobytes()).hexdigest()
            if key != last:
                saved, last = saved + 1, key
                path = args.output / f"{saved:08d}.png"
                save_image(path, pixels)
                path.with_suffix(".frame.json").write_text(json.dumps({
                    "frame_id": saved, "captured_at": datetime.now(timezone.utc).isoformat(),
                    "source": "print_window_evaluation", "client": current.asdict(),
                    "client_size": [current.width, current.height],
                    "calibrated_size": config["client_size"], "image_path": str(path.resolve())},
                    ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"saved {saved} {pixels.shape[1]}x{pixels.shape[0]}", flush=True)
        except Exception as exc:  # minimised, resized, animation: keep going
            print(f"skip: {exc}", flush=True)
        time.sleep(args.every)
    print(f"done saved={saved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
