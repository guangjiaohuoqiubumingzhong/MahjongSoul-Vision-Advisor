"""Release launcher and diagnostics using the exact bundled application stack."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback


def self_test(resources, userdata, image=None, capture_check=False, report=None, capture_hwnd=None):
    import tkinter as tk
    import psutil
    from mss import mss
    from windows_capture import WindowsCapture
    from app.advice import ScreenshotAdvice, load_screenshot, format_advice
    from app.ui.advice import AdvicePanel
    from app.vision.config import load_config

    config_path = resources / "configs/window-steam-1366-v2.json"
    config = load_config(config_path)
    if report is None:
        report = {}
    report.update({"status": "running", "frozen": bool(getattr(sys, "frozen", False)),
              "bundled_runtime": (resources / "portable-runtime.json").is_file(),
              "python": sys.version, "resources": str(resources),
              "executable": sys.executable, "config": config["version"], "checks": {}})
    root = tk.Tk()
    root.withdraw()
    panel = None
    try:
        panel = AdvicePanel(root, config_path, userdata / "screenshots")
        root.update_idletasks()
        report["checks"]["panel"] = {"title": root.title(), "buttons": list(panel.buttons)}
    finally:
        if panel:
            panel.close()
        else:
            root.destroy()
    with mss() as screen:
        report["checks"]["screen_capture"] = {"monitors": len(screen.monitors) - 1}
    report["checks"]["windows_capture_import"] = WindowsCapture.__name__
    service = ScreenshotAdvice(config)
    service.prepare()
    report["checks"]["mortal_cuda"] = service.backend.runtime.verification
    report["checks"]["tile_classifier"] = {
        "provider": service.recognizer.bank.model.provider,
        "fp32_reference_loaded": service.recognizer.bank.model._reference_model is not None}
    report["checks"]["ocr"] = service.recognizer.text_reader.session.get_providers()
    if capture_hwnd:
        from app.capture.frames import WindowCapture
        capture = WindowCapture(capture_hwnd, config)
        capture.window_content_fallback = True
        try:
            frame = capture.grab()
            report["checks"]["external_window_capture"] = {
                "shape": list(frame.pixels.shape), "capture": frame.metadata,
                "samples": [frame.pixels[y, x].tolist() for y, x in [(192, 341), (192, 1024), (576, 341), (576, 1024)]]}
            if frame.metadata.get("graphics_capture_error"):
                report.setdefault("warnings", []).append("Windows.Graphics.Capture 初始化被系统拒绝，已使用现有 PrintWindow 兼容采集。")
        finally:
            capture.close()
    if image:
        pixels, metadata = load_screenshot(image, config)
        result = service.process(pixels, metadata)
        report["checks"]["screenshot_advice"] = {
            "status": result["status"], "message": format_advice(result),
            "recommendation": result["recommendation"], "total_ms": result["total_ms"],
            "reason": result.get("reason")}
        if result["status"] != "ready":
            raise RuntimeError("测试截图未生成建议：" + str(result.get("reason")))
    if capture_check:
        from scripts.verify_window_content_capture import main as verify_capture
        output = userdata / "capture-check" / str(time.time_ns())
        verify_capture(output=output, config_path=config_path, require_wgc=False)
        report["checks"]["window_capture"] = json.loads((output / "occlusion-check.json").read_text(encoding="utf-8"))
        if report["checks"]["window_capture"]["capture_api"] != "Windows.Graphics.Capture":
            report.setdefault("warnings", []).append("窗口采集使用 PrintWindow 兼容路径；Windows.Graphics.Capture 后台采集尚未通过。")
    process = psutil.Process()
    report["native_modules"] = sorted({entry.path for entry in process.memory_maps()
                                       if entry.path.lower().endswith((".dll", ".pyd"))})
    report["sys_path"] = sys.path
    report["status"] = "passed"
    return report


def run(resources: Path, userdata: Path):
    parser = argparse.ArgumentParser(description="Majan 便携测试版")
    parser.add_argument("--self-test", action="store_true", help="检查模型、GPU、界面和采集依赖")
    parser.add_argument("--image", type=Path, help="自检时分析一张完整游戏截图")
    parser.add_argument("--capture-check", action="store_true", help="自检时验证遮挡窗口采集")
    parser.add_argument("--report", type=Path, help="自检结果文件")
    parser.add_argument("--capture-hwnd", type=int, help="自检时采集指定的测试窗口")
    args = parser.parse_args()
    if args.self_test:
        output = args.report or userdata / "self-test.json"
        report = {}
        try:
            self_test(resources, userdata, args.image, args.capture_check, report, args.capture_hwnd)
        except Exception as exc:
            report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if report["status"] == "passed" else 1
    from app.ui.advice import launch
    launch(resources / "configs/window-steam-1366-v2.json", userdata / "screenshots")
    return 0
