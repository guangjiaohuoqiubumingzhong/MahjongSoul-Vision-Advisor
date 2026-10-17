"""Launch the release with developer search paths removed, against a test HWND."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_window_content_capture import test_windows
from app.capture.window import _api


def main():
    folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "dist/Majan-portable"
    ready, state = Event(), {"phase": 0}
    thread = Thread(target=test_windows, args=(state, ready), daemon=True)
    thread.start()
    try:
        if not ready.wait(3) or state.get("error"):
            raise RuntimeError(state.get("error", "fixture did not start"))
        env = os.environ.copy()
        windows = env.get("SYSTEMROOT", r"C:\Windows")
        env["SystemRoot"] = env["WINDIR"] = windows
        env["PATH"] = windows + r"\System32;" + windows
        for key in list(env):
            if key.startswith(("PYTHON", "CUDA_PATH", "CONDA")):
                env.pop(key)
        report = folder / "userdata/external-test.json"
        command = [str(folder / "Majan.exe"), "--self-test", "--capture-check",
                   "--image", str(ROOT / "data/screenshots/user-countdown-example/window.png"),
                   "--report", str(report)]
        result = subprocess.run(command, cwd=ROOT / ".tools", env=env, timeout=90)
        data = json.loads(report.read_text(encoding="utf-8"))
        if data["status"] == "passed":
            assert data["checks"]["window_capture"]["status"] == "passed"
            assert data["checks"]["screenshot_advice"]["recommendation"]["event"]["pai"] == "S"
            internal = folder / "_internal"
            assert all(Path(p).is_relative_to(internal) for p in data["sys_path"])
            assert not any("anaconda3" in p.lower() or "/.venv/" in p.replace("\\", "/").lower()
                           for p in data["native_modules"])
        print(json.dumps({"exit_code": result.returncode, "status": data["status"], "error": data.get("error"),
                          "checks": list(data.get("checks", {})), "warnings": data.get("warnings", [])}, ensure_ascii=False))
        return result.returncode
    finally:
        user = _api()
        user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        for name in ("cover", "target"):
            if state.get(name):
                user.PostMessageW(state[name], 0x0010, 0, 0)
        thread.join(2)


if __name__ == "__main__":
    raise SystemExit(main())
