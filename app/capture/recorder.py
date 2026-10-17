from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2

from .frames import WindowCapture, save_capture_config
from .window import CaptureError


def record(hwnd: int, config: dict, output: str | Path, duration: float, fps: float = 15, screenshots_every: float = 3, game_id: str = "unknown", annotation_overlays: bool = False) -> dict:
    output = Path(output)
    if duration <= 0 or not 1 <= fps <= 60 or screenshots_every <= 0:
        raise CaptureError("采集时长、帧率或截图间隔无效。")
    if output.exists() and any(output.iterdir()):
        raise CaptureError("记录目录必须为空，避免覆盖真实回放。")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "recording", "game_id": game_id, "fps": fps, "requested_duration_s": duration,
              "config_version": config["version"], "started_at": datetime.now(timezone.utc).isoformat(),
              "frames": 0, "screenshots": 0, "late_ticks": 0, "information_scope": "screen_pixels_only_no_reference_events"}
    capture, writer, stream = None, None, None
    started = time.perf_counter()
    try:
        capture = WindowCapture(hwnd, config, annotation_overlays)
        snapshot = save_capture_config(output,capture.config)
        report.update(config_sha256=capture.config_sha256,config_snapshot=str(snapshot.resolve()),capture_clock="perf_counter_ns")
        width, height = capture.initial.width, capture.initial.height
        report["client_size"] = [width, height]
        writer = cv2.VideoWriter(str(output / "screen.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise CaptureError("MP4 编码器无法创建回放录像。")
        stream = (output / "frames.jsonl").open("w", encoding="utf-8")
        next_tick, next_screenshot, next_progress = started, started, started+10
        while time.perf_counter()-started < duration:
            if (output / "stop.request").exists():
                report["status"] = "stopped"
                break
            time.sleep(max(0, next_tick-time.perf_counter()))
            frame = capture.grab()
            frame.metadata.update({"game_id": game_id, "recording_time_ms": (time.perf_counter()-started)*1000,
                                   "video_index": report["frames"],"region_config_snapshot":str(snapshot.resolve())})
            writer.write(frame.pixels)
            now = time.perf_counter()
            if now >= next_screenshot:
                frame.save(output / "screenshots")
                report["screenshots"] += 1
                next_screenshot = now + screenshots_every
            stream.write(json.dumps(frame.metadata, ensure_ascii=False) + "\n")
            stream.flush()
            report["frames"] += 1
            next_tick += 1/fps
            if now-next_tick > 1/fps:
                missed = int((now-next_tick)*fps)
                report["late_ticks"] += missed
                next_tick += missed/fps
            if now >= next_progress:
                print(json.dumps({"frames": report["frames"], "elapsed_s": round(now-started), "screenshots": report["screenshots"]}), flush=True)
                next_progress = now+10
        else:
            report["status"] = "finished"
    except KeyboardInterrupt:
        report["status"] = "stopped"
    except Exception as exc:
        report.update({"status": "capture_error", "reason": str(exc)})
    finally:
        if writer is not None:
            writer.release()
        if stream is not None:
            stream.close()
        if capture is not None:
            capture.close()
        report["elapsed_s"] = time.perf_counter()-started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "recording.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
