from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Majan：采集、校准、静态识别与 Mortal CUDA 事件分析")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="读取 mjai JSONL 或严格事件信封")
    analyze.add_argument("input", help="JSONL 文件，- 表示 stdin")
    analyze.add_argument("--player", type=int, choices=range(4), default=0)
    analyze.add_argument("--session", default="cli")
    analyze.add_argument("--envelopes", action="store_true")
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--log-dir", type=Path)
    sub.add_parser("doctor", help="加载、预热并用 profiling 验证 CUDA 后端")
    acceptance = sub.add_parser("acceptance", help="运行固定样本与本机 GPU 性能验收")
    acceptance.add_argument("--benchmark-runs", type=int, default=100)
    sub.add_parser("windows", help="枚举可采集的 Windows 客户区")
    manual = sub.add_parser("manual-capture", help="由用户按按钮的截图采集窗口；不发送游戏输入")
    manual.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    manual.add_argument("--output", type=Path, default=Path("data/screenshots/manual-captures"))
    manual.add_argument("--client-exe", type=Path, default=Path(r"F:\Steam\steamapps\common\MahjongSoul\Jantama_MahjongSoul.exe"))
    manual.add_argument("--game-id", default="")
    manual.add_argument("--split", choices=("train","debug","test"), default="train")
    capture = sub.add_parser("capture", help="保存一张原始客户区截图及采集元数据")
    capture.add_argument("--hwnd", type=int, required=True)
    capture.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--annotation-overlays", action="store_true", help="仅供人工标注，显式记录透明覆盖层警告")
    record = sub.add_parser("record", help="录制真实回放；不发送游戏输入")
    record.add_argument("--hwnd", type=int, required=True)
    record.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--duration", type=float, default=60)
    record.add_argument("--fps", type=float, default=15)
    record.add_argument("--screenshots-every", type=float, default=3)
    record.add_argument("--game-id", required=True)
    record.add_argument("--annotation-overlays", action="store_true", help="仅供人工标注，不能证明无覆盖层")
    calibrate = sub.add_parser("calibrate", help="打开截图标注、区域校准和人工纠错工具")
    calibrate.add_argument("image", type=Path)
    calibrate.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    templates = sub.add_parser("templates", help="从已核对 train 标注构建模板")
    templates.add_argument("annotations", type=Path, nargs="+")
    templates.add_argument("--output", type=Path, default=Path("assets/templates/steam-default/manifest.json"))
    recognize = sub.add_parser("recognize", help="静态截图 → 结构化结果和带框图")
    recognize.add_argument("image", type=Path)
    recognize.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    recognize.add_argument("--output", type=Path, required=True)
    suggest = sub.add_parser("suggest", help="行动计时器截图 → 当前局面 → Mortal 建议（无需历史）")
    suggest.add_argument("image",type=Path)
    suggest.add_argument("--config",type=Path,default=Path("configs/window-steam-1366-v2.json"))
    suggest.add_argument("--review",type=Path,help="与该截图哈希绑定的当前局面修正")
    suggest.add_argument("--output",type=Path,default=Path("reports/screenshot-advice/latest.json"))
    advise = sub.add_parser("advise",help="打开截图/行动计时器自动建议面板")
    advise.add_argument("--config",type=Path,default=Path("configs/window-steam-1366-v2.json"))
    advise.add_argument("--output",type=Path,default=Path("reports/screenshot-advice"))
    advise.add_argument("--image",type=Path)
    evaluate = sub.add_parser("vision-evaluate", help="按独立对局评估静态识别，输出阶段二门槛结果")
    evaluate.add_argument("annotations", type=Path, nargs="+")
    evaluate.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    evaluate.add_argument("--output", type=Path, default=Path("reports/stage2-acceptance.json"))
    args = parser.parse_args()
    try:
        if args.command in {"suggest","advise"}:
            from app.vision.config import load_config
            config = load_config(args.config)
            if args.command == "advise":
                from app.ui.advice import launch
                launch(args.config,args.output,args.image)
                return 0
            from app.advice import ScreenshotAdvice, load_screenshot, save_advice, format_advice
            pixels,metadata = load_screenshot(args.image,config)
            review = json.loads(args.review.read_text(encoding="utf-8-sig")) if args.review else None
            result = ScreenshotAdvice(config).process(pixels,metadata,review)
            save_advice(args.output,result,pixels)
            print(json.dumps({"status":result["status"],"message":format_advice(result),
                              "recommendation":result["recommendation"],"total_ms":result["total_ms"],
                              "output":str(args.output.resolve())},ensure_ascii=False))
            return 0 if result["status"] == "ready" else 3
        if args.command in {"windows", "manual-capture", "capture", "record", "calibrate", "templates", "recognize", "vision-evaluate"}:
            return vision_command(args)
        from app.engine.runtime import OnnxRuntime
        runtime = OnnxRuntime()
        if args.command == "doctor":
            print(json.dumps({"status": "ok", "interface": runtime.interface, **runtime.verification},
                             ensure_ascii=False, indent=2))
            return 0
        if args.command == "acceptance":
            from app.acceptance import run_acceptance
            return run_acceptance(runtime, args.benchmark_runs)
        from app.backend import Backend
        backend = Backend(runtime)
        session = backend.initialize(args.session, args.player)
        stream = sys.stdin if args.input == "-" else Path(args.input).open(encoding="utf-8-sig")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
        logs = {}
        if args.log_dir:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            logs = {key: (args.log_dir / f"{key}.jsonl").open("w", encoding="utf-8")
                    for key in ("confirmed-events", "state-changes", "model-outputs")}
        result_code = 0
        try:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                data = json.loads(line)
                request = data if args.envelopes else {"session_id": args.session, "generation": session.generation,
                            "event_seq": session.event_seq + 1, "state_version": session.state_version, "event": data}
                previous = session.state_version
                response = session.receive(request)
                output.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                if logs and session.state_version > previous:
                    header = {key: response[key] for key in ("session_id", "generation", "event_seq", "state_version", "round_id")}
                    logs["confirmed-events"].write(json.dumps({**header, "event": session.history[-1]}, ensure_ascii=False) + "\n")
                    logs["state-changes"].write(json.dumps({**header, "status": response["status"], "sync_status": response["sync_status"]}, ensure_ascii=False) + "\n")
                    if response["status"] == "ready":
                        logs["model-outputs"].write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
                if response["sync_status"] != "synchronized":
                    print(f"第 {line_number} 行停止：{response['reason']}", file=sys.stderr)
                    result_code = 2
                    break
        finally:
            if stream is not sys.stdin:
                stream.close()
            if output is not sys.stdout:
                output.close()
            for log in logs.values():
                log.close()
        return result_code
    except Exception as exc:
        print(json.dumps({"status": "startup_or_input_error", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


def vision_command(args):
    from app.vision.config import load_config, write_json
    if args.command == "windows":
        from app.capture.window import windows
        print(json.dumps([area.asdict() for area in windows()], ensure_ascii=False, indent=2))
        return 0
    if args.command == "templates":
        from app.vision.templates import build_templates
        print(json.dumps(build_templates(args.annotations, args.output), ensure_ascii=False))
        return 0
    config = load_config(args.config)
    if args.command == "manual-capture":
        from app.ui.manual_capture import launch
        launch(args.config,args.output,args.client_exe,args.game_id,args.split)
        return 0
    if args.command == "calibrate":
        from app.ui.workbench import launch
        launch(args.image, args.config)
        return 0
    if args.command == "capture":
        from app.capture.frames import WindowCapture
        capture = WindowCapture(args.hwnd, config, args.annotation_overlays)
        try:
            frame = capture.grab()
            print(json.dumps({"image": str(frame.save(args.output).resolve()), "metadata": frame.metadata}, ensure_ascii=False))
        finally:
            capture.close()
        return 0
    if args.command == "record":
        from app.capture.recorder import record
        result = record(args.hwnd, config, args.output, args.duration, args.fps, args.screenshots_every, args.game_id, args.annotation_overlays)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["status"] == "capture_error" else 0
    from app.vision.recognizer import StaticRecognizer, overlay
    from app.vision.templates import TemplateBank
    from app.vision.text import TextReader
    recognizer = StaticRecognizer(config, TemplateBank(config["templates"],config.get("classifier_manifest")), TextReader())
    if args.command == "vision-evaluate":
        from app.vision.evaluate import evaluate
        result = evaluate(recognizer, args.annotations)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["gate_passed"] else 3
    from app.capture.frames import load_image, save_image
    pixels = load_image(args.image)
    frame_path = args.image.with_suffix(".frame.json")
    metadata = json.loads(frame_path.read_text(encoding="utf-8")) if frame_path.exists() else {}
    metadata["image_path"] = str(args.image.resolve())
    result = recognizer.recognize(pixels, metadata)
    write_json(args.output, result)
    save_image(args.output.with_suffix(".overlay.png"), overlay(pixels, result))
    summary = {"status": result["status"], "issues": result["issues"], "recognition_ms": result["recognition_ms"]}
    if "tile_classifier" in result:
        summary["tile_classifier"] = result["tile_classifier"]
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
