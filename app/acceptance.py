from __future__ import annotations

from datetime import datetime
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import psutil

from app.backend import Backend
from app.engine.runtime import ROOT, sha256


def hardware():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [{"index": int(row[0]), "name": row[1].strip(), "driver": row[2].strip(),
             "total_mib": float(row[3]), "used_mib_global": float(row[4])}
            for line in result.stdout.strip().splitlines() if (row := line.split(","))]


def libraries(device_id):
    """Query actual loaded runtime APIs; nvidia-smi's CUDA label is not used."""
    if os.name != "nt":
        return {"status": "not_measured", "reason": "This acceptance collector targets Windows."}
    cuda = ctypes.WinDLL("cudart64_12.dll")
    cudnn = ctypes.WinDLL("cudnn64_9.dll")
    version = ctypes.c_int()
    if cuda.cudaRuntimeGetVersion(ctypes.byref(version)) != 0 or cuda.cudaSetDevice(device_id) != 0:
        raise RuntimeError("无法查询实际 CUDA Runtime/设备。")
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        raise RuntimeError("无法查询 GPU 全局可用显存。")
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    paths = {}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    for name, lib in (("cuda", cuda), ("cudnn", cudnn)):
        buf = ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(lib._handle, buf, len(buf)):
            raise ctypes.WinError(ctypes.get_last_error())
        paths[name] = buf.value
    return {"cuda_runtime_api_version": version.value, "cudnn_api_version": cudnn.cudnnGetVersion(),
            "loaded_paths": paths, "device_id": device_id,
            "cuda_mem_total_mib": total.value / 2**20, "cuda_mem_free_mib": free.value / 2**20,
            "memory_scope": "GPU global availability; WDDM does not expose reliable per-process VRAM here"}


def request(session, event):
    return {"session_id": session.session_id, "generation": session.generation,
            "event_seq": session.event_seq + 1, "state_version": session.state_version, "event": event}


def run_acceptance(runtime, benchmark_runs=100):
    if not 10 <= benchmark_runs <= 10000:
        raise ValueError("性能验收需要 10–10000 次测量。")
    report_dir = ROOT / "reports"
    sample_dir = report_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads((ROOT / "data/fixtures/manifest.json").read_text(encoding="utf-8"))
    before_rss = psutil.Process().memory_info().rss / 2**20
    results, decisions, errors = [], [], []
    legal_variants = 0
    for fixture in fixtures:
        backend = Backend(runtime)
        session = backend.initialize(fixture["name"], fixture["player_id"])
        path = ROOT / "data/fixtures" / fixture["file"]
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expects = {item["event_seq"]: item for item in fixture.get("expectations", [])}
        checked, count = [], 0
        with (sample_dir / f'{fixture["name"]}.jsonl').open("w", encoding="utf-8") as stream:
            for event in events:
                response = session.receive(request(session, event))
                stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
                if response["sync_status"] != "synchronized":
                    errors.append({"fixture": fixture["name"], "reason": response["reason"]})
                    break
                if session.event_seq in expects:
                    expected = expects[session.event_seq]
                    ids = {entry["action_id"] for entry in response["legal_candidates"]}
                    semantics = {entry["semantic"] for entry in response["legal_candidates"]}
                    assert response["status"] == "ready"
                    assert set(expected.get("legal_ids", [])) <= ids
                    assert not set(expected.get("illegal_ids", [])) & ids
                    assert set(expected.get("semantics", [])) <= semantics
                    if "exact_legal_ids" in expected:
                        assert ids == set(expected["exact_legal_ids"])
                    if "kan_choice_ids" in expected:
                        kan = next(c for c in response["legal_candidates"] if c["action_id"] == 42)
                        assert {s["tile_action_id"] for s in kan["selection_scores"]} == set(expected["kan_choice_ids"])
                    checked.append(session.event_seq)
                if response["status"] == "ready":
                    count += 1
                    for entry in response["legal_candidates"]:
                        for variant in entry["variants"]:
                            session.state.validate_reaction(json.dumps(variant))
                            legal_variants += 1
                    session.state.validate_reaction(json.dumps(response["recommendation"]["event"]))
                    decisions.append(response)
        results.append({"name": fixture["name"], "source": fixture["source"], "events": len(events),
                        "decision_points": count, "checked_expectations": checked,
                        "passed": session.sync_status == "synchronized" and len(checked) == len(expects),
                        "sha256": sha256(path)})
    if errors:
        raise RuntimeError(f"固定样本验收失败：{errors}")
    benchmark = Backend(runtime).initialize("benchmark", 0)
    for line in (ROOT / "data/fixtures/discard_red.mjai.jsonl").read_text(encoding="utf-8").splitlines():
        benchmark.receive(request(benchmark, json.loads(line)))
    obs, mask = benchmark.state.encode_obs(3, False)
    times = [runtime.run(obs[None], mask[None])[1] for _ in range(benchmark_runs)]
    # Genuine CUDA-unavailable check in a fresh process, with no live CUDA context.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    unavailable = subprocess.run([str(Path(__import__("sys").executable)), "-m", "app.cli", "doctor"],
                                  env=env, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (report_dir / "cuda-unavailable.stderr.txt").write_text(unavailable.stderr, encoding="utf-8")
    if unavailable.returncode == 0:
        raise RuntimeError("GPU 不可用时竟然仍可启动。")
    info = {
        "accepted_at": datetime.now().astimezone().isoformat(), "status": "passed",
        "python": platform.python_version(), "os": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "hardware": hardware(), "loaded_libraries": libraries(runtime.device_id),
        "gpu_verification": runtime.verification, "interface": runtime.interface,
        "fixed_samples": results, "decision_points": len(decisions),
        "validated_action_variants": legal_variants, "recommendation_legal_rate": 1.0,
        "gpu_unavailable": {"exit_code": unavailable.returncode, "startup_stopped": True},
        "performance": {"runs": benchmark_runs, "warm_single_inference_ms": {
            "min": float(min(times)), "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)), "max": float(max(times))},
            "target_p95_ms": 500, "target_met": float(np.percentile(times, 95)) <= 500,
            "decision_total_p95_ms": float(np.percentile([d["total_ms"] for d in decisions], 95)),
            "timing_scope": "session.run until CPU-readable FP32 Q; includes H2D/D2H transfers"},
        "process_rss_mib": {"after_runtime_startup": before_rss, "after_acceptance": psutil.Process().memory_info().rss / 2**20},
        "limitations": ["Constructed prefixes and upstream exported fixtures; no user's full native Mahjong Soul replay supplied.",
                        "Room termination, renchan and scoring differences are not predicted; confirmed results are authoritative.",
                        "WDDM GPU memory is a global measurement including other applications."]}
    (report_dir / "stage1-acceptance.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "decision_points": len(decisions), "legal_rate": 1.0,
                      "gpu_p95_ms": info["performance"]["warm_single_inference_ms"]["p95"],
                      "report": str(report_dir / "stage1-acceptance.json")}, ensure_ascii=False))
    return 0
