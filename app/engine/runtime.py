from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class EnvironmentError(RuntimeError):
    """Startup cannot prove the requested GPU/feature contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_profile(path: Path) -> dict:
    nodes = [item for item in json.loads(path.read_text(encoding="utf-8"))
             if item.get("cat") == "Node" and item.get("args", {}).get("provider")]
    core_ops = {"Conv", "FusedConv", "MatMul", "Gemm", "FusedMatMul"}
    core = [item for item in nodes if item["args"].get("op_name") in core_ops]
    if not core or any(item["args"]["provider"] != "CUDAExecutionProvider" for item in core):
        raise EnvironmentError("Profiling 未证明全部核心 Conv/MatMul/Gemm 在 CUDA 执行；停止启动。")
    counts: dict[str, int] = {}
    cpu: dict[str, int] = {}
    for item in nodes:
        args = item["args"]
        provider = args["provider"]
        counts[provider] = counts.get(provider, 0) + 1
        if provider != "CUDAExecutionProvider":
            label = f'{args.get("op_name")}:{item["name"]}'
            cpu[label] = cpu.get(label, 0) + 1
    return {"path": str(path.resolve()), "provider_node_executions": counts,
            "core_node_executions": len(core), "core_all_cuda": True,
            "non_cuda_nodes": cpu}


class OnnxRuntime:
    def __init__(self, config_path: Path | str = ROOT / "configs/model.json"):
        import onnxruntime as ort
        import libriichi
        from libriichi.consts import ACTION_SPACE, obs_shape

        self.lock = threading.RLock()
        self.failed = False
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg = self.config
        self.device_id = cfg["device_id"]
        if type(self.device_id) is not int or self.device_id < 0 or cfg["use_tf32"] != 0:
            raise EnvironmentError("需要有效 GPU 编号和 use_tf32=0 的 FP32 配置。")
        if cfg["observation_version"] != 3 or obs_shape(3) != (934, 34) or ACTION_SPACE != 46:
            raise EnvironmentError("libriichi 特征编码与固定 v3 模型不兼容。")
        installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
        if "onnxruntime" in installed or "torch" in installed:
            raise EnvironmentError("请使用独立环境，仅安装 onnxruntime-gpu 和 NVIDIA 运行库，不安装 CPU 包/PyTorch。")
        if ort.__version__ != "1.23.2":
            raise EnvironmentError("ONNX Runtime 版本必须是锁定的 1.23.2。")
        manifest = json.loads((ROOT / cfg["manifest_path"]).read_text(encoding="utf-8"))
        build = json.loads((ROOT / cfg["build_manifest_path"]).read_text(encoding="utf-8"))
        if build["mortal_commit"] != manifest["source_commit"] or sha256(Path(libriichi.__file__)) != build["libriichi_sha256"]:
            raise EnvironmentError("已加载的 libriichi 扩展与固定源码构建记录不匹配，请运行 build.ps1。")
        overlay = build.get("snapshot_overlay")
        if overlay is not None:
            if sha256(ROOT / "patches/mortal-snapshot/manifest.json") != overlay["manifest_sha256"] or any(
                    sha256(ROOT / name) != digest for name, digest in overlay["files"].items()):
                raise EnvironmentError("当前局面适配源码与构建记录不一致，请重新构建。")
        model_path = ROOT / cfg["model_path"]
        meta_path = ROOT / cfg["metadata_path"]
        for path in (model_path, meta_path):
            if sha256(path) != manifest["files"][path.name]["sha256"]:
                raise EnvironmentError(f"文件校验失败：{path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["version"] != 3:
            raise EnvironmentError("模型元数据的观察版本不兼容。")
        self.model_version = f'{meta["tag"]}@{manifest["revision"]}'
        self.dll_handles = []
        if os.name == "nt":
            # cuDNN itself uses LoadLibrary for its sublibraries; PATH is needed in
            # addition to Python's DLL directories. Only this process is changed.
            dll_dirs = sorted((Path(ort.__file__).parent.parent / "nvidia").glob("*/bin"))
            self.dll_handles = [os.add_dll_directory(str(path)) for path in dll_dirs]
            os.environ["PATH"] = os.pathsep.join(str(path) for path in dll_dirs) + os.pathsep + os.environ.get("PATH", "")
        # An empty directory explicitly searches NVIDIA site-packages, never torch.
        ort.preload_dlls(directory="")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise EnvironmentError("CUDAExecutionProvider 不可用，请检查 GPU 包与独立 CUDA/cuDNN DLL。")
        options = ort.SessionOptions()
        options.enable_profiling = True
        profile_dir = ROOT / cfg["profile_dir"]
        profile_dir.mkdir(parents=True, exist_ok=True)
        options.profile_file_prefix = str(profile_dir / f"cuda-{time.time_ns()}")
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        cuda = {**cfg.get("cuda_options", {}), "device_id": str(self.device_id), "use_tf32": "0"}
        started = time.perf_counter()
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                                providers=[("CUDAExecutionProvider", cuda)])
            self.session.disable_fallback()
            actual = self.session.get_providers()
            if not actual or actual[0] != "CUDAExecutionProvider":
                raise EnvironmentError("CUDA DLL 加载失败，运行时回落 CPU；停止启动。")
            applied = self.session.get_provider_options()["CUDAExecutionProvider"]
            if int(applied["device_id"]) != self.device_id or applied["use_tf32"] != "0":
                raise EnvironmentError("会话实际 GPU 或 TF32 配置与请求不一致。")
            self._validate_interface()
            self.creation_ms = (time.perf_counter() - started) * 1000
            warmup = []
            for index in range(cfg["warmup_runs"] + 1):
                _, elapsed = self.run(np.zeros((1, 934, 34), dtype=np.float32),
                                      np.ones((1, 46), dtype=np.bool_))
                warmup.append(elapsed)
            profile_path = Path(self.session.end_profiling())
            self.verification = read_profile(profile_path)
        except Exception as exc:
            raise EnvironmentError(f"GPU 初始化/验证失败：{exc}") from exc
        self.verification.update({"available_providers": available, "session_providers": actual,
                                  "cuda_options": applied, "creation_ms": self.creation_ms,
                                  "first_inference_ms": warmup[0], "warmup_ms": warmup[1:],
                                  "model_version": self.model_version, "gpu_id": self.device_id})

    def _validate_interface(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 2 or len(outputs) != 1:
            raise EnvironmentError("模型输入/输出数量不匹配。")
        by_name = {value.name: value for value in inputs}
        if set(by_name) != {"obs", "mask"}:
            raise EnvironmentError("模型输入必须是 obs 和 mask。")
        for name, shape, dtype in (("obs", [934, 34], "tensor(float)"),
                                   ("mask", [46], "tensor(bool)")):
            item = by_name[name]
            if len(item.shape) != len(shape) + 1 or item.shape[1:] != shape or item.type != dtype:
                raise EnvironmentError(f"{name} 的形状/类型不匹配：{item.shape} {item.type}")
            if isinstance(item.shape[0], int):
                raise EnvironmentError("批维必须动态，杠选择需要批量推理。")
        out = outputs[0]
        if len(out.shape) != 2 or out.shape[1] != 46 or out.type != "tensor(float)":
            raise EnvironmentError("输出必须是 FP32 [batch,46] 模型评分。")
        self.output_name = out.name
        self.interface = {"inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in inputs],
                          "outputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in outputs]}

    def run(self, observations: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, float]:
        if self.failed:
            raise EnvironmentError("GPU 会话异常后已停止，请重新初始化。")
        if observations.dtype != np.float32 or observations.shape[1:] != (934, 34):
            raise ValueError("观察必须是 FP32 [batch,934,34]，禁止强行调整形状。")
        if masks.dtype != np.bool_ or masks.shape != (len(observations), 46):
            raise ValueError("合法动作掩码必须是 bool [batch,46]。")
        if not np.isfinite(observations).all() or not masks.any(axis=1).all():
            raise ValueError("观察有非有限数或动作掩码为空。")
        with self.lock:
            started = time.perf_counter()
            try:
                q = self.session.run([self.output_name], {"obs": np.ascontiguousarray(observations),
                                                          "mask": np.ascontiguousarray(masks)})[0]
                # session.run returns host arrays, so timing includes transfers and GPU completion.
                elapsed = (time.perf_counter() - started) * 1000
                if q.shape != masks.shape or not np.isfinite(q[masks]).all():
                    raise EnvironmentError("模型输出形状错误或合法评分非有限。")
                if not np.isneginf(q[~masks]).all():
                    raise EnvironmentError("ONNX 未按约定屏蔽非法动作。")
                if self.session.get_providers()[0] != "CUDAExecutionProvider":
                    raise EnvironmentError("执行后端发生 CPU 回落。")
                return q, elapsed
            except Exception:
                self.failed = True
                raise
