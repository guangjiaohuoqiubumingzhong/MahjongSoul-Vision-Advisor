# 阶段一 GPU 环境记录

验收日期：2026-10-08。平台：Windows x64。实际版本保存在 `reports/stage1-acceptance.json`、`configs/build.json` 与 `requirements.lock`。

| 组件 | 固定版本 / 本机结果 |
| --- | --- |
| Python | 3.12.7 x64，独立 `.venv` |
| Rust | 1.90.0，x86_64-pc-windows-msvc |
| Rustup | 1.29.1；安装器来源及 SHA256 在 `configs/sources.json` |
| MSVC | Visual Studio 2022 Build Tools，14.44.35207 |
| SDK 链接库 | Microsoft.Windows.SDK.CPP.x64 10.0.26100.9169，官方 NuGet 包 |
| NumPy | 2.2.6 |
| ONNX Runtime GPU | 1.23.2 |
| CUDA Runtime | NVIDIA wheel 12.9.79；实际 API 返回 12090（CUDA 12.9） |
| cuDNN | NVIDIA wheel 9.27.0.42；实际 API 返回 92700（cuDNN 9.27.0） |
| cuBLAS | 12.9.2.10 |
| CUDA NVRTC | 12.9.86 |
| GPU | NVIDIA GeForce RTX 4070 SUPER，编号 0，12282 MiB |
| 本机驱动 | 617.42 |
| Mortal 提交 | 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2 |
| mjai-reviewer 提交 | 2dc5ec5c8b28517cfb45f57eb21536d9a8f67aa9 |

采用 [ONNX Runtime 官方 CUDA 兼容表及 DLL 预加载方法](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。实际 CUDA/cuDNN 版本通过已加载 DLL 的 API 查询，并记录模块完整路径；没有使用驱动展示的 CUDA 支持版本代替运行库版本。

`onnxruntime-gpu`、NVIDIA CUDA 和 cuDNN 以独立 wheel 安装。环境没有 `torch` 或 CPU 发行包 `onnxruntime`。所有直接与间接 Python 依赖都固定在 `requirements.lock`；Rust 使用源码内的 Cargo.lock 和转换 CLI 的 Cargo.lock，构建时启用 `--locked`。

## 构建

先安装 Python 3.12.7、Git 和 Visual Studio 2022 C++ Build Tools，然后执行 README 中的 `setup.ps1`。脚本不会修改系统 PATH；Rust 安装使用 `--no-modify-path`，运行库 DLL 搜索路径仅在当前后台进程内添加。

本机有 MSVC 链接器，但系统没有可用 Windows SDK。通过官方 NuGet x64 包把 UM/UCRT 链接库放入 `.tools/windows-sdk`，以 `CARGO_ENCODED_RUSTFLAGS` 传入链接路径。这也支持项目路径中的空格。

构建命令的实质是：

```text
cargo build --manifest-path third_party/Mortal/Cargo.toml -p libriichi --lib --release --locked --no-default-features --features pymod
cargo build --manifest-path tools/convlog-cli/Cargo.toml --release --locked
```

禁用默认 mimalloc，使用 Rust 标准分配器，保留 `pymod`。此构建不需要额外 C 项目或 PyTorch。libriichi 现增加当前局面导入接口，覆盖文件及原件 SHA-256 保存在 `patches/mortal-snapshot/manifest.json`；构建脚本检查并应用这一已记录覆盖，拒绝其它源码改动。把 `riichi.dll` 复制为 `.venv/Lib/site-packages/libriichi.pyd` 后执行导入检查，记录扩展文件及覆盖清单 SHA256。模型启动时同时校验构建覆盖、扩展文件和模型文件 SHA256。

`scripts/verify-clean.ps1` 在全新虚拟环境和空 Rust target 目录中重新安装锁定依赖、编译规则扩展，并执行模型加载和 profiling。结果在 `reports/clean-environment.json`。

## CUDA 初始化与失败处理

显式使用 `CUDAExecutionProvider`、device 0、FP32、`use_tf32=0`。模型需要动态批维：`obs: float32 [batch,934,34]`，`mask: bool [batch,46]`，`q: float32 [batch,46]`。形状或编码版本不符直接拒绝；不调整数组尺寸迁就模型。

ONNX Runtime 的 `preload_dlls(directory="")` 只从 NVIDIA site-packages 加载独立运行库。Windows 下额外把 NVIDIA wheel 的 bin 目录加到本进程 PATH 并保留 `os.add_dll_directory` 句柄，以便 cuDNN 自己加载其子库。

本次首次运行遇到 `cudnn_engines_tensor_ir64_9.dll` 无法加载。DLL 已存在，但仅预加载顶层库不够；补全上述进程内路径后解决。GPU 启动失败立即终止，从未采用 CPU 推理绕过此故障。

会话创建后检查实际 provider 与 provider options，并禁用 Python 层运行失败回落。首次执行和预热后解析 profiling，要求所有核心 Conv/MatMul/Gemm 在 CUDA 执行；记录其他节点归属。当前模型的预热 profile 中所有 4296 次节点执行均在 CUDA，988 次核心节点执行均在 CUDA，CPU 节点为零。

遇到缺少 DLL、GPU 不可用、核心节点落到 CPU、校验失败或编码不匹配时，`doctor` / `analyze` 返回退出码 2。运行期间 GPU 调用异常会把 runtime 标记为失败，后续调用停止，需显式重新初始化。

## 测量口径

`session.run` 返回主机可读取 Q 数组后停止计时，包含上传、GPU 执行、同步和下载。完整决策耗时另计规则更新、特征编码、转换与合法性检查。100 次重复实测固定决策状态，不使用异步提交时间替代完成时间。

Windows WDDM 下无法可靠取得单进程显存，本项目记录 nvidia-smi 的全局已用显存和 CUDA API 的全局可用显存；其中包含其他应用，不能把全局值称为本模型占用。进程内存使用 psutil RSS。详细数值保存在验收 JSON。

模型与运行库共需数 GB 磁盘，干净环境复验另保留一套隔离环境。源码与工具下载均使用 `configs/sources.json` 中固定版本；模型远端 LFS SHA256 与本地一致。
