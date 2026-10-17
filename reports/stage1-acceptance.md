# 阶段一验收报告

验收日期：2026-10-08。结果：**通过已声明范围的阶段一验收**。命令行可独立读取正确 mjai 事件，通过本机 RTX 4070 SUPER 的 CUDA 后端输出合法推荐；已完成全新 Python 环境与空 Rust target 的重建验证。

## 验收结果

| 项目 | 实测结果 / 证据 |
| --- | --- |
| 独立环境 | Python 3.12.7；无 PyTorch、无 CPU onnxruntime 发行包；`requirements.lock` |
| 干净重建 | fresh venv + empty Cargo target 从零构建 libriichi，再加载 CUDA；`clean-environment.json` |
| 扩展来源 | `configs/build.json` 记录源码提交与 pyd SHA256，启动时校验 |
| ONNX 文件 | 远端固定 LFS SHA256 与本地一致；ONNX checker 通过；457 个 initializer 全部 FLOAT；`model-inspection.json` |
| 观察与动作 | v3 `[batch,934,34]` FP32、`[batch,46]` bool、动态批维；Bot 与并行可见状态的观察哈希/掩码逐决策相同 |
| 实际 GPU | 指定 device 0，use_tf32=0；988 次核心节点执行均为 CUDA |
| 完整 profile | 4296 次节点执行在 CUDA；CPU 节点 0；原始 profile 链接见 JSON |
| GPU 不可用 | 新进程设置 CUDA_VISIBLE_DEVICES=-1，启动返回退出码 2；没有 CPU 推理回落 |
| 固定样本 | 22 个样本视角，含 14 个构造样本和 2 份上游导出牌谱 × 4 座位 |
| 合法动作 | 247 个决策点，2187 个具体候选通过 libriichi；推荐合法率 100% |
| 自动检查 | 34 项通过，0 失败、0 错误；`pytest.xml` |
| 隐藏信息 | 对手暗手/摸牌任意替换后，观察哈希、候选 Q 值和推荐保持相同；传入 oracle 特征被拒绝 |
| 序号与版本 | 重复不更新、不重推理；缺失/乱序/版本冲突停止建议；reset generation 阻止旧事件 |
| 会话与重置 | 会话独立状态、共享常驻模型；结束后 reset 可开始新局；建议不会执行到实际状态 |

覆盖普通与赤牌弃牌、三种吃位置、赤/普通 consumed 选择、碰、明杠、暗杠、加杠、多个杠选择、补牌、宝牌、立直宣言/弃牌/成立、荣和、自摸、跳过、九种九牌、结束与重置。每个固定场景的独立预期在 `data/fixtures/manifest.json`；具体状态、候选和输出在 `samples/`。

## 性能与资源

| 测量 | 本次结果 |
| --- | --- |
| 会话创建 | 226.61 ms |
| 首次 GPU 推理 | 250.50 ms |
| 预热后单次推理中位数 | 9.20 ms |
| 预热后单次推理 P95 | **10.17 ms**，目标 ≤ 500 ms，通过 |
| 预热后单次推理最大值 | 12.59 ms |
| 性能样本数 | 100 次固定可见局面推理 |
| 完整决策 P95 | 12.95 ms |
| 进程 RSS：GPU 启动后 | 866.52 MiB |
| 进程 RSS：验收后 | 922.50 MiB |
| GPU 全局已用 / 总显存 | 3071 / 12282 MiB |
| 驱动 | 617.42 |

计时截止于 CPU 可读取 Q 数组，包含必要数据传输和同步。完整决策时间另含特征编码与动作校验。本机还有其他 GPU 工作负载，数值代表本次测量，不是长期性能保证。WDDM 下显存为全局值，不能等同模型单进程显存；CUDA 实际运行库版本、加载路径和全局可用显存均记录在 JSON。

## 规则与样本边界

第一阶段适配范围是正确确认事件的局内动作。连庄、终局和结算按参考牌谱确认结果驱动；没有用天凤 arena 预测雀魂房间终局。西入、三人、东风、自定义赤牌和未验收房间选项仍暂停建议。详细处理见 `rule-differences.md` 与 `configs/rules.json`。

雀魂导出转换样本来自固定版 mjai-reviewer 的 Gold Room South 测试，已验证导出 Tenhou6 → convlog → mjai → GPU 建议路径。没有用户原生 protobuf 牌谱或自有完整半庄，本报告不把这些路径声称为已验证。

人工构造样本是专项决策事件前缀，不作为完整真实比赛记录。部分样本合法结束后重置，其他前缀仅验证相应决策点。阶段二的截图识别和阶段三的连续画面验收没有在本阶段执行。

## 交付物检查

- [x] GPU 环境安装说明及固定依赖版本：`docs/environment.md`、`scripts/setup.ps1`、`requirements.lock`。
- [x] 模型配置、来源、校验、模型卡：`configs/model.json`、`models/manifest.json`、`models/README.md`。
- [x] 可读取 mjai 并输出建议的 CLI：`python -m app.cli analyze`；示例 `example-cli.jsonl`。
- [x] ONNX GPU 适配层与统一动作：`app/engine/`、`app/backend.py`。
- [x] CUDA 验证、原始 profiling、显存与内存记录：`stage1-acceptance.json`、`profiles/`、`nvidia-smi.csv`。
- [x] 固定事件、规则差异、验收及测试记录：`data/fixtures/`、本报告、`pytest.xml`。

下一阶段可以在此后台上接静态截图识别，后台仍可用固定牌谱独立验收。
