2026-10-11 更新（三）：支持 16:9 的更大客户区（1920×1080 等）。识别时整帧先缩放到 1366×768 做区域定位、检测、计时器、OCR 与模板匹配，检测框再按比例映射回原分辨率帧，切出的高分辨率牌面只交给 pjura（`pjura_evidence.input_size` 记录实际输入尺寸）。观测结果记录 `reference_size`、`native_size`、`native_scale`。`scripts/compare_native_resolution.py` 在同一批原分辨率截图上对比“原分辨率裁切”和“缩放到 1366 后裁切”。

2026-10-11 更新（二）：透视校正改为只拟合牌的**白色正面**。原先用亮色牌体轮廓（含牌侧面灰色厚度条）做单应变换，且仅在轮廓呈梯形时触发；侧面被一起拉伸，字形偏移、强制校正时准确率反而下降。现在：按白度把正面与更灰的侧面分开（Otsu 两类明显分离时才分），用与裁切短边成比例的闭运算把被密集字形切开的白色区域连成一片，凸包逼近四角，凸包/四边形面积比不在 0.90–1.12 或不凸时放弃并保留原裁切。对每个旋转假设单独变换：横牌先拉成横向矩形再转正，不再把横牌压成竖向。输出为 64×94 正面直立裁切，形状与识别稳定的手牌/宝牌指示牌一致。

`tile_classifier.rectification` 控制送入方式：`both`（默认，原裁切与校正正面都试，取最高分；两种视图都过阈值但牌种不同则判未知 `pjura_rectification_disagrees`）、`rectified`、`off`（旧行为）。`rectification_skip_roles` 默认 `["hand","draw"]`，手牌/摸牌正对镜头不做校正。牌河/副露每张牌推理数由 3 增至 6。赤五与朝向证据仍读原始像素。证据字段：`pjura_evidence.view`、`views`、`rectified_quad`、`rectification_geometry`。

在已标注截图上，非手牌的 240 张牌中 219 张能拟合正面（牌河 160/166，自家副露 30/30，右家副露 15/27）。真实模型对比：`scripts/benchmark_face_rectification.py`（仅分类器，用标注框）与 `scripts/evidence_tile_posed_corpus.py`（完整识别，off 对 both）。

2026-10-11 更新（一，已被上条替代）：牌面在送入 ViT 前先做透视校正。牌体轮廓实测为梯形（对边长度相对差达到阈值）时，把该四边形单应变换成正面直立 2:3 裁切再识别；直立矩形牌面不重采样，赤五与朝向证据仍读原始像素。受控重投影基准中严重透视下准确率由 0.275 提升到 1.000，冻结验收对局前后完全一致。详见 [牌面透视校正](tile-perspective.md)。

2026-10-10 更新：已接通行动计时器 → 当前截图状态 → Mortal CUDA 建议，新增实时建议面板与核对入口，无需历史文件。用户计时器示例自动返回“打出南”；169 项程序检查通过，12 张无计时器独立截图零误触发。实时整场尚未验收。详情见 [当前截图建议](../docs/screenshot-advice.md)。以下保留此前视觉开发与验收记录。

# pjura 截图识别接入

2026-10-16 固定牌河多边形（draft-25）：默认 `river_polygons` 指向经 SHA256 固定的 `configs/river-polygons.json`。从 train 标定 96 个牌位四角；普通牌的正面直接从原生图像映射到 64×94，减少动态轮廓拟合和旋转候选。现有检测继续确认有牌、牌体和牌位，不将空格送进 34 类网络。多边形须有白面四边支持、至少 85% 与观测牌体重合；仅新裁切超过主门槛（默认 0.9/0.04）且通过五牌颜色校验才走这条路径。其余使用原裁切和原算法，原有 >0.5/3 倍规则仍在那里生效。没有增加 GPU 模型或重新训练；共享现有模型，缓存仍绑定全部输入像素和规则。`fixed_polygon` 记录角点、牌位、来源哈希；`fixed_polygon_fallback` 记录回退。去掉 `river_polygons` 可恢复原流程；修改牌河参考布局需要重新标定或关闭该配置。完整试验及边界见 [报告](../reports/river-polygons/README.md)。

2026-10-16 候选接受规则（draft-24）：默认启用 `tile_classifier.dominant_candidate = {"minimum_confidence":0.5,"minimum_ratio":3.0}`。当首选分数严格高于 0.5 且至少是第二名的 3 倍时，可补充接受原来不足 0.9 的牌种，原 0.04 类间隔要求仍在。第二名是其余 33 类的最大值，由 `score - margin` 还原。记录 `pjura_evidence.class_acceptance` 的规则、第二名和领先倍数；关闭配置项或设为 `null` 可恢复旧门槛。倍数比较限于选中视角内部的类别，不把不同视角分数当作同一个分布；原两种视角都达到 0.9 却牌种不同的冲突检查保留。加速模型在新分数和倍数边界附近使用原 FP32 复核，模型文件、权重与预处理未改。2,457 张单牌的原精度/加速牌种与接受结果一致；新增接受 313 张，其中 1 张错误，详见 [验证与已知反例](../reports/dominant-candidate/README.md)。

2026-10-15 副露结构扩展（draft-23）：新增 `app/vision/meld_geometry.py`，三张副露、白板及四张杠牌统一按画面轴还原角点。支持嵌套模板轮廓和网络已拟合的正面几何，网络牌种/候选旋转不作为朝向证据；完整轮廓来源有矛盾时保留未知。相邻副露可由强候选给出唯一边界，但不放宽牌种确认策略。副露行保留严格直线残差和牌体间距约束，允许侧视投影产生更大倾角。原始/加速模型、输入处理、0.9/0.04 阈值及原精度复核策略均不变。选择、核对和未解决场景见 [副露专项集](../data/meld-orientation/README.md)。

2026-10-15 后续修复（draft-22）：右家吃牌中的横放赤五万受透视影响，牌种已确认而横牌位置未确认。现在对三个非空白牌面增加画面轴的相对几何校验，不将模板或网络的候选旋转角度当成物理朝向。三个拟合正面必须属于同一行，其中两个宽高比差不超过 1.25 倍，第三个至少为它们最大值的 1.5 倍；明确的相反朝向证据不会被覆盖。五牌颜色及来源映射沿用已有检查。复现与范围见 [修复记录](../reports/meld-right-chi-fix/README.md)。

2026-10-09 已默认启用。用户明确要求无论独立截图成绩如何都接入；测试结果用于记录问题，不作为是否启用的门槛。当前接入范围为截图牌种识别，尚未实现识别结果到 Mortal 连续事件的转换。

## 已接入的流程

`静态截图 → 原有区域定位、手牌/牌河/副露检测 → pjura ViT 牌种分类 → 结构化 JSON 与带框预览`

`configs/window-steam-1366-v2.json` 是默认配置，内容版本为 `steam-1366x768-pjura-draft-9`；同内容另存于 `configs/window-steam-1366-pjura.json`。CLI、标注工作台均由 `StaticRecognizer` 自动加载 pjura。工作台复用模型，区域配置变化时才重新初始化。默认使用 CUDA，原有 Mortal 模型及依赖环境保持不变。

模型仅有 34 个基本牌种，不提供赤五、牌背、文字、按钮或检测框类别。原有区域检测与文字 OCR 继续运行；可见牌背仍由已核对模板识别并排除。pjura 的五牌结果只有在现有彩色模板给出相同基本牌种时才能确定普通/赤五，否则保留未知。牌种置信度不能证明横牌朝向；朝向仅复用基本牌种一致的已接受模板证据，没有证据时保持未确认，不推断副露来源。

分类主路径使用 0.9 softmax 阈值与 0.04 类间隔，draft-24 起增加文首的首选领先补充路径。该分数未经概率校准。未满足任何接受路径的结果保留未知，不静默改回模板牌种；输出保留 `pjura_evidence`、`legacy_evidence`、模型 SHA256 和处理原因。相同裁切可复用结果，最多缓存 512 项，完整证据缓存也绑定当前接受规则。牌种分类以 pjura 为主，即使与模板不同也保留双方证据；没有对测试截图训练或微调。牌河、副露、宝牌指示牌会额外送入透视校正后的白色正面（见文首更新），并在 `pjura_evidence` 中记录 `view`、`rectification` 与 `rectified_quad`。

## 模型来源和验证

- 公开分类模型：<https://huggingface.co/pjura/mahjong_soul_vision>。
- 固定版本：`ea01ca0bcb9be0fc10a5fa09ee8356770c25adc5`。
- 原始权重 SHA256：`a3cc566c8011b4496239a5228de3985e37ba2c2dfceefd52f67bb34c5f7e49c0`。
- ONNX SHA256：`f34e4537a7b9ff41ea12fdb4843cf0ec8a6109def22ca3eabb51045b492fc75c`。
- 上游标签：`n` 为万、`p` 为筒、`b` 为索；风与三元牌逐项映射，不把 `rd`（红中）解释为赤五。
- 图像预处理与上游一致：BGR 转 RGB，Pillow 双线性缩放至 224×224，归一化至 [-1,1]。
- 转换环境独立于应用环境。只使用 `torch.load(weights_only=True)` 读取经 SHA256 核验的权重，未加载 `training_args.bin`，未下载助手的弃牌网络。
- ONNX 使用 opset 17、动态 batch、FP32。三张已核对 train 手牌裁切的 batch=1 和 batch=3 与原模型最大 logits 误差均小于 0.000004，预处理误差为 0，Top-1 一致。此项验证转换正确性，不代表真实识别准确率。

详细记录位于 `models/pjura-vision/manifest.json` 和 `reports/vision/pjura-export-validation.json`。启动时核验来源、标签顺序、预处理配方和 ONNX SHA256；缺失、损坏或指定设备不可用时明确停止，不伪称运行 pjura。

## 本地独立截图结果

沿用两场本地 test 对局的 12 张图：9 张稳定、2 张动画、1 张结算遮挡。统计稳定图的已核对目标：

| 项目 | 原模板 draft-8 | pjura 默认路径 |
| --- | --- | --- |
| 手牌 | 111/111 | 111/111 |
| 摸牌 | 2/2 | 2/2 |
| 牌河 | 37/39 | 39/39 |
| 副露牌种 | 9/12 | 9/12 |
| 宝牌指示牌 | 8/8 | 7/8 |
| 赤牌 | 3/4 | 3/4 |
| 文字 | 64/64 | 64/64 |
| 副露类型 / 来源 | 2/4、2/4 | 2/4、2/4 |
| 完整稳定局面 | 0/9 | 0/9 |

剩余 4 个牌种未知；匹配目标中未出现被接受的错误牌种。三张不确定图的相应区域均暂停。阶段二 `gate_passed=false`，但 pjura 按用户要求保持启用。测试样本很小，上游训练对局名单未知，不能证明与上游数据绝无重叠；修正横牌朝向处理时参考了第一轮运行诊断，不能把反复开发后的成绩当作最终盲测。

报告：`reports/vision/pjura-comparison.json`、`pjura-acceptance.json`、`pjura-template-baseline.json`。每张 test 截图的 JSON 和带框图位于 `reports/vision/pjura-test/`，索引记录原图。第一轮报告保留为 `*-first-pass.json`；原阶段二报告保留为 `pjura-prior-stage2.json`。

## 使用与复现

在项目根目录运行默认 pjura 识别：

```powershell
& .\.venv\Scripts\python.exe -m app.cli recognize data/screenshots/user-251228-2240-test/south01-start/00000001.png --output reports/vision/my-pjura.json
& .\.venv\Scripts\python.exe -m app.cli calibrate data/screenshots/user-251228-2240-test/south01-start/00000001.png
```

原模板方案可显式使用 `--config configs/window-steam-1366-templates.json`；默认配置不会因测试门槛未通过而切回原方案。

从公开原始权重重新生成模型（只在独立转换环境安装 Torch）：

```powershell
& .\.venv\Scripts\python.exe scripts/download_pjura.py
& .\.venv\Scripts\python.exe -m venv .tools/pjura-export
& .\.tools\pjura-export\Scripts\python.exe -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
& .\.tools\pjura-export\Scripts\python.exe -m pip install -r scripts/requirements-pjura-export.lock
& .\.tools\pjura-export\Scripts\python.exe scripts/export_pjura.py
& .\.venv\Scripts\python.exe -m scripts.validate_pjura_export
& .\.venv\Scripts\python.exe -m scripts.evaluate_pjura --baseline-config configs/window-steam-1366-templates.json
```

应用无需 Torch/Transformers 新依赖。正常使用已生成的 `models/pjura-vision/model.onnx` 即可；主环境的 `requirements.lock` 保持不变。程序回归检查见 `reports/pjura-pytest.xml`。
2026-10-15 加速更新：默认配置为 **draft-21**，认牌模型改为 ViT 算子融合与混合精度版本，输入、输出和 softmax 保持 FP32。模型清单在 `models/pjura-vision/accelerated/manifest.json`；原始 `models/pjura-vision/model.onnx` 和清单保留，供临界值复核及重建使用。Mortal 决策模型和文字 OCR 沿用原设置。

加速依据是 [ONNX Runtime Transformer 优化](https://onnxruntime.ai/docs/performance/transformers-optimization.html)及 [Float16／混合精度说明](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)。本项目使用已安装的 ONNX Runtime 1.23.2 优化器，先完成基本图优化，再按 12 个注意力头、768 隐层执行 ViT 融合；选中版本图节点 1089→121，文件约 343→172 MB。无需重新训练或在运行环境安装 PyTorch。

FP16 数值会有细微变化。网络置信度接近 0.9、背景/结构恢复分数接近 0.5，或类间隔接近 0.04、结构恢复间隔接近 0.3 时（绝对差不超过 0.005），只将相关裁切交给原始 FP32 模型复核。记录 `precision_recheck=fp32_threshold_boundary`；既有识别阈值、原始/透视视图、旋转候选及副露结构检查保持原样。自动建议开始时同时预热加速模型和原精度复核模型。

加载加速版时校验模型、原始来源清单和对应验证文件的 SHA-256，验证文件必须标明候选哈希一致且通过。该版本需要 CUDA；CPU 调试可将 `tile_classifier.manifest` 指向原始 `models/pjura-vision/manifest.json`。识别输出的 `tile_classifier.optimization` 保存精度与验证来源。

已验证：2,457 张去重公开单牌的首选牌种和确认/拒绝结果与 FP32 一致（2453 张首选正确、2122 张确认、确认错牌 0）；10 张原始截图及两尺寸共 20 个变体的牌种、物理横牌/牌位、局面、编码特征及最终建议一致。部分空白牌或未确认朝向的网络候选旋转角度会变化，作为诊断信息保留，不当作物理朝向证据。公开单牌仍有同作者素材，不是新的独立模型准确率估计；缩小图也不是新增真实局面。

纯网络测试中，16 张一批的中位耗时为 39.26→12.53 毫秒；该项不含图像预处理或临界复核。含全部识别与决策的九张图首次处理中位数为 2.21→1.15 秒。两个实际自动工作线程的截图回放，含 0.3 秒等待、行动检查和 0.1 秒界面轮询预留，约 2.23 秒；真实窗口取图、动画变化和首次模型准备另计。详见 `reports/pjura-acceleration/advice-after.json` 和 `fused-fp16-installed-validation.json`。

重建与复核（在项目根目录，用项目 Python 环境运行）：

```powershell
python -m scripts.build_pjura_candidates
python -m scripts.validate_pjura_acceleration --guard
python -m scripts.promote_pjura_acceleration --activate
python -m scripts.validate_pjura_acceleration --installed
```

完整回归 **503 项通过，0 失败、0 跳过**。来源、测速、对照验证、源码/配置校验值及测试汇总见 [认牌加速记录](../reports/pjura-acceleration/report.json)。重启建议面板后使用新配置。

以下保留此前接入和透视校正的历史记录。

