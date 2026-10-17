# 开发记录与技术说明

## 最新便携版

[下载 Windows NVIDIA 便携测试版 2](https://github.com/guangjiaohuoqiubumingzhong/majang/releases/tag/v0.1.0-portable-test2)。下载两个 ZIP 分片及 `Merge-Package.cmd`，放在同一目录，双击脚本合并并校验，再完整解压并运行 `Majan/Majan.exe`。无需另装 Python；需要 Windows x64、兼容 CUDA 12.x 的 NVIDIA 显卡和驱动。测试版先保持游戏窗口可见；独立程序的后台遮挡采集仍有兼容限制。

本仓库保存应用、脚本、测试、配置、识别资源和模型来源清单。大模型和 CUDA 运行库随便携版提供；原始截图数据集、运行日志和开发环境不放入源码仓库。源码重建与数据范围见 [SOURCE-CONTENTS.md](../SOURCE-CONTENTS.md)。

2026-10-17 界面应用 C 款米白和风：米白卡片、墨绿主按钮、金色细线与牌面建议。去掉截图预览，保留四个按钮及后台取图；Top 5 以相对百分比和进度条显示，支持默认 560×640 及 450 像素窄面板。重新打开 `启动截图建议.cmd` 生效。[实际界面与检查](../reports/ui-ivory/README.md)。

2026-10-16 固定牌河多边形（draft-25）：按固定牌位标定白色正面的四角，普通牌先拉正后识别，保留牌体/空位检查，横牌或不确定裁切回到原流程。已有核对集正确结果保持，14 个尺寸场景的局面、动作与评分不变；预热后截图到建议处理中位数约 1.22→0.92 秒，用户八筒分数约 88.85%→94.52%。公开整桌图同时做变化检查，仍有未解决项，详见 [实验、失败与采用依据](../reports/river-polygons/README.md)。重启面板生效。

2026-10-16 候选接受规则（draft-24）：增加“首选分数严格超过 50%，且至少为第二名的 3 倍”的补充路径，原 90% 路径、赤五与结构检查保留。用户倒计时 282 的截图在两个尺寸下均恢复建议。2,457 张公开单牌接受数 2,122→2,435，同时新增 1 张已知误接受；这些分数未经概率校准，放宽规则不能保证正确。规则、反例及完整回放见 [验证记录](../reports/dominant-candidate/README.md)。重启面板生效。

2026-10-15 副露专项扩展（draft-23）：联网比较公开数据后，建立覆盖四个座位的 47 个真实局部案例与 672 个受控投影案例。统一白板、一般牌与杠牌的角点顺序和牌面朝向判断；支持在多组相邻副露中保留清楚的分组，未知候选只辅助定位边界。专项 46 组真实副露正确确认 28→32，错误确认 0；尚有低清字形和特殊牌面失败保留未确认，详见 [完整数据与失败清单](../reports/meld-dataset/index.html)。重启面板生效。

2026-10-15 右家吃牌修复（draft-22）：透视下横放的牌体可能仍呈竖长形，补充同组白色牌面的相对宽高判断，先还原牌角的画面坐标顺序，再确认唯一横牌。用户“293 秒”截图恢复出牌建议，赤五万及来源保留；复现与结果见 [修复记录](../reports/meld-right-chi-fix/README.md)。重启面板生效。

已接通当前截图建议：自己的行动计时器出现（不限数字位数，每次出现等待 0.3 秒后分析当前窗口图像，并保存选中的一帧）→ pjura 识别当前手牌和公开信息 → 后台当前局面 → Mortal ONNX CUDA 合法建议，不需要历史文件。计时器示例图已无人工修正返回“打出南”。双击 `启动截图建议.cmd` 打开面板，选择 1366×768 或 1920×1080 后点击“开始自动建议”；默认自动调整游戏窗口和摆放建议面板。吃碰/和牌响应自动识别按钮与当前弃牌高亮。自己的牌河支持第三、四排和立直横牌后的偏移。用法和限制见 [截图建议说明](screenshot-advice.md)、[链路验证报告](../reports/snapshot-advice/acceptance.json)。原已验收的 mjai 事件接口继续可用。

最新开发进度见 [开发流程与当前进度](../雀魂AI助手开发流程.md)。目前核心建议链路和面板已经运行，开发重点是补齐真实对局中的牌河、副露与响应识别。最新已修复右侧白板碰牌的方向和来源判断，用户两张图在两个尺寸下均能直接给出弃牌建议。视觉使用方法及数据说明见 [视觉开发说明](vision.md)。

2026-10-15 认牌网络加速：默认 draft-21 使用 ViT 融合与混合精度，临界分数自动用原始 FP32 复核。16 张牌面的纯推理约 39.3→12.5 毫秒；九张截图首次处理中位数约 2.21→1.15 秒，含 0.3 秒等待与检查的自动工作流回放约 2.23 秒（真实窗口取图及首次准备另计）。2,457 张单牌与 20 个尺寸变体的牌种、确认结果和最终局面/建议保持一致。[方法、来源与重建](pjura.md)、[测速记录](../reports/pjura-acceleration/advice-after.json)。重启面板生效。

2026-10-15 速度优化：增加完整牌面证据缓存、提前预热模型、建议先显示后保存，并给补截设置约 5 秒的调度预算；新增累计耗时记录。9 张原始截图首次处理的中位数由 3.00 降到 2.21 秒；真实截图驱动的两个自动建议工作流，含 0.7 秒等待及界面轮询预留，约为 3.8 秒（不含真实窗口采集，首次模型准备另计）。18 次识别与推荐未改变。[测速与范围](screenshot-advice.md#响应速度2026-10-15)、[测速数据](../reports/advice-performance/after.json)。重启面板生效。

2026-10-15 界面精简：主面板只保留“刷新雀魂窗口、整理窗口、开始自动建议、刷新截图”四个按钮，删除底部说明及自动整理勾选框。候选操作只显示前五项，Q 值转为合计 100% 的相对评分，最低显示 0.1%；百分比不表示胜率，原始 Q 值仍保存在日志中。重启面板后生效。

2026-10-15：修复上家立直牌河不完整。横放东牌的厚侧面不再导致误拆成两张；前一张七筒的裁切框排除邻牌图案，按原模型及阈值重新识别。用户原始 1920×1080 图及其 1366×768 缩小图均恢复 16 张上家弃牌、保留立直，并直接给出合法建议。详见 [立直横牌与邻牌裁切修复](../reports/river-left-late-fix/report.html)。

2026-10-13 行动触发修复：提前取消吃碰、分析未完成时先出牌，不再让下一次行动漏掉提示；所有检查更新倒计时，并结合手牌/摸牌与自己的牌河区分行动，丢弃过期结果。新增“截图并分析”按钮，直接重新获取所选窗口，完成后继续已开启的自动建议。两张用户图在两个尺寸下均能区分摸牌与响应状态；检查记录见 [行动触发与手动截图修复](../reports/action-transition-fix/README.md)。

2026-10-12：自动建议优先通过 Windows.Graphics.Capture 读取游戏窗口图像，其他窗口遮挡时无需切回游戏；窗口最小化时暂停，恢复后重新取图。完全遮挡后更新画面的窗口测试已通过，前台焦点保持不变；结果见 [窗口采集验证](../reports/window-content-capture/README.md)。

2026-10-13：用公开数据集测试当前网络，2,457 张去重单牌首选正确率 99.84%；修复较小同布局截图的尺寸拒绝、副露边界牌背误判、彩色牌边造成的漏牌，以及横放牌的不合理旋转候选。35 张人工核对牌面的小规模回归中错误确认从 1 张降到 0 张、漏牌从 1 张降到 0 张，仍有 4 张未知；全部 343 项测试通过。修改前后记录见 [公开数据集测试与修复报告](../reports/public-vision-fix/report.html)。

2026-10-13 横牌后续修复：实际截图中的横放四索与八索已正确识别；此前误记为二索的人工标注已纠正。增加只针对未确认横牌的字形投票校验，使用 1,029 张公开训练图，排除测试与外部图片的完全重复哈希；白板不从字形补判。外部图片的 2,076 个受控横放样本中，正确识别 1,371→1,771，错误确认均为零；全部 357 项测试通过。详见 [横放牌修复报告](../reports/public-sideways-fix/report.html)。

2026-10-13：修复用户三筒暗杠截图。两端蓝纹牌背、中间同牌及同排几何共同确认暗杠，并接入当前局面及四张牌数量检查；被箭头遮住的一筒可在原模板、三个辅助分类器及网络候选一致时恢复。两张原始 1920×1080 图均无需修正返回合法建议，三筒暗杠缩到 1366×768 后同样通过；另一张缩小图仍有牌河置信度不足，保留核对。详见 [杠牌与箭头遮挡修复报告](../reports/kan-recognition-fix/report.html)。

2026-10-13 杠牌后续修复：发财加杠已根据叠牌和底部碰牌结构确认，用户图在 1920×1080 与 1366×768 下均无需修正返回合法建议。补充四家明杠来源、加杠追加层、薄牌检测、粘连副露分离和橙色暗杠牌背。新增 212 个公开原图文件（去重新增 207 张）；13 个局部核对样本正确 2→11、错误确认 1→0，两张暗色或特殊红背图仍未确认。完整 474 项测试通过；样本、来源和边界见 [四家杠牌识别报告](../reports/kan-all-seats-fix/report.html)。

当前默认配置为 `steam-1366x768-pjura-snapshot-draft-25`。2026-10-09 的历史识别基线：两场对局的 12 张图中，手牌 111/111、摸牌 2/2、文字 64/64、牌河 39/39、副露 9/12、指示牌 7/8；该统计不包含之后的实战修复，不作为最新功能进度。旧模板配置与历史结果保留。模型来源、使用及重建步骤见 [pjura 接入说明](pjura.md)，历史对照见 [识别比较报告](../reports/vision/pjura-comparison.json)。

识别已有截图并保存 JSON 与带框图：

```powershell
& .\.venv\Scripts\python.exe -m app.cli recognize data/screenshots/user-251228-2240-test/south01-start/00000001.png --output reports/vision/my-pjura-frame.json
```

人工采集入口：`python -m app.cli manual-capture --game-id user-250819-2005 --split train`。当前默认配置自动适用于 CLI 和标注工具；历史截图的采集配置副本保持原样。

## 在本机运行

在 PowerShell 中进入项目目录：

```powershell
Set-Location D:\py_project\Majan
& .\.venv\Scripts\python.exe -m app.cli doctor
& .\.venv\Scripts\python.exe -m app.cli analyze data/fixtures/discard_red.mjai.jsonl --player 0
```

分析已转换的雀魂导出样本，保存输出和可追溯日志：

```powershell
& .\.venv\Scripts\python.exe -m app.cli analyze data/fixtures/majsoul-gold-south.mjai.jsonl --player 3 --output reports/my-analysis.jsonl --log-dir reports/my-log
```

每条确认事件输出一行 JSON；`ready` 才有推荐。`model_score` 是模型 Q 值。`recommendation.event` 给出实际牌、赤牌、副露来源和消耗组合；立直先返回宣言，收到确认宣言后才给出合法宣言弃牌。杠动作包含具体选择及独立选择评分。

## 重建环境

需要 Windows x64、Python **3.12.7**、Git、Visual Studio 2022 C++ Build Tools 和可用 NVIDIA 驱动。所有 Python 包、Rust 工具链、CUDA/cuDNN、SDK 库和模型都放在项目的独立目录中。

```powershell
& .\scripts\setup.ps1 -PythonExe 'D:\anaconda3\python.exe' -VsDevCmd 'D:\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
```

其他机器用相应的 Python 和 `VsDevCmd.bat` 路径替换参数。此脚本使用锁定源码和依赖，校验模型，编译规则扩展与转换器，并执行实际 GPU 验证。详细版本、运行库检查和故障修复见 [环境说明](environment.md)。无需安装 PyTorch。

重新执行阶段一验收：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/pytest.xml
& .\.venv\Scripts\python.exe -m app.cli acceptance --benchmark-runs 100
& .\scripts\verify-clean.ps1
```

`verify-clean.ps1` 创建全新的虚拟环境和 Rust target 目录，从零编译并验证 CUDA；每次执行会保留独立验收环境以便检查。

## 雀魂牌谱转换

接收 mjai-reviewer 文档中所述的 **Tenhou6 JSON 格式的雀魂导出文件**。原生 `.lq.*` protobuf 文件和仅含牌谱 URL 的文本不直接支持。当前转换器拒绝三人、东风和非每色一枚赤五规则。

```powershell
& .\scripts\convert.ps1 -InputFile 'D:\path\export.json' -OutputFile 'D:\path\game.mjai.jsonl'
& .\.venv\Scripts\python.exe -m app.cli analyze 'D:\path\game.mjai.jsonl' --player 0
```

固定的雀魂金之间南场导出样本来自 mjai-reviewer 上游测试；另有天凤三局样本。两者均对四个玩家视角逐事件验收。来源和边界见 [规则差异记录](../reports/rule-differences.md)。

## 连续事件接口

离线正确 JSONL 可直接导入，CLI 自动分配序号。实际事件生产者必须使用严格信封，保留其真实连续序号；仅靠重新编号的原始牌谱不能证明没有漏事件。

```json
{"session_id":"demo","generation":0,"event_seq":1,"state_version":0,"event":{"type":"start_game","names":["A","B","C","D"]}}
```

```powershell
& .\.venv\Scripts\python.exe -m app.cli analyze 'D:\path\envelopes.jsonl' --session demo --player 0 --envelopes
```

`state_version` 是输入之前的状态版本，成功接收后增加一。`event_seq` 从 1 连续增长；完全相同的重复事件返回 `duplicate`，不重推理、不返回旧建议。缺失、乱序、重复内容冲突和版本冲突会停止建议。`reset()` 清空历史并增加 `generation`，旧 generation 事件不能进入新对局。

Python 接口为 `OnnxRuntime()`、`Backend.initialize(session_id, player_id)`、`Session.receive(envelope)`、`Session.decide(state_version)` 和 `Session.reset()`。开局和结束由已确认的 `start_game` / `start_kyoku` / `end_kyoku` / `end_game` 事件驱动。多个会话保持独立状态，共用一个常驻 CUDA 模型；模型调用串行化，保留正确的状态和输出归属。

对手开局暗手、摸牌以及 seed/oracle/额外元数据在进入状态之前移除，转换为 `?`。自己的未知手牌或摸牌会暂停。每个候选都由规则掩码产生，并验证具体动作；建议永远不会作为实际事件写回状态。

日志分为确认事件、状态变化和模型输出三个文件。第一阶段没有视觉识别记录。CLI 输入冲突或环境失败以退出码 2 结束，保留最后确认状态；可通过 Python API 显式重置并从正确历史重放。

## 交付与验收

- [环境说明](environment.md)、[完整依赖锁](../requirements.lock)、[构建来源](../configs/build.json)。
- [模型来源及校验](../models/manifest.json)、[原始模型卡](../models/README.md)、[模型配置](../configs/model.json)。
- [固定事件与独立期望](../data/fixtures/manifest.json)、[动作和规则边界](../reports/rule-differences.md)。
- [验收报告](../reports/stage1-acceptance.md)、[机器可读结果](../reports/stage1-acceptance.json)、[测试结果](../reports/pytest.xml)。
- [第三方许可记录](../THIRD_PARTY_NOTICES.md)。

阶段一限定四人、每色一枚赤五、半庄的已确认事件分析。连庄、终局及房间特有规则由输入牌谱确定；未验收规则、西入、三人和东风会暂停建议。东风回放可用于静态视觉调试，不能替代半庄连续追踪和最终验收。
