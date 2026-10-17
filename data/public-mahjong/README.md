# 雀魂公开牌面与截图样本

2026-10-12 收集完成。双击项目根目录的 `查看公开截图.cmd`，或直接打开本目录的 `index.html` 浏览。可筛选来源、画面类型、牌种和关键词；点缩略图打开原图，默认隐藏完全重复的文件。

首次收集保存 2,920 张图片（去重 2,683 张）。2026-10-13 增补杠牌相关公开资料 212 个文件，去重新增 207 张；目前合计 **3,132 张图片，按 SHA-256 去重 2,890 张**，所有清单记录均通过图片解码和 SHA-256 校验。原图保留原样，当前主网络没有用新增图重新训练。

## 已保存内容

| 来源 | 本地图片 | 内容与用途 | 上游许可说明 |
| --- | ---: | --- | --- |
| [pjura/mahjong_souls_tiles](https://huggingface.co/datasets/pjura/mahjong_souls_tiles) | 1,977 | 34 类雀魂单牌裁切，保留原 train/test 和标签目录；适合调牌种、旋转、透视识别 | 数据卡声明 Apache-2.0，原数据卡已保存 |
| [xdedss/cvmaj](https://github.com/xdedss/cvmaj) | 723 单牌 + 20 示例 | 从 pack.zip 导入带标签目录，包含大小与透视差异；示例包括牌桌、局部牌面和处理结果 | MIT，LICENSE 已保存 |
| [雀魂 Steam 社区截图](https://steamcommunity.com/app/2739990/screenshots/) | 200 | 公开原始截图，保留原尺寸，含不同桌布、语言、立直/鸣牌按钮、动画、和牌和结算画面 | 公开用户截图，没有统一开放再利用许可；原作者和游戏方版权保留，作为本地调试材料 |

首次按画面类型逐张查看了上述 **220 张场景/示例图**，分类为：

- **44 张牌桌场景**，其中 **28 张没有明显弹窗或动画遮住牌桌**；适合先检查手牌、牌河和副露布局。
- 130 张和牌、结算或特效图：适合检查大字、牌面特效及“非行动阶段不误触发”。
- 4 张牌面局部图、8 张处理结果/示意图。
- 34 张大厅、角色或其他界面：可以作非牌局样本。

画面类型分类不是逐张麻将牌的真值标注。`ground_truth_reviewed` 仍为 false；不要将自动识别结果直接作为正确答案。

2026-10-13 已回放测试当前网络与整局识别。统计及修改前后带框图见项目的 `reports/public-vision-fix/report.html`。另在 `manual-regressions.json` 保存两张截图中 **35 张人工核对牌面**的标签、位置与图片哈希，仅用于固定修复回归；这不意味着整张截图或全部数据已完成真值标注。可运行 `python -m scripts.evaluate_public_vision --output reports/public-vision-latest` 重测当前配置。

同日横牌复核：`manual-regressions.json` 中横牌二索的转录已更正为四索，普通五索更正为赤五索；原始图未改。`sideways-regressions.json` 另记录真实四索、八索的部分区域标注和三处绿色牌背白边反例。字形校验模型只使用 1,029 张 pjura train 图，cvmaj 保持留出；横放控制实验及最新回归见 `reports/public-sideways-fix/report.html`。

## 从哪里开始修问题

1. 打开图库，选择“牌桌场景”；搜索 `clear_board`，优先查看 28 张无遮挡样本。对应文件清单为 `clear-board-images.txt`。
2. 搜索 `pon`、`ron`、`riichi`、`tsumo`、`overlay`，看按钮或动画对识别的影响。标签来自画面类型检查，不代表完整对局状态。
3. 单牌问题切换到“带标签单牌”，搜索 `7p`、`8s`、`P` 等牌标记。原标签保留，`label_mjai` 只对明确对应的标签做转换。
4. 找到具体失败样本后，固定原图和人工核对结果，再修改识别逻辑；修改后重跑相同截图，比较改动前后的漏牌、错牌与来源判断。

单图可沿用项目现有入口：

```powershell
& .\.venv\Scripts\python.exe -m app.cli recognize data/public-mahjong/steam-community/images/3803922882.jpg --output reports/public-mahjong-debug/3803922882.json
```

这张示例含碰/荣和按钮。图片存在不同分辨率、语言和桌布，有些是三人场、浏览器截图或经过处理的示意图；现有中文四人场配置不能直接代表它们都应该完整识别。原图没有拉伸、裁切或改写。牌桌检查需核对客户区和场景是否适配当前配置。

## 目录和记录

- `index.html`：离线图库，不需要服务器或登录。
- `manifest.jsonl`：每张图的来源页面、原标签、尺寸、字节数、SHA-256、重复关系和场景标签。
- `validation.json`：实际数量、分类与完整性检查结果。
- `label-integrity.json`：重复图片与原始标签一致性检查。237 个重复文件保留原样，图库默认隐藏；未发现相同图像对应不同已映射牌种，但 pjura 原 train/test 有 34 组相同图片交叉，因此不要直接拿原划分统计独立准确率。
- `board-images.txt` / `clear-board-images.txt`：目前 182 张去重牌桌场景，以及初始 28 张无遮挡牌桌的相对路径列表。
- `scene-review.json`：初始 220 张示例和新增 182 张图片的场景分类，牌面真值仅在独立的局部回归标注中。
- `pjura/dataset/`：Hugging Face 原始标签和划分目录；提交版本见 `pjura/source/provenance.json`。
- `cvmaj/images/`：723 张带标签图片；原压缩包、MIT 许可和提交版本在 `cvmaj/source/`。压缩包里的 big/small 重复目录和 Python 程序没有导入。
- `cvmaj/examples/`：20 张项目示例图。
- `steam-community/images/`：200 张原始截图；来源页面链接见清单，截图页快照在 `steam-community/source/pages/`。
- `browse/`：派生缩略图和联系表，原图保持不变。

pjura 数据与当前分类器来自同一作者，适合排查牌面问题，不能自动当成独立准确率验证。该数据不提供完整牌河、操作阶段和吃碰来源标注；cvmaj 的 `x`、`5p-` 等特殊标签保留原值，未猜测转换。

## 找到但未下载的资源

| 资源 | 已确认信息 | 本次处理 |
| --- | --- | --- |
| [Roboflow：Will 的 mahjongsoul-tile-detection](https://universe.roboflow.com/will-fuogp/mahjongsoul-tile-detection/dataset/23) | 页面显示 140 张、CC BY 4.0；第 23 版导出预处理拉伸为 640×640 | 当前公开图片页请求返回 403，未下载，未计入数量 |
| [Roboflow：MajsoulBot](https://universe.roboflow.com/majsoulbot/majsoulbot) | 页面显示 232 张、19 类 UI，CC BY 4.0 | 页面关联的 GitHub 项目目前返回 404；未得到本地图片 |
| [SightRead Mahjong v0.1](https://huggingface.co/datasets/sightread-bench/mahjong-v0.1) | 6 份完整局面真值 | 数据卡明确不提供截图，因此未作为图片来源 |
| [QueHun 的第三方数据](https://github.com/jzjzzzzzzz/quehun/tree/main/dataset) | 主体来自 Google/eBay/Alibaba 的实体麻将照片 | 与雀魂画面风格不符，本次排除 |

## 复用收集工具

只增补公开截图，保留已下载原图：

```powershell
& .\.venv\Scripts\python.exe -m scripts.collect_public_mahjong --sources steam --steam-pages 20 --workers 4
& .\.venv\Scripts\python.exe -m scripts.index_public_mahjong
```

完整来源信息保留在 `discovery/`。图库仅用于本地调试，没有将这些原图上传、公开发布或直接用于重新训练现有模型。

## 2026-10-13 杠牌专项增补

新增 212 个原图文件（去重新增 207 张），来自 6 篇作者雀魂复盘、Reddit 原作者截图及 Steam 公开检索；保留原文 HTML、CDN 地址、原图尺寸和哈希。当前收集器一轮列表含 210 张，另有 2 个此前轮次已保留的文件，因此以根目录清单和 validation.json 为实际总数。作者页面的 note 标志重复文件也保留并标明重复关系。

- [杠牌专项浏览入口](kan-scenes/index.html)：13 个局部核对案例和来源，含自己白板加杠、九索大明杠、上家红中大明杠和六筒暗杠、对家东加杠、下家一筒暗杠，以及普通碰牌反例。
- [人工核对记录](kan-scenes/regressions.json)：仅对列出的副露区域标注牌种、类型和可确定的来源；不能把这些标注扩展成整张图真值。
- [修改前后报告](../../reports/kan-all-seats-fix/report.html)：13 个局部样本正确 2→11，错误确认 1→0，2 张暗色或特殊红背图仍未确认；三人场、修罗场和超宽图只测试原始像素的局部识别。
- `kan-scenes/collection.json`：最近一次收集结果及错误；全部图片解码及哈希核验以根目录 `validation.json` 为准。

场景分类现在包含 185 个牌桌文件（去重 182 张）；新增资料中 138 张去重牌桌图另走默认布局诊断，尺寸不适配的输入保留错误记录。它们没有全图牌面真值，不统计总体准确率。公开图没有统一开放再分发许可，作为本地调试材料保留作者与游戏方权利。

重跑专项：

```powershell
& .\.venv\Scripts\python.exe -m scripts.collect_kan_scenes
& .\.venv\Scripts\python.exe -m scripts.index_public_mahjong
& .\.venv\Scripts\python.exe -m scripts.benchmark_kan_scenes --stage before
& .\.venv\Scripts\python.exe -m scripts.benchmark_kan_scenes
```
