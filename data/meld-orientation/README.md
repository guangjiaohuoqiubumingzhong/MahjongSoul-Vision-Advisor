# 雀魂副露朝向与结构专项集

2026-10-15 联网检索后，从已保存的公开原图和用户实测帧中选择样本，补充副露级标注。原图引用已有文件，没有重新下载或训练主网络。`manifest.json` 固定 47 个局部区域、32 张去重原图、21 个来源家族、46 组副露和 10 个空桌面/牌背反例，覆盖四个座位。

## 为什么选择这些数据

| 来源 | 选择用途 | 限制 |
|---|---|---|
| [pjura/mahjong_souls_tiles](https://huggingface.co/datasets/pjura/mahjong_souls_tiles) | 1,977 张单牌用于已有牌种基线 | Apache-2.0；与当前模型同作者，不能当独立准确率。没有横牌/来源/杠层次标签 |
| [xdedss/cvmaj](https://github.com/xdedss/cvmaj) | 完整牌桌例图及 723 张外部单牌；本专项从完整截图补标副露结构 | MIT；单牌标签本身不提供副露来源。原项目副露提取未完成 |
| Steam 与作者复盘截图 | 有真实透视、相邻副露、白板、赤牌、明杠/暗杠/加杠、不同牌背 | 作者和游戏方保留版权，当前为本地调试材料；逐图来源保留在清单及原始公开图库 |
| 用户原始窗口帧/已核对录像截图 | 实测复现和完整建议链路回归 | 相同对局多帧共享来源家族，不能当独立对局 |

检索但未用于训练的候选：[Roboflow Will 的 140 图集](https://universe.roboflow.com/will-fuogp/mahjongsoul-tile-detection)公开页面显示 A/B 两类，[Majsoul 137 图集](https://universe.roboflow.com/majsoul/majsoul-6igdu)提供 37 类牌种检测；这些页面没有给出能直接用于本任务的横牌位置和来源真值。[SightRead](https://huggingface.co/datasets/sightread-bench/mahjong-v0.1)只有六份局面真值，不包含截图。未把不可访问的图片或页面展示数量计入本地数据量。

## 标注、评估与划分

`manifest.json` 记录原图路径、SHA-256、原生 ROI、座位、类型、完整牌列表（包含赤牌）、可确认的来源及横牌位置；预期答案只进入评分，不进入识别配置。新标注来自放大原像素的视觉核对；已有标注追溯原记录。英文辅助索引牌、低分辨率与特殊牌背失败案例照常保留。

所有样本标为 regression。没有用这批数据训练新模型，也不称其为盲测或总体准确率估计；同一作者、同一对局及多帧共享 `source_family`。这是定向调试与长期回归集。现有 pjura train/test 存在重复，不能把简单随机切分或重复帧当独立验证。

数据核对时发现并更正了三筒四张明杠被记成三张碰、绿色發被记成一索，九万被记成四万，以及九筒 ROI 左端裁掉部分牌面的初始转录/定位问题。修订均来自原生像素的独立视觉复核，记录在 `annotation-revisions.json`；最终比较对前后输出使用同一份清单和评分规则，九万修订前后该组都未确认，统计数不变。标准布局评估保留既有界面/手牌背面排除区域，非标准作者图才重新定位 ROI；避免评估器人为制造假前景。

另外有 672 个受控投影/角点顺序案例，生成器在 `scripts/stress_meld_geometry.py`。它们仅测试几何、不是 672 张真实截图或网络准确率；投影对应四个座位、碰牌来源位置、白板与有字牌及明杠位置。此控制实验不覆盖任意相机角度、贴图、遮挡、动画或低像素字形。

## 复现

```powershell
python -m scripts.build_meld_corpus
python -m scripts.evaluate_meld_corpus --stage before --baseline
python -m scripts.evaluate_meld_corpus --stage after
python -m scripts.stress_meld_geometry
python -m pytest tests/test_meld_corpus.py
```

`baseline.json` 保留冻结的 draft-22 评分基线与清单哈希。完整对照和待解决失败见 [修复报告](../../reports/meld-dataset/README.md)。
