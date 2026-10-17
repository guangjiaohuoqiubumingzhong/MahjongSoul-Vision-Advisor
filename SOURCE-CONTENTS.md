# 雀魂视觉出牌助手 · 源码与便携版

当前发布：`v0.1.0-portable-test4`（2026-10-17），包含传统日麻筒子牌面和按实际像素采集的缩放兼容更新。验证记录见 [便携测试版 4](reports/portable-release-test4/README.md)。

源码仓库包含 `app`、`scripts`、`packaging`、`tests`、`configs`、`docs`、`patches`、`tools`、识别模板、传统分类器、模型来源与校验清单、固定事件测试数据，以及界面验证截图和说明。

以下内容不进入 Git 历史：Python/Rust 开发环境、编译缓存、用户运行日志、原始游戏截图和公开训练数据集、批量生成的调试报告，以及大型 ONNX / PyTorch 权重。依赖版本固定在 `requirements.lock` 和 `configs/sources.json`。完整运行资源随 Release 中的便携版提供。

## 从源码运行

1. 按 `docs/environment.md` 安装 Python 3.12.7、Git、Visual Studio 2022 C++ Build Tools，并运行 `scripts/setup.ps1`。上游 Mortal 与 mjai-reviewer 源码由脚本按固定版本下载；本项目的修改在 `patches/mortal-snapshot`。
2. 下载并完整解压对应便携版，将 `Majan/_internal/models`、`Majan/_internal/assets` 中缺失的模型/资源复制回源码的同名目录，保留目录结构。已提交的来源清单和 SHA256 用于核对版本。也可按 `docs/pjura.md` 从锁定的公开权重重建识别模型。
3. 双击 `启动截图建议.cmd`。实际运行需要兼容 CUDA 12.x 的 NVIDIA 显卡和驱动。

仓库中的固定事件数据位于 `data/fixtures`。使用真实截图和训练集的视觉回归测试需要额外取得相应输入数据；未附带的原始截图不能仅凭源码仓库复现。发布记录列出本次在构建机器上的验证结果。

## 构建本次便携版

```powershell
./scripts/build_portable.ps1 -Version '0.1.0-portable-test4' -StageDirectory '.tools/portable-stage-test4' -OutputDirectory 'dist/Majan-portable-test4'
```

请选择尚不存在的暂存和输出目录；旧版构建保留。第三方许可证和对应源码位于便携包的 `licenses` 目录。
