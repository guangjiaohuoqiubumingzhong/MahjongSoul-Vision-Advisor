# 便携测试版 3 · 自动保存识别错误

版本：`0.1.0-portable-test3`；日期：2026-10-17。

本次发布将自动错误归档功能同步到 Windows x64 / NVIDIA 便携版。同一行动补截后仍无法确认局面时，初始图、全部补截图、识别结果、采集配置及汇总清单保存到 `userdata/recognition-errors/`。成功恢复、停止和过期行动不归档；保存失败会显示原因，后续自动建议继续运行。

源码与便携版使用同一份 `app` 文件。发布时检查应用文件 SHA256、压缩包清单、ZIP 完整性和 GitHub 附件校验值；测试结果记录在本目录。包内 `licenses/corresponding-source.zip` 保留对应应用和修改过的上游源码。

- 回归测试：**652 项通过**，无失败、错误或跳过，见 [tests.json](tests.json)。
- 便携程序：移除开发环境搜索路径后，界面、模型、CUDA 推理、截图建议和 PrintWindow 兼容窗口采集检查通过，见 [portable-check.json](portable-check.json)。
- 压缩包：文件清单、逐文件 SHA256、ZIP 完整性，以及包内应用源码和对应源码的一致性，见 [archive-check.json](archive-check.json)。

Windows.Graphics.Capture 在构建机器上仍有拒绝访问的兼容限制，请先保持雀魂窗口可见。运行需要兼容 CUDA 12.x 的 NVIDIA 显卡和驱动，尚待其他电脑试用。自动错误归档用于保存问题样本，不代表识别准确率提升。
