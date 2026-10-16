# majang

Majan 雀魂截图建议助手的 Windows NVIDIA 便携测试版。

便携包请到 [Releases](https://github.com/guangjiaohuoqiubumingzhong/majang/releases/tag/v0.1.0-portable-test1) 下载，仓库的 Source code ZIP 不包含程序。

## 下载和启动

1. 下载 `Majan-0.1.0-portable-test1-win64-nvidia.zip.001`、`Majan-0.1.0-portable-test1-win64-nvidia.zip.002` 和 `Merge-Package.cmd`，放进同一文件夹。
2. 双击 `Merge-Package.cmd`，等待合并及 SHA256 校验完成。
3. 完整解压生成的 ZIP，再双击 `Majan/Majan.exe`。

压缩包合计约 2.55 GB，解压后约 3.8 GB；合并时还需要额外约 2.55 GB 空间。

## 使用条件

Windows x64、兼容 CUDA 12.x 的 NVIDIA 显卡及驱动；无需安装 Python、Rust、CUDA Toolkit 或 cuDNN。
当前适配雀魂 Steam 客户端和默认牌面，面板支持 1366×768 / 1920×1080 预设。

当前为测试版，请先保持游戏窗口正常显示且不被遮挡。独立程序的 Windows.Graphics.Capture 在构建机器上返回拒绝访问，现有兼容采集、CUDA 推理和截图建议已通过本机验证，尚待其他电脑试用。

启动异常时运行包内“检查运行环境.cmd”，查看 `userdata/self-test.json` 和 `userdata/startup.log`。

## 校验和许可

原始 ZIP SHA256：`09e7073fb274c3d20e29ef5169e9654635aeedf62510240a0ef9c71c57da8c23`。各下载附件的 SHA256 见 Release 中的 `SHA256SUMS.txt`。

第三方许可证、模型来源和对应源码随便携包一起放在 `licenses` 文件夹。
