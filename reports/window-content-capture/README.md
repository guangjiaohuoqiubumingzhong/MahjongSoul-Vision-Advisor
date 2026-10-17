# 窗口内容采集：遮挡与焦点检查

日期：2026-10-12（Asia/Shanghai）

自动建议从指定 HWND 的 Windows.Graphics.Capture 流读取客户区，不再通过切换前台解除遮挡。使用 `windows-capture==2.0.1`，已安装在项目 `.venv`，并记录在 `pyproject.toml` 和 `requirements.lock`。

采集库声明依赖 `opencv-python`，因此主环境由 headless 包切换为同版本标准 `opencv-python==4.11.0.86`，没有同时安装两个 `cv2` 分发包；OpenCV 图像处理版本仍为 4.11.0。环境依赖检查通过。

## 行为

- WGC 是自动建议的首选图像来源，屏蔽鼠标指针，按客户区裁掉窗口装饰。
- 系统帧在回调中复制，只保留最新一帧；计时器只接收裁切区域，原图仍按行动保存。
- 来源写为 `window_graphics_capture` / `window_graphics_capture_region`，图像质量为 `window_content`。
- WGC 失败时尝试 PrintWindow；只有无遮挡时才能退回 MSS。两个窗口接口都失败且有遮挡时提示重试，不抢前台。
- 最小化时丢弃缓存流并清除建议，不自动恢复窗口。用户恢复窗口后重新连接，并重新确认计时器触发。
- 黑屏、边界不匹配、窗口采集中移动、窗口身份变化及超过 2 秒的旧系统帧会被拒绝。
- 人工标注和录像的严格屏幕采集流程保持原有入口。

## 实际窗口验证

运行 `python -m scripts.verify_window_content_capture`。脚本只创建自己的临时窗口，不发送游戏输入；测试窗口和遮挡窗口都使用 `WS_EX_NOACTIVATE`，结束时关闭。

普通 Windows 进程下的结果：

1. 从 1366×768 的测试窗口读取四个不同颜色区域。
2. 用另一个不激活窗口完全遮住目标，旧屏幕采集检查确认遮挡存在。
3. 在完全遮挡期间改变目标的四个颜色区域，WGC 仍返回新颜色，排除使用遮挡前缓存或遮挡窗口像素的情况。
4. 计时器裁切尺寸为 230×88，仍来自相同窗口流。
5. 前台窗口句柄前后相同；验证帧到读取时约 7 毫秒。

结果见 `occlusion-check.json`，图像分别在 `before-cover/` 与 `covered-and-updated/`。受限执行沙箱无法建立 WGC 项目，因此该实际窗口测试在普通 Windows 权限下运行；正常从启动脚本打开建议面板无需管理员权限。

当前雀魂客户端已关闭，实际游戏的新接口还未取图验证；没有把临时窗口验证视为游戏实战识别验证。

## 测试

- 初次窗口相关单元测试：20 项通过，`unit-tests.xml`。
- 补充最小化恢复后重新分析的检查，再运行窗口与截图建议相关测试：80 项通过，`targeted-tests.xml`。
- 依赖调整与恢复处理完成后的完整测试：320 项通过、1 项失败，`tests.xml`。失败为先前状态检查已记录的 `test_several_melds_ask_which_call_was_just_completed`：构造局面存在手牌数量冲突，最新代码先报告数量错误，而用例预期刚完成副露确认提示。本次未改动该吃碰测试或其状态逻辑。
- 两种客户区尺寸的裁切、空白拒绝、系统缓存复制、旧帧拒绝、关闭清理、窗口移动、无焦点切换、接口回退及恢复后重新分析都有针对性覆盖。

最终完整测试包含最小化恢复处理的新增检查，并在 OpenCV 包调整完成后运行；环境依赖检查没有缺失或冲突。

参考：[微软窗口捕获接口](https://learn.microsoft.com/en-us/windows/win32/api/windows.graphics.capture.interop/nf-windows-graphics-capture-interop-igraphicscaptureiteminterop-createforwindow)、[窗口采集库](https://github.com/NiiightmareXD/windows-capture/tree/main/windows-capture-python)。
