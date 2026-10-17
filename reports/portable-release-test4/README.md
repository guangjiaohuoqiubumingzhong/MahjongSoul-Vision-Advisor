# 便携测试版 4 · 筒子牌面与缩放兼容

版本：`0.1.0-portable-test4`；日期：2026-10-17。

本次发布将传统日麻筒子牌面与系统缩放兼容更新同步到 Windows x64 / NVIDIA 便携版。

- 筒子牌面采用蓝绿双环与红心，保留雀魂参考的圆点位置、大小和上下分组，并按显示尺寸抗锯齿绘制。[实际组件预览](../ui-pin-traditional/after.png)。
- 在创建界面和采集线程时启用实际像素坐标。客户区像素尺寸不变时，显示 DPI 的变化不再单独中断采集；实际尺寸变化、位置变化、屏幕边界和遮挡的检查继续保留。DPI 保存在元数据中。
- 保留测试版 3 的自动错误归档功能，错误样本仅保存在本机 `userdata/recognition-errors/`。

完整回归 **680 项通过**，无失败、错误或跳过。回归测试结果见 [tests.json](tests.json)，移除开发环境搜索路径后的便携程序检查见 [portable-check.json](portable-check.json)，压缩包完整性、文件 SHA256 和源码一致性检查见 [archive-check.json](archive-check.json)。包内 `licenses/corresponding-source.zip` 保留对应应用和修改过的上游源码。

系统缩放相关回归使用受控测试验证，不代表已在所有显示器和缩放组合上完成实机验收。Windows.Graphics.Capture 在构建机器上仍有兼容限制，请先保持雀魂窗口可见。运行需要兼容 CUDA 12.x 的 NVIDIA 显卡和驱动，尚待其他电脑试用。
