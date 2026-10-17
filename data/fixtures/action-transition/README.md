# 行动变化时显示旧建议的用户截图

2026-10-13 的两张原始桌面图保留为 `*-desktop.png`。客户区裁切为 `[1,31,1920,1080]`，原图哈希记录在 `sources.json`；`*-1920.png` 没有缩放。测试中的 1366×768 来自原尺寸客户区缩小。

- `draw-with-stale-pass`：当前有 13 张手牌加独立摸牌、自己牌河 11 张，面板仍显示较早吃牌响应的“跳过”。
- `pon-with-stale-discard`：当前有 13 张手牌、没有独立摸牌槽、自己牌河 15 张，画面出现碰牌按钮，面板仍显示较早摸牌的“打出 9筒”。

两个尺寸下，轻量牌体/牌位检测分别得到 `(13, true, 11)` 和 `(13, false, 15)`，无需牌种网络即可区分行动。测试不把本文当作运行时输入。脚本 `python -m scripts.diagnose_action_transition` 可重新裁切与检查；结果见 `reports/action-transition-fix/markers.json`。
