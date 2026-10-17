# 右家吃牌的透视朝向回归

2026-10-15 用户报告 `meld_right` 不完整。`turn-293.png` 是截图中助手预览对应的原始客户区帧（1920×1080），来自 `reports/screenshot-advice/manual-20261015T132707-7/00000001.png`；`turn-300.png` 来自同局稍后 `reports/screenshot-advice/run-20261015T132721-10/action-0006/00001047.png`，用于验证反应阶段。旁边的 `.frame.json` 原样保留。

人工核对：右家唯一一组吃牌，按该区域旋转后的排列为赤五万、四万、六万；赤五万横放并来自自家（actor=1、target=0）。前者应正常产生出牌建议，后者应正常产生反应建议。完整原帧不用于训练；1366×768 版本只在测试时缩放，是派生变体。

复现：`.venv/Scripts/python.exe -m pytest tests/test_meld_face_orientation.py`。
