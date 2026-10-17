# Majan 界面风格候选

已选定并应用：**C 款米白和风，无截图版**。运行界面及验证见 `reports/ui-ivory/README.md`，应用使用本地桌面控件实现。以下生成图与提示词保留作为设计参考及历史记录。

## 最新版本：不显示当前截图

用户已明确偏好不放截图预览。当前四款候选都移除截图区，并缩短窗口，保留四个按钮（含“刷新截图”）、窗口与分辨率选择、当前建议及 Top 5 相对评分。截图仍供后台识别使用；本轮仅更新设计示意图，尚未修改应用。

| 编号 | 最新文件 |
| --- | --- |
| A | A-fluent-light-no-preview.png |
| B | B-graphite-dark-no-preview.png |
| C | C-ivory-japanese-no-preview.png |
| D | D-lavender-glass-no-preview.png |

最新编辑提示词见 prompts-no-preview.json。以下为首版参考记录。

生成方式：内置 image_gen。用途：供用户挑选视觉方向，尚未应用到程序。界面中的建议、评分及文案是设计示例，截图仅作为预览区参考。实际实现时将使用现有数据与真实牌面资源。

| 编号 | 文件 | 方向 |
| --- | --- | --- |
| A | A-fluent-light.png | 白蓝原生，清晰轻量 |
| B | B-graphite-dark.png | 石墨深色，薄荷绿强调 |
| C | C-ivory-japanese.png | 米白和风，墨绿与少量金色 |
| D | D-lavender-glass.png | 蓝紫轻透，柔和玻璃层次 |

保持四个按钮、窗口与分辨率选择、当前建议、Top 5 相对评分、截图预览。生成图中的装饰文案和个别牌面细节不作为实现规范。

## 联网参考

- Microsoft Fluent 2: https://fluent2.microsoft.design/
- Linear, Behind the latest design refresh: https://linear.app/now/behind-the-latest-design-refresh
- Raycast Themes: https://manual.raycast.com/themes

以上为视觉方向参考，非上述产品的官方设计。米白和风是针对本项目的原创配色方向。

完整最终生成提示词见 prompts.json。
