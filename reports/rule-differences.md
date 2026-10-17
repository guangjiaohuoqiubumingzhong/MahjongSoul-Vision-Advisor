# 阶段一规则适配范围

底层规则固定为 Mortal libriichi 的指定提交。第一版输入为四人、每色一枚赤五、半庄的已确认 mjai 事件。规则配置为 `configs/rules.json`。当前不是实时雀魂房间规则引擎；阶段二及第三阶段接入界面时还需核对房间配置和可操作按钮。

| 项目 | 当前处理 | 验证 / 边界 |
| --- | --- | --- |
| 普通/赤牌弃牌 | 37 种牌编码，赤五单独动作 | constructed discard_red；物理计数合并到基本五，普通五最多三张、赤五每色一张 |
| 吃与赤牌选择 | 三种吃位置，保留具体 consumed 组合及来源 | chi_red_choices；普通/赤五组合保留，模型对同组组合共用 Q 值 |
| 食替 | 由 libriichi 掩码限制，实际弃牌也校验掩码 | chi_then_discard：吃 3m 的 4m/5mr 后，3m 与 6m 被禁止；没有放宽引擎 |
| 碰 | 具体目标玩家、目标牌和 consumed | pon；上游雀魂导出样本中的实际操作亦逐事件验证 |
| 明杠 | 与最新弃牌、来源和三张 consumed 匹配 | daiminkan |
| 暗杠 | 四张 consumed、独立杠牌选择 | ankan、two_ankan_choices；批量两组特征分别给出是否杠和选择哪张牌的评分 |
| 加杠 | 匹配之前碰的三张具体牌，保留新增 pai | kakan；上游连续事件也覆盖副露 |
| 补牌/宝牌 | 按已确认事件接受，允许引擎支持的翻牌与补牌时序 | ankan_rinshan_dora；宝牌翻开必须有对应杠，指示牌最多五张 |
| 立直 | 宣言 → 宣言弃牌 → 成立；成立后只允许引擎合法动作 | riichi_confirmation；不能把宣言直接当成已成立 |
| 荣和/自摸 | 同一模型 agari 编号转换为明确语义 | ron、tsumo；furiten/yaku 判断复用规则层 |
| 跳过 | none → pass；不当作已执行动作写回 | 吃碰/荣和响应固定样本 |
| 特殊流局 | 模型可选九种九牌 | kyushukyuhai；四家立直、四杠等结算仅接收确认结果，不擅自生成 |
| 连庄与终局条件 | 以确认 start_kyoku/end_game 为准 | 不调用天凤 arena 去决定雀魂是否连庄、止局、飞人或西入；此部分未声称等价 |
| 计分差异 | 输入分数、供托、本场和结算变化来自牌谱 | 不输出自行推算的房间结算；立直扣分的状态更新由引擎完成 |
| 西入、三人、东风、自定义赤牌 | 暂不支持并暂停建议 | start_game / start_kyoku 与转换入口有范围检查；需专项适配后开放 |

固定样本既有人工构造的合法事件前缀，也有两个上游导出样本：

- `majsoul-gold-south.export.json` 原文件为 mjai-reviewer `convlog/tests/testdata/confusing_nakis_7.json`，规则为 Gold Room South，四人且每色一枚赤五。
- `tenhou-hanchan.export.json` 原文件为 `ranked_game.json`，包含三局连续事件。

来源提交均为 `2dc5ec5c8b28517cfb45f57eb21536d9a8f67aa9`，原文件未改动；转换使用同一提交的 convlog。两份转换事件分别对全部四个玩家视角重放，自己实际操作必须通过规则检查，所有生成候选也必须合法。

雀魂路径遵循 [mjai-reviewer 的雀魂导出说明](https://github.com/Equim-chan/mjai-reviewer/blob/2dc5ec5c8b28517cfb45f57eb21536d9a8f67aa9/mjsoul.adoc)：先导出为 Tenhou6 JSON，再交给 convlog。没有访问用户游戏账号，也没有验收用户的原生 protobuf 下载或完整自有半庄。

本阶段的“适配通过”指上述固定样本中的局内动作和事件状态。未核对的房间选项、终局机制或异常序列不应被推断为已支持。状态冲突、历史缺失和不支持的规则返回明确状态并停建议，需正确历史重放或新局重建。
