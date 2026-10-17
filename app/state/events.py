from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field

TILES = [f"{i}{suit}" for suit in "mps" for i in range(1, 10)] + list("ESWNPFC") + ["5mr", "5pr", "5sr"]
BASE = TILES[:34]
ACTION_EVENTS = {"dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "reach", "hora"}


class InputError(ValueError):
    pass


class UnsupportedRule(InputError):
    pass


def base(tile: str) -> str:
    return tile[:-1] if tile.endswith("r") else tile


def visible_event(event: dict, player_id: int) -> dict:
    """Drop all extraneous fields as well as hidden hands/draws/ura indicators."""
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise InputError("事件必须包含 type。")
    keys = {
        "start_game": ("names",),
        "start_kyoku": ("bakaze", "dora_marker", "kyoku", "honba", "kyotaku", "oya", "scores", "tehais"),
        "tsumo": ("actor", "pai"), "dahai": ("actor", "pai", "tsumogiri"),
        "chi": ("actor", "target", "pai", "consumed"), "pon": ("actor", "target", "pai", "consumed"),
        "daiminkan": ("actor", "target", "pai", "consumed"), "ankan": ("actor", "consumed"),
        "kakan": ("actor", "pai", "consumed"), "dora": ("dora_marker",),
        "reach": ("actor",), "reach_accepted": ("actor",), "hora": ("actor", "target", "deltas"),
        "ryukyoku": ("deltas",), "end_kyoku": (), "end_game": (),
    }
    kind = event["type"]
    if kind not in keys:
        raise UnsupportedRule(f"不支持事件类型：{kind}")
    if kind == "start_game":
        if event.get("kyoku_first", 0) != 0 or ("aka_flag" in event and event["aka_flag"] is not True):
            raise UnsupportedRule("仅支持四人赤牌半庄（kyoku_first=0、aka_flag=true）。")
    result = {"type": kind, **{key: deepcopy(event[key]) for key in keys[kind] if key in event}}
    if kind == "start_kyoku" and "tehais" in result:
        hands = result["tehais"]
        if not isinstance(hands, list) or len(hands) != 4 or any(not isinstance(h, list) or len(h) != 13 for h in hands):
            raise InputError("四人开局必须有四组 13 张手牌。")
        result["tehais"] = [hand if actor == player_id else ["?"] * 13 for actor, hand in enumerate(hands)]
    if kind == "tsumo" and result.get("actor") != player_id:
        result["pai"] = "?"
    return result


def validate_schema(event: dict):
    kind = event["type"]
    required = {
        "start_kyoku": ["bakaze", "dora_marker", "kyoku", "honba", "kyotaku", "oya", "scores", "tehais"],
        "tsumo": ["actor", "pai"], "dahai": ["actor", "pai", "tsumogiri"],
        "chi": ["actor", "target", "pai", "consumed"], "pon": ["actor", "target", "pai", "consumed"],
        "daiminkan": ["actor", "target", "pai", "consumed"], "ankan": ["actor", "consumed"],
        "kakan": ["actor", "pai", "consumed"], "dora": ["dora_marker"],
        "reach": ["actor"], "reach_accepted": ["actor"], "hora": ["actor", "target"],
    }.get(kind, [])
    if any(key not in event for key in required):
        raise InputError(f"{kind} 缺少必需字段。")
    for key in ("actor", "target", "oya"):
        if key in event and (type(event[key]) is not int or event[key] not in range(4)):
            raise InputError(f"{key} 必须是 0–3 玩家编号。")
    if kind == "start_game" and "names" in event:
        if not isinstance(event["names"], list) or len(event["names"]) != 4 or any(not isinstance(n, str) for n in event["names"]):
            raise InputError("names 必须是四个字符串。")
    for key in ("scores", "deltas"):
        if key in event and event[key] is not None:
            if not isinstance(event[key], list) or len(event[key]) != 4 or any(type(x) is not int for x in event[key]):
                raise InputError(f"{key} 必须是四个整数。")
    if kind == "start_kyoku":
        if event["bakaze"] not in ("E", "S"):
            raise UnsupportedRule("尚未验收西入/三人/自定义房间规则，暂停建议。")
        if type(event["kyoku"]) is not int or event["kyoku"] not in range(1, 5):
            raise InputError("kyoku 必须是 1–4。")
        for key in ("honba", "kyotaku"):
            if type(event[key]) is not int or not 0 <= event[key] <= 255:
                raise InputError(f"{key} 超出引擎范围。")
        for hand in event["tehais"]:
            for tile in hand:
                if tile not in TILES and tile != "?":
                    raise InputError(f"无效牌标识：{tile}")
    if kind == "dahai" and type(event["tsumogiri"]) is not bool:
        raise InputError("摸切字段必须已确认，不能省略或使用未知值。")
    for key in ("pai", "dora_marker"):
        if key in event and event[key] not in TILES and not (kind == "tsumo" and event[key] == "?"):
            raise InputError(f"{key} 无效或不可见。")
    if "consumed" in event:
        expected = {"chi": 2, "pon": 2, "daiminkan": 3, "kakan": 3, "ankan": 4}[kind]
        consumed = event["consumed"]
        if not isinstance(consumed, list) or len(consumed) != expected or any(t not in TILES for t in consumed):
            raise InputError("consumed 必须保留指定数量的具体可见牌，包括赤牌。")
        tiles = [base(t) for t in consumed] + ([base(event["pai"])] if "pai" in event else [])
        if kind == "chi":
            if any(t not in BASE[:27] for t in tiles) or len({t[-1] for t in tiles}) != 1:
                raise InputError("吃牌必须同色连续数牌。")
            ranks = sorted(int(t[0]) for t in tiles)
            if ranks != list(range(ranks[0], ranks[0] + 3)):
                raise InputError("吃牌组合不连续。")
        elif len(set(tiles)) != 1:
            raise InputError("碰/杠组合必须是同一种基本牌。")


@dataclass
class EventLedger:
    player_id: int
    game: bool = False
    round: bool = False
    ended: bool = False
    resolved: bool = False
    round_id: int = 0
    hand: Counter = field(default_factory=Counter)
    seen: Counter = field(default_factory=Counter)
    red_seen: Counter = field(default_factory=Counter)
    normal_seen: Counter = field(default_factory=Counter)
    next_draw: int | None = None
    pending_discard: int | None = None
    last_draw: str | None = None
    last_discard: dict | None = None
    reach_declared: set = field(default_factory=set)
    reach_accepted: set = field(default_factory=set)
    melds: list = field(default_factory=lambda: [[] for _ in range(4)])
    dora_count: int = 0
    draw_count: int = 0
    pending_kan: bool = False
    kan_count: int = 0

    def _see(self, tiles):
        for tile in tiles:
            self.seen[base(tile)] += 1
            if tile.endswith("r"):
                self.red_seen[tile] += 1
            else:
                self.normal_seen[tile] += 1
        if any(n > 4 for n in self.seen.values()) or any(n > 1 for n in self.red_seen.values()):
            raise InputError("已知物理牌数量冲突（同种最多四张、每色赤五最多一张）。")
        if any(self.normal_seen[f"5{s}"] > 3 for s in "mps"):
            raise UnsupportedRule("四张普通五与每色一张赤五的固定规则不兼容。")

    def _consume(self, tiles):
        needed = Counter(tiles)
        if any(self.hand[tile] < n for tile, n in needed.items()):
            raise InputError("实际操作消耗了自己手中不存在的牌。")
        self.hand.subtract(needed)

    def accept(self, event: dict):
        validate_schema(event)
        kind = event["type"]
        actor = event.get("actor")
        if kind == "start_game":
            if self.game or self.ended:
                raise InputError("重复开局或结束后未重置。")
            self.game = True
            return
        if not self.game or self.ended:
            raise InputError("缺少 start_game 或对局已结束。")
        if kind == "start_kyoku":
            if self.round:
                raise InputError("缺少 end_kyoku。")
            if any(t == "?" for t in event["tehais"][self.player_id]):
                raise InputError("自己的开局手牌不完整。")
            self.round_id += 1
            self.round, self.resolved = True, False
            self.hand = Counter(event["tehais"][self.player_id])
            self.seen, self.red_seen, self.normal_seen = Counter(), Counter(), Counter()
            self._see([*self.hand.elements(), event["dora_marker"]])
            self.next_draw, self.pending_discard = event["oya"], None
            self.last_draw, self.last_discard = None, None
            self.reach_declared, self.reach_accepted = set(), set()
            self.melds = [[] for _ in range(4)]
            self.dora_count, self.draw_count, self.kan_count = 1, 0, 0
            self.pending_kan = False
            return
        if kind == "end_game":
            if self.round:
                raise InputError("结束对局前缺少 end_kyoku。")
            self.ended = True
            return
        if not self.round:
            raise InputError("缺少 start_kyoku。")
        if kind == "end_kyoku":
            if not self.resolved:
                raise InputError("缺少和牌或流局结果。")
            self.round = False
            return
        if self.resolved:
            if kind == "hora":  # multi-ron results are consecutive public events
                return
            raise InputError("本局已结算，应先 end_kyoku。")
        if kind == "tsumo":
            if self.pending_discard is not None or actor != self.next_draw:
                raise InputError("摸牌顺序冲突或缺少弃牌。")
            if self.draw_count >= 70:
                raise InputError("摸牌事件超过四人规则可用牌数。")
            if actor == self.player_id:
                if event["pai"] == "?":
                    raise InputError("自己的摸牌不可未知。")
                self.hand[event["pai"]] += 1
                self._see([event["pai"]])
            self.pending_discard, self.last_draw = actor, event["pai"]
            self.last_discard = None
            self.draw_count += 1
            self.pending_kan = False
        elif kind == "dahai":
            if actor != self.pending_discard:
                raise InputError("弃牌缺少该玩家的摸牌或副露。")
            if actor == self.player_id:
                if event["tsumogiri"] and event["pai"] != self.last_draw:
                    raise InputError("自己的摸切牌与实际摸牌不符。")
                self._consume([event["pai"]])
            else:
                self._see([event["pai"]])
            self.pending_discard, self.next_draw = None, (actor + 1) % 4
            self.last_discard = event
        elif kind in ("chi", "pon", "daiminkan"):
            target = event["target"]
            if not self.last_discard or self.last_discard["actor"] != target or self.last_discard["pai"] != event["pai"] or actor == target:
                raise InputError("副露来源与最新弃牌不符。")
            if kind == "chi" and actor != (target + 1) % 4:
                raise InputError("吃牌只能来自上家。")
            if self.pending_discard is not None or actor in self.reach_accepted:
                raise InputError("当前阶段不能副露。")
            if actor == self.player_id:
                self._consume(event["consumed"])
            else:
                self._see(event["consumed"])
            self.melds[actor].append(event)
            self.last_discard, self.last_draw = None, None
            if kind == "daiminkan":
                self.next_draw, self.pending_kan = actor, True
                self.kan_count += 1
            else:
                self.pending_discard = actor
        elif kind in ("ankan", "kakan"):
            if self.pending_discard != actor:
                raise InputError("暗杠/加杠必须在该玩家摸牌后。")
            if kind == "kakan":
                previous = next((m for m in self.melds[actor] if m["type"] == "pon" and base(m["pai"]) == base(event["pai"])), None)
                if previous is None or Counter([previous["pai"], *previous["consumed"]]) != Counter(event["consumed"]):
                    raise InputError("加杠必须保留之前碰牌的具体三张牌。")
                self.melds[actor].remove(previous)
            tiles = event["consumed"] if kind == "ankan" else [event["pai"]]
            if actor == self.player_id:
                self._consume(tiles)
            else:
                self._see(tiles)
            self.melds[actor].append(event)
            self.pending_discard, self.next_draw, self.pending_kan = None, actor, True
            self.last_draw, self.last_discard = None, None
            self.kan_count += 1
        elif kind == "dora":
            if self.dora_count >= 5 or self.dora_count > self.kan_count:
                raise InputError("宝牌更新缺少对应杠事件或指示牌超过五张。")
            self._see([event["dora_marker"]])
            self.dora_count += 1
        elif kind == "reach":
            if self.pending_discard != actor or actor in self.reach_declared:
                raise InputError("立直宣言顺序错误。")
            self.reach_declared.add(actor)
        elif kind == "reach_accepted":
            if actor not in self.reach_declared or actor in self.reach_accepted or not self.last_discard or self.last_discard["actor"] != actor:
                raise InputError("立直成立缺少宣言/宣言弃牌。")
            self.reach_accepted.add(actor)
        elif kind in ("hora", "ryukyoku"):
            self.resolved = True
