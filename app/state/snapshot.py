"""Turn the current screenshot's visible regions into a decision snapshot."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

from .events import TILES, base, validate_schema, InputError


class SnapshotError(InputError):
    pass


# Marks a 食替 call that the frame cannot name: the rules engine settles it.
AMBIGUOUS_CALL = object()


def resolve_last_call(snapshot, legal_ids, dimmed=None):
    """Which visible meld was just completed, when the frame cannot say.

    After a call you discard without drawing. The melds are drawn in layout
    order, not call order, so the call cannot be read off one frame; it only
    changes the decision when 食替 forbids a discard. Each visible meld is
    tried as the call that was just completed and the position is accepted
    only when every one of them yields exactly the same legal actions, which
    is when the ambiguity cannot change the advice.
    """
    if legal_ids is None:
        return None
    player = snapshot["player_id"]
    options = []
    for meld in snapshot["melds"][player]:
        candidate = deepcopy(snapshot)
        candidate["last_call"] = meld
        ids = legal_ids(candidate)
        if ids:
            options.append((meld, frozenset(ids)))
    if not options:
        return None
    if any(ids != options[0][1] for _, ids in options) and dimmed is not None:
        # The game greys out exactly the tiles 食替 forbids discarding, so the
        # call whose forbidden set equals the greyed tiles is the one just made.
        def forbidden(ids):
            return sorted(tile for tile in set(snapshot["hand"]) if TILES.index(tile) not in ids)
        matching = [meld for meld, ids in options if forbidden(ids) == sorted(set(dimmed))]
        if len(matching) == 1:
            return matching[0]
    if any(ids != options[0][1] for _, ids in options):
        raise SnapshotError("多组副露下弃牌选择随刚完成的吃/碰而不同，请核对是刚完成哪一组"
                            "（面板“核对当前局面”可勾选）。")
    return options[0][0]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                    separators=(",",":"),allow_nan=False).encode()).hexdigest()


# Four players start with 25000 each; points only move between players and
# riichi deposits (1000 each, shown as the 供托 counter), so the visible scores
# plus the deposits always add up to the starting total.
STARTING_POINTS = 25000
SCORE_MIN_OCR = .7


def visible_scores(regions, corrections, config, one, kyotaku):
    """Scores, letting the point total settle one unsure reading.

    A riichi stick drawn beside a score can push an otherwise clear OCR read
    just under the threshold (measured "24000" at 0.88-0.89). When exactly one
    score is unread, a candidate from its own OCR variants is accepted only if
    it makes the four scores plus the deposits equal the starting total.
    """
    from app.vision.text import parse_text
    names = {}
    for name, r in regions.items():
        if r["kind"] == "text" and r.get("field") == "score" and r.get("player") in range(4):
            names[r["player"]] = name
    missing = [actor for actor, name in names.items()
               if name not in corrections and not regions[name]["complete"]]
    if len(names) != 4 or len(missing) != 1 or type(kyotaku) is not int:
        return [one("text",actor,"score") for actor in range(4)]
    actor = missing[0]
    known = {other: one("text",other,"score") for other in range(4) if other != actor}
    total = 4 * config.get("starting_points", STARTING_POINTS)
    expected = total - sum(known.values()) - 1000 * kyotaku
    text = regions[names[actor]].get("text") or {}
    readings = [parse_text(v.get("text",""), "score") for v in text.get("variants",[])
                if (v.get("score") or 0) >= SCORE_MIN_OCR]
    if expected not in readings:
        return [one("text",other,"score") for other in range(4)]
    return [known.get(other, expected) for other in range(4)]


def build_snapshot(observation, config, corrections=None, context=None, legal_ids=None):
    """Corrections are explicit current-frame values; no annotations/labels load.

    Buttons and the countdown identify the *current* decision. We do not
    replay river tiles as invented ordered game events.

    ``legal_ids`` is an optional ``snapshot -> set of legal action ids`` probe
    used only after a call, to settle which visible meld was just completed.
    """
    corrections, context = corrections or {}, context or {}
    if set(corrections) - set(observation["regions"]):
        raise SnapshotError("修正中存在未知区域。")
    player = config["seat_mapping"]["self"]
    regions = observation["regions"]

    def value(name):
        r = regions[name]
        if name in corrections:
            return deepcopy(corrections[name])
        if not r["complete"]:
            raise SnapshotError(f"请核对 {name}：识别不完整（面板“核对当前局面”可修改该区域）。")
        if r["kind"] == "text":
            return deepcopy(r["text"]["value"])
        if r["kind"] == "meld":
            events = []
            for g in r.get("groups",[]):
                if g.get("status") != "confirmed":
                    raise SnapshotError(f"请核对 {name} 的副露类型和来源。")
                called = g.get("called_positions",[])
                labels = g["labels"]
                if g["type"] == "ankan" and len(labels) == 4 and not called:
                    events.append({"type":"ankan","actor":r["player"],"consumed":list(labels)})
                    continue
                if g["type"] == "kakan" and len(labels) == 4 and type(g.get("added_position")) is int and g["added_position"] in range(4):
                    index = g["added_position"]
                    events.append({"type":"kakan","actor":r["player"],"pai":labels[index],
                                   "consumed":[t for i,t in enumerate(labels) if i != index]})
                    continue
                if (g["type"] not in {"chi","pon","daiminkan"} or len(called) != 1
                        or g.get("target") is None or type(called[0]) is not int
                        or called[0] not in range(len(labels))):
                    raise SnapshotError(f"请核对 {name} 的副露类型和来源。")
                index = called[0]
                events.append({"type":g["type"],"actor":r["player"],"target":g["target"],
                               "pai":labels[index],"consumed":[t for i,t in enumerate(labels) if i != index]})
            return events
        if r["kind"] == "river":
            return [{"tile":i["label"],"riichi":i.get("horizontal") is True,
                     "tedashi":i.get("tsumogiri") is False} for i in r["items"]]
        return [i["label"] for i in r["items"]]

    def counter(field, default):
        """A counter the interface draws only when it is not zero.

        Honba and kyotaku show nothing at zero, so a missing or unreadable
        region is the zero value, not a gap in the visible position.
        """
        name = next((name for name,r in regions.items()
                     if r["kind"] == "text" and r.get("field") == field), None)
        if name is None:
            return default
        region = regions[name]
        return default if name not in corrections and not region["complete"] else value(name)

    def one(kind, actor=None, field=None):
        names = [name for name,r in regions.items() if r["kind"] == kind
                 and (actor is None or r.get("player") == actor)
                 and (field is None or r.get("field") == field)]
        if len(names) != 1:
            raise SnapshotError(f"需要唯一的 {kind}/{actor}/{field} 区域。")
        return value(names[0])

    timer = observation.get("action_timer",{})
    if not timer.get("active") and context.get("timer_confirmed") is not True:
        raise SnapshotError("未检测到自己的行动计时器。")
    hand, draw, dora = one("hand",player), one("draw",player), one("dora")
    if not isinstance(draw,list) or len(draw) > 1:
        raise SnapshotError("摸牌槽应为零张或一张牌。")
    rivers = [one("river",actor) for actor in range(4)]
    for river in rivers:
        if not isinstance(river,list):
            raise SnapshotError("牌河修正应为按槽排序的列表。")
        for i,entry in enumerate(river):
            if isinstance(entry,str):
                river[i] = {"tile":entry,"riichi":False,"tedashi":False}
            elif not isinstance(entry,dict) or set(entry)-{"tile","riichi","tedashi"}:
                raise SnapshotError("牌河项格式无效。")
            else:
                entry.setdefault("riichi",False); entry.setdefault("tedashi",False)
                if type(entry["riichi"]) is not bool or type(entry["tedashi"]) is not bool:
                    raise SnapshotError("牌河状态应为布尔值。")
    melds = [one("meld",actor) for actor in range(4)]
    for actor, groups in enumerate(melds):
        if not isinstance(groups,list) or len(groups)>4:
            raise SnapshotError("副露修正应为最多四组事件的列表。")
        for event in groups:
            if not isinstance(event,dict) or event.get("type") not in {"chi","pon","daiminkan","ankan","kakan"}:
                raise SnapshotError("副露修正类型无效。")
            event.setdefault("actor",actor)
            if event["actor"] != actor:
                raise SnapshotError("副露玩家与区域不符。")
            validate_schema(event)
    scores = visible_scores(regions, corrections, config, one, counter("kyotaku",0))
    round_ = one("text",field="round")
    wind = observation.get("self_wind",{}).get("value")
    oya = context.get("oya")
    if oya is None:
        if wind not in {"E","S","W","N"}:
            raise SnapshotError("自风无法确认；请指定当前庄家。")
        oya = (player-"ESWN".index(wind))%4
    # A riichi shows as a sideways river tile and, for the other players, as
    # their white stick on the centre board; either is enough.
    sticks = observation.get("riichi_sticks") or {}
    riichi = context.get("riichi",[any(i["riichi"] for i in river)
                                   or bool(sticks.get(actor) or sticks.get(str(actor)))
                                   for actor, river in enumerate(rivers)])
    buttons = sorted({i["label"] for r in regions.values() if r["kind"] == "buttons"
                      for i in r["items"] if i.get("clickable") is True})
    phase = context.get("phase")
    if phase is None:
        # A 13-tile concealed hand (less three per meld) has not completed a
        # call. Missing button recognition must never make it call_discard.
        waiting_count = 13 - 3 * len(melds[player])
        phase = ("draw" if draw else "reaction" if set(buttons)&{"chi","pon","kan","ron","call"}
                 or len(hand) == waiting_count else "call_discard")
    target, reaction_tile, last_call = None, None, None
    if phase == "reaction":
        target = context.get("target")
        evidence = observation.get("reaction_discard", {})
        if target is None and evidence.get("confirmed"):
            target = evidence.get("target")
        if target is None and "chi" in buttons:
            # A 吃 can only take the discard of the player on your left, so the
            # banner alone names the source; the tile is then that river's last
            # discard. 碰/杠/和 can come from any of the other three seats and
            # still need the highlighted discard or the review dialog.
            target = (player - 1) % 4
            evidence = {"target": target, "confirmed": True,
                        "evidence": "chi_source_is_left_player"}
        if type(target) is not int or target not in range(4) or target == player:
            raise SnapshotError("请指定当前待响应弃牌的玩家；牌河本身不能确定哪一家刚刚出牌"
                                "（面板“核对当前局面”可指定来源）。")
        if not rivers[target]:
            raise SnapshotError("响应来源牌河为空。")
        reaction_tile = rivers[target][-1]["tile"]
        if context.get("target") is None and evidence.get("confirmed") and evidence.get("tile") not in (None, "?") \
                and evidence.get("tile") != reaction_tile:
            raise SnapshotError("当前弃牌高亮与牌河末张不一致，请核对响应来源。")
        if not set(buttons)&{"chi","pon","kan","ron","call"} and context.get("allowed_actions") is None:
            raise SnapshotError("响应操作按钮未能确认，请核对当前可用的吃、碰、杠或和牌。")
    elif phase == "call_discard":
        last_call = context.get("last_call")
        if last_call is None and len(melds[player]) == 1:
            # After a call you discard without drawing. One visible meld names
            # the call that is being completed, which is what 食替 needs.
            last_call = melds[player][0]
        if last_call is None:
            # With several melds the visible order is the layout order, not the
            # order they were called, so 食替 cannot be read off the frame. The
            # call only matters when it forbids a discard: keep it unresolved
            # here and let the rules engine settle it below.
            last_call = AMBIGUOUS_CALL
    snapshot = {"player_id":player,"oya":oya,"bakaze":round_["bakaze"],"kyoku":round_["kyoku"],
                "honba":counter("honba",0),"kyotaku":counter("kyotaku",0),
                "scores":scores,"remaining":one("text",field="remaining"),
                "hand":hand,"draw":draw[0] if draw else None,"dora":dora,"rivers":rivers,
                "melds":melds,"riichi":riichi,"phase":phase,"target":target,
                "reaction_tile":reaction_tile,"last_call":last_call}
    if "riichi_declared" in context:
        snapshot["riichi_declared"] = deepcopy(context["riichi_declared"])
    validate_snapshot(snapshot)
    if last_call is AMBIGUOUS_CALL:
        hand_names = [name for name, r in regions.items() if r["kind"] == "hand" and r.get("player") == player]
        dimmed = None
        if len(hand_names) == 1 and hand_names[0] not in corrections:
            dimmed = [i["label"] for i in regions[hand_names[0]]["items"] if i.get("dimmed")]
        last_call = resolve_last_call(snapshot, legal_ids, dimmed)
        if last_call is None:
            raise SnapshotError("无摸牌槽的出牌阶段需要确认刚完成的吃/碰，以检查食替"
                                "（面板“核对当前局面”可勾选刚完成的那一组）。")
        snapshot["last_call"] = last_call
    allowed = context.get("allowed_actions")
    if allowed is None and phase == "reaction" and "call" not in buttons:
        allowed = sorted(set(buttons)|{"pass"})
    # An unnamed call banner leaves allowed=None: the rules engine lists the
    # legal calls and app.advice checks them against the banners on screen.
    if allowed is not None and (not isinstance(allowed,list) or not set(allowed) <= {
            "discard","chi","pon","kan","riichi","reach","ron","tsumo","pass","kyushukyuhai"}):
        raise SnapshotError("当前可用操作列表无效。")
    return snapshot, allowed


def validate_snapshot(s):
    for key in ("player_id","oya"):
        if type(s[key]) is not int or s[key] not in range(4):
            raise SnapshotError("座位编号必须是 0–3。")
    for key,upper in (("honba",255),("kyotaku",255),("remaining",69)):
        if type(s[key]) is not int or not 0 <= s[key] <= upper:
            raise SnapshotError(f"{key} 数值无效。")
    if s["bakaze"] not in {"E","S"} or type(s["kyoku"]) is not int or s["kyoku"] not in range(1,5):
        raise SnapshotError("当前仅支持四人赤牌半庄的东/南场。")
    if (len(s["scores"])!=4 or any(type(v) is not int or not -100000 <= v <= 200000 or v%100 for v in s["scores"])):
        raise SnapshotError("分数无效。")
    if len(s["riichi"]) != 4 or any(type(v) is not bool for v in s["riichi"]):
        raise SnapshotError("立直状态无效。")
    declared = s.get("riichi_declared",s["riichi"])
    if len(declared) != 4 or any(type(v) is not bool for v in declared):
        raise SnapshotError("立直宣言状态无效。")
    if not 1 <= len(s["dora"]) <= 5 or any(len(r)>24 for r in s["rivers"]):
        raise SnapshotError("宝牌/牌河数量无效。")
    visible = [*s["hand"],*s["dora"],*([s["draw"]] if s["draw"] is not None else [])]
    visible.extend(i["tile"] for r in s["rivers"] for i in r)
    for groups in s["melds"]:
        for m in groups:
            visible.extend(m["consumed"])
            if "pai" in m:
                visible.append(m["pai"])
    if any(t not in TILES for t in visible):
        raise SnapshotError("存在未知或无效牌面，请修正后再分析。")
    counts, exact = Counter(map(base,visible)), Counter(visible)
    if (any(n>4 for n in counts.values()) or any(exact[f"5{x}r"]>1 or exact[f"5{x}"]>3 for x in "mps")):
        raise SnapshotError("可见牌数量冲突，请核对重复识别或赤五。")
    phase = s["phase"]
    if phase not in {"draw","reaction","call_discard"}:
        raise SnapshotError("行动阶段无效。")
    expected = (14 if phase == "call_discard" else 13)-3*len(s["melds"][s["player_id"]])
    if len(s["hand"]) != expected or (phase=="draw") != (s["draw"] is not None):
        raise SnapshotError("手牌数量、摸牌槽与行动阶段不一致。")
