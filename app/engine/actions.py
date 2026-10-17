from __future__ import annotations

from collections import Counter
from itertools import product

from app.state.events import TILES, base


def _combinations(tiles, hand):
    choices = [[tile, tile + "r"] if tile in ("5m", "5p", "5s") else [tile] for tile in tiles]
    results = []
    for combo in product(*choices):
        counter = Counter(combo)
        if all(counter[t] <= hand[t] for t in counter) and sorted(combo) not in results:
            results.append(sorted(combo))
    # Same preference as Mortal: consume available red fives first.
    return sorted(results, key=lambda combo: -sum(t.endswith("r") for t in combo))


def action_variants(index, state, ledger):
    actor = ledger.player_id
    target = state.last_cans.target_actor
    if index <= 36:
        tile = TILES[index]
        return [{"type": "dahai", "actor": actor, "pai": tile,
                 "tsumogiri": state.last_self_tsumo() == tile}]
    if index == 37:
        return [{"type": "reach", "actor": actor}]
    if index in (38, 39, 40, 41):
        pai = state.last_kawa_tile()
        tile = base(pai)
        if index == 41:
            consumed = [tile, tile]
            kind = "pon"
        else:
            rank = int(tile[0])
            offsets = {38: (1, 2), 39: (-1, 1), 40: (-2, -1)}[index]
            consumed = [f"{rank + delta}{tile[-1]}" for delta in offsets]
            kind = "chi"
        return [{"type": kind, "actor": actor, "target": target, "pai": pai, "consumed": combo}
                for combo in _combinations(consumed, ledger.hand)]
    if index == 42:
        if state.last_cans.can_daiminkan:
            pai = state.last_kawa_tile()
            return [{"type": "daiminkan", "actor": actor, "target": target, "pai": pai, "consumed": combo}
                    for combo in _combinations([base(pai)] * 3, ledger.hand)]
        variants = []
        for tile in state.ankan_candidates():
            for combo in _combinations([tile] * 4, ledger.hand):
                variants.append({"type": "ankan", "actor": actor, "consumed": combo})
        for tile in state.kakan_candidates():
            pon = next(m for m in ledger.melds[actor] if m["type"] == "pon" and base(m["pai"]) == tile)
            for combo in _combinations([tile], ledger.hand):
                variants.append({"type": "kakan", "actor": actor, "pai": combo[0],
                                 "consumed": [pon["pai"], *pon["consumed"]]})
        return variants
    if index == 43:
        return [{"type": "hora", "actor": actor, "target": target}]
    if index == 44:
        return [{"type": "ryukyoku"}]
    if index == 45:
        return [{"type": "none"}]
    raise ValueError("未知模型动作编号。")


def semantic(event):
    kind = event["type"]
    if kind == "hora":
        return "tsumo" if event["actor"] == event["target"] else "ron"
    return {"none": "pass", "ryukyoku": "kyushukyuhai", "reach": "riichi",
            "dahai": "discard"}.get(kind, kind)
