"""Deterministic, physically possible mjai prefixes with independent expected masks.

These are constructed test scenes, not claimed to be downloaded Mahjong Soul logs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/fixtures"


def start(hand, oya=0):
    assert len(hand) == 13
    return [{"type": "start_game", "names": ["self", "shimocha", "toimen", "kamicha"]},
            {"type": "start_kyoku", "bakaze": "E", "dora_marker": "C", "kyoku": 1,
             "honba": 0, "kyotaku": 0, "oya": oya, "scores": [25000] * 4,
             "tehais": [hand, ["?"] * 13, ["?"] * 13, ["?"] * 13]}]


def draw(actor, tile):
    return {"type": "tsumo", "actor": actor, "pai": tile}


def discard(actor, tile, tsumogiri=True):
    return {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": tsumogiri}


def generate():
    DEST.mkdir(parents=True, exist_ok=True)
    samples = []

    def save(name, events, expectations):
        path = DEST / f"{name}.mjai.jsonl"
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
        samples.append({"name": name, "file": path.name, "player_id": 0,
                        "source": "constructed legal event prefix", "expectations": expectations})

    hand = "1m 2m 4m 5mr 7m 2p 3p 6p 6s 7s 9s E N".split()
    save("discard_red", start(hand) + [draw(0, "W")], [{"event_seq": 3, "legal_ids": [0, 34], "illegal_ids": [37], "semantics": ["discard"]}])
    # Four chi arrangements, including red/normal consumption choices.
    chi_hand = "1m 2m 4m 5mr 5m 6m 2p 3p 4p 6s 7s E E".split()
    chi_prefix = start(chi_hand, 3) + [draw(3, "?"), discard(3, "3m", False)]
    save("chi_red_choices", chi_prefix, [{"event_seq": 4, "legal_ids": [38, 39, 40, 45], "semantics": ["chi", "pass"]}])
    save("chi_then_discard", chi_prefix + [{"type": "chi", "actor": 0, "target": 3, "pai": "3m", "consumed": ["4m", "5mr"]}],
         [{"event_seq": 4, "legal_ids": [38, 45]}, {"event_seq": 5, "semantics": ["discard"], "illegal_ids": [2, 5]}])
    pon_hand = "E E 1m 2m 3m 4p 5pr 6p 7s 8s 9s N N".split()
    pon_prefix = start(pon_hand, 3) + [draw(3, "?"), discard(3, "E", False)]
    save("pon", pon_prefix, [{"event_seq": 4, "legal_ids": [41, 45], "semantics": ["pon", "pass"]}])
    kan_hand = "E E E 1m 2m 3m 4p 5pr 6p 7s 8s 9s N".split()
    save("daiminkan", start(kan_hand, 3) + [draw(3, "?"), discard(3, "E", False)],
         [{"event_seq": 4, "legal_ids": [42, 45], "semantics": ["daiminkan", "pass"]}])
    ankan_prefix = start(kan_hand) + [draw(0, "E")]
    save("ankan", ankan_prefix, [{"event_seq": 3, "legal_ids": [42], "semantics": ["ankan"]}])
    save("ankan_rinshan_dora", ankan_prefix + [{"type": "ankan", "actor": 0, "consumed": ["E"] * 4},
         {"type": "dora", "dora_marker": "F"}, draw(0, "N")],
         [{"event_seq": 3, "legal_ids": [42]}, {"event_seq": 6, "legal_ids": [43], "semantics": ["tsumo"]}])
    save("two_ankan_choices", start("E E E E S S S 1m 2m 3m N N P".split()) + [draw(0, "S")],
         [{"event_seq": 3, "legal_ids": [42], "kan_choice_ids": [27, 28]}])
    kakan = pon_prefix + [{"type": "pon", "actor": 0, "target": 3, "pai": "E", "consumed": ["E", "E"]}, discard(0, "N", False)]
    for actor, tile in ((1, "P"), (2, "F"), (3, "C")):
        kakan += [draw(actor, "?"), discard(actor, tile)]
    kakan += [draw(0, "E")]
    save("kakan", kakan, [{"event_seq": len(kakan), "legal_ids": [42], "semantics": ["kakan"]}])
    ready_hand = "1m 2m 3m 4m 5mr 6m 7p 8p 9p 2s 3s E E".split()
    ready_prefix = start(ready_hand) + [draw(0, "N")]
    save("riichi", ready_prefix, [{"event_seq": 3, "legal_ids": [37], "semantics": ["riichi"]}])
    reach = ready_prefix + [{"type": "reach", "actor": 0}, discard(0, "N"), {"type": "reach_accepted", "actor": 0}]
    for actor, tile in ((1, "P"), (2, "F"), (3, "C")):
        reach += [draw(actor, "?"), discard(actor, tile)]
    reach += [draw(0, "9m")]
    save("riichi_confirmation", reach, [{"event_seq": 4, "legal_ids": [30], "semantics": ["discard"]},
                                       {"event_seq": len(reach), "exact_legal_ids": [8]}])
    tsumo = start(ready_hand) + [draw(0, "4s")]
    save("tsumo", tsumo + [{"type": "hora", "actor": 0, "target": 0, "deltas": [12000, -4000, -4000, -4000]},
                           {"type": "end_kyoku"}, {"type": "end_game"}],
         [{"event_seq": 3, "legal_ids": [43], "semantics": ["tsumo"]}])
    ron = ready_prefix + [{"type": "reach", "actor": 0}, discard(0, "N"), {"type": "reach_accepted", "actor": 0},
                          draw(1, "?"), discard(1, "4s", False)]
    save("ron", ron + [{"type": "hora", "actor": 0, "target": 1, "deltas": [8000, -8000, 0, 0]},
                       {"type": "end_kyoku"}, {"type": "end_game"}],
         [{"event_seq": 8, "legal_ids": [43, 45], "semantics": ["ron", "pass"]}])
    nine = start("1m 9m 1p 9p 1s 9s E S W N P F 2m".split()) + [draw(0, "C")]
    save("kyushukyuhai", nine + [{"type": "ryukyoku", "deltas": [0, 0, 0, 0]}, {"type": "end_kyoku"}, {"type": "end_game"}],
         [{"event_seq": 3, "legal_ids": [44], "semantics": ["kyushukyuhai"]}])
    for filename in ("majsoul-gold-south", "tenhou-hanchan"):
        if (DEST / f"{filename}.mjai.jsonl").exists():
            for player_id in range(4):
                samples.append({"name": f"{filename}-player{player_id}", "file": f"{filename}.mjai.jsonl",
                                "player_id": player_id,
                                "source": "mjai-reviewer convlog upstream test export at 2dc5ec5c8b28517cfb45f57eb21536d9a8f67aa9",
                                "expectations": []})
    (DEST / "manifest.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate()
