"""The three reported call/response screenshots, from real game pixels.

Fixtures: data/fixtures/no-draw-call-discard. The whole-window screenshots the
user supplied are cropped to the game client and resampled to 1920x1080 by
scripts/build_call_discard_fixtures.py, so they behave like a real capture.

Reported problems:
* after a call there is no draw slot, and the next discard must still be advised;
* with 吃/碰/和 buttons on screen the advice stopped at
  "请指定当前待响应弃牌的玩家".

What the fixtures show is recorded here; the decision logic itself is covered by
the hand-built observations below, which use the same tile content.
"""
from pathlib import Path

import pytest

from app.capture.frames import load_image
from app.state.snapshot import build_snapshot, SnapshotError
from app.vision.config import load_config
from app.vision.recognizer import StaticRecognizer
from app.vision.templates import TemplateBank
from app.vision.text import TextReader

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/fixtures/no-draw-call-discard"
# The wind glyph is drawn beside the board centre on these captures and does not
# read; the panel lets the user set the dealer, so the tests supply it.
DEALER_CONTEXT = {"oya": 2}


@pytest.fixture(scope="module")
def recognizer():
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    return StaticRecognizer(config, TemplateBank(config["templates"], config.get("classifier_manifest")), TextReader()), config


def analyse(recognizer, name):
    recognizer, config = recognizer
    pixels = load_image(FIXTURES / f"{name}.png")
    observation = recognizer.recognize(pixels, {})
    assert observation["action_timer"]["active"], "the fixtures show our own timer"
    return observation


def observation_with(hand, groups, **overrides):
    """A complete visible position with the hand and melds under test."""
    regions = {
        "hand": {"kind": "hand", "player": 0, "complete": True,
                 "items": [{"label": label} for label in hand]},
        "draw": {"kind": "draw", "player": 0, "complete": True, "items": []},
        "dora": {"kind": "dora", "player": None, "complete": True, "items": [{"label": "F"}]},
        "buttons": {"kind": "buttons", "complete": True, "items": []},
        "round": {"kind": "text", "complete": True, "field": "round",
                  "text": {"value": {"bakaze": "E", "kyoku": 1}}},
        "remaining": {"kind": "text", "complete": True, "field": "remaining", "text": {"value": 23}},
        "honba": {"kind": "text", "complete": False, "field": "honba", "text": {"value": None}},
        "kyotaku": {"kind": "text", "complete": False, "field": "kyotaku", "text": {"value": None}},
    }
    for actor in range(4):
        regions[f"river_{actor}"] = {"kind": "river", "player": actor, "complete": True, "items": []}
        regions[f"meld_{actor}"] = {"kind": "meld", "player": actor, "complete": True,
                                    "items": [], "groups": []}
        regions[f"score_{actor}"] = {"kind": "text", "player": actor, "complete": True,
                                     "field": "score", "text": {"value": 25000}}
    regions["meld_0"]["groups"] = groups
    regions.update(overrides)
    return {"regions": regions, "self_wind": {"value": "S"}, "action_timer": {"active": True},
            "reaction_discard": {"confirmed": False, "candidates": [], "glowing_tiles": [],
                                 "evidence": "last_discard_green_outline"}}


CHI_GROUP = {"labels": ["2p", "3p", "4p"], "type": "chi", "target": 3,
             "called_positions": [0], "status": "confirmed"}


def test_the_chi_fixture_is_detected_as_a_reaction_with_call_buttons(recognizer):
    """Real pixels: 13-tile hand, four complete rivers, one call banner."""
    observation = analyse(recognizer, "chi-buttons-1920")
    regions = observation["regions"]
    assert len(regions["hand"]["items"]) == 13 and regions["hand"]["complete"]
    assert [item["label"] for item in regions["draw"]["items"]] == []
    rivers = [r for r in regions.values() if r["kind"] == "river"]
    assert len(rivers) == 4 and all(r["complete"] for r in rivers)
    assert regions["buttons"]["items"], "the 吃/碰 banner must be detected"
    assert regions["round"]["complete"] and regions["round"]["text"]["value"] == {"bakaze": "E", "kyoku": 1}


def test_the_after_call_fixture_has_no_draw_slot_and_two_melds(recognizer):
    """Real pixels after the call: 8 concealed tiles, our chi and our pon."""
    observation = analyse(recognizer, "after-chi-no-advice-1920")
    regions = observation["regions"]
    assert [item["label"] for item in regions["draw"]["items"]] == []
    assert regions["hand"]["complete"], regions["hand"].get("layout_reasons")
    groups = regions["meld_self"]["groups"]
    assert [(g["labels"], g["type"]) for g in groups] == [(["2p", "3p", "4p"], "chi"), (["E", "E", "E"], "pon")]


def test_a_call_discard_with_one_meld_still_reaches_a_snapshot(recognizer):
    """After a call you discard without drawing; the visible meld names the call.

    Figure one's position: the 2p3p4p chi is on the table and the hand holds the
    tiles left after it. The engine must accept it as call_discard, not stop at
    "无摸牌槽的出牌阶段需要确认刚完成的吃/碰".
    """
    observation = observation_with(["1m", "2m", "3m", "5m", "6m", "7m", "9m", "2p", "3p", "4p", "E"],
                                   [CHI_GROUP])
    snapshot, _ = build_snapshot(observation, recognizer[1], {}, DEALER_CONTEXT)
    assert snapshot["phase"] == "call_discard"
    assert snapshot["last_call"]["type"] == "chi"
    assert sorted(snapshot["last_call"]["consumed"]) == ["3p", "4p"]
    assert snapshot["last_call"]["pai"] == "2p", "the called tile is the sideways one"
    assert snapshot["honba"] == 0 and snapshot["kyotaku"] == 0, "counters the game hides at zero"


def test_several_melds_ask_which_call_was_just_completed(recognizer):
    """The meld list is layout order, not call order, so 食替 needs review."""
    observation = observation_with(["1m", "2m", "3m", "5m", "6m", "7m", "9m", "2p"],
                                   [CHI_GROUP, {"labels": ["E", "E", "E"], "type": "pon", "target": 2,
                                                "called_positions": [1], "status": "confirmed"}])
    with pytest.raises(SnapshotError, match="确认刚完成的吃/碰"):
        build_snapshot(observation, recognizer[1], {}, DEALER_CONTEXT)
    # Two open melds leave eight concealed tiles before the call discard.
    snapshot, _ = build_snapshot(observation, recognizer[1], {}, {**DEALER_CONTEXT, "last_call": CHI_GROUP})
    assert snapshot["phase"] == "call_discard"
    # An impossible count must still fail even with an explicitly named call.
    observation["regions"]["hand"]["items"].extend({"label":tile} for tile in ("3p","4p","E"))
    with pytest.raises(SnapshotError, match="手牌数量"):
        build_snapshot(observation, recognizer[1], {}, {**DEALER_CONTEXT, "last_call": CHI_GROUP})


def test_a_chi_banner_alone_names_the_source(recognizer):
    """A 吃 can only take your left player's discard, so the source is certain."""
    observation = observation_with(["1m", "2m", "3m", "5m", "6m", "7m", "9m", "1p", "2p", "6p", "7p", "E", "E"],
                                   [])
    observation["regions"]["buttons"]["items"] = [
        {"label": "chi", "state": "enabled", "clickable": True, "evidence": "reaction_banner_colour"}]
    observation["regions"]["river_3"]["items"] = [{"label": "9s"}]
    snapshot, allowed = build_snapshot(observation, recognizer[1], {}, DEALER_CONTEXT)
    assert snapshot["phase"] == "reaction"
    assert (snapshot["target"], snapshot["reaction_tile"]) == (3, "9s")
    assert allowed == ["chi", "pass"]


def test_a_pon_banner_without_a_highlight_still_asks_for_the_source(recognizer):
    """碰 can come from any of three seats: the frame stays for review."""
    observation = observation_with(["1m", "2m", "3m", "5m", "6m", "7m", "9m", "1p", "2p", "6p", "7p", "E", "E"], [])
    observation["regions"]["buttons"]["items"] = [
        {"label": "pon", "state": "enabled", "clickable": True, "evidence": "reaction_banner_colour"}]
    with pytest.raises(SnapshotError, match="待响应弃牌"):
        build_snapshot(observation, recognizer[1], {}, DEALER_CONTEXT)
