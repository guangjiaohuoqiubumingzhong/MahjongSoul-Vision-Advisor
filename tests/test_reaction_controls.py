from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.advice import ScreenshotAdvice, load_screenshot
from app.engine.snapshot import SnapshotBackend
from app.vision.actions import enabled_action_buttons, highlighted_discard
from app.vision.config import load_config
from app.vision.hand import hand_layout

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
@pytest.mark.parametrize("name,target,tile,allowed", [
    ("chi", 3, "6s", {"chi", "pass"}), ("ron", 1, "1m", {"ron", "pass"})])
def test_real_response_frames_go_to_response_actions_and_keep_entire_self_river(runtime, name, target, tile, allowed):
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    pixels, metadata = load_screenshot(ROOT / f"data/fixtures/screenshot-response/{name}.png", config)
    result = ScreenshotAdvice(config, backend=SnapshotBackend(runtime)).process(pixels, metadata)
    assert result["status"] == "ready", result["reason"]
    snapshot = result["snapshot"]
    assert snapshot["phase"] == "reaction" and snapshot["last_call"] is None
    assert (snapshot["target"], snapshot["reaction_tile"]) == (target, tile)
    assert {c["semantic"] for c in result["decision"]["legal_candidates"]} == allowed
    river = result["observation"]["regions"]["river_self"]
    assert river["complete"] and all(i["label"] != "?" for i in river["items"])
    if name == "ron":
        assert result["recommendation"]["semantic"] == "ron"
        assert [i["slot"] for i in river["items"]][-3:] == [[1, 4], [1, 5], [2, 0]]
        assert [i["label"] for i in river["items"]][-3:] == ["2m", "7p", "8m"]


def test_disabled_grey_glyphs_and_non_action_ocr_cannot_create_buttons():
    pixels = np.zeros((145, 856, 3), np.uint8)
    pixels[45:110, 220:295] = 230
    reader = SimpleNamespace(_infer=lambda _: ("碰", .999))
    assert enabled_action_buttons(pixels, reader) == []
    pixels[45:110, 220:295] = (100, 220, 100)
    assert enabled_action_buttons(pixels, reader)[0]["label"] == "pon"
    reader._infer = lambda _: ("吃碰", .999)
    assert enabled_action_buttons(pixels, reader) == []
    reader._infer = lambda _: ("吃", .3)
    assert enabled_action_buttons(pixels, reader) == []


def test_source_requires_unique_highlight_on_the_last_discard():
    pixels = np.zeros((120, 240, 3), np.uint8)
    regions = {"right": {"kind": "river", "player": 1, "items": [
        {"label": "1m", "bbox": [20, 30, 35, 45]}, {"label": "2m", "bbox": [70, 30, 35, 45]}]},
        "left": {"kind": "river", "player": 3, "items": [{"label": "3m", "bbox": [170, 30, 35, 45]}]}}
    assert not highlighted_discard(pixels, regions, 0)["confirmed"]
    cv2.rectangle(pixels, (16, 26), (59, 79), (80, 240, 80), 4)
    assert not highlighted_discard(pixels, regions, 0)["confirmed"]  # old tile
    cv2.rectangle(pixels, (66, 26), (109, 79), (80, 240, 80), 4)
    assert highlighted_discard(pixels, regions, 0)["target"] == 1
    cv2.rectangle(pixels, (166, 26), (209, 79), (80, 240, 80), 4)
    assert not highlighted_discard(pixels, regions, 0)["confirmed"]


@pytest.mark.parametrize("count", [2, 5, 8, 11])
def test_hand_after_a_call_keeps_the_extra_regular_tile(count):
    pixels = np.zeros((64, 650, 3), np.uint8)
    slots = [[5+45*i, 2, 40, 60] for i in range(13)]
    for x, y, w, h in slots[:count]:
        pixels[y:y+h, x:x+w] = 230
    result = hand_layout(pixels, {"slots": slots, "hand_layout": {"draw_region": "draw", "draw_gap": 15}})
    assert result["hand"] == slots[:count] and not result["draw"] and not result["reasons"]
