"""White-dragon pons have no glyph direction; their visible faces give the source."""
from copy import deepcopy
from pathlib import Path

import cv2
import pytest

from app.advice import ScreenshotAdvice, load_screenshot
from app.engine.snapshot import SnapshotBackend
from app.vision.config import load_config
from app.vision.recognizer import group_melds

ROOT = Path(__file__).resolve().parents[1]
RIGHT = {"pon_source_by_called_position":{"0":0, "1":3, "2":2}}
UPRIGHT = [[16, 8], [31, 3], [33, 44], [16, 50]]
CALLED = [[1, 9], [23, 3], [24, 32], [1, 40]]


def white_triplet(called):
    return [{"label":"P", "horizontal":False, "canonical_bbox":[35*i, 40, 35, 51],
             "legacy_evidence":{"blank_face":True,
                                "face_quad_local":deepcopy(CALLED if i == called else UPRIGHT)}}
            for i in range(3)]


@pytest.mark.parametrize("called,source", [(0,0), (1,3), (2,2)])
def test_foreshortened_white_triplet_source_uses_face_edges_not_portrait_body_boxes(called,source):
    group = group_melds(white_triplet(called), RIGHT)[0]
    assert group["status"] == "confirmed" and group["target"] == source
    assert group["called_positions"] == [called]
    assert group["called_position_evidence"] == "blank_face_relative_geometry"


@pytest.mark.parametrize("problem", ["equal", "two_wide", "missing_face", "nonblank", "kan"])
def test_white_meld_without_unique_face_geometry_stays_unresolved(problem):
    items = white_triplet(1)
    if problem == "equal":
        items = white_triplet(None)
    elif problem == "two_wide":
        items[0]["legacy_evidence"]["face_quad_local"] = deepcopy(CALLED)
    elif problem == "missing_face":
        items[0]["legacy_evidence"].pop("face_quad_local")
    elif problem == "nonblank":
        items[1]["legacy_evidence"]["blank_face"] = False
    elif problem == "kan":
        extra = deepcopy(items[-1])
        extra["canonical_bbox"][0] = 105
        items.append(extra)
    group = group_melds(items, RIGHT)[0]
    assert group["status"] == "needs_confirmation" and group["target"] is None


@pytest.mark.gpu
@pytest.mark.parametrize("turn", [62,58])
@pytest.mark.parametrize("size", [(1366,768), (1920,1080)])
def test_user_white_meld_frames_return_discard_advice_without_corrections(runtime,turn,size):
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    pixels, metadata = load_screenshot(ROOT / f"data/fixtures/white-meld/turn-{turn}.png", config)
    if size != (1920,1080):
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    result = ScreenshotAdvice(config, backend=SnapshotBackend(runtime)).process(pixels, metadata)
    meld = result["observation"]["regions"]["meld_right"]
    assert meld["complete"] and len(meld["items"]) == 3
    group = meld["groups"][0]
    assert group["labels"] == ["P"] * 3 and group["type"] == "pon"
    assert group["target"] == 3 and group["called_positions"] == [1]
    assert group["called_position_evidence"] == "blank_face_relative_geometry"
    assert result["status"] == "ready", result["reason"]
    assert result["recommendation"]["semantic"] == "discard" and result["corrections"] == {}
