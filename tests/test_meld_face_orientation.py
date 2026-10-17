"""A side-view called tile can have a portrait body despite lying sideways."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.advice import ScreenshotAdvice, load_screenshot
from app.engine.snapshot import SnapshotBackend
from app.vision.config import load_config
from app.vision.recognizer import group_melds

ROOT = Path(__file__).resolve().parents[1]
RIGHT = {"chi_source_by_called_position":{"0":0},
         "pon_source_by_called_position":{"0":0, "1":3, "2":2}}
UPRIGHT = [[15, 8], [30, 3], [32, 43], [15, 49]]
CALLED = [[16, 9], [38, 3], [40, 32], [16, 41]]


def triplet(kind="chi", called=0, corner_shift=0):
    labels = ["5mr", "4m", "6m"] if kind == "chi" else ["6p"]*3
    return [{"label":label, "horizontal":None, "canonical_bbox":[42*i, 40, 42, 60],
             "legacy_evidence":{"face_quad_local":np.roll(
                 np.asarray(CALLED if i == called else UPRIGHT), corner_shift, axis=0).tolist()}}
            for i, label in enumerate(labels)]


@pytest.mark.parametrize("kind,called,target", [("chi",0,0), ("pon",0,0), ("pon",1,3), ("pon",2,2)])
@pytest.mark.parametrize("corner_shift", range(4))
def test_foreshortened_called_face_uses_physical_edges_independent_of_glyph_order(kind,called,target,corner_shift):
    group = group_melds(triplet(kind,called,corner_shift), RIGHT)[0]
    assert group["status"] == "confirmed" and group["type"] == kind
    assert group["target"] == target and group["called_positions"] == [called]
    assert group["called_position_evidence"] == "face_relative_geometry"


@pytest.mark.parametrize("problem", ["equal", "two_wide", "unequal_upright", "missing_face", "invalid_face",
                                   "nonfinite", "degenerate", "invalid_chi", "missing_mapping",
                                   "wrong_chi_position", "upright_template", "offset_body", "unknown_tile"])
def test_incomplete_or_conflicting_side_view_evidence_does_not_confirm_a_meld(problem):
    items, region = triplet(), deepcopy(RIGHT)
    if problem == "equal":
        items[0]["legacy_evidence"]["face_quad_local"] = deepcopy(UPRIGHT)
    elif problem == "two_wide":
        items[1]["legacy_evidence"]["face_quad_local"] = deepcopy(CALLED)
    elif problem == "unequal_upright":
        items[2]["legacy_evidence"]["face_quad_local"] = [[0,0], [27,0], [27,45], [0,45]]
    elif problem == "missing_face":
        items[1]["legacy_evidence"] = {}
    elif problem == "invalid_face":
        items[1]["legacy_evidence"]["face_quad_local"].pop()
    elif problem == "nonfinite":
        items[1]["legacy_evidence"]["face_quad_local"][0][0] = float("nan")
    elif problem == "degenerate":
        items[1]["legacy_evidence"]["face_quad_local"] = [[0,0]]*4
    elif problem == "invalid_chi":
        items[2]["label"] = "9m"
    elif problem == "missing_mapping":
        region.pop("chi_source_by_called_position")
    elif problem == "wrong_chi_position":
        items = triplet(called=1)
    elif problem == "upright_template":
        items[0]["horizontal"] = False
    elif problem == "offset_body":
        items[1]["canonical_bbox"][1] += 90
    elif problem == "unknown_tile":
        items[2]["label"] = "?"
    assert all(group["status"] == "needs_confirmation" for group in group_melds(items, region))


@pytest.mark.gpu
@pytest.mark.parametrize("turn,semantic", [(293,"discard"), (300,"pass")])
@pytest.mark.parametrize("size", [(1920,1080), (1366,768)])
def test_user_right_chi_frames_produce_advice_with_the_red_called_tile(runtime,turn,semantic,size):
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    pixels, metadata = load_screenshot(ROOT / f"data/fixtures/right-chi-perspective/turn-{turn}.png", config)
    if size != (1920,1080):
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    result = ScreenshotAdvice(config, backend=SnapshotBackend(runtime)).process(pixels, metadata)
    meld = result["observation"]["regions"]["meld_right"]
    assert meld["complete"] and len(meld["items"]) == 3
    group = meld["groups"][0]
    assert group["labels"] == ["5mr", "4m", "6m"] and group["called_positions"] == [0]
    assert group["called_position_evidence"] == "face_relative_geometry"
    assert result["status"] == "ready", result["reason"]
    assert result["recommendation"]["semantic"] == semantic and result["corrections"] == {}
    assert result["snapshot"]["melds"][1] == [
        {"type":"chi", "actor":1, "target":0, "pai":"5mr", "consumed":["4m", "6m"]}]
