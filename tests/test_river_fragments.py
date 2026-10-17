"""A real seven-dot face must not create a phantom six-dot discard."""
from pathlib import Path

import cv2
import pytest

from app.advice import ScreenshotAdvice, load_screenshot
from app.engine.snapshot import SnapshotBackend
from app.vision.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)])
def test_split_seven_dot_face_yields_complete_river_and_a_discard_advice(runtime, size):
    config = load_config(ROOT / "configs/window-steam-1366-v2.json")
    pixels, metadata = load_screenshot(ROOT / "data/fixtures/river-fragments/left-7p-1920.png", config)
    if size != (1920, 1080):
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    result = ScreenshotAdvice(config, backend=SnapshotBackend(runtime)).process(pixels, metadata)
    river = result["observation"]["regions"]["river_left"]
    assert river["complete"]
    assert [item["label"] for item in river["items"]] == ["9m", "9s", "9s", "W", "2s", "7p", "C"]
    assert [item["slot"] for item in river["items"]] == [[0, i] for i in range(6)] + [[1, 0]]
    assert len(river["merged_fragments"]) == 1
    meld = result["observation"]["regions"]["meld_self"]
    assert meld["complete"] and meld["groups"][0]["labels"] == ["F", "F", "F"]
    assert any(item.get("detector_reason") == "candidate_at_region_boundary"
               for item in meld["background_candidates"])
    assert result["status"] == "ready", result["reason"]
    assert result["snapshot"]["phase"] == "draw"
    assert result["recommendation"]["semantic"] == "discard"
