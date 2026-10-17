"""Larger same-aspect clients: calibrated layout, native pixels for the classifier."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.vision.config import VisionError, native_box, pixel_rect, scaled_client, to_reference
from app.vision.detector import global_box
from app.vision.pjura import PjuraBank
from app.vision.recognizer import overlay
from app.vision.templates import rotate

CONFIG = {"client_size": [1366, 768]}


@pytest.mark.parametrize("size,expected", [((1366, 768), 1.0), ((1920, 1080), 1920 / 1366),
                                           ((1600, 900), 1600 / 1366), ((2560, 1440), 2560 / 1366),
                                           ((1280, 720), None), ((1920, 1200), None), ((1366, 800), None)])
def test_only_larger_same_aspect_clients_are_accepted(size, expected):
    value = scaled_client(*size, CONFIG)
    assert value == (pytest.approx(expected) if expected else None)


def test_native_frame_is_downscaled_for_layout_and_kept_for_the_classifier():
    native = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), np.uint8)
    reference, kept, scale = to_reference(native, CONFIG)
    assert reference.shape == (768, 1366, 3) and kept is native and scale == pytest.approx(1920 / 1366)
    same, none, one = to_reference(reference, CONFIG)
    assert same is reference and none is None and one == 1.0
    with pytest.raises(VisionError):
        to_reference(np.zeros((1200, 1920, 3), np.uint8), CONFIG)


@pytest.mark.parametrize("size", [(1280,720),(1153,651),(960,540)])
def test_smaller_offline_images_keep_their_native_tile_pixels(size):
    width,height = size
    pixels = np.random.default_rng(2).integers(0,255,(height,width,3),np.uint8)
    reference,native,scale = to_reference(pixels,CONFIG)
    assert reference.shape == (768,1366,3) and native is pixels
    assert scale == pytest.approx(width/1366)
    assert native_box([100,200,40,60],scale,native.shape)[2] > 0
    # Live window capture retains its calibrated minimum-size check.
    assert scaled_client(width,height,CONFIG) is None


def scene(width, height):
    """A deterministic table-like image; identical content at any resolution."""
    image = np.zeros((height, width, 3), np.uint8)
    for i in range(40):
        x, y = (i * 97 % 1366) / 1366, (i * 61 % 768) / 768
        cv2.rectangle(image, (int(x * width), int(y * height)), (int((x + .03) * width), int((y + .07) * height)),
                      (40 + 5 * i, 255 - 4 * i, (i * 37) % 255), -1)
        cv2.putText(image, str(i % 10), (int((x + .005) * width), int((y + .05) * height)),
                    cv2.FONT_HERSHEY_SIMPLEX, width / 1366, (0, 0, 0), 2)
    return image


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_native_crops_show_the_same_tile_in_the_same_pose(degrees):
    reference, native = scene(1366, 768), scene(1920, 1080)
    scale = 1920 / 1366
    rect = [0.28, 0.26, 0.13, 0.26]
    x, y, w, h = pixel_rect(rect, 1366, 768)
    region = rotate(reference[y:y+h, x:x+w], degrees)
    box = [12, 20, 45, 60] if degrees in (0, 180) else [20, 12, 60, 45]
    bx, by, bw, bh = box
    small = region[by:by+bh, bx:bx+bw]
    nx, ny, nw, nh = native_box(global_box(box, (x, y), (w, h), degrees), scale, native.shape)
    large = rotate(native[ny:ny+nh, nx:nx+nw], degrees)
    assert large.shape[0] == pytest.approx(small.shape[0] * scale, abs=2)
    assert large.shape[1] == pytest.approx(small.shape[1] * scale, abs=2)
    resized = cv2.resize(large, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    difference = np.abs(resized.astype(int) - small.astype(int)).mean()
    assert difference < 12, difference


class RecordingModel:
    manifest_sha256 = "m"
    manifest = {"model_sha256": "x"}
    provider = "test"

    def __init__(self):
        self.seen = []

    def predict_many(self, crops):
        self.seen.extend(crops)
        return [{"label": "3s", "score": .99, "margin": .9, "alternatives": []} for _ in crops]


def test_only_the_network_reads_native_crops():
    seen_by_templates = []
    legacy = SimpleNamespace(classify=lambda pixels, *a, **k: seen_by_templates.append(pixels) or {
        "label": "3s", "score": .9, "margin": .2, "face_rotation": 0}, sha256="t")
    model = RecordingModel()
    bank = PjuraBank(legacy, {"confidence": .9, "margin": .04, "rectification": "off"}, model=model)
    small, large = np.full((50, 30, 3), 200, np.uint8), np.full((70, 42, 3), 200, np.uint8)
    result = bank.classify_many([small], "river_left", .9, .04, (0,), model_crops=[large])[0]
    assert seen_by_templates[0].shape == small.shape
    assert model.seen[0].shape == large.shape
    assert result["pjura_evidence"]["input_size"] == [42, 70]
    with pytest.raises(VisionError):
        bank.classify_many([small], "river_left", .9, .04, (0,), model_crops=[])


def test_overlay_draws_calibrated_boxes_on_a_native_frame():
    observation = {"reference_size": [1366, 768], "regions": {}}
    assert overlay(np.zeros((1080, 1920, 3), np.uint8), observation).shape == (768, 1366, 3)


def test_window_screenshot_with_title_bar_is_cropped_at_any_supported_size(tmp_path):
    from app.advice import load_screenshot
    from app.capture.frames import save_image
    window = np.zeros((1080 + 32, 1920 + 2, 3), np.uint8)
    path = tmp_path / "window.png"
    save_image(path, window)
    pixels, metadata = load_screenshot(path, CONFIG)
    assert pixels.shape == (1080, 1920, 3) and metadata["window_crop"] == [1, 31, 1920, 1080]


def test_smaller_offline_window_is_cropped_without_accepting_a_wrong_aspect(tmp_path):
    from app.advice import load_screenshot
    from app.capture.frames import save_image
    path = tmp_path/"small-window.png"
    save_image(path,np.zeros((752,1282,3),np.uint8))
    pixels,metadata = load_screenshot(path,CONFIG)
    assert pixels.shape == (720,1280,3) and metadata["window_crop"] == [1,31,1280,720]
    with pytest.raises(VisionError):
        to_reference(np.zeros((800,1280,3),np.uint8),CONFIG)
