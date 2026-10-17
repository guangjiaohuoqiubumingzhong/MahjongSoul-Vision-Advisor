"""Bugs reproduced by the public screenshot replay, with small local fixtures."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.capture.frames import load_image
from app.vision.detector import concealed_back_mask
from app.vision.pjura import PjuraBank

ROOT = Path(__file__).resolve().parents[1]


def test_solid_orange_back_sides_are_excluded_only_inside_the_hand_corridor():
    pixels = np.zeros((80,80,3),np.uint8)
    pixels[10:65,10:65] = 235
    pixels[10:50,10:65] = (15,155,240)
    corridor = np.ones((80,80),np.uint8)*255
    mask = concealed_back_mask(pixels,corridor)
    assert np.all(mask[50:65,10:65] == 255)
    assert not concealed_back_mask(pixels,np.zeros_like(corridor)).any()
    # The orange base of an opened tile is a small strip; its white face stays.
    pixels[10:60,10:65] = 235
    pixels[60:65,10:65] = (15,155,240)
    pixels[20:45,30:40] = (40,100,40)
    assert not concealed_back_mask(pixels,corridor).any()


def test_bright_coloured_edge_does_not_turn_a_white_face_into_background():
    pixels = np.full((60,40,3),220,np.uint8)
    pixels[:10] = (100,190,255)
    model = SimpleNamespace(manifest_sha256="m",manifest={"model_sha256":"x"},provider="test",
                            predict_many=lambda patches:[{"label":"5s","score":.65,"margin":.6,
                                                           "alternatives":[]} for _ in patches])
    legacy = SimpleNamespace(sha256="t",classify=lambda *a,**k:{"label":"?","score":0,"margin":0})
    bank = PjuraBank(legacy,{"confidence":.9,"margin":.04},model=model)
    result = bank.classify(pixels,"meld_top",.9,.04)
    assert result["label"] == "?" and result["pjura_evidence"]["face_fitted"]
    assert not result["background_candidate"]


@pytest.mark.gpu
def test_public_sideways_four_bamboo_is_identified_from_its_glyph():
    from app.vision.config import load_config,native_box
    from app.vision.templates import TemplateBank,rotate
    path = ROOT/"data/public-mahjong/steam-community/images/3803049411.jpg"
    if not path.exists():
        pytest.skip("Optional public screenshot corpus is not installed")
    config = load_config(ROOT/"configs/window-steam-1366-v2.json")
    pixels = load_image(path)
    # Rechecked at native resolution: four bamboo stems and helper index '4'.
    x,y,w,h = native_box([350,25,45,26],1920/1366,pixels.shape)
    face = rotate(pixels[y:y+h,x:x+w],180)
    bank = PjuraBank(TemplateBank(config["templates"],config.get("classifier_manifest")),
                     config["tile_classifier"])
    result = bank.classify(face,"meld_top",.9,.04,(0,90,270))
    assert result["label"] == "4s"
    # The relaxed primary reading would accept 7s here. Preserve the existing
    # unanimous sideways-glyph check before using that fallback.
    assert result["pjura_evidence"]["label"] == "7s"
    assert result["pjura_evidence"]["class_acceptance"]["rule"] == "dominant_candidate"
    assert result["pjura_evidence"]["sideways_glyph"]["accepted"]
    assert result["orientation_confirmed"]
    assert result["orientation_source"] == "sideways_face_geometry"
    assert result["identity_rotation"] in (90,270)

