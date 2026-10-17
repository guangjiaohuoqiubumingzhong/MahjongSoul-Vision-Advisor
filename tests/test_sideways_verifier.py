from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.capture.frames import load_image
from app.vision.config import VisionError,load_config,native_box
from app.vision.pjura import PjuraBank
from app.vision.sideways import SidewaysVerifier
from app.vision.templates import rotate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"models/sideways-glyph/manifest.json"


def sideways_four():
    path = ROOT/"data/public-mahjong/steam-community/images/3803049411.jpg"
    if not path.exists():
        pytest.skip("Optional public screenshot corpus is not installed")
    pixels = load_image(path)
    x,y,w,h = native_box([350,25,45,26],1920/1366,pixels.shape)
    return rotate(pixels[y:y+h,x:x+w],180)


def wrapped(score=.6,label="7s",sideways=None):
    model = SimpleNamespace(manifest_sha256="m",manifest={"model_sha256":"x"},provider="test",
                            predict_many=lambda crops:[{"label":label,"score":score,"margin":.5,
                                                         "alternatives":[]} for _ in crops])
    legacy = SimpleNamespace(sha256="t",classify=lambda *a,**k:{"label":"?","score":0,"margin":0})
    result = PjuraBank(legacy,{"confidence":.9,"margin":.04},model=model)
    result.sideways_verifier = sideways
    return result


def test_real_sideways_four_has_unanimous_glyph_votes_at_both_rotations():
    reading = SidewaysVerifier(MANIFEST).verify(sideways_four())
    assert reading["accepted"] and reading["label"] == "4s"
    assert [r["rotation"] for r in reading["readings"]] == [90,270]
    assert all(r["votes"] == ["4s"]*3 for r in reading["readings"])


def test_training_has_no_public_test_or_external_image_hashes():
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [json.loads(s) for s in (ROOT/"data/public-mahjong/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    excluded = {r["sha256"] for r in records if r["source"]!="pjura" or "/test/" in r["path"]}
    assert not excluded.intersection(s["sha256"] for s in value["sources"])
    assert all("/train/" in s["path"] for s in value["sources"])


def test_blank_white_side_cannot_be_confirmed_as_a_white_dragon_by_shape():
    reading = SidewaysVerifier(MANIFEST).verify(np.full((30,80,3),232,np.uint8))
    assert not reading["accepted"]
    assert reading["reason"] == "blank_has_no_independent_glyph"


@pytest.mark.parametrize("stem",["3800908090","3803922882","3812161238"])
def test_real_green_back_sides_are_not_rescued_as_white_dragons(stem):
    path = ROOT/f"data/public-mahjong/steam-community/images/{stem}.jpg"
    if not path.exists():
        pytest.skip("Optional public screenshot corpus is not installed")
    pixels = load_image(path)
    x,y,w,h = native_box([1174,232,20,25],pixels.shape[1]/1366,pixels.shape)
    side = rotate(pixels[y:y+h,x:x+w],270)
    reading = SidewaysVerifier(MANIFEST).verify(side)
    assert not reading["accepted"]
    assert reading["reason"] == "blank_has_no_independent_glyph"


@pytest.mark.parametrize("role",["hand","draw","dora","choices"])
def test_front_hand_and_draw_never_use_sideways_fallback(role):
    def unexpected(*a):
        pytest.fail("A frontal tile must not invoke sideways verification")
    bank = wrapped(sideways=SimpleNamespace(verify=unexpected))
    assert bank.classify(np.full((40,80,3),232,np.uint8),role,.9,.04,(0,90,270))["label"] == "?"


def test_confident_primary_identity_is_preserved():
    def unexpected(*a):
        pytest.fail("Confident primary identity must not invoke sideways verification")
    bank = wrapped(score=.99,label="4s",sideways=SimpleNamespace(verify=unexpected))
    assert bank.classify(sideways_four(),"meld_top",.9,.04,(0,90,270))["label"] == "4s"


def test_uncertain_shape_evidence_cannot_replace_the_network_unknown():
    bank = wrapped(sideways=SimpleNamespace(verify=lambda *a:{"accepted":False,"reason":"glyph_rotations_disagree"}))
    result = bank.classify(sideways_four(),"meld_top",.9,.04,(0,90,270))
    assert result["label"] == "?" and result["reason"] == "pjura_low_confidence"


def test_reference_archive_checksum_is_enforced(tmp_path):
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["features"]["path"] = str(MANIFEST.parent/value["features"]["path"])
    value["features"]["sha256"] = "0"*64
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(VisionError,match="字形校验"):
        SidewaysVerifier(path)


def test_geometry_verified_five_still_requires_original_red_colour_evidence():
    proof = {"accepted":True,"label":"5s","readings":[{"similarity":.96,"class_gap":.1}]*2}
    pixels = np.full((40,80,3),232,np.uint8)
    pixels[10:30,20:60] = (30,30,200)
    result = wrapped(sideways=SimpleNamespace(verify=lambda *a:deepcopy(proof))).classify(
        pixels,"meld_top",.9,.04,(0,90,270))
    assert result["label"] == "5sr"
    assert result["score_kind"] == "glyph_cosine_not_probability"
    assert result["pjura_evidence"]["label"] == "7s"
