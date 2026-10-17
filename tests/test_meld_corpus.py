"""Frozen real-world structure regressions and controlled physical poses."""
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from app.vision.config import load_config
from app.vision.meld_geometry import relative_called_positions
from app.vision.recognizer import group_melds
from scripts.evaluate_meld_corpus import evaluate,score
from scripts.stress_meld_geometry import controlled_items

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT/"configs/window-steam-1366-v2.json")
DATA = json.loads((ROOT/"data/meld-orientation/manifest.json").read_text(encoding="utf-8"))
BASELINE = json.loads((ROOT/"data/meld-orientation/baseline.json").read_text(encoding="utf-8"))
BASE_CORRECT = {r["name"] for r in BASELINE["rows"] if r["correct"]}
FIXED = {"steam-right-three-dot-kan-seven-pon","steam-top-two-one-three-sou"}


def test_frozen_baseline_and_current_cases_use_the_same_labels_and_image_hashes():
    assert hashlib.sha256((ROOT/"data/meld-orientation/manifest.json").read_bytes()).hexdigest() == BASELINE["manifest_sha256"]


@pytest.mark.parametrize("actor",range(4))
@pytest.mark.parametrize("kind",("pon","daiminkan"))
def test_all_called_positions_under_projective_pose_and_permuted_glyph_corners(actor,kind):
    seat = ("self","right","top","left")[actor]
    for called,shift,perspective,blank in itertools.product(range(4 if kind=="daiminkan" else 3),range(4),(-.0004,0,.0004),(False,True)):
        items = controlled_items(actor,kind,called,shift,perspective,blank)
        group = group_melds(items,CONFIG["regions"]["meld_"+seat])[0]
        target = (actor+((3,2,2,1)[called] if kind=="daiminkan" else (3,2,1)[called]))%4
        assert (group["type"],group["target"],group["called_positions"],group["status"]) == (kind,target,[called],"confirmed")


def test_missing_legacy_outline_uses_fitted_geometry_without_using_network_rotation():
    items = controlled_items(1,"pon",0,0,0,False)
    expected = relative_called_positions(items)
    for item in items:
        item["pjura_evidence"] = {"face_fitted":True,"rectified_quad":item["legacy_evidence"].pop("face_quad_local"),
                                  "face_rotation":180,"label":"9m"}
    actual = relative_called_positions(items)
    assert actual[0] == expected[0] == [0]
    assert actual[1]["outline_source"] == "pjura"


def test_two_complete_outlines_disagreeing_on_called_position_are_rejected():
    items = controlled_items(1,"pon",0,0,0,False)
    other = controlled_items(1,"pon",1,0,0,False)
    for item,conflict in zip(items,other):
        item["pjura_evidence"] = {"face_fitted":True,"rectified_quad":conflict["legacy_evidence"]["face_quad_local"]}
    assert relative_called_positions(items) is None


def test_partition_of_a_kan_and_one_weak_pon_preserves_the_existing_recovery_policy():
    items = [{"label":"3p" if i<4 else "7p", "horizontal":i in (0,5),
              "canonical_bbox":[i*40,0,40,60]} for i in range(7)]
    items[-1].update(label="?",best_label="7p",score=.8,margin=.7,reason="pjura_low_confidence")
    groups = group_melds(items,CONFIG["regions"]["meld_right"])
    assert [(g["type"],g["status"]) for g in groups] == [("daiminkan","confirmed"),("pon","confirmed")]
    assert items[-1]["label"] == "7p" and items[-1]["unsure_reading"]["score"] == .8


def test_partition_proposals_cannot_turn_two_unknowns_into_confirmed_tiles():
    items = [{"label":label,"horizontal":i in (0,3),"canonical_bbox":[i*40,0,40,60]}
             for i,label in enumerate(("?","6p","?","8m","8m","8m"))]
    for i,label in ((0,"5p"),(2,"7p")):
        items[i].update(best_label=label,score=.8,margin=.7,reason="pjura_low_confidence")
    groups = group_melds(items,CONFIG["regions"]["meld_right"])
    assert len(groups) == 2 and groups[0]["status"] == "needs_confirmation" and groups[1]["status"] == "confirmed"
    assert items[0]["label"] == items[2]["label"] == "?"


def test_an_empty_best_candidate_cannot_complete_or_crash_a_partition():
    items = [{"label":label,"horizontal":False,"canonical_bbox":[i*40,0,40,60]}
             for i,label in enumerate(("?","6p","7p","8m","8m","8m"))]
    items[0].update(best_label=None,score=.8,margin=.7,reason="pjura_low_confidence")
    assert group_melds(items,CONFIG["regions"]["meld_right"])[0]["status"] == "needs_confirmation"


def test_scorer_does_not_count_an_unknown_fragment_as_a_shifted_wrong_confirmation():
    known = {"type":"pon","labels":["2s"]*3,"target":2,"called_positions":[2],"status":"confirmed"}
    region = {"complete":False,"groups":[{"status":"needs_confirmation"},known]}
    result = score(region,[{k:v for k,v in known.items() if k!="status"}])
    assert not result["correct"] and not result["wrong_confirmation"] and result["groups_correct"] == 1


@pytest.fixture(scope="module")
def bank():
    from app.vision.pjura import PjuraBank
    from app.vision.templates import TemplateBank
    return PjuraBank(TemplateBank(CONFIG["templates"],CONFIG.get("classifier_manifest")),CONFIG["tile_classifier"])


@pytest.mark.gpu
@pytest.mark.parametrize("case",DATA["cases"],ids=lambda c:c["name"])
def test_native_frozen_meld_corpus_has_no_wrong_confirmation_or_regression(case,bank):
    row,observation,_ = evaluate(CONFIG,bank,case)
    assert not row["wrong_confirmation"],row
    if case["name"] in BASE_CORRECT|FIXED:
        assert row["correct"],row
    if not case["expected_groups"]:
        assert row["safe_negative"],row
    assert observation["tile_classifier"]["confidence"] == .9
