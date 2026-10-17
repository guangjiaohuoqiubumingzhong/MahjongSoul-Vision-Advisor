from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.acceptance import candidate_acceptance, dominant_candidate_config
from app.vision.pjura import PjuraBank, PjuraModel


DOMINANT = {"minimum_confidence": .5, "minimum_ratio": 3.}


@pytest.mark.parametrize("first,second,expected", [
    (.5, .01, None), (.4999, .01, None), (.5001, .01, "dominant_candidate"),
    (.55, .15, "dominant_candidate"), (.55, .3, None),
    (.5625, .1875, "dominant_candidate"), (.5625, .1876, None),
    (.88847, .00823, "dominant_candidate"), (.9, .08, "confidence_threshold"),
    (.96, .03, "confidence_threshold"),
])
def test_strict_majority_and_runner_up_ratio(first, second, expected):
    assert candidate_acceptance({"score": first, "margin": first-second}, .9, .04, DOMINANT) == expected


def test_disabled_policy_and_margin_check_remain_effective():
    prediction = {"score": .7, "margin": .6}
    assert candidate_acceptance(prediction, .9, .04) is None
    assert candidate_acceptance(prediction, .9, .65, DOMINANT) is None
    assert candidate_acceptance({"score": .99, "margin": .01}, .9, .04, DOMINANT) is None


@pytest.mark.parametrize("value", [False, [], {"minimum_confidence": True},
    {"minimum_confidence": .49}, {"minimum_confidence": 1}, {"minimum_confidence": float("nan")},
    {"minimum_ratio": 1}, {"minimum_ratio": float("inf")}, {"minimum_ratio": "3"}])
def test_invalid_dominance_configuration_is_rejected(value):
    with pytest.raises(ValueError): dominant_candidate_config(value)


def test_defaults_are_independent_and_optional():
    assert dominant_candidate_config(None) is None
    assert dominant_candidate_config({}) == DOMINANT
    assert dominant_candidate_config({}) is not dominant_candidate_config({})


def bank(predictions, legacy_label="3s", **options):
    legacy = SimpleNamespace(classify=lambda *args, **kwargs: {
        "label": legacy_label, "score": .95, "margin": .2, "face_rotation": 0})
    model = SimpleNamespace(manifest_sha256="manifest", manifest={"model_sha256": "model"},
        provider="test", predict_many=lambda crops: [dict(predictions[i % len(predictions)],
            alternatives=[]) for i in range(len(crops))])
    return PjuraBank(legacy, {"confidence": .9, "margin": .04,
        "dominant_candidate": DOMINANT, **options}, model=model)


def face():
    pixels = np.full((40, 30, 3), 230, np.uint8)
    pixels[5:20, 5:15] = [0, 0, 255]
    return pixels


def test_dominant_reading_has_auditable_evidence_and_policy_bound_cache():
    wrapped = bank([{"label": "3s", "score": .7, "margin": .6}], rectification="off")
    result = wrapped.classify(face(), "river", .9, .04)
    assert result["label"] == "3s"
    assert result["acceptance_policy"] == "pjura_dominant_candidate_with_five_colour_evidence"
    evidence = result["pjura_evidence"]["class_acceptance"]
    assert evidence["rule"] == "dominant_candidate"
    assert evidence["second_score"] == pytest.approx(.1)
    assert evidence["top1_to_second_ratio"] == pytest.approx(7)
    wrapped.dominant_candidate = {**DOMINANT, "minimum_ratio": 8.}
    changed = wrapped.classify(face(), "river", .9, .04)
    assert changed["label"] == "?" and changed["reason"] == "pjura_low_confidence"


def test_existing_high_confidence_cross_view_conflicts_remain_unknown():
    wrapped = bank([{"label": "3s", "score": .99, "margin": .98},
                    {"label": "4s", "score": .95, "margin": .94}])
    result = wrapped.classify(face(), "river", .9, .04)
    assert len(result["pjura_evidence"]["views"]) == 2
    assert result["label"] == "?" and result["reason"] == "pjura_rectification_disagrees"


def test_weaker_fallback_view_does_not_veto_existing_high_confidence_path():
    wrapped = bank([{"label": "3s", "score": .99, "margin": .98},
                    {"label": "4s", "score": .65, "margin": .55}])
    result = wrapped.classify(face(), "river", .9, .04)
    assert result["label"] == "3s" and result["reason"] is None
    assert result["pjura_evidence"]["class_acceptance"]["rule"] == "confidence_threshold"


def test_dominance_compares_classes_within_selected_view():
    wrapped = bank([{"label": "7m", "score": .65, "margin": .59},
                    {"label": "6m", "score": .84, "margin": .81}], legacy_label="6m")
    result = wrapped.classify(face(), "river", .9, .04)
    assert result["label"] == "6m" and result["reason"] is None
    assert result["pjura_evidence"]["class_acceptance"]["rule"] == "dominant_candidate"


def test_dominance_does_not_resolve_an_unknown_red_five_or_white_dragon_rotation():
    blank = np.full((40, 30, 3), 230, np.uint8)
    result = bank([{"label": "5m", "score": .7, "margin": .6}], legacy_label="?").classify(blank, "hand", .9, .04)
    assert result["label"] == "?" and result["reason"] == "pjura_five_colour_unresolved"
    result = bank([{"label": "P", "score": .7, "margin": .6}], legacy_label="P").classify(blank, "meld_right", .9, .04)
    assert result["label"] == "P" and not result["orientation_confirmed"]


def test_fp16_rechecks_both_new_acceptance_boundaries_in_original_precision():
    model = PjuraModel.__new__(PjuraModel)
    model._reference_manifest = "source"
    model.confidence, model.margin = .9, .04
    model.dominant_candidate = DOMINANT
    model.manifest = {"optimization": {"fp32_guard_band": .005}}
    model.precision_rechecks = 0
    calls = []
    def exact(crops):
        calls.extend(crops)
        return [{"label": "3s", "score": .4999, "margin": .4},
                {"label": "4s", "score": .6, "margin": .3999}]
    model._reference_model = SimpleNamespace(predict_many=exact)
    crops = [np.full((3, 3, 3), i, np.uint8) for i in range(3)]
    result = model._precision_guard(crops, [{"label": "3s", "score": .5001, "margin": .4},
        {"label": "4s", "score": .6, "margin": .4001}, {"label": "5s", "score": .75, "margin": .6}])
    assert [int(c[0, 0, 0]) for c in calls] == [0, 1]
    assert model.precision_rechecks == 2
    assert all(candidate_acceptance(p, .9, .04, DOMINANT) is None for p in result[:2])
    assert candidate_acceptance(result[2], .9, .04, DOMINANT) == "dominant_candidate"
