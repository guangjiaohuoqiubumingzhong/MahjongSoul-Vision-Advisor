from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.config import VisionError, validate_config
from app.vision.pjura import LABELS, PjuraBank, preprocess
from app.state.events import BASE


class Model:
    manifest_sha256 = "manifest"
    manifest = {"model_sha256": "model"}
    provider = "test"

    def __init__(self, label="3s", score=.99, margin=.9):
        self.label, self.score, self.margin = label, score, margin
        self.seen = []

    def predict_many(self, crops):
        self.seen.extend(crop.copy() for crop in crops)
        return [{"label": self.label, "score": self.score, "margin": self.margin,
                 "alternatives": [{"label": self.label, "score": self.score}]} for crop in crops]


def bank(old_label="9m", *, model=None, **options):
    legacy = SimpleNamespace(classify=lambda *args, **kwargs: {
        "label": old_label, "score": .95, "margin": .2, "face_rotation": 0},
        sha256="templates", training_games={"local-train"})
    return PjuraBank(legacy, {"confidence": .9, "margin": .04, **options}, model=model or Model())


def crop():
    pixels = np.full((40, 30, 3), 230, np.uint8)
    pixels[5:20, 5:15] = [0, 0, 255]
    return pixels


def test_complete_tile_mapping_and_rgb_preprocessing():
    assert set(LABELS.values()) == set(BASE)
    assert LABELS["5b"] == "5s" and LABELS["5n"] == "5m"
    assert LABELS["wd"] == "P" and LABELS["gd"] == "F" and LABELS["rd"] == "C"
    red = np.tile(np.array([0, 0, 255], np.uint8), (10, 12, 1))
    result = preprocess(red)
    assert result.shape == (3, 224, 224) and result.dtype == np.float32
    np.testing.assert_array_equal(result[:, 10, 10], [1, -1, -1])
    with pytest.raises(VisionError):
        preprocess(np.empty((0, 10, 3), np.uint8))


def test_preprocessing_lookup_preserves_upstream_float_rounding_exactly():
    import cv2
    from PIL import Image
    pixels=np.arange(256,dtype=np.uint8).reshape(16,16)
    pixels=np.stack((pixels,np.flipud(pixels),np.fliplr(pixels)),axis=-1)
    image=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))
    rgb=np.asarray(image.resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)
    rgb=(rgb.astype(np.float64)*(1/255)).astype(np.float32)
    expected=np.ascontiguousarray(((rgb-.5)/.5).transpose(2,0,1))
    np.testing.assert_array_equal(preprocess(pixels),expected)


def test_complete_evidence_cache_is_pixel_exact_policy_bound_and_isolated():
    model=Model();wrapped=bank(model=model,rectification='off')
    calls=[];old=wrapped.legacy.classify
    wrapped.legacy.classify=lambda *args,**kwargs:calls.append(1) or old(*args,**kwargs)
    pixels=crop();native=np.repeat(np.repeat(pixels,2,axis=0),2,axis=1)
    first=wrapped.classify_many([pixels],'river',.9,.04,model_crops=[native])[0]
    first['legacy_evidence']['label']='W'
    repeated=wrapped.classify_many([pixels.copy()],'river',.9,.04,model_crops=[native.copy()])[0]
    assert repeated['legacy_evidence']['label']=='9m' and len(calls)==1
    native[0,0,0]^=1
    wrapped.classify_many([pixels],'river',.9,.04,model_crops=[native])
    pixels[0,0,0]^=1
    wrapped.classify_many([pixels],'river',.9,.04,model_crops=[native])
    wrapped.classify_many([pixels],'meld',.9,.04,model_crops=[native])
    wrapped.classify_many([pixels],'meld',.95,.04,model_crops=[native])
    wrapped.classify_many([pixels],'meld',.95,.04,(0,90),model_crops=[native])
    assert len(calls)==6
    assert len(wrapped._classification_cache)==6
    wrapped.classification_cache_size=1
    wrapped.classify_many([pixels],'meld',.95,.04,(0,90),model_crops=[native])
    assert len(wrapped._classification_cache)==1
    wrapped.classify_many([crop()],'river',.9,.04,model_crops=[np.repeat(np.repeat(crop(),2,axis=0),2,axis=1)])
    assert len(calls)==7  # The old evidence was evicted and must be recomputed.


def test_pjura_is_primary_even_when_template_disagrees_and_cache_isolated():
    model = Model()
    wrapped = bank(model=model)
    first = wrapped.classify(crop(), "hand", .9, .04)
    assert first["label"] == "3s" and first["legacy_evidence"]["label"] == "9m"
    first["pjura_evidence"]["label"] = "W"
    second = wrapped.classify(crop(), "hand", .9, .04)
    assert second["pjura_evidence"]["label"] == "3s" and len(model.seen) == 1
    assert wrapped.training_games == {"local-train"}


def blank():
    """A face with no strokes: stroke colour cannot decide red or ordinary."""
    return np.full((40, 30, 3), 230, np.uint8)


def test_white_dragon_identity_does_not_certify_an_arbitrary_template_turn():
    result = bank("P", model=Model("P")).classify(blank(), "meld_right", .9, .04, (0, 90, 270))
    assert result["label"] == "P"
    assert not result["orientation_confirmed"] and result["orientation_source"] == "unconfirmed"


def five_face(red_share):
    """A white face whose strokes are red for ``red_share`` of the ink, else black."""
    pixels = np.full((60, 40, 3), 232, np.uint8)
    rows = int(round(30 * red_share))
    pixels[10:10+rows, 10:30] = [30, 30, 200]
    pixels[10+rows:40, 10:30] = [25, 25, 25]
    return pixels


@pytest.mark.parametrize("label,expected", [("5mr", "5mr"), ("5m", "5m"), ("5pr", "?"), ("?", "?")])
def test_base_five_cannot_silently_drop_red_identity(label, expected):
    result = bank(label, model=Model("5m")).classify(blank(), "hand", .9, .04)
    assert result["label"] == expected
    if expected == "?":
        assert result["reason"] == "pjura_five_colour_unresolved"


def test_trusted_back_never_sent_to_the_34_class_model():
    model = Model()
    assert bank("back", model=model).classify(crop(), "meld", .9, .04)["label"] == "back"
    assert not model.seen


@pytest.mark.parametrize("score,margin,reason", [(.89,.5,"pjura_low_confidence"),(.99,.01,"pjura_ambiguous_classes")])
def test_uncertain_pjura_output_is_not_replaced_with_template(score, margin, reason):
    result = bank(model=Model(score=score, margin=margin)).classify(crop(), "hand", .9, .04)
    assert result["label"] == "?" and result["reason"] == reason
    assert result["pjura_evidence"]["label"] == "3s"


def test_all_crop_rotations_are_evaluated_as_separate_inputs():
    model = Model()
    result = bank(model=model, rectification="off").classify_many([crop()], "river", .9, .04, (0, 90, 270))
    assert len(model.seen) == 3 and len(result) == 1
    assert model.seen[0].shape == (40, 30, 3) and model.seen[1].shape == (30, 40, 3)


def test_both_mode_adds_one_rectified_view_per_rotation():
    model = Model()
    result = bank(model=model).classify_many([crop()], "river", .9, .04, (0, 90, 270))[0]
    assert len(model.seen) == 6
    # Originals keep the old inputs; every rectified view is an upright portrait.
    assert model.seen[0].shape == (40, 30, 3)
    assert all(view.shape[0] > view.shape[1] for view in model.seen[3:])
    assert set(result["pjura_evidence"]["views"]) == {"original", "top_face"}


def test_a_landscape_tile_cannot_use_an_impossible_portrait_hypothesis():
    class PoseModel(Model):
        def predict_many(self,crops):
            self.seen.extend(crops)
            return [{"label":"7s" if p.shape[1]>p.shape[0] else "2s",
                     "score":.99 if p.shape[1]>p.shape[0] else .98,"margin":.9,"alternatives":[]}
                    for p in crops]
    pixels = np.full((35,65,3),232,np.uint8)
    pixels[10:25,15:20] = (40,100,40)
    pixels[10:25,45:50] = (40,100,40)
    model = PoseModel()
    result = bank("?",model=model).classify(pixels,"meld_top",.9,.04,(0,90,270))
    assert result["label"] == "2s" and result["identity_rotation"] in (90,270)
    assert all(p.shape[0] > p.shape[1] for p in model.seen)


def test_identity_confidence_does_not_confirm_sideways_orientation():
    result = bank("?", model=Model("6m")).classify(crop(), "meld", .9, .04, (0, 90, 270))
    assert result["label"] == "6m" and not result["orientation_confirmed"]
    result = bank("6m", model=Model("6m")).classify(crop(), "meld", .9, .04, (0, 90, 270))
    assert result["orientation_confirmed"] and result["orientation_source"] == "legacy_template"


def test_pjura_config_rejects_invalid_threshold_and_provider():
    config = {"version":"unit", "client_size":[128,128], "calibrated":True,
              "seat_mapping":{"self":0,"right":1,"top":2,"left":3},
              "regions":{"hand":{"kind":"hand","rect":[0,0,1,1]}},
              "tile_classifier":{"name":"pjura","manifest":"model.json"}}
    validate_config(config)
    for key, value in (("confidence", float("nan")), ("margin", -1), ("provider", "unknown"), ("batch_size", 0),
                       ("rectification", "always"), ("rectification_skip_roles", "hand"),
                       ("dominant_candidate", {"minimum_confidence": .49}),
                       ("dominant_candidate", {"minimum_ratio": 1}), ("dominant_candidate", False)):
        invalid = deepcopy(config)
        invalid["tile_classifier"][key] = value
        with pytest.raises(VisionError):
            validate_config(invalid)


@pytest.mark.parametrize("share,expected", [(1.0, "5mr"), (.55, "5m"), (.8, "?")])
def test_stroke_colour_decides_a_five_the_templates_could_not(share, expected):
    result = bank("?", model=Model("5m")).classify(five_face(share), "river_left", .9, .04)
    assert result["label"] == expected
    if expected == "?":
        assert result["reason"] == "pjura_five_colour_unresolved"
    else:
        assert result["pjura_evidence"]["colour_evidence"]["source"] == "stroke_red_share"


def test_stroke_colour_and_template_disagreeing_is_unknown():
    result = bank("5m", model=Model("5m")).classify(five_face(1.0), "river_left", .9, .04)
    assert result["label"] == "?" and result["reason"] == "pjura_five_colour_conflict"


def test_ordinary_five_suits_use_their_own_red_share_limit():
    # A 5p/5s with half red strokes is not a plausible ordinary five.
    assert bank("?", model=Model("5p")).classify(five_face(.55), "river_left", .9, .04)["label"] == "?"
    assert bank("?", model=Model("5s")).classify(five_face(.2), "river_left", .9, .04)["label"] == "5s"


def test_tinted_rejected_candidates_are_flagged_as_table_art():
    feather = np.full((40, 30, 3), (235, 205, 170), np.uint8)   # light blue art
    result = bank("?", model=Model("P", score=.3, margin=.2)).classify(feather, "river_left", .9, .04)
    assert result["label"] == "?" and result["background_candidate"]
    assert not result["pjura_evidence"]["face_tint"]["neutral"]
    white = bank("?", model=Model("P", score=.3, margin=.2)).classify(blank(), "river_left", .9, .04)
    assert white["pjura_evidence"]["face_tint"]["neutral"]


def test_accepted_tiles_are_never_flagged_as_background():
    feather = np.full((40, 30, 3), (235, 205, 170), np.uint8)
    assert not bank(model=Model("3s")).classify(feather, "river_left", .9, .04)["background_candidate"]
