from copy import deepcopy
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import tkinter as tk
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.capture.frames import Frame, config_digest, load_image, save_image
from app.capture.window import CaptureError
from app.ui.workbench import Workbench
import app.ui.workbench as workbench_module
import app.vision.svm as svm_module
from app.vision.config import VisionError, pixel_rect, validate_config, write_json
from app.vision.detector import global_box, river_slot, tile_boxes, tile_candidates
from app.vision.hand import hand_layout
from app.vision.faces import FaceGallery, batch_similarity, tile_face, unrotate_quad
from app.vision.evaluate import evaluate, match_boxes
from app.vision.recognizer import StaticRecognizer, group_melds
from app.vision.templates import TemplateBank, build_templates, canonical, similarity
from app.vision.text import TextReader, parse_text
from app.vision.svm import SvmVerifier, FEATURE_PARAMETERS as SVM_FEATURE_PARAMETERS, POLICY as SVM_POLICY
from app.state.events import TILES


def config():
    return {"version":"unit", "client_size":[256,128], "calibrated":True,
            "seat_mapping":{"self":0,"right":1,"top":2,"left":3},
            "thresholds":{"tile":.9,"margin":.04,"ocr":.9,"button":.94},
            "regions":{"hand":{"kind":"hand","player":0,"rect":[0,.5,.5,.5],"required":True,
                               "slots":[[0,0,40,60]],"rotation":0}}}


def make_bank(tmp_path):
    pixels = np.zeros((128,256,3),dtype=np.uint8)
    pixels[64:124,:40] = 240
    pixels[78:108,12:25] = [30,30,200]
    pixels[10:50,150:180] = 128  # concealed-hand stand-in: deliberately outside ROIs
    image = tmp_path/"原图.png"
    save_image(image,pixels)
    annotation = {"image":str(image),"game_id":"constructed-unit-test","split":"train","reviewed":True,
                  "targets":[{"id":"a","kind":"tile","label":"5mr","bbox":[0,64,40,60],"region":"hand","roles":["hand"],"rotation":0}]}
    path = tmp_path/"labels.json"
    write_json(path,annotation)
    bank_path = tmp_path/"bank/manifest.json"
    build_templates([path],bank_path)
    return pixels, image, path, annotation, bank_path


def test_client_relative_geometry_and_invalid_regions():
    assert pixel_rect([.1,.2,.3,.4],100,200) == (10,40,30,80)
    with pytest.raises(VisionError,match="越出"):
        pixel_rect([.9,0,.2,.1],100,200)
    invalid = config()
    invalid["seat_mapping"]["top"] = 0
    with pytest.raises(VisionError,match="座位"):
        validate_config(invalid)


@pytest.mark.parametrize("degrees,expected", [(0,[13,24,7,11]),(90,[95,23,11,7]),(180,[100,65,7,11]),(270,[14,70,11,7])])
def test_inverse_rotation_preserves_original_coordinates(degrees,expected):
    assert global_box([3,4,7,11],(10,20),(100,60),degrees) == expected


def test_frame_evidence_unicode_roundtrip_and_no_overwrite(tmp_path):
    pixels = np.zeros((32,48,3),dtype=np.uint8)
    pixels[5:15,7:20] = [10,20,250]
    frame = Frame(pixels,{"frame_id":1,"source":"constructed_unit"})
    path = frame.save(tmp_path/"中文")
    assert np.array_equal(load_image(path),pixels)
    with pytest.raises(CaptureError,match="不能覆盖"):
        frame.save(tmp_path/"中文")


def test_frame_keeps_the_actual_configuration_and_rejects_mutated_snapshot(tmp_path):
    c = config()
    metadata = {"frame_id":1,"region_config_sha256":config_digest(c)}
    frame = Frame(np.zeros((128,256,3),dtype=np.uint8),metadata,c)
    frame.save(tmp_path)
    snapshot = Path(frame.metadata["region_config_snapshot"])
    assert json.loads(snapshot.read_text(encoding="utf-8")) == c
    changed = deepcopy(c)
    changed["version"] = "different"
    with pytest.raises(CaptureError,match="配置发生变化"):
        Frame(frame.pixels,{"frame_id":2,**{"region_config_sha256":config_digest(c)}},changed).save(tmp_path)
    snapshot.write_text(json.dumps(changed),encoding="utf-8")
    with pytest.raises(CaptureError,match="配置副本"):
        Frame(frame.pixels,{"frame_id":2,"region_config_sha256":config_digest(c)},c).save(tmp_path)


def test_template_source_split_and_hash_are_enforced(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    annotation["split"] = "test"
    write_json(path,annotation)
    with pytest.raises(VisionError,match="只能"):
        build_templates([path],tmp_path/"bad/manifest.json")
    manifest = json.loads(bank_path.read_text())
    template = bank_path.parent/manifest["templates"][0]["file"]
    template.write_bytes(b"corrupt")
    with pytest.raises(VisionError,match="校验"):
        TemplateBank(bank_path)


def test_template_corrections_preserve_the_old_manifest_and_bitmap(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    original = bank_path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    snapshot = bank_path.with_name(f"manifest-{digest}.json")
    entry = json.loads(original)["templates"][0]
    old_bitmap = (bank_path.parent/entry["file"]).read_bytes()
    annotation["targets"][0]["bbox"][2] = 39
    write_json(path,annotation)
    build_templates([path],bank_path)
    assert snapshot.read_bytes() == original
    assert (bank_path.parent/entry["file"]).read_bytes() == old_bitmap
    assert TemplateBank(snapshot).sha256 == digest
    assert TemplateBank(bank_path).sha256 != digest


def test_unknown_feature_version_and_changed_parameters_stop_bank_loading(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    manifest = json.loads(bank_path.read_text())
    changed = deepcopy(manifest)
    changed["feature_version"] = "unsupported-future-version"
    write_json(bank_path,changed)
    with pytest.raises(VisionError,match="版本不兼容"):
        TemplateBank(bank_path)
    manifest["face_parameters"]["blank_max_ink_fraction"] = .9
    write_json(bank_path,manifest)
    with pytest.raises(VisionError,match="参数不兼容"):
        TemplateBank(bank_path)


def test_ambiguous_tiles_and_absent_role_return_unknown(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    manifest = json.loads(bank_path.read_text())
    second = {**manifest["templates"][0],"label":"5m","id":"other"}
    manifest["templates"].append(second)
    write_json(bank_path,manifest)
    bank = TemplateBank(bank_path)
    result = bank.classify(pixels[64:124,:40],"hand",.9,.04)
    assert result["label"] == "?" and result["margin"] == 0
    assert bank.classify(pixels[64:124,:40],"river_self",.9,.04)["reason"] == "missing_role_templates"


def test_cached_body_scores_match_scalar_reference_and_keep_roles_separate(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    manifest = json.loads(bank_path.read_text())
    second = np.full((60,40,3),230,np.uint8)
    second[8:45,10:20] = 0
    file = bank_path.parent/"other.png"
    save_image(file,second)
    manifest["templates"].append({"id":"other","file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                                  "label":"1m","kind":"tile","roles":["hand"]})
    write_json(bank_path,manifest)
    bank = TemplateBank(bank_path)
    crop = pixels[64:124,:40]
    query = canonical(crop)
    expected = sorted([(entry["label"],similarity(query,value)) for entry,value in bank.entries],
                      key=lambda item:item[1],reverse=True)
    result = bank._classify_body(crop,"hand",.9,.04,(0,),kinds=("tile",))
    assert result["label"] == expected[0][0]
    assert result["score"] == pytest.approx(expected[0][1],abs=2e-5)
    assert result["margin"] == pytest.approx(expected[0][1]-expected[1][1],abs=2e-5)
    gallery = bank.body_galleries[("hand",("tile",))][1]
    bank._classify_body(crop,"hand",.9,.04,(0,),kinds=("tile",))
    assert bank.body_galleries[("hand",("tile",))][1] is gallery
    assert bank._classify_body(crop,"draw",.9,.04,(0,),kinds=("tile",))["reason"] == "missing_role_templates"
    assert bank._classify_body(crop,"hand",.9,.04,(0,),kinds=("text",))["reason"] == "missing_role_templates"


def test_body_and_face_disagreement_never_accepts_a_tile(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    bank = TemplateBank(bank_path)
    bank._classify_body = lambda *a,**k:{"label":"5mr","score":1,"margin":1,"reason":None}
    bank._classify_face = lambda *a:{"label":"1m","score":1,"blank_face":False}
    result = bank.classify(pixels[64:124,:40],"hand",.9,.04)
    assert result["label"] == "?" and result["reason"] == "conflicting_tile_features"


def test_opponent_hidden_pixels_never_change_recognized_inputs(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    # Two tile labels are required for a meaningful ambiguity margin.
    second = np.full((60,40,3),240,dtype=np.uint8)
    second[10:20,5:35] = 0
    file = bank_path.parent/"other.png"
    save_image(file,second)
    manifest = json.loads(bank_path.read_text())
    manifest["templates"].append({"id":"other","file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                                  "label":"1m","kind":"tile","roles":["hand"]})
    write_json(bank_path,manifest)
    recognizer = StaticRecognizer(config(),TemplateBank(bank_path))
    original = recognizer.recognize(pixels)
    changed = pixels.copy()
    changed[:64,128:] = np.random.default_rng(123).integers(0,256,(64,128,3),dtype=np.uint8)
    modified = recognizer.recognize(changed)
    assert original["regions"] == modified["regions"]
    assert original["regions"]["hand"]["items"][0]["label"] == "5mr"
    assert original["image_sha256"] != modified["image_sha256"]  # provenance still identifies each raw frame
    with pytest.raises(VisionError,match="尺寸"):
        recognizer.recognize(pixels[:100])


def test_river_slots_keep_gaps_and_do_not_reindex():
    pixels = np.zeros((60,140,3),dtype=np.uint8)
    pixels[:50,:30] = 240
    pixels[:50,90:120] = 240
    boxes = tile_boxes(pixels,{"slots":[[0,0,30,50],[45,0,30,50],[90,0,30,50]]})
    assert boxes == [[0,0,30,50],[90,0,30,50]]


def test_draw_is_separated_after_an_open_meld_and_regular_slots_stay_fixed():
    pixels = np.zeros((64,650,3), dtype=np.uint8)
    slots = [[5+45*i,2,40,60] for i in range(13)]
    region = {"slots":slots,"body_presence_bands":[[.08,.15],[.9,.98]],
              "hand_layout":{"draw_region":"draw","draw_gap":15}}
    for x,y,w,h in slots[:10]:
        pixels[y:y+h,x:x+w] = 230
    pixels[2:62,465:505] = 230  # ten regular tiles end at x=450; draw follows a gap
    layout = hand_layout(pixels,region)
    assert layout["hand"] == slots[:10]
    assert layout["draw"] == [[465,2,40,60]]
    assert layout["slot_indices"] == list(range(10)) and not layout["reasons"]
    pixels[2:62,450:465] = 230  # blocked gap: never silently accept an empty draw
    assert hand_layout(pixels,region)["reasons"] == ["concealed_count_or_draw_position_unconfirmed"]
    pixels[2:62,450:465] = 0
    pixels[2:62,465:505] = 0
    assert hand_layout(pixels,region)["draw"] == []
    pixels[2:62,slots[5][0]:slots[5][0]+40] = 0
    assert hand_layout(pixels,region)["reasons"] == ["noncontiguous_concealed_hand"]


def test_small_meld_in_hand_area_is_not_a_full_size_concealed_tile():
    pixels = np.zeros((64,80,3),dtype=np.uint8)
    pixels[22:62,2:42] = 230  # more than 45% bright, but its upper body band is missing
    region = {"slots":[[2,2,40,60]],"body_presence_bands":[[.08,.15],[.9,.98]]}
    assert tile_boxes(pixels,region) == []


def test_dim_tile_separators_and_sparse_bright_pixels_do_not_create_tiles():
    pixels = np.zeros((60,100,3),dtype=np.uint8)
    pixels[4:54,4:34] = 210
    pixels[4:54,36:66] = 210  # two-pixel separator must survive morphology
    assert tile_boxes(pixels,{"min_size":[20,30],"max_size":[40,60]}) == [[4,4,30,50],[36,4,30,50]]
    pixels[:] = 0
    pixels[10:13,10:13] = 250
    assert tile_boxes(pixels,{"slots":[[0,0,40,60]]}) == []


def test_sloping_river_grid_keeps_missing_slots_and_rejects_singular_config():
    grid = {"origin":[15,30],"basis":[[30,3],[0,53]],"anchor":"center"}
    assert river_slot([30,13,30,40],grid) == ([0,1],0)
    assert river_slot([90,19,30,40],grid) == ([0,3],0)  # missing column 2 is preserved
    c = config()
    c["regions"]["hand"]["grid"] = {**grid,"basis":[[1,2],[2,4]]}
    with pytest.raises(VisionError,match="基向量"):
        validate_config(c)


def test_background_tint_normalization_retains_red_glyph_identity():
    normal = np.full((60,40,3),230,dtype=np.uint8)
    normal[15:45,15:25] = [20,20,200]
    pink = (normal.astype(float)*np.array([.75,.75,1.])).astype(np.uint8)
    black = normal.copy()
    black[15:45,15:25] = 20
    assert similarity(canonical(normal),canonical(pink)) > .99
    assert similarity(canonical(normal),canonical(black)) < .9


def test_tile_face_removes_a_grey_base_and_preserves_glyph_colour():
    front = np.full((60,40,3),230,np.uint8)
    front[15:45,15:25] = [20,20,200]
    image = np.full((85,65,3),100,np.uint8)
    image[10:70,15:55] = front
    image[70:80,15:55] = 170  # grey base must not become part of the glyph
    face = tile_face(image)
    assert face is not None and not face.blank
    assert face.quad[:,1].max() < 71
    assert np.any(face.feature[:,:,2]-face.feature[:,:,1] > .5)


def test_faint_marks_are_not_blank_even_when_colour_quantisation_erases_them():
    pixels = np.full((60,40,3),240,np.uint8)
    assert tile_face(pixels).blank
    pixels[20:40,15:25] = [240,225,225]  # faint blue glyph, not a white dragon
    face = tile_face(pixels)
    assert face is not None and not face.blank


def test_specular_stripe_fallback_requires_a_large_face_and_keeps_red():
    from app.vision.faces import HIGHLIGHT_FACE_PARAMETERS
    yy,xx = np.indices((60,40))
    pixels = np.full((60,40,3),220,np.uint8)
    pixels[np.abs(xx+yy-50)<10] = 255
    pixels[35:50,15:25] = [20,20,190]
    assert tile_face(pixels) is None  # frozen v3 recipe remains reproducible
    face = tile_face(pixels,HIGHLIGHT_FACE_PARAMETERS)
    assert face is not None and face.area_fraction > .9 and not face.blank
    assert np.max(face.feature[:,:,2]-face.feature[:,:,1]) > .5
    fragment = np.full((100,100,3),30,np.uint8)
    fragment[10:90,10:45] = 220
    fragment[10:90,10:35] = 255
    assert tile_face(fragment) is None
    assert tile_face(fragment,HIGHLIGHT_FACE_PARAMETERS) is None


def test_highlight_fallback_does_not_change_a_valid_original_face():
    from app.vision.faces import HIGHLIGHT_FACE_PARAMETERS
    pixels = np.full((60,40,3),230,np.uint8)
    pixels[15:45,15:25] = [20,20,190]
    original = tile_face(pixels)
    upgraded = tile_face(pixels,HIGHLIGHT_FACE_PARAMETERS)
    assert np.array_equal(original.feature,upgraded.feature)
    assert np.array_equal(original.quad,upgraded.quad)
    assert original.blank == upgraded.blank


def test_face_recipe_is_frozen_by_manifest_and_old_bank_is_preserved(tmp_path):
    from app.vision.faces import FACE_PARAMETERS,HIGHLIGHT_FACE_PARAMETERS
    from app.vision.templates import FEATURE_VERSION,HIGHLIGHT_FEATURE_VERSION
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    payload = bank_path.read_bytes()
    upgraded_path = tmp_path/"highlight"/"manifest.json"
    build_templates([path],upgraded_path,feature_version=HIGHLIGHT_FEATURE_VERSION)
    original = TemplateBank(bank_path)
    upgraded = TemplateBank(upgraded_path)
    assert original.feature_version == FEATURE_VERSION and original.face_parameters == FACE_PARAMETERS
    assert upgraded.feature_version == HIGHLIGHT_FEATURE_VERSION and upgraded.face_parameters == HIGHLIGHT_FACE_PARAMETERS
    assert bank_path.read_bytes() == payload
    value = json.loads(upgraded_path.read_text())
    value["face_parameters"]["fallback_white_min_channel"] = .1
    write_json(upgraded_path,value)
    with pytest.raises(VisionError,match="参数不兼容"):
        TemplateBank(upgraded_path)


def test_precomputed_face_scores_keep_the_original_correlation_and_mse():
    rng = np.random.default_rng(42)
    features = rng.random((5,54,34,3),dtype=np.float32)
    features[0] = 1  # constant white dragon also exercises zero variance
    gallery = FaceGallery(features)
    for query in [features[0],features[2],rng.random((54,34,3),dtype=np.float32)]:
        expected = np.array([similarity(f,query) for f in features])
        assert np.allclose(gallery.scores(query),expected,atol=2e-5)
        assert np.allclose(batch_similarity(features,query),expected,atol=2e-5)


def test_face_points_return_to_the_original_crop_without_box_corner_offsets():
    quad = np.array([[1,2],[7,2],[7,12],[1,12]],np.float32)
    assert unrotate_quad(quad,20,10,90) == [[17,1],[17,7],[7,7],[7,1]]


def test_raw_overlay_capture_cannot_become_a_trusted_scene(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    result = StaticRecognizer(config(),TemplateBank(bank_path)).recognize(
        pixels,{"capture_quality":"requires_manual_overlay_review"})
    assert result["status"] == "needs_confirmation"
    assert {"reason":"capture_requires_manual_review","capture_quality":"requires_manual_overlay_review"} in result["issues"]


def test_meld_source_and_kan_type_require_evidence():
    tiles = [{"label":"5mr","horizontal":True,"canonical_bbox":[0,0,20,30]},
             {"label":"5m","horizontal":False,"canonical_bbox":[20,0,20,30]},
             {"label":"5m","horizontal":False,"canonical_bbox":[40,0,20,30]}]
    assert group_melds(tiles,{})[0]["status"] == "needs_confirmation"
    assert group_melds(tiles,{"source_by_called_position":{"0":3}})[0]["target"] == 3
    tiles.append({"label":"5m","horizontal":False,"canonical_bbox":[20,-10,20,30]})
    group = group_melds(tiles,{"source_by_called_position":{"0":3}})[0]
    assert group["type"] is None and group["target"] is None and group["status"] == "needs_confirmation"


def test_pon_mapping_requires_a_known_triplet_and_one_called_tile():
    items = [{"label":"2m","horizontal":i==1,"canonical_bbox":[i*25,0,25,40]} for i in range(3)]
    region = {"chi_source_by_called_position":{"0":3},"pon_source_by_called_position":{"0":3,"1":2,"2":1}}
    assert group_melds(items,region)[0]["target"] == 2
    items[0]["label"] = "?"
    assert group_melds(items,region)[0]["target"] is None
    items[0]["label"] = "2m"
    items[0]["horizontal"] = True
    assert group_melds(items,region)[0]["target"] is None


def test_oversized_foreground_cannot_be_reported_as_an_empty_meld_area():
    c = config()
    c["regions"] = {"meld":{"kind":"meld","player":0,"rect":[0,0,1,1],"required":True,
                            "min_size":[15,20],"max_size":[80,90]}}
    pixels = np.zeros((128,256,3),np.uint8)
    pixels[3:120,3:100] = 240
    found = tile_candidates(pixels,c["regions"]["meld"])
    assert not found["boxes"] and found["unresolved"][0]["reason"] == "component_outside_tile_size"
    observation = StaticRecognizer(c,SimpleNamespace(classify=lambda *a:{"label":"?","score":0,"margin":0})).recognize(pixels)
    assert observation["regions"]["meld"]["items"] == []
    assert not observation["regions"]["meld"]["complete"]
    assert observation["status"] == "needs_confirmation"
    assert observation["regions"]["meld"]["unresolved_components"][0]["bbox"] == [3,3,97,117]


def test_wide_foreground_requires_valid_calibrated_separators():
    pixels = np.zeros((100,200,3),np.uint8)
    pixels[10:70,10:130] = 240
    region = {"min_size":[15,20],"max_size":[80,90]}
    assert tile_candidates(pixels,region)["unresolved"][0]["reason"] == "unsplit_wide_component"
    region["tile_width"] = 40
    detection = tile_candidates(pixels,region)
    assert detection["boxes"] == [[10,10,40,60],[50,10,40,60],[90,10,40,60]]
    assert detection["unresolved"] == []


def test_overlapping_self_hand_body_is_not_an_unresolved_meld_foreground(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    c = config()
    c["regions"]["meld"] = {"kind":"meld","player":0,"rect":[0,.5,.5,.5],"required":True,
                            "min_size":[15,20],"max_size":[80,50]}
    original = pixels.copy()
    result = StaticRecognizer(c,TemplateBank(bank_path)).recognize(pixels)
    assert result["regions"]["meld"]["items"] == [] and result["regions"]["meld"]["complete"]
    assert not result["regions"]["meld"].get("unresolved_components")
    assert np.array_equal(pixels,original)


def test_touching_melds_require_a_unique_identity_partition():
    labels = ["6s"]*4+["8m"]*3+["1s"]*3
    items = [{"label":label,"horizontal":False,"canonical_bbox":[i*25,0,25,40]}
             for i,label in enumerate(labels)]
    groups = group_melds(items,{})
    assert [g["labels"] for g in groups] == [["6s"]*4,["8m"]*3,["1s"]*3]
    assert [g["type"] for g in groups] == [None,"pon","pon"]
    assert all(g["status"] == "needs_confirmation" for g in groups)
    items[4]["label"] = "?"
    assert len(group_melds(items,{})) == 1


def test_chi_source_mapping_does_not_confirm_an_unverified_pon_source():
    items = [{"label":label,"horizontal":i==0,"canonical_bbox":[i*25,0,25,40]}
             for i,label in enumerate(["6p","5pr","7p"])]
    region = {"chi_source_by_called_position":{"0":3}}
    assert group_melds(items,region)[0]["target"] == 3
    for item in items:
        item["label"] = "6p"
    assert group_melds(items,region)[0]["target"] is None


def test_meld_evaluation_matches_instances_independently_of_iou_pair_order(tmp_path):
    image = tmp_path/"meld.png"
    save_image(image,np.zeros((128,256,3),np.uint8))
    items = [{"label":"1m","bbox":[i*30,10,25,40]} for i in range(3)]
    targets = [{**item,"id":str(i),"kind":"tile","region":"meld","group_id":"a",
                "meld_type":"pon","target":3} for i,item in enumerate(items)]
    path = tmp_path/"meld.annotation.json"
    write_json(path,{"image":str(image),"reviewed":True,"game_id":"independent-meld-unit",
                     "split":"test","frame_quality":"stable","complete_regions":["meld"],"targets":targets})
    c = config()
    c["regions"] = {"meld":{"kind":"meld","rect":[0,0,1,1],"required":True}}
    observation = {"regions":{"meld":{"kind":"meld","items":items,
                    "groups":[{"items":items,"type":"pon","target":3}]}},"status":"recognized","recognition_ms":0}
    recognizer = SimpleNamespace(config=c,bank=SimpleNamespace(manifest={"source_games":[]}),
                                 recognize=lambda *a:observation)
    report = evaluate(recognizer,[path])
    assert report["melds"] == {"total":1,"type_correct":1,"source_correct":1}


@pytest.mark.parametrize("text,field,expected", [("余69","remaining",69),("東2局","round",{"bakaze":"E","kyoku":2}),
                                               ("南四局","round",{"bakaze":"S","kyoku":4}),("-1000","score",-1000),
                                               ("99","remaining",None),("25001","score",None),("X0","honba",None),
                                               ("東|1局","round",None),("","kyotaku",None)])
def test_ocr_values_are_not_invented(text,field,expected):
    assert parse_text(text,field) == expected


def test_acceptance_rejects_template_game_leakage_and_empty_evidence(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    recognizer = StaticRecognizer(config(),TemplateBank(bank_path))
    report = evaluate(recognizer,[path])
    assert not report["gate_passed"] and report["whole_scene_accuracy"] is None
    assert report["stable_test_frames"] == 0
    annotation["split"] = "test"
    write_json(path,annotation)
    with pytest.raises(VisionError,match="模板来源"):
        evaluate(recognizer,[path])


def test_acceptance_rejects_invalid_ground_truth_tile(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    annotation["targets"][0]["label"] = "7sr"
    write_json(path,annotation)
    with pytest.raises(VisionError,match="标注牌种"):
        evaluate(StaticRecognizer(config(),TemplateBank(bank_path)),[path])


def test_visual_variants_do_not_increase_distinct_scene_or_tile_count(tmp_path):
    pixels,image,path,annotation,bank_path=make_bank(tmp_path)
    variant_image=tmp_path/"variant.png"
    pixels[0,0]=1
    save_image(variant_image,pixels)
    variant={**annotation,"image":str(variant_image),"count_as_distinct_scene":False,"duplicate_of":str(path)}
    variant_path=tmp_path/"variant.annotation.json"
    write_json(variant_path,variant)
    report=evaluate(StaticRecognizer(config(),TemplateBank(bank_path)),[path,variant_path])
    assert report["dataset_images"]==2 and report["dataset_known_tiles"]==2
    assert report["dataset_distinct_scenes"]==1 and report["dataset_distinct_scene_known_tiles"]==1
    assert report["dataset_visual_variants"]==1
    variant["game_id"]="unrelated-game"
    write_json(variant_path,variant)
    with pytest.raises(VisionError,match="重复局面"):
        evaluate(StaticRecognizer(config(),TemplateBank(bank_path)),[path,variant_path])


def test_unknown_template_proposal_does_not_confirm_sideways_orientation(tmp_path,monkeypatch):
    pixels,image,path,annotation,bank_path=make_bank(tmp_path)
    bank=TemplateBank(bank_path)
    monkeypatch.setattr(bank,'classify',lambda *args,**kwargs:{'label':'?','best_label':'1p','face_rotation':90,
                                                            'score':.5,'reason':'low_score_or_ambiguous'})
    observation=StaticRecognizer(config(),bank).recognize(pixels)
    item=observation['regions']['hand']['items'][0]
    assert item['label']=='?' and item['horizontal'] is None
    assert item['orientation_status']=='unconfirmed'
    assert observation['status']=='needs_confirmation'


@pytest.mark.parametrize("evidence,paused",[(False,0),(True,1)])
def test_uncertain_frame_needs_specific_region_evidence_not_only_uncalibrated_status(tmp_path,evidence,paused):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    annotation.update(split="test",game_id="independent-occlusion-unit",frame_quality="occluded",
                      targets=[],complete_regions=[],expected_pause_regions=["hand"])
    write_json(path,annotation)
    observation = {"status":"needs_confirmation","recognition_ms":0,
                   "regions":{"hand":{"kind":"hand","items":[],"complete":False}},
                   "issues":[{"reason":"uncalibrated_regions"}]}
    if evidence:
        observation["issues"].append({"region":"hand","reason":"incomplete_region"})
    recognizer = SimpleNamespace(config=config(),bank=SimpleNamespace(manifest={"source_games":[]}),
                                 recognize=lambda *a:observation)
    report = evaluate(recognizer,[path])
    assert report["uncertain_frames"] == {"paused":paused,"total":1}
    assert report["frames"][0]["pause_evidence_verified"] is evidence
    assert report["stable_test_frames"] == 0 and report["dataset_known_tiles"] == 0


def test_text_templates_validate_values_and_require_alternative_labels(tmp_path):
    pixels = np.zeros((128,256,3),dtype=np.uint8)
    cv2.putText(pixels,"0",(10,45),cv2.FONT_HERSHEY_SIMPLEX,1,(230,230,230),2)
    cv2.putText(pixels,"1",(65,45),cv2.FONT_HERSHEY_SIMPLEX,1,(230,230,230),2)
    image = tmp_path/"numbers.png"
    save_image(image,pixels)
    annotation = {"image":str(image),"game_id":"numbers-unit","split":"train","reviewed":True,
                  "targets":[{"kind":"text","label":str(i),"value":i,"field":"honba","region":"honba",
                              "bbox":[i*55+5,10,40,40],"rotation":0} for i in (0,1)]}
    path = tmp_path/"numbers.json"
    write_json(path,annotation)
    bank_path = tmp_path/"numbers/manifest.json"
    build_templates([path],bank_path)
    assert TemplateBank(bank_path).text_match(pixels[10:50,5:45],"honba","honba",.98,.04)["value"] == 0
    assert TemplateBank(bank_path).text_match(pixels[10:50,5:45],"kyotaku","kyotaku",.98,.04)["value"] is None
    annotation["targets"][0]["value"] = 1
    write_json(path,annotation)
    with pytest.raises(VisionError,match="结构化值"):
        build_templates([path],tmp_path/"bad-numbers/manifest.json")


def test_conflicting_ocr_and_template_evidence_stops_text_recognition():
    c = config()
    c["regions"] = {"honba":{"kind":"text","field":"honba","rect":[0,0,.2,.2],"required":True}}
    bank = SimpleNamespace(text_match=lambda *a:{"value":0,"score":1,"raw_text":"0"})
    ocr = SimpleNamespace(read=lambda *a:{"value":1,"score":1,"raw_text":"1"})
    result = StaticRecognizer(c,bank,ocr).recognize(np.zeros((128,256,3),dtype=np.uint8))
    assert result["regions"]["honba"]["text"]["value"] is None
    assert result["regions"]["honba"]["text"]["reason"] == "conflicting_text_evidence"
    assert result["status"] == "needs_confirmation"


def test_text_cache_changes_with_pixels_field_threshold_and_cannot_be_mutated():
    reader = TextReader.__new__(TextReader)
    reader.cache_size, reader._read_cache = 2, OrderedDict()
    calls = []
    def infer(pixels):
        calls.append(int(pixels[0,0,0]))
        return str(int(pixels[0,0,0])), .99
    reader._infer = infer
    pixels = np.ones((10,10,3),np.uint8)
    result = reader.read(pixels,"honba",.9)
    result["value"] = 55
    result["variants"][0]["text"] = "55"
    repeated = reader.read(pixels,"honba",.9)
    assert repeated["value"] == 1 and repeated["variants"][0]["text"] == "1"
    assert len(calls) == 1
    pixels[0,0,0] = 2
    assert reader.read(pixels,"honba",.9)["value"] == 2
    assert reader.read(pixels,"remaining",.9)["value"] == 2
    assert len(calls) == 3 and len(reader._read_cache) == 2
    reader.read(pixels,"remaining",.95)
    assert len(calls) == 4 and len(reader._read_cache) == 2
    pixels[:] = 1  # The original entry was evicted, so this must infer again.
    reader.read(pixels,"honba",.9)
    assert len(calls) == 5


def test_clipped_region_fragments_are_preserved_and_do_not_count_as_known_empty(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    c = config()
    c["regions"] = {"meld_self":{"kind":"meld","rect":[0,.5,.5,.5],"rotation":0,"required":True,
                                 "slots":[[0,0,40,60]],"reject_boundary_fragments":True}}
    result = StaticRecognizer(c,TemplateBank(bank_path)).recognize(pixels)
    region = result["regions"]["meld_self"]
    assert region["items"] == [] and region["boundary_fragments"][0]["label"] == "?"
    assert not region["complete"] and result["status"] == "needs_confirmation"


@pytest.mark.parametrize("label,complete",[("back",True),("1m",False),("?",False)])
def test_only_a_confident_back_match_can_exclude_a_boundary_fragment(label,complete):
    c = config()
    c["regions"] = {"meld":{"kind":"meld","rect":[0,.5,.5,.5],"required":True,
                            "slots":[[0,0,40,60]],"reject_boundary_fragments":True}}
    pixels = np.zeros((128,256,3),dtype=np.uint8)
    pixels[64:124,:40] = 230
    bank = SimpleNamespace(classify=lambda *a:{"label":label,"score":1,"margin":1})
    result = StaticRecognizer(c,bank).recognize(pixels)
    assert result["regions"]["meld"]["complete"] is complete
    assert result["regions"]["meld"]["items"] == []
    if label == "back":
        assert len(result["regions"]["meld"]["backs"]) == 1
    else:
        assert len(result["regions"]["meld"]["boundary_fragments"]) == 1


@pytest.mark.parametrize("background,fitted,neutral,complete", [
    (True, False, False, True), (True, False, True, False),
    (True, True, False, False), (False, False, False, False)])
def test_boundary_decoration_needs_both_missing_face_and_colour_evidence(background,fitted,neutral,complete):
    c = config()
    c["regions"] = {"meld":{"kind":"meld","rect":[0,.5,.5,.5],"required":True,
                            "slots":[[0,0,40,60]],"reject_boundary_fragments":True}}
    pixels = np.zeros((128,256,3),dtype=np.uint8)
    pixels[64:124,:40] = 230
    bank = SimpleNamespace(classify=lambda *a:{"label":"?","score":.36,"reason":"pjura_low_confidence",
                                              "background_candidate":background,
                                              "pjura_evidence":{"face_fitted":fitted,"face_tint":{"neutral":neutral}}})
    region = StaticRecognizer(c,bank).recognize(pixels)["regions"]["meld"]
    assert region["complete"] is complete and region["items"] == []
    if complete:
        assert len(region["background_candidates"]) == 1 and not region.get("boundary_fragments")
    else:
        assert len(region["boundary_fragments"]) == 1


def test_box_matching_counts_missing_and_extra_without_label_bias():
    expected = [{"bbox":[0,0,20,30],"label":"1m"},{"bbox":[40,0,20,30],"label":"2m"}]
    actual = [{"bbox":[1,0,20,30],"label":"9m"},{"bbox":[90,0,20,30],"label":"2m"}]
    assert match_boxes(expected,actual) == ([(0,0)],[1],[1])


def test_workbench_correction_persists_original_frame_and_audit(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    image.with_suffix(".annotation.json").write_text(json.dumps(annotation),encoding="utf-8")
    config_path = tmp_path/"config.json"
    write_json(config_path,config())
    root = tk.Tk()
    root.withdraw()
    try:
        ui = Workbench(root,image,config_path)
        ui.annotation["targets"].clear()
        ui.observation = StaticRecognizer(config(),TemplateBank(bank_path)).recognize(pixels)
        ui.drag_start = [20,80]
        ui.pointer_up(SimpleNamespace(x=20*ui.scale,y=80*ui.scale))
        assert ui.selected == 0
        assert ui.annotation["targets"][0]["source"] == "recognition_needs_manual_review"
        ui.label.set("1m")
        ui.apply_correction()
        ui.reviewed.set(True)
        ui.save_annotation()
        saved = json.loads(image.with_suffix(".annotation.json").read_text(encoding="utf-8"))
        assert saved["targets"][0]["label"] == "1m"
        assert np.array_equal(load_image(image),pixels)
        history = image.with_suffix(".annotation.corrections.jsonl").read_text(encoding="utf-8")
        assert '"before"' in history and '"after"' in history
    finally:
        root.destroy()


def test_workbench_pause_regions_and_capture_warning_survive_save_and_recognition(tmp_path,monkeypatch):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    image.with_suffix(".annotation.json").write_text(json.dumps(annotation),encoding="utf-8")
    write_json(image.with_suffix(".frame.json"),{"capture_quality":"requires_manual_overlay_review","frame_id":42})
    config_path = tmp_path/"config.json"
    c = config()
    c["templates"] = str(bank_path)
    write_json(config_path,c)
    received = []
    def recognize(pixels,metadata):
        received.append(metadata)
        return {"status":"needs_confirmation","regions":{},"issues":[],"recognition_ms":0}
    monkeypatch.setattr(workbench_module,"StaticRecognizer",lambda *a:SimpleNamespace(recognize=recognize))
    monkeypatch.setattr(workbench_module,"TemplateBank",lambda *a:None)
    monkeypatch.setattr(workbench_module,"TextReader",lambda:None)
    monkeypatch.setattr(workbench_module.simpledialog,"askstring",lambda *a,**k:"hand, hand")
    monkeypatch.setattr(workbench_module.messagebox,"showerror",lambda title,message:pytest.fail(f"{title}: {message}"))
    root = tk.Tk()
    root.withdraw()
    try:
        ui = Workbench(root,image,config_path)
        ui.mark_pause_regions()
        assert ui.annotation["expected_pause_regions"] == ["hand"] and not ui.reviewed.get()
        ui.quality.set("occluded")
        ui.save_annotation()
        saved = json.loads(image.with_suffix(".annotation.json").read_text(encoding="utf-8"))
        assert saved["expected_pause_regions"] == ["hand"] and saved["frame_quality"] == "occluded"
        assert not saved["reviewed"]
        audit = image.with_suffix(".annotation.corrections.jsonl").read_text(encoding="utf-8")
        assert '"expected_pause_regions"' in audit
        ui.recognize()
        assert received[0]["capture_quality"] == "requires_manual_overlay_review" and received[0]["frame_id"] == 42
        assert received[0]["image_path"] == str(image.resolve())
        assert np.array_equal(load_image(image),pixels)
    finally:
        root.destroy()


def test_recalibrating_draw_invalidates_the_old_coupled_hand_layout(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    c = config()
    c["regions"]["hand"]["hand_layout"] = {"draw_region":"draw","draw_gap":15,"concealed_counts":[1]}
    c["regions"]["draw"] = {"kind":"draw","player":0,"rect":[.5,.5,.25,.5]}
    config_path = tmp_path/"config.json"
    write_json(config_path,c)
    root = tk.Tk()
    root.withdraw()
    try:
        ui = Workbench(root,image,config_path)
        ui.mode.set("调整区域")
        ui.region.set("draw")
        ui.drag_start = [130,65]
        ui.pointer_up(SimpleNamespace(x=190*ui.scale,y=125*ui.scale))
        assert "hand_layout" not in ui.config["regions"]["hand"]
        assert not ui.config["calibrated"]
        validate_config(ui.config)
    finally:
        root.destroy()


def test_svm_verification_rejects_unanimous_wrong_votes_without_template_support(monkeypatch):
    verifier = SvmVerifier.__new__(SvmVerifier)
    verifier.sha256 = "fixed"
    verifier.classes = sorted(TILES)
    def model(label):
        return SimpleNamespace(predict=lambda _: (0,np.array([[verifier.classes.index(label)]],np.float32)))
    monkeypatch.setattr(svm_module,"descriptor",lambda *a:np.zeros(692,np.float32))
    verifier.models = [model("6m")]*3
    pixels = np.zeros((60,40,3),np.uint8)
    proposal = {"label":"?","best_label":"3m","score":.85}
    assert not verifier.verify(pixels,proposal)["accepted"]
    verifier.models = [model("3m")]*3
    assert verifier.verify(pixels,proposal)["accepted"]
    verifier.models = [model("3m"),model("3m"),model("6m")]
    assert not verifier.verify(pixels,proposal)["accepted"]
    verifier.models = [model("3m")]*3
    assert not verifier.verify(pixels,{**proposal,"score":.7})["accepted"]
    assert not verifier.verify(pixels,{**proposal,"best_label":"P","blank_face":False})["accepted"]
    monkeypatch.setattr(svm_module,"descriptor",lambda *a:None)
    assert not verifier.verify(pixels,proposal)["accepted"]


def test_template_svm_cannot_replace_a_known_label_conflict_back_or_missing_face(monkeypatch):
    bank = TemplateBank.__new__(TemplateBank)
    bank.face_entries = [True]
    calls = []
    bank.verifier = SimpleNamespace(verify=lambda *args:calls.append(args) or {"accepted":True})
    unknown = {"label":"?","best_label":"3m","score":.85,"face_rotation":0}
    body = {**unknown,"score":.8}
    face = unknown
    monkeypatch.setattr(bank,"_classify_body",lambda *a,**k:body)
    monkeypatch.setattr(bank,"_classify_face",lambda *a,**k:face)
    pixels = np.zeros((60,40,3),np.uint8)
    refined = bank.classify(pixels,"hand",.9,.04)
    assert refined["label"] == "3m" and refined["classifier_evidence"]["accepted"]
    body = {**body,"label":"1m"}
    face = {**face,"label":"2m"}
    assert bank.classify(pixels,"hand",.9,.04)["reason"] == "conflicting_tile_features"
    body = {**body,"label":"back"}
    face = {**face,"label":"?"}
    assert bank.classify(pixels,"hand",.9,.04)["label"] == "back"
    body = unknown
    face = None
    assert bank.classify(pixels,"hand",.9,.04)["label"] == "?"
    assert len(calls) == 1


@pytest.mark.parametrize("failure",["version","policy","test_source","class_order","models","hash"])
def test_svm_loader_rejects_incompatible_or_untrusted_evidence(tmp_path,failure):
    value = {"feature_version":svm_module.FEATURE_VERSION,"feature_parameters":deepcopy(SVM_FEATURE_PARAMETERS),
             "policy":deepcopy(SVM_POLICY),"opencv_version":cv2.__version__,"classes":sorted(TILES),
             "sources":[{"game_id":"train-game","split":"train","reviewed":True}],
             "models":[{"C":10,"gamma":gamma,"path":"model.xml","sha256":"incorrect"} for gamma in (.01,.05,.1)]}
    if failure == "version":
        value["feature_version"] = "unsupported"
    elif failure == "policy":
        value["policy"]["minimum_template_proposal_score"] = 0
    elif failure == "test_source":
        value["sources"][0]["split"] = "test"
    elif failure == "class_order":
        value["classes"].reverse()
    elif failure == "models":
        value["models"].pop()
    (tmp_path/"model.xml").write_bytes(b"changed model")
    write_json(tmp_path/"manifest.json",value)
    with pytest.raises(VisionError):
        SvmVerifier(tmp_path/"manifest.json")


def test_test_game_used_only_by_svm_is_still_rejected_from_acceptance(tmp_path):
    pixels,image,path,annotation,bank_path = make_bank(tmp_path)
    annotation.update(split="test",game_id="svm-only-game")
    write_json(path,annotation)
    bank = TemplateBank(bank_path)
    bank.training_games.add("svm-only-game")
    with pytest.raises(VisionError,match="模板来源对局"):
        evaluate(StaticRecognizer(config(),bank),[path])


@pytest.mark.parametrize("label,complete",[("back",True),("1m",False),("?",False)])
def test_oversized_foreground_can_only_be_excluded_as_a_confident_back(label,complete):
    c = config()
    c["regions"] = {"meld":{"kind":"meld","rect":[.25,.125,.5,.75],"rotation":0,"required":True,
                            "min_size":[20,25],"max_size":[70,80]}}
    pixels = np.zeros((128,256,3),np.uint8)
    pixels[16:112,64:192] = 230
    bank = SimpleNamespace(classify=lambda *a:{"label":label,"score":1,"margin":1})
    region = StaticRecognizer(c,bank).recognize(pixels)["regions"]["meld"]
    assert region["complete"] is complete and region["items"] == []
    if label == "back":
        assert len(region["backs"]) == 1 and "unresolved_components" not in region
    else:
        assert len(region["unresolved_components"]) == 1
