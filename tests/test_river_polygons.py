from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from app.vision.config import VisionError, validate_config
from app.vision.river_polygons import RiverPolygonReader, face_presence, layout_signature


def fixture(tmp_path):
    region={"kind":"river","rotation":0,"rect":[0,0,1,1],"grid":{"anchor":"center",
        "origin":[22.5,30],"basis":[[35,0],[0,50]],"rows":2,"columns":3,"tolerance":13}}
    config={"version":"test","client_size":[200,180],"seat_mapping":{"self":0,"right":1,"top":2,"left":3},
        "regions":{"river_self":region},"tile_classifier":{"name":"pjura","manifest":"unused"}}
    coef=np.array([[10,10,35,10,35,50,10,50],[35,0,35,0,35,0,35,0],
                   [0,50,0,50,0,50,0,50],[0,0,0,0,0,0,0,0]],float)
    polygons=[{"slot":[r,c],"quad":(np.array([1,c,r,c*r])@coef).reshape(4,2).tolist()}
              for r in range(2) for c in range(3)]
    model={"reference_size":[200,180],"layout":layout_signature(config),
           "regions":{"river_self":{"coefficients":coef.tolist(),"polygons":polygons}}}
    path=tmp_path/"calibration.json"
    path.write_text(json.dumps(model),encoding="utf-8")
    config["river_polygons"]={"manifest":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    seen=[];fallback=[];legacy_seen=[]
    def predict(crops):
        seen.extend(crops)
        return [{"label":"3s","score":.99,"margin":.98,"alternatives":[]} for _ in crops]
    model_object=SimpleNamespace(manifest_sha256="manifest",manifest={"model_sha256":"model"},provider="test",predict_many=predict)
    legacy=SimpleNamespace(classify=lambda pixels,*a,**k:legacy_seen.append(pixels) or
        {"label":"3s","score":.99,"margin":.98,"face_rotation":0})
    def original(crops,*a,**k):
        fallback.extend(crops)
        return [{"label":"4s","score":.99,"margin":.98,"orientation_confirmed":True,"face_rotation":90} for _ in crops]
    bank=SimpleNamespace(legacy=legacy,model=model_object,classify_many=original)
    return config,bank,seen,fallback,legacy_seen,path


def test_frozen_face_uses_native_pixels_and_preserves_raw_template_crop(tmp_path):
    cfg,bank,seen,fallback,legacy_seen,_=fixture(tmp_path)
    reader=RiverPolygonReader(cfg,bank)
    reference=np.full((180,200,3),225,np.uint8);native=np.full((360,400,3),225,np.uint8)
    got=reader.classify("river_self",cfg["regions"]["river_self"],reference,native,[[8,8,29,46]],.9,.04)[0]
    assert got["label"]=="3s" and got["orientation_source"]=="calibrated_river_slot"
    assert got["face_rotation"]==0 and got["orientation_confirmed"]
    assert seen[0].shape==(94,64,3) and legacy_seen[0].shape==(46,29,3)
    assert got["fixed_polygon"]["quad_local"][0]==[10.,10.] and not fallback


def test_empty_river_never_sends_any_of_its_slots_to_the_model(tmp_path):
    cfg,bank,seen,fallback,_,_=fixture(tmp_path)
    reader=RiverPolygonReader(cfg,bank);pixels=np.zeros((180,200,3),np.uint8)
    assert reader.classify("river_self",cfg["regions"]["river_self"],pixels,pixels,[],.9,.04)==[]
    assert seen==[] and fallback==[]


def test_uncertain_new_warp_keeps_the_original_dominance_path_and_exact_native_crop(tmp_path):
    cfg,bank,seen,fallback,_,_=fixture(tmp_path)
    cfg["tile_classifier"]["dominant_candidate"]={"minimum_confidence":.5,"minimum_ratio":3.}
    bank.model.predict_many=lambda crops:[{"label":"3m","score":.64,"margin":.6,"alternatives":[]} for _ in crops]
    received=[]
    def original(crops,*args,**kwargs):
        received.extend(kwargs["model_crops"])
        return [{"label":"2m","score":.81,"margin":.7} for _ in crops]
    bank.classify_many=original
    reader=RiverPolygonReader(cfg,bank);pixels=np.full((180,200,3),225,np.uint8)
    exact=np.full((47,31,3),200,np.uint8)
    got=reader.classify("river_self",cfg["regions"]["river_self"],pixels,pixels,[[8,8,29,46]],.9,.04,
                       raw_model_crops=[exact])[0]
    assert got["label"]=="2m" and got["fixed_polygon_fallback"] and received[0] is exact


def test_misaligned_or_nonwhite_face_uses_original_reader(tmp_path):
    cfg,bank,seen,fallback,_,_=fixture(tmp_path)
    reader=RiverPolygonReader(cfg,bank);pixels=np.zeros((180,200,3),np.uint8)
    got=reader.classify("river_self",cfg["regions"]["river_self"],pixels,pixels,[[8,8,29,46]],.9,.04)[0]
    assert got["label"]=="4s" and got["fixed_polygon_fallback"] and not seen and len(fallback)==1


def test_riichi_and_the_following_tile_keep_physical_slots(tmp_path):
    cfg,bank,seen,fallback,_,_=fixture(tmp_path)
    reader=RiverPolygonReader(cfg,bank);pixels=np.full((180,200,3),225,np.uint8)
    got=reader.classify("river_self",cfg["regions"]["river_self"],pixels,pixels,[[8,12,45,30],[55,10,25,40]],.9,.04)
    assert got[0]["fixed_polygon_fallback"] and got[0]["label"]=="4s"
    assert got[1]["fixed_polygon"]["slot"]==[0,1]
    assert got[1]["fixed_polygon"]["quad_local"][0]==pytest.approx([55,10])
    assert len(seen)==1 and len(fallback)==1


def test_a_grey_side_in_half_an_empty_slot_does_not_become_a_white_dragon():
    face=np.zeros((94,64,3),np.uint8);face[:,:32]=180
    assert not face_presence(face)["occupied"]
    face[:]=215;face[20:75:12,18:50]=20
    assert face_presence(face)["occupied"]


@pytest.mark.parametrize("mutation",["hash","layout","coefficients","corners","slot_order"])
def test_changed_calibration_or_layout_is_rejected(tmp_path,mutation):
    cfg,bank,*_,path=fixture(tmp_path)
    if mutation=="hash":cfg["river_polygons"]["sha256"]="0"*64
    elif mutation=="layout":cfg["regions"]["river_self"]["rect"]=[0,0,.9,1]
    else:
        model=json.loads(path.read_text(encoding="utf-8"))
        if mutation=="coefficients":model["regions"]["river_self"]["coefficients"][0][0]=float("nan")
        elif mutation=="corners":model["regions"]["river_self"]["polygons"][0]["quad"]=[[0,0]]*4
        else:model["regions"]["river_self"]["polygons"][0]["slot"]=[1,1]
        path.write_text(json.dumps(model),encoding="utf-8")
        cfg["river_polygons"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(VisionError):RiverPolygonReader(cfg,bank)


def test_configuration_rejects_a_missing_or_malformed_manifest_hash(tmp_path):
    cfg,*_=fixture(tmp_path)
    validate_config(cfg)
    for value in (False,{}, {"manifest":"p","sha256":"short"}):
        invalid=deepcopy(cfg);invalid["river_polygons"]=value
        with pytest.raises(VisionError):validate_config(invalid)


def test_a_meld_only_component_does_not_load_unrelated_river_calibration(tmp_path):
    from app.vision.pjura import PjuraBank
    from app.vision.recognizer import StaticRecognizer
    cfg,bank,*_=fixture(tmp_path)
    cfg["regions"]={"meld_self":{"kind":"meld","rect":[0,0,1,1]}}
    cfg["client_size"]=[128,128]
    cfg["river_polygons"]["manifest"]=str(tmp_path/"not-needed.json")
    wrapped=PjuraBank(bank.legacy,cfg["tile_classifier"],model=bank.model)
    recognizer=StaticRecognizer(cfg,wrapped)
    assert recognizer.river_polygons is None
