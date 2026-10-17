"""Real public kan regressions and independent geometry/event boundary cases."""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.advice import ScreenshotAdvice
from app.capture.frames import load_image
from app.vision.config import load_config
from app.vision.detector import orange_back_evidence
from app.vision.kan import kan_structure
from app.vision.recognizer import group_melds
from app.vision.templates import TemplateBank
from scripts.benchmark_kan_scenes import evaluate

ROOT=Path(__file__).resolve().parents[1]
CONFIG=load_config(ROOT/'configs/window-steam-1366-v2.json')
SEATS=['self','right','top','left']


def open_tiles(called,tile='3p'):
    items=[];x=0
    for n in range(4):
        w,h=(60,40) if n==called else (40,60)
        items.append({'label':tile,'horizontal':n==called,'canonical_bbox':[x,60-h,w,h]})
        x+=w
    return items


def added_tiles(called,tile='F'):
    items=[];x=0
    for n in range(3):
        if n==called:
            items += [{'label':tile,'horizontal':True,'canonical_bbox':[x,0,60,40]},
                      {'label':tile,'horizontal':True,'canonical_bbox':[x+5,40,60,40]}]
            x+=60
        else:
            items.append({'label':tile,'horizontal':False,'canonical_bbox':[x,20,40,60]});x+=40
    return items


@pytest.mark.parametrize('actor',range(4))
@pytest.mark.parametrize('called',range(4))
def test_open_kan_source_for_every_actor_and_called_position(actor,called):
    region=CONFIG['regions']['meld_'+SEATS[actor]]
    group=group_melds(open_tiles(called),region)[0]
    assert group['type']=='daiminkan' and group['status']=='confirmed'
    assert group['target']==(actor+{0:3,1:2,2:2,3:1}[called])%4
    assert group['called_positions']==[called]


@pytest.mark.parametrize('actor',range(4))
@pytest.mark.parametrize('called',range(3))
def test_added_kan_keeps_the_added_layer_and_original_pon_source(actor,called):
    group=group_melds(added_tiles(called),CONFIG['regions']['meld_'+SEATS[actor]])[0]
    assert group['type']=='kakan' and group['status']=='confirmed'
    assert group['called_positions']==[called,called+1]
    assert group['added_position']==called
    assert group['target']==(actor+3-called)%4


@pytest.mark.parametrize('mutation',['four_upright','missing_source','duplicate_body','missing_fourth',
                                   'offset_body','separated_called_pair','nonadjacent_called_pair'])
def test_quad_identity_without_complete_independent_structure_is_not_confirmed(mutation):
    region=deepcopy(CONFIG['regions']['meld_self']);items=open_tiles(0)
    if mutation=='four_upright':items=open_tiles(-1)
    elif mutation=='missing_source':region.pop('kan_source_by_called_position')
    elif mutation=='duplicate_body':items[-1]['canonical_bbox']=items[-2]['canonical_bbox'].copy()
    elif mutation=='missing_fourth':items.pop()
    elif mutation=='offset_body':items[-1]['canonical_bbox'][1]+=90
    elif mutation=='separated_called_pair':
        items=added_tiles(1);items[1]['canonical_bbox'][1]-=100
    else:
        items=open_tiles(0);items[-1]['horizontal']=True
    assert kan_structure(items,region) is None


def test_red_added_tile_is_preserved_only_when_its_layer_is_proven():
    items=added_tiles(1,'5p');items[1]['label']='5pr'
    group=group_melds(items,CONFIG['regions']['meld_self'])[0]
    assert group['type']=='kakan' and group['items'][group['added_position']]['label']=='5pr'
    # Either layer could be the added tile here: a red/normal ambiguity must pause.
    items[1]['canonical_bbox'][1]=38
    assert kan_structure(items,CONFIG['regions']['meld_self']) is None


def test_orange_rims_and_green_faces_are_not_orange_kan_backs():
    face=np.full((60,40,3),225,np.uint8)
    face[-10:]=(20,155,245)
    assert orange_back_evidence(face) is None
    face[10:45,8:33]=(40,140,40)
    assert orange_back_evidence(face) is None
    face[:]=(20,155,245)
    assert orange_back_evidence(face)['orange_fraction']>.9


def test_unknown_quad_tile_without_identity_proposal_stays_unresolved():
    items=open_tiles(0)
    items[1]['label']='?'
    assert group_melds(items,CONFIG['regions']['meld_self'])[0]['status']=='needs_confirmation'


@pytest.mark.parametrize('mutation',['none','network_disagrees','template_disagrees','weak_score','weak_margin'])
def test_two_weak_closed_faces_require_independent_network_and_template_agreement(mutation):
    items=[{'label':label,'canonical_bbox':[n*35,10+n*8,35,60],'horizontal':False}
           for n,label in enumerate(['back','?','?','back'])]
    for i in items[1:3]:i.update(best_label='6p',score=.8,margin=.7,legacy_evidence={'label':'6p'})
    if mutation=='network_disagrees':items[2]['best_label']='9p'
    elif mutation=='template_disagrees':items[2]['legacy_evidence']['label']='?'
    elif mutation=='weak_score':items[2]['score']=.49
    elif mutation=='weak_margin':items[2]['margin']=.29
    group=group_melds(items,{})[0]
    assert (group['type']=='ankan' and group['status']=='confirmed')==(mutation=='none')


@pytest.fixture(scope='module')
def bank():
    from app.vision.pjura import PjuraBank
    return PjuraBank(TemplateBank(CONFIG['templates'],CONFIG.get('classifier_manifest')),CONFIG['tile_classifier'])


PUBLIC=json.loads((ROOT/'data/public-mahjong/kan-scenes/regressions.json').read_text(encoding='utf-8'))['cases']


@pytest.mark.parametrize('case',PUBLIC,ids=lambda c:c['name'])
def test_reviewed_public_kan_native_pixels(case,bank):
    row,observation,_=evaluate(case,CONFIG,bank)
    assert row['wrong_confirmation'] is False
    if case['name'] in {'top-E-added-late','top-5p-closed-red'}:
        # Darkened result screen and a red custom skin remain recorded failures.
        # These are never turned into guessed identities or backend states.
        assert row['correct'] is False
    else:assert row['correct'],row
    if case['name']=='left-C-open':
        assert observation['regions']['meld_left']['split_meld_faces']
    assert observation['tile_classifier']['confidence']==.9


@pytest.fixture(scope='module')
def advice(bank):
    from app.engine.runtime import OnnxRuntime
    from app.engine.snapshot import SnapshotBackend
    from app.vision.recognizer import StaticRecognizer
    from app.vision.text import TextReader
    return ScreenshotAdvice(CONFIG,recognizer=StaticRecognizer(CONFIG,bank,TextReader()),
                            backend=SnapshotBackend(OnnxRuntime()))


@pytest.mark.parametrize('size',[(1920,1080),(1366,768)])
def test_actual_user_added_kan_returns_fresh_legal_advice(advice,size):
    pixels=load_image(ROOT/'data/fixtures/kan-all-seats/user-self-F-kakan-1920.png')
    if size!=(1920,1080):pixels=cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
    advice.clear();advice.recognizer.river_memory.clear()
    result=advice.process(pixels)
    assert result['status']=='ready',result['reason']
    assert result['corrections']=={} and result['context']=={}
    group=result['observation']['regions']['meld_self']['groups'][0]
    assert group['labels']==['F']*4 and group['type']=='kakan'
    assert result['snapshot']['melds'][0]==[{'type':'kakan','actor':0,'pai':'F','consumed':['F']*3}]
    decision=result['decision']
    assert decision['action_mask'][decision['recommendation']['action_id']]
