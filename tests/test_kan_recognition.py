"""Closed-kan structure, screenshot event conversion and actual user regressions."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.advice import ScreenshotAdvice
from app.capture.frames import load_image
from app.state.snapshot import SnapshotError, build_snapshot
from app.vision.config import load_config
from app.vision.detector import blue_back_evidence
from app.vision.recognizer import closed_kan_labels, group_melds, meld_items_with_kan_backs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/fixtures/kan-recognition'


def closed_tiles(tile='3p'):
    return [{'label':label, 'horizontal':False, 'canonical_bbox':[index*35,10,36,39],
             'full_body':True} for index,label in enumerate(['back',tile,tile,'back'])]


def test_closed_kan_keeps_observed_backs_and_infers_only_the_group():
    items = closed_tiles()
    group = group_melds(items, {})[0]
    assert group['type'] == 'ankan' and group['status'] == 'confirmed'
    assert group['target'] is None and group['called_positions'] == []
    assert group['labels'] == ['3p']*4
    assert group['observed_labels'] == ['back','3p','3p','back']
    assert [i['label'] for i in items] == group['observed_labels']


@pytest.mark.parametrize('mutation', ['missing_back','mismatched_faces','unknown_face','offset_back',
                                      'overlapping_back','wide_back','sideways_face','all_faces','all_backs'])
def test_unproven_closed_kan_is_not_inferred(mutation):
    items = closed_tiles()
    if mutation == 'missing_back': items.pop()
    elif mutation == 'mismatched_faces': items[2]['label'] = '4p'
    elif mutation == 'unknown_face': items[2]['label'] = '?'
    elif mutation == 'offset_back': items[-1]['canonical_bbox'][1] += 30
    elif mutation == 'overlapping_back': items[-1]['canonical_bbox'][0] -= 25
    elif mutation == 'wide_back': items[-1]['canonical_bbox'][2] *= 2
    elif mutation == 'sideways_face': items[1]['horizontal'] = True
    elif mutation == 'all_faces':
        for i in items: i['label'] = '3p'
    else:
        for i in items: i['label'] = 'back'
    assert closed_kan_labels(items) is None
    assert all(g['type'] != 'ankan' or g['status'] != 'confirmed' for g in group_melds(items, {}))


@pytest.mark.parametrize('faces', [('5m','5m'), ('5mr','5m'), ('5m','5mr')])
def test_closed_four_fives_preserve_the_fixed_one_red_copy_rule(faces):
    items = closed_tiles('5m')
    items[1]['label'],items[2]['label'] = faces
    labels = closed_kan_labels(items)
    assert labels.count('5mr') == 1 and labels.count('5m') == 3
    assert labels[1:3] == list(faces)
    items[1]['label'] = items[2]['label'] = '5mr'
    assert closed_kan_labels(items) is None


def test_closed_kan_can_be_separated_from_a_touching_pon():
    items = closed_tiles()
    items += [{'label':'C','horizontal':i==0,'canonical_bbox':[140+i*35,10,35,39]} for i in range(3)]
    groups = group_melds(items, {'pon_source_by_called_position':{'0':3}})
    assert [(g['type'],g['target'],g['status']) for g in groups] == [('ankan',None,'confirmed'),('pon',3,'confirmed')]


def test_clipped_or_unrelated_hand_backs_cannot_complete_a_kan():
    items = closed_tiles()
    output = {'items':items[1:3], 'backs':[items[0],items[3]]}
    assert len(meld_items_with_kan_backs(output)) == 4
    items[0]['full_body'] = False
    assert meld_items_with_kan_backs(output) == items[1:3]
    assert meld_items_with_kan_backs({'items':[], 'backs':output['backs']}) == []


def test_blue_back_detection_does_not_name_white_or_green_tile_faces():
    patch = np.full((40,36,3), 225, np.uint8)
    assert blue_back_evidence(patch) is None
    patch[8:24,8:25] = (50,130,50)
    assert blue_back_evidence(patch) is None
    patch[8:24,8:25] = (230,200,165)
    assert blue_back_evidence(patch)['source'] == 'visible_blue_back_pattern'
    patch[:] = (230,200,165)
    assert blue_back_evidence(patch) is None  # uniform tint is not a back pattern


@pytest.fixture(scope='module')
def advice():
    from app.engine.runtime import OnnxRuntime
    from app.engine.snapshot import SnapshotBackend
    config = load_config(ROOT / 'configs/window-steam-1366-v2.json')
    return ScreenshotAdvice(config, backend=SnapshotBackend(OnnxRuntime()))


@pytest.mark.parametrize('size', [(1920,1080), (1366,768)])
@pytest.mark.parametrize('name', ['top-3p-ankan','top-C-pon-arrow'])
def test_user_screenshots_preserve_kan_structure_and_advice_limits(advice, size, name):
    pixels = load_image(DATA / (name+'-1920.png'))
    if size != (1920,1080):
        pixels = cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
    advice.clear()
    advice.recognizer.river_memory.clear()
    result = advice.process(pixels)
    if size == (1366,768) and name == 'top-C-pon-arrow':
        # One downsampled left-river glyph still has no sufficiently dominant
        # reading. The new class-acceptance rule must not guess it.
        assert result['status'] == 'needs_confirmation'
        assert result['reason'].startswith('请核对 river_left')
        assert result['corrections'] == {} and result['snapshot'] is None
        assert [g['type'] for g in result['observation']['regions']['meld_right']['groups']] == ['pon','pon']
        return
    assert result['status'] == 'ready', result['reason']
    assert result['corrections'] == {} and result['context'] == {}
    observation, snapshot = result['observation'],result['snapshot']
    if name == 'top-3p-ankan':
        assert snapshot['melds'][2] == [{'type':'ankan','actor':2,'consumed':['3p']*4}]
        group = observation['regions']['meld_top']['groups'][0]
        assert group['observed_labels'][0] == group['observed_labels'][3] == 'back'
        assert '3p' in group['observed_labels'][1:3]
        assert [i['label'] for i in observation['regions']['meld_top']['items']] == ['3p','3p']
    else:
        assert [i['label'] for i in observation['regions']['river_left']['items']] == ['W','N','S','2p','4m','2s','2p','1p','9p']
        assert [g['type'] for g in snapshot['melds'][1]] == ['pon','pon']
        assert observation['reaction_discard']['tile'] == '9p'
        assert observation['regions']['meld_self']['complete']
        assert any(i.get('detector_reason') == 'unsplit_wide_component'
                   for i in observation['regions']['meld_self']['background_candidates'])
    decision = result['decision']
    assert decision['action_mask'][decision['recommendation']['action_id']]


@pytest.mark.parametrize('mutation', ['none','wrong_proposal','weak_proposal','two_unknown'])
def test_closed_kan_may_settle_only_one_matching_strong_face_proposal(mutation):
    items = closed_tiles()
    items[2].update(label='?',best_label='3p',score=.895,margin=.89,reason='pjura_low_confidence')
    if mutation == 'wrong_proposal': items[2]['best_label'] = '2p'
    elif mutation == 'weak_proposal': items[2]['score'] = .3
    elif mutation == 'two_unknown': items[1]['label'] = '?'
    group = group_melds(items,{})[0]
    assert (group['status'] == 'confirmed') is (mutation == 'none')
    if mutation == 'none':
        assert items[2]['label'] == '3p' and items[2]['structural_reading']['label'] == '?'
        assert group['observed_labels'] == ['back','3p','?','back']


def test_closed_kan_fourth_copy_is_included_in_the_physical_copy_check(advice):
    pixels = load_image(DATA / 'top-3p-ankan-1920.png')
    # Reuse the observed group and add an independently visible impossible 3p.
    observed = advice.recognizer.recognize(pixels)
    observed['regions']['draw']['items'][0]['label'] = '3p'
    with pytest.raises(SnapshotError,match='超过|张数|数量|4'):
        build_snapshot(observed, advice.config)


@pytest.mark.parametrize('size', [(1920,1080), (1366,768)])
def test_dominant_8p_recovers_user_reaction_frame_without_review(advice, size):
    pixels = load_image(ROOT / 'data/fixtures/dominant-candidate/client-1920.png')
    if size != (1920,1080):
        pixels = cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
    advice.clear()
    advice.recognizer.river_memory.clear()
    result = advice.process(pixels)
    assert result['status'] == 'ready', result['reason']
    assert result['corrections'] == {} and result['context'] == {}
    region = result['observation']['regions']['river_right']
    tile = next(i for i in region['items'] if i.get('slot') == [1,5])
    assert region['complete'] and tile['label'] == '8p'
    if tile.get('fixed_polygon'):
        # A calibrated crop now raises this same 8p above the primary threshold.
        assert tile['score'] >= .9
        assert tile['fixed_polygon']['slot'] == [1,5]
    else:
        assert .5 < tile['score'] < .9
        evidence = tile['pjura_evidence']['class_acceptance']
        assert evidence['rule'] == 'dominant_candidate'
        assert evidence['top1_to_second_ratio'] >= 3
    reaction = result['observation']['reaction_discard']
    assert reaction['confirmed'] and reaction['target'] == 3 and reaction['tile'] == '7p'
    decision = result['decision']
    assert decision['action_mask'][decision['recommendation']['action_id']]
