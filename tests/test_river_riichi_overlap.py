"""Thick riichi bodies and overlap with the preceding upright glyph."""
from copy import deepcopy
from pathlib import Path

import cv2
import pytest

from app.capture.frames import load_image
from app.vision.detector import split_by_grid,trim_river_neighbour_sides
from tests.test_table_cleanup import LEFT_GRID,RIGHT_GRID,TOP_GRID

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('grid,box',[(LEFT_GRID,[84,127,51,44]),(RIGHT_GRID,[84,120,52,46])])
def test_one_thick_sideways_tile_is_not_two_upright_tiles(grid,box):
    assert split_by_grid(box,grid)==[box]


def test_top_river_short_upright_rows_keep_their_existing_split():
    assert len(split_by_grid([10,60,76,31],TOP_GRID))==2


def test_crop_only_the_overlapping_side_and_keep_original_bodies_as_evidence():
    boxes=[[56,116,41,55],[84,127,51,44]];original=deepcopy(boxes)
    cropped,evidence=trim_river_neighbour_sides(boxes,LEFT_GRID)
    assert cropped==[[56,116,30,55],[84,127,51,44]]
    assert boxes==original
    assert evidence[0]['bbox']==boxes[0] and evidence[0]['neighbour_bbox']==boxes[1]


@pytest.mark.parametrize('mutation',['no_neighbour','other_row','upright_neighbour','large_occlusion','off_grid'])
def test_unproven_overlap_is_never_trimmed(mutation):
    boxes=[[56,116,41,55],[84,127,51,44]]
    if mutation=='no_neighbour':boxes.pop()
    elif mutation=='other_row':boxes[1][1]+=53
    elif mutation=='upright_neighbour':boxes[1][2:]=[43,57]
    elif mutation=='large_occlusion':boxes[0][2]=58
    else:
        for box in boxes:box[1]+=400
    cropped,evidence=trim_river_neighbour_sides(boxes,LEFT_GRID)
    assert cropped==boxes and evidence==[]


@pytest.fixture(scope='module')
def advice():
    from app.advice import ScreenshotAdvice
    from app.vision.config import load_config
    from app.engine.runtime import OnnxRuntime
    from app.engine.snapshot import SnapshotBackend
    return ScreenshotAdvice(load_config(ROOT/'configs/window-steam-1366-v2.json'),
                            backend=SnapshotBackend(OnnxRuntime()))


@pytest.mark.parametrize('size',[(1920,1080),(1366,768)])
def test_real_screenshot_recovers_sixteen_tiles_and_the_riichi(advice,size):
    pixels=load_image(ROOT/'data/fixtures/river-left-late/client-1920.png')
    if size!=(1920,1080):pixels=cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)
    advice.clear();advice.recognizer.river_memory.clear()
    result=advice.process(pixels)
    assert result['status']=='ready',result['reason']
    assert result['corrections']=={} and result['context']=={}
    river=result['observation']['regions']['river_left']
    assert river['complete'] and not river.get('slot_gaps')
    assert [i['label'] for i in river['items']]==[
        'F','5sr','P','7p','9m','6m','8s','8m','3p','2p','4s','C','E','9p','7p','E']
    assert [i['slot'] for i in river['items']]==[[n//6,n%6] for n in range(16)]
    assert river['items'][-1]['horizontal'] is True
    assert result['snapshot']['riichi']==[False,False,False,True]
    assert river['items'][-1]['canonical_bbox'][2:]==[51,44]
    assert river['items'][-2]['score']>=.9
    assert len(river['neighbour_crops'])==1
    assert result['observation']['tile_classifier']['confidence']==.9
    decision=result['decision']
    assert decision['action_mask'][decision['recommendation']['action_id']]
