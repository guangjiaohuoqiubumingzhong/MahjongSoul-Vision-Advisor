"""Who discarded the tile waiting for a call, and which call buttons are shown.

Colours are the ones measured on real 1920x1080 frames reduced to 1366x768.
"""
import cv2
import numpy as np
import pytest

from app.advice import check_reaction_banners
from app.state.snapshot import SnapshotError
from app.vision.actions import call_banners, highlighted_discard

FELT = (185, 120, 70)          # BGR of the blue table (H ~101, S ~130)
GLOW = (150, 167, 95)          # pale mint outline (H ~81, S ~86, V ~167)
BAMBOO = (60, 140, 40)         # green bamboo ink (H ~72)


def table():
    pixels = np.zeros((300, 420, 3), np.uint8)
    pixels[:] = FELT
    regions = {"river_left": {"kind": "river", "player": 3, "items": []},
               "river_top": {"kind": "river", "player": 2, "items": []}}
    for name, boxes, labels in (("river_left", [[40, 40, 52, 27], [40, 68, 52, 27], [40, 96, 53, 41]], ["S", "8s", "P"]),
                                ("river_top", [[250, 40, 37, 45], [212, 40, 37, 45]], ["2m", "C"])):
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            pixels[y:y+h, x:x+w] = (225, 228, 230)
            regions[name]["items"].append({"label": label, "bbox": box})
    return pixels, regions


def glow_around(pixels, box, thickness=3):
    x, y, w, h = box
    # The neighbour above hides the top edge; the outline wraps left, right and bottom.
    cv2.rectangle(pixels, (x - 6, y + 4), (x + w + 5, y + h + 6), GLOW, thickness)
    pixels[y:y+h, x:x+w] = (225, 228, 230)


def test_pale_outline_on_the_last_discard_names_the_seat_and_tile():
    pixels, regions = table()
    assert not highlighted_discard(pixels, regions, 0)["confirmed"]
    glow_around(pixels, [40, 96, 53, 41])
    result = highlighted_discard(pixels, regions, 0)
    assert result["confirmed"] and (result["target"], result["tile"]) == (3, "P")


def test_green_bamboo_ink_next_to_a_tile_is_not_an_outline():
    pixels, regions = table()
    # Ink strokes poking out of the 8s face, like the river_left false hits.
    for dx in (0, 14, 28):
        cv2.rectangle(pixels, (93 + dx % 3, 70 + dx // 2), (100 + dx % 3, 78 + dx // 2), BAMBOO, -1)
    pixels[66:67, 30:110] = BAMBOO        # a thin line along the edge, but only on one side
    result = highlighted_discard(pixels, regions, 0)
    assert not result["confirmed"] and not result["glowing_tiles"]


def test_two_lit_seats_or_an_unreadable_tile_stay_unconfirmed():
    pixels, regions = table()
    glow_around(pixels, [40, 96, 53, 41])
    glow_around(pixels, [212, 40, 37, 45])
    assert not highlighted_discard(pixels, regions, 0)["confirmed"]
    pixels, regions = table()
    glow_around(pixels, [40, 96, 53, 41])
    regions["river_left"]["items"][-1]["label"] = "?"
    result = highlighted_discard(pixels, regions, 0)
    assert not result["confirmed"] and result["unconfirmed_target"] == 3


def banner(pixels, x, colour):
    points = np.array([[x + 12, 60], [x + 172, 60], [x + 160, 126], [x, 126]], np.int32)
    cv2.fillPoly(pixels, [points], colour)


@pytest.mark.parametrize("colour,label", [((115, 135, 35), "chi"),      # H ~77, S ~190
                                          ((200, 140, 10), "pon"),      # H ~99, S ~242
                                          ((20, 25, 180), "win")])      # H ~1, S ~227
def test_reaction_banner_colour_gives_the_call(colour, label):
    pixels = np.zeros((145, 856, 3), np.uint8)
    pixels[:] = FELT
    assert call_banners(pixels) == []
    banner(pixels, 188, colour)
    found = call_banners(pixels)
    assert [item["label"] for item in found] == [label]
    assert found[0]["state"] == "enabled" and found[0]["evidence"] == "reaction_banner_colour"


def test_grey_skip_banner_and_dark_dress_are_not_calls():
    pixels = np.zeros((145, 856, 3), np.uint8)
    pixels[:] = FELT
    banner(pixels, 397, (120, 120, 118))       # 跳過 grey
    banner(pixels, 560, (103, 40, 15))         # dress, H ~112
    assert call_banners(pixels) == []


def mask_for(*kinds):
    mask = [False] * 46
    for kind in kinds:
        mask[{"chi": 38, "pon": 41, "kan": 42, "ron": 43, "pass": 45}[kind]] = True
    return mask


def observation(*labels):
    return {"regions": {"buttons": {"kind": "buttons", "items": [
        {"label": label, "evidence": "reaction_banner_colour", "banner_colour": label} for label in labels]}}}


def test_banners_must_match_the_rule_legal_calls():
    reaction = {"phase": "reaction"}
    check_reaction_banners(observation("pon"), reaction, {"rule_mask": mask_for("pon", "pass")})
    check_reaction_banners(observation("chi", "pon"), reaction, {"rule_mask": mask_for("chi", "pon", "pass")})
    check_reaction_banners(observation("call"), reaction, {"rule_mask": mask_for("kan", "pass")})
    with pytest.raises(SnapshotError, match="不一致"):
        check_reaction_banners(observation("pon"), reaction, {"rule_mask": mask_for("pon", "ron", "pass")})
    with pytest.raises(SnapshotError, match="不一致"):
        check_reaction_banners(observation("ron"), reaction, {"rule_mask": mask_for("pass")})
    # Outside a reaction, and without banner evidence, nothing is checked.
    check_reaction_banners(observation("pon"), {"phase": "draw"}, {"rule_mask": mask_for("pass")})
    check_reaction_banners({"regions": {}}, reaction, {"rule_mask": mask_for("ron", "pass")})


def reaction_observation(buttons):
    from conftest import load_sample
    start = load_sample('discard_red')[1]
    regions = {'hand': {'kind': 'hand', 'player': 0, 'complete': True,
                        'items': [{'label': t} for t in start['tehais'][0]]},
               'draw': {'kind': 'draw', 'player': 0, 'complete': True, 'items': []},
               'dora': {'kind': 'dora', 'player': None, 'complete': True, 'items': [{'label': start['dora_marker']}]},
               'buttons': {'kind': 'buttons', 'player': None, 'complete': True, 'items': [
                   {'label': label, 'clickable': True, 'evidence': 'reaction_banner_colour', 'banner_colour': label}
                   for label in buttons]}}
    for actor in range(4):
        regions[f'river_{actor}'] = {'kind': 'river', 'player': actor, 'complete': True,
                                     'items': [{'label': 'N', 'slot': [0, 0]}] if actor == 3 else []}
        regions[f'meld_{actor}'] = {'kind': 'meld', 'player': actor, 'complete': True, 'items': [], 'groups': []}
        regions[f'score_{actor}'] = {'kind': 'text', 'player': actor, 'field': 'score', 'complete': True,
                                     'text': {'value': 25000}}
    for field, value in [('round', {'bakaze': 'E', 'kyoku': 1}), ('remaining', 69), ('honba', 0), ('kyotaku', 0)]:
        regions[field] = {'kind': 'text', 'player': None, 'field': field, 'complete': True, 'text': {'value': value}}
    return {'regions': regions, 'self_wind': {'value': 'E'}, 'action_timer': {'active': True},
            'reaction_discard': {'confirmed': True, 'target': 3, 'tile': 'N'}}


def test_coloured_banners_set_the_reaction_and_its_allowed_actions():
    from app.state.snapshot import build_snapshot
    config = {'seat_mapping': {'self': 0, 'right': 1, 'top': 2, 'left': 3}}
    snapshot, allowed = build_snapshot(reaction_observation(['pon']), config, {}, {})
    assert (snapshot['phase'], snapshot['target'], snapshot['reaction_tile']) == ('reaction', 3, 'N')
    assert allowed == ['pass', 'pon']
    # An unmeasured banner colour leaves the choice to the rules engine,
    # checked against the banners afterwards.
    assert build_snapshot(reaction_observation(['call']), config, {}, {})[1] is None
    with pytest.raises(SnapshotError, match='按钮未能确认'):
        build_snapshot(reaction_observation([]), config, {}, {})


def test_outline_split_by_the_next_tile_still_lines_two_sides():
    """Right river: only the left and right edges of the outline are visible."""
    pixels, regions = table()
    x, y, w, h = 40, 96, 53, 41
    pixels[y - 2:y + h + 12, x - 9:x - 5] = GLOW
    pixels[y - 2:y + h + 12, x + w + 4:x + w + 8] = GLOW
    result = highlighted_discard(pixels, regions, 0)
    assert result["confirmed"] and result["glowing_tiles"][0]["glow_sides"] == ["left", "right"]


def test_detector_keeps_a_dense_green_face_with_a_ragged_light_mask():
    from app.vision.detector import tile_candidates
    region = {"kind": "river", "min_size": [15, 20], "max_size": [58, 68], "min_fill": .3,
              "grid": {"anchor": "center", "origin": [26, 42], "basis": [[28.8, -2.2], [0, 56]],
                       "rows": 4, "columns": 6, "tolerance": 13}}
    pixels = np.zeros((120, 160, 3), np.uint8)
    pixels[:] = FELT
    x, y, w, h = 50, 10, 27, 53
    pixels[y:y+h, x:x+w] = BAMBOO
    # Only a thin white rim survives along two edges plus the bottom-right corner,
    # as on a sideways 8s whose sticks reach the face edges.
    pixels[y:y+4, x:x+w] = (225, 228, 230)
    pixels[y:y+h, x:x+4] = (225, 228, 230)
    pixels[y+h-4:y+h, x:x+w] = (225, 228, 230)
    assert [x, y, w, h] in tile_candidates(pixels, region)["boxes"]


def test_tile_under_the_call_arrow_reuses_the_earlier_reading_of_that_slot():
    from app.vision.recognizer import reuse_marked_tiles
    def frame(arrow, face=(225, 228, 230)):
        pixels = np.zeros((768, 1366, 3), np.uint8)
        pixels[:] = FELT
        pixels[250:292, 460:514] = face
        cv2.circle(pixels, (475, 265), 6, (40, 40, 40), -1)
        if arrow:   # light-green call arrow lying over the face
            cv2.fillPoly(pixels, [np.array([[478, 262], [500, 262], [489, 276]], np.int32)], (120, 255, 230))
        return pixels
    def result(label, box):
        return {"regions": {"river_left": {"kind": "river", "complete": label != "?", "items": [
            {"label": label, "bbox": box, "slot": [1, 3], "score": .97}]}},
            "issues": [] if label != "?" else [{"region": "river_left", "reason": "incomplete_region"}]}
    memory = {}
    reuse_marked_tiles(frame(False), result("9p", [460, 250, 54, 42]), memory)
    later = result("?", [460, 250, 54, 28])         # side face now hidden by the next tile
    reuse_marked_tiles(frame(True), later, memory)
    region = later["regions"]["river_left"]
    assert region["items"][0]["label"] == "9p" and region["complete"] and not later["issues"]
    # A different tile in the same slot (next hand) is not overwritten.
    other = result("?", [460, 250, 54, 28])
    reuse_marked_tiles(frame(True, face=(60, 60, 200)), other, memory)
    assert other["regions"]["river_left"]["items"][0]["label"] == "?"


@pytest.mark.parametrize('mutation', ['none','no_arrow','different_proposal','one_different_vote',
                                      'no_verified_template','weak_proposal','colour_ambiguous_five','blank','too_occluded'])
def test_call_arrow_glyph_fallback_requires_all_evidence(mutation):
    from app.vision.recognizer import reuse_marked_tiles
    pixels = np.full((768,1366,3),FELT,np.uint8)
    box = [460,250,54,42]
    pixels[250:292,460:514] = 225
    if mutation != 'no_arrow':
        cv2.fillPoly(pixels,[np.array([[478,262],[500,262],[489,276]],np.int32)],(120,255,230))
    tile = '5p' if mutation == 'colour_ambiguous_five' else 'P' if mutation == 'blank' else '1p'
    item = {'label':'?','best_label':tile,'reason':'pjura_low_confidence','score':.68,'margin':.65,
            'bbox':box,'slot':[1,3],'horizontal':None,
            'legacy_evidence':{'label':tile,'score':.84,'margin':.28,'face_rotation':0,
                'acceptance_policy':'unanimous_train_svm_and_existing_template_proposal',
                'classifier_evidence':{'accepted':True,'votes':[tile]*3}}}
    if mutation == 'different_proposal': item['best_label'] = '2p'
    elif mutation == 'one_different_vote': item['legacy_evidence']['classifier_evidence']['votes'][-1] = '2p'
    elif mutation == 'no_verified_template': item['legacy_evidence'].pop('acceptance_policy')
    elif mutation == 'weak_proposal': item['margin'] = .1
    elif mutation == 'too_occluded': item['bbox'] = [478,262,22,14]
    result = {'regions':{'river_left':{'kind':'river','items':[item],'complete':False}},
              'issues':[{'region':'river_left','reason':'incomplete_region'}]}
    reuse_marked_tiles(pixels,result,{})
    assert item['label'] == ('1p' if mutation == 'none' else '?')
    assert result['regions']['river_left']['complete'] is (mutation == 'none')
    if mutation == 'none':
        assert item['occluded_reading']['score'] == .68
        assert item['score_kind'] == 'template_similarity_not_probability'


def test_blank_white_dragon_pon_takes_the_called_tile_from_its_shape():
    from app.vision.recognizer import group_melds
    items = [{"label": "P", "horizontal": False, "canonical_bbox": box}
             for box in ([514, 38, 63, 68], [572, 52, 73, 54], [632, 38, 65, 68])]
    region = {"pon_source_by_called_position": {"0": 3, "1": 2, "2": 1}}
    group = group_melds(items, region)[0]
    assert (group["type"], group["called_positions"], group["target"], group["status"]) == ("pon", [1], 2, "confirmed")
    assert group["called_position_evidence"] == "tile_aspect"
    # Two wide tiles: no guess.
    items[0]["canonical_bbox"] = [514, 52, 73, 54]
    assert group_melds(items, region)[0]["status"] == "needs_confirmation"


def test_round_text_read_by_both_readers_is_accepted():
    from app.vision.recognizer import text_agreement
    ocr = {"value": None, "variants": [{"text": "東1局", "score": .85}, {"text": "東1局", "score": .81},
                                       {"text": "“東商’", "score": .36}]}
    template = {"label": "?", "best_label": "東1局", "score": .978}
    assert text_agreement(ocr, template, "round") == {"bakaze": "E", "kyoku": 1}
    assert text_agreement(ocr, {**template, "best_label": "東2局"}, "round") is None
    # A real 1920x1080 frame measures OCR .81 with template .86 for 東1局; two
    # independent readers agreeing on the parsed value is the evidence, not the
    # per-reader score, so that pair is accepted and a weak template is not.
    assert text_agreement(ocr, {**template, "score": .861}, "round") == {"bakaze": "E", "kyoku": 1}
    assert text_agreement(ocr, {**template, "score": .7}, "round") is None


def test_call_arrow_points_at_the_discard_when_the_outline_is_faint():
    from app.vision.actions import call_arrows
    pixels = np.zeros((768, 1366, 3), np.uint8)
    pixels[:] = FELT
    regions = {"river_top": {"kind": "river", "player": 2, "items": []}}
    for box, label in (([761, 172, 37, 45], "1m"), ([646, 142, 35, 30], "8s")):
        x, y, w, h = box
        pixels[y:y+h, x:x+w] = (225, 228, 230)
        regions["river_top"]["items"].append({"label": label, "bbox": box})
    assert not highlighted_discard(pixels, regions, 0)["confirmed"]
    cv2.fillPoly(pixels, [np.array([[654, 123], [673, 123], [663, 134]], np.int32)], (150, 247, 160))
    assert len(call_arrows(pixels)) == 1
    result = highlighted_discard(pixels, regions, 0)
    assert result["confirmed"] and (result["target"], result["tile"]) == (2, "8s")
    assert result["candidates"][0]["arrow"] and not result["candidates"][0]["outlined"]


def test_call_arrow_cut_by_the_glyph_of_the_tile_under_it_is_still_found():
    """Left river: the arrow over the previous discard's black glyph (2026-10-15)."""
    from app.vision.actions import call_arrows
    pixels = np.zeros((768, 1366, 3), np.uint8)
    pixels[:] = FELT
    regions = {"river_left": {"kind": "river", "player": 3, "items": []}}
    for box, label in (([448, 207, 51, 26], "S"), ([443, 233, 54, 40], "9m")):
        x, y, w, h = box
        pixels[y:y+h, x:x+w] = (225, 228, 230)
        regions["river_left"]["items"].append({"label": label, "bbox": box, "slot": [1, len(regions["river_left"]["items"])]})
    cv2.fillPoly(pixels, [np.array([[459, 211], [476, 211], [467, 223]], np.int32)], (40, 255, 205))
    for x in (464, 470):                       # dark ink strokes showing through
        pixels[211:223, x:x+2] = (20, 20, 20)
    pixels[216:218, 459:476] = (20, 20, 20)
    arrows = call_arrows(pixels)
    assert len(arrows) == 1 and arrows[0]["bbox"][:2] == [459, 211]
    result = highlighted_discard(pixels, regions, 0)
    assert result["confirmed"] and (result["target"], result["tile"]) == (3, "9m")


def test_greyed_hand_tile_after_a_call_is_still_in_the_hand():
    from app.vision.hand import hand_layout
    slots = [[5 + 67 * i, 11, 67, 108] for i in range(13)]
    region = {"slots": slots, "body_presence_bands": [[.08, .15], [.91, .98]],
              "hand_layout": {"draw_region": "draw", "draw_gap": 22, "concealed_counts": [1, 4, 7, 10, 13],
                              "gap_fill_max": .25}}
    pixels = np.zeros((124, 987, 3), np.uint8)
    pixels[:] = (90, 40, 20)
    for index in range(8):
        x, y, w, h = slots[index]
        pixels[y:y+h, x+1:x+w-1] = (134, 134, 134) if index == 1 else (223, 223, 223)
    layout = hand_layout(pixels, region)
    assert len(layout["hand"]) == 8 and layout["dimmed"] == [slots[1]] and not layout["reasons"]


def test_after_two_calls_the_greyed_tiles_name_the_call_just_made():
    """Chi 3p (2p4p) after an earlier pon 1p: the game greys the 3p in hand."""
    from app.state.snapshot import build_snapshot
    config = {'seat_mapping': {'self': 0, 'right': 1, 'top': 2, 'left': 3}}
    observation = reaction_observation([])
    hand = ['2p', '3p', '6p', '8p', '9p', 'E', 'N', 'N']
    observation['regions']['hand']['items'] = [{'label': t, 'dimmed': t == '3p'} for t in hand]
    observation['regions']['meld_0']['groups'] = [
        {'labels': ['3p', '2p', '4p'], 'type': 'chi', 'target': 3, 'called_positions': [0], 'status':'confirmed'},
        {'labels': ['1p', '1p', '1p'], 'type': 'pon', 'target': 2, 'called_positions': [1], 'status':'confirmed'}]
    from app.state.events import TILES
    def legal_ids(snapshot):
        # 食替: the called tile may not be discarded right after the call.
        banned = snapshot['last_call']['pai']
        return {TILES.index(t) for t in snapshot['hand'] if t != banned}
    snapshot, _ = build_snapshot(observation, config, {}, {}, legal_ids=legal_ids)
    assert snapshot['phase'] == 'call_discard'
    assert (snapshot['last_call']['type'], snapshot['last_call']['pai']) == ('chi', '3p')
    # Nothing greyed: the pon (whose 1p is not in hand) is the call just made.
    for item in observation['regions']['hand']['items']:
        item['dimmed'] = False
    snapshot, _ = build_snapshot(observation, config, {}, {}, legal_ids=legal_ids)
    assert snapshot['last_call']['type'] == 'pon'
    # Without the rules engine the frame cannot tell: ask.
    with pytest.raises(SnapshotError, match='确认刚完成的吃/碰'):
        build_snapshot(observation, config, {}, {})


def test_red_dot_island_inside_a_sideways_meld_tile_is_merged():
    from app.vision.detector import merge_river_fragments
    boxes, merged = merge_river_fragments([[1117, 687, 77, 54], [1142, 687, 27, 21], [1182, 673, 63, 68]],
                                          {"kind": "meld", "max_size": [90, 120]})
    assert sorted(boxes) == [[1117, 687, 77, 54], [1182, 673, 63, 68]] and len(merged) == 1


def test_orange_riichi_banner_is_not_a_win():
    pixels = np.zeros((145, 856, 3), np.uint8)
    pixels[:] = FELT
    banner(pixels, 188, (20, 110, 230))       # orange: H ~12, S ~233
    assert [item["label"] for item in call_banners(pixels)] == ["reach"]
    pixels[:] = FELT
    banner(pixels, 188, (20, 25, 180))        # red 和: H ~1
    assert [item["label"] for item in call_banners(pixels)] == ["win"]


def test_one_unsure_score_is_settled_by_the_point_total():
    """A riichi stick beside the right score: OCR "24000" at 0.89, just under 0.9."""
    from app.state.snapshot import build_snapshot
    config = {'seat_mapping': {'self': 0, 'right': 1, 'top': 2, 'left': 3}}
    observation = reaction_observation(['pon'])
    values = {0: 32700, 1: None, 2: 25000, 3: 17300}
    for actor, value in values.items():
        observation['regions'][f'score_{actor}'] = {
            'kind': 'text', 'player': actor, 'field': 'score', 'complete': value is not None,
            'text': {'value': value, 'variants': [{'text': '24000', 'score': .89}, {'text': '2400', 'score': .82}]}}
    observation['regions']['kyotaku'] = {'kind': 'text', 'player': None, 'field': 'kyotaku', 'complete': True,
                                         'text': {'value': 1}}
    snapshot, _ = build_snapshot(observation, config, {}, {})
    assert snapshot['scores'] == [32700, 24000, 25000, 17300]
    # Without the deposit the total does not close: the score stays for review.
    observation['regions']['kyotaku']['text']['value'] = 0
    with pytest.raises(SnapshotError, match='score_1'):
        build_snapshot(observation, config, {}, {})


def test_one_unsure_tile_completing_a_meld_takes_its_best_reading():
    """Right player's chi 1s-2s-3s: the sideways bird read 1s at 0.74 (next 0.02)."""
    from app.vision.recognizer import group_melds
    items = [{"label": "3s", "horizontal": True, "canonical_bbox": [231, 53, 41, 42]},
             {"label": "?", "best_label": "1s", "score": .74, "margin": .72, "reason": "pjura_low_confidence",
              "horizontal": None, "canonical_bbox": [259, 36, 34, 50]},
             {"label": "2s", "horizontal": False, "canonical_bbox": [286, 31, 26, 48]}]
    region = {"chi_source_by_called_position": {"0": 0}}
    group = group_melds([dict(item) for item in items], region)[0]
    assert (group["labels"], group["type"], group["target"], group["status"]) == (["3s", "1s", "2s"], "chi", 0, "confirmed")
    # A reading that does not complete a meld, or a weak one, stays unknown.
    for change in ({"best_label": "5s"}, {"score": .4}, {"margin": .1}):
        trial = [dict(item) for item in items]
        trial[1].update(change)
        assert group_melds(trial, region)[0]["labels"][1] == "?"


def test_white_riichi_sticks_mark_the_seats_that_declared():
    from app.vision.actions import riichi_sticks
    seats = {'self': 0, 'right': 1, 'top': 2, 'left': 3}
    pixels = np.zeros((768, 1366, 3), np.uint8)
    pixels[:] = (70, 60, 55)
    assert not any(riichi_sticks(pixels, seats).values())
    pixels[266:340, 775:791] = (235, 235, 235)              # right player's stick
    cv2.circle(pixels, (783, 303), 4, (40, 40, 220), -1)     # its red dot
    found = riichi_sticks(pixels, seats)
    assert found[1] and not found[0] and not found[2] and not found[3]
    pixels[392:395, 600:646] = (235, 235, 235)              # a thin tile edge is not a stick
    assert not riichi_sticks(pixels, seats)[0]


@pytest.mark.parametrize("shape",[(128,256),(280,790)])
def test_riichi_stick_zones_outside_the_frame_do_not_crash(shape):
    from app.vision.actions import riichi_sticks
    pixels = np.full((*shape,3),235,np.uint8)
    found = riichi_sticks(pixels,{'self':0,'right':1,'top':2,'left':3})
    assert found[0] is None and found[1] is None


def test_sideways_riichi_tile_is_found_from_its_row():
    from app.vision.recognizer import mark_sideways_by_shape
    grid = {"basis": [[44.5, 0], [-3, 49]]}
    row = [{"label": t, "horizontal": False, "slot": [1, i], "canonical_bbox": [0, 0, w, h]}
           for i, (t, w, h) in enumerate([("9m", 44, 57), ("8s", 43, 57), ("1m", 42, 57), ("9s", 44, 57),
                                          ("5s", 57, 45), ("N", 45, 57)])]
    row[4]["horizontal"] = None
    mark_sideways_by_shape(row, grid)
    assert [item["horizontal"] for item in row] == [False, False, False, False, True, False]
    # The near tile closing a side-river row is wider but also taller: not sideways.
    side = [{"label": "x", "horizontal": False, "slot": [0, i], "canonical_bbox": [0, 0, w, h]}
            for i, (w, h) in enumerate([(26, 51), (27, 53), (28, 54), (30, 55), (43, 56)])]
    mark_sideways_by_shape(side, {"basis": [[30, 2.5], [0, 53]]})
    assert not any(item["horizontal"] for item in side)
    # A riichi tile alone at the start of a row is judged against the lattice pitch.
    alone = [{"label": "5m", "horizontal": None, "slot": [2, 0], "canonical_bbox": [0, 0, 59, 47]}]
    mark_sideways_by_shape(alone, grid)
    assert alone[0]["horizontal"] is True


def test_a_riichi_stick_alone_sets_that_players_riichi():
    from app.state.snapshot import build_snapshot
    config = {'seat_mapping': {'self': 0, 'right': 1, 'top': 2, 'left': 3}}
    observation = reaction_observation(['pon'])
    observation['riichi_sticks'] = {'1': [775, 266, 16, 74], '2': None}
    snapshot, _ = build_snapshot(observation, config, {}, {})
    assert snapshot['riichi'] == [False, True, False, False]
