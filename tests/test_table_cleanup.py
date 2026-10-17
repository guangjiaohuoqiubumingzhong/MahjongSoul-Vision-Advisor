"""Merged river tiles are split at the lattice pitch; table art is not a tile."""
import numpy as np

from app.vision.detector import merge_river_fragments, split_by_grid, tile_candidates
from app.vision.recognizer import is_background

LEFT_GRID = {"anchor": "center", "origin": [15.5, 30], "basis": [[30, 2.5], [0, 53]],
             "rows": 4, "columns": 6, "tolerance": 13}
RIVER = {"kind": "river", "grid": LEFT_GRID, "min_size": [15, 20], "max_size": [58, 68], "min_fill": .3,
         "body_saturation_max": 40}


def test_single_upright_and_sideways_tiles_are_not_split():
    assert split_by_grid([3, 7, 25, 53], LEFT_GRID) == [[3, 7, 25, 53]]
    # A riichi tile lies sideways: wide but short, still one tile.
    assert split_by_grid([3, 7, 46, 30], LEFT_GRID) == [[3, 7, 46, 30]]
    assert split_by_grid([3, 7, 43, 57], LEFT_GRID) == [[3, 7, 43, 57]]


def test_two_touching_upright_tiles_are_split_at_the_column_pitch():
    pieces = split_by_grid([110, 14, 56, 54], LEFT_GRID)
    assert len(pieces) == 2 and all(abs(w - 28) <= 1 and h == 54 for _, _, w, h in pieces)


def test_two_touching_rows_are_split_at_the_row_pitch():
    pieces = split_by_grid([3, 7, 26, 106], LEFT_GRID)
    assert len(pieces) == 2 and all(h == 53 for *_, h in pieces)


def test_detector_returns_one_box_per_touching_tile():
    river = np.full((140, 200, 3), (120, 70, 40), np.uint8)
    river[10:63, 20:45] = 235
    river[10:63, 45:72] = 235          # touches the first tile: one light blob
    boxes = tile_candidates(river, RIVER)["boxes"]
    assert len(boxes) == 2


def test_disconnected_face_islands_count_as_one_tile_and_keep_the_neighbour():
    river = np.full((140, 200, 3), (120, 70, 40), np.uint8)
    river[20:76, 19:48] = 235  # neighbouring upright tile
    river[20:76, 50:94] = 235
    river[30:68, 50:70] = 0   # dense glyph cuts a white island off the face
    river[32:66, 49:65] = 235 # island extends one pixel past the main contour
    river[90:120, 7:53] = 235 # a sideways riichi tile stays independent
    detected = tile_candidates(river, RIVER)
    assert detected["boxes"] == [[19, 20, 29, 56], [49, 20, 45, 56], [7, 90, 46, 30]]
    assert len(detected["merged_fragments"]) == 1
    assert not detected["unresolved"]


def test_similarly_sized_overlapping_bodies_remain_ambiguous():
    # Two plausible tile bodies must not silently collapse into one.
    boxes = [[20, 10, 30, 54], [25, 12, 28, 50]]
    bodies, fragments = merge_river_fragments(boxes, RIVER)
    assert bodies == boxes and fragments == []


def test_fragment_merge_cannot_create_an_oversized_body():
    boxes = [[20, 10, 58, 68], [19, 20, 16, 38]]
    bodies, fragments = merge_river_fragments(boxes, RIVER)
    assert bodies == boxes and fragments == []


def item(**values):
    base = {"label": "?", "reason": "pjura_low_confidence", "background_candidate": True,
            "pjura_evidence": {"face_fitted": False}}
    return {**base, **values}


def test_table_art_is_background_but_an_unreadable_tile_on_a_slot_is_not():
    assert is_background(item(), RIVER)
    assert is_background(item(reason="river_slot_ambiguous", slot=[0, 9],
                              pjura_evidence={"face_fitted": True}), RIVER)
    # A real face on a valid slot stays an unknown tile for review.
    assert not is_background(item(slot=[1, 2], pjura_evidence={"face_fitted": True}), RIVER)
    assert not is_background(item(background_candidate=False), RIVER)
    assert not is_background(item(), {"kind": "hand"})


TOP_GRID = {"anchor": "center", "origin": [23, 25.5], "basis": [[38, 0], [1.5, 35]],
            "rows": 4, "columns": 6, "tolerance": 13}


def test_a_riichi_tile_and_the_tiles_after_it_keep_their_slots():
    from app.vision.detector import river_slots
    # Row 2: an upright tile, a sideways riichi tile 53 wide, then two uprights
    # pushed along by the riichi tile's extra width.
    boxes = [[0, 69, 41, 31], [40, 64, 53, 35], [94, 69, 36, 31], [132, 69, 36, 31]]
    slots = river_slots(boxes, TOP_GRID)
    assert [slot for slot, _ in slots] == [[2, 0], [2, 1], [2, 2], [2, 3]]
    assert all(residual <= TOP_GRID["tolerance"] for _, residual in slots)


def test_rows_without_riichi_are_unchanged():
    from app.vision.detector import river_slot, river_slots
    boxes = [[4, 10, 36, 30], [42, 10, 36, 30]]
    assert river_slots(boxes, TOP_GRID) == [river_slot(box, TOP_GRID) for box in boxes]


RIGHT_GRID = {"anchor": "center", "origin": [26, 42], "basis": [[28.8, -2.2], [0, 56]],
              "rows": 4, "columns": 6, "tolerance": 13}


def test_a_perspective_widened_upright_tile_is_not_treated_as_riichi():
    from app.vision.detector import river_slot, river_slots
    # Measured on a 1920x1080 frame: each side-river row starts with a wide but
    # tall upright tile; slots must stay exactly as the plain lattice gives them.
    boxes = [[4, 14, 43, 56], [48, 12, 30, 54], [78, 11, 28, 54], [4, 73, 43, 57], [39, 67, 38, 56],
             [77, 66, 28, 54], [106, 62, 27, 53]]
    assert river_slots(boxes, RIGHT_GRID) == [river_slot(box, RIGHT_GRID) for box in boxes]


def test_self_river_riichi_body_is_almost_square_but_orientation_confirms_row_shift():
    from app.vision.detector import river_slots
    grid = {"anchor": "center", "origin": [49, 28], "basis": [[44.5, 0], [-3, 49]],
            "rows": 4, "columns": 6, "tolerance": 13}
    boxes = [[161, 50, 42, 56], [204, 56, 57, 47], [263, 49, 44, 57]]
    slots = river_slots(boxes, grid, [False, True, False])
    assert [slot for slot, _ in slots] == [[1, 3], [1, 4], [1, 5]]
    assert all(residual <= 13 for _, residual in slots)


def test_self_river_riichi_without_orientation_evidence_still_shifts_the_row():
    from app.vision.detector import river_slots
    # Measured on a live 1920x1080 frame after the player's riichi; the
    # classifier could not confirm orientation, so geometry decides.
    grid = {"anchor": "center", "origin": [29, 28], "basis": [[44.5, 0], [-3, 49]],
            "rows": 4, "columns": 6, "tolerance": 13}
    row = [[5, 51, 44, 45], [50, 49, 44, 57], [96, 51, 42, 57], [141, 50, 42, 56],
           [184, 56, 57, 47], [243, 49, 44, 57]]
    slots = river_slots(row, grid)
    assert [slot for slot, _ in slots] == [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    assert all(residual <= 13 for _, residual in slots)


def test_a_small_faceless_candidate_on_a_slot_is_background():
    # A 26x19 felt star scored 0.74 as a white dragon and landed on a slot.
    star = item(background_candidate=False, bbox=[951, 253, 26, 19], slot=[2, 4],
                pjura_evidence={"face_fitted": False})
    assert is_background(star, RIVER)
    real = item(background_candidate=False, bbox=[951, 253, 27, 52], slot=[2, 4],
                pjura_evidence={"face_fitted": False})
    assert not is_background(real, RIVER)
