from __future__ import annotations

import numpy as np

from .detector import light_mask, tile_boxes


DIMMED_VALUE_MIN = 95


def hand_layout(pixels: np.ndarray, region: dict) -> dict:
    """Separate a drawn tile using the visible gap, without inferring meld history.

    The calibrated regular slots keep their physical indices. A drawn tile can
    follow 13, 10, 7, 4 or 1 concealed tiles; a bright patch in a regular slot is
    insufficient evidence when the upper/lower tile-body bands are absent.
    """
    layout = region["hand_layout"]
    slots = region["slots"]
    boxes = tile_boxes(pixels, region)
    # After a chi/pon the game greys out the tiles kuikae forbids discarding
    # (measured body value ~134 against ~223). They are still in the hand.
    dimmed = [box for box in tile_boxes(pixels, {**region, "body_value_min": DIMMED_VALUE_MIN})
              if box not in boxes]
    boxes = sorted(boxes + dimmed, key=slots.index)
    indices = [slots.index(box) for box in boxes]
    mask = light_mask(pixels, region.get("body_saturation_max", 100))
    gap = layout["draw_gap"]
    candidates = []
    for count in layout.get("concealed_counts", [1, 4, 7, 10, 13]):
        if indices[:count] != list(range(count)):
            continue
        last_x, y, last_w, h = slots[count-1]
        edge = last_x + last_w
        draw_box = [edge+gap, y, last_w, h]
        # Exclude the regular tile's border and the drawn tile's rounded corner.
        space = mask[y+10:y+h-10, edge+3:edge+gap-3]
        if not space.size or np.mean(space > 0) > layout.get("gap_fill_max", .1):
            continue
        draw_region = {**region, "slots": [draw_box]}
        if not tile_boxes(pixels, draw_region):
            continue
        candidates.append((count, draw_box))
    reasons = []
    draw = []
    if len(candidates) == 1:
        count, draw_box = candidates[0]
        # The shifted draw overlaps slot `count`; later full-size bodies are a
        # conflict, not extra concealed tiles to silently discard.
        if any(index > count for index in indices):
            reasons.append("unexpected_tiles_after_draw")
        boxes = slots[:count]
        indices = list(range(count))
        draw = [draw_box]
    elif candidates:
        reasons.append("ambiguous_draw_gap")
    elif indices != list(range(len(indices))):
        reasons.append("noncontiguous_concealed_hand")
    elif len(indices) not in (*layout.get("concealed_counts", [1,4,7,10,13]), 2,5,8,11):
        reasons.append("concealed_count_or_draw_position_unconfirmed")
    elif len(indices) in (2,5,8,11):
        # After a call there is one extra regular tile and no separate draw.
        # A shifted drawn tile with its gap obscured can imitate that count;
        # its body extending past the last regular slot must still be rejected.
        bx,by,bw,bh = slots[len(indices)-1]
        edge = bx+bw
        space = mask[by+10:by+bh-10,edge+3:edge+gap-3]
        possible_draw = [edge+gap,by,bw,bh]
        if ((space.size and np.mean(space > 0) > layout.get("gap_fill_max", .1))
                or tile_boxes(pixels,{**region,"slots":[possible_draw]})):
            reasons.append("concealed_count_or_draw_position_unconfirmed")
    elif indices:
        # A visible body beside a full regular hand must not be reported as an
        # empty draw region just because the gap was occluded or too bright.
        bx,by,bw,bh = slots[len(indices)-1]
        possible_draw = [bx+bw+gap,by,bw,bh]
        if tile_boxes(pixels,{**region,"slots":[possible_draw]}):
            reasons.append("draw_separation_unconfirmed")
    dimmed = [box for box in dimmed if box in boxes]
    return {"hand": boxes, "draw": draw, "slot_indices": indices, "dimmed": dimmed,
            "reasons": reasons, "evidence": "calibrated_tile_body_bands_and_visible_gap"}
