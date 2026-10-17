"""Current action controls and last-discard evidence from visible pixels."""
import re

import cv2
import numpy as np


ACTION_TEXT = {"吃": "chi", "碰": "pon", "杠": "kan", "槓": "kan",
               "和": "win", "荣和": "ron", "榮和": "ron", "自摸": "tsumo",
               "立直": "reach"}


def enabled_action_buttons(pixels, reader, threshold=.94):
    """Read the coloured enabled glyphs; grey disabled labels are excluded.

    The ornate controls cannot be read as a whole strip: the timer and table
    art get merged into their text. Isolate the green/gold glyphs first, then
    require an exact action name and a confident OCR result.
    """
    if reader is None:
        return []
    height, width = pixels.shape[:2]
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    active = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 90)
              & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 145))
    active[:round(height * .24)] = False
    active[round(height * .9):] = False
    joined = cv2.morphologyEx(active.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                             np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = {}
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (28 <= h <= 85 and 20 <= w <= 185):
            continue
        if np.count_nonzero(active[y:y+h, x:x+w]) < 180:
            continue
        left, top = max(0, x-7), max(0, y-7)
        right, bottom = min(width, x+w+7), min(height, y+h+7)
        text, score = reader._infer(pixels[top:bottom, left:right])
        label = ACTION_TEXT.get(re.sub(r"\s+", "", text))
        if label is None or score < threshold:
            continue
        match = {"label": label, "state": "enabled", "clickable": True,
                 "visible": True, "bbox": [left, top, right-left, bottom-top],
                 "score": float(score), "raw_text": text,
                 "evidence": "coloured_enabled_glyph_and_exact_ocr"}
        if label not in found or score > found[label]["score"]:
            found[label] = match
    return list(found.values())


# Reaction buttons are drawn on coloured banners next to the grey "跳過".
# Measured at 1366x768 (OpenCV HSV): 吃 green (H 72-85, S 135-195), 碰 teal
# (H 97-104, S 178-235), 和 red (H 0-12, S 176-186). The felt (H ~101,
# S ~130) and the character's dress (H 111-113) fall outside these ranges.
# The calligraphic glyphs read poorly with OCR (碰 is often read as 碟), so
# the colour is the evidence; an unmeasured colour (e.g. 杠) stays a generic
# "call". app.advice checks every banner against the rules engine's legal
# actions before advice is shown.
BANNER_COLOURS = {
    "chi": ((65, 90), 120, 70),            # label: (hue range, min saturation, min value)
    "pon": ((92, 106), 170, 80),
    "win": ((172, 18), 170, 110),          # wraps around red (立直 is orange, H ~12); the pink torii is H 158-170
    "call": ((125, 155), 150, 80),
}
BANNER_MIN_AREA = 1500
REACH_MIN_HUE = 7
BANNER_SIZE = ((100, 260), (35, 100))      # (width range, height range) at 1366x768
# Banners sit about 206 px apart, leftwards from 跳過; a cut one at the crop
# edge is still at least 100 px wide.


def call_banners(pixels, reader=None):
    """Enabled reaction banners in a buttons crop, labelled by banner colour."""
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    found = []
    for label, ((low, high), min_saturation, min_value) in BANNER_COLOURS.items():
        hues = (hue >= low) & (hue <= high) if low <= high else (hue >= low) | (hue <= high)
        mask = hues & (saturation >= min_saturation) & (value >= min_value)
        joined = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined)
        for x, y, w, h, area in (stats[i] for i in range(1, count)):
            (wl, wh), (hl, hh) = BANNER_SIZE
            if area < BANNER_MIN_AREA or not (wl <= w <= wh and hl <= h <= hh) or area < .3 * w * h:
                continue
            x, y, w, h = int(x), int(y), int(w), int(h)
            item = {"label": label, "state": "enabled", "visible": True, "bbox": [x, y, w, h],
                    "banner_colour": label, "score": None, "evidence": "reaction_banner_colour"}
            if reader is not None:
                top, bottom = max(0, int(y - .15 * h)), min(pixels.shape[0], int(y + .9 * h))
                glyph = pixels[top:bottom, int(x + .2 * w):int(x + .8 * w)]
                if glyph.size:
                    text, score = reader._infer(glyph)
                    item["ocr_hint"] = {"text": text, "score": float(score)}
            if label == "win":
                # 立直 shares the red family but is orange: measured median hue
                # 12 against 0 for 和. Its glyphs also read well ("立直" 0.74).
                piece = mask[y:y+h, x:x+w]
                wrapped = hue[y:y+h, x:x+w][piece].astype(int)
                wrapped = np.where(wrapped >= 90, wrapped - 180, wrapped)
                item["banner_hue"] = float(np.median(wrapped)) if wrapped.size else None
                text = (item.get("ocr_hint") or {}).get("text", "")
                if ("立" in text or "直" in text) or (
                        "和" not in text and item["banner_hue"] is not None and item["banner_hue"] >= REACH_MIN_HUE):
                    item["label"] = item["banner_colour"] = "reach"
            found.append(item)
    return sorted(found, key=lambda value: value["bbox"][0])


# The game outlines the discard that is waiting for a reaction (chi/pon/kan/
# ron) with a green glow that pulses. Measured on 1920x1080 frames reduced to
# 1366x768: hue 53-93 (OpenCV), saturation 50-155, value 148-238, two to six
# pixels wide and partly hidden under the neighbouring tile and its shadow.
GLOW_HUE = (35, 95)
GLOW_MIN_SATURATION = 45
GLOW_MIN_VALUE = 120
GLOW_MARGIN = 16
# The outline runs along the tile's sides, but the neighbouring tile and the
# shadow hide one or two of them, and the hidden part can split the outline
# into separate pieces (right river: only the left and right edges show).
# A side counts when one connected glow piece next to it runs along at least
# 70 % of that side. Green bamboo ink leaking out of a neighbour's face only
# forms short strokes (at most 21x10 px measured) and never lines a side.
GLOW_SIDE_COVER = .7
GLOW_MIN_SIDES = 2
GLOW_MIN_DENSITY = .5                      # lining pixels per pixel of tile perimeter
GLOW_MIN_PIXELS = 40
GLOW_UNIQUE_RATIO = .3
# Diagnostic only: the wider rim band also picks up a neighbouring green
# bamboo face, so it is reported but never used to confirm a source.
GLOW_BAND_MARGIN = 5
GLOW_INNER_RIM = 2
GLOW_PERIMETER_MIN = .30
# The green triangle the game floats above the discard waiting for a call,
# measured on real 1920x1080 frames reduced to 1366x768: about 45x28 px, filled
# from the wide top edge down to a rounded tip, so it is widest at the top and
# narrow along most of its rows.
ARROW_SIZE = ((28, 90), (16, 60))
ARROW_MIN_AREA = 250
ARROW_FILL = (.3, .95)
ARROW_NARROW_ROWS = 6
ARROW_ZONE_MARGIN = 26

def glow_mask(pixels):
    """Green/cyan glow, excluding the blue felt, white faces and red/black ink."""
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    blue, green, red = (pixels[:, :, i].astype(np.int16) for i in range(3))
    return ((hsv[:, :, 0] >= GLOW_HUE[0]) & (hsv[:, :, 0] <= GLOW_HUE[1])
            & (hsv[:, :, 1] >= GLOW_MIN_SATURATION) & (hsv[:, :, 2] >= GLOW_MIN_VALUE)
            & (green >= red + 25) & (green >= blue - 10))


def tile_glow(mask, bbox, others=()):
    """Glow lining the sides of ``bbox``, outside every river tile box.

    Returns ``(pixels, density, sides)``: the pixels of the pieces that line a
    side, per pixel of tile perimeter, and which sides ("left", "right",
    "top", "bottom") are lined.
    """
    x, y, w, h = bbox
    left, top = max(0, x - GLOW_MARGIN), max(0, y - GLOW_MARGIN)
    right, bottom = min(mask.shape[1], x + w + GLOW_MARGIN), min(mask.shape[0], y + h + GLOW_MARGIN)
    band = mask[top:bottom, left:right].copy()
    for ox, oy, ow, oh in (bbox, *others):
        band[max(0, oy - top):max(0, oy + oh - top), max(0, ox - left):max(0, ox + ow - left)] = False
    count, labels = cv2.connectedComponents(band.astype(np.uint8), connectivity=8)
    bx, by = x - left, y - top
    strips = {"left": (slice(None), slice(0, max(0, bx)), 0, h),
              "right": (slice(None), slice(bx + w, None), 0, h),
              "top": (slice(0, max(0, by)), slice(None), 1, w),
              "bottom": (slice(by + h, None), slice(None), 1, w)}
    sides, pixels = set(), 0
    for label in range(1, count):
        piece = labels == label
        lined = set()
        for side, (rows, cols, axis, length) in strips.items():
            ys, xs = np.nonzero(piece[rows, cols])
            along = ys if axis == 0 else xs
            if along.size and np.ptp(along) + 1 >= GLOW_SIDE_COVER * length:
                lined.add(side)
        if lined:
            sides |= lined
            pixels += int(piece.sum())
    return pixels, pixels / max(1, 2 * (w + h)), sorted(sides)


def glow_halo(mask, bbox, others=()):
    """Glow hugging the rim of the whole ``bbox``, just inside and outside it.

    The call highlight is drawn on the tile it marks: a thin outline around the
    tile, partly hidden by the neighbouring tile and its shadow. It is measured
    in a band a few pixels thick along the tile's own edges, on both sides of
    them, because the marked tile's *face* is also brightened. Glow that sits
    deeper inside the box is the tile's own green bamboo ink, not the outline,
    so it is ignored.

    Returns ``(pixels, coverage, sides)``: glowing pixels in the band, the
    fraction of the tile perimeter they cover, and the lined sides.
    """
    x, y, w, h = bbox
    if w <= 2 * GLOW_INNER_RIM or h <= 2 * GLOW_INNER_RIM:
        return 0, 0.0, []
    left, top = max(0, x - GLOW_BAND_MARGIN), max(0, y - GLOW_BAND_MARGIN)
    right = min(mask.shape[1], x + w + GLOW_BAND_MARGIN)
    bottom = min(mask.shape[0], y + h + GLOW_BAND_MARGIN)
    band = mask[top:bottom, left:right].copy()
    # Keep only the rim: outside the box, or within GLOW_INNER_RIM inside it.
    bx, by = x - left, y - top
    inner = np.zeros_like(band)
    inner[max(0, by + GLOW_INNER_RIM):by + h - GLOW_INNER_RIM,
          max(0, bx + GLOW_INNER_RIM):bx + w - GLOW_INNER_RIM] = True
    band &= ~inner
    for ox, oy, ow, oh in others:
        band[max(0, oy - top):max(0, oy + oh - top), max(0, ox - left):max(0, ox + ow - left)] = False
    strips = {"left": (slice(None), slice(max(0, bx - GLOW_BAND_MARGIN), bx + GLOW_INNER_RIM), h),
              "right": (slice(None), slice(bx + w - GLOW_INNER_RIM, bx + w + GLOW_BAND_MARGIN), h),
              "top": (slice(max(0, by - GLOW_BAND_MARGIN), by + GLOW_INNER_RIM), slice(bx, bx + w), w),
              "bottom": (slice(by + h - GLOW_INNER_RIM, by + h + GLOW_BAND_MARGIN), slice(bx, bx + w), w)}
    covered = {}
    for side, (rows, cols, length) in strips.items():
        if length <= 0 or band[rows, cols].size == 0:
            continue
        covered[side] = float(band[rows, cols].sum()) / length
    sides = sorted(side for side, fraction in covered.items() if fraction >= GLOW_SIDE_COVER)
    perimeter = 2.0 * (w + h)
    coverage = sum(covered.get(side, 0.0) * length for side, length in
                   (("left", h), ("right", h), ("top", w), ("bottom", w))) / max(1.0, perimeter)
    return int(band.sum()), coverage, sides


def call_arrow(mask):
    """The triangle the game floats above the discard waiting for a call.

    It is drawn above the tile, pointing down at it, in the same pale green as
    the outline. Returns the component's ``bbox`` and its bottom tip, or
    ``None``. The triangle is much wider than each single scanline of its tip,
    which separates it from bamboo strokes and from tile faces.
    """
    joined = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    best = None
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if not (ARROW_SIZE[0][0] <= w <= ARROW_SIZE[0][1] and ARROW_SIZE[1][0] <= h <= ARROW_SIZE[1][1]):
            continue
        if area < ARROW_MIN_AREA or not ARROW_FILL[0] <= area / (w * h) <= ARROW_FILL[1]:
            continue
        piece = labels[y:y + h, x:x + w] == index
        widths = piece.sum(axis=1)
        narrow = int(np.count_nonzero(widths >= 2 * max(1.0, widths.max() / 3.0)))
        if widths[0] < 2 * max(1, widths[-1]) or narrow > ARROW_NARROW_ROWS:
            continue
        candidate = {"bbox": [x, y, w, h], "area": area, "tip": [x + w // 2, y + h - 1]}
        if best is None or area > best["area"]:
            best = candidate
    return best


def arrow_target(arrow, regions, player):
    """Which seat's river the arrow points into, from the river zones only.

    The green triangle floats above the discard waiting for a call, so it
    identifies the seat even when the pulsing outline is hidden. The geometry
    differs between captures (the outline and the arrow are drawn at the game's
    own scale), so this is used as evidence for review, never as a silent
    override of the outline.
    """
    if not arrow:
        return None
    tip_x, tip_y = arrow["tip"]
    scored = []
    for name, region in regions.items():
        if region["kind"] != "river":
            continue
        bbox = region.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox
        inside = (x - ARROW_ZONE_MARGIN <= tip_x <= x + w + ARROW_ZONE_MARGIN
                  and y - ARROW_ZONE_MARGIN <= tip_y <= y + h + ARROW_ZONE_MARGIN)
        if not inside:
            continue
        seat = region.get("player")
        if seat is None or seat == player:
            continue
        distance = abs(tip_x - (x + w / 2)) + abs(tip_y - (y + h / 2))
        scored.append((distance, name, seat))
    if not scored:
        return None
    scored.sort()
    if len(scored) > 1 and scored[1][0] < .5 * scored[0][0]:
        return None
    return {"target": scored[0][2], "region": scored[0][1],
            "evidence": "call_arrow_above_last_discard", "bbox": arrow["bbox"], "tip": arrow["tip"]}


# A downward arrow also bobs just above the discard waiting for a call. It is
# light green (H 59-67, V 206-255), turning yellowish (H 22-40) where it lies
# over a white tile, and a solid triangle 10-32 x 6-22 px, wider at the top.
# Like the outline it pulses, so it adds evidence rather than replacing it.
CALL_ARROW_HUE = (20, 72)
CALL_ARROW_MIN_SATURATION = 40
CALL_ARROW_MIN_VALUE = 200
CALL_ARROW_SIZE = ((10, 32), (6, 22))
CALL_ARROW_TABLE = (330, 1060, 60, 480)         # x0, x1, y0, y1 of the rivers at 1366x768
CALL_ARROW_BELOW = (-4, 16)                      # target tile top relative to the arrow tip


def call_arrows(pixels):
    """Downward call arrows over the table: list of {bbox, tip}."""
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= CALL_ARROW_HUE[0]) & (hsv[:, :, 0] <= CALL_ARROW_HUE[1])
            & (hsv[:, :, 1] >= CALL_ARROW_MIN_SATURATION) & (hsv[:, :, 2] >= CALL_ARROW_MIN_VALUE))
    x0, x1, y0, y1 = CALL_ARROW_TABLE
    mask[:y0], mask[y1:], mask[:, :x0], mask[:, x1:] = False, False, False, False
    # Over a tile glyph the translucent arrow breaks into pieces (dark ink
    # shows through); a 3x3 closing rejoins them without merging the outline.
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    arrows = []
    (wl, wh), (hl, hh) = CALL_ARROW_SIZE
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if not (wl <= w <= wh and hl <= h <= hh) or area < .3 * w * h:
            continue
        piece = labels[y:y+h, x:x+w] == index
        third = max(1, h // 3)
        if piece[:third].sum(1).mean() < 1.5 * max(1.0, piece[-third:].sum(1).mean()):
            continue
        arrows.append({"bbox": [x, y, w, h], "tip": [x + w / 2, y + h]})
    return arrows


def arrow_points_at(arrow, bbox):
    tx, ty = arrow["tip"]
    x, y, w, h = bbox
    return x <= tx <= x + w and CALL_ARROW_BELOW[0] <= y - ty <= CALL_ARROW_BELOW[1]


def highlighted_discard(pixels, regions, player):
    """Only a unique lit discard on another seat's final river tile qualifies.

    A tile is lit by the green outline (two sides lined) or by the call arrow
    pointing at it (``call_arrows``). A light on two seats' final tiles, on the
    player's own river or around an unreadable tile leaves the source
    unconfirmed for manual review. River lengths/order never establish the
    current discarder.
    """
    mask = glow_mask(pixels)
    arrows = call_arrows(pixels)
    rivers = {name: region for name, region in regions.items() if region["kind"] == "river"}
    boxes = [item["bbox"] for region in rivers.values() for item in region["items"]]
    measured = []
    for name, region in rivers.items():
        for index, tile in enumerate(region["items"]):
            others = [box for box in boxes if box is not tile["bbox"]]
            count, density, sides = tile_glow(mask, tile["bbox"], others)
            outlined = len(sides) >= GLOW_MIN_SIDES and count >= GLOW_MIN_PIXELS and density >= GLOW_MIN_DENSITY
            pointed = any(arrow_points_at(arrow, tile["bbox"]) for arrow in arrows)
            if outlined or pointed:
                # The wider rim measurement is recorded for diagnosis only: on
                # real frames a green bamboo face lights up a neighbour's band
                # more strongly than the outline does.
                halo_pixels, coverage, halo_sides = glow_halo(mask, tile["bbox"], others)
                measured.append({"target": region.get("player"), "tile": tile["label"], "region": name,
                                 "bbox": tile["bbox"], "glow_pixels": count, "glow_density": round(density, 3),
                                 "glow_sides": sides, "rim_pixels": halo_pixels,
                                 "rim_coverage": round(coverage, 3), "rim_sides": halo_sides,
                                 "outlined": outlined, "arrow": pointed,
                                 "last": index == len(region["items"]) - 1})
    # Only the final tile of a river can be the discard waiting for a call.
    candidates = [value for value in measured if value["last"]]
    seats = {value["target"] for value in candidates}
    best = None
    if len(seats) == 1:
        best = max(candidates, key=lambda value: (value["arrow"] + value["outlined"], value["glow_density"]))
    confirmed = bool(best and best["target"] != player and best["tile"] != "?")
    result = {"confirmed": confirmed, "candidates": candidates, "glowing_tiles": measured, "arrows": arrows,
              "evidence": "last_discard_green_outline_or_call_arrow"}
    if confirmed:
        result.update({k: best[k] for k in ("target", "tile", "bbox")})
    elif best and best["target"] != player:
        # Seat is clear but the tile is unreadable.
        result["unconfirmed_target"] = best["target"]
    return result


# Accepted riichi deposits lie on the centre board in front of each player.
# The default stick is a white bar with a red dot, measured at 1366x768:
# right 775-791 x 266-340, left 575-591 x 257-330 (vertical, 16x74). A
# player's own cosmetic stick may be a different skin (the user's is an
# orange bat), so the player's own riichi also comes from the sideways river
# tile. Zones are (x, y, w, h) per seat name.
RIICHI_STICK_ZONES = {"self": (600, 345, 170, 43), "right": (758, 250, 50, 110),
                      "top": (600, 232, 170, 50), "left": (558, 250, 50, 110)}
RIICHI_STICK_MIN_LENGTH = 45
RIICHI_STICK_MIN_WIDTH = 8
RIICHI_STICK_MIN_ELONGATION = 3.5


def riichi_sticks(pixels, seat_mapping):
    """Seats whose white riichi stick is on the board: {player_id: bbox or None}."""
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    found = {}
    for seat, (x, y, w, h) in RIICHI_STICK_ZONES.items():
        if seat not in seat_mapping:
            continue
        zone = hsv[y:y+h, x:x+w]
        if zone.shape[:2] != (h,w):
            # Fixed board zones can fall outside a smaller custom calibration.
            # Never pass an empty image to OpenCV's connected components.
            found[seat_mapping[seat]] = None
            continue
        white = ((zone[:, :, 1] <= 50) & (zone[:, :, 2] >= 200)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(white)
        stick = None
        for bx, by, bw, bh, area in (stats[i] for i in range(1, count)):
            long_side, short_side = max(bw, bh), min(bw, bh)
            if (long_side >= RIICHI_STICK_MIN_LENGTH and short_side >= RIICHI_STICK_MIN_WIDTH
                    and long_side >= RIICHI_STICK_MIN_ELONGATION * short_side and area >= .5 * bw * bh):
                stick = [int(bx + x), int(by + y), int(bw), int(bh)]
        found[seat_mapping[seat]] = stick
    return found
