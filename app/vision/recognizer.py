from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import time

import cv2
import numpy as np

from app.state.events import base
from .actions import call_arrows, call_banners, enabled_action_buttons, highlighted_discard, riichi_sticks
from .config import VisionError, native_box, pixel_rect, to_reference, validate_config
from .detector import blue_back_evidence, orange_back_evidence, concealed_back_mask, global_box, river_slots, tile_boxes, tile_candidates
from .hand import hand_layout
from .faces import unrotate_quad
from .templates import TemplateBank, rotate
from .text import TextReader


class StaticRecognizer:
    def __init__(self, config: dict, bank: TemplateBank, text_reader: TextReader | None = None):
        validate_config(config)
        self.config, self.bank, self.text_reader = config, bank, text_reader
        from .turn import WindReader
        self.wind_reader = WindReader(config.get("self_wind"))
        # Last unobstructed reading of every river slot (see reuse_marked_tiles).
        self.river_memory = {}
        if (config.get("tile_classifier") or {}).get("name") == "pjura":
            from .pjura import PjuraBank
            if not isinstance(bank, PjuraBank):
                self.bank = PjuraBank(bank, config["tile_classifier"])
        self.river_polygons = None
        if config.get("river_polygons") is not None and any(
                r["kind"]=="river" and "grid" in r for r in config["regions"].values()):
            from .river_polygons import RiverPolygonReader
            self.river_polygons = RiverPolygonReader(config,self.bank)

    def recognize(self, pixels: np.ndarray, metadata: dict | None = None) -> dict:
        started = time.perf_counter()
        # Larger same-aspect clients (e.g. 1920x1080) are laid out at the
        # calibrated size; the tile classifier reads the native pixels.
        pixels, native, scale = to_reference(pixels, self.config)
        height, width = pixels.shape[:2]
        metadata = metadata or {}
        from .turn import detect_timer
        result = {"schema_version": 1, "frame": metadata, "config_version": self.config["version"],
                  "template_manifest_sha256":getattr(self.bank,"sha256",None),
                  "classifier_manifest_sha256":getattr(self.bank,"classifier_manifest_sha256",None),
                  "feature_version":getattr(self.bank,"feature_version",None),
                  "image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                  "regions": {}, "issues": [], "status": "needs_confirmation", "score_kind": "not_probability",
                  "reference_size": [width, height]}
        if native is not None:
            result["native_size"] = [int(native.shape[1]), int(native.shape[0])]
            result["native_scale"] = scale
            result["native_image_sha256"] = hashlib.sha256(native.tobytes()).hexdigest()
        if hasattr(self.bank, "tile_classifier_info"):
            result["tile_classifier"] = deepcopy(self.bank.tile_classifier_info)
        result["action_timer"] = detect_timer(pixels, self.config.get("action_timer"), self.text_reader)
        result["self_wind"] = self.wind_reader.read(pixels)
        # "window_content" is the window's own rendering: nothing drawn above it.
        if metadata.get("capture_quality") not in (None, "unoccluded", "window_content"):
            result["issues"].append({"reason": "capture_requires_manual_review",
                                     "capture_quality": metadata["capture_quality"]})
        if not self.config.get("calibrated"):
            result["issues"].append({"reason": "uncalibrated_regions"})
        thresholds = self.config["thresholds"]
        layouts = {}
        for name, region in self.config["regions"].items():
            if "hand_layout" not in region:
                continue
            hx, hy, hw, hh = pixel_rect(region["rect"], width, height)
            layout = hand_layout(pixels[hy:hy+hh, hx:hx+hw], region)
            layouts[name] = {"boxes": layout["hand"], "layout": layout}
            draw_name = region["hand_layout"]["draw_region"]
            if layout["draw"]:
                bx, by, bw, bh = layout["draw"][0]
                layouts[draw_name] = {"rect": (hx+bx,hy+by,bw,bh), "boxes": [[0,0,bw,bh]], "layout": layout}
            else:
                layouts[draw_name] = {"boxes": [], "layout": layout}
            for reason in layout["reasons"]:
                result["issues"].append({"region": name, "reason": reason})
        self_bodies = []
        for name,region in self.config["regions"].items():
            if region["kind"] not in {"hand","draw"} or region.get("player") != self.config["seat_mapping"]["self"]:
                continue
            x,y,w,h = pixel_rect(region["rect"],width,height)
            if name in layouts:
                x,y,w,h = layouts[name].get("rect",(x,y,w,h))
            degrees = region.get("rotation",0)
            patch = rotate(pixels[y:y+h,x:x+w],degrees)
            boxes = layouts[name]["boxes"] if name in layouts else tile_boxes(patch,region)
            self_bodies.extend(global_box(box,(x,y),(w,h),degrees) for box in boxes)
        visible = Counter()
        for name, region in self.config["regions"].items():
            x, y, w, h = pixel_rect(region["rect"], width, height)
            if name in layouts:
                x,y,w,h = layouts[name].get("rect", (x,y,w,h))
            degrees = region.get("rotation", 0)
            crop = rotate(pixels[y:y+h, x:x+w], degrees)
            output = {"kind": region["kind"], "bbox": [x, y, w, h], "rotation": degrees,
                      "player": region.get("player"), "items": [], "complete": True}
            if name in layouts:
                layout = layouts[name]["layout"]
                output["layout_evidence"] = layout["evidence"]
                output["layout_reasons"] = layout["reasons"]
                output["complete"] = not layout["reasons"]
            if region["kind"] == "text":
                output["field"] = region["field"]
                ocr = self.text_reader.read(crop, region["field"], thresholds["ocr"]) if self.text_reader else {
                    "value": None, "raw_text": "", "score": None, "reason": "missing_text_reader"}
                template = self.bank.text_match(crop,name,region["field"],thresholds.get("text_template",.98),thresholds["margin"])
                output["text"] = {**ocr,"template_evidence":template}
                if template["value"] is not None:
                    if ocr["value"] is None:
                        output["text"] = {**template,"ocr_evidence":ocr}
                    elif ocr["value"] != template["value"]:
                        output["text"].update(value=None,reason="conflicting_text_evidence")
                elif ocr["value"] is None:
                    agreed = text_agreement(ocr, template, region["field"])
                    if agreed is not None:
                        output["text"].update(value=agreed, reason=None, evidence="ocr_and_template_agree")
                output["complete"] = output["text"]["value"] is not None
            elif region["kind"] == "buttons":
                matches = self.bank.button_matches(crop, thresholds["button"])
                if region.get("enabled_text_ocr"):
                    matches.extend(enabled_action_buttons(crop, self.text_reader, thresholds["button"]))
                if region.get("banner_rect"):
                    # The coloured reaction banners (吃/碰/杠/和) are the stronger
                    # evidence: they replace text reads of the same button. Up to
                    # three banners extend left of the text region.
                    bx, by, bw, bh = pixel_rect(region["banner_rect"], width, height)
                    banners = call_banners(pixels[by:by+bh, bx:bx+bw], self.text_reader)
                    for banner in banners:
                        banner["bbox"][0] += bx - x
                        banner["bbox"][1] += by - y
                    def inside(item, banner):
                        ix, iy, iw, ih = item["bbox"]; bx0, by0, bw0, bh0 = banner["bbox"]
                        cx, cy = ix + iw / 2, iy + ih / 2
                        return bx0 - 10 <= cx <= bx0 + bw0 + 10 and by0 - 40 <= cy <= by0 + bh0 + 10
                    matches = [m for m in matches if not any(inside(m, b) for b in banners)] + banners
                for match in matches:
                    match["bbox"] = global_box(match["bbox"], (x, y), (w, h), degrees)
                    match["visible"] = True
                    # Enabled OCR also requires the active glyph's colour evidence.
                    match["clickable"] = {"enabled": True, "disabled": False, "clicked": False}.get(match["state"])
                output["items"] = matches
                available = bool(matches) or any(entry.get("kind") == "button" for entry, _ in self.bank.entries)
                output["complete"] = available and all(item["clickable"] is not None for item in matches)
            else:
                exclusions = []
                if region["kind"] == "meld" and region.get("player") == self.config["seat_mapping"]["self"]:
                    # The meld ROI spans both full concealed bodies and smaller
                    # open tiles after calls. Exclude already detected hand/draw
                    # slots from its mask only; keep original classifier pixels.
                    for sx,sy,sw,sh in self_bodies:
                        left,top,right,bottom = max(x,sx),max(y,sy),min(x+w,sx+sw),min(y+h,sy+sh)
                        if right>left and bottom>top:
                            exclusions.append(global_box([left-x,top-y,right-left,bottom-top],(0,0),
                                                         (crop.shape[1],crop.shape[0]),(-degrees)%360))
                exclusion_mask = None
                if region.get("concealed_hand_exclusions"):
                    exclusion_mask = np.zeros((h,w),np.uint8)
                    for polygon in region["concealed_hand_exclusions"]:
                        points = np.rint(np.asarray(polygon)*[width,height]-[x,y]).astype(np.int32)
                        cv2.fillPoly(exclusion_mask,[points],255)
                    exclusion_mask = rotate(exclusion_mask,degrees)
                    exclusion_mask = concealed_back_mask(crop,exclusion_mask,region)
                    # A back clipped by the meld ROI can show only its white
                    # side. Read a narrow halo for the missing back pattern,
                    # then cut the exclusion back to this ROI before rotating.
                    pad = max(region.get("min_size",[15,20]))
                    cx,cy = max(0,x-pad),max(0,y-pad)
                    cr,cb = min(width,x+w+pad),min(height,y+h+pad)
                    context = pixels[cy:cb,cx:cr]
                    corridor = np.zeros(context.shape[:2],np.uint8)
                    for polygon in region["concealed_hand_exclusions"]:
                        points = np.rint(np.asarray(polygon)*[width,height]-[cx,cy]).astype(np.int32)
                        cv2.fillPoly(corridor,[points],255)
                    contextual = concealed_back_mask(context,corridor,region)[y-cy:y-cy+h,x-cx:x-cx+w]
                    exclusion_mask = np.maximum(exclusion_mask,rotate(contextual,degrees))
                    output["concealed_hand_exclusions"] = deepcopy(region["concealed_hand_exclusions"])
                    output["concealed_back_excluded_pixels"] = int(np.count_nonzero(exclusion_mask))
                if region.get("exclude_polygons"):
                    # Interface art inside the ROI (e.g. the 理/和/鸣/切 panel
                    # beside the left player's melds) is never a tile.
                    ui_mask = np.zeros((h,w),np.uint8)
                    for polygon in region["exclude_polygons"]:
                        points = np.rint(np.asarray(polygon)*[width,height]-[x,y]).astype(np.int32)
                        cv2.fillPoly(ui_mask,[points],255)
                    ui_mask = rotate(ui_mask,degrees)
                    exclusion_mask = ui_mask if exclusion_mask is None else np.maximum(exclusion_mask, ui_mask)
                detected = {"boxes":layouts[name]["boxes"],"unresolved":[]} if name in layouts else tile_candidates(crop,region,exclude_boxes=exclusions,exclude_mask=exclusion_mask)
                boxes = detected["boxes"]
                if detected.get('neighbour_crops'):
                    output['neighbour_crops']=[{**part,**{key:global_box(part[key],(x,y),(w,h),degrees)
                        for key in ('bbox','cropped_bbox','neighbour_bbox')}} for part in detected['neighbour_crops']]
                if detected.get('split_meld_faces'):
                    output['split_meld_faces']=[{**s,'bbox':global_box(s['bbox'],(x,y),(w,h),degrees),
                        'parts':[global_box(b,(x,y),(w,h),degrees) for b in s['parts']]}
                        for s in detected['split_meld_faces']]
                if detected.get("merged_fragments"):
                    output["merged_fragments"] = [
                        {**part, **{key:global_box(part[key],(x,y),(w,h),degrees)
                                   for key in ("bbox","body_bbox","merged_bbox")}}
                        for part in detected["merged_fragments"]]
                if detected["unresolved"]:
                    unresolved = []
                    for part in detected["unresolved"]:
                        bx,by,bw,bh = part["bbox"]
                        match = self.bank.classify(crop[by:by+bh,bx:bx+bw],name,thresholds["tile"],thresholds["margin"],(0,90,270))
                        if match["label"] == "back":
                            output.setdefault("backs",[]).append({**match,"bbox":global_box(part["bbox"],(x,y),(w,h),degrees),
                                                                  "canonical_bbox":part["bbox"], "horizontal":None,
                                                                  "detector_reason":part["reason"]})
                        elif (region["kind"] == "meld" and match["label"] == "?"
                              and is_background(match, region)
                              and (match.get("pjura_evidence") or {}).get("face_fitted") is False
                              and ((match.get("pjura_evidence") or {}).get("face_tint") or {}).get("neutral") is False):
                            output.setdefault("background_candidates",[]).append({
                                **match,"bbox":global_box(part["bbox"],(x,y),(w,h),degrees),
                                "canonical_bbox":part["bbox"], "reason":"not_a_tile_background",
                                "rejected_reason":match.get("reason"),"detector_reason":part["reason"]})
                        else:
                            unresolved.append({**part,"bbox":global_box(part["bbox"],(x,y),(w,h),degrees)})
                    if unresolved:
                        output["unresolved_components"] = unresolved
                        output["complete"] = False
                rotations = (0, 90, 270) if region["kind"] in {"river", "meld", "choices"} else (0,)
                dimmed = layouts[name]["layout"].get("dimmed", []) if name in layouts and region["kind"] == "hand" else []
                matches = None
                if self.river_polygons is not None and region["kind"] == "river" and name in self.river_polygons.model["regions"]:
                    source = native if native is not None else pixels
                    nx,ny,nw,nh = pixel_rect(region["rect"],source.shape[1],source.shape[0])
                    hires_region = rotate(source[ny:ny+nh,nx:nx+nw],degrees)
                    raw_crops=[]
                    for box in boxes:
                        nx,ny,nw,nh = native_box(global_box(box,(x,y),(w,h),degrees),scale,source.shape)
                        raw_crops.append(rotate(source[ny:ny+nh,nx:nx+nw],degrees))
                    matches = self.river_polygons.classify(name,region,crop,hires_region,boxes,
                        thresholds["tile"],thresholds["margin"],raw_model_crops=raw_crops)
                elif hasattr(self.bank, "classify_many"):
                    faces = [crop[by:by+bh, bx:bx+bw] for bx, by, bw, bh in boxes]
                    faces = [undim(face) if box in dimmed else face for face, box in zip(faces, boxes)]
                    options = {}
                    if native is not None:
                        # Same boxes cut from the native frame, in the region's pose.
                        hires = []
                        for box in boxes:
                            nx, ny, nw, nh = native_box(global_box(box, (x, y), (w, h), degrees), scale, native.shape)
                            face = rotate(native[ny:ny+nh, nx:nx+nw], degrees)
                            hires.append(undim(face) if box in dimmed else face)
                        options["model_crops"] = hires
                    matches = self.bank.classify_many(faces, name, thresholds["tile"], thresholds["margin"], rotations,
                                                      **options)
                horizontal = ([m.get("face_rotation", 0) in (90, 270)
                               if m["label"] != "?" and m.get("orientation_confirmed", True) else None
                               for m in matches] if matches is not None else None)
                slots = river_slots(boxes, region["grid"], horizontal) if "grid" in region else None
                for index, box in enumerate(boxes):
                    bx, by, bw, bh = box
                    face = crop[by:by+bh, bx:bx+bw]
                    match = matches[index] if matches is not None else self.bank.classify(face, name, thresholds["tile"], thresholds["margin"], rotations)
                    back = blue_back_evidence(face) if region["kind"] == "meld" else None
                    if back is None and region["kind"] == "meld" and "orange" in region.get("kan_back_colours",[]):
                        back = orange_back_evidence(face)
                    if back is not None:
                        # The physical back remains a back. Only a validated
                        # back/face/face/back group may infer its kan identity.
                        match = {"label":"back", "reason":None, "back_evidence":back,
                                 "identity_reading":match, "score_kind":"colour_and_body_evidence"}
                    if match["label"] == "back":
                        # A confidently matched visible blue back is negative
                        # evidence even at the boundary. Never expose its identity.
                        output.setdefault("backs", []).append({**match, "bbox": global_box(box, (x,y), (w,h), degrees),
                                                               "canonical_bbox":box, "horizontal":None,
                                                               "full_body":bx > 0 and by > 0 and bx+bw < crop.shape[1] and by+bh < crop.shape[0]})
                        continue
                    item = {**match, "bbox": global_box(box, (x, y), (w, h), degrees), "region": name,
                            "player": region.get("player"), "index": index, "canonical_bbox": box}
                    if box in dimmed:
                        # Greyed by the game: kuikae forbids discarding it now.
                        item["dimmed"] = True
                    if region.get("reject_boundary_fragments") and (bx <= 0 or by <= 0 or
                            bx+bw >= crop.shape[1] or by+bh >= crop.shape[0]):
                        evidence = match.get("pjura_evidence") or {}
                        if (region["kind"] == "meld" and match["label"] == "?" and is_background(item, region)
                                and evidence.get("face_fitted") is False
                                and (evidence.get("face_tint") or {}).get("neutral") is False):
                            # A coloured table decoration with no fitted tile face
                            # remains background even when it crosses the ROI edge.
                            # Unknown white bodies and fitted faces still block here.
                            output.setdefault("background_candidates", []).append(
                                {**item, "reason":"not_a_tile_background", "rejected_reason":match.get("reason"),
                                 "detector_reason":"candidate_at_region_boundary"})
                            continue
                        # A cropped neighbour/hand edge is not a detected meld.
                        # Keep its position and pause the region rather than calling
                        # it an empty area or assigning a tile identity.
                        output.setdefault("boundary_fragments", []).append({
                            "bbox":global_box(box,(x,y),(w,h),degrees),"label":"?",
                            "reason":"clipped_candidate_at_region_boundary"})
                        output["complete"] = False
                        continue
                    face_evidence = match if "face_quad_local" in match else match.get("face_evidence",{})
                    if "face_quad_local" in face_evidence:
                        points = np.asarray(face_evidence["face_quad_local"],np.float32)+[bx,by]
                        item["face_quad_global"] = (np.asarray(unrotate_quad(points,w,h,degrees))+[x,y]).tolist()
                    if "slots" in region:
                        item["slot_index"] = region["slots"].index(box)
                    # A rejected glyph proposal cannot confirm orientation. In
                    # particular, a cropped circular 1p can favour a sideways
                    # template even though the visible river slot is upright.
                    item["horizontal"] = (match.get("face_rotation", 0) in (90, 270)
                                          if match["label"] != "?" and match.get("orientation_confirmed", True) else None)
                    item["orientation_status"] = ("confirmed_by_slot_geometry" if match.get("orientation_source")=="calibrated_river_slot"
                        else "confirmed_by_template" if item["horizontal"] is not None else "unconfirmed")
                    if "grid" in region:
                        grid = region["grid"]
                        item["slot"], residual = slots[index]
                        row, column = item["slot"]
                        item["slot_residual_px"] = residual
                        if (min(row, column) < 0 or row >= grid.get("rows", 100) or
                                column >= grid.get("columns", 100) or residual > grid.get("tolerance", 10)):
                            item["label"], item["reason"] = "?", "river_slot_ambiguous"
                    if item["label"] == "?" and is_background(item, region):
                        # Table art (stars, feathers, character art) is kept
                        # as evidence but neither counts as a tile nor pauses
                        # the region.
                        output.setdefault("background_candidates", []).append(
                            {**item, "reason": "not_a_tile_background", "rejected_reason": item.get("reason")})
                        continue
                    if item["label"] == "?":
                        output["complete"] = False
                    output["items"].append(item)
                if region["kind"] == "hand" and not output["items"]:
                    output["complete"] = False
                if region["kind"] == "river" and "grid" not in region:
                    output["complete"] = False
                if region["kind"] == "river" and "grid" in region:
                    mark_sideways_by_shape(output["items"], region["grid"])
                    output["items"].sort(key=lambda item: item.get("slot", [100,100]))
                    slots = Counter(tuple(item["slot"]) for item in output["items"])
                    if any(count > 1 for count in slots.values()):
                        output["complete"] = False
                        for item in output["items"]:
                            if slots[tuple(item["slot"])] > 1:
                                item["label"], item["reason"] = "?", "duplicate_river_slot"
                    # Discards fill the lattice in order and a called tile
                    # leaves no hole, so an empty slot before the last tile is
                    # a tile the detector missed.
                    columns = region["grid"].get("columns", 6)
                    order = {row*columns+column for row, column in slots}
                    gaps = [index for index in range(max(order, default=-1)+1) if index not in order]
                    if gaps:
                        output["slot_gaps"] = [[index // columns, index % columns] for index in gaps]
                        output["complete"] = False
                if region["kind"] == "meld":
                    output["groups"] = group_melds(meld_items_with_kan_backs(output), region)
                    # Grouping may settle one unsure tile from the meld's
                    # structure, so completeness is decided after it.
                    blocked = any(output.get(key) for key in ("unresolved_components", "boundary_fragments"))
                    output["complete"] = (not blocked and all(item["label"] != "?" for item in output["items"])
                                          and all(group["status"] == "confirmed" for group in output["groups"]))
            result["regions"][name] = output
            if region.get("required", False) and not output["complete"]:
                result["issues"].append({"region": name, "reason": "incomplete_region"})
        reuse_marked_tiles(pixels, result, self.river_memory)
        drawn = any(r["items"] for r in result["regions"].values() if r["kind"] == "draw")
        for region in result["regions"].values():
            if region["kind"] == "buttons":
                for item in region["items"]:
                    if item["label"] == "win":
                        item["label"] = "tsumo" if drawn else "ron"
        result["riichi_sticks"] = riichi_sticks(pixels, self.config["seat_mapping"])
        result["reaction_discard"] = highlighted_discard(
            pixels, result["regions"], self.config["seat_mapping"]["self"])
        for region in result["regions"].values():
            if region["kind"] not in {"text","buttons","choices"}:
                if region["kind"] == "meld" and region["complete"]:
                    # An evidenced closed kan accounts for all four copies,
                    # including the two backs, in the physical-copy check.
                    visible.update(base(t) for g in region["groups"] for t in g["labels"])
                else:
                    visible.update(base(item["label"]) for item in region["items"] if item["label"] != "?")
        for tile, count in visible.items():
            if count > 4:
                result["issues"].append({"reason": "physical_tile_conflict", "tile": tile, "count": count})
        if not result["issues"]:
            result["status"] = "recognized"
        result["recognition_ms"] = (time.perf_counter()-started)*1000
        # Empty regions are not evidence that the frame is a stable/complete game state.
        result["requires_temporal_confirmation"] = True
        return result


SMALL_CANDIDATE_SHARE = .5


def is_background(item: dict, region: dict) -> bool:
    """A rejected river/meld candidate that shows no real tile face.

    Only already-rejected candidates qualify. One with a fitted tile face that
    sits on a valid, calibrated river slot stays an unknown tile, so a real but
    unreadable discard is never silently dropped.
    """
    if region["kind"] not in {"river", "meld"}:
        return False
    evidence = item.get("pjura_evidence") or {}
    if "grid" in region and evidence.get("face_fitted") is False and "bbox" in item:
        # Far smaller than one lattice cell and without a tile face: a felt star
        # or similar decoration, even when it happens to land on a slot.
        column, row = (float(np.linalg.norm(vector)) for vector in region["grid"]["basis"])
        if item["bbox"][2] * item["bbox"][3] < SMALL_CANDIDATE_SHARE * column * row:
            return True
    if not item.get("background_candidate"):
        return False
    on_slot = "grid" in region and item.get("reason") != "river_slot_ambiguous" and "slot" in item
    return not (evidence.get("face_fitted") and on_slot)


DIMMED_TARGET_VALUE = 223.0


def undim(face):
    """Restore a greyed hand tile's brightness for the classifier."""
    grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gain = float(np.clip(DIMMED_TARGET_VALUE / max(1.0, float(np.percentile(grey, 75))), 1.0, 2.2))
    return np.clip(face.astype(np.float32) * gain, 0, 255).astype(np.uint8)


# While a discard waits for a call, an arrow bobs above it (see
# actions.call_arrows). In the left river "above" is the previous discard,
# whose face the arrow then covers.
MARKER_MIN_PIXELS = 12
MARKER_MARGIN = 2
MEMORY_MAX_SHIFT = 4                       # top-left corner, px at 1366x768
MEMORY_MIN_OVERLAP = .6
MEMORY_MAX_DIFFERENCE = 14.0               # mean absolute grey difference outside the arrow
MEMORY_MIN_VISIBLE = .5


def marker_mask(pixels):
    """Pixels covered by call arrows (their boxes, slightly grown)."""
    mask = np.zeros(pixels.shape[:2], bool)
    for arrow in call_arrows(pixels):
        x, y, w, h = arrow["bbox"]
        mask[max(0, y-MARKER_MARGIN):y+h+MARKER_MARGIN, max(0, x-MARKER_MARGIN):x+w+MARKER_MARGIN] = True
    return mask


def reuse_marked_tiles(pixels, result, memory):
    """Read a river tile under a call arrow using independent glyphs or memory.

    Discards never change once placed, and the player's own turn right after
    each discard shows it unobstructed. A '?' tile qualifies only when the
    arrow covers it, the same slot was read before, the box has not moved and
    every pixel the arrow leaves visible still matches that earlier view.
    """
    marker = marker_mask(pixels)
    for name, region in result["regions"].items():
        if region["kind"] != "river":
            continue
        reused = False
        for item in region["items"]:
            if "slot" not in item or min(item["slot"]) < 0:
                continue
            key = (name, tuple(item["slot"]))
            x, y, w, h = item["bbox"]
            covered = marker[y:y+h, x:x+w]
            # The existing template + three frozen train SVMs sometimes read a
            # glyph the ViT cannot accept through an arrow. Require its own
            # accepted vote evidence, agreement with the ViT's clear proposal,
            # and enough uncovered face. Normal unmarked low-confidence tiles
            # retain the primary policy; blank faces and five-colour ambiguity
            # cannot be settled by this glyph-only path.
            legacy = item.get("legacy_evidence") or {}
            verifier = legacy.get("classifier_evidence") or {}
            tile = legacy.get("label", "?")
            if (item["label"] == "?" and item.get("reason") == "pjura_low_confidence"
                    and covered.sum() >= MARKER_MIN_PIXELS and covered.mean() <= 1-MEMORY_MIN_VISIBLE
                    and tile not in {"?","P","back","5m","5mr","5p","5pr","5s","5sr"}
                    and tile == item.get("best_label")
                    and (item.get("score") or 0) >= .5 and (item.get("margin") or 0) >= .3
                    and legacy.get("acceptance_policy") == "unanimous_train_svm_and_existing_template_proposal"
                    and verifier.get("accepted") is True and verifier.get("votes") == [tile]*3
                    and (legacy.get("score") or 0) >= .75):
                item["occluded_reading"] = {"label":item["label"],"reason":item["reason"],
                                            "score":item.get("score"),"margin":item.get("margin")}
                item.update(label=tile,reason=None,score=legacy["score"],margin=legacy.get("margin"),
                            score_kind="template_similarity_not_probability",
                            evidence="call_arrow_with_unanimous_existing_glyph_verifiers",
                            acceptance_policy="call_arrow_template_and_three_svm_agreement",
                            horizontal=item.get("horizontal") is True or legacy.get("face_rotation",0) in (90,270),
                            orientation_status=item.get("orientation_status") if item.get("horizontal") is True else "confirmed_by_template")
                reused = True
            if item["label"] != "?":
                if covered.sum() < MARKER_MIN_PIXELS:
                    memory[key] = {"label": item["label"], "bbox": list(item["bbox"]), "score": item.get("score"),
                                   "pixels": cv2.cvtColor(pixels[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)}
                continue
            earlier = memory.get(key)
            if covered.sum() < MARKER_MIN_PIXELS or earlier is None:
                continue
            ex, ey, ew, eh = earlier["bbox"]
            if max(abs(ex - x), abs(ey - y)) > MEMORY_MAX_SHIFT:
                continue
            # The box shrinks when a later tile hides the side face; compare
            # the part both views share.
            left, top = max(x, ex), max(y, ey)
            right, bottom = min(x + w, ex + ew), min(y + h, ey + eh)
            if right <= left or bottom <= top or (right - left) * (bottom - top) < MEMORY_MIN_OVERLAP * w * h:
                continue
            visible = ~marker[top:bottom, left:right]
            if visible.mean() < MEMORY_MIN_VISIBLE:
                continue
            now = cv2.cvtColor(pixels[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(np.float32)
            then = earlier["pixels"][top - ey:bottom - ey, left - ex:right - ex]
            difference = float(np.abs(now - then)[visible].mean())
            if difference > MEMORY_MAX_DIFFERENCE:
                continue
            item["occluded_reading"] = {"label": item["label"], "best_label": item.get("best_label"),
                                        "reason": item.get("reason"), "score": item.get("score")}
            item.update(label=earlier["label"], reason=None, score=earlier["score"],
                        evidence="earlier_unobstructed_reading_under_call_arrow",
                        memory_difference=round(difference, 2))
            reused = True
        if (reused and region.get("complete") is False and all(i["label"] != "?" for i in region["items"])
                and not any(region.get(k) for k in ("unresolved_components", "boundary_fragments", "slot_gaps"))
                and len({tuple(i["slot"]) for i in region["items"]}) == len(region["items"])):
            region["complete"] = True
            result["issues"] = [i for i in result["issues"]
                                if not (i.get("region") == name and i.get("reason") == "incomplete_region")]


# Neither reader alone is sure (measured: OCR "東1局" 0.81, template 0.94), but
# two independent readers giving the same parsed value are.
AGREE_MIN_OCR = .75
AGREE_MIN_TEMPLATE = .8


def text_agreement(ocr, template, field):
    from .text import parse_text
    variants = [v for v in ocr.get("variants", []) if v.get("score", 0) >= AGREE_MIN_OCR]
    best = template.get("best_label") or template.get("raw_text")
    if not variants or not best or (template.get("score") or 0) < AGREE_MIN_TEMPLATE:
        return None
    expected = parse_text(best, field)
    values = {json.dumps(parse_text(v["text"], field), sort_keys=True) for v in variants}
    if expected is None or values != {json.dumps(expected, sort_keys=True)}:
        return None
    return expected


# A riichi tile lies sideways in the river. The orientation classifier often
# cannot confirm it (measured: own 5s riichi tile 57x45 left unconfirmed, the
# rest of its row 42-45 x 57), and a missed riichi silently drops the riichi
# from the position. Compared with the other tiles of its row a sideways tile
# is clearly wider and lower; perspective alone never does both (the near
# tile closing a side-river row is wider but also taller: 43x56 vs 27x52).
SIDEWAYS_ROW_WIDTH = 1.2
SIDEWAYS_ROW_HEIGHT = .85
SIDEWAYS_PITCH_WIDTH = 1.25
SIDEWAYS_PITCH_ASPECT = 1.1


def mark_sideways_by_shape(items, grid):
    column_pitch = float(np.linalg.norm(grid["basis"][0]))
    rows = {}
    for item in items:
        if "slot" in item and min(item["slot"]) >= 0 and "canonical_bbox" in item:
            rows.setdefault(item["slot"][0], []).append(item)
    for row in rows.values():
        for item in row:
            if item.get("horizontal") is True:
                continue
            width, height = item["canonical_bbox"][2:]
            others = [other["canonical_bbox"][2:] for other in row if other is not item]
            if len(others) >= 2:
                sideways = (width >= SIDEWAYS_ROW_WIDTH * float(np.median([o[0] for o in others]))
                            and height <= SIDEWAYS_ROW_HEIGHT * float(np.median([o[1] for o in others])))
            else:
                sideways = width >= SIDEWAYS_PITCH_WIDTH * column_pitch and width >= SIDEWAYS_PITCH_ASPECT * height
            if sideways:
                item["horizontal"] = True
                item["orientation_status"] = "sideways_by_shape"


# A sideways honour-like face in another player's meld (measured: the 1s bird
# turned sideways at 0.74, next best 0.02) can miss the 0.9 tile threshold
# although nothing else fits. The other tiles of the group are confident.
MELD_SETTLE_MIN_SCORE = .5
MELD_SETTLE_MIN_MARGIN = .3


def meld_types(labels):
    normalized = [base(label) for label in labels]
    if len(labels) == 3 and len(set(normalized)) == 1:
        return ["pon"]
    if (len(labels) == 3 and all(len(tile) == 2 and tile[0].isdigit() for tile in normalized)
            and len({tile[1] for tile in normalized}) == 1):
        numbers = sorted(int(tile[0]) for tile in normalized)
        if numbers == list(range(numbers[0], numbers[0]+3)):
            return ["chi"]
    if len(labels) == 4 and len(set(normalized)) == 1:
        return ["daiminkan", "ankan", "kakan"]
    return []


def settle_unknown_meld_tile(items):
    """Accept the best reading of the one unsure tile when it completes the meld.

    Only one unknown in a three- or four-tile group, a clear best reading
    (score and margin), and that reading must turn the group into a valid
    chi/pon/kan. Anything else stays unknown for review.
    """
    unknown = [item for item in items if item["label"] == "?"]
    if len(items) not in (3, 4) or len(unknown) != 1:
        return
    item = unknown[0]
    best, score, margin = item.get("best_label"), item.get("score") or 0, item.get("margin") or 0
    if (not best or best in ("?", "back") or score < MELD_SETTLE_MIN_SCORE or margin < MELD_SETTLE_MIN_MARGIN
            or item.get("reason") not in (None, "pjura_low_confidence")):
        return
    if not meld_types([best if other is item else other["label"] for other in items]):
        return
    item["unsure_reading"] = {"reason": item.get("reason"), "score": score, "margin": margin}
    item.update(label=best, reason=None, evidence="meld_structure_confirms_best_reading")


# Measured open-meld boxes: sideways 45x32, 73x54 (w/h 1.35-1.41); upright
# 35x39, 63x68 (0.90-0.93).
SIDEWAYS_MELD_ASPECT = 1.15
UPRIGHT_MELD_ASPECT = 1.0

# Side-view foreshortening can keep even a called tile's face portrait-shaped.
# Compare faces in the same triplet: the two upright faces have similar ratios,
# while the sideways face has at least 1.5 times their width/height ratio.


def blank_called_position(items: list[dict]):
    """A blank glyph needs the same physical corner ordering as other tiles."""
    from .meld_geometry import relative_called_positions
    if len(items) != 3 or any(item["label"] != "P" for item in items):
        return None
    found = relative_called_positions(items,blank=True)
    return (found[0][0],found[1]) if found is not None else None


def face_called_position(items: list[dict]):
    """Use shared physical-face evidence for a coherent non-blank triplet."""
    from .kan import row_geometry
    from .meld_geometry import relative_called_positions
    if len(items) != 3 or any(item["label"] in ("?", "P", "back") for item in items) or not row_geometry(items):
        return None
    found = relative_called_positions(items)
    return (found[0][0],found[1]) if found is not None else None


def meld_items_with_kan_backs(output):
    """Attach backs only where a complete kan structure proves their role.

    Unrelated hand backs and clipped/oversized back fragments remain negative
    evidence, as before. They do not create fictional melds or complete a kan.
    """
    items = output["items"]
    backs = [i for i in output.get("backs",[]) if i.get("full_body") is True]
    ordered = sorted([*items,*backs],key=lambda i:i["canonical_bbox"][0])
    proven = set()
    for start in range(len(ordered)-3):
        part = ordered[start:start+4]
        if closed_kan_labels(part) is not None:
            proven.update(id(i) for i in part if i["label"] == "back")
    return [*items, *[i for i in backs if id(i) in proven]]


def closed_kan_labels(items):
    """Back/face/face/back with one coherent row is an evidenced closed kan.

    Observed back labels are never overwritten. Only the group's consumed
    identities are inferred. Four fives necessarily contain exactly one red
    five under this application's fixed four-player, one-red-per-suit rules.
    """
    labels = [i["label"] for i in items]
    if (len(items) != 4 or labels[0] != "back" or labels[3] != "back"
            or "back" in labels[1:3]):
        return None
    boxes = np.asarray([i["canonical_bbox"] for i in items],dtype=float)
    if not np.all(np.isfinite(boxes)) or np.any(boxes[:,2:] <= 0) or len(set(boxes[:,0])) != 4:
        return None
    widths, heights = boxes[:,2], boxes[:,3]
    pitch = np.diff(boxes[:,0])
    # Side-seat perspective gives a sloped row. Require all four bodies to
    # follow one straight baseline, not just similar neighbouring heights.
    slope,intercept=np.polyfit(boxes[:,0],boxes[:,1],1)
    aligned=(np.ptp(boxes[:,1]) <= .25*min(heights) or
             (abs(slope) <= .35 and np.max(np.abs(boxes[:,1]-(slope*boxes[:,0]+intercept))) <= .1*min(heights)))
    if (not np.all(np.isfinite(boxes)) or min(*widths,*heights) < 15 or max(widths) > 1.5*min(widths)
            or max(heights) > 1.3*min(heights)
            or not aligned
            or np.any(pitch < .65*widths[:-1]) or np.any(pitch > 1.35*widths[:-1])
            or any(i.get("horizontal") is True for i in items)):
        return None
    # One weak face may be settled by the evidenced kan, just as one weak
    # face can be settled by an existing chi/pon. It must already propose the
    # same tile as the clear face; never infer from two unreadable faces.
    labels = list(labels)
    if labels[1:3] == ['?','?']:
        faces=items[1:3]
        proposals=[i.get('best_label') for i in faces]
        if (None in proposals or base(proposals[0]) != base(proposals[1]) or
                any((i.get('score') or 0) < MELD_SETTLE_MIN_SCORE or
                    (i.get('margin') or 0) < MELD_SETTLE_MIN_MARGIN or
                    (i.get('legacy_evidence') or {}).get('label') != i.get('best_label') for i in faces)):
            return None
        labels[1:3]=proposals
    if "?" in labels:
        index = labels.index("?")
        known = labels[3-index]
        item = items[index]
        proposal = item.get("best_label")
        if (proposal is None or base(proposal) != base(known)
                or (item.get("score") or 0) < MELD_SETTLE_MIN_SCORE
                or (item.get("margin") or 0) < MELD_SETTLE_MIN_MARGIN):
            return None
        labels[index] = proposal
    if base(labels[1]) != base(labels[2]):
        return None
    tile = base(labels[1])
    result = [tile, labels[1], labels[2], tile]
    if tile in ("5m","5p","5s"):
        red = sum(t.endswith("r") for t in result)
        if red > 1:
            return None
        if red == 0:
            result[0] += "r"
    return result


def group_melds(items: list[dict], region: dict) -> list[dict]:
    """Keep observed identities/layers. Unproven type or source stays unresolved."""
    groups = []
    gap = region.get("group_gap", 18)
    for item in sorted(items, key=lambda value: value["canonical_bbox"][0]):
        x, y, w, h = item["canonical_bbox"]
        if not groups or x - groups[-1]["right"] > gap:
            groups.append({"items": [], "right": x+w})
        groups[-1]["items"].append(item)
        groups[-1]["right"] = max(groups[-1]["right"], x+w)
    separated = []
    for group in groups:
        partitions = _meld_partitions(group["items"])
        if len(partitions) == 1:
            separated.extend({"items":part,"right":0} for part in partitions[0])
        else:
            separated.append(group)
    groups = separated
    for group in groups:
        closed = closed_kan_labels(group["items"])
        if closed is not None:
            observed = [i["label"] for i in group["items"]]
            for index,item in enumerate(group["items"]):
                if item["label"] == "?":
                    item["structural_reading"] = {"label":"?","best_label":item.get("best_label"),
                                                  "score":item.get("score"),"reason":item.get("reason")}
                    item.update(label=closed[index],reason=None,evidence="closed_kan_matching_face_proposal")
            group.update(labels=closed, observed_labels=observed,
                         type="ankan", type_hypotheses=["ankan"], target=None,
                         called_positions=[], called_position_evidence="no_called_tile_in_closed_kan",
                         structure_evidence="aligned_back_face_face_back", status="confirmed")
            group.pop("right")
            continue
        # Two clear identical faces plus two matching network AND accepted
        # template proposals may complete a geometrically proven open/added
        # kan. Four matching identities by themselves never enable this path.
        quad_items = group['items']
        unknown = [i for i in quad_items if i['label'] == '?']
        known = [i['label'] for i in quad_items if i['label'] not in {'?','back'}]
        if len(quad_items) == 4 and unknown and len(known) >= 2 and len({base(i) for i in known}) == 1:
            from .kan import kan_structure
            same = all(i.get('best_label') is not None and base(i['best_label']) == base(known[0])
                       and (i.get('score') or 0) >= MELD_SETTLE_MIN_SCORE
                       and (i.get('margin') or 0) >= MELD_SETTLE_MIN_MARGIN
                       and (i.get('legacy_evidence') or {}).get('label') == i['best_label'] for i in unknown)
            proposed = [{**i,'label':i.get('best_label','?')} if i['label'] == '?' else i for i in quad_items]
            if same and kan_structure(proposed,region) is not None:
                for item in unknown:
                    item['structural_reading'] = {k:item.get(k) for k in ('label','best_label','score','reason')}
                    item.update(label=item['best_label'],reason=None,evidence='kan_geometry_network_and_template_agree')
        settle_unknown_meld_tile(group["items"])
        labels = [item["label"] for item in group["items"]]
        normalized = [base(label) for label in labels]
        types = []
        if not {"?","back"}.intersection(labels) and len(labels) == 3:
            if len(set(normalized)) == 1:
                types = ["pon"]
            elif all(len(tile) == 2 for tile in normalized) and len({tile[1] for tile in normalized}) == 1:
                numbers = sorted(int(tile[0]) for tile in normalized)
                if numbers == list(range(numbers[0], numbers[0]+3)):
                    types = ["chi"]
        if not {"?","back"}.intersection(labels) and len(labels) == 4 and len(set(normalized)) == 1:
            from .kan import kan_structure
            kan=kan_structure(group['items'],region)
            if kan is not None:
                group.update(labels=labels,type_hypotheses=[kan['type']],status='confirmed',**kan)
                group.pop('right')
                continue
            types = ["daiminkan", "ankan", "kakan"]
        horizontal = [index for index, item in enumerate(group["items"]) if item["horizontal"]]
        evidence = "tile_orientation"
        if not horizontal and types in (["chi"], ["pon"]):
            # Blank 白 faces and some sideways tiles give no orientation. The
            # called tile still lies sideways: the only one wider than tall.
            aspects = [item["canonical_bbox"][2] / max(1, item["canonical_bbox"][3]) for item in group["items"]]
            wide = [index for index, aspect in enumerate(aspects) if aspect >= SIDEWAYS_MELD_ASPECT]
            if len(wide) == 1 and all(aspect <= UPRIGHT_MELD_ASPECT for index, aspect in enumerate(aspects)
                                      if index != wide[0]):
                horizontal, evidence = wide, "tile_aspect"
        if not horizontal and types == ["pon"]:
            blank_geometry = blank_called_position(group["items"])
            if blank_geometry is not None:
                called, geometry = blank_geometry
                horizontal, evidence = [called], "blank_face_relative_geometry"
                group["called_position_geometry"] = geometry
        if not horizontal and types in (["chi"], ["pon"]):
            face_geometry = face_called_position(group["items"])
            if face_geometry is not None:
                called, geometry = face_geometry
                horizontal, evidence = [called], "face_relative_geometry"
                group["called_position_geometry"] = geometry
        target = None
        mapping = {}
        if types == ["chi"]:
            mapping = region.get("chi_source_by_called_position",region.get("source_by_called_position",{}))
        elif types == ["pon"]:
            mapping = region.get("pon_source_by_called_position",region.get("source_by_called_position",{}))
        if len(horizontal) == 1 and str(horizontal[0]) in mapping:
            target = mapping[str(horizontal[0])]
        group.update({"labels": labels, "type": types[0] if len(types) == 1 else None, "type_hypotheses": types,
                      "target": target, "called_positions": horizontal, "called_position_evidence": evidence,
                      "status": "needs_confirmation"})
        if len(types) == 1 and target is not None:
            group["status"] = "confirmed"
        group.pop("right")
    return groups


def _meld_partitions(items: list[dict]) -> list[list[list[dict]]]:
    """Resolve touching groups only when visible identities give one partition.

    This separates e.g. a four-tile 6s group beside two three-tile pons. Unknown
    identities, partial groups and ambiguous partitions remain one unresolved
    group; a four-tile group still needs independent kan-structure evidence.
    """
    if not items:
        return [[]]
    if len(items) > 16:
        return []
    partitions = []
    for count in (3,4):
        if len(items) < count:
            continue
        head = items[:count]
        labels = [base(item["label"]) for item in head]
        same = len(set(labels)) == 1
        chi = (count == 3 and all(len(t) == 2 and t[0].isdigit() for t in labels) and
               len({t[1] for t in labels}) == 1 and
               sorted(int(t[0]) for t in labels) == list(range(min(int(t[0]) for t in labels),min(int(t[0]) for t in labels)+3)))
        # Proposals may establish a unique boundary without confirming a tile.
        # Identity recovery still runs separately under its existing policy.
        plausible = False
        if "?" in labels and "back" not in labels:
            from .kan import row_geometry
            proposed = [item["label"] if item["label"]!="?" else (item.get("best_label") or "?")
                        if item.get("reason") in (None,"pjura_low_confidence")
                        and (item.get("score") or 0)>=MELD_SETTLE_MIN_SCORE
                        and (item.get("margin") or 0)>=MELD_SETTLE_MIN_MARGIN else "?" for item in head]
            plausible = bool(meld_types(proposed)) and row_geometry(head)
        if (((same or chi) and not {"?","back"}.intersection(labels)) or plausible
                or (count == 4 and closed_kan_labels(head) is not None)):
            for rest in _meld_partitions(items[count:]):
                partitions.append([head,*rest])
                if len(partitions) > 1:
                    return partitions
    return partitions


def overlay(pixels: np.ndarray, observation: dict) -> np.ndarray:
    size = observation.get("reference_size")
    if size and [pixels.shape[1], pixels.shape[0]] != size:
        # Boxes are in calibrated coordinates; draw them on a calibrated copy.
        image = cv2.resize(pixels, tuple(size), interpolation=cv2.INTER_AREA)
    else:
        image = pixels.copy()
    for name, region in observation["regions"].items():
        x, y, w, h = region["bbox"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (190, 140, 40), 1)
        cv2.putText(image, name, (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 220, 40), 1)
        for item in region["items"]:
            bx, by, bw, bh = item["bbox"]
            color = (0, 220, 60) if item["label"] != "?" else (0, 80, 255)
            cv2.rectangle(image, (bx, by), (bx+bw, by+bh), color, 2)
            score = item.get("score")
            label = f"{item['label']} {score:.2f}" if score is not None else item["label"]
            cv2.putText(image, label, (bx, max(12, by-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
            if "face_quad_global" in item:
                cv2.polylines(image,[np.rint(item["face_quad_global"]).astype(np.int32)],True,(230,120,220),1)
        for item in region.get("boundary_fragments", []):
            bx,by,bw,bh = item["bbox"]
            cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(0,80,255),2)
            cv2.putText(image,"clipped ?",(bx,max(12,by-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,80,255),1)
        for group in region.get("groups", []):
            if group.get("type") != "ankan" or group.get("status") != "confirmed":
                continue
            for item in group["items"]:
                if item["label"] == "back":
                    bx,by,bw,bh = item["bbox"]
                    cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(240,180,60),2)
                    cv2.putText(image,"back",(bx,max(12,by-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,180,60),1)
            left = min(i["bbox"][0] for i in group["items"])
            bottom = max(i["bbox"][1]+i["bbox"][3] for i in group["items"])
            cv2.putText(image,f"ankan {base(group['labels'][0])} x4",(left,bottom+14),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(240,180,60),1)
        for item in region.get("background_candidates", []):
            bx,by,bw,bh = item["bbox"]
            cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(150,150,150),1)
            cv2.putText(image,"bg",(bx,max(12,by-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(150,150,150),1)
        for item in region.get("unresolved_components", []):
            bx,by,bw,bh = item["bbox"]
            cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(0,80,255),2)
            cv2.putText(image,"foreground ?",(bx,max(12,by-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,80,255),1)
    return image
