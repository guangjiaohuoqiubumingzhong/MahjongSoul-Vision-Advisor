from __future__ import annotations

import cv2
import numpy as np


def blue_back_evidence(pixels):
    """Positive blue-pattern evidence inside an already detected tile body.

    The legacy upright back templates do not cover the foreshortened backs of
    an opponent's closed kan. A blue back has many pale blue pixels; normal
    black/red/green tile glyphs do not. This only names a back, never its tile.
    """
    if not pixels.size or min(pixels.shape[:2]) < 15:
        return None
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0] >= 95) & (hsv[:,:,0] <= 115) & (hsv[:,:,1] > 30)
            & (hsv[:,:,1] < 110) & (hsv[:,:,2] > 175))
    light = (hsv[:,:,1] <= 110) & (hsv[:,:,2] >= 175)
    if blue.sum() < 40 or not .08 <= blue.mean() <= .4 or light.mean() < .55:
        return None
    return {"source":"visible_blue_back_pattern", "blue_pixels":int(blue.sum()),
            "blue_fraction":float(blue.mean()), "light_fraction":float(light.mean())}


def orange_back_mask(pixels):
    hsv = cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
    return (((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 35) & (hsv[:,:,1] >= 120)
             & (hsv[:,:,2] >= 145)).astype(np.uint8)*255)


def orange_back_evidence(pixels):
    """A solid orange back with its neutral side, never just an orange rim."""
    if not pixels.size or min(pixels.shape[:2]) < 15:
        return None
    orange = orange_back_mask(pixels)>0
    neutral = light_mask(pixels,40,145)>0
    if orange.mean() < .6 or (orange.mean() < .85 and neutral.mean() < .02):
        return None
    return {"source":"solid_orange_back", "orange_fraction":float(orange.mean()),
            "neutral_fraction":float(neutral.mean())}


def concealed_back_mask(pixels, corridor, region=None):
    """Exclude evidenced tile backs inside a calibrated concealed-hand corridor.

    A corridor alone never removes foreground: visible melds can occupy the
    same coordinates after calls. Require a blue pattern or a dominant solid
    orange back, rather than treating the white side of an orange back as a face.
    """
    mask = np.zeros(pixels.shape[:2],np.uint8)
    hsv = cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0] >= 95) & (hsv[:,:,0] <= 115) & (hsv[:,:,1] > 30)
            & (hsv[:,:,1] < 110) & (hsv[:,:,2] > 175))
    region = region or {}
    body = light_mask(pixels,region.get("body_saturation_max",100),region.get("body_value_min",145))
    contours,_ = cv2.findContours(body,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if (min(w,h) >= 15 and np.mean(corridor[y:y+h,x:x+w] > 0) > .5
                and blue[y:y+h,x:x+w].mean() > .025 and blue[y:y+h,x:x+w].sum() >= 20):
            mask[y:y+h,x:x+w] = 255
    orange = ((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 35) & (hsv[:,:,1] >= 120)
              & (hsv[:,:,2] >= 145))
    # Join the coloured back to its white side. These sides are individually
    # large enough to become false meld candidates on the default orange skin.
    joined = cv2.morphologyEx(np.maximum(body, orange.astype(np.uint8)*255),
                             cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(joined)
    for index in range(1,count):
        x,y,w,h,area = stats[index]
        if min(w,h) < 15 or area < 100:
            continue
        component = components[y:y+h,x:x+w] == index
        if (orange[y:y+h,x:x+w][component].mean() < .6
                or np.mean(corridor[y:y+h,x:x+w][component] > 0) < .8):
            continue
        silhouette = component.astype(np.uint8)*255
        outlines,_ = cv2.findContours(silhouette,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask[y:y+h,x:x+w],outlines,-1,255,cv2.FILLED)
    return mask


def tile_boxes(pixels: np.ndarray, region: dict) -> list[list[int]]:
    return tile_candidates(pixels,region)["boxes"]


# Dense green faces (8s, 9s, 7s seen sideways in a side river) break the
# light mask into a ragged outline: measured contour fill 0.24-0.29 while the
# convex hull still covers 0.56-0.91 of a box exactly one lattice cell in
# size. Such a component is a tile, not table art.
DENSE_FACE_HULL_FILL = .5
DENSE_FACE_SIZE = ((.75, 1.3), (.6, 1.15))  # of column pitch, of row pitch


def ink_dense_tile(box, contour, grid):
    if not grid:
        return False
    x, y, w, h = box
    column, row = (float(np.linalg.norm(vector)) for vector in grid["basis"])
    (wl, wh), (hl, hh) = DENSE_FACE_SIZE
    if not (wl * column <= w <= wh * column and hl * row <= h <= hh * row):
        return False
    return cv2.contourArea(cv2.convexHull(contour)) / (w * h) >= DENSE_FACE_HULL_FILL


def tile_candidates(pixels: np.ndarray, region: dict, *, exclude_boxes=(), exclude_mask=None) -> dict:
    """Keep oversized foreground evidence instead of treating it as empty table."""
    if "slots" in region:
        boxes = []
        for box in region["slots"]:
            x, y, w, h = box
            patch = pixels[y:y+h, x:x+w]
            if not patch.size or patch.shape[:2] != (h, w):
                continue
            mask = light_mask(patch,region.get("body_saturation_max",100),region.get("body_value_min",145))
            if np.count_nonzero(mask) / (w*h) <= 0.45:
                continue
            bands = region.get("body_presence_bands", [])
            if any(np.mean(mask[round(start*h):round(end*h)] > 0) <= .45 for start,end in bands):
                continue
            boxes.append(box)
        return {"boxes":boxes,"unresolved":[]}
    mask = light_mask(pixels,region.get("body_saturation_max",100),region.get("body_value_min",145))
    coloured = orange_back_mask(pixels) if region.get("kind") == "meld" and "orange" in region.get("kan_back_colours",[]) else None
    if exclude_mask is not None:
        if exclude_mask.shape != mask.shape:
            raise ValueError("Concealed-hand exclusion mask shape mismatch")
        mask[exclude_mask > 0] = 0
        if coloured is not None:coloured[exclude_mask > 0]=0
    for x,y,w,h in exclude_boxes:
        left,top,right,bottom = max(0,x),max(0,y),min(mask.shape[1],x+w),min(mask.shape[0],y+h)
        if right>left and bottom>top:
            mask[top:bottom,left:right] = 0
            if coloured is not None:coloured[top:bottom,left:right]=0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w, min_h = region.get("min_size", [15, 20])
    max_w, max_h = region.get("max_size", [100, 130])
    boxes, unresolved = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < min_h or w < min_w:
            continue
        pieces = split_by_grid([x, y, w, h], region.get("grid"))
        if len(pieces) > 1:
            # Touching river tiles merge into one light blob; the calibrated
            # lattice pitch says how many tiles the blob spans.
            boxes.extend(pieces)
            continue
        if h > max_h:
            unresolved.append({"bbox":[x,y,w,h],"label":"?","reason":"component_outside_tile_size"})
            continue
        if w <= max_w:
            if cv2.contourArea(contour) / (w*h) >= region.get("min_fill", 0.45):
                boxes.append([x, y, w, h])
            elif ink_dense_tile([x, y, w, h], contour, region.get("grid")):
                boxes.append([x, y, w, h])
            continue
        # Touching hand tiles: separators recur at their calibrated body width.
        nominal = region.get("tile_width")
        if not nominal:
            unresolved.append({"bbox":[x,y,w,h],"label":"?","reason":"unsplit_wide_component"})
            continue
        count = round(w / nominal)
        if count < 2 or abs(w / count - nominal) > nominal * .15:
            unresolved.append({"bbox":[x,y,w,h],"label":"?","reason":"unconfirmed_tile_separators"})
            continue
        for index in range(count):
            left, right = round(x + index*w/count), round(x+(index+1)*w/count)
            boxes.append([left, y, right-left, h])
    if coloured is not None:
        # Keep colour silhouettes separate: joining an orange back to the
        # neighbouring white face would swallow that face into one component.
        coloured=cv2.morphologyEx(coloured,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        outlines,_=cv2.findContours(coloured,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in outlines:
            x,y,w,h=cv2.boundingRect(contour)
            if (min_w<=w<=max_w and min_h<=h<=max_h and
                    cv2.contourArea(contour)/(w*h)>=.75 and
                    orange_back_evidence(pixels[y:y+h,x:x+w]) is not None):
                boxes.append([x,y,w,h])
    split_faces = []
    if region.get('kind') == 'meld' and region.get('split_touching_meld_faces'):
        from .faces import tile_face,HIGHLIGHT_FACE_PARAMETERS
        separated=[]
        for box in boxes:
            x,y,w,h=box
            neighbours=[bw for bx,by,bw,bh in boxes if [bx,by,bw,bh]!=box and
                        bw/bh <= .85 and .75*h <= bh <= 1.3*h and
                        min(abs(bx+bw-x),abs(x+w-bx)) <= region.get('group_gap',18)]
            nominal=float(np.median(neighbours)) if neighbours else None
            if nominal is None or not 1.75*nominal <= w <= 2.3*nominal or w/h > 1.2:
                separated.append(box);continue
            middle=round(w/2)
            parts=[[x,y,middle,h],[x+middle,y,w-middle,h]]
            features=[tile_face(pixels[py:py+ph,px:px+pw],HIGHLIGHT_FACE_PARAMETERS) for px,py,pw,ph in parts]
            if all(f is not None and f.area_fraction >= .45 and
                   np.ptp(f.quad[:,0]) >= .65*part[2] and np.ptp(f.quad[:,1]) >= .5*h
                   for f,part in zip(features,parts)):
                # Discard the narrow neighbour sliver below a sloped face;
                # it must not inflate one tile's baseline to the next tile's.
                trimmed=[]
                for px,py,pw,ph in parts:
                    rows=np.flatnonzero(np.mean(light_mask(pixels[py:py+ph,px:px+pw],
                        region.get('body_saturation_max',100),region.get('body_value_min',145))>0,axis=1)>=.3)
                    if not len(rows):break
                    trimmed.append([px,py+int(rows[0]),pw,int(rows[-1]-rows[0]+1)])
                if len(trimmed)==2:
                    separated.extend(trimmed)
                    split_faces.append({'bbox':box,'parts':trimmed,'reason':'two_fitted_faces_at_neighbour_body_width'})
                else:separated.append(box)
            else:separated.append(box)
        boxes=separated
    merged_fragments = []
    if region.get("kind") in ("river", "meld"):
        # Melds too: the red dot of a sideways 5p splits off a small island
        # inside the tile (measured 27x21 inside 77x54).
        boxes, merged_fragments = merge_river_fragments(boxes, region)
    neighbour_crops = []
    if region.get('kind') == 'river' and region.get('grid',{}).get('basis'):
        boxes,neighbour_crops=trim_river_neighbour_sides(boxes,region['grid'])
    result = {"boxes":sorted(boxes,key=lambda box:(round(box[1]/max(1,min_h)),box[0])),"unresolved":unresolved}
    if merged_fragments:
        result["merged_fragments"] = merged_fragments
    if split_faces:result['split_meld_faces']=split_faces
    if neighbour_crops:result['neighbour_crops']=neighbour_crops
    return result


def trim_river_neighbour_sides(boxes,grid):
    """Keep a neighbouring riichi face out of an upright tile's body crop.

    Side-river body boxes include grey thickness extending under the next
    tile. A sideways neighbour can occupy that overlap with an unrelated
    glyph. Only trim a small overlap between adjacent cells of the same row;
    keep the original body as evidence and leave large occlusions unresolved.
    """
    pitch=float(np.linalg.norm(grid['basis'][0]))
    readings=[river_slot(box,grid) for box in boxes]
    slots=[reading[0] for reading in readings]
    result=[];evidence=[]
    for index,box in enumerate(boxes):
        x,y,w,h=box;trimmed=box
        row,column=slots[index]
        if (w<h and 0<=row<grid.get('rows',100) and 0<=column<grid.get('columns',6)-1
                and readings[index][1]<=grid.get('tolerance',10)):
            for other,neighbour in enumerate(boxes):
                nx,ny,nw,nh=neighbour
                if (slots[other]!=[slots[index][0],slots[index][1]+1] or
                        readings[other][1]>grid.get('tolerance',10) or
                        nw < SIDEWAYS_WIDTH_RATIO*pitch or nw < SIDEWAYS_ASPECT*nh):
                    continue
                overlap=x+w-nx
                vertical=max(0,min(y+h,ny+nh)-max(y,ny))
                cut=nx+max(1,round(.06*pitch))
                if (.15*w<=overlap<=.4*w and vertical>=.7*h and
                        .75*pitch<=cut-x<w):
                    trimmed=[x,y,cut-x,h]
                    evidence.append({'bbox':box,'cropped_bbox':trimmed,'neighbour_bbox':neighbour,
                                     'reason':'adjacent_riichi_face_overlaps_body_side'})
                    break
        result.append(trimmed)
    return result,evidence


def merge_river_fragments(boxes: list[list[int]], region: dict) -> tuple[list[list[int]], list[dict]]:
    """Reunite small white islands inside one river tile before classification.

    Dense glyphs plus opening can disconnect the white face. RETR_EXTERNAL then
    returns separate contours whose bounding boxes almost contain each other.
    Use coverage of the smaller box rather than IoU: a small island has little
    IoU with its full face. Require a clear size difference and a union that is
    still a single tile, so neighbouring tiles and riichi bodies stay separate.
    """
    pending = sorted((list(box) for box in boxes), key=lambda box: (-box[2]*box[3], box[1], box[0]))
    max_w, max_h = region.get("max_size", [100, 130])
    bodies, merged = [], []
    while pending:
        body = pending.pop(0)
        separate = []
        for fragment in pending:
            x, y, w, h = body
            fx, fy, fw, fh = fragment
            intersection = max(0, min(x+w, fx+fw)-max(x, fx)) * max(0, min(y+h, fy+fh)-max(y, fy))
            left, top = min(x, fx), min(y, fy)
            union = [left, top, max(x+w, fx+fw)-left, max(y+h, fy+fh)-top]
            if (fw*fh <= .5*w*h and intersection >= .9*fw*fh
                    and union[2] <= max_w and union[3] <= max_h
                    and len(split_by_grid(union, region.get("grid"))) == 1):
                merged.append({"bbox":fragment, "body_bbox":body,
                               "merged_bbox":union, "reason":"contained_face_fragment"})
                body = union
            else:
                separate.append(fragment)
        bodies.append(body)
        pending = separate
    return bodies, merged


GRID_SPLIT_RATIO = 1.6
UPRIGHT_HEIGHT_RATIO = .7


def split_by_grid(box: list[int], grid: dict | None) -> list[list[int]]:
    """Split a blob of touching tiles at the calibrated river lattice pitch.

    A single upright tile is narrower than one column pitch and a sideways
    (riichi) tile is shorter than one row pitch, so a blob at least 1.6 pitches
    wide *and* of upright height, or 1.6 row pitches tall, holds several tiles.
    Anything else is returned unchanged.
    """
    if not grid or "basis" not in grid:
        return [box]
    x, y, w, h = box
    column_pitch = float(np.linalg.norm(grid["basis"][0]))
    row_pitch = float(np.linalg.norm(grid["basis"][1]))
    columns = rows = 1
    # In the side rivers a single riichi body can be 51x44: its grey thickness
    # makes it taller than 0.7 row pitches. Upright bodies still reach nearly
    # the full 53px row pitch. Other river projections retain the existing floor.
    height_ratio=.9 if row_pitch>column_pitch else UPRIGHT_HEIGHT_RATIO
    if h >= height_ratio * row_pitch and w >= GRID_SPLIT_RATIO * column_pitch:
        columns = max(1, int(round(w / column_pitch)))
    if h >= GRID_SPLIT_RATIO * row_pitch:
        rows = max(1, int(round(h / row_pitch)))
    if columns == rows == 1:
        return [box]
    pieces = []
    for row in range(rows):
        top, bottom = round(y + row*h/rows), round(y + (row+1)*h/rows)
        for column in range(columns):
            left, right = round(x + column*w/columns), round(x + (column+1)*w/columns)
            pieces.append([left, top, right-left, bottom-top])
    return pieces


SIDEWAYS_WIDTH_RATIO = 1.25
# Fallback when the classifier cannot confirm a tile's orientation: a sideways
# tile is wider than tall (measured: top river 53x35, own river 57x47 with its
# side visible), while perspective alone widens the first upright tile of a
# side river (43x56 measured) but never makes it wider than tall.
SIDEWAYS_ASPECT = 1.1


def river_slots(boxes: list[list[int]], grid: dict, horizontal=None) -> list[tuple[list[int], float]]:
    """Slots for every box of one river, allowing for sideways (riichi) tiles.

    A sideways tile is wider than one column pitch, so it and every later tile
    in its row sit further along the row than the lattice predicts. The extra
    width is measured from the tile itself and taken back before projection.
    """
    first = [river_slot(box, grid) for box in boxes]
    if "basis" not in grid:
        return first
    column_vector = np.asarray(grid["basis"][0], dtype=float)
    pitch = float(np.linalg.norm(column_vector))
    unit = column_vector / pitch
    if horizontal is not None and len(horizontal) != len(boxes):
        raise ValueError("River orientation evidence must match the detected boxes")
    sideways = [index for index, box in enumerate(boxes)
                if (horizontal[index] is True if horizontal is not None and horizontal[index] is not None
                    else box[2] >= SIDEWAYS_WIDTH_RATIO * pitch and box[2] >= SIDEWAYS_ASPECT * box[3])]
    if not sideways:
        return first
    offsets = [0.0] * len(boxes)
    for index in sideways:
        row = first[index][0][0]
        excess = boxes[index][2] - pitch
        centre = boxes[index][0] + boxes[index][2] / 2
        for other, box in enumerate(boxes):
            if other == index:
                offsets[other] += excess / 2
            elif first[other][0][0] == row and box[0] + box[2] / 2 > centre:
                offsets[other] += excess
    results = [river_slot(box, grid, offset * unit) if offset else result
               for box, offset, result in zip(boxes, offsets, first)]
    # A sideways tile also sits off the upright tiles' centre line across the
    # row; its own extra width bounds that displacement.
    for index in sideways:
        slot, residual = results[index]
        results[index] = (slot, max(0.0, residual - (boxes[index][2] - pitch) / 2))
    return results


def river_slot(box: list[int], grid: dict, shift=None) -> tuple[list[int], float]:
    """Project a tile onto a calibrated two-dimensional lattice without closing gaps.

    Side rivers have sloping rows. A centre-based basis handles that geometry and
    keeps the position of a sideways riichi tile independent of classifier output.
    The original axis-aligned top-left grid remains supported.
    """
    x, y, w, h = box
    point = np.array([x+w/2, y+h/2] if grid.get("anchor") == "center" else [x,y], dtype=float)
    if shift is not None:
        point = point - np.asarray(shift, dtype=float)
    origin = np.asarray(grid["origin"], dtype=float)
    if "basis" in grid:
        column_vector, row_vector = np.asarray(grid["basis"], dtype=float)
        matrix = np.column_stack((column_vector, row_vector))
    else:
        matrix = np.diag(grid["pitch"])
    column, row = np.rint(np.linalg.solve(matrix, point-origin)).astype(int)
    residual = float(np.max(np.abs(point-origin-matrix@np.array([column,row]))))
    return [int(row), int(column)], residual


def light_mask(pixels, saturation_max=100, value_min=145):
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, value_min], dtype=np.uint8), np.array([179, saturation_max, 255], dtype=np.uint8))
    # Closing joins the narrow dark separators between neighbouring river tiles.
    # Keep those gaps; disconnected glyph islands are reunited per river body.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))


def global_box(box, origin, unrotated_size, degrees):
    x, y, w, h = box
    width, height = unrotated_size
    if degrees == 90:
        x, y, w, h = width-y-h, x, h, w
    elif degrees == 180:
        x, y = width-x-w, height-y-h
    elif degrees == 270:
        x, y, w, h = y, height-x-w, h, w
    return [x+origin[0], y+origin[1], w, h]
