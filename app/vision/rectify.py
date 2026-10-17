"""Tile-face cropping and perspective correction.

A mahjong tile seen on the table is not a front-facing rectangle. Tiles in the
side rivers are rotated a quarter turn in the image, and every tile that is not
directly below the camera is photographed at an angle, so its visible face is a
skewed quadrilateral whose apparent size and aspect ratio depend on where it
lies on the table. Resizing the detector's axis-aligned box to a fixed size
therefore feeds the classifier a sheared, foreshortened picture of the glyph,
and the score of a sideways tile drops even though the glyph is unambiguous.

This module recovers the tile's geometry from the crop itself:

* the outer silhouette is fitted with a rotated rectangle (:func:`min_area_quad`),
  which is stable against the rounded corners and the dark side face;
* the visible white face is fitted with a quadrilateral
  (:func:`face_quad`) when it can be outlined confidently;
* the fitted quadrilateral is ordered by its own edges
  (:func:`canonical_order`) so that the tile's long axis becomes the canonical
  vertical axis regardless of how the tile is turned in the image;
* :func:`rectify_tile` warps the quadrilateral to a fixed-size upright face and
  also expands it to the tile's full body rectangle.

The silhouette above includes the tile's visible *side* (the grey thickness
strip under or beside the face), so warping it leaves the glyph off-centre and
sheared. :func:`top_face_outline` / :func:`warp_top_face` are the correction
used for classification: they separate the bright top face from the darker
side face, fit only that face, and warp it to an upright crop shaped like the
frontal hand tiles the classifier already reads reliably. Each requested
quarter turn gets its own warp, so a sideways (called / riichi) tile is
flattened to a landscape rectangle first and then turned upright instead of
being squashed into portrait proportions.

Nothing here decides tile identity. Geometry only says how the tile is lying in
the image, never which tile it is, so a failed fit returns ``None`` instead of a
guess and the caller must fall back to the uncorrected crop.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


RECTIFY_PARAMETERS = {
    "warp_size": [40, 60],
    "background_percentile": 85,
    "minimum_background": 64,
    "minimum_crop_side": 8,
    # Silhouette fit: the bright body mask refuses to join the dark side face,
    # so its rotated bounding box is the visible front face plus its top strip.
    "silhouette_value_min": 145,
    "silhouette_saturation_max": 100,
    "silhouette_open_kernel": 3,
    "minimum_silhouette_area_fraction": .35,
    # White-face fit: a colour-ratio mask keeps the white base of a tile apart
    # from the blue table felt, and the polygon is simplified relative to its
    # own outline length so a rounded corner does not turn into two corners.
    "face_channel_min": .80,
    "face_minimum_area_fraction": .12,
    "face_trim": 1,
    "source_scale": 3,
    # Top-face fit (see top_face_outline). Whiteness is the per-pixel minimum
    # channel after dividing by the crop's bright background, so table tint and
    # highlight overlays do not change it.
    "whiteness_background_percentile": 90,
    "whiteness_floor": .60,
    # The side face is noticeably greyer than the top face; split the two only
    # when Otsu finds two populations that are clearly apart.
    "face_side_separation": .07,
    "face_side_minimum_share": .08,
    "face_top_minimum_share": .20,
    # Glyph strokes cut the white face into pieces; a closing this fraction of
    # the crop's short side bridges them so the whole face is one region.
    "face_bridge_fraction": .12,
    "face_top_minimum_area_fraction": .25,
    # Region hull area / fitted quad area. Outside this range the outline is not
    # a single four-cornered face (two faces merged, or a corner clipped).
    "face_hull_fill_range": [.90, 1.12],
    "face_minimum_edge": 4,
    # A lone white piece holding this share of the merged white pixels is the
    # face; the rest are neighbour slivers across a tile seam. Glyph-split
    # faces break into comparable pieces and stay merged.
    "face_core_share": .70,
    # Upright output in the proportions of a detected hand/dora tile body.
    "face_output_size": [64, 94],
    "face_output_inset": .03,
}


@dataclass
class TileGeometry:
    """The tile's outline in one crop, plus the crop's metadata."""

    quad: np.ndarray          # 4x2 float32 (top_face: image order TL, TR, BR, BL)
    source: str               # "silhouette", "face" or "top_face"
    area_fraction: float
    aspect: float             # |long edge| / |short edge| as measured
    hull_fill: float | None = None

    @property
    def portrait(self) -> bool:
        return self.aspect >= 1.0

    def as_dict(self) -> dict:
        value = {"source": self.source, "area_fraction": float(self.area_fraction),
                 "aspect": float(self.aspect), "quad": self.quad.astype(float).tolist()}
        if self.hull_fill is not None:
            value["hull_fill"] = float(self.hull_fill)
        return value


@dataclass
class RectifiedTile:
    """A perspective-corrected tile crop in the classifier's canonical pose."""

    pixels: np.ndarray        # uint8 BGR, canonical upright tile at body resolution
    normalized: np.ndarray    # float32, background-normalized, same pose
    quad: np.ndarray
    transform: np.ndarray     # 3x3 homography mapping the warp target to the crop
    rotation: int             # quarter turns applied to make the long axis vertical
    geometry: TileGeometry

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four corners so the tile's long axis becomes the first edge.

    The starting corner is the one nearest the top of the crop and then the
    left, which keeps the mapping stable for tiles at any rotation: a tile
    turned a quarter turn in the image still yields the same canonical order as
    the same tile photographed upright.
    """
    points = np.asarray(points, np.float32).reshape(4, 2)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    long_start = int(np.argmax(lengths))
    # Trace the cycle from the corner before the longest edge.
    cycle = np.roll(points, -(long_start + 1) % 4, axis=0)
    long_length = float(np.linalg.norm(cycle[1] - cycle[0]))
    short_length = float(np.linalg.norm(cycle[2] - cycle[1]))
    if short_length > long_length:  # defensive: keep the long axis first
        cycle = np.roll(cycle, 1, axis=0)
        long_length, short_length = short_length, long_length
    top = cycle[:2]
    if top[1][1] < top[0][1] or (top[1][1] == top[0][1] and top[1][0] < top[0][0]):
        cycle = cycle[[1, 0, 3, 2]]
    return np.ascontiguousarray(cycle, np.float32)


def quad_aspect(quad: np.ndarray) -> float:
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    long_edge = np.linalg.norm(quad[1] - quad[0])
    short_edge = np.linalg.norm(quad[2] - quad[1])
    return float(long_edge / short_edge) if short_edge > 1e-6 else 0.0


def quad_skew(quad: np.ndarray) -> float:
    """How much a fitted outline tapers, as two opposite edges compare.

    A tile merely turned in the image has equal opposite edges, so this measures
    the trapezoid an oblique camera produces and not the tile's rotation. The
    result is the larger of the two relative differences, so 0 means the outline
    is a parallelogram under the current crop.
    """
    quad = np.asarray(quad, np.float64).reshape(4, 2)
    pairs = ((np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])),
             (np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    skew = 0.0
    for first, second in pairs:
        longest = max(first, second)
        if longest > 1e-6:
            skew = max(skew, abs(first - second) / longest)
    return float(skew)


def quad_ratio(quad: np.ndarray) -> float:
    """Long-edge to short-edge ratio of a fitted outline, whatever its turn."""
    quad = np.asarray(quad, np.float64).reshape(4, 2)
    first = .5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
    second = .5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
    long_edge, short_edge = max(first, second), min(first, second)
    return float(long_edge / short_edge) if short_edge > 1e-6 else 0.0


def quad_from_points(points: np.ndarray) -> np.ndarray | None:
    """Fit a quadrilateral to an outline, so a trapezoid stays a trapezoid.

    A bounding rectangle cannot represent the taper an oblique camera gives a
    tile. The outline is simplified against its own perimeter with a growing
    tolerance until exactly four corners remain; the tolerance starts small
    enough to keep a real taper and only grows far enough to absorb the rounded
    corner and anti-aliasing of a tile's edge.
    """
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if len(points) < 4:
        return None
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    if len(hull) == 4:
        return hull.astype(np.float32)
    if len(hull) < 4:
        return None
    perimeter = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    for tolerance in (0.01, 0.02, 0.04, 0.08, 0.15, 0.25):
        polygon = cv2.approxPolyDP(hull.reshape(-1, 1, 2), tolerance * perimeter, True).reshape(-1, 2).astype(np.float32)
        if len(polygon) == 4:
            return polygon
        if len(polygon) < 4:
            break
    # A concave or five-sided simplification has no four-corner reading; the
    # caller must treat the outline as unusable rather than invent one.
    rectangle = cv2.minAreaRect(hull.reshape(-1, 1, 2))
    if min(rectangle[1]) < 2:
        return None
    return None


def min_area_quad(patch: np.ndarray, parameters: dict | None = None) -> TileGeometry | None:
    """Fit the tile's outer silhouette as a quadrilateral.

    Bright tile bodies survive the mask; the blue felt and the dark side face do
    not. The fitted outline is the visible front face together with the narrow
    top strip the camera sees, which is the geometry the body templates were
    captured from, and it keeps its taper instead of being replaced by a
    bounding rectangle.
    """
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    if min(patch.shape[:2]) < parameters["minimum_crop_side"] or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, parameters["silhouette_value_min"]], np.uint8),
                       np.array([179, parameters["silhouette_saturation_max"], 255], np.uint8))
    kernel = parameters["silhouette_open_kernel"]
    if kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area_fraction = float(cv2.contourArea(contour) / (patch.shape[0] * patch.shape[1]))
    if area_fraction < parameters["minimum_silhouette_area_fraction"]:
        return None
    quad = quad_from_points(contour.reshape(-1, 2))
    if quad is None:
        return None
    ordered = order_quad(quad)
    return TileGeometry(ordered, "silhouette", area_fraction, quad_aspect(ordered))


def face_quad(patch: np.ndarray, parameters: dict | None = None) -> TileGeometry | None:
    """Fit the tile's visible white face as a quadrilateral.

    The white base is separated by channel ratio rather than a single brightness
    threshold, so the blue felt and the pinkish highlight overlays that the game
    paints on a tile do not join the face. The polygon is simplified against its
    own outline length, which keeps one corner per rounded tile corner.
    """
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    if min(patch.shape[:2]) < parameters["minimum_crop_side"] or patch.size == 0:
        return None
    background = np.percentile(patch, parameters["background_percentile"], axis=(0, 1)).astype(np.float32)
    normalized = np.clip(patch.astype(np.float32) / np.maximum(background, parameters["minimum_background"]), 0, 1)
    mask = (np.min(normalized, axis=2) >= parameters["face_channel_min"]).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area_fraction = float(cv2.contourArea(contour) / (patch.shape[0] * patch.shape[1]))
    if area_fraction < parameters["face_minimum_area_fraction"]:
        return None
    hull = cv2.convexHull(contour)
    quad = quad_from_points(hull.reshape(-1, 2))
    if quad is None:
        return None
    ordered = order_quad(quad)
    return TileGeometry(ordered, "face", area_fraction, quad_aspect(ordered))


def canonical_order(geometry: TileGeometry) -> np.ndarray:
    return geometry.quad


def _canonical_target(quad: np.ndarray, warp_size: list[int], rotation: int) -> np.ndarray:
    """Destination corners for ``rotation`` quarter turns of the fitted quad.

    ``rotation`` expresses which way round the tile is lying: 0 keeps the fitted
    long axis vertical, 90 and 270 turn it a quarter turn so a tile that the
    image shows sideways is presented upright to the classifier.
    """
    width, height = warp_size
    if rotation in (90, 270):
        width, height = height, width
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    turns = (rotation // 90) % 4
    return np.roll(corners, -turns, axis=0) if turns else corners


def background_normalized(patch: np.ndarray, parameters: dict | None = None) -> np.ndarray:
    """Divide by the crop's bright background so lighting tint stops mattering."""
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    background = np.percentile(patch, parameters["background_percentile"], axis=(0, 1)).astype(np.float32)
    return np.clip(patch.astype(np.float32) / np.maximum(background, parameters["minimum_background"]), 0, 1)


def _trim_to_fraction(image: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return image
    height, width = image.shape[:2]
    top = int(round(fraction * height))
    left = int(round(fraction * width))
    if min(height, width) <= 2 * max(top, left) + 1:
        return image
    return image[top:height - top, left:width - left]


def rectify_tile(patch: np.ndarray, parameters: dict | None = None, *,
                 rotation: int = 0, geometry: TileGeometry | None = None) -> RectifiedTile | None:
    """Warp a fitted tile outline to the classifier's fixed upright size.

    ``geometry`` may be supplied when the caller already fitted the outline (for
    example the white face while the body is warped from the silhouette). The
    returned pixels are background-normalized and use the classifier's BGR
    layout, so they can be passed to the same similarity scoring as whole-body
    templates.
    """
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    geometry = geometry or min_area_quad(patch, parameters) or face_quad(patch, parameters)
    if geometry is None:
        return None
    warp_size = [int(value) for value in parameters["warp_size"]]
    width, height = warp_size
    if rotation in (90, 270):
        width, height = height, width
    transform = cv2.getPerspectiveTransform(geometry.quad, _canonical_target(geometry.quad, warp_size, rotation))
    normalized = background_normalized(patch, parameters)
    feature = cv2.warpPerspective(normalized, transform, (width, height), flags=cv2.INTER_LINEAR)
    source_size = [int(value) for value in (parameters.get("source_size") or [width * 3, height * 3])]
    if rotation in (90, 270):
        source_size = source_size[::-1]
    source_target = _canonical_target(geometry.quad, source_size, rotation)
    source = cv2.warpPerspective(patch, cv2.getPerspectiveTransform(geometry.quad, source_target),
                                 (source_size[0], source_size[1]), flags=cv2.INTER_LINEAR)
    # Both warps are trimmed by the same fraction of their own size so the
    # normalized feature and the 8-bit crop stay pixel-aligned.
    trim = float(parameters["face_trim"]) / max(1, min(width, height))
    feature = np.ascontiguousarray(_trim_to_fraction(feature, trim), np.float32)
    source = np.ascontiguousarray(_trim_to_fraction(source, trim), np.uint8)
    return RectifiedTile(source, feature, geometry.quad, transform, rotation, geometry)


def rectify_tile_aspect(patch: np.ndarray, parameters: dict | None = None, *,
                        rotation: int = 0, geometry: TileGeometry | None = None,
                        source_scale: int | None = None) -> RectifiedTile | None:
    """Warp a fitted outline flat while keeping its measured proportions.

    A fitted outline is a trapezoid when the camera looks along the table plane,
    and resizing cannot undo a trapezoid. This warp removes that distortion and
    leaves the outline's own long/short ratio, so the classifier receives an
    undistorted tile at whatever foreshortening the camera produced. The result
    is not interchangeable with the fixed-size gallery, which is trained on
    canonical 2:3 faces, so this variant is meant for a network that resizes its
    own input.
    """
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    geometry = geometry or min_area_quad(patch, parameters) or face_quad(patch, parameters)
    if geometry is None:
        return None
    quad = np.asarray(geometry.quad, np.float64)
    short_edge = float(np.linalg.norm(quad[2] - quad[1]))
    long_edge = float(np.linalg.norm(quad[1] - quad[0]))
    if short_edge < 1e-6 or long_edge < 1e-6:
        return None
    short_pixels = int(parameters["warp_size"][0])
    long_pixels = max(short_pixels, int(round(short_pixels * long_edge / short_edge)))
    width, height = short_pixels, long_pixels
    if rotation in (90, 270):
        width, height = height, width
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    if (rotation // 90) % 4:
        target = np.roll(target, -((rotation // 90) % 4), axis=0)
    transform = cv2.getPerspectiveTransform(geometry.quad, target)
    feature = cv2.warpPerspective(background_normalized(patch, parameters), transform, (width, height),
                                  flags=cv2.INTER_LINEAR)
    scale = source_scale or parameters.get("source_scale") or 3
    if rotation in (90, 270):
        source_size = (long_pixels * scale, short_pixels * scale)
    else:
        source_size = (short_pixels * scale, long_pixels * scale)
    source_target = target * [scale, scale] + [scale / 2, scale / 2]
    source = cv2.warpPerspective(patch, cv2.getPerspectiveTransform(geometry.quad, source_target.astype(np.float32)),
                                 (int(source_size[0]), int(source_size[1])), flags=cv2.INTER_LINEAR)
    return RectifiedTile(np.ascontiguousarray(source, np.uint8), np.ascontiguousarray(feature, np.float32),
                         geometry.quad, transform, rotation, geometry)


def order_image_quad(points: np.ndarray) -> np.ndarray:
    """Order four corners as they lie in the image: TL, TR, BR, BL.

    Unlike :func:`order_quad` this does not guess which edge is the tile's long
    axis. Foreshortening can make an upright tile's face look wider than tall,
    so the caller decides the turn and the warp keeps the image's own axes.
    """
    points = np.asarray(points, np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    points = points[np.argsort(angles)]  # clockwise on screen (y grows down)
    start = int(np.argmin(points.sum(axis=1)))
    return np.ascontiguousarray(np.roll(points, -start, axis=0), np.float32)


def whiteness(patch: np.ndarray, parameters: dict | None = None) -> np.ndarray:
    """Minimum normalized channel: 1 on the white face, lower on grey and ink."""
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    pixels = patch.astype(np.float32)
    background = np.percentile(pixels, parameters["whiteness_background_percentile"], axis=(0, 1))
    return np.clip(pixels / np.maximum(background, parameters["minimum_background"]), 0, 1.2).min(axis=2)


def _top_face_level(values: np.ndarray, parameters: dict) -> float:
    """Whiteness level separating the top face from the greyer side face."""
    floor = float(parameters["whiteness_floor"])
    scaled = np.clip(np.round(values * 255), 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2's Otsu threshold is exclusive: the upper class is strictly above it.
    above = scaled > threshold
    upper, lower = values[above], values[~above]
    if (len(lower) > parameters["face_side_minimum_share"] * len(values)
            and len(upper) > parameters["face_top_minimum_share"] * len(values)
            and float(upper.mean() - lower.mean()) > parameters["face_side_separation"]):
        # Midway between the side's brightest and the face's darkest pixel.
        return max(.5 * (float(lower.max()) + float(upper.min())), floor)
    return floor


def _top_face_geometry(points: np.ndarray, patch: np.ndarray, parameters: dict) -> TileGeometry | None:
    """Fit and validate one four-cornered face outline around ``points``."""
    if points is None or len(points) < 4:
        return None
    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    area_fraction = hull_area / float(patch.shape[0] * patch.shape[1])
    if area_fraction < parameters["face_top_minimum_area_fraction"]:
        return None
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for tolerance in (.02, .03, .04, .06, .08, .10, .13):
        polygon = cv2.approxPolyDP(hull, tolerance * perimeter, True)
        if len(polygon) == 4:
            quad = polygon.reshape(4, 2).astype(np.float32)
            break
        if len(polygon) < 4:
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    quad = order_image_quad(quad)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if float(edges.min()) < parameters["face_minimum_edge"]:
        return None
    quad_area = float(cv2.contourArea(quad.reshape(-1, 1, 2)))
    if quad_area <= 0:
        return None
    fill = hull_area / quad_area
    low, high = parameters["face_hull_fill_range"]
    if not low <= fill <= high:
        return None
    return TileGeometry(quad, "top_face", area_fraction, quad_ratio(quad), fill)


def top_face_outline(patch: np.ndarray, parameters: dict | None = None) -> TileGeometry | None:
    """Fit only the tile's white top face, excluding its visible side.

    Returns ``None`` whenever the outline is not convincingly one four-cornered
    face; the caller must then keep the original crop. The returned quad is in
    image order so a caller can warp it for any quarter turn.
    """
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    if patch is None or patch.size == 0 or patch.ndim != 3 or min(patch.shape[:2]) < parameters["minimum_crop_side"]:
        return None
    white = whiteness(patch, parameters)
    candidates = white >= parameters["whiteness_floor"]
    if np.count_nonzero(candidates) < 20:
        return None
    level = _top_face_level(white[candidates], parameters)
    mask = (white >= level).astype(np.uint8)
    size = max(3, int(round(parameters["face_bridge_fraction"] * min(mask.shape))) | 1)
    bridged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, 8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # The hull is taken over genuinely white pixels only; the closing just
    # decides which pieces belong to the same face.
    union = ((labels == largest) & (mask > 0)).astype(np.uint8)
    merged = _top_face_geometry(cv2.findNonZero(union), patch, parameters)
    # The closing also bridges the thin dark seam to a neighbouring tile that
    # the detector box clipped. When the largest unbridged piece is already a
    # valid face covering nearly all of the merged outline, the extra pieces
    # are such slivers, not glyph-split parts of this face: keep the piece.
    pieces, piece_labels, piece_stats, _ = cv2.connectedComponentsWithStats(union, 8)
    if pieces > 2:
        main = 1 + int(np.argmax(piece_stats[1:, cv2.CC_STAT_AREA]))
        share = piece_stats[main, cv2.CC_STAT_AREA] / float(piece_stats[1:, cv2.CC_STAT_AREA].sum())
        if share >= parameters["face_core_share"]:
            core = _top_face_geometry(cv2.findNonZero((piece_labels == main).astype(np.uint8)), patch, parameters)
            if core is not None:
                return core
    return merged


def warp_top_face(patch: np.ndarray, geometry: TileGeometry, degrees: int = 0,
                  parameters: dict | None = None) -> RectifiedTile:
    """Warp a fitted top face upright for one quarter-turn hypothesis.

    ``degrees`` has the meaning of ``templates.rotate``: ``np.rot90(crop, d/90)``
    shows the tile upright. For 90/270 the face is flattened to a landscape
    rectangle first, so the turn produces true portrait proportions.
    """
    parameters = {**RECTIFY_PARAMETERS, **(parameters or {})}
    width, height = (int(value) for value in parameters["face_output_size"])
    turns = (int(degrees) // 90) % 4
    if turns % 2:
        width, height = height, width
    inset = float(parameters["face_output_inset"])
    dx, dy = inset * width, inset * height
    target = np.array([[dx, dy], [width - 1 - dx, dy], [width - 1 - dx, height - 1 - dy],
                       [dx, height - 1 - dy]], np.float32)
    transform = cv2.getPerspectiveTransform(np.asarray(geometry.quad, np.float32), target)
    warped = cv2.warpPerspective(patch, transform, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    upright = np.ascontiguousarray(np.rot90(warped, turns), np.uint8)
    return RectifiedTile(upright, background_normalized(upright, parameters), geometry.quad,
                         transform, int(degrees) % 360, geometry)


# Red fives: every stroke is red, while ordinary fives mix red with black
# (5m: black 伍, red 萬) or with green/black (5s, 5p). Measured on reviewed
# tiles: red share 1.00 for red fives; ordinary 5m 0.54-0.60, 5p 0.16-0.19,
# 5s 0.21-0.24. The gaps leave room for blur and lighting.
FIVE_COLOUR = {"red_min": .85, "normal_max": {"m": .75, "p": .40, "s": .40}, "minimum_ink": .03, "inset": .08}
# A tile's top face is neutral white: on reviewed tiles B-R lies in [-3, -1]
# and G-R in [-1.3, 1.2]. Table art (stars, feathers, character skin) is not.
FACE_TINT = {"blue_minus_red": (-10.0, 6.0), "green_minus_red": 5.0}


def _face_pixels(patch: np.ndarray, geometry: TileGeometry | None) -> np.ndarray:
    if geometry is not None:
        return warp_top_face(patch, geometry, 0).pixels
    return patch


def five_colour(patch: np.ndarray, geometry: TileGeometry | None = None) -> dict:
    """Share of red ink among all coloured or dark strokes on the tile face."""
    face = _face_pixels(patch, geometry)
    inset = FIVE_COLOUR["inset"]
    dy, dx = int(face.shape[0] * inset), int(face.shape[1] * inset)
    inner = face[dy:face.shape[0]-dy or None, dx:face.shape[1]-dx or None]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ink = (saturation > 70) | (value < 120)
    red = ((hue < 10) | (hue > 165)) & (saturation > 70) & (value > 60)
    count = int(np.count_nonzero(ink))
    return {"red_share": float(np.count_nonzero(red) / count) if count else 0.0,
            "ink_fraction": float(count / ink.size), "rectified": geometry is not None}


def five_verdict(label: str, colour: dict) -> str | None:
    """``"red"``, ``"normal"`` or ``None`` (undecided) for a base five label."""
    if len(label) != 2 or label[0] != "5" or colour["ink_fraction"] < FIVE_COLOUR["minimum_ink"]:
        return None
    if colour["red_share"] >= FIVE_COLOUR["red_min"]:
        return "red"
    if colour["red_share"] <= FIVE_COLOUR["normal_max"].get(label[1], 0):
        return "normal"
    return None


def face_tint(patch: np.ndarray, geometry: TileGeometry | None = None) -> dict:
    """Colour cast of the brightest face pixels; a real tile face is neutral."""
    pixels = patch.astype(np.float32)
    if geometry is not None:
        mask = np.zeros(patch.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.int32(np.round(geometry.quad))], 1)
        values = pixels[mask > 0]
    else:
        values = pixels.reshape(-1, 3)
    if len(values) == 0:
        values = pixels.reshape(-1, 3)
    luminance = values.mean(axis=1)
    bright = values[luminance >= np.percentile(luminance, 70)]
    # A coloured tile edge or helper index can contaminate the bright subset.
    # Its channel mean used to drop a real fitted face as table decoration.
    # The median retains the face's base colour while rejecting solid tinted art.
    blue, green, red = np.median(bright,axis=0)
    low, high = FACE_TINT["blue_minus_red"]
    neutral = low <= blue - red <= high and abs(green - red) <= FACE_TINT["green_minus_red"]
    return {"blue_minus_red": float(blue - red), "green_minus_red": float(green - red), "neutral": bool(neutral)}


def probe_shapes(patch: np.ndarray, region: dict | None = None) -> dict:
    """Report which outlines can be fitted in a crop; measurement helper."""
    parameters = (region or {}).get("tile_rectification")
    silhouette = min_area_quad(patch, parameters)
    face = face_quad(patch, parameters)
    top = top_face_outline(patch, parameters)
    return {"silhouette": silhouette.as_dict() if silhouette else None,
            "face": face.as_dict() if face else None,
            "top_face": top.as_dict() if top else None,
            "silhouette_skew": quad_skew(silhouette.quad) if silhouette else None,
            "face_skew": quad_skew(face.quad) if face else None}


def _classify_crop(bank, pixels: np.ndarray, region: dict) -> dict:
    match = bank.classify(pixels, region.get("kind", "hand"), 0, 0, (0, 90, 180, 270), kinds=("tile",))
    return {"label": match.get("label") or match.get("best_label"),
            "score": (match.get("pjura_evidence") or {}).get("score", match.get("score")),
            "rotation": match.get("identity_rotation", match.get("face_rotation")),
            "reason": match.get("reason")}


def probe_variants(patch: np.ndarray, region: dict | None = None, bank=None) -> dict:
    """Classify one crop as the current ``bbox`` crop and the corrected variants."""
    region = region or {}
    parameters = region.get("tile_rectification")
    result = {"silhouette": None, "face": None, "kept_aspect": None}
    if bank is not None:
        result["bbox"] = _classify_crop(bank, patch, region)
    for name, fitter in (("silhouette", min_area_quad), ("face", face_quad)):
        geometry = fitter(patch, parameters)
        if geometry is None:
            continue
        rectified = rectify_tile(patch, parameters, rotation=0, geometry=geometry)
        if rectified is None or bank is None:
            continue
        result[name] = _classify_crop(bank, rectified.pixels, region)
    geometry = min_area_quad(patch, parameters)
    if geometry is not None:
        preserved = rectify_tile_aspect(patch, parameters, geometry=geometry)
        if preserved is not None and bank is not None:
            result["kept_aspect"] = _classify_crop(bank, preserved.pixels, region)
    return result
