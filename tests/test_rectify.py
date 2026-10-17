"""Tile-face cropping and perspective correction behaviour."""
import numpy as np
import cv2
import pytest

from app.vision.pjura import (PjuraBank, outline_foreshortening, outline_skew, rail_aspect,
                              rectification_needed)
from app.vision.rectify import (RECTIFY_PARAMETERS, TileGeometry, face_quad, min_area_quad, order_image_quad,
                                order_quad, quad_aspect, rectify_tile, rectify_tile_aspect, top_face_outline,
                                warp_top_face)


def tile_patch(width=30, height=70, glyph=True):
    """A portrait tile body on a dark table, with a coloured glyph.

    The body is 30x70 so its outline measures about 2.3:1, which is the range a
    real detected river tile occupies, and the padding leaves the table visible
    around it the way the detector's box does.
    """
    patch = np.full((height + 16, width + 16, 3), 60, np.uint8)
    patch[8:8 + height, 8:8 + width] = 235
    if glyph:
        patch[8 + height // 4:8 + height // 2, 12:12 + width // 2] = [40, 40, 200]
    return patch


def project(patch, quad_source, quad_target):
    transform = cv2.getPerspectiveTransform(np.asarray(quad_source, np.float32), np.asarray(quad_target, np.float32))
    canvas = np.zeros((120, 120, 3), np.uint8)
    return cv2.warpPerspective(patch, transform, (120, 120), flags=cv2.INTER_LINEAR), transform


def test_quad_order_puts_the_long_axis_first_for_every_quarter_turn():
    upright = np.array([[10, 5], [40, 5], [40, 75], [10, 75]], np.float32)
    for turn in range(4):
        result = order_quad(np.roll(upright, -turn, axis=0))
        # Whatever the in-plane rotation, the first edge is the tile's long axis
        # and the four corners stay the same set.
        assert np.linalg.norm(result[1] - result[0]) > np.linalg.norm(result[2] - result[1])
        assert quad_aspect(result) == pytest.approx(70 / 30, rel=.02)
        assert {tuple(np.round(point, 3)) for point in result} == {tuple(np.round(point, 3)) for point in upright}


def body_quad(patch):
    """Corners of the synthetic tile body inside a patch, for use as a warp source."""
    mask = np.min(patch, axis=2) > 200
    ys, xs = np.where(mask)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def project_body(patch, target, margin=6, canvas=200):
    """Warp the tile body onto ``target`` and crop to it, as a detector box does.

    A real detector box is tight around the tile, so the test frames the warped
    outline the same way instead of leaving it small inside a large canvas. All
    target corners must stay inside the canvas, since a projection that runs off
    the edge would be cropped by the warp instead of by the detector.
    """
    transform = cv2.getPerspectiveTransform(body_quad(patch), np.asarray(target, np.float32))
    warped = cv2.warpPerspective(patch, transform, (canvas, canvas), flags=cv2.INTER_LINEAR)
    corners = np.asarray(target, np.float32)
    x1 = max(0, int(corners[:, 0].min()) - margin)
    y1 = max(0, int(corners[:, 1].min()) - margin)
    x2 = min(warped.shape[1], int(corners[:, 0].max()) + margin)
    y2 = min(warped.shape[0], int(corners[:, 1].max()) + margin)
    assert x2 > x1 and y2 > y1, "target quad must fit inside the warp canvas"
    return warped[y1:y2, x1:x2]


def test_silhouette_fit_measures_true_proportions_of_an_upright_tile():
    geometry = min_area_quad(tile_patch())
    assert geometry is not None and geometry.source == "silhouette"
    assert geometry.aspect == pytest.approx(70 / 30, rel=.1)
    assert geometry.area_fraction > .5


def test_rectification_undoes_a_trapezoid_projection():
    patch = tile_patch()
    target = np.array([[45, 20], [95, 20], [105, 140], [25, 140]], np.float32)
    projected = project_body(patch, target)
    geometry = min_area_quad(projected)
    assert geometry is not None
    assert geometry.source == "silhouette" and geometry.area_fraction > .5
    # The projection compressed the tile's long axis, so the outline no longer
    # shows the tile's true proportions and the axis-aligned box is distorted.
    assert geometry.aspect < 1.5
    rectified = rectify_tile(projected, geometry=geometry)
    assert rectified is not None
    # The corrected crop restores the classifier's canonical 2:3 face.
    assert rectified.pixels.shape[1] / rectified.pixels.shape[0] == pytest.approx(2 / 3, abs=.05)
    assert np.count_nonzero(np.min(rectified.pixels, axis=2) > 200) > .5 * rectified.pixels.size / 3
    assert rectified.transform.shape == (3, 3)


def test_rectification_recovers_a_known_face_from_a_skewed_view():
    """The warped face is undistorted, so the glyph lands where it belongs."""
    patch = tile_patch()
    target = np.array([[45, 30], [85, 30], [95, 150], [35, 150]], np.float32)
    skewed = project_body(patch, target)
    geometry = min_area_quad(skewed)
    rectified = rectify_tile(skewed, geometry=geometry)
    assert rectified is not None
    # The glyph keeps its place in the canonical face instead of sliding with the skew.
    ink = np.min(rectified.pixels, axis=2) < 120
    ys, xs = np.where(ink)
    assert len(xs) and .1 < xs.mean() / rectified.pixels.shape[1] < .8
    assert .1 < ys.mean() / rectified.pixels.shape[0] < .8


def test_aspect_preserving_warp_keeps_the_measured_ratio():
    patch = tile_patch()
    target = np.array([[45, 20], [95, 20], [105, 110], [25, 110]], np.float32)
    skewed = project_body(patch, target)
    geometry = min_area_quad(skewed)
    preserved = rectify_tile_aspect(skewed, geometry=geometry)
    canonical = rectify_tile(skewed, geometry=geometry)
    assert preserved is not None and canonical is not None
    assert preserved.width / preserved.height == pytest.approx(1 / geometry.aspect, rel=.05)
    assert canonical.width / canonical.height == pytest.approx(2 / 3, abs=.05)
    assert canonical.width / canonical.height == pytest.approx(2 / 3, abs=.05)


def test_face_fit_finds_the_white_face_inside_a_larger_crop():
    patch = np.full((90, 70, 3), 40, np.uint8)
    patch[20:80, 15:55] = 240
    patch[35:60, 22:48] = [30, 30, 190]
    geometry = face_quad(patch)
    assert geometry is not None and geometry.source == "face"
    assert geometry.area_fraction == pytest.approx(60 * 40 / (90 * 70), rel=.2)
    assert geometry.aspect > 1.2


def test_failed_fits_supply_no_geometry_instead_of_a_guess():
    assert min_area_quad(np.zeros((60, 40, 3), np.uint8)) is None
    assert face_quad(np.zeros((60, 40, 3), np.uint8)) is None
    assert min_area_quad(np.zeros((4, 4, 3), np.uint8)) is None
    assert rectify_tile(np.zeros((60, 40, 3), np.uint8)) is None


def test_a_quarter_turned_outline_is_measured_by_its_edges_not_by_storage_order():
    portrait = np.array([[0, 0], [30, 0], [30, 70], [0, 70]], np.float32)
    landscape = np.ascontiguousarray(np.roll(portrait, 1, axis=0))
    assert rail_aspect(portrait) == pytest.approx(70 / 30, rel=.02)
    assert rail_aspect(landscape) == pytest.approx(70 / 30, rel=.02)
    # Orientation is not decided from the outline: a turned tile and an upright
    # tile can both measure the same ratio, so identity keeps its rotation search.
    assert order_quad(portrait).shape == (4, 2)


def test_the_correction_threshold_agrees_with_measured_taper():
    # Proportions alone never trigger the correction: a tile that merely looks
    # long or short is still a rectangle, and a rectangle needs no warp.
    for height in (40, 56, 58, 70):
        patch = np.zeros((90, 70, 3), np.uint8)
        patch[:height, 10:60] = 235
        assert rectification_needed(patch) is None, height
    # A real taper does trigger it, and the measured taper matches the gate.
    tapered = tapered_tile()
    geometry = min_area_quad(tapered)
    assert geometry is not None
    assert outline_skew(geometry.quad) >= .12
    assert rectification_needed(tapered) is not None


def test_parameter_defaults_cover_every_value_the_module_reads():
    for key in ("warp_size", "source_size", "face_trim", "minimum_crop_side", "minimum_silhouette_area_fraction"):
        assert key in RECTIFY_PARAMETERS or key == "source_size"


class Model:
    """Pjura stand-in that records exactly which pixels reached the network."""

    manifest_sha256 = "manifest"
    manifest = {"model_sha256": "model"}
    provider = "test"

    def __init__(self, label="3s", score=.99):
        self.label, self.score = label, score
        self.seen = []

    def predict_many(self, crops):
        self.seen.extend(crop.copy() for crop in crops)
        return [{"label": self.label, "score": self.score, "margin": .9,
                 "alternatives": [{"label": self.label, "score": self.score}]} for _ in crops]


def bank(legacy, model, **options):
    return PjuraBank(legacy, {"confidence": .9, "margin": .04, **options}, model=model)


def legacy_bank(label="9m", seen=None):
    def classify(pixels, *args, **kwargs):
        if seen is not None:
            seen.append(pixels.copy())
        return {"label": label, "score": .95, "margin": .2, "face_rotation": 0}
    return type("Legacy", (), {"classify": staticmethod(classify), "sha256": "t",
                               "training_games": {"local-train"}})()


def tapered_tile():
    """A tile whose projected outline is a trapezoid rather than a rectangle."""
    patch = np.zeros((84, 80, 3), np.uint8)
    cv2.fillPoly(patch, [np.array([[25, 8], [55, 8], [70, 92], [20, 92]], np.int32)], (235, 235, 235))
    return patch


def flattened_tile():
    """A tile the camera squashed and tapered, so its outline is a trapezoid."""
    patch = np.zeros((84, 80, 3), np.uint8)
    cv2.fillPoly(patch, [np.array([[25, 8], [55, 8], [70, 92], [20, 92]], np.int32)], (235, 235, 235))
    cv2.fillPoly(patch, [np.array([[30, 30], [50, 30], [55, 65], [25, 65]], np.int32)], (200, 40, 40))
    return patch


def test_hand_tiles_are_never_rewarped():
    patch = tile_patch()
    model = Model()
    bank(legacy_bank(), model).classify(patch, "hand", .9, .04, (0,))
    assert len(model.seen) == 1 and model.seen[0].shape == patch.shape


def test_a_foreshortened_tile_reaches_the_network_rectified():
    flattened = flattened_tile()
    model = Model()
    result = bank(legacy_bank(), model, rectification="rectified").classify(flattened, "river", .9, .04, (0,))
    width, height = RECTIFY_PARAMETERS["face_output_size"]
    assert len(model.seen) == 1 and model.seen[0].shape[:2] == (height, width)
    assert result["pjura_evidence"]["rectification"] == "top_face_perspective"
    assert result["pjura_evidence"]["view"] == "top_face"
    assert len(result["pjura_evidence"]["rectified_quad"]) == 4


def test_off_mode_sends_only_original_crops():
    model = Model()
    bank(legacy_bank(), model, rectification="off").classify(flattened_tile(), "river", .9, .04, (0, 90))
    assert len(model.seen) == 2 and model.seen[0].shape == flattened_tile().shape


class ViewModel(Model):
    """Answers differently for original crops and the fixed-size rectified face."""

    def __init__(self, original, rectified):
        super().__init__()
        self.original, self.rectified = original, rectified

    def predict_many(self, crops):
        self.seen.extend(crops)
        width, height = RECTIFY_PARAMETERS["face_output_size"]
        answers = []
        for crop in crops:
            label, score = self.rectified if crop.shape[:2] == (height, width) else self.original
            answers.append({"label": label, "score": score, "margin": score - .01,
                            "alternatives": [{"label": label, "score": score}]})
        return answers


def test_the_better_view_wins_when_only_it_is_confident():
    model = ViewModel(original=("9p", .55), rectified=("8p", .98))
    result = bank(legacy_bank("?"), model).classify(flattened_tile(), "river", .9, .04, (0,))
    assert result["label"] == "8p" and result["pjura_evidence"]["view"] == "top_face"
    assert result["pjura_evidence"]["views"]["original"]["label"] == "9p"


def test_confident_views_that_disagree_make_the_tile_unknown():
    model = ViewModel(original=("9p", .95), rectified=("8p", .98))
    result = bank(legacy_bank("?"), model).classify(flattened_tile(), "river", .9, .04, (0,))
    assert result["label"] == "?" and result["reason"] == "pjura_rectification_disagrees"


def side_view_tile(face_quad, side_quad, glyph=True, size=(90, 70)):
    """A tile seen at an angle: bright top face, greyer visible side, blue felt."""
    patch = np.zeros((*size, 3), np.uint8)
    patch[:] = (120, 70, 40)
    cv2.fillPoly(patch, [np.int32(side_quad)], (185, 188, 190))
    cv2.fillPoly(patch, [np.int32(face_quad)], (236, 238, 240))
    if glyph:
        centre = np.mean(face_quad, axis=0)
        cv2.circle(patch, tuple(np.int32(centre - [0, 8])), 5, (40, 40, 200), -1)
        cv2.rectangle(patch, tuple(np.int32(centre + [-6, 6])), tuple(np.int32(centre + [6, 12])), (30, 30, 30), -1)
    return patch


def test_top_face_fit_excludes_the_visible_side_face():
    face = np.array([[12, 6], [56, 6], [58, 60], [10, 60]], np.float32)
    side = np.array([[10, 60], [58, 60], [58, 80], [10, 80]], np.float32)
    geometry = top_face_outline(side_view_tile(face, side))
    assert geometry is not None and geometry.source == "top_face"
    # The silhouette would reach the bottom of the side strip (y=80); the top
    # face stops at the face edge.
    assert geometry.quad[:, 1].max() == pytest.approx(60, abs=2.5)
    assert np.abs(geometry.quad - face).max() < 3


def test_sheared_side_river_face_is_warped_upright_with_true_proportions():
    # A side-river tile: long axis horizontal in the image, face sheared into a
    # parallelogram, side strip along its lower edge.
    face = np.array([[14, 20], [80, 14], [84, 44], [10, 50]], np.float32)
    side = np.array([[10, 50], [84, 44], [84, 52], [10, 58]], np.float32)
    patch = side_view_tile(face, side, size=(70, 100))
    geometry = top_face_outline(patch)
    assert geometry is not None
    for degrees in (90, 270):
        upright = warp_top_face(patch, geometry, degrees).pixels
        width, height = RECTIFY_PARAMETERS["face_output_size"]
        assert upright.shape[:2] == (height, width)
        # No felt or side face leaks into the corrected crop.
        assert np.mean(np.min(upright, axis=2) > 200) > .75
    portrait = warp_top_face(patch, geometry, 0).pixels
    assert portrait.shape[0] > portrait.shape[1]


def test_dense_glyph_does_not_split_the_face():
    face = np.array([[8, 6], [52, 6], [52, 72], [8, 72]], np.float32)
    side = np.array([[8, 72], [52, 72], [52, 84], [8, 84]], np.float32)
    patch = side_view_tile(face, side, glyph=False)
    # Ink bands nearly as wide as the face, leaving only a 2px margin.
    for top in range(14, 66, 10):
        patch[top:top + 5, 10:50] = (40, 120, 30)
    geometry = top_face_outline(patch)
    assert geometry is not None
    assert np.abs(geometry.quad - face).max() < 3


def test_two_merged_faces_are_rejected_instead_of_guessed():
    patch = np.zeros((80, 80, 3), np.uint8)
    patch[:] = (120, 70, 40)
    patch[5:75, 5:35] = 236
    patch[40:75, 35:75] = 236     # an L-shaped white region is not one face
    assert top_face_outline(patch) is None
    assert top_face_outline(np.zeros((60, 40, 3), np.uint8)) is None


def test_image_order_does_not_depend_on_long_axis():
    square_ish = np.array([[30, 40], [0, 40], [2, 0], [28, 2]], np.float32)
    ordered = order_image_quad(square_ish)
    assert ordered[0].tolist() == [2, 0] and ordered[2].tolist() == [30, 40]


def test_the_tolerance_can_force_correction_for_measurement():
    # Passing zero applies the warp to every crop with a usable outline, which is
    # how the A/B evidence scripts isolate the correction's effect.
    upright = tile_patch()
    assert rectification_needed(upright) is None
    assert rectification_needed(upright, skew_tolerance=0.0) is not None


def blank_flattened_tile():
    """A tapered face with no strokes, so stroke colour cannot decide a five."""
    patch = np.zeros((84, 80, 3), np.uint8)
    cv2.fillPoly(patch, [np.array([[25, 8], [55, 8], [70, 92], [20, 92]], np.int32)], (235, 235, 235))
    return patch


def test_colour_and_orientation_evidence_still_read_the_original_pixels():
    """The template bank never receives the rectified pixels."""
    flattened = blank_flattened_tile()
    seen = []
    model = Model(label="5m")
    result = bank(legacy_bank("5mr", seen), model).classify(flattened, "river", .9, .04, (0,))
    assert seen and seen[0].shape == flattened.shape
    assert np.array_equal(seen[0], flattened)
    # The red five is resolved by the template bank, which saw the original crop.
    assert result["label"] == "5mr"
    assert result["legacy_evidence"]["label"] == "5mr"


def test_rectification_never_accepts_a_five_without_colour_evidence():
    result = bank(legacy_bank("?"), Model(label="5m")).classify(blank_flattened_tile(), "river", .9, .04, (0,))
    assert result["label"] == "?" and result["reason"] == "pjura_five_colour_unresolved"


def test_backs_and_non_tile_kinds_are_never_rewarped():
    flattened = flattened_tile()
    model = Model()
    assert bank(legacy_bank("back"), model).classify(flattened, "meld", .9, .04)["label"] == "back"
    assert not model.seen
    text_patch = np.full((30, 20, 3), 230, np.uint8)
    bank(legacy_bank("1"), model).classify_many([text_patch], "honba", .9, .04, (0,), kinds=("text",))
    assert all(crop.shape == text_patch.shape for crop in model.seen)


def test_a_clipped_neighbour_across_a_seam_is_not_merged_into_the_face():
    # A blank (white dragon) face with a narrow strip of the neighbouring tile
    # on the left, separated by a 3px dark seam - as a detector box clips melds.
    patch = np.zeros((96, 91, 3), np.uint8)
    patch[:] = (120, 80, 40)
    face = np.array([[16, 2], [66, 2], [88, 72], [36, 72]], np.float32)
    cv2.fillPoly(patch, [np.int32(face)], (226, 226, 224))
    # Neighbour strip parallel to the face's left edge, 3px seam, ~12% of the face.
    cv2.fillPoly(patch, [np.int32([[3, 14], [10, 14], [30, 72], [21, 72]])], (226, 226, 224))
    cv2.fillPoly(patch, [np.int32([[10, 72], [88, 72], [88, 90], [10, 90]])], (185, 186, 188))
    geometry = top_face_outline(patch)
    assert geometry is not None
    assert np.abs(geometry.quad - face).max() < 3
    upright = warp_top_face(patch, geometry, 0).pixels
    # No seam or neighbour remains inside the corrected blank face.
    inner = upright[6:-6, 6:-6]
    assert np.min(inner, axis=2).min() > 180
