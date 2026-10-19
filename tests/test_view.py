"""Map maths and the View: projection, wraparound, clamping, braille and coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from terrascope.core import view as view_module
from terrascope.core.view import (
    BLANK_BRAILLE,
    LAT_MAX,
    LAT_MID,
    LAT_MIN,
    LAT_RANGE,
    View,
    calculate_viewport,
    clamp_center,
    normalize_longitude,
    project_location,
    viewport_aspect,
    visible,
)


def cell_center(view: View, x: int, y: int) -> tuple[float, float]:
    """Lon/lat at the middle of terminal cell (x, y)."""
    lon = view.origin_lon + (x + 0.5) * view.lon_span / view.width
    lat = view.origin_lat - (y + 0.5) * view.lat_span / view.height
    return lon, lat


def ring(min_lon, max_lon, min_lat, max_lat):
    points = [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]
    return (points, min_lon, max_lon, min_lat, max_lat)


# ---- pure maths ---------------------------------------------------------------


def test_latitude_band_comes_from_config():
    assert (LAT_MIN, LAT_MAX) == (-58.0, 84.0)
    assert LAT_RANGE == 142.0
    assert LAT_MID == 13.0


@pytest.mark.parametrize(
    "zoom, aspect, expected",
    [
        (1, 2.0, (284.0, 142.0)),
        (2, 2.0, (142.0, 71.0)),
        (1, 1.0, (142.0, 142.0)),
        (4, 3.0, (106.5, 35.5)),
    ],
)
def test_calculate_viewport_keeps_degrees_square(zoom, aspect, expected):
    lon_span, lat_span = calculate_viewport(zoom, aspect)
    assert (lon_span, lat_span) == pytest.approx(expected)
    assert lon_span / lat_span == pytest.approx(aspect)


@pytest.mark.parametrize(
    "width, height, cell_aspect, expected",
    [
        (80, 20, 2.0, 2.0),
        (120, 20, 2.0, 3.0),
        (40, 40, 2.0, 0.5),
        (80, 20, 1.0, 4.0),
        (80, 0, 2.0, 2.0),
        (80, -5, 2.0, 2.0),
    ],
)
def test_viewport_aspect(width, height, cell_aspect, expected):
    assert viewport_aspect(width, height, cell_aspect) == pytest.approx(expected)


@pytest.mark.parametrize(
    "longitude, expected",
    [
        (0.0, 0.0),
        (179.0, 179.0),
        (180.0, -180.0),
        (-180.0, -180.0),
        (181.0, -179.0),
        (-181.0, 179.0),
        (540.0, -180.0),
        (-725.0, -5.0),
    ],
)
def test_normalize_longitude(longitude, expected):
    assert normalize_longitude(longitude) == pytest.approx(expected)


def test_clamp_center_fully_zoomed_out_pins_latitude_to_the_middle():
    lon_span, lat_span = calculate_viewport(1, 2.0)
    assert clamp_center(10.0, 70.0, lon_span, lat_span) == (10.0, LAT_MID)
    assert clamp_center(10.0, -70.0, lon_span, lat_span) == (10.0, LAT_MID)


@pytest.mark.parametrize(
    "center_lat, expected",
    [
        (13.0, 13.0),
        (80.0, LAT_MAX - 35.5),
        (-80.0, LAT_MIN + 35.5),
        (48.5, 48.5),
        (-22.5, -22.5),
    ],
)
def test_clamp_center_keeps_the_view_inside_the_latitude_band(center_lat, expected):
    lon_span, lat_span = calculate_viewport(2, 2.0)  # lat_span 71
    assert clamp_center(0.0, center_lat, lon_span, lat_span)[1] == pytest.approx(expected)


def test_clamp_center_wraps_longitude():
    lon_span, lat_span = calculate_viewport(2, 2.0)
    assert clamp_center(190.0, 0.0, lon_span, lat_span)[0] == pytest.approx(-170.0)
    assert clamp_center(-190.0, 0.0, lon_span, lat_span)[0] == pytest.approx(170.0)


# ---- visible(): which antimeridian copies of a ring are on screen ------------


def test_visible_ring_in_the_middle_needs_no_shift():
    assert visible(ring(-10, 10, 0, 10), 0.0, 13.0, 100.0, 50.0) == [0.0]


@pytest.mark.parametrize(
    "lat_bounds",
    [(50, 60), (-40, -30)],  # wholly above / below a view spanning -12..38
)
def test_visible_ring_outside_the_latitude_band_is_hidden(lat_bounds):
    assert visible(ring(-10, 10, *lat_bounds), 0.0, 13.0, 100.0, 50.0) == []


def test_visible_ring_across_the_antimeridian_uses_a_wrapped_copy():
    # View centred near the date line from the west side: -210.5 .. -139.5.
    shifts = visible(ring(170, 175, 0, 10), -175.0, 13.0, 71.0, 35.5)
    assert shifts == [-360.0]
    # And from the east: the same view seen from +185 is lon 149.5 .. 220.5.
    assert visible(ring(-175, -170, 0, 10), 185.0, 13.0, 71.0, 35.5) == [360.0]


def test_visible_ring_on_a_wider_than_world_view_is_drawn_more_than_once():
    # 426 degrees wide (-213 .. 213): a ring at 170..190 shows at both edges.
    assert visible(ring(170, 190, 0, 10), 0.0, 13.0, 426.0, 142.0) == [-360.0, 0.0]


def test_visible_ring_off_to_the_side_is_hidden():
    assert visible(ring(100, 110, 0, 10), 0.0, 13.0, 100.0, 50.0) == []


# ---- project_location ---------------------------------------------------------


def test_project_location_centre_maps_to_the_middle():
    x, y = project_location(0.0, 13.0, 0.0, 13.0, 284.0, 142.0, 80, 20)
    assert (x, y) == pytest.approx((40.0, 10.0))


def test_project_location_top_left_corner_is_origin():
    x, y = project_location(-142.0, 84.0, 0.0, 13.0, 284.0, 142.0, 80, 20)
    assert (x, y) == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize(
    "longitude, latitude",
    [
        (142.0, 13.0),  # exactly on the right edge (x == width)
        (160.0, 13.0),  # beyond the right edge
        (0.0, 85.0),  # above the top
        (0.0, -58.0),  # exactly on the bottom edge (y == height)
        (0.0, -70.0),  # below the bottom
    ],
)
def test_project_location_outside_the_view_is_none(longitude, latitude):
    assert project_location(longitude, latitude, 0.0, 13.0, 284.0, 142.0, 80, 20) is None


@pytest.mark.parametrize("longitude", [-170.0, 190.0, -530.0])
def test_project_location_wraps_across_the_antimeridian(longitude):
    # View centred on the date line, spanning 144.5 .. 215.5.
    x, _ = project_location(longitude, 13.0, 180.0, 13.0, 71.0, 35.5, 80, 20)
    assert x == pytest.approx((190.0 - 144.5) / 71.0 * 80)


# ---- View ---------------------------------------------------------------------


def test_view_derives_spans_scales_and_origin():
    view = View(10.0, 13.0, 1, 80, 20)
    assert view.aspect == pytest.approx(2.0)
    assert (view.lon_span, view.lat_span) == pytest.approx((284.0, 142.0))
    assert (view.pixel_width, view.pixel_height) == (160, 80)
    assert view.scale_x == pytest.approx(160 / 284)
    assert view.scale_y == pytest.approx(80 / 142)
    assert view.origin_lon == pytest.approx(10.0 - 142.0)
    assert view.origin_lat == pytest.approx(84.0)
    image = view.new_image()
    assert image.mode == "L" and image.size == (160, 80)
    assert image.getbbox() is None  # blank


@pytest.mark.parametrize(
    "view",
    [
        View(0.0, 13.0, 1, 80, 20),
        View(-120.0, 40.0, 4, 100, 30),
        View(180.0, 0.0, 8, 61, 17),
        View(-179.5, -20.0, 64, 120, 40),
    ],
    ids=["world", "zoomed", "date-line", "max-zoom-date-line"],
)
def test_cell_centres_round_trip_through_to_cell(view):
    for x in (0, 1, view.width // 2, view.width - 1):
        for y in (0, 1, view.height // 2, view.height - 1):
            lon, lat = cell_center(view, x, y)
            assert view.to_cell(lon, lat) == (x, y)
            # The same place given as a +-360 longitude lands in the same cell.
            assert view.to_cell(lon + 360.0, lat) == (x, y)
            assert view.to_cell(lon - 360.0, lat) == (x, y)


def test_to_cell_off_screen_is_none():
    view = View(0.0, 13.0, 4, 80, 20)  # 71 x 35.5 degrees
    assert view.to_cell(100.0, 13.0) is None
    assert view.to_cell(0.0, 60.0) is None


def test_to_pixel_matches_the_braille_grid_and_honours_shift():
    view = View(0.0, 13.0, 1, 80, 20)
    assert view.to_pixel(0.0, 13.0) == pytest.approx((80.0, 40.0))
    assert view.to_pixel(-142.0, 84.0) == pytest.approx((0.0, 0.0))
    shifted = view.to_pixel(-360.0, 13.0, shift=360.0)
    assert shifted == pytest.approx(view.to_pixel(0.0, 13.0))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: project_location returns a NaN position for non-finite lon/lat, so View.to_cell raises ValueError instead of returning None",
)
@pytest.mark.parametrize(
    "longitude, latitude",
    [(math.nan, 0.0), (math.inf, 0.0), (-math.inf, 0.0), (0.0, math.nan)],
)
def test_to_cell_non_finite_coordinates_are_not_on_screen(longitude, latitude):
    assert View(0.0, 13.0, 1, 80, 20).to_cell(longitude, latitude) is None


# ---- braille ------------------------------------------------------------------


@pytest.mark.parametrize(
    "pixel, codepoint",
    [
        ((0, 0), 0x2801),
        ((0, 1), 0x2802),
        ((0, 2), 0x2804),
        ((1, 0), 0x2808),
        ((1, 1), 0x2810),
        ((1, 2), 0x2820),
        ((0, 3), 0x2840),
        ((1, 3), 0x2880),
    ],
)
def test_to_lines_maps_each_subpixel_to_its_braille_dot(pixel, codepoint):
    view = View(0.0, 13.0, 1, 1, 1)
    image = view.new_image()
    image.putpixel(pixel, 255)
    assert view.to_lines(image) == [chr(codepoint)]


def test_to_lines_blank_and_full_cells():
    view = View(0.0, 13.0, 1, 3, 2)
    image = view.new_image()
    assert view.to_lines(image) == [BLANK_BRAILLE * 3] * 2
    image.paste(255, (2, 4, 4, 8))  # fill cell (1, 1)
    assert view.to_lines(image) == [BLANK_BRAILLE * 3, BLANK_BRAILLE + "⣿" + BLANK_BRAILLE]


def test_to_lines_resizes_images_of_the_wrong_size():
    view = View(0.0, 13.0, 1, 2, 1)
    lines = view.to_lines(Image.new("L", (40, 40), 255))
    assert lines == ["⣿⣿"]


# ---- land coverage -------------------------------------------------------------


def test_to_cell_coverage_counts_set_subpixels_per_cell():
    view = View(0.0, 13.0, 1, 2, 1)
    image = view.new_image()
    image.paste(255, (0, 0, 2, 2))  # top half of cell 0
    image.putpixel((3, 3), 255)  # one dot of cell 1
    coverage = view.to_cell_coverage(image, smooth=False)
    assert coverage.shape == (1, 2)
    assert coverage[0].tolist() == pytest.approx([0.5, 0.125])


def test_to_cell_coverage_smoothing_keeps_uniform_areas_uniform():
    view = View(0.0, 13.0, 1, 6, 4)
    full = Image.new("L", (12, 16), 255)
    assert np.allclose(view.to_cell_coverage(full), 1.0)
    assert np.allclose(view.to_cell_coverage(view.new_image()), 0.0)


def test_to_cell_coverage_smoothing_spreads_a_coastline():
    view = View(0.0, 13.0, 1, 9, 5)
    image = view.new_image()
    image.paste(255, (8, 8, 10, 12))  # exactly cell (4, 2)
    hard = view.to_cell_coverage(image, smooth=False)
    soft = view.to_cell_coverage(image)
    assert hard[2, 4] == 1.0 and hard.sum() == 1.0
    assert 0.0 < soft[2, 4] < 1.0
    # Wider horizontal kernel: the fade reaches two cells sideways, one up/down.
    assert soft[2, 2] > 0 and soft[2, 6] > 0 and soft[2, 1] == 0
    assert soft[1, 4] > 0 and soft[3, 4] > 0 and soft[0, 4] == 0
    # Away from the edges, blurring moves land around without creating any.
    assert soft.sum() == pytest.approx(1.0)
    assert soft[2, 3] == pytest.approx(soft[2, 5])


def test_blur_kernel_is_normalised():
    values = np.ones((3, 7), dtype=np.float32)
    result = view_module._blur(values, axis=1, kernel=(5.0, 10.0, 5.0))
    assert np.allclose(result, 1.0)
