"""WorldMap: selection, search / filter fan-out, pan and zoom, and frame building."""

from __future__ import annotations

import numpy as np
import pytest

from terrascope.core import world as world_module
from terrascope.core.layer import Layer, LayerRender, Marker, SliderSpec
from terrascope.core.view import BLANK_BRAILLE, LAT_MAX, LAT_MID, calculate_viewport
from terrascope.core.world import (
    PAN_STEP,
    START_LAT,
    START_LON,
    ZOOM_MAX,
    ZOOM_MIN,
    ZOOM_STEP,
    Frame,
    WorldMap,
    prepare_ring_set,
)


class StubLayer(Layer):
    def __init__(
        self,
        name: str = "stub",
        markers: list[Marker] | None = None,
        *,
        enabled: bool = True,
        claims_clicks: bool = False,
        options: tuple[str, ...] = (),
        slider: SliderSpec | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.enabled = enabled
        self.last_markers = list(markers or [])
        self.claims_clicks = claims_clicks
        self.options = list(options)
        self._slider = slider
        self.searches: list[str] = []
        self.filters: list[str | None] = []
        self.clicks: list[tuple[int, int]] = []
        self.render_args: list[tuple] = []

    def click_at(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        return self.claims_clicks

    def filter_options(self) -> list[str]:
        return self.options

    def slider(self) -> SliderSpec | None:
        return self._slider

    def apply_search(self, text: str) -> None:
        self.searches.append(text)

    def apply_filter(self, value: str | None) -> None:
        self.filters.append(value)

    def render_with_land(self, view, land_mask, *extra) -> LayerRender:
        self.render_args.append((view, land_mask, *extra))
        return self.render(view)

    def build(self, view) -> LayerRender:
        return LayerRender(None, "#ffffff", list(self.last_markers))


def square(min_lon, max_lon, min_lat, max_lat):
    points = [
        (min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat),
        (min_lon, max_lat), (min_lon, min_lat),
    ]
    return (points, min_lon, max_lon, min_lat, max_lat)


def point_under(world: WorldMap, col: int, row: int, cols: int, rows: int) -> tuple[float, float]:
    """Lon/lat at the middle of map cell (col, row), the way zoom_at sees it."""
    lon_span, lat_span = calculate_viewport(world.zoom, world.aspect)
    lon = world.center_lon - lon_span / 2 + (col + 0.5) / cols * lon_span
    lat = world.center_lat + lat_span / 2 - (row + 0.5) / rows * lat_span
    return lon, lat


@pytest.fixture
def world() -> WorldMap:
    return WorldMap([], [])


# ---- construction ----------------------------------------------------------------


def test_starts_fully_zoomed_out_at_the_middle_of_the_band(world):
    assert (ZOOM_MIN, ZOOM_MAX, ZOOM_STEP, PAN_STEP) == (1, 64, 1.5, 0.05)
    assert (world.zoom, world.center_lon, world.center_lat) == (ZOOM_MIN, START_LON, START_LAT)
    assert (START_LON, START_LAT) == (0.0, LAT_MID)
    assert world.dirty and world.selection_position == (0, 0) and world.selected_layer is None


def test_rings_too_short_to_draw_are_dropped():
    single = ([(1.0, 1.0)], 1.0, 1.0, 1.0, 1.0)
    world = WorldMap([square(0, 1, 0, 1), single], [])
    assert len(world.countries) == 2
    assert len(world._rings) == 1
    assert world._country_tier[1].shape == (1, 4)
    world.provinces = [single]
    assert world._province_rings == [] and world._province_tier[1].shape == (0, 4)


def test_prepared_ring_sets_are_installed_as_built():
    rings = [square(0, 10, 0, 10), square(20, 30, 0, 10)]
    prepared = prepare_ring_set(rings)
    assert prepared[0] is rings and prepared[2].tolist() == [[0, 10, 0, 10], [20, 30, 0, 10]]
    world = WorldMap([], [])
    world.set_prepared_countries(prepared)
    assert world.countries is rings and len(world._country_tier[0]) == 2
    world.set_prepared_provinces(prepared)
    assert world.provinces is rings and world.province_resolution == "10M"
    world.set_prepared_provinces_at_resolution(prepare_ring_set([]), "50M")
    assert world.provinces == [] and world.province_resolution == "50M"


# ---- selection ----------------------------------------------------------------------


@pytest.fixture
def crowded():
    """Two layers with markers around cell (10, 10), plus a disabled one on top."""
    first = StubLayer("first", [
        Marker(11, 10, "#fff", item="a-next"),
        Marker(10, 10, "#fff", item="a-on"),
        Marker(30, 30, "#fff", item="a-far"),
    ])
    second = StubLayer("second", [
        Marker(11, 11, "#fff", item="b-diagonal"),
        Marker(10, 11, "#fff", item="b-below"),
    ])
    hidden = StubLayer("hidden", [Marker(10, 10, "#fff", item="h-on")], enabled=False)
    return WorldMap([], [first, second, hidden]), first, second, hidden


def test_select_at_orders_candidates_by_distance_across_layers(crowded):
    world, first, second, hidden = crowded
    world.dirty = False
    world.select_at(10, 10)
    assert [item for _, item in world._selection_candidates] == [
        "a-on", "a-next", "b-below", "b-diagonal",
    ]
    assert world.selection_position == (1, 4)
    assert world.selected_layer is first
    assert (first.selected, second.selected, hidden.selected) == ("a-on", None, None)
    assert world.dirty


def test_cycle_selection_wraps_both_ways_and_moves_the_selection_between_layers(crowded):
    world, first, second, _ = crowded
    world.select_at(10, 10)
    world.cycle_selection(1)
    assert world.selection_position == (2, 4) and first.selected == "a-next"
    world.cycle_selection(1)
    assert world.selected_layer is second and second.selected == "b-below"
    assert first.selected is None
    world.cycle_selection(2)
    assert world.selection_position == (1, 4) and first.selected == "a-on"
    world.cycle_selection(-1)
    assert world.selection_position == (4, 4) and second.selected == "b-diagonal"


def test_cycle_selection_needs_two_candidates():
    only = StubLayer("only", [Marker(5, 5, "#fff", item="x")])
    world = WorldMap([], [only])
    world.cycle_selection(1)  # nothing selected: no-op
    world.select_at(5, 5)
    world.cycle_selection(1)
    assert world.selection_position == (1, 1) and only.selected == "x"


def test_select_at_empty_spot_clears_the_previous_selection(crowded):
    world, first, second, _ = crowded
    world.select_at(10, 10)
    world.select_at(60, 2)
    assert world.selection_position == (0, 0)
    assert first.selected is None and second.selected is None
    assert world.selected_layer is None
    # Both active layers were offered the click as a label action, the hidden one not.
    assert first.clicks == [(60, 2)] and second.clicks == [(60, 2)]


def test_label_action_handles_a_click_only_when_no_marker_is_hit():
    labels = StubLayer("labels", claims_clicks=True)
    markers = StubLayer("markers", [Marker(4, 4, "#fff", item="m")])
    world = WorldMap([], [labels, markers])
    world.select_at(4, 4)
    assert markers.selected == "m" and labels.clicks == []
    world.select_at(20, 20)
    assert labels.clicks == [(20, 20)]
    assert markers.selected is None and world.selection_position == (0, 0)


def test_selected_layer_falls_back_to_a_layer_that_selected_on_its_own():
    first, second = StubLayer("first"), StubLayer("second")
    world = WorldMap([], [first, second])
    second.selected = "picked by search"
    assert world.selected_layer is second
    second.enabled = False
    assert world.selected_layer is None


# ---- layers, search, filter ------------------------------------------------------------


def test_search_and_filter_reach_every_layer_and_clear_the_selection(crowded):
    world, first, second, hidden = crowded
    world.select_at(10, 10)
    world.set_search("tok")
    assert world.search_text == "tok"
    assert first.searches == second.searches == hidden.searches == ["tok"]
    assert world.selection_position == (0, 0) and first.selected is None

    world.select_at(10, 10)
    world.set_filter("Airbus")
    assert world.filter_value == "Airbus"
    assert first.filters == hidden.filters == ["Airbus"]
    assert world.selection_position == (0, 0)

    world.clear_filters()
    assert (world.search_text, world.filter_value) == ("", None)
    assert first.searches[-1] == "" and first.filters[-1] is None
    assert hidden.searches[-1] == "" and hidden.filters[-1] is None


def test_filter_options_are_the_sorted_union_of_active_layers():
    world = WorldMap([], [
        StubLayer("a", options=("Boeing", "Airbus")),
        StubLayer("b", options=("Airbus", "Cessna")),
        StubLayer("off", options=("Zeppelin",), enabled=False),
    ])
    assert world.filter_options() == ["Airbus", "Boeing", "Cessna"]


def test_slider_layer_is_the_first_active_layer_with_a_slider():
    spec = SliderSpec("TIME", 5, 0)
    off = StubLayer("off", slider=spec, enabled=False)
    plain = StubLayer("plain")
    timed = StubLayer("timed", slider=spec)
    assert WorldMap([], [off, plain, timed]).slider_layer() is timed
    assert WorldMap([], [off, plain]).slider_layer() is None


def test_needs_render_tracks_the_map_and_its_layers():
    layer = StubLayer()
    world = WorldMap([], [layer])
    world.dirty = False
    layer.dirty = False
    assert world.needs_render() is False
    layer.dirty = True
    assert world.needs_render() is True


# ---- zoom ---------------------------------------------------------------------------


def test_zoom_in_and_out_stop_at_the_limits(world):
    for _ in range(30):
        world.zoom_in()
    assert world.zoom == ZOOM_MAX
    for _ in range(30):
        world.zoom_out()
    assert world.zoom == ZOOM_MIN


def test_zooming_out_pulls_an_out_of_range_centre_back_into_the_band(world):
    world.zoom = 8
    world.center_lat = 75.0
    world.zoom_out()
    lat_span = calculate_viewport(world.zoom, world.aspect)[1]
    assert world.zoom == pytest.approx(8 / ZOOM_STEP)
    assert world.center_lat == pytest.approx(LAT_MAX - lat_span / 2)


def test_zoom_at_keeps_the_point_under_the_pointer_fixed(world):
    world.zoom = 2
    world.center_lon, world.center_lat = 20.0, 13.0
    before = point_under(world, 60, 5, 80, 20)
    world.dirty = False
    world.zoom_at(1, 60, 5, 80, 20)
    assert world.zoom == pytest.approx(3)
    assert point_under(world, 60, 5, 80, 20) == pytest.approx(before)
    assert world.dirty
    world.zoom_at(-1, 60, 5, 80, 20)
    assert world.zoom == pytest.approx(2)
    assert point_under(world, 60, 5, 80, 20) == pytest.approx(before)


def test_zoom_at_near_the_pole_is_clamped_rather_than_showing_beyond_the_band(world):
    world.zoom = ZOOM_STEP
    world.center_lat = 40.0
    world.zoom_at(-1, 40, 0, 80, 20)  # zooming out to the minimum
    assert (world.zoom, world.center_lat) == (ZOOM_MIN, LAT_MID)


@pytest.mark.parametrize("direction, zoom", [(1, ZOOM_MAX), (-1, ZOOM_MIN)])
def test_zoom_at_a_limit_changes_nothing(world, direction, zoom):
    world.zoom = zoom
    world.center_lon = 12.0
    world.dirty = False
    world.zoom_at(direction, 0, 0, 80, 20)
    assert (world.zoom, world.center_lon, world.dirty) == (zoom, 12.0, False)


@pytest.mark.parametrize("cols, rows", [(0, 20), (80, 0), (-1, -1)])
def test_zoom_at_and_pan_cells_ignore_an_empty_map_area(world, cols, rows):
    world.dirty = False
    world.zoom_at(1, 0, 0, cols, rows)
    world.pan_cells(5, 5, cols, rows)
    assert (world.zoom, world.center_lon, world.dirty) == (ZOOM_MIN, START_LON, False)


# ---- pan / focus / reset ----------------------------------------------------------------


def test_pan_moves_by_a_fraction_of_the_visible_span(world):
    world.zoom = 2
    world.dirty = False
    world.pan(1, 1)
    assert world.center_lon == pytest.approx(142.0 * PAN_STEP)
    assert world.center_lat == pytest.approx(13.0 + 71.0 * PAN_STEP)
    assert world.dirty


def test_pan_wraps_around_the_antimeridian(world):
    world.center_lon = 175.0
    world.pan(1, 0)
    assert world.center_lon == pytest.approx(175.0 + 284.0 * PAN_STEP - 360.0)


def test_pan_north_stops_at_the_top_of_the_band(world):
    world.zoom = 4
    for _ in range(100):
        world.pan(0, 1)
    lat_span = calculate_viewport(4, world.aspect)[1]
    assert world.center_lat == pytest.approx(LAT_MAX - lat_span / 2)


def test_pan_cannot_move_latitude_when_fully_zoomed_out(world):
    world.pan(0, -1)
    assert world.center_lat == LAT_MID


def test_pan_cells_drags_the_map_like_grabbing_it(world):
    world.zoom = 2
    world.pan_cells(10, -4, 80, 20)
    # Dragging right shows what lies west; dragging up shows what lies south.
    assert world.center_lon == pytest.approx(-10 * 142.0 / 80)
    assert world.center_lat == pytest.approx(13.0 - 4 * 71.0 / 20)


def test_focus_location_zooms_in_and_centres(world):
    world.focus_location(200.0, 30.0)
    assert world.zoom == 8
    assert world.center_lon == pytest.approx(-160.0)
    assert world.center_lat == pytest.approx(30.0)


def test_focus_location_never_zooms_out_and_clamps_to_the_band(world):
    world.zoom = 20
    world.focus_location(0.0, 89.0, zoom=4)
    lat_span = calculate_viewport(20, world.aspect)[1]
    assert world.zoom == 20
    assert world.center_lat == pytest.approx(LAT_MAX - lat_span / 2)


def test_reset_view_restores_the_start_and_clears_the_selection(crowded):
    world, first, _, _ = crowded
    world.zoom_in()
    world.pan(1, 1)
    world.select_at(10, 10)
    world.reset_view()
    assert (world.zoom, world.center_lon, world.center_lat) == (ZOOM_MIN, START_LON, START_LAT)
    assert world.selection_position == (0, 0) and first.selected is None


# ---- frame building ----------------------------------------------------------------------


def test_render_draws_outlines_and_land_cover():
    world = WorldMap([square(-30, 30, -10, 40)], [])
    frame = world.render(80, 20)
    assert isinstance(frame, Frame)
    assert len(frame.country_lines) == 20 and {len(line) for line in frame.country_lines} == {80}
    assert frame.lon_span == pytest.approx(284.0) and frame.render_width == 80
    assert frame.land_cover.shape == (20, 80)
    # (0, 13) is cell (40, 10): solid land. (-100, -40) is open sea.
    assert frame.land_cover[10, 40] == pytest.approx(1.0)
    assert frame.land_cover[17, 12] == 0.0
    # The outline is drawn, the interior is not.
    assert frame.country_lines[10][40] == BLANK_BRAILLE
    assert any(char != BLANK_BRAILLE for line in frame.country_lines for char in line)
    assert frame.province_lines is None
    assert world.dirty is False


def test_render_follows_the_window_shape(world):
    world.render(120, 20)
    assert world.aspect == pytest.approx(3.0)


def test_render_draws_land_on_both_sides_of_the_antimeridian():
    # A 426-degree-wide view (-213 .. 213) shows a date-line island twice.
    world = WorldMap([square(170, 190, 0, 26)], [])
    frame = world.render(120, 20)
    row = 10
    left = int((-180 + 213) / 426 * 120)
    right = int((180 + 213) / 426 * 120)
    assert frame.land_cover[row, left] > 0.5
    assert frame.land_cover[row, right] > 0.5
    assert frame.land_cover[row, 60] == 0.0


def test_render_skips_rings_outside_the_view():
    world = WorldMap([square(-30, 30, -10, 40)], [])
    world.zoom = 8
    world.center_lon = 120.0
    frame = world.render(80, 20)
    assert np.all(frame.land_cover == 0.0)
    assert set("".join(frame.country_lines)) == {BLANK_BRAILLE}


def test_province_borders_only_appear_when_enabled_and_zoomed_in():
    world = WorldMap([square(-30, 30, -10, 40)], [])
    world.provinces = [([(0.0, -10.0), (0.0, 40.0)], 0.0, 0.0, -10.0, 40.0)]
    assert world.render(80, 20).province_lines is None  # opt-in
    world.show_provinces = True
    assert world.render(80, 20).province_lines is None  # zoom 1 < province_min
    world.zoom = 2
    lines = world.render(80, 20).province_lines
    assert lines is not None and len(lines) == 20
    assert any(char != BLANK_BRAILLE for line in lines for char in line)


def test_province_pixels_under_a_country_border_are_cut_out():
    border = square(-30, 30, -10, 40)
    world = WorldMap([border], [])
    world.provinces = [border]
    world.show_provinces = True
    world.zoom = 2
    lines = world.render(80, 20).province_lines
    assert lines is not None
    assert set("".join(lines)) == {BLANK_BRAILLE}


def test_province_borders_off_screen_draw_nothing():
    world = WorldMap([], [])
    world.provinces = [([(150.0, 0.0), (160.0, 5.0)], 150.0, 160.0, 0.0, 5.0)]
    world.show_provinces = True
    world.zoom = 4
    assert world.render(80, 20).province_lines is None


class CountryAwareLayer(StubLayer):
    wants_country_image = True


def test_render_passes_tab_flags_and_renders_only_enabled_layers():
    shown = StubLayer("shown", [Marker(1, 1, "#fff", item="x")])
    hidden = StubLayer("hidden", enabled=False)
    borders = CountryAwareLayer("borders")
    world = WorldMap([square(-30, 30, -10, 40)], [shown, hidden, borders])
    world.detailed = False
    world.show_places = False
    world.show_countries = False
    frame = world.render(80, 20)

    assert [render.markers for render in frame.layers] == [shown.last_markers, []]
    assert hidden.render_args == [] and hidden.dirty is False
    assert len(shown.render_args[0]) == 2
    view, land_mask, country_image = borders.render_args[0]
    assert land_mask.size == country_image.size == (view.pixel_width, view.pixel_height)
    for layer in (shown, hidden, borders):
        assert (layer.detailed, layer.show_places, layer.show_countries) == (False, False, False)


def test_land_mask_given_to_layers_excludes_the_country_outline():
    layer = StubLayer()
    world = WorldMap([square(-30, 30, -10, 40)], [layer])
    world.render(80, 20)
    view, land_mask = layer.render_args[0]
    x, y = (int(round(value)) for value in view.to_pixel(0.0, 13.0))
    assert land_mask.getpixel((x, y)) == 255  # inside the land
    edge_x, edge_y = (int(round(value)) for value in view.to_pixel(-30.0, 13.0))
    assert land_mask.getpixel((edge_x, edge_y)) == 0  # outline cut out


def test_visible_rings_of_an_empty_tier_is_empty():
    world = WorldMap([], [])
    frame = world.render(10, 5)
    assert set("".join(frame.country_lines)) == {BLANK_BRAILLE}
    assert world_module._visible_rings(([], np.empty((0, 4))), None) == []
