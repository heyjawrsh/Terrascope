"""Day/night layer: terminator shading, tint vs fill looks, keys, slider and the live TIME panel."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone

import pytest
from PIL import Image

from terrascope.core.view import BLANK_BRAILLE, View
from terrascope.layers.daynight import layer as daynight_layer
from terrascope.layers.daynight.astronomy import solar_position
from terrascope.layers.daynight.layer import (
    LOOK_KEY,
    LOOKS,
    NIGHT_CONFIG,
    PHASE_THRESHOLDS,
    SEAM_GLYPH,
    UPDATE_SECONDS,
    DayNightLayer,
    _tint,
)

UTC = timezone.utc
EQUINOX_NOON = datetime(2024, 3, 20, 12, tzinfo=UTC)
SOLSTICE_NOON = datetime(2024, 6, 21, 12, tzinfo=UTC)


@pytest.fixture
def freeze(monkeypatch):
    """freeze(moment): the layer's clock reads `moment` until the test ends."""

    def _freeze(moment: datetime) -> None:
        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return moment if tz is None else moment.astimezone(tz)

        monkeypatch.setattr(daynight_layer, "datetime", FrozenDatetime)

    return _freeze


@pytest.fixture
def layer():
    return DayNightLayer()


def phase_colors() -> list[str]:
    return [NIGHT_CONFIG[key] for _thresh, key in PHASE_THRESHOLDS if NIGHT_CONFIG.get(key)]


def subsolar_point(moment: datetime) -> tuple[float, float]:
    declination, eq_time = solar_position(moment)
    minutes = moment.hour * 60 + moment.minute
    return (720 - minutes - eq_time) / 4, math.degrees(declination)


def world_view() -> View:
    """Spans lon -52..232 and lat -58..84: both the equinox-noon subsolar point
    (about 1.9°E, 0°) and its antipode are on screen."""
    return View(90.0, 13.0, 1.0, 120, 30)


def masks(view: View, land: bool) -> tuple[Image.Image, Image.Image]:
    land_mask = Image.new("L", (view.pixel_width, view.pixel_height), 255 if land else 0)
    return land_mask, view.new_image()


def by_cell(markers) -> dict[tuple[int, int], object]:
    return {(marker.x, marker.y): marker for marker in markers}


def noon_and_midnight_cells(view: View, moment: datetime):
    lon, lat = subsolar_point(moment)
    noon = view.to_cell(lon, lat)
    midnight = view.to_cell(lon + 180, -lat)
    assert noon is not None and midnight is not None
    return noon, midnight


# ---- construction, slider, ticking ---------------------------------------------


def test_starts_in_the_configured_look_with_no_time_offset(layer):
    configured = str(NIGHT_CONFIG.get("look", "fill")).lower()
    assert layer.look == (configured if configured in LOOKS else LOOKS[0])
    spec = layer.slider()
    assert (spec.label, spec.steps, spec.index, spec.caption, spec.marker, spec.wide) == ("TIME", 49, 24, "NOW", 24, True)


def test_unknown_configured_look_falls_back_to_the_first(monkeypatch):
    monkeypatch.setitem(NIGHT_CONFIG, "look", "sparkles")
    assert DayNightLayer().look == LOOKS[0]


@pytest.mark.parametrize(
    ("index", "offset", "caption"),
    [
        (24, 0, "NOW"),
        (30, 6, "+6h"),
        (0, -24, "-24h"),
        (48, 24, "+24h"),
        (-10, -24, "-24h"),  # clamped
        (999, 24, "+24h"),  # clamped
    ],
)
def test_slider_scrubs_up_to_a_day_either_way(layer, index, offset, caption):
    layer.dirty = False
    layer.set_slider(index)
    assert layer._time_offset_hours == offset
    assert layer.slider().index == offset + 24
    assert layer.slider().caption == caption
    assert layer.dirty


def test_slider_move_forces_a_recalculation_on_the_next_tick(layer):
    layer.enabled = True
    layer.set_slider(30)
    layer.dirty = False
    layer.tick()
    assert layer.dirty


def test_tick_marks_dirty_only_once_the_update_interval_has_passed(layer):
    layer.enabled = True
    layer.dirty = False
    layer._last_update = time.monotonic()
    layer.tick()
    assert not layer.dirty
    layer._last_update = time.monotonic() - UPDATE_SECONDS - 1
    layer.tick()
    assert layer.dirty


def test_tick_does_nothing_while_disabled(layer):
    layer.enabled = False
    layer.dirty = False
    layer._last_update = 0.0
    layer.tick()
    assert not layer.dirty


def test_display_time_applies_the_slider_offset(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.set_slider(24 - 5)
    assert layer._display_time() == datetime(2024, 3, 20, 7, tzinfo=UTC)


# ---- keys and legend --------------------------------------------------------------


@pytest.mark.parametrize("key", [ord(LOOK_KEY), ord(LOOK_KEY.upper())])
def test_look_key_switches_between_fill_and_tint(layer, key):
    first = layer.look
    layer.dirty = False
    assert layer.handle_key(key) is True
    assert layer.look != first and layer.look in LOOKS
    assert layer.dirty
    assert layer.handle_key(key) is True
    assert layer.look == first


def test_toggle_key_switches_the_layer_on_and_off(layer):
    before = layer.enabled
    assert layer.handle_key(ord(DayNightLayer.toggle_key)) is True
    assert layer.enabled is not before
    assert layer.handle_key(ord(DayNightLayer.toggle_key.upper())) is True
    assert layer.enabled is before


@pytest.mark.parametrize("key", [ord("z"), 410, -1])  # unrelated letter, KEY_RESIZE, no key
def test_unrelated_keys_are_not_consumed(layer, key):
    look, enabled = layer.look, layer.enabled
    assert layer.handle_key(key) is False
    assert (layer.look, layer.enabled) == (look, enabled)


def test_legend_click_on_the_look_switch_toggles_it(layer):
    first = layer.look
    layer.press_legend("something-else")
    assert layer.look == first
    layer.press_legend("look")
    assert layer.look != first


def test_legend_lists_each_coloured_phase_then_the_look_switch(layer):
    lines = layer.legend_lines()
    assert [line.text for line in lines[:-1]] == [
        "DAY", "SUNRISE / SUNSET", "CIVIL TWILIGHT", "NAUTICAL TWILIGHT", "ASTRO TWILIGHT",
    ]
    assert [line.swatches[0] for line in lines[:-1]] == phase_colors()
    switch = lines[-1]
    assert switch.action == "look"
    assert switch.choices == tuple(look.upper() for look in LOOKS)
    assert switch.choice_index == LOOKS.index(layer.look)
    layer.toggle_look()
    assert layer.legend_lines()[-1].choice_index == LOOKS.index(layer.look)


def test_phase_with_no_colour_is_left_out_of_the_legend(layer, monkeypatch):
    monkeypatch.setitem(NIGHT_CONFIG, "astro_twilight_color", None)
    texts = [line.text for line in layer.legend_lines()]
    assert "ASTRO TWILIGHT" not in texts
    assert "NAUTICAL TWILIGHT" in texts


def test_shading_is_never_clickable(layer, freeze):
    freeze(EQUINOX_NOON)
    view = world_view()
    layer.render_with_land(view, *masks(view, land=True))
    assert layer.last_markers
    marker = layer.last_markers[0]
    assert layer.hit_test_all(marker.x, marker.y) == []


# ---- live TIME panel rows ------------------------------------------------------------


def test_clock_rows_show_utc_date_time_and_day_of_year(layer, freeze):
    freeze(datetime(2024, 6, 21, 12, 0, 5, tzinfo=UTC))
    rows = dict(layer.live_rows("CLOCK"))
    assert rows["DATE"] == "2024-06-21"
    assert rows["UTC"] == "12:00:05"
    assert rows["DAY"] == "173 of year, week 25"  # 2024 is a leap year
    assert re.match(r"^\d\d:\d\d:05\b", rows["LOCAL"])  # the zone is whatever the machine uses


def test_clock_rows_follow_the_slider(layer, freeze):
    freeze(datetime(2024, 12, 31, 20, tzinfo=UTC))
    layer.set_slider(24 + 6)
    rows = dict(layer.live_rows("CLOCK"))
    assert rows["DATE"] == "2025-01-01"
    assert rows["UTC"] == "02:00:00"
    assert rows["DAY"] == "1 of year, week 1"


def test_sun_rows_for_london_at_the_june_solstice(layer, freeze):
    freeze(SOLSTICE_NOON)
    layer._remember_view(View(-0.1278, 51.5074, 4.0, 80, 24))
    rows = dict(layer.live_rows("SUN"))
    assert rows["CENTRE"] == "51.5°N 0.1°W"
    assert rows["PHASE"].startswith("Day (+6")  # about 62° up at noon
    assert rows["RISE"] in {"03:42 UTC", "03:43 UTC", "03:44 UTC"}
    assert rows["SET"] in {"20:20 UTC", "20:21 UTC", "20:22 UTC"}
    assert rows["LENGTH"] in {"16h37m", "16h38m", "16h39m"}
    assert rows["SOLAR"] in {"11:58 solar time", "11:59 solar time"}
    assert rows["SUBSOLAR"] in {"23.4°N 0.4°E", "23.4°N 0.5°E"}  # equation of time ≈ -1.9 min
    assert "TIMES" not in rows


def test_sun_rows_report_polar_day_instead_of_times(layer, freeze):
    freeze(SOLSTICE_NOON)
    layer._remember_view(View(25.0, 78.0, 4.0, 80, 24))  # Svalbard
    rows = dict(layer.live_rows("SUN"))
    assert rows["TIMES"] == "polar day"
    assert not {"RISE", "SET", "LENGTH"} & rows.keys()


def test_sun_rows_at_midnight_in_the_southern_winter_say_night(layer, freeze):
    freeze(datetime(2024, 6, 21, 12, tzinfo=UTC))
    layer._remember_view(View(-150.0, -40.0, 4.0, 80, 24))  # local midnight-ish, mid-Pacific
    assert dict(layer.live_rows("SUN"))["PHASE"].startswith("Night (-")


def test_sun_centre_longitude_is_normalised(layer, freeze):
    freeze(SOLSTICE_NOON)
    layer._remember_view(View(190.0, 0.0, 4.0, 80, 24))
    assert dict(layer.live_rows("SUN"))["CENTRE"] == "0.0°N 170.0°W"


def test_moon_rows_at_a_known_full_moon(layer, freeze):
    freeze(datetime(2024, 4, 23, 23, 49, tzinfo=UTC))
    rows = dict(layer.live_rows("MOON"))
    assert rows["PHASE"] == "Full Moon"
    assert rows["LIT"] == "100%"
    assert rows["AGE"].endswith(" days")
    assert float(rows["NEW IN"].split()[0]) == pytest.approx(14.5, abs=0.5)
    assert set(rows) == {"PHASE", "LIT", "AGE", "NEW IN", "FULL IN"}


@pytest.mark.parametrize("title", [None, "INFO", "clock"])
def test_unknown_panel_titles_have_no_rows(layer, title):
    assert layer.live_rows(title) == []


# ---- map shading ------------------------------------------------------------------------


def test_solar_elevation_grid_peaks_at_the_subsolar_point(layer, freeze):
    freeze(EQUINOX_NOON)
    view = world_view()
    elevation = layer._solar_elevation(view)
    assert elevation.shape == (view.pixel_height, view.pixel_width)
    assert elevation.max() == pytest.approx(90, abs=1.5)
    assert elevation.min() == pytest.approx(-90, abs=1.5)
    lon, lat = subsolar_point(EQUINOX_NOON)
    px, py = view.to_pixel(lon, lat)
    assert elevation[int(py), int(px)] > 88


def test_render_records_the_view_centre_and_markers(layer, freeze):
    freeze(EQUINOX_NOON)
    view = View(12.5, 41.9, 1.0, 120, 30)
    layer.dirty = True
    result = layer.render_with_land(view, *masks(view, land=True))
    assert not layer.dirty
    assert layer.last_markers is result.markers
    assert (layer._center_lon, layer._center_lat) == (12.5, 41.9)
    assert result.lines is None


def test_fill_look_noon_land_is_lit_and_the_antipode_is_dark(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "fill"
    view = world_view()
    result = layer.render_with_land(view, *masks(view, land=True))
    noon, midnight = noon_and_midnight_cells(view, EQUINOX_NOON)
    cells = by_cell(result.markers)
    assert cells[noon].color == NIGHT_CONFIG["day_color"]
    assert cells[noon].glyph == "\u28ff"  # every dot of lit land is stippled
    assert midnight not in cells  # night: nothing drawn
    assert result.bg_cells is None


def test_fill_look_does_not_stipple_open_water(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "fill"
    view = world_view()
    result = layer.render_with_land(view, *masks(view, land=False))
    noon, _ = noon_and_midnight_cells(view, EQUINOX_NOON)
    assert by_cell(result.markers)[noon].glyph == BLANK_BRAILLE


def test_tint_look_colours_lit_land_backgrounds_only(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "tint"
    view = world_view()
    result = layer.render_with_land(view, *masks(view, land=True))
    noon, midnight = noon_and_midnight_cells(view, EQUINOX_NOON)
    assert result.bg_cells[noon] == _tint(NIGHT_CONFIG["day_color"])
    assert midnight not in result.bg_cells
    assert noon not in by_cell(result.markers)  # interior cells get a background, no glyph
    assert all(marker.glyph == SEAM_GLYPH for marker in result.markers)


def test_tint_look_leaves_open_water_untinted(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "tint"
    view = world_view()
    result = layer.render_with_land(view, *masks(view, land=False))
    assert result.bg_cells == {}
    # The terminator over water is a smooth per-dot curve, not solid blocks.
    assert result.markers
    assert all(marker.glyph != BLANK_BRAILLE for marker in result.markers)
    assert {marker.color for marker in result.markers} <= set(phase_colors())
    assert any(marker.glyph != SEAM_GLYPH for marker in result.markers)


def test_look_key_switches_the_render_between_fill_and_tint(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "fill"
    view = world_view()
    land = masks(view, land=True)
    assert layer.render_with_land(view, *land).bg_cells is None
    layer.handle_key(ord(LOOK_KEY))
    assert layer.dirty
    assert layer.render_with_land(view, *land).bg_cells


def test_tint_seams_sit_on_the_lighter_side_of_every_phase_boundary(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "tint"
    view = world_view()
    result = layer.render_with_land(view, *masks(view, land=True))
    colors = phase_colors()
    depth_of_tint = {_tint(color): index for index, color in enumerate(colors)}
    night = len(colors)

    def depth(x, y):
        return depth_of_tint[result.bg_cells[(x, y)]] if (x, y) in result.bg_cells else night

    expected = set()
    for y in range(view.height):
        for x in range(view.width):
            neighbours = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            if any(
                0 <= nx < view.width and 0 <= ny < view.height and depth(nx, ny) > depth(x, y)
                for nx, ny in neighbours
            ):
                expected.add((x, y))
    seams = by_cell(result.markers)
    assert set(seams) == expected
    assert len(expected) > view.height  # the terminator crosses the whole map
    for (x, y), marker in seams.items():
        assert marker.color == colors[depth(x, y)]


def test_every_phase_band_appears_on_the_world_map(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "fill"
    view = world_view()
    result = layer.render_with_land(view, *masks(view, land=True))
    assert {marker.color for marker in result.markers} == set(phase_colors())


def test_zoomed_in_under_the_sun_everything_is_day_with_no_seams(layer, freeze):
    freeze(EQUINOX_NOON)
    lon, lat = subsolar_point(EQUINOX_NOON)
    view = View(lon, lat, 20.0, 40, 12)
    for look in LOOKS:
        layer.look = look
        result = layer.render_with_land(view, *masks(view, land=True))
        assert {marker.color for marker in result.markers} <= {NIGHT_CONFIG["day_color"]}
        if look == "fill":
            assert len(result.markers) == view.width * view.height
        else:
            assert result.markers == []
            assert set(result.bg_cells.values()) == {_tint(NIGHT_CONFIG["day_color"])}


def test_phase_with_no_colour_is_not_drawn(layer, freeze, monkeypatch):
    freeze(EQUINOX_NOON)
    astro = NIGHT_CONFIG["astro_twilight_color"]
    monkeypatch.setitem(NIGHT_CONFIG, "astro_twilight_color", None)
    view = world_view()
    for look in LOOKS:
        layer.look = look
        result = layer.render_with_land(view, *masks(view, land=True))
        assert astro not in {marker.color for marker in result.markers}
        assert _tint(astro) not in (result.bg_cells or {}).values()


def test_no_phase_colours_draws_nothing_over_water(layer):
    view = world_view()
    elevation = layer._solar_elevation(view)
    water = masks(view, land=False)[0]
    water_cells = ~(view.to_cell_coverage(water, smooth=False) > 0)
    assert DayNightLayer._water_seam_markers(view, elevation, [], water_cells) == []


def test_twelve_hours_later_noon_and_midnight_swap(layer, freeze):
    freeze(EQUINOX_NOON)
    layer.look = "tint"
    view = world_view()
    noon, midnight = noon_and_midnight_cells(view, EQUINOX_NOON)
    layer.set_slider(24 + 12)
    result = layer.render_with_land(view, *masks(view, land=True))
    assert noon not in result.bg_cells
    assert result.bg_cells[midnight] == _tint(NIGHT_CONFIG["day_color"])


# ---- tint colour ----------------------------------------------------------------------


def test_tint_blends_the_phase_colour_into_land(monkeypatch):
    land = daynight_layer._LAND_RGB
    monkeypatch.setattr(daynight_layer, "TINT_OPACITY", 0.0)
    assert _tint("#ffee88") == "#{:02x}{:02x}{:02x}".format(*land)
    monkeypatch.setattr(daynight_layer, "TINT_OPACITY", 1.0)
    assert _tint("#ffee88") == "#ffee88"
    monkeypatch.setattr(daynight_layer, "TINT_OPACITY", 0.5)
    assert _tint("#ffffff") == "#{:02x}{:02x}{:02x}".format(*(round((255 + c) / 2) for c in land))
