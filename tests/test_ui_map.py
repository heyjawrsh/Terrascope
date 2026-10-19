"""Map box drawing in core.ui.map: background fade, markers, legend, scale bar, slider, credits."""

from __future__ import annotations

import curses
import importlib
import itertools

import numpy as np
import pytest

from conftest import FakeWindow
from terrascope.core.layer import LayerRender, Marker, SidebarLine, SliderSpec
from terrascope.core.view import BLANK_BRAILLE
from terrascope.core.world import Frame

ui_map = importlib.import_module("terrascope.core.ui.map")
constants = importlib.import_module("terrascope.core.ui.constants")

PAD_X = constants.MAP_EDGE_PAD_X
PAD_Y = constants.MAP_EDGE_PAD_Y
EDGE_STEPS = constants.EDGE_STEPS

# A 30 x 100 map box at the window origin: the map area inside it starts at
# row 1, column 2 and is 28 rows x 96 columns.
BOX = (0, 0, 30, 100)
AREA_TOP, AREA_LEFT, AREA_HEIGHT, AREA_WIDTH = 1, 2, 28, 96
FOOTER_ROW = AREA_TOP + AREA_HEIGHT - 1 - PAD_Y  # screen row shared by slider / scale / OSM credit


def _frame(layers=(), country_lines=None, land_cover=None, lon_span=360.0, width=AREA_WIDTH):
    return Frame(
        country_lines=list(country_lines or []),
        layers=list(layers),
        lon_span=lon_span,
        render_width=width,
        land_cover=land_cover,
    )


def _draw(colors, frame=None, window=None, **kwargs):
    window = window or FakeWindow(32, 110)
    hits = ui_map.draw_map_box(window, *BOX, frame or _frame(), colors, 1.0, 0.0, **kwargs)
    return window, hits


def _cell(window, row, col):
    """(char, attrs) at a map-area cell."""
    return window.cells[AREA_TOP + row][AREA_LEFT + col], window.attrs[AREA_TOP + row][AREA_LEFT + col]


# ---- colour parsing / coastline fade --------------------------------------------------


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        (None, (0, 0, 0, 0.0)),
        (False, (0, 0, 0, 0.0)),
        ("", (0, 0, 0, 0.0)),
        ("#fff", (255, 255, 255, 1.0)),
        ("#f008", (255, 0, 0, 0x88 / 255)),
        ("#102030", (16, 32, 48, 1.0)),
        ("#10203000", (16, 32, 48, 0.0)),
        (" 10203080 ", (16, 32, 48, 0x80 / 255)),
    ],
)
def test_parse_rgba(color, expected):
    assert ui_map.parse_rgba(color) == pytest.approx(expected)


@pytest.mark.parametrize("color", ["#12", "#12345", "#1234567"])
def test_parse_rgba_rejects_malformed_colours(color):
    with pytest.raises(ValueError):
        ui_map.parse_rgba(color)


def test_edge_palette_fades_from_water_to_land():
    palette = ui_map._edge_palette("#000000", "#ff0000")
    assert len(palette) == EDGE_STEPS + 1
    assert palette[0] == "#000000" and palette[-1] == "#ff0000"
    reds = [int(color[1:3], 16) for color in palette]
    assert reds == sorted(reds)


def test_edge_palette_keeps_fully_transparent_steps_transparent():
    palette = ui_map._edge_palette(None, "#00ff00")
    assert palette[0] is None
    assert palette[-1] == "#00ff00"
    # Premultiplied: the transparent side never tints the fade towards black.
    assert all(color.endswith("00") for color in palette[1:])


def test_edge_cells_are_a_colour_fade_when_both_sides_are_opaque():
    cells, shade_color = ui_map._edge_cells("#000000", "#ffffff")
    assert shade_color is None
    assert all(glyph is None for _, glyph in cells)
    assert [color for color, _ in cells] == ui_map._edge_palette("#000000", "#ffffff")


def test_edge_cells_fade_with_shade_glyphs_over_a_transparent_side():
    cells, shade_color = ui_map._edge_cells(None, "#336699")
    assert shade_color == "#336699"
    assert cells[0] == (None, None)  # open water: nothing drawn
    assert cells[-1] == ("#336699", None)  # full land: solid colour
    glyphs = [glyph for _, glyph in cells if glyph]
    assert glyphs and set(glyphs) <= set(constants.SHADE_GLYPHS)
    order = [constants.SHADE_GLYPHS.index(glyph) for glyph in glyphs]
    assert order == sorted(order)  # light -> dark towards the land


def test_background_grid_maps_land_cover_to_the_fade_and_is_cached():
    cover = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.04]])
    frame = _frame(land_cover=cover)
    grid = ui_map.background_grid(frame, "#000000", "#ff0000")
    palette = ui_map._edge_palette("#000000", "#ff0000")
    assert grid == [
        [palette[0], palette[EDGE_STEPS // 2], palette[-1]],
        [palette[-1], palette[0], palette[0]],
    ]
    assert ui_map.background_grid(frame, "#000000", "#ff0000") is grid
    assert ui_map.edge_shades(frame, "#000000", "#ff0000") is None
    # Other colours: recomputed, not served from the stale cache.
    assert ui_map.background_grid(frame, "#000000", "#00ff00")[0][2] == "#00ff00"


def test_background_grid_without_land_cover_is_empty():
    frame = _frame()
    assert ui_map.background_grid(frame, "#000000", "#ff0000") == []
    assert ui_map.edge_shades(_frame(), None, "#ff0000") == ([], "#ff0000")


def test_segments_split_rows_into_runs_of_one_background():
    row = ["a", "a", None, None, "b", "a"]
    assert list(ui_map._segments(row, 0, 6)) == [(0, 2, "a"), (2, 4, None), (4, 5, "b"), (5, 6, "a")]
    assert list(ui_map._segments(row, 1, 3)) == [(1, 2, "a"), (2, 3, None)]
    assert list(ui_map._segments(row, 3, 3)) == []


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (("✈", "#ff9f43", "Aircraft"), ("✈", "#ff9f43", "Aircraft")),
        (("#ff9f43", "Aircraft", "✈"), ("✈", "#ff9f43", "Aircraft")),
        (("Aircraft", "✈", "#ff9f43"), ("✈", "#ff9f43", "Aircraft")),
        (("●", "Quake"), ("●", None, "Quake")),
        (("#123456", "Label"), ("", "#123456", "Label")),
        ((), ("", None, "")),
    ],
)
def test_parse_legend_item_by_content(item, expected):
    assert ui_map._parse_legend_item(item) == expected


# ---- scale bar ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 1.0), (-5, 1.0), (1.0, 1), (1.4, 1), (1.5, 2), (3.4, 2), (3.5, 5), (7.4, 5), (7.5, 10), (0.03, 0.02), (1234, 1000)],
)
def test_nice_round_km(value, expected):
    assert ui_map._nice_round_km(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "text"),
    [(5000, "5,000 km"), (1000, "1,000 km"), (500, "500 km"), (1, "1 km"), (0.5, "500 m"), (0.02, "20 m")],
)
def test_format_km(value, text):
    assert ui_map._format_km(value) == text


@pytest.mark.parametrize(
    ("lon_span", "width", "max_columns"),
    [(0, 100, 50), (-1, 100, 50), (10, 0, 50), (10, 100, constants.SCALE_BAR_MIN_COLUMNS - 1)],
)
def test_compute_scale_bar_returns_none_without_a_usable_view(lon_span, width, max_columns):
    assert ui_map.compute_scale_bar(lon_span, width, 0.0, max_columns) is None


def _label_km(text: str) -> float:
    value, unit = text.rsplit(" ", 2)[-2:]
    number = float(value.replace(",", ""))
    return number / 1000 if unit == "m" else number


def _assert_bar_matches_label(lon_span, width, lat, max_columns):
    columns, text = ui_map.compute_scale_bar(lon_span, width, lat, max_columns)
    bar = text.split(" ", 1)[0]
    assert len(bar) == columns
    assert bar[0] == bar[-1] == constants._SCALE_SYMBOLS["end"]
    assert columns <= max_columns
    km_per_column = lon_span / width * constants.KM_PER_DEGREE_LAT * max(
        np.cos(np.radians(max(-85.0, min(85.0, lat)))), 0.05
    )
    # The labelled distance is what the bar really spans, give or take the
    # half column lost to rounding the bar to whole columns.
    assert abs(columns * km_per_column - _label_km(text)) <= km_per_column / 2 + 1e-9


@pytest.mark.parametrize("lon_span", [360.0, 97.3, 12.0, 1.7, 0.21, 0.013])
@pytest.mark.parametrize("lat", [0.0, 45.0, -60.0, 89.9])
@pytest.mark.parametrize("max_columns", [19, 40, 92])
def test_scale_bar_label_matches_the_bar_length(lon_span, lat, max_columns):
    _assert_bar_matches_label(lon_span, 96, lat, max_columns)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: compute_scale_bar clamps the bar to max_columns after picking the "
    "round distance, so on narrow maps the label overstates the bar (here 10 km on 8 km)",
)
def test_scale_bar_label_matches_a_clamped_bar_on_a_narrow_map():
    # 0.8 km per column, 10 columns available: a 10 km bar needs 12.5 columns.
    lon_span = 0.8 * 100 / constants.KM_PER_DEGREE_LAT
    _assert_bar_matches_label(lon_span, 100, 0.0, 10)


def test_draw_map_box_puts_the_scale_bar_right_aligned_on_the_footer_row(colors):
    window, _ = _draw(colors, _frame(lon_span=40.0))
    _, text = ui_map.compute_scale_bar(40.0, AREA_WIDTH, 0.0, AREA_WIDTH - PAD_X - 2)
    end = AREA_LEFT + AREA_WIDTH - PAD_X
    assert window.lines()[FOOTER_ROW][end - len(text) : end] == text


# ---- slider -------------------------------------------------------------------------


def test_fixed_slider_track_has_the_configured_length():
    spec = SliderSpec("SPEED", 5, 2)
    first, length = ui_map.slider_track(spec, AREA_WIDTH, _frame(), 0.0)
    assert length == constants.SLIDER_TRACK_COLUMNS
    scale_slot = constants.SCALE_BAR_TARGET_COLUMNS * 2 + 4
    right = AREA_WIDTH - PAD_X - scale_slot - 2
    assert first + length + 1 + constants.SLIDER_CAPTION_WIDTH == right


@pytest.mark.parametrize("steps", [7, 8, 40, 400])
def test_wide_slider_track_uses_one_column_per_step_when_it_can(steps):
    spec = SliderSpec("RADAR", steps, 0, wide=True)
    first, length = ui_map.slider_track(spec, 200, _frame(), 0.0)
    assert length <= steps
    if steps % 2:
        assert length % 2 == 1  # the middle step stays directly selectable
    if steps <= 40:
        assert length == steps
    assert first - len(spec.label) - 1 >= PAD_X


@pytest.mark.parametrize("width", [0, 20, 40, 50])
def test_slider_track_is_none_without_room(width):
    assert ui_map.slider_track(SliderSpec("SPEED", 5, 2), width, _frame(), 0.0) is None


def _slider_row(window):
    return window.lines()[FOOTER_ROW], window.attrs[FOOTER_ROW]


def test_draw_slider_places_label_knob_and_caption(colors):
    spec = SliderSpec("SPEED", 5, 2, caption="x60")
    window, _ = _draw(colors, slider=spec)
    first, length = ui_map.slider_track(spec, AREA_WIDTH, _frame(), 0.0)
    line, attrs = _slider_row(window)
    track = line[AREA_LEFT + first : AREA_LEFT + first + length]
    knob = round(2 * (length - 1) / 4)
    assert track[knob] == constants.SLIDER_KNOB
    assert set(track[:knob]) == {constants.SLIDER_FILLED}
    assert set(track[knob + 1 :]) == {constants.SLIDER_EMPTY}
    assert line[AREA_LEFT + first - len("SPEED") - 1 :].startswith("SPEED ")
    assert line[AREA_LEFT + first + length + 1 :].startswith("x60")
    assert attrs[AREA_LEFT + first + length - 1] & curses.A_DIM


def test_disabled_slider_is_dimmed_and_empty(colors):
    spec = SliderSpec("RADAR", 5, 2, caption="--", disabled=True)
    window, _ = _draw(colors, slider=spec)
    first, length = ui_map.slider_track(spec, AREA_WIDTH, _frame(), 0.0)
    line, attrs = _slider_row(window)
    assert set(line[AREA_LEFT + first : AREA_LEFT + first + length]) == {constants.SLIDER_EMPTY}
    label_x = AREA_LEFT + first - len("RADAR") - 1
    for x in (label_x, AREA_LEFT + first, AREA_LEFT + first + length + 1):
        assert attrs[x] & curses.A_DIM


def test_slider_marks_the_special_step_and_loaded_steps(colors):
    spec = SliderSpec("TIME", 14, 0, marker=13, loaded_steps=(5,))
    window, _ = _draw(colors, slider=spec)
    first, length = ui_map.slider_track(spec, AREA_WIDTH, _frame(), 0.0)
    assert length == 14  # one column per step
    line, attrs = _slider_row(window)
    track = line[AREA_LEFT + first : AREA_LEFT + first + length]
    assert track[0] == constants.SLIDER_KNOB
    assert track[13] == "|" and attrs[AREA_LEFT + first + 13] & curses.A_BOLD
    assert track[5] == constants.SLIDER_FILLED and not attrs[AREA_LEFT + first + 5] & curses.A_DIM
    assert track[4] == constants.SLIDER_EMPTY


def test_slider_status_sits_left_of_the_label_and_is_truncated(colors):
    spec = SliderSpec("SPEED", 5, 0, status="loading radar " * 10)
    window, _ = _draw(colors, slider=spec)
    first, _ = ui_map.slider_track(spec, AREA_WIDTH, _frame(), 0.0)
    line, attrs = _slider_row(window)
    label_x = AREA_LEFT + first - len("SPEED") - 1
    status = line[AREA_LEFT + PAD_X : label_x - 1]
    assert status.startswith("loading radar")
    assert len(status) == label_x - 1 - (AREA_LEFT + PAD_X)
    assert attrs[AREA_LEFT + PAD_X] & curses.A_DIM
    assert line[label_x:].startswith("SPEED")


# ---- border buttons --------------------------------------------------------------------


@pytest.mark.parametrize("visible", [True, False])
def test_panel_button_sits_on_the_bottom_border(visible):
    row, x, text = ui_map.panel_button_slot((2, 3, 20, 80), visible)
    assert (row, x) == (21, 5)
    assert text == f" {constants.PANEL_BUTTON_HIDE if visible else constants.PANEL_BUTTON_SHOW} "


@pytest.mark.parametrize(
    ("rect", "enabled"),
    [((2, 3, 20, 80), False), ((2, 3, 1, 80), True), ((2, 3, 20, len(constants.PANEL_BUTTON_SHOW) + 6), True)],
)
def test_panel_button_is_hidden_when_disabled_or_without_room(rect, enabled):
    assert ui_map.panel_button_slot(rect, False, enabled) is None


@pytest.mark.parametrize("width", [30, 40, 50, 60, 80, 120])
def test_border_buttons_follow_the_panel_button_and_stay_on_the_border(width):
    rect = (0, 0, 10, width)
    labels = ["CLOUDS", "STORMS", "QUAKES"]
    panel = ui_map.panel_button_slot(rect, True)
    slots = ui_map.border_button_slots(rect, True, labels)
    texts = [text for _, _, text in slots]
    assert texts == [f" {label} " for label in labels[: len(slots)]]
    spans = ([(panel[1], panel[1] + len(panel[2]))] if panel else []) + [(x, x + len(text)) for _, x, text in slots]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start > end  # at least one border cell between buttons
    for row, x, text in slots:
        assert row == 9
        assert x + len(text) + 2 <= width


def test_border_buttons_start_at_the_corner_without_a_panel_button():
    slots = ui_map.border_button_slots((0, 0, 10, 80), True, ["A"], panel_button=False)
    assert slots == [(9, 2, " A ")]
    assert ui_map.border_button_slots((0, 0, 1, 80), True, ["A"]) == []


def test_draw_map_box_draws_title_panel_button_and_layer_buttons(colors):
    window, _ = _draw(colors, panel_visible=False, buttons=["CLOUDS"])
    lines = window.lines()
    assert f" {constants.MAP_BOX_TITLE} " in lines[0]
    bottom = lines[BOX[2] - 1]
    button = f" {constants.PANEL_BUTTON_SHOW} "
    assert bottom[2:].startswith(button)
    assert " CLOUDS " in bottom
    assert window.attrs[BOX[2] - 1][2] & curses.A_BOLD
    assert bottom[0] == "└" and bottom[99] == "┘"


# ---- background / layer symbols ----------------------------------------------------


def test_country_outlines_and_layer_braille_are_drawn_in_the_map_area(colors):
    braille = "⣿⣀"
    country = [BLANK_BRAILLE * 3 + "⠁" + BLANK_BRAILLE * 2]
    layer = LayerRender([BLANK_BRAILLE * 10 + braille], "#ffcc00")
    window, _ = _draw(colors, _frame([layer], country_lines=country))
    assert _cell(window, 0, 3)[0] == "⠁"
    assert window.lines()[AREA_TOP][AREA_LEFT + 10 : AREA_LEFT + 12] == braille
    assert window.lines()[AREA_TOP][AREA_LEFT + 12] == " "  # blank braille is skipped, not painted


def test_layer_background_overrides_do_not_touch_the_cached_grid(colors):
    cover = np.zeros((AREA_HEIGHT, AREA_WIDTH))
    frame = _frame([LayerRender(None, "#fff", bg_cells={(1, 0): "#ff0000", (500, 500): "#00ff00"})], land_cover=cover)
    _draw(colors, frame)
    cached = frame.background_cache[1]
    assert cached[0][1] != "#ff0000"
    window = FakeWindow(32, 110)
    area = ui_map._MapArea(window, *BOX, frame, colors)
    assert area.bg_at(0, 1) == "#ff0000"


def test_coastline_shade_is_drawn_when_water_is_transparent(colors, monkeypatch):
    monkeypatch.setitem(ui_map.MAP_CONFIG, "water_color", None)
    monkeypatch.setitem(ui_map.MAP_CONFIG, "land_color", "#336699")
    cover = np.zeros((AREA_HEIGHT, AREA_WIDTH))
    cover[0, 4] = 0.5
    cover[0, 5] = 0.5
    # A "blank" night-fill marker on a fade cell keeps the shade glyph.
    markers = [Marker(5, 0, "#000000", glyph=BLANK_BRAILLE)]
    frame = _frame([LayerRender(None, "#fff", markers=markers)], land_cover=cover)
    window, _ = _draw(colors, frame)
    assert _cell(window, 0, 4)[0] in constants.SHADE_GLYPHS
    assert _cell(window, 0, 5)[0] == _cell(window, 0, 4)[0]
    assert _cell(window, 0, 3)[0] == " "


# ---- markers ------------------------------------------------------------------------


def _markers_frame(*markers, lines=None):
    return _frame([LayerRender(lines, "#ffffff", markers=list(markers))])


def test_marker_glyphs_are_bold_and_selected_markers_reversed(colors):
    frame = _markers_frame(
        Marker(10, 5, "#ff0000", glyph="✈"),
        Marker(12, 5, "#ff0000", glyph="✈", selected=True),
        Marker(14, 5, "#ff0000", glyph="●", dim=True),
        Marker(16, 5, "#ff0000", glyph="⣿"),
    )
    window, _ = _draw(colors, frame)
    char, attrs = _cell(window, 5, 10)
    assert char == "✈" and attrs & curses.A_BOLD and not attrs & curses.A_REVERSE
    char, attrs = _cell(window, 5, 12)
    assert char == "✈" and attrs & curses.A_REVERSE and attrs & curses.A_BOLD
    char, attrs = _cell(window, 5, 14)
    assert char == "●" and attrs & curses.A_DIM and not attrs & curses.A_BOLD
    char, attrs = _cell(window, 5, 16)
    assert char == "⣿" and not attrs & curses.A_BOLD  # braille keeps the outline weight


def test_selected_label_is_reversed_on_every_letter(colors):
    window, _ = _draw(colors, _markers_frame(Marker(20, 3, "#ffffff", label="Paris", selected=True)))
    assert window.lines()[AREA_TOP + 3][AREA_LEFT + 20 : AREA_LEFT + 25] == "Paris"
    for offset in range(5):
        assert _cell(window, 3, 20 + offset)[1] & curses.A_REVERSE


def test_markers_outside_the_map_area_are_skipped(colors):
    frame = _markers_frame(
        Marker(-1, 0, "#fff", glyph="X"),
        Marker(AREA_WIDTH, 0, "#fff", glyph="X"),
        Marker(0, -1, "#fff", glyph="X"),
        Marker(0, AREA_HEIGHT, "#fff", glyph="X"),
        Marker(-3, 2, "#fff", label="Nowhere"),
    )
    window, _ = _draw(colors, frame)
    assert "X" not in window.text()
    assert "Nowhere" not in window.text() and "where" not in window.text()


def test_labels_are_clipped_at_the_right_edge_of_the_map(colors):
    x = AREA_WIDTH - 4
    window, _ = _draw(colors, _markers_frame(Marker(x, 2, "#fff", label="Vladivostok")))
    row = window.lines()[AREA_TOP + 2]
    assert row[AREA_LEFT + x : AREA_LEFT + AREA_WIDTH] == "Vlad"
    assert row[AREA_LEFT + AREA_WIDTH :] .rstrip() == " │"  # padding then the untouched border


def test_labels_are_drawn_after_symbols(colors):
    # The label comes first in the layer, the plane after it: the label still wins.
    frame = _markers_frame(Marker(30, 4, "#fff", label="Tokyo"), Marker(31, 4, "#f00", glyph="✈"))
    window, _ = _draw(colors, frame)
    assert window.lines()[AREA_TOP + 4][AREA_LEFT + 30 : AREA_LEFT + 35] == "Tokyo"


def test_marker_without_glyph_uses_the_layer_braille_at_its_cell(colors):
    lines = [BLANK_BRAILLE * 8 + "⣤" + BLANK_BRAILLE]
    frame = _markers_frame(
        Marker(8, 0, "#00ff00", selected=True),
        Marker(9, 0, "#00ff00", selected=True),  # blank there: nothing to highlight
        Marker(3, 7, "#00ff00", selected=True),  # past the layer lines
        lines=lines,
    )
    window, _ = _draw(colors, frame)
    char, attrs = _cell(window, 0, 8)
    assert char == "⣤" and attrs & curses.A_REVERSE
    assert not _cell(window, 0, 9)[1] & curses.A_REVERSE
    assert not _cell(window, 7, 3)[1] & curses.A_REVERSE


# ---- attribution / overlay text -------------------------------------------------------


def test_attribution_stacks_layer_credits_above_the_osm_credit(colors):
    window, _ = _draw(colors, attribution="Flights: OpenSky Network · NOAA · © OpenStreetMap")
    lines = window.lines()
    x = AREA_LEFT + PAD_X
    assert lines[FOOTER_ROW][x:].startswith(constants.OSM_CREDIT)
    assert lines[FOOTER_ROW - 1][x:].startswith("NOAA")
    assert lines[FOOTER_ROW - 2][x:].startswith("OpenSky Network")
    assert "Flights" not in window.text()
    assert window.text().count("OpenStreetMap") == 1


def test_attribution_is_clipped_to_the_map_width(colors):
    window = FakeWindow(10, 30)
    frame = _frame(width=26)
    ui_map.draw_map_box(window, 0, 0, 10, 30, frame, colors, 1.0, 0.0, attribution="A" * 80)
    footer = 1 + 8 - 1 - PAD_Y
    room = 26 - 2 * PAD_X
    assert window.lines()[footer - 1][2 + PAD_X : 2 + PAD_X + room + 1] == "A" * room + " "


def test_overlay_text_is_drawn_only_inside_the_map_area(colors):
    window = FakeWindow(32, 110)
    frame = _frame()
    ui_map.draw_map_overlay_text(window, BOX, frame, colors, AREA_TOP + 2, AREA_WIDTH - 1, "LOADING", curses.A_BOLD)
    row = window.lines()[AREA_TOP + 2]
    assert row[AREA_WIDTH - 1 : AREA_LEFT + AREA_WIDTH] == "LOA"
    assert window.attrs[AREA_TOP + 2][AREA_WIDTH - 1] & curses.A_BOLD
    ui_map.draw_map_overlay_text(window, BOX, frame, colors, 0, 10, "BORDER")
    ui_map.draw_map_overlay_text(window, BOX, frame, colors, AREA_TOP + AREA_HEIGHT, 10, "BELOW")
    assert "BORDER" not in window.text() and "BELOW" not in window.text()


# ---- legend ----------------------------------------------------------------------------


def test_legend_is_right_aligned_above_the_footer_and_reports_clickable_lines(colors):
    legend = [
        SidebarLine("TEMPERATURE", bold=True),
        SidebarLine("cold", swatches=("#0000ff",)),
        SidebarLine("MODE", choices=("TINT", "FILL"), choice_index=1, action="mode"),
        SidebarLine(legend_items=(("✈", "#ff9f43", "Aircraft"),)),
    ]
    window, hits = _draw(colors, legend=legend)
    lines = window.lines()
    right = AREA_LEFT + AREA_WIDTH - PAD_X
    last_row = AREA_TOP + AREA_HEIGHT - 2 - PAD_Y
    assert lines[last_row][:right].endswith("✈ Aircraft")
    assert lines[last_row - 1][:right].endswith("MODE[ TINT  FILL ]")
    assert lines[last_row - 2][:right].endswith("cold ■")
    assert lines[last_row - 3][:right].endswith("TEMPERATURE")
    assert window.attrs[last_row - 3][right - 1] & curses.A_BOLD
    lit = lines[last_row - 1].index(" FILL ")
    assert window.attrs[last_row - 1][lit] & curses.A_REVERSE
    assert hits == [(last_row - 1, right - len("MODE[ TINT  FILL ]"), right, "mode")]


def test_legend_skips_lines_wider_than_the_map_and_empty_lines(colors):
    legend = [SidebarLine("x" * 200), SidebarLine(""), SidebarLine("ok")]
    window, hits = _draw(colors, legend=legend)
    assert "x" * 20 not in window.text()
    last_row = AREA_TOP + AREA_HEIGHT - 2 - PAD_Y
    assert window.lines()[last_row][: AREA_LEFT + AREA_WIDTH - PAD_X].endswith("ok")
    assert hits == []


def test_legend_needs_a_minimum_map_height(colors):
    window = FakeWindow(10, 60)
    hits = ui_map.draw_map_box(
        window, 0, 0, 4 + PAD_Y + 2, 60, _frame(width=56), colors, 1.0, 0.0,
        legend=[SidebarLine("HIDDEN", action="x")],
    )
    assert hits == [] and "HIDDEN" not in window.text()


def test_legend_keeps_only_the_lines_that_fit_above_the_footer(colors):
    window = FakeWindow(12, 60)
    height = 10  # map area of 8 rows
    legend = [SidebarLine(f"L{i}") for i in range(20)]
    ui_map.draw_map_box(window, 0, 0, height, 60, _frame(width=56), colors, 1.0, 0.0, legend=legend)
    shown = [i for i in range(20) if f"L{i}" in window.text()]
    assert shown == list(range(8 - 2 - PAD_Y))


# ---- layout sanity across sizes -----------------------------------------------------


@pytest.mark.parametrize(("height", "width"), list(itertools.product([3, 6, 12, 20], [8, 24, 50, 140])))
def test_draw_map_box_never_writes_outside_its_box(colors, height, width):
    window = FakeWindow(height + 4, width + 4)
    frame = _frame(
        [LayerRender(None, "#fff", markers=[Marker(width - 6, 0, "#fff", label="Somewhere far")])],
        lon_span=50.0,
        width=max(1, width - 4),
    )
    ui_map.draw_map_box(
        window, 1, 1, height, width, frame, colors, 1.0, 10.0,
        legend=[SidebarLine("LEGEND")], slider=SliderSpec("T", 30, 3, wide=True, caption="now"),
        attribution="Source", buttons=["ONE", "TWO"],
    )
    for y, line in enumerate(window.lines()):
        for x, char in enumerate(line):
            if not (1 <= y < 1 + height and 1 <= x < 1 + width):
                assert char == " ", (y, x, char)
