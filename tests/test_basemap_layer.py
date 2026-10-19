"""Basemap place labels: zoom-dependent candidates, collision placement, country clicks and loading."""

from __future__ import annotations

import threading

import pytest

from terrascope.core.config import CFG
from terrascope.core.mapdata import MapDataError
from terrascope.core.view import View
from terrascope.layers.basemap import layer as basemap_layer
from terrascope.layers.basemap.layer import (
    CAPITAL_COLOR,
    COUNTRY_COLOR,
    LOADING_TEXT,
    UNAVAILABLE_TEXT,
    BaseMapLayer,
    _dimmed,
    _map_key_lines,
    _short_population,
    city_color,
    population_floor,
)

# The shipped default.yaml zoom table, pinned so these tests don't drift with config.
CITY_LEVELS = (
    (2.0, 5_000_000), (3.0, 2_000_000), (4.5, 1_000_000), (7.0, 500_000), (10.0, 200_000),
    (15.0, 100_000), (22.0, 50_000), (35.0, 15_000), (50.0, 0),
)
CITY_COLORS = ((1_000_000, "#cfd8e3"), (100_000, "#a9b4c0"), (0, "#8c98a4"))
PADDING = 2

# (lon, lat, short code, size, full name), as load_country_labels() returns.
SAMPLE_COUNTRY_LABELS = [
    (138.0, 36.0, "JPN", 5.0, "Japan"),
    (-98.0, 39.0, "USA", 9.0, "United States of America"),
    (2.5, 46.5, "FRA", 4.0, "France"),
]


@pytest.fixture(autouse=True)
def pinned_config(monkeypatch):
    monkeypatch.setattr(basemap_layer, "COUNTRY_MAX_ZOOM", 8.0)
    monkeypatch.setattr(basemap_layer, "CAPITAL_MIN_ZOOM", 1.0)
    monkeypatch.setattr(basemap_layer, "CITY_LEVELS", CITY_LEVELS)
    monkeypatch.setattr(basemap_layer, "CITY_COLORS", CITY_COLORS)
    monkeypatch.setattr(basemap_layer, "MAJOR_CITY_MIN_POPULATION", 2_000_000)
    monkeypatch.setattr(basemap_layer, "LABEL_PADDING", PADDING)
    monkeypatch.setattr(basemap_layer, "MAX_PLACED_LABELS", 250)
    monkeypatch.setattr(basemap_layer, "COUNTRY_CLICK_EXPANDS", True)


@pytest.fixture(autouse=True)
def no_stray_loader_threads():
    yield
    for thread in threading.enumerate():
        if thread.name == "terrascope-places":
            thread.join(timeout=5)


def ready_layer(countries=(), capitals=(), cities=()) -> BaseMapLayer:
    """A layer whose places arrived through the real pending-slot hand-off."""
    layer = BaseMapLayer()
    layer._pending = (list(countries), list(capitals), list(cities))
    layer.tick()
    assert layer._state == "ready"
    return layer


def placed(markers) -> dict[str, tuple[int, int]]:
    return {marker.label: (marker.x, marker.y) for marker in markers}


# A 100 x 30 view where cities of 2M+ are candidates (zoom 4) and countries show.
VIEW = View(0.0, 0.0, 4.0, 100, 30)


def at_cell(x: int, y: int, view: View = VIEW) -> tuple[float, float]:
    """(lon, lat) that lands in the middle of cell (x, y)."""
    return (
        view.origin_lon + (x + 0.5) * view.lon_span / view.width,
        view.origin_lat - (y + 0.5) * view.lat_span / view.height,
    )


def city(name: str, x: int, y: int, population: int = 10_000_000):
    lon, lat = at_cell(x, y)
    return lon, lat, name, population


def country(code: str, full: str, x: int, y: int):
    lon, lat = at_cell(x, y)
    return lon, lat, code, full


# ---- helpers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("zoom", "floor"),
    [
        (1.0, None),
        (1.99, None),
        (2.0, 5_000_000),
        (2.99, 5_000_000),
        (3.0, 2_000_000),
        (8.5, 500_000),
        (15.0, 100_000),
        (49.0, 15_000),
        (50.0, 0),
        (200.0, 0),
    ],
)
def test_population_floor_drops_as_you_zoom_in(zoom, floor):
    assert population_floor(zoom) == floor


@pytest.mark.parametrize(
    ("population", "color"),
    [
        (35_000_000, "#cfd8e3"),
        (1_000_000, "#cfd8e3"),
        (999_999, "#a9b4c0"),
        (100_000, "#a9b4c0"),
        (99_999, "#8c98a4"),
        (0, "#8c98a4"),
        (-1, "#8c98a4"),  # below every row: falls back to the smallest colour
    ],
)
def test_city_color_by_population(population, color):
    assert city_color(population) == color


@pytest.mark.parametrize(
    ("population", "text"),
    [(1_000_000, "1M"), (2_500_000, "2.5M"), (100_000, "100K"), (999_999, "999K"), (0, "0K")],
)
def test_short_population(population, text):
    assert _short_population(population) == text


def test_dimmed_scales_each_channel():
    assert _dimmed("#ffffff") == "#999999"
    assert _dimmed("#ff8000", 0.5) == "#804000"


def test_map_key_lists_borders_capital_and_city_size_bands():
    lines = _map_key_lines()
    assert [line.text for line in lines] == [
        "COUNTRY BORDER", "PROVINCE", "CAPITAL", "CITY 1M+", "CITY 100K+", "CITY <100K",
    ]
    assert lines[2].swatches == (CAPITAL_COLOR,)
    assert [line.swatches[0] for line in lines[3:]] == [color for _, color in CITY_COLORS]


def test_map_key_province_swatch_defaults_to_a_dimmed_border(monkeypatch):
    monkeypatch.setitem(CFG["map"], "province_color", None)
    lines = {line.text: line for line in _map_key_lines()}
    assert lines["PROVINCE"].swatches == (_dimmed(CFG["map"]["country_color"]),)


def test_map_key_single_city_band_is_just_city(monkeypatch):
    monkeypatch.setattr(basemap_layer, "CITY_COLORS", ((0, "#123456"),))
    assert [line.text for line in _map_key_lines()][-1] == "CITY"


def test_prepare_places_sorts_by_importance(sample_places):
    countries, capitals, cities = BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places)
    assert [code for *_, code, _full in countries] == ["USA", "JPN", "FRA"]  # biggest label first
    assert countries[0] == (-98.0, 39.0, "USA", "United States of America")
    assert [name for *_, name in capitals] == ["Tokyo", "Paris", "Seoul"]  # by population
    assert [name for _, _, name, _ in cities] == ["New York", "São Paulo", "Sydney", "York"]
    assert cities[-1] == (-1.08, 53.96, "York", 153_717)


# ---- which labels are candidates at which zoom ----------------------------------------------


@pytest.mark.parametrize(
    ("zoom", "detailed", "expected"),
    [
        (0.5, True, ["USA", "JPN", "FRA"]),
        (1.0, True, ["USA", "JPN", "FRA", "Tokyo", "Paris", "Seoul"]),
        (2.0, True, ["USA", "JPN", "FRA", "Tokyo", "Paris", "Seoul", "New York", "São Paulo"]),
        (3.0, True, ["USA", "JPN", "FRA", "Tokyo", "Paris", "Seoul", "New York", "São Paulo", "Sydney"]),
        (8.0, True, ["USA", "JPN", "FRA", "Tokyo", "Paris", "Seoul", "New York", "São Paulo", "Sydney"]),
        # past COUNTRY_MAX_ZOOM country names give way to cities
        (8.5, True, ["Tokyo", "Paris", "Seoul", "New York", "São Paulo", "Sydney"]),
        (15.0, True, ["Tokyo", "Paris", "Seoul", "New York", "São Paulo", "Sydney", "York"]),
        # outside the MAP tab only major cities (2M+) are labelled
        (15.0, False, ["Tokyo", "Paris", "Seoul", "New York", "São Paulo", "Sydney"]),
        (1.0, False, ["USA", "JPN", "FRA", "Tokyo", "Paris", "Seoul"]),
    ],
)
def test_candidates_by_zoom(sample_places, zoom, detailed, expected):
    layer = ready_layer(*BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places))
    layer.detailed = detailed
    assert [text for _, _, text, *_ in layer._candidates(zoom)] == expected


def test_candidates_without_places_or_countries(sample_places):
    layer = ready_layer(*BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places))
    layer.show_places = False
    assert [text for _, _, text, *_ in layer._candidates(15.0)] == []
    assert [text for _, _, text, *_ in layer._candidates(3.0)] == ["USA", "JPN", "FRA"]
    layer.show_places = True
    layer.show_countries = False
    assert [text for _, _, text, *_ in layer._candidates(3.0)][:3] == ["Tokyo", "Paris", "Seoul"]


def test_candidate_colours_and_click_targets(sample_places):
    layer = ready_layer(*BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places))
    candidates = {text: (color, centred, target) for _, _, text, color, centred, target in layer._candidates(3.0)}
    assert candidates["USA"] == (COUNTRY_COLOR, True, "United States of America")
    assert candidates["Tokyo"] == (CAPITAL_COLOR, False, None)
    assert candidates["Sydney"] == (city_color(4_630_000), False, None)


def test_expanded_countries_come_first_with_their_full_name(sample_places):
    layer = ready_layer(*BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places))
    layer._expanded.add("France")
    assert [text for _, _, text, *_ in layer._candidates(0.5)] == ["France", "USA", "JPN"]


# ---- placement ----------------------------------------------------------------------------------


def test_world_view_shows_countries_and_capitals_but_no_ordinary_cities(sample_places):
    layer = ready_layer(*BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places))
    view = View(20.0, 13.0, 1.0, 200, 50)
    result = layer.build(view)
    labels = placed(result.markers)
    assert {"USA", "JPN", "FRA"} <= labels.keys()
    assert "Seoul" in labels
    assert not {"New York", "São Paulo", "Sydney", "York"} & labels.keys()
    assert result.lines is None


def test_labels_are_placed_at_their_cell_and_coloured_by_kind():
    layer = ready_layer(
        countries=[country("ABC", "Abcland", 50, 5)],
        capitals=[(*at_cell(10, 15), "Capitol")],
        cities=[city("Metro", 70, 25, population=3_000_000)],
    )
    markers = {marker.label: marker for marker in layer.build(VIEW).markers}
    assert (markers["ABC"].x, markers["ABC"].y, markers["ABC"].color) == (49, 5, COUNTRY_COLOR)  # centred
    assert (markers["Capitol"].x, markers["Capitol"].y, markers["Capitol"].color) == (10, 15, CAPITAL_COLOR)
    assert (markers["Metro"].x, markers["Metro"].y, markers["Metro"].color) == (70, 25, city_color(3_000_000))


@pytest.mark.parametrize(
    ("second_x", "second_y", "fits"),
    [
        (10 + 5 + PADDING, 10, True),  # exactly LABEL_PADDING blank columns between them
        (10 + 5 + PADDING - 1, 10, False),  # one column too close
        (12, 10, False),  # overlapping
        (10 + 5 + PADDING - 1, 9, False),  # row above counts as a collision
        (10 + 5 + PADDING - 1, 11, False),  # row below too
        (10, 12, True),  # two rows apart is fine, even straight underneath
        (10, 8, True),
        (10 - 4 - PADDING, 10, True),  # to the left, exactly padded
        (10 - 4 - PADDING + 1, 10, False),
        (10 - 4 - PADDING + 1, 11, False),
    ],
)
def test_label_collision_rules(second_x, second_y, fits):
    layer = ready_layer(cities=[city("Alpha", 10, 10, 9_000_000), city("Beta", second_x, second_y, 8_000_000)])
    labels = placed(layer.build(VIEW).markers)
    assert labels["Alpha"] == (10, 10)  # the bigger city always wins
    assert ("Beta" in labels) is fits


def test_label_cut_off_at_the_right_edge_is_skipped():
    layer = ready_layer(cities=[city("Edge", 96, 3), city("Over", 97, 10)])
    labels = placed(layer.build(VIEW).markers)
    assert labels == {"Edge": (96, 3)}  # 96 + 4 == width: just fits


def test_place_outside_the_view_is_skipped():
    layer = ready_layer(cities=[(VIEW.origin_lon + VIEW.lon_span + 5, 0.0, "Faraway", 9_000_000), city("Here", 5, 5)])
    assert placed(layer.build(VIEW).markers) == {"Here": (5, 5)}


def test_country_code_is_centred_and_clamped_at_the_left_edge():
    layer = ready_layer(countries=[country("ABC", "Abcland", 10, 5), country("XYZ", "Xyzland", 0, 20)])
    assert placed(layer.build(VIEW).markers) == {"ABC": (9, 5), "XYZ": (0, 20)}


def test_country_code_near_the_right_edge_is_dropped_but_an_expanded_name_slides_in():
    layer = ready_layer(countries=[country("LRP", "Long Republic", 99, 5)])
    assert placed(layer.build(VIEW).markers) == {}
    layer._expanded.add("Long Republic")
    assert placed(layer.build(VIEW).markers) == {"Long Republic": (VIEW.width - len("Long Republic"), 5)}


def test_expanded_name_wider_than_the_view_is_dropped():
    narrow = View(0.0, 0.0, 4.0, 10, 30)
    lon, lat = narrow.origin_lon + narrow.lon_span / 2, 0.0
    layer = ready_layer(countries=[(lon, lat, "LRP", "Long Republic")])
    layer._expanded.add("Long Republic")
    assert layer.build(narrow).markers == []


def test_duplicate_name_is_only_labelled_once():
    layer = ready_layer(
        countries=[country("SGP", "Singapore", 20, 5)],
        capitals=[(*at_cell(70, 25), "Singapore")],  # far from the country label: no collision
    )
    layer._expanded.add("Singapore")
    markers = layer.build(VIEW).markers
    assert [marker.label for marker in markers] == ["Singapore"]
    assert markers[0].color == COUNTRY_COLOR


def test_labels_stop_at_the_safety_cap(monkeypatch):
    monkeypatch.setattr(basemap_layer, "MAX_PLACED_LABELS", 3)
    cities = [city(f"C{i}", 5 + 15 * i, 5 + 4 * i, 10_000_000 - i) for i in range(6)]
    layer = ready_layer(cities=cities)
    assert [marker.label for marker in layer.build(VIEW).markers] == ["C0", "C1", "C2"]


def test_render_remembers_the_markers():
    layer = ready_layer(cities=[city("Alpha", 10, 10)])
    result = layer.render(VIEW)
    assert layer.last_markers is result.markers
    assert not layer.dirty
    assert layer.hit_test_all(10, 10) == []  # labels are never selectable markers


# ---- clickable country labels -------------------------------------------------------------------


def test_clicking_a_country_code_shows_its_full_name_and_back():
    layer = ready_layer(countries=[country("FRA", "France", 50, 10)])
    assert placed(layer.build(VIEW).markers) == {"FRA": (49, 10)}
    layer.dirty = False
    assert layer.click_at(51, 10) is True  # last column of "FRA"
    assert layer.dirty
    assert placed(layer.build(VIEW).markers) == {"France": (47, 10)}
    assert layer.click_at(47, 10) is True
    assert placed(layer.build(VIEW).markers) == {"FRA": (49, 10)}


@pytest.mark.parametrize(("x", "y"), [(52, 10), (48, 10), (50, 11), (50, 9)])
def test_clicks_beside_a_country_label_are_ignored(x, y):
    layer = ready_layer(countries=[country("FRA", "France", 50, 10)])
    layer.build(VIEW)
    assert layer.click_at(x, y) is False
    assert layer._expanded == set()


def test_city_labels_are_not_clickable():
    layer = ready_layer(cities=[city("Alpha", 10, 10)])
    layer.build(VIEW)
    assert layer.click_at(11, 10) is False


def test_country_clicks_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(basemap_layer, "COUNTRY_CLICK_EXPANDS", False)
    layer = ready_layer(countries=[country("FRA", "France", 50, 10)])
    layer.build(VIEW)
    assert layer.click_at(50, 10) is False
    assert layer._expanded == set()


def test_an_expanded_country_wins_room_over_a_bigger_neighbour():
    big = country("BIG", "Bigland", 50, 10)
    small = country("SML", "Smallland", 54, 10)
    layer = ready_layer(countries=[big, small])
    assert placed(layer.build(VIEW).markers) == {"BIG": (49, 10)}  # small one collides and is dropped
    layer._expanded.add("Smallland")
    assert placed(layer.build(VIEW).markers) == {"Smallland": (50, 10)}


# ---- loading ------------------------------------------------------------------------------------------


def _join_loader():
    for thread in threading.enumerate():
        if thread.name == "terrascope-places":
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_first_tick_loads_places_in_the_background(monkeypatch, sample_places):
    monkeypatch.setattr(basemap_layer, "load_country_labels", lambda: SAMPLE_COUNTRY_LABELS)
    monkeypatch.setattr(basemap_layer, "load_places", lambda: sample_places)
    layer = BaseMapLayer()
    assert layer._state == "idle"
    layer.tick()
    assert layer._state == "loading"
    assert layer.loading_messages() == ["LOADING PLACE NAMES"]
    assert layer.legend_lines()[-1].text == LOADING_TEXT
    _join_loader()
    layer.dirty = False
    layer.tick()
    assert layer._state == "ready"
    assert layer.dirty
    assert layer.loading_messages() == []
    assert [code for _, _, code, _ in layer._countries] == ["USA", "JPN", "FRA"]
    assert [name for *_, name in layer._capitals] == ["Tokyo", "Paris", "Seoul"]
    layer.tick()  # nothing more to load
    assert layer._state == "ready"
    assert not any(thread.name == "terrascope-places" for thread in threading.enumerate())


@pytest.mark.parametrize("error", [MapDataError("bad file"), OSError("disk"), ValueError("parse")])
def test_failed_load_shows_unavailable_and_retries_later(monkeypatch, capsys, error):
    def broken():
        raise error

    monkeypatch.setattr(basemap_layer, "load_country_labels", broken)
    layer = BaseMapLayer()
    layer._load()  # what the loader thread runs
    assert layer._state == "failed"
    assert layer.dirty
    assert "place names:" in capsys.readouterr().err
    assert layer.legend_lines()[-1].text == UNAVAILABLE_TEXT
    assert layer._load_retry_at > 0

    layer._load_retry_at = float("inf")
    layer.tick()
    assert layer._state == "failed"  # waits before trying again

    monkeypatch.setattr(basemap_layer, "load_country_labels", lambda: SAMPLE_COUNTRY_LABELS)
    monkeypatch.setattr(basemap_layer, "load_places", lambda: [])
    layer._load_retry_at = 0.0
    layer.tick()
    assert layer._state == "loading"
    _join_loader()
    layer.tick()
    assert layer._state == "ready"


def test_load_hands_prepared_places_over_through_the_pending_slot(monkeypatch, sample_places):
    monkeypatch.setattr(basemap_layer, "load_country_labels", lambda: SAMPLE_COUNTRY_LABELS)
    monkeypatch.setattr(basemap_layer, "load_places", lambda: sample_places)
    layer = BaseMapLayer()
    layer._load()
    assert layer._pending == BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places)
    assert layer._cities == []  # nothing swapped in until the UI thread ticks


def test_detailed_places_replace_the_current_set_on_tick(sample_places):
    layer = ready_layer(cities=[city("Old", 10, 10)])
    detail = BaseMapLayer._prepare_places(SAMPLE_COUNTRY_LABELS, sample_places)
    layer.dirty = False
    layer.set_prepared_detail(detail)
    assert layer.dirty
    layer.tick()
    assert (layer._countries, layer._capitals, layer._cities) == detail
    assert layer._pending_detail is None
    assert layer._state == "ready"


# ---- legend ------------------------------------------------------------------------------------------------


def test_legend_offers_a_provinces_switch_once_they_are_ready():
    layer = ready_layer()
    layer.detailed = True
    layer.provinces_ready = False
    texts = [line.text for line in layer.legend_lines()]
    assert "PROVINCES" not in texts and "PROVINCE" not in texts
    assert "CAPITAL" in texts

    layer.provinces_ready = True
    layer.provinces_visible = True
    lines = layer.legend_lines()
    assert lines[0].text == "PROVINCES"
    assert lines[0].action == "provinces"
    assert lines[0].choices == ("ON", "OFF")
    assert lines[0].choice_index == 0
    assert "PROVINCE" in [line.text for line in lines]
    layer.provinces_visible = False
    assert layer.legend_lines()[0].choice_index == 1


def test_legend_is_empty_outside_the_map_tab_once_loaded():
    layer = ready_layer()
    layer.detailed = False
    layer.provinces_ready = True
    assert layer.legend_lines() == []
