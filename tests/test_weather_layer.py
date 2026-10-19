"""WeatherLayer: keys, city placement and selection, info rows, radar slider/markers, quake filters and legend."""

from __future__ import annotations

import functools
import math
import types

import numpy as np
import pytest

from terrascope.core.layer import Marker, PanelBox
from terrascope.core.mapdata import MapDataError
from terrascope.core.view import View
from terrascope.layers.weather import city_weather, earthquakes, radar
from terrascope.layers.weather import layer as layer_mod
from terrascope.layers.weather.config import CITY_CONFIG, MARKER_PREFIX, WEATHER_SCHEMA, WEATHER_ICONS
from terrascope.layers.weather.forecast import FORECAST_ROWS, VERTICAL_DAY_SEPARATOR_ROWS, city_key

NOW = 1_760_000_000.0


class RecordingThread:
    started: list = []

    def __init__(self, target, args=(), name=None, daemon=None, **kwargs):
        self.target, self.args, self.name = target, args, name

    def start(self):
        RecordingThread.started.append(self)


@pytest.fixture
def threads(monkeypatch):
    RecordingThread.started = []
    namespace = types.SimpleNamespace(Thread=RecordingThread)
    monkeypatch.setattr(layer_mod, "threading", namespace)
    return RecordingThread.started


@pytest.fixture
def frozen_time(monkeypatch):
    clock = types.SimpleNamespace(now=NOW)
    monkeypatch.setattr(layer_mod.time, "time", lambda: clock.now)
    return clock


@pytest.fixture
def layer(tmp_path, monkeypatch, threads):
    """A WeatherLayer whose caches live in tmp_path and whose threads never start."""
    monkeypatch.setattr(city_weather, "WEATHER_CACHE_FILE", tmp_path / "weather.json")
    monkeypatch.setattr(city_weather, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(radar, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        layer_mod, "EarthquakeCache",
        functools.partial(earthquakes.EarthquakeCache, tmp_path / "quakes.json"),
    )
    instance = layer_mod.WeatherLayer()
    instance.enabled = True
    return instance


def _city(name="Tokyo", lon=139.75, lat=35.69, capital=True, population=35_676_000, country="Japan") -> dict:
    return {
        "key": city_key(name, lat, lon), "name": name, "latitude": lat, "longitude": lon,
        "capital": capital, "country": country, "population": population,
    }


def _quake(id="q1", magnitude=5.0, place="10 km E of Somewhere", lon=0.0, lat=0.0, time=None, **extra) -> dict:
    return {
        "id": id, "magnitude": magnitude, "place": place, "longitude": lon, "latitude": lat,
        "depth_km": 10.0, "time": time, "tsunami": False, "alert": None, "status": "reviewed", **extra,
    }


def _set_places(layer, places):
    layer._capitals = [place for place in places if place[3]]
    layer._cities = [place for place in places if not place[3]]


def _day(date="2026-10-19") -> dict:
    return {"date": date, "code": 1, "tmin": 3.0, "tmax": 9.0, "hours": [[0, 3.0, 1, 0, 5.0], [1, 9.0, 1, 0, 5.0]]}


def _reading(**fields) -> dict:
    reading = {
        "temperature": 18.4, "windspeed": 12.2, "winddirection": 225.0, "weathercode": 2,
        "observed_at": "2026-10-19T12:00", "forecast": [_day(), _day("2026-10-20")],
        "schema": WEATHER_SCHEMA, "fetched_at": NOW, "error": "",
    }
    reading.update(fields)
    return reading


def _frame(stamp, nowcast=False):
    return {"time": stamp, "path": f"/{stamp}", "nowcast": nowcast}


def _install_radar(layer, stamps, loaded=None, nowcast=()):
    frames = [_frame(stamp, stamp in nowcast) for stamp in stamps]
    layer.radar._frames = frames
    layer.radar._rasters = {
        stamp: np.zeros((8, 8), dtype=np.uint8) for stamp in (stamps if loaded is None else loaded)
    }
    return frames


def _text(pieces) -> str:
    return "".join(text for text, _c, _s in pieces)


# ---- keys and titles -------------------------------------------------------------


def test_construction_reads_nothing_from_the_network(layer, fake_http, threads):
    assert fake_http.calls == [] and threads == []
    assert layer.show_cities is bool(CITY_CONFIG["enabled"])
    assert layer.quakes == [] and layer.selected is None


@pytest.mark.parametrize("key", [ord("c"), ord("C")])
def test_city_toggle_key(layer, key):
    before = layer.show_cities
    layer.dirty = False
    assert layer.handle_key(key) is True
    assert layer.show_cities is (not before) and layer.dirty is True
    layer.handle_key(key)
    assert layer.show_cities is before


@pytest.mark.parametrize("key", [ord("e"), ord("E")])
def test_quake_toggle_key(layer, key):
    before = layer.show_quakes
    assert layer.handle_key(key) is True
    assert layer.show_quakes is (not before)


@pytest.mark.parametrize("key", [ord("x"), ord(" "), -1, 256, 10_000])
def test_other_keys_are_not_consumed(layer, key):
    layer.dirty = False
    assert layer.handle_key(key) is False
    assert layer.dirty is False


def test_info_title_follows_selection_kind(layer):
    assert layer.info_title == "CITY INFO"
    layer.selected = _quake()
    assert layer.info_title == "EARTHQUAKE INFO"
    layer.selected = _city()
    assert layer.info_title == "CITY INFO"


def test_panel_layout_sizes_info_box_to_its_rows(layer):
    assert layer.panel_layout_for(20, 40) == (PanelBox("info", 3 / 20), PanelBox("rich", 1 - 3 / 20, "FORECAST"))
    layer.selected = _quake()
    info, rich = layer.panel_layout_for(20, 60)
    assert math.isclose(info.ratio, (8 + 3) / 20) and math.isclose(info.ratio + rich.ratio, 1.0)
    tiny_info, _ = layer.panel_layout_for(5, 60)  # never taller than the space allows
    assert math.isclose(tiny_info.ratio, 3 / 5)
    _, zero = layer.panel_layout_for(0, 1)
    assert zero.title == "FORECAST"


# ---- forecast box --------------------------------------------------------------------


def test_rich_title_names_the_selected_city_only(layer):
    assert layer.rich_title("FORECAST") == "FORECAST"
    assert layer.rich_title(None) == ""
    layer.selected = _city()
    assert layer.rich_title("FORECAST") == "FORECAST · Tokyo"
    layer.selected = _quake()
    assert layer.rich_title("FORECAST") == "FORECAST"


def test_rich_lines_placeholder_without_a_city(layer):
    assert _text(layer.rich_lines("F", 80, 10)[0]) == "CLICK A CITY NAME"
    vertical = layer.rich_lines("F", 24, 4, vertical=True)
    assert len(vertical) == 4
    layer.selected = _quake()
    assert _text(layer.rich_lines("F", 80, 10)[0]) == "CLICK A CITY NAME"


def test_rich_lines_for_a_city(layer):
    city = _city()
    layer.selected = city
    assert layer.rich_lines("F", 80, 10) == [[("FETCHING...", None, "dim")]]
    layer.weather._entries[city["key"]] = _reading(schema=WEATHER_SCHEMA - 1)
    assert layer.rich_lines("F", 80, 10) == [[("FORECAST UNAVAILABLE", None, "dim")]]
    layer.weather._entries[city["key"]] = _reading(forecast=["junk", None])
    assert layer.rich_lines("F", 80, 10) == [[("FORECAST UNAVAILABLE", None, "dim")]]
    layer.weather._entries[city["key"]] = _reading()
    assert len(layer.rich_lines("F", 80, 10)) == FORECAST_ROWS
    assert len(layer.rich_lines("F", 30, 5, vertical=True, scroll=2)) == 5


def test_forecast_line_count(layer):
    assert layer.forecast_line_count() == 0
    city = _city()
    layer.selected = city
    assert layer.forecast_line_count() == 0
    layer.weather._entries[city["key"]] = _reading(forecast=[_day(), "junk", _day(), _day()])
    assert layer.forecast_line_count() == 3 * FORECAST_ROWS + 2 * VERTICAL_DAY_SEPARATOR_ROWS
    layer.weather._entries[city["key"]] = _reading(forecast=None)
    assert layer.forecast_line_count() == 0


# ---- quake data lifecycle -------------------------------------------------------------


def test_reload_reselects_the_same_quake_by_id(layer):
    layer.quake_cache.quakes = [_quake("a", 4.0), _quake("b", 6.0)]
    layer.selected = _quake("b", 5.5)
    layer.reload()
    assert layer.selected == _quake("b", 6.0) and len(layer.quakes) == 2


@pytest.mark.parametrize("selected", [_quake("gone"), _quake(id="")])
def test_reload_drops_a_quake_that_disappeared(layer, selected):
    layer.quake_cache.quakes = [_quake("a")]
    layer.selected = selected
    layer.reload()
    assert layer.selected is None


def test_reload_keeps_a_selected_city_and_disabled_layer_has_no_quakes(layer):
    layer.quake_cache.quakes = [_quake("a")]
    layer.selected = _city()
    layer.enabled = False
    layer.reload()
    assert layer.quakes == [] and layer.selected == _city()


def test_clear(layer):
    layer.quakes, layer.selected = [_quake()], _quake()
    layer.clear()
    assert layer.quakes == [] and layer.selected is None


def test_offline_refresh_is_never_due(layer, fake_http):
    assert layer.is_due() is False
    assert layer.refresh_if_due(force=True) is False
    layer.show_quakes = False
    assert layer.is_due() is False and layer.refresh_if_due(force=True) is False
    assert fake_http.calls == []


def test_background_weather_and_radar_updates_mark_the_layer_dirty(layer):
    layer.dirty = False
    layer.weather._on_update()
    assert layer.dirty is True
    layer.dirty = False
    layer.radar._on_update()
    assert layer.dirty is True


def test_prefetch_offline_with_ordered_startup_starts_nothing(layer, threads):
    layer.prefetch()
    assert threads == []


def test_prefetch_online_starts_places_radar_and_quakes(layer, threads, monkeypatch):
    monkeypatch.setattr(layer_mod, "FETCH_NEW_DATA_FROM_API", True)
    radar_calls = []
    monkeypatch.setattr(layer.radar, "ensure_fresh", lambda: radar_calls.append(1))
    layer._ordered_startup = False
    layer.prefetch()
    assert [thread.name for thread in threads] == ["terrascope-weather-cities", "terrascope-quakes"]
    assert layer._places_state == "loading" and layer.refresh_in_progress is True
    assert radar_calls == [1]
    threads[1].target()  # run the quake prefetch inline (the cache itself is offline)
    assert layer.refresh_in_progress is False and layer.refresh_completed is True


# ---- loading places ------------------------------------------------------------------


def test_load_places_sorts_and_warms_the_biggest_cities(layer, sample_places, monkeypatch):
    shuffled = list(reversed(sample_places))
    monkeypatch.setattr(layer_mod, "load_places", lambda: shuffled)
    monkeypatch.setattr(layer_mod, "MAX_WEATHER_CITIES", 3)
    warmed = []
    monkeypatch.setattr(layer.weather, "ensure_fresh", lambda key, lat, lon: warmed.append(key))
    layer._load_places()
    capitals, cities = layer._places_pending
    assert [place[2] for place in capitals] == ["Tokyo", "Paris", "Seoul"]
    assert [place[2] for place in cities] == ["New York", "São Paulo", "Sydney", "York"]
    assert warmed == [city_key("Tokyo", 35.69, 139.75), city_key("New York", 40.75, -73.98),
                      city_key("São Paulo", -23.55, -46.63)]
    layer.dirty = False
    layer._poll_places()
    assert layer._places_state == "ready" and layer.dirty is True
    assert layer._capitals == capitals and layer._places_pending is None


@pytest.mark.parametrize("error", [MapDataError("404"), OSError("disk"), ValueError("bad geojson")])
def test_load_places_failure_schedules_a_retry(layer, monkeypatch, capsys, error):
    def fail():
        raise error

    monkeypatch.setattr(layer_mod, "load_places", fail)
    monkeypatch.setattr(layer_mod.time, "monotonic", lambda: 1000.0)
    layer._load_places()
    assert layer._places_state == "failed"
    assert layer._places_retry_at == 1000.0 + layer_mod.PLACES_RETRY_SECONDS
    assert "weather cities" in capsys.readouterr().err


def test_poll_places_retries_only_after_the_wait(layer, threads, monkeypatch):
    clock = types.SimpleNamespace(now=0.0)
    monkeypatch.setattr(layer_mod.time, "monotonic", lambda: clock.now)
    layer._poll_places()
    assert threads == []  # ordered startup owns the first load
    layer._ordered_startup = False
    layer._places_state, layer._places_retry_at = "failed", 50.0
    clock.now = 49.9
    layer._poll_places()
    assert threads == []
    clock.now = 50.0
    layer._poll_places()
    assert len(threads) == 1 and layer._places_state == "loading"


def test_set_prepared_places_detail_is_swapped_in_on_tick(layer, sample_places):
    layer.set_prepared_places_detail(([sample_places[0]], [sample_places[1]]))
    layer.tick()
    assert layer._capitals == [sample_places[0]] and layer._places_state == "ready"


def test_tick_forces_redraw_when_a_retry_is_due(layer, frozen_time, monkeypatch):
    monkeypatch.setattr(city_weather.time, "time", lambda: frozen_time.now)
    layer.weather._entries["k"] = {"fetched_at": NOW - 10_000, "error": "boom"}
    layer.dirty = False
    layer._last_retry_scan = NOW - 0.5
    layer.tick()
    assert layer.dirty is False  # scanned less than a second ago
    frozen_time.now += 1
    layer.tick()
    assert layer.dirty is True and layer._last_retry_scan == frozen_time.now


def test_tick_does_nothing_when_cities_are_hidden(layer, sample_places):
    layer.show_cities = False
    layer.set_prepared_places_detail(([sample_places[0]], []))
    layer.tick()
    assert layer._places_state == "idle"


# ---- loading messages and legend ------------------------------------------------------


def test_loading_messages(layer):
    assert layer.loading_messages() == []
    layer._places_state = "loading"
    layer.weather._in_progress.add("x")
    layer.radar._busy = True
    layer.refresh_in_progress = True
    assert layer.loading_messages() == [
        "LOADING WEATHER PLACES", "UPDATING CITY WEATHER", "LOADING RADAR", "LOADING EARTHQUAKES",
    ]
    layer._places_state = "failed"
    layer.weather._in_progress.clear()
    layer.weather._entries["k"] = {"fetched_at": 1.0, "error": "boom"}
    layer.radar._busy, layer.radar._failed = False, True
    layer.refresh_in_progress = False
    assert layer.loading_messages() == [
        "CITY LIST UNAVAILABLE · RETRYING", "WEATHER UNAVAILABLE · RETRYING", "RADAR UNAVAILABLE · RETRYING",
    ]


def test_legend_lines(layer):
    texts = [line.text for line in layer.legend_lines()]
    assert texts == ["COLD", "HOT", "NO DATA", "<3", "3-4.4", "4.5-5.9", "6-7.4", "7.5+"]
    swatches = [line.swatches[0] for line in layer.legend_lines()]
    assert swatches[-1] == earthquakes.MAGNITUDE_COLOR_MAJOR
    assert swatches[0] == CITY_CONFIG["temperature_cold_color"]
    layer._places_state = "loading"
    assert layer.legend_lines()[3].text == "loading cities..."
    layer._places_state = "failed"
    assert layer.legend_lines()[3].text == "cities unavailable"
    layer.show_quakes = False
    assert len(layer.legend_lines()) == 4
    layer.show_cities = False
    assert layer.legend_lines() == []


# ---- city labels and placement ------------------------------------------------------------


@pytest.mark.parametrize(
    "temperature",
    ["12.3", "warm", None, True, float("nan"), float("inf")],
    ids=["numeric-string", "text", "none", "bool", "nan", "inf"],
)
def test_selected_city_with_a_malformed_temperature_gets_a_plain_label(layer, temperature):
    # PR #10: legacy cache entries can hold strings; the label must not crash.
    layer.weather._entries["k"] = _reading(temperature=temperature)
    assert layer._city_label("k", "Tokyo", selected=True) == (f"{MARKER_PREFIX}Tokyo", None)


def test_selected_city_label_shows_icon_and_temperature(layer):
    layer.weather._entries["k"] = _reading(temperature=-0.4, weathercode=71)
    glyph, color = WEATHER_ICONS["snow"]
    assert layer._city_label("k", "Oslo", selected=True) == (f"{glyph} Oslo -0°C", (glyph, color))
    assert layer._city_label("k", "Oslo", selected=False) == (f"{MARKER_PREFIX}Oslo", None)
    assert layer._city_label("missing", "Oslo", selected=True) == (f"{MARKER_PREFIX}Oslo", None)


def test_place_cities_respects_capital_zoom_and_population_floors(layer, sample_places):
    _set_places(layer, sample_places)
    world = View(0.0, 20.0, 0.5, 200, 60)
    assert layer._place_cities(world) == []  # below capital_zoom
    names = [city["name"] for city, *_ in layer._place_cities(View(30.0, 20.0, 1.0, 200, 40))]
    # capitals >= 2M only, no cities yet; Seoul's label would overlap Tokyo's on the same row
    assert set(names) == {"Tokyo", "Paris"}
    names = [city["name"] for city, *_ in layer._place_cities(View(-40.0, 40.0, 2.0, 400, 80))]
    assert "New York" in names and "York" not in names  # York is under the 5M floor


def test_small_capitals_wait_for_capital_all_zoom(layer):
    small = (0.0, 0.0, "Smallcap", True, layer_mod.CAPITAL_MIN_POPULATION - 1, "X")
    _set_places(layer, [(30.0, 10.0, "Bigcap", True, layer_mod.CAPITAL_MIN_POPULATION, "X"), small])
    below = View(15.0, 5.0, layer_mod.CAPITAL_ALL_ZOOM - 0.5, 200, 60)
    assert [city["name"] for city, *_ in layer._place_cities(below)] == ["Bigcap"]
    at = View(0.0, 0.0, layer_mod.CAPITAL_ALL_ZOOM, 200, 60)
    assert "Smallcap" in [city["name"] for city, *_ in layer._place_cities(at)]


def test_place_cities_returns_well_formed_entries(layer, sample_places):
    _set_places(layer, sample_places)
    view = View(139.75, 35.69, 4.0, 80, 30)
    placed = {city["name"]: (city, cell, label, icon) for city, cell, label, icon in layer._place_cities(view)}
    city, cell, label, icon = placed["Tokyo"]
    assert city == _city()
    assert cell == view.to_cell(139.75, 35.69)
    assert label == f"{MARKER_PREFIX}Tokyo" and icon is None


def test_place_cities_skips_offscreen_and_edge_clipped_labels(layer):
    view = View(0.0, 0.0, 4.0, 40, 20)
    edge_lon = view.origin_lon + view.lon_span * 0.95
    _set_places(layer, [
        (edge_lon, 0.0, "Edgeville", True, 9_000_000, "X"),
        (170.0, 0.0, "Faraway", True, 9_000_000, "X"),
    ])
    assert layer._place_cities(view) == []


def test_place_cities_drops_colliding_and_duplicate_labels(layer):
    view = View(0.0, 0.0, 4.0, 80, 30)
    _set_places(layer, [
        (0.0, 0.0, "Alpha", True, 9_000_000, "X"),
        (0.5, 0.0, "Beta", True, 8_000_000, "X"),  # same row, overlapping label
        (0.0, -0.4, "Gamma", True, 7_000_000, "X"),  # row below, overlapping columns
        (-10.0, 5.0, "Alpha", True, 6_000_000, "X"),  # duplicate name elsewhere
        (-10.0, -5.0, "Delta", True, 5_000_000, "X"),
    ])
    assert [city["name"] for city, *_ in layer._place_cities(view)] == ["Alpha", "Delta"]


def test_place_cities_caps_the_number_of_labels(layer, monkeypatch):
    monkeypatch.setattr(layer_mod, "MAX_WEATHER_CITIES", 2)
    _set_places(layer, [(-30.0 + 15 * index, 0.0, f"C{index}", True, 9_000_000 - index, "X") for index in range(5)])
    assert len(layer._place_cities(View(0.0, 0.0, 1.0, 200, 60))) == 2


def test_selected_city_is_placed_even_below_the_population_floor_and_over_a_collision(layer, monkeypatch):
    view = View(-1.08, 53.96, 4.0, 80, 30)
    york = _city("York", -1.08, 53.96, False, 153_717, "United Kingdom")
    _set_places(layer, [(-1.08, 53.96, "Leeds", True, 9_000_000, "United Kingdom")])  # same spot, bigger
    layer.selected = york
    placed = layer._place_cities(view)
    assert [city["name"] for city, *_ in placed] == ["York"]
    assert placed[0][0] == york
    monkeypatch.setattr(layer_mod, "MAX_WEATHER_CITIES", 1)
    layer.selected = None
    assert [city["name"] for city, *_ in layer._place_cities(view)] == ["Leeds"]
    layer.selected = york
    assert [city["name"] for city, *_ in layer._place_cities(view)] == ["York"]


def test_selected_city_at_the_right_edge_slides_in(layer):
    view = View(0.0, 0.0, 4.0, 40, 20)
    lon = view.origin_lon + view.lon_span * 0.97
    city = _city("Rightmost City", lon, 0.0, False, 100, "X")
    layer.selected = city
    layer.weather._entries[city["key"]] = _reading(temperature=31.0, weathercode=0)
    ((placed_city, (x, y), label, icon),) = layer._place_cities(view)
    assert label == f"{WEATHER_ICONS['clear'][0]} Rightmost City 31°C"
    assert icon == tuple(WEATHER_ICONS["clear"])
    assert x == view.width - len(label) and x + len(label) == view.width
    assert y == view.to_cell(lon, 0.0)[1]


def test_selected_city_offscreen_is_skipped_without_crashing(layer, sample_places):
    _set_places(layer, sample_places)
    layer.selected = _city("Sydney", 151.21, -33.87, False, 4_630_000, "Australia")
    names = [city["name"] for city, *_ in layer._place_cities(View(2.35, 48.86, 4.0, 60, 20))]
    assert "Sydney" not in names and "Paris" in names


def test_selected_quake_does_not_affect_city_placement(layer, sample_places):
    _set_places(layer, sample_places)
    view = View(0.0, 20.0, 1.0, 200, 60)
    without = layer._place_cities(view)
    layer.selected = _quake()
    assert layer._place_cities(view) == without


# ---- hit testing --------------------------------------------------------------------------


def test_hit_test_spans_the_whole_label_and_skips_radar_and_bare_cells(layer):
    city = _city()
    quake = _quake()
    layer.last_markers = [
        Marker(x=10, y=5, color="#fff", label="• Tokyo", item=city),
        Marker(x=40, y=5, color="#fff", glyph="●", item=quake),
        Marker(x=12, y=5, color="#fff", label=" ", item={"radar": True}),
        Marker(x=13, y=5, color="#fff", label="☀"),  # icon overlay, no item
    ]
    assert layer.hit_test_all(16, 5) == [(0.0, city)]  # last character of the label
    assert layer.hit_test_all(17, 5) == [(1.0, city)]
    assert layer.hit_test_all(18, 5) == []  # two cells past the label is beyond CLICK_RADIUS
    assert layer.hit_test_all(17, 5, radius=0.5) == []
    assert layer.hit_test_all(40, 6) == [(1.0, quake)]
    assert layer.hit_test_all(100, 30) == []


# ---- building markers --------------------------------------------------------------------


def test_city_markers_coloured_by_relative_temperature(layer, monkeypatch):
    monkeypatch.setattr(layer_mod, "RADAR_ENABLED", False)
    view = View(0.0, 0.0, 4.0, 80, 30)
    _set_places(layer, [
        (-10.0, 0.0, "Cold", True, 9_000_000, "X"),
        (10.0, 0.0, "Hot", True, 8_000_000, "X"),
        (0.0, 8.0, "Unknown", True, 7_000_000, "X"),
        (0.0, -8.0, "Legacy", True, 6_000_000, "X"),
    ])
    layer.weather._entries = {
        city_key("Cold", 0.0, -10.0): _reading(temperature=-5.0),
        city_key("Hot", 0.0, 10.0): _reading(temperature=30.0),
        city_key("Legacy", -8.0, 0.0): _reading(temperature="n/a"),
    }
    colours = {marker.item["name"]: marker.color for marker in layer.build(view).markers if marker.item}
    assert colours["Cold"] == CITY_CONFIG["temperature_cold_color"].lower()
    assert colours["Hot"] == CITY_CONFIG["temperature_hot_color"].lower()
    assert colours["Unknown"] == CITY_CONFIG["color"]
    assert colours["Legacy"] == CITY_CONFIG["color"]


def test_selected_city_gets_an_icon_marker_on_top(layer, monkeypatch):
    monkeypatch.setattr(layer_mod, "RADAR_ENABLED", False)
    city = _city()
    layer.selected = city
    layer.weather._entries[city["key"]] = _reading(weathercode=95)
    markers = layer._city_markers(View(139.75, 35.69, 4.0, 80, 30))
    label_marker, icon_marker = markers[0], markers[1]
    assert label_marker.selected and label_marker.item == city
    assert icon_marker.label == WEATHER_ICONS["storm"][0] and icon_marker.color == WEATHER_ICONS["storm"][1]
    assert (icon_marker.x, icon_marker.y) == (label_marker.x, label_marker.y) and icon_marker.selected


def test_city_markers_keep_label_cells_clear_of_radar(layer, monkeypatch):
    monkeypatch.setattr(layer_mod, "RADAR_ENABLED", True)
    seen = {}

    def fake_radar(view, blocked):
        seen["blocked"] = blocked
        return [Marker(x=0, y=0, color="#123456", label=" ")]

    monkeypatch.setattr(layer, "radar_markers", fake_radar)
    _set_places(layer, [(0.0, 0.0, "Alpha", True, 9_000_000, "X")])
    view = View(0.0, 0.0, 4.0, 80, 30)
    markers = layer._city_markers(view)
    x, y = view.to_cell(0.0, 0.0)
    assert seen["blocked"] == {(x + dx, y) for dx in range(len(f"{MARKER_PREFIX}Alpha"))}
    assert markers[0].color == "#123456" and markers[-1].item["name"] == "Alpha"  # radar underneath


def test_city_markers_warm_weather_only_when_online_and_started(layer, monkeypatch):
    monkeypatch.setattr(layer_mod, "RADAR_ENABLED", False)
    calls = []
    monkeypatch.setattr(layer.weather, "ensure_fresh", lambda key, lat, lon: calls.append(key))
    _set_places(layer, [(0.0, 0.0, "Alpha", True, 9_000_000, "X")])
    view = View(0.0, 0.0, 4.0, 80, 30)
    layer._startup_weather_ready = True
    layer._city_markers(view)
    assert calls == []  # offline
    monkeypatch.setattr(layer_mod, "FETCH_NEW_DATA_FROM_API", True)
    layer._startup_weather_ready = False
    layer._city_markers(view)
    assert calls == []  # startup not ready yet
    layer._startup_weather_ready = True
    layer._city_markers(view)
    assert calls == [city_key("Alpha", 0.0, 0.0)]


def test_quake_markers_sorted_weakest_first_with_dim_and_selection(layer, frozen_time):
    old = (NOW - earthquakes.QUAKE_DIM_AFTER_SECONDS - 1) * 1000
    layer.quakes = [
        _quake("big", 7.8, lon=1.0, time=NOW * 1000),
        _quake("none", None, lon=2.0, time="garbage"),
        _quake("old", 3.5, lon=3.0, time=old),
        _quake("far", 6.0, lon=170.0),
        _quake("tiny", 1.0, lon=4.0),  # below MIN_MAGNITUDE
    ]
    layer.selected = layer.quakes[2]
    layer.show_cities = False
    render = layer.build(View(0.0, 0.0, 4.0, 80, 30))
    assert render.lines is None
    ids = [marker.item["id"] for marker in render.markers]
    assert ids == ["none", "old", "big"]
    by_id = {marker.item["id"]: marker for marker in render.markers}
    assert by_id["old"].dim is True and by_id["big"].dim is False and by_id["none"].dim is False
    assert by_id["old"].selected is True and by_id["big"].selected is False
    assert by_id["big"].glyph == earthquakes.magnitude_to_glyph(7.8)
    assert by_id["none"].color == earthquakes.MAGNITUDE_COLOR_UNKNOWN


def test_quakes_hidden_when_toggled_off(layer):
    layer.quakes = [_quake()]
    layer.show_quakes = layer.show_cities = False
    assert layer.build(View(0.0, 0.0, 4.0, 80, 30)).markers == []


@pytest.mark.xfail(
    strict=True,
    reason="BUG: string magnitude from a tampered/legacy quake cache crashes visible_quakes (normalize_cached does not coerce)",
)
def test_quake_cache_with_string_magnitude_renders(layer, tmp_path):
    (tmp_path / "quakes.json").write_text(
        '{"last_fetch": 1, "quakes": [{"id": "a", "longitude": 1, "latitude": 1, "magnitude": "5.2"}]}',
        encoding="utf-8",
    )
    fresh = layer_mod.WeatherLayer()
    fresh.reload()
    fresh.build(View(0.0, 0.0, 4.0, 80, 30))


# ---- quake search / filter ---------------------------------------------------------------


def test_visible_quakes_min_magnitude_keeps_unknown(layer):
    layer.quakes = [_quake("low", 1.0), _quake("edge", earthquakes.MIN_MAGNITUDE), _quake("none", None)]
    assert [quake["id"] for quake in layer.visible_quakes()] == ["edge", "none"]


@pytest.mark.parametrize(
    "option, expected",
    [("3.0+", ["a", "b", "c"]), ("4.5+", ["b", "c"]), ("6.0+", ["c"]), ("7.5+", []), (None, ["a", "b", "c", "n"])],
)
def test_filter_options_threshold_drops_unknown_magnitudes(layer, option, expected):
    layer.quakes = [_quake("a", 3.0), _quake("b", 4.5), _quake("c", 6.2), _quake("n", None)]
    layer.apply_filter(option)
    assert [quake["id"] for quake in layer.visible_quakes()] == expected


@pytest.mark.parametrize("value", ["5.0+", "3.0", "", "abc", 4.5])
def test_apply_filter_rejects_unknown_options(layer, value):
    layer.apply_filter(value)
    assert layer.min_magnitude_filter is None
    layer.quakes = [_quake()]
    assert len(layer.visible_quakes()) == 1


def test_search_is_trimmed_and_case_insensitive(layer):
    layer.quakes = [_quake("a", place="50 km SW of Tokyo, Japan"), _quake("b", place="Alaska"), _quake("c", place="")]
    layer.apply_search("  tokyo ")
    assert [quake["id"] for quake in layer.visible_quakes()] == ["a"]
    layer.apply_search("   ")
    assert len(layer.visible_quakes()) == 3  # whitespace matches every place string


def test_filter_options_follow_the_quake_toggle(layer):
    assert layer.filter_options() == list(earthquakes.FILTER_OPTIONS)
    layer.show_quakes = False
    assert layer.filter_options() == []


# ---- info rows ---------------------------------------------------------------------------


def test_info_rows_empty_without_selection(layer):
    assert layer.info_rows() == []


def test_city_info_rows_with_a_good_reading(layer):
    city = _city()
    layer.selected = city
    layer.weather._entries[city["key"]] = _reading()
    assert layer.info_rows() == [
        ("NAME", "Tokyo"), ("COUNTRY", "Japan"), ("TYPE", "CAPITAL"),
        ("POPULATION", "35,676,000"), ("POSITION", "35.690, 139.750"),
        ("TEMP", "18.4°C"), ("CONDITIONS", "Partly cloudy"),
        ("WIND", "12 km/h @ 225°"), ("OBSERVED", "2026-10-19T12:00"),
    ]


def test_city_info_rows_before_any_reading(layer):
    layer.selected = _city("Paris", 2.35, 48.86, True, 9_904_000, None)
    rows = dict(layer.info_rows())
    assert rows["COUNTRY"] == "N/A" and rows["WEATHER"] == "FETCHING..."


@pytest.mark.parametrize(
    "temperature",
    ["18.4", "hot", None, False, float("nan"), float("-inf"), [18]],
    ids=["numeric-string", "text", "none", "bool", "nan", "-inf", "list"],
)
def test_city_info_rows_skip_non_numeric_temperatures(layer, temperature):
    # PR #10: a malformed stored temperature shows UNAVAILABLE instead of crashing.
    city = _city()
    layer.selected = city
    layer.weather._entries[city["key"]] = _reading(temperature=temperature)
    assert layer.info_rows()[-1] == ("WEATHER", "UNAVAILABLE")


@pytest.mark.parametrize(
    "wind, direction, text",
    [
        ("12", 90.0, "N/A km/h @ 90°"),
        (5.0, True, "5 km/h @ N/A°"),
        (None, float("nan"), "N/A km/h @ N/A°"),
        (float("inf"), "NW", "N/A km/h @ N/A°"),
    ],
)
def test_city_info_rows_fall_back_per_wind_field(layer, wind, direction, text):
    city = _city()
    layer.selected = city
    layer.weather._entries[city["key"]] = _reading(windspeed=wind, winddirection=direction, observed_at=None, weathercode="??")
    rows = dict(layer.info_rows())
    assert rows["WIND"] == text
    assert rows["OBSERVED"] == "N/A" and rows["CONDITIONS"] == "Unknown"


def test_quake_info_rows(layer, frozen_time, monkeypatch):
    monkeypatch.setattr(earthquakes.time, "time", lambda: NOW)
    layer.selected = _quake(
        magnitude=6.25, place=None, time=(NOW - 7200) * 1000, tsunami=True, alert="red", status=None,
        lon=-70.12345, lat=-33.5,
    )
    assert layer.info_rows() == [
        ("MAGNITUDE", "M 6.2"), ("PLACE", "UNKNOWN"), ("DEPTH", "10.0 km"), ("TIME", "2.0h ago"),
        ("TSUNAMI", "WARNING"), ("ALERT", "RED"), ("STATUS", "N/A"), ("POSITION", "-33.500, -70.123"),
    ]


def test_radar_info_rows(layer):
    layer.selected = {
        "radar": True, "name": "Rain", "kind": "OBSERVED", "intensity": 0.426,
        "latitude": 51.5, "longitude": -0.12, "time": 1_760_000_000,
    }
    assert layer.info_rows() == [
        ("SOURCE", "RAINVIEWER.COM"), ("NAME", "Rain"), ("KIND", "OBSERVED"), ("INTENSITY", "43%"),
        ("POSITION", "51.50, -0.12"), ("FRAME (UTC)", "2025-10-09 08:53"),
    ]


# ---- radar frame selection and slider -------------------------------------------------------


def test_live_index_is_the_newest_observed_frame(layer):
    frames = [(_frame(1), None), (_frame(2), None), (_frame(3, nowcast=True), None)]
    assert layer._radar_live_index(frames) == 1
    assert layer._radar_live_index([(_frame(1, True), None), (_frame(2, True), None)]) == 1
    assert layer._catalog_live_index([_frame(1), _frame(2, True)]) == 0


def test_shown_index_pinned_cached_and_animated(layer, monkeypatch):
    frames = [(_frame(100), None), (_frame(200), None), (_frame(300), None)]
    layer._radar_pin = 190
    assert layer._radar_shown_index(frames) == 1
    layer._radar_pin = None
    layer._radar_index = 7
    monkeypatch.setattr(layer_mod, "ANIMATE", True)
    assert layer._radar_shown_index(frames) == 2  # clamped
    layer._radar_index = 0
    assert layer._radar_shown_index(frames) == 0
    layer.radar._using_cache, layer.radar._rasters = True, {1: None}
    assert layer._radar_shown_index(frames) == 2  # cached fallback shows the latest, never animates
    monkeypatch.setattr(layer_mod, "ANIMATE", False)
    layer.radar._using_cache = False
    assert layer._radar_shown_index(frames) == 2


def test_slider_hidden_when_layer_disabled(layer):
    layer.enabled = False
    assert layer.slider() is None


def test_slider_disabled_while_loading_or_failed(layer):
    spec = layer.slider()
    assert spec.disabled and spec.status == "loading radar..." and spec.caption == "LIVE"
    layer.radar._failed = True
    assert layer.slider().status == "RADAR UNAVAILABLE"


def test_slider_single_cached_frame_shows_its_age(layer, frozen_time):
    _install_radar(layer, [int(NOW) - 125 * 60])
    layer.radar._using_cache = True
    spec = layer.slider()
    assert spec.disabled and spec.status == "CACHED"
    assert spec.caption == "2h05m AGO" and spec.loaded_steps == (0,)


def test_slider_over_a_fully_loaded_catalog(layer):
    _install_radar(layer, [600, 1200, 1800])
    spec = layer.slider()
    assert (spec.steps, spec.index, spec.marker, spec.caption, spec.status) == (3, 2, 2, "LIVE", "")
    assert spec.loaded_steps == (0, 1, 2) and spec.wide and not spec.disabled


def test_slider_while_frames_are_still_arriving(layer):
    _install_radar(layer, [600, 1200, 1800], loaded=[1800])
    spec = layer.slider()
    assert spec.status == "LOADING RADAR" and spec.loaded_steps == (2,)
    _install_radar(layer, [600, 1200, 1800], loaded=[])
    assert layer.slider().status == "loading radar..."
    layer.radar._failed = True
    assert layer.slider().status == "RADAR UNAVAILABLE"


def test_slider_pinned_to_an_older_frame(layer):
    _install_radar(layer, [600, 1200, 1800])
    layer.set_slider(0)
    spec = layer.slider()
    assert spec.index == 0 and spec.caption == "-20m"
    assert layer._radar_pin == 600 and layer._radar_index == 0


def test_slider_with_cached_catalog_shows_age(layer, frozen_time):
    stamps = [int(NOW) - 3 * 86400 - 600, int(NOW) - 3 * 86400]
    _install_radar(layer, stamps)
    layer.radar._using_cache = True
    spec = layer.slider()
    assert spec.status == "CACHED" and spec.caption == "3d AGO"


def test_set_slider_clamps_unpins_at_live_and_ignores_empty_catalog(layer):
    layer.set_slider(3)
    assert layer._radar_pin is None
    _install_radar(layer, [600, 1200, 1800], loaded=[600, 1800])
    layer.set_slider(-5)
    assert layer._radar_pin == 600
    layer.set_slider(1)
    assert layer._radar_pin == 1200
    assert layer._radar_index in (0, 1)  # nearest loaded frame
    layer.set_slider(99)
    assert layer._radar_pin is None and layer._radar_index == 1


@pytest.mark.parametrize(
    "seconds, caption",
    [(0, "0m AGO"), (59 * 60, "59m AGO"), (60 * 60, "1h00m AGO"), (23 * 3600 + 59 * 60, "23h59m AGO"),
     (24 * 3600, "1d AGO"), (-300, "0m AGO")],
)
def test_radar_age_caption(frozen_time, seconds, caption):
    assert layer_mod.WeatherLayer._radar_age_caption(int(NOW - seconds)) == caption


def test_radar_tick_is_static_by_default(layer, monkeypatch):
    monkeypatch.setattr(layer_mod, "RADAR_ENABLED", True)
    monkeypatch.setattr(layer_mod, "FETCH_NEW_DATA_FROM_API", True)
    monkeypatch.setattr(layer_mod, "ANIMATE", False)
    _install_radar(layer, [600, 1200])
    layer._radar_tick(NOW)
    assert layer._radar_index == 0


@pytest.mark.xfail(
    strict=True,
    reason="BUG: layer.py:482 uses FRAME_SECONDS / FRAME_HOLD_LAST_SECONDS without importing them (NameError when animate: true)",
)
def test_radar_tick_animates_through_loaded_frames(layer, monkeypatch):
    monkeypatch.setattr(layer_mod, "RADAR_ENABLED", True)
    monkeypatch.setattr(layer_mod, "FETCH_NEW_DATA_FROM_API", True)
    monkeypatch.setattr(layer_mod, "ANIMATE", True)
    _install_radar(layer, [600, 1200, 1800])
    layer._radar_last_step = 0.0
    layer._radar_tick(NOW)
    assert layer._radar_index == 1


# ---- radar markers ----------------------------------------------------------------------


def test_radar_markers_half_blocks_and_blocked_cells(layer, monkeypatch):
    _install_radar(layer, [600])
    low, high = layer_mod.INTENSITY_MIN, layer_mod.INTENSITY_MAX
    top = [0, high, high, high, 0, high, low - 1]
    bottom = [0, high, low, 0, high, high, low - 1]
    monkeypatch.setattr(layer_mod, "sample_raster", lambda raster, view: np.array([top, bottom], dtype=np.float32))
    view = View(0.0, 0.0, 1.0, 7, 1)
    markers = layer.radar_markers(view, blocked={(5, 0), (99, 99), (-1, 0)})
    by_x = {marker.x: marker for marker in markers}
    colors = radar.RADAR_LEVEL_COLORS
    assert sorted(by_x) == [1, 2, 3, 4]  # 0 empty, 5 blocked, 6 below threshold
    assert (by_x[1].label, by_x[1].color, by_x[1].bg_color) == (" ", colors[-1], colors[-1])
    assert (by_x[2].label, by_x[2].color, by_x[2].bg_color) == (layer_mod.HALF_BLOCK_UPPER, colors[-1], colors[0])
    assert (by_x[3].label, by_x[3].bg_color) == (layer_mod.HALF_BLOCK_UPPER, None)
    assert (by_x[4].label, by_x[4].color) == (layer_mod.HALF_BLOCK_LOWER, colors[-1])
    assert all(marker.item is None for marker in markers)


def test_radar_markers_empty_without_frames(layer):
    assert layer.radar_markers(View(0.0, 0.0, 1.0, 10, 5), set()) == []
