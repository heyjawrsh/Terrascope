"""FlightsLayer search, filtering, selection, history slider, airports and rendering."""

from __future__ import annotations

import time
import urllib.error
from types import SimpleNamespace

import pytest

from terrascope.core.layer import SidebarLine, SliderSpec
from terrascope.core.mapdata import MapDataError
from terrascope.core.view import BLANK_BRAILLE, View
from terrascope.layers.flights import api
from terrascope.layers.flights import cache as cache_module
from terrascope.layers.flights import layer as layer_module
from terrascope.layers.flights.cache import FlightCache
from terrascope.layers.flights.layer import FlightsLayer
from terrascope.layers.flights.settings import (
    AIRPORT_CONFIG,
    AIRPORT_GLYPH,
    AIRPORT_LOAD_RETRY_SECONDS,
    ARROW_GLYPHS,
    CATEGORY_COLORS,
    FLIGHT_CONFIG,
    OPEN_SKY_API_URL,
    UNKNOWN_HEADING_GLYPH,
)

NOW = 1_700_000_000.0


def flight(icao24, callsign="", country="Testland", lon=0.0, lat=0.0, **extra):
    return {
        "icao24": icao24,
        "callsign": callsign,
        "origin_country": country,
        "longitude": lon,
        "latitude": lat,
        "on_ground": False,
        **extra,
    }


def point(lon, lat, timestamp):
    return {"timestamp": timestamp, "longitude": lon, "latitude": lat}


def span(minutes, lon=0.0, lat=0.0):
    """Two history points `minutes` apart, ending now."""
    return [point(lon, lat, NOW - minutes * 60), point(lon + 1.0, lat, NOW)]


class SyncThread:
    """Stands in for threading.Thread: runs the target inside start()."""

    started: list[str] = []

    def __init__(self, target, name=None, daemon=None):
        self.target = target
        self.name = name

    def start(self):
        SyncThread.started.append(self.name)
        self.target()


@pytest.fixture(autouse=True)
def _frozen_and_quiet(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    monkeypatch.setattr(api, "set_live_progress", lambda *args, **kwargs: None)


@pytest.fixture
def sync_threads(monkeypatch):
    SyncThread.started = []
    monkeypatch.setattr(layer_module, "threading", SimpleNamespace(Thread=SyncThread))
    return SyncThread.started


@pytest.fixture
def layer(tmp_path):
    flights_layer = FlightsLayer()
    flights_layer.cache = FlightCache(tmp_path / "flight_cache.json")
    return flights_layer


def load(layer, flights, history=None):
    """Put flights (and history keyed by ICAO) in the cache and reload the layer."""
    layer.cache.flights = flights
    layer.cache.history = history or {}
    layer.reload()


def world_view(zoom=1.0, center_lon=0.0, center_lat=0.0):
    return View(center_lon, center_lat, zoom, 80, 24)


def lon_at_column(view, column):
    return view.origin_lon + (column + 0.5) * view.lon_span / view.width


# ---- reload and selection -------------------------------------------------------------


def test_reload_copies_flights_and_their_history(layer):
    load(layer, [flight("AAA111"), flight("BBB222")], {"AAA111": span(5), "ZZZ999": span(9)})
    assert [f["icao24"] for f in layer.flights] == ["AAA111", "BBB222"]
    assert layer.history == {"AAA111": span(5), "BBB222": []}
    assert layer.dirty is True


def test_reload_rebinds_selection_to_the_fresh_record(layer):
    load(layer, [flight("AAA111", "OLD1")])
    layer.select(layer.flights[0])
    load(layer, [flight("BBB222"), flight("AAA111", "NEW1", lon=5.0)])
    assert layer.selected["callsign"] == "NEW1"
    assert layer.selected is layer.flights[1]


def test_reload_keeps_the_last_state_of_an_aircraft_that_dropped_out(layer):
    load(layer, [flight("AAA111", "GONE1")])
    previous = layer.flights[0]
    layer.select(previous)
    load(layer, [flight("BBB222")])
    assert layer.selected is previous
    assert layer.info_rows()[0] == ("CALLSIGN", "GONE1")


@pytest.mark.parametrize("selected", [None, {"icao24": ""}, {"callsign": "NOICAO"}])
def test_reload_without_a_usable_selection_selects_nothing(layer, selected):
    layer.selected = selected
    load(layer, [flight("AAA111")])
    assert layer.selected is None


def test_reload_while_disabled_shows_no_flights(layer):
    layer.enabled = False
    load(layer, [flight("AAA111")], {"AAA111": span(5)})
    assert layer.flights == [] and layer.history == {}


def test_reload_clamps_the_slider_to_the_new_history_range(layer):
    layer.minutes_back = 30
    load(layer, [flight("AAA111")], {"AAA111": span(12)})
    assert layer.minutes_back == 12
    load(layer, [flight("AAA111")])
    assert layer.minutes_back == 0


def test_clear_drops_flights_history_and_selection(layer):
    load(layer, [flight("AAA111")], {"AAA111": span(5)})
    layer.select(layer.flights[0])
    layer.dirty = False
    layer.clear()
    assert (layer.flights, layer.history, layer.selected, layer.dirty) == ([], {}, None, True)


# ---- search and country filter -----------------------------------------------------------


FLEET = [
    flight("3C6514", "DLH400", "Germany"),
    flight("4CA2BF", "RYR8EK", "Ireland"),
    flight("39CEA0", "AFR1234", "France"),
    flight("39DE4F", "AFR66", "France"),
    flight("A1B2C3", "", "United States"),
    flight("A0A0A0", None, "United States"),
]


@pytest.mark.parametrize(
    ("search", "expected"),
    [
        ("", ["3C6514", "4CA2BF", "39CEA0", "39DE4F", "A1B2C3", "A0A0A0"]),
        ("afr", ["39CEA0", "39DE4F"]),
        ("  AFR66  ", ["39DE4F"]),
        ("dlh", ["3C6514"]),
        ("3c65", ["3C6514"]),
        ("a1b2", ["A1B2C3"]),
        ("39", ["39CEA0", "39DE4F"]),
        ("nothing", []),
    ],
)
def test_visible_flights_searches_callsign_and_icao_case_insensitively(layer, search, expected):
    load(layer, [dict(f) for f in FLEET])
    layer.apply_search(search)
    assert [f["icao24"] for f in layer.visible_flights()] == expected


@pytest.mark.parametrize(
    ("countries", "search", "expected"),
    [
        ({"France"}, "", ["39CEA0", "39DE4F"]),
        ({"France", "Ireland"}, "", ["4CA2BF", "39CEA0", "39DE4F"]),
        ({"France"}, "66", ["39DE4F"]),
        ({"Germany"}, "afr", []),
        ({"Ireland"}, "4ca", ["4CA2BF"]),
    ],
)
def test_country_filter_and_search_combine(layer, countries, search, expected):
    load(layer, [dict(f) for f in FLEET])
    layer.apply_filter(countries)
    layer.apply_search(search)
    assert [f["icao24"] for f in layer.visible_flights()] == expected


@pytest.mark.xfail(
    strict=True,
    reason="BUG: visible_flights() calls .lower() on the raw callsign; a non-string callsign (e.g. from the cache) raises AttributeError",
)
def test_visible_flights_survives_a_non_string_callsign(layer):
    load(layer, [flight("AAA111", 12345), flight("BBB222", "BAW1")])
    layer.apply_search("baw")
    assert [f["icao24"] for f in layer.visible_flights()] == ["BBB222"]


def ranked(layer, flights, query):
    load(layer, flights)
    layer.apply_search(query)
    return layer.search_target(query)


def test_search_target_prefers_exact_over_prefix_over_substring(layer):
    flights = [
        flight("AAA001", "XBAW1", lon=1.0),  # substring
        flight("AAA002", "BAW12", lon=2.0),  # prefix
        flight("AAA003", "BAW1", lon=3.0),  # exact
    ]
    assert ranked(layer, flights, "BAW1") == (3.0, 0.0)
    assert ranked(layer, flights[:2], "baw1") == (2.0, 0.0)
    assert ranked(layer, flights[:1], "baw1") == (1.0, 0.0)


def test_search_target_breaks_ties_by_list_order(layer):
    flights = [flight("AAA001", "BAW10", lon=1.0), flight("AAA002", "BAW11", lon=2.0)]
    assert ranked(layer, flights, "baw1") == (1.0, 0.0)
    assert ranked(layer, list(reversed(flights)), "baw1") == (2.0, 0.0)


def test_search_target_matches_icao_exactly_and_ignores_callsign_padding(layer):
    flights = [flight("4CA2BF", "4CA2BFX", lon=1.0), flight("ABC999", "X", lon=2.0), flight("4CA2BE", "4CA2BF  ", lon=3.0)]
    # Both the ICAO of the first and the (stripped) callsign of the third are exact; order decides.
    assert ranked(layer, flights, "4ca2bf") == (1.0, 0.0)
    assert ranked(layer, flights[1:], "4CA2BF") == (3.0, 0.0)


def test_search_target_respects_the_country_filter(layer):
    flights = [flight("AAA001", "AFR1", "France", lon=1.0), flight("AAA002", "AFR1X", "Spain", lon=2.0)]
    load(layer, flights)
    layer.apply_filter("Spain")
    layer.apply_search("afr1")
    assert layer.search_target("afr1") == (2.0, 0.0)


def test_search_target_uses_the_shown_position_when_known(layer):
    load(layer, [flight("AAA001", "BAW1", lon=1.0, lat=2.0)])
    layer._live = {"AAA001": (7.0, 8.0)}
    assert layer.search_target("BAW1") == (7.0, 8.0)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_target_ignores_blank_queries(layer, query):
    load(layer, [flight("AAA001", "BAW1")])
    assert layer.search_target(query) is None


def test_search_target_with_no_candidates_is_none(layer):
    load(layer, [flight("AAA001", "BAW1")])
    layer.apply_search("zzz")
    assert layer.search_target("zzz") is None


@pytest.mark.parametrize(("lon", "lat"), [("east", 0.0), (0.0, None), (float("nan"), 0.0)])
def test_search_target_rejects_unusable_coordinates(layer, lon, lat):
    layer.flights = [flight("AAA001", "BAW1", lon=lon, lat=lat)]
    assert layer.search_target("BAW1") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("France", {"France"}),
        (["France", "Ireland"], {"France", "Ireland"}),
        ({"Germany", "Atlantis"}, {"Germany"}),
        ("Atlantis", set()),
        (None, set()),
        ("", set()),
        ((), set()),
    ],
)
def test_apply_filter_keeps_only_known_countries(layer, value, expected):
    load(layer, [dict(f) for f in FLEET])
    layer.dirty = False
    layer.apply_filter(value)
    assert layer.filter_countries == expected
    assert layer.dirty is True


def test_filter_options_are_sorted_unique_and_skip_blank_countries(layer):
    layer.flights = [dict(f) for f in FLEET] + [flight("X1", country=""), flight("X2", country=None)]
    assert layer.filter_options() == ["France", "Germany", "Ireland", "United States"]


# ---- history slider -----------------------------------------------------------------------


def test_slider_is_disabled_without_history(layer):
    layer.minutes_back = 5
    assert layer.slider() == SliderSpec("PLANES", 2, 0, "LIVE", status="HISTORY NOT LOADED", disabled=True)
    assert layer.minutes_back == 0


def test_slider_spans_the_longest_history(layer):
    load(layer, [flight("AAA111"), flight("BBB222")], {"AAA111": span(4), "BBB222": span(10.9)})
    assert layer._history_range_minutes() == 10
    assert layer.slider() == SliderSpec("PLANES", 11, 10, "LIVE")


@pytest.mark.parametrize(
    ("index", "minutes_back", "caption"),
    [(10, 0, "LIVE"), (7, 3, "-3m"), (0, 10, "-10m"), (-5, 10, "-10m"), (99, 0, "LIVE")],
)
def test_set_slider_clamps_and_maps_index_to_minutes_back(layer, index, minutes_back, caption):
    load(layer, [flight("AAA111")], {"AAA111": span(10)})
    layer.set_slider(index)
    assert layer.minutes_back == minutes_back
    spec = layer.slider()
    assert (spec.index, spec.caption) == (10 - minutes_back, caption)


def test_slider_clamps_a_stale_minutes_back(layer):
    load(layer, [flight("AAA111")], {"AAA111": span(10)})
    layer.history = {"AAA111": span(3)}
    layer.minutes_back = 10
    assert layer.slider().caption == "-3m"
    assert layer.minutes_back == 3


@pytest.mark.parametrize(
    "points",
    [
        [],
        [point(0, 0, NOW)],
        [point(0, 0, NOW - 30), point(1, 0, NOW)],  # under a minute
        [point(0, 0, "garbage"), point(1, 0, NOW)],
        [point(0, 0, None), point(1, 0, float("nan")), point(2, 0, NOW)],
        [point(None, 0, NOW - 600), point(1, 0, NOW)],
    ],
)
def test_history_range_ignores_short_or_invalid_history(layer, points):
    layer.history = {"AAA111": points}
    assert layer._history_range_minutes() == 0


def test_history_range_skips_bad_timestamps_from_an_old_cache(layer, tmp_path):
    # Regression: invalid cached history timestamps must not crash the slider/render path.
    layer.cache.history = {
        "AAA111": [point(0, 0, "yesterday"), point(0.5, 0, NOW - 300), point(1, 0, {"t": 1}), point(2, 0, NOW)]
    }
    layer.cache.flights = [flight("AAA111")]
    layer.reload()
    assert layer.slider().steps == 6
    layer.set_slider(0)
    layer.build(world_view())


# ---- keys, ticking, legend, loading -----------------------------------------------------------


def test_toggle_key_switches_planes_and_airport_key_switches_airports(layer):
    load(layer, [flight("AAA111")])
    planes_key = ord(FLIGHT_CONFIG["toggle_key"])
    airports_key = ord(str(AIRPORT_CONFIG["toggle_key"]).upper())

    assert layer.handle_key(planes_key) is True
    assert layer.enabled is False and layer.flights == []
    assert layer.handle_key(planes_key) is True
    assert layer.enabled is True and [f["icao24"] for f in layer.flights] == ["AAA111"]

    shown = layer.show_airports
    assert layer.handle_key(airports_key) is True
    assert layer.show_airports is not shown


@pytest.mark.parametrize("key", [ord("?"), -1, 300, 0x1F600])
def test_other_keys_are_not_handled(layer, key):
    assert layer.handle_key(key) is False


def test_tick_applies_a_completed_refresh_and_swaps_in_airports(layer):
    layer.cache.flights = [flight("AAA111")]
    layer.refresh_completed = True
    layer._airports_pending = [(0.0, 0.0, "Alpha", "AAA", 0)]
    layer.tick()
    assert [f["icao24"] for f in layer.flights] == ["AAA111"]
    assert layer.refresh_completed is False
    assert layer._airports_state == "ready" and layer._airports == [(0.0, 0.0, "Alpha", "AAA", 0)]


def test_tick_does_not_start_airport_loading_during_ordered_startup(layer, sync_threads):
    layer.tick()
    assert sync_threads == [] and layer._airports_state == "idle"


def test_tick_leaves_pending_airports_alone_while_they_are_hidden(layer):
    layer.show_airports = False
    layer._airports_pending = [(0.0, 0.0, "Alpha", "AAA", 0)]
    layer.tick()
    assert layer._airports == [] and layer._airports_pending is not None


def test_is_due_and_refresh_delegate_to_the_cache(layer, monkeypatch):
    calls = []
    monkeypatch.setattr(layer.cache, "should_fetch", lambda: "due?")
    monkeypatch.setattr(layer.cache, "refresh_if_due", lambda force=False: calls.append(force) or True)
    assert layer.is_due() == "due?"
    assert layer.refresh_if_due() is True
    layer.enabled = False
    assert layer.refresh_if_due() is False
    assert layer.refresh_if_due(force=True) is True
    assert calls == [False, True]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a failed OpenSky fetch (e.g. HTTP 429) schedules no backoff; is_due() stays True so the next tick() requests again at once",
)
def test_failed_fetch_backs_off_before_the_next_request(layer, monkeypatch, fake_http):
    monkeypatch.setattr(cache_module, "FETCH_NEW_DATA_FROM_API", True)
    fake_http.responses[OPEN_SKY_API_URL] = urllib.error.HTTPError(OPEN_SKY_API_URL, 429, "Too Many Requests", {}, None)
    layer.refresh_in_progress = True  # as refresh_async() does before starting the worker
    layer._refresh_worker()
    assert len(fake_http.calls) == 1 and layer.cache.last_error
    assert time.monotonic() < layer._retry_at or not layer.is_due()


def test_legend_lists_categories_then_airports(layer):
    layer.show_airports = True
    lines = layer.legend_lines()
    assert [line.text for line in lines] == [
        "EMERGENCY", "HEAVY", "AIRLINER", "REGIONAL", "LIGHT", "HELICOPTER", "FAST/MIL", "OTHER", "AIRPORT",
    ]
    assert lines[0].swatches == (CATEGORY_COLORS["emergency"],)
    assert lines[-1].swatch_glyph == AIRPORT_GLYPH


@pytest.mark.parametrize(
    ("state", "extra"),
    [("idle", []), ("ready", []), ("loading", ["loading airports..."]), ("failed", ["airports unavailable"])],
)
def test_legend_reports_airport_loading_state(layer, state, extra):
    layer.show_airports = True
    layer._airports_state = state
    assert [line.text for line in layer.legend_lines()[9:]] == extra


def test_legend_without_airports_has_only_categories(layer):
    layer.show_airports = False
    layer._airports_state = "failed"
    assert len(layer.legend_lines()) == 8
    assert all(isinstance(line, SidebarLine) for line in layer.legend_lines())


@pytest.mark.parametrize(
    ("refreshing", "show", "state", "error", "expected"),
    [
        (False, True, "ready", "", []),
        (True, True, "ready", "", ["LOADING PLANES"]),
        (True, True, "loading", "", ["LOADING PLANES", "LOADING AIRPORTS"]),
        (False, True, "failed", "", ["AIRPORTS UNAVAILABLE"]),
        (False, True, "failed", "x" * 60, ["AIRPORTS UNAVAILABLE: " + "x" * 42]),
        (False, False, "failed", "boom", []),
    ],
)
def test_loading_messages(layer, refreshing, show, state, error, expected):
    layer.refresh_in_progress = refreshing
    layer.show_airports = show
    layer._airports_state = state
    layer._airports_error = error
    assert layer.loading_messages() == expected


# ---- background loading (run synchronously) -------------------------------------------------


def test_prefetch_waits_for_the_ordered_startup(layer, sync_threads, monkeypatch):
    monkeypatch.setattr(layer_module, "FETCH_NEW_DATA_FROM_API", True)
    layer.prefetch()
    assert sync_threads == []


def test_prefetch_loads_airports_and_warms_the_flight_cache(layer, sync_threads, monkeypatch):
    monkeypatch.setattr(layer_module, "FETCH_NEW_DATA_FROM_API", True)
    monkeypatch.setattr(layer.cache, "refresh_if_due", lambda force=False: force)
    layer._ordered_startup = False
    layer.show_airports = True

    layer.prefetch()

    assert sync_threads == ["terrascope-airports", "terrascope-planes-prefetch"]
    assert layer.refresh_completed is True and layer.refresh_in_progress is False
    assert layer._airports_state == "loading" and layer._airports_pending
    ranks = [airport[4] for airport in layer._airports_pending]
    assert ranks == sorted(ranks)

    layer.tick()
    assert layer._airports_state == "ready" and layer._airports


def test_prefetch_offline_loads_airports_only(layer, sync_threads):
    layer._ordered_startup = False
    layer.prefetch()
    assert sync_threads == ["terrascope-airports"]
    assert layer.refresh_in_progress is False


def test_prefetch_skips_planes_while_a_refresh_is_running(layer, sync_threads, monkeypatch):
    monkeypatch.setattr(layer_module, "FETCH_NEW_DATA_FROM_API", True)
    layer._ordered_startup = False
    layer.show_airports = False
    layer.refresh_in_progress = True
    layer.prefetch()
    assert sync_threads == []


def test_prefetch_worker_always_clears_the_in_progress_flag(layer, monkeypatch):
    def explode(force=False):
        raise RuntimeError("feed exploded")

    monkeypatch.setattr(layer.cache, "refresh_if_due", explode)
    layer.refresh_in_progress = True
    layer.dirty = False
    with pytest.raises(RuntimeError):
        layer._prefetch_live()
    assert layer.refresh_in_progress is False and layer.dirty is True
    assert layer.refresh_completed is False


@pytest.mark.parametrize("error", [MapDataError("no reference data"), OSError("disk"), ValueError("bad row")])
def test_airport_load_failure_is_recorded_for_retry(layer, monkeypatch, capsys, error):
    def fail():
        raise error

    monkeypatch.setattr(layer_module, "load_airports", fail)
    before = time.monotonic()
    layer._load_airports()
    assert layer._airports_state == "failed"
    assert layer._airports_error == str(error)
    assert layer._airports_retry_at >= before + AIRPORT_LOAD_RETRY_SECONDS
    assert f"airports: {error}" in capsys.readouterr().err


def test_failed_airports_retry_only_after_the_wait(layer, sync_threads, monkeypatch):
    monkeypatch.setattr(layer_module, "load_airports", lambda: [(1.0, 1.0, "B", "BBB", 5), (0.0, 0.0, "A", "AAA", 1)])
    layer._ordered_startup = False
    layer._airports_state = "failed"
    layer._airports_retry_at = time.monotonic() + 3600
    layer._poll_airports()
    assert sync_threads == []

    layer._airports_retry_at = 0.0
    layer._poll_airports()  # starts the (synchronous) load
    layer._poll_airports()  # swaps the result in
    assert sync_threads == ["terrascope-airports"]
    assert layer._airports_state == "ready" and layer._airports_error == ""
    assert [airport[3] for airport in layer._airports] == ["AAA", "BBB"]


# ---- airports on the map -------------------------------------------------------------------------


def airport_markers(layer, view, airports):
    layer._airports = sorted(airports, key=lambda airport: airport[4])
    return layer._airport_markers(view)


def test_no_airport_markers_before_the_first_zoom_level_or_without_data(layer):
    assert layer._airport_markers(world_view(5.0)) == []
    assert airport_markers(layer, world_view(0.9), [(0.0, 0.0, "Alpha", "AAA", 0)]) == []
    assert layer._airport_hits == []


def test_zoomed_out_airports_are_glyphs_filtered_by_scalerank(layer):
    view = world_view(1.0)
    markers = airport_markers(
        layer, view, [(0.0, 0.0, "Hub", "HUB", 0), (30.0, 10.0, "Small", "SML", 3), (200.0, 0.0, "Far", "FAR", 1)]
    )
    assert [(m.x, m.y, m.glyph, m.label) for m in markers] == [(*view.to_cell(0.0, 0.0), AIRPORT_GLYPH, None)]


def test_zoomed_in_airports_get_code_labels_and_avoid_collisions(layer):
    view = world_view(5.0)
    alpha = (0.0, 0.0, "Alpha International", "AAA", 0)
    crowded = (1.0, 0.0, "Bravo", "BBB", 1)  # within padding of AAA on the same row
    above = (0.0, view.lat_span / view.height, "Above", "ABV", 2)  # the row above also counts
    charlie = (10.0, 5.0, "Charlie", "CCC", 2)
    no_code = (-10.0, -5.0, "Nameless", "", 3)
    too_minor = (5.0, -5.0, "Grass strip", "GRS", 9)
    markers = airport_markers(layer, view, [alpha, crowded, above, charlie, no_code, too_minor])

    shown = {(m.label, m.glyph) for m in markers}
    assert shown == {(f"{AIRPORT_GLYPH} AAA", None), (f"{AIRPORT_GLYPH} CCC", None), (None, AIRPORT_GLYPH)}
    assert [hit[3] for hit in layer._airport_hits] == [alpha, charlie, no_code]
    x0, x1, row, _ = layer._airport_hits[0]
    assert (x0, row) == view.to_cell(0.0, 0.0) and x1 - x0 == len(f"{AIRPORT_GLYPH} AAA")


def test_label_that_would_cross_the_right_edge_falls_back_to_a_glyph(layer):
    view = world_view(5.0)
    edge = (lon_at_column(view, view.width - 2), 0.0, "Edge", "EDG", 0)
    (marker,) = airport_markers(layer, view, [edge])
    assert (marker.x, marker.label, marker.glyph) == (view.width - 2, None, AIRPORT_GLYPH)


def test_airport_marker_count_is_capped(layer, monkeypatch):
    monkeypatch.setattr(layer_module, "AIRPORT_MAX_LABELS", 2)
    view = world_view(1.0)
    airports = [(float(lon), 0.0, f"A{lon}", "", 0) for lon in (-60, -30, 0, 30, 60)]
    assert len(airport_markers(layer, view, airports)) == 2


def test_clicking_an_airport_toggles_its_full_name(layer):
    view = world_view(5.0)
    alpha = (0.0, 0.0, "Alpha International", "AAA", 0)
    airport_markers(layer, view, [alpha])
    x, y = view.to_cell(0.0, 0.0)

    assert layer.click_at(x + 1, y) is True
    (marker,) = layer._airport_markers(view)
    assert marker.label == f"{AIRPORT_GLYPH} Alpha International"

    assert layer.click_at(x + 10, y) is True  # the expanded label is wider
    (marker,) = layer._airport_markers(view)
    assert marker.label == f"{AIRPORT_GLYPH} AAA"


def test_clicks_that_miss_or_with_airports_hidden_are_ignored(layer):
    view = world_view(5.0)
    airport_markers(layer, view, [(0.0, 0.0, "Alpha", "AAA", 0)])
    x, y = view.to_cell(0.0, 0.0)
    assert layer.click_at(x, y + 3) is False
    assert layer.click_at(x - 1, y) is False
    layer.show_airports = False
    assert layer.click_at(x, y) is False
    assert layer._expanded_airports == set()


def test_expanded_label_near_the_right_edge_is_shifted_left(layer):
    view = world_view(5.0)
    name = "Very Long Airport Name"
    airport = (lon_at_column(view, view.width - 5), 0.0, name, "VLA", 0)
    layer._expanded_airports.add(airport)
    (marker,) = airport_markers(layer, view, [airport])
    assert marker.label == f"{AIRPORT_GLYPH} {name}"
    assert marker.x + len(marker.label) == view.width


def test_expanded_label_wider_than_the_view_keeps_the_short_code(layer):
    view = View(0.0, 0.0, 5.0, 12, 24)
    airport = (0.0, 0.0, "An Airport With A Long Name", "AWL", 0)
    layer._expanded_airports.add(airport)
    (marker,) = airport_markers(layer, view, [airport])
    assert marker.label == f"{AIRPORT_GLYPH} AWL"


# ---- build ---------------------------------------------------------------------------------------


def plane_markers(render):
    return [marker for marker in render.markers if marker.item is not None]


def test_build_zoomed_out_draws_braille_dots_and_coloured_markers(layer):
    layer.show_airports = False
    load(layer, [flight("AAA111", lon=10.0, lat=10.0, category=8), flight("BBB222", lon=-60.0, lat=-20.0)])
    view = world_view(1.0)
    render = layer.build(view)

    assert render.line_color == FLIGHT_CONFIG["color"]
    assert len(render.lines) == view.height
    assert sum(char != BLANK_BRAILLE for line in render.lines for char in line) == 2
    markers = {m.item["icao24"]: m for m in plane_markers(render)}
    assert markers["AAA111"].color == CATEGORY_COLORS["helicopter"]
    assert markers["BBB222"].color == CATEGORY_COLORS["other"]
    assert (markers["AAA111"].x, markers["AAA111"].y) == view.to_cell(10.0, 10.0)
    assert all(m.glyph is None for m in markers.values())
    assert layer._live == {"AAA111": (10.0, 10.0), "BBB222": (-60.0, -20.0)}


def test_build_zoomed_in_uses_heading_arrows_and_no_dots(layer):
    layer.show_airports = False
    load(layer, [flight("AAA111", true_track=90.0), flight("BBB222", lon=0.5, true_track=None)])
    render = layer.build(world_view(8.0))
    assert render.lines is None
    glyphs = {m.item["icao24"]: m.glyph for m in plane_markers(render)}
    assert glyphs == {"AAA111": ARROW_GLYPHS[2], "BBB222": UNKNOWN_HEADING_GLYPH}


def test_build_skips_offscreen_flights_and_respects_search(layer):
    layer.show_airports = False
    load(layer, [flight("AAA111", "KEEP1"), flight("BBB222", "KEEP2", lon=120.0), flight("CCC333", "DROP")])
    layer.apply_search("keep")
    render = layer.build(world_view(8.0))
    assert [m.item["icao24"] for m in plane_markers(render)] == ["AAA111"]


def test_build_zoomed_out_leaves_offscreen_flights_undrawn(layer):
    layer.show_airports = False
    load(layer, [flight("POLAR1", lat=85.0)])  # the world view stops at about 71 degrees
    render = layer.build(world_view(1.0))
    assert all(char == BLANK_BRAILLE for line in render.lines for char in line)
    assert plane_markers(render) == []


def test_build_shows_flights_across_the_antimeridian(layer):
    layer.show_airports = False
    load(layer, [flight("EAST01", lon=179.5), flight("WEST01", lon=-179.5)])
    render = layer.build(world_view(8.0, center_lon=180.0))
    xs = {m.item["icao24"]: m.x for m in plane_markers(render)}
    assert set(xs) == {"EAST01", "WEST01"} and xs["EAST01"] < xs["WEST01"]


def test_build_draws_emergencies_last_and_flags_the_selection(layer):
    layer.show_airports = False
    load(layer, [flight("EMG001", squawk="7700"), flight("AIR001", category=4), flight("LGT001", category=2)])
    layer.select(layer.flights[1])
    render = layer.build(world_view(8.0))
    order = [m.item["icao24"] for m in plane_markers(render)]
    assert order[-1] == "EMG001"
    assert order.index("LGT001") < order.index("AIR001")
    assert [m.item["icao24"] for m in plane_markers(render) if m.selected] == ["AIR001"]


def test_build_puts_airports_under_the_planes(layer):
    load(layer, [flight("AAA111")])
    layer.show_airports = True
    layer._airports = [(0.0, 0.0, "Alpha", "AAA", 0)]
    render = layer.build(world_view(1.0))
    assert render.markers[0].glyph == AIRPORT_GLYPH and render.markers[0].item is None
    assert render.markers[-1].item["icao24"] == "AAA111"


def test_build_in_the_past_interpolates_and_hides_untracked_aircraft(layer):
    layer.show_airports = False
    history = {
        "TRACK1": [point(0.0, 0.0, NOW - 600), point(10.0, 0.0, NOW)],
        "LATE01": [point(20.0, 0.0, NOW - 60), point(21.0, 0.0, NOW)],
    }
    load(layer, [flight("TRACK1", lon=10.0), flight("LATE01", lon=21.0), flight("NOHIST", lon=-5.0)], history)
    layer.set_slider(layer.slider().steps - 1 - 5)  # five minutes back
    assert layer.minutes_back == 5

    layer.build(world_view(1.0))
    assert layer._target == NOW - 300
    assert layer._live["TRACK1"] == pytest.approx((5.0, 0.0))
    assert "LATE01" not in layer._live
    assert layer._live["NOHIST"] == (-5.0, 0.0)


def test_build_live_shows_the_last_fix_without_dead_reckoning(layer):
    layer.show_airports = False
    moving = flight("AAA111", lon=1.0, lat=2.0, velocity=250.0, true_track=90.0, time_position=NOW - 600)
    load(layer, [moving], {"AAA111": [point(0.0, 2.0, NOW - 600), point(1.0, 2.0, NOW - 300)]})
    layer.build(world_view(1.0))
    assert layer._live["AAA111"] == (1.0, 2.0)


def test_trail_of_the_selected_aircraft_is_drawn_in_its_colour(layer):
    layer.show_airports = False
    trail = [point(-40.0, 0.0, NOW - 600), point(-20.0, 10.0, NOW - 300), point(99.0, 0.0, NOW + 60)]
    load(layer, [flight("AAA111", category=5), flight("BBB222", lon=30.0)], {"AAA111": trail})
    layer.select(layer.flights[0])
    render = layer.build(world_view(1.0))

    trail_markers = [m for m in render.markers if m.item is None]
    assert len(trail_markers) > 5
    assert {m.color for m in trail_markers} == {CATEGORY_COLORS["heavy"]}
    assert all(m.glyph and m.glyph != BLANK_BRAILLE for m in trail_markers)
    # The future point (t > shown time) is not part of the trail: nothing east of the plane.
    plane_x = world_view(1.0).to_cell(0.0, 0.0)[0]
    assert max(m.x for m in trail_markers) <= plane_x + 1


@pytest.mark.parametrize("selected", [None, {"icao24": ""}, {"icao24": "NOPE00", "longitude": 0.0, "latitude": 0.0}])
def test_no_trail_without_a_selection_with_history(layer, selected):
    load(layer, [flight("AAA111")], {"AAA111": span(10)})
    layer.selected = selected
    assert layer._trail_markers(world_view(1.0)) == []


def test_hit_testing_ignores_trail_cells(layer):
    layer.show_airports = False
    load(layer, [flight("AAA111")], {"AAA111": [point(-40.0, 0.0, NOW - 600), point(0.0, 0.0, NOW)]})
    layer.select(layer.flights[0])
    view = world_view(1.0)
    render = layer.render(view)
    assert any(m.item is None for m in render.markers)
    x, y = view.to_cell(0.0, 0.0)
    hits = layer.hit_test_all(x, y, radius=50)
    assert [item["icao24"] for _, item in hits] == ["AAA111"]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: cached history with numeric-string coordinates passes valid_history_point but crashes the trail (str + float in View.to_pixel)",
)
def test_trail_survives_numeric_string_coordinates_from_an_old_cache(layer):
    layer.show_airports = False
    load(layer, [flight("AAA111")], {"AAA111": [point("-40", "0", NOW - 600), point("-20", "5", NOW - 300)]})
    layer.select(layer.flights[0])
    layer.build(world_view(1.0))


# ---- info box ---------------------------------------------------------------------------------------


def test_info_rows_without_selection_are_empty(layer):
    assert layer.info_rows() == []


def test_info_rows_describe_the_selected_aircraft(layer):
    selected = flight(
        "4CA2BF", "RYR8EK", "Ireland", lon=-6.25, lat=53.4271,
        baro_altitude=1000.0, geo_altitude=1100.0, velocity=100.0, true_track=270.4,
        vertical_rate=-5.0, squawk="7700", position_source=2, category=4,
    )
    layer.selected = selected
    assert layer.info_rows() == [
        ("CALLSIGN", "RYR8EK"),
        ("TYPE", "emergency"),
        ("COUNTRY", "Ireland"),
        ("ALTITUDE", "3,281 ft"),
        ("SPEED", "194 kt"),
        ("HEADING", "270°"),
        ("VERT RATE", "-984 ft/min"),
        ("POSITION", "53.427, -6.250"),
        ("ICAO24", "4CA2BF"),
        ("SQUAWK", "7700"),
        ("GEO ALT", "3,609 ft"),
        ("SOURCE", "MLAT"),
    ]


def test_info_rows_fill_missing_values(layer):
    layer.selected = {"icao24": "", "callsign": None, "origin_country": "", "longitude": 1.0, "latitude": 2.0}
    rows = dict(layer.info_rows())
    assert rows["CALLSIGN"] == rows["COUNTRY"] == rows["ICAO24"] == "UNKNOWN"
    assert rows["TYPE"] == "other"
    assert rows["SQUAWK"] == rows["ALTITUDE"] == rows["SOURCE"] == "N/A"


def test_info_rows_use_the_shown_position(layer):
    layer.selected = flight("AAA111", lon=1.0, lat=2.0)
    layer._live = {"AAA111": (3.14159, -2.71828)}
    assert dict(layer.info_rows())["POSITION"] == "-2.718, 3.142"
