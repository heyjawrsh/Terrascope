"""TerrascopeApp drawing, layout, lifecycle, the main loop and the startup data handoff in main.py."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app_helpers import SAMPLE_FLIGHTS, TAB_INDEX, RecordedThread, layer, make_app, press  # noqa: F401
from conftest import FakeWindow
from terrascope.core.app import lifecycle, rendering, runtime

CLOCK = "2026-01-02 03:04:05 TEST"


@pytest.fixture(autouse=True)
def steady_screen(monkeypatch):
    """Draw with a fixed clock and no download / startup progress, whatever
    other tests did to the shared progress table."""
    monkeypatch.setattr(rendering, "current_time_text", lambda zone: CLOCK)
    monkeypatch.setattr(rendering, "download_progress", lambda: ())
    monkeypatch.setattr(rendering, "startup_progress", lambda: ("SETUP COMPLETE", 100, "ready", ""))
    monkeypatch.setattr(runtime, "download_progress", lambda: ())


def progress(filename, state="downloading", downloaded=0, total=None, error=""):
    return SimpleNamespace(filename=filename, state=state, downloaded=downloaded, total=total, error=error)


def drawn(app) -> str:
    app.draw()
    return app.stdscr.text()


# ---- size limits ---------------------------------------------------------------------------


@pytest.mark.parametrize("height, width", [(40, 59), (19, 120), (10, 20), (19, 59)])
def test_small_terminals_show_only_the_too_small_notice(make_app, height, width):
    app = make_app(height, width, welcome=True)
    assert drawn(app).strip() == "Terminal too small"
    assert app._welcome_rect is None


def test_the_minimum_size_draws_the_app(make_app):
    app = make_app(20, 60)
    text = drawn(app)
    assert "Terminal too small" not in text
    assert " 1 MAP " in text
    assert "WORLD MAP" in text


@pytest.mark.parametrize("tab", ["map", "time", "weather", "planes"])
@pytest.mark.parametrize("height, width", [(20, 60), (24, 80), (40, 120), (50, 200)])
def test_every_tab_draws_at_common_sizes(make_app, tab, height, width):
    app = make_app(height, width, tab=tab)
    text = drawn(app)
    assert "WORLD MAP" in text
    assert app.frame is not None
    assert app._frame_size is not None


# ---- welcome modal and tab bar ----------------------------------------------------------------


def test_welcome_modal_is_drawn_until_dismissed(make_app):
    app = make_app(welcome=True)
    assert "GITHUB" in drawn(app)
    assert app._welcome_rect is not None
    press(app, 27)
    assert "GITHUB" not in drawn(app)
    assert app._welcome_rect is None
    assert app._welcome_button is None


def test_welcome_modal_offers_province_borders_until_chosen(make_app):
    app = make_app(welcome=True)
    app.draw()
    assert app._welcome_province_yes is not None
    chosen = make_app(welcome=True, province_choice="yes")
    chosen.province_state = "ready"
    chosen.draw()
    assert chosen._welcome_province_yes is None


OSM_CREDIT = "© OpenStreetMap contributors"
CREDIT_BUG = pytest.mark.xfail(
    strict=True,
    reason="BUG: ui/map.py draws the OSM credit on the slider's footer row and the slider "
    "overwrites it, so tabs with a time slider lose the credit",
)


@pytest.mark.parametrize(
    "tab",
    [
        "map",
        pytest.param("time", marks=CREDIT_BUG),
        pytest.param("weather", marks=CREDIT_BUG),
        pytest.param("planes", marks=CREDIT_BUG),
    ],
)
def test_the_openstreetmap_credit_survives_the_slider(make_app, tab):
    app = make_app(45, 140, tab=tab)
    assert OSM_CREDIT in drawn(app)


def test_tab_bar_lists_every_tab_and_the_clock(make_app):
    text = drawn(make_app())
    assert " 1 MAP   2 TIME   3 WEATHER   4 PLANES " in text
    assert CLOCK in text


# ---- the province prompt and status lines ----------------------------------------------------


PROMPT = "STATE/PROVINCE BORDERS (~25MB)?  [Y YES]   [N NO]"


def test_province_prompt_is_drawn_on_the_map_tab_with_click_targets(make_app):
    app = make_app()
    text = drawn(app)
    assert PROMPT in text
    (row, yes_x0, yes_x1, yes), (row2, no_x0, no_x1, no) = app._province_prompt_hits
    line = app.stdscr.lines()[row]
    assert line[yes_x0:yes_x1] == "[Y YES]"
    assert line[no_x0:no_x1] == "[N NO]"
    assert (yes, no) == (True, False)


@pytest.mark.parametrize(
    "kwargs, state",
    [
        ({"tab": "time"}, "idle"),
        ({"province_choice": "no"}, "idle"),
        ({"province_choice": "yes"}, "loading"),
        ({"welcome": True}, "idle"),
    ],
)
def test_province_prompt_is_hidden_otherwise(make_app, kwargs, state):
    app = make_app(**kwargs)
    app.province_state = state
    assert PROMPT not in drawn(app)
    assert app._province_prompt_hits == []


def test_province_prompt_returns_after_a_failed_download(make_app):
    app = make_app(province_choice="yes")
    app.province_state = "failed"
    assert PROMPT in drawn(app)


def test_info_statuses_are_drawn_dim_without_buttons(make_app):
    app = make_app(province_choice="no")
    app.map_statuses["CITIES"] = "LOADING CITIES 10M"
    text = drawn(app)
    assert "LOADING CITIES 10M" in text
    assert "LOADING CITIES 10M [X]" not in text
    assert app._status_hits == []


def test_error_statuses_get_a_dismiss_button(make_app):
    app = make_app(province_choice="no")
    app.map_statuses["AIRPORTS"] = "LOADING AIRPORTS UNAVAILABLE: <urlopen error timed out>"
    text = drawn(app)
    assert "LOADING AIRPORTS UNAVAILABLE: request timed out [X]" in text  # error detail made concise
    [(row, x0, x1, key, action)] = app._status_hits
    assert app.stdscr.lines()[row][x0:x1] == "[X]"
    assert (key, action) == ("LOADING AIRPORTS UNAVAILABLE: <urlopen error timed out>", "dismiss")


def test_border_failures_get_retry_and_dismiss_buttons(make_app):
    app = make_app(tab="time", province_choice="yes")
    app.province_state = "loading"
    app.map_statuses["PROVINCE BORDERS"] = "PROVINCE BORDERS FAILED: boom"
    text = drawn(app)
    assert "PROVINCE BORDERS FAILED: boom [R] [X]" in text
    actions = {action: app.stdscr.lines()[row][x0:x1] for row, x0, x1, _, action in app._status_hits}
    assert actions == {"dismiss": "[X]", "retry-provinces": "[R]"}


def test_dismissed_and_duplicate_statuses(make_app):
    app = make_app(province_choice="no")
    app.map_statuses["A"] = "FIRST FAILED"
    app.map_statuses["B"] = "FIRST FAILED"
    app.map_statuses["C"] = "SECOND FAILED"
    text = drawn(app)
    assert text.count("FIRST FAILED [X]") == 1
    app.dismissed_statuses.add("FIRST FAILED")
    text = drawn(app)
    assert "FIRST FAILED" not in text
    assert "SECOND FAILED [X]" in text


def test_declined_borders_hide_border_statuses(make_app):
    app = make_app(tab="time", province_choice="no")
    app.map_statuses["BORDERS"] = "LOADING STATE/PROVINCE BORDERS"
    assert "BORDERS" not in drawn(app)


def test_map_error_is_shown(make_app):
    app = make_app(province_choice="no")
    app.map_error = "Could not load map data: offline"
    assert "Could not load map data: offline" in drawn(app)


def test_statuses_stop_at_the_bottom_of_the_map(make_app):
    app = make_app(20, 60, province_choice="no")
    for index in range(40):
        app.map_statuses[f"S{index}"] = f"STATUS NUMBER {index}"
    text = drawn(app)
    assert "STATUS NUMBER 0" in text
    assert "STATUS NUMBER 39" not in text


def test_statuses_are_hidden_behind_the_welcome_modal(make_app):
    app = make_app(welcome=True, province_choice="no")
    app.map_statuses["X"] = "SOMETHING FAILED"
    assert "SOMETHING FAILED" not in drawn(app)
    assert app._status_hits == []


def test_layer_loading_messages_are_shown(make_app):
    app = make_app(tab="planes")
    layer(app, "flights").refresh_in_progress = True
    assert "LOADING PLANES" in drawn(app)


# ---- radar status ---------------------------------------------------------------------------


def test_radar_failure_collapses_into_one_weather_tab_line(make_app):
    app = make_app(tab="weather")
    app.map_statuses["RADAR"] = "FETCHING PRESENT RADAR FRAME UNAVAILABLE: 503"
    text = drawn(app)
    assert "RADAR UNAVAILABLE [R RETRY] [-]" in text
    assert "503" not in text
    actions = {action: app.stdscr.lines()[row][x0:x1] for row, x0, x1, _, action in app._status_hits}
    assert actions == {"retry-radar": "[R RETRY]", "toggle-radar-status": "[-]"}
    app.radar_status_expanded = False
    text = drawn(app)
    assert "RADAR UNAVAILABLE [+]" in text
    assert [action for *_, action in app._status_hits] == ["toggle-radar-status"]


def test_radar_failure_is_not_shown_on_other_tabs(make_app):
    app = make_app(tab="planes")
    app.map_statuses["RADAR"] = "RADAR FAILED"
    assert "RADAR" not in drawn(app)


def test_radar_unavailable_from_the_layer_or_downloads(make_app, monkeypatch):
    app = make_app(tab="weather")
    weather = layer(app, "weather")
    weather.loading_messages = lambda: ["RADAR UNAVAILABLE · RETRYING", "UPDATING CITY WEATHER"]
    text = drawn(app)
    assert "RADAR UNAVAILABLE [R RETRY] [-]" in text
    assert "UPDATING CITY WEATHER" in text
    assert "RETRYING" not in text
    weather.loading_messages = lambda: []
    monkeypatch.setattr(rendering, "download_progress", lambda: (progress("WEATHER RADAR", "failed", error="x"),))
    assert "RADAR UNAVAILABLE [R RETRY] [-]" in drawn(app)


# ---- download progress ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item, line",
    [
        (progress("ne_50m_admin_0_countries.geojson", downloaded=50, total=100), "LOADING COUNTRIES 50M  50%"),
        (progress("ne_10m_admin_0_countries.geojson", downloaded=5, total=4), "LOADING COUNTRIES 10M 100%"),
        (progress("ne_10m_admin_1_states_provinces_lines.geojson"), "LOADING BORDERS 10M  ??%"),
        (progress("ne_50m_populated_places.geojson", downloaded=1, total=3), "LOADING PLACES 50M  33%"),
        (progress("ne_10m_airports.geojson", downloaded=0, total=10), "LOADING AIRPORTS   0%"),
        (progress("LIVE AIRCRAFT", downloaded=1, total=2), "LOADING AIRCRAFT  50%"),
        (progress("CITY WEATHER"), "LOADING CITY WEATHER  ??%"),
        (progress("ne_10m_lakes_big.geojson"), "LOADING 10M LAKES BIG  ??%"),
        (progress("ne_10m_airports.geojson", "failed", error="timed out"), "DOWNLOAD AIRPORTS FAILED: request timed out [X]"),
        (progress("ne_10m_airports.geojson", "failed"), "DOWNLOAD AIRPORTS FAILED [X]"),
    ],
)
def test_download_progress_lines(make_app, monkeypatch, item, line):
    monkeypatch.setattr(rendering, "download_progress", lambda: (item,))
    app = make_app(tab="time")
    assert line in drawn(app)


@pytest.mark.parametrize("state", ["ready", "waiting"])
def test_finished_or_waiting_downloads_are_not_listed(make_app, monkeypatch, state):
    monkeypatch.setattr(
        rendering, "download_progress", lambda: (progress("ne_10m_airports.geojson", state, 5, 10),)
    )
    assert "AIRPORTS" not in drawn(make_app(tab="time"))


def test_a_stage_status_gives_way_to_its_download_bar(make_app, monkeypatch):
    app = make_app(tab="time")
    app.map_statuses["MAP"] = "LOADING COUNTRIES 50M"
    app.map_statuses["CITIES"] = "LOADING CITIES 10M"
    monkeypatch.setattr(
        rendering, "download_progress",
        lambda: (progress("ne_50m_admin_0_countries.geojson", downloaded=1, total=4),),
    )
    text = drawn(app)
    assert "LOADING COUNTRIES 50M  25%" in text
    assert text.count("LOADING COUNTRIES 50M") == 1
    assert "LOADING CITIES 10M" in text


# ---- panel boxes ------------------------------------------------------------------------------------


def test_planes_panel_shows_info_search_and_filter(make_app):
    app = make_app(tab="planes")
    layer(app, "flights").flights = [dict(flight) for flight in SAMPLE_FLIGHTS]
    text = drawn(app)
    for piece in ("SEARCH PLANE", "FILTER COUNTRY", "CLICK A MARKER", "TOTAL PLANES: 4", "/ or click to search"):
        assert piece in text
    press(app, "/", "dlh")
    text = drawn(app)
    assert "dlh█" in text
    assert "Enter apply  Esc cancel" in text


def test_hidden_panel_gives_the_map_the_room(make_app):
    app = make_app(tab="planes")
    press(app, "h")
    text = drawn(app)
    assert "SEARCH PLANE" not in text
    assert "SHOW PANEL" in text
    assert app.layout(40, 120).boxes == ()


def test_horizontal_panel_sits_under_the_map(make_app):
    app = make_app(tab="planes")
    app.panel_orientation = "horizontal"
    layout = app.layout(40, 120)
    map_top, _, map_height, _ = layout.map
    assert all(rect[0] == map_top + map_height for _, rect in layout.boxes)
    assert "SEARCH PLANE" in drawn(app)


@pytest.mark.parametrize(
    "setup, kinds, title",
    [
        (lambda app: None, ["info", "search"], "SEARCH / FILTER"),
        (lambda app: press(app, "/"), ["info", "search"], "SEARCH PLANE"),
        (lambda app: press(app, "o"), ["info", "filter"], "FILTER COUNTRY"),
        (lambda app: setattr(app.world, "filter_value", "Ireland"), ["info", "filter"], "SEARCH / FILTER"),
    ],
)
def test_narrow_planes_panel_merges_search_and_filter(make_app, setup, kinds, title):
    app = make_app(30, 100, tab="planes")
    setup(app)
    layout = app.layout(30, 100)
    assert [box.kind for box, _ in layout.boxes] == kinds
    assert layout.boxes[1][0].title == title
    assert title in drawn(app)


def test_time_tab_draws_its_live_panel_and_slider(make_app):
    text = drawn(make_app(tab="time"))
    assert "TIME" in text
    assert "NOW" in text


def city(name: str, latitude: float, longitude: float) -> dict:
    return {
        "key": f"{name}|{latitude:.2f}|{longitude:.2f}",
        "name": name,
        "country": "",
        "capital": True,
        "population": 1_000_000,
        "latitude": latitude,
        "longitude": longitude,
    }


def test_forecast_scroll_resets_when_another_city_is_selected(make_app):
    app = make_app(tab="weather")
    weather = layer(app, "weather")
    weather.selected = city("Paris", 48.86, 2.35)
    app.forecast_scroll = 6
    text = drawn(app)
    assert app.forecast_scroll == 0
    assert "Paris" in text
    assert "SCROLL" in text  # the vertical forecast scrolls with the wheel
    app.forecast_scroll = 6
    app.draw()
    assert app.forecast_scroll == 6
    weather.selected = city("Oslo", 59.91, 10.75)
    app.draw()
    assert app.forecast_scroll == 0


# ---- colours and the resize cover ----------------------------------------------------------------


@pytest.mark.parametrize("age, resets", [(10.0, 1), (1.0, 0)])
def test_exhausted_colours_are_reset_at_most_every_few_seconds(make_app, monkeypatch, age, resets):
    app = make_app()
    calls = []
    app.colors.reset = lambda: calls.append("reset")
    app.colors.exhausted = True
    monkeypatch.setattr(rendering.time, "monotonic", lambda: 1000.0)
    app._colors_reset_at = 1000.0 - age
    app.draw()
    assert len(calls) == resets


def test_resize_cover_shows_the_new_size(make_app):
    app = make_app()
    cleared = []
    app.stdscr.clearok = cleared.append
    app.draw_resize_cover()
    assert "RESIZING  120 × 40" in app.stdscr.text()
    assert cleared == [True]


def test_resize_cover_in_a_tiny_window(make_app):
    app = make_app()
    app.stdscr = FakeWindow(3, 12)
    app.draw_resize_cover()
    assert app.stdscr.text().strip() == "12x3"


# ---- layout and panel owner -------------------------------------------------------------------------


@pytest.mark.parametrize("tab, owner", [("map", None), ("time", "night"), ("weather", "weather"), ("planes", "flights")])
def test_panel_owner_is_the_tabs_first_layer_with_a_panel(make_app, tab, owner):
    app = make_app(tab=tab)
    found = app.panel_owner()
    assert (found.name if found else None) == owner


def test_switching_off_the_owner_layer_drops_its_panel(make_app):
    app = make_app(tab="planes")
    press(app, "p")
    assert app.panel_owner() is None
    assert "SEARCH PLANE" not in drawn(app)


# ---- lifecycle: province download, radar retry, stderr, setup ---------------------------------------


def test_province_download_starts_only_once(make_app):
    app = make_app(province_choice="yes")
    app._start_province_download()
    app._start_province_download()
    assert len(make_app.threads) == 1
    app.province_state = "ready"
    app._start_province_download()
    assert len(make_app.threads) == 1


def test_province_download_worker_success_shows_the_borders(make_app, monkeypatch):
    from terrascope.core import mapdata

    monkeypatch.setattr(mapdata, "load_province_lines", lambda: [])
    app = make_app()
    press(app, "y")
    make_app.threads[0].run()
    assert app._collect_map_data() is True
    assert app.province_state == "ready"
    assert app.provinces_visible is True
    assert app.world.show_provinces is True
    assert layer(app, "basemap").provinces_ready is True
    assert "BORDERS" not in app.map_statuses


def test_province_download_worker_failure_reports_it(make_app, monkeypatch):
    from terrascope.core import mapdata

    def offline():
        raise mapdata.MapDataError("network disabled")

    monkeypatch.setattr(mapdata, "load_province_lines", offline)
    app = make_app()
    press(app, "y")
    make_app.threads[0].run()
    app._collect_map_data()
    assert app.province_state == "failed"
    assert app.province_error == "network disabled"
    assert app.map_statuses["PROVINCE BORDERS"] == "PROVINCE BORDERS FAILED: network disabled"
    assert "BORDERS" not in app.map_statuses
    assert "PROVINCE BORDERS FAILED: network disabled [R] [X]" in drawn(app)


def test_a_failed_border_download_bar_replaces_the_failure_status(make_app, monkeypatch):
    from terrascope.core.mapdata import PROVINCE_FILE

    monkeypatch.setattr(runtime, "download_progress", lambda: (progress(PROVINCE_FILE, "failed", error="x"),))
    app = make_app(province_choice="yes")
    app.map_data_queue.put(("province_borders", None, "x"))
    app._collect_map_data()
    assert app.province_state == "failed"
    assert "PROVINCE BORDERS" not in app.map_statuses


def test_late_borders_are_ignored_after_declining(make_app):
    app = make_app(province_choice="no")
    app.map_data_queue.put(("province_borders", ([], [], None), ""))
    app._collect_map_data()
    assert app.province_state == "idle"
    assert app.provinces_visible is False


def test_choosing_provinces_forgets_old_border_statuses(make_app):
    app = make_app()
    app.dismissed_statuses = {"PROVINCE BORDERS FAILED: x", "OTHER FAILED"}
    app.map_statuses["PROVINCE BORDERS"] = "PROVINCE BORDERS FAILED: x"
    app._choose_provinces(False)
    assert app.dismissed_statuses == {"OTHER FAILED"}
    assert "PROVINCE BORDERS" not in app.map_statuses


def test_choosing_provinces_survives_an_unwritable_cache(make_app, monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(lifecycle, "CACHE_DIR", blocker)
    app = make_app()
    app._choose_provinces(False)
    assert app.province_choice == "no"


def test_radar_retry_without_a_weather_layer_does_nothing(make_app):
    app = make_app(tab="weather")
    app.world.layers = [item for item in app.world.layers if item.name != "weather"]
    app.map_statuses["RADAR"] = "RADAR FAILED"
    app._retry_radar()
    assert app.map_statuses["RADAR"] == "RADAR FAILED"


def test_stderr_goes_to_the_log_while_running(make_app):
    app = make_app()
    original = sys.stderr
    app.redirect_stderr()
    try:
        assert sys.stderr is not original
        print("stray warning", file=sys.stderr)
    finally:
        app.restore_stderr()
    assert sys.stderr is original
    assert "stray warning" in (make_app.cache_dir / "terrascope.log").read_text()
    app.restore_stderr()  # a second restore is harmless
    assert sys.stderr is original


def test_stderr_stays_put_when_the_log_cannot_be_opened(make_app, monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(lifecycle, "CACHE_DIR", blocker)
    app = make_app()
    original = sys.stderr
    app.redirect_stderr()
    assert sys.stderr is original
    assert app._log_file is None


@pytest.fixture
def fake_curses(monkeypatch):
    import curses

    calls: list = []
    for name in ("curs_set", "mousemask", "mouseinterval"):
        monkeypatch.setattr(curses, name, lambda *args, name=name: calls.append((name, args)))
    return calls


def test_run_sets_up_and_restores_the_terminal(make_app, fake_curses):
    app = make_app()
    colors = []
    app.colors.setup = lambda: colors.append("setup")
    app.colors.restore = lambda: colors.append("restore")
    original = sys.stderr
    app.stdscr.keys = [ord("q")]
    app.run()
    assert app.running is False
    assert colors == ["setup", "restore"]
    assert ("curs_set", (0,)) in fake_curses
    assert sys.stderr is original
    assert app._log_file is None


def test_run_restores_the_terminal_after_a_crash(make_app, fake_curses):
    app = make_app()
    restored = []
    app.colors.setup = lambda: None
    app.colors.restore = lambda: restored.append(True)

    def crash():
        raise RuntimeError("boom")

    app._loop = crash
    original = sys.stderr
    with pytest.raises(RuntimeError):
        app.run()
    assert restored == [True]
    assert sys.stderr is original


# ---- the map-data queue --------------------------------------------------------------------------------


def ring(lon: float, lat: float):
    points = [(lon, lat), (lon + 1, lat), (lon + 1, lat + 1), (lon, lat)]
    return (points, lon, lat, lon + 1, lat + 1)


def test_an_empty_queue_needs_no_redraw(make_app):
    assert make_app()._collect_map_data() is False


def test_status_messages_set_and_clear(make_app):
    app = make_app()
    app.map_data_queue.put(("status", ("MAP", "LOADING COUNTRIES 50M"), ""))
    app.map_data_queue.put(("status", ("CITIES", "LOADING CITIES 10M"), ""))
    app.map_data_queue.put(("status", ("MAP", ""), ""))
    assert app._collect_map_data() is True
    assert app.map_statuses == {"CITIES": "LOADING CITIES 10M"}


def test_countries_arrive(make_app):
    from terrascope.core.world import prepare_ring_set

    app = make_app()
    app.map_loading = True
    app.map_data_queue.put(("countries_prepared", prepare_ring_set([ring(10, 10)]), ""))
    app._collect_map_data()
    assert app.map_loading is False
    assert app._map_data_received is True
    assert len(app.world.countries) == 1
    app.map_data_queue.put(("countries", [ring(1, 1), ring(2, 2)], ""))
    app._collect_map_data()
    assert len(app.world.countries) == 2


@pytest.mark.parametrize("kind", ["countries", "countries_prepared"])
def test_country_errors_are_reported(make_app, kind):
    app = make_app()
    app.map_loading = True
    app.map_data_queue.put((kind, [], "Could not load map data: offline"))
    app._collect_map_data()
    assert app.map_loading is False
    assert app.map_error == "Could not load map data: offline"
    assert app.world.countries == []


def test_province_and_detail_data_arrive(make_app):
    from terrascope.core.world import prepare_ring_set

    app = make_app()
    app.map_data_queue.put(("provinces", [ring(0, 0)], ""))
    app._collect_map_data()
    assert len(app.world.provinces) == 1
    app.map_data_queue.put(("provinces_prepared", prepare_ring_set([ring(0, 0), ring(5, 5)]), ""))
    app._collect_map_data()
    assert len(app.world.provinces) == 2
    app.map_data_queue.put(("provinces", [], "broken"))
    app.map_data_queue.put(("provinces_prepared", None, "broken"))
    app._collect_map_data()
    assert len(app.world.provinces) == 2
    app.map_data_queue.put(("detail", {"countries": [ring(3, 3)], "provinces": []}, ""))
    app._collect_map_data()
    assert len(app.world.countries) == 1
    assert app.world.provinces == []


def test_startup_layer_data_reaches_the_layers(make_app, sample_places):
    app = make_app(tab="planes")
    weather = layer(app, "weather")
    flights = layer(app, "flights")
    reloads = []
    flights.reload = lambda: reloads.append(True)
    capitals = [place for place in sample_places if place[3]]
    cities = [place for place in sample_places if not place[3]]
    airports = [("B", 0, 0, "b", 5), ("A", 0, 0, "a", 1)]
    app.map_data_queue.put(("startup_weather_places", (capitals, cities), ""))
    app.map_data_queue.put(("startup_airports", airports, ""))
    app.map_data_queue.put(("startup_flights", None, ""))
    app._collect_map_data()
    assert weather._places_pending == (capitals, cities)
    assert flights._airports_pending == [airports[1], airports[0]]
    assert reloads == [True]


# ---- the main loop ---------------------------------------------------------------------------------------


@pytest.fixture
def loop_app(make_app, monkeypatch):
    """An app whose loop runs on a fake clock: napms advances it, and layers
    don't tick for real (that would start loading threads)."""
    app = make_app()
    clock = SimpleNamespace(now=0.0, naps=[], hooks=[])

    def napms(ms):
        clock.naps.append(ms)
        clock.now += ms / 1000
        for hook in list(clock.hooks):
            hook(clock)

    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(runtime.curses, "napms", napms)
    ticks = []
    for item in app.world.layers:
        item.tick = lambda name=item.name: ticks.append(name)
    app.clock = clock
    app.ticks = ticks
    return app


def test_loop_handles_keys_draws_and_quits(loop_app):
    loop_app.stdscr.keys = [ord("2"), -1, ord("q")]
    loop_app._loop()
    assert loop_app.running is False
    assert loop_app.active_tab == TAB_INDEX["time"]
    assert " 2 TIME " in loop_app.stdscr.text()
    assert {"basemap", "night", "flights", "weather"} <= set(loop_app.ticks)


def test_loop_applies_queued_map_data_before_drawing(loop_app):
    loop_app.map_data_queue.put(("status", ("CITIES", "LOADING CITIES 10M"), ""))
    loop_app.province_choice = "no"
    loop_app.stdscr.keys = [-1, ord("q")]
    loop_app._loop()
    assert "LOADING CITIES 10M" in loop_app.stdscr.text()


def test_loop_decodes_sgr_mouse_reports(loop_app):
    x, y = 60, 20
    loop_app.stdscr.keys = [27, *map(ord, f"[<64;{x + 1};{y + 1}M"), -1, ord("q")]
    loop_app._loop()
    assert loop_app.world.zoom > 1


def test_loop_treats_a_lone_esc_as_a_key(loop_app):
    loop_app.welcome_visible = True
    loop_app.stdscr.keys = [27, -1, -1, ord("q")]
    loop_app._loop()
    assert loop_app.welcome_visible is False


def test_loop_sleeps_less_while_dragging(loop_app):
    from terrascope.core.app.shared import DRAG_FRAME_DELAY_MS, FRAME_DELAY_MS

    loop_app._mouse_event("press", 60, 20)
    loop_app.stdscr.keys = [-1, ord("q")]
    loop_app._loop()
    assert loop_app.clock.naps == [DRAG_FRAME_DELAY_MS]
    assert DRAG_FRAME_DELAY_MS < FRAME_DELAY_MS


def test_loop_covers_a_resize_then_repaints(loop_app):
    window = loop_app.stdscr
    cleared = []
    window.clearok = cleared.append
    screens = []

    def resize_then_quit(clock):
        if len(clock.naps) == 1:
            window.height, window.width = 30, 100
            window.erase()
        screens.append(window.text())
        if clock.now > 1.0 and not window.keys:
            window.keys.append(ord("q"))

    loop_app.clock.hooks.append(resize_then_quit)
    loop_app._loop()
    assert any("RESIZING  100 × 30" in screen for screen in screens)
    final = screens[-1]
    assert "RESIZING" not in final
    assert " 1 MAP " in final
    assert True in cleared
    assert len(final.splitlines()) <= 30


def test_loop_redraws_on_the_heartbeat_only(loop_app):
    draws = []
    real_draw = loop_app.draw
    loop_app.draw = lambda: (draws.append(loop_app.clock.now), real_draw())

    def quit_later(clock):
        if clock.now > 2.5:
            loop_app.stdscr.keys.append(ord("q"))

    loop_app.clock.hooks.append(quit_later)
    loop_app._loop()
    # first frame, then about one heartbeat a second while nothing changes
    assert 2 <= len(draws) <= 5


# ---- main.py: startup handoff --------------------------------------------------------------------------------


@pytest.fixture
def startup(make_app, monkeypatch, sample_places):
    """Run main._main with fake loaders and a synchronous startup thread;
    returns (app, queued messages, live progress calls)."""
    from terrascope.core import main, mapdata
    from terrascope.core.app import TerrascopeApp
    from terrascope.core.registry import build_layers

    live: list = []
    monkeypatch.setattr(mapdata, "load_map_data", lambda: [ring(0, 0)])
    monkeypatch.setattr(mapdata, "load_places", lambda: list(sample_places))
    monkeypatch.setattr(mapdata, "load_country_labels", lambda: [(2.35, 48.86, "FR", 5, "France")])
    monkeypatch.setattr(mapdata, "load_airports", lambda: [("CDG", 2.55, 49.0, "Charles de Gaulle", 1)])
    monkeypatch.setattr(mapdata, "set_live_progress", lambda name, **values: live.append((name, values["state"])))
    monkeypatch.setattr(main, "Thread", RecordedThread)
    started_apps: list = []
    monkeypatch.setattr(TerrascopeApp, "run", lambda self: started_apps.append(self))

    def launch(forced_off=(), prepare=lambda layers: None):
        layers = build_layers()
        for item in layers:
            item.startup = lambda: None  # cache reads belong to the layer tests
        prepare(layers)
        main._main(FakeWindow(), layers, list(forced_off))
        [app] = started_apps
        for thread in list(RecordedThread.started_threads):
            if thread.name == "terrascope-startup-downloads":
                thread.run()
        queued = []
        while not app.map_data_queue.empty():
            queued.append(app.map_data_queue.get_nowait())
        return app, queued, live

    return launch


def statuses(queued):
    return [data for kind, data, _ in queued if kind == "status"]


def test_startup_queues_map_data_in_order_offline(startup, sample_places):
    app, queued, live = startup()
    assert app.map_loading is True
    kinds = [kind for kind, _, _ in queued if kind != "status"]
    assert kinds == ["countries_prepared", "startup_weather_places", "startup_airports"]
    capitals, cities = next(data for kind, data, _ in queued if kind == "startup_weather_places")
    assert [place[2] for place in capitals] == ["Tokyo", "Paris", "Seoul"]
    assert "New York" in [place[2] for place in cities]
    stages = [name for name, message in statuses(queued) if message and "…" in message]
    assert stages == ["MAP", "CITIES", "AIRPORTS", "AIRCRAFT", "RADAR", "WEATHER", "RADAR HISTORY"]
    assert ("planes", "disabled") in live
    assert ("radar", "disabled") in live
    assert ("weather", "disabled") in live
    weather = layer(app, "weather")
    assert weather._startup_complete is True
    assert weather._ordered_startup is False
    assert layer(app, "basemap")._pending_detail is not None


def test_startup_queue_drives_the_app_to_a_loaded_map(startup):
    app, queued, _ = startup()
    for item in queued:
        app.map_data_queue.put(item)
    app._collect_map_data()
    assert app.map_loading is False
    assert len(app.world.countries) == 1
    assert app.map_statuses == {}
    assert app.map_error == ""


def test_startup_reports_a_map_failure_and_carries_on(startup, monkeypatch):
    from terrascope.core import mapdata

    def offline():
        raise mapdata.MapDataError("network disabled")

    monkeypatch.setattr(mapdata, "load_map_data", offline)
    app, queued, _ = startup()
    assert ("countries", [], "Could not load map data: network disabled") in queued
    for item in queued:
        app.map_data_queue.put(item)
    app._collect_map_data()
    assert app.map_error == "Could not load map data: network disabled"
    assert app.map_statuses["MAP"] == "LOADING COUNTRIES 50M UNAVAILABLE: network disabled"
    assert any(kind == "startup_airports" for kind, _, _ in queued)


def test_startup_respects_disabled_layers(startup):
    app, queued, live = startup(forced_off=["weather"])
    assert app.forced_off == frozenset({"weather"})
    app.select_tab(TAB_INDEX["weather"])
    assert not layer(app, "weather").enabled


@pytest.fixture
def online(monkeypatch):
    from terrascope.core import config
    from terrascope.layers.weather import radar

    monkeypatch.setattr(config, "FETCH_NEW_DATA_FROM_API", True)
    frames = [{"time": 100}, {"time": 200}, {"time": 300}, {"time": 400, "nowcast": True}]
    fetched: list = []
    monkeypatch.setattr(radar, "fetch_catalog", lambda mode: list(frames))

    def fetch_frame_raster(frame, mode):
        fetched.append(frame["time"])
        if frame["time"] == 100:
            raise OSError("one frame failed")
        return f"raster-{frame['time']}"

    monkeypatch.setattr(radar, "fetch_frame_raster", fetch_frame_raster)
    return SimpleNamespace(frames=frames, fetched=fetched)


def fake_live_layers(record):
    def prepare(layers):
        flights = next(item for item in layers if item.name == "flights")
        weather = next(item for item in layers if item.name == "weather")
        flights.cache.refresh_if_due = lambda force=False: record.append(("aircraft", force))
        weather.weather.ensure_fresh = lambda key, lat, lon: record.append(("weather", key))
        store = SimpleNamespace(
            _mode="test",
            catalog=[],
            installed=[],
            is_failed=False,
        )
        weather.radar = SimpleNamespace(
            _mode="test",
            install_catalog=lambda frames: store.catalog.extend(frames),
            install_raster=lambda frame, raster: store.installed.append((frame["time"], raster)),
            catalog_frames=lambda: list(store.catalog),
            loaded_frames=lambda: [({"time": t}, r) for t, r in store.installed],
            failed=lambda: store.is_failed,
            mark_failed=lambda error: (record.append(("radar-failed", error)), setattr(store, "is_failed", True)),
            ensure_fresh=lambda: record.append(("radar-refresh", None)),
        )
        record.append(("store", store))

    return prepare


def test_online_startup_fetches_live_feeds_and_warms_radar_history(startup, online):
    record: list = []
    app, queued, live = startup(prepare=fake_live_layers(record))
    store = next(value for name, value in record if name == "store")
    assert ("aircraft", True) in record
    assert ("startup_flights", None, "") in queued
    assert store.installed[0] == (300, "raster-300")  # the newest observed frame first
    assert ("radar", "ready") in live
    weather_fetches = [value for name, value in record if name == "weather"]
    assert weather_fetches and weather_fetches[0].startswith("Tokyo|")
    assert layer(app, "weather")._startup_weather_ready is True
    history = [thread for thread in RecordedThread.started_threads if thread.name == "terrascope-radar-history"]
    assert len(history) == 1
    history[0].run()
    assert (200, "raster-200") in store.installed
    assert 100 in online.fetched  # tried, failed, skipped
    assert online.fetched.count(300) == 1  # already loaded: not fetched again


def test_online_startup_survives_a_radar_outage(startup, online, monkeypatch):
    from terrascope.layers.weather import radar

    def outage(mode):
        raise OSError("503")

    monkeypatch.setattr(radar, "fetch_catalog", outage)
    record: list = []
    app, queued, live = startup(prepare=fake_live_layers(record))
    assert ("radar-failed", "503") in record
    assert ("radar", "failed") in live
    assert ("radar", "ready") in live  # the setup bar still completes
    assert ("RADAR", "FETCHING PRESENT RADAR FRAME UNAVAILABLE: 503") in statuses(queued)
    assert ("radar-refresh", None) in record  # the normal retry path takes over


def test_online_startup_with_weather_disabled_skips_its_feeds(startup, online):
    record: list = []
    app, queued, live = startup(forced_off=["weather"], prepare=fake_live_layers(record))
    assert not [name for name, _ in record if name in ("weather", "radar-refresh")]
    assert ("radar", "disabled") in live
    assert ("weather", "disabled") in live
    assert ("radar-history", "disabled") in live


def test_launch_hands_control_to_curses(monkeypatch):
    import curses

    from terrascope.core import main

    calls = []
    monkeypatch.setattr(curses, "wrapper", lambda *args: calls.append(args))
    main.launch(["layers"], {"weather"})
    assert calls == [(main._main, ["layers"], {"weather"})]
