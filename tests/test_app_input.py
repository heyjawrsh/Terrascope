"""Keyboard routing in TerrascopeApp: modals, tabs, text modes, map keys and layer hotkeys."""

from __future__ import annotations

import curses

import pytest

from app_helpers import SAMPLE_FLIGHTS, TAB_INDEX, layer, make_app, press  # noqa: F401

ESC = 27
ENTER = 10
BACKSPACE = 127


def hotkey_state(app) -> dict[str, object]:
    night = layer(app, "night")
    weather = layer(app, "weather")
    flights = layer(app, "flights")
    return {
        "night": night.enabled,
        "look": night.look,
        "cities": weather.show_cities,
        "quakes": weather.show_quakes,
        "flights": flights.enabled,
        "airports": flights.show_airports,
    }


def planes_app(make_app, **kwargs):
    app = make_app(tab="planes", **kwargs)
    layer(app, "flights").flights = [dict(flight) for flight in SAMPLE_FLIGHTS]
    return app


# ---- tabs --------------------------------------------------------------------


@pytest.mark.parametrize(
    "tab, enabled",
    [
        ("map", {"basemap"}),
        ("time", {"basemap", "night"}),
        ("weather", {"basemap", "weather"}),
        ("planes", {"basemap", "flights"}),
    ],
)
def test_select_tab_switches_on_exactly_the_tabs_layers(make_app, tab, enabled):
    app = make_app(tab=tab)
    assert app.active_tab == TAB_INDEX[tab]
    assert {item.name for item in app.world.layers if item.enabled} == enabled


def test_forced_off_layers_stay_off_whatever_the_tab_says(make_app):
    app = make_app(tab="planes", forced_off=frozenset({"flights"}))
    assert not layer(app, "flights").enabled
    app.select_tab(TAB_INDEX["weather"])
    app.select_tab(TAB_INDEX["planes"])
    assert not layer(app, "flights").enabled


def test_select_tab_resets_selection_info_and_drag(make_app):
    app = make_app()
    app.info_visible = True
    app._drag = [1, 1, 1, 1, False, 0.0]
    app.world._selection_candidates = [(layer(app, "basemap"), "x")]
    app.select_tab(TAB_INDEX["time"])
    assert app.info_visible is False
    assert app._drag is None
    assert app.world.selection_position == (0, 0)


@pytest.mark.parametrize("index", [-1, 4, 99])
def test_select_tab_ignores_out_of_range_indices(make_app, index):
    app = make_app(tab="time")
    app.select_tab(index)
    assert app.active_tab == TAB_INDEX["time"]


def test_province_borders_only_show_on_the_map_tab(make_app):
    app = make_app()
    app.provinces_visible = True
    app.select_tab(TAB_INDEX["map"])
    assert app.world.show_provinces is True
    assert layer(app, "basemap").provinces_visible is True
    app.select_tab(TAB_INDEX["weather"])
    assert app.world.show_provinces is False
    assert layer(app, "basemap").provinces_visible is False


def test_tab_flags_follow_the_tab_definition(make_app):
    app = make_app(tab="weather")
    assert app.world.show_places is False
    assert app.world.show_countries is False
    assert app.world.detailed is False
    app.select_tab(TAB_INDEX["map"])
    assert app.world.show_places is True
    assert app.world.detailed is True


@pytest.mark.parametrize("key, tab", [("1", 0), ("2", 1), ("3", 2), ("4", 3)])
def test_number_keys_switch_tabs(make_app, key, tab):
    app = make_app(tab="planes" if tab != 3 else "map")
    press(app, key)
    assert app.active_tab == tab


@pytest.mark.parametrize("key", ["5", "6", "9"])
def test_number_keys_past_the_last_tab_do_nothing(make_app, key):
    app = make_app(tab="time")
    before = hotkey_state(app)
    press(app, key)
    assert app.active_tab == TAB_INDEX["time"]
    assert hotkey_state(app) == before


# ---- welcome modal -------------------------------------------------------------


@pytest.mark.parametrize("key", ["2", "/", "w", "+", "h", "o", "z", "x"])
def test_welcome_modal_swallows_other_keys(make_app, key):
    app = make_app(welcome=True)
    view = (app.world.center_lon, app.world.center_lat, app.world.zoom)
    press(app, key)
    assert app.welcome_visible is True
    assert app.active_tab == 0
    assert app.mode == "normal"
    assert app.panel_visible is True
    assert (app.world.center_lon, app.world.center_lat, app.world.zoom) == view


@pytest.mark.parametrize("key", [ESC, ord("?")])
def test_esc_or_question_mark_dismisses_the_welcome_modal(make_app, key):
    app = make_app(welcome=True)
    press(app, key)
    assert app.welcome_visible is False
    assert app.running is True


@pytest.mark.parametrize("key", ["q", "Q"])
def test_q_quits_even_from_the_welcome_modal(make_app, key):
    app = make_app(welcome=True)
    press(app, key)
    assert app.running is False


def test_welcome_province_keys_need_the_prompt_on_screen(make_app):
    app = make_app(welcome=True)
    assert app._welcome_province_yes is None  # nothing drawn yet
    press(app, "y")
    assert app.welcome_visible is True
    assert app.province_choice is None


@pytest.mark.parametrize("key, choice", [("y", "yes"), ("N", "no")])
def test_welcome_province_prompt_answers_and_closes(make_app, key, choice):
    app = make_app(welcome=True)
    app.draw()
    assert app._welcome_province_yes is not None
    press(app, key)
    assert app.welcome_visible is False
    assert app.province_choice == choice
    assert (make_app.cache_dir / "province-borders-choice").read_text() == choice
    assert len(make_app.threads) == (1 if choice == "yes" else 0)


def test_welcome_modal_routes_mouse_events_to_the_mouse_handler(make_app):
    app = make_app(welcome=True)
    calls = []
    app.handle_mouse = lambda: calls.append("mouse")
    press(app, curses.KEY_MOUSE)
    assert calls == ["mouse"]
    assert app.welcome_visible is True


def test_welcome_modal_takes_precedence_over_the_timezone_dropdown(make_app):
    app = make_app(tab="time", welcome=True)
    app.open_tz_dropdown()
    press(app, "a")
    assert app.tz_query == ""
    press(app, ESC)
    assert app.welcome_visible is False
    assert app.tz_dropdown_open is True
    press(app, ESC)
    assert app.tz_dropdown_open is False


# ---- timezone dropdown as a modal ------------------------------------------------


def test_z_opens_the_timezone_dropdown_only_on_the_time_tab(make_app):
    app = make_app(tab="time")
    press(app, "z")
    assert app.tz_dropdown_open is True
    for tab in ("map", "weather", "planes"):
        other = make_app(tab=tab)
        press(other, "Z")
        assert other.tz_dropdown_open is False


def test_open_dropdown_captures_keys_that_would_otherwise_act(make_app):
    app = make_app(tab="time")
    press(app, "z", "q2/")
    assert app.running is True
    assert app.active_tab == TAB_INDEX["time"]
    assert app.mode == "normal"
    assert app.tz_query == "q2/"


def test_open_dropdown_routes_mouse_events_to_the_mouse_handler(make_app):
    app = make_app(tab="time")
    app.open_tz_dropdown()
    calls = []
    app.handle_mouse = lambda: calls.append("mouse")
    press(app, curses.KEY_MOUSE)
    assert calls == ["mouse"]


def test_dropdown_takes_precedence_over_a_text_mode(make_app):
    app = planes_app(make_app)
    press(app, "/")
    app.open_tz_dropdown()  # e.g. the clock was clicked while searching
    press(app, "ab")
    assert app.tz_query == "ab"
    assert app.search_buffer == ""
    press(app, ESC)
    assert app.tz_dropdown_open is False
    assert app.mode == "search"
    press(app, "ab")
    assert app.search_buffer == "ab"


# ---- search / filter: the issue #3 regression ------------------------------------


@pytest.mark.parametrize("tab", ["map", "time", "weather"])
@pytest.mark.parametrize("key", ["/", "o", "O"])
def test_search_and_filter_keys_do_not_capture_input_off_the_planes_tab(make_app, tab, key):
    app = make_app(tab=tab, province_choice="no")
    press(app, key)
    assert app.mode == "normal"
    press(app, "4")  # typing goes on working: a tab key still switches tabs
    assert app.active_tab == TAB_INDEX["planes"]


@pytest.mark.parametrize("key, mode", [("/", "search"), ("o", "filter"), ("O", "filter")])
def test_search_and_filter_keys_open_their_prompt_on_the_planes_tab(make_app, key, mode):
    app = planes_app(make_app)
    press(app, key)
    assert app.mode == mode


def test_text_mode_swallows_command_keys_as_text(make_app):
    app = planes_app(make_app)
    press(app, "/", "q1h?")
    assert app.running is True
    assert app.active_tab == TAB_INDEX["planes"]
    assert app.panel_visible is True
    assert app.welcome_visible is False
    assert app.search_buffer == "q1h?"


def test_text_mode_ignores_non_printable_keys(make_app):
    app = planes_app(make_app)
    press(app, "/", "ab", curses.KEY_UP, 1, 200)
    assert app.search_buffer == "ab"


def test_backspace_edits_the_search_buffer(make_app):
    app = planes_app(make_app)
    press(app, "/", "dlhx", BACKSPACE, curses.KEY_BACKSPACE)
    assert app.search_buffer == "dl"
    press(app, 8, 8, 8)
    assert app.search_buffer == ""


def test_esc_cancels_search_without_changing_the_applied_search(make_app):
    app = planes_app(make_app)
    app.world.set_search("EIN")
    press(app, "/")
    assert app.search_buffer == "EIN"  # the prompt starts from the applied text
    press(app, "xyz", ESC)
    assert app.mode == "normal"
    assert app.search_buffer == ""
    assert app.world.search_text == "EIN"


@pytest.mark.parametrize("confirm", [10, 13, curses.KEY_ENTER])
def test_committing_a_search_focuses_the_matching_aircraft(make_app, confirm):
    app = planes_app(make_app)
    press(app, "/", "dlh4", confirm)
    assert app.mode == "normal"
    assert app.world.search_text == "dlh4"
    assert layer(app, "flights").search_text == "dlh4"
    assert app.world.zoom >= 8
    assert app.world.center_lon == pytest.approx(8.57)
    assert app.world.center_lat == pytest.approx(50.03)


def test_search_prefers_an_exact_callsign_over_a_prefix_match(make_app):
    app = make_app(tab="planes")
    layer(app, "flights").flights = [
        {"icao24": "aaa111", "callsign": "EIN440", "longitude": 10.0, "latitude": 10.0},
        {"icao24": "bbb222", "callsign": "EIN44", "longitude": -20.0, "latitude": -20.0},
    ]
    press(app, "/", "ein44", ENTER)
    assert (app.world.center_lon, app.world.center_lat) == pytest.approx((-20.0, -20.0))


def test_search_without_a_match_keeps_the_view(make_app):
    app = planes_app(make_app)
    view = (app.world.center_lon, app.world.center_lat, app.world.zoom)
    press(app, "/", "nothing", ENTER)
    assert app.world.search_text == "nothing"
    assert (app.world.center_lon, app.world.center_lat, app.world.zoom) == view


@pytest.mark.xfail(
    strict=True,
    reason="BUG: commit_input_mode stores a whitespace-only search unstripped, leaving an "
    "invisible search applied (counter flips from TOTAL PLANES to SHOWN)",
)
def test_a_blank_search_commits_as_no_search(make_app):
    app = planes_app(make_app)
    press(app, "/", "   ", ENTER)
    assert app.world.search_text == ""
    assert app._plane_count_line() == "TOTAL PLANES: 4"


def test_committing_a_search_clears_the_selection(make_app):
    app = planes_app(make_app)
    flights = layer(app, "flights")
    flights.select(SAMPLE_FLIGHTS[0])
    press(app, "/", "uae", ENTER)
    assert flights.selected is None


def test_filter_prompt_commits_the_top_suggestion(make_app):
    app = planes_app(make_app)
    press(app, "o", "ger", ENTER)
    flights = layer(app, "flights")
    assert app.mode == "normal"
    assert app.filter_selected == {"Germany"}
    assert flights.filter_countries == {"Germany"}
    assert app.world.filter_value == "Germany"
    assert [flight["callsign"] for flight in flights.visible_flights()] == ["DLH400"]


def test_filter_prompt_without_a_match_stays_open(make_app):
    app = planes_app(make_app)
    press(app, "o", "zzz", ENTER)
    assert app.mode == "filter"
    assert app.world.filter_value is None


def test_filter_prompt_with_nothing_typed_just_closes(make_app):
    app = planes_app(make_app)
    press(app, "o", "  ", ENTER)
    assert app.mode == "normal"
    assert app.filter_buffer == ""
    assert app.world.filter_value is None


def test_filter_prompt_starts_from_the_applied_countries(make_app):
    app = planes_app(make_app)
    layer(app, "flights").apply_filter({"Ireland", "Germany"})
    press(app, "o")
    assert app.filter_selected == {"Ireland", "Germany"}
    assert app.filter_buffer == ""


def test_typing_in_the_filter_prompt_resets_its_scroll(make_app):
    app = planes_app(make_app)
    press(app, "o")
    app.filter_scroll = 3
    press(app, "g")
    assert app.filter_scroll == 0
    app.filter_scroll = 2
    press(app, BACKSPACE)
    assert app.filter_scroll == 0
    assert app.filter_buffer == ""


def test_filter_suggestions_put_prefix_matches_first(make_app):
    app = planes_app(make_app)
    app.filter_buffer = "i"
    assert app.filter_suggestions() == ["Ireland", "United Arab Emirates", "United States"]
    app.filter_buffer = ""
    assert app.filter_suggestions() == ["Germany", "Ireland", "United Arab Emirates", "United States"]
    assert app.current_filter_suggestion() is None
    app.filter_buffer = "UNI"
    assert app.current_filter_suggestion() == "United Arab Emirates"


def test_esc_in_normal_mode_clears_search_filter_and_info(make_app):
    app = planes_app(make_app)
    press(app, "o", "ire", ENTER)
    app.world.set_search("EIN")
    app.info_visible = True
    press(app, ESC)
    flights = layer(app, "flights")
    assert app.world.search_text == ""
    assert app.world.filter_value is None
    assert flights.filter_countries == set()
    assert flights.search_text == ""
    assert app.info_visible is False


@pytest.mark.parametrize(
    "setup, expected",
    [
        (lambda flights: None, "TOTAL PLANES: 4"),
        (lambda flights: flights.apply_filter("Ireland"), "SHOWN: 1 PLANE"),
        (lambda flights: flights.apply_search("ua"), "SHOWN: 2 PLANES"),
        (lambda flights: flights.apply_search("nope"), "SHOWN: 0 PLANES"),
    ],
)
def test_plane_count_line(make_app, setup, expected):
    app = planes_app(make_app)
    setup(layer(app, "flights"))
    assert app._plane_count_line() == expected


def test_plane_count_line_without_a_flights_layer(make_app):
    app = make_app()
    app.world.layers = [item for item in app.world.layers if item.name != "flights"]
    assert app._plane_count_line() == "TOTAL PLANES: 0"


def test_search_box_lines_follow_the_mode(make_app):
    app = planes_app(make_app)
    assert app.search_box_lines("callsign / ICAO")[0][0] == "/ or click to search"
    assert app.search_box_lines("/ search  ·  O filter")[1][0] == "callsign / ICAO"
    press(app, "/", "ua")
    assert app.search_box_lines(None)[0][0] == "ua█"
    assert app.search_box_lines(None)[2][0] == "Enter apply  Esc cancel"
    press(app, ENTER)
    assert app.search_box_lines(None)[0][0] == "ua"
    assert app.search_box_lines(None)[2][0] == "/ edit  Esc clear"


def test_filter_box_lines_follow_the_mode(make_app):
    app = planes_app(make_app)
    idle = app.filter_box_lines("country name", 6)
    assert [text for text, _ in idle] == ["TOTAL PLANES: 4", "O or click to filter", "country name"]
    press(app, "o", "united")
    app.filter_selected = {"United States"}
    editing = [text for text, _ in app.filter_box_lines(None, 6)]
    assert editing == [
        "united█",
        "TOTAL PLANES: 4",
        "[ ] United Arab Emirates",
        "[x] United States",
    ]
    press(app, "zz")
    assert [text for text, _ in app.filter_box_lines(None, 6)][-1] == "no match"
    press(app, ESC)
    app.filter_selected = {"Ireland"}
    app._apply_country_filter()
    applied = [text for text, _ in app.filter_box_lines(None, 6)]
    assert applied == ["Ireland", "SHOWN: 1 PLANE", "O edit  Esc clear"]


def test_filter_box_lines_respect_the_room_and_scroll(make_app):
    app = planes_app(make_app)
    press(app, "o")
    app.filter_scroll = 1
    lines = [text for text, _ in app.filter_box_lines(None, 3)]
    assert lines == ["█", "TOTAL PLANES: 4", "[ ] Ireland", "[ ] United Arab Emirates"]


# ---- normal-mode commands ---------------------------------------------------------


@pytest.mark.parametrize("key", ["q", "Q"])
def test_q_quits(make_app, key):
    app = make_app(tab="weather")
    press(app, key)
    assert app.running is False


def test_ctrl_l_forces_a_full_repaint(make_app):
    app = make_app()
    calls = []
    app.stdscr.clearok = calls.append
    app.world.dirty = False
    press(app, 12)
    assert calls == [True]
    assert app.world.dirty is True


@pytest.mark.parametrize("tab", ["time", "weather", "planes"])
def test_h_toggles_the_panel_where_there_is_one(make_app, tab):
    app = make_app(tab=tab)
    press(app, "h")
    assert app.panel_visible is False
    press(app, "H")
    assert app.panel_visible is True


def test_h_does_nothing_on_the_map_tab(make_app):
    app = make_app()
    press(app, "h")
    assert app.panel_visible is True


def test_question_mark_reopens_the_welcome_modal(make_app):
    app = make_app(tab="planes")
    press(app, "?")
    assert app.welcome_visible is True


@pytest.mark.parametrize(
    "key, call",
    [
        ("w", ("pan", (0, 1))),
        ("W", ("pan", (0, 1))),
        (curses.KEY_UP, ("pan", (0, 1))),
        ("s", ("pan", (0, -1))),
        (curses.KEY_DOWN, ("pan", (0, -1))),
        ("a", ("pan", (-1, 0))),
        (curses.KEY_LEFT, ("pan", (-1, 0))),
        ("d", ("pan", (1, 0))),
        (curses.KEY_RIGHT, ("pan", (1, 0))),
        ("+", ("zoom_in", ())),
        ("=", ("zoom_in", ())),
        ("-", ("zoom_out", ())),
        ("_", ("zoom_out", ())),
        ("0", ("reset_view", ())),
        ("<", ("cycle_selection", (-1,))),
        (">", ("cycle_selection", (1,))),
    ],
)
@pytest.mark.parametrize("tab", ["map", "planes"])
def test_map_keys(make_app, tab, key, call):
    app = make_app(tab=tab)
    calls = []
    for name in ("pan", "zoom_in", "zoom_out", "reset_view", "cycle_selection"):
        setattr(app.world, name, lambda *args, name=name: calls.append((name, args)))
    press(app, key)
    assert calls == [call]


def test_zoom_and_reset_keys_move_the_real_view(make_app):
    from terrascope.core.world import ZOOM_MIN, ZOOM_STEP

    app = make_app()
    press(app, "+", "+")
    assert app.world.zoom == pytest.approx(ZOOM_MIN * ZOOM_STEP**2)
    start_lon = app.world.center_lon
    press(app, "d")
    assert app.world.center_lon > start_lon
    press(app, "0")
    assert app.world.zoom == ZOOM_MIN


def test_any_key_ends_a_drag_but_mouse_events_do_not(make_app):
    app = make_app()
    app.handle_mouse = lambda: None
    app._drag = [5, 5, 5, 5, False, 0.0]
    press(app, curses.KEY_MOUSE)
    assert app._drag is not None
    press(app, "w")
    assert app._drag is None


def test_normal_mode_routes_mouse_events_to_the_mouse_handler(make_app):
    app = make_app(tab="weather")
    calls = []
    app.handle_mouse = lambda: calls.append("mouse")
    press(app, curses.KEY_MOUSE)
    assert calls == ["mouse"]


# ---- slider keys ------------------------------------------------------------------


def test_bracket_keys_step_the_time_slider(make_app):
    app = make_app(tab="time")
    night = layer(app, "night")
    press(app, "]")
    assert night._time_offset_hours == 1
    press(app, "[", "[")
    assert night._time_offset_hours == -1


def test_slider_keys_stop_at_the_ends(make_app):
    app = make_app(tab="time")
    night = layer(app, "night")
    night.set_slider(48)
    press(app, "]")
    assert night._time_offset_hours == 24
    night.set_slider(0)
    press(app, "[")
    assert night._time_offset_hours == -24


def test_slider_keys_leave_a_disabled_slider_alone(make_app):
    app = make_app(tab="planes")
    flights = layer(app, "flights")
    assert flights.slider().disabled
    press(app, "[")
    assert flights.minutes_back == 0


def test_slider_keys_do_nothing_without_a_slider(make_app):
    app = make_app()
    assert app.world.slider_layer() is None
    before = hotkey_state(app)
    press(app, "]")
    assert hotkey_state(app) == before


# ---- layer hotkeys --------------------------------------------------------------------


@pytest.mark.parametrize(
    "tab, key, changed",
    [
        ("time", "n", "night"),
        ("time", "L", "look"),
        ("weather", "c", "cities"),
        ("weather", "E", "quakes"),
        ("planes", "p", "flights"),
        ("planes", "K", "airports"),
    ],
)
def test_layer_hotkeys_work_on_their_own_tab(make_app, tab, key, changed):
    app = make_app(tab=tab)
    before = hotkey_state(app)
    press(app, key)
    after = hotkey_state(app)
    assert {name for name in before if before[name] != after[name]} == {changed}


@pytest.mark.parametrize(
    "tab, keys",
    [
        ("map", "pkcel"),
        ("time", "pkce"),
        ("weather", "nlpk"),
        ("planes", "nlce"),
    ],
)
def test_layer_hotkeys_of_other_tabs_do_nothing(make_app, tab, keys):
    app = make_app(tab=tab, province_choice="no")
    before = hotkey_state(app)
    for key in keys:
        press(app, key)
        assert hotkey_state(app) == before, f"{key!r} acted on the {tab} tab"


def test_a_layer_hotkey_clears_the_selection(make_app):
    app = planes_app(make_app)
    flights = layer(app, "flights")
    app.world._selection_candidates = [(flights, SAMPLE_FLIGHTS[0])]
    app.world._activate_selection()
    app.info_visible = True
    press(app, "k")
    assert flights.selected is None
    assert app.info_visible is False


# ---- status keys: radar retry, dismiss, province retry ----------------------------------


def test_r_retries_a_failed_radar_on_the_weather_tab(make_app):
    app = make_app(tab="weather")
    layer(app, "weather").radar.failed = lambda: True
    app.map_statuses["RADAR"] = "RADAR FAILED"
    press(app, "r")
    assert make_app.radar_retries == ["retry"]
    assert "RADAR" not in app.map_statuses


def test_r_does_not_retry_a_healthy_radar(make_app):
    app = make_app(tab="weather")
    press(app, "r")
    assert make_app.radar_retries == []


def test_r_does_not_retry_the_radar_from_another_tab(make_app):
    app = make_app(tab="planes")
    layer(app, "weather").radar.failed = lambda: True
    press(app, "R")
    assert make_app.radar_retries == []


def test_x_dismisses_the_first_dismissable_status(make_app):
    app = make_app(tab="time")
    app._status_hits = [
        (4, 10, 20, "RADAR UNAVAILABLE", "retry-radar"),
        (5, 10, 13, "DOWNLOAD AIRPORTS FAILED", "dismiss"),
        (6, 10, 13, "SOMETHING ELSE FAILED", "dismiss"),
    ]
    press(app, "x")
    assert app.dismissed_statuses == {"DOWNLOAD AIRPORTS FAILED"}


def test_x_without_dismissable_statuses_does_nothing(make_app):
    app = make_app(tab="time")
    press(app, "X")
    assert app.dismissed_statuses == set()


def test_r_retries_failed_province_borders(make_app):
    app = make_app(tab="time", province_choice="yes")
    app.province_state = "failed"
    press(app, "r")
    assert app.province_state == "loading"
    assert [thread.name for thread in make_app.threads] == ["terrascope-province-borders"]


# ---- the province y/n prompt on the MAP tab ------------------------------------------------


def test_y_on_the_map_tab_accepts_province_borders(make_app):
    app = make_app()
    assert app.province_choice is None
    press(app, "y")
    assert app.province_choice == "yes"
    assert app.province_state == "loading"
    assert app.map_statuses["BORDERS"] == "LOADING STATE/PROVINCE BORDERS"
    assert (make_app.cache_dir / "province-borders-choice").read_text() == "yes"
    assert len(make_app.threads) == 1


def test_n_on_the_map_tab_declines_province_borders(make_app):
    app = make_app()
    press(app, "N")
    assert app.province_choice == "no"
    assert app.province_state == "idle"
    assert (make_app.cache_dir / "province-borders-choice").read_text() == "no"
    assert make_app.threads == []


def test_province_prompt_keys_only_apply_on_the_map_tab(make_app):
    app = make_app(tab="time")
    press(app, "y")
    assert app.province_choice is None
    press(app, "n")  # the night layer's own hotkey on this tab
    assert app.province_choice is None
    assert layer(app, "night").enabled is False


def test_province_prompt_keys_reopen_after_a_failed_download(make_app):
    app = make_app(province_choice="yes")
    app.province_state = "failed"
    press(app, "n")
    assert app.province_choice == "no"


def test_province_prompt_is_gone_once_answered(make_app):
    app = make_app(province_choice="no")
    press(app, "y")
    assert app.province_choice == "no"


def test_saved_province_choice_is_read_at_launch(make_app):
    (make_app.cache_dir / "province-borders-choice").write_text("yes")
    app = make_app()
    assert app.province_choice == "yes"
    assert app.province_state == "loading"
    assert len(make_app.threads) == 1
