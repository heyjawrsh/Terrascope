"""Mouse handling in TerrascopeApp: SGR / ncurses decoding, modals, tabs, drag vs click, wheel zoom."""

from __future__ import annotations

import curses

import pytest

from app_helpers import SAMPLE_FLIGHTS, TAB_INDEX, layer, make_app  # noqa: F401
from terrascope.core.app import mouse
from terrascope.core.app.shared import (
    DRAG_RESUME_SECONDS,
    MAP_EDGE_PAD_Y,
    SCROLL_DOWN,
    SCROLL_UP,
    TABS,
    _welcome_skipped,
    info_arrow_positions,
    panel_button_slot,
    tab_slots,
    timezone_choices,
)
from terrascope.core.view import calculate_viewport


def map_inner(app) -> tuple[int, int, int, int]:
    """(inner_left, inner_top, inner_width, inner_height) of the map box."""
    top, left, height, width = app.layout(*app.stdscr.getmaxyx()).map
    return left + 2, top + 1, width - 4, height - 2


def map_middle(app) -> tuple[int, int]:
    left, top, width, height = map_inner(app)
    return left + width // 2, top + height // 2


def spy(target, name: str) -> list:
    calls: list = []
    setattr(target, name, lambda *args: calls.append(args))
    return calls


def lon_lat_under(app, col: int, row: int) -> tuple[float, float]:
    world = app.world
    _, _, cols, rows = map_inner(app)
    lon_span, lat_span = calculate_viewport(world.zoom, world.aspect)
    return (
        world.center_lon - lon_span / 2 + (col + 0.5) / cols * lon_span,
        world.center_lat + lat_span / 2 - (row + 0.5) / rows * lat_span,
    )


# ---- ncurses KEY_MOUSE decoding ---------------------------------------------------------


@pytest.mark.parametrize(
    "state, kind",
    [
        (SCROLL_UP, "scroll_up"),
        pytest.param(
            SCROLL_DOWN,
            "scroll_down",
            marks=pytest.mark.skipif(not SCROLL_DOWN, reason="this curses build has no BUTTON5_PRESSED"),
        ),
        (curses.BUTTON1_RELEASED, "release"),
        (curses.REPORT_MOUSE_POSITION, "motion"),
        (curses.BUTTON1_PRESSED, "press"),
        (curses.BUTTON1_PRESSED | curses.REPORT_MOUSE_POSITION, "motion"),
    ],
)
def test_handle_mouse_decodes_ncurses_states(make_app, monkeypatch, state, kind):
    app = make_app()
    monkeypatch.setattr(mouse.curses, "getmouse", lambda: (0, 12, 7, 0, state))
    events = spy(app, "_mouse_event")
    app.handle_mouse()
    assert events == [(kind, 12, 7)]


def test_handle_mouse_ignores_other_buttons(make_app, monkeypatch):
    app = make_app()
    monkeypatch.setattr(mouse.curses, "getmouse", lambda: (0, 1, 1, 0, curses.BUTTON3_PRESSED))
    events = spy(app, "_mouse_event")
    app.handle_mouse()
    assert events == []


def test_handle_mouse_survives_getmouse_errors(make_app, monkeypatch):
    def broken():
        raise curses.error("no mouse event")

    app = make_app()
    monkeypatch.setattr(mouse.curses, "getmouse", broken)
    events = spy(app, "_mouse_event")
    app.handle_mouse()
    assert events == []


# ---- SGR (ESC [ < b ; x ; y M|m) decoding ----------------------------------------------


@pytest.fixture
def ungot(monkeypatch):
    """curses.ungetch needs a real terminal: record what would be pushed back."""
    pushed: list[int] = []
    monkeypatch.setattr(mouse.curses, "ungetch", pushed.append)
    return pushed


@pytest.mark.parametrize(
    "report, event",
    [
        ("[<0;10;5M", ("press", 9, 4)),
        ("[<0;10;5m", ("release", 9, 4)),
        ("[<32;11;6M", ("motion", 10, 5)),
        ("[<64;3;4M", ("scroll_up", 2, 3)),
        ("[<65;3;4M", ("scroll_down", 2, 3)),
        ("[<1;3;4M", None),  # middle button: unused
        ("[<2;3;4M", None),  # right button: unused
        ("[<0;x;4M", None),  # garbled: swallowed
        ("[<0;10;5", None),  # cut short: swallowed
    ],
)
def test_sgr_reports_are_decoded_and_consumed(make_app, ungot, report, event):
    app = make_app()
    app.stdscr.keys = [ord(char) for char in report]
    events = spy(app, "_mouse_event")
    assert app._try_sgr_mouse() is True
    assert events == ([event] if event else [])
    assert ungot == []
    assert app.stdscr.keys == []


def test_sgr_press_always_starts_a_new_grab(make_app, ungot):
    app = make_app()
    app._drag = [1, 1, 1, 1, True, 0.0]
    app.stdscr.keys = [ord(char) for char in "[<0;200;200M"]  # off the map
    app._try_sgr_mouse()
    assert app._drag is None


def test_plain_esc_is_not_a_mouse_report(make_app, ungot):
    app = make_app()
    assert app._try_sgr_mouse() is False
    assert ungot == []


def test_esc_followed_by_another_key_puts_it_back(make_app, ungot):
    app = make_app()
    app.stdscr.keys = [ord("x")]
    assert app._try_sgr_mouse() is False
    assert ungot == [ord("x")]


def test_esc_bracket_without_lt_puts_both_back(make_app, ungot):
    app = make_app()
    app.stdscr.keys = [ord("["), ord("A")]
    assert app._try_sgr_mouse() is False
    assert ungot == [ord("A"), ord("[")]


def test_esc_bracket_alone_puts_the_bracket_back(make_app, ungot):
    app = make_app()
    app.stdscr.keys = [ord("[")]
    assert app._try_sgr_mouse() is False
    assert ungot == [ord("[")]


# ---- the welcome modal ---------------------------------------------------------------------


@pytest.fixture
def welcome_app(make_app):
    app = make_app(welcome=True)
    app.draw()  # lays out the modal's clickable parts
    assert app._welcome_rect is not None
    return app


def test_click_outside_the_welcome_modal_closes_it(welcome_app):
    top, left, height, width = welcome_app._welcome_rect
    welcome_app._mouse_event("press", left - 1, top)
    assert welcome_app.welcome_visible is False


def test_click_on_blank_modal_space_keeps_it_open(welcome_app):
    top, left, height, width = welcome_app._welcome_rect
    welcome_app._mouse_event("press", left + 1, top + 1)
    assert welcome_app.welcome_visible is True


def test_welcome_ignores_everything_but_presses(welcome_app):
    x, y = map_middle(welcome_app)
    for kind in ("release", "motion", "scroll_up", "scroll_down"):
        welcome_app._mouse_event(kind, x, y)
    assert welcome_app.welcome_visible is True
    assert welcome_app.world.zoom == 1


def test_clicking_the_github_link_opens_a_browser(welcome_app, monkeypatch):
    opened = []
    monkeypatch.setattr(mouse.webbrowser, "open", opened.append)
    link = welcome_app._welcome_link
    welcome_app._mouse_event("press", link.left, link.row)
    assert opened == [mouse.GITHUB_URL]
    assert welcome_app.welcome_visible is True


def test_a_browser_failure_is_swallowed(welcome_app, monkeypatch):
    def broken(url):
        raise RuntimeError("no browser")

    monkeypatch.setattr(mouse.webbrowser, "open", broken)
    link = welcome_app._welcome_link
    welcome_app._mouse_event("press", link.left + link.width - 1, link.row)
    assert welcome_app.welcome_visible is True


def test_skip_button_closes_and_remembers(welcome_app, make_app):
    button = welcome_app._welcome_button
    welcome_app._mouse_event("press", button.left, button.row)
    assert welcome_app.welcome_visible is False
    assert (make_app.cache_dir / "welcome_skip").exists()
    assert _welcome_skipped() is True  # the next launch starts without it


@pytest.mark.parametrize("part, choice", [("_welcome_province_yes", "yes"), ("_welcome_province_no", "no")])
def test_province_buttons_in_the_welcome_modal(welcome_app, part, choice):
    target = getattr(welcome_app, part)
    welcome_app._mouse_event("press", target.left, target.row)
    assert welcome_app.province_choice == choice
    assert welcome_app.welcome_visible is False


def test_any_click_closes_a_welcome_modal_that_was_never_drawn(make_app):
    app = make_app(welcome=True)
    app._mouse_event("press", 50, 20)
    assert app.welcome_visible is False


# ---- the timezone dropdown -------------------------------------------------------------------


@pytest.fixture
def dropdown_app(make_app):
    app = make_app(tab="time")
    app.open_tz_dropdown()
    app.draw()
    assert app._dropdown_layout is not None
    return app


def test_clicking_a_dropdown_entry_picks_it(dropdown_app):
    layout = dropdown_app._dropdown_layout
    row = layout.option_rows[2]
    top, left, height, width = layout.rect
    dropdown_app._mouse_event("press", left + 3, row)
    assert dropdown_app.selected_timezone == timezone_choices("")[layout.first_index + 2][1]
    assert dropdown_app.tz_dropdown_open is False


def test_clicking_dropdown_padding_keeps_it_open(dropdown_app):
    top, left, height, width = dropdown_app._dropdown_layout.rect
    dropdown_app._mouse_event("press", left, top)
    assert dropdown_app.tz_dropdown_open is True


def test_clicking_outside_the_dropdown_closes_it_without_choosing(dropdown_app):
    x, y = map_middle(dropdown_app)
    dropdown_app._mouse_event("press", 0, y)
    assert dropdown_app.tz_dropdown_open is False
    assert dropdown_app.selected_timezone is None
    assert dropdown_app._drag is None  # the map underneath never saw the click


def test_wheel_moves_the_dropdown_highlight_not_the_map(dropdown_app):
    x, y = map_middle(dropdown_app)
    dropdown_app._mouse_event("scroll_down", x, y)
    dropdown_app._mouse_event("scroll_down", x, y)
    dropdown_app._mouse_event("scroll_up", x, y)
    assert dropdown_app.tz_index == 1
    assert dropdown_app.world.zoom == 1
    dropdown_app._mouse_event("release", x, y)
    dropdown_app._mouse_event("motion", x, y)
    assert dropdown_app.tz_dropdown_open is True


def test_dropdown_without_a_layout_closes_on_click(make_app):
    app = make_app(tab="time")
    app.open_tz_dropdown()
    app._mouse_event("press", 5, 5)
    assert app.tz_dropdown_open is False


# ---- tab bar and controls ----------------------------------------------------------------------


@pytest.mark.parametrize("index", range(len(TABS)))
def test_clicking_a_tab_selects_it(make_app, index):
    app = make_app(tab="planes" if index != 3 else "map")
    tabs_top, tabs_left, _, tabs_width = app.layout(40, 120).tabs
    _, start, text = tab_slots(TABS, tabs_left, tabs_width)[index]
    app._mouse_event("press", start + len(text) - 1, tabs_top)
    assert app.active_tab == index


@pytest.mark.xfail(
    strict=True,
    reason="BUG: select_tab keeps self.mode, so clicking a tab while the plane search / filter "
    "prompt is open leaves a hidden text mode capturing every key (issue #3 via the mouse)",
)
@pytest.mark.parametrize("key", ["/", "o"])
def test_clicking_a_tab_while_a_planes_prompt_is_open_releases_the_keyboard(make_app, key):
    app = make_app(tab="planes")
    app.handle_key(ord(key))
    assert app.mode != "normal"
    tabs_top, tabs_left, _, tabs_width = app.layout(40, 120).tabs
    _, start, _ = tab_slots(TABS, tabs_left, tabs_width)[0]
    app._mouse_event("press", start, tabs_top)
    assert app.active_tab == TAB_INDEX["map"]
    app.handle_key(ord("2"))
    assert app.active_tab == TAB_INDEX["time"]


def test_clicks_between_tab_bar_items_do_nothing(make_app):
    app = make_app(tab="time")
    app.draw()
    tabs_top, tabs_left, _, tabs_width = app.layout(40, 120).tabs
    slots = tab_slots(TABS, tabs_left, tabs_width)
    _, start, text = slots[-1]
    app._mouse_event("press", start + len(text) + 3, tabs_top)
    assert app.active_tab == TAB_INDEX["time"]
    assert app.tz_dropdown_open is False
    assert app.controls_expanded is False
    assert app._drag is None


def test_clicking_the_clock_opens_the_timezone_dropdown(make_app):
    app = make_app(tab="weather")
    app.draw()
    controls = app._controls_layout
    app._mouse_event("press", controls.datetime_x + 1, app.layout(40, 120).tabs[0])
    assert app.tz_dropdown_open is True


def test_controls_button_expands_and_collapses(make_app):
    app = make_app()
    app.draw()
    row = app.layout(40, 120).tabs[0]
    controls = app._controls_layout
    app._mouse_event("press", controls.button_x, row)
    assert app.controls_expanded is True
    app.draw()
    controls = app._controls_layout
    app._mouse_event("press", controls.button_x, row)
    assert app.controls_expanded is False


def test_expanded_controls_switch_panel_orientation(make_app):
    app = make_app(tab="time")
    app.controls_expanded = True
    app.draw()
    controls = app._controls_layout
    row = app.layout(40, 120).tabs[0]
    start = app.panel_orientation
    app._mouse_event("press", controls.orientation_x, row)
    assert app.panel_orientation != start
    app._mouse_event("press", controls.orientation_x, row)
    assert app.panel_orientation == start


@pytest.mark.parametrize("action", ["about", "quit"])
def test_expanded_controls_help_links(make_app, action):
    app = make_app(tab="time")
    app.controls_expanded = True
    app.draw()
    hits = {name: (x, width) for name, x, width in app._controls_layout.help_hits}
    assert action in hits
    app._mouse_event("press", hits[action][0], app.layout(40, 120).tabs[0])
    if action == "about":
        assert app.welcome_visible is True
    else:
        assert app.running is False


def test_panel_button_toggles_the_panel(make_app):
    app = make_app(tab="weather")
    row, x, text = panel_button_slot(app.layout(40, 120).map, app.panel_visible)
    app._mouse_event("press", x + len(text) - 1, row)
    assert app.panel_visible is False
    row, x, text = panel_button_slot(app.layout(40, 120).map, app.panel_visible)
    app._mouse_event("press", x, row)
    assert app.panel_visible is True


def test_map_tab_has_no_panel_button(make_app):
    app = make_app()
    row, x, text = panel_button_slot(app.layout(40, 120).map, app.panel_visible)
    app._mouse_event("press", x, row)
    assert app.panel_visible is True


# ---- drag to pan vs click to select ----------------------------------------------------------


def test_press_and_release_in_place_selects_under_the_pointer(make_app):
    app = make_app()
    left, top, _, _ = map_inner(app)
    selects = spy(app.world, "select_at")
    app._mouse_event("press", left + 10, top + 4)
    assert app._drag is not None
    app._mouse_event("release", left + 10, top + 4)
    assert selects == [(10, 4)]
    assert app._drag is None
    assert app.info_visible is False  # nothing was there


def test_a_click_on_a_marker_shows_the_info_box(make_app):
    app = make_app(tab="planes")
    flights = layer(app, "flights")

    def select_at(x, y):
        app.world._selection_candidates = [(flights, SAMPLE_FLIGHTS[0])]
        app.world._activate_selection()

    app.world.select_at = select_at
    x, y = map_middle(app)
    app._mouse_event("press", x, y)
    app._mouse_event("release", x, y)
    assert app.info_visible is True
    assert flights.selected == SAMPLE_FLIGHTS[0]


def test_dragging_pans_by_the_cells_moved_and_selects_nothing(make_app):
    app = make_app()
    left, top, width, height = map_inner(app)
    x, y = map_middle(app)
    pans = spy(app.world, "pan_cells")
    selects = spy(app.world, "select_at")
    app._mouse_event("press", x, y)
    app._mouse_event("motion", x + 3, y + 1)
    app._mouse_event("motion", x + 3, y + 1)  # no movement: no pan
    app._mouse_event("motion", x + 1, y - 2)
    app._mouse_event("release", x + 1, y - 2)
    assert pans == [(3, 1, width, height), (-2, -3, width, height)]
    assert selects == []


def test_real_drag_moves_the_map_the_way_the_pointer_goes(make_app):
    app = make_app()
    app.world.zoom_in()
    app.world.zoom_in()
    start = app.world.center_lon
    x, y = map_middle(app)
    app._mouse_event("press", x, y)
    app._mouse_event("motion", x + 5, y)
    assert app.world.center_lon < start  # grabbing the map and pulling it right shows the west


def test_motion_or_release_without_a_press_does_nothing(make_app):
    app = make_app()
    pans = spy(app.world, "pan_cells")
    selects = spy(app.world, "select_at")
    x, y = map_middle(app)
    app._mouse_event("motion", x, y)
    app._mouse_event("release", x, y)
    assert pans == [] and selects == []


def test_a_press_after_a_pause_regrabs_instead_of_jumping(make_app, monkeypatch):
    app = make_app()
    clock = [100.0]
    monkeypatch.setattr(mouse.time, "monotonic", lambda: clock[0])
    pans = spy(app.world, "pan_cells")
    x, y = map_middle(app)
    app._mouse_event("press", x, y)
    clock[0] += DRAG_RESUME_SECONDS + 0.1  # the release got lost
    app._mouse_event("press", x + 8, y + 3)
    assert pans == []
    assert app._drag[:2] == [x + 8, y + 3]
    clock[0] += 0.01
    app._mouse_event("press", x + 9, y + 3)  # a quick repeat counts as movement
    assert len(pans) == 1


def test_press_outside_the_map_starts_no_drag(make_app):
    app = make_app()
    app._mouse_event("press", 0, 39)
    assert app._drag is None


# ---- wheel zoom ---------------------------------------------------------------------------------


def test_wheel_zooms_around_the_pointer(make_app):
    app = make_app()
    left, top, width, height = map_inner(app)
    calls = spy(app.world, "zoom_at")
    app._mouse_event("scroll_up", left + 7, top + 5)
    app._mouse_event("scroll_down", left + 1, top + 2)
    assert calls == [(1, 7, 5, width, height), (-1, 1, 2, width, height)]


def test_wheel_zoom_keeps_the_spot_under_the_pointer_in_place(make_app):
    app = make_app()
    app.draw()  # sets the real aspect ratio
    left, top, _, _ = map_inner(app)
    x, y = map_middle(app)
    col, row = x - left + 6, y - top - 3
    before = lon_lat_under(app, col, row)
    app._mouse_event("scroll_up", left + col, top + row)
    assert app.world.zoom > 1
    assert lon_lat_under(app, col, row) == pytest.approx(before)


def test_wheel_outside_the_map_does_not_zoom(make_app):
    app = make_app()
    app._mouse_event("scroll_up", 0, 0)
    assert app.world.zoom == 1


# ---- panel boxes: search, filter, info, forecast ------------------------------------------------


def many_countries_app(make_app, count: int = 20):
    app = make_app(tab="planes")
    layer(app, "flights").flights = [
        {"icao24": f"{index:06x}", "callsign": f"T{index}", "origin_country": f"Country {index:02d}",
         "longitude": 0.0, "latitude": 0.0}
        for index in range(count)
    ]
    return app


def test_clicking_the_search_box_starts_a_search(make_app):
    app = make_app(tab="planes")
    top, left, height, width = app.layout(40, 120).rect_of("search")
    app._mouse_event("press", left + 2, top + 1)
    assert app.mode == "search"


def test_clicking_the_filter_box_opens_then_toggles_options(make_app):
    app = make_app(tab="planes")
    layer(app, "flights").flights = [dict(flight) for flight in SAMPLE_FLIGHTS]
    top, left, height, width = app.layout(40, 120).rect_of("filter")
    app._mouse_event("press", left + 2, top + 1)
    assert app.mode == "filter"
    app._mouse_event("press", left + 2, top + 3 + 1)  # second option: Ireland
    assert app.filter_selected == {"Ireland"}
    assert app.world.filter_value == "Ireland"
    app._mouse_event("press", left + 2, top + 3)  # first option: Germany
    assert app.world.filter_value == "Germany, Ireland"
    app._mouse_event("press", left + 2, top + 3 + 1)
    assert app.filter_selected == {"Germany"}
    app._mouse_event("press", left + 2, top + 1)  # not an option row
    assert app.filter_selected == {"Germany"}


def test_wheel_over_the_filter_box_scrolls_its_options(make_app):
    app = many_countries_app(make_app)
    top, left, height, width = app.layout(40, 120).rect_of("filter")
    room = max(1, height - 4)
    for _ in range(50):
        app._mouse_event("scroll_down", left + 1, top + 1)
    assert app.mode == "filter"
    assert app.filter_scroll == 20 - room
    app._mouse_event("scroll_up", left + 1, top + 1)
    assert app.filter_scroll == 20 - room - 1
    assert app.world.zoom == 1


def test_clicking_a_scrolled_filter_list_picks_the_visible_option(make_app):
    app = many_countries_app(make_app)
    top, left, height, width = app.layout(40, 120).rect_of("filter")
    app._mouse_event("scroll_down", left + 1, top + 1)
    app._mouse_event("scroll_down", left + 1, top + 1)
    app._mouse_event("press", left + 2, top + 3)
    assert app.filter_selected == {"Country 02"}


def test_info_box_arrows_cycle_the_selection(make_app):
    app = make_app(tab="planes")
    flights = layer(app, "flights")
    app.world._selection_candidates = [(flights, flight) for flight in SAMPLE_FLIGHTS[:3]]
    app.world._activate_selection()
    app.info_visible = True
    info_top, info_left, _, _ = app.layout(40, 120).info
    left_arrow, right_arrow = info_arrow_positions(
        info_left, app.world.selected_layer, app.world.selection_position
    )
    app._mouse_event("press", right_arrow, info_top)
    assert app.world.selection_position == (2, 3)
    app._mouse_event("press", left_arrow, info_top)
    app._mouse_event("press", left_arrow, info_top)
    assert app.world.selection_position == (3, 3)
    app._mouse_event("press", info_left + 2, info_top + 2)  # inside, not an arrow
    assert app.world.selection_position == (3, 3)
    assert app._drag is None


def test_wheel_over_the_vertical_forecast_scrolls_it(make_app):
    app = make_app(tab="weather")
    weather = layer(app, "weather")
    weather.forecast_line_count = lambda: 40
    top, left, height, width = app.layout(40, 120).rect_of("rich")
    app._mouse_event("scroll_down", left + 1, top + 1)
    assert app.forecast_scroll == 3
    for _ in range(30):
        app._mouse_event("scroll_down", left + 1, top + 1)
    assert app.forecast_scroll == 40 - (height - 3)
    app._mouse_event("scroll_up", left + 1, top + 1)
    assert app.forecast_scroll == 40 - (height - 3) - 3
    assert app.world.zoom == 1


# ---- legend, status and prompt hit areas (set by draw()) -----------------------------------------


def test_clicking_a_legend_toggle_presses_it_on_the_layers(make_app):
    app = make_app(tab="time")
    night = layer(app, "night")
    look = night.look
    app._legend_hits = [(10, 50, 60, "look")]
    app._drag = [1, 1, 1, 1, False, 0.0]
    app._mouse_event("press", 55, 10)
    assert night.look != look
    assert app._drag is None


def test_clicking_the_province_legend_toggle_shows_borders_on_the_map_tab(make_app):
    app = make_app()
    app._legend_hits = [(10, 50, 60, "provinces")]
    app._mouse_event("press", 50, 10)
    assert app.provinces_visible is True
    assert app.world.show_provinces is True
    assert layer(app, "basemap").provinces_visible is True
    app._mouse_event("press", 59, 10)
    assert app.provinces_visible is False


@pytest.mark.parametrize(
    "action, check",
    [
        ("dismiss", lambda app, built: app.dismissed_statuses == {"MSG FAILED"}),
        ("retry-provinces", lambda app, built: app.province_choice == "yes" and len(built.threads) == 1),
        ("retry-radar", lambda app, built: built.radar_retries == ["retry"]),
        ("toggle-radar-status", lambda app, built: app.radar_status_expanded is False),
    ],
)
def test_status_line_buttons(make_app, action, check):
    app = make_app(tab="weather")
    app._status_hits = [(8, 20, 23, "MSG FAILED", action)]
    app._mouse_event("press", 22, 8)
    assert check(app, make_app)
    assert app._drag is None


def test_dismissing_a_province_failure_clears_its_error(make_app):
    app = make_app()
    app.province_error = "boom"
    app._status_hits = [(8, 20, 23, "PROVINCE BORDERS FAILED: boom", "dismiss")]
    app._mouse_event("press", 20, 8)
    assert app.province_error == ""


@pytest.mark.parametrize("enabled, choice", [(True, "yes"), (False, "no")])
def test_province_prompt_buttons_on_the_map(make_app, enabled, choice):
    app = make_app()
    app._province_prompt_hits = [(6, 30, 37, enabled)]
    app._mouse_event("press", 31, 6)
    assert app.province_choice == choice
    assert app._drag is None


# ---- the time slider ---------------------------------------------------------------------------


def slider_row_and_track(app):
    left, top, width, height = map_inner(app)
    app.draw()  # a frame is needed to place the track
    _, _, first, length = app._slider_track(width)
    return top + height - 1 - MAP_EDGE_PAD_Y, left + first, length


def test_dragging_the_time_slider(make_app):
    app = make_app(tab="time")
    night = layer(app, "night")
    row, start, length = slider_row_and_track(app)
    app._mouse_event("press", start, row)
    assert app._slider_drag is True
    assert night._time_offset_hours == -24
    app._mouse_event("motion", start + length - 1, row - 5)  # rows don't matter while dragging
    assert night._time_offset_hours == 24
    app._mouse_event("release", start, row)
    assert app._slider_drag is False
    assert night._time_offset_hours == 24
    assert app._drag is None


def test_a_press_elsewhere_ends_a_slider_drag(make_app):
    app = make_app(tab="time")
    row, start, length = slider_row_and_track(app)
    app._mouse_event("press", start + length // 2, row)
    x, y = map_middle(app)
    app._mouse_event("press", x, y)
    assert app._slider_drag is False
    assert app._drag is not None


def test_a_press_off_the_track_on_the_slider_row_grabs_the_map(make_app):
    app = make_app(tab="time")
    row, start, length = slider_row_and_track(app)
    left, _, _, _ = map_inner(app)
    app._mouse_event("press", left + 1, row)
    assert app._slider_drag is False
    assert app._drag is not None
