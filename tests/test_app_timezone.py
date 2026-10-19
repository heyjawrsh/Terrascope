"""The timezone dropdown: opening, typing to search, keyboard navigation, choosing and closing."""

from __future__ import annotations

import curses

import pytest

from app_helpers import make_app, press  # noqa: F401
from terrascope.core.app import rendering
from terrascope.core.app.shared import timezone_choices

ESC = 27
ENTER = 10
BACKSPACE = 127


@pytest.fixture
def tz_app(make_app):
    app = make_app(tab="time")
    press(app, "z")
    assert app.tz_dropdown_open is True
    return app


def test_opening_starts_with_an_empty_search_on_the_first_entry(tz_app):
    assert tz_app.tz_query == ""
    assert tz_app.tz_index == 0
    assert tz_app.world.dirty is True


def test_opening_highlights_the_zone_in_use(make_app):
    choices = timezone_choices("")
    app = make_app(tab="time")
    app.selected_timezone = choices[5][1]
    app.tz_query = "leftover"
    app.open_tz_dropdown()
    assert app.tz_index == 5
    assert app.tz_query == ""


def test_typing_searches_and_resets_the_highlight(tz_app):
    press(tz_app, curses.KEY_DOWN, curses.KEY_DOWN)
    press(tz_app, "hel")
    assert tz_app.tz_query == "hel"
    assert tz_app.tz_index == 0
    assert "Europe/Helsinki" in [name for _, name in timezone_choices("hel")]


def test_backspace_edits_the_search(tz_app):
    press(tz_app, "tokx", BACKSPACE)
    assert tz_app.tz_query == "tok"
    press(tz_app, curses.KEY_DOWN, curses.KEY_BACKSPACE)
    assert tz_app.tz_query == "to"
    assert tz_app.tz_index == 0
    press(tz_app, 8, 8, 8)
    assert tz_app.tz_query == ""


@pytest.mark.parametrize(
    "keys, index",
    [
        ([curses.KEY_DOWN], 1),
        ([curses.KEY_DOWN, curses.KEY_DOWN, curses.KEY_UP], 1),
        ([curses.KEY_UP], 0),
        ([curses.KEY_NPAGE], 8),
        ([curses.KEY_NPAGE, curses.KEY_PPAGE, curses.KEY_PPAGE], 0),
    ],
)
def test_arrow_and_page_keys_move_the_highlight(tz_app, keys, index):
    press(tz_app, *keys)
    assert tz_app.tz_index == index


def test_highlight_stops_at_the_end_of_a_filtered_list(tz_app):
    press(tz_app, "helsinki")
    total = len(timezone_choices("helsinki"))
    press(tz_app, *[curses.KEY_NPAGE] * 5)
    assert tz_app.tz_index == total - 1


def test_ignored_keys_change_nothing(tz_app):
    press(tz_app, curses.KEY_LEFT, 1, 300)
    assert tz_app.tz_query == ""
    assert tz_app.tz_index == 0
    assert tz_app.tz_dropdown_open is True


@pytest.mark.parametrize("confirm", [10, 13, curses.KEY_ENTER])
def test_enter_picks_the_highlighted_zone_and_closes(tz_app, confirm):
    press(tz_app, "tokyo")
    expected = timezone_choices("tokyo")[0][1]
    press(tz_app, confirm)
    assert tz_app.selected_timezone == expected == "Asia/Tokyo"
    assert tz_app.tz_dropdown_open is False
    assert tz_app.tz_query == ""


def test_enter_picks_the_moved_highlight(tz_app):
    press(tz_app, curses.KEY_DOWN, curses.KEY_DOWN, ENTER)
    assert tz_app.selected_timezone == timezone_choices("")[2][1]


def test_enter_with_no_match_keeps_the_dropdown_open(tz_app):
    press(tz_app, "zzzzqqq")
    assert timezone_choices("zzzzqqq") == []
    press(tz_app, ENTER)
    assert tz_app.tz_dropdown_open is True
    assert tz_app.selected_timezone is None


def test_esc_closes_without_changing_the_zone(tz_app):
    tz_app.selected_timezone = "Europe/Paris"
    press(tz_app, "tokyo", curses.KEY_DOWN, ESC)
    assert tz_app.tz_dropdown_open is False
    assert tz_app.selected_timezone == "Europe/Paris"
    assert tz_app.tz_query == ""


def test_choosing_an_out_of_range_index_clamps_to_the_list(tz_app):
    press(tz_app, "tokyo")
    tz_app._tz_choose(10_000)
    assert tz_app.selected_timezone == timezone_choices("tokyo")[-1][1]


def test_keys_go_back_to_the_map_once_closed(tz_app):
    press(tz_app, ESC, "3")
    assert tz_app.active_tab == 2


def test_local_time_entry_resets_to_the_system_zone(tz_app):
    tz_app.selected_timezone = "Asia/Tokyo"
    local = next(index for index, (_, name) in enumerate(timezone_choices("")) if name is None)
    tz_app._tz_choose(local)
    assert tz_app.selected_timezone is None


# ---- drawing --------------------------------------------------------------------------------


@pytest.fixture
def fixed_clock(monkeypatch):
    seen: list = []

    def clock(zone):
        seen.append(zone)
        return "2026-01-02 03:04:05 TEST"

    monkeypatch.setattr(rendering, "current_time_text", clock)
    return seen


def test_time_tab_clock_shows_the_dropdown_key(make_app, fixed_clock):
    app = make_app(tab="time")
    app.draw()
    assert "Z TZ 2026-01-02 03:04:05 TEST" in app.stdscr.text()
    other = make_app(tab="weather")
    other.draw()
    assert "TZ 2026" not in other.stdscr.text()
    assert "2026-01-02 03:04:05 TEST" in other.stdscr.text()


def test_clock_uses_the_chosen_zone(make_app, fixed_clock):
    app = make_app(tab="time")
    app.selected_timezone = "Asia/Tokyo"
    app.draw()
    assert fixed_clock == ["Asia/Tokyo"]


def test_open_dropdown_is_drawn_with_the_search_and_entries(tz_app, fixed_clock):
    tz_app.draw()
    text = tz_app.stdscr.text()
    assert "TIMEZONE" in text
    assert "> search all timezones" in text
    assert timezone_choices("")[0][0] in text
    press(tz_app, "tokyo")
    tz_app.draw()
    text = tz_app.stdscr.text()
    assert "> tokyo█" in text
    assert "Tokyo" in text


def test_dropdown_closes_when_its_clock_does_not_fit(make_app):
    app = make_app(20, 60, tab="time")
    app.open_tz_dropdown()
    app.draw()
    assert app._controls_layout is None
    assert app.tz_dropdown_open is False
    assert app._dropdown_layout is None
