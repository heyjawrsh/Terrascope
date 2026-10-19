"""Screen layout, tab / controls bar, timezone picker and box helpers in core.ui.layout."""

from __future__ import annotations

import curses
import importlib
import itertools
import re
from datetime import datetime, timezone

import pytest

from conftest import FakeWindow
from terrascope.core.config import CFG
from terrascope.core.layer import Layer, PanelBox
from terrascope.core.tabs import TABS

layout = importlib.import_module("terrascope.core.ui.layout")
constants = importlib.import_module("terrascope.core.ui.constants")

MIN_HEIGHT = CFG["app"]["min_terminal_height"]
MIN_WIDTH = CFG["app"]["min_terminal_width"]
MARGIN_X, MARGIN_Y = constants.OUTER_MARGIN

SCREEN_SIZES = [
    (MIN_HEIGHT, MIN_WIDTH),
    (MIN_HEIGHT + 1, MIN_WIDTH + 1),
    (24, 80),
    (27, 73),
    (30, 100),
    (40, 120),
    (50, 200),
    (80, 300),
]

# The real bottom-row layouts used by the layers (see default.yaml panel_ratios).
PANEL_SPECS = {
    "default": None,
    "flights": (PanelBox("info", 0.5), PanelBox("search", 0.25, "SEARCH"), PanelBox("filter", 0.25, "FILTER")),
    "weather": (PanelBox("info", 0.22), PanelBox("rich", 0.78, "FORECAST")),
    "daynight": (PanelBox("live", 0.28, "CLOCK"), PanelBox("live", 0.40, "SUN"), PanelBox("live", 0.32, "MOON")),
    "search": (PanelBox("info", 0.43), PanelBox("search", 0.57, "SEARCH")),
}


def _inside(rect, height, width) -> bool:
    top, left, h, w = rect
    return top >= 0 and left >= 0 and h >= 1 and w >= 1 and top + h <= height and left + w <= width


def _overlap(a, b) -> bool:
    at, al, ah, aw = a
    bt, bl, bh, bw = b
    return at < bt + bh and bt < at + ah and al < bl + bw and bl < al + aw


def _row_text(window, y, x, length) -> str:
    return "".join(window.cells[y][x : x + length])


# ---- safe_addstr / concise_error_message / draw_box ---------------------------


def test_safe_addstr_never_writes_the_last_column():
    window = FakeWindow(3, 10)
    layout.safe_addstr(window, 1, 5, "abcdefgh")
    assert window.lines()[1] == "     abcd "


def test_safe_addstr_writes_nothing_at_or_past_the_right_edge():
    window = FakeWindow(3, 10)
    layout.safe_addstr(window, 1, 9, "x")
    layout.safe_addstr(window, 1, 20, "y")
    assert window.text().strip() == ""


def test_safe_addstr_swallows_curses_errors():
    class RaisingWindow(FakeWindow):
        def addnstr(self, *args, **kwargs):
            raise curses.error("bottom-right corner")

    layout.safe_addstr(RaisingWindow(3, 10), 2, 0, "text")  # does not raise


def test_safe_addstr_passes_attributes_through():
    window = FakeWindow(2, 10)
    layout.safe_addstr(window, 0, 0, "ab", curses.A_BOLD)
    assert window.attrs[0][:3] == [curses.A_BOLD, curses.A_BOLD, 0]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("<urlopen error [Errno 8] nodename nor servname provided, or not known>", "network/DNS unavailable"),
        ("[Errno -2] Name or service not known", "network/DNS unavailable"),
        ("Temporary failure in name resolution", "network/DNS unavailable"),
        ("getaddrinfo failed", "network/DNS unavailable"),
        ("<urlopen error timed out>", "request timed out"),
        ("Read TIMEOUT after 10s", "request timed out"),
        ("HTTP Error 404: Not Found", "HTTP 404"),
        ("http error 503: Service Unavailable", "HTTP 503"),
        ("Could not download ne_10m_airports.geojson", "download failed"),
        ("  something   odd\n happened ", "something odd happened"),
        ("", ""),
    ],
)
def test_concise_error_message_shortens_common_transport_errors(error, expected):
    assert layout.concise_error_message(error) == expected


def test_concise_error_message_accepts_exception_objects():
    assert layout.concise_error_message(TimeoutError("timed out")) == "request timed out"


@pytest.mark.parametrize("limit", [2, 5, 10, 34, 42])
def test_concise_error_message_truncates_long_text_with_an_ellipsis(limit):
    text = layout.concise_error_message("x" * 200, limit)
    assert len(text) <= limit
    assert text.endswith("…")


def test_concise_error_message_keeps_text_exactly_at_the_limit():
    assert layout.concise_error_message("a" * 42) == "a" * 42


def test_draw_box_draws_corners_edges_and_title():
    window = FakeWindow(6, 20)
    layout.draw_box(window, 1, 2, 4, 12, "HI")
    lines = window.lines()
    assert lines[1][2:14] == "┌─ HI ─────┐"
    assert lines[2][2] == "│" and lines[2][13] == "│"
    assert lines[4][2:14] == "└" + "─" * 10 + "┘"
    assert lines[1][4:8] == " HI "


def test_draw_box_drops_a_title_that_does_not_fit():
    window = FakeWindow(4, 20)
    layout.draw_box(window, 0, 0, 3, 8, "TOO LONG")
    assert window.lines()[0][:8] == "┌──────┐"


@pytest.mark.parametrize(("height", "width"), [(1, 10), (5, 1), (0, 0)])
def test_draw_box_too_small_draws_nothing(height, width):
    window = FakeWindow(6, 20)
    layout.draw_box(window, 0, 0, height, width, "X")
    assert window.text().strip() == ""


# ---- compute_layout -------------------------------------------------------------


@pytest.mark.parametrize(("height", "width"), SCREEN_SIZES)
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("spec_name", list(PANEL_SPECS))
def test_layout_rects_stay_on_screen_and_never_overlap(height, width, orientation, spec_name):
    result = layout.compute_layout(
        height, width, True, panel_layout=PANEL_SPECS[spec_name], panel_orientation=orientation
    )
    rects = [result.map, result.tabs] + [rect for _, rect in result.boxes]
    for rect in rects:
        assert _inside(rect, height, width), rect
    for a, b in itertools.combinations(rects, 2):
        assert not _overlap(a, b), (a, b)


@pytest.mark.parametrize(("height", "width"), SCREEN_SIZES)
def test_horizontal_panel_boxes_share_one_row_and_fill_its_width(height, width):
    spec = PANEL_SPECS["flights"]
    result = layout.compute_layout(height, width, True, panel_layout=spec)
    assert [box for box, _ in result.boxes] == list(spec)
    tops = {rect[0] for _, rect in result.boxes}
    heights = {rect[2] for _, rect in result.boxes}
    assert len(tops) == 1 and len(heights) == 1
    map_top, map_left, map_height, map_width = result.map
    assert tops.pop() == map_top + map_height  # directly under the map
    first, last = result.boxes[0][1], result.boxes[-1][1]
    assert first[1] == map_left
    assert last[1] + last[3] == map_left + map_width  # last box takes the remainder
    for (_, a), (_, b) in zip(result.boxes, result.boxes[1:]):
        assert b[1] - (a[1] + a[3]) == constants.BOX_GAP


@pytest.mark.parametrize(("height", "width"), SCREEN_SIZES)
def test_vertical_panel_boxes_stack_and_fill_the_panel_height(height, width):
    spec = PANEL_SPECS["daynight"]
    result = layout.compute_layout(height, width, True, panel_layout=spec, panel_orientation="vertical")
    map_top, map_left, map_height, _ = result.map
    boxes = [rect for _, rect in result.boxes]
    assert boxes[0][0] == map_top
    for a, b in zip(boxes, boxes[1:]):
        assert b[0] == a[0] + a[2]
    assert boxes[-1][0] + boxes[-1][2] == map_top + map_height
    assert {rect[1] for rect in boxes} == {MARGIN_X}
    assert map_left == MARGIN_X + boxes[0][3] + constants.BOX_GAP


def test_vertical_panel_is_at_most_the_configured_width():
    result = layout.compute_layout(40, 200, True, panel_orientation="vertical")
    assert result.info[3] == constants.PANEL_WIDTH


@pytest.mark.parametrize(("height", "width"), SCREEN_SIZES)
def test_hidden_panel_gives_the_map_all_the_room(height, width):
    result = layout.compute_layout(height, width, False)
    assert result.boxes == ()
    assert result.info is None
    assert result.map == (
        MARGIN_Y + constants.LAYER_BAR_HEIGHT,
        MARGIN_X,
        height - 2 * MARGIN_Y - constants.LAYER_BAR_HEIGHT,
        width - 2 * MARGIN_X,
    )
    assert result.tabs == (MARGIN_Y, MARGIN_X, constants.LAYER_BAR_HEIGHT, width - 2 * MARGIN_X)


def test_horizontal_panel_is_dropped_when_the_terminal_is_too_short():
    # Too short for both the minimum map and the minimum panel height.
    height = 2 * MARGIN_Y + constants.LAYER_BAR_HEIGHT + constants.PANEL_MIN_MAP_HEIGHT + constants.PANEL_MIN_HEIGHT - 1
    result = layout.compute_layout(height, 100, True)
    assert result.boxes == ()
    assert result.map[2] == height - 2 * MARGIN_Y - constants.LAYER_BAR_HEIGHT
    taller = layout.compute_layout(height + 1, 100, True)
    assert taller.info is not None and taller.info[2] == constants.PANEL_MIN_HEIGHT


def test_horizontal_panel_height_is_capped_and_map_keeps_its_minimum():
    tall = layout.compute_layout(100, 120, True)
    assert tall.info[2] == constants.PANEL_HEIGHT
    short = layout.compute_layout(MIN_HEIGHT, 120, True)
    assert short.map[2] >= constants.PANEL_MIN_MAP_HEIGHT


def test_vertical_panel_falls_back_to_the_bottom_row_when_too_narrow():
    # box width - 28 < 16 columns: no room for a side panel.
    width = 2 * MARGIN_X + 28 + 15
    result = layout.compute_layout(40, width, True, panel_orientation="vertical")
    info = result.info
    assert info is not None
    assert info[0] == result.map[0] + result.map[2]  # under the map, not beside it
    assert info[3] == result.map[3]


def test_owner_panel_layout_is_used_when_no_explicit_layout_is_given():
    class Owner(Layer):
        panel_layout = PANEL_SPECS["weather"]

    result = layout.compute_layout(40, 120, True, owner=Owner())
    assert [box.kind for box, _ in result.boxes] == ["info", "rich"]
    assert result.rect_of("rich") is not None
    assert result.rect_of("search") is None


def test_owner_can_adapt_the_vertical_layout_to_the_room_it_gets():
    calls = []

    class Owner(Layer):
        panel_layout = PANEL_SPECS["weather"]

        def panel_layout_for(self, available_height, panel_width):
            calls.append((available_height, panel_width))
            return (PanelBox("info", 0.5), PanelBox("live", 0.5, "CLOCK"))

    result = layout.compute_layout(40, 120, True, owner=Owner(), panel_orientation="vertical")
    assert calls == [(40 - 2 * MARGIN_Y - constants.LAYER_BAR_HEIGHT, constants.PANEL_WIDTH)]
    assert [box.kind for box, _ in result.boxes] == ["info", "live"]

    # An explicit panel_layout wins over the owner's adaptive one.
    calls.clear()
    explicit = layout.compute_layout(
        40, 120, True, owner=Owner(), panel_layout=PANEL_SPECS["search"], panel_orientation="vertical"
    )
    assert calls == []
    assert [box.kind for box, _ in explicit.boxes] == ["info", "search"]


# ---- tab bar / controls bar -------------------------------------------------------


def test_tab_slots_number_the_tabs_and_keep_the_gap():
    slots = layout.tab_slots(TABS, 2, 200)
    assert [index for index, _, _ in slots] == list(range(len(TABS)))
    for (index, x, text), tab in zip(slots, TABS):
        assert text == f" {index + 1} {tab.label} "
    for (_, x1, t1), (_, x2, _) in zip(slots, slots[1:]):
        assert x2 == x1 + len(t1) + constants.LAYER_TAB_GAP
    assert slots[0][1] == 2


@pytest.mark.parametrize("width", [0, 5, 8, 12, 20, 30, 40, 200])
def test_tab_slots_only_include_tabs_that_fit(width):
    slots = layout.tab_slots(TABS, 3, width)
    for _, x, text in slots:
        assert 3 <= x and x + len(text) <= 3 + width
    if len(slots) < len(TABS):
        next_text = f" {len(slots) + 1} {TABS[len(slots)].label} "
        next_x = slots[-1][1] + len(slots[-1][2]) + constants.LAYER_TAB_GAP if slots else 3
        assert next_x + len(next_text) > 3 + width


def test_draw_tab_bar_lights_the_active_tab(colors):
    window = FakeWindow(3, 120)
    rect = (0, 2, 1, 116)
    layout.draw_tab_bar(window, rect, TABS, 1, colors, False, "2024-01-15 12:00:00 UTC")
    slots = layout.tab_slots(TABS, 2, 116)
    for index, x, text in slots:
        assert _row_text(window, 0, x, len(text)) == text
        attrs = window.attrs[0][x]
        if index == 1:
            assert attrs & curses.A_REVERSE and attrs & curses.A_BOLD
        else:
            assert attrs & curses.A_DIM and not attrs & curses.A_REVERSE


def test_draw_tab_bar_without_a_row_draws_nothing(colors):
    window = FakeWindow(3, 120)
    assert layout.draw_tab_bar(window, (0, 2, 0, 116), TABS, 0, colors, False, "12:00") is None
    assert window.text().strip() == ""


@pytest.mark.parametrize("width", [MIN_WIDTH, 72, 80, 100, 120, 160, 240])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_controls_bar_never_overlaps_the_tabs(colors, width, expanded, orientation):
    time_text = "2024-01-15 12:00:00 UTC"
    window = FakeWindow(3, width)
    rect = layout.compute_layout(MIN_HEIGHT, width, True).tabs
    top, left, _, box_width = rect
    controls = layout.draw_tab_bar(window, rect, TABS, 0, colors, expanded, time_text, orientation)
    slots = layout.tab_slots(TABS, left, box_width)
    tabs_end = slots[-1][1] + len(slots[-1][2])
    if controls is None:
        return
    assert controls.datetime_x + controls.datetime_width == left + box_width
    assert _row_text(window, top, controls.datetime_x, len(time_text)) == time_text
    assert controls.button_x >= tabs_end
    assert controls.button_x + controls.button_width < controls.datetime_x
    button = constants.CONTROLS_EXPANDED_BUTTON if expanded else constants.CONTROLS_COLLAPSED_BUTTON
    assert _row_text(window, top, controls.button_x, len(button)) == button
    if expanded:
        orientation_text = "LEFT" if orientation == "vertical" else "BOTTOM"
        assert controls.orientation_x >= tabs_end
        assert _row_text(window, top, controls.orientation_x, len(orientation_text)) == orientation_text
        for action, x, hit_width in controls.help_hits:
            assert tabs_end < x and x + hit_width < controls.orientation_x
        # the tab text itself is untouched by the hints
        for _, x, text in slots:
            assert _row_text(window, top, x, len(text)) == text
    else:
        assert controls.help_hits == ()


def test_controls_bar_slots_give_up_when_there_is_no_room():
    assert layout.controls_bar_slots(50, 60, "2024-01-15 12:00:00 UTC") is None
    assert layout.controls_bar_slots(0, 60, "12:00") is not None


def test_expanded_controls_need_room_for_the_orientation_label():
    time_text = "12:00"
    collapsed = layout.controls_bar_slots(0, 100, time_text)
    # Just enough room for the collapsed controls, not the extra label.
    tabs_end = collapsed.button_x
    assert layout.controls_bar_slots(tabs_end, 100, time_text, expanded=False) is not None
    assert layout.controls_bar_slots(tabs_end, 100, time_text, expanded=True) is None


def _help_width(level: int) -> int:
    return sum(len(text) for text, _, _ in layout._help_segments(level))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_help_bar_uses_the_fullest_wording_that_fits(level):
    room = _help_width(level)
    right = 10 + constants.HELP_MIN_GAP + room
    window = FakeWindow(1, right + 5)
    hits = layout.draw_help_bar(window, 0, 10, right)
    text = window.lines()[0]
    full_key = constants.HELP_ITEMS[0][0]
    short_key = constants.HELP_SHORT_KEYS[full_key]
    if level == 0:
        assert f"{full_key} PAN" in text
    elif level == 1:
        assert f"{short_key} PAN" in text and full_key not in text
    else:
        assert "PAN" not in text and short_key in text
    assert {action for action, _, _ in hits} == {"about", "quit"}
    for action, x, width in hits:
        assert text[x : x + width].strip()
    assert text.rstrip().endswith(text[right - 1]) and text[right:].strip() == ""


def test_help_bar_clickable_hits_cover_their_key_text():
    window = FakeWindow(1, 200)
    hits = dict((action, (x, width)) for action, x, width in layout.draw_help_bar(window, 0, 0, 190))
    text = window.lines()[0]
    x, width = hits["quit"]
    assert text[x : x + width] == f"{constants.QUIT_KEY} QUIT"
    x, width = hits["about"]
    assert text[x : x + width] == "? ABOUT"


def test_help_bar_draws_nothing_when_even_the_keys_do_not_fit():
    window = FakeWindow(1, 40)
    assert layout.draw_help_bar(window, 0, 10, 10 + _help_width(2)) == ()
    assert window.text().strip() == ""


# ---- timezones / clock ----------------------------------------------------------

FAKE_ZONES = {
    "UTC",  # no "/": never listed as an extra
    "Etc/GMT+1",
    "posix/Europe/Helsinki",
    "right/UTC",
    "Europe/Helsinki",  # already curated
    "America/Hermosillo",
    "Asia/Chelyabinsk",
    "America/Argentina/Buenos_Aires",
    "Africa/Abidjan",
}


@pytest.fixture
def fake_zones(monkeypatch):
    monkeypatch.setattr(layout, "available_timezones", lambda: set(FAKE_ZONES))


def test_empty_query_lists_curated_zones_first_then_the_rest_a_to_z(fake_zones):
    choices = layout.timezone_choices("")
    curated = list(constants.TIMEZONE_OPTIONS)
    assert choices[: len(curated)] == curated
    assert choices[len(curated) :] == [
        ("Africa/Abidjan", "Africa/Abidjan"),
        ("America/Argentina/Buenos Aires", "America/Argentina/Buenos_Aires"),
        ("America/Hermosillo", "America/Hermosillo"),
        ("Asia/Chelyabinsk", "Asia/Chelyabinsk"),
    ]


def test_whitespace_query_counts_as_empty(fake_zones):
    assert layout.timezone_choices("   ") == layout.timezone_choices("")


def test_word_prefix_matches_come_before_substring_matches(fake_zones):
    choices = layout.timezone_choices("hel")
    names = [name for _, name in choices]
    assert names.index("Europe/Helsinki") < names.index("Asia/Chelyabinsk")
    assert names[0] == "Europe/Helsinki"  # curated "Helsinki / Athens"
    assert "America/Hermosillo" not in names


def test_query_is_case_insensitive_and_matches_words_split_on_slashes_and_underscores(fake_zones):
    assert layout.timezone_choices("BUENOS") == [
        ("America/Argentina/Buenos Aires", "America/Argentina/Buenos_Aires")
    ]
    assert layout.timezone_choices("  aires ") == layout.timezone_choices("buenos")


def test_local_entry_is_found_by_its_label(fake_zones):
    assert ("Local", None) in layout.timezone_choices("loc")


def test_query_without_matches_lists_nothing(fake_zones):
    assert layout.timezone_choices("zzzz") == []


def test_missing_tz_database_falls_back_to_the_curated_list(monkeypatch):
    def broken():
        raise OSError("no tzdata")

    monkeypatch.setattr(layout, "available_timezones", broken)
    assert layout.timezone_choices("") == list(constants.TIMEZONE_OPTIONS)


def test_real_tz_database_keeps_curated_first_and_prefix_matches_first():
    everything = layout.timezone_choices("")
    curated = list(constants.TIMEZONE_OPTIONS)
    assert everything[: len(curated)] == curated
    names = [name for _, name in everything[len(curated) :]]
    assert names == sorted(names)
    assert not any(name.startswith(constants._IGNORED_ZONE_PREFIXES) for name in names)
    assert len(set(name for _, name in everything)) == len(everything)

    needle = "new"
    ranked = layout.timezone_choices(needle)

    def is_prefix(entry):
        label, name = entry
        text = f"{label} {name or ''}".lower()
        words = text.replace("/", " ").replace("_", " ").split()
        return text.startswith(needle) or any(word.startswith(needle) for word in words)

    flags = [is_prefix(entry) for entry in ranked]
    assert flags == sorted(flags, reverse=True)  # all prefix matches, then substrings
    assert ("New York", "America/New_York") == ranked[0]


class _FrozenDateTime(datetime):
    """datetime with now() pinned to 2024-01-15 12:00:00 UTC."""

    FROZEN = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.FROZEN.astimezone(tz) if tz is not None else cls.FROZEN.replace(tzinfo=None)


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(layout, "datetime", _FrozenDateTime)


@pytest.mark.parametrize(
    ("zone", "expected"),
    [
        ("UTC", "2024-01-15 12:00:00 UTC"),
        ("Asia/Tokyo", "2024-01-15 21:00:00 JST"),
        ("America/New_York", "2024-01-15 07:00:00 EST"),
        ("Asia/Kolkata", "2024-01-15 17:30:00 IST"),
        ("Australia/Sydney", "2024-01-15 23:00:00 AEDT"),
        ("Pacific/Kiritimati", "2024-01-16 02:00:00 +14"),
    ],
)
def test_current_time_text_for_a_chosen_zone(frozen_clock, zone, expected):
    assert layout.current_time_text(zone) == expected


@pytest.mark.parametrize("zone", ["Not/A_Zone", "../etc/passwd"])
def test_current_time_text_falls_back_to_utc_for_unknown_zones(frozen_clock, zone):
    assert layout.current_time_text(zone) == "2024-01-15 12:00:00 UTC"


def test_current_time_text_for_local_time_has_the_clock_shape(frozen_clock):
    # The local zone depends on the host, so only the shape is checked.
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d( \S+)?", layout.current_time_text(None))


# ---- timezone dropdown --------------------------------------------------------------


@pytest.mark.parametrize(("screen_height", "screen_width"), SCREEN_SIZES)
@pytest.mark.parametrize("anchor_right_offset", [0, 5, 40])
def test_dropdown_rect_stays_on_screen(screen_height, screen_width, anchor_right_offset):
    anchor_top = MARGIN_Y + 1
    anchor_right = screen_width - MARGIN_X - anchor_right_offset
    top, left, height, width = layout.dropdown_rect(anchor_top, anchor_right, screen_height, screen_width)
    assert top == anchor_top + constants.DROPDOWN_MARGIN_Y
    assert constants.DROPDOWN_MIN_HEIGHT <= height <= constants.DROPDOWN_MAX_HEIGHT
    assert top + height <= screen_height - constants.DROPDOWN_MARGIN_Y
    assert width <= constants.DROPDOWN_WIDTH
    assert left >= constants.DROPDOWN_MARGIN_X
    assert left + width <= screen_width - constants.DROPDOWN_MARGIN_X
    # right-aligned under the anchor whenever the screen allows it
    if anchor_right - width >= constants.DROPDOWN_MARGIN_X:
        assert left + width == min(anchor_right, screen_width - constants.DROPDOWN_MARGIN_X)


def test_dropdown_rect_keeps_a_minimum_width_on_tiny_screens():
    assert layout.dropdown_rect(0, 5, 30, 8)[3] == 10


@pytest.mark.parametrize(("visible", "total"), [(1, 1), (1, 5), (3, 3), (5, 3), (5, 40), (6, 7)])
def test_dropdown_first_index_keeps_the_cursor_row_visible(visible, total):
    for cursor in range(total):
        first = layout.dropdown_first_index(cursor, visible, total)
        assert 0 <= first <= max(0, total - visible)
        assert first <= cursor < first + visible


@pytest.mark.parametrize(("cursor", "visible", "total"), [(4, 0, 10), (4, -1, 10), (2, 10, 10), (0, 5, 0)])
def test_dropdown_first_index_starts_at_the_top_when_everything_fits(cursor, visible, total):
    assert layout.dropdown_first_index(cursor, visible, total) == 0


def _dropdown(window, colors, rect=(2, 10, 12, 34), selected=None, query="", cursor=0):
    return layout.draw_dropdown(window, rect, selected, colors, query, cursor)


def test_dropdown_lists_entries_and_highlights_the_cursor(fake_zones, colors):
    window = FakeWindow(20, 60)
    result = _dropdown(window, colors, selected="Europe/London", cursor=2)
    top, left, height, width = result.rect
    rows = height - 4  # borders, search row, rule row
    assert result.first_index == 0
    assert len(result.option_rows) == rows
    assert result.option_rows == tuple(range(top + 3, top + 3 + rows))
    x = left + 1 + constants.DROPDOWN_PADDING_X
    options = constants.TIMEZONE_OPTIONS
    for offset, row in enumerate(result.option_rows):
        label, zone = options[offset]
        mark = constants.DROPDOWN_ACTIVE_MARK if zone == "Europe/London" else constants.DROPDOWN_INACTIVE_MARK
        assert window.lines()[row][x:].startswith(mark + label)
        reversed_ = bool(window.attrs[row][x] & curses.A_REVERSE)
        assert reversed_ == (offset == 2)
    # every listed row stays inside the box, left of the right border
    for row in result.option_rows:
        assert top < row < top + height - 1
        assert window.lines()[row][left + width - 1] == "│"


def test_dropdown_shows_the_placeholder_and_position_counter(fake_zones, colors):
    window = FakeWindow(20, 60)
    _dropdown(window, colors, cursor=4)
    total = len(layout.timezone_choices(""))
    text = window.text()
    assert constants.DROPDOWN_PLACEHOLDER in text
    assert f" 5/{total} " in text
    assert f" {constants.DROPDOWN_TITLE} " in text


def test_dropdown_scrolls_so_the_cursor_row_is_visible(fake_zones, colors):
    window = FakeWindow(20, 60)
    total = len(layout.timezone_choices(""))
    result = _dropdown(window, colors, cursor=total - 1)
    rows = len(result.option_rows)
    assert result.first_index == total - rows
    last_row = result.option_rows[-1]
    x = result.rect[1] + 1 + constants.DROPDOWN_PADDING_X
    assert window.attrs[last_row][x] & curses.A_REVERSE
    assert "Asia/Chelyabinsk" in window.lines()[last_row]


def test_dropdown_clamps_an_out_of_range_cursor(fake_zones, colors):
    window = FakeWindow(20, 60)
    total = len(layout.timezone_choices(""))
    result = _dropdown(window, colors, cursor=999)
    assert f" {total}/{total} " in window.text()
    assert result.first_index == total - len(result.option_rows)


def test_dropdown_echoes_the_query_with_a_block_cursor(fake_zones, colors):
    window = FakeWindow(20, 60)
    result = _dropdown(window, colors, query="hel")
    search_row = result.rect[0] + 1 + constants.DROPDOWN_PADDING_Y
    assert "> hel\u2588" in window.lines()[search_row]
    assert " 1/2 " in window.text()
    assert len(result.option_rows) == 2


def test_dropdown_long_query_shows_its_tail(fake_zones, colors):
    window = FakeWindow(20, 60)
    query = "abcdefghijklmnopqrstuvwxyz0123456789"
    result = _dropdown(window, colors, query=query)
    top, left, height, width = result.rect
    search_row = top + 1 + constants.DROPDOWN_PADDING_Y
    line = window.lines()[search_row]
    assert "789\u2588" in line
    assert line[left + width - 1] == "│"


def test_dropdown_without_matches_says_so(fake_zones, colors):
    window = FakeWindow(20, 60)
    result = _dropdown(window, colors, query="zzzz")
    assert result.option_rows == ()
    assert constants.DROPDOWN_NO_MATCH in window.text()
    assert " 0 " in window.text()


def test_dropdown_paints_its_margin_and_border(fake_zones, colors):
    window = FakeWindow(20, 60)
    window.cells = [["#"] * 60 for _ in range(20)]
    top, left, height, width = _dropdown(window, colors).rect
    lines = window.lines()
    mx, my = constants.DROPDOWN_MARGIN_X, constants.DROPDOWN_MARGIN_Y
    assert lines[top][left] == "┌" and lines[top + height - 1][left + width - 1] == "┘"
    assert lines[top + 1][left - mx : left] == " " * mx  # margin cleared
    assert lines[top + 1][left - mx - 1] == "#"  # outside the margin untouched
    assert lines[top + height + my][left] == "#"


# ---- derived constants ------------------------------------------------------------------


def test_first_key_takes_the_first_binding():
    assert constants._first_key(["+", "="]) == "+"
    assert constants._first_key(("a",)) == "a"
    assert constants._first_key("q") == "q"
    assert constants.QUIT_KEY == constants._first_key(CFG["app"]["keys"]["quit"]).upper()
    assert constants.PANEL_BUTTON_SHOW == f"[{constants.PANEL_KEY}] SHOW PANEL"
    assert (constants.QUIT_KEY, "QUIT", "quit") in constants.HELP_ITEMS


def test_derived_sizes_respect_their_floors_and_caps():
    assert constants.DROPDOWN_WIDTH <= 34
    assert constants.DROPDOWN_MIN_HEIGHT <= 8 and constants.DROPDOWN_MAX_HEIGHT <= 14
    assert constants.DROPDOWN_MIN_HEIGHT <= constants.DROPDOWN_MAX_HEIGHT
    assert constants.MODAL_PADDING_X >= 4 and constants.MODAL_PADDING_Y >= 2
    assert constants.MODAL_MARGIN_X >= 2 and constants.MODAL_MARGIN_Y >= 1
    assert constants.MAP_EDGE_PAD_Y >= 1
    assert len(constants.OUTER_MARGIN) == 2 and min(constants.OUTER_MARGIN) >= 0
    # The default panel fits a minimum-size terminal next to the minimum map.
    assert constants.PANEL_MIN_HEIGHT + constants.PANEL_MIN_MAP_HEIGHT <= MIN_HEIGHT


def test_box_characters_are_single_cells():
    assert set(constants._BOX) == {"top_left", "top_right", "bottom_left", "bottom_right", "horizontal", "vertical"}
    assert all(len(char) == 1 for char in constants._BOX.values())


def test_curated_timezones_are_real_zones():
    assert constants.TIMEZONE_OPTIONS[0][1] is None  # local time comes first
    for label, zone in constants.TIMEZONE_OPTIONS:
        assert isinstance(label, str) and label
        if zone is not None:
            constants.ZoneInfo(zone)  # raises if unknown


def test_banner_font_covers_the_banner_text():
    for letter in constants.BANNER_TEXT.upper():
        glyph = constants._BANNER_FONT[letter]
        assert len(glyph) == constants.BANNER_ROWS
        assert {len(row) for row in glyph} == {5}
        assert set("".join(glyph)) <= {"X", "."}


def test_attribution_colour_is_the_legend_colour_at_half_brightness():
    ui_map = importlib.import_module("terrascope.core.ui.map")
    legend = constants.LEGEND_TEXT_COLOR.lstrip("#")
    expected = "#" + "".join(f"{int(legend[i:i + 2], 16) // 2:02x}" for i in (0, 2, 4))
    assert ui_map.ATTRIBUTION_COLOR == expected
