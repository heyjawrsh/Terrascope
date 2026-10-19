"""Welcome modal layout and the info / text / live / rich panel boxes in core.ui.dialogs."""

from __future__ import annotations

import curses
import importlib

import pytest

from conftest import FakeWindow
from terrascope.core.config import CFG
from terrascope.core.layer import InfoImage, Layer
from terrascope.core.tabs import TABS

dialogs = importlib.import_module("terrascope.core.ui.dialogs")
constants = importlib.import_module("terrascope.core.ui.constants")

MIN_HEIGHT = CFG["app"]["min_terminal_height"]
MIN_WIDTH = CFG["app"]["min_terminal_width"]
LINK_TEXT = f"GITHUB   {constants.GITHUB_URL}"


def _text_at(window, y, x, length) -> str:
    return "".join(window.cells[y][x : x + length])


# ---- welcome modal ------------------------------------------------------------------


@pytest.mark.parametrize(("height", "width"), [(13, 120), (40, 29), (5, 5)])
def test_welcome_modal_needs_a_minimum_screen(colors, height, width):
    window = FakeWindow(height, width)
    assert dialogs.draw_welcome_modal(window, height, width, colors) == (None,) * 5
    assert window.text().strip() == ""


MODAL_SIZES = [
    (MIN_HEIGHT, MIN_WIDTH),
    (22, MIN_WIDTH),
    (24, 80),
    (30, 100),
    (41, 99),
    (45, 140),
    (60, 200),
]


@pytest.mark.parametrize(("height", "width"), MODAL_SIZES)
@pytest.mark.parametrize("province_prompt", [False, True])
def test_welcome_modal_and_its_click_targets_stay_inside_the_card(colors, height, width, province_prompt):
    window = FakeWindow(height, width)
    rect, link, button, yes, no = dialogs.draw_welcome_modal(
        window, height, width, colors, province_prompt=province_prompt
    )
    top, left, modal_height, modal_width = rect
    assert top >= 0 and left >= 0
    assert top + modal_height <= height and left + modal_width <= width
    assert window.cells[top][left] == "┌"
    assert window.cells[top + modal_height - 1][left] == "└"

    targets = [target for target in (link, button, yes, no) if target is not None]
    padding_x = constants.MODAL_PADDING_X
    for target in targets:
        assert top < target.row < top + modal_height - 1
        assert left + 1 + padding_x <= target.left
        assert target.left + target.width <= left + modal_width - 1 - padding_x
        assert target.width > 0
    if link is not None:
        drawn = _text_at(window, link.row, link.left, link.width)
        assert LINK_TEXT.startswith(drawn)
        assert window.attrs[link.row][link.left] & curses.A_UNDERLINE
    if button is not None:
        assert _text_at(window, button.row, button.left, button.width) == constants.WELCOME_SKIP_BUTTON
        assert window.attrs[button.row][button.left] & curses.A_REVERSE
    if yes is not None and no is not None:
        assert yes.row == no.row
        assert yes.left + yes.width < no.left
        assert "Y DOWNLOAD" in _text_at(window, yes.row, yes.left, yes.width)
        assert "N KEEP OFF" in _text_at(window, no.row, no.left, no.width)
        assert button is None or button.row > yes.row
    rows = [target.row for target in (link, button, yes) if target is not None]
    assert len(rows) == len(set(rows))


@pytest.mark.parametrize(("height", "width"), [(24, 80), (45, 140)])
def test_welcome_modal_offers_every_button_when_there_is_room(colors, height, width):
    window = FakeWindow(height, width)
    _, link, button, yes, no = dialogs.draw_welcome_modal(window, height, width, colors, province_prompt=True)
    assert None not in (link, button, yes, no)
    assert link.width == len(LINK_TEXT)


def test_welcome_modal_without_the_prompt_has_no_province_buttons(colors):
    window = FakeWindow(45, 140)
    _, _, button, yes, no = dialogs.draw_welcome_modal(window, 45, 140, colors)
    assert button is not None and yes is None and no is None
    assert "Y DOWNLOAD" not in window.text()


@pytest.mark.parametrize("width", [30, 36, 40])
@pytest.mark.xfail(
    strict=True,
    reason="BUG: draw_welcome_modal floors the card width at 40, so on screens 30-40 wide "
    "(allowed with --min-width) the card overflows the right edge",
)
def test_welcome_modal_fits_narrow_screens_it_agrees_to_draw_on(colors, width):
    window = FakeWindow(20, width)
    rect, *_ = dialogs.draw_welcome_modal(window, 20, width, colors)
    top, left, _, modal_width = rect
    assert left + modal_width <= width
    # the right border is visible somewhere on the top row
    assert window.cells[top][left + modal_width - 1] == "┐"


def test_full_welcome_modal_shows_the_banner_tagline_and_tab_guide(colors):
    window = FakeWindow(45, 140)
    dialogs.draw_welcome_modal(window, 45, 140, colors)
    text = window.text()
    assert constants.WELCOME_TAGLINE in text
    assert constants.BANNER_FILL * 4 in text
    assert "EXPLORE" in text and "CONTROLS" in text
    for index, tab in enumerate(TABS):
        assert f"{index + 1} {tab.label}" in text
        if tab.blurb:
            assert tab.blurb[:20] in text
    assert constants.WELCOME_DISMISS_HINT in text and constants.WELCOME_REOPEN_HINT in text


def test_compact_welcome_modal_uses_short_text(colors):
    window = FakeWindow(24, 80)
    dialogs.draw_welcome_modal(window, 24, 80, colors)
    text = window.text()
    assert "TERRA SCOPE" in text
    assert "A live world map, drawn in braille." in text
    assert constants.WELCOME_TAGLINE not in text
    assert "EXPLORE" not in text


def _headline(colors, status) -> str:
    window = FakeWindow(45, 140)
    dialogs.draw_welcome_modal(window, 45, 140, colors, startup_status=status)
    lines = window.lines()
    index = next(i for i, line in enumerate(lines) if "FIRST RUN  /  DATA DOWNLOADS" in line)
    return lines[index + 1].strip(" │")


@pytest.mark.parametrize(
    ("step", "state", "expected"),
    [
        ("SETUP COMPLETE", "ready", "READY  /  SETUP COMPLETE"),
        ("LIVE AIRCRAFT", "downloading", "DOWNLOADING  /  AIRCRAFT FEED"),
        ("WEATHER RADAR", "waiting", "QUEUED  /  PRESENT RADAR"),
        ("CITY WEATHER", "ready", "READY  /  CITY WEATHER"),
        ("RADAR HISTORY", "downloading", "DOWNLOADING  /  OTHER RADAR TIMES"),
        ("ne_50m_admin_0_countries.geojson", "downloading", "DOWNLOADING  /  WORLD MAP"),
        ("ne_10m_admin_1_states_provinces.geojson", "waiting", "QUEUED  /  DETAILED BORDERS"),
        ("ne_10m_populated_places.geojson", "ready", "READY  /  DETAILED CITIES"),
        ("ne_10m_airports.geojson", "ready", "READY  /  AIRPORTS"),
        ("ne_10m_lakes.geojson", "paused", "PAUSED  /  10M LAKES"),
        ("WEATHER RADAR", "failed", "RADAR UNAVAILABLE"),
    ],
)
def test_welcome_modal_names_the_current_download_step(colors, step, state, expected):
    assert _headline(colors, (step, 50, state, "")) == expected


def test_welcome_modal_appends_a_short_error(colors):
    headline = _headline(colors, ("LIVE AIRCRAFT", 20, "failed", "HTTP Error 503: Service Unavailable"))
    assert headline == "FAILED  /  AIRCRAFT FEED  ·  HTTP 503"


def test_radar_failure_hides_the_raw_error(colors):
    assert _headline(colors, ("WEATHER RADAR", 20, "failed", "timed out")) == "RADAR UNAVAILABLE"


@pytest.mark.parametrize(("percent", "fraction"), [(0, 0.0), (50, 0.5), (100, 1.0), (-20, 0.0), (180, 1.0)])
def test_welcome_progress_bar_fill_is_clamped(colors, percent, fraction):
    window = FakeWindow(45, 140)
    dialogs.draw_welcome_modal(window, 45, 140, colors, startup_status=("LIVE AIRCRAFT", percent, "downloading", ""))
    line = next(line for line in window.lines() if f"{percent:3d}%" in line and "[" in line)
    bar = line[line.index("[") + 1 : line.index("]")]
    assert set(bar) <= {"=", "-"}
    assert bar.count("=") == round(len(bar) * fraction)


def test_banner_lines_are_rectangular_and_scale(colors):
    small = dialogs._banner_lines(1, "terra")
    big = dialogs._banner_lines(2, "terra")
    assert len(small) == len(big) == constants.BANNER_ROWS
    assert len({len(line) for line in small}) == 1
    # 5 letters x 5 columns + 4 gaps, all doubled at scale 2
    assert len(small[0]) == 5 * 5 + 4
    assert len(big[0]) == 2 * len(small[0])
    assert set("".join(small)) == {constants.BANNER_FILL, " "}


def test_modal_body_expands_the_tab_list():
    rows = dialogs._flatten_modal_body()
    texts = [text for text, _, _ in rows]
    assert "@tabs" not in texts
    tab_rows = [row for row in rows if row[0].startswith(" 1 ")]
    assert tab_rows and tab_rows[0][2] is True  # left-aligned
    assert ("EXPLORE", True, True) in rows
    assert ("CONTROLS", True, False) in rows
    assert sum(1 for text in texts if text.lstrip()[:1].isdigit()) == len(TABS)


# ---- info box -----------------------------------------------------------------------


class InfoLayer(Layer):
    info_title = "PLANE"

    def __init__(self, rows=(), image=None):
        super().__init__()
        self.rows = list(rows)
        self.image = image
        self.image_requests = []

    def info_rows(self):
        return self.rows

    def info_image(self, rows, max_cols):
        self.image_requests.append((rows, max_cols))
        return self.image


@pytest.mark.parametrize(
    ("layer", "position", "expected"),
    [
        (None, (0, 0), "INFO"),
        (None, (2, 5), "INFO  < 2 of 5 >"),
        (InfoLayer(), (1, 1), "PLANE"),
        (InfoLayer(), (3, 4), "PLANE  < 3 of 4 >"),
    ],
)
def test_info_title_text(layer, position, expected):
    assert dialogs.info_title_text(layer, position) == expected


@pytest.mark.parametrize(("layer", "position"), [(None, (1, 3)), (InfoLayer(), (1, 1)), (InfoLayer(), (0, 0))])
def test_info_arrows_are_hidden_without_several_selections(layer, position):
    assert dialogs.info_arrow_positions(10, layer, position) is None


@pytest.mark.parametrize("position", [(1, 2), (3, 9), (12, 140)])
def test_info_arrow_positions_match_the_drawn_title(position):
    layer = InfoLayer()
    window = FakeWindow(10, 60)
    dialogs.draw_info_box(window, 1, 4, 8, 50, layer, position)
    left_x, right_x = dialogs.info_arrow_positions(4, layer, position)
    assert window.cells[1][left_x] == "<"
    assert window.cells[1][right_x] == ">"


@pytest.mark.parametrize(("height", "width"), [(2, 40), (8, constants.INFO_BOX_MIN_WIDTH - 1)])
def test_info_box_too_small_draws_nothing(height, width):
    window = FakeWindow(10, 60)
    dialogs.draw_info_box(window, 0, 0, height, width, InfoLayer([("A", "b")]))
    assert window.text().strip() == ""


def test_info_box_without_a_layer_shows_the_placeholder():
    window = FakeWindow(10, 40)
    dialogs.draw_info_box(window, 0, 0, 8, 30, None)
    lines = window.lines()
    assert " INFO " in lines[0]
    for offset, line in enumerate(constants.INFO_PLACEHOLDER):
        if line:
            assert lines[1 + offset][2:].startswith(line)


def test_info_box_wraps_values_and_truncates_labels():
    label_width = constants.INFO_BOX_LABEL_WIDTH
    window = FakeWindow(12, 60)
    layer = InfoLayer([("CALLSIGN", "KLM1234"), ("A" * (label_width + 5), "short"), ("ROUTE", "word " * 12)])
    dialogs.draw_info_box(window, 0, 0, 10, 40, layer)
    lines = window.lines()
    value_x = 2 + label_width + 1
    assert lines[1][2:].startswith("CALLSIGN")
    assert lines[1][value_x:].startswith("KLM1234")
    assert lines[2][2 : 2 + label_width] == "A" * label_width
    assert lines[2][2 + label_width] == " "
    assert lines[3][value_x:].startswith("word")
    assert lines[4][value_x:].startswith("word")  # wrapped onto the next row
    assert lines[4][2 : 2 + label_width].strip() == ""
    for line in lines[:10]:
        assert line[39] in "│┐┘"  # nothing spills over the right border


def test_info_box_rows_stop_at_the_bottom_border():
    window = FakeWindow(12, 60)
    layer = InfoLayer([(f"ROW{i}", "x " * 30) for i in range(10)])
    dialogs.draw_info_box(window, 0, 0, 5, 40, layer)
    lines = window.lines()
    assert lines[4][:40] == "└" + "─" * 38 + "┘"
    assert lines[5].strip() == ""


def test_info_box_draws_the_layer_picture_on_the_right(colors):
    image = InfoImage(4, ((("ab", "#ff0000", None), ("cdef", None, None)), (("WXYZ", None, "#000000"),)))
    layer = InfoLayer([("NAME", "value")], image)
    window = FakeWindow(12, 80)
    dialogs.draw_info_box(window, 0, 0, 8, 60, layer, colors=colors)
    label_width = constants.INFO_BOX_LABEL_WIDTH
    room = 60 - (label_width + 3) - constants.INFO_IMAGE_MIN_VALUE_WIDTH - constants.INFO_IMAGE_GAP - 2
    assert layer.image_requests == [(6, room)]
    image_left = 60 - 2 - image.width
    assert _text_at(window, 1, image_left, 4) == "abcd"  # clipped to the picture width
    assert _text_at(window, 2, image_left, 4) == "WXYZ"
    assert window.cells[1][59] == "│"


def test_info_box_skips_the_picture_when_there_is_no_room(colors):
    layer = InfoLayer([("NAME", "value")], InfoImage(4, ((("ab", None, None),),)))
    window = FakeWindow(12, 80)
    dialogs.draw_info_box(window, 0, 0, 8, 30, layer, colors=colors)
    assert layer.image_requests == []
    dialogs.draw_info_box(window, 0, 0, 8, 60, layer)  # no colours: no picture either
    assert layer.image_requests == []


# ---- text / live / rich boxes --------------------------------------------------------


def test_text_box_truncates_lines_instead_of_overflowing():
    window = FakeWindow(10, 40)
    lines = [("x" * 100, curses.A_BOLD), ("short", 0), ("third", 0), ("dropped", 0)]
    dialogs.draw_text_box(window, (1, 2, 5, 20), "LOG", lines)
    rows = window.lines()
    assert rows[2][4:20] == "x" * 16
    assert rows[2][20] == " "  # one column of padding before the border
    assert rows[2][21] == "│"
    assert window.attrs[2][4] == curses.A_BOLD
    assert rows[3][4:9] == "short"
    assert rows[4][4:9] == "third"
    assert "dropped" not in window.text()
    assert " LOG " in rows[1]


@pytest.mark.parametrize("draw", ["text", "live", "rich"])
def test_boxes_too_small_draw_nothing(colors, draw):
    window = FakeWindow(10, 40)
    for rect in ((0, 0, 2, 30), (0, 0, 6, constants.INFO_BOX_MIN_WIDTH - 1)):
        if draw == "text":
            dialogs.draw_text_box(window, rect, "T", [("a", 0)])
        elif draw == "live":
            dialogs.draw_live_box(window, rect, "T", [("a", "b")])
        else:
            dialogs.draw_rich_box(window, rect, "T", [[("a", None, "")]], colors)
    assert window.text().strip() == ""


def test_live_box_label_column_fits_the_longest_label():
    window = FakeWindow(10, 50)
    dialogs.draw_live_box(window, (0, 0, 6, 40), "SUN", [("RISE", "06:12"), ("SET", "18:40")])
    lines = window.lines()
    assert lines[1][2:].startswith("RISE 06:12")
    assert lines[2][2:].startswith("SET  18:40")
    assert window.attrs[1][2] & curses.A_BOLD
    assert not window.attrs[1][7] & curses.A_BOLD


def test_live_box_label_column_is_capped():
    cap = constants.LIVE_BOX_MAX_LABEL_WIDTH
    window = FakeWindow(10, 60)
    dialogs.draw_live_box(window, (0, 0, 6, 50), "SUN", [("L" * (cap + 8), "value")])
    line = window.lines()[1]
    assert line[2 : 2 + cap] == "L" * cap
    assert line[2 + cap + 1 :].startswith("value")


def test_live_box_without_rows_draws_just_the_box():
    window = FakeWindow(10, 40)
    dialogs.draw_live_box(window, (0, 0, 5, 30), "CLOCK", [])
    assert " CLOCK " in window.lines()[0]
    assert window.lines()[1][1:29].strip() == ""


def test_rich_box_applies_styles_and_clips_to_the_box(colors):
    window = FakeWindow(10, 40)
    lines = [
        [("BOLD", None, "bold"), (" dim", "#aabbcc", "dim"), (" plain", None, "")],
        [("y" * 50, None, "")],
    ]
    dialogs.draw_rich_box(window, (0, 0, 5, 20), "FORECAST", lines, colors)
    rows = window.lines()
    assert rows[1][2:18] == "BOLD dim plain  "
    assert window.attrs[1][2] == curses.A_BOLD
    assert window.attrs[1][7] == curses.A_DIM
    assert window.attrs[1][11] == 0
    assert rows[2][2:18] == "y" * 16
    assert rows[2][19] == "│"
