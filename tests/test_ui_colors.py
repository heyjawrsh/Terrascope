"""Colour conversions and ColorManager pair allocation in core.ui.colors."""

from __future__ import annotations

import curses
import importlib
import io

import pytest

colors_mod = importlib.import_module("terrascope.core.ui.colors")
ColorManager = colors_mod.ColorManager


class FakeCurses:
    """Just enough of the curses module for ColorManager, recording every
    palette / pair call instead of talking to a terminal."""

    error = curses.error
    A_COLOR = curses.A_COLOR
    COLOR_BLACK = curses.COLOR_BLACK
    COLOR_RED = curses.COLOR_RED
    COLOR_GREEN = curses.COLOR_GREEN
    COLOR_YELLOW = curses.COLOR_YELLOW
    COLOR_BLUE = curses.COLOR_BLUE
    COLOR_MAGENTA = curses.COLOR_MAGENTA
    COLOR_CYAN = curses.COLOR_CYAN
    COLOR_WHITE = curses.COLOR_WHITE

    def __init__(self, colors=256, pairs=32767, can_change=True, has_colors=True):
        self.COLORS = colors
        self.COLOR_PAIRS = pairs
        self._can_change = can_change
        self._has_colors = has_colors
        self.started = False
        self.default_colors_error = False
        self.init_pair_fails = set()  # backgrounds init_pair refuses
        self.init_color_fails = False
        self.pairs: dict[int, tuple[int, int]] = {}
        self.palette: dict[int, tuple[int, int, int]] = {}
        self.calls: list[str] = []

    def has_colors(self):
        return self._has_colors

    def start_color(self):
        self.started = True

    def use_default_colors(self):
        if self.default_colors_error:
            raise curses.error("no default colours")

    def can_change_color(self):
        return self._can_change

    def init_pair(self, pair_id, fg, bg):
        self.calls.append("init_pair")
        if bg in self.init_pair_fails:
            raise curses.error("bad pair")
        self.pairs[pair_id] = (fg, bg)

    def init_color(self, slot, r, g, b):
        self.calls.append("init_color")
        if self.init_color_fails:
            raise curses.error("palette is read-only")
        self.palette[slot] = (r, g, b)

    def color_pair(self, pair_id):
        return pair_id << 8


@pytest.fixture
def fake_curses(monkeypatch):
    fake = FakeCurses()
    monkeypatch.setattr(colors_mod, "curses", fake)
    return fake


def _manager(fake, monkeypatch, *, redefine=True, max_pairs=None):
    monkeypatch.setitem(colors_mod.TERMINAL_CONFIG, "redefine_palette", redefine)
    monkeypatch.setitem(colors_mod.TERMINAL_CONFIG, "max_color_pairs", max_pairs)
    manager = ColorManager()
    manager.setup()
    return manager


# ---- conversions --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "rgb"),
    [
        ("#000000", (0, 0, 0)),
        ("#ffffff", (255, 255, 255)),
        ("#FFFFFF", (255, 255, 255)),
        ("1E2C39", (30, 44, 57)),
        ("#abc", (170, 187, 204)),
        ("  #f00  ", (255, 0, 0)),
    ],
)
def test_hex_to_rgb(text, rgb):
    assert colors_mod.hex_to_rgb(text) == rgb


@pytest.mark.parametrize("text", ["", "#", "#12345", "#1234567", "#ggg", "red"])
def test_hex_to_rgb_rejects_malformed_colours(text):
    with pytest.raises(ValueError):
        colors_mod.hex_to_rgb(text)


@pytest.mark.parametrize(
    ("index", "rgb"),
    [
        (0, (0, 0, 0)),
        (9, (255, 0, 0)),
        (15, (255, 255, 255)),
        (16, (0, 0, 0)),  # first cube entry
        (21, (0, 0, 255)),
        (196, (255, 0, 0)),
        (231, (255, 255, 255)),  # last cube entry
        (232, (8, 8, 8)),  # first grey
        (244, (128, 128, 128)),
        (255, (238, 238, 238)),  # last grey
    ],
)
def test_xterm_256_index_to_rgb_boundaries(index, rgb):
    assert colors_mod.xterm_256_index_to_rgb(index) == rgb


def test_every_xterm_colour_maps_back_to_an_identical_colour():
    for index in range(256):
        rgb = colors_mod.xterm_256_index_to_rgb(index)
        nearest = colors_mod.rgb_to_nearest_xterm_256(rgb)
        assert colors_mod.xterm_256_index_to_rgb(nearest) == rgb


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((255, 0, 0), 9),  # ties go to the lowest index
        ((128, 128, 128), 244),
        ((1, 2, 3), 0),
        ((250, 250, 250), 15),
        ((90, 0, 140), 54),  # (95, 0, 135)
    ],
)
def test_rgb_to_nearest_xterm_256(rgb, expected):
    assert colors_mod.rgb_to_nearest_xterm_256(rgb) == expected


def test_nearest_xterm_skips_slots_the_app_redefined():
    assert colors_mod.rgb_to_nearest_xterm_256((255, 0, 0), exclude={9}) == 196
    assert colors_mod.rgb_to_nearest_xterm_256((0, 0, 0), exclude={0, 16}) not in (0, 16)


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((0, 0, 0), curses.COLOR_BLACK),
        ((255, 255, 255), curses.COLOR_WHITE),
        ((220, 10, 10), curses.COLOR_RED),
        ((0, 0, 255), curses.COLOR_BLUE),
        ((0, 200, 210), curses.COLOR_CYAN),
        ((210, 200, 0), curses.COLOR_YELLOW),
    ],
)
def test_rgb_to_nearest_basic(rgb, expected):
    assert colors_mod.rgb_to_nearest_basic(rgb) == expected


# ---- ColorManager ------------------------------------------------------------------


def test_manager_that_was_never_set_up_hands_out_pair_zero(fake_curses):
    manager = ColorManager()
    assert manager.get_pair("#ff0000") == 0
    assert manager.get_pair("#00ff00", "#000000") == 0
    assert fake_curses.calls == []


def test_setup_without_colour_support_stays_disabled(monkeypatch):
    fake = FakeCurses(has_colors=False)
    monkeypatch.setattr(colors_mod, "curses", fake)
    manager = _manager(fake, monkeypatch)
    assert not manager.enabled and not fake.started
    assert manager.get_pair("#ff0000") == 0


def test_setup_survives_a_terminal_without_default_colours(fake_curses, monkeypatch):
    fake_curses.default_colors_error = True
    manager = _manager(fake_curses, monkeypatch)
    assert manager.enabled


@pytest.mark.parametrize(
    ("colors", "can_change", "redefine", "exact"),
    [
        (256, True, True, True),
        (256, False, True, False),
        (256, True, False, False),
        (16, True, True, False),  # no slot above the 16 basic ones
    ],
)
def test_setup_decides_whether_palette_slots_can_be_redefined(monkeypatch, colors, can_change, redefine, exact):
    fake = FakeCurses(colors=colors, can_change=can_change)
    monkeypatch.setattr(colors_mod, "curses", fake)
    assert _manager(fake, monkeypatch, redefine=redefine).supports_exact_color is exact


@pytest.mark.parametrize(
    ("pairs", "configured", "limit"),
    [
        (32767, None, (curses.A_COLOR >> 8) + 1),  # capped by the attribute bits
        (64, None, 64),
        (32767, 10, 11),
        (1, None, 2),  # never below 2
    ],
)
def test_setup_caps_the_pair_budget(monkeypatch, pairs, configured, limit):
    fake = FakeCurses(pairs=pairs)
    monkeypatch.setattr(colors_mod, "curses", fake)
    assert _manager(fake, monkeypatch, max_pairs=configured)._pair_limit == limit


def test_pairs_are_cached_per_colour_combination(fake_curses, monkeypatch):
    manager = _manager(fake_curses, monkeypatch)
    red = manager.get_pair("#ff0000")
    assert red == fake_curses.color_pair(1)
    assert manager.get_pair(" #FF0000 ") == red  # normalised before lookup
    assert fake_curses.calls.count("init_pair") == 1
    on_black = manager.get_pair("#ff0000", "#000000")
    assert on_black not in (0, red)
    assert manager.get_pair("#ff0000", "#000000") == on_black
    assert fake_curses.calls.count("init_pair") == 2


def test_exact_colours_redefine_free_palette_slots(fake_curses, monkeypatch):
    manager = _manager(fake_curses, monkeypatch)
    manager.get_pair("#ff8000", "#000000")
    first = ColorManager.FIRST_FREE_SLOT
    assert fake_curses.palette == {first: (1000, 502, 0), first + 1: (0, 0, 0)}
    assert fake_curses.pairs[1] == (first, first + 1)
    # The same colour as foreground or background reuses its slot.
    manager.get_pair("#000000", "#ff8000")
    assert fake_curses.pairs[2] == (first + 1, first)
    assert fake_curses.calls.count("init_color") == 2


def test_without_exact_colour_the_nearest_xterm_colour_is_used(monkeypatch):
    fake = FakeCurses(can_change=False)
    monkeypatch.setattr(colors_mod, "curses", fake)
    manager = _manager(fake, monkeypatch)
    manager.get_pair("#fe0101")
    assert fake.pairs[1] == (9, -1)  # -1 = the terminal's default background
    assert fake.palette == {}


def test_eight_colour_terminals_get_the_nearest_basic_colour(monkeypatch):
    fake = FakeCurses(colors=8, can_change=False)
    monkeypatch.setattr(colors_mod, "curses", fake)
    manager = _manager(fake, monkeypatch)
    manager.get_pair("#10e020", "#0000f0")
    assert fake.pairs[1] == (curses.COLOR_GREEN, curses.COLOR_BLUE)


def test_a_read_only_palette_falls_back_to_nearest_colours(fake_curses, monkeypatch):
    fake_curses.init_color_fails = True
    manager = _manager(fake_curses, monkeypatch)
    manager.get_pair("#ff0000")
    assert fake_curses.pairs[1] == (9, -1)
    assert manager._redefined == set()
    assert manager._next_color_slot == ColorManager.FIRST_FREE_SLOT


def test_slot_exhaustion_falls_back_to_nearest_colours(monkeypatch):
    fake = FakeCurses(colors=ColorManager.FIRST_FREE_SLOT + 1)
    monkeypatch.setattr(colors_mod, "curses", fake)
    monkeypatch.setitem(colors_mod.TERMINAL_CONFIG, "redefine_palette", True)
    monkeypatch.setitem(colors_mod.TERMINAL_CONFIG, "max_color_pairs", None)
    manager = ColorManager()
    manager.setup()
    manager.get_pair("#123456")  # takes the only free slot
    manager.get_pair("#ffffff")  # nothing left: nearest basic colour
    assert fake.pairs[1][0] == ColorManager.FIRST_FREE_SLOT
    assert fake.pairs[2][0] == colors_mod.rgb_to_nearest_basic((255, 255, 255))


def test_unsupported_default_background_falls_back_to_black(fake_curses, monkeypatch):
    fake_curses.init_pair_fails = {-1}
    manager = _manager(fake_curses, monkeypatch, redefine=False)
    assert manager.get_pair("#ff0000") != 0
    assert fake_curses.pairs[1] == (9, curses.COLOR_BLACK)


def test_a_pair_the_terminal_refuses_is_attribute_zero(fake_curses, monkeypatch):
    fake_curses.init_pair_fails = {-1, curses.COLOR_BLACK}
    manager = _manager(fake_curses, monkeypatch, redefine=False)
    assert manager.get_pair("#ff0000") == 0


@pytest.mark.parametrize(("fg", "bg"), [("not-a-colour", None), ("#ff0000", "#12")])
def test_malformed_colours_get_attribute_zero_without_using_a_pair(fake_curses, monkeypatch, fg, bg):
    manager = _manager(fake_curses, monkeypatch)
    assert manager.get_pair(fg, bg) == 0
    assert fake_curses.calls == []
    assert manager.get_pair("#00ff00") == fake_curses.color_pair(1)


def test_exhausted_budget_reuses_the_closest_existing_pair(fake_curses, monkeypatch):
    manager = _manager(fake_curses, monkeypatch, max_pairs=3)
    red_on_blue = manager.get_pair("#ff0000", "#0000ff")
    dark_red = manager.get_pair("#c80000")
    green = manager.get_pair("#00ff00")
    assert len({red_on_blue, dark_red, green}) == 3
    assert not manager.exhausted
    # No pair left: plain red is closest to dark red (same, absent background)
    # rather than to exact red on blue.
    assert manager.get_pair("#ff0000") == dark_red
    assert manager.exhausted
    assert manager.get_pair("#10f010") == green
    assert manager.get_pair("#ff1010", "#0000f0") == red_on_blue
    assert fake_curses.calls.count("init_pair") == 3


def test_reset_frees_the_pair_budget_but_keeps_palette_slots(fake_curses, monkeypatch):
    manager = _manager(fake_curses, monkeypatch, max_pairs=1)
    manager.get_pair("#ff0000")
    manager.get_pair("#00ff00")
    assert manager.exhausted
    colors_before = fake_curses.calls.count("init_color")

    manager.reset()
    assert not manager.exhausted
    assert manager.get_pair("#ff0000") == fake_curses.color_pair(1)
    assert fake_curses.pairs[1][0] == ColorManager.FIRST_FREE_SLOT  # red's old slot
    assert fake_curses.calls.count("init_color") == colors_before
    assert fake_curses.calls.count("init_pair") == 2


class _Stdout(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.mark.parametrize("tty", [True, False])
def test_restore_resets_a_redefined_palette_only_on_a_terminal(fake_curses, monkeypatch, tty):
    stdout = _Stdout(tty)
    monkeypatch.setattr(colors_mod, "sys", type("FakeSys", (), {"stdout": stdout}))
    manager = _manager(fake_curses, monkeypatch)
    manager.get_pair("#ff0000")
    manager.restore()
    assert stdout.getvalue() == ("\033]104\007" if tty else "")
    assert (manager._redefined == set()) is tty


def test_restore_does_nothing_when_the_palette_was_never_touched(monkeypatch):
    stdout = _Stdout(True)
    monkeypatch.setattr(colors_mod, "sys", type("FakeSys", (), {"stdout": stdout}))
    ColorManager().restore()
    assert stdout.getvalue() == ""


def test_restore_ignores_a_closed_terminal(fake_curses, monkeypatch):
    class BrokenStdout(_Stdout):
        def write(self, text):
            raise OSError("terminal went away")

    monkeypatch.setattr(colors_mod, "sys", type("FakeSys", (), {"stdout": BrokenStdout(True)}))
    manager = _manager(fake_curses, monkeypatch)
    manager.get_pair("#ff0000")
    manager.restore()  # does not raise
