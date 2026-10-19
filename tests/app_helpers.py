"""Builders shared by the TerrascopeApp tests: an app on a FakeWindow with no threads, disk or network."""

from __future__ import annotations

import threading

import pytest

from conftest import FakeWindow

_UNSET = object()

TAB_INDEX = {"map": 0, "time": 1, "weather": 2, "planes": 3}

SAMPLE_FLIGHTS = [
    {
        "icao24": "a1b2c3",
        "callsign": "UAL123 ",
        "origin_country": "United States",
        "longitude": -97.04,
        "latitude": 32.9,
    },
    {
        "icao24": "4ca7b5",
        "callsign": "EIN44",
        "origin_country": "Ireland",
        "longitude": -6.27,
        "latitude": 53.42,
    },
    {
        "icao24": "3c6444",
        "callsign": "DLH400",
        "origin_country": "Germany",
        "longitude": 8.57,
        "latitude": 50.03,
    },
    {
        "icao24": "c0ffee",
        "callsign": "UAE2",
        "origin_country": "United Arab Emirates",
        "longitude": 55.36,
        "latitude": 25.25,
    },
]


class RecordedThread:
    """Stands in for threading.Thread in the app modules: remembers what would
    have run so a test can run it synchronously (or not at all)."""

    def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
        self.target = target
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}
        self.started = False

    def start(self) -> None:
        self.started = True
        RecordedThread.started_threads.append(self)

    def run(self) -> None:
        self.target(*self.args, **self.kwargs)

    started_threads: list[RecordedThread] = []


def layer(app, name: str):
    return next(item for item in app.world.layers if item.name == name)


def press(app, *keys) -> None:
    """Feed keys to handle_key; str items are typed one character at a time."""
    for key in keys:
        if isinstance(key, str):
            for char in key:
                app.handle_key(ord(char))
        else:
            app.handle_key(key)


@pytest.fixture
def make_app(monkeypatch, tmp_path):
    """Factory: make_app(height=40, width=120, tab="map", welcome=False, ...).

    The app's cache directory (welcome-skip and province-choice markers) is a
    per-test tmp dir, the province download thread is recorded instead of
    started, and radar retries are counted instead of fetched."""
    from terrascope.core.app import TerrascopeApp, lifecycle, shared
    from terrascope.core.registry import build_layers
    from terrascope.core.world import WorldMap

    monkeypatch.setattr(shared, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lifecycle, "CACHE_DIR", tmp_path)
    RecordedThread.started_threads = []
    monkeypatch.setattr(lifecycle, "Thread", RecordedThread)
    before = {thread.ident for thread in threading.enumerate()}
    radar_retries: list[str] = []

    def build(
        height: int = 40,
        width: int = 120,
        *,
        tab: str = "map",
        welcome: bool = False,
        forced_off: frozenset[str] = frozenset(),
        province_choice: str | None | object = _UNSET,
    ):
        window = FakeWindow(height, width)
        app = TerrascopeApp(window, WorldMap([], build_layers()), forced_off)
        app.welcome_visible = welcome
        if province_choice is not _UNSET:
            app.province_choice = province_choice
        layer(app, "weather").radar.retry = lambda: radar_retries.append("retry")
        if tab != "map":
            app.select_tab(TAB_INDEX[tab])
        return app

    build.cache_dir = tmp_path
    build.threads = RecordedThread.started_threads
    build.radar_retries = radar_retries
    yield build
    stray = [
        thread
        for thread in threading.enumerate()
        if thread.ident not in before and thread.is_alive()
    ]
    assert not stray, f"test left background threads running: {stray}"
