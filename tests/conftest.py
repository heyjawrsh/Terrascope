"""Shared test setup: an isolated config / cache, no network, fake curses.

terrascope reads (and on first run writes) its config and cache under the
XDG directories at *import* time, and several layers fetch live data. So
before any terrascope module is imported, this file points XDG_CONFIG_HOME /
XDG_CACHE_HOME at a throwaway directory, forces offline mode, and replaces
urllib's urlopen with a guard: a test never touches the user's files or the
network. Tests that need an HTTP response register one with `fake_http`.
"""

from __future__ import annotations

import io
import json
import os
import socket
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---- isolation (must run before terrascope is imported) ---------------------

_SANDBOX = Path(tempfile.mkdtemp(prefix="terrascope-tests-"))
os.environ["XDG_CONFIG_HOME"] = str(_SANDBOX / "config")
os.environ["XDG_CACHE_HOME"] = str(_SANDBOX / "cache")
os.environ["terrascope_OFFLINE"] = "1"
os.environ.pop("terrascope_CONFIG", None)


class _NetworkGuard:
    """Stands in for urllib.request.urlopen for the whole session.

    Unregistered URLs fail like an unreachable host (URLError), so code under
    test takes its normal offline / error path; every attempt is recorded so a
    test can assert that nothing was requested."""

    def __init__(self) -> None:
        self.responses: dict[str, object] = {}
        self.calls: list[str] = []

    def __call__(self, request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        self.calls.append(url)
        for prefix, response in self.responses.items():
            if url.startswith(prefix):
                if isinstance(response, BaseException):
                    raise response
                return _FakeResponse(response, url)
        raise urllib.error.URLError("network disabled in tests")


class _FakeResponse(io.BytesIO):
    def __init__(self, body, url: str) -> None:
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode("utf-8")
        super().__init__(bytes(body))
        self.url = url
        self.status = 200
        self.headers = {}

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url


NETWORK = _NetworkGuard()
urllib.request.urlopen = NETWORK


def _no_sockets(self, *args, **kwargs):
    raise OSError("network disabled in tests")


socket.socket.connect = _no_sockets  # belt and braces for anything not using urlopen


# ---- fixtures -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_network():
    NETWORK.responses.clear()
    NETWORK.calls.clear()
    yield
    NETWORK.responses.clear()


@pytest.fixture
def fake_http():
    """Register canned HTTP responses: fake_http.responses[url_prefix] = body,
    where body is bytes, a JSON-serialisable value, or an exception to raise.
    fake_http.calls lists every URL requested during the test."""
    return NETWORK


@pytest.fixture
def cache_dir():
    """The sandboxed cache directory terrascope.core.config resolved."""
    from terrascope.core import config

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR


class FakeWindow:
    """Minimal curses window: records text into a grid so drawing helpers can
    be tested without a terminal. Writes past the edge are clipped, the way
    safe_addstr expects addnstr to behave."""

    def __init__(self, height: int = 40, width: int = 120) -> None:
        self.height = height
        self.width = width
        self.keys: list[int] = []
        self.erase()

    # drawing
    def addnstr(self, y, x, text, n, attributes=0):
        if not 0 <= y < self.height:
            return
        for offset, char in enumerate(str(text)[: max(0, n)]):
            column = x + offset
            if 0 <= column < self.width:
                self.cells[y][column] = char
                self.attrs[y][column] = attributes

    def addstr(self, y, x, text, attributes=0):
        self.addnstr(y, x, text, len(str(text)), attributes)

    def erase(self):
        self.cells = [[" "] * self.width for _ in range(self.height)]
        self.attrs = [[0] * self.width for _ in range(self.height)]

    clear = erase

    def getmaxyx(self):
        return self.height, self.width

    # input
    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    # no-ops the app calls on its screen
    def refresh(self):
        pass

    def clearok(self, flag):
        pass

    def keypad(self, flag):
        pass

    def nodelay(self, flag):
        pass

    # helpers for assertions
    def lines(self) -> list[str]:
        return ["".join(row) for row in self.cells]

    def text(self) -> str:
        return "\n".join(line.rstrip() for line in self.lines())


@pytest.fixture
def fake_window():
    """A 40x120 FakeWindow; build other sizes with FakeWindow(height, width)."""
    return FakeWindow()


@pytest.fixture
def colors():
    """A real ColorManager that was never set up: every pair is 0, no curses
    calls happen, so drawing code runs without a terminal."""
    from terrascope.core.ui import ColorManager

    return ColorManager()


@pytest.fixture
def sample_places():
    """(lon, lat, name, is_capital, population, country) tuples, biggest first,
    in the shape mapdata.load_places() returns."""
    return [
        (139.75, 35.69, "Tokyo", True, 35_676_000, "Japan"),
        (-73.98, 40.75, "New York", False, 19_040_000, "United States of America"),
        (-46.63, -23.55, "São Paulo", False, 18_845_000, "Brazil"),
        (126.998, 37.568, "Seoul", True, 9_796_000, "South Korea"),
        (2.35, 48.86, "Paris", True, 9_904_000, "France"),
        (-1.08, 53.96, "York", False, 153_717, "United Kingdom"),
        (151.21, -33.87, "Sydney", False, 4_630_000, "Australia"),
    ]
