"""End to end: the real terrascope executable in a pseudo-terminal, read back through pyte."""

from __future__ import annotations

import os
import select
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.name != "posix", reason="needs a POSIX pseudo-terminal"),
]

if os.name == "posix":
    import fcntl
    import pty
    import termios

pyte = pytest.importorskip("pyte")

ESC = b"\x1b"
MAP_BOX = "┌─ WORLD MAP"  # the map box title ("WORLD MAP" alone also appears in the guide)
TABS = ("1 MAP", "2 TIME", "3 WEATHER", "4 PLANES")


def terrascope_command() -> list[str]:
    beside_python = Path(sys.executable).with_name("terrascope")
    if beside_python.exists():
        return [str(beside_python)]
    found = shutil.which("terrascope")
    return [found] if found else [sys.executable, "-m", "terrascope"]


class Terminal:
    """The app running in a pty of a given size, its output fed to a pyte screen."""

    def __init__(self, tmp_path: Path, columns: int, rows: int) -> None:
        self.master, self.slave = pty.openpty()
        # select() can report the master readable with nothing to read (macOS,
        # once the app has exited), so a blocking read could hang the test.
        os.set_blocking(self.master, False)
        self.screen = pyte.Screen(columns, rows)
        self.stream = pyte.ByteStream(self.screen)
        self.raw = bytearray()
        self._set_size(columns, rows)
        env = {
            key: value for key, value in os.environ.items() if not key.startswith("terrascope_")
        }
        env.update(
            TERM="xterm-256color",
            terrascope_OFFLINE="1",
            ESCDELAY="25",  # ncurses otherwise holds a bare Esc for a second
            XDG_CONFIG_HOME=str(tmp_path / "config"),
            XDG_CACHE_HOME=str(tmp_path / "cache"),
        )
        env.pop("COLUMNS", None)
        env.pop("LINES", None)
        self.process = subprocess.Popen(
            terrascope_command(),
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            env=env,
            cwd=tmp_path,
            start_new_session=True,
            close_fds=True,
        )

    def _set_size(self, columns: int, rows: int) -> None:
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))

    def resize(self, columns: int, rows: int) -> None:
        self._set_size(columns, rows)
        self.screen.resize(rows, columns)
        self.process.send_signal(signal.SIGWINCH)

    def pump(self, seconds: float = 0.05) -> None:
        end = time.monotonic() + seconds
        while True:
            remaining = end - time.monotonic()
            ready, _, _ = select.select([self.master], [], [], max(0.0, min(0.05, remaining)))
            if ready:
                try:
                    data = os.read(self.master, 65536)
                except BlockingIOError:
                    data = b""  # "readable" with nothing to read: see __init__
                except OSError:
                    return
                self.raw.extend(data)
                self.stream.feed(data)
            if remaining <= 0:
                return

    def text(self) -> str:
        return "\n".join(line.rstrip() for line in self.screen.display)

    def wait_for(self, predicate, timeout: float = 10.0, what: str = "") -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(0.05)
            if predicate():
                return
        raise AssertionError(f"timed out waiting for {what or predicate}; screen:\n{self.text()}")

    def wait_for_text(self, text: str, timeout: float = 10.0) -> None:
        self.wait_for(lambda: text in self.text(), timeout, repr(text))

    def wait_for_gone(self, text: str, timeout: float = 5.0) -> None:
        self.wait_for(lambda: text not in self.text(), timeout, f"{text!r} to disappear")

    def send(self, keys: bytes | str) -> None:
        os.write(self.master, keys.encode() if isinstance(keys, str) else keys)
        self.pump(0.05)

    def active_tab(self) -> str | None:
        """Label of the tab drawn in reverse video on the tab bar."""
        for row, line in enumerate(self.screen.display):
            if TABS[0] not in line or TABS[-1] not in line:
                continue
            lit = [label for label in TABS if self.screen.buffer[row][line.index(label)].reverse]
            return lit[0] if len(lit) == 1 else None
        return None

    def wait_for_tab(self, label: str, timeout: float = 5.0) -> None:
        self.wait_for(lambda: self.active_tab() == label, timeout, f"tab {label} to be lit")

    def click(self, text: str) -> None:
        """Left click (SGR 1006 press + release) on the first cell of `text`."""
        for row, line in enumerate(self.screen.display):
            if text in line:
                x, y = line.index(text) + 1, row + 1  # reports are 1-based
                self.send(f"\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m")
                return
        raise AssertionError(f"{text!r} is not on screen:\n{self.text()}")

    def wait_exit(self, timeout: float = 10.0) -> int:
        """Exit status, reading output meanwhile: a full pty buffer would
        otherwise block the app mid-write and it could never see the key."""
        self.wait_for(lambda: self.process.poll() is not None, timeout, "the app to exit")
        return self.process.returncode

    def local_flags(self) -> int | None:
        """The pty's lflag bits, or None once it is gone: on macOS the app is
        the session leader, and its exit revokes the terminal."""
        try:
            return termios.tcgetattr(self.slave)[3]
        except termios.error:
            return None

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.fixture
def terminal(tmp_path):
    """Factory: terminal(columns=140, rows=45) starts the app; every app is killed afterwards."""
    opened: list[Terminal] = []

    def start(columns: int = 140, rows: int = 45) -> Terminal:
        session = Terminal(tmp_path, columns, rows)
        opened.append(session)
        return session

    try:
        yield start
    finally:
        for session in opened:
            session.close()


def dismissed(session: Terminal) -> Terminal:
    session.wait_for_text("GITHUB")
    session.send(ESC)
    session.wait_for_gone("GITHUB")
    session.wait_for_text(MAP_BOX)
    return session


def test_starts_with_the_welcome_guide_and_esc_dismisses_it(terminal):
    session = terminal()
    session.wait_for_text("GITHUB")
    assert "DON'T SHOW AGAIN" in session.text()
    assert session.local_flags() & termios.ECHO == 0  # curses owns the terminal
    session.send(ESC)
    session.wait_for_gone("GITHUB")
    session.wait_for_text(MAP_BOX)
    assert session.active_tab() == "1 MAP"
    # offline with an empty cache: the app says why there is no map
    session.wait_for_text("offline mode")


def test_number_keys_switch_the_highlighted_tab(terminal):
    session = dismissed(terminal())
    for key, label in (("2", "2 TIME"), ("3", "3 WEATHER"), ("4", "4 PLANES"), ("1", "1 MAP")):
        session.send(key)
        session.wait_for_tab(label)
    session.send("2")
    session.wait_for_text("Z TZ")  # the Time tab's clock opens the timezone picker


def test_slash_on_the_planes_tab_opens_the_search_prompt(terminal):
    session = dismissed(terminal())
    session.send("4")
    session.wait_for_tab("4 PLANES")
    session.wait_for_text("SEARCH PLANE")
    session.send("/")
    session.wait_for_text("Enter apply  Esc cancel")
    session.send("ual1")
    session.wait_for_text("ual1█")
    assert session.active_tab() == "4 PLANES"  # "1" was typed, not a tab switch
    session.send(ESC)
    session.wait_for_gone("Enter apply  Esc cancel")
    session.send("1")
    session.wait_for_tab("1 MAP")


@pytest.mark.parametrize("start, then", [("1", "2"), ("2", "3"), ("3", "1")])
def test_slash_off_the_planes_tab_does_not_capture_input(terminal, start, then):
    """Issue #3: "/" used to switch to a search prompt that is only drawn on the
    Planes tab, so every key after it vanished into an invisible modal."""
    session = dismissed(terminal())
    session.send(start)
    session.wait_for_tab(TABS[int(start) - 1])
    session.send("/")
    session.send("o")
    session.send(then)
    session.wait_for_tab(TABS[int(then) - 1])
    assert "Enter apply" not in session.text()


def test_clicking_a_tab_selects_it(terminal):
    session = dismissed(terminal())
    session.click("3 WEATHER")
    session.wait_for_tab("3 WEATHER")
    session.click("4 PLANES")
    session.wait_for_tab("4 PLANES")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: clicking a tab while the plane search prompt is open leaves the hidden "
    "search mode capturing every key (issue #3, reached with the mouse)",
)
def test_clicking_a_tab_while_searching_does_not_capture_input(terminal):
    session = dismissed(terminal())
    session.send("4")
    session.wait_for_tab("4 PLANES")
    session.send("/")
    session.wait_for_text("Enter apply  Esc cancel")
    session.click("1 MAP")
    session.wait_for_tab("1 MAP")
    session.send("2")
    session.wait_for_tab("2 TIME", timeout=2.0)


def test_question_mark_reopens_the_guide(terminal):
    session = dismissed(terminal())
    session.send("?")
    session.wait_for_text("GITHUB")
    session.send("2")  # swallowed by the guide
    session.pump(0.3)
    assert session.active_tab() == "1 MAP"
    session.send(ESC)
    session.wait_for_gone("GITHUB")


def test_q_quits_cleanly_and_restores_the_terminal(terminal):
    session = dismissed(terminal())
    assert session.local_flags() & (termios.ECHO | termios.ICANON) == 0
    session.send("q")
    assert session.wait_exit() == 0
    session.pump(0.2)
    flags = session.local_flags()
    if flags is not None:  # still readable (Linux): curses put the modes back
        assert flags & termios.ECHO
        assert flags & termios.ICANON
    assert b"\x1b[?1049l" in session.raw  # left the alternate screen
    assert not session.screen.cursor.hidden
    assert b"\x1b[?1006l" in session.raw  # SGR mouse reporting switched back off


def test_the_minimum_size_works_and_smaller_says_too_small(terminal):
    from terrascope.core.config import CFG

    width = CFG["app"]["min_terminal_width"]
    height = CFG["app"]["min_terminal_height"]
    session = dismissed(terminal(width, height))
    assert "Terminal too small" not in session.text()
    session.send("3")
    session.wait_for_tab("3 WEATHER")
    session.resize(width - 1, height)
    session.wait_for_text("Terminal too small")
    assert MAP_BOX not in session.text()
    session.resize(width, height - 1)
    session.wait_for_text("Terminal too small")
    session.resize(140, 45)
    session.wait_for_gone("Terminal too small")
    session.wait_for_text(MAP_BOX)
    assert session.active_tab() == "3 WEATHER"  # state survives the resizes
    session.send("q")
    assert session.wait_exit() == 0
