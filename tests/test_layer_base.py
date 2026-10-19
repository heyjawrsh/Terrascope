"""The Layer base class: toggling, hit-testing, background refresh and backoff, defaults."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from terrascope.core import layer as layer_module
from terrascope.core.layer import (
    CLICK_RADIUS,
    DEFAULT_INFO_TITLE,
    RETRY_BASE_SECONDS,
    RETRY_MAX_SECONDS,
    InfoImage,
    Layer,
    LayerRender,
    Marker,
    PanelBox,
    SidebarLine,
    SliderSpec,
)
from terrascope.core.view import View


class RecordingLayer(Layer):
    name = "recording"
    toggle_key = "r"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.due = False
        self.refresh_result = True
        self.refresh_error: Exception | None = None

    def reload(self) -> None:
        self.calls.append("reload")

    def clear(self) -> None:
        self.calls.append("clear")

    def is_due(self) -> bool:
        return self.due

    def refresh_if_due(self, force: bool = False) -> bool:
        self.calls.append("refresh")
        if self.refresh_error is not None:
            raise self.refresh_error
        return self.refresh_result

    def build(self, view: View) -> LayerRender:
        self.calls.append("build")
        return LayerRender(None, "#ffffff", [Marker(1, 2, "#ff0000", item="m")])


class SyncThread:
    """Stands in for threading.Thread: runs the target when started."""

    created: list[SyncThread] = []

    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon
        SyncThread.created.append(self)

    def start(self):
        self.target()


@pytest.fixture
def sync_threads(monkeypatch):
    SyncThread.created = []
    monkeypatch.setattr(layer_module.threading, "Thread", SyncThread)
    return SyncThread.created


@pytest.fixture
def clock(monkeypatch):
    """Freeze layer.py's view of time.monotonic(); set clock.now to move it."""
    state = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(layer_module, "time", SimpleNamespace(monotonic=lambda: state.now))
    return state


# ---- toggling and keys ---------------------------------------------------------


def test_toggle_off_clears_and_toggle_on_reloads():
    layer = RecordingLayer()
    layer.dirty = False
    layer.toggle()
    assert layer.enabled is False and layer.calls == ["clear"] and layer.dirty
    layer.dirty = False
    layer.toggle()
    assert layer.enabled is True and layer.calls == ["clear", "reload"] and layer.dirty


@pytest.mark.parametrize(
    "key, handled",
    [(ord("r"), True), (ord("R"), True), (ord("x"), False), (300, False), (-1, False)],
)
def test_handle_key_toggles_on_the_toggle_key_in_either_case(key, handled):
    layer = RecordingLayer()
    assert layer.handle_key(key) is handled
    assert layer.enabled is (not handled)


def test_handle_key_without_a_toggle_key_is_ignored():
    layer = RecordingLayer()
    layer.toggle_key = ""
    assert layer.handle_key(ord("r")) is False
    assert layer.enabled is True


def test_startup_loads_the_cache():
    layer = RecordingLayer()
    layer.startup()
    assert layer.calls == ["reload"]


# ---- rendering and hit-testing ---------------------------------------------------


def test_render_builds_clears_dirty_and_remembers_markers():
    layer = RecordingLayer()
    view = View(0.0, 13.0, 1, 10, 5)
    result = layer.render_with_land(view, view.new_image())
    assert layer.calls == ["build"]
    assert layer.dirty is False
    assert layer.last_markers == result.markers
    assert result.markers[0].item == "m"


def test_base_build_is_abstract():
    with pytest.raises(NotImplementedError):
        Layer().render(View(0.0, 13.0, 1, 10, 5))


def test_hit_test_all_returns_hits_within_the_click_radius_nearest_first():
    layer = Layer()
    layer.last_markers = [
        Marker(12, 10, "#fff", item="two-away"),
        Marker(11, 11, "#fff", item="diagonal"),
        Marker(10, 10, "#fff", item="on-it"),
        Marker(11, 10, "#fff", item="next-door"),
    ]
    hits = layer.hit_test_all(10, 10)
    assert CLICK_RADIUS == 1.5
    assert [item for _, item in hits] == ["on-it", "next-door", "diagonal"]
    assert [distance for distance, _ in hits] == pytest.approx([0.0, 1.0, 2**0.5])


def test_hit_test_all_with_a_custom_radius():
    layer = Layer()
    layer.last_markers = [Marker(10, 10, "#fff", item="a"), Marker(13, 14, "#fff", item="b")]
    assert [item for _, item in layer.hit_test_all(10, 10, radius=0)] == ["a"]
    assert [item for _, item in layer.hit_test_all(10, 10, radius=5)] == ["a", "b"]
    assert layer.hit_test_all(50, 50) == []


def test_select_stores_the_item_and_marks_dirty():
    layer = Layer()
    layer.dirty = False
    layer.select("item")
    assert layer.selected == "item" and layer.dirty


# ---- background refresh -----------------------------------------------------------


def test_refresh_async_runs_the_worker_on_a_daemon_thread(sync_threads):
    layer = RecordingLayer()
    layer.refresh_async()
    assert len(sync_threads) == 1 and sync_threads[0].daemon is True
    assert layer.calls == ["refresh"]
    assert layer.refresh_completed is True
    assert layer.refresh_in_progress is False


def test_refresh_async_is_skipped_while_a_refresh_is_running(sync_threads):
    layer = RecordingLayer()
    layer.refresh_in_progress = True
    layer.refresh_async()
    assert sync_threads == [] and layer.calls == []


def test_refresh_with_nothing_new_does_not_flag_completion(sync_threads):
    layer = RecordingLayer()
    layer.refresh_result = False
    layer.refresh_async()
    assert layer.refresh_completed is False


def test_tick_applies_a_completed_refresh_on_the_ui_thread():
    layer = RecordingLayer()
    layer.refresh_completed = True
    layer.tick()
    assert layer.calls == ["reload"]
    assert layer.refresh_completed is False
    layer.tick()
    assert layer.calls == ["reload"]


@pytest.mark.parametrize(
    "failures, delay",
    [
        (1, RETRY_BASE_SECONDS),
        (2, RETRY_BASE_SECONDS * 2),
        (3, RETRY_BASE_SECONDS * 4),
        (5, RETRY_BASE_SECONDS * 16),
        (6, RETRY_MAX_SECONDS),
        (20, RETRY_MAX_SECONDS),
    ],
)
def test_failed_refresh_backs_off_exponentially_up_to_the_cap(clock, failures, delay):
    assert (RETRY_BASE_SECONDS, RETRY_MAX_SECONDS) == (15, 300)
    layer = RecordingLayer()
    layer.refresh_error = RuntimeError("provider down")
    layer._failures = failures - 1
    layer.refresh_in_progress = True
    layer._refresh_worker()  # must swallow the error
    assert layer._failures == failures
    assert layer._retry_at == pytest.approx(clock.now + delay)
    assert layer.refresh_in_progress is False
    assert layer.refresh_completed is False


def test_successful_refresh_resets_the_backoff():
    layer = RecordingLayer()
    layer._failures = 4
    layer._retry_at = 9999.0
    layer._refresh_worker()
    assert (layer._failures, layer._retry_at) == (0, 0.0)


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setattr(layer_module, "FETCH_NEW_DATA_FROM_API", True)


def spy_refresh(layer: Layer) -> list[bool]:
    started: list[bool] = []
    layer.refresh_async = lambda: started.append(True)
    return started


def test_tick_starts_a_refresh_when_online_due_and_not_backing_off(online, clock):
    layer = RecordingLayer()
    layer.due = True
    started = spy_refresh(layer)
    layer.tick()
    assert started == [True]


@pytest.mark.parametrize(
    "setup",
    [
        lambda layer, clock: setattr(layer, "due", False),
        lambda layer, clock: setattr(layer, "enabled", False),
        lambda layer, clock: setattr(layer, "_ordered_startup", True),
        lambda layer, clock: setattr(layer, "_retry_at", clock.now + 1),
    ],
    ids=["not-due", "disabled", "ordered-startup", "backing-off"],
)
def test_tick_holds_off_refreshing(online, clock, setup):
    layer = RecordingLayer()
    layer.due = True
    setup(layer, clock)
    started = spy_refresh(layer)
    layer.tick()
    assert started == []


def test_tick_never_refreshes_in_offline_mode(clock):
    assert layer_module.FETCH_NEW_DATA_FROM_API is False
    layer = RecordingLayer()
    layer.due = True
    started = spy_refresh(layer)
    layer.tick()
    assert started == []


def test_tick_retries_once_the_backoff_has_passed(online, clock):
    layer = RecordingLayer()
    layer.due = True
    layer._retry_at = clock.now + 15
    started = spy_refresh(layer)
    layer.tick()
    clock.now += 15
    layer.tick()
    assert started == [True]


# ---- defaults and data classes ------------------------------------------------------


def test_base_layer_defaults_are_inert():
    layer = Layer()
    assert layer.info_title == DEFAULT_INFO_TITLE == "INFO"
    assert layer.enabled and layer.dirty and layer.selected is None
    assert layer.panel_layout is None
    assert layer.prefetch() is None
    assert layer.is_due() is False
    assert layer.refresh_if_due(force=True) is False
    assert layer.click_at(1, 1) is False
    assert layer.info_rows() == []
    assert layer.info_image(5, 20) is None
    assert layer.live_rows("x") == []
    assert layer.rich_title("Box") == "Box"
    assert layer.rich_title(None) == ""
    assert layer.rich_lines("x", 10, 5, vertical=True, scroll=2) == []
    assert layer.border_buttons() == []
    assert layer.slider() is None
    assert layer.loading_messages() == []
    assert layer.filter_options() == []
    assert layer.legend_lines() == []
    # Hooks that only exist to be overridden accept their arguments quietly.
    layer.reload()
    layer.clear()
    layer.press_border_button(0)
    layer.press_legend("action")
    layer.set_slider(3)
    layer.apply_search("text")
    layer.apply_filter(None)


def test_marker_and_render_defaults():
    marker = Marker(3, 4, "#123456")
    assert (marker.glyph, marker.label, marker.item, marker.bg_color) == (None, None, None, None)
    assert marker.selected is False and marker.dim is False
    first, second = LayerRender(None, "#fff"), LayerRender(None, "#fff")
    first.markers.append(marker)
    assert second.markers == []  # no shared mutable default
    assert first.bg_cells is None


@pytest.mark.parametrize(
    "instance, field",
    [
        (PanelBox("info", 0.5), "ratio"),
        (SliderSpec("TIME", 10, 3), "index"),
        (SidebarLine("Legend"), "text"),
        (InfoImage(4, ()), "width"),
    ],
)
def test_layout_records_are_frozen(instance, field):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, 1)


def test_frozen_record_defaults():
    assert PanelBox("search", 0.3).title is None and PanelBox("search", 0.3).hint is None
    slider = SliderSpec("TIME", 10, 3)
    assert (slider.caption, slider.marker, slider.wide, slider.status) == ("", None, False, "")
    assert slider.loaded_steps is None and slider.disabled is False
    line = SidebarLine()
    assert (line.text, line.swatches, line.legend_items, line.choices) == ("", (), (), ())
    assert line.swatch_glyph == "■" and line.bold is False
    assert line.choice_index == 0 and line.action is None
