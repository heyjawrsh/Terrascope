"""RainViewer catalog parsing, palette levels, raster sampling and RadarStore caching / fallback."""

from __future__ import annotations

import io
import json
import threading
import types
import urllib.error

import numpy as np
import pytest
from PIL import Image

from terrascope.core.view import View
from terrascope.layers.weather import radar

NOW = 1_760_000_000.0
CATALOG_URL = f"{radar.RAINVIEWER_URL}/public/weather-maps.json"
HOST = "https://tiles.example.test"
TILE = 4  # pixels per tile in these tests (the real one is 512)


def _catalog(*times, host=HOST, **extra) -> dict:
    past = [{"time": stamp, "path": f"/v2/radar/{stamp}"} for stamp in times]
    return {"host": host, "radar": {"past": past, **extra}}


def _png(color=(255, 238, 0, 255)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (TILE, TILE), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def frozen_time(monkeypatch):
    clock = types.SimpleNamespace(now=NOW)
    monkeypatch.setattr(radar.time, "time", lambda: clock.now)
    return clock


@pytest.fixture
def progress(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(radar, "set_live_progress", lambda name, **kw: events.append({"name": name, **kw}))
    return events


@pytest.fixture
def small_world(monkeypatch, tmp_path, progress):
    """A 2x2-tile world of TILE px tiles, a private cache dir, and threads that
    run synchronously inside start() so nothing outlives the test."""

    class InlineThread:
        def __init__(self, target, args=(), **kwargs):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(radar, "WORLD_ZOOM", 1)
    monkeypatch.setattr(radar, "TILE_SIZE", TILE)
    monkeypatch.setattr(radar, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(radar, "threading", types.SimpleNamespace(Thread=InlineThread, Lock=threading.Lock))
    return tmp_path


def _raster(value=153) -> np.ndarray:
    return np.full((2 * TILE, 2 * TILE), value, dtype=np.uint8)


# ---- palette -------------------------------------------------------------------------


def test_palette_endpoints_and_clamping():
    first, last = radar._PALETTE_STOPS[0][1].lower(), radar._PALETTE_STOPS[-1][1].lower()
    assert radar.intensity_to_color(0.0, "radar") == first
    assert radar.intensity_to_color(-5.0, "radar") == first
    assert radar.intensity_to_color(1.0, "radar") == last
    assert radar.intensity_to_color(7.0, "radar") == last
    assert radar.intensity_to_color(1.0, "satellite") == radar._CLOUD_STOPS[-1][1].lower()


def test_palette_snaps_to_colour_steps():
    colours = {radar.intensity_to_color(step / 1000, "radar") for step in range(1001)}
    assert len(colours) <= radar._RADAR_COLOR_STEPS + 1


def test_interpolate_handles_zero_width_segments_and_overshoot():
    stops = [(0.0, "#000000"), (0.5, "#ff0000"), (0.5, "#00ff00"), (0.9, "#0000ff")]
    assert radar._interpolate_palette(stops, 0.5, snap=False) == "#ff0000"
    assert radar._interpolate_palette(stops, 0.95, snap=False) == "#0000ff"  # past the last stop


def test_level_colours_and_lookup_clamp():
    assert len(radar.RADAR_LEVEL_COLORS) == radar._LEVEL_COUNT >= 2
    assert radar.radar_level_color(-1.0) == radar.RADAR_LEVEL_COLORS[0]
    assert radar.radar_level_color(0.0) == radar.RADAR_LEVEL_COLORS[0]
    assert radar.radar_level_color(0.999) == radar.RADAR_LEVEL_COLORS[-1]
    assert radar.radar_level_color(5.0) == radar.RADAR_LEVEL_COLORS[-1]


def test_fade_blends_towards_the_background(monkeypatch):
    monkeypatch.setattr(radar, "RADAR_OPACITY", 0.0)
    monkeypatch.setattr(radar, "_RADAR_BLEND_RGB", (16, 32, 48))
    assert radar._fade("#ffffff") == "#102030"
    monkeypatch.setattr(radar, "RADAR_OPACITY", 1.0)
    assert radar._fade("#abcdef") == "#abcdef"


# ---- fetch_catalog (PR #9: reject non-int times early) --------------------------------


def test_catalog_frames_sorted_oldest_first_with_host(fake_http):
    fake_http.responses[CATALOG_URL] = _catalog(300, 100, 200, host=HOST + "/")
    frames = radar.fetch_catalog("radar")
    assert [frame["time"] for frame in frames] == [100, 200, 300]
    assert all(frame["host"] == HOST and frame["nowcast"] is False for frame in frames)


def test_catalog_coerces_and_rejects_hostile_times(fake_http):
    past = [
        {"time": "1700000600", "path": "/a"},
        {"time": True, "path": "/bool"},
        {"time": None, "path": "/none"},
        {"time": "soon", "path": "/text"},
        {"time": [1], "path": "/list"},
        {"time": float("inf"), "path": "/inf"},
        {"time": float("nan"), "path": "/nan"},
        {"path": "/no-time"},
        "not-a-dict",
        None,
        {"time": 1700000000, "path": "/b"},
        {"time": 1700001200},  # no path: dropped
    ]
    fake_http.responses[CATALOG_URL] = {"host": HOST, "radar": {"past": past}}
    frames = radar.fetch_catalog("radar")
    assert [(frame["time"], frame["path"]) for frame in frames] == [(1700000000, "/b"), (1700000600, "/a")]
    assert all(type(frame["time"]) is int for frame in frames)


def test_catalog_keeps_only_the_newest_past_frames(fake_http, monkeypatch):
    monkeypatch.setattr(radar, "PAST_FRAMES", 3)
    fake_http.responses[CATALOG_URL] = _catalog(*range(0, 6000, 600))
    assert [frame["time"] for frame in radar.fetch_catalog("radar")] == [4200, 4800, 5400]


def test_catalog_history_cutoff(fake_http, monkeypatch):
    monkeypatch.setattr(radar, "HISTORY_MINUTES", 20.0)
    monkeypatch.setattr(radar, "PAST_FRAMES", 100)
    fake_http.responses[CATALOG_URL] = _catalog(0, 600, 1200, 1800, 2400)
    assert [frame["time"] for frame in radar.fetch_catalog("radar")] == [1200, 1800, 2400]


@pytest.mark.parametrize(
    "body",
    [
        urllib.error.URLError("down"),
        urllib.error.HTTPError(CATALOG_URL, 500, "oops", {}, None),
        TimeoutError(),
        b"{truncated",
        b"",
        {"radar": {"past": [{"time": 1, "path": "/a"}]}},
        {"host": "ftp://tiles", "radar": {"past": [{"time": 1, "path": "/a"}]}},
        {"host": 42, "radar": {}},
        {"host": HOST},
        {"host": HOST, "radar": {"past": []}},
        {"host": HOST, "radar": {"past": [{"time": "x", "path": "/a"}]}},
    ],
    ids=[
        "url-error", "http-500", "timeout", "truncated", "empty", "no-host", "ftp-host",
        "host-int", "no-radar", "no-frames", "only-bad-times",
    ],
)
def test_catalog_failures_raise_radar_fetch_error(fake_http, body):
    fake_http.responses[CATALOG_URL] = body
    with pytest.raises(radar.RadarFetchError):
        radar.fetch_catalog("radar")


def test_satellite_mode_is_refused_without_a_request(fake_http):
    with pytest.raises(radar.RadarFetchError):
        radar.fetch_catalog("satellite")
    assert fake_http.calls == []


def test_tile_url_shape(monkeypatch):
    monkeypatch.setattr(radar, "TILE_SIZE", 256)
    monkeypatch.setattr(radar, "WORLD_ZOOM", 3)
    frame = {"host": HOST, "path": "/v2/radar/abc"}
    assert radar.tile_url(frame, "radar", 5, 6) == f"{HOST}/v2/radar/abc/256/3/5/6/2/1_0.png"


# ---- rgba_to_intensity / fetch_frame_raster -----------------------------------------


def test_rgba_to_intensity_maps_palette_colours_monotonically():
    colours = [(130, 123, 105), (0, 163, 224), (255, 238, 0), (193, 0, 0), (255, 255, 255)]
    rgba = np.array([[(*colour, 255) for colour in colours] + [(255, 255, 255, 0)]], dtype=np.uint8)
    values = radar.rgba_to_intensity(rgba)
    assert values.shape == (1, 6)
    visible = values[0, :5].tolist()
    assert visible == sorted(visible) and visible[0] >= 1 and visible[-1] == 255
    assert values[0, 5] == 0  # transparent = no data


def test_rgba_to_intensity_handles_more_pixels_than_one_chunk():
    rgba = np.zeros((130, 130, 4), dtype=np.uint8)
    rgba[-1, -1] = (255, 238, 0, 255)
    values = radar.rgba_to_intensity(rgba)
    assert values[-1, -1] == 153 and values.sum() == 153


def test_fetch_frame_raster_stitches_tiles_and_tolerates_holes(fake_http, small_world):
    frame = {"host": HOST, "path": "/p", "time": 1}
    fake_http.responses[radar.tile_url(frame, "radar", 0, 0)] = _png()
    fake_http.responses[radar.tile_url(frame, "radar", 1, 1)] = b"not a png"
    raster = radar.fetch_frame_raster(frame, "radar")
    assert raster.shape == (2 * TILE, 2 * TILE) and raster.dtype == np.uint8
    assert (raster[:TILE, :TILE] == 153).all()
    assert (raster[TILE:, :] == 0).all() and (raster[:, TILE:] == 0).all()
    assert len(fake_http.calls) == 4


def test_fetch_frame_raster_fails_when_no_tile_loads(fake_http, small_world):
    with pytest.raises(radar.RadarFetchError):
        radar.fetch_frame_raster({"host": HOST, "path": "/p", "time": 1}, "radar")


# ---- sample_raster --------------------------------------------------------------------


def test_sample_raster_shape_is_two_half_cells_per_row():
    view = View(0.0, 0.0, 1.0, 30, 10)
    values = radar.sample_raster(np.full((64, 64), 100, dtype=np.uint8), view)
    assert values.shape == (20, 30)
    assert np.allclose(values, 100.0)


def test_sample_raster_follows_longitude_and_wraps_the_antimeridian():
    raster = np.zeros((64, 64), dtype=np.uint8)
    raster[:, :32] = 200  # western hemisphere
    centred = radar.sample_raster(raster, View(0.0, 0.0, 4.0, 20, 6))
    assert centred[:, 0].min() > 150 and centred[:, -1].max() < 50
    wrapped = radar.sample_raster(raster, View(180.0, 0.0, 4.0, 20, 6))
    assert wrapped[:, 0].max() < 50 and wrapped[:, -1].min() > 150


def test_sample_raster_north_is_up_and_poles_are_blank():
    raster = np.zeros((64, 64), dtype=np.uint8)
    raster[:32, :] = 200  # northern hemisphere in Web Mercator
    values = radar.sample_raster(raster, View(0.0, 0.0, 4.0, 20, 6))
    assert values[0].min() > 150 and values[-1].max() < 50
    polar = radar.sample_raster(np.full((64, 64), 200, dtype=np.uint8), View(0.0, 85.0, 1.0, 20, 10))
    assert polar[0].max() == 0.0  # beyond the Mercator limit there is no data


# ---- RadarStore: on-disk cache ------------------------------------------------------------


def _write_manifest(directory, frames, version=2, mode="radar"):
    (directory / f"radar-{mode}-cache.json").write_text(
        json.dumps({"version": version, "frames": frames}), encoding="utf-8"
    )


def test_store_restores_cached_frames(small_world):
    np.save(small_world / "radar-radar-200.npy", _raster(9))
    np.save(small_world / "radar-radar-100.npy", _raster(7))
    _write_manifest(small_world, [{"time": 200}, {"time": "100", "nowcast": 1}])
    store = radar.RadarStore("radar")
    assert [frame["time"] for frame in store.catalog_frames()] == [100, 200]
    assert store.catalog_frames()[0]["nowcast"] is True
    assert store.cached_fallback() is True
    assert [int(raster[0, 0]) for _frame, raster in store.loaded_frames()] == [7, 9]


@pytest.mark.parametrize(
    "manifest",
    ["{bad json", '"text"', '{"version": 1, "frames": []}', '{"version": 2, "frames": {}}', "[]"],
    ids=["truncated", "string", "old-version", "frames-dict", "list"],
)
def test_store_ignores_bad_manifests(small_world, manifest):
    np.save(small_world / "radar-radar-100.npy", _raster())
    (small_world / "radar-radar-cache.json").write_text(manifest, encoding="utf-8")
    store = radar.RadarStore("radar")
    assert store.loaded_frames() == [] and store.cached_fallback() is False


def test_store_skips_unusable_cached_rasters(small_world):
    np.save(small_world / "radar-radar-1.npy", np.zeros((3, 3), dtype=np.uint8))  # wrong shape
    np.save(small_world / "radar-radar-2.npy", _raster().astype(np.float32))  # wrong dtype
    np.save(small_world / "radar-radar-3.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    (small_world / "radar-radar-4.npy").write_bytes(b"\x93NUMPY truncated")
    np.save(small_world / "radar-radar-5.npy", _raster())
    _write_manifest(
        small_world,
        [{"time": 1}, {"time": 2}, {"time": 3}, {"time": 4}, {"time": 5}, {"time": 6},
         {"time": "x"}, {"time": 1e400}, {"nope": 1}, "junk"],
    )
    store = radar.RadarStore("radar")
    assert [frame["time"] for frame, _raster in store.loaded_frames()] == [5]


def test_store_save_cache_round_trips_and_clears_when_empty(small_world):
    store = radar.RadarStore("radar")
    store.install_catalog([{"time": 10, "path": "/a", "nowcast": False}, {"time": 20, "path": "/b", "nowcast": True}])
    manifest = small_world / "radar-radar-cache.json"
    assert not manifest.exists()  # nothing downloaded yet
    store.install_raster({"time": 20}, _raster())
    assert json.loads(manifest.read_text()) == {"version": 2, "frames": [{"time": 20, "nowcast": True}]}
    assert (small_world / "radar-radar-20.npy").exists()
    store.install_catalog([{"time": 30, "path": "/c", "nowcast": False}])
    assert not manifest.exists()
    assert not (small_world / "radar-radar-20.npy").exists()


def test_prune_keeps_wanted_rasters_and_other_modes(small_world):
    for name in ("radar-radar-1.npy", "radar-radar-2.npy", "radar-radar-latest.npy", "radar-satellite-1.npy"):
        (small_world / name).write_bytes(b"")
    radar.RadarStore("radar")._prune_cache({2})
    assert sorted(path.name for path in small_world.iterdir()) == [
        "radar-radar-2.npy", "radar-radar-latest.npy", "radar-satellite-1.npy",
    ]


def test_install_raster_clears_failure_and_notifies(small_world, frozen_time):
    calls = []
    store = radar.RadarStore("radar", on_update=lambda: calls.append(1))
    store.mark_failed("boom")
    assert store.failed() is True
    store.install_catalog([{"time": 5, "path": "/a", "nowcast": False}])
    store.install_raster({"time": 5}, _raster())
    assert store.failed() is False and store.cached_fallback() is False
    assert calls == [1]


def test_mark_failed_falls_back_to_cache_only_when_rasters_exist(small_world, frozen_time):
    store = radar.RadarStore("radar")
    store.mark_failed("down")
    assert store.failed() is True and store.cached_fallback() is False
    assert store._last_error == "down" and store._checked_at == NOW
    store._rasters[1] = _raster()
    store.mark_failed("down")
    assert store.cached_fallback() is True


# ---- RadarStore: refresh scheduling and fallback -------------------------------------------


def test_ensure_fresh_respects_refresh_and_retry_intervals(small_world, frozen_time, monkeypatch):
    runs = []
    store = radar.RadarStore("radar")
    monkeypatch.setattr(store, "_refresh", lambda: runs.append(1))
    store._checked_at = NOW - radar.CATALOG_REFRESH_SECONDS + 1
    store.ensure_fresh()
    assert runs == []
    frozen_time.now += 1
    store.ensure_fresh()
    assert runs == [1]
    store._busy = False
    store._failed = True
    store._checked_at = frozen_time.now - radar.CATALOG_RETRY_SECONDS
    store.ensure_fresh()
    assert runs == [1, 1]


def test_ensure_fresh_and_retry_do_nothing_while_busy(small_world, frozen_time, monkeypatch, progress):
    runs = []
    store = radar.RadarStore("radar")
    monkeypatch.setattr(store, "_refresh", lambda: runs.append(1))
    store._busy = True
    store.ensure_fresh()
    store.retry()
    assert runs == [] and progress == []
    assert store.loading() is True


def test_retry_clears_backoff_and_refetches(small_world, frozen_time, monkeypatch, progress):
    runs = []
    store = radar.RadarStore("radar")
    monkeypatch.setattr(store, "_refresh", lambda: runs.append(1))
    store.mark_failed("down")
    store.retry()
    assert runs == [1]
    assert store.failed() is False and store._last_error == ""
    assert progress[0] == {"name": "radar", "state": "downloading"}


def test_refresh_downloads_every_missing_frame(fake_http, small_world, frozen_time, progress):
    updates = []
    fake_http.responses[CATALOG_URL] = _catalog(100, 200)
    fake_http.responses[HOST] = _png()
    store = radar.RadarStore("radar", on_update=lambda: updates.append(1))
    store.ensure_fresh()
    assert store.loading() is False
    assert [frame["time"] for frame, _raster in store.loaded_frames()] == [100, 200]
    assert store.failed() is False and store.cached_fallback() is False
    assert store._checked_at == NOW
    assert progress[-1]["state"] == "ready" and progress[-1]["downloaded"] == 2
    assert len(updates) == 3  # one per frame plus the final one
    tile_requests = [url for url in fake_http.calls if url.startswith(HOST)]
    assert len(tile_requests) == 2 * 4
    assert tile_requests[0].startswith(f"{HOST}/v2/radar/200/")  # newest frame first


def test_refresh_reuses_rasters_already_loaded(fake_http, small_world, frozen_time):
    fake_http.responses[CATALOG_URL] = _catalog(100, 200)
    fake_http.responses[HOST] = _png()
    store = radar.RadarStore("radar")
    store._rasters = {100: _raster(1), 50: _raster(2)}
    store._refresh()
    assert sorted(store._rasters) == [100, 200]
    assert int(store._rasters[100][0, 0]) == 1  # not refetched
    assert not any("/v2/radar/100/" in url for url in fake_http.calls)


def test_refresh_with_partial_failures_falls_back_to_cache(fake_http, small_world, frozen_time, progress):
    fake_http.responses[CATALOG_URL] = _catalog(100, 200)
    fake_http.responses[f"{HOST}/v2/radar/200/"] = _png()  # frame 100 has no tiles
    store = radar.RadarStore("radar")
    store._refresh()
    assert [frame["time"] for frame, _raster in store.loaded_frames()] == [200]
    assert store.failed() is True and store.cached_fallback() is True
    assert progress[-1]["state"] == "failed"
    assert progress[-1]["error"] == "Some radar frames could not be downloaded"


def test_refresh_catalog_failure_keeps_cached_frames(fake_http, small_world, frozen_time, progress):
    np.save(small_world / "radar-radar-100.npy", _raster())
    _write_manifest(small_world, [{"time": 100}])
    store = radar.RadarStore("radar")
    store._refresh()  # catalog URL unregistered: network error
    assert store.failed() is True and store.cached_fallback() is True
    assert [frame["time"] for frame, _raster in store.loaded_frames()] == [100]
    assert store.loading() is False
    assert progress[-1]["state"] == "failed"


@pytest.mark.parametrize(
    "body",
    [[1, 2], None, {"host": HOST, "radar": [1]}, {"host": HOST, "radar": {"past": 5}}],
    ids=["list", "null", "radar-list", "past-int"],
)
def test_refresh_survives_malformed_catalog_shapes(fake_http, small_world, frozen_time, body):
    fake_http.responses[CATALOG_URL] = body
    store = radar.RadarStore("radar")
    store._refresh()
    assert store.failed() is True and store.loading() is False
    assert store._last_error


def test_refresh_never_downloads_satellite(fake_http, small_world, frozen_time):
    store = radar.RadarStore("satellite")
    store._refresh()
    assert store.failed() is True
    assert fake_http.calls == []
