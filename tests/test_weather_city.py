"""Open-Meteo fetch, value coercion, forecast parsing and the per-city WeatherStationCache."""

from __future__ import annotations

import json
import threading
import types
import urllib.error

import pytest

from terrascope.layers.weather import city_weather as cw
from terrascope.layers.weather.config import CITY_CONFIG

NOW = 1_760_000_000.0


@pytest.fixture
def frozen_time(monkeypatch):
    clock = types.SimpleNamespace(now=NOW)
    monkeypatch.setattr(cw.time, "time", lambda: clock.now)
    return clock


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "cache" / "weather_cache.json"
    monkeypatch.setattr(cw, "WEATHER_CACHE_FILE", path)
    monkeypatch.setattr(cw, "CACHE_DIR", path.parent)
    return path


@pytest.fixture
def threads(monkeypatch):
    """Capture threads ensure_fresh would start instead of running them."""
    started: list = []

    class RecordingThread:
        def __init__(self, target, args=(), daemon=None, **kwargs):
            self.target, self.args = target, args

        def start(self):
            started.append(self)

        def run(self):
            self.target(*self.args)

    monkeypatch.setattr(cw, "threading", types.SimpleNamespace(Thread=RecordingThread, Lock=threading.Lock))
    return started


def _payload(temperature=12.3, **current) -> dict:
    return {
        "current": {
            "temperature_2m": temperature,
            "wind_speed_10m": 14.0,
            "wind_direction_10m": 270,
            "weather_code": 3,
            "time": "2026-10-19T12:00",
            **current,
        },
        "daily": {
            "time": ["2026-10-19", "2026-10-20"],
            "weather_code": [3, 61],
            "temperature_2m_max": [15.0, 13.0],
            "temperature_2m_min": [7.0, 6.0],
        },
        "hourly": {
            "time": ["2026-10-19T00:00", "2026-10-19T01:00", "2026-10-20T00:00"],
            "temperature_2m": [8.0, 7.5, 6.5],
        },
    }


# ---- describe / coerce / colour ---------------------------------------------------


@pytest.mark.parametrize(
    "code, text",
    [(0, "Clear sky"), ("63", "Rain"), (99.0, "Thunderstorm, heavy hail"), (4, "Unknown"), (None, "Unknown"), ("x", "Unknown")],
)
def test_describe_weather_code(code, text):
    assert cw.describe_weather_code(code) == text


@pytest.mark.parametrize(
    "value, expected",
    [
        (5, 5.0), (-3.5, -3.5), ("7.25", 7.25), (0, 0.0),
        (True, None), (False, None), (None, None), ("warm", None), ([], None),
        (float("nan"), None), (float("inf"), None), ("-inf", None),
    ],
)
def test_as_finite(value, expected):
    assert cw._as_finite(value) == expected


def test_temperature_colour_endpoints_and_clamping():
    cold = CITY_CONFIG["temperature_cold_color"].lower()
    hot = CITY_CONFIG["temperature_hot_color"].lower()
    assert cw.temperature_to_color(0, 0, 10) == cold
    assert cw.temperature_to_color(10, 0, 10) == hot
    assert cw.temperature_to_color(-40, 0, 10) == cold
    assert cw.temperature_to_color(55, 0, 10) == hot


def test_temperature_colour_with_zero_or_inverted_span_is_the_midpoint():
    middle = cw.temperature_to_color(5, 0, 10)
    assert cw.temperature_to_color(3, 3, 3) == middle
    assert cw.temperature_to_color(3, 10, 0) == middle


def test_temperature_colour_snaps_to_colour_steps():
    colours = {cw.temperature_to_color(step / 10, 0, 100) for step in range(1001)}
    assert len(colours) == cw._COLOR_STEPS + 1


@pytest.mark.parametrize("value", [None, "warm", True, float("nan"), float("-inf"), {}])
def test_temperature_colour_is_none_for_unusable_readings(value):
    assert cw.temperature_to_color(value, 0, 10) is None


def test_temperature_colour_accepts_legacy_numeric_strings():
    assert cw.temperature_to_color("10", 0, 10) == cw.temperature_to_color(10.0, 0, 10)


# ---- fetch_weather_snapshot --------------------------------------------------------


def test_fetch_snapshot_normalises_current_fields(fake_http):
    fake_http.responses[cw.WEATHER_API_URL] = _payload()
    snapshot = cw.fetch_weather_snapshot(35.6895, 139.6917)
    assert snapshot["current"] == {
        "temperature": 12.3, "windspeed": 14.0, "winddirection": 270, "weathercode": 3,
        "time": "2026-10-19T12:00",
    }
    assert snapshot["daily"]["time"] == ["2026-10-19", "2026-10-20"]
    (url,) = fake_http.calls
    assert "latitude=35.6895&longitude=139.6917" in url
    assert f"forecast_days={cw.FORECAST_DAYS_AHEAD + 1}" in url


def test_fetch_snapshot_accepts_legacy_current_weather(fake_http):
    fake_http.responses[cw.WEATHER_API_URL] = {
        "current_weather": {"temperature": -2, "windspeed": 5, "winddirection": 90, "weathercode": 71},
        "daily": "not a dict",
        "hourly": None,
    }
    snapshot = cw.fetch_weather_snapshot(0, 0)
    assert snapshot["current"]["temperature"] == -2
    assert snapshot["current"]["weathercode"] == 71
    assert snapshot["current"]["time"] is None
    assert snapshot["daily"] == {} and snapshot["hourly"] == {}


@pytest.mark.parametrize(
    "body",
    [
        urllib.error.URLError("down"),
        urllib.error.HTTPError(cw.WEATHER_API_URL, 429, "slow down", {}, None),
        TimeoutError(),
        b'{"current": {"temperature_2m": 1',
        b"",
        {"daily": {}},
        {"current": "sunny"},
        {"current": None, "current_weather": [1, 2]},
    ],
    ids=["url-error", "http-429", "timeout", "truncated", "empty", "no-current", "current-string", "both-bad"],
)
def test_fetch_snapshot_wraps_failures(fake_http, body):
    fake_http.responses[cw.WEATHER_API_URL] = body
    with pytest.raises(cw.WeatherFetchError):
        cw.fetch_weather_snapshot(1, 2)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a non-object JSON body (list/null) or non-UTF-8 bytes escape as AttributeError/UnicodeDecodeError, not WeatherFetchError",
)
@pytest.mark.parametrize("body", [[], None, b"\x80\x81"], ids=["list", "null", "non-utf8"])
def test_fetch_snapshot_wraps_non_object_bodies(fake_http, body):
    fake_http.responses[cw.WEATHER_API_URL] = body
    with pytest.raises(cw.WeatherFetchError):
        cw.fetch_weather_snapshot(1, 2)


# ---- parse_forecast ----------------------------------------------------------------


def test_parse_forecast_groups_hours_by_day_and_accepts_old_field_names():
    daily = {
        "time": ["2026-10-19", "2026-10-20"],
        "weathercode": [1, 2],
        "temperature_2m_max": [20.0],
        "windspeed_10m_max": [10, 12],
        "wind_gusts_10m_max": [30, 31],
        "uv_index_max": "high",  # not a list: treated as missing
    }
    hourly = {
        "time": ["2026-10-19T05:00", "2026-10-19T06:00", "2026-10-20T00:00", "bad", None, "2026-10-21T00:00"],
        "temperature_2m": [1.0, 2.0, 3.0, 4.0],
        "weather_code": [0, 1, 2, 3, 4, 5],
        "precipitation_probability": [],
        "wind_speed_10m": [5, 6, 7],
    }
    days = cw.parse_forecast(daily, hourly)
    assert [day["date"] for day in days] == ["2026-10-19", "2026-10-20"]
    first, second = days
    assert first["code"] == 1 and second["code"] == 2
    assert first["tmax"] == 20.0 and second["tmax"] is None
    assert first["wind"] == 10 and first["gust"] == 30
    assert first["uv"] is None and first["tmin"] is None
    assert first["hours"] == [[5, 1.0, 0, None, 5], [6, 2.0, 1, None, 6]]
    assert second["hours"] == [[0, 3.0, 2, None, 7]]


def test_parse_forecast_caps_days_and_handles_empty_blocks():
    many = {"time": [f"2026-11-{day:02d}" for day in range(1, 20)]}
    assert len(cw.parse_forecast(many, {})) == cw.FORECAST_DAYS_AHEAD + 1
    assert cw.parse_forecast({}, {}) == []
    assert cw.parse_forecast({"time": None}, {"time": "2026"}) == []


# ---- WeatherStationCache: persistence ---------------------------------------------


def test_cache_starts_empty_without_a_file(cache_file):
    cache = cw.WeatherStationCache()
    assert cache.get("x") is None
    assert cache.temperature_range() is None
    assert (cache.loading_count(), cache.failed_count(), cache.first_error()) == (0, 0, "")


@pytest.mark.parametrize("content", ["", "{", "[1, 2]", "42", "￾"])
def test_cache_ignores_unreadable_files(cache_file, content):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(content, encoding="utf-8")
    assert cw.WeatherStationCache()._entries == {}


def test_cache_load_keeps_only_entries_with_a_finite_fetch_time(cache_file):
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        '{"good": {"fetched_at": "100", "temperature": 3},'
        ' "nan": {"fetched_at": NaN}, "inf": {"fetched_at": Infinity},'
        ' "none": {"fetched_at": null}, "missing": {"temperature": 3},'
        ' "text": {"fetched_at": "yesterday"}, "list": [1], "number": 5}',
        encoding="utf-8",
    )
    entries = cw.WeatherStationCache()._entries
    assert entries == {"good": {"fetched_at": 100.0, "temperature": 3}}


def test_cache_save_writes_atomically_and_round_trips(cache_file):
    cw.WeatherStationCache._save_to_disk({"a": {"fetched_at": 1.0, "temperature": 2.0}})
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {"a": {"fetched_at": 1.0, "temperature": 2.0}}
    assert not cache_file.with_suffix(".json.tmp").exists()
    assert cw.WeatherStationCache().get("a") == {"fetched_at": 1.0, "temperature": 2.0}


def test_cache_save_swallows_os_errors(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(cw, "CACHE_DIR", blocker / "sub")  # parent is a file: mkdir fails
    monkeypatch.setattr(cw, "WEATHER_CACHE_FILE", blocker / "sub" / "w.json")
    cw.WeatherStationCache._save_to_disk({"a": {"fetched_at": 1.0}})


def test_get_returns_a_copy(cache_file):
    cache = cw.WeatherStationCache()
    cache._entries["k"] = {"fetched_at": 1.0, "temperature": 5}
    cache.get("k")["temperature"] = 99
    assert cache.get("k")["temperature"] == 5


# ---- temperature_range / counts ----------------------------------------------------


def test_temperature_range_skips_legacy_and_bad_values(cache_file):
    cache = cw.WeatherStationCache()
    cache._entries = {
        "a": {"fetched_at": 1, "temperature": 21.5},
        "b": {"fetched_at": 1, "temperature": "-4"},  # legacy string, still numeric
        "c": {"fetched_at": 1, "temperature": "hot"},
        "d": {"fetched_at": 1, "temperature": True},
        "e": {"fetched_at": 1, "temperature": None},
        "f": {"fetched_at": 1},
        "g": {"fetched_at": 1, "temperature": float("nan")},
    }
    assert cache.temperature_range() == (-4.0, 21.5)


def test_temperature_range_none_when_nothing_usable(cache_file):
    cache = cw.WeatherStationCache()
    cache._entries = {"a": {"fetched_at": 1, "temperature": "n/a"}}
    assert cache.temperature_range() is None


def test_failed_count_and_first_error(cache_file):
    cache = cw.WeatherStationCache()
    cache._entries = {
        "a": {"fetched_at": 1, "error": ""},
        "b": {"fetched_at": 1, "error": "timeout"},
        "c": {"fetched_at": 1, "error": "429"},
    }
    assert cache.failed_count() == 2
    assert cache.first_error() == "timeout"


# ---- freshness and retry backoff --------------------------------------------------


def _entry(age, error="", schema=cw.WEATHER_SCHEMA) -> dict:
    return {"fetched_at": NOW - age, "error": error, "schema": schema, "temperature": 1.0}


@pytest.mark.parametrize(
    "entry, stale",
    [
        (None, True),
        (_entry(0), False),
        (_entry(cw.WEATHER_REFRESH_INTERVAL_SECONDS - 1), False),
        (_entry(cw.WEATHER_REFRESH_INTERVAL_SECONDS), True),
        (_entry(0, schema=cw.WEATHER_SCHEMA - 1), True),
        (_entry(0, schema=None), True),
        (_entry(cw.WEATHER_RETRY_INTERVAL_SECONDS - 1, error="boom"), False),
        (_entry(cw.WEATHER_RETRY_INTERVAL_SECONDS, error="boom"), True),
        # an errored entry on an old schema waits out the retry interval instead
        # of being refetched on every frame
        (_entry(0, error="boom", schema=1), False),
    ],
    ids=[
        "missing", "fresh", "just-fresh", "expired", "old-schema", "no-schema",
        "retry-wait", "retry-due", "errored-old-schema",
    ],
)
def test_is_stale(cache_file, frozen_time, entry, stale):
    cache = cw.WeatherStationCache()
    if entry is not None:
        cache._entries["k"] = entry
    assert cache._is_stale("k") is stale


def test_retry_interval_is_shorter_than_refresh():
    assert cw.WEATHER_RETRY_INTERVAL_SECONDS < cw.WEATHER_REFRESH_INTERVAL_SECONDS


def test_has_pending_retry(cache_file, frozen_time):
    cache = cw.WeatherStationCache()
    cache._entries = {"ok": _entry(10_000), "waiting": _entry(1, error="x")}
    assert cache.has_pending_retry() is False
    cache._entries["due"] = _entry(cw.WEATHER_RETRY_INTERVAL_SECONDS, error="x")
    assert cache.has_pending_retry() is True
    cache._in_progress.add("due")
    assert cache.has_pending_retry() is False


# ---- ensure_fresh: concurrency limits ------------------------------------------------


def test_ensure_fresh_starts_one_worker_per_stale_city(cache_file, frozen_time, threads):
    cache = cw.WeatherStationCache()
    cache._entries["fresh"] = _entry(0)
    cache.ensure_fresh("new", 1.0, 2.0)
    cache.ensure_fresh("new", 1.0, 2.0)  # already in progress
    cache.ensure_fresh("fresh", 1.0, 2.0)  # not stale
    assert [thread.args for thread in threads] == [("new", 1.0, 2.0)]
    assert cache.loading_count() == 1


def test_ensure_fresh_caps_concurrent_requests(cache_file, frozen_time, threads):
    cache = cw.WeatherStationCache()
    for index in range(cw.MAX_CONCURRENT_WEATHER_REQUESTS + 3):
        cache.ensure_fresh(f"city{index}", 0.0, 0.0)
    assert len(threads) == cw.MAX_CONCURRENT_WEATHER_REQUESTS
    assert cache.loading_count() == cw.MAX_CONCURRENT_WEATHER_REQUESTS


def test_worker_completes_and_frees_its_slot(cache_file, frozen_time, threads, fake_http):
    updates = []
    cache = cw.WeatherStationCache(on_update=lambda: updates.append(1))
    fake_http.responses[cw.WEATHER_API_URL] = _payload()
    cache.ensure_fresh("tokyo", 35.0, 139.0)
    threads[0].run()
    assert cache.loading_count() == 0
    assert updates == [1]
    entry = cache.get("tokyo")
    assert entry["temperature"] == 12.3 and entry["schema"] == cw.WEATHER_SCHEMA
    assert entry["fetched_at"] == NOW and entry["error"] == ""
    assert [day["date"] for day in entry["forecast"]] == ["2026-10-19", "2026-10-20"]
    assert entry["forecast"][0]["hours"] == [[0, 8.0, None, None, None], [1, 7.5, None, None, None]]
    assert json.loads(cache_file.read_text(encoding="utf-8"))["tokyo"]["temperature"] == 12.3
    cache.ensure_fresh("tokyo", 35.0, 139.0)  # fresh now
    assert len(threads) == 1


# ---- _fetch_worker: ingest coercion and fallbacks ------------------------------------


def test_failed_fetch_keeps_previous_reading_and_schedules_a_retry(cache_file, frozen_time, fake_http):
    cache = cw.WeatherStationCache()
    previous = {
        "temperature": 4.0, "windspeed": 3.0, "winddirection": 10.0, "weathercode": 2,
        "observed_at": "earlier", "forecast": [{"date": "2026-10-18"}], "schema": cw.WEATHER_SCHEMA,
        "fetched_at": NOW - 5000, "error": "",
    }
    cache._entries["k"] = dict(previous)
    cache._in_progress.add("k")
    cache._fetch_worker("k", 1.0, 2.0)  # unregistered URL: network error
    entry = cache.get("k")
    for field in ("temperature", "windspeed", "winddirection", "weathercode", "observed_at", "forecast", "schema"):
        assert entry[field] == previous[field]
    assert entry["error"].startswith("Weather request failed")
    assert entry["fetched_at"] == NOW
    assert "k" not in cache._in_progress
    assert cache._is_stale("k") is False
    frozen_time.now = NOW + cw.WEATHER_RETRY_INTERVAL_SECONDS
    assert cache._is_stale("k") is True
    assert cache.has_pending_retry() is True


def test_first_fetch_failure_records_an_error_entry(cache_file, frozen_time):
    cache = cw.WeatherStationCache()
    cache._fetch_worker("k", 1.0, 2.0)
    entry = cache.get("k")
    assert entry["temperature"] is None and entry["forecast"] == []
    assert entry["schema"] is None and entry["error"]
    assert cache.failed_count() == 1


@pytest.mark.parametrize("bad", ["12.3abc", None, True, "NaN", [], {"v": 1}])
def test_bad_feed_values_keep_the_previous_good_reading(cache_file, frozen_time, fake_http, bad):
    cache = cw.WeatherStationCache()
    cache._entries["k"] = {"fetched_at": NOW - 10_000, "temperature": 9.0, "windspeed": 4.0, "winddirection": 80.0}
    fake_http.responses[cw.WEATHER_API_URL] = _payload(
        temperature=bad, wind_speed_10m=bad, wind_direction_10m=bad
    )
    cache._fetch_worker("k", 1.0, 2.0)
    entry = cache.get("k")
    assert (entry["temperature"], entry["windspeed"], entry["winddirection"]) == (9.0, 4.0, 80.0)
    assert entry["error"] == "" and entry["schema"] == cw.WEATHER_SCHEMA


def test_numeric_strings_from_the_feed_are_stored_as_floats(cache_file, frozen_time, fake_http):
    cache = cw.WeatherStationCache()
    fake_http.responses[cw.WEATHER_API_URL] = _payload(temperature="-3.5", wind_speed_10m="7")
    cache._fetch_worker("k", 1.0, 2.0)
    entry = cache.get("k")
    assert entry["temperature"] == -3.5 and entry["windspeed"] == 7.0


def test_missing_daily_block_keeps_previous_forecast(cache_file, frozen_time, fake_http):
    cache = cw.WeatherStationCache()
    cache._entries["k"] = {"fetched_at": 1.0, "forecast": [{"date": "old"}]}
    payload = _payload()
    payload["daily"] = {}
    fake_http.responses[cw.WEATHER_API_URL] = payload
    cache._fetch_worker("k", 1.0, 2.0)
    assert cache.get("k")["forecast"] == [{"date": "old"}]


def test_worker_releases_its_slot_even_when_the_callback_raises(cache_file, frozen_time):
    def explode():
        raise RuntimeError("ui gone")

    cache = cw.WeatherStationCache(on_update=explode)
    cache._in_progress.add("k")
    with pytest.raises(RuntimeError):
        cache._fetch_worker("k", 0.0, 0.0)
    assert cache.loading_count() == 0
