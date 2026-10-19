"""FlightCache refresh scheduling, history merging and persistence in layers/flights/cache.py."""

from __future__ import annotations

import json
import time

import pytest

from terrascope.layers.flights import api
from terrascope.layers.flights import cache as cache_module
from terrascope.layers.flights.cache import FlightCache
from terrascope.layers.flights.settings import (
    FLIGHT_HISTORY_POINTS,
    FLIGHT_REFRESH_INTERVAL_SECONDS,
    OPEN_SKY_API_URL,
)

NOW = 1_700_000_000.0


def state_row(icao24="abc123", lon=10.0, lat=20.0, last_contact=NOW - 5, callsign="TEST1"):
    return [
        icao24, callsign, "Testland", NOW - 10, last_contact, lon, lat,
        9000.0, False, 200.0, 45.0, 0.0, None, 9100.0, None, False, 0,
    ]


@pytest.fixture(autouse=True)
def _quiet_progress(monkeypatch):
    """Keep fetches from writing the global startup-progress table."""
    monkeypatch.setattr(api, "set_live_progress", lambda *args, **kwargs: None)


@pytest.fixture
def clock(monkeypatch):
    """A controllable time.time()."""

    class Clock:
        now = NOW

    monkeypatch.setattr(time, "time", lambda: Clock.now)
    return Clock


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setattr(cache_module, "FETCH_NEW_DATA_FROM_API", True)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "flight_cache.json"


def write_cache(path, data) -> None:
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def flight(icao24="ABC123", lon=1.0, lat=2.0, **extra):
    return {"icao24": icao24, "longitude": lon, "latitude": lat, "on_ground": False, **extra}


# ---- load -------------------------------------------------------------------------------


def test_missing_cache_file_starts_empty(path):
    cache = FlightCache(path)
    assert (cache.last_fetch, cache.flights, cache.history, cache.last_error) == (None, [], {}, "")
    assert cache.age_seconds() is None


@pytest.mark.parametrize("text", ["", "{not json", "[]", '"flights"', "42", "null"])
def test_corrupt_or_non_object_cache_is_ignored(path, text):
    write_cache(path, text)
    cache = FlightCache(path)
    assert (cache.last_fetch, cache.flights, cache.history) == (None, [], {})


def test_unreadable_cache_path_is_ignored(tmp_path):
    directory = tmp_path / "flight_cache.json"
    directory.mkdir()
    cache = FlightCache(directory)
    assert cache.flights == [] and cache.last_fetch is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("null", None),
        ("1700000000.5", 1700000000.5),
        ("0", 0.0),
        ('"1700000000"', 1700000000.0),
        ('"yesterday"', None),
        ('"nan"', None),
        ("NaN", None),
        ("Infinity", None),
        ('"-inf"', None),
        ("[1]", None),
        ('{"t": 1}', None),
    ],
)
def test_last_fetch_is_loaded_only_when_finite(path, raw, expected):
    write_cache(path, f'{{"last_fetch": {raw}}}')
    assert FlightCache(path).last_fetch == expected


def test_missing_last_fetch_key_is_none(path):
    write_cache(path, {"flights": []})
    assert FlightCache(path).last_fetch is None


def test_cached_flights_are_normalised_on_load(path):
    write_cache(path, {"flights": [flight(icao24="abc123"), flight(on_ground=True), "junk", flight(lon="x")]})
    cache = FlightCache(path)
    assert [f["icao24"] for f in cache.flights] == ["ABC123"]


@pytest.mark.parametrize("history", [[], "history", 5, None])
def test_non_dict_history_is_ignored(path, history):
    write_cache(path, {"history": history})
    assert FlightCache(path).history == {}


def test_cached_history_drops_invalid_points_and_uppercases_keys(path):
    good = {"timestamp": NOW, "longitude": 1.0, "latitude": 2.0}
    write_cache(
        path,
        {
            "history": {
                "abc123": [good, {"longitude": None, "latitude": 1}, "junk", {"latitude": 1}],
                "def456": "not a list",
                "ghi789": [{"longitude": 999, "latitude": 0}],
                "jkl000": [],
            }
        },
    )
    cache = FlightCache(path)
    assert cache.history == {"ABC123": [good]}


def test_cached_history_keeps_points_with_bad_timestamps_for_the_layer_to_skip(path):
    # load() validates only coordinates; timestamps are filtered where they are
    # used (position_at, the slider range), so a bad one never crashes loading.
    points = [
        {"timestamp": "garbage", "longitude": 1.0, "latitude": 2.0},
        {"timestamp": None, "longitude": 1.5, "latitude": 2.0},
        {"timestamp": NOW, "longitude": 2.0, "latitude": 2.0},
    ]
    write_cache(path, {"history": {"ABC123": points}})
    assert FlightCache(path).get_history("abc123") == points


def test_cached_history_is_trimmed_to_the_newest_points(path):
    points = [{"timestamp": NOW + i, "longitude": float(i), "latitude": 0.0} for i in range(FLIGHT_HISTORY_POINTS + 5)]
    write_cache(path, {"history": {"ABC123": points}})
    assert FlightCache(path).history["ABC123"] == points[-FLIGHT_HISTORY_POINTS:]


# ---- save -------------------------------------------------------------------------------


def test_save_round_trips_through_load(tmp_path):
    path = tmp_path / "nested" / "dir" / "flight_cache.json"
    cache = FlightCache(path)
    cache.last_fetch = NOW
    cache.flights = [flight()]
    cache.history = {"ABC123": [{"timestamp": NOW, "longitude": 1.0, "latitude": 2.0}]}
    cache.save()

    assert not path.with_suffix(".json.tmp").exists()
    reloaded = FlightCache(path)
    assert reloaded.last_fetch == NOW
    assert reloaded.flights == cache.flights
    assert reloaded.history == cache.history


# ---- scheduling -------------------------------------------------------------------------


def test_offline_never_fetches(path, fake_http):
    cache = FlightCache(path)
    assert cache.should_fetch() is False
    assert cache.refresh_if_due() is False
    assert cache.refresh_if_due(force=True) is False
    assert fake_http.calls == []


def test_age_seconds_is_clamped_for_future_timestamps(path, clock):
    cache = FlightCache(path)
    cache.last_fetch = NOW + 100
    assert cache.age_seconds() == 0.0
    cache.last_fetch = NOW - 42
    assert cache.age_seconds() == 42.0


@pytest.mark.parametrize(
    ("last_fetch", "due"),
    [
        (None, True),
        (NOW, False),
        (NOW - FLIGHT_REFRESH_INTERVAL_SECONDS + 1, False),
        (NOW - FLIGHT_REFRESH_INTERVAL_SECONDS, True),
        (NOW - 10 * FLIGHT_REFRESH_INTERVAL_SECONDS, True),
        (NOW + 3600, False),
    ],
)
def test_should_fetch_respects_the_refresh_interval(path, clock, online, last_fetch, due):
    cache = FlightCache(path)
    cache.last_fetch = last_fetch
    assert cache.should_fetch() is due


# ---- refresh ----------------------------------------------------------------------------


def test_refresh_fetches_merges_and_saves(path, clock, online, fake_http):
    fake_http.responses[OPEN_SKY_API_URL] = {"states": [state_row("abc123"), state_row("def456", lon=-50.0)]}
    cache = FlightCache(path)

    assert cache.refresh_if_due() is True
    assert fake_http.calls == [OPEN_SKY_API_URL]
    assert cache.last_fetch == NOW
    assert cache.last_error == ""
    assert [f["icao24"] for f in cache.get_flights()] == ["ABC123", "DEF456"]
    assert cache.get_history("abc123") == [{"timestamp": NOW - 5, "latitude": 20.0, "longitude": 10.0}]

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["last_fetch"] == NOW
    assert set(saved["history"]) == {"ABC123", "DEF456"}


def test_refresh_is_skipped_until_due_unless_forced(path, clock, online, fake_http):
    fake_http.responses[OPEN_SKY_API_URL] = {"states": [state_row()]}
    cache = FlightCache(path)
    assert cache.refresh_if_due() is True

    clock.now = NOW + 10
    assert cache.refresh_if_due() is False
    assert len(fake_http.calls) == 1

    assert cache.refresh_if_due(force=True) is True
    assert len(fake_http.calls) == 2

    clock.now = NOW + 10 + FLIGHT_REFRESH_INTERVAL_SECONDS
    assert cache.refresh_if_due() is True
    assert len(fake_http.calls) == 3


def test_failed_refresh_keeps_previous_data_and_records_error(path, clock, online, fake_http):
    cache = FlightCache(path)
    cache.flights = [flight()]
    cache.last_fetch = NOW - 2 * FLIGHT_REFRESH_INTERVAL_SECONDS

    assert cache.refresh_if_due() is False  # nothing registered: unreachable host
    assert cache.last_error.startswith("OpenSky request failed")
    assert cache.flights == [flight()]
    assert cache.last_fetch == NOW - 2 * FLIGHT_REFRESH_INTERVAL_SECONDS
    assert not path.exists()


def test_successful_refresh_clears_a_previous_error(path, clock, online, fake_http):
    cache = FlightCache(path)
    cache.last_error = "OpenSky request failed: earlier"
    fake_http.responses[OPEN_SKY_API_URL] = {"states": []}
    assert cache.refresh_if_due() is True
    assert cache.last_error == ""
    assert cache.flights == []


def test_save_failure_is_reported_but_refresh_still_succeeds(tmp_path, clock, online, fake_http):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    cache = FlightCache(blocker / "flight_cache.json")
    fake_http.responses[OPEN_SKY_API_URL] = {"states": [state_row()]}

    assert cache.refresh_if_due() is True
    assert cache.last_error.startswith("Could not save flight cache:")
    assert [f["icao24"] for f in cache.flights] == ["ABC123"]


def test_refresh_re_checks_due_inside_the_lock(path, clock, online, monkeypatch):
    # Another thread finished a fetch between the unlocked and locked check.
    cache = FlightCache(path)
    checks = iter([True, False])
    monkeypatch.setattr(cache, "should_fetch", lambda: next(checks))
    monkeypatch.setattr(cache_module, "fetch_opensky_states", lambda: pytest.fail("fetched twice"))
    assert cache.refresh_if_due() is False


# ---- merge (history accumulation) -------------------------------------------------------


def test_merge_appends_moving_aircraft_and_skips_stationary_ones(path, clock):
    cache = FlightCache(path)
    cache._merge([state_row(lon=10.0, last_contact=NOW - 100)])
    cache._merge([state_row(lon=10.0005, last_contact=NOW - 50)])  # moved < 0.001 degrees
    cache._merge([state_row(lon=11.0, last_contact=NOW)])
    assert cache.get_history("ABC123") == [
        {"timestamp": NOW - 100, "latitude": 20.0, "longitude": 10.0},
        {"timestamp": NOW, "latitude": 20.0, "longitude": 11.0},
    ]


def test_merge_trims_history_to_the_configured_length(path, clock):
    cache = FlightCache(path)
    for step in range(FLIGHT_HISTORY_POINTS + 4):
        cache._merge([state_row(lon=float(step), last_contact=NOW + step)])
    history = cache.get_history("ABC123")
    assert len(history) == FLIGHT_HISTORY_POINTS
    assert history[-1]["longitude"] == float(FLIGHT_HISTORY_POINTS + 3)
    assert history[0]["longitude"] == 4.0


def test_merge_prunes_history_of_aircraft_no_longer_reported(path, clock):
    cache = FlightCache(path)
    cache._merge([state_row("aaa111"), state_row("bbb222")])
    cache._merge([state_row("bbb222", lon=12.0)])
    assert set(cache.history) == {"BBB222"}
    assert [f["icao24"] for f in cache.flights] == ["BBB222"]
    assert cache.get_history("aaa111") == []


@pytest.mark.parametrize(
    ("last_contact", "expected"),
    [
        (NOW - 30, NOW - 30),
        (0, 0.0),  # a real zero is kept, not replaced by "now"
        ("1699999990", 1699999990.0),
        ("soon", NOW),
        (None, NOW),
        (float("nan"), NOW),
        (float("inf"), NOW),
        ([1], NOW),
        ({"t": 1}, NOW),
    ],
)
def test_merge_coerces_last_contact_into_a_finite_timestamp(path, clock, last_contact, expected):
    cache = FlightCache(path)
    cache._merge([state_row(last_contact=last_contact)])
    (point,) = cache.get_history("abc123")
    assert point["timestamp"] == expected
    assert isinstance(point["timestamp"], float)


def test_merge_with_bad_last_contact_saves_valid_json(path, clock):
    cache = FlightCache(path)
    cache._merge([state_row(last_contact={"weird": True})])
    cache.save()
    reloaded = FlightCache(path)
    assert reloaded.get_history("ABC123")[0]["timestamp"] == NOW


def test_merge_ignores_unusable_rows(path, clock):
    cache = FlightCache(path)
    cache._merge([None, ["short"], state_row(lon="x"), state_row("good01")])
    assert list(cache.history) == ["GOOD01"]


# ---- accessors --------------------------------------------------------------------------


def test_getters_return_copies(path):
    cache = FlightCache(path)
    cache.flights = [flight()]
    cache.history = {"ABC123": [{"timestamp": NOW, "longitude": 1.0, "latitude": 2.0}]}

    cache.get_flights()[0]["icao24"] = "CHANGED"
    cache.get_history("abc123")[0]["longitude"] = 99.0
    assert cache.flights[0]["icao24"] == "ABC123"
    assert cache.history["ABC123"][0]["longitude"] == 1.0
    assert cache.get_history("unknown") == []
