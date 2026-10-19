"""OpenSky state-vector fetching and flight normalisation in layers/flights/api.py."""

from __future__ import annotations

import io
import json
import math
import urllib.error
import urllib.request

import pytest

from terrascope.layers.flights import api
from terrascope.layers.flights.api import (
    FlightFetchError,
    fetch_opensky_states,
    normalize_flights,
    normalize_states,
)
from terrascope.layers.flights.settings import OPEN_SKY_API_URL, REQUEST_TIMEOUT_SECONDS

NAN = float("nan")


def state_row(
    icao24="abc123",
    callsign="BAW123  ",
    country="United Kingdom",
    lon=-0.45,
    lat=51.47,
    on_ground=False,
    category=None,
):
    """One OpenSky state vector (17 fields, or 18 with a category)."""
    row = [
        icao24, callsign, country,
        1_700_000_000, 1_700_000_005,  # time_position, last_contact
        lon, lat,
        10_000.0,  # baro_altitude
        on_ground,
        230.0, 90.0, -2.5,  # velocity, true_track, vertical_rate
        None, 10_200.0, "1234", False, 0,  # sensors, geo_altitude, squawk, spi, position_source
    ]
    if category is not None:
        row.append(category)
    return row


@pytest.fixture
def progress(monkeypatch):
    """Record set_live_progress calls instead of touching the global progress table."""
    calls: list[dict] = []

    def record(name, **values):
        calls.append({"name": name, **values})

    monkeypatch.setattr(api, "set_live_progress", record)
    return calls


# ---- fetch_opensky_states -----------------------------------------------------------


def test_fetch_returns_the_states_list(fake_http, progress):
    rows = [state_row(), state_row(icao24="def456")]
    fake_http.responses[OPEN_SKY_API_URL] = {"time": 1, "states": rows}

    assert fetch_opensky_states() == rows
    assert fake_http.calls == [OPEN_SKY_API_URL]
    assert progress[0] == {"name": "planes", "state": "downloading"}
    assert progress[-1] == {"name": "planes", "state": "ready", "downloaded": 1, "total": 1}
    assert {call["name"] for call in progress} == {"planes"}


def test_fetch_without_states_key_is_an_empty_sky(fake_http, progress):
    fake_http.responses[OPEN_SKY_API_URL] = {"time": 1}
    assert fetch_opensky_states() == []
    assert progress[-1]["state"] == "ready"


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError(OPEN_SKY_API_URL, 429, "Too Many Requests", {}, None),
        urllib.error.HTTPError(OPEN_SKY_API_URL, 503, "Service Unavailable", {}, None),
        urllib.error.URLError("name resolution failed"),
        TimeoutError("timed out"),
        ConnectionResetError("reset by peer"),
    ],
    ids=["http-429", "http-503", "url-error", "timeout", "connection-reset"],
)
def test_fetch_wraps_network_errors(fake_http, progress, error):
    fake_http.responses[OPEN_SKY_API_URL] = error
    with pytest.raises(FlightFetchError, match="OpenSky request failed") as caught:
        fetch_opensky_states()
    assert caught.value.__cause__ is error
    assert progress[-1]["state"] == "failed"
    assert progress[-1]["error"] == str(error)


def test_fetch_unreachable_host_is_a_fetch_error(progress):
    # Nothing registered: the network guard raises URLError like a dead host.
    with pytest.raises(FlightFetchError):
        fetch_opensky_states()
    assert progress[-1]["state"] == "failed"


@pytest.mark.parametrize("body", [b"", b"{not json", b"<html>502 Bad Gateway</html>", b'{"states": ['])
def test_fetch_rejects_malformed_json(fake_http, progress, body):
    fake_http.responses[OPEN_SKY_API_URL] = body
    with pytest.raises(FlightFetchError, match="OpenSky request failed"):
        fetch_opensky_states()
    assert progress[-1]["state"] == "failed"


@pytest.mark.parametrize(
    "payload",
    [[], [1, 2], "states", 42, None, {"states": {"abc": []}}, {"states": "none"}, {"states": 7}],
)
def test_fetch_rejects_unexpected_response_shapes(fake_http, progress, payload):
    fake_http.responses[OPEN_SKY_API_URL] = payload
    with pytest.raises(FlightFetchError, match="unexpected response format"):
        fetch_opensky_states()
    assert progress[-1] == {"name": "planes", "state": "failed", "error": "Unexpected response format"}


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a non-UTF-8 body (e.g. gzip from a proxy) raises UnicodeDecodeError instead of FlightFetchError",
)
def test_fetch_wraps_undecodable_bodies(fake_http, progress):
    fake_http.responses[OPEN_SKY_API_URL] = b"\x1f\x8b\x08\x00\x00\x00\x00\x00"
    with pytest.raises(FlightFetchError):
        fetch_opensky_states()


class _SizedResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict) -> None:
        super().__init__(body)
        self.headers = headers


def test_fetch_sends_headers_and_reports_chunked_progress(monkeypatch, progress):
    rows = [state_row(icao24=f"{index:06x}") for index in range(600)]
    body = json.dumps({"states": rows}).encode("utf-8")
    assert len(body) > 2 * 32 * 1024  # spans several read() chunks
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        return _SizedResponse(body, {"Content-Length": str(len(body))})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert fetch_opensky_states() == rows
    request = seen["request"]
    assert request.full_url == OPEN_SKY_API_URL
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("User-agent")
    assert seen["timeout"] == REQUEST_TIMEOUT_SECONDS

    downloads = [call for call in progress if "downloaded" in call and call["state"] == "downloading"]
    assert len(downloads) >= 3
    assert all(call["total"] == len(body) for call in downloads)
    assert [call["downloaded"] for call in downloads] == sorted(call["downloaded"] for call in downloads)
    assert downloads[-1]["downloaded"] == len(body)


@pytest.mark.parametrize("length", ["", "abc", "-5", "12.5"])
def test_fetch_ignores_a_non_numeric_content_length(monkeypatch, progress, length):
    body = json.dumps({"states": []}).encode("utf-8")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: _SizedResponse(body, {"Content-Length": length}))
    assert fetch_opensky_states() == []
    assert all(call.get("total") in (None, 1) for call in progress)


# ---- normalize_states -----------------------------------------------------------------


def test_normalize_states_maps_every_field():
    (flight,) = normalize_states([state_row(icao24=" abc123 ", category=4)])
    assert flight == {
        "icao24": "ABC123",
        "callsign": "BAW123",
        "origin_country": "United Kingdom",
        "time_position": 1_700_000_000,
        "last_contact": 1_700_000_005,
        "longitude": -0.45,
        "latitude": 51.47,
        "baro_altitude": 10_000.0,
        "on_ground": False,
        "velocity": 230.0,
        "true_track": 90.0,
        "vertical_rate": -2.5,
        "sensors": None,
        "geo_altitude": 10_200.0,
        "squawk": "1234",
        "spi": False,
        "position_source": 0,
        "category": 4,
    }


def test_normalize_states_without_category_column_has_no_category():
    (flight,) = normalize_states([state_row()])
    assert flight["category"] is None


def test_normalize_states_fills_blank_identity_fields():
    (flight,) = normalize_states([state_row(icao24=None, callsign=None, country=None)])
    assert (flight["icao24"], flight["callsign"], flight["origin_country"]) == ("", "", "Unknown")


def test_normalize_states_coerces_numeric_string_coordinates():
    (flight,) = normalize_states([state_row(lon="12.5", lat="-7")])
    assert (flight["longitude"], flight["latitude"]) == (12.5, -7.0)


@pytest.mark.parametrize(("lon", "lat"), [(180, 90), (-180, -90), (179.999, 0), (-179.999, 0)])
def test_normalize_states_keeps_positions_on_the_antimeridian_and_poles(lon, lat):
    assert len(normalize_states([state_row(lon=lon, lat=lat)])) == 1


@pytest.mark.parametrize(
    "row",
    [
        None,
        "abc123",
        {"icao24": "abc123"},
        tuple(state_row()),
        state_row()[:16],
        [],
        state_row(on_ground=True),
        state_row(on_ground=None),
        state_row(on_ground=0),
        state_row(on_ground="false"),
        state_row(lon=None),
        state_row(lat=None),
        state_row(lon="west"),
        state_row(lat=[51]),
        state_row(lon=NAN),
        state_row(lat=float("inf")),
        state_row(lon=180.01),
        state_row(lon=-181),
        state_row(lat=90.5),
        state_row(lat=-91),
    ],
)
def test_normalize_states_drops_unusable_rows(row):
    assert normalize_states([row]) == []


def test_normalize_states_keeps_good_rows_among_bad_ones():
    rows = [None, state_row(icao24="aaa111"), state_row(lon="x"), state_row(icao24="bbb222")]
    assert [flight["icao24"] for flight in normalize_states(rows)] == ["AAA111", "BBB222"]


def test_normalize_states_of_nothing_is_empty():
    assert normalize_states([]) == []


# ---- normalize_flights (cached flights) -------------------------------------------------


def cached(**fields):
    flight = {"icao24": "abc123", "longitude": 1.0, "latitude": 2.0, "on_ground": False}
    flight.update(fields)
    return flight


@pytest.mark.parametrize("value", [None, {}, "flights", 5, {"flights": []}])
def test_normalize_flights_requires_a_list(value):
    assert normalize_flights(value) == []


@pytest.mark.parametrize(
    "item",
    [
        None,
        "abc123",
        ["abc123", 1.0, 2.0],
        {"icao24": "abc123", "latitude": 2.0, "on_ground": False},
        cached(longitude=None),
        cached(latitude="north"),
        cached(longitude=NAN),
        cached(latitude=float("-inf")),
        cached(on_ground=True),
        cached(on_ground=None),
        cached(on_ground=0),
        {"icao24": "abc123", "longitude": 1.0, "latitude": 2.0},
    ],
)
def test_normalize_flights_drops_unusable_entries(item):
    assert normalize_flights([item]) == []


def test_normalize_flights_normalises_identity_and_coordinates_without_mutating_input():
    original = cached(icao24=" abc123 ", longitude="10.5", latitude="-3", callsign="X", extra=[1])
    (flight,) = normalize_flights([original])
    assert flight["icao24"] == "ABC123"
    assert (flight["longitude"], flight["latitude"]) == (10.5, -3.0)
    assert flight["callsign"] == "X" and flight["extra"] == [1]
    assert original["icao24"] == " abc123 " and original["longitude"] == "10.5"


def test_normalize_flights_blank_icao24_becomes_empty_string():
    (flight,) = normalize_flights([cached(icao24=None)])
    assert flight["icao24"] == ""


def test_normalize_flights_round_trips_normalize_states_output():
    flights = normalize_states([state_row(category=5), state_row(icao24="def456")])
    assert normalize_flights(json.loads(json.dumps(flights))) == flights


def test_normalize_flights_coordinates_are_finite_floats():
    (flight,) = normalize_flights([cached(longitude=3, latitude=4)])
    assert isinstance(flight["longitude"], float) and math.isfinite(flight["longitude"])
