"""USGS earthquake feed parsing, magnitude bands, formatting helpers and the on-disk cache."""

from __future__ import annotations

import json
import math
import urllib.error

import pytest

from terrascope.layers.weather import earthquakes as eq

NOW = 1_760_000_000.0


@pytest.fixture
def frozen_time(monkeypatch):
    monkeypatch.setattr(eq.time, "time", lambda: NOW)
    return NOW


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setattr(eq, "FETCH_NEW_DATA_FROM_API", True)


def _feature(lon=142.4, lat=38.3, depth=10.0, mag=5.1, **properties) -> dict:
    coordinates = [lon, lat] if depth is ... else [lon, lat, depth]
    return {
        "id": properties.pop("id", "us1"),
        "geometry": {"type": "Point", "coordinates": coordinates},
        "properties": {"mag": mag, "place": "off the coast", "time": 1_759_990_000_000, **properties},
    }


# ---- magnitude colour / glyph bands --------------------------------------------


def test_bands_are_read_from_config_in_ascending_order():
    limits = [limit for limit, _color in eq.MAGNITUDE_COLOR_BANDS]
    assert limits == sorted(limits)
    assert len(eq.LEGEND_LABELS) == len(eq.MAGNITUDE_COLOR_BANDS) + 1


def test_magnitude_colour_boundaries_belong_to_the_next_band():
    bands = eq.MAGNITUDE_COLOR_BANDS
    assert eq.magnitude_to_color(None) == eq.MAGNITUDE_COLOR_UNKNOWN
    assert eq.magnitude_to_color(-1.0) == bands[0][1]
    for index, (limit, color) in enumerate(bands):
        assert eq.magnitude_to_color(limit - 0.01) == color
        following = bands[index + 1][1] if index + 1 < len(bands) else eq.MAGNITUDE_COLOR_MAJOR
        assert eq.magnitude_to_color(limit) == following
    assert eq.magnitude_to_color(9.5) == eq.MAGNITUDE_COLOR_MAJOR


def test_magnitude_glyph_boundaries():
    glyphs = eq.QUAKE_GLYPHS
    assert eq.magnitude_to_glyph(None) == glyphs[0][1]
    assert eq.magnitude_to_glyph(-2.0) == glyphs[0][1]
    for index, (minimum, glyph) in enumerate(glyphs[1:], start=1):
        assert eq.magnitude_to_glyph(minimum) == glyph
        assert eq.magnitude_to_glyph(minimum - 0.01) == glyphs[index - 1][1]
    assert eq.magnitude_to_glyph(10.0) == glyphs[-1][1]


# ---- formatting -------------------------------------------------------------------


@pytest.mark.parametrize(
    "magnitude, text", [(None, "N/A"), (4.25, "M 4.2"), (0, "M 0.0"), (-0.5, "M -0.5")]
)
def test_format_magnitude(magnitude, text):
    assert eq.format_magnitude(magnitude) == text


@pytest.mark.parametrize("depth, text", [(None, "N/A"), (10, "10.0 km"), (0.04, "0.0 km")])
def test_format_depth(depth, text):
    assert eq.format_depth(depth) == text


@pytest.mark.parametrize(
    "seconds_ago, text",
    [
        (0, "0m ago"),
        (59 * 60, "59m ago"),
        (3599, "60m ago"),
        (3600, "1.0h ago"),
        (5400, "1.5h ago"),
        (86399, "24.0h ago"),
        (86400, "1.0d ago"),
        (3 * 86400, "3.0d ago"),
        (-600, "0m ago"),  # a timestamp from the future clamps to now
    ],
)
def test_format_time_ago_bands(frozen_time, seconds_ago, text):
    assert eq.format_time_ago((NOW - seconds_ago) * 1000) == text


@pytest.mark.parametrize("value", [None, "yesterday", [], {}])
def test_format_time_ago_rejects_garbage(frozen_time, value):
    assert eq.format_time_ago(value) == "N/A"


def test_format_time_ago_accepts_numeric_strings(frozen_time):
    assert eq.format_time_ago(str((NOW - 120) * 1000)) == "2m ago"


@pytest.mark.parametrize("flag, text", [(True, "WARNING"), (1, "WARNING"), (False, "NO"), (0, "NO"), (None, "NO")])
def test_format_tsunami(flag, text):
    assert eq.format_tsunami(flag) == text


# ---- fetch_usgs_earthquakes -------------------------------------------------------


def test_fetch_returns_features(fake_http):
    fake_http.responses[eq.USGS_FEED_URL] = {"type": "FeatureCollection", "features": [_feature()]}
    assert eq.fetch_usgs_earthquakes() == [_feature()]
    assert fake_http.calls == [eq.USGS_FEED_URL]


def test_fetch_treats_missing_features_as_empty(fake_http):
    fake_http.responses[eq.USGS_FEED_URL] = {"type": "FeatureCollection"}
    assert eq.fetch_usgs_earthquakes() == []


@pytest.mark.parametrize(
    "body",
    [
        urllib.error.URLError("down"),
        urllib.error.HTTPError(eq.USGS_FEED_URL, 503, "busy", {}, None),
        TimeoutError("slow"),
        b'{"features": [',  # truncated
        b"",
        b"<html>maintenance</html>",
        [1, 2, 3],
        None,
        "features",
        {"features": {"0": {}}},
        {"features": "none"},
    ],
    ids=[
        "url-error", "http-error", "timeout", "truncated", "empty", "html",
        "list", "null", "string", "features-dict", "features-string",
    ],
)
def test_fetch_wraps_every_failure_in_earthquake_fetch_error(fake_http, body):
    fake_http.responses[eq.USGS_FEED_URL] = body
    with pytest.raises(eq.EarthquakeFetchError):
        eq.fetch_usgs_earthquakes()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a non-UTF-8 body raises UnicodeDecodeError, not EarthquakeFetchError (escapes the prefetch thread)",
)
def test_fetch_wraps_undecodable_bodies(fake_http):
    fake_http.responses[eq.USGS_FEED_URL] = b"\x80\x81\xfe"
    with pytest.raises(eq.EarthquakeFetchError):
        eq.fetch_usgs_earthquakes()


# ---- normalize_features -----------------------------------------------------------


def test_normalize_feature_full_record():
    feature = _feature(
        tsunami=1, alert="orange", felt=12, status="reviewed", type="earthquake",
        url="https://usgs/1", updated=5,
    )
    (quake,) = eq.normalize_features([feature])
    assert quake == {
        "id": "us1",
        "place": "off the coast",
        "magnitude": 5.1,
        "depth_km": 10.0,
        "time": 1_759_990_000_000,
        "updated": 5,
        "tsunami": True,
        "alert": "orange",
        "felt": 12,
        "status": "reviewed",
        "event_type": "earthquake",
        "url": "https://usgs/1",
        "longitude": 142.4,
        "latitude": 38.3,
    }


def test_normalize_feature_defaults_for_sparse_properties():
    feature = {"geometry": {"coordinates": ["10", "20"]}, "properties": None}
    (quake,) = eq.normalize_features([feature])
    assert quake["id"] == ""
    assert quake["place"] == "Unknown location"
    assert quake["magnitude"] is None
    assert quake["depth_km"] is None
    assert quake["tsunami"] is False
    assert quake["event_type"] == "earthquake"
    assert (quake["longitude"], quake["latitude"]) == (10.0, 20.0)


@pytest.mark.parametrize("mag, expected", [("4.7", 4.7), ("strong", None), ([5], None), ({}, None), (None, None)])
def test_normalize_feature_coerces_magnitude(mag, expected):
    (quake,) = eq.normalize_features([_feature(mag=mag)])
    assert quake["magnitude"] == expected


@pytest.mark.parametrize(
    "feature",
    [
        None,
        "quake",
        {"geometry": None, "properties": {}},
        {"geometry": {"coordinates": [1]}, "properties": {}},
        {"geometry": {"coordinates": "1,2"}, "properties": {}},
        {"geometry": {"coordinates": [1, 2]}, "properties": "bad"},
        {"geometry": ["not", "a", "dict"], "properties": {}},
        _feature(lon="east"),
        _feature(lat=None),
        _feature(depth="deep"),
        _feature(lon=180.01),
        _feature(lat=-90.5),
        _feature(lon=float("nan")),
        _feature(lat=float("inf")),
    ],
    ids=[
        "none", "string", "no-geometry", "one-coordinate", "coordinates-string",
        "properties-string", "geometry-list", "lon-text", "lat-null", "depth-text",
        "lon-out-of-range", "lat-out-of-range", "lon-nan", "lat-inf",
    ],
)
def test_normalize_feature_drops_unplaceable_records(feature):
    assert eq.normalize_features([feature]) == []


@pytest.mark.parametrize("lon, lat", [(-180, -90), (180, 90), (0, 0)])
def test_normalize_feature_accepts_coordinate_extremes(lon, lat):
    assert len(eq.normalize_features([_feature(lon=lon, lat=lat)])) == 1


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a null depth (valid GeoJSON) drops the whole quake instead of depth_km=None",
)
def test_normalize_feature_keeps_a_quake_with_null_depth():
    (quake,) = eq.normalize_features([_feature(depth=None)])
    assert quake["depth_km"] is None


def test_normalize_feature_two_coordinate_point_has_no_depth():
    (quake,) = eq.normalize_features([_feature(depth=...)])
    assert quake["depth_km"] is None


def test_normalize_features_keeps_good_records_among_bad_ones():
    quakes = eq.normalize_features([None, _feature(id="a"), {"geometry": {}}, _feature(id="b")])
    assert [quake["id"] for quake in quakes] == ["a", "b"]


# ---- normalize_cached ---------------------------------------------------------------


def test_normalize_cached_rejects_non_lists():
    assert eq.normalize_cached({"quakes": []}) == []
    assert eq.normalize_cached(None) == []


def test_normalize_cached_filters_and_coerces_positions():
    cached = [
        {"id": "ok", "longitude": "12.5", "latitude": -3, "magnitude": 4.0},
        {"id": "missing-lat", "longitude": 1},
        {"id": "text", "longitude": "x", "latitude": 1},
        {"id": "nan", "longitude": float("nan"), "latitude": 1},
        {"id": "inf", "longitude": 1, "latitude": float("-inf")},
        {"id": "range", "longitude": 200, "latitude": 1},
        "junk",
        None,
    ]
    result = eq.normalize_cached(cached)
    assert result == [{"id": "ok", "longitude": 12.5, "latitude": -3.0, "magnitude": 4.0}]
    assert result[0] is not cached[0]  # a copy, not the loaded dict


@pytest.mark.xfail(
    strict=True,
    reason="BUG: normalize_cached does not coerce magnitude, so a string magnitude in the cache crashes band maths",
)
@pytest.mark.parametrize("magnitude", ["5.0", "big"])
def test_normalize_cached_coerces_magnitude(magnitude):
    (quake,) = eq.normalize_cached([{"longitude": 1, "latitude": 2, "magnitude": magnitude}])
    assert quake["magnitude"] is None or type(quake["magnitude"]) is float
    eq.magnitude_to_color(quake["magnitude"])


# ---- EarthquakeCache ---------------------------------------------------------------


def test_cache_starts_empty_without_a_file(tmp_path):
    cache = eq.EarthquakeCache(tmp_path / "quakes.json")
    assert cache.quakes == [] and cache.last_fetch is None
    assert cache.age_seconds() is None


@pytest.mark.parametrize(
    "content",
    ["", "{not json", "[]", '"text"', "null", "\x00\x01"],
    ids=["empty", "truncated", "list", "string", "null", "binary"],
)
def test_cache_ignores_corrupt_files(tmp_path, content):
    path = tmp_path / "quakes.json"
    path.write_text(content, encoding="utf-8")
    cache = eq.EarthquakeCache(path)
    assert cache.quakes == [] and cache.last_fetch is None


@pytest.mark.parametrize(
    "last_fetch, expected",
    [(NOW - 10, NOW - 10), ("12.5", 12.5), ("soon", None), ([1], None), (None, None)],
)
def test_cache_load_coerces_last_fetch(tmp_path, last_fetch, expected):
    path = tmp_path / "quakes.json"
    path.write_text(json.dumps({"last_fetch": last_fetch, "quakes": []}), encoding="utf-8")
    assert eq.EarthquakeCache(path).last_fetch == expected


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_cache_load_rejects_non_finite_last_fetch(tmp_path, token):
    path = tmp_path / "quakes.json"
    path.write_text('{"last_fetch": %s, "quakes": []}' % token, encoding="utf-8")
    assert eq.EarthquakeCache(path).last_fetch is None


def test_cache_save_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "quakes.json"
    cache = eq.EarthquakeCache(path)
    cache.quakes = eq.normalize_features([_feature(id="a"), _feature(id="b", mag=None)])
    cache.last_fetch = NOW
    cache.save()
    assert not path.with_suffix(".json.tmp").exists()
    reloaded = eq.EarthquakeCache(path)
    assert reloaded.last_fetch == NOW
    assert reloaded.quakes == cache.quakes


def test_get_quakes_returns_copies(tmp_path):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    cache.quakes = [{"id": "a", "longitude": 1.0, "latitude": 2.0}]
    copy = cache.get_quakes()
    copy[0]["id"] = "changed"
    assert cache.quakes[0]["id"] == "a"


def test_age_never_negative(tmp_path, frozen_time):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    cache.last_fetch = NOW + 500  # clock went backwards
    assert cache.age_seconds() == 0.0
    cache.last_fetch = NOW - 42
    assert cache.age_seconds() == 42


def test_offline_never_fetches(tmp_path, fake_http):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    assert eq.FETCH_NEW_DATA_FROM_API is False
    assert cache.should_fetch() is False
    assert cache.refresh_if_due() is False
    assert cache.refresh_if_due(force=True) is False
    assert fake_http.calls == []


@pytest.mark.parametrize(
    "age, due",
    [(None, True), (0, False), (eq.EARTHQUAKE_REFRESH_INTERVAL_SECONDS - 1, False), (eq.EARTHQUAKE_REFRESH_INTERVAL_SECONDS, True)],
)
def test_should_fetch_respects_refresh_interval(tmp_path, online, frozen_time, age, due):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    cache.last_fetch = None if age is None else NOW - age
    assert cache.should_fetch() is due


def test_refresh_success_replaces_quakes_and_persists(tmp_path, online, frozen_time, fake_http):
    path = tmp_path / "q.json"
    fake_http.responses[eq.USGS_FEED_URL] = {"features": [_feature(id="a"), None, _feature(id="b")]}
    cache = eq.EarthquakeCache(path)
    cache.last_error = "old"
    assert cache.refresh_if_due() is True
    assert [quake["id"] for quake in cache.get_quakes()] == ["a", "b"]
    assert cache.last_fetch == NOW and cache.last_error == ""
    assert json.loads(path.read_text(encoding="utf-8"))["last_fetch"] == NOW


def test_refresh_not_due_skips_the_network(tmp_path, online, frozen_time, fake_http):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    cache.last_fetch = NOW - 1
    assert cache.refresh_if_due() is False
    assert fake_http.calls == []
    fake_http.responses[eq.USGS_FEED_URL] = {"features": []}
    assert cache.refresh_if_due(force=True) is True


def test_refresh_failure_keeps_last_good_quakes(tmp_path, online, frozen_time, fake_http):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    cache.quakes = [{"id": "kept", "longitude": 1.0, "latitude": 1.0}]
    cache.last_fetch = NOW - 10_000
    fake_http.responses[eq.USGS_FEED_URL] = b"{truncated"
    assert cache.refresh_if_due() is False
    assert cache.quakes[0]["id"] == "kept"
    assert cache.last_fetch == NOW - 10_000
    assert cache.last_error.startswith("USGS request failed")


def test_refresh_reports_a_failed_save_but_keeps_the_data(tmp_path, online, frozen_time, fake_http, monkeypatch):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    fake_http.responses[eq.USGS_FEED_URL] = {"features": [_feature()]}

    def refuse():
        raise OSError("disk full")

    monkeypatch.setattr(cache, "save", refuse)
    assert cache.refresh_if_due() is True
    assert len(cache.quakes) == 1
    assert "disk full" in cache.last_error


def test_refresh_rechecks_freshness_under_the_lock(tmp_path, online, frozen_time, fake_http, monkeypatch):
    # Another thread finished a fetch between the unlocked and locked checks.
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    answers = iter([True, False])
    monkeypatch.setattr(cache, "should_fetch", lambda: next(answers))
    assert cache.refresh_if_due() is False
    assert fake_http.calls == []


def test_magnitudes_survive_json_round_trip_as_floats(tmp_path):
    cache = eq.EarthquakeCache(tmp_path / "q.json")
    cache.quakes = eq.normalize_features([_feature(mag=7)])
    cache.save()
    (quake,) = eq.EarthquakeCache(tmp_path / "q.json").quakes
    assert isinstance(quake["magnitude"], float) and math.isclose(quake["magnitude"], 7.0)
