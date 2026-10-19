"""Natural Earth loading: download / cache / retry, GeoJSON parsing, progress and bundled data."""

from __future__ import annotations

import dataclasses
import io
import json
import math
import urllib.error

import pytest

from terrascope.core import mapdata
from terrascope.core.mapdata import MapDataError


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Every test gets its own empty cache dir and leaves download progress as
    it found it (it is module-global state)."""
    monkeypatch.setattr(mapdata, "CACHE_DIR", tmp_path / "cache")
    saved = {key: dataclasses.replace(item) for key, item in mapdata._DOWNLOAD_PROGRESS.items()}
    yield tmp_path / "cache"
    for key, item in saved.items():
        for field in dataclasses.fields(item):
            setattr(mapdata._DOWNLOAD_PROGRESS[key], field.name, getattr(item, field.name))


@pytest.fixture
def online(monkeypatch):
    monkeypatch.setattr(mapdata, "FETCH_NEW_DATA_FROM_API", True)


def feature(geometry, **properties):
    return {"type": "Feature", "properties": properties, "geometry": geometry}


def polygon(*rings):
    return {"type": "Polygon", "coordinates": [list(map(list, ring)) for ring in rings]}


def collection(*features) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def write_geojson(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # json.dumps writes NaN / Infinity literals, which json.loads accepts:
    # exactly what a hostile or broken file can contain.
    path.write_text(json.dumps(data), encoding="utf-8")


SQUARE = [(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)]


def progress(filename):
    return mapdata._DOWNLOAD_PROGRESS[filename]


# ---- ring building ---------------------------------------------------------------


@pytest.mark.parametrize(
    "points, expected",
    [
        ([], []),
        ([(10, 0), (20, 0)], [(10, 0), (20, 0)]),
        # east across the date line: keep counting up past 180
        ([(170, 0), (179, 0), (-179, 0), (-170, 0)], [(170, 0), (179, 0), (181, 0), (190, 0)]),
        # west across it: keep counting down past -180
        ([(-170, 0), (-179, 0), (179, 0), (170, 0)], [(-170, 0), (-179, 0), (-181, 0), (-190, 0)]),
        # there and back again
        ([(179, 0), (-179, 0), (179, 1)], [(179, 0), (181, 0), (179, 1)]),
    ],
)
def test_unwrap_ring(points, expected):
    assert mapdata.unwrap_ring(points) == expected


def test_make_ring_returns_points_and_bounding_box():
    ring = mapdata.make_ring([[0, 0], [10, -5], [4, 7]])
    assert ring == ([(0.0, 0.0), (10.0, -5.0), (4.0, 7.0)], 0.0, 10.0, -5.0, 7.0)


def test_make_ring_unwraps_before_computing_the_box():
    points, min_lon, max_lon, _, _ = mapdata.make_ring([[178, 0], [-178, 0], [-178, 1]])
    assert (min_lon, max_lon) == (178, 182)
    assert points[-1] == (182, 1)


@pytest.mark.parametrize(
    "points",
    [
        None,
        [],
        [[1, 2]],
        [[1], [2], [3]],
        [[1, 2], [math.nan, 3], [4, math.inf]],
        [[math.inf, math.inf], [-math.inf, 0]],
    ],
)
def test_make_ring_needs_two_usable_points(points):
    assert mapdata.make_ring(points) is None


def test_make_ring_drops_non_finite_and_short_points_and_keeps_the_rest():
    ring = mapdata.make_ring([[1, 2], [math.nan, 3], [4], [5, -math.inf], [6, 7, 100]])
    assert ring == ([(1.0, 2.0), (6.0, 7.0)], 1.0, 6.0, 2.0, 7.0)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: make_ring only filters NaN/inf; a non-numeric or non-sequence coordinate raises TypeError/ValueError/KeyError",
)
@pytest.mark.parametrize(
    "bad_point",
    [[None, 1], ["east", 1], 5, {"lon": 1, "lat": 2}],
    ids=["null", "text", "number", "object"],
)
def test_make_ring_skips_non_numeric_points(bad_point):
    assert mapdata.make_ring([bad_point, [1, 2], [3, 4]]) == ([(1.0, 2.0), (3.0, 4.0)], 1.0, 3.0, 2.0, 4.0)


# ---- country polygons --------------------------------------------------------------


def test_load_polygons_reads_polygons_and_multipolygon_outer_rings(tmp_path):
    path = tmp_path / "countries.geojson"
    hole = [(2, 1), (3, 1), (3, 2), (2, 1)]
    write_geojson(path, collection(
        feature(polygon(SQUARE, hole), ADMIN="Squareland"),
        feature({"type": "MultiPolygon", "coordinates": [
            [[[20, 20], [21, 20], [21, 21]]],
            [],  # empty part
            [[[30, 30], [31, 30], [31, 31]], [[30.2, 30.2], [30.3, 30.2]]],
        ]}),
    ))
    rings = mapdata.load_polygons(path)
    assert [ring[1:] for ring in rings] == [(0, 10, 0, 5), (20, 21, 20, 21), (30, 31, 30, 31)]


@pytest.mark.parametrize(
    "properties",
    [
        {"CONTINENT": "Antarctica"},
        {"continent": "Antarctica"},
        {"ADM0_A3": "ATA"},
        {"iso_a3": "ATA"},
        {"ADMIN": "Antarctica"},
        {"name": "Antarctica"},
    ],
)
def test_load_polygons_leaves_out_antarctica(tmp_path, properties):
    path = tmp_path / "countries.geojson"
    write_geojson(path, collection(feature(polygon(SQUARE), **properties)))
    assert mapdata.load_polygons(path) == []


@pytest.mark.parametrize(
    "bad_feature",
    [
        "not a feature",
        None,
        {"properties": {}},
        {"properties": {}, "geometry": "Polygon"},
        {"properties": None, "geometry": {"type": "Polygon", "coordinates": []}},
        {"properties": "x", "geometry": {"type": "Polygon"}},
        {"geometry": {"type": "MultiPolygon", "coordinates": None}},
        {"geometry": {"type": "Point", "coordinates": [1, 2]}},
        {"geometry": {"type": "LineString", "coordinates": [[1, 2], [3, 4]]}},
        {"geometry": {"type": "Polygon", "coordinates": [[[1, 2]]]}},
        {"geometry": {"type": "Polygon", "coordinates": [[[math.nan, 2], [3, math.inf], [5, 6]]]}},
    ],
)
def test_load_polygons_skips_malformed_features(tmp_path, bad_feature):
    path = tmp_path / "countries.geojson"
    write_geojson(path, collection(bad_feature, feature(polygon(SQUARE))))
    assert [ring[1:] for ring in mapdata.load_polygons(path)] == [(0, 10, 0, 5)]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: load_polygons crashes (TypeError/ValueError, not MapDataError) on non-numeric coordinates instead of skipping them",
)
@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Polygon", "coordinates": [[[None, 0], [1, 1], [2, 0]]]},
        {"type": "Polygon", "coordinates": [[["1°E", 0], [1, 1], [2, 0]]]},
        {"type": "Polygon", "coordinates": 5},
        {"type": "MultiPolygon", "coordinates": [7]},
    ],
    ids=["null-lon", "text-lon", "number-coordinates", "number-polygon"],
)
def test_load_polygons_survives_non_numeric_coordinates(tmp_path, geometry):
    path = tmp_path / "countries.geojson"
    write_geojson(path, collection(feature(geometry), feature(polygon(SQUARE))))
    assert (0, 10, 0, 5) in [ring[1:] for ring in mapdata.load_polygons(path)]


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"\xff\xfe\x00garbage",
        b"[]",
        b'{"features": {"a": 1}}',
        b'{"type": "FeatureCollection"}',
        b"",
    ],
    ids=["broken-json", "bad-utf8", "list", "features-not-list", "no-features", "empty"],
)
def test_unparseable_files_raise_map_data_error(tmp_path, content):
    path = tmp_path / "broken.geojson"
    path.write_bytes(content)
    with pytest.raises(MapDataError, match="Could not parse broken.geojson"):
        mapdata.load_polygons(path)


def test_missing_file_raises_map_data_error(tmp_path):
    with pytest.raises(MapDataError):
        mapdata.load_polygons(tmp_path / "nowhere.geojson")


# ---- province lines ----------------------------------------------------------------


def test_load_province_lines_reads_every_line_shape(tmp_path):
    path = tmp_path / "provinces.geojson"
    write_geojson(path, collection(
        feature({"type": "LineString", "coordinates": [[0, 0], [1, 1]]}),
        feature({"type": "MultiLineString", "coordinates": [[[2, 2], [3, 3]], [[4, 4], [5, 5]]]}),
        feature(polygon([(6, 6), (7, 6), (7, 7)], [(6.2, 6.2), (6.4, 6.4)])),
        feature({"type": "MultiPolygon", "coordinates": [[[[8, 8], [9, 9]]], [[[10, 10], [11, 11]]]]}),
        feature({"type": "Point", "coordinates": [12, 12]}),
        feature({"type": "LineString", "coordinates": []}),
        feature({"type": "LineString", "coordinates": [[13, 13], [math.nan, 14]]}),
    ))
    rings = mapdata.load_province_lines_file(path)
    assert [ring[1] for ring in rings] == [0, 2, 4, 6, 6.2, 8, 10]


@pytest.mark.parametrize(
    "properties",
    [
        {"featurecla": "Adm-0 country"},
        {"FEATURECLA": "International country boundary"},
        {"adm0_a3": "ATA"},
        {"ADM0_A3": "ATA"},
    ],
)
def test_load_province_lines_skips_country_borders_and_antarctica(tmp_path, properties):
    path = tmp_path / "provinces.geojson"
    write_geojson(path, collection(feature({"type": "LineString", "coordinates": [[0, 0], [1, 1]]}, **properties)))
    assert mapdata.load_province_lines_file(path) == []


def test_load_province_lines_tolerates_odd_features(tmp_path):
    path = tmp_path / "provinces.geojson"
    line = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
    write_geojson(path, collection(
        42,
        {"properties": None, "geometry": line},  # no properties: still a line
        {"properties": {"featurecla": "Adm-1 boundary"}, "geometry": None},
        {"properties": {"featurecla": None}, "geometry": line},
    ))
    assert len(mapdata.load_province_lines_file(path)) == 2


# ---- populated places -----------------------------------------------------------------


def place(lon, lat, **properties):
    return feature({"type": "Point", "coordinates": [lon, lat]}, **properties)


def load_places(tmp_path, *features):
    path = tmp_path / "places.geojson"
    write_geojson(path, collection(*features))
    return mapdata.load_places_file(path)


def test_load_places_reads_name_capital_population_and_country(tmp_path):
    places = load_places(
        tmp_path,
        place(139.75, 35.69, name="Tokyo", featurecla="Admin-0 capital", pop_max=35676000,
              pop_min=8336599, adm0name="Japan"),
        place(-1.08, 53.96, NAME="York", FEATURECLA="Populated place", POP_MAX=153717,
              SOV0NAME="United Kingdom"),
    )
    assert places == [
        (139.75, 35.69, "Tokyo", True, 35676000, "Japan"),
        (-1.08, 53.96, "York", False, 153717, "United Kingdom"),
    ]


@pytest.mark.parametrize(
    "properties, is_capital",
    [
        ({"featurecla": "Admin-0 capital"}, True),
        ({"featurecla": "Admin-0 capital alt"}, True),
        ({"featurecla": "Admin-1 capital"}, False),
        ({"featurecla": "Populated place", "adm0cap": 1}, True),
        ({"adm0cap": "1.0"}, True),
        ({"adm0cap": " 1 "}, True),
        ({"adm0cap": 0}, False),
        ({"adm0cap": None}, False),
        ({}, False),
    ],
)
def test_load_places_capital_flag(tmp_path, properties, is_capital):
    [(_, _, _, capital, _, _)] = load_places(tmp_path, place(0, 0, name="X", **properties))
    assert capital is is_capital


@pytest.mark.parametrize(
    "properties, population",
    [
        ({"pop_max": 100, "pop_min": 250}, 250),
        ({"pop_max": "1200.7"}, 1200),
        ({"pop_max": None, "pop_min": None}, 0),
        ({"pop_max": "unknown", "pop_min": 50}, 50),
        ({"pop_max": [1, 2]}, 0),
        ({"pop_max": math.nan}, 0),
        ({"pop_max": -5}, 0),
        ({}, 0),
    ],
)
def test_load_places_population_tolerates_junk(tmp_path, properties, population):
    [(_, _, _, _, pop, _)] = load_places(tmp_path, place(0, 0, name="X", **properties))
    assert pop == population


@pytest.mark.xfail(
    strict=True,
    reason="BUG: load_places_file only catches TypeError/ValueError; int(float('inf')) raises OverflowError on an Infinity population",
)
def test_load_places_survives_an_infinite_population(tmp_path):
    places = load_places(tmp_path, place(0, 0, name="Big", pop_max=math.inf, pop_min=10))
    assert places == [(0.0, 0.0, "Big", False, 10, "")]


def test_load_places_name_falls_back_to_nameascii(tmp_path):
    [entry] = load_places(tmp_path, place(0, 0, name="", nameascii="Sao Paulo"))
    assert entry[2] == "Sao Paulo"


def test_load_places_accepts_numeric_strings_and_extra_dimensions(tmp_path):
    point = feature({"type": "Point", "coordinates": ["12.5", "-3", 100]}, name="Z")
    [entry] = load_places(tmp_path, point)
    assert entry[:2] == (12.5, -3.0)


@pytest.mark.parametrize(
    "bad",
    [
        "junk",
        {"properties": {"name": "X"}},
        {"properties": None, "geometry": {"type": "Point", "coordinates": [0, 0]}},
        feature({"type": "MultiPoint", "coordinates": [[0, 0]]}, name="X"),
        feature({"type": "Point", "coordinates": [0]}, name="X"),
        feature({"type": "Point", "coordinates": "0,0"}, name="X"),
        feature({"type": "Point", "coordinates": {"lon": 0, "lat": 0}}, name="X"),
        feature({"type": "Point", "coordinates": [None, 0]}, name="X"),
        feature({"type": "Point", "coordinates": ["east", 0]}, name="X"),
        feature({"type": "Point", "coordinates": [[0], 0]}, name="X"),
        feature({"type": "Point", "coordinates": [math.nan, 0]}, name="X"),
        feature({"type": "Point", "coordinates": [0, math.inf]}, name="X"),
        place(0, 0),
        place(0, 0, name=None, nameascii=""),
    ],
)
def test_load_places_skips_malformed_features(tmp_path, bad):
    places = load_places(tmp_path, bad, place(1, 2, name="Good"))
    assert [entry[2] for entry in places] == ["Good"]


def test_load_places_stringifies_odd_names_and_countries(tmp_path):
    [entry] = load_places(tmp_path, place(0, 0, name=12345, adm0name=7))
    assert (entry[2], entry[5]) == ("12345", "7")


# ---- download_map: cache, offline, network --------------------------------------------


def test_cached_file_is_used_without_touching_the_network(isolated, fake_http, online):
    target = isolated / "countries.geojson"
    write_geojson(target, collection())
    assert mapdata.download_map("countries.geojson") == target
    assert fake_http.calls == []


def test_cached_file_marks_its_progress_ready(isolated):
    write_geojson(isolated / mapdata.MAP_FILE, collection())
    mapdata.download_map(mapdata.MAP_FILE)
    entry = progress(mapdata.MAP_FILE)
    assert entry.state == "ready" and entry.downloaded == (isolated / mapdata.MAP_FILE).stat().st_size


def test_offline_without_a_cache_fails_without_a_request(isolated, fake_http):
    assert mapdata.FETCH_NEW_DATA_FROM_API is False
    with pytest.raises(MapDataError, match="offline mode"):
        mapdata.download_map("countries.geojson")
    assert fake_http.calls == []
    assert isolated.is_dir()  # cache dir is created even so


def test_download_writes_the_file_and_reports_progress(isolated, fake_http, online):
    body = json.dumps(collection(feature(polygon(SQUARE)))).encode()
    fake_http.responses[mapdata.NATURAL_EARTH_BASE_URL] = body
    path = mapdata.download_map(mapdata.MAP_FILE)
    assert fake_http.calls == [mapdata.NATURAL_EARTH_BASE_URL + mapdata.MAP_FILE]
    assert path == isolated / mapdata.MAP_FILE and path.read_bytes() == body
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    entry = progress(mapdata.MAP_FILE)
    assert (entry.state, entry.downloaded, entry.total) == ("ready", len(body), None)


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("unreachable"),
        urllib.error.HTTPError("https://x", 404, "Not Found", {}, None),
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
    ],
    ids=["url-error", "http-404", "timeout", "reset"],
)
def test_download_failure_raises_and_leaves_nothing_behind(isolated, fake_http, online, error):
    fake_http.responses[mapdata.NATURAL_EARTH_BASE_URL] = error
    with pytest.raises(MapDataError, match=f"Could not download {mapdata.MAP_FILE}"):
        mapdata.download_map(mapdata.MAP_FILE)
    assert list(isolated.iterdir()) == []
    entry = progress(mapdata.MAP_FILE)
    assert entry.state == "failed" and entry.error


class StreamingResponse:
    """A urlopen() result with headers and a body that can fail part-way."""

    def __init__(self, body: bytes, headers: dict, fail_after: int | None = None) -> None:
        self.stream = io.BytesIO(body)
        self.headers = headers
        self.fail_after = fail_after
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size: int = -1) -> bytes:
        if self.fail_after is not None and self.reads >= self.fail_after:
            raise ConnectionResetError("connection dropped")
        self.reads += 1
        return self.stream.read(size)


def serve(monkeypatch, response: StreamingResponse) -> list:
    requests = []

    def urlopen(request, timeout=None):
        requests.append((request, timeout))
        return response

    monkeypatch.setattr(mapdata.urllib.request, "urlopen", urlopen)
    return requests


def test_download_streams_in_chunks_and_records_the_content_length(isolated, monkeypatch, online):
    body = b"x" * (150 * 1024)
    response = StreamingResponse(body, {"Content-Length": str(len(body))})
    requests = serve(monkeypatch, response)
    path = mapdata.download_map(mapdata.PLACES_FILE)
    assert path.read_bytes() == body
    assert response.reads == 4  # 64K + 64K + 22K + the empty read that ends it
    entry = progress(mapdata.PLACES_FILE)
    assert (entry.state, entry.downloaded, entry.total) == ("ready", len(body), len(body))
    request, timeout = requests[0]
    assert timeout == mapdata.MAP_DOWNLOAD_TIMEOUT_SECONDS
    assert request.get_header("User-agent") == mapdata.USER_AGENT
    assert request.get_header("Accept") == "application/json"


@pytest.mark.parametrize("length", ["unknown", "-1", ""])
def test_download_ignores_a_bogus_content_length(isolated, monkeypatch, online, length):
    serve(monkeypatch, StreamingResponse(b"{}", {"Content-Length": length}))
    mapdata.download_map(mapdata.PLACES_FILE)
    assert progress(mapdata.PLACES_FILE).total is None


def test_download_dropped_mid_stream_removes_the_partial_file(isolated, monkeypatch, online):
    serve(monkeypatch, StreamingResponse(b"y" * (200 * 1024), {}, fail_after=1))
    with pytest.raises(MapDataError, match="connection dropped"):
        mapdata.download_map(mapdata.PLACES_FILE)
    assert list(isolated.iterdir()) == []
    assert progress(mapdata.PLACES_FILE).state == "failed"


def test_one_lock_per_file():
    assert mapdata._download_lock("a.geojson") is mapdata._download_lock("a.geojson")
    assert mapdata._download_lock("a.geojson") is not mapdata._download_lock("b.geojson")


# ---- _load_with_retry via the public loaders --------------------------------------------


def test_loaders_read_their_cached_files(isolated, monkeypatch):
    write_geojson(isolated / mapdata.MAP_FILE, collection(feature(polygon(SQUARE))))
    write_geojson(isolated / mapdata.PROVINCE_FILE, collection(
        feature({"type": "LineString", "coordinates": [[0, 0], [1, 1]]})
    ))
    write_geojson(isolated / mapdata.PLACES_FILE, collection(place(1, 2, name="P")))
    assert [ring[1:] for ring in mapdata.load_map_data()] == [(0, 10, 0, 5)]
    assert len(mapdata.load_province_lines()) == 1
    assert mapdata.load_places() == [(1.0, 2.0, "P", False, 0, "")]


def test_corrupt_cache_is_replaced_by_one_fresh_download(isolated, fake_http, online):
    (isolated).mkdir(parents=True)
    (isolated / mapdata.MAP_FILE).write_text('{"type": "FeatureColl', encoding="utf-8")
    fake_http.responses[mapdata.NATURAL_EARTH_BASE_URL] = collection(feature(polygon(SQUARE)))
    rings = mapdata.load_map_data()
    assert [ring[1:] for ring in rings] == [(0, 10, 0, 5)]
    assert len(fake_http.calls) == 1
    assert json.loads((isolated / mapdata.MAP_FILE).read_text())["type"] == "FeatureCollection"


def test_corrupt_cache_offline_is_removed_and_reported(isolated, fake_http):
    isolated.mkdir(parents=True)
    (isolated / mapdata.PLACES_FILE).write_text("truncated", encoding="utf-8")
    with pytest.raises(MapDataError, match="offline mode"):
        mapdata.load_places()
    assert not (isolated / mapdata.PLACES_FILE).exists()
    assert fake_http.calls == []


def test_retry_gives_up_after_a_second_bad_download(isolated, fake_http, online):
    isolated.mkdir(parents=True)
    (isolated / mapdata.PROVINCE_FILE).write_text("[]", encoding="utf-8")
    fake_http.responses[mapdata.NATURAL_EARTH_BASE_URL] = b"<html>rate limited</html>"
    with pytest.raises(MapDataError, match="Could not parse"):
        mapdata.load_province_lines()
    assert len(fake_http.calls) == 1


# ---- progress reporting -------------------------------------------------------------------


def test_set_live_progress_keeps_errors_only_for_failures():
    mapdata.set_live_progress("planes", state="downloading", downloaded=3, total=10, error="ignored")
    entry = progress("api:planes")
    assert (entry.state, entry.downloaded, entry.total, entry.error) == ("downloading", 3, 10, "")
    mapdata.set_live_progress("planes", state="failed", error="HTTP 503")
    assert (entry.state, entry.error) == ("failed", "HTTP 503")


def test_set_live_progress_for_an_unknown_feed_is_ignored():
    mapdata.set_live_progress("submarines", state="ready")
    assert "api:submarines" not in mapdata._DOWNLOAD_PROGRESS


def test_download_progress_is_a_snapshot(isolated):
    snapshot = {item.filename: item for item in mapdata.download_progress()}
    snapshot["LIVE AIRCRAFT"].state = "tampered"
    snapshot[mapdata.MAP_FILE].state = "tampered"
    assert progress("api:planes").state != "tampered"
    assert progress(mapdata.MAP_FILE).state != "tampered"


def test_download_progress_reports_cached_files_as_ready(isolated):
    write_geojson(isolated / mapdata.MAP_FILE, collection())
    size = (isolated / mapdata.MAP_FILE).stat().st_size
    progress(mapdata.MAP_FILE).state = "waiting"
    snapshot = {item.filename: item for item in mapdata.download_progress()}
    assert (snapshot[mapdata.MAP_FILE].state, snapshot[mapdata.MAP_FILE].total) == ("ready", size)
    # ...unless a download of it is under way or just failed.
    progress(mapdata.MAP_FILE).state = "failed"
    snapshot = {item.filename: item for item in mapdata.download_progress()}
    assert snapshot[mapdata.MAP_FILE].state == "failed"


def test_download_progress_hides_optional_files_nobody_asked_for():
    entry = progress(mapdata.PROVINCE_FILE)
    entry.optional, entry.state = True, "waiting"
    names = [item.filename for item in mapdata.download_progress()]
    assert mapdata.PROVINCE_FILE not in names
    entry.state = "downloading"
    names = [item.filename for item in mapdata.download_progress()]
    assert mapdata.PROVINCE_FILE in names


def set_states(**states):
    keys = {
        "map": mapdata.MAP_FILE, "places": mapdata.PLACES_FILE,
        "planes": "api:planes", "radar": "api:radar",
        "weather": "api:weather", "history": "api:radar-history",
    }
    for name, state in states.items():
        progress(keys[name]).state = state


def test_startup_progress_points_at_the_first_unfinished_step():
    set_states(map="ready", places="downloading", planes="waiting", radar="waiting",
               weather="waiting", history="waiting")
    entry = progress(mapdata.PLACES_FILE)
    entry.downloaded, entry.total = 50, 100
    # Step 2 of 6, half done: (1 + 0.5) / 6.
    assert mapdata.startup_progress() == (mapdata.PLACES_FILE, 25, "downloading", "")


def test_startup_progress_without_a_known_size_counts_the_step_as_started():
    set_states(map="ready", places="ready", planes="ready", radar="downloading",
               weather="waiting", history="waiting")
    progress("api:radar").total = None
    assert mapdata.startup_progress() == ("WEATHER RADAR", 50, "downloading", "")


def test_startup_progress_surfaces_a_failure():
    set_states(map="ready", places="ready", planes="failed", radar="ready",
               weather="ready", history="ready")
    progress("api:planes").error = "HTTP 503"
    assert mapdata.startup_progress() == ("LIVE AIRCRAFT", 33, "failed", "HTTP 503")


def test_startup_progress_complete_when_every_step_is_ready_or_disabled():
    set_states(map="ready", places="ready", planes="ready", radar="disabled",
               weather="ready", history="disabled")
    assert mapdata.startup_progress() == ("SETUP COMPLETE", 100, "ready", "")


def test_startup_progress_with_no_steps_is_complete(monkeypatch):
    monkeypatch.setattr(mapdata, "STARTUP_PROGRESS_KEYS", ())
    assert mapdata.startup_progress() == ("SETUP COMPLETE", 100, "ready", "")


# ---- bundled reference data --------------------------------------------------------------


def test_country_labels_apply_code_overrides():
    labels = mapdata.load_country_labels()
    assert len(labels) > 200
    by_name = {label[4]: label for label in labels}
    assert by_name["United Kingdom"][2] == "UK"  # GB -> UK override
    assert by_name["Zimbabwe"][2] == "ZW"
    for lon, lat, code, size, name in labels:
        assert isinstance(lon, float) and isinstance(size, float) and isinstance(name, str)
        assert -180 <= lon <= 180 and -90 <= lat <= 90 and code


def test_airports_are_typed_tuples():
    airports = mapdata.load_airports()
    assert len(airports) > 200
    codes = {airport[3] for airport in airports}
    assert {"HKG", "TPE"} <= codes
    for lon, lat, name, code, rank in airports:
        assert isinstance(lon, float) and isinstance(rank, int) and name


@pytest.fixture
def fresh_reference_cache():
    mapdata._reference_data.cache_clear()
    yield
    mapdata._reference_data.cache_clear()


@pytest.mark.parametrize("content", [None, b"{oops", b"\xff\xfe\xfa"], ids=["missing", "json", "utf8"])
def test_unreadable_reference_data_raises_map_data_error(tmp_path, monkeypatch, fresh_reference_cache, content):
    path = tmp_path / "reference_data.json"
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setattr(mapdata, "REFERENCE_DATA_FILE", path)
    with pytest.raises(MapDataError, match="bundled reference data"):
        mapdata.load_airports()
