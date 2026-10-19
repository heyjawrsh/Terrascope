"""Aircraft formatting, categorisation and position maths in layers/flights/formatting.py."""

from __future__ import annotations

import math

import pytest

from terrascope.layers.flights import formatting
from terrascope.layers.flights.formatting import (
    airport_scalerank_floor,
    categorize,
    distance_between_points,
    finite_float,
    format_altitude,
    format_heading,
    format_position_source,
    format_speed,
    format_vertical_rate,
    heading_to_arrow,
    live_position,
    position_at,
    valid_history_point,
)
from terrascope.layers.flights.settings import (
    ARROW_GLYPHS,
    EARTH_RADIUS_M,
    MAX_EXTRAPOLATE_AGE_SECONDS,
    UNKNOWN_HEADING_GLYPH,
)

NAN = float("nan")
INF = float("inf")


def point(lon, lat, timestamp):
    return {"longitude": lon, "latitude": lat, "timestamp": timestamp}


# ---- finite_float / valid_history_point -----------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (2.5, 2.5),
        ("2.5", 2.5),
        (" -3 ", -3.0),
        (0, 0.0),
        (True, 1.0),  # bools are ints to float(); documented, not rejected
        (None, None),
        ("", None),
        ("abc", None),
        ([], None),
        ({}, None),
        (NAN, None),
        (INF, None),
        (-INF, None),
        ("nan", None),
        ("inf", None),
    ],
)
def test_finite_float_accepts_numbers_and_rejects_garbage(value, expected):
    assert finite_float(value) == expected


@pytest.mark.parametrize(
    "candidate",
    [
        point(0, 0, 1),
        point(180, 90, 1),
        point(-180, -90, 1),
        point("10.5", "-20", 1),
        {"longitude": 1, "latitude": 2},  # timestamp is not this function's concern
        point(1, 2, "not a time"),
    ],
)
def test_valid_history_point_accepts_in_range_coordinates(candidate):
    assert valid_history_point(candidate) is True


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        [0, 0],
        "0,0",
        {},
        {"longitude": 0},
        {"latitude": 0},
        point(None, 0, 1),
        point(0, None, 1),
        point("east", 0, 1),
        point(NAN, 0, 1),
        point(0, INF, 1),
        point(180.0001, 0, 1),
        point(-181, 0, 1),
        point(0, 90.5, 1),
        point(0, -91, 1),
        point([1], 0, 1),
    ],
)
def test_valid_history_point_rejects_malformed_points(candidate):
    assert valid_history_point(candidate) is False


def test_distance_between_points_is_planar_degrees_and_coerces_strings():
    assert distance_between_points(point(0, 0, 0), point(3, 4, 0)) == pytest.approx(5.0)
    assert distance_between_points(point("1", "1", 0), point(1, 1, 0)) == 0.0


# ---- airport zoom levels ------------------------------------------------------------


@pytest.mark.parametrize(
    ("zoom", "expected"),
    [(0.5, None), (0.999, None), (1.0, 2), (2.9, 2), (3.0, 4), (5.0, 6), (7.99, 6), (8.0, 8), (12.0, 10), (500.0, 10)],
)
def test_airport_scalerank_floor_follows_configured_levels(zoom, expected):
    assert airport_scalerank_floor(zoom) == expected


def test_airport_scalerank_floor_uses_the_last_level_reached(monkeypatch):
    monkeypatch.setattr(formatting, "AIRPORT_LEVELS", ((2.0, 3), (4.0, 7)))
    assert airport_scalerank_floor(1.0) is None
    assert airport_scalerank_floor(3.0) == 3
    assert airport_scalerank_floor(4.0) == 7


# ---- headings and units ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("track", "index"),
    [
        (0, 0),
        (22.49, 0),
        (22.5, 1),
        (45, 1),
        (90, 2),
        (135, 3),
        (180, 4),
        (225, 5),
        (270, 6),
        (315, 7),
        (337.49, 7),
        (337.5, 0),
        (359.9, 0),
        (360, 0),
        (720 + 90, 2),
        (-90, 6),
        (-1, 0),
        ("90", 2),
    ],
)
def test_heading_to_arrow_picks_the_nearest_of_eight_directions(track, index):
    assert heading_to_arrow(track) == ARROW_GLYPHS[index]


@pytest.mark.parametrize("track", [None, "", "north", NAN, INF, -INF, [], {}])
def test_heading_to_arrow_falls_back_for_unknown_headings(track):
    assert heading_to_arrow(track) == UNKNOWN_HEADING_GLYPH


@pytest.mark.parametrize(
    ("formatter", "value", "expected"),
    [
        (format_altitude, 1000, "3,281 ft"),
        (format_altitude, 0, "0 ft"),
        (format_altitude, "12000", "39,370 ft"),
        (format_altitude, -30, "-98 ft"),
        (format_speed, 100, "194 kt"),
        (format_speed, 0, "0 kt"),
        (format_speed, "250.5", "487 kt"),
        (format_vertical_rate, 5, "+984 ft/min"),
        (format_vertical_rate, -5, "-984 ft/min"),
        (format_vertical_rate, 0, "+0 ft/min"),
        (format_vertical_rate, 30, "+5,906 ft/min"),
        (format_heading, 90.4, "90°"),
        (format_heading, "359.6", "360°"),
        (format_heading, 0, "0°"),
    ],
)
def test_unit_formatters_convert_to_aviation_units(formatter, value, expected):
    assert formatter(value) == expected


@pytest.mark.parametrize("formatter", [format_altitude, format_speed, format_vertical_rate, format_heading])
@pytest.mark.parametrize("value", [None, "", "high", NAN, INF, [], {}])
def test_unit_formatters_show_not_available_for_garbage(formatter, value):
    assert formatter(value) == "N/A"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "ADS-B"),
        (1, "ASTERIX"),
        (2, "MLAT"),
        (3, "FLARM"),
        ("2", "MLAT"),
        (2.9, "MLAT"),
        (7, "7"),
        (-1, "-1"),
        (None, "N/A"),
        ("radar", "N/A"),
        ([], "N/A"),
    ],
)
def test_format_position_source_names_known_sources(value, expected):
    assert format_position_source(value) == expected


# ---- categorize ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flight", "expected"),
    [
        ({"squawk": "7700", "category": 4}, "emergency"),
        ({"squawk": "7600"}, "emergency"),
        ({"squawk": "7500", "category": 8}, "emergency"),
        ({"squawk": 7700}, "emergency"),  # numeric squawk still matches
        ({"squawk": "07700", "category": 4}, "airliner"),
        ({"squawk": "1200", "category": 2}, "light"),
        ({"category": 12}, "light"),
        ({"category": 3}, "regional"),
        ({"category": 4}, "airliner"),
        ({"category": 5}, "heavy"),
        ({"category": 6}, "heavy"),
        ({"category": 7}, "fast"),
        ({"category": 8}, "helicopter"),
        ({"category": 9}, "other"),
        ({"category": 14}, "other"),
        ({"category": 5.0}, "heavy"),
    ],
)
def test_categorize_prefers_emergency_squawk_then_category_code(flight, expected):
    assert categorize(flight) == expected


@pytest.mark.parametrize(
    ("flight", "expected"),
    [
        ({}, "other"),
        ({"category": 0, "velocity": None}, "other"),
        ({"category": 1, "velocity": "fast"}, "other"),
        ({"velocity": NAN}, "other"),
        ({"category": "4", "velocity": 50}, "light"),  # string codes are not looked up
        ({"velocity": 94.9}, "light"),
        ({"velocity": 95.0}, "regional"),
        ({"velocity": 120, "baro_altitude": 3000}, "regional"),
        ({"velocity": 120, "baro_altitude": None}, "regional"),
        ({"velocity": 120, "baro_altitude": "garbage"}, "regional"),
        ({"velocity": 120, "baro_altitude": 7000}, "airliner"),
        ({"velocity": 120, "baro_altitude": 11000}, "airliner"),
        ({"velocity": 150.0}, "airliner"),
        ({"velocity": "230"}, "airliner"),
    ],
)
def test_categorize_guesses_from_speed_and_altitude_without_a_code(flight, expected):
    assert categorize(flight) == expected


@pytest.mark.xfail(
    strict=True,
    reason="BUG: categorize() raises TypeError when 'category' is unhashable (list/dict from a malformed feed or cache)",
)
@pytest.mark.parametrize("category", [[4], {"code": 4}])
def test_categorize_survives_an_unhashable_category(category):
    assert categorize({"category": category, "velocity": 200}) == "airliner"


# ---- live_position (dead reckoning) ---------------------------------------------------


def moving(lon=0.0, lat=0.0, velocity=100.0, track=90.0, time_position=1000.0, last_contact=None):
    return {
        "longitude": lon,
        "latitude": lat,
        "velocity": velocity,
        "true_track": track,
        "time_position": time_position,
        "last_contact": last_contact,
    }


@pytest.mark.parametrize(
    "flight",
    [
        moving(velocity=None),
        moving(velocity=0),
        moving(velocity=-50),
        moving(velocity="fast"),
        moving(track=None),
        moving(track=NAN),
        moving(time_position=None, last_contact=None),
        moving(time_position="soon", last_contact="later"),
    ],
)
def test_live_position_stays_put_without_usable_motion(flight):
    assert live_position(flight, 5000.0) == (0.0, 0.0)


def test_live_position_stays_put_for_times_before_the_report():
    assert live_position(moving(), 900.0) == (0.0, 0.0)
    assert live_position(moving(), 1000.0) == (0.0, 0.0)


def test_live_position_moves_east_along_the_equator():
    lon, lat = live_position(moving(velocity=100.0, track=90.0), 1600.0)
    assert lon == pytest.approx(math.degrees(100.0 * 600 / EARTH_RADIUS_M))
    assert lat == pytest.approx(0.0, abs=1e-9)


def test_live_position_moves_north_on_heading_zero():
    lon, lat = live_position(moving(velocity=100.0, track=0.0), 1600.0)
    assert lon == pytest.approx(0.0, abs=1e-9)
    assert lat == pytest.approx(math.degrees(100.0 * 600 / EARTH_RADIUS_M))


def test_live_position_caps_extrapolation_age():
    capped = live_position(moving(), 1000.0 + MAX_EXTRAPOLATE_AGE_SECONDS)
    much_later = live_position(moving(), 1000.0 + MAX_EXTRAPOLATE_AGE_SECONDS * 50)
    assert much_later == pytest.approx(capped)
    assert capped[0] > 0


def test_live_position_falls_back_to_last_contact_when_time_position_missing():
    by_contact = live_position(moving(time_position=None, last_contact=1000.0), 1600.0)
    by_position = live_position(moving(time_position=1000.0), 1600.0)
    assert by_contact == pytest.approx(by_position)


def test_live_position_wraps_across_the_antimeridian():
    lon, lat = live_position(moving(lon=179.9, velocity=250.0, track=90.0), 1000.0 + 600)
    travelled = math.degrees(250.0 * 600 / EARTH_RADIUS_M)
    assert lon == pytest.approx(179.9 + travelled - 360.0)
    assert lat == pytest.approx(0.0, abs=1e-9)


# ---- position_at (history interpolation) ------------------------------------------------


def stationary(lon, lat):
    return {"longitude": lon, "latitude": lat, "velocity": None, "true_track": None}


def test_position_at_without_history_uses_the_live_position():
    assert position_at(stationary(5.0, 6.0), [], 0.0) == (5.0, 6.0)


def test_position_at_before_tracking_started_is_none():
    history = [point(0.0, 0.0, 100.0), point(10.0, 10.0, 200.0)]
    assert position_at(stationary(10.0, 10.0), history, 99.9) is None


def test_position_at_after_the_last_point_uses_the_live_position():
    history = [point(0.0, 0.0, 100.0), point(10.0, 10.0, 200.0)]
    assert position_at(stationary(11.0, 12.0), history, 200.0) == (11.0, 12.0)
    assert position_at(stationary(11.0, 12.0), history, 999.0) == (11.0, 12.0)


@pytest.mark.parametrize(
    ("at_time", "expected"),
    [(100.0, (0.0, 0.0)), (125.0, (2.5, -5.0)), (150.0, (5.0, -10.0)), (199.0, (9.9, -19.8))],
)
def test_position_at_interpolates_between_points(at_time, expected):
    history = [point(0.0, 0.0, 100.0), point(10.0, -20.0, 200.0)]
    assert position_at(stationary(10.0, -20.0), history, at_time) == pytest.approx(expected)


def test_position_at_picks_the_bracketing_segment():
    history = [point(0.0, 0.0, 0.0), point(10.0, 0.0, 10.0), point(10.0, 10.0, 20.0)]
    assert position_at(stationary(10.0, 10.0), history, 15.0) == pytest.approx((10.0, 5.0))


@pytest.mark.parametrize(
    ("fraction", "expected_lon"),
    [(0.25, 179.5), (0.75, -179.5)],
)
def test_position_at_interpolates_the_short_way_across_the_antimeridian(fraction, expected_lon):
    history = [point(179.0, 0.0, 0.0), point(-179.0, 0.0, 100.0)]
    lon, _ = position_at(stationary(-179.0, 0.0), history, 100.0 * fraction)
    assert lon == pytest.approx(expected_lon)


def test_position_at_midpoint_on_the_antimeridian_stays_in_range():
    history = [point(179.0, 0.0, 0.0), point(-179.0, 0.0, 100.0)]
    lon, _ = position_at(stationary(-179.0, 0.0), history, 50.0)
    assert lon == pytest.approx(-180.0)


def test_position_at_ignores_points_with_bad_coordinates_or_timestamps():
    history = [
        point(0.0, 0.0, "yesterday"),
        point(0.0, 0.0, None),
        point(0.0, 0.0, NAN),
        point(NAN, 0.0, 50.0),
        point(500.0, 0.0, 60.0),
        "garbage",
        point(0.0, 0.0, 100.0),
        point(10.0, 0.0, 200.0),
    ]
    # Tracking starts at the first *valid* point (t=100), not the garbage at t=50.
    assert position_at(stationary(10.0, 0.0), history, 75.0) is None
    assert position_at(stationary(10.0, 0.0), history, 150.0) == pytest.approx((5.0, 0.0))


def test_position_at_with_only_invalid_history_falls_back_to_live_position():
    history = [point(0.0, 0.0, "bad"), {"longitude": "x"}]
    assert position_at(stationary(3.0, 4.0), history, 0.0) == (3.0, 4.0)


def test_position_at_accepts_numeric_string_timestamps():
    history = [point(0.0, 0.0, "100"), point(10.0, 0.0, "200")]
    assert position_at(stationary(10.0, 0.0), history, 150.0) == pytest.approx((5.0, 0.0))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: position_at() validates string coordinates with float() but then does arithmetic on the raw strings (TypeError)",
)
def test_position_at_interpolates_numeric_string_coordinates():
    history = [point("0", "0", 100.0), point("10", "0", 200.0)]
    assert position_at(stationary(10.0, 0.0), history, 150.0) == pytest.approx((5.0, 0.0))
