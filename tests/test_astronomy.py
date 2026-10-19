"""Solar position, sunrise / sunset and moon phase against published reference values."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from terrascope.layers.daynight.astronomy import (
    MOON_EPOCH,
    SYNODIC_MONTH_DAYS,
    format_hemisphere,
    hhmm,
    julian_day,
    moon_info,
    solar_position,
    sun_elevation,
    sun_phase_name,
    sunrise_sunset,
)

UTC = timezone.utc


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def minutes_apart(a: float, b: float) -> float:
    """Distance between two times of day in minutes, across midnight."""
    difference = abs(a - b) % 1440
    return min(difference, 1440 - difference)


# ---- Julian Day ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (utc(2000, 1, 1, 12), 2451545.0),  # J2000.0
        (utc(1957, 10, 4, 19, 26, 24), 2436116.31),  # Meeus example 7.a (Sputnik 1)
        (utc(1987, 1, 27), 2446822.5),  # Meeus table 7.a, January (month <= 2 branch)
        (utc(1988, 6, 19, 12), 2447332.0),  # Meeus table 7.a
        (utc(2024, 2, 29), 2460369.5),  # leap day
    ],
)
def test_julian_day_matches_meeus_reference_dates(moment, expected):
    assert julian_day(moment) == pytest.approx(expected, abs=1e-6)


# ---- solar position -----------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "declination_degrees"),
    [
        (utc(2024, 3, 20, 3, 6), 0.0),  # March equinox
        (utc(2024, 6, 20, 20, 51), 23.44),  # June solstice
        (utc(2024, 9, 22, 12, 44), 0.0),  # September equinox
        (utc(2024, 12, 21, 9, 20), -23.44),  # December solstice
    ],
)
def test_solar_declination_at_equinoxes_and_solstices(moment, declination_degrees):
    declination, _ = solar_position(moment)
    assert math.degrees(declination) == pytest.approx(declination_degrees, abs=0.02)


@pytest.mark.parametrize(
    ("moment", "eq_time_minutes"),
    [
        (utc(2024, 11, 3, 12), 16.4),  # yearly maximum, sundial fast
        (utc(2024, 2, 11, 12), -14.2),  # yearly minimum, sundial slow
        (utc(2024, 4, 15, 12), 0.0),  # one of the four zero crossings
    ],
)
def test_equation_of_time_extremes_and_zero_crossing(moment, eq_time_minutes):
    _, eq_time = solar_position(moment)
    assert eq_time == pytest.approx(eq_time_minutes, abs=0.5)


def test_sun_is_overhead_at_the_subsolar_point_and_opposite_at_the_antipode():
    moment = utc(2024, 3, 20, 12)
    declination, eq_time = solar_position(moment)
    subsolar_lat = math.degrees(declination)
    subsolar_lon = (720 - 12 * 60 - eq_time) / 4
    assert sun_elevation(subsolar_lat, subsolar_lon, moment, declination, eq_time) == pytest.approx(90, abs=0.1)
    assert sun_elevation(-subsolar_lat, subsolar_lon + 180, moment, declination, eq_time) == pytest.approx(-90, abs=0.1)
    # A quarter of the way round, the sun is on the horizon.
    assert sun_elevation(0, subsolar_lon + 90, moment, declination, eq_time) == pytest.approx(0, abs=0.1)


def test_sun_elevation_at_the_pole_equals_declination():
    moment = utc(2024, 6, 20, 20, 51)
    declination, eq_time = solar_position(moment)
    for lon in (-120, 0, 75):
        assert sun_elevation(90, lon, moment, declination, eq_time) == pytest.approx(math.degrees(declination), abs=1e-6)


# ---- sunrise / sunset ---------------------------------------------------------


@pytest.mark.parametrize(
    ("place", "lat", "lon", "moment", "rise_utc", "set_utc"),
    [
        # NOAA solar calculator: London 04:43 / 21:21 BST
        ("London summer solstice", 51.5074, -0.1278, utc(2024, 6, 21, 12), "03:43", "20:21"),
        # New York 07:16 / 16:32 EST
        ("New York winter solstice", 40.7128, -74.006, utc(2024, 12, 21, 17), "12:16", "21:32"),
        # Los Angeles 05:42 / 20:08 PDT: sunset falls after midnight UTC
        ("Los Angeles summer solstice", 34.05, -118.24, utc(2024, 6, 21, 19), "12:42", "03:08"),
        # Sydney 07:00 / 16:54 AEST on the June solstice (southern winter)
        ("Sydney winter solstice", -33.87, 151.21, utc(2024, 6, 21, 2), "21:00", "06:54"),
    ],
)
def test_sunrise_and_sunset_match_noaa_within_two_minutes(place, lat, lon, moment, rise_utc, set_utc):
    declination, eq_time = solar_position(moment)
    rise, sunset = sunrise_sunset(lat, lon, declination, eq_time)
    expected_rise = int(rise_utc[:2]) * 60 + int(rise_utc[3:])
    expected_set = int(set_utc[:2]) * 60 + int(set_utc[3:])
    assert minutes_apart(rise, expected_rise) <= 2, place
    assert minutes_apart(sunset, expected_set) <= 2, place
    assert 0 <= rise < 1440 and 0 <= sunset < 1440  # wrapped into one UTC day


def test_equator_day_is_a_little_over_twelve_hours_at_the_equinox():
    declination, eq_time = solar_position(utc(2024, 3, 20, 12))
    rise, sunset = sunrise_sunset(0.0, 0.0, declination, eq_time)
    day_length = (sunset - rise) % 1440
    # Refraction plus the sun's radius add about 6-7 minutes to a 12 h day.
    assert 720 < day_length < 735


@pytest.mark.parametrize(
    ("lat", "moment", "expected"),
    [
        (69.65, utc(2024, 6, 21, 12), "polar day"),  # Tromsø, midnight sun
        (69.65, utc(2024, 12, 21, 12), "polar night"),  # Tromsø, polar night
        (-77.85, utc(2024, 12, 21, 12), "polar day"),  # McMurdo, southern summer
        (-77.85, utc(2024, 6, 21, 12), "polar night"),
        (90.0, utc(2024, 6, 21, 12), "polar day"),  # exactly at the pole
        (-90.0, utc(2024, 6, 21, 12), "polar night"),
    ],
)
def test_polar_day_and_night_have_no_sunrise(lat, moment, expected):
    declination, eq_time = solar_position(moment)
    assert sunrise_sunset(lat, 18.95, declination, eq_time) == expected


def test_sunrise_and_sunset_are_symmetric_about_solar_noon():
    declination, eq_time = solar_position(utc(2024, 6, 21, 12))
    rise, sunset = sunrise_sunset(45.0, 30.0, declination, eq_time)
    noon = 720 - 4 * 30.0 - eq_time
    assert (rise + sunset) / 2 == pytest.approx(noon, abs=1e-6)


# ---- named phases -------------------------------------------------------------


@pytest.mark.parametrize(
    ("elevation", "name"),
    [
        (45.0, "Day"),
        (0.0, "Day"),
        (-0.01, "Civil twilight"),
        (-6.0, "Civil twilight"),
        (-6.01, "Nautical twilight"),
        (-12.0, "Nautical twilight"),
        (-17.99, "Astronomical twilight"),
        (-18.0, "Astronomical twilight"),
        (-18.01, "Night"),
        (-90.0, "Night"),
    ],
)
def test_sun_phase_name_thresholds(elevation, name):
    assert sun_phase_name(elevation) == name


# ---- moon -----------------------------------------------------------------------


def test_moon_epoch_is_a_new_moon():
    moon = moon_info(MOON_EPOCH)
    assert moon["name"] == "New Moon"
    assert moon["age"] == pytest.approx(0)
    assert moon["illumination"] == pytest.approx(0)
    assert moon["to_new"] == pytest.approx(SYNODIC_MONTH_DAYS)
    assert moon["to_full"] == pytest.approx(SYNODIC_MONTH_DAYS / 2)


@pytest.mark.parametrize(
    ("moment", "name", "min_lit", "max_lit"),
    [
        (utc(2024, 1, 25, 17, 54), "Full Moon", 0.97, 1.0),
        (utc(2024, 4, 23, 23, 49), "Full Moon", 0.97, 1.0),
        (utc(2024, 4, 8, 18, 21), "New Moon", 0.0, 0.03),  # total solar eclipse
        (utc(2024, 10, 2, 18, 49), "New Moon", 0.0, 0.03),  # annular solar eclipse
        (utc(2024, 4, 15, 19, 13), "First Quarter", 0.35, 0.65),
        (utc(2024, 5, 1, 11, 27), "Last Quarter", 0.35, 0.65),
    ],
)
def test_moon_phase_on_known_2024_dates(moment, name, min_lit, max_lit):
    moon = moon_info(moment)
    assert moon["name"] == name
    assert min_lit <= moon["illumination"] <= max_lit


def test_moon_countdowns_are_consistent_with_age():
    moon = moon_info(utc(2024, 7, 4, 9))
    assert 0 <= moon["age"] < SYNODIC_MONTH_DAYS
    assert moon["age"] + moon["to_new"] == pytest.approx(SYNODIC_MONTH_DAYS)
    assert 0 <= moon["to_full"] < SYNODIC_MONTH_DAYS
    assert (moon["age"] + moon["to_full"]) % SYNODIC_MONTH_DAYS == pytest.approx(SYNODIC_MONTH_DAYS / 2)


def test_moon_before_the_epoch_still_has_a_positive_age():
    moon = moon_info(utc(1999, 12, 22, 17, 31))  # solstice full moon, two weeks before the epoch
    assert 0 <= moon["age"] < SYNODIC_MONTH_DAYS
    assert moon["name"] == "Full Moon"
    assert moon["illumination"] > 0.97


# ---- formatting -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("minutes", "text"),
    [
        (0, "00:00"),
        (754.4, "12:34"),
        (754.5, "12:34"),  # round half to even
        (1439.4, "23:59"),
        (1439.6, "00:00"),  # rounds up past midnight and wraps
        (-1, "23:59"),
        (1500, "01:00"),
    ],
)
def test_hhmm_rounds_and_wraps(minutes, text):
    assert hhmm(minutes) == text


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (51.5074, "51.5°N"),
        (0.0, "0.0°N"),
        (-33.87, "33.9°S"),
        (-0.04, "0.0°S"),
    ],
)
def test_format_hemisphere(value, text):
    assert format_hemisphere(value, "N", "S") == text
