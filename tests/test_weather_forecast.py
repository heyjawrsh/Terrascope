"""Forecast panel building: icons, colour steps, population floors, city keys and line layout."""

from __future__ import annotations

import math

import pytest

from terrascope.layers.weather import forecast
from terrascope.layers.weather.config import (
    CITY_CONFIG,
    CITY_LEVELS,
    FORECAST_DAYS_AHEAD,
    FORECAST_MIN_COLUMN_WIDTH,
    FORECAST_TODAY_COLOR,
    FORECAST_WEEKEND_COLOR,
    WEATHER_ICONS,
)


def _text(pieces) -> str:
    return "".join(text for text, _color, _style in pieces)


def _width(pieces) -> int:
    return sum(len(text) for text, _color, _style in pieces)


def _day(date="2026-10-19", **overrides) -> dict:
    hours = [[hour, 10.0 + hour / 2, 1, 20, 12.0] for hour in range(24)]
    day = {
        "date": date,
        "code": 61,
        "tmin": 4.4,
        "tmax": 17.6,
        "precip": 2.25,
        "pop": 70,
        "wind": 18.0,
        "gust": 45.0,
        "sunrise": f"{date}T06:51",
        "sunset": f"{date}T18:02",
        "uv": 3.2,
        "hours": hours,
    }
    day.update(overrides)
    return day


# ---- city_population_floor / city_key / item_kind ---------------------------


def test_population_floor_is_none_below_the_first_level():
    first_zoom = CITY_LEVELS[0][0]
    assert forecast.city_population_floor(first_zoom - 0.01) is None
    assert forecast.city_population_floor(0) is None


def test_population_floor_takes_the_last_reached_level_at_each_boundary():
    for zoom, population in CITY_LEVELS:
        assert forecast.city_population_floor(zoom) == population


def test_population_floor_never_increases_with_zoom():
    floors = [forecast.city_population_floor(zoom / 4) for zoom in range(0, 80)]
    numeric = [floor for floor in floors if floor is not None]
    assert numeric == sorted(numeric, reverse=True)
    assert forecast.city_population_floor(1000) == CITY_LEVELS[-1][1]


def test_city_key_rounds_position_to_two_decimals():
    assert forecast.city_key("Tokyo", 35.6895, 139.6917) == "Tokyo|35.69|139.69"
    assert forecast.city_key("Victoria", -4.62, 55.45) != forecast.city_key("Victoria", 48.43, -123.37)


@pytest.mark.parametrize(
    "item, kind",
    [
        ({"radar": True, "magnitude": 5.0}, "radar"),
        ({"radar": False, "magnitude": None}, "quake"),
        ({"magnitude": 4.2}, "quake"),
        ({"name": "Paris"}, "city"),
        (("Paris",), "city"),
        (None, "city"),
    ],
)
def test_item_kind(item, kind):
    assert forecast.item_kind(item) == kind


# ---- weather categories and icons ---------------------------------------------


@pytest.mark.parametrize(
    "code, category",
    [
        (0, "clear"), (1, "clear"), (2, "partly"), (3, "cloud"),
        (45, "fog"), (48, "fog"),
        (50, "unknown"), (51, "drizzle"), (57, "drizzle"), (58, "unknown"),
        (61, "rain"), (67, "rain"), (80, "rain"), (82, "rain"), (83, "unknown"),
        (71, "snow"), (77, "snow"), (85, "snow"), (86, "snow"),
        (94, "unknown"), (95, "storm"), (99, "storm"),
        ("3", "cloud"), (2.9, "partly"), (-1, "unknown"),
        (None, "unknown"), ("rain", "unknown"), ([], "unknown"),
    ],
)
def test_weather_category(code, category):
    assert forecast.weather_category(code) == category


def test_weather_icon_falls_back_to_unknown_glyph():
    assert forecast.weather_icon(None) == WEATHER_ICONS["unknown"]
    assert forecast.weather_icon(0) == WEATHER_ICONS["clear"]


def test_weather_icon_falls_back_when_a_category_has_no_icon(monkeypatch):
    icons = {name: value for name, value in WEATHER_ICONS.items() if name != "storm"}
    monkeypatch.setattr(forecast, "WEATHER_ICONS", icons)
    assert forecast.weather_icon(95) == WEATHER_ICONS["unknown"]


def test_forecast_temp_color_uses_neutral_colour_when_missing():
    assert forecast.forecast_temp_color(None) == CITY_CONFIG["color"]
    assert forecast.forecast_temp_color(-100) == CITY_CONFIG["temperature_cold_color"].lower()
    assert forecast.forecast_temp_color(100) == CITY_CONFIG["temperature_hot_color"].lower()


@pytest.mark.parametrize(
    "value, steps, expected",
    [
        (0.0, forecast.PRECIP_STEPS, "#6b7785"),
        (0.1, forecast.PRECIP_STEPS, "#6cb4ee"),
        (0.99, forecast.PRECIP_STEPS, "#6cb4ee"),
        (1.0, forecast.PRECIP_STEPS, "#3f8cff"),
        (15.0, forecast.PRECIP_STEPS, "#c77dff"),
        (19.99, forecast.WIND_STEPS, "#6bcb77"),
        (20, forecast.WIND_STEPS, "#ffd93d"),
        (60, forecast.WIND_STEPS, "#ff4d4d"),
        (11, forecast.UV_STEPS, "#c77dff"),
        (math.inf, forecast.UV_STEPS, "#c77dff"),
    ],
)
def test_step_color_band_boundaries(value, steps, expected):
    assert forecast._step_color(value, steps) == expected


def test_step_color_past_every_limit_takes_the_last_colour():
    assert forecast._step_color(100, ((1, "#a"), (2, "#b"))) == "#b"


# ---- _fit -----------------------------------------------------------------------


def test_fit_pads_short_lines_and_cuts_long_ones():
    assert forecast._fit([("abc", "#fff", "bold")], 5) == [("abc", "#fff", "bold"), ("  ", None, "")]
    cut = forecast._fit([("abc", None, ""), ("defg", None, "dim"), ("h", None, "")], 5)
    assert _text(cut) == "abcde"
    assert cut[-1] == ("de", None, "dim")
    assert forecast._fit([], 3) == [("   ", None, "")]
    assert forecast._fit([("abc", None, "")], 0) == []


# ---- day rows -------------------------------------------------------------------


def test_day_header_today_and_weekend():
    today = forecast._day_header({"date": "2026-10-19"}, today=True)
    assert today[0] == ("TODAY ", FORECAST_TODAY_COLOR, "bold")
    assert today[1][0] == "Mon 10-19"
    saturday = forecast._day_header({"date": "2026-10-24"}, today=False)
    assert saturday == [("Sat 10-24", FORECAST_WEEKEND_COLOR, "bold")]
    weekday = forecast._day_header({"date": "2026-10-21"}, today=False)
    assert weekday == [("Wed 10-21", None, "bold")]


@pytest.mark.parametrize("value", [None, "", "tomorrow", "2026-13-40", 12345])
def test_day_header_survives_bad_dates(value):
    assert forecast._day_header({"date": value}, today=False) == [("?", None, "bold")]


def test_day_conditions_names_the_code():
    glyph, color = WEATHER_ICONS["rain"]
    assert forecast._day_conditions({"code": 63}) == [(glyph + " ", color, "bold"), ("Rain", color, "")]
    assert _text(forecast._day_conditions({})) == WEATHER_ICONS["unknown"][0] + " Unknown"


def test_temp_extremes():
    assert _text(forecast._temp_extremes({"tmin": -3.6, "tmax": 21.4})) == "▼ -4°  →  ▲ 21°"
    assert forecast._temp_extremes({"tmin": None, "tmax": 3}) == [("N/A", None, "dim")]
    assert forecast._temp_extremes({"tmax": 3}) == [("N/A", None, "dim")]


def test_sparkline_spans_the_requested_width_and_the_full_block_range():
    hours = [[hour, float(hour), 0, 0, 0] for hour in range(24)]
    line = forecast._temp_sparkline(hours, 16)
    assert len(line) == 16
    assert line[0][0] == forecast.SPARK_BLOCKS[0]
    assert line[-1][0] == forecast.SPARK_BLOCKS[-1]


def test_sparkline_flat_day_and_missing_temperatures():
    flat = forecast._temp_sparkline([[0, 5.0, 0, 0, 0], [1, 5.0, 0, 0, 0]], 4)
    assert _text(flat) == forecast.SPARK_BLOCKS[3] * 4
    assert forecast._temp_sparkline([[0, None, 0, 0, 0]], 10) is forecast.NO_DATA
    assert forecast._temp_sparkline([], 10) is forecast.NO_DATA
    assert len(forecast._temp_sparkline([[0, 1.0, 0, 0, 0], [1, 2.0, 0, 0, 0]], 0)) == 1


def test_sparkline_single_reading():
    assert len(forecast._temp_sparkline([[0, 12.0, 0, 0, 0]], 5)) == 5


def test_precip_row_variants():
    assert _text(forecast._precip_row({"precip": 0.0, "pop": 5})) == "☂ dry  5%"
    assert _text(forecast._precip_row({"precip": 2.25, "pop": None})) == "☂ 2.2 mm"
    assert _text(forecast._precip_row({"precip": None})) == "☂ N/A"


def test_wind_row_colours_by_gust_when_known():
    with_gust = forecast._wind_row({"wind": 10, "gust": 65})
    assert _text(with_gust) == "≈ 10 km/h  gust 65"
    assert with_gust[0][1] == "#ff4d4d"
    calm = forecast._wind_row({"wind": 10, "gust": None})
    assert _text(calm) == "≈ 10 km/h"
    assert calm[0][1] == "#6bcb77"
    assert _text(forecast._wind_row({"wind": None, "gust": 80})) == "≈ N/A"


def test_sun_row_variants():
    full = forecast._sun_row({"sunrise": "2026-10-19T06:51", "sunset": "2026-10-19T18:02", "uv": 6.4})
    assert _text(full) == "↑06:51 ↓18:02  UV 6"
    assert full[-1][1] == "#ff9f43"
    assert _text(forecast._sun_row({"sunset": "2026-10-19T18:02"})) == "↓18:02"
    assert _text(forecast._sun_row({"uv": 0})) == "  UV 0"
    assert forecast._sun_row({}) is forecast.NO_DATA
    assert forecast._sun_row({"sunrise": "short", "sunset": None}) is forecast.NO_DATA


def test_day_rows_ignores_malformed_hour_rows():
    day = _day(hours=[[0, 1.0], "junk", None, [1, 2.0, 0, 0, 0], [2, 9.0, 0, 0, 0]])
    rows = forecast._day_rows(day, today=False, cell=6)
    assert len(rows) == forecast.FORECAST_ROWS
    assert len(rows[3]) == 6  # two valid hours drive the sparkline
    assert forecast._day_rows(_day(hours=None), False, 6)[3] is forecast.NO_DATA


# ---- forecast_lines -------------------------------------------------------------


def test_forecast_lines_fill_the_width_exactly_with_one_column_per_fitting_day():
    days = [_day(f"2026-10-{19 + index}") for index in range(FORECAST_DAYS_AHEAD + 1)]
    gap = len(forecast.FORECAST_SEPARATOR)
    width = 3 * FORECAST_MIN_COLUMN_WIDTH + 2 * gap + 4  # room for three columns only
    lines = forecast.forecast_lines(days, width)
    assert len(lines) == forecast.FORECAST_ROWS
    for line in lines:
        assert _width(line) == width
        assert sum(text == forecast.FORECAST_SEPARATOR for text, _c, _s in line) == 2
    assert _text(lines[0]).startswith("TODAY Mon 10-19")
    assert "Wed 10-21" in _text(lines[0])
    assert "Thu 10-22" not in _text(lines[0])


def test_forecast_lines_always_show_at_least_one_day_even_when_narrow():
    lines = forecast.forecast_lines([_day(), _day("2026-10-20")], 5)
    assert all(_width(line) == 5 for line in lines)
    assert all(forecast.FORECAST_SEPARATOR not in _text(line) for line in lines)


def test_forecast_lines_survive_missing_fields():
    lines = forecast.forecast_lines([{"date": None}], 40)
    assert len(lines) == forecast.FORECAST_ROWS
    assert all(_width(line) == 40 for line in lines)
    assert "N/A" in _text(lines[2])


@pytest.mark.xfail(
    strict=True,
    reason="BUG: parse_forecast stores feed values uncoerced, so non-numeric/NaN forecast values crash forecast_lines",
)
@pytest.mark.parametrize(
    "overrides",
    [
        {"tmin": "4", "tmax": "17"},
        {"precip": "2.5"},
        {"hours": [[0, 1.0, 0, 0, 0], [1, float("nan"), 0, 0, 0], [2, 3.0, 0, 0, 0]]},
    ],
    ids=["string-temperatures", "string-precip", "nan-hourly-temp"],
)
def test_forecast_lines_tolerate_non_numeric_feed_values(overrides):
    lines = forecast.forecast_lines([_day(**overrides)], 40)
    assert len(lines) == forecast.FORECAST_ROWS


# ---- vertical layout and placeholders -------------------------------------------


def test_vertical_lines_stack_days_with_separators_and_scroll():
    days = [_day("2026-10-19"), _day("2026-10-20"), _day("2026-10-21")]
    total = 3 * forecast.FORECAST_ROWS + 2 * forecast.VERTICAL_DAY_SEPARATOR_ROWS
    everything = forecast.forecast_vertical_lines(days, 30, 1000)
    assert len(everything) == total
    separator = everything[forecast.FORECAST_ROWS]
    assert separator == [("─" * 30, None, "dim")]
    assert everything[forecast.FORECAST_ROWS + 1] == []
    window = forecast.forecast_vertical_lines(days, 30, 5, scroll=forecast.FORECAST_ROWS)
    assert window == everything[forecast.FORECAST_ROWS : forecast.FORECAST_ROWS + 5]


@pytest.mark.parametrize("scroll, start", [(-10, 0), (10_000, None)])
def test_vertical_lines_clamp_scroll(scroll, start):
    days = [_day("2026-10-19"), _day("2026-10-20")]
    everything = forecast.forecast_vertical_lines(days, 30, 1000)
    window = forecast.forecast_vertical_lines(days, 30, 4, scroll=scroll)
    expected_start = len(everything) - 4 if start is None else start
    assert window == everything[expected_start : expected_start + 4]


def test_vertical_lines_empty_inputs():
    assert forecast.forecast_vertical_lines([], 30, 10) == []
    assert forecast.forecast_vertical_lines([_day()], 30, 0) == []


def test_placeholder_lists_every_icon_and_the_temperature_scale():
    lines = forecast.forecast_placeholder()
    text = "\n".join(_text(line) for line in lines)
    assert "CLICK A CITY NAME" in text
    for name in ("clear", "partly", "cloud", "fog", "drizzle", "rain", "snow", "storm"):
        assert WEATHER_ICONS[name][0] in text
    assert f"{int(forecast.FORECAST_COLD_C)}°" in text
    assert f"{int(forecast.FORECAST_HOT_C)}°C" in text


def test_vertical_placeholder_is_fitted_to_width_and_height():
    lines = forecast.forecast_vertical_placeholder(24, 6)
    assert len(lines) == 6
    assert all(_width(line) == 24 for line in lines)
    assert forecast.forecast_vertical_placeholder(24, 0) == []
