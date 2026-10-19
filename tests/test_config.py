"""config.py: validation rules, deep_merge, user config files, dotted paths and derived values."""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from terrascope.core import config
from terrascope.core.config import ConfigError


def _restore_in_place(target: dict, saved: dict) -> None:
    # Copy `saved` back into the existing dicts, so modules that kept a
    # reference to a nested section (CFG["map"], ...) see the restored values.
    for key in list(target):
        if key not in saved:
            del target[key]
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _restore_in_place(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


@pytest.fixture(autouse=True)
def restore_config():
    """CFG is a module-level global shared with every other test: put it, the
    loaded-file list, the derived values and terrascope_OFFLINE back afterwards."""
    top_level = dict(config.CFG)
    snapshot = copy.deepcopy(config.CFG)
    loaded = list(config.LOADED_FILES)
    derived = (config.CACHE_DIR, config.FETCH_NEW_DATA_FROM_API, config.USER_AGENT)
    offline = os.environ.get("terrascope_OFFLINE")
    yield
    config.CFG.clear()
    config.CFG.update(top_level)
    _restore_in_place(config.CFG, snapshot)
    config.LOADED_FILES[:] = loaded
    config.CACHE_DIR, config.FETCH_NEW_DATA_FROM_API, config.USER_AGENT = derived
    if offline is None:
        os.environ.pop("terrascope_OFFLINE", None)
    else:
        os.environ["terrascope_OFFLINE"] = offline


@pytest.fixture
def candidate():
    """A private copy of the bundled settings to break in one place."""
    return copy.deepcopy(config._DEFAULTS)


def _set(values: dict, dotted: str, value) -> dict:
    *parents, last = dotted.split(".")
    for part in parents:
        values = values[part]
    values[last] = value
    return values


def _write(tmp_path: Path, text: str, name: str = "user.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---- validate_config: the bundled defaults ------------------------------------


def test_bundled_defaults_are_valid(candidate):
    config.validate_config(candidate)


def test_validate_config_without_argument_checks_the_live_settings():
    config.validate_config()
    config.CFG["layers"]["night"]["look"] = "stripes"
    with pytest.raises(ConfigError, match="layers.night.look must be tint or fill"):
        config.validate_config()


# ---- validate_config: type checks against the bundled defaults ----------------


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        # bool settings take only true / false, not 1 or "yes"
        ("app.panel_visible_at_start", 1, "app.panel_visible_at_start: expected true or false"),
        ("app.panel_visible_at_start", "yes", "app.panel_visible_at_start: expected true or false"),
        # int settings: bool is an int subclass in Python, but must be refused
        ("app.min_terminal_width", True, "app.min_terminal_width: expected an integer"),
        ("app.min_terminal_width", 60.0, "app.min_terminal_width: expected an integer"),
        ("app.min_terminal_width", "60", "app.min_terminal_width: expected an integer"),
        # float settings: no bools, strings or non-finite numbers
        ("map.cell_aspect", True, "map.cell_aspect: expected a finite number"),
        ("map.cell_aspect", "2.0", "map.cell_aspect: expected a finite number"),
        ("map.cell_aspect", math.inf, "map.cell_aspect: expected a finite number"),
        ("map.cell_aspect", math.nan, "map.cell_aspect: expected a finite number"),
        # strings, and colours (a default starting with "#" makes it a colour)
        ("data.user_agent", 7, "data.user_agent: expected a string"),
        ("map.country_color", "white", "map.country_color: expected a hex color"),
        ("map.country_color", "#12345", "map.country_color: expected a hex color"),
        ("map.country_color", "ffffff", "map.country_color: expected a hex color"),
        ("map.water_color", None, "map.water_color: expected a string"),
        ("map.province_color", "#12", "map.province_color: expected a hex color"),
        # mappings and lists
        ("map", [], "map: expected a mapping"),
        ("map.zoom", 5, "map.zoom: expected a mapping"),
        ("map.coast_blur_vertical", "1,2,1", "map.coast_blur_vertical: expected a list"),
        ("map.coast_blur_vertical", [1, 2.5], r"map.coast_blur_vertical\[1\]: expected an integer"),
        ("ui.timezones", [["Local", None], ["Mars", 5]], r"ui.timezones\[1\]\[1\]: expected a string"),
        # free-form sections still need string keys and values
        ("map.country_code_overrides", {"DE": 1}, "map.country_code_overrides: expected string keys and values"),
        ("map.country_code_overrides", {1: "X"}, "map.country_code_overrides: expected string keys and values"),
    ],
)
def test_wrong_types_are_rejected_with_the_setting_path(candidate, path, value, message):
    _set(candidate, path, value)
    with pytest.raises(ConfigError, match=message):
        config.validate_config(candidate)


def test_optional_keys_must_be_strings_when_given(candidate):
    candidate["layers"]["night"]["seam_glyph"] = 5
    with pytest.raises(ConfigError, match="layers.night.seam_glyph: expected a string"):
        config.validate_config(candidate)
    candidate["layers"]["night"]["seam_glyph"] = "|"
    config.validate_config(candidate)


def test_a_non_mapping_top_level_is_rejected():
    with pytest.raises(ConfigError, match="expected a mapping"):
        config.validate_config([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("map.cell_aspect", 2),  # an int is fine where the default is a float
        ("map.water_color", False),  # false = transparent ocean
        ("map.land_color", False),
        ("map.land_color", "#0f0f1580"),  # #rrggbbaa
        ("layers.night.day_color", None),  # night colours can be switched off
        ("layers.night.astro_twilight_color", False),
        ("map.country_color", "#abc"),
        ("map.country_color", "#ABCD"),
        ("map.country_color", "#AaBbCc"),
        ("map.country_color", "#aabbccdd"),
        ("terminal.max_color_pairs", 2),
        ("terminal.max_color_pairs", None),
        ("map.transparent_blend_color", "#000"),
        ("ui.modal_bg_color", "#101018"),
        ("paths.cache_dir", "/tmp/terrascope-cache"),
        ("map.country_code_overrides", {"DE": "GER"}),
        ("map.brand_new_setting", 1),  # unknown keys are not type-checked
        ("tabs.start_tab", 3),
        ("map.zoom.pan_step", 1),
        ("layers.weather.cities.forecast_days_ahead", 0),
        ("layers.weather.cities.forecast_days_ahead", 15),
        ("layers.weather.radar.mode", "satellite"),
        ("layers.night.look", "fill"),
        ("layers.weather.radar.opacity", 0),
        ("layers.weather.radar.opacity", 1),
        ("data.natural_earth_base_url", "http://localhost:8000/geojson/"),
        ("app.log_file", "debug.log"),
    ],
)
def test_valid_values_pass(candidate, path, value):
    _set(candidate, path, value)
    config.validate_config(candidate)


def test_panel_orientation_may_be_left_out(candidate):
    del candidate["ui"]["panel_orientation"]
    config.validate_config(candidate)


# ---- validate_config: value and safety rules ---------------------------------


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("paths.cache_dir", "   ", "paths.cache_dir: expected a non-empty path or null"),
        ("paths.cache_dir", 5, "paths.cache_dir: expected a non-empty path or null"),
        ("terminal.max_color_pairs", True, "terminal.max_color_pairs: expected int or null"),
        ("terminal.max_color_pairs", "8", "terminal.max_color_pairs: expected int or null"),
        ("terminal.max_color_pairs", 1, "terminal.max_color_pairs must be at least 2 or null"),
        ("map.transparent_blend_color", 0, "map.transparent_blend_color: expected str or null"),
        ("map.transparent_blend_color", "black", "map.transparent_blend_color: expected a hex color or null"),
        ("ui.modal_bg_color", "#ggg", "ui.modal_bg_color: expected a hex color or null"),
        ("ui.panel_orientation", "diagonal", "ui.panel_orientation must be horizontal or vertical"),
        # the log file must stay inside the cache directory
        ("app.log_file", "", "app.log_file must be a filename inside the cache directory"),
        ("app.log_file", ".", "app.log_file must be a filename inside the cache directory"),
        ("app.log_file", "..", "app.log_file must be a filename inside the cache directory"),
        ("app.log_file", "logs/terrascope.log", "app.log_file must be a filename inside the cache directory"),
        ("app.log_file", "../../.bashrc", "app.log_file must be a filename inside the cache directory"),
        ("app.log_file", "..\\evil.log", "app.log_file must be a filename inside the cache directory"),
        # the User-Agent goes into HTTP headers: no blank value, no header injection
        ("data.user_agent", "   ", "data.user_agent must be non-empty and contain no control characters"),
        ("data.user_agent", "terrascope\r\nX-Evil: 1", "data.user_agent must be non-empty"),
        ("data.user_agent", "terrascope\t1.0", "data.user_agent must be non-empty"),
        # URLs must be http(s) with a host
        ("data.natural_earth_base_url", "ftp://example.com/", "data.natural_earth_base_url: expected an http or https URL"),
        ("layers.flights.api_url", "https://", "layers.flights.api_url: expected an http or https URL"),
        ("layers.weather.cities.api_url", "file:///etc/passwd", "layers.weather.cities.api_url: expected an http or https URL"),
        ("layers.weather.earthquakes.api_url", "example.com/feed", "expected an http or https URL"),
        ("layers.weather.radar.url", "http://[::1", r"layers.weather.radar.url: invalid URL"),
        # positive numbers
        ("app.min_terminal_width", 0, "app.min_terminal_width: must be a positive finite number"),
        ("map.cell_aspect", -1.5, "map.cell_aspect: must be a positive finite number"),
        ("map.zoom.step", 0.0, "map.zoom.step: must be a positive finite number"),
        ("layers.weather.cities.max_concurrent_requests", -1, "must be a positive finite number"),
        # 0..1
        ("layers.weather.radar.opacity", 1.5, "layers.weather.radar.opacity: must be between 0 and 1"),
        ("layers.night.tint_opacity", -0.1, "layers.night.tint_opacity: must be between 0 and 1"),
        # map geometry
        ("map.lat_min", 84.0, "map.lat_min must be less than map.lat_max"),
        ("map.lat_min", -100.0, "map latitude bounds must be ordered and within -90..90"),
        ("map.lat_max", 91.0, "map latitude bounds must be ordered and within -90..90"),
        ("map.zoom.min", 65, "map.zoom.min must not exceed map.zoom.max"),
        ("map.zoom.step", 1.0, "map.zoom.step must be greater than 1"),
        ("map.zoom.pan_step", 0.0, "map.zoom.pan_step must be greater than 0 and at most 1"),
        ("map.zoom.pan_step", 1.5, "map.zoom.pan_step must be greater than 0 and at most 1"),
        # enumerations and ranges
        ("layers.weather.cities.forecast_days_ahead", 16, "forecast_days_ahead must be between 0 and 15"),
        ("layers.weather.cities.forecast_days_ahead", -1, "forecast_days_ahead must be between 0 and 15"),
        ("layers.weather.radar.mode", "infrared", "layers.weather.radar.mode must be radar or satellite"),
        ("layers.night.look", "stripes", "layers.night.look must be tint or fill"),
        ("tabs.start_tab", 4, "tabs.start_tab is outside the configured tab list"),
        ("tabs.start_tab", -1, "tabs.start_tab is outside the configured tab list"),
    ],
)
def test_invalid_values_are_rejected(candidate, path, value, message):
    _set(candidate, path, value)
    with pytest.raises(ConfigError, match=message):
        config.validate_config(candidate)


def test_start_tab_is_checked_against_a_shortened_tab_list(candidate):
    candidate["tabs"]["items"] = candidate["tabs"]["items"][:2]
    candidate["tabs"]["start_tab"] = 2
    with pytest.raises(ConfigError, match="tabs.start_tab is outside"):
        config.validate_config(candidate)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: settings whose YAML default is an int (map.zoom.min/max, timeouts, ...) reject floats",
)
def test_fractional_zoom_is_accepted(candidate):
    candidate["map"]["zoom"]["min"] = 1.5
    config.validate_config(candidate)


# ---- deep_merge ------------------------------------------------------------


def test_deep_merge_merges_dicts_key_by_key_and_returns_base():
    base = {"map": {"zoom": {"min": 1, "max": 64}, "color": "#fff"}, "app": {"q": 1}}
    result = config.deep_merge(base, {"map": {"zoom": {"max": 128}}})
    assert result is base
    assert base == {"map": {"zoom": {"min": 1, "max": 128}, "color": "#fff"}, "app": {"q": 1}}


@pytest.mark.parametrize(
    ("base", "extra", "expected"),
    [
        # lists are replaced, never concatenated
        ({"blur": [1, 2, 3, 2, 1]}, {"blur": [1]}, {"blur": [1]}),
        # a scalar over a dict and a dict over a scalar both replace
        ({"a": {"b": 1}}, {"a": 5}, {"a": 5}),
        ({"a": 5}, {"a": {"b": 1}}, {"a": {"b": 1}}),
        # new keys are added
        ({"a": 1}, {"b": {"c": 2}}, {"a": 1, "b": {"c": 2}}),
        # None replaces too (that is how nullable settings are reset)
        ({"a": {"b": "#fff"}}, {"a": {"b": None}}, {"a": {"b": None}}),
        ({"a": 1}, {}, {"a": 1}),
    ],
)
def test_deep_merge_replaces_everything_but_dicts(base, extra, expected):
    assert config.deep_merge(base, extra) == expected


# ---- unknown-key detection ---------------------------------------------------


def test_unknown_paths_reports_typos_with_their_dotted_path():
    extra = {
        "map": {"zoom": {"maxx": 3, "max": 3}, "colour": "#fff"},
        "mapp": {"zoom": 1},
        "ui": {"panel_orientation": "vertical"},
    }
    assert config._unknown_paths(config._DEFAULTS, extra) == ["map.zoom.maxx", "map.colour", "mapp"]


def test_unknown_paths_allows_optional_and_free_form_keys():
    base = {"map": {"country_code_overrides": {"GB": "UK"}}, "layers": {"night": {}}}
    extra = {
        "map": {"country_code_overrides": {"DE": "GER"}, "province_color": "#123456"},
        "layers": {"night": {"seam_glyph": "|"}},
    }
    assert config._unknown_paths(base, extra) == []


def test_unknown_paths_does_not_descend_into_non_mapping_defaults():
    # map.lat_min is a number: its "children" are a type error, not unknown keys
    assert config._unknown_paths(config._DEFAULTS, {"map": {"lat_min": {"x": 1}}}) == []


# ---- merge_user_config -------------------------------------------------------


def test_merge_user_config_merges_over_current_settings(tmp_path):
    path = _write(tmp_path, "map:\n  zoom:\n    max: 128\nui:\n  tab_gap: 4\n")
    config.merge_user_config(path)
    assert config.CFG["map"]["zoom"]["max"] == 128
    assert config.CFG["map"]["zoom"]["min"] == config._DEFAULTS["map"]["zoom"]["min"]
    assert config.CFG["ui"]["tab_gap"] == 4
    assert config.LOADED_FILES[-1] == path


def test_merge_user_config_warns_about_unknown_keys_but_keeps_them(tmp_path, capsys):
    path = _write(tmp_path, "map:\n  zoom:\n    maxx: 3\n")
    config.merge_user_config(path)
    err = capsys.readouterr().err
    assert f"terrascope: warning: {path}: unknown setting 'map.zoom.maxx'" in err
    assert config.CFG["map"]["zoom"]["maxx"] == 3


def test_merge_user_config_does_not_warn_about_free_form_keys(tmp_path, capsys):
    path = _write(tmp_path, "map:\n  country_code_overrides:\n    DE: GER\n")
    config.merge_user_config(path)
    assert capsys.readouterr().err == ""
    assert config.CFG["map"]["country_code_overrides"] == {"GB": "UK", "DE": "GER"}


@pytest.mark.parametrize("text", ["", "# only a comment\n", "null\n"])
def test_an_empty_user_config_changes_nothing(tmp_path, text):
    before = copy.deepcopy(config.CFG)
    loaded = list(config.LOADED_FILES)
    config.merge_user_config(_write(tmp_path, text))
    assert config.CFG == before
    assert config.LOADED_FILES == loaded


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- map\n- ui\n", "the top level must be a mapping of sections"),
        ("just a string\n", "the top level must be a mapping of sections"),
        ("map: {zoom: [\n", "invalid YAML in"),
        ("map:\n  zoom:\n    step: 1\n", "map.zoom.step must be greater than 1"),
        ("app:\n  log_file: ../escape.log\n", "app.log_file must be a filename"),
    ],
)
def test_a_bad_user_config_raises_and_leaves_settings_untouched(tmp_path, text, message):
    before = copy.deepcopy(config.CFG)
    loaded = list(config.LOADED_FILES)
    with pytest.raises(ConfigError, match=message):
        config.merge_user_config(_write(tmp_path, text))
    assert config.CFG == before
    assert config.LOADED_FILES == loaded


def test_a_missing_user_config_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read .*missing.yaml"):
        config.merge_user_config(tmp_path / "missing.yaml")


def test_merge_user_config_expands_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write(tmp_path, "ui:\n  tab_gap: 7\n", "terrascope.yaml")
    config.merge_user_config("~/terrascope.yaml")
    assert config.CFG["ui"]["tab_gap"] == 7
    assert config.LOADED_FILES[-1] == tmp_path / "terrascope.yaml"


def test_later_user_configs_win(tmp_path):
    config.merge_user_config(_write(tmp_path, "ui:\n  tab_gap: 2\n  box_gap: 3\n", "a.yaml"))
    config.merge_user_config(_write(tmp_path, "ui:\n  tab_gap: 5\n", "b.yaml"))
    assert (config.CFG["ui"]["tab_gap"], config.CFG["ui"]["box_gap"]) == (5, 3)
    assert config.LOADED_FILES[-2:] == [tmp_path / "a.yaml", tmp_path / "b.yaml"]


# ---- get_path / set_path / get_path_from -------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("layers.flights.toggle_key", "p"),
        ("map.zoom", config._DEFAULTS["map"]["zoom"]),
        ("map.zoom.nope", None),
        ("map.lat_min.deeper", None),  # walking through a number
        ("nope", None),
    ],
)
def test_get_path(path, expected):
    assert config.get_path(path) == expected


def test_get_path_returns_the_given_default_for_missing_paths():
    assert config.get_path("map.nope", default="fallback") == "fallback"


def test_get_path_from_raises_for_missing_keys():
    assert config.get_path_from({"a": {"b": 2}}, "a.b") == 2
    with pytest.raises(KeyError):
        config.get_path_from({"a": {}}, "a.b")


def test_set_path_changes_an_existing_key():
    config.set_path("map.zoom.max", 128)
    assert config.CFG["map"]["zoom"]["max"] == 128


@pytest.mark.parametrize(
    "path",
    [
        "map.zoom.nope",  # key does not exist
        "nope.zoom",  # section does not exist
        "map.lat_min.deeper",  # parent is not a section
        "nope",
    ],
)
def test_set_path_refuses_unknown_keys(path):
    with pytest.raises(KeyError):
        config.set_path(path, 1)


def test_set_path_creates_a_key_only_when_asked():
    config.set_path("layers.night.seam_glyph", "|", create=True)
    assert config.CFG["layers"]["night"]["seam_glyph"] == "|"
    with pytest.raises(KeyError):
        config.set_path("nope.seam_glyph", "|", create=True)  # sections are never created


# ---- derived values ----------------------------------------------------------


def test_refresh_derived_uses_the_configured_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config.CFG["paths"]["cache_dir"] = "~/maps"
    config.refresh_derived()
    assert config.CACHE_DIR == tmp_path / "maps"


def test_refresh_derived_falls_back_to_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    config.CFG["paths"]["cache_dir"] = None
    config.refresh_derived()
    assert config.CACHE_DIR == tmp_path / "terrascope"


def test_refresh_derived_falls_back_to_home_cache_without_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config.CFG["paths"]["cache_dir"] = None
    config.refresh_derived()
    assert config.CACHE_DIR == tmp_path / ".cache" / "terrascope"


@pytest.mark.parametrize(
    ("setting", "offline_env", "expected"),
    [
        (True, None, True),
        (True, "1", False),  # terrascope_OFFLINE=1 always wins
        (True, "0", True),  # only exactly "1" means offline
        (False, None, False),
    ],
)
def test_refresh_derived_live_request_switch(setting, offline_env, expected):
    config.CFG["data"]["fetch_new_data_from_api"] = setting
    if offline_env is None:
        os.environ.pop("terrascope_OFFLINE", None)
    else:
        os.environ["terrascope_OFFLINE"] = offline_env
    config.refresh_derived()
    assert config.FETCH_NEW_DATA_FROM_API is expected


def test_refresh_derived_copies_the_user_agent():
    config.CFG["data"]["user_agent"] = "terrascope-tests/9"
    config.refresh_derived()
    assert config.USER_AGENT == "terrascope-tests/9"


# ---- first-launch install of the editable default config ---------------------


def test_first_launch_copies_the_bundled_config(tmp_path, monkeypatch):
    user_file = tmp_path / "terrascope" / "default.yaml"
    monkeypatch.setattr(config, "USER_CONFIG_DIR", user_file.parent)
    monkeypatch.setattr(config, "USER_DEFAULT_CONFIG_FILE", user_file)
    assert config._install_default_config() == user_file
    assert user_file.read_bytes() == config.BUNDLED_DEFAULT_CONFIG_FILE.read_bytes()


def test_an_existing_user_config_is_never_replaced(tmp_path, monkeypatch):
    user_file = _write(tmp_path, "# my edits\n", "default.yaml")
    monkeypatch.setattr(config, "USER_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "USER_DEFAULT_CONFIG_FILE", user_file)
    assert config._install_default_config() == user_file
    assert user_file.read_text(encoding="utf-8") == "# my edits\n"


def test_an_unwritable_config_dir_falls_back_to_the_bundled_file(tmp_path, monkeypatch):
    blocker = _write(tmp_path, "", "not-a-dir")  # mkdir under a file fails
    monkeypatch.setattr(config, "USER_CONFIG_DIR", blocker / "terrascope")
    monkeypatch.setattr(config, "USER_DEFAULT_CONFIG_FILE", blocker / "terrascope" / "default.yaml")
    assert config._install_default_config() == config.BUNDLED_DEFAULT_CONFIG_FILE


def test_the_sandboxed_user_config_matches_the_bundled_defaults():
    assert config.DEFAULT_CONFIG_FILE == config.USER_DEFAULT_CONFIG_FILE
    assert config._read_yaml(config.BUNDLED_DEFAULT_CONFIG_FILE) == config._DEFAULTS


# ---- terrascope_CONFIG is applied at import time (needs a fresh interpreter) --


def _fresh_python(tmp_path: Path, code: str, user_config: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "terrascope_OFFLINE": "1",
        "terrascope_CONFIG": str(user_config),
    }
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )


def test_terrascope_config_env_is_merged_at_import(tmp_path):
    user_config = _write(tmp_path, "map:\n  zoom:\n    max: 99\n")
    result = _fresh_python(
        tmp_path,
        "from terrascope.core import config; "
        "print(config.CFG['map']['zoom']['max'], config.LOADED_FILES[-1])",
        user_config,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["99", str(user_config)]


def test_a_broken_terrascope_config_env_makes_the_cli_exit_cleanly(tmp_path):
    user_config = _write(tmp_path, "map: [\n")
    result = _fresh_python(tmp_path, "import terrascope.core.cli", user_config)
    assert result.returncode == 1
    assert "terrascope: could not load configuration: invalid YAML in" in result.stderr
    assert "Traceback" not in result.stderr
