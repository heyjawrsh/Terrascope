"""cli.py: value parsers, flag handling, override precedence, information commands, version."""

from __future__ import annotations

import argparse
import copy
import os
import re
import runpy
import sys
from pathlib import Path

import pytest
import yaml

import terrascope
from terrascope.core import cli, config


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
    """Every CLI call rewrites the shared CFG (and --offline / --online the
    environment): put it all back so other tests see the defaults."""
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


def apply(*argv: str) -> argparse.Namespace:
    """Parse argv and fold it into config.CFG, as run() does."""
    parser = cli.build_parser()
    args = parser.parse_args(list(argv))
    cli.apply_settings(args, parser)
    return args


def run_cli(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["terrascope", *argv])
    cli.run()


def usage_error(capsys, *argv: str) -> str:
    """argparse's exit status 2 plus the message it printed."""
    with pytest.raises(SystemExit) as exit_info:
        apply(*argv)
    assert exit_info.value.code == 2
    return capsys.readouterr().err


def _write(tmp_path: Path, text: str, name: str = "user.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---- value parsers ------------------------------------------------------------


@pytest.mark.parametrize(
    ("parser", "text", "expected"),
    [
        (cli.INT, "-3", -3),
        (cli.FLOAT, "2.5", 2.5),
        (cli.POS_INT, "1", 1),
        (cli.NONNEG_INT, "0", 0),
        (cli.POS_FLOAT, "0.001", 0.001),
        (cli.NONNEG_FLOAT, "0", 0.0),
        (cli.UNIT_FLOAT, "1", 1.0),
        (cli.UNIT_FLOAT, "0", 0.0),
        (cli._color, "#ABCDEF", "#abcdef"),
        (cli._color, "abc", "#abc"),  # the # is optional on the command line
        (cli._color, "  #123456 ", "#123456"),
        (cli._color_alpha, "#0B1D2E80", "#0b1d2e80"),
        (cli._color_alpha, "0b1d2e", "#0b1d2e"),
        (cli._color_alpha, "#abcd", "#abcd"),
        (cli._color_alpha, "None", False),
        (cli._color_alpha, "off", False),
        (cli._color_alpha, "transparent", False),
        (cli._color_alpha, "FALSE", False),
        (cli._pair, "2,1", [2, 1]),
        (cli._pair, "0, 0", [0, 0]),
    ],
)
def test_value_parsers_accept(parser, text, expected):
    assert parser(text) == expected


@pytest.mark.parametrize(
    ("parser", "text", "message"),
    [
        (cli.INT, "1.5", "'1.5' is not an integer"),
        (cli.INT, "", "'' is not an integer"),
        (cli.FLOAT, "fast", "'fast' is not a number"),
        (cli.FLOAT, "nan", "'nan' is not a finite number"),
        (cli.FLOAT, "inf", "'inf' is not a finite number"),
        (cli.POS_INT, "0", "must be >= 1"),
        (cli.NONNEG_INT, "-1", "must be >= 0"),
        (cli.POS_FLOAT, "0", "must be > 0"),
        (cli.POS_FLOAT, "-0.5", "must be > 0"),
        (cli.UNIT_FLOAT, "1.01", "must be <= 1"),
        (cli.UNIT_FLOAT, "-0.01", "must be >= 0"),
        (cli._color, "#abcd", "is not a colour like #rrggbb"),  # no alpha here
        (cli._color, "red", "is not a colour like #rrggbb"),
        (cli._color, "none", "is not a colour like #rrggbb"),
        (cli._color_alpha, "#12345", "is not a colour like #rrggbb or #rrggbbaa"),
        (cli._color_alpha, "blue", "is not a colour like #rrggbb or #rrggbbaa"),
        (cli._pair, "2", "is not two integers like 2,1"),
        (cli._pair, "1,2,3", "is not two integers like 2,1"),
        (cli._pair, "a,b", "is not two integers like 2,1"),
        (cli._pair, "-1,2", "values must be >= 0"),
    ],
)
def test_value_parsers_reject(parser, text, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        parser(text)


def test_number_parsers_are_named_after_their_type():
    # argparse uses the callable's name in "invalid int value" messages
    assert (cli.POS_INT.__name__, cli.UNIT_FLOAT.__name__) == ("int", "float")


# ---- the option table ------------------------------------------------------------


def test_every_option_points_at_an_existing_setting():
    for opt in cli.OPTIONS:
        assert config.get_path(opt.path) is not None or opt.path in config._NULLABLE_PATHS, opt.path


def test_option_flags_and_destinations_are_unique():
    flags = [flag for opt in cli.OPTIONS for flag in opt.flags]
    assert len(flags) == len(set(flags))
    dests = [opt.dest for opt in cli.OPTIONS]
    assert len(dests) == len(set(dests))


_FLOAT_FLAGS_WITH_INT_DEFAULTS = {
    "--download-timeout", "--zoom-min", "--zoom-max", "--province-zoom",
    "--flights-timeout", "--radar-history-minutes", "--quake-dim-hours", "--night-update",
}


def _default_as_text(opt: cli.Opt) -> str:
    value = config.get_path(opt.path)
    return ",".join(str(item) for item in value) if isinstance(value, list) else str(value)


@pytest.mark.parametrize(
    "opt",
    [
        pytest.param(
            opt,
            id=opt.flags[0],
            marks=[pytest.mark.xfail(
                strict=True,
                reason="BUG: flag parses to float but its YAML default is an int, so validation "
                       "rejects every value (config._check_types)",
            )] if opt.flags[0] in _FLOAT_FLAGS_WITH_INT_DEFAULTS else [],
        )
        for opt in cli.OPTIONS
        if opt.kind != "bool" and config.get_path(opt.path) is not None
    ],
)
def test_every_value_flag_accepts_its_own_default(opt):
    text = _default_as_text(opt)
    apply(opt.flags[0], text)
    assert config.get_path(opt.path) == opt.kind(text)


@pytest.mark.parametrize(
    "opt", [opt for opt in cli.OPTIONS if opt.kind == "bool"], ids=lambda opt: opt.flags[0]
)
def test_every_switch_has_on_and_off_forms(opt):
    flag = opt.flags[0]
    apply(flag)
    assert config.get_path(opt.path) is True
    apply("--no-" + flag[2:])
    assert config.get_path(opt.path) is False


# ---- applying flags ------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "path", "expected"),
    [
        (["--water-color", "none"], "map.water_color", False),
        (["--land-color", "#0B1D2E80"], "map.land_color", "#0b1d2e80"),
        (["--country-color", "FFF"], "map.country_color", "#fff"),
        (["--outer-margin", "3,2"], "ui.outer_margin", [3, 2]),
        (["--panel-orientation", "horizontal"], "ui.panel_orientation", "horizontal"),
        (["--radar-opacity", "0.8"], "layers.weather.radar.opacity", 0.8),
        (["--forecast-days", "15"], "layers.weather.cities.forecast_days_ahead", 15),
        (["--max-color-pairs", "16"], "terminal.max_color_pairs", 16),
        (["--lat-min", "-60", "--lat-max", "80"], "map.lat_max", 80.0),
        (["--log-file", "debug.log"], "app.log_file", "debug.log"),
        (["--user-agent", "my-terrascope/2"], "data.user_agent", "my-terrascope/2"),
        (["--opensky-url", "https://example.com/states"], "layers.flights.api_url", "https://example.com/states"),
    ],
)
def test_flags_set_their_setting(argv, path, expected):
    apply(*argv)
    assert config.get_path(path) == expected


def test_no_flags_leave_the_settings_alone():
    before = copy.deepcopy(config.CFG)
    apply()
    assert config.CFG == before


def test_user_agent_flag_updates_the_derived_value():
    apply("--user-agent", "my-terrascope/2")
    assert config.USER_AGENT == "my-terrascope/2"


def test_cache_dir_is_expanded_and_becomes_the_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    apply("--cache-dir", "~/maps")
    assert config.CFG["paths"]["cache_dir"] == str(tmp_path / "maps")
    assert config.CACHE_DIR == tmp_path / "maps"


def test_offline_forces_live_requests_off():
    os.environ.pop("terrascope_OFFLINE", None)
    apply("--offline")
    assert os.environ["terrascope_OFFLINE"] == "1"
    assert config.CFG["data"]["fetch_new_data_from_api"] is False
    assert config.FETCH_NEW_DATA_FROM_API is False


def test_online_overrides_the_environment_and_config(tmp_path):
    os.environ["terrascope_OFFLINE"] = "1"
    user_config = _write(tmp_path, "data:\n  fetch_new_data_from_api: false\n")
    apply("--config", str(user_config), "--online")
    assert "terrascope_OFFLINE" not in os.environ
    assert config.FETCH_NEW_DATA_FROM_API is True


def test_offline_and_online_together_are_refused(capsys):
    assert "--offline and --online cannot be used together" in usage_error(capsys, "--offline", "--online")


# ---- precedence: default.yaml < --config files < flags < --set --------------------


def test_flags_beat_config_files_and_set_beats_flags(tmp_path):
    user_config = _write(tmp_path, "ui:\n  tab_gap: 2\n  box_gap: 2\n  legend_text_color: '#000000'\n")
    apply(
        "--config", str(user_config),
        "--tab-gap", "3", "--legend-color", "#111111",
        "--set", "ui.tab_gap=5",
    )
    assert config.CFG["ui"]["tab_gap"] == 5  # --set
    assert config.CFG["ui"]["legend_text_color"] == "#111111"  # flag
    assert config.CFG["ui"]["box_gap"] == 2  # config file


def test_several_config_files_merge_in_order(tmp_path):
    first = _write(tmp_path, "ui:\n  tab_gap: 2\n  box_gap: 3\n", "a.yaml")
    second = _write(tmp_path, "ui:\n  tab_gap: 4\n", "b.yaml")
    apply("-c", str(first), "--config", str(second))
    assert (config.CFG["ui"]["tab_gap"], config.CFG["ui"]["box_gap"]) == (4, 3)
    assert config.LOADED_FILES[-2:] == [first, second]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ui: [\n", "invalid YAML in"),
        ("ui:\n  panel_orientation: diagonal\n", "ui.panel_orientation must be horizontal or vertical"),
        ("- a list\n", "the top level must be a mapping"),
    ],
)
def test_a_bad_config_file_is_a_usage_error(tmp_path, capsys, text, message):
    assert message in usage_error(capsys, "--config", str(_write(tmp_path, text)))


def test_a_missing_config_file_is_a_usage_error(tmp_path, capsys):
    assert "cannot read" in usage_error(capsys, "--config", str(tmp_path / "missing.yaml"))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a tab item without 'name' passes validate_config, then cli._tab_names raises KeyError",
)
def test_a_tab_without_a_name_is_a_usage_error(tmp_path, capsys):
    user_config = _write(tmp_path, "tabs:\n  items:\n    - {label: X, color: '#ffffff'}\n")
    usage_error(capsys, "--config", str(user_config))


# ---- --set ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "path", "expected"),
    [
        ("map.zoom.step=2", "map.zoom.step", 2),
        ("map.land_color=\"#101018\"", "map.land_color", "#101018"),
        ("map.water_color=false", "map.water_color", False),
        (" ui.tab_gap =3", "ui.tab_gap", 3),  # the key is stripped
        ("paths.cache_dir=null", "paths.cache_dir", None),
        ("app.log_file=a=b.log", "app.log_file", "a=b.log"),  # only the first = splits
        ("map.coast_blur_vertical=[1, 4, 1]", "map.coast_blur_vertical", [1, 4, 1]),
        ("map.country_code_overrides={DE: GER}", "map.country_code_overrides", {"DE": "GER"}),
        ("layers.night.seam_glyph=\"|\"", "layers.night.seam_glyph", "|"),  # optional key: created
    ],
)
def test_set_takes_yaml_values(item, path, expected):
    apply("--set", item)
    assert config.get_path(path) == expected


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("map.zoom.step", "--set expects KEY=VALUE, got 'map.zoom.step'"),
        ("map.zoom.stepp=2", "--set: unknown setting 'map.zoom.stepp'"),
        ("ui.new_key=1", "--set: unknown setting 'ui.new_key'"),
        ("map.lat_min.x=1", "--set: unknown setting 'map.lat_min.x'"),
        ("map.zoom.max=[1", "--set map.zoom.max: value is not valid YAML"),
        # an unquoted colour is a YAML comment, i.e. null
        ("map.land_color=#101018", "map.land_color: expected a string"),
        ("ui.panel_orientation=diagonal", "ui.panel_orientation must be horizontal or vertical"),
        ("app.min_terminal_width=true", "app.min_terminal_width: expected an integer"),
        ("map=1", "map: expected a mapping"),
        ("tabs.start_tab=9", "tabs.start_tab must be 0-3, got 9"),
        ("tabs.start_tab=first", "tabs.start_tab must be 0-3, got 'first'"),
    ],
)
def test_bad_set_values_are_usage_errors(capsys, item, message):
    assert message in usage_error(capsys, "--set", item)


# ---- --tab -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "index"),
    [("map", 0), ("PLANES", 3), ("Weather", 2), ("2", 1), (" 4 ", 3), ("1", 0)],
)
def test_tab_by_name_or_number(spec, index):
    apply("--tab", spec)
    assert config.CFG["tabs"]["start_tab"] == index


@pytest.mark.parametrize("spec", ["0", "5", "-1", "radar", ""])
def test_unknown_tabs_are_usage_errors(capsys, spec):
    err = usage_error(capsys, "-t", spec)
    assert f"unknown tab {spec!r}; choose a number 1-4 or one of: map, time, weather, planes" in err


# ---- argparse-level checks ---------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--panel-orientation", "diagonal"], "invalid choice: 'diagonal'"),
        (["--radar-mode", "infrared"], "invalid choice: 'infrared'"),
        (["--forecast-days", "16"], "must be <= 15"),
        (["--max-color-pairs", "1"], "must be >= 2"),
        (["--zoom-step", "1"], "must be > 1"),
        (["--pan-step", "0"], "must be > 0"),
        (["--radar-world-zoom", "5"], "must be <= 4"),
        (["--outer-margin", "2"], "is not two integers"),
        (["--water-color", "blue"], "is not a colour like #rrggbb or #rrggbbaa"),
        (["--off"], "unrecognized arguments: --off"),  # no abbreviations
        (["--no-panel-orientation"], "unrecognized arguments"),
    ],
)
def test_bad_flag_values_are_usage_errors(capsys, argv, message):
    assert message in usage_error(capsys, *argv)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        # each flag is valid alone; the combination fails validation
        (["--lat-min", "80", "--lat-max", "10"], "map.lat_min must be less than map.lat_max"),
        (["--user-agent", "a\tb"], "data.user_agent must be non-empty and contain no control characters"),
        (["--log-file", "../x.log"], "app.log_file must be a filename inside the cache directory"),
        (["--map-data-url", "ftp://example.com/"], "data.natural_earth_base_url: expected an http or https URL"),
        (["--cache-dir", " "], "paths.cache_dir: expected a non-empty path or null"),
    ],
)
def test_values_are_validated_after_all_overrides(capsys, argv, message):
    assert message in usage_error(capsys, *argv)


# ---- help and version ---------------------------------------------------------------


def test_help_lists_every_group_and_flag(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "--help")
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for title in cli.GROUP_ORDER:
        assert f"{title}:" in out
    for opt in cli.OPTIONS:
        assert opt.flags[0] in out
    assert "tabs: 1=map, 2=time, 3=weather, 4=planes" in out
    assert "--set 'map.country_code_overrides={DE: \"GER\"}'" in out


def test_option_help_shows_choices_defaults_and_escapes_percent():
    opt = cli.Opt(cli.G_LAYOUT, ("--x",), "ui.panel_orientation", str, "Use 100% width.",
                  "MODE", ("horizontal", "vertical"))
    assert cli._option_help(opt) == (
        "Use 100%% width. One of: horizontal, vertical. (default: vertical)"
    )
    switch = cli.Opt(cli.G_STARTUP, ("--x",), "app.panel_visible_at_start", "bool", "Panel.")
    assert cli._option_help(switch) == "Panel. (default: on)"
    pair = cli.Opt(cli.G_LAYOUT, ("--x",), "ui.outer_margin", cli._pair, "Margin.")
    assert cli._option_help(pair) == "Margin. (default: 2,1)"
    nullable = cli.Opt(cli.G_TERMINAL, ("--x",), "terminal.max_color_pairs", cli.POS_INT, "Pairs.")
    assert cli._option_help(nullable) == "Pairs."


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version_flag(monkeypatch, capsys, flag):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, flag)
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"terrascope {terrascope.__version__}\n"


def test_python_dash_m_terrascope_runs_the_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["terrascope", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("terrascope", run_name="__main__")
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("terrascope ")


def test_project_version_without_a_toml_parser(monkeypatch):
    # Python 3.10 has no tomllib: the regex fallback must find the same version.
    expected = terrascope._project_version()
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", None)
    assert terrascope._project_version() == expected == terrascope.__version__


# ---- information commands ---------------------------------------------------------


def test_list_layers_shows_state_and_toggle_keys(monkeypatch, capsys):
    run_cli(monkeypatch, "--list-layers", "-d", "flights")
    rows = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        ["basemap", "on", "-"],
        ["night", "off", "N"],
        ["flights", "off", "P"],
        ["weather", "on", "-"],
    ]


def test_unknown_disabled_layer_is_a_usage_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(monkeypatch, "--list-layers", "-d", "radar", "-d", "clouds")
    assert exit_info.value.code == 2
    assert (
        "unknown layer name(s): clouds, radar (available: basemap, flights, night, weather)"
        in capsys.readouterr().err
    )


def test_list_tabs_marks_the_start_tab(monkeypatch, capsys):
    run_cli(monkeypatch, "--list-tabs", "--tab", "weather")
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[:3] for line in lines] == [
        [" 1", "map", "basemap"],
        [" 2", "time", "basemap,night"],
        ["*3", "weather", "basemap,weather"],
        [" 4", "planes", "basemap,flights"],
    ]
    assert lines[3].endswith("live aircraft")


def test_list_keys_shows_configured_and_fixed_keys(monkeypatch, capsys):
    run_cli(monkeypatch, "--list-keys", "--set", "app.keys.quit=x")
    rows = dict(re.split(r" {2,}", line, maxsplit=1) for line in capsys.readouterr().out.splitlines())
    assert rows["quit"] == "x"
    assert rows["zoom in"] == "+ ="
    assert rows["planes on/off"] == "p"
    assert rows["tint / fill look"] == "l"
    assert rows["select tab"] == "1-9"
    assert rows["mouse"] == "drag = pan, wheel = zoom, click = select"


def test_list_keys_aligns_its_columns(monkeypatch, capsys):
    run_cli(monkeypatch, "--list-keys")
    lines = capsys.readouterr().out.splitlines()
    width = max(len(line.split("  ")[0]) for line in lines)
    assert all(line[width:width + 2] == "  " for line in lines)


def test_list_keys_fills_in_timezone_and_skips_unbound_layer_keys(monkeypatch, capsys):
    del config.CFG["app"]["keys"]["timezone"]
    config.CFG["layers"]["flights"]["airports"]["toggle_key"] = ""
    cli.print_key_list()
    out = capsys.readouterr().out
    assert any(line.split() == ["timezone", "z"] for line in out.splitlines())
    assert "airports on/off" not in out


def test_show_paths(monkeypatch, capsys, tmp_path):
    user_config = _write(tmp_path, "ui:\n  tab_gap: 2\n")
    run_cli(monkeypatch, "--show-paths", "--cache-dir", str(tmp_path / "cache"),
            "--config", str(user_config), "--offline")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"cache dir      {tmp_path / 'cache'}"
    assert lines[1] == f"log file       {tmp_path / 'cache' / 'terrascope.log'}"
    assert lines[2] == f"config files   {config.DEFAULT_CONFIG_FILE}"
    assert lines[3] == f"               {user_config}"
    assert lines[4] == "live requests  off (offline)"


def test_show_paths_reports_live_requests_when_online(monkeypatch, capsys):
    run_cli(monkeypatch, "--show-paths", "--online")
    assert "live requests  on" in capsys.readouterr().out


def test_dump_config_prints_the_effective_settings(monkeypatch, capsys):
    run_cli(monkeypatch, "--dump-config", "--tab-gap", "6", "--set", "map.zoom.max=128")
    dumped = yaml.safe_load(capsys.readouterr().out)
    assert dumped == config.CFG
    assert dumped["ui"]["tab_gap"] == 6
    assert dumped["map"]["zoom"]["max"] == 128


def test_information_commands_combine_without_launching(monkeypatch, capsys):
    monkeypatch.setattr("terrascope.core.main.launch", lambda *args: pytest.fail("launched"))
    run_cli(monkeypatch, "--list-tabs", "--list-keys", "--list-layers")
    out = capsys.readouterr().out
    assert "\tplanes\t" in out and "select tab" in out and "basemap\ton\t-" in out


# ---- --clear-cache ------------------------------------------------------------------


def test_clear_cache_removes_only_terrascope_cache_files(monkeypatch, capsys, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("ne_50m_admin_0_countries.geojson", "download.tmp", "flights_cache.json",
                 "terrascope.log", "notes.json", "welcome_skip"):
        (cache / name).write_text("x", encoding="utf-8")
    (cache / "folder.tmp").mkdir()
    run_cli(monkeypatch, "--clear-cache", "--cache-dir", str(cache))
    assert capsys.readouterr().out == f"removed 3 cached item(s) from {cache}\n"
    assert sorted(path.name for path in cache.iterdir()) == [
        "folder.tmp", "notes.json", "terrascope.log", "welcome_skip",
    ]


def test_clear_cache_with_no_cache_dir(monkeypatch, capsys, tmp_path):
    run_cli(monkeypatch, "--clear-cache", "--cache-dir", str(tmp_path / "absent"))
    assert capsys.readouterr().out == f"nothing to clear: {tmp_path / 'absent'} does not exist\n"


def test_clear_cache_reports_files_it_cannot_remove(monkeypatch, capsys, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a_cache.json").write_text("x", encoding="utf-8")
    (cache / "b_cache.json").write_text("x", encoding="utf-8")
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == "a_cache.json":
            raise PermissionError("read-only")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    run_cli(monkeypatch, "--clear-cache", "--cache-dir", str(cache))
    captured = capsys.readouterr()
    assert "could not remove" in captured.err and "a_cache.json" in captured.err
    assert captured.out == f"removed 1 cached item(s) from {cache}\n"


# ---- launching -----------------------------------------------------------------------


def test_run_hands_the_layers_to_launch(monkeypatch):
    calls = []
    monkeypatch.setattr("terrascope.core.main.launch", lambda layers, disabled: calls.append((layers, disabled)))
    run_cli(monkeypatch, "-d", "night", "-d", "weather", "--tab", "planes")
    [(layers, disabled)] = calls
    assert disabled == ["night", "weather"]
    assert {layer.name: layer.enabled for layer in layers} == {
        "basemap": True, "night": False, "flights": True, "weather": False,
    }
    assert config.CFG["tabs"]["start_tab"] == 3
