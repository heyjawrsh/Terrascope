"""registry.py and tabs.py: toggle-key clashes, layer building, and tabs built from the config."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from terrascope.core import config, registry, tabs
from terrascope.core.tabs import Tab


def _layer(name: str, toggle_key: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, toggle_key=toggle_key)


# ---- check_toggle_keys ---------------------------------------------------------


@pytest.mark.parametrize(
    "layers",
    [
        [],
        [_layer("a", "p"), _layer("b", "k")],
        [_layer("a", ""), _layer("b", ""), _layer("c", "x")],  # no key = no clash
    ],
)
def test_distinct_toggle_keys_pass(layers):
    registry.check_toggle_keys(layers)


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ([_layer("flights", "p"), _layer("ports", "p")], "Layers 'flights' and 'ports' both use key 'p'"),
        # letters work in both cases, so P and p clash
        ([_layer("flights", "p"), _layer("ports", "P")], "Layers 'flights' and 'ports' both use key 'p'"),
        ([_layer("a", "x"), _layer("b", "y"), _layer("c", "X")], "Layers 'a' and 'c' both use key 'x'"),
    ],
)
def test_duplicate_toggle_keys_are_refused(layers, message):
    with pytest.raises(ValueError, match=message):
        registry.check_toggle_keys(layers)


# ---- build_layers -----------------------------------------------------------------


def test_build_layers_makes_fresh_instances_each_time():
    first, second = registry.build_layers(), registry.build_layers()
    assert [type(layer) for layer in first] == list(registry.ALL_LAYER_CLASSES)
    assert all(a is not b for a, b in zip(first, second))


def test_build_layers_checks_toggle_keys(monkeypatch):
    class Planes:
        name, toggle_key = "planes", "p"

    class Ports:
        name, toggle_key = "ports", "P"

    monkeypatch.setattr(registry, "ALL_LAYER_CLASSES", (Planes, Ports))
    with pytest.raises(ValueError, match="Layers 'planes' and 'ports' both use key 'p'"):
        registry.build_layers()


def test_layer_keys_do_not_clash_with_app_keys():
    # default.yaml: a layer's toggle key "must not clash with another layer or
    # with a key bound under app.keys" -- including the sub-toggles and look key.
    app_keys = set()
    for spec in config.CFG["app"]["keys"].values():
        app_keys.update(str(key).lower() for key in (spec if isinstance(spec, list) else [spec]))
    layer_keys = [layer.toggle_key.lower() for layer in registry.build_layers() if layer.toggle_key]
    layer_keys += [
        str(config.get_path(path)).lower()
        for path in ("layers.flights.airports.toggle_key", "layers.weather.cities.toggle_key",
                     "layers.weather.earthquakes.toggle_key", "layers.night.look_key")
    ]
    assert len(layer_keys) == len(set(layer_keys))
    assert not app_keys & set(layer_keys)
    assert not set(layer_keys) & set("123456789")  # digits select tabs


# ---- tabs ----------------------------------------------------------------------------


def test_tabs_follow_the_configured_items():
    items = config.CFG["tabs"]["items"]
    assert [tab.name for tab in tabs.TABS] == [item["name"] for item in items]
    assert [tab.label for tab in tabs.TABS] == ["MAP", "TIME", "WEATHER", "PLANES"]
    assert tabs.START_TAB == config.CFG["tabs"]["start_tab"] == 0


def test_bundled_tab_options():
    by_name = {tab.name: tab for tab in tabs.TABS}
    assert by_name["map"].detailed and by_name["map"].provinces  # provinces follows detailed
    assert by_name["weather"].provinces is False  # explicitly switched off
    assert not by_name["weather"].place_names and not by_name["weather"].country_names
    assert by_name["planes"].layers == ("basemap", "flights")
    assert by_name["time"].attribution == "NOAA, Meeus"


def test_every_tab_names_real_layers():
    known = {layer.name for layer in registry.build_layers()}
    for tab in tabs.TABS:
        assert tab.layers, tab.name
        assert set(tab.layers) <= known, tab.name


def test_a_minimal_tab_item_gets_the_defaults():
    tab = tabs._build({"name": "x", "label": "X", "color": "#ffffff"})
    assert tab == Tab(name="x", label="X", color="#ffffff", blurb="", layers=(),
                      detailed=False, provinces=False, place_names=True,
                      country_names=True, attribution="")


@pytest.mark.parametrize(
    ("extra", "field", "expected"),
    [
        ({"detailed": True}, "provinces", True),  # provinces defaults to detailed
        ({"detailed": True, "provinces": False}, "provinces", False),
        ({"provinces": True}, "detailed", False),
        ({"layers": ["basemap", "night"]}, "layers", ("basemap", "night")),
        ({"layers": None}, "layers", ()),
        ({"blurb": None, "attribution": None}, "blurb", ""),
        ({"attribution": None}, "attribution", ""),
        ({"place_names": False}, "place_names", False),
        ({"country_names": 0}, "country_names", False),  # coerced to bool
    ],
)
def test_tab_item_options(extra, field, expected):
    tab = tabs._build({"name": "x", "label": "X", "color": "#ffffff", **extra})
    assert getattr(tab, field) == expected


def test_tab_fields_are_coerced_to_strings():
    tab = tabs._build({"name": 7, "label": 8, "color": "#fff", "blurb": 9})
    assert (tab.name, tab.label, tab.blurb) == ("7", "8", "9")


def test_tabs_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        tabs.TABS[0].name = "other"  # type: ignore[misc]
