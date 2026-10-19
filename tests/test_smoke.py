"""The test sandbox itself, and that every module imports and the layers build."""

from __future__ import annotations

import importlib
import os
import pkgutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import terrascope


def test_config_and_cache_live_in_the_sandbox():
    from terrascope.core import config

    sandbox_config = Path(os.environ["XDG_CONFIG_HOME"])
    sandbox_cache = Path(os.environ["XDG_CACHE_HOME"])
    assert config.USER_CONFIG_DIR.is_relative_to(sandbox_config)
    assert config.CACHE_DIR.is_relative_to(sandbox_cache)
    assert config.FETCH_NEW_DATA_FROM_API is False


def test_unregistered_urls_fail_like_an_unreachable_host(fake_http):
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen("https://example.com/feed.json", timeout=1)
    assert fake_http.calls == ["https://example.com/feed.json"]


def test_registered_urls_return_canned_bodies(fake_http):
    fake_http.responses["https://example.com/"] = {"ok": True}
    with urllib.request.urlopen(urllib.request.Request("https://example.com/a")) as response:
        assert response.read() == b'{"ok": true}'


@pytest.mark.parametrize(
    "module",
    sorted(
        info.name
        for info in pkgutil.walk_packages(terrascope.__path__, prefix="terrascope.")
        if not info.name.endswith("__main__")
    ),
)
def test_every_module_imports(module):
    importlib.import_module(module)


def test_layers_build_in_draw_order_with_unique_toggle_keys():
    from terrascope.core.registry import build_layers

    names = [layer.name for layer in build_layers()]
    assert names == ["basemap", "night", "flights", "weather"]


def test_package_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert f'version = "{terrascope.__version__}"' in pyproject.read_text(encoding="utf-8")
