"""Smoke tests — package imports and CLI version round-trips.

Proves the skeleton holds together end-to-end before any real code lands.
"""

from __future__ import annotations

import importlib

import pytest

import verbum


def test_version_exposed() -> None:
    assert isinstance(verbum.__version__, str)
    assert verbum.__version__  # non-empty


@pytest.mark.parametrize(
    "module",
    [
        "verbum.client",
        "verbum.probes",
        "verbum.results",
        "verbum.lambda_ast",
        "verbum.analysis",
        "verbum.cli",
        "verbum.config",
    ],
)
def test_module_importable(module: str) -> None:
    importlib.import_module(module)


def test_cli_version_command() -> None:
    """Typer CLI exposes `version` and prints __version__."""
    from typer.testing import CliRunner

    from verbum.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert verbum.__version__ in result.stdout


def test_settings_defaults() -> None:
    """Settings construct with no env, default points at local llama.cpp."""
    from verbum.config import Settings

    s = Settings()
    assert s.llama_server_url.startswith("http")
    assert s.http_timeout_s > 0
