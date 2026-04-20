"""Runtime settings.

Sourced from `VERBUM_*` environment variables and an optional `.env` file.
Composes with `pydantic-settings` per AGENTS.md S1 λ tooling.

Local llama.cpp server has no API key by default; if a remote endpoint
is added later, an `api_key: SecretStr | None` field belongs here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Override any field with `VERBUM_<FIELD>` env var."""

    model_config = SettingsConfigDict(
        env_prefix="VERBUM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base URL of the running llama.cpp server (HTTP API).
    llama_server_url: str = "http://127.0.0.1:8080"

    # Default request timeout (seconds) for non-streaming HTTP calls.
    http_timeout_s: float = 120.0


def load_settings() -> Settings:
    """Build a Settings instance from environment + .env."""
    return Settings()
