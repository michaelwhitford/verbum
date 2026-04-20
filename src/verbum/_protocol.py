"""Wire-level pydantic models for the llama.cpp HTTP API.

Tolerant ingest (S2 λ lambda_text): `extra="allow"` so unknown server
fields survive round-trip. Strict outputs (we send only what we name).

Contract mirrors `specs/llama_server.openapi.yaml`. Grow by use: when the
client first needs a field, add it here and to the spec.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CompletionResult",
    "HealthStatus",
    "ServerProps",
    "StreamEvent",
    "Timings",
    "TokenizeResult",
]


class _Wire(BaseModel):
    """Base for server-emitted payloads — tolerant to unknown fields."""

    model_config = ConfigDict(extra="allow", frozen=False)


class HealthStatus(_Wire):
    """`GET /health` response. Server reports readiness state."""

    status: str = Field(default="unknown")


class ServerProps(_Wire):
    """`GET /props` response. Consumed at run-start for `meta.json` provenance.

    Fields modeled here are the ones we actively record. Other fields the
    server emits are preserved via `extra="allow"` and reachable through
    `.model_dump()`.
    """

    default_generation_settings: dict[str, Any] = Field(default_factory=dict)
    total_slots: int | None = None
    chat_template: str | None = None
    model_path: str | None = None
    n_ctx: int | None = None


class TokenizeResult(_Wire):
    """`POST /tokenize` response."""

    tokens: list[int] = Field(default_factory=list)


class Timings(_Wire):
    """Per-completion timing block from llama.cpp (ms-scaled floats)."""

    prompt_n: int | None = None
    prompt_ms: float | None = None
    prompt_per_token_ms: float | None = None
    prompt_per_second: float | None = None
    predicted_n: int | None = None
    predicted_ms: float | None = None
    predicted_per_token_ms: float | None = None
    predicted_per_second: float | None = None


class CompletionResult(_Wire):
    """Full completion result — non-streaming, or streaming accumulated.

    `error` and `partial` are verbum extensions (not emitted by the server).
    Set when a streaming call breaks mid-flight, preserving whatever text
    arrived before the break per S2 λ result_format.
    """

    content: str = ""
    stop: bool = False
    tokens_predicted: int | None = None
    tokens_evaluated: int | None = None
    truncated: bool | None = None
    stopped_word: bool | None = None
    stopped_eos: bool | None = None
    stopped_limit: bool | None = None
    stopping_word: str | None = None
    timings: Timings | None = None
    generation_settings: dict[str, Any] | None = None
    model: str | None = None

    # verbum extensions
    error: str | None = None
    partial: bool = False


class StreamEvent(_Wire):
    """One SSE event from `/completion` with `stream: true`.

    Chunk events carry `content` (delta) and `stop=False`.
    The final event carries `stop=True` plus `timings`, `tokens_predicted`,
    etc. The `error`/`partial` fields are verbum extensions emitted by the
    client when a stream breaks before the server's final event.
    """

    content: str = ""
    stop: bool = False
    tokens_predicted: int | None = None
    tokens_evaluated: int | None = None
    timings: Timings | None = None

    # verbum extensions
    error: str | None = None
    partial: bool = False
