"""llama.cpp HTTP client — sync and async mirror.

Mirrors `specs/llama_server.openapi.yaml` (hand-curated, grown by use per
AGENTS.md S2 λ spec_artifact). Exposes both `Client` and `AsyncClient` so
callers in either runtime can use the same surface.

Streaming uses Server-Sent Events via `httpx-sse`. Partial results on
broken streams are preserved per S2 λ result_format — the stream iterator
yields a final `StreamEvent(error=..., partial=True)` instead of raising,
so probe runners can record whatever text arrived before the break.

Non-streaming calls raise on HTTP errors; the probe runner is responsible
for catching and writing the failed-row JSONL entry.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from types import TracebackType
from typing import Any

import httpx
import httpx_sse
import structlog

from verbum._protocol import (
    CompletionResult,
    HealthStatus,
    ServerProps,
    StreamEvent,
    TokenizeResult,
)
from verbum.config import Settings

__all__ = [
    "AsyncClient",
    "Client",
    "accumulate_stream",
    "accumulate_stream_async",
]

_LOG = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT_S = 120.0
_STREAM_READ_TIMEOUT_S = 600.0  # streams can idle during long predictions


# ─────────────────────────── shared helpers ───────────────────────────


def _build_completion_body(
    prompt: str,
    *,
    n_predict: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int | None = None,
    grammar: str | None = None,
    stop: list[str] | None = None,
    n_probs: int = 0,
    cache_prompt: bool = True,
    stream: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for POST /completion.

    Only fields we actively use are typed; `extra` passes through anything
    else (grow by use — once a new knob becomes standard, promote it to a
    named argument).
    """
    body: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "cache_prompt": cache_prompt,
        "stream": stream,
        "n_probs": n_probs,
    }
    if seed is not None:
        body["seed"] = seed
    if grammar is not None:
        body["grammar"] = grammar
    if stop is not None:
        body["stop"] = stop
    if extra:
        body.update(extra)
    return body


def _parse_sse_data(raw: str) -> dict[str, Any] | None:
    """Decode a single SSE `data:` payload from llama.cpp. Returns None on
    empty / comment lines (keep-alive)."""
    if not raw or raw.isspace():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        _LOG.warning("sse.decode_failed", raw=raw[:200], error=repr(exc))
        return None


def accumulate_stream(events: Iterable[StreamEvent]) -> CompletionResult:
    """Collapse a synchronous `StreamEvent` iterator into a `CompletionResult`.

    Preserves `partial=True` + `error` from a mid-stream break.
    """
    parts: list[str] = []
    final: StreamEvent | None = None
    error: str | None = None
    partial = False
    for ev in events:
        parts.append(ev.content)
        if ev.error is not None:
            error = ev.error
            partial = True
            break
        if ev.stop:
            final = ev
    return _result_from_stream(parts, final, error=error, partial=partial)


async def accumulate_stream_async(
    events: AsyncIterator[StreamEvent],
) -> CompletionResult:
    """Async variant of `accumulate_stream`."""
    parts: list[str] = []
    final: StreamEvent | None = None
    error: str | None = None
    partial = False
    async for ev in events:
        parts.append(ev.content)
        if ev.error is not None:
            error = ev.error
            partial = True
            break
        if ev.stop:
            final = ev
    return _result_from_stream(parts, final, error=error, partial=partial)


def _result_from_stream(
    parts: list[str],
    final: StreamEvent | None,
    *,
    error: str | None,
    partial: bool,
) -> CompletionResult:
    return CompletionResult(
        content="".join(parts),
        stop=final.stop if final else False,
        tokens_predicted=final.tokens_predicted if final else None,
        tokens_evaluated=final.tokens_evaluated if final else None,
        timings=final.timings if final else None,
        error=error,
        partial=partial,
    )


def _default_base_url() -> str:
    """Resolve base URL from Settings when caller doesn't supply one."""
    return str(Settings().llama_server_url)


# ─────────────────────────── sync client ──────────────────────────────


class Client:
    """Synchronous llama.cpp HTTP client.

    Context-manager aware. Use `with Client(...) as c:` or call `.close()`
    explicitly. Defaults to `VERBUM_LLAMA_SERVER_URL` via `Settings`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url or _default_base_url()
        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT_S,
            transport=transport,
        )

    # lifecycle ---------------------------------------------------------

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # endpoints ---------------------------------------------------------

    def health(self) -> HealthStatus:
        r = self._http.get("/health")
        r.raise_for_status()
        return HealthStatus.model_validate(r.json())

    def props(self) -> ServerProps:
        r = self._http.get("/props")
        r.raise_for_status()
        return ServerProps.model_validate(r.json())

    def tokenize(self, content: str, *, add_special: bool = True) -> list[int]:
        r = self._http.post(
            "/tokenize",
            json={"content": content, "add_special": add_special},
        )
        r.raise_for_status()
        return TokenizeResult.model_validate(r.json()).tokens

    def detokenize(self, tokens: list[int]) -> str:
        r = self._http.post("/detokenize", json={"tokens": tokens})
        r.raise_for_status()
        data = r.json()
        return str(data.get("content", ""))

    def complete(
        self,
        prompt: str,
        *,
        n_predict: int = -1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = None,
        grammar: str | None = None,
        stop: list[str] | None = None,
        n_probs: int = 0,
        cache_prompt: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        body = _build_completion_body(
            prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            grammar=grammar,
            stop=stop,
            n_probs=n_probs,
            cache_prompt=cache_prompt,
            stream=False,
            extra=extra,
        )
        r = self._http.post("/completion", json=body)
        r.raise_for_status()
        return CompletionResult.model_validate(r.json())

    def stream_complete(
        self,
        prompt: str,
        *,
        n_predict: int = -1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = None,
        grammar: str | None = None,
        stop: list[str] | None = None,
        n_probs: int = 0,
        cache_prompt: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream completion events via SSE. Partial-result safe.

        If the stream breaks mid-flight (network drop, timeout, server
        error, etc.), the iterator yields one final synthetic event with
        `partial=True` and `error` populated, then terminates cleanly.
        """
        body = _build_completion_body(
            prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            grammar=grammar,
            stop=stop,
            n_probs=n_probs,
            cache_prompt=cache_prompt,
            stream=True,
            extra=extra,
        )
        try:
            with httpx_sse.connect_sse(
                self._http,
                "POST",
                "/completion",
                json=body,
                timeout=_STREAM_READ_TIMEOUT_S,
            ) as source:
                source.response.raise_for_status()
                for sse in source.iter_sse():
                    data = _parse_sse_data(sse.data)
                    if data is None:
                        continue
                    yield StreamEvent.model_validate(data)
                    if data.get("stop"):
                        return
        except (httpx.HTTPError, RuntimeError) as exc:
            _LOG.warning("stream.break", error=repr(exc))
            yield StreamEvent(error=repr(exc), partial=True)


# ─────────────────────────── async client ─────────────────────────────


class AsyncClient:
    """Asynchronous llama.cpp HTTP client. Mirror of `Client`."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url or _default_base_url()
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT_S,
            transport=transport,
        )

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def health(self) -> HealthStatus:
        r = await self._http.get("/health")
        r.raise_for_status()
        return HealthStatus.model_validate(r.json())

    async def props(self) -> ServerProps:
        r = await self._http.get("/props")
        r.raise_for_status()
        return ServerProps.model_validate(r.json())

    async def tokenize(self, content: str, *, add_special: bool = True) -> list[int]:
        r = await self._http.post(
            "/tokenize",
            json={"content": content, "add_special": add_special},
        )
        r.raise_for_status()
        return TokenizeResult.model_validate(r.json()).tokens

    async def detokenize(self, tokens: list[int]) -> str:
        r = await self._http.post("/detokenize", json={"tokens": tokens})
        r.raise_for_status()
        data = r.json()
        return str(data.get("content", ""))

    async def complete(
        self,
        prompt: str,
        *,
        n_predict: int = -1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = None,
        grammar: str | None = None,
        stop: list[str] | None = None,
        n_probs: int = 0,
        cache_prompt: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        body = _build_completion_body(
            prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            grammar=grammar,
            stop=stop,
            n_probs=n_probs,
            cache_prompt=cache_prompt,
            stream=False,
            extra=extra,
        )
        r = await self._http.post("/completion", json=body)
        r.raise_for_status()
        return CompletionResult.model_validate(r.json())

    async def stream_complete(
        self,
        prompt: str,
        *,
        n_predict: int = -1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int | None = None,
        grammar: str | None = None,
        stop: list[str] | None = None,
        n_probs: int = 0,
        cache_prompt: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        body = _build_completion_body(
            prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            grammar=grammar,
            stop=stop,
            n_probs=n_probs,
            cache_prompt=cache_prompt,
            stream=True,
            extra=extra,
        )
        try:
            async with httpx_sse.aconnect_sse(
                self._http,
                "POST",
                "/completion",
                json=body,
                timeout=_STREAM_READ_TIMEOUT_S,
            ) as source:
                source.response.raise_for_status()
                async for sse in source.aiter_sse():
                    data = _parse_sse_data(sse.data)
                    if data is None:
                        continue
                    yield StreamEvent.model_validate(data)
                    if data.get("stop"):
                        return
        except (httpx.HTTPError, RuntimeError) as exc:
            _LOG.warning("stream.break", error=repr(exc))
            yield StreamEvent(error=repr(exc), partial=True)
