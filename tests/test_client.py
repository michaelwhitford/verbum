"""Client tests — sync + async, mocked via `httpx.MockTransport`.

No running server required. Verifies:
  - non-streaming endpoints parse correctly
  - streaming yields events and terminates on `stop: true`
  - stream breaks emit `partial=True` event rather than raising
  - async mirror matches sync behaviour
  - `accumulate_stream*` collapses events into `CompletionResult`
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from verbum import AsyncClient, Client, CompletionResult, StreamEvent
from verbum.client import accumulate_stream, accumulate_stream_async

# ─────────────────────────── fixtures / helpers ────────────────────────


def _sse_body(events: list[dict]) -> bytes:
    """Encode a list of dicts as an SSE text/event-stream body."""
    lines: list[str] = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}")
        lines.append("")  # blank line terminates event
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _handler(responses: dict[tuple[str, str], httpx.Response]):
    """Build an httpx MockTransport handler from a (method, path) → response map."""

    def handle(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in responses:
            raise AssertionError(f"unexpected request: {key}")
        return responses[key]

    return handle


# ─────────────────────────── sync tests ────────────────────────────────


def test_health_parses() -> None:
    transport = httpx.MockTransport(
        _handler({("GET", "/health"): httpx.Response(200, json={"status": "ok"})})
    )
    with Client(base_url="http://srv", transport=transport) as c:
        h = c.health()
    assert h.status == "ok"


def test_props_tolerates_unknown_fields() -> None:
    payload = {
        "n_ctx": 32768,
        "total_slots": 1,
        "model_path": "/models/qwen3-35b-a3b.gguf",
        "chat_template": "...",
        "default_generation_settings": {"temperature": 0.0},
        "some_future_field": [1, 2, 3],  # unknown; must survive
    }
    transport = httpx.MockTransport(
        _handler({("GET", "/props"): httpx.Response(200, json=payload)})
    )
    with Client(base_url="http://srv", transport=transport) as c:
        p = c.props()
    assert p.n_ctx == 32768
    assert p.model_path is not None and p.model_path.endswith(".gguf")
    # unknown field preserved via extra="allow"
    dumped = p.model_dump()
    assert dumped["some_future_field"] == [1, 2, 3]


def test_tokenize_detokenize_roundtrip() -> None:
    transport = httpx.MockTransport(
        _handler(
            {
                ("POST", "/tokenize"): httpx.Response(200, json={"tokens": [1, 2, 3]}),
                ("POST", "/detokenize"): httpx.Response(200, json={"content": "hi"}),
            }
        )
    )
    with Client(base_url="http://srv", transport=transport) as c:
        assert c.tokenize("hi") == [1, 2, 3]
        assert c.detokenize([1, 2, 3]) == "hi"


def test_complete_nonstreaming() -> None:
    payload = {
        "content": "λx. x",
        "stop": True,
        "tokens_predicted": 6,
        "tokens_evaluated": 12,
        "timings": {"predicted_ms": 120.5, "predicted_n": 6},
    }
    transport = httpx.MockTransport(
        _handler({("POST", "/completion"): httpx.Response(200, json=payload)})
    )
    with Client(base_url="http://srv", transport=transport) as c:
        r = c.complete("identity", n_predict=6)
    assert isinstance(r, CompletionResult)
    assert r.content == "λx. x"
    assert r.stop is True
    assert r.tokens_predicted == 6
    assert r.timings is not None and r.timings.predicted_ms == 120.5
    assert r.partial is False
    assert r.error is None


def test_complete_raises_on_http_error() -> None:
    transport = httpx.MockTransport(
        _handler({("POST", "/completion"): httpx.Response(500, json={"error": "boom"})})
    )
    with (
        Client(base_url="http://srv", transport=transport) as c,
        pytest.raises(httpx.HTTPStatusError),
    ):
        c.complete("x")


def test_stream_complete_yields_and_terminates() -> None:
    events = [
        {"content": "λ", "stop": False},
        {"content": "x", "stop": False},
        {"content": ". x", "stop": False},
        {
            "content": "",
            "stop": True,
            "tokens_predicted": 4,
            "timings": {"predicted_ms": 99.0},
        },
    ]
    transport = httpx.MockTransport(
        _handler(
            {
                ("POST", "/completion"): httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse_body(events),
                )
            }
        )
    )
    with Client(base_url="http://srv", transport=transport) as c:
        collected: list[StreamEvent] = list(c.stream_complete("identity"))
    assert len(collected) == 4
    assert collected[0].content == "λ"
    assert collected[-1].stop is True
    assert collected[-1].tokens_predicted == 4
    assert all(ev.error is None for ev in collected)


def test_stream_complete_handles_break_without_raising() -> None:
    """Server returns 500 mid-handshake; iterator yields partial event."""
    transport = httpx.MockTransport(
        _handler({("POST", "/completion"): httpx.Response(500, json={"error": "boom"})})
    )
    with Client(base_url="http://srv", transport=transport) as c:
        events = list(c.stream_complete("identity"))
    assert len(events) == 1
    assert events[0].partial is True
    assert events[0].error is not None
    assert events[0].content == ""


def test_accumulate_stream_preserves_partial() -> None:
    events: Iterator[StreamEvent] = iter(
        [
            StreamEvent(content="λx"),
            StreamEvent(content=". "),
            StreamEvent(error="RemoteProtocolError('disconnected')", partial=True),
        ]
    )
    r = accumulate_stream(events)
    assert r.content == "λx. "
    assert r.partial is True
    assert r.error is not None
    assert r.stop is False


def test_accumulate_stream_full_path_sets_final_timings() -> None:
    events = [
        StreamEvent(content="a"),
        StreamEvent(content="b"),
        StreamEvent(content="", stop=True, tokens_predicted=2),
    ]
    r = accumulate_stream(events)
    assert r.content == "ab"
    assert r.stop is True
    assert r.tokens_predicted == 2
    assert r.partial is False
    assert r.error is None


# ─────────────────────────── async tests ───────────────────────────────


async def test_async_health() -> None:
    transport = httpx.MockTransport(
        _handler({("GET", "/health"): httpx.Response(200, json={"status": "ok"})})
    )
    async with AsyncClient(base_url="http://srv", transport=transport) as c:
        h = await c.health()
    assert h.status == "ok"


async def test_async_complete_nonstreaming() -> None:
    payload = {"content": "λx. x", "stop": True, "tokens_predicted": 6}
    transport = httpx.MockTransport(
        _handler({("POST", "/completion"): httpx.Response(200, json=payload)})
    )
    async with AsyncClient(base_url="http://srv", transport=transport) as c:
        r = await c.complete("identity")
    assert r.content == "λx. x"
    assert r.stop is True


async def test_async_stream_complete_yields_and_terminates() -> None:
    events = [
        {"content": "λ", "stop": False},
        {"content": "y", "stop": False},
        {"content": "", "stop": True, "tokens_predicted": 2},
    ]
    transport = httpx.MockTransport(
        _handler(
            {
                ("POST", "/completion"): httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse_body(events),
                )
            }
        )
    )
    async with AsyncClient(base_url="http://srv", transport=transport) as c:
        collected = [ev async for ev in c.stream_complete("identity")]
    assert len(collected) == 3
    assert collected[-1].stop is True


async def test_async_stream_complete_handles_break() -> None:
    transport = httpx.MockTransport(
        _handler({("POST", "/completion"): httpx.Response(500, json={"error": "boom"})})
    )
    async with AsyncClient(base_url="http://srv", transport=transport) as c:
        events = [ev async for ev in c.stream_complete("identity")]
    assert len(events) == 1
    assert events[0].partial is True
    assert events[0].error is not None


async def test_accumulate_stream_async_collapses() -> None:
    async def gen():
        yield StreamEvent(content="a")
        yield StreamEvent(content="b")
        yield StreamEvent(content="", stop=True, tokens_predicted=2)

    r = await accumulate_stream_async(gen())
    assert r.content == "ab"
    assert r.stop is True
    assert r.tokens_predicted == 2
