"""Decompression-cap rejections must surface as body-too-large, never 400.

``RequestBodyTooLarge`` is a ``ValueError`` subclass, so a generic
``except (json.JSONDecodeError, ValueError)`` handler clause would translate
a decompression-bomb rejection into ``400 invalid_json``. Every handler
path that reads a request body must instead return the configured
body-too-large status (default 413, see ``get_body_too_large_status``), and
the Bedrock adapter must never fail open: its old ``except Exception``
path forwarded the compressed bomb verbatim to the upstream — exactly the
allocation this cap exists to prevent.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.batch import BatchHandlerMixin
from headroom.proxy.handlers.bedrock import BedrockHandlerMixin
from headroom.proxy.handlers.gemini import GeminiHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.helpers import (
    RequestBodyTooLarge,
    get_body_too_large_status,
)

_TOO_LARGE = RequestBodyTooLarge("gzip request body exceeds the 100-byte decompression limit")


class _FakeState:
    auth_mode = None


class _FakeRequest:
    """Minimal Starlette Request stand-in: headers, state, url, body()."""

    def __init__(self, *, path: str = "/v1/messages", headers: dict | None = None) -> None:
        self.headers = headers or {}
        self.state = _FakeState()
        self.url = SimpleNamespace(path=path, query="")
        self.query_params: dict[str, str] = {}

    async def body(self) -> bytes:
        return b"{}"


async def _raise_too_large(request) -> None:  # noqa: ANN001
    raise _TOO_LARGE


def _error_payload(response) -> dict:
    return json.loads(response.body)


class _FakeHttpClient:
    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.post_response = SimpleNamespace(
            status_code=200,
            content=b"{}",
            headers={"content-type": "application/json"},
        )

    async def post(self, url: str, **kwargs):  # noqa: ANN003, ANN201
        self.posts.append({"url": url, **kwargs})
        return self.post_response


class _OpenAIHandler(OpenAIHandlerMixin):
    async def _next_request_id(self) -> str:
        return "req-1"


class _AnthropicHandler(AnthropicHandlerMixin):
    async def _next_request_id(self) -> str:
        return "req-1"


class _BedrockHandler(BedrockHandlerMixin):
    def __init__(self) -> None:
        self.forwarded: list[dict] = []

    def _bedrock_upstream_base(self) -> str:
        return "https://bedrock.example"

    async def _next_request_id(self) -> str:
        return "req-1"

    async def _forward_bedrock(self, **kwargs):  # noqa: ANN003, ANN201
        self.forwarded.append(kwargs)
        from fastapi.responses import Response

        return Response(status_code=599, content=b"should-not-be-forwarded")


class _BatchHandler(BatchHandlerMixin):
    OPENAI_API_URL = "https://openai.example"
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self) -> None:
        self.http_client = _FakeHttpClient()

    async def _next_request_id(self) -> str:
        return "req-1"

    async def _record_request_outcome(self, outcome):
        self.outcome = outcome


class _GeminiHandler(GeminiHandlerMixin):
    async def _next_request_id(self) -> str:
        return "req-1"


# ---------------------------------------------------------------------------
# OpenAI /v1/chat/completions and /v1/responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_bomb_returns_body_too_large_status(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", _raise_too_large)
    response = await _OpenAIHandler().handle_openai_chat(_FakeRequest(path="/v1/chat/completions"))
    assert response.status_code == get_body_too_large_status()
    error = _error_payload(response)["error"]
    assert error["code"] == "request_too_large"
    assert "decompression limit" in error["message"]


@pytest.mark.asyncio
async def test_openai_responses_bomb_returns_body_too_large_status(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.helpers.read_request_json_with_bytes", _raise_too_large)
    response = await _OpenAIHandler().handle_openai_responses(_FakeRequest(path="/v1/responses"))
    assert response.status_code == get_body_too_large_status()
    assert _error_payload(response)["error"]["code"] == "request_too_large"


@pytest.mark.asyncio
async def test_openai_chat_bomb_status_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_BODY_TOO_LARGE_STATUS", "429")
    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", _raise_too_large)
    response = await _OpenAIHandler().handle_openai_chat(_FakeRequest())
    assert response.status_code == 429


# ---------------------------------------------------------------------------
# Anthropic /v1/messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_messages_bomb_returns_body_too_large_status(monkeypatch) -> None:
    async def _noop(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    monkeypatch.setattr("headroom.proxy.helpers.read_request_json_with_bytes", _raise_too_large)
    monkeypatch.setattr("headroom.proxy.handlers.anthropic.emit_stage_timings_log", _noop)
    response = await _AnthropicHandler().handle_anthropic_messages(
        _FakeRequest(path="/v1/messages")
    )
    assert response.status_code == get_body_too_large_status()
    assert _error_payload(response)["error"]["type"] == "request_too_large"


# ---------------------------------------------------------------------------
# Bedrock: must never fail open and forward the expanding body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bedrock_invoke_bomb_never_fails_open(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.helpers.read_request_json_with_bytes", _raise_too_large)
    handler = _BedrockHandler()
    response = await handler.handle_bedrock_invoke(
        _FakeRequest(path="/model/x/invoke"), "amazon.nova-pro-v1:0", stream=False
    )
    assert response.status_code == get_body_too_large_status()
    assert handler.forwarded == []  # the compressed bomb is never forwarded


# ---------------------------------------------------------------------------
# Batch paths (OpenAI create + Google create)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_batch_create_bomb_returns_body_too_large_status(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", _raise_too_large)
    response = await _BatchHandler().handle_batch_create(_FakeRequest(path="/v1/batches"))
    assert response.status_code == get_body_too_large_status()
    assert _error_payload(response)["error"]["code"] == "request_too_large"


@pytest.mark.asyncio
async def test_google_batch_create_bomb_returns_body_too_large_status(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", _raise_too_large)
    response = await _BatchHandler().handle_google_batch_create(
        _FakeRequest(path="/v1beta/models/gemini-pro:batchGenerateContent"),
        "gemini-pro",
    )
    assert response.status_code == get_body_too_large_status()
    assert _error_payload(response)["error"]["code"] == get_body_too_large_status()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_generate_content_bomb_returns_body_too_large_status(
    monkeypatch,
) -> None:
    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", _raise_too_large)
    response = await _GeminiHandler().handle_gemini_generate_content(
        _FakeRequest(path="/v1beta/models/gemini-pro:generateContent"), "gemini-pro"
    )
    assert response.status_code == get_body_too_large_status()
    assert _error_payload(response)["error"]["code"] == get_body_too_large_status()


# ---------------------------------------------------------------------------
# Batch passthrough best-effort read: size-policy rejections must propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_passthrough_re_raises_body_too_large(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.helpers._read_request_body_bytes", _raise_too_large)
    handler = _BatchHandler()
    with pytest.raises(RequestBodyTooLarge):
        await handler._batch_passthrough(_FakeRequest(), {"model": "x"})
    assert handler.http_client.posts == []  # nothing forwarded


@pytest.mark.asyncio
async def test_batch_passthrough_other_read_errors_still_best_effort(monkeypatch) -> None:
    async def _boom(request) -> None:  # noqa: ANN001
        raise RuntimeError("body already consumed upstream")

    monkeypatch.setattr("headroom.proxy.helpers._read_request_body_bytes", _boom)
    handler = _BatchHandler()
    response = await handler._batch_passthrough(_FakeRequest(), {"model": "x"})
    assert response.status_code == 200
    assert handler.http_client.posts  # forwarded via canonical re-serialization
