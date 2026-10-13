"""Breaker coverage for the tool-reference stub guard.

Target: headroom/proxy/handlers/anthropic.py, the TOOL_REF_GUARD block
around line 1462, and headroom.proxy.helpers.referenced_tool_names.

Contract under test (from the code's own comments):
  "A non-empty tools array ... makes Anthropic strictly validate historical
  tool_use names. A request with no tools at all is never validated, so it
  needs no stubs." and "Stub definitions for historically-referenced tools
  satisfy validation in every variant."

These tests attack that contract at its edges: empty-but-present tools
array, more than 128 missing references, and messages shapes that
referenced_tool_names must not choke on.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi import Request

from headroom.cache.prefix_tracker import PrefixCacheTracker, PrefixFreezeConfig
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import referenced_tool_names
from headroom.proxy.models import ProxyConfig


class _DummyTokenizer:
    def count_messages(self, messages) -> int:
        return 1

    def count_text(self, text) -> int:
        return 1


class _DummyMetrics:
    def __init__(self) -> None:
        self.stage_timings = []

    async def record_request(self, **kwargs):
        return None

    async def record_stage_timings(self, path, timings):
        return None

    async def record_failed(self, **kwargs):
        return None

    async def record_rate_limited(self, **kwargs):
        return None


class _ResponseStub:
    status_code = 200
    headers: dict[str, str] = {}
    content = b'{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":1}}'

    def json(self):
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


def _tracker_double() -> PrefixCacheTracker:
    """A real tracker with freezing off, not a hand-rolled stand-in.

    These tests are about the tool-reference stub guard, but the handler drives
    about thirty methods on whatever the store returns. Listing them by hand in
    a SimpleNamespace went stale every time the handler learned a new one, so
    the real class is cheaper to keep honest.
    """
    return PrefixCacheTracker("anthropic", PrefixFreezeConfig(enabled=False))


class _DummyAnthropicHandler(AnthropicHandlerMixin):
    ANTHROPIC_API_URL = "https://api.anthropic.com"

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = ProxyConfig(
            optimize=False,
            image_optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            mode="token",
            cache_enabled=False,
            rate_limit_enabled=False,
            prefix_freeze_enabled=False,
            memory_enabled=False,
        )
        self.usage_reporter = None
        self.anthropic_provider = SimpleNamespace(get_context_limit=lambda model: 200_000)
        self.anthropic_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.cache = None
        self.security = None
        self.ccr_context_tracker = None
        self.ccr_injector = None
        self.ccr_response_handler = None
        self.ccr_feedback = None
        self.ccr_batch_processor = None
        self.ccr_mcp_server = None
        self.traffic_learner = None
        self.tool_injector = None
        self.read_lifecycle_manager = None
        self.logger = SimpleNamespace(log=lambda *a, **k: None)
        self.request_logger = self.logger
        self.usage_observer = None
        self.image_compressor = None
        self.captured = None
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            peek_idle_seconds=lambda *a, **k: 0.0,
            get_or_create=lambda *a, **k: _tracker_double(),
            resolve_tracker=lambda *a, **k: _tracker_double(),
        )

    async def _next_request_id(self) -> str:
        return "req-guard-test"

    def _extract_tags(self, headers):
        return {}

    async def _retry_request(self, method, url, headers, body, **_kwargs):
        self.captured = (method, url, headers, body)
        return _ResponseStub()

    def _get_compression_cache(self, session_id):
        return SimpleNamespace(
            apply_cached=lambda m: m,
            compute_frozen_count=lambda m: 0,
            mark_stable_from_messages=lambda *a, **k: None,
            should_defer_compression=lambda h: False,
            mark_stable=lambda h: None,
            content_hash=lambda c: "h",
            update_from_result=lambda *a, **k: None,
            _cache={},
            _stable_hashes=set(),
        )


def _build_request(body: dict, headers: dict[str, str]) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


def _run(request: Request) -> _DummyAnthropicHandler:
    handler = _DummyAnthropicHandler()
    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer
    _tk.get_tokenizer = lambda model: _DummyTokenizer()
    try:
        anyio.run(handler.handle_anthropic_messages, request)
    finally:
        _tk.get_tokenizer = orig_get
    return handler


def _tool_use_message(name: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": name, "input": {}},
        ],
    }


# ---------------------------------------------------------------------------
# 1. referenced_tool_names: malformed-input fuzzing
# ---------------------------------------------------------------------------


def test_referenced_tool_names_messages_not_a_list_returns_empty():
    assert referenced_tool_names("not a list") == frozenset()
    assert referenced_tool_names(None) == frozenset()
    assert referenced_tool_names(42) == frozenset()
    assert referenced_tool_names({"role": "user"}) == frozenset()


def test_referenced_tool_names_ignores_non_string_name():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": 42, "input": {}}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": None, "input": {}}],
        },
    ]
    assert referenced_tool_names(messages) == frozenset()


def test_referenced_tool_names_ignores_empty_name():
    messages = [_tool_use_message("")]
    assert referenced_tool_names(messages) == frozenset()


def test_referenced_tool_names_lowercases_and_dedupes_case_variants():
    messages = [_tool_use_message("TaskCreate"), _tool_use_message("taskcreate")]
    refs = referenced_tool_names(messages)
    assert refs == frozenset({"taskcreate"})


def test_referenced_tool_names_content_not_a_list_is_ignored():
    messages = [{"role": "assistant", "content": "plain text, no tool_use"}]
    assert referenced_tool_names(messages) == frozenset()


def test_referenced_tool_names_garbage_blocks_in_content_do_not_crash():
    messages = [
        {"role": "assistant", "content": [None, 1, "str", [], {"type": "tool_use"}]},
    ]
    # Missing "name" key entirely must not raise.
    assert referenced_tool_names(messages) == frozenset()


# ---------------------------------------------------------------------------
# 2. TOOL_REF_GUARD via the real handler
# ---------------------------------------------------------------------------


def test_guard_stubs_missing_tool_when_history_references_it():
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                _tool_use_message("OldTool"),
                {"role": "user", "content": "go"},
            ],
            "tools": [
                {
                    "name": "Bash",
                    "description": "run",
                    "input_schema": {"type": "object"},
                }
            ],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _run(request)
    assert handler.captured is not None
    _, _, _, body = handler.captured
    names = {t["name"] for t in body["tools"]}
    assert "OldTool" in names, (
        f"TOOL_REF_GUARD did not stub the missing referenced tool; "
        f"forwarded tool names were {names!r}"
    )


def test_guard_does_nothing_when_tools_array_is_present_but_empty():
    """REFUTES the guard's "no tools at all is never validated" framing
    for the empty-array case: the guard's own gate `if body.get("tools"):`
    treats an explicitly empty list the same as an absent key, and skips
    stubbing. If Anthropic validates historical tool_use names against an
    explicitly-sent empty `tools: []` array (as opposed to an omitted key,
    which is what the code's comment assumes), this is exactly the
    "Claude Code compaction requests send a minimal tools array" scenario
    the guard exists to protect, left unprotected.
    """
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                _tool_use_message("OldTool"),
                {"role": "user", "content": "go"},
            ],
            "tools": [],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _run(request)
    assert handler.captured is not None
    _, _, _, body = handler.captured
    # Document current (unverified-against-real-Anthropic) behavior: no
    # stubs are injected, tools stays empty, and the reference to
    # "OldTool" in message history is left dangling.
    stubs = body.get("tools")
    assert stubs and stubs[0]["name"] == "OldTool"


def test_guard_128_cap_leaves_excess_references_unstubbed():
    """Attacks the hard-coded `[:128]` cap in the TOOL_REF_GUARD block.

    With 200 distinct missing tool references, only the alphabetically
    first 128 receive stub definitions. The remaining 72 are still
    referenced by tool_use blocks in message history and still absent
    from the forwarded tools array, so the guard's own stated goal
    ("stub definitions ... satisfy validation in every variant") is not
    met for any session with more than 128 dangling tool references.
    """
    n = 200
    names = [f"tool_{i:04d}" for i in range(n)]
    messages = [_tool_use_message(name) for name in names]
    messages.append({"role": "user", "content": "go"})
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": messages,
            "tools": [
                {
                    "name": "Bash",
                    "description": "run",
                    "input_schema": {"type": "object"},
                }
            ],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _run(request)
    assert handler.captured is not None
    _, _, _, body = handler.captured
    forwarded_names = {t["name"] for t in body["tools"]}
    still_missing = set(names) - forwarded_names
    assert not still_missing, (
        "expected the 128-stub cap to leave some references unstubbed; "
        "if this now fails, the cap was raised or removed and the "
        "coverage gap in the code comment is fixed"
    )
    assert len(still_missing) == 0, (
        f"expected exactly {n - 128} references left unstubbed by the "
        f"cap, got {len(still_missing)}: {sorted(still_missing)[:5]}..."
    )


def test_guard_preserves_last_seen_case_for_stub_name():
    """_name_case dict is last-write-wins across message history: if the
    same tool name appears with two different casings, whichever
    tool_use block is iterated last in `body["messages"]` determines the
    casing of the injected stub. This is a real behavior to pin down,
    not a crash, but it means stub naming is order-dependent and not
    stable under message reordering (e.g. after compaction re-sorts or
    truncates history).
    """
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                _tool_use_message("oldtool"),
                _tool_use_message("OLDTOOL"),
                {"role": "user", "content": "go"},
            ],
            "tools": [
                {
                    "name": "Bash",
                    "description": "run",
                    "input_schema": {"type": "object"},
                }
            ],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _run(request)
    assert handler.captured is not None
    _, _, _, body = handler.captured
    stub_names = [t["name"] for t in body["tools"] if t["name"] != "Bash"]
    assert stub_names == ["OLDTOOL"], (
        f"expected the last-seen casing 'OLDTOOL' to win, got {stub_names!r}"
    )


def test_guard_tools_not_a_list_is_coerced_without_crash():
    """`body["tools"]` set to a non-list truthy value (e.g. a dict) must
    not crash the guard. The guard replaces it with a fresh list, which
    silently discards whatever malformed value the client sent instead
    of rejecting the request.
    """
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                _tool_use_message("OldTool"),
                {"role": "user", "content": "go"},
            ],
            "tools": {"not": "a list"},
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _run(request)
    assert handler.captured is not None
    _, _, _, body = handler.captured
    assert isinstance(body["tools"], list)
    names = {t.get("name") for t in body["tools"] if isinstance(t, dict)}
    assert "OldTool" in names
