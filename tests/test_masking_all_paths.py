"""Observation masking coverage across Anthropic forwarding branches."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx

from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.proxy.hybrid_mode import HybridModeConfig, HybridModeDecision, HybridPhase

_STABILITY_SPEC = importlib.util.spec_from_file_location(
    "_masking_stability_helpers",
    Path(__file__).with_name("test_proxy_anthropic_cache_stability.py"),
)
assert _STABILITY_SPEC is not None and _STABILITY_SPEC.loader is not None
_STABILITY = importlib.util.module_from_spec(_STABILITY_SPEC)
_STABILITY_SPEC.loader.exec_module(_STABILITY)
_FakePrefixTracker = _STABILITY._FakePrefixTracker
_make_proxy_client = _STABILITY._make_proxy_client


class _HybridMaskingTracker(_FakePrefixTracker):
    def __init__(self, *, client_prefix_alive: float, frozen_count: int = 0) -> None:
        super().__init__(frozen_count=frozen_count)
        self._client_prefix_alive = client_prefix_alive

    def observe_client_churn(self, messages, head_fingerprint=None):  # noqa: ANN001, ANN201
        return self._client_prefix_alive

    @property
    def hybrid_controller(self):  # noqa: ANN201
        return SimpleNamespace(
            config=replace(
                HybridModeConfig(),
                adaptive_ttl=False,
                subagent_ttl_5m=False,
                observation_masking=True,
                mask_after_turns=0,
                mask_min_tokens=1,
            ),
            decide=lambda **_: HybridModeDecision(
                phase=HybridPhase.LIVE_DELTA,
                frozen_message_count=self._frozen_count,
            ),
        )


class _Policy:
    def __init__(self, gain: float) -> None:
        self.gain = gain

    def net_mutation_gain(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return self.gain


class _Store:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def store(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return kwargs["explicit_hash"]


def _messages() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Read",
                    "input": {"path": "/tmp/example"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "large historical observation " * 200,
                }
            ],
        },
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
    ]


def _without_cache_control(messages: list[dict]) -> list[dict]:
    cleaned = deepcopy(messages)
    for message in cleaned:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    return cleaned


async def _run_request(
    monkeypatch,
    *,
    alive: float,
    gain: float,
    frozen_count: int = 0,
    mode: str = "balanced",
    previous_forwarded: list[dict] | None = None,
):  # noqa: ANN001, ANN202
    captured: dict = {}
    store = _Store()
    messages = _messages()

    test_client = _make_proxy_client()
    app = test_client.app
    proxy = app.state.proxy
    try:
        proxy.config.optimize = True
        proxy.config.mode = mode
        proxy.config.image_optimize = False
        proxy.config.ccr_inject_tool = True
        tracker = _HybridMaskingTracker(
            client_prefix_alive=alive, frozen_count=frozen_count
        )
        if previous_forwarded is not None:
            tracker._last_forwarded_messages = deepcopy(previous_forwarded)
        # The fake tracker has no prior turn to confirm against.
        proxy._strict_previous_turn_frozen_count = lambda messages, count: count
        proxy.session_tracker_store.compute_session_id = (
            lambda request, model, messages, system=None: f"mask-all-paths-{alive}"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider, *_: tracker
        proxy.anthropic_pipeline.apply = lambda **kwargs: SimpleNamespace(
            messages=kwargs["messages"],
            transforms_applied=[],
            timing={},
            tokens_before=1000,
            tokens_after=1000,
            waste_signals=None,
        )
        monkeypatch.setattr(
            "headroom.transforms.compression_policy.resolve_policy",
            lambda auth_mode: _Policy(gain),
        )
        monkeypatch.setattr(
            "headroom.cache.compression_store.get_compression_store",
            lambda: store,
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_mask_paths",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "tools": [
                        {
                            "name": CCR_TOOL_NAME,
                            "description": "Retrieve compressed content",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    "messages": messages,
                },
            )
    finally:
        test_client.close()

    assert response.status_code == 200
    return captured["body"]["messages"], deepcopy(messages), store


async def test_ccr_tool_already_injected_path_applies_admitted_masking(
    monkeypatch,
) -> None:
    forwarded, original, store = await _run_request(
        monkeypatch, alive=0.0, gain=-1.0
    )

    assert forwarded != original
    assert forwarded[1]["content"][0]["content"].startswith("[Tool result masked:")
    assert len(store.calls) == 1
    assert store.calls[0]["compression_strategy"] == "observation_masking"
    assert store.calls[0]["explicit_hash"]


async def test_ccr_tool_already_injected_path_is_byte_identical_when_declined(
    monkeypatch,
) -> None:
    forwarded, original, store = await _run_request(
        monkeypatch, alive=1.0, gain=-1.0
    )

    # Cache-control placement is a separate forwarding normalization. Masking
    # itself must leave every message byte unchanged when its gate declines.
    assert _without_cache_control(forwarded) == original
    assert store.calls == []


async def test_admitted_masking_leaves_frozen_prefix_bytes_unchanged(
    monkeypatch,
) -> None:
    forwarded, original, store = await _run_request(
        monkeypatch, alive=1.0, gain=1.0, frozen_count=3, mode="hybrid"
    )

    assert _without_cache_control(forwarded) == original
    assert store.calls == []


async def test_declined_gate_still_replays_masks_the_cached_prefix_carries(
    monkeypatch,
) -> None:
    masked_turn, _, _ = await _run_request(
        monkeypatch, alive=1.0, gain=1.0, mode="hybrid"
    )
    masked_turn = _without_cache_control(masked_turn)

    forwarded, original, _ = await _run_request(
        monkeypatch,
        alive=1.0,
        gain=-1.0,
        frozen_count=3,
        mode="hybrid",
        previous_forwarded=masked_turn,
    )

    assert _without_cache_control(forwarded) != original
    assert _without_cache_control(forwarded)[:3] == masked_turn[:3]
