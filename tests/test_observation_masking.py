from __future__ import annotations

from copy import deepcopy

import pytest

from headroom.transforms.observation_masking import (
    _MASKABLE_INPUT_KEYS,
    apply_candidates,
    discover_candidates,
    masking_gate_gain,
)


def _count(text: str) -> int:
    if text.startswith("[Tool result masked:") or text.startswith("[Tool input masked:"):
        return 1
    return len(text.split())


def _write_messages(payload: str, later_turns: int = 3) -> list[dict]:
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "write-1",
                    "name": "Write",
                    "input": {"file_path": "/tmp/x.py", "content": payload},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "write-1", "content": "ok"}
            ],
        },
    ]
    for turn in range(later_turns):
        messages.extend(
            [
                {"role": "assistant", "content": f"turn {turn}"},
                {"role": "user", "content": "continue"},
            ]
        )
    return messages


def _messages(payload: str, later_turns: int = 3) -> list[dict]:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": payload}],
        },
    ]
    for turn in range(later_turns):
        messages.extend(
            [
                {"role": "assistant", "content": f"turn {turn}"},
                {"role": "user", "content": "continue"},
            ]
        )
    return messages


class _Store:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    def store(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("store unavailable")
        return kwargs["explicit_hash"]


@pytest.mark.parametrize("later_turns, expected", [(2, 0), (3, 1), (4, 1)])
def test_age_threshold(later_turns: int, expected: int):
    candidates = discover_candidates(
        _messages("word " * 20, later_turns),
        count_tokens=_count,
        mask_min_tokens=10,
    )
    assert len(candidates) == expected


@pytest.mark.parametrize("tokens, expected", [(9, 0), (10, 1), (11, 1)])
def test_size_threshold(tokens: int, expected: int):
    candidates = discover_candidates(
        _messages("word " * tokens),
        count_tokens=_count,
        mask_min_tokens=10,
    )
    assert len(candidates) == expected


def test_placeholder_and_ccr_content():
    payload = "\n  first\tline with spacing  \n" + "word " * 30
    candidates = discover_candidates(
        _messages(payload), count_tokens=_count, mask_min_tokens=10
    )
    candidate = candidates[0]
    assert "tool=Read" in candidate.marker
    assert f"tokens={_count(payload)}" in candidate.marker
    assert f"bytes={len(payload.encode('utf-8'))}" in candidate.marker
    assert 'head="first line with spacing"' in candidate.marker
    assert f"Retrieve original: hash={candidate.content_hash}" in candidate.marker

    store = _Store()
    result = apply_candidates(_messages(payload), candidates, compression_store=store)
    assert result.masked_count == 1
    assert store.calls[0]["compression_strategy"] == "observation_masking"
    assert store.calls[0]["original"] == payload


def test_idempotence_and_existing_recovery_marker():
    payload = "word " * 30
    messages = _messages(payload)
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    first = apply_candidates(messages, candidates, compression_store=_Store())
    frozen = deepcopy(first.messages)
    second_candidates = discover_candidates(
        first.messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert second_candidates == []
    assert first.messages == frozen

    recovery = _messages("old. Retrieve original: hash=0123456789abcdef01234567 " + payload)
    assert discover_candidates(recovery, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS) == []


def test_non_string_and_store_failure_leave_messages_unchanged():
    messages = _messages("word " * 30)
    messages[1]["content"].append(
        {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text"}]}
    )
    original = deepcopy(messages)
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    store = _Store(fail=True)
    result = apply_candidates(messages, candidates, compression_store=store)
    assert result.messages == original
    assert result.messages is messages
    assert result.masked_count == 0


@pytest.mark.parametrize("later_turns, expected", [(2, 0), (3, 1)])
def test_tool_use_input_age_threshold(later_turns: int, expected: int):
    candidates = discover_candidates(
        _write_messages("word " * 20, later_turns),
        count_tokens=_count,
        mask_min_tokens=10,
        mask_input_keys=_MASKABLE_INPUT_KEYS,
    )
    assert len(candidates) == expected
    if expected:
        assert candidates[0].input_key == "content"


def test_tool_use_input_masks_and_recovers():
    payload = "word " * 30
    messages = _write_messages(payload)
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert [c.input_key for c in candidates] == ["content"]
    marker = candidates[0].marker
    assert marker.startswith("[Tool input masked: tool=Write, key=content")

    store = _Store()
    result = apply_candidates(messages, candidates, compression_store=store)
    assert result.masked_count == 1
    assert store.calls[0]["original"] == payload
    masked_input = result.messages[0]["content"][0]["input"]
    assert masked_input["content"] == marker
    assert masked_input["file_path"] == "/tmp/x.py"
    # source list untouched (copy on write)
    assert messages[0]["content"][0]["input"]["content"] == payload

    second = discover_candidates(result.messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert second == []


def test_tool_use_non_allowlisted_and_non_dict_input_untouched():
    messages = _write_messages("word " * 30)
    messages[0]["content"][0]["input"] = {"command": "word " * 30}
    assert discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS) == []
    messages[0]["content"][0]["input"] = "word " * 30
    assert discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS) == []


def test_tool_use_multiple_keys_same_block():
    messages = _write_messages("word " * 30)
    messages[0]["content"][0]["name"] = "Edit"
    messages[0]["content"][0]["input"] = {
        "file_path": "/tmp/x.py",
        "old_string": "old " * 30,
        "new_string": "new " * 40,
    }
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert sorted(c.input_key for c in candidates if c.input_key) == [
        "new_string",
        "old_string",
    ]
    result = apply_candidates(messages, candidates, compression_store=_Store())
    assert result.masked_count == 2
    masked_input = result.messages[0]["content"][0]["input"]
    assert masked_input["old_string"].startswith("[Tool input masked:")
    assert masked_input["new_string"].startswith("[Tool input masked:")
    assert masked_input["file_path"] == "/tmp/x.py"


class _Policy:
    def net_mutation_gain(self, delta_t, suffix_tokens, expected_reads, p_alive, write_multiplier):  # noqa: ANN001, ANN201
        return delta_t * expected_reads - p_alive * suffix_tokens * write_multiplier


def test_gate_uses_exact_masking_delta():
    candidates = discover_candidates(
        _messages("word " * 30), count_tokens=_count, mask_min_tokens=10
    )
    gain = masking_gate_gain(
        candidates,
        compression_policy=_Policy(),
        suffix_tokens=10,
        expected_reads=4,
        p_alive=0.5,
        write_multiplier=2,
    )
    assert gain == candidates[0].tokens_saved * 4 - 10


def test_gate_prices_through_the_real_policy():
    """The stub above accepts any call shape. The real policy takes
    ``write_multiplier`` keyword-only, and a positional pass raised
    TypeError on every gated request, which the handler swallowed as
    "Optimization failed" and forwarded the request unoptimized."""
    from headroom.transforms.compression_policy import (
        CACHE_WRITE_MULTIPLIER_1H,
        policy_default_payg,
    )

    policy = policy_default_payg()
    candidates = discover_candidates(
        _messages("word " * 30), count_tokens=_count, mask_min_tokens=10
    )
    gain = masking_gate_gain(
        candidates,
        compression_policy=policy,
        suffix_tokens=10,
        expected_reads=4,
        p_alive=0.5,
        write_multiplier=CACHE_WRITE_MULTIPLIER_1H,
    )
    assert gain == policy.net_mutation_gain(
        candidates[0].tokens_saved, 10, 4, 0.5, write_multiplier=CACHE_WRITE_MULTIPLIER_1H
    )


def test_skip_result_tools_leaves_owned_results_alone():
    """Read maturation owns Read results while it runs, so masking must not
    replace a Read before maturation's quiet turns have passed."""
    messages = _messages("word " * 30)
    assert discover_candidates(messages, count_tokens=_count, mask_min_tokens=10)
    assert not discover_candidates(
        messages,
        count_tokens=_count,
        mask_min_tokens=10,
        skip_result_tools=frozenset({"Read"}),
    )


def test_mask_replay_ledger_books_each_masked_block_once():
    """The client re-sends the raw block every turn, so the masked copy goes out
    again each request. Only the first appearance may count as a saving."""
    from headroom.transforms.observation_masking import MaskReplayLedger

    original = _messages("word " * 30)
    candidates = discover_candidates(original, count_tokens=_count, mask_min_tokens=10)
    masked = apply_candidates(original, candidates, compression_store=_Store()).messages
    ledger = MaskReplayLedger()

    assert ledger.replayed_token_debt(original, masked, _count) == 0
    expected = _count("word " * 30) - _count(candidates[0].marker)
    assert ledger.replayed_token_debt(original, masked, _count) == expected
    assert ledger.replayed_token_debt(original, masked, _count) == expected
    # A turn that forwards the block unmasked owes nothing.
    assert ledger.replayed_token_debt(original, original, _count) == 0
