"""Pure-function tests for the replay corpus audit."""

from __future__ import annotations

import json

from headroom.proxy.replay_audit import (
    AuditSummary,
    audit_session,
    group_sessions,
    prefix_divergence,
    session_key,
    turn_from_record,
)


class _Tok:
    """Tokenizer stand-in: one token per character of the JSON form."""

    def count_text(self, text: str) -> int:
        return len(text)

    def count_messages(self, messages):
        return sum(len(json.dumps(m, sort_keys=True)) for m in messages)


def _record(session: str, ts: float, model: str = "claude-opus-5", n_msgs: int = 1, **usage):
    return {
        "request_id": f"r{ts}",
        "model": model,
        "timestamp": ts,
        "request": {
            "model": model,
            "metadata": {"user_id": json.dumps({"device_id": "d", "session_id": session})},
            "system": "sys",
            "tools": [{"name": "Read"}],
            "messages": [{"role": "user", "content": f"m{i}"} for i in range(n_msgs)],
        },
        "response_usage": {
            "input_tokens": usage.get("input", 10),
            "cache_read_input_tokens": usage.get("read", 0),
            "cache_creation_input_tokens": usage.get("write", 0),
        },
        "transforms_applied": [],
    }


SID_A = "2c0f0f85-6ab0-46e3-a867-f874385918b7"
SID_B = "11111111-2222-3333-4444-555555555555"


def test_session_key_parses_json_string_and_dict_forms() -> None:
    assert session_key(_record(SID_A, 1.0)) == SID_A
    as_dict = _record(SID_A, 1.0)
    as_dict["request"]["metadata"]["user_id"] = {"session_id": SID_B}
    assert session_key(as_dict) == SID_B
    assert session_key({"request": {"metadata": {"user_id": "no id here"}}}) is None
    assert session_key({"request": {}}) is None


def test_turn_from_record_sums_every_billed_tier() -> None:
    turn = turn_from_record(_record(SID_A, 1.0, input=5, read=100, write=20))
    assert turn is not None
    assert turn.billed_input == 125
    assert turn.cache_read == 100
    assert turn.cache_write == 20


def test_group_sessions_sorts_by_timestamp_and_splits_on_model() -> None:
    records = [
        _record(SID_A, 3.0),
        _record(SID_A, 1.0),
        _record(SID_A, 2.0, model="claude-sonnet-5"),
        _record(SID_B, 5.0),
        _record(SID_A, 4.0, model="claude-haiku-4-5-20251001"),
        {"request": {"messages": []}, "timestamp": 9.0},
    ]
    groups = group_sessions(records, min_turns=1)
    assert set(groups) == {
        (SID_A, "claude-opus-5"),
        (SID_A, "claude-sonnet-5"),
        (SID_B, "claude-opus-5"),
    }
    assert [t.timestamp for t in groups[(SID_A, "claude-opus-5")]] == [1.0, 3.0]
    assert group_sessions(records, min_turns=2).keys() == {(SID_A, "claude-opus-5")}
    with_haiku = group_sessions(records, min_turns=1, exclude_model_prefixes=())
    assert (SID_A, "claude-haiku-4-5-20251001") in with_haiku
    assert group_sessions(records, model_filter="claude-sonnet-5").keys() == {
        (SID_A, "claude-sonnet-5")
    }


def test_prefix_divergence_orders_sections_like_the_provider() -> None:
    base = {"tools": [1], "system": "s", "messages": [{"a": 1}, {"b": 2}]}
    assert prefix_divergence(None, base).section == "first"
    assert prefix_divergence(base, dict(base)).section == "none"
    assert prefix_divergence(base, {**base, "tools": [2]}).section == "tools"
    assert prefix_divergence(base, {**base, "system": "x"}).section == "system"
    extended = {**base, "messages": base["messages"] + [{"c": 3}]}
    div = prefix_divergence(base, extended)
    assert div.section == "none" and div.shared_messages == 2
    edited = {**base, "messages": [{"a": 1}, {"b": 9}, {"c": 3}]}
    div = prefix_divergence(base, edited)
    assert div.section == "messages_tail" and div.message_index == 1 and div.shared_messages == 1
    deep = {**base, "messages": [{"a": 9}, {"b": 2}, {"c": 3}]}
    assert prefix_divergence(base, deep).section == "messages"


def test_prefix_divergence_ignores_moved_cache_control_markers() -> None:
    marked = {
        "tools": [{"name": "Read", "cache_control": {"type": "ephemeral"}}],
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}],
            },
        ],
    }
    moved = {
        "tools": [{"name": "Read"}],
        "system": [{"type": "text", "text": "s"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}}],
            },
        ],
    }
    div = prefix_divergence(marked, moved)
    assert div.section == "none" and div.shared_messages == 2


def test_audit_session_predicts_reads_from_the_stable_prefix() -> None:
    turns = group_sessions(
        [
            _record(SID_A, 1.0, n_msgs=1, input=50),
            _record(SID_A, 2.0, n_msgs=2, input=10, read=40),
            _record(SID_A, 3.0, n_msgs=3, input=10, read=50),
        ]
    )[(SID_A, "claude-opus-5")]
    audits = audit_session(turns, _Tok())
    assert [a.divergence.section for a in audits] == ["first", "none", "none"]
    assert audits[0].predicted_stable_tokens == 0
    assert audits[1].predicted_stable_tokens > 0
    assert audits[1].predicted_stable_tokens < audits[1].local_tokens
    assert audits[2].predicted_stable_tokens > audits[1].predicted_stable_tokens
    summary = AuditSummary()
    summary.add(audits)
    assert summary.turns == 3
    assert summary.billed_input == 160
    assert summary.cache_read == 90
    assert 0 < summary.predicted_read_fraction < 1
    assert summary.as_dict()["divergence_by_section"] == {"first": 1, "none": 2}


def test_audit_session_zeroes_prediction_when_the_head_changes() -> None:
    first = _record(SID_A, 1.0)
    second = _record(SID_A, 2.0, n_msgs=2)
    second["request"]["system"] = "different"
    turns = group_sessions([first, second])[(SID_A, "claude-opus-5")]
    audits = audit_session(turns, _Tok())
    assert audits[1].divergence.section == "system"
    assert audits[1].predicted_stable_tokens == 0
