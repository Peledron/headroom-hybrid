"""A prefix_change names WHICH cache-key segment moved.

`reason=prefix_change` says the cached bytes moved; `changed_segment` says where,
which is the difference between an unexplained cost and a fixable one: a tools
change usually means the tool set moved mid-session (MCP server connecting late,
async plugin/skill load), a system change means the prompt was rebuilt, a messages
change means history was rewritten rather than appended to.

The mechanism is provider-neutral on purpose — `classify_cache_miss` is shared by
the Anthropic and OpenAI handlers, so segment names/ordering are the caller's
(Anthropic: tools+system; OpenAI chat: tools only, system lives in messages).
"""

from __future__ import annotations

from headroom.cache.prefix_tracker import (
    MISS_PREFIX_CHANGE,
    MISS_TTL_EXPIRY,
    MISS_UNKNOWN,
    SEGMENT_MESSAGES,
    SEGMENT_SYSTEM,
    SEGMENT_TOOLS,
    PrefixCacheTracker,
    segment_fingerprint,
)

PREFIX = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
TOOLS = [{"name": "read", "input_schema": {"type": "object"}}]
SYSTEM = [{"type": "text", "text": "you are a coding agent"}]


def _warm_tracker(provider: str = "anthropic") -> PrefixCacheTracker:
    """A tracker that believes last turn left a cacheable prefix."""
    t = PrefixCacheTracker(provider)
    t._cached_token_count = 50_000
    t._last_forwarded_messages = [dict(m) for m in PREFIX]
    return t


def _seed(t: PrefixCacheTracker, segments: dict) -> None:
    """Establish the baseline hashes (turn N-1), reported as a hit."""
    t.classify_cache_miss(
        cache_read_tokens=50_000, current_forwarded_messages=PREFIX, segments=segments
    )


def test_tools_change_is_named() -> None:
    t = _warm_tracker()
    _seed(t, {SEGMENT_TOOLS: TOOLS, SEGMENT_SYSTEM: SYSTEM})
    grew = [*TOOLS, {"name": "mcp__late_server__query", "input_schema": {}}]
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=PREFIX,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: grew, SEGMENT_SYSTEM: SYSTEM},
    )
    assert miss.is_miss and miss.reason == MISS_PREFIX_CHANGE
    assert miss.changed_segment == SEGMENT_TOOLS


def test_system_change_is_named() -> None:
    t = _warm_tracker()
    _seed(t, {SEGMENT_TOOLS: TOOLS, SEGMENT_SYSTEM: SYSTEM})
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=PREFIX,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: TOOLS, SEGMENT_SYSTEM: [{"type": "text", "text": "rebuilt"}]},
    )
    assert miss.changed_segment == SEGMENT_SYSTEM


def test_earliest_segment_wins_when_several_move() -> None:
    """tools renders before system, so tools is the root cause, not a consequence."""
    t = _warm_tracker()
    _seed(t, {SEGMENT_TOOLS: TOOLS, SEGMENT_SYSTEM: SYSTEM})
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=PREFIX,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: [], SEGMENT_SYSTEM: [{"type": "text", "text": "also moved"}]},
    )
    assert miss.changed_segment == SEGMENT_TOOLS


def test_messages_is_the_fallback_when_no_named_segment_moved() -> None:
    t = _warm_tracker()
    _seed(t, {SEGMENT_TOOLS: TOOLS, SEGMENT_SYSTEM: SYSTEM})
    rewritten = [{"role": "user", "content": [{"type": "text", "text": "REWRITTEN"}]}]
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=rewritten,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: TOOLS, SEGMENT_SYSTEM: SYSTEM},
    )
    assert miss.reason == MISS_PREFIX_CHANGE
    assert miss.changed_segment == SEGMENT_MESSAGES


def test_omitting_segments_keeps_previous_behavior() -> None:
    """Callers that pass nothing still get prefix_change, just unnamed."""
    t = _warm_tracker()
    rewritten = [{"role": "user", "content": [{"type": "text", "text": "REWRITTEN"}]}]
    miss = t.classify_cache_miss(
        cache_read_tokens=0, current_forwarded_messages=rewritten, idle_seconds=1.0
    )
    assert miss.reason == MISS_PREFIX_CHANGE
    assert miss.changed_segment == SEGMENT_MESSAGES


def test_first_sighting_cannot_claim_a_change() -> None:
    """No baseline for a segment -> we must not blame it.

    With a stable message prefix and no segment baseline there is nothing to
    attribute, so this stays `unknown` rather than inventing a culprit.
    """
    t = _warm_tracker()
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=PREFIX,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: TOOLS},
    )
    assert miss.reason == MISS_UNKNOWN
    assert miss.changed_segment is None


def test_segment_blind_call_drops_the_baseline() -> None:
    """A mixed session must not blame a change on the wrong turn.

    Hashes only roll forward on calls that pass segments. If a streaming turn
    (segment-blind) sat between two segment-aware turns, keeping the old baseline
    would report the change on the later turn instead of when it happened. Dropping
    the baseline turns that false positive into an honest "can't tell".
    """
    t = _warm_tracker()
    _seed(t, {SEGMENT_TOOLS: TOOLS})
    assert t._last_segment_hashes  # baseline established

    # A turn that cannot report segments.
    t.classify_cache_miss(cache_read_tokens=50_000, current_forwarded_messages=PREFIX)
    assert t._last_segment_hashes == {}  # forgotten, not stale

    # Tools differ from the original baseline, but we no longer claim to know.
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=PREFIX,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: [{"name": "totally_different"}]},
    )
    assert miss.changed_segment is None
    assert miss.reason == MISS_UNKNOWN


def test_ttl_expiry_does_not_name_a_segment() -> None:
    """The entry was already gone; a coincident edit didn't cause the miss."""
    t = _warm_tracker()
    _seed(t, {SEGMENT_TOOLS: TOOLS})
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=PREFIX,
        idle_seconds=99_999.0,  # far beyond any TTL
        segments={SEGMENT_TOOLS: []},  # changed too, but moot
    )
    assert miss.reason == MISS_TTL_EXPIRY
    assert miss.changed_segment is None


def test_hit_names_no_segment() -> None:
    t = _warm_tracker()
    _seed(t, {SEGMENT_TOOLS: TOOLS})
    miss = t.classify_cache_miss(
        cache_read_tokens=50_000,
        current_forwarded_messages=PREFIX,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: []},
    )
    assert not miss.is_miss
    assert miss.changed_segment is None


def test_openai_shape_needs_no_system_segment() -> None:
    """OpenAI chat keeps the system prompt in messages — tools-only is valid."""
    t = _warm_tracker("openai")
    _seed(t, {SEGMENT_TOOLS: TOOLS})
    miss = t.classify_cache_miss(
        cache_read_tokens=0,
        current_forwarded_messages=PREFIX,
        idle_seconds=1.0,
        segments={SEGMENT_TOOLS: [{"name": "shell"}]},
    )
    assert miss.changed_segment == SEGMENT_TOOLS


# ---------------------------------------------------------------- fingerprinting


def test_moved_cache_control_is_not_a_change() -> None:
    """Clients re-mark the breakpoint every turn; that must not read as a change."""
    a = [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    b = [{"type": "text", "text": "sys"}]
    assert segment_fingerprint(a) == segment_fingerprint(b)


def test_key_order_is_not_a_change() -> None:
    a = {"name": "read", "description": "d"}
    b = {"description": "d", "name": "read"}
    assert segment_fingerprint(a) == segment_fingerprint(b)


def test_real_content_change_is_a_change() -> None:
    assert segment_fingerprint(TOOLS) != segment_fingerprint([*TOOLS, {"name": "write"}])


def test_none_and_empty_are_distinguishable_but_stable() -> None:
    assert segment_fingerprint(None) == segment_fingerprint(None)
    assert segment_fingerprint(None) != segment_fingerprint([])


def test_unserializable_value_does_not_raise() -> None:
    """A stable repr is enough; attribution must never break the request path."""

    class Weird:
        pass

    assert segment_fingerprint({"t": Weird()})  # no exception
