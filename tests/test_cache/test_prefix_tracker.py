"""Tests for PrefixCacheTracker — cache-aware compression."""

import copy
import time

import pytest

from headroom.cache.prefix_tracker import (
    MISS_COLD_START,
    MISS_PREFIX_CHANGE,
    MISS_TTL_EXPIRY,
    MISS_UNKNOWN,
    FreezeStats,
    PrefixCacheTracker,
    PrefixFreezeConfig,
    SessionTrackerStore,
    overlay_cached_prefix,
)


class TestPrefixCacheTracker:
    """Test PrefixCacheTracker core functionality."""

    @pytest.fixture
    def tracker(self):
        return PrefixCacheTracker("anthropic")

    @pytest.fixture
    def openai_tracker(self):
        return PrefixCacheTracker("openai")

    def test_turn_0_no_freeze(self, tracker):
        """First turn should never freeze — no cache state yet."""
        assert tracker.get_frozen_message_count() == 0

    def test_turn_1_with_cache_hit_freezes(self, tracker):
        """After turn 1 with cache hits, turn 2 should freeze."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant." * 100},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        # Simulate: provider cached 2000 tokens (system + user)
        token_counts = [1500, 50, 500]

        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=2050,
            messages=messages,
            message_token_counts=token_counts,
        )

        # On turn 2, the first 2 messages (1500 + 50 = 1550 <= 2050) are frozen
        assert tracker.get_frozen_message_count() == 3  # All 3 fit within 2050

    def test_partial_freeze(self, tracker):
        """Only messages that fit within cached tokens are frozen."""
        messages = [
            {"role": "system", "content": "System prompt" * 50},
            {"role": "user", "content": "First question" * 50},
            {"role": "assistant", "content": "First answer" * 50},
            {"role": "user", "content": "Second question"},
        ]
        token_counts = [2000, 500, 500, 50]

        tracker.update_from_response(
            cache_read_tokens=2500,
            cache_write_tokens=0,
            messages=messages,
            message_token_counts=token_counts,
        )

        # 2000 + 500 = 2500 <= 2500, but 2000 + 500 + 500 = 3000 > 2500
        assert tracker.get_frozen_message_count() == 2

    def test_cold_start_no_freeze(self, tracker):
        """If cache_read=0 and cache_write=0, don't freeze."""
        messages = [{"role": "user", "content": "Hello"}]

        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=0,
            messages=messages,
        )

        assert tracker.get_frozen_message_count() == 0

    def test_cache_write_freezes_next_turn(self, tracker):
        """Cache writes (new cache entries) should be frozen on the next turn."""
        messages = [
            {"role": "system", "content": "System" * 200},
            {"role": "user", "content": "Hello"},
        ]
        token_counts = [1500, 50]

        # Turn 1: provider writes to cache (above min threshold)
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=1550,
            messages=messages,
            message_token_counts=token_counts,
        )

        # Turn 2: should freeze what was written
        assert tracker.get_frozen_message_count() == 2

    def test_min_cached_tokens_threshold(self):
        """Below min_cached_tokens, no freeze."""
        config = PrefixFreezeConfig(min_cached_tokens=2000)
        tracker = PrefixCacheTracker("anthropic", config)

        messages = [{"role": "user", "content": "Hello"}]

        # Turn 1: only 500 tokens cached — below threshold
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=500,
            messages=messages,
            message_token_counts=[500],
        )

        assert tracker.get_frozen_message_count() == 0

    def test_disabled_config(self):
        """Disabled config always returns 0."""
        config = PrefixFreezeConfig(enabled=False)
        tracker = PrefixCacheTracker("anthropic", config)

        messages = [{"role": "system", "content": "System" * 500}]

        tracker.update_from_response(
            cache_read_tokens=5000,
            cache_write_tokens=0,
            messages=messages,
            message_token_counts=[5000],
        )

        assert tracker.get_frozen_message_count() == 0

    def test_turn_number_increments(self, tracker):
        """Turn number should increment on each update."""
        messages = [{"role": "user", "content": "Hello"}]

        assert tracker._turn_number == 0

        tracker.update_from_response(0, 0, messages)
        assert tracker._turn_number == 1

        tracker.update_from_response(0, 0, messages)
        assert tracker._turn_number == 2

    def test_stats_tracking(self, tracker):
        """Stats should reflect tracker state."""
        stats = tracker.stats
        assert isinstance(stats, FreezeStats)
        assert stats.busts_avoided == 0
        assert stats.tokens_preserved == 0
        assert stats.turn_number == 0

    def test_record_bust_avoided(self, tracker):
        """Recording bust avoided should update stats."""
        tracker.record_bust_avoided(tokens_preserved=5000, compression_foregone=500)
        tracker.record_bust_avoided(tokens_preserved=3000, compression_foregone=200)

        stats = tracker.stats
        assert stats.busts_avoided == 2
        assert stats.tokens_preserved == 8000
        assert stats.compression_foregone_tokens == 700
        assert stats.net_benefit_tokens == 7300

    def test_estimate_message_tokens(self):
        """Token estimation should roughly match character / 3.5."""
        messages = [
            {"role": "system", "content": "A" * 350},  # ~100 tokens
            {"role": "user", "content": "B" * 70},  # ~20 tokens
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 2
        assert counts[0] > counts[1]  # System should have more tokens

    def test_estimate_content_blocks(self):
        """Token estimation should handle Anthropic content blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A" * 350},
                    {"type": "text", "text": "B" * 350},
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        assert counts[0] > 100

    def test_estimate_tool_result_content(self):
        """Token estimation should count tool_result content field."""
        tool_content = "x" * 3500  # ~1000 tokens
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": tool_content,
                    }
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        # Should be ~1000 tokens, definitely > 100
        assert counts[0] > 100

    def test_estimate_tool_use_input(self):
        """Token estimation should count tool_use input field."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "/very/long/path/" + "x" * 700},
                    }
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        # Should count the serialized input dict
        assert counts[0] > 50

    def test_estimate_tool_result_nested_blocks(self):
        """Token estimation should handle nested content blocks in tool_result."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "A" * 3500},
                        ],
                    }
                ],
            },
        ]
        counts = PrefixCacheTracker._estimate_message_tokens(messages)
        assert len(counts) == 1
        assert counts[0] > 100

    def test_session_ttl_expiry(self):
        """Tracker should report as expired after TTL."""
        config = PrefixFreezeConfig(session_ttl_seconds=1)
        tracker = PrefixCacheTracker("anthropic", config)

        assert tracker.is_expired is False

        # Simulate time passing
        tracker._last_activity = time.time() - 2
        assert tracker.is_expired is True


class TestSessionTrackerStore:
    """Test SessionTrackerStore management."""

    @pytest.fixture
    def store(self):
        return SessionTrackerStore()

    def test_get_or_create_new(self, store):
        """Should create a new tracker for unknown session."""
        tracker = store.get_or_create("session-1", "anthropic")
        assert isinstance(tracker, PrefixCacheTracker)
        assert tracker.provider == "anthropic"

    def test_get_or_create_existing(self, store):
        """Should return the same tracker for the same session."""
        tracker1 = store.get_or_create("session-1", "anthropic")
        tracker2 = store.get_or_create("session-1", "anthropic")
        assert tracker1 is tracker2

    def test_different_sessions(self, store):
        """Different sessions should get different trackers."""
        tracker1 = store.get_or_create("session-1", "anthropic")
        tracker2 = store.get_or_create("session-2", "openai")
        assert tracker1 is not tracker2
        assert tracker1.provider == "anthropic"
        assert tracker2.provider == "openai"

    def test_active_sessions_count(self, store):
        """Should track the number of active sessions."""
        assert store.active_sessions == 0

        store.get_or_create("s1", "anthropic")
        assert store.active_sessions == 1

        store.get_or_create("s2", "openai")
        assert store.active_sessions == 2

    def test_cleanup_expired(self, store):
        """Should remove expired sessions on cleanup."""
        config = PrefixFreezeConfig(session_ttl_seconds=1)
        store = SessionTrackerStore(default_config=config)

        tracker = store.get_or_create("expired-session", "anthropic")
        tracker._last_activity = time.time() - 2

        # Force cleanup
        store._last_cleanup = 0
        store._maybe_cleanup()

        assert store.active_sessions == 0

    def test_compute_session_id_from_header(self, store):
        """Should use x-headroom-session-id header if present."""

        class MockRequest:
            headers = {"x-headroom-session-id": "explicit-id-123"}

        session_id = store.compute_session_id(
            MockRequest(), "claude-3", [{"role": "user", "content": "Hi"}]
        )
        assert session_id == "explicit-id-123"

    def test_compute_session_id_from_hash(self, store):
        """Should hash model + system prompt as fallback."""

        class MockRequest:
            headers = {}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        id1 = store.compute_session_id(MockRequest(), "claude-3", messages)
        id2 = store.compute_session_id(MockRequest(), "claude-3", messages)
        assert id1 == id2  # Stable hash
        assert len(id1) == 16

        # Different model = different session
        id3 = store.compute_session_id(MockRequest(), "gpt-4", messages)
        assert id3 != id1

    def test_compute_session_id_includes_anthropic_top_level_system(self, store):
        class MockRequest:
            headers = {}

        messages = [{"role": "user", "content": "same user turn"}]
        id_a = store.compute_session_id(
            MockRequest(),
            "claude-sonnet-5",
            messages,
            system=[{"type": "text", "text": "session A"}],
        )
        id_b = store.compute_session_id(
            MockRequest(),
            "claude-sonnet-5",
            messages,
            system=[{"type": "text", "text": "session B"}],
        )

        assert id_a != id_b

    def test_compute_session_id_distinguishes_leading_system_run(self, store):
        """Different dynamic LEADING system messages should not collide."""

        class MockRequest:
            headers = {}

        static_prompt = "framework prompt " * 80
        conv_a = [
            {"role": "system", "content": [{"type": "text", "text": static_prompt}]},
            {"role": "system", "content": [{"type": "text", "text": "context: session A"}]},
            {"role": "user", "content": "hello"},
        ]
        conv_b = [
            {"role": "system", "content": [{"type": "text", "text": static_prompt}]},
            {"role": "system", "content": [{"type": "text", "text": "context: session B"}]},
            {"role": "user", "content": "hello"},
        ]

        id_a = store.compute_session_id(MockRequest(), "claude-3", conv_a)
        id_b = store.compute_session_id(MockRequest(), "claude-3", conv_b)

        assert id_a != id_b

    def test_compute_session_id_distinguishes_top_level_system(self, store):
        """Anthropic carries the system prompt as a top-level field (not a
        role:'system' message). The handler folds it in as a synthetic system
        message so two conversations with the same model and turns but different
        system prompts get distinct ids — otherwise they share one tracker and
        their sticky state cross-contaminates. This exercises that mechanism."""

        class MockRequest:
            headers = {}

        turns = [{"role": "user", "content": "hello"}]

        def with_system(system):
            # Mirror what handlers/anthropic.py does for the top-level system.
            return [{"role": "system", "content": system}, *turns]

        id_a = store.compute_session_id(
            MockRequest(), "claude-3", with_system("You are a Python expert.")
        )
        id_b = store.compute_session_id(
            MockRequest(), "claude-3", with_system("You are a Rust expert.")
        )
        assert id_a != id_b

        # A list-of-text-blocks system folds the same text as the string form.
        id_a_list = store.compute_session_id(
            MockRequest(),
            "claude-3",
            with_system([{"type": "text", "text": "You are a Python expert."}]),
        )
        assert id_a_list == id_a

    def test_compute_session_id_is_stable_when_only_non_system_turns_change(self, store):
        """Appending non-system turns should keep the same fallback session id."""

        class MockRequest:
            headers = {}

        base_messages = [
            {"role": "system", "content": [{"type": "text", "text": "framework prompt"}]},
            {"role": "system", "content": [{"type": "text", "text": "context: session A"}]},
            {"role": "user", "content": "hello"},
        ]
        extended_messages = base_messages + [{"role": "assistant", "content": "hi there"}]

        id1 = store.compute_session_id(MockRequest(), "claude-3", base_messages)
        id2 = store.compute_session_id(MockRequest(), "claude-3", extended_messages)

        assert id1 == id2

    def test_compute_session_id_no_system(self, store):
        """Should work without system messages."""

        class MockRequest:
            headers = {}

        messages = [{"role": "user", "content": "Hi"}]
        session_id = store.compute_session_id(MockRequest(), "claude-3", messages)
        assert isinstance(session_id, str)
        assert len(session_id) == 16

    def test_mid_conversation_system_turns_do_not_rotate_session_id(self, store):
        """Claude Code sends <system-reminder> turns as role:"system" MESSAGES
        interleaved into the history (hook outputs, skills lists, truncation
        notices). Hashing those into the fallback id rotates the session id
        mid-conversation — orphaning the prefix tracker and every other
        session-sticky subsystem (beta headers, CCR/memory registries, the
        compression cache) each time a reminder lands. Only the LEADING run of
        system messages is session identity; later system turns are content.
        """

        class MockRequest:
            headers = {}

        leading = {"role": "system", "content": "You are an agent. " * 40}
        turn1 = [leading, {"role": "user", "content": "read file A"}]
        turn2 = turn1 + [
            {"role": "assistant", "content": "read it"},
            {"role": "user", "content": "tool result ..."},
            {
                "role": "system",
                "content": "<system-reminder>Truncated: PARTIAL view</system-reminder>",
            },
            {"role": "user", "content": "continue"},
        ]

        id1 = store.compute_session_id(MockRequest(), "claude-sonnet-5", turn1)
        id2 = store.compute_session_id(MockRequest(), "claude-sonnet-5", turn2)
        assert id1 == id2


class TestConversationLineageResolution:
    """resolve_tracker: one PrefixCacheTracker per conversation lineage (#2085).

    Concurrent conversations that share a fallback session id (same model +
    same system prompt — e.g. a Claude Code session and its parallel subagents)
    must not thrash one tracker's frozen-prefix state: interleaved turns would
    each see the *other* conversation's prefix, freeze never stabilizes, and
    the provider prompt cache is re-written on every call.

    resolve_tracker keys trackers by message lineage instead: an incoming
    history that extends a known lineage reuses its tracker; a diverging or
    rewritten history gets a fresh one. The session id itself is never changed,
    so session-sticky state keyed on it elsewhere (beta headers, CCR/memory
    registries, the compression cache) is unaffected.
    """

    @pytest.fixture
    def store(self):
        return SessionTrackerStore()

    @staticmethod
    def _history(name: str, turn: int) -> list[dict]:
        """Client-shaped request messages for `turn` (1-based): u0,a0,...,u_{turn-1}."""
        messages: list[dict] = []
        for t in range(turn):
            messages.append({"role": "user", "content": f"[{name}] user {t} " + "x" * 200})
            if t < turn - 1:
                messages.append(
                    {"role": "assistant", "content": f"[{name}] assistant {t} " + "y" * 200}
                )
        return messages

    @staticmethod
    def _block_history(turn: int, cc_on: int | None) -> list[dict]:
        """Block-content history; cache_control breakpoint on message `cc_on` (or none)."""
        messages: list[dict] = []
        for t in range(turn):
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": f"user {t} " + "x" * 200}]}
            )
            if t < turn - 1:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"assistant {t} " + "y" * 200}],
                    }
                )
        if cc_on is not None:
            msg = messages[cc_on]
            blocks = [dict(b) for b in msg["content"]]
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            messages[cc_on] = {**msg, "content": blocks}
        return messages

    def test_interleaved_conversations_resolve_to_independent_trackers(self, store):
        """The #2085 production shape: two conversations, one session id,
        alternating requests. Each must keep its own tracker and per-turn state
        (on a shared tracker, _turn_number would count both conversations)."""
        sid = "shared-fallback-id"
        trackers: dict[str, PrefixCacheTracker] = {}
        for turn in range(1, 5):
            for name in ("A", "B"):
                history = self._history(name, turn)
                tracker = store.resolve_tracker(sid, "anthropic", messages=history)
                trackers.setdefault(name, tracker)
                assert tracker is trackers[name], f"[{name}] turn {turn} switched trackers"
                tracker.update_from_response(
                    cache_read_tokens=1000 * turn,
                    cache_write_tokens=500,
                    messages=history,
                )
        assert trackers["A"] is not trackers["B"]
        assert trackers["A"]._turn_number == 4
        assert trackers["B"]._turn_number == 4

    def test_identical_first_turns_share_until_divergence_then_split(self, store):
        """Templated fan-outs send byte-identical first turns. While histories
        are identical, sharing a tracker is harmless (the provider cache line
        is identical too); they must split as soon as the histories diverge."""
        sid = "shared"
        first = [{"role": "user", "content": "verify the fix " + "p" * 300}]
        t_a1 = store.resolve_tracker(sid, "anthropic", messages=first)
        t_b1 = store.resolve_tracker(sid, "anthropic", messages=first)
        assert t_b1 is t_a1

        a2 = first + [
            {"role": "assistant", "content": "answer A"},
            {"role": "user", "content": "next A"},
        ]
        b2 = first + [
            {"role": "assistant", "content": "answer B"},
            {"role": "user", "content": "next B"},
        ]
        t_a2 = store.resolve_tracker(sid, "anthropic", messages=a2)
        t_b2 = store.resolve_tracker(sid, "anthropic", messages=b2)
        assert t_a2 is not t_b2

        a3 = a2 + [
            {"role": "assistant", "content": "answer A2"},
            {"role": "user", "content": "next A2"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=a3) is t_a2

    @pytest.mark.parametrize(
        "cc_turn2",
        [0, -1, None],
        ids=["breakpoint-stays", "breakpoint-moved-to-last", "breakpoint-removed"],
    )
    def test_cache_control_movement_does_not_split_lineage(self, store, cc_turn2):
        """Clients move the cache_control breakpoint every turn; that must not
        read as a rewritten history."""
        sid = "shared"
        t1 = store.resolve_tracker(sid, "anthropic", messages=self._block_history(1, cc_on=0))
        t2 = store.resolve_tracker(
            sid, "anthropic", messages=self._block_history(2, cc_on=cc_turn2)
        )
        assert t2 is t1

    def test_same_message_block_append_reuses_lineage(self, store):
        sid = "shared"
        first = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "stable-1"},
                    {"type": "text", "text": "stable-2"},
                ],
            },
            {"role": "assistant", "content": "ack"},
        ]
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        grown = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "stable-1"},
                    {"type": "text", "text": "stable-2"},
                    {"type": "text", "text": "appended"},
                ],
            },
            {"role": "assistant", "content": "ack"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=grown) is tracker

    def test_same_message_block_edit_starts_new_lineage(self, store):
        sid = "shared"
        first = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "stable-1"},
                    {"type": "text", "text": "stable-2"},
                ],
            },
            {"role": "assistant", "content": "ack"},
        ]
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        edited = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "edited"},
                    {"type": "text", "text": "stable-2"},
                    {"type": "text", "text": "appended"},
                ],
            },
            {"role": "assistant", "content": "ack"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=edited) is not tracker

    @pytest.mark.parametrize(
        ("label", "blocks"),
        [
            ("removed", ["stable-1"]),
            ("reordered", ["stable-2", "stable-1"]),
            ("inserted", ["stable-1", "inserted", "stable-2"]),
            ("replaced", ["stable-1", "other", "appended"]),
        ],
    )
    def test_non_append_block_rewrites_start_new_lineage(self, store, label, blocks):
        from headroom.cache.prefix_tracker import classify_append_only_prefix

        sid = f"shared-{label}"
        first = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "stable-1"},
                    {"type": "text", "text": "stable-2"},
                ],
            },
            {"role": "assistant", "content": "ack"},
        ]
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        rewritten = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text} for text in blocks],
            },
            {"role": "assistant", "content": "ack"},
        ]
        assert classify_append_only_prefix(rewritten, first) is None
        assert store.resolve_tracker(sid, "anthropic", messages=rewritten) is not tracker

    def test_block_append_with_changed_message_field_is_not_append_only(self, store):
        """Growing blocks while any non-content field moves is a replacement."""
        from headroom.cache.prefix_tracker import classify_append_only_prefix

        first = [
            {
                "role": "user",
                "name": "alice",
                "content": [{"type": "text", "text": "stable-1"}],
            },
        ]
        renamed = [
            {
                "role": "user",
                "name": "bob",
                "content": [
                    {"type": "text", "text": "stable-1"},
                    {"type": "text", "text": "appended"},
                ],
            },
        ]
        assert classify_append_only_prefix(renamed, first) is None

    def test_append_only_classifier_rejects_short_empty_and_non_dict_histories(self):
        from headroom.cache.prefix_tracker import (
            _classify_append_only_canonical,
            classify_append_only_prefix,
        )

        history = [{"role": "user", "content": "hello"}]

        assert classify_append_only_prefix([], history) is None
        assert classify_append_only_prefix(history, []) is None
        assert classify_append_only_prefix(history, ["not a message"]) is None
        assert _classify_append_only_canonical([], history) is None

    def test_second_changed_message_rejects_the_append_only_classification(self):
        from headroom.cache.prefix_tracker import classify_append_only_prefix

        previous = [
            {"role": "user", "content": [{"type": "text", "text": "A"}]},
            {"role": "assistant", "content": "old"},
        ]
        current = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B"},
                ],
            },
            {"role": "assistant", "content": "new"},
        ]

        assert classify_append_only_prefix(current, previous) is None

    def test_dropped_current_prefix_message_refuses_raw_indexing(self):
        from headroom.cache.prefix_tracker import classify_append_only_prefix

        previous = [{"role": "user", "content": [{"type": "text", "text": "A"}]}]
        current = [{"cachePoint": {"type": "default"}}]

        assert classify_append_only_prefix(current, previous) is None

    def test_overlay_falls_back_when_frontier_is_outside_optimized_output(self):
        from headroom.cache.prefix_tracker import overlay_cached_prefix

        previous_original = [
            {"role": "user", "content": [{"type": "text", "text": "A"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "old"}]},
        ]
        previous_forwarded = [
            {"role": "user", "content": [{"type": "text", "text": "A_compressed"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "old_compressed"}]},
        ]
        current = [
            previous_original[0],
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "old"},
                    {"type": "text", "text": "appended"},
                ],
            },
        ]

        assert overlay_cached_prefix(
            [previous_forwarded[0]], current, previous_original, previous_forwarded
        ) == [previous_forwarded[0]]

    def test_overlay_falls_back_when_forwarded_message_count_differs(self):
        from headroom.cache.prefix_tracker import overlay_cached_prefix

        original = [{"role": "user", "content": [{"type": "text", "text": "A"}]}]
        forwarded = [
            original[0],
            {"role": "assistant", "content": [{"type": "text", "text": "extra"}]},
        ]

        assert overlay_cached_prefix(original, original, original, forwarded) == original

    def test_overlay_logs_and_replays_prefix_before_a_divergent_tail(self):
        from headroom.cache.prefix_tracker import overlay_cached_prefix

        previous_original = [
            {"role": "user", "content": [{"type": "text", "text": "A" * 64}]},
            {"role": "assistant", "content": "old original text"},
        ]
        previous_forwarded = [
            {"role": "user", "content": [{"type": "text", "text": "Ac"}]},
            {"role": "assistant", "content": "old_c"},
        ]
        current = [
            previous_original[0],
            {"role": "assistant", "content": "new"},
        ]

        assert overlay_cached_prefix(current, current, previous_original, previous_forwarded) == [
            previous_forwarded[0],
            current[1],
        ]

    @pytest.mark.parametrize(
        "requote",
        [
            lambda m: {**m, "content": [{"type": "text", "text": m["content"]}]},
            lambda m: {
                **m,
                "content": [{"type": "text", "text": m["content"], "index": 0}],
            },
            lambda m: {
                **m,
                "content": [
                    {"type": "text", "text": m["content"]},
                    {"cachePoint": {"type": "default"}},
                ],
            },
        ],
        ids=["string-to-block-sugar", "streaming-index-annotation", "bedrock-cachepoint-block"],
    )
    def test_representation_churn_does_not_split_lineage(self, store, requote):
        """Clients re-encode history turn-to-turn without changing content
        (litellm flips string<->block sugar, streaming assembly adds `index`,
        Bedrock moves its cachePoint block). Lineage matching must use the
        same canonical equivalence as the cache-stable delta path."""
        sid = "shared"
        first = {"role": "user", "content": "hello " + "x" * 200}
        t1 = store.resolve_tracker(sid, "anthropic", messages=[first])
        grown = [
            requote(first),
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "next"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=grown) is t1

    @pytest.mark.parametrize(
        "rewrite",
        [
            lambda h: [{"role": "user", "content": "[summary of the conversation so far]"}],
            lambda h: [h[0], {"role": "assistant", "content": "EDITED"}, *h[2:]],
            lambda h: h[:-2],
        ],
        ids=["compacted", "middle-edited", "truncated"],
    )
    def test_rewritten_history_gets_fresh_tracker(self, store, rewrite):
        """A rewritten history (client-side /compact, edits, truncation) means
        the provider cache line is gone anyway: start a fresh lineage, keep the
        old tracker until TTL, and never touch the session id."""
        sid = "shared"
        history = self._history("A", 3)
        tracker = store.resolve_tracker(sid, "anthropic", messages=history)
        fresh = store.resolve_tracker(sid, "anthropic", messages=rewrite(history))
        assert fresh is not tracker
        assert store.active_sessions == 2

    @staticmethod
    def _edit_late(history: list[dict], index: int) -> list[dict]:
        """History with one old message rewritten, as a client stripping an
        ephemeral <system-reminder> block out of its own transcript does."""
        edited = [dict(m) for m in history]
        edited[index] = {**edited[index], "content": "trimmed by the client"}
        return edited

    def test_late_edit_keeps_the_lineage(self, store):
        """One rewritten message near the tail must not discard the agreeing
        head. Strict prefix matching threw away the whole lineage, so prev_fwd
        went to 0 and overlay_cached_prefix had nothing to replay. Measured on
        the replay corpus: histories of 20+ messages broke lineage on 18.1
        percent of turns at mean mismatch depth 167."""
        sid = "shared"
        history = self._history("A", 40)
        tracker = store.resolve_tracker(sid, "anthropic", messages=history)
        late = self._edit_late(history, len(history) - 4)
        assert store.resolve_tracker(sid, "anthropic", messages=late) is tracker
        assert store.active_sessions == 1

    def test_early_divergence_still_gets_a_fresh_tracker(self, store):
        """The #2085 guarantee. Two conversations under one session id share a
        short head and then diverge for good: they must stay apart."""
        sid = "shared"
        history = self._history("A", 40)
        tracker = store.resolve_tracker(sid, "anthropic", messages=history)
        branched = self._edit_late(history, 4)
        assert store.resolve_tracker(sid, "anthropic", messages=branched) is not tracker
        assert store.active_sessions == 2

    def test_short_overlap_is_rejected_even_at_a_high_fraction(self):
        """The absolute floor is the primary defence against merging siblings,
        because a fraction alone cannot see scale. Two short conversations that
        share their opening score a high fraction on a tiny chain, which is
        exactly the #2085 shape: same system prompt, same first exchange, then
        independent work. Nine shared messages out of ten is 0.9, well over the
        fraction guard, so only the message floor keeps them apart."""
        store = SessionTrackerStore(
            default_config=PrefixFreezeConfig(
                lineage_rematch_min_messages=16,
                lineage_rematch_fraction=0.5,
            )
        )
        sid = "shared"
        history = self._history("A", 5)
        tracker = store.resolve_tracker(sid, "anthropic", messages=history)
        sibling = self._edit_late(history, len(history) - 1)
        assert store.resolve_tracker(sid, "anthropic", messages=sibling) is not tracker
        assert store.active_sessions == 2

    def test_strict_prefix_is_preferred_over_a_partial_match(self, store):
        """A chain that genuinely prefixes the history wins outright, so the
        fallback can never pull a conversation onto a worse-matching lineage."""
        sid = "shared"
        exact = store.resolve_tracker(sid, "anthropic", messages=self._history("A", 40))
        store.resolve_tracker(sid, "anthropic", messages=self._edit_late(self._history("A", 40), 4))
        grown = self._history("A", 41)
        assert store.resolve_tracker(sid, "anthropic", messages=grown) is exact

    def test_retained_lineage_replays_the_forwarded_head_byte_identical(self, store):
        """The payoff, end to end. Retaining the tracker is only worth anything
        because overlay_cached_prefix can then replay what was forwarded last
        turn. This asserts the bytes, not just the tracker identity: every
        message before the client's edit must come back exactly as forwarded,
        and the edited message onward must be this turn's fresh output."""
        sid = "shared"
        history = self._history("A", 40)
        store.resolve_tracker(sid, "anthropic", messages=history)

        # What the proxy forwarded last turn: the head was compressed, so it is
        # deliberately NOT equal to the client's original.
        forwarded = [{**m, "content": f"compressed-{i}"} for i, m in enumerate(history)]

        edit_at = len(history) - 4
        late = self._edit_late(history, edit_at)
        assert store.resolve_tracker(sid, "anthropic", messages=late) is not None

        # This turn's pipeline output, recompressed from scratch and drifted.
        fresh = [{**m, "content": f"recompressed-{i}"} for i, m in enumerate(late)]

        result = overlay_cached_prefix(
            fresh,
            late,
            previous_original_messages=history,
            previous_forwarded_messages=forwarded,
        )

        assert result[:edit_at] == forwarded[:edit_at]
        assert result[edit_at:] == fresh[edit_at:]

    def test_fraction_above_one_restores_strict_matching(self):
        """Opt-out: the fallback is off when no agreement can satisfy it."""
        store = SessionTrackerStore(PrefixFreezeConfig(lineage_rematch_fraction=1.5))
        sid = "shared"
        history = self._history("A", 40)
        tracker = store.resolve_tracker(sid, "anthropic", messages=history)
        late = self._edit_late(history, len(history) - 4)
        assert store.resolve_tracker(sid, "anthropic", messages=late) is not tracker

    @pytest.mark.parametrize("messages", [None, []], ids=["none", "empty"])
    def test_resolve_without_messages_matches_legacy_get_or_create(self, store, messages):
        tracker = store.resolve_tracker("sid", "anthropic", messages=messages)
        assert tracker is store.get_or_create("sid", "anthropic")

    def test_resolve_with_freeze_disabled_matches_legacy_get_or_create(self):
        """With prefix freeze off there is no frozen state to protect — skip
        lineage bookkeeping entirely."""
        store = SessionTrackerStore(PrefixFreezeConfig(enabled=False))
        t_a = store.resolve_tracker("sid", "anthropic", messages=self._history("A", 1))
        t_b = store.resolve_tracker("sid", "anthropic", messages=self._history("B", 1))
        assert t_a is t_b
        assert t_a is store.get_or_create("sid", "anthropic")

    def test_over_cap_conversations_share_one_overflow_tracker(self):
        """A fan-out storm on one session id must not grow trackers unbounded:
        past the cap, new conversations share one overflow tracker instead of
        evicting an established lineage."""
        store = SessionTrackerStore(PrefixFreezeConfig(max_lineages_per_session=4))
        sid = "storm"
        overflow = set()
        for i in range(9):
            tracker = store.resolve_tracker(
                sid, "anthropic", messages=[{"role": "user", "content": f"task {i} " + "z" * 100}]
            )
            if i >= 4:
                overflow.add(id(tracker))
        assert store.active_sessions == 5  # 4 lineages + 1 shared overflow
        assert len(overflow) == 1

    def test_established_lineages_survive_cap_overflow(self):
        """Filling the family must never evict an established conversation —
        under round-robin any eviction victim is the next requester, which
        would degrade EVERY conversation to a cold tracker per turn."""
        store = SessionTrackerStore(PrefixFreezeConfig(max_lineages_per_session=2))
        sid = "shared"
        parent_history = self._history("parent", 4)
        parent = store.resolve_tracker(sid, "anthropic", messages=parent_history)
        shorty = store.resolve_tracker(sid, "anthropic", messages=self._history("shorty", 1))
        # Third divergent conversation lands on the overflow tracker; both
        # established lineages keep their trackers.
        newcomer = store.resolve_tracker(sid, "anthropic", messages=self._history("new", 1))
        assert newcomer is not parent
        assert newcomer is not shorty
        grown = parent_history + [
            {"role": "assistant", "content": "[parent] assistant 3 " + "y" * 200},
            {"role": "user", "content": "[parent] user 4 " + "x" * 200},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=grown) is parent
        assert (
            store.resolve_tracker(sid, "anthropic", messages=self._history("shorty", 2)) is shorty
        )

    def test_no_cliff_at_cap_plus_one_round_robin(self):
        """cap+1 conversations round-robining turns: the in-cap conversations
        keep their trackers on every round (no eviction churn); only the
        over-cap tail shares the overflow tracker."""
        store = SessionTrackerStore(PrefixFreezeConfig(max_lineages_per_session=3))
        sid = "shared"
        trackers: dict[str, PrefixCacheTracker] = {}
        for turn in range(1, 4):
            for name in ("A", "B", "C", "D"):
                tracker = store.resolve_tracker(
                    sid, "anthropic", messages=self._history(name, turn)
                )
                if turn == 1:
                    trackers[name] = tracker
                elif name != "D":
                    assert tracker is trackers[name], f"[{name}] turn {turn} lost its tracker"
                else:
                    assert tracker is trackers["D"]  # stable overflow tracker

    def test_empty_canonical_history_falls_back_to_legacy(self):
        """A history whose every message projects away (pure directive
        content) carries no lineage signal — behave like get_or_create."""
        store = SessionTrackerStore()
        tracker = store.resolve_tracker("sid", "anthropic", messages=[{}])
        assert tracker is store.get_or_create("sid", "anthropic")

    def test_nan_in_tool_payload_does_not_split_lineage(self):
        """json.loads accepts bare NaN, and NaN != NaN — a resent history
        containing one must still read as the same conversation."""
        store = SessionTrackerStore()
        sid = "shared"
        turn1 = [
            {"role": "user", "content": "run the tool " + "x" * 200},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "score", "input": {"v": float("nan")}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]
        t1 = store.resolve_tracker(sid, "anthropic", messages=turn1)
        turn2 = turn1 + [
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=turn2) is t1

    def test_expired_lineages_are_cleaned_up(self):
        config = PrefixFreezeConfig(session_ttl_seconds=1)
        store = SessionTrackerStore(default_config=config)
        for name in ("A", "B"):
            tracker = store.resolve_tracker("sid", "anthropic", messages=self._history(name, 1))
            tracker._last_activity = time.time() - 2

        store._last_cleanup = 0
        store._maybe_cleanup()
        assert store.active_sessions == 0

        # The lineage index must not resurrect evicted trackers: extending an
        # evicted conversation starts cold.
        fresh = store.resolve_tracker("sid", "anthropic", messages=self._history("A", 2))
        assert fresh._turn_number == 0

    def test_shared_session_id_is_not_rotated(self, store):
        """Composition guard: lineage resolution must not leak into session-id
        derivation — beta stickiness and the compression cache key on it."""

        class MockRequest:
            headers = {}

        msgs_a = [{"role": "system", "content": "S"}, *self._history("A", 1)]
        msgs_b = [{"role": "system", "content": "S"}, *self._history("B", 1)]
        id_a = store.compute_session_id(MockRequest(), "claude-3", msgs_a)
        id_b = store.compute_session_id(MockRequest(), "claude-3", msgs_b)
        assert id_a == id_b


    @staticmethod
    def _history(name: str, turn: int) -> list[dict]:
        """Client-shaped request messages for `turn` (1-based): u0,a0,...,u_{turn-1}."""
        messages: list[dict] = []
        for t in range(turn):
            messages.append({"role": "user", "content": f"[{name}] user {t} " + "x" * 200})
            if t < turn - 1:
                messages.append(
                    {"role": "assistant", "content": f"[{name}] assistant {t} " + "y" * 200}
                )
        return messages

    @staticmethod
    def _block_history(turn: int, cc_on: int | None) -> list[dict]:
        """Block-content history; cache_control breakpoint on message `cc_on` (or none)."""
        messages: list[dict] = []
        for t in range(turn):
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": f"user {t} " + "x" * 200}]}
            )
            if t < turn - 1:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"assistant {t} " + "y" * 200}],
                    }
                )
        if cc_on is not None:
            msg = messages[cc_on]
            blocks = [dict(b) for b in msg["content"]]
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            messages[cc_on] = {**msg, "content": blocks}
        return messages

    @staticmethod
    def _in_place_growth(
        stable: int, churn: int, tail: str, suffix: str = "end of transcript"
    ) -> list[dict]:
        """The sub-call shape: a transcript packed into ONE block-style message
        whose leading blocks are stable, whose middle churns, and which keeps a
        fixed instruction pinned at the very end. This history grows at BLOCK
        granularity, not message granularity.

        The pinned suffix mirrors the captured production shape, where the final
        blocks are byte-identical turn over turn while the blocks before them are
        rewritten. ``suffix`` is what distinguishes sibling conversations that
        pack the same parent transcript.
        """
        blocks = [{"type": "text", "text": f"stable {i} " + "x" * 200} for i in range(stable)]
        blocks += [{"type": "text", "text": f"{tail} churn {i}"} for i in range(churn)]
        blocks += [{"type": "text", "text": suffix}]
        return [{"role": "user", "content": "kickoff"}, {"role": "user", "content": blocks}]

    def test_in_place_message_growth_keeps_its_lineage(self, store):
        """Lineage matching assumes histories append MESSAGES. A conversation
        whose newest message grows in place used to fail that test on every turn
        and get a FRESH tracker each time — no frozen prefix, no overlay replay,
        and one leaked lineage per turn. It must reuse its own tracker instead."""
        sid = "sub-call"
        first = None
        for turn, churn in enumerate([2, 4, 7, 11], start=1):
            history = self._in_place_growth(20, churn, f"t{turn}")
            tracker = store.resolve_tracker(sid, "anthropic", messages=history)
            first = first or tracker
            assert tracker is first, f"turn {turn} switched trackers"
            # The previous turn's state has to actually ARRIVE, not just exist.
            if turn > 1:
                assert tracker.get_last_forwarded_messages(), f"turn {turn} lost prev messages"
            tracker.update_from_response(
                cache_read_tokens=1000,
                cache_write_tokens=500,
                messages=history,
                original_messages=history,
            )
        assert store.active_sessions == 1, "one conversation must not leak lineages"
        assert first._turn_number == 4

    def test_fan_out_diverging_in_its_newest_message_still_splits(self, store):
        """Guard on the loosened match: templated fan-outs share the head and
        differ from the first block of the newest message. They must keep
        separate trackers (that thrash is what lineages exist to prevent)."""
        sid = "shared-fallback-id"
        a = self._in_place_growth(0, 6, "agent-a")
        b = self._in_place_growth(0, 6, "agent-b")
        assert store.resolve_tracker(sid, "anthropic", messages=a) is not store.resolve_tracker(
            sid, "anthropic", messages=b
        )

    def test_mostly_rewritten_newest_message_gets_a_fresh_tracker(self, store):
        """Continuation requires MOST of the recorded message to survive. A
        message rewritten past that point is a new conversation, not growth."""
        sid = "shared"
        tracker = store.resolve_tracker(
            sid, "anthropic", messages=self._in_place_growth(2, 20, "t1")
        )
        rewritten = self._in_place_growth(2, 20, "t2")  # only 2 of 22 blocks survive
        assert store.resolve_tracker(sid, "anthropic", messages=rewritten) is not tracker

    def test_shrinking_newest_message_gets_a_fresh_tracker(self, store):
        """Growth only. A shorter block list means the client dropped content,
        so the provider's cache line for it is gone."""
        sid = "shared"
        tracker = store.resolve_tracker(
            sid, "anthropic", messages=self._in_place_growth(20, 8, "t1")
        )
        shrunk = self._in_place_growth(20, 2, "t1")
        assert store.resolve_tracker(sid, "anthropic", messages=shrunk) is not tracker

    def test_shared_preamble_does_not_merge_parallel_subagents(self, store):
        """Parallel subagents of the same type open with the same injected
        boilerplate and then diverge. Sharing a preamble is not a conversation
        identity — they must keep separate trackers, or they thrash one."""
        sid = "shared-fallback-id"
        preamble = {"type": "text", "text": "<system-reminder>shared boilerplate</system-reminder>"}

        def spawn(name: str) -> list[dict]:
            return [
                {"role": "user", "content": [preamble, {"type": "text", "text": f"task {name}"}]}
            ]

        a = store.resolve_tracker(sid, "anthropic", messages=spawn("a"))
        b = store.resolve_tracker(sid, "anthropic", messages=spawn("b"))
        assert a is not b
        assert store.active_sessions == 2

    def test_boilerplate_on_both_ends_is_not_enough_to_share_a_lineage(self, store):
        """Injected boilerplate can bracket a message on BOTH sides — a preamble
        in front and a reminder pinned at the end. Two conversations then agree
        on their first blocks and their last block while differing in the only
        part that is theirs, so a proportional test alone ("most of the recorded
        blocks survived") would merge them. A few shared blocks is not identity.
        """
        sid = "shared-fallback-id"
        head = [{"type": "text", "text": f"boilerplate {i}"} for i in range(2)]
        pinned = {"type": "text", "text": "<system-reminder>stay on task</system-reminder>"}

        def spawn(name: str) -> list[dict]:
            content = [*head, {"type": "text", "text": f"task {name}"}, pinned]
            return [{"role": "user", "content": content}]

        a = store.resolve_tracker(sid, "anthropic", messages=spawn("a"))
        b = store.resolve_tracker(sid, "anthropic", messages=spawn("b"))
        assert a is not b
        assert store.active_sessions == 2

    def test_sibling_sub_calls_over_one_transcript_do_not_ping_pong(self, store):
        """Two sub-call streams can pack the SAME parent transcript and differ
        only in the instruction appended after it. Their leading blocks match
        almost entirely, so without a tail check they would trade one tracker
        back and forth on every turn — forever, since neither ever gains a
        message to tell them apart."""
        sid = "shared-fallback-id"
        seen: dict[str, PrefixCacheTracker] = {}
        for turn, churn in enumerate([2, 4, 7], start=1):
            for stream in ("summarize", "title"):
                history = self._in_place_growth(
                    30, churn, f"t{turn}", suffix=f"instruction: {stream}"
                )
                tracker = store.resolve_tracker(sid, "anthropic", messages=history)
                seen.setdefault(stream, tracker)
                assert tracker is seen[stream], f"[{stream}] turn {turn} switched trackers"
                tracker.update_from_response(
                    cache_read_tokens=1000,
                    cache_write_tokens=500,
                    messages=history,
                    original_messages=history,
                )
        assert seen["summarize"] is not seen["title"]

    def test_added_message_plus_rewritten_older_one_is_not_in_place_growth(self, store):
        """Growing in place means the message COUNT held still. A history that
        gained a message AND rewrote an earlier one was rewritten, not grown."""
        sid = "shared"
        base = self._in_place_growth(30, 4, "t1")
        tracker = store.resolve_tracker(sid, "anthropic", messages=base)
        grown_and_rewritten = [*self._in_place_growth(30, 9, "t2"), {"role": "assistant", "c": 1}]
        assert store.resolve_tracker(sid, "anthropic", messages=grown_and_rewritten) is not tracker

    def test_strict_prefix_match_wins_over_in_place_growth(self, store):
        """An exact continuation must never be displaced by a looser one."""
        sid = "shared"
        base = self._in_place_growth(20, 4, "t1")
        exact = store.resolve_tracker(sid, "anthropic", messages=base)
        # A second lineage that would ALSO match `base` under in-place growth.
        store.resolve_tracker(sid, "anthropic", messages=self._in_place_growth(20, 2, "t1"))
        assert store.resolve_tracker(sid, "anthropic", messages=base) is exact

class TestMultiTurnScenario:
    """Integration-style tests simulating multi-turn conversations."""

    def test_five_turn_conversation(self):
        """Simulate a 5-turn conversation with growing prefix."""
        tracker = PrefixCacheTracker("anthropic")

        # Turn 1: System + User (cold start, no cache)
        messages_t1 = [
            {"role": "system", "content": "System prompt" * 200},
            {"role": "user", "content": "Question 1"},
        ]
        token_counts_t1 = [2000, 50]

        assert tracker.get_frozen_message_count() == 0  # No freeze on turn 1

        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=2050,
            messages=messages_t1,
            message_token_counts=token_counts_t1,
        )

        # Turn 2: Previous messages cached, new user message added
        messages_t2 = messages_t1 + [
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        token_counts_t2 = [2000, 50, 200, 50]

        frozen = tracker.get_frozen_message_count()
        assert frozen == 2  # System + User1 frozen

        tracker.update_from_response(
            cache_read_tokens=2050,
            cache_write_tokens=250,
            messages=messages_t2,
            message_token_counts=token_counts_t2,
        )

        # Turn 3: Even more cached
        messages_t3 = messages_t2 + [
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
        ]
        token_counts_t3 = [2000, 50, 200, 50, 200, 50]

        frozen = tracker.get_frozen_message_count()
        assert frozen == 4  # System + User1 + Asst1 + User2 frozen

        tracker.update_from_response(
            cache_read_tokens=2300,
            cache_write_tokens=250,
            messages=messages_t3,
            message_token_counts=token_counts_t3,
        )

        # Verify turn count
        assert tracker._turn_number == 3

    def test_cache_bust_resets_freeze(self):
        """If cache is busted (0 read, 0 write), freeze should reset."""
        tracker = PrefixCacheTracker("anthropic")

        messages = [
            {"role": "system", "content": "System" * 200},
            {"role": "user", "content": "Hello"},
        ]

        # Turn 1: Cache established
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=2000,
            messages=messages,
            message_token_counts=[1500, 500],
        )
        assert tracker.get_frozen_message_count() == 2  # Both fit within 2000

        # Turn 2: Cache bust (0 reads, system prompt changed)
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=0,
            messages=messages,
            message_token_counts=[1500, 500],
        )

        # After a bust with 0 total, freeze should reset
        assert tracker.get_frozen_message_count() == 0


class TestClassifyCacheMiss:
    """Cache-miss attribution (#1313): TTL lapse vs prefix change vs unknown."""

    BASE = [
        {"role": "system", "content": "x" * 4000},
        {"role": "user", "content": "hello"},
    ]
    CHANGED = [
        {"role": "system", "content": "DIFFERENT" * 400},
        {"role": "user", "content": "hello"},
    ]

    def _warm(self, tracker, messages, read=500, write=500):
        """Simulate a turn that left `messages` cached."""
        tracker.update_from_response(
            cache_read_tokens=read, cache_write_tokens=write, messages=messages
        )

    def test_cold_start_is_not_a_miss(self):
        """No prior cached prefix → cold start, is_miss False."""
        tracker = PrefixCacheTracker("anthropic")
        result = tracker.classify_cache_miss(0, self.BASE)
        assert result.is_miss is False
        assert result.reason == MISS_COLD_START

    def test_cache_read_is_a_hit(self):
        """A non-zero read on an expected-cached prefix is a hit, not a miss."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(800, self.BASE)
        assert result.is_miss is False
        assert result.reason == "hit"

    def test_ttl_expiry_when_idle_exceeds_ttl(self):
        """Idle longer than the cache TTL → ttl_expiry."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.BASE, idle_seconds=400)
        assert result.is_miss is True
        assert result.reason == MISS_TTL_EXPIRY
        assert result.ttl_exceeded is True
        assert result.cache_ttl_seconds == 300

    def test_ttl_wins_tie_when_prefix_also_changed(self):
        """When idle past TTL AND prefix changed, TTL expiry wins (docstring)."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.CHANGED, idle_seconds=400)
        assert result.reason == MISS_TTL_EXPIRY
        assert result.ttl_exceeded is True
        assert result.prefix_changed is True

    def test_prefix_change_within_ttl(self):
        """Within TTL but the forwarded prefix differs → prefix_change."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.CHANGED, idle_seconds=10)
        assert result.is_miss is True
        assert result.reason == MISS_PREFIX_CHANGE
        assert result.prefix_changed is True
        assert result.ttl_exceeded is False

    def test_unknown_when_stable_prefix_within_ttl(self):
        """Within TTL, prefix unchanged, but still no read → unknown."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        result = tracker.classify_cache_miss(0, self.BASE, idle_seconds=10)
        assert result.is_miss is True
        assert result.reason == MISS_UNKNOWN

    def test_growing_prefix_is_stable(self):
        """A turn that appends to last turn's forwarded prefix is not a change."""
        tracker = PrefixCacheTracker("anthropic")
        self._warm(tracker, self.BASE)
        grown = self.BASE + [{"role": "assistant", "content": "hi back"}]
        result = tracker.classify_cache_miss(0, grown, idle_seconds=10)
        # Prefix preserved (only appended) → not a prefix_change.
        assert result.prefix_changed is False
        assert result.reason == MISS_UNKNOWN

    def test_one_hour_ttl_override(self):
        """cache_ttl_seconds override widens the TTL window (1h breakpoint)."""
        tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(cache_ttl_seconds=3600))
        self._warm(tracker, self.BASE)
        # 400s idle is past the 300s default but within 3600s → not TTL expiry.
        result = tracker.classify_cache_miss(0, self.BASE, idle_seconds=400)
        assert result.cache_ttl_seconds == 3600
        assert result.ttl_exceeded is False
        assert result.reason == MISS_UNKNOWN

    def test_resolved_ttl_falls_back_to_provider_default(self):
        assert PrefixCacheTracker("anthropic").resolved_cache_ttl_seconds() == 300
        assert (
            PrefixCacheTracker(
                "anthropic", PrefixFreezeConfig(cache_ttl_seconds=3600)
            ).resolved_cache_ttl_seconds()
            == 3600
        )


class TestForwardedPrefixWillChange:
    """The gates ask this before spending a rewrite, so it has to be exact.

    A false positive hands the effort router and the compaction advisor a free
    pass they have not earned. A false negative charges them for a write that
    was already being paid. The region that matters is the one the provider
    billed us a read for, not the whole list we sent last turn.
    """

    @staticmethod
    def _msgs(*texts: str) -> list[dict]:
        return [{"role": "user", "content": t} for t in texts]

    @staticmethod
    def _warm(cached_messages: list[dict], read_tokens: int = 4000) -> PrefixCacheTracker:
        """A tracker that has billed a read covering exactly these messages."""
        tracker = PrefixCacheTracker("anthropic")
        tracker.update_from_response(
            cache_read_tokens=read_tokens,
            cache_write_tokens=0,
            messages=cached_messages,
            message_token_counts=[read_tokens // len(cached_messages)] * len(cached_messages),
        )
        return tracker

    def test_a_cold_lineage_is_not_busting(self):
        """Nothing cached yet, so there is no prefix to invalidate."""
        tracker = PrefixCacheTracker("anthropic")
        assert tracker.forwarded_prefix_will_change(self._msgs("a", "b")) is False

    def test_an_appended_turn_keeps_the_prefix(self):
        cached = self._msgs("a", "b")
        tracker = self._warm(cached)
        assert tracker.forwarded_prefix_will_change(cached + self._msgs("c")) is False

    def test_rewriting_the_uncached_tail_is_not_a_bust(self):
        """The regression this method exists to avoid.

        Last turn's uncached tail is this turn's mid-prefix. Recompressing it
        changes the list without touching what the provider cached, and a
        whole-list compare called that a bust on every turn of a healthy
        lineage.
        """
        tracker = self._warm(self._msgs("a", "b"))
        # Two messages were cached. Send those verbatim, everything after
        # rewritten.
        current = self._msgs("a", "b", "tail-recompressed", "d")
        assert tracker.forwarded_prefix_will_change(current) is False

    def test_rewriting_the_head_of_the_cached_prefix_is_a_bust(self):
        cached = self._msgs(*[f"m{i}" for i in range(8)])
        tracker = self._warm(cached)
        current = self._msgs("CHANGED", *[f"m{i}" for i in range(1, 8)])
        assert tracker.forwarded_prefix_will_change(current) is True

    def test_rewriting_the_last_cached_message_is_not_already_paid(self):
        """Providers match a longest common prefix, so depth is the question.

        Recompressing the final cached message leaves everything ahead of it
        readable. Handing a caller a free-rewrite discount for that spends the
        rest of the prefix on a mutation it never covered.
        """
        cached = self._msgs(*[f"m{i}" for i in range(8)])
        tracker = self._warm(cached)
        current = self._msgs(*[f"m{i}" for i in range(7)], "CHANGED")
        assert tracker.forwarded_prefix_will_change(current) is False

    def test_a_shorter_list_than_the_cached_prefix_is_a_bust(self):
        tracker = self._warm(self._msgs("a", "b"))
        assert tracker.forwarded_prefix_will_change(self._msgs("a")) is True

    def test_a_non_list_body_is_not_treated_as_a_bust(self):
        """Best-effort: an unreadable body is no evidence of anything."""
        tracker = self._warm(self._msgs("a", "b"))
        assert tracker.forwarded_prefix_will_change(None) is False

    def test_a_moved_cache_breakpoint_is_not_a_bust(self):
        """Claude Code slides the breakpoint forward on every call.

        The same cached message carries cache_control on one turn and not the
        next. That is a directive about marker placement, not content, and
        treating it as a change reported a bust on every turn of a lineage
        billing 97 to 100% cache hits.
        """
        base = [
            {"role": "user", "content": [{"type": "text", "text": f"m{i}"}]} for i in range(4)
        ]
        turn1 = copy.deepcopy(base)
        turn1[2]["content"][0]["cache_control"] = {"type": "ephemeral"}
        tracker = PrefixCacheTracker("anthropic")
        tracker.update_from_response(
            cache_read_tokens=400,
            cache_write_tokens=0,
            messages=turn1,
            message_token_counts=[100] * 4,
        )

        turn2 = copy.deepcopy(base)
        turn2[3]["content"][0]["cache_control"] = {"type": "ephemeral"}
        assert tracker.forwarded_prefix_will_change(turn2) is False

        # Content still counts, marker or no marker.
        turn3 = copy.deepcopy(turn2)
        turn3[1]["content"][0]["text"] = "rewritten"
        assert tracker.forwarded_prefix_will_change(turn3) is True

    def test_the_projected_assistant_tail_is_excluded(self):
        """Callers record what they sent plus the assistant reply they parsed.

        That reconstruction never byte-matches the client's own echo of the same
        reply, so including it in the compare reports a bust on every single
        turn. Only the region actually sent upstream is evidence of anything.
        """
        sent = self._msgs("a", "b")
        reconstructed = {"role": "assistant", "content": [{"type": "text", "text": "reply"}]}
        tracker = PrefixCacheTracker("anthropic")
        tracker.update_from_response(
            cache_read_tokens=300,
            cache_write_tokens=0,
            messages=sent + [reconstructed],
            message_token_counts=[100] * 3,
            sent_message_count=len(sent),
        )
        # The provider's read covered all three, but only two were ever sent.
        assert tracker._cached_message_count == 3

        # Next turn the client echoes its own, differently-shaped, version.
        client_echo = {
            "role": "assistant",
            "content": [{"type": "text", "text": "reply"}],
            "id": "msg_abc",
        }
        assert (
            tracker.forwarded_prefix_will_change(sent + [client_echo] + self._msgs("c"))
            is False
        )

    def test_rewriting_a_sent_message_is_still_a_bust(self):
        """Excluding the projected tail must not blind it to real changes."""
        sent = self._msgs(*[f"m{i}" for i in range(8)])
        tracker = PrefixCacheTracker("anthropic")
        tracker.update_from_response(
            cache_read_tokens=900,
            cache_write_tokens=0,
            messages=sent + [{"role": "assistant", "content": "reply"}],
            message_token_counts=[100] * 9,
            sent_message_count=len(sent),
        )
        current = self._msgs("CHANGED", *[f"m{i}" for i in range(1, 8)])
        assert tracker.forwarded_prefix_will_change(current) is True

    def test_the_sent_count_survives_a_snapshot_round_trip(self):
        """A restored lineage keeps reading its live prefix.

        The exported message list carries a projected assistant tail. Without
        the sent count travelling alongside it, the restored tracker compares
        against that tail and calls the first turn after a restart a bust.
        """
        sent = self._msgs("a", "b")
        tracker = PrefixCacheTracker("anthropic")
        tracker.update_from_response(
            cache_read_tokens=300,
            cache_write_tokens=0,
            messages=sent + [{"role": "assistant", "content": "reply"}],
            message_token_counts=[100] * 3,
            sent_message_count=len(sent),
        )

        restored = PrefixCacheTracker("anthropic")
        restored.restore_state(tracker.export_state())
        assert restored._last_sent_message_count == len(sent)

        client_echo = {"role": "assistant", "content": [{"type": "text", "text": "reply"}]}
        assert (
            restored.forwarded_prefix_will_change(sent + [client_echo] + self._msgs("c"))
            is False
        )

    def test_an_older_snapshot_without_the_sent_count_still_loads(self):
        blob = PrefixCacheTracker("anthropic").export_state()
        blob.pop("last_sent_message_count")
        restored = PrefixCacheTracker("anthropic")
        restored.restore_state(blob)
        assert restored._last_sent_message_count == 0
