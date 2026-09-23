"""Prefix Cache Tracker — session-scoped state for cache-aware compression.

Tracks provider prefix cache state between turns so the transform pipeline
can freeze already-cached messages and only compress new content.

Problem: Clients like Claude Code already manage prefix caching (up to 4
cache_control breakpoints, growing-prefix strategy). If Headroom compresses
or modifies messages in the cached prefix, it invalidates the cache —
replacing a 90% read discount (Anthropic) or 50% (OpenAI) with a 25%
write penalty.

Solution: After each API response, record how many tokens the provider
cached. On the next turn, freeze that many messages so the transform
pipeline skips them entirely.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import os
import time
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from headroom.proxy.hybrid_mode import HybridModeController

logger = logging.getLogger(__name__)

# Provider cache economics for cost comparisons
_PROVIDER_READ_DISCOUNT = {
    "anthropic": 0.9,  # 90% discount on reads
    "openai": 0.5,  # 50% discount on reads
    "gemini": 0.9,
    "bedrock": 0.9,
}

# How much of a cached prefix has to be going down with this turn's own body
# before a caller may treat its rewrite as already paid for. Providers match a
# longest common prefix, so a turn that recompresses the last cached message
# still reads everything ahead of it. Set strict on purpose: claiming a free
# rewrite that is not free spends real tokens, while missing one only leaves a
# structural mutation waiting for a later turn.
ALREADY_BUSTING_SURVIVING_FRACTION = 0.25

_PROVIDER_WRITE_PENALTY = {
    "anthropic": 0.25,  # 25% surcharge on writes
    "openai": 0.0,  # No write penalty
    "gemini": 0.0,
    "bedrock": 0.25,
}

# A warm turn appends a user message plus one assistant reply and its tool
# results. Measured over 865 warm turns of live traffic that lands around 7k
# tokens, so anything past this ceiling is a re-write of already-cached history
# rather than an append, and must not pollute the steady-state write rate.
_STEADY_WRITE_CEILING = 32_000

# Anthropic bills a 1h cache write at 2x base and a 5m write at 1.25x, so the
# premium for holding the longer tier is 0.75x of whatever gets written.
_TTL_1H_PREMIUM = 0.75
_TTL_5M_WRITE_PENALTY = 1.25

# Below this the prefix is too small for a re-write to outweigh the premium,
# whatever the cadence, so the reactive cadence rule keeps the decision.
_TTL_SIZE_FLOOR_TOKENS = 40_000

# Horizon for pricing an edit whose saving outlives the current cache run.
# Sessions are heavy tailed, so remaining length is estimated Lindy style: one
# that has already run n turns is expected to run about n more. The cap keeps a
# long session from promising an unbounded payback, and the minimum keeps the
# estimator off until a session has shown it is not a one-shot call.
_SESSION_READ_HORIZON_CAP = 200.0
_SESSION_READ_MIN_TURNS = 8

# Smoothing factor for the compression-ratio predictor. The cost gate has to
# estimate how many tokens the NEXT compression will remove before running it,
# which is the same shape as an OS scheduler predicting the next CPU burst from
# past bursts. The textbook answer is exponential averaging: weight the newest
# observation by alpha and decay the running estimate by (1 - alpha), so one
# anomalous turn cannot yank the estimate the way a single last-sample can.
# 0.3 tracks a genuine regime change within a few turns without chasing noise.
_KEPT_EWMA_ALPHA = 0.3

# Default prompt-cache lifetime per provider, in seconds. Used by
# `classify_cache_miss` to decide whether a miss is most likely a TTL
# lapse (idle longer than this) versus a prefix-content change. Anthropic's
# default ephemeral cache is 5 minutes (matches
# headroom.cache.anthropic.ANTHROPIC_CACHE_TTL_SECONDS); the others are best-
# effort defaults and only matter once those providers are wired in. A
# session that opts into Anthropic's 1h cache breakpoint can override this
# via the tracker config (see PrefixFreezeConfig.cache_ttl_seconds).
_PROVIDER_CACHE_TTL_SECONDS = {
    "anthropic": 300,  # 5 minutes (default ephemeral cache)
    "openai": 300,  # automatic prefix cache, ~5-10 min; conservative floor
    "gemini": 300,
    "bedrock": 300,
}


@dataclass
class PrefixFreezeConfig:
    """Configuration for cache-aware prefix freezing."""

    enabled: bool = True
    min_cached_tokens: int = 1024  # Min cached tokens to activate freeze
    session_ttl_seconds: int = 600  # Session tracker cleanup TTL
    force_compress_threshold: float = 0.5  # Bust cache if compression saves > this fraction
    # Provider prompt-cache lifetime used by `classify_cache_miss` to tell a
    # TTL lapse from a prefix change. `None` falls back to the per-provider
    # default in `_PROVIDER_CACHE_TTL_SECONDS`. Set to 3600 for a session that
    # uses Anthropic's 1h cache breakpoint so idle-gap attribution stays honest.
    cache_ttl_seconds: int | None = None
    # Cap on concurrent conversation lineages tracked per session id (#2085).
    # A fan-out storm (many parallel subagents sharing one fallback id) evicts
    # the shortest-chain lineage instead of growing without bound. Raise it
    # for workspaces that genuinely run more concurrent conversations on one
    # model + system prompt.
    max_lineages_per_session: int = 32
    # How much of a recorded lineage chain the incoming history must still
    # match for the tracker to be reused when it is no longer a strict prefix.
    #
    # Client histories are not append-only. Claude Code strips
    # <system-reminder> blocks out of old user messages, so one edit deep in a
    # long history used to fail the strict prefix test, start a fresh lineage,
    # and throw away every message before the edit as well. Measured on the
    # replay corpus: histories of 20+ messages broke lineage on 18.1 percent of
    # turns at a mean mismatch depth of 167, discarding 24.1M tokens of prefix
    # that was still byte-valid.
    #
    # Two guards, tuned together against that corpus. The absolute floor is what
    # keeps sibling conversations apart: a subagent that shares only the system
    # prompt and opening message overlaps by one to three messages, never eight.
    # The fraction keeps client compaction out, which lands near 0.01.
    #
    # min 8 messages with fraction 0.5 recovers 24.3M of the 25.4M (95.8 percent)
    # over 191 of 222 broken lineages. Requiring fraction 0.9 instead recovers
    # only 79.7 percent, because a reminder stripped a dozen messages from the
    # tail of an 80 message history scores 0.85 and would be refused.
    #
    # Set lineage_rematch_min_messages to 0 to restore strict-prefix matching.
    lineage_rematch_min_messages: int = 8
    lineage_rematch_fraction: float = 0.5


@dataclass
class FreezeStats:
    """Statistics from prefix freezing for metrics/dashboard."""

    busts_avoided: int = 0
    tokens_preserved: int = 0
    compression_foregone_tokens: int = 0
    net_benefit_tokens: int = 0  # tokens_preserved - compression_foregone
    frozen_message_count: int = 0
    turn_number: int = 0


# Cache-miss attribution verdicts. `reason` is one of these literals so
# metrics/dashboard can bucket without re-deriving the logic. See
# PrefixCacheTracker.classify_cache_miss.
MISS_TTL_EXPIRY = "ttl_expiry"
MISS_PREFIX_CHANGE = "prefix_change"
MISS_COLD_START = "cold_start"  # no prior cached prefix to miss against
MISS_UNKNOWN = "unknown"  # expected a hit, content stable, idle within TTL

# Which segment of the cache key changed, when we can tell. Reporting the
# earliest changed segment is what makes a prefix_change actionable: a tool-list
# change usually means the tool set moved mid-session (an MCP server connecting
# late, a plugin/skill loading asynchronously), a system-prompt change means the
# prompt was rebuilt (an injected timestamp, a re-synced memory block, a per-turn
# reminder), and a messages change means history was rewritten rather than
# appended to. Each is a different fix.
#
# These are the WELL-KNOWN names, not a closed set: segment naming and ordering
# are per-provider request shapes and belong to the caller, which is why
# classify_cache_miss takes an open ordered mapping. For reference:
#   * Anthropic Messages: top-level `tools` and `system`, then `messages`.
#   * OpenAI Chat Completions: `tools`; the system prompt is messages[0], so it
#     is already covered by the messages comparison (no separate segment).
#   * OpenAI Responses: `tools` and `instructions`, then `input`.
#   * Gemini: `tools` and `systemInstruction`, then `contents`.
SEGMENT_TOOLS = "tools"
SEGMENT_SYSTEM = "system"
SEGMENT_MESSAGES = "messages"  # the fallback: prefix moved, no named segment did


@dataclass
class CacheMissAttribution:
    """Why a turn that expected a prompt-cache hit missed instead.

    Produced by :meth:`PrefixCacheTracker.classify_cache_miss`. ``is_miss``
    is False when the turn actually hit cache (or there was nothing to hit),
    in which case ``reason`` is informational only.
    """

    is_miss: bool
    reason: str  # one of the MISS_* literals
    idle_seconds: float = 0.0
    cache_ttl_seconds: int = 0
    expected_cached_tokens: int = 0
    cache_read_tokens: int = 0
    prefix_changed: bool = False
    ttl_exceeded: bool = False
    # Which cache-key segment diverged, when we can tell: one of the SEGMENT_*
    # literals. ``reason == "prefix_change"`` says the bytes moved; this says
    # WHERE, which is the difference between "unexplained" and "fixable".
    changed_segment: str | None = None


def segment_fingerprint(value: Any) -> str:
    """Stable content hash for one cache-key segment (tools, system, ...).

    Provider-neutral on purpose: it hashes whatever the caller hands it, so each
    handler can pass its own request shape without this module knowing about any
    of them.

    ``cache_control`` is stripped first. Clients move the caching breakpoint from
    turn to turn (Claude Code re-marks the newest block every call), and Headroom
    itself re-places message markers, so the annotation flips on bytes the model
    never sees differently. Including it would report a segment "changed" on
    every turn and drown the real signal.

    ``sort_keys`` makes the hash independent of dict ordering, so a client that
    serializes the same tool schema with keys in a different order does not read
    as a change.
    """
    canonical = json.dumps(
        _strip_cache_control(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,  # never raise on an exotic value; a stable repr is enough
    )
    return hashlib.md5(canonical.encode()).hexdigest()[:16]  # nosec B324


def _strip_cache_control(obj: Any) -> Any:
    """Recursively drop ``cache_control`` for content-only equality checks.

    Clients (notably Claude Code) move the cache_control breakpoint to the newest
    message on every call, so the exact same message carries cache_control on one
    turn and not the next. That per-call annotation must be ignored when deciding
    whether this turn append-only-extends the previous one — otherwise a moved
    marker spuriously fails the check and we skip the byte-identical replay,
    busting the cache."""
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(v) for v in obj]
    return obj


# Keys that carry NO semantic payload for the model — transport / caching-directive
# / telemetry / client-routing annotations that clients attach and vary turn-to-turn.
# Grounded in provider API docs (Anthropic Messages, OpenAI Chat+Responses, Bedrock
# Converse) + client-library field inventories (litellm, Vercel AI SDK, opencode,
# Claude Code, Cline). Dropped from the cross-turn prefix-equality key ONLY.
#
# NOTE ON SAFETY: this projection is a COMPARISON KEY, never a source to rebuild
# forwarded bytes — the cache-stable-delta path always forwards the previously
# forwarded bytes + the raw appended delta. So dropping these can't deprive the
# model. What we must NOT do is drop a *semantic* field (that would mask a real
# divergence and replay a stale prefix), which is why: (1) reasoning SIGNATURES are
# NOT in this set (Anthropic 400s if a thinking block is altered/missing, and a
# present/absent flip is a real divergence we want to detect); (2) tool inputs /
# arguments / json payloads are treated as OPAQUE and compared verbatim (see
# _OPAQUE_PAYLOAD_KEYS) so a user key that happens to be named "index"/"state" is
# never stripped from inside a tool call.
_NON_SEMANTIC_KEYS = frozenset(
    {
        # cache-breakpoint markers (moved to the newest block every turn)
        "cache_control",  # Anthropic (per-block)
        "cachePoint",  # Bedrock (per-block content block)
        # litellm unified-message / tool annotations
        "caller",  # litellm programmatic-tool tag on tool_use
        "provider_specific_fields",
        "reasoning_content",  # litellm display echo (the paired signature is separate)
        "reasoning_items",
        "annotations",  # citation/display metadata
        # OpenAI response echoes that can ride on assistant messages
        "system_fingerprint",
        "service_tier",
        # Vercel AI SDK / opencode part transport
        "providerMetadata",
        "providerOptions",
        "callProviderMetadata",
        "state",
        "providerExecuted",
        "synthetic",
        "ignored",
        # streaming-assembly artifact
        "index",
    }
)

# Values under these keys are opaque semantic payloads (tool-call input, OpenAI
# stringified arguments, Bedrock tool_result json). They are compared VERBATIM — we
# never recurse into them to strip "noise" keys, because arbitrary user data there
# may legitimately contain keys that collide with _NON_SEMANTIC_KEYS (e.g. an
# `input` of {"state": "CA", "index": 3}). Recursing would corrupt the comparison.
_OPAQUE_PAYLOAD_KEYS = frozenset({"input", "arguments", "json"})


def _canonicalize_for_prefix_compare(obj: Any) -> Any:
    """Representation-agnostic canonical form for cross-turn prefix equality.

    Providers accept several *equivalent* encodings for the same message, and real
    clients vary them turn-to-turn; a raw-dict prefix compare then fails spuriously
    and drops cache mode to raw (uncompressed) forwarding. This normalizes ONLY
    representation:
      * drops non-semantic annotation / cache-directive / telemetry keys
        (_NON_SEMANTIC_KEYS) at any message/block level;
      * wraps a bare string ``content`` into ``[{"type": "text", "text": ...}]``
        (Anthropic's string sugar, which litellm flips per turn);
      * leaves tool ``input`` / ``arguments`` / ``json`` payloads verbatim
        (_OPAQUE_PAYLOAD_KEYS) so user data is never corrupted;
      * KEEPS all real content (text, tool name/input, tool_result content, reasoning
        signatures, ids) so two messages canonicalize-equal iff they are semantically
        identical.

    Used ONLY as a comparison key for the cache-stable delta path; the original,
    unmodified messages are always what gets forwarded.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in _NON_SEMANTIC_KEYS:
                continue
            if key in _OPAQUE_PAYLOAD_KEYS:
                out[key] = value  # verbatim — do not recurse into user payloads
            elif key == "content" and isinstance(value, str):
                out[key] = [{"type": "text", "text": value}]
            else:
                out[key] = _canonicalize_for_prefix_compare(value)
        return out
    if isinstance(obj, list):
        canon = [_canonicalize_for_prefix_compare(value) for value in obj]
        # Drop blocks that projected to {} — a pure cache-directive content block
        # (e.g. Bedrock {"cachePoint": {...}}) whose only key was non-semantic. Left
        # in place it would be an empty-dict entry, so a directive block moving
        # position across turns would spuriously fail the length/order compare.
        return [value for value in canon if value != {}]
    return obj


RELATION_EXACT = "exact"
RELATION_MESSAGE_APPEND = "message_append"
RELATION_BLOCK_APPEND = "block_append"
RELATION_BLOCK_REWRITE_TAIL = "block_rewrite_tail"
RELATION_DIVERGED = "diverged"


@dataclass(frozen=True)
class HistoryRelation:
    """How a current message history relates to one recorded last turn.

    ``block_append`` and ``block_rewrite_tail`` are deliberately distinct:
    Anthropic should keep the breakpoint on the newest block for a pure append,
    but anchor it to ``stable_prefix_blocks - 1`` when the previous tail was
    rewritten and therefore can never match a prior cache write (#2671).
    """

    kind: str
    message_index: int | None = None
    stable_prefix_blocks: int = 0
    stable_suffix_blocks: int = 0
    previous_block_count: int = 0
    current_block_count: int = 0


# A rewritten-tail match is intentionally conservative.  The production shape
# behind #2671 has a hundred-plus-block stable prefix and a fixed two-block
# suffix.  Requiring both avoids merging sibling sub-calls which merely share a
# short injected preamble or a single generic reminder at the end.
_MIN_REWRITE_PREFIX_BLOCKS = 8
_MIN_REWRITE_SUFFIX_BLOCKS = 2


def _message_fields_outside_content(message: dict[str, Any]) -> dict[str, Any]:
    """Return message identity fields, excluding the block list itself."""
    return {key: value for key, value in message.items() if key != "content"}


def _stable_trailing_block_run(current: list[Any], previous: list[Any], *, leading_run: int) -> int:
    """Non-overlapping canonical-equal suffix length."""
    limit = min(len(current), len(previous)) - leading_run
    run = 0
    while run < limit and current[-(run + 1)] == previous[-(run + 1)]:
        run += 1
    return run


def _classify_history_canonical(
    current_messages: list[Any], previous_messages: list[Any]
) -> HistoryRelation:
    """Classify two already-canonical, structurally snapshotted histories."""
    if not previous_messages or len(current_messages) < len(previous_messages):
        return HistoryRelation(RELATION_DIVERGED)

    changed: HistoryRelation | None = None
    for index, previous_message in enumerate(previous_messages):
        current_message = current_messages[index]
        if current_message == previous_message:
            continue
        if changed is not None:
            return HistoryRelation(RELATION_DIVERGED)
        if not isinstance(previous_message, dict) or not isinstance(current_message, dict):
            return HistoryRelation(RELATION_DIVERGED)
        if _message_fields_outside_content(previous_message) != _message_fields_outside_content(
            current_message
        ):
            return HistoryRelation(RELATION_DIVERGED)

        previous_blocks = previous_message.get("content")
        current_blocks = current_message.get("content")
        if not isinstance(previous_blocks, list) or not isinstance(current_blocks, list):
            return HistoryRelation(RELATION_DIVERGED)

        previous_count = len(previous_blocks)
        current_count = len(current_blocks)
        leading = _stable_leading_block_run(current_blocks, previous_blocks)

        # Pure block append.  The previous write remains intact and Anthropic's
        # lookback can find it, so the breakpoint must advance to the newest
        # block and cover the newly appended tail.
        if current_count > previous_count and leading == previous_count:
            changed = HistoryRelation(
                RELATION_BLOCK_APPEND,
                message_index=index,
                stable_prefix_blocks=leading,
                previous_block_count=previous_count,
                current_block_count=current_count,
            )
            continue

        # Rewritten-tail growth.  This is narrower than a fuzzy prefix match:
        # message count may not change, content may not shrink, most of the old
        # prefix must survive, and a substantial fixed suffix must identify the
        # sub-call.  Crucially ``leading < previous_count`` proves this is NOT a
        # pure append (the bug in the original #2702 discriminator).
        trailing = _stable_trailing_block_run(current_blocks, previous_blocks, leading_run=leading)
        if (
            len(current_messages) == len(previous_messages)
            and current_count >= previous_count
            and _MIN_REWRITE_PREFIX_BLOCKS <= leading < previous_count
            and leading * 2 >= previous_count
            and leading * 2 >= current_count
            and trailing >= _MIN_REWRITE_SUFFIX_BLOCKS
        ):
            changed = HistoryRelation(
                RELATION_BLOCK_REWRITE_TAIL,
                message_index=index,
                stable_prefix_blocks=leading,
                stable_suffix_blocks=trailing,
                previous_block_count=previous_count,
                current_block_count=current_count,
            )
            continue

        return HistoryRelation(RELATION_DIVERGED)

    if changed is not None:
        return changed
    return HistoryRelation(
        RELATION_EXACT
        if len(current_messages) == len(previous_messages)
        else RELATION_MESSAGE_APPEND
    )


def classify_history_relation(
    current_messages: list[dict[str, Any]],
    previous_messages: list[dict[str, Any]],
) -> HistoryRelation:
    """Return the canonical cross-turn relationship for two raw histories.

    The canonical projection may drop a whole directive-only message.  Refuse
    classification when that would shift raw message indices: block replay
    always slices the raw lists and must never consume a canonical index as a
    raw one.
    """
    if not current_messages or not previous_messages:
        return HistoryRelation(RELATION_DIVERGED)
    current = _lineage_snapshot(_canonicalize_for_prefix_compare(current_messages))
    previous = _lineage_snapshot(_canonicalize_for_prefix_compare(previous_messages))
    prefix_len = len(previous_messages)
    if len(previous) != prefix_len:
        return HistoryRelation(RELATION_DIVERGED)
    if len(_canonicalize_for_prefix_compare(current_messages[:prefix_len])) != prefix_len:
        return HistoryRelation(RELATION_DIVERGED)
    return _classify_history_canonical(current, previous)


def segment_fingerprint(value: Any) -> str:
    """Stable hash for non-message provider cache-key segments.

    Cache-control placement and transport annotations are deliberately ignored;
    semantic tool/model/thinking changes remain visible.  The hash is affinity
    metadata only and is never used to reconstruct or forward request content.
    """
    canonical = _lineage_snapshot(_canonicalize_for_prefix_compare(value))
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class AppendOnlyClassification:
    """Canonical relationship between two consecutive message histories."""

    block_frontier: tuple[int, int] | None = None


def _classify_append_only_canonical(
    current_messages: list[Any],
    previous_messages: list[Any],
) -> AppendOnlyClassification | None:
    """Classify a history that preserves the previous history's semantics."""
    if len(current_messages) < len(previous_messages):
        return None

    block_frontier: tuple[int, int] | None = None
    for index, previous_message in enumerate(previous_messages):
        current_message = current_messages[index]
        if current_message == previous_message:
            continue

        # A single existing message may grow by appending content blocks. Any
        # replacement, insertion, or second changed message is a divergence.
        if block_frontier is not None:
            return None
        if not isinstance(previous_message, dict) or not isinstance(current_message, dict):
            return None
        # Everything outside ``content`` must be unchanged. A message whose
        # role, name, tool ids, or any other field moved is a replacement, not a
        # block append: ``overlay_cached_prefix`` replays the previously
        # forwarded message's fields under the merged block list, so a changed
        # field would silently forward last turn's metadata.
        if _message_fields_outside_content(previous_message) != _message_fields_outside_content(
            current_message
        ):
            return None
        previous_content = previous_message.get("content")
        current_content = current_message.get("content")
        if (
            not isinstance(previous_content, list)
            or not isinstance(current_content, list)
            or len(current_content) <= len(previous_content)
            or current_content[: len(previous_content)] != previous_content
        ):
            return None
        # Record the frontier unconditionally. An empty previous content list
        # still means this message grew, and reporting no frontier would let
        # ``extract_cache_stable_delta`` read the history as a pure
        # whole-message append and slice the appended blocks away. A canonical
        # content list is empty whenever every block projected to ``{}``, which
        # is what a pure cache-directive block does.
        block_frontier = (index, len(previous_content) - 1)

    return AppendOnlyClassification(block_frontier=block_frontier)


def extract_cache_stable_delta(
    current_messages: list[dict[str, Any]],
    previous_original_messages: list[dict[str, Any]] | None,
    previous_forwarded_messages: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return ``(stable_forwarded_prefix, appended_delta_messages)`` when the current
    request append-only-extends the previous one, else ``None``.

    Provider-agnostic delta engine for cache mode. "Append-only" is decided by comparing
    the *canonicalized* prefix (:func:`_canonicalize_for_prefix_compare`, which ignores
    per-turn transport / cache-directive / client-annotation noise across
    Anthropic / OpenAI / Bedrock and the common clients), so a moved cache marker or
    shape churn does not spuriously collapse cache mode to raw forwarding. On a match the
    caller replays the byte-identical previously-forwarded prefix and compresses ONLY the
    appended delta.

    This is a COMPARISON + slice only: the returned prefix is the previously-forwarded
    bytes verbatim and the delta is the raw appended messages — never a rebuild from the
    canonical projection — so the projection dropping non-semantic fields is safe.
    """
    if not previous_original_messages or previous_forwarded_messages is None:
        return None
    relation = classify_history_relation(current_messages, previous_original_messages)
    if relation.kind not in (RELATION_EXACT, RELATION_MESSAGE_APPEND):
        # A same-message block append needs a block-level splice in
        # ``overlay_cached_prefix``; slicing only whole messages would silently
        # discard its new blocks.  Rewritten tails are not append-only deltas.
        return None
    prefix_len = len(previous_original_messages)
    return (
        copy.deepcopy(previous_forwarded_messages),
        copy.deepcopy(current_messages[prefix_len:]),
    )


def overlay_cached_prefix(
    optimized_messages: list[dict[str, Any]],
    current_original_messages: list[dict[str, Any]],
    previous_original_messages: list[dict[str, Any]] | None,
    previous_forwarded_messages: list[dict[str, Any]] | None,
    *,
    confirmed_frozen_count: int | None = None,
) -> list[dict[str, Any]]:
    """Replay a positional, non-inflating cached prefix when it is safe.

    Provider-agnostic cache-safety guard for the freeze path. When a message is
    "frozen", the compression pipeline may emit the agent's ORIGINAL bytes for
    it — but the provider cached whatever we FORWARDED last turn (the compressed
    form). Forwarding the original then mismatches the cached prefix and busts
    the prompt cache from that point (100% of observed misses were this
    ``prefix_change``). This overlays the exact previously-forwarded prefix onto
    the corresponding leading messages so the forwarded prefix stays byte-for-byte
    what the provider hashed for its cache key.

    Safe only when this turn extends the previous turn in a proven positional
    shape: either whole-message append or pure block append inside one message.
    There must be exactly one previous forwarded message per original. Otherwise
    the previous bytes may not correspond to the same positions, so we return
    ``optimized_messages`` unchanged (accept a possible bust rather than forward
    wrong content).

    The optimized and current-original lists must be positionally aligned, and
    compact UTF-8 JSON for the replayed result must not exceed the optimized
    candidate. These bounds prefer a cache miss to corrupting or inflating a
    client's live history.

    ``confirmed_frozen_count`` bounds UNCONDITIONAL replay. Leading positions
    the provider has already confirmed cached (a message count derived from
    ``cache_read_input_tokens``) are always replayed byte-identical: the
    replay source there is exactly what the provider hashed, so changing
    those bytes can only bust the cache. Beyond the floor the size bound
    still arbitrates each turn: a shrinking replay (the pipeline emitted
    original bytes for a frozen message) is repaired, while an inflating one
    (fresh compression improved on the forwarded form) is declined so the
    improvement reaches the wire. When the provider count collapses (cold
    cache, TTL lapse) the floor collapses with it and every accumulated
    improvement lands at once - the natural re-baselining that keeps
    long-session growth bounded (#3026). Callers with no provider-confirmed
    count pass None and keep the fully size-bounded behavior.
    """
    prev_orig = previous_original_messages
    prev_fwd = previous_forwarded_messages
    if not prev_orig or not prev_fwd:
        return optimized_messages
    if len(optimized_messages) != len(current_original_messages):
        logger.debug(
            "overlay: optimized/current-original length mismatch (optimized=%d, current=%d) "
            "— skipping positional cached-prefix replay",
            len(optimized_messages),
            len(current_original_messages),
        )
        return optimized_messages
    n = len(prev_orig)
    # Positional 1:1 correspondence between prev_orig[i] and prev_fwd[i] holds
    # only when last turn forwarded exactly one message per original (the
    # append-only, no-injection shape update_from_response records). If the
    # counts differ, an injected / dropped / merged message shifted the
    # mapping, so replaying prev_fwd[i] at position i could forward the wrong
    # content — bail (leave this turn's output untouched) rather than risk it.
    mapping: dict[int, int] | None = None
    if len(prev_fwd) != n:
        mapping = _align_forwarded_to_original(prev_orig, prev_fwd)
        if not mapping:
            logger.debug(
                "overlay: forwarded/original count mismatch (prev_fwd=%d, "
                "prev_orig=%d) and no tool-id anchor aligned them — skipping "
                "cached-prefix replay (possible bust)",
                len(prev_fwd),
                n,
            )
            return optimized_messages
        logger.debug(
            "overlay: forwarded/original count mismatch (prev_fwd=%d, prev_orig=%d) "
            "— realigned %d messages on tool-id anchors",
            len(prev_fwd),
            n,
            len(mapping),
        )
    relation = classify_history_relation(current_original_messages, prev_orig)
    if relation.kind == RELATION_BLOCK_APPEND and relation.message_index is not None:
        message_index = relation.message_index
        if message_index < len(optimized_messages):
            previous_message = prev_fwd[message_index]
            previous_original_message = prev_orig[message_index]
            current_message = optimized_messages[message_index]
            previous_content = (
                previous_message.get("content") if isinstance(previous_message, dict) else None
            )
            previous_original_content = (
                previous_original_message.get("content")
                if isinstance(previous_original_message, dict)
                else None
            )
            current_content = (
                current_message.get("content") if isinstance(current_message, dict) else None
            )
            # The frontier index comes from the CANONICAL projection, so it
            # cannot be used to slice raw block lists: canonicalization drops
            # pure directive blocks, and compression can change the count too.
            # Re-establish the split in raw terms instead. Last turn must have
            # forwarded one block per original block, and this turn's leading
            # blocks must canonicalize-equal last turn's originals. Then, and
            # only then, replaying the forwarded blocks and appending the rest
            # is exactly the growth.
            split = (
                len(previous_original_content)
                if isinstance(previous_original_content, list)
                else -1
            )
            if (
                isinstance(previous_content, list)
                and isinstance(previous_original_content, list)
                and isinstance(current_content, list)
                and len(previous_content) == split
                and len(current_content) >= split
                and _canonicalize_for_prefix_compare(current_content[:split])
                == _canonicalize_for_prefix_compare(previous_original_content)
            ):
                merged = copy.deepcopy(previous_message)
                merged["content"] = copy.deepcopy(previous_content) + copy.deepcopy(
                    current_content[split:]
                )
                logger.debug(
                    "overlay: replayed %d forwarded blocks and appended %d new blocks "
                    "inside message %d",
                    split,
                    len(current_content) - split,
                    message_index,
                )
                replayed = (
                    list(prev_fwd[:message_index])
                    + [merged]
                    + list(optimized_messages[message_index + 1 :])
                )
                if message_index >= max(confirmed_frozen_count or 0, 0):
                    replayed_bytes = _compact_json_bytes(replayed)
                    optimized_bytes = _compact_json_bytes(optimized_messages)
                    if (
                        replayed_bytes is None
                        or optimized_bytes is None
                        or len(replayed_bytes) > len(optimized_bytes)
                    ):
                        logger.debug("overlay: block replay inflated compact JSON — skipping")
                        return optimized_messages
                return replayed
    # Append-only guard on CONTENT ONLY, message-by-message. Replay the
    # previously-forwarded (cached, compressed) bytes for the longest LEADING
    # run of messages that is byte-for-byte (content-canonical) identical to
    # what we forwarded last turn, and stop at the first divergence.
    #
    # This is the cache-safety centerpiece for token mode (which relies solely
    # on this replay; cache mode is already byte-stable by construction). The
    # prior all-or-nothing guard busted the ENTIRE cached prefix the moment any
    # single leading message failed to canonicalize-equal last turn — most
    # commonly the just-added assistant turn, whose client-resent form can
    # differ trivially from the copy we reconstructed and recorded. Stopping at
    # the first divergence instead keeps the (much larger) cache-hit region
    # up to that point and only re-forwards from the changed message onward.
    #
    # Comparison uses the shared canonicalizer (not just cache_control
    # stripping) so it is robust to ALL per-turn transport / annotation churn —
    # cache_control movement (Anthropic), litellm `caller`,
    # provider_specific_fields, streaming `index`, string<->block content shape,
    # etc. Content stability is what the provider's prefix cache actually keys
    # on. Safe by construction: we only replay prev_fwd[k] where
    # current_original[k] canonicalize-equals prev_orig[k], and prev_fwd[k]
    # positionally corresponds to prev_orig[k] (guaranteed by the count check
    # above), so no wrong bytes are ever forwarded.
    limit = min(n, len(current_original_messages), len(optimized_messages))
    if mapping is not None:
        # Only replay up to the last original index we could align to a
        # forwarded message; beyond it the correspondence is unknown again.
        limit = min(limit, max(mapping) + 1)
    k = 0
    while k < limit and _canonicalize_for_prefix_compare(
        current_original_messages[k]
    ) == _canonicalize_for_prefix_compare(prev_orig[k]):
        k += 1
    while mapping is not None and k > 0 and (k - 1) not in mapping:
        k -= 1
    if k == 0:
        logger.debug(
            "overlay: prefix diverged at message 0 — no cached-prefix replay "
            "(cold prefix or client rewrote history head)"
        )
        return optimized_messages
    if k < n:
        logger.debug(
            "overlay: cached-prefix replay for %d/%d leading messages "
            "(diverged at %d — re-forwarding tail fresh)",
            k,
            n,
            k,
        )
    # Replay the cached (compressed) prefix byte-identical up to the first
    # divergence; keep this turn's freshly-produced output for the rest.
    # Under anchor realignment the forwarded run reaching original index k-1
    # ends at mapping[k - 1], which also carries any message injected inside
    # that run, so the replayed bytes stay exactly what the provider billed.
    forwarded_end = k if mapping is None else mapping[k - 1] + 1
    replayed = list(prev_fwd[:forwarded_end]) + list(optimized_messages[k:])
    replayed_bytes = _compact_json_bytes(replayed)
    optimized_bytes = _compact_json_bytes(optimized_messages)
    if (
        replayed_bytes is None
        or optimized_bytes is None
        or len(replayed_bytes) > len(optimized_bytes)
    ):
        # Something in the replay is byte-larger than this turn's fresh form:
        # fresh compression improved on already-forwarded bytes. Landing the
        # improvement is only safe OUTSIDE the provider-confirmed prefix -
        # inside it, the improvement would change bytes the provider has
        # already cached and bust the whole suffix. Split at the confirmed
        # floor: replay the confirmed region unconditionally, forward
        # everything beyond it fresh so the improvement reaches the wire.
        floor = min(k, max(confirmed_frozen_count or 0, 0))
        if floor <= 0:
            logger.debug("overlay: replay inflated compact JSON — skipping cached-prefix replay")
            return optimized_messages
        logger.debug(
            "overlay: replay inflated beyond the confirmed floor — replaying %d/%d "
            "confirmed messages, forwarding the rest fresh",
            floor,
            k,
        )
        return list(prev_fwd[:floor]) + list(optimized_messages[floor:])
    return replayed


def _tool_anchor(msg: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return an identity anchor that survives every headroom transform.

    Tool-call ids are minted by the provider and no transform rewrites them, so
    they still identify a forwarded message after its text has been masked,
    compressed or re-shaped. Messages carrying no tool id fall back to a
    role-only anchor and are aligned positionally.
    """
    role = str(msg.get("role", ""))
    ids: list[str] = []
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_id = block.get("id") or block.get("tool_use_id")
            if isinstance(block_id, str) and block_id:
                ids.append(block_id)
    return (role, tuple(ids))


def _align_forwarded_to_original(
    previous_original_messages: list[dict[str, Any]],
    previous_forwarded_messages: list[dict[str, Any]],
) -> dict[int, int]:
    """Map original index -> forwarded index for last turn's pair of lists.

    Positional 1:1 correspondence only holds when the turn forwarded exactly one
    message per original. Injection, dropping or merging shifts the mapping, and
    the previous behaviour was to bail out of cached-prefix replay entirely for
    the rest of the session. Anchoring on tool ids recovers the mapping instead,
    which is what keeps the already-billed prefix byte-stable.
    """
    mapping: dict[int, int] = {}
    forwarded_anchors = [_tool_anchor(msg) for msg in previous_forwarded_messages]
    # Fewer forwarded than original means a message was dropped or merged, and
    # nothing says where. A tool id still pins its own message, but position no
    # longer does, so unanchored messages get no positional fallback at all.
    allow_positional = len(previous_forwarded_messages) >= len(previous_original_messages)
    cursor = 0
    for index, original in enumerate(previous_original_messages):
        if cursor >= len(previous_forwarded_messages):
            break
        role, ids = _tool_anchor(original)
        if ids:
            scan = cursor
            while scan < len(forwarded_anchors) and forwarded_anchors[scan] != (
                role,
                ids,
            ):
                scan += 1
            if scan >= len(forwarded_anchors):
                break
            mapping[index] = scan
            cursor = scan + 1
            continue
        # A message carrying no tool id has nothing to verify a pairing against,
        # so it may only be paired positionally while the two lists have not
        # drifted apart yet. Once an anchor has shifted the cursor, pairing an
        # unanchored message on position alone could silently skip a dropped
        # message and replay the wrong bytes, so stop the run here instead.
        if not allow_positional or cursor != index or forwarded_anchors[cursor][0] != role:
            break
        mapping[index] = cursor
        cursor += 1
    return mapping


def latest_message_cache_control_ttl(messages: list[dict[str, Any]]) -> str | None:
    """Return the newest valid message cache TTL before markers are stripped."""
    for msg in reversed(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for block in reversed(content):
                if not isinstance(block, dict):
                    continue
                cache_control = block.get("cache_control")
                if isinstance(cache_control, dict) and cache_control.get("ttl") in (
                    "5m",
                    "1h",
                ):
                    return str(cache_control["ttl"])

        cache_control = msg.get("cache_control")
        if isinstance(cache_control, dict) and cache_control.get("ttl") in ("5m", "1h"):
            return str(cache_control["ttl"])
    return None


def _stable_leading_block_run(
    current_blocks: list[Any],
    previous_blocks: list[Any] | None,
) -> int:
    """Length of the longest leading run of content blocks that canonicalize-equal
    the previous turn's blocks.

    Uses :func:`_canonicalize_for_prefix_compare`, which drops ``cache_control``
    and other non-semantic keys, so a moved breakpoint or per-turn annotation
    churn does not shorten the run. Two blocks canonicalize-equal iff their
    forwarded content is the same, which is exactly what the provider's prefix
    cache keys on, so this run is the part of a message that can still be read
    from cache.
    """
    if not previous_blocks:
        return 0
    limit = min(len(current_blocks), len(previous_blocks))
    k = 0
    while k < limit and _canonicalize_for_prefix_compare(
        current_blocks[k]
    ) == _canonicalize_for_prefix_compare(previous_blocks[k]):
        k += 1
    return k


# A leading run shorter than this is not conversation identity. Injected
# boilerplate can bracket a message on both sides, so two unrelated sub-calls
# under one session id can agree on their first few blocks by construction.
_MIN_CONTINUATION_RUN_BLOCKS = 8


def _is_message_continuation(recorded: Any, incoming: Any) -> bool:
    """Return True iff ``incoming`` is ``recorded`` grown in place.

    Client histories are append-only at MESSAGE granularity, which is what
    lineage matching assumes, except for sub-call shapes that pack a transcript
    into one block-style message and extend that message's block list each turn.
    Such a turn is the same conversation continuing, but its newest message is
    not equal to the recorded one, so the whole-message prefix test rejects it.

    "Grown in place" is deliberately narrow, because a false match makes two
    concurrent conversations share one tracker, the thrash lineages exist to
    prevent. All of the following must hold:

    * same role, block-style content on both sides, and no blocks lost.
    * a leading run of byte-stable blocks that is both substantial in absolute
      terms and MOST of the recorded version. A few shared blocks is not a
      conversation identity.
    * an unchanged FINAL block. Conversations that pack the same parent
      transcript differ in the instruction they append after it, so the tail is
      what distinguishes siblings from a continuation of one stream. The shapes
      this exists for keep a fixed suffix pinned at the end while the blocks
      before it churn.

    Anything that fails these keeps its own lineage, which is the pre-existing
    behaviour and merely forgoes the cache win.
    """
    if not (isinstance(recorded, dict) and isinstance(incoming, dict)):
        return False
    if recorded.get("role") != incoming.get("role"):
        return False
    old = recorded.get("content")
    new = incoming.get("content")
    if not (isinstance(old, list) and isinstance(new, list)):
        return False
    # Strictly longer, not merely no-shorter. "Grown in place" means the message
    # gained content, so an incoming message of the SAME block count is not
    # growth however well its edges line up. Accepting equal lengths let two
    # unrelated conversations under one session id be judged continuations of
    # each other whenever they shared a large enough artifact: quote one
    # 20-block document into two sibling sub-agent prompts, end both with the
    # same boilerplate instruction, and the leading-run and final-block tests
    # both pass while the messages differ only in the middle. That is #2085,
    # and the second caller's first-ever message would take over the first
    # caller's tracker.
    if not old or len(new) <= len(old):
        return False
    if old[-1] != new[-1]:
        return False
    run = _stable_leading_block_run(new, old)
    return run >= _MIN_CONTINUATION_RUN_BLOCKS and run * 2 >= len(old)


# Env kill switch for the stable-boundary breakpoint placement below. Default on.
# Set to 0/false/no/off to restore the newest-block placement unconditionally.
# This sits on the cache-key path of every Anthropic request, so it needs a
# rollback that does not require shipping a new build.
_STABLE_BOUNDARY_ENV = "HEADROOM_STABLE_BOUNDARY_BREAKPOINT"

# Below this many blocks a message cannot benefit from relocation: the provider
# walks back up to 20 content-block boundaries from the breakpoint looking for a
# previous write, so a short message's prefix is still found from the newest
# block, and anchoring backwards would only shrink what gets cached.
_MIN_BLOCKS_FOR_RELOCATION = 20


def _stable_boundary_enabled() -> bool:
    return os.environ.get(_STABLE_BOUNDARY_ENV, "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _breakpoint_index(
    content: list[Any],
    message: dict[str, Any],
    message_index: int,
    previous_forwarded_messages: list[dict[str, Any]] | None,
) -> int:
    """Choose newest for appends, stable-prefix end for rewritten tails."""
    newest = len(content) - 1
    if (
        not previous_forwarded_messages
        or not _stable_boundary_enabled()
        or len(content) < _MIN_BLOCKS_FOR_RELOCATION
        or message_index >= len(previous_forwarded_messages)
    ):
        return newest
    previous = previous_forwarded_messages[message_index]
    if not isinstance(previous, dict):
        return newest
    relation = classify_history_relation([message], [previous])
    if relation.kind != RELATION_BLOCK_REWRITE_TAIL:
        return newest
    logger.debug(
        "cache breakpoint anchored to stable run %d/%d blocks in message %d "
        "(previous=%d, stable_suffix=%d)",
        relation.stable_prefix_blocks,
        relation.current_block_count,
        message_index,
        relation.previous_block_count,
        relation.stable_suffix_blocks,
    )
    return relation.stable_prefix_blocks - 1



def _compact_json_bytes(value: Any) -> bytes | None:
    """Return compact JSON bytes, or ``None`` when sizing cannot be proved."""
    try:
        return json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError):
        return None


def _client_marker_positions(
    client_messages: list[dict[str, Any]],
) -> list[tuple[int, int, dict[str, Any]]]:
    """(message index, block index, marker) for every CLIENT cache_control.

    Block-level, not one-per-message: clients mark multiple blocks within a
    single long message (Claude Code does this on 1-2-message requests with a
    large first message), and the ~20-block lookback applies within a message
    just as it does across messages. Only block-style content carries markers.
    """
    positions: list[tuple[int, int, dict[str, Any]]] = []
    for i, msg in enumerate(client_messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for bi, b in enumerate(content):
            if isinstance(b, dict) and isinstance(b.get("cache_control"), dict):
                positions.append((i, bi, b["cache_control"]))
    return positions


def normalize_message_cache_control(
    messages: list[dict[str, Any]],
    previous_forwarded_messages: list[dict[str, Any]] | None = None,
    *,
    client_messages: list[dict[str, Any]] | None = None,
    force_ttl: str | None = None,
) -> list[dict[str, Any]]:
    """Own message-level cache_control placement so breakpoints stay bounded.

    Two forces pile up cache_control markers turn over turn: clients move the
    breakpoint to the newest message each call, and ``overlay_cached_prefix``
    replays the markers that rode on each turn's then-newest message. Anthropic
    hard-errors at **>4 cache_control blocks total** (system + tools + messages),
    so on a long conversation the accumulation eventually 400s.

    Fix: strip EVERY message-level cache_control, then re-place markers at the
    positions the CLIENT's current request marks (``client_messages``). The
    client's positions are load-bearing, not redundant: Anthropic resolves each
    breakpoint by walking back **at most ~20 content blocks** for a prior cache
    entry, and agentic clients (Claude Code) keep a marker on the previous
    turn's newest message precisely so the new turn's write can chain to the
    old entry. Collapsing to a single newest-block marker breaks that chain
    whenever one turn adds >20 blocks (typical for tool-heavy turns): the
    lookback misses, the entire message history silently re-bills as cache
    creation, and a marker anchored short of the final block leaves the tail
    billing as fully uncached input. Mirroring the client's positions bounds
    accumulation identically (the client manages its own 4-marker budget) while
    preserving its read/write chaining.

    The provider's cache key is message CONTENT, not marker presence (moving
    the breakpoint forward is the documented client pattern and it hits), so
    stripping replay leftovers and re-placing markers never busts. system/tools
    breakpoints live outside ``messages`` and are left untouched.

    Headroom owns WHICH BLOCK carries each marker; the client owns the message
    positions and the marker values, so an explicit ``ttl`` (e.g. ``"1h"``)
    survives per position instead of silently downgrading (#2375). The newest
    position uses stable-run anchoring for proven rewritten tails; earlier
    positions go on their message's last block.

    Without ``client_messages`` (or when the transformed list no longer aligns
    with it), falls back to the legacy single-marker consolidation. Only
    block-style (list) content can carry cache_control; string content is left
    as-is. Returns the input unchanged when there is nothing to normalize.

    ``force_ttl`` overrides the ttl written onto every re-placed breakpoint. The
    default (None) reuses each client marker verbatim (the #2375 behavior
    above), right for a long-lived main session that may idle past 5
    minutes. A caller that knows the request is short-lived and non-resuming (a
    Claude Code sub-agent, measured median ~3 min) passes ``force_ttl="5m"`` so
    message-prefix writes land in the 1.25x tier instead of the 2x 1h tier the
    client would otherwise pay retention for. The ttl is a pure retention/cost
    knob: the provider keys the cache on content, not ttl, so this only changes
    write price and retention, never correctness.
    """
    # Anthropic accepts only "5m"/"1h" as a ttl; forwarding anything else 400s
    # the live request. force_ttl is a cost knob, never a correctness lever, so
    # an unrecognized value is ignored (fall back to the client's kept_ttl)
    # rather than propagated. "" is not a pass-through for "provider default"
    # here; it is treated as unset.
    if force_ttl not in ("5m", "1h", None):
        force_ttl = None
    changed = False
    out: list[dict[str, Any]] = []
    last_block_idx = -1
    # Preserve the client's cache marker across normalization: re-placing a bare
    # ephemeral marker silently downgrades a 1h client breakpoint to the 5m tier,
    # so any idle gap over 5 minutes lapses a cache the client paid 2x write
    # premium to keep for an hour. Reuse the newest client marker verbatim so an
    # explicit ttl (and any future cache_control field) survives (#2375). The
    # newest marker in message order is the client's current intent; older ones
    # are replay leftovers.
    last_marker: dict[str, Any] | None = None
    for i, msg in enumerate(messages):
        # A cache_control sitting on the MESSAGE, sibling to content rather than
        # inside a block, is a shape the API accepts and clients do send. The
        # block scan below never sees it, so it used to survive every turn
        # untouched: six such messages left seven live markers, past the hard
        # four-breakpoint limit this function exists to enforce. Strip it here
        # and let it feed marker selection like any block-level one.
        if isinstance(msg, dict) and "cache_control" in msg:
            if isinstance(msg["cache_control"], dict):
                last_marker = msg["cache_control"]
            msg = {k: v for k, v in msg.items() if k != "cache_control"}
            changed = True
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            had = False
            for b in content:
                if isinstance(b, dict) and "cache_control" in b:
                    had = True
                    if isinstance(b["cache_control"], dict):
                        last_marker = b["cache_control"]
            stripped = [
                {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                for b in content
            ]
            out.append({**msg, "content": stripped} if had else msg)
            changed = changed or had
            if stripped and isinstance(stripped[-1], dict):
                last_block_idx = i
        else:
            out.append(msg)

    def _final_marker(marker: dict[str, Any]) -> dict[str, Any]:
        # Reuse the client marker verbatim (#2375), then let an explicit
        # force_ttl override the retention tier for short-lived sub-agents.
        m = dict(marker)
        if force_ttl is not None:
            m["ttl"] = force_ttl
        return m

    def _place(
        target_idx: int,
        marker: dict[str, Any],
        *,
        anchor: bool,
        block_idx: int | None = None,
    ) -> bool:
        msg = out[target_idx]
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            return False
        content = list(content)
        if block_idx is not None and 0 <= block_idx < len(content):
            # Transforms can shift block indices (e.g. a dropped thinking
            # block); a slightly-off placement still lands on a stable block
            # in the same message, which is harmless — markers are not part
            # of the provider's cache key.
            breakpoint_index = block_idx
        elif anchor:
            breakpoint_index = _breakpoint_index(
                content, msg, target_idx, previous_forwarded_messages
            )
        else:
            breakpoint_index = len(content) - 1
        # Anthropic content blocks are dictionaries, but callers can still
        # supply mixed list content. Fall back to the newest block rather than
        # attempting ``**`` on a scalar stable-boundary element, and skip the
        # message entirely when even that is not a dict.
        if not isinstance(content[breakpoint_index], dict):
            breakpoint_index = len(content) - 1
        if not isinstance(content[breakpoint_index], dict):
            return False
        content[breakpoint_index] = {**content[breakpoint_index], "cache_control": _final_marker(marker)}
        out[target_idx] = {**msg, "content": content}
        return True

    # Preferred: mirror the client's marker positions 1:1, block-level. The
    # transform pipeline preserves message count, so index alignment is the
    # invariant; fall back to legacy consolidation if it ever does not hold,
    # or when the client marked nothing (legacy still places one so the
    # prefix caches). The newest client marker keeps stable-run anchoring
    # when the client placed it on its message's final block (intent: "cache
    # through the end"); an explicit mid-message marker is honored verbatim.
    if client_messages is not None and len(client_messages) == len(messages):
        positions = _client_marker_positions(client_messages)
        if positions:
            placed_any = False
            newest_mi, newest_bi, _ = positions[-1]
            newest_client_content = client_messages[newest_mi].get("content")
            newest_on_final_block = (
                isinstance(newest_client_content, list)
                and newest_bi == len(newest_client_content) - 1
            )
            for mi, bi, marker in positions:
                is_newest = (mi, bi) == (newest_mi, newest_bi)
                if is_newest and newest_on_final_block:
                    placed = _place(mi, marker, anchor=True)
                else:
                    placed = _place(mi, marker, anchor=False, block_idx=bi)
                placed_any = placed or placed_any
            if placed_any or changed:
                return out
            return messages

    # Legacy: re-place exactly one breakpoint on the last block-style message.
    if last_block_idx >= 0:
        marker = dict(last_marker) if last_marker else {"type": "ephemeral"}
        _place(last_block_idx, marker, anchor=True)
        changed = True
    return out if changed else messages


class PrefixCacheTracker:
    """Tracks provider prefix cache state across turns in a session.

    Usage:
        tracker = PrefixCacheTracker("anthropic")

        # Before compression (turn 2+):
        frozen = tracker.get_frozen_message_count()
        result = pipeline.apply(messages, model, frozen_message_count=frozen)

        # After API response:
        tracker.update_from_response(
            cache_read_tokens=usage["cache_read_input_tokens"],
            cache_write_tokens=usage["cache_creation_input_tokens"],
            messages=optimized_messages,
            tokenizer=tokenizer,
        )
    """

    def __init__(self, provider: str, config: PrefixFreezeConfig | None = None):
        self.provider = provider
        self.config = config or PrefixFreezeConfig()
        self._cached_token_count: int = 0
        self._cached_message_count: int = 0
        self._turn_number: int = 0
        self._last_activity: float = time.time()
        self._last_original_messages: list[dict[str, Any]] = []
        # Fingerprint of the bytes ahead of every message in the cache prefix
        # (system + tools). Deferred tool loading grows the tools array
        # mid-session, which kills the whole cached suffix while the messages
        # stay byte-identical, so message comparison alone reports a warm
        # prefix on a turn that is in fact a total bust. See
        # observe_client_churn and docs/prefix-waste-2026-07-25.md.
        self._last_head_fingerprint: str | None = None
        self._last_forwarded_messages: list[dict[str, Any]] = []
        # How many of the above were actually sent upstream, as opposed to
        # projected onto the tail from the response. See update_from_response.
        self._last_sent_message_count: int = 0
        # Recent observed inter-turn gaps (seconds between this session's
        # successive requests), for adaptive cache-TTL tier selection. A short,
        # bounded ring: only the recent cadence matters and old gaps should age
        # out. Fed by the handler with the pre-refresh idle gap each turn.
        self._turn_gaps: deque[float] = deque(maxlen=8)
        self._ttl_recommendation: str | None = None
        # Smoothed incremental cache write per warm turn, and how many turns
        # this session has run per observed TTL breach. Together they say
        # whether the 1h write premium is cheaper than re-writing the prefix.
        self._steady_write_tokens: float | None = None
        self._ttl_breaches: int = 0
        # Depth fractions (0..1 of last turn's original prefix) where client
        # churn structurally diverged the history, recorded by
        # observe_client_churn. Consumed by anchor placement: the empirical
        # churn-depth profile says where breakpoints stop paying. Bounded ring,
        # recent churn behavior is what matters for this session.
        self._churn_depth_fractions: deque[float] = deque(maxlen=32)
        # Absolute message indices where 1h anchors were actually attached last
        # turn. Anchor placement re-solves against inputs that move every turn
        # (churn fractions rescale with message count, the ring turns over), so
        # without a memory of where the live anchors sit the argmin walks and
        # re-writes the span it crosses. Absolute, not fractional: in steady
        # state messages are only appended, so an anchor that does not move
        # keeps the same index and the same bytes.
        self._placed_anchor_depths: list[int] = []
        # Token-mode cost-aware prefix gate state. Once the break-even math
        # decides to compress this session, it latches on: pressure and expected
        # reads only grow, so re-deciding every turn could flip compressed back to
        # original and bust the compressed cache. ``_last_compression_kept`` is the
        # fraction of tokens the last compression KEPT (after/before), used to
        # estimate the next turn's saving before actually running compression.
        self._compress_latched: bool = False
        self._last_compression_kept: float | None = None
        # Exponentially-weighted estimate of the KEPT fraction and its variance.
        # ``_kept_ewma`` is the smoothed prediction the cost gate reads;
        # ``_kept_var`` is an EWMA of the squared deviation, so its square root
        # is a confidence spread the gate discounts by (assume less saving when
        # the estimate is noisy). Both stay None/0 until the first observation.
        self._kept_ewma: float | None = None
        self._kept_var: float = 0.0
        # First-class hybrid mode owns its prefix generation and rebase
        # hysteresis here so it follows the same session affinity and expiry as
        # provider cache observations.
        self.hybrid_controller = HybridModeController(provider)
        # Idle gap (seconds) since the PREVIOUS turn's response, captured by
        # SessionTrackerStore.get_or_create at fetch time — BEFORE it refreshes
        # _last_activity. Without this snapshot, seconds_since_activity() reads
        # ~0 on every request (the fetch itself bumps the clock), so the
        # net-cost/TTL P_alive gate could never see idle time. The handler reads
        # this and forwards it to the pipeline as `idle_seconds`.
        self._idle_seconds_at_fetch: float = 0.0

        # Hashes of the non-message cache-key segments as last forwarded, so a
        # prefix_change can name WHICH segment moved. Segment NAMES are supplied
        # by the caller (each provider's request shape differs — see
        # classify_cache_miss), so this stays an open mapping rather than fixed
        # per-provider fields. Only hashes are kept: they answer "did these exact
        # bytes change?" without retaining a second copy of the tool schemas.
        self._last_segment_hashes: dict[str, str] = {}

        # Session-scoped ReadMaturationManager (Mechanism B), created
        # lazily by the handler when read maturation is enabled. Rides
        # here so it shares the session's affinity and TTL cleanup.
        self.read_maturation_manager: Any = None

        # The output-shaping verbosity level this conversation's forwarded
        # system prompt was established at, or ``None`` if the handler has
        # never shaped it. The Anthropic handler pins this level for as long as
        # the frozen prefix stands: the tail is part of the cached prefix, so
        # adding it, dropping it, or changing its level all invalidate the
        # provider cache. Level 0 is a real pinned value (no tail) and is
        # deliberately distinct from ``None`` (never shaped).
        self.output_shaping_level: int | None = None

        # Stats
        self._busts_avoided: int = 0
        self._tokens_preserved: int = 0
        self._compression_foregone_tokens: int = 0

    def export_state(self) -> dict[str, Any]:
        """Plain-data snapshot of everything a restart must not lose.

        The two message blobs are the point of the exercise: without them
        ``overlay_cached_prefix`` has nothing to replay, so every live session
        re-writes its whole history after a restart (measured at roughly 195k
        write tokens each). The scalars ride along because rebuilding them from
        a cold tracker would re-run cold-start behavior on a warm provider
        cache: turn 0 skips freezing entirely, and a reset compression latch
        can flip compressed back to original and bust the very prefix it
        preserved.

        Config is excluded on purpose. It is rebuilt from the environment on
        load so a restart that changes a knob takes the new value.
        ``read_maturation_manager`` is excluded too: the handler creates it
        lazily and it holds no cross-restart value.
        """
        return {
            "provider": self.provider,
            "last_original_messages": self._last_original_messages,
            "last_forwarded_messages": self._last_forwarded_messages,
            # Travels with the list it indexes into. Without it a restored
            # lineage cannot tell which of those messages it actually sent,
            # falls back to the whole list, and reports a bust on its first
            # turn back purely because the list carries a projected tail.
            "last_sent_message_count": self._last_sent_message_count,
            "last_head_fingerprint": self._last_head_fingerprint,
            "turn_number": self._turn_number,
            "cached_token_count": self._cached_token_count,
            "cached_message_count": self._cached_message_count,
            "last_activity": self._last_activity,
            "steady_write_tokens": self._steady_write_tokens,
            "ttl_breaches": self._ttl_breaches,
            "ttl_recommendation": self._ttl_recommendation,
            "turn_gaps": list(self._turn_gaps),
            "churn_depth_fractions": list(self._churn_depth_fractions),
            "placed_anchor_depths": list(self._placed_anchor_depths),
            "compress_latched": self._compress_latched,
            "last_compression_kept": self._last_compression_kept,
            "kept_ewma": self._kept_ewma,
            "kept_var": self._kept_var,
            "busts_avoided": self._busts_avoided,
            "tokens_preserved": self._tokens_preserved,
            "compression_foregone_tokens": self._compression_foregone_tokens,
            "hybrid": self.hybrid_controller.export_state(),
        }

    def restore_state(self, blob: dict[str, Any]) -> None:
        """Reinstate a snapshot from :meth:`export_state`.

        Every field is read defensively. A snapshot written by an older build
        can be missing keys, and a half-restored tracker is worse than a cold
        one: it would claim a frozen prefix it cannot actually replay.
        """
        self._last_original_messages = blob.get("last_original_messages") or []
        self._last_forwarded_messages = blob.get("last_forwarded_messages") or []
        # A snapshot from a build before this field existed leaves it at 0,
        # which reads as "unknown" and falls back to the cached count. That
        # is the old behaviour, not a new failure.
        self._last_sent_message_count = int(blob.get("last_sent_message_count", 0) or 0)
        self._last_head_fingerprint = blob.get("last_head_fingerprint")
        self._turn_number = int(blob.get("turn_number", 0) or 0)
        self._cached_token_count = int(blob.get("cached_token_count", 0) or 0)
        self._cached_message_count = int(blob.get("cached_message_count", 0) or 0)
        self._last_activity = float(blob.get("last_activity", time.time()))
        steady = blob.get("steady_write_tokens")
        self._steady_write_tokens = None if steady is None else float(steady)
        self._ttl_breaches = int(blob.get("ttl_breaches", 0) or 0)
        self._ttl_recommendation = blob.get("ttl_recommendation")
        self._turn_gaps = deque(blob.get("turn_gaps") or [], maxlen=8)
        self._churn_depth_fractions = deque(
            blob.get("churn_depth_fractions") or [], maxlen=32
        )
        self._placed_anchor_depths = list(blob.get("placed_anchor_depths") or [])
        self._compress_latched = bool(blob.get("compress_latched", False))
        kept = blob.get("last_compression_kept")
        self._last_compression_kept = None if kept is None else float(kept)
        ewma = blob.get("kept_ewma")
        self._kept_ewma = None if ewma is None else float(ewma)
        self._kept_var = float(blob.get("kept_var", 0.0) or 0.0)
        self._busts_avoided = int(blob.get("busts_avoided", 0) or 0)
        self._tokens_preserved = int(blob.get("tokens_preserved", 0) or 0)
        self._compression_foregone_tokens = int(
            blob.get("compression_foregone_tokens", 0) or 0
        )
        hybrid = blob.get("hybrid")
        if isinstance(hybrid, dict):
            self.hybrid_controller.restore_state(hybrid)
        # A restored tracker has not been fetched this process, so there is no
        # meaningful in-process idle gap yet. get_or_create computes the real
        # one from _last_activity on the next fetch.
        self._idle_seconds_at_fetch = 0.0

    def get_frozen_message_count(self) -> int:
        """How many leading messages to skip compression on the next turn.

        Returns 0 on turn 0 (cold start) or if caching is disabled/below threshold.
        """
        if not self.config.enabled:
            return 0
        if self._turn_number == 0:
            return 0
        if self._cached_token_count < self.config.min_cached_tokens:
            return 0
        return self._cached_message_count

    def record_turn_gap(self, gap_seconds: float | None) -> None:
        """Record the observed gap since this session's previous request.

        The handler peeks the idle gap BEFORE ``get_or_create`` refreshes
        ``_last_activity``, so ``gap_seconds`` is the real request-to-request
        cadence (assistant generation time plus user think time), which is
        exactly what decides whether the prompt cache is still warm when the
        next request lands. ``None`` (unknown session, first turn) and
        non-finite/negative values are ignored.
        """
        # Reject None and bool (a bool is an int subclass, so `float(True)` is a
        # silent 1.0 that would poison the cadence with a flag mistaken for a
        # duration). OverflowError covers a Python int too large for a float.
        if gap_seconds is None or isinstance(gap_seconds, bool):
            return
        try:
            g = float(gap_seconds)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(g) or g < 0.0:
            return
        self._turn_gaps.append(g)
        # A gap past the 5m tier would have lapsed a 5m entry. Counting these
        # over the whole session gives the breach rate that
        # prefers_long_ttl weighs against the 1h write premium.
        if g > 300.0:
            self._ttl_breaches += 1

    def recommended_ttl(
        self,
        *,
        tier_boundary_seconds: float = 300.0,
        margin_seconds: float = 60.0,
    ) -> str | None:
        """Pick the cheaper cache TTL tier this session can safely use.

        Anthropic sells exactly two ephemeral tiers: 5m (1.25x write) and 1h
        (2x write). The 1h premium only pays off when a gap between requests
        exceeds 5 minutes, so a fast-cadence session (the common case, measured
        median ~3 min per sub-agent turn) that pins 1h burns the 2x premium for
        retention it never uses. This reads the session's recent inter-turn gaps
        and returns ``"5m"`` when the cadence stays safely inside the 5-minute
        window, ``"1h"`` when a recent gap has already breached it, or ``None``
        when there is not enough history (or the cadence sits in the ambiguous
        band) to change the client's own choice.

        The decision is on the recent MAX gap, not the mean: the risk is a single
        long idle lapsing a 5m cache, so one breach flips the session back to 1h.
        ``margin_seconds`` keeps a safety buffer below the 5-minute boundary, and
        a one-tier hysteresis (the ambiguous band holds the last recommendation)
        stops the tier from flapping turn to turn.
        """
        if len(self._turn_gaps) < 2:
            return None
        recent_max = max(self._turn_gaps)
        if recent_max <= tier_boundary_seconds - margin_seconds:
            self._ttl_recommendation = "5m"
        elif recent_max > tier_boundary_seconds:
            self._ttl_recommendation = "1h"
        # else: ambiguous band -> keep the previous recommendation (hysteresis).
        if self._ttl_recommendation == "5m" and self.prefers_long_ttl():
            self._ttl_recommendation = "1h"
        return self._ttl_recommendation

    def prefers_long_ttl(self) -> bool:
        """Whether the 1h premium is cheaper than re-writing this prefix once.

        Cadence alone under-buys retention on a large prefix. Staying on 5m
        costs ``1.25 x prefix`` the first time an idle gap outruns the window,
        while 1h costs an extra ``0.75 x delta`` per warm turn, and the delta is
        the appended tail, not the whole prefix. So 1h pays for itself whenever
        a breach lands within ``1.25 x prefix / (0.75 x delta)`` turns. Measured
        on this proxy's own traffic: ~200k prefix against ~7k steady writes puts
        break-even near 46 turns, and breaches arrived roughly every 30, so a
        long conversation wants 1h even when the recent cadence looks fast.

        Small prefixes stay on 5m: below the size floor a re-write is cheap and
        the premium is not worth paying against a cadence that may stay fast.
        """
        prefix_tokens = self._cached_token_count
        steady_write = self._steady_write_tokens
        if prefix_tokens < _TTL_SIZE_FLOOR_TOKENS or not steady_write:
            return False
        breakeven_turns = (_TTL_5M_WRITE_PENALTY * prefix_tokens) / (
            _TTL_1H_PREMIUM * steady_write
        )
        # Observed turns per breach, with the whole session as the sample. No
        # breach yet means no evidence for the premium, so 5m holds.
        if self._ttl_breaches <= 0:
            return False
        turns_per_breach = self._turn_number / self._ttl_breaches
        return turns_per_breach < breakeven_turns

    def observe_client_churn(
        self,
        current_original_messages: list[dict[str, Any]],
        head_fingerprint: str | None = None,
    ) -> float:
        """Surviving fraction of last turn's ORIGINAL prefix in this turn's bytes.

        Structural bust detection, the pre-forward counterpart of
        :meth:`classify_cache_miss`. Compares this turn's client messages
        against last turn's recorded originals with the shared canonicalizer,
        client bytes against client bytes, so the proxy's own transforms can
        never register as churn. Returns ``k / n`` where ``k`` is the first
        divergent message index and ``n`` last turn's length: ``1.0`` means the
        prefix is intact (or there is no history to compare), ``0.0`` means the
        client rewrote the head and the entire cached suffix is already dead.

        The mutation gates multiply their bust penalty by this scale: content
        below the divergence point re-writes this turn regardless, so mutating
        it is free and only the surviving fraction still carries a penalty.
        Message-count granularity approximates the token split, which is the
        conservative direction only when churn hits size-typical messages;
        callers treat it as an estimate, not an exact token ratio.

        ``head_fingerprint`` covers the bytes ahead of every message in the
        cache prefix (system + tools). Those bytes are not messages, so message
        comparison cannot see them change, yet a single added tool schema
        invalidates the entire transcript behind it. Measured at 14% of all
        billed write, and it silently suppressed the free-rebase path: the turn
        was a total bust that reported as a warm prefix, so queued compression
        never flushed on the one turn where rewriting costs nothing. When the
        fingerprint moves, the surviving fraction is 0.0 by construction.

        Call BEFORE :meth:`update_from_response` (which overwrites
        ``_last_original_messages``), once per request. A real divergence
        (``k < n``) is also recorded into the churn-depth ring consumed by
        anchor placement.
        """
        previous_head = self._last_head_fingerprint
        if head_fingerprint is not None:
            self._last_head_fingerprint = head_fingerprint

        if (
            head_fingerprint is not None
            and previous_head is not None
            and previous_head != head_fingerprint
        ):
            # Checked ahead of the no-history guard: the head sits in front of
            # every message, so it busts the prefix whether or not this tracker
            # recorded last turn's originals. Nothing behind it survives, so
            # every message is already being rewritten and mutating them is free.
            self._churn_depth_fractions.append(0.0)
            return 0.0

        prev = self._last_original_messages
        if not prev:
            return 1.0

        n = len(prev)
        limit = min(n, len(current_original_messages))
        k = 0
        while k < limit and _canonicalize_for_prefix_compare(
            current_original_messages[k]
        ) == _canonicalize_for_prefix_compare(prev[k]):
            k += 1
        if k >= n:
            return 1.0
        fraction = k / n
        self._churn_depth_fractions.append(fraction)
        return fraction

    @property
    def churn_depth_samples(self) -> list[float]:
        """Recent structural-churn depth fractions (0 = head, 1 = tail)."""
        return list(self._churn_depth_fractions)

    @property
    def placed_anchor_depths(self) -> list[int]:
        """Absolute message indices carrying a live 1h anchor from last turn."""
        return list(self._placed_anchor_depths)

    def record_placed_anchors(self, depths: list[int]) -> None:
        """Remember where 1h anchors were actually attached this turn.

        The caller passes the indices it attached, not the ones it asked for.
        Attachment scans backwards for a usable message, so the two differ, and
        recording the request instead of the result would make next turn read a
        move where none happened.
        """
        self._placed_anchor_depths = sorted({int(d) for d in depths if int(d) > 0})

    def forget_placed_anchors(self) -> None:
        """Drop anchor memory when the prefix those anchors sat on is gone.

        After a rebase or a full bust the old indices name bytes that are no
        longer in the cache, so keeping them would price a move against an
        entry that cannot be read back anyway.
        """
        self._placed_anchor_depths = []

    def survival_p_alive(
        self, ttl_seconds: float, linear_fallback: float
    ) -> float:
        """P(the prefix written now is read again before its TTL lapses).

        Empirical hazard estimate from this session's observed inter-turn gaps:
        the fraction of recent gaps that fit inside ``ttl_seconds``. The linear
        ``1 - idle/ttl`` proxy this replaces predicts the NEXT gap from the
        CURRENT idle, which mis-prices heavy-tailed cadences (a bursty session
        looks half-dead at 4 minutes idle when its gap history says the next
        request lands in seconds). With fewer than two samples the fallback is
        returned unchanged, and the blend weight ``n / (n + 4)`` walks from the
        fallback toward the empirical rate as evidence accumulates, so a single
        outlier gap cannot swing the estimate.
        """
        gaps = self._turn_gaps
        n = len(gaps)
        if n < 2 or ttl_seconds <= 0:
            return min(max(linear_fallback, 0.0), 1.0)
        empirical = sum(1 for g in gaps if g <= ttl_seconds) / n
        weight = n / (n + 4.0)
        blended = weight * empirical + (1.0 - weight) * linear_fallback
        return min(max(blended, 0.0), 1.0)

    def expected_reads_within_ttl(
        self, ttl_seconds: float, fallback: float
    ) -> float:
        """Forecast of future same-session reads before a TTL-lapsing gap.

        The net-mutation break-even amortizes a bust over expected future
        reads ``R``. A fixed ``R`` misprices both extremes: a rapid-fire
        session amortizes far more reads than a constant admits, a sporadic
        one far fewer. With ``p`` the observed fraction of this session's
        gaps that fit inside ``ttl_seconds``, the expected run of consecutive
        within-TTL turns ahead is the geometric run length ``p / (1 - p)``.
        ``p`` is capped at 0.95 (a 19-read forecast) so a streak of quick
        turns cannot promise an unbounded amortization horizon. Same blend
        discipline as :meth:`survival_p_alive`: with fewer than two gaps the
        fallback is returned unchanged, then evidence weight ``n / (n + 4)``
        walks toward the forecast.
        """
        gaps = self._turn_gaps
        n = len(gaps)
        if n < 2 or ttl_seconds <= 0:
            return max(fallback, 0.0)
        p = min(sum(1 for g in gaps if g <= ttl_seconds) / n, 0.95)
        forecast = p / (1.0 - p)
        weight = n / (n + 4.0)
        return max(weight * forecast + (1.0 - weight) * max(fallback, 0.0), 0.0)

    def expected_session_reads(self, ttl_seconds: float, fallback: float) -> float:
        """Forecast reads for an edit whose saving outlives the current cache run.

        :meth:`expected_reads_within_ttl` answers "how many turns until this
        cache lapses", which is the right horizon for a mutation whose value
        dies with the cache. Masking is not such a mutation. Once a result is
        masked it stays masked for the rest of the session, so it keeps paying
        on every later turn, and on the turn after a lapse it saves a 1.25x
        rewrite rather than a 0.1x read. Pricing it over a single TTL run caps
        R at 19 and declines batches that pay back several times over.

        Remaining turns are estimated Lindy style, a session that has run n
        turns is expected to run about n more, bounded by
        ``_SESSION_READ_HORIZON_CAP``. The within-TTL forecast is the floor, so
        this is never more optimistic than refusing to look past the cache and
        never more conservative than the estimate it replaces.
        """
        run = self.expected_reads_within_ttl(ttl_seconds, fallback)
        turns = self.turn_number()
        if turns < _SESSION_READ_MIN_TURNS:
            return run
        return max(run, min(float(turns), _SESSION_READ_HORIZON_CAP))

    @property
    def compress_latched(self) -> bool:
        """Whether the cost gate has already committed this session to compress."""
        return self._compress_latched

    def latch_compress(self) -> None:
        """Commit this session to compression for the rest of its life.

        The break-even inputs (expected reads, context pressure) only grow, so a
        session that crosses into "compress" never economically returns to
        "forward original". Latching makes that explicit and prevents an
        oscillation that would bust the compressed cache.
        """
        self._compress_latched = True

    def note_compression(self, tokens_before: int, tokens_after: int) -> None:
        """Record the fraction of tokens the last compression kept.

        Feeds the exponentially-weighted predictor ``recent_compression_ratio``
        reads, so the next turn's break-even can estimate its saving without
        first running compression. Ignores non-positive or inflating results
        (nothing learned from them). The first accepted sample seeds the EWMA
        directly, so a single-observation session reports that sample exactly.
        """
        # ``math.isfinite`` raises OverflowError on a Python int too large for a
        # float (e.g. a corrupted or adversarial counter), so guard it like
        # ``record_turn_gap`` does: an oversized value is just another invalid
        # sample to ignore, never a crash on the request path.
        try:
            valid = (
                math.isfinite(tokens_before)
                and math.isfinite(tokens_after)
                and tokens_before > 0
                and 0 < tokens_after <= tokens_before
            )
        except (TypeError, OverflowError):
            return
        if valid:
            sample = tokens_after / tokens_before
            self._last_compression_kept = sample
            if self._kept_ewma is None:
                self._kept_ewma = sample
                self._kept_var = 0.0
            else:
                prev = self._kept_ewma
                self._kept_ewma = _KEPT_EWMA_ALPHA * sample + (1.0 - _KEPT_EWMA_ALPHA) * prev
                dev = sample - prev
                self._kept_var = (
                    _KEPT_EWMA_ALPHA * (dev * dev) + (1.0 - _KEPT_EWMA_ALPHA) * self._kept_var
                )

    def recent_compression_ratio(self, default: float = 0.8) -> float:
        """Fraction of tokens compression is expected to KEEP (after/before).

        Returns the exponentially-weighted estimate of the kept fraction, or
        ``default`` before any compression has run. ``1 - ratio`` is the mean
        saving fraction; the cost gate uses ``conservative_compression_ratio``
        for the actual decision so a noisy estimate does not over-commit.
        """
        return self._kept_ewma if self._kept_ewma is not None else default

    def compression_ratio_stddev(self) -> float:
        """Spread of the kept-fraction estimate (root of the EWMA variance).

        Zero before the second observation. The cost gate widens its saving
        estimate downward by this much per unit of confidence ``k``.
        """
        return math.sqrt(self._kept_var) if self._kept_var > 0.0 else 0.0

    def conservative_compression_ratio(self, *, default: float = 0.8, k: float = 1.0) -> float:
        """Confidence-discounted KEPT fraction for the irreversible latch.

        The gate commits to compression before it can measure the real saving,
        and the commit is one-way (forwarding original again busts the
        compressed cache). So instead of the mean estimate it uses a lower
        confidence bound on the SAVING: assume compression keeps
        ``mean + k * stddev`` of the tokens, capped at 1.0. A high-variance
        session therefore under-estimates its own saving and waits for the
        estimate to settle before latching, while a session with a stable
        ratio behaves like the plain mean. This is the UCB-style pessimism
        that pairs with the EWMA smoothing.
        """
        if self._kept_ewma is None:
            return default
        # The discount is one-directional: it may only assume LESS saving than
        # the mean, never more. A negative k (e.g. a mis-set env override) would
        # invert that and make the irreversible latch MORE aggressive, so clamp
        # it out. NaN k collapses to 0 (no discount) for the same reason.
        if not math.isfinite(k) or k < 0.0:
            k = 0.0
        return min(1.0, self._kept_ewma + k * self.compression_ratio_stddev())

    def cached_token_count(self) -> int:
        """Tokens the provider currently has cached for this session's prefix.

        This is the suffix a fresh compression would invalidate, i.e. the S term
        in the net-cost break-even.
        """
        return self._cached_token_count

    def turn_number(self) -> int:
        """Turns seen so far, a proxy for expected remaining reads of the cache."""
        return self._turn_number

    def update_from_response(
        self,
        cache_read_tokens: int,
        cache_write_tokens: int,
        messages: list[dict[str, Any]],
        message_token_counts: list[int] | None = None,
        original_messages: list[dict[str, Any]] | None = None,
        sent_message_count: int | None = None,
    ) -> None:
        """Update tracker with cache metrics from the API response.

        Called after every API call. Computes how many messages to freeze
        on the next turn based on the cache_read_tokens reported.

        Args:
            cache_read_tokens: Tokens read from cache (cache hit portion).
            cache_write_tokens: Tokens written to cache (new cache entries).
            messages: The messages that were sent to the API.
            message_token_counts: Pre-computed token counts per message.
                If None, estimates from content length.
        """
        self._last_activity = time.time()
        self._turn_number += 1
        if original_messages is None:
            logger.warning(
                "PrefixCacheTracker[%s]: update_from_response called without "
                "original_messages — falling back to forwarded messages as "
                "originals, which busts the overlay's append-only cache-safety "
                "check on the next turn.",
                self.provider,
            )
        self._last_original_messages = copy.deepcopy(original_messages or messages)
        self._last_forwarded_messages = copy.deepcopy(messages)
        # Callers pass the projected next-turn state: what we forwarded, plus the
        # assistant reply reconstructed from the response. That projection is
        # right for freezing, but it is not a record of bytes any provider
        # hashed. The reconstruction never byte-matches the client's own echo of
        # the same reply, so anything comparing against it has to stop at the
        # boundary between what was sent and what was projected.
        self._last_sent_message_count = (
            len(messages) if sent_message_count is None else max(0, sent_message_count)
        )

        # Steady-state write rate feeds the TTL break-even in recommended_ttl.
        # Only cheap incremental writes belong in it: a full re-write is the
        # cost being avoided, so folding it in would inflate the rate and argue
        # against the very tier that prevents it.
        if 0 < cache_write_tokens <= _STEADY_WRITE_CEILING and cache_read_tokens > 0:
            if self._steady_write_tokens is None:
                self._steady_write_tokens = float(cache_write_tokens)
            else:
                self._steady_write_tokens = (
                    0.7 * self._steady_write_tokens + 0.3 * cache_write_tokens
                )

        # Compute total cached tokens (read + write = what's in cache now)
        total_cached = cache_read_tokens + cache_write_tokens

        if total_cached == 0:
            self._cached_token_count = 0
            self._cached_message_count = 0
            return

        # Estimate per-message token counts if not provided
        if message_token_counts is None:
            message_token_counts = self._estimate_message_tokens(messages)

        # Walk messages from the start, accumulating tokens until we exceed
        # the cached amount. All messages within the cached prefix are frozen.
        accumulated = 0
        frozen_count = 0
        for i, tok_count in enumerate(message_token_counts):
            accumulated += tok_count
            if accumulated <= total_cached:
                frozen_count = i + 1
            else:
                break

        self._cached_token_count = total_cached
        self._cached_message_count = frozen_count

        logger.debug(
            "PrefixCacheTracker[%s]: turn=%d, cached=%d tokens, "
            "frozen=%d/%d messages (read=%d, write=%d)",
            self.provider,
            self._turn_number,
            total_cached,
            frozen_count,
            len(messages),
            cache_read_tokens,
            cache_write_tokens,
        )

    def get_last_original_messages(self) -> list[dict[str, Any]]:
        """Returns internal state — treat as immutable; do not mutate messages or blocks."""
        return self._last_original_messages

    def get_last_forwarded_messages(self) -> list[dict[str, Any]]:
        """Returns internal state — treat as immutable; do not mutate messages or blocks."""
        return self._last_forwarded_messages

    def record_returned(
        self,
        original_messages: list[dict[str, Any]],
        returned_messages: list[dict[str, Any]],
    ) -> None:
        """Record the compressed form handed back to a compress-only caller.

        Sidecar mode (session-aware ``/v1/compress``): Headroom does not
        forward upstream, but whatever it RETURNS is what the caller forwards
        — the same fact ``update_from_response`` records in proxy mode, just
        captured at return time instead of send time. Only the transcript
        snapshots and the activity clock move here; frozen-prefix counts are
        left untouched because no provider response has confirmed anything
        yet — they advance when the caller relays usage via ``/v1/usage``
        (``update_from_response``), or stay at their conservative local value.
        """
        self._last_activity = time.time()
        self._last_original_messages = copy.deepcopy(original_messages)
        self._last_forwarded_messages = copy.deepcopy(returned_messages)

    def resolved_cache_ttl_seconds(self) -> int:
        """Effective prompt-cache lifetime for this session's provider."""
        if self.config.cache_ttl_seconds is not None:
            return self.config.cache_ttl_seconds
        return _PROVIDER_CACHE_TTL_SECONDS.get(self.provider, 300)

    def classify_cache_miss(
        self,
        cache_read_tokens: int,
        current_forwarded_messages: list[dict[str, Any]],
        idle_seconds: float | None = None,
        segments: Mapping[str, Any] | None = None,
    ) -> CacheMissAttribution:
        """Attribute *this turn's* cache outcome: hit, TTL lapse, or prefix change.

        Call this BEFORE :meth:`update_from_response` — it reads the state
        captured from the *previous* turn (``_cached_token_count``,
        ``_last_forwarded_messages``, ``_last_activity``), all of which
        ``update_from_response`` overwrites.

        Attribution only fires when the previous turn left a cacheable prefix
        (``_cached_token_count > 0``); the very first warm turn has nothing to
        miss against, so it is reported as ``cold_start`` with ``is_miss=False``.

        When a hit was expected but ``cache_read_tokens == 0``:

        * If the idle gap since the last turn exceeded the provider cache TTL,
          the cache entry had already lapsed — ``ttl_expiry``. **TTL wins ties:**
          once the entry expired, a coincident prefix change is moot (the issue
          asks "should I move 5m → 1h?", which only the TTL signal answers).
        * Otherwise, if the forwarded prefix changed versus last turn, the new
          bytes couldn't match the cached prefix — ``prefix_change``.
        * If neither signal fires (stable prefix, within TTL) we can't explain
          it from local state — ``unknown`` (e.g. provider-side eviction).

        ``segments`` optionally names the non-message parts of the cache key so a
        ``prefix_change`` can say WHICH part moved. Pass an ordered mapping of
        ``{segment_name: forwarded_value}`` **outermost-first** — i.e. in the
        order the provider hashes them, so the first entry that changed is the one
        that invalidated everything after it. Names are the caller's to choose
        (see the ``SEGMENT_*`` constants for the well-known ones and the
        per-provider shapes); this method never inspects a request body itself.
        Pass the values as forwarded upstream, not as received from the client —
        those are the bytes the provider actually hashed, so Headroom's own
        mutations are correctly attributed to Headroom.

        Omitting ``segments`` keeps the previous behavior exactly: a prefix change
        is still detected from the forwarded messages, just reported without a
        segment name.

        A partial read (``0 < cache_read_tokens``) counts as a hit here; the
        existing model-aware bust detection in PrometheusMetrics already covers
        partial-invalidation accounting, and double-counting it as a "miss"
        would muddy the 5m-vs-1h signal this method exists to provide.

        Returns a :class:`CacheMissAttribution`; ``is_miss`` is False for hits
        and cold starts.
        """
        if idle_seconds is None:
            idle_seconds = self.seconds_since_activity()
        ttl = self.resolved_cache_ttl_seconds()
        expected = self._cached_token_count
        # Always roll the segment hashes forward, hit or miss, so the next turn
        # compares against what we actually forwarded this turn.
        changed_segment = self._diff_segments(segments)

        # Nothing was cached last turn → cold start, not a miss.
        if expected <= 0:
            return CacheMissAttribution(
                is_miss=False,
                reason=MISS_COLD_START,
                idle_seconds=idle_seconds,
                cache_ttl_seconds=ttl,
                expected_cached_tokens=expected,
                cache_read_tokens=cache_read_tokens,
            )

        # We expected a hit. A non-zero read means the prefix cache worked.
        if cache_read_tokens > 0:
            return CacheMissAttribution(
                is_miss=False,
                reason="hit",
                idle_seconds=idle_seconds,
                cache_ttl_seconds=ttl,
                expected_cached_tokens=expected,
                cache_read_tokens=cache_read_tokens,
            )

        # Full miss on a prefix we expected cached. Attribute it.
        ttl_exceeded = idle_seconds > ttl
        # The cache key is the WHOLE prefix, not just the message list: a changed
        # tool schema or system prompt invalidates it while `messages` stays
        # byte-identical. Checking only the messages filed every tools/system
        # change under `unknown` — and tools render at position 0, so that is the
        # class that invalidates the most and was the least explained.
        messages_changed = not self._forwarded_prefix_stable(current_forwarded_messages)
        prefix_changed = messages_changed or changed_segment is not None

        if ttl_exceeded:
            reason = MISS_TTL_EXPIRY  # TTL wins ties (see docstring)
        elif prefix_changed:
            reason = MISS_PREFIX_CHANGE
        else:
            reason = MISS_UNKNOWN

        if reason == MISS_PREFIX_CHANGE:
            # A named segment moved -> report it. Otherwise the forwarded history
            # itself diverged, which is the messages segment by elimination.
            changed_segment = changed_segment or SEGMENT_MESSAGES
        else:
            # On a TTL lapse the entry was already gone, so a coincident content
            # change didn't cause the miss — naming it sends the reader after the
            # wrong fix. `unknown` has nothing to name by definition.
            changed_segment = None

        return CacheMissAttribution(
            is_miss=True,
            reason=reason,
            idle_seconds=idle_seconds,
            cache_ttl_seconds=ttl,
            expected_cached_tokens=expected,
            cache_read_tokens=cache_read_tokens,
            prefix_changed=prefix_changed,
            ttl_exceeded=ttl_exceeded,
            changed_segment=changed_segment,
        )

    def _diff_segments(self, segments: Mapping[str, Any] | None) -> str | None:
        """Roll segment hashes forward; return the first changed segment's name.

        ``segments`` is ordered outermost-first, so the first entry whose hash
        moved is the one that invalidated everything after it — later changes are
        consequences, not causes, and reporting them would point at the wrong fix.

        Returns ``None`` when nothing changed, when no segments were supplied, or
        on the first turn a segment is seen (there is no baseline to compare
        against, so we cannot claim it changed).

        A call that supplies NO segments DROPS any stored baseline. Hashes only
        roll forward on calls that pass them, so a session mixing segment-aware
        and segment-blind calls would otherwise compare against a baseline from
        several turns ago and report a change on the wrong turn — right culprit,
        wrong moment. "I cannot tell" is the honest answer there, so we forget
        instead of guessing. Sessions that always pass segments never clear, and
        providers that never pass them have nothing to clear.
        """
        if not segments:
            self._last_segment_hashes.clear()
            return None
        first_changed: str | None = None
        for name, value in segments.items():
            fingerprint = segment_fingerprint(value)
            previous = self._last_segment_hashes.get(name)
            self._last_segment_hashes[name] = fingerprint
            if previous is not None and previous != fingerprint and first_changed is None:
                first_changed = name
        return first_changed

    def _forwarded_prefix_stable(self, current_forwarded_messages: list[dict[str, Any]]) -> bool:
        """True if last turn's forwarded prefix is still an exact prefix of this turn's.

        The cached prefix is whatever we forwarded last turn. If those exact
        messages still lead the current forwarded list, the bytes the provider
        hashed for its cache key are unchanged, so a miss can't be blamed on
        content. Anything else (a frozen message rewritten, the prefix
        reordered, the list now shorter) counts as a prefix change.
        """
        prev = self._last_forwarded_messages
        if not prev:
            # No recorded prefix to compare — can't claim it changed.
            return True
        if len(current_forwarded_messages) < len(prev):
            return False
        return current_forwarded_messages[: len(prev)] == prev

    def forwarded_prefix_will_change(
        self, current_forwarded_messages: list[dict[str, Any]] | None
    ) -> bool:
        """Whether this turn's body already invalidates the provider's cached prefix.

        For the gates that price a rewrite before the request goes out. On a
        turn that is busting anyway, a structural mutation is free, so this
        decides whether they get that discount.

        Two things this deliberately is not. It is not
        ``body_mutation_tracker.mutated``: that flag says a transform touched
        the body, which only turns off byte-faithful forwarding, and a
        re-serialized body with an unchanged lead still hits the cache. And it
        is not ``_forwarded_prefix_stable``, which compares the whole previous
        forwarded list. Last turn's uncached tail is this turn's mid-prefix, and
        recompressing it fails that compare while the region the provider
        actually cached is untouched. Measured 2026-07-25: that comparison
        reported a bust on every turn of a lineage billing 97 to 100% cache
        hits.

        What the provider cached is ``_cached_message_count`` messages, sized
        from the read it billed us, and the comparable region stops where the
        recorded list stops being a record of what we sent.

        Depth is the last thing that matters, and the reason this returns a
        judgement rather than a raw comparison. A prefix match is a longest
        common prefix, not all or nothing. Recompressing the final cached
        message rewrites twelve characters and leaves the other 99% of the
        prefix readable, which is nothing like the rewrite a caller gets a
        discount for. Measured 2026-07-25: divergence sat at message 76 of 77
        on turn after turn while the same lineage billed near-total hits. So a
        rewrite counts as already paid only once most of the cached prefix is
        going down with it.
        """
        if not isinstance(current_forwarded_messages, list):
            return False
        cached_count = min(
            self._cached_message_count,
            self._last_sent_message_count or self._cached_message_count,
        )
        if cached_count <= 0:
            # Nothing cached yet, so there is no prefix to invalidate. A cold
            # lineage pays its first write regardless of what we do here.
            return False
        prev = self._last_forwarded_messages
        if len(current_forwarded_messages) < cached_count:
            # This turn sends fewer messages than were cached, so history was
            # rewritten from underneath us. Compaction looks exactly like this.
            logger.info(
                "PREFIX_DIVERGE truncated: cached=%d now=%d",
                cached_count,
                len(current_forwarded_messages),
            )
            return True
        if len(prev) < cached_count:
            # No record to compare against, which is not evidence of a bust.
            # Callers get no discount rather than a guessed one.
            return False
        # The breakpoint slides forward every turn, so the same cached message
        # carries cache_control on one turn and not the next. That is a
        # directive about where to place a marker, not content, and comparing it
        # reports a bust on a prefix the provider is still reading happily.
        now = _strip_cache_control(current_forwarded_messages[:cached_count])
        was = _strip_cache_control(prev[:cached_count])
        if now == was:
            return False
        diverge_at = next(
            (i for i, (a, b) in enumerate(zip(now, was, strict=False)) if a != b),
            len(now),
        )
        counts = self._estimate_message_tokens(was)
        cached_tokens = sum(counts) or 1
        surviving = sum(counts[:diverge_at]) / cached_tokens
        already_busting = surviving <= ALREADY_BUSTING_SURVIVING_FRACTION
        # A bust report with no depth is not actionable, and this only fires on
        # turns whose prefix moved at all, so it stays at INFO. If it ever
        # becomes chatty that is itself the finding.
        logger.info(
            "PREFIX_DIVERGE at %d/%d surviving=%.3f busting=%s",
            diverge_at,
            cached_count,
            surviving,
            already_busting,
        )
        return already_busting

    def record_bust_avoided(self, tokens_preserved: int, compression_foregone: int) -> None:
        """Record when we chose to preserve cache over compressing."""
        self._busts_avoided += 1
        self._tokens_preserved += tokens_preserved
        self._compression_foregone_tokens += compression_foregone

    @property
    def is_expired(self) -> bool:
        """Check if this tracker has been idle beyond TTL."""
        return (time.time() - self._last_activity) > self.config.session_ttl_seconds

    def seconds_since_activity(self) -> float:
        """Wall-clock seconds since this tracker last saw activity.

        #856 P3b feeds this to the net-cost gate as an idle signal: as it
        approaches the provider's prompt-cache TTL (~300s for Anthropic),
        P_alive decays toward 0 and deep edits near cache lapse become free.
        Distinct from :attr:`is_expired`, which uses the much longer
        session-tracker *cleanup* TTL (``session_ttl_seconds``), not the cache
        TTL.

        Wiring caveat: ``SessionTrackerStore.get_or_create`` refreshes
        ``_last_activity`` on access, so a caller that wants the idle gap
        since the *previous turn's response* must read this before fetching
        the tracker for the current request (or the store must capture it at
        fetch time). ``update_from_response`` is the per-turn activity stamp.
        """
        return max(0.0, time.time() - self._last_activity)

    @property
    def stats(self) -> FreezeStats:
        """Return stats for dashboard/metrics."""
        return FreezeStats(
            busts_avoided=self._busts_avoided,
            tokens_preserved=self._tokens_preserved,
            compression_foregone_tokens=self._compression_foregone_tokens,
            net_benefit_tokens=self._tokens_preserved - self._compression_foregone_tokens,
            frozen_message_count=self._cached_message_count,
            turn_number=self._turn_number,
        )

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, Any]]) -> list[int]:
        """Rough token count per message (chars / 3.5).

        Counts text, tool_result content, and tool_use input fields
        for accurate Anthropic-format estimation.
        """
        counts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                chars = len(content)
            elif isinstance(content, list):
                chars = 0
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")
                    if block_type == "text":
                        chars += len(block.get("text", ""))
                    elif block_type == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, str):
                            chars += len(inner)
                        elif isinstance(inner, list):
                            chars += sum(
                                len(b.get("text", "")) for b in inner if isinstance(b, dict)
                            )
                    elif block_type == "tool_use":
                        inp = block.get("input")
                        if isinstance(inp, str):
                            chars += len(inp)
                        elif isinstance(inp, dict):
                            chars += len(json.dumps(inp, separators=(",", ":")))
                    else:
                        text = block.get("text", "")
                        if text:
                            chars += len(text)
            else:
                chars = 0
            # OpenAI function-calling: the assistant's command lives in the
            # top-level `tool_calls` (or legacy `function_call`) field, NOT in
            # `content` (which is empty/None on a tool-call turn). Anthropic puts
            # the equivalent in a `tool_use` content BLOCK (counted above), but
            # the OpenAI shape was never counted here. That under-counted every
            # tool-based assistant turn to ~0, so the frozen-prefix estimate
            # overshot the real cache boundary and froze the NEWEST delta — which
            # is why OpenAI/Kimi (fireworks) tool harnesses got ~zero compression
            # while text/back-tick harnesses (command in `content`) compressed.
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    chars += len(str(fn.get("name", ""))) + len(str(fn.get("arguments", "")))
            fc = msg.get("function_call")
            if isinstance(fc, dict):
                chars += len(str(fc.get("name", ""))) + len(str(fc.get("arguments", "")))
            # Add overhead for role, block structure, etc.
            chars += 20
            counts.append(max(1, int(chars / 3.5)))
        return counts


def _lineage_snapshot(obj: Any) -> Any:
    """Structural copy of a canonical projection for lineage-chain storage.

    Copies dict/list structure (immutable leaves are shared, so this costs
    structure, not bytes) to isolate the stored chain from downstream in-place
    mutation of the request, and normalizes NaN to a sentinel — ``json.loads``
    accepts bare ``NaN`` and ``NaN != NaN``, so a byte-identical resend would
    otherwise read as a rewritten history on every turn. Values inside opaque
    tool payloads are walked for copying only; no keys are dropped (see
    ``_OPAQUE_PAYLOAD_KEYS``).
    """
    if isinstance(obj, dict):
        return {k: _lineage_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_lineage_snapshot(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return "\x00nan"
    return obj


def classify_append_only_prefix(
    current_messages: list[dict[str, Any]],
    previous_messages: list[dict[str, Any]],
) -> AppendOnlyClassification | None:
    """Return the shared append-only classification for two message histories."""
    if not current_messages or not previous_messages:
        return None
    current = _lineage_snapshot(_canonicalize_for_prefix_compare(current_messages))
    previous = _lineage_snapshot(_canonicalize_for_prefix_compare(previous_messages))
    # Callers slice raw message lists by raw counts and index them by the
    # frontier's message index, so the canonical projection has to stay
    # positionally 1:1 with its input over the region they address. It does not
    # when a whole message projects to ``{}`` and the list comprehension in
    # ``_canonicalize_for_prefix_compare`` drops it, which happens when every key
    # on that message is non-semantic. Refuse rather than hand back an index into
    # a list the caller does not have.
    #
    # Only the compared prefix matters. A dropped message past it cannot shift
    # any index the callers use, and the delta they take is the raw tail, so
    # refusing there would give up a sound replay for nothing.
    prefix_len = len(previous_messages)
    if len(previous) != prefix_len:
        return None
    if len(_canonicalize_for_prefix_compare(current_messages[:prefix_len])) != prefix_len:
        return None
    return _classify_append_only_canonical(current, previous)


class SessionTrackerStore:
    """Manages PrefixCacheTracker instances across sessions.

    Keyed by session ID (from x-headroom-session-id header or computed hash).
    Within one session id, ``resolve_tracker`` keys trackers by conversation
    lineage so concurrent conversations sharing a fallback id do not thrash
    one tracker's frozen-prefix state (#2085).
    Automatically cleans up expired sessions.
    """

    def __init__(self, default_config: PrefixFreezeConfig | None = None):
        self._trackers: dict[str, PrefixCacheTracker] = {}
        self._default_config = default_config or PrefixFreezeConfig()
        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 60.0  # Cleanup every 60s
        # Conversation lineages per session id: tracker key -> snapshot of the
        # canonicalized messages of the last request that lineage served. The
        # first lineage lives under the bare session id (single-conversation
        # sessions behave exactly as before); later lineages get unique
        # "<session_id>\x00<n>" keys — NUL cannot appear in an HTTP header
        # value, so a synthetic key can never collide with a client-supplied
        # x-headroom-session-id.
        self._lineages: dict[str, OrderedDict[str, list[Any]]] = {}
        # Exact non-message cache-key affinity per tracker.  Anthropic renders
        # tools before system/messages, so two sub-calls with identical history
        # but different tool profiles must never share frozen-prefix state.
        self._lineage_affinities: dict[str, str | None] = {}
        self._lineage_counter = itertools.count(1)

    def peek(self, session_id: str) -> PrefixCacheTracker | None:
        """Return the live tracker for ``session_id``, else None.

        Never creates: lookup paths that must not leave a footprint (e.g. the
        ``/v1/usage`` unknown-session check, where ``get_or_create`` would let
        a flood of novel ids grow the store unboundedly within each TTL
        window) use this instead of :meth:`get_or_create`.

        A TTL-expired-but-unswept tracker answers None too: the sweep runs
        lazily from get_or_create at 60s granularity, so without this check an
        expired session would keep answering with stale pre-expiry state — and
        a caller that then touched it (``update_from_response`` stamps
        ``_last_activity``) would resurrect the dead tracker indefinitely,
        making the documented 404-on-expired contract nondeterministic.
        """
        tracker = self._trackers.get(session_id)
        if tracker is None or tracker.is_expired:
            return None
        return tracker

    def get_or_create(self, session_id: str, provider: str) -> PrefixCacheTracker:
        """Get existing tracker or create a new one for this session."""
        self._maybe_cleanup()

        if session_id in self._trackers:
            tracker = self._trackers[session_id]
            # Snapshot idle-since-last-response BEFORE bumping the access clock,
            # so the net-cost/TTL gate sees the true gap (see the attribute's
            # docstring in PrefixCacheTracker.__init__).
            tracker._idle_seconds_at_fetch = max(0.0, time.time() - tracker._last_activity)
            tracker._last_activity = time.time()
            return tracker

        tracker = PrefixCacheTracker(provider, self._default_config)
        tracker._idle_seconds_at_fetch = 0.0  # cold start: nothing cached to lapse
        self._trackers[session_id] = tracker
        return tracker

    def _rematch_partial_lineage(
        self,
        family: dict[str, list[str]],
        snap: list[str],
    ) -> str | None:
        """Deepest chain that agrees with `snap` for most of its length.

        Returns None when no chain agrees closely enough, which keeps the
        caller on its existing new-lineage path.

        Both guards matter, and they catch different things. Two sibling
        requests under one session id share a short head and then diverge for
        good, and merging those is the cross-contamination this store exists to
        prevent. They overlap by one to three messages, so the absolute floor
        rejects them. Client compaction agrees on almost nothing relative to the
        chain it replaces, so the fraction rejects that. A client that edits one
        old message clears both.
        """
        floor = self._default_config.lineage_rematch_min_messages
        if floor <= 0:
            return None
        threshold = self._default_config.lineage_rematch_fraction

        best_key: str | None = None
        best_common = 0
        deepest_common = 0
        deepest_len = 0
        for key, chain in family.items():
            if not chain:
                continue
            common = 0
            for recorded, incoming in zip(chain, snap):
                if recorded != incoming:
                    break
                common += 1
            if common > deepest_common:
                deepest_common, deepest_len = common, len(chain)
            if common <= best_common or common < floor:
                continue
            if common < threshold * len(chain):
                continue
            best_key, best_common = key, common

        if best_key is None and deepest_len:
            # A rejection here discards the whole recorded prefix, so record how
            # close it came. An early divergence means the provider cache was
            # dead regardless and starting fresh costs nothing. A late one means
            # a guard threw away bytes that were still worth replaying, which is
            # the case worth retuning for.
            logger.info(
                "LINEAGE_REJECT: deepest_common=%d chain_len=%d incoming=%d "
                "fraction=%.3f floor=%d threshold=%.2f",
                deepest_common,
                deepest_len,
                len(snap),
                deepest_common / deepest_len if deepest_len else 0.0,
                floor,
                threshold,
            )
        return best_key

    def resolve_tracker(
        self,
        session_id: str,
        provider: str,
        messages: list[dict[str, Any]] | None = None,
        cache_affinity: str | None = None,
    ) -> PrefixCacheTracker:
        """Resolve the tracker for THIS conversation within a session id (#2085).

        Concurrent conversations often share a fallback session id — same
        model + system prompt covers a Claude Code session together with its
        parallel subagents, or several sessions in one workspace. On a shared
        tracker their interleaved histories cross-contaminate the
        frozen-prefix state: freeze never stabilizes and the provider prompt
        cache is re-written on nearly every call.

        Lineage resolution keys trackers by conversation content instead:
        reuse the tracker whose previous request messages are a prefix of the
        incoming history. It also recognizes a conservative block-level shape
        where a large leading run and two-block identity suffix survive while
        the middle tail is regenerated; all other rewrites start a fresh
        lineage. This keeps #2671's stable cache boundary attached without
        merging unrelated parallel sub-calls.
        Byte-identical histories (templated fan-outs before they diverge)
        intentionally share a tracker: their provider cache line is identical
        too, so sharing is harmless.

        The session id itself is never altered, so session-sticky state keyed
        on it elsewhere (beta headers, CCR/memory registries, the compression
        cache) is unaffected.

        Args:
            session_id: Session identity from :meth:`compute_session_id`.
            provider: Provider name for a newly created tracker.
            messages: The request's original client messages, as captured
                right after the INPUT_RECEIVED pipeline extension and before
                security-scan/hook/image-compression mutation — the same
                snapshot ``update_from_response`` records, so the chain
                compares like against like across turns. ``None``/empty
                (legacy callers, stub stores in tests) falls back to plain
                :meth:`get_or_create`.
            cache_affinity: Stable fingerprint of the provider's non-message
                cache-key segments (model/tools/tool choice/thinking). Lineages
                with different affinity never share a tracker.

        Returns:
            The ``PrefixCacheTracker`` for this conversation's lineage.
        """
        if not messages or not self._default_config.enabled:
            # No lineage signal, or prefix freeze is disabled (there is no
            # frozen state to protect): legacy one-tracker-per-session-id.
            return self.get_or_create(session_id, provider)

        # Prune expired trackers BEFORE matching, so a dead lineage cannot win
        # the match. This also arms the cleanup interval: the get_or_create
        # calls below cannot re-trigger a prune mid-function, so the family
        # read here stays attached through the stamp at the end.
        self._maybe_cleanup()

        # The repo's canonical cross-turn equivalence, shared with the
        # cache-stable delta path: a moved cache breakpoint, string<->block
        # content sugar, or per-turn transport annotations must not read as
        # a rewritten history. Object comparison plus one structural snapshot
        # per request (~1ms on a 2MB history — same order as the handler's
        # existing request deepcopy; no serialization or hashing).
        canon = _canonicalize_for_prefix_compare(messages)
        if not canon:
            # Degenerate: every message projected away (pure directive
            # content) — no lineage signal to match on.
            return self.get_or_create(session_id, provider)
        snap = _lineage_snapshot(canon)

        family = self._lineages.setdefault(session_id, OrderedDict())

        # Strict whole-message continuations win first, then pure block appends.
        # Rewritten-tail matches are deliberately last and require a unique best
        # structural score; ambiguity starts a fresh lineage instead of making
        # sibling sub-calls ping-pong one tracker.
        by_length = sorted(family.items(), key=lambda item: len(item[1]), reverse=True)
        best_key: str | None = None
        for accepted in (
            (RELATION_EXACT, RELATION_MESSAGE_APPEND),
            (RELATION_BLOCK_APPEND,),
        ):
            for key, chain in by_length:
                if self._lineage_affinities.get(key) != cache_affinity:
                    continue
                relation = _classify_history_canonical(snap, chain)
                if relation.kind in accepted:
                    best_key = key
                    break
            if best_key is not None:
                break

        if best_key is None:
            rewrite_candidates: list[tuple[tuple[int, int, int], str]] = []
            for key, chain in by_length:
                if self._lineage_affinities.get(key) != cache_affinity:
                    continue
                relation = _classify_history_canonical(snap, chain)
                if relation.kind == RELATION_BLOCK_REWRITE_TAIL:
                    rewrite_candidates.append(
                        (
                            (
                                relation.stable_prefix_blocks,
                                relation.stable_suffix_blocks,
                                relation.previous_block_count,
                            ),
                            key,
                        )
                    )
            rewrite_candidates.sort(reverse=True)
            if rewrite_candidates and (
                len(rewrite_candidates) == 1 or rewrite_candidates[0][0] != rewrite_candidates[1][0]
            ):
                best_key = rewrite_candidates[0][1]

        if best_key is None:
            # No chain prefixes the incoming history at MESSAGE granularity, but
            # a sub-call shape packs its transcript into one block-style message
            # and rewrites that message's blocks each turn. That is the same
            # conversation continuing, and the strict test above cannot see it
            # because the newest message is not equal to the recorded one.
            #
            # Narrower than the strict pass, not merely weaker: the message
            # COUNT must be unchanged, every message before the last must still
            # match exactly, and only the last may differ, by having grown in
            # place. A history that both gained a message and rewrote an older
            # one did not grow in place and starts a fresh lineage. Longest
            # chain first, so an exact continuation is never displaced.
            for key, chain in sorted(
                family.items(), key=lambda kv: len(kv[1]), reverse=True
            ):
                if len(chain) != len(snap) or not chain:
                    continue
                head = len(chain) - 1
                if snap[:head] == chain[:head] and _is_message_continuation(
                    chain[head], snap[head]
                ):
                    best_key = key
                    break

        if best_key is None:
            # No chain prefixes the incoming history, but the client may have
            # edited one old message rather than started a new conversation.
            # Strict matching treated those alike and discarded the whole
            # lineage, including the forwarded bytes for every message BEFORE
            # the edit — the run that overlay_cached_prefix could still have
            # replayed for free. Fall back to the deepest common prefix and
            # keep the lineage when it still covers most of the recorded chain.
            #
            # Replaying past the edit is not a risk here: overlay_cached_prefix
            # re-checks every message against the incoming history and stops at
            # the first divergence, so a partial match can only ever restore the
            # agreeing head.
            best_key = self._rematch_partial_lineage(family, snap)

        if best_key is None:
            cap = self._default_config.max_lineages_per_session
            if len(family) >= cap:
                # Family is full: over-cap conversations share one overflow
                # tracker instead of evicting an established lineage. Any
                # eviction policy degrades EVERY conversation once the
                # working set exceeds the cap (under round-robin the victim
                # is always the conversation about to arrive), while overflow
                # sharing degrades only the over-cap tail — to exactly the
                # pre-lineage shared-tracker behavior — and established
                # lineages keep their state. A cap <= 0 therefore disables
                # lineage splitting entirely.
                overflow_key = f"{session_id}\x00overflow"
                if overflow_key not in self._trackers:
                    logger.info(
                        "SessionTrackerStore: lineage cap %d reached for session %s; "
                        "over-cap conversations share an overflow tracker (raise "
                        "PrefixFreezeConfig.max_lineages_per_session if this "
                        "workspace genuinely runs more concurrent conversations)",
                        cap,
                        session_id,
                    )
                return self.get_or_create(overflow_key, provider)
            if not family:
                # First lineage rides the bare session id; this also adopts a
                # tracker created earlier via plain get_or_create.
                best_key = session_id
            else:
                best_key = f"{session_id}\x00{next(self._lineage_counter)}"

        # get_or_create (not a private fetch) so test stubs that patch the
        # instance method keep intercepting tracker creation; its internal
        # cleanup is interval-gated and was armed above, so it cannot prune
        # the family before the stamp below.
        tracker = self.get_or_create(best_key, provider)
        family[best_key] = snap
        self._lineage_affinities[best_key] = cache_affinity
        return tracker

    _OVERFLOW_SUFFIX = "\x00overflow"

    def export_state(self) -> dict[str, Any]:
        """Plain-data snapshot of every live tracker, for cross-restart reuse.

        The lineage families are exported verbatim. An earlier version rebuilt
        them on load from each tracker's ``_last_original_messages``, on the
        assumption that those are the same messages ``resolve_tracker`` stamps
        into ``family[key]``. Measured against the live proxy that assumption is
        wrong: the rebuilt chain never matched, so every restart minted a fresh
        lineage and rewrote the whole history anyway, which is the exact cost
        this is meant to remove. ``family[key]`` is the object the prefix
        comparison actually runs against, so it is the object that gets saved.

        Expired trackers are dropped here rather than on load, so a snapshot
        never carries state that the in-memory store would already have pruned.
        """
        trackers: dict[str, Any] = {}
        for key, tracker in self._trackers.items():
            if tracker.is_expired:
                continue
            try:
                trackers[key] = tracker.export_state()
            except Exception:  # pragma: no cover - defensive, never fail a save
                logger.debug("SessionTrackerStore: skipping unexportable tracker")

        lineages: dict[str, Any] = {}
        for session_id, family in self._lineages.items():
            kept = {key: snap for key, snap in family.items() if key in trackers}
            if kept:
                lineages[session_id] = kept

        return {"trackers": trackers, "lineages": lineages}

    def restore_state(self, blob: dict[str, Any]) -> int:
        """Rebuild trackers and lineage families from :meth:`export_state`.

        Returns the number of trackers restored. Existing in-memory trackers
        win: this is startup rehydration, not a merge, and a tracker created by
        live traffic reflects the provider's actual cache state better than a
        snapshot does.
        """
        trackers = blob.get("trackers")
        if not isinstance(trackers, dict):
            return 0

        raw_lineages = blob.get("lineages")
        saved_lineages: dict[str, Any] = (
            raw_lineages if isinstance(raw_lineages, dict) else {}
        )

        restored = 0
        highest_lineage = 0
        for key, state in trackers.items():
            if not isinstance(key, str) or not isinstance(state, dict):
                continue
            if key in self._trackers:
                continue
            try:
                tracker = PrefixCacheTracker(
                    str(state.get("provider") or "anthropic"), self._default_config
                )
                tracker.restore_state(state)
                if tracker.is_expired:
                    # Snapshot sat on disk past the cleanup TTL. Dropping it
                    # matches _maybe_cleanup, and matters for correctness as
                    # well as hygiene: a dead lineage must not win a match in
                    # resolve_tracker.
                    continue
                self._trackers[key] = tracker
                restored += 1
            except Exception:  # pragma: no cover - defensive
                logger.debug("SessionTrackerStore: skipping unrestorable tracker")
                continue

            session_id, _, suffix = key.partition("\x00")
            if suffix == "overflow":
                # Over-cap conversations share this tracker without owning a
                # lineage, so it must not be matched against as one.
                continue
            if suffix.isdigit():
                highest_lineage = max(highest_lineage, int(suffix))

            snap = saved_lineages.get(session_id, {}).get(key)
            if snap is None:
                continue
            family = self._lineages.setdefault(session_id, OrderedDict())
            family[key] = snap

        if highest_lineage:
            # Never hand out a synthetic key that a restored tracker already
            # holds: a collision would silently merge two conversations.
            self._lineage_counter = itertools.count(highest_lineage + 1)
        return restored

    def peek_idle_seconds(self, session_id: str) -> float | None:
        """Idle gap for an existing session without refreshing activity.

        ``get_or_create`` stamps ``_last_activity`` on access, so the
        net-cost gate (#856 P3b) must read the gap before fetching the
        tracker for the current request. Returns None for an unknown
        session, in which case the gate keeps its env-constant P_alive.
        """
        tracker = self._trackers.get(session_id)
        if tracker is None:
            return None
        return tracker.seconds_since_activity()

    def peek_tracker(self, session_id: str) -> PrefixCacheTracker | None:
        """An existing tracker without creating one or stamping activity.

        Model routing has to price a switch before the request reaches
        ``resolve_tracker``, because the model chooses the tokenizer that
        compression runs under. Peeking keeps that early read from creating a
        tracker for a session the request may never establish, and from
        moving ``_last_activity`` ahead of the idle-gap read that follows.
        Returns None for an unknown session, where a cold prefix is the right
        assumption anyway.
        """
        return self._trackers.get(session_id)

    def compute_session_id(
        self,
        request: Any,
        model: str,
        messages: list[dict[str, Any]],
        *,
        system: Any | None = None,
    ) -> str:
        """Compute a session ID from the request.

        Priority:
        1. x-headroom-session-id header (explicit)
        2. Hash of (model + system prompt) — stable per conversation

        The system prompt is harvested from the LEADING run of ``role:"system"``
        entries in ``messages`` (everything before the first non-system turn).
        Anthropic carries the system prompt as a top-level ``body["system"]``
        field instead, so its handler prepends that as a synthetic
        ``role:"system"`` message before calling this — otherwise every
        Anthropic conversation on the same model would collapse to one session id
        and their session-sticky state (CCR/memory tools, beta headers, frozen
        prefix) would cross-contaminate.

        Only the leading run counts: agentic clients (Claude Code) interleave
        ``role:"system"`` reminder turns INTO the history as it grows (hook
        output, skills lists, truncation notices). Hashing those would rotate
        the session id mid-conversation, orphaning the prefix tracker and every
        other session-sticky subsystem each time a reminder lands (#2085).
        """
        # Check for explicit session header
        if hasattr(request, "headers"):
            session_header = request.headers.get("x-headroom-session-id")
            if session_header:
                return str(session_header)

        # Fall back to hashing model + the leading system-text run. Anthropic
        # carries its system prompt outside ``messages``, so callers must pass
        # that top-level value or unrelated Claude sessions collapse to a
        # model-only key.
        system_parts: list[str] = []

        def _append_system_content(content: Any) -> None:
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(str(block.get("text", "")))

        _append_system_content(system)
        for msg in messages:
            if msg.get("role") != "system":
                break
            _append_system_content(msg.get("content", ""))

        system_content = json.dumps(system_parts, ensure_ascii=False, separators=(",", ":"))
        key = f"{model}:{system_content}"
        return hashlib.md5(key.encode()).hexdigest()[:16]  # nosec B324

    def _maybe_cleanup(self) -> None:
        """Remove expired trackers periodically."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        expired = [sid for sid, tracker in self._trackers.items() if tracker.is_expired]
        for sid in expired:
            del self._trackers[sid]

        if expired:
            # Keep the lineage index in step with the tracker map.
            for base in list(self._lineages):
                family = self._lineages[base]
                for key in [k for k in family if k not in self._trackers]:
                    del family[key]
                    self._lineage_affinities.pop(key, None)
                if not family:
                    del self._lineages[base]
            logger.debug("SessionTrackerStore: cleaned up %d expired sessions", len(expired))

        self._last_cleanup = now

    @property
    def active_sessions(self) -> int:
        """Number of active session trackers (one per conversation lineage)."""
        return len(self._trackers)


