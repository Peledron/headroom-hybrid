"""Compression Store for CCR (Compress-Cache-Retrieve) architecture.

This module implements reversible compression: when SmartCrusher compresses
tool outputs, the original data is cached here for on-demand retrieval.

Key insight from research: REVERSIBLE compression beats irreversible compression.
If the LLM needs data that was compressed away, it can retrieve it instantly.

Features:
- Thread-safe in-memory storage with TTL expiration
- BM25-based search within cached content
- Retrieval event tracking for feedback loop
- Automatic eviction when capacity is reached

Usage:
    store = get_compression_store()

    # Store compressed content
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_tokens=1000,
        compressed_tokens=100,
        tool_name="search_api",
    )

    # Retrieve later (by hash; always returns the full original content)
    entry = store.retrieve(hash_key)
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats
    from .backends import CompressionStoreBackend

logger = logging.getLogger(__name__)

# A marker stays referenceable for as long as the compressed tool result sits
# in the transcript, which is the whole session, not half an hour. When an
# entry expires early the marker guard cannot expand it and blocks the tool
# call outright, so the client loses work that was never at risk. Capacity
# (``max_entries``, LRU) is the real bound; this TTL only reaps stale sessions.
# It is an idle window, not an absolute lifetime (#2604): the clock restarts on
# every successful retrieve(), so an entry the agent keeps touching survives.
DEFAULT_CCR_TTL_SECONDS = 86400  # 24h idle; override via HEADROOM_CCR_TTL_SECONDS
CCR_TTL_SECONDS_ENV = "HEADROOM_CCR_TTL_SECONDS"

# Absolute ceiling so a repeatedly-touched entry cannot live forever (and
# so originals, which may contain sensitive tool output, do not sit on disk
# indefinitely). Expressed as a multiple of the entry's idle TTL.
DEFAULT_CCR_MAX_LIFETIME_MULTIPLIER = 8

# One marker per compressed tool result, and a long session runs tens of
# thousands of tool calls. At 1000 the store evicted markers the transcript was
# still quoting, which surfaced to the client as a blocked tool call. Entries
# are small next to the payloads they stand in for, so the bound is set where a
# whole session fits and eviction becomes the exception rather than routine.
DEFAULT_CCR_MAX_ENTRIES = 50000

_RETRIEVAL_LOG_PREVIEW_CHARS = 4096
# Previews carry verbatim tool-result content (post-redaction) — source code,
# credentials the redactor does not recognize, customer data. That is not
# something the always-on runtime log should hold, so previews are OFF unless
# an operator turns them on with 1/true/yes/on; the log then records byte
# counts only. (The log file is created owner-only either way — see
# ``headroom/proxy/helpers.py:_OwnerOnlyRotatingFileHandler`` — because other
# switches put request content in the same file. That is a second line of
# defence, not a reason to log the payload.)
PAYLOAD_PREVIEW_ENV = "HEADROOM_LOG_PAYLOAD_PREVIEW"
_SECRET_KEY_VALUE_RE = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_-]*)"
    r"(\s*[:=]\s*)([\"']?)([^\"'\s,}]+)"
)
_AUTH_VALUE_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}")
_API_KEY_VALUE_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def _get_env_default_ttl_seconds() -> int:
    raw_value = os.environ.get(CCR_TTL_SECONDS_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CCR_TTL_SECONDS

    try:
        ttl_seconds = int(raw_value)
    except ValueError:
        logger.warning(
            "%s must be a positive integer number of seconds, got %r; using %s",
            CCR_TTL_SECONDS_ENV,
            raw_value,
            DEFAULT_CCR_TTL_SECONDS,
        )
        return DEFAULT_CCR_TTL_SECONDS

    if ttl_seconds <= 0:
        logger.warning(
            "%s must be greater than 0, got %s; using %s",
            CCR_TTL_SECONDS_ENV,
            ttl_seconds,
            DEFAULT_CCR_TTL_SECONDS,
        )
        return DEFAULT_CCR_TTL_SECONDS

    return ttl_seconds


PROCESS_LOCAL_STORE_DETAIL = (
    " — this worker's CCR store is in-process only, so entries stored by a "
    "different worker or before a restart are not visible here"
)


def _active_store_is_process_local() -> bool:
    """True when the store serving this request keeps entries in-process only.

    Reads the already-built store (request-scoped first, then the global
    singleton) rather than calling ``get_compression_store()``, so a
    diagnostics call can never be the thing that constructs the singleton.
    """
    store = _request_ccr_store.get() or _compression_store
    if store is None:
        return False
    return store.backend_is_process_local


def format_retrieval_miss_detail(status: dict[str, Any]) -> str:
    """Return an operator-facing miss reason for CCR retrieval failures.

    A *missing* entry is only a TTL story when the store could have held it in
    the first place. With an in-process-only store — which is what
    ``HEADROOM_CCR_BACKEND=memory`` and ``--stateless`` give you — a retrieval
    that lands on a different worker than the compression misses no matter what
    the TTL is, and at ``--workers 4`` that is most retrievals. Say so, instead
    of pointing every operator at a TTL that is not the cause.
    """
    default_ttl = status.get("default_ttl_seconds", DEFAULT_CCR_TTL_SECONDS)
    ttl_seconds = status.get("ttl_seconds", default_ttl)

    entry_status = status.get("status")

    if entry_status == "expired":
        age_seconds = status.get("age_seconds")
        idle_seconds = status.get("idle_seconds")
        reason = status.get("expiry_reason")
        if reason == "max_lifetime":
            if isinstance(age_seconds, (int, float)):
                return (
                    "Entry expired (CCR max lifetime: "
                    f"{status.get('max_lifetime_seconds')} seconds; "
                    f"age: {age_seconds:.0f} seconds)"
                )
            return f"Entry expired (CCR max lifetime: {status.get('max_lifetime_seconds')} seconds)"
        if isinstance(idle_seconds, (int, float)):
            return (
                f"Entry expired (CCR idle TTL: {ttl_seconds} seconds; "
                f"idle: {idle_seconds:.0f} seconds)"
            )
        return f"Entry expired (CCR idle TTL: {ttl_seconds} seconds)"

    # The entry is gone from the store. A tombstone tells us *why*, which
    # is the difference between "raise the TTL" and "raise max_entries".
    if entry_status == "evicted":
        return f"Entry evicted for capacity (CCR store limit: {status.get('max_entries')} entries)"
    if entry_status == "expired_and_purged":
        if status.get("expiry_reason") == "max_lifetime":
            return (
                "Entry expired and was purged (CCR max lifetime: "
                f"{status.get('max_lifetime_seconds')} seconds)"
            )
        return f"Entry expired and was purged (CCR idle TTL: {ttl_seconds} seconds)"

    # No tombstone either. That is not proof the hash was never stored: reasons
    # are process-local (see CompressionStore._tombstones), so a backend sweep
    # in another worker, or any purge that happened before this process started,
    # leaves none behind. Name both possibilities rather than implying the first.
    detail = (
        "Entry not found - never stored, or expired/evicted with no reason "
        f"recorded (CCR idle TTL: {default_ttl} seconds)"
    )
    process_local = status.get("store_is_process_local")
    if process_local is None:
        process_local = _active_store_is_process_local()
    if process_local:
        detail += PROCESS_LOCAL_STORE_DETAIL
    return detail


def _redact_retrieval_log_payload(payload: str) -> str:
    redacted = _SECRET_KEY_VALUE_RE.sub(r"\1\2\3[REDACTED]", payload)
    redacted = _AUTH_VALUE_RE.sub(r"\1 [REDACTED]", redacted)
    return _API_KEY_VALUE_RE.sub("sk-[REDACTED]", redacted)


def _payload_preview_enabled() -> bool:
    """Whether the retrieval log may carry a slice of the retrieved payload.

    Off by default, on only when an operator opts in explicitly. Always off
    under stateless mode regardless of the env var: the retrieval log goes to
    the always-on runtime log file under the workspace, so a preview there
    would put tool-result content on disk, the thing a stateless deployment
    selected the flag to avoid.
    """
    from ..paths import process_is_stateless

    if process_is_stateless():
        return False
    raw = os.environ.get(PAYLOAD_PREVIEW_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _payload_for_retrieval_log(payload: str) -> dict[str, Any]:
    if not _payload_preview_enabled():
        return {
            "payload_chars": len(payload),
            "payload_preview_chars": 0,
            "payload_truncated": len(payload) > 0,
            "payload_preview": "",
        }
    redacted = _redact_retrieval_log_payload(payload)
    preview = redacted[:_RETRIEVAL_LOG_PREVIEW_CHARS]
    truncated = len(redacted) > len(preview)
    return {
        "payload_chars": len(payload),
        "payload_preview_chars": len(preview),
        "payload_truncated": truncated,
        "payload_preview": preview,
    }


# Single source of truth for the retrieval-miss message. Actionable by
# design: the model still has the marker in context (Read markers carry
# the file path), so tell it how to recover instead of just reporting
# the miss.
CCR_MISS_MESSAGE = (
    "Entry not found or expired. To recover: if the compression marker "
    "references a file Read, re-read that file (the path is in the "
    "marker; disk is the source of truth). If it was command output, "
    "re-run the command. Entries expire after the store goes untouched "
    "for the idle TTL (default 24 hours, restarted on every retrieval; "
    "configurable via HEADROOM_CCR_TTL_SECONDS), and when the CCR store is "
    "in-process only (HEADROOM_CCR_BACKEND=memory, which --stateless forces) "
    "they are also invisible to other workers and lost on restart, so a miss "
    "need not mean the TTL elapsed."
)


@dataclass
class CompressionEntry:
    """A cached compression entry with metadata for retrieval and feedback."""

    hash: str
    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    original_item_count: int
    compressed_item_count: int
    tool_name: str | None
    tool_call_id: str | None
    query_context: str | None
    created_at: float
    ttl: int = DEFAULT_CCR_TTL_SECONDS
    # Absolute ceiling measured from created_at. None means
    # ttl * DEFAULT_CCR_MAX_LIFETIME_MULTIPLIER.
    max_lifetime: int | None = None

    # TOIN integration: Store the tool signature hash for retrieval correlation
    # This MUST match the hash used by SmartCrusher when recording compression
    tool_signature_hash: str | None = None
    compression_strategy: str | None = None  # Strategy used for compression

    # Which observation-masking gate admitted this block: "age" for the normal
    # turn-age gate, "episode" for the closed-episode relaxation. Read by
    # get_stats to report retrieval rate per gate, which is the only evidence
    # that says whether the episode boundary masks material still in play.
    mask_gate: str | None = None

    # Feedback tracking
    retrieval_count: int = 0
    search_queries: list[str] = field(default_factory=list)
    last_accessed: float | None = None

    @property
    def max_lifetime_seconds(self) -> int:
        """Absolute lifetime ceiling, derived from ttl when unset."""
        if self.max_lifetime is not None:
            return self.max_lifetime
        return self.ttl * DEFAULT_CCR_MAX_LIFETIME_MULTIPLIER

    @property
    def idle_since(self) -> float:
        """Timestamp the idle window is measured from."""
        return self.last_accessed if self.last_accessed is not None else self.created_at

    @property
    def expires_at(self) -> float:
        """Earliest moment this entry can expire, given current access state."""
        return min(
            self.idle_since + self.ttl,
            self.created_at + self.max_lifetime_seconds,
        )

    def expiry_reason(self, now: float | None = None) -> str | None:
        """Return why this entry is expired, or None if it is still live."""
        now = time.time() if now is None else now
        if now - self.created_at > self.max_lifetime_seconds:
            return "max_lifetime"
        if now - self.idle_since > self.ttl:
            return "idle"
        return None

    def is_expired(self) -> bool:
        """Check if this entry has expired.

        The TTL is an *idle* window: record_access() moves ``last_accessed``,
        so an entry the agent keeps retrieving stays alive until it goes
        untouched for ``ttl`` seconds, bounded by ``max_lifetime_seconds``.
        """
        return self.expiry_reason() is not None

    def record_access(self, query: str | None = None) -> None:
        """Record an access to this entry for feedback tracking."""
        self.retrieval_count += 1
        self.last_accessed = time.time()
        if query and query not in self.search_queries:
            self.search_queries.append(query)
            # Keep only last 10 queries
            if len(self.search_queries) > 10:
                self.search_queries = self.search_queries[-10:]


@dataclass
class RetrievalEvent:
    """Event logged when content is retrieved from cache."""

    hash: str
    query: str | None
    items_retrieved: int
    total_items: int
    tool_name: str | None
    timestamp: float
    retrieval_type: str  # always "full" (retrieval is by hash)
    tool_signature_hash: str | None = None  # For TOIN correlation


class CompressionStore:
    """Thread-safe store for compressed content with retrieval support.

    This is the core of the CCR architecture. When SmartCrusher compresses
    an array, the original content is stored here. If the LLM needs more
    data, it can retrieve from this cache instantly.

    Design principles:
    - Zero external dependencies (pure Python)
    - Thread-safe for concurrent access
    - Idle-based expiration (default 3600 seconds since last access,
      env-configurable) with an absolute lifetime ceiling
    - LRU-style eviction when capacity is reached
    - Hash-keyed retrieval that always returns the full original content
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_CCR_MAX_ENTRIES,
        default_ttl: int = DEFAULT_CCR_TTL_SECONDS,
        enable_feedback: bool = True,
        backend: CompressionStoreBackend | None = None,
        default_max_lifetime: int | None = None,
    ):
        """Initialize the compression store.

        Args:
            max_entries: Maximum number of entries to store.
            default_ttl: Default idle TTL in seconds (default 60 minutes,
                restarted on every retrieval).
            default_max_lifetime: Absolute lifetime ceiling in seconds.
                Defaults to ``default_ttl * DEFAULT_CCR_MAX_LIFETIME_MULTIPLIER``.
            enable_feedback: Whether to track retrieval events.
            backend: Storage backend to use. Defaults to InMemoryBackend
                     when constructed directly; `get_compression_store()`
                     defaults to SQLiteBackend for restart/multi-worker
                     safety. Custom backends can be passed for
                     persistence (MongoDB, Redis).
        """
        # Import here to avoid circular imports
        from .backends import InMemoryBackend

        self._backend: CompressionStoreBackend = backend or InMemoryBackend()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._default_max_lifetime = (
            default_max_lifetime
            if default_max_lifetime is not None
            else default_ttl * DEFAULT_CCR_MAX_LIFETIME_MULTIPLIER
        )
        self._enable_feedback = enable_feedback

        # Why a hash went away, kept after the entry itself is deleted so a
        # miss can say "expired" / "evicted for capacity" instead of the
        # indistinguishable "not found" that #2604 reported. Bounded ring —
        # this is diagnostics, not state.
        #
        # Best-effort by construction, and deliberately so: a reason is only
        # recorded when *this* store instance performs the deletion. The
        # backend also drops expired rows on its own — SQLiteBackend._open()
        # sweeps at startup and _maybe_purge() sweeps on write — and a process
        # restart takes the whole ring with it. Those misses degrade to a plain
        # "not found", which format_retrieval_miss_detail() reports as
        # ambiguous. Persisting reasons would need backend support every
        # backend can opt out of (Redis/Mongo load by entry point), so the
        # honest wording is required regardless; see the follow-up issue.
        self._tombstones: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_tombstones = 512

        # Feedback tracking. maxlen caps the display history, replacing an
        # append-then-reslice that re-copied 1000 pointers on every retrieval.
        self._max_events = 1000  # Keep last 1000 events
        self._retrieval_events: deque[RetrievalEvent] = deque(maxlen=self._max_events)
        # Deliberately NOT bounded: this is a drain-by-swap queue, and every
        # event in it still owes a feedback notification.
        self._pending_feedback_events: list[RetrievalEvent] = []

        # MEDIUM FIX #16: Use a min-heap for O(log n) eviction instead of O(n)
        # Heap entries are (idle_since, hash_key) tuples — keyed on last
        # access, not creation, so eviction drops the coldest entry rather
        # than the oldest one the agent is still actively retrieving.
        self._eviction_heap: list[tuple[float, str]] = []
        # CRITICAL FIX: Track stale entries count to know when heap cleanup is needed
        self._stale_heap_entries = 0
        # Threshold for triggering heap rebuild (when 50% are stale)
        self._heap_rebuild_threshold = 0.5

        # Lifetime masking-gate counters, keyed by mask_gate. These live on the
        # store rather than being derived from entries because the interesting
        # ratio spans evicted and expired entries: a block that was masked,
        # never retrieved, and then aged out is exactly the success case, and
        # summing over live entries would drop it from the denominator.
        # _first_retrieved counts distinct hashes retrieved at least once, so a
        # block fetched three times still counts as one block that was needed.
        self._mask_gate_stored: dict[str, int] = {}
        self._mask_gate_first_retrieved: dict[str, int] = {}
        self._mask_gate_retrievals: dict[str, int] = {}

    def _record_tombstone(self, hash_key: str, reason: str, entry: CompressionEntry) -> None:
        """Remember why a hash was removed. Must be called with lock held."""
        self._tombstones.pop(hash_key, None)
        self._tombstones[hash_key] = {
            "reason": reason,
            "at": time.time(),
            "ttl_seconds": entry.ttl,
            "max_lifetime_seconds": entry.max_lifetime_seconds,
            "created_at": entry.created_at,
            "last_accessed": entry.last_accessed,
            "retrieval_count": entry.retrieval_count,
        }
        while len(self._tombstones) > self._max_tombstones:
            self._tombstones.popitem(last=False)

    @property
    def default_ttl_seconds(self) -> int:
        """Default TTL applied to new entries when callers do not override it."""
        return self._default_ttl

    @property
    def backend_is_process_local(self) -> bool:
        """True when this store's entries live only in this process."""
        with self._lock:
            return bool(getattr(self._backend, "is_process_local", False))

    @property
    def backend_is_stateless_safe(self) -> bool:
        """True when this store's backend may be kept in a stateless process.

        See ``backend_is_stateless_safe()`` (module level) for the rule.
        """
        with self._lock:
            return backend_is_stateless_safe(self._backend)

    @property
    def backend_class_name(self) -> str:
        """Class name of the current backend, for logs and diagnostics."""
        with self._lock:
            return type(self._backend).__name__

    def use_process_local_backend(self) -> bool:
        """Swap this store's backend for a fresh in-process one.

        This is how stateless mode stops the CCR store writing tool-result
        originals to disk. Two things it deliberately does NOT do:

        - It does not ``clear()``. The on-disk backend is left exactly as it
          was: ``ccr_store.db`` is shared with the other worker processes and
          survives restarts by design, so deleting its rows would destroy
          entries live workers are still serving.
        - It does not rebuild the store object. Callers that already captured
          this store (``CompressionFeedback._store``, request-scoped
          stores, anything holding the singleton) keep their reference and
          stop writing to disk with it. Dropping the module global alone would
          leave those writers happily persisting originals.

        Entries already loaded from the shared store are NOT carried over: a
        stateless process should not go on serving — or holding in RAM —
        originals it was told to stop touching.

        A backend that ``backend_is_stateless_safe()`` accepts is left alone,
        so this is a no-op on a store that is already in-process only and on
        one an operator declared off-machine.

        Returns True if a backend was actually swapped out.
        """
        from .backends import InMemoryBackend

        with self._lock:
            previous = self._backend
            if backend_is_stateless_safe(previous):
                return False
            self._backend = InMemoryBackend()
            self._retrieval_events.clear()
            self._pending_feedback_events.clear()
            self._eviction_heap.clear()
            self._stale_heap_entries = 0

        close = getattr(previous, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - never fail startup over a close
                logger.debug("Closing previous CCR backend failed", exc_info=True)
        return True

    def store(
        self,
        original: str,
        compressed: str,
        *,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        original_item_count: int = 0,
        compressed_item_count: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        query_context: str | None = None,
        tool_signature_hash: str | None = None,
        compression_strategy: str | None = None,
        ttl: int | None = None,
        max_lifetime: int | None = None,
        explicit_hash: str | None = None,
        mask_gate: str | None = None,
    ) -> str:
        """Store compressed content and return hash for retrieval.

        Args:
            original: Original JSON content before compression.
            compressed: Compressed JSON content.
            original_tokens: Token count of original content.
            compressed_tokens: Token count of compressed content.
            original_item_count: Number of items in original array.
            compressed_item_count: Number of items after compression.
            tool_name: Name of the tool that produced this output.
            tool_call_id: ID of the tool call.
            query_context: User query context for relevance matching.
            tool_signature_hash: Hash from ToolSignature for TOIN correlation.
            compression_strategy: Strategy used for compression.
            ttl: Custom idle TTL in seconds (uses default if not specified).
            max_lifetime: Custom absolute lifetime ceiling in seconds. Defaults
                to the store default when ttl is also default, otherwise to
                ttl * DEFAULT_CCR_MAX_LIFETIME_MULTIPLIER.
            explicit_hash: Use this exact hex hash as the storage key
                instead of computing SHA-256(original)[:24]. Required when
                the marker that points at this entry was emitted by a
                producer with its own hash function (e.g. SmartCrusher's
                Rust row-drop path uses SHA-256[:12]). If not a hex
                string, raises ``ValueError``. The marker hash and the
                store key MUST match — otherwise ``/v1/retrieve/{hash}``
                returns 404 even though the data is present.

        Returns:
            Hash key for retrieving this content.
        """
        # Generate hash from original content. Default: SHA-256[:24] of the
        # original. When the caller provides `explicit_hash`, use it
        # verbatim — required when the hash that ends up in the prompt
        # marker is produced by another component (e.g. the Rust
        # SmartCrusher row-drop path emits SHA-256[:12], which the
        # Python store has to mirror so /v1/retrieve resolves it).
        # 24 chars (96 bits) was chosen for collision resistance under the
        # birthday bound: 50% collision probability at ~280 trillion entries
        # (2^48), versus ~4 billion (2^32) for the previous 16-char default.
        if explicit_hash is not None:
            # Validate as hex. Bail loudly per `feedback_no_silent_fallbacks`
            # — silently falling back to the default hash when the caller
            # asked for a specific key would defeat the marker/store
            # consistency we're trying to preserve.
            if not explicit_hash or not all(c in "0123456789abcdefABCDEF" for c in explicit_hash):
                raise ValueError(
                    f"explicit_hash must be a non-empty hex string, got {explicit_hash!r}"
                )
            hash_key = explicit_hash.lower()
        else:
            # SHA-256 truncated to 24 hex chars (96 bits) — same collision
            # space as the MD5[:24] this replaced. Switched from MD5 in
            # PR #395 to silence CodeQL's `py/weak-sensitive-data-hashing`
            # rule (the `usedforsecurity=False` parameter and the `lgtm`
            # comment marker both failed to suppress it). The cache is
            # in-memory, so changing the hash function on upgrade has no
            # persistence-side effect — the same content always hashes
            # deterministically under whichever function is in use.
            hash_key = hashlib.sha256(original.encode()).hexdigest()[:24]

        # Refuse to persist a bare CCR marker as an entry's "original"
        # (#2694). A marker is a *pointer* to content, never content: an
        # entry like `hash=abc123 -> "<<ccr:abc123,base64,2.0KB>>"` answers a
        # retrieve with the very placeholder the caller is trying to resolve,
        # and (worse) can overwrite a good entry with a useless one. Any
        # producer that gets here has lost the source bytes upstream, so fail
        # loudly rather than silently converting "retrievable" into "gone".
        # Narrow by design: only a *bare* marker is rejected. Legitimate
        # originals may legally CONTAIN markers (nested offloads, a tool that
        # echoed one), and refusing those would drop recoverable data.
        #
        # The rejected value is never echoed into the log. It is provably a
        # bare marker here, but `original` is the store's credential-bearing
        # payload in the general case (this issue was reported against an
        # OAuth token), and an error path is exactly where that sort of leak
        # survives review. `hash_key` already identifies the entry.
        stripped = original.strip()
        if stripped.startswith("<<ccr:") and stripped.endswith(">>") and "\n" not in stripped:
            logger.error(
                "CCR store: refusing to persist a bare retrieval marker as "
                "original_content (hash=%s tool=%s strategy=%s len=%d) — the "
                "producer lost the source bytes; retrieval for this hash will "
                "miss instead of returning a placeholder",
                hash_key,
                tool_name,
                compression_strategy,
                len(stripped),
            )
            return hash_key

        effective_ttl = ttl if ttl is not None else self._default_ttl
        if max_lifetime is not None:
            effective_max_lifetime = max_lifetime
        elif ttl is None:
            effective_max_lifetime = self._default_max_lifetime
        else:
            effective_max_lifetime = effective_ttl * DEFAULT_CCR_MAX_LIFETIME_MULTIPLIER

        entry = CompressionEntry(
            hash=hash_key,
            original_content=original,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_item_count=original_item_count,
            compressed_item_count=compressed_item_count,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            query_context=query_context,
            created_at=time.time(),
            ttl=effective_ttl,
            max_lifetime=effective_max_lifetime,
            tool_signature_hash=tool_signature_hash,
            compression_strategy=compression_strategy,
            mask_gate=mask_gate,
        )

        # Process pending feedback BEFORE acquiring lock for eviction.
        # This ensures feedback from entries about to be evicted is captured.
        if self._enable_feedback:
            self.process_pending_feedback()

        with self._lock:
            # Decide whether this is a NEW key before evicting. Evicting to make
            # room only applies to a genuinely new entry; a re-store of an
            # existing key overwrites in place (no room needed). Evicting first
            # for a duplicate would needlessly destroy a live, unrelated entry
            # and drop the store below capacity, making that entry's <<ccr:...>>
            # marker (still sitting in the conversation) unredeemable — a 404.
            # The CCR mirror bridge re-stores the same explicit_hash on every
            # turn a marker is re-encountered, so duplicate stores are common.
            existing = self._backend.get(hash_key)
            if existing is None:
                # Counted here, inside the new-key branch, because the CCR
                # mirror bridge re-stores the same hash on every turn a marker
                # survives. Counting each call would inflate the denominator by
                # roughly the number of turns the marker lives.
                if mask_gate:
                    self._mask_gate_stored[mask_gate] = (
                        self._mask_gate_stored.get(mask_gate, 0) + 1
                    )
                self._evict_if_needed()
            else:
                # Hash already present. Different content means a true (extremely
                # rare with SHA256[:24]) collision; same content is a duplicate
                # re-store. Either way we overwrite in place.
                if existing.original_content != original:
                    logger.warning(
                        "Hash collision detected: hash=%s tool=%s (existing_len=%d, new_len=%d)",
                        hash_key,
                        tool_name,
                        len(existing.original_content),
                        len(original),
                    )
                else:
                    logger.debug(
                        "Duplicate store for hash=%s, updating entry",
                        hash_key,
                    )
                # Mark old heap entry as stale since we're replacing it.
                self._stale_heap_entries += 1

            self._backend.set(hash_key, entry)
            # MEDIUM FIX #16: Add to eviction heap for O(log n) eviction
            heapq.heappush(self._eviction_heap, (entry.idle_since, hash_key))
            # A re-store makes the hash live again; drop any stale tombstone
            # so a later miss is not misreported with an old reason.
            self._tombstones.pop(hash_key, None)

        return hash_key

    def retrieve(
        self,
        hash_key: str,
        query: str | None = None,
    ) -> CompressionEntry | None:
        """Retrieve original content by hash.

        Args:
            hash_key: Hash key returned by store().
            query: Optional query for feedback tracking.

        Returns:
            CompressionEntry if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._backend.get(hash_key)

            if entry is None:
                return None

            expiry_reason = entry.expiry_reason()
            if expiry_reason is not None:
                self._backend.delete(hash_key)
                self._record_tombstone(hash_key, f"expired:{expiry_reason}", entry)
                # CRITICAL FIX: Track stale heap entry
                self._stale_heap_entries += 1
                return None

            # Counted before record_access, which is what advances the count
            # from zero. A block fetched repeatedly is still one block the
            # model needed back, so the first-retrieval tally is the one that
            # divides against _mask_gate_stored.
            if entry.mask_gate:
                if entry.retrieval_count == 0:
                    self._mask_gate_first_retrieved[entry.mask_gate] = (
                        self._mask_gate_first_retrieved.get(entry.mask_gate, 0) + 1
                    )
                self._mask_gate_retrievals[entry.mask_gate] = (
                    self._mask_gate_retrievals.get(entry.mask_gate, 0) + 1
                )

            # Track access for feedback. This also restarts the idle window,
            # so an entry in active use never expires under the agent.
            entry.record_access(query)
            # Update the backend with the modified entry
            self._backend.set(hash_key, entry)
            # Re-key the entry in the eviction heap on its new access time.
            # The old tuple becomes stale; the counter drives the rebuild.
            heapq.heappush(self._eviction_heap, (entry.idle_since, hash_key))
            self._stale_heap_entries += 1

            # Log retrieval event
            if self._enable_feedback:
                self._log_retrieval(
                    hash_key=hash_key,
                    query=query,
                    items_retrieved=entry.original_item_count,
                    total_items=entry.original_item_count,
                    tool_name=entry.tool_name,
                    retrieval_type="full",
                    tool_signature_hash=entry.tool_signature_hash,
                )
            self._log_retrieval_payload(
                hash_key=hash_key,
                query=query,
                retrieval_type="full",
                payload=entry.original_content,
                items_retrieved=entry.original_item_count,
                total_items=entry.original_item_count,
                entry=entry,
            )

            # CRITICAL: Make a deep copy to return
            # (entry could be modified/evicted after lock release)
            # The entry contains mutable fields (search_queries list) that must be copied
            result_entry = replace(entry, search_queries=list(entry.search_queries))

        # Process feedback immediately to ensure TOIN learns in real-time
        if self._enable_feedback:
            self.process_pending_feedback()

        return result_entry

    def get_metadata(
        self,
        hash_key: str,
    ) -> dict[str, Any] | None:
        """Get metadata about a stored entry without retrieving full content.

        Useful for context tracking to know what was compressed without
        fetching the entire original content.

        Args:
            hash_key: Hash key returned by store().

        Returns:
            Dict with metadata if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._backend.get(hash_key)

            if entry is None:
                return None

            reason = entry.expiry_reason()
            if reason is not None:
                self._backend.delete(hash_key)
                # Record why, so a later get_entry_status() on this hash still
                # reports the expiry instead of a generic "not found".
                self._record_tombstone(hash_key, f"expired:{reason}", entry)
                self._stale_heap_entries += 1
                return None

            return {
                "hash": entry.hash,
                "tool_name": entry.tool_name,
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "query_context": entry.query_context,
                "compressed_content": entry.compressed_content,
                "original_content_preview": entry.original_content[:2000],
                "created_at": entry.created_at,
                "ttl": entry.ttl,
            }

    def _log_retrieval_payload(
        self,
        *,
        hash_key: str,
        query: str | None,
        retrieval_type: str,
        payload: str,
        items_retrieved: int,
        total_items: int,
        entry: CompressionEntry,
    ) -> None:
        event = {
            "event": "headroom_retrieve",
            "hash": hash_key,
            "retrieval_type": retrieval_type,
            "query": query,
            "items_retrieved": items_retrieved,
            "total_items": total_items,
            "tool_name": entry.tool_name,
            "tool_call_id": entry.tool_call_id,
            "compression_strategy": entry.compression_strategy,
            "tool_signature_hash": entry.tool_signature_hash,
            "original_tokens": entry.original_tokens,
            "compressed_tokens": entry.compressed_tokens,
            "original_item_count": entry.original_item_count,
            "compressed_item_count": entry.compressed_item_count,
            **_payload_for_retrieval_log(payload),
        }
        logger.info(
            "event=headroom_retrieve %s",
            json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        )

    def exists(self, hash_key: str, clean_expired: bool = False) -> bool:
        """Check if a hash key exists and is not expired.

        Args:
            hash_key: The hash key to check.
            clean_expired: If True, delete the entry if expired.
                          LOW FIX #20: Default False to make this a pure check.

        Returns:
            True if the entry exists and is not expired.
        """
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return False
            reason = entry.expiry_reason()
            if reason is not None:
                # LOW FIX #20: Only delete if explicitly requested
                # This makes exists() a pure check by default
                if clean_expired:
                    self._backend.delete(hash_key)
                    self._record_tombstone(hash_key, f"expired:{reason}", entry)
                    # CRITICAL FIX: Track stale heap entry
                    self._stale_heap_entries += 1
                return False
            return True

    def stored_hashes(self) -> list[str]:
        """Every hash currently held, expired or not.

        The marker guards use this to look for a near miss when a hash does
        not resolve, so an entry that is present but expired is still worth
        naming: it tells the caller the hash was right and the content aged
        out, which is a different problem from a mistyped hash.
        """
        with self._lock:
            return list(self._backend.keys())

    def get_entry_status(
        self,
        hash_key: str,
        *,
        clean_expired: bool = False,
    ) -> dict[str, Any]:
        """Return availability and TTL metadata for a stored entry."""
        now = time.time()
        with self._lock:
            process_local = bool(getattr(self._backend, "is_process_local", False))
            entry = self._backend.get(hash_key)
            if entry is None:
                status = self._missing_status(hash_key)
                # Lets format_retrieval_miss_detail explain a cross-worker
                # miss instead of blaming the TTL.
                status["store_is_process_local"] = process_local
                return status

            reason = entry.expiry_reason(now)
            status = {
                "hash": hash_key,
                "status": "expired" if reason else "available",
                "ttl_seconds": entry.ttl,
                "max_lifetime_seconds": entry.max_lifetime_seconds,
                "default_ttl_seconds": self._default_ttl,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "expires_at": entry.expires_at,
                "age_seconds": now - entry.created_at,
                "idle_seconds": now - entry.idle_since,
                "expiry_reason": reason,
            }

            if reason and clean_expired:
                self._backend.delete(hash_key)
                self._record_tombstone(hash_key, f"expired:{reason}", entry)
                self._stale_heap_entries += 1

            return status

    def _missing_status(self, hash_key: str) -> dict[str, Any]:
        """Status for a hash the store no longer holds. Lock must be held."""
        status: dict[str, Any] = {
            "hash": hash_key,
            "status": "missing",
            "default_ttl_seconds": self._default_ttl,
            "max_entries": self._max_entries,
        }
        tombstone = self._tombstones.get(hash_key)
        if tombstone is None:
            return status

        reason = tombstone["reason"]
        status["removed_at"] = tombstone["at"]
        status["ttl_seconds"] = tombstone["ttl_seconds"]
        status["max_lifetime_seconds"] = tombstone["max_lifetime_seconds"]
        status["retrieval_count"] = tombstone["retrieval_count"]
        if reason == "evicted":
            status["status"] = "evicted"
        elif reason.startswith("expired:"):
            status["status"] = "expired_and_purged"
            status["expiry_reason"] = reason.split(":", 1)[1]
        return status

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics for monitoring."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()

            # Get all entries for statistics
            entries = [entry for _, entry in self._backend.items()]
            total_original_tokens = sum(e.original_tokens for e in entries)
            total_compressed_tokens = sum(e.compressed_tokens for e in entries)
            total_retrievals = sum(e.retrieval_count for e in entries)

            # Include backend stats
            backend_stats = self._backend.get_stats()

            return {
                "entry_count": self._backend.count(),
                "max_entries": self._max_entries,
                "default_ttl_seconds": self._default_ttl,
                "default_max_lifetime_seconds": self._default_max_lifetime,
                "total_original_tokens": total_original_tokens,
                "total_compressed_tokens": total_compressed_tokens,
                "total_retrievals": total_retrievals,
                "event_count": len(self._retrieval_events),
                "backend": backend_stats,
                "mask_gates": self._mask_gate_stats(),
            }

    def _mask_gate_stats(self) -> dict[str, dict[str, float | int]]:
        """Per-gate masking outcome, for judging the closed-episode boundary.

        ``retrieval_rate`` is the number that matters. The age gate is the
        control: it waits for the model to demonstrably move on, so its rate is
        the floor for how often masking guesses wrong. If the episode gate runs
        materially above that floor it is closing tasks that are still in play,
        and the boundary needs to be stricter (two consecutive user asks, or a
        minimum token distance) rather than firing on the first new ask.

        Caller holds ``self._lock``.
        """
        gates = (
            set(self._mask_gate_stored)
            | set(self._mask_gate_first_retrieved)
            | set(self._mask_gate_retrievals)
        )
        stats: dict[str, dict[str, float | int]] = {}
        for gate in sorted(gates):
            stored = self._mask_gate_stored.get(gate, 0)
            first_retrieved = self._mask_gate_first_retrieved.get(gate, 0)
            stats[gate] = {
                "stored": stored,
                "retrieved": first_retrieved,
                "retrievals": self._mask_gate_retrievals.get(gate, 0),
                "retrieval_rate": (first_retrieved / stored) if stored else 0.0,
            }
        return stats

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        with self._lock:
            # Get backend stats which include bytes_used
            backend_stats = self._backend.get_stats()
            bytes_used = backend_stats.get("bytes_used", 0)

            # Add retrieval events memory
            import sys

            bytes_used += sys.getsizeof(self._retrieval_events)
            for event in self._retrieval_events:
                bytes_used += sys.getsizeof(event)

            # Add eviction heap memory
            bytes_used += sys.getsizeof(self._eviction_heap)

            return ComponentStats(
                name="compression_store",
                entry_count=self._backend.count(),
                size_bytes=bytes_used,
                budget_bytes=None,  # No budget set yet
                hits=sum(1 for _, e in self._backend.items() if e.retrieval_count > 0),
                misses=0,  # CompressionStore doesn't track misses directly
                evictions=0,  # Would need to track this separately
            )

    def get_retrieval_events(
        self,
        limit: int = 100,
        tool_name: str | None = None,
    ) -> list[RetrievalEvent]:
        """Get recent retrieval events for feedback analysis.

        Args:
            limit: Maximum number of events to return.
            tool_name: Filter by tool name if specified.

        Returns:
            List of recent retrieval events (copies to prevent mutation).
        """
        with self._lock:
            # MEDIUM FIX #17: Take a slice copy immediately to avoid race conditions
            # if another thread modifies _retrieval_events after we release the lock
            events_copy = list(self._retrieval_events)

        # Filter and slice outside lock (safe since we have a copy)
        if tool_name:
            events_copy = [e for e in events_copy if e.tool_name == tool_name]

        return list(reversed(events_copy[-limit:]))

    def clear(self) -> None:
        """Clear all entries. Mainly for testing."""
        with self._lock:
            self._backend.clear()
            self._retrieval_events.clear()
            self._pending_feedback_events.clear()
            self._eviction_heap.clear()  # MEDIUM FIX #16: Clear heap too
            self._tombstones.clear()
            self._stale_heap_entries = 0  # CRITICAL FIX: Reset stale counter

    def _evict_if_needed(self) -> None:
        """Evict old entries if at capacity. Must be called with lock held.

        MEDIUM FIX #16: Use heap for O(log n) eviction instead of O(n) scan.
        CRITICAL FIX: Track and clean stale heap entries to prevent memory leak.
        """
        # First, remove expired entries
        self._clean_expired()

        # CRITICAL FIX: Rebuild heap if too many stale entries
        # This prevents unbounded heap growth when entries are deleted/replaced
        heap_size = len(self._eviction_heap)
        if heap_size > 0:
            stale_ratio = self._stale_heap_entries / heap_size
            if stale_ratio >= self._heap_rebuild_threshold:
                self._rebuild_heap()

        # If still at capacity, remove oldest entries using heap
        deferred: list[tuple[float, str]] = []
        while self._backend.count() >= self._max_entries and self._eviction_heap:
            # Pop coldest from heap (O(log n))
            idle_since, hash_key = heapq.heappop(self._eviction_heap)

            # Check if entry still exists and matches timestamp
            # (entry might have been deleted, replaced, or accessed since —
            # an accessed entry has a newer tuple further up the heap)
            entry = self._backend.get(hash_key)
            if entry is None or entry.idle_since != idle_since:
                # CRITICAL FIX: This was a stale entry, decrement counter
                # (we already popped it, so the stale entry is now gone)
                if self._stale_heap_entries > 0:
                    self._stale_heap_entries -= 1
                continue
            if entry.retrieval_count > 0:
                # A retrieval is proof the marker is still live in a transcript:
                # the client quoted it back at us. Dropping one of those makes the
                # response-side marker guard block a legitimate tool call over a
                # payload only headroom lost, which costs the client real work.
                # Ordering by created_at alone made this the common case, because
                # the oldest entries in a long session are exactly the ones still
                # being quoted. Spare it and let a never-retrieved entry go first.
                deferred.append((idle_since, hash_key))
                continue
            # Entry was never retrieved = compression was successful
            # Notify feedback system so it knows this strategy worked
            if self._enable_feedback:
                self._record_eviction_success(entry)
            self._backend.delete(hash_key)
            self._record_tombstone(hash_key, "evicted", entry)

        # Everything spared goes back, keyed by idle_since so the staleness check
        # above still matches. If every entry is live the loop drains the heap and
        # capacity is exceeded on purpose: a marker that is still quoted is worth
        # more than the bound, and the TTL sweep reclaims it when the session ends.
        for spared in deferred:
            heapq.heappush(self._eviction_heap, spared)

    def _clean_expired(self) -> None:
        """Remove expired entries. Must be called with lock held.

        CRITICAL FIX: Track stale heap entries when deleting to prevent memory leak.
        """
        purge_expired = getattr(self._backend, "purge_expired", None)
        if callable(purge_expired):
            self._stale_heap_entries += purge_expired()
            return

        expired = [
            (key, entry, reason)
            for key, entry in self._backend.items()
            if (reason := entry.expiry_reason()) is not None
        ]
        for key, entry, reason in expired:
            self._backend.delete(key)
            self._record_tombstone(key, f"expired:{reason}", entry)
            # CRITICAL FIX: Increment stale counter - the heap still has an entry
            # for this key that will be stale when we try to evict
            self._stale_heap_entries += 1

    def _rebuild_heap(self) -> None:
        """Rebuild heap from current store entries. Must be called with lock held.

        CRITICAL FIX: This removes stale heap entries that accumulate when entries
        are deleted or replaced. Without this, the heap grows unboundedly.
        """
        # Build new heap from current store entries only
        self._eviction_heap = [
            (entry.idle_since, hash_key) for hash_key, entry in self._backend.items()
        ]
        heapq.heapify(self._eviction_heap)
        # Reset stale counter - heap is now clean
        self._stale_heap_entries = 0
        logger.debug(
            "Rebuilt eviction heap: %d entries",
            len(self._eviction_heap),
        )

    def _record_eviction_success(self, entry: CompressionEntry) -> None:
        """Record successful compression when an entry is evicted without retrieval.

        HIGH FIX: State divergence on eviction
        When an entry is evicted and was NEVER retrieved, this indicates the
        compression was fully successful - the LLM never needed the original data.
        We notify the feedback system so it can learn from this success.

        Must be called with lock held (entry data access).
        Actual feedback notification happens outside lock.

        Args:
            entry: The entry being evicted.
        """
        # Capture entry data while we have the lock
        tool_name = entry.tool_name
        sig_hash = entry.tool_signature_hash
        strategy = entry.compression_strategy

        # We can't call feedback while holding the lock (would cause deadlock)
        # Instead, queue this for deferred processing
        if sig_hash is not None and strategy is not None:
            # Create a synthetic "success" event that we'll process later
            # Use a special retrieval type to indicate this was an eviction success
            success_event = RetrievalEvent(
                hash=entry.hash,
                query=None,
                items_retrieved=0,  # No retrieval happened
                total_items=entry.original_item_count,
                tool_name=tool_name,
                timestamp=time.time(),
                retrieval_type="eviction_success",  # Special marker
                tool_signature_hash=sig_hash,
            )
            self._pending_feedback_events.append(success_event)
            logger.debug(
                "Recorded eviction success: hash=%s strategy=%s",
                entry.hash[:8],
                strategy,
            )

    def _log_retrieval(
        self,
        hash_key: str,
        query: str | None,
        items_retrieved: int,
        total_items: int,
        tool_name: str | None,
        retrieval_type: str,
        tool_signature_hash: str | None = None,
    ) -> None:
        """Log a retrieval event. Must be called with lock held."""
        event = RetrievalEvent(
            hash=hash_key,
            query=query,
            items_retrieved=items_retrieved,
            total_items=total_items,
            tool_name=tool_name,
            timestamp=time.time(),
            retrieval_type=retrieval_type,
            tool_signature_hash=tool_signature_hash,
        )

        # maxlen keeps this bounded; no trim needed here.
        self._retrieval_events.append(event)

        # Queue event for feedback processing (will be processed after lock release)
        # This is safe because process_pending_feedback() uses the lock to atomically
        # swap out the pending list before processing
        self._pending_feedback_events.append(event)

    def process_pending_feedback(self) -> None:
        """Process pending feedback events.

        Forwards events to:
        1. CompressionFeedback - for learning compression hints
        2. TelemetryCollector - for the data flywheel
        3. TOIN - for cross-user intelligence network

        This is called automatically on each retrieval to ensure the
        feedback loop operates in real-time.
        """
        from ..telemetry import get_telemetry_collector
        from ..telemetry.toin import get_toin
        from .compression_feedback import get_compression_feedback

        # Get pending events and related entry data atomically
        with self._lock:
            events = self._pending_feedback_events
            self._pending_feedback_events = []

            # Gather entry data while holding lock to avoid race conditions
            # Tuple: (event, tool_name, sig_hash, strategy, compressed_content)
            event_data: list[
                tuple[RetrievalEvent, str | None, str | None, str | None, str | None]
            ] = []
            for event in events:
                entry = self._backend.get(event.hash)
                if entry:
                    # Use the ACTUAL tool_signature_hash stored during compression
                    # This MUST match the hash used by SmartCrusher
                    event_data.append(
                        (
                            event,
                            entry.tool_name,
                            entry.tool_signature_hash,  # The correct hash!
                            entry.compression_strategy,
                            entry.compressed_content,  # For TOIN field-level learning
                        )
                    )
                else:
                    event_data.append((event, None, None, None, None))

        # Process outside lock
        if event_data:
            feedback = get_compression_feedback()
            telemetry = get_telemetry_collector()
            toin = get_toin()

            for event, _tool_name, sig_hash, strategy, compressed_content in event_data:
                # Notify feedback system (pass strategy for success rate tracking)
                feedback.record_retrieval(event, strategy=strategy)

                # Extract query fields if present
                query_fields = None
                if event.query:
                    # Extract field:value patterns
                    query_fields = re.findall(r"(\w+)[=:]", event.query)

                # Notify telemetry for data flywheel
                try:
                    if sig_hash is not None:
                        telemetry.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query_fields=query_fields,
                        )
                except Exception:
                    # Telemetry should never break the feedback loop
                    logger.debug("Telemetry record_retrieval failed", exc_info=True)

                # Parse compressed content to extract items for TOIN field-level learning
                retrieved_items: list[dict[str, Any]] | None = None
                if compressed_content:
                    try:
                        parsed = json.loads(compressed_content)
                        # Handle both direct arrays and wrapped arrays
                        if isinstance(parsed, list):
                            # Filter to dicts only (field learning needs dict items)
                            retrieved_items = [item for item in parsed if isinstance(item, dict)]
                        elif isinstance(parsed, dict):
                            # Check for common wrapper patterns: {"items": [...], "results": [...]}
                            for key in ("items", "results", "data", "records"):
                                if key in parsed and isinstance(parsed[key], list):
                                    retrieved_items = [
                                        item for item in parsed[key] if isinstance(item, dict)
                                    ]
                                    break
                    except (json.JSONDecodeError, TypeError):
                        # Invalid JSON - skip field learning for this retrieval
                        pass

                # Notify TOIN for cross-user learning
                try:
                    if sig_hash is not None:
                        toin.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query=event.query,
                            query_fields=query_fields,
                            strategy=strategy,  # Pass strategy for success rate tracking
                            retrieved_items=retrieved_items,  # For field-level learning
                        )
                except Exception:
                    # TOIN should never break the feedback loop
                    logger.debug("TOIN record_retrieval failed", exc_info=True)


# Request-scoped store (for multi-tenant SaaS: one store per request/tenant)
_request_ccr_store: ContextVar[CompressionStore | None] = ContextVar(
    "headroom_request_ccr_store", default=None
)

# Global store instance (lazy initialization)
_compression_store: CompressionStore | None = None
_store_lock = threading.Lock()


def backend_is_stateless_safe(backend: Any) -> bool:
    """True when a stateless process may keep holding originals in ``backend``.

    One rule, applied wherever a backend object reaches the CCR store —
    ``_create_default_ccr_backend()``, an explicit ``backend=`` argument, a
    request-scoped tenant store, and the in-place swap on an already-built
    singleton:

    * a *process-local* backend (``is_process_local = True``) is safe: nothing
      leaves this process, which is what ``--stateless`` promises;
    * a backend that *declares itself off-machine* (``writes_local_disk =
      False``) is safe. That is the escape hatch for the external adapters
      (e.g. a ``redis`` backend registered under ``headroom.ccr_backend``):
      they write nothing to this machine's filesystem and are the only way to
      share retrieval across workers, so stateless mode keeps them;
    * anything else **fails closed**. ``SQLiteBackend`` says so explicitly, and
      an undeclared third-party backend is treated the same way, because from
      here there is no way to tell whether it is putting verbatim tool-result
      originals in a file. A backend that stores off-machine opts in with one
      class attribute.
    """
    if backend is None:
        return True  # CompressionStore() falls back to InMemoryBackend
    if getattr(backend, "is_process_local", False):
        return True
    declared = getattr(backend, "writes_local_disk", None)
    return declared is not None and not declared


def _enforce_stateless_backend(store: CompressionStore, source: str) -> CompressionStore:
    """Make ``store`` safe for a stateless process, in place. Returns ``store``.

    This is the enforcement boundary for the "stateless wins over every backend
    choice" invariant. ``_create_default_ccr_backend()`` alone cannot hold it:
    it is only consulted when this process builds the *unconfigured global*
    store, while ``get_compression_store()`` also hands back request-scoped
    (SaaS per-tenant) stores and honours an explicit ``backend=`` argument —
    either of which can be a SQLite or plugin backend created long after
    stateless mode was turned on.

    The store object is never replaced, only its backend, so callers that hold
    their own reference (``CompressionFeedback.store``, middleware that kept
    the tenant store it installed) stop writing to disk too.
    """
    from ..paths import process_is_stateless

    if not process_is_stateless() or store.backend_is_stateless_safe:
        return store  # hot path: one flag read and one backend attribute read
    previous_backend = store.backend_class_name
    if store.use_process_local_backend():
        logger.warning(
            "Stateless mode: %s CCR store was backed by %s, which may write "
            "tool-result originals to local disk. Swapped it for an in-process "
            "store; retrieval is not shared across workers and does not "
            "survive a restart. Use an external backend (it must declare "
            "`writes_local_disk = False`) to keep retrieval shared without "
            "writing to this machine's filesystem.",
            source,
            previous_backend,
        )
    return store


def set_request_compression_store(store: CompressionStore | None) -> None:
    """Set the compression store for the current request context.

    Used by middleware (e.g. SaaS) to provide a tenant-scoped store.
    When set, get_compression_store() returns this store instead of the global one.

    Stateless mode is enforced here, at installation, and not only on the way
    out of ``get_compression_store()``: middleware that keeps its own reference
    to the tenant store it just installed must not be able to write originals
    to disk through it.

    Args:
        store: CompressionStore to use for this request, or None to clear.
    """
    if store is not None:
        _enforce_stateless_backend(store, "request-scoped")
    _request_ccr_store.set(store)


def clear_request_compression_store() -> None:
    """Clear the request-scoped compression store."""
    _request_ccr_store.set(None)


# HEADROOM_CCR_BACKEND values that put entries on the local filesystem.
# "" is the unset default, which resolves to sqlite.
_LOCAL_DISK_CCR_BACKENDS = frozenset({"", "sqlite"})


def ccr_backend_writes_local_disk(backend_type: str | None) -> bool:
    """True when this ``HEADROOM_CCR_BACKEND`` value stores entries on local disk.

    Only the local-disk backends are what ``--stateless`` is about. The
    entry-point adapters (e.g. a ``redis`` backend) keep nothing on the
    machine's filesystem, and they are the only way to share retrieval across
    workers and restarts — precisely what a multi-worker stateless deployment
    needs — so stateless mode keeps them.
    """
    return (backend_type or "").strip().lower() in _LOCAL_DISK_CCR_BACKENDS


def _create_default_ccr_backend() -> CompressionStoreBackend | None:
    """Create a CCR backend from env (e.g. HEADROOM_CCR_BACKEND=redis).

    Default (env unset or "sqlite"): SQLiteBackend at workspace_dir()/ccr_store.db
    — restart-safe and shared across worker processes, which the
    session-scale idle TTL assumes.
    "memory" opts back into the in-process dict. Other values load
    adapters via setuptools entry point 'headroom.ccr_backend'.
    Returns None to use InMemoryBackend.

    Stateless mode overrides the *local-disk* backends only, and does so here
    as well as in the CLI because embedders reach this through
    ``ProxyConfig(stateless=True)`` without touching the environment. An
    external backend is left alone: ``--stateless`` is a promise about this
    machine's filesystem, not a reason to silently downgrade an operator's
    shared store to a per-worker dict.
    """
    from ..paths import process_is_stateless

    backend_type = (os.environ.get("HEADROOM_CCR_BACKEND") or "").strip().lower()
    stateless = process_is_stateless()

    if stateless and ccr_backend_writes_local_disk(backend_type):
        logger.info(
            "Stateless mode: CCR retrieval store is in-process only "
            "(HEADROOM_CCR_BACKEND=%s would write tool-result originals to "
            "%s). Retrieval will not survive a restart and is not shared "
            "across workers — set an external backend (e.g. "
            "HEADROOM_CCR_BACKEND=redis) to keep it shared without writing to "
            "local disk.",
            backend_type or "sqlite",
            "workspace_dir()/ccr_store.db",
        )
        return None
    if backend_type == "memory":
        return None
    if not backend_type or backend_type == "sqlite":
        try:
            from .backends.sqlite import SQLiteBackend

            return SQLiteBackend()
        except Exception as e:
            logger.warning(
                "Failed to initialize SQLite CCR backend (%s); "
                "falling back to in-memory store. Retrieval will not "
                "survive proxy restarts.",
                e,
            )
            return None
    try:
        from importlib.metadata import entry_points

        all_eps = entry_points(group="headroom.ccr_backend")
        ep = next((e for e in all_eps if e.name == backend_type), None)
        if ep is None:
            logger.warning(
                "HEADROOM_CCR_BACKEND=%s but no entry point headroom.ccr_backend[%s]",
                backend_type,
                backend_type,
            )
            return None
        fn = ep.load()
        kwargs = {
            "url": os.environ.get("HEADROOM_REDIS_URL", ""),
            "tenant_prefix": os.environ.get("HEADROOM_CCR_TENANT_PREFIX", ""),
        }
        backend: CompressionStoreBackend = fn(**kwargs)
        if stateless:
            if not backend_is_stateless_safe(backend):
                logger.warning(
                    "Stateless mode: CCR backend %r (%s) does not declare "
                    "`writes_local_disk = False`, so it may be putting "
                    "verbatim tool-result originals on this machine's disk. "
                    "Falling back to an in-process store. Adapters that store "
                    "off-machine should set that class attribute to be kept "
                    "under --stateless.",
                    backend_type,
                    type(backend).__name__,
                )
                close = getattr(backend, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001 - never fail startup over a close
                        logger.debug("Closing rejected CCR backend failed", exc_info=True)
                return None
            logger.info(
                "Stateless mode: keeping external CCR backend %r. It declares "
                "no local-disk writes and keeps retrieval shared across "
                "workers; --stateless does not downgrade it.",
                backend_type,
            )
        return backend
    except Exception as e:
        logger.warning("Failed to load CCR backend %s: %s", backend_type, e)
        return None


def get_compression_store(
    max_entries: int = DEFAULT_CCR_MAX_ENTRIES,
    default_ttl: int | None = None,
    backend: CompressionStoreBackend | None = None,
) -> CompressionStore:
    """Get the compression store instance.

    If a request-scoped store was set (e.g. by SaaS middleware), returns it.
    Otherwise uses lazy-initialized global singleton. Backend can be supplied
    explicitly or created from env (HEADROOM_CCR_BACKEND) when building the global.

    Stateless mode is enforced on the way out, on whichever store is returned
    (see ``_enforce_stateless_backend``). That is what makes "stateless wins
    over every backend choice" true of the request-scoped and explicit
    ``backend=`` paths, which never reach ``_create_default_ccr_backend()``.
    The store *object* is preserved — only its backend is swapped — so callers
    that compare identity, and callers that kept their own reference, both keep
    working.

    Args:
        max_entries: Maximum entries (only used on first call for global store).
        default_ttl: Default TTL (only used on first call for global store).
            When omitted, HEADROOM_CCR_TTL_SECONDS overrides the 86400-second default.
        backend: Custom storage backend (only used on first call for global store).
                 Defaults to InMemoryBackend if not provided; env backend used if backend is None.

    Returns:
        Request-scoped CompressionStore if set, else global CompressionStore instance.
    """
    request_store = _request_ccr_store.get()
    if request_store is not None:
        # Enforced on every hand-out, not just at installation: the tenant
        # store may have been put in the ContextVar directly, or before this
        # process entered stateless mode.
        return _enforce_stateless_backend(request_store, "request-scoped")

    global _compression_store
    if _compression_store is None:
        with _store_lock:
            if _compression_store is None:
                if backend is None:
                    backend = _create_default_ccr_backend()
                effective_default_ttl = (
                    default_ttl if default_ttl is not None else _get_env_default_ttl_seconds()
                )
                _compression_store = CompressionStore(
                    max_entries=max_entries,
                    default_ttl=effective_default_ttl,
                    backend=backend,
                )
    # Covers the explicitly supplied `backend=` (which bypasses
    # `_create_default_ccr_backend()` entirely) and a singleton built before
    # stateless mode was turned on.
    return _enforce_stateless_backend(_compression_store, "global")


def make_compression_store_stateless() -> bool:
    """Stop the live CCR store writing to disk, without destroying what is there.

    This is the *transition* half of stateless enforcement; the steady-state
    half is ``_enforce_stateless_backend()`` on every store hand-out. The
    documented lifecycle for a process that already holds a persistent
    singleton when a stateless proxy is built:

    1. **The store object survives.** Its backend is swapped in place (see
       ``CompressionStore.use_process_local_backend``). Components that
       captured the store earlier — notably ``CompressionFeedback.store`` —
       keep their reference and stop writing to disk with it. Clearing the
       module global instead would leave those writers persisting verbatim
       originals after the stateless proxy is up.
    2. **The database is left exactly as it is.** We do *not* call
       ``reset_compression_store()``: its ``clear()`` is ``DELETE FROM
       ccr_entries`` against a file shared with other worker processes that
       are still serving from it, and that exists precisely to outlive this
       process. Deleting it would be data loss for them.
    3. **Rows written before the switch stay on disk.** ``--stateless`` is
       forward-looking: it stops this process persisting new originals, it
       does not retroactively erase what an earlier stateful run wrote.
       Purging that file is an explicit operator action (delete
       ``workspace_dir()/ccr_store.db``), deliberately not something proxy
       startup does behind an operator's back.
    4. **Entries already in RAM are dropped**, not migrated to the new
       backend: a stateless process should not go on serving originals it was
       just told to stop touching.
    5. **Stores created later are not this function's job.** A per-request
       tenant store or an explicit ``backend=`` argument can appear long after
       startup; those are caught at the boundary, in
       ``set_request_compression_store()`` and ``get_compression_store()``.

    Returns True if a backend was actually swapped out.
    """
    with _store_lock:
        store = _compression_store

    swapped = store.use_process_local_backend() if store is not None else False

    # A request-scoped store (SaaS tenant store) that already exists at
    # proxy-construction time gets the same treatment. Ones created later go
    # through set_request_compression_store()/get_compression_store().
    request_store = _request_ccr_store.get()
    if request_store is not None and request_store is not store:
        swapped = request_store.use_process_local_backend() or swapped

    return swapped


def reset_compression_store() -> None:
    """Reset the global compression store. Mainly for testing.

    WARNING: this ``clear()``s the store first, which on the SQLite backend is
    ``DELETE FROM ccr_entries`` against a database shared with other worker
    processes. Do not call it to "drop the singleton" in a live proxy — use
    ``make_compression_store_stateless()`` for that.
    """
    global _compression_store

    with _store_lock:
        if _compression_store is not None:
            _compression_store.clear()
        _compression_store = None
