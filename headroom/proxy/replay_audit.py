"""Offline audit over the replay corpus written by :mod:`replay_capture`.

The corpus holds the request body that went upstream after every transform,
plus the usage block the provider billed for it. That is enough to answer
two questions without any network call:

* Calibration: how far the local token count of a forwarded body sits from
  the billed input, per model. The cache gates price decisions in local
  tokens, so a drift here scales every gate's error.
* Prefix stability: between consecutive turns of one session, which section
  of the forwarded body diverged first (tools, system, or a message index),
  and how the byte-stable prefix predicts the cache read the provider
  actually reported.

Everything here is pure over parsed records. Grouping keys on the Claude
Code session id carried in ``metadata.user_id`` and on the model, and sorts
by timestamp, because the capture writer did not keep files in write order
across restarts.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

_SESSION_RE = re.compile(r'"session_id"\s*:\s*"([0-9a-fA-F-]{36})"')
DEFAULT_EXCLUDED_MODEL_PREFIXES: tuple[str, ...] = ("claude-haiku",)


class TokenCounter(Protocol):
    """The two counting calls the audit needs from a tokenizer."""

    def count_messages(self, messages: list[dict[str, Any]]) -> int: ...

    def count_text(self, text: str) -> int: ...


@dataclass(frozen=True)
class ReplayTurn:
    """One captured request with the usage the provider billed for it."""

    timestamp: float
    request_id: str
    model: str
    body: dict[str, Any]
    usage: dict[str, Any]
    transforms_applied: tuple[str, ...] = ()

    @property
    def messages(self) -> list[dict[str, Any]]:
        messages = self.body.get("messages")
        return messages if isinstance(messages, list) else []

    @property
    def uncached_input(self) -> int:
        return int(self.usage.get("input_tokens", 0) or 0)

    @property
    def cache_read(self) -> int:
        return int(self.usage.get("cache_read_input_tokens", 0) or 0)

    @property
    def cache_write(self) -> int:
        return int(self.usage.get("cache_creation_input_tokens", 0) or 0)

    @property
    def billed_input(self) -> int:
        """Whole prompt as the provider saw it, every tier added up."""
        return self.uncached_input + self.cache_read + self.cache_write


def session_key(record: dict[str, Any]) -> str | None:
    """Claude Code session id from ``metadata.user_id``, or None."""
    request = record.get("request")
    if not isinstance(request, dict):
        return None
    metadata = request.get("metadata")
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get("user_id")
    if isinstance(user_id, dict):
        value = user_id.get("session_id")
        return str(value) if value else None
    if isinstance(user_id, str):
        match = _SESSION_RE.search(user_id)
        if match is not None:
            return match.group(1)
        try:
            parsed = json.loads(user_id)
        except (TypeError, ValueError):
            return None
        if isinstance(parsed, dict) and parsed.get("session_id"):
            return str(parsed["session_id"])
    return None


def turn_from_record(record: dict[str, Any]) -> ReplayTurn | None:
    """Project one capture record onto :class:`ReplayTurn`."""
    request = record.get("request")
    if not isinstance(request, dict):
        return None
    usage = record.get("response_usage")
    transforms = record.get("transforms_applied") or []
    return ReplayTurn(
        timestamp=float(record.get("timestamp") or 0.0),
        request_id=str(record.get("request_id") or ""),
        model=str(record.get("model") or request.get("model") or "?"),
        body=request,
        usage=usage if isinstance(usage, dict) else {},
        transforms_applied=tuple(str(t) for t in transforms),
    )


def group_sessions(
    records: Iterable[dict[str, Any]],
    *,
    min_turns: int = 1,
    model_filter: str | None = None,
    exclude_model_prefixes: tuple[str, ...] = DEFAULT_EXCLUDED_MODEL_PREFIXES,
) -> dict[tuple[str, str], list[ReplayTurn]]:
    """Group records into per-session turn lists sorted by timestamp.

    The key is ``(session_id, model)``. A session that switches model gets
    one lineage per model, which matches how the provider caches. Records
    without a session id are dropped, and so are models matching one of
    ``exclude_model_prefixes``. The default excludes the Haiku side calls
    Claude Code makes for titles and summaries, which are not lineages.
    """
    groups: dict[tuple[str, str], list[ReplayTurn]] = {}
    for record in records:
        key = session_key(record)
        if key is None:
            continue
        turn = turn_from_record(record)
        if turn is None:
            continue
        if model_filter and turn.model != model_filter:
            continue
        if any(turn.model.startswith(prefix) for prefix in exclude_model_prefixes):
            continue
        groups.setdefault((key, turn.model), []).append(turn)
    result: dict[tuple[str, str], list[ReplayTurn]] = {}
    for key, turns in groups.items():
        turns.sort(key=lambda t: t.timestamp)
        if len(turns) >= min_turns:
            result[key] = turns
    return result


@dataclass(frozen=True)
class Divergence:
    """Where two consecutive forwarded bodies stop being byte-identical."""

    section: str
    """``none``, ``tools``, ``system``, ``messages_tail``, ``messages``, or
    ``first`` (no prior turn). ``messages_tail`` means only the last message
    of the previous turn changed, which is the client rewriting its final
    turn, while ``messages`` is an edit deeper in history."""

    message_index: int
    """First differing message index for the two message sections, else -1."""

    shared_messages: int
    """Leading messages identical to the previous turn."""


def _strip_cache_control(value: Any) -> Any:
    """Copy of ``value`` without ``cache_control`` keys at any depth."""
    if isinstance(value, dict):
        return {k: _strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_control(v) for v in value]
    return value


def _same_bytes(a: Any, b: Any) -> bool:
    """Structural equality that ignores cache_control breakpoints.

    The provider hashes content without the breakpoint markers, so a client
    that moves its marker every turn still gets a full read. Plain equality
    runs first because it is the common case and far cheaper than stripping.
    """
    if a == b:
        return True
    return _strip_cache_control(a) == _strip_cache_control(b)


def prefix_divergence(previous: dict[str, Any] | None, current: dict[str, Any]) -> Divergence:
    """Compare two forwarded bodies in provider hashing order.

    The provider hashes tools, then system, then messages, so a change in an
    earlier section busts everything after it. Equality is structural over
    the parsed JSON with cache_control markers ignored.
    """
    if previous is None:
        return Divergence("first", -1, 0)
    if not _same_bytes(previous.get("tools"), current.get("tools")):
        return Divergence("tools", -1, 0)
    if not _same_bytes(previous.get("system"), current.get("system")):
        return Divergence("system", -1, 0)
    prev_messages = previous.get("messages") or []
    cur_messages = current.get("messages") or []
    shared = 0
    for prev_msg, cur_msg in zip(prev_messages, cur_messages, strict=False):
        if not _same_bytes(prev_msg, cur_msg):
            break
        shared += 1
    if shared < len(prev_messages):
        section = "messages_tail" if shared == len(prev_messages) - 1 else "messages"
        return Divergence(section, shared, shared)
    return Divergence("none", -1, shared)


@dataclass(frozen=True)
class TurnAudit:
    """Local and billed view of one turn side by side."""

    request_id: str
    timestamp: float
    local_tokens: int
    billed_input: int
    cache_read: int
    cache_write: int
    predicted_stable_tokens: int
    divergence: Divergence

    @property
    def billed_read_fraction(self) -> float:
        return self.cache_read / self.billed_input if self.billed_input else 0.0

    @property
    def predicted_read_fraction(self) -> float:
        return self.predicted_stable_tokens / self.local_tokens if self.local_tokens else 0.0


def _head_tokens(body: dict[str, Any], count_text: Callable[[str], int]) -> int:
    """Token estimate for the tools and system head of a body."""
    total = 0
    for key in ("tools", "system"):
        value = body.get(key)
        if value:
            total += count_text(json.dumps(value, sort_keys=True, default=str))
    return total


def audit_session(turns: list[ReplayTurn], tokenizer: TokenCounter) -> list[TurnAudit]:
    """Per-turn audit of one session in timestamp order."""
    audits: list[TurnAudit] = []
    previous: ReplayTurn | None = None
    for turn in turns:
        head = _head_tokens(turn.body, tokenizer.count_text)
        message_tokens = tokenizer.count_messages(turn.messages) if turn.messages else 0
        local = head + message_tokens
        divergence = prefix_divergence(previous.body if previous else None, turn.body)
        if divergence.section in ("none", "messages", "messages_tail"):
            stable = head + (
                tokenizer.count_messages(turn.messages[: divergence.shared_messages])
                if divergence.shared_messages
                else 0
            )
        else:
            stable = 0
        audits.append(
            TurnAudit(
                request_id=turn.request_id,
                timestamp=turn.timestamp,
                local_tokens=local,
                billed_input=turn.billed_input,
                cache_read=turn.cache_read,
                cache_write=turn.cache_write,
                predicted_stable_tokens=stable,
                divergence=divergence,
            )
        )
        previous = turn
    return audits


@dataclass
class AuditSummary:
    """Aggregate over any number of audited sessions."""

    sessions: int = 0
    turns: int = 0
    local_tokens: int = 0
    billed_input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    predicted_stable_tokens: int = 0
    divergence_by_section: Counter[str] = field(default_factory=Counter)

    @property
    def calibration(self) -> float:
        """Local tokens per billed token. 1.0 means the tokenizer is exact."""
        return self.local_tokens / self.billed_input if self.billed_input else 0.0

    @property
    def billed_read_fraction(self) -> float:
        return self.cache_read / self.billed_input if self.billed_input else 0.0

    @property
    def predicted_read_fraction(self) -> float:
        return self.predicted_stable_tokens / self.local_tokens if self.local_tokens else 0.0

    def add(self, audits: list[TurnAudit]) -> None:
        self.sessions += 1
        for audit in audits:
            self.turns += 1
            self.local_tokens += audit.local_tokens
            self.billed_input += audit.billed_input
            self.cache_read += audit.cache_read
            self.cache_write += audit.cache_write
            self.predicted_stable_tokens += audit.predicted_stable_tokens
            self.divergence_by_section[audit.divergence.section] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "turns": self.turns,
            "local_tokens": self.local_tokens,
            "billed_input": self.billed_input,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "calibration": round(self.calibration, 4),
            "billed_read_fraction": round(self.billed_read_fraction, 4),
            "predicted_read_fraction": round(self.predicted_read_fraction, 4),
            "divergence_by_section": dict(self.divergence_by_section),
        }
