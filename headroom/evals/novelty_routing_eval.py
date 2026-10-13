"""Offline novelty-routing eval (Workstream B, see docs/optimization-plan-2026-07-19.md).

Answers one question with real Claude Code transcripts and zero API calls: how
often would a masking rule hide a tool output that later actually mattered.
"needed later" is defined by three deterministic rules (re-fetch, CCR-hash
retrieve, rare-token overlap with a later assistant message), never by a
model call and never by guessing. The eval then scores several masking
policies, including the two real production transforms
(``cross_turn_dedup.dedup_blocks`` and ``ReadLifecycleManager``), against
those labels.

This script makes no network calls and never imports an embedding backend
at module load time. Steps 0 through 4 run against a plain JSONL corpus
with the stdlib only. The ``--with-embeddings`` flag is a stub path only,
see the module docstring on ``_embedding_stub_report`` for what it does
and does not do.

Run as::

    .venv/bin/python -m headroom.evals.novelty_routing_eval --corpus ~/.claude/projects
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom.transforms.cross_turn_dedup import DedupBlock, dedup_blocks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 0/1: corpus loading
# ---------------------------------------------------------------------------

# Below this many sessions or tool outputs, treat the corpus as too small to
# support a real result and mark the run a smoke test instead of pretending
# the numbers mean anything.
MIN_SESSIONS_FOR_REAL_RUN = 30
MIN_TOOL_OUTPUTS_FOR_REAL_RUN = 1000

# Rare-token rule (Step 2c). A token is "rare" if it appears in fewer than
# this fraction of all tool outputs in the corpus, counting one occurrence
# per tool output at most (document frequency, not raw term frequency).
RARE_TOKEN_DOC_FREQ_THRESHOLD = 0.01
# Number of shared rare tokens required between a tool output and a later
# assistant message for the overlap rule to fire.
RARE_TOKEN_MIN_OVERLAP = 2

# Step 3: a single rule accounting for more than this fraction of positives
# is flagged as a concentration risk (the label set might really be testing
# one heuristic, not three).
RULE_CONCENTRATION_FLAG_THRESHOLD = 0.90
# Step 3: sessions whose positive rate lands within this distance of 0 or 1
# are flagged as degenerate (all-positive or all-negative sessions distort
# per-session statistics and are worth a human look).
PER_SESSION_RATE_EPSILON = 0.02

# Step 4: turns-since-appearance thresholds for the mask-by-age baseline family.
MASK_BY_AGE_THRESHOLDS = (5, 10, 20, 40)
# Which mask-by-age variant the rate-matched random baseline mirrors.
RANDOM_BASELINE_MATCHES_AGE_THRESHOLD = 20
RANDOM_BASELINE_SEED = 20260719
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_RESAMPLES = 1000

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass
class TranscriptEvent:
    """One typed event in a session's ordered stream.

    ``kind`` is one of ``tool_output``, ``assistant_text``,
    ``compaction_boundary``, or ``tool_use`` (an assistant-issued tool call,
    kept separately from ``tool_output`` because the labeler needs both the
    call, for retrieve-style tool names and inputs, and the result).

    ``is_sidechain`` marks an event that belongs to a subagent's own nested
    transcript rather than the main thread. Sidechain events are kept in the
    stream (for corpus-wide stats) but excluded from main-thread-only rules
    like re-fetch detection, since a subagent re-reading a file its parent
    already read is not the same signal as the main agent repeating itself.
    """

    kind: str
    index: int
    is_sidechain: bool = False
    text: str = ""
    tool_name: str = ""
    tool_use_id: str = ""
    normalized_target: str = ""
    raw_input: str = ""


@dataclass
class SessionData:
    """One parsed session: its ordered event stream plus lookup indices."""

    session_id: str
    events: list[TranscriptEvent] = field(default_factory=list)


def _normalize_target(tool_name: str, raw_input: Any) -> str:
    """Fold a tool call into a comparable "same thing fetched again" key.

    Deliberately simple: lowercase the tool name, and for a dict input pull
    the first present field out of a short list of common target-shaped
    keys (path, file_path, query, command, url). Anything else falls back to
    a compact stringified form of the whole input. Volatile bits (an
    explicit "timestamp" or "cache_bust" key some tools carry) are dropped
    before stringifying, so two calls that differ only in a nonce still
    normalize to the same target. This is not a general path canonicalizer:
    it does not resolve symlinks or walk ``..`` segments, since transcript
    inputs are already whatever string the model typed.
    """
    name = (tool_name or "").strip().lower()
    if isinstance(raw_input, dict):
        cleaned = {k: v for k, v in raw_input.items() if k not in ("timestamp", "cache_bust")}
        for key in ("file_path", "path", "query", "command", "url", "pattern"):
            if key in cleaned and cleaned[key] is not None:
                return f"{name}:{cleaned[key]}"
        target = json.dumps(cleaned, sort_keys=True, default=str)
    else:
        target = str(raw_input)
    return f"{name}:{target}"


def _is_sidechain(entry: dict[str, Any]) -> bool:
    return bool(entry.get("isSidechain"))


def _is_compaction_boundary(entry: dict[str, Any]) -> bool:
    """True if this JSONL line marks a compaction boundary.

    Checked, in order: an ``isCompactSummary`` flag (entry-level or on
    ``message``), a ``type == "summary"`` entry, or a ``system`` entry whose
    ``subtype`` contains "compact". No sample session in the schema
    investigation actually contained one of these, so this path is
    defensive rather than exercised against real data, but a synthetic
    fixture drives it in tests.
    """
    if entry.get("isCompactSummary") or (entry.get("message") or {}).get("isCompactSummary"):
        return True
    if entry.get("type") == "summary":
        return True
    if entry.get("type") == "system" and "compact" in str(entry.get("subtype", "")).lower():
        return True
    return False


def _extract_text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def load_session(path: Path) -> SessionData:
    """Parse one JSONL session file into an ordered event stream.

    Reads the file line by line, preserving file order as the event order
    (per the schema notes, ``type: system`` entries with
    ``subtype: turn_duration`` and similar are structural, not content, and
    are skipped entirely rather than turned into events). Malformed JSON
    lines are skipped rather than raising, since a corpus walk over real
    ``~/.claude/projects`` data should not die on one bad line.
    """
    session_id = path.stem
    events: list[TranscriptEvent] = []
    # tool_use_id -> (tool_name, normalized_target, raw_input_str)
    pending_tool_use: dict[str, tuple[str, str, str]] = {}

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSONL line %d in %s", line_no, path)
                continue

            if _is_compaction_boundary(entry):
                events.append(TranscriptEvent(kind="compaction_boundary", index=len(events)))
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            sidechain = _is_sidechain(entry)
            message = entry.get("message") or {}
            role = message.get("role")
            content = message.get("content")

            if role == "assistant":
                for text in _extract_text_blocks(content):
                    if text.strip():
                        events.append(
                            TranscriptEvent(
                                kind="assistant_text",
                                index=len(events),
                                is_sidechain=sidechain,
                                text=text,
                            )
                        )
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_use":
                            continue
                        name = block.get("name", "")
                        tc_id = block.get("id", "")
                        raw_input = block.get("input", {})
                        normalized = _normalize_target(name, raw_input)
                        raw_input_str = json.dumps(raw_input, sort_keys=True, default=str)
                        if tc_id:
                            pending_tool_use[tc_id] = (name, normalized, raw_input_str)
                        events.append(
                            TranscriptEvent(
                                kind="tool_use",
                                index=len(events),
                                is_sidechain=sidechain,
                                tool_name=name,
                                tool_use_id=tc_id,
                                normalized_target=normalized,
                                raw_input=raw_input_str,
                            )
                        )

            elif role == "user":
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tc_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        text = "\n".join(_extract_text_blocks(result_content))
                    else:
                        text = str(result_content)
                    tool_name, normalized, _raw = pending_tool_use.get(
                        tc_id, ("", f"unknown:{tc_id}", "")
                    )
                    events.append(
                        TranscriptEvent(
                            kind="tool_output",
                            index=len(events),
                            is_sidechain=sidechain,
                            text=text,
                            tool_name=tool_name,
                            tool_use_id=tc_id,
                            normalized_target=normalized,
                        )
                    )

    return SessionData(session_id=session_id, events=events)


def walk_corpus(corpus_root: Path) -> list[SessionData]:
    """Load every session under ``<corpus_root>/*/*.jsonl``."""
    sessions: list[SessionData] = []
    if not corpus_root.exists():
        return sessions
    for project_dir in sorted(corpus_root.iterdir()):
        if not project_dir.is_dir():
            continue
        for session_path in sorted(project_dir.glob("*.jsonl")):
            try:
                sessions.append(load_session(session_path))
            except OSError as exc:
                logger.warning("could not read %s: %s", session_path, exc)
    return sessions


# ---------------------------------------------------------------------------
# Step 2: deterministic labeler
# ---------------------------------------------------------------------------

_HASH_TOKEN_RE = re.compile(r"\bhash=([0-9a-fA-F]{6,})")


@dataclass
class LabeledEvent:
    """A labeled ``tool_output`` event, tied back to its session and index."""

    session_id: str
    event_index: int
    normalized_target: str
    needed_later: bool
    rules_fired: list[str]
    snippet: str
    turns_since_prior_appearance_hint: int = 0


def _rare_token_doc_freq(sessions: list[SessionData]) -> dict[str, float]:
    """Corpus-wide document frequency of tokens across tool outputs.

    Document frequency here means: fraction of tool outputs (corpus-wide,
    across all sessions, main thread and sidechain both, since rarity is a
    corpus property) containing the token at least once. A token appearing
    in exactly one output out of a thousand has document frequency 0.001.
    """
    doc_count: Counter[str] = Counter()
    total_outputs = 0
    for session in sessions:
        for event in session.events:
            if event.kind != "tool_output":
                continue
            total_outputs += 1
            tokens = {t.lower() for t in _WORD_RE.findall(event.text)}
            doc_count.update(tokens)
    if total_outputs == 0:
        return {}
    return {tok: count / total_outputs for tok, count in doc_count.items()}


def label_session(
    session: SessionData, rare_token_doc_freq: dict[str, float]
) -> list[LabeledEvent]:
    """Label every ``tool_output`` event in one session.

    Positive (``needed_later=True``) if any of:

    (a) re-fetch: a later main-thread ``tool_output`` shares the same
        ``normalized_target``.
    (b) CCR-hash retrieve: the output text contains a ``hash=<hex>`` token
        (the real marker shape emitted by ``compression_store``/
        ``mcp_server.py``, see module docstring in the calling script's
        design doc) and a later ``tool_use`` event names a tool containing
        "retrieve" whose stringified input contains that same hash token.
        This is real string matching against the actual identifier format
        the production CCR marker uses, not a fabricated heuristic.
    (c) rare-token overlap: the output shares at least
        ``RARE_TOKEN_MIN_OVERLAP`` rare tokens (document frequency below
        ``RARE_TOKEN_DOC_FREQ_THRESHOLD``) with a later ``assistant_text``
        event.

    Rule (a) only considers main-thread events on both ends (a subagent
    re-fetching its own prior read is a different signal, see
    ``TranscriptEvent`` docstring). Rules (b) and (c) look at the whole
    ordered stream, since a hash marker or an assistant summary can
    legitimately span into or out of a sidechain.
    """
    labeled: list[LabeledEvent] = []
    events = session.events

    for i, event in enumerate(events):
        if event.kind != "tool_output":
            continue

        rules_fired: list[str] = []

        # Rule (a): re-fetch, main thread only.
        if not event.is_sidechain:
            for later in events[i + 1 :]:
                if later.is_sidechain:
                    continue
                if (
                    later.kind == "tool_output"
                    and later.normalized_target == event.normalized_target
                ):
                    rules_fired.append("refetch")
                    break

        # Rule (b): CCR hash retrieve.
        hash_matches = _HASH_TOKEN_RE.findall(event.text)
        if hash_matches:
            hash_set = set(hash_matches)
            for later in events[i + 1 :]:
                if later.kind != "tool_use":
                    continue
                if "retrieve" not in later.tool_name.lower():
                    continue
                if any(h in later.raw_input for h in hash_set):
                    rules_fired.append("ccr_retrieve")
                    break

        # Rule (c): rare-token overlap with a later assistant message.
        output_tokens = {t.lower() for t in _WORD_RE.findall(event.text)}
        rare_output_tokens = {
            t
            for t in output_tokens
            if rare_token_doc_freq.get(t, 1.0) < RARE_TOKEN_DOC_FREQ_THRESHOLD
        }
        if rare_output_tokens:
            for later in events[i + 1 :]:
                if later.kind != "assistant_text":
                    continue
                later_tokens = {t.lower() for t in _WORD_RE.findall(later.text)}
                if len(rare_output_tokens & later_tokens) >= RARE_TOKEN_MIN_OVERLAP:
                    rules_fired.append("rare_token_overlap")
                    break

        labeled.append(
            LabeledEvent(
                session_id=session.session_id,
                event_index=i,
                normalized_target=event.normalized_target,
                needed_later=bool(rules_fired),
                rules_fired=rules_fired,
                snippet=event.text[:200],
            )
        )

    return labeled


# ---------------------------------------------------------------------------
# Step 3: label audit
# ---------------------------------------------------------------------------


def audit_labels(all_labels: list[LabeledEvent]) -> dict[str, Any]:
    """Sample positives/negatives and compute the concentration and
    per-session-rate flags called for by the pre-registered design."""
    positives = [lbl for lbl in all_labels if lbl.needed_later]
    negatives = [lbl for lbl in all_labels if not lbl.needed_later]

    def _sample(items: list[LabeledEvent], n: int) -> list[dict[str, Any]]:
        return [
            {
                "session_id": lbl.session_id,
                "normalized_target": lbl.normalized_target,
                "rules_fired": lbl.rules_fired,
                "snippet": lbl.snippet,
            }
            for lbl in items[:n]
        ]

    rule_counts: Counter[str] = Counter()
    for lbl in positives:
        for rule in lbl.rules_fired:
            rule_counts[rule] += 1

    concentration_flag = None
    if positives:
        for rule, count in rule_counts.items():
            share = count / len(positives)
            if share > RULE_CONCENTRATION_FLAG_THRESHOLD:
                concentration_flag = {"rule": rule, "share": share}
                break

    per_session: dict[str, list[bool]] = defaultdict(list)
    for lbl in all_labels:
        per_session[lbl.session_id].append(lbl.needed_later)

    rates = {sid: (sum(flags) / len(flags) if flags else 0.0) for sid, flags in per_session.items()}
    rate_values = list(rates.values())
    degenerate_sessions = [
        sid
        for sid, rate in rates.items()
        if rate <= PER_SESSION_RATE_EPSILON or rate >= 1.0 - PER_SESSION_RATE_EPSILON
    ]

    return {
        "total_labeled": len(all_labels),
        "total_positive": len(positives),
        "total_negative": len(negatives),
        "sample_positives": _sample(positives, 20),
        "sample_negatives": _sample(negatives, 20),
        "rule_counts": dict(rule_counts),
        "rule_concentration_flag": concentration_flag,
        "per_session_rate_min": min(rate_values) if rate_values else None,
        "per_session_rate_max": max(rate_values) if rate_values else None,
        "per_session_rate_mean": (sum(rate_values) / len(rate_values)) if rate_values else None,
        "degenerate_session_count": len(degenerate_sessions),
        "degenerate_session_epsilon": PER_SESSION_RATE_EPSILON,
    }


# ---------------------------------------------------------------------------
# Step 4: baselines
# ---------------------------------------------------------------------------


def _tool_output_events_with_turn(session: SessionData) -> list[tuple[TranscriptEvent, int]]:
    """Main-thread tool_output events paired with a 1-based turn ordinal.

    The "turn" for age-based baselines is the position of the event among
    all main-thread events in the session (sidechain events do not consume
    a turn, since they are not part of the thread the age policies reason
    about).
    """
    out = []
    turn = 0
    for event in session.events:
        if event.is_sidechain:
            continue
        turn += 1
        if event.kind == "tool_output":
            out.append((event, turn))
    return out


def _session_current_turn(session: SessionData) -> int:
    return sum(1 for e in session.events if not e.is_sidechain)


def _mask_nothing(session: SessionData) -> set[int]:
    return set()


def _mask_by_age(session: SessionData, threshold: int) -> set[int]:
    """Mask any tool_output whose turn distance from the end of the session
    exceeds ``threshold``."""
    current_turn = _session_current_turn(session)
    masked = set()
    for event, turn in _tool_output_events_with_turn(session):
        if current_turn - turn > threshold:
            masked.add(event.index)
    return masked


def _mask_random_matched(session: SessionData, rate: float, rng: random.Random) -> set[int]:
    masked = set()
    for event in session.events:
        if event.kind != "tool_output":
            continue
        if rng.random() < rate:
            masked.add(event.index)
    return masked


def _build_dedup_blocks(session: SessionData) -> list[DedupBlock]:
    """Build the ``DedupBlock`` list the real production transform expects.

    ``turn`` is the event's position among main-thread tool_output events
    (stable, increasing, matches how ``dedup_blocks`` expects an absolute
    ordinal), so it's consistent with how the transform is used at runtime,
    where blocks are indexed as they occur in the request.
    """
    blocks = []
    for event, turn in _tool_output_events_with_turn(session):
        blocks.append(DedupBlock(text=event.text, turn=turn, protected=False))
    return blocks


def _mask_by_dedup(session: SessionData) -> set[int]:
    pairs = _tool_output_events_with_turn(session)
    if not pairs:
        return set()
    blocks = _build_dedup_blocks(session)
    try:
        new_blocks, _stats = dedup_blocks(blocks)
    except Exception:  # dedup_blocks documents "never raises" but stay defensive
        logger.warning("dedup_blocks raised unexpectedly for session %s", session.session_id)
        return set()
    masked = set()
    for (event, _turn), original, rewritten in zip(pairs, blocks, new_blocks):
        if rewritten.text != original.text:
            masked.add(event.index)
    return masked


def _build_anthropic_messages(session: SessionData) -> list[dict[str, Any]]:
    """Reconstruct a minimal Anthropic-shaped messages list from a session's
    main-thread events, enough for ``ReadLifecycleManager`` to classify Read
    outputs as stale or superseded. Only carries the fields the manager
    reads: tool_use blocks with name/id/input, and tool_result blocks with
    tool_use_id/content.
    """
    messages: list[dict[str, Any]] = []
    pending_assistant: list[dict[str, Any]] = []

    def _flush_assistant() -> None:
        if pending_assistant:
            messages.append({"role": "assistant", "content": list(pending_assistant)})
            pending_assistant.clear()

    for event in session.events:
        if event.is_sidechain:
            continue
        if event.kind == "tool_use":
            pending_assistant.append(
                {
                    "type": "tool_use",
                    "id": event.tool_use_id or f"tu_{event.index}",
                    "name": event.tool_name,
                    "input": _parse_raw_input(event.raw_input),
                }
            )
        elif event.kind == "assistant_text":
            pending_assistant.append({"type": "text", "text": event.text})
        elif event.kind == "tool_output":
            _flush_assistant()
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": event.tool_use_id or f"tu_{event.index}",
                            "content": event.text,
                        }
                    ],
                }
            )
    _flush_assistant()
    return messages


def _parse_raw_input(raw_input: str) -> dict[str, Any]:
    if not raw_input:
        return {}
    try:
        parsed = json.loads(raw_input)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _mask_by_read_lifecycle(session: SessionData) -> set[int]:
    """A tool_output counts as masked if ReadLifecycleManager rewrote its
    Read into a stale/superseded marker. Imported lazily inside this
    function's caller graph is unnecessary here (this module is not the
    flagged embeddings path), but the transform is only invoked, never
    reimplemented, per the workstream constraint.
    """
    from headroom.config import ReadLifecycleConfig
    from headroom.transforms.read_lifecycle import ReadLifecycleManager

    messages = _build_anthropic_messages(session)
    if not messages:
        return set()

    tool_use_id_by_output_index: dict[str, int] = {}
    original_text_by_tc_id: dict[str, str] = {}
    for event in session.events:
        if event.kind == "tool_output" and not event.is_sidechain:
            tc_id = event.tool_use_id or f"tu_{event.index}"
            tool_use_id_by_output_index[tc_id] = event.index
            original_text_by_tc_id[tc_id] = event.text

    manager = ReadLifecycleManager(ReadLifecycleConfig())
    try:
        result = manager.apply(messages)
    except Exception:
        logger.warning("ReadLifecycleManager.apply raised for session %s", session.session_id)
        return set()

    masked = set()
    if not result.transforms_applied:
        return masked
    # A tool_result counts as masked if the manager rewrote its content
    # away from what the session originally had, keyed by tool_use_id
    # rather than by guessing the marker's wording (the manager's replaced
    # bytes_after is always smaller than bytes_before by construction, but
    # comparing text directly is the honest way to tell "this one changed").
    for msg in result.messages:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tc_id = block.get("tool_use_id", "")
            content = block.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content)
            original = original_text_by_tc_id.get(tc_id)
            if original is not None and text != original:
                idx = tool_use_id_by_output_index.get(tc_id)
                if idx is not None:
                    masked.add(idx)
    return masked


@dataclass
class BaselineResult:
    name: str
    per_1000_point: float
    per_1000_ci_low: float
    per_1000_ci_high: float
    mask_rate: float


def _per_1000_needed_but_masked(
    sessions_subset: list[SessionData],
    labels_by_session: dict[str, dict[int, bool]],
    masked_indices_by_session: dict[str, set[int]],
) -> float:
    needed_but_masked = 0
    total_outputs = 0
    for session in sessions_subset:
        labels = labels_by_session.get(session.session_id, {})
        masked = masked_indices_by_session.get(session.session_id, set())
        for idx, needed in labels.items():
            total_outputs += 1
            if needed and idx in masked:
                needed_but_masked += 1
    if total_outputs == 0:
        return 0.0
    return (needed_but_masked / total_outputs) * 1000.0


def _mask_rate(
    sessions_subset: list[SessionData], masked_indices_by_session: dict[str, set[int]]
) -> float:
    total = 0
    masked_total = 0
    for session in sessions_subset:
        for event in session.events:
            if event.kind != "tool_output":
                continue
            total += 1
            if event.index in masked_indices_by_session.get(session.session_id, set()):
                masked_total += 1
    return masked_total / total if total else 0.0


def _bootstrap_ci(
    sessions: list[SessionData],
    labels_by_session: dict[str, dict[int, bool]],
    masked_indices_by_session: dict[str, set[int]],
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    """Cluster bootstrap over sessions: resample sessions with replacement,
    recompute the per-1000 metric each time, report the 2.5/97.5 percentile
    interval. Resampling at session granularity (not event granularity)
    respects that events within a session are not independent draws."""
    if not sessions:
        return (0.0, 0.0)
    if len(sessions) == 1:
        # Resampling one session with replacement returns that session every
        # time, so the interval collapses to a point that looks like perfect
        # certainty. NaN bounds force downstream readers to treat n=1 as
        # "no confidence interval" instead of citing a zero-width one.
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(sessions)
    values = []
    for _ in range(resamples):
        resample = [sessions[rng.randrange(n)] for _ in range(n)]
        values.append(
            _per_1000_needed_but_masked(resample, labels_by_session, masked_indices_by_session)
        )
    values.sort()
    lo_idx = int(0.025 * len(values))
    hi_idx = min(int(0.975 * len(values)), len(values) - 1)
    return (values[lo_idx], values[hi_idx])


def run_baselines(
    sessions: list[SessionData],
    labels_by_session: dict[str, dict[int, bool]],
    bootstrap_resamples: int,
) -> list[BaselineResult]:
    baseline_masks: dict[str, dict[str, set[int]]] = {}

    baseline_masks["mask_nothing"] = {s.session_id: _mask_nothing(s) for s in sessions}
    for threshold in MASK_BY_AGE_THRESHOLDS:
        name = f"mask_by_age_{threshold}"
        baseline_masks[name] = {s.session_id: _mask_by_age(s, threshold) for s in sessions}

    matched_name = f"mask_by_age_{RANDOM_BASELINE_MATCHES_AGE_THRESHOLD}"
    matched_rate = _mask_rate(sessions, baseline_masks[matched_name])
    rng = random.Random(RANDOM_BASELINE_SEED)
    baseline_masks["random_matched_rate"] = {
        s.session_id: _mask_random_matched(s, matched_rate, rng) for s in sessions
    }

    baseline_masks["production_dedup"] = {s.session_id: _mask_by_dedup(s) for s in sessions}
    baseline_masks["production_read_lifecycle"] = {
        s.session_id: _mask_by_read_lifecycle(s) for s in sessions
    }

    results = []
    for name, masked_by_session in baseline_masks.items():
        point = _per_1000_needed_but_masked(sessions, labels_by_session, masked_by_session)
        ci_low, ci_high = _bootstrap_ci(
            sessions, labels_by_session, masked_by_session, BOOTSTRAP_SEED, bootstrap_resamples
        )
        results.append(
            BaselineResult(
                name=name,
                per_1000_point=point,
                per_1000_ci_low=ci_low,
                per_1000_ci_high=ci_high,
                mask_rate=_mask_rate(sessions, masked_by_session),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Steps 5-8: embeddings stubs, only reachable behind --with-embeddings
# ---------------------------------------------------------------------------


def _embedding_stub_report() -> dict[str, Any]:
    """Placeholder for candidates 1 to 4 of the embeddings-based extension.

    Not implemented. When wired up, this would lazily import
    ``headroom.relevance.embedding`` and compute: candidate 1, cosine
    similarity between a tool output's embedding and later assistant
    message embeddings, candidate 2, a query-aware variant that also
    embeds the user's original request, candidate 3, the comparison
    plumbing to line this up against the rule-based labels from step 2,
    and candidate 4, a verdict section stating whether the embedding
    signal adds anything the deterministic rules did not already catch.
    None of that is implemented here. This function only imports the
    embedding module (to prove the import boundary is real and lazy) and
    returns a stub result marked not implemented.
    """
    from headroom import relevance  # noqa: F401  (import boundary check only)
    from headroom.relevance import embedding  # noqa: F401

    return {
        "implemented": False,
        "candidates": {
            "candidate_1_cosine_similarity": "not implemented, would call relevance/embedding.py",
            "candidate_2_query_aware_variant": "not implemented, would call relevance/embedding.py",
            "candidate_3_comparison_plumbing": "not implemented, would call relevance/embedding.py",
            "candidate_4_verdict_section": "not implemented, would call relevance/embedding.py",
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_eval(corpus_root: Path, bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    sessions = walk_corpus(corpus_root)
    total_tool_outputs = sum(1 for s in sessions for e in s.events if e.kind == "tool_output")
    smoke_test_only = (
        len(sessions) < MIN_SESSIONS_FOR_REAL_RUN
        or total_tool_outputs < MIN_TOOL_OUTPUTS_FOR_REAL_RUN
    )
    if smoke_test_only:
        logger.warning(
            "corpus has %d sessions / %d tool outputs, below the %d/%d real-run "
            "gate, treating this run as a labeled smoke test only",
            len(sessions),
            total_tool_outputs,
            MIN_SESSIONS_FOR_REAL_RUN,
            MIN_TOOL_OUTPUTS_FOR_REAL_RUN,
        )

    rare_doc_freq = _rare_token_doc_freq(sessions)

    all_labels: list[LabeledEvent] = []
    labels_by_session: dict[str, dict[int, bool]] = {}
    for session in sessions:
        session_labels = label_session(session, rare_doc_freq)
        all_labels.extend(session_labels)
        labels_by_session[session.session_id] = {
            lbl.event_index: lbl.needed_later for lbl in session_labels
        }

    audit = audit_labels(all_labels)
    baselines = run_baselines(sessions, labels_by_session, bootstrap_resamples)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(corpus_root),
        "smoke_test_only": smoke_test_only,
        "corpus_survey": {
            "sessions": len(sessions),
            "tool_outputs": total_tool_outputs,
            "min_sessions_for_real_run": MIN_SESSIONS_FOR_REAL_RUN,
            "min_tool_outputs_for_real_run": MIN_TOOL_OUTPUTS_FOR_REAL_RUN,
        },
        "label_audit": audit,
        "baselines": [
            {
                "name": b.name,
                "needed_but_masked_per_1000": b.per_1000_point,
                "ci_2.5": b.per_1000_ci_low,
                "ci_97.5": b.per_1000_ci_high,
                "mask_rate": b.mask_rate,
            }
            for b in baselines
        ],
        "bootstrap_resamples": bootstrap_resamples,
        "methodology_notes": {
            "ccr_hash_rule": (
                "Rule 2(b) regex-extracts hash=<hex> tokens from tool output text "
                "(the real marker shape emitted by compression_store.py / mcp_server.py) "
                "and checks whether a later tool_use event names a tool containing "
                "'retrieve' whose stringified input contains that same hash token. This "
                "is honest best-effort string matching against a real, introspectable "
                "mechanism's real identifier format. It is process-local at runtime (an "
                "in-memory store), so it cannot be replayed exactly offline from JSONL "
                "alone, it only detects the case where the transcript itself shows both "
                "the hash-bearing marker and a later retrieve call referencing it. If a "
                "corpus has no such markers, this rule fires zero times."
            ),
            "rare_token_definition": (
                f"a token is rare if it appears in fewer than "
                f"{RARE_TOKEN_DOC_FREQ_THRESHOLD} of all tool outputs corpus-wide "
                f"(document frequency, one count per output). Overlap rule fires at "
                f"{RARE_TOKEN_MIN_OVERLAP} or more shared rare tokens with a later "
                f"assistant text message."
            ),
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    survey = report["corpus_survey"]
    print("Novelty routing eval (Workstream B)")
    print(f"  corpus: {report['corpus_root']}")
    print(f"  sessions: {survey['sessions']}, tool_outputs: {survey['tool_outputs']}")
    if report["smoke_test_only"]:
        print("  WARNING: smoke_test_only=true, corpus below the real-run size gate")
        print("  all numbers below are a labeled smoke test, not a real result")
    audit = report["label_audit"]
    print(
        f"  labels: {audit['total_positive']} positive / {audit['total_negative']} negative "
        f"of {audit['total_labeled']} total"
    )
    if audit["rule_concentration_flag"]:
        flag = audit["rule_concentration_flag"]
        print(f"  FLAG: rule '{flag['rule']}' accounts for {flag['share']:.0%} of positives")
    print(f"  degenerate sessions (near 0/1 positive rate): {audit['degenerate_session_count']}")
    print("  baselines (needed-but-masked per 1000 tool outputs):")
    for b in report["baselines"]:
        print(
            f"    {b['name']:28s} {b['needed_but_masked_per_1000']:8.2f} "
            f"[{b['ci_2.5']:.2f}, {b['ci_97.5']:.2f}]  mask_rate={b['mask_rate']:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline novelty-routing eval over Claude Code session transcripts"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="~/.claude/projects",
        help="Root directory of <project>/<session>.jsonl transcripts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report path (default: headroom/evals/reports/novelty_routing_eval.json)",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=BOOTSTRAP_RESAMPLES,
        help="Number of session-level bootstrap resamples for the confidence interval",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Attach the (unimplemented) embeddings-based extension stub to the report",
    )
    args = parser.parse_args()

    corpus_root = Path(args.corpus).expanduser()
    report = run_eval(corpus_root, bootstrap_resamples=args.bootstrap_resamples)

    if args.with_embeddings:
        report["embeddings_extension"] = _embedding_stub_report()

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "reports" / "novelty_routing_eval.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(report, fh, indent=2)

    _print_summary(report)
    print(f"\nfull report written to {output_path}")


if __name__ == "__main__":
    main()
