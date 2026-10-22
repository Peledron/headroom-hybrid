"""Cache-gated masking of old, large Anthropic tool observations."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MASK_PREFIX = "[Tool result masked:"
INPUT_MASK_PREFIX = "[Tool input masked:"
_RECOVERY_MARKER = "Retrieve original: hash="
_EXCERPT_CHARS = 120

# tool_use input keys that MAY be masked when a caller explicitly opts in.
# DEFAULT OFF (discover_candidates masks no input keys unless asked): tool
# inputs are MODEL-authored, and masking them plants marker text in the
# position the model generates into. On 2026-07-17 this caused live file
# corruption: after a masked rebase, the model emitted a fabricated
# "[Tool input masked: ...]" marker as a Write's content, the file landed
# on disk as the marker, and the invented hash was unrecoverable. Tool
# RESULTS are environment-authored and stay safe to mask.
_MASKABLE_INPUT_KEYS = ("content", "file_text", "new_string", "old_string")

# How many assistant turns an assistant TEXT block must age before
# sweep_assistant_text may replace it with a CCR marker.
#
# This guards model-authored content, not tool results, so it is deliberately
# conservative and stays at 3. The 2026-07-17 mimicry incident came from
# marker text planted in a position the model can imitate on its next turn
# (see the module docstring and tests/test_breaker3_sweep_chaos.py). Pulling
# the floor closer to the tail moves that text further into the model's
# imitation window, and assistant prose is only 8.4 percent of appended
# content (docs/rewrite-mechanisms-2026-07-25.md), so there is little to buy.
#
# Tool-result masking has no turn-age gate of its own. It is priced by
# masking_gate_gain instead, which is the correct shape: it compares the
# rewrite cost against the read saving rather than guessing from position.
ASSISTANT_TEXT_SWEEP_AGE = 3


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    """One eligible Anthropic tool result and its prepared marker."""

    message_index: int
    block_index: int
    tool_use_id: str
    tool_name: str
    original: str
    marker: str
    content_hash: str
    original_tokens: int
    marker_tokens: int
    original_bytes: int
    input_key: str | None = None
    """None targets tool_result content; a key name targets that field of a
    tool_use block's input dict instead."""
    text_block: bool = False
    """True when the tool_result content is the Claude Code block-list form
    [{"type": "text", "text": ...}] rather than a plain string. The marker
    replaces the inner text, preserving sibling keys like cache_control."""
    closed_episode: bool = False
    """True when the closed-episode relaxation admitted this block rather than
    the turn-age gate. Recorded on the CCR entry so retrieval rate can be
    compared per gate (see CompressionStore._mask_gate_stats)."""

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.marker_tokens)


@dataclass(frozen=True, slots=True)
class MaskResult:
    messages: list[dict[str, Any]]
    masked_count: int = 0
    tokens_saved: int = 0
    bytes_saved: int = 0
    swept_count: int = 0


def masking_gate_gain(
    candidates: list[MaskCandidate],
    *,
    compression_policy: Any,
    suffix_tokens: int,
    expected_reads: float,
    p_alive: float,
    write_multiplier: float,
) -> float:
    """Price an exact masking batch through the shared mutation formula."""
    delta_t = sum(candidate.tokens_saved for candidate in candidates)
    return float(
        compression_policy.net_mutation_gain(
            delta_t,
            suffix_tokens,
            expected_reads,
            p_alive,
            write_multiplier=write_multiplier,
        )
    )


def _already_compact(content: str) -> bool:
    stripped = content.lstrip()
    return (
        stripped.startswith(MASK_PREFIX)
        or stripped.startswith(INPUT_MASK_PREFIX)
        or _RECOVERY_MARKER in content
        or "<<ccr:" in content
        or stripped.startswith("[Read content stale:")
        or stripped.startswith("[Read content superseded:")
        or stripped.startswith("[Read content matured:")
    )


def sweep_history(
    messages: list[dict[str, Any]],
    *,
    router: Any,
    tokenizer: Any,
    compression_store: Any,
    sweep_assistant_text: bool = False,
    minimum_savings_fraction: float = 0.15,
) -> MaskResult:
    """Compress old result residue after an admitted cache-busting mutation."""
    from headroom.transforms.compression_units import (
        CompressionUnit,
        compress_unit_with_router,
    )

    metadata, assistant_turns = _tool_metadata(messages)
    replacements: dict[tuple[int, int], tuple[str, str, bool]] = {}
    tokens_saved = 0
    bytes_saved = 0

    def prepare(
        *,
        message_index: int,
        block_index: int,
        original: str,
        role: str,
        item_type: str,
        tool_name: str,
        tool_use_id: str,
        text_block: bool,
    ) -> None:
        nonlocal tokens_saved, bytes_saved
        if not original or _already_compact(original):
            return
        unit = CompressionUnit(
            text=original,
            provider="anthropic",
            endpoint="/v1/messages",
            role=role,
            item_type=item_type,
            min_bytes=0,
            metadata={"compress_assistant": "true"} if role == "assistant" else {},
        )
        result = compress_unit_with_router(unit, router=router, tokenizer=tokenizer)
        if not result.modified:
            return
        try:
            encoded = original.encode("utf-8")
        except UnicodeEncodeError:
            return
        content_hash = hashlib.sha256(encoded).hexdigest()[:24]
        marker = f"<<ccr:{content_hash},string,{len(encoded)}B>>"
        replacement = f"{result.compressed.rstrip()}\n{marker}"
        replacement_bytes = len(replacement.encode("utf-8"))
        if replacement_bytes > len(encoded) * (1.0 - minimum_savings_fraction):
            return
        replacement_tokens = max(0, int(tokenizer.count_text(replacement)))
        original_tokens = max(0, int(tokenizer.count_text(original)))
        if replacement_tokens >= original_tokens:
            return
        try:
            compression_store.store(
                original=original,
                compressed=replacement,
                original_tokens=original_tokens,
                compressed_tokens=replacement_tokens,
                tool_name=tool_name,
                tool_call_id=tool_use_id,
                compression_strategy="history_sweep",
                explicit_hash=content_hash,
            )
        except Exception as exc:  # noqa: BLE001
            # A transient store fault must cost one block, not the whole
            # request. Skipping keeps the invariant: no marker without a
            # stored original.
            logger.warning(
                "history_sweep: CCR store failed for %s: %s", tool_use_id, exc
            )
            return
        replacements[(message_index, block_index)] = (
            original,
            replacement,
            text_block,
        )
        tokens_saved += original_tokens - replacement_tokens
        bytes_saved += len(encoded) - replacement_bytes

    assistant_turn = 0
    for message_index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            if (
                sweep_assistant_text
                and role == "assistant"
                and isinstance(content, str)
                and assistant_turns - assistant_turn - 1 >= ASSISTANT_TEXT_SWEEP_AGE
            ):
                prepare(
                    message_index=message_index,
                    block_index=-1,
                    original=content,
                    role="assistant",
                    item_type="text",
                    tool_name="assistant",
                    tool_use_id="",
                    text_block=False,
                )
            if role == "assistant":
                assistant_turn += 1
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                original = block.get("content")
                text_block = False
                if (
                    isinstance(original, list)
                    and len(original) == 1
                    and isinstance(original[0], dict)
                    and original[0].get("type") == "text"
                    and isinstance(original[0].get("text"), str)
                ):
                    original = original[0]["text"]
                    text_block = True
                if not isinstance(original, str):
                    continue
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    tool_use_id = ""
                tool_name, source_turn = metadata.get(
                    tool_use_id, ("unknown", assistant_turns)
                )
                if assistant_turns - source_turn - 1 < 1:
                    continue
                prepare(
                    message_index=message_index,
                    block_index=block_index,
                    original=original,
                    role="tool",
                    item_type="tool_result",
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    text_block=text_block,
                )
            elif (
                sweep_assistant_text
                and role == "assistant"
                and block_type == "text"
                and isinstance(block.get("text"), str)
                and assistant_turns - assistant_turn - 1 >= ASSISTANT_TEXT_SWEEP_AGE
            ):
                prepare(
                    message_index=message_index,
                    block_index=block_index,
                    original=block["text"],
                    role="assistant",
                    item_type="text",
                    tool_name="assistant",
                    tool_use_id="",
                    text_block=True,
                )
        if role == "assistant":
            assistant_turn += 1

    if not replacements:
        return MaskResult(messages=messages)
    output = list(messages)
    for (message_index, block_index), (original, replacement, text_block) in replacements.items():
        message = output[message_index]
        if block_index == -1:
            if message.get("content") != original:
                continue
            output[message_index] = {**message, "content": replacement}
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        block = content[block_index]
        if not isinstance(block, dict):
            continue
        new_block = block
        if block.get("type") == "tool_result":
            block_content = block.get("content")
            if text_block:
                if not (
                    isinstance(block_content, list)
                    and len(block_content) == 1
                    and isinstance(block_content[0], dict)
                    and block_content[0].get("text") == original
                ):
                    continue
                new_block = {
                    **block,
                    "content": [{**block_content[0], "text": replacement}],
                }
            elif block_content == original:
                new_block = {**block, "content": replacement}
            else:
                continue
        elif block.get("type") == "text" and block.get("text") == original:
            new_block = {**block, "text": replacement}
        else:
            continue
        new_content = list(content)
        new_content[block_index] = new_block
        output[message_index] = {**message, "content": new_content}
    return MaskResult(
        messages=output,
        tokens_saved=tokens_saved,
        bytes_saved=bytes_saved,
        swept_count=len(replacements),
    )


def _excerpt(content: str) -> str:
    line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    normalized = " ".join(line.split())
    if len(normalized) > _EXCERPT_CHARS:
        normalized = normalized[: _EXCERPT_CHARS - 3] + "..."
    return json.dumps(normalized, ensure_ascii=True)


def _tool_metadata(messages: list[dict[str, Any]]) -> tuple[dict[str, tuple[str, int]], int]:
    metadata: dict[str, tuple[str, int]] = {}
    assistant_turn = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_use_id = block.get("id")
                if isinstance(tool_use_id, str) and tool_use_id:
                    name = block.get("name")
                    metadata[tool_use_id] = (
                        name if isinstance(name, str) and name else "unknown",
                        assistant_turn,
                    )
        assistant_turn += 1
    return metadata, assistant_turn


def closed_episode_turn(messages: list[dict[str, Any]]) -> int | None:
    """Assistant-turn index at which the most recent finished task ended.

    A new user ask closes the task before it. Everything that task produced is
    finished work: the model has been told what to do next and will not be
    reasoning from those tool results again, whatever their age in turns.

    Returns None when the conversation has no closed task yet, either because
    the last message is not a user ask or because the ask is the first thing in
    the conversation. A user message that only carries tool results is the
    client returning work the model asked for, not a new ask, so it does not
    close anything.
    """
    boundary_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            kinds = {b.get("type") for b in content if isinstance(b, dict)}
            if "tool_result" in kinds:
                # Tool results coming back mid-task, not a new instruction.
                return None
        boundary_index = index
        break

    if boundary_index is None or boundary_index == 0:
        return None

    _, closed_turns = _tool_metadata(messages[:boundary_index])
    return closed_turns or None


def discover_candidates(
    messages: list[dict[str, Any]],
    *,
    count_tokens: Callable[[str], int],
    mask_after_turns: int = 3,
    mask_min_tokens: int = 400,
    mask_input_keys: tuple[str, ...] = (),
    episode_closed_turn: int | None = None,
    skip_result_tools: frozenset[str] = frozenset(),
) -> list[MaskCandidate]:
    """Prepare eligible markers without mutating messages or writing CCR.

    ``skip_result_tools`` names tools whose results another transform owns.
    The handler passes the Read tools while read maturation runs: maturation
    holds a fresh Read for ``quiesce_turns`` quiet turns before replacing it,
    and masking the same block first would cut that wait short.

    ``episode_closed_turn`` names the assistant turn where the last finished
    task ended (see ``closed_episode_turn``). Results produced before it skip
    the turn-age gate: that gate is a proxy for "the model has moved on", and a
    closed task is the direct evidence the proxy stands in for. It only relaxes
    the age test, never the size or already-compact tests.
    """
    metadata, assistant_turns = _tool_metadata(messages)
    candidates: list[MaskCandidate] = []

    def prepare(
        message_index: int,
        block_index: int,
        tool_use_id: str,
        tool_name: str,
        source_turn: int,
        original: str,
        input_key: str | None,
        text_block: bool = False,
    ) -> MaskCandidate | None:
        if _already_compact(original):
            return None
        if input_key is None and tool_name in skip_result_tools:
            return None
        # Results only. A tool input sits where the model generates, and the
        # 2026-07-17 mimicry incident came from marker text landing there; the
        # age gate is part of what keeps that text away from the tail, so it
        # does not get relaxed for inputs however cold the task is.
        in_closed_episode = (
            input_key is None
            and episode_closed_turn is not None
            and source_turn < episode_closed_turn
        )
        if not in_closed_episode and assistant_turns - source_turn - 1 < max(0, mask_after_turns):
            return None
        original_tokens = max(0, int(count_tokens(original)))
        if original_tokens < max(0, mask_min_tokens):
            return None
        try:
            encoded = original.encode("utf-8")
        except UnicodeEncodeError:
            # Unpaired surrogates cannot round-trip through the CCR
            # store, so the block is left untouched.
            return None
        original_bytes = len(encoded)
        content_hash = hashlib.sha256(encoded).hexdigest()[:24]
        if input_key is None:
            marker = (
                f"[Tool result masked: tool={tool_name}, tokens={original_tokens}, "
                f"bytes={original_bytes}, head={_excerpt(original)}. Not deleted, recover "
                f"the full text by requesting: Retrieve original: hash={content_hash}]"
            )
        else:
            marker = (
                f"[Tool input masked: tool={tool_name}, key={input_key}, "
                f"tokens={original_tokens}, bytes={original_bytes}, "
                f"head={_excerpt(original)}. Not deleted, recover the full text by "
                f"requesting: Retrieve original: hash={content_hash}]"
            )
        marker_tokens = max(0, int(count_tokens(marker)))
        if marker_tokens >= original_tokens:
            return None
        return MaskCandidate(
            message_index=message_index,
            block_index=block_index,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            original=original,
            marker=marker,
            content_hash=content_hash,
            original_tokens=original_tokens,
            marker_tokens=marker_tokens,
            original_bytes=original_bytes,
            input_key=input_key,
            text_block=text_block,
            closed_episode=in_closed_episode,
        )

    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        is_assistant = message.get("role") == "assistant"
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                original = block.get("content")
                text_block = False
                if isinstance(original, list):
                    # Claude Code sends tool_result content as a block list,
                    # almost always a single text block. Mask that inner text;
                    # multi-block or non-text lists stay untouched.
                    if (
                        len(original) == 1
                        and isinstance(original[0], dict)
                        and original[0].get("type") == "text"
                        and isinstance(original[0].get("text"), str)
                    ):
                        original = original[0]["text"]
                        text_block = True
                    else:
                        continue
                if not isinstance(original, str):
                    continue
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    tool_use_id = ""
                tool_name, source_turn = metadata.get(
                    tool_use_id, ("unknown", assistant_turns)
                )
                candidate = prepare(
                    message_index,
                    block_index,
                    tool_use_id,
                    tool_name,
                    source_turn,
                    original,
                    None,
                    text_block,
                )
                if candidate is not None:
                    candidates.append(candidate)
            elif block_type == "tool_use" and is_assistant and mask_input_keys:
                tool_use_id = block.get("id")
                if not isinstance(tool_use_id, str) or tool_use_id not in metadata:
                    continue
                block_input = block.get("input")
                if not isinstance(block_input, dict):
                    continue
                tool_name, source_turn = metadata[tool_use_id]
                for input_key in mask_input_keys:
                    original = block_input.get(input_key)
                    if not isinstance(original, str):
                        continue
                    candidate = prepare(
                        message_index,
                        block_index,
                        tool_use_id,
                        tool_name,
                        source_turn,
                        original,
                        input_key,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates


def apply_candidates(
    messages: list[dict[str, Any]],
    candidates: list[MaskCandidate],
    *,
    compression_store: Any,
) -> MaskResult:
    """Persist and apply an admitted batch, retaining originals on store failure."""
    if not candidates:
        return MaskResult(messages=messages)
    replacements: dict[tuple[int, int, str | None], MaskCandidate] = {}
    for candidate in candidates:
        try:
            stored_hash = compression_store.store(
                original=candidate.original,
                compressed="",
                tool_name=candidate.tool_name,
                tool_call_id=candidate.tool_use_id,
                compression_strategy="observation_masking",
                explicit_hash=candidate.content_hash,
                mask_gate="episode" if candidate.closed_episode else "age",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("observation_masking: CCR store failed for %s: %s", candidate.tool_use_id, exc)
            continue
        if stored_hash != candidate.content_hash:
            logger.warning("observation_masking: CCR store returned an unexpected hash")
            continue
        replacements[
            (candidate.message_index, candidate.block_index, candidate.input_key)
        ] = candidate
    if not replacements:
        return MaskResult(messages=messages)

    output: list[dict[str, Any]] = []
    tokens_saved = 0
    bytes_saved = 0
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            output.append(message)
            continue
        new_content = list(content)
        changed = False
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            candidate = replacements.get((message_index, block_index, None))
            if candidate is not None:
                block_content = block.get("content")
                if candidate.text_block:
                    if (
                        isinstance(block_content, list)
                        and len(block_content) == 1
                        and isinstance(block_content[0], dict)
                        and block_content[0].get("text") == candidate.original
                    ):
                        block = {
                            **block,
                            "content": [
                                {**block_content[0], "text": candidate.marker}
                            ],
                        }
                        new_content[block_index] = block
                        tokens_saved += candidate.tokens_saved
                        bytes_saved += candidate.original_bytes - len(
                            candidate.marker.encode("utf-8")
                        )
                        changed = True
                elif block_content == candidate.original:
                    block = {**block, "content": candidate.marker}
                    new_content[block_index] = block
                    tokens_saved += candidate.tokens_saved
                    bytes_saved += candidate.original_bytes - len(
                        candidate.marker.encode("utf-8")
                    )
                    changed = True
            for input_key in _MASKABLE_INPUT_KEYS:
                candidate = replacements.get((message_index, block_index, input_key))
                if candidate is None:
                    continue
                block_input = block.get("input")
                if (
                    not isinstance(block_input, dict)
                    or block_input.get(input_key) != candidate.original
                ):
                    continue
                block = {
                    **block,
                    "input": {**block_input, input_key: candidate.marker},
                }
                new_content[block_index] = block
                tokens_saved += candidate.tokens_saved
                bytes_saved += candidate.original_bytes - len(
                    candidate.marker.encode("utf-8")
                )
                changed = True
        output.append({**message, "content": new_content} if changed else message)
    return MaskResult(
        messages=output,
        masked_count=len(replacements),
        tokens_saved=tokens_saved,
        bytes_saved=max(0, bytes_saved),
    )


def _masked_values(messages: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Maskable text per slot: ``(tool_use_id, "")`` for a tool result and
    ``(tool_use_id, key)`` for a tool_use input field. Same shapes
    discover_candidates reads, so a masked slot and its original line up."""
    values: dict[tuple[str, str], str] = {}
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                inner = block.get("content")
                if (
                    isinstance(inner, list)
                    and len(inner) == 1
                    and isinstance(inner[0], dict)
                    and inner[0].get("type") == "text"
                ):
                    inner = inner[0].get("text")
                if isinstance(inner, str):
                    values[(str(block.get("tool_use_id", "")), "")] = inner
            elif block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                for key, value in block["input"].items():
                    if isinstance(value, str):
                        values[(str(block.get("id", "")), str(key))] = value
    return values


def keep_cached_prefix(
    candidates: list[MaskCandidate],
    *,
    previous_forwarded: list[dict[str, Any]] | None,
    frozen_count: int,
) -> list[MaskCandidate]:
    """Drop candidates that would change bytes the provider already cached.

    A slot inside the first ``frozen_count`` messages may be masked only when
    the previous request already forwarded that exact marker, which re-sends
    identical bytes. Any other mask there moves the prefix and forces a full
    cache rewrite to save a few hundred tokens. Slots past the frozen boundary
    are uncached and pass through for the gate to price.
    """
    if frozen_count <= 0:
        return candidates
    sent = _masked_values(previous_forwarded or [])
    return [
        candidate
        for candidate in candidates
        if candidate.message_index >= frozen_count
        or sent.get((candidate.tool_use_id, candidate.input_key or "")) == candidate.marker
    ]


class MaskReplayLedger:
    """Books each masked block once per session.

    The client re-sends the raw conversation every turn and masking replaces
    the same block again, so a plain original-vs-forwarded diff books one
    masking on every request until the session ends. This is the masking twin
    of ``ReadMaturationManager.replayed_token_debt``: the first turn a slot
    goes out masked books in full, and every later turn is returned as debt.
    """

    def __init__(self) -> None:
        self._booked: set[tuple[str, str]] = set()
        self._deltas: dict[tuple[str, str], int] = {}

    def replayed_token_debt(
        self,
        original_messages: list[dict[str, Any]],
        outbound_messages: list[dict[str, Any]],
        count_text: Callable[[str], int],
    ) -> int:
        """Tokens this request re-saved by re-masking an already-booked slot.

        Call once per request on the final outbound messages. The first
        booking is recorded as a side effect, so a second call would charge
        the request for its own first appearance.
        """
        forwarded = _masked_values(outbound_messages)
        debt = 0
        for slot, original in _masked_values(original_messages).items():
            sent = forwarded.get(slot)
            if sent is None or sent == original:
                continue
            if not (sent.startswith(MASK_PREFIX) or sent.startswith(INPUT_MASK_PREFIX)):
                continue
            if slot not in self._booked:
                self._booked.add(slot)
                continue
            delta = self._deltas.get(slot)
            if delta is None:
                delta = max(0, int(count_text(original)) - int(count_text(sent)))
                self._deltas[slot] = delta
            debt += delta
        return debt
