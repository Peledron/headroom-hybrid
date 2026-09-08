"""Opt-in request and response capture for offline replay.

Workstream E, phase 0: a small jsonl sink that records what actually went
upstream (post-transform) alongside the billed usage block, so
``benchmarks/claude_stack_canary.py`` can replay real traffic instead of a
fixed prompt list. Off by default and zero cost when off: the proxy checks
``HEADROOM_REPLAY_CAPTURE`` once at startup, and every call into this module
is defensively wrapped so a capture failure never fails the request it is
describing.

Determinism note: this module only records, it never rewrites the request
that gets sent. Nothing here can become a cache buster, because nothing
here changes request bytes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("headroom.proxy")

ENV_VAR = "HEADROOM_REPLAY_CAPTURE"
DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
_REDACTED = "[redacted]"
_SEQUENCE_RE = re.compile(r"replay-(\d+)\.jsonl$")
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "x-api-key",
    "authorization",
    "anthropic-api-key",
    "proxy-authorization",
}

# Credential-shaped substrings inside free text, for example a bearer token
# pasted into a curl command in a Bash tool_use block. Key-based redaction
# cannot catch these, so string values get a content scan too.
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)"
    r"(?:bearer\s+[a-z0-9._~+/-]{16,}=*"
    r"|sk-(?:ant-)?[a-z0-9_-]{16,}"
    r"|(?:api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[a-z0-9._~+/-]{12,})"
)


def redact(value: Any) -> Any:
    """Deep-copy ``value`` with sensitive keys and values replaced.

    Walks dicts, lists, and tuples recursively so api keys and authorization
    headers are stripped no matter how deep they are nested. Keys are matched
    case-insensitively against ``_SENSITIVE_KEYS``, and string values are
    scanned for credential-shaped substrings (bearer tokens, sk- keys,
    key=value assignments) which are masked in place.
    """
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, sub_value in value.items():
            if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
                result[key] = _REDACTED
            else:
                result[key] = redact(sub_value)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str) and _CREDENTIAL_PATTERN.search(value):
        return _CREDENTIAL_PATTERN.sub(_REDACTED, value)
    return value


class ReplayCapture:
    """Bounded jsonl sink for request and response replay records.

    Rotates to a fresh numbered file when the active file would exceed
    ``max_file_bytes``. Never raises: every public method swallows its own
    errors and logs at debug level, since a capture failure must not
    surface as a proxy failure.
    """

    def __init__(
        self,
        directory: Path | str,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._directory = Path(directory)
        self._max_file_bytes = max_file_bytes
        self._lock = threading.Lock()
        self._sequence = 0
        self._active_path: Path | None = None

    def record(
        self,
        *,
        request_body: dict[str, Any],
        response_usage: dict[str, Any] | None,
        transforms_applied: list[str],
        model: str,
        request_id: str,
        provider: str,
        timestamp: float,
        headers: dict[str, Any] | None = None,
    ) -> None:
        """Append one capture record. Best-effort, never raises."""
        try:
            self._record(
                request_body=request_body,
                response_usage=response_usage,
                transforms_applied=transforms_applied,
                model=model,
                request_id=request_id,
                provider=provider,
                timestamp=timestamp,
                headers=headers,
            )
        except Exception:
            logger.debug("[%s] replay_capture: record failed, skipping", request_id, exc_info=True)

    def _record(
        self,
        *,
        request_body: dict[str, Any],
        response_usage: dict[str, Any] | None,
        transforms_applied: list[str],
        model: str,
        request_id: str,
        provider: str,
        timestamp: float,
        headers: dict[str, Any] | None,
    ) -> None:
        payload = {
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "timestamp": timestamp,
            "request": redact(request_body),
            "response_usage": redact(response_usage) if response_usage else {},
            "transforms_applied": list(transforms_applied),
        }
        if headers:
            payload["headers"] = redact(headers)
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        with self._lock:
            self._directory.mkdir(parents=True, exist_ok=True)
            path = self._current_path_locked(len(line.encode("utf-8")))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def _current_path_locked(self, incoming_bytes: int) -> Path:
        """Pick the active file, rotating if it would cross the size cap.

        On first use the writer resumes at the highest sequence already in
        the directory. Starting from zero on every restart walked one full
        file per record, which left a stray out-of-order record in each
        earlier file before the writer reached the open tail.
        """
        if self._active_path is None:
            self._sequence = self._resume_sequence_locked()
            self._active_path = self._directory / f"replay-{self._sequence:04d}.jsonl"
        while True:
            existing_size = 0
            if self._active_path.exists():
                existing_size = self._active_path.stat().st_size
            if existing_size + incoming_bytes > self._max_file_bytes and existing_size > 0:
                self._sequence += 1
                self._active_path = self._directory / f"replay-{self._sequence:04d}.jsonl"
                continue
            return self._active_path

    def _resume_sequence_locked(self) -> int:
        """Highest sequence number present in the directory, or zero."""
        highest = 0
        for path in self._directory.glob("replay-*.jsonl"):
            match = _SEQUENCE_RE.match(path.name)
            if match is not None:
                highest = max(highest, int(match.group(1)))
        return highest


def iter_replay_records(source: Path | str) -> Iterator[dict[str, Any]]:
    """Yield parsed capture records from a file or a directory of them.

    A directory is walked for ``*.jsonl`` files in sorted (rotation) order,
    which puts capture output back in write order for a single sequence.
    Malformed lines are skipped rather than raising, since a replay corpus
    should tolerate a truncated last line from a killed process.
    """
    path = Path(source)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    elif path.is_file():
        files = [path]
    else:
        return
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def iter_replay_pairs(source: Path | str) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(request, response_usage)`` pairs, the shape a replay driver needs.

    Thin projection over :func:`iter_replay_records` for callers that only
    want the request body and the billed usage, not the full envelope.
    """
    for record in iter_replay_records(source):
        request = record.get("request")
        usage = record.get("response_usage")
        if isinstance(request, dict):
            yield request, usage if isinstance(usage, dict) else {}


_capture: ReplayCapture | None = None
_capture_lock = threading.Lock()
_capture_checked = False


def get_replay_capture() -> ReplayCapture | None:
    """Process-wide singleton, ``None`` unless ``HEADROOM_REPLAY_CAPTURE`` is set.

    Reads the env var once and caches the result (including the "unset"
    result) so the hot path after the first call is a single attribute
    check, not an ``os.environ`` lookup per request.
    """
    global _capture, _capture_checked
    if _capture_checked:
        return _capture
    with _capture_lock:
        if not _capture_checked:
            directory = os.environ.get(ENV_VAR)
            if directory:
                max_bytes_raw = os.environ.get("HEADROOM_REPLAY_CAPTURE_MAX_BYTES")
                try:
                    max_bytes = int(max_bytes_raw) if max_bytes_raw else DEFAULT_MAX_FILE_BYTES
                except ValueError:
                    max_bytes = DEFAULT_MAX_FILE_BYTES
                _capture = ReplayCapture(directory, max_file_bytes=max_bytes)
            _capture_checked = True
    return _capture


def reset_replay_capture_cache() -> None:
    """Test-only: forget the cached singleton so the env var is re-read."""
    global _capture, _capture_checked
    with _capture_lock:
        _capture = None
        _capture_checked = False
