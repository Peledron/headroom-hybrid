"""Cross-restart persistence for SessionTrackerStore lineages.

``SessionTrackerStore`` keeps every lineage in memory, so restarting the proxy
loses ``prev_orig`` and ``prev_fwd`` for every live session.
``overlay_cached_prefix`` then has nothing to replay and each session rewrites
its whole history at the provider's 1.25x write rate. Measured on 2026-07-25 at
roughly 195k write tokens per restart, which is a standing tax on shipping any
fix at all.

This module is only the file layer: snapshot in, snapshot out, atomically and
off the request path. What belongs in a snapshot is decided by
``SessionTrackerStore.export_state``.

Two properties matter more than speed here. The write must be atomic, because a
truncated snapshot would restore a partial history and every later turn would
compare against a prefix that was never actually forwarded. And nothing in this
module may raise into the caller: a persistence failure has to degrade to the
old in-memory behavior, never fail the turn it was trying to make cheaper.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("headroom.cache")

ENV_VAR = "HEADROOM_LINEAGE_PERSIST"
DIR_ENV_VAR = "HEADROOM_LINEAGE_DIR"
SCHEMA_VERSION = 1
DEFAULT_FLUSH_INTERVAL = 20.0


def default_state_path() -> Path:
    """Where snapshots live, alongside the other session-scoped stores.

    Overridable so a test run cannot write live conversation content into the
    developer's own state file, or read it back on the next run.
    """
    override = os.environ.get(DIR_ENV_VAR, "").strip()
    if override:
        return Path(os.path.expanduser(override)) / "state.json.gz"
    return Path(os.path.expanduser("~/.headroom")) / "lineages" / "state.json.gz"


def persistence_enabled() -> bool:
    """Whether to persist lineages. On unless explicitly disabled.

    Default-on because the cost it removes is unconditional: every restart pays
    the full rewrite otherwise. The snapshot holds conversation content, which
    is consistent with the CCR store already on disk, and is written 0600.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


class LineagePersistence:
    """Atomic, debounced snapshot file for lineage state.

    Flushing runs on a background daemon thread rather than in
    ``update_from_response``: the message blobs run to megabytes, and paying a
    multi-megabyte JSON dump on every turn would trade cache-write tokens for
    response latency. A snapshot is written only when the store actually
    changed, detected by a cheap fingerprint rather than by threading a dirty
    flag through the request path.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self.path = Path(path) if path is not None else default_state_path()
        self.flush_interval = max(1.0, float(flush_interval))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._export: Callable[[], dict[str, Any]] | None = None
        self._fingerprint: Callable[[], Any] | None = None
        self._last_fingerprint: Any = None

    def load(self) -> dict[str, Any] | None:
        """Read the snapshot, or None when there is nothing usable to read.

        A missing file is the normal first-run case and is not an error. A
        corrupt or future-versioned file is discarded rather than partially
        applied, which costs one restart's worth of rewrite instead of
        poisoning every subsequent prefix comparison.
        """
        try:
            if not self.path.exists():
                return None
            with gzip.open(self.path, "rt", encoding="utf-8") as handle:
                blob = json.load(handle)
        except Exception as exc:
            logger.warning(
                "LineagePersistence: discarding unreadable snapshot at %s (%s)",
                self.path,
                exc,
            )
            return None

        if not isinstance(blob, dict):
            return None
        version = blob.get("version")
        if version != SCHEMA_VERSION:
            logger.info(
                "LineagePersistence: ignoring snapshot with schema version %r "
                "(this build reads %d)",
                version,
                SCHEMA_VERSION,
            )
            return None
        payload = blob.get("state")
        return payload if isinstance(payload, dict) else None

    def save(self, state: dict[str, Any]) -> bool:
        """Write ``state`` atomically. Returns whether the write landed.

        Writes to a temp file in the destination directory and renames, so a
        crash mid-write leaves the previous good snapshot in place rather than
        a truncated one. ``os.replace`` is atomic within a filesystem, which
        holds because the temp file is created in the destination directory.
        """
        tmp_path: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            blob = {
                "version": SCHEMA_VERSION,
                "saved_at": time.time(),
                "state": state,
            }
            handle_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=".state-", suffix=".tmp"
            )
            os.close(handle_fd)
            # 0600 before any content lands: the snapshot holds conversation
            # text, and a default-mode window between create and chmod would be
            # a real exposure however brief.
            os.chmod(tmp_path, 0o600)
            with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
                json.dump(blob, handle, ensure_ascii=False)
            os.replace(tmp_path, self.path)
            tmp_path = None
            return True
        except Exception as exc:
            logger.warning("LineagePersistence: snapshot write failed (%s)", exc)
            return False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def start(
        self,
        export: Callable[[], dict[str, Any]],
        fingerprint: Callable[[], Any],
    ) -> None:
        """Begin periodic flushing of ``export()`` output.

        ``fingerprint`` must be cheap: it runs every interval and its only job
        is to say whether anything changed since the last successful write.
        """
        with self._lock:
            if self._thread is not None:
                return
            self._export = export
            self._fingerprint = fingerprint
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="headroom-lineage-persist", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.flush_interval):
            self.flush()

    def flush(self, *, force: bool = False) -> bool:
        """Write a snapshot if the store changed. Returns whether it wrote."""
        export = self._export
        fingerprint = self._fingerprint
        if export is None:
            return False
        try:
            current = fingerprint() if fingerprint is not None else None
            if not force and current == self._last_fingerprint:
                return False
            state = export()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LineagePersistence: snapshot export failed (%s)", exc)
            return False
        if self.save(state):
            self._last_fingerprint = current
            return True
        return False

    def stop(self) -> None:
        """Stop flushing and write one final snapshot.

        Forced, because shutdown is exactly when the fingerprint is most likely
        to be unchanged since the last periodic flush while the tail of the
        session is still worth keeping.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            self._thread = None
        self.flush(force=True)
