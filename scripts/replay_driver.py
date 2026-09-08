#!/usr/bin/env python3
"""Offline replay of the capture corpus against the current proxy code.

Reads ``~/.headroom/replay`` (or ``--corpus``), rebuilds sessions from the
Claude Code session id and the model, and reports two things per session
and in aggregate:

* ``audit`` (default): the byte-stable prefix between consecutive forwarded
  bodies versus the cache read the provider billed, plus tokenizer
  calibration (local count over billed input). No pipeline runs, so this is
  a measurement of what the proxy that produced the corpus actually did.
* ``--pipeline``: additionally pushes each recorded body through the current
  TransformPipeline with a fresh prefix tracker and compression cache, the
  same wiring ``benchmarks/proxy_mode_benchmark.py`` uses, and reports how
  often the current code would change bytes the recorded proxy had already
  forwarded. The corpus holds post-transform bodies, so a nonzero change rate
  here means a code change would have busted a prefix the provider had
  cached, which is the regression signal workstream E asked for.

Usage::

    uv run python scripts/replay_driver.py
    uv run python scripts/replay_driver.py --model claude-opus-5 --limit-sessions 5
    uv run python scripts/replay_driver.py --pipeline --json-out /tmp/replay.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from headroom.proxy.replay_audit import (  # noqa: E402
    AuditSummary,
    ReplayTurn,
    audit_session,
    group_sessions,
    prefix_divergence,
)
from headroom.proxy.replay_capture import iter_replay_records  # noqa: E402

DEFAULT_CORPUS = Path.home() / ".headroom" / "replay"


def _pipeline_replay(turns: list[ReplayTurn], tokenizer: Any) -> dict[str, Any]:
    """Re-forward every recorded body through the current pipeline.

    Returns counts of turns whose forwarded bytes the current code would
    change, the token delta of those changes, and how many of the re-forwarded
    turns keep a byte-stable prefix against the previous re-forwarded turn.
    """
    from benchmarks.proxy_mode_benchmark import _make_proxy
    from headroom.cache.compression_cache import CompressionCache
    from headroom.cache.prefix_tracker import PrefixCacheTracker
    from headroom.proxy.modes import PROXY_MODE_HYBRID
    from headroom.proxy.session_engine import finalize_turn
    from headroom.utils import extract_user_query

    proxy = _make_proxy(PROXY_MODE_HYBRID)
    tracker = PrefixCacheTracker("anthropic")
    comp_cache = CompressionCache()
    changed = 0
    tokens_before = 0
    tokens_after = 0
    stable = 0
    previous_forwarded: list[dict[str, Any]] | None = None
    for turn in turns:
        messages = copy.deepcopy(turn.messages)
        if not messages:
            continue
        frozen = tracker.get_frozen_message_count()
        working = comp_cache.apply_cached(messages)
        frozen = min(frozen, comp_cache.compute_frozen_count(messages))
        context_limit = proxy.anthropic_provider.get_context_limit(turn.model) or 200_000
        result = proxy.anthropic_pipeline.apply(
            messages=working,
            model=turn.model,
            model_limit=context_limit,
            context=extract_user_query(working),
            frozen_message_count=frozen,
        )
        comp_cache.update_from_result(messages, result.messages)
        final = finalize_turn(
            result.messages,
            messages,
            tracker.get_last_original_messages(),
            tracker.get_last_forwarded_messages(),
            count_tokens=tokenizer.count_messages,
            confirmed_frozen_count=frozen,
        )
        forwarded = final.messages
        before = tokenizer.count_messages(messages)
        after = final.tokens if final.tokens is not None else tokenizer.count_messages(forwarded)
        tokens_before += before
        tokens_after += after
        if forwarded != messages:
            changed += 1
        if previous_forwarded is not None:
            shared = prefix_divergence({"messages": previous_forwarded}, {"messages": forwarded})
            if shared.section == "none":
                stable += 1
        tracker.update_from_response(
            cache_read_tokens=turn.cache_read,
            cache_write_tokens=turn.cache_write,
            messages=forwarded,
            message_token_counts=[tokenizer.count_message(m) for m in forwarded],
            original_messages=messages,
        )
        previous_forwarded = copy.deepcopy(forwarded)
    return {
        "turns": len(turns),
        "changed_turns": changed,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "stable_prefix_turns": stable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", default=None, help="Filter to one model id")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--include-haiku", action="store_true")
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        print(f"no corpus at {args.corpus}")
        return 1
    from headroom.tokenizers import get_tokenizer

    groups = group_sessions(
        iter_replay_records(args.corpus),
        min_turns=args.min_turns,
        model_filter=args.model,
        exclude_model_prefixes=() if args.include_haiku else ("claude-haiku",),
    )
    ordered = sorted(groups.items(), key=lambda kv: kv[1][0].timestamp)
    if args.limit_sessions is not None:
        ordered = ordered[: args.limit_sessions]
    if not ordered:
        print("no sessions matched")
        return 1

    summary = AuditSummary()
    rows: list[dict[str, Any]] = []
    print(
        f"{'session':<10}{'model':<22}{'turns':>6}{'calib':>7}{'read%':>8}"
        f"{'pred%':>8}{'tools':>6}{'sys':>5}{'msgs':>5}"
        + (f"{'chg':>5}{'stable':>7}" if args.pipeline else "")
    )
    for (session_id, model), turns in ordered:
        tokenizer = get_tokenizer(model)
        audits = audit_session(turns, tokenizer)
        local = AuditSummary()
        local.add(audits)
        summary.add(audits)
        row: dict[str, Any] = {"session": session_id, "model": model, **local.as_dict()}
        line = (
            f"{session_id[:8]:<10}{model:<22}{local.turns:>6}{local.calibration:>7.2f}"
            f"{local.billed_read_fraction:>8.1%}{local.predicted_read_fraction:>8.1%}"
            f"{local.divergence_by_section.get('tools', 0):>6}"
            f"{local.divergence_by_section.get('system', 0):>5}"
            f"{local.divergence_by_section.get('messages', 0):>5}"
        )
        if args.pipeline:
            replay = _pipeline_replay(turns, tokenizer)
            row["pipeline"] = replay
            line += f"{replay['changed_turns']:>5}{replay['stable_prefix_turns']:>7}"
        rows.append(row)
        print(line)

    agg = summary.as_dict()
    print(
        f"\naggregate: sessions={agg['sessions']} turns={agg['turns']} "
        f"calibration={agg['calibration']:.3f} billed_read={agg['billed_read_fraction']:.1%} "
        f"predicted_read={agg['predicted_read_fraction']:.1%} "
        f"divergence={agg['divergence_by_section']}"
    )
    if args.pipeline:
        changed = sum(r["pipeline"]["changed_turns"] for r in rows)
        total = sum(r["pipeline"]["turns"] for r in rows)
        print(f"pipeline: current code changes {changed}/{total} recorded bodies")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps({"aggregate": agg, "sessions": rows}, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
