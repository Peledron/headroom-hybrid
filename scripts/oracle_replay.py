#!/usr/bin/env python3
"""Hindsight-optimal replay of logged sessions: how much did the live
prefix-mutation policy leave on the table?

Reads the replay corpus (``~/.headroom/replay``, one record per forwarded
request with the billed usage block) when it exists, else the persistent
savings history (``~/.headroom/proxy_savings.json``, cumulative counters
capped at 5000 points, so only the last few days). Either way it rebuilds
per-session turn series and runs a small dynamic program that picks the cost-optimal compress/hold decision per
turn with full knowledge of the future. The gap between the oracle's cost and
the actual recorded cost is the total budget available to ANY smarter online
policy. If the gap is small, the current gate is near-optimal and further
policy sophistication is wasted effort. That measurement is this script's only
job, it changes nothing.

Model and its assumptions (all approximations, stated so the number is read
with the right error bars):

* Corpus turns carry the provider's own usage split per request and are
  grouped on the Claude Code session id plus model, so they are exact
  lineages. Savings-history turns come from deltas of cumulative per-model
  counters with session boundaries at idle gaps above ``--session-gap``
  seconds, and concurrent sessions of the same model interleave there, so
  treat those per-session numbers as indicative and the aggregate as the
  headline.
* Prices are normalized to input-token units: cache read 0.1, 5m write 1.25,
  1h write 2.0 (Anthropic multipliers). The oracle state is (compressed?,
  ttl tier), transitions charge a full prefix re-write on compress and read
  costs per turn otherwise. Compression keeps ``--kept`` of the prefix
  (default from the observed session mean when derivable, else 0.55).
* The oracle can also do nothing, so it never scores worse than the better of
  hold-everything and compress-at-the-single-best-turn.

Usage::

    uv run python scripts/oracle_replay.py
    uv run python scripts/oracle_replay.py --model claude-fable-5 --days 3
    uv run python scripts/oracle_replay.py --source savings
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CORPUS = Path.home() / ".headroom" / "replay"
READ_MULT = 0.10
WRITE_MULT = {"5m": 1.25, "1h": 2.00}


@dataclass
class Turn:
    timestamp: datetime
    input_tokens: int
    cache_read_tokens: int
    cost_usd: float


@dataclass
class Session:
    model: str
    turns: list[Turn]

    @property
    def actual_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)


def load_history(path: Path, days: float | None, model_filter: str | None) -> list[dict]:
    data = json.loads(path.read_text())
    entries = data.get("history", [])
    if model_filter:
        entries = [e for e in entries if e.get("model") == model_filter]
    if days is not None and entries:
        last = datetime.fromisoformat(entries[-1]["timestamp"].replace("Z", "+00:00"))
        floor = last - timedelta(days=days)
        entries = [
            e
            for e in entries
            if datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) >= floor
        ]
    return entries


def sessions_from_history(entries: list[dict], session_gap_s: float) -> list[Session]:
    """Delta the cumulative counters into turns, split on idle gaps."""
    by_model: dict[str, list[dict]] = {}
    for e in entries:
        by_model.setdefault(e.get("model", "?"), []).append(e)

    sessions: list[Session] = []
    for model, rows in by_model.items():
        prev: dict | None = None
        current: list[Turn] = []
        last_ts: datetime | None = None
        for row in rows:
            ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            if prev is not None:
                d_in = row["total_input_tokens"] - prev["total_input_tokens"]
                d_read = row["cache_read_tokens"] - prev["cache_read_tokens"]
                d_cost = row["total_input_cost_usd"] - prev["total_input_cost_usd"]
                # Counter resets (proxy restart) show up as negative deltas.
                if d_in >= 0 and d_read >= 0:
                    if last_ts is not None and (ts - last_ts).total_seconds() > session_gap_s:
                        if len(current) >= 3:
                            sessions.append(Session(model, current))
                        current = []
                    current.append(Turn(ts, d_in, d_read, max(d_cost, 0.0)))
                    last_ts = ts
            prev = row
        if len(current) >= 3:
            sessions.append(Session(model, current))
    return sessions


def sessions_from_corpus(
    source: Path,
    *,
    model_filter: str | None,
    days: float | None,
    min_turns: int,
    include_haiku: bool,
) -> list[Session]:
    """Exact lineages from the replay corpus, one per session id and model."""
    from headroom.proxy.replay_audit import group_sessions
    from headroom.proxy.replay_capture import iter_replay_records

    groups = group_sessions(
        iter_replay_records(source),
        min_turns=min_turns,
        model_filter=model_filter,
        exclude_model_prefixes=() if include_haiku else ("claude-haiku",),
    )
    if not groups:
        return []
    floor: float | None = None
    if days is not None:
        last = max(t.timestamp for turns in groups.values() for t in turns)
        floor = last - days * 86400.0
    sessions: list[Session] = []
    for (_, model), turns in sorted(groups.items(), key=lambda kv: kv[1][0].timestamp):
        series = [
            Turn(
                datetime.fromtimestamp(t.timestamp, tz=timezone.utc),
                t.billed_input,
                t.cache_read,
                0.0,
            )
            for t in turns
            if floor is None or t.timestamp >= floor
        ]
        if len(series) >= min_turns:
            sessions.append(Session(model, series))
    return sessions


def oracle_cost_tokens(turns: list[Turn], kept: float, ttl: str) -> float:
    """Minimal token-unit cost of serving the observed turn sizes in hindsight.

    DP state per turn: 0 = uncompressed history, 1 = compressed (the one-way
    latch the live gate also uses, so oracle and policy share an action space
    and the comparison isolates TIMING, the thing the user can actually tune).

    Scored on PREFIX-SERVING cost only. This turn's fresh delta costs the
    same under every policy (and under the recorded actuals), so it cancels
    out of the regret and stays excluded from BOTH sides. What remains is
    exactly the controllable quantity: whether the carried prefix is served
    as a cache read, a compressed cache read, or a re-write. The compress
    transition charges the kept prefix at the tier's write premium once.

    Turns where the ACTUAL series shows uncached input beyond the fresh delta
    were busted by the client or provider. No mutation-timing policy dodges
    those, a rewritten history head busts a compressed prefix identically, so
    the oracle re-pays the carried prefix at full price on those turns too
    (scaled by its kept fraction when compressed). Without this the oracle
    scores an unreachable world and the regret is fiction.
    """
    w = WRITE_MULT[ttl]
    inf = float("inf")
    # cost[state]
    cost = [0.0, inf]
    prev_n = 0.0
    for t in turns:
        n = float(t.input_tokens)
        delta = max(n - prev_n, 0.0)
        prefix = min(prev_n, n)
        busted = (t.input_tokens - t.cache_read_tokens - delta) > 0.05 * n
        serve_full = prefix * (1.0 if busted else READ_MULT)
        serve_kept = prefix * kept * (1.0 if busted else READ_MULT)
        hold_un = cost[0] + serve_full
        compress_now = cost[0] + prefix * kept * w + serve_kept
        stay_comp = cost[1] + serve_kept if cost[1] < inf else inf
        cost = [hold_un, min(compress_now, stay_comp)]
        prev_n = n
    return min(cost)


def actual_cost_tokens(turns: list[Turn]) -> float:
    """Prefix-serving token-unit cost the session actually paid.

    Mirrors the oracle's scope: the fresh per-turn delta is excluded (it costs
    the same under every policy), so uncached input BEYOND the delta is what
    busts and re-writes actually charged, at 1.0, plus cached reads at the
    discount."""
    total = 0.0
    prev_n = 0
    for t in turns:
        delta = max(t.input_tokens - prev_n, 0)
        uncached = max(0, t.input_tokens - t.cache_read_tokens - delta)
        total += uncached * 1.0 + t.cache_read_tokens * READ_MULT
        prev_n = t.input_tokens
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--savings",
        type=Path,
        default=Path.home() / ".headroom" / "proxy_savings.json",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--source",
        choices=("auto", "corpus", "savings"),
        default="auto",
        help="auto uses the corpus when the directory exists, else the savings history",
    )
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument(
        "--include-haiku",
        action="store_true",
        help="Keep Haiku side calls (titles, summaries), which are not prefix lineages",
    )
    parser.add_argument("--model", default=None, help="Filter to one model id")
    parser.add_argument("--days", type=float, default=None, help="Only the last N days")
    parser.add_argument(
        "--session-gap",
        type=float,
        default=1800.0,
        help="Idle seconds that split two sessions (default 1800)",
    )
    parser.add_argument(
        "--kept",
        type=float,
        default=0.55,
        help="Fraction of tokens compression keeps (default 0.55)",
    )
    parser.add_argument("--ttl", choices=("5m", "1h"), default="5m")
    args = parser.parse_args()

    use_corpus = args.source == "corpus" or (args.source == "auto" and args.corpus.is_dir())
    if use_corpus:
        sessions = sessions_from_corpus(
            args.corpus,
            model_filter=args.model,
            days=args.days,
            min_turns=args.min_turns,
            include_haiku=args.include_haiku,
        )
        print(f"source: corpus {args.corpus}")
    else:
        entries = load_history(args.savings, args.days, args.model)
        if not entries:
            print("no history entries matched")
            return
        sessions = sessions_from_history(entries, args.session_gap)
        print(f"source: savings history {args.savings}")
    if not sessions:
        print(f"no sessions reconstructed (need {args.min_turns}+ turns per session)")
        return

    print(
        f"{'model':<20}{'turns':>6}{'actual(tok-units)':>19}{'oracle(tok-units)':>19}{'regret':>9}"
    )
    total_actual = 0.0
    total_oracle = 0.0
    for s in sessions:
        actual = actual_cost_tokens(s.turns)
        oracle = oracle_cost_tokens(s.turns, args.kept, args.ttl)
        total_actual += actual
        total_oracle += oracle
        regret = (actual - oracle) / actual if actual > 0 else 0.0
        print(f"{s.model:<20}{len(s.turns):>6}{actual:>19,.0f}{oracle:>19,.0f}{regret:>8.1%}")
    agg = (total_actual - total_oracle) / total_actual if total_actual else 0.0
    print(
        f"\naggregate: actual={total_actual:,.0f} oracle={total_oracle:,.0f} "
        f"regret={agg:.1%} of spend was addressable by better mutation timing"
    )
    print(
        "(model assumptions in the module docstring; treat single sessions as "
        "indicative, the aggregate as the headline)"
    )


if __name__ == "__main__":
    main()
