# Replay corpus measurement, 2026-09-08

Source: `~/.headroom/replay`, the capture written by `headroom/proxy/replay_capture.py`
under `HEADROOM_REPLAY_CAPTURE=1`. Each record is the request body that went upstream
after every transform, plus the usage block the provider billed for it. The corpus does
not hold response bodies.

## What is on disk

| Measure | Value |
|---|---|
| Size | 6.9 GB in 14 files |
| Records, all with usage | 19,822 |
| Time span | 2026-07-21 to 2026-09-08 |
| Sessions (Claude Code session id plus model), 3 or more turns | 94 without Haiku |
| Sessions with 20 or more turns | 72, holding 19,543 records |
| Haiku side-call records, excluded by default | 6,198 |

The 2026-07-25 lineage fix boundary sits inside the span with fewer than 900 records
before it, so the corpus is almost entirely post-fix traffic.

## Capture writer bug, fixed

The writer restarted its sequence at zero on every process start and rotated one file
per record, so each proxy restart appended one stray record to every full file before it
reached the open tail. That is why every file's time span ended on the restart day. The
writer now resumes at the highest existing sequence and skips full files in a loop. The
existing corpus is still out of write order across files, so every reader groups by
session id and sorts by timestamp instead of trusting file order.

## Tools

- `scripts/oracle_replay.py` reads the corpus by default (`--source savings` restores the
  old savings-history path, which is capped at 5000 points and covers only the last few
  days). Sessions are exact lineages keyed on session id and model.
- `scripts/replay_driver.py` audits the corpus offline. Default mode reports per session
  the tokenizer calibration (local count over billed input), the billed cache read
  fraction, the read fraction predicted from the byte-stable prefix, and where consecutive
  bodies diverge first. `--pipeline` additionally pushes each recorded body through the
  current TransformPipeline with a fresh tracker and reports how many recorded bodies the
  current code would change. Because the corpus is post-transform, a nonzero change rate is
  a cache-bust regression signal, not a savings number.
- `headroom/proxy/replay_audit.py` holds the pure functions both scripts use, with tests
  in `tests/test_replay_audit.py`.

## Results

Oracle, all non-Haiku sessions, 5m tier, kept fraction 0.55:

```
aggregate: actual=256,662,769 oracle=165,576,636 regret=35.5%
```

A third of prefix-serving spend was addressable by better mutation timing under the
oracle's own assumptions. The per-session spread is wide (7% to 95%), so the aggregate is
the headline and single sessions are indicative only.

Audit over 94 sessions and 13,684 turns, comparison ignores cache_control markers:

| Model | Turns | Calibration | Billed read | Deep edits | Tail edits | Tools | System |
|---|---|---|---|---|---|---|---|
| claude-opus-5 | 4,504 | 1.005 | 95.8% | 189 | 564 | 158 | 56 |
| claude-sonnet-5 | 4,408 | 0.893 | 94.2% | 1,133 | 448 | 401 | 159 |
| claude-fable-5-1 | 2,148 | 1.050 | 95.1% | 866 | 1,099 | 102 | 55 |
| claude-fable-5 | 2,137 | 1.099 | 97.2% | 22 | 1,213 | 64 | 2 |
| claude-opus-4-8 | 487 | 1.297 | 89.5% | 76 | 14 | 13 | 3 |

Divergence classes over all turns: none 6,953, tail edit 3,338, deep edit 2,286, tools
738, system 275, first turn 94. Predicted read fraction 88.9% against 95.4% billed.

Three things follow:

1. The tokenizer is off by 11% on Sonnet and by 30% on Opus 4.8. Every gate that prices
   in local tokens inherits that error on those models.
2. Deep history edits are the dominant real bust on Sonnet and Fable 5.1 sessions, 26%
   and 40% of their turns. Tools and system changes together are under 8% everywhere.
   This is the STRUCTURAL-BUST finding from `handover-2026-07-24.md` measured on 17 times
   more data.
3. Tail edits are a quarter of all turns. They cost at most one message of re-write and
   the corpus cannot say whether the client or a proxy transform made them. Splitting that
   needs the original client body next to the forwarded one in the capture.

## What the corpus cannot answer

The quality term of the pruning design, an accuracy versus context-length curve, needs
assistant output text. The capture stores only usage, so that gate needs response-body
capture first and a fresh collection window after it.

## Commands

```
uv run python scripts/oracle_replay.py
uv run python scripts/oracle_replay.py --model claude-opus-5 --days 14
uv run python scripts/replay_driver.py --json-out /tmp/replay.json
uv run python scripts/replay_driver.py --pipeline --limit-sessions 5
```
