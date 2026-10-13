# Self-optimizing context pruning, design (2026-07-21)

Companion to optimization-plan-2026-07-19.md workstream K. K names the judge
and the exits. This doc turns it into a self-learning loop and phases the build.

## The objective, two paybacks

Minimize `tokens + lambda * accuracy_loss_from_context_length`. The token half
is the cache cost model already priced in workstreams C, H, and J: a cut saves
reads at 0.1x every turn, and at a bust the suffix rewrite is sunk so a deep
cut is nearly free. The quality half is new: models degrade at long context
(arXiv 2505.06120, 39 percent multi-turn drop with unreliable history, plus the
lost-in-the-middle effect). So a cut that is cost-neutral can still pay back in
accuracy. lambda is the exchange rate between the two, and it is the one number
this system has to learn rather than assume.

## The flywheel, one self-improving loop

1. Capture. The replay recorder (workstream E, enabled 2026-07-21) writes every
   request and response, redacted, to disk. This is the training set.
2. Label, for free, from the system's own operation. A cut was wrong if the cut
   content is retrieved later (a headroom_retrieve call), re-read, or shares
   rare tokens with a later assistant turn. "Needed later" is derived from real
   events, so no human labeling is required.
3. Learn, offline. JEPA trains self-supervised on the corpus (predict the next
   representation from context). lambda is fit from the measured accuracy versus
   context-length curve. The keep threshold is tuned from the observed
   retrieve-rate.
4. Decide, live. The judge scores each region on three axes: solved
   (supersession rules in audit/reads.py), irrelevant (novelty, JEPA or the
   cheaper embedding score), active (matches the task line, the structural
   ledger). Cut hardest at busts and the frontier, never rewrite deep-warm
   content mid-conversation.
5. Measure. The closed-loop reconciliation (workstream H) gives the token
   payback. The retrieve-rate gives the mistake rate. An accuracy proxy
   (retry rate, error rate, task success) gives the quality signal.
6. Back to capture. Live decisions generate new labeled data, the corpus grows,
   the judge improves. The loop is the product.

## What makes it self-learning, specifically

- JEPA is self-supervised, so it needs no human labels, it learns the structure
  of this workload's context and retrains as the corpus grows.
- The needed-later labels come free from the system's own retrieval events, so
  the judge learns from its own mistakes without a labeling step.
- lambda re-fits as the accuracy-versus-context curve sharpens with more data.
- The keep threshold self-tunes: too many cuts retrieved means raise it, a
  context that stays bloated with dead content means lower it. A ski-rental or
  bandit controller owns that one scalar.

## The world model, the shared predictive substrate

The JEPA is not only the novelty scorer, it is a world model of the conversation
in the LeCun sense: trained self-supervised to predict the next representation of
the context stream, it learns the dynamics of how this agentic context evolves.
That one learned object replaces three separate guesses the system makes today:

- Prediction error is the novelty signal. A block the model predicts well is
  redundant (cut), a block it predicts badly is new (keep). That is the
  irrelevant axis of the judge, just the model's loss on the incoming block.
- Forward rollout is the future-need estimate. Roll the model a few steps and it
  estimates whether a region will be read or needed again. That is the
  needed-later label the judge wants, and it is the same expected_reads and
  p_alive that the cache DP (workstream C) and the rebase gain formula
  (hybrid_mode.net_rebase_gain) currently plug in as point-estimate guesses.
- So it is the shared substrate. Pruning guesses relevance by age, caching
  guesses expected_reads, TTL guesses reuse, the rebase guesses p_alive. Every
  one is a separate heuristic estimate of "what will be needed next." The world
  model replaces all of them with one learned prediction. Pruning, caching, and
  TTL are all downstream of that single question.

Two boundaries. It is the end of the ladder, not the start: cheap heuristics
first (age, cosine similarity), the world model only if they prove insufficient
(rule of three). And rollout is short-horizon only, world models drift and
compound error on long rollouts, so the forward prediction is trusted for a few
steps, not fifty. Even fully trained it stays a predictor, not a policy-maker.

## Guardrails, non-negotiable

Per the project stance (the model generates, deterministic mechanisms plus a
gated graduation decide) and the ml-eval-protocol:

- Learn offline, never live-mutate policy unguarded.
- Graduate a new judge only when it beats a fixed baseline on a held-out replay
  split, not the split it trained on.
- Canary every live change on both token savings (H reconciliation) and the
  retrieve-rate (the quality half), the same two-sided gate the rest of the
  plan uses.
- Kill-switch: if retrieve-rate or error-rate spikes past a bound, auto-revert
  to the last-good judge.
- No reward hacking: the objective already counts retrieve-rate as a cost, so a
  judge cannot win by cutting everything. Re-anchor periodically to a
  human-audited label set to catch slow drift.

## Phased build, dependency order

- Phase 0, done: replay corpus enabled, accumulating the training set.
- Phase 1: the labeler, derive needed-later from retrieval, re-read, and
  rare-token overlap (workstream B step 2), then audit the labels by hand.
- Phase 2: baseline judge, supersession (already live) plus the embedding
  novelty score (B candidates 1 and 2), log-only, measured against the age
  baseline on retrieve-rate.
- Phase 3: if novelty beats age, train JEPA (B candidate 3) self-supervised,
  log-only, compared on the held-out split. If the cheaper scorers already win,
  JEPA may never be needed.
- Phase 4: assemble the full K judge, three axes plus the reference,
  summarize-to-memory, and drop exits, still log-only, logging the would-be
  token and accuracy delta per cut.
- Phase 5: fit lambda from the accuracy-versus-context curve. Ship cost-only
  cutting live first, quality term still log-only.
- Phase 6: graduate to quality-driven live cutting behind the canary and the
  kill-switch.
- Phase 7: close the flywheel, continuous offline retrain, gated graduation,
  self-tuning threshold.

## The honest hard parts

- The accuracy signal is the weak link. Online accuracy has no ground truth, so
  the proxies (retry, error, retrieve rate) are noisy, and lambda stays a guess
  until the curve sharpens. This is the gating uncertainty for the whole quality
  half, so it ships log-only longest.
- Feedback-loop collapse. A judge trained on its own decisions can drift.
  Mitigated by held-out validation, the fixed-baseline gate, the kill-switch,
  and periodic re-anchoring to a human-audited set.
- JEPA training cost and uncertainty. Gated behind cheaper scorers proving
  signal first, per the rule of three, so the expensive build only happens if
  the concept is already validated.

## Safety rule for the summarize-to-memory exit

A generated summary can hallucinate or drop the one detail needed later, so the
summary is additive to the safety net, never a replacement. Keep the retrievable
reference (the CCR marker) even when a region is summarized, so the failure mode
is "the agent retrieves the raw," not "the information is gone." The structural
ledger (workstream D) provides the deterministic state, the model only writes
the prose summary, and it runs at a bust or offline, never per-request in the
warm path.

## 2026-10-12: goal-aware forgetting and JEPA

The scorer should estimate whether a block will be needed for the next few
steps of the current task. Its inputs are the explicit goal, active constraints,
the current plan, recent turns, and the candidate block. A block can be familiar
and still essential. Novelty alone is insufficient for this decision.

The closest verified research match is
[FlashMemory for DeepSeek V4](https://arxiv.org/html/2606.09079v3).
It predicts future context demand with a separately trained memory indexer and
recalls selected KV blocks from a CPU store. Its authors include Tencent,
university, and independent researchers. The paper reports a failure on dense
global-memory tasks, which matters for any attempt to select only a few blocks.
DeepSeek's own
[sparse attention selector](https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md)
selects attention entries inside the model. Headroom can borrow the selection
idea, but an API proxy has access to message blocks rather than hidden states
or provider KV caches.

[I-JEPA](https://arxiv.org/abs/2301.08243) learns representations by predicting
target embeddings from context. For this task, a JEPA representation still needs
a goal-conditioned decision head and calibration against future use. The
comment-lint findings in `docs/decisions/code-heads.md`, especially D-045 and
D-057, support testing explicit intent conditioning. Those experiments concern
code correctness and do not establish the quality of context relevance models.

### Candidate and evaluation contract

1. Start with the existing age, supersession, and deduplication policies. Compare
   a goal-conditioned lexical scorer, then a frozen text encoder with a small
   supervised relevance head. Add JEPA pretraining if it improves held-out
   needed-block recall at the same retained token budget and acceptable latency.
2. Give the scorer the information available at the decision time. Later
   retrievals, re-reads, and assistant references supply labels, never features.
   Split by complete session and project. Treat the absence of a later reference
   as an uncertain negative until audited.
3. Measure missed needed blocks, successful task outcomes, recovery latency,
   billed token cost, cache writes, and scorer cost. A lower retrieval rate alone
   can hide cases where the agent fails to notice missing evidence.
4. Run in shadow mode first. Keep system instructions, active user constraints,
   unresolved tool calls, signed content, and required tool definitions protected
   by deterministic rules. The classifier supplies a score to those rules.
5. Price each proposed rewrite through the existing cache economics gate.
   Expected read savings must cover the cache rewrite, scoring, and expected
   retrieval costs. Prefer an already admitted rebase or cache bust for changes
   to a warm prefix.
6. Store the exact original in CCR before replacing it with a reference. Confirm
   the reference remains live before replaying a cached optimized result. Goal
   changes should trigger rescoring and allow relevant blocks to be restored.

### Current implementation boundary

The novelty-routing evaluator still contains an embedding stub. No learned
goal-aware judge or JEPA training is implemented by this upstream integration.
Request replay capture alone also does not provide paired original histories
and assistant responses for an outcome-quality curve. That evaluation needs a
suitable transcript corpus or an explicitly configured capture window before
any quality-based live pruning can graduate.
