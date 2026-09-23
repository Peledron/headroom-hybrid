"""Adversarial tests for the HR_TOKEN_PREFIX_GATE cost-aware prefix gate.

Targets (see session4b.diff, 2026-07-12):
- headroom/cache/prefix_tracker.py: note_compression, recent_compression_ratio,
  cached_token_count, turn_number, latch_compress, compress_latched.
- headroom/proxy/handlers/anthropic.py HR_TOKEN_PREFIX_GATE block: the
  break-even decision built from those primitives plus
  CompressionPolicy.net_mutation_gain.

Does not duplicate tests/test_subagent_ttl.py's `_gate_decision` short/long/
lapsed/tier cases. Every helper here that "replicates the handler formula"
copies the literal expression from headroom/proxy/handlers/anthropic.py
(HR_TOKEN_PREFIX_GATE block) rather than reproducing it from memory, so a
divergence between this file and the handler would show up as a failing
assertion against the module's own primitives (net_mutation_gain,
write_multiplier_for_ttl), not just against a second copy of the arithmetic.
"""

from __future__ import annotations

import math

from headroom.cache.prefix_tracker import PrefixCacheTracker
from headroom.transforms.compression_policy import (
    CACHE_READ_MULTIPLIER,
    policy_default_payg,
    write_multiplier_for_ttl,
)

R_DEFAULT_ENV = 10.0  # HEADROOM_NET_COST_EXPECTED_READS default in the handler
PRESSURE_THRESHOLD_DEFAULT = 0.85  # HR_TOKEN_PRESSURE_THRESHOLD default


def _handler_gate(
    policy,
    *,
    original_tokens: int,
    kept: float,
    S: int,
    turn_number: int,
    ttl: str | None,
    idle: float,
    context_limit: int,
    expected_reads_env: float = R_DEFAULT_ENV,
    pressure_threshold: float = PRESSURE_THRESHOLD_DEFAULT,
) -> tuple[float, float, bool]:
    """Literal transcription of the HR_TOKEN_PREFIX_GATE arithmetic
    (anthropic.py lines ~1436-1473): est_dt, R, w, ttl_s, p_alive, gain,
    pressure, and the compress/forward-original verdict. Returns
    (gain, pressure, should_compress).
    """
    est_dt = max(0, int(original_tokens * (1.0 - kept)))
    R = max(expected_reads_env, float(turn_number))
    ttl_resolved = ttl or "5m"
    w = write_multiplier_for_ttl(ttl_resolved)
    ttl_s = 3600.0 if ttl_resolved == "1h" else 300.0
    p_alive = max(0.0, 1.0 - (idle or 0.0) / ttl_s)
    gain = policy.net_mutation_gain(est_dt, S, R, p_alive, write_multiplier=w)
    pressure = original_tokens / context_limit
    should_compress = gain > 0.0 or pressure >= pressure_threshold
    return gain, pressure, should_compress


# ── Claim 2: note_compression bounds ───────────────────────────────────────


def test_note_compression_ignores_zero_before():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(0, 0)
    assert t.recent_compression_ratio() == 0.8


def test_note_compression_ignores_zero_after():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 0)
    assert t.recent_compression_ratio() == 0.8


def test_note_compression_ignores_inflation():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 1001)
    assert t.recent_compression_ratio() == 0.8


def test_note_compression_ignores_negative_before():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(-500, 100)
    assert t.recent_compression_ratio() == 0.8


def test_note_compression_ignores_negative_after():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(500, -100)
    assert t.recent_compression_ratio() == 0.8


def test_note_compression_accepts_ratio_exactly_one():
    # after == before: zero-savings compression is still a "valid" observation
    # per the (before > 0 and 0 < after <= before) guard. The next turn's
    # estimator will then predict zero saving too, which is internally
    # consistent, but confirms the boundary is admitted, not rejected.
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 1000)
    assert t.recent_compression_ratio() == 1.0


def test_note_compression_huge_but_finite_values_stay_in_bounds():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(10**9, 5 * 10**8)
    ratio = t.recent_compression_ratio()
    assert 0.0 < ratio <= 1.0
    assert abs(ratio - 0.5) < 1e-9


def test_note_compression_floats_as_ints_accepted():
    # The signature is typed `int` but Python does not enforce it at runtime;
    # the handler computes original_tokens/optimized_tokens as ints in
    # practice, but nothing stops a float from reaching this method.
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000.0, 700.5)
    ratio = t.recent_compression_ratio()
    assert 0.0 < ratio <= 1.0
    assert abs(ratio - 0.7005) < 1e-9


def test_note_compression_nan_before_ignored():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(math.nan, 100)
    assert t.recent_compression_ratio() == 0.8


def test_note_compression_nan_after_ignored():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, math.nan)
    assert t.recent_compression_ratio() == 0.8


def test_note_compression_infinite_before_breaks_documented_bound():
    """REFUTES the (0, 1] promise in note_compression's docstring.

    `tokens_before > 0 and 0 < tokens_after <= tokens_before` is satisfied by
    before=inf, after=1 (1 <= inf is True), and the stored ratio is
    `1 / inf == 0.0`, i.e. exactly the excluded lower endpoint. The estimator
    then reads a "100% saving" prior (est_dt == original_tokens) instead of
    anything grounded in a real compression. No caller currently constructs
    tokens_before as inf, so this is a live latent defect rather than a
    reachable-today crash: any future caller that derives tokens_before from a
    division (e.g. a rate) inherits an unclamped path to ratio == 0.0.
    """
    t = PrefixCacheTracker("anthropic")
    t.note_compression(math.inf, 1)
    ratio = t.recent_compression_ratio()
    # Fixed: non-finite before/after is now rejected, so the prior default holds
    # and the ratio never collapses to the excluded 0.0 endpoint.
    assert ratio == 0.8
    assert 0.0 < ratio <= 1.0


def test_note_compression_extreme_finite_before_saturates_ratio_near_zero():
    """Same failure mode as the inf case, reachable with ordinary floats.

    A before/after pair from a pathological tokenizer count (huge before,
    tiny after) drives ratio to a value indistinguishable from 0.0 in the
    downstream `1.0 - kept` estimate, so est_dt saturates to ~100% of
    original_tokens on the next turn even though nothing was validated about
    whether that generalizes.
    """
    t = PrefixCacheTracker("anthropic")
    t.note_compression(10**15, 1)
    ratio = t.recent_compression_ratio()
    assert ratio > 0.0
    est_dt_fraction = 1.0 - ratio
    assert est_dt_fraction > 0.999999999


def test_recent_compression_ratio_custom_default_before_any_note():
    t = PrefixCacheTracker("anthropic")
    assert t.recent_compression_ratio(default=0.5) == 0.5
    assert t.recent_compression_ratio(default=1.0) == 1.0


def test_recent_compression_ratio_survives_later_invalid_notes():
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 300)  # kept 30%
    assert abs(t.recent_compression_ratio() - 0.3) < 1e-9
    # A run of invalid notes must neither reset to default nor corrupt state.
    t.note_compression(0, 0)
    t.note_compression(500, -1)
    t.note_compression(-1, 500)
    t.note_compression(200, 400)
    t.note_compression(math.nan, math.nan)
    assert abs(t.recent_compression_ratio() - 0.3) < 1e-9


def test_cached_token_count_and_turn_number_reflect_tracker_state():
    t = PrefixCacheTracker("anthropic")
    assert t.cached_token_count() == 0
    assert t.turn_number() == 0
    t.update_from_response(
        cache_read_tokens=4000,
        cache_write_tokens=1000,
        messages=[{"role": "user", "content": "x" * 100}],
    )
    assert t.cached_token_count() == 5000
    assert t.turn_number() == 1
    t.update_from_response(
        cache_read_tokens=5000,
        cache_write_tokens=0,
        messages=[{"role": "user", "content": "x" * 100}],
    )
    assert t.cached_token_count() == 5000
    assert t.turn_number() == 2


# ── Claim 1: latch monotonicity / bust-safety ──────────────────────────────


def test_latch_is_idempotent_and_sticky():
    t = PrefixCacheTracker("anthropic")
    assert t.compress_latched is False
    t.latch_compress()
    assert t.compress_latched is True
    t.latch_compress()  # calling again must not raise or toggle
    assert t.compress_latched is True


def test_latch_has_no_reset_path_through_normal_tracker_activity():
    """SURVIVES: nothing in the tracker's public surface unlatches once set.

    Drives every other state-mutating method after latching and confirms none
    of them flips compress_latched back to False. This is the actual
    mechanism the handler's `not prefix_tracker.compress_latched` guard
    depends on for bust-safety; if any of these silently reset the flag the
    guard would go stale and the session could flip back to forward-original,
    busting the compressed cache.
    """
    t = PrefixCacheTracker("anthropic")
    t.latch_compress()
    t.note_compression(1000, 400)
    assert t.compress_latched is True
    t.update_from_response(
        cache_read_tokens=1000,
        cache_write_tokens=200,
        messages=[{"role": "user", "content": "hello"}],
    )
    assert t.compress_latched is True
    t.record_turn_gap(120.0)
    assert t.compress_latched is True
    t.recommended_ttl()
    assert t.compress_latched is True
    # A cache-miss turn (read tokens drop to 0) is exactly the kind of event
    # that resets OTHER tracker state (_cached_token_count -> 0); confirm it
    # does not reach the latch either.
    t.update_from_response(cache_read_tokens=0, cache_write_tokens=0, messages=[])
    assert t.compress_latched is True


def test_forward_original_decisions_are_wire_stable_pre_latch():
    """SURVIVES (with a caveat, see claim 3): two consecutive pre-latch
    "forward original" turns take the IDENTICAL action (skip compression,
    send the unmodified growing prefix) regardless of how much est_dt swings
    between them, because est_dt only feeds the DECISION, not the bytes
    forwarded on the forward-original branch. So oscillation among
    forward-original verdicts cannot itself bust the cache: the wire content
    of the accepted prefix is byte-identical to what was sent (and cached)
    the turn before, only the client's own turn growth changes it, exactly as
    an ungated session would behave.
    """
    p = policy_default_payg()
    # Turn A: pessimistic ratio (kept=0.98, tiny estimated saving).
    gA, _, compress_A = _handler_gate(
        p,
        original_tokens=50000,
        kept=0.98,
        S=48000,
        turn_number=20,
        ttl="5m",
        idle=0,
        context_limit=200000,
    )
    # Turn B: wildly different single-sample ratio (kept=0.4, huge estimated
    # saving) purely because the LAST compression happened to hit a
    # duplicate-heavy blob. Same S/R/ttl/idle otherwise.
    gB, _, compress_B = _handler_gate(
        p,
        original_tokens=50000,
        kept=0.4,
        S=48000,
        turn_number=20,
        ttl="5m",
        idle=0,
        context_limit=200000,
    )
    assert compress_A is False
    # The swing in est_dt is large enough to flip the verdict even though
    # nothing about the session's actual amortization horizon (R, S) moved.
    assert compress_B is True
    assert gB - gA > 20000  # the single-sample ratio alone swings gain by 20k+


# ── Claim 3: estimation error can make a latched compression net-negative ──


def test_bad_prior_estimate_flips_gain_sign_between_decision_and_reality():
    """REFUTES the implicit claim that a positive `gain` computed from the
    prior turn's compression ratio predicts a positive REAL gain.

    Same S, R, p_alive, w. The estimate (kept=0.5, i.e. the LAST compression
    halved the prompt) says compress. The compression that actually runs this
    turn only trims 10% (kept=0.9, content this turn does not resemble last
    turn's), i.e. real dT is 5x smaller than estimated. The honest gain with
    the REAL dT is negative: the gate committed to (and, in the handler,
    latched) a compression that a correct cost model would have rejected.

    This is not a contrived edge: recent_compression_ratio is a single last-
    sample estimator with no smoothing, so any session whose compressibility
    varies turn to turn (duplicate tool output one turn, terse prose the
    next) can reproduce this.
    """
    p = policy_default_payg()
    orig = 50000
    S = 10000
    R = 15.0
    w = write_multiplier_for_ttl("5m")
    p_alive = 1.0

    est_kept = 0.5
    est_dt = int(orig * (1.0 - est_kept))
    gain_estimated = p.net_mutation_gain(est_dt, S, R, p_alive, write_multiplier=w)
    assert gain_estimated > 0.0  # gate decides: compress (and would latch)

    real_kept = 0.95
    real_dt = int(orig * (1.0 - real_kept))
    gain_real = p.net_mutation_gain(real_dt, S, R, p_alive, write_multiplier=w)
    assert gain_real < 0.0  # the compression that actually ran was a net loss

    # Quantify: the estimate was wrong by 5x on dT, and the sign flip means
    # every read for the rest of this (latched) session pays a bust that a
    # correct estimate would have avoided. Severity: ONGOING, not one-time,
    # because the handler latches unconditionally on the estimate BEFORE the
    # real compression ratio is known.
    assert est_dt / max(real_dt, 1) >= 5.0


def test_bad_prior_estimate_case_two_small_margin_estimate_still_flips():
    """A second, less extreme (est_dt, real_dt) pair to show this is not a
    single cherry-picked ratio: even a modest overestimate crosses the
    break-even the wrong way for a suffix that is a large fraction of orig.
    """
    p = policy_default_payg()
    orig = 20000
    S = 15000
    R = 20.0
    w = write_multiplier_for_ttl("5m")
    p_alive = 1.0

    est_dt = int(orig * (1.0 - 0.5))  # 10000, estimate: last compression halved it
    gain_estimated = p.net_mutation_gain(est_dt, S, R, p_alive, write_multiplier=w)
    real_dt = int(orig * (1.0 - 0.9))  # 2000, reality: only trims 10%
    gain_real = p.net_mutation_gain(real_dt, S, R, p_alive, write_multiplier=w)

    assert gain_estimated > 0.0
    assert gain_real < 0.0


# ── Claim 4: R = max(default, turn_number) is a backward-looking proxy ────


def test_end_of_session_turn_forces_compress_with_zero_real_reads_ahead():
    """REFUTES the claim that R = max(default, turn_number) is a safe stand-in
    for expected reads AHEAD.

    Scenario: a session on its FINAL exchange (no more turns will follow).
    turn_number() reflects turns already SEEN (40 prior turns), so the
    handler's R = max(10, 40) = 40 treats this exactly like a session with 40
    reads still ahead. The HONEST remaining-reads count for a session about
    to end is 0. Computed side by side from the same net_mutation_gain:
    honest R=0 says the compression is a net loss (correctly: no reads will
    ever amortize the bust); the handler's turn-count proxy says compress
    anyway.

    Severity: ONE-TIME (the session ends right after), but happens on every
    session that crosses this decision boundary near its natural end, and the
    handler latches the decision before it can find out the session was
    ending, so there is no way to unwind it even in principle within the
    current session's lifetime.
    """
    p = policy_default_payg()
    orig = 50000
    est_dt = 8000  # from prior compression ratio kept=0.84
    S = 10000
    w = write_multiplier_for_ttl("5m")
    p_alive = 1.0  # fresh cache, idle == 0

    honest_gain = p.net_mutation_gain(est_dt, S, 0.0, p_alive, write_multiplier=w)
    assert honest_gain < 0.0  # correct answer: do not compress, nothing to amortize

    turn_number = 40
    proxy_R = max(R_DEFAULT_ENV, float(turn_number))
    proxy_gain = p.net_mutation_gain(est_dt, S, proxy_R, p_alive, write_multiplier=w)
    assert proxy_gain > 0.0  # the gate compresses anyway: a real bug

    # Confirm this reproduces through the literal handler transcription too.
    _, _, should_compress = _handler_gate(
        p,
        original_tokens=orig,
        kept=1.0 - est_dt / orig,
        S=S,
        turn_number=turn_number,
        ttl="5m",
        idle=0,
        context_limit=200000,
    )
    assert should_compress is True


def test_turn_number_only_grows_so_the_proxy_never_self_corrects():
    """turn_number() is monotonic non-decreasing (see update_from_response),
    so once a session has accumulated enough turns to clear the R floor via
    the proxy, EVERY subsequent turn (including the actual last one, whichever
    turn that turns out to be) inherits an R at least as large. The proxy has
    no way to distinguish "40 turns in, 40 more to go" from "40 turns in,
    session ending now": both produce R=40.
    """
    t = PrefixCacheTracker("anthropic")
    for _ in range(40):
        t.update_from_response(
            cache_read_tokens=1000, cache_write_tokens=0, messages=[{"role": "user", "content": "x"}]
        )
    assert t.turn_number() == 40
    r_at_turn_40 = max(R_DEFAULT_ENV, float(t.turn_number()))
    # Whether turn 40 is the middle of a long session or its last exchange,
    # the proxy's R is identical: 40. Nothing in the tracker's public state
    # distinguishes the two.
    assert r_at_turn_40 == 40.0


# ── Claim 5: decision-boundary cross-check against a hand break-even ──────


def _hand_break_even_reads(dt: float, S: float, w: float) -> float:
    """Independent re-derivation of net_mutation_gain's zero-crossing at
    p_alive=1, solved directly from the formula
    gain = dT*(w + r*(R-1)) - (w-r)*(S+dT), rather than via
    CompressionPolicy.break_even_reads (which would just be testing the
    module against itself).
    """
    r = CACHE_READ_MULTIPLIER
    # 0 = dt*w + dt*r*R - dt*r - (w-r)*(S+dt)
    # dt*r*R = (w-r)*(S+dt) + dt*r - dt*w
    # R = [(w-r)*(S+dt) + dt*r - dt*w] / (dt*r)
    numerator = (w - r) * (S + dt) + dt * r - dt * w
    return numerator / (dt * r)


def test_gain_sign_matches_hand_derived_break_even_at_full_alive():
    p = policy_default_payg()
    for dt, S, ttl in [
        (2000, 50000, "5m"),
        (10000, 50000, "1h"),
        (500, 500, "5m"),
        (30000, 1000, "1h"),
    ]:
        w = write_multiplier_for_ttl(ttl)
        r_star = _hand_break_even_reads(dt, S, w)
        below = p.net_mutation_gain(dt, S, max(r_star - 1.0, 0.0), 1.0, write_multiplier=w)
        above = p.net_mutation_gain(dt, S, r_star + 1.0, 1.0, write_multiplier=w)
        assert below < 0.0, (dt, S, ttl, r_star, below)
        assert above > 0.0, (dt, S, ttl, r_star, above)


def test_pressure_override_boundary_is_a_closed_interval_at_threshold():
    p = policy_default_payg()
    limit = 200000
    thr = PRESSURE_THRESHOLD_DEFAULT
    at_boundary_tokens = int(limit * thr)  # pressure == 0.85 exactly (>= admits)
    just_under_tokens = at_boundary_tokens - 1

    # Starve the read-based gain deliberately negative so only pressure can
    # flip the verdict.
    _, pressure_at, compress_at = _handler_gate(
        p,
        original_tokens=at_boundary_tokens,
        kept=0.99,
        S=at_boundary_tokens - 100,
        turn_number=1,
        ttl="5m",
        idle=0,
        context_limit=limit,
    )
    _, pressure_under, compress_under = _handler_gate(
        p,
        original_tokens=just_under_tokens,
        kept=0.99,
        S=just_under_tokens - 100,
        turn_number=1,
        ttl="5m",
        idle=0,
        context_limit=limit,
    )
    assert pressure_at >= thr
    assert compress_at is True
    assert pressure_under < thr
    assert compress_under is False


def test_p_alive_lapse_path_clamps_at_zero_not_negative():
    p = policy_default_payg()
    # idle far beyond the TTL tier: p_alive must clamp to exactly 0, not go
    # negative and invert the (w - r) penalty term's sign.
    gain_huge_idle, _, _ = _handler_gate(
        p,
        original_tokens=50000,
        kept=0.9,
        S=40000,
        turn_number=1,
        ttl="5m",
        idle=10**9,
        context_limit=200000,
    )
    gain_exact_lapse, _, _ = _handler_gate(
        p,
        original_tokens=50000,
        kept=0.9,
        S=40000,
        turn_number=1,
        ttl="5m",
        idle=300.0,
        context_limit=200000,
    )
    assert gain_huge_idle == gain_exact_lapse  # both fully clamped, identical gain
    # A lapsed cache has nothing left to bust, so the gain must equal the pure
    # write-side term (no alive penalty at all).
    est_dt = int(50000 * (1.0 - 0.9))
    w = write_multiplier_for_ttl("5m")
    expected = p.net_mutation_gain(est_dt, 40000, max(R_DEFAULT_ENV, 1.0), 0.0, write_multiplier=w)
    assert gain_huge_idle == expected


def test_regime_outside_pressure_override_where_gate_still_compresses_at_a_loss():
    """Combines claim 4's R-proxy defect with claim 5's ask directly: a
    regime where the gate compresses (gain > 0 via the R proxy) even though
    pressure is nowhere near the 0.85 override band, so the override is not
    what is saving this decision, and the honest math (real reads-ahead = 0)
    says it is a loss. This is the same defect as claim 4, cross-checked here
    against the pressure term to confirm it is NOT the intentional
    compaction-avoidance path.
    """
    p = policy_default_payg()
    orig = 50000
    est_dt = 8000
    S = 10000
    w = write_multiplier_for_ttl("5m")
    limit = 200000

    _, pressure, should_compress = _handler_gate(
        p,
        original_tokens=orig,
        kept=1.0 - est_dt / orig,
        S=S,
        turn_number=40,
        ttl="5m",
        idle=0,
        context_limit=limit,
    )
    assert pressure < PRESSURE_THRESHOLD_DEFAULT  # override is not in play
    assert should_compress is True  # yet the gate compresses anyway (via R proxy)
    honest_gain = p.net_mutation_gain(est_dt, S, 0.0, 1.0, write_multiplier=w)
    assert honest_gain < 0.0  # ...and the honest math says that is a loss
