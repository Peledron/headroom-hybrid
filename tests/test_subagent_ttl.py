"""HR_SUBAGENT_TTL_5M: sub-agent detection + 5m message-prefix ttl override.

Covers the deterministic sub-agent signal (system-head "[1m]" marker) and the
force_ttl path in normalize_message_cache_control that downgrades a short-lived
sub-agent's single message breakpoint from the client's 1h (2x write) to 5m
(1.25x), while leaving the shared system/tools head untouched.
"""

from headroom.cache.prefix_tracker import (
    _strip_cache_control,
    normalize_message_cache_control,
)
from headroom.proxy.handlers.anthropic import (
    _flatten_system_text,
    _system_looks_subagent,
)


def _blk(role, text, ttl=None):
    b = {"type": "text", "text": text}
    if ttl is not None:
        b["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    return {"role": role, "content": [b]}


def _sole_ttl(messages):
    ttls = [
        b["cache_control"].get("ttl")
        for m in messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and isinstance(b.get("cache_control"), dict)
    ]
    assert len(ttls) == 1, f"expected exactly one breakpoint, got {len(ttls)}"
    return ttls[0]


# ── force_ttl override in normalize_message_cache_control ─────────────────────

def test_default_preserves_client_1h():
    msgs = [_blk("user", "a", ttl="1h"), _blk("user", "b", ttl="1h")]
    out = normalize_message_cache_control(msgs)
    assert _sole_ttl(out) == "1h"  # long-lived main keeps 1h retention
    assert _strip_cache_control(out) == _strip_cache_control(msgs)


def test_force_5m_downgrades_client_1h():
    msgs = [_blk("user", "a", ttl="1h"), _blk("user", "b", ttl="1h")]
    out = normalize_message_cache_control(msgs, force_ttl="5m")
    assert _sole_ttl(out) == "5m"  # sub-agent pays 1.25x, not 2x
    # content is byte-identical; only the ttl knob changed
    assert _strip_cache_control(out) == _strip_cache_control(msgs)


def test_force_5m_when_client_sent_no_breakpoint():
    msgs = [_blk("user", "a"), _blk("assistant", "b")]
    out = normalize_message_cache_control(msgs, force_ttl="5m")
    assert _sole_ttl(out) == "5m"


def test_force_5m_idempotent_across_turns():
    # A sub-agent forwards force_ttl every turn; the ttl stays 5m (byte-stable),
    # so the rewrite never self-busts the growing prefix.
    conv = []
    for t in range(1, 6):
        conv = conv + [_blk("user", f"turn-{t}", ttl="1h")]
        conv = normalize_message_cache_control(conv, force_ttl="5m")
        assert _sole_ttl(conv) == "5m"


def _all_ttls(messages):
    """Every cache_control ttl in message order, at ANY nesting depth."""
    found = []

    def walk(obj):
        if isinstance(obj, dict):
            cc = obj.get("cache_control")
            if isinstance(cc, dict):
                found.append(cc.get("ttl", "5m"))
            walk(obj.get("content"))
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(messages)
    return found


def test_nested_tool_result_marker_is_stripped():
    # Regression: a transcript packed into a tool_result re-forwards the inner
    # conversation's cache_control markers. The provider processes nested
    # markers too, so a leftover ttl='1h' after the re-placed 5m breakpoint
    # 400s: "a ttl='1h' cache_control block must not come after a ttl='5m'
    # cache_control block".
    packed = {
        "role": "user",
        "content": [
            {"type": "text", "text": "context"},
            {"type": "text", "text": "stable tail"},
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [
                    {
                        "type": "text",
                        "text": "inner transcript",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
        ],
    }
    msgs = [_blk("user", "a"), _blk("assistant", "b"), packed]
    out = normalize_message_cache_control(msgs, force_ttl="5m")
    ttls = _all_ttls(out)
    assert ttls == ["5m"], f"nested 1h leftover must not survive: {ttls}"
    # semantic content untouched
    assert _strip_cache_control(out) == _strip_cache_control(msgs)


def test_nested_marker_never_becomes_client_intent():
    # A nested leftover must not be promoted to the re-placed breakpoint's
    # marker: with no force_ttl, the breakpoint stays a bare ephemeral (5m
    # default) rather than inheriting the packed transcript's 1h.
    packed = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [
                    {
                        "type": "text",
                        "text": "inner",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {"type": "text", "text": "tail"},
        ],
    }
    out = normalize_message_cache_control([packed])
    ttls = _all_ttls(out)
    assert "1h" not in ttls, f"leftover 1h reused as breakpoint: {ttls}"


# ── sub-agent detection from the system head ──────────────────────────────────

MODEL = "claude-opus-4-8"


def test_main_1m_head_is_not_subagent():
    system = [{"type": "text", "text": f"The exact model ID is {MODEL}[1m]."}]
    assert _system_looks_subagent(system, MODEL) is False


def test_bare_model_head_is_subagent():
    system = [{"type": "text", "text": f"The exact model ID is {MODEL}."}]
    assert _system_looks_subagent(system, MODEL) is True


def test_non_claude_code_head_not_matched():
    system = [{"type": "text", "text": "You are a helpful assistant."}]
    assert _system_looks_subagent(system, MODEL) is False


def test_string_system_head():
    assert _system_looks_subagent(f"model {MODEL}[1m] here", MODEL) is False
    assert _system_looks_subagent(f"model {MODEL} here", MODEL) is True


def test_empty_and_missing():
    assert _system_looks_subagent(None, MODEL) is False
    assert _system_looks_subagent([], MODEL) is False
    assert _system_looks_subagent("anything", None) is False


def test_flatten_joins_blocks():
    # Separator-free, so a marker straddling a block boundary stays intact for
    # the substring detectors that are this function's only callers.
    system = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _flatten_system_text(system) == "ab"
    assert _flatten_system_text("raw") == "raw"
    assert _flatten_system_text(None) == ""


# ── token-mode prefix-mutation gate ───────────────────────────────────────────

from headroom.proxy.handlers.anthropic import _token_prefix_mutation_worth_it  # noqa: E402


def test_warm_roomy_prefix_not_mutated():
    # p_alive high (fresh cache), low context pressure -> re-writing a warm prefix
    # loses (0.1x reads traded for 1.25-2x writes). Do not mutate.
    assert _token_prefix_mutation_worth_it(context_pressure=0.30, p_alive=1.0) is False


def test_near_lapse_prefix_mutated():
    # cache about to expire: the write is coming regardless, so mutation is ~free.
    assert _token_prefix_mutation_worth_it(context_pressure=0.30, p_alive=0.10) is True


def test_high_pressure_prefix_mutated():
    # near the context limit: compressing averts a forced overflow/compaction.
    assert _token_prefix_mutation_worth_it(context_pressure=0.90, p_alive=1.0) is True


def test_thresholds_are_boundaries():
    assert _token_prefix_mutation_worth_it(context_pressure=0.85, p_alive=1.0) is True
    assert _token_prefix_mutation_worth_it(context_pressure=0.84, p_alive=1.0) is False
    assert _token_prefix_mutation_worth_it(context_pressure=0.0, p_alive=0.25) is True
    assert _token_prefix_mutation_worth_it(context_pressure=0.0, p_alive=0.26) is False


# ── adaptive TTL tier selection from observed inter-turn cadence ───────────────

from headroom.cache.prefix_tracker import PrefixCacheTracker  # noqa: E402


def _tracker_with_gaps(gaps):
    t = PrefixCacheTracker("anthropic")
    for g in gaps:
        t.record_turn_gap(g)
    return t


def test_no_recommendation_without_history():
    assert PrefixCacheTracker("anthropic").recommended_ttl() is None
    assert _tracker_with_gaps([30.0]).recommended_ttl() is None  # need >= 2 samples


def test_fast_cadence_picks_5m():
    # all gaps well under the 5-min boundary (with the 60s margin -> 240s)
    assert _tracker_with_gaps([20.0, 45.0, 90.0]).recommended_ttl() == "5m"


def test_long_gap_picks_1h():
    # a single breach past 5 min flips the session to 1h (decide on recent MAX)
    assert _tracker_with_gaps([20.0, 400.0]).recommended_ttl() == "1h"


def test_ambiguous_band_holds_previous():
    t = _tracker_with_gaps([20.0, 30.0])  # -> 5m
    assert t.recommended_ttl() == "5m"
    t.record_turn_gap(270.0)  # in (240, 300] ambiguous band -> hold 5m
    assert t.recommended_ttl() == "5m"
    t.record_turn_gap(500.0)  # clear breach -> 1h
    assert t.recommended_ttl() == "1h"


def test_record_ignores_bad_values():
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(None)
    t.record_turn_gap(-5.0)
    t.record_turn_gap(float("nan"))
    t.record_turn_gap(float("inf"))
    assert t.recommended_ttl() is None  # nothing valid recorded


def test_ring_buffer_ages_out_old_gaps():
    # maxlen 8: once eight fast gaps push an old long gap out, tier returns to 5m
    t = _tracker_with_gaps([600.0] + [30.0] * 8)
    assert t.recommended_ttl() == "5m"


# ── ttl-aware net-cost calculation ────────────────────────────────────────────

from headroom.transforms.compression_policy import (  # noqa: E402
    policy_default_payg,
    write_multiplier_for_ttl,
)


def test_write_multiplier_for_ttl():
    assert write_multiplier_for_ttl("1h") == 2.0
    assert write_multiplier_for_ttl("5m") == 1.25
    assert write_multiplier_for_ttl(None) == 1.25
    assert write_multiplier_for_ttl("bogus") == 1.25


def test_break_even_reads_scales_with_tier():
    p = policy_default_payg()
    dt, s = 1000, 50000
    be_5m = p.break_even_reads(dt, s)
    be_1h = p.break_even_reads(dt, s, write_multiplier=2.0)
    assert round(be_5m, 1) == round(11.5 * s / dt, 1)
    assert round(be_1h, 1) == round(19.0 * s / dt, 1)
    assert be_1h > be_5m  # a 1h bust needs more remaining reads to pay off


def test_net_gain_charges_1h_more():
    p = policy_default_payg()
    # warm cache (p_alive=1), modest reads: busting a 1h suffix should score
    # strictly worse (lower gain) than a 5m one for the same mutation.
    args = dict(delta_t=2000, suffix_tokens=50000, expected_reads=10.0, p_alive=1.0)
    g_5m = p.net_mutation_gain(**args)
    g_1h = p.net_mutation_gain(**args, write_multiplier=2.0)
    assert g_1h < g_5m


def test_default_multiplier_unchanged():
    p = policy_default_payg()
    args = dict(delta_t=2000, suffix_tokens=50000, expected_reads=10.0, p_alive=1.0)
    assert p.net_mutation_gain(**args) == p.net_mutation_gain(**args, write_multiplier=1.25)


# ── broadened sub-agent detection: "no [1m] marker" ───────────────────────────

from headroom.proxy.handlers.anthropic import _system_lacks_1m_marker  # noqa: E402


def test_lacks_1m_marker_main_session_false():
    # main 1M session carries <id>[1m] -> not a sub-agent
    assert _system_lacks_1m_marker([{"type": "text", "text": "model claude-opus-4-8[1m]."}], "claude-opus-4-8") is False
    assert _system_lacks_1m_marker("...claude-opus-4-8[1m]...", "claude-opus-4-8") is False


def test_lacks_1m_marker_subagent_true_even_without_model_id():
    # the gap case the A/B exposed: sub-agent head with NO model id and no [1m]
    assert _system_lacks_1m_marker([{"type": "text", "text": "You are a subagent."}], "claude-opus-4-8") is True
    assert _system_lacks_1m_marker("some system prompt without the marker", "claude-opus-4-8") is True


def test_lacks_1m_marker_empty_is_false():
    assert _system_lacks_1m_marker(None, "claude-opus-4-8") is False
    assert _system_lacks_1m_marker([], "claude-opus-4-8") is False
    assert _system_lacks_1m_marker("", "claude-opus-4-8") is False


def test_lacks_1m_subagent_quoting_marker_still_frozen():
    # a sub-agent that merely QUOTES [1m] (no <model>[1m] rendering) is still a sub-agent
    sys = [{"type": "text", "text": "The main model id is claude-opus-4-8[1m], but you are a helper."}]
    # exact <model>[1m] present -> looks like main (rare paste case)
    assert _system_lacks_1m_marker(sys, "claude-opus-4-8") is False
    sys2 = [{"type": "text", "text": "Note: the 1M marker is written [1m] in the id line."}]
    # generic [1m] mention, no <model>[1m] -> still a sub-agent, gets frozen
    assert _system_lacks_1m_marker(sys2, "claude-opus-4-8") is True


# ── cost-aware prefix gate: tracker state + break-even decision ────────────────

from headroom.transforms.compression_policy import (  # noqa: E402
    write_multiplier_for_ttl as _wmt,
)


def test_compress_latch_and_ratio_tracking():
    t = PrefixCacheTracker("anthropic")
    assert t.compress_latched is False
    assert t.recent_compression_ratio() == 0.8  # prior before any compression
    t.note_compression(1000, 700)  # kept 70%
    assert abs(t.recent_compression_ratio() - 0.7) < 1e-9
    t.note_compression(1000, 0)  # non-positive after -> ignored
    t.note_compression(1000, 1200)  # inflation -> ignored
    assert abs(t.recent_compression_ratio() - 0.7) < 1e-9
    t.latch_compress()
    assert t.compress_latched is True


def _gate_decision(policy, *, orig, cached, kept, R, ttl, idle):
    """Replicate the handler gate's break-even to test the decision boundary."""
    est_dt = max(0, int(orig * (1.0 - kept)))
    w = _wmt(ttl)
    ttl_s = 3600.0 if ttl == "1h" else 300.0
    p_alive = max(0.0, 1.0 - idle / ttl_s)
    gain = policy.net_mutation_gain(est_dt, cached, R, p_alive, write_multiplier=w)
    return gain


def test_short_warm_session_does_not_compress():
    p = policy_default_payg()
    # few expected reads, warm cache, whole prefix cached: busting it loses
    g = _gate_decision(p, orig=50000, cached=48000, kept=0.8, R=10, ttl="5m", idle=0)
    assert g <= 0.0  # forward original


def test_long_warm_session_compresses():
    p = policy_default_payg()
    # many expected reads amortize the one-time bust -> compress
    g = _gate_decision(p, orig=50000, cached=48000, kept=0.8, R=400, ttl="5m", idle=0)
    assert g > 0.0


def test_ttl_lapsed_cache_makes_compression_free():
    p = policy_default_payg()
    # idle beyond the 5m tier -> p_alive 0 -> no bust penalty -> compress even short
    g = _gate_decision(p, orig=50000, cached=48000, kept=0.8, R=10, ttl="5m", idle=600)
    assert g > 0.0


def test_1h_tier_charges_more_so_needs_more_reads():
    p = policy_default_payg()
    # same session, 1h tier (w=2.0) is more conservative than 5m (w=1.25)
    g_5m = _gate_decision(p, orig=50000, cached=48000, kept=0.8, R=120, ttl="5m", idle=0)
    g_1h = _gate_decision(p, orig=50000, cached=48000, kept=0.8, R=120, ttl="1h", idle=0)
    assert g_1h < g_5m


# ── EWMA compression-ratio predictor + confidence discount ────────────────────


def test_ewma_seeds_on_first_sample():
    # A single-observation session reports that sample exactly (EWMA seed), so
    # the estimator is a drop-in for the old last-sample behaviour.
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 650)
    assert abs(t.recent_compression_ratio() - 0.65) < 1e-9
    assert t.compression_ratio_stddev() == 0.0


def test_ewma_smooths_a_spike():
    # After a stable run, one anomalous sample moves the estimate by only alpha,
    # not all the way. Old last-sample logic would have jumped to the spike.
    t = PrefixCacheTracker("anthropic")
    for _ in range(6):
        t.note_compression(1000, 700)  # steady 0.70
    assert abs(t.recent_compression_ratio() - 0.70) < 1e-9
    t.note_compression(1000, 300)  # spike to 0.30
    # EWMA moves 0.3 of the way: 0.7 + 0.3 * (0.3 - 0.7) = 0.58
    assert abs(t.recent_compression_ratio() - 0.58) < 1e-9


def test_ewma_converges_to_a_regime_change():
    # A sustained shift is tracked within a few turns (not stuck on the old mean).
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 700)
    for _ in range(10):
        t.note_compression(1000, 400)  # new steady regime 0.40
    assert abs(t.recent_compression_ratio() - 0.40) < 0.02


def test_confidence_discount_widens_saving_estimate_downward():
    # A noisy session has stddev > 0, so the conservative kept fraction sits
    # above the mean (assume LESS saving), and 1-kept is a lower bound.
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 700)
    t.note_compression(1000, 300)
    t.note_compression(1000, 800)
    mean = t.recent_compression_ratio()
    cons = t.conservative_compression_ratio(k=1.0)
    assert t.compression_ratio_stddev() > 0.0
    assert cons > mean  # discounted toward "keeps more, saves less"
    assert cons <= 1.0  # capped


def test_confidence_discount_noop_on_stable_session():
    # Zero variance -> conservative estimate equals the mean (no penalty).
    t = PrefixCacheTracker("anthropic")
    for _ in range(5):
        t.note_compression(1000, 700)
    assert abs(t.conservative_compression_ratio(k=1.0) - t.recent_compression_ratio()) < 1e-9


def test_confidence_discount_prior_before_any_sample():
    t = PrefixCacheTracker("anthropic")
    assert t.conservative_compression_ratio(default=0.8, k=1.0) == 0.8


def test_confidence_discount_capped_at_one():
    # Large k on a high-variance session must not exceed a 1.0 kept fraction
    # (which would make est_dt negative in the gate).
    t = PrefixCacheTracker("anthropic")
    t.note_compression(1000, 950)
    t.note_compression(1000, 100)
    assert t.conservative_compression_ratio(k=100.0) == 1.0


def test_high_variance_blocks_latch_that_mean_would_allow():
    # End-to-end: the confidence discount can flip a borderline compress to a
    # forward-original when the ratio estimate is unreliable.
    p = policy_default_payg()
    t = PrefixCacheTracker("anthropic")
    # Alternating extreme samples -> high stddev, mean around 0.55.
    for a in (900, 100, 900, 100, 900):
        t.note_compression(1000, a)
    mean_kept = t.recent_compression_ratio()
    cons_kept = t.conservative_compression_ratio(k=1.0)
    g_mean = _gate_decision(p, orig=60000, cached=55000, kept=mean_kept, R=90, ttl="5m", idle=0)
    g_cons = _gate_decision(p, orig=60000, cached=55000, kept=cons_kept, R=90, ttl="5m", idle=0)
    # Discounted saving yields a strictly smaller (more cautious) gain.
    assert g_cons < g_mean
