"""Breaker suite for the TTL-aware net-cost write multiplier (#856 follow-up).

Targets:
  - headroom/transforms/compression_policy.py: write_multiplier_for_ttl,
    the write_multiplier param on net_mutation_gain / should_mutate_deep /
    break_even_reads, CACHE_WRITE_MULTIPLIER_1H.
  - headroom/transforms/content_router.py: ContentRouter._net_cost_allows
    deriving the multiplier from an explicit arg, then
    HEADROOM_NET_COST_WRITE_TTL, else the 5m default.

Every test here is an attempt to falsify a specific claim about that change,
not a demonstration of the happy path. See the per-class docstrings for the
claim under test.
"""

from __future__ import annotations

import itertools
import math
import os

import pytest

from headroom.transforms.compression_policy import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER_1H,
    CompressionPolicy,
    write_multiplier_for_ttl,
)
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


@pytest.fixture
def policy() -> CompressionPolicy:
    # Field values are irrelevant to net_mutation_gain / break_even_reads,
    # only the write/read multiplier constants matter, so any concrete
    # CompressionPolicy instance is representative.
    return CompressionPolicy(
        live_zone_only=True,
        cache_aligner_enabled=True,
        volatile_token_threshold=32,
        max_lossy_ratio=0.25,
        toin_read_only=True,
    )


@pytest.fixture
def router() -> ContentRouter:
    return ContentRouter(ContentRouterConfig())


def _net_cost_allows(router, *, delta_t, suffix, reads_env, p_alive_env, ttl_env, monkeypatch,
                      write_multiplier=None):
    """Call the private gate directly with a controlled env, mirroring the
    setup in tests/test_netcost_gate.py but isolating write_multiplier."""
    if ttl_env is None:
        monkeypatch.delenv("HEADROOM_NET_COST_WRITE_TTL", raising=False)
    else:
        monkeypatch.setenv("HEADROOM_NET_COST_WRITE_TTL", ttl_env)
    monkeypatch.setenv("HEADROOM_NET_COST_EXPECTED_READS", str(reads_env))
    monkeypatch.setenv("HEADROOM_NET_COST_P_ALIVE", str(p_alive_env))
    original = delta_t + 1000
    compressed = original - delta_t
    return router._net_cost_allows(
        slot_idx=0,
        original_tokens=original,
        compressed_tokens=compressed,
        suffix_tokens=[0, suffix],
        route_counts={},
        transforms_applied=[],
        write_multiplier=write_multiplier,
    )


# ─────────────────────────────────────────────────────────────────────────
# 1. REGRESSION: default path (write_multiplier=None / env unset) must be
#    bit-identical to pre-change (w=1.25) behaviour, across a wide grid and
#    at the exact #856 anchor points.
# ─────────────────────────────────────────────────────────────────────────


class TestDefaultPathRegressionGuard:
    """Claim: leaving write_multiplier unset changes nothing. Any deviation
    here is a HIGH-severity regression on the pre-existing, shipped #856
    net-cost gate."""

    GRID_DT = [0, 1, 5, 1000, 50_000, 2_000_000]
    GRID_SUFFIX = [0, 1, 5, 10_000, 50_000, 10**8]
    GRID_READS = [0.0, 0.5, 1, 10, 1e6]
    GRID_ALIVE = [0.0, 0.3, 1.0, float("nan")]

    def test_net_mutation_gain_none_equals_explicit_1_25(self, policy):
        mismatches = []
        for dt, suf, reads, alive in itertools.product(
            self.GRID_DT, self.GRID_SUFFIX, self.GRID_READS, self.GRID_ALIVE
        ):
            a = policy.net_mutation_gain(dt, suf, reads, alive, write_multiplier=None)
            b = policy.net_mutation_gain(dt, suf, reads, alive, write_multiplier=CACHE_WRITE_MULTIPLIER)
            same = a == b or (math.isnan(a) and math.isnan(b))
            if not same:
                mismatches.append((dt, suf, reads, alive, a, b))
        assert not mismatches, f"default-path divergence at: {mismatches[:10]}"

    def test_should_mutate_deep_none_equals_explicit_1_25(self, policy):
        mismatches = []
        for dt, suf, reads, alive in itertools.product(
            self.GRID_DT, self.GRID_SUFFIX, self.GRID_READS, self.GRID_ALIVE
        ):
            a = policy.should_mutate_deep(dt, suf, reads, alive, write_multiplier=None)
            b = policy.should_mutate_deep(dt, suf, reads, alive, write_multiplier=CACHE_WRITE_MULTIPLIER)
            if a != b:
                mismatches.append((dt, suf, reads, alive, a, b))
        assert not mismatches, f"default-path decision divergence at: {mismatches[:10]}"

    def test_break_even_reads_none_equals_explicit_1_25(self, policy):
        mismatches = []
        for dt, suf in itertools.product(self.GRID_DT, self.GRID_SUFFIX):
            a = policy.break_even_reads(dt, suf, write_multiplier=None)
            b = policy.break_even_reads(dt, suf, write_multiplier=CACHE_WRITE_MULTIPLIER)
            if a != b:
                mismatches.append((dt, suf, a, b))
        assert not mismatches, f"default-path break_even_reads divergence at: {mismatches[:10]}"

    def test_break_even_reads_856_anchor_2000_50000(self, policy):
        # #856 anchor: dT=2000, S=50000 -> 287.5 exactly (mod float noise).
        assert policy.break_even_reads(2000, 50_000) == pytest.approx(287.5)

    def test_break_even_reads_856_anchor_50000_10000(self, policy):
        # #856 anchor: dT=50000, S=10000 -> 2.3 exactly (mod float noise).
        assert policy.break_even_reads(50_000, 10_000) == pytest.approx(2.3)

    def test_default_write_multiplier_constant_unchanged(self):
        # CACHE_WRITE_MULTIPLIER (the 5m tier) must still be 1.25 -- the new
        # CACHE_WRITE_MULTIPLIER_1H constant must not have been substituted
        # for it or aliased onto it.
        assert CACHE_WRITE_MULTIPLIER == 1.25
        assert CACHE_WRITE_MULTIPLIER_1H == 2.0
        assert CACHE_WRITE_MULTIPLIER != CACHE_WRITE_MULTIPLIER_1H

    def test_router_default_path_env_unset_matches_explicit_1_25_gate(
        self, router, monkeypatch
    ):
        # End-to-end through _net_cost_allows: env unset, no explicit arg,
        # must match the explicit-1.25 call bit for bit across a scenario
        # that sits right at the boundary (borderline admit/reject) where a
        # multiplier error is most visible.
        monkeypatch.delenv("HEADROOM_NET_COST_WRITE_TTL", raising=False)
        for reads in (170, 172.5, 175, 285, 290):
            monkeypatch.setenv("HEADROOM_NET_COST_EXPECTED_READS", str(reads))
            monkeypatch.setenv("HEADROOM_NET_COST_P_ALIVE", "1")
            unset_result = router._net_cost_allows(
                slot_idx=0,
                original_tokens=2000,
                compressed_tokens=1000,
                suffix_tokens=[0, 15_000],
                route_counts={},
                transforms_applied=[],
            )
            explicit_result = router._net_cost_allows(
                slot_idx=0,
                original_tokens=2000,
                compressed_tokens=1000,
                suffix_tokens=[0, 15_000],
                route_counts={},
                transforms_applied=[],
                write_multiplier=CACHE_WRITE_MULTIPLIER,
            )
            assert unset_result == explicit_result, f"diverged at reads={reads}"


# ─────────────────────────────────────────────────────────────────────────
# 2. write_multiplier_for_ttl must map ONLY the exact literal "1h" -> 2.0.
# ─────────────────────────────────────────────────────────────────────────


class TestWriteMultiplierForTtlMapping:
    """Claim: only the exact string "1h" selects the 2x tier; everything
    else (including near-miss strings, other TTL spellings, and falsy
    values) must fall back to 1.25."""

    @pytest.mark.parametrize(
        "ttl",
        [
            "1H",  # case
            " 1h",  # leading space
            "1h ",  # trailing space
            " 1h ",  # both
            "1hr",  # unit suffix
            "1 h",  # internal space
            "3600",  # seconds-form
            "3600s",
            None,
            "5m",
            "",
            "1",
            "h",
            "01h",
            "1h\n",
            "1h\t",
            "\t1h",
            "one hour",
            "true",
            "1H ",
        ],
    )
    def test_non_exact_matches_fall_back_to_5m_tier(self, ttl):
        assert write_multiplier_for_ttl(ttl) == CACHE_WRITE_MULTIPLIER

    def test_exact_1h_maps_to_1h_tier(self):
        assert write_multiplier_for_ttl("1h") == CACHE_WRITE_MULTIPLIER_1H

    def test_no_leak_across_similar_looking_strings(self):
        # A grid of near-miss strings, none may leak into the 2x tier.
        near_misses = ["1h" + c for c in " \t\n!,.;01hH"] + ["1h" * 2, "1h1h"]
        leaked = [s for s in near_misses if write_multiplier_for_ttl(s) == CACHE_WRITE_MULTIPLIER_1H]
        assert not leaked, f"leaked into 1h tier: {leaked!r}"


# ─────────────────────────────────────────────────────────────────────────
# 3. MONOTONICITY / economic correctness for a warm bust (p_alive=1):
#    higher w must give strictly lower gain and strictly higher (or equal,
#    at dT=0) break_even_reads. Any counterexample defeats the point of the
#    change.
# ─────────────────────────────────────────────────────────────────────────


class TestMonotonicityWarmBust:
    def test_1h_gain_never_exceeds_5m_gain_warm(self, policy):
        # p_alive=1 (fully warm): charging the pricier tier must never look
        # MORE attractive than charging the cheaper tier.
        violations = []
        for dt, suf, reads in itertools.product(
            [1, 5, 100, 1000, 50_000, 2_000_000],
            [0, 1, 100, 10_000, 50_000, 10**8],
            [0.0, 1, 10, 287.5, 1000, 1e6],
        ):
            gain_5m = policy.net_mutation_gain(dt, suf, reads, 1.0, write_multiplier=CACHE_WRITE_MULTIPLIER)
            gain_1h = policy.net_mutation_gain(dt, suf, reads, 1.0, write_multiplier=CACHE_WRITE_MULTIPLIER_1H)
            if gain_1h > gain_5m:
                violations.append((dt, suf, reads, gain_5m, gain_1h))
        assert not violations, f"1h tier scored HIGHER than 5m tier at: {violations[:10]}"

    def test_1h_gain_strictly_lower_when_suffix_positive(self, policy):
        # When there's an actual suffix to protect (suffix>0), the 1h charge
        # must be strictly more conservative, not merely tied.
        for dt, suf, reads in itertools.product(
            [1, 1000, 50_000], [1, 100, 50_000], [0.0, 10, 1000]
        ):
            gain_5m = policy.net_mutation_gain(dt, suf, reads, 1.0, write_multiplier=CACHE_WRITE_MULTIPLIER)
            gain_1h = policy.net_mutation_gain(dt, suf, reads, 1.0, write_multiplier=CACHE_WRITE_MULTIPLIER_1H)
            assert gain_1h < gain_5m, f"not strictly lower at dt={dt} suf={suf} reads={reads}"

    def test_break_even_reads_higher_for_1h_tier(self, policy):
        for dt, suf in itertools.product([1, 1000, 50_000], [1, 100, 50_000]):
            be_5m = policy.break_even_reads(dt, suf, write_multiplier=CACHE_WRITE_MULTIPLIER)
            be_1h = policy.break_even_reads(dt, suf, write_multiplier=CACHE_WRITE_MULTIPLIER_1H)
            assert be_1h > be_5m, f"1h break-even not higher at dt={dt} suf={suf}"

    def test_break_even_ratio_is_exactly_19_over_11_5(self, policy):
        # R = ((w - r) / r) * (S/dT): the 1h/5m break-even ratio is a pure
        # constant (independent of dT, S) equal to 19.0 / 11.5.
        expected_ratio = (CACHE_WRITE_MULTIPLIER_1H - CACHE_READ_MULTIPLIER) / (
            CACHE_WRITE_MULTIPLIER - CACHE_READ_MULTIPLIER
        )
        assert expected_ratio == pytest.approx(19.0 / 11.5, rel=1e-9)
        for dt, suf in itertools.product([1, 7, 1000, 50_000], [1, 3, 100, 50_000]):
            be_5m = policy.break_even_reads(dt, suf, write_multiplier=CACHE_WRITE_MULTIPLIER)
            be_1h = policy.break_even_reads(dt, suf, write_multiplier=CACHE_WRITE_MULTIPLIER_1H)
            assert be_1h / be_5m == pytest.approx(expected_ratio, rel=1e-9)

    def test_1h_tier_can_flip_an_admit_into_a_reject(self, policy):
        # Direct demonstration that the tier choice is load-bearing: a
        # scenario that admits at the 5m rate must reject at the 1h rate
        # for some achievable (dt, suffix, reads).
        dt, suf, reads = 1000, 15_000, 200  # break-even: 172.5 (5m) / 285 (1h)
        assert policy.should_mutate_deep(dt, suf, reads, 1.0, write_multiplier=CACHE_WRITE_MULTIPLIER) is True
        assert policy.should_mutate_deep(dt, suf, reads, 1.0, write_multiplier=CACHE_WRITE_MULTIPLIER_1H) is False


# ─────────────────────────────────────────────────────────────────────────
# 4. _net_cost_allows env parsing: junk / "1h" / "5m" / empty / whitespace
#    must never crash and must default safely to the 1.25 tier on anything
#    but the exact "1h" literal.
# ─────────────────────────────────────────────────────────────────────────


class TestNetCostAllowsEnvParsing:
    # dt=1000, suffix=15000 -> break-even 172.5 (5m) / 285 (1h); reads=200
    # sits strictly between the two, so the tier choice is the only thing
    # that can flip the decision -- proving the env value is actually
    # consumed, not merely tolerated.
    DT, SUF, READS = 1000, 15_000, 200

    @pytest.mark.parametrize(
        "ttl_env",
        [None, "", "5m", "junk", "not-a-ttl", "1H", " 1h", "1h ", "1hr", "3600", "\t", "  "],
    )
    def test_non_1h_env_values_use_5m_tier_and_admit(self, router, monkeypatch, ttl_env):
        allowed = _net_cost_allows(
            router,
            delta_t=self.DT,
            suffix=self.SUF,
            reads_env=self.READS,
            p_alive_env=1,
            ttl_env=ttl_env,
            monkeypatch=monkeypatch,
        )
        assert allowed is True, f"env={ttl_env!r} unexpectedly used the 1h tier (rejected)"

    def test_exact_1h_env_uses_1h_tier_and_rejects(self, router, monkeypatch):
        allowed = _net_cost_allows(
            router,
            delta_t=self.DT,
            suffix=self.SUF,
            reads_env=self.READS,
            p_alive_env=1,
            ttl_env="1h",
            monkeypatch=monkeypatch,
        )
        assert allowed is False, "env='1h' failed to select the pricier tier"

    @pytest.mark.parametrize(
        "ttl_env",
        [None, "", "junk", "1h", "5m", "💥", "a" * 10_000, "1h\n1h"],
    )
    def test_never_raises_regardless_of_env_garbage(self, router, monkeypatch, ttl_env):
        # Must not crash the request path no matter how hostile the env
        # value is. (NUL bytes are excluded: POSIX environ itself rejects
        # them before any application code runs, so that's not a value the
        # code under test could ever observe.)
        try:
            _net_cost_allows(
                router,
                delta_t=self.DT,
                suffix=self.SUF,
                reads_env=self.READS,
                p_alive_env=1,
                ttl_env=ttl_env,
                monkeypatch=monkeypatch,
            )
        except Exception as exc:  # noqa: BLE001 - explicitly checking for absence of any raise
            pytest.fail(f"env={ttl_env!r} raised {exc!r}")

    def test_explicit_write_multiplier_beats_env(self, router, monkeypatch):
        # Explicit kwarg takes priority over the env var (priority order
        # documented at content_router.py _net_cost_allows call site).
        monkeypatch.setenv("HEADROOM_NET_COST_WRITE_TTL", "1h")
        monkeypatch.setenv("HEADROOM_NET_COST_EXPECTED_READS", str(self.READS))
        monkeypatch.setenv("HEADROOM_NET_COST_P_ALIVE", "1")
        # Env says 1h (would reject at reads=200) but the explicit arg
        # forces the 5m rate -> must admit.
        allowed = router._net_cost_allows(
            slot_idx=0,
            original_tokens=self.DT + 1000,
            compressed_tokens=1000,
            suffix_tokens=[0, self.SUF],
            route_counts={},
            transforms_applied=[],
            write_multiplier=CACHE_WRITE_MULTIPLIER,
        )
        assert allowed is True, "explicit write_multiplier did not override env"


# ─────────────────────────────────────────────────────────────────────────
# 5. Clamping with a custom w: NaN/inf/negative/huge w must not produce a
#    nonsense admit. Direct attack on the "no validation of write_multiplier"
#    surface.
# ─────────────────────────────────────────────────────────────────────────


class TestAbsurdWriteMultiplierClamping:
    # A realistic warm-cache, huge-suffix scenario that correctly BLOCKS
    # (gain << 0) at any sane multiplier (1.25 or 2.0).
    DT, SUF, READS, ALIVE = 1000, 50_000, 1.0, 1.0

    def test_sane_multipliers_reject_as_baseline(self, policy):
        assert policy.net_mutation_gain(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=1.25) < 0
        assert policy.net_mutation_gain(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=2.0) < 0

    def test_nan_write_multiplier_fails_closed(self, policy):
        gain = policy.net_mutation_gain(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=float("nan"))
        assert math.isnan(gain)
        assert policy.should_mutate_deep(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=float("nan")) is False

    def test_positive_infinite_write_multiplier_fails_closed(self, policy):
        gain = policy.net_mutation_gain(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=float("inf"))
        # inf - inf during the subtraction of two divergent terms -> nan;
        # should_mutate_deep must still not admit.
        assert math.isnan(gain) or gain <= 0
        assert (
            policy.should_mutate_deep(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=float("inf"))
            is False
        )

    def test_negative_infinite_write_multiplier_fails_closed(self, policy):
        assert (
            policy.should_mutate_deep(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=float("-inf"))
            is False
        )

    @pytest.mark.parametrize("w", [0.0, -0.01, -1.0, -100.0, -1000.0, -1e9])
    def test_negative_or_zero_write_multiplier_is_clamped_no_busting_admit(self, policy, w):
        """FIXED: a caller-supplied write_multiplier <= 0 used to flip the
        sign of the warm-cache penalty term and force an admit for a
        mutation that genuinely busts a large warm cache -- the exact
        scenario that net_mutation_gain exists to reject.

        ``_clamp_write_multiplier`` (headroom/transforms/compression_policy.py)
        now clamps any non-NaN write_multiplier into ``[CACHE_WRITE_MULTIPLIER,
        CACHE_WRITE_MULTIPLIER_1H]`` = ``[1.25, 2.0]`` before it reaches the
        formula, so any value at or below zero clamps up to the 1.25 floor
        instead of inverting the penalty term. The gain now matches the sane
        1.25-tier baseline (see test_sane_multipliers_reject_as_baseline) and
        the gate correctly refuses to admit.
        """
        gain = policy.net_mutation_gain(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=w)
        admitted = policy.should_mutate_deep(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=w)
        assert gain < 0, f"expected the clamped (sane) negative gain at w={w}, got {gain}"
        assert admitted is False

    @pytest.mark.parametrize("w", [0.0, -1.0, -1000.0])
    def test_break_even_reads_stays_non_negative_for_clamped_w(self, policy, w):
        """FIXED companion: break_even_reads used to return a NEGATIVE
        required-read count for w <= r (here r=0.1), which is economically
        meaningless (you cannot need "-50 reads" to break even). The same
        ``_clamp_write_multiplier`` floor now applies here too, so w <= 0
        clamps to 1.25 and break_even_reads returns the same non-negative
        value as the sane 1.25-tier baseline."""
        be = policy.break_even_reads(self.DT, self.SUF, write_multiplier=w)
        assert be >= 0, f"expected a non-negative break-even at w={w}, got {be}"

    def test_huge_positive_write_multiplier_stays_conservative_when_suffix_positive(
        self, policy
    ):
        # A huge but positive w does NOT break the gate when suffix > 0 --
        # it makes the decision MORE conservative (gain decreases as w
        # grows, for p_alive=1 and suffix>0), consistent with the
        # monotonicity claim in section 3. This isolates the negative/zero
        # case above as the actual defect, not "any extreme w".
        gain_normal = policy.net_mutation_gain(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=1.25)
        gain_huge = policy.net_mutation_gain(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=1e6)
        assert gain_huge < gain_normal < 0
        assert (
            policy.should_mutate_deep(self.DT, self.SUF, self.READS, self.ALIVE, write_multiplier=1e6) is False
        )

    def test_zero_suffix_makes_gain_independent_of_write_multiplier_at_full_alive(self, policy):
        # Sanity/documentation check: when suffix=0 (e.g. the batch-reclaim
        # S=0 path) and p_alive=1, the coefficient of w in the gain formula
        # is exactly 0 (dt - alive*(0+dt) = 0), so ANY write_multiplier,
        # sane or absurd, yields the identical gain. This is expected
        # (batch reclaim intentionally charges no incremental write cost)
        # and is not itself a new bug, but it means the multiplier choice
        # is silently inert whenever the caller charges S=0.
        g1 = policy.net_mutation_gain(1000, 0, 10.0, 1.0, write_multiplier=1.25)
        g2 = policy.net_mutation_gain(1000, 0, 10.0, 1.0, write_multiplier=2.0)
        g3 = policy.net_mutation_gain(1000, 0, 10.0, 1.0, write_multiplier=-500.0)
        assert g1 == g2 == g3

    def test_clamped_write_multiplier_via_net_cost_allows_end_to_end(self, router, monkeypatch):
        # Same fix, exercised through the router's public gate rather than
        # the bare CompressionPolicy method, to confirm the clamp is
        # reachable from the call site that plumbs write_multiplier through.
        monkeypatch.delenv("HEADROOM_NET_COST_WRITE_TTL", raising=False)
        monkeypatch.setenv("HEADROOM_NET_COST_EXPECTED_READS", "1")
        monkeypatch.setenv("HEADROOM_NET_COST_P_ALIVE", "1")
        allowed_sane = router._net_cost_allows(
            slot_idx=0,
            original_tokens=51_000,
            compressed_tokens=50_000,  # delta_t=1000
            suffix_tokens=[0, 50_000],
            route_counts={},
            transforms_applied=[],
            write_multiplier=1.25,
        )
        allowed_bad = router._net_cost_allows(
            slot_idx=0,
            original_tokens=51_000,
            compressed_tokens=50_000,
            suffix_tokens=[0, 50_000],
            route_counts={},
            transforms_applied=[],
            write_multiplier=-1.0,
        )
        assert allowed_sane is False
        assert allowed_bad is False, "negative write_multiplier must clamp to 1.25, not force an admit"
