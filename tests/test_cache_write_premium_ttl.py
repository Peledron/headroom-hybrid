"""Cache-write premium is priced per TTL bucket.

Anthropic bills a 1-hour-TTL cache write at 2.0x base input, not the 1.25x a
5-minute write costs. Reporting every write at 1.25x under-states the premium on
a session using ENABLE_PROMPT_CACHING_1H by 60%, which makes net savings look
better than the invoice.
"""

from __future__ import annotations

import pytest

from headroom.proxy.cost import _CACHE_ECONOMICS, cache_write_premium_usd

PRICE = 1e-6  # $1 per 1M tokens keeps the arithmetic readable


def test_all_5m_matches_the_flat_legacy_rate() -> None:
    """With no 1h tokens the result is exactly the old flat 1.25x behavior."""
    premium = cache_write_premium_usd(
        write_tokens=1_000_000,
        write_1h_tokens=0,
        input_price_per_token=PRICE,
        write_mult=1.25,
        write_mult_1h=2.0,
    )
    assert premium == 0.25  # 1M * $1/M * 0.25


def test_all_1h_charges_the_2x_rate() -> None:
    premium = cache_write_premium_usd(
        write_tokens=1_000_000,
        write_1h_tokens=1_000_000,
        input_price_per_token=PRICE,
        write_mult=1.25,
        write_mult_1h=2.0,
    )
    assert premium == 1.0  # 1M * $1/M * 1.0 — 4x the 5m premium


def test_mixed_buckets_are_priced_separately() -> None:
    premium = cache_write_premium_usd(
        write_tokens=1_000_000,
        write_1h_tokens=400_000,
        input_price_per_token=PRICE,
        write_mult=1.25,
        write_mult_1h=2.0,
    )
    # 600k at 0.25 + 400k at 1.0
    assert premium == pytest.approx(0.15 + 0.4)


def test_unbucketed_tokens_bill_at_the_5m_rate() -> None:
    """A provider that reports no split must behave exactly as before."""
    premium = cache_write_premium_usd(
        write_tokens=1_000_000,
        write_1h_tokens=0,
        input_price_per_token=PRICE,
        write_mult=1.25,
        write_mult_1h=1.25,
    )
    assert premium == 0.25


def test_bucket_cannot_exceed_total() -> None:
    """A bogus/oversized 1h count can't inflate the premium past the total."""
    premium = cache_write_premium_usd(
        write_tokens=100_000,
        write_1h_tokens=999_999_999,
        input_price_per_token=PRICE,
        write_mult=1.25,
        write_mult_1h=2.0,
    )
    assert premium == pytest.approx(0.1)  # all 100k at the 1h rate, nothing more


def test_no_premium_when_multiplier_is_not_above_one() -> None:
    assert (
        cache_write_premium_usd(
            write_tokens=1_000_000,
            write_1h_tokens=500_000,
            input_price_per_token=PRICE,
            write_mult=1.0,
            write_mult_1h=1.0,
        )
        == 0.0
    )


def test_zero_inputs_are_free() -> None:
    assert (
        cache_write_premium_usd(
            write_tokens=0,
            write_1h_tokens=0,
            input_price_per_token=PRICE,
            write_mult=1.25,
            write_mult_1h=2.0,
        )
        == 0.0
    )
    assert (
        cache_write_premium_usd(
            write_tokens=1_000,
            write_1h_tokens=0,
            input_price_per_token=0.0,
            write_mult=1.25,
            write_mult_1h=2.0,
        )
        == 0.0
    )


def test_ttl_aware_providers_declare_the_1h_rate() -> None:
    """Anthropic-family providers have a 1h TTL; the others deliberately don't."""
    for provider in ("anthropic", "bedrock"):
        assert _CACHE_ECONOMICS[provider]["write_multiplier_1h"] == 2.0
    for provider in ("openai", "gemini"):
        assert "write_multiplier_1h" not in _CACHE_ECONOMICS[provider]
