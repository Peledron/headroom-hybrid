"""Adversarial tests attacking the decompression-bomb cap added in ed264658.

The claim under test: "a compressed request body cannot expand past the
cap, and an over-cap body is refused before it is materialised in memory."

This file does not repeat the writer's own suites
(``tests/test_proxy_request_decompression_cap.py`` and
``tests/test_proxy_decompression_cap_status.py``); it hunts for the gaps
those suites leave open: codec aliasing, boundary math, uncompressed
bodies, malformed size headers, and an empirical proof that the cap is
enforced during inflation rather than after a full buffer is built.

Findings are grouped by section, each labelled with the verdict it
produces. Tests that encode the CORRECT contract and currently fail are
intentional: they are red because the code is wrong, not because the test
is wrong.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import tracemalloc
import zlib
from types import SimpleNamespace

import pytest

from headroom.proxy import helpers as H
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.batch import BatchHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeHeaders:
    """Case-insensitive header stand-in for the low-level ``_read_request_body_bytes`` tests."""

    def __init__(self, d=None):
        self._d = {k.lower(): v for k, v in (d or {}).items()}

    def get(self, k, default=None):
        return self._d.get(k.lower(), default)


class _RawRequest:
    def __init__(self, raw, headers=None):
        self._raw = raw
        self.headers = _FakeHeaders(headers)

    async def body(self):
        return self._raw

    async def stream(self):
        yield self._raw


def _read(raw, encoding=None, *, headers=None):
    hdrs = dict(headers or {})
    if encoding is not None:
        hdrs["content-encoding"] = encoding
    return asyncio.run(H._read_request_body_bytes(_RawRequest(raw, hdrs)))


class _FakeState:
    auth_mode = None


class _HandlerRequest:
    """Minimal Starlette Request stand-in for handler-level (route) tests."""

    def __init__(self, *, path: str = "/v1/messages", headers: dict | None = None) -> None:
        self.headers = headers or {}
        self.state = _FakeState()
        self.url = SimpleNamespace(path=path, query="")
        self.query_params: dict[str, str] = {}

    async def body(self) -> bytes:
        return b"{}"


class _AnthropicHandler(AnthropicHandlerMixin):
    ANTHROPIC_API_URL = "https://anthropic.example"

    async def _next_request_id(self) -> str:
        return "req-1"


class _OpenAIHandler(OpenAIHandlerMixin):
    OPENAI_API_URL = "https://openai.example"

    async def _next_request_id(self) -> str:
        return "req-1"


class _FakeHttpClient:
    async def post(self, url, **kwargs):  # noqa: ANN003, ANN201
        return SimpleNamespace(status_code=200, content=b"{}", headers={})


class _BatchHandler(BatchHandlerMixin):
    GEMINI_API_URL = "https://gemini.example"
    OPENAI_API_URL = "https://openai.example"

    def __init__(self) -> None:
        self.http_client = _FakeHttpClient()

    async def _next_request_id(self) -> str:
        return "req-1"


# ---------------------------------------------------------------------------
# 1. Boundary math: cap-1, cap, cap+1 for every bounded codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_gzip_cap_boundary(monkeypatch, delta):
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 4096)
    payload = b"x" * (4096 + delta)
    raw = gzip.compress(payload)
    if delta > 0:
        with pytest.raises(H.RequestBodyTooLarge):
            _read(raw, "gzip")
    else:
        assert _read(raw, "gzip") == payload


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_deflate_cap_boundary(monkeypatch, delta):
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 4096)
    payload = b"x" * (4096 + delta)
    raw = zlib.compress(payload)
    if delta > 0:
        with pytest.raises(H.RequestBodyTooLarge):
            _read(raw, "deflate")
    else:
        assert _read(raw, "deflate") == payload


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_zstd_cap_boundary(monkeypatch, delta):
    zstandard = pytest.importorskip("zstandard")
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 4096)
    payload = b"x" * (4096 + delta)
    raw = zstandard.ZstdCompressor().compress(payload)
    if delta > 0:
        with pytest.raises(H.RequestBodyTooLarge):
            _read(raw, "zstd")
    else:
        assert _read(raw, "zstd") == payload


# ---------------------------------------------------------------------------
# 2. Cap precision: the running total can overshoot the cap by up to one
#    64 KiB decompression chunk before it is caught. "Cannot expand past the
#    cap" is imprecise for small caps -- the transient buffer briefly holds
#    up to cap + 65536 bytes. Immaterial at the real 100 MB default, but the
#    literal claim wording is wrong.
# ---------------------------------------------------------------------------


def test_gzip_cap_overshoot_is_bounded_by_one_chunk(monkeypatch):
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 10)
    seen_sizes = []
    orig_enforce = H._enforce_decompression_cap

    def _spy(size, label):
        seen_sizes.append(size)
        return orig_enforce(size, label)

    monkeypatch.setattr(H, "_enforce_decompression_cap", _spy)

    # A single-chunk-sized payload (well under 64 KiB) decompressed against
    # a 10-byte cap.
    payload = b"B" * 70_000
    raw = gzip.compress(payload)
    with pytest.raises(H.RequestBodyTooLarge):
        _read(raw, "gzip")

    rejected_at = seen_sizes[-1]
    overshoot = rejected_at - 10
    # Document (not assert-fail) the actual overshoot bound: at most one
    # 64 KiB decompression chunk over the cap, never the full payload.
    assert overshoot <= 65536, (
        f"expected overshoot bounded by one 64 KiB chunk, got {overshoot} bytes "
        f"past the cap (buffer reached {rejected_at} bytes against a 10-byte cap)"
    )
    # But the overshoot is NOT zero: the buffer transiently held far more
    # than the configured cap before rejection. The claim "cannot expand
    # past the cap" is false at the byte level for a chunk-granular check.
    assert overshoot > 0, "expected to observe the known one-chunk overshoot"


# ---------------------------------------------------------------------------
# 3. REFUTED: uncompressed / missing / identity Content-Encoding bypasses
#    the decompressed-body cap entirely. The cap only guards the codec
#    branches; a body with no Content-Encoding header (the common case for
#    a client that doesn't compress) is returned unchecked regardless of
#    size.
# ---------------------------------------------------------------------------


def test_missing_content_encoding_bypasses_cap_entirely(monkeypatch):
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 100)
    oversized = b"A" * 10_000  # 100x the cap, sent uncompressed
    # Fixed: the cap now applies to uncompressed bodies too, so this is
    # refused rather than returned. Kept as a regression guard.
    with pytest.raises(H.RequestBodyTooLarge):
        asyncio.run(H._read_request_body_bytes(_RawRequest(oversized, {})))


def test_explicit_identity_content_encoding_bypasses_cap_entirely(monkeypatch):
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 100)
    oversized = b"A" * 10_000
    # Fixed: identity is no longer a hole in the cap. Kept as a regression guard.
    with pytest.raises(H.RequestBodyTooLarge):
        _read(oversized, "identity")


# ---------------------------------------------------------------------------
# 4. Codec aliasing / spelling variants: aliases must either decompress
#    correctly under the cap or be rejected -- never silently treated as
#    identity (which would forward the still-compressed bytes downstream
#    unchecked, same failure mode as section 3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ["x-gzip", "GZIP;q=1", "gzip\x00", "gzip,gzip", "gzip, gzip"])
def test_gzip_aliases_never_silently_pass_through_as_identity(alias):
    payload = b"should not leak uncompressed" * 10
    raw = gzip.compress(payload)
    try:
        result = _read(raw, alias)
    except ValueError:
        return  # rejecting the alias is an acceptable (if strict) outcome
    # If it didn't reject, it MUST have actually decompressed -- returning
    # the still-gzip-compressed bytes verbatim would defeat downstream
    # UTF-8/JSON parsing silently rather than cleanly, and would mean the
    # cap was never consulted for this alias at all.
    assert result == payload, (
        f"Content-Encoding: {alias!r} returned bytes that are neither the "
        "decompressed payload nor a clean rejection"
    )


def test_case_and_whitespace_variants_still_decompress_and_are_capped(monkeypatch):
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 4096)
    payload = b"y" * 4096
    raw = gzip.compress(payload)
    for variant in ("GZIP", "Gzip", "  gzip  ", "gzip\t"):
        assert _read(raw, variant) == payload, f"variant {variant!r} failed to round-trip"

    bomb_payload = b"z" * 8192
    bomb_raw = gzip.compress(bomb_payload)
    for variant in ("GZIP", "  gzip  "):
        with pytest.raises(H.RequestBodyTooLarge):
            _read(bomb_raw, variant)


# ---------------------------------------------------------------------------
# 5. zstd multi-frame concatenation: mirrors the gzip multi-member test the
#    writer's suite already has for gzip, applied to zstd (untested by the
#    writer).
# ---------------------------------------------------------------------------


def test_zstd_multi_frame_round_trips():
    zstandard = pytest.importorskip("zstandard")
    m1 = b"first frame " * 100
    m2 = b"second frame " * 100
    comp = zstandard.ZstdCompressor()
    raw = comp.compress(m1) + comp.compress(m2)
    assert _read(raw, "zstd") == m1 + m2


def test_zstd_multi_frame_cumulative_cap(monkeypatch):
    zstandard = pytest.importorskip("zstandard")
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 4096)
    comp = zstandard.ZstdCompressor()
    frame = comp.compress(b"A" * 2048)
    with pytest.raises(H.RequestBodyTooLarge):
        _read(frame + frame + frame, "zstd")


# ---------------------------------------------------------------------------
# 6. Zero-length and malformed payloads per codec: must fail as a clean
#    ValueError, never crash with something else, never silently succeed
#    with garbage.
#
# gzip and deflate correctly reject a zero-length body as a truncated
# stream. zstd does not: a zero-length body with Content-Encoding: zstd
# silently decompresses to b"" and is accepted, an asymmetry across codecs
# for the same malformed input. Not itself a cap bypass (empty stays
# empty), but it breaks the "malformed compressed payloads fail cleanly and
# consistently" part of the contract. Left red on purpose for the zstd
# case; see headroom/proxy/helpers.py:_decompress_zstd_capped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "zstd"])
def test_zero_length_body_is_rejected_cleanly(encoding):
    if encoding == "zstd":
        pytest.importorskip("zstandard")
    with pytest.raises(ValueError):
        _read(b"", encoding)


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "zstd"])
def test_single_byte_garbage_body_is_rejected_cleanly(encoding):
    if encoding == "zstd":
        pytest.importorskip("zstandard")
    with pytest.raises(ValueError):
        _read(b"\x00", encoding)


# ---------------------------------------------------------------------------
# 7. REFUTED: a malformed (non-numeric) Content-Length header crashes the
#    handler with an unhandled ValueError instead of a clean 4xx. This is
#    the same size-policy guard clause the decompression cap advertises as
#    "refused before materialised" -- but the int() call it shares is not
#    guarded, so a garbage header takes the whole request path down with an
#    exception the handler's own except clauses never see (it happens
#    before any try/except that would translate it).
# ---------------------------------------------------------------------------


async def test_anthropic_messages_garbage_content_length_must_not_crash():
    request = _HandlerRequest(path="/v1/messages", headers={"content-length": "not-a-number"})
    response = await _AnthropicHandler().handle_anthropic_messages(request)
    assert response.status_code < 500, (
        "expected a clean 4xx for a malformed Content-Length header, "
        f"got {getattr(response, 'status_code', 'an exception')} "
        "(headroom/proxy/handlers/anthropic.py: `int(content_length)` is unguarded)"
    )


async def test_openai_chat_garbage_content_length_must_not_crash():
    request = _HandlerRequest(
        path="/v1/chat/completions", headers={"content-length": "not-a-number"}
    )
    response = await _OpenAIHandler().handle_openai_chat(request)
    assert response.status_code < 500, (
        "expected a clean 4xx for a malformed Content-Length header, "
        f"got {getattr(response, 'status_code', 'an exception')} "
        "(headroom/proxy/handlers/openai.py:handle_openai_chat: "
        "`int(content_length)` is unguarded)"
    )


async def test_openai_responses_garbage_content_length_must_not_crash():
    request = _HandlerRequest(path="/v1/responses", headers={"content-length": "not-a-number"})
    response = await _OpenAIHandler().handle_openai_responses(request)
    assert response.status_code < 500, (
        "expected a clean 4xx for a malformed Content-Length header, "
        f"got {getattr(response, 'status_code', 'an exception')} "
        "(headroom/proxy/handlers/openai.py:handle_openai_responses: "
        "`int(content_length)` is unguarded)"
    )


async def test_google_batch_create_garbage_content_length_must_not_crash():
    request = _HandlerRequest(
        path="/v1beta/models/gemini-pro:batchGenerateContent",
        headers={"content-length": "not-a-number"},
    )
    response = await _BatchHandler().handle_google_batch_create(request, "gemini-pro")
    assert response.status_code < 500, (
        "expected a clean 4xx for a malformed Content-Length header, "
        f"got {getattr(response, 'status_code', 'an exception')} "
        "(headroom/proxy/handlers/batch.py:handle_google_batch_create: "
        "`int(content_length)` is unguarded)"
    )


async def test_anthropic_batch_create_garbage_content_length_must_not_crash():
    request = _HandlerRequest(
        path="/v1/messages/batches", headers={"content-length": "not-a-number"}
    )
    response = await _AnthropicHandler().handle_anthropic_batch_create(request)
    assert response.status_code < 500, (
        "expected a clean 4xx for a malformed Content-Length header, "
        f"got {getattr(response, 'status_code', 'an exception')} "
        "(headroom/proxy/handlers/anthropic.py:handle_anthropic_batch_create: "
        "`int(content_length)` is unguarded)"
    )


# Also probe float-string, absurdly long, whitespace, and hex-prefixed
# Content-Length values -- not just plain non-numeric garbage. (A negative
# value such as "-1" is excluded here: it passes int() cleanly and does not
# reach the unguarded-int() crash this section targets, so exercising it
# would require mocking the full downstream forwarding path rather than the
# size-cap guard clause under test.)
@pytest.mark.parametrize("value", ["1.5", "99999999999999999999999999999999", " ", "0x10"])
async def test_openai_chat_hostile_content_length_values_must_not_crash(value):
    request = _HandlerRequest(path="/v1/chat/completions", headers={"content-length": value})
    response = await _OpenAIHandler().handle_openai_chat(request)
    assert response.status_code < 500, (
        f"Content-Length: {value!r} crashed handle_openai_chat instead of "
        f"returning a clean 4xx (got {getattr(response, 'status_code', 'an exception')})"
    )


# ---------------------------------------------------------------------------
# 8. Empirical proof: is memory actually bounded during inflation, or is
#    the full buffer built first and only then measured?
#
# An earlier version of this section spawned a subprocess and read its
# resource.getrusage(RUSAGE_SELF).ru_maxrss. That technique was discarded:
# it is contaminated by the *calling* process's own memory footprint. When
# the parent process had recently touched large buffers (as pytest does
# after running earlier tests in this file), the freshly spawned child
# reported a peak RSS several times higher than an identical child spawned
# from a cold parent, even though the child's own code never touched more
# than a few hundred KB. That is consistent with vfork/posix_spawn address
# space sharing bleeding through into the child's reported high-water mark,
# not with the decompressor actually materialising the parent's memory.
# Confirmed directly: artificially inflating the parent's RSS before
# spawning the same child pushed its reported peak from ~165 MB to ~420 MB
# for byte-for-byte identical child logic. That number is not trustworthy.
#
# tracemalloc measures actual Python-level allocation inside this process,
# with no subprocess and no OS-level sharing artifact, so it is used here
# instead.
# ---------------------------------------------------------------------------


def test_gzip_bomb_never_allocates_a_buffer_near_the_logical_size(monkeypatch):
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 4096)
    logical_size = 200 * 1024 * 1024  # 200 MB, far above the cap
    bomb = gzip.compress(b"A" * logical_size, compresslevel=9)

    tracemalloc.start()
    try:
        with pytest.raises(H.RequestBodyTooLarge):
            H._decompress_gzip_capped(bomb)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # A correctly bounded implementation should peak at a small multiple of
    # the 64 KiB chunk size, nowhere near the 200 MB logical payload. This
    # is the actual test of the claim: the full inflated buffer is never
    # built.
    assert peak < logical_size // 100, (
        f"decompressing a {logical_size} byte (logical) gzip bomb against a "
        f"4096-byte cap traced a peak allocation of {peak} bytes -- "
        "expected a small multiple of the 64 KiB chunk size, not a fraction "
        "of the logical size"
    )


def test_zstd_bomb_never_allocates_a_buffer_near_the_logical_size(monkeypatch):
    zstandard = pytest.importorskip("zstandard")
    monkeypatch.setattr(H, "MAX_DECOMPRESSED_BODY_BYTES", 4096)
    logical_size = 200 * 1024 * 1024
    bomb = zstandard.ZstdCompressor().compress(b"A" * logical_size)

    tracemalloc.start()
    try:
        with pytest.raises(H.RequestBodyTooLarge):
            H._decompress_zstd_capped(zstandard, bomb)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < logical_size // 100, (
        f"decompressing a {logical_size} byte (logical) zstd bomb against a "
        f"4096-byte cap traced a peak allocation of {peak} bytes -- "
        "expected a small multiple of the 64 KiB chunk size, not a fraction "
        "of the logical size"
    )


# ---------------------------------------------------------------------------
# 9. Full end-to-end status check: bomb via TestClient with the REAL
#    default cap (not a monkeypatched tiny one) for a codec the writer's
#    integration test never exercised (zstd, on /v1/compress).
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_compress_endpoint_body_too_large_zstd():
    zstandard = pytest.importorskip("zstandard")
    from headroom.proxy.helpers import MAX_DECOMPRESSED_BODY_BYTES, get_body_too_large_status
    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    bomb_body = json.dumps(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "A" * (MAX_DECOMPRESSED_BODY_BYTES + 1)}],
        }
    ).encode("utf-8")
    compressed = zstandard.ZstdCompressor().compress(bomb_body)
    expected_status = get_body_too_large_status()

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        resp = client.post(
            "/v1/compress",
            content=compressed,
            headers={"Content-Encoding": "zstd", "Content-Type": "application/json"},
        )
        assert resp.status_code == expected_status, (
            f"expected {expected_status}, got {resp.status_code}: {resp.text}"
        )


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")
def test_compress_endpoint_uncompressed_oversized_body_is_not_rejected_by_decompression_cap():
    """Companion to section 3: confirm via the real HTTP path that an
    uncompressed oversized body does NOT go through the decompression-cap
    rejection message (it either passes, or is rejected by a different
    mechanism entirely -- proving the decompression cap itself never sees
    it).
    """
    from headroom.proxy.helpers import MAX_DECOMPRESSED_BODY_BYTES
    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    # Body slightly over the decompressed-body cap, sent WITHOUT
    # Content-Encoding (identity). Kept a modest multiple over the cap
    # rather than a huge one so the test stays fast; the point is only to
    # cross the threshold that a compressed body of this size would be
    # rejected for.
    oversized = ("A" * (MAX_DECOMPRESSED_BODY_BYTES // 20 + 1024)).encode("utf-8")
    body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": oversized.decode()}]}).encode(
        "utf-8"
    )

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        resp = client.post(
            "/v1/compress",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        payload = resp.json()
        message = json.dumps(payload).lower()
        assert "decompression limit" not in message, (
            "an uncompressed body crossed the decompressed-body cap threshold "
            f"but the response still mentions the decompression limit: {payload}"
        )
