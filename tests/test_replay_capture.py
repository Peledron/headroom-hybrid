import json

import pytest

from headroom.proxy.replay_capture import (
    ENV_VAR,
    ReplayCapture,
    get_replay_capture,
    iter_replay_pairs,
    iter_replay_records,
    redact,
    reset_replay_capture_cache,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("HEADROOM_REPLAY_CAPTURE_MAX_BYTES", raising=False)
    reset_replay_capture_cache()
    yield
    reset_replay_capture_cache()


def test_redact_strips_sensitive_keys_at_any_depth() -> None:
    value = {
        "api_key": "sk-secret",
        "headers": {"Authorization": "Bearer xyz", "X-Api-Key": "abc"},
        "nested": [{"authorization": "token"}],
        "keep": "value",
    }
    result = redact(value)
    assert result["api_key"] == "[redacted]"
    assert result["headers"]["Authorization"] == "[redacted]"
    assert result["headers"]["X-Api-Key"] == "[redacted]"
    assert result["nested"][0]["authorization"] == "[redacted]"
    assert result["keep"] == "value"
    # Original is untouched.
    assert value["api_key"] == "sk-secret"


def test_record_writes_one_jsonl_line(tmp_path) -> None:
    capture = ReplayCapture(tmp_path)
    capture.record(
        request_body={"model": "claude", "messages": [{"role": "user", "content": "hi"}]},
        response_usage={"input_tokens": 10, "output_tokens": 5},
        transforms_applied=["dedupe"],
        model="claude-x",
        request_id="req-1",
        provider="anthropic",
        timestamp=123.5,
        headers={"authorization": "Bearer secret"},
    )
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request_id"] == "req-1"
    assert record["provider"] == "anthropic"
    assert record["model"] == "claude-x"
    assert record["timestamp"] == 123.5
    assert record["transforms_applied"] == ["dedupe"]
    assert record["response_usage"]["input_tokens"] == 10
    assert record["headers"]["authorization"] == "[redacted]"
    assert record["request"]["messages"][0]["content"] == "hi"


def test_record_never_raises_on_bad_directory(tmp_path) -> None:
    # A file where a directory is expected makes mkdir/open fail underneath;
    # record() must swallow that rather than propagate it.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a dir")
    capture = ReplayCapture(blocker)
    capture.record(
        request_body={"a": 1},
        response_usage=None,
        transforms_applied=[],
        model="m",
        request_id="req-2",
        provider="anthropic",
        timestamp=1.0,
    )  # must not raise


def test_rotation_starts_a_new_file_past_max_size(tmp_path) -> None:
    capture = ReplayCapture(tmp_path, max_file_bytes=1)
    for i in range(3):
        capture.record(
            request_body={"i": i, "padding": "x" * 50},
            response_usage={"output_tokens": i},
            transforms_applied=[],
            model="m",
            request_id=f"req-{i}",
            provider="anthropic",
            timestamp=float(i),
        )
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 3
    total_records = sum(len(f.read_text().splitlines()) for f in files)
    assert total_records == 3


def test_iter_replay_records_reads_directory_in_rotation_order(tmp_path) -> None:
    (tmp_path / "replay-0000.jsonl").write_text(
        json.dumps({"request_id": "a", "request": {}, "response_usage": {}}) + "\n"
    )
    (tmp_path / "replay-0001.jsonl").write_text(
        json.dumps({"request_id": "b", "request": {}, "response_usage": {}}) + "\n" + "not json\n"
    )
    records = list(iter_replay_records(tmp_path))
    assert [r["request_id"] for r in records] == ["a", "b"]


def test_iter_replay_pairs_projects_request_and_usage(tmp_path) -> None:
    path = tmp_path / "replay-0000.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "a",
                "request": {"model": "claude", "messages": []},
                "response_usage": {"output_tokens": 3},
            }
        )
        + "\n"
    )
    pairs = list(iter_replay_pairs(path))
    assert len(pairs) == 1
    request, usage = pairs[0]
    assert request["model"] == "claude"
    assert usage == {"output_tokens": 3}


def test_iter_replay_records_missing_source_yields_nothing(tmp_path) -> None:
    assert list(iter_replay_records(tmp_path / "missing")) == []


def test_get_replay_capture_off_by_default() -> None:
    assert get_replay_capture() is None


def test_get_replay_capture_on_when_env_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    capture = get_replay_capture()
    assert capture is not None
    assert isinstance(capture, ReplayCapture)
    # Cached: a second call returns the same instance without re-reading env.
    assert get_replay_capture() is capture


def test_get_replay_capture_respects_max_bytes_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    monkeypatch.setenv("HEADROOM_REPLAY_CAPTURE_MAX_BYTES", "1024")
    capture = get_replay_capture()
    assert capture._max_file_bytes == 1024


def test_get_replay_capture_ignores_bad_max_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    monkeypatch.setenv("HEADROOM_REPLAY_CAPTURE_MAX_BYTES", "not-an-int")
    capture = get_replay_capture()
    from headroom.proxy.replay_capture import DEFAULT_MAX_FILE_BYTES

    assert capture._max_file_bytes == DEFAULT_MAX_FILE_BYTES


def test_restart_resumes_at_the_open_tail(tmp_path) -> None:
    """A new writer must not touch the full files an earlier one left behind."""
    first = ReplayCapture(tmp_path, max_file_bytes=100)
    for i in range(3):
        first.record(
            request_body={"i": i, "padding": "x" * 80},
            response_usage={},
            transforms_applied=[],
            model="m",
            request_id=f"req-{i}",
            provider="anthropic",
            timestamp=float(i),
        )
    before = {f.name: f.read_bytes() for f in tmp_path.glob("*.jsonl")}
    assert len(before) == 3

    second = ReplayCapture(tmp_path, max_file_bytes=100)
    second.record(
        request_body={"i": 99, "padding": "x" * 80},
        response_usage={},
        transforms_applied=[],
        model="m",
        request_id="req-99",
        provider="anthropic",
        timestamp=99.0,
    )
    after = {f.name: f.read_bytes() for f in tmp_path.glob("*.jsonl")}
    for name, content in before.items():
        assert after[name] == content
    assert sorted(after) == sorted(before) + ["replay-0003.jsonl"]
    assert json.loads(after["replay-0003.jsonl"])["request_id"] == "req-99"
