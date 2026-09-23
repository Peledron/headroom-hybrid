"""`headroom wrap claude` prefix-cache stability + opt-in prompt trims.

A Claude Code session's cached prefix is tools + system + history, keyed on exact
bytes, so anything that mutates tools/system mid-session forces a full cold
re-write. Skills/plugins load async by default, so their descriptions can land
after turn 1 — these defaults move that work to startup instead.

Prompt trims are separate and OFF by default: each one hides a capability from
the model, which is the class the quiet-CLI defaults deliberately refuse to touch.
"""

from __future__ import annotations

from headroom.cli.wrap import (
    _configure_prefix_stable_env,
    _configure_trim_prompt_env,
    _prefix_stable_enabled,
    _trim_prompt_enabled,
)


def test_prefix_stable_off_by_default(monkeypatch) -> None:
    """Certain latency cost + unproven benefit => the operator opts in, not us."""
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    monkeypatch.delenv("HEADROOM_WRAP_PREFIX_STABLE", raising=False)
    assert _prefix_stable_enabled() is False
    env: dict[str, str] = {}
    assert _configure_prefix_stable_env(env) == []
    assert env == {}


def test_prefix_stable_defaults_injected_when_opted_in(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    monkeypatch.setenv("HEADROOM_WRAP_PREFIX_STABLE", "1")
    env: dict[str, str] = {}
    written = _configure_prefix_stable_env(env)
    assert env["CLAUDE_CODE_SYNC_SKILLS"] == "1"
    assert env["CLAUDE_CODE_SYNC_PLUGINS"] == "1"
    assert set(written) == {"CLAUDE_CODE_SYNC_SKILLS", "CLAUDE_CODE_SYNC_PLUGINS"}


def test_prefix_stable_user_value_wins(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    monkeypatch.setenv("HEADROOM_WRAP_PREFIX_STABLE", "1")
    env = {"CLAUDE_CODE_SYNC_SKILLS": "0"}
    written = _configure_prefix_stable_env(env)
    assert env["CLAUDE_CODE_SYNC_SKILLS"] == "0"  # untouched
    assert "CLAUDE_CODE_SYNC_SKILLS" not in written
    assert env["CLAUDE_CODE_SYNC_PLUGINS"] == "1"  # absent one still filled


def test_prefix_stable_honors_the_shared_kill_switch(monkeypatch) -> None:
    """HEADROOM_WRAP_QUIET=0 means "leave the launched agent alone" — even if opted in."""
    monkeypatch.setenv("HEADROOM_WRAP_PREFIX_STABLE", "1")
    monkeypatch.setenv("HEADROOM_WRAP_QUIET", "0")
    env: dict[str, str] = {}
    assert _configure_prefix_stable_env(env) == []
    assert env == {}


def test_prefix_stable_accepts_the_usual_truthy_spellings(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    for on in ("1", "true", "yes", "ON"):
        monkeypatch.setenv("HEADROOM_WRAP_PREFIX_STABLE", on)
        assert _prefix_stable_enabled() is True


def test_prefix_stable_defaults_only_reorder_never_remove() -> None:
    """Guard the safety rule: nothing here may disable a capability.

    Both knobs are `SYNC_*` (when work happens), not `DISABLE_*` (whether a
    capability exists). A `DISABLE_*`/`SKIP_*` name landing in this set would be
    a silent capability removal, which belongs behind the opt-in trim flag.
    """
    from headroom.cli.wrap import _PREFIX_STABLE_DEFAULTS

    for name in _PREFIX_STABLE_DEFAULTS:
        assert "DISABLE" not in name and "SKIP" not in name, name


def test_trim_prompt_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_TRIM_PROMPT", raising=False)
    assert _trim_prompt_enabled() is False
    env: dict[str, str] = {}
    assert _configure_trim_prompt_env(env) == []
    assert env == {}  # capability-removing knobs are never set implicitly


def test_trim_prompt_opt_in(monkeypatch) -> None:
    for on in ("1", "true", "yes", "ON"):
        monkeypatch.setenv("HEADROOM_WRAP_TRIM_PROMPT", on)
        assert _trim_prompt_enabled() is True
        env: dict[str, str] = {}
        written = _configure_trim_prompt_env(env)
        assert env["CLAUDE_CODE_DISABLE_BUNDLED_SKILLS"] == "1"
        assert env["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"] == "1"
        assert len(written) == 4


def test_trim_prompt_user_value_wins(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_WRAP_TRIM_PROMPT", "1")
    env = {"CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "0"}
    written = _configure_trim_prompt_env(env)
    assert env["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"] == "0"
    assert "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS" not in written
