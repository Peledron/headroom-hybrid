"""The output side is the half of the bill nothing in this proxy touches.

These tests pin the accounting, not any particular number. The load-bearing
property is that the two turn shapes stay separated: a mean taken across both
would hide whether the cost lives in tool loops or in direct answers, which is
the only thing this counter exists to say.
"""

from __future__ import annotations

import pytest

from headroom.proxy.output_accounting import (
    TURN_SHAPE_CONTINUATION,
    TURN_SHAPE_FRESH,
    TURN_SHAPE_UNKNOWN,
    OutputTokenLedger,
    classify_turn_shape,
    shared_output_ledger,
)


def _tool_result_turn(text: str = "ok") -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "read the file"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": text}],
        },
    ]


class TestClassification:
    def test_a_returned_tool_result_means_the_model_is_mid_task(self):
        assert classify_turn_shape(_tool_result_turn()) == TURN_SHAPE_CONTINUATION

    def test_a_plain_user_message_is_a_fresh_ask(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "why?"}]}]
        assert classify_turn_shape(messages) == TURN_SHAPE_FRESH

    def test_a_string_content_body_is_a_fresh_ask(self):
        assert classify_turn_shape([{"role": "user", "content": "why?"}]) == TURN_SHAPE_FRESH

    def test_only_the_last_message_decides(self):
        """Every session past its first tool call contains tool_result blocks."""
        messages = _tool_result_turn()
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "next"}]})
        assert classify_turn_shape(messages) == TURN_SHAPE_FRESH

    def test_a_mixed_block_still_counts_as_a_continuation(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                    {"type": "text", "text": "and also rename it"},
                ],
            }
        ]
        assert classify_turn_shape(messages) == TURN_SHAPE_CONTINUATION

    @pytest.mark.parametrize("messages", [None, [], "not a list", [None], [["nested"]], 42])
    def test_an_unreadable_shape_is_reported_not_guessed(self, messages):
        """A wrong bucket corrupts the comparison the counter exists to make."""
        assert classify_turn_shape(messages) == TURN_SHAPE_UNKNOWN


class TestLedger:
    def test_a_response_is_recorded_against_its_shape(self):
        ledger = OutputTokenLedger()
        observation = ledger.observe(TURN_SHAPE_FRESH, 500)
        assert observation is not None
        assert observation.count == 1
        assert observation.mean == 500

    def test_the_shapes_do_not_pool(self):
        ledger = OutputTokenLedger()
        ledger.observe(TURN_SHAPE_FRESH, 100)
        ledger.observe(TURN_SHAPE_CONTINUATION, 900)
        shapes = ledger.snapshot()["shapes"]
        assert shapes[TURN_SHAPE_FRESH]["mean"] == 100
        assert shapes[TURN_SHAPE_CONTINUATION]["mean"] == 900

    def test_a_missing_usage_block_is_not_a_zero_token_response(self):
        """Errors and some streaming paths report nothing. Counting those as
        zero would drag every mean down and make the split unreadable."""
        ledger = OutputTokenLedger()
        assert ledger.observe(TURN_SHAPE_FRESH, 0) is None
        assert ledger.observe(TURN_SHAPE_FRESH, -5) is None
        assert ledger.snapshot()["responses"] == 0

    def test_a_nameless_shape_is_dropped(self):
        ledger = OutputTokenLedger()
        assert ledger.observe("", 500) is None

    def test_a_non_integer_count_is_dropped(self):
        ledger = OutputTokenLedger()
        assert ledger.observe(TURN_SHAPE_FRESH, "500") is None  # type: ignore[arg-type]

    def test_the_max_survives_a_later_small_response(self):
        ledger = OutputTokenLedger()
        ledger.observe(TURN_SHAPE_FRESH, 8_000)
        ledger.observe(TURN_SHAPE_FRESH, 10)
        assert ledger.snapshot()["shapes"][TURN_SHAPE_FRESH]["max"] == 8_000

    def test_the_median_ignores_a_single_outlier(self):
        """The mean is what a 36k turn moves. The median says whether that turn
        was the workload or an accident."""
        ledger = OutputTokenLedger()
        for _ in range(20):
            ledger.observe(TURN_SHAPE_FRESH, 100)
        ledger.observe(TURN_SHAPE_FRESH, 36_000)
        shape = ledger.snapshot()["shapes"][TURN_SHAPE_FRESH]
        assert shape["median"] == 100
        assert shape["mean"] > 100

    def test_lifetime_totals_outlive_the_median_window(self):
        from headroom.proxy.output_accounting import RECENT_WINDOW

        ledger = OutputTokenLedger()
        for _ in range(RECENT_WINDOW + 50):
            ledger.observe(TURN_SHAPE_FRESH, 10)
        shape = ledger.snapshot()["shapes"][TURN_SHAPE_FRESH]
        assert shape["count"] == RECENT_WINDOW + 50
        assert shape["total"] == (RECENT_WINDOW + 50) * 10
        assert shape["window"] == RECENT_WINDOW

    def test_an_empty_ledger_reports_nothing_rather_than_dividing_by_zero(self):
        assert OutputTokenLedger().snapshot() == {
            "shapes": {},
            "total_output_tokens": 0,
            "responses": 0,
        }

    def test_reset_clears_the_shapes(self):
        ledger = OutputTokenLedger()
        ledger.observe(TURN_SHAPE_FRESH, 500)
        ledger.reset()
        assert ledger.snapshot()["responses"] == 0


class TestLogLine:
    def test_the_log_line_carries_no_prompt_text(self):
        """This runs on every response, so it must never hold conversation."""
        ledger = OutputTokenLedger()
        observation = ledger.observe(TURN_SHAPE_CONTINUATION, 1_234)
        assert observation is not None
        fields = observation.as_log_fields()
        assert "out=1234" in fields
        assert f"shape={TURN_SHAPE_CONTINUATION}" in fields
        assert "p50=" in fields


class TestSharedLedger:
    def test_the_handlers_write_to_one_process_wide_ledger(self):
        assert shared_output_ledger() is shared_output_ledger()


class TestHandlerWiring:
    """The counter is worthless if the response path never calls it.

    The ledger and the classifier can both be correct while the handler misses
    the call site, and that failure is silent by construction, since the helper
    swallows its own exceptions. So drive a real request through the proxy.
    """

    @pytest.fixture(autouse=True)
    def _clean_ledger(self):
        shared_output_ledger().reset()
        yield
        shared_output_ledger().reset()

    def _app(self):
        from headroom.proxy.server import ProxyConfig, create_app

        return create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                ccr_inject_tool=False,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                mode="token",
            )
        )

    def _install_fake_upstream(self, proxy, output_tokens: int):
        from unittest.mock import AsyncMock, MagicMock

        import httpx

        response = httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 10, "output_tokens": output_tokens},
            },
            request=httpx.Request("POST", "http://upstream/v1/messages"),
        )
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        client.request = AsyncMock(return_value=response)
        client.send = AsyncMock(return_value=response)
        client.build_request = MagicMock(
            return_value=httpx.Request("POST", "http://upstream/v1/messages", content=b"{}")
        )
        client.aclose = AsyncMock()
        proxy.http_client = client
        return client

    def _post(self, content):
        from fastapi.testclient import TestClient

        app = self._app()
        with TestClient(app) as client:
            self._install_fake_upstream(client.app.state.proxy, 777)
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": content}],
                },
            )
        assert resp.status_code == 200
        return shared_output_ledger().snapshot()

    def test_a_buffered_response_lands_in_the_ledger(self):
        snapshot = self._post("hi")
        assert snapshot["responses"] == 1
        assert snapshot["total_output_tokens"] == 777
        assert TURN_SHAPE_FRESH in snapshot["shapes"]

    def test_a_tool_continuation_is_bucketed_apart_end_to_end(self):
        snapshot = self._post([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}])
        assert list(snapshot["shapes"]) == [TURN_SHAPE_CONTINUATION]

    def test_both_response_paths_observe(self):
        """The streaming path bills output the same way the buffered one does.

        Driving a full SSE upstream through the test client costs far more than
        it proves here, so pin the call sites instead. Losing one of the two
        would halve the sample without any test going red.
        """
        import inspect

        from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

        src = inspect.getsource(AnthropicHandlerMixin.handle_anthropic_messages)
        assert src.count("_observe_output_tokens(") == 2

    def test_the_diag_line_reaches_the_log(self, caplog: pytest.LogCaptureFixture):
        """Attached directly because ``setup_logging`` clears propagation.

        `headroom/proxy/helpers.py:995` sets ``propagate = False`` on the proxy
        logger, so caplog's root handler never sees these records once an app
        has been created.
        """
        import logging

        proxy_logger = logging.getLogger("headroom.proxy")
        proxy_logger.addHandler(caplog.handler)
        previous = proxy_logger.level
        proxy_logger.setLevel(logging.INFO)
        try:
            self._post("hi")
        finally:
            proxy_logger.removeHandler(caplog.handler)
            proxy_logger.setLevel(previous)

        lines = [r.getMessage() for r in caplog.records if "OUTPUT_DIAG" in r.getMessage()]
        assert len(lines) == 1
        assert f"shape={TURN_SHAPE_FRESH}" in lines[0]
        assert "out=777" in lines[0]
