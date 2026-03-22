"""
Tests for prysmai.session — unified proxy + governance session scope.
"""

import pytest
from unittest.mock import patch, MagicMock

from prysmai import PrysmClient
from prysmai.context import prysm_context
from prysmai.session import PrysmSession, PrysmSessionIdentifiers


VALID_KEY = "sk-prysm-test1234567890abcdef"
VALID_URL = "https://prysmai.io/api/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("PRYSM_API_KEY", raising=False)
    monkeypatch.delenv("PRYSM_BASE_URL", raising=False)
    prysm_context.clear()


class TestPrysmSession:
    def test_session_without_governance_sets_context(self):
        prysm = PrysmClient(prysm_key=VALID_KEY, base_url=VALID_URL)

        with prysm.session(
            user_id="user_123",
            session_id="sess_abc",
            metadata={"feature": "chat"},
        ) as run:
            assert isinstance(run, PrysmSession)
            assert run.is_active is True
            assert run.governance is None
            assert run.governance_session_id is None

            ctx = prysm_context.get()
            assert ctx.user_id == "user_123"
            assert ctx.session_id == "sess_abc"
            assert ctx.metadata == {"feature": "chat"}

            ids = run.identifiers
            assert isinstance(ids, PrysmSessionIdentifiers)
            assert ids.session_id == "sess_abc"
            assert ids.user_id == "user_123"

        ctx = prysm_context.get()
        assert ctx.user_id is None
        assert ctx.session_id is None
        assert run.is_active is False

    def test_session_openai_delegates_to_prysm_client(self):
        prysm = PrysmClient(prysm_key=VALID_KEY, base_url=VALID_URL)
        with patch.object(prysm, "openai", return_value="proxy-client") as mock_openai:
            with prysm.session() as run:
                result = run.openai()
        assert result == "proxy-client"
        mock_openai.assert_called_once()

    @patch("prysmai.session.GovernanceSession")
    def test_session_with_governance_links_context(self, mock_gov_cls):
        mock_gov = MagicMock()
        mock_gov.session_id = "gov_123"
        mock_gov.is_active = True
        mock_gov.end.return_value = "report"
        mock_gov_cls.return_value = mock_gov

        prysm = PrysmClient(prysm_key=VALID_KEY, base_url=VALID_URL)

        with prysm.session(
            user_id="user_123",
            session_id="sess_abc",
            metadata={"feature": "agent"},
            governance_task="Investigate issue",
            agent_type="codex",
            available_tools=["search"],
            governance_context={"repo": "demo"},
            auto_check_interval=3,
        ) as run:
            assert run.governance is mock_gov
            assert run.governance_session_id == "gov_123"

            ctx = prysm_context.get()
            assert ctx.session_id == "sess_abc"
            assert ctx.governance_session_id == "gov_123"

            run.report_event("tool_call", {"tool_name": "search"})
            run.scan_code("print('ok')", "python")
            run.check_behavior([{"event_type": "decision", "data": {"ok": True}}])

        mock_gov_cls.assert_called_once()
        mock_gov.start.assert_called_once()
        mock_gov.report_event.assert_called_once()
        mock_gov.scan_code.assert_called_once()
        mock_gov.check_behavior.assert_called_once()
        mock_gov.end.assert_called_once()
        mock_gov.close.assert_called_once()
        assert run.governance_report == "report"

    @patch("prysmai.session.GovernanceSession")
    def test_session_parity_helpers_use_governance_session(self, mock_gov_cls):
        mock_gov = MagicMock()
        mock_gov.session_id = "gov_123"
        mock_gov.is_active = True
        mock_gov.end.return_value = "report"
        mock_gov_cls.return_value = mock_gov

        prysm = PrysmClient(prysm_key=VALID_KEY, base_url=VALID_URL)
        mock_mcp = MagicMock()
        mock_mcp.record_llm_call.return_value = {"trace_id": "trace_1"}
        mock_mcp.record_tool_call.return_value = {"event_id": 11}
        mock_mcp.record_decision.return_value = {"event_id": 12}
        mock_mcp.record_file_change.return_value = {"event_id": 13}

        with patch.object(prysm, "mcp", return_value=mock_mcp):
            with prysm.session(
                governance_task="Investigate issue",
                agent_type="codex",
            ) as run:
                run.record_llm_call(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                )
                run.record_tool_call(
                    tool_name="search",
                    tool_input={"query": "docs"},
                )
                run.record_decision(
                    description="Escalate",
                    selected_action="escalate",
                )
                run.record_file_change(
                    operation="write",
                    path="src/app.py",
                )

        mock_mcp.record_llm_call.assert_called_once_with(
            session_id="gov_123",
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello"}],
            provider=None,
            completion=None,
            finish_reason=None,
            status="success",
            status_code=None,
            error_message=None,
            latency_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost_usd=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            metadata=None,
            logprobs=None,
            run_security_scan=True,
        )
        mock_mcp.record_tool_call.assert_called_once_with(
            session_id="gov_123",
            tool_name="search",
            tool_input={"query": "docs"},
            tool_output=None,
            success=None,
            duration_ms=None,
            external_trace_id=None,
            metadata=None,
        )
        mock_mcp.record_decision.assert_called_once_with(
            session_id="gov_123",
            description="Escalate",
            rationale=None,
            selected_action="escalate",
            severity=None,
            external_trace_id=None,
            metadata=None,
        )
        mock_mcp.record_file_change.assert_called_once_with(
            session_id="gov_123",
            operation="write",
            path="src/app.py",
            language=None,
            content=None,
            external_trace_id=None,
            metadata=None,
        )

    @patch("prysmai.session.GovernanceSession")
    def test_run_tool_records_success(self, mock_gov_cls):
        mock_gov = MagicMock()
        mock_gov.session_id = "gov_123"
        mock_gov.is_active = True
        mock_gov.end.return_value = "report"
        mock_gov_cls.return_value = mock_gov

        prysm = PrysmClient(prysm_key=VALID_KEY, base_url=VALID_URL)
        mock_mcp = MagicMock()
        mock_mcp.record_tool_call.return_value = {"event_id": 1}

        with patch.object(prysm, "mcp", return_value=mock_mcp):
            with prysm.session(
                governance_task="Investigate issue",
                agent_type="codex",
            ) as run:
                result = run.run_tool(
                    "search",
                    lambda query: {"answer": "ok", "query": query},
                    "docs",
                    tool_input={"query": "docs"},
                )

        assert result == {"answer": "ok", "query": "docs"}
        _, kwargs = mock_mcp.record_tool_call.call_args
        assert kwargs["session_id"] == "gov_123"
        assert kwargs["tool_name"] == "search"
        assert kwargs["tool_input"] == {"query": "docs"}
        assert kwargs["tool_output"] == {"answer": "ok", "query": "docs"}
        assert kwargs["success"] is True
        assert isinstance(kwargs["duration_ms"], int)

    @patch("prysmai.session.GovernanceSession")
    def test_run_tool_records_failure(self, mock_gov_cls):
        mock_gov = MagicMock()
        mock_gov.session_id = "gov_123"
        mock_gov.is_active = True
        mock_gov.end.return_value = "report"
        mock_gov_cls.return_value = mock_gov

        prysm = PrysmClient(prysm_key=VALID_KEY, base_url=VALID_URL)
        mock_mcp = MagicMock()
        mock_mcp.record_tool_call.return_value = {"event_id": 1}

        def _boom():
            raise RuntimeError("search failed")

        with patch.object(prysm, "mcp", return_value=mock_mcp):
            with pytest.raises(RuntimeError, match="search failed"):
                with prysm.session(
                    governance_task="Investigate issue",
                    agent_type="codex",
                ) as run:
                    run.run_tool("search", _boom, tool_input={"query": "docs"})

        _, kwargs = mock_mcp.record_tool_call.call_args
        assert kwargs["session_id"] == "gov_123"
        assert kwargs["tool_name"] == "search"
        assert kwargs["tool_input"] == {"query": "docs"}
        assert kwargs["success"] is False
        assert kwargs["tool_output"]["error"] == "search failed"
        assert kwargs["tool_output"]["exception_type"] == "RuntimeError"

    def test_governance_methods_raise_when_not_enabled(self):
        prysm = PrysmClient(prysm_key=VALID_KEY, base_url=VALID_URL)
        with prysm.session() as run:
            with pytest.raises(RuntimeError, match="Governance is not enabled"):
                run.report_event("tool_call", {"tool_name": "search"})
            with pytest.raises(RuntimeError, match="Governance is not enabled"):
                run.record_tool_call(tool_name="search")
