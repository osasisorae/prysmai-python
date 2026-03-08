"""
Tests for prysmai.governance — GovernanceSession, MCP transport, and data classes.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from prysmai.governance import (
    GovernanceSession,
    GovernanceError,
    SessionNotActiveError,
    CheckResult,
    ScanResult,
    SessionReport,
    BehavioralFlag,
    Vulnerability,
    _parse_sse_response,
    _extract_tool_result,
    _McpTransport,
)


VALID_KEY = "sk-prysm-test1234567890abcdef"
VALID_URL = "https://prysmai.io/api/v1"


# ─── Fixtures ───


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure clean environment for each test."""
    monkeypatch.delenv("PRYSM_API_KEY", raising=False)
    monkeypatch.delenv("PRYSM_BASE_URL", raising=False)


# ─── SSE Parser Tests ───


class TestSSEParser:
    def test_parse_plain_json_with_result(self):
        raw = '{"jsonrpc":"2.0","id":1,"result":{"session_id":"abc123"}}'
        result = _parse_sse_response(raw)
        assert result == {"session_id": "abc123"}

    def test_parse_sse_format(self):
        raw = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
        result = _parse_sse_response(raw)
        assert result == {"ok": True}

    def test_parse_sse_with_multiple_data_lines(self):
        raw = (
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"first":true}}\n\n'
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":2,"result":{"second":true}}\n\n'
        )
        result = _parse_sse_response(raw)
        # Should return the last valid result
        assert result == {"second": True}

    def test_parse_sse_error_raises(self):
        raw = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"error":{"code":-32603,"message":"Internal error"}}\n\n'
        with pytest.raises(GovernanceError, match="Internal error"):
            _parse_sse_response(raw)

    def test_parse_plain_json_error_raises(self):
        raw = '{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Invalid request"}}'
        with pytest.raises(GovernanceError, match="Invalid request"):
            _parse_sse_response(raw)

    def test_parse_unparseable_raises(self):
        raw = "this is not json or sse"
        with pytest.raises(GovernanceError, match="Failed to parse"):
            _parse_sse_response(raw)

    def test_parse_empty_data_lines_skipped(self):
        raw = 'event: message\ndata: \ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
        result = _parse_sse_response(raw)
        assert result == {"ok": True}


class TestExtractToolResult:
    def test_extract_json_content(self):
        mcp_result = {
            "content": [{"type": "text", "text": '{"session_id":"abc","started_at":"2026-03-08"}'}]
        }
        result = _extract_tool_result(mcp_result)
        assert result["session_id"] == "abc"
        assert result["started_at"] == "2026-03-08"

    def test_extract_empty_content(self):
        assert _extract_tool_result({"content": []}) == {}
        assert _extract_tool_result({}) == {}

    def test_extract_non_json_content(self):
        mcp_result = {"content": [{"type": "text", "text": "not json"}]}
        result = _extract_tool_result(mcp_result)
        assert result == {"raw": "not json"}


# ─── Data Class Tests ───


class TestDataClasses:
    def test_check_result_has_flags(self):
        cr = CheckResult(
            events_ingested=5,
            flags=[BehavioralFlag(detector="early_stopping", severity=75)],
            violations=[],
            recommendations=["Review agent behavior"],
            raw={},
        )
        assert cr.has_flags is True
        assert cr.max_severity == 75

    def test_check_result_no_flags(self):
        cr = CheckResult(
            events_ingested=5, flags=[], violations=[], recommendations=[], raw={}
        )
        assert cr.has_flags is False
        assert cr.max_severity == 0

    def test_scan_result_is_clean(self):
        sr = ScanResult(
            language="python",
            file_path="app.py",
            vulnerability_count=0,
            max_severity="none",
            threat_score=0,
            vulnerabilities=[],
            recommendations=[],
            raw={},
        )
        assert sr.is_clean is True

    def test_scan_result_has_vulnerabilities(self):
        sr = ScanResult(
            language="python",
            file_path="app.py",
            vulnerability_count=2,
            max_severity="high",
            threat_score=85,
            vulnerabilities=[
                Vulnerability(type="command_injection", severity="high", description="Unsafe subprocess call"),
            ],
            recommendations=["Use subprocess with shell=False"],
            raw={},
        )
        assert sr.is_clean is False

    def test_behavioral_flag_defaults(self):
        flag = BehavioralFlag(detector="test", severity=50)
        assert flag.evidence == []

    def test_session_report_fields(self):
        report = SessionReport(
            session_id="sess-123",
            outcome="completed",
            behavior_score=85,
            detectors=[{"name": "early_stopping", "triggered": False}],
            summary="Clean session",
            recommendations=[],
            violations=[],
            raw={},
        )
        assert report.session_id == "sess-123"
        assert report.behavior_score == 85


# ─── McpTransport Tests ───


class TestMcpTransport:
    def test_mcp_url_derivation(self):
        t = _McpTransport(base_url="https://prysmai.io/api/v1", api_key=VALID_KEY)
        assert t.mcp_url == "https://prysmai.io/api/mcp"

    def test_mcp_url_strips_v1(self):
        t = _McpTransport(base_url="https://custom.host/v1", api_key=VALID_KEY)
        assert t.mcp_url == "https://custom.host/api/mcp"

    def test_mcp_url_strips_api(self):
        t = _McpTransport(base_url="https://custom.host/api", api_key=VALID_KEY)
        assert t.mcp_url == "https://custom.host/api/mcp"

    def test_mcp_url_no_suffix(self):
        t = _McpTransport(base_url="https://custom.host", api_key=VALID_KEY)
        assert t.mcp_url == "https://custom.host/api/mcp"

    @patch("prysmai.governance.httpx.Client")
    def test_call_tool_sends_correct_payload(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{\\"ok\\":true}"}]}}\n\n'
        )
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        t = _McpTransport(base_url=VALID_URL, api_key=VALID_KEY)
        result = t.call_tool("prysm_session_start", {"task_instructions": "test"})

        assert result == {"ok": True}
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "prysm_session_start"
        assert payload["params"]["arguments"]["task_instructions"] == "test"
        assert "Bearer" in call_args[1]["headers"]["Authorization"]

    @patch("prysmai.governance.httpx.Client")
    def test_call_tool_raises_on_tool_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            'event: message\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"isError":true,"content":[{"type":"text","text":"{\\"error\\":\\"Session not found\\"}"}]}}\n\n'
        )
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        t = _McpTransport(base_url=VALID_URL, api_key=VALID_KEY)
        with pytest.raises(GovernanceError, match="Session not found"):
            t.call_tool("prysm_check_behavior", {"session_id": "bad"})


# ─── GovernanceSession Tests ───


class TestGovernanceSessionInit:
    def test_init_with_direct_key(self):
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test task")
        assert gov._api_key == VALID_KEY
        assert gov._task == "test task"
        assert gov.is_active is False
        assert gov.session_id is None

    def test_init_with_client(self):
        client = MagicMock()
        client.prysm_key = VALID_KEY
        client.base_url = VALID_URL
        gov = GovernanceSession(client=client, task="test", agent_type="claude_code")
        assert gov._api_key == VALID_KEY
        assert gov._agent_type == "claude_code"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("PRYSM_API_KEY", VALID_KEY)
        gov = GovernanceSession(task="env test")
        assert gov._api_key == VALID_KEY

    def test_init_no_key_raises(self):
        with pytest.raises(ValueError, match="API key is required"):
            GovernanceSession(task="no key")

    def test_repr(self):
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test")
        assert "inactive" in repr(gov)
        assert "GovernanceSession" in repr(gov)

    def test_stats(self):
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test")
        stats = gov.stats
        assert stats["active"] is False
        assert stats["total_events_reported"] == 0
        assert stats["checks_performed"] == 0


class TestGovernanceSessionLifecycle:
    @patch.object(_McpTransport, "call_tool")
    def test_start_sets_active(self, mock_call):
        mock_call.return_value = {
            "session_id": "sess-abc",
            "started_at": "2026-03-08T00:00:00Z",
            "policies": [{"name": "no-eval"}],
            "message": "Session started",
        }
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test")
        result = gov.start()

        assert gov.is_active is True
        assert gov.session_id == "sess-abc"
        assert len(gov.policies) == 1
        mock_call.assert_called_once_with("prysm_session_start", {
            "task_instructions": "test",
            "agent_type": "custom",
        })

    @patch.object(_McpTransport, "call_tool")
    def test_start_twice_raises(self, mock_call):
        mock_call.return_value = {"session_id": "sess-abc", "started_at": "now", "policies": []}
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test")
        gov.start()
        with pytest.raises(GovernanceError, match="already active"):
            gov.start()

    @patch.object(_McpTransport, "call_tool")
    def test_end_returns_report(self, mock_call):
        mock_call.side_effect = [
            # start
            {"session_id": "sess-abc", "started_at": "now", "policies": []},
            # end
            {
                "session_id": "sess-abc",
                "outcome": "completed",
                "report": {
                    "behavior_score": 92,
                    "detectors": [{"name": "early_stopping", "triggered": False}],
                    "summary": "Clean session",
                    "recommendations": [],
                },
                "violations": [],
            },
        ]
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test")
        gov.start()
        report = gov.end(outcome="completed")

        assert isinstance(report, SessionReport)
        assert report.behavior_score == 92
        assert report.outcome == "completed"
        assert gov.is_active is False

    @patch.object(_McpTransport, "call_tool")
    def test_end_without_start_raises(self, mock_call):
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test")
        with pytest.raises(GovernanceError, match="No active session"):
            gov.end()

    @patch.object(_McpTransport, "call_tool")
    def test_context_manager(self, mock_call):
        mock_call.side_effect = [
            {"session_id": "sess-ctx", "started_at": "now", "policies": []},
            {"session_id": "sess-ctx", "outcome": "completed", "report": {}, "violations": []},
        ]
        with GovernanceSession(prysm_key=VALID_KEY, task="ctx test") as gov:
            assert gov.is_active is True
        assert gov.is_active is False

    @patch.object(_McpTransport, "call_tool")
    def test_context_manager_on_exception(self, mock_call):
        mock_call.side_effect = [
            {"session_id": "sess-err", "started_at": "now", "policies": []},
            {"session_id": "sess-err", "outcome": "failed", "report": {}, "violations": []},
        ]
        try:
            with GovernanceSession(prysm_key=VALID_KEY, task="err test") as gov:
                raise ValueError("something broke")
        except ValueError:
            pass

        assert gov.is_active is False
        # The end call should have been made with outcome="failed"
        end_call = mock_call.call_args_list[-1]
        assert end_call[0][1]["outcome"] == "failed"


class TestGovernanceSessionOperations:
    @patch.object(_McpTransport, "call_tool")
    def test_check_behavior(self, mock_call):
        mock_call.side_effect = [
            {"session_id": "sess-chk", "started_at": "now", "policies": []},
            {
                "events_ingested": 3,
                "flags": [{"detector": "early_stopping", "severity": 80, "evidence": []}],
                "violations": [],
                "recommendations": ["Review early stopping"],
            },
        ]
        gov = GovernanceSession(prysm_key=VALID_KEY, task="check test")
        gov.start()
        result = gov.check_behavior([
            {"event_type": "llm_call", "data": {"model": "gpt-4"}},
            {"event_type": "tool_call", "data": {"tool_name": "bash"}},
            {"event_type": "llm_call", "data": {"model": "gpt-4"}},
        ])

        assert isinstance(result, CheckResult)
        assert result.events_ingested == 3
        assert result.has_flags is True
        assert result.max_severity == 80
        assert len(result.flags) == 1
        assert gov._total_events_reported == 3
        assert gov._checks_performed == 1

    @patch.object(_McpTransport, "call_tool")
    def test_check_behavior_not_active_raises(self, mock_call):
        gov = GovernanceSession(prysm_key=VALID_KEY, task="test")
        with pytest.raises(SessionNotActiveError):
            gov.check_behavior([{"event_type": "llm_call", "data": {}}])

    @patch.object(_McpTransport, "call_tool")
    def test_scan_code(self, mock_call):
        mock_call.side_effect = [
            {"session_id": "sess-scan", "started_at": "now", "policies": []},
            {
                "language": "python",
                "file_path": "app.py",
                "vulnerability_count": 1,
                "max_severity": "high",
                "threat_score": 85,
                "vulnerabilities": [
                    {"type": "command_injection", "severity": "high", "description": "Unsafe subprocess"}
                ],
                "recommendations": ["Use subprocess with shell=False"],
            },
        ]
        gov = GovernanceSession(prysm_key=VALID_KEY, task="scan test")
        gov.start()
        result = gov.scan_code(
            code="subprocess.call(user_input, shell=True)",
            language="python",
            file_path="app.py",
        )

        assert isinstance(result, ScanResult)
        assert result.is_clean is False
        assert result.vulnerability_count == 1
        assert result.threat_score == 85
        assert len(result.vulnerabilities) == 1
        assert gov._scans_performed == 1

    @patch.object(_McpTransport, "call_tool")
    def test_report_event_auto_check(self, mock_call):
        mock_call.side_effect = [
            {"session_id": "sess-auto", "started_at": "now", "policies": []},
            {"events_ingested": 3, "flags": [], "violations": [], "recommendations": []},
        ]
        gov = GovernanceSession(
            prysm_key=VALID_KEY, task="auto check", auto_check_interval=3
        )
        gov.start()

        # First two events should just buffer
        r1 = gov.report_event("llm_call", {"model": "gpt-4"})
        assert r1 is None
        r2 = gov.report_event("tool_call", {"tool_name": "bash"})
        assert r2 is None

        # Third event should trigger auto-check
        r3 = gov.report_event("llm_call", {"model": "gpt-4"})
        assert isinstance(r3, CheckResult)
        assert r3.events_ingested == 3

    @patch.object(_McpTransport, "call_tool")
    def test_all_flags_accumulate(self, mock_call):
        mock_call.side_effect = [
            {"session_id": "sess-flags", "started_at": "now", "policies": []},
            {"events_ingested": 1, "flags": [{"detector": "d1", "severity": 50}], "violations": [], "recommendations": []},
            {"events_ingested": 1, "flags": [{"detector": "d2", "severity": 70}], "violations": [], "recommendations": []},
        ]
        gov = GovernanceSession(prysm_key=VALID_KEY, task="flags test")
        gov.start()
        gov.check_behavior([{"event_type": "llm_call", "data": {}}])
        gov.check_behavior([{"event_type": "tool_call", "data": {}}])

        assert len(gov.all_flags) == 2
        assert gov.all_flags[0].detector == "d1"
        assert gov.all_flags[1].detector == "d2"
