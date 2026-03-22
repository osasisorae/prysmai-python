"""
Tests for prysmai.mcp — public MCP client surface.
"""

import pytest
from unittest.mock import patch

from prysmai import PrysmClient
from prysmai.governance import GovernanceSession, _McpTransport
from prysmai.mcp import PrysmMCPClient, PrysmMCPConfig


VALID_KEY = "sk-prysm-test1234567890abcdef"
VALID_URL = "https://prysmai.io/api/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("PRYSM_API_KEY", raising=False)
    monkeypatch.delenv("PRYSM_BASE_URL", raising=False)


class TestPrysmMCPClientInit:
    def test_init_with_direct_key(self):
        client = PrysmMCPClient(prysm_key=VALID_KEY)
        assert client.prysm_key == VALID_KEY
        assert client.base_url == VALID_URL

    def test_init_with_prysm_client(self):
        prysm = PrysmClient(prysm_key=VALID_KEY, base_url="https://custom.host/api/v1")
        client = prysm.mcp()
        assert isinstance(client, PrysmMCPClient)
        assert client.prysm_key == VALID_KEY
        assert client.base_url == "https://custom.host/api/v1"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Prysm API key is required"):
            PrysmMCPClient()

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Invalid Prysm API key format"):
            PrysmMCPClient(prysm_key="sk-openai-bad")


class TestPrysmMCPClientAPI:
    def test_mcp_url_derived_from_base_url(self):
        client = PrysmMCPClient(
            prysm_key=VALID_KEY,
            base_url="https://custom.host/api/v1",
        )
        assert client.mcp_url == "https://custom.host/api/mcp"

    def test_connection_config(self):
        client = PrysmMCPClient(prysm_key=VALID_KEY)
        config = client.connection_config(extra_headers={"X-Test": "1"})

        assert isinstance(config, PrysmMCPConfig)
        assert config.server_url == "https://prysmai.io/api/mcp"
        assert config.transport == "streamable_http"
        assert config.headers["Authorization"] == f"Bearer {VALID_KEY}"
        assert config.headers["X-Test"] == "1"

    @patch.object(_McpTransport, "list_tools")
    def test_list_tools_delegates_to_transport(self, mock_list_tools):
        mock_list_tools.return_value = [{"name": "prysm_session_start"}]
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        tools = client.list_tools()

        assert tools == [{"name": "prysm_session_start"}]
        mock_list_tools.assert_called_once()

    @patch.object(_McpTransport, "list_resources")
    def test_list_resources_delegates_to_transport(self, mock_list_resources):
        mock_list_resources.return_value = [{"uri": "prysm://policies"}]
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        resources = client.list_resources()

        assert resources == [{"uri": "prysm://policies"}]
        mock_list_resources.assert_called_once()

    @patch.object(_McpTransport, "read_resource")
    def test_list_policies_reads_policy_resource(self, mock_read_resource):
        mock_read_resource.return_value = {
            "contents": [
                {
                    "uri": "prysm://policies",
                    "text": '[{"name": "EU AI Act Human Oversight"}]',
                }
            ]
        }
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        policies = client.list_policies()

        assert policies == [{"name": "EU AI Act Human Oversight"}]
        mock_read_resource.assert_called_once_with("prysm://policies")

    @patch.object(_McpTransport, "read_resource")
    def test_get_session_report_reads_report_resource(self, mock_read_resource):
        mock_read_resource.return_value = {
            "contents": [
                {
                    "uri": "prysm://session/sess_123/report",
                    "text": '{"sessionId": "sess_123", "violations": []}',
                }
            ]
        }
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        report = client.get_session_report("sess_123")

        assert report == {"sessionId": "sess_123", "violations": []}
        mock_read_resource.assert_called_once_with(
            "prysm://session/sess_123/report"
        )

    @patch.object(_McpTransport, "call_tool")
    def test_call_tool_delegates_to_transport(self, mock_call_tool):
        mock_call_tool.return_value = {"ok": True}
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        result = client.call_tool("prysm_session_start", {"task_instructions": "test"})

        assert result == {"ok": True}
        mock_call_tool.assert_called_once_with(
            "prysm_session_start",
            {"task_instructions": "test"},
        )

    @patch.object(_McpTransport, "call_tool")
    def test_record_tool_call_uses_structured_helper(self, mock_call_tool):
        mock_call_tool.return_value = {"event_id": 1}
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        result = client.record_tool_call(
            session_id="sess_123",
            tool_name="search",
            tool_input={"query": "docs"},
            tool_output={"hits": 3},
            success=True,
            duration_ms=42,
        )

        assert result == {"event_id": 1}
        mock_call_tool.assert_called_once_with(
            "prysm_record_tool_call",
            {
                "session_id": "sess_123",
                "tool_name": "search",
                "input": {"query": "docs"},
                "output": {"hits": 3},
                "success": True,
                "duration_ms": 42,
            },
        )

    @patch.object(_McpTransport, "call_tool")
    def test_record_decision_uses_structured_helper(self, mock_call_tool):
        mock_call_tool.return_value = {"event_id": 2}
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        result = client.record_decision(
            session_id="sess_123",
            description="Escalate to human review",
            selected_action="escalate",
            severity="warning",
        )

        assert result == {"event_id": 2}
        mock_call_tool.assert_called_once_with(
            "prysm_record_decision",
            {
                "session_id": "sess_123",
                "description": "Escalate to human review",
                "selected_action": "escalate",
                "severity": "warning",
            },
        )

    @patch.object(_McpTransport, "call_tool")
    def test_record_file_change_uses_structured_helper(self, mock_call_tool):
        mock_call_tool.return_value = {"event_id": 3}
        client = PrysmMCPClient(prysm_key=VALID_KEY)

        result = client.record_file_change(
            session_id="sess_123",
            operation="write",
            path="src/app.py",
            language="python",
            content="print('ok')",
        )

        assert result == {"event_id": 3}
        mock_call_tool.assert_called_once_with(
            "prysm_record_file_change",
            {
                "session_id": "sess_123",
                "operation": "write",
                "path": "src/app.py",
                "language": "python",
                "content": "print('ok')",
            },
        )

    def test_governance_session_uses_same_auth(self):
        client = PrysmMCPClient(prysm_key=VALID_KEY, base_url=VALID_URL, timeout=45.0)

        gov = client.governance_session(task="Review agent run", agent_type="codex")

        assert isinstance(gov, GovernanceSession)
        assert gov._api_key == VALID_KEY
        assert gov._base_url == VALID_URL
        assert gov._timeout == 45.0

    @patch.object(_McpTransport, "close")
    def test_close_closes_transport(self, mock_close):
        client = PrysmMCPClient(prysm_key=VALID_KEY)
        client.close()
        mock_close.assert_called_once()
