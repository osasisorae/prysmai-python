"""
Tests for prysmai.integrations.langgraph — PrysmGraphMonitor.

Uses mocks for langchain_core dependencies so tests run without
the full LangGraph/LangChain stack installed.
"""
import json
import time
import uuid
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# Mock langchain_core before importing the module
import sys

# Create mock modules for langchain_core
mock_callbacks = MagicMock()
mock_outputs = MagicMock()
mock_messages = MagicMock()
mock_agents = MagicMock()

# Create mock classes
class MockBaseCallbackHandler:
    pass

class MockLLMResult:
    def __init__(self, generations=None, llm_output=None):
        self.generations = generations or []
        self.llm_output = llm_output

class MockChatGeneration:
    def __init__(self, text="", message=None):
        self.text = text
        self.message = message

class MockGeneration:
    def __init__(self, text=""):
        self.text = text

class MockBaseMessage:
    def __init__(self, content="", msg_type="human"):
        self.content = content
        self.type = msg_type

class MockAgentAction:
    def __init__(self, tool="", tool_input="", log=""):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

class MockAgentFinish:
    def __init__(self, return_values=None, log=""):
        self.return_values = return_values or {}
        self.log = log

mock_callbacks.BaseCallbackHandler = MockBaseCallbackHandler
mock_outputs.LLMResult = MockLLMResult
mock_outputs.ChatGeneration = MockChatGeneration
mock_outputs.Generation = MockGeneration
mock_messages.BaseMessage = MockBaseMessage
mock_agents.AgentAction = MockAgentAction
mock_agents.AgentFinish = MockAgentFinish

sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.callbacks"] = mock_callbacks
sys.modules["langchain_core.outputs"] = mock_outputs
sys.modules["langchain_core.messages"] = mock_messages
sys.modules["langchain_core.agents"] = mock_agents

from prysmai.integrations.langgraph import PrysmGraphMonitor, _safe_serialize


VALID_KEY = "sk-prysm-test1234567890abcdef"
VALID_URL = "https://prysmai.io/api/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("PRYSM_API_KEY", raising=False)
    monkeypatch.delenv("PRYSM_BASE_URL", raising=False)


@pytest.fixture
def monitor():
    """Create a PrysmGraphMonitor with mocked HTTP client."""
    with patch("prysmai.integrations.langgraph.httpx.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        m = PrysmGraphMonitor(api_key=VALID_KEY, batch_size=100)
        m._mock_http = mock_client
        yield m


# ─── Init Tests ───


class TestPrysmGraphMonitorInit:
    def test_init_with_key(self):
        with patch("prysmai.integrations.langgraph.httpx.Client"):
            m = PrysmGraphMonitor(api_key=VALID_KEY)
        assert m.api_key == VALID_KEY
        assert m.base_url == VALID_URL

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("PRYSM_API_KEY", VALID_KEY)
        with patch("prysmai.integrations.langgraph.httpx.Client"):
            m = PrysmGraphMonitor()
        assert m.api_key == VALID_KEY

    def test_init_no_key_raises(self):
        with pytest.raises(ValueError, match="API key is required"):
            PrysmGraphMonitor()

    def test_session_id_auto_generated(self):
        with patch("prysmai.integrations.langgraph.httpx.Client"):
            m = PrysmGraphMonitor(api_key=VALID_KEY)
        assert m.session_id is not None
        assert len(m.session_id) > 0

    def test_custom_session_id(self):
        with patch("prysmai.integrations.langgraph.httpx.Client"):
            m = PrysmGraphMonitor(api_key=VALID_KEY, session_id="custom-123")
        assert m.session_id == "custom-123"


# ─── Node Tracking Tests ───


class TestNodeTracking:
    def test_on_chain_start_tracks_node(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={"id": ["langchain", "RunnableSequence"]},
            inputs={"query": "hello"},
            run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        assert "agent" in monitor.node_execution_order
        assert monitor._current_node == "agent"

    def test_state_transitions_tracked(self, monitor):
        # Execute node A
        run_a = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_a,
            metadata={"langgraph_node": "node_a"},
        )
        monitor.on_chain_end(outputs={}, run_id=run_a)

        # Execute node B
        run_b = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_b,
            metadata={"langgraph_node": "node_b"},
        )
        monitor.on_chain_end(outputs={}, run_id=run_b)

        transitions = monitor.state_transitions
        assert len(transitions) == 1
        assert transitions[0]["from_node"] == "node_a"
        assert transitions[0]["to_node"] == "node_b"

    def test_node_timings_recorded(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={"langgraph_node": "timed_node"},
        )
        time.sleep(0.01)  # Small delay to ensure non-zero duration
        monitor.on_chain_end(outputs={}, run_id=run_id)

        timings = monitor.node_timings
        assert "timed_node" in timings
        assert timings["timed_node"]["duration_ms"] > 0

    def test_graph_summary(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        monitor.on_chain_end(outputs={}, run_id=run_id)

        summary = monitor.graph_summary
        assert summary["nodes_executed"] == 1
        assert "agent" in summary["execution_order"]

    def test_reset_clears_tracking(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        monitor.on_chain_end(outputs={}, run_id=run_id)

        monitor.reset()
        assert monitor.node_execution_order == []
        assert monitor.node_timings == {}
        assert monitor._current_node is None


# ─── LLM Callback Tests ───


class TestLLMCallbacks:
    def test_on_llm_start_and_end(self, monitor):
        run_id = uuid.uuid4()
        monitor._current_node = "agent"

        monitor.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}, "id": ["langchain", "ChatOpenAI"]},
            prompts=["What is 2+2?"],
            run_id=run_id,
        )

        # Create mock LLM result
        mock_gen = MockGeneration(text="4")
        mock_result = MockLLMResult(
            generations=[[mock_gen]],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        monitor.on_llm_end(response=mock_result, run_id=run_id)

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event["event_type"] == "llm_call"
        assert event["model"] == "gpt-4o"
        assert event["prompt_tokens"] == 10
        assert event["langgraph_node"] == "agent"

    def test_on_chat_model_start(self, monitor):
        run_id = uuid.uuid4()
        msg = MockBaseMessage(content="Hello", msg_type="human")

        monitor.on_chat_model_start(
            serialized={"kwargs": {"model_name": "claude-4"}, "id": ["langchain", "ChatAnthropic"]},
            messages=[[msg]],
            run_id=run_id,
        )

        assert str(run_id) in monitor._run_map
        assert monitor._run_map[str(run_id)]["model"] == "claude-4"

    def test_on_llm_error(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["test"],
            run_id=run_id,
        )
        monitor.on_llm_error(
            error=RuntimeError("Rate limit exceeded"),
            run_id=run_id,
        )

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event["event_type"] == "llm_error"
        assert "Rate limit" in event["error"]


# ─── Tool Callback Tests ───


class TestToolCallbacks:
    def test_on_tool_start_and_end(self, monitor):
        monitor._current_node = "tools"
        run_id = uuid.uuid4()

        monitor.on_tool_start(
            serialized={"name": "web_search"},
            input_str="latest AI news",
            run_id=run_id,
        )

        assert "web_search" in monitor._tool_names_seen
        assert "web_search" in monitor._node_to_tools.get("tools", [])

        monitor.on_tool_end(output="Found 10 results", run_id=run_id)

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event["event_type"] == "tool_invocation"
        assert event["tool_name"] == "web_search"
        assert event["langgraph_node"] == "tools"

    def test_on_tool_error(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_tool_start(
            serialized={"name": "database_query"},
            input_str="SELECT * FROM users",
            run_id=run_id,
        )
        monitor.on_tool_error(
            error=ConnectionError("Database unavailable"),
            run_id=run_id,
        )

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event["event_type"] == "tool_error"
        assert "Database unavailable" in event["error"]

    def test_tool_names_tracked_per_node(self, monitor):
        monitor._current_node = "node_a"
        run_a = uuid.uuid4()
        monitor.on_tool_start(serialized={"name": "tool_1"}, input_str="", run_id=run_a)
        monitor.on_tool_end(output="", run_id=run_a)

        monitor._current_node = "node_b"
        run_b = uuid.uuid4()
        monitor.on_tool_start(serialized={"name": "tool_2"}, input_str="", run_id=run_b)
        monitor.on_tool_end(output="", run_id=run_b)

        assert monitor._node_to_tools["node_a"] == ["tool_1"]
        assert monitor._node_to_tools["node_b"] == ["tool_2"]


# ─── Agent Callback Tests ───


class TestAgentCallbacks:
    def test_on_agent_action(self, monitor):
        run_id = uuid.uuid4()
        monitor._current_node = "agent"
        action = MockAgentAction(
            tool="calculator", tool_input="2+2", log="I need to calculate 2+2"
        )
        monitor.on_agent_action(action=action, run_id=run_id)

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event["event_type"] == "agent_action"
        assert event["tool"] == "calculator"

    def test_on_agent_finish(self, monitor):
        run_id = uuid.uuid4()
        monitor._current_node = "agent"
        finish = MockAgentFinish(
            return_values={"output": "The answer is 4"}, log="Final answer"
        )
        monitor.on_agent_finish(finish=finish, run_id=run_id)

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event["event_type"] == "agent_finish"


# ─── Chain Error Tests ───


class TestChainErrors:
    def test_on_chain_error_with_node(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={"langgraph_node": "failing_node"},
        )
        monitor.on_chain_error(
            error=RuntimeError("Node failed"), run_id=run_id,
        )

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event["event_type"] == "node_error"
        assert event["node_name"] == "failing_node"

    def test_on_chain_error_without_node(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={},
        )
        monitor.on_chain_error(
            error=RuntimeError("Chain failed"), run_id=run_id,
        )

        event = monitor._events[0]
        assert event["event_type"] == "chain_error"


# ─── Flush Tests ───


class TestFlush:
    def test_flush_sends_events(self, monitor):
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        monitor.on_chain_end(outputs={"result": "done"}, run_id=run_id)

        assert len(monitor._events) == 1
        monitor.flush()
        assert len(monitor._events) == 0

        # Verify HTTP call was made
        monitor._mock_http.post.assert_called_once()
        call_args = monitor._mock_http.post.call_args
        assert "telemetry/events" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["source"] == "langgraph"
        assert len(payload["events"]) == 1

    def test_auto_flush_on_batch_size(self):
        with patch("prysmai.integrations.langgraph.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            m = PrysmGraphMonitor(api_key=VALID_KEY, batch_size=2)

            # First event — no flush
            run1 = uuid.uuid4()
            m.on_chain_start(serialized={}, inputs={}, run_id=run1, metadata={"langgraph_node": "a"})
            m.on_chain_end(outputs={}, run_id=run1)
            assert mock_client.post.call_count == 0

            # Second event — triggers flush
            run2 = uuid.uuid4()
            m.on_chain_start(serialized={}, inputs={}, run_id=run2, metadata={"langgraph_node": "b"})
            m.on_chain_end(outputs={}, run_id=run2)
            assert mock_client.post.call_count == 1

    def test_flush_retains_events_on_failure(self, monitor):
        monitor._mock_http.post.side_effect = Exception("Network error")
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        monitor.on_chain_end(outputs={}, run_id=run_id)

        event_count = len(monitor._events)
        monitor.flush()
        # Events should be re-added to buffer on failure
        assert len(monitor._events) == event_count


# ─── Governance Integration Tests ───


class TestGovernanceIntegration:
    @patch("prysmai.governance.GovernanceSession")
    def test_start_governance(self, mock_gov_cls, monitor):
        monitor._governance_enabled = True
        mock_session = MagicMock()
        mock_session.session_id = "gov-123"
        mock_session.is_active = True
        mock_gov_cls.return_value = mock_session

        monitor.start_governance(
            task="Test graph execution",
            available_tools=["search", "calculator"],
        )

        mock_gov_cls.assert_called_once()
        mock_session.start.assert_called_once()
        assert monitor._gov_session is mock_session

    def test_end_governance_returns_report(self, monitor):
        monitor._governance_enabled = True
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_report = MagicMock()
        mock_report.behavior_score = 90
        mock_session.end.return_value = mock_report
        monitor._gov_session = mock_session

        report = monitor.end_governance(outcome="completed")

        assert report is mock_report
        mock_session.end.assert_called_once()
        mock_session.close.assert_called_once()

    @patch("prysmai.governance.GovernanceSession")
    def test_auto_start_on_first_event(self, mock_gov_cls, monitor):
        monitor._governance_enabled = True
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.session_id = "auto-gov"
        mock_gov_cls.return_value = mock_session

        # First chain event should auto-start governance
        run_id = uuid.uuid4()
        monitor.on_chain_start(
            serialized={}, inputs={}, run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )

        mock_gov_cls.assert_called_once()
        mock_session.start.assert_called_once()

    def test_governance_events_buffered_and_checked(self, monitor):
        monitor._governance_enabled = True
        monitor._gov_check_interval = 2

        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.session_id = "buf-gov"
        mock_check_result = MagicMock()
        mock_check_result.has_flags = False
        mock_session.check_behavior.return_value = mock_check_result
        monitor._gov_session = mock_session

        # Report 2 governance events (should trigger check)
        monitor._report_governance_event("llm_call", {"model": "gpt-4"})
        assert mock_session.check_behavior.call_count == 0

        monitor._report_governance_event("tool_call", {"tool_name": "search"})
        assert mock_session.check_behavior.call_count == 1


# ─── Safe Serialize Tests ───


class TestSafeSerialize:
    def test_primitives(self):
        assert _safe_serialize(None) is None
        assert _safe_serialize(42) == 42
        assert _safe_serialize(3.14) == 3.14
        assert _safe_serialize(True) is True
        assert _safe_serialize("hello") == "hello"

    def test_long_string_truncated(self):
        long_str = "a" * 3000
        result = _safe_serialize(long_str)
        assert len(result) < 3000
        assert result.endswith("...[truncated]")

    def test_dict_serialized(self):
        d = {"key": "value", "num": 42}
        assert _safe_serialize(d) == {"key": "value", "num": 42}

    def test_list_serialized(self):
        lst = [1, 2, 3]
        assert _safe_serialize(lst) == [1, 2, 3]

    def test_unserializable_object(self):
        class Weird:
            def __str__(self):
                raise RuntimeError("nope")
        result = _safe_serialize(Weird())
        assert result == "<unserializable>"
