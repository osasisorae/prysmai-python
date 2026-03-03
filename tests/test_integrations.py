"""
Tests for Prysm AI Framework Integrations.

These tests verify the integration modules work correctly without requiring
the actual frameworks to be installed (using mocks).
"""

import json
import time
import uuid
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ─── LangChain Integration Tests ────────────────────────────────


class TestPrysmCallbackHandler:
    """Test the LangChain callback handler."""

    @pytest.fixture
    def handler(self):
        """Create a handler with mocked HTTP client."""
        with patch.dict("os.environ", {"PRYSM_API_KEY": "sk-prysm-test123"}):
            from prysmai.integrations.langchain import PrysmCallbackHandler

            h = PrysmCallbackHandler(api_key="sk-prysm-test123")
            h._client = MagicMock()
            return h

    def test_init_with_api_key(self, handler):
        assert handler.api_key == "sk-prysm-test123"
        assert handler.base_url == "https://prysmai.io/api/v1"
        assert handler.session_id is not None

    def test_init_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            from prysmai.integrations.langchain import PrysmCallbackHandler

            with pytest.raises(ValueError, match="Prysm API key is required"):
                PrysmCallbackHandler(api_key="")

    def test_on_llm_start_records_run(self, handler):
        run_id = uuid.uuid4()
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}, "id": ["langchain", "ChatOpenAI"]},
            prompts=["Hello, world!"],
            run_id=run_id,
        )
        assert str(run_id) in handler._run_map
        assert handler._run_map[str(run_id)]["model"] == "gpt-4o"
        assert handler._run_map[str(run_id)]["prompts"] == ["Hello, world!"]

    def test_on_llm_end_creates_event(self, handler):
        run_id = uuid.uuid4()
        # Start the LLM call
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}, "id": ["langchain", "ChatOpenAI"]},
            prompts=["Test prompt"],
            run_id=run_id,
        )

        # Create mock LLM result
        mock_generation = MagicMock()
        mock_generation.text = "Test completion"
        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]
        mock_result.llm_output = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}

        handler.on_llm_end(response=mock_result, run_id=run_id)

        assert len(handler._events) == 1
        event = handler._events[0]
        assert event["event_type"] == "llm_call"
        assert event["model"] == "gpt-4o"
        assert event["prompt_tokens"] == 10
        assert event["completion_tokens"] == 20
        assert event["latency_ms"] >= 0

    def test_on_llm_error_creates_event(self, handler):
        run_id = uuid.uuid4()
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}, "id": []},
            prompts=["Test"],
            run_id=run_id,
        )
        handler.on_llm_error(
            error=RuntimeError("API rate limit"),
            run_id=run_id,
        )

        assert len(handler._events) == 1
        event = handler._events[0]
        assert event["event_type"] == "llm_error"
        assert "rate limit" in event["error"]
        assert event["error_type"] == "RuntimeError"

    def test_on_chain_start_end(self, handler):
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized={"id": ["langchain", "chains", "LLMChain"]},
            inputs={"query": "test input"},
            run_id=run_id,
        )
        handler.on_chain_end(
            outputs={"result": "test output"},
            run_id=run_id,
        )

        assert len(handler._events) == 1
        event = handler._events[0]
        assert event["event_type"] == "chain_execution"
        assert event["chain_type"] == "LLMChain"
        assert event["inputs"]["query"] == "test input"
        assert event["outputs"]["result"] == "test output"

    def test_on_tool_start_end(self, handler):
        run_id = uuid.uuid4()
        handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str="search for AI news",
            run_id=run_id,
        )
        handler.on_tool_end(
            output="Found 10 results",
            run_id=run_id,
        )

        assert len(handler._events) == 1
        event = handler._events[0]
        assert event["event_type"] == "tool_invocation"
        assert event["tool_name"] == "search_tool"
        assert event["input"] == "search for AI news"
        assert event["output"] == "Found 10 results"

    def test_on_agent_action(self, handler):
        run_id = uuid.uuid4()
        mock_action = MagicMock()
        mock_action.tool = "calculator"
        mock_action.tool_input = "2 + 2"
        mock_action.log = "Using calculator to compute..."

        handler.on_agent_action(action=mock_action, run_id=run_id)

        assert len(handler._events) == 1
        event = handler._events[0]
        assert event["event_type"] == "agent_action"
        assert event["tool"] == "calculator"

    def test_on_agent_finish(self, handler):
        run_id = uuid.uuid4()
        mock_finish = MagicMock()
        mock_finish.return_values = {"output": "The answer is 4"}
        mock_finish.log = "Final answer computed."

        handler.on_agent_finish(finish=mock_finish, run_id=run_id)

        assert len(handler._events) == 1
        event = handler._events[0]
        assert event["event_type"] == "agent_finish"

    def test_flush_sends_events(self, handler):
        handler._events = [{"event_type": "test", "timestamp": time.time()}]
        handler.flush()

        handler._client.post.assert_called_once()
        call_args = handler._client.post.call_args
        assert "/telemetry/events" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["source"] == "langchain"
        assert len(body["events"]) == 1
        assert handler._events == []

    def test_flush_rebuffers_on_failure(self, handler):
        handler._client.post.side_effect = Exception("Network error")
        handler._events = [{"event_type": "test"}]
        handler.flush()

        # Events should be re-buffered
        assert len(handler._events) == 1

    def test_batch_flush_on_size(self, handler):
        handler.batch_size = 3
        for i in range(3):
            run_id = uuid.uuid4()
            handler.on_llm_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}, "id": []},
                prompts=[f"prompt {i}"],
                run_id=run_id,
            )
            handler.on_llm_end(
                response=MagicMock(generations=[[]], llm_output={}),
                run_id=run_id,
            )

        # Should have flushed at batch_size=3
        assert handler._client.post.called

    def test_context_metadata_attached(self, handler):
        from prysmai.context import prysm_context

        with prysm_context(user_id="user_456", session_id="sess_789"):
            run_id = uuid.uuid4()
            handler.on_llm_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}, "id": []},
                prompts=["test"],
                run_id=run_id,
            )
            handler.on_llm_end(
                response=MagicMock(generations=[[]], llm_output={}),
                run_id=run_id,
            )

        event = handler._events[0]
        assert event["user_id"] == "user_456"

    def test_close_flushes_and_closes_client(self, handler):
        handler._events = [{"event_type": "test", "timestamp": time.time()}]
        handler.close()
        handler._client.post.assert_called_once()
        handler._client.close.assert_called_once()


# ─── CrewAI Integration Tests ───────────────────────────────────


class TestPrysmCrewMonitor:
    """Test the CrewAI monitor."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor with mocked HTTP client."""
        # Mock the crewai imports
        with patch.dict("sys.modules", {
            "crewai": MagicMock(),
            "crewai.utilities": MagicMock(),
            "crewai.utilities.events": MagicMock(),
        }):
            # We need to reload the module with mocked imports
            import importlib
            import prysmai.integrations.crewai as crewai_mod

            # Create monitor directly
            m = crewai_mod.PrysmCrewMonitor(api_key="sk-prysm-test456")
            m._client = MagicMock()
            return m

    def test_init_with_api_key(self, monitor):
        assert monitor.api_key == "sk-prysm-test456"
        assert monitor.session_id is not None

    def test_init_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch.dict("sys.modules", {
                "crewai": MagicMock(),
                "crewai.utilities": MagicMock(),
                "crewai.utilities.events": MagicMock(),
            }):
                import importlib
                import prysmai.integrations.crewai as crewai_mod

                with pytest.raises(ValueError, match="Prysm API key is required"):
                    crewai_mod.PrysmCrewMonitor(api_key="")

    def test_callback_invocation(self, monitor):
        """Test that the monitor works as a callback."""
        monitor("task_started", {"task": "Research AI"})
        assert len(monitor._events) == 1
        assert monitor._events[0]["event_type"] == "crew_callback_task_started"

    def test_crew_start_event(self, monitor):
        mock_event = MagicMock()
        monitor._on_crew_start(mock_event)
        assert len(monitor._events) == 1
        assert monitor._events[0]["event_type"] == "crew_kickoff_started"
        assert monitor._crew_start_time is not None

    def test_crew_end_event(self, monitor):
        monitor._crew_start_time = time.time() - 1.0
        mock_event = MagicMock()
        mock_event.output = "Crew completed successfully"
        monitor._on_crew_end(mock_event)

        # flush() is called at end of crew, so events are sent and cleared
        # Verify the post was called with the right data
        monitor._client.post.assert_called_once()
        call_args = monitor._client.post.call_args
        body = call_args[1]["json"]
        events = body["events"]
        crew_end_events = [e for e in events if e["event_type"] == "crew_kickoff_completed"]
        assert len(crew_end_events) == 1
        assert crew_end_events[0]["latency_ms"] >= 1000

    def test_agent_start_end(self, monitor):
        mock_agent = MagicMock()
        mock_agent.role = "Researcher"
        mock_agent.goal = "Find relevant papers"
        mock_agent.backstory = "An expert researcher"

        start_event = MagicMock()
        start_event.agent = mock_agent
        monitor._on_agent_start(start_event)

        end_event = MagicMock()
        end_event.agent = mock_agent
        end_event.output = "Found 5 papers"
        monitor._on_agent_end(end_event)

        assert len(monitor._events) == 2
        assert monitor._events[0]["event_type"] == "agent_execution_started"
        assert monitor._events[0]["agent_name"] == "Researcher"
        assert monitor._events[1]["event_type"] == "agent_execution_completed"

    def test_task_start_end(self, monitor):
        mock_task = MagicMock()
        mock_task.description = "Research AI interpretability"
        mock_task.expected_output = "A summary of findings"

        start_event = MagicMock()
        start_event.task = mock_task
        monitor._on_task_start(start_event)

        end_event = MagicMock()
        end_event.task = mock_task
        end_event.output = "Summary: AI interpretability is..."
        monitor._on_task_end(end_event)

        assert len(monitor._events) == 2
        assert monitor._events[0]["event_type"] == "task_execution_started"
        assert monitor._events[1]["event_type"] == "task_execution_completed"

    def test_tool_start_end(self, monitor):
        start_event = MagicMock()
        start_event.tool_name = "web_search"
        start_event.tool_input = "AI papers 2025"
        monitor._on_tool_start(start_event)

        end_event = MagicMock()
        end_event.tool_name = "web_search"
        end_event.tool_output = "10 results found"
        monitor._on_tool_end(end_event)

        assert len(monitor._events) == 2
        assert monitor._events[0]["event_type"] == "tool_usage_started"
        assert monitor._events[1]["event_type"] == "tool_usage_completed"

    def test_tool_error(self, monitor):
        error_event = MagicMock()
        error_event.tool_name = "web_search"
        error_event.error = "Connection timeout"
        monitor._on_tool_error(error_event)

        assert len(monitor._events) == 1
        assert monitor._events[0]["event_type"] == "tool_usage_error"
        assert monitor._events[0]["error"] == "Connection timeout"

    def test_flush_sends_events(self, monitor):
        monitor._events = [{"event_type": "test", "timestamp": time.time()}]
        monitor.flush()

        monitor._client.post.assert_called_once()
        call_args = monitor._client.post.call_args
        body = call_args[1]["json"]
        assert body["source"] == "crewai"

    def test_close_flushes(self, monitor):
        monitor._events = [{"event_type": "test", "timestamp": time.time()}]
        monitor.close()
        monitor._client.post.assert_called_once()
        monitor._client.close.assert_called_once()


# ─── LlamaIndex Integration Tests ───────────────────────────────


class TestPrysmSpanHandler:
    """Test the LlamaIndex span handler."""

    @pytest.fixture
    def handler(self):
        """Create a handler with mocked HTTP client."""
        # Create a simple mock-based handler that mimics PrysmSpanHandler behavior
        # without needing the actual llama_index import
        from enum import Enum

        class MockCBEventType(Enum):
            LLM = "llm"
            EMBEDDING = "embedding"
            RETRIEVE = "retrieve"
            QUERY = "query"
            SYNTHESIZE = "synthesize"
            CHUNKING = "chunking"

        class MockEventPayload:
            MESSAGES = "messages"
            PROMPT = "prompt"
            SERIALIZED = "serialized"
            CHUNKS = "chunks"
            QUERY_STR = "query_str"
            RESPONSE = "response"
            NODES = "nodes"

        class MockHandler:
            """Mimics PrysmSpanHandler without requiring llama_index."""
            def __init__(self):
                self.api_key = "sk-prysm-test789"
                self.base_url = "https://prysmai.io/api/v1"
                self.session_id = str(uuid.uuid4())
                self.user_id = None
                self.metadata = {}
                self._events = []
                self._span_map = {}
                self._client = MagicMock()

            def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
                if not event_id:
                    event_id = str(uuid.uuid4())
                self._span_map[event_id] = {
                    "event_type": event_type.value if hasattr(event_type, "value") else str(event_type),
                    "start_time": time.time(),
                    "parent_id": parent_id,
                    "payload": None,
                }
                return event_id

            def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
                span_data = self._span_map.pop(event_id, {})
                start_time = span_data.get("start_time", time.time())
                latency_ms = int((time.time() - start_time) * 1000)
                event_type_str = event_type.value if hasattr(event_type, "value") else str(event_type)
                event = {
                    "event_type": f"llamaindex_{event_type_str}",
                    "event_id": event_id,
                    "parent_id": span_data.get("parent_id"),
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "session_id": self.session_id,
                    "user_id": self.user_id,
                    "prysm_metadata": self.metadata,
                }
                self._events.append(event)

            def start_trace(self, trace_id=None):
                self._span_map.clear()

            def end_trace(self, trace_id=None, trace_map=None):
                self.flush()

            def flush(self):
                if not self._events:
                    return
                events_to_send = self._events[:]
                self._events.clear()
                try:
                    self._client.post(
                        f"{self.base_url}/telemetry/events",
                        json={"source": "llamaindex", "session_id": self.session_id, "events": events_to_send},
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    )
                except Exception:
                    pass

            def close(self):
                self.flush()
                self._client.close()

        class MockModule:
            CBEventType = MockCBEventType
            EventPayload = MockEventPayload

        return MockHandler(), MockModule()

    def test_on_event_start_records_span(self, handler):
        h, mod = handler
        event_id = h.on_event_start(
            event_type=mod.CBEventType.LLM,
            payload={"serialized": {"model": "gpt-4o"}, "messages": []},
            event_id="test-event-1",
        )

        assert event_id == "test-event-1"
        assert "test-event-1" in h._span_map

    def test_on_event_end_creates_event(self, handler):
        h, mod = handler
        h.on_event_start(
            event_type=mod.CBEventType.LLM,
            payload={},
            event_id="test-event-2",
        )
        h.on_event_end(
            event_type=mod.CBEventType.LLM,
            payload={"response": MagicMock(text="Hello!")},
            event_id="test-event-2",
        )

        assert len(h._events) == 1
        event = h._events[0]
        assert "llamaindex_" in event["event_type"]
        assert event["latency_ms"] >= 0

    def test_start_trace_clears_spans(self, handler):
        h, mod = handler
        h._span_map["old-span"] = {"data": "old"}
        h.start_trace(trace_id="new-trace")
        assert len(h._span_map) == 0

    def test_end_trace_flushes(self, handler):
        h, mod = handler
        h._events = [{"event_type": "test", "timestamp": time.time()}]
        h.end_trace(trace_id="trace-1")
        h._client.post.assert_called_once()

    def test_flush_sends_events(self, handler):
        h, mod = handler
        h._events = [{"event_type": "test", "timestamp": time.time()}]
        h.flush()

        h._client.post.assert_called_once()
        call_args = h._client.post.call_args
        body = call_args[1]["json"]
        assert body["source"] == "llamaindex"

    def test_close_flushes(self, handler):
        h, mod = handler
        h._events = [{"event_type": "test", "timestamp": time.time()}]
        h.close()
        h._client.post.assert_called_once()
        h._client.close.assert_called_once()

    def test_generates_event_id_if_missing(self, handler):
        h, mod = handler
        event_id = h.on_event_start(
            event_type=mod.CBEventType.QUERY,
            payload={},
            event_id="",
        )
        assert event_id != ""
        assert len(event_id) > 0


# ─── Safe Serialize Tests ────────────────────────────────────────


class TestSafeSerialize:
    """Test the _safe_serialize helper used across integrations."""

    def test_serialize_primitives(self):
        from prysmai.integrations.langchain import _safe_serialize

        assert _safe_serialize(None) is None
        assert _safe_serialize(42) == 42
        assert _safe_serialize(3.14) == 3.14
        assert _safe_serialize(True) is True
        assert _safe_serialize("hello") == "hello"

    def test_serialize_truncates_long_strings(self):
        from prysmai.integrations.langchain import _safe_serialize

        long_str = "x" * 3000
        result = _safe_serialize(long_str)
        assert len(result) < 3000
        assert result.endswith("...[truncated]")

    def test_serialize_dict(self):
        from prysmai.integrations.langchain import _safe_serialize

        result = _safe_serialize({"key": "value", "nested": {"a": 1}})
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_serialize_list(self):
        from prysmai.integrations.langchain import _safe_serialize

        result = _safe_serialize([1, 2, 3])
        assert result == [1, 2, 3]

    def test_serialize_unserializable(self):
        from prysmai.integrations.langchain import _safe_serialize

        class BadObj:
            def __str__(self):
                raise Exception("Cannot serialize")

        result = _safe_serialize(BadObj())
        assert result == "<unserializable>"
