"""
Tests for Prysm AI — Microsoft Agent Framework Integration

Tests the PrysmAgentFrameworkMonitor middleware system using mock
Agent Framework context objects (no real agent-framework dependency needed).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ─── Mock Agent Framework types (so tests don't require the real package) ────


class MockAgentMiddleware:
    """Mock base class for agent middleware."""
    pass


class MockFunctionMiddleware:
    """Mock base class for function middleware."""
    pass


class MockChatMiddleware:
    """Mock base class for chat middleware."""
    pass


@dataclass
class MockAgent:
    name: str = "TestAgent"


@dataclass
class MockAgentContext:
    agent: MockAgent = field(default_factory=MockAgent)
    messages: List[Dict[str, str]] = field(default_factory=list)
    is_streaming: bool = False
    terminate: bool = False
    result: Optional[str] = None


@dataclass
class MockFunction:
    name: str = "search_web"


@dataclass
class MockFunctionInvocationContext:
    function: MockFunction = field(default_factory=MockFunction)
    arguments: Dict[str, Any] = field(default_factory=dict)
    terminate: bool = False
    result: Optional[str] = None


@dataclass
class MockUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MockMessage:
    content: str = "Hello, world!"
    role: str = "assistant"


@dataclass
class MockChoice:
    message: MockMessage = field(default_factory=MockMessage)


@dataclass
class MockChatResult:
    model: str = "gpt-4o"
    usage: MockUsage = field(default_factory=MockUsage)
    choices: List[MockChoice] = field(default_factory=lambda: [MockChoice()])


@dataclass
class MockChatContext:
    messages: List[Dict[str, str]] = field(default_factory=list)
    is_streaming: bool = False
    options: Dict[str, Any] = field(default_factory=dict)
    terminate: bool = False
    result: Optional[MockChatResult] = None


# ─── Patch the import system so agent_framework module loads with mocks ──────

import sys

mock_af_module = MagicMock()
mock_af_module.AgentMiddleware = MockAgentMiddleware
mock_af_module.FunctionMiddleware = MockFunctionMiddleware
mock_af_module.ChatMiddleware = MockChatMiddleware
mock_af_module.AgentContext = MockAgentContext
mock_af_module.FunctionInvocationContext = MockFunctionInvocationContext
mock_af_module.ChatContext = MockChatContext

sys.modules["agent_framework"] = mock_af_module

# Now import the integration (it will find agent_framework in sys.modules)
from prysmai.integrations.agent_framework import (
    PrysmAgentFrameworkMonitor,
    _PrysmAgentMiddleware,
    _PrysmFunctionMiddleware,
    _PrysmChatMiddleware,
    _safe_serialize,
    _extract_message_content,
)


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def monitor():
    """Create a monitor with a large batch_size so events stay buffered for inspection."""
    m = PrysmAgentFrameworkMonitor(
        api_key="sk-prysm-test-key-1234567890",
        base_url="https://test.prysmai.io/api/v1",
        session_id="test-session-001",
        user_id="test-user",
        batch_size=100,  # large so events don't auto-flush
        flush_interval=9999,
    )
    return m


# ─── Helper to run async middleware ──────────────────────────────


def run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Unit Tests ──────────────────────────────────────────────────


class TestSafeSerialize:
    def test_none(self):
        assert _safe_serialize(None) is None

    def test_primitives(self):
        assert _safe_serialize(42) == 42
        assert _safe_serialize(3.14) == 3.14
        assert _safe_serialize(True) is True
        assert _safe_serialize("hello") == "hello"

    def test_long_string_truncation(self):
        long_str = "x" * 3000
        result = _safe_serialize(long_str)
        assert result.endswith("...[truncated]")
        assert len(result) < 3000

    def test_dict(self):
        result = _safe_serialize({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

    def test_list(self):
        result = _safe_serialize([1, 2, 3])
        assert result == [1, 2, 3]

    def test_nested(self):
        result = _safe_serialize({"a": [1, {"b": "c"}]})
        assert result == {"a": [1, {"b": "c"}]}

    def test_unserializable(self):
        class Weird:
            def __str__(self):
                raise RuntimeError("nope")
        result = _safe_serialize(Weird())
        assert result == "<unserializable>"


class TestExtractMessageContent:
    def test_empty(self):
        assert _extract_message_content(None) == []
        assert _extract_message_content([]) == []

    def test_dict_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = _extract_message_content(msgs)
        assert result == [{"role": "user", "content": "hi"}]

    def test_object_messages(self):
        msg = MockMessage(content="hello", role="assistant")
        result = _extract_message_content([msg])
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "hello"


class TestMonitorInit:
    def test_basic_init(self, monitor):
        assert monitor.api_key == "sk-prysm-test-key-1234567890"
        assert monitor.session_id == "test-session-001"
        assert monitor.user_id == "test-user"
        assert monitor._events == []
        assert monitor._agent_runs == []
        assert monitor._tool_calls == []
        assert monitor._llm_calls == []

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                PrysmAgentFrameworkMonitor(api_key="")

    def test_middleware_returns_three(self, monitor):
        mw = monitor.middleware()
        assert len(mw) == 3
        assert isinstance(mw[0], _PrysmAgentMiddleware)
        assert isinstance(mw[1], _PrysmFunctionMiddleware)
        assert isinstance(mw[2], _PrysmChatMiddleware)


class TestAgentMiddleware:
    def test_successful_agent_run(self, monitor):
        ctx = MockAgentContext(
            agent=MockAgent(name="TestBot"),
            messages=[{"role": "user", "content": "Hello"}],
        )

        async def mock_next(c):
            c.result = "Done"

        run_async(monitor._agent_mw.process(ctx, mock_next))

        # Should have 2 events: agent_run_start + agent_run_end
        assert len(monitor._events) == 2
        assert monitor._events[0]["event_type"] == "agent_run_start"
        assert monitor._events[0]["agent_name"] == "TestBot"
        assert monitor._events[1]["event_type"] == "agent_run_end"
        assert monitor._events[1]["success"] is True
        assert monitor._events[1]["latency_ms"] >= 0

        # Should track the run
        assert len(monitor._agent_runs) == 1
        assert monitor._agent_runs[0]["agent_name"] == "TestBot"
        assert monitor._agent_runs[0]["success"] is True

    def test_failed_agent_run(self, monitor):
        ctx = MockAgentContext(agent=MockAgent(name="FailBot"))

        async def mock_next(c):
            raise RuntimeError("Agent crashed")

        with pytest.raises(RuntimeError, match="Agent crashed"):
            run_async(monitor._agent_mw.process(ctx, mock_next))

        assert len(monitor._events) == 2
        assert monitor._events[1]["event_type"] == "agent_run_end"
        assert monitor._events[1]["success"] is False
        assert monitor._events[1]["error"] == "Agent crashed"
        assert monitor._events[1]["error_type"] == "RuntimeError"

        assert monitor._agent_runs[0]["success"] is False


class TestFunctionMiddleware:
    def test_successful_tool_call(self, monitor):
        ctx = MockFunctionInvocationContext(
            function=MockFunction(name="search_web"),
            arguments={"query": "test query"},
        )

        async def mock_next(c):
            c.result = "Search results here"

        run_async(monitor._function_mw.process(ctx, mock_next))

        assert len(monitor._events) == 2
        assert monitor._events[0]["event_type"] == "tool_call_start"
        assert monitor._events[0]["tool_name"] == "search_web"
        assert monitor._events[0]["arguments"] == {"query": "test query"}
        assert monitor._events[1]["event_type"] == "tool_call_end"
        assert monitor._events[1]["success"] is True
        assert monitor._events[1]["latency_ms"] >= 0

        assert len(monitor._tool_calls) == 1
        assert monitor._tool_calls[0]["tool_name"] == "search_web"
        assert "search_web" in monitor._tool_names_seen

    def test_failed_tool_call(self, monitor):
        ctx = MockFunctionInvocationContext(
            function=MockFunction(name="db_query"),
            arguments={"sql": "SELECT *"},
        )

        async def mock_next(c):
            raise ConnectionError("Database offline")

        with pytest.raises(ConnectionError):
            run_async(monitor._function_mw.process(ctx, mock_next))

        assert monitor._events[1]["event_type"] == "tool_call_end"
        assert monitor._events[1]["success"] is False
        assert monitor._events[1]["error"] == "Database offline"

    def test_multiple_tools_tracked(self, monitor):
        for name in ["search", "calculate", "search"]:
            ctx = MockFunctionInvocationContext(
                function=MockFunction(name=name),
            )

            async def mock_next(c):
                pass

            run_async(monitor._function_mw.process(ctx, mock_next))

        assert len(monitor._tool_calls) == 3
        # Unique tool names
        assert monitor._tool_names_seen == ["search", "calculate"]


class TestChatMiddleware:
    def test_successful_llm_call(self, monitor):
        ctx = MockChatContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        async def mock_next(c):
            c.result = MockChatResult(
                model="gpt-4o",
                usage=MockUsage(prompt_tokens=100, completion_tokens=50),
                choices=[MockChoice(message=MockMessage(content="Hi there!"))],
            )

        run_async(monitor._chat_mw.process(ctx, mock_next))

        assert len(monitor._events) == 2
        assert monitor._events[0]["event_type"] == "llm_call_start"
        assert monitor._events[0]["message_count"] == 1
        assert monitor._events[1]["event_type"] == "llm_call_end"
        assert monitor._events[1]["success"] is True
        assert monitor._events[1]["model"] == "gpt-4o"
        assert monitor._events[1]["prompt_tokens"] == 100
        assert monitor._events[1]["completion_tokens"] == 50
        assert monitor._events[1]["completion_preview"] == "Hi there!"

        assert len(monitor._llm_calls) == 1
        assert monitor._llm_calls[0]["model"] == "gpt-4o"
        assert monitor._llm_calls[0]["total_tokens"] == 150

    def test_failed_llm_call(self, monitor):
        ctx = MockChatContext(messages=[])

        async def mock_next(c):
            raise TimeoutError("LLM timeout")

        with pytest.raises(TimeoutError):
            run_async(monitor._chat_mw.process(ctx, mock_next))

        assert monitor._events[1]["success"] is False
        assert monitor._events[1]["error_type"] == "TimeoutError"


class TestExecutionSummary:
    def test_empty_summary(self, monitor):
        summary = monitor.execution_summary
        assert summary["agent_runs"] == 0
        assert summary["tool_calls"] == 0
        assert summary["llm_calls"] == 0
        assert summary["tools_used"] == []
        assert summary["total_llm_tokens"] == 0

    def test_populated_summary(self, monitor):
        # Simulate some execution
        monitor._agent_runs.append({"run_id": "r1", "success": True, "latency_ms": 500})
        monitor._tool_calls.append({"tool_name": "search", "success": True, "latency_ms": 100})
        monitor._tool_calls.append({"tool_name": "calc", "success": False, "latency_ms": 200})
        monitor._llm_calls.append({"model": "gpt-4o", "total_tokens": 150, "latency_ms": 300})
        monitor._tool_names_seen = ["search", "calc"]

        summary = monitor.execution_summary
        assert summary["agent_runs"] == 1
        assert summary["tool_calls"] == 2
        assert summary["llm_calls"] == 1
        assert summary["tools_used"] == ["search", "calc"]
        assert summary["total_llm_tokens"] == 150
        assert summary["avg_tool_latency_ms"] == 150.0
        assert summary["tool_success_rate"] == 0.5


class TestFlush:
    def test_flush_sends_events(self, monitor):
        monitor._buffer_event({"event_type": "test_event"})
        assert len(monitor._events) == 1

        with patch.object(monitor._client, "post") as mock_post:
            monitor.flush()
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert body["source"] == "agent_framework"
            assert body["session_id"] == "test-session-001"
            assert len(body["events"]) == 1

        assert len(monitor._events) == 0

    def test_flush_empty_noop(self, monitor):
        with patch.object(monitor._client, "post") as mock_post:
            monitor.flush()
            mock_post.assert_not_called()

    def test_flush_failure_rebuffers(self, monitor):
        monitor._buffer_event({"event_type": "test_event"})

        with patch.object(monitor._client, "post", side_effect=Exception("Network error")):
            monitor.flush()

        # Events should be re-buffered
        assert len(monitor._events) == 1


class TestReset:
    def test_reset_clears_state(self, monitor):
        monitor._agent_runs.append({"test": True})
        monitor._tool_calls.append({"test": True})
        monitor._llm_calls.append({"test": True})
        monitor._tool_names_seen.append("tool1")

        monitor.reset()

        assert monitor._agent_runs == []
        assert monitor._tool_calls == []
        assert monitor._llm_calls == []
        assert monitor._tool_names_seen == []


class TestClose:
    def test_close_flushes_and_closes(self, monitor):
        monitor._buffer_event({"event_type": "test"})

        with patch.object(monitor._client, "post"):
            with patch.object(monitor._client, "close") as mock_close:
                monitor.close()
                mock_close.assert_called_once()


class TestEndToEndFlow:
    """Simulate a realistic agent execution flow."""

    def test_full_agent_execution(self, monitor):
        """Simulate: agent starts → LLM call → tool call → LLM call → agent ends."""
        agent_ctx = MockAgentContext(
            agent=MockAgent(name="ResearchBot"),
            messages=[{"role": "user", "content": "Find info about AI"}],
        )

        async def agent_execution(ctx):
            # Step 1: LLM decides to use a tool
            chat_ctx = MockChatContext(
                messages=[{"role": "user", "content": "Find info about AI"}],
            )

            async def llm_1(c):
                c.result = MockChatResult(
                    model="gpt-4o",
                    usage=MockUsage(prompt_tokens=50, completion_tokens=20),
                )

            await monitor._chat_mw.process(chat_ctx, llm_1)

            # Step 2: Execute the tool
            func_ctx = MockFunctionInvocationContext(
                function=MockFunction(name="web_search"),
                arguments={"query": "AI research 2026"},
            )

            async def tool_exec(c):
                c.result = "Found 10 results about AI research"

            await monitor._function_mw.process(func_ctx, tool_exec)

            # Step 3: LLM processes tool results
            chat_ctx2 = MockChatContext(
                messages=[
                    {"role": "user", "content": "Find info about AI"},
                    {"role": "assistant", "content": "Let me search..."},
                    {"role": "tool", "content": "Found 10 results"},
                ],
            )

            async def llm_2(c):
                c.result = MockChatResult(
                    model="gpt-4o",
                    usage=MockUsage(prompt_tokens=150, completion_tokens=100),
                    choices=[MockChoice(message=MockMessage(content="Here is what I found..."))],
                )

            await monitor._chat_mw.process(chat_ctx2, llm_2)
            ctx.result = "Research complete"

        run_async(monitor._agent_mw.process(agent_ctx, agent_execution))

        # Verify complete event sequence
        event_types = [e["event_type"] for e in monitor._events]
        assert event_types == [
            "agent_run_start",
            "llm_call_start",
            "llm_call_end",
            "tool_call_start",
            "tool_call_end",
            "llm_call_start",
            "llm_call_end",
            "agent_run_end",
        ]

        # Verify tracking
        assert len(monitor._agent_runs) == 1
        assert len(monitor._tool_calls) == 1
        assert len(monitor._llm_calls) == 2
        assert monitor._tool_names_seen == ["web_search"]

        # Verify summary
        summary = monitor.execution_summary
        assert summary["agent_runs"] == 1
        assert summary["tool_calls"] == 1
        assert summary["llm_calls"] == 2
        assert summary["total_llm_tokens"] == 320  # 70 + 250
