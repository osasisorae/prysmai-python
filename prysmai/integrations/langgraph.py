"""
Prysm AI — LangGraph Integration

Provides PrysmGraphMonitor that captures node executions, state transitions,
LLM calls, tool invocations, and routing decisions from LangGraph graphs
and routes telemetry through the Prysm proxy.

Usage:
    from prysmai.integrations.langgraph import PrysmGraphMonitor

    monitor = PrysmGraphMonitor(api_key="sk-prysm-...")
    for chunk in graph.stream(inputs, stream_mode="debug", config={"callbacks": [monitor]}):
        ...
    monitor.flush()

With governance (v0.5.0):
    monitor = PrysmGraphMonitor(api_key="sk-prysm-...", governance=True)
    monitor.start_governance(task="Analyze customer support ticket")
    for chunk in graph.stream(inputs, stream_mode="debug", config={"callbacks": [monitor]}):
        ...
    report = monitor.end_governance()

Blueprint Reference: Section 9.3 (LangGraph), v0.5.0
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult, ChatGeneration, Generation
    from langchain_core.messages import BaseMessage
    from langchain_core.agents import AgentAction, AgentFinish
except ImportError:
    raise ImportError(
        "LangGraph integration requires langchain-core. "
        "Install it with: pip install prysmai[langgraph]"
    )

from prysmai.context import prysm_context

logger = logging.getLogger("prysmai.integrations.langgraph")


class PrysmGraphMonitor(BaseCallbackHandler):
    """
    LangGraph callback handler that captures telemetry from graph execution
    and sends it to Prysm.

    Captures:
        - LLM calls (model, prompts, completions, token usage)
        - Tool invocations (tool name, input, output)
        - Node executions (via on_chain_start/end with graph metadata)
        - State transitions (node-to-node flow)
        - Agent routing decisions
        - Error events

    When governance=True, events are forwarded to a GovernanceSession
    for behavioral analysis, security scanning, and policy enforcement.

    Graph-specific features:
        - Tracks node execution order and timing
        - Detects conditional routing decisions
        - Captures subgraph execution context
        - Maps LLM/tool calls to their parent graph nodes
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        governance: bool = False,
        governance_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Prysm LangGraph monitor.

        Args:
            api_key: Prysm API key (sk-prysm-...). Falls back to PRYSM_API_KEY env var.
            base_url: Prysm proxy base URL. Falls back to PRYSM_BASE_URL.
            session_id: Optional session ID for grouping related graph executions.
            user_id: Optional user ID for attribution.
            metadata: Optional metadata dict attached to all events.
            batch_size: Number of events to buffer before sending (default 10).
            flush_interval: Max seconds between flushes (default 5.0).
            governance: Enable governance monitoring (v0.5.0+). When True,
                events are forwarded to a GovernanceSession for behavioral analysis.
            governance_context: Additional context for the governance session
                (e.g., {"environment": "production", "team": "backend"}).
        """
        import os

        self.api_key = api_key or os.environ.get("PRYSM_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("PRYSM_BASE_URL", "https://prysmai.io/api/v1")
        )
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.metadata = metadata or {}
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        if not self.api_key:
            raise ValueError(
                "Prysm API key is required. Pass api_key= or set PRYSM_API_KEY env var."
            )

        # Internal state
        self._events: List[Dict[str, Any]] = []
        self._run_map: Dict[str, Dict[str, Any]] = {}
        self._last_flush = time.time()
        self._client = httpx.Client(timeout=30.0)

        # Graph-specific tracking
        self._node_execution_order: List[str] = []
        self._node_timings: Dict[str, Dict[str, float]] = {}
        self._current_node: Optional[str] = None
        self._tool_names_seen: List[str] = []
        self._node_to_tools: Dict[str, List[str]] = {}
        self._state_transitions: List[Dict[str, Any]] = []

        # Governance state (v0.5.0)
        self._governance_enabled = governance
        self._governance_context = governance_context or {}
        self._gov_session: Any = None
        self._gov_event_buffer: List[Dict[str, Any]] = []
        self._gov_check_interval = 5
        self._governance_report: Any = None

    # ─── Properties ─────────────────────────────────────────────

    @property
    def governance_report(self) -> Any:
        """The governance report from the last session (if governance=True)."""
        return self._governance_report

    @property
    def node_execution_order(self) -> List[str]:
        """Ordered list of node names as they were executed."""
        return self._node_execution_order[:]

    @property
    def node_timings(self) -> Dict[str, Dict[str, float]]:
        """Timing data for each node: {node_name: {start, end, duration_ms}}."""
        return dict(self._node_timings)

    @property
    def state_transitions(self) -> List[Dict[str, Any]]:
        """List of state transitions: [{from_node, to_node, timestamp}]."""
        return self._state_transitions[:]

    @property
    def graph_summary(self) -> Dict[str, Any]:
        """Summary of the graph execution for telemetry/debugging."""
        return {
            "session_id": self.session_id,
            "nodes_executed": len(self._node_execution_order),
            "execution_order": self._node_execution_order,
            "tools_used": self._tool_names_seen,
            "node_to_tools": dict(self._node_to_tools),
            "transitions": len(self._state_transitions),
        }

    # ─── Governance Lifecycle ───────────────────────────────────

    def start_governance(
        self,
        task: str = "LangGraph graph execution",
        agent_type: str = "langgraph",
        available_tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start a governance session for this graph execution.

        Call this before invoking your graph. If governance=True was set
        in the constructor, this is called automatically on the first
        node execution event. Calling it explicitly gives you control
        over the task description and available tools.

        Args:
            task: Description of the task being performed.
            agent_type: Agent type identifier (default "langgraph").
            available_tools: List of tool names available to the graph.
            context: Additional context for the governance session.
        """
        if self._gov_session and self._gov_session.is_active:
            logger.warning("Governance session already active. Ending previous session.")
            self.end_governance()

        try:
            from prysmai.governance import GovernanceSession

            merged_context = {
                "framework": "langgraph",
                "session_id": self.session_id,
                **self._governance_context,
                **(context or {}),
            }

            self._gov_session = GovernanceSession(
                prysm_key=self.api_key,
                base_url=self.base_url,
                task=task,
                agent_type=agent_type,
                available_tools=available_tools or self._tool_names_seen or None,
                context=merged_context,
            )
            self._gov_session.start()
            logger.info(
                f"Governance session started for LangGraph: {self._gov_session.session_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to start governance session: {e}")
            self._gov_session = None

    def end_governance(
        self,
        outcome: str = "completed",
        output_summary: Optional[str] = None,
    ) -> Any:
        """
        End the governance session and return the report.

        Args:
            outcome: How the task ended ("completed", "failed", "partial", "timeout").
            output_summary: Brief summary of what was produced.

        Returns:
            SessionReport if governance was active, None otherwise.
        """
        if not self._gov_session or not self._gov_session.is_active:
            return None

        try:
            # Flush remaining governance events
            if self._gov_event_buffer:
                try:
                    self._gov_session.check_behavior(self._gov_event_buffer)
                except Exception:
                    pass
                self._gov_event_buffer.clear()

            self._governance_report = self._gov_session.end(
                outcome=outcome,
                output_summary=output_summary or json.dumps(self.graph_summary),
            )
            logger.info(
                f"Governance session ended for LangGraph: "
                f"score={self._governance_report.behavior_score}"
            )
            return self._governance_report
        except Exception as e:
            logger.warning(f"Failed to end governance session: {e}")
            return None
        finally:
            if self._gov_session:
                self._gov_session.close()
                self._gov_session = None

    def _auto_start_governance(self) -> None:
        """Auto-start governance on first event if governance=True and not yet started."""
        if self._governance_enabled and self._gov_session is None:
            self.start_governance(
                task="LangGraph execution (auto-started)",
                available_tools=self._tool_names_seen if self._tool_names_seen else None,
            )

    def _report_governance_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Buffer a governance event and auto-check when interval is reached."""
        if not self._governance_enabled or not self._gov_session:
            return

        # Enrich with graph context
        if self._current_node:
            data["langgraph_node"] = self._current_node

        self._gov_event_buffer.append({
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
        })

        if len(self._gov_event_buffer) >= self._gov_check_interval:
            try:
                events = self._gov_event_buffer[:]
                self._gov_event_buffer.clear()
                result = self._gov_session.check_behavior(events)
                if result.has_flags:
                    logger.warning(
                        f"[Governance] Behavioral flags in LangGraph execution: "
                        f"{[f.detector for f in result.flags]}"
                    )
            except Exception as e:
                logger.warning(f"Failed to check governance behavior: {e}")

    # ─── Node / Chain Callbacks (Graph Structure) ────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when a chain/node execution begins.

        In LangGraph, each node execution triggers on_chain_start.
        The metadata dict contains 'langgraph_node' identifying which
        graph node is executing.
        """
        self._auto_start_governance()

        if serialized is None:
            serialized = {}

        # Extract the graph node name from LangGraph metadata
        lg_metadata = metadata or {}
        node_name = lg_metadata.get("langgraph_node", None)
        chain_type = serialized.get("id", ["", ""])[-1] if serialized.get("id") else "unknown"

        # Track node execution
        if node_name:
            prev_node = self._current_node
            self._current_node = node_name

            if node_name not in self._node_execution_order:
                self._node_execution_order.append(node_name)

            self._node_timings[node_name] = {
                "start": time.time(),
                "end": 0.0,
                "duration_ms": 0.0,
            }

            # Record state transition
            if prev_node and prev_node != node_name:
                self._state_transitions.append({
                    "from_node": prev_node,
                    "to_node": node_name,
                    "timestamp": time.time(),
                })

        self._run_map[str(run_id)] = {
            "type": "node" if node_name else "chain",
            "node_name": node_name,
            "chain_type": chain_type,
            "inputs": _safe_serialize(inputs),
            "start_time": time.time(),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": lg_metadata,
        }

        # Report node start to governance
        if node_name:
            self._report_governance_event("decision", {
                "node": node_name,
                "chain_type": chain_type,
                "reasoning": f"Executing graph node: {node_name}",
            })

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node execution completes."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)
        node_name = run_data.get("node_name")

        # Update node timing
        if node_name and node_name in self._node_timings:
            self._node_timings[node_name]["end"] = time.time()
            self._node_timings[node_name]["duration_ms"] = latency_ms

        event_type = "node_execution" if node_name else "chain_execution"
        event = {
            "event_type": event_type,
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
            "node_name": node_name,
            "chain_type": run_data.get("chain_type", "unknown"),
            "inputs": run_data.get("inputs"),
            "outputs": _safe_serialize(outputs),
            "latency_ms": latency_ms,
            "tags": run_data.get("tags"),
            "metadata": run_data.get("metadata"),
        }
        self._buffer_event(event)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain/node execution fails."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)
        node_name = run_data.get("node_name")

        event = {
            "event_type": "node_error" if node_name else "chain_error",
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
            "node_name": node_name,
            "chain_type": run_data.get("chain_type", "unknown"),
            "error": str(error),
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
        }
        self._buffer_event(event)

        self._report_governance_event("error", {
            "error": str(error),
            "error_type": type(error).__name__,
            "node": node_name,
        })

    # ─── LLM Callbacks ───────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call begins (within a graph node)."""
        self._auto_start_governance()

        if serialized is None:
            serialized = {}
        model_name = serialized.get("kwargs", {}).get("model_name", "")
        if not model_name:
            model_name = serialized.get("id", ["", ""])[-1] if serialized.get("id") else "unknown"

        invocation_params = kwargs.get("invocation_params", {})

        self._run_map[str(run_id)] = {
            "type": "llm",
            "model": model_name or invocation_params.get("model_name", "unknown"),
            "prompts": prompts,
            "start_time": time.time(),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
            "invocation_params": invocation_params,
            "langgraph_node": self._current_node,
        }

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model call begins (within a graph node)."""
        self._auto_start_governance()

        if serialized is None:
            serialized = {}
        model_name = serialized.get("kwargs", {}).get("model_name", "")
        if not model_name:
            model_name = serialized.get("id", ["", ""])[-1] if serialized.get("id") else "unknown"

        invocation_params = kwargs.get("invocation_params", {})

        serialized_messages = []
        for msg_list in messages:
            for msg in msg_list:
                serialized_messages.append({
                    "role": msg.type,
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                })

        self._run_map[str(run_id)] = {
            "type": "chat_model",
            "model": model_name or invocation_params.get("model_name", "unknown"),
            "messages": serialized_messages,
            "start_time": time.time(),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
            "langgraph_node": self._current_node,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call completes."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        completions = []
        for gen_list in response.generations:
            for gen in gen_list:
                if isinstance(gen, ChatGeneration) and gen.message:
                    completions.append(gen.message.content)
                elif isinstance(gen, Generation):
                    completions.append(gen.text)

        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        event = {
            "event_type": "llm_call",
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
            "model": run_data.get("model", "unknown"),
            "prompts": run_data.get("prompts", []),
            "completions": completions,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "langgraph_node": run_data.get("langgraph_node"),
            "tags": run_data.get("tags"),
            "metadata": run_data.get("metadata"),
        }
        self._buffer_event(event)

        self._report_governance_event("llm_call", {
            "model": run_data.get("model", "unknown"),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "latency_ms": latency_ms,
        })

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call fails."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        event = {
            "event_type": "llm_error",
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
            "model": run_data.get("model", "unknown"),
            "error": str(error),
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
            "langgraph_node": run_data.get("langgraph_node"),
        }
        self._buffer_event(event)

        self._report_governance_event("error", {
            "error": str(error),
            "error_type": type(error).__name__,
            "model": run_data.get("model", "unknown"),
        })

    # ─── Tool Callbacks ──────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool invocation begins (within a graph node)."""
        if serialized is None:
            serialized = {}
        tool_name = serialized.get("name", "unknown")

        # Track tool names globally and per-node
        if tool_name not in self._tool_names_seen:
            self._tool_names_seen.append(tool_name)

        if self._current_node:
            if self._current_node not in self._node_to_tools:
                self._node_to_tools[self._current_node] = []
            if tool_name not in self._node_to_tools[self._current_node]:
                self._node_to_tools[self._current_node].append(tool_name)

        self._run_map[str(run_id)] = {
            "type": "tool",
            "tool_name": tool_name,
            "input": input_str,
            "start_time": time.time(),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
            "langgraph_node": self._current_node,
        }

        self._report_governance_event("tool_call", {
            "tool_name": tool_name,
            "input": _safe_serialize(input_str),
        })

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool invocation completes."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        event = {
            "event_type": "tool_invocation",
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
            "tool_name": run_data.get("tool_name", "unknown"),
            "input": run_data.get("input"),
            "output": _safe_serialize(output),
            "latency_ms": latency_ms,
            "langgraph_node": run_data.get("langgraph_node"),
            "tags": run_data.get("tags"),
            "metadata": run_data.get("metadata"),
        }
        self._buffer_event(event)

        self._report_governance_event("tool_result", {
            "tool_name": run_data.get("tool_name", "unknown"),
            "output": _safe_serialize(output),
            "latency_ms": latency_ms,
            "success": True,
        })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool invocation fails."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        event = {
            "event_type": "tool_error",
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
            "tool_name": run_data.get("tool_name", "unknown"),
            "error": str(error),
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
            "langgraph_node": run_data.get("langgraph_node"),
        }
        self._buffer_event(event)

        self._report_governance_event("error", {
            "tool_name": run_data.get("tool_name", "unknown"),
            "error": str(error),
            "error_type": "tool_error",
        })

    # ─── Agent Callbacks ─────────────────────────────────────────

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent decides on an action (e.g., in a ReAct-style node)."""
        event = {
            "event_type": "agent_action",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tool": action.tool,
            "tool_input": _safe_serialize(action.tool_input),
            "log": action.log,
            "langgraph_node": self._current_node,
        }
        self._buffer_event(event)

        self._report_governance_event("decision", {
            "tool": action.tool,
            "tool_input": _safe_serialize(action.tool_input),
            "reasoning": action.log,
        })

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent completes."""
        event = {
            "event_type": "agent_finish",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "output": _safe_serialize(finish.return_values),
            "log": finish.log,
            "langgraph_node": self._current_node,
        }
        self._buffer_event(event)

    # ─── Event Buffering & Flushing ──────────────────────────────

    def _buffer_event(self, event: Dict[str, Any]) -> None:
        """Add event to buffer and flush if needed."""
        ctx = prysm_context.get()
        event["session_id"] = self.session_id
        event["user_id"] = self.user_id or ctx.user_id
        event["timestamp"] = time.time()
        event["prysm_metadata"] = {**self.metadata, **ctx.metadata}

        self._events.append(event)

        if (
            len(self._events) >= self.batch_size
            or time.time() - self._last_flush > self.flush_interval
        ):
            self.flush()

    def flush(self) -> None:
        """Send buffered events to Prysm telemetry endpoint."""
        if not self._events:
            return

        events_to_send = self._events[:]
        self._events.clear()
        self._last_flush = time.time()

        try:
            self._client.post(
                f"{self.base_url}/telemetry/events",
                json={
                    "source": "langgraph",
                    "session_id": self.session_id,
                    "events": events_to_send,
                    "graph_summary": self.graph_summary,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to send telemetry events to Prysm: {e}")
            if len(self._events) < self.batch_size * 2:
                self._events.extend(events_to_send)

    def close(self) -> None:
        """Flush remaining events, end governance session, and close the HTTP client."""
        self.flush()
        if self._gov_session and self._gov_session.is_active:
            try:
                self.end_governance(outcome="partial")
            except Exception:
                pass
        self._client.close()

    def reset(self) -> None:
        """Reset graph-specific tracking state for a new graph execution."""
        self._node_execution_order.clear()
        self._node_timings.clear()
        self._current_node = None
        self._node_to_tools.clear()
        self._state_transitions.clear()

    def __del__(self) -> None:
        """Ensure events are flushed on garbage collection."""
        try:
            self.flush()
        except Exception:
            pass


# ─── Helpers ─────────────────────────────────────────────────────

def _safe_serialize(obj: Any, max_length: int = 2000) -> Any:
    """Safely serialize an object for JSON transmission, truncating if needed."""
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, str) and len(obj) > max_length:
                return obj[:max_length] + "...[truncated]"
            return obj
        if isinstance(obj, (list, tuple)):
            return [_safe_serialize(item, max_length) for item in obj[:50]]
        if isinstance(obj, dict):
            return {
                str(k): _safe_serialize(v, max_length)
                for k, v in list(obj.items())[:50]
            }
        s = str(obj)
        if len(s) > max_length:
            return s[:max_length] + "...[truncated]"
        return s
    except Exception:
        return "<unserializable>"
