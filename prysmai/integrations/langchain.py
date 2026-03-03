"""
Prysm AI — LangChain Integration

Provides PrysmCallbackHandler that captures LLM calls, chain executions,
and tool invocations from LangChain and routes telemetry through the Prysm proxy.

Usage:
    from prysmai.integrations.langchain import PrysmCallbackHandler

    handler = PrysmCallbackHandler(api_key="sk-prysm-...")
    chain.invoke(input, config={"callbacks": [handler]})

Blueprint Reference: Section 9.3, page 28
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
        "LangChain integration requires langchain-core. "
        "Install it with: pip install prysmai[langchain]"
    )

from prysmai.context import prysm_context

logger = logging.getLogger("prysmai.integrations.langchain")


class PrysmCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures telemetry and sends it to Prysm.

    Captures:
        - LLM calls (model, prompts, completions, token usage)
        - Chain executions (chain type, inputs, outputs)
        - Tool invocations (tool name, input, output)
        - Agent actions and decisions
        - Error events

    All captured data is sent to the Prysm telemetry endpoint as structured events.
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
    ):
        """
        Initialize the Prysm LangChain callback handler.

        Args:
            api_key: Prysm API key (sk-prysm-...). Falls back to PRYSM_API_KEY env var.
            base_url: Prysm proxy base URL. Falls back to PRYSM_BASE_URL or https://prysmai.io/api/v1.
            session_id: Optional session ID for grouping related chain executions.
            user_id: Optional user ID for attribution.
            metadata: Optional metadata dict attached to all events.
            batch_size: Number of events to buffer before sending (default 10).
            flush_interval: Max seconds between flushes (default 5.0).
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
        self._run_map: Dict[str, Dict[str, Any]] = {}  # run_id -> run metadata
        self._last_flush = time.time()
        self._client = httpx.Client(timeout=30.0)

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
        """Called when an LLM call begins."""
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

        # Extract completion text
        completions = []
        for gen_list in response.generations:
            for gen in gen_list:
                if isinstance(gen, ChatGeneration) and gen.message:
                    completions.append(gen.message.content)
                elif isinstance(gen, Generation):
                    completions.append(gen.text)

        # Extract token usage
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
            "tags": run_data.get("tags"),
            "metadata": run_data.get("metadata"),
        }
        self._buffer_event(event)

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
        }
        self._buffer_event(event)

    # ─── Chat Model Callbacks ────────────────────────────────────

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
        """Called when a chat model call begins."""
        model_name = serialized.get("kwargs", {}).get("model_name", "")
        if not model_name:
            model_name = serialized.get("id", ["", ""])[-1] if serialized.get("id") else "unknown"

        invocation_params = kwargs.get("invocation_params", {})

        # Serialize messages for telemetry
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
        }

    # ─── Chain Callbacks ─────────────────────────────────────────

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
        """Called when a chain execution begins."""
        chain_type = serialized.get("id", ["", ""])[-1] if serialized.get("id") else "unknown"

        self._run_map[str(run_id)] = {
            "type": "chain",
            "chain_type": chain_type,
            "inputs": _safe_serialize(inputs),
            "start_time": time.time(),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
        }

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain execution completes."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        event = {
            "event_type": "chain_execution",
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
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
        """Called when a chain execution fails."""
        run_data = self._run_map.pop(str(run_id), {})
        start_time = run_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        event = {
            "event_type": "chain_error",
            "run_id": str(run_id),
            "parent_run_id": run_data.get("parent_run_id"),
            "chain_type": run_data.get("chain_type", "unknown"),
            "error": str(error),
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
        }
        self._buffer_event(event)

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
        """Called when a tool invocation begins."""
        tool_name = serialized.get("name", "unknown")

        self._run_map[str(run_id)] = {
            "type": "tool",
            "tool_name": tool_name,
            "input": input_str,
            "start_time": time.time(),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata,
        }

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
            "tags": run_data.get("tags"),
            "metadata": run_data.get("metadata"),
        }
        self._buffer_event(event)

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
        }
        self._buffer_event(event)

    # ─── Agent Callbacks ─────────────────────────────────────────

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent decides on an action."""
        event = {
            "event_type": "agent_action",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tool": action.tool,
            "tool_input": _safe_serialize(action.tool_input),
            "log": action.log,
        }
        self._buffer_event(event)

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
        }
        self._buffer_event(event)

    # ─── Event Buffering & Flushing ──────────────────────────────

    def _buffer_event(self, event: Dict[str, Any]) -> None:
        """Add event to buffer and flush if needed."""
        # Merge context metadata
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
                    "source": "langchain",
                    "session_id": self.session_id,
                    "events": events_to_send,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to send telemetry events to Prysm: {e}")
            # Re-buffer events on failure (up to 2x batch_size to prevent unbounded growth)
            if len(self._events) < self.batch_size * 2:
                self._events.extend(events_to_send)

    def close(self) -> None:
        """Flush remaining events and close the HTTP client."""
        self.flush()
        self._client.close()

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
        # For other types, try str()
        s = str(obj)
        if len(s) > max_length:
            return s[:max_length] + "...[truncated]"
        return s
    except Exception:
        return "<unserializable>"
