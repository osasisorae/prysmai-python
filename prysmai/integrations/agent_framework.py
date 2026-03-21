"""
Prysm AI — Microsoft Agent Framework Integration

Provides PrysmAgentFrameworkMonitor that uses Agent Framework's middleware
system to capture agent runs, function/tool calls, and chat/LLM calls,
routing all telemetry through the Prysm proxy.

Agent Framework supports three middleware types:
  - Agent middleware: intercepts agent run execution
  - Function middleware: intercepts tool/function calls
  - Chat middleware: intercepts LLM calls to AI models

Usage:
    from prysmai.integrations.agent_framework import PrysmAgentFrameworkMonitor

    monitor = PrysmAgentFrameworkMonitor(api_key="sk-prysm-...")

    # Register as agent-level middleware (persistent across all runs)
    agent = client.as_agent(
        name="MyAgent",
        instructions="...",
        middleware=monitor.middleware(),
    )

    # Or register as run-level middleware (per-invocation)
    await agent.run("prompt", middleware=monitor.middleware())

    # Access telemetry
    print(monitor.execution_summary)
    monitor.flush()

With governance (v0.5.0):
    monitor = PrysmAgentFrameworkMonitor(api_key="sk-prysm-...", governance=True)
    monitor.start_governance(task="Process customer request")
    await agent.run("prompt", middleware=monitor.middleware())
    report = monitor.end_governance()

Blueprint Reference: Section 9.3 (Agent Framework), v0.6.0
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

try:
    from agent_framework import (
        AgentMiddleware,
        FunctionMiddleware,
        ChatMiddleware,
        AgentContext,
        FunctionInvocationContext,
        ChatContext,
    )

    _AF_AVAILABLE = True
except ImportError:
    _AF_AVAILABLE = False

from prysmai.context import prysm_context
from prysmai.config import resolve_prysm_connection

logger = logging.getLogger("prysmai.integrations.agent_framework")


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


def _extract_message_content(messages: Any) -> List[Dict[str, Any]]:
    """Extract serializable content from chat messages."""
    if not messages:
        return []
    result = []
    for msg in messages:
        try:
            if isinstance(msg, dict):
                result.append(msg)
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                result.append({
                    "role": str(getattr(msg, "role", "unknown")),
                    "content": _safe_serialize(getattr(msg, "content", "")),
                })
            else:
                result.append({"raw": _safe_serialize(msg)})
        except Exception:
            result.append({"raw": "<unserializable>"})
    return result


# ─── Middleware Classes ──────────────────────────────────────────


class _PrysmAgentMiddleware(AgentMiddleware if _AF_AVAILABLE else object):
    """Agent middleware that captures agent run execution telemetry."""

    def __init__(self, monitor: "PrysmAgentFrameworkMonitor"):
        self._monitor = monitor

    async def process(
        self,
        context: "AgentContext",
        next: Callable[["AgentContext"], Awaitable[None]],
    ) -> None:
        """Intercept agent run execution."""
        run_id = str(uuid.uuid4())
        agent_name = getattr(context.agent, "name", "unknown") if context.agent else "unknown"
        is_streaming = getattr(context, "is_streaming", False)

        start_time = time.time()
        self._monitor._active_runs[run_id] = {
            "agent_name": agent_name,
            "start_time": start_time,
            "is_streaming": is_streaming,
        }

        # Pre-processing event
        self._monitor._buffer_event({
            "event_type": "agent_run_start",
            "run_id": run_id,
            "agent_name": agent_name,
            "is_streaming": is_streaming,
            "message_count": len(context.messages) if context.messages else 0,
            "messages_preview": _extract_message_content(
                context.messages[:3] if context.messages else []
            ),
        })

        self._monitor._report_governance_event("decision", {
            "action": "agent_run_start",
            "agent_name": agent_name,
            "reasoning": f"Starting agent run: {agent_name}",
        })

        error_occurred = None
        try:
            await next(context)
        except Exception as e:
            error_occurred = e
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)
            self._monitor._active_runs.pop(run_id, None)

            # Track agent execution
            self._monitor._agent_runs.append({
                "run_id": run_id,
                "agent_name": agent_name,
                "latency_ms": latency_ms,
                "success": error_occurred is None,
                "is_streaming": is_streaming,
                "terminated": getattr(context, "terminate", False),
            })

            event_data: Dict[str, Any] = {
                "event_type": "agent_run_end",
                "run_id": run_id,
                "agent_name": agent_name,
                "latency_ms": latency_ms,
                "terminated": getattr(context, "terminate", False),
            }

            if error_occurred:
                event_data["error"] = str(error_occurred)
                event_data["error_type"] = type(error_occurred).__name__
                event_data["success"] = False
                self._monitor._report_governance_event("error", {
                    "error": str(error_occurred),
                    "error_type": type(error_occurred).__name__,
                    "agent_name": agent_name,
                })
            else:
                event_data["success"] = True
                # Capture result preview if available
                result = getattr(context, "result", None)
                if result is not None:
                    event_data["result_preview"] = _safe_serialize(result, 500)

            self._monitor._buffer_event(event_data)


class _PrysmFunctionMiddleware(FunctionMiddleware if _AF_AVAILABLE else object):
    """Function middleware that captures tool/function call telemetry."""

    def __init__(self, monitor: "PrysmAgentFrameworkMonitor"):
        self._monitor = monitor

    async def process(
        self,
        context: "FunctionInvocationContext",
        next: Callable[["FunctionInvocationContext"], Awaitable[None]],
    ) -> None:
        """Intercept function/tool execution."""
        func = context.function if hasattr(context, "function") else None
        func_name = getattr(func, "name", "unknown") if func else "unknown"
        arguments = _safe_serialize(
            getattr(context, "arguments", {}), 1000
        )

        call_id = str(uuid.uuid4())
        start_time = time.time()

        # Pre-processing event
        self._monitor._buffer_event({
            "event_type": "tool_call_start",
            "call_id": call_id,
            "tool_name": func_name,
            "arguments": arguments,
        })

        self._monitor._report_governance_event("tool_call", {
            "tool_name": func_name,
            "input": arguments,
        })

        error_occurred = None
        try:
            await next(context)
        except Exception as e:
            error_occurred = e
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            # Track tool usage
            if func_name not in self._monitor._tool_names_seen:
                self._monitor._tool_names_seen.append(func_name)

            tool_record = {
                "call_id": call_id,
                "tool_name": func_name,
                "latency_ms": latency_ms,
                "success": error_occurred is None,
            }
            self._monitor._tool_calls.append(tool_record)

            event_data: Dict[str, Any] = {
                "event_type": "tool_call_end",
                "call_id": call_id,
                "tool_name": func_name,
                "arguments": arguments,
                "latency_ms": latency_ms,
                "terminated": getattr(context, "terminate", False),
            }

            if error_occurred:
                event_data["error"] = str(error_occurred)
                event_data["error_type"] = type(error_occurred).__name__
                event_data["success"] = False
                self._monitor._report_governance_event("error", {
                    "tool_name": func_name,
                    "error": str(error_occurred),
                    "error_type": "tool_error",
                })
            else:
                event_data["success"] = True
                result = getattr(context, "result", None)
                if result is not None:
                    event_data["result_preview"] = _safe_serialize(result, 500)
                self._monitor._report_governance_event("tool_result", {
                    "tool_name": func_name,
                    "success": True,
                    "latency_ms": latency_ms,
                })

            self._monitor._buffer_event(event_data)


class _PrysmChatMiddleware(ChatMiddleware if _AF_AVAILABLE else object):
    """Chat middleware that captures LLM call telemetry."""

    def __init__(self, monitor: "PrysmAgentFrameworkMonitor"):
        self._monitor = monitor

    async def process(
        self,
        context: "ChatContext",
        next: Callable[["ChatContext"], Awaitable[None]],
    ) -> None:
        """Intercept chat/LLM calls to AI models."""
        messages = _extract_message_content(
            getattr(context, "messages", [])
        )
        is_streaming = getattr(context, "is_streaming", False)
        options = _safe_serialize(getattr(context, "options", {}), 500)

        call_id = str(uuid.uuid4())
        start_time = time.time()

        # Pre-processing event
        self._monitor._buffer_event({
            "event_type": "llm_call_start",
            "call_id": call_id,
            "message_count": len(messages),
            "messages_preview": messages[:3],
            "is_streaming": is_streaming,
            "options": options,
        })

        self._monitor._report_governance_event("llm_call", {
            "message_count": len(messages),
            "is_streaming": is_streaming,
        })

        error_occurred = None
        try:
            await next(context)
        except Exception as e:
            error_occurred = e
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            # Track LLM call
            llm_record: Dict[str, Any] = {
                "call_id": call_id,
                "latency_ms": latency_ms,
                "success": error_occurred is None,
                "is_streaming": is_streaming,
            }

            # Extract token usage from result if available
            result = getattr(context, "result", None)
            if result is not None and error_occurred is None:
                usage = getattr(result, "usage", None)
                if usage:
                    llm_record["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
                    llm_record["completion_tokens"] = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
                    llm_record["total_tokens"] = (
                        llm_record["prompt_tokens"] + llm_record["completion_tokens"]
                    )
                model = getattr(result, "model", None)
                if model:
                    llm_record["model"] = model

            self._monitor._llm_calls.append(llm_record)

            event_data: Dict[str, Any] = {
                "event_type": "llm_call_end",
                "call_id": call_id,
                "latency_ms": latency_ms,
                "is_streaming": is_streaming,
                "terminated": getattr(context, "terminate", False),
            }

            if error_occurred:
                event_data["error"] = str(error_occurred)
                event_data["error_type"] = type(error_occurred).__name__
                event_data["success"] = False
            else:
                event_data["success"] = True
                if result is not None:
                    # Extract key response metadata
                    usage = getattr(result, "usage", None)
                    if usage:
                        event_data["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
                        event_data["completion_tokens"] = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
                    model = getattr(result, "model", None)
                    if model:
                        event_data["model"] = model
                    # Capture completion preview
                    choices = getattr(result, "choices", None)
                    if choices and len(choices) > 0:
                        first_choice = choices[0]
                        msg = getattr(first_choice, "message", None)
                        if msg:
                            event_data["completion_preview"] = _safe_serialize(
                                getattr(msg, "content", ""), 500
                            )

            self._monitor._buffer_event(event_data)


# ─── Main Monitor Class ─────────────────────────────────────────


class PrysmAgentFrameworkMonitor:
    """
    Microsoft Agent Framework monitor that captures execution telemetry
    via the framework's middleware system and sends it to Prysm.

    Captures:
        - Agent run execution (start/end, timing, messages, results)
        - Function/tool calls (name, arguments, results, timing, errors)
        - Chat/LLM calls (messages, model, tokens, response, timing)

    When governance=True, events are forwarded to a GovernanceSession
    for behavioral analysis, security scanning, and policy enforcement.

    All captured data is sent to the Prysm telemetry endpoint as structured events.
    """

    def __init__(
        self,
        client: Any = None,
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
        Initialize the Prysm Agent Framework monitor.

        Args:
            client: Optional PrysmClient or PrysmMCPClient providing shared auth and base URL.
            api_key: Prysm API key (sk-prysm-...). Falls back to PRYSM_API_KEY env var.
            base_url: Prysm proxy base URL. Falls back to PRYSM_BASE_URL.
            session_id: Optional session ID for grouping related executions.
            user_id: Optional user ID for attribution.
            metadata: Optional metadata dict attached to all events.
            batch_size: Number of events to buffer before sending (default 10).
            flush_interval: Max seconds between flushes (default 5.0).
            governance: Enable governance monitoring (v0.5.0+). When True,
                events are forwarded to a GovernanceSession for behavioral analysis.
            governance_context: Additional context for the governance session.
        """
        if not _AF_AVAILABLE:
            raise ImportError(
                "Microsoft Agent Framework integration requires agent-framework. "
                "Install it with: pip install prysmai[agent-framework]"
            )

        resolved = resolve_prysm_connection(
            client=client,
            prysm_key=api_key,
            base_url=base_url,
        )
        self.api_key = resolved.prysm_key
        self.base_url = resolved.base_url
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.metadata = metadata or {}
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Internal state
        self._events: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._client = httpx.Client(timeout=30.0)

        # Execution tracking
        self._agent_runs: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._llm_calls: List[Dict[str, Any]] = []
        self._tool_names_seen: List[str] = []
        self._active_runs: Dict[str, Dict[str, Any]] = {}

        # Middleware instances (created once, reused)
        self._agent_mw = _PrysmAgentMiddleware(self)
        self._function_mw = _PrysmFunctionMiddleware(self)
        self._chat_mw = _PrysmChatMiddleware(self)

        # Governance state (v0.5.0)
        self._governance_enabled = governance
        self._governance_context = governance_context or {}
        self._gov_session: Any = None
        self._gov_event_buffer: List[Dict[str, Any]] = []
        self._gov_check_interval = 5
        self._governance_report: Any = None

    # ─── Middleware Access ──────────────────────────────────────

    def middleware(self) -> List:
        """
        Return the list of middleware instances to register with an agent or run.

        Usage:
            agent = client.as_agent(name="Bot", middleware=monitor.middleware())
            # or
            await agent.run("prompt", middleware=monitor.middleware())
        """
        return [self._agent_mw, self._function_mw, self._chat_mw]

    @property
    def agent_middleware(self) -> "_PrysmAgentMiddleware":
        """The agent middleware instance (for selective registration)."""
        return self._agent_mw

    @property
    def function_middleware(self) -> "_PrysmFunctionMiddleware":
        """The function middleware instance (for selective registration)."""
        return self._function_mw

    @property
    def chat_middleware(self) -> "_PrysmChatMiddleware":
        """The chat middleware instance (for selective registration)."""
        return self._chat_mw

    # ─── Properties ────────────────────────────────────────────

    @property
    def governance_report(self) -> Any:
        """The governance report from the last session (if governance=True)."""
        return self._governance_report

    @property
    def execution_summary(self) -> Dict[str, Any]:
        """Summary of all captured execution data."""
        total_llm_tokens = sum(c.get("total_tokens", 0) for c in self._llm_calls)
        avg_tool_latency = (
            sum(c["latency_ms"] for c in self._tool_calls) / len(self._tool_calls)
            if self._tool_calls
            else 0
        )
        avg_llm_latency = (
            sum(c["latency_ms"] for c in self._llm_calls) / len(self._llm_calls)
            if self._llm_calls
            else 0
        )
        tool_success_rate = (
            sum(1 for c in self._tool_calls if c.get("success")) / len(self._tool_calls)
            if self._tool_calls
            else 1.0
        )

        return {
            "session_id": self.session_id,
            "agent_runs": len(self._agent_runs),
            "tool_calls": len(self._tool_calls),
            "llm_calls": len(self._llm_calls),
            "tools_used": self._tool_names_seen[:],
            "total_llm_tokens": total_llm_tokens,
            "avg_tool_latency_ms": round(avg_tool_latency, 1),
            "avg_llm_latency_ms": round(avg_llm_latency, 1),
            "tool_success_rate": round(tool_success_rate, 3),
        }

    # ─── Governance Lifecycle ──────────────────────────────────

    def start_governance(
        self,
        task: str = "Agent Framework execution",
        agent_type: str = "agent_framework",
        available_tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start a governance session for this execution.

        Call this before invoking your agent. If governance=True was set
        in the constructor, this is called automatically on the first
        event. Calling it explicitly gives you control over the task
        description and available tools.

        Args:
            task: Description of the task being performed.
            agent_type: Agent type identifier (default "agent_framework").
            available_tools: List of tool names available to the agent.
            context: Additional context for the governance session.
        """
        if self._gov_session and self._gov_session.is_active:
            logger.warning("Governance session already active. Ending previous session.")
            self.end_governance()

        try:
            from prysmai.governance import GovernanceSession

            merged_context = {
                "framework": "agent_framework",
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
                f"Governance session started for Agent Framework: {self._gov_session.session_id}"
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
                output_summary=output_summary or json.dumps(self.execution_summary),
            )
            logger.info(
                f"Governance session ended for Agent Framework: "
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
                task="Agent Framework execution (auto-started)",
                available_tools=self._tool_names_seen if self._tool_names_seen else None,
            )

    def _report_governance_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Buffer a governance event and auto-check when interval is reached."""
        if not self._governance_enabled:
            return

        self._auto_start_governance()

        if not self._gov_session:
            return

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
                        f"[Governance] Behavioral flags in Agent Framework execution: "
                        f"{[f.detector for f in result.flags]}"
                    )
            except Exception as e:
                logger.warning(f"Failed to check governance behavior: {e}")

    # ─── Event Buffering & Flushing ────────────────────────────

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
                    "source": "agent_framework",
                    "session_id": self.session_id,
                    "events": events_to_send,
                    "execution_summary": self.execution_summary,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to send telemetry events to Prysm: {e}")
            # Re-buffer events if there's room
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
        """Reset tracking state for a new execution."""
        self._agent_runs.clear()
        self._tool_calls.clear()
        self._llm_calls.clear()
        self._tool_names_seen.clear()
        self._active_runs.clear()

    def __del__(self) -> None:
        """Ensure events are flushed on garbage collection."""
        try:
            self.flush()
        except Exception:
            pass
