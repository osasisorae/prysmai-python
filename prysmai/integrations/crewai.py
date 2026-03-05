"""
Prysm AI — CrewAI Integration

Provides PrysmCrewMonitor that captures agent executions, task completions,
tool calls, and delegation events from CrewAI crews.

Usage:
    from prysmai.integrations.crewai import PrysmCrewMonitor

    monitor = PrysmCrewMonitor(api_key="sk-prysm-...")
    crew = Crew(agents=[...], tasks=[...], callbacks=[monitor])

Blueprint Reference: Section 9.3, page 28
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from typing import Any, Dict, List, Optional

import httpx

try:
    from crewai.utilities.events import (
        crewai_event_bus,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        TaskExecutionStartedEvent,
        TaskExecutionCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
        ToolUsageErrorEvent,
        CrewKickoffStartedEvent,
        CrewKickoffCompletedEvent,
    )

    _CREWAI_EVENTS_AVAILABLE = True
except ImportError:
    try:
        # Fallback: older CrewAI versions may not have the event bus
        from crewai import Crew  # noqa: F401

        _CREWAI_EVENTS_AVAILABLE = False
    except ImportError:
        raise ImportError(
            "CrewAI integration requires crewai. "
            "Install it with: pip install prysmai[crewai]"
        )

from prysmai.context import prysm_context

logger = logging.getLogger("prysmai.integrations.crewai")


class PrysmCrewMonitor:
    """
    CrewAI monitor that captures crew execution telemetry and sends it to Prysm.

    Captures:
        - Crew kickoff and completion (total execution time)
        - Agent executions (agent name, role, goal, backstory)
        - Task executions (task description, expected output, actual output)
        - Tool usage per agent (tool name, input, output, errors)
        - Delegation events between agents

    All captured data is sent to the Prysm telemetry endpoint as structured events.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Prysm CrewAI monitor.

        Args:
            api_key: Prysm API key (sk-prysm-...). Falls back to PRYSM_API_KEY env var.
            base_url: Prysm proxy base URL. Falls back to PRYSM_BASE_URL or https://prysmai.io/api/v1.
            session_id: Optional session ID for grouping related crew executions.
            user_id: Optional user ID for attribution.
            metadata: Optional metadata dict attached to all events.
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

        if not self.api_key:
            raise ValueError(
                "Prysm API key is required. Pass api_key= or set PRYSM_API_KEY env var."
            )

        # Internal state
        self._events: List[Dict[str, Any]] = []
        self._agent_timers: Dict[str, float] = {}
        self._task_timers: Dict[str, float] = {}
        self._tool_timers: Dict[str, float] = {}
        self._crew_start_time: Optional[float] = None
        self._client = httpx.Client(timeout=30.0)
        self._subscribed = False

    def subscribe(self) -> None:
        """Subscribe to CrewAI event bus events."""
        if self._subscribed:
            return

        if _CREWAI_EVENTS_AVAILABLE:
            crewai_event_bus.on(CrewKickoffStartedEvent, self._on_crew_start)
            crewai_event_bus.on(CrewKickoffCompletedEvent, self._on_crew_end)
            crewai_event_bus.on(AgentExecutionStartedEvent, self._on_agent_start)
            crewai_event_bus.on(AgentExecutionCompletedEvent, self._on_agent_end)
            crewai_event_bus.on(TaskExecutionStartedEvent, self._on_task_start)
            crewai_event_bus.on(TaskExecutionCompletedEvent, self._on_task_end)
            crewai_event_bus.on(ToolUsageStartedEvent, self._on_tool_start)
            crewai_event_bus.on(ToolUsageFinishedEvent, self._on_tool_end)
            crewai_event_bus.on(ToolUsageErrorEvent, self._on_tool_error)
            self._subscribed = True
            logger.info("Prysm CrewAI monitor subscribed to event bus")
        else:
            logger.warning(
                "CrewAI event bus not available. "
                "Using callback-based monitoring (limited telemetry)."
            )

    # ─── CrewAI Callback Interface (for Crew(callbacks=[monitor])) ──

    def __call__(self, event_type: str, data: Any = None) -> None:
        """
        Called by CrewAI's callback system.
        This is the fallback for older CrewAI versions without the event bus.
        """
        if not self._subscribed:
            self.subscribe()

        # Handle string-based callbacks from older CrewAI
        event = {
            "event_type": f"crew_callback_{event_type}",
            "data": _safe_serialize(data),
            "timestamp": time.time(),
        }
        self._buffer_event(event)

    # ─── Event Bus Handlers ──────────────────────────────────────

    def _on_crew_start(self, event: Any) -> None:
        """Called when a crew kickoff begins."""
        self._crew_start_time = time.time()
        ev = {
            "event_type": "crew_kickoff_started",
            "crew_id": str(uuid.uuid4()),
        }
        self._buffer_event(ev)

    def _on_crew_end(self, event: Any) -> None:
        """Called when a crew kickoff completes."""
        latency_ms = 0
        if self._crew_start_time:
            latency_ms = int((time.time() - self._crew_start_time) * 1000)

        output = getattr(event, "output", None)
        ev = {
            "event_type": "crew_kickoff_completed",
            "latency_ms": latency_ms,
            "output": _safe_serialize(output),
        }
        self._buffer_event(ev)
        self.flush()

    def _on_agent_start(self, event: Any) -> None:
        """Called when an agent execution begins."""
        agent = getattr(event, "agent", None)
        agent_name = getattr(agent, "role", "unknown") if agent else "unknown"
        agent_id = str(id(agent)) if agent else str(uuid.uuid4())
        self._agent_timers[agent_id] = time.time()

        ev = {
            "event_type": "agent_execution_started",
            "agent_name": agent_name,
            "agent_goal": getattr(agent, "goal", None) if agent else None,
            "agent_backstory": _truncate(getattr(agent, "backstory", None), 500) if agent else None,
        }
        self._buffer_event(ev)

    def _on_agent_end(self, event: Any) -> None:
        """Called when an agent execution completes."""
        agent = getattr(event, "agent", None)
        agent_name = getattr(agent, "role", "unknown") if agent else "unknown"
        agent_id = str(id(agent)) if agent else ""
        start_time = self._agent_timers.pop(agent_id, time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        output = getattr(event, "output", None)
        ev = {
            "event_type": "agent_execution_completed",
            "agent_name": agent_name,
            "output": _safe_serialize(output),
            "latency_ms": latency_ms,
        }
        self._buffer_event(ev)

    def _on_task_start(self, event: Any) -> None:
        """Called when a task execution begins."""
        task = getattr(event, "task", None)
        task_id = str(id(task)) if task else str(uuid.uuid4())
        self._task_timers[task_id] = time.time()

        ev = {
            "event_type": "task_execution_started",
            "task_description": _truncate(getattr(task, "description", None), 500) if task else None,
            "expected_output": _truncate(getattr(task, "expected_output", None), 500) if task else None,
        }
        self._buffer_event(ev)

    def _on_task_end(self, event: Any) -> None:
        """Called when a task execution completes."""
        task = getattr(event, "task", None)
        task_id = str(id(task)) if task else ""
        start_time = self._task_timers.pop(task_id, time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        output = getattr(event, "output", None)
        ev = {
            "event_type": "task_execution_completed",
            "task_description": _truncate(getattr(task, "description", None), 500) if task else None,
            "output": _safe_serialize(output),
            "latency_ms": latency_ms,
        }
        self._buffer_event(ev)

    def _on_tool_start(self, event: Any) -> None:
        """Called when a tool usage begins."""
        try:
            tool_name = getattr(event, "tool_name", "unknown")
            tool_id = f"{tool_name}_{time.time()}"
            self._tool_timers[tool_id] = time.time()

            # BUG-003 fix: Safely serialize tool_input — delegation tools
            # (DelegateWorkToolSchema) can have malformed data from gpt-4o-mini
            tool_input = None
            try:
                raw_input = getattr(event, "tool_input", None)
                if raw_input is not None:
                    tool_input = _safe_serialize(raw_input)
            except Exception:
                tool_input = "<input serialization failed>"

            ev = {
                "event_type": "tool_usage_started",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "_tool_id": tool_id,
            }
            self._buffer_event(ev)
        except Exception as e:
            logger.warning(f"Failed to capture tool_start event: {e}")

    def _on_tool_end(self, event: Any) -> None:
        """Called when a tool usage completes."""
        tool_name = getattr(event, "tool_name", "unknown")
        # Find matching timer (most recent for this tool)
        matching_key = None
        for k in reversed(list(self._tool_timers.keys())):
            if k.startswith(tool_name):
                matching_key = k
                break
        start_time = self._tool_timers.pop(matching_key, time.time()) if matching_key else time.time()
        latency_ms = int((time.time() - start_time) * 1000)

        ev = {
            "event_type": "tool_usage_completed",
            "tool_name": tool_name,
            "tool_output": _safe_serialize(getattr(event, "tool_output", None)),
            "latency_ms": latency_ms,
        }
        self._buffer_event(ev)

    def _on_tool_error(self, event: Any) -> None:
        """Called when a tool usage fails."""
        try:
            tool_name = getattr(event, "tool_name", "unknown")
            error = getattr(event, "error", "Unknown error")

            # BUG-003 fix: Safely handle delegation tool errors
            ev = {
                "event_type": "tool_usage_error",
                "tool_name": tool_name,
                "error": str(error),
            }
            self._buffer_event(ev)
        except Exception as e:
            logger.warning(f"Failed to capture tool_error event: {e}")

    # ─── Event Buffering & Flushing ──────────────────────────────

    def _buffer_event(self, event: Dict[str, Any]) -> None:
        """Add event to buffer."""
        ctx = prysm_context.get()
        event["session_id"] = self.session_id
        event["user_id"] = self.user_id or ctx.user_id
        event["timestamp"] = event.get("timestamp", time.time())
        event["prysm_metadata"] = {**self.metadata, **ctx.metadata}
        self._events.append(event)

    def flush(self) -> None:
        """Send buffered events to Prysm telemetry endpoint."""
        if not self._events:
            return

        events_to_send = self._events[:]
        self._events.clear()

        try:
            self._client.post(
                f"{self.base_url}/telemetry/events",
                json={
                    "source": "crewai",
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
    """Safely serialize an object for JSON transmission."""
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


def _truncate(text: Optional[str], max_length: int = 500) -> Optional[str]:
    """Truncate text to max_length."""
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...[truncated]"
