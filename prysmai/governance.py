"""
Prysm AI — Governance Session Client

Provides GovernanceSession, a context manager that wraps the Prysm MCP
governance endpoint. It manages the full session lifecycle (start → check
behavior → scan code → end) and parses SSE-wrapped JSON-RPC responses
from the Streamable HTTP transport.

Usage:
    from prysmai import PrysmClient
    from prysmai.governance import GovernanceSession

    client = PrysmClient(prysm_key="sk-prysm-...")

    with GovernanceSession(
        client,
        task="Fix authentication bypass in login handler",
        agent_type="claude_code",
        available_tools=["bash", "file_write", "file_read"],
    ) as gov:
        # Report events as the agent works
        result = gov.check_behavior([
            {"event_type": "llm_call", "data": {"model": "claude-4", "prompt_tokens": 500}},
            {"event_type": "tool_call", "data": {"tool_name": "bash", "input": "git diff"}},
        ])

        # Scan generated code
        scan = gov.scan_code(
            code="import subprocess; subprocess.call(user_input, shell=True)",
            language="python",
            file_path="app/utils.py",
        )

    # Session auto-ends with outcome="completed" (or "failed" on exception)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

logger = logging.getLogger("prysmai.governance")


# ─── SSE Response Parser ────────────────────────────────────────────


def _parse_sse_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse an SSE-wrapped JSON-RPC response from the MCP Streamable HTTP transport.

    The MCP server returns responses in Server-Sent Events format:
        event: message
        data: {"jsonrpc":"2.0","id":1,"result":{...}}

    This function extracts the JSON-RPC result from the SSE envelope.
    If the response is already plain JSON (not SSE), it parses it directly.
    """
    text = raw_text.strip()

    # Try plain JSON first (some transports may return raw JSON)
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if "result" in parsed:
                return parsed["result"]
            if "error" in parsed:
                raise GovernanceError(
                    f"JSON-RPC error: {parsed['error'].get('message', 'Unknown error')}",
                    code=parsed["error"].get("code"),
                )
            return parsed
        except json.JSONDecodeError:
            pass

    # Parse SSE format — find the last "data:" line with JSON content
    result = None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                parsed = json.loads(data_str)
                if "result" in parsed:
                    result = parsed["result"]
                elif "error" in parsed:
                    raise GovernanceError(
                        f"JSON-RPC error: {parsed['error'].get('message', 'Unknown error')}",
                        code=parsed["error"].get("code"),
                    )
                else:
                    result = parsed
            except json.JSONDecodeError:
                continue

    if result is None:
        raise GovernanceError(f"Failed to parse MCP response: {text[:500]}")

    return result


def _extract_tool_result(mcp_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the parsed JSON from an MCP tool call result.

    MCP tool results are wrapped in:
        {"content": [{"type": "text", "text": "{...json...}"}]}

    This function extracts and parses the inner JSON.
    """
    content = mcp_result.get("content", [])
    if not content:
        return {}

    text_content = content[0].get("text", "{}")
    try:
        return json.loads(text_content)
    except json.JSONDecodeError:
        return {"raw": text_content}


# ─── Exceptions ─────────────────────────────────────────────────────


class GovernanceError(Exception):
    """Raised when a governance operation fails."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class SessionNotActiveError(GovernanceError):
    """Raised when trying to operate on a non-active session."""
    pass


# ─── Data Classes ───────────────────────────────────────────────────


@dataclass
class BehavioralFlag:
    """A behavioral flag detected during a governance check."""
    detector: str
    severity: int
    evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Vulnerability:
    """A vulnerability found during a code scan."""
    type: str
    severity: str
    description: str


@dataclass
class CheckResult:
    """Result from a check_behavior call."""
    events_ingested: int
    flags: List[BehavioralFlag]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    raw: Dict[str, Any]

    @property
    def has_flags(self) -> bool:
        return len(self.flags) > 0

    @property
    def max_severity(self) -> int:
        if not self.flags:
            return 0
        return max(f.severity for f in self.flags)


@dataclass
class ScanResult:
    """Result from a scan_code call."""
    language: str
    file_path: Optional[str]
    vulnerability_count: int
    max_severity: str
    threat_score: int
    vulnerabilities: List[Vulnerability]
    recommendations: List[str]
    raw: Dict[str, Any]

    @property
    def is_clean(self) -> bool:
        return self.vulnerability_count == 0


@dataclass
class SessionReport:
    """Final report from a governance session."""
    session_id: str
    outcome: str
    behavior_score: Optional[int]
    detectors: List[Dict[str, Any]]
    summary: Optional[str]
    recommendations: List[str]
    violations: List[Dict[str, Any]]
    raw: Dict[str, Any]


# ─── MCP Transport ──────────────────────────────────────────────────


class _McpTransport:
    """
    Low-level transport for calling the Prysm MCP endpoint.

    Handles:
    - JSON-RPC request construction
    - SSE response parsing
    - Authentication via Bearer token
    - Request ID auto-increment
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._request_id = 0
        self._client = httpx.Client(timeout=timeout)

    @property
    def mcp_url(self) -> str:
        """Derive the MCP endpoint URL from the proxy base URL."""
        # The base_url is typically https://prysmai.io/api/v1 (the proxy endpoint)
        # The MCP endpoint is at /api/mcp on the same host
        # Strip /api/v1 or similar suffix and append /api/mcp
        url = self._base_url
        for suffix in ["/api/v1", "/v1", "/api"]:
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        return f"{url}/api/mcp"

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool and return the parsed result.

        Args:
            tool_name: Name of the MCP tool (e.g., "prysm_session_start")
            arguments: Tool arguments as a dictionary

        Returns:
            Parsed tool result as a dictionary

        Raises:
            GovernanceError: If the call fails
        """
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        try:
            response = self._client.post(
                self.mcp_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GovernanceError(
                f"MCP request failed with status {e.response.status_code}: {e.response.text[:500]}"
            )
        except httpx.RequestError as e:
            raise GovernanceError(f"MCP request failed: {e}")

        mcp_result = _parse_sse_response(response.text)
        tool_result = _extract_tool_result(mcp_result)

        # Check for tool-level errors
        if mcp_result.get("isError"):
            error_msg = tool_result.get("error", "Unknown tool error")
            raise GovernanceError(f"Tool {tool_name} failed: {error_msg}")

        return tool_result

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/list",
            "params": {},
        }

        try:
            response = self._client.post(
                self.mcp_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GovernanceError(
                f"MCP request failed with status {e.response.status_code}: {e.response.text[:500]}"
            )
        except httpx.RequestError as e:
            raise GovernanceError(f"MCP request failed: {e}")

        result = _parse_sse_response(response.text)
        return result.get("tools", [])

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


# ─── GovernanceSession ──────────────────────────────────────────────


class GovernanceSession:
    """
    A governance session that monitors agent behavior through the Prysm MCP endpoint.

    Use as a context manager for automatic session lifecycle management:

        with GovernanceSession(client, task="...", agent_type="claude_code") as gov:
            gov.check_behavior([...])
            gov.scan_code(code="...", language="python")

    Or manage manually:

        gov = GovernanceSession(client, task="...", agent_type="claude_code")
        gov.start()
        gov.check_behavior([...])
        gov.end(outcome="completed")

    Args:
        client: A PrysmClient instance (provides auth and base_url).
            Alternatively, pass prysm_key and base_url directly.
        task: The task instructions given to the agent.
        agent_type: Type of agent ("claude_code", "codex", "langchain", "crewai", "custom", etc.)
        available_tools: List of tool names available to the agent.
        context: Additional context (repo name, branch, constraints, etc.)
        prysm_key: Prysm API key (alternative to passing a PrysmClient).
        base_url: Prysm base URL (alternative to passing a PrysmClient).
        timeout: HTTP timeout in seconds for MCP calls (default 60).
        auto_check_interval: If set, automatically call check_behavior every N events
            reported via report_event(). Default is None (manual only).
    """

    def __init__(
        self,
        client: Any = None,
        task: str = "",
        agent_type: str = "custom",
        available_tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        prysm_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        auto_check_interval: Optional[int] = None,
    ):
        # Resolve auth from client or direct params
        if client is not None:
            self._api_key = client.prysm_key
            self._base_url = client.base_url
        else:
            self._api_key = prysm_key or os.environ.get("PRYSM_API_KEY", "")
            self._base_url = base_url or os.environ.get(
                "PRYSM_BASE_URL", "https://prysmai.io/api/v1"
            )

        if not self._api_key:
            raise ValueError(
                "Prysm API key is required. Pass a PrysmClient, prysm_key=, or set PRYSM_API_KEY."
            )

        self._task = task
        self._agent_type = agent_type
        self._available_tools = available_tools
        self._context = context
        self._timeout = timeout
        self._auto_check_interval = auto_check_interval

        # State
        self._session_id: Optional[str] = None
        self._started_at: Optional[str] = None
        self._policies: List[Dict[str, Any]] = []
        self._active = False
        self._event_buffer: List[Dict[str, Any]] = []
        self._total_events_reported = 0
        self._checks_performed = 0
        self._scans_performed = 0
        self._flags: List[BehavioralFlag] = []
        self._detectors: List[Any] = []  # BaseDetector instances
        self._detections: List[Any] = []  # Detection instances from detectors

        # Transport
        self._transport = _McpTransport(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=timeout,
        )

    # ─── Detectors ──────────────────────────────────────────────

    def attach_detector(self, detector: Any) -> "GovernanceSession":
        """
        Attach a behavioral detector to this session.

        Detectors process events locally (client-side) before events are
        sent to the server. They can detect financial anomalies, resource
        access violations, loops, and multi-agent coordination issues.

        Args:
            detector: A BaseDetector instance (e.g., FinancialAnomalyDetector).

        Returns:
            self (for chaining).
        """
        self._detectors.append(detector)
        logger.info(f"Attached detector: {getattr(detector, 'name', type(detector).__name__)}")
        return self

    def detach_detector(self, detector_name: str) -> bool:
        """
        Detach a detector by name.

        Returns:
            True if a detector was removed, False if not found.
        """
        for i, d in enumerate(self._detectors):
            if getattr(d, 'name', '') == detector_name:
                self._detectors.pop(i)
                return True
        return False

    @property
    def detections(self) -> List[Any]:
        """All detections raised by attached detectors."""
        return self._detections

    @property
    def detector_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries from all attached detectors."""
        return [d.get_summary() for d in self._detectors]

    def _run_detectors(self, events: List[Dict[str, Any]]) -> List[Any]:
        """Run all attached detectors against events, return detections."""
        all_detections = []
        for event in events:
            for detector in self._detectors:
                try:
                    detections = detector.process_event(event)
                    all_detections.extend(detections)
                    self._detections.extend(detections)
                except Exception as e:
                    logger.warning(f"Detector {getattr(detector, 'name', '?')} error: {e}")
        return all_detections

    # ─── Properties ─────────────────────────────────────────────

    @property
    def session_id(self) -> Optional[str]:
        """The governance session ID (set after start())."""
        return self._session_id

    @property
    def is_active(self) -> bool:
        """Whether the session is currently active."""
        return self._active

    @property
    def policies(self) -> List[Dict[str, Any]]:
        """Active governance policies returned at session start."""
        return self._policies

    @property
    def all_flags(self) -> List[BehavioralFlag]:
        """All behavioral flags detected across all checks in this session."""
        return self._flags

    @property
    def stats(self) -> Dict[str, Any]:
        """Session statistics."""
        return {
            "session_id": self._session_id,
            "active": self._active,
            "total_events_reported": self._total_events_reported,
            "checks_performed": self._checks_performed,
            "scans_performed": self._scans_performed,
            "flags_detected": len(self._flags),
        }

    # ─── Lifecycle ──────────────────────────────────────────────

    def start(self) -> Dict[str, Any]:
        """
        Start the governance session.

        Returns:
            Dict with session_id, started_at, policies, and message.

        Raises:
            GovernanceError: If the session fails to start.
        """
        if self._active:
            raise GovernanceError("Session is already active. Call end() first.")

        args: Dict[str, Any] = {
            "task_instructions": self._task,
            "agent_type": self._agent_type,
        }
        if self._available_tools:
            args["available_tools"] = self._available_tools
        if self._context:
            args["context"] = self._context

        result = self._transport.call_tool("prysm_session_start", args)

        self._session_id = result.get("session_id")
        self._started_at = result.get("started_at")
        self._policies = result.get("policies", [])
        self._active = True

        logger.info(
            f"Governance session started: {self._session_id} "
            f"({len(self._policies)} active policies)"
        )

        return result

    def end(self, outcome: str = "completed", output_summary: Optional[str] = None,
            files_modified: Optional[List[str]] = None) -> SessionReport:
        """
        End the governance session and get the final report.

        Args:
            outcome: How the task ended ("completed", "failed", "partial", "timeout").
            output_summary: Brief summary of what the agent produced.
            files_modified: List of file paths created or modified.

        Returns:
            SessionReport with behavior score, detectors, and violations.

        Raises:
            GovernanceError: If the session fails to end.
        """
        if not self._active:
            raise GovernanceError("No active session to end.")

        # Flush any buffered events first
        if self._event_buffer:
            try:
                self.check_behavior(self._event_buffer)
            except GovernanceError:
                pass  # Best effort flush
            self._event_buffer.clear()

        args: Dict[str, Any] = {
            "session_id": self._session_id,
            "outcome": outcome,
        }
        if output_summary:
            args["output_summary"] = output_summary
        if files_modified:
            args["files_modified"] = files_modified

        result = self._transport.call_tool("prysm_session_end", args)
        self._active = False

        report_data = result.get("report", {})

        # Merge local detector summaries into the report
        server_detectors = report_data.get("detectors", [])
        local_detector_summaries = self.detector_summaries
        all_detectors = server_detectors + local_detector_summaries

        # Merge local violations into the report
        server_violations = result.get("violations", [])
        local_violations = [
            {
                "detector": d.detector,
                "category": d.category,
                "message": d.message,
                "severity": d.severity,
                "evidence": d.evidence,
            }
            for d in self._detections
            if d.severity in ("critical", "halt")
        ]
        all_violations = server_violations + local_violations

        report = SessionReport(
            session_id=result.get("session_id", self._session_id or ""),
            outcome=result.get("outcome", outcome),
            behavior_score=report_data.get("behavior_score"),
            detectors=all_detectors,
            summary=report_data.get("summary"),
            recommendations=report_data.get("recommendations", []),
            violations=all_violations,
            raw=result,
        )

        logger.info(
            f"Governance session ended: {self._session_id} "
            f"outcome={outcome} score={report.behavior_score}"
        )

        return report

    # ─── Operations ─────────────────────────────────────────────

    def check_behavior(self, events: List[Dict[str, Any]]) -> CheckResult:
        """
        Report events and receive behavioral feedback.

        Args:
            events: List of event dicts. Each must have "event_type" and "data".
                Supported event_type values:
                    llm_call, tool_call, tool_result, code_generated,
                    code_executed, file_read, file_write, decision,
                    error, delegation

        Returns:
            CheckResult with flags, violations, and recommendations.

        Raises:
            GovernanceError: If the check fails.
            SessionNotActiveError: If the session is not active.
        """
        if not self._active:
            raise SessionNotActiveError("Cannot check behavior: session is not active.")

        # Ensure events have required fields
        normalized_events = []
        for event in events:
            normalized = {
                "event_type": event.get("event_type", "unknown"),
                "data": event.get("data", {}),
            }
            if "timestamp" in event:
                normalized["timestamp"] = event["timestamp"]
            normalized_events.append(normalized)

        # Run local detectors first (client-side)
        local_detections = self._run_detectors(normalized_events)

        # Check for halt-level detections — include in result even before server call
        halt_detections = [d for d in local_detections if d.severity == "halt"]

        result = self._transport.call_tool("prysm_check_behavior", {
            "session_id": self._session_id,
            "events": normalized_events,
        })

        self._total_events_reported += len(normalized_events)
        self._checks_performed += 1

        # Parse server flags
        flags = []
        for f in result.get("flags", []):
            flag = BehavioralFlag(
                detector=f.get("detector", "unknown"),
                severity=f.get("severity", 0),
                evidence=f.get("evidence", []),
            )
            flags.append(flag)
            self._flags.append(flag)

        # Merge local detector detections as additional flags
        for detection in local_detections:
            severity_map = {"info": 1, "warning": 3, "critical": 5, "halt": 7}
            flag = BehavioralFlag(
                detector=detection.detector,
                severity=severity_map.get(detection.severity, 3),
                evidence=[detection.message, detection.evidence],
            )
            flags.append(flag)
            self._flags.append(flag)

        # Merge local violations
        violations = result.get("violations", [])
        for detection in local_detections:
            if detection.severity in ("critical", "halt"):
                violations.append({
                    "detector": detection.detector,
                    "category": detection.category,
                    "message": detection.message,
                    "severity": detection.severity,
                    "evidence": detection.evidence,
                })

        check_result = CheckResult(
            events_ingested=result.get("events_ingested", 0),
            flags=flags,
            violations=violations,
            recommendations=result.get("recommendations", []),
            raw=result,
        )

        if check_result.has_flags:
            logger.warning(
                f"Behavioral flags detected: {[f.detector for f in flags]} "
                f"(max severity: {check_result.max_severity})"
            )

        return check_result

    def scan_code(
        self,
        code: str,
        language: str,
        file_path: Optional[str] = None,
    ) -> ScanResult:
        """
        Scan code for security vulnerabilities, PII, and policy violations.

        Args:
            code: The source code to scan.
            language: Programming language ("python", "typescript", "javascript",
                "sql", "bash", "go", "rust", "java", "other").
            file_path: Target file path for context.

        Returns:
            ScanResult with vulnerabilities, threat score, and recommendations.

        Raises:
            GovernanceError: If the scan fails.
            SessionNotActiveError: If the session is not active.
        """
        if not self._active:
            raise SessionNotActiveError("Cannot scan code: session is not active.")

        args: Dict[str, Any] = {
            "session_id": self._session_id,
            "code": code,
            "language": language,
        }
        if file_path:
            args["file_path"] = file_path

        result = self._transport.call_tool("prysm_scan_code", args)
        self._scans_performed += 1

        vulnerabilities = [
            Vulnerability(
                type=v.get("type", "unknown"),
                severity=v.get("severity", "info"),
                description=v.get("description", ""),
            )
            for v in result.get("vulnerabilities", [])
        ]

        scan_result = ScanResult(
            language=result.get("language", language),
            file_path=result.get("file_path", file_path),
            vulnerability_count=result.get("vulnerability_count", 0),
            max_severity=result.get("max_severity", "info"),
            threat_score=result.get("threat_score", 0),
            vulnerabilities=vulnerabilities,
            recommendations=result.get("recommendations", []),
            raw=result,
        )

        if not scan_result.is_clean:
            logger.warning(
                f"Code scan found {scan_result.vulnerability_count} vulnerabilities "
                f"(max severity: {scan_result.max_severity}, threat score: {scan_result.threat_score})"
            )

        return scan_result

    def report_event(self, event_type: str, data: Dict[str, Any],
                     timestamp: Optional[float] = None) -> Optional[CheckResult]:
        """
        Buffer a single event. If auto_check_interval is set and the buffer
        reaches that size, automatically calls check_behavior.

        Args:
            event_type: Type of event (llm_call, tool_call, etc.)
            data: Event data dictionary
            timestamp: Optional Unix timestamp (defaults to now)

        Returns:
            CheckResult if auto-check was triggered, None otherwise.
        """
        event: Dict[str, Any] = {
            "event_type": event_type,
            "data": data,
        }
        if timestamp:
            event["timestamp"] = timestamp
        else:
            event["timestamp"] = time.time()

        self._event_buffer.append(event)

        if (
            self._auto_check_interval
            and len(self._event_buffer) >= self._auto_check_interval
        ):
            events = self._event_buffer[:]
            self._event_buffer.clear()
            return self.check_behavior(events)

        return None

    # ─── Context Manager ────────────────────────────────────────

    def __enter__(self) -> "GovernanceSession":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._active:
            outcome = "completed" if exc_type is None else "failed"
            error_summary = None
            if exc_val is not None:
                error_summary = f"{type(exc_val).__name__}: {exc_val}"
            try:
                self.end(outcome=outcome, output_summary=error_summary)
            except GovernanceError as e:
                logger.warning(f"Failed to end governance session: {e}")

    # ─── Cleanup ────────────────────────────────────────────────

    def close(self) -> None:
        """Close the transport. Call this if not using as a context manager."""
        self._transport.close()

    def __del__(self) -> None:
        try:
            self._transport.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return (
            f"GovernanceSession(session_id={self._session_id!r}, "
            f"status={status}, agent_type={self._agent_type!r})"
        )
