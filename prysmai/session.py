"""
Prysm AI — unified session scope across proxy and MCP surfaces.

This module links proxy request context and governance session context so a
single logical run can be correlated across the Prysm control plane.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, TypeVar

from prysmai.context import prysm_context
from prysmai.governance import CheckResult, GovernanceSession, ScanResult, SessionReport

T = TypeVar("T")


@dataclass(frozen=True)
class PrysmSessionIdentifiers:
    """Correlation identifiers for a unified Prysm session."""

    session_id: str
    user_id: Optional[str] = None
    governance_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrysmSession:
    """
    Unified session scope across proxy traffic and MCP/governance activity.

    Typical usage:

        prysm = PrysmClient(prysm_key="sk-prysm-...")

        with prysm.session(
            user_id="user_123",
            metadata={"feature": "support"},
            governance_task="Resolve support request",
            agent_type="codex",
        ) as run:
            client = run.openai()
            response = client.chat.completions.create(...)
            run.report_event("tool_call", {"tool_name": "search"})

    Inside the scope:
    - proxy requests inherit session/user/governance identifiers through prysm_context
    - governance events and scans can be reported through the same run object
    """

    def __init__(
        self,
        client: Any,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        governance_task: Optional[str] = None,
        agent_type: str = "custom",
        available_tools: Optional[List[str]] = None,
        governance_context: Optional[Dict[str, Any]] = None,
        auto_check_interval: Optional[int] = None,
        governance_timeout: float = 60.0,
    ):
        self._client = client
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.governance_task = governance_task
        self.agent_type = agent_type
        self.available_tools = available_tools
        self.governance_context = governance_context
        self.auto_check_interval = auto_check_interval
        self.governance_timeout = governance_timeout

        self._context_scope: Any = None
        self._governance: Optional[GovernanceSession] = None
        self._governance_report: Optional[SessionReport] = None
        self._active = False

    @property
    def identifiers(self) -> PrysmSessionIdentifiers:
        """Correlation identifiers for this run."""
        return PrysmSessionIdentifiers(
            session_id=self.session_id,
            user_id=self.user_id,
            governance_session_id=self.governance_session_id,
            metadata=self.metadata,
        )

    @property
    def governance(self) -> Optional[GovernanceSession]:
        """Underlying governance session, if enabled."""
        return self._governance

    @property
    def governance_session_id(self) -> Optional[str]:
        """Governance session identifier, if governance is enabled."""
        if self._governance is None:
            return None
        return self._governance.session_id

    @property
    def governance_report(self) -> Optional[SessionReport]:
        """Final governance report after the session exits or closes."""
        return self._governance_report

    @property
    def is_active(self) -> bool:
        return self._active

    def openai(self, **kwargs: Any) -> Any:
        """Backward-compatible alias for llm() within this session."""
        return self._client.openai(**kwargs)

    def llm(self, **kwargs: Any) -> Any:
        """Create a proxied OpenAI-compatible client scoped to this session."""
        if hasattr(self._client, "llm"):
            return self._client.llm(**kwargs)
        return self._client.openai(**kwargs)

    def async_openai(self, **kwargs: Any) -> Any:
        """Backward-compatible alias for async_llm() within this session."""
        return self._client.async_openai(**kwargs)

    def async_llm(self, **kwargs: Any) -> Any:
        """Create an async proxied OpenAI-compatible client scoped to this session."""
        if hasattr(self._client, "async_llm"):
            return self._client.async_llm(**kwargs)
        return self._client.async_openai(**kwargs)

    def mcp(self, timeout: Optional[float] = None) -> Any:
        """Create an MCP client for the same Prysm deployment."""
        if hasattr(self._client, "mcp"):
            return self._client.mcp(timeout=timeout or self.governance_timeout)
        return self._client

    def check_behavior(self, events: List[Dict[str, Any]]) -> CheckResult:
        """Delegate behavior checks to the governance session."""
        if self._governance is None:
            raise RuntimeError("Governance is not enabled for this Prysm session.")
        return self._governance.check_behavior(events)

    def _require_governance_session_id(self) -> str:
        if self._governance is None or self.governance_session_id is None:
            raise RuntimeError(
                "Governance is not enabled for this Prysm session."
            )
        return self.governance_session_id

    @staticmethod
    def _coerce_payload(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, (str, int, float, bool)):
            return {"value": value}
        if isinstance(value, list):
            return {"items": value}
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "model_dump") and callable(value.model_dump):
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {"value": dumped}
        if hasattr(value, "dict") and callable(value.dict):
            dumped = value.dict()
            return dumped if isinstance(dumped, dict) else {"value": dumped}
        return {"repr": repr(value)}

    def _session_event_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        governance_context = self.governance_context or {}

        if isinstance(governance_context, dict):
            for key in (
                "agent_id",
                "delegated_authority",
                "approved_by",
                "approval_id",
                "human_approved",
            ):
                value = governance_context.get(key)
                if value is not None:
                    defaults[key] = value

            parent_session_id = governance_context.get("parent_session_id")
            if parent_session_id is not None:
                defaults["parent_session_id"] = parent_session_id

        if "agent_id" not in defaults:
            defaults["agent_id"] = self.agent_type

        if "parent_session_id" not in defaults:
            defaults["parent_session_id"] = (
                self.governance_session_id or self.session_id
            )

        if (
            "human_approved" not in defaults
            and (
                defaults.get("approved_by") is not None
                or defaults.get("approval_id") is not None
            )
        ):
            defaults["human_approved"] = True

        return defaults

    def _merge_event_metadata(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        merged = self._session_event_defaults()
        if metadata:
            merged.update(metadata)
        return merged

    def _normalize_report_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(data)

        if event_type in {
            "llm_call",
            "tool_call",
            "decision",
            "file_read",
            "file_write",
            "delegation",
        }:
            existing_metadata = payload.get("metadata")
            if isinstance(existing_metadata, dict):
                payload["metadata"] = self._merge_event_metadata(existing_metadata)
            else:
                payload["metadata"] = self._merge_event_metadata()

        if event_type == "delegation":
            from_agent = payload.get("from_agent") or payload.get("sender")
            if from_agent is None:
                from_agent = self._session_event_defaults().get("agent_id")
            if from_agent is not None:
                payload["from_agent"] = from_agent

            to_agent = (
                payload.get("to_agent")
                or payload.get("target")
                or payload.get("agent_id")
            )
            if to_agent is not None:
                payload["to_agent"] = to_agent

            if payload.get("parent_session_id") is None:
                payload["parent_session_id"] = (
                    self.governance_session_id or self.session_id
                )

            if payload.get("delegation_chain") is None and to_agent is not None:
                payload["delegation_chain"] = [
                    str(from_agent),
                    str(to_agent),
                ]

        return payload

    def record_llm_call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        provider: Optional[str] = None,
        completion: Optional[str] = None,
        finish_reason: Optional[str] = None,
        status: str = "success",
        status_code: Optional[int] = None,
        error_message: Optional[str] = None,
        latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        logprobs: Optional[Dict[str, Any]] = None,
        run_security_scan: bool = True,
    ) -> Dict[str, Any]:
        """Record an external model interaction under the active Prysm governance session."""
        session_id = self._require_governance_session_id()
        mcp = self.mcp(timeout=self.governance_timeout)
        return mcp.record_llm_call(
            session_id=session_id,
            model=model,
            messages=messages,
            provider=provider,
            completion=completion,
            finish_reason=finish_reason,
            status=status,
            status_code=status_code,
            error_message=error_message,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            metadata=self._merge_event_metadata(metadata),
            logprobs=logprobs,
            run_security_scan=run_security_scan,
        )

    def record_tool_call(
        self,
        *,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
        duration_ms: Optional[int] = None,
        external_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a tool call under the active Prysm governance session."""
        session_id = self._require_governance_session_id()
        mcp = self.mcp(timeout=self.governance_timeout)
        return mcp.record_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            success=success,
            duration_ms=duration_ms,
            external_trace_id=external_trace_id,
            metadata=self._merge_event_metadata(metadata),
        )

    def record_decision(
        self,
        *,
        description: str,
        rationale: Optional[str] = None,
        selected_action: Optional[str] = None,
        severity: Optional[str] = None,
        external_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a reviewable decision under the active Prysm governance session."""
        session_id = self._require_governance_session_id()
        mcp = self.mcp(timeout=self.governance_timeout)
        return mcp.record_decision(
            session_id=session_id,
            description=description,
            rationale=rationale,
            selected_action=selected_action,
            severity=severity,
            external_trace_id=external_trace_id,
            metadata=self._merge_event_metadata(metadata),
        )

    def record_file_change(
        self,
        *,
        operation: str,
        path: str,
        language: Optional[str] = None,
        content: Optional[str] = None,
        external_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a file read or write under the active Prysm governance session."""
        session_id = self._require_governance_session_id()
        mcp = self.mcp(timeout=self.governance_timeout)
        return mcp.record_file_change(
            session_id=session_id,
            operation=operation,
            path=path,
            language=language,
            content=content,
            external_trace_id=external_trace_id,
            metadata=self._merge_event_metadata(metadata),
        )

    def record_delegation(
        self,
        *,
        to_agent: str,
        from_agent: Optional[str] = None,
        delegated_authority: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        delegation_chain: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[CheckResult]:
        """Record a delegation event with canonical lineage metadata."""
        payload: Dict[str, Any] = {
            "to_agent": to_agent,
        }
        if from_agent is not None:
            payload["from_agent"] = from_agent
        if delegated_authority is not None:
            payload["delegated_authority"] = delegated_authority
        if parent_session_id is not None:
            payload["parent_session_id"] = parent_session_id
        if delegation_chain is not None:
            payload["delegation_chain"] = delegation_chain
        if metadata:
            payload["metadata"] = metadata
        return self.report_event("delegation", payload, timestamp=timestamp)

    def run_tool(
        self,
        tool_name: str,
        func: Callable[..., T],
        *args: Any,
        tool_input: Optional[Dict[str, Any]] = None,
        external_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute a tool callable and automatically record the result in Prysm.

        This is the session-level automation helper for MCP parity. It executes
        the tool, captures duration, serializes the output, and records the
        corresponding `prysm_record_tool_call` event.
        """
        started = perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            duration_ms = int((perf_counter() - started) * 1000)
            self.record_tool_call(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output={
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                },
                success=False,
                duration_ms=duration_ms,
                external_trace_id=external_trace_id,
                metadata=metadata,
            )
            raise

        duration_ms = int((perf_counter() - started) * 1000)
        self.record_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=self._coerce_payload(result),
            success=True,
            duration_ms=duration_ms,
            external_trace_id=external_trace_id,
            metadata=metadata,
        )
        return result

    def report_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> Optional[CheckResult]:
        """Buffer or report a governance event within this unified session."""
        if self._governance is None:
            raise RuntimeError("Governance is not enabled for this Prysm session.")
        return self._governance.report_event(
            event_type=event_type,
            data=self._normalize_report_event(event_type, data),
            timestamp=timestamp,
        )

    def scan_code(
        self,
        code: str,
        language: str,
        file_path: Optional[str] = None,
    ) -> ScanResult:
        """Scan generated code within this unified session."""
        if self._governance is None:
            raise RuntimeError("Governance is not enabled for this Prysm session.")
        return self._governance.scan_code(
            code=code,
            language=language,
            file_path=file_path,
        )

    def __enter__(self) -> "PrysmSession":
        governance_session_id = None

        if self.governance_task:
            self._governance = GovernanceSession(
                client=self._client,
                task=self.governance_task,
                agent_type=self.agent_type,
                available_tools=self.available_tools,
                context=self.governance_context,
                timeout=self.governance_timeout,
                auto_check_interval=self.auto_check_interval,
            )
            self._governance.start()
            governance_session_id = self._governance.session_id

        self._context_scope = prysm_context(
            user_id=self.user_id,
            session_id=self.session_id,
            governance_session_id=governance_session_id,
            metadata=self.metadata,
        )
        self._context_scope.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._context_scope is not None:
            self._context_scope.__exit__(exc_type, exc_val, exc_tb)
            self._context_scope = None

        if self._governance is not None and self._governance.is_active:
            outcome = "completed" if exc_type is None else "failed"
            error_summary = None
            if exc_val is not None:
                error_summary = f"{type(exc_val).__name__}: {exc_val}"
            try:
                self._governance_report = self._governance.end(
                    outcome=outcome,
                    output_summary=error_summary,
                )
            finally:
                self._governance.close()

        self._active = False

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return (
            f"PrysmSession(session_id={self.session_id!r}, "
            f"governance_session_id={self.governance_session_id!r}, "
            f"status={status})"
        )
