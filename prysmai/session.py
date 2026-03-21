"""
Prysm AI — unified session scope across proxy and MCP surfaces.

This module links proxy request context and governance session context so a
single logical run can be correlated across the Prysm control plane.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prysmai.context import prysm_context
from prysmai.governance import CheckResult, GovernanceSession, ScanResult, SessionReport


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
        """Create a proxied OpenAI client scoped to this session."""
        return self._client.openai(**kwargs)

    def async_openai(self, **kwargs: Any) -> Any:
        """Create an async proxied OpenAI client scoped to this session."""
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
            data=data,
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
