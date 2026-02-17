"""
Prysm context management for request-level metadata tagging.

Usage:
    from prysmai import prysm_context

    # Set metadata for all subsequent requests in this context
    with prysm_context(user_id="user_123", session_id="sess_abc"):
        response = monitored.chat.completions.create(...)

    # Or set globally
    prysm_context.set(user_id="user_123")
"""

from __future__ import annotations

import json
import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PrysmContext:
    """Holds metadata that gets attached to every proxied request."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_headers(self) -> Dict[str, str]:
        """Convert context to Prysm custom headers."""
        headers: Dict[str, str] = {}
        if self.user_id:
            headers["X-Prysm-User-Id"] = self.user_id
        if self.session_id:
            headers["X-Prysm-Session-Id"] = self.session_id
        if self.metadata:
            headers["X-Prysm-Metadata"] = json.dumps(self.metadata)
        return headers


# Context variable for async-safe per-request metadata
_prysm_ctx: contextvars.ContextVar[PrysmContext] = contextvars.ContextVar(
    "prysm_context", default=PrysmContext()
)


class _ContextManager:
    """
    Dual-purpose context manager:
    - Use as a context manager: `with prysm_context(user_id="x"): ...`
    - Use imperatively: `prysm_context.set(user_id="x")`
    """

    def set(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set global context metadata for all subsequent requests."""
        ctx = _prysm_ctx.get()
        if user_id is not None:
            ctx.user_id = user_id
        if session_id is not None:
            ctx.session_id = session_id
        if metadata is not None:
            ctx.metadata = metadata

    def get(self) -> PrysmContext:
        """Get the current context."""
        return _prysm_ctx.get()

    def clear(self) -> None:
        """Reset context to defaults."""
        _prysm_ctx.set(PrysmContext())

    def __call__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "_ContextScope":
        """Use as a context manager for scoped metadata."""
        return _ContextScope(user_id=user_id, session_id=session_id, metadata=metadata)


class _ContextScope:
    """Scoped context manager that restores previous context on exit."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._user_id = user_id
        self._session_id = session_id
        self._metadata = metadata
        self._token: Optional[contextvars.Token[PrysmContext]] = None

    def __enter__(self) -> PrysmContext:
        old = _prysm_ctx.get()
        new_ctx = PrysmContext(
            user_id=self._user_id or old.user_id,
            session_id=self._session_id or old.session_id,
            metadata={**old.metadata, **(self._metadata or {})},
        )
        self._token = _prysm_ctx.set(new_ctx)
        return new_ctx

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _prysm_ctx.reset(self._token)


# Singleton instance
prysm_context = _ContextManager()
