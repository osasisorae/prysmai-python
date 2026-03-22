"""
Prysm AI — MCP integration surface.

This module makes MCP a first-class public SDK surface rather than leaving it
implicit behind GovernanceSession.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from prysmai.config import resolve_prysm_connection
from prysmai.governance import GovernanceSession, _McpTransport


@dataclass(frozen=True)
class PrysmMCPConfig:
    """Reusable MCP connection config for agent runtimes and tool clients."""

    server_url: str
    headers: Dict[str, str]
    transport: str = "streamable_http"


class PrysmMCPClient:
    """
    Public MCP-facing client for the Prysm control plane.

    This client shares the same auth and base URL concepts as PrysmClient, but
    exposes the MCP surface directly:

    - derive the MCP server URL
    - return MCP connection config
    - list and call MCP tools
    - create GovernanceSession instances
    """

    def __init__(
        self,
        client: Any = None,
        prysm_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        resolved = resolve_prysm_connection(
            client=client,
            prysm_key=prysm_key,
            base_url=base_url,
        )
        self.prysm_key = resolved.prysm_key
        self.base_url = resolved.base_url

        self.timeout = timeout
        self._transport = _McpTransport(
            base_url=self.base_url,
            api_key=self.prysm_key,
            timeout=timeout,
        )

    @property
    def mcp_url(self) -> str:
        """Resolved MCP server URL for the configured Prysm deployment."""
        return self._transport.mcp_url

    @property
    def auth_headers(self) -> Dict[str, str]:
        """Bearer auth header for MCP-compatible clients."""
        return {"Authorization": f"Bearer {self.prysm_key}"}

    def connection_config(
        self,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> PrysmMCPConfig:
        """
        Return reusable MCP connection config for agent runtimes.

        Example shape:
            {
                "server_url": ".../api/mcp",
                "headers": {"Authorization": "Bearer sk-prysm-..."},
                "transport": "streamable_http",
            }
        """
        headers = {**self.auth_headers, **(extra_headers or {})}
        return PrysmMCPConfig(server_url=self.mcp_url, headers=headers)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available Prysm MCP tools."""
        return self._transport.list_tools()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool directly."""
        return self._transport.call_tool(tool_name, arguments)

    def record_llm_call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        session_id: Optional[str] = None,
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
        """Record an externally executed model call into the Prysm control plane."""
        arguments: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "status": status,
            "run_security_scan": run_security_scan,
        }
        if session_id is not None:
            arguments["session_id"] = session_id
        if provider is not None:
            arguments["provider"] = provider
        if completion is not None:
            arguments["completion"] = completion
        if finish_reason is not None:
            arguments["finish_reason"] = finish_reason
        if status_code is not None:
            arguments["status_code"] = status_code
        if error_message is not None:
            arguments["error_message"] = error_message
        if latency_ms is not None:
            arguments["latency_ms"] = latency_ms
        if prompt_tokens is not None:
            arguments["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            arguments["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            arguments["total_tokens"] = total_tokens
        if cost_usd is not None:
            arguments["cost_usd"] = cost_usd
        if temperature is not None:
            arguments["temperature"] = temperature
        if max_tokens is not None:
            arguments["max_tokens"] = max_tokens
        if top_p is not None:
            arguments["top_p"] = top_p
        if metadata is not None:
            arguments["metadata"] = metadata
        if logprobs is not None:
            arguments["logprobs"] = logprobs
        return self.call_tool("prysm_record_llm_call", arguments)

    def record_tool_call(
        self,
        *,
        session_id: str,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
        duration_ms: Optional[int] = None,
        external_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record external tool activity inside an active Prysm session."""
        arguments: Dict[str, Any] = {
            "session_id": session_id,
            "tool_name": tool_name,
        }
        if tool_input is not None:
            arguments["input"] = tool_input
        if tool_output is not None:
            arguments["output"] = tool_output
        if success is not None:
            arguments["success"] = success
        if duration_ms is not None:
            arguments["duration_ms"] = duration_ms
        if external_trace_id is not None:
            arguments["external_trace_id"] = external_trace_id
        if metadata is not None:
            arguments["metadata"] = metadata
        return self.call_tool("prysm_record_tool_call", arguments)

    def record_decision(
        self,
        *,
        session_id: str,
        description: str,
        rationale: Optional[str] = None,
        selected_action: Optional[str] = None,
        severity: Optional[str] = None,
        external_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a reviewable agent decision inside an active Prysm session."""
        arguments: Dict[str, Any] = {
            "session_id": session_id,
            "description": description,
        }
        if rationale is not None:
            arguments["rationale"] = rationale
        if selected_action is not None:
            arguments["selected_action"] = selected_action
        if severity is not None:
            arguments["severity"] = severity
        if external_trace_id is not None:
            arguments["external_trace_id"] = external_trace_id
        if metadata is not None:
            arguments["metadata"] = metadata
        return self.call_tool("prysm_record_decision", arguments)

    def record_file_change(
        self,
        *,
        session_id: str,
        operation: str,
        path: str,
        language: Optional[str] = None,
        content: Optional[str] = None,
        external_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a file read or write inside an active Prysm session."""
        arguments: Dict[str, Any] = {
            "session_id": session_id,
            "operation": operation,
            "path": path,
        }
        if language is not None:
            arguments["language"] = language
        if content is not None:
            arguments["content"] = content
        if external_trace_id is not None:
            arguments["external_trace_id"] = external_trace_id
        if metadata is not None:
            arguments["metadata"] = metadata
        return self.call_tool("prysm_record_file_change", arguments)

    def governance_session(
        self,
        task: str = "",
        agent_type: str = "custom",
        available_tools: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        auto_check_interval: Optional[int] = None,
    ) -> GovernanceSession:
        """Create a GovernanceSession using this MCP client as the auth source."""
        return GovernanceSession(
            client=self,
            task=task,
            agent_type=agent_type,
            available_tools=available_tools,
            context=context,
            timeout=self.timeout,
            auto_check_interval=auto_check_interval,
        )

    def close(self) -> None:
        """Close the underlying MCP transport."""
        self._transport.close()
