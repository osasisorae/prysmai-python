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
