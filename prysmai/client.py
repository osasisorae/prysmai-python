"""
Core Prysm client — wraps the OpenAI client to route through the Prysm proxy.

The monitor() function is the primary entry point. It takes an existing OpenAI
client and returns a new one that routes all traffic through Prysm's proxy,
capturing metrics, costs, and full request/response data.

Architecture:
    - We create a new OpenAI client pointing at the Prysm proxy base_url
    - The Prysm API key (sk-prysm-*) is used for authentication
    - The proxy handles upstream routing using the project's stored provider config
    - Custom httpx transport injects Prysm context headers (user_id, session_id, metadata)
    - Optional: dynamic upstream API key and custom header forwarding for CI/CD integrations
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
import openai

from prysmai.config import resolve_prysm_connection
from prysmai.context import _prysm_ctx


# ─── Custom transport that injects Prysm headers ───


class _PrysmTransport(httpx.BaseTransport):
    """
    Wraps an httpx transport to inject Prysm context headers
    (X-Prysm-User-Id, X-Prysm-Session-Id, X-Prysm-Metadata)
    and optional dynamic upstream key / forward headers
    into every outgoing request.
    """

    def __init__(
        self,
        wrapped: httpx.BaseTransport,
        upstream_api_key: Optional[str] = None,
        forward_headers: Optional[Dict[str, str]] = None,
    ):
        self._wrapped = wrapped
        self._upstream_api_key = upstream_api_key
        self._forward_headers = forward_headers

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        ctx = _prysm_ctx.get()
        headers = ctx.to_headers()
        for key, value in headers.items():
            request.headers[key] = value

        # Inject dynamic upstream key
        if self._upstream_api_key:
            request.headers["X-Prysm-Upstream-Key"] = self._upstream_api_key

        # Inject forward headers as JSON
        if self._forward_headers:
            request.headers["X-Prysm-Forward-Headers"] = json.dumps(self._forward_headers)

        return self._wrapped.handle_request(request)


class _PrysmAsyncTransport(httpx.AsyncBaseTransport):
    """Async version of the Prysm header-injecting transport."""

    def __init__(
        self,
        wrapped: httpx.AsyncBaseTransport,
        upstream_api_key: Optional[str] = None,
        forward_headers: Optional[Dict[str, str]] = None,
    ):
        self._wrapped = wrapped
        self._upstream_api_key = upstream_api_key
        self._forward_headers = forward_headers

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        ctx = _prysm_ctx.get()
        headers = ctx.to_headers()
        for key, value in headers.items():
            request.headers[key] = value

        # Inject dynamic upstream key
        if self._upstream_api_key:
            request.headers["X-Prysm-Upstream-Key"] = self._upstream_api_key

        # Inject forward headers as JSON
        if self._forward_headers:
            request.headers["X-Prysm-Forward-Headers"] = json.dumps(self._forward_headers)

        return await self._wrapped.handle_async_request(request)


# ─── PrysmClient: the monitored OpenAI client ───


class PrysmClient:
    """
    A thin configuration object that produces monitored OpenAI clients.

    Usage:
        prysm = PrysmClient(prysm_key="sk-prysm-...", base_url="https://prysmai.io/api/v1")
        client = prysm.openai()
        response = client.chat.completions.create(...)

    For CI/CD integrations (e.g., GitLab AI Gateway):
        prysm = PrysmClient(
            prysm_key="sk-prysm-...",
            upstream_api_key=os.environ["AI_GATEWAY_TOKEN"],
            forward_headers={"X-Gitlab-Instance-Id": "..."},
        )
    """

    def __init__(
        self,
        prysm_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        upstream_api_key: Optional[str] = None,
        forward_headers: Optional[Dict[str, str]] = None,
    ):
        resolved = resolve_prysm_connection(
            prysm_key=prysm_key,
            base_url=base_url,
        )
        self.prysm_key = resolved.prysm_key
        self.base_url = resolved.base_url
        self.timeout = timeout
        self.upstream_api_key = upstream_api_key
        self.forward_headers = forward_headers

    def openai(self, **kwargs: Any) -> openai.OpenAI:
        """
        Create a sync OpenAI client routed through Prysm.

        Any extra kwargs are passed to openai.OpenAI().
        """
        transport = _PrysmTransport(
            httpx.HTTPTransport(retries=2),
            upstream_api_key=self.upstream_api_key,
            forward_headers=self.forward_headers,
        )

        http_client = httpx.Client(
            transport=transport,
            timeout=self.timeout,
        )

        return openai.OpenAI(
            api_key=self.prysm_key,
            base_url=self.base_url,
            http_client=http_client,
            **kwargs,
        )

    def async_openai(self, **kwargs: Any) -> openai.AsyncOpenAI:
        """
        Create an async OpenAI client routed through Prysm.

        Any extra kwargs are passed to openai.AsyncOpenAI().
        """
        transport = _PrysmAsyncTransport(
            httpx.AsyncHTTPTransport(retries=2),
            upstream_api_key=self.upstream_api_key,
            forward_headers=self.forward_headers,
        )

        http_client = httpx.AsyncClient(
            transport=transport,
            timeout=self.timeout,
        )

        return openai.AsyncOpenAI(
            api_key=self.prysm_key,
            base_url=self.base_url,
            http_client=http_client,
            **kwargs,
        )

    def mcp(self, timeout: float = 60.0) -> "PrysmMCPClient":
        """
        Create an MCP client for the same Prysm deployment.

        This exposes Prysm's MCP surface as a first-class SDK entry point for
        agent runtimes and tool-based integrations.
        """
        from prysmai.mcp import PrysmMCPClient

        return PrysmMCPClient(client=self, timeout=timeout)

    def session(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        governance_task: Optional[str] = None,
        agent_type: str = "custom",
        available_tools: Optional[list[str]] = None,
        governance_context: Optional[Dict[str, Any]] = None,
        auto_check_interval: Optional[int] = None,
        governance_timeout: float = 60.0,
    ) -> "PrysmSession":
        """
        Create a unified Prysm session linking proxy context and governance.

        This is the SDK entry point for correlated runs across the control
        plane. Proxy requests inside the scope inherit the same session and,
        when enabled, governance session identifiers.
        """
        from prysmai.session import PrysmSession

        return PrysmSession(
            self,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            governance_task=governance_task,
            agent_type=agent_type,
            available_tools=available_tools,
            governance_context=governance_context,
            auto_check_interval=auto_check_interval,
            governance_timeout=governance_timeout,
        )

    def langgraph_monitor(self, **kwargs: Any) -> Any:
        """Create a LangGraph monitor using this Prysm client's connection config."""
        from prysmai.integrations.langgraph import PrysmGraphMonitor

        return PrysmGraphMonitor(client=self, **kwargs)

    def crewai_monitor(self, **kwargs: Any) -> Any:
        """Create a CrewAI monitor using this Prysm client's connection config."""
        from prysmai.integrations.crewai import PrysmCrewMonitor

        return PrysmCrewMonitor(client=self, **kwargs)

    def agent_framework_monitor(self, **kwargs: Any) -> Any:
        """Create an Agent Framework monitor using this Prysm client's connection config."""
        from prysmai.integrations.agent_framework import PrysmAgentFrameworkMonitor

        return PrysmAgentFrameworkMonitor(client=self, **kwargs)

    def llamaindex_handler(self, **kwargs: Any) -> Any:
        """Create a LlamaIndex handler using this Prysm client's connection config."""
        from prysmai.integrations.llamaindex import PrysmSpanHandler

        return PrysmSpanHandler(client=self, **kwargs)


# ─── monitor(): the one-line integration ───


def monitor(
    client: openai.OpenAI | openai.AsyncOpenAI,
    prysm_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    upstream_api_key: Optional[str] = None,
    forward_headers: Optional[Dict[str, str]] = None,
) -> openai.OpenAI | openai.AsyncOpenAI:
    """
    Wrap an existing OpenAI client to route all traffic through Prysm.

    This is the primary entry point for the SDK. It takes your existing
    OpenAI client and returns a new one that sends all requests through
    the Prysm proxy for observability.

    Args:
        client: An existing openai.OpenAI or openai.AsyncOpenAI instance.
        prysm_key: Your Prysm API key (sk-prysm-...). Falls back to PRYSM_API_KEY env var.
        base_url: Prysm proxy URL. Falls back to PRYSM_BASE_URL env var or https://prysmai.io/api/v1.
        timeout: Request timeout in seconds (default 120).
        upstream_api_key: Dynamic upstream provider API key. Overrides the stored key in the project config.
            Useful for CI/CD integrations where the API key is injected at runtime (e.g., GitLab AI Gateway).
        forward_headers: Custom headers to forward to the upstream provider. Merged into the upstream
            request. Cannot override Content-Type or Authorization. Useful for passing platform-specific
            headers (e.g., X-Gitlab-Instance-Id).

    Returns:
        A new OpenAI client instance routed through Prysm.

    Example:
        import openai
        from prysmai import monitor

        client = openai.OpenAI(api_key="sk-...")
        monitored = monitor(client, prysm_key="sk-prysm-...")

        # This call is now tracked by Prysm
        response = monitored.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )

    GitLab AI Gateway example:
        import os, openai
        from prysmai import monitor

        client = openai.OpenAI()
        monitored = monitor(
            client,
            prysm_key="sk-prysm-...",
            upstream_api_key=os.environ["AI_FLOW_AI_GATEWAY_TOKEN"],
            forward_headers={
                "X-Gitlab-Instance-Id": os.environ.get("CI_SERVER_HOST", ""),
                "X-Gitlab-Realm": "saas",
            },
        )
    """
    prysm = PrysmClient(
        prysm_key=prysm_key,
        base_url=base_url,
        timeout=timeout,
        upstream_api_key=upstream_api_key,
        forward_headers=forward_headers,
    )

    if isinstance(client, openai.AsyncOpenAI):
        return prysm.async_openai()
    else:
        return prysm.openai()
