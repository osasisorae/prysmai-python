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
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
import openai

from prysmai.context import _prysm_ctx


# ─── Custom transport that injects Prysm headers ───


class _PrysmTransport(httpx.BaseTransport):
    """
    Wraps an httpx transport to inject Prysm context headers
    (X-Prysm-User-Id, X-Prysm-Session-Id, X-Prysm-Metadata)
    into every outgoing request.
    """

    def __init__(self, wrapped: httpx.BaseTransport):
        self._wrapped = wrapped

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        ctx = _prysm_ctx.get()
        headers = ctx.to_headers()
        for key, value in headers.items():
            request.headers[key] = value
        return self._wrapped.handle_request(request)


class _PrysmAsyncTransport(httpx.AsyncBaseTransport):
    """Async version of the Prysm header-injecting transport."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport):
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        ctx = _prysm_ctx.get()
        headers = ctx.to_headers()
        for key, value in headers.items():
            request.headers[key] = value
        return await self._wrapped.handle_async_request(request)


# ─── PrysmClient: the monitored OpenAI client ───


class PrysmClient:
    """
    A thin configuration object that produces monitored OpenAI clients.

    Usage:
        prysm = PrysmClient(prysm_key="sk-prysm-...", base_url="https://proxy.prysmai.io")
        client = prysm.openai()
        response = client.chat.completions.create(...)
    """

    def __init__(
        self,
        prysm_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.prysm_key = prysm_key or os.environ.get("PRYSM_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "PRYSM_BASE_URL", "https://proxy.prysmai.io/v1"
        )
        self.timeout = timeout

        if not self.prysm_key:
            raise ValueError(
                "Prysm API key is required. Pass prysm_key= or set PRYSM_API_KEY env var."
            )

        if not self.prysm_key.startswith("sk-prysm-"):
            raise ValueError(
                f"Invalid Prysm API key format. Expected 'sk-prysm-...' but got '{self.prysm_key[:12]}...'"
            )

    def openai(self, **kwargs: Any) -> openai.OpenAI:
        """
        Create a sync OpenAI client routed through Prysm.

        Any extra kwargs are passed to openai.OpenAI().
        """
        transport = _PrysmTransport(
            httpx.HTTPTransport(retries=2)
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
            httpx.AsyncHTTPTransport(retries=2)
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


# ─── monitor(): the one-line integration ───


def monitor(
    client: openai.OpenAI | openai.AsyncOpenAI,
    prysm_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
) -> openai.OpenAI | openai.AsyncOpenAI:
    """
    Wrap an existing OpenAI client to route all traffic through Prysm.

    This is the primary entry point for the SDK. It takes your existing
    OpenAI client and returns a new one that sends all requests through
    the Prysm proxy for observability.

    Args:
        client: An existing openai.OpenAI or openai.AsyncOpenAI instance.
        prysm_key: Your Prysm API key (sk-prysm-...). Falls back to PRYSM_API_KEY env var.
        base_url: Prysm proxy URL. Falls back to PRYSM_BASE_URL env var or https://proxy.prysmai.io/v1.
        timeout: Request timeout in seconds (default 120).

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
    """
    prysm = PrysmClient(prysm_key=prysm_key, base_url=base_url, timeout=timeout)

    if isinstance(client, openai.AsyncOpenAI):
        return prysm.async_openai()
    else:
        return prysm.openai()
