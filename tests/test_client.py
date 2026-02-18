"""
Tests for prysm.client — monitor(), PrysmClient, and transport layers.
"""

import os
import json
import pytest
import httpx
import openai

from prysmai import monitor, PrysmClient, __version__
from prysmai.client import _PrysmTransport, _PrysmAsyncTransport
from prysmai.context import prysm_context, PrysmContext, _prysm_ctx


# ─── Fixtures ───


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure clean environment for each test."""
    monkeypatch.delenv("PRYSM_API_KEY", raising=False)
    monkeypatch.delenv("PRYSM_BASE_URL", raising=False)
    prysm_context.clear()


VALID_KEY = "sk-prysm-test1234567890abcdef"
VALID_URL = "https://prysmai.io/api/v1"


# ─── PrysmClient init ───


class TestPrysmClientInit:
    def test_valid_key(self):
        pc = PrysmClient(prysm_key=VALID_KEY)
        assert pc.prysm_key == VALID_KEY
        assert pc.base_url == VALID_URL

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PRYSM_API_KEY", VALID_KEY)
        pc = PrysmClient()
        assert pc.prysm_key == VALID_KEY

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("PRYSM_API_KEY", VALID_KEY)
        monkeypatch.setenv("PRYSM_BASE_URL", "https://custom.proxy/v1")
        pc = PrysmClient()
        assert pc.base_url == "https://custom.proxy/v1"

    def test_custom_base_url(self):
        pc = PrysmClient(prysm_key=VALID_KEY, base_url="http://localhost:3000/v1")
        assert pc.base_url == "http://localhost:3000/v1"

    def test_custom_timeout(self):
        pc = PrysmClient(prysm_key=VALID_KEY, timeout=30.0)
        assert pc.timeout == 30.0

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Prysm API key is required"):
            PrysmClient()

    def test_invalid_key_prefix_raises(self):
        with pytest.raises(ValueError, match="Invalid Prysm API key format"):
            PrysmClient(prysm_key="sk-openai-bad")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="Prysm API key is required"):
            PrysmClient(prysm_key="")


# ─── PrysmClient.openai() ───


class TestPrysmClientOpenAI:
    def test_returns_openai_client(self):
        pc = PrysmClient(prysm_key=VALID_KEY)
        client = pc.openai()
        assert isinstance(client, openai.OpenAI)

    def test_client_uses_prysm_key(self):
        pc = PrysmClient(prysm_key=VALID_KEY)
        client = pc.openai()
        assert client.api_key == VALID_KEY

    def test_client_uses_prysm_base_url(self):
        pc = PrysmClient(prysm_key=VALID_KEY, base_url="http://localhost:3000/v1")
        client = pc.openai()
        assert str(client.base_url) == "http://localhost:3000/v1/"

    def test_returns_async_client(self):
        pc = PrysmClient(prysm_key=VALID_KEY)
        client = pc.async_openai()
        assert isinstance(client, openai.AsyncOpenAI)

    def test_async_client_uses_prysm_key(self):
        pc = PrysmClient(prysm_key=VALID_KEY)
        client = pc.async_openai()
        assert client.api_key == VALID_KEY


# ─── monitor() function ───


class TestMonitor:
    def test_monitor_sync_client(self):
        original = openai.OpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY)
        assert isinstance(monitored, openai.OpenAI)
        assert monitored.api_key == VALID_KEY

    def test_monitor_async_client(self):
        original = openai.AsyncOpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY)
        assert isinstance(monitored, openai.AsyncOpenAI)
        assert monitored.api_key == VALID_KEY

    def test_monitor_preserves_type(self):
        """monitor(sync) returns sync, monitor(async) returns async."""
        sync_orig = openai.OpenAI(api_key="sk-test")
        async_orig = openai.AsyncOpenAI(api_key="sk-test")

        sync_mon = monitor(sync_orig, prysm_key=VALID_KEY)
        async_mon = monitor(async_orig, prysm_key=VALID_KEY)

        assert type(sync_mon) is openai.OpenAI
        assert type(async_mon) is openai.AsyncOpenAI

    def test_monitor_custom_base_url(self):
        original = openai.OpenAI(api_key="sk-test")
        monitored = monitor(
            original,
            prysm_key=VALID_KEY,
            base_url="http://localhost:3000/v1",
        )
        assert str(monitored.base_url) == "http://localhost:3000/v1/"

    def test_monitor_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PRYSM_API_KEY", VALID_KEY)
        original = openai.OpenAI(api_key="sk-test")
        monitored = monitor(original)
        assert monitored.api_key == VALID_KEY

    def test_monitor_missing_key_raises(self):
        original = openai.OpenAI(api_key="sk-test")
        with pytest.raises(ValueError, match="Prysm API key is required"):
            monitor(original)


# ─── Context ───


class TestContext:
    def test_default_context_empty(self):
        ctx = prysm_context.get()
        assert ctx.user_id is None
        assert ctx.session_id is None
        assert ctx.metadata == {}

    def test_set_context(self):
        prysm_context.set(user_id="u1", session_id="s1")
        ctx = prysm_context.get()
        assert ctx.user_id == "u1"
        assert ctx.session_id == "s1"

    def test_set_metadata(self):
        prysm_context.set(metadata={"env": "prod", "version": "1.0"})
        ctx = prysm_context.get()
        assert ctx.metadata == {"env": "prod", "version": "1.0"}

    def test_clear_context(self):
        prysm_context.set(user_id="u1", session_id="s1")
        prysm_context.clear()
        ctx = prysm_context.get()
        assert ctx.user_id is None

    def test_context_manager_scoped(self):
        prysm_context.set(user_id="outer")
        with prysm_context(user_id="inner", session_id="sess"):
            ctx = prysm_context.get()
            assert ctx.user_id == "inner"
            assert ctx.session_id == "sess"
        # Restored after exit
        ctx = prysm_context.get()
        assert ctx.user_id == "outer"
        assert ctx.session_id is None

    def test_context_manager_merges_metadata(self):
        prysm_context.set(metadata={"a": 1})
        with prysm_context(metadata={"b": 2}):
            ctx = prysm_context.get()
            assert ctx.metadata == {"a": 1, "b": 2}
        ctx = prysm_context.get()
        assert ctx.metadata == {"a": 1}

    def test_context_to_headers_empty(self):
        ctx = PrysmContext()
        assert ctx.to_headers() == {}

    def test_context_to_headers_full(self):
        ctx = PrysmContext(
            user_id="u1",
            session_id="s1",
            metadata={"env": "test"},
        )
        headers = ctx.to_headers()
        assert headers["X-Prysm-User-Id"] == "u1"
        assert headers["X-Prysm-Session-Id"] == "s1"
        assert json.loads(headers["X-Prysm-Metadata"]) == {"env": "test"}

    def test_context_to_headers_partial(self):
        ctx = PrysmContext(user_id="u1")
        headers = ctx.to_headers()
        assert "X-Prysm-User-Id" in headers
        assert "X-Prysm-Session-Id" not in headers
        assert "X-Prysm-Metadata" not in headers


# ─── Transport header injection ───


class TestTransport:
    def test_sync_transport_injects_headers(self):
        """Verify the sync transport adds context headers to requests."""
        prysm_context.set(user_id="test_user", session_id="test_session")

        captured_headers = {}

        class MockTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured_headers.update(dict(request.headers))
                return httpx.Response(200, json={"ok": True})

        transport = _PrysmTransport(MockTransport())
        request = httpx.Request("GET", "https://example.com/test")
        transport.handle_request(request)

        assert captured_headers.get("x-prysm-user-id") == "test_user"
        assert captured_headers.get("x-prysm-session-id") == "test_session"

    @pytest.mark.asyncio
    async def test_async_transport_injects_headers(self):
        """Verify the async transport adds context headers to requests."""
        prysm_context.set(user_id="async_user")

        captured_headers = {}

        class MockAsyncTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured_headers.update(dict(request.headers))
                return httpx.Response(200, json={"ok": True})

        transport = _PrysmAsyncTransport(MockAsyncTransport())
        request = httpx.Request("GET", "https://example.com/test")
        await transport.handle_async_request(request)

        assert captured_headers.get("x-prysm-user-id") == "async_user"


# ─── Version ───


class TestVersion:
    def test_version_string(self):
        assert __version__ == "0.1.3"

    def test_version_semver_format(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
