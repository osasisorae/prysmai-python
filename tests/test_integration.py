"""
Integration tests — verify the SDK works end-to-end against a mock HTTP server.

These tests spin up a real httpx mock to simulate the Prysm proxy and verify
that requests are correctly routed, headers are injected, and responses are
properly returned.
"""

import json
import pytest
import httpx
import respx
import openai

from prysmai import monitor, PrysmClient
from prysmai.context import prysm_context


VALID_KEY = "sk-prysm-integration-test-key"
MOCK_BASE = "http://mock-prysm.local/v1"


@pytest.fixture(autouse=True)
def clean_context():
    prysm_context.clear()
    yield
    prysm_context.clear()


# ─── Mock response payloads ───

CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1709000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}


# ─── Sync integration tests ───


class TestSyncIntegration:
    @respx.mock
    def test_chat_completion_routed_through_proxy(self):
        """Full sync flow: monitor → create → response."""
        route = respx.post(f"{MOCK_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)
        )

        original = openai.OpenAI(api_key="sk-original-key")
        monitored = monitor(original, prysm_key=VALID_KEY, base_url=MOCK_BASE)

        response = monitored.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert route.called
        assert response.choices[0].message.content == "Hello! How can I help you today?"
        assert response.usage.total_tokens == 18

    @respx.mock
    def test_prysm_key_sent_as_bearer(self):
        """Verify the Prysm API key is sent as Authorization: Bearer."""
        captured = {}

        def capture_request(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)

        respx.post(f"{MOCK_BASE}/chat/completions").mock(side_effect=capture_request)

        original = openai.OpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY, base_url=MOCK_BASE)

        monitored.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
        )

        assert captured["auth"] == f"Bearer {VALID_KEY}"

    @respx.mock
    def test_context_headers_injected(self):
        """Verify context headers are injected into the request."""
        captured = {}

        def capture_request(request):
            captured["user_id"] = request.headers.get("x-prysm-user-id")
            captured["session_id"] = request.headers.get("x-prysm-session-id")
            captured["metadata"] = request.headers.get("x-prysm-metadata")
            return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)

        respx.post(f"{MOCK_BASE}/chat/completions").mock(side_effect=capture_request)

        prysm_context.set(
            user_id="user_42",
            session_id="sess_abc",
            metadata={"env": "test"},
        )

        original = openai.OpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY, base_url=MOCK_BASE)

        monitored.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
        )

        assert captured["user_id"] == "user_42"
        assert captured["session_id"] == "sess_abc"
        assert json.loads(captured["metadata"]) == {"env": "test"}

    @respx.mock
    def test_prysm_client_direct(self):
        """Test PrysmClient.openai() directly."""
        respx.post(f"{MOCK_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)
        )

        pc = PrysmClient(prysm_key=VALID_KEY, base_url=MOCK_BASE)
        client = pc.openai()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
        )

        assert response.id == "chatcmpl-test123"

    @respx.mock
    def test_scoped_context_changes(self):
        """Verify scoped context is used within the with-block."""
        captured_calls = []

        def capture_request(request):
            captured_calls.append({
                "user_id": request.headers.get("x-prysm-user-id"),
            })
            return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)

        respx.post(f"{MOCK_BASE}/chat/completions").mock(side_effect=capture_request)

        prysm_context.set(user_id="global_user")

        pc = PrysmClient(prysm_key=VALID_KEY, base_url=MOCK_BASE)
        client = pc.openai()

        # First call with global context
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "call 1"}],
        )

        # Second call with scoped context
        with prysm_context(user_id="scoped_user"):
            client2 = pc.openai()
            client2.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "call 2"}],
            )

        assert captured_calls[0]["user_id"] == "global_user"
        assert captured_calls[1]["user_id"] == "scoped_user"


# ─── Async integration tests ───


class TestAsyncIntegration:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_chat_completion(self):
        """Full async flow: monitor → create → response."""
        respx.post(f"{MOCK_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)
        )

        original = openai.AsyncOpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY, base_url=MOCK_BASE)

        response = await monitored.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello async!"}],
        )

        assert response.choices[0].message.content == "Hello! How can I help you today?"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_context_headers(self):
        """Verify context headers work in async mode."""
        captured = {}

        def capture_request(request):
            captured["user_id"] = request.headers.get("x-prysm-user-id")
            return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)

        respx.post(f"{MOCK_BASE}/chat/completions").mock(side_effect=capture_request)

        prysm_context.set(user_id="async_user_99")

        original = openai.AsyncOpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY, base_url=MOCK_BASE)

        await monitored.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
        )

        assert captured["user_id"] == "async_user_99"


# ─── Error handling ───


class TestErrorHandling:
    @respx.mock
    def test_proxy_401_raises(self):
        """Verify that a 401 from the proxy surfaces as an auth error."""
        respx.post(f"{MOCK_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        )

        original = openai.OpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY, base_url=MOCK_BASE)

        with pytest.raises(openai.AuthenticationError):
            monitored.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
            )

    @respx.mock
    def test_proxy_500_raises(self):
        """Verify that a 500 from the proxy surfaces as an API error."""
        respx.post(f"{MOCK_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                500,
                json={
                    "error": {
                        "message": "Internal server error",
                        "type": "server_error",
                        "code": "internal_error",
                    }
                },
            )
        )

        original = openai.OpenAI(api_key="sk-original")
        monitored = monitor(original, prysm_key=VALID_KEY, base_url=MOCK_BASE)

        with pytest.raises(openai.InternalServerError):
            monitored.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
            )
