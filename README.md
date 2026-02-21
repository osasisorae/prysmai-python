# Prysm AI — Python SDK

**The observability layer for LLM applications. One line of code. Full visibility.**

Prysm AI sits between your application and your LLM provider, capturing every request and response with full metrics — latency, token counts, cost, errors, and complete prompt/completion data. The Python SDK makes integration a single line change.

[![PyPI version](https://img.shields.io/pypi/v/prysmai.svg)](https://pypi.org/project/prysmai/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```
Your App  →  Prysm Proxy  →  LLM Provider
              ↓               (OpenAI, Anthropic, vLLM, Ollama, or any OpenAI-compatible endpoint)
         Full observability
         (latency, tokens, cost,
          errors, alerts, traces)
```

---

## What You Get

| Feature | Description |
|---------|-------------|
| **Multi-provider proxy** | OpenAI, Anthropic (auto-translated), vLLM, Ollama, any OpenAI-compatible endpoint |
| **Full trace capture** | Every request/response logged with tokens, latency, cost, model, and custom metadata |
| **Real-time dashboard** | Live metrics charts, request explorer, model usage breakdown, WebSocket live feed |
| **3 proxy endpoints** | Chat completions, text completions, and embeddings |
| **Streaming support** | SSE passthrough with Time to First Token (TTFT) measurement |
| **Alerting engine** | Email, Slack, Discord, and custom webhook alerts on metric thresholds |
| **Team management** | Invite members via email, assign roles, manage access per organization |
| **API key auth** | `sk-prysm-*` keys with SHA-256 hashing, create/revoke from dashboard |
| **Cost tracking** | Automatic cost calculation for 30+ models, custom pricing for any model |
| **Tool calling & logprobs** | Captured and displayed in the trace detail panel |
| **Latency percentiles** | Pre-aggregated p50, p95, p99 latency and TTFT metrics |
| **Usage enforcement** | Free tier limit (10K requests/month) with configurable plan limits |

---

## Installation

```bash
pip install prysmai
```

Requires Python 3.9+ and depends on `openai` (v1.0+) and `httpx` (v0.24+), both installed automatically.

---

## Quick Start

### Option 1: PrysmClient (Recommended)

The simplest way to get started. No OpenAI API key needed in your code — the proxy uses the credentials stored in your project settings.

```python
from prysmai import PrysmClient

client = PrysmClient(prysm_key="sk-prysm-...").openai()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)

print(response.choices[0].message.content)
```

### Option 2: Wrap an Existing Client

If you already have a configured OpenAI client and want to add observability on top:

```python
from openai import OpenAI
from prysmai import monitor

client = OpenAI()  # Uses OPENAI_API_KEY env var
monitored = monitor(client, prysm_key="sk-prysm-...")

response = monitored.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
```

### Option 3: Environment Variable

Set `PRYSM_API_KEY` in your environment and skip the `prysm_key` parameter entirely:

```bash
export PRYSM_API_KEY="sk-prysm-your-key-here"
```

```python
from prysmai import PrysmClient

# Reads PRYSM_API_KEY automatically
client = PrysmClient().openai()
```

Open your Prysm dashboard. The request appears in the live feed within seconds, with full metrics.

---

## Supported Providers

The Prysm proxy supports any LLM provider. Configure your provider in the project settings — the SDK handles the rest.

| Provider | Base URL | Notes |
|----------|----------|-------|
| **OpenAI** | `https://api.openai.com/v1` | Default. All models supported (GPT-4o, GPT-4o-mini, o1, o3-mini, etc.) |
| **Anthropic** | `https://api.anthropic.com` | Auto-translated to/from OpenAI format. Use OpenAI SDK syntax — Prysm handles the conversion. |
| **vLLM** | `http://your-server:8000/v1` | Any vLLM-served model (Llama, Mistral, Qwen, etc.) |
| **Ollama** | `http://localhost:11434/v1` | Local models via Ollama |
| **Custom** | Any URL | Any OpenAI-compatible endpoint (Together AI, Groq, Fireworks, etc.) |

### Anthropic Example

Use standard OpenAI SDK syntax — the proxy translates automatically:

```python
from prysmai import PrysmClient

# Your project is configured with Anthropic as the provider
client = PrysmClient(prysm_key="sk-prysm-...").openai()

# Use OpenAI format — Prysm translates to Anthropic's API and back
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
```

---

## API Reference

### `PrysmClient(prysm_key, base_url, timeout)`

The primary entry point. Creates sync or async OpenAI clients routed through the Prysm proxy.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prysm_key` | `str` | `PRYSM_API_KEY` env var | Your Prysm API key (`sk-prysm-...`) |
| `base_url` | `str` | `https://prysmai.io/api/v1` | Prysm proxy URL |
| `timeout` | `float` | `120.0` | Request timeout in seconds |

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")

# Sync client
client = prysm.openai()

# Async client
async_client = prysm.async_openai()
```

### `monitor(client, prysm_key, base_url, timeout)`

Alternative entry point for wrapping an existing OpenAI client.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `OpenAI` or `AsyncOpenAI` | *required* | An existing OpenAI client instance |
| `prysm_key` | `str` | `PRYSM_API_KEY` env var | Your Prysm API key |
| `base_url` | `str` | `https://prysmai.io/api/v1` | Prysm proxy URL |
| `timeout` | `float` | `120.0` | Request timeout in seconds |

**Returns:** A new OpenAI client of the same type (sync or async) routed through Prysm.

```python
from openai import OpenAI
from prysmai import monitor

monitored = monitor(OpenAI(), prysm_key="sk-prysm-...")
```

### `prysm_context` — Request Metadata

Attach metadata to every request for filtering and grouping in your dashboard. Tag requests with user IDs, session IDs, or any custom key-value pairs.

```python
from prysmai import prysm_context

# Set globally — all subsequent requests include these
prysm_context.set(
    user_id="user_123",
    session_id="sess_abc",
    metadata={"env": "production", "version": "1.2.0"}
)

# Scoped — only applies within the block
with prysm_context(user_id="user_456", metadata={"feature": "chat"}):
    response = client.chat.completions.create(...)
    # Tagged with user_456

# Outside the block, reverts to user_123
```

| Method | Description |
|--------|-------------|
| `prysm_context.set(user_id, session_id, metadata)` | Set global context for all subsequent requests |
| `prysm_context.get()` | Get the current context object |
| `prysm_context.clear()` | Reset context to defaults |
| `prysm_context(user_id, session_id, metadata)` | Use as a context manager for scoped metadata |

Metadata is sent via custom HTTP headers (`X-Prysm-User-Id`, `X-Prysm-Session-Id`, `X-Prysm-Metadata`) and appears in the trace detail panel on your dashboard.

---

## Streaming

Streaming works exactly as you'd expect — no changes needed. The proxy captures Time to First Token (TTFT), total latency, and the full streamed content.

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Async Support

Full async support with the same API:

```python
import asyncio
from prysmai import PrysmClient

async def main():
    client = PrysmClient(prysm_key="sk-prysm-...").async_openai()

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello async!"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## Proxy Endpoints

The Prysm proxy exposes three OpenAI-compatible endpoints. You can also use them directly via REST without the Python SDK.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/completions` | POST | Chat completions (GPT-4o, Claude, Llama, etc.) |
| `/api/v1/completions` | POST | Text completions (legacy) |
| `/api/v1/embeddings` | POST | Embedding generation (text-embedding-3-small, etc.) |
| `/api/v1/health` | GET | Proxy health check |

### Direct REST Usage (cURL)

```bash
curl -X POST https://prysmai.io/api/v1/chat/completions \
  -H "Authorization: Bearer sk-prysm-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Custom Headers

| Header | Description |
|--------|-------------|
| `X-Prysm-User-Id` | Tag the request with a user identifier |
| `X-Prysm-Session-Id` | Tag the request with a session identifier |
| `X-Prysm-Metadata` | JSON string of custom key-value pairs |

---

## What Gets Captured

Every request through the SDK is logged with:

| Field | Description |
|-------|-------------|
| **Model** | Which model was called (gpt-4o, claude-sonnet-4-20250514, llama-3, etc.) |
| **Provider** | Which provider handled the request (openai, anthropic, vllm, ollama, custom) |
| **Latency** | Total request duration in milliseconds |
| **TTFT** | Time to first token for streaming requests |
| **Prompt tokens** | Input token count |
| **Completion tokens** | Output token count |
| **Cost** | Calculated cost based on model pricing (30+ models built-in, custom pricing supported) |
| **Status** | `success`, `error`, or provider-specific error code |
| **Request body** | Full messages array and parameters |
| **Response body** | Complete model response |
| **Tool calls** | Function/tool call names, arguments, and results (if present) |
| **Logprobs** | Token log probabilities (if requested) |
| **User ID** | From `prysm_context` or `X-Prysm-User-Id` header |
| **Session ID** | From `prysm_context` or `X-Prysm-Session-Id` header |
| **Custom metadata** | Any key-value pairs from `prysm_context` or `X-Prysm-Metadata` header |

---

## Dashboard Features

Once traces are flowing, your Prysm dashboard provides:

**Overview** — Real-time metrics cards (total requests, average latency, error rate, total cost), request volume chart, latency distribution, cost accumulation, error rate over time, model usage breakdown, and a WebSocket-powered live trace feed.

**Request Explorer** — Searchable, filterable table of all traces. Click any trace to see the full prompt, completion, token counts, latency breakdown, tool calls, logprobs, cost, and metadata in a detail panel.

**API Keys** — Create, view, and revoke `sk-prysm-*` keys. Each key shows its prefix, creation date, and last used timestamp.

**Settings** — Project configuration (provider, base URL, model, API key), team management (invite/remove members), alert configuration, custom model pricing, and usage tracking.

---

## Alerting

Configure alerts in the dashboard to get notified when metrics cross thresholds. Supported channels:

| Channel | Configuration |
|---------|--------------|
| **Email** | Sends to any email address via Resend |
| **Slack** | Webhook URL — posts to any Slack channel |
| **Discord** | Webhook URL — posts to any Discord channel |
| **Custom webhook** | Any HTTP endpoint — receives JSON payload |

Supported metrics: `error_rate`, `latency_p50`, `latency_p95`, `latency_p99`, `request_count`, `total_cost`.

Supported conditions: `>`, `>=`, `<`, `<=`, `=`.

---

## Cost Tracking

Prysm automatically calculates cost for 30+ models with built-in pricing:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4-turbo | $10.00 | $30.00 |
| o1 | $15.00 | $60.00 |
| o3-mini | $1.10 | $4.40 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-5-haiku | $0.80 | $4.00 |
| claude-3-opus | $15.00 | $75.00 |
| text-embedding-3-small | $0.02 | $0.00 |
| text-embedding-3-large | $0.13 | $0.00 |

For models not in the built-in list (open-source, self-hosted, etc.), add custom pricing in **Settings > Pricing** with your own cost-per-token rates.

---

## Self-Hosted Proxy

If you're running the Prysm proxy on your own infrastructure:

```python
from prysmai import PrysmClient

client = PrysmClient(
    prysm_key="sk-prysm-...",
    base_url="http://localhost:3000/api/v1",
).openai()
```

---

## Error Handling

The SDK preserves OpenAI's error types. If the upstream API returns an error, you get the same exception you'd get without Prysm:

```python
import openai

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "test"}],
    )
except openai.AuthenticationError:
    print("Invalid API key")
except openai.RateLimitError:
    print("Rate limited")
except openai.APIError as e:
    print(f"API error: {e}")
```

Prysm-specific errors:

| HTTP Status | Meaning |
|-------------|---------|
| `401` | Invalid or missing Prysm API key |
| `429` | Usage limit exceeded (free tier: 10K requests/month) |
| `502` | Upstream provider error (forwarded from OpenAI/Anthropic/etc.) |
| `503` | Proxy temporarily unavailable |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PRYSM_API_KEY` | Your Prysm API key (used if `prysm_key` is not passed) |
| `PRYSM_BASE_URL` | Custom proxy URL (used if `base_url` is not passed) |

---

## Development

```bash
git clone https://github.com/osasisorae/prysmai-python.git
cd prysmai-python

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

The SDK includes 41 tests covering client initialization, environment variable fallbacks, sync/async client creation, `monitor()` behavior, context management (global, scoped, nested), header injection, full integration tests with mock HTTP server, and error propagation.

---

## Links

- **Website:** [prysmai.io](https://prysmai.io)
- **Documentation:** [prysmai.io/docs](https://prysmai.io/docs)
- **PyPI:** [pypi.org/project/prysmai](https://pypi.org/project/prysmai/)
- **GitHub:** [github.com/osasisorae/prysmai-python](https://github.com/osasisorae/prysmai-python)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Built by [Prysm AI](https://prysmai.io)** — See inside your AI.
