# Prysm AI — Python SDK

**Observability for LLM applications. One line of code.**

Prysm wraps your existing OpenAI client and routes every request through the Prysm proxy, capturing latency, token usage, cost, errors, and full request/response payloads — with zero changes to your application logic.

[![PyPI version](https://img.shields.io/pypi/v/prysmai.svg)](https://pypi.org/project/prysmai/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Installation

```bash
pip install prysmai
```

## Quick Start

```python
import openai
from prysmai import monitor

# Your existing OpenAI client
client = openai.OpenAI(api_key="sk-...")

# Wrap it with Prysm — that's it
monitored = monitor(client, prysm_key="sk-prysm-...")

# Every call is now tracked
response = monitored.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)

print(response.choices[0].message.content)
```

Your dashboard at [app.prysmai.io](https://app.prysmai.io) now shows the request with full metrics: latency, tokens, cost, model, and the complete request/response.

---

## How It Works

The SDK creates a new OpenAI client that points at the Prysm proxy instead of the OpenAI API directly. The proxy:

1. **Authenticates** the request using your `sk-prysm-*` API key
2. **Forwards** the request to OpenAI (or any configured provider) using your project's stored credentials
3. **Captures** the full request, response, timing, token counts, and cost
4. **Returns** the response to your application — unchanged

Your application code stays exactly the same. The only difference is the client instance.

```
Your App  →  Prysm Proxy  →  OpenAI API
              ↓
         Metrics stored
         (latency, tokens,
          cost, errors)
```

---

## API Reference

### `monitor(client, prysm_key, base_url, timeout)`

The primary entry point. Wraps an existing OpenAI client.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `openai.OpenAI` or `openai.AsyncOpenAI` | *required* | Your existing OpenAI client |
| `prysm_key` | `str` | `PRYSM_API_KEY` env var | Your Prysm API key (`sk-prysm-...`) |
| `base_url` | `str` | `https://prysmai.io/api/v1` | Prysm proxy URL |
| `timeout` | `float` | `120.0` | Request timeout in seconds |

**Returns:** A new OpenAI client of the same type (sync or async) routed through Prysm.

```python
# Sync
monitored = monitor(openai.OpenAI(api_key="sk-..."), prysm_key="sk-prysm-...")

# Async
monitored = monitor(openai.AsyncOpenAI(api_key="sk-..."), prysm_key="sk-prysm-...")
```

### `PrysmClient(prysm_key, base_url, timeout)`

Lower-level client for more control.

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")

# Create sync client
client = prysm.openai()

# Create async client
async_client = prysm.async_openai()
```

### `prysm_context` — Request Metadata

Attach metadata (user ID, session ID, custom tags) to every request for filtering and grouping in your dashboard.

```python
from prysmai import prysm_context

# Set globally
prysm_context.set(user_id="user_123", session_id="sess_abc")

# Or use scoped context
with prysm_context(user_id="user_456", metadata={"env": "production"}):
    response = monitored.chat.completions.create(...)
    # This request is tagged with user_456

# Outside the block, context reverts to user_123
```

| Method | Description |
|--------|-------------|
| `prysm_context.set(user_id, session_id, metadata)` | Set global context for all subsequent requests |
| `prysm_context.get()` | Get the current context object |
| `prysm_context.clear()` | Reset context to defaults |
| `prysm_context(user_id, session_id, metadata)` | Use as a context manager for scoped metadata |

---

## Environment Variables

The SDK reads these environment variables as fallbacks:

| Variable | Description |
|----------|-------------|
| `PRYSM_API_KEY` | Your Prysm API key (used if `prysm_key` is not passed) |
| `PRYSM_BASE_URL` | Custom proxy URL (used if `base_url` is not passed) |

```bash
export PRYSM_API_KEY="sk-prysm-your-key-here"
```

```python
from prysmai import monitor
import openai

# No need to pass prysm_key — reads from env
monitored = monitor(openai.OpenAI(api_key="sk-..."))
```

---

## Streaming

Streaming works exactly as you'd expect — no changes needed:

```python
stream = monitored.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

The proxy captures Time to First Token (TTFT), total latency, and full streamed content.

---

## Async Support

Full async support with the same API:

```python
import asyncio
import openai
from prysmai import monitor

async def main():
    client = openai.AsyncOpenAI(api_key="sk-...")
    monitored = monitor(client, prysm_key="sk-prysm-...")

    response = await monitored.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello async!"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## Self-Hosted Proxy

If you're running the Prysm proxy on your own infrastructure:

```python
monitored = monitor(
    client,
    prysm_key="sk-prysm-...",
    base_url="http://localhost:3000/api/v1",  # Your self-hosted proxy
)
```

---

## What Gets Captured

Every request through the SDK is logged with:

| Metric | Description |
|--------|-------------|
| **Model** | Which model was called (gpt-4o, gpt-4o-mini, etc.) |
| **Latency** | Total request duration in milliseconds |
| **TTFT** | Time to first token (streaming requests) |
| **Prompt tokens** | Input token count |
| **Completion tokens** | Output token count |
| **Cost** | Calculated cost based on model pricing |
| **Status** | Success, error, or timeout |
| **Request body** | Full messages array and parameters |
| **Response body** | Complete model response |
| **User ID** | From `prysm_context` (if set) |
| **Session ID** | From `prysm_context` (if set) |
| **Custom metadata** | Any key-value pairs from `prysm_context` |

---

## Error Handling

The SDK preserves OpenAI's error types. If the upstream API returns an error, you get the same exception you'd get without Prysm:

```python
try:
    response = monitored.chat.completions.create(
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

Prysm-specific errors (invalid Prysm key, proxy unreachable) also surface as standard OpenAI exceptions so your existing error handling works unchanged.

---

## Requirements

- Python 3.9+
- `openai >= 1.0.0`
- `httpx >= 0.24.0`

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

### Test Coverage

The SDK includes 41 tests covering:

- Client initialization and validation
- Environment variable fallbacks
- Sync and async client creation
- `monitor()` function behavior
- Context management (global, scoped, nested)
- Header injection via custom transports
- Full integration tests with mock HTTP server
- Error propagation (401, 500)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Built by [Prysm AI](https://prysmai.io)** — See inside your AI.
