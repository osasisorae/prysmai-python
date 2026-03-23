# Prysm AI Python SDK

PrysmAI is the control plane for production AI.

This SDK gives you two integration paths into the same Prysm control plane:

- **Proxy path** for application traffic you route through Prysm
- **MCP path** for agent runtimes that connect to Prysm as a governance and evidence surface

Both paths should produce the same operational outcome in Prysm:

- request traces
- security findings
- policy decisions
- governance sessions
- reviewable evidence

[![PyPI version](https://img.shields.io/pypi/v/prysmai.svg)](https://pypi.org/project/prysmai/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```text
Your App          -> Prysm Proxy (/api/v1) -> Model Provider
Agent Runtime     -> Prysm MCP   (/api/mcp) -> Same control plane
```

## Installation

```bash
pip install prysmai

# Optional integrations
pip install prysmai[langgraph]
pip install prysmai[crewai]
pip install prysmai[agent-framework]
pip install prysmai[all]
```

Requires Python 3.9+.

## The Golden Paths

### 1. Proxy path

Use this when you are building an AI application directly and want Prysm in the
request path.

```python
from prysmai import PrysmClient

prysm = PrysmClient(
    prysm_key="sk-prysm-...",
    base_url="https://prysmai.io/api/v1",
)

client = prysm.llm()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum computing simply."}],
)

print(response.choices[0].message.content)
```

### 2. Wrap an existing OpenAI client

Use this when you already have an OpenAI client and want to add Prysm without
rewriting the rest of your app.

```python
from openai import OpenAI
from prysmai import monitor

client = OpenAI()
monitored = monitor(client, prysm_key="sk-prysm-...")

response = monitored.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize the meeting notes."}],
)
```

### 3. MCP path

Use this when your runtime connects to MCP-compatible tools and you want Prysm
to act as the control and evidence layer.

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")
mcp = prysm.mcp()

config = mcp.connection_config()

print(config.server_url)
print(config.headers)
```

For MCP-compatible runtimes, hand them:

- `config.server_url`
- `config.headers`

Then use Prysm's MCP tools and resources to record model calls, tool activity,
decisions, file changes, and governance evidence.

### 4. Unified session scope

Use `PrysmClient.session(...)` when you want one correlated run across proxy
traffic and governance activity.

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")

with prysm.session(
    user_id="user_123",
    metadata={"feature": "support"},
    governance_task="Resolve a customer support request safely.",
    agent_type="codex",
    auto_check_interval=1,
) as run:
    client = run.llm()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Draft a short response."}],
    )

    run.record_decision(
        description="Send a short and safe reply",
        selected_action="respond",
        severity="low",
    )

    run.run_tool(
        "search_docs",
        lambda query: {"result_count": 2, "query": query},
        "refund policy",
        tool_input={"query": "refund policy"},
    )

print(run.identifiers.session_id)
print(run.identifiers.governance_session_id)
```

## Choosing The Right Path

Use the **proxy path** when:

- your app already talks directly to an LLM provider
- you want request/response capture automatically
- you want security scanning on proxied traffic with minimal code changes

Use the **MCP path** when:

- your runtime is MCP-native
- you are connecting Prysm to an external agent runtime
- you want session, decision, tool, and file evidence even when the model call
  happens outside Prysm's HTTP proxy

Use a **unified session** when:

- one run spans model calls, tools, file changes, and governance activity
- you want one correlated session in the Prysm dashboard

## Core SDK Surface

### `PrysmClient`

The root client for the Prysm control plane.

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")

proxy_client = prysm.llm()
mcp_client = prysm.mcp()
session = prysm.session(governance_task="Review a change", agent_type="codex")
```

`prysm.openai()` still works as a backward-compatible alias. The newer
`prysm.llm()` name is more honest because Prysm can route to Claude, Gemini,
vLLM, Ollama, or another configured provider behind the same OpenAI-compatible
surface.

### `prysm_context`

Attach user, session, and metadata to proxied requests.

```python
from prysmai import PrysmClient, prysm_context

client = PrysmClient(prysm_key="sk-prysm-...").openai()

with prysm_context(
    user_id="user_42",
    session_id="sess_checkout",
    metadata={"tenant": "acme", "feature": "checkout"},
):
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Help me check out."}],
    )
```

### `PrysmSession`

Use `PrysmSession` helpers when you need to record governance-side events
explicitly:

- `record_llm_call(...)`
- `record_tool_call(...)`
- `record_decision(...)`
- `record_file_change(...)`
- `record_delegation(...)`
- `run_tool(...)`
- `scan_code(...)`

## What Appears In Prysm

With the SDK wired correctly, Prysm can show:

- model traces
- latency, tokens, and cost
- threat and policy findings
- session events such as tool calls, decisions, and file changes
- governance reports and reviewable evidence

## Framework Integrations

The SDK also includes integrations for:

- LangGraph
- CrewAI
- Microsoft Agent Framework
- LlamaIndex

You can initialize these from the shared `PrysmClient` so they use the same
auth and base URL model.

## Configuration

The SDK resolves connection settings from:

- explicit arguments
- then environment variables

Environment variables:

- `PRYSM_API_KEY`
- `PRYSM_BASE_URL`

Default base URL:

```text
https://prysmai.io/api/v1
```

## Local Development

For local Prysm development:

```python
from prysmai import PrysmClient

prysm = PrysmClient(
    prysm_key="sk-prysm-...",
    base_url="http://localhost:3000/api/v1",
)
```

The MCP server for that same deployment will resolve to:

```text
http://localhost:3000/api/mcp
```

## More Documentation

- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [SDK control plane note](docs/SDK_CONTROL_PLANE.md)
- [Examples](examples/README.md)
- [PyPI package](https://pypi.org/project/prysmai/)

## Status

The SDK is still early, but the core product direction is now:

- one control plane
- two integration paths
- shared evidence and governance outcomes
