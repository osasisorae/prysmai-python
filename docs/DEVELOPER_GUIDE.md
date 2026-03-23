# PrysmAI Python SDK Developer Guide

This guide is for developers who want to build AI applications with PrysmAI
without guessing how the pieces fit together.

Prysm has two first-class integration paths:

- **Proxy path** for application traffic
- **MCP path** for agent runtimes

Both should feed the same control plane.

## Before You Start

You need:

1. a Prysm project
2. a Prysm API key from the dashboard
3. a Prysm base URL

Production:

```text
https://prysmai.io/api/v1
```

Local development:

```text
http://localhost:3000/api/v1
```

Set environment variables if you want the SDK to pick them up automatically:

```bash
export PRYSM_API_KEY="sk-prysm-..."
export PRYSM_BASE_URL="http://localhost:3000/api/v1"
```

## Choose Your Integration Path

Choose the **proxy path** when:

- your app already calls a model provider directly
- you want the easiest path to traces, metrics, and security scanning
- you want Prysm in the request path

Choose the **MCP path** when:

- your runtime is MCP-native
- you want Prysm as the governance and evidence layer for an agent runtime
- the runtime may execute model calls or tool actions outside Prysm's HTTP proxy

Choose a **unified session** when:

- one run spans requests, tool calls, file changes, and decisions
- you want a single session in the Prysm dashboard

## Proxy Path

### Start with `PrysmClient`

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")
client = prysm.openai()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain retrieval augmented generation."}],
)
```

This does three things:

1. routes traffic through Prysm
2. authenticates with your Prysm API key
3. lets Prysm use the provider credentials stored in your project

### Wrap an Existing Client

```python
from openai import OpenAI
from prysmai import monitor

client = OpenAI()
monitored = monitor(client, prysm_key="sk-prysm-...")
```

Use this when you want minimal code churn.

### Add Request Metadata

Use `prysm_context` to tag requests with user and session information.

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

That metadata is attached as Prysm headers and shows up in the control plane.

## MCP Path

Use the MCP client when your runtime connects to tools over MCP.

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")
mcp = prysm.mcp()

config = mcp.connection_config()

print(config.server_url)
print(config.headers)
```

Provide that configuration to your MCP-compatible runtime.

The MCP server URL is derived from the same base URL you use for the proxy.
For example:

- proxy base URL: `http://localhost:3000/api/v1`
- MCP URL: `http://localhost:3000/api/mcp`

### What The MCP Surface Can Record

The MCP client can record:

- external LLM calls
- tool calls
- decisions
- file changes
- governance sessions and reports

Examples:

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")

with prysm.session(
    governance_task="Review a code change",
    agent_type="codex",
) as run:
    run.record_decision(
        description="Reject direct secret access",
        selected_action="block",
        severity="high",
    )

    run.record_file_change(
        operation="write",
        path="app/auth.py",
        language="python",
        content="print('patched')",
    )
```

## Unified Session Pattern

This is the most important pattern in the SDK right now.

Use one `PrysmSession` when you want:

- proxied model traffic
- governance evidence
- one session timeline in the Prysm dashboard

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")

with prysm.session(
    user_id="user_123",
    metadata={"feature": "support"},
    governance_task="Resolve a support request safely.",
    agent_type="codex",
    governance_context={
        "agent_id": "triage-agent",
        "delegated_authority": "support_reply",
    },
    auto_check_interval=1,
) as run:
    client = run.openai()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Draft a refund reply."}],
    )

    run.record_decision(
        description="Respond with a refund explanation",
        selected_action="respond",
        severity="medium",
    )
```

## Automatic Tool Recording

`run_tool(...)` executes a callable and records success or failure in Prysm.

```python
from prysmai import PrysmClient

prysm = PrysmClient(prysm_key="sk-prysm-...")

with prysm.session(
    governance_task="Answer a support question",
    agent_type="codex",
) as run:
    result = run.run_tool(
        "search_docs",
        lambda query: {"query": query, "hits": 3},
        "refund policy",
        tool_input={"query": "refund policy"},
    )
```

This is the cleanest way to make tool activity visible in Prysm without adding
manual recording around every call.

## Delegation And Approval Metadata

The session helpers can carry metadata such as:

- `agent_id`
- `parent_session_id`
- `delegated_authority`
- `approved_by`
- `approval_id`
- `human_approved`

That metadata matters when your policies depend on:

- human review checkpoints
- delegated authority
- multi-agent lineage
- domain or resource boundaries

## What You Should See In Prysm

After a healthy integration, Prysm should show:

- request traces
- session detail
- threat findings
- policy violations
- tool and file activity
- decisions and review evidence

If you only see model requests but not tools or decisions, your runtime is
probably using the proxy path correctly but not yet recording governance-side
events.

## Framework Integrations

The SDK also includes integrations for:

- LangGraph
- CrewAI
- Microsoft Agent Framework
- LlamaIndex

Use these when you want framework-native hooks, but keep the same control-plane
model in mind:

- proxy traffic when applicable
- session events where appropriate
- shared evidence in Prysm

## Local Testing

If Prysm is running locally:

```python
from prysmai import PrysmClient

prysm = PrysmClient(
    prysm_key="sk-prysm-...",
    base_url="http://localhost:3000/api/v1",
)
```

Then verify:

1. a proxy request appears in the dashboard
2. a session can be opened and closed
3. tool or decision events appear under that session

## Common Mistakes

### Wrong base URL

Use the Prysm base URL, not the upstream provider URL.

Correct:

```text
https://prysmai.io/api/v1
```

Not:

```text
https://api.openai.com/v1
```

### Missing Prysm API key

The SDK expects a key that starts with `sk-prysm-`.

### Expecting MCP to behave like the proxy

The MCP path is not pretending to be an HTTP reverse proxy.
Its purpose is to let agent runtimes produce the same operator-visible outcomes
inside Prysm.

## Related Docs

- [README](../README.md)
- [SDK control plane note](./SDK_CONTROL_PLANE.md)
