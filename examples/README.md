# PrysmAI Python SDK Examples

These examples mirror the three main ways to use PrysmAI:

1. **Proxy path** for application traffic
2. **MCP path** for agent runtimes
3. **Unified session path** for correlated runs across requests and governance evidence

## Setup

Set your Prysm credentials:

```bash
export PRYSM_API_KEY="sk-prysm-..."
export PRYSM_BASE_URL="http://localhost:3000/api/v1"
```

Then run any example:

```bash
python examples/proxy_chat.py
python examples/mcp_agent_runtime.py
python examples/unified_session.py
```

## Files

- `proxy_chat.py`
  - Smallest application example
  - Routes an OpenAI-style request through Prysm

- `mcp_agent_runtime.py`
  - MCP-oriented example
  - Shows how to derive MCP connection config and record external runtime events

- `unified_session.py`
  - Best example of the current Prysm product shape
  - Correlates model traffic, decisions, tools, and file changes under one run
