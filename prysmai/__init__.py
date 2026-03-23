"""
PrysmAI Python SDK.

PrysmAI is one control plane with two first-class SDK surfaces:

- the proxy path for application traffic
- the MCP path for agent runtimes

Start with `PrysmClient`:

    from prysmai import PrysmClient

    prysm = PrysmClient(prysm_key="sk-prysm-...")

    # Proxy path
    client = prysm.openai()
    response = client.chat.completions.create(...)

    # MCP path
    mcp = prysm.mcp()
    tools = mcp.list_tools()

Use `prysm_context` to tag proxied traffic:

    from prysmai import prysm_context

    with prysm_context(user_id="user_123", session_id="sess_checkout"):
        client.chat.completions.create(...)

Use `PrysmClient.session(...)` when you want one correlated run across proxy
traffic and governance evidence:

    with prysm.session(
        user_id="user_123",
        metadata={"feature": "support"},
        governance_task="Handle customer request safely",
        agent_type="codex",
    ) as run:
        client = run.openai()
        client.chat.completions.create(...)
        run.record_decision(description="Respond safely")
        run.run_tool("search", lambda query: {"hits": 2}, "refund policy")
"""

from prysmai.client import monitor, PrysmClient
from prysmai.config import PrysmConnectionConfig
from prysmai.context import prysm_context, PrysmContext
from prysmai.governance import GovernanceSession
from prysmai.mcp import PrysmMCPClient, PrysmMCPConfig
from prysmai.session import PrysmSession, PrysmSessionIdentifiers
from prysmai.detectors import (
    FinancialAnomalyDetector,
    ResourceAccessDetector,
    LoopDetector,
    MultiAgentMonitor,
    BaseDetector,
    Detection,
)

__version__ = "0.7.0"
__all__ = [
    "monitor",
    "PrysmClient",
    "PrysmConnectionConfig",
    "PrysmMCPClient",
    "PrysmMCPConfig",
    "PrysmSession",
    "PrysmSessionIdentifiers",
    "prysm_context",
    "PrysmContext",
    "GovernanceSession",
    "FinancialAnomalyDetector",
    "ResourceAccessDetector",
    "LoopDetector",
    "MultiAgentMonitor",
    "BaseDetector",
    "Detection",
    "__version__",
]
