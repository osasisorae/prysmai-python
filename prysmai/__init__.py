"""
Prysm AI — Observability & Governance SDK for LLM applications.

Usage:
    import openai
    from prysmai import monitor

    client = openai.OpenAI(api_key="sk-...")
    monitored = monitor(client, prysm_key="sk-prysm-...")

    # Every call is now tracked through Prysm.
    response = monitored.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Framework integrations (v0.5.0):
    # LangGraph
    from prysmai.integrations.langgraph import PrysmGraphMonitor
    monitor = PrysmGraphMonitor(api_key="sk-prysm-...")
    for chunk in graph.stream(inputs, config={"callbacks": [monitor]}):
        ...
    monitor.flush()

    # Or create integrations from a shared PrysmClient
    prysm = PrysmClient(prysm_key="sk-prysm-...")
    monitor = prysm.langgraph_monitor()

    # CrewAI
    from prysmai.integrations.crewai import PrysmCrewMonitor
    mon = PrysmCrewMonitor(prysm_key="sk-prysm-...")
    mon.monitor_crew(crew)

    # LlamaIndex
    from prysmai.integrations.llamaindex import PrysmSpanHandler
    handler = PrysmSpanHandler(prysm_key="sk-prysm-...")
    Settings.callback_manager.add_handler(handler)

    # Microsoft Agent Framework
    from prysmai.integrations.agent_framework import PrysmAgentFrameworkMonitor
    monitor = PrysmAgentFrameworkMonitor(api_key="sk-prysm-...")
    agent = client.as_agent(name="Bot", middleware=monitor.middleware())
    await agent.run("prompt")
    monitor.flush()

Governance (v0.5.0):
    from prysmai import PrysmClient
    from prysmai.governance import GovernanceSession

    client = PrysmClient(prysm_key="sk-prysm-...")
    with GovernanceSession(client, task="Fix auth bug", agent_type="claude_code") as gov:
        gov.check_behavior([{"event_type": "llm_call", "data": {...}}])
        gov.scan_code(code="...", language="python")
    # Session auto-ends, report generated

Advanced Detectors (v0.7.0):
    from prysmai.detectors import (
        FinancialAnomalyDetector,
        ResourceAccessDetector,
        LoopDetector,
        MultiAgentMonitor,
    )

    gov = GovernanceSession(client, task="...", agent_type="langgraph")
    gov.attach_detector(FinancialAnomalyDetector(budget_limit=50.0))
    gov.attach_detector(ResourceAccessDetector(allowed_tools=["search", "calc"]))
    gov.attach_detector(LoopDetector(max_repeated_calls=5))
    gov.attach_detector(MultiAgentMonitor(expected_agents=["planner", "executor"]))
    gov.start()
    # Detectors run locally on every check_behavior() call
    gov.check_behavior([{"event_type": "tool_call", "data": {"tool_name": "search"}}])
    report = gov.end()  # Includes detector summaries and violations

MCP surface:
    from prysmai import PrysmClient

    prysm = PrysmClient(prysm_key="sk-prysm-...")
    mcp = prysm.mcp()

    config = mcp.connection_config()
    tools = mcp.list_tools()

Unified session scope:
    from prysmai import PrysmClient

    prysm = PrysmClient(prysm_key="sk-prysm-...")

    with prysm.session(
        user_id="user_123",
        metadata={"feature": "support"},
        governance_task="Handle customer request",
        agent_type="codex",
    ) as run:
        client = run.openai()
        response = client.chat.completions.create(...)
        run.report_event("tool_call", {"tool_name": "search"})
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
