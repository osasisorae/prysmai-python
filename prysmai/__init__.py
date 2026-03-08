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

    # CrewAI
    from prysmai.integrations.crewai import PrysmCrewMonitor
    mon = PrysmCrewMonitor(prysm_key="sk-prysm-...")
    mon.monitor_crew(crew)

    # LlamaIndex
    from prysmai.integrations.llamaindex import PrysmSpanHandler
    handler = PrysmSpanHandler(prysm_key="sk-prysm-...")
    Settings.callback_manager.add_handler(handler)

Governance (v0.5.0):
    from prysmai import PrysmClient
    from prysmai.governance import GovernanceSession

    client = PrysmClient(prysm_key="sk-prysm-...")
    with GovernanceSession(client, task="Fix auth bug", agent_type="claude_code") as gov:
        gov.check_behavior([{"event_type": "llm_call", "data": {...}}])
        gov.scan_code(code="...", language="python")
    # Session auto-ends, report generated
"""

from prysmai.client import monitor, PrysmClient
from prysmai.context import prysm_context, PrysmContext
from prysmai.governance import GovernanceSession

__version__ = "0.5.0"
__all__ = [
    "monitor",
    "PrysmClient",
    "prysm_context",
    "PrysmContext",
    "GovernanceSession",
    "__version__",
]
