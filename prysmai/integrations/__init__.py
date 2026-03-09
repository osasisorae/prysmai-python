"""
Prysm AI Framework Integrations

Auto-instrumentation for popular LLM frameworks.
Each integration is lazily imported to avoid requiring all framework dependencies.

Available integrations:
    - langgraph: PrysmGraphMonitor for LangGraph stateful agent graphs
    - crewai: PrysmCrewMonitor for CrewAI crews
    - llamaindex: PrysmSpanHandler for LlamaIndex query engines
    - agent_framework: PrysmAgentFrameworkMonitor for Microsoft Agent Framework

Usage:
    from prysmai.integrations.langgraph import PrysmGraphMonitor
    from prysmai.integrations.crewai import PrysmCrewMonitor
    from prysmai.integrations.llamaindex import PrysmSpanHandler
    from prysmai.integrations.agent_framework import PrysmAgentFrameworkMonitor
"""

__all__ = [
    "langgraph",
    "crewai",
    "llamaindex",
    "agent_framework",
]
