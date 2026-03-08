"""
Prysm AI Framework Integrations

Auto-instrumentation for popular LLM frameworks.
Each integration is lazily imported to avoid requiring all framework dependencies.

Available integrations:
    - langgraph: PrysmGraphMonitor for LangGraph stateful agent graphs
    - crewai: PrysmCrewMonitor for CrewAI crews
    - llamaindex: PrysmSpanHandler for LlamaIndex query engines

Usage:
    from prysmai.integrations.langgraph import PrysmGraphMonitor
    from prysmai.integrations.crewai import PrysmCrewMonitor
    from prysmai.integrations.llamaindex import PrysmSpanHandler
"""

__all__ = [
    "langgraph",
    "crewai",
    "llamaindex",
]
