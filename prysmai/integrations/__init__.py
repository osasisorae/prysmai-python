"""
Prysm AI Framework Integrations

Auto-instrumentation for popular LLM frameworks.
Each integration is lazily imported to avoid requiring all framework dependencies.

Available integrations:
    - langchain: PrysmCallbackHandler for LangChain chains and agents
    - crewai: PrysmCrewMonitor for CrewAI crews
    - llamaindex: PrysmSpanHandler for LlamaIndex query engines

Usage:
    from prysmai.integrations.langchain import PrysmCallbackHandler
    from prysmai.integrations.crewai import PrysmCrewMonitor
    from prysmai.integrations.llamaindex import PrysmSpanHandler
"""

__all__ = [
    "langchain",
    "crewai",
    "llamaindex",
]
