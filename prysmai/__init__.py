"""
Prysm AI — Observability SDK for LLM applications.

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

Framework integrations (v0.4.0):
    # LangChain
    from prysmai.integrations.langchain import PrysmCallbackHandler
    handler = PrysmCallbackHandler(prysm_key="sk-prysm-...")
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

    # CrewAI
    from prysmai.integrations.crewai import PrysmCrewMonitor
    mon = PrysmCrewMonitor(prysm_key="sk-prysm-...")
    mon.monitor_crew(crew)

    # LlamaIndex
    from prysmai.integrations.llamaindex import PrysmSpanHandler
    handler = PrysmSpanHandler(prysm_key="sk-prysm-...")
    Settings.callback_manager.add_handler(handler)
"""

from prysmai.client import monitor, PrysmClient
from prysmai.context import prysm_context, PrysmContext

__version__ = "0.4.1"
__all__ = ["monitor", "PrysmClient", "prysm_context", "PrysmContext", "__version__"]
