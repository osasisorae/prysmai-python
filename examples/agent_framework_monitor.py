"""
Microsoft Agent Framework monitoring example.

Requires:
    pip install prysmai[agent-framework]

Run:
    export PRYSM_API_KEY="sk-prysm-..."
    export PRYSM_BASE_URL="http://localhost:3000/api/v1"
    python examples/agent_framework_monitor.py
"""

from prysmai import PrysmClient


def main() -> None:
    prysm = PrysmClient()
    monitor = prysm.agent_framework_monitor(
        user_id="demo-user",
        metadata={"example": "agent_framework_monitor", "framework": "agent_framework"},
        governance=True,
    )

    # Replace this section with your real Agent Framework runtime wiring:
    #
    # from agent_framework import AIProjectClient
    # client = AIProjectClient.from_connection_string("...")
    # agent = client.as_agent(
    #     name="SupportBot",
    #     instructions="Help support users safely.",
    #     middleware=monitor.middleware(),
    # )
    # await agent.run("Handle a duplicate charge request.")
    #
    # The important part is that Prysm middleware wraps the agent execution.

    print("Attach `monitor.middleware()` to your Agent Framework agent or run.")
    print("Prysm will record agent activity, tool calls, and chat-model events.")

    monitor.close()


if __name__ == "__main__":
    main()
