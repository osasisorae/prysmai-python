"""
LangGraph monitoring example.

Requires:
    pip install prysmai[langgraph]

Run:
    export PRYSM_API_KEY="sk-prysm-..."
    export PRYSM_BASE_URL="http://localhost:3000/api/v1"
    python examples/langgraph_monitor.py
"""

from prysmai import PrysmClient


def main() -> None:
    prysm = PrysmClient()
    monitor = prysm.langgraph_monitor(
        user_id="demo-user",
        metadata={"example": "langgraph_monitor", "framework": "langgraph"},
        governance=True,
    )

    monitor.start_governance(
        task="Run a customer support graph safely.",
        available_tools=["search_docs"],
    )

    # Replace this section with your real LangGraph graph execution:
    #
    # graph = build_support_graph()
    # for chunk in graph.stream(
    #     {"question": "A user says they were charged twice."},
    #     config={"callbacks": [monitor]},
    # ):
    #     print(chunk)
    #
    # The important part is that Prysm is passed as a callback handler.

    print("Attach `monitor` as a LangGraph callback handler during graph execution.")

    report = monitor.end_governance()
    if report is not None:
        print("Outcome:", report.outcome)

    monitor.close()


if __name__ == "__main__":
    main()
