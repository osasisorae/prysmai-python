"""
Prysm MCP-path example for an external agent runtime.

This example does not pretend Prysm is the HTTP proxy. Instead, it shows how an
MCP-native runtime can connect to Prysm and produce the same operator-visible
evidence: sessions, LLM calls, tool activity, decisions, and reports.

Run:
    export PRYSM_API_KEY="sk-prysm-..."
    export PRYSM_BASE_URL="http://localhost:3000/api/v1"
    python examples/mcp_agent_runtime.py
"""

from prysmai import PrysmClient


def main() -> None:
    prysm = PrysmClient()
    mcp = prysm.mcp()

    config = mcp.connection_config()
    print("MCP server:", config.server_url)
    print("Auth header:", config.headers["Authorization"][:24] + "...")

    governance = mcp.governance_session(
        task="Review a code change and decide whether to apply it.",
        agent_type="codex",
        available_tools=["search_docs", "write_file"],
        context={
            "agent_id": "reviewer",
            "delegated_authority": "code_review",
        },
        auto_check_interval=1,
    )

    with governance as run:
        session_id = run.session_id
        print("Governance session:", session_id)

        llm_result = mcp.record_llm_call(
            session_id=session_id,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            messages=[
                {"role": "user", "content": "Review the patch and suggest a safe change."}
            ],
            completion="Use parameterized queries and reject direct string interpolation.",
            status="success",
            prompt_tokens=48,
            completion_tokens=14,
            total_tokens=62,
            latency_ms=420,
            metadata={
                "example": "mcp_agent_runtime",
                "surface": "mcp",
                "agent_id": "reviewer",
                "human_approved": False,
            },
        )

        trace_id = llm_result.get("trace_id")
        print("Recorded trace:", trace_id)

        mcp.record_tool_call(
            session_id=session_id,
            tool_name="search_docs",
            tool_input={"query": "parameterized queries python"},
            tool_output={"hits": 3},
            success=True,
            duration_ms=120,
            external_trace_id=trace_id,
            metadata={
                "example": "mcp_agent_runtime",
                "surface": "mcp",
                "agent_id": "reviewer",
            },
        )

        mcp.record_decision(
            session_id=session_id,
            description="Reject the unsafe patch and recommend a safer alternative.",
            selected_action="reject_patch",
            rationale="The patch uses string interpolation in a database query.",
            severity="medium",
            external_trace_id=trace_id,
            metadata={
                "example": "mcp_agent_runtime",
                "surface": "mcp",
                "agent_id": "reviewer",
            },
        )

    report = mcp.get_session_report(session_id)
    print("Outcome:", report.get("outcome"))

    mcp.close()


if __name__ == "__main__":
    main()
