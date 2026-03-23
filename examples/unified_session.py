"""
Unified Prysm session example.

This is the best example of the current Prysm product shape:
one run, one correlated session, multiple types of evidence.

Run:
    export PRYSM_API_KEY="sk-prysm-..."
    export PRYSM_BASE_URL="http://localhost:3000/api/v1"
    python examples/unified_session.py
"""

from prysmai import PrysmClient


def search_docs(query: str) -> dict:
    return {
        "query": query,
        "hits": [
            "Refunds are approved by support for duplicate charges.",
            "Escalate if user requests an exception outside policy.",
        ],
    }


def main() -> None:
    prysm = PrysmClient()

    with prysm.session(
        user_id="support-user-123",
        metadata={
            "example": "unified_session",
            "feature": "support",
            "surface": "proxy+mcp",
        },
        governance_task="Handle a customer refund request safely.",
        agent_type="codex",
        governance_context={
            "agent_id": "refund-agent",
            "delegated_authority": "refund_support",
        },
        auto_check_interval=1,
    ) as run:
        client = run.llm()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Help support agents answer safely and concisely.",
                },
                {
                    "role": "user",
                    "content": "A customer says they were charged twice. Draft a reply.",
                },
            ],
            max_tokens=160,
        )

        print("Model reply:")
        print(response.choices[0].message.content)

        tool_result = run.run_tool(
            "search_docs",
            search_docs,
            "duplicate charge refund policy",
            tool_input={"query": "duplicate charge refund policy"},
        )
        print("Tool hits:", len(tool_result["hits"]))

        run.record_decision(
            description="Reply with a refund-review message and ask for the charge details.",
            selected_action="reply_with_refund_guidance",
            rationale="The user needs a safe response that stays within documented policy.",
            severity="low",
        )

        run.record_file_change(
            operation="write",
            path="outputs/refund_reply.txt",
            language="text",
            content=response.choices[0].message.content or "",
        )

    print("Session:", run.identifiers.session_id)
    print("Governance session:", run.identifiers.governance_session_id)
    if run.governance_report is not None:
        print("Outcome:", run.governance_report.outcome)
        print("Violations:", len(run.governance_report.violations))


if __name__ == "__main__":
    main()
