"""
Real integration tests for Prysm's control-plane surfaces.

These tests are intended to run against a real Prysm deployment using a real
Prysm API key. They validate the two integration surfaces that now define the
SDK direction:

- Proxy surface via PrysmClient.openai()
- MCP surface via PrysmClient.mcp()
- Unified session scope across both via PrysmClient.session()
"""

import os


PRYSM_API_KEY = os.environ["PRYSM_API_KEY"]
PRYSM_BASE_URL = os.environ.get("PRYSM_BASE_URL", "https://prysmai.io/api/v1")


def test_mcp_lists_expected_tools():
    """The MCP surface should expose Prysm's core governance tools."""
    from prysmai import PrysmClient

    prysm = PrysmClient(prysm_key=PRYSM_API_KEY, base_url=PRYSM_BASE_URL)
    mcp = prysm.mcp(timeout=30.0)

    tools = mcp.list_tools()
    tool_names = {tool.get("name") for tool in tools}

    assert tool_names.issuperset(
        {
            "prysm_session_start",
            "prysm_check_behavior",
            "prysm_scan_code",
            "prysm_session_end",
        }
    )

    config = mcp.connection_config()
    assert config.server_url.endswith("/api/mcp")
    assert config.headers["Authorization"].startswith("Bearer ")

    mcp.close()


def test_unified_session_runs_proxy_and_governance_together():
    """
    A unified session should:
    - create a governance session over MCP
    - route model traffic through the proxy
    - close with a governance report
    """
    from prysmai import PrysmClient

    prysm = PrysmClient(prysm_key=PRYSM_API_KEY, base_url=PRYSM_BASE_URL)

    run = prysm.session(
        user_id="real-test-user",
        metadata={"source": "test_control_plane_real"},
        governance_task="Answer a trivial prompt safely and exit cleanly.",
        agent_type="codex",
        auto_check_interval=1,
    )

    with run:
        client = run.openai()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with the single word ok.",
                }
            ],
            max_tokens=5,
        )

        content = (response.choices[0].message.content or "").strip().lower()
        assert "ok" in content

        check = run.report_event(
            "tool_call",
            {
                "tool_name": "noop",
                "tool_input": {"kind": "real-test"},
            },
        )
        assert check is not None

    assert run.identifiers.session_id
    assert run.identifiers.governance_session_id
    assert run.governance_report is not None
    assert run.governance_report.outcome == "completed"
