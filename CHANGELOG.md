# Changelog

## 0.5.0 (2026-03-08)

### Features — Governance Layer

- **GovernanceSession** — New `prysmai.governance.GovernanceSession` class that wraps the Prysm MCP governance endpoint. Manages the full session lifecycle (start, check behavior, scan code, end) with context manager support. Parses SSE-wrapped JSON-RPC responses from the Streamable HTTP transport automatically.
- **Behavioral Detection** — `check_behavior()` reports agent events (LLM calls, tool calls, decisions) and receives real-time behavioral feedback including early stopping detection and tool undertriggering analysis. Flags accumulate across checks via `all_flags` property.
- **Code Security Scanning** — `scan_code()` scans generated code for vulnerabilities (command injection, SQL injection, path traversal, PII exposure) and returns structured `ScanResult` with threat scores and remediation recommendations.
- **Auto-Check Interval** — `GovernanceSession(auto_check_interval=N)` automatically calls `check_behavior` every N events reported via `report_event()`, reducing boilerplate for streaming agent architectures.

### Features — LangGraph Integration (replaces LangChain)

- **PrysmGraphMonitor** — New `prysmai.integrations.langgraph.PrysmGraphMonitor` callback handler built for LangGraph's stateful graph architecture. Tracks node execution order, state transitions, node-to-tool mappings, and timing data.
- **Graph-Specific Telemetry** — Captures `langgraph_node` metadata from LangGraph's callback system, mapping every LLM call and tool invocation to its parent graph node. Exposes `node_execution_order`, `node_timings`, `state_transitions`, and `graph_summary` properties.
- **Governance Integration** — `PrysmGraphMonitor(governance=True)` auto-starts a governance session on the first graph event. Use `start_governance()` / `end_governance()` for explicit control. Events are enriched with graph node context before being forwarded to the behavioral detection engine.

### Features — CrewAI Governance Extension

- **`governance=True` flag** — `PrysmCrewMonitor(governance=True)` wraps crew execution in a GovernanceSession. Agent executions, task completions, tool calls, and delegation events are automatically forwarded for behavioral analysis.
- **Governance Report** — Access the final report via `monitor.governance_report` after crew completion, including behavior score, detector results, and policy violations.

### Features — Context Propagation

- **`governance_session_id`** — New field on `PrysmContext` for propagating the active governance session ID through `contextvars`. Integrations automatically read and write this field.

### Breaking Changes

- **LangChain integration removed** — `prysmai.integrations.langchain` has been removed. Use `prysmai.integrations.langgraph` with `PrysmGraphMonitor` instead. LangGraph uses the same `langchain_core` callback system, so migration is a class rename.
- **Optional dependency renamed** — `pip install prysmai[langchain]` is now `pip install prysmai[langgraph]`.

### Internal

- New data classes: `CheckResult`, `ScanResult`, `SessionReport`, `BehavioralFlag`, `Vulnerability`
- New exceptions: `GovernanceError`, `SessionNotActiveError`
- `_McpTransport` handles JSON-RPC over SSE with automatic response parsing
- 71 new unit tests covering governance, LangGraph integration, and data classes

## 0.4.1 (2026-03-05)

### Bug Fixes

- **BUG-001: LangChain `serialized=None` crash** — Fixed `AttributeError: 'NoneType' object has no attribute 'get'` in `PrysmCallbackHandler` when LangChain passes `None` for the `serialized` parameter in `on_chain_start`, `on_llm_start`, `on_chat_model_start`, and `on_tool_start`. This occurs with certain chain types (e.g., `RunnableSequence`). Added null guards that default to an empty dict.

- **BUG-003: CrewAI delegation tool serialization crash** — Fixed crash in `PrysmCrewMonitor._on_tool_start` when CrewAI's `DelegateWorkToolSchema` receives malformed arguments from `gpt-4o-mini`. The monitor now wraps tool event handlers in defensive try/catch blocks and safely serializes tool inputs, preventing the monitor from crashing the entire crew execution.

## 0.4.0 (2026-02-28)

### Features

- LangChain integration via `PrysmCallbackHandler`
- CrewAI integration via `PrysmCrewMonitor`
- LlamaIndex integration via `PrysmSpanHandler`
- Context propagation via `prysm_context`
- Multi-provider support (OpenAI, Anthropic, Google Gemini, vLLM, Ollama)
