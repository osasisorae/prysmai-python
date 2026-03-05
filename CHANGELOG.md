# Changelog

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
