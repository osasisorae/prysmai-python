# Prysm AI SDK — Control Plane Plan

This document defines the SDK direction for Prysm AI as a single control plane with two first-class integration surfaces:

- the **Proxy surface**
- the **MCP surface**

The SDK should make both surfaces explicit.

---

## Goal

Users should be able to reach the same Prysm outcomes through either:

- building through the Prysm proxy
- connecting an agent runtime to Prysm through MCP

The shared outcomes are:

- traces
- policy decisions
- security findings
- governance sessions
- reviewable evidence

---

## Current State

The SDK already contains the building blocks, but they are not presented as a clean product model.

### What exists today

- `PrysmClient` and `monitor()` for proxy-based traffic
- `prysm_context` for request metadata
- `GovernanceSession` over MCP
- framework-specific telemetry integrations for LangGraph, CrewAI, LlamaIndex, and Agent Framework
- local detector support attached to governance sessions

### What is missing

- an explicit public MCP integration surface
- a clearer relationship between proxy traffic and MCP-driven governance
- a shared conceptual model in the public API

Right now, MCP exists mostly as the transport behind governance. That is too narrow.

---

## Target SDK Model

The public SDK should present:

### 1. Proxy Surface

Used when developers route application traffic through Prysm.

Examples:

- `PrysmClient(...).openai()`
- `monitor(existing_client, ...)`

Responsibilities:

- proxy routing
- request/response capture
- metadata tagging
- upstream key/header controls

### 2. MCP Surface

Used when developers connect agent runtimes to Prysm through MCP.

Examples:

- obtaining MCP connection config
- listing tools
- calling tools
- creating governance sessions from an MCP-native client

Responsibilities:

- MCP connection details
- tool access
- governance entry point
- shared auth/base URL handling

### 3. Shared Concepts

Both surfaces should align around:

- request identity
- session identity
- policy decisions
- governance reports
- evidence for review

---

## Implementation Plan

### Phase 1: Make MCP a first-class API surface

- add a public MCP client type
- let `PrysmClient` produce that MCP client
- make MCP connection config explicit and reusable
- keep `GovernanceSession` working, but stop treating it as the only MCP-facing abstraction

### Phase 2: Normalize shared concepts

- define shared identifiers and metadata expectations
- ensure proxy and MCP paths can be correlated cleanly
- document how traces, sessions, and policy decisions connect

### Phase 3: Unify framework integrations

- align framework monitors around the MCP/control-plane model
- make it obvious when a framework path uses proxy traffic, telemetry events, governance, or all three
- reduce accidental fragmentation in the SDK story

### Phase 4: Simplify the mental model

- update docs and examples
- explain the two entry points clearly:
  - build through Prysm
  - connect Prysm MCP to your agent runtime

---

## Near-Term Design Rules

- Do not treat MCP as a hidden governance implementation detail.
- Do not treat framework integrations as isolated products.
- Do not expose shared control-plane outcomes through unrelated public APIs.
- Keep the proxy path simple.
- Keep the MCP path explicit.

---

## First Implementation Slice

The first code change should introduce:

- `PrysmMCPClient`
- a reusable MCP connection config object
- `PrysmClient.mcp()` as the bridge from proxy credentials to MCP usage

That gives the SDK a visible two-surface model without breaking the existing governance API.
