"""
Prysm AI — Advanced Governance Detectors

Four behavioral detectors that plug into GovernanceSession to provide
real-time detection of financial anomalies, unauthorized resource access,
agent loops, and multi-agent coordination issues.

Usage:
    from prysmai.detectors import (
        FinancialAnomalyDetector,
        ResourceAccessDetector,
        LoopDetector,
        MultiAgentMonitor,
    )

    # Attach detectors to a GovernanceSession
    gov = GovernanceSession(client, task="...", agent_type="langgraph")
    gov.attach_detector(FinancialAnomalyDetector(
        cost_baseline=0.05,       # $0.05 per call baseline
        alert_threshold=3.0,      # alert at 3x baseline
        halt_threshold=10.0,      # halt at 10x baseline
        budget_limit=50.0,        # hard budget cap $50
    ))
    gov.attach_detector(ResourceAccessDetector(
        allowed_tools=["search", "calculator"],
        allowed_domains=["api.example.com"],
        allowed_file_patterns=["data/*.csv"],
    ))
    gov.attach_detector(LoopDetector(
        max_repeated_calls=5,
        window_seconds=60,
        circuit_breaker_threshold=10,
    ))
    gov.attach_detector(MultiAgentMonitor(
        expected_agents=["planner", "executor", "reviewer"],
    ))
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("prysmai.detectors")


# ─── Base Detector ────────────────────────────────────────────────────


@dataclass
class Detection:
    """A detection event raised by a detector."""

    detector: str
    severity: str  # "info", "warning", "critical", "halt"
    category: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }


class BaseDetector(ABC):
    """Abstract base class for governance detectors."""

    name: str = "base"
    _prysm_key: Optional[str] = None
    _base_url: Optional[str] = None

    def configure_reporting(self, prysm_key: str, base_url: str) -> None:
        """Enable reporting detections to the PrysmAI backend."""
        self._prysm_key = prysm_key
        self._base_url = base_url.rstrip("/")

    def _report_to_backend(self, detection: "Detection") -> None:
        """POST a detection to the backend threat events endpoint (best-effort)."""
        if not self._prysm_key or not self._base_url:
            return
        try:
            import httpx as _httpx
            url = self._base_url
            for suffix in ["/api/v1", "/v1"]:
                if url.endswith(suffix):
                    url = url[: -len(suffix)]
                    break
            _httpx.post(
                f"{url}/threat-events",
                headers={"Authorization": f"Bearer {self._prysm_key}", "Content-Type": "application/json"},
                json={
                    "detector": detection.detector,
                    "severity": detection.severity,
                    "category": detection.category,
                    "message": detection.message,
                    "evidence": detection.evidence,
                    "timestamp": detection.timestamp,
                },
                timeout=5,
            )
        except Exception:
            pass  # Never block the agent

    @abstractmethod
    def process_event(self, event: Dict[str, Any]) -> List["Detection"]:
        """
        Process a single event and return any detections.

        Args:
            event: Event dict with "event_type", "data", and optional "timestamp".

        Returns:
            List of Detection objects (empty if no issues detected).
        """
        ...

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the detector's state for reporting."""
        ...

    def reset(self) -> None:
        """Reset detector state. Override in subclasses."""
        pass

    def _process_and_report(self, event: Dict[str, Any]) -> List["Detection"]:
        """Call process_event and report any detections to the backend."""
        detections = self.process_event(event)
        for d in detections:
            self._report_to_backend(d)
        return detections


# ─── 1. Financial Anomaly Detector ───────────────────────────────────


# Default cost-per-1k-tokens for common models (USD)
_DEFAULT_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-4-sonnet": {"input": 0.003, "output": 0.015},
    "claude-4-opus": {"input": 0.015, "output": 0.075},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
}


class FinancialAnomalyDetector(BaseDetector):
    """
    Detects when an agent's cost trajectory deviates from baseline.

    Monitors cumulative cost, per-call cost spikes, cost velocity (rate of
    spend), and hard budget limits. Can signal alerts or halt execution.

    Args:
        cost_baseline: Expected cost per LLM call in USD (default $0.01).
        alert_threshold: Multiplier over baseline to trigger alert (default 3x).
        halt_threshold: Multiplier over baseline to trigger halt (default 10x).
        budget_limit: Hard budget cap in USD for the session (default None = no limit).
        velocity_window: Seconds over which to measure cost velocity (default 60).
        velocity_limit: Max USD per velocity_window before alert (default None).
        model_costs: Custom model cost table (overrides defaults).
    """

    name = "financial_anomaly"

    def __init__(
        self,
        cost_baseline: float = 0.01,
        alert_threshold: float = 3.0,
        halt_threshold: float = 10.0,
        budget_limit: Optional[float] = None,
        velocity_window: float = 60.0,
        velocity_limit: Optional[float] = None,
        model_costs: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.cost_baseline = cost_baseline
        self.alert_threshold = alert_threshold
        self.halt_threshold = halt_threshold
        self.budget_limit = budget_limit
        self.velocity_window = velocity_window
        self.velocity_limit = velocity_limit
        self.model_costs = {**_DEFAULT_MODEL_COSTS, **(model_costs or {})}

        # State
        self._total_cost: float = 0.0
        self._call_count: int = 0
        self._cost_history: deque = deque()  # (timestamp, cost) tuples
        self._highest_single_cost: float = 0.0
        self._alerts_raised: int = 0
        self._halts_raised: int = 0

    def _estimate_cost(self, data: Dict[str, Any]) -> float:
        """Estimate cost from event data."""
        # If cost is directly provided, use it
        if "cost" in data:
            return float(data["cost"])

        model = data.get("model", "")
        prompt_tokens = data.get("prompt_tokens", 0) or data.get("input_tokens", 0)
        completion_tokens = data.get("completion_tokens", 0) or data.get("output_tokens", 0)

        if not model or (not prompt_tokens and not completion_tokens):
            return 0.0

        # Find matching model costs (partial match)
        costs = None
        model_lower = model.lower()
        for model_key, model_cost in self.model_costs.items():
            if model_key in model_lower or model_lower in model_key:
                costs = model_cost
                break

        if costs is None:
            # Fallback: use baseline as estimate
            return self.cost_baseline

        input_cost = (prompt_tokens / 1000) * costs.get("input", 0)
        output_cost = (completion_tokens / 1000) * costs.get("output", 0)
        return input_cost + output_cost

    def _get_velocity(self, now: float) -> float:
        """Calculate cost velocity (USD per window)."""
        cutoff = now - self.velocity_window
        window_cost = sum(c for t, c in self._cost_history if t >= cutoff)
        return window_cost

    def process_event(self, event: Dict[str, Any]) -> List[Detection]:
        detections: List[Detection] = []
        event_type = event.get("event_type", "")
        data = event.get("data", {})
        ts = event.get("timestamp", time.time())

        if event_type != "llm_call":
            return detections

        cost = self._estimate_cost(data)
        if cost <= 0:
            return detections

        self._total_cost += cost
        self._call_count += 1
        self._cost_history.append((ts, cost))
        if cost > self._highest_single_cost:
            self._highest_single_cost = cost

        # Prune old entries from cost history
        cutoff = ts - self.velocity_window * 2
        while self._cost_history and self._cost_history[0][0] < cutoff:
            self._cost_history.popleft()

        # Check 1: Per-call cost spike
        if cost > self.cost_baseline * self.halt_threshold:
            self._halts_raised += 1
            detections.append(Detection(
                detector=self.name,
                severity="halt",
                category="cost_spike",
                message=(
                    f"Single call cost ${cost:.4f} exceeds halt threshold "
                    f"({self.halt_threshold}x baseline of ${self.cost_baseline:.4f})"
                ),
                evidence={
                    "call_cost": cost,
                    "baseline": self.cost_baseline,
                    "multiplier": round(cost / self.cost_baseline, 2),
                    "model": data.get("model", "unknown"),
                    "total_cost": self._total_cost,
                },
                timestamp=ts,
            ))
        elif cost > self.cost_baseline * self.alert_threshold:
            self._alerts_raised += 1
            detections.append(Detection(
                detector=self.name,
                severity="warning",
                category="cost_spike",
                message=(
                    f"Single call cost ${cost:.4f} exceeds alert threshold "
                    f"({self.alert_threshold}x baseline of ${self.cost_baseline:.4f})"
                ),
                evidence={
                    "call_cost": cost,
                    "baseline": self.cost_baseline,
                    "multiplier": round(cost / self.cost_baseline, 2),
                    "model": data.get("model", "unknown"),
                    "total_cost": self._total_cost,
                },
                timestamp=ts,
            ))

        # Check 2: Budget limit
        if self.budget_limit and self._total_cost > self.budget_limit:
            self._halts_raised += 1
            detections.append(Detection(
                detector=self.name,
                severity="halt",
                category="budget_exceeded",
                message=(
                    f"Session budget exceeded: ${self._total_cost:.4f} > "
                    f"${self.budget_limit:.2f} limit"
                ),
                evidence={
                    "total_cost": self._total_cost,
                    "budget_limit": self.budget_limit,
                    "call_count": self._call_count,
                },
                timestamp=ts,
            ))

        # Check 3: Cost velocity
        if self.velocity_limit:
            velocity = self._get_velocity(ts)
            if velocity > self.velocity_limit:
                self._alerts_raised += 1
                detections.append(Detection(
                    detector=self.name,
                    severity="critical",
                    category="cost_velocity",
                    message=(
                        f"Cost velocity ${velocity:.4f}/{self.velocity_window}s exceeds "
                        f"limit of ${self.velocity_limit:.4f}"
                    ),
                    evidence={
                        "velocity": velocity,
                        "velocity_limit": self.velocity_limit,
                        "window_seconds": self.velocity_window,
                        "total_cost": self._total_cost,
                    },
                    timestamp=ts,
                ))

        return detections

    def get_summary(self) -> Dict[str, Any]:
        return {
            "detector": self.name,
            "total_cost": round(self._total_cost, 6),
            "call_count": self._call_count,
            "avg_cost_per_call": round(self._total_cost / max(self._call_count, 1), 6),
            "highest_single_cost": round(self._highest_single_cost, 6),
            "budget_limit": self.budget_limit,
            "budget_remaining": round(self.budget_limit - self._total_cost, 6) if self.budget_limit else None,
            "alerts_raised": self._alerts_raised,
            "halts_raised": self._halts_raised,
        }

    def reset(self) -> None:
        self._total_cost = 0.0
        self._call_count = 0
        self._cost_history.clear()
        self._highest_single_cost = 0.0
        self._alerts_raised = 0
        self._halts_raised = 0


# ─── 2. Resource Access Detector ─────────────────────────────────────


class ResourceAccessDetector(BaseDetector):
    """
    Detects when an agent accesses resources outside its allowed envelope.

    Monitors tool calls, file access, API calls, and domain access against
    a configurable allowlist. Violations are flagged in real time.

    Args:
        allowed_tools: List of tool names the agent is permitted to use.
            If None, all tools are allowed.
        allowed_domains: List of domain patterns the agent can access.
            Supports wildcards (e.g., "*.example.com").
        allowed_file_patterns: List of file path patterns (glob-style).
            E.g., ["data/*.csv", "/tmp/**"].
        blocked_tools: Explicit blocklist (takes precedence over allowlist).
        blocked_domains: Explicit domain blocklist.
        blocked_file_patterns: Explicit file path blocklist.
        severity_on_violation: Default severity for violations ("warning" or "critical").
    """

    name = "resource_access"

    def __init__(
        self,
        allowed_tools: Optional[List[str]] = None,
        allowed_domains: Optional[List[str]] = None,
        allowed_file_patterns: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        blocked_file_patterns: Optional[List[str]] = None,
        severity_on_violation: str = "critical",
    ):
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.allowed_domains = allowed_domains
        self.allowed_file_patterns = allowed_file_patterns or []
        self.blocked_tools = set(blocked_tools) if blocked_tools else set()
        self.blocked_domains = blocked_domains or []
        self.blocked_file_patterns = blocked_file_patterns or []
        self.severity_on_violation = severity_on_violation

        # State
        self._violations: List[Dict[str, Any]] = []
        self._tools_used: Set[str] = set()
        self._domains_accessed: Set[str] = set()
        self._files_accessed: Set[str] = set()
        self._total_events: int = 0

    def _match_domain(self, domain: str, pattern: str) -> bool:
        """Check if a domain matches a pattern (supports * wildcard)."""
        pattern = pattern.lower().strip()
        domain = domain.lower().strip()
        if pattern.startswith("*."):
            suffix = pattern[1:]  # ".example.com"
            return domain.endswith(suffix) or domain == pattern[2:]
        return domain == pattern

    def _match_file_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if a file path matches a glob-like pattern."""
        # Convert glob to regex
        regex = pattern.replace(".", r"\.")
        regex = regex.replace("**", "§DOUBLESTAR§")
        regex = regex.replace("*", r"[^/]*")
        regex = regex.replace("§DOUBLESTAR§", r".*")
        regex = f"^{regex}$"
        return bool(re.match(regex, file_path))

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from a URL."""
        url = url.strip()
        # Remove protocol
        for prefix in ["https://", "http://", "ftp://"]:
            if url.startswith(prefix):
                url = url[len(prefix):]
                break
        # Get domain part
        domain = url.split("/")[0].split(":")[0]
        return domain if domain else None

    def _check_tool(self, tool_name: str, ts: float) -> Optional[Detection]:
        """Check if a tool call is allowed."""
        self._tools_used.add(tool_name)

        if tool_name in self.blocked_tools:
            violation = {
                "type": "blocked_tool",
                "tool": tool_name,
                "timestamp": ts,
            }
            self._violations.append(violation)
            return Detection(
                detector=self.name,
                severity=self.severity_on_violation,
                category="blocked_tool",
                message=f"Agent used blocked tool: '{tool_name}'",
                evidence=violation,
                timestamp=ts,
            )

        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            violation = {
                "type": "unauthorized_tool",
                "tool": tool_name,
                "allowed_tools": list(self.allowed_tools),
                "timestamp": ts,
            }
            self._violations.append(violation)
            return Detection(
                detector=self.name,
                severity=self.severity_on_violation,
                category="unauthorized_tool",
                message=f"Agent used unauthorized tool: '{tool_name}' (not in allowed list)",
                evidence=violation,
                timestamp=ts,
            )
        return None

    def _check_domain(self, url: str, ts: float) -> Optional[Detection]:
        """Check if a domain access is allowed."""
        domain = self._extract_domain(url)
        if not domain:
            return None

        self._domains_accessed.add(domain)

        # Check blocklist first
        for pattern in self.blocked_domains:
            if self._match_domain(domain, pattern):
                violation = {
                    "type": "blocked_domain",
                    "domain": domain,
                    "url": url,
                    "matched_pattern": pattern,
                    "timestamp": ts,
                }
                self._violations.append(violation)
                return Detection(
                    detector=self.name,
                    severity=self.severity_on_violation,
                    category="blocked_domain",
                    message=f"Agent accessed blocked domain: '{domain}'",
                    evidence=violation,
                    timestamp=ts,
                )

        # Check allowlist
        if self.allowed_domains is not None:
            allowed = any(
                self._match_domain(domain, p) for p in self.allowed_domains
            )
            if not allowed:
                violation = {
                    "type": "unauthorized_domain",
                    "domain": domain,
                    "url": url,
                    "allowed_domains": self.allowed_domains,
                    "timestamp": ts,
                }
                self._violations.append(violation)
                return Detection(
                    detector=self.name,
                    severity="warning",
                    category="unauthorized_domain",
                    message=f"Agent accessed unauthorized domain: '{domain}'",
                    evidence=violation,
                    timestamp=ts,
                )
        return None

    def _check_file(self, file_path: str, ts: float) -> Optional[Detection]:
        """Check if a file access is allowed."""
        self._files_accessed.add(file_path)

        # Check blocklist
        for pattern in self.blocked_file_patterns:
            if self._match_file_pattern(file_path, pattern):
                violation = {
                    "type": "blocked_file",
                    "file_path": file_path,
                    "matched_pattern": pattern,
                    "timestamp": ts,
                }
                self._violations.append(violation)
                return Detection(
                    detector=self.name,
                    severity=self.severity_on_violation,
                    category="blocked_file",
                    message=f"Agent accessed blocked file: '{file_path}'",
                    evidence=violation,
                    timestamp=ts,
                )

        # Check allowlist
        if self.allowed_file_patterns:
            allowed = any(
                self._match_file_pattern(file_path, p) for p in self.allowed_file_patterns
            )
            if not allowed:
                violation = {
                    "type": "unauthorized_file",
                    "file_path": file_path,
                    "allowed_patterns": self.allowed_file_patterns,
                    "timestamp": ts,
                }
                self._violations.append(violation)
                return Detection(
                    detector=self.name,
                    severity="warning",
                    category="unauthorized_file",
                    message=f"Agent accessed file outside allowed patterns: '{file_path}'",
                    evidence=violation,
                    timestamp=ts,
                )
        return None

    def process_event(self, event: Dict[str, Any]) -> List[Detection]:
        detections: List[Detection] = []
        event_type = event.get("event_type", "")
        data = event.get("data", {})
        ts = event.get("timestamp", time.time())
        self._total_events += 1

        # Check tool calls
        if event_type in ("tool_call", "tool_result"):
            tool_name = data.get("tool_name", "") or data.get("name", "")
            if tool_name:
                d = self._check_tool(tool_name, ts)
                if d:
                    detections.append(d)

            # Check URLs in tool arguments
            for key in ("url", "endpoint", "api_url", "base_url"):
                url = data.get(key, "")
                if not url and isinstance(data.get("input"), dict):
                    url = data.get("input", {}).get(key, "")
                if url and isinstance(url, str) and ("http" in url or "ftp" in url):
                    d = self._check_domain(url, ts)
                    if d:
                        detections.append(d)

        # Check file access
        if event_type in ("file_read", "file_write"):
            file_path = data.get("file_path", "") or data.get("path", "")
            if file_path:
                d = self._check_file(file_path, ts)
                if d:
                    detections.append(d)

        # Check LLM calls for function calling that might reference tools
        if event_type == "llm_call":
            functions = data.get("functions", []) or data.get("tools", [])
            for func in functions:
                fname = func.get("name", "") if isinstance(func, dict) else ""
                if fname:
                    d = self._check_tool(fname, ts)
                    if d:
                        detections.append(d)

        return detections

    def get_summary(self) -> Dict[str, Any]:
        return {
            "detector": self.name,
            "total_events_processed": self._total_events,
            "violations_count": len(self._violations),
            "tools_used": sorted(self._tools_used),
            "domains_accessed": sorted(self._domains_accessed),
            "files_accessed": sorted(self._files_accessed),
            "violations": self._violations[-20:],  # Last 20 violations
        }

    def reset(self) -> None:
        self._violations.clear()
        self._tools_used.clear()
        self._domains_accessed.clear()
        self._files_accessed.clear()
        self._total_events = 0


# ─── 3. Loop Detector ────────────────────────────────────────────────


class LoopDetector(BaseDetector):
    """
    Detects circular tool call patterns and repetitive state transitions.

    Monitors for:
    - Repeated identical tool calls within a time window
    - Circular sequences (A→B→A→B or A→B→C→A→B→C)
    - Repetitive LLM calls with similar prompts
    - Configurable circuit breaker that triggers halt

    Args:
        max_repeated_calls: Max identical tool calls in window before alert (default 5).
        window_seconds: Time window for pattern detection (default 60).
        circuit_breaker_threshold: Total repeated pattern count before halt (default 10).
        sequence_length: Length of sequences to check for cycles (default 3).
        similarity_threshold: Prompt similarity threshold for LLM loop detection (0-1, default 0.85).
    """

    name = "loop_detection"

    def __init__(
        self,
        max_repeated_calls: int = 5,
        window_seconds: float = 60.0,
        circuit_breaker_threshold: int = 10,
        sequence_length: int = 3,
        similarity_threshold: float = 0.85,
    ):
        self.max_repeated_calls = max_repeated_calls
        self.window_seconds = window_seconds
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.sequence_length = sequence_length
        self.similarity_threshold = similarity_threshold

        # State
        self._tool_calls: deque = deque()  # (timestamp, tool_name, args_hash)
        self._tool_sequence: List[str] = []  # ordered tool name sequence
        self._llm_prompts: deque = deque()  # (timestamp, prompt_hash)
        self._pattern_violations: int = 0
        self._detected_loops: List[Dict[str, Any]] = []
        self._circuit_broken: bool = False

    def _simple_hash(self, text: str) -> str:
        """Simple string hash for comparison."""
        if not text:
            return ""
        # Use first 200 chars for comparison
        return text[:200].strip().lower()

    def _check_repeated_tools(self, tool_name: str, ts: float) -> Optional[Detection]:
        """Check for repeated identical tool calls."""
        cutoff = ts - self.window_seconds
        recent = [(t, n) for t, n, _ in self._tool_calls if t >= cutoff and n == tool_name]

        if len(recent) >= self.max_repeated_calls:
            self._pattern_violations += 1
            loop_info = {
                "type": "repeated_tool",
                "tool_name": tool_name,
                "count": len(recent),
                "window_seconds": self.window_seconds,
                "timestamp": ts,
            }
            self._detected_loops.append(loop_info)

            severity = "warning"
            if self._pattern_violations >= self.circuit_breaker_threshold:
                severity = "halt"
                self._circuit_broken = True

            return Detection(
                detector=self.name,
                severity=severity,
                category="repeated_tool_call",
                message=(
                    f"Tool '{tool_name}' called {len(recent)} times in "
                    f"{self.window_seconds}s (limit: {self.max_repeated_calls})"
                ),
                evidence=loop_info,
                timestamp=ts,
            )
        return None

    def _check_circular_sequence(self, ts: float) -> Optional[Detection]:
        """Check for circular patterns in tool call sequence."""
        seq = self._tool_sequence
        if len(seq) < self.sequence_length * 2:
            return None

        # Check for repeating subsequences of various lengths
        for pattern_len in range(2, self.sequence_length + 1):
            if len(seq) < pattern_len * 2:
                continue

            recent = seq[-pattern_len:]
            preceding = seq[-(pattern_len * 2):-pattern_len]

            if recent == preceding:
                # Found a repeating pattern — check how many times it repeats
                repeat_count = 1
                for i in range(2, len(seq) // pattern_len + 1):
                    start = len(seq) - pattern_len * i
                    end = start + pattern_len
                    if start < 0:
                        break
                    if seq[start:end] == recent:
                        repeat_count += 1
                    else:
                        break

                if repeat_count >= 2:
                    self._pattern_violations += 1
                    loop_info = {
                        "type": "circular_sequence",
                        "pattern": recent,
                        "repeat_count": repeat_count,
                        "timestamp": ts,
                    }
                    self._detected_loops.append(loop_info)

                    severity = "warning"
                    if self._pattern_violations >= self.circuit_breaker_threshold:
                        severity = "halt"
                        self._circuit_broken = True

                    return Detection(
                        detector=self.name,
                        severity=severity,
                        category="circular_sequence",
                        message=(
                            f"Circular tool pattern detected: "
                            f"{' → '.join(recent)} repeated {repeat_count} times"
                        ),
                        evidence=loop_info,
                        timestamp=ts,
                    )
        return None

    def _check_llm_loop(self, prompt_hash: str, ts: float) -> Optional[Detection]:
        """Check for repetitive LLM calls with similar prompts."""
        cutoff = ts - self.window_seconds
        recent_same = sum(1 for t, h in self._llm_prompts if t >= cutoff and h == prompt_hash)

        if recent_same >= self.max_repeated_calls:
            self._pattern_violations += 1
            loop_info = {
                "type": "llm_loop",
                "repeated_count": recent_same,
                "window_seconds": self.window_seconds,
                "timestamp": ts,
            }
            self._detected_loops.append(loop_info)

            severity = "warning"
            if self._pattern_violations >= self.circuit_breaker_threshold:
                severity = "halt"
                self._circuit_broken = True

            return Detection(
                detector=self.name,
                severity=severity,
                category="llm_loop",
                message=(
                    f"Repetitive LLM calls detected: same prompt sent "
                    f"{recent_same} times in {self.window_seconds}s"
                ),
                evidence=loop_info,
                timestamp=ts,
            )
        return None

    def process_event(self, event: Dict[str, Any]) -> List[Detection]:
        detections: List[Detection] = []
        event_type = event.get("event_type", "")
        data = event.get("data", {})
        ts = event.get("timestamp", time.time())

        if self._circuit_broken:
            detections.append(Detection(
                detector=self.name,
                severity="halt",
                category="circuit_breaker",
                message="Circuit breaker is active — agent execution should be halted",
                evidence={"total_violations": self._pattern_violations},
                timestamp=ts,
            ))
            return detections

        if event_type in ("tool_call", "tool_result"):
            tool_name = data.get("tool_name", "") or data.get("name", "")
            if tool_name:
                args_str = str(data.get("input", "") or data.get("arguments", ""))
                args_hash = self._simple_hash(args_str)
                self._tool_calls.append((ts, tool_name, args_hash))
                self._tool_sequence.append(tool_name)

                # Prune old entries
                cutoff = ts - self.window_seconds * 2
                while self._tool_calls and self._tool_calls[0][0] < cutoff:
                    self._tool_calls.popleft()

                # Keep sequence manageable
                if len(self._tool_sequence) > 100:
                    self._tool_sequence = self._tool_sequence[-50:]

                d = self._check_repeated_tools(tool_name, ts)
                if d:
                    detections.append(d)

                d = self._check_circular_sequence(ts)
                if d:
                    detections.append(d)

        if event_type == "llm_call":
            prompt = data.get("prompt", "") or str(data.get("messages", ""))
            prompt_hash = self._simple_hash(prompt)
            if prompt_hash:
                self._llm_prompts.append((ts, prompt_hash))

                # Prune
                cutoff = ts - self.window_seconds * 2
                while self._llm_prompts and self._llm_prompts[0][0] < cutoff:
                    self._llm_prompts.popleft()

                d = self._check_llm_loop(prompt_hash, ts)
                if d:
                    detections.append(d)

        return detections

    def get_summary(self) -> Dict[str, Any]:
        return {
            "detector": self.name,
            "pattern_violations": self._pattern_violations,
            "circuit_broken": self._circuit_broken,
            "detected_loops": self._detected_loops[-10:],
            "unique_tools_in_sequence": len(set(self._tool_sequence)),
            "sequence_length": len(self._tool_sequence),
        }

    def reset(self) -> None:
        self._tool_calls.clear()
        self._tool_sequence.clear()
        self._llm_prompts.clear()
        self._pattern_violations = 0
        self._detected_loops.clear()
        self._circuit_broken = False


# ─── 4. Multi-Agent Monitor ──────────────────────────────────────────


class MultiAgentMonitor(BaseDetector):
    """
    Monitors multi-agent coordination for communication issues and conflicts.

    Tracks inter-agent communication, detects conflicting instructions,
    monitors delegation chains, and provides a unified view of the agent network.

    Args:
        expected_agents: List of expected agent names/IDs in the network.
            If provided, unexpected agents trigger alerts.
        max_delegation_depth: Maximum delegation chain depth before alert (default 5).
        conflict_detection: Enable conflict detection between agent instructions (default True).
        orphan_timeout: Seconds before a delegated task is considered orphaned (default 300).
    """

    name = "multi_agent"

    def __init__(
        self,
        expected_agents: Optional[List[str]] = None,
        max_delegation_depth: int = 5,
        conflict_detection: bool = True,
        orphan_timeout: float = 300.0,
    ):
        self.expected_agents = set(expected_agents) if expected_agents else None
        self.max_delegation_depth = max_delegation_depth
        self.conflict_detection = conflict_detection
        self.orphan_timeout = orphan_timeout

        # State
        self._agents_seen: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata
        self._communications: List[Dict[str, Any]] = []
        self._delegations: List[Dict[str, Any]] = []  # (from, to, task, timestamp)
        self._active_delegations: Dict[str, Dict[str, Any]] = {}  # delegation_id -> info
        self._conflicts: List[Dict[str, Any]] = []
        self._agent_instructions: Dict[str, List[str]] = defaultdict(list)
        self._total_events: int = 0

    def _register_agent(self, agent_id: str, agent_type: str = "", ts: float = 0) -> None:
        """Register an agent in the network."""
        if agent_id not in self._agents_seen:
            self._agents_seen[agent_id] = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "first_seen": ts,
                "last_seen": ts,
                "event_count": 0,
                "delegations_sent": 0,
                "delegations_received": 0,
            }
        self._agents_seen[agent_id]["last_seen"] = ts
        self._agents_seen[agent_id]["event_count"] += 1

    def _check_unexpected_agent(self, agent_id: str, ts: float) -> Optional[Detection]:
        """Check if an agent is unexpected."""
        if self.expected_agents and agent_id not in self.expected_agents:
            return Detection(
                detector=self.name,
                severity="warning",
                category="unexpected_agent",
                message=f"Unexpected agent detected in network: '{agent_id}'",
                evidence={
                    "agent_id": agent_id,
                    "expected_agents": list(self.expected_agents),
                    "timestamp": ts,
                },
                timestamp=ts,
            )
        return None

    def _check_delegation_depth(self, from_agent: str, to_agent: str, ts: float) -> Optional[Detection]:
        """Check delegation chain depth."""
        # Walk backward from from_agent to find the root of the chain
        chain = [from_agent, to_agent]
        current = from_agent
        visited = {from_agent, to_agent}
        for deleg in reversed(self._delegations[:-1]):  # Exclude the current delegation
            if deleg["to_agent"] == current and deleg["from_agent"] not in visited:
                chain.insert(0, deleg["from_agent"])
                visited.add(deleg["from_agent"])
                current = deleg["from_agent"]

        depth = len(chain) - 1  # Number of delegation hops
        if depth > self.max_delegation_depth:
            return Detection(
                detector=self.name,
                severity="warning",
                category="deep_delegation",
                message=(
                    f"Delegation chain depth ({depth}) exceeds limit "
                    f"({self.max_delegation_depth}): {' → '.join(chain)}"
                ),
                evidence={
                    "chain": chain,
                    "depth": depth,
                    "max_depth": self.max_delegation_depth,
                    "timestamp": ts,
                },
                timestamp=ts,
            )
        return None

    def _check_circular_delegation(self, from_agent: str, to_agent: str, ts: float) -> Optional[Detection]:
        """Check for circular delegation (A delegates to B, B delegates back to A)."""
        # Check if to_agent has previously delegated to from_agent
        for deleg in self._delegations:
            if deleg["from_agent"] == to_agent and deleg["to_agent"] == from_agent:
                return Detection(
                    detector=self.name,
                    severity="critical",
                    category="circular_delegation",
                    message=(
                        f"Circular delegation detected: '{from_agent}' → '{to_agent}' → '{from_agent}'"
                    ),
                    evidence={
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "original_delegation_time": deleg["timestamp"],
                        "timestamp": ts,
                    },
                    timestamp=ts,
                )
        return None

    def _check_instruction_conflicts(self, agent_id: str, instruction: str, ts: float) -> List[Detection]:
        """Check for conflicting instructions given to the same agent."""
        detections = []
        if not self.conflict_detection or not instruction:
            return detections

        existing = self._agent_instructions.get(agent_id, [])

        # Simple conflict detection: look for contradictory keywords
        contradiction_pairs = [
            ("always", "never"),
            ("include", "exclude"),
            ("allow", "deny"),
            ("enable", "disable"),
            ("accept", "reject"),
            ("increase", "decrease"),
            ("add", "remove"),
            ("start", "stop"),
        ]

        instruction_lower = instruction.lower()
        for prev_instruction in existing:
            prev_lower = prev_instruction.lower()
            for word_a, word_b in contradiction_pairs:
                if (word_a in instruction_lower and word_b in prev_lower) or \
                   (word_b in instruction_lower and word_a in prev_lower):
                    conflict = {
                        "agent_id": agent_id,
                        "instruction_a": prev_instruction[:200],
                        "instruction_b": instruction[:200],
                        "conflicting_terms": (word_a, word_b),
                        "timestamp": ts,
                    }
                    self._conflicts.append(conflict)
                    detections.append(Detection(
                        detector=self.name,
                        severity="warning",
                        category="instruction_conflict",
                        message=(
                            f"Potentially conflicting instructions for agent '{agent_id}': "
                            f"'{word_a}' vs '{word_b}'"
                        ),
                        evidence=conflict,
                        timestamp=ts,
                    ))
                    break

        self._agent_instructions[agent_id].append(instruction)
        return detections

    def _check_orphaned_delegations(self, ts: float) -> List[Detection]:
        """Check for delegations that haven't been completed."""
        detections = []
        for deleg_id, deleg in list(self._active_delegations.items()):
            age = ts - deleg["timestamp"]
            if age > self.orphan_timeout:
                detections.append(Detection(
                    detector=self.name,
                    severity="warning",
                    category="orphaned_delegation",
                    message=(
                        f"Delegation from '{deleg['from_agent']}' to '{deleg['to_agent']}' "
                        f"has been pending for {age:.0f}s (timeout: {self.orphan_timeout}s)"
                    ),
                    evidence={
                        "delegation_id": deleg_id,
                        "from_agent": deleg["from_agent"],
                        "to_agent": deleg["to_agent"],
                        "age_seconds": round(age, 1),
                        "timestamp": ts,
                    },
                    timestamp=ts,
                ))
                del self._active_delegations[deleg_id]
        return detections

    def process_event(self, event: Dict[str, Any]) -> List[Detection]:
        detections: List[Detection] = []
        event_type = event.get("event_type", "")
        data = event.get("data", {})
        ts = event.get("timestamp", time.time())
        self._total_events += 1

        # Extract agent info from event
        agent_id = data.get("agent_id", "") or data.get("agent_name", "")
        agent_type = data.get("agent_type", "")

        if agent_id:
            self._register_agent(agent_id, agent_type, ts)
            d = self._check_unexpected_agent(agent_id, ts)
            if d:
                detections.append(d)

        # Handle delegation events
        if event_type == "delegation":
            from_agent = data.get("from_agent", "") or agent_id
            to_agent = data.get("to_agent", "") or data.get("delegate_to", "")
            task = data.get("task", "") or data.get("instruction", "")

            if from_agent and to_agent:
                self._register_agent(from_agent, "", ts)
                self._register_agent(to_agent, "", ts)
                self._agents_seen[from_agent]["delegations_sent"] += 1
                self._agents_seen[to_agent]["delegations_received"] += 1

                deleg_info = {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "task": task[:200],
                    "timestamp": ts,
                }
                self._delegations.append(deleg_info)
                self._communications.append({
                    "type": "delegation",
                    **deleg_info,
                })

                # Track active delegation
                deleg_id = f"{from_agent}->{to_agent}:{ts}"
                self._active_delegations[deleg_id] = deleg_info

                # Check for issues
                d = self._check_delegation_depth(from_agent, to_agent, ts)
                if d:
                    detections.append(d)

                d = self._check_circular_delegation(from_agent, to_agent, ts)
                if d:
                    detections.append(d)

                d = self._check_unexpected_agent(to_agent, ts)
                if d:
                    detections.append(d)

            # Check instruction conflicts
            if to_agent and task:
                conflict_detections = self._check_instruction_conflicts(to_agent, task, ts)
                detections.extend(conflict_detections)

        # Handle inter-agent communication
        if event_type in ("message", "communication", "agent_message"):
            from_agent = data.get("from_agent", "") or data.get("sender", "") or agent_id
            to_agent = data.get("to_agent", "") or data.get("recipient", "")
            content = data.get("content", "") or data.get("message", "")

            if from_agent:
                self._register_agent(from_agent, "", ts)
            if to_agent:
                self._register_agent(to_agent, "", ts)

            self._communications.append({
                "type": "message",
                "from_agent": from_agent,
                "to_agent": to_agent,
                "content_preview": content[:100],
                "timestamp": ts,
            })

        # Handle delegation completion
        if event_type == "delegation_complete":
            from_agent = data.get("from_agent", "")
            to_agent = data.get("to_agent", "")
            # Remove from active delegations
            for deleg_id in list(self._active_delegations.keys()):
                deleg = self._active_delegations[deleg_id]
                if deleg["from_agent"] == from_agent and deleg["to_agent"] == to_agent:
                    del self._active_delegations[deleg_id]
                    break

        # Periodically check for orphaned delegations
        if self._total_events % 10 == 0:
            detections.extend(self._check_orphaned_delegations(ts))

        return detections

    def get_summary(self) -> Dict[str, Any]:
        return {
            "detector": self.name,
            "agents_in_network": len(self._agents_seen),
            "agents": {
                aid: {
                    "type": info["agent_type"],
                    "event_count": info["event_count"],
                    "delegations_sent": info["delegations_sent"],
                    "delegations_received": info["delegations_received"],
                }
                for aid, info in self._agents_seen.items()
            },
            "total_communications": len(self._communications),
            "total_delegations": len(self._delegations),
            "active_delegations": len(self._active_delegations),
            "conflicts_detected": len(self._conflicts),
            "total_events_processed": self._total_events,
        }

    def reset(self) -> None:
        self._agents_seen.clear()
        self._communications.clear()
        self._delegations.clear()
        self._active_delegations.clear()
        self._conflicts.clear()
        self._agent_instructions.clear()
        self._total_events = 0
