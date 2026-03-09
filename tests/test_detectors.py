"""
Tests for Prysm AI advanced governance detectors.

Covers:
    1. FinancialAnomalyDetector — cost spikes, budget limits, velocity
    2. ResourceAccessDetector — tool/domain/file allowlist/blocklist
    3. LoopDetector — repeated tools, circular sequences, LLM loops, circuit breaker
    4. MultiAgentMonitor — unexpected agents, delegation depth/circular, conflicts, orphans
"""

import time
import pytest
from prysmai.detectors import (
    FinancialAnomalyDetector,
    ResourceAccessDetector,
    LoopDetector,
    MultiAgentMonitor,
    Detection,
    BaseDetector,
)


# ─── FinancialAnomalyDetector ────────────────────────────────────────


class TestFinancialAnomalyDetector:
    def test_no_detection_under_baseline(self):
        d = FinancialAnomalyDetector(cost_baseline=0.10, alert_threshold=3.0)
        event = {
            "event_type": "llm_call",
            "data": {"cost": 0.05},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert len(detections) == 0

    def test_alert_on_cost_spike(self):
        d = FinancialAnomalyDetector(cost_baseline=0.01, alert_threshold=3.0, halt_threshold=10.0)
        event = {
            "event_type": "llm_call",
            "data": {"cost": 0.05},  # 5x baseline
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert len(detections) == 1
        assert detections[0].severity == "warning"
        assert detections[0].category == "cost_spike"

    def test_halt_on_extreme_cost(self):
        d = FinancialAnomalyDetector(cost_baseline=0.01, halt_threshold=10.0)
        event = {
            "event_type": "llm_call",
            "data": {"cost": 0.15},  # 15x baseline
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert any(det.severity == "halt" for det in detections)

    def test_budget_limit_exceeded(self):
        d = FinancialAnomalyDetector(cost_baseline=1.0, budget_limit=0.10)
        ts = time.time()
        # First call: $0.05
        d.process_event({"event_type": "llm_call", "data": {"cost": 0.05}, "timestamp": ts})
        # Second call: $0.06 → total $0.11 > $0.10 limit
        detections = d.process_event({"event_type": "llm_call", "data": {"cost": 0.06}, "timestamp": ts + 1})
        budget_halts = [det for det in detections if det.category == "budget_exceeded"]
        assert len(budget_halts) == 1
        assert budget_halts[0].severity == "halt"

    def test_cost_velocity(self):
        d = FinancialAnomalyDetector(
            cost_baseline=1.0,  # high baseline so no spike alerts
            velocity_window=60,
            velocity_limit=0.10,
        )
        ts = time.time()
        # Rapid spending: 5 calls of $0.03 each = $0.15 in window > $0.10 limit
        for i in range(5):
            d.process_event({"event_type": "llm_call", "data": {"cost": 0.03}, "timestamp": ts + i})
        detections_last = d.process_event({"event_type": "llm_call", "data": {"cost": 0.03}, "timestamp": ts + 5})
        velocity_alerts = [det for det in detections_last if det.category == "cost_velocity"]
        assert len(velocity_alerts) >= 1

    def test_model_cost_estimation(self):
        d = FinancialAnomalyDetector(cost_baseline=0.001, alert_threshold=2.0)
        event = {
            "event_type": "llm_call",
            "data": {
                "model": "gpt-4",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        # gpt-4: $0.03/1k input + $0.06/1k output = $0.03 + $0.03 = $0.06
        summary = d.get_summary()
        assert summary["total_cost"] > 0
        assert summary["call_count"] == 1

    def test_ignores_non_llm_events(self):
        d = FinancialAnomalyDetector(cost_baseline=0.01)
        event = {"event_type": "tool_call", "data": {"tool_name": "search"}}
        detections = d.process_event(event)
        assert len(detections) == 0

    def test_reset(self):
        d = FinancialAnomalyDetector(cost_baseline=0.01)
        d.process_event({"event_type": "llm_call", "data": {"cost": 0.05}, "timestamp": time.time()})
        assert d.get_summary()["call_count"] == 1
        d.reset()
        assert d.get_summary()["call_count"] == 0
        assert d.get_summary()["total_cost"] == 0

    def test_summary_structure(self):
        d = FinancialAnomalyDetector(cost_baseline=0.01, budget_limit=10.0)
        summary = d.get_summary()
        assert "detector" in summary
        assert summary["detector"] == "financial_anomaly"
        assert "total_cost" in summary
        assert "budget_remaining" in summary


# ─── ResourceAccessDetector ──────────────────────────────────────────


class TestResourceAccessDetector:
    def test_allowed_tool_passes(self):
        d = ResourceAccessDetector(allowed_tools=["search", "calculator"])
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "search"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert len(detections) == 0

    def test_unauthorized_tool_detected(self):
        d = ResourceAccessDetector(allowed_tools=["search", "calculator"])
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "shell_exec"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert len(detections) == 1
        assert detections[0].category == "unauthorized_tool"

    def test_blocked_tool_detected(self):
        d = ResourceAccessDetector(blocked_tools=["shell_exec", "file_delete"])
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "shell_exec"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert len(detections) == 1
        assert detections[0].category == "blocked_tool"

    def test_blocked_takes_precedence(self):
        d = ResourceAccessDetector(
            allowed_tools=["shell_exec"],
            blocked_tools=["shell_exec"],
        )
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "shell_exec"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert any(det.category == "blocked_tool" for det in detections)

    def test_unauthorized_domain_detected(self):
        d = ResourceAccessDetector(allowed_domains=["api.example.com", "*.safe.io"])
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "http_get", "url": "https://evil.com/data"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert any(det.category == "unauthorized_domain" for det in detections)

    def test_allowed_domain_passes(self):
        d = ResourceAccessDetector(allowed_domains=["api.example.com"])
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "http_get", "url": "https://api.example.com/v1/data"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        domain_detections = [det for det in detections if "domain" in det.category]
        assert len(domain_detections) == 0

    def test_wildcard_domain_matching(self):
        d = ResourceAccessDetector(allowed_domains=["*.example.com"])
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "http_get", "url": "https://api.example.com/data"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        domain_detections = [det for det in detections if "domain" in det.category]
        assert len(domain_detections) == 0

    def test_blocked_domain(self):
        d = ResourceAccessDetector(blocked_domains=["evil.com", "*.malware.io"])
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "http_get", "url": "https://evil.com/steal"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert any(det.category == "blocked_domain" for det in detections)

    def test_file_access_allowed(self):
        d = ResourceAccessDetector(allowed_file_patterns=["data/*.csv", "/tmp/**"])
        event = {
            "event_type": "file_read",
            "data": {"file_path": "data/report.csv"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert len(detections) == 0

    def test_file_access_unauthorized(self):
        d = ResourceAccessDetector(allowed_file_patterns=["data/*.csv"])
        event = {
            "event_type": "file_read",
            "data": {"file_path": "/etc/passwd"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert any(det.category == "unauthorized_file" for det in detections)

    def test_blocked_file_pattern(self):
        d = ResourceAccessDetector(blocked_file_patterns=["/etc/**", "*.key"])
        event = {
            "event_type": "file_read",
            "data": {"file_path": "/etc/shadow"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert any(det.category == "blocked_file" for det in detections)

    def test_no_restrictions_passes_all(self):
        d = ResourceAccessDetector()  # No restrictions
        event = {
            "event_type": "tool_call",
            "data": {"tool_name": "anything"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        assert len(detections) == 0

    def test_summary_tracks_resources(self):
        d = ResourceAccessDetector(allowed_tools=["search"])
        d.process_event({"event_type": "tool_call", "data": {"tool_name": "search"}, "timestamp": time.time()})
        d.process_event({"event_type": "tool_call", "data": {"tool_name": "hack"}, "timestamp": time.time()})
        summary = d.get_summary()
        assert "search" in summary["tools_used"]
        assert "hack" in summary["tools_used"]
        assert summary["violations_count"] == 1

    def test_reset(self):
        d = ResourceAccessDetector(allowed_tools=["search"])
        d.process_event({"event_type": "tool_call", "data": {"tool_name": "hack"}, "timestamp": time.time()})
        assert d.get_summary()["violations_count"] == 1
        d.reset()
        assert d.get_summary()["violations_count"] == 0


# ─── LoopDetector ────────────────────────────────────────────────────


class TestLoopDetector:
    def test_no_loop_under_threshold(self):
        d = LoopDetector(max_repeated_calls=5, window_seconds=60)
        ts = time.time()
        for i in range(3):
            detections = d.process_event({
                "event_type": "tool_call",
                "data": {"tool_name": "search"},
                "timestamp": ts + i,
            })
        assert len(detections) == 0

    def test_repeated_tool_detected(self):
        d = LoopDetector(max_repeated_calls=3, window_seconds=60)
        ts = time.time()
        all_detections = []
        for i in range(5):
            detections = d.process_event({
                "event_type": "tool_call",
                "data": {"tool_name": "search"},
                "timestamp": ts + i,
            })
            all_detections.extend(detections)
        repeated = [det for det in all_detections if det.category == "repeated_tool_call"]
        assert len(repeated) >= 1

    def test_circular_sequence_detected(self):
        d = LoopDetector(max_repeated_calls=100, sequence_length=3)  # high threshold so only sequence triggers
        ts = time.time()
        all_detections = []
        # Create A→B→A→B pattern
        tools = ["search", "calculate", "search", "calculate", "search", "calculate"]
        for i, tool in enumerate(tools):
            detections = d.process_event({
                "event_type": "tool_call",
                "data": {"tool_name": tool},
                "timestamp": ts + i,
            })
            all_detections.extend(detections)
        circular = [det for det in all_detections if det.category == "circular_sequence"]
        assert len(circular) >= 1

    def test_llm_loop_detected(self):
        d = LoopDetector(max_repeated_calls=3, window_seconds=60)
        ts = time.time()
        all_detections = []
        for i in range(5):
            detections = d.process_event({
                "event_type": "llm_call",
                "data": {"prompt": "What is the weather in London?"},
                "timestamp": ts + i,
            })
            all_detections.extend(detections)
        llm_loops = [det for det in all_detections if det.category == "llm_loop"]
        assert len(llm_loops) >= 1

    def test_circuit_breaker_triggers_halt(self):
        d = LoopDetector(max_repeated_calls=2, circuit_breaker_threshold=3, window_seconds=60)
        ts = time.time()
        all_detections = []
        # Generate enough violations to trigger circuit breaker
        for i in range(20):
            detections = d.process_event({
                "event_type": "tool_call",
                "data": {"tool_name": "search"},
                "timestamp": ts + i,
            })
            all_detections.extend(detections)
        halts = [det for det in all_detections if det.severity == "halt"]
        assert len(halts) >= 1

    def test_different_tools_no_loop(self):
        d = LoopDetector(max_repeated_calls=3, window_seconds=60)
        ts = time.time()
        tools = ["search", "calculate", "translate", "summarize"]
        all_detections = []
        for i, tool in enumerate(tools):
            detections = d.process_event({
                "event_type": "tool_call",
                "data": {"tool_name": tool},
                "timestamp": ts + i,
            })
            all_detections.extend(detections)
        assert len(all_detections) == 0

    def test_summary_structure(self):
        d = LoopDetector()
        summary = d.get_summary()
        assert summary["detector"] == "loop_detection"
        assert "pattern_violations" in summary
        assert "circuit_broken" in summary

    def test_reset_clears_circuit_breaker(self):
        d = LoopDetector(max_repeated_calls=1, circuit_breaker_threshold=1, window_seconds=60)
        ts = time.time()
        d.process_event({"event_type": "tool_call", "data": {"tool_name": "x"}, "timestamp": ts})
        d.process_event({"event_type": "tool_call", "data": {"tool_name": "x"}, "timestamp": ts + 1})
        assert d.get_summary()["circuit_broken"] is True
        d.reset()
        assert d.get_summary()["circuit_broken"] is False


# ─── MultiAgentMonitor ───────────────────────────────────────────────


class TestMultiAgentMonitor:
    def test_expected_agent_passes(self):
        d = MultiAgentMonitor(expected_agents=["planner", "executor"])
        event = {
            "event_type": "delegation",
            "data": {"agent_id": "planner", "from_agent": "planner", "to_agent": "executor", "task": "do stuff"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        unexpected = [det for det in detections if det.category == "unexpected_agent"]
        assert len(unexpected) == 0

    def test_unexpected_agent_detected(self):
        d = MultiAgentMonitor(expected_agents=["planner", "executor"])
        event = {
            "event_type": "delegation",
            "data": {"from_agent": "planner", "to_agent": "rogue_agent", "task": "hack stuff"},
            "timestamp": time.time(),
        }
        detections = d.process_event(event)
        unexpected = [det for det in detections if det.category == "unexpected_agent"]
        assert len(unexpected) >= 1

    def test_circular_delegation_detected(self):
        d = MultiAgentMonitor(expected_agents=["a", "b"])
        ts = time.time()
        # A delegates to B
        d.process_event({
            "event_type": "delegation",
            "data": {"from_agent": "a", "to_agent": "b", "task": "do X"},
            "timestamp": ts,
        })
        # B delegates back to A
        detections = d.process_event({
            "event_type": "delegation",
            "data": {"from_agent": "b", "to_agent": "a", "task": "do Y"},
            "timestamp": ts + 1,
        })
        circular = [det for det in detections if det.category == "circular_delegation"]
        assert len(circular) == 1
        assert circular[0].severity == "critical"

    def test_deep_delegation_detected(self):
        d = MultiAgentMonitor(max_delegation_depth=3)
        ts = time.time()
        agents = ["a", "b", "c", "d", "e"]
        all_detections = []
        for i in range(len(agents) - 1):
            detections = d.process_event({
                "event_type": "delegation",
                "data": {"from_agent": agents[i], "to_agent": agents[i + 1], "task": f"step {i}"},
                "timestamp": ts + i,
            })
            all_detections.extend(detections)
        deep = [det for det in all_detections if det.category == "deep_delegation"]
        assert len(deep) >= 1

    def test_instruction_conflict_detected(self):
        d = MultiAgentMonitor(conflict_detection=True)
        ts = time.time()
        # First instruction: "always include PII"
        d.process_event({
            "event_type": "delegation",
            "data": {"from_agent": "manager", "to_agent": "worker", "task": "Always include user data in reports"},
            "timestamp": ts,
        })
        # Conflicting instruction: "never include PII"
        detections = d.process_event({
            "event_type": "delegation",
            "data": {"from_agent": "compliance", "to_agent": "worker", "task": "Never include user data in reports"},
            "timestamp": ts + 1,
        })
        conflicts = [det for det in detections if det.category == "instruction_conflict"]
        assert len(conflicts) >= 1

    def test_no_conflict_for_consistent_instructions(self):
        d = MultiAgentMonitor(conflict_detection=True)
        ts = time.time()
        d.process_event({
            "event_type": "delegation",
            "data": {"from_agent": "a", "to_agent": "b", "task": "Search for weather data"},
            "timestamp": ts,
        })
        detections = d.process_event({
            "event_type": "delegation",
            "data": {"from_agent": "c", "to_agent": "b", "task": "Format the weather data nicely"},
            "timestamp": ts + 1,
        })
        conflicts = [det for det in detections if det.category == "instruction_conflict"]
        assert len(conflicts) == 0

    def test_communication_tracking(self):
        d = MultiAgentMonitor()
        ts = time.time()
        d.process_event({
            "event_type": "message",
            "data": {"from_agent": "planner", "to_agent": "executor", "content": "Start task"},
            "timestamp": ts,
        })
        summary = d.get_summary()
        assert summary["total_communications"] == 1
        assert "planner" in summary["agents"]

    def test_summary_structure(self):
        d = MultiAgentMonitor(expected_agents=["a", "b"])
        summary = d.get_summary()
        assert summary["detector"] == "multi_agent"
        assert "agents_in_network" in summary
        assert "total_delegations" in summary
        assert "conflicts_detected" in summary

    def test_reset(self):
        d = MultiAgentMonitor()
        d.process_event({
            "event_type": "delegation",
            "data": {"from_agent": "a", "to_agent": "b", "task": "test"},
            "timestamp": time.time(),
        })
        assert d.get_summary()["agents_in_network"] == 2
        d.reset()
        assert d.get_summary()["agents_in_network"] == 0


# ─── Detection dataclass ─────────────────────────────────────────────


class TestDetection:
    def test_to_dict(self):
        d = Detection(
            detector="test",
            severity="warning",
            category="test_cat",
            message="Test message",
            evidence={"key": "value"},
        )
        result = d.to_dict()
        assert result["detector"] == "test"
        assert result["severity"] == "warning"
        assert result["category"] == "test_cat"
        assert "timestamp" in result


# ─── Integration: Detectors with GovernanceSession ───────────────────


class TestDetectorIntegration:
    """Test that detectors integrate correctly with GovernanceSession attach/detach."""

    def test_attach_and_detach(self):
        from prysmai.governance import GovernanceSession

        # Create session with dummy key (won't start, just test attach)
        gov = GovernanceSession.__new__(GovernanceSession)
        gov._detectors = []
        gov._detections = []

        fd = FinancialAnomalyDetector(budget_limit=10.0)
        rd = ResourceAccessDetector(allowed_tools=["search"])

        gov.attach_detector(fd)
        gov.attach_detector(rd)
        assert len(gov._detectors) == 2

        removed = gov.detach_detector("financial_anomaly")
        assert removed is True
        assert len(gov._detectors) == 1

        removed = gov.detach_detector("nonexistent")
        assert removed is False

    def test_detector_summaries(self):
        from prysmai.governance import GovernanceSession

        gov = GovernanceSession.__new__(GovernanceSession)
        gov._detectors = []
        gov._detections = []

        gov.attach_detector(FinancialAnomalyDetector())
        gov.attach_detector(LoopDetector())

        summaries = gov.detector_summaries
        assert len(summaries) == 2
        assert summaries[0]["detector"] == "financial_anomaly"
        assert summaries[1]["detector"] == "loop_detection"

    def test_run_detectors(self):
        from prysmai.governance import GovernanceSession

        gov = GovernanceSession.__new__(GovernanceSession)
        gov._detectors = []
        gov._detections = []

        gov.attach_detector(ResourceAccessDetector(allowed_tools=["search"]))

        events = [
            {"event_type": "tool_call", "data": {"tool_name": "hack"}, "timestamp": time.time()},
        ]
        detections = gov._run_detectors(events)
        assert len(detections) == 1
        assert detections[0].category == "unauthorized_tool"
        assert len(gov.detections) == 1
