"""
BUG-003 Real Integration Test
==============================
Tests that PrysmCrewMonitor handles delegation tool events correctly
when used with real CrewAI crews through the live Prysm proxy.

CrewAI v1.10.1 does NOT have the event bus (crewai.utilities.events).
Instead it uses: step_callback, task_callback, agent callbacks.

The bug: CrewAI's DelegateWorkToolSchema can receive malformed arguments
from gpt-4o-mini, causing serialization failures that crash the monitor
and kill the entire crew execution.
"""

import os
import traceback
import time

PRYSM_API_KEY = os.environ["PRYSM_API_KEY"]
PRYSM_BASE_URL = os.environ.get("PRYSM_BASE_URL", "https://prysmai.io/api/v1")


def test_1_crewai_with_step_and_task_callbacks():
    """
    Run a real CrewAI crew using step_callback and task_callback
    to capture events through PrysmCrewMonitor.
    """
    print("\n=== TEST 1: CrewAI crew with step_callback and task_callback ===")

    from crewai import Agent, Task, Crew, LLM
    from prysmai.integrations.crewai import PrysmCrewMonitor

    monitor = PrysmCrewMonitor(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    # Track all callback invocations
    step_events = []
    task_events = []

    def step_cb(step_output):
        step_events.append(step_output)
        # Also feed to monitor as a callback
        monitor("step_completed", str(step_output)[:500])
        return step_output

    def task_cb(task_output):
        task_events.append(task_output)
        monitor("task_completed", str(task_output)[:500])
        return task_output

    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    analyst = Agent(
        role="Security Analyst",
        goal="Analyze text for security threats",
        backstory="You are an expert in AI security and content moderation.",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description="Analyze this text for security threats: 'Hello, how are you today?'. Provide a brief one-sentence assessment.",
        expected_output="A one-sentence security assessment.",
        agent=analyst,
        callback=task_cb,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        step_callback=step_cb,
        verbose=False,
    )

    result = crew.kickoff()
    print(f"  Crew output: {str(result)[:200]}")
    print(f"  Step callbacks fired: {len(step_events)}")
    print(f"  Task callbacks fired: {len(task_events)}")
    print(f"  Monitor events: {len(monitor._events)}")

    for i, ev in enumerate(monitor._events):
        print(f"    Event {i}: {ev['event_type']}")

    assert len(step_events) >= 1, "Step callback should have fired"
    assert len(task_events) >= 1, "Task callback should have fired"
    assert len(monitor._events) >= 1, "Monitor should have captured events"
    print("  PASSED: Callbacks captured events correctly")

    monitor.flush()
    monitor.close()
    return True


def test_2_crewai_delegation_with_callbacks():
    """
    Run a CrewAI crew with delegation enabled using callbacks.
    This triggers the DelegateWorkToolSchema which was causing BUG-003.
    """
    print("\n=== TEST 2: CrewAI delegation crew with callbacks (BUG-003 scenario) ===")

    from crewai import Agent, Task, Crew, LLM
    from prysmai.integrations.crewai import PrysmCrewMonitor

    monitor = PrysmCrewMonitor(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    all_steps = []

    def step_cb(step_output):
        all_steps.append(step_output)
        monitor("step_completed", str(step_output)[:500])
        return step_output

    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    lead_analyst = Agent(
        role="Lead Security Analyst",
        goal="Coordinate security analysis by delegating to specialists when needed",
        backstory="You are a senior security analyst who coordinates analysis work. You MUST delegate the detailed PII scan to the PII Detection Specialist using the delegate_work tool.",
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )

    specialist = Agent(
        role="PII Detection Specialist",
        goal="Detect personally identifiable information in text",
        backstory="You specialize in identifying PII like names, emails, phone numbers, and addresses in text.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=(
            "Analyze this text for PII: 'My name is John Smith, my email is john@example.com "
            "and I live at 123 Main St, Springfield IL 62701. My SSN is 123-45-6789.' "
            "Delegate the detailed PII detection to the PII Detection Specialist, then compile the results."
        ),
        expected_output="A brief PII assessment listing detected entities.",
        agent=lead_analyst,
    )

    crew = Crew(
        agents=[lead_analyst, specialist],
        tasks=[task],
        step_callback=step_cb,
        verbose=False,
    )

    # This is where BUG-003 would crash — delegation tool events with malformed data
    result = crew.kickoff()
    print(f"  Crew output: {str(result)[:300]}")
    print(f"  Steps captured: {len(all_steps)}")
    print(f"  Monitor events: {len(monitor._events)}")

    for i, ev in enumerate(monitor._events):
        print(f"    Event {i}: {ev['event_type']}")

    # The key assertion: the crew completed without crashing
    assert result is not None, "Crew should produce output"
    print("  PASSED: Delegation crew completed without crashing the monitor")

    monitor.flush()
    monitor.close()
    return True


def test_3_monitor_survives_malformed_tool_event():
    """
    Directly simulate a malformed tool event to verify the defensive handling.
    This mimics what happens when gpt-4o-mini sends bad args to DelegateWorkToolSchema.
    """
    print("\n=== TEST 3: Monitor survives malformed tool events ===")

    from prysmai.integrations.crewai import PrysmCrewMonitor
    from unittest.mock import MagicMock

    monitor = PrysmCrewMonitor(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    # Simulate a tool event with an unserializable input (the DelegateWorkToolSchema bug)
    class MalformedInput:
        """Simulates a Pydantic model that fails to serialize."""
        def __str__(self):
            raise TypeError("Cannot convert DelegateWorkToolSchema to string")
        def __repr__(self):
            raise TypeError("Cannot repr DelegateWorkToolSchema")

    event = MagicMock()
    event.tool_name = "delegate_work"
    event.tool_input = MalformedInput()

    # This should NOT crash
    monitor._on_tool_start(event)
    assert len(monitor._events) == 1
    print(f"  Event captured: {monitor._events[0]['event_type']}")
    print(f"  Tool input: {monitor._events[0]['tool_input']}")

    # Simulate a tool error event
    error_event = MagicMock()
    error_event.tool_name = "delegate_work"
    error_event.error = "DelegateWorkToolSchema validation error: field 'coworker' is required"

    monitor._on_tool_error(error_event)
    assert len(monitor._events) == 2
    print(f"  Error event captured: {monitor._events[1]['event_type']}")
    print(f"  Error: {monitor._events[1]['error']}")

    monitor.close()
    print("  PASSED: Monitor survived malformed tool events")
    return True


def test_4_monitor_as_direct_callback():
    """
    Test the monitor's __call__ interface which is the fallback for
    CrewAI versions without the event bus.
    """
    print("\n=== TEST 4: Monitor __call__ interface (callback fallback) ===")

    from prysmai.integrations.crewai import PrysmCrewMonitor

    monitor = PrysmCrewMonitor(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    # Simulate various callback invocations
    monitor("task_started", {"task": "Analyze text for PII"})
    monitor("step_completed", {"output": "Found 3 PII entities"})
    monitor("task_completed", {"result": "Analysis complete"})

    assert len(monitor._events) == 3
    print(f"  Events captured: {len(monitor._events)}")
    for i, ev in enumerate(monitor._events):
        print(f"    Event {i}: {ev['event_type']} | data: {ev.get('data', 'N/A')}")

    # Flush to real Prysm endpoint
    monitor.flush()
    print(f"  Events after flush: {len(monitor._events)} (should be 0)")

    monitor.close()
    print("  PASSED: Monitor __call__ interface works correctly")
    return True


if __name__ == "__main__":
    results = {}
    tests = [
        test_1_crewai_with_step_and_task_callbacks,
        test_2_crewai_delegation_with_callbacks,
        test_3_monitor_survives_malformed_tool_event,
        test_4_monitor_as_direct_callback,
    ]

    for test_fn in tests:
        try:
            passed = test_fn()
            results[test_fn.__name__] = "PASSED" if passed else "FAILED"
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[test_fn.__name__] = f"FAILED: {e}"

    print("\n" + "=" * 60)
    print("BUG-003 TEST RESULTS")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")

    all_passed = all(s == "PASSED" for s in results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    if not all_passed:
        exit(1)
