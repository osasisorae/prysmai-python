"""
BUG-001 Real Integration Test
==============================
Tests that PrysmCallbackHandler handles serialized=None correctly
when used with real LangChain chains through the live Prysm proxy.

This is the exact scenario from the stress test report:
  - LangChain RunnableSequence passes serialized=None to on_chain_start
  - Before the fix, this caused: AttributeError: 'NoneType' object has no attribute 'get'
"""

import os
import traceback

PRYSM_API_KEY = os.environ["PRYSM_API_KEY"]
PRYSM_BASE_URL = os.environ.get("PRYSM_BASE_URL", "https://prysmai.io/api/v1")


def test_1_langchain_chain_with_callback():
    """
    Run a real LangChain chain (ChatOpenAI → StrOutputParser) with PrysmCallbackHandler.
    This triggers RunnableSequence which passes serialized=None to on_chain_start.
    """
    print("\n=== TEST 1: LangChain chain with PrysmCallbackHandler ===")

    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from prysmai.integrations.langchain import PrysmCallbackHandler

    # Create handler pointing at real Prysm
    handler = PrysmCallbackHandler(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    # Create LLM routed through Prysm proxy
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=PRYSM_API_KEY,
        openai_api_base=PRYSM_BASE_URL,
        max_tokens=30,
    )

    # Build a chain — this creates a RunnableSequence which triggers the bug
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Reply in one sentence."),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # Invoke with the callback handler — this is where serialized=None was crashing
    result = chain.invoke(
        {"input": "What is 2+2?"},
        config={"callbacks": [handler]},
    )

    print(f"  Chain output: {result}")
    print(f"  Events captured: {len(handler._events)}")
    print(f"  Run map entries: {len(handler._run_map)}")

    # Verify events were captured (not crashed)
    assert isinstance(result, str) and len(result) > 0, "Chain should produce output"
    print("  PASSED: Chain completed without AttributeError")

    # Flush events to Prysm
    handler.flush()
    print(f"  Events after flush: {len(handler._events)} (should be 0)")
    handler.close()
    return True


def test_2_langchain_direct_llm_call():
    """
    Direct LLM call (no chain) — this should work with normal serialized dict.
    Verifies the fix doesn't break normal flow.
    """
    print("\n=== TEST 2: Direct LLM call with PrysmCallbackHandler ===")

    from langchain_openai import ChatOpenAI
    from prysmai.integrations.langchain import PrysmCallbackHandler

    handler = PrysmCallbackHandler(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=PRYSM_API_KEY,
        openai_api_base=PRYSM_BASE_URL,
        max_tokens=20,
    )

    result = llm.invoke(
        "Say 'test passed' in exactly two words.",
        config={"callbacks": [handler]},
    )

    print(f"  LLM output: {result.content}")
    print(f"  Events captured: {len(handler._events)}")

    assert result.content is not None and len(result.content) > 0
    assert len(handler._events) >= 1, "Should have at least one event"

    # Check the event has proper structure
    event = handler._events[0]
    print(f"  Event type: {event['event_type']}")
    print(f"  Model: {event.get('model', 'N/A')}")
    print(f"  Latency: {event.get('latency_ms', 'N/A')}ms")

    handler.flush()
    handler.close()
    print("  PASSED: Direct LLM call works correctly")
    return True


def test_3_manually_trigger_none_serialized():
    """
    Manually call on_chain_start with serialized=None to directly verify the fix.
    This is the exact code path that was crashing.
    """
    print("\n=== TEST 3: Manual on_chain_start(serialized=None) ===")

    import uuid
    from prysmai.integrations.langchain import PrysmCallbackHandler

    handler = PrysmCallbackHandler(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    run_id = uuid.uuid4()

    # This was the exact crash — serialized=None
    handler.on_chain_start(
        serialized=None,
        inputs={"query": "test input"},
        run_id=run_id,
    )

    assert str(run_id) in handler._run_map, "Run should be recorded"
    assert handler._run_map[str(run_id)]["chain_type"] == "unknown", "Chain type should default to 'unknown'"
    print(f"  Run map entry: {handler._run_map[str(run_id)]}")

    # Complete the chain
    handler.on_chain_end(
        outputs={"result": "test output"},
        run_id=run_id,
    )

    assert len(handler._events) == 1, "Should have one chain_execution event"
    event = handler._events[0]
    assert event["event_type"] == "chain_execution"
    assert event["chain_type"] == "unknown"
    print(f"  Event: {event['event_type']} | chain_type: {event['chain_type']}")

    handler.close()
    print("  PASSED: on_chain_start(serialized=None) handled correctly")
    return True


def test_4_none_serialized_on_all_handlers():
    """
    Test serialized=None on all callback methods that accept it:
    on_llm_start, on_chat_model_start, on_chain_start, on_tool_start
    """
    print("\n=== TEST 4: serialized=None on all handler methods ===")

    import uuid
    from unittest.mock import MagicMock
    from prysmai.integrations.langchain import PrysmCallbackHandler

    handler = PrysmCallbackHandler(
        api_key=PRYSM_API_KEY,
        base_url=PRYSM_BASE_URL,
    )

    # on_llm_start with None
    run_id_1 = uuid.uuid4()
    handler.on_llm_start(serialized=None, prompts=["test"], run_id=run_id_1)
    assert str(run_id_1) in handler._run_map
    assert handler._run_map[str(run_id_1)]["model"] == "unknown"
    print("  on_llm_start(serialized=None) — OK")

    # on_chat_model_start with None
    run_id_2 = uuid.uuid4()
    mock_msg = MagicMock()
    mock_msg.type = "human"
    mock_msg.content = "Hello"
    handler.on_chat_model_start(serialized=None, messages=[[mock_msg]], run_id=run_id_2)
    assert str(run_id_2) in handler._run_map
    assert handler._run_map[str(run_id_2)]["model"] == "unknown"
    print("  on_chat_model_start(serialized=None) — OK")

    # on_chain_start with None
    run_id_3 = uuid.uuid4()
    handler.on_chain_start(serialized=None, inputs={"q": "test"}, run_id=run_id_3)
    assert str(run_id_3) in handler._run_map
    assert handler._run_map[str(run_id_3)]["chain_type"] == "unknown"
    print("  on_chain_start(serialized=None) — OK")

    # on_tool_start with None
    run_id_4 = uuid.uuid4()
    handler.on_tool_start(serialized=None, input_str="test query", run_id=run_id_4)
    assert str(run_id_4) in handler._run_map
    assert handler._run_map[str(run_id_4)]["tool_name"] == "unknown"
    print("  on_tool_start(serialized=None) — OK")

    handler.close()
    print("  PASSED: All handler methods handle serialized=None")
    return True


if __name__ == "__main__":
    results = {}
    tests = [
        test_1_langchain_chain_with_callback,
        test_2_langchain_direct_llm_call,
        test_3_manually_trigger_none_serialized,
        test_4_none_serialized_on_all_handlers,
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
    print("BUG-001 TEST RESULTS")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")

    all_passed = all(s == "PASSED" for s in results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    if not all_passed:
        exit(1)
