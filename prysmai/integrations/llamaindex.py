"""
Prysm AI — LlamaIndex Integration

Provides PrysmSpanHandler that captures query engine operations, retrieval events,
LLM calls, and embedding calls from LlamaIndex pipelines.

Usage:
    from prysmai.integrations.llamaindex import PrysmSpanHandler
    from llama_index.core import Settings

    Settings.callback_manager.add_handler(PrysmSpanHandler(api_key="sk-prysm-..."))

Blueprint Reference: Section 9.3, page 28
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List, Optional

import httpx

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install it with: pip install prysmai[llamaindex]"
    )

from prysmai.context import prysm_context

logger = logging.getLogger("prysmai.integrations.llamaindex")


class PrysmSpanHandler(BaseCallbackHandler):
    """
    LlamaIndex callback handler that captures pipeline telemetry and sends it to Prysm.

    Captures:
        - LLM calls (model, prompt, completion, token usage)
        - Embedding calls (model, text count, dimensions)
        - Retrieval events (query, retrieved nodes, scores)
        - Query engine operations (query text, response)
        - Synthesis events (prompt template, response)
        - Chunking events (chunk count, sizes)

    All captured data is sent to the Prysm telemetry endpoint as structured events.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ):
        """
        Initialize the Prysm LlamaIndex span handler.

        Args:
            api_key: Prysm API key (sk-prysm-...). Falls back to PRYSM_API_KEY env var.
            base_url: Prysm proxy base URL. Falls back to PRYSM_BASE_URL or https://prysmai.io/api/v1.
            session_id: Optional session ID for grouping related queries.
            user_id: Optional user ID for attribution.
            metadata: Optional metadata dict attached to all events.
            event_starts_to_ignore: Event types to ignore on start.
            event_ends_to_ignore: Event types to ignore on end.
        """
        import os

        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

        self.api_key = api_key or os.environ.get("PRYSM_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("PRYSM_BASE_URL", "https://prysmai.io/api/v1")
        )
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.metadata = metadata or {}

        if not self.api_key:
            raise ValueError(
                "Prysm API key is required. Pass api_key= or set PRYSM_API_KEY env var."
            )

        # Internal state
        self._events: List[Dict[str, Any]] = []
        self._span_map: Dict[str, Dict[str, Any]] = {}  # event_id -> span data
        self._client = httpx.Client(timeout=30.0)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Called when a LlamaIndex event begins."""
        if not event_id:
            event_id = str(uuid.uuid4())

        self._span_map[event_id] = {
            "event_type": event_type.value if hasattr(event_type, "value") else str(event_type),
            "start_time": time.time(),
            "parent_id": parent_id,
            "payload": _extract_start_payload(event_type, payload),
        }

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when a LlamaIndex event ends."""
        span_data = self._span_map.pop(event_id, {})
        start_time = span_data.get("start_time", time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        event_type_str = event_type.value if hasattr(event_type, "value") else str(event_type)

        event = {
            "event_type": f"llamaindex_{event_type_str}",
            "event_id": event_id,
            "parent_id": span_data.get("parent_id"),
            "latency_ms": latency_ms,
            "start_payload": span_data.get("payload"),
            "end_payload": _extract_end_payload(event_type, payload),
        }

        self._buffer_event(event)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Called when a new trace begins (e.g., a new query)."""
        self._span_map.clear()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Called when a trace ends. Flush all buffered events."""
        self.flush()

    # ─── Event Buffering & Flushing ──────────────────────────────

    def _buffer_event(self, event: Dict[str, Any]) -> None:
        """Add event to buffer."""
        ctx = prysm_context.get()
        event["session_id"] = self.session_id
        event["user_id"] = self.user_id or ctx.user_id
        event["timestamp"] = time.time()
        event["prysm_metadata"] = {**self.metadata, **ctx.metadata}
        self._events.append(event)

    def flush(self) -> None:
        """Send buffered events to Prysm telemetry endpoint."""
        if not self._events:
            return

        events_to_send = self._events[:]
        self._events.clear()

        try:
            self._client.post(
                f"{self.base_url}/telemetry/events",
                json={
                    "source": "llamaindex",
                    "session_id": self.session_id,
                    "events": events_to_send,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to send telemetry events to Prysm: {e}")

    def close(self) -> None:
        """Flush remaining events and close the HTTP client."""
        self.flush()
        self._client.close()

    def __del__(self) -> None:
        """Ensure events are flushed on garbage collection."""
        try:
            self.flush()
        except Exception:
            pass


# ─── Payload Extraction Helpers ──────────────────────────────────

def _extract_start_payload(
    event_type: CBEventType, payload: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Extract relevant data from event start payload."""
    if not payload:
        return None

    result: Dict[str, Any] = {}

    try:
        if event_type == CBEventType.LLM:
            # LLM call start: capture model, messages/prompt
            messages = payload.get(EventPayload.MESSAGES)
            if messages:
                result["messages"] = [
                    {
                        "role": getattr(m, "role", "unknown"),
                        "content": _truncate(
                            m.content if hasattr(m, "content") else str(m), 1000
                        ),
                    }
                    for m in (messages[:10] if isinstance(messages, list) else [messages])
                ]
            prompt = payload.get(EventPayload.PROMPT)
            if prompt:
                result["prompt"] = _truncate(str(prompt), 2000)
            serialized = payload.get(EventPayload.SERIALIZED)
            if serialized and isinstance(serialized, dict):
                result["model"] = serialized.get("model", "unknown")

        elif event_type == CBEventType.EMBEDDING:
            # Embedding call start
            chunks = payload.get(EventPayload.CHUNKS)
            if chunks:
                result["chunk_count"] = len(chunks) if isinstance(chunks, list) else 1
            serialized = payload.get(EventPayload.SERIALIZED)
            if serialized and isinstance(serialized, dict):
                result["model"] = serialized.get("model_name", "unknown")

        elif event_type == CBEventType.RETRIEVE:
            # Retrieval start: capture query
            query_str = payload.get(EventPayload.QUERY_STR)
            if query_str:
                result["query"] = _truncate(str(query_str), 1000)

        elif event_type == CBEventType.QUERY:
            # Query engine start
            query_str = payload.get(EventPayload.QUERY_STR)
            if query_str:
                result["query"] = _truncate(str(query_str), 1000)

        elif event_type == CBEventType.SYNTHESIZE:
            # Synthesis start
            query_str = payload.get(EventPayload.QUERY_STR)
            if query_str:
                result["query"] = _truncate(str(query_str), 1000)

        elif event_type == CBEventType.CHUNKING:
            # Chunking start
            chunks = payload.get(EventPayload.CHUNKS)
            if chunks and isinstance(chunks, list):
                result["input_chunk_count"] = len(chunks)

    except Exception as e:
        logger.debug(f"Error extracting start payload for {event_type}: {e}")

    return result if result else None


def _extract_end_payload(
    event_type: CBEventType, payload: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Extract relevant data from event end payload."""
    if not payload:
        return None

    result: Dict[str, Any] = {}

    try:
        if event_type == CBEventType.LLM:
            # LLM call end: capture response, token usage
            response = payload.get(EventPayload.RESPONSE)
            if response:
                if hasattr(response, "text"):
                    result["completion"] = _truncate(response.text, 2000)
                elif hasattr(response, "message"):
                    msg = response.message
                    result["completion"] = _truncate(
                        msg.content if hasattr(msg, "content") else str(msg), 2000
                    )
                # Token usage
                raw = getattr(response, "raw", None)
                if raw and hasattr(raw, "usage"):
                    usage = raw.usage
                    result["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                    result["completion_tokens"] = getattr(usage, "completion_tokens", 0)
                    result["total_tokens"] = getattr(usage, "total_tokens", 0)

        elif event_type == CBEventType.EMBEDDING:
            # Embedding end: capture dimensions
            chunks = payload.get(EventPayload.CHUNKS)
            if chunks and isinstance(chunks, list) and len(chunks) > 0:
                first = chunks[0]
                if isinstance(first, list):
                    result["dimensions"] = len(first)
                result["embedding_count"] = len(chunks)

        elif event_type == CBEventType.RETRIEVE:
            # Retrieval end: capture nodes
            nodes = payload.get(EventPayload.NODES)
            if nodes and isinstance(nodes, list):
                result["node_count"] = len(nodes)
                result["nodes"] = [
                    {
                        "score": getattr(n, "score", None),
                        "text_preview": _truncate(
                            getattr(n, "text", str(n)), 200
                        ),
                    }
                    for n in nodes[:5]
                ]

        elif event_type == CBEventType.QUERY:
            # Query end: capture response
            response = payload.get(EventPayload.RESPONSE)
            if response:
                result["response"] = _truncate(str(response), 2000)

        elif event_type == CBEventType.SYNTHESIZE:
            # Synthesis end: capture response
            response = payload.get(EventPayload.RESPONSE)
            if response:
                result["response"] = _truncate(str(response), 2000)

        elif event_type == CBEventType.CHUNKING:
            # Chunking end
            chunks = payload.get(EventPayload.CHUNKS)
            if chunks and isinstance(chunks, list):
                result["output_chunk_count"] = len(chunks)
                result["avg_chunk_size"] = (
                    sum(len(str(c)) for c in chunks) / len(chunks) if chunks else 0
                )

    except Exception as e:
        logger.debug(f"Error extracting end payload for {event_type}: {e}")

    return result if result else None


def _truncate(text: Optional[str], max_length: int = 500) -> Optional[str]:
    """Truncate text to max_length."""
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + "...[truncated]"
