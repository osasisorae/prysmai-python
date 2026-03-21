"""
Shared Prysm connection configuration helpers.

These helpers keep proxy, MCP, governance, and framework integrations aligned
around the same auth and base URL model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PrysmConnectionConfig:
    """Canonical Prysm connection settings shared across SDK surfaces."""

    prysm_key: str
    base_url: str


def resolve_prysm_connection(
    *,
    client: Any = None,
    prysm_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> PrysmConnectionConfig:
    """
    Resolve Prysm auth and base URL from a client, explicit params, or env vars.

    Accepted client shape:
    - object with `prysm_key` and `base_url` attributes
    """
    if client is not None:
        resolved_key = getattr(client, "prysm_key", "") or ""
        resolved_url = getattr(client, "base_url", "") or ""
    else:
        resolved_key = prysm_key or os.environ.get("PRYSM_API_KEY", "")
        resolved_url = base_url or os.environ.get(
            "PRYSM_BASE_URL", "https://prysmai.io/api/v1"
        )

    if not resolved_key:
        raise ValueError(
            "Prysm API key is required. Pass a Prysm client, prysm_key=, or set PRYSM_API_KEY."
        )

    if not resolved_key.startswith("sk-prysm-"):
        raise ValueError(
            f"Invalid Prysm API key format. Expected 'sk-prysm-...' but got '{resolved_key[:12]}...'"
        )

    if not resolved_url:
        resolved_url = "https://prysmai.io/api/v1"

    return PrysmConnectionConfig(
        prysm_key=resolved_key,
        base_url=resolved_url,
    )
