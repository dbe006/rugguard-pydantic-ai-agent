"""Unit tests for the in-memory TTL DecisionCache."""

from __future__ import annotations

import time
from unittest.mock import patch

from rugguard_pydantic_ai_agent.cache import DecisionCache


def test_cache_hit_within_ttl():
    c = DecisionCache(ttl_seconds=60)
    c.put("base", "0xABC", {"score": 12})
    assert c.get("base", "0xABC") == {"score": 12}


def test_cache_miss_after_ttl():
    """Past the TTL window the entry is treated as missing."""
    c = DecisionCache(ttl_seconds=1)
    c.put("base", "0xABC", {"score": 12})
    with patch(
        "rugguard_pydantic_ai_agent.cache.time.monotonic",
        return_value=time.monotonic() + 100,
    ):
        assert c.get("base", "0xABC") is None


def test_cache_chain_normalization():
    """Chain is normalized to lowercase ; contract is NOT (Solana case-sensitive)."""
    c = DecisionCache(ttl_seconds=60)
    c.put("BASE", "0xABC", {"score": 1})
    # different case of chain → same entry
    assert c.get("base", "0xABC") == {"score": 1}
    # different case of contract → MISS (we don't want to bridge Solana into EVM)
    assert c.get("base", "0xabc") is None


def test_cache_clear():
    c = DecisionCache(ttl_seconds=60)
    c.put("base", "0xA", {"x": 1})
    c.put("solana", "Bar", {"x": 2})
    assert len(c) == 2
    c.clear()
    assert len(c) == 0
    assert c.get("base", "0xA") is None
