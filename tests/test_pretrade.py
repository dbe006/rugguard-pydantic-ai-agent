"""Tests for the pre-trade check tool, framework-agnostic path + Pydantic AI surface.

No LLM key needed — the Pydantic AI surface is exercised via `FunctionModel`
which short-circuits to a known tool-call without touching any LLM provider.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from rugguard_pydantic_ai_agent import (
    DecisionCache,
    PreTradeCheckResult,
    pretrade_check_async,
    register_rugguard_tool,
)
from rugguard_pydantic_ai_agent.pretrade import PreTradeCheckError


def _canned_response(
    *,
    policy_recommendation: str = "allow",
    risk_score: int = 12,
    verdict: str = "safe",
    intended_trade_usd: float = 250.0,
) -> dict[str, Any]:
    return {
        "scan_id": "test-01",
        "chain": "base",
        "contract": "0xABC",
        "policy_recommendation": policy_recommendation,
        "policy": "balanced",
        "risk_score": risk_score,
        "verdict": verdict,
        "confidence": "high",
        "reason": [],
        "max_suggested_exposure_usd": intended_trade_usd
        if policy_recommendation == "allow"
        else (intended_trade_usd * 0.2 if policy_recommendation == "caution" else 0.0),
        "intended_trade_usd": intended_trade_usd,
        "scanned_at": "2026-05-17T12:00:00Z",
        "cache_hit": False,
        "disclaimer": "policy_recommendation is not an order ; etc.",
        "signature": None,
        "key_fingerprint": None,
    }


# --- framework-agnostic async function ---


@pytest.mark.asyncio
async def test_pretrade_returns_typed_result_on_200():
    """Happy path: the function decodes the server JSON into a typed Pydantic model."""
    response = _canned_response(
        policy_recommendation="caution", risk_score=60, verdict="medium_risk"
    )

    async def fake_paid_post(*, url, json_body, **_kw):
        assert url.endswith("/v1/pretrade/check")
        assert json_body["chain"] == "base"
        assert json_body["intended_trade_usd"] == 250.0
        return 200, response

    with patch("rugguard_pydantic_ai_agent.pretrade.paid_post", new=fake_paid_post):
        result = await pretrade_check_async(
            chain="base",
            contract="0xABC",
            intended_trade_usd=250.0,
            private_key_hex="0x" + "ab" * 32,
        )

    assert isinstance(result, PreTradeCheckResult)
    assert result.policy_recommendation == "caution"
    assert result.risk_score == 60
    assert result.max_suggested_exposure_usd == 50.0  # 20% of 250


@pytest.mark.asyncio
async def test_pretrade_missing_credentials_returns_structured_error(monkeypatch):
    """No private key → typed PreTradeCheckError, not crash."""
    monkeypatch.delenv("RUGGUARD_X402_PRIVATE_KEY", raising=False)
    result = await pretrade_check_async(
        chain="base", contract="0xABC", intended_trade_usd=100.0
    )
    assert isinstance(result, PreTradeCheckError)
    assert result.error == "missing_credentials"


@pytest.mark.asyncio
async def test_pretrade_payment_error_surfaces_typed(monkeypatch):
    """A payment rejection bubbles up as PreTradeCheckError(payment_failed)."""
    monkeypatch.setenv("RUGGUARD_X402_PRIVATE_KEY", "0x" + "ab" * 32)

    from rugguard_pydantic_ai_agent.x402_pay import X402PaymentError

    async def failing(*, url, json_body, **_kw):
        raise X402PaymentError("payment_rejected:PAYMENT_VERIFY_FAILED")

    with patch("rugguard_pydantic_ai_agent.pretrade.paid_post", new=failing):
        result = await pretrade_check_async(
            chain="base", contract="0xABC", intended_trade_usd=100.0
        )

    assert isinstance(result, PreTradeCheckError)
    assert result.error == "payment_failed"


@pytest.mark.asyncio
async def test_pretrade_non_200_surfaces_typed():
    """Server 4xx (e.g. INVALID_POLICY) → PreTradeCheckError(non_200, status=400)."""

    async def fake(*, url, json_body, **_kw):
        return 400, {"detail": {"code": "INVALID_POLICY", "error": "bad"}}

    with patch("rugguard_pydantic_ai_agent.pretrade.paid_post", new=fake):
        result = await pretrade_check_async(
            chain="base",
            contract="0xABC",
            intended_trade_usd=100.0,
            private_key_hex="0x" + "ab" * 32,
        )

    assert isinstance(result, PreTradeCheckError)
    assert result.error == "non_200"
    assert result.status == 400


# --- DecisionCache ---


@pytest.mark.asyncio
async def test_cache_hit_skips_payment():
    """Second call for the same (chain, contract) inside the TTL window
    returns the cached result with cache_hit=True and never calls paid_post."""
    cache = DecisionCache(ttl_seconds=300)
    response = _canned_response()

    calls: list[int] = []

    async def fake(*, url, json_body, **_kw):
        calls.append(1)
        return 200, response

    with patch("rugguard_pydantic_ai_agent.pretrade.paid_post", new=fake):
        a = await pretrade_check_async(
            chain="base",
            contract="0xABC",
            intended_trade_usd=250.0,
            private_key_hex="0x" + "ab" * 32,
            cache=cache,
        )
        b = await pretrade_check_async(
            chain="base",
            contract="0xABC",
            intended_trade_usd=250.0,
            private_key_hex="0x" + "ab" * 32,
            cache=cache,
        )

    assert len(calls) == 1, "paid_post should be called exactly once across two pretrade calls"
    assert isinstance(a, PreTradeCheckResult)
    assert isinstance(b, PreTradeCheckResult)
    assert a.cache_hit is False
    assert b.cache_hit is True


# --- Pydantic AI tool surface ---


def test_register_rugguard_tool_smoke():
    """Smoke: registering the tool on a real Agent shouldn't raise, and the
    Agent should be able to run end-to-end with TestModel + mocked paid_post.

    TestModel auto-calls every registered tool with synthesized args, so this
    test pins (a) the registration doesn't error and (b) the tool signature is
    callable by the Pydantic AI machinery.
    """
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent: Agent[None, str] = Agent(model=TestModel(), system_prompt="test")
    register_rugguard_tool(agent, policy="balanced", private_key_hex="0x" + "ab" * 32)

    # Mock paid_post so the tool can complete without touching the network.
    async def fake_paid_post(*, url, json_body, **_kw):
        return 200, _canned_response()

    with patch("rugguard_pydantic_ai_agent.pretrade.paid_post", new=fake_paid_post):
        result = agent.run_sync("Should I buy 0xABC on base for $100?")

    # TestModel returns a synthesized string; we just verify the run completes
    # without raising (the tool was wired correctly).
    output = getattr(result, "output", None) or getattr(result, "data", None)
    assert output is not None


# --- v0.1.1 security batch 1: https + max_amount_usdc regression tests ---


@pytest.mark.asyncio
async def test_pretrade_rejects_plaintext_api_url():
    """A plaintext api_url leaks trade intent + lets a MITM tamper the
    policy_recommendation. v0.1.1 refuses non-https unless loopback."""
    result = await pretrade_check_async(
        chain="base",
        contract="0xABC",
        intended_trade_usd=100.0,
        private_key_hex="0x" + "ab" * 32,
        api_url="http://attacker.example",
    )
    assert isinstance(result, PreTradeCheckError)
    assert result.error == "request_failed"
    assert "https" in result.message.lower()


@pytest.mark.asyncio
async def test_pretrade_allows_loopback_http_for_dev():
    """Dev against a local RugGuard should still work over http://localhost."""

    async def fake(*, url, json_body, **_kw):
        assert url.startswith("http://localhost"), url
        return 200, _canned_response()

    with patch("rugguard_pydantic_ai_agent.pretrade.paid_post", new=fake):
        result = await pretrade_check_async(
            chain="base",
            contract="0xABC",
            intended_trade_usd=100.0,
            private_key_hex="0x" + "ab" * 32,
            api_url="http://localhost:8000",
        )
    assert isinstance(result, PreTradeCheckResult)


@pytest.mark.asyncio
async def test_pretrade_passes_max_amount_usdc_to_paid_post():
    """v0.1.1 plumbs max_amount_usdc to paid_post so callers can refuse
    a 402 advertising more than the expected price."""
    captured: dict = {}

    async def fake(*, url, json_body, private_key_hex, max_amount_usdc=None, **_kw):
        captured["max_amount_usdc"] = max_amount_usdc
        return 200, _canned_response()

    with patch("rugguard_pydantic_ai_agent.pretrade.paid_post", new=fake):
        await pretrade_check_async(
            chain="base",
            contract="0xABC",
            intended_trade_usd=100.0,
            private_key_hex="0x" + "ab" * 32,
            max_amount_usdc=0.02,
        )

    assert captured["max_amount_usdc"] == 0.02
