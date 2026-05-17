"""Pre-trade safety as a Pydantic AI tool.

The kit exports two surfaces:

1. **Framework-agnostic** — `pretrade_check_async(...)`: a typed async
   function any runtime can call. Returns a `PreTradeCheckResult` Pydantic
   model.

2. **Pydantic AI tool** — `register_rugguard_tool(agent, ...)`: registers
   the pretrade check as a typed tool on an existing `Agent`. The LLM
   sees a function it can call before each trade, with all the structured
   args (chain, contract, intended_trade_usd, policy) and a typed return.

The Pydantic AI surface is a thin wrapper over the agnostic function —
the typing flows directly because RugGuard's API is Pydantic-native.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from rugguard_pydantic_ai_agent.cache import DecisionCache
from rugguard_pydantic_ai_agent.x402_pay import X402PaymentError, paid_post

DEFAULT_API_URL = "https://rugguard.redfleet.fr"

Policy = Literal["conservative", "balanced", "aggressive"]
PolicyRecommendation = Literal["block", "caution", "allow"]
Verdict = Literal["safe", "low_risk", "medium_risk", "high_risk", "critical", "uncertain"]
Confidence = Literal["high", "medium", "low", "insufficient_data"]


class PreTradeFlag(BaseModel):
    """One flag in the prescriptive `reason` list."""

    code: str = Field(description="Stable flag identifier, e.g. OWNER_NOT_RENOUNCED.")
    severity: str = Field(description="low | medium | high | critical.")


class PreTradeCheckResult(BaseModel):
    """Typed RugGuard /v1/pretrade/check response.

    Mirrors the server-side Pydantic schema in `rugguard.api.schemas.
    PreTradeCheckResponse`. The signature + key_fingerprint fields are
    populated when the deployment has Ed25519 signing configured; null
    in the pre-deploy state. Verify offline with `rugguard-verify`
    (`pip install rugguard-verify`).
    """

    scan_id: str
    chain: str
    contract: str
    policy_recommendation: PolicyRecommendation
    policy: Policy
    risk_score: int = Field(ge=0, le=100)
    verdict: Verdict
    confidence: Confidence
    reason: list[PreTradeFlag] = Field(
        description="Top 3 flags that drove the recommendation, severity-ordered."
    )
    max_suggested_exposure_usd: float = Field(
        ge=0,
        description=(
            "Clamped trade size. allow → 100% of intended ; "
            "caution → 20% ; block → 0."
        ),
    )
    intended_trade_usd: float
    scanned_at: str
    cache_hit: bool = False
    disclaimer: str = Field(
        description=(
            "Mandatory legal disclaimer. Stripping or rewriting this field "
            "breaks signature verification by design — it is inside the "
            "signed canonical bytes."
        ),
    )
    signature: str | None = None
    key_fingerprint: str | None = None


class PreTradeCheckError(BaseModel):
    """Returned to the LLM when the call fails. Structured so the agent
    can branch (retry, skip, abort) without parsing strings."""

    error: Literal["payment_failed", "non_200", "request_failed", "missing_credentials"]
    message: str
    status: int | None = None


async def pretrade_check_async(
    *,
    chain: str,
    contract: str,
    intended_trade_usd: float,
    policy: Policy = "balanced",
    private_key_hex: str | None = None,
    api_url: str | None = None,
    cache: DecisionCache | None = None,
) -> PreTradeCheckResult | PreTradeCheckError:
    """Call RugGuard /v1/pretrade/check.

    Args:
        chain: `base` or `solana`.
        contract: Token address (EVM 0x… or Solana base58).
        intended_trade_usd: Trade size in USD.
        policy: Agent risk tolerance ; default `balanced`. `uncertain`
            verdicts return `caution` regardless of policy.
        private_key_hex: EOA private key for the x402 payer. Falls back
            to env var `RUGGUARD_X402_PRIVATE_KEY` if None.
        api_url: Override the default RugGuard endpoint (for self-hosting
            or testnet). Falls back to env var `RUGGUARD_API_URL`.
        cache: Optional `DecisionCache` instance. Hits short-circuit
            without paying, so the agent can call the check multiple
            times during a single trade decision without double-billing.

    Returns:
        `PreTradeCheckResult` on success. `PreTradeCheckError` on
        recoverable failure (missing creds, payment rejected, non-200).
        The agent can pattern-match the `error` field to branch.
    """
    pk = private_key_hex or os.environ.get("RUGGUARD_X402_PRIVATE_KEY")
    if not pk:
        return PreTradeCheckError(
            error="missing_credentials",
            message=(
                "No x402 private key configured. Pass `private_key_hex=...` "
                "or set RUGGUARD_X402_PRIVATE_KEY in the environment. The "
                "wallet must hold USDC on Base (≥ $0.05 recommended)."
            ),
        )

    if cache is not None:
        cached = cache.get(chain, contract)
        if cached is not None:
            # Re-construct the typed model from the cached dict, flagging
            # cache_hit=True so the caller can decide if it matters.
            cached_with_flag = {**cached, "cache_hit": True}
            return PreTradeCheckResult(**cached_with_flag)

    base_url = (api_url or os.environ.get("RUGGUARD_API_URL") or DEFAULT_API_URL).rstrip("/")
    url = f"{base_url}/v1/pretrade/check"
    body = {
        "chain": chain,
        "contract": contract,
        "intended_trade_usd": intended_trade_usd,
        "policy": policy,
    }

    try:
        status, response = await paid_post(url=url, json_body=body, private_key_hex=pk)
    except X402PaymentError as exc:
        return PreTradeCheckError(error="payment_failed", message=str(exc))
    except Exception as exc:
        return PreTradeCheckError(
            error="request_failed", message=f"{type(exc).__name__}: {exc}"
        )

    if status != 200:
        return PreTradeCheckError(
            error="non_200",
            status=status,
            message=f"server returned {status}: {response!s:.200}",
        )

    result = PreTradeCheckResult(**response)
    if cache is not None:
        cache.put(chain, contract, response)
    return result


# --- Pydantic AI surface ---


def register_rugguard_tool(
    agent: Any,
    *,
    policy: Policy = "balanced",
    private_key_hex: str | None = None,
    api_url: str | None = None,
    cache: DecisionCache | None = None,
    tool_name: str = "pretrade_check",
) -> None:
    """Register the RugGuard pre-trade check as a typed tool on a Pydantic AI Agent.

    After registering, the LLM driving `agent` can call the tool by name
    before any trade. The tool returns a typed `PreTradeCheckResult` (or
    `PreTradeCheckError`) which the LLM consumes natively.

    Usage:
        from pydantic_ai import Agent
        from rugguard_pydantic_ai_agent import register_rugguard_tool, DecisionCache

        agent = Agent("openai:gpt-4o-mini", system_prompt="You are a trading agent...")
        register_rugguard_tool(agent, policy="balanced", cache=DecisionCache())

        result = agent.run_sync("Should I buy 250 USDC of 0xABC on Base?")

    The `policy`, `private_key_hex`, `api_url`, and `cache` are bound at
    registration time — the LLM never sees them, so it cannot accidentally
    or maliciously change them.

    Args:
        agent: The Pydantic AI `Agent` instance.
        policy: Risk policy locked at registration. The LLM cannot override.
        private_key_hex: x402 payer key. Defaults to env var.
        api_url: Override RugGuard endpoint (self-host / testnet).
        cache: Optional DecisionCache for de-duplication.
        tool_name: Override the tool name surfaced to the LLM.
    """

    @agent.tool_plain(name=tool_name)
    async def _rugguard_pretrade_check(
        chain: Annotated[str, "Chain id: 'base' or 'solana'."],
        contract: Annotated[str, "Token contract address (EVM 0x… or Solana base58)."],
        intended_trade_usd: Annotated[
            float, "Intended trade size in USD. Must be > 0 and ≤ 1B."
        ],
    ) -> PreTradeCheckResult | PreTradeCheckError:
        """RugGuard pre-trade safety check. Call this BEFORE any buy. Returns a
        prescriptive policy_recommendation (block | caution | allow) + a
        clamped max_suggested_exposure_usd. Pays $0.01 USDC via x402."""
        return await pretrade_check_async(
            chain=chain,
            contract=contract,
            intended_trade_usd=intended_trade_usd,
            policy=policy,
            private_key_hex=private_key_hex,
            api_url=api_url,
            cache=cache,
        )
