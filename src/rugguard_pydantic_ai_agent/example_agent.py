"""Example: a Pydantic AI agent that uses RugGuard before every trade.

Two run modes:

  rugguard-pydantic-ai-demo --demo         # no LLM, no network. Mocks the
                                           # x402 round trip and walks
                                           # through 3 sample scenarios.

  rugguard-pydantic-ai-demo --live         # real LLM (OPENAI_API_KEY) +
                                           # real x402 payment. Needs a
                                           # funded wallet via
                                           # RUGGUARD_X402_PRIVATE_KEY.

The demo mode is what most readers will run first. Copy this file as a
starting point for your own agent; the only thing you need to swap is
the LLM model string and your trade-execution function.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any
from unittest.mock import patch

from rugguard_pydantic_ai_agent.cache import DecisionCache
from rugguard_pydantic_ai_agent.pretrade import (
    PreTradeCheckResult,
    pretrade_check_async,
    register_rugguard_tool,
)

# --- Demo scenarios used by `--demo` mode (no network, no LLM) ---

# Each scenario is a (label, RugGuard response) pair. The response shape
# mirrors what /v1/pretrade/check returns when signing is configured.
_DEMO_SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    (
        "Established USDC (allow)",
        {
            "scan_id": "demo-01",
            "chain": "base",
            "contract": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "policy_recommendation": "allow",
            "policy": "balanced",
            "risk_score": 12,
            "verdict": "safe",
            "confidence": "high",
            "reason": [],
            "max_suggested_exposure_usd": 250.0,
            "intended_trade_usd": 250.0,
            "scanned_at": "2026-05-17T12:00:00Z",
            "cache_hit": False,
            "disclaimer": (
                "policy_recommendation is not an order; "
                "final responsibility remains with the agent. "
                "Signed reports are point-in-time attestations — check "
                "scanned_at and re-scan before relying on an old verdict "
                "(chain state changes). "
                "RugGuard is a data analytics tool, not a security "
                "guarantee. Best-effort heuristic scoring."
            ),
            "signature": "DEMO_SIG_BASE64",
            "key_fingerprint": "a0c71156d8747078",
        },
    ),
    (
        "Fresh memecoin, balanced policy (caution + downsize)",
        {
            "scan_id": "demo-02",
            "chain": "base",
            "contract": "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed",
            "policy_recommendation": "caution",
            "policy": "balanced",
            "risk_score": 62,
            "verdict": "medium_risk",
            "confidence": "medium",
            "reason": [
                {"code": "OWNER_NOT_RENOUNCED", "severity": "high"},
                {"code": "TOP10_CONCENTRATION_HIGH", "severity": "high"},
            ],
            "max_suggested_exposure_usd": 50.0,
            "intended_trade_usd": 250.0,
            "scanned_at": "2026-05-17T12:00:00Z",
            "cache_hit": False,
            "disclaimer": (
                "policy_recommendation is not an order; "
                "final responsibility remains with the agent. "
                "RugGuard is a data analytics tool, not a security guarantee."
            ),
            "signature": "DEMO_SIG_BASE64",
            "key_fingerprint": "a0c71156d8747078",
        },
    ),
    (
        "Drained pool + dev-concentrated (block)",
        {
            "scan_id": "demo-03",
            "chain": "base",
            "contract": "0xfc0482b1abd9da4a90a512305eeac472ffb88e1f",
            "policy_recommendation": "block",
            "policy": "balanced",
            "risk_score": 95,
            "verdict": "critical",
            "confidence": "high",
            "reason": [
                {"code": "TOP10_CONCENTRATION_HIGH", "severity": "critical"},
                {"code": "LP_INSUFFICIENT_LIQUIDITY", "severity": "critical"},
                {"code": "OWNER_NOT_RENOUNCED", "severity": "high"},
            ],
            "max_suggested_exposure_usd": 0.0,
            "intended_trade_usd": 250.0,
            "scanned_at": "2026-05-17T12:00:00Z",
            "cache_hit": False,
            "disclaimer": (
                "policy_recommendation is not an order; "
                "final responsibility remains with the agent. "
                "RugGuard is a data analytics tool, not a security guarantee."
            ),
            "signature": "DEMO_SIG_BASE64",
            "key_fingerprint": "a0c71156d8747078",
        },
    ),
]


def _print_result(label: str, result: PreTradeCheckResult) -> None:
    badge = {"allow": "[ALLOW]", "caution": "[CAUTION]", "block": "[BLOCK]"}[
        result.policy_recommendation
    ]
    print(f"\n=== {label} ===")
    print(f"  contract:        {result.chain}:{result.contract}")
    print(f"  verdict:         {result.verdict}  (score={result.risk_score})")
    print(f"  recommendation:  {badge}  (policy={result.policy})")
    print(f"  suggested cap:   ${result.max_suggested_exposure_usd:.2f} "
          f"(intended ${result.intended_trade_usd:.2f})")
    if result.reason:
        print("  top reasons:")
        for r in result.reason:
            print(f"    - {r.code}  ({r.severity})")
    if result.signature:
        print(f"  signed:          fingerprint={result.key_fingerprint}")
    print(f"  scan_id:         {result.scan_id}")


async def run_demo() -> int:
    """Walk through 3 canned scenarios. No network, no LLM, no payment."""
    print("RugGuard pre-trade safety demo — no LLM, no network, no payment.\n")
    print("This shows what the kit returns for representative tokens:")
    print(f"  1. {_DEMO_SCENARIOS[0][0]}")
    print(f"  2. {_DEMO_SCENARIOS[1][0]}")
    print(f"  3. {_DEMO_SCENARIOS[2][0]}")

    cache = DecisionCache(ttl_seconds=300)

    for label, mocked_response in _DEMO_SCENARIOS:
        # Mock the x402 paid_post at the level pretrade_check_async calls it.
        # The default-arg capture `_mocked=mocked_response` binds the current
        # loop iteration's value, so the closure doesn't capture by reference
        # (would otherwise B023 + all 3 scenarios would print the last response).
        async def fake_paid_post(
            *, url: str, json_body: dict, _mocked=mocked_response, **_kw: Any
        ) -> tuple[int, dict]:
            return 200, _mocked

        with patch(
            "rugguard_pydantic_ai_agent.pretrade.paid_post", new=fake_paid_post
        ):
            result = await pretrade_check_async(
                chain=mocked_response["chain"],
                contract=mocked_response["contract"],
                intended_trade_usd=mocked_response["intended_trade_usd"],
                policy="balanced",
                private_key_hex="0x" + "ab" * 32,  # mock — never sent over wire
                cache=cache,
            )
        if isinstance(result, PreTradeCheckResult):
            _print_result(label, result)
        else:
            print(f"\n=== {label} === ERROR: {result.error}: {result.message}")

    print("\nCache hit demo — second call for the same contract uses the cache:")
    async def fake_paid_post_must_not_fire(**_kw: Any) -> tuple[int, dict]:
        raise AssertionError("paid_post should NOT be called on a cache hit")
    with patch(
        "rugguard_pydantic_ai_agent.pretrade.paid_post",
        new=fake_paid_post_must_not_fire,
    ):
        result = await pretrade_check_async(
            chain="base",
            contract=_DEMO_SCENARIOS[0][1]["contract"],
            intended_trade_usd=100.0,
            policy="balanced",
            private_key_hex="0x" + "ab" * 32,
            cache=cache,
        )
    if isinstance(result, PreTradeCheckResult):
        print(f"  cache_hit={result.cache_hit}  (paid 0 USDC)")

    print("\nDone. Next: read src/rugguard_pydantic_ai_agent/pretrade.py "
          "and copy register_rugguard_tool into your own Pydantic AI agent.")
    return 0


async def run_live(*, model: str, prompt: str) -> int:
    """Real LLM + real x402 payment. Requires:

      - OPENAI_API_KEY (or whatever the chosen `model` needs)
      - RUGGUARD_X402_PRIVATE_KEY pointing to a Base-mainnet wallet
        funded with USDC (≥ $0.05 recommended).
    """
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # pragma: no cover
        print(f"pydantic-ai not installed: {exc}", file=sys.stderr)
        return 2

    if not os.environ.get("RUGGUARD_X402_PRIVATE_KEY"):
        print(
            "error: RUGGUARD_X402_PRIVATE_KEY is not set. The live demo pays "
            "$0.01 USDC per pretrade_check call. Generate a funded wallet at "
            "https://docs.coinbase.com/cdp/docs/x402 and export the private "
            "key.",
            file=sys.stderr,
        )
        return 2

    cache = DecisionCache(ttl_seconds=300)
    agent = Agent(
        model,
        system_prompt=(
            "You are a careful crypto trading assistant. Before recommending "
            "any token purchase, ALWAYS call the pretrade_check tool with "
            "the chain, contract address, and intended trade size in USD. "
            "If the tool returns policy_recommendation='block', refuse the "
            "trade. If 'caution', recommend down-sizing to "
            "max_suggested_exposure_usd. If 'allow', proceed. Always cite "
            "the top reasons from the tool's response."
        ),
    )
    register_rugguard_tool(agent, policy="balanced", cache=cache)

    print(f"User prompt: {prompt}\n")
    result = await agent.run(prompt)
    print("Agent response:")
    print(result.output if hasattr(result, "output") else result.data)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rugguard-pydantic-ai-demo",
        description=(
            "Demo of the RugGuard Pydantic AI integration kit. "
            "Run --demo for an offline walk-through ; --live for a real "
            "LLM + paid call against rugguard.redfleet.fr."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="(default) Offline walk-through of 3 canned scenarios. No LLM, no network.",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Real LLM + real x402 payment. Requires OPENAI_API_KEY + RUGGUARD_X402_PRIVATE_KEY.",
    )
    parser.add_argument(
        "--model",
        default="openai:gpt-4o-mini",
        help="Pydantic AI model string for --live mode. Default: openai:gpt-4o-mini.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "I'm considering buying $250 of token "
            "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed on Base. Should I?"
        ),
        help="User prompt for --live mode.",
    )
    args = parser.parse_args(argv)

    if args.live:
        return asyncio.run(run_live(model=args.model, prompt=args.prompt))
    return asyncio.run(run_demo())


if __name__ == "__main__":
    raise SystemExit(main())


# Tiny helper kept here (and not under tests/) so a curious dev reading
# example_agent.py end-to-end sees how the tool result decodes from JSON.
# Pydantic does the typing for free — this is just a doctest-style hint.
_example_response_doctest = json.dumps({"see": "_DEMO_SCENARIOS above"})
