"""Pydantic AI integration kit for RugGuard.

Exports:

  - `register_rugguard_tool(agent, ...)`  — register the prescriptive
    pretrade_check tool on a Pydantic AI Agent. The LLM gets a typed
    tool it can call before any trade.

  - `pretrade_check_async(...)`           — framework-agnostic async function
    backing the tool. Call this directly from any non-Pydantic-AI runtime.

  - `PreTradeCheckResult`                  — typed Pydantic model of the
    response (policy_recommendation, risk_score, signature, etc.).

  - `DecisionCache`                        — in-process TTL cache for
    decisions, so back-to-back calls for the same (chain, contract)
    inside the window pay only once.

Kit footprint is ~300 LOC total. Read it end-to-end before forking.
For production-grade spend caps + asset whitelist + signed-report
verification, also install `rugguard-mcp` and `rugguard-verify`.
"""

from rugguard_pydantic_ai_agent.cache import DecisionCache
from rugguard_pydantic_ai_agent.pretrade import (
    PreTradeCheckResult,
    pretrade_check_async,
    register_rugguard_tool,
)
from rugguard_pydantic_ai_agent.x402_pay import X402PaymentError, paid_post

__all__ = [
    "DecisionCache",
    "PreTradeCheckResult",
    "X402PaymentError",
    "paid_post",
    "pretrade_check_async",
    "register_rugguard_tool",
]

__version__ = "0.1.0"
