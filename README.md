# rugguard-pydantic-ai-agent

[Pydantic AI](https://ai.pydantic.dev) integration kit for [RugGuard](https://rugguard.redfleet.fr). Add a typed `pretrade_check` tool to your AI trading agent in ~3 lines of code. Every call returns a Pydantic model with a prescriptive `policy_recommendation` of `block | caution | allow`, plus a clamped `max_suggested_exposure_usd` and a signed JSON report (Ed25519). Pay-per-call via x402 micropayments on Base mainnet.

## Why

If your agent buys tokens, it should run a pre-trade check first. RugGuard wraps 14 heuristics on Base + 5 on Solana SPL into a single $0.01 USDC call. The response is a typed Pydantic model with everything an LLM needs to make a sane sizing decision.

Two surfaces in this kit:

- **`pretrade_check_async(...)`** — a framework-agnostic async function. Call it directly from any runtime.
- **`register_rugguard_tool(agent, ...)`** — a one-liner that registers the check as a typed Pydantic AI tool on your existing `Agent`. The LLM sees the tool, calls it with structured args, and consumes the typed response natively.

## Install

```bash
pip install rugguard-pydantic-ai-agent
```

## 30-second tour

```python
from pydantic_ai import Agent
from rugguard_pydantic_ai_agent import register_rugguard_tool, DecisionCache

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You are a careful crypto trading assistant. Always call "
        "pretrade_check before recommending any buy. If it returns "
        "'block', refuse. If 'caution', downsize to "
        "max_suggested_exposure_usd. If 'allow', proceed."
    ),
)

# Three lines. Done. The LLM now sees a typed `pretrade_check` tool.
register_rugguard_tool(agent, policy="balanced", cache=DecisionCache())

result = agent.run_sync(
    "Should I buy $250 of 0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed on Base?"
)
print(result.output)
```

The LLM will call `pretrade_check(chain="base", contract="0x4ed4E862...", intended_trade_usd=250.0)` automatically before answering. The tool's response is a `PreTradeCheckResult` Pydantic model so the LLM can reason over typed fields, not JSON strings.

## Run the demo (no LLM, no network)

```bash
pip install rugguard-pydantic-ai-agent
rugguard-pydantic-ai-demo --demo
```

Walks through 3 canned scenarios (safe USDC, fresh memecoin, drained pool) so you can see what the tool returns without paying or wiring up an LLM.

## Run the live demo (real LLM + real $0.01 payment)

```bash
# 1. Get a funded x402 wallet (Base mainnet, ≥ $0.05 USDC)
#    Generate one with: python -m rugguard_mcp init  (from rugguard-mcp)
export RUGGUARD_X402_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HEX
export OPENAI_API_KEY=sk-YOUR_KEY

# 2. Run
rugguard-pydantic-ai-demo --live
```

The agent will reason about the prompt, call `pretrade_check` (which pays $0.01 USDC under the hood via x402), and respond with a sized recommendation.

## What `pretrade_check` returns

```python
class PreTradeCheckResult(BaseModel):
    scan_id: str
    chain: str
    contract: str
    policy_recommendation: Literal["block", "caution", "allow"]
    policy: Literal["conservative", "balanced", "aggressive"]
    risk_score: int                       # 0-100
    verdict: Literal["safe", "low_risk", "medium_risk", "high_risk", "critical", "uncertain"]
    confidence: Literal["high", "medium", "low", "insufficient_data"]
    reason: list[PreTradeFlag]            # top 3 flags, severity-ordered
    max_suggested_exposure_usd: float     # 100% if allow, 20% if caution, 0 if block
    intended_trade_usd: float
    scanned_at: str
    disclaimer: str                       # MANDATORY — see below
    signature: str | None                 # Ed25519 base64 (when configured)
    key_fingerprint: str | None           # routing identifier
```

## Policy modes

| Policy | Blocks at | Cautions at | Allows below |
|---|---|---|---|
| `conservative` | score ≥ 51 (medium_risk) | score 26-50 | score ≤ 25 |
| `balanced` *(default)* | score ≥ 71 (high_risk)   | score 51-70 | score ≤ 50 |
| `aggressive`           | score ≥ 91 (critical)    | score 71-90 | score ≤ 70 |

An `uncertain` verdict (sparse data) returns `caution` in all modes. Absence of evidence is not evidence of safety.

## Signed reports

When the deployment has Ed25519 signing configured (production rugguard.redfleet.fr does as of 2026-05-17, fingerprint `a0c71156d8747078`), the response carries `signature` and `key_fingerprint` fields. Verify offline:

```bash
pip install rugguard-verify
# inside an agent that has a result:
echo "$result_json" | rugguard-verify --report -
```

The `disclaimer` field is **inside** the signed canonical bytes. Stripping or rewriting it breaks signature verification by design.

## Safety

This kit is intentionally minimal (~300 LOC across all modules) so you can read it end-to-end before forking. It is **not** spend-capped. For production use:

- Install [`rugguard-mcp`](https://pypi.org/project/rugguard-mcp/) and import its `x402_client.paid_post` instead — that ships session caps + 24h caps + asset whitelist + EIP-712 domain enforcement.
- Add your own monitoring + retry policy + circuit breaker.
- Use a dedicated x402 wallet, funded only with the USDC you are willing to spend.

The asset whitelist IS enforced in this kit (USDC on Base / Base Sepolia only). A malicious 402 trying to drain a different EIP-3009 token in your wallet is rejected before signing.

## How `pretrade_check` works under the hood

1. Agent calls `pretrade_check(chain, contract, intended_trade_usd)`.
2. Kit checks the in-memory `DecisionCache`. If hit, returns the cached result with `cache_hit=True`. No payment.
3. Otherwise: POST `https://rugguard.redfleet.fr/v1/pretrade/check` with the body. Server returns `402 Payment Required` with x402 spec body.
4. Kit signs an EIP-3009 `TransferWithAuthorization` for $0.01 USDC to RugGuard's receiving wallet. Retries the POST with `X-Payment` header.
5. Server settles via Coinbase CDP facilitator, returns `200` with the typed response + signature.
6. Kit parses into `PreTradeCheckResult`, caches it, returns to the agent.

The whole round trip is ~300-500ms on a cache miss, ~1ms on a cache hit.

## Self-host / testnet

```python
register_rugguard_tool(
    agent,
    api_url="https://my-rugguard.example.com",  # or http://localhost:8000 for dev
)
```

The kit also reads `RUGGUARD_API_URL` from the environment if no `api_url` argument is passed.

## License

MIT. See [LICENSE](LICENSE).

## See also

- [RugGuard](https://rugguard.redfleet.fr) — the pre-trade safety API
- [`rugguard-mcp`](https://pypi.org/project/rugguard-mcp/) — MCP server for Claude Desktop / Cursor / LangGraph
- [`rugguard-verify`](https://pypi.org/project/rugguard-verify/) — stand-alone Ed25519 signed-report verifier
- [Pydantic AI](https://ai.pydantic.dev) — the agent framework this kit plugs into
