# Repository Operating Rules

These rules apply to the entire repository.

## Environment and installation

- Work in an isolated checkout or worktree; do not modify a user's active checkout.
- Python must satisfy `>=3.11,<3.13`. Create a disposable environment and install with `uv pip sync requirements.lock`; `requirements.lock` is authoritative for tests.
- Frontend dependencies are installed with `npm ci` in `web/frontend/`; its `package-lock.json` is authoritative.
- Never persist API keys, cookies, tokens, account data, or raw credential-bearing responses in tests, logs, fixtures, commits, or audit ledgers.

## Canonical checks

- Python: `pytest -q`
- Golden product cases: `pytest -q tests/golden/test_frozen_product_cases.py tests/unit/test_web/test_analyze_single_ticker_golden.py`
- Frontend: `cd web/frontend && npm ci && npm run lint && npm run build`
- Run the CI workflow's security and policy checks before a merge claim; a subset does not clear the repository.

## Quantitative and data boundaries

- Preserve point-in-time semantics. No observation, quote, earnings date, outcome, or model fit may use information unavailable at `as_of`.
- Every recommendation must expose provider provenance, quote timestamps, quality gates, uncertainty, transaction-cost assumptions, and no-trade reasons.
- Frozen fixtures are the primary regression evidence. Mutable live APIs are supplemental and must never silently update expected outputs.
- Trials, exclusions, calibration sets, and forward results must be counted explicitly. Backtest or paper evidence is not live profitability evidence.
- Crossed markets, absent Greeks, stale earnings dates, sparse history, malformed values, partial chains, and provider outages must fail closed or degrade visibly.

## Product and architecture

- Establish the primary user and decision before recommending integration with PIVOT_QUANT.
- Prefer shared shell/design/platform services unless evidence shows the analytical engines share inputs, semantics, validation, release cadence, and failure policy.
- Do not publish, deploy, modify hosted credentials, or alter provider accounts without explicit authorization and verified rollback.
- Broad audits must satisfy `docs/AUDIT_DEFINITION_OF_DONE.md`; inventory is not completion.
