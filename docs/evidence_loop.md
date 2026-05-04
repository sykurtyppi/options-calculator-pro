# Evidence Loop

## Purpose
The evidence loop measures whether recommendations improve over time. It is designed for auditability first, not for claiming live trading performance.

## Flow
1. A recommendation is written to the recommendation ledger.
2. The forward loop may create a paper trade linked by `recommendation_id`.
3. Exit detection records paper outcome fields.
4. Finalization updates calibration observations and structure priors.
5. Diagnostics summarize calibration state, provenance, provider health, and forward paper performance.

## What Is Empirical
Resolved paper outcomes are empirical observations of the paper workflow. They are not live fills and should not be interpreted as realized brokerage performance.

## What Is Diagnostic
Selector confidence, expected edge tier, and expected return signal are score-derived ranking diagnostics. They help compare setups, but they are not calibrated forecasts.

## Failure Modes
- Duplicate or missing recommendations weaken the learning loop.
- Stale earnings dates can contaminate outcome alignment.
- Research quotes may differ from executable bid/ask fills.
- Small sample sizes can make apparent performance unstable.

## Validation Approach
The system tests idempotency, provenance persistence, recommendation-to-outcome linkage, calibration diagnostics, and paper forward-performance aggregation.
