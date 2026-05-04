# Earnings Volatility Decision Engine Architecture

## Purpose
This project is a research-grade event-volatility decision system. It separates market-state measurement from structure scoring, selection, diagnostics, and feedback-loop learning.

## Layered Architecture
1. `VolSnapshot` computes the canonical event-volatility state for one symbol and earnings event.
2. `StructureScorecard` evaluates supported option structures independently from the same snapshot.
3. `StructureSelector` ranks eligible structures and can abstain.
4. `RecommendationLedger` records every recommendation and its evidence surface.
5. `OutcomeRecorder` tracks forward paper trades and resolved outcomes.
6. Diagnostics services inspect data quality, provider health, calibration state, and forward performance.

## Main Data Flow
Screener or single-ticker analysis builds a snapshot, scorecards, and selector output. The ledger persists the decision. Forward paper trades link to the ledger by `recommendation_id`. Resolved outcomes update calibration and structure priors, then diagnostics report what changed.

## Trust Boundaries
The UI is decision-first and intentionally compact. Detailed explanations live in model cards and diagnostics docs. Numeric expected-return fields are score-derived diagnostics unless explicitly labeled as empirical paper outcomes.

## Production Caveats
This is not an execution engine, broker, or financial adviser. Quotes may be delayed, stale, or research-quality only. Live deployment requires stronger provider SLAs, audit logging, auth, monitoring, and execution-grade fill validation.
