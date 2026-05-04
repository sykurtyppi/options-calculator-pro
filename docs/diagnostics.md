# Diagnostics Guide

## Purpose
Diagnostics answer whether the engine is learning from usable evidence and whether decisions are being made from healthy data.

## Learning Diagnostics
Shows calibration phase, observation counts, data provenance, structure priors, and learning-health warnings.

## Ledger Diagnostics
Shows recent recommendations, stale-source flags, data quality, selector explanation, quote provenance, and paper trade linkage.

## Data-Quality Diagnostics
Aggregates stale earnings sources, low-quality recommendations, missing option-chain evidence, and provider/source breakdowns.

## Provider Telemetry
Records provider request health, failure categories, latency, fallback use, and stale-cache use. Telemetry is best-effort and must never break analysis.

## Forward Performance
Summarizes resolved paper/research outcomes by structure, score bucket, stale-source state, quality tier, and symbol.

## What Users Should Not Infer
Diagnostics are observability surfaces. They do not guarantee future returns, execution quality, or strategy profitability.
