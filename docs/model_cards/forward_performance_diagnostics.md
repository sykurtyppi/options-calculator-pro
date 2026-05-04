# Model Card: Forward Performance Diagnostics

## Purpose
Forward performance diagnostics summarize resolved paper/research outcomes linked to recommendations.

## Inputs
Recommendation ledger rows, outcome recorder rows, modeled return diagnostics at entry, realized paper return, realized IV expansion, data-quality fields, and stale-source flags.

## Outputs
Win/loss by structure, modeled-vs-realized paper return, no-trade count, stale-source comparison, quality-tier comparison, symbol counts, confidence buckets, recent resolved outcomes, and warnings.

## Limitations
Results are paper/research unless explicitly labeled live. They are observational and sample-size dependent.

## Known Failure Modes
Small samples, stale source data, paper quotes, missing recommendation linkage, or duplicated paper workflows can distort conclusions.

## Data Dependencies
Recommendation ledger, outcome store, quote provenance, and source-quality metadata.

## Validation Approach
Tests cover empty systems, structure aggregation, quality/stale comparisons, score buckets, recent outcomes, and paper/research warnings.

## Do Not Infer
Do not infer future profitability, execution quality, or broker-fill performance from paper diagnostics.
