# Model Card: StructureSelector

## Purpose
`StructureSelector` turns scorecards into a conservative decision object: best structure, recommendation tier, confidence, thesis, risks, and alternatives.

## Inputs
VolSnapshot and all StructureScorecards for the event.

## Outputs
Recommendation tier, selected structure, decision quality, edge tier, primary thesis, risks, why-this-structure, why-not-others, runner-ups, and data-quality fields.

## Limitations
It does not execute trades, size positions, or produce calibrated probabilities. Confidence is decision quality, not win probability.

## Known Failure Modes
Close score gaps, low-quality data, stale earnings dates, thin walk-forward evidence, or high execution friction can make recommendations unstable.

## Data Dependencies
Scorecards, snapshot data quality, walk-forward evidence, and structure eligibility.

## Validation Approach
Tests cover clear winners, close scores, negative edge, execution failure, conflicting signals, and deterministic output.

## Do Not Infer
Do not infer financial advice, guaranteed edge, or a precise return forecast from selector output.
