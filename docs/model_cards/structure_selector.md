# Model Card: StructureSelector

## Purpose
`StructureSelector` turns scorecards into a conservative decision object: best structure, recommendation tier, confidence, thesis, risks, and alternatives.

## Inputs
VolSnapshot and all StructureScorecards for the event.

## Outputs
Recommendation tier, selected structure, decision quality, edge tier, primary thesis, risks, why-this-structure, why-not-others, runner-ups, passed-over structures, and data-quality fields.

## Decision Rule
Eligible structures are ranked by composite score. The selector decides on the highest-ranked structure that clears the card-level hard gates. A structure that fails only economic gates (`non_positive_edge`, `execution_penalty_high`, `execution_cost_dominates_edge`) is passed over and recorded in `passed_over_structures`; its score is not used in the chosen structure's score gap. A structure with a non-finite input stops the walk and the event is No Trade (fail closed), as do snapshot-level gates (data quality). If every structure is passed over, the decision is made on the top-ranked structure exactly as before, so its gates are the binding gates. Weak history, watch and promotion gates apply to the structure chosen.

## Limitations
It does not execute trades, size positions, or produce calibrated probabilities. Confidence is decision quality, not win probability.

## Known Failure Modes
Close score gaps, low-quality data, stale earnings dates, thin walk-forward evidence, or high execution friction can make recommendations unstable.

## Data Dependencies
Scorecards, snapshot data quality, walk-forward evidence, and structure eligibility.

## Validation Approach
Tests cover clear winners, close scores, negative edge, execution failure, conflicting signals, deterministic output, and passing over a rejected winner (`tests/unit/test_services/test_selector_pass_over.py`).

## Do Not Infer
Do not infer financial advice, guaranteed edge, or a precise return forecast from selector output.
