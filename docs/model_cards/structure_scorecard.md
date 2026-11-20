# Model Card: StructureScorecard

## Purpose
`StructureScorecard` evaluates each supported structure independently using the same `VolSnapshot`.

## Inputs
VolSnapshot fields, walk-forward prior data when available, execution/liquidity metrics, historical move ratios, term structure, and surface shape.

## Outputs
Eligibility, expected edge diagnostics, expected return signal, penalties, walk-forward evidence, composite score, and rationale bullets.

## Limitations
Scores are deterministic and interpretable, not machine-learned predictions. Expected values are score-derived diagnostics unless backed by separate empirical calibration.

## Known Failure Modes
Thin walk-forward history, stale priors, wide spreads, unpriced structures, or sparse surface points can make a score less reliable.

## Data Dependencies
VolSnapshot, structure prior store, historical walk-forward outputs, and current option-chain liquidity.

## Validation Approach
Tests cover structure differentiation, monotonic penalties, no-trade eligibility, determinism, and low-sample behavior.

Accuracy is measured by the walk-forward pricing harness (`docs/PRICING_HARNESS.md`): the archive is replayed point in time, the scorecards are given priors learned only from outcomes that exited before each entry (`build_structure_scorecards(..., priors=...)`, which then reads neither the report files nor the prior store), and each prediction is compared with the conservative-fill return of the same contracts at the T-1 exit.

## Do Not Infer
Do not infer that the highest score guarantees profit or that score differences are precise expected-return differences.

## Iron Condor Horizon
Every structure exits the session before earnings, so the iron condor is scored for that window: it never collects the post-earnings crush or faces the earnings move, and the crush classifier and earnings-move ratios do not enter its score. Under the event split's assumption (non-event implied vol = HAR forecast) the premium that decays before the exit is priced at fair value at best, so the snapshot signal tops out at zero; a positive condor edge can only come from real condor outcomes through the walk-forward prior. A hold-through-earnings condor would be a different strategy with its own exit rule.

## Execution Is Priced on the Booked Legs
When the snapshot is built from a chain, it records the exact legs each structure would book on the trade expiry (`structure_legs`, from `services/structure_legs.py`): the first expiry that spans the earnings reaction, and the same strike rules the forward loop books by. Each scorecard's execution penalty uses those legs' mean spread (OTM wings for strangles and condors, the back leg for calendars). The analysis resolves the trade and back expiries from the complete expiry listing and fetches those chains the way entry does (`ENTRY_STRIKE_LIMIT` strikes from the provider, the full chain from yfinance, the same provider-to-yfinance fallback); only legs from that fetch have coverage `complete`. With complete coverage a structure the chain cannot form is ineligible (`structure_legs_unavailable`), and discovery applies entry's opening gates (condor credit, calendar debit). Legs from the sampled analysis chain (narrow strike window, tenors chosen for IV interpolation) have coverage `sampled`: they neither price nor rule out a structure, and the near-term ATM spreads are used, as for a snapshot built without a chain. At entry, the forward loop records whether it booked the scored legs (`scored_legs_check` in the trade notes).
