# Edge Analysis V1 Structure Selector Spec (2026-04-20)

## Objective
Rebuild edge analysis from a hybrid crush/expansion gate stack into a single-name **earnings volatility structure selector**.

The new engine should answer:

- What is the market pricing into this earnings event?
- How does that compare with the stock's realized earnings behavior?
- Which supported option structure is historically best suited to this setup after costs?
- How confident is that recommendation?

This spec is intentionally scoped for **v1**:

- deterministic
- interpretable
- walk-forward calibrated
- cost-aware
- structure-aware

It is not a black-box ML spec.

## Product Thesis
Retail traders may pay for "institutional-style" volatility analysis, but the premium product is not the existence of Yang-Zhang or HAR-RV alone. Those estimators are public and reproducible.

The real product advantage is the combination of:

- earnings-event specific variance extraction
- term-structure and smile-state interpretation
- structure-specific trade selection
- transaction-cost-aware ranking
- confidence and data-quality controls
- clear explanations for why one structure is favored over another

## External Research Basis
- Yang-Zhang is a strong realized-volatility baseline for equities because it is drift-independent and handles opening jumps well.  
  Source: [Yang & Zhang (2000)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=229190)
- HAR-RV remains a practical and parsimonious realized-volatility forecasting framework.  
  Source: [Corsi (2009)](https://academic.oup.com/jfec/article/7/2/174/856522)
- Option-implied volatility contains incremental information about future realized volatility partly because it predicts scheduled news intensity.  
  Source: [Chen & Li (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0378426623002108)
- Earnings-event uncertainty is distinct from ordinary day-to-day volatility and should be modeled separately.  
  Source: [Dubinsky et al. (2019)](https://academic.oup.com/rfs/article/32/2/646/5001193)
- For pre-earnings long-vol trades, the relation between historical earnings move and implied earnings move is more informative than generic IV-vs-RV alone.  
  Source: [Milian (2023)](https://www.mdpi.com/2291876)
- Execution frictions materially affect whether option-market signals survive in implementable trading.  
  Source: [Govindaraj, Li, Zhao (2020)](https://econpapers.repec.org/RePEc:bla:jbfnac:v:47:y:2020:i:5-6:p:615-644)
- Combination forecasts often outperform relying on one volatility forecast family alone.  
  Source: [Becker & Clements (2008)](https://ideas.repec.org/a/eee/intfor/v24y2008i1p122-133.html)
- Strong pre-earnings smile concavity signals event risk, but some long-vol structures perform worse when the market overpays for that tail/gamma protection.  
  Source: [Alexiou et al. (2025)](https://academic.oup.com/rof/article/29/4/963/8079062)

## Core Design Principles
- One canonical feature snapshot per ticker/date.
- One scoring pass per supported structure.
- Cost-adjusted expectancy beats raw signal strength.
- Missing history should reduce confidence, not force contradictory hard rejections.
- The engine must be allowed to recommend `No Trade`.
- Signals must be monotonic or otherwise interpretable by design.
- Walk-forward structure history should drive calibration.
- Calendar analytics can survive, but only as one structure family among several.

## V1 Supported Structures
The structure universe should be deliberately small.

- `atm_straddle`
- `otm_strangle`
- `call_calendar`
- `put_calendar`

Rationale:

- All four already exist in the multi-structure backtest framework.
- They cover the main pre-earnings long-vol and event-premium expressions already explored in-repo.
- Expanding the universe before the selector is stable will make validation harder.

Out of scope for v1:

- diagonals
- broken-wing flies
- ratio spreads
- iron structures
- post-earnings short premium as a primary recommendation path

## Engine Contract
The new edge analysis should become a 3-layer pipeline:

1. `vol_snapshot`
2. `structure_scorecards`
3. `selector_output`

### Layer 1: Canonical Vol Snapshot
This layer computes the same feature state for all structures. It should not contain structure-specific gates.

Required snapshot fields for v1:

- `symbol`
- `as_of_date`
- `earnings_date`
- `release_timing`
- `days_to_earnings`
- `underlying_price`
- `data_quality`
- `option_source`
- `underlying_source`
- `price_staleness_minutes`
- `chain_staleness_minutes`

Realized-vol fields:

- `rv30_yang_zhang`
- `rv30_estimator`
- `rv_har_forecast`
- `rv_percentile_rank`
- `vol_regime_label`

Implied-vol fields:

- `iv30`
- `iv45`
- `near_term_dte`
- `near_term_atm_iv`
- `back_term_dte`
- `back_term_atm_iv`
- `near_back_iv_ratio`
- `term_structure_slope`

Event pricing fields:

- `near_term_implied_move_pct`
- `non_event_move_pct_har`
- `event_implied_move_pct`
- `event_move_share_of_total`

Historical earnings behavior fields:

- `historical_event_count`
- `historical_median_move_pct`
- `historical_avg_last4_move_pct`
- `historical_p90_move_pct`
- `historical_move_std_pct`
- `historical_move_anchor_pct`
- `historical_move_uncertainty_pct`
- `historical_vs_implied_move_ratio`
- `tail_vs_implied_move_ratio`

Surface-shape fields:

- `smile_curvature`
- `smile_concavity_flag`
- `smile_points`

Execution fields:

- `near_term_spread_pct`
- `near_term_liquidity_proxy`
- `atm_call_spread_pct`
- `atm_put_spread_pct`
- `atm_total_open_interest`
- `atm_total_volume`
- `liquidity_tier`

Derived relationship fields:

- `iv_rv_yz`
- `iv_rv_har`
- `cheapness_score`
- `event_risk_score`
- `execution_score`
- `timing_score`
- `data_quality_score`

### Snapshot Interpretation Rules
V1 should standardize these signal directions:

- Lower `iv_rv_yz` or `iv_rv_har` is generally better for pre-event long-vol entry, unless extremely low because of bad data.
- Higher `historical_vs_implied_move_ratio` is better for straddle/strangle style structures.
- Normal upward term structure can be favorable for long-vega expansion trades.
- Inverted front-end term structure is not automatically good or bad; it is structure-dependent.
- Strong smile concavity is a risk-price signal, not a universal buy signal.
- Wide spreads and low liquidity are penalties across all structures.

## Layer 2: Structure Scorecards
Each supported structure receives its own scorecard built from the same snapshot.

Each scorecard should contain:

- `structure`
- `eligible`
- `eligibility_flags`
- `expected_edge_pct`
- `expected_return_pct`
- `expected_iv_contribution_pct`
- `expected_move_fit_score`
- `theta_drag_penalty`
- `execution_penalty`
- `crowding_penalty`
- `sample_confidence`
- `walk_forward_history_count`
- `walk_forward_win_rate`
- `walk_forward_avg_return_pct`
- `walk_forward_avg_pnl`
- `walk_forward_rank_score`
- `composite_structure_score`
- `rationale_bullets`

### Hard Eligibility vs Soft Penalty
Hard ineligibility should be minimal and limited to execution or missing-core-data conditions:

- no earnings date
- no option chain
- cannot price required legs
- spread too wide to execute
- liquidity below absolute minimum
- invalid DTE for the structure

Everything else should be soft:

- expensive IV
- weak term structure
- low historical sample size
- low regime confidence
- weak move-fit
- concavity penalty

### Structure-Specific Logic

#### ATM Straddle
Best suited when:

- historical earnings move is large relative to event-implied move
- event risk is real but not obviously overbid
- front expiry exists with acceptable spreads
- timing window is favorable

Primary positive inputs:

- `historical_vs_implied_move_ratio`
- `tail_vs_implied_move_ratio`
- `event_implied_move_pct`
- `historical_move_anchor_pct`

Primary penalties:

- extreme smile concavity
- very high `iv_rv`
- high theta for short DTE
- poor spread quality

#### OTM Strangle
Best suited when:

- event move potential is strong
- cheaper convexity is preferred over ATM exposure
- ATM structure is too expensive relative to tail opportunity

Primary positive inputs:

- high tail-vs-implied ratio
- favorable event risk
- acceptable OTM leg liquidity

Primary penalties:

- poor wing liquidity
- too-small historical moves
- too-much premium leakage from wide spreads

#### Call Calendar / Put Calendar
Best suited when:

- long-vega expansion is expected into the event
- near term is not already over-inverted
- back leg offers enough vega support
- expected move does not imply high probability of blowing through the body

Primary positive inputs:

- favorable long-vega cheapness
- room for front-end IV build
- moderate move anchor
- supportive term structure shape

Primary penalties:

- already-rich front-end event premium
- too-large historical move tails relative to breakeven geometry
- bad same-strike pairing or far-leg illiquidity

## Composite Score Formula
V1 should use a deterministic weighted blend, calibrated later, not opaque ML.

Recommended v1 structure score:

`composite_structure_score`
= `0.30 * walk_forward_rank_score`
+ `0.20 * expected_edge_score`
+ `0.15 * move_fit_score`
+ `0.10 * cheapness_score`
+ `0.10 * timing_score`
+ `0.10 * execution_score`
+ `0.05 * data_quality_score`
- penalties

Penalties:

- `theta_drag_penalty`
- `crowding_penalty`
- `concavity_penalty`
- `sample_uncertainty_penalty`

Notes:

- `walk_forward_rank_score` should be the anchor because it reflects realized structure behavior under similar implementation rules.
- `expected_edge_score` should be capped when calibration is thin.
- `sample_uncertainty_penalty` should widen materially for low-count structure histories.

## Calibration Strategy
V1 should use **walk-forward scorecard calibration**, not a pure model fit.

Calibration inputs:

- structure-level realized return distributions
- adjusted P&L under selected execution profiles
- win rate by regime bucket
- average return by top/bottom feature quartiles

Calibration sources already available:

- `scripts/backtest_iv_expansion_study.py`
- `scripts/backtest_pre_earnings_otm_strangle.py`
- `services/calibration_service.py`
- `services/execution_cost_model.py`

Calibration outputs for v1:

- expected return bands by structure
- confidence bands by structure
- minimum trade count thresholds
- feature-to-return monotonicity checks

## Regime Bucketing
The selector should be auditable by regime. At minimum, scorecards should be analyzable by:

- `days_to_earnings` bucket
- release timing bucket
- volatility percentile bucket
- liquidity tier
- `historical_vs_implied_move_ratio` bucket
- `iv_rv_har` bucket
- term-structure bucket
- smile-concavity bucket

This supports explainability and later threshold tuning.

## Selector Output Contract
The final engine output should be strategy-first, not score-first.

Top-level response shape for v1:

- `symbol`
- `as_of`
- `earnings_date`
- `release_timing`
- `recommendation`
- `best_structure`
- `confidence_pct`
- `expected_edge_pct`
- `expected_return_pct`
- `primary_thesis`
- `primary_risks`
- `why_not_no_trade`
- `why_this_structure`
- `runner_up_structures`
- `vol_snapshot`
- `structure_scorecards`
- `data_quality`

Recommendation enum for v1:

- `Best Candidate`
- `Candidate`
- `Watch`
- `No Trade`

`No Trade` must be returned when:

- no structure is executable
- no structure has positive cost-adjusted expectancy
- confidence is too low because of thin or low-quality evidence

## UI Contract
The frontend should pivot from "calendar/crush analyzer" to "best structure for this event."

Primary card should show:

- best structure
- confidence
- expected edge after costs
- expected move vs historical move
- why this structure fits
- top execution risks

Secondary cards should show:

- vol snapshot
- structure comparison table
- historical evidence panel
- data quality / staleness panel

The current calendar-specific panel can remain, but only when the selected structure is a calendar or when the user expands a structure detail view.

## Implementation Mapping

### Reuse with Minimal Changes
- [web/api/edge_engine.py](/Users/tristanalejandro/Desktop/options_calculator_pro/web/api/edge_engine.py)
  - reuse:
    - Yang-Zhang RV
    - HAR-RV forecast
    - term structure extraction
    - smile curvature
    - historical earnings move profile
    - move anchor / uncertainty
    - risk-free lookup
- [scripts/backtest_iv_expansion_study.py](/Users/tristanalejandro/Desktop/options_calculator_pro/scripts/backtest_iv_expansion_study.py)
  - reuse:
    - structure universe
    - exact-contract trade realization
    - walk-forward summaries
    - breakdown tables
- [services/execution_cost_model.py](/Users/tristanalejandro/Desktop/options_calculator_pro/services/execution_cost_model.py)
  - reuse as friction layer
- [services/calibration_service.py](/Users/tristanalejandro/Desktop/options_calculator_pro/services/calibration_service.py)
  - reuse ideas and storage pattern, but adapt from one-score expansion calibration to structure-level calibration

### Replace or Heavily Refactor
- current `hard_no_trade` crush gates
- current `IV Expansion Candidate` gate stack
- current universal `_score_setup()` logic
- current expansion summary that hardcodes `same_strike_call_calendar`
- current recommendation router that branches between crush and expansion modes

### New Modules Recommended
- `services/earnings_vol_snapshot.py`
  - builds canonical snapshot
- `services/structure_selector.py`
  - scores structures from snapshot
- `services/structure_calibration.py`
  - stores and applies structure-level empirical calibration

`web/api/edge_engine.py` can remain as the API orchestration layer, but heavy modeling logic should move into services.

## Phased Delivery Plan

### Phase 1: Canonical Snapshot Extraction
Deliverables:

- one reusable snapshot builder
- unit tests for feature extraction and signal direction
- no UI change required yet

Success criteria:

- screener and edge analysis use the same feature definitions
- no strategy-specific gates inside snapshot generation

### Phase 2: Structure Scorecards
Deliverables:

- deterministic scorecard for each structure
- reuse walk-forward summaries from existing backtests
- structure comparison table in API output

Success criteria:

- same ticker/date can produce different structure rankings
- `No Trade` is possible

### Phase 3: Selector Output and UI Rewrite
Deliverables:

- new API response contract
- frontend best-structure presentation
- calendar details demoted to structure detail

Success criteria:

- UI tells a coherent story
- no crush-specific language remains in primary path

### Phase 4: Calibration and Threshold Tuning
Deliverables:

- structure-level calibration store
- expected-return bands and confidence bands
- threshold tuning from walk-forward results

Success criteria:

- expected edge and confidence are empirically anchored
- top-ranked structures remain robust after cost assumptions

## Validation Plan
The selector should not ship without these validations.

### Required Tests
- Snapshot feature extraction consistency tests.
- Strategy-direction tests ensuring cheap-IV setups are not rejected by legacy gates.
- Structure eligibility tests for all four supported structures.
- Recommendation tests where different synthetic setups prefer different structures.
- Regression tests ensuring `No Trade` occurs when costs erase edge.

### Required Research Runs
- walk-forward ranking by structure and regime
- strict-liquidity subset analysis
- execution-profile sensitivity
- entry/exit timing sensitivity
- feature monotonicity review for:
  - `historical_vs_implied_move_ratio`
  - `iv_rv_har`
  - term structure
  - smile concavity

### Metrics To Track
- total adjusted P&L
- average return per trade
- win rate
- return-based Sharpe
- drawdown
- structure selection frequency
- top-1 minus runner-up expectancy gap
- confidence calibration error

## Risks
- Sparse data by symbol/structure/regime may create unstable structure rankings.
- Calendar structures may look attractive in theory but fail under execution realism.
- Smile-concavity signals may identify expensive insurance rather than tradable edge.
- One universal score can quietly reintroduce the same thesis conflicts if feature directions are not standardized first.

## Non-Goals for V1
- end-to-end auto-ML
- symbol-specific deep models
- intraday dynamic hedging
- portfolio optimization across many names
- post-earnings short-vol selector

## Recommended Immediate Next Step
Implement **Phase 1 only** first:

- extract the canonical `vol_snapshot`
- remove strategy-specific gating from feature generation
- add tests for feature direction and snapshot consistency

Reason:

- every later layer depends on a clean, strategy-neutral snapshot
- it lets the screener and edge engine share one truth source
- it avoids baking today’s contradictions into a more complex selector

## Repo Decision
This spec recommends **evolving edge analysis into a structure selector**, not deleting the existing calendar work and not merging old logic blindly.

The calendar research remains useful, but only as one structure family inside a broader earnings-vol decision engine.
